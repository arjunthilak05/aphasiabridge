"""Task 2 — TED-LIUM + AphasiaBank mixed manifest builder.

Implements the mixing strategy from arXiv:2506.06566 (AS-ASR): pure aphasia
training lets the language prior collapse onto canonical task scripts; mixing
fluent English (TED-LIUM) anchors normal vocabulary while still teaching
aphasic patterns. They saw substantial gains on Whisper-Tiny going from
AphasiaBank-only to TED-LIUM + AphasiaBank.

Pipeline:
  1. Stream LIUM/tedlium release3 from HuggingFace.
  2. Filter: dur ∈ [0.5, 30] s, word_count ≥ 2, no `<unk>`, no empty text.
  3. Save 16 kHz mono WAVs to /data/kl_dev/dataset_wan/data_gemma/tedlium/clips/
  4. Subsample to ~7.4 hours of audio (matching aphasia paper-filter).
  5. Emit JSONL rows in the same shape as aphasiabank_si_train.jsonl
     (audio_path, transcript, severity="control", duration_ms, ...).
  6. Write `aphasiabank_si_train_mixed.jsonl` = aphasia + TED-LIUM
     interleaved (50/50 by hours).

Output:
  $APHASIA_ROOT/../tedlium/clips/<id>.wav        (TED-LIUM wavs)
  $APHASIA_ROOT/aphasiabank_si_train_mixed.jsonl (combined manifest)
  $APHASIA_ROOT/tedlium_subsample.jsonl          (TED-LIUM only, for diagnostics)
"""
from __future__ import annotations
import argparse, json, os, random, sys, time
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import load_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aphasia-manifest", required=True,
                    help="Path to aphasiabank_si_train.jsonl")
    ap.add_argument("--out-clips-dir", required=True,
                    help="Where to save TED-LIUM WAVs (separate dir, not under aphasiabank/)")
    ap.add_argument("--out-tedlium-manifest", required=True,
                    help="Output path for TED-LIUM-only manifest")
    ap.add_argument("--out-mixed-manifest", required=True,
                    help="Output path for the combined mixed manifest")
    ap.add_argument("--target-hours", type=float, default=None,
                    help="Match this many hours of TED-LIUM. Default = aphasia hours.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # --- Load aphasia manifest, compute target hours -----------------------
    print(f"[{time.strftime('%H:%M:%S')}] loading aphasia manifest ...", flush=True)
    with open(args.aphasia_manifest) as f:
        aphasia = [json.loads(l) for l in f if l.strip()]
    aphasia_h = sum(r.get("duration_ms", 0) for r in aphasia) / 1000 / 3600
    target_h = args.target_hours if args.target_hours else aphasia_h
    target_s = target_h * 3600
    print(f"  aphasia: {len(aphasia)} clips / {aphasia_h:.2f} h")
    print(f"  TED-LIUM target: {target_h:.2f} h")

    out_clips = Path(args.out_clips_dir)
    out_clips.mkdir(parents=True, exist_ok=True)

    # --- Stream TED-LIUM release3 -----------------------------------------
    print(f"[{time.strftime('%H:%M:%S')}] streaming LIUM/tedlium release3 train split ...", flush=True)
    ds = load_dataset("LIUM/tedlium", "release3", split="train", streaming=True)

    tedlium_rows = []
    accumulated_s = 0.0
    n_seen = n_dropped = 0

    # Stream the FIRST N clips that pass filters (deterministic).
    # We stop once we have enough hours.
    for ex in ds:
        n_seen += 1
        text = (ex.get("text") or "").strip().lower()
        # Filter: empty / <unk> / contains weird tokens
        if not text or "<unk>" in text or text.startswith("ignore_time_segment"):
            n_dropped += 1; continue
        words = text.split()
        if len(words) < 2:
            n_dropped += 1; continue
        audio = ex["audio"]
        arr = np.asarray(audio["array"], dtype=np.float32)
        sr = int(audio["sampling_rate"])
        dur_s = arr.shape[0] / sr if sr else 0.0
        if dur_s < 0.5 or dur_s > 30.0:
            n_dropped += 1; continue

        # Resample if needed (TED-LIUM is 16 kHz already)
        if sr != 16000:
            n_dropped += 1; continue   # skip rather than risk lossy resample on the fly

        # Save WAV
        clip_id = f"tedlium_{n_seen:07d}"
        wav_path = out_clips / f"{clip_id}.wav"
        sf.write(wav_path, arr, 16000, subtype="PCM_16")
        tedlium_rows.append({
            "audio_path": str(wav_path),
            "clip_path":  str(wav_path),
            "transcript": text,
            "transcript_raw": text,
            "transcript_v2": text,
            "transcript_clean": text,
            "duration_ms": int(round(dur_s * 1000)),
            "word_count": len(words),
            "participant_id": (ex.get("speaker_id") or f"tedlium_speaker_{n_seen:07d}"),
            "severity": "control",
            "corpus": "TED-LIUM-3",
            "folder": "TED",
            "condition": "control",
            "source": "tedlium",
            "session_wav": str(wav_path),
            "aphasia_type": None, "qab_overall": None,
            "has_paraphasia_marker": False,
            "has_filler": False,
            "utt_index": n_seen, "start_ms": 0, "end_ms": int(round(dur_s * 1000)),
        })
        accumulated_s += dur_s
        if (len(tedlium_rows)) % 250 == 0:
            print(f"  [{time.strftime('%H:%M:%S')}] kept {len(tedlium_rows)} clips / "
                  f"{accumulated_s/3600:.2f} h (target {target_h:.2f} h); seen={n_seen} dropped={n_dropped}",
                  flush=True)
        if accumulated_s >= target_s:
            break

    print(f"\n[{time.strftime('%H:%M:%S')}] TED-LIUM done. kept={len(tedlium_rows)} "
          f"({accumulated_s/3600:.3f} h); seen={n_seen} dropped={n_dropped}", flush=True)

    # --- Write outputs ----------------------------------------------------
    with open(args.out_tedlium_manifest, "w") as f:
        for r in tedlium_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {args.out_tedlium_manifest}")

    mixed = aphasia[:] + tedlium_rows
    rng.shuffle(mixed)
    with open(args.out_mixed_manifest, "w") as f:
        for r in mixed:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    mixed_h = sum(r.get("duration_ms", 0) for r in mixed) / 1000 / 3600
    print(f"  wrote {args.out_mixed_manifest}  ({len(mixed)} clips / {mixed_h:.2f} h)")

    print(f"\n  aphasia: {len(aphasia)} clips / {aphasia_h:.2f} h")
    print(f"  TED-LIUM: {len(tedlium_rows)} clips / {accumulated_s/3600:.2f} h")
    print(f"  mixed:   {len(mixed)} clips / {mixed_h:.2f} h")


if __name__ == "__main__":
    main()

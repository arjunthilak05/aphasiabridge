"""Task 2 — LibriSpeech + AphasiaBank mixed manifest builder.

Implements the mixing strategy from arXiv:2506.06566 (AS-ASR): pure aphasia
training lets the language prior collapse onto canonical task scripts; mixing
fluent English anchors normal vocabulary while still teaching aphasic
patterns. They used TED-LIUM. We use LibriSpeech (`openslr/librispeech_asr`
clean train-100) instead — TED-LIUM is no longer on HF Hub
(LIUM/tedlium → 404). Same hypothesis: fluent English anchors the language prior.

Pipeline:
  1. Stream openslr/librispeech_asr config "clean" split "train.100" from HF.
  2. Filter: dur ∈ [0.5, 30] s, word_count ≥ 2, no empty text.
  3. Save 16 kHz mono WAVs.
  4. Subsample to ~7.4 hours (matching aphasia paper-filter).
  5. Emit JSONL rows in the same shape as aphasiabank_si_train.jsonl.
  6. Write the combined mixed manifest interleaved (50/50 by hours).

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
                    help="Where to save fluent-corpus WAVs (separate dir, not under aphasiabank/)")
    ap.add_argument("--out-tedlium-manifest", required=True,
                    help="Output path for fluent-corpus-only manifest (kept name for compat)")
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

    # --- Stream LibriSpeech (clean, train-100) ----------------------------
    print(f"[{time.strftime('%H:%M:%S')}] streaming openslr/librispeech_asr clean train.100 ...", flush=True)
    ds = load_dataset("openslr/librispeech_asr", "clean", split="train.100", streaming=True)

    fluent_rows = []
    accumulated_s = 0.0
    n_seen = n_dropped = 0

    # Stream rows until we hit the target hours.
    for ex in ds:
        n_seen += 1
        text = (ex.get("text") or "").strip().lower()
        if not text:
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
        if sr != 16000:
            n_dropped += 1; continue

        clip_id = f"librispeech_{n_seen:07d}"
        wav_path = out_clips / f"{clip_id}.wav"
        sf.write(wav_path, arr, 16000, subtype="PCM_16")
        fluent_rows.append({
            "audio_path": str(wav_path),
            "clip_path":  str(wav_path),
            "transcript": text,
            "transcript_raw": text,
            "transcript_v2": text,
            "transcript_clean": text,
            "duration_ms": int(round(dur_s * 1000)),
            "word_count": len(words),
            "participant_id": (str(ex.get("speaker_id")) if ex.get("speaker_id") is not None
                               else f"librispeech_speaker_{n_seen:07d}"),
            "severity": "control",
            "corpus": "LibriSpeech-clean-100",
            "folder": "LibriSpeech",
            "condition": "control",
            "source": "librispeech",
            "session_wav": str(wav_path),
            "aphasia_type": None, "qab_overall": None,
            "has_paraphasia_marker": False,
            "has_filler": False,
            "utt_index": n_seen, "start_ms": 0, "end_ms": int(round(dur_s * 1000)),
        })
        accumulated_s += dur_s
        if (len(fluent_rows)) % 250 == 0:
            print(f"  [{time.strftime('%H:%M:%S')}] kept {len(fluent_rows)} clips / "
                  f"{accumulated_s/3600:.2f} h (target {target_h:.2f} h); seen={n_seen} dropped={n_dropped}",
                  flush=True)
        if accumulated_s >= target_s:
            break

    print(f"\n[{time.strftime('%H:%M:%S')}] LibriSpeech done. kept={len(fluent_rows)} "
          f"({accumulated_s/3600:.3f} h); seen={n_seen} dropped={n_dropped}", flush=True)

    # --- Write outputs ----------------------------------------------------
    with open(args.out_tedlium_manifest, "w") as f:
        for r in fluent_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {args.out_tedlium_manifest}")

    mixed = aphasia[:] + fluent_rows
    rng.shuffle(mixed)
    with open(args.out_mixed_manifest, "w") as f:
        for r in mixed:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    mixed_h = sum(r.get("duration_ms", 0) for r in mixed) / 1000 / 3600
    print(f"  wrote {args.out_mixed_manifest}  ({len(mixed)} clips / {mixed_h:.2f} h)")

    print(f"\n  aphasia: {len(aphasia)} clips / {aphasia_h:.2f} h")
    print(f"  fluent:  {len(fluent_rows)} clips / {accumulated_s/3600:.2f} h")
    print(f"  mixed:   {len(mixed)} clips / {mixed_h:.2f} h")


if __name__ == "__main__":
    main()

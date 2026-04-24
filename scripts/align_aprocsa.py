"""Slice each APROCSA session audio into per-utterance clips using the CHAT timing.

Reads:
  - data/aprocsa_par_utts.jsonl   (from parse_aprocsa_cha.py)
  - D:/Downloads/speech_data/aprocsa/audio/aprocsa{ID}.wav  (16 kHz mono)
Writes:
  - data/aprocsa_clips/aprocsa{ID}_{utt_index:04d}.wav
  - data/aprocsa_manifest.jsonl   (one record per clip, pointing to wav + transcript)

Each manifest row:
  {clip_path, participant_id, utt_index, start_ms, end_ms, duration_ms,
   transcript_clean, transcript_raw, has_paraphasia_marker, has_filler,
   condition, source, severity}

Usage:
    python scripts/align_aprocsa.py \
        --utts   data/aprocsa_par_utts.jsonl \
        --audio-dir D:/Downloads/speech_data/aprocsa/audio \
        --out-dir   data/aprocsa_clips \
        --manifest  data/aprocsa_manifest.jsonl
"""
from __future__ import annotations
import argparse, json, os, sys
import soundfile as sf
import numpy as np

# APROCSA severity per Table 1 in Ezzes et al. 2022 (QAB Overall score)
APROCSA_SEVERITY = {
    "aprocsa1738a": {"qab": 7.72, "severity": "mild",     "age": 72, "sex": "M", "tpo_months": 120},
    "aprocsa1944a": {"qab": 7.69, "severity": "mild",     "age": 71, "sex": "F", "tpo_months": 151},
    "aprocsa1713a": {"qab": 8.84, "severity": "mild",     "age": 63, "sex": "F", "tpo_months":  23},
    "aprocsa1554a": {"qab": 7.96, "severity": "mild",     "age": 46, "sex": "F", "tpo_months":  35},
    "aprocsa1833a": {"qab": 7.52, "severity": "moderate", "age": 67, "sex": "M", "tpo_months":  18},
    "aprocsa1731a": {"qab": 3.74, "severity": "severe",   "age": 48, "sex": "M", "tpo_months":  52},
}

MIN_DUR_MS = 300
MAX_DUR_MS = 30_000  # Gemma 4 E4B audio encoder cap
MIN_WORDS = 1


def load_audio(path: str):
    data, sr = sf.read(path, dtype="int16")
    if data.ndim > 1:
        data = data.mean(axis=1).astype(np.int16)
    return data, sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--utts", required=True)
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--manifest", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.manifest), exist_ok=True)

    # Cache of (pid -> (audio_int16, sr))
    audio_cache = {}

    kept = dropped_short = dropped_long = dropped_missing_time = dropped_no_audio = 0
    written = []

    with open(args.utts, "r", encoding="utf-8") as f:
        utts = [json.loads(ln) for ln in f]

    for u in utts:
        pid = u["participant_id"]
        if u["start_ms"] is None or u["end_ms"] is None:
            dropped_missing_time += 1
            continue
        dur = u["end_ms"] - u["start_ms"]
        if dur < MIN_DUR_MS or u["word_count"] < MIN_WORDS:
            dropped_short += 1
            continue
        if dur > MAX_DUR_MS:
            dropped_long += 1
            continue
        # Load audio
        if pid not in audio_cache:
            # audio file: aprocsa{ID}.wav (no trailing 'a')
            wav_path = os.path.join(args.audio_dir, f"{pid[:-1]}.wav")
            if not os.path.exists(wav_path):
                dropped_no_audio += 1
                audio_cache[pid] = None
                continue
            audio_cache[pid] = load_audio(wav_path)
        if audio_cache[pid] is None:
            dropped_no_audio += 1
            continue
        pcm, sr = audio_cache[pid]
        s = int(u["start_ms"] * sr / 1000)
        e = int(u["end_ms"] * sr / 1000)
        e = min(e, pcm.shape[0])
        if e <= s:
            dropped_short += 1
            continue
        clip = pcm[s:e]
        clip_name = f"{pid[:-1]}_{u['utt_index']:04d}.wav"
        clip_path = os.path.join(args.out_dir, clip_name)
        sf.write(clip_path, clip, sr, subtype="PCM_16")

        sev = APROCSA_SEVERITY.get(pid, {})
        rec = {
            "clip_path": os.path.abspath(clip_path),
            "participant_id": pid,
            "utt_index": u["utt_index"],
            "start_ms": u["start_ms"],
            "end_ms": u["end_ms"],
            "duration_ms": dur,
            "transcript_clean": u["clean"],
            "transcript_raw": u["raw"],
            "has_paraphasia_marker": u["has_paraphasia_marker"],
            "has_filler": u["has_filler"],
            "word_count": u["word_count"],
            "condition": "aphasia",
            "source": "aprocsa",
            "severity": sev.get("severity"),
            "qab_overall": sev.get("qab"),
        }
        written.append(rec)
        kept += 1

    with open(args.manifest, "w", encoding="utf-8") as f:
        for r in written:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Kept:                  {kept}", file=sys.stderr)
    print(f"Dropped (no timing):   {dropped_missing_time}", file=sys.stderr)
    print(f"Dropped (short):       {dropped_short}", file=sys.stderr)
    print(f"Dropped (>30s):        {dropped_long}", file=sys.stderr)
    print(f"Dropped (no audio):    {dropped_no_audio}", file=sys.stderr)
    print(f"Wrote:                 {args.manifest}", file=sys.stderr)


if __name__ == "__main__":
    main()

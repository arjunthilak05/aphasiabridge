"""Extract TORGO audio + transcripts from the abnerh/TORGO-database HF parquet shards
into per-clip WAVs + a training manifest matching APROCSA schema.

Reads: D:/Downloads/huggingface_cache/hub/datasets--abnerh--TORGO-database/.../data/*.parquet
Writes:
  - D:/Downloads/torgo_extracted/clips/torgo_<idx:06d>.wav   (16 kHz mono)
  - data/torgo_manifest.jsonl                                 (one row per clip)
"""
from __future__ import annotations
import os, re, sys, json, glob, io
import soundfile as sf
import numpy as np

HF_CACHE = r"D:\Downloads\huggingface_cache\hub\datasets--abnerh--TORGO-database"
OUT_CLIPS = r"D:\Downloads\torgo_extracted\clips"
OUT_MANIFEST = r"data/torgo_manifest.jsonl"

MIN_DUR = 0.3
MAX_DUR = 30.0
SKIP_MARKERS = re.compile(r"^\[(xxx|laughter|cough|noise|null|silence)\]$", re.IGNORECASE)


def clean_transcript(text: str) -> str:
    if text is None: return ""
    text = text.strip()
    if not text or SKIP_MARKERS.match(text):
        return ""
    # Remove standalone bracketed annotations
    text = re.sub(r"\[[^\]]*\]", "", text).strip()
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text


def map_severity(status: str) -> str:
    s = (status or "").lower()
    if "healthy" in s or "control" in s:
        return "control"
    # Otherwise treat as dysarthric; TORGO actual severity needs external lookup per speaker
    return "dysarthric"


def main():
    os.makedirs(OUT_CLIPS, exist_ok=True)
    os.makedirs(os.path.dirname(OUT_MANIFEST), exist_ok=True)
    shards = sorted(glob.glob(os.path.join(HF_CACHE, "snapshots", "*", "data", "*.parquet")))
    if not shards:
        print("No parquet shards found. Download first.", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(shards)} parquet shards", file=sys.stderr)

    import pyarrow.parquet as pq
    kept = skipped_short = skipped_long = skipped_empty = 0
    idx = 0

    with open(OUT_MANIFEST, "w", encoding="utf-8") as fout:
        for shard in shards:
            print(f"Reading {os.path.basename(shard)} ...", file=sys.stderr)
            tbl = pq.read_table(shard)
            for row in tbl.to_pylist():
                transcript = clean_transcript(row.get("transcription"))
                if not transcript:
                    skipped_empty += 1
                    continue
                dur = row.get("duration") or 0
                if dur < MIN_DUR:
                    skipped_short += 1; continue
                if dur > MAX_DUR:
                    skipped_long += 1; continue
                audio_blob = row["audio"]["bytes"]
                if not audio_blob:
                    skipped_empty += 1; continue
                # Decode and re-write as 16 kHz mono PCM_16
                data, sr = sf.read(io.BytesIO(audio_blob))
                if data.ndim > 1: data = data.mean(axis=1)
                if sr != 16000:
                    # light resample via linear (rare — dataset should already be 16 kHz)
                    import librosa
                    data = librosa.resample(data.astype(np.float32), orig_sr=sr, target_sr=16000)
                    sr = 16000
                data = np.asarray(data, dtype=np.float32)
                peak = max(1e-6, np.abs(data).max())
                data = (data / peak * 0.98 * 32767).astype(np.int16)
                clip_name = f"torgo_{idx:06d}.wav"
                clip_path = os.path.join(OUT_CLIPS, clip_name)
                sf.write(clip_path, data, sr, subtype="PCM_16")

                rec = {
                    "audio_path": clip_path,
                    "transcript": transcript,
                    "participant_id": f"torgo-{row.get('speech_status','?')}-{row.get('gender','?')}",
                    "severity": map_severity(row.get("speech_status")),
                    "source": "torgo",
                    "condition": "dysarthria" if map_severity(row.get("speech_status")) != "control" else "control",
                    "duration_ms": int(dur * 1000),
                    "speech_status": row.get("speech_status"),
                    "gender": row.get("gender"),
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1
                idx += 1
                if idx % 1000 == 0:
                    print(f"  {idx} clips written", file=sys.stderr)

    print(f"\nKept:           {kept}", file=sys.stderr)
    print(f"Skipped empty:  {skipped_empty}", file=sys.stderr)
    print(f"Skipped short:  {skipped_short}", file=sys.stderr)
    print(f"Skipped long:   {skipped_long}", file=sys.stderr)
    print(f"Manifest:       {OUT_MANIFEST}", file=sys.stderr)


if __name__ == "__main__":
    main()

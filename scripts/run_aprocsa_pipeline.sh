#!/usr/bin/env bash
# End-to-end APROCSA pipeline: extract audio from completed MP4s,
# align transcripts, build manifest, launch training.
# Safe to re-run — skips steps whose outputs already exist.
set -e
cd "$(dirname "$0")/.."
PY="$PWD/.venv/Scripts/python.exe"

export HF_HOME="D:\\Downloads\\huggingface_cache"
export HF_HUB_CACHE="D:\\Downloads\\huggingface_cache\\hub"
export TORCHDYNAMO_DISABLE=1
export PYTHONIOENCODING=utf-8

MP4_DIR="D:/Downloads/speech_data/aprocsa/mp4"
AUDIO_DIR="D:/Downloads/speech_data/aprocsa/audio"
CLIP_DIR="data/aprocsa_clips"
UTTS_JSONL="data/aprocsa_par_utts.jsonl"
MANIFEST="data/aprocsa_manifest.jsonl"
OUTPUT="outputs/lora_aprocsa_phase1"

mkdir -p "$AUDIO_DIR" "$CLIP_DIR" "outputs"

echo "==> extracting audio from completed MP4s (ffprobe-validated)"
for mp4 in "$MP4_DIR"/*.mp4; do
    id=$(basename "$mp4" .mp4)
    wav="$AUDIO_DIR/aprocsa${id}.wav"
    if [ -f "$wav" ]; then
        echo "  [done] $wav already exists"
        continue
    fi
    # ffprobe returns non-zero / empty on truncated files
    dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$mp4" 2>/dev/null || echo "")
    if [ -z "$dur" ] || [ "$dur" = "N/A" ]; then
        echo "  [skip] ${id}.mp4 unreadable (still downloading?)"
        continue
    fi
    echo "  [extract] ${id}.mp4 (${dur}s) -> $(basename "$wav")"
    ffmpeg -y -nostdin -loglevel error \
        -i "$mp4" -vn -ac 1 -ar 16000 -c:a pcm_s16le "$wav"
done

echo "==> aligning APROCSA utterances to clips"
"$PY" scripts/align_aprocsa.py \
    --utts "$UTTS_JSONL" \
    --audio-dir "$AUDIO_DIR" \
    --out-dir "$CLIP_DIR" \
    --manifest "$MANIFEST"

echo "==> building training manifest (assistant target = clean transcript)"
# align_aprocsa.py already writes the training-ready manifest with transcript_clean field
# rename the field to 'transcript' so train_gemma4_lora.py can read it
"$PY" - <<PY
import json, os
src = r"$MANIFEST"
tmp = src + ".tmp"
with open(src, "r", encoding="utf-8") as f, open(tmp, "w", encoding="utf-8") as o:
    n = 0
    for line in f:
        r = json.loads(line)
        if "transcript" not in r:
            r["transcript"] = r.get("transcript_clean", "")
        if not r["transcript"].strip():
            continue
        if "audio_path" not in r:
            r["audio_path"] = r.get("clip_path")
        o.write(json.dumps(r) + "\n")
        n += 1
os.replace(tmp, src)
print(f"  manifest has {n} trainable clips")
PY

echo "==> launching APROCSA Phase-1 training with 15% held-out eval"
"$PY" scripts/train_gemma4_lora.py \
    --manifest "$MANIFEST" \
    --output-dir "$OUTPUT" \
    --max-steps 200 \
    --batch-size 1 \
    --grad-accum 8 \
    --lora-r 16 \
    --lora-alpha 16 \
    --lr 5e-5 \
    --max-seq-length 4096 \
    --holdout-frac 0.15

echo "==> pipeline complete. LoRA saved to $OUTPUT/lora_final"

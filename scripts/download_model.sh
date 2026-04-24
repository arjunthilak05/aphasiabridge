#!/usr/bin/env bash
# Download Gemma 4 E4B 4-bit weights to D:\Downloads\huggingface_cache
set -e
cd "$(dirname "$0")/.."
PY="$PWD/.venv/Scripts/python.exe"

export HF_HOME="D:\\Downloads\\huggingface_cache"
export HF_HUB_CACHE="D:\\Downloads\\huggingface_cache\\hub"
# Token is saved at this path by earlier step
TOKEN=$(cat "D:/Downloads/huggingface_cache/token")
export HF_TOKEN="$TOKEN"

"$PY" - <<'PY'
import os
from huggingface_hub import snapshot_download
MODEL_ID = "unsloth/gemma-4-E4B-it-unsloth-bnb-4bit"
print(f"Downloading {MODEL_ID} to {os.environ.get('HF_HUB_CACHE')}")
path = snapshot_download(
    repo_id=MODEL_ID,
    token=os.environ["HF_TOKEN"],
    max_workers=4,
)
print(f"Model ready at: {path}")
# report disk usage
total = 0
for root, _, files in os.walk(path):
    for f in files:
        try: total += os.path.getsize(os.path.join(root, f))
        except OSError: pass
print(f"Total size: {total/1e9:.2f} GB")
PY

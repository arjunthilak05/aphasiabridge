#!/usr/bin/env bash
# Install Unsloth + deps into a fresh venv on Windows.
# Uses uv with --torch-backend=auto so uv picks the correct CUDA torch build
# to match the detected GPU (this is what Unsloth's own install.ps1 does).
set -e
cd "$(dirname "$0")/.."
VENV="$PWD/.venv"
PY="$VENV/Scripts/python.exe"

export HF_HOME="D:\\Downloads\\huggingface_cache"
export HF_HUB_CACHE="D:\\Downloads\\huggingface_cache\\hub"

echo "==> create venv"
python -m venv "$VENV"

echo "==> bootstrap pip + uv"
"$PY" -m pip install --quiet --upgrade pip wheel setuptools
"$PY" -m pip install --quiet uv

UV="$PY -m uv"

echo "==> unsloth + torch (auto-picks CUDA build)"
$UV pip install --python "$PY" --torch-backend=auto \
    "unsloth>=2026.4.5" unsloth-zoo

echo "==> verify CUDA torch survived"
"$PY" - <<'PY'
import torch
print(f"torch {torch.__version__}  cuda_available={torch.cuda.is_available()}  device_count={torch.cuda.device_count()}")
assert torch.cuda.is_available(), "CUDA not available after install"
print(f"device[0]: {torch.cuda.get_device_name(0)}")
PY

echo "==> audio + misc deps"
$UV pip install --python "$PY" --torch-backend=auto \
    soundfile librosa jiwer pylangacq pandas tqdm regex

echo "==> final verify"
"$PY" - <<'PY'
import torch
print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}  bf16={torch.cuda.is_bf16_supported()}")
import unsloth; print(f"unsloth {getattr(unsloth,'__version__','?')}")
import transformers; print(f"transformers {transformers.__version__}")
import bitsandbytes as bnb; print(f"bitsandbytes {bnb.__version__}")
import peft, trl, datasets, accelerate
print(f"peft {peft.__version__}  trl {trl.__version__}  datasets {datasets.__version__}  accelerate {accelerate.__version__}")
import soundfile, librosa, jiwer, pylangacq
print("audio libs OK")
PY

echo "==> install complete."

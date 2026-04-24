# H100 Runbook — AphasiaBank Full Fine-Tune

**Target reader: a Claude (or human) sitting at a freshly rented H100 instance
with this repo cloned.** Walks through the full pipeline from cold environment
to trained LoRA adapter + WER numbers.

Budget: ~$2-3/hour on Lambda/RunPod/Vast H100 SXM, aiming for ≤$40 total per plan.

---

## 0. Quick orientation

- Project: **AphasiaBridge** — Gemma 4 E4B audio LoRA for aphasic speech transcription.
  Hackathon deadline ~2026-05-21.
- What you're doing here: rerunning the **AphasiaBank fine-tune with more compute**
  than the A4000 at home can provide (batch size > 1, longer training, larger
  subsample, and/or **full fine-tuning instead of LoRA**).
- Full context in [`STATUS.md`](STATUS.md). Fallback plan in [`CONTINGENCY.md`](CONTINGENCY.md).
- Prior results on A4000: Phase-1 APROCSA WER 75.3% → 55.3%. Phase-2 AphasiaBank LoRA
  v2 is in-flight as of the last commit; check `outputs/aphasiabank_lora_r32_v2/train.log`.

---

## 1. What success looks like

On completion you should have:

1. `outputs/<run_name>/lora_final/` (LoRA) or `outputs/<run_name>/merged/` (full-FT).
2. `outputs/<run_name>/run_summary.json` with base-vs-after WER + per-severity breakdown.
3. WER target: **overall ≤ 40 %**, severe ≤ 50 %. Matches or beats published
   Whisper-tiny fine-tune numbers on AphasiaBank.
4. Clean git commit + push back to this repo with the summary (adapter weights are
   too big for git; upload those to HF Hub instead — see §9).

---

## 2. Environment bootstrap

H100 is a Linux box (Ubuntu 22.04 typical). Install uv + Python 3.11:

```bash
# Pick a workspace on fast local NVMe, NOT the persistent volume — faster I/O
cd /workspace  # or wherever fast disk is
git clone <this repo url> aphasiabridge
cd aphasiabridge

# uv is much faster than pip for this stack
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Create 3.11 venv
uv venv --python 3.11
source .venv/bin/activate

# Core install. --torch-backend=auto makes uv pick the CUDA 12.x torch wheel
# that matches the H100 driver; without it you get CPU torch.
uv pip install --torch-backend=auto \
  "unsloth>=2026.4.5" unsloth-zoo \
  "transformers>=5.5,<6" "trl>=0.24" peft accelerate bitsandbytes \
  datasets soundfile jiwer pylangacq requests
```

Verify GPU + torch:
```bash
python - <<'PY'
import torch
print("cuda:", torch.cuda.is_available(), "device:", torch.cuda.get_device_name(0))
print("mem:", torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")
print("torch:", torch.__version__)
PY
```

Should print `cuda: True device: NVIDIA H100 ... mem: 80.0 GB`. If it says
CPU or 0, reinstall torch with the right backend.

### Required env vars

```bash
export TORCHDYNAMO_DISABLE=1      # avoids triton_key import bug on Windows; harmless on Linux
export PYTHONIOENCODING=utf-8     # avoids Unicode print crashes with IPA transcripts
export HF_HOME=/workspace/hf_cache
export HF_HUB_CACHE=/workspace/hf_cache/hub
mkdir -p "$HF_HOME"

# Optional: push merged adapter to HF Hub after training
# export HF_TOKEN=hf_...
```

---

## 3. Get the data onto the H100

The AphasiaBank audio does NOT live in this repo (101 GB WAVs + license says no
redistribution). You have two options:

### Option A: Re-download with AphasiaBank credentials (takes ~6 h, works from scratch)

The login credentials live in env vars (never commit them):

```bash
export TB_USER='mariya.c@saiuniversity.edu.in'   # AphasiaBank member account
export TB_PASS='<password>'                       # ask the human for this
```

Run the two-stage downloader:

```bash
# Stage A: all CHA transcripts (~160 MB, ~15 min)
python scripts/download_aphasiabank.py \
  --stage cha --folders Protocol NonProtocol Script \
  --root "$HOME/aphasiabank"

# Build the per-session catalog.csv
python scripts/download_aphasiabank.py --stage catalog --root "$HOME/aphasiabank"

# Stage B: audio WAV via ffmpeg-from-URL. ~4-6 h at 4 parallel workers.
# Marshall corpus is separately gated (skipped).
python scripts/download_aphasiabank_media.py \
  --root "$HOME/aphasiabank" --jobs 4
```

Build per-utterance clips + split manifests + clean/subsample:

```bash
python scripts/build_aphasiabank_manifest.py \
  --root "$HOME/aphasiabank" --split
python scripts/reclean_aphasiabank_manifests.py \
  --root "$HOME/aphasiabank"
python scripts/subsample_aphasiabank.py \
  --input  "$HOME/aphasiabank/aphasiabank_train_v2.jsonl" \
  --output "$HOME/aphasiabank/aphasiabank_train_v2_50k.jsonl" \
  --target 50000 --per-speaker-cap 80 --min-words 2
python scripts/subsample_aphasiabank.py \
  --input  "$HOME/aphasiabank/aphasiabank_val_v2.jsonl" \
  --output "$HOME/aphasiabank/aphasiabank_val_v2_500.jsonl" \
  --target 500 --per-speaker-cap 5 --min-words 2
```

**Note the larger subsample (50 K vs the 25 K in Run 3).** With 80 GB VRAM on H100
you can handle batch=4, and the whole pipeline becomes data-bound rather than
memory-bound; bigger subsample is directly usable.

### Option B: upload the already-built manifest + clips from the local box (~41 GB)

If you want to skip re-downloading, the local Windows box has the prepared
clips at `D:\Downloads\aphasiabank\clips\` (347 K WAVs, 41 GB) and manifests
at `D:\Downloads\aphasiabank\aphasiabank_*_v2.jsonl`. `rsync` or `rclone` them
to the H100; rewrite the `clip_path` in each JSONL to the H100 path:

```bash
# on H100, after upload
sed -i 's|D:\\\\Downloads\\\\aphasiabank\\\\clips\\\\|/workspace/aphasia_clips/|g' \
  aphasiabank_train_v2*.jsonl aphasiabank_val_v2_500.jsonl
```

(Whichever option, end state is: balanced train + val JSONLs pointing to WAV clips on local NVMe.)

---

## 4. Download the model

```bash
python - <<'PY'
from unsloth import FastModel
import os
os.environ["HF_HOME"] = os.environ.get("HF_HOME", "/workspace/hf_cache")
FastModel.from_pretrained(
    model_name="unsloth/gemma-4-E4B-it-unsloth-bnb-4bit",
    max_seq_length=4096, load_in_4bit=True, full_finetuning=False,
)
print("model cached")
PY
```

~11 GB download, one-off.

---

## 5. Training

Three recipes depending on what you're testing.

### Recipe 1: LoRA, same as Run 3, bigger everything (~3 h on H100)

Baseline check that H100 reproduces the A4000 run at higher throughput, plus
larger LoRA + subsample:

```bash
python scripts/train_gemma4_lora.py \
  --manifest    "$HOME/aphasiabank/aphasiabank_train_v2_50k.jsonl" \
  --output-dir  outputs/h100_lora_r64 \
  --holdout-manifest "$HOME/aphasiabank/aphasiabank_val_v2_500.jsonl" \
  --max-steps 5000 \
  --batch-size 4 --grad-accum 4 \
  --lora-r 64 --lora-alpha 128 \
  --lr 2e-4 \
  --max-seq-length 4096
```

Peak VRAM ≈ 30 GB. Expected WER: our Run 3 was ~45-55%; this should be 35-45%.

### Recipe 2: Full fine-tune (~4-6 h on H100) ⭐ the main H100 experiment

This is the reason to rent H100: full fine-tuning with real gradients on every
parameter. `full_finetuning=True` in `FastModel.from_pretrained`. Requires
editing the training script once; patch below (also see
[CONTINGENCY.md](CONTINGENCY.md) Strategy B).

Edit `scripts/train_gemma4_lora.py` around line ~163:

```python
model, processor = FastModel.from_pretrained(
    model_name=MODEL_ID,
    dtype=None,
    max_seq_length=args.max_seq_length,
    load_in_4bit=False,       # CHANGED: no quantization for full-FT
    full_finetuning=True,     # CHANGED
)
# And REMOVE the FastModel.get_peft_model(...) block entirely.
```

Then run:

```bash
python scripts/train_gemma4_lora.py \
  --manifest    "$HOME/aphasiabank/aphasiabank_train_v2_50k.jsonl" \
  --output-dir  outputs/h100_full_ft \
  --holdout-manifest "$HOME/aphasiabank/aphasiabank_val_v2_500.jsonl" \
  --max-steps 5000 \
  --batch-size 4 --grad-accum 4 \
  --lora-r 0 --lora-alpha 0 \
  --lr 3e-5 \
  --max-seq-length 4096
```

Lr is **lower** (3e-5) because full-FT is unstable at LoRA's 2e-4. Peak VRAM
expected 60-70 GB. Published delta LoRA→full-FT on ASR: +3-8 pp WER reduction.

### Recipe 3: Curriculum (mild → moderate → severe), LoRA (~4 h)

Published trick for imbalanced severity distributions:

```bash
# Phase 1: mild only, 1500 steps
python scripts/subsample_aphasiabank.py \
  --input "$HOME/aphasiabank/aphasiabank_train_v2.jsonl" \
  --output "$HOME/aphasiabank/aphasiabank_train_mild.jsonl" \
  --target 15000 --per-speaker-cap 30 \
  --severity-only mild     # <- add this flag; see patch note §7

python scripts/train_gemma4_lora.py \
  --manifest "$HOME/aphasiabank/aphasiabank_train_mild.jsonl" \
  --output-dir outputs/h100_curriculum_mild \
  --holdout-manifest "$HOME/aphasiabank/aphasiabank_val_v2_500.jsonl" \
  --max-steps 1500 --batch-size 4 --grad-accum 4 \
  --lora-r 32 --lora-alpha 64 --lr 2e-4

# Phase 2: add moderate+severe — resume from phase-1 checkpoint
# (requires a small edit to accept --resume-from; see §7)
```

---

## 6. Eval

The training script runs an in-process holdout eval automatically (Phase A: base
model, Phase B: LoRA/fine-tuned) over `aphasiabank_val_v2_500.jsonl`. Results go
to `outputs/<run_name>/run_summary.json`.

For a paper-grade Fridriksson-Scripts benchmark:

```bash
python - <<'PY'
import json
with open("/workspace/aphasia_manifest/aphasiabank_val_v2.jsonl") as f:
    rows = [json.loads(l) for l in f]
fs = [r for r in rows if r["corpus"] == "Fridriksson" and r["folder"] == "Script"]
with open("/workspace/aphasia_manifest/fridriksson_scripts_val.jsonl", "w") as f:
    for r in fs:
        f.write(json.dumps(r) + "\n")
print(len(fs), "Fridriksson-Scripts val clips")
PY
```

Then pass that as `--holdout-manifest` on a short eval-only pass.

---

## 7. Patches you may need

Depending on which recipe above, a couple of small flag additions help. Keep
them in a branch rather than on main:

- `--resume-from <checkpoint>` in `train_gemma4_lora.py` (reload LoRA + optimizer state).
- `--severity-only mild|moderate|severe|very_severe` in `subsample_aphasiabank.py`
  (filter a single severity bucket).
- Optionally `--skip-post-eval` in `train_gemma4_lora.py` so the hardcoded UASpeech
  post-training eval (§10 of `STATUS.md`) doesn't try to access paths that only
  exist on the Windows box.

All three are <10-line edits. Commit them on a `h100-run` branch and push back.

---

## 8. Monitoring + safety

- H100 runs on rental are $$ — set `max-steps` explicitly, never leave loops open.
- Use `nohup ... > train.log 2>&1 &` + `tail -f train.log | grep "'loss'"` so a
  dropped SSH doesn't kill the run.
- Monitor GPU: `watch -n 10 nvidia-smi`. If util drops below 70% consistently,
  data loading is the bottleneck; increase `num_workers` in the dataloader or
  prefetch clips.
- Budget check: the 5000-step full-FT on 50 K clips at batch=4 should cost ~5-6 h
  wall. At $3/hr H100 that's ~$18. Leaves room for Recipe 1 or a retry.

---

## 9. What to push back to this repo

After successful training:

1. Do NOT commit LoRA/merged weights (> 1 GB). Upload to Hugging Face Hub:
   ```bash
   huggingface-cli upload arjunthilakeros/aphasiabridge-gemma4-e4b-lora \
     outputs/h100_lora_r64/lora_final --create
   ```
2. DO commit:
   - The `run_summary.json` (tiny, has WER numbers)
   - The modified `scripts/train_gemma4_lora.py` + any flag patches from §7
   - A short `outputs/h100_lora_r64/README.md` with: final WER, hyperparams, wall time, notes
   - Update `STATUS.md` Section 6 with the H100 result row
3. Commit + push:
   ```bash
   git checkout -b h100-run-1
   git add scripts/ outputs/h100_lora_r64/run_summary.json outputs/h100_lora_r64/README.md STATUS.md
   git commit -m "H100 full fine-tune: overall WER <X>%, per-severity <...>"
   git push origin h100-run-1
   ```
4. Open a PR back to main with the summary.

---

## 10. Known gotchas

- **Unsloth's `save_pretrained_merged()` for audio is untested** — see STATUS.md
  §7. If merging for deployment fails, you can `model.save_pretrained()` and
  load via `FastModel.from_pretrained(lora_dir)` **in the same Python process
  you trained in** — cross-process audio reload is currently broken.
- **RAM ceiling**: loading 50 K audio clips as float32 at dataset build time
  needs ~50 GB RAM. H100 hosts usually have 128-256 GB, fine. But the training
  script uses `Dataset.set_transform` for lazy loading — don't revert that.
- **Collator marker** `[105, 4368, 107]` is validated at training start; check
  the log line `[collator] assistant-start marker = [...]`. If it shows a
  different value, the Gemma tokenizer was updated — retest before trusting
  loss values.
- **IPA contamination check**: always run `reclean_aphasiabank_manifests.py`
  before training. The pre-clean v1 manifests had 8.6 % IPA phonetic symbols
  that caused a catastrophic regression in Run 1 (WER 3.66). v2 manifests are
  the safe ones.

---

## 11. Emergency rollback

If anything breaks and you're running out of rental hours:

1. Kill the run: `kill $(pgrep -f train_gemma4_lora)`.
2. Tar up the latest checkpoint: `tar czf checkpoint.tgz outputs/*/checkpoint-*/`.
3. Upload to a cloud bucket / HF Hub.
4. Write one paragraph in STATUS.md saying what you learned and what to try next.
5. Terminate the instance.

---

## Quick-reference checklist

- [ ] H100 visible, torch CUDA works
- [ ] `TORCHDYNAMO_DISABLE=1` + `PYTHONIOENCODING=utf-8` set
- [ ] HF_HOME on fast NVMe
- [ ] AphasiaBank data on disk (either re-downloaded or rsync'd)
- [ ] Clean v2 manifests: 50K train, 500 val (balanced)
- [ ] Gemma 4 E4B weights cached
- [ ] Pick recipe: LoRA-big / full-FT / curriculum
- [ ] Launch with nohup, log to file
- [ ] Monitor first 50 steps for collator marker + loss sanity
- [ ] After done: WER breakdown saved to JSON, adapter to HF Hub, code + summary committed

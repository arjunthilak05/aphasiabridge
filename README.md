# AphasiaBridge — Gemma 4 E4B Fine-Tuning for Aphasic Speech

Hackathon submission (Gemma 4 Good Hackathon, deadline 2026-05-21). On-device
Android app that uses a fine-tuned Gemma 4 E4B to help post-stroke aphasia
patients turn fragmented speech into clear messages and smart-home actions.

> **Headline so far**: APROCSA held-out WER 75.3% → 55.3% (Phase-1, 15 min on
> an RTX A4000, 6 speakers, 200 steps).
> Phase-2 on full AphasiaBank (889 h, 1,826 speakers) is in progress.

---

## Where to look first

- [`STATUS.md`](STATUS.md) — **single source of truth** on data, scripts,
  training recipe, experiment results, known issues. Read this before anything.
- [`CONTINGENCY.md`](CONTINGENCY.md) — what to do if the current Phase-2 LoRA
  run underperforms. Decision tree + fallbacks + web-research citations.
- [`H100_RUNBOOK.md`](H100_RUNBOOK.md) — complete runbook for running full
  fine-tuning on a rented H100, end-to-end, in under a day.
- [`scripts/`](scripts/) — all training / data / eval code.

---

## The pipeline, in 6 commands

Assumes Windows 11 + RTX A4000 locally, with AphasiaBank credentials in env vars.

```bash
# 1. Download AphasiaBank CHA transcripts (~160 MB, minutes)
python scripts/download_aphasiabank.py --stage cha --folders Protocol NonProtocol Script

# 2. Build session catalog with severity + media URLs
python scripts/download_aphasiabank.py --stage catalog

# 3. Download session audio as 16 kHz mono WAV (~100 GB, ~6 h on consumer internet)
python scripts/download_aphasiabank_media.py --jobs 4

# 4. Slice into per-utterance clips + speaker-independent splits (~40 GB output)
python scripts/build_aphasiabank_manifest.py --split

# 5. Re-clean transcripts (IPA-strip, [: target] substitution, 0word removal)
python scripts/reclean_aphasiabank_manifests.py

# 6. Train LoRA (~4 h on A4000)
python scripts/train_gemma4_lora.py \
  --manifest "D:\Downloads\aphasiabank\aphasiabank_train_v2_25k.jsonl" \
  --output-dir outputs/aphasiabank_lora_r32_v2 \
  --max-steps 3000 --batch-size 1 --grad-accum 8 \
  --lora-r 32 --lora-alpha 64 --lr 2e-4 \
  --holdout-manifest "D:\Downloads\aphasiabank\aphasiabank_val_v2_500.jsonl"
```

See `STATUS.md §9` for full env-var setup and commands to resubsample / reconfigure.

---

## Credentials (never committed)

```bash
export TB_USER='<your AphasiaBank email>'
export TB_PASS='<your AphasiaBank password>'
export HF_HOME="D:\Downloads\huggingface_cache"
export HF_HUB_CACHE="D:\Downloads\huggingface_cache\hub"
export TORCHDYNAMO_DISABLE=1
export PYTHONIOENCODING=utf-8
```

AphasiaBank access is CC BY-NC-SA 3.0 — **research / hackathon use only**,
cannot train commercial models. Ship-grade models must use APROCSA + synthetic
+ personal corpora.

---

## Repo layout

```
scripts/                Training, download, parse, eval scripts
data/                   Small manifests (clips excluded via .gitignore)
outputs/                Training checkpoints (git-ignored; regenerate or push to HF Hub)
STATUS.md               Project state + results
CONTINGENCY.md          If-LoRA-fails plan
H100_RUNBOOK.md         Scale-up on rental H100
requirements.txt        Pinned dependency list
```

Reusable scripts worth reading:
- `scripts/train_gemma4_lora.py` — SFT with audio-aware collator and
  assistant-only loss masking. Works for any (audio, transcript) JSONL manifest.
- `scripts/build_aphasiabank_manifest.py` — CHAT parser + clip slicer. The
  `strip_chat_annotation()` is the heart of it; handles `[: target]`,
  `0word` null morphemes, CHAT retrace/repetition markers.
- `scripts/probe_gemma4_modules.py` — diagnostic for verifying your LoRA
  `target_modules` actually match the Gemma 4 audio tower structure. Saved us
  in Run 1 where 12 of 14 listed modules matched zero Linears.

---

## License

Code: MIT.

Data: AphasiaBank is CC BY-NC-SA 3.0 — research/non-commercial only. APROCSA is
separately licensed (see TalkBank). No aphasia audio is committed here; all
downloads happen at runtime with member credentials.

# AphasiaBridge — Project Status (2026-04-24)

End-to-end status. Use this as the pickup point when conversation context is compacted.

---

## 1. Project

**AphasiaBridge** — hackathon submission (Gemma 4 Good Hackathon). On-device Android app using fine-tuned Gemma 4 E4B to help post-stroke aphasia patients turn fragmented speech into actions (SMS, calls, smart-home).

Deadline: ~2026-05-21 (timeline started 2026-04-21; ~4 weeks remaining).

Hackathon scoring: Impact & Vision 40 pts + Video & Storytelling 30 pts + Technical Depth 30 pts. Tracks: Health & Sciences + Digital Equity.

Target headline: intent-recovery ~30% → 75%+ (base vs. fine-tuned on held-out user utterances).

---

## 2. Hardware + environment

- **Local box**: Windows 11, RTX A4000 16 GB (SM 8.6, CUDA 12.8 driver), Python 3.11.9, 128 GB RAM.
- **Venv**: `C:\Users\Hp\Desktop\gemmafinetune\.venv\`
  - torch 2.8.0+cu129, unsloth 2026.4.6, unsloth-zoo 2026.4.8
  - transformers 5.5.0, trl 0.24.0, peft 0.19.1, accelerate 1.13.0, datasets 4.3.0
  - bitsandbytes 0.49.2, xformers (FA2 broken, falls back), torchcodec 0.11.1, timm 1.0.26
  - soundfile, librosa, jiwer, pylangacq, ffmpeg 8.0
- **HF cache**: `D:\Downloads\huggingface_cache\`
- **D: usage**: 819 GB used / 135 GB free (after AphasiaBank download + manifest build).

### Windows gotchas
- **MUST** set `TORCHDYNAMO_DISABLE=1` before any Unsloth script (torch 2.8 + triton-windows 3.6 has a `triton_key` import bug that crashes inductor)
- `export PYTHONIOENCODING=utf-8` so Greek/IPA Unicode prints don't crash
- Install via `uv pip install --torch-backend=auto unsloth-zoo "unsloth>=2026.4.5"` — `--torch-backend=auto` prevents pip from overwriting CUDA torch with CPU torch.

---

## 3. Data assets (what's on disk)

### Real corpora

| Corpus | Path | Contents | Role |
|---|---|---|---|
| **AphasiaBank** (English PWA) | `D:\Downloads\aphasiabank\` | **2,493 session WAVs (102 GB, 889 h, 16 kHz mono), 2,739 CHA transcripts** (60 corpora across Protocol/NonProtocol/Script) | **Primary large-scale training data** |
| APROCSA (aphasia) | `D:\Downloads\speech_data\aprocsa\` | 6 CHAT transcripts + 6 session MP4s + 6 WAVs | Legacy Phase-1 training data (superseded by AphasiaBank) |
| UASpeech (dysarthria/CP) | `D:\Downloads\speech_data\<speakers>\` | 135,977 isolated-word WAVs, 15 dysarthric + 13 control speakers | Not useful for aphasia (wrong condition); used only for zero-shot sanity baseline |
| TORGO (dysarthria/CP+ALS) | `D:\Downloads\torgo_extracted\clips\` | 16,546 clips (10,976 healthy + 5,570 dysarthric) | **Harmful for aphasia — see §6**, unused |

### AphasiaBank derived

| File | Description |
|---|---|
| `D:\Downloads\aphasiabank\cha/` | 2,739 per-corpus CHA transcripts (160 MB) |
| `D:\Downloads\aphasiabank\wav/` | 2,493 session WAVs (16 kHz mono, 102 GB, 889 h) |
| `D:\Downloads\aphasiabank\catalog.csv` | Session-level catalog: corpus, folder, severity, WAB-AQ, aphasia_type, media_url |
| `D:\Downloads\aphasiabank\clips/` | **365,991 per-utterance clips (347K on disk, 41 GB)** — produced by `build_aphasiabank_manifest.py` |
| `aphasiabank_manifest.jsonl` | Full manifest, all 366K clips (273 MB) |
| `aphasiabank_{train,val,test}.jsonl` | Speaker-independent splits (301K / 35K / 30K) |
| `aphasiabank_{train,val,test}_v2.jsonl` | **v2 cleaned versions**: IPA stripped, `[: target]` substituted, `0word` markers removed, lowercased |
| `aphasiabank_train_v2_25k.jsonl` | **Balanced 19,311-clip training subsample** (883 speakers, 25.6 h, severity-stratified) — **current Run 3 training manifest** |
| `aphasiabank_val_v2_500.jsonl` | **Balanced 409-clip eval subsample** (109 speakers, 0.6 h) — current in-process holdout |
| `media.log`, `build_manifest.log` | Download + build logs |
| `media_failures.txt` | 67 failed downloads (62 Marshall gated + 5 server 404/400) |

### APROCSA derived (legacy, Phase-1)

| File | Description |
|---|---|
| `data/aprocsa_manifest.jsonl` | 1,208 per-utterance clips from 6 speakers (Phase-1 training data) |
| `data/holdout_aprocsa_only.jsonl` | 181 APROCSA-only held-out clips (seed 1337) |

### Model

- `D:\Downloads\huggingface_cache\hub\models--unsloth--gemma-4-E4B-it-unsloth-bnb-4bit\...\` — 10.87 GB, 12 files.

### Access status

- **AphasiaBank**: ✅ **ACCESS GRANTED** (mariya.c@saiuniversity.edu.in credentials). Full English PWA corpus downloaded 2026-04-23.
  - License: CC BY-NC-SA 3.0. **Research/hackathon only — cannot train commercial LLMs.** Ship-grade model must use APROCSA + synthetic + personal corpora only.
  - 1 corpus gated separately (Marshall, 62 sessions) — needs its own password key; skipped.
- **PSST**: form at `psst.study/announcement/` NOT filled. Low priority.
- **Personal corpus**: not started.

---

## 4. Scripts (all in `scripts/`)

| Script | Purpose |
|---|---|
| `install_env.sh` | Reproducible env install (uv + --torch-backend=auto) |
| `download_model.sh` | Fetch Gemma 4 E4B 4-bit weights to D: |
| `download_aphasiabank.py` | **AphasiaBank downloader** — cookie auth + per-corpus `?f=zip` for CHA, parses CHA headers into catalog.csv |
| `download_aphasiabank_media.py` | **AphasiaBank media pipeline** — parallel ffmpeg direct-from-URL to 16 kHz WAV, no MP4 on disk, gated-corpus auto-blacklist |
| `build_aphasiabank_manifest.py` | Parse CHA + slice WAV → per-utterance clips + full manifest + speaker-independent split. **v2 `strip_chat_annotation` handles IPA, `[: target]`, `0word`, all CHAT markers** |
| `reclean_aphasiabank_manifests.py` | Re-run improved cleaner over `transcript_raw` in existing manifests (no WAV re-slicing) |
| `subsample_aphasiabank.py` | Balanced severity-stratified, speaker-capped subsample (round-robin bug fixed) |
| `probe_gemma4_modules.py` | Diagnostic: dump Gemma 4 E4B audio_tower module names (how we caught the target_modules bug) |
| `diagnose_lora.py` | Load a trained LoRA and print sample predictions |
| `parse_aprocsa_cha.py` | Legacy CHAT parser (APROCSA) |
| `align_aprocsa.py` | Legacy per-utterance slicer (APROCSA) |
| `generate_synthetic_aphasic.py` | Text perturbation (mild/moderate/severe error models) |
| `baseline_zeroshot_uaspeech.py` | UASpeech zero-shot WER sweep |
| `verify_gemma4_audio.py` | Smoke test: load model, transcribe 4 UASpeech clips |
| `train_gemma4_lora.py` | **Main training script** — SFT with audio collator + assistant-only loss masking + in-process base/after WER eval |
| `eval_with_lora.py` | Reload+eval path (broken for audio LoRA — see §7) |
| `clean_aphasiabank_manifests.py` | Legacy post-process filter (superseded by `reclean_aphasiabank_manifests.py`) |

---

## 5. Training recipe (verified working on A4000)

```python
model, processor = FastModel.from_pretrained(
    model_name="unsloth/gemma-4-E4B-it-unsloth-bnb-4bit",
    dtype=None, max_seq_length=4096, load_in_4bit=True, full_finetuning=False,
)

model = FastModel.get_peft_model(
    model,
    finetune_vision_layers=False, finetune_language_layers=True,
    finetune_attention_modules=True, finetune_mlp_modules=True,
    r=32, lora_alpha=64, lora_dropout=0.1, bias="none",
    target_modules=[
        # Language tower (42 layers × 7 proj types)
        "q_proj","k_proj","v_proj","o_proj",
        "gate_proj","up_proj","down_proj",
        # Audio encoder — MUST use the INNER linears; outer wrappers silently miss.
        "ffw_layer_1.linear",   # FFN-1 inner (60 layers)
        "ffw_layer_2.linear",   # FFN-2 inner (60 layers)
        "relative_k_proj",      # relative-pos attention (12)
        "input_proj_linear",    # audio input projection (1)
        "output_proj",          # audio output projection (1)
        "embedding_projection", # audio embedding projection (1)
    ],
)

# Custom collator wraps UnslothVisionDataCollator to mask loss to assistant tokens only.
# Marker for Gemma 4: [105, 4368, 107] = <|turn>, model, \n
masked = AssistantOnlyLossCollator(UnslothVisionDataCollator(model, processor), processor)

# IMPORTANT: lazy audio loading via Dataset.set_transform.
# Previous version materialised all audio in RAM — MemoryError at 25K clips (150 GB float64).
ds = Dataset.from_list(light_records).with_transform(_lazy_messages)

trainer = SFTTrainer(
    model=model, train_dataset=ds,
    processing_class=processor.tokenizer, data_collator=masked,
    args=SFTConfig(
        per_device_train_batch_size=1, gradient_accumulation_steps=8,
        warmup_ratio=0.03, max_steps=3000,
        learning_rate=2e-4,                    # Unsloth Gemma-recommended
        optim="adamw_8bit", weight_decay=0.001,
        lr_scheduler_type="cosine", bf16=True, fp16=False,
        dataset_text_field="", dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False, max_length=4096,
    ),
)
```

**A4000 footprint (current v2 config)**: peak ~14–15 GB VRAM (1–2 GB headroom), ~4.5 s/step, 3000 steps ≈ 4 h.

---

## 6. Experiments & headline results

### Phase-1 APROCSA-only (legacy baseline, 2026-04-21) ⭐

**Config**: 200 steps, 1,027 train clips (5 speakers), LoRA r=16, LR 5e-5, batch 1 × accum 8.
Held-out on 181 APROCSA clips:

| Severity | Base WER | After LoRA WER | Δ |
|---|---|---|---|
| **Overall** | **75.3%** | **55.3%** | **−20.0 pp** |
| mild | 63.8% | 53.4% | −10.4 |
| moderate | 102.6% | 56.1% | −46.5 |
| severe | 106.5% | 64.5% | −42.0 |

15.2 min on A4000. Adapter at `outputs/lora_aprocsa_phase1/lora_final/`.

### Phase-1 APROCSA + TORGO mix (NEGATIVE result) ❌

Adding TORGO dysarthric data hurts aphasia WER by **+17 pp overall**. Confirmed: do NOT mix dysarthria corpora into aphasia training. Both TORGO and UASpeech stay on disk but unused.

### Phase-2 AphasiaBank **Run 1 — IPA CONTAMINATION** ❌ (2026-04-24)

**Config**: 25K clips (dirty cleaner), r=32, α=64, lr=1e-4, 3000 steps, 6.6 h wall-time.

| | Base | After LoRA |
|---|---|---|
| Overall WER | 0.695 | **3.664** (366%) |
| mild | 0.532 | 2.651 |
| moderate | 0.727 | 4.776 |
| severe | 0.749 | 5.103 |
| very_severe | 1.431 | 3.594 |

**Catastrophic regression.** Root causes found after the run:

1. **8.6% of training transcripts contained IPA phonetic symbols** (`fivəm butter tʃɛrlɪ`). CHAT transcribers preserved actual paraphasic sounds; our cleaner left them in. Model learned to output phonetics.
2. **Audio encoder target_modules list was mostly wrong.** 12 of 14 listed audio modules (`post`, `linear_start`, `ffw_layer_1`, `ffw_layer_2`, etc.) matched **zero** real Linears — they're wrapper Modules. Actual Linears sit one level deeper at `ffw_layer_1.linear`. So LoRA was effectively language-only.
3. **`[: target]` replacement annotations discarded.** CHAT writes paraphasic surface form followed by intended target: `haf [: half]`. We stripped the whole bracket, keeping only `haf`. Published benchmarks use the target.
4. **`0word` null morphemes leaked** (`0det`, `0pres`, `0pl`) as literal tokens.
5. **`lora_dropout=0`**, `lr=1e-4` — Unsloth docs recommend 0.1 and 2e-4.

### Phase-2 AphasiaBank **Run 2 — aborted** (2026-04-24)

Retrained on IPA-stripped data only (first round of cleaning) but before we caught bugs #2–5. Aborted at step ~300 after audit. Loss was healthy (3.1 at step 300) but wouldn't have hit target.

### Phase-2 AphasiaBank **Run 3 — v2 (IN PROGRESS)** 🏃 (2026-04-24)

**All audit fixes applied.**

**Config**:
- Manifest: `aphasiabank_train_v2_25k.jsonl` — 19,311 clips / 883 speakers / 25.6 h, severity-stratified
- r=32, α=64, **lora_dropout=0.1**, **lr=2e-4**, 3000 steps
- **Audio target_modules fixed**: `ffw_layer_1.linear`, `ffw_layer_2.linear`, `relative_k_proj`, `input_proj_linear`, `output_proj`, `embedding_projection` + language attn/MLP
- Collator verified: 8.7% tokens unmasked (loss only on assistant response)
- Holdout: `aphasiabank_val_v2_500.jsonl` (409 clips, 109 speakers)
- Started 22:56:13, ETA ~4 h

**Loss trajectory (20-step rolling avg) vs Run 1 at same steps**:

| Step | Run 1 (dirty) | **Run 3 (v2)** |
|---|---|---|
| 1 | 13.00 | 15.38 |
| 50 | 4.69 | **3.98** |
| 100 | 3.95 | **3.69** |
| 200 | 3.61 | **3.47** |

Consistently 5–10% lower loss. Gap expected to widen as audio LoRA actually receives gradients for the first time.

**WER expectations after Run 3** (published Whisper zero-shot numbers for comparison):

| | Base Gemma | Whisper zero-shot | Run 3 target |
|---|---|---|---|
| Overall | 69.5% | ~50% | **45–55% (≤40% stretch)** |
| mild | 53% | 32.6% | 30–40% |
| moderate | 73% | 40.6% | 45–55% |
| severe | 75% | 43.7% | 55–65% |
| very_severe | 143% | 65.3% | 65–80% |

### UASpeech digits baseline (secondary, legacy)

Zero-shot base Gemma 4 on UASpeech D1–D9, 28 speakers, control consensus as GT: control 20.4% · High 16.7% · Mid 38.9% · Low 85.2% · Very Low 166.7%. Clean severity gradient. Used only as reference; not a target metric.

---

## 7. Known issues / unresolved

- **LoRA reload from disk fails** with "Audio features and audio tokens do not match" — Unsloth's audio patches don't re-apply when loading saved LoRA via `FastModel.from_pretrained(lora_dir)`. Workaround: train + eval in the same Python process. For deployment, use `model.save_pretrained_merged()` (untested).
- **Marshall corpus (62 sessions) is separately password-gated** with its own key doc. Permanently blacklisted in `download_aphasiabank_media.py:--skip-corpora`.
- **Server-side 404/400 on ~5 Fridriksson-2 sessions** — benign, logged in `media_failures.txt`.
- **Kurland-BATS zip endpoint dies mid-stream** from talkbank.org. Missing ~10 sessions; not worth fighting.
- **Training RAM ceiling**: loading 25K clips' audio as float64 upfront needs 150 GB; lazy loading via `set_transform` fixed this. Beyond ~50K clips even lazy loading risks OOM if many parallel workers.
- **Symlink warning** from HF cache on Windows — benign (uses hardlinks).
- **Unicode stdout crashes** on Windows cp1252 when transcripts contain IPA — fixed by ASCII-only cleaning; diagnostic scripts use `errors="replace"`.

---

## 8. What's NOT done yet (gap to full AphasiaBridge)

1. **Intent-recovery metric** — we measure WER, not fragment→intent reconstruction. Hackathon pitch promises the latter. Needs paired aphasic-audio → intended-sentence data. Sources: personal corpus + synthetic TTS pipeline.
2. **Synthetic corpus at scale** — generator works (75 sample utterances). Needs scaling to 3–5 k samples paired with TTS audio. ~1 day of work.
3. **Personal corpus** — not started. Recruit 1 consented user, ~1 week of recordings.
4. **Function calling** — no training data for SMS / call / smart-home actions yet. Phase-3.
5. **Android app / LiteRT-LM** — model trained, not deployed to phone. Separate workstream.
6. **PSST corpus** — form not filled; low priority given AphasiaBank access.

---

## 9. Commands to reproduce / continue

Env activation:
```bash
export HF_HOME="D:\Downloads\huggingface_cache"
export HF_HUB_CACHE="D:\Downloads\huggingface_cache\hub"
export TORCHDYNAMO_DISABLE=1
export PYTHONIOENCODING=utf-8
cd C:/Users/Hp/Desktop/gemmafinetune
PY=.venv/Scripts/python.exe
```

**Run 3 launch command (current training)**:
```bash
$PY scripts/train_gemma4_lora.py \
  --manifest "D:\Downloads\aphasiabank\aphasiabank_train_v2_25k.jsonl" \
  --output-dir outputs/aphasiabank_lora_r32_v2 \
  --max-steps 3000 --batch-size 1 --grad-accum 8 \
  --lora-r 32 --lora-alpha 64 --lr 2e-4 \
  --holdout-manifest "D:\Downloads\aphasiabank\aphasiabank_val_v2_500.jsonl"
```

**Rebuild manifests from scratch (if CHAT cleaner is updated)**:
```bash
$PY scripts/reclean_aphasiabank_manifests.py
$PY scripts/subsample_aphasiabank.py \
  --input  "D:\Downloads\aphasiabank\aphasiabank_train_v2.jsonl" \
  --output "D:\Downloads\aphasiabank\aphasiabank_train_v2_25k.jsonl" \
  --target 25000 --per-speaker-cap 30 --min-words 2
$PY scripts/subsample_aphasiabank.py \
  --input  "D:\Downloads\aphasiabank\aphasiabank_val_v2.jsonl" \
  --output "D:\Downloads\aphasiabank\aphasiabank_val_v2_500.jsonl" \
  --target 500 --per-speaker-cap 5 --min-words 2
```

---

## 10. Recommended next moves (priority order)

1. **Wait for Run 3 completion** (~3 h) and inspect Phase-A vs Phase-B WER per severity.
2. **If Run 3 overall WER ≤ 50%**: hackathon headline ("70% → 50% WER on aphasic speech"); move to intent-recovery metric + synthetic corpus.
3. **If 50 % < WER ≤ 65%**: iterate — longer run (5000 steps), larger subsample (50–80 K clips, target reduce per-speaker-cap bias), possibly r=64.
4. **If WER still > 65%**: diagnose with `diagnose_lora.py` — inspect 20 sample predictions; look for residual data issues (punctuation handling, rare aphasia types).
5. **Scale synthetic corpus** (Phase-3) — unlocks the intent-recovery metric. Independent of Run 3 outcome.
6. **Start personal-corpus recruitment** (parallel track, no code dep).
7. **Merge LoRA to base weights** and verify on-phone deployment via LiteRT-LM (pre-demo work).

Deprioritize / skip: TORGO, UASpeech, PSST beyond filling the form.

---

## 11. Key memory files (persistent across sessions)

At `C:\Users\Hp\.claude\projects\C--Users-Hp-Desktop-gemmafinetune\memory\`:
- `project_aphasiabridge.md` — project overview
- `project_finetune_strategy.md` — two-corpus plan
- `project_access_status.md` — who we emailed
- `project_dataset_status.md` — what's on disk
- `project_training_recipe_verified.md` — working training config
- `project_headline_result.md` — Phase-1 numbers (55.3% overall)
- `project_torgo_negative_result.md` — TORGO A/B test
- `reference_aphasiabank.md` — external pointer

`MEMORY.md` indexes them all. **Update required**: add the AphasiaBank full-download milestone + Run 1 failure diagnosis as durable memories.

---

## 12. 2026-04-23 / 24 — today's work (summary)

1. **AphasiaBank download** (cookie-based auth, per-corpus `?f=zip` for CHA, ffmpeg HTTP Range for MP4→WAV, 4 parallel workers, 11 h wall-time): 2,493 sessions / 889 h / 102 GB on `D:\Downloads\aphasiabank\`.
2. **Manifest build** (`build_aphasiabank_manifest.py`): 365,991 per-utterance clips + speaker-independent splits.
3. **LoRA Run 1** (25K clips, r=32, 3000 steps): 6.6 h training → **WER 0.695 → 3.664 catastrophic regression**.
4. **Diagnosis** (web research + repo audit): IPA leak (8.6%), audio target_modules bug (12 of 14 listed didn't match), `[: target]` discarded, `0word` markers, lr/dropout tuning.
5. **v2 cleaning + fix pass**: all 6 issues addressed in place.
6. **LoRA Run 3 launched** with fixed pipeline. In-flight at the time of this writing (step ~250, loss 3.47 (20-step avg)). ETA ~3.5 h.

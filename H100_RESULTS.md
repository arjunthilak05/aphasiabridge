# H100 Run Log — A to Z

Single-host work log on `claudeuser@spk-hgx8-h100-11`. Branch: `h100-run-1`.
Started 2026-04-24 18:33 UTC. This document is the **authoritative narrative**
of every decision, fix, run, and result in this fine-tune attempt.

---

## A. Box & environment

- **Hardware**: 8× NVIDIA H100 80 GB HBM3 on a shared erosinvest cluster.
  Other tenants run `tritonserver` (production inference, GPU 0) +
  `miniconda3/envs/eros` workloads (GPUs 4, 7). We picked **fully idle GPUs only**.
- 2 TB RAM, 28 TB ext4 over 6× NVMe RAID (8.4 TB free at start).
- Ubuntu 22.04, NVIDIA driver 575.57.08, system CUDA 12.6.

### Software stack we built
- Python 3.11.15 in an isolated `uv venv` at `aphasiabridge/.venv`.
- torch 2.10.0+**cu129**, unsloth 2026.4.8, unsloth-zoo 2026.4.9,
  transformers 5.5.0, trl 0.24.0, peft 0.19.1, accelerate 1.13.0,
  bitsandbytes 0.49.2, soundfile, jiwer, pylangacq.
- **CUDA shimming**: torch ships cu129 wheels; system has cu126 in
  `/usr/local/cuda-12.6/`. Unfixed, torch fails to import (`undefined symbol:
  __nvJitLinkGetErrorLogSize_12_9`). Fixed via `LD_LIBRARY_PATH` prepending the
  venv's `nvidia/*/lib` paths so torch's bundled libs win.
- **Secrets**: `HF_TOKEN` (Arjunthilak HF user with `repo.write` on `Arjunthilak/*`),
  `TB_USER`/`TB_PASS` (AphasiaBank credentials) live in
  `aphasiabridge/.env.local`, mode 600, gitignored via `.env.*` rule. Sourced
  at the start of every Bash call (Bash sessions don't persist env between calls).
- `CUDA_VISIBLE_DEVICES` overridden per-run to pick a specific GPU.

---

## B. Data acquisition + base manifests

| Stage | Wall | Output |
|---|---|---|
| `download_aphasiabank.py --stage cha` | ~6 min | 2,739 CHA transcripts / 160 MB |
| `download_aphasiabank.py --stage catalog` | ~2 s | `catalog.csv` with severity / WAB-AQ / media URLs |
| `download_aphasiabank_media.py --jobs 6 --skip-corpora Marshall` | **3 h** | **2,498 session WAVs / 110 GB** (65 fail = Marshall blacklist + benign 404s; runbook estimated 6 h, we got it in half) |
| `build_aphasiabank_manifest.py --split` | ~25 min | 346,896 per-utterance clips + speaker-independent train/val/test JSONL |
| `reclean_aphasiabank_manifests.py` | ~3 min | v2 manifests: IPA stripped, `[: target]` substituted, `0word` removed, lowercased. Verified by Agent B (transcripts 100% clean on a 300-sample audit) |

### Subsamples we built
- `aphasiabank_train_v2_50k.jsonl` — 39,690 clips / 884 speakers / 51.0 h (cap 80 — used for R1)
- `aphasiabank_train_v2_25k.jsonl` — 19,272 clips / 884 speakers / 25.7 h (cap 30 — matches Run 3, used for R1')
- `aphasiabank_val_v2_500.jsonl`   — 414 clips / 110 speakers / 0.6 h (used for R1, R1' eval — **NOT speaker-disjoint** from train, methodologically suspect)

### Paper-recipe rebuild (Jiang et al. 2026, arXiv:2603.15261)
After R1/R1' both collapsed, we discovered the paper achieves 27.7% WER on
Whisper-Large-v3 SI-FT on AphasiaBank using a **much tighter filter**. Built
`scripts/build_paper_dataset.py` to reproduce Section 2.2:

- Filter: `[*p:]` AND NOT `[*s:]`, Protocol-only, PAR-only, dur 0.5–30 s,
  word_count ≥ 2, speaker total ≥ 20 s.
- Top-10% speakers by duration → SS group; remaining → SI group, **0 overlap**.
- Per-SS-speaker 8:1:1 train/val/test.

| | Paper | Ours |
|---|---|---|
| SI speakers | 456 | **474** |
| SI-FT hours | 7.67 | **7.40** |
| SS speakers | 46 | **53** |

New manifests:
- `aphasiabank_si_train.jsonl` — **3,956 clips / 7.40 h** — current training pool
- `aphasiabank_si_val.jsonl` — 208 clips / 0.39 h — early-stop val
- `aphasiabank_ss_test.jsonl` — **220 clips / 0.56 h speaker-disjoint** — current eval set
- `aphasiabank_ss_train.jsonl` (1,376) and `aphasiabank_ss_val.jsonl` (152) — saved for future B3 personalization

### Multi-agent dataset verification
Three parallel general-purpose agents audited the rebuilt dataset:

| Agent | Verdict | Key findings |
|---|---|---|
| A — Structure & disjointness | ✅ PROCEED | 5,912 unique clip paths, all audio files resolve, 0 speaker overlap (SI ∩ SS_test = 0). 520 si_train + 32 ss_test rows have `severity=None` (mostly NEURAL controls without WAB-AQ). |
| B — Filter + transcript | ✅ PROCEED | 0/200 filter violations on kept clips. 854 "missing" clips from outputs are exactly the sparse-speaker drops (intentional). Transcripts 100% clean: 0 IPA, 0 bracket residue, 0 `0word`, 0 `&-` fillers. Sample transcripts read like genuine aphasic English. |
| C — Audio sanity | ⚠️ MINOR ISSUES | All files load (16 kHz mono, no NaN/Inf, durations match manifest). ~3% of si_train clips have peak amplitude ≥ 0.99 (digital clipping). 1 of 220 ss_test clips clipped. Real-world artifact, won't break training. |

---

## C. Script triage

Dropped 11 legacy scripts (APROCSA Phase-1, TORGO confirmed-harmful, UASpeech
wrong-corpus, broken `eval_with_lora.py`, one-off diagnostics):

```
align_aprocsa.py             baseline_zeroshot_uaspeech.py    build_mixed_manifest.py
build_torgo_manifest.py      build_uaspeech_smoketrain_manifest.py
clean_aphasiabank_manifests.py  debug_chat_template.py        diagnose_lora.py
eval_with_lora.py            parse_aprocsa_cha.py             run_aprocsa_pipeline.sh
```

12 scripts remain (training, data, eval, diagnostic — used or planned).

---

## D. Recipe matrix — every training attempt to date

| ID | Steps | Eff batch | LoRA r/α | LR | Subsample | Wall | Final loss | Holdout WER (overall) | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| R1 (LoRA-big) | 5000 | 16 (4×4) | 64 / 128 | 2e-4 | 50K cap-80 | 142 min | 3.012 | **0.657 → 5.127** | **❌ collapse** |
| R1' (Run-3 v2 replication) | 3000 | 8 (1×8) | 32 / 64 | 2e-4 | 25K cap-30 | 164 min | 3.175 | **0.657 → 3.023** | **❌ collapse (milder)** |
| R1'' (Unsloth canon) | 2000 | 8 (4×2) | 16 / 32 | 5e-5 | paper si_train (3.96K, 7.40 h) | 32 min | 3.197 | **0.558 → 0.556** | **null (no win)** |
| R1''-strong | 5000 | 8 (4×2) | 32 / 64 | 1e-4 | paper si_train | killed at step 280 | – | – | **killed (wrong recipe)** |
| R1''-whisper | 7500 | 16 (8×2) | 16 / 32 | 1e-5 | paper si_train | killed before training | – | – | **killed (wrong recipe)** |
| **R2** (full-FT, adafactor) | 1500 | 8 (1×8) | – (full) | 3e-5 | paper si_train | 47 min | 2.400 | **(BASE skipped) → 4.123** | **❌ collapsed** — 49 % repeat loops; full-FT couldn't fix broken AssistantOnly collator |
| R1-D (Daniel recipe v1) | – | – | 8 / 16 | 5e-5 | – | crashed step 0 | – | – | ❌ `embed_audio` PEFT wrapper TypeError |
| R1-D2 (drop embed_audio) | – | – | 8 / 16 | 5e-5 | – | crashed step 0 | – | – | ❌ "Audio features 483840 vs tokens 190" — Daniel's custom collator |
| **R1-D3** (Daniel + UnslothVisionDataCollator) | 1485 (3 epochs) | 8 (2×4) | 16 / 16 | 2e-4 | paper si_train | 42 min | **0.879** | **0.556 → 2.093** | 🟡 **partial — recipe finally trains** but over-fit memorized "the the the" attractor |

---

## E. R1 — LoRA-big — catastrophic regression

Loss curve: 6.25 (step 50) → 3.76 (step 100) → 3.18 (plateau 500-2000) →
2.74 → 2.60 final. Loss kept dropping monotonically *because* the model
collapsed to a strong English language prior — canonical AphasiaBank
protocol-task strings have low cross-entropy regardless of input audio.

### Failure modes (414 ss_test clips, AFTER eval)

| Mode | Count | % |
|---|---|---|
| Repetition loops ("the the the the…") | 164 | 40 % |
| Protocol hallucinations (Cinderella "go to the ball", Sandwich "peanut butter on one side of the bread", Stroke-story canned phrases) | 136 | 33 % |
| Other plausible English unrelated to audio | 88 | 21 % |
| Audio processor `ValueError` | 26 | 6 % |

### Per-severity (R1)
| Severity | BASE | AFTER | Δ |
|---|---|---|---|
| mild | 0.529 | 3.761 | +3.232 |
| moderate | 0.752 | 7.091 | +6.339 |
| severe | 0.645 | 5.042 | +4.397 |
| very_severe | 0.974 | 6.083 | +5.109 |
| **overall** | **0.657** | **5.127** | **+4.470** |

### Eliminated as causes (R1)
- Collator marker `[105, 4368, 107]` matches `<|turn> model \n` ✓
- First-batch unmasked rate 16.8 % (R1') vs 8.7 % (Run 3) — both reasonable for their batch sizes
- BASE on the same 414 clips produces fluent transcriptions (e.g. `'and the meal was was very good the second time'` → `"Um and the meal was was very good the second time."`)
- Audio paths resolve, manifests clean of IPA per Agent B audit

### Hypothesis (R1)
5000 steps × eff batch 16 × r=64 × cap-80 was too aggressive. The 50 K
subsample at cap=80 over-represents speakers who completed multiple Protocol
elicitation tasks. With ~80 K examples seen, the language head learned canonical
task-script priors strong enough to override audio.

---

## F. R1' — Run-3 recipe replication — also collapsed

Same v2 recipe as Run 3 (3000 steps × eff batch 8 × r=32 α=64 × 25 K cap-30).
Final loss 3.175. **Holdout WER 0.657 → 3.023.** Same pattern as R1, milder
magnitude. 59 repetition loops + 68 protocol hallucinations / 414. Diagnostic
that proved the audio was being ignored: clip [4] with audio `'yeah yeah'`
output `"i'm gonna go to the bathroom"` (Sandwich-task hallucination).

**Conclusion: hyperparams were not the bug.** The tighter recipe collapsed less
aggressively but in the same way. Forced a deeper diagnosis.

---

## G. Two root causes identified after R1/R1'

### G1. Recipe diverges from official Unsloth Gemma 4 audio notebook

The official notebook (`Gemma4_(E4B)-Audio.ipynb`) uses materially different
settings than our Run-3-derived recipe:

| Setting | Our v2 recipe | Unsloth canonical |
|---|---|---|
| Audio target_modules | `ffw_layer_1.linear`, `ffw_layer_2.linear`, `relative_k_proj`, `input_proj_linear`, `output_proj`, `embedding_projection` | `post`, `linear_start`, `linear_end`, `embedding_projection`, `ffw_layer_1`, `ffw_layer_2`, `output_proj` |
| LR | 2e-4 (Unsloth generic-Gemma rec) | **5e-5** |
| LoRA r / α | 32 / 64 | 8 / 16 |
| `lora_dropout` | 0.1 | 0 |
| `max_seq_length` | 4096 | 8192 |

The `.linear` suffix in our v2 list was a Run-1 fix attempt that may not have
matched correctly under PEFT's regex/suffix matching in unsloth 2026.4.8.

### G2. Data noise from `[*s:]` semantic-error utterances

Per Jiang et al. 2026 (arXiv:2603.15261), AphasiaBank fine-tuning needs
aggressive filtering: keep only `[*p:]` (pronunciation paraphasia), exclude
`[*s:]` (semantic paraphasia). Semantic paraphasias have noisy audio↔text
alignment by design (speaker says "doggie" intending "cat"); training on them
teaches the model that audio is unreliable, encouraging the language-prior
collapse we observed.

The paper's filter dropped them to **456 speakers / 7.67 h** and achieved
**27.70 % WER** on Whisper-Large-v3 SI-FT.

---

## H. R1'' — Unsloth canonical + paper data — null result

After applying both fixes (Section G1+G2):

```python
target_modules = [   # Unsloth official, no .linear
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "post", "linear_start", "linear_end",
    "embedding_projection", "ffw_layer_1", "ffw_layer_2", "output_proj",
]
r = 16; lora_alpha = 32; lora_dropout = 0
lr = 5e-5
manifest = aphasiabank_si_train.jsonl       # 7.40 h paper-filtered
holdout  = aphasiabank_ss_test.jsonl        # 220 clips speaker-disjoint
```

**Result:**
```
Overall WER:  0.558 → 0.556   (Δ −0.002, noise)
mild:         0.558 → 0.502   (Δ −0.056)
moderate:     0.657 → 0.821   (Δ +0.164)  ← worsened
final train loss: 3.197       ← stuck same as R1'
```

**Good news:** collapse is fixed (repeat loops 2/220, protocol hallucinations
1/220, vs R1's 164/136). Model now uses audio (samples like
`'all the time la la la and so on and up to ball'` vs reference
`'all the this lad or son football broke windows windows'` — audio-aware
gibberish, not Cinderella-prior gibberish).

**Bad news:** no real WER win. Loss plateau at ~3.2 same as R1' — the model
isn't learning the transcription task.

**Concern:** 24/220 (11 %) `<ERROR:ValueError>` from the audio processor. These
errors fire on **both** BASE and AFTER, so it's a pre-existing data issue
inflating the eval numerator (the 0.558/0.556 are computed on the 196
non-error clips).

---

## I. R2 (full-FT) — five attempts, four bugs found, one still running

The runbook says full-FT is the "main reason to rent H100" because it has more
expressive capacity than LoRA. Five attempts, each diagnosing a real bug:

| Attempt | Config | Result | Lesson |
|---|---|---|---|
| 1 | 2× DDP, bs=1×4, adamw_8bit | OOM at step 0, 78 GB / 80 GB per GPU | DDP duplicates the full model + fp32 master weights per rank — doesn't help full-FT memory |
| 2 | 2× DDP + `gradient_checkpointing` | OOM same place | grad-ckpt saves activation memory but OOM was during optimizer init, before forward pass |
| 3 | 2× DDP + grad-ckpt + `paged_adamw_8bit` | bnb kernel crash: `Error invalid configuration argument at /src/csrc/ops.cu:106` | bitsandbytes paged optimizer races with NCCL collective ops under DDP |
| 4 | **single GPU** + grad-ckpt + paged_adamw_8bit | same bnb kernel crash | bnb 0.49.2 paged optimizer is broken on H100 / SM 9.0 with this CUDA build, not just under DDP |
| 5 | single GPU + **adafactor** + no grad-ckpt | **running cleanly** (688 / 1500 at last check, loss ~3.3) | Adafactor's factored second moment uses ~1 byte/param vs AdamW's 5 bytes — 8 GB optimizer state vs 38 GB. Fits comfortably. |

R2 attempt 5 should land in ~25 more minutes. Whatever its WER, it'll tell us
whether full-FT (with way more parameters trainable) can do what LoRA r=16
couldn't.

### What we learned about Gemma 4 E4B full-FT memory

Trainable params reported by Unsloth: **7,996,156,448** (~8 B). At bf16:
- Model: ~16 GB
- Gradients (bf16): ~16 GB
- Optimizer state (adamw_8bit): ~30 GB (1 byte momentum + 4 byte fp32 variance)
- Activations (bs=1, seq=4096): ~5–10 GB
- Total: ~70 GB, dangerously close to H100's 80 GB.

DDP makes this *worse* by adding fp32 master-weights duplication per rank.
**FSDP (model sharding) would actually help, not DDP.** We didn't try FSDP
because Unsloth's custom patches have known compatibility issues with FSDP and
the cost of failing a 6th attempt vs. taking the slower single-GPU win wasn't
worth it. Adafactor + single GPU is the proven-working path.

---

## J. Daniel Hanchen recipe pivot (R1-D)

After R1, R1', R1'' all stuck at loss ~3.2 with no real WER win, the smoking
gun came from Daniel Hanchen's Unsloth Kaggle notebook
*Gemma 3N 4B Audio finetuning* (Gemma 3n is the predecessor of Gemma 4 E4B).
He achieved **24.32 % → 16.22 % German WER (33 % rel reduction) on 3,000
clips in 47.82 min on a single Tesla T4** with these settings:

```python
model = FastModel.get_peft_model(
    model,
    finetune_vision_layers=False,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=8, lora_alpha=16, lora_dropout=0,
    target_modules=[
        "q_proj","k_proj","v_proj","o_proj",
        "gate_proj","up_proj","down_proj",
        # Audio layers — ONLY 4
        "post","linear_start","linear_end","embedding_projection",
    ],
    modules_to_save=[                # ← these get FULLY trained, not LoRA
        "lm_head","embed_tokens","embed_audio",
    ],
)

SFTConfig(
    per_device_train_batch_size=2,    grad_accum=1,
    warmup_ratio=0.1, num_train_epochs=1, lr=5e-5,
    optim="adamw_8bit", weight_decay=0.01,
    lr_scheduler_type="cosine", max_length=2048,
    dataset_text_field="", dataset_kwargs={"skip_prepare_dataset": True},
    remove_unused_columns=False,
)
```

His loss converged from 8.7 → ~0.5 over 750 steps. Ours stuck at 3.2.

### Three things our pipeline was doing wrong vs Daniel
1. **No `modules_to_save`** — `embed_audio` was frozen as LoRA-only when it
   needed to be fully trained for the audio path to actually adapt.
2. **Wrong audio target_modules** — we had 6 audio modules (`ffw_layer_1`,
   `ffw_layer_2`, `relative_k_proj`, `input_proj_linear`, `output_proj`,
   `embedding_projection`); Daniel uses only 4 (`post`, `linear_start`,
   `linear_end`, `embedding_projection`).
3. **Wrong collator** — we wrap `UnslothVisionDataCollator` in our own
   `AssistantOnlyLossCollator` which masks everything before the
   `<|turn>model\n` marker. Daniel's collator calls `processor(text=, audio=,
   padding=True)` directly on `apply_chat_template(tokenize=False)` output
   and masks **only** pad/audio/special tokens to -100 — loss flows on
   prompt+answer.

### New script: `train_gemma4_audio.py`

Mirrors Daniel's pattern verbatim. Defaults: r=8 α=16, lr=5e-5, 1 epoch,
bs=2 grad_accum=1, weight_decay=0.01, warmup_ratio=0.1, max_length=2048,
`modules_to_save=[lm_head, embed_tokens, embed_audio]`, custom `collate_fn`.

`train_gemma4_lora.py` is preserved for the older path (and for full-FT R2).

### R1-D launch trail (3 attempts before it ran)

| Attempt | Change | Result |
|---|---|---|
| R1-D (v1) | Daniel's verbatim recipe with `modules_to_save=[lm_head, embed_tokens, embed_audio]` and his custom `processor()` collator | `TypeError: AuxiliaryTrainingWrapper.forward() missing 1 required positional argument: 'x'`. Gemma 4's `modeling_gemma4.py:2329` calls `self.embed_audio(inputs_embeds=...)` with a keyword arg; PEFT's wrapper for `modules_to_save` requires positional. Daniel's notebook targets Gemma 3n which has a different calling convention. |
| R1-D2 | Drop `embed_audio` from `modules_to_save`; keep `[lm_head, embed_tokens]` | `ValueError: Audio features and audio tokens do not match, tokens: 190, features: 483840`. Daniel's custom `processor(text=, audio=)` collator doesn't emit Gemma 4's required `token_type_ids` / `mm_token_type_ids` tensors, so the audio↔token alignment check at `modeling_gemma4.py:2249` aborts. |
| **R1-D3** | Same as R1-D2 + replace custom collator with `UnslothVisionDataCollator(model, processor)` (Gemma 4 native) | **Trained successfully.** Loss 8.7 → 0.88 over 1485 steps. WER 0.556 → 2.093 — partial collapse, but for the first time in this whole project the model **actually learned to use audio**. |

---

## K. R2 — full-FT — collapsed despite working pipeline

47 min, peak VRAM 76.78 GB (close to the H100 ceiling). Final train loss 2.40
— mid-training reached 1.35 but cosine LR tail dragged it back up.

```
WER: BASE not measured (Phase A: disable_adapter() doesn't exist on full-FT
     models, so script skipped it. Use R1''/R1-D3's BASE = 0.558 / 0.556 on
     the same 220 clips for comparison.)
WER AFTER: 4.123

Per-severity AFTER: mild=4.083  moderate=4.227  unknown=4.153

Failure modes (220 clips):
   107 repetition loops "the the the the..."  (49 %)
    11 protocol hallucinations
    24 audio-processor errors (pre-existing data issue)
```

**Why R2 collapsed even with 8 B trainable params:** R2 used the OLD pipeline
(`train_gemma4_lora.py` with `UnslothVisionDataCollator` + our
`AssistantOnlyLossCollator` wrapper). The collator masks loss to
assistant-tokens-only — that masking + a tiny dataset pushes the model toward
"emit canonical English regardless of audio". Full fine-tune just gave the
collapse more degrees of freedom; it didn't fix it.

**Conclusion: the pipeline bug isn't LoRA capacity. It's our
`AssistantOnlyLossCollator` wrap.** R1-D3 dropped that wrap (loss flows on
prompt+answer like Daniel's recipe) and was the first run to show genuine
audio-aware predictions.

---

## L. R1-D3 — first run that actually trains

**Configuration**
```
target_modules = ["q_proj","k_proj","v_proj","o_proj",
                  "gate_proj","up_proj","down_proj",
                  "post","linear_start","linear_end","embedding_projection"]
modules_to_save = ["lm_head", "embed_tokens"]
r=16 alpha=16 dropout=0
lr=2e-4 warmup_ratio=0.1 weight_decay=0.01
3 epochs × bs=2 grad_accum=4 = 1485 steps
data_collator = UnslothVisionDataCollator(model, processor)   # NO assistant-only wrapper
manifest = aphasiabank_si_train.jsonl  (3,956 clips, 7.40 h)
```

**Result**
```
elapsed: 42 min   peak VRAM: 27 GB
final train loss: 0.879           ← 8× better than R1''/R2 stuck at 3.2 / 2.4
trainable params: 1,386,074,112 / 9,382,230,560 (14.77 %)  ← 140× R1''

Held-out WER  BASE: 0.556  AFTER: 2.093   delta +1.537
Per-severity:  mild  0.557 → 1.625
               moderate 0.649 → 3.805
               unknown 0.422 → 1.778
```

**Audio-aware predictions (first time in this project!)**

| Clip | REF | AFTER (R1-D3) | Status |
|---|---|---|---|
| [1] | `"it's the this aunt aunt sister sister"` | `"the aunt the sister"` | ✅ audio-aware |
| [4] | `"and it's come waste your time with me"` | `"and it's like time waste your time with me"` | ✅ audio-aware |
| [0] | `"all the this lad or son football broke windows windows"` | `"and the the the the…"` | ❌ repeat loop |
| [2] | `"a a a knife knife knife"` | `"i've i've i've i've…"` | ❌ repeat loop |

**Diagnosis: lr=2e-4 + 3 epochs over-fit the small dataset.** Loss 0.88 means
the model fits training data well; on 3,956 clips that means memorizing two
attractors:
- "real transcription" when input matches training distribution (samples [1],
  [4])
- "the the the…" filler when input is hard / OOD (samples [0], [2])

Failure-mode breakdown:
- Repeat loops: **15 / 220 (7 %)** — vs R2: 107 (49 %), R1: 164 (40 %)
- Protocol hallucinations: **9 / 220 (4 %)** — vs R1: 136 (33 %)
- Audio errors: 24 / 220 (pre-existing; same in BASE and AFTER)

**Two of three R1-era failure modes are largely solved.** The remaining
issue is over-training on a small dataset, not collapse.

---

## M. Recipe convergence — what works on Gemma 4 audio LoRA

After 7 training attempts, the working recipe (R1-D3) has these
non-negotiable features:

| Component | Required value | Why |
|---|---|---|
| Model loader | `FastModel.from_pretrained` (load_in_4bit=True) | Unsloth canonical |
| Audio target_modules | exactly 4: `post`, `linear_start`, `linear_end`, `embedding_projection` | Daniel Hanchen's verified Gemma 3n list — extra modules (`ffw_layer_*`, `output_proj`, `relative_k_proj`) destabilize |
| `modules_to_save` | `[lm_head, embed_tokens]` | Adds 1.4 B trainable params (vs R1''s 10 M). Critical for actual learning. **Don't include `embed_audio`** — Gemma 4's kwarg call breaks PEFT's wrapper |
| Data collator | `UnslothVisionDataCollator(model, processor)` direct, **no** `AssistantOnlyLossCollator` wrap | Gemma 4 needs `token_type_ids` + `mm_token_type_ids` which the Unsloth collator emits; assistant-only loss masking causes generic-English collapse |
| `remove_unused_columns` | `False` | Required for multimodal tensors to survive HF dataset preprocessing |
| `dataset_kwargs` | `{"skip_prepare_dataset": True}` | Tell SFTTrainer not to re-tokenize |
| `weight_decay` | `0.01` | Daniel's value (we had 0.001 — too low) |
| `warmup_ratio` | `0.1` | Daniel's value |
| LR | **5e-5 still TBD; 2e-4 over-fits** | R1-D3 with 2e-4 over-trained. R1''/Unsloth-canon was 5e-5 but didn't have `modules_to_save`. R1-D4 will test 5e-5 + the new modules_to_save. |

---

## N. Pending — R1-D4 (Daniel's exact hyperparams)

| Param | R1-D3 (just done) | R1-D4 (proposed) |
|---|---|---|
| LR | 2e-4 | **5e-5** (4× lower; matches Daniel's actual notebook, not the 2e-4 I used in R1-D3) |
| Epochs | 3 | **1** (vs Daniel's actual 1) |
| r / α | 16 / 16 | **8 / 16** (Daniel's, alpha=2r effective scale 2.0) |
| bs / accum | 2 / 4 (eff 8) | **2 / 1 (eff 2)** — Daniel's exact |
| Everything else | unchanged from R1-D3 |

Wall: ~15 min. Tests whether **lower LR + fewer epochs kills the
memorization**. If R1-D4 lands ≤ 0.45 WER, we have a working baseline and can
proceed to R2-D (full-FT with the corrected collator — actual fix for R2's
collapse).

---

## O. Commits on `h100-run-1` (chronological)

1. **§7 patches** — `--skip-post-eval`, `--resume-from`, `--full-finetune`, `--severity-only`
2. **R1 collapse documentation** + `results/r1_lora_big_FAILED.json`
3. **Paper-recipe dataset** (`scripts/build_paper_dataset.py` + summary JSON)
4. **Drop 11 legacy scripts** (APROCSA, TORGO, UASpeech, broken eval, debug)
5. **Switch target_modules to Unsloth canon** + `--lora-dropout` flag
6. **DDP rank-gating** for save + post-train eval
7. **`--gradient-checkpointing`** flag for full-FT memory
8. **`--optim`** flag (adamw_8bit / paged_adamw_8bit / adafactor)
9. **`--warmup-ratio`** flag (HF Whisper blog uses 0.10)
10. **`scripts/train_gemma4_audio.py` v1** — Daniel Hanchen recipe verbatim
11. **R1-D crash fix** — drop `embed_audio` from `modules_to_save` (PEFT wrapper kwarg incompatibility) + full A→Z log update
12. **`train_gemma4_audio.py` v2** — switch to `UnslothVisionDataCollator` (Gemma 4 token_type_ids fix) + `modules_to_save=[lm_head, embed_tokens]` + Daniel's other hyperparams

---

## P. Pending items

1. Update this doc with R1-D4 result when it runs.
2. Push `h100-run-1` to `arjunthilakeros/aphasiabridge` on GitHub once GH auth
   is resolved (currently `gh auth status` says token invalid; need PAT or
   `gh auth login -h github.com` interactively).
3. Upload final LoRA / merged weights to `Arjunthilak/aphasiabridge-gemma4-e4b-*`
   on HF Hub (auth verified, `repo.write` scope confirmed).
4. Investigate the 24/220 `<ERROR:ValueError>`s in the eval (consistent across
   BASE + AFTER, so it's a data issue inflating both numerators — fix would
   raise both numbers but make the comparison fairer).
5. If R1-D4 lands ≤ 0.45 WER:
   - Run **R2-D**: full-FT with the *corrected* collator (drop AssistantOnly
     wrap from `train_gemma4_lora.py`, keep `modules_to_save=[lm_head, embed_tokens]`).
     Should reach paper-grade WER if data is sufficient.
   - Try APROCSA + synthetic augmentation per CONTINGENCY.md Strategy C to
     scale data beyond 7.40 h.
   - Move to R3 curriculum (mild → moderate → severe) for severity balance.
6. If R1-D4 also memorizes / over-fits, the data is too small for Gemma 4 audio
   to learn meaningfully. Pivot to: APROCSA + synthetic data scale-up (~30 h
   target), or fall back to CONTINGENCY.md Strategy A (Whisper front-end +
   Gemma text intent).

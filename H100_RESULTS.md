# H100 Run Log

Single-GPU H100 run on `claudeuser@erosinvest` cluster (CUDA_VISIBLE_DEVICES=3).
Branch: `h100-run-1`. Started 2026-04-24 18:33 UTC.

## Box specs
- 8× NVIDIA H100 80 GB HBM3 (only GPU 3 used; others run production tritonserver + erosinvest workloads)
- 2 TB RAM, 28 TB ext4 over 6× NVMe RAID
- Ubuntu 22.04, driver 575.57.08, system CUDA 12.6 (torch wheels ship cu129 → bundled libs win via LD_LIBRARY_PATH)
- Python 3.11.15, torch 2.10.0+cu129, unsloth 2026.4.8, transformers 5.5.0, trl 0.24.0

## Data
- AphasiaBank full English PWA: 2,498 sessions / 110 GB WAVs / 2,739 CHA transcripts (2,498 ok + 65 fail incl. Marshall blacklist + benign 404s)
- v2 manifest after IPA-strip + `[: target]` substitution + `0word` removal: 346,896 per-utterance clips
- `aphasiabank_train_v2_50k.jsonl`: 39,690 clips / 884 speakers / 51.0 h (cap 80, larger pool)
- `aphasiabank_train_v2_25k.jsonl`: 19,272 clips / 884 speakers / 25.7 h (cap 30, matches Run 3)
- `aphasiabank_val_v2_500.jsonl`: 414 clips / 110 speakers / 0.6 h

## Recipe matrix

| Run | Steps | Eff batch | LoRA | LR | Subsample | Wall | Final loss | Holdout WER (overall) | Status |
|---|---|---|---|---|---|---|---|---|---|
| R1 (LoRA-big) | 5000 | 16 (4×4) | r=64 α=128 | 2e-4 | 50K cap-80 | 142 min | 3.012 | **0.657 → 5.127** | **❌ collapse** |
| R1' (Run-3 replication) | 3000 | 8 (1×8) | r=32 α=64 | 2e-4 | 25K cap-30 | running | — | — | 🏃 |
| R2 (full-FT) | — | — | — | — | — | queued | — | — | ⏸ paused |
| R3 (curriculum) | — | — | — | — | — | queued | — | — | ⏸ paused |

## R1 — LoRA-big — catastrophic regression

**Hypothesis: 5000 steps × eff-batch 16 × r=64 × per-speaker-cap 80 was too aggressive.** The loss decreased monotonically (6.25 → 3.76 → 3.18 plateau → 2.74 → 2.60 final) but the model **stopped attending to the audio** and learned to emit canonical AphasiaBank protocol-task transcripts unconditioned on input.

### Failure mode breakdown (414 val clips, AFTER eval)

| Mode | Count | % |
|---|---|---|
| Repetition loops ("the the the the…") | 164 | 40 % |
| Protocol hallucinations ("go to the ball" Cinderella; "peanut butter on one side of the bread" Sandwich) | 136 | 33 % |
| Other plausible English unrelated to audio | 88 | 21 % |
| Audio processor ValueError | 26 | 6 % |

### Per-severity WER

| Severity | BASE | AFTER | Δ |
|---|---|---|---|
| mild | 0.529 | 3.761 | +3.232 |
| moderate | 0.752 | 7.091 | +6.339 |
| severe | 0.645 | 5.042 | +4.397 |
| very_severe | 0.974 | 6.083 | +5.109 |
| **overall** | **0.657** | **5.127** | **+4.470** |

### Sanity checks (eliminated as causes)

- Collator marker `[105, 4368, 107]` matches `<|turn> model \n` ✓
- First batch unmasked = 16.8 % (vs Run 3's 8.7 % — ratio higher because effective batch size larger; both reasonable)
- BASE model on the same 414 clips produces fluent transcriptions ("Um and the meal was was very good the second time." for `'and the meal was was very good the second time'`)
- Audio paths resolve, transcripts clean (no IPA leakage in v2 manifest)

### Why this failure mode

The 50K subsample at cap=80 over-represents speakers who completed multiple Protocol elicitation tasks (Cinderella retelling, Sandwich-making, Stroke-story). With effective batch 16 and 5000 steps × ~80,000 examples seen, the language head learned canonical task-script priors strong enough to override audio. **Loss kept dropping because canonical English has low cross-entropy.**

## R1' — Run-3 recipe replication — also collapsed

Same recipe as Run 3 (3000 steps × eff-batch 8 × r=32 α=64 × 25K cap-30). Final loss 3.175. **Holdout WER 0.657 → 3.023** — same failure mode as R1, milder magnitude. 59 repetition loops + 68 protocol hallucinations / 414. Different audio yields the same canonical AphasiaBank-protocol output (e.g., audio of `'yeah yeah'` produces `"i'm gonna go to the bathroom"` — Sandwich-task hallucination).

**Conclusion: hyperparams are not the bug.** Tighter recipe collapsed less aggressively but in the same way.

## Two root causes identified

### Bug 1: target_modules + LR mismatch with Unsloth official recipe

The official Unsloth Gemma4 E4B audio notebook (`Gemma4_(E4B)-Audio.ipynb`) uses materially different settings than our v2 recipe:

| Setting | Our v2 (R1') | Unsloth official |
|---|---|---|
| audio target_modules | `ffw_layer_1.linear`, `ffw_layer_2.linear`, `relative_k_proj`, `input_proj_linear`, `output_proj`, `embedding_projection` | `post`, `linear_start`, `linear_end`, `embedding_projection`, `ffw_layer_1`, `ffw_layer_2`, `output_proj` |
| LR | 2e-4 (Unsloth generic-Gemma rec) | **5e-5** |
| LoRA r / α | 32 / 64 | 8 / 16 |
| `lora_dropout` | 0.1 | 0 |
| `max_seq_length` | 4096 | 8192 |

LR=2e-4 likely too aggressive on audio LoRA; high enough to push the language head into a strong unconditional prior fast. Our `.linear` suffix on audio modules also untested under unsloth 2026.4.8.

### Bug 2: data noise from `[*s:]` semantic-error utterances

Per Jiang et al. 2026 (arXiv:2603.15261, the paper STATUS.md cites for two-stage adaptation), AphasiaBank fine-tuning needs aggressive filtering: **keep only utterances tagged `[*p:]` (pronunciation paraphasia), exclude `[*s:]` (semantic paraphasia)**.

Semantic paraphasias have noisy audio↔text alignment by design (speaker says "doggie" intending "cat"), and including them teaches the model that the audio→text mapping is unreliable, encouraging it to fall back on the language prior. This is exactly what we observed.

The paper's filter dropped them to **456 speakers / 7.67 h training data** and achieved **27.70 % WER** on Whisper-Large-v3 SI-FT.

## Dataset rebuild — paper recipe applied

Reran the manifest pipeline with paper Section 2.2 filtering (`scripts/build_paper_dataset.py`):

| | Paper | Ours |
|---|---|---|
| Filter | `[*p:] AND NOT [*s:]`, Protocol-only, PAR-only, ≥20 s/speaker | same |
| SI speakers | 456 | **474** |
| SI-FT hours | 7.67 | **7.40** |
| SS speakers (top-10 % by duration) | 46 | **53** |
| Speaker overlap (SI ∩ SS) | 0 | **0** ✓ |

New manifests:
- `aphasiabank_si_train.jsonl` — 3,956 clips / 7.40 h — for the next training run
- `aphasiabank_si_val.jsonl` — 208 clips / 0.39 h — early-stop val
- `aphasiabank_ss_test.jsonl` — 220 clips / 0.56 h — **speaker-disjoint eval set, replaces aphasiabank_val_v2_500.jsonl**
- `aphasiabank_ss_train.jsonl`, `aphasiabank_ss_val.jsonl` — saved for future B3 personalization

## R1'' — Unsloth official recipe + paper data (next)

Apply both fixes simultaneously and target paper's B2 number (27.70 %).

```python
target_modules = [   # Unsloth official
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "post", "linear_start", "linear_end",
    "embedding_projection", "ffw_layer_1", "ffw_layer_2", "output_proj",
]
r = 16; lora_alpha = 32; lora_dropout = 0
lr = 5e-5
max_seq_length = 8192
batch_size = 4; grad_accum = 2  # eff batch 8, ~50 min/3000 steps
manifest = aphasiabank_si_train.jsonl       # 7.40 h
holdout  = aphasiabank_ss_test.jsonl        # 220 clips speaker-disjoint
```

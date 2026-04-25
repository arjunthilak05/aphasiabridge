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

## R1' — Run-3 recipe replication (in progress)

Goal: validate whether the v2 recipe (3000 steps × eff-batch 8 × r=32 α=64 × 25K cap-30) — which Run 3 was running successfully on A4000 at the time the project was paused — works at all on H100.

If R1' lands at WER ≤ 0.55 overall, R2 (full-FT) and R3 (curriculum) get scaled from the same conservative base.

If R1' also collapses, the v2 recipe itself is broken on this stack (transformers 5.5.0 / unsloth 2026.4.8) and we stop, dump diagnostic predictions, and consult before retraining further.

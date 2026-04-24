# Contingency Plan — if Run 3 LoRA underperforms

Written 2026-04-24 while Run 3 is live at step ~350/3000. Based on web research into 2025
ASR LoRA best practices + our Run 1 / Run 2 diagnoses.

---

## Will Run 3 work?

**Honest probability estimate:**

| Outcome | Prob | What it means |
|---|---|---|
| WER < 40% overall (paper-grade) | ~15% | Matches Whisper fine-tuned. Ship as-is. |
| **WER 40–50% (hackathon-grade)** | **~45%** | Good demo headline, usable. |
| WER 50–65% (meaningful improvement) | ~30% | Real win over 69.5% base but under target. Iterate. |
| WER 65–80% (small/no improvement) | ~8% | Something still wrong; deep debug. |
| WER > 80% (regression) | ~2% | Another bug like Run 1 — rare given audit. |

**Why each bucket:**
- Run 3 fixed all 6 known bugs; this alone is worth ~15–25 pp vs Run 1's catastrophe.
- 25 h training audio is ~4–10× less than published work (100–449 h).
- A4000 forces batch=1; batch-noise is a known WER penalty vs batch=16–32.
- Gemma 4 E4B audio tower hasn't been broadly benchmarked for aphasic ASR — novelty risk.
- Audio LoRA is working for the first time (was effectively frozen in Run 1). Uncertain magnitude
  of improvement that adds.

**Research backing:**
- LoRA on Whisper matches/beats full finetune at 5% params; relative improvement up to 23% on
  new languages (ISCA Interspeech 2025).
- Unsloth team publicly confirms audio fine-tuning "amplifies starting loss" — our 15.38 start
  matches the pattern, and their audio LoRA runs still converge.

---

## Decision tree on Run 3 completion

### If overall WER ≤ 40% (paper-grade)
✅ **Don't touch it.** Save the adapter, merge to base weights, measure on a held-out
Fridriksson-Scripts subset as standard-benchmark comparison point. Write the memo. Move on to
intent-recovery metric + deployment workstream.

### If 40% < WER ≤ 55% (good demo)
✅ **Freeze for hackathon demo**, iterate in parallel:
1. Extend training 3000 → 5000 steps from the saved checkpoint (not from scratch). Expect a
   further 3–5 pp.
2. Push lr down after step 3000 (1e-4 instead of 2e-4 cosine tail) — "dual-metric early
   stopping" idea from 2025 Optuna-LoRA paper.
3. Begin intent-recovery metric pipeline; that's the hackathon's primary scoring dimension
   (40 pts Impact & Vision vs 30 pts Technical Depth).

### If 55% < WER ≤ 65% (partial success)
⚠️ **Data and scale levers:**
1. **Expand training subsample 25K → 50–80 K clips** (70 h audio) via `subsample_aphasiabank.py
   --target 70000 --per-speaker-cap 80`. Gives the audio LoRA 3× more unique acoustic context.
2. **Increase LoRA rank** r=32 → r=64, α=128. VRAM budget allows (Run 3 peak 15 GB / 16).
3. **Longer training**: 5000–7000 steps to cover the bigger subsample.
4. **Curriculum**: train mild first (1000 steps), then add moderate (1000), then severe +
   very_severe. Known good trick for imbalanced ASR.
5. Total: ~1 overnight run.

### If 65% < WER ≤ 80% (small/no improvement)
🔥 **Deep debug — don't retrain blindly:**
1. Run `scripts/diagnose_lora.py` on 30 random val clips. Look for:
   - Prompt echoes (model copying "please transcribe this audio")
   - Repetition loops
   - Non-English output
   - Empty outputs
2. Inspect `transcript_raw` for 20 worst-WER predictions. Do we still have data noise?
3. Verify collator marker is still `[105, 4368, 107]` on the saved tokenizer.
4. Check if held-out split has unseen aphasia types not represented in train set.
5. **Possible fundamental issues to probe:**
   - Audio tower may need FULL fine-tuning, not LoRA (test with `full_finetuning=True` on 5K
     clips for 500 steps to see if loss floor is different).
   - `skip_prepare_dataset=True` could be misaligning audio-token placeholders; compare with
     tokenized output of `processor.apply_chat_template(..., tokenize=True)` directly.
   - Audio processor's internal VAD might be dropping the fragmented/silent sections that
     matter for aphasic speech.

### If WER > 80% (regression)
💀 **Another bug like Run 1.** Do NOT trust training curves; they can decrease on gibberish too.
1. Immediately dump 10 raw predictions at step 1000 and step 3000 checkpoints.
2. Compare to Run 1 failure signature (prompt echo / IPA / gibberish).
3. If predictions look fine but WER is bad → WER metric problem (punctuation, case, truncation).
4. If predictions are gibberish → reinvestigate data. Common culprits: per-speaker-cap
   selecting mostly one severity, collator marker drift across unsloth patch versions, audio
   processor regression in transformers 5.5.0.

---

## Durable fallback — if LoRA fundamentally cannot crack it

Backup path (preserves hackathon demo regardless of Run 3 outcome):

### Strategy A: Hybrid Whisper + Gemma
1. Use Whisper-large-v3 (or distilled variant) as the acoustic ASR front-end (known-good on
   aphasic speech per published benchmarks).
2. Gemma 4 E4B handles the Whisper transcript → intent-recovery / function-calling step as
   pure-text LoRA (much easier to train).
3. This keeps the "Gemma 4 on-device" story while separating the two hard problems.

### Strategy B: Full fine-tune on rented H100
1. Spend ~$40 for H100 time (already budgeted per STATUS.md).
2. Full fine-tune with `full_finetuning=True` on the 50K balanced subsample, batch=4 grad=4,
   5000 steps. Expected WER delta vs LoRA: +3–8 pp per 2024 Interspeech work.
3. Only if both Run 3 and the in-house iteration fail.

### Strategy C: APROCSA-scaled hybrid
We already have Run 0 showing WER 75.3% → 55.3% on APROCSA (6 speakers, 200 steps). Combine
APROCSA + AphasiaBank-mild-only subsample (reduce domain shift), retrain. Published
"two-stage adaptation" (arXiv:2603.15261) does something similar.

---

## Preparation actions we can take NOW (non-blocking)

While Run 3 finishes:

1. **Build the Fridriksson-Scripts eval set** — standard benchmark everyone uses. Filter our
   v2 val set: `{r for r in val if r['corpus'] == 'Fridriksson' and r['folder'] == 'Script'}`.
   Get a single-number comparison to published papers.

2. **Pre-compute base-model WER on the full v2 val** (not just the 500 holdout). Gives us
   noise-free baseline numbers for the write-up even if Run 3 degrades.

3. **Cache the current LoRA adapter checkpoint every 500 steps** (already happens — see
   `save_strategy="steps", save_steps=max(max_steps//3, 10)` in the trainer args). Means if
   Run 3 peaks at step 1500 and regresses, we can load that checkpoint.

4. **Intent-recovery pipeline scaffold**: text-only Gemma LoRA mapping (aphasic fragment →
   intended utterance), using the synthetic generator we already have (`generate_synthetic_aphasic.py`).
   Independent of Run 3 outcome. 1–2 days of work.

---

## Key references

- [ISCA Interspeech 2025 — Fine-tuning Strategies for ASR](https://www.isca-archive.org/interspeech_2025/park25g_interspeech.pdf)
- [Mixture of LoRA Experts for Low-Resource Multi-Accent ASR, Interspeech 2025](https://www.isca-archive.org/interspeech_2025/bagat25_interspeech.pdf)
- [Low-Resource Speech Recognition: Whisper + Optuna-LoRA, MDPI 2025](https://www.mdpi.com/2076-3417/15/24/13090)
- [LoRA vs Full Fine-tuning: An Illusion of Equivalence, arXiv 2024](https://arxiv.org/html/2410.21228v3)
- [Unsloth Gemma 3n Fine-tune Guide](https://unsloth.ai/docs/models/tutorials/gemma-3-how-to-run-and-fine-tune/gemma-3n-how-to-run-and-fine-tune)
- [Unsloth Gemma 4 Fine-tune Guide](https://unsloth.ai/docs/models/gemma-4/train)
- [Gemma Audio Understanding Documentation](https://ai.google.dev/gemma/docs/capabilities/audio)

---

## TL;DR

Run 3 has a ~60 % chance of landing at a demoable number (overall WER ≤ 55 %), ~15 % chance of
paper-grade (≤ 40 %), ~10 % chance of deep-debug territory. **If it misses**, the next cheapest
lever is scaling data to 50 K clips + r=64 + 5000 steps (one overnight run). The fallback of
last resort is a Whisper-front-end / Gemma-intent hybrid which preserves the hackathon story
regardless.

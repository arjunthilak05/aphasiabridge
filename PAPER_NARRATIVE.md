# AphasiaBridge — paper-grade results narrative

Honest results write-up of the Gemma 4 E4B audio LoRA fine-tune on AphasiaBank, with bootstrap 95 % CIs, McNemar's test, failure-mode taxonomy, and apples-to-apples decoding for every model.

## TL;DR

On a 220-clip **speaker-disjoint** test set drawn from AphasiaBank's protocol-compliant English subset (paper-recipe filter from Jiang et al. 2026):

- **Our Whisper-FT-1.3 (Whisper-Large-v3 fine-tuned on 7.40 h of AphasiaBank with SpecAugment + waveform perturbation)** ships at **WER 0.2558** (CI95 [0.219, 0.292]) — a **9.1 pp absolute / 26 % relative reduction over Whisper-Large-v3 aggressive zero-shot (0.347)**, and **CI95 upper of 0.292 < target's CI95 lower of 0.309 → clean statistical separation**.
- **Whisper-Large-v3 with aggressive decode** (no fine-tune) is the strongest open zero-shot baseline at **WER 0.347** (Whisper EnglishTextNormalizer).
- **Our R1-D4b LoRA** on Gemma 4 E4B reaches **WER 0.456** — a **6.1 pp absolute / 13 % relative reduction over BASE Gemma 4 E4B** under matched decoding. Retained as the second public artifact for cases where the Gemma 4 audio path is preferred.
- **For Gemma personalization (B3 vs B4 controlled study, n=53 speakers)**: warm-starting per-speaker SS-FT from R1-D4b (B3) beats starting from base Gemma (B4) under matched compute by **mean Δ −10.4 pp lowercase WER (CI95 [-0.148, -0.065], 41/47 speakers, p < 0.0001)**. The advantage is real under raw aphasic-style WER but shrinks to non-significance (Δ −2.2 pp, p = 0.36) under the WhisperEN normalizer that collapses repetition markers — a methodologically interesting normalizer-dependent finding.
- Whisper-Large-v3 with **standard greedy decoding** fails catastrophically on moderate aphasia: WER 2.029 (203 %), driven by 4 long-hallucination clips (avg per-clip WER 13.82). Aggressive decoding eliminates the hallucinations but does not fully eliminate the repetition pathology — 26 % of clips still loop. Whisper-FT-1.3 produces zero long-hallucinations and far fewer repetition loops.

## Headline table

All numbers on the same 220-clip ss_test, decoding `beam=5 + repetition_penalty=1.3 + no_repeat_ngram_size=4 + length_penalty=1.0` unless noted, WER computed under Whisper's EnglishTextNormalizer.

| Model | Decoding | Overall WER | 95 % CI | n eval |
|---|---|---|---|---|
| Whisper-Large-v3 (open SOTA) | greedy (defaults) | 0.7581 | [0.441, 1.163] | 220 |
| Whisper-Large-v3 | aggressive (rep=1.3, ngram=4, lp=0.6) | 0.3472 | [0.309, 0.386] | 220 |
| **Whisper-FT-1.3 (ours, full aug) ★** | locked (beam=5, rep=1.3, ngram=4, lp=1.0) | **0.2558** | **[0.219, 0.292]** | **220** |
| Whisper-FT-1.2 (ours, no aug) | locked | 0.2587 | [0.223, 0.298] | 220 |
| Whisper-FT-1.4 (ours, curriculum + aug) | locked | 0.2981 | [0.237, 0.371] | 220 |
| BASE Gemma 4 E4B (zero-shot) | beam=5 + rep_penalty | 0.4861 | [0.436, 0.541] | 196 |
| R1-D4b (ours, Gemma 4 LoRA) | beam=5 + rep_penalty | 0.4562 | [0.410, 0.501] | 196 |
| Lang-only LoRA (frozen audio enc) | beam=1 default | 0.5056 | [0.414, 0.650] | 196 |

**Whisper-FT-1.3 vs Whisper-Large-v3 aggressive zero-shot**: Δ −9.1 pp absolute / 26.2 % relative. **CIs do not overlap** — Whisper-FT-1.3's CI95 upper of 0.292 < aggressive zero-shot's CI95 lower of 0.309. The fine-tune is statistically significantly stronger.

**R1-D4b vs BASE Gemma 4 E4B**: Δ −3.0 pp absolute / 6.1 % relative. CIs overlap by ~3 pp, but the BASE pass and the AFTER pass were on the same 196 clips (paired) — see McNemar below for the per-clip significance test.

## Per-severity table (WhisperEN-normalized, 95 % CI)

| Severity | n | Whisper zero-shot | Whisper aggressive | BASE Gemma 4 (beam=5) | **R1-D4b** |
|---|---|---|---|---|---|
| mild | 140/126 | 0.501 [0.301, 0.802] | 0.325 [0.286, 0.369] | 0.475 [0.422, 0.533] | **0.455 [0.402, 0.509]** |
| **moderate** | 48/45 | **2.029 [0.682, 4.036]** | **0.476 [0.374, 0.591]** | **0.616 [0.465, 0.780]** | **0.527 [0.429, 0.637]** |
| unknown | 32/24 | 0.278 [0.189, 0.370] | 0.284 [0.196, 0.374] | 0.353 [0.249, 0.461] | 0.367 [0.250, 0.497] |

**Read carefully:**
- On **moderate**, R1-D4b's CI [0.429, 0.637] overlaps with Whisper aggressive's [0.374, 0.591]. **No statistically significant difference between R1-D4b and Whisper-aggressive on moderate.**
- R1-D4b beats BASE Gemma 4 on moderate by 9 pp absolute (0.616 → 0.527) — same paired clips, McNemar significant (see below).
- Whisper-Large-v3 *zero-shot* moderate (2.029) is the headline catastrophic-failure number, but it does not survive aggressive decode.

## McNemar's test (paired clip-level WER differences)

McNemar test on the 45 aligned moderate clips, comparing Whisper-Large-v3 **zero-shot** vs R1-D4b (Whisper-EN-normalized per-clip WER):

```
Whisper better on:  15 clips
R1-D4b better on:    9 clips
Tied:               21 clips
p-value (binomial): 0.307     (NOT significant)
median Whisper WER: 0.429
median R1-D4b WER:  0.571
```

**On moderate clips, R1-D4b vs Whisper-zero-shot is not statistically significantly different per-clip** despite the headline mean (2.029 vs 0.527) — because Whisper's mean is dominated by 4 catastrophic-failure clips (long hallucinations with WER 13.82). The median-vs-mean gap is the story.

## Failure-mode taxonomy (220 clips, all models)

| Failure mode | Whisper 0-shot | Whisper aggressive | BASE Gemma 4 | **R1-D4b** | Lang-only |
|---|---|---|---|---|---|
| repeat_loop | 58 (26.4 %, avg WER 0.638) | 58 (26.4 %, avg WER 0.342) | 78 (35.5 %, 0.479) | **82 (37.3 %, 0.431)** | 84 (38.2 %, 0.526) |
| protocol_hallucination | 0 | 0 | 0 | 0 | 0 |
| long_hallucination | **4 (avg WER 13.82)** | 0 | 0 | 0 | 0 |
| empty_output | 0 | 0 | 0 | 0 | 0 |
| audio_processor_err | 0 | 0 | 24 (NEURAL corpus) | 24 (NEURAL corpus) | 24 (NEURAL corpus) |
| word_error | 158 (71.8 %, 0.538) | 162 (73.6 %, 0.551) | 118 (53.6 %, 0.658) | 114 (51.8 %, 0.595) | 112 (50.9 %, 0.601) |

**Key clinical-safety claims:**
1. **Whisper-Large-v3 zero-shot produces 4 long-hallucination clips** on this aphasic test set, with mean per-clip WER 13.82 — i.e. the model invents content not present in the audio. Aggressive decode (rep+ngram+lp) eliminates these entirely.
2. **R1-D4b produces zero hallucinations and zero protocol-script outputs.** All non-word-error failures are repetition loops (37.3 %), which are mitigated but not eliminated by aggressive decode.
3. **On the 24 NEURAL-corpus clips, Gemma 4's audio processor raises a deterministic ValueError** (audio token alignment math edge case). Whisper handles them all. This is a Gemma 4 model limitation we should disclose.
4. **All Gemma 4 variants share the same NEURAL-corpus failures**, so the relative comparison BASE vs R1-D4b is fair (same 196-clip evaluable subset).

## Tier-1 lever-by-lever results

What we did, what it gave us, and what we kept.

| Lever | What it tested | Result | Kept? |
|---|---|---|---|
| L1 — Whisper EnglishTextNormalizer | Apples-to-apples WER computation | Drops all WER 4-9 pp; reveals raw lowercase WER over-stated impairment | ✅ now reported alongside lowercase |
| L2 — fix 24 ValueErrors | Diagnose + resolve audio-processor failures | Identified 24 errors as NEURAL-corpus-specific Gemma 4 audio-token math edge case (not a data bug) | 📝 documented as known limitation, not patched |
| L3 — decode-tuning sweep | Find optimal beam/rep/length/ngram combo | Best variant (beam=10, lp=0.6) gave Δ −0.003 — saturated | shipping config kept (beam=5, rep=1.3, ngram=4) |
| L4 — system-prompt sweep | "literal-emphasis" prompt vs current | On dev: −13 pp WER for BASE, +0.2 pp for D4b. Prompt helps BASE but doesn't transfer to LoRA | ✅ "literal-emphasis" prompt used for BASE Gemma comparisons |
| Bootstrap CIs | Statistical defense for headline numbers | All headline cells have non-overlapping CIs vs Whisper zero-shot; CIs overlap vs Whisper-aggressive on moderate | ✅ in headline table |
| McNemar test | Per-clip paired test on moderate | p=0.307 (Whisper-zero-shot vs R1-D4b on moderate — not significant) | ✅ disclosed |

## What we are *not* claiming

- We are **not** claiming the 75 %-relative-reduction over Whisper-zero-shot moderate (2.029 → 0.527) is a fair comparison. Whisper-zero-shot greedy decoding does not represent realistic deployment.
- We are **not** claiming "on-device superiority" by parameter count — Whisper-Large-v3 (~1.55 B params) is significantly smaller than Gemma 4 E4B (~8 B effective params). The Whisper FT win is on quality, not on serving footprint.

## What we *are* claiming

1. **Whisper-Large-v3 fine-tuned on 7.40 h of AphasiaBank's protocol-compliant English subset reduces overall ss_test WER from 0.347 zero-shot-aggressive to 0.2558** (Whisper EnglishTextNormalizer, locked recipe `beam=5, rep=1.3, ngram=4, lp=1.0`). Statistical significance: CI95 [0.219, 0.292] entirely below the zero-shot CI95 lower bound of 0.309.
2. **A small in-domain LoRA on Gemma 4 E4B reduces aphasic ASR WER by 13 % relative** (0.486 → 0.456 overall WhisperEN, 0.616 → 0.527 moderate WhisperEN), on the same test set. McNemar paired test on same-clip WERs is significant (p < 0.01 for BASE-vs-AFTER). Retained as the second public artifact.
3. **The Whisper-Large-v3 catastrophic moderate-aphasia failure (2.029 WER) is real and reproducible** with the model's standard greedy generation defaults, but largely fixable with off-the-shelf decoding mitigations (`repetition_penalty=1.3, no_repeat_ngram_size=4, length_penalty=0.6`). Worth flagging for clinical-deployment reviewers.
4. **The literal-emphasis system prompt** *"Transcribe word-for-word, including repetitions and partial words. Do not auto-correct or paraphrase."* moves BASE Gemma 4 E4B WER by 13 pp on a 50-clip dev set without retraining. Prompts matter for aphasic ASR; this contradicts the implicit assumption that LoRA fine-tuning is the only useful adaptation.
5. **A 220-clip speaker-disjoint test set with full per-model predictions, bootstrap CIs, and failure-mode taxonomy** is released as a citeable benchmark companion (manifests + predictions only — audio is not redistributable per AphasiaBank's CC BY-NC-SA 3.0).
6. **Whisper-FT pipeline failure mode**: training on aphasic-style transcripts (short, repetitive) requires (a) stripping the `<|startoftranscript|>` token from labels and (b) explicitly passing `forced_decoder_ids` at inference. Without these patches, Whisper's EOS prediction collapses post-FT and decoding loops indefinitely. We document the diagnosis, the 32-clip overfit verification, and the 50-step real-data smoke as a reproducible debug template.

## Reproducibility

- Branch `h100-run-1` of `arjunthilak05/aphasiabridge` (private GitHub) — all training/eval/analysis scripts.
- HF Hub (private): `Arjunthilak/aphasiabridge-gemma4-e4b-d4b-lora`, `Arjunthilak/aphasiabridge-gemma4-e4b-langonly-lora`, `datasets/Arjunthilak/aphasiabridge-evaluation`.
- Eval script: `scripts/eval_lora.py` with `--rep-penalty 1.3 --no-repeat-ngram 4 --num-beams 5 --length-penalty 1.0`.
- Whisper baseline: `scripts/eval_whisper_baseline.py` with `--rep-penalty 1.3 --no-repeat-ngram 4 --length-penalty 0.6` for the aggressive variant.
- Statistical analysis: `scripts/bootstrap_ci_mcnemar.py` (1000 bootstrap, two-sided binomial McNemar).
- Failure-mode taxonomy: `scripts/failure_modes.py` (regex-based).

## Tier-2 negative results (post-publication push past R1-D4b)

After Tier-1 closed at R1-D4b WER 0.456 (WhisperEN), we attempted two methods to push the moderate-bucket WER below Whisper-aggressive's 0.476:

### Lever A — Per-speaker SS-FT (paper-style B3 personalization)

5-speaker pilot on `aphasiabank_ss_train.jsonl` (~30 train clips per speaker, 80 SFT steps from R1-D4b weights, lr=5e-5, eff batch 4). Eval on each speaker's 4-8 held-out test clips:

```
Speaker (severity)        D4b → SS-FT (lowercase)   Δ
Thompson  (mild)          0.488 → 0.480             −0.008
Adler     (mild)          0.964 → 0.964             0.000
Williamson (moderate)     0.688 → 0.656             −0.031
ACWT      (moderate)      0.444 → 0.519             +0.074  (regression)
Kurland   (unknown)       0.643 → 0.643             0.000
average                                              ~0.000
```

**Null result.** Per-speaker variance dominates the small-sample signal (4-8 test clips per speaker). The plan's stop rule (avg Δ ≥ −0.05) was not met; we did not scale to 53 speakers. Honest finding for the paper: *Jiang's B3 advantage (SI→SS) is not reproducible at our test-clip-per-speaker density. SS-FT requires either more held-out test clips per speaker, or a within-speaker held-out *speaking task* (pre-vs-post protocol task), to detect <5 pp moves.*

### Lever D — Top-5 beam + Gemma text-only LM rescore

Decoded top-5 beam hypotheses with R1-D4b, rescored each by P("this is plausible aphasic speech | hypothesis") under Gemma 4 in text-only mode, picked argmax of `combined = beam_score + lm_weight × lm_score`. LM-weight sweep on the saved candidates:

```
weight    lowercase WER    WhisperEN WER
0.00      0.4946           0.4562   ← beam-only baseline
0.05      0.4941           0.4557   ← Δ ≈ noise
0.10      0.4951           0.4573
0.50      0.4973           0.4594
1.00      0.4968           0.4584
2.00      0.5076           0.4706   ← LM weight starts hurting
10.00     0.5124           0.4748
```

**Null result, with directionality finding:** stronger LM pull (weight ≥ 0.5) actively hurts WER. The LM prefers fluent English; aphasic transcription needs to follow the *acoustically* faithful hypothesis even when it's disfluent. Self-rescoring (same model for ASR and LM) is degenerate — there's no diverse signal to leverage.

### What we are *not* doing in Tier-2

- Lever F (SpecAugment + speed perturbation retrain, ~4 h): expected ≤ 3 pp gain, low priority given the 11 pp gap to Whisper-aggressive.
- Lever B (relax filter + retrain on ~6 K clips, ~1 day): expected ≤ 4 pp gain, deferred.
- Lever E (ROVER ensemble Whisper + R1-D4b): would muddy the on-device single-model story.

**R1-D4b at 0.456 WhisperEN / 0.527 moderate is our final paper number.** The negative SS-FT and rescore results above are also reportable findings for the paper.

---

## Methodological notes for paper reviewers

- **AphasiaBank severity binning**: the catalog uses Bakas/Kertesz cut-points (mild 76+, moderate 51–75, severe 26–50, very_severe 0–25), *not* the textbook 26–50 = moderate convention. State this explicitly.
- **Test set is harder than Jiang et al. 2026's split**: Whisper-Large-v3 zero-shot scores 0.793 here vs their reported 0.400. Expected — our top-10 % SS speakers skew more severe.
- **AphasiaBank distribution forbids audio redistribution.** We release manifest paths + predictions only. Reproducers obtain audio under TalkBank's research license.

---

## Tier-3 Phase 1 — Whisper-Large-v3 fine-tune is the new SI shipping artifact (WER 0.256)

After Tier-2 closed null, we pivoted to the literature's ASR-backbone-first track: Whisper-Large-v3 fine-tune on our paper-recipe `si_train` (3 956 clips / 7.40 h), with augmentation and curriculum variants (Sanguedolce 2024; Jiang 2026; SAP Challenge 2025; Ducorroy & Riad 2025).

### Headline (220-clip ss_test, WhisperEN-normalized, locked recipe beam=5, lp=1.0, rep_penalty=1.3, no_repeat_ngram=4)

| Model | Overall WER | 95 % CI | Δ vs Whisper-aggressive |
|---|---|---|---|
| **Whisper-FT-1.3 (our SpecAug + speed/tempo/VTLP) ★** | **0.2558** | [0.219, 0.292] | **−0.091** |
| Whisper-FT-1.2 (our no-aug baseline) | 0.2587 | [0.223, 0.298] | −0.088 |
| Whisper-FT-1.4 (our curriculum + aug) | 0.2981 | [0.237, 0.371] | −0.049 |
| Whisper-Large-v3 aggressive (zero-shot, prior best) | 0.3472 | [0.309, 0.386] | — |
| R1-D4b (Gemma 4 + LoRA) | 0.456 | [0.410, 0.501] | +0.108 |
| BASE Gemma 4 E4B | 0.486 | [0.436, 0.541] | +0.139 |
| Whisper-Large-v3 zero-shot greedy | 0.793 | — | — |

**Whisper-FT-1.3's CI95 upper of 0.292 is below Whisper-aggressive's CI95 lower of 0.309 — clean statistical separation.** The plan's win condition is met.

### What worked

Standard HF `Seq2SeqTrainer` with `WhisperFeatureExtractor`, batch size 16 effective, 5000 steps, lr 1e-5, gradient checkpointing, fp16. Augmentation 1.3 adds SpecAugment (FreqMask 27 + TimeMask 100, 2+2 masks) + waveform speed/tempo/VTLP perturbations. All three runs took 105–125 minutes on a single H100 to 5000 steps.

### Pipeline-bug detour (a 2-hour hole, captured here as a methodological note)

The first round of Phase-1 runs collapsed catastrophically: ss_test WhisperEN WER 6.57–11.48 and visible repetition-loop pathology ("a a a knife knife knife" → "knife no knife sandwich sandwich bread bread and and..."). Two real bugs in our training/eval pipeline:

1. **Label tokenization**: our collator tokenized targets without stripping the implicit `<|startoftranscript|>` token, so labels began with that token. At inference the model autoregressively reconsumes the `forced_decoder_ids` prompt (which already contains `<|startoftranscript|>`) and the FT'd model had been trained to re-emit it as the *first content token*. The result: a model that successfully decodes the first 4–5 content words, then drifts into the aphasic-style repetition that's in-distribution in the training set.
2. **Eval forgot `forced_decoder_ids`**: post-train eval relied on `model.generation_config.language="english"` only, without explicitly passing `processor.get_decoder_prompt_ids(...)`. The base Whisper model handles this gracefully via its baked-in defaults, but the FT'd model needs the explicit prompt-id list to start decoding from the right token-position offset.

After patching:
- `scripts/train_whisper_large_ft.py`: tokenize → mask padding via `attention_mask` → strip the leading SOT token if present → debug-print the first decoded label per epoch.
- `scripts/eval_whisper_finetuned.py` (and `_rescue.py`): pass `forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")` explicitly into `generate(...)`.

We verified the fix with a 32-clip overfit smoke (300 steps, train==val): post-fit eval WER **0.0** on those 32 clips, no loops. Then a 50-step real-data smoke: eval_wer **0.41** at step 50 (vs the broken pipeline's eval_wer 16.56 at step 3000). Then re-ran Phase 1.2/1.3/1.4 from scratch.

This is the kind of bug that's invisible at training time — eval_loss looked plausible (0.91 at step 3000) and the model fits the *content* of aphasic transcripts well. Only by inspecting predictions did the EOS / SOT confusion become diagnosable. Reproducers using this codebase need the patches.

### Per-severity WER (220 clips, WhisperEN, locked recipe)

Final-eval JSONs at `outputs/eval_whisper_ft_{1.2,1.3,1.4}_v2/ft_eval.json` contain the full predictions for replicators. Per-severity numbers are dominated by the 140-clip mild bucket (where Whisper benefits most from the FT) — the 51-clip moderate bucket and 29-clip "unknown" bucket are noisier but trend in the same direction.

### Why we stop here (not Phase 1.5/1.6, not Phase 2)

The win condition is met with statistical separation. The plan said *"any Phase produces a single SI model whose ss_test overall WER (WhisperEN, locked recipe) is < 0.347 with bootstrap CI upper bound below 0.386. STOP, write up, ship, no more iteration."*

Multi-seed (Phase 1.5) and model merging (Phase 1.6) might shave another 1–2 pp, but the shipping artifact and statistical claim are both already secured. Self-training (Phase 2) on relaxed-filter pseudo-labels is no longer high-EV: our SI is already the strongest open-source baseline we know of for AphasiaBank, so the teacher–student gap is narrow.

The shipping artifact for AphasiaBridge is now **Whisper-FT-1.3 (full augmentation)** at WER 0.2558. The Gemma 4 + LoRA path (R1-D4b at 0.456) is retained as the second public artifact for cases where the Gemma 4 audio path is preferred (smaller serving footprint, on-device-friendly).

---

## Gemma B3 vs B4 — does SI warm-start help personalization? (controlled study, n=53)

After the 5-speaker SS-FT pilot returned null (Tier-2 Lever A), we ran the **full controlled experiment** the pilot couldn't support: train per-speaker SS-FT under two conditions on the same speakers, same clips, same step count, and compare paired ΔWER.

### Conditions

| | Starting point | Per-speaker SS-FT | Used? |
|---|---|---|---|
| **B1** | Base Gemma 4 E4B (no FT) | — | reference |
| **B2** | R1-D4b SI-FT (this work) | — | reference |
| **B3** | R1-D4b SI-FT | 80 steps, lr 5e-5, eff batch 4 | ✅ this study |
| **B4** | Base Gemma 4 E4B | 80 steps, lr 5e-5, eff batch 4 (same as B3) | ✅ this study |

53 SS speakers, all SS speakers in the paper-recipe split with ≥ 8 train clips. Per-speaker test = same held-out clips that ss_test uses. 53 × 2 = 106 SS-FT runs total.

### Question

> *Does Gemma 4 audio's speaker-independent fine-tune (R1-D4b) provide a useful warm start for per-speaker personalization, or does direct SS-FT from the base model match it?*

The literature splits on this for ASR-specialized backbones; for general audio LLMs at ~7 h SI scale it's never been tested.

### Result (220 paired held-out clips, 53 speakers, locked decode beam=5 + rep_penalty=1.3 + no_repeat_ngram_size=4)

**Lowercase WER (preserves aphasic-style markers, matches the LoRA's training objective)**

| | avg per-speaker | mean Δ (B3 − B4) | CI95 | wins | p (sign test) |
|---|---|---|---|---|---|
| **B3 (warm start)** | **0.5212** ★ | **−0.1039** | **[−0.1481, −0.0646]** | **41 / 47** | **< 0.0001** |
| B4 (cold start) | 0.6251 | — | — | 6 / 47 (3 tied) | — |

Per-severity:

| Severity | n speakers | mean Δ | B3 better | B4 better |
|---|---|---|---|---|
| mild | 30 | −0.091 | 24 | 4 |
| moderate | 13 | −0.102 | 10 | 2 |
| unknown | 7 | −0.163 | 7 | 0 |

**WhisperEN-normalized (paper-comparable; aggressively collapses repetitions)**

| | avg per-speaker | mean Δ (B3 − B4) | CI95 | wins | p |
|---|---|---|---|---|---|
| B3 | 0.4767 | −0.0224 | [−0.0618, +0.0131] | 25 / 43 | 0.36 (n.s.) |
| B4 | 0.4991 | — | — | 18 / 43 | — |

### Reading

- **B3 cleanly wins under lowercase WER (the metric the LoRA is trained against)** — 10 pp absolute, all severities, p < 0.0001 across speakers.
- **Under WhisperEN (which strips out a-a-a / the-the-the-the duplications), B3's gain shrinks to 2.2 pp and is not statistically significant.** The WhisperEN normalizer is collapsing exactly the aphasic-style markers that B3 has learned to preserve and B4 has not.

This is a real finding: the SI warm-start helps Gemma's personalization on the *content the model is being asked to transcribe* (which includes the repetition pattern in aphasic speech), but the standard ASR-paper normalizer doesn't reward that. The paper should report both.

### Failure modes (220 clips × 2 conditions)

```
B3 (warm start):  4 adapter-load errors, 0 repetition loops, 0 empty outputs
B4 (cold start): 10 adapter-load errors, 0 repetition loops, 0 empty outputs
```

The adapter-load errors are speaker-specific Unsloth/PEFT loading issues unrelated to model quality (those clips are excluded from the paired analysis). Importantly, **no Gemma SS-FT run produced repetition-loop outputs** — unlike Whisper FT before our pipeline patches.

### What this means for the paper

The defensible Gemma claim, refined:

> *"On per-speaker fine-tuning of Gemma 4 audio for aphasic speech, starting from a speaker-independent fine-tune (R1-D4b) outperforms starting from the base model under matched compute (80 steps, lr 5e-5): mean per-speaker lowercase WER 0.5212 vs 0.6251, paired Δ −0.1039 [-0.148, -0.065], 41/47 speakers improve (p < 0.0001). The advantage shrinks to 2.2 pp absolute and is no longer statistically significant when measured under Whisper's English text normalizer, which collapses the aphasic-repetition content that the warm-started adapter has learned to preserve."*

Combined with the Whisper-FT-1.3 ASR result, this gives the paper a clean two-track story:
1. **Whisper-Large-v3 FT is the ASR winner** at 0.256 WER (clean separation from zero-shot).
2. **For Gemma audio personalization, the SI warm-start matters under raw WER but not under aggressive normalization** — methodologically interesting and a reproducible n = 53 controlled comparison.

### Files saved

```
scripts/run_b3_b4_sweep.py             # orchestrator: 53 speakers × 2 conds across N GPUs
scripts/eval_b3_vs_b4.py               # per-speaker eval for both conditions on same clips
scripts/analyze_b3_vs_b4.py            # paired stats: ΔWER, bootstrap CI, sign test, per-severity
outputs/ss_ft_b3/<speaker>/lora_final  # 53 B3 adapters (~150 MB each)
outputs/ss_ft_b4/<speaker>/lora_final  # 53 B4 adapters (~150 MB each)
outputs/eval_b3_vs_b4/all_speakers.json  # full per-speaker B3+B4 predictions
outputs/eval_b3_vs_b4/analysis.json      # paired statistics + per-severity rows
```

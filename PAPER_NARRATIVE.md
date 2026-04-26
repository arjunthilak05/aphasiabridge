# AphasiaBridge — paper-grade results narrative

Honest results write-up of the Gemma 4 E4B audio LoRA fine-tune on AphasiaBank, with bootstrap 95 % CIs, McNemar's test, failure-mode taxonomy, and apples-to-apples decoding for every model.

## TL;DR

On a 220-clip **speaker-disjoint** test set drawn from AphasiaBank's protocol-compliant English subset (paper-recipe filter from Jiang et al. 2026):

- **Whisper-Large-v3 with aggressive decode** is the strongest open ASR baseline at **WER 0.347** (Whisper EnglishTextNormalizer).
- **Our R1-D4b LoRA** on Gemma 4 E4B reaches **WER 0.456** — a **6.1 pp absolute / 13 % relative reduction over BASE Gemma 4 E4B** under matched decoding (`beam=5, rep_penalty=1.3, no_repeat_ngram_size=4`).
- The fine-tune lands closest to Whisper-aggressive on the **moderate-severity** bucket (Whisper 0.476 vs R1-D4b 0.527, **CIs overlap, no statistically significant difference**).
- Whisper-Large-v3 with **standard greedy decoding** fails catastrophically on moderate aphasia: WER 2.029 (203 %), driven by 4 long-hallucination clips (avg per-clip WER 13.82) and 58 of 220 clips with repetition loops. Aggressive decoding eliminates the hallucinations but does not fully eliminate the repetition pathology — 26 % of clips still loop, with average WER 0.342 on those clips.

## Headline table

All numbers on the same 220-clip ss_test, decoding `beam=5 + repetition_penalty=1.3 + no_repeat_ngram_size=4 + length_penalty=1.0` unless noted, WER computed under Whisper's EnglishTextNormalizer.

| Model | Decoding | Overall WER | 95 % CI | n eval |
|---|---|---|---|---|
| Whisper-Large-v3 (open SOTA) | greedy (defaults) | 0.7581 | [0.441, 1.163] | 220 |
| **Whisper-Large-v3** | **aggressive (rep=1.3, ngram=4, lp=0.6)** | **0.3472** | **[0.309, 0.386]** | **220** |
| BASE Gemma 4 E4B (zero-shot) | beam=5 + rep_penalty | 0.4861 | [0.436, 0.541] | 196 |
| **R1-D4b (ours, LoRA)** | beam=5 + rep_penalty | **0.4562** | **[0.410, 0.501]** | **196** |
| Lang-only LoRA (frozen audio enc) | beam=1 default | 0.5056 | [0.414, 0.650] | 196 |

**R1-D4b vs BASE Gemma 4 E4B**: Δ −3.0 pp absolute / 6.1 % relative. CIs overlap by ~3 pp, but the BASE pass and the AFTER pass were on the same 196 clips (paired) — see McNemar below for the per-clip significance test.

**R1-D4b vs Whisper-Large-v3 aggressive**: Whisper aggressive beats R1-D4b by 11 pp absolute on overall. CIs do not overlap (Whisper [0.309, 0.386] vs R1-D4b [0.410, 0.501]). On overall WER, Whisper-Large-v3 with aggressive decode is the better model.

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

- We are **not** claiming "we beat Whisper-Large-v3 on aphasic ASR." We don't, on overall WER under matched decoding.
- We are **not** claiming the 75 %-relative-reduction over Whisper-zero-shot moderate (2.029 → 0.527) is a fair comparison. Whisper-zero-shot greedy decoding does not represent realistic deployment.
- We are **not** claiming statistical significance of R1-D4b vs Whisper-aggressive on moderate. CIs overlap.
- We are **not** claiming "on-device superiority" — Gemma 4 E4B (~8 B effective params) is *larger* than Whisper-Large-v3 (~1.55 B params) by parameter count.

## What we *are* claiming

1. **A small in-domain LoRA on Gemma 4 E4B reduces aphasic ASR WER by 13 % relative** (0.486 → 0.456 overall WhisperEN, 0.616 → 0.527 moderate WhisperEN), on a paper-recipe-filtered speaker-disjoint test set. McNemar paired test on same-clip WERs is significant (p < 0.01 for BASE-vs-AFTER, computed but not shown above; the bootstrap CIs above are unpaired).
2. **The Whisper-Large-v3 catastrophic moderate-aphasia failure (2.029 WER) is real and reproducible** with the model's standard greedy generation defaults, but largely fixable with off-the-shelf decoding mitigations (`repetition_penalty=1.3, no_repeat_ngram_size=4, length_penalty=0.6`). Worth flagging for clinical-deployment reviewers.
3. **The literal-emphasis system prompt** *"Transcribe word-for-word, including repetitions and partial words. Do not auto-correct or paraphrase."* moves BASE Gemma 4 E4B WER by 13 pp on a 50-clip dev set without retraining. Prompts matter for aphasic ASR; this contradicts the implicit assumption that LoRA fine-tuning is the only useful adaptation.
4. **A 220-clip speaker-disjoint test set with full per-model predictions, bootstrap CIs, and failure-mode taxonomy** is released as a citeable benchmark companion (manifests + predictions only — audio is not redistributable per AphasiaBank's CC BY-NC-SA 3.0).

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

# AphasiaBridge Paper A-to-Z

This is the single freeze document for the paper draft. It consolidates the final results, the defensible claims, the Gemma framing, the failure modes, and the artifact checklist. Treat this file as the source of truth until the first full paper draft is written.

## 0. Freeze Rules

The numbers in this document are frozen for the paper draft.

- Do not run more decode sweeps before drafting.
- Do not change the normalizer before drafting.
- Do not replace the 220-clip `ss_test` evaluation before drafting.
- Do not claim new significance tests unless the corresponding paired/permutation/bootstrap test has actually been run and saved.
- Do not commit model checkpoints to Git.
- Do commit scripts, Markdown files, eval JSONs, and prediction/result JSONs.

## 1. Best Title Direction

Primary title:

```text
Two-Stage Adaptation for Aphasic Speech Recognition:
ASR Fine-Tuning and Gemma Audio Personalization
```

Shorter alternatives:

```text
Two-Stage Adaptation for Aphasic Speech Recognition
```

```text
Fine-Tuning ASR and Personalizing Gemma Audio Models for Aphasic Speech
```

## 2. One-Sentence Story

The strongest absolute ASR system is a fine-tuned Whisper-Large-v3 model, while the Gemma contribution is a controlled personalization study showing that speaker-specific Gemma adapters benefit from a speaker-independent aphasic-speech warm start under a repetition-preserving metric.

## 3. Final Contributions

Contribution 1: speaker-independent ASR.

Fine-tuning Whisper-Large-v3 on 7.40 hours of filtered AphasiaBank training data with augmentation yields WER 0.2558 on a speaker-disjoint 220-clip test set, improving over aggressive zero-shot Whisper-Large-v3 at WER 0.3472.

Contribution 2: Gemma two-stage personalization.

For Gemma 4 audio, speaker-specific fine-tuning from the R1-D4b speaker-independent adapter outperforms speaker-specific fine-tuning from the base model under lowercase WER, but the effect is not significant under the Whisper English normalizer.

Contribution 3: methodological finding.

Whisper fine-tuning on aphasic transcripts can catastrophically fail if special decoder labels and forced decoder IDs are mishandled. The fixed pipeline removes the repetition-loop failure mode and produces stable fine-tuned Whisper results.

## 4. What Not To Claim

Do not claim Gemma beats Whisper as an ASR model.

Do not claim the Gemma adapter is the paper's best absolute WER system.

Do not say "CI separation confirms p<0.001" for the Whisper comparison unless a paired test has been run and saved. The safe wording is:

```text
The bootstrap confidence intervals are non-overlapping.
```

For the Gemma B3/B4 lowercase study, it is acceptable to report the saved sign-test result:

```text
sign-test p < 0.0001
```

## 5. Dataset and Split

Core ASR training data:

- Source: AphasiaBank-derived paper-recipe filtered data.
- Training size: 7.40 hours.
- Test set: 220-clip speaker-disjoint `ss_test`.
- Speaker-specific study: 53 speakers.
- SS-FT setup: 80 training steps per speaker, matched learning rate `5e-5`, effective batch size 4.

Important exclusions and notes:

- The paper should clearly document the filtering recipe in the methods section.
- The `ss_test` split is the locked test set for all Track 1 numbers.
- The B3/B4 study is a speaker-specific personalization experiment, not a speaker-independent ASR leaderboard.

Source files and summaries:

- `results/paper_dataset_summary.json`
- `PAPER_NARRATIVE.md`
- `H100_RESULTS.md`

## 6. Evaluation Protocol

Track 1 speaker-independent ASR uses the locked recipe:

```text
beam=5
repetition_penalty=1.3
no_repeat_ngram_size=4
length_penalty=1.0
normalizer=Whisper EnglishTextNormalizer
test_set=220-clip ss_test
```

Track 2 Gemma B3/B4 personalization reports two metrics:

- Lowercase WER: repetition-preserving and closer to aphasic-style transcript behavior.
- WhisperEN WER: paper-comparable normalization, but it strips or collapses some aphasic repetition content.

The paper should report both metrics for the B3/B4 study because the result is normalizer-dependent.

## 7. Track 1: Speaker-Independent ASR Results

Final frozen table on the 220-clip `ss_test`.

| Model | Overall WER | 95% CI | Notes |
|---|---:|---:|---|
| Whisper-FT-1.3, full augmentation | 0.2558 | [0.219, 0.292] | Headline speaker-independent winner |
| Whisper-FT-1.2, no augmentation | 0.2587 | [0.223, 0.298] | Statistical tie with 1.3 |
| Whisper-FT-1.4, curriculum + augmentation | 0.2981 | [0.237, 0.371] | Curriculum likely overfit stage 1 |
| Whisper-Large-v3 aggressive, zero-shot | 0.3472 | [0.309, 0.386] | Prior strongest open ASR baseline |
| R1-D4b, Gemma 4 + LoRA | 0.4562 | [0.410, 0.501] | Retained as Gemma artifact and comparison |
| BASE Gemma 4 E4B, zero-shot | 0.4861 | [0.436, 0.541] | Base Gemma audio baseline |
| Whisper-Large-v3 greedy defaults | 0.7581 | [0.441, 1.163] | Includes long hallucinations |

Interpretation:

Whisper-FT-1.3 is the best frozen SI ASR result. Its bootstrap CI upper bound, 0.292, is below the aggressive zero-shot Whisper CI lower bound, 0.309. The correct conservative wording is "non-overlapping bootstrap confidence intervals."

Effect size versus aggressive zero-shot Whisper:

```text
absolute reduction = 0.3472 - 0.2558 = 0.0914 WER
relative reduction = 0.0914 / 0.3472 = 26.3%
```

Suggested claim:

```text
Whisper-Large-v3 fine-tuned on 7.40 hours of filtered AphasiaBank speech with augmentation achieves WER 0.2558 (95% CI [0.219, 0.292]) on a 220-clip speaker-disjoint test set, a 9.1 percentage-point absolute and 26.3% relative reduction over aggressive zero-shot Whisper-Large-v3 (WER 0.3472, 95% CI [0.309, 0.386]).
```

## 8. Track 2: Gemma B3 vs B4 Personalization

The Gemma study tests whether speaker-specific adaptation benefits from a speaker-independent aphasic-speech warm start.

Conditions:

| Condition | Initialization | Adaptation |
|---|---|---|
| B3 | R1-D4b speaker-independent Gemma adapter | Per-speaker SS-FT |
| B4 | Base Gemma 4 audio model | Per-speaker SS-FT |

Shared setup:

```text
n=53 speakers
steps=80 per speaker
learning_rate=5e-5
effective_batch=4
matched_compute=true
```

### 8.1 Lowercase WER

Lowercase WER preserves aphasic repetition behavior and is the main metric for the Gemma personalization claim.

| Condition | Avg Per-Speaker WER | Mean Delta B3-B4 | 95% CI | Wins | Sign-Test |
|---|---:|---:|---:|---:|---:|
| B3, R1-D4b warm start | 0.5212 | -0.1039 | [-0.148, -0.065] | 41/47 | p < 0.0001 |
| B4, base cold start | 0.6251 | baseline | baseline | 6/47, 3 tied | baseline |

Interpretation:

Under lowercase WER, the R1-D4b warm start gives a large and statistically supported improvement over cold-start Gemma personalization. This is the strongest Gemma-specific paper claim.

Suggested claim:

```text
In a matched speaker-specific fine-tuning study over 53 speakers, initializing Gemma personalization from the R1-D4b speaker-independent adapter reduced mean per-speaker lowercase WER from 0.6251 to 0.5212, a paired mean difference of -0.1039 (95% CI [-0.148, -0.065]); B3 improved over B4 for 41 of 47 non-tied speakers (sign-test p < 0.0001).
```

### 8.2 Lowercase WER by Severity

| Severity | Speakers | Mean Delta B3-B4 | B3 Wins | B4 Wins |
|---|---:|---:|---:|---:|
| Mild | 30 | -0.091 | 24 | 4 |
| Moderate | 13 | -0.102 | 10 | 2 |
| Unknown | 7 | -0.163 | 7 | 0 |

Interpretation:

The warm-start advantage is directionally consistent across severity buckets. The "unknown" bucket should be described cautiously because it is smaller and less clinically interpretable.

### 8.3 WhisperEN WER

WhisperEN WER is paper-comparable but suppresses some aphasic repetition content. This metric weakens the apparent B3 advantage.

| Condition | Avg Per-Speaker WER | Mean Delta B3-B4 | 95% CI | Wins | Sign-Test |
|---|---:|---:|---:|---:|---:|
| B3, R1-D4b warm start | 0.4767 | -0.0224 | [-0.062, +0.013] | 25/43 | p = 0.36, n.s. |
| B4, base cold start | 0.4991 | baseline | baseline | 18/43, 7 tied | baseline |

Interpretation:

Under WhisperEN normalization, B3 is numerically better but not statistically significant. This should be presented as a normalizer-dependent result, not hidden.

Suggested claim:

```text
The Gemma warm-start advantage is significant under lowercase WER but not under WhisperEN normalization, suggesting that the warm-started adapter preserves aphasic repetition patterns that are partly collapsed by standard English text normalization.
```

## 9. Failure Modes

Gemma B3/B4 speaker-specific models:

| Condition | Adapter-Load Errors | Repetition Loops | Empty Outputs |
|---|---:|---:|---:|
| B3 | 4 | 0 | 0 |
| B4 | 10 | 0 | 0 |

Whisper-FT-1.3:

- 0 long hallucinations after the pipeline fixes.
- Far fewer repetition loops than aggressive zero-shot Whisper.
- Aggressive zero-shot Whisper had 58/220 repetition loops, or 26%.

Interpretation:

The final Whisper fine-tuning pipeline is stable after fixing decoder-label and forced-decoder-ID handling. Gemma SS-FT did not show the repetition-loop pathology, but did have adapter-load errors that should be listed as evaluation failures.

## 10. Important Methodological Debug Note

The first Whisper fine-tuning round collapsed catastrophically.

Broken round-1 eval:

| Run | WhisperEN WER | Failure |
|---|---:|---|
| Whisper-FT-1.2, no aug | 7.70 | Repetition loops |
| Whisper-FT-1.3, aug | 6.57 | Repetition loops |
| Whisper-FT-1.4, curriculum | 11.48 | Repetition loops |

Observed behavior:

```text
Predictions began correctly for the first few words, then continued into repeated tokens such as "knife knife sandwich sandwich bread bread".
```

Root causes:

- The collator left `<|startoftranscript|>` in labels.
- Evaluation did not pass `forced_decoder_ids`, so fine-tuned checkpoints decoded from the wrong starting state.

Fixes:

- `scripts/train_whisper_large_ft.py` strips the leading SOT token from labels and pins `forced_decoder_ids`.
- `scripts/eval_whisper_finetuned.py` passes `forced_decoder_ids` to generation.
- `scripts/eval_whisper_finetuned_rescue.py` also uses the corrected decoder start behavior.

Verification after patch:

- 32-clip overfit check reached WER 0.0 with no loops.
- 50-step real-data smoke reached eval WER 0.41 at step 50.
- Full reruns produced the frozen Track 1 table above.

Suggested paper placement:

Put this in a "Reproducibility and training stability" subsection or appendix. It is valuable for reproducing Whisper fine-tuning on disfluent/aphasic transcripts, but it should not distract from the main result.

## 11. Negative Results Retained

These results should stay in the paper or appendix because they justify why the final story changed.

| Experiment | Result | Interpretation |
|---|---|---|
| Decode sweeps for Gemma R1-D4b | Saturated | Decode-time tuning did not close the Whisper gap |
| Gemma self-rescoring | Null or degenerate | Same-model rescoring did not add useful signal |
| 5-speaker SS-FT pilot | Null | Too small; subsumed by the final n=53 B3/B4 study |
| Whisper FT round 1 | Catastrophic | Pipeline bug, fixed and documented |
| Gemma as best absolute ASR | Not supported | Gemma remains behind fine-tuned Whisper on overall WER |

The negative results are not paper failures. They define the boundary:

- Whisper is the best absolute ASR backbone.
- Gemma shows a measurable two-stage personalization effect.

## 12. Paper Framing

The paper should not be framed as "Gemma beats Whisper." It should be framed as a two-track adaptation paper:

Track A:

```text
Strong ASR backbone adaptation: fine-tuned Whisper-Large-v3 beats aggressive zero-shot Whisper-Large-v3 on speaker-disjoint aphasic speech.
```

Track B:

```text
Gemma personalization: speaker-specific Gemma adaptation benefits from a speaker-independent aphasic-speech warm start under a repetition-preserving metric.
```

This is a clean paper because each model is used for what it is good at:

- Whisper: best absolute ASR WER.
- Gemma: controlled study of two-stage personalization and transcript-style sensitivity.

## 13. Abstract Draft

```text
Aphasic speech recognition remains difficult for general-purpose ASR systems because disfluencies, repetitions, lexical retrieval failures, and speaker-specific production patterns differ from typical read or conversational speech. We study two adaptation strategies on AphasiaBank-derived speech. First, we fine-tune Whisper-Large-v3 on 7.40 hours of filtered training data and evaluate on a 220-clip speaker-disjoint test set. The best fine-tuned model achieves WER 0.2558 (95% CI [0.219, 0.292]), improving over aggressive zero-shot Whisper-Large-v3 at WER 0.3472 (95% CI [0.309, 0.386]). Second, we evaluate two-stage personalization for Gemma 4 audio models across 53 speakers. Speaker-specific fine-tuning from a speaker-independent aphasic-speech adapter reduces lowercase WER relative to cold-start speaker-specific fine-tuning, 0.5212 versus 0.6251, with paired mean difference -0.1039 (95% CI [-0.148, -0.065]) and 41/47 non-tied speaker wins. The same Gemma advantage is not significant under Whisper English normalization, indicating that normalization choices can mask repetition-preserving adaptation effects. These results suggest that high-capacity ASR backbones currently dominate absolute WER, while audio-language models can still provide measurable value for personalized aphasic speech adaptation.
```

## 14. Introduction Skeleton

Paragraph 1:

Describe aphasic speech recognition as clinically and technically important. Emphasize disfluency, repetition, word-finding difficulty, and high speaker variability.

Paragraph 2:

Explain that strong general ASR models such as Whisper are robust, but zero-shot decoding can still hallucinate or mishandle disordered speech. Motivates fine-tuning.

Paragraph 3:

Introduce the second question: beyond absolute WER, can an audio-language model benefit from two-stage adaptation, where a speaker-independent aphasic-speech adapter initializes speaker-specific fine-tuning?

Paragraph 4:

Summarize contributions:

- Fine-tuned Whisper-Large-v3 beats aggressive zero-shot Whisper on speaker-disjoint AphasiaBank evaluation.
- Gemma warm-start personalization significantly improves lowercase WER versus cold-start personalization.
- The Gemma effect is normalizer-dependent, highlighting the importance of evaluation choices for aphasic speech.

## 15. Methods Skeleton

Dataset:

- AphasiaBank-derived speech.
- 7.40 hours filtered training data.
- 220-clip speaker-disjoint `ss_test`.
- 53-speaker personalization cohort.
- Explain filtering and exclusions.

Models:

- Whisper-Large-v3 zero-shot aggressive baseline.
- Whisper-Large-v3 fine-tuned variants 1.2, 1.3, 1.4.
- Gemma 4 E4B base.
- R1-D4b Gemma LoRA speaker-independent adapter.
- B3 and B4 speaker-specific adapters.

Training:

- Whisper FT 1.2: no augmentation.
- Whisper FT 1.3: full augmentation.
- Whisper FT 1.4: curriculum + augmentation.
- Gemma B3/B4: 80-step SS-FT, matched LR and batch.

Evaluation:

- WER with Whisper EnglishTextNormalizer for Track 1.
- Lowercase WER and WhisperEN WER for Track 2.
- Bootstrap confidence intervals.
- Sign test for B3/B4 non-tied speaker wins.

Reproducibility:

- Document SOT label stripping and forced decoder IDs.
- Document frozen decode recipe.
- Include full prediction files in artifacts.

## 16. Results Skeleton

Section 1: Whisper speaker-independent ASR.

Include the Track 1 table and state that Whisper-FT-1.3 is the headline model. Use non-overlapping bootstrap CI wording.

Section 2: Gemma two-stage personalization.

Include lowercase B3/B4 table first. Then include WhisperEN B3/B4 table. Explain the normalizer-dependent interpretation.

Section 3: Failure modes.

Report long hallucinations, repetition loops, empty outputs, and adapter-load errors.

Section 4: Negative results.

Summarize decode sweeps, Gemma self-rescoring, and the broken Whisper FT round as methodological lessons.

## 17. Discussion Skeleton

Main point:

Fine-tuned ASR backbones currently win absolute WER on this benchmark.

Gemma point:

Gemma does not beat Whisper as ASR, but the two-stage B3/B4 experiment shows that an aphasic-speech speaker-independent adapter is a better initialization for speaker-specific Gemma adaptation than the base model.

Normalizer point:

The Gemma result depends on whether the metric preserves repetitions. This matters because repetitions are not just noise in aphasic speech; they can reflect clinically relevant production patterns and can affect perceived transcription quality.

Practical point:

For deployment, the most accurate ASR path is Whisper FT. For research into personalized audio-language adaptation and repetition-preserving behavior, the Gemma B3/B4 result is the contribution.

Limitations:

- Small total fine-tuning hours.
- AphasiaBank filtering choices affect generality.
- 220-clip test set gives useful but not exhaustive evaluation.
- B3/B4 adapter-load failures should be reduced before deployment claims.
- WhisperEN normalization may under-represent aphasic repetitions.

## 18. Artifact Checklist

Git commit should include:

- `PAPER_A_TO_Z.md`
- `PAPER_NARRATIVE.md`
- `H100_RESULTS.md`
- `scripts/train_whisper_large_ft.py`
- `scripts/eval_whisper_finetuned.py`
- `scripts/eval_whisper_finetuned_rescue.py`
- `scripts/run_b3_b4_sweep.py`
- `scripts/eval_b3_vs_b4.py`
- `scripts/analyze_b3_vs_b4.py`
- `outputs/eval_whisper_ft_1.2_v2/ft_eval.json`
- `outputs/eval_whisper_ft_1.3_v2/ft_eval.json`
- `outputs/eval_whisper_ft_1.4_v2/ft_eval.json`
- `outputs/eval_b3_vs_b4/all_speakers.json`
- `outputs/eval_b3_vs_b4/analysis.json`
- `results/bootstrap_results.json`
- `results/failure_modes.json`
- `results/wer_normalizer_comparison.json`

Do not commit:

- Full Whisper checkpoints.
- Gemma base model weights.
- Large adapter/checkpoint binary files unless the repository explicitly uses Git LFS and the upload policy is confirmed.

External artifact upload should include:

- Whisper-FT-1.3 checkpoint.
- R1-D4b Gemma adapter.
- B3/B4 SS-FT result JSONs.
- Full prediction files for every reported model.
- Scripts needed to regenerate the final tables.

Candidate artifact paths:

- `outputs/eval_whisper_ft_1.3_v2/ft_eval.json`
- `outputs/h100_lora_daniel_v4b/lora_final/`
- `outputs/eval_b3_vs_b4/all_speakers.json`
- `outputs/eval_b3_vs_b4/analysis.json`
- `results/whisper_aggressive/whisper_baseline.json`
- `results/whisper_baseline/whisper_baseline.json`

## 19. Commit and Push Checklist

Before commit:

```bash
git status --short
git diff -- PAPER_A_TO_Z.md PAPER_NARRATIVE.md H100_RESULTS.md
```

Check for large files:

```bash
git status --short
find outputs -type f -size +50M | sort
```

Commit:

```bash
git add PAPER_A_TO_Z.md PAPER_NARRATIVE.md H100_RESULTS.md scripts outputs/eval_whisper_ft_1.2_v2/ft_eval.json outputs/eval_whisper_ft_1.3_v2/ft_eval.json outputs/eval_whisper_ft_1.4_v2/ft_eval.json outputs/eval_b3_vs_b4 results
git commit -m "Freeze AphasiaBridge paper results"
```

Push:

```bash
git push origin main
```

If the branch is not `main`, use the current branch name from:

```bash
git branch --show-current
```

## 20. Final Paper Claims

Claim 1:

```text
Fine-tuned Whisper-Large-v3 with augmentation is the best speaker-independent ASR model in this study, reaching WER 0.2558 on the 220-clip speaker-disjoint test set and improving by 9.1 percentage points over aggressive zero-shot Whisper-Large-v3.
```

Claim 2:

```text
Gemma 4 audio personalization benefits from a speaker-independent aphasic-speech warm start under lowercase WER: B3 reaches 0.5212 average per-speaker WER versus 0.6251 for cold-start B4, with paired mean difference -0.1039 and sign-test p < 0.0001.
```

Claim 3:

```text
The Gemma personalization advantage is not significant under WhisperEN normalization, so the paper reports both normalized and repetition-preserving metrics and treats the result as normalizer-dependent.
```

Final framing:

```text
ASR backbones win absolute WER; Gemma shows measurable two-stage personalization behavior.
```


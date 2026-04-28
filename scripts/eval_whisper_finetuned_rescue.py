"""Rescue eval — same as eval_whisper_finetuned.py but with aggressive
length-capping decode, since the fine-tuned model has broken EOS prediction
and falls into repetition loops past ~30 tokens.

Each HYP from the locked-recipe eval starts with a faithful transcription
of the first ~5 words then explodes. So we cap max_new_tokens hard and
add no_repeat_ngram=2 to kill consecutive-word loops.
"""
from __future__ import annotations
import os, json, argparse, time, traceback

import torch
import soundfile as sf

from transformers import WhisperProcessor, WhisperForConditionalGeneration

from eval_recipe import (
    normalizer, lowercase_norm, aggregate_pairs, bootstrap_wer_ci,
    per_severity_wer, pad_audio_to_safe_length, SS_TEST_PATH,
)


def load_manifest(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def find_model_dir(d):
    if os.path.isdir(os.path.join(d, "final")):
        return os.path.join(d, "final")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--holdout", default=SS_TEST_PATH)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=50,
                    help="Hard cap. Aphasic refs are <40 chars (~12 tokens).")
    ap.add_argument("--length-penalty", type=float, default=0.3,
                    help="<1 favors short sequences. Aggressive.")
    ap.add_argument("--no-repeat-ngram", type=int, default=2,
                    help="Kill consecutive 2-grams (catches a-a-a, knife-knife-knife loops).")
    ap.add_argument("--rep-penalty", type=float, default=1.5,
                    help="Stronger than locked recipe's 1.3.")
    ap.add_argument("--num-beams", type=int, default=5)
    ap.add_argument("--early-stopping", action="store_true", default=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model_dir = find_model_dir(args.model_dir)
    print(f"[{time.strftime('%H:%M:%S')}] loading {model_dir} ...", flush=True)
    processor = WhisperProcessor.from_pretrained(model_dir)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_dir, torch_dtype=torch.float16,
    ).cuda().eval()
    model.generation_config.language = "english"
    model.generation_config.task = "transcribe"
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        num_beams=args.num_beams,
        length_penalty=args.length_penalty,
        repetition_penalty=args.rep_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram,
        early_stopping=args.early_stopping,
    )

    holdout = load_manifest(args.holdout)
    if args.limit:
        holdout = holdout[: args.limit]
    print(f"[{time.strftime('%H:%M:%S')}] rescue eval on {len(holdout)} clips", flush=True)
    print(f"  gen_kwargs: {gen_kwargs}")

    outs = []
    n_errors = 0
    for i, r in enumerate(holdout):
        try:
            audio, sr = sf.read(r["audio_path"])
            assert sr == 16000
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            audio = pad_audio_to_safe_length(audio, sr=16000, quantum_s=1.0)
            inputs = processor.feature_extractor(
                audio, sampling_rate=16000, return_tensors="pt",
            )
            input_features = inputs.input_features.to("cuda", dtype=torch.float16)
            with torch.no_grad():
                pred = model.generate(
                    input_features,
                    forced_decoder_ids=forced_decoder_ids,
                    **gen_kwargs,
                )
            text = processor.tokenizer.batch_decode(
                pred, skip_special_tokens=True,
            )[0].strip()
        except Exception as e:
            if n_errors < 3:
                print(f"\n=== TRACEBACK clip {i} ===")
                traceback.print_exc()
            n_errors += 1
            text = f"<ERROR:{type(e).__name__}>"
        outs.append({
            "ref": r["transcript"], "hyp": text,
            "severity": r.get("severity"),
            "speaker": r.get("participant_id"),
        })
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(holdout)} done", flush=True)

    result = {"name": os.path.basename(args.model_dir.rstrip("/")) + "_rescue",
              "n_predictions": len(outs), "n_errors": n_errors}
    for label, norm in (("lowercase", lowercase_norm), ("whisperEN", normalizer)):
        rows = aggregate_pairs(outs, norm=norm)
        result[f"{label}_overall"] = bootstrap_wer_ci(rows)
        result[f"{label}_per_severity"] = per_severity_wer(rows)

    print()
    print("=" * 70)
    print(f"  model: {model_dir}")
    print(f"  rescue gen_kwargs: {gen_kwargs}")
    print(f"  errors: {n_errors}/{len(outs)}")
    we = result.get("whisperEN_overall") or {}
    lo = result.get("lowercase_overall") or {}
    print(f"  --- WhisperEN ---")
    print(f"    overall WER: {we.get('wer')}  CI95 [{we.get('ci95_lo')}, {we.get('ci95_hi')}]")
    print(f"    per-severity: {result['whisperEN_per_severity']}")
    print(f"  --- lowercase ---")
    print(f"    overall WER: {lo.get('wer')}  CI95 [{lo.get('ci95_lo')}, {lo.get('ci95_hi')}]")
    print(f"    per-severity: {result['lowercase_per_severity']}")
    print()
    print(f"  Reference: Whisper-aggressive zero-shot WhisperEN = 0.347")
    print(f"             R1-D4b WhisperEN                       = 0.456")
    print("=" * 70)

    out_path = os.path.join(args.output_dir, "ft_eval_rescue.json")
    with open(out_path, "w") as f:
        json.dump({
            "model_dir": model_dir,
            "holdout": args.holdout,
            "gen_kwargs": gen_kwargs,
            "result": result,
            "predictions": outs,
        }, f, indent=2)
    print(f"\n  Saved {out_path}")


if __name__ == "__main__":
    main()

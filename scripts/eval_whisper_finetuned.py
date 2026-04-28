"""Eval a fine-tuned Whisper checkpoint with the locked Tier-3 recipe.

Loads the saved Whisper FT directory and runs LOCKED_GEN_KWARGS on ss_test.
Reports lowercase + WhisperEN WER, per-severity, bootstrap CIs.

Usage:
    python scripts/eval_whisper_finetuned.py \\
        --model-dir outputs/whisper_ft_1.2_baseline/final \\
        --holdout   $APHASIA_ROOT/aphasiabank_ss_test.jsonl \\
        --output-dir outputs/eval_whisper_ft_1.2

If --model-dir doesn't have a `final/` subdir, it will use the dir as-is
(useful for evaluating the load-best-model-at-end checkpoint at stageN/).
"""
from __future__ import annotations
import os, json, argparse, time, traceback

import torch
import soundfile as sf

from transformers import WhisperProcessor, WhisperForConditionalGeneration

from eval_recipe import (
    LOCKED_GEN_KWARGS, normalizer, lowercase_norm,
    aggregate_pairs, bootstrap_wer_ci, per_severity_wer,
    pad_audio_to_safe_length, SS_TEST_PATH,
)


def load_manifest(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def find_model_dir(d):
    """If `d/final` exists, use it; else use `d` directly."""
    if os.path.isdir(os.path.join(d, "final")):
        return os.path.join(d, "final")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True,
                    help="Path to saved Whisper FT (or its parent — auto-detects /final)")
    ap.add_argument("--holdout", default=SS_TEST_PATH)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-pad", action="store_true",
                    help="Skip 1-second pad (Whisper has its own padding)")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model_dir = find_model_dir(args.model_dir)
    print(f"[{time.strftime('%H:%M:%S')}] loading {model_dir} ...", flush=True)
    processor = WhisperProcessor.from_pretrained(model_dir)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_dir, torch_dtype=torch.float16,
    ).cuda().eval()

    # English-only generation; mirror train_whisper_large_ft.py settings
    model.generation_config.language = "english"
    model.generation_config.task = "transcribe"
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")

    holdout = load_manifest(args.holdout)
    if args.limit:
        holdout = holdout[: args.limit]
    print(f"[{time.strftime('%H:%M:%S')}] eval on {len(holdout)} clips with locked recipe", flush=True)
    print(f"  gen_kwargs: {LOCKED_GEN_KWARGS}")

    outs = []
    n_errors = 0
    for i, r in enumerate(holdout):
        try:
            audio, sr = sf.read(r["audio_path"])
            assert sr == 16000, f"expected 16 kHz got {sr}"
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if not args.no_pad:
                audio = pad_audio_to_safe_length(audio, sr=16000, quantum_s=1.0)
            inputs = processor.feature_extractor(
                audio, sampling_rate=16000, return_tensors="pt",
            )
            input_features = inputs.input_features.to("cuda", dtype=torch.float16)
            with torch.no_grad():
                pred = model.generate(
                    input_features,
                    forced_decoder_ids=forced_decoder_ids,
                    **LOCKED_GEN_KWARGS,
                )
            text = processor.tokenizer.batch_decode(
                pred, skip_special_tokens=True
            )[0].strip()
        except Exception as e:
            if n_errors < 3:
                print(f"\n=== TRACEBACK clip {i} ({r['audio_path']}) ===")
                traceback.print_exc()
                print("=== END ===\n", flush=True)
            n_errors += 1
            text = f"<ERROR:{type(e).__name__}>"
        outs.append({
            "ref": r["transcript"], "hyp": text,
            "severity": r.get("severity"),
            "speaker": r.get("participant_id"),
        })
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(holdout)} done", flush=True)

    # Score with both normalizers
    result = {"name": os.path.basename(args.model_dir.rstrip("/")),
              "n_predictions": len(outs), "n_errors": n_errors}
    for label, norm in (("lowercase", lowercase_norm), ("whisperEN", normalizer)):
        rows = aggregate_pairs(outs, norm=norm)
        result[f"{label}_overall"] = bootstrap_wer_ci(rows)
        result[f"{label}_per_severity"] = per_severity_wer(rows)

    # Print headline
    print()
    print("=" * 70)
    print(f"  model: {model_dir}")
    print(f"  errors: {n_errors}/{len(outs)}")
    we = result.get("whisperEN_overall") or {}
    lo = result.get("lowercase_overall") or {}
    print(f"  --- WhisperEN (paper-comparable) ---")
    print(f"    overall WER: {we.get('wer')}  CI95 [{we.get('ci95_lo')}, {we.get('ci95_hi')}]  n={we.get('n')}")
    print(f"    per-severity: {result['whisperEN_per_severity']}")
    print(f"  --- lowercase ---")
    print(f"    overall WER: {lo.get('wer')}  CI95 [{lo.get('ci95_lo')}, {lo.get('ci95_hi')}]")
    print(f"    per-severity: {result['lowercase_per_severity']}")
    print()
    print(f"  Reference: Whisper-aggressive zero-shot WhisperEN = 0.347 (CI95 [0.309, 0.386])")
    print(f"             R1-D4b WhisperEN                       = 0.456 (CI95 [0.410, 0.501])")
    print(f"             BASE Gemma 4 WhisperEN                 = 0.486 (CI95 [0.436, 0.541])")
    print("=" * 70)

    out_path = os.path.join(args.output_dir, "ft_eval.json")
    with open(out_path, "w") as f:
        json.dump({
            "model_dir": model_dir,
            "holdout": args.holdout,
            "gen_kwargs": LOCKED_GEN_KWARGS,
            "result": result,
            "predictions": outs,
        }, f, indent=2)
    print(f"\n  Saved {out_path}")


if __name__ == "__main__":
    main()

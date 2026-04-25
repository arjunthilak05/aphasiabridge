"""Zero-shot Whisper-Large-v3 baseline on the same ss_test 220 clips.

Two purposes:
  1. Paper comparability — Jiang et al. 2026 used Whisper-Large-v3 baseline = 39.97% WER
     (their B1) on AphasiaBank protocol-compliant subset. Different test split, but
     same model, so we should land in a similar ballpark.
  2. Architecture decision for CONTINGENCY.md Strategy A: if zero-shot Whisper >>
     BASE Gemma 4 (0.556) on aphasic speech, the right shipping plan is
     Whisper(ASR) → Gemma(intent), not Gemma direct.

WER is computed three ways for fair comparisons:
  - Lowercase only (matches our LoRA eval — directly comparable)
  - HuggingFace Whisper EnglishTextNormalizer (paper-comparable)
  - Both also report CER alongside.
"""
from __future__ import annotations
import os, sys, json, argparse, time, traceback
from collections import defaultdict

_WIN_HF = r"D:\Downloads\huggingface_cache"
os.environ.setdefault("HF_HOME", _WIN_HF if os.path.isdir(_WIN_HF) else os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("HF_HUB_CACHE", os.path.join(os.environ["HF_HOME"], "hub"))

import torch
import soundfile as sf
import jiwer

from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer


def load_manifest(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/whisper-large-v3")
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[{time.strftime('%H:%M:%S')}] loading {args.model} ...", flush=True)
    processor = WhisperProcessor.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
    ).cuda().eval()
    print(f"  loaded; VRAM={torch.cuda.memory_allocated()/1e9:.2f} GB")

    forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
    normalizer = EnglishTextNormalizer({})

    holdout = load_manifest(args.holdout)
    if args.limit:
        holdout = holdout[:args.limit]
    print(f"[{time.strftime('%H:%M:%S')}] eval on {len(holdout)} clips ...", flush=True)

    outs = []
    n_errors = 0
    for i, r in enumerate(holdout):
        try:
            audio, sr = sf.read(r["audio_path"])
            assert sr == 16000, f"expected 16 kHz got {sr}"
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
            input_features = inputs.input_features.to("cuda", dtype=torch.float16)
            with torch.no_grad():
                pred = model.generate(
                    input_features,
                    forced_decoder_ids=forced_decoder_ids,
                    max_new_tokens=256,
                    num_beams=5,           # paper used beam=5
                    do_sample=False,
                )
            text = processor.batch_decode(pred, skip_special_tokens=True)[0].strip()
        except Exception as e:
            if n_errors < 3:
                print(f"\n=== TRACEBACK clip {i} ({r['audio_path']}) dur={r.get('duration_ms')} ===")
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

    # Compute WER three ways
    def _agg(outs, transform):
        by_sev = defaultdict(lambda: {"refs": [], "hyps": []})
        good = [o for o in outs if not o["hyp"].startswith("<ERROR")]
        for o in good:
            r = transform(o["ref"]); h = transform(o["hyp"])
            if not r or not h: continue
            by_sev[o["severity"] or "unknown"]["refs"].append(r)
            by_sev[o["severity"] or "unknown"]["hyps"].append(h)
        refs = [transform(o["ref"]) for o in good]
        hyps = [transform(o["hyp"]) for o in good]
        # filter empties (normalizer can leave empty strings)
        pairs = [(r, h) for r, h in zip(refs, hyps) if r and h]
        if not pairs:
            return {"wer": None, "cer": None, "n": 0, "per_severity_wer": {}}
        rs, hs = zip(*pairs)
        return {
            "wer": round(jiwer.wer(list(rs), list(hs)), 4),
            "cer": round(jiwer.cer(list(rs), list(hs)), 4),
            "n": len(pairs),
            "per_severity_wer": {
                k: round(jiwer.wer(v["refs"], v["hyps"]), 4)
                for k, v in by_sev.items() if v["refs"]
            },
        }

    lower = _agg(outs, lambda s: s.lower().strip())
    norm  = _agg(outs, lambda s: normalizer(s))

    print()
    print("=" * 64)
    print(f"Whisper-Large-v3 zero-shot on {len(holdout)} ss_test clips:")
    print(f"  errors: {n_errors}/{len(holdout)} ({n_errors/len(holdout)*100:.1f}%)")
    print(f"  --- lowercase-only (comparable to our LoRA evals) ---")
    print(f"    WER: {lower['wer']}    CER: {lower['cer']}    n={lower['n']}")
    print(f"    per-severity: {lower['per_severity_wer']}")
    print(f"  --- Whisper EnglishTextNormalizer (comparable to paper) ---")
    print(f"    WER: {norm['wer']}    CER: {norm['cer']}    n={norm['n']}")
    print(f"    per-severity: {norm['per_severity_wer']}")
    print()
    print(f"Reference: paper Jiang et al. 2026 B1 (Whisper-Large-v3) = 39.97% WER on their AphasiaBank test split")
    print("=" * 64)

    out_path = os.path.join(args.output_dir, "whisper_baseline.json")
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "holdout": args.holdout,
            "n_clips": len(holdout),
            "n_errors": n_errors,
            "wer_lowercase": lower,
            "wer_normalized": norm,
            "predictions": outs,
        }, f, indent=2)
    print(f"\n  Saved {out_path}")


if __name__ == "__main__":
    main()

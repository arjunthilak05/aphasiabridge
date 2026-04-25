"""Tier-1 Lever 4: 5-prompt sweep on a 50-clip dev subset using BASE Gemma 4 (no LoRA).

Why BASE only: the LoRA was trained with the current SYSTEM_PROMPT/USER_PROMPT pair.
Changing prompts at inference may degrade adapter performance. Running the sweep on
BASE first answers "is there a better prompt" without confounding from the adapter.
If a winner emerges, we then test it on D4b.

Output: a small JSON with 5 rows × WER (lowercase + WhisperEN + per-severity).
The dev set is 50 clips sampled deterministically from aphasiabank_si_val.jsonl.
"""
from __future__ import annotations
import argparse, json, os, time, traceback
from collections import defaultdict
from pathlib import Path
import random

_WIN_HF = r"D:\Downloads\huggingface_cache"
os.environ.setdefault("HF_HOME", _WIN_HF if os.path.isdir(_WIN_HF) else os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("HF_HUB_CACHE", os.path.join(os.environ["HF_HOME"], "hub"))
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
torch._dynamo.config.disable = True
torch._dynamo.config.suppress_errors = True

from unsloth import FastModel
import soundfile as sf
import jiwer
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

MODEL_ID = "unsloth/gemma-4-E4B-it-unsloth-bnb-4bit"

VARIANTS = [
    {
        "name": "control",
        "system": "You are an assistant that transcribes speech accurately.",
        "user":   "Please transcribe this audio.",
        "no_system": False,
    },
    {
        "name": "aphasia-aware",
        "system": "This is speech from a person with aphasia. Preserve repetitions, fillers, and partial words exactly. Do not auto-correct.",
        "user":   "Please transcribe this audio.",
        "no_system": False,
    },
    {
        "name": "literal-emphasis",
        "system": "Transcribe word-for-word, including repetitions and partial words. Do not auto-correct or paraphrase.",
        "user":   "Please transcribe this audio.",
        "no_system": False,
    },
    {
        "name": "asr-prefix",
        "system": "You are an assistant that transcribes speech accurately.",
        "user":   "Transcribe the following audio:",
        "no_system": False,
    },
    {
        "name": "no-system-literal",
        "system": "",
        "user":   "Transcribe the audio verbatim, preserving all repetitions and partial words.",
        "no_system": True,
    },
]


def load_manifest(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-manifest", required=True)
    ap.add_argument("--n-dev", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--rep-penalty", type=float, default=1.3)
    ap.add_argument("--no-repeat-ngram", type=int, default=4)
    ap.add_argument("--num-beams", type=int, default=5)
    ap.add_argument("--length-penalty", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    pool = load_manifest(args.dev_manifest)
    rng = random.Random(args.seed)
    dev = rng.sample(pool, min(args.n_dev, len(pool)))
    print(f"[{time.strftime('%H:%M:%S')}] dev set: {len(dev)} clips from {args.dev_manifest}", flush=True)

    print(f"[{time.strftime('%H:%M:%S')}] loading BASE {MODEL_ID} ...", flush=True)
    model, processor = FastModel.from_pretrained(
        model_name=MODEL_ID, dtype=None,
        max_seq_length=2048, load_in_4bit=True, full_finetuning=False,
    )
    model.eval()
    en_norm = EnglishTextNormalizer({})

    all_results = []
    for v in VARIANTS:
        print(f"\n=== variant: {v['name']} ===", flush=True)
        outs = []
        n_err = 0
        for i, r in enumerate(dev):
            try:
                audio, sr = sf.read(r["audio_path"])
                if sr != 16000:
                    outs.append({"ref": r["transcript"], "hyp": "<ERROR:bad_sr>",
                                 "severity": r.get("severity")})
                    continue
                msgs = []
                if not v["no_system"] and v["system"]:
                    msgs.append({"role": "system",
                                 "content": [{"type": "text", "text": v["system"]}]})
                msgs.append({"role": "user",
                             "content": [
                                 {"type": "audio", "audio": audio},
                                 {"type": "text",  "text": v["user"]},
                             ]})
                inputs = processor.apply_chat_template(
                    msgs, add_generation_prompt=True, tokenize=True,
                    return_dict=True, return_tensors="pt",
                )
                inputs = {k: x.to(model.device) if hasattr(x, "to") else x for k, x in inputs.items()}
                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens, do_sample=False,
                        temperature=None, top_p=None,
                        repetition_penalty=args.rep_penalty,
                        no_repeat_ngram_size=args.no_repeat_ngram,
                        num_beams=args.num_beams,
                        length_penalty=args.length_penalty,
                    )
                input_len = inputs["input_ids"].shape[1]
                text = processor.decode(out[0, input_len:], skip_special_tokens=True).strip()
            except Exception:
                n_err += 1
                text = "<ERROR>"
            outs.append({"ref": r["transcript"], "hyp": text,
                         "severity": r.get("severity"),
                         "speaker": r.get("participant_id")})

        # WER (lowercase + WhisperEN)
        good = [o for o in outs if not o["hyp"].startswith("<ERROR")]
        lc_refs = [o["ref"].lower().strip() for o in good]
        lc_hyps = [o["hyp"].lower().strip() for o in good]
        en_refs = [en_norm(o["ref"]) for o in good]
        en_hyps = [en_norm(o["hyp"]) for o in good]
        # filter empties post-normalization
        lc_pairs = [(r, h) for r, h in zip(lc_refs, lc_hyps) if r and h]
        en_pairs = [(r, h) for r, h in zip(en_refs, en_hyps) if r and h]
        wer_lc = round(jiwer.wer([r for r,_ in lc_pairs], [h for _,h in lc_pairs]), 4) if lc_pairs else None
        wer_en = round(jiwer.wer([r for r,_ in en_pairs], [h for _,h in en_pairs]), 4) if en_pairs else None
        # per-severity (en)
        by_sev = defaultdict(lambda: {"r": [], "h": []})
        for o, nr, nh in zip(good, en_refs, en_hyps):
            if not nr or not nh: continue
            by_sev[o.get("severity") or "unknown"]["r"].append(nr)
            by_sev[o.get("severity") or "unknown"]["h"].append(nh)
        per_sev = {k: round(jiwer.wer(v["r"], v["h"]), 4) for k, v in by_sev.items() if v["r"]}

        result = {
            "variant": v["name"], "system": v["system"], "user": v["user"],
            "no_system": v["no_system"],
            "n_total": len(outs), "n_errors": n_err,
            "wer_lowercase": wer_lc, "wer_whisperEN": wer_en,
            "per_severity_whisperEN": per_sev,
            "predictions": outs[:10],   # save first 10 for spot-check
        }
        all_results.append(result)
        print(f"  WER lowercase={wer_lc}  WhisperEN={wer_en}  per-sev(EN)={per_sev}  errors={n_err}/{len(outs)}", flush=True)

    # rank
    all_results.sort(key=lambda r: r["wer_whisperEN"] if r["wer_whisperEN"] is not None else 99)
    print(f"\n{'='*80}\nRanking (best → worst, by WhisperEN WER):\n{'='*80}")
    for r in all_results:
        print(f"  {r['variant']:24s}  WER_lc={r['wer_lowercase']}  WER_EN={r['wer_whisperEN']}")

    out_path = os.path.join(args.output_dir, "prompt_sweep_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "model": MODEL_ID, "dev_manifest": args.dev_manifest, "n_dev": args.n_dev,
            "decoding": {
                "rep_penalty": args.rep_penalty, "no_repeat_ngram": args.no_repeat_ngram,
                "num_beams": args.num_beams, "length_penalty": args.length_penalty,
                "max_new_tokens": args.max_new_tokens,
            },
            "variants": all_results,
        }, f, indent=2)
    print(f"\n  Saved {out_path}")


if __name__ == "__main__":
    main()

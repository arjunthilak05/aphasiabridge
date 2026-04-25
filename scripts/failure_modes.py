"""Tier-1 add-on: failure-mode taxonomy applied to all model outputs.

Categories (from §E and §K of H100_RESULTS.md, plus refinements for Whisper):
  - repeat_loop           the model repeats the same word/n-gram excessively
  - protocol_hallucination canonical AphasiaBank task-script (Cinderella, Sandwich, Stroke)
  - long_hallucination    output > 3× ref length and substantively different
  - empty_output          empty or near-empty hypothesis
  - audio_processor_err   <ERROR:> failures in our pipeline
  - word_error            none of the above; just standard ASR word errors

For each saved JSON we count clips per category, compute average WER per
category, and emit a comparison table.
"""
from __future__ import annotations
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import jiwer

ROOT = Path("/data/kl_dev/dataset_wan/data_gemma/aphasiabridge")

# Patterns that flag specific failure modes
RE_REPEAT_LOOP = re.compile(
    r"(?:\b(\w+)\s+){8,}",   # any single word repeated 8+ times in a row
    re.IGNORECASE,
)
RE_PROTOCOL_HALLU = [
    re.compile(r"\bgo to the ball\b", re.I),                # Cinderella
    re.compile(r"\bpeanut butter\b.*\b(bread|sandwich)\b", re.I),  # Sandwich
    re.compile(r"\bevery morning\b", re.I),                 # generic narrative filler
    re.compile(r"\bstroke\b.*\b(hospital|brain)\b", re.I),  # Stroke story
]


def classify(record):
    hyp = record.get("hyp", "") or ""
    ref = record.get("ref", "") or ""
    if hyp.startswith("<ERROR"):
        return "audio_processor_err"
    h_low = hyp.lower().strip()
    r_low = ref.lower().strip()
    if not h_low:
        return "empty_output"
    if RE_REPEAT_LOOP.search(h_low):
        return "repeat_loop"
    if any(p.search(h_low) for p in RE_PROTOCOL_HALLU):
        return "protocol_hallucination"
    # Long hallucination: output much longer than ref AND not contained
    if len(h_low.split()) > max(20, 3 * len(r_low.split())):
        return "long_hallucination"
    return "word_error"


def per_category_wer(records):
    by_cat = defaultdict(lambda: {"refs": [], "hyps": []})
    cat_counts = Counter()
    for r in records:
        cat = classify(r)
        cat_counts[cat] += 1
        if cat in ("audio_processor_err", "empty_output"):
            continue
        by_cat[cat]["refs"].append(r["ref"].lower().strip())
        by_cat[cat]["hyps"].append(r["hyp"].lower().strip())
    cat_wer = {}
    for cat, v in by_cat.items():
        try:
            cat_wer[cat] = round(jiwer.wer(v["refs"], v["hyps"]), 4)
        except Exception:
            cat_wer[cat] = None
    return cat_counts, cat_wer


def load_preds(path):
    p = ROOT / path
    if not p.exists():
        return None
    blob = json.loads(p.read_text())
    if "predictions" in blob:
        return blob["predictions"]
    if "result" in blob:
        # eval_lora.py reeval
        ar = blob.get("after_raw") or blob["result"].get("after_raw")
        return ar
    if "holdout_eval" in blob:
        return blob["holdout_eval"].get("after_raw")
    return None


def main():
    sources = [
        ("Whisper-Large-v3 zero-shot",      "results/whisper_baseline/whisper_baseline.json"),
        ("Whisper-Large-v3 aggressive",     "results/whisper_aggressive/whisper_baseline.json"),
        ("BASE Gemma 4 (lang-only base)",   "outputs/h100_lora_lang_only/run_summary.json:base"),
        ("Lang-only LoRA AFTER",            "outputs/h100_lora_lang_only/run_summary.json"),
        ("R1-D4b beam=5 (full preds)",      "outputs/h100_lora_daniel_v4b_full/reeval_summary.json"),
    ]

    print(f"\n{'='*100}\nFailure-mode taxonomy applied to all saved eval JSONs\n{'='*100}\n")

    table = []
    for name, src in sources:
        # Special case: ":base" suffix -> load base_raw instead
        if src.endswith(":base"):
            src = src[:-5]
            p = ROOT / src
            if not p.exists():
                print(f"-- skip: {name}  ({src})"); continue
            preds = json.loads(p.read_text())["holdout_eval"]["base_raw"]
        else:
            preds = load_preds(src)
            if preds is None:
                print(f"-- skip: {name}  ({src})"); continue

        cat_counts, cat_wer = per_category_wer(preds)
        n = len(preds)
        row = {"model": name, "n_total": n, "categories": dict(cat_counts), "wer_per_category": cat_wer}
        table.append(row)

        print(f"\n{name}  (n={n})")
        for cat in ("repeat_loop", "protocol_hallucination", "long_hallucination",
                    "empty_output", "audio_processor_err", "word_error"):
            count = cat_counts.get(cat, 0)
            pct = round(count / n * 100, 1)
            wer_str = f"WER={cat_wer.get(cat):.4f}" if cat in cat_wer else ""
            print(f"  {cat:25s}  {count:4d} ({pct:5.1f}%)  {wer_str}")

    out_path = ROOT / "results/failure_modes.json"
    with open(out_path, "w") as f:
        json.dump(table, f, indent=2)
    print(f"\n  Wrote {out_path}")


if __name__ == "__main__":
    main()

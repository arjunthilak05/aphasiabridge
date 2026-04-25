"""Tier-1 Lever 1: recompute WER on every saved eval JSON using
Whisper's EnglishTextNormalizer (the same normalizer Jiang et al. 2026
use). Pure post-processing; no model load.

Reads either schema:
  - reeval_summary.json (eval_lora.py output, has 'result.after_raw' / 'result.base_raw')
  - run_summary.json    (train_gemma4_audio.py output, has 'holdout_eval.{after_raw,base_raw}')
  - whisper_baseline.json (eval_whisper_baseline.py output, has 'predictions')

Outputs a side-by-side WER table (lowercase / Whisper EnglishTextNormalizer / BasicTextNormalizer)
for every JSON found, plus per-severity breakdowns under the Whisper normalizer.
"""
from __future__ import annotations
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import jiwer
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer, BasicTextNormalizer

ROOT = Path("/data/kl_dev/dataset_wan/data_gemma/aphasiabridge")

TARGETS = [
    ("D4a beam=5",       "outputs/h100_lora_daniel_v4_beam5/reeval_summary.json"),
    ("D4b beam=5 ★",     "outputs/h100_lora_daniel_v4b_beam5/reeval_summary.json"),
    ("Lang-only beam=5", "outputs/h100_lora_lang_only_beam5/reeval_summary.json"),
    ("Mixed beam=5",     "outputs/h100_lora_mixed_beam5/reeval_summary.json"),
    ("D4a beam=1 (Phase-1)", "outputs/h100_lora_daniel_v4/reeval_summary.json"),
    ("D4b beam=1 (Phase-1)", "outputs/h100_lora_daniel_v4b/reeval_summary.json"),
    ("Lang-only in-script (beam=1)", "outputs/h100_lora_lang_only/run_summary.json"),
    ("Mixed in-script (beam=1)",     "outputs/h100_lora_mixed/run_summary.json"),
    ("Whisper-Large-v3 zero-shot",   "results/whisper_baseline/whisper_baseline.json"),
]


def extract_pairs(blob: dict, tag: str):
    """Return (label, refs[], hyps[], severities[]) tuples for BASE and AFTER passes."""
    out = []
    # eval_lora.py
    if "result" in blob:
        r = blob["result"]
        for label, raw_key in (("BASE", "base_raw"), ("AFTER", "after_raw")):
            raw = blob.get(raw_key)
            if raw is None and label.lower() in r:
                raw = r.get(f"{label.lower()}_raw")
            if raw:
                out.append((
                    label,
                    [o["ref"] for o in raw],
                    [o["hyp"] for o in raw],
                    [o.get("severity") for o in raw],
                ))
    # train_gemma4_audio.py
    elif "holdout_eval" in blob:
        ho = blob["holdout_eval"]
        for label, raw_key in (("BASE", "base_raw"), ("AFTER", "after_raw")):
            raw = ho.get(raw_key)
            if raw:
                out.append((
                    label,
                    [o["ref"] for o in raw],
                    [o["hyp"] for o in raw],
                    [o.get("severity") for o in raw],
                ))
    # whisper_baseline.json
    elif "predictions" in blob:
        preds = blob["predictions"]
        out.append((
            "Whisper-zero-shot",
            [o["ref"] for o in preds],
            [o["hyp"] for o in preds],
            [o.get("severity") for o in preds],
        ))
    return out


def compute_wer(refs, hyps, sevs, normalizer):
    """Apply normalizer + filter ERROR/empty rows + return overall + per-severity WER."""
    pairs = []
    by_sev = defaultdict(lambda: {"refs": [], "hyps": []})
    for r, h, sev in zip(refs, hyps, sevs):
        if h.startswith("<ERROR"):
            continue
        nr = normalizer(r)
        nh = normalizer(h)
        if not nr.strip() or not nh.strip():
            continue
        pairs.append((nr, nh))
        sev_key = sev or "unknown"
        by_sev[sev_key]["refs"].append(nr)
        by_sev[sev_key]["hyps"].append(nh)
    if not pairs:
        return None, {}, 0
    rs, hs = zip(*pairs)
    overall = jiwer.wer(list(rs), list(hs))
    per_sev = {
        k: jiwer.wer(v["refs"], v["hyps"])
        for k, v in by_sev.items() if v["refs"]
    }
    return overall, per_sev, len(pairs)


def lower_norm(s: str) -> str:
    """Match our existing convention: lowercase + strip."""
    return s.lower().strip()


def main():
    en_norm = EnglishTextNormalizer({})
    basic_norm = BasicTextNormalizer()

    results = []
    print(f"\n{'='*100}")
    print(f"Recomputing WER on saved eval JSONs (Lever 1: Whisper EnglishTextNormalizer)")
    print(f"{'='*100}\n")

    for tag, rel_path in TARGETS:
        p = ROOT / rel_path
        if not p.exists():
            print(f"-- skip (missing): {tag}  {rel_path}")
            continue
        try:
            blob = json.loads(p.read_text())
        except Exception as e:
            print(f"-- skip (bad JSON): {tag}  {e}")
            continue

        passes = extract_pairs(blob, tag)
        if not passes:
            print(f"-- skip (no raw predictions): {tag}")
            continue

        for label, refs, hyps, sevs in passes:
            row = {"tag": tag, "pass": label, "n_total": len(refs)}
            for nname, norm in (("lowercase", lower_norm),
                                ("WhisperEN", en_norm),
                                ("BasicNorm", basic_norm)):
                wer, per_sev, n = compute_wer(refs, hyps, sevs, norm)
                row[f"WER_{nname}"] = round(wer, 4) if wer is not None else None
                row[f"n_{nname}"] = n
                if nname == "WhisperEN":
                    row["per_severity_WhisperEN"] = {k: round(v, 4) for k, v in per_sev.items()}
            results.append(row)

    # --- print summary table -----------------------------------------------
    if not results:
        print("No results found."); return

    print(f"{'Run':38s} {'pass':6s} {'n_total':>7s} {'lowercase':>10s} {'WhisperEN':>10s} {'BasicNorm':>10s}  Δ_norm_vs_lower")
    print("-" * 110)
    for row in results:
        lower = row["WER_lowercase"]
        wn    = row["WER_WhisperEN"]
        bn    = row["WER_BasicNorm"]
        delta = round(wn - lower, 4) if (lower is not None and wn is not None) else "NA"
        delta_str = (f"{delta:+.4f}" if isinstance(delta, float) else str(delta))
        print(f"{row['tag']:38s} {row['pass']:6s} {row['n_total']:>7d}   {str(lower):>8s}   {str(wn):>8s}   {str(bn):>8s}   {delta_str:>9s}")

    print()
    print("Per-severity WER (Whisper EnglishTextNormalizer)")
    print("-" * 100)
    for row in results:
        if not row.get("per_severity_WhisperEN"):
            continue
        print(f"{row['tag']:38s} {row['pass']:6s}   {row['per_severity_WhisperEN']}")

    out_path = ROOT / "results" / "wer_normalizer_comparison.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Wrote {out_path}")


if __name__ == "__main__":
    main()

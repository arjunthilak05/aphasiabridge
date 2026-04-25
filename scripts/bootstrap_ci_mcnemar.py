"""Tier-1 add-on: bootstrap 95% CIs + McNemar test (Whisper vs D4b on moderate).

Reads:
  - results/whisper_baseline/whisper_baseline.json   (220 Whisper preds)
  - outputs/h100_lora_daniel_v4b_full/reeval_summary.json  (220 D4b preds, after rerun)
  - outputs/h100_lora_lang_only/run_summary.json   (BASE Gemma 4 preds via lang-only base pass)

Outputs:
  - results/bootstrap_results.json       full table + CIs
  - prints headline results
"""
from __future__ import annotations
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import jiwer
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

ROOT = Path("/data/kl_dev/dataset_wan/data_gemma/aphasiabridge")

en_norm = EnglishTextNormalizer({})
N_BOOT = 1000
SEED = 42


def per_clip_wer(refs, hyps):
    """jiwer.wer is corpus-level; compute per-clip WER manually for McNemar."""
    out = []
    for r, h in zip(refs, hyps):
        if not r:
            out.append(0.0); continue
        try:
            out.append(jiwer.wer([r], [h]))
        except Exception:
            out.append(1.0)
    return out


def aggregate(preds, normalizer):
    """Apply normalizer + filter ERROR/empty, return aligned (ref, hyp, sev) tuples."""
    rows = []
    for p in preds:
        h = p["hyp"]
        if h.startswith("<ERROR"):
            continue
        nr = normalizer(p["ref"])
        nh = normalizer(h)
        if not nr or not nh:
            continue
        rows.append((nr, nh, p.get("severity") or "unknown"))
    return rows


def bootstrap_wer_ci(rows, n_boot=N_BOOT, seed=SEED):
    if not rows:
        return None
    rng = np.random.default_rng(seed)
    idxs = np.arange(len(rows))
    refs = [r[0] for r in rows]; hyps = [r[1] for r in rows]
    overall = jiwer.wer(refs, hyps)
    samples = []
    for _ in range(n_boot):
        sel = rng.choice(idxs, size=len(idxs), replace=True)
        sel_refs = [refs[i] for i in sel]
        sel_hyps = [hyps[i] for i in sel]
        try:
            samples.append(jiwer.wer(sel_refs, sel_hyps))
        except Exception:
            pass
    samples.sort()
    lo = samples[int(0.025 * len(samples))]
    hi = samples[int(0.975 * len(samples))]
    return {"wer": round(overall, 4),
            "ci95_lo": round(lo, 4), "ci95_hi": round(hi, 4),
            "n": len(rows)}


def per_severity_with_ci(rows, n_boot=N_BOOT):
    by_sev = defaultdict(list)
    for r, h, sev in rows:
        by_sev[sev].append((r, h, sev))
    return {sev: bootstrap_wer_ci(rs, n_boot) for sev, rs in by_sev.items()}


def mcnemar_per_clip(rows_a, rows_b):
    """Pair clips by (ref, sev) ordering — only works if both lists evaluated on
    the same 220 clips. Each row has the per-clip WER. McNemar table: how many
    clips where A < B vs B < A vs ties.
    """
    # Both should be in same clip order — caller must ensure
    if len(rows_a) != len(rows_b):
        return {"error": f"mismatched lengths: {len(rows_a)} vs {len(rows_b)}"}
    wer_a = per_clip_wer([r[0] for r in rows_a], [r[1] for r in rows_a])
    wer_b = per_clip_wer([r[0] for r in rows_b], [r[1] for r in rows_b])
    a_better = sum(1 for x, y in zip(wer_a, wer_b) if x < y - 1e-9)
    b_better = sum(1 for x, y in zip(wer_a, wer_b) if y < x - 1e-9)
    tied = sum(1 for x, y in zip(wer_a, wer_b) if abs(x - y) < 1e-9)
    # Two-sided binomial test on (a_better, b_better) under H0 = 0.5
    from scipy.stats import binomtest
    n_disagree = a_better + b_better
    if n_disagree == 0:
        return {"a_better": 0, "b_better": 0, "tied": tied, "p_value": 1.0}
    res = binomtest(min(a_better, b_better), n=n_disagree, p=0.5, alternative="two-sided")
    return {"a_better": a_better, "b_better": b_better, "tied": tied,
            "p_value": float(res.pvalue), "n_pairs": n_disagree,
            "median_a_wer": round(float(np.median(wer_a)), 4),
            "median_b_wer": round(float(np.median(wer_b)), 4)}


def main():
    print(f"\n{'='*100}\nBootstrap 95% CIs + McNemar Whisper-vs-D4b (Tier-1 statistical defense)\n{'='*100}\n")

    sources = {}
    # Whisper-Large-v3 zero-shot
    p = ROOT / "results/whisper_baseline/whisper_baseline.json"
    if p.exists():
        sources["Whisper-Large-v3 (zero-shot)"] = json.loads(p.read_text())["predictions"]

    # Whisper-Large-v3 aggressive decode (when it lands)
    p = ROOT / "results/whisper_aggressive/whisper_baseline.json"
    if p.exists():
        sources["Whisper-Large-v3 (aggressive decode)"] = json.loads(p.read_text())["predictions"]

    # BASE Gemma 4 (from lang-only's BASE pass — full 220)
    p = ROOT / "outputs/h100_lora_lang_only/run_summary.json"
    if p.exists():
        sources["BASE Gemma 4 E4B (beam=1)"] = json.loads(p.read_text())["holdout_eval"]["base_raw"]

    # D4b full preds
    p = ROOT / "outputs/h100_lora_daniel_v4b_full/reeval_summary.json"
    if p.exists():
        sources["R1-D4b (beam=5, rep=1.3, ngram=4)"] = json.loads(p.read_text())["result"]["after_raw"]
        sources["BASE Gemma 4 E4B (beam=5, same decode as D4b)"] = json.loads(p.read_text())["result"]["base_raw"]

    # Lang-only AFTER (full)
    p = ROOT / "outputs/h100_lora_lang_only/run_summary.json"
    if p.exists():
        sources["Lang-only LoRA (beam=1 in-script)"] = json.loads(p.read_text())["holdout_eval"]["after_raw"]

    full_results = {}
    for name, preds in sources.items():
        rows_lc  = aggregate(preds, lambda s: s.lower().strip())
        rows_en  = aggregate(preds, en_norm)
        full_results[name] = {
            "lowercase_overall":   bootstrap_wer_ci(rows_lc),
            "lowercase_per_severity": per_severity_with_ci(rows_lc),
            "whisperEN_overall":   bootstrap_wer_ci(rows_en),
            "whisperEN_per_severity": per_severity_with_ci(rows_en),
        }

    # --- print summary -----------------------------------------------------
    print(f"{'Model':45s}  {'overall_lc':>14s}  {'CI95_lc':>16s}  {'overall_EN':>14s}  {'CI95_EN':>16s}")
    print("-" * 120)
    for name, r in full_results.items():
        olc = r["lowercase_overall"]
        oen = r["whisperEN_overall"]
        if olc is None or oen is None:
            continue
        print(f"{name:45s}  {olc['wer']:>14.4f}  [{olc['ci95_lo']:.3f},{olc['ci95_hi']:.3f}]  "
              f"{oen['wer']:>14.4f}  [{oen['ci95_lo']:.3f},{oen['ci95_hi']:.3f}]")

    print(f"\nPer-severity (WhisperEN, lower=better) with 95% CI")
    print("-" * 120)
    headline_severities = ["mild", "moderate", "severe", "very_severe", "unknown"]
    for name, r in full_results.items():
        if r["whisperEN_per_severity"] is None:
            continue
        bits = []
        for sev in headline_severities:
            v = r["whisperEN_per_severity"].get(sev)
            if v is None:
                bits.append(f"{sev}=–")
            else:
                bits.append(f"{sev}={v['wer']:.3f} [{v['ci95_lo']:.3f},{v['ci95_hi']:.3f}] (n={v['n']})")
        print(f"  {name:45s}  " + "  ".join(bits))

    # McNemar: Whisper-Large-v3 (zero-shot) vs R1-D4b on moderate clips
    print(f"\nMcNemar: Whisper-Large-v3 zero-shot vs R1-D4b on moderate-severity clips (WhisperEN-normalized)")
    print("-" * 100)
    if "Whisper-Large-v3 (zero-shot)" in sources and "R1-D4b (beam=5, rep=1.3, ngram=4)" in sources:
        # Build clip-aligned arrays
        whisp = aggregate(sources["Whisper-Large-v3 (zero-shot)"], en_norm)
        d4b   = aggregate(sources["R1-D4b (beam=5, rep=1.3, ngram=4)"], en_norm)

        # Filter to moderate clips, align on (ref, sev) since clip order should match
        whisp_mod = [(r, h, s) for r, h, s in whisp if s == "moderate"]
        d4b_mod   = [(r, h, s) for r, h, s in d4b   if s == "moderate"]
        # Match by ref text (assumes deterministic clip ordering)
        ref_to_d4b = {r[0]: r for r in d4b_mod}
        aligned_w, aligned_d = [], []
        for w in whisp_mod:
            d = ref_to_d4b.get(w[0])
            if d is None:
                continue
            aligned_w.append(w); aligned_d.append(d)
        if aligned_w:
            res = mcnemar_per_clip(aligned_w, aligned_d)
            print(f"  n_aligned moderate clips: {len(aligned_w)}")
            print(f"  Whisper better on  : {res['a_better']} clips")
            print(f"  R1-D4b better on   : {res['b_better']} clips")
            print(f"  Tied               : {res['tied']} clips")
            print(f"  p-value (binomial) : {res['p_value']:.2e}")
            print(f"  median Whisper WER : {res['median_a_wer']}")
            print(f"  median R1-D4b WER  : {res['median_b_wer']}")
        else:
            print("  unable to align clips by ref")

    out_path = ROOT / "results/bootstrap_results.json"
    with open(out_path, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"\n  Wrote {out_path}")


if __name__ == "__main__":
    main()

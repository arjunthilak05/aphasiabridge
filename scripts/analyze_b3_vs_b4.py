"""Paired statistical analysis of B3 vs B4 SS-FT predictions.

Reads the JSON produced by eval_b3_vs_b4.py and reports:
  - Per-speaker WER for B3 and B4 (lowercase + WhisperEN)
  - Paired ΔWER = WER_B3 − WER_B4 per speaker (negative = B3 better)
  - Mean / median Δ across speakers
  - Win rate (fraction of speakers where B3 < B4)
  - Bootstrap 95 % CI on mean Δ
  - Sign test (binomial) p-value
  - Per-severity breakdown
  - Failure modes (errors, repetition loops, empty outputs)
"""
from __future__ import annotations
import argparse, json, re
from collections import defaultdict
from statistics import mean, median

import numpy as np
import jiwer
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

EN_NORM = EnglishTextNormalizer({})


def lowercase(s: str) -> str:
    return s.lower().strip()


def whisper_en(s: str) -> str:
    return EN_NORM(s).strip()


def per_clip_wer(preds, normalizer):
    """Returns list of (ref_norm, hyp_norm) for non-error clips."""
    out = []
    for p in preds:
        if p["hyp"].startswith("<ERROR"):
            continue
        nr = normalizer(p["ref"])
        nh = normalizer(p["hyp"])
        if not nr or not nh:
            continue
        out.append((nr, nh))
    return out


def speaker_wer(preds, normalizer):
    pairs = per_clip_wer(preds, normalizer)
    if not pairs:
        return None, 0
    refs, hyps = zip(*pairs)
    return jiwer.wer(list(refs), list(hyps)), len(pairs)


def detect_loop(s: str) -> bool:
    """Check if hypothesis has a repetition loop (8+ consecutive same word)."""
    return bool(re.search(r'\b(\w+)\b(?:\s+\1\b){7,}', s.lower()))


def bootstrap_mean_ci(deltas, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(deltas)
    samples = []
    arr = np.array(deltas)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        samples.append(arr[idx].mean())
    samples.sort()
    return {
        "mean": float(arr.mean()),
        "ci95_lo": float(samples[int(0.025 * n_boot)]),
        "ci95_hi": float(samples[int(0.975 * n_boot)]),
        "n": n,
    }


def sign_test(deltas):
    """Binomial p-value: how likely is the observed B3<B4 win-count under null Δ symmetric around 0?"""
    from scipy.stats import binomtest
    n_b3_better = sum(1 for d in deltas if d < 0)
    n_b4_better = sum(1 for d in deltas if d > 0)
    n_relevant = n_b3_better + n_b4_better
    if n_relevant == 0:
        return {"p_value": None, "n_b3_better": 0, "n_b4_better": 0, "n_tied": len(deltas)}
    res = binomtest(n_b3_better, n_relevant, p=0.5, alternative="two-sided")
    return {
        "p_value": float(res.pvalue),
        "n_b3_better": n_b3_better,
        "n_b4_better": n_b4_better,
        "n_tied": len(deltas) - n_relevant,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="JSON from eval_b3_vs_b4.py")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    data = json.load(open(args.input))
    preds_by_spk = data["predictions"]

    rows_lc = []
    rows_we = []
    severity_by_spk = {}

    for spk, two_conds in preds_by_spk.items():
        b3 = two_conds.get("b3", [])
        b4 = two_conds.get("b4", [])
        # severity: take first non-empty
        for p in b3 + b4:
            if p.get("severity"):
                severity_by_spk[spk] = p["severity"]; break

        # Paired clip-level: only include clips where BOTH B3 and B4 have non-error hyps
        b3_ok = {p["audio_path"]: p for p in b3 if not p["hyp"].startswith("<ERROR")}
        b4_ok = {p["audio_path"]: p for p in b4 if not p["hyp"].startswith("<ERROR")}
        common = sorted(set(b3_ok) & set(b4_ok))

        for label, norm in (("lowercase", lowercase), ("whisperEN", whisper_en)):
            refs, b3hyps, b4hyps = [], [], []
            for ap_ in common:
                rb = norm(b3_ok[ap_]["ref"])
                hb3 = norm(b3_ok[ap_]["hyp"])
                hb4 = norm(b4_ok[ap_]["hyp"])
                if not rb:
                    continue
                if not hb3 or not hb4:
                    continue
                refs.append(rb); b3hyps.append(hb3); b4hyps.append(hb4)
            if not refs:
                continue
            w_b3 = jiwer.wer(refs, b3hyps)
            w_b4 = jiwer.wer(refs, b4hyps)
            row = {
                "speaker": spk,
                "severity": severity_by_spk.get(spk, "unknown"),
                "n_clips": len(refs),
                "wer_b3": round(w_b3, 4),
                "wer_b4": round(w_b4, 4),
                "delta": round(w_b3 - w_b4, 4),  # negative = B3 better
            }
            (rows_lc if label == "lowercase" else rows_we).append(row)

    # Failure modes (across all speakers / both conditions, on non-error clips)
    failure_modes = {}
    for cond in ("b3", "b4"):
        n_clips = 0
        n_err = 0
        n_loops = 0
        n_empty = 0
        for spk, conds in preds_by_spk.items():
            for p in conds.get(cond, []):
                n_clips += 1
                if p["hyp"].startswith("<ERROR"):
                    n_err += 1
                    continue
                if not p["hyp"].strip():
                    n_empty += 1
                if detect_loop(p["hyp"]):
                    n_loops += 1
        failure_modes[cond] = {
            "n_clips": n_clips,
            "errors": n_err,
            "empty": n_empty,
            "loops": n_loops,
        }

    def summarize(rows, label):
        if not rows:
            return None
        deltas = [r["delta"] for r in rows]
        # Per-severity
        by_sev = defaultdict(list)
        for r in rows:
            by_sev[r["severity"]].append(r["delta"])
        sev_summary = {
            sev: {
                "n_speakers": len(deltas_),
                "mean_delta": round(mean(deltas_), 4),
                "median_delta": round(median(deltas_), 4),
                "n_b3_better": sum(1 for d in deltas_ if d < 0),
                "n_b4_better": sum(1 for d in deltas_ if d > 0),
            }
            for sev, deltas_ in by_sev.items()
        }
        b3_overall = mean(r["wer_b3"] for r in rows)
        b4_overall = mean(r["wer_b4"] for r in rows)
        return {
            "label": label,
            "n_speakers": len(rows),
            "speaker_avg_wer_b3": round(b3_overall, 4),
            "speaker_avg_wer_b4": round(b4_overall, 4),
            "mean_delta": round(mean(deltas), 4),
            "median_delta": round(median(deltas), 4),
            "min_delta": round(min(deltas), 4),
            "max_delta": round(max(deltas), 4),
            "bootstrap_mean_ci": bootstrap_mean_ci(deltas),
            "sign_test": sign_test(deltas),
            "per_severity": sev_summary,
            "rows": rows,
        }

    out = {
        "input_file": args.input,
        "lowercase": summarize(rows_lc, "lowercase"),
        "whisperEN": summarize(rows_we, "whisperEN"),
        "failure_modes": failure_modes,
    }

    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)

    # Print headline
    print(f"\n{'='*78}")
    print(f"  B3 vs B4 — paired SS-FT WER analysis")
    print(f"  ({len(preds_by_spk)} speakers; same per-speaker held-out clips)")
    print(f"{'='*78}")
    for label in ("lowercase", "whisperEN"):
        s = out[label]
        if not s: continue
        print(f"\n  --- {label} ---")
        print(f"    avg per-speaker WER:   B3 = {s['speaker_avg_wer_b3']}   B4 = {s['speaker_avg_wer_b4']}")
        bs = s["bootstrap_mean_ci"]
        st = s["sign_test"]
        print(f"    paired Δ (B3 − B4):    mean={s['mean_delta']:+.4f}  median={s['median_delta']:+.4f}")
        print(f"                           CI95 = [{bs['ci95_lo']:+.4f}, {bs['ci95_hi']:+.4f}]")
        print(f"    sign test:             B3 better on {st['n_b3_better']} speakers, B4 better on {st['n_b4_better']}, tied {st['n_tied']}")
        print(f"                           binomial p-value = {st['p_value']:.4f}" if st['p_value'] is not None else "                           p = NA")
        print(f"    per-severity mean Δ:")
        for sev, v in s["per_severity"].items():
            print(f"      {sev:<10s} n={v['n_speakers']:<3d}  mean Δ {v['mean_delta']:+.4f}  median {v['median_delta']:+.4f}  B3>{v['n_b3_better']}  B4>{v['n_b4_better']}")
    print(f"\n  --- failure modes ---")
    for cond, fm in out["failure_modes"].items():
        print(f"    {cond}: errors={fm['errors']}/{fm['n_clips']}  loops={fm['loops']}  empty={fm['empty']}")
    print(f"\n  saved {args.output}")


if __name__ == "__main__":
    main()

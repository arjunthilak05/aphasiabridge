"""Merge APROCSA + TORGO manifests into one training manifest with balanced weighting.

APROCSA is the primary (aphasia) target; TORGO is atypical-speech-robustness.
We keep all APROCSA clips and downsample TORGO so the ratio is tunable.

Usage:
    python scripts/build_mixed_manifest.py \
        --aprocsa data/aprocsa_manifest.jsonl \
        --torgo   data/torgo_manifest.jsonl \
        --out     data/mixed_aprocsa_torgo.jsonl \
        --torgo-ratio 0.5   # torgo records = 0.5 x aprocsa records
"""
from __future__ import annotations
import argparse, json, os, random

def load(p):
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aprocsa", required=True)
    ap.add_argument("--torgo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--torgo-ratio", type=float, default=0.5,
                    help="TORGO size as fraction of APROCSA size")
    ap.add_argument("--drop-torgo-controls", action="store_true",
                    help="Drop TORGO control (healthy) speakers — keep only dysarthric")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    apr = load(args.aprocsa)
    tor = load(args.torgo)
    if args.drop_torgo_controls:
        tor = [r for r in tor if r.get("severity") != "control"]

    rng = random.Random(args.seed)
    rng.shuffle(tor)
    n_torgo = int(len(apr) * args.torgo_ratio)
    tor = tor[:n_torgo]

    merged = apr + tor
    rng.shuffle(merged)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in merged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    by_src = Counter(r.get("source") for r in merged)
    print(f"Total: {len(merged)}  (APROCSA={len(apr)}  TORGO sampled={len(tor)})")
    print(f"By source: {dict(by_src)}")
    print(f"Wrote: {args.out}")

if __name__ == "__main__":
    main()

"""Build the paper-recipe dataset (Jiang et al. 2026, arXiv:2603.15261).

Pipeline:
  1. Load v2 manifests (already IPA-stripped + [: target] substituted + 0word removed).
  2. Filter to utterances with [*p:] (pronunciation paraphasia) AND NOT [*s:] (semantic).
  3. Restrict to PAR participant lines only (drop INV interviewer lines).
  4. Speaker-disjoint split: top-10% speakers by total duration → SS group, rest → SI.
  5. SS speakers → 8:1:1 train/val/test per speaker.
  6. SI speakers → all utterances → SI training set; carve a small per-speaker val for
     early-stopping.

Outputs (written to APHASIA_ROOT):
  - aphasiabank_si_train.jsonl       (SI-FT training, paper's B2 input)
  - aphasiabank_si_val.jsonl         (early-stop val, drawn from SI speakers)
  - aphasiabank_ss_test.jsonl        (held-out for evaluation = paper's SS test)
  - aphasiabank_ss_train.jsonl       (per-speaker train, for future B3 personalization)
  - aphasiabank_ss_val.jsonl         (per-speaker val)
  - aphasiabank_paper_dataset.json   (manifest stats and provenance)
"""
from __future__ import annotations
import argparse, json, os, random, re
from collections import defaultdict


P_TAG = re.compile(r"\[\*\s*p[:\s]")     # pronunciation paraphasia
S_TAG = re.compile(r"\[\*\s*s[:\s]")     # semantic paraphasia


def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(p, rows):
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def keep_clip(r, protocol_only: bool = True):
    raw = r.get("transcript_raw") or ""
    if not P_TAG.search(raw):
        return False
    if S_TAG.search(raw):
        return False
    # Paper uses "protocol-compliant English subset only"
    if protocol_only and (r.get("folder") or "") != "Protocol":
        return False
    # PAR-only: skip if condition explicitly marks INV
    cond = (r.get("condition") or "").lower()
    if cond == "inv":
        return False
    # Drop very short / very long clips (alignment / OOM safety)
    dur_ms = r.get("duration_ms") or 0
    if dur_ms < 500 or dur_ms > 30000:
        return False
    if (r.get("word_count") or 0) < 2:
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("APHASIA_ROOT"))
    ap.add_argument("--ss-frac", type=float, default=0.10,
                    help="Top fraction of speakers (by duration) reserved as SS group (paper: 0.10)")
    ap.add_argument("--min-speaker-seconds", type=float, default=20.0,
                    help="Drop speakers whose total filtered audio is below this (paper had densely-sampled speakers)")
    ap.add_argument("--si-val-frac", type=float, default=0.05,
                    help="Fraction of SI clips reserved for early-stop val (random, NOT per-speaker)")
    ap.add_argument("--protocol-only", action="store_true", default=True,
                    help="Restrict to Protocol corpora only (matches paper)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if not args.root:
        raise SystemExit("APHASIA_ROOT not set")

    rng = random.Random(args.seed)

    # ----- Step 1: load all v2 splits and merge (we re-split by speaker) -----
    print("[1/6] Loading v2 manifests")
    rows = []
    for sp in ("train", "val", "test"):
        p = os.path.join(args.root, f"aphasiabank_{sp}_v2.jsonl")
        if os.path.exists(p):
            n_before = len(rows)
            rows.extend(load_jsonl(p))
            print(f"     {sp}: +{len(rows)-n_before} (total {len(rows)})")

    # ----- Step 2: paper filter ([*p:] AND NOT [*s:], Protocol-only, PAR only, length sane) -----
    print(f"[2/6] Filtering: [*p:] AND NOT [*s:], Protocol-only={args.protocol_only}, PAR-only, dur 0.5-30s, words >=2")
    kept = [r for r in rows if keep_clip(r, protocol_only=args.protocol_only)]
    dropped = len(rows) - len(kept)
    print(f"     kept: {len(kept):,} / {len(rows):,}  ({len(kept)/len(rows)*100:.2f}%)  dropped: {dropped:,}")
    if not kept:
        raise SystemExit("No clips survived filter — check transcript_raw markers")

    # ----- Step 3: aggregate per-speaker durations + min-speaker-seconds filter -----
    print("[3/6] Aggregating per-speaker durations + min-speaker-seconds filter")
    by_speaker = defaultdict(list)
    for r in kept:
        by_speaker[r["participant_id"]].append(r)
    speaker_durations = {pid: sum(r.get("duration_ms", 0) for r in clips) / 1000.0
                         for pid, clips in by_speaker.items()}
    n_before = len(by_speaker)
    sparse = [pid for pid, d in speaker_durations.items() if d < args.min_speaker_seconds]
    for pid in sparse:
        del by_speaker[pid]
        del speaker_durations[pid]
    print(f"     dropped {len(sparse):,} speakers under {args.min_speaker_seconds:.0f}s   ({n_before:,} → {len(by_speaker):,})")
    sorted_spk = sorted(speaker_durations.items(), key=lambda x: -x[1])
    total_h = sum(speaker_durations.values()) / 3600.0
    print(f"     remaining speakers: {len(by_speaker):,}    filtered audio total: {total_h:.2f} h")
    print(f"     longest 5: " + ", ".join(f"{p}={d:.0f}s" for p,d in sorted_spk[:5]))

    # ----- Step 4: top-10% speakers -> SS group -----
    n_ss = max(1, int(round(len(sorted_spk) * args.ss_frac)))
    ss_speakers = {pid for pid, _ in sorted_spk[:n_ss]}
    si_speakers = {pid for pid, _ in sorted_spk[n_ss:]}
    si_h = sum(speaker_durations[p] for p in si_speakers) / 3600.0
    ss_h = sum(speaker_durations[p] for p in ss_speakers) / 3600.0
    print(f"[4/6] Speaker-disjoint split (top-{args.ss_frac*100:.0f}% by duration)")
    print(f"     SS speakers: {len(ss_speakers):3d}  ({ss_h:.2f} h)")
    print(f"     SI speakers: {len(si_speakers):3d}  ({si_h:.2f} h)")

    # ----- Step 5: SS speakers -> per-speaker 8:1:1 -----
    print("[5/6] Per-speaker 8:1:1 split for SS group")
    ss_train = []; ss_val = []; ss_test = []
    for pid in ss_speakers:
        clips = by_speaker[pid][:]
        rng.shuffle(clips)
        n = len(clips)
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        ss_train.extend(clips[:n_train])
        ss_val.extend(clips[n_train:n_train+n_val])
        ss_test.extend(clips[n_train+n_val:])
    print(f"     ss_train: {len(ss_train):,}   ss_val: {len(ss_val):,}   ss_test: {len(ss_test):,}")

    # ----- Step 6: SI training set + small randomly-sampled val -----
    print("[6/6] SI training set (all SI-speaker clips) + random fractional val")
    si_pool = [c for pid in si_speakers for c in by_speaker[pid]]
    rng.shuffle(si_pool)
    n_val = max(50, int(len(si_pool) * args.si_val_frac))
    si_val = si_pool[:n_val]
    si_train = si_pool[n_val:]
    si_train_h = sum(r.get("duration_ms",0) for r in si_train) / 1000 / 3600
    si_val_h   = sum(r.get("duration_ms",0) for r in si_val)   / 1000 / 3600
    ss_test_h  = sum(r.get("duration_ms",0) for r in ss_test)  / 1000 / 3600
    print(f"     si_train: {len(si_train):,}  ({si_train_h:.2f} h)   si_val: {len(si_val):,}  ({si_val_h:.2f} h)")
    print(f"     ss_test:  {len(ss_test):,}  ({ss_test_h:.2f} h)")

    # Sanity: paper's SI-FT data was 7.67 h on 456 speakers
    print(f"\n     PAPER REFERENCE: 456 speakers / 7.67 h SI-FT data")
    print(f"     OURS:            {len(si_speakers)} speakers / {si_train_h:.2f} h SI-FT data")

    # ----- Write -----
    out_paths = {}
    for name, data in [
        ("si_train",   si_train),
        ("si_val",     si_val),
        ("ss_train",   ss_train),
        ("ss_val",     ss_val),
        ("ss_test",    ss_test),
    ]:
        p = os.path.join(args.root, f"aphasiabank_{name}.jsonl")
        write_jsonl(p, data)
        out_paths[name] = p
        print(f"     wrote {p}  ({len(data):,} rows)")

    # Manifest of manifests
    summary = {
        "source": "Jiang et al. 2026 (arXiv:2603.15261) Section 2.2 recipe",
        "filter": "[*p:] AND NOT [*s:], PAR-only, 0.5-30s clips, word_count >= 2",
        "ss_fraction_top_speakers": args.ss_frac,
        "speakers": {
            "ss_count": len(ss_speakers), "si_count": len(si_speakers),
            "ss_total_hours": round(ss_h, 3), "si_total_hours": round(si_h, 3),
        },
        "splits": {
            "si_train": {"clips": len(si_train), "hours": round(si_train_h, 3)},
            "si_val":   {"clips": len(si_val),   "hours": round(si_val_h, 3)},
            "ss_train": {"clips": len(ss_train)},
            "ss_val":   {"clips": len(ss_val)},
            "ss_test":  {"clips": len(ss_test),  "hours": round(ss_test_h, 3)},
        },
        "paths": out_paths,
        "paper_reference": {"speakers": 456, "hours": 7.67},
        "seed": args.seed,
    }
    summary_path = os.path.join(args.root, "aphasiabank_paper_dataset.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\n     wrote {summary_path}")


if __name__ == "__main__":
    main()

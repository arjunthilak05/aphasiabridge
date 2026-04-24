"""Build a balanced, speaker-diverse training subsample from aphasiabank_train.jsonl.

Strategy:
  - Cap to N clips per speaker for diversity.
  - Stratify by severity bucket, with target totals per bucket.
  - Prefer clips with non-trivial word count (>=2 words) to avoid one-word filler clips.

Default target: 40K clips ~40 h audio, fits in RAM at float32.
"""
from __future__ import annotations
import argparse, json, random
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  default=r"D:\Downloads\aphasiabank\aphasiabank_train.jsonl")
    ap.add_argument("--output", default=r"D:\Downloads\aphasiabank\aphasiabank_train_40k.jsonl")
    ap.add_argument("--per-speaker-cap", type=int, default=30)
    ap.add_argument("--target", type=int, default=40000)
    ap.add_argument("--min-words", type=int, default=2,
                    help="drop clips with transcript word_count below this")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--severity-only", choices=["mild", "moderate", "severe", "very_severe"],
                    default=None, help="Restrict sampling to a single severity bucket (R3 curriculum)")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.input, encoding="utf-8") if l.strip()]
    print(f"input: {len(rows)} clips")

    rows = [r for r in rows if (r.get("word_count") or 0) >= args.min_words]
    print(f"after word_count>={args.min_words}: {len(rows)}")

    # Group by (severity_bucket, speaker) for diversity cap
    def bucket(r):
        sev = r.get("severity") or "unlabeled"
        return sev

    by_sev = defaultdict(list)
    for r in rows:
        by_sev[bucket(r)].append(r)
    for k, v in sorted(by_sev.items()):
        print(f"  bucket {k!r}: {len(v)} clips")

    # Target proportions per bucket (boost hard severity, keep some controls)
    if args.severity_only:
        targets = {k: 0 for k in ("mild", "moderate", "severe", "very_severe", "unlabeled")}
        targets[args.severity_only] = args.target
    else:
        targets = {
            "mild":         12000,
            "moderate":     10000,
            "severe":       10000,
            "very_severe":   8000,
            "unlabeled":     0,   # skip — mostly controls + unscored
        }
    # Scale to requested total
    s = sum(targets.values())
    if s != args.target:
        factor = args.target / s
        targets = {k: int(v * factor) for k, v in targets.items()}

    rng = random.Random(args.seed)
    chosen = []
    for sev, target in targets.items():
        if target <= 0 or sev not in by_sev:
            continue
        pool = by_sev[sev][:]
        rng.shuffle(pool)
        by_speaker: dict[str, list] = defaultdict(list)
        for r in pool:
            by_speaker[r["participant_id"]].append(r)
        # Round-robin: iterate speakers in cycles, pick one clip each pass,
        # track per-speaker counts for cap.  Filter speakers with remaining quota
        # at the start of every cycle — no delete-while-iterate.
        bucket_rows = []
        speaker_count: dict[str, int] = defaultdict(int)
        speakers = list(by_speaker.keys())
        rng.shuffle(speakers)
        while len(bucket_rows) < target:
            active = [pid for pid in speakers
                      if by_speaker[pid] and speaker_count[pid] < args.per_speaker_cap]
            if not active:
                break
            for pid in active:
                if len(bucket_rows) >= target:
                    break
                if speaker_count[pid] >= args.per_speaker_cap:
                    continue
                if not by_speaker[pid]:
                    continue
                bucket_rows.append(by_speaker[pid].pop(0))
                speaker_count[pid] += 1
        print(f"  {sev!r}: kept {len(bucket_rows)} (target {target}, "
              f"{len(speaker_count)} speakers)")
        chosen.extend(bucket_rows)

    rng.shuffle(chosen)
    out = Path(args.output)
    with open(out, "w", encoding="utf-8") as f:
        for r in chosen:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total_h = sum(r["duration_ms"] for r in chosen) / 1000 / 3600
    speakers = {r["participant_id"] for r in chosen}
    print(f"\ntotal kept: {len(chosen)} clips  /  {len(speakers)} speakers  /  {total_h:.1f} h audio")
    print(f"wrote: {out}")


if __name__ == "__main__":
    main()

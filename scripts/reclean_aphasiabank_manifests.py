"""Re-run the improved `strip_chat_annotation` over transcript_raw in existing manifests
(no WAV re-slicing). Outputs v2 jsonl files with cleaner transcripts.

Also applies the final ASCII-only + punctuation-normalization pass from
clean_aphasiabank_manifests.py so the output is immediately trainable.
"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_aphasiabank_manifest import strip_chat_annotation  # improved version

PUNCT_RE = re.compile(r"[^a-z' ]")
WS_RE    = re.compile(r"\s+")


def has_bad_marker(t: str) -> bool:
    t_low = t.lower()
    return ("xxx" in t_low or " yyy" in t_low or t_low.startswith("yyy")
            or "www" in t_low)


def normalize(t: str) -> str:
    t = t.lower()
    t = PUNCT_RE.sub(" ", t)
    t = WS_RE.sub(" ", t).strip()
    return t


def process(src: Path, dst: Path) -> tuple[int, int, int]:
    total = kept = dropped_ascii = 0
    with open(src, encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip(): continue
            total += 1
            r = json.loads(line)
            raw = r.get("transcript_raw") or r.get("transcript", "")
            # Re-clean from raw CHA-body
            cleaned = strip_chat_annotation(raw)
            if has_bad_marker(cleaned):
                continue
            # Strip non-ASCII (IPA etc.)
            if any(ord(c) > 127 for c in cleaned):
                dropped_ascii += 1
                continue
            norm = normalize(cleaned)
            if len(norm.split()) < 2:
                continue
            r["transcript_v2"]      = norm
            r["transcript_clean"]   = norm
            r["transcript"]         = norm
            r["word_count"]         = len(norm.split())
            fout.write(json.dumps(r, ensure_ascii=True) + "\n")
            kept += 1
    return total, kept, dropped_ascii


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(r"D:\Downloads\aphasiabank"))
    args = ap.parse_args()

    for name in ("train", "val", "test"):
        src = args.root / f"aphasiabank_{name}.jsonl"
        dst = args.root / f"aphasiabank_{name}_v2.jsonl"
        if not src.exists():
            print(f"skip: {src} not found"); continue
        total, kept, dropped = process(src, dst)
        print(f"{name:5s}: {total:7d} -> {kept:7d} kept  (dropped non-ASCII {dropped})  "
              f"-> {dst.name}")


if __name__ == "__main__":
    main()

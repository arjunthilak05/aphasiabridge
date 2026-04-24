"""Post-process AphasiaBank manifests to produce ASR-clean transcripts.

Filters:
  - drop rows whose transcript contains any non-ASCII char (IPA phonetics)
  - drop rows with CHAT unintelligible markers (xxx/yyy/www) or quote continuations (+")
  - drop rows with < 2 clean words

Normalizes:
  - lowercase
  - strip all punctuation except apostrophe
  - collapse whitespace

Writes new JSONLs alongside originals with `_clean` suffix.
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path

PUNCT_RE = re.compile(r"[^a-z' ]")
WS_RE    = re.compile(r"\s+")


def has_bad_marker(t: str) -> bool:
    t_low = t.lower()
    return ("xxx" in t_low or " yyy" in t_low or t_low.startswith("yyy")
            or "www" in t_low or '+"' in t or "+<" in t)


def normalize(t: str) -> str:
    t = t.lower()
    t = PUNCT_RE.sub(" ", t)
    t = WS_RE.sub(" ", t).strip()
    return t


def is_clean(t: str) -> bool:
    if any(ord(c) > 127 for c in t):
        return False
    if has_bad_marker(t):
        return False
    return True


def process(path: Path, out_path: Path) -> tuple[int, int]:
    kept = 0; total = 0
    with open(path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            total += 1
            r = json.loads(line)
            raw_t = r.get("transcript", "")
            if not is_clean(raw_t):
                continue
            norm = normalize(raw_t)
            if len(norm.split()) < 2:
                continue
            r["transcript_raw_cha"] = r.get("transcript_raw", raw_t)  # keep original
            r["transcript"]         = norm
            r["transcript_clean"]   = norm
            r["word_count"]         = len(norm.split())
            fout.write(json.dumps(r, ensure_ascii=True) + "\n")
            kept += 1
    return total, kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(r"D:\Downloads\aphasiabank"))
    args = ap.parse_args()

    for name in ("train", "val", "test"):
        src = args.root / f"aphasiabank_{name}.jsonl"
        dst = args.root / f"aphasiabank_{name}_clean.jsonl"
        if not src.exists():
            print(f"skip (missing): {src}")
            continue
        total, kept = process(src, dst)
        print(f"{name:5s}: {total:7d} -> {kept:7d} kept ({kept/total*100:.1f}%)  -> {dst.name}")


if __name__ == "__main__":
    main()

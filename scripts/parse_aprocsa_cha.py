"""Parse APROCSA CHAT (.cha) transcripts into a clean JSONL manifest.

One record per PAR (participant = aphasic speaker) utterance:
  {participant_id, utt_index, start_ms, end_ms, raw, clean, word_count, has_para, has_filler}

Usage:
    python scripts/parse_aprocsa_cha.py \
        --cha-dir "D:/Downloads/speech_data/aprocsa" \
        --out     "data/aprocsa_par_utts.jsonl"
"""
from __future__ import annotations
import argparse, json, os, re, sys, glob

PAR_TIER_RE = re.compile(r"^\*PAR:\s*(.*)$")
CONT_TIER_RE = re.compile(r"^\t(?!%)(.*)$")
# CHAT timing markers use NAK (\x15) as delimiter: "text\x15<start>_<end>\x15"
TIME_RE = re.compile(r"\x15(\d+)_(\d+)\x15")

PARAPHASIA_MARKERS = ("@u", "@q", "@n", "[:", "[*", "[/")
FILLER_RE = re.compile(r"&-(?:um|uh|er|mm|eh)\b", re.IGNORECASE)


def strip_chat_annotation(text: str) -> str:
    text = TIME_RE.sub("", text)
    text = re.sub(r"\[[^\]]*\]", "", text)
    text = re.sub(r"&=[^\s]+", "", text)
    text = re.sub(r"&-[^\s]+", "", text)
    text = re.sub(r"&\+[^\s]+", "", text)
    text = re.sub(r"\(\.{1,}\)", "", text)
    text = re.sub(r"[‡↑↓↗↘]", "", text)
    text = re.sub(r"\+<", "", text)
    text = re.sub(r"\+\.{2,}", "", text)
    text = re.sub(r"\+/\.", ".", text)
    text = re.sub(r"\+/\?", "?", text)
    text = re.sub(r"<([^<>]+)>", r"\1", text)
    text = re.sub(r"@[a-z:]+", "", text)
    text = re.sub(r"\(([a-z']+)\)", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip(" .?!,")
    return text


def extract_times(raw: str):
    m = TIME_RE.search(raw)
    if not m:
        return None, None, raw
    start, end = int(m.group(1)), int(m.group(2))
    body = (raw[: m.start()] + raw[m.end():]).strip()
    return start, end, body


def iter_par_utterances(cha_path: str):
    with open(cha_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    buf = None
    for ln in lines:
        ln = ln.rstrip("\n")
        m = PAR_TIER_RE.match(ln)
        if m:
            if buf is not None:
                yield buf
            buf = m.group(1)
            continue
        m2 = CONT_TIER_RE.match(ln)
        if m2 and buf is not None and not ln.lstrip().startswith("%"):
            buf += " " + m2.group(1).strip()
            continue
        if ln.startswith(("*", "%", "@")):
            if buf is not None:
                yield buf
                buf = None
    if buf is not None:
        yield buf


def parse_file(cha_path: str):
    pid = os.path.basename(cha_path).split(".")[0]
    out = []
    for i, raw in enumerate(iter_par_utterances(cha_path)):
        start_ms, end_ms, body = extract_times(raw)
        clean = strip_chat_annotation(body)
        if not clean:
            continue
        out.append({
            "participant_id": pid,
            "utt_index": i,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "duration_ms": (end_ms - start_ms) if (start_ms is not None and end_ms is not None) else None,
            "raw": body,
            "clean": clean,
            "word_count": len(clean.split()),
            "has_paraphasia_marker": any(m in body for m in PARAPHASIA_MARKERS),
            "has_filler": bool(FILLER_RE.search(body)),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cha-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    all_records = []
    for cha in sorted(glob.glob(os.path.join(args.cha_dir, "aprocsa*.cha"))):
        recs = parse_file(cha)
        all_records.extend(recs)
        words = sum(r["word_count"] for r in recs)
        print(f"{os.path.basename(cha):25s}  utts={len(recs):4d}  words={words:5d}", file=sys.stderr)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total_words = sum(r["word_count"] for r in all_records)
    total_dur = sum(r["duration_ms"] or 0 for r in all_records) / 1000
    print(f"\nTOTAL: {len(all_records)} utterances, {total_words} words, {total_dur:.1f} s audio", file=sys.stderr)
    print(f"Wrote: {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()

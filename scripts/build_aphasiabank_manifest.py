"""Build AphasiaBank training dataset: per-utterance clips + JSONL manifest.

Inputs:
  D:/Downloads/aphasiabank/cha/<Folder>/<Corpus>/.../*.cha   (transcripts)
  D:/Downloads/aphasiabank/wav/<Folder>/<Corpus>/.../*.wav   (16 kHz mono sessions)
  D:/Downloads/aphasiabank/catalog.csv                        (severity + AQ + media map)

Outputs:
  D:/Downloads/aphasiabank/clips/<corpus>_<sess>_<utt_idx:04d>.wav    (sliced clips)
  D:/Downloads/aphasiabank/aphasiabank_manifest.jsonl                 (one line per clip)
  D:/Downloads/aphasiabank/aphasiabank_train.jsonl   (speaker-independent split)
  D:/Downloads/aphasiabank/aphasiabank_val.jsonl
  D:/Downloads/aphasiabank/aphasiabank_test.jsonl

Each manifest row matches the APROCSA schema so the existing trainer picks it up:
  {clip_path, participant_id, utt_index, start_ms, end_ms, duration_ms,
   transcript_clean, transcript_raw, has_paraphasia_marker, has_filler,
   condition, source, severity, qab_overall, corpus, folder, aphasia_type,
   audio_path, transcript}
"""
from __future__ import annotations
import argparse, csv, json, os, random, re, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

# ---- CHAT parsing (lifted from parse_aprocsa_cha.py, same rules) ----

PAR_TIER_RE  = re.compile(r"^\*PAR:\s*(.*)$")
CONT_TIER_RE = re.compile(r"^\t(?!%)(.*)$")
TIME_RE      = re.compile(r"\x15(\d+)_(\d+)\x15")

PARAPHASIA_MARKERS = ("@u", "@q", "@n", "[:", "[*", "[/")
FILLER_RE = re.compile(r"&-(?:um|uh|er|mm|eh)\b", re.IGNORECASE)


def strip_chat_annotation(text: str) -> str:
    """Convert CHAT-format utterance body to ASR-style text.

    Order matters. [: target] substitution and explicit-marker stripping must run
    before the generic bracket strip, otherwise we lose the target word.
    """
    # Remove timing markers \x15<start>_<end>\x15
    text = TIME_RE.sub("", text)

    # [: target] — replace the preceding surface form with the intended target word(s).
    # e.g. "haf [: half]" -> "half", "gonna [: going to]" -> "going to".
    # This is what published AphasiaBank ASR benchmarks use as the ground-truth label.
    text = re.sub(r"\S+\s*\[:\s*([^\]]+)\]", r"\1", text)

    # Error codes like [* pw], [* s], [* n:k], [+ gram]: drop them (informational annotations).
    text = re.sub(r"\[\*[^\]]*\]", "", text)
    text = re.sub(r"\[\+[^\]]*\]", "", text)

    # Retrace / repetition markers [/], [//], [///], [x N]: drop. The repeated words stay.
    text = re.sub(r"\[/{1,3}\]", "", text)
    text = re.sub(r"\[x\s*\d+\]", "", text)

    # Any remaining bracketed annotation (percent tiers, comments)
    text = re.sub(r"\[[^\]]*\]", "", text)

    # Paralinguistic events &=laugh / &=throat:clears
    text = re.sub(r"&=\S+", "", text)
    # Fillers and interruptions  &-uh / &+bot / &~
    text = re.sub(r"&[-+~]\S+", "", text)
    # Bare & (code-switch) and the & marker itself at word boundary
    text = re.sub(r"(?:^|\s)&\S*", " ", text)

    # Pauses (.), (..), (...)
    text = re.sub(r"\(\.+\)", "", text)
    # Lexical marks (.), prosody arrows, turn markers
    text = re.sub(r"[‡↑↓↗↘]", "", text)

    # Utterance-continuation / quotation markers:
    #   +" quoted speech, +, self-completion, +< latching, +^ fast-start,
    #   +// interruption, +/. +/? trailing-off
    text = re.sub(r'\+"', "", text)
    text = re.sub(r"\+<", "", text)
    text = re.sub(r"\+,", "", text)
    text = re.sub(r"\+\^", "", text)
    text = re.sub(r"\+\.{2,}", "", text)
    text = re.sub(r"\+//?", "", text)
    text = re.sub(r"\+/\.", ".", text)
    text = re.sub(r"\+/\?", "?", text)

    # Reformulation brackets <text> — keep the text inside.
    text = re.sub(r"<([^<>]+)>", r"\1", text)

    # Special-form markers on words: word@u (unintelligible), word@q (quotation),
    # word@n (novel), word@wp (word play), etc.  Drop the @tag, keep the word.
    text = re.sub(r"@[a-z]+(?::[a-z]+)?", "", text)

    # Optional letters: (g)onna -> gonna, (be)cause -> because
    text = re.sub(r"\(([a-z']+)\)", r"\1", text)

    # CHAT "null morpheme" markers: 0det, 0pres, 0pl, etc. — grammatical elements the
    # speaker omitted. Drop entirely, they are NOT spoken.
    text = re.sub(r"\b0[a-z]+\b", "", text)
    # Bare 0 as a placeholder.
    text = re.sub(r"(?:^|\s)0(?:\s|$)", " ", text)

    # Collapse whitespace and strip sentence punctuation around edges
    text = re.sub(r"\s+", " ", text).strip(" .?!,")
    return text


def extract_times(raw: str):
    m = TIME_RE.search(raw)
    if not m:
        return None, None, raw
    start, end = int(m.group(1)), int(m.group(2))
    body = (raw[: m.start()] + raw[m.end():]).strip()
    return start, end, body


def iter_par_utterances(cha_path: Path):
    with open(cha_path, "r", encoding="utf-8", errors="replace") as f:
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


# ---- Severity derivation (from WAB-AQ) ----

def derive_severity(aq):
    if aq is None or aq == "":
        return None
    try:
        aq = float(aq)
    except (TypeError, ValueError):
        return None
    if aq >= 76: return "mild"
    if aq >= 51: return "moderate"
    if aq >= 26: return "severe"
    return "very_severe"


# ---- Clip slicing ----

MIN_DUR_MS = 300
MAX_DUR_MS = 30_000     # Gemma 4 E4B audio encoder context cap
MIN_WORDS  = 1


def participant_id_for(row: dict) -> str:
    """Unique speaker proxy: <corpus>_<session_stem>.
    Works well because AphasiaBank stores one speaker per session file across most corpora."""
    stem = Path(row["file"]).stem
    return f"{row['corpus']}_{stem}"


def wav_path_for(row: dict, wav_root: Path) -> Path | None:
    """Map a catalog row to its extracted WAV path (same layout as download_aphasiabank_media.py)."""
    if not row.get("media_url"):
        return None
    # media_url: https://media.talkbank.org/aphasia/English/<tail>.mp4
    media_url = row["media_url"]
    marker = "/aphasia/English/"
    if marker not in media_url:
        return None
    tail = media_url.split(marker, 1)[1]
    # Strip extension, append .wav
    stem = tail.rsplit(".", 1)[0]
    return wav_root / (stem + ".wav")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",   type=Path, default=Path(r"D:\Downloads\aphasiabank"))
    ap.add_argument("--clips-dir", type=Path, default=None,
                    help="where per-utterance WAVs go (default: <root>/clips)")
    ap.add_argument("--manifest", type=Path, default=None,
                    help="output manifest JSONL (default: <root>/aphasiabank_manifest.jsonl)")
    ap.add_argument("--limit-sessions", type=int, default=0,
                    help="stop after N sessions (for smoke testing)")
    ap.add_argument("--corpora", nargs="*", help="restrict to these corpora")
    ap.add_argument("--folders", nargs="*", help="restrict to these folders")
    ap.add_argument("--split", action="store_true",
                    help="also write speaker-independent train/val/test jsonl")
    ap.add_argument("--val-frac",  type=float, default=0.10)
    ap.add_argument("--test-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    clips_dir = args.clips_dir or (args.root / "clips")
    manifest_path = args.manifest or (args.root / "aphasiabank_manifest.jsonl")
    catalog_path  = args.root / "catalog.csv"
    wav_root      = args.root / "wav"

    if not catalog_path.exists():
        sys.exit(f"missing catalog: {catalog_path}")

    clips_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(open(catalog_path, encoding="utf-8")))
    if args.folders: rows = [r for r in rows if r["folder"] in args.folders]
    if args.corpora: rows = [r for r in rows if r["corpus"] in args.corpora]
    # Only sessions with media (we have WAV for these)
    rows = [r for r in rows if r.get("media_url")]

    print(f"[plan] {len(rows)} sessions in catalog (after filters)",
          file=sys.stderr, flush=True)

    kept = dropped_short = dropped_long = dropped_missing_time = 0
    dropped_no_wav = dropped_empty_text = 0
    dropped_clip_oob = 0
    all_rows: list[dict] = []

    for sidx, row in enumerate(rows, 1):
        if args.limit_sessions and sidx > args.limit_sessions:
            break

        wav_path = wav_path_for(row, wav_root)
        if wav_path is None or not wav_path.exists() or wav_path.stat().st_size == 0:
            dropped_no_wav += 1
            continue

        cha_path = Path(row["cha_path"])
        if not cha_path.exists():
            dropped_no_wav += 1
            continue

        # Load session audio once
        try:
            pcm, sr = sf.read(str(wav_path), dtype="int16")
        except Exception as e:
            print(f"[warn] {wav_path}: cannot read ({e})", file=sys.stderr)
            dropped_no_wav += 1
            continue
        if pcm.ndim > 1:
            pcm = pcm.mean(axis=1).astype(np.int16)
        if sr != 16000:
            print(f"[warn] {wav_path}: sr={sr}, expected 16000", file=sys.stderr)

        pid     = participant_id_for(row)
        sev     = row.get("severity") or derive_severity(row.get("par_wab_aq"))
        try:
            qab = float(row.get("par_wab_aq")) if row.get("par_wab_aq") else None
        except (TypeError, ValueError):
            qab = None
        corpus  = row["corpus"]
        folder  = row["folder"]
        ap_type = row.get("par_type") or None

        utts_written = 0
        for utt_index, raw in enumerate(iter_par_utterances(cha_path)):
            start_ms, end_ms, body = extract_times(raw)
            if start_ms is None or end_ms is None:
                dropped_missing_time += 1
                continue
            dur = end_ms - start_ms
            if dur < MIN_DUR_MS:
                dropped_short += 1
                continue
            if dur > MAX_DUR_MS:
                dropped_long += 1
                continue

            clean = strip_chat_annotation(body)
            if not clean:
                dropped_empty_text += 1
                continue
            wc = len(clean.split())
            if wc < MIN_WORDS:
                dropped_empty_text += 1
                continue

            s = int(start_ms * sr / 1000)
            e = int(end_ms   * sr / 1000)
            e = min(e, pcm.shape[0])
            if e <= s or s < 0:
                dropped_clip_oob += 1
                continue
            clip = pcm[s:e]

            clip_name = f"{corpus}_{Path(row['file']).stem}_{utt_index:04d}.wav"
            clip_path = clips_dir / clip_name
            sf.write(str(clip_path), clip, sr, subtype="PCM_16")

            rec = {
                "clip_path": str(clip_path.resolve()),
                "audio_path": str(clip_path.resolve()),
                "participant_id": pid,
                "utt_index": utt_index,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "duration_ms": dur,
                "transcript_clean": clean,
                "transcript": clean,
                "transcript_raw": body,
                "word_count": wc,
                "has_paraphasia_marker": any(m in body for m in PARAPHASIA_MARKERS),
                "has_filler": bool(FILLER_RE.search(body)),
                "condition": "control" if (ap_type and ap_type.lower().startswith("control")) else "aphasia",
                "source": "aphasiabank",
                "severity": sev,
                "qab_overall": qab,
                "corpus": corpus,
                "folder": folder,
                "aphasia_type": ap_type,
                "session_wav": str(wav_path.resolve()),
            }
            all_rows.append(rec)
            utts_written += 1
            kept += 1

        if sidx % 50 == 0 or sidx == len(rows):
            print(f"[build] {sidx}/{len(rows)}  kept={kept}  "
                  f"no_wav={dropped_no_wav}  no_time={dropped_missing_time}  "
                  f"short={dropped_short}  long={dropped_long}  empty={dropped_empty_text}",
                  file=sys.stderr, flush=True)

    # Write unified manifest
    with open(manifest_path, "w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[done] clips={kept}  manifest={manifest_path}", file=sys.stderr, flush=True)
    print(f"[done] dropped: no_wav={dropped_no_wav}  no_time={dropped_missing_time}  "
          f"short={dropped_short}  long={dropped_long}  empty={dropped_empty_text}  "
          f"oob={dropped_clip_oob}", file=sys.stderr, flush=True)

    # Optional: speaker-independent splits, stratified by severity
    if args.split and all_rows:
        rng = random.Random(args.seed)
        # group rows by participant
        by_pid: dict[str, list[dict]] = defaultdict(list)
        for r in all_rows:
            by_pid[r["participant_id"]].append(r)
        # severity key per speaker (take majority or first non-empty)
        speaker_sev: dict[str, str] = {}
        for pid, recs in by_pid.items():
            sevs = [x["severity"] for x in recs if x["severity"]]
            speaker_sev[pid] = sevs[0] if sevs else "unknown"
        # stratified shuffle: bucket speakers by severity, then randomly assign
        buckets: dict[str, list[str]] = defaultdict(list)
        for pid, sv in speaker_sev.items():
            buckets[sv].append(pid)
        train_pids, val_pids, test_pids = set(), set(), set()
        for sv, pids in buckets.items():
            rng.shuffle(pids)
            n = len(pids)
            n_test = max(1, int(round(n * args.test_frac)))  if n >= 3 else 0
            n_val  = max(1, int(round(n * args.val_frac)))   if n >= 3 else 0
            test_pids.update(pids[:n_test])
            val_pids.update(pids[n_test:n_test+n_val])
            train_pids.update(pids[n_test+n_val:])

        splits = {"train": [], "val": [], "test": []}
        for r in all_rows:
            pid = r["participant_id"]
            if pid in train_pids: splits["train"].append(r)
            elif pid in val_pids:  splits["val"].append(r)
            else:                  splits["test"].append(r)

        for name, recs in splits.items():
            p = args.root / f"aphasiabank_{name}.jsonl"
            with open(p, "w", encoding="utf-8") as f:
                for r in recs:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            hrs = sum(r["duration_ms"] for r in recs) / 1000 / 3600
            spks = len({r["participant_id"] for r in recs})
            print(f"[split] {name}: {len(recs)} clips, {spks} speakers, {hrs:.1f} h -> {p}",
                  file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()

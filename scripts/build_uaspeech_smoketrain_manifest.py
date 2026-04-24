"""Build a small UASpeech training manifest for a pipeline smoke-test training run.

We use digits D1..D9 (D10 skipped — not at B1) with labels discovered from our
zero-shot baseline on controls: D1="one" ... D9="nine". For each dysarthric
speaker (where we have data), we keep a handful of clips per digit across mics.

Writes `data/uaspeech_smoketrain_manifest.jsonl` with records:
  {audio_path, transcript, speaker, severity, source, condition, word_id}

Usage:
    .venv/Scripts/python.exe scripts/build_uaspeech_smoketrain_manifest.py
"""
from __future__ import annotations
import os, glob, json, random

ROOT = r"D:\Downloads\speech_data"
OUT = r"data/uaspeech_smoketrain_manifest.jsonl"

# Labels discovered by consensus of control speakers on base Gemma 4 E4B
DIGIT_LABELS = {
    "D1": "one",   "D2": "two",   "D3": "three",
    "D4": "four",  "D5": "five",  "D6": "six",
    "D7": "seven", "D8": "eight", "D9": "nine",
}
SEVERITY = {
    "F02": "Low",  "F03": "Very Low", "F04": "Low",  "F05": "Mid",
    "M01": "Very Low", "M04": "Very Low", "M05": "Mid",     "M07": "Low",
    "M08": "Mid",      "M09": "High",     "M10": "Mid",     "M11": "Very Low",
    "M12": "Low",      "M14": "High",     "M16": "Low",
}

# Use 3 mics to get some acoustic diversity; all 3 blocks
MICS = ["M2", "M5", "M8"]
BLOCKS = ["B1", "B2", "B3"]


def main():
    records = []
    for spk, sev in SEVERITY.items():
        spk_dir = os.path.join(ROOT, spk)
        if not os.path.isdir(spk_dir):
            continue
        for digit, label in DIGIT_LABELS.items():
            for blk in BLOCKS:
                for mic in MICS:
                    fn = f"{spk}_{blk}_{digit}_{mic}.wav"
                    p = os.path.join(spk_dir, fn)
                    if os.path.exists(p):
                        records.append({
                            "audio_path": p,
                            "transcript": label,
                            "speaker": spk,
                            "severity": sev,
                            "source": "uaspeech",
                            "condition": "dysarthria",
                            "word_id": digit,
                        })

    # Also keep a smaller set of controls for calibration (don't overwhelm the mix)
    controls = ["CF02","CF03","CM01","CM05","CM09","CM13"]
    for spk in controls:
        cdir = os.path.join(ROOT, "control", spk)
        if not os.path.isdir(cdir):
            continue
        for digit, label in DIGIT_LABELS.items():
            # one mic only for controls
            for blk in BLOCKS:
                fn = f"{spk}_{blk}_{digit}_M2.wav"
                p = os.path.join(cdir, fn)
                if os.path.exists(p):
                    records.append({
                        "audio_path": p,
                        "transcript": label,
                        "speaker": spk,
                        "severity": "control",
                        "source": "uaspeech",
                        "condition": "control",
                        "word_id": digit,
                    })

    # Shuffle for training
    rng = random.Random(42)
    rng.shuffle(records)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    # Stats
    from collections import Counter
    by_spk = Counter(r["speaker"] for r in records)
    by_sev = Counter(r["severity"] for r in records)
    print(f"Wrote {len(records)} records to {OUT}")
    print(f"Per severity: {dict(by_sev)}")
    print(f"Per speaker (first 10): {dict(list(by_spk.items())[:10])}")


if __name__ == "__main__":
    main()

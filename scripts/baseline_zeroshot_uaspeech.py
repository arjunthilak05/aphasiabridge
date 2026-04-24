"""Zero-shot baseline: Gemma 4 E4B on UASpeech digits (D1-D10).

Method:
  1. Run base model on D1-D10 clips from BOTH control and dysarthric speakers,
     using Block 1 + Microphone 2 (one clip per digit per speaker) for a fair
     speaker-disjoint sweep.
  2. Ground truth per digit is taken as the MAJORITY output from control speakers
     (controls have clean articulation; their consensus is a reliable pseudo-label
     since we do not yet have the UASpeech wordlist file).
  3. Per-utterance WER (via jiwer) and per-severity aggregates using published
     UASpeech intelligibility ratings.

Output: outputs/baseline_uaspeech_digits.json  (raw outputs + per-speaker + per-severity WER)
"""
from __future__ import annotations
import os, sys, time, json, glob, re
from collections import Counter, defaultdict

os.environ.setdefault("HF_HOME", r"D:\Downloads\huggingface_cache")
os.environ.setdefault("HF_HUB_CACHE", r"D:\Downloads\huggingface_cache\hub")
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"

import torch
torch._dynamo.config.disable = True
torch._dynamo.config.suppress_errors = True

from unsloth import FastModel
from transformers import AutoProcessor
import soundfile as sf
import jiwer

MODEL_ID = "unsloth/gemma-4-E4B-it-unsloth-bnb-4bit"
ROOT = r"D:\Downloads\speech_data"

# Published UASpeech intelligibility ratings (see Kim et al. 2008)
SEVERITY = {
    "F02": "Low",  "F03": "Very Low", "F04": "Low",  "F05": "Mid",
    "M01": "Very Low", "M04": "Very Low", "M05": "Mid",     "M07": "Low",
    "M08": "Mid",      "M09": "High",     "M10": "Mid",     "M11": "Very Low",
    "M12": "Low",      "M14": "High",     "M16": "Low",
}
CONTROLS = ["CF02","CF03","CF04","CF05","CM01","CM04","CM05","CM06","CM08","CM09","CM10","CM12","CM13"]

DIGITS = [f"D{i}" for i in range(1, 11)]  # D1..D10
BLOCK = "B1"
MIC = "M2"

PROMPT = "Transcribe the audio. Output only the spoken word or words, nothing else."

CLEAN = re.compile(r"[^a-z\s']")


def clean(text: str) -> str:
    return CLEAN.sub(" ", text.lower()).strip()


def find_wav(speaker_dir: str, speaker_id: str, digit: str) -> str | None:
    # Filename: <SPK>_<BLOCK>_<WORDID>_<MIC>.wav
    pattern = os.path.join(speaker_dir, f"{speaker_id}_{BLOCK}_{digit}_{MIC}.wav")
    return pattern if os.path.exists(pattern) else None


def transcribe(model, processor, wav_path: str) -> str:
    audio, sr = sf.read(wav_path)
    assert sr == 16000
    messages = [{"role":"user","content":[
        {"type":"audio","audio":audio},
        {"type":"text","text":PROMPT},
    ]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    )
    inputs = {k: v.to(model.device) if hasattr(v,"to") else v for k,v in inputs.items()}
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=16, do_sample=False, temperature=None, top_p=None)
    input_len = inputs["input_ids"].shape[1]
    text = processor.decode(out[0, input_len:], skip_special_tokens=True).strip()
    return text


def main():
    print(f"[{time.strftime('%H:%M:%S')}] loading model ...", flush=True)
    model, tokenizer = FastModel.from_pretrained(
        model_name=MODEL_ID, max_seq_length=2048, load_in_4bit=True, full_finetuning=False,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model.eval()

    # Build speaker list with paths
    speakers = []
    for s in CONTROLS:
        d = os.path.join(ROOT, "control", s)
        if os.path.isdir(d): speakers.append((s, d, "control"))
    for s in SEVERITY:
        d = os.path.join(ROOT, s)
        if os.path.isdir(d): speakers.append((s, d, SEVERITY[s]))

    print(f"[{time.strftime('%H:%M:%S')}] running {len(speakers)} speakers x {len(DIGITS)} digits = {len(speakers)*len(DIGITS)} inferences", flush=True)

    results = []  # list of {speaker, severity, digit, wav, output}
    for spk, sdir, sev in speakers:
        for d in DIGITS:
            wav = find_wav(sdir, spk, d)
            if not wav:
                continue
            t0 = time.time()
            try:
                out = transcribe(model, processor, wav)
            except Exception as e:
                out = f"<ERROR:{type(e).__name__}>"
            results.append({
                "speaker": spk, "severity": sev, "digit": d,
                "output": out, "output_clean": clean(out),
                "dur_s": round(time.time()-t0, 2),
            })
        done = sum(1 for r in results if r["speaker"]==spk)
        print(f"  {spk} ({sev}): {done}/{len(DIGITS)} clips", flush=True)

    # Consensus ground truth from controls: majority vote of control outputs per digit
    gt = {}
    for d in DIGITS:
        ctrl_outs = [r["output_clean"] for r in results if r["digit"]==d and r["severity"]=="control" and not r["output"].startswith("<ERROR")]
        if not ctrl_outs:
            gt[d] = None
            continue
        most_common = Counter(ctrl_outs).most_common(1)[0]
        gt[d] = most_common[0]
        print(f"  GT[{d}] = {most_common[0]!r}  (vote {most_common[1]}/{len(ctrl_outs)})")

    # WER per speaker
    per_speaker_wer = {}
    for spk, _, sev in speakers:
        refs, hyps = [], []
        for r in results:
            if r["speaker"] != spk: continue
            ref = gt.get(r["digit"])
            if not ref: continue
            refs.append(ref)
            hyps.append(r["output_clean"] if not r["output"].startswith("<ERROR") else "")
        if refs:
            wer = jiwer.wer(refs, hyps)
            per_speaker_wer[spk] = {"severity": sev, "n": len(refs), "wer": round(wer, 3)}

    # Aggregate by severity
    bucket = defaultdict(lambda: {"refs":[], "hyps":[], "n":0})
    for spk, info in per_speaker_wer.items():
        for r in results:
            if r["speaker"] != spk: continue
            ref = gt.get(r["digit"])
            if not ref: continue
            bucket[info["severity"]]["refs"].append(ref)
            bucket[info["severity"]]["hyps"].append(r["output_clean"] if not r["output"].startswith("<ERROR") else "")
            bucket[info["severity"]]["n"] += 1
    per_severity = {k: {"n": v["n"], "wer": round(jiwer.wer(v["refs"], v["hyps"]), 3)} for k,v in bucket.items() if v["refs"]}

    print(f"\n=== Per-speaker WER ===")
    for spk, info in sorted(per_speaker_wer.items(), key=lambda x: (x[1]["severity"], x[0])):
        print(f"  {spk:6s} {info['severity']:10s}  n={info['n']:2d}  WER={info['wer']:.3f}")
    print(f"\n=== Per-severity WER ===")
    for sev, info in sorted(per_severity.items(), key=lambda x: ["control","High","Mid","Low","Very Low"].index(x[0]) if x[0] in ["control","High","Mid","Low","Very Low"] else 99):
        print(f"  {sev:10s}  n={info['n']:3d}  WER={info['wer']:.3f}")

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/baseline_uaspeech_digits.json", "w", encoding="utf-8") as f:
        json.dump({
            "model": MODEL_ID,
            "prompt": PROMPT,
            "ground_truth": gt,
            "per_speaker_wer": per_speaker_wer,
            "per_severity_wer": per_severity,
            "raw": results,
        }, f, indent=2)
    print("\nSaved: outputs/baseline_uaspeech_digits.json")


if __name__ == "__main__":
    main()

"""Eval Gemma 4 E4B + LoRA on UASpeech digits, using the same probe set as
scripts/baseline_zeroshot_uaspeech.py so the numbers are directly comparable.

Loads a LoRA adapter directory (Unsloth saves one via model.save_pretrained)
and re-runs the digit sweep. Prints before-vs-after WER per severity.
"""
from __future__ import annotations
import os, sys, time, json, re, argparse
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
from peft import PeftModel
import soundfile as sf
import jiwer

MODEL_ID = "unsloth/gemma-4-E4B-it-unsloth-bnb-4bit"
ROOT = r"D:\Downloads\speech_data"
BASELINE_JSON = r"outputs/baseline_uaspeech_digits.json"

SEVERITY = {
    "F02": "Low",  "F03": "Very Low", "F04": "Low",  "F05": "Mid",
    "M01": "Very Low", "M04": "Very Low", "M05": "Mid",     "M07": "Low",
    "M08": "Mid",      "M09": "High",     "M10": "Mid",     "M11": "Very Low",
    "M12": "Low",      "M14": "High",     "M16": "Low",
}
CONTROLS = ["CF02","CF03","CF04","CF05","CM01","CM04","CM05","CM06","CM08","CM09","CM10","CM12","CM13"]

DIGITS = [f"D{i}" for i in range(1, 11)]
BLOCK = "B1"
MIC = "M2"
PROMPT = "Please transcribe this audio."
SYSTEM = "You are an assistant that transcribes speech accurately."

CLEAN = re.compile(r"[^a-z\s']")
def clean(t: str) -> str:
    return CLEAN.sub(" ", t.lower()).strip()


def find_wav(spk_dir, spk, digit):
    p = os.path.join(spk_dir, f"{spk}_{BLOCK}_{digit}_{MIC}.wav")
    return p if os.path.exists(p) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora-dir", required=True, help="Path to LoRA adapter dir (must contain adapter_model.safetensors)")
    ap.add_argument("--out", default="outputs/eval_with_lora_digits.json")
    args = ap.parse_args()

    # Load via FastModel passing the LoRA dir directly — Unsloth handles its
    # custom Gemma4ClippableLinear modules internally (stock PEFT cannot wrap them).
    # Use a FRESH processor from the base model to avoid any drift in the saved one.
    print(f"[{time.strftime('%H:%M:%S')}] loading LoRA via FastModel from {args.lora_dir} ...", flush=True)
    model, _ = FastModel.from_pretrained(
        model_name=args.lora_dir,
        max_seq_length=2048,
        load_in_4bit=True,
        full_finetuning=False,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model.eval()

    speakers = []
    for s in CONTROLS:
        d = os.path.join(ROOT, "control", s)
        if os.path.isdir(d): speakers.append((s, d, "control"))
    for s in SEVERITY:
        d = os.path.join(ROOT, s)
        if os.path.isdir(d): speakers.append((s, d, SEVERITY[s]))

    print(f"[{time.strftime('%H:%M:%S')}] running sweep ({len(speakers)} speakers x {len(DIGITS)} digits)", flush=True)
    results = []
    for spk, sdir, sev in speakers:
        for d in DIGITS:
            wav = find_wav(sdir, spk, d)
            if not wav: continue
            audio, sr = sf.read(wav)
            messages = [
                {"role":"system","content":[{"type":"text","text":SYSTEM}]},
                {"role":"user","content":[
                    {"type":"audio","audio":audio},
                    {"type":"text","text":PROMPT},
                ]},
            ]
            inputs = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt",
            )
            inputs = {k: v.to(model.device) if hasattr(v,"to") else v for k,v in inputs.items()}
            t0 = time.time()
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=16, do_sample=False,
                                     temperature=None, top_p=None)
            input_len = inputs["input_ids"].shape[1]
            text = processor.decode(out[0, input_len:], skip_special_tokens=True).strip()
            results.append({
                "speaker": spk, "severity": sev, "digit": d,
                "output": text, "output_clean": clean(text),
                "dur_s": round(time.time()-t0, 2),
            })
        done = sum(1 for r in results if r["speaker"]==spk)
        print(f"  {spk:6s} ({sev:10s}): {done}/{len(DIGITS)}", flush=True)

    # Use SAME ground truth as baseline: majority vote of controls on this new run
    gt = {}
    for d in DIGITS:
        ctrl = [r["output_clean"] for r in results if r["digit"]==d and r["severity"]=="control"]
        if ctrl:
            gt[d] = Counter(ctrl).most_common(1)[0][0]

    # Per-speaker WER
    per_speaker = {}
    for spk, _, sev in speakers:
        refs, hyps = [], []
        for r in results:
            if r["speaker"] != spk: continue
            ref = gt.get(r["digit"])
            if not ref: continue
            refs.append(ref); hyps.append(r["output_clean"])
        if refs:
            per_speaker[spk] = {"severity": sev, "n": len(refs), "wer": round(jiwer.wer(refs, hyps), 3)}

    # Per-severity aggregate
    bucket = defaultdict(lambda: {"refs":[], "hyps":[]})
    for spk, info in per_speaker.items():
        for r in results:
            if r["speaker"] != spk: continue
            ref = gt.get(r["digit"])
            if not ref: continue
            bucket[info["severity"]]["refs"].append(ref)
            bucket[info["severity"]]["hyps"].append(r["output_clean"])
    per_sev_after = {k: round(jiwer.wer(v["refs"], v["hyps"]), 3) for k, v in bucket.items() if v["refs"]}

    # Load baseline for comparison
    per_sev_before = {}
    if os.path.exists(BASELINE_JSON):
        with open(BASELINE_JSON, "r", encoding="utf-8") as f:
            base = json.load(f)
        per_sev_before = {k: v["wer"] for k, v in base.get("per_severity_wer", {}).items()}

    print(f"\n=== BEFORE vs AFTER LoRA (WER) ===")
    print(f"{'Severity':12s}  {'Before':>8s}  {'After':>8s}  {'Δ':>8s}")
    for sev in ["control", "High", "Mid", "Low", "Very Low"]:
        b = per_sev_before.get(sev)
        a = per_sev_after.get(sev)
        if a is None: continue
        if b is None:
            print(f"{sev:12s}  {'--':>8s}  {a:>8.3f}  {'--':>8s}")
        else:
            delta = a - b
            print(f"{sev:12s}  {b:>8.3f}  {a:>8.3f}  {delta:>+8.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({
            "lora_dir": args.lora_dir,
            "ground_truth": gt,
            "per_speaker_wer": per_speaker,
            "per_severity_wer": per_sev_after,
            "baseline_per_severity_wer": per_sev_before,
            "raw": results,
        }, f, indent=2)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()

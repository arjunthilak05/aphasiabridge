"""Tier-2 Lever A: aggregated SS-FT eval across speakers.

For each participant_id passed via --speakers, load BASE + the per-speaker
SS-FT adapter (from outputs/ss_ft/<pid>/lora_final/), filter ss_test to that
speaker's clips, run inference with the same gen kwargs as eval_lora.py, and
aggregate WER + per-severity. Also runs a D4b-zero-shot baseline pass on the
SAME aligned clips so we can compute BASE-vs-SS-FT delta on the per-speaker
subset.

Outputs:
  outputs/ss_ft/aggregated_ss_test_results.json  per-speaker + overall WER
  prints headline + bootstrap-friendly raw predictions
"""
from __future__ import annotations
import os, json, argparse, time, traceback
from collections import defaultdict
from pathlib import Path

_WIN_HF = r"D:\Downloads\huggingface_cache"
os.environ.setdefault("HF_HOME", _WIN_HF if os.path.isdir(_WIN_HF) else os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("HF_HUB_CACHE", os.path.join(os.environ["HF_HOME"], "hub"))
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
torch._dynamo.config.disable = True
torch._dynamo.config.suppress_errors = True

from unsloth import FastModel
import soundfile as sf
import jiwer
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
from peft import PeftModel

MODEL_ID = "unsloth/gemma-4-E4B-it-unsloth-bnb-4bit"
SYSTEM_PROMPT = "You are an assistant that transcribes speech accurately."
USER_PROMPT = "Please transcribe this audio."


def load_manifest(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def run_inference(model, processor, clips, gen_kwargs):
    outs = []
    for i, r in enumerate(clips):
        try:
            audio, sr = sf.read(r["audio_path"])
            if sr != 16000:
                outs.append({"ref": r["transcript"], "hyp": "<ERROR:bad_sr>",
                             "severity": r.get("severity"),
                             "speaker": r.get("participant_id")})
                continue
            messages = [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user",
                 "content": [{"type": "audio", "audio": audio},
                             {"type": "text", "text": USER_PROMPT}]},
            ]
            inputs = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt",
            )
            inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
            with torch.no_grad():
                out = model.generate(**inputs, **gen_kwargs)
            input_len = inputs["input_ids"].shape[1]
            text = processor.decode(out[0, input_len:], skip_special_tokens=True).strip()
        except Exception:
            text = f"<ERROR:{type(__import__('sys').exc_info()[1]).__name__}>"
        outs.append({"ref": r["transcript"], "hyp": text,
                     "severity": r.get("severity"),
                     "speaker": r.get("participant_id"),
                     "audio_path": r["audio_path"]})
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--si-lora-dir", required=True,
                    help="Path to R1-D4b lora_final (used for the D4b baseline pass)")
    ap.add_argument("--ss-ft-root", required=True,
                    help="Root dir holding per-speaker subdirs with lora_final/")
    ap.add_argument("--speakers", required=True,
                    help="Comma-separated participant_ids (or path to a text file with one per line)")
    ap.add_argument("--ss-test", required=True,
                    help="Path to aphasiabank_ss_test.jsonl")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--rep-penalty", type=float, default=1.3)
    ap.add_argument("--no-repeat-ngram", type=int, default=4)
    ap.add_argument("--num-beams", type=int, default=5)
    ap.add_argument("--length-penalty", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    args = ap.parse_args()

    if os.path.isfile(args.speakers):
        speakers = [l.strip() for l in open(args.speakers) if l.strip()]
    else:
        speakers = [s.strip() for s in args.speakers.split(",") if s.strip()]
    print(f"[{time.strftime('%H:%M:%S')}] eval-ing {len(speakers)} speakers", flush=True)

    os.makedirs(args.output_dir, exist_ok=True)
    en_norm = EnglishTextNormalizer({})

    ss_test = load_manifest(args.ss_test)
    by_speaker = defaultdict(list)
    for r in ss_test:
        by_speaker[r.get("participant_id")].append(r)

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens, do_sample=False,
        temperature=None, top_p=None,
        repetition_penalty=args.rep_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram,
        num_beams=args.num_beams,
        length_penalty=args.length_penalty,
    )

    # Load BASE model + D4b LoRA once for baseline pass
    print(f"[{time.strftime('%H:%M:%S')}] loading BASE + R1-D4b for baseline pass ...", flush=True)
    base_model, processor = FastModel.from_pretrained(
        model_name=MODEL_ID, dtype=None,
        max_seq_length=args.max_new_tokens + 1024, load_in_4bit=True,
        full_finetuning=False,
    )
    d4b_model = PeftModel.from_pretrained(base_model, args.si_lora_dir, is_trainable=False)
    d4b_model.eval()
    print(f"  baseline model loaded; VRAM={torch.cuda.memory_allocated()/1e9:.2f} GB")

    all_d4b_preds = {}
    print(f"[{time.strftime('%H:%M:%S')}] D4b baseline pass on {len(speakers)} speakers' test clips ...", flush=True)
    for spk in speakers:
        clips = by_speaker.get(spk, [])
        if not clips:
            print(f"  {spk}: no test clips, skipping baseline")
            continue
        all_d4b_preds[spk] = run_inference(d4b_model, processor, clips, gen_kwargs)
        print(f"  {spk}: D4b baseline {len(clips)} clips done", flush=True)

    # Free model so SS-FT models can load
    del d4b_model, base_model
    torch.cuda.empty_cache()

    # Now load each SS-FT adapter and eval that speaker's clips
    all_ss_preds = {}
    print(f"\n[{time.strftime('%H:%M:%S')}] SS-FT pass per speaker ...", flush=True)
    for spk in speakers:
        ss_dir = os.path.join(args.ss_ft_root, spk, "lora_final")
        if not os.path.isdir(ss_dir):
            print(f"  {spk}: missing SS-FT adapter at {ss_dir}, skipping")
            continue
        clips = by_speaker.get(spk, [])
        if not clips:
            continue
        # Reload model fresh each speaker (otherwise PEFT adapter swapping is messy)
        base_model, processor = FastModel.from_pretrained(
            model_name=MODEL_ID, dtype=None,
            max_seq_length=args.max_new_tokens + 1024, load_in_4bit=True,
            full_finetuning=False,
        )
        ss_model = PeftModel.from_pretrained(base_model, ss_dir, is_trainable=False)
        ss_model.eval()
        all_ss_preds[spk] = run_inference(ss_model, processor, clips, gen_kwargs)
        print(f"  {spk}: SS-FT {len(clips)} clips done", flush=True)
        del ss_model, base_model
        torch.cuda.empty_cache()

    # Aggregate
    def wer_table(preds_by_spk, normalizer):
        all_refs, all_hyps = [], []
        per_spk = {}
        per_sev = defaultdict(lambda: {"r": [], "h": []})
        for spk, preds in preds_by_spk.items():
            spk_refs, spk_hyps = [], []
            for p in preds:
                if p["hyp"].startswith("<ERROR"): continue
                nr = normalizer(p["ref"]); nh = normalizer(p["hyp"])
                if not nr or not nh: continue
                spk_refs.append(nr); spk_hyps.append(nh)
                all_refs.append(nr); all_hyps.append(nh)
                per_sev[p.get("severity") or "unknown"]["r"].append(nr)
                per_sev[p.get("severity") or "unknown"]["h"].append(nh)
            if spk_refs:
                per_spk[spk] = round(jiwer.wer(spk_refs, spk_hyps), 4)
        overall = round(jiwer.wer(all_refs, all_hyps), 4) if all_refs else None
        sev_wer = {k: round(jiwer.wer(v["r"], v["h"]), 4) for k, v in per_sev.items() if v["r"]}
        return {"overall": overall, "n_eval": len(all_refs), "per_speaker": per_spk, "per_severity": sev_wer}

    out = {
        "speakers": speakers,
        "gen_kwargs": gen_kwargs,
        "d4b_baseline_lowercase": wer_table(all_d4b_preds, lambda s: s.lower().strip()),
        "ss_ft_lowercase":        wer_table(all_ss_preds,  lambda s: s.lower().strip()),
        "d4b_baseline_whisperEN": wer_table(all_d4b_preds, en_norm),
        "ss_ft_whisperEN":        wer_table(all_ss_preds,  en_norm),
        "d4b_predictions":        all_d4b_preds,
        "ss_ft_predictions":      all_ss_preds,
    }

    out_path = os.path.join(args.output_dir, "aggregated_ss_test_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n{'='*80}\nSS-FT vs D4b baseline (same {sum(len(v) for v in all_d4b_preds.values())} test clips)\n{'='*80}")
    for label, key in [("lowercase", "lowercase"), ("WhisperEN", "whisperEN")]:
        d4b = out[f"d4b_baseline_{key}"]; ssf = out[f"ss_ft_{key}"]
        if d4b["overall"] is None or ssf["overall"] is None:
            continue
        delta = round(ssf["overall"] - d4b["overall"], 4)
        print(f"\n  {label}:")
        print(f"    D4b baseline overall: {d4b['overall']}  (n={d4b['n_eval']})")
        print(f"    SS-FT      overall: {ssf['overall']}  (n={ssf['n_eval']})  Δ={delta:+.4f}")
        print(f"    D4b per-severity:   {d4b['per_severity']}")
        print(f"    SS-FT per-severity: {ssf['per_severity']}")
        print(f"    Per-speaker (D4b → SS-FT):")
        for spk in speakers:
            a = d4b['per_speaker'].get(spk); b = ssf['per_speaker'].get(spk)
            if a is None or b is None: continue
            print(f"      {spk:32s} {a:.4f} → {b:.4f}  Δ={b-a:+.4f}")

    print(f"\n  Wrote {out_path}")


if __name__ == "__main__":
    main()

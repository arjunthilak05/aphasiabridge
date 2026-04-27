"""Standalone re-eval: load a saved LoRA and re-evaluate with custom generate kwargs.

Phase-1 cheap win: R1-D3 had 15/220 repetition loops at decode time. Whisper's
standard fix is no_repeat_ngram_size=4 + repetition_penalty. This evaluates
already-trained LoRAs with those settings, no retraining.

Also dumps full traceback for the first N <ERROR:ValueError> clips so we can
fix or consistently filter the 24/220 audio-processor failures.
"""
from __future__ import annotations
import os, sys, json, argparse, time, traceback
from collections import defaultdict

_WIN_HF = r"D:\Downloads\huggingface_cache"
os.environ.setdefault("HF_HOME", _WIN_HF if os.path.isdir(_WIN_HF) else os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("HF_HUB_CACHE", os.path.join(os.environ["HF_HOME"], "hub"))
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
torch._dynamo.config.disable = True
torch._dynamo.config.suppress_errors = True

from unsloth import FastModel
import numpy as np
import soundfile as sf
import jiwer


def pad_audio_to_safe_length(audio, sr=16000, quantum_s=1.0):
    """Pad audio to the next multiple of `quantum_s` seconds with trailing zeros.

    Phase 0a: works around Gemma 4's `Audio features and audio tokens do not match`
    edge case (24 NEURAL-corpus clips on our ss_test). The processor's audio-token-
    placeholder count and the audio-encoder's feature-count diverge for certain
    durations; rounding the input duration up to the next 1-second boundary makes
    the math align cleanly. Whisper baselines unaffected (they use their own
    feature extractor)."""
    target = int(np.ceil(audio.shape[0] / (sr * quantum_s)) * sr * quantum_s)
    if target <= audio.shape[0]:
        return audio
    pad = target - audio.shape[0]
    return np.concatenate([audio, np.zeros(pad, dtype=audio.dtype)])

MODEL_ID = "unsloth/gemma-4-E4B-it-unsloth-bnb-4bit"
SYSTEM_PROMPT = "You are an assistant that transcribes speech accurately."
USER_PROMPT = "Please transcribe this audio."


def load_manifest(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora-dir", required=True,
                    help="Path containing adapter_config.json + adapter_model.safetensors")
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--rep-penalty", type=float, default=1.3)
    ap.add_argument("--no-repeat-ngram", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--num-beams", type=int, default=1)
    ap.add_argument("--length-penalty", type=float, default=1.0,
                    help="Beam search length penalty (>1 favors longer; <1 favors shorter). 0.6 kills hallucinated continuations.")
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dump-tracebacks", type=int, default=3,
                    help="Print full traceback for first N failing clips")
    ap.add_argument("--skip-base", action="store_true",
                    help="Skip BASE (LoRA-disabled) eval pass")
    ap.add_argument("--system-prompt", default=None,
                    help="Override SYSTEM_PROMPT. Default: 'You are an assistant that transcribes speech accurately.'")
    ap.add_argument("--user-prompt", default=None,
                    help="Override USER_PROMPT. Default: 'Please transcribe this audio.'")
    ap.add_argument("--no-system-prompt", action="store_true",
                    help="Drop the system prompt entirely (Lever-4 variant 5).")
    args = ap.parse_args()
    # Apply prompt overrides
    sys_prompt = SYSTEM_PROMPT if args.system_prompt is None else args.system_prompt
    usr_prompt = USER_PROMPT   if args.user_prompt   is None else args.user_prompt

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[{time.strftime('%H:%M:%S')}] loading base model {MODEL_ID} ...", flush=True)
    model, processor = FastModel.from_pretrained(
        model_name=MODEL_ID,
        dtype=None,
        max_seq_length=args.max_length,
        load_in_4bit=True,
        full_finetuning=False,
    )

    print(f"[{time.strftime('%H:%M:%S')}] loading LoRA from {args.lora_dir} ...", flush=True)
    # First apply LoRA scaffolding via get_peft_model with same target_modules
    # then load weights from disk. This is the workaround for the audio-LoRA
    # reload bug noted in STATUS.md §7.
    from peft import PeftModel
    try:
        model = PeftModel.from_pretrained(model, args.lora_dir, is_trainable=False)
        print(f"  PeftModel.from_pretrained OK")
    except Exception as e:
        print(f"  PeftModel.from_pretrained FAILED: {e}")
        print(f"  falling back to FastModel.from_pretrained(lora_dir) ...")
        model, processor = FastModel.from_pretrained(
            model_name=args.lora_dir,
            dtype=None,
            max_seq_length=args.max_length,
            load_in_4bit=True,
        )

    holdout = load_manifest(args.holdout)
    if args.limit:
        holdout = holdout[:args.limit]
    print(f"[{time.strftime('%H:%M:%S')}] eval on {len(holdout)} clips with "
          f"rep_penalty={args.rep_penalty} no_repeat_ngram={args.no_repeat_ngram} "
          f"max_new_tokens={args.max_new_tokens} beams={args.num_beams}", flush=True)

    model.eval()
    traceback_dumps_remaining = args.dump_tracebacks

    def run_eval(recs, label):
        nonlocal traceback_dumps_remaining
        outs = []
        for i, r in enumerate(recs):
            try:
                audio, sr = sf.read(r["audio_path"])
                if sr != 16000:
                    outs.append({"ref": r["transcript"], "hyp": "<ERROR:bad_sr>",
                                 "severity": r.get("severity")})
                    continue
                # Phase 0a: pad to next 1-second boundary to avoid Gemma 4's
                # audio-token alignment edge case on certain durations.
                audio = pad_audio_to_safe_length(audio, sr=16000, quantum_s=1.0)
                messages = []
                if not args.no_system_prompt:
                    messages.append({"role": "system",
                                     "content": [{"type": "text", "text": sys_prompt}]})
                messages.append({"role": "user",
                                 "content": [
                                     {"type": "audio", "audio": audio},
                                     {"type": "text",  "text": usr_prompt},
                                 ]})
                inputs = processor.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True,
                    return_dict=True, return_tensors="pt",
                )
                inputs = {k: v.to(model.device) if hasattr(v,"to") else v
                          for k,v in inputs.items()}
                gen_kwargs = dict(
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    temperature=None, top_p=None,
                    repetition_penalty=args.rep_penalty,
                    no_repeat_ngram_size=args.no_repeat_ngram,
                    num_beams=args.num_beams,
                    length_penalty=args.length_penalty,
                )
                with torch.no_grad():
                    out = model.generate(**inputs, **gen_kwargs)
                input_len = inputs["input_ids"].shape[1]
                text = processor.decode(out[0, input_len:], skip_special_tokens=True).strip()
            except Exception as e:
                if traceback_dumps_remaining > 0:
                    print(f"\n=== TRACEBACK for clip {i} ({label}) ===", flush=True)
                    print(f"  audio_path: {r['audio_path']}")
                    print(f"  duration_ms: {r.get('duration_ms')}")
                    print(f"  audio.shape: {audio.shape if 'audio' in dir() else 'NA'}")
                    print(f"  sr: {sr if 'sr' in dir() else 'NA'}")
                    traceback.print_exc()
                    print(f"=== END TRACEBACK ===\n", flush=True)
                    traceback_dumps_remaining -= 1
                text = f"<ERROR:{type(e).__name__}>"
            outs.append({"ref": r["transcript"], "hyp": text,
                         "severity": r.get("severity"),
                         "speaker": r.get("participant_id")})
            if (i+1) % 25 == 0:
                print(f"    [{label}] {i+1}/{len(recs)} done", flush=True)
        return outs

    base_outs = None
    if not args.skip_base:
        print("Phase A: BASE (LoRA disabled)")
        try:
            with model.disable_adapter():
                base_outs = run_eval(holdout, "BASE")
        except Exception as e:
            print(f"  disable_adapter failed: {e}; skipping base pass")

    print("Phase B: AFTER (LoRA active)")
    after_outs = run_eval(holdout, "AFTER")

    def aggregate(outs):
        by_sev = defaultdict(lambda: {"refs":[], "hyps":[]})
        for o in outs:
            if o["hyp"].startswith("<ERROR"): continue
            by_sev[o["severity"] or "unknown"]["refs"].append(o["ref"].lower())
            by_sev[o["severity"] or "unknown"]["hyps"].append(o["hyp"].lower())
        refs = [o["ref"].lower() for o in outs if not o["hyp"].startswith("<ERROR")]
        hyps = [o["hyp"].lower() for o in outs if not o["hyp"].startswith("<ERROR")]
        return {
            "overall_wer": round(jiwer.wer(refs, hyps), 4) if refs else None,
            "per_severity_wer": {k: round(jiwer.wer(v["refs"], v["hyps"]), 4)
                                 for k,v in by_sev.items() if v["refs"]},
            "n_eval": len(refs),
            "n_errors": sum(1 for o in outs if o["hyp"].startswith("<ERROR")),
            "n_total": len(outs),
        }

    import re
    def failure_modes(outs):
        repeat = sum(1 for r in outs if re.search(r'(\bthe ){8,}|(\band ){8,}|(\bher ){8,}|(\bi\'ve ){8,}', r["hyp"].lower()))
        hallu  = sum(1 for r in outs if 'go to the ball' in r["hyp"].lower() or 'peanut butter' in r["hyp"].lower() or "i'm gonna go" in r["hyp"].lower())
        errs   = sum(1 for r in outs if r["hyp"].startswith("<ERROR"))
        return {"repeat_loops": repeat, "protocol_hallucinations": hallu, "errors": errs}

    result = {"after": aggregate(after_outs), "after_failure_modes": failure_modes(after_outs)}
    if base_outs is not None:
        result["base"] = aggregate(base_outs)
        result["base_failure_modes"] = failure_modes(base_outs)
        b = result['base']['overall_wer']; a = result['after']['overall_wer']
        print(f"\n>>> WER  BASE: {b}  AFTER: {a}  delta: {round(a-b,4) if (a and b) else 'NA'}")
        print(f"  Per-severity BASE:  {result['base']['per_severity_wer']}")
        print(f"  Per-severity AFTER: {result['after']['per_severity_wer']}")
        print(f"  BASE failure modes:  {result['base_failure_modes']}")
        print(f"  AFTER failure modes: {result['after_failure_modes']}")
    else:
        print(f"\n>>> WER AFTER only: {result['after']['overall_wer']}")
        print(f"  AFTER failure modes: {result['after_failure_modes']}")

    out_path = os.path.join(args.output_dir, "reeval_summary.json")
    with open(out_path, "w") as f:
        json.dump({
            "lora_dir": args.lora_dir,
            "holdout": args.holdout,
            "gen_kwargs": {
                "rep_penalty": args.rep_penalty,
                "no_repeat_ngram": args.no_repeat_ngram,
                "max_new_tokens": args.max_new_tokens,
                "num_beams": args.num_beams,
            },
            "result": result,
            "after_raw": after_outs,   # all predictions for downstream WER recompute
            "base_raw": (base_outs if base_outs else None),
        }, f, indent=2)
    print(f"\n  Saved {out_path}")


if __name__ == "__main__":
    main()

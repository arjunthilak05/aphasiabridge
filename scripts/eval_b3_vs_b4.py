"""B3 vs B4 paired evaluation: same speakers, same per-speaker test clips.

For each SS speaker, load the B3 adapter (D4b → SS-FT) and the B4 adapter
(base → SS-FT) and run inference on the same held-out clips. Saves
per-speaker per-condition predictions and a summary JSON.

Used by analyze_b3_vs_b4.py to compute paired ΔWER, bootstrap CI, sign test.
"""
from __future__ import annotations
import os, json, argparse, time, traceback
from collections import defaultdict

_WIN_HF = r"D:\Downloads\huggingface_cache"
os.environ.setdefault("HF_HOME", _WIN_HF if os.path.isdir(_WIN_HF) else os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("HF_HUB_CACHE", os.path.join(os.environ["HF_HOME"], "hub"))
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import torch
torch._dynamo.config.disable = True
torch._dynamo.config.suppress_errors = True

from unsloth import FastModel
import soundfile as sf

SYSTEM_PROMPT = "You are an assistant that transcribes speech accurately."
USER_PROMPT = "Please transcribe this audio."


def load_manifest(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def pad_audio(audio, sr=16000, quantum_s=1.0):
    import numpy as np
    target = int(np.ceil(audio.shape[0] / (sr * quantum_s)) * sr * quantum_s)
    if target <= audio.shape[0]:
        return audio
    pad = target - audio.shape[0]
    return np.concatenate([audio, np.zeros(pad, dtype=audio.dtype)])


def run_inference(model, processor, clips, gen_kwargs):
    outs = []
    for i, r in enumerate(clips):
        try:
            audio, sr = sf.read(r["audio_path"])
            assert sr == 16000
            audio = pad_audio(audio, sr=16000, quantum_s=1.0)
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
        outs.append({
            "ref": r["transcript"], "hyp": text,
            "severity": r.get("severity"),
            "speaker": r.get("participant_id"),
            "audio_path": r["audio_path"],
        })
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b3-root", required=True,
                    help="Root holding outputs/ss_ft_b3/<spk>/lora_final adapters")
    ap.add_argument("--b4-root", required=True,
                    help="Root holding outputs/ss_ft_b4/<spk>/lora_final adapters")
    ap.add_argument("--ss-test", required=True)
    ap.add_argument("--speakers", default=None,
                    help="Optional comma-sep list or path-to-file. Default: all speakers under b3-root.")
    ap.add_argument("--output", required=True,
                    help="Output JSON path with per-speaker predictions for both conditions")
    ap.add_argument("--rep-penalty", type=float, default=1.3)
    ap.add_argument("--no-repeat-ngram", type=int, default=4)
    ap.add_argument("--num-beams", type=int, default=5)
    ap.add_argument("--length-penalty", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--limit-speakers", type=int, default=None)
    args = ap.parse_args()

    if args.speakers:
        if os.path.isfile(args.speakers):
            speakers = [l.strip() for l in open(args.speakers) if l.strip()]
        else:
            speakers = [s.strip() for s in args.speakers.split(",") if s.strip()]
    else:
        speakers = sorted([
            d for d in os.listdir(args.b3_root)
            if os.path.isdir(os.path.join(args.b3_root, d, "lora_final"))
        ])
    if args.limit_speakers:
        speakers = speakers[: args.limit_speakers]
    print(f"[{time.strftime('%H:%M:%S')}] eval-ing {len(speakers)} speakers", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)

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
    print(f"  gen_kwargs: {gen_kwargs}")

    # For memory: load each speaker's B3 + B4 adapter sequentially, run on that speaker's clips, free.
    out = {"gen_kwargs": gen_kwargs, "speakers": speakers, "predictions": {}}

    for s_idx, spk in enumerate(speakers):
        clips = by_speaker.get(spk, [])
        if not clips:
            print(f"  [{s_idx+1}/{len(speakers)}] {spk}: NO TEST CLIPS, skip", flush=True)
            continue
        b3_dir = os.path.join(args.b3_root, spk, "lora_final")
        b4_dir = os.path.join(args.b4_root, spk, "lora_final")
        if not (os.path.isdir(b3_dir) and os.path.isdir(b4_dir)):
            print(f"  [{s_idx+1}/{len(speakers)}] {spk}: missing adapter (b3={os.path.isdir(b3_dir)}, b4={os.path.isdir(b4_dir)}), skip", flush=True)
            continue

        spk_out = {}
        for cond, adapter_dir in (("b3", b3_dir), ("b4", b4_dir)):
            t0 = time.time()
            try:
                model, processor = FastModel.from_pretrained(
                    model_name=adapter_dir, dtype=None,
                    max_seq_length=args.max_length, load_in_4bit=True,
                    full_finetuning=False,
                )
                model.eval()
                preds = run_inference(model, processor, clips, gen_kwargs)
                del model
                torch.cuda.empty_cache()
            except Exception as e:
                preds = [{"ref": r["transcript"], "hyp": f"<ERROR_LOAD:{type(e).__name__}>",
                          "severity": r.get("severity"), "speaker": spk,
                          "audio_path": r["audio_path"]} for r in clips]
            spk_out[cond] = preds
            print(f"  [{s_idx+1}/{len(speakers)}] {spk} {cond}: {len(preds)} clips in {time.time()-t0:.0f}s", flush=True)
        out["predictions"][spk] = spk_out

        # Incremental save so progress isn't lost
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)

    print(f"\n[{time.strftime('%H:%M:%S')}] DONE. wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()

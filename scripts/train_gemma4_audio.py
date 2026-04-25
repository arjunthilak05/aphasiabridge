"""Gemma 4 / 3n audio LoRA fine-tune — Daniel Hanchen recipe (proven working).

Reference: Unsloth Kaggle notebook "Gemma 3N 4B Audio finetuning" (German ASR demo).
That recipe achieved 24.32% -> 16.22% WER (33% relative reduction) on 3000 German
clips in 47.82 min on a single T4. Our prior pipeline (UnslothVisionDataCollator +
AssistantOnlyLossCollator + missing modules_to_save) plateaued at loss ~3.2 with
no real WER win. This script mirrors Daniel's pattern verbatim.

Key differences from train_gemma4_lora.py:
  - target_modules audio list: ONLY post, linear_start, linear_end,
    embedding_projection (drops ffw_layer_1, ffw_layer_2, output_proj — these
    aren't in Daniel's recipe)
  - modules_to_save = [lm_head, embed_tokens, embed_audio] — the audio embedding
    is FULLY trained (not LoRA), which is what unblocks the audio path
  - Custom collate_fn: processor.apply_chat_template(..., tokenize=False) then
    processor(text=, audio=, return_tensors='pt', padding=True). Mask only
    pad/audio/special tokens to -100, NOT the prompt — Daniel's recipe trains on
    everything except padding and audio placeholder positions
  - max_length 2048 (not 4096) — matches notebook exactly
  - weight_decay 0.01 (not 0.001)
  - warmup_ratio 0.1 (not 0.03)
"""
from __future__ import annotations
import os, sys, json, argparse, time
from collections import Counter, defaultdict

_WIN_HF = r"D:\Downloads\huggingface_cache"
os.environ.setdefault("HF_HOME", _WIN_HF if os.path.isdir(_WIN_HF) else os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("HF_HUB_CACHE", os.path.join(os.environ["HF_HOME"], "hub"))
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"

import torch
torch._dynamo.config.disable = True
torch._dynamo.config.suppress_errors = True

from unsloth import FastModel
from trl import SFTTrainer, SFTConfig
from datasets import Dataset
import soundfile as sf
import jiwer


MODEL_ID = "unsloth/gemma-4-E4B-it-unsloth-bnb-4bit"
SYSTEM_PROMPT = "You are an assistant that transcribes speech accurately."
USER_PROMPT = "Please transcribe this audio."


def load_manifest(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def build_collate_fn(processor):
    """Daniel Hanchen's collator: apply_chat_template(tokenize=False) then
    processor(text=, audio=, padding=True). Mask pad / audio placeholder /
    special boi/eoi tokens to -100; loss on everything else (prompt + answer).
    """
    tok = processor.tokenizer

    def collate_fn(examples):
        texts, audios = [], []
        for ex in examples:
            audio, sr = sf.read(ex["audio_path"])
            assert sr == 16000, f"expected 16 kHz got {sr} at {ex['audio_path']}"
            messages = [
                {"role": "system",
                 "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user",
                 "content": [
                     {"type": "audio", "audio": audio},
                     {"type": "text",  "text": USER_PROMPT},
                 ]},
                {"role": "assistant",
                 "content": [{"type": "text", "text": ex["transcript"]}]},
            ]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            ).strip()
            texts.append(text)
            audios.append(audio)
        batch = processor(
            text=texts, audio=audios, return_tensors="pt", padding=True,
        )
        labels = batch["input_ids"].clone()
        labels[labels == tok.pad_token_id] = -100
        for attr in ("audio_token_id", "image_token_id",
                     "boi_token_id", "eoi_token_id",
                     "boa_token_id", "eoa_token_id"):
            if hasattr(tok, attr):
                tid = getattr(tok, attr)
                if tid is not None:
                    labels[labels == tid] = -100
        batch["labels"] = labels
        return batch

    return collate_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--holdout-manifest", default=None)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-steps", type=int, default=-1,
                    help="If -1, use --num-epochs instead. Daniel's notebook used 1 epoch.")
    ap.add_argument("--num-epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=2)        # Daniel's value
    ap.add_argument("--grad-accum", type=int, default=1)        # Daniel's value
    ap.add_argument("--lora-r", type=int, default=8)            # Daniel's value
    ap.add_argument("--lora-alpha", type=int, default=16)       # Daniel's value
    ap.add_argument("--lr", type=float, default=5e-5)           # Daniel's value
    ap.add_argument("--warmup-ratio", type=float, default=0.1)  # Daniel's value
    ap.add_argument("--weight-decay", type=float, default=0.01) # Daniel's value
    ap.add_argument("--max-length", type=int, default=2048)     # Daniel's value
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=3407)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[{time.strftime('%H:%M:%S')}] loading model {MODEL_ID} ...", flush=True)
    model, processor = FastModel.from_pretrained(
        model_name=MODEL_ID,
        dtype=None,
        max_seq_length=args.max_length,
        load_in_4bit=True,
        full_finetuning=False,
    )
    print(f"  loaded in cuda; VRAM={torch.cuda.memory_allocated()/1e9:.2f} GB")

    print(f"[{time.strftime('%H:%M:%S')}] applying LoRA r={args.lora_r} alpha={args.lora_alpha} (Daniel recipe) ...", flush=True)
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias="none",
        random_state=args.seed,
        use_rslora=False,
        loftq_config=None,
        target_modules=[
            # Verbatim from Daniel Hanchen's Gemma 3N 4B audio notebook (proven working)
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
            # Audio layers (only the 4 from Daniel's recipe)
            "post", "linear_start", "linear_end",
            "embedding_projection",
        ],
        # Daniel's recipe targets Gemma 3n where embed_audio takes positional args.
        # In Gemma 4, embed_audio is called as `self.embed_audio(inputs_embeds=...)` and
        # PEFT's AuxiliaryTrainingWrapper barfs (TypeError: missing 1 required positional
        # argument: 'x'). Drop it from modules_to_save; lm_head + embed_tokens still
        # captures most of Daniel's "fully train the heads" win.
        modules_to_save=[
            "lm_head",
            "embed_tokens",
        ],
    )

    # Build dataset (lazy audio loading via collator)
    records = load_manifest(args.manifest)
    if args.limit:
        records = records[:args.limit]
    holdout_records = load_manifest(args.holdout_manifest) if args.holdout_manifest else []
    print(f"[{time.strftime('%H:%M:%S')}] train={len(records)}  holdout={len(holdout_records)}", flush=True)

    light = [{"audio_path": r["audio_path"], "transcript": r["transcript"]} for r in records]
    train_ds = Dataset.from_list(light)

    collate_fn = build_collate_fn(processor)

    sft_args = dict(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=args.warmup_ratio,
        learning_rate=args.lr,
        logging_steps=10,
        save_strategy="steps",
        save_steps=500,
        optim="adamw_8bit",
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        seed=args.seed,
        output_dir=args.output_dir,
        report_to="none",
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        dataset_num_proc=2,
        max_length=args.max_length,
        bf16=True,
        fp16=False,
    )
    if args.max_steps > 0:
        sft_args["max_steps"] = args.max_steps
    else:
        sft_args["num_train_epochs"] = args.num_epochs

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        processing_class=processor.tokenizer,
        data_collator=collate_fn,
        args=SFTConfig(**sft_args),
    )

    print(f"[{time.strftime('%H:%M:%S')}] starting training ...", flush=True)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    stats = trainer.train()
    elapsed = time.time() - t0
    peak = torch.cuda.max_memory_reserved() / 1e9

    print(f"\n=== done ===")
    print(f"  elapsed:          {elapsed/60:.1f} min")
    print(f"  peak VRAM:        {peak:.2f} GB")
    print(f"  final train loss: {stats.metrics.get('train_loss')}")

    save_dir = os.path.join(args.output_dir, "lora_final")
    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)
    print(f"  LoRA saved to: {save_dir}")

    # In-process held-out eval (BASE vs AFTER)
    if holdout_records:
        print(f"\n[{time.strftime('%H:%M:%S')}] held-out eval on {len(holdout_records)} clips ...", flush=True)
        model.eval()

        def run_eval(recs):
            outs = []
            for i, r in enumerate(recs):
                try:
                    audio, sr = sf.read(r["audio_path"])
                    if sr != 16000:
                        outs.append({"ref": r["transcript"], "hyp": "<ERROR:bad_sr>",
                                     "severity": r.get("severity")})
                        continue
                    messages = [
                        {"role": "system",
                         "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                        {"role": "user",
                         "content": [
                             {"type": "audio", "audio": audio},
                             {"type": "text",  "text": USER_PROMPT},
                         ]},
                    ]
                    inputs = processor.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=True,
                        return_dict=True, return_tensors="pt",
                    )
                    inputs = {k: v.to(model.device) if hasattr(v,"to") else v
                              for k,v in inputs.items()}
                    with torch.no_grad():
                        out = model.generate(**inputs, max_new_tokens=128, do_sample=False,
                                             temperature=None, top_p=None)
                    input_len = inputs["input_ids"].shape[1]
                    text = processor.decode(out[0, input_len:], skip_special_tokens=True).strip()
                except Exception as e:
                    text = f"<ERROR:{type(e).__name__}>"
                outs.append({"ref": r["transcript"], "hyp": text,
                             "severity": r.get("severity"),
                             "speaker": r.get("participant_id")})
                if (i+1) % 25 == 0:
                    print(f"    {i+1}/{len(recs)} done", flush=True)
            return outs

        print("  Phase A: BASE (LoRA disabled)")
        try:
            with model.disable_adapter():
                base_outs = run_eval(holdout_records)
        except Exception as e:
            print(f"  disable_adapter failed: {e}; skipping base pass")
            base_outs = None

        print("  Phase B: AFTER (LoRA active)")
        after_outs = run_eval(holdout_records)

        def aggregate(outs):
            by_sev = defaultdict(lambda: {"refs":[], "hyps":[]})
            for o in outs:
                if o["hyp"].startswith("<ERROR"): continue
                by_sev[o["severity"] or "unknown"]["refs"].append(o["ref"].lower())
                by_sev[o["severity"] or "unknown"]["hyps"].append(o["hyp"].lower())
            refs = [o["ref"].lower() for o in outs if not o["hyp"].startswith("<ERROR")]
            hyps = [o["hyp"].lower() for o in outs if not o["hyp"].startswith("<ERROR")]
            return {
                "overall_wer": round(jiwer.wer(refs, hyps), 3) if refs else None,
                "per_severity_wer": {k: round(jiwer.wer(v["refs"], v["hyps"]), 3)
                                     for k,v in by_sev.items() if v["refs"]},
                "n_eval": len(refs), "n_errors": sum(1 for o in outs if o["hyp"].startswith("<ERROR")),
            }

        result = {"after": aggregate(after_outs), "after_raw": after_outs}
        if base_outs is not None:
            result["base"] = aggregate(base_outs)
            result["base_raw"] = base_outs
            print(f"\n  Held-out WER  BASE: {result['base']['overall_wer']}  AFTER: {result['after']['overall_wer']}")
            print(f"  Per-severity BASE:  {result['base']['per_severity_wer']}")
            print(f"  Per-severity AFTER: {result['after']['per_severity_wer']}")
        else:
            print(f"\n  Held-out WER  AFTER only: {result['after']['overall_wer']}")

        with open(os.path.join(args.output_dir, "run_summary.json"), "w") as f:
            json.dump({
                "model_id": MODEL_ID,
                "manifest": args.manifest,
                "records": len(records),
                "holdout_records": len(holdout_records),
                "max_steps": args.max_steps,
                "num_epochs": args.num_epochs,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lr": args.lr,
                "warmup_ratio": args.warmup_ratio,
                "weight_decay": args.weight_decay,
                "max_length": args.max_length,
                "elapsed_s": elapsed,
                "peak_vram_gb": peak,
                "train_loss": stats.metrics.get("train_loss"),
                "holdout_eval": result,
            }, f, indent=2)
        print(f"\n  Saved run_summary to {args.output_dir}/run_summary.json")


if __name__ == "__main__":
    main()

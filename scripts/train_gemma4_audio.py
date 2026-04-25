"""Gemma 4 E4B audio LoRA fine-tune — Daniel-recipe wins layered on Gemma 4 collator.

After R1-D attempt 1 (TypeError: AuxiliaryTrainingWrapper missing 'x' — Gemma 4
calls embed_audio with kwarg, PEFT wrapper expects positional) and R1-D
attempt 2 (ValueError: audio features 483840 vs tokens 190 — Daniel's custom
processor() collator doesn't emit Gemma 4's required token_type_ids /
mm_token_type_ids tensors), this version uses Unsloth's collator (correct for
Gemma 4) but keeps Daniel's other proven wins:

  - modules_to_save = [lm_head, embed_tokens]   (NOT embed_audio — Gemma 4
    incompatible with PEFT's AuxiliaryTrainingWrapper for that module)
  - Audio target_modules: 4 modules (post, linear_start, linear_end,
    embedding_projection) instead of our prior 6
  - weight_decay = 0.01, warmup_ratio = 0.1
  - max_length = 2048
  - r = 16, lora_alpha = 16, lora_dropout = 0
  - lr default = 2e-4 (the user-suggested LR for this corrected setup —
    higher than Daniel's 5e-5; the rationale is that fully-trained heads
    via modules_to_save absorb noisy gradient that would otherwise destabilize
    the LoRA at this LR)
  - 3 epochs (vs Daniel's 1) since our dataset (3,956 clips) is small
  - Loss flows on prompt+answer (UnslothVisionDataCollator default) —
    NOT assistant-only like our prior train_gemma4_lora.py
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
from unsloth.trainer import UnslothVisionDataCollator
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--holdout-manifest", default=None)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-steps", type=int, default=-1,
                    help="If -1, use --num-epochs instead.")
    ap.add_argument("--num-epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)         # eff batch 8
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--freeze-audio", action="store_true",
                    help="Drop audio modules from target_modules. Adapts only the language path; "
                         "audio encoder stays at strong pretrained weights. R1-Lang config.")
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
    print(f"  loaded; VRAM={torch.cuda.memory_allocated()/1e9:.2f} GB")

    print(f"[{time.strftime('%H:%M:%S')}] applying LoRA r={args.lora_r} alpha={args.lora_alpha} "
          f"+ modules_to_save=[lm_head, embed_tokens] ...", flush=True)
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
        target_modules=(
            # --freeze-audio: language-only LoRA (audio encoder stays at base weights)
            ["q_proj", "k_proj", "v_proj", "o_proj",
             "gate_proj", "up_proj", "down_proj"]
            if args.freeze_audio else
            # Default: Daniel Hanchen Gemma 3n notebook target_modules (+ 4 audio modules)
            ["q_proj", "k_proj", "v_proj", "o_proj",
             "gate_proj", "up_proj", "down_proj",
             "post", "linear_start", "linear_end",
             "embedding_projection"]
        ),
        # Critical: lm_head + embed_tokens fully trained. embed_audio NOT included
        # because Gemma 4's modeling_gemma4.py:2329 calls it with kwarg
        # `inputs_embeds=` and PEFT's AuxiliaryTrainingWrapper requires positional.
        modules_to_save=[
            "lm_head",
            "embed_tokens",
        ],
    )

    # Lazy-loaded dataset
    records = load_manifest(args.manifest)
    if args.limit:
        records = records[:args.limit]
    holdout_records = load_manifest(args.holdout_manifest) if args.holdout_manifest else []
    print(f"[{time.strftime('%H:%M:%S')}] train={len(records)}  holdout={len(holdout_records)}", flush=True)

    light = [{"audio_path": r["audio_path"], "transcript": r["transcript"]} for r in records]
    train_ds = Dataset.from_list(light)

    def _lazy_messages(batch):
        msgs = []
        for p, t in zip(batch["audio_path"], batch["transcript"]):
            audio, sr = sf.read(p)
            assert sr == 16000, f"expected 16 kHz got {sr} at {p}"
            msgs.append([
                {"role": "system",
                 "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user",
                 "content": [
                     {"type": "audio", "audio": audio},
                     {"type": "text",  "text": USER_PROMPT},
                 ]},
                {"role": "assistant",
                 "content": [{"type": "text", "text": t}]},
            ])
        return {"messages": msgs}

    train_ds.set_transform(_lazy_messages)
    print(f"  dataset (lazy): {len(train_ds)} records, audio loaded on access")

    # CRITICAL: UnslothVisionDataCollator handles Gemma 4's audio token expansion
    # math correctly (emits token_type_ids + mm_token_type_ids). Don't wrap it
    # in AssistantOnlyLossCollator — Daniel's recipe trains on prompt+answer.
    data_collator = UnslothVisionDataCollator(model, processor)

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
        # Required for multimodal — keep mm_token_type_ids etc.
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
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
        data_collator=data_collator,
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

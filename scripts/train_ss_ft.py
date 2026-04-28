"""Per-speaker SS-FT for the B3 vs B4 controlled study.

  B3: --start-from d4b  → load BASE+R1-D4b LoRA, fine-tune the existing adapter
                          on per-speaker clips. Uses --si-lora-dir.
  B4: --start-from base → load BASE Gemma, apply a *fresh* LoRA with the same
                          r/alpha/target_modules/modules_to_save as R1-D4b,
                          then fine-tune from scratch on per-speaker clips.

Same hyperparams (lr, steps, eff batch) for both — that's the whole point of
the controlled comparison.

Filters ss_train.jsonl to --participant-id, saves the speaker-specific
adapter to outputs/ss_ft_{b3,b4}/<participant_id>/lora_final/.
"""
from __future__ import annotations
import os, json, argparse, time

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

MODEL_ID = "unsloth/gemma-4-E4B-it-unsloth-bnb-4bit"
SYSTEM_PROMPT = "You are an assistant that transcribes speech accurately."
USER_PROMPT = "Please transcribe this audio."


def load_manifest(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-from", choices=["d4b", "base"], default="d4b",
                    help="d4b = warm-start from R1-D4b LoRA (B3); base = fresh LoRA on BASE Gemma (B4)")
    ap.add_argument("--si-lora-dir", default=None,
                    help="Path to R1-D4b's lora_final directory (required when --start-from d4b)")
    ap.add_argument("--manifest", required=True,
                    help="ss_train manifest (will be filtered to --participant-id)")
    ap.add_argument("--participant-id", required=True,
                    help="Single speaker to fine-tune for")
    ap.add_argument("--output-dir", required=True,
                    help="Where to save the per-speaker adapter")
    ap.add_argument("--max-steps", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=3407)
    # B4 LoRA hyperparams (must match R1-D4b for fair comparison)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=16)
    args = ap.parse_args()

    if args.start_from == "d4b" and not args.si_lora_dir:
        raise SystemExit("--si-lora-dir is required when --start-from d4b")

    os.makedirs(args.output_dir, exist_ok=True)

    if args.start_from == "d4b":
        # B3: load BASE + R1-D4b LoRA via FastModel (handles Gemma4ClippableLinear)
        print(f"[{time.strftime('%H:%M:%S')}] B3: loading BASE+SI-FT LoRA from {args.si_lora_dir} ...", flush=True)
        model, processor = FastModel.from_pretrained(
            model_name=args.si_lora_dir,
            dtype=None, max_seq_length=args.max_length,
            load_in_4bit=True, full_finetuning=False,
        )
        print(f"  loaded; VRAM={torch.cuda.memory_allocated()/1e9:.2f} GB")
        model.train()
        for n, p in model.named_parameters():
            if "lora_" in n.lower() or n.endswith("modules_to_save.default.weight") or n.endswith("modules_to_save.default.bias"):
                p.requires_grad_(True)
    else:
        # B4: load BASE Gemma, apply fresh LoRA matching R1-D4b's config
        print(f"[{time.strftime('%H:%M:%S')}] B4: loading BASE {MODEL_ID} ...", flush=True)
        model, processor = FastModel.from_pretrained(
            model_name=MODEL_ID,
            dtype=None, max_seq_length=args.max_length,
            load_in_4bit=True, full_finetuning=False,
        )
        print(f"  loaded; VRAM={torch.cuda.memory_allocated()/1e9:.2f} GB")
        print(f"[{time.strftime('%H:%M:%S')}] B4: applying fresh LoRA r={args.lora_r} alpha={args.lora_alpha} ...", flush=True)
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
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
                "post", "linear_start", "linear_end",
                "embedding_projection",
            ],
            modules_to_save=["lm_head", "embed_tokens"],
        )
        model.train()

    # Filter manifest to single speaker
    all_records = load_manifest(args.manifest)
    speaker_records = [r for r in all_records if r.get("participant_id") == args.participant_id]
    print(f"[{time.strftime('%H:%M:%S')}] speaker {args.participant_id}: {len(speaker_records)} clips "
          f"(out of {len(all_records)} total in manifest)")
    if not speaker_records:
        raise SystemExit(f"No clips for participant_id={args.participant_id}")

    light = [{"audio_path": r["audio_path"], "transcript": r["transcript"]} for r in speaker_records]
    train_ds = Dataset.from_list(light)

    def _lazy(batch):
        msgs = []
        for p, t in zip(batch["audio_path"], batch["transcript"]):
            audio, sr = sf.read(p)
            assert sr == 16000, f"expected 16 kHz got {sr} at {p}"
            msgs.append([
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user",
                 "content": [{"type": "audio", "audio": audio},
                             {"type": "text", "text": USER_PROMPT}]},
                {"role": "assistant", "content": [{"type": "text", "text": t}]},
            ])
        return {"messages": msgs}
    train_ds.set_transform(_lazy)

    collator = UnslothVisionDataCollator(model, processor)

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        processing_class=processor.tokenizer,
        data_collator=collator,
        args=SFTConfig(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            warmup_ratio=args.warmup_ratio,
            max_steps=args.max_steps,
            learning_rate=args.lr,
            logging_steps=10,
            save_strategy="no",   # only save final
            optim="adamw_8bit",
            weight_decay=args.weight_decay,
            lr_scheduler_type="cosine",
            seed=args.seed,
            output_dir=args.output_dir,
            report_to="none",
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_length=args.max_length,
            bf16=True, fp16=False,
        ),
    )

    print(f"[{time.strftime('%H:%M:%S')}] training {args.max_steps} steps "
          f"(eff batch {args.batch_size * args.grad_accum}) ...", flush=True)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    stats = trainer.train()
    elapsed = time.time() - t0

    print(f"\n=== {args.participant_id} done ===")
    print(f"  elapsed: {elapsed/60:.1f} min")
    print(f"  peak VRAM: {torch.cuda.max_memory_reserved()/1e9:.2f} GB")
    print(f"  final loss: {stats.metrics.get('train_loss')}")

    save_dir = os.path.join(args.output_dir, "lora_final")
    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)

    # tiny stats record (no full preds — eval is a separate script)
    with open(os.path.join(args.output_dir, "ss_ft_stats.json"), "w") as f:
        json.dump({
            "participant_id": args.participant_id,
            "start_from": args.start_from,
            "n_train_clips": len(speaker_records),
            "max_steps": args.max_steps,
            "lr": args.lr,
            "elapsed_s": elapsed,
            "train_loss": stats.metrics.get("train_loss"),
        }, f, indent=2)
    print(f"  saved to {save_dir}")


if __name__ == "__main__":
    main()

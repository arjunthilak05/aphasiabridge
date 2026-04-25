"""Fine-tune Gemma 4 E4B audio path with LoRA on an audio+transcript manifest.

Follows Unsloth's canonical Gemma4_(E4B)-Audio notebook recipe:
  - FastModel.from_pretrained(..., load_in_4bit=True, full_finetuning=False)
  - FastModel.get_peft_model with target_modules that include audio encoder layers
  - UnslothVisionDataCollator (unified for vision/audio)
  - SFTConfig with dataset_text_field="", skip_prepare_dataset=True

Manifest format (JSONL, one record per clip):
  {audio_path, transcript, ...metadata fields}

Usage:
    .venv/Scripts/python.exe scripts/train_gemma4_lora.py \
        --manifest data/uaspeech_smoketrain_manifest.jsonl \
        --output-dir outputs/lora_smoketest \
        --max-steps 30 \
        --batch-size 1 --grad-accum 8 \
        --lora-r 8 --lora-alpha 16 \
        --lr 5e-5
"""
from __future__ import annotations
import os, sys, json, argparse, time

_WIN_HF = r"D:\Downloads\huggingface_cache"
os.environ.setdefault("HF_HOME", _WIN_HF if os.path.isdir(_WIN_HF) else os.path.expanduser("~/.cache/huggingface"))
os.environ.setdefault("HF_HUB_CACHE", os.path.join(os.environ["HF_HOME"], "hub"))
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"

import torch
torch._dynamo.config.disable = True
torch._dynamo.config.suppress_errors = True
torch._dynamo.config.recompile_limit = 64

from unsloth import FastModel
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig
from datasets import Dataset, Audio
import soundfile as sf
import re
from collections import Counter, defaultdict
import jiwer


class AssistantOnlyLossCollator:
    """Wraps UnslothVisionDataCollator and rewrites `labels` so only the tokens
    AFTER the `<start_of_turn>model\\n` marker contribute to loss. Everything
    else (system prompt, user prompt, audio placeholder tokens) gets -100.

    This is the multimodal equivalent of TRL's DataCollatorForCompletionOnlyLM.
    Essential for audio fine-tuning — without it, the model learns to reproduce
    the system+user prompt instead of transcribing.
    """
    def __init__(self, base_collator, processor):
        self.base = base_collator
        tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        # Gemma 4 chat template uses SPECIAL tokens that differ from the literal
        # string "<start_of_turn>model\n". Compute the real marker by tokenizing
        # a reference template and finding the "model" token position.
        ref_msgs = [
            {"role": "user",      "content": [{"type": "text", "text": "x"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "y"}]},
        ]
        try:
            ref_ids = tok.apply_chat_template(ref_msgs, tokenize=True, add_generation_prompt=False)
            if hasattr(ref_ids, "tolist"): ref_ids = ref_ids.tolist()
            if ref_ids and isinstance(ref_ids[0], list): ref_ids = ref_ids[0]
            # Find 'model' token (id should be same regardless of text length)
            model_token_id = tok("model", add_special_tokens=False).input_ids[-1]
            # The marker is: <turn-start-token>, model, <newline>
            try:
                idx = ref_ids.index(model_token_id)
                start = max(0, idx - 1)
                end = min(len(ref_ids), idx + 2)
                self.marker_ids = ref_ids[start:end]
            except ValueError:
                self.marker_ids = [105, 4368, 107]  # known-good Gemma 4 fallback
        except Exception:
            self.marker_ids = [105, 4368, 107]
        self._debug_logged = False
        print(f"[collator] assistant-start marker = {self.marker_ids} "
              f"({[tok.decode([i]) for i in self.marker_ids]})", flush=True)

    def __call__(self, features):
        batch = self.base(features)
        if "labels" not in batch:
            return batch
        import torch as _t
        labels = batch["labels"].clone()
        input_ids = batch["input_ids"]
        marker = self.marker_ids
        m = len(marker)
        for i in range(labels.size(0)):
            seq = input_ids[i].tolist()
            # Find last occurrence of marker; assistant response starts after it
            last = None
            for j in range(len(seq) - m + 1):
                if seq[j:j+m] == marker:
                    last = j
            if last is None:
                labels[i, :] = -100
            else:
                labels[i, :last + m] = -100
        batch["labels"] = labels
        # One-time debug: how many tokens still contribute to loss
        if not self._debug_logged:
            unmasked = (batch["labels"] != -100).sum().item()
            total = batch["labels"].numel()
            print(f"[collator] first batch: {unmasked}/{total} tokens unmasked "
                  f"({unmasked/total*100:.1f}%) — loss only on assistant response", flush=True)
            self._debug_logged = True
        return batch

MODEL_ID = "unsloth/gemma-4-E4B-it-unsloth-bnb-4bit"

SYSTEM_PROMPT = "You are an assistant that transcribes speech accurately."
USER_PROMPT = "Please transcribe this audio."


def load_manifest(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def to_messages(rec: dict) -> dict:
    # Read audio into numpy array (float64) once — avoids per-epoch disk IO
    audio, sr = sf.read(rec["audio_path"])
    assert sr == 16000, f"expected 16 kHz got {sr} at {rec['audio_path']}"
    return {"messages": [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user",   "content": [
            {"type": "audio", "audio": audio},
            {"type": "text",  "text": USER_PROMPT},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": rec["transcript"]}]},
    ]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lora-r", type=int, default=16)         # Unsloth audio recipe: 8-32 range
    ap.add_argument("--lora-alpha", type=int, default=32)     # alpha = 2*r per Unsloth note
    ap.add_argument("--lora-dropout", type=float, default=0.0) # Unsloth audio notebook uses 0
    ap.add_argument("--lr", type=float, default=5e-5)         # Unsloth audio default; NOT 2e-4
    ap.add_argument("--max-seq-length", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=None, help="Cap number of records for quick tests")
    ap.add_argument("--holdout-frac", type=float, default=0.0,
                    help="If >0, split that fraction of manifest into a held-out set "
                         "and run base vs after-LoRA eval on it in the same process")
    ap.add_argument("--holdout-seed", type=int, default=1337)
    ap.add_argument("--holdout-manifest", default=None,
                    help="Optional explicit held-out JSONL. Overrides --holdout-frac. "
                         "Training uses --manifest as-is; eval uses this file.")
    ap.add_argument("--skip-post-eval", action="store_true",
                    help="Skip the UASpeech Windows-only post-training eval sweep (H100 runs)")
    ap.add_argument("--resume-from", default=None,
                    help="Checkpoint directory to resume training from (R3 curriculum phase 2)")
    ap.add_argument("--full-finetune", action="store_true",
                    help="Full fine-tune instead of LoRA: load_in_4bit=False, full_finetuning=True, "
                         "skip get_peft_model. Use lower LR (~3e-5). R2 of H100 runbook.")
    ap.add_argument("--gradient-checkpointing", action="store_true",
                    help="Trade compute for activation memory (~10 GB savings, ~25%% slower). "
                         "Required for full-FT in DDP mode where each rank holds the full model.")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[{time.strftime('%H:%M:%S')}] loading model {MODEL_ID} "
          f"(full_finetune={args.full_finetune}) ...", flush=True)
    model, processor = FastModel.from_pretrained(
        model_name=MODEL_ID,
        dtype=None,
        max_seq_length=args.max_seq_length,
        load_in_4bit=not args.full_finetune,
        full_finetuning=args.full_finetune,
    )
    print(f"  loaded in cuda; VRAM={torch.cuda.memory_allocated()/1e9:.2f} GB")

    if not args.full_finetune:
        print(f"[{time.strftime('%H:%M:%S')}] applying LoRA (r={args.lora_r}, alpha={args.lora_alpha}) ...", flush=True)
        model = FastModel.get_peft_model(
            model,
            finetune_vision_layers=False,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
            # Verbatim target_modules from official Unsloth Gemma4_(E4B)-Audio.ipynb
            # (cell 18). Two prior runs that used a hand-crafted v2 list with
            # `.linear` suffix collapsed catastrophically (WER 0.66 -> 5.13 / 3.02);
            # the model learned the language prior and stopped attending to audio.
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
                # Audio layers (Unsloth canonical list)
                "post", "linear_start", "linear_end",
                "embedding_projection",
                "ffw_layer_1", "ffw_layer_2",
                "output_proj",
            ],
        )
    else:
        print(f"[{time.strftime('%H:%M:%S')}] FULL FINE-TUNE mode — no LoRA adapters. "
              f"All parameters trainable.", flush=True)

    # Load manifest and convert to Dataset
    records = load_manifest(args.manifest)
    if args.limit:
        records = records[:args.limit]

    # Optionally split a held-out set for source-domain eval (e.g. APROCSA test)
    holdout_records = []
    if args.holdout_manifest:
        holdout_records = load_manifest(args.holdout_manifest)
        print(f"[{time.strftime('%H:%M:%S')}] explicit holdout: train={len(records)}  holdout={len(holdout_records)}")
    elif args.holdout_frac > 0:
        import random as _r
        rng = _r.Random(args.holdout_seed)
        shuffled = records[:]
        rng.shuffle(shuffled)
        n_hold = max(1, int(len(shuffled) * args.holdout_frac))
        holdout_records = shuffled[:n_hold]
        records = shuffled[n_hold:]
        print(f"[{time.strftime('%H:%M:%S')}] split: train={len(records)}  holdout={len(holdout_records)}")

    print(f"[{time.strftime('%H:%M:%S')}] building dataset from {len(records)} records ...", flush=True)

    # Lazy: only keep path+transcript columns in the Dataset; load audio per-batch
    # on access via set_transform. Keeps RAM usage low for large manifests.
    light_records = [{"audio_path": r["audio_path"], "transcript": r["transcript"]}
                     for r in records]
    ds = Dataset.from_list(light_records)

    def _lazy_messages(batch):
        # batch is a dict of lists (one entry per column)
        msgs = []
        for p, t in zip(batch["audio_path"], batch["transcript"]):
            audio, sr = sf.read(p)
            assert sr == 16000, f"expected 16 kHz got {sr} at {p}"
            msgs.append([
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {"role": "user",   "content": [
                    {"type": "audio", "audio": audio},
                    {"type": "text",  "text": USER_PROMPT},
                ]},
                {"role": "assistant", "content": [{"type": "text", "text": t}]},
            ])
        return {"messages": msgs}

    ds.set_transform(_lazy_messages)
    print(f"  dataset (lazy): {len(ds)} records, audio loaded on access")

    print(f"[{time.strftime('%H:%M:%S')}] setting up SFTTrainer ...", flush=True)
    base_collator = UnslothVisionDataCollator(model, processor)
    masked_collator = AssistantOnlyLossCollator(base_collator, processor)
    trainer = SFTTrainer(
        model=model,
        train_dataset=ds,
        processing_class=processor.tokenizer,
        data_collator=masked_collator,
        args=SFTConfig(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            warmup_ratio=0.03,
            max_steps=args.max_steps,
            learning_rate=args.lr,
            logging_steps=1,
            save_strategy="steps",
            save_steps=max(args.max_steps // 3, 10),
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="cosine",
            seed=3407,
            output_dir=args.output_dir,
            report_to="none",
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_length=args.max_seq_length,
            bf16=True,
            fp16=False,
            gradient_checkpointing=args.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False} if args.gradient_checkpointing else None,
        ),
    )

    print(f"[{time.strftime('%H:%M:%S')}] starting training ({args.max_steps} steps"
          f"{', resume=' + args.resume_from if args.resume_from else ''}) ...", flush=True)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    stats = trainer.train(resume_from_checkpoint=args.resume_from) if args.resume_from else trainer.train()
    elapsed = time.time() - t0
    peak = torch.cuda.max_memory_reserved() / 1e9

    # Under DDP/accelerate launch, only rank 0 should print + save + eval.
    # Single-GPU runs have LOCAL_RANK unset, so this is a no-op there.
    is_main = int(os.environ.get("LOCAL_RANK", 0)) == 0

    if is_main:
        print(f"\n=== done ===")
        print(f"  elapsed:          {elapsed/60:.1f} min ({elapsed:.1f}s)")
        print(f"  peak VRAM:        {peak:.2f} GB")
        print(f"  final train loss: {stats.metrics.get('train_loss')}")

    # Save final weights — LoRA adapter or full merged, depending on mode
    # All ranks must call save_pretrained for trainer's internal collective sync;
    # but only rank 0 writes the final files (HF trainer handles this).
    out_name = "merged" if args.full_finetune else "lora_final"
    save_dir = os.path.join(args.output_dir, out_name)
    if is_main:
        model.save_pretrained(save_dir)
        processor.save_pretrained(save_dir)
        print(f"  {'full-FT weights' if args.full_finetune else 'LoRA'} saved to: {save_dir}")

    # Post-train eval is single-GPU only — short-circuit on non-main ranks.
    if not is_main:
        return

    # === In-process source-domain held-out eval (APROCSA base vs LoRA) ===
    holdout_eval = {}
    if holdout_records:
        print(f"\n[{time.strftime('%H:%M:%S')}] source-domain held-out eval on {len(holdout_records)} clips ...", flush=True)
        model.eval()
        import jiwer as _jw

        def _run(recs):
            outs = []
            for i, r in enumerate(recs):
                try:
                    audio, sr = sf.read(r["audio_path"])
                    if sr != 16000: continue
                    messages = [
                        {"role":"system","content":[{"type":"text","text":SYSTEM_PROMPT}]},
                        {"role":"user","content":[
                            {"type":"audio","audio":audio},
                            {"type":"text","text":USER_PROMPT},
                        ]},
                    ]
                    inputs = processor.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=True,
                        return_dict=True, return_tensors="pt",
                    )
                    inputs = {k: v.to(model.device) if hasattr(v,"to") else v for k,v in inputs.items()}
                    with torch.no_grad():
                        out = model.generate(**inputs, max_new_tokens=64, do_sample=False,
                                             temperature=None, top_p=None)
                    input_len = inputs["input_ids"].shape[1]
                    text = processor.decode(out[0, input_len:], skip_special_tokens=True).strip()
                except Exception as e:
                    text = f"<ERROR:{type(e).__name__}>"
                outs.append({"ref": r["transcript"], "hyp": text,
                             "severity": r.get("severity"), "speaker": r.get("participant_id") or r.get("speaker")})
                if (i+1) % 25 == 0:
                    print(f"    {i+1}/{len(recs)} done", flush=True)
            return outs

        # Phase A: base model (LoRA disabled)
        print("  Phase A: BASE (LoRA disabled)")
        try:
            with model.disable_adapter():
                base_outs = _run(holdout_records)
        except Exception as e:
            print(f"  disable_adapter failed: {e}; skipping base pass")
            base_outs = None

        # Phase B: LoRA fine-tuned
        print("  Phase B: AFTER (LoRA active)")
        after_outs = _run(holdout_records)

        def _agg(outs):
            by_sev = defaultdict(lambda: {"refs":[], "hyps":[]})
            for o in outs:
                if o["hyp"].startswith("<ERROR"): continue
                by_sev[o["severity"] or "unknown"]["refs"].append(o["ref"].lower())
                by_sev[o["severity"] or "unknown"]["hyps"].append(o["hyp"].lower())
            overall_refs = [r.lower() for o in outs for r in [o["ref"]] if not o["hyp"].startswith("<ERROR")]
            overall_hyps = [o["hyp"].lower() for o in outs if not o["hyp"].startswith("<ERROR")]
            return {
                "overall_wer": round(_jw.wer(overall_refs, overall_hyps), 3) if overall_refs else None,
                "per_severity_wer": {k: round(_jw.wer(v["refs"], v["hyps"]), 3) for k,v in by_sev.items() if v["refs"]},
            }

        holdout_eval["after"] = _agg(after_outs)
        holdout_eval["after_raw"] = after_outs
        if base_outs is not None:
            holdout_eval["base"] = _agg(base_outs)
            holdout_eval["base_raw"] = base_outs
            print(f"\n  APROCSA held-out WER — BASE: {holdout_eval['base']['overall_wer']}  AFTER: {holdout_eval['after']['overall_wer']}")
            print(f"  Per-severity BASE:  {holdout_eval['base']['per_severity_wer']}")
            print(f"  Per-severity AFTER: {holdout_eval['after']['per_severity_wer']}")
        else:
            print(f"  APROCSA held-out WER — AFTER only: {holdout_eval['after']['overall_wer']}")

    # === UASpeech cross-domain eval (secondary sanity check) ===
    # Windows-only paths — skip on H100. `--skip-post-eval` short-circuits + writes
    # the run_summary with just the holdout eval results.
    per_sev_after: dict = {}
    per_sev_before: dict = {}
    eval_results: list = []
    gt: dict = {}
    if args.skip_post_eval:
        print(f"\n[{time.strftime('%H:%M:%S')}] --skip-post-eval set; skipping UASpeech sweep", flush=True)
        with open(os.path.join(args.output_dir, "run_summary.json"), "w", encoding="utf-8") as f:
            json.dump({
                "model_id": MODEL_ID,
                "manifest": args.manifest,
                "records": len(records),
                "holdout_records": len(holdout_records),
                "max_steps": args.max_steps,
                "batch_size": args.batch_size,
                "grad_accum": args.grad_accum,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "lr": args.lr,
                "full_finetune": args.full_finetune,
                "resume_from": args.resume_from,
                "elapsed_s": elapsed,
                "peak_vram_gb": peak,
                "metrics": stats.metrics,
                "aprocsa_holdout_eval": holdout_eval,
            }, f, indent=2)
        print(f"Saved run_summary (holdout-only) to {args.output_dir}/run_summary.json")
        return
    # We run the SAME speaker x digit sweep as baseline_zeroshot_uaspeech.py,
    # avoiding the LoRA-reload audio patch bug by staying in this process.
    print(f"\n[{time.strftime('%H:%M:%S')}] running post-training eval ...", flush=True)
    model.eval()
    ROOT = r"D:\Downloads\speech_data"
    SEVERITY_MAP = {
        "F02":"Low","F03":"Very Low","F04":"Low","F05":"Mid",
        "M01":"Very Low","M04":"Very Low","M05":"Mid","M07":"Low",
        "M08":"Mid","M09":"High","M10":"Mid","M11":"Very Low",
        "M12":"Low","M14":"High","M16":"Low",
    }
    CONTROLS = ["CF02","CF03","CF04","CF05","CM01","CM04","CM05","CM06","CM08","CM09","CM10","CM12","CM13"]
    DIGITS = [f"D{i}" for i in range(1, 10)]
    CLEAN = re.compile(r"[^a-z\s']")
    eval_results = []
    speakers = [(s, os.path.join(ROOT,"control",s), "control") for s in CONTROLS] + \
               [(s, os.path.join(ROOT,s), SEVERITY_MAP[s]) for s in SEVERITY_MAP]
    for spk, sdir, sev in speakers:
        if not os.path.isdir(sdir): continue
        for d in DIGITS:
            wav = os.path.join(sdir, f"{spk}_B1_{d}_M2.wav")
            if not os.path.exists(wav): continue
            audio, sr = sf.read(wav)
            messages = [
                {"role":"system","content":[{"type":"text","text":SYSTEM_PROMPT}]},
                {"role":"user","content":[
                    {"type":"audio","audio":audio},
                    {"type":"text","text":USER_PROMPT},
                ]},
            ]
            inputs = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt",
            )
            inputs = {k: v.to(model.device) if hasattr(v,"to") else v for k,v in inputs.items()}
            try:
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=16, do_sample=False,
                                         temperature=None, top_p=None)
                input_len = inputs["input_ids"].shape[1]
                text = processor.decode(out[0, input_len:], skip_special_tokens=True).strip()
            except Exception as e:
                text = f"<ERROR:{type(e).__name__}>"
            eval_results.append({"speaker":spk,"severity":sev,"digit":d,"output":text,
                                 "output_clean":CLEAN.sub(" ", text.lower()).strip()})
    # Consensus GT from controls
    gt = {}
    for d in DIGITS:
        ctrl = [r["output_clean"] for r in eval_results if r["digit"]==d and r["severity"]=="control" and not r["output"].startswith("<ERROR")]
        if ctrl: gt[d] = Counter(ctrl).most_common(1)[0][0]
    # Per-severity WER
    bucket = defaultdict(lambda: {"refs":[],"hyps":[]})
    for r in eval_results:
        ref = gt.get(r["digit"])
        if not ref or r["output"].startswith("<ERROR"): continue
        bucket[r["severity"]]["refs"].append(ref)
        bucket[r["severity"]]["hyps"].append(r["output_clean"])
    per_sev_after = {k: round(jiwer.wer(v["refs"], v["hyps"]), 3) for k,v in bucket.items() if v["refs"]}

    # Load baseline for comparison
    per_sev_before = {}
    base_p = "outputs/baseline_uaspeech_digits.json"
    if os.path.exists(base_p):
        with open(base_p, "r", encoding="utf-8") as f:
            per_sev_before = {k:v["wer"] for k,v in json.load(f).get("per_severity_wer",{}).items()}

    # Save JSON FIRST so any downstream print failure can't lose data
    with open(os.path.join(args.output_dir, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump({
            "model_id": MODEL_ID,
            "manifest": args.manifest,
            "records": len(records),
            "holdout_records": len(holdout_records),
            "max_steps": args.max_steps,
            "batch_size": args.batch_size,
            "grad_accum": args.grad_accum,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lr": args.lr,
            "elapsed_s": elapsed,
            "peak_vram_gb": peak,
            "metrics": stats.metrics,
            "uaspeech_eval_per_severity_wer_before": per_sev_before,
            "uaspeech_eval_per_severity_wer_after": per_sev_after,
            "uaspeech_eval_raw": eval_results,
            "uaspeech_eval_ground_truth": gt,
            "aprocsa_holdout_eval": holdout_eval,
        }, f, indent=2)
    print(f"\nSaved run_summary with eval to {args.output_dir}/run_summary.json")

    print(f"\n=== BEFORE vs AFTER ({args.max_steps}-step LoRA) ===")
    print(f"{'Severity':12s}  {'Before':>8s}  {'After':>8s}  {'Delta':>8s}")
    for sev in ["control","High","Mid","Low","Very Low"]:
        b = per_sev_before.get(sev); a = per_sev_after.get(sev)
        if a is None: continue
        if b is None:
            print(f"  {sev:12s}  {'--':>8s}  {a:>8.3f}  {'--':>8s}")
        else:
            print(f"  {sev:12s}  {b:>8.3f}  {a:>8.3f}  {a-b:>+8.3f}")


if __name__ == "__main__":
    main()

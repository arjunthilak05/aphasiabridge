"""Phase 1: Whisper-Large-v3 fine-tune on AphasiaBank paper-recipe data.

The single highest-EV experiment in Tier-3 — the literature on
disordered-speech ASR converges on "stronger ASR backbone + augmentation +
curriculum + multi-seed + merging beats decode tricks". This is the SI
backbone for the rest of the Tier-3 stack.

Loads our manifest format (audio_path, transcript, severity), runs HF
Seq2SeqTrainer with WhisperFeatureExtractor, optional augmentation chain
(--augment {none,specaug,full}), optional severity curriculum
(--severity-stages mild|all).
"""
from __future__ import annotations
import argparse, json, os, time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import soundfile as sf
import jiwer
from datasets import Dataset
from transformers import (
    WhisperFeatureExtractor, WhisperTokenizer, WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
)
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

from audio_augmentations import (
    augment_audio_waveform, SpecAugmentMask
)


def load_manifest(p: str):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


@dataclass
class WhisperCollator:
    processor: WhisperProcessor
    augment: str = "none"      # 'none' | 'specaug' | 'full'
    rng: Any = None
    logged_label_debug: bool = False

    def __post_init__(self):
        import random
        self.rng = self.rng or random.Random(42)
        self.specaug = SpecAugmentMask()

    def __call__(self, features):
        # features: [{"audio_path": ..., "transcript": ...}]
        audios, transcripts = [], []
        for ex in features:
            audio, sr = sf.read(ex["audio_path"])
            assert sr == 16000, f"got {sr} for {ex['audio_path']}"
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if self.augment == "full":
                audio = augment_audio_waveform(audio, sr=16000, rng=self.rng)
            audios.append(audio.astype(np.float32))
            transcripts.append(ex["transcript"])

        # Mel features
        feats = self.processor.feature_extractor(
            audios, sampling_rate=16000, return_tensors="pt"
        )
        input_features = feats["input_features"]

        # SpecAugment on log-mel
        if self.augment in ("specaug", "full"):
            arr = input_features.numpy()
            for b in range(arr.shape[0]):
                arr[b] = self.specaug.apply(arr[b], rng=self.rng)
            input_features = torch.from_numpy(arr)

        # Tokenize labels with the standard Whisper recipe: mask padding using
        # the tokenizer attention mask, then remove the decoder-start token if
        # the tokenizer added it to every target.
        label_batch = self.processor.tokenizer(
            transcripts, padding=True, return_tensors="pt"
        )
        labels = label_batch["input_ids"].masked_fill(
            label_batch["attention_mask"].ne(1), -100
        )
        start_ids = [
            self.processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>"),
            self.processor.tokenizer.bos_token_id,
        ]
        start_ids = [x for x in start_ids if x is not None and x != self.processor.tokenizer.unk_token_id]
        if (
            labels.shape[1] > 0
            and start_ids
            and any((labels[:, 0] == start_id).all().cpu().item() for start_id in start_ids)
        ):
            labels = labels[:, 1:]

        if not self.logged_label_debug:
            debug_labels = labels[0].clone()
            debug_labels[debug_labels == -100] = self.processor.tokenizer.pad_token_id
            print(
                "[collator] first label decodes to: "
                f"{self.processor.tokenizer.decode(debug_labels, skip_special_tokens=False)[:300]}",
                flush=True,
            )
            self.logged_label_debug = True

        return {"input_features": input_features, "labels": labels}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="paper-recipe si_train")
    ap.add_argument("--val-manifest", default=None, help="optional dev set for eval-during-train")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model", default="openai/whisper-large-v3")
    ap.add_argument("--max-steps", type=int, default=5000)
    ap.add_argument("--warmup-steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--augment", choices=["none", "specaug", "full"], default="none")
    ap.add_argument("--severity-stages", default=None,
                    help="comma-separated stages, each comma-separated severities. "
                         "e.g. 'mild|all' for curriculum: stage1=mild only, stage2=all")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"[{time.strftime('%H:%M:%S')}] loading {args.model} ...", flush=True)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model)
    tokenizer = WhisperTokenizer.from_pretrained(args.model, language="en", task="transcribe")
    processor = WhisperProcessor.from_pretrained(args.model, language="en", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.float32,
    )
    # English-only generation. HF Whisper-FT blog uses these exact strings.
    # Don't clear suppress_tokens — Whisper relies on them to prevent rambling.
    model.generation_config.language = "english"
    model.generation_config.task = "transcribe"
    forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
    model.generation_config.forced_decoder_ids = forced_decoder_ids
    model.config.forced_decoder_ids = forced_decoder_ids

    # Load + filter manifest
    records = load_manifest(args.manifest)
    if args.limit:
        records = records[: args.limit]
    print(f"[{time.strftime('%H:%M:%S')}] training on {len(records)} clips", flush=True)

    # Curriculum: split into stages by severity
    stages = []
    if args.severity_stages:
        for stage in args.severity_stages.split("|"):
            keep = stage.strip()
            if keep == "all":
                stages.append(records[:])
            else:
                allowed = set(keep.split(","))
                stages.append([r for r in records if (r.get("severity") or "unknown") in allowed])
    else:
        stages = [records]
    for i, s in enumerate(stages):
        print(f"  stage {i}: {len(s)} clips")

    collator = WhisperCollator(processor=processor, augment=args.augment)

    val_dataset = None
    if args.val_manifest:
        val_records = load_manifest(args.val_manifest)
        if args.limit:
            val_records = val_records[: max(1, args.limit // 5)]
        val_dataset = Dataset.from_list([
            {"audio_path": r["audio_path"], "transcript": r["transcript"]}
            for r in val_records
        ])

    en_norm = EnglishTextNormalizer({})

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        # replace -100 with pad
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        # Whisper-EN normalize
        pred_norm = [en_norm(s) for s in pred_str]
        label_norm = [en_norm(s) for s in label_str]
        # filter empties
        pairs = [(r, h) for r, h in zip(label_norm, pred_norm) if r and h]
        if not pairs:
            return {"wer": 1.0}
        return {"wer": float(jiwer.wer([r for r, _ in pairs], [h for _, h in pairs]))}

    # Train each stage in sequence
    for stage_idx, stage_records in enumerate(stages):
        print(f"\n[{time.strftime('%H:%M:%S')}] === stage {stage_idx + 1} / {len(stages)} ===")
        stage_ds = Dataset.from_list([
            {"audio_path": r["audio_path"], "transcript": r["transcript"]}
            for r in stage_records
        ])
        # Steps per stage scaled by stage size (rough heuristic)
        if len(stages) > 1:
            stage_steps = max(500, int(args.max_steps * len(stage_records) / sum(len(s) for s in stages)))
        else:
            stage_steps = args.max_steps

        train_args = Seq2SeqTrainingArguments(
            output_dir=os.path.join(args.output_dir, f"stage{stage_idx}"),
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            warmup_steps=args.warmup_steps if stage_idx == 0 else 0,
            max_steps=stage_steps,
            gradient_checkpointing=True,
            fp16=True,
            eval_strategy="steps" if val_dataset is not None else "no",
            eval_steps=1000,
            per_device_eval_batch_size=8,
            predict_with_generate=True,
            generation_max_length=200,
            save_steps=1000,
            logging_steps=50,
            report_to=["none"],
            load_best_model_at_end=True if val_dataset is not None else False,
            metric_for_best_model="wer", greater_is_better=False,
            seed=args.seed,
            data_seed=args.seed,
            remove_unused_columns=False,
        )

        trainer = Seq2SeqTrainer(
            args=train_args,
            model=model,
            train_dataset=stage_ds,
            eval_dataset=val_dataset,
            data_collator=collator,
            compute_metrics=compute_metrics if val_dataset is not None else None,
            processing_class=processor.feature_extractor,
        )
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        trainer.train()
        elapsed = time.time() - t0
        peak = torch.cuda.max_memory_reserved() / 1e9
        print(f"\n  stage {stage_idx} done in {elapsed/60:.1f} min, peak VRAM {peak:.2f} GB")

    save_dir = os.path.join(args.output_dir, "final")
    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)
    print(f"\n[{time.strftime('%H:%M:%S')}] saved Whisper FT to {save_dir}")


if __name__ == "__main__":
    main()

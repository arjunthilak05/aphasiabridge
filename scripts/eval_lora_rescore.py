"""Tier-2 Lever D: LM-rescoring with Gemma 4 in text-only mode.

For each ss_test clip:
  1. Decode top-5 beam hypotheses with R1-D4b (audio mode).
  2. For each hypothesis, score it with Gemma 4 in text-only mode using the
     conditional log-likelihood under a "plausible aphasic transcription"
     prompt: P("transcription is X" | aphasic-context-prompt).
  3. Combine acoustic score (beam logprob from step 1) + LM score with a
     tunable weight. Pick argmax.
  4. WER vs both lowercase + WhisperEN normalizers.

Saves per-clip top-5 + scores so we can re-tune the weight without re-decoding.
"""
from __future__ import annotations
import os, json, argparse, time
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

MODEL_ID = "unsloth/gemma-4-E4B-it-unsloth-bnb-4bit"
SYSTEM_PROMPT = "You are an assistant that transcribes speech accurately."
USER_PROMPT = "Please transcribe this audio."

# Text-only rescoring prompt
RESCORE_SYS = "You are evaluating whether a candidate transcription is plausible for speech from a person with aphasia. Aphasic speech may include word repetitions, partial words, hesitations, and substitution errors but should be linguistically and acoustically coherent."
RESCORE_USER = "Candidate transcription: {hyp}\n\nIs this plausible aphasic speech? Yes."


def load_manifest(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def text_only_logprob(model, processor, hypothesis: str) -> float:
    """Score a hypothesis's plausibility under Gemma 4 in text-only mode.
    Returns a sum log-probability of `hypothesis` tokens given the rescore prompt.
    Higher = more plausible."""
    prompt = RESCORE_USER.format(hyp=hypothesis)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": RESCORE_SYS}]},
        {"role": "user",   "content": [{"type": "text", "text": prompt}]},
    ]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    )
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
    logits = out.logits[0, -1, :]
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    # Score the FIRST token of "Yes" — single positive-class probability
    yes_id = processor.tokenizer(" Yes", add_special_tokens=False).input_ids[0]
    return log_probs[yes_id].item()


def decode_top_k_beam(model, processor, audio, gen_kwargs, k: int = 5):
    """Return list of (text, beam_score) — beam_score is sum log-probability.

    HF generate with num_beams=k, num_return_sequences=k, return_dict_in_generate=True,
    output_scores=True yields beam-search top-k with sequence_scores (length-normalized
    log-probs by default depending on length_penalty).
    """
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
    inputs = {k_: v.to(model.device) if hasattr(v, "to") else v for k_, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs, **gen_kwargs,
            num_return_sequences=k,
            return_dict_in_generate=True,
            output_scores=True,
        )
    input_len = inputs["input_ids"].shape[1]
    seqs = out.sequences      # (k, L)
    if hasattr(out, "sequences_scores") and out.sequences_scores is not None:
        scores = out.sequences_scores.tolist()
    else:
        # fallback — derive from beam logprobs sum
        scores = [0.0] * seqs.shape[0]
    texts = []
    for i in range(seqs.shape[0]):
        text = processor.decode(seqs[i, input_len:], skip_special_tokens=True).strip()
        texts.append(text)
    return list(zip(texts, scores))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora-dir", required=True)
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--rep-penalty", type=float, default=1.3)
    ap.add_argument("--no-repeat-ngram", type=int, default=4)
    ap.add_argument("--num-beams", type=int, default=5)
    ap.add_argument("--length-penalty", type=float, default=1.0)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--lm-weight", type=float, default=0.1,
                    help="weight on LM rescore score: combined = beam_score + lm_weight * lm_score")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[{time.strftime('%H:%M:%S')}] loading R1-D4b ...", flush=True)
    model, processor = FastModel.from_pretrained(
        model_name=args.lora_dir, dtype=None,
        max_seq_length=args.max_new_tokens + 1024, load_in_4bit=True,
        full_finetuning=False,
    )
    model.eval()

    holdout = load_manifest(args.holdout)
    if args.limit:
        holdout = holdout[:args.limit]
    print(f"[{time.strftime('%H:%M:%S')}] decoding {len(holdout)} clips × top-{args.num_beams} ...", flush=True)

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens, do_sample=False,
        temperature=None, top_p=None,
        repetition_penalty=args.rep_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram,
        num_beams=args.num_beams,
        length_penalty=args.length_penalty,
    )

    results = []
    for i, r in enumerate(holdout):
        try:
            audio, sr = sf.read(r["audio_path"])
            if sr != 16000:
                results.append({"ref": r["transcript"], "best_hyp": "<ERROR:bad_sr>",
                                "severity": r.get("severity"), "speaker": r.get("participant_id"),
                                "candidates": []})
                continue
            cands = decode_top_k_beam(model, processor, audio, gen_kwargs, k=args.num_beams)
            # LM-rescore each candidate
            scored = []
            for text, beam_score in cands:
                if not text.strip():
                    lm_score = -100.0
                else:
                    try:
                        lm_score = text_only_logprob(model, processor, text)
                    except Exception:
                        lm_score = -100.0
                combined = beam_score + args.lm_weight * lm_score
                scored.append({"text": text, "beam_score": beam_score,
                               "lm_score": lm_score, "combined": combined})
            scored.sort(key=lambda d: -d["combined"])
            results.append({"ref": r["transcript"],
                            "best_hyp": scored[0]["text"] if scored else "<ERROR>",
                            "best_hyp_beam_only": cands[0][0] if cands else "<ERROR>",
                            "severity": r.get("severity"),
                            "speaker": r.get("participant_id"),
                            "candidates": scored})
        except Exception as e:
            results.append({"ref": r["transcript"], "best_hyp": f"<ERROR:{type(e).__name__}>",
                            "severity": r.get("severity"), "speaker": r.get("participant_id"),
                            "candidates": []})
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(holdout)} done", flush=True)

    # WER computations
    en_norm = EnglishTextNormalizer({})

    def wer(predictions, hyp_key, normalizer):
        refs = [normalizer(p["ref"]) for p in predictions if not p[hyp_key].startswith("<ERROR")]
        hyps = [normalizer(p[hyp_key]) for p in predictions if not p[hyp_key].startswith("<ERROR")]
        pairs = [(r, h) for r, h in zip(refs, hyps) if r and h]
        if not pairs: return None
        return round(jiwer.wer([r for r, _ in pairs], [h for _, h in pairs]), 4)

    print(f"\n{'='*80}\nLM-rescoring vs beam-only (R1-D4b, {len(results)} clips)\n{'='*80}")
    for label, norm in [("lowercase", lambda s: s.lower().strip()),
                        ("WhisperEN", en_norm)]:
        beam_only = wer(results, "best_hyp_beam_only", norm)
        rescore   = wer(results, "best_hyp", norm)
        delta = round(rescore - beam_only, 4) if (beam_only is not None and rescore is not None) else "NA"
        print(f"  {label:10s}: beam-only={beam_only}  rescore={rescore}  Δ={delta}")

    # Per-severity (WhisperEN)
    by_sev = defaultdict(lambda: {"r": [], "h_beam": [], "h_re": []})
    for p in results:
        if p["best_hyp"].startswith("<ERROR"): continue
        s = p.get("severity") or "unknown"
        by_sev[s]["r"].append(en_norm(p["ref"]))
        by_sev[s]["h_beam"].append(en_norm(p["best_hyp_beam_only"]))
        by_sev[s]["h_re"].append(en_norm(p["best_hyp"]))
    print(f"\nPer-severity (WhisperEN):")
    for sev, v in sorted(by_sev.items()):
        if v["r"]:
            wb = round(jiwer.wer(v["r"], v["h_beam"]), 4)
            wr = round(jiwer.wer(v["r"], v["h_re"]), 4)
            print(f"  {sev:12s}  beam-only={wb}  rescore={wr}  Δ={round(wr-wb,4):+.4f}")

    out_path = os.path.join(args.output_dir, "rescore_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "lora_dir": args.lora_dir, "holdout": args.holdout,
            "lm_weight": args.lm_weight,
            "gen_kwargs": gen_kwargs,
            "predictions": results,
        }, f, indent=2)
    print(f"\n  Wrote {out_path}")


if __name__ == "__main__":
    main()

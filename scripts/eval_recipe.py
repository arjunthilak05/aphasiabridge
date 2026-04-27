"""Phase 0b: locked eval recipe — one source of truth.

Every Tier-3 evaluation script imports from here so we never accidentally
report a number under different decode kwargs / normalizer / clip-set.

Usage:
    from scripts.eval_recipe import (
        LOCKED_GEN_KWARGS, normalizer, bootstrap_wer_ci,
        SS_TEST_PATH, pad_audio_to_safe_length,
    )

The 220-clip ss_test path and the headline normalizer are pinned here.
"""
from __future__ import annotations
import os
from collections import defaultdict
from typing import Iterable

import numpy as np
import jiwer
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

# --- Paths ----------------------------------------------------------------
APHASIA_ROOT = os.environ.get("APHASIA_ROOT",
                              "/data/kl_dev/dataset_wan/data_gemma/aphasiabank")
SS_TEST_PATH = os.path.join(APHASIA_ROOT, "aphasiabank_ss_test.jsonl")
SS_TRAIN_PATH = os.path.join(APHASIA_ROOT, "aphasiabank_ss_train.jsonl")
SS_VAL_PATH = os.path.join(APHASIA_ROOT, "aphasiabank_ss_val.jsonl")
SI_TRAIN_PATH = os.path.join(APHASIA_ROOT, "aphasiabank_si_train.jsonl")
SI_VAL_PATH = os.path.join(APHASIA_ROOT, "aphasiabank_si_val.jsonl")

# --- Locked decoding ------------------------------------------------------
LOCKED_GEN_KWARGS = dict(
    max_new_tokens=200,
    do_sample=False,
    temperature=None, top_p=None,
    num_beams=5,
    length_penalty=1.0,
    repetition_penalty=1.3,
    no_repeat_ngram_size=4,
)

# Whisper-Large-v3 aggressive (the comparator) uses these (note length_penalty=0.6):
WHISPER_AGGRESSIVE_GEN = dict(
    max_new_tokens=256,
    do_sample=False,
    num_beams=5,
    length_penalty=0.6,
    repetition_penalty=1.3,
    no_repeat_ngram_size=4,
)

# --- Normalizer -----------------------------------------------------------
_en_norm = EnglishTextNormalizer({})

def normalizer(s: str) -> str:
    """The single canonical normalizer for all reported numbers in Tier-3.
    Whisper EnglishTextNormalizer matches Jiang et al. 2026 exactly."""
    return _en_norm(s).strip()

def lowercase_norm(s: str) -> str:
    """Legacy lowercase-only normalizer for cross-comparison with Tier-1/2."""
    return s.lower().strip()

# --- Audio padding (Phase 0a) --------------------------------------------
def pad_audio_to_safe_length(audio: np.ndarray, sr: int = 16000,
                             quantum_s: float = 1.0) -> np.ndarray:
    """Pad audio to next multiple of `quantum_s` seconds.

    Works around Gemma 4's audio-token-alignment edge case (24 NEURAL
    clips on ss_test). Whisper baselines unaffected; safe to apply
    universally."""
    target = int(np.ceil(audio.shape[0] / (sr * quantum_s)) * sr * quantum_s)
    if target <= audio.shape[0]:
        return audio
    pad = target - audio.shape[0]
    return np.concatenate([audio, np.zeros(pad, dtype=audio.dtype)])

# --- WER aggregation + bootstrap ------------------------------------------
def aggregate_pairs(records, norm=normalizer):
    """records: list of dicts with 'ref' + 'hyp' + 'severity'.
    Filter <ERROR> rows + post-norm empties. Returns aligned (ref, hyp, sev) tuples."""
    out = []
    for r in records:
        if r.get("hyp", "").startswith("<ERROR"):
            continue
        nr = norm(r["ref"]); nh = norm(r["hyp"])
        if not nr or not nh:
            continue
        out.append((nr, nh, r.get("severity") or "unknown"))
    return out


def overall_wer(rows) -> float | None:
    if not rows:
        return None
    refs = [r[0] for r in rows]; hyps = [r[1] for r in rows]
    return round(jiwer.wer(refs, hyps), 4)


def per_severity_wer(rows) -> dict:
    by_sev = defaultdict(lambda: {"r": [], "h": []})
    for r, h, s in rows:
        by_sev[s]["r"].append(r); by_sev[s]["h"].append(h)
    return {s: round(jiwer.wer(v["r"], v["h"]), 4)
            for s, v in by_sev.items() if v["r"]}


def bootstrap_wer_ci(rows, n_boot: int = 1000, seed: int = 42) -> dict | None:
    if not rows:
        return None
    rng = np.random.default_rng(seed)
    refs = [r[0] for r in rows]; hyps = [r[1] for r in rows]
    overall = jiwer.wer(refs, hyps)
    samples = []
    idxs = np.arange(len(rows))
    for _ in range(n_boot):
        sel = rng.choice(idxs, size=len(idxs), replace=True)
        try:
            samples.append(jiwer.wer([refs[i] for i in sel],
                                     [hyps[i] for i in sel]))
        except Exception:
            pass
    samples.sort()
    return {
        "wer": round(overall, 4),
        "ci95_lo": round(samples[int(0.025 * len(samples))], 4),
        "ci95_hi": round(samples[int(0.975 * len(samples))], 4),
        "n": len(rows),
    }


def report_block(name: str, predictions: list) -> dict:
    """One-stop summary for a model: lowercase + WhisperEN, overall + per-severity + CI."""
    out = {"name": name, "n_predictions": len(predictions)}
    for label, fn in (("lowercase", lowercase_norm), ("whisperEN", normalizer)):
        rows = aggregate_pairs(predictions, norm=fn)
        out[f"{label}_overall"] = bootstrap_wer_ci(rows)
        out[f"{label}_per_severity"] = per_severity_wer(rows)
    return out


# --- Reference fixed numbers from Tier-1+2 (so we never lose them) -------
REFERENCE_NUMBERS = {
    "Whisper-Large-v3 zero-shot (greedy defaults)": {
        "lowercase": 0.7927, "whisperEN": 0.7581,
        "moderate_whisperEN": 2.0293,
    },
    "Whisper-Large-v3 aggressive (rep+ngram+lp=0.6)": {
        "lowercase": 0.4660, "whisperEN": 0.3472,
        "moderate_whisperEN": 0.4756,
    },
    "BASE Gemma 4 E4B (locked recipe)": {
        "lowercase": 0.5658, "whisperEN": 0.4861,
        "moderate_whisperEN": 0.6160,
    },
    "R1-D4b (Gemma 4 + LoRA, locked recipe)": {
        "lowercase": 0.4946, "whisperEN": 0.4562,
        "moderate_whisperEN": 0.5274,
    },
}
TARGET_WER = 0.347              # Whisper-aggressive overall WhisperEN
TARGET_CI_HI = 0.386            # Whisper-aggressive CI95 upper bound

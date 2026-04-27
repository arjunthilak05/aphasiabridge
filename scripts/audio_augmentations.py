"""Phase 0c: SpecAugment + speed/tempo/VTLP stack for disordered-speech ASR FT.

Standard practice for ASR fine-tuning. Used by SpecAugment (Park et al. 2019),
Geng et al. 2020 (disordered speech), and the 2025 SAP Challenge top systems.

API:
    augment_audio(audio, sr=16000, p_speed=0.5, p_specaug=0.5, ...) -> np.ndarray

For Whisper-style training (where the model takes log-mel features), we
augment at the WAVEFORM level (speed/tempo/VTLP) and let the feature extractor
re-compute mels. SpecAugment applies AFTER mel computation — used as a
post-processing wrapper around the feature extractor's output.

Two entry points:
    - augment_audio_waveform(np.ndarray) for waveform-level perturbations
    - SpecAugmentMask(...).apply(mel) for log-mel masking
"""
from __future__ import annotations
import random
import numpy as np


# ---------- Waveform-level augmentations -----------------------------------

def speed_perturb(audio: np.ndarray, sr: int = 16000, factor: float = 1.0) -> np.ndarray:
    """Resample to change speed (also changes pitch). factor in {0.9, 1.0, 1.1} typical."""
    if abs(factor - 1.0) < 1e-6:
        return audio
    new_len = int(audio.shape[0] / factor)
    # Simple linear interpolation — good enough; full quality wants soxr or torchaudio.functional.speed
    indices = np.linspace(0, audio.shape[0] - 1, new_len)
    return np.interp(indices, np.arange(audio.shape[0]), audio).astype(audio.dtype)


def tempo_perturb(audio: np.ndarray, sr: int = 16000, factor: float = 1.0) -> np.ndarray:
    """Change tempo without changing pitch (phase-vocoder-style).
    Approximated here by speed_perturb since librosa+rubberband may not be
    installed; for production use librosa.effects.time_stretch."""
    try:
        import librosa
        return librosa.effects.time_stretch(y=audio.astype(np.float32), rate=factor)
    except Exception:
        return speed_perturb(audio, sr, factor)


def vtlp(audio: np.ndarray, sr: int = 16000, alpha: float = 1.0) -> np.ndarray:
    """Vocal Tract Length Perturbation — approximated as low-cost
    frequency-warp via resample-then-pad/truncate. alpha in [0.9, 1.1]."""
    if abs(alpha - 1.0) < 1e-6:
        return audio
    perturbed = speed_perturb(audio, sr, alpha)
    if perturbed.shape[0] >= audio.shape[0]:
        return perturbed[: audio.shape[0]]
    padded = np.zeros(audio.shape[0], dtype=audio.dtype)
    padded[: perturbed.shape[0]] = perturbed
    return padded


def augment_audio_waveform(audio: np.ndarray, sr: int = 16000,
                            p_speed: float = 0.5, p_tempo: float = 0.3,
                            p_vtlp: float = 0.3, rng: random.Random | None = None) -> np.ndarray:
    """Apply waveform-level augmentation chain stochastically.
    Each augmentation independently rolls; multiple may compose."""
    if rng is None:
        rng = random
    out = audio.astype(np.float32, copy=True)
    if rng.random() < p_speed:
        factor = rng.choice([0.9, 1.1])
        out = speed_perturb(out, sr, factor)
    if rng.random() < p_tempo:
        factor = rng.choice([0.9, 1.1])
        out = tempo_perturb(out, sr, factor)
    if rng.random() < p_vtlp:
        alpha = rng.uniform(0.9, 1.1)
        out = vtlp(out, sr, alpha)
    return out


# ---------- Log-mel-level SpecAugment --------------------------------------

class SpecAugmentMask:
    """Park et al. 2019: random frequency-mask + time-mask on log-mel spectrogram.

    Apply AFTER feature extractor produces log-mel.
        mel = feature_extractor(audio).input_features[0]  # shape: (n_mels, T)
        mel = SpecAugmentMask().apply(mel)
    """

    def __init__(self, n_freq_masks: int = 2, n_time_masks: int = 2,
                 freq_mask_param: int = 27, time_mask_param: int = 100,
                 mask_value: float = 0.0):
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.mask_value = mask_value

    def apply(self, mel: np.ndarray, rng: random.Random | None = None) -> np.ndarray:
        rng = rng or random
        n_mels, T = mel.shape
        out = mel.copy()
        for _ in range(self.n_freq_masks):
            f = rng.randint(0, min(self.freq_mask_param, n_mels))
            f0 = rng.randint(0, max(1, n_mels - f))
            out[f0:f0 + f, :] = self.mask_value
        for _ in range(self.n_time_masks):
            t = rng.randint(0, min(self.time_mask_param, T))
            t0 = rng.randint(0, max(1, T - t))
            out[:, t0:t0 + t] = self.mask_value
        return out


# ---------- Combined collator hook ----------------------------------------

def make_augmenting_collator(base_collator, p_aug: float = 0.5,
                              specaug: SpecAugmentMask | None = None):
    """Wrap any HF collator. Pre-collate: waveform perturbation. Post-collate:
    SpecAugment on input_features. Used by train_whisper_large_ft.py."""
    if specaug is None:
        specaug = SpecAugmentMask()
    rng = random.Random(0)

    def call(features):
        # features: list of dicts; for Whisper, may have 'input_features' or 'audio'
        for ex in features:
            if "audio" in ex and rng.random() < p_aug:
                ex["audio"] = augment_audio_waveform(ex["audio"], rng=rng)
        batch = base_collator(features)
        if "input_features" in batch and rng.random() < p_aug:
            # batch["input_features"]: (B, n_mels, T)
            arr = batch["input_features"]
            for b in range(arr.shape[0]):
                arr[b] = specaug.apply(np.asarray(arr[b]), rng=rng)
            batch["input_features"] = arr
        return batch
    return call

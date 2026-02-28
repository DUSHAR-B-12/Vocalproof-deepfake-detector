"""
VoiceShield v2 — Log-Mel Spectrogram Feature Extraction
Replaces MFCC with per-sample normalized log-mel spectrograms.
"""

import numpy as np
import librosa
from training.config import (
    SAMPLE_RATE, CLIP_SAMPLES, N_FFT, HOP_LENGTH,
    N_MELS, FMIN, FMAX,
)


def load_and_pad(path: str, sr: int = SAMPLE_RATE,
                 target_len: int = CLIP_SAMPLES) -> np.ndarray:
    """Load wav file, resample to sr, crop or zero-pad to target_len."""
    y, _ = librosa.load(path, sr=sr, mono=True)
    if len(y) >= target_len:
        y = y[:target_len]
    else:
        y = np.pad(y, (0, target_len - len(y)), mode="constant")
    return y


def log_mel_spectrogram(y: np.ndarray,
                        sr: int = SAMPLE_RATE,
                        n_fft: int = N_FFT,
                        hop_length: int = HOP_LENGTH,
                        n_mels: int = N_MELS,
                        fmin: float = FMIN,
                        fmax: float = FMAX) -> np.ndarray:
    """
    Compute log-mel spectrogram from waveform.
    Returns: (n_mels, T) float32 array, per-sample normalized to zero mean / unit var.
    """
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,
        n_mels=n_mels, fmin=fmin, fmax=fmax,
    )
    log_S = librosa.power_to_db(S, ref=np.max)

    # Per-sample normalization (zero mean, unit variance)
    mean = log_S.mean()
    std = log_S.std()
    if std > 1e-6:
        log_S = (log_S - mean) / std
    else:
        log_S = log_S - mean

    return log_S.astype(np.float32)


def extract_features(path: str) -> np.ndarray:
    """
    Full pipeline: load → crop/pad → log-mel → normalize.
    Returns: (n_mels, T, 1) array ready for CNN input.
    """
    y = load_and_pad(path)
    spec = log_mel_spectrogram(y)
    return spec[..., np.newaxis]        # add channel dim

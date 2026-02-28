"""
VoiceShield v2 — tf.data Dataset Pipeline
Streams audio from disk → log-mel spectrogram on-the-fly.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
from training.config import (
    DATASET_ROOT, METADATA_CSV, BATCH_SIZE,
    SAMPLE_RATE, CLIP_SAMPLES, N_FFT, HOP_LENGTH,
    N_MELS, FMIN, FMAX, SEED,
)
from training.feature_extraction import extract_features


def _load_metadata(split: str) -> pd.DataFrame:
    """Load metadata_clean.csv and filter by split."""
    df = pd.read_csv(METADATA_CSV)
    df = df[df["split"] == split].reset_index(drop=True)
    return df


def _numpy_extract(file_path_bytes: bytes) -> np.ndarray:
    """Wrapper for tf.py_function: loads audio and extracts features."""
    if hasattr(file_path_bytes, "numpy"):
        path = file_path_bytes.numpy().decode("utf-8")
    else:
        path = file_path_bytes.decode("utf-8")

    full_path = DATASET_ROOT / path
    if not full_path.exists():
        # After resplit, file may be in a different split dir on disk.
        # Try all split dirs with same class_dir/filename.
        parts = path.replace("\\", "/").split("/")
        class_and_file = "/".join(parts[1:])  # e.g. "real/xxx.wav"
        for alt_split in ("train", "val", "test"):
            alt = DATASET_ROOT / alt_split / class_and_file
            if alt.exists():
                full_path = alt
                break

    spec = extract_features(str(full_path))   # (n_mels, T, 1)
    return spec


def _tf_extract(file_path: tf.Tensor, label: tf.Tensor):
    """Map function for tf.data pipeline."""
    spec = tf.py_function(
        func=_numpy_extract,
        inp=[file_path],
        Tout=tf.float32,
    )
    # Set static shape so Keras can build the model
    T = (CLIP_SAMPLES // HOP_LENGTH) + 1
    spec.set_shape([N_MELS, T, 1])
    return spec, label


def build_dataset(split: str,
                  batch_size: int = BATCH_SIZE,
                  shuffle: bool = True,
                  repeat: bool = False) -> tf.data.Dataset:
    """
    Build a tf.data.Dataset for the given split.

    Parameters
    ----------
    split : 'train', 'val', or 'test'
    batch_size : batch size
    shuffle : whether to shuffle (should be True for train)
    repeat : if True, repeat indefinitely

    Returns
    -------
    tf.data.Dataset yielding (spectrogram, label) batches
    """
    df = _load_metadata(split)
    file_paths = df["file_path"].values
    labels = df["label"].values.astype(np.float32)

    ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))

    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(df), 10_000), seed=SEED)

    ds = ds.map(_tf_extract, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=(split == "train"))
    ds = ds.prefetch(tf.data.AUTOTUNE)

    if repeat:
        ds = ds.repeat()

    return ds


def get_steps(split: str, batch_size: int = BATCH_SIZE) -> int:
    """Return number of steps per epoch for the given split."""
    df = _load_metadata(split)
    n = len(df)
    if split == "train":
        return n // batch_size        # drop_remainder
    return int(np.ceil(n / batch_size))

"""
VoiceShield v2 — ResNet-style binary classifier for deepfake voice detection.

Architecture:
  Input (n_mels, T, 1)
  → SpecAugment (training only)
  → Conv2D stem (3×3, BN, ReLU)
  → N residual stages (each with K residual blocks, stride 2 downsampling)
  → Global Average Pooling
  → Dense → Dropout → Dense(1, sigmoid)
"""

import tensorflow as tf
from tensorflow.keras import layers, Model

from training.config import (
    N_MELS, CLIP_SAMPLES, HOP_LENGTH,
    RESNET_FILTERS, RESNET_BLOCKS_PER_STAGE,
    DENSE_UNITS, DROPOUT_RATE,
)
from training.spec_augment import SpecAugment


def _compute_time_frames() -> int:
    """Calculate number of time frames for the configured clip duration."""
    return (CLIP_SAMPLES // HOP_LENGTH) + 1


# ── Building blocks ──────────────────────────────────────────────

def _conv_bn_relu(x, filters, kernel_size=(3, 3), strides=(1, 1)):
    x = layers.Conv2D(filters, kernel_size, strides=strides,
                      padding="same", use_bias=False,
                      kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x


def _residual_block(x, filters, strides=(1, 1)):
    """Basic residual block: two 3×3 convs with skip connection."""
    shortcut = x

    out = layers.Conv2D(filters, (3, 3), strides=strides,
                        padding="same", use_bias=False,
                        kernel_initializer="he_normal")(x)
    out = layers.BatchNormalization()(out)
    out = layers.ReLU()(out)

    out = layers.Conv2D(filters, (3, 3), strides=(1, 1),
                        padding="same", use_bias=False,
                        kernel_initializer="he_normal")(out)
    out = layers.BatchNormalization()(out)

    # Match dimensions when stride > 1 or filter count changes
    if strides != (1, 1) or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, (1, 1), strides=strides,
                                 padding="same", use_bias=False,
                                 kernel_initializer="he_normal")(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    out = layers.Add()([out, shortcut])
    out = layers.ReLU()(out)
    return out


# ── Full model ────────────────────────────────────────────────────

def build_resnet(input_shape=None, name="VoiceShieldResNet"):
    """
    Build the ResNet-style binary classifier.

    Parameters
    ----------
    input_shape : tuple or None
        (n_mels, time_frames, 1). Auto-computed from config if None.
    name : str
        Model name.

    Returns
    -------
    tf.keras.Model
    """
    if input_shape is None:
        T = _compute_time_frames()
        input_shape = (N_MELS, T, 1)

    inp = layers.Input(shape=input_shape, name="mel_input")

    # SpecAugment (training only)
    x = SpecAugment(name="spec_augment")(inp)

    # Stem conv
    x = _conv_bn_relu(x, RESNET_FILTERS[0], kernel_size=(3, 3))

    # Residual stages
    for stage_idx, filters in enumerate(RESNET_FILTERS):
        for block_idx in range(RESNET_BLOCKS_PER_STAGE):
            strides = (2, 2) if block_idx == 0 and stage_idx > 0 else (1, 1)
            x = _residual_block(x, filters, strides=strides)

    # Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(DENSE_UNITS, activation="relu",
                     kernel_initializer="he_normal")(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    out = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = Model(inputs=inp, outputs=out, name=name)
    return model


if __name__ == "__main__":
    model = build_resnet()
    model.summary()

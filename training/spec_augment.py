"""
VoiceShield v2 — SpecAugment as a Keras layer
Time masking + frequency masking applied only during training.
Reference: Park et al., "SpecAugment" (2019)
"""

import tensorflow as tf
from training.config import (
    FREQ_MASK_PARAM, TIME_MASK_PARAM,
    NUM_FREQ_MASKS, NUM_TIME_MASKS,
)


class SpecAugment(tf.keras.layers.Layer):
    """
    Applies SpecAugment (frequency + time masking) on a
    (batch, freq, time, 1) spectrogram tensor.
    Only active during training.
    """

    def __init__(self,
                 freq_mask_param: int = FREQ_MASK_PARAM,
                 time_mask_param: int = TIME_MASK_PARAM,
                 num_freq_masks: int = NUM_FREQ_MASKS,
                 num_time_masks: int = NUM_TIME_MASKS,
                 **kwargs):
        super().__init__(**kwargs)
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks

    def call(self, x, training=None):
        if not training:
            return x
        return self._augment(x)

    @tf.function
    def _augment(self, x):
        shape = tf.shape(x)                      # (B, F, T, 1)
        freq_dim = shape[1]
        time_dim = shape[2]

        mask = tf.ones_like(x)

        # Frequency masking
        for _ in range(self.num_freq_masks):
            f = tf.random.uniform([], 0, self.freq_mask_param, dtype=tf.int32)
            f0 = tf.random.uniform([], 0, freq_dim - f, dtype=tf.int32)
            indices = tf.range(freq_dim)
            freq_mask = tf.cast(
                tf.logical_or(indices < f0, indices >= f0 + f),
                x.dtype,
            )
            freq_mask = tf.reshape(freq_mask, [1, freq_dim, 1, 1])
            mask = mask * freq_mask

        # Time masking
        for _ in range(self.num_time_masks):
            t = tf.random.uniform([], 0, self.time_mask_param, dtype=tf.int32)
            t0 = tf.random.uniform([], 0, time_dim - t, dtype=tf.int32)
            indices = tf.range(time_dim)
            time_mask = tf.cast(
                tf.logical_or(indices < t0, indices >= t0 + t),
                x.dtype,
            )
            time_mask = tf.reshape(time_mask, [1, 1, time_dim, 1])
            mask = mask * time_mask

        return x * mask

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "freq_mask_param": self.freq_mask_param,
            "time_mask_param": self.time_mask_param,
            "num_freq_masks": self.num_freq_masks,
            "num_time_masks": self.num_time_masks,
        })
        return cfg

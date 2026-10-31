"""Raw-epoch SimCLR-style components; no spectral view or stage labels."""

from dataclasses import asdict, dataclass

import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.models.multiview import build_waveform_encoder


@dataclass(frozen=True)
class AugmentationConfig:
    gain_min: float = 0.8
    gain_max: float = 1.2
    noise_rms_fraction: float = 0.02
    max_mask_seconds: float = 1.0

    def __post_init__(self):
        values = list(asdict(self).values())
        if not np.isfinite(values).all() or not 0 < self.gain_min <= self.gain_max:
            raise ValueError("Augmentation gains must be positive, ordered and finite.")
        if not 0 <= self.noise_rms_fraction <= 0.2 or not 0 <= self.max_mask_seconds <= 1.:
            raise ValueError("Initial EEG augmentations allow noise <= 0.2 RMS and masks <= one second.")


def augment_waveform(waveforms, seed, config=None, sampling_frequency=100.):
    """Stateless independent per-epoch gain, RMS-relative noise, and short mask.

    No temporal permutation, wraparound, frequency masking, or spectral targets.
    These are candidate invariances to evaluate on source data, not guarantees
    of preservation of every sleep-specific transient.
    """
    config = config or AugmentationConfig()
    x = tf.convert_to_tensor(waveforms, dtype=tf.float32)
    tf.debugging.assert_rank(x, 3)
    tf.debugging.assert_equal(tf.shape(x)[-1], 1)
    tf.debugging.assert_all_finite(x, "Nonfinite augmentation input")
    seeds = tf.random.experimental.stateless_split(tf.convert_to_tensor(seed, tf.int32), 4)
    shape = (tf.shape(x)[0], 1, 1)
    if config.gain_min == config.gain_max:
        gain = tf.fill(shape, tf.cast(config.gain_min, tf.float32))
    else:
        gain = tf.random.stateless_uniform(shape, seeds[0], config.gain_min, config.gain_max)
    rms = tf.sqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True))
    augmented = x * gain + tf.random.stateless_normal(tf.shape(x), seeds[1]) * rms * config.noise_rms_fraction
    max_mask = int(round(config.max_mask_seconds * sampling_frequency))
    if max_mask:
        tf.debugging.assert_less(max_mask, tf.shape(x)[1], message="Mask must be shorter than an epoch")
        widths = tf.random.stateless_uniform(shape, seeds[2], 0, max_mask + 1, dtype=tf.int32)
        # Uniform start conditional on each width, including the final sample.
        starts = tf.cast(tf.random.stateless_uniform(shape, seeds[3]) * tf.cast(tf.shape(x)[1] - widths + 1, tf.float32), tf.int32)
        positions = tf.range(tf.shape(x)[1])[None, :, None]
        mask = (positions >= starts) & (positions < starts + widths)
        augmented = tf.where(mask, tf.zeros_like(augmented), augmented)
    return augmented


def build_contrastive_model(config, projection_dim=64):
    raw = keras.Input((config.samples_per_epoch, 1), name="signal")
    tokens = build_waveform_encoder(config)(raw)
    pooled = keras.layers.GlobalAveragePooling1D(name="ssl_pool")(tokens)
    hidden = keras.layers.Dense(config.token_dim, activation="gelu", name="projection_hidden")(pooled)
    projected = keras.layers.Dense(projection_dim, name="projection_output", dtype="float32")(hidden)
    return keras.Model(raw, projected, name="raw_contrastive_model")


def nt_xent(first, second, temperature=0.2, allowed_negatives=None):
    """Symmetric NT-Xent, self removed, paired view retained in denominator.

    Return loss and retrieval/cosine diagnostics. B original epochs give 2B
    anchors and at most 2(B-1) negatives each, regardless of optimizer strategy.
    """
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("Contrastive temperature must be positive and finite.")
    first, second = tf.cast(first, tf.float32), tf.cast(second, tf.float32)
    tf.debugging.assert_rank(first, 2)
    tf.debugging.assert_equal(tf.shape(first), tf.shape(second))
    tf.debugging.assert_all_finite(first, "Nonfinite first projection")
    tf.debugging.assert_all_finite(second, "Nonfinite second projection")
    batch = tf.shape(first)[0]
    tf.debugging.assert_greater_equal(batch, 2, message="NT-Xent requires at least two original epochs")
    z = tf.math.l2_normalize(tf.concat([first, second], axis=0), axis=-1)
    cosine = tf.matmul(z, z, transpose_b=True)
    positive = (tf.range(2 * batch) + batch) % (2 * batch)
    positive_mask = tf.one_hot(positive, 2 * batch, on_value=True, off_value=False)
    if allowed_negatives is None:
        allowed_negatives = ~tf.eye(batch, dtype=tf.bool)
    allowed_negatives = tf.convert_to_tensor(allowed_negatives, tf.bool)
    tf.debugging.assert_equal(tf.shape(allowed_negatives), [batch, batch])
    tf.debugging.assert_equal(allowed_negatives, tf.transpose(allowed_negatives), message="Negative mask must be symmetric")
    allowed_negatives = allowed_negatives & ~tf.eye(batch, dtype=tf.bool)
    negatives = tf.tile(allowed_negatives, [2, 2])
    tf.debugging.assert_equal(tf.reduce_all(tf.reduce_any(negatives, axis=1)), True,
                              message="Every contrastive anchor needs at least one negative")
    logits = tf.where(negatives | positive_mask, cosine / temperature, tf.constant(-1e9, tf.float32))
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=positive, logits=logits))
    positive_cosine = tf.reduce_mean(tf.gather(cosine, positive, axis=1, batch_dims=1))
    negative_cosine = tf.reduce_mean(tf.boolean_mask(cosine, negatives))
    return loss, {"retrieval_accuracy": tf.reduce_mean(tf.cast(tf.argmax(logits, axis=1, output_type=tf.int32) == positive, tf.float32)),
                  "positive_cosine": positive_cosine, "negative_cosine": negative_cosine,
                  "mean_negatives_per_anchor": tf.reduce_mean(tf.reduce_sum(tf.cast(negatives, tf.float32), axis=1))}


def representation_diagnostics(encoder, waveforms):
    """Unaugmented pooled encoder diagnostics; projector is deliberately absent."""
    tokens = encoder(waveforms, training=False).numpy()
    pooled = tokens.mean(axis=1).astype(np.float64)
    normalized = pooled / np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)
    centered = normalized - normalized.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    energy = singular ** 2
    if energy.sum() > 1e-20:
        mass = energy[energy > 0] / energy.sum()
        effective_rank = float(np.exp(-np.sum(mass * np.log(mass))))
    else:
        effective_rank = 0.
    return {"epochs": len(pooled), "embedding_dimension": pooled.shape[-1],
            "mean_dimension_std": float(normalized.std(axis=0).mean()),
            "covariance_effective_rank": effective_rank,
            "mean_embedding_norm": float(np.linalg.norm(pooled, axis=1).mean())}

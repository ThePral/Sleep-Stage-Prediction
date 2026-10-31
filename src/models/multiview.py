"""Compact epoch encoders and matched temporal heads for raw/spectral controls."""

from dataclasses import asdict, dataclass

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.preprocessing.normalization import ALL_FEATURES
from src.preprocessing.sequences import CLASS_LABELS


@dataclass(frozen=True)
class ModelConfig:
    variant: str = "gated"
    sequence_length: int = 10
    samples_per_epoch: int = 3000
    filters: tuple = (16, 32, 64)
    kernels: tuple = (31, 15, 7)
    strides: tuple = (4, 5, 5)
    token_dim: int = 64
    attention_heads: int = 4
    temporal_units: int = 64
    dropout: float = 0.3

    def __post_init__(self):
        if self.variant not in {"raw", "concat", "gated"}:
            raise ValueError("variant must be raw, concat, or gated.")
        for name in ["sequence_length", "samples_per_epoch", "token_dim", "attention_heads", "temporal_units"]:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        for name in ["filters", "kernels", "strides"]:
            values = tuple(getattr(self, name))
            if not values or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in values):
                raise ValueError(f"{name} must contain positive integers.")
            object.__setattr__(self, name, values)
        if not len(self.filters) == len(self.kernels) == len(self.strides):
            raise ValueError("filters, kernels, and strides must describe the same CNN blocks.")
        if self.token_dim % self.attention_heads:
            raise ValueError("token_dim must be divisible by attention_heads.")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1).")

    def to_dict(self):
        return asdict(self)


@keras.utils.register_keras_serializable(package="sleep_stage")
class SpectralTokenizer(layers.Layer):
    """One value-dependent token per named feature, with a learned identity."""

    def __init__(self, token_dim=64, n_features=10, **kwargs):
        super().__init__(**kwargs)
        self.token_dim = int(token_dim)
        self.n_features = int(n_features)

    def build(self, input_shape):
        if input_shape[-1] != self.n_features:
            raise ValueError("Spectral feature count disagrees with the tokenizer.")
        self.value_weights = self.add_weight(
            name="value_weights", shape=(self.n_features, self.token_dim), initializer="glorot_uniform",
        )
        self.feature_identity = self.add_weight(
            name="feature_identity", shape=(self.n_features, self.token_dim),
            initializer=keras.initializers.RandomNormal(stddev=0.02),
        )
        super().build(input_shape)

    def call(self, inputs):
        return inputs[..., :, None] * self.value_weights + self.feature_identity

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (self.n_features, self.token_dim)

    def get_config(self):
        return {**super().get_config(), "token_dim": self.token_dim, "n_features": self.n_features}


@keras.utils.register_keras_serializable(package="sleep_stage")
class GatedCrossAttention(layers.Layer):
    """Waveform queries attend over feature keys/values within the same epoch."""

    def __init__(self, token_dim=64, num_heads=4, dropout=0.3, **kwargs):
        super().__init__(**kwargs)
        if token_dim < 1 or num_heads < 1 or token_dim % num_heads:
            raise ValueError("token_dim must be positive and divisible by num_heads.")
        self.token_dim = int(token_dim)
        self.num_heads = int(num_heads)
        self.dropout_rate = float(dropout)
        self.attention = layers.MultiHeadAttention(
            num_heads=self.num_heads, key_dim=self.token_dim // self.num_heads,
            dropout=self.dropout_rate, name="cross_attention",
        )
        self.gate = layers.Dense(self.token_dim, activation="sigmoid", name="spectral_gate")
        self.context_dropout = layers.Dropout(self.dropout_rate)
        self.normalization = layers.LayerNormalization(epsilon=1e-5, name="fusion_norm")

    def build(self, input_shape):
        raw_shape, spectral_shape = input_shape
        if len(raw_shape) != 3 or len(spectral_shape) != 3:
            raise ValueError("Attention expects per-epoch token tensors of rank three.")
        if raw_shape[-1] != self.token_dim or spectral_shape[-1] != self.token_dim:
            raise ValueError("Both views must use the configured token dimension.")
        if spectral_shape[-2] is not None and spectral_shape[-2] < 2:
            raise ValueError("Cross-attention requires multiple spectral key tokens.")
        self.attention.build(raw_shape, spectral_shape, spectral_shape)
        self.gate.build(tuple(raw_shape[:-1]) + (2 * self.token_dim,))
        self.normalization.build(raw_shape)
        super().build(input_shape)

    def call(self, inputs, training=None, return_attention_scores=False):
        raw, spectral = inputs
        context, scores = self.attention(
            query=raw, value=spectral, key=spectral, training=training,
            return_attention_scores=True,
        )
        gates = self.gate(tf.concat([raw, context], axis=-1))
        fused = self.normalization(raw + gates * self.context_dropout(context, training=training))
        if return_attention_scores:
            return fused, scores, gates
        return fused

    def get_config(self):
        return {**super().get_config(), "token_dim": self.token_dim,
                "num_heads": self.num_heads, "dropout": self.dropout_rate}


@keras.utils.register_keras_serializable(package="sleep_stage")
class EpochSequenceEncoder(layers.Layer):
    """Apply the same epoch model to B*L epochs, then restore temporal order."""

    def __init__(self, epoch_encoder, **kwargs):
        super().__init__(**kwargs)
        self.epoch_encoder = epoch_encoder

    def build(self, input_shape):
        if len(input_shape["waveform"]) != 4:
            raise ValueError("Expected waveform windows shaped (batch, epochs, samples, 1).")
        super().build(input_shape)

    def call(self, inputs, training=None):
        raw = inputs["waveform"]
        shape = tf.shape(raw)
        flattened = {"waveform": tf.reshape(raw, (-1, shape[-2], shape[-1]))}
        if "spectral" in inputs:
            spectral = inputs["spectral"]
            flattened["spectral"] = tf.reshape(spectral, (-1, tf.shape(spectral)[-1]))
        embeddings = self.epoch_encoder(flattened, training=training)
        return tf.reshape(embeddings, (shape[0], shape[1], self.epoch_encoder.output_shape[-1]))

    def compute_output_shape(self, input_shape):
        return tuple(input_shape["waveform"][:2]) + (self.epoch_encoder.output_shape[-1],)

    def get_config(self):
        return {**super().get_config(), "epoch_encoder": keras.utils.serialize_keras_object(self.epoch_encoder)}

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        config["epoch_encoder"] = keras.utils.deserialize_keras_object(config["epoch_encoder"])
        return cls(**config)


def build_waveform_encoder(config=None):
    """Return within-epoch waveform tokens; no batch-dependent normalization."""
    config = config or ModelConfig()
    signal = keras.Input((config.samples_per_epoch, 1), name="signal")
    tokens = signal
    for index, (filters, kernel, stride) in enumerate(zip(config.filters, config.kernels, config.strides)):
        tokens = layers.Conv1D(filters, kernel, strides=stride, padding="same", name=f"conv_{index}")(tokens)
        tokens = layers.LayerNormalization(epsilon=1e-5, name=f"norm_{index}")(tokens)
        tokens = layers.Activation("gelu", name=f"gelu_{index}")(tokens)
    tokens = layers.Dense(config.token_dim, name="token_projection")(tokens)
    return keras.Model(signal, tokens, name="waveform_encoder")


def build_epoch_encoder(config=None):
    config = config or ModelConfig()
    inputs = {"waveform": keras.Input((config.samples_per_epoch, 1), name="waveform")}
    raw_tokens = build_waveform_encoder(config)(inputs["waveform"])
    raw_pool = layers.GlobalAveragePooling1D(name="waveform_pool")
    if config.variant == "raw":
        pooled = raw_pool(raw_tokens)
    else:
        inputs["spectral"] = keras.Input((len(ALL_FEATURES),), name="spectral")
        spectral = SpectralTokenizer(config.token_dim, len(ALL_FEATURES), name="spectral_tokens")(inputs["spectral"])
        if config.variant == "concat":
            spectral = layers.Dense(config.token_dim, activation="gelu", name="spectral_transform")(spectral)
            spectral_pool = layers.GlobalAveragePooling1D(name="spectral_pool")(spectral)
            pooled = layers.Concatenate(name="view_concatenation")([raw_pool(raw_tokens), spectral_pool])
        else:
            fused = GatedCrossAttention(
                config.token_dim, config.attention_heads, config.dropout, name="gated_fusion",
            )([raw_tokens, spectral])
            pooled = raw_pool(fused)
    embedding = layers.Dense(config.token_dim, activation="gelu", name="epoch_embedding")(pooled)
    return keras.Model(inputs, embedding, name=f"{config.variant}_epoch_encoder")


def build_multiview_model(config=None):
    config = config or ModelConfig()
    inputs = {"waveform": keras.Input((config.sequence_length, config.samples_per_epoch, 1), name="waveform")}
    if config.variant != "raw":
        inputs["spectral"] = keras.Input((config.sequence_length, len(ALL_FEATURES)), name="spectral")
    sequence = EpochSequenceEncoder(build_epoch_encoder(config), name="epoch_sequence_encoder")(inputs)
    temporal = layers.GRU(config.temporal_units, name="temporal_gru")(sequence)
    temporal = layers.Dropout(config.dropout, name="temporal_dropout")(temporal)
    output = layers.Dense(len(CLASS_LABELS), activation="softmax", dtype="float32", name="sleep_stage")(temporal)
    return keras.Model(inputs, output, name=f"{config.variant}_sleep_stage_model")


def get_waveform_encoder(model):
    """Expose the shared encoder for subsequent SSL checkpoint transfer."""
    return model.get_layer("epoch_sequence_encoder").epoch_encoder.get_layer("waveform_encoder")


def compile_multiview_model(model, learning_rate=1e-3):
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model

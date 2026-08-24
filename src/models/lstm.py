from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_baseline_lstm(
    sequence_length: int,
    n_features: int,
    n_classes: int,
    lstm_units: int = 64,
    dropout_rate: float = 0.30,
) -> keras.Model:
    """
    Build the single-layer baseline LSTM.

    Architecture
    ------------
    Input
      ↓
    LSTM
      ↓
    Dropout
      ↓
    Dense(softmax)
    """

    inputs = keras.Input(
        shape=(
            sequence_length,
            n_features,
        ),
        name="eeg_features",
    )

    x = layers.LSTM(
        lstm_units,
        name="lstm",
    )(inputs)

    x = layers.Dropout(
        dropout_rate,
        name="dropout",
    )(x)

    outputs = layers.Dense(
        n_classes,
        activation="softmax",
        name="sleep_stage",
    )(x)

    return keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="baseline_sleep_stage_lstm",
    )


def compile_baseline_lstm(
    model: keras.Model,
    learning_rate: float = 1e-3,
) -> keras.Model:
    """
    Compile an LSTM model with Adam and sparse categorical
    cross-entropy.
    """

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=learning_rate
        ),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            keras.metrics.SparseCategoricalAccuracy(
                name="accuracy"
            )
        ],
    )

    return model


def build_stacked_lstm(
    sequence_length: int,
    n_features: int,
    n_classes: int,
    lstm_units: int = 32,
    dropout_rate: float = 0.30,
) -> keras.Model:
    """
    Build a two-layer stacked LSTM.

    Architecture
    ------------
    Input
      ↓
    LSTM(units, return_sequences=True)
      ↓
    LSTM(units)
      ↓
    Dropout
      ↓
    Dense(softmax)
    """

    inputs = keras.Input(
        shape=(
            sequence_length,
            n_features,
        ),
        name="eeg_features",
    )

    x = layers.LSTM(
        lstm_units,
        return_sequences=True,
        name="lstm_1",
    )(inputs)

    x = layers.LSTM(
        lstm_units,
        name="lstm_2",
    )(x)

    x = layers.Dropout(
        dropout_rate,
        name="dropout",
    )(x)

    outputs = layers.Dense(
        n_classes,
        activation="softmax",
        name="sleep_stage",
    )(x)

    return keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="stacked_sleep_stage_lstm",
    )


def build_bidirectional_lstm(
    sequence_length: int,
    n_features: int,
    n_classes: int,
    lstm_units: int = 32,
    dropout_rate: float = 0.30,
) -> keras.Model:
    """
    Build a two-layer bidirectional LSTM.

    Architecture
    ------------
    Input
      ↓
    Bidirectional(
        LSTM(units, return_sequences=True)
    )
      ↓
    Bidirectional(
        LSTM(units)
    )
      ↓
    Dropout
      ↓
    Dense(softmax)
    """

    inputs = keras.Input(
        shape=(
            sequence_length,
            n_features,
        ),
        name="eeg_features",
    )

    x = layers.Bidirectional(
        layers.LSTM(
            lstm_units,
            return_sequences=True,
        ),
        name="bidirectional_lstm_1",
    )(inputs)

    x = layers.Bidirectional(
        layers.LSTM(
            lstm_units,
        ),
        name="bidirectional_lstm_2",
    )(x)

    x = layers.Dropout(
        dropout_rate,
        name="dropout",
    )(x)

    outputs = layers.Dense(
        n_classes,
        activation="softmax",
        name="sleep_stage",
    )(x)

    return keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="bidirectional_sleep_stage_lstm",
    )
    
class TemporalAttention(layers.Layer):
    """
    Learnable temporal attention over a sequence.

    Input shape:
        (batch_size, time_steps, features)

    Output:
        context:
            (batch_size, features)

        attention_weights:
            (batch_size, time_steps)
    """

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(
            **kwargs
        )

        self.score_dense = (
            layers.Dense(
                1,
                activation="tanh",
            )
        )

    def call(
        self,
        inputs,
    ):
        # Shape:
        # (batch, time_steps, 1)
        scores = self.score_dense(
            inputs
        )

        # Remove the last dimension.
        # Shape:
        # (batch, time_steps)
        scores = tf.squeeze(
            scores,
            axis=-1,
        )

        # Normalize across time.
        weights = tf.nn.softmax(
            scores,
            axis=1,
        )

        # Restore feature multiplication dimension.
        # Shape:
        # (batch, time_steps, 1)
        weights_expanded = tf.expand_dims(
            weights,
            axis=-1,
        )

        # Weighted temporal representations.
        weighted_inputs = (
            inputs
            * weights_expanded
        )

        # Sum across time.
        # Shape:
        # (batch, features)
        context = tf.reduce_sum(
            weighted_inputs,
            axis=1,
        )

        return context, weights

    def get_config(self):
        config = super().get_config()
        return config
    
def build_attention_lstm(
    sequence_length: int,
    n_features: int,
    n_classes: int,
    lstm_units: int = 32,
    dropout_rate: float = 0.30,
) -> keras.Model:
    """
    Build a stacked LSTM with temporal attention.

    Architecture
    ------------
    Input
      ↓
    LSTM(32, return_sequences=True)
      ↓
    LSTM(32, return_sequences=True)
      ↓
    TemporalAttention
      ↓
    Dropout
      ↓
    Dense(5, softmax)
    """

    inputs = keras.Input(
        shape=(
            sequence_length,
            n_features,
        ),
        name="eeg_features",
    )

    x = layers.LSTM(
        lstm_units,
        return_sequences=True,
        name="lstm_1",
    )(inputs)

    x = layers.LSTM(
        lstm_units,
        return_sequences=True,
        name="lstm_2",
    )(x)

    context, attention_weights = (
        TemporalAttention(
            name="temporal_attention"
        )(x)
    )

    context = layers.Dropout(
        dropout_rate,
        name="dropout",
    )(context)

    outputs = layers.Dense(
        n_classes,
        activation="softmax",
        name="sleep_stage",
    )(context)

    # We use the classification output as the main model output.
    # The attention weights will be exposed through a separate
    # inspection model later.
    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="attention_sleep_stage_lstm",
    )

    return model
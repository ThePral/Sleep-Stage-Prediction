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
    Build the baseline LSTM for sleep-stage prediction.

    Architecture
    ------------
    Input
      ↓
    LSTM(64)
      ↓
    Dropout(0.30)
      ↓
    Dense(n_classes, softmax)

    Parameters
    ----------
    sequence_length:
        Number of temporal steps.

    n_features:
        Number of features per time step.

    n_classes:
        Number of target sleep stages.

    lstm_units:
        Number of LSTM hidden units.

    dropout_rate:
        Dropout applied after the LSTM.

    Returns
    -------
    keras.Model
        Uncompiled baseline LSTM.
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

    model = keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="baseline_sleep_stage_lstm",
    )

    return model


def compile_baseline_lstm(
    model: keras.Model,
    learning_rate: float = 1e-3,
) -> keras.Model:
    """
    Compile the baseline LSTM.
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
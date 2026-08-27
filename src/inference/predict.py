from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf


CLASS_NAMES = [
    "W",
    "N1",
    "N2",
    "N3",
    "REM",
]

FEATURE_COLUMNS = [
    "delta_absolute",
    "theta_absolute",
    "alpha_absolute",
    "beta_absolute",
    "delta_relative",
    "theta_relative",
    "alpha_relative",
    "beta_relative",
    "spectral_entropy",
    "dominant_frequency",
]


class SleepStagePredictor:
    """
    Load the trained sleep-stage model and perform inference.

    Expected input:
        20 consecutive epochs × 10 features

    Output:
        predicted sleep stage and class probabilities.
    """

    def __init__(
        self,
        model_path: str | Path,
        scaler_path: str | Path,
        config_path: str | Path,
    ) -> None:

        self.model_path = Path(
            model_path
        )

        self.scaler_path = Path(
            scaler_path
        )

        self.config_path = Path(
            config_path
        )

        self._validate_paths()

        self.model = tf.keras.models.load_model(
            self.model_path
        )

        self.scaler = joblib.load(
            self.scaler_path
        )

        with open(
            self.config_path,
            "r",
            encoding="utf-8",
        ) as file:
            self.config = json.load(file)

        self.sequence_length = int(
            self.config[
                "sequence_length_epochs"
            ]
        )

        self.n_features = len(
            self.config[
                "features"
            ]
        )

        self.class_names = list(
            self.config[
                "classes"
            ]
        )

        if self.class_names != CLASS_NAMES:
            raise ValueError(
                "Class ordering in config.json "
                "does not match the expected "
                "class ordering."
            )

    def _validate_paths(self) -> None:
        paths = {
            "model": self.model_path,
            "scaler": self.scaler_path,
            "config": self.config_path,
        }

        for name, path in paths.items():
            if not path.exists():
                raise FileNotFoundError(
                    f"{name.capitalize()} file "
                    f"not found: {path}"
                )

    def _prepare_features(
        self,
        data: pd.DataFrame | np.ndarray,
    ) -> np.ndarray:
        """
        Prepare and scale inference features.
        """

        if isinstance(data, pd.DataFrame):

            missing_columns = [
                column
                for column in FEATURE_COLUMNS
                if column not in data.columns
            ]

            if missing_columns:
                raise ValueError(
                    "Missing feature columns: "
                    f"{missing_columns}"
                )

            features = data[
                FEATURE_COLUMNS
            ].to_numpy(
                dtype=np.float32
            )

        else:

            features = np.asarray(
                data,
                dtype=np.float32,
            )

        if features.ndim != 2:
            raise ValueError(
                "Expected 2D feature input "
                "with shape "
                "(epochs, features). "
                f"Received shape: "
                f"{features.shape}"
            )

        if features.shape[1] != self.n_features:
            raise ValueError(
                "Incorrect number of features. "
                f"Expected {self.n_features}, "
                f"received {features.shape[1]}."
            )

        if not np.isfinite(
            features
        ).all():
            raise ValueError(
                "Input contains NaN or infinite "
                "values."
            )

        scaled = self.scaler.transform(
            pd.DataFrame(
                features,
                columns=FEATURE_COLUMNS,
            )
        )

        return np.asarray(
            scaled,
            dtype=np.float32,
        )

    def predict_sequence(
        self,
        sequence: pd.DataFrame | np.ndarray,
    ) -> dict:
        """
        Predict one 20-epoch sequence.
        """

        features = self._prepare_features(
            sequence
        )

        expected_shape = (
            self.sequence_length,
            self.n_features,
        )

        if features.shape != expected_shape:
            raise ValueError(
                "Incorrect sequence shape. "
                f"Expected {expected_shape}, "
                f"received {features.shape}."
            )

        model_input = np.expand_dims(
            features,
            axis=0,
        )

        probabilities = self.model.predict(
            model_input,
            verbose=0,
        )[0]

        predicted_index = int(
            np.argmax(
                probabilities
            )
        )

        return {
            "predicted_stage": self.class_names[
                predicted_index
            ],
            "predicted_index": predicted_index,
            "probabilities": {
                stage: float(
                    probability
                )
                for stage, probability in zip(
                    self.class_names,
                    probabilities,
                )
            },
        }

    def predict_batch(
        self,
        sequences: Sequence[
            pd.DataFrame | np.ndarray
        ] | np.ndarray,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
    ]:
        """
        Predict a batch of 20-epoch sequences.

        Input
        -----
        Either:

        1. A NumPy array containing raw features:
        (samples, 20, 10)

        2. A sequence of DataFrames, each containing:
        20 rows × 10 raw features

        The method performs the same preprocessing
        used during training.

        Returns
        -------
        predictions:
            Integer class indices with shape (samples,)

        probabilities:
            Class probabilities with shape (samples, 5)
        """

        # ---------------------------------------------------------
        # Case 1: NumPy array
        # ---------------------------------------------------------

        if isinstance(
            sequences,
            np.ndarray,
        ):
            values = np.asarray(
                sequences,
                dtype=np.float32,
            )

            if values.ndim != 3:
                raise ValueError(
                    "Expected 3D input with shape "
                    "(samples, epochs, features). "
                    f"Received {values.shape}"
                )

            expected_shape = (
                self.sequence_length,
                self.n_features,
            )

            if values.shape[1:] != expected_shape:
                raise ValueError(
                    "Incorrect sequence shape. "
                    f"Expected (*, {expected_shape[0]}, "
                    f"{expected_shape[1]}), "
                    f"received {values.shape}."
                )

            # A NumPy array is assumed to contain RAW features.
            scaled_sequences = []

            for sequence in values:

                scaled = self._prepare_features(
                    sequence
                )

                scaled_sequences.append(
                    scaled
                )

            model_input = np.stack(
                scaled_sequences,
                axis=0,
            )

        # ---------------------------------------------------------
        # Case 2: sequence of DataFrames
        # ---------------------------------------------------------

        else:

            scaled_sequences = []

            for sequence in sequences:

                scaled = self._prepare_features(
                    sequence
                )

                scaled_sequences.append(
                    scaled
                )

            model_input = np.stack(
                scaled_sequences,
                axis=0,
            )

        probabilities = self.model.predict(
            model_input,
            verbose=0,
        )

        predictions = np.argmax(
            probabilities,
            axis=1,
        )

        return (
            predictions,
            probabilities,
        )
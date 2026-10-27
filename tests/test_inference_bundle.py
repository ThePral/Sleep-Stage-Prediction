"""Exercise actual Keras serialization and feature-scaler bundle compatibility."""

import json
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from src.inference.predict import CLASS_NAMES, FEATURE_COLUMNS, SleepStagePredictor
from src.preprocessing.normalization import SleepFeatureScaler


class InferenceBundleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        rng = np.random.default_rng(42)
        self.features = pd.DataFrame(rng.uniform(0.1, 1, (20, 10)), columns=FEATURE_COLUMNS)
        self.scaler = SleepFeatureScaler().fit(self.features)
        joblib.dump(self.scaler, self.directory / "scaler.pkl")
        self.config = {"sequence_length_epochs": 10, "features": FEATURE_COLUMNS, "classes": CLASS_NAMES}
        self.write_config()

    def write_config(self):
        (self.directory / "config.json").write_text(json.dumps(self.config), encoding="utf-8")

    def save_model(self, sequence_length=10, n_classes=5):
        tf.keras.utils.set_random_seed(42)
        inputs = tf.keras.Input((sequence_length, 10))
        pooled = tf.keras.layers.GlobalAveragePooling1D()(inputs)
        output = tf.keras.layers.Dense(n_classes, activation="softmax")(pooled)
        model = tf.keras.Model(inputs, output)
        model.save(self.directory / "stacked_lstm_32_32_10_epochs.keras")

    def test_canonical_bundle_roundtrip_and_batch_prediction_agree(self):
        self.save_model()
        predictor = SleepStagePredictor.from_directory(self.directory)
        sequence = self.features.iloc[:10]
        result = predictor.predict_sequence(sequence)
        predicted, probabilities = predictor.predict_batch(np.stack([sequence.to_numpy()] * 2))
        np.testing.assert_allclose(probabilities[0], list(result["probabilities"].values()), atol=1e-6)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
        self.assertEqual(predicted.tolist(), [result["predicted_index"]] * 2)

    def test_historical_model_shape_is_rejected_at_initialization(self):
        self.save_model(sequence_length=20)
        with self.assertRaisesRegex(ValueError, "Model/configuration mismatch"):
            SleepStagePredictor.from_directory(self.directory)

    def test_feature_order_mismatch_is_rejected(self):
        self.save_model()
        self.config["features"] = list(reversed(FEATURE_COLUMNS))
        self.write_config()
        with self.assertRaisesRegex(ValueError, "Feature ordering"):
            SleepStagePredictor.from_directory(self.directory)

    def test_output_class_count_mismatch_is_rejected(self):
        self.save_model(n_classes=4)
        with self.assertRaisesRegex(ValueError, "output size"):
            SleepStagePredictor.from_directory(self.directory)


if __name__ == "__main__":
    unittest.main()

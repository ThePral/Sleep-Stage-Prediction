"""Architecture, source-only data, normalization, and training integration checks."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import mne
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score, f1_score
import tensorflow as tf

from src.data.epoch_store import EpochStore, write_recording
from src.data.research import prepare_recording
from src.data.supervised import SourcePartition, WindowBatches
from src.evaluation.classification import ClassificationAccumulator
from src.inference.multiview import MultiviewPredictor
from src.models.multiview import (GatedCrossAttention, ModelConfig, SpectralTokenizer,
                                  build_epoch_encoder, build_multiview_model,
                                  compile_multiview_model, get_waveform_encoder)
from src.preprocessing.normalization import ALL_FEATURES
from src.preprocessing.research_normalization import ResearchNormalizer, log_spectral
from src.training.supervised import TrainingConfig, train_supervised


def small_config(variant="gated"):
    return ModelConfig(variant=variant, sequence_length=2, samples_per_epoch=128,
                       filters=(8, 16), kernels=(7, 5), strides=(4, 4),
                       token_dim=16, attention_heads=2, temporal_units=16, dropout=0.)


def make_source_stores(root):
    for name, split, amplitude in [("SC4001", "train", 1.), ("SC4011", "val", 3.)]:
        values = np.random.default_rng(42).normal(0, amplitude * 1e-5, (1, 12000))
        raw = mne.io.RawArray(values, mne.create_info(["EEG"], 100, ["eeg"]), verbose=False)
        labels = pd.DataFrame({"epoch_index": np.arange(4), "raw_stage": ["W", "N1", "N2", "REM"],
                               "target_stage": ["W", "N1", "N2", "REM"]})
        result = prepare_recording(raw, labels, dataset="sleepedf", participant_id=name[:5],
                                   recording_id=name, channel_as_recorded="EEG", derivation_used="EEG", split=split)
        write_recording(result, Path(root) / "sleepedf" / name)


class ArchitectureTests(unittest.TestCase):
    def setUp(self):
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(42)

    def test_default_variants_have_correct_shapes_and_compact_parameter_counts(self):
        self.counts = {}
        for variant in ["raw", "concat", "gated"]:
            model = build_multiview_model(ModelConfig(variant=variant))
            inputs = {"waveform": np.ones((1, 10, 3000, 1), dtype=np.float32)}
            if variant != "raw":
                inputs["spectral"] = np.ones((1, 10, 10), dtype=np.float32)
            probabilities = model(inputs, training=False).numpy()
            self.assertEqual(probabilities.shape, (1, 5))
            np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
            self.assertLess(model.count_params(), 500_000)
            self.assertEqual(get_waveform_encoder(model).output_shape, (None, 30, 64))
            self.counts[variant] = model.count_params()
        self.assertLess(self.counts["raw"], self.counts["gated"])

    def test_feature_tokens_have_distinct_identities_and_value_sensitivity(self):
        layer = SpectralTokenizer(16)
        original = layer(tf.zeros((2, 10))).numpy()
        self.assertEqual(original.shape, (2, 10, 16))
        self.assertFalse(np.allclose(original[:, 0], original[:, 1]))
        changed = np.zeros((2, 10), dtype=np.float32)
        changed[:, 3] = 1
        altered = layer(changed).numpy()
        np.testing.assert_array_equal(original[:, :3], altered[:, :3])
        self.assertFalse(np.allclose(original[:, 3], altered[:, 3]))

    def test_attention_uses_ten_keys_and_backpropagates_into_both_views(self):
        raw = tf.Variable(tf.random.normal((2, 8, 16)))
        spectral = tf.Variable(tf.random.normal((2, 10, 16)))
        layer = GatedCrossAttention(16, 2, dropout=0.)
        with tf.GradientTape() as tape:
            fused, attention, gates = layer([raw, spectral], return_attention_scores=True)
            loss = tf.reduce_sum(fused[..., 0])
        gradients = tape.gradient(loss, [raw, spectral])
        self.assertEqual(attention.shape, (2, 2, 8, 10))
        np.testing.assert_allclose(attention.numpy().sum(axis=-1), 1, atol=1e-6)
        self.assertGreater(float(np.max(np.abs(attention[:, :, 0] - attention[:, :, 1]))), 1e-5)
        self.assertTrue(np.all((gates.numpy() >= 0) & (gates.numpy() <= 1)))
        for gradient in gradients:
            self.assertIsNotNone(gradient)
            self.assertTrue(np.isfinite(gradient.numpy()).all())
            self.assertGreater(float(tf.linalg.global_norm([gradient])), 1e-6)
        with self.assertRaisesRegex(ValueError, "multiple"):
            GatedCrossAttention(16, 2)([raw, spectral[:, :1]])

    def test_each_variant_serializes_without_custom_objects(self):
        rng = np.random.default_rng(42)
        for variant in ["raw", "concat", "gated"]:
            model = build_multiview_model(small_config(variant))
            inputs = {"waveform": rng.normal(size=(2, 2, 128, 1)).astype(np.float32)}
            if variant != "raw":
                inputs["spectral"] = rng.normal(size=(2, 2, 10)).astype(np.float32)
            expected = model(inputs, training=False).numpy()
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "model.keras"
                model.save(path)
                loaded = tf.keras.models.load_model(path, compile=False)
                np.testing.assert_allclose(expected, loaded(inputs, training=False).numpy(), atol=1e-6)

    def test_attention_diagnostics_can_be_exposed_as_a_functional_model(self):
        raw = tf.keras.Input((8, 16))
        spectral = tf.keras.Input((10, 16))
        outputs = GatedCrossAttention(16, 2, dropout=0.)([raw, spectral], return_attention_scores=True)
        diagnostic = tf.keras.Model([raw, spectral], outputs)
        fused, attention, gates = diagnostic([tf.ones((1, 8, 16)), tf.ones((1, 10, 16))])
        self.assertEqual(fused.shape, (1, 8, 16))
        self.assertEqual(attention.shape, (1, 2, 8, 10))
        self.assertEqual(gates.shape, (1, 8, 16))

    def test_epoch_encoder_does_not_mix_batch_members(self):
        encoder = build_epoch_encoder(small_config())
        inputs = {"waveform": tf.random.normal((2, 128, 1)), "spectral": tf.random.normal((2, 10))}
        together = encoder(inputs, training=False).numpy()
        separately = np.concatenate([encoder({key: value[i:i + 1] for key, value in inputs.items()}, training=False).numpy()
                                      for i in range(2)])
        np.testing.assert_allclose(together, separately, atol=1e-6)

    def test_seeded_initialization_is_reproducible(self):
        first = build_multiview_model(small_config()).get_weights()
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(42)
        second = build_multiview_model(small_config()).get_weights()
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)

    def test_backbone_weights_transfer_between_comparison_models(self):
        raw_model = build_multiview_model(small_config("raw"))
        waveform = tf.random.normal((2, 128, 1))
        expected = get_waveform_encoder(raw_model)(waveform).numpy()
        for variant in ["concat", "gated"]:
            model = build_multiview_model(small_config(variant))
            get_waveform_encoder(model).set_weights(get_waveform_encoder(raw_model).get_weights())
            np.testing.assert_allclose(expected, get_waveform_encoder(model)(waveform).numpy(), atol=1e-6)

    def test_model_can_overfit_a_small_balanced_problem(self):
        config = small_config("concat")
        model = compile_multiview_model(build_multiview_model(config), learning_rate=0.02)
        labels = np.tile(np.arange(5), 2)
        time = np.arange(128) / 128
        raw = np.stack([np.sin(2 * np.pi * (label + 1) * time) for label in labels])
        raw = np.repeat(raw[:, None, :, None], 2, axis=1).astype(np.float32)
        features = np.zeros((10, 2, 10), dtype=np.float32)
        features[..., :5] = (np.eye(5)[labels, None, :] * 3) - 1
        inputs = {"waveform": raw, "spectral": features}
        initial = float(np.mean(tf.keras.losses.sparse_categorical_crossentropy(labels, model(inputs))))
        for _ in range(60):
            model.train_on_batch(inputs, labels)
        probabilities = model(inputs, training=False).numpy()
        final = float(np.mean(tf.keras.losses.sparse_categorical_crossentropy(labels, probabilities)))
        self.assertGreaterEqual(float(np.mean(probabilities.argmax(axis=1) == labels)), 0.9)
        self.assertLess(final, initial * 0.3)

    def test_invalid_architecture_configuration_is_rejected(self):
        for change in [{"variant": "unknown"}, {"sequence_length": 0}, {"attention_heads": 3},
                       {"filters": [8]}, {"dropout": 1.}, {"strides": [0, 2, 2]}]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                ModelConfig(**change)


class SourceDataAndNormalizationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        make_source_stores(self.root)

    def test_normalization_is_fitted_once_per_unique_training_epoch(self):
        with SourcePartition(self.root, "train", sequence_length=2) as train, \
             SourcePartition(self.root, "val", sequence_length=2) as validation:
            normalizer = ResearchNormalizer().fit(train)
            self.assertEqual(normalizer.state["fit_epochs"], 4)
            self.assertEqual(len(train), 3)
            np.testing.assert_allclose(normalizer.state["spectral_mean"], log_spectral(train.recordings[0].spectral).mean(axis=0))
            with self.assertRaisesRegex(ValueError, "training"):
                ResearchNormalizer().fit(validation)
            train_raw, _, _, _ = train.gather(np.array([0]))
            validation_raw, _, _, _ = validation.gather(np.array([0]))
            ratio = np.std(normalizer.transform_waveform(validation_raw)) / np.std(normalizer.transform_waveform(train_raw))
            self.assertAlmostEqual(float(ratio), 3., places=4)
            path = self.root / "normalizer.json"
            normalizer.save(path)
            restored = ResearchNormalizer.load(path)
            np.testing.assert_array_equal(restored.transform_waveform(train_raw), normalizer.transform_waveform(train_raw))
            restored.state["training_data"]["split_manifest"]["path"] = "relocated/frozen_split.csv"
            restored.assert_compatible(train, require_training_match=True)

    def test_capped_validation_spans_time_without_label_based_selection(self):
        with SourcePartition(self.root, "train", sequence_length=2) as train, \
             SourcePartition(self.root, "val", sequence_length=2) as validation:
            normalizer = ResearchNormalizer().fit(train)
            batches = WindowBatches(validation, normalizer, variant="raw", batch_size=2, max_batches=1)
            _, _, metadata = next(batches.iter_batches(epoch=0))
            self.assertEqual([row["epoch_index"] for row in metadata], [1, 3])

    def test_external_stores_are_not_discovered_as_source_inputs(self):
        raw = mne.io.RawArray(np.random.default_rng(4).normal(0, 1e-5, (1, 12000)),
                             mne.create_info(["C3-A2"], 100, ["eeg"]), verbose=False)
        labels = pd.DataFrame({"epoch_index": np.arange(4), "raw_stage": ["W"] * 4, "target_stage": ["W"] * 4})
        external = prepare_recording(raw, labels, dataset="isruc", participant_id="1", recording_id="1",
                                     channel_as_recorded="C3-A2", derivation_used="C3-A2", split="external")
        write_recording(external, self.root / "isruc/1")
        with SourcePartition(self.root, "train", sequence_length=2) as train:
            self.assertEqual([r.recording_id for r in train.recordings], ["SC4001"])
        with self.assertRaisesRegex(ValueError, "not found"):
            SourcePartition(self.root, "train", sequence_length=2, recordings=["1"])

    def test_batch_shapes_alignment_and_deterministic_shuffle(self):
        with SourcePartition(self.root, "train", sequence_length=2, max_open_recordings=1) as train:
            normalizer = ResearchNormalizer().fit(train)
            batches = WindowBatches(train, normalizer, variant="gated", batch_size=2, shuffle=True, seed=42)
            first = list(batches.iter_batches(epoch=0))
            repeated = list(batches.iter_batches(epoch=0))
            self.assertEqual(len(first), 2)
            self.assertEqual(first[0][0]["waveform"].shape, (2, 2, 3000, 1))
            self.assertEqual(first[0][0]["spectral"].shape, (2, 2, 10))
            self.assertEqual(first[1][0]["waveform"].shape[0], 1)
            for a, b in zip(first, repeated):
                np.testing.assert_array_equal(a[1], b[1])
                self.assertEqual(a[2], b[2])
                for label, row in zip(a[1], a[2]):
                    self.assertEqual(row["true_stage"], ["W", "N1", "N2", "N3", "REM"][label])
                    self.assertEqual(row["epoch_index"], int(row["epoch_id"].split(":")[-1]))
            self.assertLessEqual(len(train._maps), 1)
            tf_batches = list(batches.as_dataset().as_numpy_iterator())
            self.assertEqual(sum(len(y) for _, y in tf_batches), 3)

    def test_explicit_held_out_recording_cannot_enter_training(self):
        with self.assertRaisesRegex(ValueError, "not in the train"):
            SourcePartition(self.root, "train", sequence_length=2, recordings=["SC4011"])
        with self.assertRaisesRegex(ValueError, "source train/val/test"):
            SourcePartition(self.root, "external", sequence_length=2)

    def test_missing_duplicate_recordings_and_invalid_batches_rejected(self):
        for selection in [["missing"], ["SC4001", "SC4001"]]:
            with self.assertRaises(ValueError):
                SourcePartition(self.root, "train", sequence_length=2, recordings=selection)
        with SourcePartition(self.root, "train", sequence_length=2) as train:
            normalizer = ResearchNormalizer().fit(train)
            for batch_size in [0, -1, True]:
                with self.assertRaises(ValueError):
                    WindowBatches(train, normalizer, variant="raw", batch_size=batch_size)
            with self.assertRaises(IndexError):
                train.gather(np.array([-1]))

    def test_normalization_rejects_missing_and_invalid_statistics(self):
        with self.assertRaises(RuntimeError):
            ResearchNormalizer().transform_waveform(np.ones((2, 128, 1)))
        with SourcePartition(self.root, "train", sequence_length=2) as train:
            normalizer = ResearchNormalizer().fit(train)
            features = train.recordings[0].spectral.copy()
            features[0, 0] = 0
            with self.assertRaisesRegex(ValueError, "positive"):
                normalizer.transform_spectral(features)
            normalizer.state["training_data"]["split"] = "external"
            path = self.root / "bad_normalizer.json"
            path.write_text(json.dumps(normalizer.state), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source-training-only"):
                ResearchNormalizer.load(path)


class MetricTests(unittest.TestCase):
    def test_streaming_metrics_agree_with_sklearn(self):
        true = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        predicted = np.array([0, 1, 1, 2, 2, 2, 3, 0, 4, 1])
        probabilities = np.full((10, 5), 0.05)
        probabilities[np.arange(10), predicted] = 0.8
        metrics = ClassificationAccumulator()
        metrics.update(true[:4], probabilities[:4])
        metrics.update(true[4:], probabilities[4:])
        result = metrics.result()
        for actual, expected in [(result["accuracy"], accuracy_score(true, predicted)),
                                 (result["macro_f1"], f1_score(true, predicted, average="macro")),
                                 (result["balanced_accuracy"], balanced_accuracy_score(true, predicted)),
                                 (result["cohen_kappa"], cohen_kappa_score(true, predicted))]:
            self.assertAlmostEqual(actual, expected)

    def test_absent_class_policy_is_explicit(self):
        metrics = ClassificationAccumulator()
        metrics.update(np.array([0, 0]), np.array([[1., 0, 0, 0, 0]] * 2))
        self.assertAlmostEqual(metrics.result()["macro_f1"], 0.2)
        self.assertAlmostEqual(metrics.result()["balanced_accuracy"], 1.)
        with self.assertRaisesRegex(ValueError, "sum to one"):
            metrics.update(np.array([0]), np.array([[1., 1., 0, 0, 0]]))


class TrainingIntegrationTests(unittest.TestCase):
    def test_training_checkpoint_export_and_raw_input_inference(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            make_source_stores(root)
            run = root / "run"
            config = replace(small_config(), samples_per_epoch=3000)
            summary = train_supervised(root, run, config, TrainingConfig(batch_size=2, epochs=2),
                                       max_train_batches=1, max_validation_batches=1, verbose=0)
            self.assertTrue(summary["limited_data_run"])
            self.assertEqual(summary["epochs_completed"], 2)
            self.assertIn(summary["best_epoch"], [1, 2])
            self.assertLessEqual(summary["checkpoint_reload_max_probability_difference"], 1e-6)
            predictions = pd.read_csv(run / "validation_predictions.csv")
            self.assertEqual(len(predictions), 2)
            self.assertEqual(set(predictions.participant_id), {"SC401"})
            self.assertIn("val_macro_f1", pd.read_csv(run / "training_history.csv"))
            predictor = MultiviewPredictor(run)
            with EpochStore(root / "sleepedf/SC4011") as store:
                positions = store.sequence_positions(2)
                # Replay the exported target IDs, including evenly spaced
                # capped validation windows rather than assuming the first rows.
                selected = np.stack([positions[positions[:, -1] == epoch][0] for epoch in predictions.epoch_index])
                batch = store.gather(selected)
                _, probabilities = predictor.predict_batch(batch["waveform"], batch["spectral"])
                np.testing.assert_allclose(probabilities, predictions[[f"prob_{s}" for s in ["W", "N1", "N2", "N3", "REM"]]], atol=1e-6)
                with self.assertRaisesRegex(ValueError, "align"):
                    predictor.predict_batch(batch["waveform"], batch["spectral"][:, :1])
            with self.assertRaises(FileExistsError):
                train_supervised(root, run, config)
            with (run / "normalizer.json").open("a", encoding="utf-8") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(ValueError, "integrity"):
                MultiviewPredictor(run)


if __name__ == "__main__":
    unittest.main()

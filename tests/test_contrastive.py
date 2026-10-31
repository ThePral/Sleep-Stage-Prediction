"""Contrastive objective correctness, label independence, resume and transfer."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import mne
import numpy as np
import pandas as pd
import tensorflow as tf

from src.data.contrastive import EpochBatches, SourceEpochPool, negative_mask
from src.data.epoch_store import write_recording
from src.data.research import prepare_recording
from src.data.supervised import SourcePartition
from src.inference.multiview import MultiviewPredictor
from src.models.contrastive import (AugmentationConfig, augment_waveform, build_contrastive_model,
                                    nt_xent, representation_diagnostics)
from src.models.multiview import ModelConfig, build_linear_probe, build_multiview_model, get_waveform_encoder
from src.preprocessing.research_normalization import ResearchNormalizer
from src.preprocessing.waveform_normalization import WaveformNormalizer
from src.training.contrastive import PretrainedEncoder, PretrainingConfig, train_contrastive
from src.training.supervised import TrainingConfig, train_supervised


def model_config(variant="raw", length=1):
    return ModelConfig(variant=variant, sequence_length=length, samples_per_epoch=3000,
                       filters=(4, 8), kernels=(7, 5), strides=(10, 10), token_dim=8,
                       attention_heads=2, temporal_units=8, dropout=0.)


def fixtures(root, hide_labels=False):
    for name, split in [("SC4001", "train"), ("SC4011", "val")]:
        values = np.random.default_rng(8).normal(0, 1e-5, (1, 8 * 3000))
        values[:, 7 * 3000:] = 0  # QC-ineligible regardless of labels.
        raw = mne.io.RawArray(values, mne.create_info(["EEG"], 100, ["eeg"]), verbose=False)
        stages = [None] * 8 if hide_labels else ["W", "N1", "N2", "N3", "REM", "W", None, "W"]
        labels = pd.DataFrame({"epoch_index": np.arange(8), "raw_stage": stages, "target_stage": stages})
        result = prepare_recording(raw, labels, dataset="sleepedf", participant_id=name[:5],
                                   recording_id=name, channel_as_recorded="EEG", derivation_used="EEG", split=split)
        write_recording(result, root / "sleepedf" / name)


class ContrastiveMathTests(unittest.TestCase):
    def test_loss_matches_direct_numpy_formula_and_gradients(self):
        rng = np.random.default_rng(8)
        first = tf.Variable(rng.normal(size=(3, 4)).astype(np.float32))
        second = tf.Variable(rng.normal(size=(3, 4)).astype(np.float32))
        allowed = np.array([[False, True, False], [True, False, True], [False, True, False]])
        with tf.GradientTape() as tape:
            loss, metrics = nt_xent(first, second, 0.2, allowed)
        z = np.concatenate([first.numpy(), second.numpy()]).astype(np.float64)
        z /= np.linalg.norm(z, axis=-1, keepdims=True)
        similarities = z @ z.T / 0.2
        losses = []
        for i in range(6):
            positive = (i + 3) % 6
            candidates = [j for j in range(6) if allowed[i % 3, j % 3] or j == positive]
            scores = similarities[i, candidates]
            maximum = scores.max()
            losses.append(maximum + np.log(np.exp(scores - maximum).sum()) - similarities[i, positive])
        self.assertAlmostEqual(float(loss), float(np.mean(losses)), places=5)
        self.assertAlmostEqual(float(metrics["mean_negatives_per_anchor"]), 8 / 3, places=5)
        for gradient in tape.gradient(loss, [first, second]):
            self.assertTrue(np.isfinite(gradient).all())
            self.assertGreater(float(tf.linalg.global_norm([gradient])), 0.)

    def test_constant_representation_has_known_loss_and_bad_diagnostics(self):
        loss, _ = nt_xent(tf.ones((4, 8)), tf.ones((4, 8)))
        self.assertAlmostEqual(float(loss), float(np.log(7)), places=5)
        inputs = tf.keras.Input((128, 1))
        # A constant but connected encoder.
        output = tf.keras.layers.Conv1D(8, 1, kernel_initializer="zeros", bias_initializer="ones")(inputs)
        encoder = tf.keras.Model(inputs, output)
        diagnostics = representation_diagnostics(encoder, np.ones((4, 128, 1), dtype=np.float32))
        self.assertEqual(diagnostics["covariance_effective_rank"], 0.)
        self.assertEqual(diagnostics["mean_dimension_std"], 0.)

    def test_objective_rejects_invalid_negatives_and_temperature(self):
        for temperature in [0., -1., float("nan")]:
            with self.assertRaises(ValueError):
                nt_xent(tf.ones((2, 8)), tf.ones((2, 8)), temperature)
        with self.assertRaises(tf.errors.InvalidArgumentError):
            nt_xent(tf.ones((1, 8)), tf.ones((1, 8)))
        with self.assertRaises(tf.errors.InvalidArgumentError):
            nt_xent(tf.ones((2, 8)), tf.ones((2, 8)), allowed_negatives=np.zeros((2, 2), bool))

    def test_temporal_mask_uses_only_recording_and_epoch_ids(self):
        metadata = [{"recording_id": r, "epoch_index": e} for r, e in [("a", 0), ("a", 1), ("a", 3), ("b", 0)]]
        allowed = negative_mask(metadata, 1)
        self.assertFalse(allowed[0, 1])
        self.assertTrue(allowed[0, 2])
        self.assertTrue(allowed[0, 3])
        self.assertFalse(allowed.diagonal().any())
        np.testing.assert_array_equal(allowed, allowed.T)

    def test_augmentations_are_stateless_independent_and_bounded(self):
        raw = tf.ones((8, 3000, 1))
        first = augment_waveform(raw, [42, 0]).numpy()
        np.testing.assert_array_equal(first, augment_waveform(raw, [42, 0]).numpy())
        self.assertFalse(np.array_equal(first, augment_waveform(raw, [42, 1]).numpy()))
        self.assertEqual(first.shape, raw.shape)
        self.assertTrue(np.isfinite(first).all())
        mask_only = AugmentationConfig(1., 1., 0., 1.)
        masked = augment_waveform(raw, [8, 0], mask_only).numpy()[..., 0]
        self.assertTrue(((masked == 0).sum(axis=1) <= 100).all())
        for row in masked:
            removed = np.flatnonzero(row == 0)
            if len(removed):
                self.assertEqual(removed[-1] - removed[0] + 1, len(removed))
        doubled = augment_waveform(raw, [42, 0], AugmentationConfig(2., 2., 0., 0.)).numpy()
        np.testing.assert_array_equal(doubled, raw.numpy() * 2)

    def test_invalid_configuration_is_rejected(self):
        for change in [{"batch_size": 1}, {"temperature": 0}, {"exclusion_epochs": -1}, {"seed": True}]:
            with self.assertRaises(ValueError):
                PretrainingConfig(**change)
        for change in [{"gain_min": 0}, {"noise_rms_fraction": .3}, {"max_mask_seconds": 2.}]:
            with self.assertRaises(ValueError):
                AugmentationConfig(**change)

    def test_projection_model_serializes_and_backpropagates_to_encoder(self):
        tf.keras.utils.set_random_seed(42)
        config = replace(model_config(), samples_per_epoch=128)
        model = build_contrastive_model(config, 8)
        inputs = tf.random.normal((4, 128, 1))
        with tf.GradientTape() as tape:
            projections = model(inputs)
            loss, _ = nt_xent(projections[:2], projections[2:])
        gradients = tape.gradient(loss, model.get_layer("waveform_encoder").trainable_variables)
        self.assertTrue(all(g is not None for g in gradients))
        self.assertGreater(float(tf.linalg.global_norm(gradients)), 0.)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.keras"
            model.save(path)
            loaded = tf.keras.models.load_model(path, compile=False)
            np.testing.assert_allclose(model(inputs), loaded(inputs), atol=1e-6)


class LabelFreeDataTests(unittest.TestCase):
    def test_labels_do_not_change_pool_waveforms_order_or_statistics(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixtures(root / "scored")
            fixtures(root / "unscored", hide_labels=True)
            with SourceEpochPool(root / "scored") as scored, SourceEpochPool(root / "unscored") as unscored:
                self.assertEqual(len(scored), 7)
                self.assertEqual(len(unscored), 7)
                self.assertIsNone(scored.recordings[0].labels)
                self.assertIsNone(scored.recordings[0].spectral)
                a, b = WaveformNormalizer().fit(scored), WaveformNormalizer().fit(unscored)
                self.assertEqual(a.state["waveform_scale"], b.state["waveform_scale"])
                self.assertEqual(a.state["fit_epochs"], 7)
                first = EpochBatches(scored, a, batch_size=4, seed=42)
                second = EpochBatches(unscored, b, batch_size=4, seed=42)
                for one, two in zip(first.iter_batches(0), second.iter_batches(0)):
                    for x, y in zip(one[:2], two[:2]):
                        np.testing.assert_array_equal(x, y)
                    self.assertEqual(one[2], two[2])
                    self.assertTrue(all("true_stage" not in m for m in one[2]))
                path = root / "normalizer.json"
                a.save(path)
                restored = WaveformNormalizer.load(path)
                np.testing.assert_array_equal(restored.transform_waveform(scored.gather(np.array([6]))[0]),
                                              a.transform_waveform(scored.gather(np.array([6]))[0]))

    def test_pool_rejects_held_out_records_and_invalid_indices(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixtures(root)
            with self.assertRaisesRegex(ValueError, "not in the train"):
                SourceEpochPool(root, recordings=["SC4011"])
            with SourceEpochPool(root) as pool:
                with self.assertRaises(IndexError):
                    pool.gather(np.array([-1]))
                with self.assertRaises(ValueError):
                    pool.gather(np.array([.5]))
                normalizer = WaveformNormalizer().fit(pool)
                with self.assertRaises(ValueError):
                    EpochBatches(pool, normalizer, batch_size=8)
            with SourcePartition(root, "val", sequence_length=1) as validation:
                with self.assertRaises(ValueError):
                    WaveformNormalizer().fit(validation)

    def test_unlabeled_waveform_scale_and_supervised_spectral_fit_are_explicit(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixtures(root)
            with SourceEpochPool(root) as pool, SourcePartition(root, "train", sequence_length=2) as train:
                waveform = WaveformNormalizer().fit(pool)
                normalizer = ResearchNormalizer().fit(train).use_pretraining_waveform(waveform, train)
                self.assertEqual(normalizer.state["fit_epochs"], 6)
                self.assertEqual(normalizer.state["pretraining_waveform"]["fit_epochs"], 7)
                raw, _, _, _ = train.gather(np.array([0]))
                np.testing.assert_array_equal(normalizer.transform_waveform(raw), waveform.transform_waveform(raw))
                normalizer.save(root / "combined.json")
                ResearchNormalizer.load(root / "combined.json")
                waveform.state["training_data"]["recordings"][0]["manifest_sha256"] = "wrong"
                with self.assertRaisesRegex(ValueError, "selections disagree"):
                    normalizer.use_pretraining_waveform(waveform, train)


class PretrainingIntegrationTests(unittest.TestCase):
    def test_epoch_resume_matches_uninterrupted_updates_then_transfers_and_probes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            fixtures(root)
            config = model_config()
            settings = PretrainingConfig(batch_size=4, epochs=2, projection_dim=8, diagnostic_epochs=4)
            whole = train_contrastive(root, root / "whole", config, settings, max_batches=1, verbose=0)
            original_iterator = EpochBatches.iter_batches

            def interrupt_second_epoch(batches, epoch):
                if epoch == 1:
                    raise RuntimeError("simulated interruption")
                yield from original_iterator(batches, epoch)

            with mock.patch.object(EpochBatches, "iter_batches", interrupt_second_epoch):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    train_contrastive(root, root / "first", config, settings, max_batches=1, verbose=0)
            self.assertEqual(json.loads((root / "first/status.json").read_text())["status"], "failed")
            resumed = train_contrastive(root, root / "resumed", config, settings,
                                        max_batches=1, resume_from=root / "first", verbose=0)
            self.assertEqual(resumed["optimizer_iterations"], 2)
            self.assertEqual(resumed["resumed_after_epoch"], 1)
            self.assertEqual(resumed["source_quality_epochs_available"], 7)
            self.assertGreater(resumed["encoder_max_weight_update_this_run"], 0)
            self.assertLessEqual(whole["encoder_reload_max_difference"], 1e-6)
            expected, artifact = PretrainedEncoder(root / "whole"), PretrainedEncoder(root / "resumed")
            for a, b in zip(expected.encoder.get_weights(), artifact.encoder.get_weights()):
                np.testing.assert_allclose(a, b, atol=1e-6, rtol=1e-6)
            first_history = pd.read_csv(root / "whole/training_history.csv")
            second_history = pd.read_csv(root / "resumed/training_history.csv")
            np.testing.assert_allclose(first_history.loss, second_history.loss, atol=1e-6)
            with SourcePartition(root, "train", sequence_length=2) as train:
                for variant in ["raw", "concat", "gated"]:
                    downstream = build_multiview_model(model_config(variant, 2))
                    artifact.transfer(downstream, model_config(variant, 2), train)
                    for a, b in zip(artifact.encoder.get_weights(), get_waveform_encoder(downstream).get_weights()):
                        np.testing.assert_array_equal(a, b)
                with self.assertRaisesRegex(ValueError, "architecture"):
                    artifact.transfer(downstream, replace(model_config("gated", 2), token_dim=16), train)
            probe = train_supervised(root, root / "probe", config,
                                     TrainingConfig(batch_size=2, epochs=1), max_train_batches=1,
                                     max_validation_batches=1, pretrained_path=root / "resumed", linear_probe=True, verbose=0)
            self.assertEqual(probe["trainable_parameters"], (config.token_dim + 1) * 5)
            self.assertEqual(probe["training_mode"], "linear_probe")
            predictor = MultiviewPredictor(root / "probe")
            for a, b in zip(artifact.encoder.get_weights(), get_waveform_encoder(predictor.model).get_weights()):
                np.testing.assert_array_equal(a, b)
            summary = train_supervised(root, root / "finetune", model_config("gated", 2),
                                       TrainingConfig(batch_size=2, epochs=1), max_train_batches=1,
                                       max_validation_batches=1, pretrained_path=root / "resumed", verbose=0)
            self.assertTrue(summary["pretrained_waveform"])
            final = MultiviewPredictor(root / "finetune")
            self.assertEqual(final.normalizer.state["waveform_scale"], artifact.normalizer.state["waveform_scale"])
            self.assertTrue(any(not np.array_equal(a, b) for a, b in zip(artifact.encoder.get_weights(), get_waveform_encoder(final.model).get_weights())))
            scratch = train_supervised(root, root / "scratch_probe", config,
                                       TrainingConfig(batch_size=2, epochs=1), max_train_batches=1,
                                       max_validation_batches=1, linear_probe=True,
                                       waveform_normalizer_path=root / "resumed/waveform_normalizer.json", verbose=0)
            self.assertFalse(scratch["pretrained_waveform"])
            control = MultiviewPredictor(root / "scratch_probe")
            self.assertEqual(control.normalizer.state["waveform_scale"], artifact.normalizer.state["waveform_scale"])
            self.assertFalse(all(np.array_equal(a, b) for a, b in zip(artifact.encoder.get_weights(), get_waveform_encoder(control.model).get_weights())))
            with self.assertRaisesRegex(ValueError, "settings disagree"):
                train_contrastive(root, root / "wrong_temperature", config, replace(settings, temperature=.1),
                                  max_batches=1, resume_from=root / "first", verbose=0)
            with self.assertRaises(FileExistsError):
                train_contrastive(root, root / "whole", config, settings, verbose=0)
            # Neither bundle use nor resume accepts modified artifacts.
            with (root / "resumed/encoder.keras").open("ab") as stream:
                stream.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "integrity"):
                PretrainedEncoder(root / "resumed")
            (root / "first/checkpoints/epoch_0001/state.index").write_bytes(b"tampered")
            with self.assertRaisesRegex(ValueError, "integrity"):
                train_contrastive(root, root / "bad_resume", config, settings, max_batches=1,
                                  resume_from=root / "first", verbose=0)

    def test_linear_probe_rejects_temporal_or_fusion_models(self):
        for config in [model_config("raw", 2), model_config("gated", 1)]:
            with self.assertRaisesRegex(ValueError, "single-epoch"):
                build_linear_probe(config)


if __name__ == "__main__":
    unittest.main()

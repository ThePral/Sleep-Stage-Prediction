"""BiLSTM trainability, portable inference, and source-only SSL transfer."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import tensorflow as tf

from src.compute.preflight import PreflightConfig
from src.data.epoch_store import EpochStore, sha256_file
from src.data.supervised import SourcePartition
from src.inference.multiview import MultiviewPredictor
from src.models.multiview import ModelConfig, build_multiview_model, compile_multiview_model, get_waveform_encoder
from src.training.contrastive import PretrainedEncoder, PretrainingConfig, train_contrastive
from src.training.supervised import TrainingConfig, train_supervised
from tests.test_multiview import small_config
from tests.test_contrastive import fixtures, model_config


class BiLSTMArchitectureTests(unittest.TestCase):
    def setUp(self):
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(42)

    def test_all_views_train_both_directions_and_roundtrip(self):
        rng = np.random.default_rng(42)
        for variant in ['spectral', 'raw', 'concat', 'gated']:
            with self.subTest(variant=variant):
                config = replace(small_config(variant), temporal_head='bilstm', temporal_units=8)
                model = build_multiview_model(config)
                inputs = {} if variant == 'spectral' else {'waveform': rng.normal(size=(2, 2, 128, 1)).astype(np.float32)}
                if variant != 'raw':
                    inputs['spectral'] = rng.normal(size=(2, 2, 10)).astype(np.float32)
                head = model.get_layer('temporal_bilstm')
                self.assertEqual(head.output.shape[-1], 16)
                self.assertFalse(head.forward_layer.go_backwards)
                self.assertTrue(head.backward_layer.go_backwards)
                with tf.GradientTape() as tape:
                    probabilities = model(inputs, training=True)
                    loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(tf.constant([0, 4]), probabilities))
                gradients = tape.gradient(loss, model.trainable_variables)
                for gradient in gradients:
                    self.assertIsNotNone(gradient)
                    self.assertTrue(np.isfinite(gradient.numpy()).all())
                by_id = {id(variable): gradient for variable, gradient in zip(model.trainable_variables, gradients)}
                for direction in [head.forward_layer, head.backward_layer]:
                    gradient = by_id[id(direction.cell.kernel)]
                    self.assertGreater(float(tf.linalg.global_norm([gradient])), 1e-7)
                expected = model(inputs, training=False).numpy()
                np.testing.assert_allclose(expected.sum(axis=1), 1., atol=1e-6)
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / 'model.keras'
                    model.save(path)
                    restored = tf.keras.models.load_model(path, compile=False)
                    np.testing.assert_allclose(restored(inputs, training=False), expected, atol=1e-6)

    def test_gated_bilstm_can_overfit_a_tiny_multiclass_batch(self):
        config = replace(small_config(), temporal_head='bilstm', temporal_units=8)
        model = compile_multiview_model(build_multiview_model(config), .02)
        labels = np.arange(10) % 5
        time = np.arange(128) / 128
        raw = np.stack([np.sin(2 * np.pi * (label + 1) * time) for label in labels])
        raw = np.repeat(raw[:, None, :, None], 2, axis=1).astype(np.float32)
        spectral = np.zeros((10, 2, 10), dtype=np.float32)
        spectral[..., :5] = np.eye(5)[labels, None, :] * 3 - 1
        inputs = {'waveform': raw, 'spectral': spectral}
        initial = float(tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, model(inputs))))
        for _ in range(80):
            model.train_on_batch(inputs, labels)
        probabilities = model(inputs, training=False).numpy()
        final = float(tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(labels, probabilities)))
        self.assertGreaterEqual(float(np.mean(probabilities.argmax(axis=1) == labels)), .9)
        self.assertLess(final, initial * .3)

    def test_legacy_configuration_defaults_keep_the_previous_architecture(self):
        old = small_config().to_dict()
        old.pop('temporal_head')
        config = ModelConfig(**old)
        self.assertEqual(config.temporal_head, 'legacy')
        self.assertIsInstance(build_multiview_model(config).get_layer('temporal_gru'), tf.keras.layers.GRU)
        self.assertEqual(build_multiview_model(ModelConfig()).count_params(), 82757)
        spectral = build_multiview_model(ModelConfig(variant='spectral'))
        self.assertIsInstance(spectral.get_layer('lstm_2'), tf.keras.layers.LSTM)
        for factory in [ModelConfig, PreflightConfig]:
            with self.assertRaisesRegex(ValueError, 'temporal_head'):
                factory(temporal_head='unknown')


class BiLSTMTransferTests(unittest.TestCase):
    def test_ssl_transfer_finetuning_export_and_declared_head_agree(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixtures(root)
            settings = PretrainingConfig(batch_size=4, epochs=1, projection_dim=8, diagnostic_epochs=4)
            train_contrastive(root, root / 'ssl', model_config(), settings, max_batches=1, verbose=0)
            pretrained = PretrainedEncoder(root / 'ssl')
            with SourcePartition(root, 'train', sequence_length=2) as partition:
                for variant in ['raw', 'concat', 'gated']:
                    config = replace(model_config(variant, 2), temporal_head='bilstm')
                    model = build_multiview_model(config)
                    pretrained.transfer(model, config, partition)
                    for before, after in zip(pretrained.encoder.get_weights(), get_waveform_encoder(model).get_weights()):
                        np.testing.assert_array_equal(before, after)
            config = replace(model_config('gated', 2), temporal_head='bilstm')
            summary = train_supervised(root, root / 'finetune', config, TrainingConfig(batch_size=2, epochs=1),
                                       max_train_batches=1, max_validation_batches=1, pretrained_path=root / 'ssl', verbose=0)
            self.assertTrue(summary['pretrained_waveform'])
            self.assertEqual(summary['temporal_head'], 'bilstm')
            self.assertLessEqual(summary['checkpoint_reload_max_probability_difference'], 1e-6)
            predictor = MultiviewPredictor(root / 'finetune')
            self.assertTrue(any(not np.array_equal(a, b) for a, b in zip(
                pretrained.encoder.get_weights(), get_waveform_encoder(predictor.model).get_weights())))
            predictions = pd.read_csv(root / 'finetune/validation_predictions.csv')
            with EpochStore(root / 'sleepedf/SC4011') as store:
                positions = store.sequence_positions(2)
                selected = np.stack([positions[positions[:, -1] == epoch][0] for epoch in predictions.epoch_index])
                batch = store.gather(selected)
                _, actual = predictor.predict_batch(batch['waveform'], batch['spectral'])
                np.testing.assert_allclose(actual, predictions[[f'prob_{s}' for s in ['W', 'N1', 'N2', 'N3', 'REM']]], atol=1e-6)
            # A consistently rehashed but wrongly described head must still fail.
            path = root / 'finetune/model_config.json'
            changed = json.loads(path.read_text())
            changed['temporal_units'] += 1
            path.write_text(json.dumps(changed))
            bundle_path = root / 'finetune/bundle.json'
            bundle = json.loads(bundle_path.read_text())
            bundle['files']['model_config.json'] = sha256_file(path)
            bundle_path.write_text(json.dumps(bundle))
            with self.assertRaisesRegex(ValueError, 'BiLSTM head'):
                MultiviewPredictor(root / 'finetune')

    def test_linear_probe_ignores_the_unused_temporal_choice(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixtures(root)
            config = replace(model_config(), temporal_head='bilstm')
            train_supervised(root, root / 'probe', config, TrainingConfig(batch_size=2, epochs=1),
                             linear_probe=True, max_train_batches=1, max_validation_batches=1, verbose=0)
            predictor = MultiviewPredictor(root / 'probe')
            self.assertEqual(predictor.model.name, 'linear_sleep_stage_probe')


if __name__ == '__main__':
    unittest.main()

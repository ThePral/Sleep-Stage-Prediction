"""Recovery/inference gates without training calls or optimizer updates."""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.experiments.run import archive_stale_lock, code_hashes, execute_study, main, supervised_resume_source
from src.data.supervised import WindowBatches
from src.models.multiview import ModelConfig, build_multiview_model
from src.training.recovery_state import load_recovery, publish_recovery
from src.training.supervised import TrainingConfig, evaluate_batches, train_supervised, validation_predictor
from src.training.supervised_recovery import EpochEvents, SupervisedRecovery


class Variable:
    def __init__(self, path, value):
        self.path = path
        self.value = np.asarray(value)
        self.shape = self.value.shape
        self.dtype = str(self.value.dtype)
    def numpy(self):
        return self.value.copy()
    def assign(self, value):
        self.value = np.asarray(value, dtype=self.dtype).copy()


class RecoveryStateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.contract = {'code': 'v1', 'data': 'source-only'}
        self.arrays = {'model_0': np.arange(3, dtype=np.float32)}

    def test_roundtrip_best_file_and_retained_generations(self):
        best = self.root / 'best.keras'
        best.write_bytes(b'best-model-fixture')
        first = publish_recovery(self.root / 'recovery', self.contract, 0, {'history': []}, self.arrays)
        second = publish_recovery(self.root / 'recovery', self.contract, 1,
                                  {'history': [{'epoch': 0}]}, self.arrays, best)
        generation, state, arrays = load_recovery(self.root, self.contract)
        self.assertEqual(generation, second)
        self.assertTrue(first.exists())
        self.assertEqual(state['completed_epochs'], 1)
        self.assertEqual((generation / 'best.keras').read_bytes(), best.read_bytes())
        np.testing.assert_array_equal(arrays['model_0'], self.arrays['model_0'])

    def test_failed_pointer_publication_keeps_previous_epoch(self):
        first = publish_recovery(self.root, self.contract, 0, {'history': []}, self.arrays)
        from src.training import recovery_state
        original = recovery_state.atomic_json
        def interrupted(path, value):
            if Path(path).name == 'latest.json':
                raise OSError('simulated power loss')
            original(path, value)
        with mock.patch.object(recovery_state, 'atomic_json', side_effect=interrupted):
            with self.assertRaises(OSError):
                publish_recovery(self.root, self.contract, 1, {'history': [{'epoch': 0}]}, self.arrays)
        self.assertEqual(load_recovery(self.root, self.contract)[0], first)

    def test_corrupt_state_and_changed_contract_are_rejected(self):
        generation = publish_recovery(self.root, self.contract, 0, {'history': []}, self.arrays)
        with self.assertRaisesRegex(ValueError, 'contract changed'):
            load_recovery(self.root, {'code': 'v2', 'data': 'source-only'})
        (generation / 'arrays.npz').write_bytes(b'truncated')
        with self.assertRaisesRegex(ValueError, 'artifact changed'):
            load_recovery(self.root, self.contract)

    def test_scheduler_selects_greatest_published_epoch_without_touching_attempts(self):
        contract = {'code_sha256': code_hashes()}
        first = self.root / 'jobs/job/attempt_0001/artifacts'
        second = self.root / 'jobs/job/attempt_0002/artifacts'
        publish_recovery(first / 'recovery', contract, 1, {'history': [{'epoch': 0}]}, self.arrays)
        publish_recovery(second / 'recovery', contract, 0, {'history': []}, self.arrays)
        self.assertEqual(supervised_resume_source(self.root, 'job'), first)
        self.assertTrue((first / 'recovery/latest.json').exists())
        self.assertTrue((second / 'recovery/latest.json').exists())

    def test_pointer_cannot_escape_and_nonfinite_arrays_cannot_publish(self):
        (self.root / 'latest.json').write_text(json.dumps({'schema_version': 1, 'generation': '../outside'}))
        with self.assertRaises(ValueError):
            load_recovery(self.root)
        with self.assertRaisesRegex(ValueError, 'finite'):
            publish_recovery(self.root, self.contract, 0, {'history': []}, {'x': np.array([np.nan])})

    def test_variable_rng_callback_and_best_state_restore_without_optimizer(self):
        model_var = Variable('model/weight', [1., 2.])
        optimizer_var = Variable('optimizer/iteration', np.int64(7))
        rng = Variable('global_rng', np.array([12, 3], dtype=np.int64))
        tf_rng = SimpleNamespace(algorithm=1, state=Variable('tf_rng', np.array([1, 2, 3], dtype=np.int64)))
        early = SimpleNamespace(best=.7, wait=2, stopped_epoch=0, best_epoch=0, best_weights=[np.array([4., 5.])])
        reduce = SimpleNamespace(best=.4, wait=1, cooldown_counter=0)
        best = SimpleNamespace(path=self.root / 'model.keras', best=.7)
        best.path.write_bytes(b'best-fixture')
        callback = SupervisedRecovery(self.root, self.contract, best, early, reduce, tf_rng)
        callback.set_model(SimpleNamespace(variables=[model_var], optimizer=SimpleNamespace(variables=[optimizer_var]), stop_training=False))
        callback.history = [{'epoch': 0, 'val_macro_f1': .7}]
        random.seed(13)
        np.random.seed(13)
        with mock.patch('src.training.supervised_recovery.keras_global_rng', return_value=rng):
            callback.save()
            expected_random = (random.random(), np.random.random())
            model_var.assign([9., 9.])
            optimizer_var.assign(99)
            early.wait, reduce.wait = 99, 99
            rng.assign([9, 9])
            initial = callback.restore(self.root)
            self.assertEqual(initial, 1)
            self.assertEqual((random.random(), np.random.random()), expected_random)
        np.testing.assert_array_equal(model_var.numpy(), [1., 2.])
        self.assertEqual(optimizer_var.numpy(), 7)
        self.assertEqual((early.wait, reduce.wait), (2, 1))
        np.testing.assert_array_equal(rng.numpy(), [12, 3])

    def test_epoch_event_forwarding_does_not_repeat_begin_or_restore_best(self):
        callback = mock.Mock()
        wrapped = EpochEvents(SimpleNamespace(callbacks=[callback]))
        logs = {'loss': .5}
        wrapped.on_train_begin()
        wrapped.on_epoch_end(3, logs)
        wrapped.on_train_end()
        callback.on_train_begin.assert_not_called()
        callback.on_train_end.assert_not_called()
        callback.on_epoch_end.assert_called_once_with(3, logs)


class ValidationAndOwnershipTests(unittest.TestCase):
    def test_larger_validation_batches_preserve_targets_predictions_and_metrics(self):
        class Partition:
            split = 'val'
            def __len__(self):
                return 11
            def gather(self, indices, include_waveform=True):
                features = np.stack([np.full((2, 10), index / 11, dtype=np.float32) for index in indices])
                metadata = [{'dataset': 'sleepedf', 'participant_id': 'SC401', 'recording_id': 'SC4011',
                             'epoch_index': int(index), 'epoch_id': f'sleepedf:SC4011:{index:06d}',
                             'start_time': int(index) * 30, 'true_stage': ['W', 'N1', 'N2', 'N3', 'REM'][index % 5]}
                            for index in indices]
                return None, features, np.asarray(indices) % 5, metadata
        partition = Partition()
        normalizer = SimpleNamespace(transform_spectral=lambda value: value, assert_compatible=lambda partition: None)
        model = build_multiview_model(ModelConfig(variant='spectral', sequence_length=2, temporal_head='bilstm', temporal_units=4))
        with tempfile.TemporaryDirectory() as folder:
            first, second = Path(folder) / 'first.csv', Path(folder) / 'second.csv'
            reference = evaluate_batches(model, WindowBatches(partition, normalizer, variant='spectral', batch_size=4), first)
            compiled = evaluate_batches(model, WindowBatches(partition, normalizer, variant='spectral', batch_size=8), second, compiled=True)
            import pandas as pd
            left, right = pd.read_csv(first), pd.read_csv(second)
            self.assertEqual(left.epoch_id.tolist(), right.epoch_id.tolist())
            self.assertEqual(left.predicted_stage.tolist(), right.predicted_stage.tolist())
            np.testing.assert_allclose(left.filter(like='prob_'), right.filter(like='prob_'), atol=1e-6, rtol=1e-6)
            for key in ('macro_f1', 'accuracy', 'loss'):
                self.assertAlmostEqual(reference[key], compiled[key], places=6)
            self.assertEqual(reference['confusion_matrix'], compiled['confusion_matrix'])

    def test_compiled_inference_matches_eager_for_all_architectures_and_tail_batch(self):
        # Inference only: no compile(), fit(), training=True or optimizer exists.
        keras.utils.set_random_seed(42)
        for variant in ('raw', 'concat', 'gated', 'spectral'):
            with self.subTest(variant=variant):
                config = ModelConfig(variant=variant, temporal_head='bilstm', sequence_length=2,
                                     samples_per_epoch=100, filters=(4,), kernels=(7,), strides=(5,),
                                     token_dim=8, attention_heads=2, temporal_units=4, dropout=.3)
                model = build_multiview_model(config)
                predict = validation_predictor(model, True)
                for size in (4, 1):
                    inputs = {}
                    if variant != 'spectral':
                        inputs['waveform'] = np.random.default_rng(1).normal(size=(size, 2, 100, 1)).astype('float32')
                    if variant != 'raw':
                        inputs['spectral'] = np.random.default_rng(2).normal(size=(size, 2, 10)).astype('float32')
                    expected = model(inputs, training=False).numpy()
                    actual = predict(inputs).numpy()
                    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)
                    np.testing.assert_array_equal(actual.argmax(1), expected.argmax(1))
                self.assertEqual(predict.experimental_get_tracing_count(), 1)

    def test_bad_config_and_capped_mask_changes_fail_before_data_or_training(self):
        for change in ({'validation_batch_size': 0}, {'recovery': 'yes'}, {'compiled_validation': 1}):
            with self.assertRaises(ValueError):
                TrainingConfig(**change)
        with self.assertRaisesRegex(ValueError, 'capped validation'):
            train_supervised('unused', 'unused', training_config=TrainingConfig(validation_batch_size=8), max_validation_batches=2)
        with self.assertRaisesRegex(ValueError, 'requires'):
            train_supervised('unused', 'unused', resume_from='unused')

    def test_live_lock_is_never_archived_and_dead_lock_bytes_are_preserved(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / '.runner.lock'
            path.write_text('123')
            with mock.patch('src.experiments.run.psutil.pid_exists', return_value=True):
                with self.assertRaisesRegex(ValueError, 'alive'):
                    archive_stale_lock(folder)
            self.assertTrue(path.exists())
            with mock.patch('src.experiments.run.psutil.pid_exists', return_value=False), \
                 mock.patch('src.experiments.run.psutil.process_iter', return_value=[
                     SimpleNamespace(pid=os.getpid(), info={'cmdline': ['python', '-m', 'src.experiments.run']})]):
                archived = archive_stale_lock(folder)
            self.assertEqual(archived.read_text(), '123')
            self.assertFalse(path.exists())

    def test_explicit_recovery_requires_retry_and_cli_only_forwards(self):
        with self.assertRaisesRegex(ValueError, 'retry-failed'):
            execute_study('unused', resume_supervised=True)
        with mock.patch('src.experiments.run.execute_study', return_value={}) as execute, redirect_stdout(io.StringIO()):
            main(['execute', '--study-dir', 'unused', '--retry-failed', '--resume-supervised', '--recover-stale-lock'])
        self.assertTrue(execute.call_args.kwargs['resume_supervised'])
        self.assertTrue(execute.call_args.kwargs['recover_stale_lock'])


if __name__ == '__main__':
    unittest.main()

"""Live-output handoff checks with no model construction or optimizer updates."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from src.experiments.run import execute_study, main, run_worker_live
from src.experiments.worker import run_task
from src.training.progress import BatchProgress
from src.training.supervised import SourceTrainingProgress


class TerminalProgressTests(unittest.TestCase):
    def test_progress_reports_first_interval_and_last_batches_with_metrics(self):
        output = io.StringIO()
        with mock.patch('src.training.progress.time.perf_counter', side_effect=[0., 1., 100., 101.]), redirect_stdout(output):
            progress = BatchProgress('SSL', 1, 10, 101, 100)
            for batch in range(1, 102):
                progress.update(batch, {'loss': .25, 'retrieval_accuracy': .5})
        text = output.getvalue()
        self.assertIn('batch 1/101', text)
        self.assertIn('batch 100/101', text)
        self.assertIn('batch 101/101', text)
        self.assertNotIn('batch 2/101', text)
        self.assertIn('loss=0.25000', text)
        self.assertIn('ETA~00:00:00', text)

    def test_disabled_progress_is_silent_and_bad_intervals_fail(self):
        output = io.StringIO()
        with redirect_stdout(output):
            BatchProgress('SSL', 1, 1, 3).update(1, {'loss': 1.})
        self.assertEqual(output.getvalue(), '')
        for interval in [-1, True, '100']:
            with self.assertRaises(ValueError):
                BatchProgress('SSL', 1, 1, 3, interval)

    def test_supervised_callback_reports_batches_and_validation_handoff(self):
        output = io.StringIO()
        with redirect_stdout(output):
            callback = SourceTrainingProgress(2, 3, 100)
            callback.on_epoch_begin(0)
            callback.on_train_batch_end(0, {'loss': 1., 'accuracy': .5})
            callback.on_train_batch_end(1, {'loss': .8, 'accuracy': .6})
        self.assertIn('Supervised epoch 1/3 batch 2/2', output.getvalue())
        self.assertIn('computing source-validation metrics', output.getvalue())

    def test_live_worker_keeps_stdout_stderr_and_exit_code_in_terminal_and_log(self):
        terminal, log = io.StringIO(), io.StringIO()
        # This child only prints strings; it imports no project/model code.
        command = [sys.executable, '-u', '-c', "import sys; print('first'); print('second', file=sys.stderr); sys.exit(7)"]
        with redirect_stdout(terminal):
            code = run_worker_live(command, log)
        self.assertEqual(code, 7)
        self.assertEqual(terminal.getvalue(), log.getvalue())
        self.assertIn('first', log.getvalue())
        self.assertIn('second', log.getvalue())

    def test_interrupt_cleans_up_its_worker_tree_and_reraises(self):
        class Interrupted:
            def __iter__(self):
                raise KeyboardInterrupt()
            def close(self):
                pass
        process = SimpleNamespace(stdout=Interrupted())
        with mock.patch('src.experiments.run.subprocess.Popen', return_value=process), \
             mock.patch('src.experiments.run.stop_worker_tree') as stop:
            with self.assertRaises(KeyboardInterrupt):
                run_worker_live(['unused'], io.StringIO())
        stop.assert_called_once_with(process)

    def test_cli_forwards_live_without_scheduling_training(self):
        with mock.patch('src.experiments.run.execute_study', return_value={} ) as execute, redirect_stdout(io.StringIO()):
            main(['execute', '--study-dir', 'unused', '--live'])
        self.assertTrue(execute.call_args.kwargs['live'])
        self.assertFalse(execute.call_args.kwargs['require_gpu'])
        with self.assertRaisesRegex(ValueError, 'live'):
            execute_study('unused', live='yes')

    def test_workers_enable_periodic_progress_only_for_live_tasks(self):
        for kind in ['pretrain', 'supervised']:
            for live in [False, True]:
                with self.subTest(kind=kind, live=live), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    dependencies = {'waveform_statistics': str(root / 'waveform/artifacts')} if kind == 'pretrain' else {
                        'normalizer_draw42_n1': str(root / 'normalizer/artifacts')}
                    job = {'job_id': 'job', 'kind': kind, 'seed': 42, 'dependencies': list(dependencies),
                           'budget_id': 'draw42_n1', 'variant': 'gated', 'pretrained': False}
                    study = {'fingerprint': 'fixture', 'jobs': [job], 'config': {
                        'model': {}, 'pretraining': {}, 'augmentation': {}, 'training': {}, 'train_participants': None,
                        'limits': {'max_pretraining_batches': None, 'max_train_batches': None, 'max_validation_batches': None}}}
                    plan = {'source_records': [{'recording_id': 'SC4001', 'split': 'train'}]}
                    task = root / 'task.json'
                    task.write_text(json.dumps({'study_directory': str(root), 'job_id': 'job', 'dependencies': dependencies,
                                                'data_root': str(root), 'split_file': str(root / 'split.csv'), 'live': live}))
                    with mock.patch('src.experiments.worker.load_study', return_value=(study, plan)), \
                         mock.patch('src.experiments.worker.verify_runtime'), mock.patch('src.experiments.worker.verify_job_result'), \
                         mock.patch('src.experiments.worker.train_contrastive') as ssl, \
                         mock.patch('src.experiments.worker.train_supervised') as supervised:
                        run_task(task)
                    selected, other = (ssl, supervised) if kind == 'pretrain' else (supervised, ssl)
                    selected.assert_called_once()
                    other.assert_not_called()
                    self.assertEqual(selected.call_args.kwargs['progress_interval'], 100 if live else 0)


if __name__ == '__main__':
    unittest.main()

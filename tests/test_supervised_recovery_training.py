"""USER-RUN ONLY: optimizer updates on small synthetic EEG, fresh worker processes."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

import numpy as np
import pandas as pd

from src.training.recovery_state import load_recovery


class SupervisedRecoveryTrainingTests(unittest.TestCase):
    def test_fresh_process_resume_matches_dropout_optimizer_callbacks_and_best_model(self):
        from tests.test_experiments import make_fixture
        from src.inference.multiview import MultiviewPredictor
        root = Path(__file__).resolve().parents[1]
        # Retain every worker log and synthetic artifact for inspection; no research data.
        import uuid
        output = root / 'runs/recovery_gate' / uuid.uuid4().hex
        output.mkdir(parents=True)
        data = output / 'synthetic_data'
        data.mkdir()
        make_fixture(data)
        environment = {**os.environ, 'PYTHONHASHSEED': '42', 'TF_NUM_INTRAOP_THREADS': '1', 'TF_NUM_INTEROP_THREADS': '1'}
        print(f'User-run recovery gate artifacts: {output}', flush=True)
        for name, flags in [('baseline', []), ('interrupted', ['--interrupt']),
                             ('resumed', ['--resume-from', str(output / 'interrupted')])]:
            command = [sys.executable, '-u', '-X', 'utf8', '-m', 'tests.recovery_training_worker',
                       '--data-root', str(data), '--run-dir', str(output / name), *flags]
            with (output / f'{name}.log').open('w', encoding='utf-8') as log:
                result = subprocess.run(command, cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT)
            self.assertEqual(result.returncode, 0, f'See {output / (name + ".log")}')
        _, before, before_arrays = load_recovery(output / 'baseline')
        _, after, after_arrays = load_recovery(output / 'resumed')
        self.assertEqual(before['completed_epochs'], 4)
        self.assertEqual(after['completed_epochs'], 4)
        self.assertEqual(before['early'], after['early'])
        self.assertEqual(before['reduce'], after['reduce'])
        self.assertEqual(before['early']['wait'], 2)
        self.assertTrue(before['stop_training'])
        self.assertEqual(before['history'], after['history'])
        self.assertEqual(before['python_rng'], after['python_rng'])
        self.assertEqual(before['numpy_rng'], after['numpy_rng'])
        self.assertEqual(set(before_arrays), set(after_arrays))
        for name in before_arrays:
            if np.issubdtype(before_arrays[name].dtype, np.integer):
                np.testing.assert_array_equal(before_arrays[name], after_arrays[name], err_msg=name)
            else:
                np.testing.assert_allclose(before_arrays[name], after_arrays[name], atol=1e-7, rtol=1e-7, err_msg=name)
        self.assertTrue(any('seed_generator' in value['path'] for value in before['model_layout']))
        for index, variable in enumerate(before['optimizer_layout']):
            if variable['path'].endswith('/iteration'):
                self.assertEqual(int(before_arrays[f'optimizer_{index}']), 12)
            if variable['path'].endswith('/learning_rate'):
                self.assertAlmostEqual(float(before_arrays[f'optimizer_{index}']), .0005, places=8)
        baseline = json.loads((output / 'baseline/summary.json').read_text())
        resumed = json.loads((output / 'resumed/summary.json').read_text())
        self.assertEqual((baseline['best_epoch'], resumed['best_epoch']), (2, 2))
        self.assertEqual(resumed['resumed_after_epoch'], 3)
        # Final exports are real inference on the same synthetic validation fixture.
        left = pd.read_csv(output / 'baseline/validation_predictions.csv')
        right = pd.read_csv(output / 'resumed/validation_predictions.csv')
        self.assertEqual(left.epoch_id.tolist(), right.epoch_id.tolist())
        np.testing.assert_allclose(left.filter(like='prob_'), right.filter(like='prob_'), atol=1e-7, rtol=1e-7)
        # Also load the sealed models through the production inference loader.
        for name in ('baseline', 'resumed'):
            MultiviewPredictor(output / name)
        print('Resume equivalence gate passed on synthetic data; this is not research performance.', flush=True)


if __name__ == '__main__':
    unittest.main()

"""Inspect user-produced synthetic gate artifacts; never execute training."""
import importlib.metadata
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.experiments.run import code_hashes
from src.training.recovery_state import load_recovery
from src.data.epoch_store import sha256_file

gate = ROOT / 'runs/recovery_gate/3149ea07f23440278975a4e95008ac5a'
baseline_gen, baseline, baseline_arrays = load_recovery(gate / 'baseline')
resumed_gen, resumed, resumed_arrays = load_recovery(gate / 'resumed')
_, interrupted, _ = load_recovery(gate / 'interrupted')
assert baseline['contract_sha256'] == resumed['contract_sha256'] == interrupted['contract_sha256']
assert baseline['contract']['code_sha256'] == code_hashes()
packages = {dist.metadata['Name']: dist.version for dist in importlib.metadata.distributions()}
assert baseline['contract']['packages'] == packages
assert baseline['completed_epochs'] == resumed['completed_epochs'] == 4
assert interrupted['completed_epochs'] == 3
for key in ('history', 'early', 'reduce', 'python_rng', 'numpy_rng', 'stop_training', 'best_score'):
    assert baseline[key] == resumed[key], key
assert baseline['stop_training'] and baseline['early']['best_epoch'] == 1
assert baseline['early']['wait'] == 2
assert set(baseline_arrays) == set(resumed_arrays)
maximum_float_difference = 0.
integer_count = 0
for key, value in baseline_arrays.items():
    other = resumed_arrays[key]
    if np.issubdtype(value.dtype, np.integer):
        np.testing.assert_array_equal(value, other, err_msg=key)
        integer_count += 1
    else:
        np.testing.assert_allclose(value, other, atol=1e-7, rtol=1e-7, err_msg=key)
        maximum_float_difference = max(maximum_float_difference, float(np.max(np.abs(value - other))))
scalars = {}
for index, variable in enumerate(baseline['optimizer_layout']):
    if variable['path'].endswith('/iteration'):
        scalars['iterations'] = int(baseline_arrays[f'optimizer_{index}'])
    elif variable['path'].endswith('/learning_rate'):
        scalars['learning_rate'] = float(baseline_arrays[f'optimizer_{index}'])
assert scalars['iterations'] == 12
assert np.isclose(scalars['learning_rate'], .0005)
left = pd.read_csv(gate / 'baseline/validation_predictions.csv')
right = pd.read_csv(gate / 'resumed/validation_predictions.csv')
assert left.epoch_id.tolist() == right.epoch_id.tolist()
assert left.predicted_stage.tolist() == right.predicted_stage.tolist()
np.testing.assert_allclose(left.filter(like='prob_'), right.filter(like='prob_'), atol=1e-7, rtol=1e-7)
probability_difference = float(np.max(np.abs(left.filter(like='prob_').to_numpy() - right.filter(like='prob_').to_numpy())))
for name in ('baseline', 'resumed'):
    summary = json.loads((gate / name / 'summary.json').read_text())
    assert summary['best_epoch'] == 2 and summary['epochs_completed'] == 4
    if name == 'resumed':
        assert summary['resumed_after_epoch'] == 3
    with zipfile.ZipFile(gate / name / 'model.keras') as archive:
        assert archive.testzip() is None
assert 'tensorflow' not in sys.modules
evidence = {
    'passed': True, 'training_executed_by': 'user',
    'gate_directory': str(gate.relative_to(ROOT)), 'user_reported_test_seconds': 42.455,
    'interrupted_after_epoch': 3, 'baseline_and_resumed_completed_epochs': 4,
    'best_epoch': 2, 'final_optimizer': scalars,
    'arrays_checked': len(baseline_arrays), 'integer_arrays_equal': integer_count,
    'maximum_float_array_difference': maximum_float_difference,
    'maximum_exported_probability_difference': probability_difference,
    'python_numpy_and_callback_state_equal': True, 'checkpoint_seals_verified': True,
    'source_and_packages_match_gate_contract': True,
    'gate_contract_sha256': baseline['contract_sha256'],
    'worker_log_sha256': {name: sha256_file(gate / f'{name}.log') for name in ('baseline', 'interrupted', 'resumed')},
    'assistant_optimizer_updates': False, 'tensorflow_imported_by_inspection': False,
    'full_regression_passed': None, 'real_data_parity_passed': None, 'speedup_measured': False,
    'scope': 'One tiny synthetic gated-BiLSTM, dropout, CPU, fixed runtime. Not historical-implementation equivalence or arbitrary-device recovery.'}
with (Path(__file__).parent / 'verification.json').open('x', encoding='utf-8') as stream:
    json.dump(evidence, stream, indent=2, allow_nan=False)
    stream.write('\n')
print(json.dumps(evidence, indent=2))

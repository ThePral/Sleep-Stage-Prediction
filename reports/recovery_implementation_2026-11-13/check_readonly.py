"""Run explicit non-optimizer checks and preserve historical artifact evidence."""
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.experiments.run import code_hashes, load_study, normalize_config, verify_job_result
from src.data.epoch_store import sha256_file

out = Path(__file__).resolve().parent
for name in ['tests/recovery_training_worker.py', 'tests/test_supervised_recovery_training.py',
             'src/training/recovery_state.py', 'src/training/supervised_recovery.py']:
    compile((ROOT / name).read_text(encoding='utf-8'), name, 'exec')
recipe = normalize_config(json.loads((ROOT / 'configs/sleepedf/experiments/experiment_bilstm_recovery_pilot.json').read_text()))
assert recipe['training']['recovery'] and recipe['training']['compiled_validation']
suite = unittest.defaultTestLoader.loadTestsFromNames([
    'tests.test_supervised_recovery', 'tests.test_terminal_progress', 'tests.test_training_handoff'])
with (out / 'checks.txt').open('w', encoding='utf-8') as stream:
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
if not result.wasSuccessful():
    print((out / 'checks.txt').read_text(), flush=True)
    raise SystemExit(1)
directory = ROOT / 'runs/bilstm_terminal_2026-11-02'
study, _ = load_study(directory)
ledger = json.loads((directory / 'ledger.json').read_text())
for job in study['jobs']:
    entry = ledger['jobs'][job['job_id']]
    assert entry['status'] == 'completed'
    verify_job_result(directory / entry['attempt'], study, job['job_id'])
baseline = json.loads((ROOT / 'reports/implementation_2026-10-27/evidence.json').read_text())['preserved_artifact_hashes']
assert all(sha256_file(ROOT / name) == digest for name, digest in baseline.items())
evidence = {'non_optimizer_tests_passed': result.testsRun, 'optimizer_updates_performed': False,
            'training_equivalence_gate_executed': False, 'full_regression_suite_executed': False,
            'training_gate_sources_compile': True, 'fresh_pilot_config_validated_without_execution': True,
            'historical_completed_job_inventories_verified': len(study['jobs']),
            'historical_baseline_hashes_match': True, 'current_code_sha256': code_hashes(),
            'historical_study_code_matches_current': study['code_sha256'] == code_hashes(),
            'new_studies_required': True,
            'claims': 'Atomic-state and inference checks pass. Training resume equivalence, real speedup and full regression await user-run gates.'}
with (out / 'evidence.json').open('w', encoding='utf-8') as stream:
    json.dump(evidence, stream, indent=2)
    stream.write('\n')
print(json.dumps({key: value for key, value in evidence.items() if key != 'current_code_sha256'}, indent=2), flush=True)

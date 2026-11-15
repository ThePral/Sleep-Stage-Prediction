"""Inspect a user-run regression log; run only the new pure parity-helper tests."""
import importlib.metadata
import json
from pathlib import Path
import re
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.epoch_store import sha256_file
from src.experiments.run import code_hashes, load_study, verify_job_result
from src.training.recovery_state import load_recovery

out = Path(__file__).resolve().parent
log = ROOT / 'reports/recovery_validation_2026-11-13/full_regression_01.log'
payload = log.read_bytes()
content = payload.decode('utf-16' if payload.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8-sig')
match = re.search(r'Ran\s+(\d+)\s+tests\s+in\s+([\d.]+)s\s+OK\s*$', content)
assert match, 'No final successful unittest summary in the retained log.'
assert int(match.group(1)) == 136 and float(match.group(2)) == 147.021
original = json.loads((ROOT / 'reports/recovery_implementation_2026-11-13/evidence.json').read_text())
assert original['current_code_sha256'] == code_hashes()
_, gate, _ = load_recovery(ROOT / 'runs/recovery_gate/3149ea07f23440278975a4e95008ac5a/resumed')
assert gate['contract']['code_sha256'] == code_hashes()
packages = {dist.metadata['Name']: dist.version for dist in importlib.metadata.distributions()}
assert packages == gate['contract']['packages']
study_dir = ROOT / 'runs/bilstm_terminal_2026-11-02'
study, _ = load_study(study_dir)
ledger = json.loads((study_dir / 'ledger.json').read_text())
for job in study['jobs']:
    entry = ledger['jobs'][job['job_id']]
    assert entry['status'] == 'completed'
    verify_job_result(study_dir / entry['attempt'], study, job['job_id'])
baseline = json.loads((ROOT / 'reports/implementation_2026-10-27/evidence.json').read_text())['preserved_artifact_hashes']
assert all(sha256_file(ROOT / name) == digest for name, digest in baseline.items())
suite = unittest.defaultTestLoader.loadTestsFromName('tests.test_validation_parity')
with (out / 'parity_helper_tests.txt').open('w', encoding='utf-8') as stream:
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
assert result.wasSuccessful() and result.testsRun == 5
assert 'tensorflow' not in sys.modules
evidence = {'full_regression_passed': True, 'tests_reported': int(match.group(1)),
            'elapsed_seconds_reported': float(match.group(2)), 'full_regression_executed_by': 'user',
            'log': str(log.relative_to(ROOT)), 'log_sha256': sha256_file(log),
            'source_matches_prior_gate_and_readonly_audit': True, 'packages_match_prior_gate': True,
            'new_script_helper_tests_passed': 5, 'helper_tests_performed_optimizer_updates': False,
            'new_helper_tests_added_after_user_full_suite': True,
            'completed_historical_job_inventories_verified': 18, 'baseline_hashes_preserved': True,
            'tensorflow_imported_by_inspection': False,
            'real_data_parity_executed': False, 'performance_measured': False,
            'research_training_launched': False, 'test_or_ISRUC_scoring_performed': False}
with (out / 'verification.json').open('x', encoding='utf-8') as stream:
    json.dump(evidence, stream, indent=2)
    stream.write('\n')
print(json.dumps(evidence, indent=2))

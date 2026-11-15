"""Inspect user-produced parity evidence and preserved artifacts; no inference."""
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.validation_parity import require_target_identity
from src.data.epoch_store import sha256_file
from src.experiments.run import code_hashes, load_study, verify_job_result

OUTPUT = Path(__file__).resolve().parent
INPUT = ROOT / 'reports/validation_parity_2026-11-15_m3_eager_full'
STUDY = ROOT / 'runs/bilstm_terminal_2026-11-02'


def main():
    result = json.loads((INPUT / 'result.json').read_text(encoding='utf-8'))
    policy = json.loads((INPUT / 'policy.json').read_text(encoding='utf-8'))
    assert policy == result['policy']
    assert result['passed'] and result['full_cohort_checked'] and not result['diagnostic_only']
    assert not result['optimizer_updates_performed'] and not result['test_or_ISRUC_scoring']
    assert policy['execution_mode'] == 'eager' and policy['eager_batch_size'] == 4
    assert policy['compiled_batch_size'] is None and policy['samples'] == 'all'
    assert policy['atol'] == policy['rtol'] == 1e-6
    assert code_hashes() == policy['current_source_sha256']
    assert sha256_file(ROOT / 'scripts/validation_parity.py') == policy['script_sha256']
    assert len(result['models']) == 1
    model = result['models'][0]
    assert model['job_id'] == 'M3_draw42_n70_seed42' and model['examples'] == 65133
    check = model['saved_vs_eager']
    assert model['passed'] and check['passed'] and check['probabilities_within_tolerance']
    assert check['argmax_mismatches'] == 0 and check['confusion_matrices_equal']
    assert check['metric_differences']['accuracy'] == check['metric_differences']['macro_f1'] == 0
    assert abs(check['metric_differences']['loss']) <= policy['atol']
    assert 'saved_vs_compiled' not in model
    study, _ = load_study(STUDY)
    assert study['fingerprint'] == policy['study_fingerprint']
    ledger = json.loads((STUDY / 'ledger.json').read_text(encoding='utf-8'))
    entry = ledger['jobs'][model['job_id']]
    assert entry['status'] == 'completed'
    attempt = STUDY / entry['attempt']
    verify_job_result(attempt, study, model['job_id'])
    reference = attempt / 'artifacts/validation_predictions.csv'
    assert sha256_file(reference) == model['export_sha256']
    targets = pd.read_csv(INPUT / 'selected_targets.csv')
    assert len(targets) == model['examples'] and targets.epoch_id.is_unique
    require_target_identity(pd.read_csv(reference), targets)
    supervised = [job['job_id'] for job in study['jobs'] if job['kind'] == 'supervised']
    assert len(supervised) == 14 and model['job_id'] in supervised
    remaining = [name for name in supervised if name != model['job_id']]
    assert all(ledger['jobs'][name]['status'] == 'completed' for name in remaining)
    (OUTPUT / 'remaining_models.txt').write_text('\n'.join(remaining) + '\n', encoding='utf-8')
    evidence = {'review_date': '2026-11-15', 'input_sha256': {
                    name: sha256_file(INPUT / name) for name in ('result.json', 'policy.json', 'selected_targets.csv')},
                'current_source_and_parity_script_match': True,
                'preserved_m3_job_seal_verified': True, 'all_target_ids_labels_and_order_match_saved_export': True,
                'model_id': model['job_id'], 'targets': model['examples'], 'execution_mode': 'eager', 'batch_size': 4,
                'comparison': check, 'inference_seconds': model['eager_seconds'],
                'total_reviewed_execution_seconds': result['elapsed_seconds'],
                'full_cohort_eager_passed_models': [model['job_id']], 'remaining_models': remaining,
                'compiled_parity_accepted': False, 'full_family_parity_accepted': False,
                'optimizer_updates_by_reviewer': False, 'model_inference_by_reviewer': False,
                'tensorflow_imported_by_reviewer': 'tensorflow' in sys.modules,
                'training_speedup_established': False}
    assert not evidence['tensorflow_imported_by_reviewer']
    (OUTPUT / 'verification.json').write_text(json.dumps(evidence, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'verified': True, 'full_cohort_eager_models': 1, 'remaining_models': len(remaining),
                      'maximum_probability_difference': check['maximum_probability_difference']}))


if __name__ == '__main__':
    main()

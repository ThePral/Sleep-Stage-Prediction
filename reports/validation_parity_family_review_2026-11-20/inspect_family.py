"""Verify retained user-run parity and artifact seals without model execution."""
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
STUDY = ROOT / 'runs/bilstm_terminal_2026-11-02'
INPUTS = [ROOT / 'reports/validation_parity_2026-11-15_m3_eager_full',
          ROOT / 'reports/validation_parity_2026-11-17_remaining_eager_full']


def main():
    study, _ = load_study(STUDY)
    ledger = json.loads((STUDY / 'ledger.json').read_text(encoding='utf-8'))
    expected = {job['job_id'] for job in study['jobs'] if job['kind'] == 'supervised'}
    source_hashes = code_hashes()
    rows, inputs, seen = [], [], set()
    cohort = None
    for directory in INPUTS:
        result = json.loads((directory / 'result.json').read_text(encoding='utf-8'))
        policy = json.loads((directory / 'policy.json').read_text(encoding='utf-8'))
        assert policy == result['policy']
        assert result['passed'] and result['full_cohort_checked'] and not result['diagnostic_only']
        assert not result['optimizer_updates_performed'] and not result['test_or_ISRUC_scoring']
        assert policy['execution_mode'] == 'eager' and policy['eager_batch_size'] == 4
        assert policy['samples'] == 'all' and policy['compiled_batch_size'] is None
        assert policy['cohort'] == 'source_validation_only'
        assert policy['atol'] == policy['rtol'] == 1e-6
        assert policy['study_fingerprint'] == study['fingerprint']
        assert policy['current_source_sha256'] == source_hashes
        assert policy['script_sha256'] == sha256_file(ROOT / 'scripts/validation_parity.py')
        assert policy['model_ids'] == [model['job_id'] for model in result['models']]
        targets = pd.read_csv(directory / 'selected_targets.csv')
        assert len(targets) == 65133 and targets.epoch_id.is_unique
        if cohort is not None:
            require_target_identity(cohort, targets)
        cohort = targets
        inputs.append({'directory': directory.relative_to(ROOT).as_posix(),
                       'observed_at_utc': result['observed_at_utc'],
                       'elapsed_seconds': result['elapsed_seconds'],
                       'files_sha256': {name: sha256_file(directory / name) for name in
                                        ('result.json', 'policy.json', 'selected_targets.csv')}})
        for model in result['models']:
            job_id = model['job_id']
            assert job_id in expected and job_id not in seen
            seen.add(job_id)
            check = model['saved_vs_eager']
            assert model['passed'] and model['examples'] == len(targets)
            assert check['passed'] and check['probabilities_within_tolerance']
            assert check['argmax_mismatches'] == 0 and check['confusion_matrices_equal']
            assert check['metric_differences']['accuracy'] == check['metric_differences']['macro_f1'] == 0
            assert abs(check['metric_differences']['loss']) <= policy['atol']
            assert 'saved_vs_compiled' not in model
            entry = ledger['jobs'][job_id]
            assert entry['status'] == 'completed'
            attempt = STUDY / entry['attempt']
            verify_job_result(attempt, study, job_id)
            reference = attempt / 'artifacts/validation_predictions.csv'
            assert sha256_file(reference) == model['export_sha256']
            require_target_identity(pd.read_csv(reference), targets)
            rows.append({'job_id': job_id, 'examples': model['examples'],
                         'comparison': check, 'eager_seconds': model['eager_seconds']})
    assert seen == expected and len(seen) == 14
    evidence = {'review_date': '2026-11-20', 'study_fingerprint': study['fingerprint'],
                'inputs': inputs, 'full_family_eager_parity_passed': True,
                'models_verified': len(rows), 'targets_per_model': len(cohort),
                'model_target_evaluations': sum(item['examples'] for item in rows),
                'execution_mode': 'eager', 'batch_size': 4, 'atol': 1e-6, 'rtol': 1e-6,
                'maximum_probability_difference': max(item['comparison']['maximum_probability_difference'] for item in rows),
                'maximum_absolute_loss_delta': max(abs(item['comparison']['metric_differences']['loss']) for item in rows),
                'total_argmax_mismatches': sum(item['comparison']['argmax_mismatches'] for item in rows),
                'all_target_ids_labels_order_and_artifact_seals_verified': True,
                'current_source_and_script_match_both_parity_runs': True,
                'models': rows, 'compiled_validation_accepted': False,
                'training_speedup_established': False, 'real_data_recovery_training_verified': False,
                'optimizer_updates_by_reviewer': False, 'inference_by_reviewer': False,
                'tensorflow_imported_by_reviewer': 'tensorflow' in sys.modules}
    assert not evidence['tensorflow_imported_by_reviewer']
    (OUTPUT / 'verification.json').write_text(json.dumps(evidence, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({name: evidence[name] for name in ('full_family_eager_parity_passed', 'models_verified',
                     'model_target_evaluations', 'maximum_probability_difference', 'total_argmax_mismatches')}))


if __name__ == '__main__':
    main()

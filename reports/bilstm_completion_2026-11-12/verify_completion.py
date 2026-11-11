"""Audit completed source study and analyze saved validation predictions only."""
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import sys
import zipfile

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.epoch_store import sha256_file
from src.data.label_budgets import authorize_waveform_pool
from src.experiments.run import load_study, verify_job_result, verify_runtime
from src.evaluation.artifacts import inspect_bundle
from src.evaluation.uncertainty import AnalysisConfig, analyze_registry
from src.preprocessing.waveform_normalization import WaveformNormalizer

REPORT = Path(__file__).resolve().parent
STUDY = ROOT / 'runs/bilstm_terminal_2026-11-02'
study, plan = load_study(STUDY)
verify_runtime(study, plan, Path(study['data_root']), Path(study['split_file']))
ledger = json.loads((STUDY / 'ledger.json').read_text())
assert len(ledger['jobs']) == len(study['jobs']) == 18
assert not (STUDY / '.runner.lock').exists()
assert not study['limited_run']
jobs = []
summaries = {}
registries = {budget['budget_id']: [] for budget in plan['budgets']}
normalizers = {key: set() for key in registries}
ssl_artifacts = STUDY / ledger['jobs']['ssl_seed42']['attempt'] / 'artifacts'
ssl_bundle_hash = sha256_file(ssl_artifacts / 'pretraining_bundle.json')
ssl_bundle = json.loads((ssl_artifacts / 'pretraining_bundle.json').read_text())
ssl_waveform = WaveformNormalizer.load(ssl_artifacts / 'waveform_normalizer.json')
authorize_waveform_pool(plan, ssl_waveform)
for job in study['jobs']:
    entry = ledger['jobs'][job['job_id']]
    assert entry['status'] == 'completed'
    attempt = STUDY / entry['attempt']
    seal = verify_job_result(attempt, study, job['job_id'])
    artifacts = attempt / 'artifacts'
    summary = json.loads((artifacts / 'summary.json').read_text())
    summaries[job['job_id']] = summary
    row = {'job_id': job['job_id'], 'kind': job['kind'], 'attempt': entry['attempt'],
           'artifact_files_verified': len(seal['files'])}
    for key in ('epochs_completed', 'best_epoch', 'elapsed_seconds', 'elapsed_seconds_this_run', 'optimizer_iterations'):
        if key in summary:
            row[key] = summary[key]
    if job['kind'] == 'supervised':
        inspected = inspect_bundle(artifacts)
        assert inspected['normalizer'].state['pretraining_waveform'] == ssl_waveform.state
        assert inspected['config']['temporal_head'] == 'bilstm'
        assert inspected['config']['temporal_units'] == 32
        assert not summary['limited_data_run']
        history = pd.read_csv(artifacts / 'training_history.csv')
        assert len(history) == summary['epochs_completed']
        assert history.epoch.tolist() == list(range(len(history)))
        assert np.isfinite(history.to_numpy()).all()
        assert int(history.val_macro_f1.to_numpy().argmax()) + 1 == summary['best_epoch']
        assert np.isclose(history.val_macro_f1.max(), summary['source_validation']['macro_f1'], atol=1e-12, rtol=1e-12)
        checked = []
        with zipfile.ZipFile(artifacts / 'model.keras') as archive:
            assert archive.testzip() is None
            with h5py.File(io.BytesIO(archive.read('model.weights.h5')), 'r') as weights:
                def inspect_weight(name, tensor):
                    if isinstance(tensor, h5py.Dataset):
                        assert np.isfinite(np.asarray(tensor)).all(), name
                        checked.append(name)
                weights.visititems(inspect_weight)
        if job['pretrained']:
            transfer = inspected['resolved']['pretraining_transfer']
            assert transfer['bundle_sha256'] == ssl_bundle_hash
            assert transfer['encoder_sha256'] == ssl_bundle['files']['encoder.keras']
            assert transfer['frozen_waveform_encoder'] is False
        budget_id = job['budget_id']
        normalizers[budget_id].add(sha256_file(artifacts / 'normalizer.json'))
        registries[budget_id].append({'model_id': job['job_id'], 'condition': job['condition'], 'training_seed': job['seed'],
                                      'predictions': str(artifacts / 'validation_predictions.csv'),
                                      'bundle_directory': str(artifacts)})
        row.update(condition=job['condition'], budget_id=budget_id,
                   validation_macro_f1=summary['source_validation']['macro_f1'],
                   checkpoint_tensors_checked=len(checked), checkpoint_zip_and_tensors_valid=True)
    jobs.append(row)

assert all(len(values) == 1 for values in normalizers.values())
assert all(len(rows) == 7 for rows in registries.values())
assert summaries['ssl_seed42']['epochs_completed'] == 10
assert summaries['ssl_seed42']['optimizer_iterations'] == 101060
print('Verified 18 jobs, 14 model archives, histories, normalization and common SSL transfer.', flush=True)

config = AnalysisConfig(bootstrap_replicates=2000, confidence_level=.95, seed=42, calibration_bins=15)
pairs = [('M4', 'M1'), ('M5', 'M2'), ('M6', 'M3'), ('M3', 'M2'), ('M6', 'M5')]
condition_rows, paired_rows, class_rows = [], [], []
identity = None
for budget_id, records in registries.items():
    registry = REPORT / f'{budget_id}_registry.csv'
    pd.DataFrame(records).to_csv(registry, index=False)
    print(f'Analyzing {budget_id}: saved validation predictions, 2000 participant resamples.', flush=True)
    analysis = analyze_registry(registry, REPORT / f'{budget_id}_analysis', pairs=pairs, config=config)
    assert analysis['participants'] == 15
    if identity is None:
        identity = analysis['target_identity_sha256']
    assert analysis['target_identity_sha256'] == identity
    for record in records:
        metrics = analysis['models'][record['model_id']]
        summary = summaries[record['model_id']]
        assert metrics['examples'] == 65133
        for key in ('macro_f1', 'accuracy', 'balanced_accuracy', 'cohen_kappa'):
            assert np.isclose(metrics[key], summary['source_validation'][key], atol=1e-12, rtol=1e-12)
        estimate = analysis['conditions'][record['condition']]
        condition_rows.append({'budget_id': budget_id, 'condition': record['condition'],
                               'pooled_macro_f1': metrics['macro_f1'], 'accuracy': metrics['accuracy'],
                               'participant_mean_macro_f1': estimate['participant_mean_macro_f1'],
                               'participant_mean_ci_lower': estimate['confidence_interval']['lower'],
                               'participant_mean_ci_upper': estimate['confidence_interval']['upper'],
                               'brier_score': metrics['brier_score'], 'ece': metrics['expected_calibration_error'],
                               'epochs_completed': summary['epochs_completed'], 'best_epoch': summary['best_epoch']})
        for stage, values in metrics['per_stage'].items():
            class_rows.append({'budget_id': budget_id, 'condition': record['condition'], 'stage': stage, **values})
    for contrast in analysis['paired_comparisons']:
        first = next(row for row in condition_rows if row['budget_id'] == budget_id and row['condition'] == contrast['first'])
        second = next(row for row in condition_rows if row['budget_id'] == budget_id and row['condition'] == contrast['second'])
        paired_rows.append({'budget_id': budget_id, 'first': contrast['first'], 'second': contrast['second'],
                            'pooled_macro_f1_difference': first['pooled_macro_f1'] - second['pooled_macro_f1'],
                            'participant_mean_macro_f1_difference': contrast['participant_mean_macro_f1_difference'],
                            'participant_mean_ci_lower': contrast['confidence_interval']['lower'],
                            'participant_mean_ci_upper': contrast['confidence_interval']['upper']})
pd.DataFrame(condition_rows).to_csv(REPORT / 'validation_results.csv', index=False)
pd.DataFrame(paired_rows).to_csv(REPORT / 'validation_contrasts.csv', index=False)
pd.DataFrame(class_rows).to_csv(REPORT / 'per_stage_results.csv', index=False)

baseline = json.loads((ROOT / 'reports/implementation_2026-10-27/evidence.json').read_text())['preserved_artifact_hashes']
assert all(sha256_file(ROOT / name) == digest for name, digest in baseline.items())
assert 'tensorflow' not in sys.modules
classifiers = [row for row in jobs if row['kind'] == 'supervised']
hours = (sum(row['elapsed_seconds'] for row in classifiers) + summaries['ssl_seed42']['elapsed_seconds_this_run']) / 3600
result = {'observed_at_utc': datetime.now(timezone.utc).isoformat(), 'verified': True,
          'study_fingerprint': study['fingerprint'], 'limited_run': False, 'jobs': jobs,
          'completed_jobs': len(jobs), 'completed_classifiers': len(classifiers),
          'completed_classifier_epochs': sum(row['epochs_completed'] for row in classifiers),
          'classifier_plus_ssl_recorded_hours': hours, 'validation_participants': 15,
          'validation_examples_per_model': 65133, 'common_validation_target_identity': identity,
          'common_normalizer_within_each_budget': True, 'common_SSL_encoder_for_all_six_finetunes': True,
          'source_code_packages_and_scope_match': True, 'historical_baseline_hashes_match': True,
          'tensorflow_imported': False, 'optimizer_updates_performed': False,
          'new_source_test_or_ISRUC_scoring_performed': False,
          'analysis_limitations': ['Source-validation results used for checkpoint selection; exploratory intervals without selection correction.',
                                   'One optimization seed and one label draw; intervals conditional on seed 42.',
                                   'Participant-mean macro-F1 intervals differ from pooled macro-F1 checkpoint-selection scores.',
                                   'No multiple-comparison correction; no held-out performance claims.']}
with (REPORT / 'verification.json').open('x', encoding='utf-8') as stream:
    json.dump(result, stream, indent=2, allow_nan=False)
    stream.write('\n')
print(json.dumps({'completed_jobs': len(jobs), 'completed_classifiers': len(classifiers),
                  'classifier_epochs': result['completed_classifier_epochs'], 'recorded_hours': hours,
                  'contrasts': paired_rows, 'tensorflow_imported': False}, indent=2), flush=True)

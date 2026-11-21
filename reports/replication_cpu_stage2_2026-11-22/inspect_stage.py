"""Verify all three SSL fits and the next supervised block without TensorFlow."""
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.epoch_store import sha256_file
from src.data.label_budgets import fingerprint
from src.experiments.run import load_study, verify_job_result, verify_runtime

DIRECTORY = ROOT / 'runs/bilstm_replication_cpu_2026-11-20'
OUTPUT = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verify_files(directory, files):
    for name, digest in files.items():
        path = (directory / name).resolve()
        assert path.is_relative_to(directory.resolve()) and sha256_file(path) == digest


def main():
    study, plan = load_study(DIRECTORY)
    verify_runtime(study, plan, Path(study['data_root']), Path(study['split_file']))
    freeze = read_json(DIRECTORY / 'protocol_freeze.json')
    assert freeze['fingerprint'] == fingerprint({k: v for k, v in freeze.items() if k != 'fingerprint'})
    assert freeze['study_fingerprint'] == study['fingerprint']
    for key, path in [('study_manifest_sha256', DIRECTORY / 'study.json'),
                      ('budget_manifest_sha256', DIRECTORY / 'label_budgets.json'),
                      ('protocol_sha256', DIRECTORY / 'protocol.md'),
                      ('recipe_sha256', ROOT / 'configs/sleepedf/experiments/experiment_bilstm_replication_cpu.json'),
                      ('CPU_launcher_sha256', ROOT / 'scripts/research_cpu.ps1')]:
        assert sha256_file(path) == freeze[key]
    assert freeze['packages_all_distributions'] == {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}
    assert not (DIRECTORY / '.runner.lock').exists()
    ledger = read_json(DIRECTORY / 'ledger.json')
    expected = [job['job_id'] for job in study['jobs'][:6]]
    assert set(ledger['jobs']) == set(expected)
    execution = read_json(DIRECTORY / 'execution_summary.json')
    assert execution == {'jobs_planned': 48, 'jobs_completed': 6, 'jobs_executed_this_call': 2, 'limited_run': False}
    artifacts, seals = {}, {}
    for job_id in expected:
        entry = ledger['jobs'][job_id]
        assert entry['status'] == 'completed'
        attempt = DIRECTORY / entry['attempt']
        verify_job_result(attempt, study, job_id)
        artifacts[job_id] = attempt / 'artifacts'
        seals[job_id] = sha256_file(attempt / 'job_result.json')
    waveform_path = artifacts['waveform_statistics'] / 'waveform_normalizer.json'
    waveform = read_json(waveform_path)
    source_manifest = read_json(artifacts['waveform_statistics'] / 'summary.json')['source_data']
    train_records = {r['recording_id'] for r in plan['source_records'] if r['split'] == 'train'}
    assert len(train_records) == 138 and len(plan['training_participants']) == 70
    for budget in plan['budgets']:
        directory = artifacts['normalizer_' + budget['budget_id']]
        summary = read_json(directory / 'summary.json')
        assert summary['label_budget'] == budget
        assert summary['spectral_fit_epochs'] == budget['annotation_cost']['supervised_epochs']
        assert summary['waveform_fit_epochs'] == 323423
        assert read_json(directory / 'normalizer.json')['pretraining_waveform'] == waveform
    rows = []
    common_model = None
    for seed in (17, 42, 123):
        directory = artifacts[f'ssl_seed{seed}']
        bundle = read_json(directory / 'pretraining_bundle.json')
        assert bundle['completed'] and bundle['kind'] == 'source_raw_contrastive'
        verify_files(directory, bundle['files'])
        resume = read_json(directory / 'resume_state.json')
        assert resume['completed'] and resume['completed_epochs'] == 10
        verify_files(directory, resume['files'])
        resolved = read_json(directory / 'resolved_config.json')
        assert resolved['pretraining'] == {**study['config']['pretraining'], 'seed': seed}
        assert resolved['augmentation'] == study['config']['augmentation']
        assert resolved['max_batches'] is None and resolved['limited_data_run'] is False
        if common_model is not None:
            assert resolved['model'] == common_model
        common_model = resolved['model']
        manifest = read_json(directory / 'data_manifest.json')
        assert manifest == source_manifest
        assert {r['recording_id'] for r in manifest['recordings']} == train_records
        assert {r['participant_id'] for r in manifest['recordings']} == set(plan['training_participants'])
        assert all(r['split'] == 'train' for r in manifest['recordings'])
        assert sha256_file(directory / 'waveform_normalizer.json') == sha256_file(waveform_path)
        summary = read_json(directory / 'summary.json')
        assert summary['epochs_completed'] == 10 and summary['resumed_after_epoch'] == 0
        assert summary['source_quality_epochs_available'] == summary['normalization_fit_epochs'] == 323423
        assert summary['batches_per_epoch'] == 10106 and summary['optimizer_iterations'] == 101060
        assert summary['encoder_reload_max_difference'] == 0 and not summary['limited_data_run']
        history = pd.read_csv(directory / 'training_history.csv')
        assert history.epoch.tolist() == list(range(1, 11)) and np.isfinite(history.to_numpy()).all()
        for key, value in summary['last_epoch'].items():
            np.testing.assert_allclose(history.iloc[-1][key], value, atol=1e-12, rtol=1e-12)
        log = directory.parent.with_suffix('.log')
        assert 'SSL epoch 10/10:' in log.read_text(encoding='utf-8')
        rows.append({'seed': seed, 'epochs_completed': 10, 'optimizer_iterations_reported': 101060,
                     'elapsed_minutes_reported': summary['elapsed_seconds_this_run'] / 60,
                     'encoder_reload_max_difference_reported': summary['encoder_reload_max_difference'],
                     'encoder_sha256': sha256_file(directory / 'encoder.keras'),
                     'native_worker_log_sha256': sha256_file(log)})
    next_jobs = study['jobs'][6:13]
    assert [job['job_id'] for job in next_jobs] == [f'M{i}_draw42_n18_seed17' for i in range(7)]
    for job in next_jobs:
        assert job['kind'] == 'supervised' and job['budget_id'] == 'draw42_n18' and job['seed'] == 17
        assert all(dep in artifacts for dep in job['dependencies'])
    old_review = read_json(ROOT / 'reports/bilstm_completion_2026-11-12/verification.json')
    historical_hours = sum(row['elapsed_seconds'] for row in old_review['jobs'] if row.get('budget_id') == 'draw42_n18') / 3600
    evidence = {'review_date': '2026-11-22', 'stage_two_verified': True,
                'study_fingerprint': study['fingerprint'], 'jobs_planned': 48, 'jobs_completed': 6,
                'completed_job_seal_sha256': seals, 'source_packages_budget_and_protocol_match_freeze': True,
                'SSL_runs': rows, 'SSL_epochs_total': 30, 'SSL_optimizer_iterations_total_reported': 303180,
                'SSL_source_training_participants': 70, 'SSL_source_training_recordings': 138,
                'SSL_quality_epochs': 323423, 'shared_normalization_and_exact_source_manifest_verified': True,
                'supervised_jobs_started': 0, 'next_stage_job_ids': [job['job_id'] for job in next_jobs],
                'next_stage_max_new_jobs': 7, 'next_stage_expected_total_completed': 13,
                'historical_seven_model_n18_block_recorded_hours': historical_hours,
                'runtime_context_limit': 'Historical seed 42 under the older training procedure; not a forecast for new seed 17.',
                'optimizer_checkpoint_tensors_independently_decoded': False,
                'optimizer_updates_by_reviewer': False, 'model_inference_by_reviewer': False,
                'tensorflow_imported_by_reviewer': 'tensorflow' in sys.modules,
                'new_source_test_or_ISRUC_predictive_scoring': False,
                'downstream_performance_established_by_SSL_diagnostics': False}
    assert not evidence['tensorflow_imported_by_reviewer']
    (OUTPUT / 'verification.json').write_text(json.dumps(evidence, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: evidence[key] for key in ('stage_two_verified', 'jobs_completed', 'SSL_epochs_total',
                     'SSL_optimizer_iterations_total_reported', 'next_stage_max_new_jobs', 'next_stage_expected_total_completed')}))


if __name__ == '__main__':
    main()

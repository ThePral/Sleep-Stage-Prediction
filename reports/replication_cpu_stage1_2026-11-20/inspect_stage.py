"""Verify the completed preparation/SSL-17 stage without TensorFlow or training."""
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

OUTPUT = Path(__file__).resolve().parent
DIRECTORY = ROOT / 'runs/bilstm_replication_cpu_2026-11-20'


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
    assert freeze['fingerprint'] == fingerprint({key: value for key, value in freeze.items() if key != 'fingerprint'})
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
    expected = [job['job_id'] for job in study['jobs'][:4]]
    assert expected == freeze['first_stage_job_ids'] and set(ledger['jobs']) == set(expected)
    execution = read_json(DIRECTORY / 'execution_summary.json')
    assert execution == {'jobs_planned': 48, 'jobs_completed': 4, 'jobs_executed_this_call': 4, 'limited_run': False}
    artifacts, job_seals = {}, {}
    for job_id in expected:
        entry = ledger['jobs'][job_id]
        assert entry['status'] == 'completed'
        attempt = DIRECTORY / entry['attempt']
        verify_job_result(attempt, study, job_id)
        artifacts[job_id] = attempt / 'artifacts'
        job_seals[job_id] = sha256_file(attempt / 'job_result.json')
    waveform = read_json(artifacts['waveform_statistics'] / 'waveform_normalizer.json')
    waveform_summary = read_json(artifacts['waveform_statistics'] / 'summary.json')
    assert waveform_summary['participants'] == 70 and waveform_summary['fit_epochs'] == 323423
    for budget in plan['budgets']:
        directory = artifacts['normalizer_' + budget['budget_id']]
        summary = read_json(directory / 'summary.json')
        state = read_json(directory / 'normalizer.json')
        assert summary['label_budget'] == budget
        assert summary['spectral_fit_epochs'] == budget['annotation_cost']['supervised_epochs']
        assert summary['waveform_fit_epochs'] == 323423 and state['pretraining_waveform'] == waveform
    directory = artifacts['ssl_seed17']
    bundle = read_json(directory / 'pretraining_bundle.json')
    assert bundle['completed'] and bundle['kind'] == 'source_raw_contrastive'
    verify_files(directory, bundle['files'])
    resume = read_json(directory / 'resume_state.json')
    assert resume['completed'] and resume['completed_epochs'] == 10
    verify_files(directory, resume['files'])
    manifest = read_json(directory / 'data_manifest.json')
    assert manifest == waveform_summary['source_data']
    train_records = {record['recording_id'] for record in plan['source_records'] if record['split'] == 'train'}
    assert {record['recording_id'] for record in manifest['recordings']} == train_records
    assert {record['participant_id'] for record in manifest['recordings']} == set(plan['training_participants'])
    assert all(record['split'] == 'train' for record in manifest['recordings'])
    summary = read_json(directory / 'summary.json')
    assert summary['epochs_completed'] == 10 and summary['resumed_after_epoch'] == 0
    assert summary['source_quality_epochs_available'] == summary['normalization_fit_epochs'] == 323423
    assert summary['batches_per_epoch'] == 10106 and summary['optimizer_iterations'] == 101060
    assert summary['encoder_reload_max_difference'] == 0 and not summary['limited_data_run']
    assert sha256_file(directory / 'waveform_normalizer.json') == sha256_file(artifacts['waveform_statistics'] / 'waveform_normalizer.json')
    history = pd.read_csv(directory / 'training_history.csv')
    assert history.epoch.tolist() == list(range(1, 11))
    assert np.isfinite(history.to_numpy()).all()
    for name, value in summary['last_epoch'].items():
        np.testing.assert_allclose(history.iloc[-1][name], value, atol=1e-12, rtol=1e-12)
    next_jobs = [job['job_id'] for job in study['jobs'][4:6]]
    assert next_jobs == ['ssl_seed42', 'ssl_seed123']
    assert all(job['kind'] == 'pretrain' for job in study['jobs'][4:6])
    log = (directory.parent).with_suffix('.log')
    # The sibling log belongs to attempt_0001, not its artifacts child.
    assert log.exists() and 'SSL epoch 10/10:' in log.read_text(encoding='utf-8')
    evidence = {'review_date': '2026-11-20', 'stage_one_verified': True,
                'study_fingerprint': study['fingerprint'], 'jobs_completed': 4, 'jobs_planned': 48,
                'completed_job_seal_sha256': job_seals, 'source_packages_budget_and_protocol_match_freeze': True,
                'SSL_seed': 17, 'SSL_epochs_completed': 10, 'SSL_optimizer_iterations_reported': 101060,
                'SSL_elapsed_minutes_reported': summary['elapsed_seconds_this_run'] / 60,
                'SSL_encoder_reload_max_difference_reported': summary['encoder_reload_max_difference'],
                'SSL_pretraining_bundle_and_final_checkpoint_hashes_verified': True,
                'SSL_source_training_participants': len(plan['training_participants']),
                'SSL_source_training_recordings': len(train_records), 'SSL_quality_epochs': 323423,
                'normalizer_budget_participants': [len(budget['participants']) for budget in plan['budgets']],
                'supervised_jobs_started': 0, 'next_stage_job_ids': next_jobs,
                'next_stage_max_new_jobs': 2, 'next_stage_expected_total_completed': 6,
                'native_SSL_progress_log_present': True, 'SSL_progress_log_sha256': sha256_file(log),
                'optimizer_checkpoint_tensors_independently_decoded': False,
                'optimizer_updates_by_reviewer': False, 'model_inference_by_reviewer': False,
                'tensorflow_imported_by_reviewer': 'tensorflow' in sys.modules,
                'new_source_test_or_ISRUC_predictive_scoring': False,
                'representation_quality_or_sleep_staging_performance_established_by_SSL_diagnostics': False}
    assert not evidence['tensorflow_imported_by_reviewer']
    (OUTPUT / 'verification.json').write_text(json.dumps(evidence, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: evidence[key] for key in ('stage_one_verified', 'jobs_completed', 'SSL_epochs_completed',
                     'SSL_optimizer_iterations_reported', 'SSL_elapsed_minutes_reported', 'next_stage_job_ids')}))


if __name__ == '__main__':
    main()

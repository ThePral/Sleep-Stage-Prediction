"""Read-only outage inspection: no model execution, lock mutation or training."""
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys

import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.validation_parity import require_target_identity
from src.data.epoch_store import sha256_file
from src.data.label_budgets import fingerprint
from src.evaluation.artifacts import inspect_bundle, verify_validation_frame
from src.experiments.run import load_study, supervised_resume_source, verify_job_result, verify_runtime
from src.training.recovery_state import load_recovery

DIRECTORY = ROOT / 'runs/bilstm_replication_cpu_2026-11-20'
OUTPUT = Path(__file__).resolve().parent


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


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
    packages = {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}
    assert packages == freeze['packages_all_distributions']
    ledger = read_json(DIRECTORY / 'ledger.json')
    assert set(ledger['jobs']) == {job['job_id'] for job in study['jobs'][:22]}
    seals = {}
    for job in study['jobs'][:21]:
        entry = ledger['jobs'][job['job_id']]
        assert entry['status'] == 'completed'
        attempt = DIRECTORY / entry['attempt']
        verify_job_result(attempt, study, job['job_id'])
        seals[job['job_id']] = sha256_file(attempt / 'job_result.json')
    completed_id = 'M0_draw42_n18_seed123'
    completed = DIRECTORY / ledger['jobs'][completed_id]['attempt'] / 'artifacts'
    summary = read_json(completed / 'summary.json')
    inspect = inspect_bundle(completed)
    predictions = pd.read_csv(completed / 'validation_predictions.csv')
    verify_validation_frame(predictions, inspect)
    previous = DIRECTORY / ledger['jobs']['M0_draw42_n18_seed17']['attempt'] / 'artifacts/validation_predictions.csv'
    require_target_identity(pd.read_csv(previous), predictions)
    completed_generation, completed_state, _ = load_recovery(completed)
    assert completed_state['completed_epochs'] == summary['epochs_completed'] == 11
    assert summary['best_epoch'] == 6 and len(predictions) == 65133
    assert sha256_file(completed_generation / 'best.keras') == sha256_file(completed / 'model.keras')
    job_id = 'M1_draw42_n18_seed123'
    entry = ledger['jobs'][job_id]
    assert entry['status'] == 'running'
    artifacts = DIRECTORY / entry['attempt'] / 'artifacts'
    generation, state, arrays = load_recovery(artifacts)
    assert state['completed_epochs'] == 6 and len(state['history']) == 6
    assert state['contract']['code_sha256'] == study['code_sha256']
    assert state['contract']['training'] == {**study['config']['training'], 'seed': 123}
    assert state['contract']['packages'] == packages
    assert state['contract']['platform'] == platform.platform()
    assert state['contract']['python'] == platform.python_version()
    assert state['contract']['processor'] == platform.processor()
    assert state['contract']['execution_environment'] == {key: os.environ.get(key) for key in state['contract']['execution_environment']}
    assert not state['stop_training'] and state['early']['best_epoch'] + 1 == 6
    index = [i for i, v in enumerate(state['optimizer_layout']) if v['path'].endswith('/iteration')]
    assert len(index) == 1 and int(arrays[f'optimizer_{index[0]}']) == 113532
    assert sha256_file(generation / 'best.keras') == sha256_file(artifacts / 'model.keras')
    epochs = []
    for candidate in sorted((artifacts / 'recovery').glob('epoch_*')):
        _, saved, _ = load_recovery(candidate, state['contract'])
        epochs.append(saved['completed_epochs'])
    assert epochs == list(range(7))
    assert supervised_resume_source(DIRECTORY, job_id).resolve() == artifacts.resolve()
    lock = DIRECTORY / '.runner.lock'
    pid = int(lock.read_text())
    assert not psutil.pid_exists(pid)
    visible = []
    for process in psutil.process_iter(['pid', 'cmdline']):
        args = process.info['cmdline'] or []
        if '-m' in args and args[args.index('-m') + 1] in ('src.experiments.run', 'src.experiments.worker'):
            visible.append(process.pid)
    assert not visible
    next_ids = [job['job_id'] for job in study['jobs'][21:27]]
    assert next_ids == [f'M{i}_draw42_n18_seed123' for i in range(1, 7)]
    evidence = {'review_date': '2026-11-28', 'recovery_ready': True, 'study_fingerprint': study['fingerprint'],
                'completed_jobs_verified': 21, 'completed_job_seal_sha256': seals,
                'new_completed_job': {'job_id': completed_id, 'attempt': ledger['jobs'][completed_id]['attempt'],
                                      'epochs_completed': 11, 'best_epoch': 6,
                                      'resumed_after_epoch_reported': summary['resumed_after_epoch']},
                'interrupted_job': job_id, 'interrupted_attempt': entry['attempt'],
                'latest_sealed_epoch': 6, 'next_epoch_to_replay': 7, 'optimizer_steps_saved': 113532,
                'best_epoch_saved': 6, 'recovery_generations_verified': epochs,
                'recovery_generation': generation.name, 'generation_seal_sha256': sha256_file(generation / 'seal.json'),
                'source_packages_data_protocol_runtime_environment_match': True,
                'dead_lock_pid': pid, 'lock_sha256': sha256_file(lock), 'lock_archived_by_inspector': False,
                'visible_experiment_processes': visible, 'remaining_stage_job_ids': next_ids,
                'max_new_jobs_for_restart': 6, 'expected_completed_after_stage': 27,
                'last_execution_summary_is_older_than_ledger': read_json(DIRECTORY / 'execution_summary.json'),
                'optimizer_updates_by_inspector': False, 'inference_by_inspector': False,
                'tensorflow_imported_by_inspector': 'tensorflow' in sys.modules}
    assert not evidence['tensorflow_imported_by_inspector']
    (OUTPUT / 'verification.json').write_text(json.dumps(evidence, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: evidence[key] for key in ('recovery_ready', 'completed_jobs_verified', 'interrupted_job',
                     'latest_sealed_epoch', 'optimizer_steps_saved', 'max_new_jobs_for_restart', 'expected_completed_after_stage')}))


if __name__ == '__main__':
    main()

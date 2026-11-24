"""Inspect saved supervised results and recovery generations; no model execution."""
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.validation_parity import require_target_identity
from src.data.epoch_store import sha256_file
from src.data.label_budgets import fingerprint
from src.evaluation.artifacts import inspect_bundle, verify_validation_frame
from src.evaluation.classification import ClassificationAccumulator
from src.experiments.run import load_study, verify_job_result, verify_runtime
from src.preprocessing.sequences import CLASS_LABELS
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
    assert freeze['packages_all_distributions'] == {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}
    ledger = read_json(DIRECTORY / 'ledger.json')
    assert set(ledger['jobs']) == {job['job_id'] for job in study['jobs'][:13]}
    assert not (DIRECTORY / '.runner.lock').exists()
    assert read_json(DIRECTORY / 'execution_summary.json') == {
        'jobs_planned': 48, 'jobs_completed': 13, 'jobs_executed_this_call': 7, 'limited_run': False}
    seals = {}
    for job in study['jobs'][:13]:
        entry = ledger['jobs'][job['job_id']]
        assert entry['status'] == 'completed'
        attempt = DIRECTORY / entry['attempt']
        verify_job_result(attempt, study, job['job_id'])
        seals[job['job_id']] = sha256_file(attempt / 'job_result.json')
    budget = next(b for b in plan['budgets'] if b['budget_id'] == 'draw42_n18')
    normalizer = DIRECTORY / ledger['jobs']['normalizer_draw42_n18']['attempt'] / 'artifacts/normalizer.json'
    ssl = DIRECTORY / ledger['jobs']['ssl_seed17']['attempt'] / 'artifacts'
    rows, cohort, total_generations = [], None, 0
    for job in study['jobs'][6:13]:
        assert job['job_id'] == f"{job['condition']}_draw42_n18_seed17"
        artifacts = DIRECTORY / ledger['jobs'][job['job_id']]['attempt'] / 'artifacts'
        inspected = inspect_bundle(artifacts)
        resolved = inspected['resolved']
        summary = read_json(artifacts / 'summary.json')
        assert resolved['training'] == {**study['config']['training'], 'seed': 17}
        assert resolved['selection']['max_train_batches'] is None and resolved['selection']['max_validation_batches'] is None
        assert not summary['limited_data_run'] and summary['recovery_enabled']
        assert summary['resumed_after_epoch'] == 0 and summary['label_budget'] == budget
        assert sha256_file(artifacts / 'normalizer.json') == sha256_file(normalizer)
        assert summary['pretrained_waveform'] == job['pretrained']
        if job['pretrained']:
            assert Path(resolved['pretrained_from']).resolve() == ssl.resolve()
        else:
            assert resolved['pretrained_from'] is None
        frame = pd.read_csv(artifacts / 'validation_predictions.csv')
        verify_validation_frame(frame, inspected)
        assert len(frame) == 65133 and frame.epoch_id.is_unique
        if cohort is not None:
            require_target_identity(cohort, frame)
        cohort = frame
        accumulator = ClassificationAccumulator()
        accumulator.update(frame.true_stage.map({s: i for i, s in enumerate(CLASS_LABELS)}).to_numpy(),
                           frame[['prob_' + s for s in CLASS_LABELS]].to_numpy())
        metrics = accumulator.result()
        assert metrics['confusion_matrix'] == summary['source_validation']['confusion_matrix']
        for key in ('loss', 'accuracy', 'macro_f1'):
            np.testing.assert_allclose(metrics[key], summary['source_validation'][key], atol=1e-12, rtol=1e-12)
        history = pd.read_csv(artifacts / 'training_history.csv')
        epochs = summary['epochs_completed']
        assert 1 <= epochs <= 30 and history.epoch.tolist() == list(range(epochs))
        assert summary['best_epoch'] == int(history.val_macro_f1.to_numpy().argmax()) + 1
        for key in ('loss', 'accuracy', 'macro_f1'):
            np.testing.assert_allclose(history.iloc[summary['best_epoch'] - 1]['val_' + key], metrics[key], atol=1e-12, rtol=1e-12)
        generation_epochs = []
        for generation in sorted((artifacts / 'recovery').glob('epoch_*')):
            _, state, arrays = load_recovery(generation)
            epoch = state['completed_epochs']
            generation_epochs.append(epoch)
            assert state['contract']['code_sha256'] == study['code_sha256']
            assert state['contract']['training'] == resolved['training']
            assert state['contract']['normalizer_sha256'] == sha256_file(normalizer)
            indices = [i for i, v in enumerate(state['optimizer_layout']) if v['path'].endswith('/iteration')]
            assert len(indices) == 1
            assert int(arrays[f'optimizer_{indices[0]}']) == epoch * summary['train_batches_per_epoch']
            assert (generation / 'best.keras').exists() == (epoch > 0)
        assert sorted(generation_epochs) == list(range(epochs + 1))
        latest, state, arrays = load_recovery(artifacts)
        assert state['completed_epochs'] == epochs
        assert state['early']['best_epoch'] + 1 == summary['best_epoch']
        assert sha256_file(latest / 'best.keras') == sha256_file(artifacts / 'model.keras')
        for row in state['history']:
            for key, value in row.items():
                np.testing.assert_allclose(history.iloc[row['epoch']][key], value, atol=1e-12, rtol=1e-12)
        assert summary['checkpoint_reload_max_probability_difference'] <= 1e-6
        total_generations += len(generation_epochs)
        rows.append({'condition': job['condition'], 'job_id': job['job_id'], 'epochs_completed': epochs,
                     'best_epoch': summary['best_epoch'], 'recovery_generations': len(generation_epochs),
                     'optimizer_iterations': epochs * summary['train_batches_per_epoch'],
                     'pooled_source_validation_macro_f1': metrics['macro_f1'],
                     'recorded_hours': summary['elapsed_seconds'] / 3600,
                     'checkpoint_reload_difference_reported': summary['checkpoint_reload_max_probability_difference']})
    next_ids = [job['job_id'] for job in study['jobs'][13:20]]
    assert next_ids == [f'M{i}_draw42_n18_seed42' for i in range(7)]
    evidence = {'review_date': '2026-11-24', 'stage_three_verified': True, 'jobs_completed': 13,
                'study_fingerprint': study['fingerprint'], 'completed_job_seal_sha256': seals,
                'source_packages_data_and_protocol_match_freeze': True, 'supervised_models_verified': 7,
                'validation_targets_per_model': 65133, 'shared_target_labels_and_order_verified': True,
                'shared_normalizer_and_seed17_SSL_initialization_verified': True,
                'supervised_epochs_completed': sum(row['epochs_completed'] for row in rows),
                'recovery_generations_verified': total_generations,
                'recorded_supervised_hours': sum(row['recorded_hours'] for row in rows), 'models': rows,
                'next_stage_job_ids': next_ids, 'next_stage_max_new_jobs': 7,
                'next_stage_expected_total_completed': 20, 'new_source_test_or_ISRUC_predictive_scoring': False,
                'optimizer_updates_by_reviewer': False, 'inference_by_reviewer': False,
                'tensorflow_imported_by_reviewer': 'tensorflow' in sys.modules,
                'limitations': 'One optimization seed and one label budget; validation selected checkpoints. No multi-seed, held-out, speedup or resume-equivalence claim from this block.'}
    assert not evidence['tensorflow_imported_by_reviewer']
    (OUTPUT / 'verification.json').write_text(json.dumps(evidence, indent=2) + '\n', encoding='utf-8')
    pd.DataFrame(rows).to_csv(OUTPUT / 'source_validation_summary.csv', index=False)
    print(json.dumps({key: evidence[key] for key in ('stage_three_verified', 'supervised_models_verified',
                     'supervised_epochs_completed', 'recovery_generations_verified', 'recorded_supervised_hours',
                     'next_stage_expected_total_completed')}))


if __name__ == '__main__':
    main()

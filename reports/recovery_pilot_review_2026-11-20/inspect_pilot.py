"""Inspect completed pilot seals and recovery arrays without TensorFlow."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.validation_parity import require_target_identity
from src.data.epoch_store import sha256_file
from src.experiments.run import load_study, verify_job_result, verify_runtime
from src.training.recovery_state import load_recovery

OUTPUT = Path(__file__).resolve().parent
DIRECTORY = ROOT / 'runs/bilstm_recovery_eager_pilot_2026-11-20'


def main():
    study, plan = load_study(DIRECTORY)
    verify_runtime(study, plan, Path(study['data_root']), Path(study['split_file']))
    ledger = json.loads((DIRECTORY / 'ledger.json').read_text(encoding='utf-8'))
    execution = json.loads((DIRECTORY / 'execution_summary.json').read_text(encoding='utf-8'))
    assert execution == {'jobs_planned': 18, 'jobs_completed': 18, 'jobs_executed_this_call': 18, 'limited_run': True}
    assert len(ledger['jobs']) == len(study['jobs']) == 18
    assert not (DIRECTORY / '.runner.lock').exists()
    models, cohort, m6_handoff = [], None, None
    for job in study['jobs']:
        entry = ledger['jobs'][job['job_id']]
        assert entry['status'] == 'completed'
        attempt = DIRECTORY / entry['attempt']
        verify_job_result(attempt, study, job['job_id'])
        artifacts = attempt / 'artifacts'
        summary = json.loads((artifacts / 'summary.json').read_text(encoding='utf-8'))
        if job['kind'] == 'pretrain':
            assert summary['epochs_completed'] == 1 and summary['optimizer_iterations'] == 2
            assert summary['encoder_reload_max_difference'] <= 1e-6
            assert summary['limited_data_run'] is True
        if job['kind'] != 'supervised':
            continue
        assert summary['epochs_completed'] == 2 and summary['recovery_enabled'] is True
        assert summary['resumed_after_epoch'] == 0 and summary['limited_data_run'] is True
        assert summary['train_batches_per_epoch'] == summary['validation_batches'] == 2
        assert summary['source_validation']['examples'] == 8
        assert summary['checkpoint_reload_max_probability_difference'] <= 1e-6
        frame = pd.read_csv(artifacts / 'validation_predictions.csv')
        assert len(frame) == 8 and frame.epoch_id.is_unique
        if cohort is not None:
            require_target_identity(cohort, frame)
        cohort = frame
        generations = sorted((artifacts / 'recovery').glob('epoch_*'))
        assert len(generations) == 3
        epochs, epoch_one = [], None
        for generation in generations:
            _, state, arrays = load_recovery(generation)
            epoch = state['completed_epochs']
            epochs.append(epoch)
            assert state['contract']['code_sha256'] == study['code_sha256']
            assert state['contract']['training']['recovery'] is True
            assert state['contract']['training']['compiled_validation'] is False
            assert state['contract']['training']['validation_batch_size'] == 4
            iteration_indices = [i for i, item in enumerate(state['optimizer_layout']) if item['path'].endswith('/iteration')]
            assert len(iteration_indices) == 1
            assert int(arrays[f'optimizer_{iteration_indices[0]}']) == epoch * 2
            assert (generation / 'best.keras').exists() == (epoch > 0)
            if epoch == 1:
                epoch_one = generation
        assert sorted(epochs) == [0, 1, 2]
        latest, state, arrays = load_recovery(artifacts)
        assert state['completed_epochs'] == 2
        assert sha256_file(latest / 'best.keras') == sha256_file(artifacts / 'model.keras')
        assert state['early']['best_epoch'] + 1 == summary['best_epoch']
        history = pd.read_csv(artifacts / 'training_history.csv')
        assert history.epoch.tolist() == [0, 1]
        for row in state['history']:
            for key, value in row.items():
                np.testing.assert_allclose(history.iloc[row['epoch']][key], value, atol=1e-12, rtol=1e-12)
        models.append({'job_id': job['job_id'], 'epochs_completed': 2, 'recovery_epochs': epochs,
                       'optimizer_iterations': 4, 'best_epoch': summary['best_epoch'],
                       'validation_targets': 8, 'state_array_count': len(arrays),
                       'elapsed_seconds_this_call': summary['elapsed_seconds_this_call'],
                       'reload_max_probability_difference': summary['checkpoint_reload_max_probability_difference']})
        if job['job_id'] == 'M6_draw42_n2_seed42':
            resolved = json.loads((artifacts / 'resolved_config.json').read_text(encoding='utf-8'))
            config_path = OUTPUT / 'resume_m6_config.json'
            config_path.write_text(json.dumps({key: resolved[key] for key in ('model', 'training')}, indent=2) + '\n', encoding='utf-8')
            m6_handoff = {'baseline_artifacts': str(artifacts), 'epoch_one_generation': str(epoch_one),
                          'config': str(config_path), 'data_root': study['data_root'], 'split_file': study['split_file'],
                          'normalizer': resolved['normalizer_reused_from'], 'pretrained': resolved['pretrained_from'],
                          'budget_plan': str(DIRECTORY / 'label_budgets.json'), 'budget_id': job['budget_id'],
                          'resume_output': str(ROOT / 'runs/bilstm_real_resume_m6_2026-11-20'),
                          'expected_resumed_after_epoch': 1, 'expected_completed_epochs': 2,
                          'optimizer_updates_in_replay': 2, 'executed': False}
    assert len(models) == 14 and m6_handoff is not None
    (OUTPUT / 'resume_handoff.json').write_text(json.dumps(m6_handoff, indent=2) + '\n', encoding='utf-8')
    evidence = {'review_date': '2026-11-20', 'pilot_completed_and_artifacts_verified': True,
                'study_fingerprint': study['fingerprint'], 'execution_summary': execution,
                'source_packages_and_data_scope_verified': True, 'completed_job_seals_verified': 18,
                'supervised_models_verified': 14, 'recovery_generations_verified': 42,
                'supervised_optimizer_steps_observed': 56, 'ssl_optimizer_steps_observed': 2,
                'validation_targets_per_model': 8, 'models': models,
                'epoch_resume_equivalence_on_real_data_tested': False,
                'compiled_validation_accepted': False, 'training_speedup_established': False,
                'optimizer_updates_by_reviewer': False, 'model_inference_by_reviewer': False,
                'tensorflow_imported_by_reviewer': 'tensorflow' in sys.modules,
                'study_manifest_sha256': sha256_file(DIRECTORY / 'study.json'),
                'execution_summary_sha256': sha256_file(DIRECTORY / 'execution_summary.json')}
    assert not evidence['tensorflow_imported_by_reviewer']
    (OUTPUT / 'verification.json').write_text(json.dumps(evidence, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: evidence[key] for key in ('pilot_completed_and_artifacts_verified',
                     'completed_job_seals_verified', 'supervised_models_verified', 'recovery_generations_verified')}))


if __name__ == '__main__':
    main()

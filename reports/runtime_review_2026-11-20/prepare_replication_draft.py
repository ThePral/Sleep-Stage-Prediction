"""Prepare a CPU-compatible draft recipe and scope counts; do not freeze or run."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.epoch_store import sha256_file
from src.experiments.run import load_study, make_jobs


def main():
    study, plan = load_study(ROOT / 'runs/bilstm_terminal_2026-11-02')
    config = deepcopy(study['config'])
    config['training'].update({'recovery': True, 'compiled_validation': False, 'validation_batch_size': 4})
    config['training_seeds'] = [17, 42, 123]
    assert config['training']['batch_size'] == 4 and config['training']['epochs'] == 30
    assert config['fractions'] == [0.25, 1.0] and config['draw_seeds'] == [42]
    assert config['train_participants'] is None and config['validation_participants'] is None
    assert config['limits'] == {'max_train_batches': None, 'max_validation_batches': None, 'max_pretraining_batches': None}
    jobs = make_jobs(config, plan['budgets'])
    counts = Counter(job['kind'] for job in jobs)
    assert counts == {'waveform_statistics': 1, 'normalizer': 2, 'pretrain': 3, 'supervised': 42}
    path = ROOT / 'configs/sleepedf/experiments/experiment_bilstm_replication_draft.json'
    path.write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    completion = json.loads((ROOT / 'reports/bilstm_completion_2026-11-12/verification.json').read_text())
    eager = json.loads((ROOT / 'reports/validation_parity_2026-11-15_m3_eager_full/result.json').read_text())
    pass_seconds = eager['models'][0]['eager_seconds']
    scenarios = [{'epochs_per_model': epochs, 'model_count': 42,
                  'validation_passes_including_export': 42 * (epochs + 1),
                  'illustrative_validation_hours_at_M3_rate': 42 * (epochs + 1) * pass_seconds / 3600}
                 for epochs in (5, 10, 30)]
    evidence = {'prepared_on': '2026-11-20', 'status': 'draft_not_frozen_or_executed',
                'recipe': path.relative_to(ROOT).as_posix(), 'recipe_sha256': sha256_file(path),
                'optimization_seeds': config['training_seeds'], 'label_draw_seed': 42,
                'labeled_participant_budgets': [len(budget['participants']) for budget in plan['budgets']],
                'source_training_participants': len(plan['training_participants']),
                'source_validation_participants': len(plan['validation_participants']),
                'job_kinds': dict(counts), 'jobs_total': len(jobs), 'jobs': jobs,
                'historical_optimization_seed_42_pooled_with_new_version': False,
                'historical_one_seed_recorded_classifier_plus_SSL_hours': completion['classifier_plus_ssl_recorded_hours'],
                'historical_three_times_hours_arithmetic_only': 3 * completion['classifier_plus_ssl_recorded_hours'],
                'illustrative_M3_validation_pass_seconds': pass_seconds, 'capacity_scenarios': scenarios,
                'scenario_limit': 'Illustrations assume every model has the measured M3 eager validation cost. Architectures differ; excludes training, SSL, preparation, downtime and thermal changes. Not a runtime guarantee or lower bound.',
                'study_directory_created': False, 'optimizer_updates_performed': False,
                'pending_decisions': ['CPU versus separately validated GPU/cloud execution', 'Compute allocation and staging before a fresh study freeze']}
    (Path(__file__).resolve().parent / 'replication_draft.json').write_text(json.dumps(evidence, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'draft_recipe_prepared': True, 'jobs_total': len(jobs), 'job_kinds': dict(counts),
                      'study_created': False, 'training_started': False}))


if __name__ == '__main__':
    main()

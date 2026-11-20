"""Check a planned integration study without model construction or training."""
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.epoch_store import sha256_file
from src.experiments.run import load_study, verify_runtime


def main():
    directory = ROOT / 'runs/bilstm_recovery_eager_pilot_2026-11-20'
    study, plan = load_study(directory)
    verify_runtime(study, plan, Path(study['data_root']), Path(study['split_file']))
    config = study['config']
    training = config['training']
    assert training['recovery'] is True and training['compiled_validation'] is False
    assert training['batch_size'] == training['validation_batch_size'] == 4
    assert training['epochs'] == 2 and config['pretraining']['epochs'] == 1
    assert config['model']['temporal_head'] == 'bilstm'
    assert set(config['conditions']) == {f'M{i}' for i in range(7)}
    assert config['limits'] == {'max_train_batches': 2, 'max_validation_batches': 2, 'max_pretraining_batches': 2}
    counts = Counter(job['kind'] for job in study['jobs'])
    assert counts == {'waveform_statistics': 1, 'normalizer': 2, 'pretrain': 1, 'supervised': 14}
    ledger = json.loads((directory / 'ledger.json').read_text(encoding='utf-8'))
    assert not ledger['jobs'] and not (directory / '.runner.lock').exists()
    assert study['limited_run'] is True
    evidence = {'review_date': '2026-11-20', 'study_directory': directory.relative_to(ROOT).as_posix(),
                'study_fingerprint': study['fingerprint'], 'source_packages_and_data_scope_verified': True,
                'jobs_planned': len(study['jobs']), 'job_kinds': dict(counts), 'jobs_started': 0,
                'recovery': True, 'compiled_validation': False, 'validation_batch_size': 4,
                'limited_run': True, 'training_owner': 'user', 'optimizer_updates_by_reviewer': False,
                'training_config': training, 'limits': config['limits'],
                'study_manifest_sha256': sha256_file(directory / 'study.json'),
                'budget_manifest_sha256': sha256_file(directory / 'label_budgets.json'),
                'recipe_sha256': sha256_file(ROOT / 'configs/sleepedf/experiments/experiment_bilstm_recovery_eager_pilot.json'),
                'tensorflow_imported_by_reviewer': 'tensorflow' in sys.modules}
    assert not evidence['tensorflow_imported_by_reviewer']
    destination = Path(__file__).resolve().parent / 'pilot_plan_verification.json'
    destination.write_text(json.dumps(evidence, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'pilot_plan_verified': True, 'jobs_planned': 18, 'jobs_started': 0, 'limited_run': True}))


if __name__ == '__main__':
    main()

"""Publish/inspect the CPU plan freeze; never construct models or run training."""
import argparse
from collections import Counter
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.epoch_store import sha256_file
from src.data.label_budgets import fingerprint
from src.experiments.run import load_study, verify_runtime

OUTPUT = Path(__file__).resolve().parent
DIRECTORY = ROOT / 'runs/bilstm_replication_cpu_2026-11-20'
RECIPE = ROOT / 'configs/sleepedf/experiments/experiment_bilstm_replication_cpu.json'
PROTOCOL = ROOT / 'docs/REPLICATION_PROTOCOL_CPU_2026-11-20.md'


def write_new_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publish', action='store_true', help='Archive protocol and freeze manifest once, before any execution.')
    args = parser.parse_args()
    study, plan = load_study(DIRECTORY)
    verify_runtime(study, plan, Path(study['data_root']), Path(study['split_file']))
    old, old_plan = load_study(ROOT / 'runs/bilstm_terminal_2026-11-02')
    assert plan['fingerprint'] == old_plan['fingerprint']
    assert study['config'] == json.loads(RECIPE.read_text(encoding='utf-8'))
    config = study['config']
    assert config['training_seeds'] == [17, 42, 123] and config['draw_seeds'] == [42]
    assert config['conditions'] == [f'M{i}' for i in range(7)]
    assert config['training']['recovery'] and not config['training']['compiled_validation']
    assert config['training']['batch_size'] == config['training']['validation_batch_size'] == 4
    assert config['training']['epochs'] == 30 and config['pretraining']['epochs'] == 10
    assert config['pretraining']['batch_size'] == 32 and all(value is None for value in config['limits'].values())
    assert not study['limited_run']
    assert len(plan['training_participants']) == 70 and len(plan['validation_participants']) == 15
    assert [len(budget['participants']) for budget in plan['budgets']] == [18, 70]
    counts = Counter(job['kind'] for job in study['jobs'])
    assert counts == {'waveform_statistics': 1, 'normalizer': 2, 'pretrain': 3, 'supervised': 42}
    ids = [job['job_id'] for job in study['jobs']]
    assert len(ids) == len(set(ids)) == 48
    assert ids[:4] == ['waveform_statistics', 'normalizer_draw42_n18', 'normalizer_draw42_n70', 'ssl_seed17']
    positions = {job_id: i for i, job_id in enumerate(ids)}
    for job in study['jobs']:
        assert all(positions[dependency] < positions[job['job_id']] for dependency in job['dependencies'])
    ledger = json.loads((DIRECTORY / 'ledger.json').read_text())
    assert ledger['study_fingerprint'] == study['fingerprint'] and not ledger['jobs']
    assert not (DIRECTORY / '.runner.lock').exists() and not (DIRECTORY / 'jobs').exists()
    runtime_spec = json.loads((ROOT / 'reports/recovery_replay_review_2026-11-20/runtime_spec.json').read_text())
    assert study['code_sha256'] == runtime_spec['current_source_sha256']
    packages = {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}
    assert packages == runtime_spec['current_packages']
    protocol_copy = DIRECTORY / 'protocol.md'
    freeze_path = DIRECTORY / 'protocol_freeze.json'
    if args.publish:
        if protocol_copy.exists() or freeze_path.exists():
            raise ValueError('An existing protocol freeze must be preserved; use inspection without --publish.')
        envelope = {'schema_version': 1, 'frozen_on': '2026-11-20', 'platform_route': 'native_windows_cpu',
                    'training_owner': 'user', 'study_fingerprint': study['fingerprint'],
                    'historical_study_fingerprint': old['fingerprint'], 'historical_seed42_pooled': False,
                    'study_manifest_sha256': sha256_file(DIRECTORY / 'study.json'),
                    'budget_manifest_sha256': sha256_file(DIRECTORY / 'label_budgets.json'),
                    'recipe_sha256': sha256_file(RECIPE), 'protocol_sha256': sha256_file(PROTOCOL),
                    'CPU_launcher_sha256': sha256_file(ROOT / 'scripts/research_cpu.ps1'),
                    'packages_all_distributions': packages, 'python': platform.python_version(),
                    'platform': platform.platform(), 'processor': platform.processor(),
                    'analysis': {'checkpoint_selection': 'maximum_pooled_validation_macro_f1_earliest_best_tie',
                                 'reporting': 'equal_participant_macro_f1_and_seedwise_results',
                                 'primary_budget': 'draw42_n18', 'primary_contrasts': [['M6', 'M3'], ['M6', 'M5']],
                                 'seed_aggregation': 'mean_paired_participant_difference_across_17_42_123',
                                 'bootstrap_unit': 'participant', 'bootstrap_replicates': 2000,
                                 'bootstrap_seed': 42, 'confidence_level': 0.95,
                                 'source_intervals': 'exploratory_percentile_no_selection_or_multiplicity_correction'},
                    'first_stage_job_ids': ids[:4], 'first_stage_max_new_jobs': 4,
                    'supervised_scheduler_epoch_resume': True, 'SSL_scheduler_epoch_resume': False,
                    'source_test_scoring_opened': False, 'ISRUC_predictive_scoring_opened': False}
        envelope['fingerprint'] = fingerprint(envelope)
        with protocol_copy.open('xb') as stream:
            stream.write(PROTOCOL.read_bytes())
        write_new_json(freeze_path, envelope)
    envelope = json.loads(freeze_path.read_text())
    assert envelope['fingerprint'] == fingerprint({key: value for key, value in envelope.items() if key != 'fingerprint'})
    assert envelope['study_fingerprint'] == study['fingerprint']
    assert envelope['study_manifest_sha256'] == sha256_file(DIRECTORY / 'study.json')
    assert envelope['budget_manifest_sha256'] == sha256_file(DIRECTORY / 'label_budgets.json')
    assert envelope['recipe_sha256'] == sha256_file(RECIPE)
    assert envelope['protocol_sha256'] == sha256_file(PROTOCOL) == sha256_file(protocol_copy)
    assert envelope['CPU_launcher_sha256'] == sha256_file(ROOT / 'scripts/research_cpu.ps1')
    assert envelope['packages_all_distributions'] == packages
    evidence = {'review_date': '2026-11-20', 'frozen_CPU_replication_plan_verified': True,
                'study_directory': DIRECTORY.relative_to(ROOT).as_posix(), 'study_fingerprint': study['fingerprint'],
                'protocol_freeze_fingerprint': envelope['fingerprint'], 'jobs_planned': 48, 'job_kinds': dict(counts),
                'jobs_started': 0, 'limited_run': False, 'first_stage_job_ids': ids[:4],
                'first_stage_expected_completed_jobs': 4, 'first_stage_is_full_campaign_completion': False,
                'source_packages_scope_protocol_recipe_and_launcher_verified': True,
                'label_budget_fingerprint_matches_historical_study': True,
                'D_free_GiB_at_inspection': shutil.disk_usage(ROOT).free / 1024**3,
                'optimizer_updates_performed': False, 'inference_performed': False,
                'tensorflow_imported_by_inspector': 'tensorflow' in sys.modules,
                'files_sha256': {name: sha256_file(DIRECTORY / name) for name in
                                 ('study.json', 'label_budgets.json', 'protocol.md', 'protocol_freeze.json')}}
    assert not evidence['tensorflow_imported_by_inspector']
    (OUTPUT / 'verification.json').write_text(json.dumps(evidence, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: evidence[key] for key in ('frozen_CPU_replication_plan_verified', 'jobs_planned',
                     'jobs_started', 'limited_run', 'first_stage_job_ids', 'study_fingerprint')}))


if __name__ == '__main__':
    main()

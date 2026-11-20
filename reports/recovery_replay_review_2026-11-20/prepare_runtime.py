"""Freeze a runtime-only benchmark specification; no model construction."""
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.epoch_store import sha256_file
from src.data.label_budgets import verify_plan_scope
from src.experiments.run import code_hashes, load_study, verify_job_result


def main():
    directory = ROOT / 'runs/bilstm_terminal_2026-11-02'
    study, plan = load_study(directory)
    verify_plan_scope(plan, Path(study['data_root']), Path(study['split_file']))
    ledger = json.loads((directory / 'ledger.json').read_text())
    ids = ['M6_draw42_n70_seed42', 'normalizer_draw42_n70', 'ssl_seed42']
    artifacts = {}
    for job_id in ids:
        entry = ledger['jobs'][job_id]
        assert entry['status'] == 'completed'
        attempt = directory / entry['attempt']
        verify_job_result(attempt, study, job_id)
        artifacts[job_id] = attempt / 'artifacts'
    resolved = json.loads((artifacts[ids[0]] / 'resolved_config.json').read_text())
    summary = json.loads((artifacts[ids[0]] / 'summary.json').read_text())
    assert summary['validation_windows_available'] == 65133
    assert len(plan['training_participants']) == 70 and len(plan['validation_participants']) == 15
    assert resolved['model']['temporal_head'] == 'bilstm' and resolved['model']['variant'] == 'gated'
    training = {**resolved['training'], 'epochs': 1, 'batch_size': 4, 'recovery': True,
                'compiled_validation': False, 'validation_batch_size': 4}
    output = ROOT / 'runs/bilstm_recovery_runtime_2026-11-20'
    assert not output.exists(), 'Preserve an existing benchmark rather than replacing it.'
    spec = {'purpose': 'runtime_only_not_research_performance', 'prepared_on': '2026-11-20',
            'model': resolved['model'], 'training': training,
            'data_root': study['data_root'], 'split_file': study['split_file'],
            'budget_plan': str(directory / 'label_budgets.json'), 'budget_id': 'draw42_n70',
            'normalizer': str(artifacts['normalizer_draw42_n70'] / 'normalizer.json'),
            'pretrained': str(artifacts['ssl_seed42']), 'output_directory': str(output),
            'max_train_batches': 64, 'max_validation_batches': None,
            'expected_validation_targets': 65133,
            'initialization_study': str(directory), 'initialization_study_fingerprint': study['fingerprint'],
            'initialization_jobs': ids, 'current_source_sha256': code_hashes(),
            'current_packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
            'input_files_sha256': {str(path): sha256_file(path) for path in
                                   [directory / 'label_budgets.json', artifacts['normalizer_draw42_n70'] / 'normalizer.json']},
            'optimizer_updates_when_user_runs': 64, 'optimizer_updates_by_preparer': False}
    destination = Path(__file__).resolve().parent / 'runtime_spec.json'
    destination.write_text(json.dumps(spec, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'runtime_benchmark_prepared': True, 'executed': False, 'training_batches': 64,
                      'full_validation_targets_per_pass': 65133, 'validation_passes': 2}))


if __name__ == '__main__':
    main()

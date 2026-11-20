"""Verify retained runtime outputs without model execution or optimizer updates."""
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
from src.data.label_budgets import verify_plan_scope
from src.evaluation.artifacts import inspect_bundle, verify_validation_frame
from src.evaluation.classification import ClassificationAccumulator
from src.experiments.run import code_hashes, load_study, verify_job_result
from src.preprocessing.sequences import CLASS_LABELS
from src.training.recovery_state import load_recovery

OUTPUT = Path(__file__).resolve().parent
DIRECTORY = ROOT / 'runs/bilstm_recovery_runtime_2026-11-20'


def main():
    benchmark = json.loads((DIRECTORY / 'benchmark.json').read_text())
    spec = json.loads((DIRECTORY / 'spec.json').read_text())
    assert spec == json.loads((ROOT / 'reports/recovery_replay_review_2026-11-20/runtime_spec.json').read_text())
    assert json.loads((DIRECTORY / 'status.json').read_text())['status'] == 'completed'
    assert benchmark['completed'] and benchmark['purpose'] == 'runtime_only_not_research_performance'
    assert benchmark['optimizer_iterations'] == 64 and benchmark['validation_passes'] == 2
    assert benchmark['validation_targets_per_pass'] == 65133
    assert benchmark['compiled_validation'] is False and benchmark['recovery'] is True
    assert benchmark['validation_batch_size'] == 4 and not benchmark['training_speedup_established']
    assert code_hashes() == spec['current_source_sha256']
    assert {d.metadata['Name']: d.version for d in importlib.metadata.distributions()} == spec['current_packages']
    assert sha256_file(ROOT / 'reports/recovery_replay_review_2026-11-20/run_runtime_benchmark.py') == benchmark['script_sha256']
    study_dir = Path(spec['initialization_study'])
    study, plan = load_study(study_dir)
    assert study['fingerprint'] == spec['initialization_study_fingerprint']
    verify_plan_scope(plan, Path(spec['data_root']), Path(spec['split_file']))
    ledger = json.loads((study_dir / 'ledger.json').read_text())
    for job_id in spec['initialization_jobs']:
        entry = ledger['jobs'][job_id]
        assert entry['status'] == 'completed'
        verify_job_result(study_dir / entry['attempt'], study, job_id)
    artifacts = DIRECTORY / 'artifacts'
    inspected = inspect_bundle(artifacts)
    resolved = inspected['resolved']
    assert resolved['model'] == spec['model'] and resolved['training'] == spec['training']
    assert resolved['selection']['max_train_batches'] == 64
    assert resolved['selection']['max_validation_batches'] is None
    summary = json.loads((artifacts / 'summary.json').read_text())
    assert summary['epochs_completed'] == 1 and summary['best_epoch'] == 1
    assert summary['train_batches_per_epoch'] == 64 and summary['validation_batches'] == 16284
    assert summary['validation_windows_available'] == summary['source_validation']['examples'] == 65133
    assert summary['limited_data_run'] and summary['recovery_enabled']
    assert summary['resumed_after_epoch'] == 0 and summary['checkpoint_reload_max_probability_difference'] <= 1e-6
    predictions = pd.read_csv(artifacts / 'validation_predictions.csv')
    verify_validation_frame(predictions, inspected)
    reference = study_dir / ledger['jobs']['M6_draw42_n70_seed42']['attempt'] / 'artifacts/validation_predictions.csv'
    require_target_identity(pd.read_csv(reference), predictions)
    accumulator = ClassificationAccumulator()
    accumulator.update(predictions.true_stage.map({stage: i for i, stage in enumerate(CLASS_LABELS)}).to_numpy(),
                       predictions[['prob_' + stage for stage in CLASS_LABELS]].to_numpy())
    metrics = accumulator.result()
    for key in ('loss', 'accuracy', 'macro_f1'):
        np.testing.assert_allclose(metrics[key], summary['source_validation'][key], atol=1e-12, rtol=1e-12)
    generations = sorted((artifacts / 'recovery').glob('epoch_*'))
    assert len(generations) == 2
    states = {}
    for generation in generations:
        _, state, arrays = load_recovery(generation)
        epoch = state['completed_epochs']
        assert state['contract']['code_sha256'] == spec['current_source_sha256']
        index = [i for i, item in enumerate(state['optimizer_layout']) if item['path'].endswith('/iteration')]
        assert len(index) == 1 and int(arrays[f'optimizer_{index[0]}']) == epoch * 64
        states[epoch] = state
    assert set(states) == {0, 1}
    latest, final, _ = load_recovery(artifacts)
    assert final['completed_epochs'] == 1
    assert sha256_file(latest / 'best.keras') == sha256_file(artifacts / 'model.keras')
    peak = benchmark['peak_sampled_host_rss_bytes']
    baseline = benchmark['host_rss_at_sampling_start_bytes']
    assert peak >= baseline > 0
    assert benchmark['worker_elapsed_seconds'] >= benchmark['training_call_seconds'] >= final['elapsed_seconds'] > 0
    transcript = ROOT / 'reports/recovery_replay_review_2026-11-20/runtime_terminal.log'
    transcript_text = transcript.read_text(encoding='utf-8-sig', errors='replace')
    transcript_has_native_output = 'Runtime benchmark active:' in transcript_text
    evidence = {'review_date': '2026-11-20', 'runtime_benchmark_verified': True,
                'source_packages_scope_model_bundle_initialization_and_recovery_seals_verified': True,
                'optimizer_iterations': 64, 'completed_epochs': 1, 'full_validation_targets': 65133,
                'validation_passes': 2, 'training_call_minutes': benchmark['training_call_seconds'] / 60,
                'worker_minutes': benchmark['worker_elapsed_seconds'] / 60,
                'peak_sampled_host_rss_MiB': peak / 1024**2,
                'sampled_host_rss_increase_MiB': (peak - baseline) / 1024**2,
                'elapsed_at_epoch_zero_capture_seconds': states[0]['elapsed_seconds'],
                'elapsed_at_epoch_one_capture_seconds': final['elapsed_seconds'],
                'elapsed_after_epoch_one_capture_seconds': summary['elapsed_seconds'] - final['elapsed_seconds'],
                'phase_timing_limit': 'Capture boundaries are not isolated training or validation timers. Full training-pass and SSL duration are not measured.',
                'native_terminal_output_retained_in_transcript': transcript_has_native_output,
                'durable_evidence': 'benchmark/spec/status JSON, recovery state, summary, history and full prediction export',
                'runtime_output_files_sha256': {str(path.relative_to(DIRECTORY)): sha256_file(path)
                                              for path in sorted(DIRECTORY.rglob('*')) if path.is_file()},
                'training_speedup_established': False, 'GPU_VRAM_measured': False,
                'optimizer_updates_by_reviewer': False, 'inference_by_reviewer': False,
                'tensorflow_imported_by_reviewer': 'tensorflow' in sys.modules}
    assert not evidence['tensorflow_imported_by_reviewer']
    (OUTPUT / 'verification.json').write_text(json.dumps(evidence, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: evidence[key] for key in ('runtime_benchmark_verified', 'training_call_minutes',
                     'worker_minutes', 'peak_sampled_host_rss_MiB', 'native_terminal_output_retained_in_transcript')}))


if __name__ == '__main__':
    main()

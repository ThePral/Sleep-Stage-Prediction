"""Compare retained real-data replay state without model loading or training."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.validation_parity import compare_predictions, require_target_identity
from src.data.epoch_store import sha256_file
from src.experiments.run import load_study, verify_job_result, verify_runtime
from src.training.recovery_state import load_recovery

OUTPUT = Path(__file__).resolve().parent


def main():
    handoff = json.loads((ROOT / 'reports/recovery_pilot_review_2026-11-20/resume_handoff.json').read_text())
    baseline, replay = Path(handoff['baseline_artifacts']), Path(handoff['resume_output'])
    directory = ROOT / 'runs/bilstm_recovery_eager_pilot_2026-11-20'
    study, plan = load_study(directory)
    verify_runtime(study, plan, Path(study['data_root']), Path(study['split_file']))
    verify_job_result(baseline.parent, study, 'M6_draw42_n2_seed42')
    before_generation, before, before_arrays = load_recovery(baseline)
    after_generation, after, after_arrays = load_recovery(replay, before['contract'])
    _, boundary, boundary_arrays = load_recovery(handoff['epoch_one_generation'], before['contract'])
    assert boundary['completed_epochs'] == 1
    assert before['completed_epochs'] == after['completed_epochs'] == 2
    assert set(before) == set(after)
    fields = [name for name in before if name != 'elapsed_seconds']
    assert all(before[name] == after[name] for name in fields)
    assert set(before_arrays) == set(after_arrays)
    comparisons = []
    for name, left in before_arrays.items():
        right = after_arrays[name]
        assert left.shape == right.shape and left.dtype == right.dtype
        if np.issubdtype(left.dtype, np.integer):
            np.testing.assert_array_equal(left, right, err_msg=name)
        else:
            np.testing.assert_allclose(left, right, atol=1e-7, rtol=1e-7, err_msg=name)
        comparisons.append({'name': name, 'shape': list(left.shape), 'dtype': str(left.dtype),
                            'exact': bool(np.array_equal(left, right)),
                            'maximum_difference': float(np.max(np.abs(left.astype(np.float64) - right.astype(np.float64))))})
    summaries = [json.loads((path / 'summary.json').read_text()) for path in (baseline, replay)]
    assert summaries[0]['resumed_after_epoch'] == 0 and summaries[1]['resumed_after_epoch'] == 1
    assert summaries[0]['epochs_completed'] == summaries[1]['epochs_completed'] == 2
    assert summaries[0]['best_epoch'] == summaries[1]['best_epoch'] == 1
    assert all(item['limited_data_run'] and item['recovery_enabled'] for item in summaries)
    assert all(item['checkpoint_reload_max_probability_difference'] == 0 for item in summaries)
    assert summaries[0]['source_validation'] == summaries[1]['source_validation']
    for generation, path in ((before_generation, baseline), (after_generation, replay)):
        assert sha256_file(generation / 'best.keras') == sha256_file(path / 'model.keras')
    assert sha256_file(baseline / 'model.keras') == sha256_file(replay / 'model.keras')
    history_before, history_after = [pd.read_csv(path / 'training_history.csv') for path in (baseline, replay)]
    pd.testing.assert_frame_equal(history_before, history_after, check_exact=True)
    left, right = [pd.read_csv(path / 'validation_predictions.csv') for path in (baseline, replay)]
    require_target_identity(left, right)
    classes = ['W', 'N1', 'N2', 'N3', 'REM']
    columns = ['prob_' + name for name in classes]
    targets = left.true_stage.map({name: i for i, name in enumerate(classes)}).to_numpy()
    probabilities = compare_predictions(left[columns].to_numpy(), right[columns].to_numpy(), targets,
                                        atol=1e-7, rtol=1e-7)
    assert probabilities['passed'] and len(left) == 8
    iteration = [i for i, item in enumerate(after['optimizer_layout']) if item['path'].endswith('/iteration')]
    assert len(iteration) == 1 and int(after_arrays[f'optimizer_{iteration[0]}']) == 4
    assert boundary['optimizer_layout'] == after['optimizer_layout']
    assert int(boundary_arrays[f'optimizer_{iteration[0]}']) == 2
    status = json.loads((replay / 'status.json').read_text())
    assert status['status'] == 'completed'
    evidence = {'review_date': '2026-11-20', 'real_data_m6_capped_resume_equivalence_passed': True,
                'baseline': str(baseline), 'replay': str(replay), 'completed_epochs': 2,
                'resumed_after_epoch': 1, 'optimizer_steps_at_boundary': 2,
                'optimizer_steps_at_finish': 4, 'optimizer_steps_replayed': 2, 'best_epoch': 1,
                'float_atol': 1e-7, 'float_rtol': 1e-7, 'integer_comparison': 'exact',
                'compared_state_fields': fields, 'excluded_state_fields': ['elapsed_seconds'],
                'state_arrays': comparisons, 'arrays_compared': len(comparisons),
                'all_arrays_exactly_equal': all(row['exact'] for row in comparisons),
                'maximum_state_array_difference': max(row['maximum_difference'] for row in comparisons),
                'history_exactly_equal': True, 'best_model_file_sha256_equal': True,
                'validation_predictions': probabilities, 'validation_targets': len(left),
                'source_packages_data_and_baseline_job_seal_verified': True,
                'replay_files_sha256': {str(path.relative_to(replay)): sha256_file(path)
                                       for path in sorted(replay.rglob('*')) if path.is_file()},
                'optimizer_updates_by_reviewer': False, 'inference_by_reviewer': False,
                'tensorflow_imported_by_reviewer': 'tensorflow' in sys.modules,
                'training_speedup_established': False,
                'scope': 'M6 SSL-initialized gated BiLSTM, seed 42, two capped real-data epochs, same native CPU environment; not all configurations or full-schedule recovery.'}
    assert not evidence['tensorflow_imported_by_reviewer']
    (OUTPUT / 'verification.json').write_text(json.dumps(evidence, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({name: evidence[name] for name in ('real_data_m6_capped_resume_equivalence_passed',
                     'arrays_compared', 'maximum_state_array_difference', 'best_epoch')}))


if __name__ == '__main__':
    main()

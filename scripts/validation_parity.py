"""User-run saved-model validation parity. No training or test/ISRUC inference.

Default: a deterministic 64-window check of every completed classifier. 'all'
checks the full validation cohort against stored exports, omitting the expensive
eager replay. Timings describe warmed inference passes, not total epoch speed.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.epoch_store import sha256_file
from src.data.label_budgets import fingerprint, verify_plan_scope
from src.evaluation.classification import ClassificationAccumulator
from src.preprocessing.sequences import CLASS_LABELS, CLASS_TO_INDEX


def selected_indices(total, samples):
    if isinstance(total, bool) or not isinstance(total, int) or total < 1:
        raise ValueError('A nonempty cohort is required.')
    if samples == 'all':
        return np.arange(total, dtype=np.int64)
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError('Samples must be positive or all.')
    return np.linspace(0, total - 1, min(total, samples), dtype=np.int64)


def compare_predictions(reference, candidate, targets, atol=1e-6, rtol=1e-6):
    reference, candidate = np.asarray(reference), np.asarray(candidate)
    if reference.shape != candidate.shape or reference.shape != (len(targets), 5):
        raise ValueError('Prediction shape/target count mismatch.')
    metrics = []
    for values in (reference, candidate):
        accumulator = ClassificationAccumulator()
        accumulator.update(np.asarray(targets), values)
        metrics.append(accumulator.result())
    maximum = float(np.max(np.abs(reference - candidate)))
    mismatches = int(np.sum(reference.argmax(1) != candidate.argmax(1)))
    differences = {key: float(metrics[1][key] - metrics[0][key]) for key in ('loss', 'accuracy', 'macro_f1')}
    probabilities_match = bool(np.allclose(reference, candidate, atol=atol, rtol=rtol))
    return {'passed': probabilities_match and mismatches == 0 and
            all(abs(value) <= atol for value in differences.values()),
            'probabilities_within_tolerance': probabilities_match, 'maximum_probability_difference': maximum,
            'argmax_mismatches': mismatches, 'metric_differences': differences,
            'confusion_matrices_equal': metrics[0]['confusion_matrix'] == metrics[1]['confusion_matrix']}


def require_target_identity(expected, actual):
    columns = ['dataset', 'participant_id', 'recording_id', 'epoch_id', 'epoch_index', 'true_stage']
    if (len(expected) != len(actual) or any(expected[column].tolist() != actual[column].tolist() for column in columns)):
        raise ValueError('Validation targets, labels or ordering changed.')


def canonical_targets(partition):
    rows = []
    for record in partition.recordings:
        for epoch in record.positions[:, -1]:
            rows.append({'dataset': 'sleepedf', 'participant_id': record.participant_id,
                         'recording_id': record.recording_id, 'epoch_index': int(epoch),
                         'epoch_id': f'sleepedf:{record.recording_id}:{epoch:06d}',
                         'true_stage': CLASS_LABELS[record.labels[epoch]]})
    return pd.DataFrame(rows)


def run(args):
    # Imports which can construct models are confined to explicit execution.
    import tensorflow as tf
    from tensorflow import keras
    from src.data.supervised import SourcePartition
    from src.evaluation.artifacts import inspect_bundle, verify_validation_frame
    from src.experiments.run import code_hashes, load_study, verify_job_result
    from src.inference.multiview import MultiviewPredictor
    from src.training.supervised import validation_predictor
    directory, output = args.study_dir.resolve(), args.output_dir.resolve()
    if output == directory or output.is_relative_to(directory):
        raise ValueError('Save parity evidence outside the preserved study.')
    study, plan = load_study(directory)
    root = (args.data_root or Path(study['data_root'])).resolve()
    split = Path(study['split_file'])
    verify_plan_scope(plan, root, split)
    ledger = json.loads((directory / 'ledger.json').read_text())
    jobs = [job for job in study['jobs'] if job['kind'] == 'supervised']
    if not jobs:
        raise ValueError('No supervised model family to check.')
    if any(ledger['jobs'].get(job['job_id'], {}).get('status') != 'completed' for job in jobs):
        raise ValueError('Parity review requires a completed model family.')
    output.mkdir(parents=True, exist_ok=False)
    results = []
    started = time.perf_counter()
    policy = {'samples': args.samples, 'compiled_batch_size': args.validation_batch_size,
              'eager_batch_size': 4, 'atol': 1e-6, 'rtol': 1e-6,
              'cohort': 'source_validation_only', 'current_source_sha256': code_hashes(),
              'script_sha256': sha256_file(Path(__file__)), 'study_fingerprint': study['fingerprint']}
    (output / 'policy.json').write_text(json.dumps(policy, indent=2) + '\n', encoding='utf-8')
    tf.config.experimental.enable_op_determinism()
    records = [record['recording_id'] for record in plan['source_records'] if record['split'] == 'val']
    try:
        with SourcePartition(root, 'val', sequence_length=plan['sequence_length'], recordings=records, split_file=split) as partition:
            target_frame = canonical_targets(partition)
            chosen = selected_indices(len(partition), args.samples)
            expected_targets = target_frame.iloc[chosen].reset_index(drop=True)
            expected_targets.to_csv(output / 'selected_targets.csv', index=False)
            for job in jobs:
                artifacts = directory / ledger['jobs'][job['job_id']]['attempt'] / 'artifacts'
                verify_job_result(artifacts.parent, study, job['job_id'])
                inspected = inspect_bundle(artifacts)
                if inspected['data']['validation'] != partition.manifest():
                    raise ValueError('Bundle validation scope differs from the prepared cohort.')
                reference_frame = pd.read_csv(artifacts / 'validation_predictions.csv')
                verify_validation_frame(reference_frame, inspected)
                require_target_identity(target_frame, reference_frame)
                reference = reference_frame.iloc[chosen][['prob_' + label for label in CLASS_LABELS]].to_numpy()
                targets = expected_targets.true_stage.map(CLASS_TO_INDEX).to_numpy(dtype=np.int64)
                keras.backend.clear_session()
                predictor = MultiviewPredictor(artifacts)
                predictor.normalizer.assert_compatible(partition)
                compiled = validation_predictor(predictor.model, True)
                def inputs_for(indices):
                    raw, spectral, labels, metadata = partition.gather(indices, include_waveform=predictor.config.variant != 'spectral')
                    values = {}
                    if predictor.config.variant != 'spectral':
                        values['waveform'] = predictor.normalizer.transform_waveform(raw)
                    if predictor.config.variant != 'raw':
                        values['spectral'] = predictor.normalizer.transform_spectral(spectral)
                    return values, metadata
                def replay(function, batch_size):
                    values, _ = inputs_for(chosen[:1])
                    function(values).numpy()  # Warm tracing outside the reported pass.
                    probabilities, metadata = [], []
                    tick = time.perf_counter()
                    for start in range(0, len(chosen), batch_size):
                        values, identities = inputs_for(chosen[start:start + batch_size])
                        probabilities.append(function(values).numpy())
                        metadata.extend(identities)
                    elapsed = time.perf_counter() - tick
                    require_target_identity(expected_targets, pd.DataFrame(metadata))
                    return np.concatenate(probabilities), elapsed
                item = {'job_id': job['job_id'], 'examples': len(chosen),
                        'export_sha256': sha256_file(artifacts / 'validation_predictions.csv')}
                if args.samples != 'all':
                    eager, eager_seconds = replay(validation_predictor(predictor.model), 4)
                    item['saved_vs_eager'] = compare_predictions(reference, eager, targets)
                    item['eager_seconds'] = eager_seconds
                new, seconds = replay(compiled, args.validation_batch_size)
                item['saved_vs_compiled'] = compare_predictions(reference, new, targets)
                if args.samples != 'all':
                    item['eager_vs_compiled'] = compare_predictions(eager, new, targets)
                item['compiled_seconds'] = seconds
                item['passed'] = all(value['passed'] for value in item.values() if isinstance(value, dict) and 'passed' in value)
                results.append(item)
                (output / 'progress.json').write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
                print(f"{job['job_id']}: {'PASS' if item['passed'] else 'FAIL'}, {len(chosen)} targets", flush=True)
        if code_hashes() != policy['current_source_sha256']:
            raise ValueError('Implementation changed during parity review.')
        result = {'passed': all(item['passed'] for item in results), 'models': results,
                  'observed_at_utc': datetime.now(timezone.utc).isoformat(), 'policy': policy,
                  'full_cohort_checked': args.samples == 'all', 'elapsed_seconds': time.perf_counter() - started,
                  'timing_limit': 'Warm inference passes include gathering/scaling, exclude initial integrity/model loading/tracing; not total training epoch speedup.',
                  'optimizer_updates_performed': False, 'test_or_ISRUC_scoring': False}
        (output / 'result.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
        return result
    except BaseException as exc:
        (output / 'failure.json').write_text(json.dumps({'error': type(exc).__name__, 'message': str(exc)}) + '\n', encoding='utf-8')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--samples', default='64', help='Positive count or all; all skips eager replay.')
    parser.add_argument('--validation-batch-size', type=int, default=32)
    args = parser.parse_args()
    if args.samples != 'all':
        try:
            args.samples = int(args.samples)
        except ValueError:
            parser.error('--samples must be a positive integer or all.')
        if args.samples < 1:
            parser.error('--samples must be positive.')
    if args.validation_batch_size < 1:
        parser.error('--validation-batch-size must be positive.')
    result = run(args)
    print(json.dumps({'passed': result['passed'], 'models_checked': len(result['models']),
                      'full_cohort_checked': result['full_cohort_checked']}, indent=2))
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()

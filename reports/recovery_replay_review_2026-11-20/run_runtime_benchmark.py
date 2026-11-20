"""USER-RUN: one recovery-enabled epoch, 64 optimizer updates, full validation."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.epoch_store import sha256_file
from src.data.label_budgets import verify_plan_scope
from src.experiments.run import atomic_json, code_hashes, load_study, verify_job_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inspect-only', action='store_true', help='Verify inputs without importing TensorFlow or starting training.')
    args = parser.parse_args()
    started = time.perf_counter()
    spec = json.loads((Path(__file__).resolve().parent / 'runtime_spec.json').read_text())
    output = Path(spec['output_directory'])
    if output.exists():
        raise ValueError('Use a fresh benchmark output; preserve the existing one for review.')
    if code_hashes() != spec['current_source_sha256']:
        raise ValueError('Source changed since runtime specification; prepare and review a new benchmark.')
    if {d.metadata['Name']: d.version for d in importlib.metadata.distributions()} != spec['current_packages']:
        raise ValueError('Packages changed since runtime specification.')
    for name, digest in spec['input_files_sha256'].items():
        if sha256_file(Path(name)) != digest:
            raise ValueError('Benchmark budget or normalizer changed.')
    study_dir = Path(spec['initialization_study'])
    study, plan = load_study(study_dir)
    if study['fingerprint'] != spec['initialization_study_fingerprint']:
        raise ValueError('Initialization study changed.')
    ledger = json.loads((study_dir / 'ledger.json').read_text())
    for job_id in spec['initialization_jobs']:
        entry = ledger['jobs'][job_id]
        if entry['status'] != 'completed':
            raise ValueError('Benchmark initialization job is incomplete.')
        verify_job_result(study_dir / entry['attempt'], study, job_id)
    verify_plan_scope(plan, Path(spec['data_root']), Path(spec['split_file']))
    if args.inspect_only:
        assert 'tensorflow' not in sys.modules
        print(json.dumps({'inputs_verified': True, 'optimizer_updates_performed': False,
                          'tensorflow_imported': False, 'training_batches': spec['max_train_batches'],
                          'validation_targets_per_pass': spec['expected_validation_targets'],
                          'output_directory_created': output.exists()}))
        return
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / 'spec.json', spec)
    atomic_json(output / 'status.json', {'status': 'running'})
    try:
        # Training-capable imports occur only after explicit user execution.
        from src.evaluation.profile import HostMemorySampler
        from src.models.multiview import ModelConfig
        from src.training.supervised import TrainingConfig, train_supervised
        from src.training.recovery_state import load_recovery
        stop = threading.Event()
        with HostMemorySampler() as memory:
            def heartbeat():
                while not stop.wait(30):
                    elapsed = time.perf_counter() - started
                    rss = memory.process.memory_info().rss
                    atomic_json(output / 'progress.json', {'elapsed_seconds': elapsed, 'host_rss_bytes': rss})
                    print(f'Runtime benchmark active: {elapsed / 60:.1f} min elapsed, {rss / 1024**3:.2f} GiB RSS', flush=True)
            thread = threading.Thread(target=heartbeat, daemon=True)
            thread.start()
            try:
                call_started = time.perf_counter()
                summary = train_supervised(
                    Path(spec['data_root']), output / 'artifacts', ModelConfig(**spec['model']),
                    TrainingConfig(**spec['training']), split_file=Path(spec['split_file']),
                    max_train_batches=spec['max_train_batches'], max_validation_batches=None,
                    normalizer_path=Path(spec['normalizer']), pretrained_path=Path(spec['pretrained']),
                    budget_plan_path=Path(spec['budget_plan']), budget_id=spec['budget_id'],
                    progress_interval=16)
                call_seconds = time.perf_counter() - call_started
            finally:
                stop.set()
                thread.join()
        _, state, arrays = load_recovery(output / 'artifacts')
        iterations = [int(arrays[f'optimizer_{i}']) for i, item in enumerate(state['optimizer_layout'])
                      if item['path'].endswith('/iteration')]
        assert iterations == [64] and state['completed_epochs'] == 1
        assert summary['source_validation']['examples'] == spec['expected_validation_targets']
        result = {'purpose': spec['purpose'], 'completed': True,
                  'worker_elapsed_seconds': time.perf_counter() - started,
                  'training_call_seconds': call_seconds, 'peak_sampled_host_rss_bytes': memory.maximum,
                  'host_rss_at_sampling_start_bytes': memory.baseline, 'optimizer_iterations': 64,
                  'validation_targets_per_pass': spec['expected_validation_targets'], 'validation_passes': 2,
                  'compiled_validation': False, 'validation_batch_size': 4, 'recovery': True,
                  'script_sha256': sha256_file(Path(__file__)),
                  'training_speedup_established': False,
                  'timing_scope': 'Training call includes setup, 64 training batches, full epoch validation, best/recovery saves, reload and full best-model prediction export. Training is capped; this does not measure a full training pass or SSL.',
                  'memory_scope': 'Host RSS sampled every 10 ms after TensorFlow import, before model construction, through training and export; excludes initial import and is not whole-system memory.'}
        atomic_json(output / 'benchmark.json', result)
        atomic_json(output / 'status.json', {'status': 'completed'})
        print(json.dumps(result, indent=2), flush=True)
    except BaseException as exc:
        atomic_json(output / 'status.json', {'status': 'failed', 'error': type(exc).__name__, 'message': str(exc)})
        raise


if __name__ == '__main__':
    main()

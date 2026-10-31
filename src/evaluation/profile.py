"""Profile source-only inference or disposable training steps in one process.

Invoke the CLI separately for each batch size/mode so allocator history does
not carry across measurements. Training changes only the in-memory loaded model.
"""

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import platform
import threading
import time

import numpy as np
import psutil
import tensorflow as tf

from src.data.epoch_store import sha256_file
from src.data.prepare_research_data import ROOT
from src.data.supervised import SourcePartition
from src.evaluation.artifacts import inspect_bundle
from src.features.spectral import extract_frequency_features_batch
from src.inference.multiview import MultiviewPredictor
from src.models.multiview import compile_multiview_model
from src.training.artifacts import write_json


@dataclass(frozen=True)
class ProfileConfig:
    batch_size: int = 4
    warmup_steps: int = 2
    measured_steps: int = 10
    mode: str = "inference"
    split: str = "train"
    seed: int = 42

    def __post_init__(self):
        for key in ["batch_size", "warmup_steps", "measured_steps"]:
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{key} must be a positive integer.")
        if self.mode not in {"inference", "training"} or self.split not in {"train", "val"}:
            raise ValueError("Profiling supports inference/training on source train/val only.")
        if self.mode == "training" and self.split != "train":
            raise ValueError("Disposable training-step profiling requires source training participants.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2**31:
            raise ValueError("Profile seed must be an integer in [0, 2**31).")


class HostMemorySampler:
    def __enter__(self):
        self.process = psutil.Process()
        self.baseline = self.process.memory_info().rss
        self.maximum = self.baseline
        self.stop = threading.Event()

        def sample():
            while not self.stop.wait(.01):
                self.maximum = max(self.maximum, self.process.memory_info().rss)

        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.maximum = max(self.maximum, self.process.memory_info().rss)
        self.stop.set()
        self.thread.join()


def timing_summary(durations, batch_size):
    values = np.asarray(durations)
    return {"steps": len(values), "median_seconds": float(np.median(values)),
            "p95_seconds": float(np.quantile(values, .95)), "mean_seconds": float(values.mean()),
            "examples_per_second": float(batch_size / values.mean())}


def profile_bundle(bundle_directory, root, output_directory, config=None, *, recordings=None, split_file=None):
    config = config or ProfileConfig()
    inspected = inspect_bundle(bundle_directory)
    manifest = inspected["data"]["train" if config.split == "train" else "validation"]
    allowed = [r["recording_id"] for r in manifest["recordings"]]
    selected = list(recordings) if recordings is not None else allowed
    if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(allowed):
        raise ValueError("Profile recordings must belong to the model's declared source partition.")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "status.json", {"status": "running"})
    before = {name: sha256_file(Path(bundle_directory) / name) for name in ["model.keras", "normalizer.json", "bundle.json"]}
    try:
        tf.keras.utils.set_random_seed(config.seed)
        tf.config.experimental.enable_op_determinism()
        gpus = tf.config.list_physical_devices("GPU")
        memory_growth_errors = []
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as exc:
                memory_growth_errors.append(str(exc))
        with HostMemorySampler() as memory:
            started = time.perf_counter()
            predictor = MultiviewPredictor(bundle_directory)
            load_model_seconds = time.perf_counter() - started
            gpu_device = "GPU:0" if gpus else None
            gpu_memory_error = None
            if gpu_device:
                try:
                    tf.config.experimental.reset_memory_stats(gpu_device)
                except (ValueError, RuntimeError) as exc:
                    gpu_memory_error = str(exc)
            started = time.perf_counter()
            with SourcePartition(root, config.split, sequence_length=predictor.config.sequence_length,
                                 recordings=selected, split_file=split_file) as partition:
                verification_seconds = time.perf_counter() - started
                predictor.normalizer.assert_compatible(partition)
                if len(partition) < config.batch_size:
                    raise ValueError("The selected source partition has fewer windows than the requested fixed batch.")

                def batch(number):
                    indices = (np.arange(config.batch_size) + number * config.batch_size) % len(partition)
                    raw, spectral, targets, metadata = partition.gather(indices, include_waveform=predictor.config.variant != "spectral")
                    inputs = {}
                    if raw is not None:
                        inputs["waveform"] = predictor.normalizer.transform_waveform(raw)
                    if predictor.config.variant != "raw":
                        inputs["spectral"] = predictor.normalizer.transform_spectral(spectral)
                    return inputs, targets, raw, metadata

                initial, targets, raw, metadata = batch(0)
                if config.mode == "training":
                    compile_multiview_model(predictor.model, inspected["resolved"]["training"]["learning_rate"])

                def step(inputs, labels):
                    if config.mode == "training":
                        return predictor.model.train_on_batch(inputs, labels, return_dict=True)
                    # numpy synchronizes device execution before stopping the timer.
                    return predictor.model(inputs, training=False).numpy()

                for _ in range(config.warmup_steps):
                    step(initial, targets)
                model_times, end_to_end_times, loader_times = [], [], []
                for number in range(config.measured_steps):
                    start = time.perf_counter()
                    step(initial, targets)
                    model_times.append(time.perf_counter() - start)
                    start = time.perf_counter()
                    inputs, labels, _, _ = batch(number)
                    loaded = time.perf_counter()
                    step(inputs, labels)
                    end = time.perf_counter()
                    loader_times.append(loaded - start)
                    end_to_end_times.append(end - start)
                feature_timing = None
                if raw is not None:
                    unique, keys = [], set()
                    for window, item in zip(raw[..., 0], metadata):
                        for offset, epoch in enumerate(window):
                            key = (item["recording_id"], item["epoch_index"] - predictor.config.sequence_length + 1 + offset)
                            if key not in keys:
                                unique.append(epoch)
                                keys.add(key)
                    start = time.perf_counter()
                    extract_frequency_features_batch(np.stack(unique), partition.preprocessing.sampling_frequency,
                                                     nperseg=partition.preprocessing.welch_samples)
                    feature_timing = {"unique_epochs": len(unique), "seconds": time.perf_counter() - start}
                    feature_timing["seconds_per_epoch"] = feature_timing["seconds"] / len(unique)
                gpu_memory = None
                if gpu_device:
                    try:
                        gpu_memory = tf.config.experimental.get_memory_info(gpu_device)
                    except (ValueError, RuntimeError) as exc:
                        gpu_memory_error = str(exc)
                source_manifest = partition.manifest()
                parameters = predictor.model.count_params()
        after = {name: sha256_file(Path(bundle_directory) / name) for name in before}
        if before != after:
            raise ValueError("Profiling changed an original saved artifact.")
        result = {"schema_version": 1, "config": asdict(config), "parameters": parameters,
                  "bundle_sha256": before["bundle.json"], "source_data": source_manifest,
                  "source_recording_subset": selected, "python": platform.python_version(), "tensorflow": tf.__version__,
                  "devices": [d.name for d in tf.config.list_physical_devices()],
                  "model_load_seconds": load_model_seconds, "source_integrity_verification_seconds": verification_seconds,
                  "model_only": timing_summary(model_times, config.batch_size),
                  "streaming_cached_epochs_including_normalization": timing_summary(end_to_end_times, config.batch_size),
                  "loader_and_normalization": timing_summary(loader_times, config.batch_size), "spectral_extraction": feature_timing,
                  "host_memory": {"baseline_rss_bytes": memory.baseline, "maximum_sampled_rss_bytes": memory.maximum, "sampling_interval_seconds": .01},
                  "tensorflow_gpu_allocator_bytes": gpu_memory, "gpu_measurement_error": gpu_memory_error,
                  "memory_growth_configuration_errors": memory_growth_errors,
                  "original_saved_artifacts_unchanged": True,
                  "limitations": ["Host RSS is sampled, not an exact peak. TensorFlow memory excludes driver-reserved GPU memory.",
                                  "Cached-epoch timing excludes original EDF decoding, resampling/filtering and integrity verification.",
                                  "Feature extraction is measured separately on unique prefiltered waveform epochs when available.",
                                  "Training profiles use a fresh in-memory optimizer and discard updates; no checkpoint is saved.",
                                  "Use a fresh CLI process for each mode/batch. Few-step pilots are timing checks, not stable hardware benchmarks."]}
        write_json(output / "profile.json", result)
        write_json(output / "status.json", {"status": "completed"})
        return result
    except Exception as exc:
        write_json(output / "status.json", {"status": "failed", "error": type(exc).__name__, "message": str(exc)})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/processed/research_v1")
    parser.add_argument("--split-file", type=Path)
    parser.add_argument("--recordings", nargs="+")
    parser.add_argument("--mode", choices=["inference", "training"], default="inference")
    parser.add_argument("--split", choices=["train", "val"], default="train")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--measured-steps", type=int, default=10)
    args = parser.parse_args(argv)
    result = profile_bundle(args.bundle, args.data_root, args.output_dir,
                            ProfileConfig(batch_size=args.batch_size, warmup_steps=args.warmup_steps,
                                          measured_steps=args.measured_steps, mode=args.mode, split=args.split),
                            recordings=args.recordings, split_file=args.split_file)
    print(json.dumps({"devices": result["devices"], "streaming": result["streaming_cached_epochs_including_normalization"],
                      "host_memory": result["host_memory"], "gpu_allocator": result["tensorflow_gpu_allocator_bytes"]}, indent=2))


if __name__ == "__main__":
    main()

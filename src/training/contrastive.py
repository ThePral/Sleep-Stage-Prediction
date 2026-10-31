"""Source-only raw EEG pretraining, recoverable at completed epoch boundaries."""

import argparse
import csv
from dataclasses import asdict, dataclass
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.data.contrastive import EpochBatches, SourceEpochPool
from src.data.epoch_store import sha256_file
from src.data.prepare_research_data import ROOT
from src.models.contrastive import (AugmentationConfig, augment_waveform, build_contrastive_model,
                                    nt_xent, representation_diagnostics)
from src.models.multiview import ModelConfig, get_waveform_encoder
from src.preprocessing.waveform_normalization import WaveformNormalizer, source_identity
from src.training.artifacts import write_json


@dataclass(frozen=True)
class PretrainingConfig:
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 0.001
    temperature: float = 0.2
    projection_dim: int = 64
    seed: int = 42
    exclusion_epochs: int = 1
    diagnostic_epochs: int = 64

    def __post_init__(self):
        for name in ["batch_size", "epochs", "projection_dim", "diagnostic_epochs"]:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < (2 if name in {"batch_size", "diagnostic_epochs"} else 1):
                raise ValueError(f"Invalid positive integer {name}.")
        for name in ["learning_rate", "temperature"]:
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2**31:
            raise ValueError("seed must be an integer in [0, 2**31).")
        if isinstance(self.exclusion_epochs, bool) or not isinstance(self.exclusion_epochs, int) or self.exclusion_epochs < 0:
            raise ValueError("exclusion_epochs must be a nonnegative integer.")


def atomic_json(path, value):
    path = Path(path)
    staging = path.with_suffix(path.suffix + ".tmp")
    write_json(staging, value)
    os.replace(staging, path)


def verify_inventory(directory, seal):
    directory = Path(directory).resolve()
    if seal.get("schema_version") != 1 or seal.get("completed") is not True or not seal.get("files"):
        raise ValueError("Pretraining requires a complete hashed artifact inventory.")
    for name, expected in seal["files"].items():
        target = (directory / name).resolve()
        if not target.is_relative_to(directory) or sha256_file(target) != expected:
            raise ValueError(f"Pretraining artifact integrity check failed: {name}")


class PretrainedEncoder:
    """Verified waveform encoder and source-fitted input convention."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.bundle = json.loads((self.directory / "pretraining_bundle.json").read_text(encoding="utf-8"))
        verify_inventory(self.directory, self.bundle)
        required = {"encoder.keras", "waveform_normalizer.json", "model_config.json", "data_manifest.json", "resolved_config.json"}
        if self.bundle.get("kind") != "source_raw_contrastive" or not required <= set(self.bundle["files"]):
            raise ValueError("Unsupported pretraining bundle.")
        self.config = ModelConfig(**json.loads((self.directory / "model_config.json").read_text(encoding="utf-8")))
        self.normalizer = WaveformNormalizer.load(self.directory / "waveform_normalizer.json")
        manifest = json.loads((self.directory / "data_manifest.json").read_text(encoding="utf-8"))
        if manifest != self.normalizer.state["training_data"]:
            raise ValueError("Pretraining normalization and pool manifests disagree.")
        self.encoder = keras.models.load_model(self.directory / "encoder.keras", compile=False)
        if (self.encoder.input_shape != (None, self.config.samples_per_epoch, 1)
                or self.encoder.output_shape[-1] != self.config.token_dim):
            raise ValueError("Pretraining encoder and configuration dimensions disagree.")

    def transfer(self, model, config, partition, *, freeze=False, allow_source_subset=False):
        self.normalizer.assert_compatible(partition, require_training_match=True, allow_source_subset=allow_source_subset)
        for key in ["samples_per_epoch", "filters", "kernels", "strides", "token_dim"]:
            if getattr(config, key) != getattr(self.config, key):
                raise ValueError(f"Pretrained waveform architecture disagrees: {key}")
        target = get_waveform_encoder(model)
        target.set_weights(self.encoder.get_weights())
        target.trainable = not freeze
        return {"bundle_sha256": sha256_file(self.directory / "pretraining_bundle.json"),
                "encoder_sha256": self.bundle["files"]["encoder.keras"], "frozen_waveform_encoder": freeze}


def train_contrastive(root, run_directory, model_config=None, pretraining_config=None, augmentation_config=None, *,
                      train_recordings=None, split_file=None, max_batches=None, resume_from=None, verbose=2,
                      normalizer_path=None):
    config = model_config or ModelConfig(variant="raw")
    training = pretraining_config or PretrainingConfig()
    augmentation = augmentation_config or AugmentationConfig()
    if config.variant != "raw":
        raise ValueError("The initial SSL objective pretrains the raw waveform encoder only.")
    run = Path(run_directory).resolve()
    run.mkdir(parents=True, exist_ok=False)
    write_json(run / "status.json", {"status": "running"})
    started = time.perf_counter()
    try:
        keras.utils.set_random_seed(training.seed)
        tf.config.experimental.enable_op_determinism()
        with SourceEpochPool(root, recordings=train_recordings, split_file=split_file) as pool:
            samples = int(round(pool.preprocessing.epoch_duration * pool.preprocessing.sampling_frequency))
            if config.samples_per_epoch != samples:
                raise ValueError("Pretraining model sample count disagrees with stored waveforms.")
            resolved = {
                "model": config.to_dict(), "pretraining": asdict(training), "augmentation": asdict(augmentation),
                "max_batches": max_batches, "train_recordings": train_recordings,
                "limited_data_run": train_recordings is not None or max_batches is not None,
                "checkpoint_selection": "last_epoch_of_declared_fixed_schedule_no_held_out_inputs",
                "sampling": "global_epoch_shuffle_without_replacement_drop_incomplete_batch",
                "rng_policy": "stateless_views_and_sampler_seeded_by_seed_epoch_batch_at_epoch_boundaries",
                "data_root": str(Path(root).resolve()), "python": platform.python_version(),
                "packages": {p: importlib.metadata.version(p) for p in ["tensorflow", "keras", "numpy", "pandas"]},
                "devices": [device.name for device in tf.config.list_physical_devices()],
                "resume_from": str(Path(resume_from).resolve()) if resume_from else None,
                "normalizer_reused_from": str(Path(normalizer_path).resolve()) if normalizer_path else None,
                "code_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): sha256_file(p) for p in [
                    Path(__file__), ROOT / "src/models/contrastive.py", ROOT / "src/models/multiview.py",
                    ROOT / "src/data/contrastive.py", ROOT / "src/data/supervised.py",
                    ROOT / "src/preprocessing/waveform_normalization.py"]},
            }
            # Compare the same portable representation used by saved JSON;
            # dataclass tuples become lists when a previous run is loaded.
            resolved = json.loads(json.dumps(resolved))
            model = build_contrastive_model(config, training.projection_dim)
            encoder = model.get_layer("waveform_encoder")
            optimizer = keras.optimizers.Adam(training.learning_rate, clipnorm=1.)
            optimizer.build(model.trainable_variables)
            epoch_state = tf.Variable(0, dtype=tf.int64, trainable=False, name="completed_epochs")
            checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer, completed_epochs=epoch_state)
            history = []
            if resume_from:
                previous = Path(resume_from).resolve()
                seal = json.loads((previous / "resume_state.json").read_text(encoding="utf-8"))
                verify_inventory(previous, seal)
                prior = json.loads((previous / "resolved_config.json").read_text(encoding="utf-8"))
                # Epoch counts may extend a constant-LR schedule; all update and
                # augmentation settings, data identities and code must match.
                for key in ["model", "augmentation", "max_batches", "code_sha256", "packages"]:
                    if prior[key] != resolved[key]:
                        raise ValueError(f"Resume configuration disagrees: {key}")
                previous_training, current_training = dict(prior["pretraining"]), dict(resolved["pretraining"])
                previous_training.pop("epochs")
                current_training.pop("epochs")
                if previous_training != current_training:
                    raise ValueError("Resume pretraining settings disagree.")
                normalizer = WaveformNormalizer.load(previous / "waveform_normalizer.json")
                normalizer.assert_compatible(pool, require_training_match=True)
                if source_identity(json.loads((previous / "data_manifest.json").read_text(encoding="utf-8"))) != source_identity(pool.manifest()):
                    raise ValueError("Resume source data disagree.")
                prefix = str((previous / seal["checkpoint_prefix"]).resolve())
                if not Path(prefix).is_relative_to(previous):
                    raise ValueError("Resume checkpoint escaped its run directory.")
                checkpoint.read(prefix).assert_consumed()
                if int(epoch_state.numpy()) != seal["completed_epochs"]:
                    raise ValueError("Checkpoint epoch counter disagrees with its seal.")
                history = json.loads((previous / seal["history_file"]).read_text(encoding="utf-8"))
                if len(history) != int(epoch_state.numpy()):
                    raise ValueError("Resume history disagrees with the checkpoint.")
            else:
                normalizer = WaveformNormalizer.load(normalizer_path) if normalizer_path else WaveformNormalizer().fit(pool)
                normalizer.assert_compatible(pool, require_training_match=True)
            if resume_from and normalizer_path:
                supplied = WaveformNormalizer.load(normalizer_path)
                supplied.assert_compatible(pool, require_training_match=True)
                if supplied.state["waveform_scale"] != normalizer.state["waveform_scale"]:
                    raise ValueError("Resume waveform scale disagrees with supplied normalization.")
            initial_epoch = int(epoch_state.numpy())
            if initial_epoch >= training.epochs:
                raise ValueError("Resume target epochs must exceed the saved completed epoch count.")
            normalizer.save(run / "waveform_normalizer.json")
            write_json(run / "model_config.json", config.to_dict())
            write_json(run / "data_manifest.json", pool.manifest())
            write_json(run / "resolved_config.json", resolved)
            batches = EpochBatches(pool, normalizer, batch_size=training.batch_size, seed=training.seed,
                                   max_batches=max_batches, exclusion_epochs=training.exclusion_epochs)
            diagnostic_indices = np.linspace(0, len(pool) - 1, min(training.diagnostic_epochs, len(pool)), dtype=np.int64)
            diagnostic_raw, diagnostic_ids = pool.gather(diagnostic_indices)
            diagnostic_raw = normalizer.transform_waveform(diagnostic_raw)
            initial_weights = [weight.copy() for weight in encoder.get_weights()]
            initial_diagnostics = representation_diagnostics(encoder, diagnostic_raw)

            @tf.function(reduce_retracing=True)
            def update(raw, allowed, seed):
                seeds = tf.random.experimental.stateless_split(seed, 2)
                first = augment_waveform(raw, seeds[0], augmentation, pool.preprocessing.sampling_frequency)
                second = augment_waveform(raw, seeds[1], augmentation, pool.preprocessing.sampling_frequency)
                with tf.GradientTape() as tape:
                    projected = model(tf.concat([first, second], axis=0), training=True)
                    loss, diagnostics = nt_xent(projected[:training.batch_size], projected[training.batch_size:],
                                                training.temperature, allowed)
                gradients = tape.gradient(loss, model.trainable_variables)
                tf.debugging.assert_all_finite(loss, "Nonfinite contrastive loss")
                for gradient in gradients:
                    tf.debugging.assert_all_finite(gradient, "Nonfinite contrastive gradient")
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                return {"loss": loss, **diagnostics}

            base_names = ["waveform_normalizer.json", "model_config.json", "data_manifest.json", "resolved_config.json"]
            for epoch in range(initial_epoch, training.epochs):
                values = []
                for batch, (raw, allowed, _) in enumerate(batches.iter_batches(epoch)):
                    # fold_in avoids collisions between epoch and batch numbers.
                    seed = tf.random.experimental.stateless_fold_in([training.seed, epoch], batch)
                    result = update(raw, allowed, seed)
                    values.append({key: float(value.numpy()) for key, value in result.items()})
                row = {"epoch": epoch + 1, **{key: float(np.mean([v[key] for v in values])) for key in values[0]}}
                diagnostics = representation_diagnostics(encoder, diagnostic_raw)
                row.update({f"encoder_{key}": value for key, value in diagnostics.items()})
                history.append(row)
                epoch_state.assign(epoch + 1)
                snapshot = run / "checkpoints" / f"epoch_{epoch + 1:04d}"
                snapshot.mkdir(parents=True, exist_ok=False)
                prefix = snapshot / "state"
                checkpoint.write(str(prefix))
                history_file = snapshot / "history.json"
                write_json(history_file, history)
                names = base_names + [p.relative_to(run).as_posix() for p in sorted(snapshot.iterdir())]
                atomic_json(run / "resume_state.json", {
                    "schema_version": 1, "completed": True, "completed_epochs": epoch + 1,
                    "checkpoint_prefix": prefix.relative_to(run).as_posix(),
                    "history_file": history_file.relative_to(run).as_posix(),
                    "files": {name: sha256_file(run / name) for name in names},
                })
                if verbose:
                    print(f"SSL epoch {epoch + 1}/{training.epochs}: loss={row['loss']:.5f}, retrieval={row['retrieval_accuracy']:.3f}, encoder rank={diagnostics['covariance_effective_rank']:.2f}", flush=True)
            encoder.save(run / "encoder.keras")
            loaded = keras.models.load_model(run / "encoder.keras", compile=False)
            reload_difference = float(np.max(np.abs(encoder(diagnostic_raw, training=False).numpy() - loaded(diagnostic_raw, training=False).numpy())))
            if reload_difference > 1e-6:
                raise ValueError("Pretraining encoder reload disagrees with trained weights.")
            weight_difference = max(float(np.max(np.abs(a - b))) for a, b in zip(initial_weights, encoder.get_weights()))
            write_json(run / "diagnostics.json", {"initial": initial_diagnostics, "final": diagnostics,
                                                  "epoch_ids": [m["epoch_id"] for m in diagnostic_ids]})
            with (run / "training_history.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(history[0]))
                writer.writeheader()
                writer.writerows(history)
            summary = {
                "epochs_completed": training.epochs, "resumed_after_epoch": initial_epoch,
                "source_quality_epochs_available": len(pool), "normalization_fit_epochs": normalizer.state["fit_epochs"],
                "original_epochs_per_batch": training.batch_size, "batches_per_epoch": len(batches),
                "original_epochs_used_per_epoch": len(batches) * training.batch_size,
                "incomplete_tail_epochs_per_pass": len(pool) % training.batch_size,
                "maximum_negatives_per_anchor": 2 * (training.batch_size - 1),
                "optimizer_iterations": int(optimizer.iterations.numpy()), "parameters": model.count_params(),
                "encoder_parameters": encoder.count_params(), "limited_data_run": resolved["limited_data_run"],
                "elapsed_seconds_this_run": time.perf_counter() - started,
                "encoder_max_weight_update_this_run": weight_difference,
                "encoder_reload_max_difference": reload_difference, "last_epoch": history[-1],
            }
            write_json(run / "summary.json", summary)
            names = base_names + ["encoder.keras", "diagnostics.json", "training_history.csv", "summary.json", "resume_state.json"]
            write_json(run / "pretraining_bundle.json", {
                "schema_version": 1, "completed": True, "kind": "source_raw_contrastive",
                "files": {name: sha256_file(run / name) for name in names},
            })
        write_json(run / "status.json", {"status": "completed"})
        return summary
    except Exception as exc:
        write_json(run / "status.json", {"status": "failed", "error": type(exc).__name__, "message": str(exc)})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/processed/research_v1")
    parser.add_argument("--split-file", type=Path)
    parser.add_argument("--train-recordings", nargs="+")
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--normalizer", type=Path)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if set(config) - {"model", "pretraining", "augmentation"}:
        parser.error("Unknown top-level configuration keys.")
    training = dict(config.get("pretraining", {}))
    for key in ["epochs", "batch_size"]:
        if getattr(args, key) is not None:
            training[key] = getattr(args, key)
    summary = train_contrastive(args.data_root, args.run_dir, ModelConfig(**config.get("model", {"variant": "raw"})),
                                PretrainingConfig(**training), AugmentationConfig(**config.get("augmentation", {})),
                                train_recordings=args.train_recordings, split_file=args.split_file,
                                max_batches=args.max_batches, resume_from=args.resume_from, normalizer_path=args.normalizer)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

"""Train compact EEG controls using source training and validation participants.

No test or external partition is loaded by this entry point. A run directory is
immutable once created; completed inference bundles are sealed with file hashes.
"""

import argparse
import csv
from dataclasses import asdict, dataclass
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from src.data.epoch_store import sha256_file
from src.data.prepare_research_data import ROOT
from src.data.supervised import SourcePartition, WindowBatches
from src.evaluation.classification import ClassificationAccumulator
from src.models.multiview import ModelConfig, build_multiview_model, compile_multiview_model
from src.preprocessing.normalization import ALL_FEATURES
from src.preprocessing.research_normalization import ResearchNormalizer
from src.preprocessing.sequences import CLASS_LABELS


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 4
    epochs: int = 30
    learning_rate: float = 0.001
    seed: int = 42
    patience: int = 5

    def __post_init__(self):
        for name in ["batch_size", "epochs", "patience"]:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32).")


def evaluate_batches(model, batches, prediction_path=None):
    """Evaluate one deterministic source-validation pass, optionally exporting IDs."""
    if batches.partition.split != "val":
        raise ValueError("Development evaluation only accepts source validation participants.")
    accumulator = ClassificationAccumulator()
    stream = None
    writer = None
    if prediction_path is not None:
        stream = Path(prediction_path).open("w", encoding="utf-8", newline="")
        writer = csv.DictWriter(stream, fieldnames=[
            "dataset", "participant_id", "recording_id", "epoch_index", "epoch_id",
            "start_time", "true_stage", "predicted_stage", *[f"prob_{s}" for s in CLASS_LABELS],
        ])
        writer.writeheader()
    try:
        for inputs, targets, metadata in batches.iter_batches(epoch=0):
            probabilities = model(inputs, training=False).numpy()
            accumulator.update(targets, probabilities)
            if writer:
                for item, probability in zip(metadata, probabilities):
                    writer.writerow({**item, "predicted_stage": CLASS_LABELS[int(probability.argmax())],
                                     **{f"prob_{s}": float(p) for s, p in zip(CLASS_LABELS, probability)}})
    finally:
        if stream:
            stream.close()
    return accumulator.result()


class SourceValidation(keras.callbacks.Callback):
    """Compute val loss/F1 once per epoch before checkpoint/early-stop callbacks."""

    def __init__(self, batches):
        super().__init__()
        self.batches = batches

    def on_epoch_end(self, epoch, logs=None):
        metrics = evaluate_batches(self.model, self.batches)
        for key in ["loss", "accuracy", "macro_f1", "balanced_accuracy"]:
            logs[f"val_{key}"] = metrics[key]


def write_json(path, content):
    Path(path).write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")


def train_supervised(
    root, run_directory, model_config=None, training_config=None, *,
    train_recordings=None, validation_recordings=None, split_file=None,
    max_train_batches=None, max_validation_batches=None, normalizer_path=None, verbose=2,
):
    model_config = model_config or ModelConfig()
    training_config = training_config or TrainingConfig()
    run_directory = Path(run_directory).resolve()
    # Refuse to replace a previous experiment, including an interrupted run.
    run_directory.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    write_json(run_directory / "status.json", {"status": "running"})
    try:
        keras.utils.set_random_seed(training_config.seed)
        tf.config.experimental.enable_op_determinism()
        selection = {
            "train_recordings": train_recordings, "validation_recordings": validation_recordings,
            "max_train_batches": max_train_batches, "max_validation_batches": max_validation_batches,
        }
        limited = any(value is not None for value in selection.values())
        with SourcePartition(root, "train", sequence_length=model_config.sequence_length,
                             recordings=train_recordings, split_file=split_file) as train, \
             SourcePartition(root, "val", sequence_length=model_config.sequence_length,
                             recordings=validation_recordings, split_file=split_file) as validation:
            if train.preprocessing != validation.preprocessing:
                raise ValueError("Training and validation preprocessing disagree.")
            samples = int(round(train.preprocessing.epoch_duration * train.preprocessing.sampling_frequency))
            if samples != model_config.samples_per_epoch:
                raise ValueError("Model sample count disagrees with the prepared EEG epochs.")
            training_participants = {r.participant_id for r in train.recordings}
            validation_participants = {r.participant_id for r in validation.recordings}
            if training_participants & validation_participants:
                raise ValueError("Training and validation participants overlap.")
            if normalizer_path is None:
                normalizer = ResearchNormalizer().fit(train)
            else:
                normalizer = ResearchNormalizer.load(normalizer_path)
                normalizer.assert_compatible(train, require_training_match=True)
            normalizer.save(run_directory / "normalizer.json")
            data_manifest = {"train": train.manifest(), "validation": validation.manifest()}
            write_json(run_directory / "data_manifest.json", data_manifest)
            write_json(run_directory / "model_config.json", model_config.to_dict())
            resolved = {
                "model": model_config.to_dict(), "training": asdict(training_config),
                "data_root": str(Path(root).resolve()), "selection": selection,
                "limited_data_run": limited, "normalizer_reused_from": str(normalizer_path) if normalizer_path else None,
                "checkpoint_selection": "maximum_source_validation_fixed_five_class_macro_f1",
                "capped_validation_selection": "evenly_spaced_windows_without_using_labels",
                "classes": CLASS_LABELS, "features": ALL_FEATURES,
                "python": platform.python_version(),
                "packages": {p: importlib.metadata.version(p) for p in ["tensorflow", "keras", "numpy", "pandas", "scikit-learn"]},
                "devices": [device.name for device in tf.config.list_physical_devices()],
                "code_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): sha256_file(p)
                                for p in [Path(__file__), ROOT / "src/models/multiview.py",
                                          ROOT / "src/data/supervised.py", ROOT / "src/preprocessing/research_normalization.py"]},
            }
            write_json(run_directory / "resolved_config.json", resolved)
            training_batches = WindowBatches(
                train, normalizer, variant=model_config.variant, batch_size=training_config.batch_size,
                shuffle=True, seed=training_config.seed, max_batches=max_train_batches,
            )
            validation_batches = WindowBatches(
                validation, normalizer, variant=model_config.variant, batch_size=training_config.batch_size,
                max_batches=max_validation_batches,
            )
            # Seeding occurs before constructing any layers/initializers.
            keras.utils.set_random_seed(training_config.seed)
            model = compile_multiview_model(build_multiview_model(model_config), training_config.learning_rate)
            checkpoint = run_directory / "model.keras"
            callbacks = [
                SourceValidation(validation_batches),
                keras.callbacks.ModelCheckpoint(checkpoint, monitor="val_macro_f1", mode="max", save_best_only=True),
                keras.callbacks.EarlyStopping(monitor="val_macro_f1", mode="max", patience=training_config.patience,
                                             restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
                keras.callbacks.CSVLogger(run_directory / "training_history.csv"),
            ]
            model.fit(training_batches.as_dataset(), epochs=training_config.epochs, callbacks=callbacks,
                      shuffle=False, verbose=verbose)
            # Verify the actual serialized checkpoint, not only the in-memory model.
            loaded = keras.models.load_model(checkpoint, compile=False)
            sample, _, _ = next(validation_batches.iter_batches(epoch=0))
            reload_difference = float(np.max(np.abs(model(sample, training=False).numpy() - loaded(sample, training=False).numpy())))
            if reload_difference > 1e-6:
                raise ValueError("Reloaded best checkpoint predictions disagree with restored training weights.")
            metrics = evaluate_batches(loaded, validation_batches, run_directory / "validation_predictions.csv")
            write_json(run_directory / "validation_metrics.json", metrics)
            pd.DataFrame(metrics["confusion_matrix"], index=CLASS_LABELS, columns=CLASS_LABELS).to_csv(run_directory / "validation_confusion_matrix.csv")
            history = pd.read_csv(run_directory / "training_history.csv")
            summary = {
                "variant": model_config.variant, "parameters": model.count_params(),
                "epochs_completed": len(history),
                "best_epoch": int(history["val_macro_f1"].to_numpy().argmax()) + 1,
                "train_windows_available": len(train), "validation_windows_available": len(validation),
                "train_batches_per_epoch": len(training_batches), "validation_batches": len(validation_batches),
                "normalization_fit_epochs": normalizer.state["fit_epochs"],
                "limited_data_run": limited, "elapsed_seconds": time.perf_counter() - started,
                "checkpoint_reload_max_probability_difference": reload_difference,
                "source_validation": metrics,
            }
            write_json(run_directory / "summary.json", summary)
            write_json(run_directory / "bundle.json", {
                "schema_version": 1, "completed": True, "classes": CLASS_LABELS, "features": ALL_FEATURES,
                "preprocessing_fingerprint": train.preprocessing.fingerprint,
                "files": {name: sha256_file(run_directory / name) for name in [
                    "model.keras", "normalizer.json", "model_config.json", "data_manifest.json", "resolved_config.json",
                ]},
            })
        write_json(run_directory / "status.json", {"status": "completed"})
        return summary
    except Exception as exc:
        write_json(run_directory / "status.json", {"status": "failed", "error": type(exc).__name__, "message": str(exc)})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/processed/research_v1")
    parser.add_argument("--split-file", type=Path)
    parser.add_argument("--train-recordings", nargs="+")
    parser.add_argument("--validation-recordings", nargs="+")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-validation-batches", type=int)
    parser.add_argument("--normalizer", type=Path)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if set(config) - {"model", "training"}:
        parser.error("Unknown top-level configuration keys.")
    training = dict(config.get("training", {}))
    for name in ["epochs", "batch_size"]:
        if getattr(args, name) is not None:
            training[name] = getattr(args, name)
    summary = train_supervised(
        args.data_root, args.run_dir, ModelConfig(**config.get("model", {})), TrainingConfig(**training),
        train_recordings=args.train_recordings, validation_recordings=args.validation_recordings,
        split_file=args.split_file, max_train_batches=args.max_train_batches,
        max_validation_batches=args.max_validation_batches, normalizer_path=args.normalizer,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

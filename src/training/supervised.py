"""Train compact EEG controls using source training and validation participants.

No test or external partition is loaded by this entry point. A run directory is
immutable once created; completed inference bundles are sealed with file hashes.
"""

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
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from src.data.epoch_store import sha256_file
from src.data.prepare_research_data import ROOT
from src.data.supervised import SourcePartition, WindowBatches
from src.data.label_budgets import (authorize_waveform_pool, get_budget, load_budget_plan, verify_plan_scope)
from src.evaluation.classification import ClassificationAccumulator
from src.models.multiview import (ModelConfig, build_linear_probe, build_multiview_model,
                                  compile_multiview_model, get_waveform_encoder)
from src.preprocessing.normalization import ALL_FEATURES
from src.preprocessing.research_normalization import ResearchNormalizer
from src.preprocessing.sequences import CLASS_LABELS
from src.preprocessing.waveform_normalization import WaveformNormalizer
from src.training.artifacts import write_json
from src.training.progress import BatchProgress


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 4
    epochs: int = 30
    learning_rate: float = 0.001
    seed: int = 42
    patience: int = 5
    validation_batch_size: int | None = None
    compiled_validation: bool = False
    recovery: bool = False

    def __post_init__(self):
        for name in ["batch_size", "epochs", "patience"]:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32).")
        if self.validation_batch_size is not None and (isinstance(self.validation_batch_size, bool)
                or not isinstance(self.validation_batch_size, int) or self.validation_batch_size < 1):
            raise ValueError('validation_batch_size must be a positive integer or null.')
        if not isinstance(self.compiled_validation, bool) or not isinstance(self.recovery, bool):
            raise ValueError('compiled_validation and recovery must be boolean.')


def validation_predictor(model, compiled=False):
    if not compiled:
        return lambda inputs: model(inputs, training=False)
    signature = {name: tf.TensorSpec((None, *tensor.shape[1:]), tf.as_dtype(tensor.dtype))
                 for name, tensor in model.input.items()}
    return tf.function(lambda inputs: model(inputs, training=False), input_signature=[signature], autograph=False)


def evaluate_batches(model, batches, prediction_path=None, *, compiled=False, predictor=None):
    """Evaluate one deterministic source-validation pass, optionally exporting IDs."""
    if batches.partition.split != "val":
        raise ValueError("Development evaluation only accepts source validation participants.")
    predictor = predictor or validation_predictor(model, compiled)
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
            probabilities = predictor(inputs).numpy()
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

    def __init__(self, batches, compiled=False):
        super().__init__()
        self.batches = batches
        self.compiled = compiled
        self.predictor = None

    def on_epoch_end(self, epoch, logs=None):
        if self.predictor is None:
            self.predictor = validation_predictor(self.model, self.compiled)
        metrics = evaluate_batches(self.model, self.batches, predictor=self.predictor)
        for key in ["loss", "accuracy", "macro_f1", "balanced_accuracy"]:
            logs[f"val_{key}"] = metrics[key]


class SourceTrainingProgress(keras.callbacks.Callback):
    def __init__(self, batches, epochs, interval):
        super().__init__()
        self.batches, self.epochs, self.interval = batches, epochs, interval

    def on_epoch_begin(self, epoch, logs=None):
        self.progress = BatchProgress("Supervised", epoch + 1, self.epochs, self.batches, self.interval)

    def on_train_batch_end(self, batch, logs=None):
        self.progress.update(batch + 1, logs)
        if batch + 1 == self.batches:
            print("Training pass complete; computing source-validation metrics...", flush=True)


def train_supervised(
    root, run_directory, model_config=None, training_config=None, *,
    train_recordings=None, validation_recordings=None, split_file=None,
    max_train_batches=None, max_validation_batches=None, normalizer_path=None, verbose=2,
    pretrained_path=None, freeze_backbone=False, linear_probe=False,
    waveform_normalizer_path=None,
    budget_plan_path=None, budget_id=None, progress_interval=0, resume_from=None,
):
    model_config = model_config or ModelConfig()
    training_config = training_config or TrainingConfig()
    if resume_from is not None and not training_config.recovery:
        raise ValueError('Supervised resume requires training.recovery=true.')
    if training_config.recovery and (os.environ.get('PYTHONHASHSEED') is None
            or not os.environ['PYTHONHASHSEED'].isdigit()
            or not 0 <= int(os.environ['PYTHONHASHSEED']) < 2**32):
        raise ValueError('Recovery requires a fixed PYTHONHASHSEED before Python starts; use scripts/research_cpu.ps1.')
    validation_size = training_config.validation_batch_size or training_config.batch_size
    if max_validation_batches is not None and validation_size != training_config.batch_size:
        raise ValueError('A changed validation batch size would change the capped validation target mask.')
    if linear_probe and (model_config.variant != "raw" or model_config.sequence_length != 1):
        raise ValueError("Linear probes require raw single-epoch configuration.")
    if pretrained_path is not None and waveform_normalizer_path is not None:
        raise ValueError("Pretrained bundles already supply their waveform normalizer.")
    if model_config.variant == "spectral" and (pretrained_path is not None or freeze_backbone):
        raise ValueError("The spectral reference has no waveform backbone to transfer or freeze.")
    budget_plan, budget = None, None
    if (budget_plan_path is None) != (budget_id is None):
        raise ValueError("A label budget requires both its plan and budget ID.")
    if budget_plan_path is not None:
        budget_plan = load_budget_plan(budget_plan_path)
        verify_plan_scope(budget_plan, root, split_file)
        budget = get_budget(budget_plan, budget_id)
        if train_recordings is not None or validation_recordings is not None:
            raise ValueError("A label budget controls complete training/validation recording selections.")
        if model_config.sequence_length != budget_plan["sequence_length"]:
            raise ValueError("Model context disagrees with label budget planning.")
        train_recordings = budget["recordings"]
        validation_recordings = [r["recording_id"] for r in budget_plan["source_records"] if r["split"] == "val"]
    run_directory = Path(run_directory).resolve()
    # Refuse to replace a previous experiment, including an interrupted run.
    run_directory.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    write_json(run_directory / "status.json", {"status": "running"})
    try:
        if training_config.recovery:
            keras.backend.clear_session()
        keras.utils.set_random_seed(training_config.seed)
        tf.config.experimental.enable_op_determinism()
        if training_config.recovery:
            tf.random.set_global_generator(tf.random.Generator.from_seed(training_config.seed))
        selection = {
            "train_recordings": train_recordings, "validation_recordings": validation_recordings,
            "max_train_batches": max_train_batches, "max_validation_batches": max_validation_batches,
        }
        limited = any(value is not None for value in selection.values())
        if budget_plan:
            limited = budget_plan["limited_participant_scope"] or max_train_batches is not None or max_validation_batches is not None
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
            pretrained = None
            if pretrained_path is not None:
                from src.training.contrastive import PretrainedEncoder
                pretrained = PretrainedEncoder(pretrained_path)
                if budget_plan:
                    authorize_waveform_pool(budget_plan, pretrained.normalizer)
                normalizer.use_pretraining_waveform(pretrained.normalizer, train, allow_source_subset=budget_plan is not None)
            elif waveform_normalizer_path is not None:
                # Match preprocessing in randomly initialized SSL controls.
                waveform = WaveformNormalizer.load(waveform_normalizer_path)
                if budget_plan:
                    authorize_waveform_pool(budget_plan, waveform)
                normalizer.use_pretraining_waveform(waveform, train, allow_source_subset=budget_plan is not None)
            if budget_plan:
                if "pretraining_waveform" not in normalizer.state:
                    raise ValueError("Budget comparisons require the fixed full-pool waveform normalizer.")
                authorize_waveform_pool(budget_plan, WaveformNormalizer(normalizer.state["pretraining_waveform"]))
                if (len(train) != budget["annotation_cost"]["windows"]
                        or train.preprocessing.fingerprint != budget_plan["preprocessing_fingerprint"]):
                    raise ValueError("Actual training windows disagree with the planned label budget.")
            normalizer.save(run_directory / "normalizer.json")
            data_manifest = {"train": train.manifest(), "validation": validation.manifest()}
            write_json(run_directory / "data_manifest.json", data_manifest)
            write_json(run_directory / "model_config.json", model_config.to_dict())
            resolved = {
                "model": model_config.to_dict(), "training": asdict(training_config),
                "data_root": str(Path(root).resolve()), "selection": selection,
                "limited_data_run": limited, "normalizer_reused_from": str(normalizer_path) if normalizer_path else None,
                "training_mode": "linear_probe" if linear_probe else "spectral_reference" if model_config.variant == "spectral" else "temporal_classifier",
                "pretrained_from": str(Path(pretrained_path).resolve()) if pretrained_path else None,
                "waveform_normalizer_from": str(Path(waveform_normalizer_path).resolve()) if waveform_normalizer_path else None,
                "frozen_waveform_encoder": freeze_backbone or linear_probe,
                "label_budget": {"plan": budget_plan, "budget_id": budget_id} if budget_plan else None,
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
            resolved['resume_from'] = str(Path(resume_from).resolve()) if resume_from else None
            resolved['validation_execution'] = {'batch_size': validation_size, 'compiled': training_config.compiled_validation}
            resolved['sampler_policy'] = ('explicit_seed_plus_completed_epoch' if training_config.recovery
                                          else 'legacy_generator_iteration')
            training_batches = WindowBatches(
                train, normalizer, variant=model_config.variant, batch_size=training_config.batch_size,
                shuffle=True, seed=training_config.seed, max_batches=max_train_batches,
            )
            validation_batches = WindowBatches(
                validation, normalizer, variant=model_config.variant, batch_size=validation_size,
                max_batches=max_validation_batches,
            )
            # Seeding occurs before constructing any layers/initializers.
            keras.utils.set_random_seed(training_config.seed)
            model = build_linear_probe(model_config) if linear_probe else build_multiview_model(model_config)
            if pretrained:
                resolved["pretraining_transfer"] = pretrained.transfer(
                    model, model_config, train, freeze=freeze_backbone or linear_probe, allow_source_subset=budget_plan is not None)
            elif freeze_backbone:
                get_waveform_encoder(model).trainable = False
            frozen_weights = [v.copy() for v in get_waveform_encoder(model).get_weights()] if freeze_backbone or linear_probe else None
            model = compile_multiview_model(model, training_config.learning_rate)
            write_json(run_directory / "resolved_config.json", resolved)
            checkpoint = run_directory / "model.keras"
            initial_epoch, recovery = 0, None
            validation_callback = SourceValidation(validation_batches, training_config.compiled_validation)
            callbacks = [
                validation_callback,
                keras.callbacks.ModelCheckpoint(checkpoint, monitor="val_macro_f1", mode="max", save_best_only=True),
                keras.callbacks.EarlyStopping(monitor="val_macro_f1", mode="max", patience=training_config.patience,
                                             restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
                keras.callbacks.CSVLogger(run_directory / "training_history.csv"),
            ]
            if progress_interval:
                callbacks.append(SourceTrainingProgress(len(training_batches), training_config.epochs, progress_interval))
            if training_config.recovery:
                from src.training.supervised_recovery import fit_recoverable
                from src.experiments.run import code_hashes
                # Absolute data provenance, complete source hashes and runtime identity
                # intentionally forbid silently migrating an exact-resume claim.
                contract = {'model': model_config.to_dict(), 'training': asdict(training_config),
                            'data': data_manifest, 'normalizer_sha256': sha256_file(run_directory / 'normalizer.json'),
                            'selection': selection, 'training_mode': resolved['training_mode'],
                            'frozen_waveform_encoder': resolved['frozen_waveform_encoder'],
                            'initialization': resolved.get('pretraining_transfer'),
                            'code_sha256': code_hashes(), 'python': platform.python_version(),
                            'platform': platform.platform(), 'processor': platform.processor(),
                            'execution_environment': {key: os.environ.get(key) for key in
                                ('TF_ENABLE_ONEDNN_OPTS', 'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS',
                                 'OMP_NUM_THREADS', 'CUDA_VISIBLE_DEVICES', 'PYTHONHASHSEED')},
                            'precision_policy': keras.mixed_precision.global_policy().name,
                            'jit': tf.config.optimizer.get_jit(),
                            'packages': {distribution.metadata['Name']: distribution.version
                                         for distribution in importlib.metadata.distributions()},
                            'devices': [{'name': device.name, 'details': tf.config.experimental.get_device_details(device)}
                                        for device in tf.config.list_physical_devices()],
                            'sampler_policy': resolved['sampler_policy']}
                initial_epoch, recovery = fit_recoverable(model, training_batches, run_directory, training_config,
                    validation_callback, contract, SourceTrainingProgress(len(training_batches), training_config.epochs, progress_interval)
                    if progress_interval else None, resume_from=resume_from, verbose=verbose, elapsed_start=started)
            else:
                model.fit(training_batches.as_dataset(), epochs=training_config.epochs, callbacks=callbacks,
                          shuffle=False, verbose=verbose)
            if frozen_weights is not None:
                for before, after in zip(frozen_weights, get_waveform_encoder(model).get_weights()):
                    if not np.array_equal(before, after):
                        raise ValueError("Frozen waveform weights changed during head training.")
            # Verify the actual serialized checkpoint, not only the in-memory model.
            loaded = keras.models.load_model(checkpoint, compile=False)
            sample, _, _ = next(validation_batches.iter_batches(epoch=0))
            reload_difference = float(np.max(np.abs(model(sample, training=False).numpy() - loaded(sample, training=False).numpy())))
            if reload_difference > 1e-6:
                raise ValueError("Reloaded best checkpoint predictions disagree with restored training weights.")
            metrics = evaluate_batches(loaded, validation_batches, run_directory / "validation_predictions.csv",
                                       compiled=training_config.compiled_validation)
            write_json(run_directory / "validation_metrics.json", metrics)
            pd.DataFrame(metrics["confusion_matrix"], index=CLASS_LABELS, columns=CLASS_LABELS).to_csv(run_directory / "validation_confusion_matrix.csv")
            history = pd.read_csv(run_directory / "training_history.csv")
            summary = {
                "variant": model_config.variant, "parameters": model.count_params(),
                "temporal_head": model_config.temporal_head, "temporal_units": model_config.temporal_units,
                "training_mode": resolved["training_mode"],
                "trainable_parameters": int(sum(np.prod(v.shape) for v in model.trainable_weights)),
                "pretrained_waveform": pretrained is not None,
                "label_budget": budget,
                "validation_annotation_cost": budget_plan["validation_annotation_cost"] if budget_plan else None,
                "epochs_completed": len(history),
                "best_epoch": int(history["val_macro_f1"].to_numpy().argmax()) + 1,
                "train_windows_available": len(train), "validation_windows_available": len(validation),
                "train_batches_per_epoch": len(training_batches), "validation_batches": len(validation_batches),
                "normalization_fit_epochs": normalizer.state["fit_epochs"],
                "waveform_normalization_fit_epochs": normalizer.state.get("pretraining_waveform", normalizer.state)["fit_epochs"],
                "limited_data_run": limited,
                "elapsed_seconds": time.perf_counter() - started + (recovery.previous_elapsed if recovery else 0.),
                'elapsed_seconds_this_call': time.perf_counter() - started,
                'elapsed_seconds_policy': 'current_call_plus_published_prior_segments_excluding_uncommitted_interruption_time',
                "checkpoint_reload_max_probability_difference": reload_difference,
                'resumed_after_epoch': initial_epoch,
                'recovery_enabled': training_config.recovery,
                'elapsed_seconds_previous_completed_segments': recovery.previous_elapsed if recovery else 0.,
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
    parser.add_argument('--validation-batch-size', type=int)
    parser.add_argument('--compiled-validation', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--recovery', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--resume-from', type=Path)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-validation-batches", type=int)
    parser.add_argument("--normalizer", type=Path)
    parser.add_argument("--pretrained", type=Path)
    parser.add_argument("--waveform-normalizer", type=Path)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--linear-probe", action="store_true")
    parser.add_argument("--budget-plan", type=Path)
    parser.add_argument("--budget-id")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if set(config) - {"model", "training"}:
        parser.error("Unknown top-level configuration keys.")
    training = dict(config.get("training", {}))
    for name in ["epochs", "batch_size", 'validation_batch_size', 'compiled_validation', 'recovery']:
        if getattr(args, name) is not None:
            training[name] = getattr(args, name)
    summary = train_supervised(
        args.data_root, args.run_dir, ModelConfig(**config.get("model", {})), TrainingConfig(**training),
        train_recordings=args.train_recordings, validation_recordings=args.validation_recordings,
        split_file=args.split_file, max_train_batches=args.max_train_batches,
        max_validation_batches=args.max_validation_batches, normalizer_path=args.normalizer,
        pretrained_path=args.pretrained, freeze_backbone=args.freeze_backbone, linear_probe=args.linear_probe,
        waveform_normalizer_path=args.waveform_normalizer,
        budget_plan_path=args.budget_plan, budget_id=args.budget_id, resume_from=args.resume_from,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

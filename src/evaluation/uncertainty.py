"""Participant-cluster uncertainty, paired contrasts and probability diagnostics.

Participants, including all their recordings and trained seeds, are resampled
together. Intervals are conditional on the supplied trained seed set; they do
not reinterpret participant-seed pairs as independent people.
"""

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.label_budgets import fingerprint
from src.evaluation.classification import ClassificationAccumulator
from src.evaluation.artifacts import inspect_bundle, verify_validation_frame
from src.preprocessing.sequences import CLASS_LABELS, CLASS_TO_INDEX
from src.training.artifacts import write_json


@dataclass(frozen=True)
class AnalysisConfig:
    bootstrap_replicates: int = 2000
    confidence_level: float = .95
    seed: int = 42
    calibration_bins: int = 15

    def __post_init__(self):
        for name, minimum in [("bootstrap_replicates", 100), ("calibration_bins", 2)]:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        if not np.isfinite(self.confidence_level) or not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must be in (0, 1).")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2**32:
            raise ValueError("Analysis seed must be an integer in [0, 2**32).")


def validate_predictions(frame):
    required = {"dataset", "participant_id", "recording_id", "epoch_id", "epoch_index", "true_stage",
                *["prob_" + s for s in CLASS_LABELS]}
    if required - set(frame) or frame.empty or frame[list(required)].isna().any().any():
        raise ValueError("Predictions require complete epoch identities, true stages and five probabilities.")
    if frame.epoch_id.duplicated().any() or not frame.true_stage.isin(CLASS_LABELS).all():
        raise ValueError("Prediction IDs must be unique and targets must use the fixed five classes.")
    if frame.dataset.nunique() != 1:
        raise ValueError("One analysis cohort cannot mix datasets.")
    indices = frame.epoch_index.to_numpy(dtype=float)
    if not np.isfinite(indices).all() or np.any(indices < 0) or not np.equal(indices, np.floor(indices)).all():
        raise ValueError("Prediction epoch indices must be nonnegative integers.")
    for record, group in frame.groupby("recording_id", sort=False):
        if group.participant_id.nunique() != 1:
            raise ValueError("A recording cannot belong to multiple participants.")
        expected = [f"{row.dataset}:{record}:{int(row.epoch_index):06d}" for row in group.itertuples()]
        if group.epoch_id.tolist() != expected:
            raise ValueError("Prediction epoch IDs disagree with their physical identities.")
    probabilities = frame[["prob_" + s for s in CLASS_LABELS]].to_numpy(dtype=np.float64)
    accumulator = ClassificationAccumulator()
    targets = frame.true_stage.map(CLASS_TO_INDEX).to_numpy(dtype=np.int64)
    accumulator.update(targets, probabilities)
    if "predicted_stage" in frame and frame.predicted_stage.tolist() != [CLASS_LABELS[i] for i in probabilities.argmax(axis=1)]:
        raise ValueError("Exported predicted stages disagree with probability argmax.")
    if "eligible" in frame and not frame.eligible.eq(True).all():
        raise ValueError("Analysis expects the declared eligible target set only.")
    return targets, probabilities


def summarize_frame(frame, calibration_bins=15):
    if isinstance(calibration_bins, bool) or not isinstance(calibration_bins, int) or calibration_bins < 2:
        raise ValueError("calibration_bins must be an integer >= 2.")
    targets, probabilities = validate_predictions(frame)
    accumulator = ClassificationAccumulator()
    accumulator.update(targets, probabilities)
    metrics = accumulator.result()
    one_hot = np.eye(5)[targets]
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == targets
    bins = np.minimum((confidence * calibration_bins).astype(int), calibration_bins - 1)
    reliability = []
    ece = 0.
    for number in range(calibration_bins):
        selected = bins == number
        count = int(selected.sum())
        accuracy = float(correct[selected].mean()) if count else None
        mean_confidence = float(confidence[selected].mean()) if count else None
        if count:
            ece += count / len(frame) * abs(accuracy - mean_confidence)
        reliability.append({"bin": number, "lower": number / calibration_bins, "upper": (number + 1) / calibration_bins,
                            "count": count, "accuracy": accuracy, "mean_confidence": mean_confidence})
    metrics.update({"brier_score": float(np.square(probabilities - one_hot).sum(axis=1).mean()),
                    "brier_policy": "mean_sum_over_five_classes_squared_error_range_zero_to_two",
                    "expected_calibration_error": float(ece), "calibration_bins": reliability,
                    "missing_truth_classes": [s for s in CLASS_LABELS if metrics["per_stage"][s]["support"] == 0]})
    if "near_transition" in frame:
        metrics["transition_subgroups"] = {}
        for label, mask in [("near_transition", frame.near_transition.eq(True)), ("other", frame.near_transition.eq(False))]:
            metrics["transition_subgroups"][label] = summarize_frame(frame.loc[mask].drop(columns="near_transition"), calibration_bins) if mask.any() else None
    return metrics


def target_identity(frame):
    columns = ["dataset", "participant_id", "recording_id", "epoch_id", "epoch_index", "true_stage"]
    return fingerprint(frame.sort_values("epoch_id")[columns].astype({"epoch_index": int}).to_dict("records"))


def interval(samples, participants, confidence_level):
    if participants < 2:
        return {"lower": None, "upper": None, "status": "insufficient_independent_participants"}
    alpha = (1 - confidence_level) / 2
    lower, upper = np.quantile(samples, [alpha, 1 - alpha])
    return {"lower": float(lower), "upper": float(upper), "status": "percentile_participant_cluster_bootstrap"}


def analyze_models(models, pairs=(), config=None):
    """models contain model_id, condition, training_seed and predictions DataFrame.

    The condition endpoint averages per-participant F1 across trained seeds,
    then across participants. Seed SD is computed separately from per-seed
    participant-mean endpoints. Pooled scores are per-model secondary results.
    """
    config = config or AnalysisConfig()
    if not models or len({m["model_id"] for m in models}) != len(models):
        raise ValueError("Analysis requires nonempty, unique model IDs.")
    for model in models:
        if (not isinstance(model["condition"], str) or not model["condition"].strip()
                or isinstance(model["training_seed"], bool) or not isinstance(model["training_seed"], (int, np.integer))
                or model["training_seed"] < 0):
            raise ValueError("Analysis requires named conditions and nonnegative integer training seeds.")
    if len({(m["condition"], m["training_seed"]) for m in models}) != len(models):
        raise ValueError("A condition may have only one model per training seed.")
    expected_identity = None
    participants = None
    summaries, participant_rows, vectors = {}, [], {}
    for model in models:
        frame = model["predictions"].copy()
        for column in ["dataset", "participant_id", "recording_id", "epoch_id"]:
            frame[column] = frame[column].astype(str)
        validate_predictions(frame)
        identity = target_identity(frame)
        if expected_identity is None:
            expected_identity = identity
            participants = sorted(frame.participant_id.unique())
        elif expected_identity != identity:
            raise ValueError("Paired analysis requires identical epoch identities, participants and true labels.")
        summaries[model["model_id"]] = summarize_frame(frame, config.calibration_bins)
        values = []
        for participant in participants:
            selected = frame.loc[frame.participant_id == participant]
            summary = summarize_frame(selected, config.calibration_bins)
            values.append(summary["macro_f1"])
            participant_rows.append({"model_id": model["model_id"], "condition": model["condition"], "training_seed": model["training_seed"],
                                     "participant_id": participant, "recordings": selected.recording_id.nunique(), "examples": len(selected),
                                     "macro_f1": summary["macro_f1"], "accuracy": summary["accuracy"], "loss": summary["loss"],
                                     "brier_score": summary["brier_score"], "missing_truth_classes": summary["missing_truth_classes"],
                                     "stage_support": {s: summary["per_stage"][s]["support"] for s in CLASS_LABELS}})
        vectors[model["model_id"]] = np.asarray(values)
    # One common participant draw matrix preserves pairing across all models/seeds.
    count = len(participants)
    draws = np.random.default_rng(config.seed).integers(0, count, (config.bootstrap_replicates, count))
    conditions = {}
    condition_vectors = {}
    for condition in sorted({m["condition"] for m in models}):
        selected = sorted([m for m in models if m["condition"] == condition], key=lambda m: m["training_seed"])
        matrix = np.stack([vectors[m["model_id"]] for m in selected])
        person_means = matrix.mean(axis=0)
        condition_vectors[condition] = person_means
        per_seed = matrix.mean(axis=1)
        conditions[condition] = {"participant_mean_macro_f1": float(person_means.mean()),
                                 "confidence_interval": interval(person_means[draws].mean(axis=1), count, config.confidence_level),
                                 "trained_seeds": [m["training_seed"] for m in selected],
                                 "per_seed_participant_mean_macro_f1": {str(m["training_seed"]): float(value) for m, value in zip(selected, per_seed)},
                                 "seed_standard_deviation": float(per_seed.std(ddof=1)) if len(per_seed) > 1 else None,
                                 "seed_range": [float(per_seed.min()), float(per_seed.max())]}
    comparisons = []
    for first, second in pairs:
        if first not in conditions or second not in conditions or first == second:
            raise ValueError("A declared contrast references missing or identical conditions.")
        if conditions[first]["trained_seeds"] != conditions[second]["trained_seeds"]:
            raise ValueError("Paired condition contrasts require matching training seed sets.")
        difference = condition_vectors[first] - condition_vectors[second]
        comparisons.append({"first": first, "second": second, "participant_mean_macro_f1_difference": float(difference.mean()),
                            "confidence_interval": interval(difference[draws].mean(axis=1), count, config.confidence_level)})
    return {"schema_version": 1, "analysis_config": asdict(config), "participants": count, "participant_ids": participants,
            "target_identity_sha256": expected_identity, "models": summaries, "conditions": conditions,
            "participant_metrics": participant_rows, "paired_comparisons": comparisons,
            "primary_endpoint": "participant_mean_fixed_five_class_macro_f1_averaged_over_supplied_trained_seeds",
            "uncertainty_policy": "resample_participants_with_all_recordings_and_seeds_together_percentile_interval",
            "seed_uncertainty_policy": "participant_CI_conditional_on_trained_seeds_seed_variation_reported_separately_no_seed_bootstrap",
            "interval_limitations": "Percentile intervals are exploratory for small cohorts; one participant yields no interval. No multiplicity adjustment or model-selection correction is applied."}


def analyze_registry(registry_path, output_directory, *, pairs=(), config=None, source_validation_only=True):
    registry_path = Path(registry_path)
    registry = pd.read_csv(registry_path, dtype={"model_id": str, "condition": str})
    if not {"model_id", "condition", "training_seed", "predictions", "bundle_directory"} <= set(registry):
        raise ValueError("Prediction registry requires model_id, condition, training_seed, predictions and bundle_directory.")
    models = []
    for row in registry.itertuples(index=False):
        path = Path(row.predictions)
        path = path if path.is_absolute() else registry_path.parent / path
        frame = pd.read_csv(path, dtype={c: str for c in ["dataset", "participant_id", "recording_id", "epoch_id"]})
        if source_validation_only:
            directory = Path(row.bundle_directory)
            directory = directory if directory.is_absolute() else registry_path.parent / directory
            inspected = inspect_bundle(directory)
            verify_validation_frame(frame, inspected)
            if row.training_seed != inspected["resolved"]["training"]["seed"]:
                raise ValueError("Analysis registry training seed disagrees with its model bundle.")
        models.append({"model_id": row.model_id, "condition": row.condition, "training_seed": int(row.training_seed), "predictions": frame})
    result = analyze_models(models, pairs, config)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "analysis.json", result)
    pd.DataFrame(result["participant_metrics"]).to_csv(output / "participant_metrics.csv", index=False)
    pd.DataFrame(result["paired_comparisons"]).to_csv(output / "paired_comparisons.csv", index=False)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pair", nargs=2, action="append", default=[])
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    result = analyze_registry(args.registry, args.output_dir, pairs=args.pair,
                              config=AnalysisConfig(bootstrap_replicates=args.bootstrap_replicates, seed=args.seed))
    print(json.dumps({"participants": result["participants"], "conditions": result["conditions"]}, indent=2))


if __name__ == "__main__":
    main()

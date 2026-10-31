"""Freeze source-selected models before opening a declared evaluation cohort.

The scoring orchestrator executes the entire frozen family, serially, with no
training or normalization fitting. Confirmatory freezes reject capped models.
"""

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

from src.data.epoch_store import EpochStore, sha256_file
from src.data.label_budgets import default_split, fingerprint
from src.data.prepare_research_data import ROOT, participant_assignments
from src.evaluation.artifacts import BUNDLE_FILES, inspect_bundle, read_json, verify_validation_frame
from src.evaluation.uncertainty import (AnalysisConfig, analyze_models, summarize_frame, target_identity, validate_predictions)
from src.experiments.run import PACKAGES, atomic_json, code_hashes, completed_attempt, load_study
from src.preprocessing.normalization import ALL_FEATURES
from src.preprocessing.sequences import CLASS_LABELS
from src.training.artifacts import write_json


def safe_name(name):
    if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError("Model and condition names must use letters, digits, underscores, dots or hyphens.")
    return name


def normalize_protocol(config):
    if set(config) - {"purpose", "target", "analysis", "paired_conditions", "transition_radius_epochs"}:
        raise ValueError("Unknown evaluation protocol fields.")
    purpose = config.get("purpose", "confirmatory")
    if purpose not in {"confirmatory", "integration"}:
        raise ValueError("Evaluation purpose must be confirmatory or integration.")
    target = dict(config.get("target", {}))
    if set(target) - {"dataset", "split", "expected_participants", "expected_recordings", "recordings", "scorer", "electrode_policy", "max_windows_per_record"}:
        raise ValueError("Unknown target cohort fields.")
    dataset, split = target.get("dataset"), target.get("split")
    if (dataset, split) not in {("isruc", "external"), ("sleepedf", "test"), ("sleepedf", "val")}:
        raise ValueError("Evaluation supports ISRUC external or Sleep-EDF test/validation cohorts.")
    for key in ["expected_participants", "expected_recordings"]:
        value = target.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError("Expected cohort participant/recording counts must be explicit positive integers.")
    records = target.get("recordings")
    if records is not None and (not records or len(set(records)) != len(records) or any(not re.fullmatch(r"[A-Za-z0-9_-]+", r) for r in records)):
        raise ValueError("Explicit evaluation recording IDs must be nonempty, unique and safe.")
    cap = target.get("max_windows_per_record")
    if cap is not None and (isinstance(cap, bool) or not isinstance(cap, int) or cap < 1):
        raise ValueError("Evaluation window caps must be positive integers.")
    if purpose == "confirmatory" and (cap is not None or dataset == "isruc" and (target["expected_participants"] != 100 or target["expected_recordings"] != 100)):
        raise ValueError("Confirmatory ISRUC evaluation requires all 100 participants/records without window caps.")
    if dataset == "isruc" and (isinstance(target.get("scorer"), bool) or target.get("scorer") not in {1, 2} or target.get("electrode_policy") not in {"C3", "F3"}):
        raise ValueError("ISRUC evaluation must explicitly freeze scorer and electrode policy.")
    radius = config.get("transition_radius_epochs", 1)
    if isinstance(radius, bool) or not isinstance(radius, int) or radius < 0:
        raise ValueError("Transition radius must be a nonnegative integer.")
    pairs = config.get("paired_conditions", [])
    if any(not isinstance(p, (list, tuple)) or len(p) != 2 or p[0] == p[1] for p in pairs):
        raise ValueError("Paired conditions must contain distinct names in pairs.")
    return {"purpose": purpose, "target": target, "analysis": asdict(AnalysisConfig(**config.get("analysis", {}))),
            "paired_conditions": pairs, "transition_radius_epochs": radius}


def registry_from_study(study_directory, output_path, budget_id, conditions=None):
    study, _ = load_study(study_directory)
    conditions = conditions or ["M" + str(i) for i in range(7)]
    rows = []
    for job in study["jobs"]:
        if job["kind"] != "supervised" or job["budget_id"] != budget_id or job["condition"] not in conditions:
            continue
        attempt = completed_attempt(study_directory, study, job["job_id"])
        if attempt is None:
            raise ValueError("Every selected evaluation model must have a verified completed study job.")
        rows.append({"model_id": job["job_id"], "condition": job["condition"], "training_seed": job["seed"],
                     "bundle_directory": str((attempt / "artifacts").resolve()),
                     "predictions": str((attempt / "artifacts/validation_predictions.csv").resolve()),
                     "selection_basis": "source_validation_macro_f1_checkpoint_of_predeclared_condition"})
    if not rows or {r["condition"] for r in rows} != set(conditions):
        raise ValueError("Selected budget/conditions do not have all requested completed models.")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="") as stream:
        pd.DataFrame(rows).to_csv(stream, index=False)
    return rows


def freeze_evaluation(registry_path, configuration, directory):
    """No target path or target-data reader is accepted by this operation."""
    protocol = normalize_protocol(configuration)
    registry_path = Path(registry_path)
    registry = pd.read_csv(registry_path, dtype={"model_id": str, "condition": str})
    required = {"model_id", "condition", "training_seed", "bundle_directory", "selection_basis"}
    if registry.empty or required - set(registry) or registry[list(required)].isna().any().any():
        raise ValueError("A frozen registry requires named models, condition, seed, bundle directory and source selection basis.")
    if registry.model_id.duplicated().any() or registry.duplicated(["condition", "training_seed"]).any():
        raise ValueError("Model IDs and condition/seed pairs must be unique.")
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "status.json", {"status": "freezing"})
    models = []
    common = None
    condition_signatures = {}
    try:
        for row in registry.itertuples(index=False):
            model_id, condition = safe_name(row.model_id), safe_name(row.condition)
            original = Path(row.bundle_directory)
            original = original if original.is_absolute() else registry_path.parent / original
            inspected = inspect_bundle(original)
            seed = row.training_seed
            if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed != inspected["resolved"]["training"]["seed"]:
                raise ValueError("Frozen registry seed disagrees with the source model.")
            if protocol["purpose"] == "confirmatory" and inspected["resolved"].get("limited_data_run", True):
                raise ValueError("Confirmatory evaluation cannot freeze limited-data integration models.")
            if not str(row.selection_basis).strip():
                raise ValueError("Each model needs an explicit source-only selection basis.")
            validation_path = original / "validation_predictions.csv"
            validation_sha256 = sha256_file(validation_path)
            validation = pd.read_csv(validation_path, dtype={c: str for c in ["dataset", "participant_id", "recording_id", "epoch_id"]})
            verify_validation_frame(validation, inspected)
            validate_predictions(validation)
            data = inspected["data"]
            identity = (inspected["config"]["sequence_length"], inspected["bundle"]["preprocessing_fingerprint"],
                        inspected["normalizer"].state["training_data"]["split_manifest"]["sha256"], target_identity(validation))
            if common is None:
                common = identity
            elif common != identity:
                raise ValueError("Frozen models require matching context, preprocessing, frozen source split and validation targets.")
            training_settings = {k: v for k, v in inspected["resolved"]["training"].items() if k != "seed"}
            signature = fingerprint({"model": inspected["config"], "training": training_settings,
                                     "source_records": [{k: r[k] for k in ["recording_id", "participant_id", "manifest_sha256"]}
                                                        for r in data["train"]["recordings"]],
                                     "spectral_mean": inspected["normalizer"].state["spectral_mean"],
                                     "spectral_scale": inspected["normalizer"].state["spectral_scale"],
                                     "waveform_scale": inspected["normalizer"].state["waveform_scale"]})
            if condition in condition_signatures and condition_signatures[condition] != signature:
                raise ValueError("Seeds grouped into one condition must share architecture, training policy, labeled scope and normalization.")
            condition_signatures[condition] = signature
            snapshot = output / "models" / model_id
            snapshot.mkdir(parents=True)
            for name in sorted(BUNDLE_FILES | {"bundle.json", "validation_predictions.csv"}):
                shutil.copyfile(original / name, snapshot / name)
            inspect_bundle(snapshot)
            if sha256_file(snapshot / "validation_predictions.csv") != validation_sha256:
                raise ValueError("Source validation evidence changed while freezing.")
            models.append({"model_id": model_id, "condition": condition, "training_seed": int(seed),
                           "selection_basis": row.selection_basis, "original_directory": str(original.resolve()),
                           "limited_training_run": inspected["resolved"].get("limited_data_run", True),
                           "source_validation": summarize_frame(validation, protocol["analysis"]["calibration_bins"]),
                           "snapshot": snapshot.relative_to(output).as_posix()})
        for first, second in protocol["paired_conditions"]:
            left = {m["training_seed"] for m in models if m["condition"] == first}
            right = {m["training_seed"] for m in models if m["condition"] == second}
            if not left or left != right:
                raise ValueError("Frozen paired contrasts require existing conditions with matching seed sets.")
        frozen = {"schema_version": 1, "completed": True, "protocol": protocol, "models": models,
                  "sequence_length": common[0], "preprocessing_fingerprint": common[1], "source_split_sha256": common[2],
                  "source_validation_target_sha256": common[3], "classes": CLASS_LABELS, "features": ALL_FEATURES,
                  "eligibility_policy": "all_context_epochs_supervised_valid_no_gaps_same_targets_for_every_model",
                  "normalization_policy": "saved_source_statistics_only_no_target_fitting_or_updates",
                  "source_test_history": "historically_inspected_source_test_not_a_fresh_unseen_holdout",
                  "code_sha256": code_hashes(), "packages": {p: importlib.metadata.version(p) for p in PACKAGES},
                  "files": {p.relative_to(output).as_posix(): sha256_file(p) for p in sorted((output / "models").rglob("*")) if p.is_file()}}
        frozen["fingerprint"] = fingerprint(frozen)
        atomic_json(output / "freeze.json", frozen)
        write_json(output / "status.json", {"status": "frozen"})
        return frozen
    except Exception as exc:
        write_json(output / "status.json", {"status": "failed", "error": type(exc).__name__, "message": str(exc)})
        raise


def load_freeze(directory, *, require_runtime=False):
    directory = Path(directory).resolve()
    frozen = read_json(directory / "freeze.json")
    if (frozen.get("schema_version") != 1 or frozen.get("completed") is not True
            or fingerprint({k: v for k, v in frozen.items() if k != "fingerprint"}) != frozen.get("fingerprint")):
        raise ValueError("Evaluation freeze integrity check failed.")
    for name, expected in frozen["files"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or sha256_file(path) != expected:
            raise ValueError(f"Frozen model snapshot integrity check failed: {name}")
    if require_runtime and (frozen["code_sha256"] != code_hashes() or frozen["packages"] != {p: importlib.metadata.version(p) for p in PACKAGES}):
        raise ValueError("Evaluation implementation/packages changed after freezing.")
    return frozen


def selected_positions(store, frozen):
    positions = store.sequence_positions(frozen["sequence_length"])
    cap = frozen["protocol"]["target"].get("max_windows_per_record")
    if cap is not None and len(positions) > cap:
        positions = positions[np.linspace(0, len(positions) - 1, cap, dtype=np.int64)]
    return positions


def inventory_cohort(frozen, root, split_file=None):
    target = frozen["protocol"]["target"]
    assignments = None
    if target["dataset"] == "sleepedf":
        split_file = Path(split_file or default_split())
        if sha256_file(split_file) != frozen["source_split_sha256"]:
            raise ValueError("Evaluation source split disagrees with the frozen model selection.")
        assignments = participant_assignments(split_file)
    selected = set(target["recordings"]) if target.get("recordings") is not None else None
    records = []
    for path in sorted((Path(root) / target["dataset"]).glob("*/manifest.json")):
        if selected is not None and path.parent.name not in selected:
            continue
        head = pd.read_csv(path.parent / "epochs.csv", nrows=1,
                           usecols=["dataset", "participant_id", "recording_id", "split"], dtype=str)
        if len(head) != 1:
            raise ValueError("Evaluation store has no identity metadata.")
        row = head.iloc[0]
        if row.split != target["split"]:
            if selected is not None:
                raise ValueError("Explicit evaluation record is in the wrong partition.")
            continue
        if row.dataset != target["dataset"] or row.recording_id != path.parent.name:
            raise ValueError("Evaluation dataset/recording identity mismatch.")
        if target["dataset"] == "isruc" and row.participant_id != row.recording_id:
            raise ValueError("ISRUC evaluation requires canonical participant/recording identity.")
        if assignments is not None and (assignments.get(row.participant_id) != target["split"] or row.participant_id != row.recording_id[:5]):
            raise ValueError("Evaluation participant disagrees with the frozen source split.")
        with EpochStore(path.parent) as store:
            if store.config.fingerprint != frozen["preprocessing_fingerprint"]:
                raise ValueError("Evaluation preprocessing differs from the frozen model family.")
            if target["dataset"] == "isruc":
                for key in ["scorer", "electrode_policy"]:
                    if store.manifest["provenance"].get(key) != target[key]:
                        raise ValueError("ISRUC scorer/channel policy differs from the frozen declaration.")
                if store.manifest["provenance"].get("annotation_policy") != "explicit-workbook-epochs":
                    raise ValueError("ISRUC annotation policy is not the declared workbook authority.")
            positions = selected_positions(store, frozen)
            if not len(positions):
                raise ValueError("An enrolled evaluation recording has no eligible windows; it cannot silently disappear.")
            frame = store.epochs
            ids = frame.epoch_id.to_numpy()[positions[:, -1]].tolist()
            records.append({"recording_id": row.recording_id, "participant_id": row.participant_id,
                            "manifest_sha256": sha256_file(path), "physical_epochs": len(frame),
                            "quality_epochs": int(frame.quality_valid.sum()), "annotated_epochs": int(frame.label_valid.sum()),
                            "supervised_epochs": int(frame.supervised_valid.sum()),
                            "eligible_windows_uncapped": len(store.sequence_positions(frozen["sequence_length"])),
                            "selected_windows": len(positions), "target_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest()})
    if selected is not None and {r["recording_id"] for r in records} != selected:
        raise ValueError("Explicit evaluation recordings were not all found.")
    if len(records) != target["expected_recordings"] or len({r["participant_id"] for r in records}) != target["expected_participants"]:
        raise ValueError("Evaluation cohort is incomplete or differs from its declared enrollment counts.")
    if target["dataset"] == "isruc" and frozen["protocol"]["purpose"] == "confirmatory" and {r["participant_id"] for r in records} != {str(i) for i in range(1, 101)}:
        raise ValueError("Confirmatory ISRUC enrollment must contain canonical participants 1 through 100.")
    return records


def transition_mask(frame, radius):
    labels = frame.target_stage.to_numpy()
    known = frame.label_valid.to_numpy()
    boundaries = np.flatnonzero(known[1:] & known[:-1] & (labels[1:] != labels[:-1])) + 1
    mask = np.zeros(len(frame), bool)
    for offset in range(-radius, radius + 1):
        indices = boundaries + offset
        mask[indices[(indices >= 0) & (indices < len(frame))]] = True
    return mask


def predict_frozen_model(freeze_directory, cohort_path, root, model_id, output_path, batch_size=4):
    from src.inference.multiview import MultiviewPredictor
    frozen = load_freeze(freeze_directory, require_runtime=True)
    cohort = read_json(cohort_path)
    if cohort["freeze_fingerprint"] != frozen["fingerprint"]:
        raise ValueError("Evaluation cohort belongs to a different model freeze.")
    model = next((m for m in frozen["models"] if m["model_id"] == model_id), None)
    if model is None or isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("Invalid frozen model ID or inference batch size.")
    predictor = MultiviewPredictor(Path(freeze_directory) / model["snapshot"])
    import csv
    columns = ["run_id", "condition", "training_seed", "dataset", "participant_id", "recording_id", "epoch_index", "epoch_id",
               "start_time", "scorer", "eligible", "near_transition", "true_stage", "predicted_stage", *["prob_" + s for s in CLASS_LABELS]]
    with Path(output_path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for record in cohort["records"]:
            path = Path(root) / frozen["protocol"]["target"]["dataset"] / record["recording_id"]
            if sha256_file(path / "manifest.json") != record["manifest_sha256"]:
                raise ValueError("Evaluation recording changed after cohort enrollment.")
            with EpochStore(path) as store:
                positions = selected_positions(store, frozen)
                ids = store.epochs.epoch_id.to_numpy()[positions[:, -1]].tolist()
                if hashlib.sha256("\n".join(ids).encode()).hexdigest() != record["target_ids_sha256"]:
                    raise ValueError("Frozen evaluation targets changed between models.")
                spectral = store.epochs[ALL_FEATURES].to_numpy(dtype=np.float64)
                transition = transition_mask(store.epochs, frozen["protocol"]["transition_radius_epochs"])
                for start in range(0, len(positions), batch_size):
                    rows = positions[start:start + batch_size]
                    raw = np.asarray(store.waveforms[rows])[..., None] if predictor.config.variant != "spectral" else None
                    _, probabilities = predictor.predict_batch(raw, spectral[rows] if predictor.config.variant != "raw" else None)
                    for epoch, probability in zip(rows[:, -1], probabilities):
                        item = store.epochs.iloc[int(epoch)]
                        writer.writerow({"run_id": model_id, "condition": model["condition"], "training_seed": model["training_seed"],
                                         "dataset": item.dataset, "participant_id": item.participant_id, "recording_id": item.recording_id,
                                         "epoch_index": int(epoch), "epoch_id": item.epoch_id, "start_time": item.start_time,
                                         "scorer": frozen["protocol"]["target"].get("scorer", "source_hypnogram"), "eligible": True,
                                         "near_transition": bool(transition[epoch]), "true_stage": item.target_stage,
                                         "predicted_stage": CLASS_LABELS[int(probability.argmax())],
                                         **{"prob_" + s: float(p) for s, p in zip(CLASS_LABELS, probability)}})


def score_frozen(freeze_directory, root, output_directory, *, split_file=None, batch_size=4, isolated_workers=True):
    frozen = load_freeze(freeze_directory, require_runtime=True)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("Evaluation batch size must be a positive integer.")
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "status.json", {"status": "running"})
    try:
        cohort = {"freeze_fingerprint": frozen["fingerprint"], "records": inventory_cohort(frozen, root, split_file)}
        cohort_path = output / "cohort.json"
        atomic_json(cohort_path, cohort)
        models = []
        for model in frozen["models"]:
            destination = output / (model["model_id"] + "_predictions.csv")
            if isolated_workers:
                with (output / (model["model_id"] + ".log")).open("w", encoding="utf-8") as log:
                    result = subprocess.run([sys.executable, "-u", "-X", "utf8", "-m", "src.evaluation.frozen", "_predict",
                                             "--freeze-dir", str(Path(freeze_directory).resolve()), "--cohort", str(cohort_path),
                                             "--data-root", str(Path(root).resolve()), "--model-id", model["model_id"],
                                             "--output", str(destination), "--batch-size", str(batch_size)], cwd=ROOT,
                                            stdout=log, stderr=subprocess.STDOUT)
                if result.returncode:
                    raise RuntimeError("Frozen scoring worker failed; no complete family report was published.")
            else:
                predict_frozen_model(freeze_directory, cohort_path, root, model["model_id"], destination, batch_size)
            frame = pd.read_csv(destination, dtype={c: str for c in ["dataset", "participant_id", "recording_id", "epoch_id"]})
            validate_predictions(frame)
            if set(frame.recording_id) != {r["recording_id"] for r in cohort["records"]}:
                raise ValueError("A frozen model did not produce the complete enrolled recording set.")
            for record in cohort["records"]:
                selected = frame.loc[frame.recording_id == record["recording_id"]].sort_values("epoch_index")
                if (len(selected) != record["selected_windows"] or not selected.participant_id.eq(record["participant_id"]).all()
                        or hashlib.sha256("\n".join(selected.epoch_id).encode()).hexdigest() != record["target_ids_sha256"]):
                    raise ValueError("A frozen model did not produce the complete declared target mask.")
            models.append({**model, "predictions": frame})
        load_freeze(freeze_directory, require_runtime=True)
        analysis = analyze_models(models, frozen["protocol"]["paired_conditions"], AnalysisConfig(**frozen["protocol"]["analysis"]))
        analysis.update({"freeze_fingerprint": frozen["fingerprint"], "evaluation_purpose": frozen["protocol"]["purpose"],
                         "cohort_policy": frozen["protocol"]["target"], "source_test_history": frozen["source_test_history"]})
        write_json(output / "analysis.json", analysis)
        pd.DataFrame(analysis["participant_metrics"]).to_csv(output / "participant_metrics.csv", index=False)
        pd.DataFrame(analysis["paired_comparisons"]).to_csv(output / "paired_comparisons.csv", index=False)
        files = {p.name: sha256_file(p) for p in sorted(output.iterdir()) if p.is_file() and p.name not in {"status.json"} and p.suffix != ".log"}
        atomic_json(output / "evaluation_bundle.json", {"schema_version": 1, "completed": True,
                                                       "freeze_fingerprint": frozen["fingerprint"], "files": files})
        write_json(output / "status.json", {"status": "completed"})
        return analysis
    except Exception as exc:
        write_json(output / "status.json", {"status": "failed", "error": type(exc).__name__, "message": str(exc)})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    registry = sub.add_parser("registry")
    registry.add_argument("--study-dir", type=Path, required=True)
    registry.add_argument("--budget-id", required=True)
    registry.add_argument("--conditions", nargs="+")
    registry.add_argument("--output", type=Path, required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--registry", type=Path, required=True)
    freeze.add_argument("--config", type=Path, required=True)
    freeze.add_argument("--freeze-dir", type=Path, required=True)
    score = sub.add_parser("score")
    score.add_argument("--freeze-dir", type=Path, required=True)
    score.add_argument("--data-root", type=Path, default=ROOT / "data/processed/research_v1")
    score.add_argument("--split-file", type=Path)
    score.add_argument("--output-dir", type=Path, required=True)
    score.add_argument("--batch-size", type=int, default=4)
    worker = sub.add_parser("_predict")
    for flag in ["freeze-dir", "cohort", "data-root", "output"]:
        worker.add_argument("--" + flag, type=Path, required=True)
    worker.add_argument("--model-id", required=True)
    worker.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args(argv)
    if args.command == "registry":
        result = registry_from_study(args.study_dir, args.output, args.budget_id, args.conditions)
        print(json.dumps({"registered_models": len(result)}, indent=2))
    elif args.command == "freeze":
        result = freeze_evaluation(args.registry, read_json(args.config), args.freeze_dir)
        print(json.dumps({"fingerprint": result["fingerprint"], "models": len(result["models"]), "purpose": result["protocol"]["purpose"]}, indent=2))
    elif args.command == "score":
        result = score_frozen(args.freeze_dir, args.data_root, args.output_dir, split_file=args.split_file, batch_size=args.batch_size)
        print(json.dumps({"participants": result["participants"], "conditions": result["conditions"]}, indent=2))
    else:
        predict_frozen_model(args.freeze_dir, args.cohort, args.data_root, args.model_id, args.output, args.batch_size)


if __name__ == "__main__":
    main()

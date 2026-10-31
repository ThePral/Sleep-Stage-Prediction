"""Plan and execute paired source-only comparisons in isolated processes."""

import argparse
from dataclasses import asdict
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd

from src.data.epoch_store import sha256_file
from src.data.label_budgets import (build_budget_plan, default_split, fingerprint, load_budget_plan, verify_plan_scope)
from src.data.prepare_research_data import ROOT
from src.training.artifacts import write_json


CONDITIONS = {
    "M0": ("spectral", False), "M1": ("raw", False), "M2": ("concat", False), "M3": ("gated", False),
    "M4": ("raw", True), "M5": ("concat", True), "M6": ("gated", True),
}
PACKAGES = ["tensorflow", "keras", "numpy", "pandas", "scikit-learn", "mne"]


def atomic_json(path, value):
    path = Path(path)
    staging = path.with_suffix(".tmp")
    write_json(staging, value)
    os.replace(staging, path)


def code_hashes():
    return {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in sorted((ROOT / "src").rglob("*.py"))}


def normalize_config(config):
    from src.models.contrastive import AugmentationConfig
    from src.models.multiview import ModelConfig
    from src.training.contrastive import PretrainingConfig
    from src.training.supervised import TrainingConfig
    allowed = {"model", "training", "pretraining", "augmentation", "fractions", "draw_seeds", "training_seeds",
               "conditions", "train_participants", "validation_participants", "limits"}
    if set(config) - allowed:
        raise ValueError("Unknown experiment configuration keys.")
    if "variant" in config.get("model", {}) or "seed" in config.get("training", {}) or "seed" in config.get("pretraining", {}):
        raise ValueError("Condition determines variant; training_seeds determines optimization/pretraining seeds.")
    seeds = config.get("training_seeds", [42])
    if not seeds or len(set(seeds)) != len(seeds) or any(isinstance(s, bool) or not isinstance(s, int) or not 0 <= s < 2**31 for s in seeds):
        raise ValueError("training_seeds must be unique integers in [0, 2**31).")
    conditions = config.get("conditions", list(CONDITIONS))
    if not conditions or len(set(conditions)) != len(conditions) or set(conditions) - set(CONDITIONS):
        raise ValueError("Conditions must be unique IDs from M0 through M6.")
    model = asdict(ModelConfig(**config.get("model", {})))
    model.pop("variant")
    training = asdict(TrainingConfig(**config.get("training", {})))
    pretraining = asdict(PretrainingConfig(**config.get("pretraining", {})))
    training.pop("seed")
    pretraining.pop("seed")
    limits = {key: None for key in ["max_train_batches", "max_validation_batches", "max_pretraining_batches"]}
    if set(config.get("limits", {})) - set(limits):
        raise ValueError("Unknown experiment limit.")
    limits.update(config.get("limits", {}))
    if any(v is not None and (isinstance(v, bool) or not isinstance(v, int) or v < 1) for v in limits.values()):
        raise ValueError("Batch caps must be positive integers or null.")
    value = {"model": model, "training": training, "pretraining": pretraining,
             "augmentation": asdict(AugmentationConfig(**config.get("augmentation", {}))),
             "fractions": config.get("fractions", [0.25, 1.]), "draw_seeds": config.get("draw_seeds", [42]),
             "training_seeds": seeds, "conditions": conditions, "limits": limits,
             "train_participants": config.get("train_participants"), "validation_participants": config.get("validation_participants")}
    return json.loads(json.dumps(value))


def make_jobs(config, budgets):
    jobs = [{"job_id": "waveform_statistics", "kind": "waveform_statistics", "dependencies": []}]
    for budget in budgets:
        jobs.append({"job_id": f"normalizer_{budget['budget_id']}", "kind": "normalizer", "budget_id": budget["budget_id"],
                     "dependencies": ["waveform_statistics"]})
    if any(CONDITIONS[c][1] for c in config["conditions"]):
        for seed in config["training_seeds"]:
            jobs.append({"job_id": f"ssl_seed{seed}", "kind": "pretrain", "seed": seed, "dependencies": ["waveform_statistics"]})
    for budget in budgets:
        for seed in config["training_seeds"]:
            for condition in config["conditions"]:
                variant, pretrained = CONDITIONS[condition]
                dependencies = [f"normalizer_{budget['budget_id']}"]
                if pretrained:
                    dependencies.append(f"ssl_seed{seed}")
                jobs.append({"job_id": f"{condition}_{budget['budget_id']}_seed{seed}", "kind": "supervised",
                             "budget_id": budget["budget_id"], "seed": seed, "condition": condition,
                             "variant": variant, "pretrained": pretrained, "dependencies": dependencies})
    return jobs


def create_study(root, directory, config, *, split_file=None):
    config = normalize_config(config)
    plan = build_budget_plan(root, fractions=config["fractions"], draw_seeds=config["draw_seeds"],
                             sequence_length=config["model"]["sequence_length"], split_file=split_file,
                             train_participants=config["train_participants"], validation_participants=config["validation_participants"])
    study = {"schema_version": 1, "config": config, "budget_fingerprint": plan["fingerprint"],
             "jobs": make_jobs(config, plan["budgets"]), "data_root": str(Path(root).resolve()),
             "split_file": str(Path(split_file or default_split()).resolve()), "code_sha256": code_hashes(),
             "packages": {p: importlib.metadata.version(p) for p in PACKAGES},
             "limited_run": plan["limited_participant_scope"] or any(v is not None for v in config["limits"].values()),
             "selection_policy": "source_validation_only_no_source_test_or_external_prediction_jobs",
             "comparison_policy": "paired_budget_seed_normalizer_validation_targets_and_ssl_checkpoint",
             "M0_policy": "fresh_two_layer_32_unit_spectral_LSTM_by_default_not_the_historical_frozen_checkpoint"}
    study["fingerprint"] = fingerprint(study)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "label_budgets.json", plan)
    write_json(directory / "study.json", study)
    atomic_json(directory / "ledger.json", {"schema_version": 1, "study_fingerprint": study["fingerprint"], "jobs": {}})
    return study


def load_study(directory):
    directory = Path(directory)
    study = json.loads((directory / "study.json").read_text(encoding="utf-8"))
    if study.get("schema_version") != 1 or fingerprint({k: v for k, v in study.items() if k != "fingerprint"}) != study.get("fingerprint"):
        raise ValueError("Study manifest integrity check failed.")
    plan = load_budget_plan(directory / "label_budgets.json")
    if plan["fingerprint"] != study["budget_fingerprint"] or study["jobs"] != make_jobs(study["config"], plan["budgets"]):
        raise ValueError("Study jobs disagree with the frozen label budget manifest.")
    return study, plan


def verify_runtime(study, plan, root, split_file):
    if study["code_sha256"] != code_hashes():
        raise ValueError("Implementation changed after study planning; create a new study.")
    if study["packages"] != {p: importlib.metadata.version(p) for p in PACKAGES}:
        raise ValueError("Package versions changed after study planning.")
    verify_plan_scope(plan, root, split_file)


def verify_job_result(directory, study, job_id):
    directory = Path(directory).resolve()
    result = json.loads((directory / "job_result.json").read_text(encoding="utf-8"))
    if (result.get("schema_version") != 1 or result.get("completed") is not True
            or result.get("study_fingerprint") != study["fingerprint"] or result.get("job_id") != job_id or not result.get("files")):
        raise ValueError("Job result does not match this study/job.")
    for name, digest in result["files"].items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or sha256_file(path) != digest:
            raise ValueError(f"Completed job artifact integrity check failed: {job_id}/{name}")
    return result


def completed_attempt(directory, study, job_id):
    candidates = sorted((Path(directory) / "jobs" / job_id).glob("attempt_*"), reverse=True)
    for candidate in candidates:
        if candidate.is_dir() and (candidate / "job_result.json").exists():
            verify_job_result(candidate, study, job_id)
            return candidate
    return None


def write_tables(directory, study, ledger):
    rows = []
    for job in study["jobs"]:
        item = ledger["jobs"].get(job["job_id"], {})
        if job["kind"] != "supervised" or item.get("status") != "completed":
            continue
        attempt = Path(directory) / item["attempt"]
        summary = json.loads((attempt / "artifacts/summary.json").read_text(encoding="utf-8"))
        predictions = pd.read_csv(attempt / "artifacts/validation_predictions.csv", usecols=["epoch_id", "true_stage"])
        budget = summary["label_budget"]
        rows.append({"job_id": job["job_id"], "condition": job["condition"], "budget_id": job["budget_id"],
                     "training_seed": job["seed"], "draw_seed": budget["draw_seed"], "fraction": budget["fraction"],
                     "labeled_training_participants": len(budget["participants"]),
                     "training_annotated_hours": budget["annotation_cost"]["annotated_hours"],
                     "validation_annotated_hours": summary["validation_annotation_cost"]["annotated_hours"],
                     "validation_examples": len(predictions), "validation_macro_f1": summary["source_validation"]["macro_f1"],
                     "validation_accuracy": summary["source_validation"]["accuracy"], "parameters": summary["parameters"],
                     "epochs_completed": summary["epochs_completed"], "best_epoch": summary["best_epoch"],
                     "normalizer_sha256": sha256_file(attempt / "artifacts/normalizer.json"),
                     "validation_targets_sha256": fingerprint(predictions.to_dict("records")),
                     "limited_run": study["limited_run"]})
    columns = ["job_id", "condition", "budget_id", "training_seed", "draw_seed", "fraction", "labeled_training_participants",
               "training_annotated_hours", "validation_annotated_hours", "validation_examples", "validation_macro_f1",
               "validation_accuracy", "parameters", "epochs_completed", "best_epoch", "normalizer_sha256", "validation_targets_sha256", "limited_run"]
    frame = pd.DataFrame(rows, columns=columns)
    frame.to_csv(Path(directory) / "source_validation_results.csv", index=False)
    comparisons = []
    for (budget, seed), group in frame.groupby(["budget_id", "training_seed"]):
        if group.normalizer_sha256.nunique() != 1 or group.validation_targets_sha256.nunique() != 1:
            raise ValueError("Paired comparisons have different normalization or validation targets.")
        lookup = {r["condition"]: r for r in group.to_dict("records")}
        for first, second in [("M4", "M1"), ("M5", "M2"), ("M6", "M3"), ("M3", "M2"), ("M6", "M5")]:
            if first in lookup and second in lookup:
                comparisons.append({"budget_id": budget, "training_seed": seed, "first": first, "second": second,
                                    "source_validation_macro_f1_difference": lookup[first]["validation_macro_f1"] - lookup[second]["validation_macro_f1"],
                                    "limited_run": study["limited_run"]})
    pd.DataFrame(comparisons, columns=["budget_id", "training_seed", "first", "second", "source_validation_macro_f1_difference", "limited_run"]).to_csv(
        Path(directory) / "paired_source_validation.csv", index=False)


def execute_study(directory, *, root=None, split_file=None, max_jobs=None, retry_failed=False, executor=None):
    directory = Path(directory).resolve()
    study, plan = load_study(directory)
    root = Path(root or study["data_root"]).resolve()
    split_file = Path(split_file or study["split_file"]).resolve()
    if max_jobs is not None and (isinstance(max_jobs, bool) or not isinstance(max_jobs, int) or max_jobs < 1):
        raise ValueError("max_jobs must be a positive integer.")
    verify_runtime(study, plan, root, split_file)
    lock = directory / ".runner.lock"
    # Exclusive creation prevents two serial schedulers from sharing a study.
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    try:
        ledger = json.loads((directory / "ledger.json").read_text(encoding="utf-8"))
        if ledger.get("study_fingerprint") != study["fingerprint"]:
            raise ValueError("Ledger belongs to a different study.")
        attempted = 0
        for job in study["jobs"]:
            job_id = job["job_id"]
            completed = completed_attempt(directory, study, job_id)
            if completed:
                ledger["jobs"][job_id] = {"status": "completed", "attempt": completed.relative_to(directory).as_posix()}
                continue
            if max_jobs is not None and attempted >= max_jobs:
                break
            prior = ledger["jobs"].get(job_id, {})
            if prior.get("status") in {"failed", "running"} and not retry_failed:
                raise ValueError(f"Job {job_id} needs --retry-failed to preserve and replace its interrupted/failed attempt.")
            dependencies = {}
            for dependency in job["dependencies"]:
                found = completed_attempt(directory, study, dependency)
                if found is None:
                    raise ValueError(f"Job dependency is incomplete: {dependency}")
                dependencies[dependency] = str(found / "artifacts")
            job_directory = directory / "jobs" / job_id
            job_directory.mkdir(parents=True, exist_ok=True)
            attempt_number = 1
            while (job_directory / f"attempt_{attempt_number:04d}").exists():
                attempt_number += 1
            attempt = job_directory / f"attempt_{attempt_number:04d}"
            attempt.mkdir()
            write_json(attempt / "task.json", {"study_directory": str(directory), "job_id": job_id,
                                               "data_root": str(root), "split_file": str(split_file), "dependencies": dependencies})
            ledger["jobs"][job_id] = {"status": "running", "attempt": attempt.relative_to(directory).as_posix()}
            atomic_json(directory / "ledger.json", ledger)
            print(f"Running {job_id} ({attempt.name})", flush=True)
            try:
                if executor is None:
                    with attempt.with_suffix(".log").open("w", encoding="utf-8") as log:
                        result = subprocess.run([sys.executable, "-u", "-X", "utf8", "-m", "src.experiments.worker",
                                                 "--task", str(attempt / "task.json")], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                    if result.returncode:
                        raise RuntimeError(f"Job {job_id} exited {result.returncode}; see {attempt.with_suffix('.log')}")
                else:
                    executor(attempt / "task.json")
                verify_job_result(attempt, study, job_id)
            except BaseException as exc:
                ledger["jobs"][job_id].update({"status": "failed", "error": type(exc).__name__, "message": str(exc)})
                atomic_json(directory / "ledger.json", ledger)
                raise
            ledger["jobs"][job_id]["status"] = "completed"
            atomic_json(directory / "ledger.json", ledger)
            attempted += 1
        atomic_json(directory / "ledger.json", ledger)
        write_tables(directory, study, ledger)
        summary = {"jobs_planned": len(study["jobs"]), "jobs_completed": sum(v["status"] == "completed" for v in ledger["jobs"].values()),
                   "jobs_executed_this_call": attempted, "limited_run": study["limited_run"]}
        atomic_json(directory / "execution_summary.json", summary)
        return summary
    finally:
        lock.unlink()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--config", required=True, type=Path)
    plan.add_argument("--study-dir", required=True, type=Path)
    plan.add_argument("--data-root", type=Path, default=ROOT / "data/processed/research_v1")
    plan.add_argument("--split-file", type=Path)
    run = sub.add_parser("execute")
    run.add_argument("--study-dir", required=True, type=Path)
    run.add_argument("--data-root", type=Path)
    run.add_argument("--split-file", type=Path)
    run.add_argument("--max-jobs", type=int)
    run.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "plan":
        study = create_study(args.data_root, args.study_dir, json.loads(args.config.read_text(encoding="utf-8")), split_file=args.split_file)
        print(json.dumps({"jobs_planned": len(study["jobs"]), "limited_run": study["limited_run"], "study_fingerprint": study["fingerprint"]}, indent=2))
    else:
        print(json.dumps(execute_study(args.study_dir, root=args.data_root, split_file=args.split_file,
                                       max_jobs=args.max_jobs, retry_failed=args.retry_failed), indent=2))


if __name__ == "__main__":
    main()

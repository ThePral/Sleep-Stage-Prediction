"""Execute exactly one source-only study job; process exit releases TF memory."""

import argparse
import json
from pathlib import Path

from src.data.contrastive import SourceEpochPool
from src.data.epoch_store import sha256_file
from src.data.label_budgets import authorize_waveform_pool, get_budget, verify_plan_scope
from src.data.supervised import SourcePartition
from src.experiments.run import atomic_json, load_study, verify_job_result
from src.models.contrastive import AugmentationConfig
from src.models.multiview import ModelConfig
from src.preprocessing.research_normalization import ResearchNormalizer
from src.preprocessing.waveform_normalization import WaveformNormalizer
from src.training.artifacts import write_json
from src.training.contrastive import PretrainingConfig, train_contrastive
from src.training.supervised import TrainingConfig, train_supervised


def run_task(task_path):
    task_path = Path(task_path).resolve()
    task = json.loads(task_path.read_text(encoding="utf-8"))
    study_dir = Path(task["study_directory"])
    study, plan = load_study(study_dir)
    matches = [j for j in study["jobs"] if j["job_id"] == task["job_id"]]
    if len(matches) != 1:
        raise ValueError("Unknown study job.")
    job = matches[0]
    if set(task["dependencies"]) != set(job["dependencies"]):
        raise ValueError("Task dependencies disagree with the planned job.")
    for name, path in task["dependencies"].items():
        verify_job_result(Path(path).parent, study, name)
    root, split = Path(task["data_root"]), Path(task["split_file"])
    verify_plan_scope(plan, root, split)
    config = study["config"]
    output = task_path.parent / "artifacts"
    training_records = [r["recording_id"] for r in plan["source_records"] if r["split"] == "train"]
    if job["kind"] == "waveform_statistics":
        output.mkdir(exist_ok=False)
        with SourceEpochPool(root, recordings=training_records, split_file=split) as pool:
            normalizer = WaveformNormalizer().fit(pool)
            authorize_waveform_pool(plan, normalizer)
            normalizer.save(output / "waveform_normalizer.json")
            write_json(output / "summary.json", {"fit_epochs": normalizer.state["fit_epochs"], "participants": len(plan["training_participants"]),
                                                 "label_use": "none_for_selection_or_fitting", "source_data": pool.manifest()})
    elif job["kind"] == "normalizer":
        output.mkdir(exist_ok=False)
        budget = get_budget(plan, job["budget_id"])
        waveform = WaveformNormalizer.load(Path(task["dependencies"]["waveform_statistics"]) / "waveform_normalizer.json")
        authorize_waveform_pool(plan, waveform)
        with SourcePartition(root, "train", sequence_length=plan["sequence_length"], recordings=budget["recordings"], split_file=split) as partition:
            normalizer = ResearchNormalizer().fit(partition).use_pretraining_waveform(waveform, partition, allow_source_subset=True)
            normalizer.save(output / "normalizer.json")
            write_json(output / "summary.json", {"label_budget": budget, "spectral_fit_epochs": normalizer.state["fit_epochs"],
                                                 "waveform_fit_epochs": waveform.state["fit_epochs"]})
    elif job["kind"] == "pretrain":
        train_contrastive(root, output, ModelConfig(**{**config["model"], "variant": "raw", "sequence_length": 1}),
                          PretrainingConfig(**{**config["pretraining"], "seed": job["seed"]}), AugmentationConfig(**config["augmentation"]),
                          train_recordings=training_records if config["train_participants"] is not None else None, split_file=split,
                          max_batches=config["limits"]["max_pretraining_batches"],
                          normalizer_path=Path(task["dependencies"]["waveform_statistics"]) / "waveform_normalizer.json")
    elif job["kind"] == "supervised":
        pretrained = Path(task["dependencies"][f"ssl_seed{job['seed']}"]) if job["pretrained"] else None
        train_supervised(root, output, ModelConfig(**{**config["model"], "variant": job["variant"]}),
                         TrainingConfig(**{**config["training"], "seed": job["seed"]}), split_file=split,
                         max_train_batches=config["limits"]["max_train_batches"], max_validation_batches=config["limits"]["max_validation_batches"],
                         normalizer_path=Path(task["dependencies"][f"normalizer_{job['budget_id']}"]) / "normalizer.json",
                         pretrained_path=pretrained, budget_plan_path=study_dir / "label_budgets.json", budget_id=job["budget_id"])
    else:
        raise ValueError("Unsupported study job kind.")
    # Seal all durable outputs, including predictions/metrics and optimizer state.
    files = {p.relative_to(task_path.parent).as_posix(): sha256_file(p) for p in sorted(task_path.parent.rglob("*")) if p.is_file()}
    atomic_json(task_path.parent / "job_result.json", {"schema_version": 1, "completed": True,
                                                     "study_fingerprint": study["fingerprint"], "job_id": job["job_id"], "files": files})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True)
    args = parser.parse_args(argv)
    run_task(args.task)


if __name__ == "__main__":
    main()

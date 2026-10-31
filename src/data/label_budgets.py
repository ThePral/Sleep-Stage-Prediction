"""Nested participant label budgets with a fixed unlabeled source pool."""

from dataclasses import asdict
from decimal import Decimal, ROUND_CEILING
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.epoch_store import EpochStore, sha256_file
from src.data.prepare_research_data import ROOT, participant_assignments
from src.preprocessing.sequences import CLASS_LABELS


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def default_split():
    return ROOT / "final_model_corrected/participant_split.csv"


def discover_scope(root, assignments, train_participants, validation_participants):
    """Read identities only; target labels of test/external stores are not read."""
    requested = {"train": set(train_participants), "val": set(validation_participants)}
    records = []
    for manifest in sorted((Path(root) / "sleepedf").glob("*/manifest.json")):
        head = pd.read_csv(manifest.parent / "epochs.csv", nrows=1,
                           usecols=["dataset", "participant_id", "recording_id", "split"], dtype=str)
        if len(head) != 1:
            raise ValueError("An epoch store has no identity row.")
        row = head.iloc[0]
        if (row.dataset != "sleepedf" or row.recording_id != manifest.parent.name
                or row.participant_id != row.recording_id[:5] or assignments.get(row.participant_id) != row.split):
            raise ValueError("Source identity disagrees with the frozen split.")
        if row.split in requested and row.participant_id in requested[row.split]:
            records.append({"recording_id": row.recording_id, "participant_id": row.participant_id,
                            "split": row.split, "manifest_sha256": sha256_file(manifest)})
    for split, participants in requested.items():
        if {r["participant_id"] for r in records if r["split"] == split} != participants:
            raise ValueError(f"Missing source {split} participants; a study scope cannot silently shrink.")
    return records


def annotation_counts(records, epoch_duration):
    return {"participants": len({r["participant_id"] for r in records}), "recordings": len(records),
            **{key: sum(r[key] for r in records) for key in ["physical_epochs", "annotated_epochs", "supervised_epochs", "windows"]},
            "recorded_hours": sum(r["physical_epochs"] for r in records) * epoch_duration / 3600,
            "annotated_hours": sum(r["annotated_epochs"] for r in records) * epoch_duration / 3600,
            "supervised_hours": sum(r["supervised_epochs"] for r in records) * epoch_duration / 3600,
            "target_stage_counts": {s: sum(r["target_stage_counts"][s] for r in records) for s in CLASS_LABELS}}


def make_entries(plan):
    training = sorted(plan["training_participants"])
    fractions = plan["fractions"]
    sizes = [int((Decimal(str(f)) * len(training)).to_integral_value(rounding=ROUND_CEILING)) for f in fractions]
    if len(set(sizes)) != len(sizes):
        raise ValueError("Fractions produce duplicate participant counts; use distinct budgets for this scope.")
    entries = []
    for seed in plan["draw_seeds"]:
        order = np.random.default_rng(seed).permutation(training).tolist()
        for fraction, size in zip(fractions, sizes):
            participants = sorted(order[:size])
            records = [r for r in plan["source_records"] if r["split"] == "train" and r["participant_id"] in participants]
            counts = annotation_counts(records, plan["preprocessing"]["epoch_duration"])
            missing = [s for s in CLASS_LABELS if counts["target_stage_counts"][s] == 0]
            if missing:
                raise ValueError(f"Draw {seed}, budget {size} lacks eligible target classes {missing}; no label-driven redraw is performed.")
            entries.append({"budget_id": f"draw{seed}_n{size}", "draw_seed": seed, "fraction": fraction,
                            "actual_fraction": size / len(training), "participants": participants,
                            "recordings": sorted(r["recording_id"] for r in records), "annotation_cost": counts})
    return entries


def build_budget_plan(root, *, fractions=(0.25, 1.), draw_seeds=(42,), sequence_length=10,
                      split_file=None, train_participants=None, validation_participants=None):
    fractions = list(fractions)
    seeds = list(draw_seeds)
    if (not fractions or any(isinstance(f, bool) or not np.isfinite(f) or not 0 < f <= 1 for f in fractions)
            or fractions != sorted(set(fractions))):
        raise ValueError("Label fractions must be unique, increasing and in (0, 1].")
    if not seeds or len(set(seeds)) != len(seeds) or any(isinstance(s, bool) or not isinstance(s, int) or not 0 <= s < 2**32 for s in seeds):
        raise ValueError("Draw seeds must be unique integers in [0, 2**32).")
    if isinstance(sequence_length, bool) or not isinstance(sequence_length, int) or sequence_length < 1:
        raise ValueError("sequence_length must be a positive integer.")
    split_file = Path(split_file) if split_file else default_split()
    assignments = participant_assignments(split_file)

    def scope(selection, split):
        values = sorted(p for p, s in assignments.items() if s == split) if selection is None else list(selection)
        if not values or len(set(values)) != len(values) or any(assignments.get(p) != split for p in values):
            raise ValueError(f"Study {split} scope must contain unique frozen {split} participants.")
        return sorted(values)

    training, validation = scope(train_participants, "train"), scope(validation_participants, "val")
    records = discover_scope(root, assignments, training, validation)
    preprocessing = None
    for record in records:
        with EpochStore(Path(root) / "sleepedf" / record["recording_id"]) as store:
            if preprocessing is None:
                preprocessing = store.config
            elif store.config != preprocessing:
                raise ValueError("Study source stores mix preprocessing versions.")
            frame = store.epochs
            positions = store.sequence_positions(sequence_length)
            targets = frame.iloc[positions[:, -1]]
            record.update({"physical_epochs": len(frame), "annotated_epochs": int(frame.label_valid.sum()),
                           "supervised_epochs": int(frame.supervised_valid.sum()), "quality_epochs": int(frame.quality_valid.sum()),
                           "windows": len(positions), "target_stage_counts": {s: int(targets.target_stage.eq(s).sum()) for s in CLASS_LABELS},
                           "target_ids_sha256": hashlib.sha256("\n".join(targets.epoch_id).encode()).hexdigest()})
    plan = {"schema_version": 1, "dataset": "sleepedf", "split_sha256": sha256_file(split_file),
            "preprocessing": asdict(preprocessing), "preprocessing_fingerprint": preprocessing.fingerprint,
            "sequence_length": sequence_length, "fractions": fractions, "draw_seeds": seeds,
            "training_participants": training, "validation_participants": validation,
            "limited_participant_scope": train_participants is not None or validation_participants is not None,
            "source_records": records,
            "sampling_policy": "seeded_participant_permutation_nested_prefixes_ceil_no_label_driven_redraw",
            "unlabeled_policy": "all_quality_valid_epochs_from_fixed_training_scope_for_every_budget",
            "validation_policy": "fixed_fully_labeled_validation_scope_training_label_efficiency_only"}
    plan["budgets"] = make_entries(plan)
    plan["validation_annotation_cost"] = annotation_counts([r for r in records if r["split"] == "val"], preprocessing.epoch_duration)
    plan["fingerprint"] = fingerprint(plan)
    return plan


def validate_plan(plan):
    payload = {k: v for k, v in plan.items() if k != "fingerprint"}
    if plan.get("schema_version") != 1 or plan.get("dataset") != "sleepedf" or fingerprint(payload) != plan.get("fingerprint"):
        raise ValueError("Label budget manifest integrity check failed.")
    if not plan["training_participants"] or set(plan["training_participants"]) & set(plan["validation_participants"]):
        raise ValueError("Label budget participant scopes overlap or are empty.")
    if plan["budgets"] != make_entries(plan):
        raise ValueError("Label budgets disagree with their declared nested draws.")
    return plan


def load_budget_plan(path):
    return validate_plan(json.loads(Path(path).read_text(encoding="utf-8")))


def verify_plan_scope(plan, root, split_file=None):
    validate_plan(plan)
    split_file = Path(split_file) if split_file else default_split()
    if sha256_file(split_file) != plan["split_sha256"]:
        raise ValueError("Label budget frozen split has changed.")
    actual = discover_scope(root, participant_assignments(split_file), plan["training_participants"], plan["validation_participants"])
    keys = ["recording_id", "participant_id", "split", "manifest_sha256"]
    if actual != [{k: r[k] for k in keys} for r in plan["source_records"]]:
        raise ValueError("Source recording scope or manifest content changed after budget planning.")


def get_budget(plan, budget_id):
    validate_plan(plan)
    entries = [b for b in plan["budgets"] if b["budget_id"] == budget_id]
    if len(entries) != 1:
        raise ValueError(f"Unknown label budget: {budget_id}")
    return entries[0]


def authorize_waveform_pool(plan, normalizer):
    """Require the complete planned unlabeled pool, not an arbitrary superset."""
    normalizer.validate()
    data = normalizer.state["training_data"]
    keys = ["recording_id", "participant_id", "split", "manifest_sha256"]
    expected = [{k: r[k] for k in keys} for r in plan["source_records"] if r["split"] == "train"]
    actual = [{k: r[k] for k in keys} for r in data["recordings"]]
    if (data["split_manifest"]["sha256"] != plan["split_sha256"] or actual != expected
            or normalizer.state["preprocessing_fingerprint"] != plan["preprocessing_fingerprint"]):
        raise ValueError("Waveform normalization/SSL pool disagrees with the fixed budget training scope.")

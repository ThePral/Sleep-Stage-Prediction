"""Read-only verification of model bundles and their source-validation role."""

import json
from pathlib import Path

from src.data.epoch_store import sha256_file
from src.preprocessing.normalization import ALL_FEATURES
from src.preprocessing.research_normalization import ResearchNormalizer
from src.preprocessing.sequences import CLASS_LABELS


BUNDLE_FILES = {"model.keras", "normalizer.json", "model_config.json", "data_manifest.json", "resolved_config.json"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def inspect_bundle(directory):
    directory = Path(directory)
    bundle = read_json(directory / "bundle.json")
    if (bundle.get("schema_version") != 1 or bundle.get("completed") is not True
            or bundle.get("classes") != CLASS_LABELS or bundle.get("features") != ALL_FEATURES
            or set(bundle.get("files", {})) != BUNDLE_FILES):
        raise ValueError("Evaluation requires a completed supervised model bundle.")
    for name in BUNDLE_FILES:
        if sha256_file(directory / name) != bundle["files"][name]:
            raise ValueError(f"Model bundle integrity check failed: {name}")
    data, resolved, config = [read_json(directory / name) for name in ["data_manifest.json", "resolved_config.json", "model_config.json"]]
    normalizer = ResearchNormalizer.load(directory / "normalizer.json")
    if normalizer.state["preprocessing_fingerprint"] != bundle["preprocessing_fingerprint"]:
        raise ValueError("Evaluation bundle preprocessing disagrees with its normalization.")
    for partition, expected in [("train", "train"), ("validation", "val")]:
        manifest = data[partition]
        if (manifest.get("dataset") != "sleepedf" or manifest.get("split") != expected or not manifest.get("recordings")
                or any(r["split"] != expected for r in manifest["recordings"])):
            raise ValueError("Model fitting/selection must declare source training and validation participants only.")
    train = {r["participant_id"] for r in data["train"]["recordings"]}
    validation = {r["participant_id"] for r in data["validation"]["recordings"]}
    if train & validation or resolved.get("checkpoint_selection") != "maximum_source_validation_fixed_five_class_macro_f1":
        raise ValueError("Model selection is not the declared disjoint source-validation policy.")
    return {"bundle": bundle, "data": data, "resolved": resolved, "config": config, "normalizer": normalizer}


def verify_validation_frame(frame, inspected):
    records = {r["recording_id"]: r["participant_id"] for r in inspected["data"]["validation"]["recordings"]}
    if (frame.empty or not frame.dataset.eq("sleepedf").all()
            or any(records.get(str(r.recording_id)) != str(r.participant_id) for r in frame.itertuples())):
        raise ValueError("Development predictions include records outside the model's source-validation partition.")

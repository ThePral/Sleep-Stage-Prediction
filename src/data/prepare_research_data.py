"""Prepare versioned research data without overwriting the existing feature cache.

Example: python -m src.data.prepare_research_data --dataset sleepedf --recordings SC4002 --compare-cached
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.adapters.isruc import prepare_isruc
from src.data.adapters.sleepedf import prepare_sleepedf
from src.data.epoch_store import EpochStore, sha256_file, write_recording
from src.data.pairing import build_recording_pairs, validate_recording_pairs
from src.data.research import PreprocessingConfig, validate_feature_dataframe
from src.preprocessing.normalization import ALL_FEATURES

ROOT = Path(__file__).resolve().parents[2]


def participant_assignments(path):
    frame = pd.read_csv(path, dtype={"participant_id": str})
    if not {"participant_id", "split"}.issubset(frame.columns) or frame.empty:
        raise ValueError("Participant manifest requires participant_id and split columns.")
    if frame[["participant_id", "split"]].isna().any().any() or frame["participant_id"].duplicated().any():
        raise ValueError("Every participant must have exactly one nonmissing split assignment.")
    if not frame["split"].isin(["train", "val", "test"]).all():
        raise ValueError("Unexpected source split name.")
    return frame.set_index("participant_id")["split"].to_dict()


def compare_cached(recording, path):
    cached = validate_feature_dataframe(pd.read_csv(path))
    actual = recording.epochs.loc[recording.epochs["supervised_valid"]]
    if cached["subject_id"].nunique() != 1 or cached["subject_id"].iloc[0] != actual["recording_id"].iloc[0]:
        raise ValueError("Preserved feature cache belongs to a different recording.")
    if not np.array_equal(actual["epoch_index"].to_numpy(), cached["epoch_index"].to_numpy()):
        raise ValueError("Regenerated supervised epoch positions differ from the preserved cache.")
    if not np.array_equal(actual["target_stage"].to_numpy(), cached["target_stage"].to_numpy()):
        raise ValueError("Regenerated stage labels differ from the preserved cache.")
    if not np.allclose(actual[ALL_FEATURES], cached[ALL_FEATURES], rtol=1e-6, atol=1e-20):
        raise ValueError("Regenerated spectral features differ from the preserved cache.")
    return {"cached_rows": len(cached), "all_features_match": True}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["sleepedf", "isruc"], required=True)
    select = parser.add_mutually_exclusive_group(required=True)
    select.add_argument("--recordings", nargs="+")
    select.add_argument("--all", action="store_true")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/processed/research_v1")
    parser.add_argument("--split-file", type=Path, default=ROOT / "final_model_corrected/participant_split.csv")
    parser.add_argument("--electrode", choices=["C3", "F3"], default="C3")
    parser.add_argument("--scorer", type=int, choices=[1, 2], default=1)
    parser.add_argument("--compare-cached", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = PreprocessingConfig()
    if args.compare_cached and args.dataset != "sleepedf":
        parser.error("--compare-cached applies only to the preserved Sleep-EDF features.")
    default_raw = "data/raw" if args.dataset == "sleepedf" else "data/external/ISRUC-Sleep-I"
    raw_root = (args.data_root or ROOT / default_raw).resolve()
    output_root = args.output_root.resolve()
    if output_root == raw_root or raw_root in output_root.parents or output_root in raw_root.parents:
        raise ValueError("Output and raw-data directories must not contain one another.")
    selected = set(args.recordings or [])
    if args.dataset == "sleepedf":
        pairs = build_recording_pairs(raw_root)
        if not validate_recording_pairs(pairs)["is_valid"]:
            raise ValueError("Incomplete PSG/hypnogram pairings.")
        assignments = participant_assignments(args.split_file)
        plans = []
        for row in pairs.itertuples(index=False):
            if selected and row.recording_id not in selected:
                continue
            participant = row.recording_id[:5]
            if participant not in assignments:
                raise ValueError(f"No frozen assignment for {participant}.")
            plans.append((row.recording_id, [Path(row.psg), Path(row.hypnogram)], assignments[participant]))
    else:
        plans = []
        for folder in sorted(raw_root.iterdir(), key=lambda p: p.name):
            if folder.is_dir() and folder.name.isdigit():
                name = str(int(folder.name))
                if not selected or name in selected:
                    plans.append((name, [folder / f"{name}.rec", folder / f"{name}_{args.scorer}.xlsx"], "external"))
    if selected - {name for name, _, _ in plans}:
        raise ValueError(f"Recordings not found: {sorted(selected - {name for name, _, _ in plans})}")
    if not plans:
        raise ValueError("No recordings selected.")
    for name, paths, split in plans:
        if any(not path.is_file() for path in paths):
            raise FileNotFoundError(f"Missing canonical input for {name}.")
        destination = output_root / args.dataset / name
        if args.dry_run:
            print(json.dumps({"recording": name, "split": split, "destination": str(destination)}))
            continue
        if destination.exists():
            store = EpochStore(destination)
            provenance = store.manifest["provenance"]
            expected_sources = {str(path.resolve()): sha256_file(path) for path in paths}
            actual_sources = {item["path"]: item["sha256"] for item in provenance["sources"]}
            if actual_sources != expected_sources or store.config != config or store.epochs["split"].iloc[0] != split:
                raise ValueError(f"Existing store {name} has a different source/configuration/split. Use a new output root.")
            if args.dataset == "isruc" and (provenance["scorer"] != args.scorer or provenance["electrode_policy"] != args.electrode):
                raise ValueError("Existing ISRUC store uses a different scorer/channel policy. Use a new output root.")
            recording = None
            summary = {"recording": name, "status": "verified_existing", "rows": len(store.epochs)}
            if args.compare_cached:
                summary.update(compare_cached(store, ROOT / f"data/features/individual/{name}_features.csv"))
        else:
            if args.dataset == "sleepedf":
                recording = prepare_sleepedf(*paths, split=split, config=config)
                recording.provenance["split_manifest"] = {"path": str(args.split_file.resolve()), "sha256": sha256_file(args.split_file)}
            else:
                recording = prepare_isruc(paths[0].parent, config=config, electrode=args.electrode, scorer=args.scorer)
            summary = {"recording": name, "status": "prepared", "rows": len(recording.epochs), "supervised_rows": int(recording.epochs["supervised_valid"].sum())}
            if args.compare_cached:
                summary.update(compare_cached(recording, ROOT / f"data/features/individual/{name}_features.csv"))
            write_recording(recording, destination)
        print(json.dumps(summary))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Audit Sleep-EDF and ISRUC-Sleep files before cross-dataset experiments.

Run from the root of the Sleep-Stage-Prediction project:
    python audit_sleep_dataset_compatibility.py \
        --sleep-edf data/raw \
        --isruc data/external/isruc \
        --features data/features/sleep_edf_features.csv \
        --split data/features/participant_split.csv

The script reads metadata and annotation summaries only; it does not preprocess
signals, train models, or modify any source files.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def print_section(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def find_files(root: Path, patterns: list[str]) -> list[Path]:
    found: set[Path] = set()
    if not root.exists():
        return []
    for pattern in patterns:
        found.update(p for p in root.rglob(pattern) if p.is_file())
    return sorted(found)


def inspect_signal_files(files: list[Path], dataset_name: str, max_files: int) -> list[dict[str, Any]]:
    print_section(f"{dataset_name}: signal-file metadata")
    if not files:
        print("No EDF/REC signal files found under the configured path.")
        return []

    print(f"Discovered {len(files)} candidate signal files. Showing up to {max_files}.")
    try:
        import mne
    except ImportError:
        print("ERROR: MNE is not installed. Install it with: pip install mne")
        return []

    records: list[dict[str, Any]] = []
    for path in files[:max_files]:
        row: dict[str, Any] = {"file": str(path)}
        try:
            raw = mne.io.read_raw_edf(str(path), preload=False, verbose="ERROR")
            row.update({
                "sfreq_hz": float(raw.info["sfreq"]),
                "n_channels": int(len(raw.ch_names)),
                "duration_seconds": round(float(raw.n_times / raw.info["sfreq"]), 3),
                "n_samples": int(raw.n_times),
                "channels": list(raw.ch_names),
                "channel_types": list(raw.get_channel_types()),
            })
            print(f"\nFile: {path}")
            print(f"  Sampling rate: {row['sfreq_hz']} Hz")
            print(f"  Duration: {row['duration_seconds'] / 3600:.3f} hours")
            print(f"  Channels ({row['n_channels']}): {', '.join(row['channels'])}")
            print(f"  Types: {', '.join(row['channel_types'])}")
            raw.close()
        except Exception as exc:  # continue to inspect other files
            row["error"] = f"{type(exc).__name__}: {exc}"
            print(f"\nFile: {path}\n  FAILED TO READ: {row['error']}")
        records.append(row)
    return records


def inspect_annotations(root: Path, dataset_name: str, max_files: int) -> dict[str, Any]:
    print_section(f"{dataset_name}: annotation/label files")
    result: dict[str, Any] = {}
    if not root.exists():
        print(f"Path does not exist: {root}")
        return result

    # Sleep-EDF uses a separate EDF+ hypnogram file. ISRUC releases vary in
    # packaging: BIDS-style events TSVs, spreadsheets, or text annotations.
    patterns = ["*Hypnogram.edf", "*_events.tsv", "*.xlsx", "*.xls", "*.csv", "*.tsv", "*.txt"]
    files = find_files(root, patterns)
    by_suffix = Counter(p.suffix.lower() for p in files)
    print(f"Candidate annotation/label files: {len(files)}")
    print("Counts by extension:", dict(by_suffix))
    result["candidate_count"] = len(files)
    result["counts_by_extension"] = dict(by_suffix)
    result["examples"] = [str(p) for p in files[:max_files]]

    # Show one table-like label file's columns and sample rows without assuming
    # that every version uses the same annotation schema.
    # Prefer event/label tables over channel metadata tables when present.
    event_tables = [p for p in files if p.name.lower().endswith("_events.tsv")]
    other_tables = [
        p for p in files
        if p.suffix.lower() in {".tsv", ".csv", ".xlsx", ".xls"}
        and not p.name.lower().endswith("_channels.tsv")
        and p not in event_tables
    ]
    table_files = event_tables + other_tables
    if table_files:
        try:
            import pandas as pd
            p = table_files[0]
            if p.suffix.lower() == ".tsv":
                frame = pd.read_csv(p, sep="\t", nrows=8)
            elif p.suffix.lower() == ".csv":
                frame = pd.read_csv(p, nrows=8)
            else:
                frame = pd.read_excel(p, nrows=8)
            result["sample_table"] = {
                "file": str(p),
                "columns": [str(c) for c in frame.columns],
                "first_rows": frame.astype(str).to_dict(orient="records"),
            }
            print(f"\nSample tabular label/metadata file: {p}")
            print("Columns:", list(frame.columns))
            print(frame.to_string(index=False))
        except Exception as exc:
            print(f"Could not preview tabular file: {type(exc).__name__}: {exc}")

    hypnograms = [p for p in files if p.name.lower().endswith("hypnogram.edf")]
    if hypnograms:
        try:
            import mne
            p = hypnograms[0]
            ann = mne.read_annotations(str(p), verbose="ERROR")
            descriptions = Counter(str(x) for x in ann.description)
            result["sample_hypnogram"] = {
                "file": str(p),
                "annotation_count": int(len(ann)),
                "descriptions": dict(descriptions),
            }
            print(f"\nSample Sleep-EDF hypnogram: {p}")
            print("Annotation descriptions and counts:", dict(descriptions))
        except Exception as exc:
            print(f"Could not parse sample hypnogram: {type(exc).__name__}: {exc}")
    return result


def inspect_project_tables(features_path: Path, split_path: Path) -> dict[str, Any]:
    print_section("Existing project: feature table and participant split")
    result: dict[str, Any] = {}
    try:
        import pandas as pd
    except ImportError:
        print("ERROR: pandas is not installed. Install it with: pip install pandas")
        return result

    if features_path.exists():
        expected = [
            "subject_id", "epoch_index", "start_time", "end_time", "raw_stage", "target_stage",
            "delta_absolute", "theta_absolute", "alpha_absolute", "beta_absolute",
            "delta_relative", "theta_relative", "alpha_relative", "beta_relative",
            "spectral_entropy", "dominant_frequency",
        ]
        print(f"Features: {features_path}")
        print(f"File size: {features_path.stat().st_size / (1024**2):.2f} MiB")
        counts = Counter()
        recordings: set[str] = set()
        stages: Counter[str] = Counter()
        missing_by_col: Counter[str] = Counter()
        columns: list[str] | None = None
        for chunk in pd.read_csv(features_path, chunksize=100_000):
            if columns is None:
                columns = [str(c) for c in chunk.columns]
            counts["rows"] += len(chunk)
            if "subject_id" in chunk:
                recordings.update(chunk["subject_id"].dropna().astype(str).unique())
            if "target_stage" in chunk:
                stages.update(chunk["target_stage"].dropna().astype(str).value_counts().to_dict())
            for col in chunk.columns:
                missing_by_col[str(col)] += int(chunk[col].isna().sum())
        result["features"] = {
            "path": str(features_path),
            "rows": counts["rows"],
            "recording_ids": len(recordings),
            "stages": dict(stages),
            "columns": columns or [],
            "missing_values_by_column": dict(missing_by_col),
            "missing_expected_columns": sorted(set(expected) - set(columns or [])),
            "participant_id_rule_for_existing_project": "first 5 characters of subject_id (verify in preprocessing code)",
        }
        print(f"Rows: {counts['rows']:,}; recording IDs: {len(recordings)}")
        print("Target stages/counts:", dict(stages))
        print("Missing expected columns:", result["features"]["missing_expected_columns"])
        total_missing = sum(missing_by_col.values())
        print(f"Total missing cells: {total_missing:,}")
    else:
        print(f"Feature CSV not found: {features_path}")

    if split_path.exists():
        split = pd.read_csv(split_path)
        print(f"\nParticipant split: {split_path}")
        print("Columns:", list(split.columns))
        if "participant_id" not in split or "split" not in split:
            print("WARNING: Expected participant_id and split columns are absent.")
        else:
            duplicate_ids = int(split["participant_id"].duplicated().sum())
            counts_split = split.groupby("split")["participant_id"].nunique().to_dict()
            overlap_summary: dict[str, int] = {}
            sets = {name: set(g["participant_id"].astype(str))
                    for name, g in split.groupby("split")}
            names = sorted(sets)
            for i, left in enumerate(names):
                for right in names[i + 1:]:
                    overlap_summary[f"{left} ∩ {right}"] = len(sets[left] & sets[right])
            type_counts = {}
            if "dataset_type" in split:
                type_counts = pd.crosstab(split["dataset_type"], split["split"]).to_dict()
            result["split"] = {
                "path": str(split_path),
                "rows": len(split),
                "unique_participants": int(split["participant_id"].nunique()),
                "participant_counts_by_split": {str(k): int(v) for k, v in counts_split.items()},
                "duplicate_participant_rows": duplicate_ids,
                "pairwise_overlap_counts": overlap_summary,
                "dataset_type_by_split": type_counts,
            }
            print(f"Unique participants: {split['participant_id'].nunique()}")
            print("Participants by split:", counts_split)
            print("Duplicate participant rows:", duplicate_ids)
            print("Pairwise overlaps:", overlap_summary)
            if "dataset_type" in split:
                print("Dataset type by split:")
                print(pd.crosstab(split["dataset_type"], split["split"]).to_string())
    else:
        print(f"Participant split CSV not found: {split_path}")

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sleep-edf", type=Path, default=Path("data/raw"), help="Root containing Sleep-EDF files")
    parser.add_argument("--isruc", type=Path, default=Path("data/external/isruc"), help="Root containing ISRUC files")
    parser.add_argument("--features", type=Path, default=Path("data/features/sleep_edf_features.csv"), help="Existing feature CSV")
    parser.add_argument("--split", type=Path, default=Path("data/features/participant_split.csv"), help="Participant split CSV")
    parser.add_argument("--max-files", type=int, default=5, help="Number of signal files to inspect for each dataset")
    parser.add_argument("--report", type=Path, default=Path("results/dataset_compatibility_audit.json"), help="JSON report path")
    args = parser.parse_args()

    if args.max_files < 1:
        parser.error("--max-files must be >= 1")

    print("Sleep-stage cross-dataset compatibility audit")
    print(f"Sleep-EDF root: {args.sleep_edf.resolve()}")
    print(f"ISRUC root:     {args.isruc.resolve()}")

    sleep_files = find_files(args.sleep_edf, ["*-PSG.edf", "*PSG.edf"])
    isruc_files = find_files(args.isruc, ["*_eeg.edf", "*.edf", "*.rec"])
    # Avoid duplicate matches from overlapping glob patterns.
    sleep_files = sorted(set(sleep_files))
    isruc_files = sorted(set(isruc_files))

    report: dict[str, Any] = {
        "configured_paths": {
            "sleep_edf": str(args.sleep_edf),
            "isruc": str(args.isruc),
            "features": str(args.features),
            "split": str(args.split),
        },
        "sleep_edf_signal_files_discovered": len(sleep_files),
        "isruc_signal_files_discovered": len(isruc_files),
    }
    report["sleep_edf_signals"] = inspect_signal_files(sleep_files, "Sleep-EDF", args.max_files)
    report["isruc_signals"] = inspect_signal_files(isruc_files, "ISRUC-Sleep", args.max_files)
    report["sleep_edf_annotations"] = inspect_annotations(args.sleep_edf, "Sleep-EDF", args.max_files)
    report["isruc_annotations"] = inspect_annotations(args.isruc, "ISRUC-Sleep", args.max_files)
    report["existing_project"] = inspect_project_tables(args.features, args.split)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print_section("Audit saved")
    print(f"JSON report: {args.report.resolve()}")
    print("Note: inspect the printed channel/annotation details manually before choosing the target channel.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

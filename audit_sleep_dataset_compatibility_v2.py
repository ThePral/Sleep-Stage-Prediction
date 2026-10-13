#!/usr/bin/env python3
"""Audit Sleep-EDF and ISRUC-Sleep compatibility before cross-dataset experiments.

This is a read-only audit of source data. For ISRUC .rec signal files, the script
creates a temporary hard-link with an .edf suffix in the same folder so MNE can
read the EDF-formatted content. The original file is not renamed or modified.

Run from the project root, for example:
  python audit_sleep_dataset_compatibility_v2.py \
    --sleep-edf data/raw \
    --isruc data/external/ISRUC-Sleep-I \
    --features data/features/sleep_edf_features.csv \
    --split data/features/participant_split.csv \
    --max-files 101 \
    --report results/dataset_compatibility_audit_v2.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

EXPECTED_STAGES = {"W", "N1", "N2", "N3", "N4", "REM", "R", "S0", "S1", "S2", "S3", "S4", "Wake", "Wakefulness"}


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


def read_edf_compatible(path: Path):
    """Read EDF or EDF-formatted .rec, returning (raw, temporary_alias).

    The caller must close ``raw`` before removing the alias; this ordering is
    important on Windows, where an open file can prevent alias deletion.
    """
    import mne

    alias: Path | None = None
    target = path
    if path.suffix.lower() == ".rec":
        alias = path.with_name(f".{path.stem}_audit_alias.edf")
        if alias.exists():
            # Avoid overwriting anything not created by this invocation.
            alias = path.with_name(f".{path.stem}_audit_alias_{os.getpid()}.edf")
        try:
            os.link(path, alias)
        except OSError as exc:
            raise RuntimeError(
                "Cannot create a temporary hard-link alias beside this .rec file. "
                "The audit deliberately does not copy or rename large source files. "
                f"Hard-link error: {type(exc).__name__}: {exc}"
            ) from exc
        target = alias

    try:
        try:
            raw = mne.io.read_raw_edf(
                str(target), preload=False, infer_types=True, verbose="ERROR"
            )
        except TypeError:
            # Compatibility with older MNE versions without infer_types.
            raw = mne.io.read_raw_edf(str(target), preload=False, verbose="ERROR")
        return raw, alias
    except Exception:
        if alias is not None:
            try:
                alias.unlink(missing_ok=True)
            except OSError:
                pass
        raise


def inspect_signal_files(files: list[Path], dataset_name: str, max_files: int) -> list[dict[str, Any]]:
    print_section(f"{dataset_name}: signal-file metadata")
    if not files:
        print("No candidate signal files found under this path.")
        return []

    chosen = files if max_files <= 0 else files[:max_files]
    print(f"Discovered {len(files)} candidate signal files. Inspecting {len(chosen)}.")
    try:
        import mne  # noqa: F401
    except ImportError:
        print("ERROR: MNE is not installed. Install with: pip install mne")
        return []

    records: list[dict[str, Any]] = []
    sfreqs: Counter[str] = Counter()
    channel_signatures: Counter[str] = Counter()
    for path in chosen:
        row: dict[str, Any] = {"file": str(path)}
        raw = None
        alias = None
        try:
            raw, alias = read_edf_compatible(path)
            ch_types = list(raw.get_channel_types())
            row.update({
                "sfreq_hz": float(raw.info["sfreq"]),
                "n_channels": int(len(raw.ch_names)),
                "duration_seconds": round(float(raw.n_times / raw.info["sfreq"]), 3),
                "n_samples": int(raw.n_times),
                "channels": list(raw.ch_names),
                "channel_types": ch_types,
                "format_suffix": path.suffix.lower(),
            })
            sfreqs[str(row["sfreq_hz"])] += 1
            channel_signatures[" | ".join(row["channels"])] += 1
            print(f"\nFile: {path}")
            print(f"  Sampling rate: {row['sfreq_hz']} Hz")
            print(f"  Duration: {row['duration_seconds'] / 3600:.3f} hours")
            print(f"  Channels ({row['n_channels']}): {', '.join(row['channels'])}")
            print(f"  Types: {', '.join(ch_types)}")
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            print(f"\nFile: {path}\n  FAILED TO READ: {row['error']}")
        finally:
            if raw is not None:
                raw.close()
            if alias is not None:
                try:
                    alias.unlink(missing_ok=True)
                except OSError as exc:
                    print(f"  WARNING: Could not remove temporary alias {alias}: {exc}")
        records.append(row)

    print("\nSampling-rate counts among successfully read files:", dict(sfreqs))
    print("Distinct channel signatures:", len(channel_signatures))
    return records


def normalize_stage(value: Any) -> str | None:
    """Normalize common stage strings without silently mapping ambiguous labels."""
    if value is None:
        return None
    try:
        import pandas as pd
        if pd.isna(value):
            return None
    except Exception:
        pass
    s = str(value).strip().upper()
    if not s or s in {"NAN", "NONE", "NA", "N/A", "-", "?"}:
        return None
    aliases = {
        "WAKE": "W", "WAKEFULNESS": "W", "S0": "W", "0": "W",
        "S1": "N1", "1": "N1", "STAGE 1": "N1", "NREM1": "N1",
        "S2": "N2", "2": "N2", "STAGE 2": "N2", "NREM2": "N2",
        "S3": "N3", "3": "N3", "STAGE 3": "N3", "NREM3": "N3",
        "S4": "N4", "4": "N4", "STAGE 4": "N4", "NREM4": "N4",
        "R": "REM", "SREM": "REM", "STAGE REM": "REM",
    }
    if s.startswith("STAGE "):
        s = s.replace("STAGE ", "", 1)
    return aliases.get(s, s)


def inspect_isruc_excel_annotations(root: Path, max_files: int) -> dict[str, Any]:
    """Inspect ISRUC Excel tables and summarize stage columns by workbook.

    ISRUC Excel files may include a metadata/statistics block after the epoch rows.
    We therefore keep only rows whose Epoch cell is a positive integer and whose
    Stage cell looks like a sleep-stage value. The report retains raw stage values.
    """
    print_section("ISRUC: Excel annotation audit")
    result: dict[str, Any] = {"files_found": 0, "workbooks_inspected": 0, "files": []}
    files = find_files(root, ["*.xlsx", "*.xls"])
    result["files_found"] = len(files)
    if not files:
        print("No Excel annotation files found.")
        return result

    try:
        import pandas as pd
    except ImportError:
        print("ERROR: pandas is required to inspect Excel annotations. Install with: pip install pandas openpyxl")
        return result

    # Prefer epoch-label sheets, avoid the metadata summary files if their names are not session-based.
    selected = files if max_files <= 0 else files[:max_files]
    all_raw_stages: Counter[str] = Counter()
    all_normalized: Counter[str] = Counter()
    session_summaries: list[dict[str, Any]] = []
    for p in selected:
        summary: dict[str, Any] = {"file": str(p)}
        try:
            frame = pd.read_excel(p, usecols=None)
            cols = {str(c).strip().lower(): c for c in frame.columns}
            epoch_col = cols.get("epoch")
            stage_col = cols.get("stage")
            if epoch_col is None or stage_col is None:
                summary["skipped_reason"] = "No columns named Epoch and Stage"
                summary["columns"] = [str(c) for c in frame.columns]
                session_summaries.append(summary)
                continue

            epochs = pd.to_numeric(frame[epoch_col], errors="coerce")
            raw_stage_strings = frame[stage_col].astype(str).str.strip()
            valid_epoch = epochs.notna() & (epochs >= 1) & (epochs % 1 == 0)
            # Keep rows with stage labels matching common ISRUC stage tokens.
            allowed = raw_stage_strings.str.upper().isin({
                "W", "WAKE", "WAKEFULNESS", "N1", "N2", "N3", "N4", "REM", "R",
                "S0", "S1", "S2", "S3", "S4", "0", "1", "2", "3", "4",
                "STAGE 1", "STAGE 2", "STAGE 3", "STAGE 4", "STAGE REM"
            })
            valid = valid_epoch & allowed
            raw_counts = Counter(raw_stage_strings[valid].tolist())
            norm_counts = Counter(x for x in (normalize_stage(v) for v in raw_stage_strings[valid].tolist()) if x)
            session_id = p.stem.replace("_", "/")
            summary.update({
                "session_id_from_filename": session_id,
                "rows_in_workbook": int(len(frame)),
                "valid_epoch_label_rows": int(valid.sum()),
                "first_epoch": int(epochs[valid].min()) if valid.any() else None,
                "last_epoch": int(epochs[valid].max()) if valid.any() else None,
                "raw_stage_counts": dict(raw_counts),
                "normalized_stage_counts_pre_harmonization": dict(norm_counts),
                "columns": [str(c) for c in frame.columns],
            })
            all_raw_stages.update(raw_counts)
            all_normalized.update(norm_counts)
            session_summaries.append(summary)
        except Exception as exc:
            summary["error"] = f"{type(exc).__name__}: {exc}"
            session_summaries.append(summary)

    result["workbooks_inspected"] = len(selected)
    result["aggregate_raw_stage_counts_across_inspected_workbooks"] = dict(all_raw_stages)
    result["aggregate_normalized_stage_counts_pre_harmonization"] = dict(all_normalized)
    result["files"] = session_summaries
    print(f"Excel annotation files found: {len(files)}; inspected: {len(selected)}")
    print("Aggregate raw stage counts across inspected workbooks:", dict(all_raw_stages))
    print("Normalized stage counts before N3/N4 harmonization:", dict(all_normalized))
    print("NOTE: aggregate counts can double-count sessions/experts if files contain duplicate or alternate scoring exports.")
    return result


def inspect_isruc_text_annotations(root: Path, max_files: int) -> dict[str, Any]:
    """List TXT annotation candidates and preview the first few nonempty lines."""
    print_section("ISRUC: text annotation preview")
    files = find_files(root, ["*.txt"])
    chosen = files if max_files <= 0 else files[:max_files]
    previews = []
    for p in chosen[:min(10, len(chosen))]:
        try:
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
            previews.append({"file": str(p), "first_nonempty_lines": [s.strip() for s in lines if s.strip()][:12]})
        except Exception as exc:
            previews.append({"file": str(p), "error": f"{type(exc).__name__}: {exc}"})
    print(f"TXT files found: {len(files)}; showing previews for {len(previews)}")
    for item in previews:
        print(f"\n{item['file']}")
        if "error" in item:
            print(item["error"])
        else:
            for line in item["first_nonempty_lines"]:
                print("  ", line)
    return {"files_found": len(files), "preview": previews}


def inspect_sleep_edf_hypnograms(root: Path, max_files: int) -> dict[str, Any]:
    print_section("Sleep-EDF: hypnogram audit")
    files = find_files(root, ["*Hypnogram.edf"])
    chosen = files if max_files <= 0 else files[:max_files]
    result: dict[str, Any] = {"files_found": len(files), "inspected": []}
    try:
        import mne
    except ImportError:
        result["error"] = "MNE is not installed"
        return result
    for p in chosen:
        try:
            ann = mne.read_annotations(str(p), verbose="ERROR")
            result["inspected"].append({
                "file": str(p),
                "annotation_count": int(len(ann)),
                "descriptions": dict(Counter(str(x) for x in ann.description)),
            })
        except Exception as exc:
            result["inspected"].append({"file": str(p), "error": f"{type(exc).__name__}: {exc}"})
    print(f"Hypnogram files found: {len(files)}; inspected: {len(chosen)}")
    if result["inspected"]:
        print("First inspected hypnogram:", json.dumps(result["inspected"][0], indent=2, ensure_ascii=False))
    return result


def inspect_project_tables(features_path: Path, split_path: Path) -> dict[str, Any]:
    print_section("Existing project: feature table and participant split")
    result: dict[str, Any] = {}
    try:
        import pandas as pd
    except ImportError:
        print("ERROR: pandas is not installed. Install with: pip install pandas")
        return result

    if features_path.exists():
        expected = [
            "subject_id", "epoch_index", "start_time", "end_time", "raw_stage", "target_stage",
            "delta_absolute", "theta_absolute", "alpha_absolute", "beta_absolute",
            "delta_relative", "theta_relative", "alpha_relative", "beta_relative",
            "spectral_entropy", "dominant_frequency",
        ]
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
            "path": str(features_path), "rows": counts["rows"], "recording_ids": len(recordings),
            "stages": dict(stages), "columns": columns or [],
            "missing_values_by_column": dict(missing_by_col),
            "missing_expected_columns": sorted(set(expected) - set(columns or [])),
        }
        print(f"Feature rows: {counts['rows']:,}; recording IDs: {len(recordings)}")
        print("Target stages:", dict(stages))
        print("Missing expected columns:", result["features"]["missing_expected_columns"])
    else:
        result["features_error"] = f"Feature CSV not found: {features_path}"
        print(result["features_error"])

    if split_path.exists():
        split = pd.read_csv(split_path)
        print(f"\nParticipant split: {split_path}")
        if not {"participant_id", "split"}.issubset(split.columns):
            result["split_error"] = "Expected participant_id and split columns"
            print(result["split_error"])
        else:
            sets = {str(name): set(group["participant_id"].astype(str)) for name, group in split.groupby("split")}
            overlap: dict[str, int] = {}
            names = sorted(sets)
            for i, left in enumerate(names):
                for right in names[i + 1:]:
                    overlap[f"{left} ∩ {right}"] = len(sets[left] & sets[right])
            split_result = {
                "rows": int(len(split)),
                "unique_participants": int(split["participant_id"].nunique()),
                "participant_counts_by_split": {str(k): int(v) for k, v in split.groupby("split")["participant_id"].nunique().items()},
                "duplicate_participant_rows": int(split["participant_id"].duplicated().sum()),
                "pairwise_overlap_counts": overlap,
            }
            if "dataset_type" in split:
                split_result["dataset_type_by_split"] = pd.crosstab(split["dataset_type"], split["split"]).to_dict()
            result["split"] = split_result
            print(json.dumps(split_result, indent=2, ensure_ascii=False))
    else:
        result["split_error"] = f"Split CSV not found: {split_path}"
        print(result["split_error"])
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sleep-edf", type=Path, default=Path("data/raw"), help="Root containing Sleep-EDF PSG and hypnogram files")
    parser.add_argument("--isruc", type=Path, default=Path("data/external/ISRUC-Sleep-I"), help="Root containing ISRUC .rec and annotation files")
    parser.add_argument("--features", type=Path, default=Path("data/features/sleep_edf_features.csv"), help="Existing Sleep-EDF feature CSV")
    parser.add_argument("--split", type=Path, default=Path("data/features/participant_split.csv"), help="Existing participant split CSV")
    parser.add_argument("--max-files", type=int, default=5, help="Maximum signal files and Excel files to inspect per source; use 0 to inspect all")
    parser.add_argument("--report", type=Path, default=Path("results/dataset_compatibility_audit_v2.json"), help="Output JSON report path")
    args = parser.parse_args()
    if args.max_files < 0:
        parser.error("--max-files must be >= 0 (0 means all files)")

    print("Sleep-stage cross-dataset compatibility audit v2")
    print(f"Sleep-EDF root: {args.sleep_edf.resolve()}")
    print(f"ISRUC root:     {args.isruc.resolve()}")
    if not args.sleep_edf.exists():
        print(f"WARNING: Sleep-EDF path does not exist: {args.sleep_edf}")
    if not args.isruc.exists():
        print(f"WARNING: ISRUC path does not exist: {args.isruc}")

    sleep_signals = sorted(set(find_files(args.sleep_edf, ["*-PSG.edf", "*PSG.edf"])))
    isruc_signals = sorted(set(find_files(args.isruc, ["*.edf", "*.rec"])))
    # Ignore the temporary aliases if a previous run was interrupted.
    isruc_signals = [p for p in isruc_signals if "_audit_alias" not in p.name.lower()]

    report: dict[str, Any] = {
        "audit_version": 2,
        "configured_paths": {
            "sleep_edf": str(args.sleep_edf), "isruc": str(args.isruc),
            "features": str(args.features), "split": str(args.split),
        },
        "sleep_edf_signal_files_discovered": len(sleep_signals),
        "isruc_signal_files_discovered": len(isruc_signals),
        "note": "For ISRUC .rec files, metadata are read through a temporary hard-link alias; original files are not modified.",
    }
    report["sleep_edf_signals"] = inspect_signal_files(sleep_signals, "Sleep-EDF", args.max_files)
    report["isruc_signals"] = inspect_signal_files(isruc_signals, "ISRUC-Sleep", args.max_files)
    report["sleep_edf_annotations"] = inspect_sleep_edf_hypnograms(args.sleep_edf, args.max_files)
    report["isruc_annotations_excel"] = inspect_isruc_excel_annotations(args.isruc, args.max_files)
    report["isruc_annotations_text"] = inspect_isruc_text_annotations(args.isruc, args.max_files)
    report["existing_project"] = inspect_project_tables(args.features, args.split)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print_section("Audit saved")
    print(f"JSON report: {args.report.resolve()}")
    print("Review the channel compatibility and annotation alignment before starting model experiments.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

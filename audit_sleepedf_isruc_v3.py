#!/usr/bin/env python3
"""Robust, read-only audit of Sleep-EDF Expanded and ISRUC-Sleep Cohort I.

This version fixes the MNE read_annotations API issue, identifies the duplicate
alternate-named recording in ISRUC subject 65, and parses ISRUC annotation files
without assuming Excel headers are consistent. It does NOT train a model or
modify source data.

Run from the project root:
  python audit_sleepedf_isruc_v3.py \\
    --sleep-edf data/raw \\
    --isruc data/external/ISRUC-Sleep-I \\
    --features data/features/sleep_edf_features.csv \\
    --split data/features/participant_split.csv \\
    --report results/dataset_compatibility_audit_v3.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SLEEPEDF_STAGE_RE = re.compile(r"sleep\s*stage\s*([a-z0-9?]+)", re.I)
ISRUC_NAME_RE = re.compile(r"^(\d+)_(\d+)\.(txt|xlsx)$", re.I)


def find_files(root: Path, patterns: list[str]) -> list[Path]:
    found: set[Path] = set()
    if root.exists():
        for pattern in patterns:
            found.update(p for p in root.rglob(pattern) if p.is_file())
    return sorted(found)


def read_edf_compatible(path: Path):
    """Read EDF or EDF-compatible .rec through a temporary hard-link alias."""
    import mne

    alias: Path | None = None
    target = path
    if path.suffix.lower() == ".rec":
        alias = path.with_name(f".{path.stem}_audit_alias_{os.getpid()}.edf")
        if alias.exists():
            raise FileExistsError(f"Temporary alias already exists: {alias}")
        try:
            os.link(path, alias)
        except OSError as exc:
            raise RuntimeError(
                "Could not create temporary hard link beside the .rec file; "
                "the source file was not modified. "
                f"Hard-link error: {type(exc).__name__}: {exc}"
            ) from exc
        target = alias

    try:
        try:
            raw = mne.io.read_raw_edf(
                str(target), preload=False, infer_types=True, verbose="ERROR"
            )
        except TypeError:
            raw = mne.io.read_raw_edf(str(target), preload=False, verbose="ERROR")
        return raw, alias
    except Exception:
        if alias is not None:
            alias.unlink(missing_ok=True)
        raise


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sleepedf_record_key(path: Path) -> str | None:
    """Return recording key such as SC4001 from PSG/hypnogram filenames."""
    m = re.match(r"^((?:SC|ST)\d{4})", path.name, flags=re.I)
    return m.group(1).upper() if m else None


def sleepedf_participant_key(path: Path) -> str | None:
    """Sleep-EDF participant key (e.g. SC400 for recordings SC4001/SC4002)."""
    key = sleepedf_record_key(path)
    return key[:5] if key else None


def normalize_sleepedf_stage(description: Any) -> str | None:
    text = str(description).strip()
    match = SLEEPEDF_STAGE_RE.search(text)
    label = match.group(1).upper() if match else text.upper()
    aliases = {
        "W": "W", "R": "REM", "REM": "REM",
        "1": "N1", "N1": "N1", "2": "N2", "N2": "N2",
        "3": "N3", "N3": "N3", "4": "N3", "N4": "N3",
        "M": "MOVEMENT", "MOVEMENT": "MOVEMENT", "?": "UNKNOWN",
        "UNKNOWN": "UNKNOWN", "UNSCORED": "UNKNOWN",
    }
    return aliases.get(label, None)


def normalize_isruc_excel_stage(value: Any) -> str | None:
    if value is None:
        return None
    try:
        import pandas as pd
        if pd.isna(value):
            return None
    except Exception:
        pass
    s = str(value).strip().upper()
    if not s or s in {"NAN", "NONE", "NA", "N/A", "?", "-"}:
        return None
    aliases = {
        "W": "W", "WAKE": "W", "N1": "N1", "N2": "N2", "N3": "N3",
        "R": "REM", "REM": "REM", "U": "UNKNOWN", "UNKNOWN": "UNKNOWN",
        "N4": "AMBIGUOUS_N4", "4": "AMBIGUOUS_4",
    }
    if s in aliases:
        return aliases[s]
    if s in {"0", "1", "2", "3", "5", "6"}:
        return {"0": "W", "1": "N1", "2": "N2", "3": "N3", "5": "REM", "6": "UNKNOWN"}[s]
    return None


def normalize_isruc_numeric_code(value: int) -> str:
    # Codes follow the ISRUC numeric-label convention used in current dataset conversions.
    # Code 4 is deliberately left ambiguous; do not silently map it to N3.
    mapping = {0: "W", 1: "N1", 2: "N2", 3: "N3", 5: "REM", 6: "UNKNOWN"}
    return mapping.get(value, f"AMBIGUOUS_CODE_{value}")


def read_isruc_txt_labels(path: Path) -> dict[str, Any]:
    """Read a numeric stage sequence from a TXT file; report rather than hide odd tokens."""
    labels: list[int] = []
    bad_lines: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        text = line.strip().lstrip("\ufeff")
        if not text:
            continue
        # The audited files appear to contain one integer label per line.
        # Allow a one-column CSV/whitespace representation, but flag multi-value rows.
        tokens = re.split(r"[\s,;\t]+", text)
        if len(tokens) == 1 and re.fullmatch(r"[-+]?\d+", tokens[0]):
            labels.append(int(tokens[0]))
        elif len(tokens) > 1 and re.fullmatch(r"[-+]?\d+", tokens[0]):
            bad_lines.append({"line": line_no, "text": text, "reason": "multiple tokens; not consumed"})
        else:
            bad_lines.append({"line": line_no, "text": text[:120], "reason": "not an integer label"})
    mapped = [normalize_isruc_numeric_code(x) for x in labels]
    return {
        "file": str(path),
        "n_labels": len(labels),
        "raw_code_counts": dict(Counter(labels)),
        "mapped_stage_counts_including_unknown_or_ambiguous": dict(Counter(mapped)),
        "bad_line_count": len(bad_lines),
        "bad_line_examples": bad_lines[:10],
        "labels": labels,
        "mapped_labels": mapped,
    }


def read_isruc_excel_labels(path: Path) -> dict[str, Any]:
    """Parse epoch/stage from both headed and headerless ISRUC Excel annotation sheets."""
    import pandas as pd

    frame = pd.read_excel(path, header=None, dtype=object)
    if frame.empty or frame.shape[1] < 2:
        return {"file": str(path), "n_labels": 0, "error": "Workbook has fewer than two columns"}

    labels: list[dict[str, Any]] = []
    for row_i in range(len(frame)):
        epoch_v = frame.iat[row_i, 0]
        stage_v = frame.iat[row_i, 1]
        try:
            epoch_num = float(str(epoch_v).strip())
        except (TypeError, ValueError):
            continue
        if not math.isfinite(epoch_num) or epoch_num < 1 or not epoch_num.is_integer():
            continue
        stage = normalize_isruc_excel_stage(stage_v)
        if stage is None:
            continue
        labels.append({"epoch": int(epoch_num), "stage": stage, "raw_stage": str(stage_v).strip()})

    labels.sort(key=lambda x: x["epoch"])
    epochs = [x["epoch"] for x in labels]
    return {
        "file": str(path),
        "n_labels": len(labels),
        "first_epoch": min(epochs) if epochs else None,
        "last_epoch": max(epochs) if epochs else None,
        "duplicate_epoch_count": len(epochs) - len(set(epochs)),
        "nonconsecutive_epoch_count": sum(1 for a, b in zip(epochs, epochs[1:]) if b != a + 1),
        "raw_stage_counts": dict(Counter(x["raw_stage"].upper() for x in labels)),
        "mapped_stage_counts": dict(Counter(x["stage"] for x in labels)),
        "stages": [x["stage"] for x in labels],
        "labels": labels,
    }


def inspect_sleepedf(sleep_root: Path) -> dict[str, Any]:
    import mne

    psgs = find_files(sleep_root, ["*PSG.edf"])
    hypnos = find_files(sleep_root, ["*Hypnogram.edf"])
    hypno_by_key: dict[str, list[Path]] = defaultdict(list)
    for p in hypnos:
        key = sleepedf_record_key(p)
        if key:
            hypno_by_key[key].append(p)

    records = []
    unmatched_psgs = []
    all_stage_counts: Counter[str] = Counter()
    annotation_errors = []
    for psg in psgs:
        key = sleepedf_record_key(psg)
        match = hypno_by_key.get(key or "", [])
        row: dict[str, Any] = {
            "psg_file": str(psg),
            "recording_key": key,
            "participant_key": sleepedf_participant_key(psg),
            "matched_hypnograms": [str(x) for x in match],
        }
        raw = None
        alias = None
        try:
            raw, alias = read_edf_compatible(psg)
            row.update({
                "sfreq_hz": float(raw.info["sfreq"]),
                "n_channels": int(len(raw.ch_names)),
                "duration_seconds": round(float(raw.n_times / raw.info["sfreq"]), 3),
                "channels": list(raw.ch_names),
            })
        except Exception as exc:
            row["signal_error"] = f"{type(exc).__name__}: {exc}"
        finally:
            if raw is not None:
                raw.close()
            if alias is not None:
                alias.unlink(missing_ok=True)

        if len(match) != 1:
            unmatched_psgs.append({"psg": str(psg), "key": key, "match_count": len(match)})
        elif match:
            try:
                # No verbose argument: compatible with MNE versions that reject it.
                ann = mne.read_annotations(str(match[0]))
                labels = []
                epoch_counts: Counter[str] = Counter()
                duration_counts: Counter[str] = Counter()
                for onset, duration, desc in zip(ann.onset, ann.duration, ann.description):
                    label = normalize_sleepedf_stage(desc)
                    if label:
                        labels.append(label)
                        duration_counts[label] += float(duration)
                        # Sleep-EDF labels usually have 30-second epochs.
                        epoch_counts[label] += int(round(float(duration) / 30.0))
                row["annotation_count"] = len(ann)
                row["annotation_stage_counts_by_30s_equivalent"] = dict(epoch_counts)
                row["annotation_stage_duration_seconds"] = dict(duration_counts)
                row["annotation_unmapped_descriptions"] = sorted({
                    str(desc) for desc in ann.description if normalize_sleepedf_stage(desc) is None
                })
                all_stage_counts.update(epoch_counts)
            except Exception as exc:
                row["annotation_error"] = f"{type(exc).__name__}: {exc}"
                annotation_errors.append({"hypnogram": str(match[0]), "error": row["annotation_error"]})
        records.append(row)

    return {
        "psg_files_found": len(psgs),
        "hypnogram_files_found": len(hypnos),
        "psg_records_inspected": len(records),
        "unique_recording_keys": len({r.get("recording_key") for r in records if r.get("recording_key")}),
        "unique_participant_keys_by_filename_prefix": len({r.get("participant_key") for r in records if r.get("participant_key")}),
        "unmatched_or_ambiguous_psg_hypnogram_pairs": unmatched_psgs,
        "annotation_read_errors": annotation_errors,
        "aggregate_annotation_stage_counts_30s_equivalent": dict(all_stage_counts),
        "records": records,
    }


def inspect_isruc(isruc_root: Path) -> dict[str, Any]:
    # Use numeric subject folder and exactly <subject>/<subject>.rec as the canonical file.
    subject_dirs = sorted(
        [p for p in isruc_root.iterdir() if p.is_dir() and p.name.isdigit()],
        key=lambda p: int(p.name),
    ) if isruc_root.exists() else []
    canonical: list[tuple[int, Path]] = []
    missing_numeric_recordings: list[int] = []
    for folder in subject_dirs:
        sid = int(folder.name)
        p = folder / f"{sid}.rec"
        if p.exists():
            canonical.append((sid, p))
        else:
            missing_numeric_recordings.append(sid)

    signals: list[dict[str, Any]] = []
    channel_presence: Counter[str] = Counter()
    duplicate_extra_files: list[dict[str, Any]] = []
    signal_errors: list[dict[str, Any]] = []
    for sid, path in canonical:
        row: dict[str, Any] = {"participant_id": sid, "file": str(path)}
        raw = None
        alias = None
        try:
            raw, alias = read_edf_compatible(path)
            channels = list(raw.ch_names)
            channel_presence.update(channels)
            row.update({
                "sfreq_hz": float(raw.info["sfreq"]),
                "n_channels": int(len(channels)),
                "duration_seconds": round(float(raw.n_times / raw.info["sfreq"]), 3),
                "expected_30s_epochs": int(round(float(raw.n_times / raw.info["sfreq"]) / 30.0)),
                "channels": channels,
            })
        except Exception as exc:
            row["signal_error"] = f"{type(exc).__name__}: {exc}"
            signal_errors.append({"file": str(path), "error": row["signal_error"]})
        finally:
            if raw is not None:
                raw.close()
            if alias is not None:
                alias.unlink(missing_ok=True)
        signals.append(row)

        # Report noncanonical .rec files in each numeric participant folder.
        for extra in sorted(path.parent.glob("*.rec")):
            if extra.resolve() == path.resolve():
                continue
            extra_row: dict[str, Any] = {"participant_id": sid, "canonical_file": str(path), "extra_file": str(extra)}
            try:
                extra_row["same_size"] = path.stat().st_size == extra.stat().st_size
                extra_row["canonical_sha256"] = sha256_file(path)
                extra_row["extra_sha256"] = sha256_file(extra)
                extra_row["byte_identical"] = extra_row["canonical_sha256"] == extra_row["extra_sha256"]
            except Exception as exc:
                extra_row["hash_error"] = f"{type(exc).__name__}: {exc}"
            duplicate_extra_files.append(extra_row)

    txt_files = find_files(isruc_root, ["*.txt"])
    xlsx_files = find_files(isruc_root, ["*.xlsx", "*.xls"])
    txt_by_key: dict[tuple[int, int], Path] = {}
    xlsx_by_key: dict[tuple[int, int], Path] = {}
    ignored_annotation_files: list[str] = []
    for p in txt_files:
        m = ISRUC_NAME_RE.match(p.name)
        if m and int(m.group(1)) in range(1, 1000) and int(m.group(2)) in (1, 2):
            txt_by_key[(int(m.group(1)), int(m.group(2)))] = p
        else:
            ignored_annotation_files.append(str(p))
    for p in xlsx_files:
        m = ISRUC_NAME_RE.match(p.name)
        if m and m.group(3).lower() == "xlsx" and int(m.group(2)) in (1, 2):
            xlsx_by_key[(int(m.group(1)), int(m.group(2)))] = p
        else:
            ignored_annotation_files.append(str(p))

    txt_results = []
    txt_counts_by_scorer: dict[str, Counter[str]] = {"1": Counter(), "2": Counter()}
    txt_length_mismatches = []
    txt_bad_files = []
    txt_lookup: dict[tuple[int, int], dict[str, Any]] = {}
    signal_by_id = {r["participant_id"]: r for r in signals}
    for key, p in sorted(txt_by_key.items()):
        item = read_isruc_txt_labels(p)
        sid, scorer = key
        item["participant_id"] = sid
        item["scorer_id"] = scorer
        expected = signal_by_id.get(sid, {}).get("expected_30s_epochs")
        item["expected_signal_epochs"] = expected
        item["length_matches_signal_epochs"] = (item["n_labels"] == expected) if expected is not None else None
        item.pop("labels", None)
        item.pop("mapped_labels", None)
        txt_results.append(item)
        txt_lookup[key] = read_isruc_txt_labels(p)
        txt_counts_by_scorer[str(scorer)].update(txt_lookup[key]["mapped_stage_counts_including_unknown_or_ambiguous"])
        if expected is not None and item["n_labels"] != expected:
            txt_length_mismatches.append({"file": str(p), "n_labels": item["n_labels"], "expected_epochs": expected})
        if item["bad_line_count"]:
            txt_bad_files.append({"file": str(p), "bad_line_count": item["bad_line_count"], "examples": item["bad_line_examples"]})

    excel_results = []
    excel_counts_by_scorer: dict[str, Counter[str]] = {"1": Counter(), "2": Counter()}
    excel_length_mismatches = []
    excel_parse_errors = []
    excel_lookup: dict[tuple[int, int], dict[str, Any]] = {}
    for key, p in sorted(xlsx_by_key.items()):
        sid, scorer = key
        try:
            item = read_isruc_excel_labels(p)
            item["participant_id"] = sid
            item["scorer_id"] = scorer
            expected = signal_by_id.get(sid, {}).get("expected_30s_epochs")
            item["expected_signal_epochs"] = expected
            item["length_matches_signal_epochs"] = (item["n_labels"] == expected) if expected is not None else None
            stages = item.pop("stages", [])
            labels = item.pop("labels", [])
            excel_lookup[key] = {**item, "stages": stages, "labels": labels}
            excel_results.append(item)
            excel_counts_by_scorer[str(scorer)].update(item.get("mapped_stage_counts", {}))
            if expected is not None and item["n_labels"] != expected:
                excel_length_mismatches.append({"file": str(p), "n_labels": item["n_labels"], "expected_epochs": expected})
        except Exception as exc:
            err = {"file": str(p), "error": f"{type(exc).__name__}: {exc}"}
            excel_parse_errors.append(err)
            excel_results.append(err)

    # Compare text-coded labels and Excel labels only when lengths match.
    txt_excel_comparisons = []
    for key in sorted(set(txt_lookup) & set(excel_lookup)):
        t = txt_lookup[key]["mapped_labels"]
        e = excel_lookup[key]["stages"]
        if len(t) == len(e) and len(t) > 0:
            exact = sum(a == b for a, b in zip(t, e))
            txt_excel_comparisons.append({
                "participant_id": key[0], "scorer_id": key[1], "n_epochs": len(t),
                "exact_agreement": exact / len(t), "matching_epochs": exact,
            })
        else:
            txt_excel_comparisons.append({
                "participant_id": key[0], "scorer_id": key[1],
                "txt_epochs": len(t), "excel_epochs": len(e),
                "comparison_status": "length mismatch or missing labels; agreement not computed",
            })

    return {
        "numeric_subject_directories": len(subject_dirs),
        "numeric_subject_ids": [int(p.name) for p in subject_dirs],
        "canonical_numeric_recordings_found": len(canonical),
        "missing_numeric_recording_ids": missing_numeric_recordings,
        "signal_errors": signal_errors,
        "sampling_rate_counts_hz": dict(Counter(str(r.get("sfreq_hz")) for r in signals if "sfreq_hz" in r)),
        "channel_count_distribution": dict(Counter(str(r.get("n_channels")) for r in signals if "n_channels" in r)),
        "duration_range_seconds": {
            "min": min((r["duration_seconds"] for r in signals if "duration_seconds" in r), default=None),
            "max": max((r["duration_seconds"] for r in signals if "duration_seconds" in r), default=None),
        },
        "channel_presence_counts": dict(channel_presence),
        "noncanonical_extra_recordings": duplicate_extra_files,
        "annotation_files": {
            "txt_files_found": len(txt_files),
            "recognized_subject_scorer_txt_files": len(txt_by_key),
            "xlsx_or_xls_files_found": len(xlsx_files),
            "recognized_subject_scorer_xlsx_files": len(xlsx_by_key),
            "ignored_extra_annotation_files": sorted(set(ignored_annotation_files)),
            "txt_label_summaries": txt_results,
            "txt_stage_counts_by_scorer": {k: dict(v) for k, v in txt_counts_by_scorer.items()},
            "txt_length_mismatches": txt_length_mismatches,
            "txt_bad_files": txt_bad_files,
            "excel_label_summaries": excel_results,
            "excel_stage_counts_by_scorer": {k: dict(v) for k, v in excel_counts_by_scorer.items()},
            "excel_length_mismatches": excel_length_mismatches,
            "excel_parse_errors": excel_parse_errors,
            "txt_vs_excel_sequence_comparisons": txt_excel_comparisons,
        },
        "signals": signals,
    }


def inspect_existing_features(features_path: Path, split_path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    try:
        import pandas as pd
    except ImportError:
        return {"error": "pandas is required for feature/split checks"}

    if features_path.exists():
        df = pd.read_csv(features_path)
        stage_counts = df["target_stage"].fillna("<NA>").value_counts().to_dict() if "target_stage" in df else {}
        expected_features = [
            "delta_absolute", "theta_absolute", "alpha_absolute", "beta_absolute",
            "delta_relative", "theta_relative", "alpha_relative", "beta_relative",
            "spectral_entropy", "dominant_frequency",
        ]
        missing_cols = [c for c in expected_features if c not in df.columns]
        numeric = [c for c in expected_features if c in df.columns]
        nonfinite = {c: int((~pd.to_numeric(df[c], errors="coerce").map(math.isfinite)).sum()) for c in numeric}
        result["features"] = {
            "path": str(features_path), "rows": len(df),
            "recordings": int(df["subject_id"].nunique()) if "subject_id" in df else None,
            "participant_id_rule_note": "Existing participant ID derivation must be checked in preprocessing code; do not assume recording ID and participant ID are interchangeable.",
            "stages": stage_counts, "missing_expected_feature_columns": missing_cols,
            "nonfinite_counts_by_feature": nonfinite,
        }
    else:
        result["features"] = {"path": str(features_path), "error": "file not found"}

    if split_path.exists():
        split = pd.read_csv(split_path)
        # Try common schema names without silently choosing wrong columns.
        participant_col = next((c for c in ["participant_id", "subject_id", "participant"] if c in split.columns), None)
        split_col = next((c for c in ["split", "set", "subset"] if c in split.columns), None)
        if participant_col and split_col:
            rows_by_split = split.groupby(split_col)[participant_col].nunique().to_dict()
            sets = {str(name): set(group[participant_col].astype(str)) for name, group in split.groupby(split_col)}
            overlaps = {}
            keys = list(sets)
            for i, a in enumerate(keys):
                for b in keys[i+1:]:
                    overlaps[f"{a} ∩ {b}"] = len(sets[a] & sets[b])
            result["split"] = {
                "path": str(split_path), "rows": len(split),
                "participant_column": participant_col, "split_column": split_col,
                "unique_participants": int(split[participant_col].nunique()),
                "unique_participants_by_split": rows_by_split,
                "overlap_counts": overlaps,
            }
        else:
            result["split"] = {"path": str(split_path), "columns": list(split.columns), "error": "could not identify participant and split columns"}
    else:
        result["split"] = {"path": str(split_path), "error": "file not found"}
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sleep-edf", type=Path, required=True)
    parser.add_argument("--isruc", type=Path, required=True)
    parser.add_argument("--features", type=Path, default=Path("data/features/sleep_edf_features.csv"))
    parser.add_argument("--split", type=Path, default=Path("data/features/participant_split.csv"))
    parser.add_argument("--report", type=Path, default=Path("results/dataset_compatibility_audit_v3.json"))
    args = parser.parse_args()

    report = {
        "audit_version": 3,
        "configured_paths": {"sleep_edf": str(args.sleep_edf), "isruc": str(args.isruc),
                              "features": str(args.features), "split": str(args.split)},
        "notes": [
            "Read-only audit; source recordings are not modified.",
            "ISRUC scorer files are evaluated separately; the same EEG recording is not counted twice.",
            "Numeric code 4 is left ambiguous pending confirmation from the local dataset codebook.",
            "EEG montage differs between Sleep-EDF and ISRUC and must be described as an external-domain shift.",
        ],
    }

    print("Auditing Sleep-EDF annotations and signals...")
    report["sleep_edf"] = inspect_sleepedf(args.sleep_edf)
    print("Auditing ISRUC recordings and both annotation formats...")
    report["isruc"] = inspect_isruc(args.isruc)
    print("Checking existing features and participant split...")
    report["existing_project"] = inspect_existing_features(args.features, args.split)

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    se = report["sleep_edf"]
    ir = report["isruc"]
    anns = ir["annotation_files"]
    print("\nAUDIT SUMMARY")
    print(f"Sleep-EDF: {se['psg_files_found']} PSG, {se['hypnogram_files_found']} hypnograms")
    print(f"Sleep-EDF annotation errors: {len(se['annotation_read_errors'])}; unmatched pairs: {len(se['unmatched_or_ambiguous_psg_hypnogram_pairs'])}")
    print(f"ISRUC numeric subject directories: {ir['numeric_subject_directories']}; canonical recordings: {ir['canonical_numeric_recordings_found']}")
    print(f"ISRUC signal errors: {len(ir['signal_errors'])}; sampling rates: {ir['sampling_rate_counts_hz']}")
    print(f"ISRUC TXT scorer files recognized: {anns['recognized_subject_scorer_txt_files']}; Excel scorer files recognized: {anns['recognized_subject_scorer_xlsx_files']}")
    print(f"TXT length mismatches: {len(anns['txt_length_mismatches'])}; Excel length mismatches: {len(anns['excel_length_mismatches'])}")
    print(f"Report saved to: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

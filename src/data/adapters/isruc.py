"""ISRUC-I numeric recordings and explicitly selected workbook/scorer policy.

The central derivation is the fixed default. Actual auricular/mastoid reference
names are retained. No predictive scores are used for channel/file selection.
"""

from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import tempfile

import mne
import numpy as np
import pandas as pd

from src.data.epoch_store import source_manifest
from src.data.research import prepare_recording

LABELS = {
    "W": "W", "N1": "N1", "N2": "N2", "N3": "N3", "R": "REM", "REM": "REM",
    "0": "W", "1": "N1", "2": "N2", "3": "N3", "5": "REM",
    "U": None, "UNKNOWN": None, "?": None, "6": None,
}


def read_workbook_labels(path):
    """Read explicit one-based epoch numbers; reject unknown/ambiguous codes."""
    frame = pd.read_excel(path, header=None, dtype=object, usecols=[0, 1])
    rows = []
    headers = []
    for row_number, (epoch, stage) in enumerate(frame.itertuples(index=False, name=None)):
        if pd.isna(epoch) and pd.isna(stage):
            continue
        if row_number == 0 and str(stage).strip().lower() == "stage":
            try:
                float(epoch)
            except (TypeError, ValueError):
                # Workbook 29 has a damaged first-column header ("hich").
                # Only the first explicit Stage header is skipped; numbered
                # rows still require valid labels and full signal coverage.
                headers.append({"row": 1, "epoch_header": str(epoch), "stage_header": str(stage)})
                continue
        try:
            number = float(epoch)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Unexpected annotation row in {Path(path).name}: {epoch!r}") from exc
        if not np.isfinite(number) or number < 1 or not number.is_integer():
            raise ValueError("ISRUC workbook epoch numbers must be positive integers.")
        text = str(stage).strip().upper()
        if isinstance(stage, (int, float, np.number)) and np.isfinite(stage) and float(stage).is_integer():
            text = str(int(stage))
        if text not in LABELS:
            raise ValueError(f"Unrecognized/ambiguous ISRUC stage {stage!r} in {Path(path).name}.")
        rows.append({"epoch_index": int(number) - 1, "raw_stage": text, "target_stage": LABELS[text]})
    result = pd.DataFrame(rows, columns=["epoch_index", "raw_stage", "target_stage"])
    if result.empty or result["epoch_index"].duplicated().any():
        raise ValueError("ISRUC workbook must contain unique, explicit epoch labels.")
    result = result.sort_values("epoch_index").reset_index(drop=True)
    result.attrs["header_rows"] = headers
    return result


@contextmanager
def edf_alias(path):
    """Provide MNE an EDF suffix without changing files in the raw dataset."""
    path = Path(path).resolve()
    # Prefer a same-volume hard link. All temporary paths remain under the
    # project's processed-data directory, never alongside source recordings.
    temporary_root = Path(__file__).resolve().parents[3] / "data" / "processed" / ".edf_read"
    temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="isruc-", dir=temporary_root) as folder:
        folder = Path(folder).resolve()
        if folder.parent != temporary_root.resolve():
            raise ValueError("EDF alias escaped its intended temporary directory.")
        alias = folder / "signal.edf"
        try:
            os.link(path, alias)
        except OSError:
            shutil.copyfile(path, alias)
        yield alias


def select_derivation(raw, electrode="C3"):
    """Return one stored bipolar or explicitly reconstructed derivation."""
    if electrode not in {"C3", "F3"}:
        raise ValueError("Supported prospective electrodes are C3 and F3.")
    for reference in ["M2", "A2"]:
        name = f"{electrode}-{reference}"
        if name in raw.ch_names:
            return raw.copy().pick([name]), name, name
    for reference in ["M2", "A2"]:
        if electrode in raw.ch_names and reference in raw.ch_names:
            values = raw.get_data(picks=[electrode]) - raw.get_data(picks=[reference])
            name = f"{electrode}-{reference}"
            info = mne.create_info([name], raw.info["sfreq"], ch_types=["eeg"])
            result = mne.io.RawArray(values, info, verbose=False)
            result.set_meas_date(raw.info["meas_date"])
            return result, f"{electrode},{reference}", name
    raise ValueError(f"Requested {electrode} contralateral derivation is unavailable; no channel fallback is permitted.")


def prepare_isruc(subject_directory, *, config=None, electrode="C3", scorer=1):
    subject_directory = Path(subject_directory)
    if not subject_directory.name.isdigit() or scorer not in {1, 2}:
        raise ValueError("ISRUC requires a numeric participant directory and scorer 1 or 2.")
    participant = str(int(subject_directory.name))
    signal_path = subject_directory / f"{participant}.rec"
    annotation_path = subject_directory / f"{participant}_{scorer}.xlsx"
    provenance = {
        "sources": source_manifest([signal_path, annotation_path]),
        "adapter": "isruc-i-v1", "annotation_policy": "explicit-workbook-epochs",
        "scorer": scorer, "electrode_policy": electrode,
    }
    labels = read_workbook_labels(annotation_path)
    provenance["annotation_headers"] = labels.attrs["header_rows"]
    include = [f"{electrode}-M2", f"{electrode}-A2", electrode, "M2", "A2"]
    with edf_alias(signal_path) as alias:
        with mne.io.read_raw_edf(alias, include=include, preload=True, verbose=False) as raw:
            channel, original, derivation = select_derivation(raw, electrode)
            try:
                duration = config.epoch_duration if config is not None else 30.0
                expected = int(np.floor(raw.n_times / raw.info["sfreq"] / duration))
                if not np.array_equal(labels["epoch_index"].to_numpy(), np.arange(expected)):
                    raise ValueError("Workbook epoch coverage disagrees with the complete signal grid; resolve the discrepancy explicitly.")
                result = prepare_recording(
                    channel, labels, dataset="isruc", participant_id=participant,
                    recording_id=participant, channel_as_recorded=original,
                    derivation_used=derivation, split="external", config=config,
                    provenance=provenance,
                )
            finally:
                channel.close()
    result.epochs["annotation_file"] = str(annotation_path.resolve())
    result.epochs["scorer_id"] = str(scorer)
    return result.validate()

"""Sleep-EDF adapter preserving the existing Fpz-Cz feature convention."""

from pathlib import Path
import re

import mne

from src.data.epoch_store import source_manifest
from src.data.pairing import get_hypnogram_recording_id, get_psg_recording_id
from src.data.research import prepare_recording
from src.preprocessing.annotations import build_epoch_metadata


def prepare_sleepedf(psg_path, hypnogram_path, *, split, config=None, channel="EEG Fpz-Cz"):
    recording_id = get_psg_recording_id(psg_path)
    if recording_id != get_hypnogram_recording_id(hypnogram_path):
        raise ValueError("PSG and hypnogram recording IDs disagree.")
    if not re.fullmatch(r"(?:SC|ST)\d{4}", recording_id):
        raise ValueError(f"Unexpected Sleep-EDF recording ID: {recording_id}")
    if split not in {"train", "val", "test"}:
        raise ValueError("Sleep-EDF requires a frozen train/val/test participant assignment.")
    provenance = {"sources": source_manifest([psg_path, hypnogram_path]), "adapter": "sleepedf-v1"}
    with mne.io.read_raw_edf(psg_path, include=[channel], preload=True, verbose=False) as raw:
        annotations = mne.read_annotations(hypnogram_path)
        if annotations.orig_time is not None and raw.info["meas_date"] is not None:
            if annotations.orig_time != raw.info["meas_date"]:
                raise ValueError("PSG and hypnogram time origins differ; explicit alignment is required.")
        duration = config.epoch_duration if config is not None else 30.0
        labels = build_epoch_metadata(
            annotations, recording_id, raw.n_times / raw.info["sfreq"], duration,
            include_excluded=True,
        )
        result = prepare_recording(
            raw, labels, dataset="sleepedf", participant_id=recording_id[:5],
            recording_id=recording_id, channel_as_recorded=channel,
            derivation_used=channel, split=split, config=config, provenance=provenance,
        )
    result.epochs["annotation_file"] = str(Path(hypnogram_path).resolve())
    result.epochs["scorer_id"] = Path(hypnogram_path).name.split("-")[0][-1]
    return result.validate()

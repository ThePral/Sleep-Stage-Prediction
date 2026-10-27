"""Shared, label-independent waveform/spectral epoch representation.

Signals remain in volts and spectral powers in volts squared, preserving the
existing feature convention. Scaling is a later training-only operation.
"""

from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np
import pandas as pd

from src.features.spectral import extract_frequency_features_batch
from src.preprocessing.epoching import extract_all_eeg_epochs, samples_per_epoch
from src.preprocessing.filtering import filter_eeg
from src.preprocessing.normalization import ABSOLUTE_POWER_FEATURES, ALL_FEATURES
from src.preprocessing.sequences import CLASS_LABELS


@dataclass(frozen=True)
class PreprocessingConfig:
    sampling_frequency: float = 100.0
    epoch_duration: float = 30.0
    low_frequency: float = 0.3
    high_frequency: float = 35.0
    welch_samples: int = 1024
    flat_std_threshold: float = 1e-10
    version: str = "research-epochs-v1"

    def __post_init__(self):
        samples_per_epoch(self.sampling_frequency, self.epoch_duration)
        if not 0 < self.low_frequency < self.high_frequency < self.sampling_frequency / 2:
            raise ValueError("Filter frequencies must lie strictly inside the Nyquist range.")
        if isinstance(self.welch_samples, bool) or not isinstance(self.welch_samples, int) or self.welch_samples < 2:
            raise ValueError("welch_samples must be an integer >= 2.")
        if not np.isfinite(self.flat_std_threshold) or self.flat_std_threshold < 0:
            raise ValueError("flat_std_threshold must be nonnegative and finite.")

    @property
    def fingerprint(self):
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass
class ProcessedRecording:
    epochs: pd.DataFrame
    waveforms: np.ndarray
    config: PreprocessingConfig
    provenance: dict

    def validate(self):
        frame = self.epochs
        required = {
            "dataset", "participant_id", "recording_id", "epoch_id", "epoch_index",
            "start_time", "end_time", "original_start_sample", "canonical_start_sample",
            "canonical_end_sample_exclusive", "quality_valid", "label_valid",
            "supervised_valid", "target_stage", "exclusion_reason", "split",
            "channel_as_recorded", "derivation_used", "original_sfreq", "canonical_sfreq",
            *ALL_FEATURES,
        }
        if required - set(frame):
            raise ValueError(f"Missing epoch columns: {sorted(required - set(frame))}")
        if frame.empty:
            raise ValueError("A recording must contain at least one complete epoch.")
        for column in ["dataset", "participant_id", "recording_id", "split"]:
            if frame[column].isna().any() or frame[column].astype(str).str.len().eq(0).any() or frame[column].nunique() != 1:
                raise ValueError(f"A store must contain exactly one nonmissing {column}.")
        if frame["epoch_id"].isna().any() or frame["epoch_id"].duplicated().any():
            raise ValueError("epoch_id must be nonmissing and unique.")
        if not np.array_equal(frame["epoch_index"].to_numpy(), np.arange(len(frame))):
            raise ValueError("The label-independent store must preserve the complete epoch grid.")
        expected_ids = [f"{frame.dataset.iloc[0]}:{frame.recording_id.iloc[0]}:{i:06d}" for i in range(len(frame))]
        if frame["epoch_id"].tolist() != expected_ids:
            raise ValueError("Stable epoch IDs disagree with the recording timeline.")
        if not np.allclose(frame["canonical_sfreq"], self.config.sampling_frequency, rtol=0, atol=1e-9):
            raise ValueError("Canonical sampling frequency disagrees with preprocessing.")
        if frame["original_sfreq"].nunique() != 1:
            raise ValueError("Original sampling frequency must be constant within a recording.")
        original_count = samples_per_epoch(float(frame["original_sfreq"].iloc[0]), self.config.epoch_duration)
        count = samples_per_epoch(self.config.sampling_frequency, self.config.epoch_duration)
        if self.waveforms.shape != (len(frame), count) or not np.isfinite(self.waveforms).all():
            raise ValueError("Waveforms must be finite and aligned with the complete epoch grid.")
        starts = frame["epoch_index"].to_numpy() * self.config.epoch_duration
        for column, expected in [
            ("start_time", starts), ("end_time", starts + self.config.epoch_duration),
            ("canonical_start_sample", frame["epoch_index"].to_numpy() * count),
            ("canonical_end_sample_exclusive", (frame["epoch_index"].to_numpy() + 1) * count),
            ("original_start_sample", frame["epoch_index"].to_numpy() * original_count),
        ]:
            if not np.allclose(frame[column].to_numpy(dtype=float), expected, rtol=0, atol=1e-6):
                raise ValueError(f"Misaligned {column}.")
        if not np.isfinite(frame[ALL_FEATURES].to_numpy(dtype=float)).all():
            raise ValueError("Spectral features must be finite.")
        for column in ["quality_valid", "label_valid", "supervised_valid"]:
            if not pd.api.types.is_bool_dtype(frame[column].dtype):
                raise ValueError(f"{column} must be boolean.")
        if not np.array_equal(frame["label_valid"], frame["target_stage"].isin(CLASS_LABELS)):
            raise ValueError("label_valid disagrees with the mapped stage.")
        if not np.array_equal(frame["supervised_valid"], frame["quality_valid"] & frame["label_valid"]):
            raise ValueError("supervised_valid disagrees with the QC/label masks.")
        if (frame.loc[frame["quality_valid"], ABSOLUTE_POWER_FEATURES] <= 0).any().any():
            raise ValueError("Usable features require positive absolute power for log scaling.")
        if frame["dataset"].iloc[0] == "isruc" and frame["split"].iloc[0] != "external":
            raise ValueError("ISRUC stores must be marked external for the source-only study.")
        if frame["dataset"].iloc[0] == "sleepedf" and frame["split"].iloc[0] not in {"train", "val", "test"}:
            raise ValueError("Sleep-EDF stores require a train/val/test participant assignment.")
        return self


def prepare_recording(
    raw, labels, *, dataset, participant_id, recording_id, channel_as_recorded,
    derivation_used, split, config=None, provenance=None,
):
    """Process a selected single channel without fitting or using labels for QC.

    ``labels`` maps physical ``epoch_index`` to ``raw_stage``/``target_stage``.
    Missing labels and rejected epochs remain in the store to preserve time.
    """
    config = config or PreprocessingConfig()
    if len(raw.ch_names) != 1:
        raise ValueError("Select or reconstruct exactly one EEG derivation first.")
    original_sfreq = float(raw.info["sfreq"])
    original_count = samples_per_epoch(original_sfreq, config.epoch_duration)
    original_n_epochs = raw.n_times // original_count
    if not original_n_epochs:
        raise ValueError("Recording contains no complete epochs.")
    if not np.isfinite(raw.get_data()).all():
        raise ValueError("Nonfinite continuous samples would contaminate filtering.")
    signal = raw.copy()
    if not np.isclose(original_sfreq, config.sampling_frequency, rtol=0, atol=1e-9):
        signal.resample(config.sampling_frequency, verbose=False)
    unfiltered = extract_all_eeg_epochs(signal, config.epoch_duration)
    if len(unfiltered) != original_n_epochs:
        raise ValueError("Resampling changed the number of complete physical epochs.")
    quality = np.std(unfiltered, axis=1) >= config.flat_std_threshold
    filtered = filter_eeg(signal, config.low_frequency, config.high_frequency)
    waveforms = extract_all_eeg_epochs(filtered, config.epoch_duration)
    features = extract_frequency_features_batch(
        waveforms, config.sampling_frequency, nperseg=config.welch_samples,
    )[ALL_FEATURES]
    positive = (features[ABSOLUTE_POWER_FEATURES] > 0).all(axis=1).to_numpy()
    quality &= positive
    indices = np.arange(len(waveforms), dtype=np.int64)
    count = waveforms.shape[1]
    frame = pd.DataFrame({
        "dataset": dataset, "participant_id": str(participant_id),
        "recording_id": str(recording_id), "subject_id": str(recording_id),
        "epoch_id": [f"{dataset}:{recording_id}:{i:06d}" for i in indices],
        "epoch_index": indices, "start_time": indices * config.epoch_duration,
        "end_time": (indices + 1) * config.epoch_duration,
        "original_start_sample": indices * original_count,
        "canonical_start_sample": indices * count,
        "canonical_end_sample_exclusive": (indices + 1) * count,
        "channel_as_recorded": channel_as_recorded, "derivation_used": derivation_used,
        "original_sfreq": original_sfreq, "canonical_sfreq": config.sampling_frequency,
        "signal_unit": "V", "power_unit": "V^2", "split": split,
    })
    if labels["epoch_index"].duplicated().any():
        raise ValueError("Duplicate annotation epoch indices.")
    label_indices = labels["epoch_index"].to_numpy(dtype=float)
    if not np.isfinite(label_indices).all() or np.any(label_indices < 0) or np.any(label_indices >= len(frame)) or not np.equal(label_indices, np.floor(label_indices)).all():
        raise ValueError("Annotation epochs fall outside the recording grid.")
    frame = frame.merge(labels[["epoch_index", "raw_stage", "target_stage"]], on="epoch_index", how="left", validate="one_to_one")
    frame["quality_valid"] = quality
    frame["label_valid"] = frame["target_stage"].isin(CLASS_LABELS)
    frame["supervised_valid"] = frame["quality_valid"] & frame["label_valid"]
    frame["exclusion_reason"] = np.select(
        [~quality, ~frame["label_valid"].to_numpy()],
        ["signal_quality", "unscored_or_excluded"], default="",
    )
    for column in ALL_FEATURES:
        frame[column] = features[column].to_numpy()
    return ProcessedRecording(
        frame, waveforms.astype(np.float32), config, provenance or {},
    ).validate()


def validate_feature_dataframe(frame, epoch_duration=30.0):
    """Check legacy feature caches without assuming a header proves completion."""
    required = {"subject_id", "epoch_index", "start_time", "end_time", "target_stage", *ALL_FEATURES}
    if frame.empty or required - set(frame):
        raise ValueError("Feature file is empty or missing required columns.")
    if frame[list(required)].isna().any().any():
        raise ValueError("Feature file contains missing values.")
    if frame.duplicated(["subject_id", "epoch_index"]).any():
        raise ValueError("Feature file contains duplicate epochs.")
    numeric = frame[["epoch_index", "start_time", "end_time", *ALL_FEATURES]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or not frame["target_stage"].isin(CLASS_LABELS).all():
        raise ValueError("Feature file contains nonfinite values or unknown targets.")
    indices = frame["epoch_index"].to_numpy(dtype=float)
    if np.any(indices < 0) or not np.equal(indices, np.floor(indices)).all():
        raise ValueError("Epoch indices must be nonnegative integers.")
    if not np.allclose(frame["start_time"], indices * epoch_duration, rtol=0, atol=1e-6):
        raise ValueError("Feature timeline indices disagree with start times.")
    if not np.allclose(frame["end_time"] - frame["start_time"], epoch_duration, rtol=0, atol=1e-6):
        raise ValueError("Feature file has invalid epoch durations.")
    if (frame[ABSOLUTE_POWER_FEATURES] <= 0).any().any():
        raise ValueError("Absolute powers must be positive.")
    return frame

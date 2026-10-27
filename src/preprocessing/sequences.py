import numpy as np
from numbers import Integral


CLASS_LABELS = [
    "W",
    "N1",
    "N2",
    "N3",
    "REM",
]

CLASS_TO_INDEX = {
    label: index
    for index, label in enumerate(CLASS_LABELS)
}


def find_contiguous_runs(dataframe, time_tolerance=1e-6):
    """Split one recording at time gaps, preserving its input index labels."""
    if not np.isfinite(time_tolerance) or time_tolerance < 0:
        raise ValueError("time_tolerance must be nonnegative and finite.")
    if dataframe.empty:
        return []
    ordered = dataframe.sort_values("start_time", kind="stable").copy()
    times = ordered[["start_time", "end_time"]].to_numpy(dtype=float)
    if not np.isfinite(times).all() or np.any(times[:, 1] <= times[:, 0]):
        raise ValueError("Epoch times must be finite and have positive duration.")
    gaps = times[1:, 0] - times[:-1, 1]
    if np.any(gaps < -time_tolerance):
        raise ValueError("Overlapping or duplicate epochs cannot form temporal windows.")
    boundaries = np.r_[0, np.flatnonzero(np.abs(gaps) > time_tolerance) + 1, len(ordered)]
    return [ordered.iloc[a:b].copy() for a, b in zip(boundaries[:-1], boundaries[1:])]


def build_sequence_positions(dataframe, sequence_length=10, time_tolerance=1e-6):
    """Return input-row positions without materializing overlapping EEG windows.

    Identifying columns isolate datasets, participants, recordings, and splits
    when present. Input index labels may be arbitrary or duplicated.
    """
    if isinstance(sequence_length, bool) or not isinstance(sequence_length, Integral) or sequence_length < 1:
        raise ValueError("sequence_length must be a positive integer.")
    frame = dataframe.reset_index(drop=True)
    recording = "recording_id" if "recording_id" in frame else "subject_id"
    if recording not in frame:
        raise ValueError("A recording_id or subject_id column is required.")
    groups = [c for c in ["dataset", "participant_id", recording, "split"] if c in frame]
    if frame[groups].isna().any().any():
        raise ValueError("Recording/group identifiers cannot be missing.")
    windows = []
    for _, group in frame.groupby(groups, sort=False):
        for run in find_contiguous_runs(group, time_tolerance):
            if len(run) >= sequence_length:
                windows.append(np.lib.stride_tricks.sliding_window_view(
                    run.index.to_numpy(dtype=np.int64), sequence_length,
                ).copy())
    return np.concatenate(windows) if windows else np.empty((0, sequence_length), dtype=np.int64)


def build_sequences_from_dataframe(dataframe, feature_values, sequence_length=10, time_tolerance=1e-6):
    """Build final-epoch targets from features aligned row-for-row with the frame.

    Pandas index labels may be arbitrary or duplicated. All indexing into the
    external feature matrix uses input row positions.
    """
    values = np.asarray(feature_values, dtype=np.float32)
    if values.ndim != 2 or len(dataframe) != len(values):
        raise ValueError("feature_values must be 2D and aligned row-for-row with dataframe.")
    positions = build_sequence_positions(dataframe, sequence_length, time_tolerance)
    targets = dataframe["target_stage"].to_numpy(dtype=object)[positions[:, -1]]
    return values[positions], targets


def encode_sleep_stage_labels(
    labels,
):
    """
    Convert sleep-stage strings into integer class labels.

    W   -> 0
    N1  -> 1
    N2  -> 2
    N3  -> 3
    REM -> 4
    """

    labels = np.asarray(
        labels,
        dtype=object,
    )

    unknown_labels = (
        set(labels)
        - set(CLASS_TO_INDEX)
    )

    if unknown_labels:
        raise ValueError(
            f"Unknown sleep-stage labels: "
            f"{unknown_labels}"
        )

    return np.asarray(
        [
            CLASS_TO_INDEX[label]
            for label in labels
        ],
        dtype=np.int64,
    )

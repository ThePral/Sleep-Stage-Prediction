import numpy as np
import pandas as pd


STAGE_MAPPING = {
    "Sleep stage W": "W",
    "Sleep stage 1": "N1",
    "Sleep stage 2": "N2",
    "Sleep stage 3": "N3",
    "Sleep stage 4": "N3",
    "Sleep stage R": "REM",
}


EXCLUDED_STAGES = {
    "Movement time",
    "?",
    "Sleep stage ?",
}


TARGET_STAGES = [
    "W",
    "N1",
    "N2",
    "N3",
    "REM",
]


def map_sleep_stage(raw_stage):
    """
    Convert a Sleep-EDF annotation label into the target label.

    Sleep stage 3 and Sleep stage 4 are combined into N3.

    Movement time and unknown annotations are excluded.
    """

    if raw_stage in EXCLUDED_STAGES:
        return None

    return STAGE_MAPPING.get(raw_stage)


def build_epoch_metadata(
    annotations,
    recording_id,
    recording_duration,
    epoch_duration=30.0,
    *,
    include_excluded=False,
    time_tolerance=1e-6,
):
    """
    Convert Sleep-EDF annotations into complete 30-second
    epoch metadata.

    Epochs are generated from the actual hypnogram annotation
    intervals. This preserves the annotation-based behavior
    validated during dataset exploration.

    Unknown/unusable annotations such as Movement time and
    Sleep stage ? are ignored.

    Parameters
    ----------
    annotations : mne.Annotations
        Sleep-EDF hypnogram annotations.

    recording_id : str
        Recording identifier.

    recording_duration : float
        PSG duration in seconds.

    epoch_duration : float
        Epoch duration in seconds.

    Returns
    -------
    pandas.DataFrame
        One row per valid annotated epoch.
    """

    if not np.isfinite(epoch_duration) or epoch_duration <= 0:
        raise ValueError("epoch_duration must be positive and finite.")
    if not np.isfinite(recording_duration) or recording_duration < 0:
        raise ValueError("recording_duration must be nonnegative and finite.")
    if not np.isfinite(time_tolerance) or time_tolerance < 0:
        raise ValueError("time_tolerance must be nonnegative and finite.")
    records = []

    for onset, duration, description in zip(
        annotations.onset,
        annotations.duration,
        annotations.description,
    ):

        target_stage = map_sleep_stage(
            description
        )

        # Ignore Movement time, ?, and any other
        # annotation without a target stage.
        if target_stage is None and not include_excluded:
            continue

        if not np.isfinite([onset, duration]).all() or duration < 0:
            raise ValueError("Annotation onset/duration must be finite; duration cannot be negative.")

        # Number of complete 30-second epochs
        # represented by this annotation.
        n_epochs = int(
            np.floor(
                (min(duration, recording_duration - onset) + time_tolerance) / epoch_duration
            )
        )

        for i in range(n_epochs):

            start_time = (
                onset
                + i * epoch_duration
            )

            end_time = (
                start_time
                + epoch_duration
            )

            if start_time < -time_tolerance:
                continue
            # The physical timeline, not the retained row number, determines
            # which EEG samples belong to this label after excluded intervals.
            epoch_index = int(round(start_time / epoch_duration))
            grid_start = epoch_index * epoch_duration
            if abs(start_time - grid_start) > time_tolerance:
                raise ValueError(
                    f"Annotation at {start_time}s in {recording_id} is not "
                    f"aligned to the {epoch_duration}s EEG grid."
                )
            start_time = grid_start
            end_time = grid_start + epoch_duration

            # Do not create an epoch extending
            # beyond the actual PSG.
            if end_time > recording_duration + time_tolerance:
                continue

            records.append(
                {
                    "subject_id": recording_id,
                    "epoch_index": epoch_index,
                    "start_time": start_time,
                    "end_time": end_time,
                    "raw_stage": description,
                    "target_stage": target_stage,
                }
            )

    result = pd.DataFrame(records, columns=[
        "subject_id", "epoch_index", "start_time", "end_time", "raw_stage", "target_stage",
    ])
    if result["epoch_index"].duplicated().any():
        raise ValueError(f"Overlapping annotation epochs in {recording_id}.")
    return result.sort_values("epoch_index", kind="stable").reset_index(drop=True)

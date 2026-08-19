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

    records = []

    epoch_index = 0

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
        if target_stage is None:
            continue

        # Number of complete 30-second epochs
        # represented by this annotation.
        n_epochs = int(
            np.floor(
                duration / epoch_duration
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

            # Do not create an epoch extending
            # beyond the actual PSG.
            if end_time > recording_duration:
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

            epoch_index += 1

    return pd.DataFrame(records)
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

    Movement time and unknown annotations are excluded
    from the classification dataset.
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
    Build 30-second epoch metadata from Sleep-EDF annotations.

    Only complete epochs inside the PSG recording are retained.

    Unknown/unusable annotations are ignored.

    Parameters
    ----------
    annotations : mne.Annotations
        Sleep-EDF hypnogram annotations.

    recording_id : str
        Recording identifier, e.g. SC4001.

    recording_duration : float
        PSG duration in seconds.

    epoch_duration : float
        Epoch duration in seconds. Default is 30 seconds.

    Returns
    -------
    pandas.DataFrame
        One row per complete epoch.
    """

    n_epochs = int(
        np.floor(
            recording_duration / epoch_duration
        )
    )

    rows = []

    for epoch_index in range(n_epochs):

        start_time = (
            epoch_index * epoch_duration
        )

        end_time = (
            start_time + epoch_duration
        )

        raw_stage = None

        # Find the annotation covering this epoch.
        for onset, duration, description in zip(
            annotations.onset,
            annotations.duration,
            annotations.description,
        ):

            annotation_start = onset
            annotation_end = onset + duration

            if (
                start_time >= annotation_start
                and end_time <= annotation_end
            ):
                raw_stage = description
                break

        target_stage = map_sleep_stage(
            raw_stage
        )

        # Ignore unknown stages such as "?"
        # and annotations outside the useful PSG range.
        if target_stage is None:
            continue

        rows.append(
            {
                "subject_id": recording_id,
                "epoch_index": epoch_index,
                "start_time": start_time,
                "end_time": end_time,
                "raw_stage": raw_stage,
                "target_stage": target_stage,
            }
        )

    return pd.DataFrame(rows)
import numpy as np
import pandas as pd
import mne

def build_epoch_metadata(
    annotations,
    recording_id,
    recording_duration,
    epoch_duration=30.0
):
    """
    Convert Sleep-EDF annotations into complete 30-second
    epoch metadata.

    Unknown/unusable annotations such as "Sleep stage ?"
    are ignored.
    """

    stage_mapping = {
        "Sleep stage W": "W",
        "Sleep stage 1": "N1",
        "Sleep stage 2": "N2",
        "Sleep stage 3": "N3",
        "Sleep stage 4": "N3",
        "Sleep stage R": "REM",
    }

    records = []
    epoch_index = 0

    for onset, duration, description in zip(
        annotations.onset,
        annotations.duration,
        annotations.description
    ):

        if description not in stage_mapping:
            continue

        target_stage = stage_mapping[description]

        # Number of complete 30-second epochs
        n_epochs = int(duration // epoch_duration)

        for i in range(n_epochs):

            start_time = onset + i * epoch_duration
            end_time = start_time + epoch_duration

            # Only keep epochs fully contained in the PSG
            if end_time > recording_duration:
                continue

            records.append({
                "subject_id": recording_id,
                "epoch_index": epoch_index,
                "start_time": start_time,
                "end_time": end_time,
                "raw_stage": description,
                "target_stage": target_stage,
            })

            epoch_index += 1

    return pd.DataFrame(records)
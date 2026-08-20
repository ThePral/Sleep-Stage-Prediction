import numpy as np
import pandas as pd


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


def find_contiguous_runs(
    dataframe,
    time_tolerance=1e-6,
):
    """
    Split a subject's epochs into temporally contiguous runs.

    The original dataframe indices are preserved so that the
    returned runs can safely be used to index an external
    feature matrix aligned with the original dataframe.
    """

    if dataframe.empty:
        return []

    sorted_df = (
        dataframe
        .sort_values("start_time")
        .copy()
    )

    runs = []

    run_start_position = 0

    for position in range(
        1,
        len(sorted_df),
    ):

        previous_end = sorted_df.iloc[
            position - 1
        ]["end_time"]

        current_start = sorted_df.iloc[
            position
        ]["start_time"]

        is_contiguous = (
            abs(
                current_start
                - previous_end
            )
            <= time_tolerance
        )

        if not is_contiguous:

            runs.append(
                sorted_df.iloc[
                    run_start_position:position
                ].copy()
            )

            run_start_position = position

    runs.append(
        sorted_df.iloc[
            run_start_position:
        ].copy()
    )

    return runs


def build_sequences_from_dataframe(
    dataframe,
    feature_values,
    sequence_length=10,
    time_tolerance=1e-6,
):
    """
    Build many-to-one temporal sequences.

    Each sequence contains `sequence_length` consecutive
    epochs from the same subject and the same contiguous
    time run.

    The target is the stage of the final epoch.

    `feature_values` must be aligned row-for-row with
    `dataframe`.
    """

    if len(dataframe) != len(feature_values):
        raise ValueError(
            "dataframe and feature_values must "
            "contain the same number of rows."
        )

    if sequence_length < 1:
        raise ValueError(
            "sequence_length must be >= 1."
        )

    working_df = dataframe.copy()

    feature_values = np.asarray(
        feature_values,
        dtype=np.float32,
    )

    sequences = []
    targets = []

    for subject_id, subject_df in (
        working_df.groupby(
            "subject_id",
            sort=False,
        )
    ):

        runs = find_contiguous_runs(
            subject_df,
            time_tolerance=time_tolerance,
        )

        for run in runs:

            if len(run) < sequence_length:
                continue

            # IMPORTANT:
            # These are the ORIGINAL dataframe indices.
            run_positions = run.index.to_numpy(
                dtype=int
            )

            for start in range(
                0,
                len(run_positions)
                - sequence_length
                + 1,
            ):

                window_positions = (
                    run_positions[
                        start:
                        start + sequence_length
                    ]
                )

                sequence = feature_values[
                    window_positions
                ]

                target_position = (
                    window_positions[-1]
                )

                target_stage = working_df.loc[
                    target_position,
                    "target_stage",
                ]

                sequences.append(
                    sequence
                )

                targets.append(
                    target_stage
                )

    if not sequences:

        n_features = (
            feature_values.shape[1]
            if feature_values.ndim == 2
            else 0
        )

        return (
            np.empty(
                (
                    0,
                    sequence_length,
                    n_features,
                ),
                dtype=np.float32,
            ),
            np.empty(
                (0,),
                dtype=object,
            ),
        )

    return (
        np.stack(sequences).astype(
            np.float32
        ),
        np.asarray(
            targets,
            dtype=object,
        ),
    )


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
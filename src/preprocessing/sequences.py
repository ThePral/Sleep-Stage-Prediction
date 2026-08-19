import numpy as np
import pandas as pd


def find_contiguous_runs(
    dataframe,
    time_tolerance=1e-6,
):
    """
    Split a subject's epochs into temporally contiguous runs.

    Two consecutive rows belong to the same run only if:

        next.start_time ≈ current.end_time

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Must contain:
            subject_id
            start_time
            end_time

    time_tolerance : float
        Allowed floating-point tolerance in seconds.

    Returns
    -------
    list[pandas.DataFrame]
        Contiguous temporal runs.
    """

    if dataframe.empty:
        return []

    dataframe = (
        dataframe
        .sort_values("start_time")
        .reset_index(drop=True)
    )

    runs = []

    run_start = 0

    for i in range(1, len(dataframe)):

        previous_end = dataframe.loc[
            i - 1,
            "end_time",
        ]

        current_start = dataframe.loc[
            i,
            "start_time",
        ]

        is_contiguous = (
            abs(
                current_start
                - previous_end
            )
            <= time_tolerance
        )

        if not is_contiguous:

            runs.append(
                dataframe.iloc[
                    run_start:i
                ].copy()
            )

            run_start = i

    runs.append(
        dataframe.iloc[
            run_start:
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

    Each sequence contains `sequence_length`
    consecutive epochs from the same subject.

    The target is the sleep-stage label of
    the final epoch in the sequence.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        Epoch metadata and labels.

    feature_values : numpy.ndarray
        Normalized feature matrix aligned row-for-row
        with `dataframe`.

    sequence_length : int
        Number of time steps per sequence.

    time_tolerance : float
        Allowed timing tolerance.

    Returns
    -------
    X : numpy.ndarray
        Shape:
            (n_sequences, sequence_length, n_features)

    y : numpy.ndarray
        Shape:
            (n_sequences,)
    """

    if len(dataframe) != len(
        feature_values
    ):
        raise ValueError(
            "dataframe and feature_values "
            "must contain the same number "
            "of rows."
        )

    if sequence_length < 1:
        raise ValueError(
            "sequence_length must be >= 1."
        )

    # Work with dataframe row positions.
    working_df = (
        dataframe
        .reset_index(drop=True)
        .copy()
    )

    feature_values = np.asarray(
        feature_values,
        dtype=np.float32,
    )

    sequences = []
    targets = []

    # Process each subject independently.
    for subject_id, subject_df in (
        working_df
        .groupby("subject_id", sort=False)
    ):

        subject_positions = (
            subject_df.index.to_numpy()
        )

        # Split the subject into contiguous
        # temporal runs.
        runs = find_contiguous_runs(
            subject_df,
            time_tolerance=time_tolerance,
        )

        for run in runs:

            if len(run) < sequence_length:
                continue

            run_positions = (
                run.index.to_numpy()
            )

            for start in range(
                0,
                len(run) - sequence_length + 1,
            ):

                window_positions = (
                    run_positions[
                        start:
                        start + sequence_length
                    ]
                )

                X_sequence = (
                    feature_values[
                        window_positions
                    ]
                )

                target_position = (
                    window_positions[-1]
                )

                target_stage = (
                    working_df.loc[
                        target_position,
                        "target_stage",
                    ]
                )

                sequences.append(
                    X_sequence
                )

                targets.append(
                    target_stage
                )

    if not sequences:
        return (
            np.empty(
                (
                    0,
                    sequence_length,
                    feature_values.shape[1],
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
    

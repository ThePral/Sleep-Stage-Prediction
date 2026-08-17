import numpy as np


DEFAULT_FLAT_STD_THRESHOLD = 1e-10


def extract_eeg_epochs(
    eeg_raw,
    epochs_df,
):
    """
    Extract EEG samples corresponding to each epoch.

    Parameters
    ----------
    eeg_raw : mne.io.Raw
        Continuous EEG recording.

    epochs_df : pandas.DataFrame
        Epoch metadata containing start_time and end_time.

    Returns
    -------
    np.ndarray
        EEG epochs with shape:
        (n_epochs, n_samples)
    """

    sfreq = eeg_raw.info["sfreq"]

    epochs = []

    for _, row in epochs_df.iterrows():

        start_sample = int(
            round(
                row["start_time"] * sfreq
            )
        )

        end_sample = int(
            round(
                row["end_time"] * sfreq
            )
        )

        epoch = eeg_raw.get_data(
            start=start_sample,
            stop=end_sample,
        )[0]

        if len(epoch) == 0:
            raise ValueError(
                f"Empty EEG epoch at "
                f"epoch_index={row['epoch_index']}"
            )

        epochs.append(epoch)

    if not epochs:
        return np.empty(
            (0, 0),
            dtype=np.float64,
        )

    return np.asarray(
        epochs,
        dtype=np.float64,
    )


def find_valid_epoch_indices(
    eeg_raw,
    epochs_df,
    flat_std_threshold=DEFAULT_FLAT_STD_THRESHOLD,
):
    """
    Identify epochs whose raw EEG is not flat or near-flat.

    Returns
    -------
    np.ndarray
        Row indices of valid epochs in epochs_df.
    """

    valid_indices = []

    sfreq = eeg_raw.info["sfreq"]

    for dataframe_index, (_, row) in enumerate(
        epochs_df.iterrows()
    ):

        start_sample = int(
            round(
                row["start_time"] * sfreq
            )
        )

        end_sample = int(
            round(
                row["end_time"] * sfreq
            )
        )

        epoch = eeg_raw.get_data(
            start=start_sample,
            stop=end_sample,
        )[0]

        if len(epoch) == 0:
            continue

        epoch_std = np.std(epoch)

        if epoch_std < flat_std_threshold:
            continue

        valid_indices.append(
            dataframe_index
        )

    return np.asarray(
        valid_indices,
        dtype=int,
    )
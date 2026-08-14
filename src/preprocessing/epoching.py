import numpy as np


def extract_eeg_epochs(
    eeg_raw,
    epochs_df,
):
    """
    Extract EEG samples corresponding to each epoch.

    Parameters
    ----------
    eeg_raw : mne.io.Raw
        Filtered continuous EEG recording containing one EEG channel.

    epochs_df : pandas.DataFrame
        Epoch metadata containing start_time and end_time.

    Returns
    -------
    numpy.ndarray
        Array with shape:

        (number_of_epochs, samples_per_epoch)
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
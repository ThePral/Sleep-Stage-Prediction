import numpy as np


DEFAULT_FLAT_STD_THRESHOLD = 1e-10


def extract_all_eeg_epochs(
    eeg_raw,
    epoch_duration=30.0,
):
    """
    Extract all complete 30-second epochs from a continuous EEG
    recording in one operation.

    Returns
    -------
    np.ndarray
        Shape:
        (n_epochs, samples_per_epoch)
    """

    sfreq = eeg_raw.info["sfreq"]

    epoch_samples = int(
        epoch_duration * sfreq
    )

    total_samples = eeg_raw.n_times

    n_complete_epochs = (
        total_samples // epoch_samples
    )

    if n_complete_epochs == 0:
        return np.empty(
            (0, epoch_samples),
            dtype=np.float64,
        )

    usable_samples = (
        n_complete_epochs * epoch_samples
    )

    # Load the continuous signal only once.
    signal = eeg_raw.get_data(
        start=0,
        stop=usable_samples,
    )[0]

    return signal.reshape(
        n_complete_epochs,
        epoch_samples,
    )


def find_valid_epoch_indices(
    eeg_raw,
    epochs_df,
    epoch_duration=30.0,
    flat_std_threshold=DEFAULT_FLAT_STD_THRESHOLD,
):
    """
    Identify valid EEG epochs using a raw-signal
    standard-deviation threshold.

    The continuous EEG is loaded once and reshaped into
    30-second epochs, making the quality check much faster
    than repeatedly calling get_data() for each epoch.

    Returns
    -------
    np.ndarray
        Original epoch indices from epochs_df that passed
        the quality check.
    """

    all_epochs = extract_all_eeg_epochs(
        eeg_raw=eeg_raw,
        epoch_duration=epoch_duration,
    )

    if len(all_epochs) == 0:
        return np.empty(
            (0,),
            dtype=int,
        )

    epoch_stds = np.std(
        all_epochs,
        axis=1,
    )

    # epochs_df["epoch_index"] refers to the original
    # 30-second timeline, so use it directly.
    epoch_indices = (
        epochs_df["epoch_index"]
        .to_numpy(dtype=int)
    )

    # Keep only indices that are actually present
    # in the loaded EEG recording.
    valid_timeline_mask = (
        epoch_indices < len(all_epochs)
    )

    epoch_indices = (
        epoch_indices[
            valid_timeline_mask
        ]
    )

    valid_mask = (
        epoch_stds[epoch_indices]
        >= flat_std_threshold
    )

    return epoch_indices[
        valid_mask
    ]


def extract_eeg_epochs_by_indices(
    eeg_raw,
    epoch_indices,
    epoch_duration=30.0,
):
    """
    Extract selected epochs from a continuous EEG recording.

    The continuous signal is loaded only once.

    Parameters
    ----------
    eeg_raw : mne.io.Raw
        Continuous EEG recording.

    epoch_indices : array-like
        Original 30-second epoch indices.

    epoch_duration : float
        Epoch duration in seconds.

    Returns
    -------
    np.ndarray
        Selected EEG epochs.
    """

    all_epochs = extract_all_eeg_epochs(
        eeg_raw=eeg_raw,
        epoch_duration=epoch_duration,
    )

    epoch_indices = np.asarray(
        epoch_indices,
        dtype=int,
    )

    if len(epoch_indices) == 0:
        return np.empty(
            (0, 0),
            dtype=np.float64,
        )

    if np.any(
        epoch_indices >= len(all_epochs)
    ):
        raise IndexError(
            "One or more requested epoch indices "
            "are outside the available EEG data."
        )

    return all_epochs[
        epoch_indices
    ]
import numpy as np


DEFAULT_FLAT_STD_THRESHOLD = 1e-10


def samples_per_epoch(sfreq, epoch_duration=30.0):
    value = float(sfreq) * float(epoch_duration)
    if not np.isfinite([sfreq, epoch_duration, value]).all() or sfreq <= 0 or epoch_duration <= 0:
        raise ValueError("Sampling frequency and epoch duration must be positive and finite.")
    count = int(round(value))
    if count < 1 or not np.isclose(value, count, rtol=0, atol=1e-6):
        raise ValueError("An epoch must contain a positive integer number of samples.")
    return count


def validate_epoch_indices(epoch_indices, n_epochs):
    values = np.asarray(epoch_indices)
    if values.ndim != 1:
        raise ValueError("epoch_indices must be one-dimensional.")
    try:
        numeric = values.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("epoch_indices must contain finite integers.") from exc
    if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
        raise ValueError("epoch_indices must contain finite integers.")
    if np.any(numeric < 0) or np.any(numeric >= n_epochs):
        raise IndexError("Requested epoch indices are outside the available EEG data.")
    return numeric.astype(np.int64)


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

    if len(eeg_raw.ch_names) != 1:
        raise ValueError("Select exactly one EEG channel before epoch extraction.")
    epoch_samples = samples_per_epoch(sfreq, epoch_duration)

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

    if not np.isfinite(flat_std_threshold) or flat_std_threshold < 0:
        raise ValueError("flat_std_threshold must be nonnegative and finite.")
    all_epochs = extract_all_eeg_epochs(
        eeg_raw=eeg_raw,
        epoch_duration=epoch_duration,
    )

    epoch_stds = np.std(
        all_epochs,
        axis=1,
    )

    # epochs_df["epoch_index"] refers to the original
    # 30-second timeline, so use it directly.
    epoch_indices = validate_epoch_indices(epochs_df["epoch_index"].to_numpy(), len(all_epochs))
    for column, expected in [
        ("start_time", epoch_indices * epoch_duration),
        ("end_time", (epoch_indices + 1) * epoch_duration),
    ]:
        if column in epochs_df and not np.allclose(
            epochs_df[column].to_numpy(dtype=float), expected, rtol=0, atol=1e-6,
        ):
            raise ValueError(f"{column} disagrees with absolute epoch_index values.")

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
    valid_mask &= np.isfinite(all_epochs[epoch_indices]).all(axis=1)

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

    epoch_indices = validate_epoch_indices(epoch_indices, len(all_epochs))

    if len(epoch_indices) == 0:
        return np.empty(
            (0, all_epochs.shape[1]),
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

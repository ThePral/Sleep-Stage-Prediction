import mne


def filter_eeg(
    raw,
    low_freq=0.3,
    high_freq=35.0,
):
    """
    Apply a zero-phase FIR band-pass filter to the EEG.

    Filtering is performed on the continuous EEG signal
    before epoch extraction.

    Parameters
    ----------
    raw : mne.io.Raw
        Continuous EEG recording.

    low_freq : float
        Lower cutoff frequency in Hz.

    high_freq : float
        Upper cutoff frequency in Hz.

    Returns
    -------
    mne.io.Raw
        Filtered copy of the recording.
    """

    filtered_raw = raw.copy()

    filtered_raw.filter(
        l_freq=low_freq,
        h_freq=high_freq,
        method="fir",
        phase="zero",
        verbose=False,
    )

    return filtered_raw
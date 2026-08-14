import mne


def filter_eeg(
    raw,
    low_freq=0.3,
    high_freq=35.0
):
    """
    Apply a zero-phase FIR band-pass filter
    to the continuous EEG signal.
    """

    filtered_raw = raw.copy()

    filtered_raw.filter(
        l_freq=low_freq,
        h_freq=high_freq,
        method="fir",
        phase="zero",
        verbose=False
    )

    return filtered_raw
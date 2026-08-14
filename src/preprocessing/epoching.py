import numpy as np
import pandas as pd
import mne

EPOCH_DURATION = 30.0

def extract_eeg_epochs(eeg_raw, epochs_df):
    """
    Extract EEG samples for every epoch in epochs_df.

    Parameters
    ----------
    eeg_raw : mne.io.Raw
        Preloaded raw EEG recording.
    epochs_df : pandas.DataFrame
        DataFrame containing start_time and end_time for each epoch.

    Returns
    -------
    X : numpy.ndarray
        EEG epochs with shape:
        (number_of_epochs, samples_per_epoch)
    """
    
    sfreq = eeg_raw.info["sfreq"]
    epoch_samples = int(EPOCH_DURATION * sfreq)

    X = []

    for _, row in epochs_df.iterrows():
        start_sample = int(round(row["start_time"] * sfreq))
        end_sample = start_sample + epoch_samples

        epoch = eeg_raw.get_data(
            start=start_sample,
            stop=end_sample
        )[0]

        # Validate epoch length
        if len(epoch) != epoch_samples:
            raise ValueError(
                f"Unexpected epoch length for epoch "
                f"{row['epoch_index']}: "
                f"expected {epoch_samples}, got {len(epoch)}"
            )

        X.append(epoch)

    return np.asarray(X)
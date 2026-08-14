import mne
import numpy as np

from src.preprocessing.annotations import (
    build_epoch_metadata,
)

from src.preprocessing.epoching import (
    extract_eeg_epochs,
)

from src.preprocessing.filtering import (
    filter_eeg,
)

from src.features.spectral import (
    FREQUENCY_BANDS,
    ANALYSIS_LOW,
    ANALYSIS_HIGH,
    extract_frequency_features_batch,
)


def process_sleep_edf_subject(
    psg_path,
    hypnogram_path,
    recording_id,
    eeg_channel="EEG Fpz-Cz",
    low_freq=0.3,
    high_freq=35.0,
    epoch_duration=30.0,
    nperseg=1024,
):
    """
    Complete processing pipeline for one Sleep-EDF recording.

    Pipeline:

        PSG
        ↓
        Load EEG
        ↓
        Load hypnogram
        ↓
        Build epoch metadata
        ↓
        Filter continuous EEG
        ↓
        Extract 30-second epochs
        ↓
        Extract spectral features
        ↓
        Merge metadata + features

    Returns
    -------
    pandas.DataFrame
        One row per valid epoch.
    """

    # ========================================================
    # 1. Load EEG
    # ========================================================

    raw = mne.io.read_raw_edf(
        psg_path,
        include=[eeg_channel],
        preload=True,
        verbose=False,
    )

    sfreq = raw.info["sfreq"]

    # ========================================================
    # 2. Recording duration
    # ========================================================

    recording_duration = (
        raw.n_times / sfreq
    )

    # ========================================================
    # 3. Load hypnogram annotations
    # ========================================================

    annotations = mne.read_annotations(
        hypnogram_path
    )

    # ========================================================
    # 4. Build epoch metadata
    # ========================================================

    epochs_df = build_epoch_metadata(
        annotations=annotations,
        recording_id=recording_id,
        recording_duration=recording_duration,
        epoch_duration=epoch_duration,
    )

    # ========================================================
    # 5. Filter continuous EEG
    # ========================================================

    filtered_raw = filter_eeg(
        raw,
        low_freq=low_freq,
        high_freq=high_freq,
    )

    # ========================================================
    # 6. Extract EEG epochs
    # ========================================================

    eeg_epochs = extract_eeg_epochs(
        eeg_raw=filtered_raw,
        epochs_df=epochs_df,
    )

    # ========================================================
    # 7. Extract spectral features
    # ========================================================

    features_df = (
        extract_frequency_features_batch(
            epochs=eeg_epochs,
            sfreq=sfreq,
            frequency_bands=FREQUENCY_BANDS,
            analysis_low=ANALYSIS_LOW,
            analysis_high=ANALYSIS_HIGH,
            nperseg=nperseg,
        )
    )

    # ========================================================
    # 8. Add epoch index
    # ========================================================

    features_df.insert(
        0,
        "epoch_index",
        np.arange(
            len(features_df)
        ),
    )

    # ========================================================
    # 9. Merge metadata and features
    # ========================================================

    processed_df = epochs_df.merge(
        features_df,
        on="epoch_index",
        how="inner",
    )

    return processed_df
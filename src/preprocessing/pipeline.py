import numpy as np
import mne

from src.preprocessing.annotations import (
    build_epoch_metadata,
)

from src.preprocessing.epoching import (
    extract_eeg_epochs,
    find_valid_epoch_indices,
    DEFAULT_FLAT_STD_THRESHOLD,
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
    flat_std_threshold=DEFAULT_FLAT_STD_THRESHOLD,
):
    """
    Process one Sleep-EDF recording into a labeled
    spectral-feature DataFrame.

    Flat/near-flat raw EEG epochs are excluded.
    """

    # --------------------------------------------------------
    # 1. Load EEG
    # --------------------------------------------------------

    raw = mne.io.read_raw_edf(
        psg_path,
        include=[eeg_channel],
        preload=True,
        verbose=False,
    )

    sfreq = raw.info["sfreq"]

    # --------------------------------------------------------
    # 2. Recording duration
    # --------------------------------------------------------

    recording_duration = (
        raw.n_times / sfreq
    )

    # --------------------------------------------------------
    # 3. Load hypnogram
    # --------------------------------------------------------

    annotations = mne.read_annotations(
        hypnogram_path
    )

    # --------------------------------------------------------
    # 4. Build annotation-aligned epochs
    # --------------------------------------------------------

    epochs_df = build_epoch_metadata(
        annotations=annotations,
        recording_id=recording_id,
        recording_duration=recording_duration,
        epoch_duration=epoch_duration,
    )

    if epochs_df.empty:
        raise ValueError(
            f"No labeled epochs found "
            f"for {recording_id}."
        )

    # --------------------------------------------------------
    # 5. Detect flat raw EEG epochs
    # --------------------------------------------------------

    valid_indices = find_valid_epoch_indices(
        eeg_raw=raw,
        epochs_df=epochs_df,
        flat_std_threshold=flat_std_threshold,
    )

    if len(valid_indices) == 0:
        raise ValueError(
            f"No valid EEG epochs found "
            f"for {recording_id}."
        )

    valid_epochs_df = (
        epochs_df.iloc[
            valid_indices
        ]
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # 6. Filter continuous EEG
    # --------------------------------------------------------

    filtered_raw = filter_eeg(
        raw,
        low_freq=low_freq,
        high_freq=high_freq,
    )

    # --------------------------------------------------------
    # 7. Extract exactly the valid epochs
    # --------------------------------------------------------

    filtered_epochs = (
        extract_eeg_epochs(
            eeg_raw=filtered_raw,
            epochs_df=valid_epochs_df,
        )
    )

    expected_shape = (
        len(valid_epochs_df),
        int(
            epoch_duration * sfreq
        ),
    )

    if filtered_epochs.shape != expected_shape:
        raise RuntimeError(
            f"Unexpected EEG epoch shape for "
            f"{recording_id}: "
            f"{filtered_epochs.shape}; "
            f"expected {expected_shape}"
        )

    # --------------------------------------------------------
    # 8. Extract spectral features
    # --------------------------------------------------------

    features_df = (
        extract_frequency_features_batch(
            epochs=filtered_epochs,
            sfreq=sfreq,
            frequency_bands=FREQUENCY_BANDS,
            analysis_low=ANALYSIS_LOW,
            analysis_high=ANALYSIS_HIGH,
            nperseg=nperseg,
        )
    )

    # --------------------------------------------------------
    # 9. Preserve original epoch indices
    # --------------------------------------------------------

    features_df.insert(
        0,
        "epoch_index",
        valid_epochs_df[
            "epoch_index"
        ].to_numpy(),
    )

    # --------------------------------------------------------
    # 10. Merge metadata + features
    # --------------------------------------------------------

    processed_df = valid_epochs_df.merge(
        features_df,
        on="epoch_index",
        how="inner",
    )

    return processed_df
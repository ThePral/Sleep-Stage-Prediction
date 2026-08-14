import numpy as np
import pandas as pd
from scipy.signal import welch


# ============================================================
# Frequency configuration
# ============================================================

FREQUENCY_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}

ANALYSIS_LOW = 0.5
ANALYSIS_HIGH = 30.0


# ============================================================
# Basic spectral helpers
# ============================================================

def compute_psd(
    epoch,
    sfreq,
    nperseg=1024,
):
    """
    Compute the Power Spectral Density using Welch's method.

    Parameters
    ----------
    epoch : numpy.ndarray
        One EEG epoch.

    sfreq : float
        Sampling frequency.

    nperseg : int
        Number of samples per Welch segment.

    Returns
    -------
    frequencies : numpy.ndarray
    psd : numpy.ndarray
    """

    frequencies, psd = welch(
        epoch,
        fs=sfreq,
        nperseg=min(
            nperseg,
            len(epoch),
        ),
    )

    return frequencies, psd


def band_power(
    frequencies,
    psd,
    low_freq,
    high_freq,
):
    """
    Calculate absolute power inside a frequency band.
    """

    mask = (
        (frequencies >= low_freq)
        & (frequencies < high_freq)
    )

    if not np.any(mask):
        return 0.0

    return np.trapezoid(
        psd[mask],
        frequencies[mask],
    )


def compute_band_powers(
    frequencies,
    psd,
    frequency_bands,
):
    """
    Compute absolute power for all configured bands.
    """

    powers = {}

    for band_name, (
        low_freq,
        high_freq,
    ) in frequency_bands.items():

        powers[
            f"{band_name}_absolute"
        ] = band_power(
            frequencies,
            psd,
            low_freq,
            high_freq,
        )

    return powers


def compute_relative_band_powers(
    absolute_powers,
):
    """
    Convert absolute band powers into relative powers.
    """

    total_power = sum(
        absolute_powers.values()
    )

    relative_powers = {}

    if total_power <= 0:
        for column in absolute_powers:
            band_name = column.replace(
                "_absolute",
                "",
            )

            relative_powers[
                f"{band_name}_relative"
            ] = 0.0

        return relative_powers

    for column, value in absolute_powers.items():

        band_name = column.replace(
            "_absolute",
            "",
        )

        relative_powers[
            f"{band_name}_relative"
        ] = value / total_power

    return relative_powers


# ============================================================
# Spectral entropy
# ============================================================

def compute_spectral_entropy(
    psd,
):
    """
    Calculate normalized spectral entropy.

    The PSD is converted into a probability distribution,
    then Shannon entropy is normalized to [0, 1].
    """

    psd = np.asarray(
        psd,
        dtype=np.float64,
    )

    total_power = np.sum(psd)

    if total_power <= 0:
        return 0.0

    probability = psd / total_power

    probability = probability[
        probability > 0
    ]

    entropy = -np.sum(
        probability
        * np.log2(probability)
    )

    normalization = np.log2(
        len(probability)
    )

    if normalization == 0:
        return 0.0

    return entropy / normalization


# ============================================================
# Dominant frequency
# ============================================================

def compute_dominant_frequency(
    frequencies,
    psd,
    analysis_low=ANALYSIS_LOW,
    analysis_high=ANALYSIS_HIGH,
):
    """
    Find the frequency with the highest PSD
    inside the analysis range.
    """

    mask = (
        (frequencies >= analysis_low)
        & (frequencies <= analysis_high)
    )

    if not np.any(mask):
        return 0.0

    analysis_psd = psd[mask]
    analysis_frequencies = frequencies[
        mask
    ]

    index = np.argmax(
        analysis_psd
    )

    return analysis_frequencies[index]


# ============================================================
# Single epoch
# ============================================================

def extract_frequency_features(
    epoch,
    sfreq,
    frequency_bands=FREQUENCY_BANDS,
    analysis_low=ANALYSIS_LOW,
    analysis_high=ANALYSIS_HIGH,
    nperseg=1024,
):
    """
    Extract all spectral features from one EEG epoch.

    Returns
    -------
    dict
        Absolute band powers,
        relative band powers,
        spectral entropy,
        dominant frequency.
    """

    frequencies, psd = compute_psd(
        epoch,
        sfreq,
        nperseg=nperseg,
    )

    # Restrict PSD to the main analysis range.
    analysis_mask = (
        (frequencies >= analysis_low)
        & (frequencies <= analysis_high)
    )

    analysis_frequencies = frequencies[
        analysis_mask
    ]

    analysis_psd = psd[
        analysis_mask
    ]

    absolute_powers = compute_band_powers(
        analysis_frequencies,
        analysis_psd,
        frequency_bands,
    )

    relative_powers = (
        compute_relative_band_powers(
            absolute_powers
        )
    )

    spectral_entropy = (
        compute_spectral_entropy(
            analysis_psd
        )
    )

    dominant_frequency = (
        compute_dominant_frequency(
            frequencies,
            psd,
            analysis_low,
            analysis_high,
        )
    )

    features = {}

    features.update(
        absolute_powers
    )

    features.update(
        relative_powers
    )

    features[
        "spectral_entropy"
    ] = spectral_entropy

    features[
        "dominant_frequency"
    ] = dominant_frequency

    return features


# ============================================================
# Batch / vectorized-style extraction
# ============================================================

def extract_frequency_features_batch(
    epochs,
    sfreq,
    frequency_bands=FREQUENCY_BANDS,
    analysis_low=ANALYSIS_LOW,
    analysis_high=ANALYSIS_HIGH,
    nperseg=1024,
):
    """
    Extract spectral features for multiple EEG epochs.

    Parameters
    ----------
    epochs : numpy.ndarray
        Shape:

        (number_of_epochs, samples_per_epoch)

    sfreq : float
        Sampling frequency.

    Returns
    -------
    pandas.DataFrame
        One row per epoch.
    """

    epochs = np.asarray(
        epochs,
        dtype=np.float64,
    )

    if epochs.ndim != 2:
        raise ValueError(
            "epochs must be a 2D array "
            "with shape "
            "(n_epochs, n_samples)."
        )

    n_epochs = epochs.shape[0]

    if n_epochs == 0:
        columns = []

        for band in frequency_bands:
            columns.append(
                f"{band}_absolute"
            )

        for band in frequency_bands:
            columns.append(
                f"{band}_relative"
            )

        columns.extend(
            [
                "spectral_entropy",
                "dominant_frequency",
            ]
        )

        return pd.DataFrame(
            columns=columns
        )

    # --------------------------------------------------------
    # Welch PSD for all epochs
    # --------------------------------------------------------

    frequencies, psd = welch(
        epochs,
        fs=sfreq,
        nperseg=min(
            nperseg,
            epochs.shape[1],
        ),
        axis=1,
    )

    # --------------------------------------------------------
    # Analysis frequency range
    # --------------------------------------------------------

    analysis_mask = (
        (frequencies >= analysis_low)
        & (frequencies <= analysis_high)
    )

    analysis_frequencies = frequencies[
        analysis_mask
    ]

    analysis_psd = psd[
        :,
        analysis_mask,
    ]

    # --------------------------------------------------------
    # Absolute band powers
    # --------------------------------------------------------

    feature_data = {}

    for band_name, (
        low_freq,
        high_freq,
    ) in frequency_bands.items():

        band_mask = (
            (analysis_frequencies >= low_freq)
            & (
                analysis_frequencies
                < high_freq
            )
        )

        if np.any(band_mask):

            feature_data[
                f"{band_name}_absolute"
            ] = np.trapezoid(
                analysis_psd[
                    :,
                    band_mask,
                ],
                analysis_frequencies[
                    band_mask
                ],
                axis=1,
            )

        else:

            feature_data[
                f"{band_name}_absolute"
            ] = np.zeros(
                n_epochs,
                dtype=np.float64,
            )

    # --------------------------------------------------------
    # Relative band powers
    # --------------------------------------------------------

    total_power = np.sum(
        np.stack(
            [
                feature_data[
                    f"{band}_absolute"
                ]
                for band in frequency_bands
            ],
            axis=1,
        ),
        axis=1,
    )

    for band_name in frequency_bands:

        absolute_column = (
            f"{band_name}_absolute"
        )

        relative_column = (
            f"{band_name}_relative"
        )

        feature_data[
            relative_column
        ] = np.divide(
            feature_data[
                absolute_column
            ],
            total_power,
            out=np.zeros_like(
                feature_data[
                    absolute_column
                ]
            ),
            where=total_power > 0,
        )

    # --------------------------------------------------------
    # Spectral entropy
    # --------------------------------------------------------

    psd_sum = np.sum(
        analysis_psd,
        axis=1,
        keepdims=True,
    )

    probability = np.divide(
        analysis_psd,
        psd_sum,
        out=np.zeros_like(
            analysis_psd
        ),
        where=psd_sum > 0,
    )

    log_probability = np.zeros_like(
        probability
    )

    positive_mask = (
        probability > 0
    )

    log_probability[
        positive_mask
    ] = np.log2(
        probability[
            positive_mask
        ]
    )

    entropy = -np.sum(
        probability * log_probability,
        axis=1,
    )

    number_of_bins = (
        analysis_psd.shape[1]
    )

    if number_of_bins > 1:

        spectral_entropy = (
            entropy
            / np.log2(number_of_bins)
        )

    else:

        spectral_entropy = np.zeros(
            n_epochs,
            dtype=np.float64,
        )

    feature_data[
        "spectral_entropy"
    ] = spectral_entropy

    # --------------------------------------------------------
    # Dominant frequency
    # --------------------------------------------------------

    dominant_indices = np.argmax(
        analysis_psd,
        axis=1,
    )

    feature_data[
        "dominant_frequency"
    ] = analysis_frequencies[
        dominant_indices
    ]

    # --------------------------------------------------------
    # DataFrame
    # --------------------------------------------------------

    return pd.DataFrame(
        feature_data
    )
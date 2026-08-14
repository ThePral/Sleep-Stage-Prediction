import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.integrate import trapezoid
from scipy.stats import entropy

def extract_frequency_features_batch(
    epochs,
    sfreq,
    frequency_bands,
    analysis_low=0.5,
    analysis_high=30.0,
    nperseg=1024
):
    """
    Extract frequency-domain features from multiple EEG epochs.

    Parameters
    ----------
    epochs : numpy.ndarray
        Shape: (n_epochs, n_samples)

    sfreq : float
        Sampling frequency.

    frequency_bands : dict
        Frequency bands such as delta/theta/alpha/beta.

    analysis_low : float
        Lower frequency limit for spectral entropy
        and dominant frequency.

    analysis_high : float
        Upper frequency limit for spectral entropy
        and dominant frequency.

    nperseg : int
        Welch PSD segment length.

    Returns
    -------
    features_df : pandas.DataFrame
        One row per epoch.
    """

    frequencies, psd = welch(
        epochs,
        fs=sfreq,
        nperseg=nperseg,
        axis=1
    )

    feature_data = {}

    # Absolute and relative band power
    absolute_powers = {}

    for band, (low, high) in frequency_bands.items():

        mask = (
            (frequencies >= low)
            & (frequencies < high)
        )

        power = trapezoid(
            psd[:, mask],
            frequencies[mask],
            axis=1
        )

        absolute_powers[band] = power

        feature_data[f"{band}_absolute"] = power

    total_power = np.sum(
        list(absolute_powers.values()),
        axis=0
    )

    for band, power in absolute_powers.items():

        feature_data[f"{band}_relative"] = np.divide(
            power,
            total_power,
            out=np.zeros_like(power),
            where=total_power > 0
        )

    # Restrict spectrum for entropy and dominant frequency
    analysis_mask = (
        (frequencies >= analysis_low)
        & (frequencies <= analysis_high)
    )

    analysis_frequencies = frequencies[
        analysis_mask
    ]

    analysis_psd = psd[
        :,
        analysis_mask
    ]

    # Normalize PSD to probability distributions
    psd_sum = np.sum(
        analysis_psd,
        axis=1,
        keepdims=True
    )

    psd_probability = np.divide(
        analysis_psd,
        psd_sum,
        out=np.zeros_like(analysis_psd),
        where=psd_sum > 0
    )

    # Spectral entropy
    entropy_values = entropy(
        psd_probability,
        axis=1,
        base=2
    )

    if psd_probability.shape[1] > 1:
        entropy_values /= np.log2(
            psd_probability.shape[1]
        )

    feature_data["spectral_entropy"] = (
        entropy_values
    )

    # Dominant frequency
    dominant_indices = np.argmax(
        analysis_psd,
        axis=1
    )

    feature_data["dominant_frequency"] = (
        analysis_frequencies[dominant_indices]
    )

    return pd.DataFrame(feature_data)
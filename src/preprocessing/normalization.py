import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


ABSOLUTE_POWER_FEATURES = [
    "delta_absolute",
    "theta_absolute",
    "alpha_absolute",
    "beta_absolute",
]

STANDARD_FEATURES = [
    "delta_relative",
    "theta_relative",
    "alpha_relative",
    "beta_relative",
    "spectral_entropy",
    "dominant_frequency",
]

ALL_FEATURES = (
    ABSOLUTE_POWER_FEATURES
    + STANDARD_FEATURES
)


class SleepFeatureScaler:
    """
    Feature transformation pipeline for sleep-stage prediction.

    Absolute spectral powers are transformed with log10 first,
    then all features are standardized.

    The scaler must be fitted using training data only.
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.is_fitted = False

    def _prepare_features(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Apply feature-specific transformations.

        Absolute powers:
            log10(power)

        Other features:
            unchanged
        """

        transformed = dataframe[
            ALL_FEATURES
        ].copy()

        # Absolute power values produced by PSD are
        # strictly positive in our dataset.
        for column in ABSOLUTE_POWER_FEATURES:

            values = transformed[column].to_numpy(
                dtype=np.float64
            )

            if np.any(values <= 0):
                raise ValueError(
                    f"Feature '{column}' contains "
                    "non-positive values, so log10 "
                    "cannot be safely applied."
                )

            transformed[column] = np.log10(
                values
            )

        return transformed

    def fit(
        self,
        dataframe: pd.DataFrame,
    ):
        """
        Fit the scaler on training data only.
        """

        prepared = self._prepare_features(
            dataframe
        )

        self.scaler.fit(
            prepared[ALL_FEATURES]
        )

        self.is_fitted = True

        return self

    def transform(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Transform a dataset using the scaler
        previously fitted on training data.
        """

        if not self.is_fitted:
            raise RuntimeError(
                "The scaler has not been fitted yet."
            )

        prepared = self._prepare_features(
            dataframe
        )

        scaled = self.scaler.transform(
            prepared[ALL_FEATURES]
        )

        result = pd.DataFrame(
            scaled,
            columns=ALL_FEATURES,
            index=dataframe.index,
        )

        return result

    def fit_transform(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Fit on training data and transform it.
        """

        self.fit(dataframe)

        return self.transform(
            dataframe
        )
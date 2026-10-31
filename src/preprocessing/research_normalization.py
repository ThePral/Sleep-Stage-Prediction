"""Streaming source-training statistics with portable, explicit JSON state."""

import json
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.preprocessing.normalization import ALL_FEATURES
from src.preprocessing.waveform_normalization import WaveformNormalizer, source_identity, source_subset


def log_spectral(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim < 2 or values.shape[-1] != len(ALL_FEATURES) or not np.isfinite(values).all():
        raise ValueError("Expected finite spectral features in the declared ten-feature order.")
    if np.any(values[..., :4] <= 0):
        raise ValueError("Absolute spectral powers must be positive before log transformation.")
    values = values.copy()
    values[..., :4] = np.log10(values[..., :4])
    return values


class ResearchNormalizer:
    """Per-epoch mean removal and a single source-fitted amplitude scale.

    No per-epoch variance standardization or target-cohort statistics are used.
    Spectral statistics are fitted on unique eligible source epochs, never on
    the repeated copies induced by overlapping temporal windows.
    """

    def __init__(self):
        self.state = None

    def fit(self, partition):
        if partition.split != "train":
            raise ValueError("Normalizer fitting requires source training participants.")
        manifest = partition.manifest()
        if manifest["dataset"] != "sleepedf" or any(r["split"] != "train" for r in manifest["recordings"]):
            raise ValueError("Target/held-out recordings cannot fit normalization.")
        scaler = StandardScaler()
        squares, samples, epochs = 0.0, 0, 0
        for raw, spectral in partition.iter_fit_batches():
            centered = raw.astype(np.float64) - raw.mean(axis=1, keepdims=True, dtype=np.float64)
            squares += float(np.square(centered).sum(dtype=np.float64))
            samples += centered.size
            epochs += len(raw)
            scaler.partial_fit(log_spectral(spectral))
        if not epochs or not np.isfinite(squares) or squares <= 0:
            raise ValueError("No non-flat source training epochs available for normalization.")
        self.state = {
            "schema_version": 1, "features": ALL_FEATURES, "signal_unit": "V", "power_unit": "V^2",
            "waveform_epoch_mean_removal": True, "waveform_scale": float(np.sqrt(squares / samples)),
            "spectral_mean": scaler.mean_.tolist(), "spectral_scale": scaler.scale_.tolist(),
            "fit_epochs": epochs, "fit_policy": "unique_supervised_source_training_epochs",
            "preprocessing_fingerprint": partition.preprocessing.fingerprint,
            "training_data": manifest,
        }
        self._validate()
        return self

    def _validate(self):
        state = self.state
        if state is None:
            raise RuntimeError("Normalizer has not been fitted.")
        if state.get("schema_version") != 1 or state.get("features") != ALL_FEATURES:
            raise ValueError("Unsupported normalization schema or feature ordering.")
        if state.get("signal_unit") != "V" or state.get("power_unit") != "V^2" or state.get("waveform_epoch_mean_removal") is not True:
            raise ValueError("Unexpected waveform/spectral normalization convention.")
        mean, scale = np.asarray(state["spectral_mean"]), np.asarray(state["spectral_scale"])
        if mean.shape != (10,) or scale.shape != (10,) or not np.isfinite([mean, scale]).all() or np.any(scale <= 0):
            raise ValueError("Invalid saved spectral normalization statistics.")
        if not np.isfinite(state["waveform_scale"]) or state["waveform_scale"] <= 0 or state["fit_epochs"] < 1:
            raise ValueError("Invalid saved waveform normalization statistics.")
        training = state["training_data"]
        if training["dataset"] != "sleepedf" or training["split"] != "train" or not training["recordings"] or any(r["split"] != "train" for r in training["recordings"]):
            raise ValueError("Normalization state must declare source-training-only fitting.")
        if state.get("fit_policy") != "unique_supervised_source_training_epochs":
            raise ValueError("Unsupported normalization fitting policy.")
        if training["preprocessing_fingerprint"] != state["preprocessing_fingerprint"]:
            raise ValueError("Normalization provenance disagrees with its preprocessing fingerprint.")
        if state["fit_epochs"] != sum(r["fit_epochs"] for r in training["recordings"]):
            raise ValueError("Normalization epoch count disagrees with its provenance.")
        if "pretraining_waveform" in state:
            waveform = WaveformNormalizer(state["pretraining_waveform"])
            policy = state.get("pretraining_waveform_policy", "same_source_training_selection")
            matches = source_identity(training) == source_identity(waveform.state["training_data"])
            if policy == "label_budget_source_training_superset":
                matches = source_subset(training, waveform.state["training_data"])
            elif policy != "same_source_training_selection":
                raise ValueError("Unknown pretraining waveform fitting policy.")
            if (state["waveform_scale"] != waveform.state["waveform_scale"]
                    or state["preprocessing_fingerprint"] != waveform.state["preprocessing_fingerprint"]
                    or not matches):
                raise ValueError("Pretraining waveform statistics disagree with the downstream state.")

    def use_pretraining_waveform(self, waveform_normalizer, partition, *, allow_source_subset=False):
        """Keep the SSL input scale unchanged; spectral fitting remains supervised."""
        self.assert_compatible(partition, require_training_match=True)
        waveform_normalizer.assert_compatible(partition, require_training_match=True, allow_source_subset=allow_source_subset)
        # JSON copy prevents later mutation of the source object's provenance.
        self.state["pretraining_waveform"] = json.loads(json.dumps(waveform_normalizer.state))
        self.state["waveform_scale"] = waveform_normalizer.state["waveform_scale"]
        self.state["pretraining_waveform_policy"] = "label_budget_source_training_superset" if allow_source_subset else "same_source_training_selection"
        self._validate()
        return self

    def assert_compatible(self, partition, *, require_training_match=False):
        self._validate()
        if self.state["preprocessing_fingerprint"] != partition.preprocessing.fingerprint:
            raise ValueError("Normalizer and data preprocessing fingerprints disagree.")
        if require_training_match:
            def identity(manifest):
                # Logical identities and content hashes survive relocation to a
                # cloud machine; absolute paths and temporal context do not fit statistics.
                return (manifest["dataset"], manifest["split"], manifest["split_manifest"]["sha256"],
                        sorted((r["recording_id"], r["participant_id"], r["split"],
                                r["manifest_sha256"], r["fit_epochs"]) for r in manifest["recordings"]))
            if identity(self.state["training_data"]) != identity(partition.manifest()):
                raise ValueError("Normalizer was fitted on a different source training selection.")

    def transform_waveform(self, waveforms):
        self._validate()
        waveforms = np.asarray(waveforms, dtype=np.float32)
        if waveforms.ndim < 2 or waveforms.shape[-1] != 1 or not np.isfinite(waveforms).all():
            raise ValueError("Expected finite waveform samples with a single channel, in volts.")
        centered = waveforms - waveforms.mean(axis=-2, keepdims=True, dtype=np.float64)
        return np.asarray(centered / self.state["waveform_scale"], dtype=np.float32)

    def transform_spectral(self, spectral):
        self._validate()
        values = log_spectral(spectral)
        return np.asarray((values - self.state["spectral_mean"]) / self.state["spectral_scale"], dtype=np.float32)

    def save(self, path):
        self._validate()
        Path(path).write_text(json.dumps(self.state, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path):
        result = cls()
        result.state = json.loads(Path(path).read_text(encoding="utf-8"))
        result._validate()
        return result

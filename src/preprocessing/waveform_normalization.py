"""Quality-only source amplitude fitting, independent of annotation availability."""

import json
from pathlib import Path

import numpy as np


def source_identity(manifest):
    return (manifest["dataset"], manifest["split"], manifest["split_manifest"]["sha256"],
            sorted((r["recording_id"], r["participant_id"], r["split"], r["manifest_sha256"])
                   for r in manifest["recordings"]))


class WaveformNormalizer:
    def __init__(self, state=None):
        self.state = state
        if state is not None:
            self.validate()

    def fit(self, pool):
        manifest = pool.manifest()
        if manifest.get("dataset") != "sleepedf" or manifest.get("split") != "train" or manifest.get("eligibility") != "quality":
            raise ValueError("Waveform fitting requires the quality-only source training pool.")
        squares, samples, epochs = 0., 0, 0
        for raw in pool.iter_waveform_fit_batches():
            centered = raw.astype(np.float64) - raw.mean(axis=-2, keepdims=True, dtype=np.float64)
            squares += float(np.square(centered).sum())
            samples += centered.size
            epochs += len(raw)
        if not epochs or not np.isfinite(squares) or squares <= 0:
            raise ValueError("No usable source waveforms for amplitude fitting.")
        self.state = {"schema_version": 1, "signal_unit": "V", "waveform_epoch_mean_removal": True,
                      "waveform_scale": float(np.sqrt(squares / samples)), "fit_epochs": epochs,
                      "fit_policy": "unique_quality_source_training_epochs",
                      "preprocessing_fingerprint": pool.preprocessing.fingerprint, "training_data": manifest}
        self.validate()
        return self

    def validate(self):
        s = self.state
        if s is None:
            raise RuntimeError("Waveform normalizer has not been fitted.")
        if s.get("schema_version") != 1 or s.get("signal_unit") != "V" or s.get("waveform_epoch_mean_removal") is not True:
            raise ValueError("Invalid waveform normalization convention.")
        data = s["training_data"]
        if (s.get("fit_policy") != "unique_quality_source_training_epochs" or data.get("dataset") != "sleepedf"
                or data.get("split") != "train" or data.get("eligibility") != "quality" or not data["recordings"]
                or any(r["split"] != "train" for r in data["recordings"])):
            raise ValueError("Waveform normalization requires quality-only source-training provenance.")
        if (not np.isfinite(s["waveform_scale"]) or s["waveform_scale"] <= 0
                or s["fit_epochs"] < 1 or s["fit_epochs"] != sum(r["fit_epochs"] for r in data["recordings"])
                or s["preprocessing_fingerprint"] != data["preprocessing_fingerprint"]):
            raise ValueError("Invalid waveform statistics or provenance.")

    def assert_compatible(self, partition, *, require_training_match=False):
        self.validate()
        if self.state["preprocessing_fingerprint"] != partition.preprocessing.fingerprint:
            raise ValueError("Waveform preprocessing fingerprints disagree.")
        if require_training_match and source_identity(self.state["training_data"]) != source_identity(partition.manifest()):
            raise ValueError("Pretraining and downstream source training selections disagree.")

    def transform_waveform(self, values):
        self.validate()
        values = np.asarray(values, dtype=np.float32)
        if values.ndim < 2 or values.shape[-1] != 1 or not np.isfinite(values).all():
            raise ValueError("Expected finite single-channel waveform samples in volts.")
        return np.asarray((values - values.mean(axis=-2, keepdims=True, dtype=np.float64)) / self.state["waveform_scale"], dtype=np.float32)

    def save(self, path):
        self.validate()
        Path(path).write_text(json.dumps(self.state, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

"""Inference for sealed supervised waveform/spectral bundles, with frozen scaling."""

import json
from pathlib import Path

import numpy as np
from tensorflow import keras

from src.data.epoch_store import sha256_file
from src.models.multiview import ModelConfig  # Registers the custom Keras layers.
from src.preprocessing.normalization import ALL_FEATURES
from src.preprocessing.research_normalization import ResearchNormalizer
from src.preprocessing.sequences import CLASS_LABELS


class MultiviewPredictor:
    def __init__(self, directory):
        directory = Path(directory)
        bundle = json.loads((directory / "bundle.json").read_text(encoding="utf-8"))
        required = {"model.keras", "normalizer.json", "model_config.json", "data_manifest.json", "resolved_config.json"}
        if bundle.get("schema_version") != 1 or bundle.get("completed") is not True:
            raise ValueError("Inference requires a completed model bundle.")
        if bundle.get("classes") != CLASS_LABELS or bundle.get("features") != ALL_FEATURES:
            raise ValueError("Bundle class/feature ordering disagrees with the inference pipeline.")
        if set(bundle.get("files", {})) != required:
            raise ValueError("Bundle has an unexpected file inventory.")
        for name in required:
            if sha256_file(directory / name) != bundle["files"][name]:
                raise ValueError(f"Model bundle integrity check failed: {name}")
        self.config = ModelConfig(**json.loads((directory / "model_config.json").read_text(encoding="utf-8")))
        self.normalizer = ResearchNormalizer.load(directory / "normalizer.json")
        if self.normalizer.state["preprocessing_fingerprint"] != bundle["preprocessing_fingerprint"]:
            raise ValueError("Model bundle and normalizer preprocessing fingerprints disagree.")
        self.model = keras.models.load_model(directory / "model.keras", compile=False)
        actual = {tensor.name.split(":")[0]: tuple(tensor.shape[1:]) for tensor in self.model.inputs}
        expected = {"waveform": (self.config.sequence_length, self.config.samples_per_epoch, 1)}
        if self.config.variant != "raw":
            expected["spectral"] = (self.config.sequence_length, len(ALL_FEATURES))
        if actual != expected or self.model.output_shape[-1] != len(CLASS_LABELS):
            raise ValueError("Model/configuration input or output dimensions disagree.")

    def predict_batch(self, waveforms, spectral_features=None):
        """Predict raw windows in volts, and raw spectral powers in volts squared.

        Epochs must already use the bundle's declared preprocessing. Scaling is
        applied once here; no model or normalization statistics are updated.
        """
        waveforms = np.asarray(waveforms)
        expected = (self.config.sequence_length, self.config.samples_per_epoch, 1)
        if waveforms.ndim != 4 or waveforms.shape[1:] != expected or not len(waveforms):
            raise ValueError(f"Expected a nonempty waveform batch shaped (batch, {expected}).")
        inputs = {"waveform": self.normalizer.transform_waveform(waveforms)}
        if self.config.variant != "raw":
            features = np.asarray(spectral_features)
            if features.shape != (len(waveforms), self.config.sequence_length, len(ALL_FEATURES)):
                raise ValueError("Spectral feature windows must align with waveform windows.")
            inputs["spectral"] = self.normalizer.transform_spectral(features)
        probabilities = self.model(inputs, training=False).numpy()
        return probabilities.argmax(axis=1), probabilities

    def predict_sequence(self, waveform, spectral_features=None):
        features = None if spectral_features is None else np.asarray(spectral_features)[None]
        predicted, probabilities = self.predict_batch(np.asarray(waveform)[None], features)
        return {"predicted_stage": CLASS_LABELS[int(predicted[0])],
                "probabilities": {stage: float(p) for stage, p in zip(CLASS_LABELS, probabilities[0])}}

"""Atomic per-recording stores; waveform windows are gathered only on demand."""

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

from src.data.research import PreprocessingConfig, ProcessedRecording
from src.preprocessing.normalization import ALL_FEATURES
from src.preprocessing.sequences import build_sequence_positions


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_manifest(paths):
    return [{"path": str(Path(p).resolve()), "sha256": sha256_file(p)} for p in paths]


def write_recording(recording, destination):
    """Publish a complete new directory in one rename; never replace a dataset."""
    recording.validate()
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"Store already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".epoch-store-", dir=destination.parent) as folder:
        staging = Path(folder).resolve()
        if staging.parent != destination.parent:
            raise ValueError("Staging directory escaped the intended store parent.")
        np.save(staging / "waveforms.npy", recording.waveforms.astype(np.float32), allow_pickle=False)
        recording.epochs.to_csv(staging / "epochs.csv", index=False)
        manifest = {
            "schema_version": 1, "completed": True,
            "rows": len(recording.epochs), "waveform_shape": list(recording.waveforms.shape),
            "waveform_dtype": "float32", "features": ALL_FEATURES,
            "preprocessing": asdict(recording.config),
            "preprocessing_fingerprint": recording.config.fingerprint,
            "provenance": recording.provenance,
            "files": {name: sha256_file(staging / name) for name in ["waveforms.npy", "epochs.csv"]},
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        # On Windows, rename to an existing directory fails rather than merges.
        # The explicit existence check also preserves this policy on other OSes.
        if destination.exists():
            raise FileExistsError(f"Store already exists: {destination}")
        os.rename(staging, destination)
    return destination


class EpochStore:
    def close(self):
        """Release the memory map explicitly, including on Windows."""
        waveforms = getattr(self, "waveforms", None)
        mapping = getattr(waveforms, "_mmap", None)
        if mapping is not None and not mapping.closed:
            mapping.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        self.close()

    def __init__(self, directory, *, verify=True):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text(encoding="utf-8"))
        manifest = self.manifest
        if manifest.get("schema_version") != 1 or manifest.get("completed") is not True:
            raise ValueError("Unsupported or incomplete epoch store.")
        if manifest.get("features") != ALL_FEATURES:
            raise ValueError("Unexpected spectral feature order.")
        if set(manifest.get("files", {})) != {"waveforms.npy", "epochs.csv"}:
            raise ValueError("Store manifest has an unexpected file inventory.")
        if verify:
            for name, expected in manifest["files"].items():
                if sha256_file(self.directory / name) != expected:
                    raise ValueError(f"Epoch store integrity check failed: {name}")
        self.epochs = pd.read_csv(
            self.directory / "epochs.csv",
            dtype={c: str for c in ["dataset", "participant_id", "recording_id", "subject_id", "epoch_id"]},
        )
        self.waveforms = np.load(self.directory / "waveforms.npy", mmap_mode="r", allow_pickle=False)
        self.config = PreprocessingConfig(**manifest["preprocessing"])
        if self.config.fingerprint != manifest["preprocessing_fingerprint"]:
            raise ValueError("Preprocessing fingerprint mismatch.")
        if len(self.epochs) != manifest["rows"] or list(self.waveforms.shape) != manifest["waveform_shape"] or self.waveforms.dtype != np.float32:
            raise ValueError("Store row counts, waveform shape, or dtype disagree with the manifest.")
        ProcessedRecording(self.epochs, self.waveforms, self.config, manifest["provenance"]).validate()

    def sequence_positions(self, sequence_length=10, *, supervised=True):
        mask = self.epochs["supervised_valid" if supervised else "quality_valid"].to_numpy()
        rows = np.flatnonzero(mask)
        positions = build_sequence_positions(self.epochs.iloc[rows], sequence_length)
        return rows[positions]

    def gather(self, positions, *, supervised=True):
        """Gather only the requested batch; return raw, features, and target IDs.

        Features remain unscaled. Training must fit its scaler on source training
        rows only. This loader never estimates statistics from external data.
        """
        positions = np.asarray(positions)
        if positions.ndim != 2 or not np.issubdtype(positions.dtype, np.integer) or positions.shape[1] < 1:
            raise ValueError("positions must be a 2D integer array of epoch windows.")
        if np.any(positions < 0) or np.any(positions >= len(self.epochs)):
            raise IndexError("Window positions fall outside the epoch store.")
        if positions.shape[1] > 1 and not np.equal(np.diff(positions, axis=1), 1).all():
            raise ValueError("Windows must contain consecutive physical epochs.")
        eligible = self.epochs["supervised_valid" if supervised else "quality_valid"].to_numpy()
        if not eligible[positions].all():
            raise ValueError("Window contains epochs excluded by the selected eligibility policy.")
        return {
            "waveform": np.asarray(self.waveforms[positions])[..., None],
            "spectral": self.epochs[ALL_FEATURES].to_numpy(dtype=np.float32)[positions],
            "target_epoch_id": self.epochs["epoch_id"].to_numpy()[positions[:, -1]],
            "target_stage": self.epochs["target_stage"].to_numpy()[positions[:, -1]],
        }

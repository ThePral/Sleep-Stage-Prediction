"""Source-only, bounded-memory windows from verified research epoch stores."""

from collections import OrderedDict
from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.epoch_store import EpochStore, sha256_file
from src.data.prepare_research_data import ROOT, participant_assignments
from src.preprocessing.normalization import ALL_FEATURES
from src.preprocessing.sequences import CLASS_LABELS, CLASS_TO_INDEX


@dataclass
class RecordingIndex:
    directory: Path
    recording_id: str
    participant_id: str
    split: str
    manifest_sha256: str
    positions: np.ndarray
    spectral: np.ndarray
    labels: np.ndarray
    fit_rows: np.ndarray
    waveform_stat: tuple


class SourcePartition:
    """Only Sleep-EDF records agreeing with a frozen participant manifest.

    Full integrity checks run once on selected stores. Subsequent waveform reads
    use a small LRU of read-only memory maps; no full window tensor is cached.
    """

    def __init__(self, root, split, *, sequence_length=10, recordings=None,
                 split_file=None, max_open_recordings=2):
        if split not in {"train", "val", "test"}:
            raise ValueError("SourcePartition only accepts source train/val/test splits.")
        if max_open_recordings < 1:
            raise ValueError("At least one recording memory map must be allowed.")
        self.root = Path(root)
        self.split = split
        self.sequence_length = sequence_length
        self.max_open_recordings = max_open_recordings
        self._maps = OrderedDict()
        self.recordings = []
        self.preprocessing = None
        split_file = Path(split_file) if split_file else ROOT / "final_model_corrected/participant_split.csv"
        self.split_manifest = {"path": str(split_file.resolve()), "sha256": sha256_file(split_file)}
        assignments = participant_assignments(split_file)
        selected = set(recordings) if recordings is not None else None
        if selected is not None and (not selected or len(selected) != len(recordings)):
            raise ValueError("Explicit recording selections must be nonempty and unique.")
        for manifest in sorted((self.root / "sleepedf").glob("*/manifest.json")):
            if selected is not None and manifest.parent.name not in selected:
                continue
            head = pd.read_csv(manifest.parent / "epochs.csv", nrows=1,
                               usecols=["dataset", "participant_id", "recording_id", "split"],
                               dtype={"participant_id": str, "recording_id": str})
            if head.empty:
                raise ValueError("Source recording has no metadata rows.")
            row = head.iloc[0]
            if row["dataset"] != "sleepedf" or row["participant_id"] != str(row["recording_id"])[:5]:
                raise ValueError("Only correctly identified Sleep-EDF recordings may enter source training.")
            if assignments.get(row["participant_id"]) != row["split"]:
                raise ValueError("Source store disagrees with the frozen participant split.")
            if row["split"] != split:
                if selected is not None:
                    raise ValueError(f"Requested recording {manifest.parent.name} is not in the {split} split.")
                continue
            with EpochStore(manifest.parent) as store:
                frame = store.epochs
                if frame.recording_id.iloc[0] != manifest.parent.name:
                    raise ValueError("Source store identity disagrees with its directory.")
                if self.preprocessing is None:
                    self.preprocessing = store.config
                elif self.preprocessing != store.config:
                    raise ValueError("A source partition cannot mix preprocessing versions.")
                targets = frame.target_stage.map(CLASS_TO_INDEX).fillna(-1).to_numpy(dtype=np.int64)
                stat = (manifest.parent / "waveforms.npy").stat()
                self.recordings.append(RecordingIndex(
                    manifest.parent, str(frame.recording_id.iloc[0]), str(frame.participant_id.iloc[0]),
                    split, sha256_file(manifest), store.sequence_positions(sequence_length),
                    frame[ALL_FEATURES].to_numpy(dtype=np.float64, copy=True), targets,
                    np.flatnonzero(frame.supervised_valid.to_numpy()), (stat.st_size, stat.st_mtime_ns),
                ))
        found = {record.recording_id for record in self.recordings}
        if selected is not None and selected != found:
            raise ValueError(f"Requested source recordings were not found: {sorted(selected - found)}")
        if not self.recordings:
            raise ValueError(f"No verified source {split} recordings found.")
        self.offsets = np.r_[0, np.cumsum([len(r.positions) for r in self.recordings])]
        if not self.offsets[-1]:
            raise ValueError(f"No eligible {sequence_length}-epoch source windows in {split}.")

    def __len__(self):
        return int(self.offsets[-1])

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        for mapped in self._maps.values():
            mapped._mmap.close()
        self._maps.clear()

    def _waveforms(self, record):
        key = record.recording_id
        path = record.directory / "waveforms.npy"
        stat = path.stat()
        if (stat.st_size, stat.st_mtime_ns) != record.waveform_stat:
            raise ValueError("A waveform file changed after its integrity check.")
        if key in self._maps:
            self._maps.move_to_end(key)
            return self._maps[key]
        while len(self._maps) >= self.max_open_recordings:
            _, old = self._maps.popitem(last=False)
            old._mmap.close()
        mapped = np.load(path, mmap_mode="r", allow_pickle=False)
        self._maps[key] = mapped
        return mapped

    def iter_fit_batches(self, batch_epochs=64):
        if self.split != "train":
            raise ValueError("Normalization can only be fitted on source training participants.")
        for record in self.recordings:
            for start in range(0, len(record.fit_rows), batch_epochs):
                rows = record.fit_rows[start:start + batch_epochs]
                yield np.asarray(self._waveforms(record)[rows]), record.spectral[rows]

    def manifest(self):
        return {
            "dataset": "sleepedf", "split": self.split, "sequence_length": self.sequence_length,
            "preprocessing_fingerprint": self.preprocessing.fingerprint,
            "split_manifest": self.split_manifest,
            "windows": len(self),
            "recordings": [{"recording_id": r.recording_id, "participant_id": r.participant_id,
                            "split": r.split, "manifest_sha256": r.manifest_sha256,
                            "windows": len(r.positions), "fit_epochs": len(r.fit_rows)} for r in self.recordings],
        }

    def gather(self, indices):
        indices = np.asarray(indices)
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("Window indices must be a one-dimensional integer array.")
        if np.any(indices < 0) or np.any(indices >= len(self)):
            raise IndexError("Window indices fall outside the source partition.")
        length = self.sequence_length
        samples = int(round(self.preprocessing.epoch_duration * self.preprocessing.sampling_frequency))
        raw = np.empty((len(indices), length, samples, 1), dtype=np.float32)
        spectral = np.empty((len(indices), length, len(ALL_FEATURES)), dtype=np.float64)
        targets = np.empty(len(indices), dtype=np.int64)
        metadata = [None] * len(indices)
        record_numbers = np.searchsorted(self.offsets[1:], indices, side="right")
        for number in np.unique(record_numbers):
            record = self.recordings[number]
            selected = np.flatnonzero(record_numbers == number)
            positions = record.positions[indices[selected] - self.offsets[number]]
            raw[selected, ..., 0] = self._waveforms(record)[positions]
            spectral[selected] = record.spectral[positions]
            targets[selected] = record.labels[positions[:, -1]]
            for target, epoch in zip(selected, positions[:, -1]):
                metadata[int(target)] = {
                    "dataset": "sleepedf", "participant_id": record.participant_id,
                    "recording_id": record.recording_id, "epoch_index": int(epoch),
                    "epoch_id": f"sleepedf:{record.recording_id}:{epoch:06d}",
                    "start_time": float(epoch * self.preprocessing.epoch_duration),
                    "true_stage": CLASS_LABELS[record.labels[epoch]],
                }
        return raw, spectral, targets, metadata


class WindowBatches:
    def __init__(self, partition, normalizer, *, variant, batch_size=4,
                 shuffle=False, seed=42, max_batches=None):
        if variant not in {"raw", "concat", "gated"}:
            raise ValueError("Unknown supervised model variant.")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        if max_batches is not None and (isinstance(max_batches, bool) or not isinstance(max_batches, int) or max_batches < 1):
            raise ValueError("max_batches must be a positive integer when supplied.")
        normalizer.assert_compatible(partition)
        self.partition, self.normalizer, self.variant = partition, normalizer, variant
        self.batch_size, self.shuffle, self.seed = batch_size, shuffle, seed
        self.max_batches, self._iteration = max_batches, 0

    def __len__(self):
        count = math.ceil(len(self.partition) / self.batch_size)
        return min(count, self.max_batches) if self.max_batches is not None else count

    def iter_batches(self, *, epoch=None):
        if epoch is None:
            epoch = self._iteration
            self._iteration += 1
        order = np.arange(len(self.partition), dtype=np.int64)
        if self.shuffle:
            np.random.default_rng(self.seed + epoch).shuffle(order)
        elif self.max_batches is not None:
            # A capped validation pilot should span the selected nights instead
            # of evaluating only their initial Wake epochs. Selection uses no labels.
            count = min(len(order), len(self) * self.batch_size)
            order = np.linspace(0, len(order) - 1, count, dtype=np.int64)
        for batch in range(len(self)):
            indices = order[batch * self.batch_size:(batch + 1) * self.batch_size]
            raw, spectral, targets, metadata = self.partition.gather(indices)
            inputs = {"waveform": self.normalizer.transform_waveform(raw)}
            if self.variant != "raw":
                inputs["spectral"] = self.normalizer.transform_spectral(spectral)
            yield inputs, targets, metadata

    def as_dataset(self):
        import tensorflow as tf
        length = self.partition.sequence_length
        samples = int(round(self.partition.preprocessing.epoch_duration * self.partition.preprocessing.sampling_frequency))
        signature = {"waveform": tf.TensorSpec((None, length, samples, 1), tf.float32)}
        if self.variant != "raw":
            signature["spectral"] = tf.TensorSpec((None, length, len(ALL_FEATURES)), tf.float32)

        def generate():
            for inputs, targets, _ in self.iter_batches():
                yield inputs, targets

        dataset = tf.data.Dataset.from_generator(generate, output_signature=(signature, tf.TensorSpec((None,), tf.int64)))
        dataset = dataset.apply(tf.data.experimental.assert_cardinality(len(self)))
        options = tf.data.Options()
        options.deterministic = True
        options.threading.private_threadpool_size = 1
        return dataset.with_options(options).prefetch(1)

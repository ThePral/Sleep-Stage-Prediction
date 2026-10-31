"""Label-free source-training epochs and reproducible contrastive batches."""

import numpy as np

from src.data.supervised import SourcePartition


class SourceEpochPool(SourcePartition):
    """Full-store integrity is checked, but stage labels never select SSL inputs.

    Labels present in a store are validated by EpochStore's common integrity
    checker. They are not retained by this pool or returned to pretraining.
    """

    def __init__(self, root, *, recordings=None, split_file=None, max_open_recordings=2):
        super().__init__(root, "train", sequence_length=1, recordings=recordings,
                         split_file=split_file, max_open_recordings=max_open_recordings,
                         eligibility="quality")

    def gather(self, indices):
        indices = np.asarray(indices)
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("Epoch indices must be a one-dimensional integer array.")
        if np.any(indices < 0) or np.any(indices >= len(self)):
            raise IndexError("Epoch indices fall outside the source training pool.")
        samples = int(round(self.preprocessing.epoch_duration * self.preprocessing.sampling_frequency))
        raw = np.empty((len(indices), samples, 1), dtype=np.float32)
        metadata = [None] * len(indices)
        numbers = np.searchsorted(self.offsets[1:], indices, side="right")
        for number in np.unique(numbers):
            record = self.recordings[number]
            selected = np.flatnonzero(numbers == number)
            rows = record.positions[indices[selected] - self.offsets[number], 0]
            raw[selected, ..., 0] = self._waveforms(record)[rows]
            for target, epoch in zip(selected, rows):
                metadata[int(target)] = {
                    "participant_id": record.participant_id, "recording_id": record.recording_id,
                    "epoch_index": int(epoch), "epoch_id": f"sleepedf:{record.recording_id}:{epoch:06d}",
                }
        return raw, metadata

    def iter_waveform_fit_batches(self, batch_epochs=64):
        for start in range(0, len(self), batch_epochs):
            yield self.gather(np.arange(start, min(start + batch_epochs, len(self)), dtype=np.int64))[0]


def negative_mask(metadata, exclusion_epochs=1):
    """Allow nonlocal same-recording and all other-recording negatives.

    Same-stage false negatives remain possible; labels are deliberately absent.
    The matrix indexes original epochs, before duplication into two views.
    """
    records = np.asarray([m["recording_id"] for m in metadata])
    epochs = np.asarray([m["epoch_index"] for m in metadata])
    return (records[:, None] != records[None, :]) | (np.abs(epochs[:, None] - epochs[None, :]) > exclusion_epochs)


class EpochBatches:
    def __init__(self, pool, normalizer, *, batch_size=32, seed=42, max_batches=None, exclusion_epochs=1):
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 2:
            raise ValueError("Contrastive batch_size must be an integer >= 2.")
        if max_batches is not None and (isinstance(max_batches, bool) or not isinstance(max_batches, int) or max_batches < 1):
            raise ValueError("max_batches must be a positive integer.")
        if isinstance(exclusion_epochs, bool) or not isinstance(exclusion_epochs, int) or exclusion_epochs < 0:
            raise ValueError("exclusion_epochs must be a nonnegative integer.")
        normalizer.assert_compatible(pool)
        self.pool, self.normalizer = pool, normalizer
        self.batch_size, self.seed, self.max_batches = batch_size, seed, max_batches
        self.exclusion_epochs = exclusion_epochs
        if not len(self):
            raise ValueError("Not enough quality-valid source epochs for one full contrastive batch.")

    def __len__(self):
        count = len(self.pool) // self.batch_size
        return min(count, self.max_batches) if self.max_batches is not None else count

    def iter_batches(self, epoch):
        order = np.random.default_rng(np.random.SeedSequence([self.seed, epoch])).permutation(len(self.pool))
        for batch in range(len(self)):
            indices = order[batch * self.batch_size:(batch + 1) * self.batch_size]
            raw, metadata = self.pool.gather(indices)
            allowed = negative_mask(metadata, self.exclusion_epochs)
            if not allowed.any(axis=1).all():
                raise ValueError("A contrastive anchor has no negatives after temporal exclusion; increase batch size or reduce exclusion_epochs.")
            yield self.normalizer.transform_waveform(raw), allowed, metadata

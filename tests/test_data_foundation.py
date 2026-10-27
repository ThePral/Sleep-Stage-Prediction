"""Regression tests for the audit's alignment defects and shared data contract."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout, redirect_stderr
import io

import mne
import numpy as np
import pandas as pd

from src.data.adapters.isruc import read_workbook_labels, select_derivation
from src.data.epoch_store import EpochStore, write_recording
from src.data.prepare_research_data import participant_assignments
from src.data.process_all import is_valid_feature_file
from src.data.research import PreprocessingConfig, prepare_recording
from src.preprocessing.annotations import build_epoch_metadata
from src.preprocessing.epoching import extract_eeg_epochs_by_indices, find_valid_epoch_indices
from src.preprocessing.sequences import build_sequence_positions, build_sequences_from_dataframe


def raw_signal(n_epochs=4, sfreq=100, *, flat_epoch=None):
    n = int(30 * sfreq)
    time = np.arange(n) / sfreq
    epochs = np.stack([
        (i + 1) * 1e-5 * np.sin(2 * np.pi * (i + 1) * time)
        + 2e-6 * np.sin(2 * np.pi * 15 * time)
        for i in range(n_epochs)
    ])
    if flat_epoch is not None:
        epochs[flat_epoch] = 0
    return mne.io.RawArray(epochs.reshape(1, -1), mne.create_info(["EEG"], sfreq, ["eeg"]), verbose=False)


def annotations():
    return mne.Annotations([0, 30, 60, 90], [30] * 4,
                           ["Sleep stage W", "Movement time", "Sleep stage 2", "Sleep stage R"])


def recording(*, sfreq=100, flat_epoch=None, dataset="sleepedf", split="train"):
    raw = raw_signal(sfreq=sfreq, flat_epoch=flat_epoch)
    labels = build_epoch_metadata(annotations(), "SC4002", 120, include_excluded=True)
    return prepare_recording(
        raw, labels, dataset=dataset, participant_id="SC400", recording_id="SC4002",
        channel_as_recorded="EEG", derivation_used="EEG", split=split,
    )


class AnnotationTests(unittest.TestCase):
    def test_excluded_middle_epoch_preserves_physical_position(self):
        frame = build_epoch_metadata(annotations(), "SC4002", 120)
        self.assertEqual(frame.epoch_index.tolist(), [0, 2, 3])
        raw = mne.io.RawArray(np.repeat([1., 2., 3., 4.], 3000)[None], mne.create_info(["EEG"], 100), verbose=False)
        selected = extract_eeg_epochs_by_indices(raw, frame.epoch_index)
        np.testing.assert_array_equal(selected.mean(axis=1), [1., 3., 4.])

    def test_nonzero_start_and_recording_end(self):
        ann = mne.Annotations([60], [100], ["Sleep stage 2"])
        frame = build_epoch_metadata(ann, "s", 135)
        self.assertEqual(frame.epoch_index.tolist(), [2, 3])
        self.assertEqual(frame.end_time.tolist(), [90, 120])

    def test_off_grid_annotation_rejected(self):
        with self.assertRaisesRegex(ValueError, "aligned"):
            build_epoch_metadata(mne.Annotations([1], [30], ["Sleep stage W"]), "s", 120)

    def test_overlapping_annotations_rejected(self):
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            build_epoch_metadata(mne.Annotations([0, 0], [30, 30], ["Sleep stage W"] * 2), "s", 120)

    def test_empty_metadata_retains_schema(self):
        frame = build_epoch_metadata(mne.Annotations([0], [30], ["Movement time"]), "s", 120)
        self.assertTrue(frame.empty)
        self.assertIn("epoch_index", frame)

    def test_unknowns_can_be_retained_without_inventing_labels(self):
        frame = build_epoch_metadata(annotations(), "s", 120, include_excluded=True)
        self.assertEqual(frame.epoch_index.tolist(), [0, 1, 2, 3])
        self.assertTrue(pd.isna(frame.target_stage.iloc[1]))


class EpochExtractionTests(unittest.TestCase):
    def test_invalid_indices_are_never_coerced_or_wrapped(self):
        raw = raw_signal()
        for indices in [[-1], [4], [1.5], [np.nan], [[1]]]:
            with self.subTest(indices=indices), self.assertRaises((ValueError, IndexError)):
                extract_eeg_epochs_by_indices(raw, indices)

    def test_qc_rejects_metadata_signal_misalignment(self):
        frame = build_epoch_metadata(annotations(), "s", 120)
        frame.loc[1, "epoch_index"] = 1
        with self.assertRaisesRegex(ValueError, "disagrees"):
            find_valid_epoch_indices(raw_signal(), frame)

    def test_flat_epoch_removed_before_filtering(self):
        frame = build_epoch_metadata(annotations(), "s", 120, include_excluded=True)
        self.assertEqual(find_valid_epoch_indices(raw_signal(flat_epoch=2), frame).tolist(), [0, 1, 3])

    def test_empty_extraction_keeps_sample_dimension(self):
        self.assertEqual(extract_eeg_epochs_by_indices(raw_signal(), []).shape, (0, 3000))


class SequenceTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({"subject_id": ["s"] * 3, "start_time": [0, 30, 60],
                             "end_time": [30, 60, 90], "target_stage": ["W", "N1", "N2"]})

    def test_arbitrary_and_duplicate_index_labels_do_not_reorder_features(self):
        for index in [[2, 0, 1], [8, 8, 8], ["a", "b", "c"]]:
            frame = self.frame().set_axis(index)
            X, y = build_sequences_from_dataframe(frame, [[10], [20], [30]], 2)
            np.testing.assert_array_equal(X[..., 0], [[10, 20], [20, 30]])
            self.assertEqual(y.tolist(), ["N1", "N2"])

    def test_sorting_preserves_original_feature_correspondence(self):
        frame = self.frame().iloc[[2, 0, 1]]
        X, y = build_sequences_from_dataframe(frame, [[30], [10], [20]], 2)
        np.testing.assert_array_equal(X[..., 0], [[10, 20], [20, 30]])
        self.assertEqual(y.tolist(), ["N1", "N2"])

    def test_windows_do_not_cross_gaps_or_recordings(self):
        frame = self.frame()
        frame.loc[2, ["start_time", "end_time"]] = [120, 150]
        np.testing.assert_array_equal(build_sequence_positions(frame, 2), [[0, 1]])
        frame.loc[1, "subject_id"] = "other"
        self.assertEqual(build_sequence_positions(frame, 2).shape, (0, 2))

    def test_datasets_and_splits_are_isolated(self):
        frame = self.frame()
        frame["dataset"] = ["source", "target", "source"]
        self.assertEqual(build_sequence_positions(frame, 2).shape, (0, 2))
        frame["dataset"] = "source"
        frame["split"] = ["train", "val", "train"]
        self.assertEqual(build_sequence_positions(frame, 2).shape, (0, 2))

    def test_position_and_legacy_feature_apis_agree(self):
        frame = self.frame()
        positions = build_sequence_positions(frame, 2)
        X, _ = build_sequences_from_dataframe(frame, [[10], [20], [30]], 2)
        np.testing.assert_array_equal(X, np.array([[10], [20], [30]])[positions])


class ResearchStoreTests(unittest.TestCase):
    def test_unscored_epochs_are_available_only_to_label_independent_selection(self):
        result = recording()
        self.assertEqual(result.epochs.epoch_index.tolist(), [0, 1, 2, 3])
        self.assertEqual(result.epochs.supervised_valid.tolist(), [True, False, True, True])
        self.assertTrue(result.epochs.quality_valid.all())

    def test_signal_qc_preserves_grid_and_breaks_windows(self):
        result = recording(flat_epoch=2)
        self.assertEqual(len(result.epochs), 4)
        self.assertFalse(result.epochs.quality_valid.iloc[2])

    def test_isruc_resampling_preserves_epoch_boundaries(self):
        result = recording(sfreq=200)
        self.assertEqual(result.waveforms.shape, (4, 3000))
        self.assertEqual(result.epochs.original_start_sample.tolist(), [0, 6000, 12000, 18000])
        self.assertEqual(result.epochs.canonical_start_sample.tolist(), [0, 3000, 6000, 9000])

    def test_external_cannot_be_marked_train(self):
        with self.assertRaisesRegex(ValueError, "external"):
            recording(dataset="isruc", split="train")

    def test_roundtrip_mmap_and_supervised_vs_ssl_windows(self):
        result = recording()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "recording"
            write_recording(result, path)
            store = EpochStore(path)
            self.assertIsInstance(store.waveforms, np.memmap)
            np.testing.assert_array_equal(store.sequence_positions(2), [[2, 3]])
            np.testing.assert_array_equal(store.sequence_positions(2, supervised=False), [[0, 1], [1, 2], [2, 3]])
            batch = store.gather(store.sequence_positions(2))
            self.assertEqual(batch["waveform"].shape, (1, 2, 3000, 1))
            self.assertEqual(batch["spectral"].shape, (1, 2, 10))
            self.assertEqual(batch["target_stage"].tolist(), ["REM"])
            with self.assertRaises(FileExistsError):
                write_recording(result, path)
            store.close()

    def test_tampering_and_incomplete_stores_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "recording"
            write_recording(recording(), path)
            with (path / "epochs.csv").open("a") as stream:
                stream.write("\n")
            with self.assertRaisesRegex(ValueError, "integrity"):
                EpochStore(path)
            manifest = json.loads((path / "manifest.json").read_text())
            manifest["completed"] = False
            (path / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                EpochStore(path)

    def test_preprocessing_fingerprint_changes_with_parameters(self):
        config = PreprocessingConfig()
        self.assertNotEqual(config.fingerprint, replace(config, low_frequency=0.5).fingerprint)

    def test_misaligned_waveforms_are_rejected_before_publication(self):
        result = recording()
        result.waveforms = result.waveforms[:-1]
        with self.assertRaisesRegex(ValueError, "aligned"):
            result.validate()

    def test_amplitude_scaling_preserves_relative_powers_and_squares_absolute_power(self):
        raw = raw_signal()
        labels = build_epoch_metadata(annotations(), "s", 120, include_excluded=True)
        options = dict(dataset="sleepedf", participant_id="p", recording_id="s",
                       channel_as_recorded="EEG", derivation_used="EEG", split="train")
        original = prepare_recording(raw, labels, **options)
        amplified = mne.io.RawArray(raw.get_data() * 2, raw.info.copy(), verbose=False)
        changed = prepare_recording(amplified, labels, **options)
        absolute = ["delta_absolute", "theta_absolute", "alpha_absolute", "beta_absolute"]
        relative = ["delta_relative", "theta_relative", "alpha_relative", "beta_relative"]
        np.testing.assert_allclose(changed.epochs[absolute], original.epochs[absolute] * 4, rtol=1e-6)
        np.testing.assert_allclose(changed.epochs[relative], original.epochs[relative], rtol=1e-6)

    def test_gather_rejects_invalid_context_and_skipped_physical_epochs(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "recording"
            write_recording(recording(), path)
            with EpochStore(path) as store:
                with self.assertRaisesRegex(ValueError, "consecutive"):
                    store.gather(np.array([[0, 2]]))
                with self.assertRaisesRegex(ValueError, "excluded"):
                    store.gather(np.array([[0, 1]]))
                self.assertEqual(store.gather(np.array([[0, 1]]), supervised=False)["waveform"].shape, (1, 2, 3000, 1))


class AdapterAndCacheTests(unittest.TestCase):
    def test_failed_legacy_processing_preserves_the_previous_combined_dataset(self):
        from src.data import process_dataset
        pairs = pd.DataFrame({"recording_id": ["SC4001", "SC4002"],
                              "psg": ["first.edf", "second.edf"], "hypnogram": ["first-label.edf", "second-label.edf"]})
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            previous = directory / "sleep_edf_features.csv"
            previous.write_text("previous complete cohort", encoding="utf-8")
            with patch.object(process_dataset, "FEATURES_DIR", directory), \
                 patch.object(process_dataset, "prepare_directories"), \
                 patch.object(process_dataset, "build_recording_pairs", return_value=pairs), \
                 patch.object(process_dataset, "validate_recording_pairs", return_value={"is_valid": True}), \
                 patch.object(process_dataset, "save_recording_features"), \
                 patch.object(process_dataset, "process_one_recording", side_effect=[pd.DataFrame({"target_stage": ["W"]}), RuntimeError("incomplete signal")]), \
                 redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "partial cohort"):
                    process_dataset.main()
            self.assertEqual(previous.read_text(encoding="utf-8"), "previous complete cohort")

    def test_workbook_headers_and_explicit_epoch_numbers(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "1_1.xlsx"
            pd.DataFrame([["Epoch", "Stage"], [1, "W"], [2, "U"], [3, "R"]]).to_excel(path, header=False, index=False)
            frame = read_workbook_labels(path)
            self.assertEqual(frame.epoch_index.tolist(), [0, 1, 2])
            self.assertEqual(frame.target_stage.iloc[[0, 2]].tolist(), ["W", "REM"])
            self.assertTrue(pd.isna(frame.target_stage.iloc[1]))
            pd.DataFrame([[1, "4"]]).to_excel(path, header=False, index=False)
            with self.assertRaisesRegex(ValueError, "ambiguous"):
                read_workbook_labels(path)

    def test_bipolar_derivation_is_not_subtracted_again(self):
        raw = raw_signal().rename_channels({"EEG": "C3-A2"})
        selected, original, derivation = select_derivation(raw)
        np.testing.assert_array_equal(selected.get_data(), raw.get_data())
        self.assertEqual((original, derivation), ("C3-A2", "C3-A2"))
        with self.assertRaisesRegex(ValueError, "fallback"):
            select_derivation(raw, "F3")

    def test_damaged_first_header_is_recorded_without_skipping_numbered_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "29_1.xlsx"
            pd.DataFrame([["hich", "Stage"], [1, "W"], [2, "N1"]]).to_excel(path, header=False, index=False)
            frame = read_workbook_labels(path)
            self.assertEqual(frame.epoch_index.tolist(), [0, 1])
            self.assertEqual(frame.attrs["header_rows"][0]["epoch_header"], "hich")
            pd.DataFrame([["Epoch", "Stage"], [1, "Stage"]]).to_excel(path, header=False, index=False)
            with self.assertRaisesRegex(ValueError, "ambiguous"):
                read_workbook_labels(path)

    def test_separate_electrodes_are_subtracted_in_physical_units(self):
        values = np.stack([np.arange(100, dtype=float), np.ones(100)]) * 1e-6
        raw = mne.io.RawArray(values, mne.create_info(["C3", "A2"], 100, ["eeg"] * 2), verbose=False)
        selected, _, derivation = select_derivation(raw)
        np.testing.assert_allclose(selected.get_data()[0], values[0] - values[1])
        self.assertEqual(derivation, "C3-A2")

    def test_legacy_header_only_and_misaligned_caches_rejected(self):
        frame = recording().epochs
        frame = frame[frame.supervised_valid]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "features.csv"
            frame.iloc[:0].to_csv(path, index=False)
            self.assertFalse(is_valid_feature_file(path))
            frame.to_csv(path, index=False)
            self.assertTrue(is_valid_feature_file(path))
            frame = frame.copy()
            frame.loc[frame.index[-1], "epoch_index"] = 1
            frame.to_csv(path, index=False)
            self.assertFalse(is_valid_feature_file(path))

    def test_conflicting_participant_assignments_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "split.csv"
            pd.DataFrame({"participant_id": ["SC400", "SC400"], "split": ["train", "test"]}).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "exactly one"):
                participant_assignments(path)


if __name__ == "__main__":
    unittest.main()

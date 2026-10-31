"""Cluster inference, model freezing, synthetic external scoring and profiling."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import mne
import numpy as np
import pandas as pd

from src.data.epoch_store import sha256_file, write_recording
from src.data.research import prepare_recording
from src.evaluation.artifacts import inspect_bundle, verify_validation_frame
from src.evaluation.frozen import (freeze_evaluation, inventory_cohort, load_freeze, normalize_protocol,
                                    predict_frozen_model, score_frozen, transition_mask)
from src.evaluation.profile import ProfileConfig, profile_bundle
from src.evaluation.uncertainty import (AnalysisConfig, analyze_models, analyze_registry, summarize_frame,
                                        validate_predictions)
from src.models.multiview import ModelConfig
from src.training.artifacts import write_json
from src.training.supervised import TrainingConfig, train_supervised
from tests.test_experiments import make_fixture


def predictions(participants=("p1", "p2"), perfect_second=False):
    rows = []
    for participant in participants:
        for night, count in [(1, 9 if participant == "p1" else 1), (2, 1)]:
            for epoch in range(count):
                correct = participant == "p1" or perfect_second
                probability = np.eye(5)[0 if correct else 1]
                record = participant + "n" + str(night)
                rows.append({"dataset": "synthetic", "participant_id": participant, "recording_id": record,
                             "epoch_index": epoch, "epoch_id": f"synthetic:{record}:{epoch:06d}", "true_stage": "W",
                             **{"prob_" + s: float(p) for s, p in zip(["W", "N1", "N2", "N3", "REM"], probability)}})
    return pd.DataFrame(rows)


class ClusterAnalysisTests(unittest.TestCase):
    def test_participants_not_nights_or_epochs_define_primary_endpoint_and_pairing(self):
        first = predictions()
        second = predictions(perfect_second=True)
        result = analyze_models([{"model_id": "a", "condition": "A", "training_seed": 42, "predictions": first},
                                 {"model_id": "b", "condition": "B", "training_seed": 42, "predictions": second}],
                                [("B", "A")], AnalysisConfig(bootstrap_replicates=1000))
        self.assertEqual(result["participants"], 2)
        self.assertAlmostEqual(result["conditions"]["A"]["participant_mean_macro_f1"], .1)
        self.assertAlmostEqual(result["conditions"]["B"]["participant_mean_macro_f1"], .2)
        self.assertNotAlmostEqual(result["models"]["a"]["macro_f1"], .1)
        self.assertAlmostEqual(result["paired_comparisons"][0]["participant_mean_macro_f1_difference"], .1)
        self.assertEqual(result["conditions"]["A"]["confidence_interval"]["lower"], 0.)
        self.assertEqual(result["conditions"]["A"]["confidence_interval"]["upper"], .2)
        self.assertTrue(all(row["recordings"] == 2 for row in result["participant_metrics"]))

    def test_trained_seeds_do_not_increase_independent_participant_count(self):
        models = [{"model_id": "a42", "condition": "A", "training_seed": 42, "predictions": predictions()},
                  {"model_id": "a43", "condition": "A", "training_seed": 43, "predictions": predictions(perfect_second=True)}]
        result = analyze_models(models, config=AnalysisConfig(bootstrap_replicates=200))
        self.assertEqual(result["participants"], 2)
        self.assertAlmostEqual(result["conditions"]["A"]["participant_mean_macro_f1"], .15)
        self.assertAlmostEqual(result["conditions"]["A"]["seed_standard_deviation"], .1 / np.sqrt(2))
        self.assertEqual(result, analyze_models(models, config=AnalysisConfig(bootstrap_replicates=200)))
        models.append({"model_id": "b42", "condition": "B", "training_seed": 42, "predictions": predictions()})
        with self.assertRaisesRegex(ValueError, "matching training seed"):
            analyze_models(models, [("A", "B")])

    def test_one_participant_produces_no_confidence_interval(self):
        result = analyze_models([{"model_id": "a", "condition": "A", "training_seed": 42, "predictions": predictions(("p1",))}])
        interval = result["conditions"]["A"]["confidence_interval"]
        self.assertIsNone(interval["lower"])
        self.assertEqual(interval["status"], "insufficient_independent_participants")

    def test_calibration_has_known_uniform_and_perfect_probability_values(self):
        frame = predictions().iloc[:3].copy()
        frame["true_stage"] = ["W", "N1", "N2"]
        for stage in ["W", "N1", "N2", "N3", "REM"]:
            frame["prob_" + stage] = .2
        summary = summarize_frame(frame)
        self.assertAlmostEqual(summary["loss"], np.log(5))
        self.assertAlmostEqual(summary["brier_score"], .8)
        self.assertAlmostEqual(summary["expected_calibration_error"], 1 / 3 - .2)
        self.assertEqual(summary["missing_truth_classes"], ["N3", "REM"])
        for number, stage in enumerate(["W", "N1", "N2", "N3", "REM"]):
            frame["prob_" + stage] = np.eye(5)[:3, number]
        perfect = summarize_frame(frame)
        self.assertEqual(perfect["brier_score"], 0.)
        self.assertEqual(perfect["expected_calibration_error"], 0.)

    def test_mismatched_targets_duplicate_ids_and_probabilities_are_rejected(self):
        first = predictions()
        with self.assertRaisesRegex(ValueError, "unique"):
            validate_predictions(pd.concat([first, first.iloc[:1]]))
        second = first.copy()
        second.loc[0, "true_stage"] = "N2"
        with self.assertRaisesRegex(ValueError, "identical epoch"):
            analyze_models([{"model_id": "a", "condition": "A", "training_seed": 1, "predictions": first},
                            {"model_id": "b", "condition": "B", "training_seed": 1, "predictions": second}])
        first.loc[0, "prob_W"] = .9
        with self.assertRaisesRegex(ValueError, "sum to one"):
            validate_predictions(first)
        for change in [{"bootstrap_replicates": 2}, {"confidence_level": 1.}, {"seed": True}]:
            with self.assertRaises(ValueError):
                AnalysisConfig(**change)

    def test_transition_analysis_preserves_physical_unknown_label_gaps(self):
        frame = pd.DataFrame({"target_stage": ["W", None, "N1", "N2"], "label_valid": [True, False, True, True]})
        np.testing.assert_array_equal(transition_mask(frame, 0), [False, False, False, True])
        np.testing.assert_array_equal(transition_mask(frame, 1), [False, False, True, True])


class FrozenEvaluationAndProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.split = make_fixture(cls.root)
        cls.bundles = {}
        for variant in ["raw", "spectral"]:
            config = ModelConfig(variant=variant, sequence_length=2, samples_per_epoch=3000, filters=(4, 8),
                                 kernels=(7, 5), strides=(10, 10), token_dim=8, attention_heads=2,
                                 temporal_units=8, spectral_units=4, dropout=0.)
            directory = cls.root / (variant + "_model")
            train_supervised(cls.root, directory, config, TrainingConfig(batch_size=2, epochs=1),
                              train_recordings=["SC4001", "SC4002"], validation_recordings=["SC4011", "SC4012"],
                              split_file=cls.split, max_train_batches=1, max_validation_batches=2, verbose=0)
            cls.bundles[variant] = directory
        registry = [{"model_id": v, "condition": v, "training_seed": 42, "bundle_directory": str(p),
                     "predictions": str(p / "validation_predictions.csv"), "selection_basis": "source_validation_only"} for v, p in cls.bundles.items()]
        cls.registry = cls.root / "registry.csv"
        pd.DataFrame(registry).to_csv(cls.registry, index=False)
        for participant in ["1", "2"]:
            raw = mne.io.RawArray(np.random.default_rng(int(participant)).normal(0, 1e-5, (1, 8 * 3000)),
                                 mne.create_info(["C3-A2"], 100, ["eeg"]), verbose=False)
            stages = ["W", "W", "N1", "N2", "N3", "REM", None, "W"]
            labels = pd.DataFrame({"epoch_index": np.arange(8), "raw_stage": stages, "target_stage": stages})
            processed = prepare_recording(raw, labels, dataset="isruc", participant_id=participant, recording_id=participant,
                                          split="external", channel_as_recorded="C3-A2", derivation_used="C3-A2",
                                          provenance={"synthetic_fixture": True, "annotation_policy": "explicit-workbook-epochs", "scorer": 1, "electrode_policy": "C3"})
            write_recording(processed, cls.root / "isruc" / participant)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def output(self, name):
        return self.root / (self._testMethodName + "_" + name)

    def configuration(self):
        return {"purpose": "integration", "target": {"dataset": "isruc", "split": "external", "expected_participants": 2,
                                                       "expected_recordings": 2, "scorer": 1, "electrode_policy": "C3"},
                "analysis": {"bootstrap_replicates": 200}, "paired_conditions": [["raw", "spectral"]]}

    def test_freezing_does_not_read_targets_and_snapshots_survive_original_relocation(self):
        directory = self.output("freeze")
        with mock.patch("src.evaluation.frozen.EpochStore", side_effect=AssertionError("target data read during freeze")):
            frozen = freeze_evaluation(self.registry, self.configuration(), directory)
        self.assertEqual(len(frozen["models"]), 2)
        original = self.bundles["raw"] / "model.keras"
        renamed = original.with_suffix(".temporarily_moved")
        original.rename(renamed)
        try:
            self.assertEqual(load_freeze(directory)["fingerprint"], frozen["fingerprint"])
        finally:
            renamed.rename(original)
        with self.assertRaises(FileExistsError):
            freeze_evaluation(self.registry, self.configuration(), directory)

    def test_synthetic_external_scoring_is_complete_paired_and_does_not_modify_models(self):
        directory = self.output("freeze")
        frozen = freeze_evaluation(self.registry, self.configuration(), directory)
        before = sha256_file(directory / "freeze.json")
        result = score_frozen(directory, self.root, self.output("scored"), split_file=self.split, batch_size=2, isolated_workers=False)
        self.assertEqual(result["participants"], 2)
        self.assertEqual(set(result["conditions"]), {"raw", "spectral"})
        self.assertEqual(result["models"]["raw"]["examples"], 10)
        self.assertEqual(result["models"]["raw"]["examples"], result["models"]["spectral"]["examples"])
        self.assertEqual(sha256_file(directory / "freeze.json"), before)
        self.assertTrue((self.output("scored") / "evaluation_bundle.json").exists())
        self.assertEqual({r["recordings"] for r in result["participant_metrics"]}, {1})
        exported = pd.read_csv(self.output("scored") / "raw_predictions.csv")
        self.assertEqual(set(exported.scorer), {1})
        self.assertTrue(exported.eligible.all())
        for name, digest in frozen["files"].items():
            self.assertEqual(sha256_file(directory / name), digest)

    def test_confirmatory_freeze_rejects_capped_models_before_target_access(self):
        configuration = self.configuration()
        configuration["purpose"] = "confirmatory"
        configuration["target"].update(expected_participants=100, expected_recordings=100)
        with mock.patch("src.evaluation.frozen.EpochStore", side_effect=AssertionError("target access")):
            with self.assertRaisesRegex(ValueError, "limited-data"):
                freeze_evaluation(self.registry, configuration, self.output("freeze"))

    def test_scorer_counts_and_preprocessing_are_checked_before_predictions(self):
        for name, changes, message in [("count", {"expected_participants": 3}, "incomplete"),
                                       ("scorer", {"scorer": 2}, "scorer/channel")]:
            configuration = self.configuration()
            configuration["target"].update(changes)
            directory = self.output(name)
            frozen = freeze_evaluation(self.registry, configuration, directory)
            with self.assertRaisesRegex(ValueError, message):
                inventory_cohort(frozen, self.root, self.split)

    def test_snapshot_tampering_and_runtime_code_changes_are_rejected(self):
        directory = self.output("freeze")
        frozen = freeze_evaluation(self.registry, self.configuration(), directory)
        with mock.patch("src.evaluation.frozen.code_hashes", return_value={}):
            with self.assertRaisesRegex(ValueError, "changed after freezing"):
                load_freeze(directory, require_runtime=True)
        path = directory / frozen["models"][0]["snapshot"] / "normalizer.json"
        with path.open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "snapshot integrity"):
            load_freeze(directory)

    def test_target_record_changes_and_incomplete_model_exports_do_not_publish_results(self):
        directory = self.output("freeze")
        frozen = freeze_evaluation(self.registry, self.configuration(), directory)
        output = self.output("score")
        with mock.patch("src.evaluation.frozen.predict_frozen_model") as predict:
            def truncated(freeze_dir, cohort, root, model_id, path, batch_size):
                frame = predictions(("p1",))
                frame.to_csv(path, index=False)
            predict.side_effect = truncated
            with self.assertRaisesRegex(ValueError, "enrolled recording"):
                score_frozen(directory, self.root, output, isolated_workers=False)
        self.assertFalse((output / "evaluation_bundle.json").exists())
        self.assertEqual(json.loads((output / "status.json").read_text())["status"], "failed")

    def test_development_analysis_checks_actual_validation_role(self):
        result = analyze_registry(self.registry, self.output("analysis"), pairs=[("raw", "spectral")], config=AnalysisConfig(bootstrap_replicates=200))
        self.assertEqual(result["participants"], 1)  # both source nights are SC401.
        inspected = inspect_bundle(self.bundles["raw"])
        fake = pd.read_csv(self.bundles["raw"] / "validation_predictions.csv")
        fake["recording_id"] = "SC4021"
        fake["participant_id"] = "SC402"
        with self.assertRaisesRegex(ValueError, "outside the model's source-validation"):
            verify_validation_frame(fake, inspected)

    def test_enrolled_record_content_changes_are_detected_before_prediction(self):
        directory = self.output("freeze")
        frozen = freeze_evaluation(self.registry, self.configuration(), directory)
        cohort = self.output("cohort.json")
        write_json(cohort, {"freeze_fingerprint": frozen["fingerprint"], "records": inventory_cohort(frozen, self.root, self.split)})
        manifest = self.root / "isruc/1/manifest.json"
        original = manifest.read_bytes()
        try:
            manifest.write_bytes(original + b"\n")
            with self.assertRaisesRegex(ValueError, "changed after cohort enrollment"):
                predict_frozen_model(directory, cohort, self.root, "raw", self.output("predictions.csv"), 2)
        finally:
            manifest.write_bytes(original)

    def test_profiles_use_source_data_and_leave_original_artifacts_unchanged(self):
        bundle = self.bundles["raw"]
        before = sha256_file(bundle / "model.keras")
        for mode in ["inference", "training"]:
            result = profile_bundle(bundle, self.root, self.output(mode),
                                     ProfileConfig(batch_size=2, warmup_steps=1, measured_steps=1, mode=mode),
                                     recordings=["SC4001"], split_file=self.split)
            self.assertTrue(result["original_saved_artifacts_unchanged"])
            self.assertGreater(result["model_only"]["examples_per_second"], 0.)
            self.assertGreaterEqual(result["host_memory"]["maximum_sampled_rss_bytes"], result["host_memory"]["baseline_rss_bytes"])
            self.assertEqual(sha256_file(bundle / "model.keras"), before)
        with self.assertRaisesRegex(ValueError, "declared source partition"):
            profile_bundle(bundle, self.root, self.output("wrong_partition"), recordings=["SC4021"], split_file=self.split)
        with self.assertRaises(ValueError):
            ProfileConfig(mode="training", split="val")
        with self.assertRaises(ValueError):
            ProfileConfig(batch_size=0)

    def test_protocol_invalid_caps_and_target_roles_are_rejected(self):
        for changes in [{"dataset": "isruc", "split": "train"}, {"expected_participants": 0}, {"scorer": 3}, {"max_windows_per_record": 0}]:
            configuration = self.configuration()
            configuration["target"].update(changes)
            with self.assertRaises(ValueError):
                normalize_protocol(configuration)


if __name__ == "__main__":
    unittest.main()

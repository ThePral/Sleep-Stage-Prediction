"""Participant label budgets, matched statistics and recoverable study execution."""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import mne
import numpy as np
import pandas as pd

from src.data.contrastive import SourceEpochPool
from src.data.epoch_store import EpochStore, sha256_file, write_recording
from src.data.label_budgets import (authorize_waveform_pool, build_budget_plan, fingerprint, get_budget,
                                    load_budget_plan, validate_plan, verify_plan_scope)
from src.data.research import prepare_recording
from src.data.supervised import SourcePartition, WindowBatches
from src.experiments.run import (CONDITIONS, completed_attempt, create_study, execute_study,
                                 load_study, make_jobs, normalize_config, write_tables)
from src.experiments.worker import run_task
from src.inference.multiview import MultiviewPredictor
from src.preprocessing.research_normalization import ResearchNormalizer
from src.preprocessing.waveform_normalization import WaveformNormalizer
from src.training.artifacts import write_json
from src.training.supervised import TrainingConfig, train_supervised
from src.models.multiview import ModelConfig


def make_fixture(root):
    assignment = {"SC400": "train", "SC403": "train", "SC404": "train", "SC405": "train", "SC401": "val", "SC402": "test"}
    split = root / "split.csv"
    pd.DataFrame([{"participant_id": p, "split": s} for p, s in assignment.items()]).to_csv(split, index=False)
    for number, (participant, partition) in enumerate(assignment.items()):
        if partition == "test":
            # A held-out identity row exists, but no valid training store or labels.
            directory = root / "sleepedf" / (participant + "1")
            directory.mkdir(parents=True)
            (directory / "manifest.json").write_text("not a training manifest")
            pd.DataFrame([{"dataset": "sleepedf", "participant_id": participant,
                           "recording_id": participant + "1", "split": "test", "target_stage": "DO_NOT_READ"}]).to_csv(directory / "epochs.csv", index=False)
            continue
        for night in [1, 2]:
            name = participant + str(night)
            values = np.random.default_rng(number + night).normal(0, (number + 1) * 1e-5, (1, 8 * 3000))
            raw = mne.io.RawArray(values, mne.create_info(["EEG"], 100, ["eeg"]), verbose=False)
            stages = ["W", "W", "N1", "N2", "N3", "REM", None, "W"]
            labels = pd.DataFrame({"epoch_index": np.arange(8), "raw_stage": stages, "target_stage": stages})
            processed = prepare_recording(raw, labels, dataset="sleepedf", participant_id=participant,
                                          recording_id=name, split=partition, channel_as_recorded="EEG", derivation_used="EEG")
            write_recording(processed, root / "sleepedf" / name)
    return split


def tiny_study_config(conditions=None):
    return {"model": {"sequence_length": 2, "samples_per_epoch": 3000,
                       "filters": [4, 8], "kernels": [7, 5], "strides": [10, 10],
                       "token_dim": 8, "attention_heads": 2, "temporal_units": 8, "spectral_units": 4, "dropout": 0.},
            "training": {"epochs": 1, "batch_size": 2, "patience": 1},
            "pretraining": {"epochs": 1, "batch_size": 4, "projection_dim": 8, "diagnostic_epochs": 4},
            "fractions": [.25], "training_seeds": [42], "draw_seeds": [42],
            "conditions": conditions or ["M0", "M1", "M4"],
            "limits": {"max_train_batches": 1, "max_validation_batches": 2, "max_pretraining_batches": 1}}


class LabelBudgetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.split = make_fixture(self.root)

    def test_draws_are_nested_whole_participants_and_annotation_costs_include_validation(self):
        plan = build_budget_plan(self.root, fractions=[.25, .5, 1.], draw_seeds=[42, 43], sequence_length=2, split_file=self.split)
        again = build_budget_plan(self.root, fractions=[.25, .5, 1.], draw_seeds=[42, 43], sequence_length=2, split_file=self.split)
        self.assertEqual(plan, again)
        self.assertEqual(len(plan["budgets"]), 6)
        for seed in [42, 43]:
            entries = [b for b in plan["budgets"] if b["draw_seed"] == seed]
            self.assertEqual([len(b["participants"]) for b in entries], [1, 2, 4])
            self.assertTrue(set(entries[0]["participants"]) < set(entries[1]["participants"]) < set(entries[2]["participants"]))
            for entry in entries:
                self.assertEqual(len(entry["recordings"]), 2 * len(entry["participants"]))
                self.assertEqual(entry["annotation_cost"]["annotated_epochs"], 14 * len(entry["participants"]))
                self.assertAlmostEqual(entry["annotation_cost"]["annotated_hours"], 14 * len(entry["participants"]) / 120)
                self.assertTrue(all(entry["annotation_cost"]["target_stage_counts"].values()))
        self.assertEqual(plan["validation_annotation_cost"]["participants"], 1)
        self.assertEqual(plan["validation_annotation_cost"]["annotated_epochs"], 14)
        self.assertNotIn("SC402", {r["participant_id"] for r in plan["source_records"]})
        verify_plan_scope(plan, self.root, self.split)

    def test_invalid_draws_budgets_and_held_out_scopes_are_rejected(self):
        for change in [{"fractions": [0.]}, {"fractions": [.5, .25]}, {"fractions": [.01, .02]},
                       {"draw_seeds": [42, 42]}, {"train_participants": ["SC402"]}, {"validation_participants": ["SC400"]}]:
            with self.subTest(change=change), self.assertRaises(ValueError):
                build_budget_plan(self.root, split_file=self.split, sequence_length=2, **change)
        # All five classes exist as epochs, but an overly long context leaves no targets.
        with self.assertRaisesRegex(ValueError, "lacks eligible target classes"):
            build_budget_plan(self.root, split_file=self.split, sequence_length=10)

    def test_manifest_tampering_and_source_scope_changes_are_detected(self):
        plan = build_budget_plan(self.root, split_file=self.split, sequence_length=2)
        path = self.root / "budgets.json"
        write_json(path, plan)
        self.assertEqual(load_budget_plan(path), plan)
        altered = json.loads(json.dumps(plan))
        altered["budgets"][0]["participants"] = ["SC402"]
        with self.assertRaisesRegex(ValueError, "integrity"):
            validate_plan(altered)
        altered["fingerprint"] = fingerprint({k: v for k, v in altered.items() if k != "fingerprint"})
        with self.assertRaisesRegex(ValueError, "nested draws"):
            validate_plan(altered)
        with self.assertRaisesRegex(ValueError, "Unknown label budget"):
            get_budget(plan, "not-a-budget")
        with (self.root / "sleepedf/SC4001/manifest.json").open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "changed"):
            verify_plan_scope(plan, self.root, self.split)

    def test_full_unlabeled_pool_transfer_requires_explicit_content_matched_subset(self):
        plan = build_budget_plan(self.root, split_file=self.split, sequence_length=2)
        budget = plan["budgets"][0]
        with SourceEpochPool(self.root, split_file=self.split) as pool, \
             SourcePartition(self.root, "train", sequence_length=2, recordings=budget["recordings"], split_file=self.split) as labeled:
            waveform = WaveformNormalizer().fit(pool)
            self.assertEqual(waveform.state["fit_epochs"], 64)
            authorize_waveform_pool(plan, waveform)
            with self.assertRaisesRegex(ValueError, "selections disagree"):
                waveform.assert_compatible(labeled, require_training_match=True)
            normalizer = ResearchNormalizer().fit(labeled).use_pretraining_waveform(waveform, labeled, allow_source_subset=True)
            self.assertEqual(normalizer.state["fit_epochs"], 14)
            self.assertEqual(normalizer.state["pretraining_waveform"]["fit_epochs"], 64)
            normalizer.save(self.root / "normalizer.json")
            ResearchNormalizer.load(self.root / "normalizer.json")
            batches = WindowBatches(labeled, normalizer, variant="spectral", batch_size=2)
            with mock.patch.object(labeled, "_waveforms", side_effect=AssertionError("spectral batching must not read raw samples")):
                inputs, targets, _ = next(batches.iter_batches(epoch=0))
                self.assertEqual(set(inputs), {"spectral"})
                self.assertEqual(inputs["spectral"].shape, (2, 2, 10))
            changed = WaveformNormalizer(json.loads(json.dumps(waveform.state)))
            changed.state["training_data"]["recordings"][0]["manifest_sha256"] = "wrong"
            with self.assertRaisesRegex(ValueError, "fixed budget training scope"):
                authorize_waveform_pool(plan, changed)

    def test_standalone_budget_training_rejects_recording_overrides_or_missing_fixed_scale(self):
        plan = build_budget_plan(self.root, split_file=self.split, sequence_length=2)
        path = self.root / "plan.json"
        write_json(path, plan)
        config = ModelConfig(sequence_length=2)
        kwargs = {"split_file": self.split, "budget_plan_path": path, "budget_id": plan["budgets"][0]["budget_id"]}
        with self.assertRaisesRegex(ValueError, "controls complete"):
            train_supervised(self.root, self.root / "override", config, train_recordings=["SC4001"], **kwargs)
        with self.assertRaisesRegex(ValueError, "fixed full-pool"):
            train_supervised(self.root, self.root / "no_scale", config, TrainingConfig(epochs=1), verbose=0, **kwargs)


class StudyRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.split = make_fixture(self.root)

    def test_core_matrix_shares_budget_normalizers_seeds_and_ssl_checkpoints(self):
        config = normalize_config({"fractions": [.25, 1.], "draw_seeds": [42, 43], "training_seeds": [42, 43, 44]})
        plan = build_budget_plan(self.root, fractions=[.25, 1.], draw_seeds=[42, 43], sequence_length=2, split_file=self.split)
        jobs = make_jobs(config, plan["budgets"])
        self.assertEqual(len([j for j in jobs if j["kind"] == "pretrain"]), 3)
        self.assertEqual(len([j for j in jobs if j["kind"] == "supervised"]), 4 * 3 * 7)
        self.assertEqual(len({j["job_id"] for j in jobs}), len(jobs))
        for budget in plan["budgets"]:
            for seed in [42, 43, 44]:
                rows = [j for j in jobs if j.get("budget_id") == budget["budget_id"] and j.get("seed") == seed]
                self.assertEqual({j["condition"] for j in rows}, set(CONDITIONS))
                self.assertTrue(all(j["dependencies"][0] == "normalizer_" + budget["budget_id"] for j in rows))
                self.assertTrue(all("ssl_seed" + str(seed) in j["dependencies"] for j in rows if j["pretrained"]))

    def test_configuration_rejects_unpaired_or_unknown_options(self):
        for config in [{"conditions": ["ISRUC"]}, {"conditions": ["M1", "M1"]}, {"training_seeds": [42, 42]},
                       {"model": {"variant": "gated"}}, {"training": {"seed": 42}}, {"limits": {"max_train_batches": 0}}]:
            with self.subTest(config=config), self.assertRaises(ValueError):
                normalize_config(config)

    def test_scheduler_skips_complete_jobs_preserves_failed_attempts_and_detects_tampering(self):
        study_dir = self.root / "study"
        study = create_study(self.root, study_dir, tiny_study_config(["M1"]), split_file=self.split)
        first = execute_study(study_dir, max_jobs=1, executor=run_task)
        self.assertEqual(first["jobs_completed"], 1)
        existing = completed_attempt(study_dir, study, "waveform_statistics")
        digest = sha256_file(existing / "job_result.json")

        def fail(task):
            raise RuntimeError("simulated worker failure")

        with self.assertRaisesRegex(RuntimeError, "simulated worker"):
            execute_study(study_dir, max_jobs=1, executor=fail)
        with self.assertRaisesRegex(ValueError, "retry-failed"):
            execute_study(study_dir, max_jobs=1, executor=run_task)
        result = execute_study(study_dir, max_jobs=1, retry_failed=True, executor=run_task)
        self.assertEqual(result["jobs_completed"], 2)
        normalizer = completed_attempt(study_dir, study, "normalizer_" + load_study(study_dir)[1]["budgets"][0]["budget_id"])
        self.assertEqual(normalizer.name, "attempt_0002")
        self.assertTrue((normalizer.parent / "attempt_0001/task.json").exists())
        self.assertEqual(sha256_file(existing / "job_result.json"), digest)
        with (existing / "artifacts/waveform_normalizer.json").open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "integrity"):
            execute_study(study_dir, max_jobs=1, executor=run_task)
        self.assertFalse((study_dir / ".runner.lock").exists())

    def test_planned_code_change_or_concurrent_runner_prevents_execution(self):
        study_dir = self.root / "study"
        create_study(self.root, study_dir, tiny_study_config(["M1"]), split_file=self.split)
        with mock.patch("src.experiments.run.code_hashes", return_value={}):
            with self.assertRaisesRegex(ValueError, "Implementation changed"):
                execute_study(study_dir, executor=run_task)
        (study_dir / ".runner.lock").write_text("another active scheduler")
        with self.assertRaises(FileExistsError):
            execute_study(study_dir, executor=run_task)
        self.assertTrue((study_dir / ".runner.lock").exists())
        with self.assertRaises(FileExistsError):
            create_study(self.root, study_dir, tiny_study_config(["M1"]), split_file=self.split)

    def test_interrupted_completion_publication_does_not_leave_a_partial_final_seal(self):
        study_dir = self.root / "study"
        study = create_study(self.root, study_dir, tiny_study_config(["M1"]), split_file=self.split)
        original_replace = os.replace

        def interrupted_replace(source, destination):
            if Path(destination).name == "job_result.json":
                raise RuntimeError("interrupted before final completion publication")
            return original_replace(source, destination)

        def interrupted_worker(task):
            with mock.patch("src.experiments.run.os.replace", side_effect=interrupted_replace):
                run_task(task)

        with self.assertRaisesRegex(RuntimeError, "interrupted before final"):
            execute_study(study_dir, max_jobs=1, executor=interrupted_worker)
        failed = study_dir / "jobs/waveform_statistics/attempt_0001"
        self.assertFalse((failed / "job_result.json").exists())
        self.assertTrue((failed / "job_result.tmp").exists())
        result = execute_study(study_dir, max_jobs=1, retry_failed=True, executor=run_task)
        self.assertEqual(result["jobs_completed"], 1)
        self.assertEqual(completed_attempt(study_dir, study, "waveform_statistics").name, "attempt_0002")

    def test_budgeted_spectral_scratch_and_ssl_training_export_matching_inference(self):
        study_dir = self.root / "study"
        study = create_study(self.root, study_dir, tiny_study_config(), split_file=self.split)
        summary = execute_study(study_dir, executor=run_task)
        self.assertEqual(summary["jobs_completed"], 6)
        self.assertEqual(summary["jobs_planned"], 6)
        rows = pd.read_csv(study_dir / "source_validation_results.csv")
        self.assertEqual(set(rows.condition), {"M0", "M1", "M4"})
        self.assertEqual(rows.normalizer_sha256.nunique(), 1)
        self.assertEqual(rows.validation_targets_sha256.nunique(), 1)
        self.assertEqual(rows.labeled_training_participants.tolist(), [1, 1, 1])
        paired = pd.read_csv(study_dir / "paired_source_validation.csv")
        self.assertEqual(paired[["first", "second"]].values.tolist(), [["M4", "M1"]])
        for job in [j for j in study["jobs"] if j["kind"] == "supervised"]:
            attempt = completed_attempt(study_dir, study, job["job_id"])
            predictor = MultiviewPredictor(attempt / "artifacts")
            predictions = pd.read_csv(attempt / "artifacts/validation_predictions.csv")
            self.assertEqual(predictor.normalizer.state["fit_epochs"], 14)
            self.assertEqual(predictor.normalizer.state["pretraining_waveform"]["fit_epochs"], 64)
            batches = []
            with EpochStore(self.root / "sleepedf" / predictions.recording_id.iloc[0]) as store:
                positions = store.sequence_positions(2)
                selected = np.stack([positions[positions[:, -1] == e][0] for e in predictions[predictions.recording_id == store.epochs.recording_id.iloc[0]].epoch_index])
                data = store.gather(selected)
                _, probabilities = predictor.predict_batch(data["waveform"], data["spectral"])
                if job["condition"] == "M0":
                    _, spectral_only = predictor.predict_batch(spectral_features=data["spectral"])
                    np.testing.assert_array_equal(spectral_only, probabilities)
                expected = predictions[predictions.recording_id == store.epochs.recording_id.iloc[0]][["prob_" + s for s in ["W", "N1", "N2", "N3", "REM"]]].to_numpy()
                np.testing.assert_allclose(probabilities, expected, atol=1e-6)
        # Continuing a completed study changes no model artifacts or training.
        repeated = execute_study(study_dir, executor=lambda _: self.fail("completed jobs were rerun"))
        self.assertEqual(repeated["jobs_executed_this_call"], 0)
        # A comparison must also reject a normalization/target mismatch.
        ledger = json.loads((study_dir / "ledger.json").read_text())
        changed = completed_attempt(study_dir, study, rows[rows.condition == "M4"].job_id.iloc[0])
        with (changed / "artifacts/normalizer.json").open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "different normalization"):
            write_tables(study_dir, study, ledger)


if __name__ == "__main__":
    unittest.main()

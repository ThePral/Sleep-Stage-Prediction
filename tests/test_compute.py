"""Safe source-only transport and actual synthetic training kernel checks."""

import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest import mock

import tensorflow as tf

from src.compute.capsule import create_capsule, unpack_capsule, verify_extracted, MANIFEST
from src.compute.preflight import PreflightConfig, run_preflight
from src.data.epoch_store import sha256_file
from src.data.label_budgets import fingerprint
from tests.test_experiments import make_fixture


class SourceCapsuleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.project = self.root / "project"
        (self.project / "src").mkdir(parents=True)
        (self.project / "src/example.py").write_text("VALUE = 42\n")
        (self.project / "scripts").mkdir()
        (self.project / "scripts/research.ps1").write_text("Write-Output 'launcher fixture'\n")
        (self.project / "scripts/configure_wsl_cuda.py").write_text("# Environment configuration fixture.\n")
        (self.project / "TRAINING_RUNBOOK.md").write_text("User-run training commands.\n")
        (self.project / "AGENTS.md").write_text("Training belongs to the user.\n")
        (self.project / ".env").write_text("SECRET=do-not-copy")
        (self.project / "src/.private.py").write_text("SECRET=do-not-copy")
        (self.project / ".aws").mkdir()
        (self.project / ".aws/credentials").write_text("do-not-copy")
        self.data = self.root / "prepared"
        self.data.mkdir()
        self.split = make_fixture(self.data)
        (self.data / "isruc").mkdir()
        (self.data / "isruc/private-target-labels.csv").write_text("do-not-copy")
        self.archive = self.root / "source.tar"
        self.result = create_capsule(self.project, self.data, self.split, self.archive)

    def test_roundtrip_preserves_hashes_and_excludes_held_out_data_and_private_files(self):
        with tarfile.open(self.archive, "r:") as tar:
            names = tar.getnames()
        self.assertNotIn(".env", names)
        for name in ["scripts/research.ps1", "scripts/configure_wsl_cuda.py", "TRAINING_RUNBOOK.md", "AGENTS.md"]:
            self.assertIn(name, names)
        self.assertFalse(any(".aws" in n or ".private" in n or "isruc" in n or "SC4021" in n for n in names))
        self.assertEqual(len(self.result["source_scope"]["training_participants"]), 4)
        destination = self.root / "restored"
        unpack_capsule(self.archive, destination, self.result["archive_sha256"])
        verified = verify_extracted(destination)
        self.assertEqual(verified["training_participants"], 4)
        self.assertEqual(verified["validation_participants"], 1)
        self.assertEqual(verified["manifest_fingerprint"], self.result["manifest_fingerprint"])
        self.assertEqual((destination / "src/example.py").read_bytes(), (self.project / "src/example.py").read_bytes())

    def test_wrong_archive_hash_fails_before_creating_destination(self):
        destination = self.root / "wrong"
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            unpack_capsule(self.archive, destination, "0" * 64)
        self.assertFalse(destination.exists())

    def test_existing_artifacts_are_not_overwritten(self):
        with self.assertRaises(FileExistsError):
            create_capsule(self.project, self.data, self.split, self.archive)
        destination = self.root / "existing"
        destination.mkdir()
        (destination / "preserve.txt").write_text("user work")
        with self.assertRaises(FileExistsError):
            unpack_capsule(self.archive, destination, self.result["archive_sha256"])
        self.assertEqual((destination / "preserve.txt").read_text(), "user work")

    def altered_archive(self, mode):
        with tarfile.open(self.archive, "r:") as source:
            contents = [(member, source.extractfile(member).read()) for member in source.getmembers()]
        manifest = json.loads(contents[0][1])
        if mode == "traversal":
            manifest["files"]["../outside.py"] = manifest["files"].pop("src/example.py")
        if mode == "held_out":
            manifest["source_scope"]["recordings"][0]["split"] = "test"
        manifest["fingerprint"] = fingerprint({k: v for k, v in manifest.items() if k != "fingerprint"})
        contents[0] = (contents[0][0], json.dumps(manifest).encode())
        path = self.root / (mode + ".tar")
        with tarfile.open(path, "w") as target:
            for old, data in contents:
                name = "../outside.py" if mode == "traversal" and old.name == "src/example.py" else old.name
                member = tarfile.TarInfo(name)
                member.size = len(data)
                if mode == "link" and name == "src/example.py":
                    member.type = tarfile.SYMTYPE
                    member.linkname = "../../outside.py"
                    target.addfile(member)
                else:
                    target.addfile(member, io.BytesIO(data))
        return path

    def test_path_traversal_links_and_held_out_scope_are_rejected_before_extraction(self):
        for mode in ["traversal", "link", "held_out"]:
            archive = self.altered_archive(mode)
            destination = self.root / ("rejected_" + mode)
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                unpack_capsule(archive, destination, sha256_file(archive))
            self.assertFalse(destination.exists())

    def test_modified_extracted_content_is_detected(self):
        destination = self.root / "restored"
        unpack_capsule(self.archive, destination, self.result["archive_sha256"])
        (destination / "src/example.py").write_text("modified")
        with self.assertRaisesRegex(ValueError, "content changed"):
            verify_extracted(destination)


class TrainingKernelPreflightTests(unittest.TestCase):
    def test_configuration_rejects_invalid_batches(self):
        for changes in [{"contrastive_batch_size": 1}, {"supervised_batch_size": 0}, {"measured_steps": 0}, {"require_gpu": "yes"}]:
            with self.assertRaises(ValueError):
                PreflightConfig(**changes)

    def test_missing_gpu_fails_closed_without_training(self):
        original = tf.config.list_physical_devices

        def devices(kind=None):
            return [] if kind == "GPU" else original(kind)

        with tempfile.TemporaryDirectory() as folder, mock.patch("tensorflow.config.list_physical_devices", side_effect=devices):
            result = run_preflight(Path(folder) / "report.json", PreflightConfig(measured_steps=1, warmup_steps=1))
            self.assertFalse(result["passed"])
            self.assertIn("no GPU", result["error"]["message"])
            self.assertEqual(result["checks"], {})

    def test_synthetic_supervised_and_contrastive_updates_execute_and_reports_are_immutable(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "report.json"
            config = PreflightConfig(supervised_batch_size=1, contrastive_batch_size=2, warmup_steps=1, measured_steps=1, require_gpu=False)
            result = run_preflight(path, config)
            self.assertTrue(result["passed"], result.get("error"))
            self.assertEqual(result["checks"]["gated_supervised_updates"]["parameters"], 82757)
            self.assertEqual(result["checks"]["contrastive_updates"]["maximum_negatives_per_anchor"], 2)
            self.assertEqual(result["data_access"], "synthetic_only_no_recordings_or_labels_loaded")
            with self.assertRaises(FileExistsError):
                run_preflight(path, config)


if __name__ == "__main__":
    unittest.main()

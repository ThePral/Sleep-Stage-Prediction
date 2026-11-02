"""Training ownership checks using mocks only; no optimizer steps execute."""

from pathlib import Path
import json
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from src.compute.devices import require_gpu_device
from src.experiments.run import execute_study, main


class GPUHandoffTests(unittest.TestCase):
    def test_worker_rechecks_frozen_runtime_before_creating_outputs(self):
        from src.experiments.worker import run_task

        with tempfile.TemporaryDirectory() as folder:
            task = Path(folder) / "task.json"
            task.write_text(json.dumps({"study_directory": folder, "job_id": "first",
                                        "dependencies": {}, "data_root": folder, "split_file": "split.csv"}))
            study = {"jobs": [{"job_id": "first", "dependencies": []}]}
            with mock.patch("src.experiments.worker.load_study", return_value=(study, {})), \
                 mock.patch("src.experiments.worker.verify_runtime", side_effect=ValueError("Implementation changed")) as verify:
                with self.assertRaisesRegex(ValueError, "Implementation changed"):
                    run_task(task)
            verify.assert_called_once_with(study, {}, Path(folder), Path("split.csv"))
            self.assertFalse((Path(folder) / "artifacts").exists())

    def test_no_gpu_fails_without_building_a_model(self):
        tensorflow = SimpleNamespace(config=SimpleNamespace(list_physical_devices=mock.Mock(return_value=[])))
        with mock.patch.dict("sys.modules", {"tensorflow": tensorflow}):
            with self.assertRaisesRegex(RuntimeError, "exposes no GPU"):
                require_gpu_device()
        tensorflow.config.list_physical_devices.assert_called_once_with("GPU")

    def test_visible_devices_are_returned_without_executing_kernels(self):
        tensorflow = SimpleNamespace(config=SimpleNamespace(list_physical_devices=mock.Mock(
            return_value=[SimpleNamespace(name="/physical_device:GPU:0")])))
        with mock.patch.dict("sys.modules", {"tensorflow": tensorflow}):
            self.assertEqual(require_gpu_device(), ["/physical_device:GPU:0"])

    def test_execution_guard_precedes_study_reads_and_worker_execution(self):
        with mock.patch("src.compute.devices.require_gpu_device", side_effect=RuntimeError("GPU unavailable")), \
             mock.patch("src.experiments.run.load_study") as load, \
             mock.patch("src.experiments.run.subprocess.run") as subprocess_run:
            with self.assertRaisesRegex(RuntimeError, "GPU unavailable"):
                execute_study("unused", require_gpu=True)
            load.assert_not_called()
            subprocess_run.assert_not_called()

    def test_device_policy_rejects_nonboolean_values(self):
        with self.assertRaisesRegex(ValueError, "boolean"):
            execute_study("unused", require_gpu="yes")

    def test_cli_forwards_required_gpu_without_executing_a_worker(self):
        with mock.patch("src.experiments.run.execute_study", return_value={}) as execute:
            main(["execute", "--study-dir", "unused", "--require-gpu", "--max-jobs", "2"])
        self.assertTrue(execute.call_args.kwargs["require_gpu"])
        self.assertEqual(execute.call_args.kwargs["max_jobs"], 2)

    def test_cli_keeps_cpu_execution_an_explicitly_available_path(self):
        with mock.patch("src.experiments.run.execute_study", return_value={}) as execute:
            main(["execute", "--study-dir", "unused"])
        self.assertFalse(execute.call_args.kwargs["require_gpu"])
        self.assertEqual(execute.call_args.args, (Path("unused"),))


if __name__ == "__main__":
    unittest.main()

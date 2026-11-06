"""Inspect a frozen source study without importing TensorFlow or changing it."""

import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.experiments.run import code_hashes, load_study, verify_job_result
from src.data.label_budgets import verify_plan_scope


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def scheduler_process(directory):
    lock = directory / ".runner.lock"
    if not lock.exists():
        return None, "no_scheduler_lock"
    try:
        process = psutil.Process(int(lock.read_text().strip()))
        args = process.cmdline()
        module = args[args.index("-m") + 1]
        target = Path(args[args.index("--study-dir") + 1])
        if not target.is_absolute():
            target = Path(process.cwd()) / target
        if module != "src.experiments.run" or "execute" not in args or target.resolve() != directory:
            return None, "lock_pid_does_not_match_this_scheduler"
        return process, "verified_live_scheduler"
    except (psutil.NoSuchProcess, ProcessLookupError):
        return None, "lock_pid_is_no_longer_alive"
    except psutil.AccessDenied:
        return None, "scheduler_identity_unavailable"
    except (ValueError, IndexError, OSError):
        return None, "scheduler_identity_unverified"


def process_sample(scheduler):
    result = {}
    if scheduler is None:
        return result
    try:
        processes = [scheduler, *scheduler.children(recursive=True)]
    except psutil.Error:
        return result
    for process in processes:
        try:
            with process.oneshot():
                cpu = process.cpu_times()
                result[process.pid] = {"pid": process.pid, "created": process.create_time(),
                    "rss_bytes": process.memory_info().rss, "cpu_seconds": cpu.user + cpu.system}
        except psutil.Error:
            continue
    return result


def inspect_data_scope(study, plan):
    """Check frozen source identities/manifests without loading waveforms or labels."""
    result = {"data_root": study["data_root"], "split_file": study["split_file"],
              "verified": False, "error": None}
    try:
        verify_plan_scope(plan, Path(study["data_root"]), Path(study["split_file"]))
    except (ValueError, OSError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    else:
        result["verified"] = True
    return result


def inspect_study(directory, *, sample_seconds=3., verify_artifacts=False, verify_data_scope=False):
    directory = Path(directory).resolve()
    study, plan = load_study(directory)
    ledger = read_json(directory / "ledger.json")
    if ledger["study_fingerprint"] != study["fingerprint"]:
        raise ValueError("Ledger does not belong to this study.")
    scheduler, identity = scheduler_process(directory)
    before = process_sample(scheduler)
    if scheduler is not None:
        time.sleep(sample_seconds)
    after = process_sample(scheduler)
    cpu_delta = sum(max(0., item["cpu_seconds"] - before[pid]["cpu_seconds"])
                    for pid, item in after.items() if pid in before and item["created"] == before[pid]["created"])
    counts = {name: 0 for name in ["completed", "running", "failed", "pending"]}
    active = []
    verified = []
    for job in study["jobs"]:
        entry = ledger["jobs"].get(job["job_id"], {"status": "pending"})
        counts[entry["status"]] += 1
        if entry["status"] == "completed" and verify_artifacts:
            verify_job_result(directory / entry["attempt"], study, job["job_id"])
            verified.append(job["job_id"])
        if entry["status"] != "running":
            continue
        attempt = (directory / entry["attempt"]).resolve()
        if not attempt.is_relative_to(directory / "jobs"):
            raise ValueError("Attempt path escaped its study.")
        artifacts = attempt / "artifacts"
        item = {"job_id": job["job_id"], "kind": job["kind"], "log": str(attempt.with_suffix(".log"))}
        if (artifacts / "status.json").exists():
            item["recorded_status"] = read_json(artifacts / "status.json")
        if (artifacts / "resolved_config.json").exists():
            item["devices"] = read_json(artifacts / "resolved_config.json").get("devices")
        if job["kind"] == "pretrain":
            seal_path = artifacts / "resume_state.json"
            # Only an atomically published epoch seal advances this count.
            item["published_epochs"] = read_json(seal_path)["completed_epochs"] if seal_path.exists() else 0
            item["scheduled_epochs"] = study["config"]["pretraining"]["epochs"]
        elif job["kind"] == "supervised" and (artifacts / "training_history.csv").exists():
            with (artifacts / "training_history.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            item["logged_epochs"] = len(rows)
            item["maximum_epochs"] = study["config"]["training"]["epochs"]
        active.append(item)
    packages_match = all(importlib.metadata.version(p) == version for p, version in study["packages"].items())
    status = (("completed_artifacts_verified" if verify_artifacts else "completed_in_ledger")
              if counts["completed"] == len(study["jobs"]) else
              "failed_attempt_present" if counts["failed"] else
              "running" if scheduler is not None and after else "no_verified_live_scheduler")
    return {"observed_at_utc": datetime.now(timezone.utc).isoformat(), "status": status,
            "study_directory": str(directory), "study_fingerprint": study["fingerprint"],
            "limited_run": study["limited_run"], "jobs_planned": len(study["jobs"]), "jobs": counts,
            "scheduler_identity": identity, "processes": list(after.values()),
            "sample_seconds": sample_seconds, "observed_process_tree_cpu_seconds": round(cpu_delta, 4),
            "sampled_process_tree_rss_bytes": sum(p["rss_bytes"] for p in after.values()),
            "host_available_memory_bytes": psutil.virtual_memory().available,
            "workspace_free_bytes": shutil.disk_usage(ROOT).free,
            "c_drive_free_bytes": shutil.disk_usage("C:/").free if sys.platform == "win32" else None,
            "source_hashes_match": study["code_sha256"] == code_hashes(), "packages_match": packages_match,
            "data_scope": inspect_data_scope(study, plan) if verify_data_scope else None,
            "training_participants": len(plan["training_participants"]),
            "validation_participants": len(plan["validation_participants"]),
            "completed_artifacts_verified": verified if verify_artifacts else None,
            "active_jobs": active, "tensorflow_imported": "tensorflow" in sys.modules,
            "limitations": ["A process sample is not proof of optimizer progress; epoch seals provide stronger evidence.",
                            "Ledger and process observations can span a job transition; rerun if they disagree.",
                            "Published epoch counts do not establish convergence; no predictive evaluation is performed."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--sample-seconds", type=float, default=3.)
    parser.add_argument("--verify-artifacts", action="store_true")
    parser.add_argument("--verify-data-scope", action="store_true",
                        help="Verify frozen source participants, recording manifests and split at the saved data root.")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if not 0 < args.sample_seconds <= 30:
        parser.error("--sample-seconds must be in (0, 30].")
    if args.report is not None and args.report.exists():
        parser.error("Use a new report path; previous observations are preserved.")
    if args.report is not None and args.report.resolve().is_relative_to(args.study_dir.resolve()):
        parser.error("Save monitoring reports outside the frozen study directory.")
    result = inspect_study(args.study_dir, sample_seconds=args.sample_seconds, verify_artifacts=args.verify_artifacts,
                           verify_data_scope=args.verify_data_scope)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

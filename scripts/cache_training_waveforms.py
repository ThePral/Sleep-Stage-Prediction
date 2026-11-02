"""Cache source-training waveforms in existing Linux shared memory.

All records remain enrolled. Metadata and validation files stay on the source
drive; copied waveform bytes must match the prepared-store checksums. No model
is imported, no original permissions change, and no disk image is created.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare_cache(source_root, split_file, cache_root, *, free_bytes=None, available_memory=None):
    source, cache = Path(source_root).resolve(), Path(cache_root).resolve()
    if cache.exists():
        raise FileExistsError("Preserve existing cache directories; do not overwrite an active cache.")
    with Path(split_file).open(newline="") as stream:
        assignments = {r["participant_id"]: r["split"] for r in csv.DictReader(stream)}
    expected = {s: {p for p, role in assignments.items() if role == s} for s in ["train", "val"]}
    records = []
    for directory in sorted((source / "sleepedf").iterdir()):
        if not directory.is_dir():
            continue
        role = assignments.get(directory.name[:5])
        if role == "test":
            continue
        if role not in expected or directory.is_symlink():
            raise ValueError("Unexpected source-record identity or directory link.")
        with (directory / "epochs.csv").open(newline="") as stream:
            row = next(csv.DictReader(stream))
        if (row["dataset"] != "sleepedf" or row["recording_id"] != directory.name
                or row["participant_id"] != directory.name[:5] or row["split"] != role):
            raise ValueError("Source identity disagrees with the frozen participant split.")
        manifest = json.loads((directory / "manifest.json").read_text())
        if set(manifest["files"]) != {"epochs.csv", "waveforms.npy"}:
            raise ValueError("Unexpected prepared-store inventory.")
        if digest(directory / "epochs.csv") != manifest["files"]["epochs.csv"]:
            raise ValueError("Source epoch metadata checksum mismatch.")
        records.append({"recording_id": directory.name, "participant_id": row["participant_id"], "split": role,
                        "waveform_bytes": (directory / "waveforms.npy").stat().st_size,
                        "waveform_sha256": manifest["files"]["waveforms.npy"],
                        "manifest_sha256": digest(directory / "manifest.json")})
    for role, participants in expected.items():
        if not participants or participants != {r["participant_id"] for r in records if r["split"] == role}:
            raise ValueError("A RAM cache cannot shrink the source training/validation cohort.")
    required = sum(r["waveform_bytes"] for r in records if r["split"] == "train")
    if free_bytes is None:
        free_bytes = shutil.disk_usage(cache.parent).free
    if available_memory is None:
        available_memory = next(int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines()
                                if line.startswith("MemAvailable:"))
    result = {"source_root": str(source), "data_root": str(cache), "training_waveform_bytes": required,
              "shared_memory_free_before": free_bytes, "memory_available_before": available_memory,
              "split_sha256": digest(split_file), "records": records,
              "policy": "all_source_train_validation_records_same_bytes_no_test_or_external_payload"}
    if required + 128 * 1024**2 > free_bytes or required + 2 * 1024**3 > available_memory:
        return {**result, "status": "skipped_insufficient_headroom", "data_root": str(source)}
    cache.mkdir(mode=0o700)
    (cache / "sleepedf").mkdir()
    for number, record in enumerate(records):
        original = source / "sleepedf" / record["recording_id"]
        target = cache / "sleepedf" / record["recording_id"]
        if record["split"] == "val":
            target.symlink_to(original, target_is_directory=True)
            continue
        target.mkdir()
        for name in ["epochs.csv", "manifest.json"]:
            (target / name).symlink_to(original / name)
        checksum = hashlib.sha256()
        with (original / "waveforms.npy").open("rb") as reader, (target / "waveforms.npy").open("xb") as writer:
            for block in iter(lambda: reader.read(1024 * 1024), b""):
                writer.write(block)
                checksum.update(block)
        if checksum.hexdigest() != record["waveform_sha256"]:
            raise ValueError("Cached waveform checksum mismatch; no complete cache manifest published.")
        if number % 20 == 0:
            print(json.dumps({"copied_training_record": record["recording_id"]}), flush=True)
    result["status"] = "completed"
    with (cache / "cache_manifest.json").open("x") as stream:
        json.dump(result, stream, indent=2)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    if sys.platform != "linux" or args.cache_root.resolve().parent != Path("/dev/shm").resolve():
        raise ValueError("The local cache must be a new directory directly inside existing /dev/shm.")
    if not args.cache_root.name.startswith("sleep-stage-prediction-") or args.report.exists():
        raise ValueError("Use a project-specific cache name and a new report path.")
    result = prepare_cache(args.source_root, args.split_file, args.cache_root)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({k: result[k] for k in ["status", "data_root", "training_waveform_bytes"]}), flush=True)


if __name__ == "__main__":
    main()

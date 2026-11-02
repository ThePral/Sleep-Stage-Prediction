"""Create and verify an offline source-training capsule; no upload is performed.

Only explicit code/configuration paths, the frozen split, and prepared Sleep-EDF
train/validation stores enter the archive. Test and external epoch data do not.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile

from src.data.epoch_store import EpochStore, sha256_file
from src.data.label_budgets import discover_scope, fingerprint
from src.data.prepare_research_data import ROOT, participant_assignments


MANIFEST = "training_capsule.json"
# Legacy names remain valid when verifying archives created before organization.
LEGACY_DOCUMENTS = {"IMPLEMENTATION_STATUS.md", "COMPUTE_GUIDE.md", "TRAINING_RUNBOOK.md",
                    "requirements-research-common.txt", "requirements-linux-gpu.txt",
                    "constraints-cuda125-pascal.txt"}
DOCUMENTS = LEGACY_DOCUMENTS | {
    "README.md", "AGENTS.md", "docs/README.md", "docs/PROJECT_STRUCTURE.md", "docs/RESEARCH_DIRECTION.md",
    "docs/SLEEPEDF_BILSTM.md",
    "docs/IMPLEMENTATION_STATUS.md", "docs/operations/COMPUTE_GUIDE.md", "docs/operations/TRAINING_RUNBOOK.md",
    "docs/reference/IMPLEMENTED_PIPELINE.md", "docs/archive/2026-10/PROJECT_AUDIT_2026-10-26.md",
    "docs/archive/2026-10/RESEARCH_ROADMAP_2026-10-26.md", "environments/README.md",
    "environments/requirements-research-common.txt", "environments/requirements-linux-gpu.txt",
    "environments/constraints-cuda125-pascal.txt", "environments/requirements-windows-py313.lock.txt",
    "configs/README.md", "scripts/README.md", "scripts/research.ps1", "scripts/research_cpu.ps1",
    "scripts/training_status.py", "scripts/configure_wsl_cuda.py",
}


def safe_member(name):
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or "\\" in name or ":" in name
            or any(part in {"", ".", ".."} for part in name.split("/"))):
        raise ValueError("Unsafe training capsule member path.")
    return path


def validate_manifest(manifest):
    if (manifest.get("schema_version") != 1 or manifest.get("kind") != "source_training_capsule"
            or fingerprint({k: v for k, v in manifest.items() if k != "fingerprint"}) != manifest.get("fingerprint")):
        raise ValueError("Training capsule manifest integrity check failed.")
    records = manifest["source_scope"]["recordings"]
    if not records or len({r["recording_id"] for r in records}) != len(records) or any(r["split"] not in {"train", "val"} for r in records):
        raise ValueError("A training capsule cannot declare held-out epoch records.")
    scope = manifest["source_scope"]
    if (set(scope["training_participants"]) != {r["participant_id"] for r in records if r["split"] == "train"}
            or set(scope["validation_participants"]) != {r["participant_id"] for r in records if r["split"] == "val"}
            or set(scope["training_participants"]) & set(scope["validation_participants"])):
        raise ValueError("Training capsule participant scope disagrees with its records.")
    data_files = {f"data/processed/research_v1/sleepedf/{r['recording_id']}/{name}" for r in records
                  for name in ["manifest.json", "epochs.csv", "waveforms.npy"]}
    for name, item in manifest["files"].items():
        path = safe_member(name)
        code = (path.parts[0] in {"src", "tests"} and path.suffix == ".py" or path.parts[0] == "configs" and path.suffix == ".json") and not any(p.startswith(".") for p in path.parts)
        if not (code or name in DOCUMENTS or name == "final_model_corrected/participant_split.csv" or name in data_files):
            raise ValueError("Training capsule contains a payload outside the source-only allowlist.")
        if not isinstance(item["bytes"], int) or item["bytes"] < 0:
            raise ValueError("Invalid capsule payload size.")
    if (not data_files <= set(manifest["files"]) or "final_model_corrected/participant_split.csv" not in manifest["files"]
            or manifest["total_payload_bytes"] != sum(i["bytes"] for i in manifest["files"].values())):
        raise ValueError("Training capsule inventory is incomplete.")
    return manifest


def _payload(project_root, data_root, split_file):
    project_root, data_root, split_file = Path(project_root).resolve(), Path(data_root).resolve(), Path(split_file).resolve()
    files = {}
    for folder, suffix in [("src", ".py"), ("tests", ".py"), ("configs", ".json")]:
        for path in sorted((project_root / folder).rglob("*" + suffix)):
            relative = path.relative_to(project_root)
            if any(part.startswith(".") for part in relative.parts):
                continue
            if path.is_symlink() or not path.resolve().is_relative_to(project_root):
                raise ValueError("Code/configuration symlinks cannot enter a training capsule.")
            files[relative.as_posix()] = path
    for name in sorted(DOCUMENTS):
        path = project_root / name
        if path.is_file():
            if path.is_symlink() or not path.resolve().is_relative_to(project_root):
                raise ValueError("Capsule documentation/dependencies cannot be symlinks.")
            files[name] = path
    assignments = participant_assignments(split_file)
    training = sorted(p for p, split in assignments.items() if split == "train")
    validation = sorted(p for p, split in assignments.items() if split == "val")
    if not training or not validation:
        raise ValueError("A training capsule requires both source training and validation participants.")
    records = discover_scope(data_root, assignments, training, validation)
    config = None
    for record in records:
        directory = data_root / "sleepedf" / record["recording_id"]
        with EpochStore(directory) as store:
            if config is None:
                config = store.config
            elif store.config != config:
                raise ValueError("Training capsule cannot mix preprocessing versions.")
            if store.epochs.split.iloc[0] not in {"train", "val"}:
                raise ValueError("Held-out epoch data cannot enter the training capsule.")
        for filename in ["manifest.json", "epochs.csv", "waveforms.npy"]:
            path = directory / filename
            if path.is_symlink() or not path.resolve().is_relative_to(data_root):
                raise ValueError("Prepared source stores must be regular files inside the data root.")
            files[f"data/processed/research_v1/sleepedf/{record['recording_id']}/{filename}"] = path
    files["final_model_corrected/participant_split.csv"] = split_file
    return files, {"training_participants": training, "validation_participants": validation, "recordings": records,
                   "split_sha256": sha256_file(split_file), "preprocessing_fingerprint": config.fingerprint,
                   "data_policy": "prepared_sleepedf_train_and_validation_only_no_test_or_isruc_epoch_payloads"}


class DigestReader:
    def __init__(self, stream):
        self.stream, self.digest = stream, hashlib.sha256()

    def read(self, size=-1):
        value = self.stream.read(size)
        self.digest.update(value)
        return value


def create_capsule(project_root, data_root, split_file, archive_path):
    archive = Path(archive_path).resolve()
    staging = archive.with_suffix(archive.suffix + ".partial")
    if archive.exists() or staging.exists():
        raise FileExistsError("Capsules and interrupted staging files are never overwritten.")
    files, scope = _payload(project_root, data_root, split_file)
    inventory = {name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)} for name, path in sorted(files.items())}
    manifest = {"schema_version": 1, "kind": "source_training_capsule", "source_scope": scope, "files": inventory,
                "total_payload_bytes": sum(item["bytes"] for item in inventory.values()), "compression": "uncompressed_tar"}
    manifest["fingerprint"] = fingerprint(manifest)
    validate_manifest(manifest)
    archive.parent.mkdir(parents=True, exist_ok=True)
    with staging.open("xb") as stream, tarfile.open(fileobj=stream, mode="w") as tar:
        encoded = (json.dumps(manifest, indent=2) + "\n").encode()
        header = tarfile.TarInfo(MANIFEST)
        header.size = len(encoded)
        header.mode = 0o644
        tar.addfile(header, io.BytesIO(encoded))
        for name, path in sorted(files.items()):
            safe_member(name)
            header = tarfile.TarInfo(name)
            header.size = inventory[name]["bytes"]
            header.mode = 0o644
            with path.open("rb") as source:
                reader = DigestReader(source)
                tar.addfile(header, reader)
                if reader.digest.hexdigest() != inventory[name]["sha256"] or source.read(1):
                    raise ValueError("A capsule input changed while it was being copied.")
    if archive.exists():
        raise FileExistsError("Capsule destination appeared while publishing.")
    staging.rename(archive)
    result = {"archive": str(archive), "archive_sha256": sha256_file(archive), "archive_bytes": archive.stat().st_size,
              "manifest_fingerprint": manifest["fingerprint"], "files": len(inventory), "source_scope": scope}
    return result


def unpack_capsule(archive_path, destination, expected_sha256):
    archive = Path(archive_path)
    if not expected_sha256 or len(expected_sha256) != 64 or sha256_file(archive) != expected_sha256:
        raise ValueError("Training capsule archive SHA-256 verification failed.")
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError("Extract into a new directory; no existing checkout is replaced.")
    with tarfile.open(archive, "r:") as tar:
        members = tar.getmembers()
        if not members or members[0].name != MANIFEST or members[0].size > 10 * 1024 * 1024 or not members[0].isreg():
            raise ValueError("Training capsule has an invalid manifest header.")
        manifest_bytes = tar.extractfile(members[0]).read()
        manifest = json.loads(manifest_bytes)
        validate_manifest(manifest)
        names = [m.name for m in members]
        if len(set(names)) != len(names) or set(names) != {MANIFEST, *manifest["files"]}:
            raise ValueError("Training capsule members disagree with its explicit inventory.")
        for member in members:
            safe_member(member.name)
            if not member.isreg() or member.issym() or member.islnk():
                raise ValueError("Training capsules cannot contain links or special filesystem objects.")
            if member.name != MANIFEST and member.size != manifest["files"][member.name]["bytes"]:
                raise ValueError("Training capsule member size disagrees with its manifest.")
        destination.mkdir(parents=True, exist_ok=False)
        for member in members:
            if member.name == MANIFEST:
                continue
            target = destination / member.name
            if not target.resolve().is_relative_to(destination):
                raise ValueError("Training capsule path escaped its new destination.")
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            with tar.extractfile(member) as source, target.open("xb") as stream:
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    stream.write(block)
                    digest.update(block)
            if member.name != MANIFEST and digest.hexdigest() != manifest["files"][member.name]["sha256"]:
                raise ValueError("Extracted training payload SHA-256 mismatch.")
        # Publish the manifest last, after every payload verifies successfully.
        with (destination / MANIFEST).open("xb") as stream:
            stream.write(manifest_bytes)
    return {"destination": str(destination), "manifest_fingerprint": manifest["fingerprint"], "files_verified": len(manifest["files"])}


def verify_extracted(destination):
    destination = Path(destination).resolve()
    manifest = validate_manifest(json.loads((destination / MANIFEST).read_text(encoding="utf-8")))
    for name, expected in manifest["files"].items():
        path = destination / name
        if path.is_symlink() or not path.resolve().is_relative_to(destination) or path.stat().st_size != expected["bytes"] or sha256_file(path) != expected["sha256"]:
            raise ValueError("Extracted training capsule content changed or is incomplete.")
    assignments = participant_assignments(destination / "final_model_corrected/participant_split.csv")
    import pandas as pd
    for record in manifest["source_scope"]["recordings"]:
        path = destination / "data/processed/research_v1/sleepedf" / record["recording_id"]
        row = pd.read_csv(path / "epochs.csv", nrows=1, usecols=["dataset", "recording_id", "participant_id", "split"], dtype=str).iloc[0]
        if (row.dataset != "sleepedf" or row.recording_id != record["recording_id"] or row.participant_id != record["participant_id"]
                or row.split != record["split"] or assignments.get(row.participant_id) != row.split):
            raise ValueError("Extracted source identities disagree with the frozen participant split.")
    return {"manifest_fingerprint": manifest["fingerprint"], "files_verified": len(manifest["files"]),
            "training_participants": len(manifest["source_scope"]["training_participants"]),
            "validation_participants": len(manifest["source_scope"]["validation_participants"])}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("create")
    build.add_argument("--project-root", type=Path, default=ROOT)
    build.add_argument("--data-root", type=Path, default=ROOT / "data/processed/research_v1")
    build.add_argument("--split-file", type=Path, default=ROOT / "final_model_corrected/participant_split.csv")
    build.add_argument("--archive", type=Path, required=True)
    unpack = sub.add_parser("unpack")
    unpack.add_argument("--archive", type=Path, required=True)
    unpack.add_argument("--destination", type=Path, required=True)
    unpack.add_argument("--expected-sha256", required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "create":
        result = create_capsule(args.project_root, args.data_root, args.split_file, args.archive)
    elif args.command == "unpack":
        result = unpack_capsule(args.archive, args.destination, args.expected_sha256)
    else:
        result = verify_extracted(args.destination)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

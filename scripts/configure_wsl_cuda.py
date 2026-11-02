"""Expose pip-installed CUDA libraries inside this virtual environment only.

Implements TensorFlow's documented virtual-environment shared-library links:
https://www.tensorflow.org/install/pip#6_gpu_only_virtual_environment_configuration
Existing files are never replaced. No TensorFlow import or optimizer work occurs.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import sysconfig


def configure(venv_root, site_packages):
    root, packages = Path(venv_root).resolve(), Path(site_packages).resolve()
    if not (root / "pyvenv.cfg").is_file() or not packages.is_relative_to(root):
        raise ValueError("CUDA links require an isolated virtual environment.")
    tensorflow = packages / "tensorflow"
    if not tensorflow.is_dir() or not tensorflow.resolve().is_relative_to(root):
        raise ValueError("Install TensorFlow before configuring its CUDA library links.")
    sources = sorted((packages / "nvidia").glob("*/lib/*.so*"))
    if not sources:
        raise ValueError("No pip-installed NVIDIA shared libraries were found.")
    planned = {}
    for source in sources:
        if not source.is_file() or not source.resolve().is_relative_to(root):
            raise ValueError("A CUDA library escaped the inspected virtual environment.")
        alias = tensorflow / source.name
        if alias in planned and planned[alias].resolve() != source.resolve():
            raise ValueError("Different CUDA libraries have the same link name.")
        planned[alias] = source
    ptxas = packages / "nvidia/cuda_nvcc/bin/ptxas"
    if ptxas.is_file():
        if not ptxas.resolve().is_relative_to(root):
            raise ValueError("The CUDA compiler escaped the inspected virtual environment.")
        planned[root / "bin/ptxas"] = ptxas
    # Check every conflict before creating anything; never replace package files.
    for alias, source in planned.items():
        if not alias.parent.resolve().is_relative_to(root):
            raise ValueError("A CUDA link destination escaped the inspected virtual environment.")
        if os.path.lexists(alias) and (not alias.is_symlink() or alias.resolve() != source.resolve()):
            raise FileExistsError(f"Preserve conflicting environment file: {alias}")
    created = []
    for alias, source in planned.items():
        if not os.path.lexists(alias):
            alias.symlink_to(os.path.relpath(source, alias.parent))
            created.append(str(alias.relative_to(root)))
    return {"environment": str(root), "created": created, "verified_links": len(planned),
            "links": {str(alias.relative_to(root)): str(source.relative_to(root)) for alias, source in planned.items()},
            "scope": "project_virtual_environment_only_no_driver_or_system_changes"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.report.exists():
        raise FileExistsError("Preserve the previous CUDA environment report.")
    if sys.platform != "linux" or sys.prefix == sys.base_prefix:
        raise RuntimeError("Run this command with the isolated Linux/WSL Python.")
    result = configure(sys.prefix, sysconfig.get_path("purelib"))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"created": len(result["created"]), "verified_links": result["verified_links"]}))


if __name__ == "__main__":
    main()

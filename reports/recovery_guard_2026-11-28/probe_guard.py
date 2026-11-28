"""Read-only guard probe. Trailing -m arguments are inspected, never executed."""
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

import psutil

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.epoch_store import sha256_file
from src.experiments.run import archive_stale_lock


class ArchiveReached(Exception):
    pass


def main():
    root = ROOT / 'runs/bilstm_replication_cpu_2026-11-20'
    lock = root / '.runner.lock'
    before = sha256_file(lock)
    current = psutil.Process(os.getpid())
    excluded = {current.pid}
    command = current.cmdline()[1:]
    for parent in current.parents():
        try:
            if parent.cmdline()[1:] != command:
                break
            excluded.add(parent.pid)
        except psutil.Error:
            break
    blockers = []
    for process in psutil.process_iter(['pid', 'name', 'cmdline']):
        args = process.info['cmdline'] or []
        if process.pid not in excluded and '-m' in args and args[args.index('-m') + 1] in ('src.experiments.run', 'src.experiments.worker'):
            blockers.append({**process.info, 'is_ancestor': process.pid in {p.pid for p in current.parents()}})
    # Stop at the mutation boundary even if every production guard check passes.
    with patch.object(Path, 'rename', side_effect=ArchiveReached('rename prevented by read-only probe')):
        try:
            archive_stale_lock(root)
        except ArchiveReached:
            outcome = 'all_checks_passed_archive_prevented'
        except ValueError as exc:
            outcome = str(exc)
        else:
            raise AssertionError('Expected an existing lock and a prevented rename.')
    assert sha256_file(lock) == before
    result = {'outcome': outcome, 'blockers': blockers, 'lock_unchanged': True,
              'lock_sha256': before, 'optimizer_updates_performed': False,
              'tensorflow_imported': 'tensorflow' in sys.modules}
    assert not result['tensorflow_imported']
    name = 'file_wrapper.json' if blockers else 'command_wrapper.json'
    (Path(__file__).resolve().parent / name).write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'outcome': outcome, 'blocking_processes': [{'name': p['name'], 'is_ancestor': p['is_ancestor']} for p in blockers],
                      'lock_unchanged': True, 'training_started': False}))


if __name__ == '__main__':
    main()

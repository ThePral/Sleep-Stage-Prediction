"""Sealed epoch generations. Publishing a new pointer never overwrites old state.

Numeric arrays use NPZ without pickle; compatibility is a content fingerprint.
No TensorFlow import or optimizer operation is performed in this module.
"""
import json
import os
from pathlib import Path
import shutil
import uuid

import numpy as np

from src.data.epoch_store import sha256_file
from src.data.label_budgets import fingerprint


def atomic_json(path, value):
    path = Path(path)
    staging = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with staging.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(staging, path)


def load_recovery(source, contract=None):
    """Accept an artifact root, recovery root, or explicitly sealed generation."""
    source = Path(source).resolve()
    if (source / 'recovery').is_dir():
        source = source / 'recovery'
    if (source / 'latest.json').exists():
        pointer = json.loads((source / 'latest.json').read_text(encoding='utf-8'))
        name = pointer.get('generation', '')
        if (pointer.get('schema_version') != 1 or not isinstance(name, str)
                or not name.startswith('epoch_') or Path(name).name != name or '/' in name or '\\' in name):
            raise ValueError('Invalid supervised recovery pointer.')
        generation = source / name
        if generation.is_symlink() or sha256_file(generation / 'seal.json') != pointer.get('seal_sha256'):
            raise ValueError('Supervised recovery pointer integrity failure.')
    else:
        generation = source
    seal = json.loads((generation / 'seal.json').read_text(encoding='utf-8'))
    if (seal.get('schema_version') != 1 or seal.get('kind') != 'supervised_epoch_recovery'
            or set(seal.get('files', {})) not in ({'state.json', 'arrays.npz'}, {'state.json', 'arrays.npz', 'best.keras'})):
        raise ValueError('Unsupported supervised recovery seal.')
    for name, digest in seal['files'].items():
        path = generation / name
        if path.is_symlink() or sha256_file(path) != digest:
            raise ValueError(f'Supervised recovery artifact changed: {name}')
    state = json.loads((generation / 'state.json').read_text(encoding='utf-8'))
    epoch = state.get('completed_epochs')
    if (isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0
            or len(state.get('history', [])) != epoch
            or state.get('contract_sha256') != fingerprint(state.get('contract'))):
        raise ValueError('Invalid supervised recovery state.')
    if [row.get('epoch') for row in state['history']] != list(range(epoch)):
        raise ValueError('Recovery history epoch order changed.')
    if contract is not None and state['contract_sha256'] != fingerprint(contract):
        raise ValueError('Supervised resume contract changed (code/packages/data/configuration/runtime).')
    with np.load(generation / 'arrays.npz', allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    if not arrays or any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError('Nonfinite or empty supervised recovery arrays.')
    return generation, state, arrays


def publish_recovery(root, contract, completed_epochs, state, arrays, best_path=None):
    """Commit all files, then atomically replace latest.json. Keep prior generations."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if isinstance(completed_epochs, bool) or not isinstance(completed_epochs, int) or completed_epochs < 0:
        raise ValueError('completed_epochs must be a nonnegative integer.')
    if len(state.get('history', [])) != completed_epochs:
        raise ValueError('Recovery history does not match epoch boundary.')
    if not arrays or any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError('Recovery arrays must be finite and nonempty.')
    identifier = uuid.uuid4().hex
    staging = root / ('.pending_' + identifier)
    staging.mkdir()  # Inherit normal project permissions, unlike Windows mkdtemp(0700).
    payload = {**state, 'completed_epochs': completed_epochs, 'contract': contract,
               'contract_sha256': fingerprint(contract)}
    atomic_json(staging / 'state.json', payload)
    with (staging / 'arrays.npz').open('xb') as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    names = ['state.json', 'arrays.npz']
    if best_path is not None:
        with Path(best_path).open('rb') as source, (staging / 'best.keras').open('xb') as destination:
            shutil.copyfileobj(source, destination)
            destination.flush()
            os.fsync(destination.fileno())
        names.append('best.keras')
    atomic_json(staging / 'seal.json', {'schema_version': 1, 'kind': 'supervised_epoch_recovery',
                                      'files': {name: sha256_file(staging / name) for name in names}})
    destination = root / f'epoch_{completed_epochs:04d}_{identifier}'
    staging.rename(destination)
    atomic_json(root / 'latest.json', {'schema_version': 1, 'generation': destination.name,
                                     'seal_sha256': sha256_file(destination / 'seal.json')})
    return destination

"""Copy the reviewed draft to a CPU recipe; no study execution or model import."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    source = ROOT / 'configs/sleepedf/experiments/experiment_bilstm_replication_draft.json'
    target = ROOT / 'configs/sleepedf/experiments/experiment_bilstm_replication_cpu.json'
    config = json.loads(source.read_text(encoding='utf-8'))
    assert config['training_seeds'] == [17, 42, 123]
    assert config['training']['recovery'] and not config['training']['compiled_validation']
    assert config['training']['batch_size'] == config['training']['validation_batch_size'] == 4
    assert all(value is None for value in config['limits'].values())
    with target.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(config, indent=2) + '\n')
    print(json.dumps({'CPU_recipe_created': True, 'training_started': False}))


if __name__ == '__main__':
    main()

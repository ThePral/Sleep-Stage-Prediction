"""Read retained diagnostic outputs only; no model loading or optimizer updates."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = Path(__file__).resolve().parent
DIAGNOSTIC = ROOT / 'reports/validation_parity_2026-11-15_m3_diagnostic'
SMOKE = ROOT / 'reports/validation_parity_2026-11-15_smoke'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    result = json.loads((DIAGNOSTIC / 'result.json').read_text(encoding='utf-8'))
    assert result['diagnostic_only'] and not result['passed'] and not result['full_cohort_checked']
    assert not result['optimizer_updates_performed'] and not result['test_or_ISRUC_scoring']
    assert len(result['models']) == 1
    model = result['models'][0]
    assert model['job_id'] == 'M3_draw42_n70_seed42' and model['examples'] == 64
    assert model['saved_vs_eager']['passed'] and not model['saved_vs_compiled']['passed']
    assert digest(DIAGNOSTIC / 'selected_targets.csv') == digest(SMOKE / 'selected_targets.csv')
    policy = result['policy']
    assert policy['atol'] == policy['rtol'] == 1e-6
    changed = [name for name, expected in policy['current_source_sha256'].items()
               if digest(ROOT / name) != expected]
    assert not changed, changed
    rows = []
    diagnostics = model['batch_diagnostics']
    assert [item['batch_size'] for item in diagnostics] == [4, 8, 16, 32]
    for item in diagnostics:
        assert item['compiled_traces'] == 1
        assert item['saved_vs_eager']['passed'] == (item['batch_size'] in (4, 8))
        assert not item['saved_vs_compiled']['passed']
        for name in ('saved_vs_eager', 'saved_vs_compiled', 'eager_vs_compiled'):
            check = item[name]
            assert check['argmax_mismatches'] == 0 and check['confusion_matrices_equal']
            assert check['metric_differences']['accuracy'] == check['metric_differences']['macro_f1'] == 0
        rows.append({'batch_size': item['batch_size'],
                     'eager_vs_saved': item['saved_vs_eager'],
                     'compiled_vs_saved': item['saved_vs_compiled'],
                     'compiled_traces': item['compiled_traces']})
    evidence = {'review_date': '2026-11-15', 'diagnostic_gate_passed': False,
                'input_sha256': {name: digest(DIAGNOSTIC / name) for name in ('result.json', 'policy.json', 'selected_targets.csv')},
                'identical_sample_to_smoke': True, 'current_source_matches_diagnostic': True,
                'sample_count': 64, 'model_id': model['job_id'], 'comparisons': rows,
                'full_cohort_checked': False, 'optimizer_updates_by_reviewer': False,
                'decision': 'Keep compiled validation unaccepted; retain conservative eager batch 4 pending full-cohort review. Do not relax tolerances.',
                'next_check': 'User-run full-cohort eager-only M3 comparison; no claim about compiled parity or training speedup.'}
    (OUTPUT / 'verification.json').write_text(json.dumps(evidence, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'review_complete': True, 'diagnostic_gate_passed': False, 'identical_sample': True}))


if __name__ == '__main__':
    main()

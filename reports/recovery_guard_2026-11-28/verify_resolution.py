"""Recheck outage state and retain guard probe evidence without mutation."""
import importlib.util
import json
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent
ROOT = OUTPUT.parents[1]


def main():
    source = ROOT / 'reports/replication_power_loss_2026-11-28/inspect_recovery.py'
    spec = importlib.util.spec_from_file_location('outage_readonly_audit', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.OUTPUT = OUTPUT  # Retain a new observation; preserve the original report.
    module.main()
    verification = json.loads((OUTPUT / 'verification.json').read_text())
    first = json.loads((OUTPUT / 'file_wrapper.json').read_text())
    second = json.loads((OUTPUT / 'command_wrapper.json').read_text())
    assert first['outcome'] == 'An experiment process is alive; refuse stale-lock recovery.'
    assert len(first['blockers']) == 1
    assert first['blockers'][0]['name'].lower() == 'powershell.exe' and first['blockers'][0]['is_ancestor']
    assert second['outcome'] == 'all_checks_passed_archive_prevented' and not second['blockers']
    assert first['lock_sha256'] == second['lock_sha256'] == verification['lock_sha256']
    assert first['lock_unchanged'] and second['lock_unchanged']
    verification['guard_resolution'] = {
        'false_positive_identified': 'outer PowerShell -File launcher contains standalone -m src.experiments.run arguments',
        'command_wrapper_passed_all_guard_checks_in_readonly_probe': True,
        'archive_operation_prevented_by_probe': True, 'frozen_source_modified': False,
        'CPU_launcher_script_modified': False, 'training_launched': False,
        'next_invocation': 'same CPU launcher via powershell -Command rather than -File'}
    (OUTPUT / 'verification.json').write_text(json.dumps(verification, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'guard_false_positive_confirmed': True, 'command_wrapper_verified_without_training': True,
                      'completed_jobs': verification['completed_jobs_verified'],
                      'sealed_epoch': verification['latest_sealed_epoch'], 'lock_unchanged': True}))


if __name__ == '__main__':
    main()

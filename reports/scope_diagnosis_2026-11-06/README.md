# Dataset-scope startup diagnosis — 6 November 2026

The user reported `Missing source train participants` when executing the
prepared BiLSTM terminal study with `--live --max-jobs 4`, without a data-root
override. That command is valid. The original cause could not be reproduced;
no claim is made that a dataset or model defect was repaired.

The same native Windows environment now passes `verify_runtime`, including
frozen source/package versions and `verify_plan_scope`. Discovery sees 197
Sleep-EDF recording manifests. The planned source training/validation identities
and manifest hashes match. The ledger has no attempts, all 18 jobs remain
pending, and no scheduler or worker is running.

`status.json` records the monitor's new optional `--verify-data-scope` check.
`checks.json` records a passing real-scope check and a deliberately nonexistent
data-root check, which correctly reports failure. The fault injection uses a
copied in-memory study dictionary and changes no frozen artifact. These checks
do not import TensorFlow, load waveform payloads or perform optimizer updates.

Only the monitoring script and documentation changed; `src/`, packages, the
study, datasets and historical checkpoints remain unchanged. No training was
launched. The user can retry the original command; `--retry-failed` is not
needed because the startup error occurred before any job attempt.

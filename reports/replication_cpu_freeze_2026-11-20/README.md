# CPU replication freeze review

The user chose the current CPU route. The source-only replication is frozen at
`runs/bilstm_replication_cpu_2026-11-20` with fingerprint
`1e994b1116d15494f9137c56f7133bcc1afe807867381e7717e268a169351572`.
There are 48 planned jobs and no optimizer work was launched by the assistant.

| Item | Frozen value |
|---|---|
| Optimization seeds | 17, 42, 123 |
| Label draw and budgets | Draw 42; 18 and 70 participants |
| Source participants | 70 train, 15 validation |
| Model family | M0 through M6, BiLSTM |
| Supervised schedule | Up to 30 epochs; batch 4; recovery enabled |
| Validation | Full cohort, eager batch 4; compiled disabled |
| SSL | Three fresh ten-epoch fits; batch 32 |
| Jobs | 42 supervised, 3 SSL, 2 normalizers, 1 waveform-statistics fit |
| Data/batch caps | None |

The budget fingerprint is identical to the historical study's: source scope,
participant label selections and eligible records did not shrink or change.
Current source and all installed package versions also match the verified
runtime environment. The protocol, recipe, launcher and study/budget hashes
are archived in the study's `protocol_freeze.json`, which has its own content
fingerprint; `protocol.md` holds the archived human-readable protocol.

`inspect_frozen_plan.py --publish` archived the protocol before execution.
The same script without `--publish` reproduces the pre-execution checks without
TensorFlow imports. `verification.json` records the result and D: free space.
The planner imported configuration classes only; it did not construct models
or run optimizer operations. Core source and packages were not edited.

The new seed-42 fits use the same procedure as 17/123. Historical seed-42
results remain separate exploratory evidence. This freeze covers source
training and source-development analysis; source-test scoring requires its
separate final evaluation policy, and ISRUC scoring stays unopened.

The [terminal handoff](../../docs/operations/REPLICATION_CPU.md) supplies a
single `execute --live --max-jobs 4` command for the first preparation/SSL-17
stage. It must be launched by the user. Its expected completion count is four
of forty-eight, with `limited_run: false`; no supervised jobs start in that
stage. Further stages wait for inspection. SSL scheduler resume is not added
by the supervised recovery flag.

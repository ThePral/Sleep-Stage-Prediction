# Supervised recovery and validation execution — 13 November 2026

This implementation is opt-in and awaits the user-run optimizer equivalence
gate. Atomic-state/inference tests pass; exact training continuation and an
end-to-end speed improvement have not yet been demonstrated. No research
training or optimizer-based gate was launched by the assistant.

## Run the small training gate first

From the project root, run in your terminal:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/research_cpu.ps1 -m unittest tests.test_supervised_recovery_training -v
```

This performs short synthetic gated-BiLSTM fits with dropout in three fresh
processes: uninterrupted, interrupted after epoch three, and resumed. Controlled
test-only validation inputs exercise early stopping and learning-rate reduction;
they are not research performance scores. Expected training is 24 optimizer
updates total. No real Sleep-EDF or ISRUC training data are used. Logs and fixture
artifacts are retained at the printed `runs/recovery_gate/<id>/` directory.

The gate checks integer state exactly and float state within `1e-7` absolute
and relative tolerance, including optimizer slots, dropout seed variables,
global RNG states, callback counters, complete history, best epoch and final
prediction parity. It also loads the completed inference bundles. Report its
result before launching a new research campaign. A pass applies to the tested
environment/architecture, not arbitrary hardware or all possible model changes.

After that gate passes, run the full regression suite yourself:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/research_cpu.ps1 -m unittest discover -s tests -v
```

The full suite includes optimizer updates and remains user-owned. No prior
full-suite result is claimed for the new implementation. The assistant-run
non-optimizer checks are recorded in
`reports/recovery_implementation_2026-11-13/`.

## Configuration and state

`TrainingConfig` adds fields with conservative defaults:

```json
{"recovery": false, "compiled_validation": false, "validation_batch_size": null}
```

Enable `recovery: true` only in a fresh configuration/study after its gates pass.
It uses explicit `seed + epoch` batch ordering and one-epoch `fit` calls while
preserving callback lifetime across the whole schedule. This is a new execution
path and is not declared bitwise equivalent to the historical implicit-generator
training path. The CPU launcher provides a fixed `PYTHONHASHSEED` (42 unless the
caller supplied another value) and restores the caller's environment afterward.
Direct Python recovery calls require a fixed hash seed set before Python starts.

Each run has two distinct outputs:

- `model.keras`: the best source-validation macro-F1 model, atomically replaced.
- `recovery/latest.json`: an atomic pointer to the latest complete epoch state.
  Each sealed generation contains every model variable (including dropout seed
  state), every optimizer variable, Python/NumPy/Keras/TensorFlow RNG state,
  early-stop/LR callback state, history and a copy of the selected best model.

The first generation captures epoch zero, before any optimizer updates. Later
generations publish after validation, early stopping and LR reduction finish.
Old generations remain available; incomplete pending generations are not chosen
by `latest.json`. A failed publication keeps the previous pointer. CRC/whole-file
hash guards detect damaged files; atomic rename and fsync reduce write risk but
cannot guarantee storage survives arbitrary hardware/power failure.

The recovery contract covers model/training settings, target/data manifests,
normalization, initialization, full source hashes, installed package versions,
Python/platform/device identity, precision/JIT and relevant execution environment.
Changed contracts are rejected; this is not silent cloud/platform migration.
An epoch whose validation/save did not finish is replayed from the preceding
published boundary. Epoch-boundary recovery does not preserve an unfinished batch.

## Restart a future recovery-enabled study

For a study that was planned under the current unchanged code and enabled
recovery, the restart command is:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/research_cpu.ps1 -m src.experiments.run execute --study-dir runs/YOUR_NEW_STUDY --live --retry-failed --resume-supervised --recover-stale-lock
```

The flags are explicit. Completed jobs are verified/skipped; old attempts are
retained. For a supervised job the scheduler finds the greatest published epoch
across its attempts, and the worker verifies that it belongs to a previous
attempt of that same study/job. Without a published state, it announces a restart
from the declared initialization. Corrupt recovery data are rejected rather than
silently ignored. SSL retains its separate existing resume interface; this flag
does not add scheduler-level SSL resume.

Stale-lock recovery archives the dead PID lock; it refuses a live PID or another
experiment process. The current CLI and verified identical-command Windows venv
launcher parents are excluded from its process check. It does not kill processes.
Standalone supervised training accepts `--resume-from <previous_artifact_root>`
with recovery enabled and identical arguments/configuration; it creates a fresh
output directory and checks the same contract.

The completed `runs/bilstm_terminal_2026-11-02` study has no new recovery states
and must not be restarted under edited source. Its artifacts still verify;
`source_hashes_match: false` against current source is now expected historical
version separation, not data corruption. The old best checkpoints cannot be
retrofitted with missing callback/RNG/sampler state.

## Validation execution and next benchmark

`compiled_validation: true` caches a TensorFlow inference function with a
dynamic batch dimension. `validation_batch_size` can differ from the optimizer
batch size for full-cohort validation. Defaults preserve the prior eager path
and batch size. Optimizer batch size, labels, normalization and metric definitions
are unchanged by these switches. The final prediction export uses the selected
validation execution mode too.

When validation is capped, changing its batch size would change the selected
target mask. The trainer therefore rejects that combination. The capped
`experiment_bilstm_recovery_pilot.json` uses matching batch sizes of two,
two supervised epochs and two batches per phase; it is an integration recipe,
not evidence of convergence. Do not execute it before the gates above pass.

After those gates, plan a fresh pilot and inspect it before execution:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/research_cpu.ps1 -m src.experiments.run plan --config configs/sleepedf/experiments/experiment_bilstm_recovery_pilot.json --study-dir runs/my_recovery_pilot
```

Subsequent user-run pilot/benchmark commands will be supplied after reviewing
the gate results. A separate benchmark should measure full epoch time including
data loading, validation and checkpoint saving. Current inference checks cover
all four representations, tail batches, target identity, predicted classes,
probabilities, confusion matrices and metrics with `1e-6` tolerance. They do not
establish real-data full-cohort parity or a measured speedup, and must not be
used to pool old/new research results without the planned equivalence review.

Primary API references: [TensorFlow checkpoints](https://www.tensorflow.org/api_docs/python/tf/train/Checkpoint)
and [Keras serialization](https://keras.io/guides/serialization_and_saving/).
Recovery here uses explicit numeric arrays to include RNG variables and callback
state beyond the inference model; a best-model file alone is insufficient.

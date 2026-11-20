# Supervised recovery and validation execution — 13 November 2026

This implementation is opt-in. The user-run fresh-process synthetic optimizer
gate passed on 13 November: saved state and exported predictions matched exactly
for the tested gated BiLSTM on CPU. The [artifact review](../../reports/recovery_validation_2026-11-13/README.md)
records the evidence and its limits. Full regression subsequently passed on
15 November: 136 tests in 147.021 seconds, verified against the retained log.
M3/n70 subsequently passed full-cohort eager batch-4 parity on all 65,133
targets. The remaining thirteen also passed on 17 November; the 20 November
review verified full-family eager parity. The user also completed all eighteen
capped recovery/eager pilot jobs; all job seals and forty-two recovery generations
verify. Real-data resume equivalence and an end-to-end speed improvement remain
pending. No research training or
optimizer-based gate was launched by the assistant.

## Passed synthetic training gate (reproduction)

The user ran the following command successfully. It is retained for reproduction;
the next action is full regression below, rather than repeating this gate alone:

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

The user also passed the full regression command below; it is retained for reproduction:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/research_cpu.ps1 -m unittest discover -s tests -v
```

The full suite includes optimizer updates and remains user-owned. The bounded
saved-model check subsequently passed 13/14 models. M3 at n70 failed the
probability tolerance at compiled batch 32; eager replay passed and predicted
classes were unchanged on the 64 targets. M3/n70 then passed full-cohort eager
batch-4 parity (maximum probability difference 1.11e-16, zero class changes).
The remaining thirteen also passed full-cohort eager batch-4 parity. The next
action is the user-run [M6 epoch-one replay](../../reports/recovery_pilot_review_2026-11-20/README.md),
comparing resumed real-data training with its completed uninterrupted pilot.
The matched-batch diagnostic failed compiled 4/8/16/32 and eager 16/32; eager
4/8 passed on the sample. Keep compiled validation unaccepted and retain
eager batch 4, which passed for all fourteen saved classifiers. The original
recovery pilot enables compiled validation; the new eager pilot uses a separate
recipe and fresh study.
Do not relax tolerances based on this observation or interpret a passing
alternative batch size as a pass of the original declared mode. Real-data
resume equivalence and measured training throughput remain pending.

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
`experiment_bilstm_recovery_pilot.json` uses compiled validation and remains
unaccepted for the fallback. The separate
`experiment_bilstm_recovery_eager_pilot.json` uses eager validation, matching
batch sizes of four, two supervised epochs and two batches per phase. It is an
integration recipe, not evidence of convergence.

The user completed the eager pilot below; this command is retained for provenance:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/research_cpu.ps1 -m src.experiments.run execute --study-dir runs/bilstm_recovery_eager_pilot_2026-11-20 --live
```

All eighteen job seals and forty-two recovery generations have been inspected.
The next user-run command is the M6 replay in the linked pilot review. A separate
benchmark should measure full epoch time including
data loading, validation and checkpoint saving. Current inference checks cover
all four representations, tail batches, target identity, predicted classes,
probabilities, confusion matrices and metrics with `1e-6` tolerance. They do not
establish optimizer/callback equivalence or a measured speedup. The separate
user-run saved-model check passed real-data full-cohort eager batch-4 parity;
it does not establish equivalence of old and recovery-enabled training paths.

Primary API references: [TensorFlow checkpoints](https://www.tensorflow.org/api_docs/python/tf/train/Checkpoint)
and [Keras serialization](https://keras.io/guides/serialization_and_saving/).
Recovery here uses explicit numeric arrays to include RNG variables and callback
state beyond the inference model; a best-model file alone is insufficient.

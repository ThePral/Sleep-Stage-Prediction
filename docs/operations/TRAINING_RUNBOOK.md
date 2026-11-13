# Reproducible training commands

13 November engineering update: opt-in supervised recovery and validation
execution controls are implemented, with non-optimizer tests passed. Start
with the small user-run optimizer gate in
[SUPERVISED_RECOVERY.md](SUPERVISED_RECOVERY.md). Full regression, real-data
parity and runtime benchmarking remain pending; do not start a replication
matrix yet. Existing completed artifacts remain intact, but source hashes now
differ from their historical freeze, so new execution requires a fresh study.

Current state (12 November 2026): all 18 jobs in
`runs/bilstm_terminal_2026-11-02` are complete and verified. No training is running.
The [completion review](../../reports/bilstm_completion_2026-11-12/README.md)
contains audited validation results and analysis. The handoff/restart commands
below remain for reproduction; no more training is needed to complete this
one-seed matrix. New seeds or changed recipes require a separately frozen study
and user-launched training. Test and ISRUC scoring were not opened in the review.

## Current handoff: user-run terminal training

On 2 November 2026 the user took back training execution so progress is visible
in their own terminal. The background BiLSTM controller is stopped. Its capped
full-source benchmark completed all five jobs, and the disposable training
profile completed successfully. The old full matrix was interrupted during
waveform-statistics preparation; no full-schedule optimizer training started.
All completed artifacts remain intact.

`--live` now streams each worker's output to the terminal and its durable
attempt log. SSL and supervised phases print the first batch, every 100th
batch and the final batch, with loss, applicable accuracy, elapsed time and an
approximate remaining phase time. Source-validation summaries follow each
supervised epoch. Initial ETA includes compilation and should not be treated
as a stable throughput estimate. Preparation/integrity checks precede training
and can take several minutes without batch updates.

A fresh **18-job uncapped study has been planned, not executed**, under
`runs/bilstm_terminal_2026-11-02`. Fresh planning is required because live-output
code changed after the earlier study freeze. From a PowerShell terminal opened
at the checkout root, start it yourself:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/research_cpu.ps1 -m src.experiments.run execute --study-dir runs/bilstm_terminal_2026-11-02 --live
```

This runs three preparation jobs, ten-epoch SSL, then 14 supervised classifiers
across both label budgets. Each classifier allows up to 30 epochs with patience
five. Checkpoints/logs/caches stay on D: through the CPU launcher. No ISRUC or
new source-test predictive scoring is scheduled. The assistant will provide
commands and inspect outputs; it will not start training in the background.

For staged execution, add `--max-jobs 4` on the first call to complete the three
preparation jobs plus SSL and return before the classifiers. Later run the same
command without that limit to skip verified completed jobs and continue.

To check dataset availability before launching, run this read-only command:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/research_cpu.ps1 scripts/training_status.py --study-dir runs/bilstm_terminal_2026-11-02 --verify-data-scope --verify-artifacts
```

Require `data_scope.verified: true`, `source_hashes_match: true` and
`packages_match: true` before execution. Without `--verify-data-scope`, the
monitor reports `data_scope: null`; code/package agreement alone does not
establish dataset availability. This check verifies source identities, frozen
split and manifest hashes; it does not scan waveform payload integrity.
`Missing source train participants` means discovery at the chosen data root
did not find the complete frozen training scope. Do not shrink that scope or
disable the guard to proceed. Check the reported `data_root` and its recording
folders first. A failure before scheduling leaves no job attempt to retry.

On 6 November, a repeated startup failure was traced to protected Windows
access rules on prepared recording stores: the sandbox owner could read them,
but the normal checkout owner (`ELECOMP`) lacked access. Python's glob discovery
can omit inaccessible directories, producing the generic missing-participant
error. Passing checks from the assistant's account do not establish readability
from the user's terminal. Explicit read/traverse access was restored for that
account, with owners, file sizes/content-write times and existing grants retained.
The backup and repair evidence are under `reports/sleepedf_access_2026-11-06/`.

If this specific permission problem recurs, inspect the intended account/scope
with the following command. It performs no permission changes without `-Apply`:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/repair_sleepedf_access.ps1
```

The repair script targets only `data/processed/research_v1/sleepedf`, backs up
access rules before applying changes, and grants read access to the checkout
owner (or an explicitly supplied `-Account`). It does not take ownership,
grant write access, alter training code or start training.

Ctrl+C stops the worker tree and records an interrupted attempt. To retry in the
same unchanged study, use the execution command with `--live --retry-failed`.
Completed jobs are verified and skipped; the failed job starts a new attempt.
This scheduler retry does **not** resume within-job optimizer state. SSL's
separate checked epoch-resume API is documented in the implementation reference.
Do not edit source files or update packages once the study is frozen.

An example progress line is:

```text
SSL epoch 1/10 batch 100/10106 (1.0%) loss=... retrieval_accuracy=... elapsed=... ETA~...
```

The 116-test regression pass below is historical evidence from before live
logging changes. The handoff adds 14 passing non-optimizer progress/ownership
checks; it does not automatically rerun optimizer-based tests under user-owned
training. Details are in `reports/bilstm_terminal_handoff_2026-11-02/`.

## Earlier automated execution and reproduction history

The previous campaign was stopped explicitly by the user on 2 November 2026,
with three complete SSL epochs preserved and no supervised jobs started.
See [the stop record](../../reports/organization_2026-11-02/training_stop.json).
The BiLSTM replacement is now implemented and its full 116-test regression
suite passes. [SLEEPEDF_BILSTM.md](../SLEEPEDF_BILSTM.md) describes the architecture
and new configuration names. A fresh CPU staged runner uses
`runs/bilstm_2026-11-02/status.json`: seven-condition integration pilot, full-source
capped benchmark, disposable training profile, then the uncapped source matrix.
Every stage must pass before the next starts. Read that status and job ledgers
for actual progress; startup is not completion or convergence.

For independent reproduction, use the commands in the BiLSTM guide and fresh
output directories. To plan the uncapped version after successful checks:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/research_cpu.ps1 -m src.experiments.run plan --config configs/sleepedf/experiments/experiment_bilstm_matrix.json --study-dir runs/my_bilstm_matrix
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/research_cpu.ps1 -m src.experiments.run execute --study-dir runs/my_bilstm_matrix
```

The later sections retain earlier GRU/LSTM recipes for reproducibility. The old
frozen GRU matrix must not be restarted under changed code. The new campaign
starts source-only SSL afresh and does not bypass the stopped checkpoint's
code/resume guards. No real ISRUC or new source-test scoring is scheduled.

## Current route: native CPU, runtime files on D: (2 November 2026)

The WSL/GPU pipeline is stopped. Use the existing Windows environment without
installing packages or invoking WSL. The CPU launcher sets process-local temp
and cache paths under `runs/cpu_runtime`, disables CUDA and bytecode writes, and
restores the caller's environment afterward. Child experiment workers inherit
these settings. Data, study files and logs remain in this D: checkout.
The Python base installation can still be read from C:. Windows may write paging
or system files there; this launcher cannot guarantee zero operating-system I/O
or shrink an existing WSL image. No C: files or system settings were changed.

From the project root, these commands use a process-only execution-policy option
because script execution is disabled on this machine:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/research_cpu.ps1 -m pip check
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/research_cpu.ps1 -m src.compute.preflight --allow-cpu --report runs/my_cpu_preflight.json
```

The stopped CPU campaign used the configuration now at `configs/sleepedf/experiments/experiment_matrix.json`:
supervised batch 4 and SSL batch 32, no epoch/batch caps, seven conditions,
18/70 labeled training participants, 15 validation participants, seed 42.
Use a fresh directory for independent reproduction:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/research_cpu.ps1 -m src.experiments.run plan --config configs/sleepedf/experiments/experiment_matrix.json --study-dir runs/my_cpu_source_matrix
if ($LASTEXITCODE -ne 0) { throw 'Study planning failed.' }
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/research_cpu.ps1 -m src.experiments.run execute --study-dir runs/my_cpu_source_matrix
```

Do not pass `--require-gpu` for CPU reproduction. The stopped run's directory is
`runs/training_execution_2026-11-02_cpu/source_matrix`; inspect `ledger.json`
and the active job's `attempt_0001.log`. Its outer process status is
`runs/training_execution_2026-11-02_cpu/status.json`. To continue after a normal
stop, use the same execute command and study path; failed attempts additionally
require `--retry-failed`. Never start a second scheduler while its lock/process
is active. Completion is established by all 18 sealed jobs and the execution
summary, not by a stale `running` status. Keep source code/packages unchanged
throughout this campaign. The earlier CPU five-job benchmark was capped and
does not establish convergence. Real ISRUC/source-test scoring remains unopened.

### Inspect a live CPU study

This read-only command checks the scheduler PID and its command line, samples
its process tree for five seconds, compares frozen source/package versions,
and verifies every completed job's artifact hashes. It imports no TensorFlow,
performs no optimizer updates and does not alter the study:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/research_cpu.ps1 scripts/training_status.py --study-dir runs/training_execution_2026-11-02_cpu/source_matrix --sample-seconds 5 --verify-artifacts
```

Add `--report runs/my_progress_01.json` to preserve a snapshot; use a new filename
each time, outside the study directory. A verified live scheduler and observed
CPU activity distinguish the process from a stale `running` record. CPU activity
alone does not prove optimizer progress. SSL publishes its epoch count and
checkpoint only after a full epoch, so `published_epochs: 0` can persist while
the first epoch runs. Supervised history is also written at epoch boundaries.
No ETA is inferred before real epoch timings are available. Final status
`completed_in_ledger` is a recorded claim; use `--verify-artifacts` to require
all completed artifact seals before `completed_artifacts_verified` is reported.
If the scheduler is missing or its PID identity cannot be checked, inspect the
logs/processes before any recovery. The monitor never deletes a lock or restarts
a job automatically. It opens no source-test or ISRUC predictions.

The actual campaign's source scope, annotation-hour costs, normalization pools,
M0-M6 definitions and planned comparisons are recorded in
`reports/source_campaign_2026-11-02/METHODS.md` with a machine-readable JSON
companion. This is a methods record, not an executed external freeze. The first
full SSL epoch's saved counter/hash/finiteness checks are recorded separately in
`reports/cpu_training_2026-11-02/ssl_epoch01_verification.json`.

## Historical WSL/GPU route (stopped; retain for reproduction)

The user subsequently delegated training execution to the assistant on
1 November 2026. The manual commands below remain available for reproduction.
The dependency helper installs packages only; a separate authorized execution
pipeline gates GPU preflight, the capped benchmark, source profiling and the full
matrix on successful checks. Its live status is
`runs/training_execution_2026-11-01_gpu/status.json`. A failed stage stops the pipeline.

The measured GPU configuration uses supervised batch **32** and SSL batch
**128**. Both synthetic workloads passed on the GTX 1050; the combined TF
allocator peak was about 199 MiB, excluding driver reservations. These choices
use `configs/compute/gpu/experiment_gpu_benchmark.json` and `configs/compute/gpu/experiment_gpu_matrix.json`.
The original CPU/reference configurations remain unchanged.

The original prepared-store directories have traversal-only permissions in WSL,
so discovery cannot enumerate their manifests there. The authorized run uses the
verified source-only capsule extraction instead. All 167 included manifest hashes
match the original stores. Set its path for the commands below:

```powershell
$sourceData = 'runs/compute_2026-10-31/restored_source_2026-11-01/data/processed/research_v1'
```

Run the commands below in PowerShell from the project root. The launcher selects
the existing Ubuntu-24.04 WSL Python and places framework caches under `runs/`.
It never falls back to native Windows Python. No administrator shell is needed.
If script execution is blocked, prefix an invocation with
`powershell -NoProfile -ExecutionPolicy Bypass -File scripts/research.ps1` instead
of `.\scripts\research.ps1`; this affects only that process.

## 1. Wait for dependency installation

```powershell
Get-Content runs/compute_2026-10-31/gpu_setup_status_2026-11-01.json
Get-Content runs/compute_2026-10-31/dependency_only_install_2026-11-01.log -Tail 10
```

Continue when status is `dependencies_installed_pending_user_gpu_checks`.
The current installation has advanced to `dependencies_installed_gpu_preflight_passed`;
use the pipeline status above to inspect actual source-training progress.
`installing_dependencies_only` means downloads/installation are still running;
`dependency_installation_failed` means setup needs investigation. A stale status
after shutting down WSL is not proof that an installer is still active. Partial
wheels are retained for verified resumption.

The native Windows report `reports/compute_2026-11-01/windows_gpu_required.json`
is expected to report no GPU for that environment. The next check must run in WSL.
Existing WSL warnings about missing `F:` entries in the Windows PATH can appear;
the launcher argument check succeeded despite them. Use the command exit code
and saved report to assess the actual check.

## 2. Run the GPU training-kernel check yourself

```powershell
.\scripts\research.ps1 -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Resolve dependency errors before continuing.' }

.\scripts\research.ps1 scripts/configure_wsl_cuda.py --report runs/user_cuda_links_2026-11-01.json
if ($LASTEXITCODE -ne 0) { throw 'Resolve CUDA library discovery before continuing.' }

.\scripts\research.ps1 -m src.compute.preflight --supervised-batch-size 32 --contrastive-batch-size 128 --report runs/user_gpu_preflight_2026-11-01.json
if ($LASTEXITCODE -ne 0) { throw 'GPU preflight failed. Inspect the saved report before training.' }

Get-Content runs/user_gpu_preflight_2026-11-01.json
```

This command performs synthetic optimizer updates. Require `passed: true`,
`requested_device: /GPU:0`, and passed supervised and
contrastive checks. The command checks 32 temporal examples and 128 original
contrastive epochs. The report records actual output devices and framework
allocator memory. A report cannot overwrite an older one; use a new filename for
another attempt. Do not add `--allow-cpu` to establish GPU readiness.

The CUDA configuration command creates shared-library links inside the virtual
environment, following TensorFlow's documented GPU discovery fix. It also links
the bundled `ptxas` executable there. It never replaces existing files or changes
drivers/system paths. Existing matching links are verified on later invocations;
reports still require new filenames. This step fixes discovery, not kernel
compatibility ([TensorFlow setup](https://www.tensorflow.org/install/pip)).
This WSL environment also needs explicit library search paths. The launcher
sets `LD_LIBRARY_PATH` for its child process using the installed NVIDIA library
directories and WSL driver directory; it changes no global environment settings.

## 3. Run the bounded GPU benchmark

Use a new study because the launcher/GPU-guard implementation was added after
the historical CPU study. Planning reads source data and freezes the current
implementation, packages, configuration and participant budget; it performs no
optimizer updates.

```powershell
.\scripts\research.ps1 -m src.experiments.run plan --config configs/compute/gpu/experiment_gpu_benchmark.json --study-dir runs/user_gpu_benchmark_2026-11-01 --data-root $sourceData
if ($LASTEXITCODE -ne 0) { throw 'Study planning failed; inspect the error.' }

.\scripts\research.ps1 -m src.experiments.run execute --study-dir runs/user_gpu_benchmark_2026-11-01 --require-gpu
```

Expect five completed jobs: source waveform statistics, the full-label-budget
normalizer, SSL, M3 gated scratch, and M6 gated fine-tuning. All 70 source training
and 15 validation participants are in scope; schedules remain capped at two
epochs, 128 training/pretraining batches and 64 validation batches. Both
classifiers use identical normalization and 2,048 validation targets. This checks
real loading, checkpointing and transfer; it cannot establish convergence.

`--require-gpu` rejects execution before reading a study or scheduling any job if
TensorFlow exposes no GPU. It checks device visibility; the earlier preflight is
still needed to check kernels. It does not substitute for measuring GPU memory.

Inspect these outputs:

```powershell
Get-Content runs/user_gpu_benchmark_2026-11-01/execution_summary.json
Import-Csv runs/user_gpu_benchmark_2026-11-01/source_validation_results.csv | Format-Table condition,validation_macro_f1,validation_accuracy,epochs_completed,best_epoch
Get-Content runs/user_gpu_benchmark_2026-11-01/paired_source_validation.csv
Get-Content runs/user_gpu_benchmark_2026-11-01/jobs/ssl_seed42/attempt_0001/artifacts/diagnostics.json
```

The historical capped CPU run had lower M6 than M3 F1. It does not establish an
SSL benefit or harm and does not justify tuning on ISRUC.

## 4. Measure real source training on the GPU

After a successful benchmark, run this disposable optimizer-based profile:

```powershell
.\scripts\research.ps1 -m src.evaluation.profile --bundle runs/user_gpu_benchmark_2026-11-01/jobs/M3_draw42_n70_seed42/attempt_0001/artifacts --data-root $sourceData --output-dir runs/user_gpu_profile_2026-11-01 --recordings SC4001 --mode training --batch-size 32 --warmup-steps 5 --measured-steps 50
Get-Content runs/user_gpu_profile_2026-11-01/profile.json
```

Profiling discards its optimizer updates and verifies unchanged saved artifacts.
Host RSS and GPU allocator bytes are separate measurements. One cached recording
does not measure full-source random-loader performance or campaign duration.
The preflight's SSL workload separately checks batch 128. Use fresh report/output
paths for retries.

## 5. Start the first full source matrix

Once GPU execution and memory fit are established, use the uncapped source
configuration. This is a substantial training run.

```powershell
.\scripts\research.ps1 -m src.experiments.run plan --config configs/compute/gpu/experiment_gpu_matrix.json --study-dir runs/user_source_matrix_seed42_2026-11-01 --data-root $sourceData
if ($LASTEXITCODE -ne 0) { throw 'Study planning failed; inspect the error.' }

.\scripts\research.ps1 -m src.experiments.run execute --study-dir runs/user_source_matrix_seed42_2026-11-01 --require-gpu
```

The matrix has **18 jobs**, **seven conditions**, labeled budgets of **18 and 70
participants**, **15 validation participants**, and optimization seed **42**.
SSL uses all source-training quality-valid epochs for ten epochs. Supervised
models train for at most 30 epochs with patience five and checkpoint selection
by source-validation macro-F1. M0 is a freshly trained spectral reference.
M1/M2/M3 are raw/concatenated/gated scratch models; M4/M5/M6 are their SSL versions.

This is the first full-seed experiment. Repeated seeds and any final protocol
decisions remain required before publication claims and external evaluation.
Do not change source files or package versions during a frozen study.

## Continuing an interrupted study

The same execution command verifies and skips completed jobs. To stop between
jobs, add `--max-jobs 1` (or another positive count) to an execution call. This
limits newly attempted jobs, not epochs or batches within a job.

If an attempt failed or was interrupted:

```powershell
.\scripts\research.ps1 -m src.experiments.run execute --study-dir runs/user_source_matrix_seed42_2026-11-01 --require-gpu --retry-failed
```

Retries preserve the prior attempt and start that job afresh. The scheduler does
not resume supervised optimizer state; standalone SSL has separate checked
checkpoint recovery. A stale `.runner.lock` after a hard shutdown must be
investigated before starting another scheduler. Use a fresh study directory if
code or packages changed; do not edit old seals to bypass integrity checks.

## Outputs to review next

The GPU preflight JSON, benchmark execution summary, source-validation CSV,
SSL diagnostics and GPU profile are enough to review compute readiness. After
the full matrix, inspect full learning curves, selected epochs, class-level
metrics and paired source results before choosing the next scientific step.
No command here opens source-test or ISRUC predictive performance.

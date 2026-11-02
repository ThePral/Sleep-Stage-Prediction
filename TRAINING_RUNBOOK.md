# Reproducible training commands

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

The active CPU campaign uses the original `configs/experiment_matrix.json`:
supervised batch 4 and SSL batch 32, no epoch/batch caps, seven conditions,
18/70 labeled training participants, 15 validation participants, seed 42.
Use a fresh directory for independent reproduction:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/research_cpu.ps1 -m src.experiments.run plan --config configs/experiment_matrix.json --study-dir runs/my_cpu_source_matrix
if ($LASTEXITCODE -ne 0) { throw 'Study planning failed.' }
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/research_cpu.ps1 -m src.experiments.run execute --study-dir runs/my_cpu_source_matrix
```

Do not pass `--require-gpu`. The current run's directory is
`runs/training_execution_2026-11-02_cpu/source_matrix`; inspect `ledger.json`
and the active job's `attempt_0001.log`. Its outer process status is
`runs/training_execution_2026-11-02_cpu/status.json`. To continue after a normal
stop, use the same execute command and study path; failed attempts additionally
require `--retry-failed`. Never start a second scheduler while its lock/process
is active. Completion is established by all 18 sealed jobs and the execution
summary, not by a stale `running` status. Keep source code/packages unchanged
throughout this campaign. The earlier CPU five-job benchmark was capped and
does not establish convergence. Real ISRUC/source-test scoring remains unopened.

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
use `configs/experiment_gpu_benchmark.json` and `configs/experiment_gpu_matrix.json`.
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
.\scripts\research.ps1 -m src.experiments.run plan --config configs/experiment_gpu_benchmark.json --study-dir runs/user_gpu_benchmark_2026-11-01 --data-root $sourceData
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
.\scripts\research.ps1 -m src.experiments.run plan --config configs/experiment_gpu_matrix.json --study-dir runs/user_source_matrix_seed42_2026-11-01 --data-root $sourceData
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

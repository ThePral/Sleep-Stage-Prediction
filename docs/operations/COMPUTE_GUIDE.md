# Source-training compute workflow

The prior CPU matrix is now stopped at the user's request, with three sealed
SSL epochs preserved. Sleep-EDF SSL/cross-attention/BiLSTM is now implemented
and a fresh staged CPU workflow runs under `runs/bilstm_2026-11-02` after its
regression/kernel gates passed. See [the implementation guide](../SLEEPEDF_BILSTM.md)
for new recipes and inspect live status for actual stage completion. GPU routes
below are retained historical recipes. The current runtime route remains native CPU
with project-managed storage on D:. See [research direction](../RESEARCH_DIRECTION.md).

The bounded source-only benchmark completed on CPU before that matrix. Follow
[TRAINING_RUNBOOK.md](TRAINING_RUNBOOK.md) for
the exact PowerShell commands. The benchmark uses
all 70 training and 15 validation participants, the default architecture, and
short capped schedules. Its scores cannot establish convergence or support a
confirmatory ISRUC freeze.

## Local environment

Hardware inspection found an NVIDIA GeForce GTX 1050 with 4,096 MiB VRAM, compute
capability 6.1 (Pascal), driver 581.80, approximately 16 GiB host RAM and 12 logical
CPUs. Existing Ubuntu 24.04 under WSL2 exposes the same GPU and Python 3.12.3.
The existing native Windows Python 3.13 environment runs TensorFlow 2.21 on CPU.
Native Windows GPU support ended at TensorFlow 2.10; WSL2 is the supported path
for current releases ([TensorFlow installation](https://www.tensorflow.org/install/pip)).

`environments/requirements-linux-gpu.txt` pins the framework and direct research dependencies.
Its CUDA constraint file pins CUDA 12.5 libraries and cuDNN 9.3, matching the
documented TensorFlow 2.21 build combination
([tested builds](https://www.tensorflow.org/install/source)). cuDNN 9.3 supports
Pascal; NVIDIA recommends CUDA 11.8 for tuning on pre-Volta hardware, while its
support matrix also lists CUDA 12.5
([cuDNN 9.3 matrix](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.3.0/reference/support-matrix.html)).
Broad dependency ranges can select newer cuDNN releases that drop Pascal
([cuDNN 9.27 matrix](https://docs.nvidia.com/deeplearning/cudnn/backend/v9.27.0/reference/support-matrix.html)).
Pins alone do not establish that the TensorFlow wheel contains usable kernels
for this GPU. Run the preflight below before accepting the environment.

For a standard Linux/WSL Python with venv/pip available, run from the checkout:

```bash
python3 -m venv .venv-linux
.venv-linux/bin/python -m pip install -r environments/requirements-linux-gpu.txt
.venv-linux/bin/python -m pip check
.venv-linux/bin/python -m pip freeze > requirements-linux-installed.lock.txt
.venv-linux/bin/python scripts/configure_wsl_cuda.py --report runs/cuda_links.json
.venv-linux/bin/python -u -m src.compute.preflight --report runs/gpu_preflight.json
```

The inspected local Ubuntu lacks ensurepip. The project-local setup attempt uses
`runs/compute_2026-10-31/linux_env` and bootstraps pip from the existing trusted
Windows pip code, then installs Linux packages. It changes no system packages or
drivers. Dependency downloads and logs stay under that ignored run directory.
Do not mix Windows and Linux site-packages for research execution.

Use a fresh CLI process for preflight and each profile. Preflight executes actual
gated supervised updates (batch four, ten epochs of context per example) and
contrastive updates (32 original epochs, 64 augmented views). It requires a GPU
by default, records build/device information and finite losses, and fails closed
when no GPU is visible or a kernel fails. `--allow-cpu` exercises the same check
on a CPU-only installation; that result does not establish GPU fit. Synthetic
checks do not measure disk loading, convergence, or the full experiment duration.
Reports are immutable; choose a new path for every attempt.

The first installed WSL preflight failed GPU discovery because the loader could
not find pip-installed CUDA libraries. `scripts/configure_wsl_cuda.py` implements
TensorFlow's documented virtual-environment shared-library links. It operates
inside the inspected venv, checks all conflicts first and never replaces existing
files. It changes no drivers or system library paths. A fresh preflight is still
required after configuration; discovery alone does not establish usable kernels.

This WSL setup also requires process-local CUDA library search paths. The
PowerShell launcher provides them without changing system paths. Measured
synthetic GPU checks pass at supervised batch 32 and SSL batch 128 (about 199 MiB
combined TF allocator peak). GPU recipes live in `configs/compute/gpu/experiment_gpu_*.json`;
the original CPU/reference configs remain intact. Follow the runbook's explicit
data-root selection: WSL cannot enumerate the original traversal-only store
directories, so the active run reads the verified source-only capsule copy.

Recorded on 1 November 2026: all 103 regression tests passed. A fresh-process CPU
preflight completed both default-batch workloads. The required-GPU check correctly
failed on native Windows. The WSL GPU runtime download is still in progress at
the evidence snapshot; Linux dependency compatibility and GPU kernels remain
unverified. The initially queued automatic preflight/profile was stopped when the
user initially requested ownership of all training. The dependency continuation installs
dependencies only, then records `dependencies_installed_pending_user_gpu_checks`
in `runs/compute_2026-10-31/gpu_setup_status_2026-11-01.json`. It executes no model
or optimizer checks. The user subsequently delegated training to the assistant.
A separate execution pipeline now gates preflight, profiling and source training
on successful checks; inspect `runs/training_execution_2026-11-01_gpu/status.json`.
The runbook retains the exact reproduction commands. Interruption or download
failure can leave setup incomplete.

```powershell
Get-Content runs/compute_2026-10-31/gpu_setup_status_2026-11-01.json
```

The tracked snapshot is in `reports/compute_2026-11-01/evidence.json`. The live
status and subsequent GPU results stay under ignored `runs/` until reviewed.

## Bounded full-source benchmark

Use the Python executable of the verified environment for every command:

```bash
python -u -m src.experiments.run plan \
  --config configs/sleepedf/experiments/experiment_source_benchmark.json \
  --study-dir runs/source_benchmark
python -u -m src.experiments.run execute \
  --study-dir runs/source_benchmark --require-gpu
```

The five jobs fit full-source waveform statistics, fit the 100% labeled-budget
normalizer, pretrain a source-only encoder, train gated fusion from scratch (M3),
and fine-tune the same fusion architecture from that encoder (M6). Each model has
at most two epochs; each epoch is capped at 128 training/pretraining batches and
64 validation batches. All source recordings enter the declared scope, but a
capped epoch visits only a subset of their targets. These are implementation and
workload checks. The pretraining checkpoint follows the fixed short schedule;
fine-tuning uses source validation only. Both conditions share normalization,
label budget, seed and validation targets. No test or external prediction job is
scheduled.

Review `execution_summary.json`, worker logs, SSL diagnostics, sealed bundles,
`source_validation_results.csv` and `paired_source_validation.csv`. Profile a
completed bundle in a fresh process, using records included in its source scope:

```bash
python -u -m src.evaluation.profile \
  --bundle <completed-supervised-attempt>/artifacts \
  --output-dir runs/gated_gpu_training_b4 \
  --mode training --batch-size 4 --warmup-steps 5 --measured-steps 50
```

Host RSS and TF allocator GPU bytes measure different resources. Allocator bytes
exclude driver reservations; check available VRAM alongside them. Profiling uses
disposable optimizer updates and verifies that saved files stay unchanged. Random
access to many memory-mapped stores on `/mnt/d` may behave differently from a
small cached-record profile or native Linux storage.

Study planning freezes all source-code hashes and package versions. Create a new
study after changing code or environment; retain previous results as historical
evidence. `execute --max-jobs N` stops between jobs and later skips verified
completed jobs. `--retry-failed` preserves an interrupted attempt and starts that
job again. The scheduler does not resume supervised optimizer state. Standalone
SSL has a separate checked resume mechanism.

## Portable source-only training capsule

The capsule supports moving a reviewed training workload to another Linux GPU
machine. Creating it is local and performs no upload or cloud provisioning.

```powershell
.venv\Scripts\python.exe -m src.compute.capsule create --archive runs/source_training.tar
```

The archive contains an explicit allowlist of Python source/tests, JSON configs,
compute documentation/dependencies, the frozen participant split, and prepared
Sleep-EDF training/validation stores only. The split CSV contains all participant
assignments, including test identities; test epoch files are excluded. ISRUC,
raw recordings, old checkpoints, run directories and hidden credential files are
excluded. Each included store is validated before packaging. File hashes and an
archive SHA-256 support integrity checks; they do not anonymize EEG. Review data
use terms and the selected destination before any transfer.

On a new machine, use a trusted checkout of this capsule tool to verify the
externally retained archive SHA-256 and extract into a new directory:

```bash
python -m src.compute.capsule unpack --archive source_training.tar \
  --destination restored_source --expected-sha256 <recorded-archive-sha256>
python -m src.compute.capsule verify --destination restored_source
```

Extraction checks archive/member hashes and rejects traversal, links, duplicate
members, undeclared files and held-out epoch scope. It never replaces an existing
destination. Failed extraction can leave a partial new directory; no completed
manifest is published until payload verification succeeds. Preserve the reported
archive hash separately from the archive. Create a fresh study at the destination
so runtime paths and packages are recorded there.

## Full campaign and external evaluation

After reviewing the benchmark, plan a fresh study with
`configs/sleepedf/experiments/experiment_matrix.json`. Its default one-seed matrix is the first full
source experiment, not a repeated-seed result. Freeze any additional seeds,
budgets and ablations before confirmatory comparisons. Estimate compute from
actual loader/training timings and epoch counts; early stopping makes the total
uncertain. Keep checkpoints and logs on persistent storage.

The `--require-gpu` execution flag fails before scheduling any job when TensorFlow
exposes no GPU. It checks visibility rather than kernel compatibility; complete
the user-run preflight first. CPU execution remains available by omitting the flag.

## Native CPU route with runtime storage on D: (2 November 2026)

Use `scripts/research_cpu.ps1`, as documented in `docs/operations/TRAINING_RUNBOOK.md`, instead
of the WSL launcher. It uses the existing `.venv` on D:, disables CUDA, and
redirects Python temporary files, pip/Keras/Matplotlib caches and worker scratch
files under `runs/cpu_runtime`. Checkpoints and studies also stay on D:.
No new installation is required. The original batch sizes (4 supervised, 32 SSL)
keep working memory small; the uncapped matrix can take days on this CPU.
Synthetic timings do not establish full-campaign duration or convergence.

The stopped GPU benchmark retains two completed preparation jobs and an
interrupted SSL attempt; the GPU full matrix never started. The shared-memory
cache was absent when inspected on 2 November. No original stores were changed,
no global WSL shutdown was issued, and no WSL images/pagefiles/C: files were
modified. Windows can still write its own logs or pagefile to C:; the CPU route
controls project-managed storage rather than all operating-system writes.
Current CPU evidence is in `reports/cpu_training_2026-11-02` and live state in
`runs/training_execution_2026-11-02_cpu`.

Cloud training means renting a remote GPU and running these same commands there.
No provider account, paid instance, upload or spending commitment is needed for
the local capsule/benchmark. If cloud compute becomes necessary, choose hardware,
persistent disk and a cost limit using the measured workload before provisioning.
ISRUC remains local and unopened for predictive scoring until the complete model
family and evaluation protocol have been source-selected and frozen.

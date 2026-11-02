# Current scope and organization - 2 November 2026

The current paper direction is Sleep-EDF Expanded with SSL, cross-attention
and a planned BiLSTM temporal head. The previous GRU matrix was stopped at the
user's request after three saved SSL epochs, before supervised training.
Cross-dataset work is deferred and preserved. BiLSTM integration is not yet
implemented, and no replacement campaign is running.

See [research direction](RESEARCH_DIRECTION.md), [project structure](PROJECT_STRUCTURE.md)
and [organization evidence](../reports/organization_2026-11-02/README.md).
The dated milestones below describe the earlier implementation/history.
Current configuration paths are indexed in [configs/README.md](../configs/README.md).
Only capsule packaging metadata changed under `src/` during organization;
new studies must freeze the current code before further research training.

---

**Research implementation — started 27 October 2026.**

The one-month objective is the complete implementation and validation of the
multi-view, self-supervised, and source-only external-evaluation workflows.
This file tracks delivered code separately from research conclusions.

| Milestone | Status | Deliverable |
|---|---|---|
| Timeline and sequence-index repairs | Implemented; regression tests pass | Physical indices survive excluded annotations; arbitrary pandas indices no longer reorder features |
| Shared waveform/spectral epoch format | Implemented | Full epoch grids, stable IDs, quality/label masks, declared units, and no fitted normalization |
| Sleep-EDF adapter | Implemented | Corrected frozen participant split; PSG/hypnogram identity/time-origin checks |
| ISRUC-I adapter | Implemented | Canonical numeric recordings; fixed central derivation; explicit workbook epoch labels and scorer |
| Recording storage and preparation CLI | Implemented | Atomic publication, hashes, memory maps, verified resume, no conflicting-store overwrite |
| Full dataset regeneration | Complete and validated | 197 Sleep-EDF and 100 ISRUC-I recordings; all 457,577 source supervised feature rows match the preserved cache |
| Canonical feature-model inference | Implemented | Directory loader and early rejection of model/configuration shape or feature-order mismatches |
| Collection and baseline verification CLI | Passed on full collections | Identity/split/coverage checks and exact reproduction of the source confusion matrix on 69,539 windows |
| Waveform encoder, feature tokens, and fusion models | Implemented | Shared strided CNN, ten feature tokens, raw-only/concatenation/gated-attention controls, and a temporal GRU |
| Source-only normalization and streaming batches | Implemented | Unique training-epoch statistics, frozen split checks, bounded memory maps, reproducible window ordering |
| Supervised trainer and multiview inference | Implemented; source integration pilots passed | Validation macro-F1 checkpoints, hashed completed bundles, prediction exports, and checkpoint reload checks |
| Contrastive pretraining and fine-tuning | Implemented; CPU integration pilots passed | Quality-only source pool, stateless augmentations, masked NT-Xent, recoverable optimizer state, verified encoder transfer |
| Frozen-encoder linear probe | Implemented | Single-epoch mean pooling and affine classifier; CNN weights checked unchanged |
| Matched participant label budgets | Implemented; automated checks pass | Nested whole-participant draws, class-coverage checks, full unlabeled pool, and explicit annotation costs |
| Source experiment runner | Implemented; real-data CLI pilots passed | M0–M6 dependency graph, shared normalization/checkpoints, isolated serial workers, sealed attempts and paired source tables |
| Source compute profiler | Implemented; CPU pilots passed | Disposable training/inference steps, cached-loader timing, host RSS and optional TF GPU allocator measurements |
| Participant uncertainty and calibration | Implemented; automated checks pass | Participant-cluster paired intervals, separate seed variation, Brier/ECE and class support |
| Frozen evaluation workflow | Implemented; synthetic external tests pass | Source-only freeze, immutable snapshots, enrollment/mask checks and whole-family scoring |
| Full research campaign and real external evaluation | Pending | Adequately trained source models, GPU validation, repeated-seed results and prespecified ISRUC scoring |

**Validation and data notes**

The data-foundation milestone passed **36 automated tests**. Data regression checks are in
`tests/test_data_foundation.py`; model/scaler serialization and incompatible
bundle rejection checks are in `tests/test_inference_bundle.py`. They cover
annotation gaps, recording bounds, index-label independence, signal QC, resampling,
relative/absolute power behavior, workbook parsing, channel reconstruction,
atomic cache round trips, tampering, split conflicts, and window eligibility.
Run them with the command documented in the README.

Prepared stores are written under `data/processed/research_v1`. Logs and the final
collection-validation report belong under `reports/implementation_2026-10-27`.
Original signal files, feature caches, model files, and saved experiment results
are preserved. The notebooks already modified when work began remain historical
inputs rather than the extension's execution interface.

The complete collection validation found 461,870 physical Sleep-EDF epochs,
including 457,577 supervised-eligible epochs, and 90,187 physical ISRUC epochs,
including 89,995 scorer-1 supervised-eligible epochs. The source participant
split remains 70/15/15 with no participant overlap. All 100 ISRUC participants
remain external. Signal-valid unlabeled epochs are retained separately for
source-only pretraining eligibility.

**Supervised model milestone**

At the supervised milestone, the suite passed **56 tests**: the 36 data/bundle checks from the
first milestone and 20 model/training checks. The recorded source pilots and
replay checks are in `reports/supervised_2026-10-27/pilot_summary.json`.

The default raw, concatenation, and gated-attention models have 56,453, 65,989,
and 82,757 parameters respectively. All share the same waveform encoder and
temporal context. The gated model attends over ten spectral feature tokens;
diagnostic attention/gate tensors can be exposed through a Keras model.

All three variants completed two-epoch source-only integration pilots using
SC4001/SC4002 for training and SC4011 for validation, with two batches per
epoch/validation pass. Normalization was shared across variants and fitted on
5,479 unique source training epochs. Each best checkpoint reloaded with zero
probability difference in the sampled check. These deliberately capped runs
verify the implementation; they are not trained-model performance evidence.
No new source-test or ISRUC predictions were used for model selection.
Replaying the exported validation epoch IDs through the raw-input inference
wrapper agreed within 1e-6 for all three models in a fresh process.

`tests/test_multiview.py` adds checks for attention keys and gradients, batch
independence, reproducible initialization, backbone transfer, serialization,
tiny-problem overfitting, source-only fitting, capped validation sampling,
metric agreement, and complete training-to-inference bundle execution.
Reproduction commands are in the README. Full source training, GPU memory and
throughput measurements, full-budget SSL, and the subsequent frozen external experiment
remain distinct experimental/implementation steps.

**Contrastive implementation — 31 October 2026**

The source-only raw SSL path uses individual signal-valid epochs, including
unscored epochs. Store integrity validation can inspect annotations, but the pool
retains neither stage labels nor spectral features. Hiding every annotation in
the test fixture preserves epoch eligibility, waveform statistics and seeded
batch order. Held-out source participants and ISRUC cannot enter the pool.

Two independently augmented views feed the shared CNN and a discarded projection
head. NT-Xent retains the positive in the denominator, excludes self-similarity,
and masks nearby same-recording negatives. Mild gain/noise and a short contiguous
mask are explicit initial assumptions; their physiological suitability and
same-stage false negatives remain research questions. The default pretraining
model has 35,328 parameters, of which 27,008 belong to the transferable CNN.

An atomic epoch-boundary state records encoder/projector/optimizer tensors and
history. Sampler/view randomness is stateless and reproducible from the seed,
epoch and batch. Resume requires unchanged source content, update settings,
package versions and relevant code, and writes a new run directory. The test
suite compares recovery after a simulated interruption with uninterrupted
updates. Completed bundles verify file hashes and reloaded encoder outputs.

Downstream transfer validates architecture, source identities and preprocessing.
Ordinary SSL transfer requires the same source recording selection; the later
budget workflow authorizes labeled subsets of a verified full training pool. The
SSL waveform scale is preserved, while the spectral scaler retains its separate
supervised source fitting population. Scratch controls can reuse that exact
waveform scale without loading pretrained weights. Linear probes freeze the CNN
and fit only a 325-parameter affine head on a single epoch; frozen temporal/fusion
head training is also supported and is explicitly distinguished from probing.

At the contrastive milestone, the suite passed **68 tests**, including 12 contrastive/data/transfer
checks. CPU pilots completed raw pretraining, epoch-boundary resume, gated
fine-tuning, a matched scratch control and a frozen linear probe. The complete
70-participant source-training pool contains 323,423 quality-valid epochs. Its
waveform statistics were fitted on that full pool, and a deliberately capped
two-batch optimization pass completed successfully. The smaller transfer pilot
fitted waveform statistics on 5,480 source epochs and spectral statistics on
5,479 supervised source epochs from SC4001/SC4002, validating on SC4011.

Implementation pilots, test totals and limitations are recorded in
`reports/contrastive_2026-10-31/pilot_summary.json`. These runs verify execution
and source boundaries; they do not demonstrate convergence, label efficiency,
physiological invariance, or external generalization. No new test/ISRUC predictive
scores are used. Participant label budgets and the source runner are delivered
in the next milestone below; uncertainty estimation and the frozen external
scoring batch remain upcoming work.

**Participant-budget and experiment-runner implementation — 31 October 2026**

Nested participant draws retain all recordings for each chosen participant.
Fractions round up, so the 70-participant source pool yields 7/18/35/70 labeled
participants for 10/25/50/100% budgets. Class coverage is checked on eligible
training targets after selection, without label-driven redrawing. The manifest
reports recording hours, available five-class annotation hours and usable hours,
plus the fixed validation annotation cost. This is training-label efficiency
with a fully labeled validation set, rather than a total annotation-budget claim.

Waveform statistics use the fixed full quality-valid training pool for every
condition. Spectral statistics use only each labeled budget. Paired models share
the same normalizer file; one SSL checkpoint per optimization seed is reused
across draws and budgets. Transfer permits a subset only when its frozen split,
recording contents and complete unlabeled pool agree with the budget manifest.

The runner includes the fresh spectral stacked-LSTM reference and all six
raw/fusion conditions. It executes jobs in separate serial processes, verifies
dependencies, preserves every failed attempt, and skips verified completed jobs.
Study definitions freeze configuration, source/package hashes and participant
scope. Scheduler retries start a fresh job; standalone SSL recovery remains
available separately. Source-validation result tables and paired differences
require matching normalization and validation-target hashes. Source-test/ISRUC
predictive scoring is excluded from the scheduler.

At the experiment-runner milestone, the suite passed **79 tests**, including 11 new budget/runner
checks. A real-data CLI study completed all 18 jobs (14 supervised models across
the two pilot budgets), continued after a four-job boundary, and subsequently
verified/skipped every completed job. All fourteen saved prediction exports
replayed within 3e-8 in a fresh process. The two-participant pilot uses smaller
models and capped one-epoch schedules; its nominal 25% fraction selects one
participant, so its actual fraction is 50%. Canonical full-source planning
separately validated the 18- and 70-participant budgets and the fixed 15-participant
validation scope without executing the full training campaign.

After that matrix pilot, final completion-manifest publication was made atomic.
The updated suite injects an interruption immediately before publication and
verifies successful retry with no partial final seal. A subsequent three-job CLI
study validated the hardened worker. Historical matrix and current code hashes
are recorded separately; only the worker's publication logic changed between
those CLI checks.

Evidence is recorded under `reports/experiments_2026-10-31`, including canonical
budget definitions, capped source-validation tables and the pilot summary.
Full-budget convergence, repeated-seed scientific
comparisons, GPU measurements and real external scoring remain separate work.

**Profiling and frozen-evaluation implementation — 31 October 2026**

Participant-level analysis pools both source nights before computing five-class
F1, resamples participants jointly across conditions/seeds, and reports seed
variation separately. Calibration uses fixed-bin ECE and multiclass Brier score.
One participant yields no confidence interval; conditions cannot gain fictitious
sample size from repeated seeds. Missing-class support and physical transition
subgroups remain explicit.

The profiler measures the saved architecture using source records only, with
disposable in-memory training updates and unchanged-file verification. It separates
model/load/normalization/feature timing, sampled host memory and optional TF GPU
allocator memory. Short CPU pilots do not establish GPU fit or full-campaign speed;
SSL-specific profiling remains separate.

Evaluation registries contain source-selected bundles. Freezing snapshots model,
normalization, selection evidence, code/packages and the declared cohort/analysis
policy without reading target stores. Confirmatory freezes reject limited pilot
models. Scoring checks complete enrollment, provenance, preprocessing and identical
target masks across the entire family. Models run in isolated processes, and a
completed report is sealed only after all succeed. Real ISRUC/source-test scores
remain unopened in this milestone; synthetic external stores and capped source
validation exercise the scoring path.

The full current suite passes **95 tests**, including 16 evaluation/profiling
checks. Three fresh-process CPU profiles exercised the default 82,757-parameter
gated model: inference at batch four and disposable training at batches one/four.
Saved artifacts stayed unchanged. Sampled host RSS stayed below 530 MiB in these
limited runs; this measures host memory, not GPU fit. Five timed steps per phase
are timing checks, not a stable campaign benchmark, and inference/training use
different execution modes.

A source-only registry, freeze and isolated-worker scoring CLI completed M0/M3/M6
on eight predeclared validation targets each from SC4011/SC4012. Analysis counted
one independent participant and returned no confidence interval. Synthetic
external fixtures separately exercised paired scoring and provenance/enrollment
failures. No real external freeze or real ISRUC/source-test scoring was performed.

Evidence is recorded in `reports/evaluation_2026-10-31`, including CPU profiles,
source-validation coverage/analysis, participant tables and the evidence manifest.
Full-budget convergence, repeated-seed scientific
comparisons, GPU/SSL profiling, secondary scorer agreement and real external
scoring remain outstanding research/validation work.

**Full-source execution and compute workflow — 1 November 2026**

The suite now passes **103 tests**, including eight checks for source-only capsule
integrity and actual synthetic supervised/contrastive training kernels. Required
GPU preflight fails closed on native Windows; CPU-allowed preflight completes the
default gated batch-four and SSL batch-32 workloads.

The bounded full-source study completed all five jobs using 70 training and 15
validation participants (138/29 recordings). Waveform statistics use all 323,423
quality-valid source training epochs. Two short SSL epochs and two short epochs
for each gated scratch/fine-tuned model exercise the default architectures,
checked transfer, source validation, saving and reloading. Each epoch is capped
at 128 training/pretraining batches and 64 validation batches. Both models export
the same 256 validation targets across all 15 validation participants and share
normalization. M6 did not outperform M3 in this limited run; the capped schedules
do not establish an SSL benefit, harm, convergence, or confirmatory performance.

A separate 50-step CPU training profile of the saved gated model measured about
110 examples/second including cached loading/normalization and sampled peak host
RSS below 535 MiB. This one-record CPU profile excludes preprocessing/integrity
startup and does not measure GPU memory or full-campaign throughput.

The source-only capsule contains 573 allowlisted files, including all 167 source
training/validation stores. Its 4,891,279,360-byte archive passed a complete
extraction, per-file verification and source-identity check. Test epoch data,
ISRUC, raw recordings, checkpoints and hidden credentials are excluded. The
frozen split retains test identities. This is a local archive; no upload or paid
compute was initiated. The archive reflects its creation snapshot; later
documentation/evidence changes require a new capsule before transfer.

Hardware inspection identified a GTX 1050 with 4 GB VRAM and Pascal capability
6.1. Existing Ubuntu 24.04/WSL2 exposes that GPU. Project-local Linux setup uses
pinned TensorFlow/CUDA/cuDNN requirements, but its 2.52 GB GPU-wheel download is
still in progress at the tracked evidence snapshot. No passed GPU check is
claimed. An unattended continuation records dependency/import checks, required-GPU
training preflight and, if successful, a 50-step source-only GPU profile under
`runs/compute_2026-10-31`; it starts no full campaign. Inspect
`gpu_setup_status_2026-11-01.json` there for the live result. System packages,
drivers and cloud accounts were not changed.

Evidence is in `reports/compute_2026-11-01`; operational instructions are in
`COMPUTE_GUIDE.md`. All ten preserved historical artifact hashes remain unchanged.
No new source-test or real ISRUC predictive scores were opened. Full-budget
convergence, repeated seeds and confirmatory external evaluation remain ahead.

Evaluating the frozen feature model from the newly prepared source data reproduced
accuracy **0.8266296611**, macro-F1 **0.6533745630**, and every entry in the saved
confusion matrix. These checks validate the data migration and baseline; no
predictive model performance has been measured on ISRUC.

ISRUC workbook 29 has `hich` instead of `Epoch` in the first header cell. Its
first-row `Stage` header identifies it as a header; the adapter records this
metadata and still requires complete, unique numbered epoch coverage. Numeric
rows with invalid stage labels are rejected. This behavior has a regression test.

The source-only study uses scorer-1 workbooks as the declared initial annotation
policy. TXT/Excel disagreements, reference-electrode differences, and scorer-2
sensitivity analyses remain research-protocol considerations. Preparing ISRUC
does not establish external model performance or resolve every annotation
provenance question.

**User-run training handoff — 1 November 2026**

The user now owns research training, pretraining, fine-tuning and optimizer-based
hardware checks. The previously queued automatic GPU preflight/profile was
stopped. The current background helper installs dependencies only, then waits
for user-run GPU verification; GPU readiness remains unverified.

`TRAINING_RUNBOOK.md` contains exact PowerShell commands for the WSL preflight,
capped GPU benchmark, disposable source profile, first full source matrix and
job recovery. `scripts/research.ps1` selects the existing project-local Linux
Python and redirects framework caches into the workspace. Its real argument
forwarding check preserved a string containing spaces and selected the expected
Linux executable without running any model.

`execute --require-gpu` checks TensorFlow GPU visibility before reading a study
or scheduling workers. This guard does not replace kernel preflight. Existing
CPU callers retain their prior behavior when the flag is omitted. Capsules now
include the launcher, runbook and training-ownership instructions.

Eleven targeted tests passed: six mocked GPU/CLI handoff checks and five capsule
checks, with no model-weight updates. The earlier 103-test full suite is
historical evidence; the optimizer-based suite was not rerun under the user's
new training-ownership policy. A full regression command is provided below for
the user to run if desired:

```powershell
.venv\Scripts\python.exe -W ignore::DeprecationWarning -X utf8 -m unittest discover -s tests -v
```

Fresh studies are required after these source changes. No research training,
synthetic optimizer preflight, training-mode profile or real ISRUC/source-test
scoring was started in this handoff. Evidence is in
`reports/training_handoff_2026-11-01`.

**Delegated source training execution — 1 November 2026**

The user subsequently delegated training execution to the assistant. The current
authorization includes GPU preflight, profiling, source-only pretraining,
fine-tuning and the source matrix. Manual commands remain available for
reproduction; the earlier user-only handoff is historical.

Each worker now rechecks frozen code and package versions immediately before its
job, in addition to the scheduler's startup check. This prevents an active study
from silently mixing implementations after an edit during a long campaign.
The full regression suite passes **110 tests** after this change.

The staged execution pipeline records live state under
`runs/training_execution_2026-11-01_gpu/status.json`. It waits for dependency setup,
checks the tested source hashes, then requires a passed GPU kernel preflight,
five completed bounded benchmark jobs and a successful GPU source-training
profile before starting the 18-job uncapped source matrix. A failed stage stops
execution and preserves its logs. Only source training/validation data are used;
no real source-test or ISRUC prediction is scheduled. Recorded execution evidence
is under `reports/training_execution_2026-11-01`; live status determines which
stages have actually completed.

**Native CPU training and D: runtime storage - 2 November 2026**

The user's C: storage concern supersedes the WSL/GPU route for the current
campaign. That pipeline is stopped: two preparation jobs completed, SSL was
interrupted, and the full GPU matrix never started. No project jobs remained
in WSL and the optional RAM cache was absent when checked. Historical attempts
and original data remain intact.

`scripts/research_cpu.ps1` uses the existing D: Windows environment, disables
CUDA and bytecode writes, and routes process/worker temporary files and framework
caches to `runs/cpu_runtime`. The real temporary-directory and worker-inheritance
probe passes, as does dependency consistency. No packages or source modules
were changed for this switch; the prior 110-test regression snapshot still
matches the source. Windows system/paging writes to C: remain possible.

The CPU campaign uses a fresh study under
`runs/training_execution_2026-11-02_cpu/source_matrix` and the original uncapped
batch-4/SSL-32 matrix. Source-only scope, frozen worker checks, and all 18 planned
jobs are retained. Track actual progress in its ledger and logs; startup does
not establish completion, convergence or an SSL benefit. ISRUC and new real
source-test predictive scoring remain unopened. Evidence for runtime routing,
CPU optimizer checks and the transition is in `reports/cpu_training_2026-11-02`.

The CPU continuation has three sealed preparation jobs completed and SSL
pretraining active. `scripts/training_status.py` checks the actual scheduler
identity, samples its process tree, checks frozen source/package versions and
optionally verifies completed artifact hashes. It imports no TensorFlow and
leaves the study unchanged. Reproducible commands are in `TRAINING_RUNBOOK.md`;
dated progress snapshots and monitor checks are retained with the CPU evidence.
Initial activity samples precede the first SSL epoch seal, so no epoch timing,
convergence or comparative performance claim is available yet.

The first full SSL epoch subsequently completed. Its saved state passed all
seven declared file hashes, epoch/optimizer-counter readback (1 epoch and
10,106 updates), and finiteness checks on 57 numeric tensors. This is checkpoint
integrity evidence; no model was restored by this check and no sleep-stage
benefit is established. Training continues for the declared ten SSL epochs
before the 14 supervised jobs. The first checkpoint interval was 15.9 minutes,
including startup; subsequent epoch timing and total matrix duration remain
unmeasured. Evidence is in `ssl_epoch01_verification.json` alongside the CPU
progress snapshots. The frozen source methods and exact budgets are recorded
in `reports/source_campaign_2026-11-02/METHODS.md` and `methods.json` without
opening target predictions or changing source/environment/configuration.

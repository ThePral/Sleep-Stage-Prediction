# Project working instructions

## Training ownership

13 November engineering update: opt-in supervised epoch recovery and compiled
validation are implemented. Non-optimizer checks passed. The user-run fresh-process
synthetic optimizer gate also passed: state and predictions match exactly for
the tested tiny gated-BiLSTM CPU case. The user also passed all 136 full-regression
tests on 15 November (147.021 seconds); the retained log and source/package
provenance were verified. The 15 November real-data smoke check passed 13/14
models; M3 at n70 failed the probability tolerance at compiled batch 32 despite
unchanged classes/metrics on 64 targets. Matched-batch diagnostics then failed
compiled batches 4/8/16/32 and eager batches 16/32; eager 4/8 passed the sample.
M3/n70 then passed full-cohort eager batch-4 parity on all 65,133 targets
(maximum probability difference 1.11e-16, zero class changes). Keep compiled
validation unaccepted. The remaining thirteen also passed on 17 November;
the 20 November review verified all 14 model seals, source/script hashes and
65,133 target identities per model, with maximum probability difference
1.11e-16 and zero class changes. Full-family eager batch-4 inference parity is
accepted for these saved models. The user then completed all 18 jobs in
`runs/bilstm_recovery_eager_pilot_2026-11-20`; the review verified all job seals
and 42 recovery generations (epochs 0/1/2 for each of 14 supervised models).
The user then replayed M6 from epoch one into
`runs/bilstm_real_resume_m6_2026-11-20`: all 161 state arrays, RNG/callback state,
history, best-model hash and eight validation predictions match the uninterrupted
pilot exactly. This passes the tested capped real-data M6 CPU recovery case,
not full-schedule or cross-hardware recovery. The user also completed the
64-training-batch M6 runtime benchmark: 59.47 minutes for the training call,
including two full validation passes/export, with 664.42 MiB peak sampled host
RSS. Its bundle, recovery counters and full target identities verify. It does
not measure an uncapped epoch, SSL, GPU VRAM or a speedup. The user selected
the current CPU route. A fresh three-seed replication study (17/42/123, 48 jobs)
is frozen at `runs/bilstm_replication_cpu_2026-11-20`. The user completed the
first six jobs. The 22 November review verified all preparation/normalizer seals
and SSL seeds 17/42/123: ten epochs and 101,060 recorded optimizer steps each,
matching source-only 70-participant pool, bundle/checkpoint hashes and zero
reported encoder reload difference. The user then completed all seven M0–M6
fits at draw42_n18/seed17. The 24 November review verified 13 completed-job seals,
seven model bundles whose validation exports share 65,133 target identities, 69 supervised
epochs and 76 sealed recovery generations (42.48 recorded supervised hours).
The 27 November review also verified M0–M6 at draw42_n18/seed42: twenty completed
job seals, 52 additional epochs, 59 recovery generations and matching validation
target identities (31.62 recorded hours). The ledger additionally records an
interrupted M0_draw42_n18_seed123 attempt with KeyboardInterrupt; its valid
epoch-zero checkpoint has zero completed epochs/optimizer steps, so epoch one
must be replayed. Artifact status still says running but no scheduler lock or
visible experiment process was found. Preserve that attempt and all completed jobs.
The 28 November outage inspection supersedes that restart status: twenty-one
completed-job seals verify, including M0/seed123 attempt 0003 (eleven epochs,
best epoch six). M1/seed123 attempt 0001 retains six sealed epochs, 113,532
optimizer steps and best epoch six; epoch seven must be replayed. The scheduler
lock PID 94504 is dead and no visible experiment process was found. All source,
packages, data and recorded runtime checks match; no lock was changed or training
launched by the inspector.
The initial -File restart hit a confirmed guard false positive on its own
PowerShell parent. Read-only probes reproduced it and verified that invoking
the same CPU launcher via powershell -Command passes all guard checks. Lock
rename was mocked/prevented in probes; no training or lock mutation occurred.
Use the corrected -Command restart in docs/operations/REPLICATION_CPU.md;
do not edit frozen src or bypass the actual process/lock safety checks.
Source/packages and original participant budgets match; protocol/recipe hashes
are archived with the study. The next user-run restart is exactly six supervised
jobs, M1–M6 at draw42_n18/seed123, with --retry-failed --resume-supervised
--recover-stale-lock to archive the dead lock and restore M1 after epoch six,
stopping before budget 70 and reporting the expected
27/48 summary for inspection. Do not advance stages
automatically. SSL scheduler epoch resume is not implemented by the supervised
resume flag. The historical seed-42 study is not pooled as a new-version seed.
See `docs/operations/REPLICATION_CPU.md`, `docs/REPLICATION_PROTOCOL_CPU_2026-11-20.md`
and `reports/recovery_guard_2026-11-28/`. Do not generalize
the synthetic recovery result to untested configurations or claim measured speedup. The
completed 18-job seed-42 study remains preserved historical evidence; edited
source requires new study directories. See `docs/operations/SUPERVISED_RECOVERY.md`.

Current ownership (2 November 2026): the user now wants to launch training in
their own terminal using assistant-provided commands so progress is visible.
This supersedes all earlier delegation below. Do not start research training,
pretraining, fine-tuning, optimizer kernel checks, training-mode profiles, or
optimizer-based regression suites without a new explicit request. Do not launch
background training or automatically advance staged controllers. The assistant
may implement/review code, inspect artifacts, plan fresh studies and run checks
that perform no optimizer updates. Provide `execute --live` commands through
`scripts/research_cpu.ps1`, retain logs, and keep project-managed files on D:.

The automatic BiLSTM controller has been stopped for terminal handoff after
its benchmark/profile completed. Preserve its completed pilot and benchmark;
the full matrix was interrupted during initial waveform-statistics preparation,
before full-schedule optimizer training. Logging implementation changes require
a fresh frozen study. See `reports/bilstm_terminal_handoff_2026-11-02/`.

The earlier execution authorizations in this history no longer assign current
training ownership to the assistant.

On 1 November 2026 the user superseded their earlier request to run training
personally and explicitly delegated execution to the assistant ("I've changed
my mind, do it yourself please."). The assistant may run dependency setup, GPU
kernel preflight, profiling, tests, source-only pretraining, fine-tuning and the
source experiment matrix. Continue within this authorization without requesting
the same permission again. Retain the manual launcher/runbook for reproducibility.

On 2 November 2026 the user requested CPU training to avoid filling C:. Use
native Windows `scripts/research_cpu.ps1` with the existing environment on D:;
keep project-managed temporary files, caches, checkpoints and logs on D:.
Do not restart the WSL/GPU pipeline or create a RAM cache for this campaign.
Windows paging/system writes are outside this guarantee. No system storage,
pagefile, WSL image or C: cleanup changes are authorized by this preference.

The user then explicitly stopped the previous CPU campaign during workspace
organization on 2 November 2026. Preserve its latest sealed SSL checkpoint
(three completed epochs); no supervised jobs started. Do not restart that GRU
matrix automatically. The current paper direction is Sleep-EDF Expanded with
SSL, cross-attention and a planned BiLSTM head. Cross-dataset work is deferred,
not deleted. Organization is not authorization to claim BiLSTM implementation
or start an unvalidated replacement campaign.

The user subsequently requested implementing SSL, cross-attention and BiLSTM.
New studies may run after the BiLSTM regression/kernel/integration gates pass;
use the `experiment_bilstm_*.json` recipes and fresh directories. Preserve the
stopped GRU campaign and never bypass its code/resume seals to reuse partial
training. Old recipes/bundles default to the explicit `legacy` temporal mode.

Current guides are under `docs/operations/`, environment specifications under
`environments/`, and source configurations under `configs/sleepedf/`. See
`docs/PROJECT_STRUCTURE.md` for pinned artifact paths and the migration map.

## Research protocol

Keep real ISRUC predictive scoring unopened until the full model family and
evaluation policy have been source-selected and frozen. Use fresh study
directories after implementation or package changes; preserve historical runs
and their evidence. Never describe capped integration results as convergence or
confirmatory performance.
Inspect active run status before editing source. Keep `src/` and the environment
unchanged during a frozen campaign; each worker rejects changed code/packages.

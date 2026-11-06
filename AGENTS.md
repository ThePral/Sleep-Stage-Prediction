# Project working instructions

## Training ownership

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

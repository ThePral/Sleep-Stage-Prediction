# Project working instructions

## Training ownership

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

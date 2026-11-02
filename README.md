# Sleep-EDF EEG Sleep Staging

[![XERJ Corpus Hub contributor](https://hub.xerj.org/contributors/ThePral/badge.svg)](https://hub.xerj.org/contributors/ThePral)

The current paper direction is **Sleep-EDF Expanded**, combining self-supervised
EEG pretraining, raw/spectral cross-attention and a planned BiLSTM temporal head.
The existing research models use a GRU head, with a stacked-LSTM spectral
reference; the BiLSTM replacement is the next implementation step.

Cross-dataset work is deferred and preserved for later. Its ISRUC adapter,
prepared data, evaluation code, audit tools and configuration remain available.
Real ISRUC predictive scoring remains unopened.

## Start here

- [Project structure and path rules](docs/PROJECT_STRUCTURE.md)
- [Current research direction and deferred work](docs/RESEARCH_DIRECTION.md)
- [Documentation index](docs/README.md)
- [Implementation status](docs/IMPLEMENTATION_STATUS.md)
- [CPU training and recovery runbook](docs/operations/TRAINING_RUNBOOK.md)
- [Compute and environment guide](docs/operations/COMPUTE_GUIDE.md)
- [Existing pipeline, API examples and historical baseline details](docs/reference/IMPLEMENTED_PIPELINE.md)

## Repository layout

```text
src/                           Shared data, models, training, inference and evaluation
configs/
  sleepedf/models/             Existing supervised and linear-probe configurations
  sleepedf/pretraining/        Source-only SSL configuration
  sleepedf/experiments/        Existing source experiment matrices and pilots
  sleepedf/evaluation/         Source-validation integration specification
  compute/gpu/                 Retained GPU benchmark/matrix recipes
  deferred/cross_dataset/      ISRUC evaluation specification for later
docs/                          Direction, structure, operations and reference material
environments/                  Pinned dependencies, platform lock and CUDA constraints
scripts/                       CPU/WSL launchers and read-only training monitor
  audits/cross_dataset/        Preserved dataset-compatibility audit tools
notebooks/legacy/              Earlier baseline and paper-plot notebooks
data/                          Raw, prepared and feature stores (large data are ignored)
runs/                          Local studies, checkpoints, logs and runtime caches (ignored)
reports/                       Versioned research evidence and progress snapshots
```

`experiments/`, `final_model/`, `final_model_corrected/` and `results/` retain
historical artifacts at their original paths. Frozen studies and baseline
evidence refer to these paths; see the project structure guide before moving
them. The participant split remains
`final_model_corrected/participant_split.csv`.

## Execution and reproducibility

Run commands from the checkout root. Use the existing native Windows environment
on D: through `scripts/research_cpu.ps1`; it routes project-managed temporary
files and caches to D:. Windows system/paging activity can still use C:.
Dependency specifications are in [environments/](environments/README.md).
The installed environments have not been moved or reinstalled.

The previous CPU matrix was **stopped at the user's request** before any
supervised jobs started. Three complete SSL epochs and their checkpoint are
preserved. The interruption is recorded in
[training_stop.json](reports/organization_2026-11-02/training_stop.json).
No BiLSTM campaign is running. Do not automatically restart the older GRU matrix.

The [workspace migration report](reports/organization_2026-11-02/README.md)
records moved paths and validation. Old reports and study manifests keep their
original path strings as historical evidence; current commands use the new
configuration locations. Start a fresh study after source/package changes.

The earlier notebooks preserve their outputs but can write to legacy artifact
directories when executed. Their [index](notebooks/README.md) distinguishes them
from the maintained Python pipeline. The source-only training capsule is an
explicit subset of the repository; it excludes test/ISRUC epoch payloads,
historical checkpoints and notebooks.

# Project structure and path rules

The workspace separates maintained modules, runnable configurations, operational
tools, historical notebooks and research evidence. File placement does not
change experimental methodology or select new models.

| Location | Contents and role |
|---|---|
| `src/data/` | Epoch stores, participant budgets, streaming loaders and dataset adapters |
| `src/features/`, `src/preprocessing/` | Feature extraction and source-fitted normalization |
| `src/models/`, `src/training/`, `src/inference/` | Maintained model, training and prediction implementations |
| `src/experiments/` | Frozen source-only study planner and workers |
| `src/evaluation/` | Shared classification/statistics/profiling and frozen evaluation; source and external protocols both use it |
| `configs/sleepedf/` | Source model, SSL, experiment and validation configurations |
| `configs/compute/gpu/` | Earlier GPU recipes, retained for reproducibility |
| `configs/deferred/cross_dataset/` | Deferred external-cohort specification |
| `scripts/` | Launchers, monitor and operational helpers |
| `scripts/audits/cross_dataset/` | Versioned standalone compatibility audits; run from the checkout root |
| `environments/` | Dependency specifications; installed environments remain in their existing locations |
| `notebooks/legacy/baseline/` | Earlier numbered exploration, training, evaluation and inference notebooks |
| `notebooks/legacy/paper/` | Earlier plotting notebooks |
| `reports/` | Immutable dated evidence plus a navigation index |
| `runs/` | Local runtime state, temporary caches, studies, checkpoints and logs |

The `src` hierarchy already follows module responsibilities, so import paths
such as `src.models.multiview` and `src.data.adapters.isruc` remain stable.
Shared evaluation modules are retained in place because source-only validation
also depends on them. Deferring ISRUC does not require deleting those modules.

## Paths deliberately kept stable

Prepared and raw datasets, the installed `.venv`, local studies, saved model
directories, historical results and dated report contents stay at their original
paths. In particular:

- `final_model_corrected/participant_split.csv` is the frozen participant split.
- `final_model/` and `final_model_corrected/` contain baseline artifacts used by inference and integrity checks.
- `experiments/` and `results/` contain earlier model-selection/evaluation artifacts.
- `data/processed/research_v1/` contains the shared prepared stores for Sleep-EDF and ISRUC.
- `runs/training_execution_2026-11-02_cpu/source_matrix/` preserves the stopped study and its SSL checkpoint.

Moving those files would invalidate absolute references, hashes or historical
reproduction. A future migration would need a separate data/artifact manifest;
adding folder indexes is sufficient for this organizational pass.

## New work

Add the BiLSTM implementation under the existing model/training modules and
give it explicit Sleep-EDF configurations. Do not rename an existing GRU recipe
to imply that it is BiLSTM. Use fresh study directories and keep older results
separate. New figures and tables should be exported to a named run/results
directory rather than the source tree or raw data folders.

Run CLI commands from the checkout root. Moved notebooks discover that root by
walking their working-directory ancestors, so their depth does not determine
imports or dataset paths. Outputs and execution counts remain historical;
no notebook was re-executed during migration.

Current commands use the new configuration paths. Old reports retain original
strings, with the translation recorded in
[`relocations.json`](../reports/organization_2026-11-02/relocations.json).
Capsule validation accepts both old root-level document/dependency members and
the organized paths so earlier archives can still be verified and unpacked.

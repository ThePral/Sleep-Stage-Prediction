# Research evidence index

Dated report folders retain their original paths and contents. Path strings and
source/configuration hashes in older reports describe the project at that time;
they are not rewritten during file organization.

| Evidence | Location |
|---|---|
| Dataset-scope startup diagnosis and read-only monitor verification | `scope_diagnosis_2026-11-06/` |
| User-owned terminal handoff, background stop and non-optimizer live-progress checks | `bilstm_terminal_handoff_2026-11-02/` |
| BiLSTM regression/kernel checks, architecture counts and staged source execution | `bilstm_2026-11-02/` |
| Workspace organization, path map and explicit training stop | `organization_2026-11-02/` |
| Stopped CPU campaign, preflight, checkpoint and progress records | `cpu_training_2026-11-02/` |
| Methods for the earlier GRU source campaign | `source_campaign_2026-11-02/` |
| Earlier GPU execution/setup and handoff | `training_execution_2026-11-01/`, `training_handoff_2026-11-01/`, `compute_2026-11-01/` |
| Evaluator/uncertainty/profiling integration | `evaluation_2026-10-31/` |
| Source experiment matrix and SSL integration pilots | `experiments_2026-10-31/`, `contrastive_2026-10-31/` |
| Supervised integration and canonical data preparation | `supervised_2026-10-27/`, `implementation_2026-10-27/` |
| Original project audit | `audit_2026-10-26/` |

New BiLSTM execution evidence is separate from earlier GRU artifacts. Capped pilots remain integration checks rather
than convergence evidence. The stopped SSL checkpoint has three completed
epochs and is not a completed ten-epoch encoder-transfer bundle. Raw run state
and checkpoints remain under `runs/`; these reports are selected evidence.

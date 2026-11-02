# Sleep-EDF SSL/cross-attention/BiLSTM implementation - 2 November 2026

The new temporal head has 32 LSTM units per direction, concatenated into 64
features. All four representation controls use it. `architectures.json`
records actual constructed model shapes and parameter counts: spectral 11,333;
raw 56,325; concatenation 65,861; gated cross-attention 82,629. SSL transfers
only the waveform encoder; the fusion and temporal components learn during
supervised source-only fine-tuning.

`regression_tests.txt` and `regression_summary.json` record **116 passed tests**
in 123.443 seconds, with exact source/test hashes. New checks cover gradients
through both LSTM directions, serialization for all four views, tiny multiclass
overfit, SSL transfer/fine-tuning/export, explicit temporal-head mismatch
rejection and the separate linear-probe mode. Existing data-leakage, sampler,
checkpoint-resume, baseline inference, statistics, capsule and frozen-worker
checks also pass.

`cpu_preflight.json` passes supervised batch four and SSL batch 32 on native
Windows CPU. It uses the actual BiLSTM architecture. Five timed synthetic steps
follow two warm-up steps. It ran alongside regressions; timing is not a clean
performance benchmark or an estimate for full data loading/convergence.

The staged controller is `runs/bilstm_2026-11-02/orchestrate.py`; live state is
its sibling `status.json`. It requires the passing source-hash snapshot and
CPU kernel check before a real-source, seven-condition, two-budget pilot.
That pilot uses two training participants, one validation participant, full
architecture, one epoch and two batches per stage. All 18 jobs must seal
successfully. A full-70/15-participant capped M3/M6 benchmark and disposable
training profile must then pass before the uncapped source matrix starts.
Failures preserve evidence and stop later stages.

The full matrix is a fresh one-seed campaign with 18/70 labeled participants,
the fixed 70-person unlabeled pool, 15 validation participants, ten SSL epochs
and up to 30 supervised epochs with patience five. It trains a new spectral
BiLSTM reference and all raw/fusion scratch/SSL controls. It schedules no real
ISRUC or new source-test predictions. Repeated seeds and scientific conclusions
remain subsequent work.

The earlier stopped GRU study and three-epoch checkpoint remain unchanged.
The new study starts SSL afresh rather than bypassing strict code/resume seals.
Installed packages and prepared datasets are unchanged. Source/package hashes
are frozen for every new worker. Do not edit `src/` or update packages during
the staged campaign. Current manual commands are in `docs/SLEEPEDF_BILSTM.md`
and `docs/operations/TRAINING_RUNBOOK.md`.

The real-source pilot has completed all 18 sealed jobs. `pilot_validation.json`
verifies all 14 classifier bundles, exact BiLSTM settings, checkpoint reload
tolerance, one shared SSL initialization bundle, matched normalization within
each budget and identical validation target IDs. Each model has just four
validation targets. The pilot tables are retained as
`pilot_source_validation_results.csv` and `pilot_paired_source_validation.csv`;
they are execution evidence, not a basis for selecting a winning architecture.
All ten preserved baseline artifact hashes and all seven files in the earlier
three-epoch SSL checkpoint remain unchanged. Documentation links also resolve.
The full-source benchmark is the next staged gate; consult live status rather
than interpreting pilot completion as completion of the research campaign.

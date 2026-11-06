# Research direction - 2 November 2026

The supervisor-recommended paper direction is Sleep-EDF Expanded with
self-supervised pretraining, waveform/spectral cross-attention and a BiLSTM
temporal classifier. External-dataset evaluation is deferred for a later phase.

Training ownership now belongs to the user, who launches commands in a visible
terminal. The assistant provides commands, implementation and read-only review;
it does not start background training. The ready `execute --live` command and
fresh planned study are described in [the runbook](operations/TRAINING_RUNBOOK.md).

The maintained implementation now supports BiLSTM heads for the spectral, raw,
concatenation and gated cross-attention controls. The initial recipe uses 32
units per direction. Source-only waveform SSL transfers into the new BiLSTM
classifiers. Earlier GRU/stacked-LSTM configurations and the legacy notebooks
remain historical comparisons. See [SLEEPEDF_BILSTM.md](SLEEPEDF_BILSTM.md) for
the implemented data flow, configuration paths and staged validation policy.

## Immediate sequence

1. Validate BiLSTM gradient flow, serialization, waveform SSL transfer and inference.
2. Run the full seven-condition integration pilot on source participants only.
3. Check the full 70/15-participant scope with a bounded M3/M6 source benchmark.
4. Freeze and execute a fresh uncapped source matrix after checks pass.
5. Review complete learning curves and source-only comparisons before freezing any additional seeds or scoring held-out data.

The previous CPU GRU matrix was stopped explicitly by the user during
organization. Three complete SSL epochs are retained with model and optimizer
state; the final ten-epoch transfer bundle was not produced. No supervised jobs
had started. Do not silently reinterpret that partial run as the new BiLSTM
experiment or choose its stopping epoch using downstream results.

## Deferred cross-dataset phase

The ISRUC data and adapter, frozen-evaluation engine, earlier audits, source-only
capsule support and external specification are preserved. The specification now
lives at `configs/deferred/cross_dataset/evaluation_isruc.json`.
The earlier audit and roadmap are in `docs/archive/2026-10/`.

Keep real ISRUC predictive performance unopened while the model family and
selection procedure are changing. When returning to that strategy, review the
preserved annotation/channel provenance, finalize source-selected models and
the complete external policy, then freeze before scoring. Deferral does not
invalidate the prepared data or authorize target-based tuning.

This file records scope and sequencing, not positive results, novelty claims
or a final publication protocol. The initial architecture and validation
configuration are explicit; convergence, repeated-seed evidence and scientific
conclusions require the subsequent experiment results.

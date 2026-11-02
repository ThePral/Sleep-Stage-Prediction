# Research direction - 2 November 2026

The supervisor-recommended paper direction is Sleep-EDF Expanded with
self-supervised pretraining, waveform/spectral cross-attention and a BiLSTM
temporal classifier. External-dataset evaluation is deferred for a later phase.

This organizational change does not implement or train the new BiLSTM variant.
The current `multiview` implementation already has raw/concatenation/gated
cross-attention variants and source-only contrastive pretraining. Its fusion
models use a 64-unit GRU; its spectral reference uses two 32-unit LSTMs.
The legacy exploration notebook also contains earlier BiLSTM experiments;
those do not constitute the new end-to-end SSL/cross-attention/BiLSTM system.

## Immediate sequence

1. Complete and validate file organization while preserving historical evidence.
2. Specify a BiLSTM temporal head and matched raw/concatenation/cross-attention controls in the maintained modules.
3. Keep participant-independent Sleep-EDF splits and source-fitted normalization; define context and evaluation eligibility explicitly.
4. Verify SSL transfer, checkpoint reload and inference for the BiLSTM variants in small integration runs.
5. Freeze a fresh source-only study and its label budgets, seeds and selection policy before full training.

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
or a final publication protocol. The first-version architecture and validation
decisions still need to be made explicitly in the next implementation stage.

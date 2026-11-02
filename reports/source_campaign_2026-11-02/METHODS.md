# Frozen source campaign methods - 2 November 2026

This record describes the running experiment. It contains no model performance results
and does not freeze or execute external evaluation. Counts come from the study plan
and three completed preparation jobs, whose artifact seals were verified.

Study fingerprint: `92a0c9a6993467f2ff6fa12a6688b8fac4d2469e83edfd400887de9eb502e4a0`. Training seed 42 and budget draw seed 42.
All 18 jobs are uncapped. The source comprises 70 training participants/138 recordings
and 15 validation participants/29 recordings; no target labels select models.

| Training budget | Participants | Recordings | Annotated hours | Eligible training windows | Spectral fit epochs |
|---|---:|---:|---:|---:|---:|
| draw42_n18 | 18 | 35 | 638.458 | 75,687 | 76,604 |
| draw42_n70 | 70 | 138 | 2681.200 | 319,226 | 321,688 |

The nominal 25% budget is actually 25.71% of training participants
after ceiling rounding and 23.81% of the full budget's annotated hours.
Validation uses the same 65,133 eligible windows and 547.917 annotated hours for every model.

| Target stage | Smaller-budget training | Full-budget training | Validation |
|---|---:|---:|---:|
| W | 46,404 | 202,168 | 41,060 |
| N1 | 4,598 | 16,984 | 3,336 |
| N2 | 14,461 | 61,539 | 14,041 |
| N3 | 4,249 | 14,982 | 2,016 |
| REM | 5,975 | 23,553 | 4,680 |

Windows contain ten contiguous eligible 30-second epochs (100 Hz, 3,000 samples per epoch),
ending at the predicted epoch. Context epochs require valid signal and mapped labels;
gaps, unusable epochs and unknown labels break supervised sequences. Full recording
coverage is retained rather than trimming using sleep-stage labels. Wake support is
therefore substantial; report macro-F1 and stage support alongside accuracy.

Raw waveforms and the ten spectral features are aligned to the same epoch grid.
Preprocessing uses the frozen 0.3-35 Hz offline filter and Welch length 1,024.
The ten features are absolute and relative power in delta (0.5-4 Hz), theta
(4-8 Hz), alpha (8-13 Hz) and beta (13-30 Hz), normalized spectral entropy,
and dominant frequency. Spectral summaries use the 0.5-30 Hz analysis range.
Waveforms remain in volts and absolute powers in volts squared before scaling.
The four absolute powers are log10 transformed; all ten features are standardized
using unique supervised epochs from that labeled training budget. Waveforms undergo
per-epoch mean removal followed by one amplitude scale fitted to the fixed unlabeled
70-participant pool (323,423 quality-valid epochs). The same scale is used by
scratch and SSL models; no validation/ISRUC statistics are fitted. These offline
preprocessing and eligibility choices do not establish online deployment performance.

| Condition | View | Initialization | Temporal head |
|---|---|---|---|
| M0 | spectral | scratch | two 32-unit LSTMs |
| M1 | raw | scratch | 64-unit GRU |
| M2 | concatenation | scratch | 64-unit GRU |
| M3 | gated cross-attention | scratch | 64-unit GRU |
| M4 | raw | SSL waveform encoder | 64-unit GRU |
| M5 | concatenation | SSL waveform encoder | 64-unit GRU |
| M6 | gated cross-attention | SSL waveform encoder | 64-unit GRU |

The raw, concatenation and gated variants use the same CNN architecture and 64-unit GRU.
Gated cross-attention uses waveform queries and ten value/feature-identity spectral
tokens. It is physiology-informed feature fusion; attention weights do not establish
physiological causality. M0 is a freshly trained spectral reference, separate from the
historical model. The architectures have different capacities, so retain parameter
counts and avoid calling their differences equal-parameter mechanism tests.

SSL uses 323,423 quality-valid source epochs without stage-based selection,
batch 32, temperature 0.2, and ten fixed epochs. Each pass has 10,106
batches (323,392 epochs used, 31 shuffled tail epochs dropped).
The complete schedule has 101,060 optimizer steps.
Two views use gain 0.8-1.2, noise at 0.02 of waveform RMS and a mask of at most one
second. Temporal neighbors within one epoch in a recording are excluded as negatives;
same-stage false negatives can remain. The maximum is 62 negatives per anchor.
Only waveform encoder weights transfer into M4-M6. Their heads/fusion layers are
newly initialized and the transferred encoders are fine-tuned. The last declared
SSL epoch is used; no held-out scores select the pretraining checkpoint.

Supervised models use batch four, Adam at 0.001 with clipnorm one, up to 30 epochs,
and patience five. Checkpoints maximize source-validation macro-F1. Validation loss
reduces the learning rate by 0.5 after two plateau epochs, down to 1e-6.

The existing comparison pairs are M4-M1, M5-M2 and M6-M3 (SSL within architecture),
plus M3-M2 and M6-M5 (gated fusion versus concatenation). Analyze each label budget
separately using identical participant/epoch targets. Preserve all contrasts and
per-stage results, including null/negative effects. Participant-cluster intervals
must not treat overlapping windows or participant-seed combinations as independent
people. The candidate external analysis uses 2,000 bootstrap replicates, 95%
intervals and seed 42. Multiple intervals are not automatically multiplicity-adjusted.
A single training seed cannot characterize initialization variability.

Before any real ISRUC scoring: finish and verify the source campaign; review complete
learning curves, source validation, representation diagnostics and reload parity;
record repeated-seed requirements and source-selected model/normalization choices;
specify every budget/seed/condition entering the external batch; then seal the full
family registry and evaluation policy. The current ISRUC configuration specifies
100 records/participants, C3 and scorer 1, with no target adaptation. Its presence
is not an executed freeze. No target performance may inform later selection.

Disclose the historical source-test inspection. Neither this methods export nor
the running campaign opens new source-test predictions or any real ISRUC scores.
The JSON companion retains exact budgets, class counts, configuration, versions,
fingerprints and input hashes for later manuscript tables. No positive-result,
convergence, novelty or journal-readiness claim is established by this record.

**Research implementation — started 27 October 2026.**

The one-month objective is the complete implementation and validation of the
multi-view, self-supervised, and source-only external-evaluation workflows.
This file tracks delivered code separately from research conclusions.

| Milestone | Status | Deliverable |
|---|---|---|
| Timeline and sequence-index repairs | Implemented; regression tests pass | Physical indices survive excluded annotations; arbitrary pandas indices no longer reorder features |
| Shared waveform/spectral epoch format | Implemented | Full epoch grids, stable IDs, quality/label masks, declared units, and no fitted normalization |
| Sleep-EDF adapter | Implemented | Corrected frozen participant split; PSG/hypnogram identity/time-origin checks |
| ISRUC-I adapter | Implemented | Canonical numeric recordings; fixed central derivation; explicit workbook epoch labels and scorer |
| Recording storage and preparation CLI | Implemented | Atomic publication, hashes, memory maps, verified resume, no conflicting-store overwrite |
| Full dataset regeneration | Complete and validated | 197 Sleep-EDF and 100 ISRUC-I recordings; all 457,577 source supervised feature rows match the preserved cache |
| Canonical feature-model inference | Implemented | Directory loader and early rejection of model/configuration shape or feature-order mismatches |
| Collection and baseline verification CLI | Passed on full collections | Identity/split/coverage checks and exact reproduction of the source confusion matrix on 69,539 windows |
| Waveform encoder, feature tokens, and fusion models | Implemented | Shared strided CNN, ten feature tokens, raw-only/concatenation/gated-attention controls, and a temporal GRU |
| Source-only normalization and streaming batches | Implemented | Unique training-epoch statistics, frozen split checks, bounded memory maps, reproducible window ordering |
| Supervised trainer and multiview inference | Implemented; source integration pilots passed | Validation macro-F1 checkpoints, hashed completed bundles, prediction exports, and checkpoint reload checks |
| Contrastive pretraining and fine-tuning | Implemented; CPU integration pilots passed | Quality-only source pool, stateless augmentations, masked NT-Xent, recoverable optimizer state, verified encoder transfer |
| Frozen-encoder linear probe | Implemented | Single-epoch mean pooling and affine classifier; CNN weights checked unchanged |
| Matched participant label budgets | Planned | Nested training-participant selections shared across model/initialization comparisons |
| Full experiment runner and external predictive evaluation | Planned | Repeated comparisons, label budgets, participant uncertainty, and a frozen external scoring batch |

**Validation and data notes**

The data-foundation milestone passed **36 automated tests**. Data regression checks are in
`tests/test_data_foundation.py`; model/scaler serialization and incompatible
bundle rejection checks are in `tests/test_inference_bundle.py`. They cover
annotation gaps, recording bounds, index-label independence, signal QC, resampling,
relative/absolute power behavior, workbook parsing, channel reconstruction,
atomic cache round trips, tampering, split conflicts, and window eligibility.
Run them with the command documented in the README.

Prepared stores are written under `data/processed/research_v1`. Logs and the final
collection-validation report belong under `reports/implementation_2026-10-27`.
Original signal files, feature caches, model files, and saved experiment results
are preserved. The notebooks already modified when work began remain historical
inputs rather than the extension's execution interface.

The complete collection validation found 461,870 physical Sleep-EDF epochs,
including 457,577 supervised-eligible epochs, and 90,187 physical ISRUC epochs,
including 89,995 scorer-1 supervised-eligible epochs. The source participant
split remains 70/15/15 with no participant overlap. All 100 ISRUC participants
remain external. Signal-valid unlabeled epochs are retained separately for
source-only pretraining eligibility.

**Supervised model milestone**

At the supervised milestone, the suite passed **56 tests**: the 36 data/bundle checks from the
first milestone and 20 model/training checks. The recorded source pilots and
replay checks are in `reports/supervised_2026-10-27/pilot_summary.json`.

The default raw, concatenation, and gated-attention models have 56,453, 65,989,
and 82,757 parameters respectively. All share the same waveform encoder and
temporal context. The gated model attends over ten spectral feature tokens;
diagnostic attention/gate tensors can be exposed through a Keras model.

All three variants completed two-epoch source-only integration pilots using
SC4001/SC4002 for training and SC4011 for validation, with two batches per
epoch/validation pass. Normalization was shared across variants and fitted on
5,479 unique source training epochs. Each best checkpoint reloaded with zero
probability difference in the sampled check. These deliberately capped runs
verify the implementation; they are not trained-model performance evidence.
No new source-test or ISRUC predictions were used for model selection.
Replaying the exported validation epoch IDs through the raw-input inference
wrapper agreed within 1e-6 for all three models in a fresh process.

`tests/test_multiview.py` adds checks for attention keys and gradients, batch
independence, reproducible initialization, backbone transfer, serialization,
tiny-problem overfitting, source-only fitting, capped validation sampling,
metric agreement, and complete training-to-inference bundle execution.
Reproduction commands are in the README. Full source training, GPU memory and
throughput measurements, full-budget SSL, and the subsequent frozen external experiment
remain distinct experimental/implementation steps.

**Contrastive implementation — 31 October 2026**

The source-only raw SSL path uses individual signal-valid epochs, including
unscored epochs. Store integrity validation can inspect annotations, but the pool
retains neither stage labels nor spectral features. Hiding every annotation in
the test fixture preserves epoch eligibility, waveform statistics and seeded
batch order. Held-out source participants and ISRUC cannot enter the pool.

Two independently augmented views feed the shared CNN and a discarded projection
head. NT-Xent retains the positive in the denominator, excludes self-similarity,
and masks nearby same-recording negatives. Mild gain/noise and a short contiguous
mask are explicit initial assumptions; their physiological suitability and
same-stage false negatives remain research questions. The default pretraining
model has 35,328 parameters, of which 27,008 belong to the transferable CNN.

An atomic epoch-boundary state records encoder/projector/optimizer tensors and
history. Sampler/view randomness is stateless and reproducible from the seed,
epoch and batch. Resume requires unchanged source content, update settings,
package versions and relevant code, and writes a new run directory. The test
suite compares recovery after a simulated interruption with uninterrupted
updates. Completed bundles verify file hashes and reloaded encoder outputs.

Downstream transfer validates architecture, source identities and preprocessing.
SSL and downstream currently require the same source recording selection. The
SSL waveform scale is preserved, while the spectral scaler retains its separate
supervised source fitting population. Scratch controls can reuse that exact
waveform scale without loading pretrained weights. Linear probes freeze the CNN
and fit only a 325-parameter affine head on a single epoch; frozen temporal/fusion
head training is also supported and is explicitly distinguished from probing.

The full current suite passes **68 tests**, including 12 contrastive/data/transfer
checks. CPU pilots completed raw pretraining, epoch-boundary resume, gated
fine-tuning, a matched scratch control and a frozen linear probe. The complete
70-participant source-training pool contains 323,423 quality-valid epochs. Its
waveform statistics were fitted on that full pool, and a deliberately capped
two-batch optimization pass completed successfully. The smaller transfer pilot
fitted waveform statistics on 5,480 source epochs and spectral statistics on
5,479 supervised source epochs from SC4001/SC4002, validating on SC4011.

Implementation pilots, test totals and limitations are recorded in
`reports/contrastive_2026-10-31/pilot_summary.json`. These runs verify execution
and source boundaries; they do not demonstrate convergence, label efficiency,
physiological invariance, or external generalization. No new test/ISRUC predictive
scores are used. Participant label budgets, the experiment runner, uncertainty
estimation and the frozen external scoring batch remain upcoming work.

Evaluating the frozen feature model from the newly prepared source data reproduced
accuracy **0.8266296611**, macro-F1 **0.6533745630**, and every entry in the saved
confusion matrix. These checks validate the data migration and baseline; no
predictive model performance has been measured on ISRUC.

ISRUC workbook 29 has `hich` instead of `Epoch` in the first header cell. Its
first-row `Stage` header identifies it as a header; the adapter records this
metadata and still requires complete, unique numbered epoch coverage. Numeric
rows with invalid stage labels are rejected. This behavior has a regression test.

The source-only study uses scorer-1 workbooks as the declared initial annotation
policy. TXT/Excel disagreements, reference-electrode differences, and scorer-2
sensitivity analyses remain research-protocol considerations. Preparing ISRUC
does not establish external model performance or resolve every annotation
provenance question.

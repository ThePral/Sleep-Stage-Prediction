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
| Contrastive pretraining and fine-tuning | Next milestone | Source-training-only pool, reproducible augmentations and checkpoint transfer |
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

The full current suite passes **56 tests**: the 36 data/bundle checks from the
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
throughput measurements, SSL, and the subsequent frozen external experiment
remain distinct experimental/implementation steps.

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

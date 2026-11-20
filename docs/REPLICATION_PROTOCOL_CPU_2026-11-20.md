# Sleep EDF CPU replication protocol

Frozen on 20 November 2026 for the user's current native Windows CPU route.
This protocol fixes the source-only training family and analysis for the new
recovery-enabled replication. The user launches staged training in a visible
terminal. The study is `runs/bilstm_replication_cpu_2026-11-20`; its recipe is
`configs/sleepedf/experiments/experiment_bilstm_replication_cpu.json`.

## Data and normalization

Reuse the existing Sleep-EDF Expanded participant split and prepared
`research_v1` stores: 70 source-training participants, 15 source-validation
participants and the existing source-test partition. Preserve the eligible
windows, ten-epoch context ending at the target, preprocessing, ten spectral
features and fixed five-class mapping. Offline preprocessing and bidirectional
context do not establish a real-time causal method.

Keep label-draw seed 42 and both participant budgets: 18 and 70. The source-only
unlabeled waveform pool contains the same 70 training participants for every
SSL seed. No validation, source-test or ISRUC recording enters SSL or
normalization fitting. Waveform scaling uses the fixed source-training pool;
spectral scaling uses the corresponding labeled budget. Share normalization
within each budget across conditions and seeds. The frozen budget manifests
must match the earlier participant selections and eligible source records.

## Model family and schedule

| Condition | Representation | Initialization |
|---|---|---|
| M0 | Spectral features and BiLSTM | Fresh |
| M1 | Raw waveform and BiLSTM | Fresh |
| M2 | Waveform and spectral concatenation with BiLSTM | Fresh |
| M3 | Gated cross-attention fusion with BiLSTM | Fresh |
| M4 | Raw waveform and BiLSTM | Source-only waveform SSL |
| M5 | Concatenation with BiLSTM | Source-only waveform SSL |
| M6 | Gated cross-attention fusion with BiLSTM | Source-only waveform SSL |

Use optimization seeds 17, 42 and 123 with separate fresh SSL pretraining for
each seed. Share that seed's encoder initialization across its SSL conditions
and both label budgets, then fine-tune the encoder with the classifier.
Keep all model dimensions, augmentation and contrastive settings from the
completed recipe, including BiLSTM units 32 per direction and dropout 0.3.

SSL uses ten epochs, batch 32 and learning rate 0.001. Supervised fits use at
most thirty epochs, batch four and learning rate 0.001. Enable epoch recovery,
eager validation at batch four, full validation each epoch, and no batch caps.
Select the maximum pooled validation macro-F1 checkpoint, retaining the earliest
best on ties. Early stopping uses macro-F1 patience five; reduce LR on validation
loss plateaus by factor 0.5, patience two, minimum 1e-6. Keep latest recovery
state separately from the selected best model.

The study has 48 jobs: one waveform-statistics fit, two budget normalizers,
three SSL fits and 42 supervised fits. All three seeds, including 42, are trained
under this procedure. Historical seed-42 results remain separate exploratory
evidence because inference parity does not establish equivalence of the older
implicit training sampler and the new explicit epoch sampler. Capped pilot and
runtime benchmark scores do not enter research results.

## Source analysis

Checkpoint selection uses pooled validation macro-F1. Primary source reporting
uses equal-participant macro-F1 with fixed five-class definitions. The two
primary low-budget contrasts are M6 minus M3 (SSL within attention) and M6 minus
M5 (attention versus concatenation under SSL). Report negative and null outcomes
alongside positive ones. These choices follow source-development observations;
they are not independently registered confirmatory hypotheses.

Report the same contrasts at full labels as secondary outcomes. Supporting
controls include M4 minus M1, M5 minus M2, M3 minus M2 and the spectral reference.
Retain all seeds and conditions regardless of ranking. Show per-seed results,
participant means, pooled metrics, per-class F1/confusion counts and actual
selected/completed epochs. Calibration analyses retain existing definitions.

Compute paired participant differences within each seed, average those
differences across seeds, and bootstrap the fifteen participants with 2,000
paired resamples, bootstrap seed 42 and 95% percentile intervals. Show seed
variation separately. Do not treat the repeated participants as forty-five
independent people. These exploratory source-validation intervals condition on
the three trained seeds, have no multiplicity or checkpoint-selection correction,
and are not independent confirmation. Do not infer a physiological mechanism or
BiLSTM superiority over LSTM from a family that does not isolate those claims.

## Staged CPU execution

Use `scripts/research_cpu.ps1` and the existing D: environment. Preserve `src/`,
packages, data and study manifests during the frozen campaign. The assistant
inspects and supplies commands; the user performs all optimizer work. No GPU,
WSL, cloud setup, system storage change or spending is part of this protocol.

The first invocation is limited to four additional jobs: waveform statistics,
the two budget normalizers and `ssl_seed17`. Stop for artifact inspection before
the next stage. `--max-jobs` counts newly executed jobs on each invocation;
repeating the command advances the study further. Worker logs and checkpoints
remain inside the new study on D:. A partial first stage still reports
`limited_run: false` because its individual jobs use uncapped schedules.

For supervised interruption, preserve attempts and use the explicit documented
retry/resume interface after inspection. `--resume-supervised` does not resume
SSL at scheduler level. If SSL is interrupted, retain its completed epoch states
and request inspection before retrying; do not assume the same flag applies.
Never delete locks or bypass compatibility seals to force continuation.

## Evaluation boundary

Before new source-test scoring, separately freeze the evaluated family, seed
aggregation, endpoints, paired comparisons, multiplicity treatment and uncertainty
policy. Historical baseline source-test exposure must be disclosed; the split
was not pristine throughout development. ISRUC predictive scoring remains
unopened and cross-dataset work remains deferred. This training protocol does
not open either held-out evaluation path.

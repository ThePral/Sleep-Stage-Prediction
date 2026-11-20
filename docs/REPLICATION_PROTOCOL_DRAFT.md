# Draft Sleep EDF replication protocol

This draft specifies a consistent repeated-seed comparison of raw waveform
learning, waveform/spectral fusion and source-only SSL with a BiLSTM classifier.
The tested CPU eager path has passed its engineering checks. Resource allocation
and execution platform remain undecided, so this is not a frozen study or an
instruction to start the full campaign.

## Data and normalization

Reuse the existing Sleep-EDF Expanded participant split and prepared
`research_v1` stores: 70 source-training participants, 15 source-validation
participants and the existing source-test partition. Keep the eligible windows,
ten-epoch context ending at the target, preprocessing, spectral features and
fixed five-class mapping unchanged. Offline preprocessing and the bidirectional
head do not support a real-time causal claim.

Keep participant label-draw seed 42 and both labeled budgets: 18 participants
and all 70. The source-only unlabeled waveform pool contains the same 70
training participants for every SSL seed. No validation, source-test or ISRUC
recording enters SSL or normalization fitting. Waveform scaling uses the fixed
source-training pool; spectral scaling uses the corresponding labeled training
budget. Share normalization within each budget across conditions and seeds.

## Model family and optimization

| Condition | Representation | Initialization |
|---|---|---|
| M0 | Spectral features and BiLSTM | Fresh |
| M1 | Raw waveform and BiLSTM | Fresh |
| M2 | Waveform and spectral concatenation with BiLSTM | Fresh |
| M3 | Gated cross-attention fusion with BiLSTM | Fresh |
| M4 | Raw waveform and BiLSTM | Source-only waveform SSL |
| M5 | Concatenation with BiLSTM | Source-only waveform SSL |
| M6 | Gated cross-attention fusion with BiLSTM | Source-only waveform SSL |

Use optimization seeds 17, 42 and 123, with separate fresh SSL pretraining for
each seed and a shared encoder initialization for that seed's SSL conditions
at both budgets. Keep model dimensions, augmentation, contrastive objective and
learning-rate settings from the completed recipe. SSL uses ten epochs and batch
32; supervised fits use at most thirty epochs, batch four, learning rate 0.001
and validation-macro-F1 early-stopping patience five. Reduce learning rate on
validation-loss plateaus by a factor of 0.5, patience two and minimum 1e-6,
as in the existing trainer. No training or validation batch caps apply to
the replication recipe.

Enable epoch recovery and eager validation at batch four. Keep compiled
validation disabled. Use full validation every epoch and the existing maximum
pooled validation macro-F1 checkpoint policy; ties retain the earliest best
model. Retain latest recovery state separately from the selected model.

The recipe defines 42 supervised fits, three SSL fits and three preparation
jobs. The new seed-42 fits are necessary for a three-seed summary under one
training procedure. The historical seed-42 study used a different sampling/fit
path and remains separate exploratory evidence. The capped pilot and runtime
benchmark do not enter research metric tables.

## Source analysis plan

Keep checkpoint selection by pooled validation macro-F1 distinct from reporting
equal-participant macro-F1. For the label-efficiency question, the two primary
low-budget contrasts are M6 minus M3 (SSL within attention) and M6 minus M5
(attention versus concatenation under SSL). Report both directions of effect,
including a negative or null result. These are proposed contrasts for the new
replication, not previously registered confirmatory hypotheses.

Report the same contrasts at full labels as secondary outcomes. Include raw
SSL versus raw scratch, concatenation SSL versus scratch, scratch attention
versus scratch concatenation, and the spectral reference as supporting controls.
Do not remove conditions based on their validation ranking. Report per-seed
results, participant means, pooled metrics, per-class F1/confusion counts and
actual selected/completed epochs. Descriptive calibration and uncertainty
analyses can reuse the existing evaluation definitions.

Average paired participant differences across the three seeds before a paired
participant-cluster bootstrap, and show seed-level variation separately. Fifteen
validation participants reused across seeds are not forty-five independent
participants. Participant intervals condition on the trained seeds; they are
not unrestricted inference over all optimization randomness. Source validation
was used for checkpoint selection and prior development, so these results
remain source-development evidence rather than independent confirmation.

The family does not isolate BiLSTM versus LSTM superiority or establish a
physiological mechanism for attention. Such claims require additional matched
comparisons or analyses, rather than reinterpretation of these results.

## Execution and evaluation boundaries

Choose CPU or a separately validated GPU environment before freezing a fresh
study. The CPU recipe alone does not authorize GPU/WSL setup or expenditure.
Current host memory measurements do not predict GPU VRAM. Preserve old
artifacts and never bypass source/package/resume seals to migrate a checkpoint.

After the platform and recipe are fixed, create and inspect a fresh study,
then provide staged `execute --live` commands through `scripts/research_cpu.ps1`
for CPU execution. The user launches all optimizer work. Start with preparation
and the first SSL job, inspect its sealed completion, then advance manually.
Keep code and packages fixed during the frozen campaign. Preserve failed
attempts and use the documented explicit recovery interface without selectively
dropping seeds or conditions.

Before new source-test scoring, freeze the evaluated family, seed aggregation,
primary endpoints, paired comparisons, multiplicity treatment and uncertainty
policy. Historical source-test baseline exposure must be disclosed; it is not
a pristine split throughout the project's history. ISRUC predictive scoring
remains unopened and cross-dataset work remains deferred.

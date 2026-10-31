# Sleep Stage Prediction from EEG Using Spectral Features and LSTM
[![XERJ Corpus Hub contributor](https://hub.xerj.org/contributors/ThePral/badge.svg)](https://hub.xerj.org/contributors/ThePral)

The research extension adds aligned waveform/spectral data for multi-view models,
source-only self-supervised pretraining, and external ISRUC evaluation. See
[implementation status](IMPLEMENTATION_STATUS.md) and the
[research roadmap](RESEARCH_ROADMAP_2026-10-26.md) for current progress.

A sleep-stage classification system based on **single-channel EEG**, **frequency-domain features**, and **Long Short-Term Memory (LSTM)** neural networks.

The project investigates whether a compact set of handcrafted spectral EEG features can capture enough information for reliable automatic sleep-stage prediction while modeling the temporal dependencies between consecutive sleep epochs.

The final system uses a **stacked LSTM architecture** trained on sequences of consecutive 30-second EEG epochs and predicts five sleep stages:

- Wake (W)
- N1
- N2
- N3
- REM

---

## Overview

Automatic sleep staging is an important component of sleep analysis and is traditionally performed by trained specialists using polysomnography (PSG). Manual scoring is time-consuming and subjective, motivating the development of automated sleep-stage classification methods.

This project focuses on a lightweight alternative to large end-to-end deep-learning systems.

Instead of feeding raw EEG directly into a complex neural network, each 30-second EEG epoch is represented using a compact set of spectral features:

- Absolute delta power
- Absolute theta power
- Absolute alpha power
- Absolute beta power
- Relative delta power
- Relative theta power
- Relative alpha power
- Relative beta power
- Spectral entropy
- Dominant frequency

These features are then normalized and organized into temporal sequences before being processed by an LSTM-based classifier.

---

# Project Goals

The main goals of the project are:

1. Extract meaningful frequency-domain characteristics from EEG.
2. Model temporal dependencies between consecutive sleep epochs.
3. Investigate the effect of temporal context length.
4. Compare different loss strategies for imbalanced sleep-stage data.
5. Compare different LSTM capacities and architectures.
6. Evaluate the final model using participant-independent data splitting.
7. Analyze the model's strengths and weaknesses at the individual sleep-stage level.
8. Provide a reproducible inference pipeline.

---

# Dataset

## Sleep-EDF Expanded

The project uses the **Sleep-EDF Expanded** dataset available through PhysioNet.

The dataset contains:

- **197 whole-night recordings**
- **100 participants**
- Sleep Cassette (SC) and Sleep Telemetry (ST) recordings
- EEG channel: **EEG Fpz-Cz**
- Sampling frequency: **100 Hz**

The project uses five target sleep stages:

| Label | Sleep Stage |
|---|---|
| W | Wake |
| N1 | NREM Stage 1 |
| N2 | NREM Stage 2 |
| N3 | NREM Stage 3 |
| REM | Rapid Eye Movement |

### Important dataset detail

The 197 recordings are **not 197 independent participants**.

Multiple recordings can belong to the same participant. Therefore, the project uses **participant-level splitting** to prevent recordings belonging to the same participant from appearing in different subsets.

This is important for avoiding participant-level data leakage.

### Data source

Sleep-EDF Expanded:

https://physionet.org/content/sleep-edfx/1.0.0/

---

# Methodology

The overall pipeline is:

```text
Sleep-EDF Expanded
        │
        ▼
EEG Fpz-Cz
        │
        ▼
30-second epochs
        │
        ▼
EEG preprocessing
        │
        ▼
Power spectral analysis
        │
        ▼
10 handcrafted spectral features
        │
        ▼
Quality control
        │
        ▼
Participant-level train/validation/test split
        │
        ▼
Log transformation + standardization
        │
        ▼
Temporal sequence construction
        │
        ▼
Stacked LSTM
        │
        ▼
5-class sleep-stage prediction
        │
        ▼
Evaluation
```

**Research data workflow**

Run commands from the repository root. The new workflow writes separate,
versioned stores under `data/processed/research_v1`; it verifies completed stores
before reusing them and refuses conflicting inputs or policies. Use a new output
root when changing preprocessing, channels, scorers, or split assignments.

```powershell
# Regression checks (no extra test package required).
.venv\Scripts\python.exe -X utf8 -m unittest discover -s tests -v

# Preview inputs and destinations without processing signals.
.venv\Scripts\python.exe -X utf8 -m src.data.prepare_research_data --dataset sleepedf --recordings SC4002 --dry-run

# Prepare source data and compare every supervised feature with the old cache.
.venv\Scripts\python.exe -u -X utf8 -m src.data.prepare_research_data --dataset sleepedf --all --compare-cached

# Prepare external data using the fixed C3 contralateral derivation and scorer-1 workbook.
.venv\Scripts\python.exe -u -X utf8 -m src.data.prepare_research_data --dataset isruc --all

# Check all stores and reproduce the frozen source model from the new features.
.venv\Scripts\python.exe -X utf8 -m src.evaluation.validate_research_data --require-complete --check-baseline --report reports/research_data_validation.json
```

Each recording contains `waveforms.npy`, `epochs.csv`, and a completion manifest
with preprocessing parameters, source-file hashes, and output-file hashes. Signals
remain in volts and powers in volts squared. No normalization is fitted during
dataset preparation. Every complete physical epoch is retained, including
unscored/quality-rejected epochs with explicit masks. The authoritative source
split is `final_model_corrected/participant_split.csv`; ISRUC is marked `external`.

Waveforms are memory-mapped and overlapping windows are gathered by index:

```python
from src.data.epoch_store import EpochStore

with EpochStore("data/processed/research_v1/sleepedf/SC4002") as store:
    positions = store.sequence_positions(sequence_length=10)
    batch = store.gather(positions[:8])
    # waveform: (batch, 10, 3000, 1); spectral: (batch, 10, 10)
```

For unlabeled epoch/window selection, `supervised=False` uses signal QC rather
than stage availability. A pretraining caller must additionally restrict its
inputs to Sleep-EDF training participants. Feature scaling remains a separate
training-only operation. The store rejects skipped physical positions and
ineligible contexts rather than silently joining gaps.

The canonical feature-model loader avoids the historical 20-epoch bundle:

```python
from src.inference.predict import SleepStagePredictor

predictor = SleepStagePredictor.from_directory()
```

Earlier notebooks contain historical experiments, including the old recording
split, and can write incompatible artifacts into legacy result directories.
Use the research commands and corrected bundle for the extension. The local
dependency snapshot is `requirements-windows-py313.lock.txt`; it records the
tested Windows/Python 3.13 environment and is not a cloud GPU environment lock.

**Supervised waveform and fusion models**

Three controls share a strided waveform CNN and a 64-unit temporal GRU:

| Configuration | Epoch representation | Parameters |
|---|---|---:|
| `configs/supervised_raw.json` | Waveform only | 56,453 |
| `configs/supervised_concat.json` | Waveform plus spectral-token concatenation | 65,989 |
| `configs/supervised_gated.json` | Waveform queries over ten spectral keys/values, with gated residual fusion | 82,757 |

The default input is ten 30-second epochs. CNN downsampling produces 30 waveform
tokens per epoch before fusion. Layer normalization avoids fitting target-batch
statistics. The waveform encoder is accessible through `get_waveform_encoder()`
for self-supervised pretraining and checked downstream transfer.

Run a short integration pilot first:

```powershell
.venv\Scripts\python.exe -u -X utf8 -m src.training.supervised --config configs/supervised_gated.json --run-dir runs/my_gated_pilot --epochs 2 --train-recordings SC4001 SC4002 --validation-recordings SC4011 --max-train-batches 2 --max-validation-batches 2
```

This example trains on two source recordings and evaluates eight evenly spaced
validation windows. It checks execution, gradient updates, and checkpoint reload;
its metrics do not establish model quality. Capped validation sampling uses no
stage labels, and all limited runs are marked in their saved configuration.

For a full source training run, omit the recording selections and batch caps:

```powershell
.venv\Scripts\python.exe -u -X utf8 -m src.training.supervised --config configs/supervised_gated.json --run-dir runs/gated_seed42
```

The default is 30 epochs and batch size four. Benchmark the configured device and
adjust the batch size before a large experiment; the current integration pilots
ran on CPU. Each run requires a new output directory. `runs/` is ignored by Git;
selected experiment summaries should be retained under `reports/`.

The trainer opens only source training and validation stores matching the frozen
participant manifest. It fits normalization once on unique eligible training
epochs: per-epoch waveform mean removal with a single source RMS amplitude scale,
and log-power/standard spectral scaling. No per-epoch variance normalization or
target-cohort fitting occurs. Use `--normalizer <previous-run>/normalizer.json`
to reuse the same statistics for a matched comparison; recording identities,
manifest hashes, and preprocessing must agree.

Checkpoints maximize source-validation macro-F1 over the fixed five classes;
absent classes contribute zero F1. Validation loss drives learning-rate reduction.
Training windows shuffle reproducibly, and memory maps plus one prefetched batch
bound loader memory without caching overlapping raw windows.

Completed runs contain `model.keras`, `normalizer.json`, resolved configuration,
data/code provenance, training history, validation predictions with epoch IDs,
metrics, confusion matrix, and a hashed `bundle.json`. The bundle is published
only after the best checkpoint reloads with matching predictions. Source test and
ISRUC predictive evaluation remain outside this development trainer.

Predict from raw prepared inputs using the frozen bundle:

```python
from src.data.epoch_store import EpochStore
from src.inference.multiview import MultiviewPredictor

predictor = MultiviewPredictor("runs/my_gated_pilot")
with EpochStore("data/processed/research_v1/sleepedf/SC4011") as store:
    batch = store.gather(store.sequence_positions(10)[:4])
    predicted_indices, probabilities = predictor.predict_batch(
        batch["waveform"], batch["spectral"]
    )
```

Waveforms must be in volts and spectral powers in volts squared, before scaling.
The inference wrapper applies the saved normalization and rejects incomplete,
modified, or dimensionally incompatible bundles. For the raw-only control,
`predict_batch` only requires the waveform argument.

**Source-only contrastive pretraining and probing**

The initial SSL method uses two independently augmented versions of the same
30-second waveform. The shared CNN feeds mean pooling and a two-layer projection
head, trained with symmetric NT-Xent ([SimCLR](https://proceedings.mlr.press/v119/chen20j.html)).
Downstream models receive the CNN weights before pooling/projector; the projector
is discarded. This implementation does not pretrain the spectral branch.

`SourceEpochPool` opens only frozen Sleep-EDF training participants. Eligibility
uses `quality_valid`, including unscored epochs, and never uses stage availability.
The shared store checker validates existing annotations for integrity, but the
SSL pool retains neither stage labels nor spectral arrays. A separate waveform
normalizer fits mean-removed RMS amplitude on unique quality-valid source epochs.
Validation, test, and ISRUC signals cannot enter this fitting path.

Defaults are gain 0.8–1.2, additive noise at 0.02 times each input epoch's RMS,
and a randomly placed contiguous mask of zero to one second. These are candidate
EEG invariances, requiring source-data inspection and augmentation ablations.
No frequency masking, permutation, wraparound, or heavy cropping is applied.
Clean spectral features are not paired with augmented raw signals during SSL.

Epochs shuffle globally without replacement each pass; incomplete final batches
are dropped. Two views of each original epoch are the positive pair. Other
epochs supply negatives, except epochs from the same recording within one
physical epoch (30 seconds) by default. Temporally separated same-participant
negatives remain eligible. This exclusion is a declared starting assumption,
not an optimized EEG rule; same-stage false negatives and identity shortcuts
remain possible. If exclusion removes every negative for any anchor, training
fails with an actionable message rather than silently changing the loss.

Batch size 32 means 64 augmented inputs and at most 62 negatives per anchor.
Changing gradient accumulation would not enlarge that denominator. The default
ten-epoch budget is a starting diagnostic schedule, not evidence of convergence
or a measured fit on a 4 GB GPU. Current pilots use CPU.

Run a limited real-data pilot and recover its completed epoch into a new run:

```powershell
.venv\Scripts\python.exe -u -X utf8 -m src.training.contrastive --config configs/pretrain_raw.json --run-dir runs/my_ssl_epoch1 --train-recordings SC4001 SC4002 --epochs 1 --max-batches 2

.venv\Scripts\python.exe -u -X utf8 -m src.training.contrastive --config configs/pretrain_raw.json --run-dir runs/my_ssl_epoch2 --train-recordings SC4001 SC4002 --epochs 2 --max-batches 2 --resume-from runs/my_ssl_epoch1
```

For full source pretraining omit recording selections and batch caps:

```powershell
.venv\Scripts\python.exe -u -X utf8 -m src.training.contrastive --config configs/pretrain_raw.json --run-dir runs/raw_ssl_seed42
```

SSL saves the last epoch of the declared schedule; it does not select a checkpoint
using held-out signals or pretraining loss. A sealed `resume_state.json` is
published after each complete epoch, with model/projector/optimizer tensors,
epoch counter and history. Stateless augmentations and a seeded sampler reproduce
the next epoch without an opaque RNG snapshot. Resume uses a new output directory,
requires matching data, update settings, package versions and implementation
hashes, and replays an interrupted partial epoch from its last complete boundary.
The total epoch count can extend the constant-learning-rate schedule; batch caps
and other training settings must match. Resume is an SSL capability; supervised
training currently exports checkpoints but does not resume optimizer state.

Completed runs contain `encoder.keras`, `waveform_normalizer.json`, configurations,
data/code provenance, history, a hash-sealed pretraining bundle, and diagnostics.
Diagnostics include positive/negative cosine similarity, contrastive retrieval,
and normalized pooled-encoder variance/covariance effective rank on a fixed,
label-free source training sample. Retrieval is augmentation-pair retrieval,
not sleep-stage accuracy; these diagnostics alone establish no task improvement.

Fine-tune the pilot encoder in a temporal classifier:

```powershell
.venv\Scripts\python.exe -u -X utf8 -m src.training.supervised --config configs/supervised_gated.json --run-dir runs/my_ssl_gated --pretrained runs/my_ssl_epoch2 --train-recordings SC4001 SC4002 --validation-recordings SC4011 --epochs 2 --max-train-batches 2 --max-validation-batches 2
```

Use `supervised_raw.json` or `supervised_concat.json` for the other controls.
The trainer verifies backbone architecture, frozen source assignment, recording
identities, content hashes and preprocessing before transfer. The current transfer
policy requires the same source recording selection for SSL and downstream
training unless a verified participant-budget plan authorizes a labeled subset
of the fixed source pool. The waveform scale stays fixed at its SSL value; spectral statistics
are fitted on unique supervised source training epochs. Both fitting populations
are recorded separately in the downstream normalizer. Inference applies this
combined frozen state to unscaled prepared inputs.

A matched scratch control uses the same input scale without loading SSL weights:

```powershell
.venv\Scripts\python.exe -u -X utf8 -m src.training.supervised --config configs/supervised_gated.json --run-dir runs/my_matched_scratch --waveform-normalizer runs/my_ssl_epoch2/waveform_normalizer.json --train-recordings SC4001 SC4002 --validation-recordings SC4011 --epochs 2 --max-train-batches 2 --max-validation-batches 2
```

For a strict linear probe, freeze the CNN, mean-pool a single epoch, and train
only a five-class affine softmax head. No GRU or hidden classifier is included:

```powershell
.venv\Scripts\python.exe -u -X utf8 -m src.training.supervised --config configs/linear_probe.json --linear-probe --run-dir runs/my_ssl_probe --pretrained runs/my_ssl_epoch2 --train-recordings SC4001 SC4002 --validation-recordings SC4011 --epochs 2 --max-train-batches 2 --max-validation-batches 2
```

The default probe has 325 trainable parameters. For a frozen random-encoder
control, replace `--pretrained <run>` with `--waveform-normalizer <run>/waveform_normalizer.json`.
`--freeze-backbone` on a temporal/fusion model trains its other modules; this is
distinct from `--linear-probe`. All downstream checkpoints still maximize source
validation macro-F1, and the existing inference wrapper loads both probe and
fine-tuning bundles.

The SSL tests include a direct numerical loss comparison, augmentation bounds,
fully hidden-label invariance, temporal exclusion, an interrupted-run recovery
comparison, exact transfer into all three variants, frozen-probe weight checks,
and pretrain-to-fine-tune-to-inference execution. Participant label budgets and
serial study execution are documented below. GPU profiling and frozen external
scoring remain subsequent milestones. Capped pilot metrics do not test the research hypotheses.

**Participant budgets and the source-only experiment runner**

The study runner implements the seven roadmap conditions:

| ID | Representation | Initialization |
|---|---|---|
| M0 | Ten features and two stacked LSTM layers (32 units by default) | Scratch |
| M1 | Waveform CNN + temporal GRU | Scratch |
| M2 | Waveform + spectral concatenation + GRU | Scratch |
| M3 | Waveform + gated spectral attention + GRU | Scratch |
| M4 | Same architecture as M1 | Raw SSL |
| M5 | Same architecture as M2 | Same raw SSL |
| M6 | Same architecture as M3 | Same raw SSL |

M0 is a newly trained reference using the historical spectral architecture and
the new shared training policy. It does not replace or reproduce the preserved
historical checkpoint's training history. `configs/supervised_spectral.json`
also runs this path independently. Its inference bundle accepts
`predict_batch(spectral_features=features)` without waveform inputs.

Plan an entire study before executing it. Planning validates source stores and
writes a frozen participant-budget manifest, configuration/code/package hashes,
and a dependency graph; it does not train models. Start with the deliberately
small real-data pilot:

```powershell
.venv\Scripts\python.exe -u -X utf8 -m src.experiments.run plan --config configs/experiment_pilot.json --study-dir runs/my_matrix_pilot

# Execute four jobs, then return with the rest pending.
.venv\Scripts\python.exe -u -X utf8 -m src.experiments.run execute --study-dir runs/my_matrix_pilot --max-jobs 4

# Verify/skip completed artifacts and execute the remaining jobs.
.venv\Scripts\python.exe -u -X utf8 -m src.experiments.run execute --study-dir runs/my_matrix_pilot
```

The pilot covers two source training participants, one validation participant,
one seed and two nested budgets. Its 25% fraction rounds up to one of those two
training participants (50% actual); the full pilot budget uses both. It uses
a smaller CNN/LSTM and capped one-epoch runs. These are execution checks, with
the actual fractions and costs recorded, not a 25%-of-70-participant study or
a performance comparison.

For the canonical full scope, use a new directory and the main configuration:

```powershell
.venv\Scripts\python.exe -u -X utf8 -m src.experiments.run plan --config configs/experiment_matrix.json --study-dir runs/source_matrix_seed42

# Launch only after choosing an appropriate measured compute budget.
.venv\Scripts\python.exe -u -X utf8 -m src.experiments.run execute --study-dir runs/source_matrix_seed42
```

The main configuration plans 18 jobs: one waveform-statistics job, two budget
normalizers, one SSL checkpoint, and fourteen supervised conditions. It uses all
70 source training participants as the unlabeled pool and the fixed 15 labeled
validation participants. Its default fractions 0.25 and 1.0 select **18 and 70**
training participants. Fractions 0.1 and 0.5 are also supported (**7 and 35**).
Participant counts use ceiling, and actual fractions are saved explicitly.

Budget draws use a seeded permutation of participant IDs and nested prefixes,
including every discovered recording of each selected participant. Selection
uses no stage balancing. Source-training labels are subsequently used to report
costs and assert that all five stages occur among eligible training targets.
If a draw fails this check, planning fails; it never redraws based on labels.
Duplicate fractions/counts or repeated seeds are rejected. Every condition uses
the same draw and optimization seed. Add `draw_seeds` and `training_seeds` in
the JSON configuration before planning repeated experiments; the initial
single-seed screen is exploratory.

`label_budgets.json` reports physical/annotated/usable epochs, labeled hours,
participants, recordings, target class counts and validation cost separately.
Annotated hours count available five-class labels, including QC-rejected epochs;
usable hours count supervised-eligible epochs. Whole-recording hours are also
recorded. A fixed fully labeled validation set remains an annotation expense:
this protocol measures **training-label efficiency**, not total annotation-cost
reduction. Planning statistics exclude source-test annotations. Discovery reads
identity columns from Sleep-EDF roster rows before filtering held-out records;
ISRUC stores are not opened by the study planner or runner.

All budgets share waveform RMS statistics fitted once on the full quality-valid
training scope. The spectral scaler is fitted separately on unique supervised
epochs from each labeled budget. That combined normalizer is shared byte-for-byte
across its M0–M6 conditions and optimization seeds. SSL is trained once per
optimization seed on the same full unlabeled pool, then reused across all draws,
budgets and SSL conditions. No hidden labels determine SSL batches or statistics.
Scratch controls receive the same unlabeled-pool waveform scale without SSL
weights, isolating encoder initialization in each matched comparison.

Content hashes and the frozen split authorize transfer from the full SSL pool
to the smaller labeled budget. An arbitrary source superset is rejected; it must
be exactly the planned unlabeled scope. Standalone training can use a generated
manifest with `--budget-plan <study>/label_budgets.json --budget-id draw42_n18`,
plus `--pretrained <ssl-artifacts>` or a compatible fixed-pool normalizer. Budget
mode controls complete recording selections; do not combine it with recording
overrides. Ordinary training keeps the stricter matching-selection policy.

Jobs run serially in fresh Python processes. Each dependency is verified before
use, and every completed attempt seals model, configuration, normalization,
predictions, metrics and optimizer-state files. The final completion manifest is
published atomically; an interrupted write cannot leave a partial completed seal.
The ledger and per-job logs show
progress. Repeating `execute` verifies and skips completed attempts. Use
`--retry-failed` to start a fresh attempt for a failed/interrupted job; earlier
attempts are preserved. This scheduler retry restarts that job's training;
the standalone SSL entry point separately supports epoch-boundary optimizer
resume. Two schedulers cannot share an active study. After a hard-killed
scheduler, inspect the PID in `.runner.lock` and remove that file only once
the scheduler and its worker have stopped.

Study definitions and relevant source/package versions must stay unchanged
between planning and execution. Create a new study after implementation/config
changes. For relocation, `execute --data-root <new-root> --split-file <new-file>`
supports new absolute paths when the frozen split and recording contents match.

Completed supervised jobs appear in `source_validation_results.csv` with seed,
budget, labeled hours, parameters and source-validation scores. The paired table
records M4–M1, M5–M2, M6–M3, M3–M2 and M6–M5 macro-F1 differences only when both
conditions completed with identical normalizer and validation target hashes.
These are pooled source-validation summaries. They do not provide confidence
intervals, independent source-test results or external generalization estimates.
Participant uncertainty, compute profiling and a predeclared frozen external
scoring batch remain upcoming work.

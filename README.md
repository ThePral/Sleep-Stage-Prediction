# Sleep Stage Prediction from EEG Using Spectral Features and LSTM

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

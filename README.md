# Sleep Stage Prediction from EEG Using Spectral Features and LSTM

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
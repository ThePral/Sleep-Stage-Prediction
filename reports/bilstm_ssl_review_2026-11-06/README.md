# Completed BiLSTM study SSL review — 6 November 2026

The user completed the first four jobs of
`runs/bilstm_terminal_2026-11-02`: quality-only waveform statistics, spectral
normalizers for 18/70 labeled participants, and ten fixed-schedule SSL epochs.
All four job artifact inventories, the final pretraining bundle and epoch-ten
checkpoint seal pass hash verification. Source code, packages and the planned
dataset scope match the original freeze. Fourteen supervised jobs remain pending;
no training process is active at this observation.

`status.json` records read-only study status. `verification.json` records the
additional artifact review: ten consecutive finite history rows agree with
checkpoint history; the recorded 101,060 optimizer iterations match ten passes
of 10,106 full batches; all 27,008 exported encoder parameters are finite when
read directly from the Keras archive. This review does not restore or execute
an optimizer. The user-run trainer recorded a zero encoder reload difference.

The SSL pool contains 323,423 quality-valid epochs from 138 recordings and
70 training participants. Normalizer provenance matches that frozen pool.
Both label budgets share exactly the SSL waveform statistics; spectral fitting
uses their own 76,604/321,688 eligible labeled epochs. All 64 fixed diagnostic
epoch identities belong to source training recordings. No ISRUC or new
source-test predictive evaluation was performed.

Loss decreased from 0.557325 to 0.459575. Final augmented-view retrieval was
1.0; it was already nearly saturated in epoch one. Final encoder covariance
effective rank was 14.93 on the fixed 64-epoch diagnostic sample, compared with
16.30 before training. These diagnostics do not establish staging accuracy,
transfer benefit, full-data representation quality or convergence. Saturated
view retrieval means it is unsuitable for judging downstream benefit. Preserve
the frozen schedule and evaluate transfer through the planned supervised pairs.

To continue in the user's terminal, omit the first-stage job limit:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/research_cpu.ps1 -m src.experiments.run execute --study-dir runs/bilstm_terminal_2026-11-02 --live
```

This skips the four verified completed jobs and runs seven supervised conditions
at each of the two participant budgets. Each classifier has up to 30 epochs,
with early stopping and source-validation-only checkpoint selection. Keep
source files and packages unchanged. The assistant performed artifact analysis
only: no TensorFlow import, optimizer updates or training launch in this review.

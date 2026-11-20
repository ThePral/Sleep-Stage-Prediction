# Recovery enabled pilot completion review

The user completed all eighteen jobs in
`runs/bilstm_recovery_eager_pilot_2026-11-20`. The artifact review verifies every
job seal, current source/package and source-data scope, fourteen supervised
models and forty-two sealed recovery generations. No training code or environment
change was needed.

Each supervised job completed two epochs with two training batches per epoch:
four optimizer updates per model. All saved generations for epochs zero, one
and two load successfully with finite numeric arrays and matching history lengths
and optimizer counters. Latest recovery states contain the same selected best
model files as the completed inference bundles. History CSVs match sealed
histories. Every supervised export uses the same eight capped validation targets.
SSL completed one capped epoch and two optimizer updates. These are integration
results; the tiny validation sample does not support research performance claims.

`inspect_pilot.py` reproduces the seal, state-array, counter and export checks
without model execution or TensorFlow imports. The completed pilot shows that
recovery states were published; it does not yet show that restoring one gives
the same subsequent real-data training trajectory. Full-family eager inference
parity remains passed, and compiled validation remains unaccepted.

## Replay from an earlier saved epoch

The next user-run check resumes the SSL-initialized cross-attention BiLSTM
`M6_draw42_n2_seed42` from its sealed epoch-one generation. It uses the exact
same model, normalization, SSL initialization, data, seed and two-epoch schedule
as the completed pilot, in a fresh output directory:
`runs/bilstm_real_resume_m6_2026-11-20`.

This replays only epoch two and performs two optimizer updates. The completed
pilot and its ledger remain intact. Because M6 selected epoch one as its best
model, the replay also exercises preservation of a best model from before the
resume boundary. A physical power interruption is unnecessary for this check.

Run from the repository root in PowerShell:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File reports/recovery_pilot_review_2026-11-20/resume_m6.ps1
```

The launcher delegates to `scripts/research_cpu.ps1` and retains terminal output
in `m6_resume_terminal.log` here. `-InspectOnly` prints arguments without launching
Python; that is the only launcher mode used by the assistant. The output directory
and transcript must be new. A failed run is preserved for diagnosis.

The final summary should report `resumed_after_epoch: 1`, `epochs_completed: 2`
and `limited_data_run: true`. Report any contract mismatch instead of changing
the environment or bypassing the seal. After completion, compare final model and
optimizer arrays, RNG state, history, callbacks, best-epoch selection and saved
validation probabilities against the original uninterrupted M6 run. Use the
existing synthetic-gate float tolerances of `atol=rtol=1e-7`, with integer state
checked exactly; exclude wall-clock fields. A pass applies to this capped real-data
M6 configuration. Representative runtime measurement and a frozen replication
protocol remain separate steps before a long campaign.

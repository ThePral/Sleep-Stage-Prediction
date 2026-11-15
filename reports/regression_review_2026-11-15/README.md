# Full regression passed and real-data parity handoff — 15 November 2026

The user ran the full suite: **136 tests passed in 147.021 seconds**. The final
successful summary is present in
`reports/recovery_validation_2026-11-13/full_regression_01.log`.
`verification.json` records its hash and read-only audit. Source hashes match
the earlier implementation/recovery audit, installed packages match the passed
synthetic gate contract, all eighteen historical completed-job inventories
verify, and baseline hashes remain unchanged.

The assistant did not rerun optimizer-based tests, train models or execute
real-data parity/performance measurements. Those remain user-owned. No `src/`
or package changes were made in this handoff.

## Next check: bounded real-data inference parity

`scripts/validation_parity.py` replays already completed model checkpoints on
source validation only. It verifies original job and model-bundle hashes,
source split/manifests, the exact validation identity/label/order, probability
tolerance, predicted classes, confusion matrices and metric differences. It
loads existing normalization and never fits normalization or updates an optimizer.

First run the bounded 64-window check across all fourteen saved classifiers:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/research_cpu.ps1 scripts/validation_parity.py --study-dir runs/bilstm_terminal_2026-11-02 --samples 64 --validation-batch-size 32 --output-dir reports/validation_parity_2026-11-15_smoke
```

The shared window selection spans the ordered validation cohort without using
prediction outcomes. Each model's saved probabilities are compared with eager
batch-four replay and compiled batch-32 replay. Probability tolerances are
`atol=rtol=1e-6`; predicted classes must agree exactly, and reported loss,
accuracy and macro-F1 differences must not exceed `1e-6`. Any failure yields
a nonzero exit status and preserved diagnostics. It never overwrites an output
directory. Send the final summary and inspect any failure before proceeding.
The expected successful summary has `models_checked: 14`, `passed: true` and
`full_cohort_checked: false`. A sampled pass does not establish full-cohort parity.

After reviewing the sampled result, a separate full-cohort check can use
`--samples all` and a new output directory. That mode compares compiled replay
against all stored validation exports and skips the expensive full eager replay.
It still checks all identities, classes and metrics, without opening source test
or ISRUC scores. Do not bypass changed targets, numerical differences or class
flips merely to obtain a pass.

Each output records policy/source/script hashes, selected targets, per-model
progress and results. Warm inference-pass timings include window gathering and
scaling but exclude initial integrity scanning, model loading and tracing.
They are not total training-epoch timings or evidence of end-to-end speedup.
Fresh user-run integration/throughput studies still follow parity review.

Five new pure helper tests passed separately in 0.005 seconds in the initial
check; their retained rerun is `parity_helper_tests.txt`. They cover deterministic
selection, shape/probability checks, labels/order and a near-tie class flip that
would otherwise hide inside probability tolerance. These tests were added after
the user's 136-test suite; this is not a claim of a single 141-test suite run.
No TensorFlow or real EEG execution is needed for these helper tests.

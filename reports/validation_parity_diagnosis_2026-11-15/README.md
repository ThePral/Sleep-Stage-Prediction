# Sampled validation parity diagnosis — 15 November 2026

The user's retained smoke result in `../validation_parity_2026-11-15_smoke/`
passed 13 of 14 models. Each model used the same 64 deterministically selected
source-validation targets. This does not cover the full validation cohort.

`M3_draw42_n70_seed42` failed only the probability comparison:

| Comparison | Maximum probability difference | Classes changed | Loss delta | Result |
|---|---:|---:|---:|---|
| Saved vs eager, batch 4 | 1.048e-9 | 0 | -1.110e-16 | PASS |
| Saved vs compiled, batch 32 | 5.484e-6 | 0 | 5.038e-7 | FAIL |

Accuracy, macro-F1 and confusion matrices were unchanged. The declared NumPy
allclose tolerances remain `atol=1e-6, rtol=1e-6`. Unchanged classes do not
override a failed probability tolerance. The existing observation compares
different execution modes AND batch sizes; it does not isolate the cause.
TensorFlow warning messages alone cannot establish the cause either.

The diagnostic adds exact model filtering, eager/compiled comparisons at each
requested batch size, trace counts and the largest discrepancy location.
It never replaces the original gate result with an alternative passing batch.
Only the script, pure helper tests and documentation changed; `src/`, packages
and preserved training artifacts were not edited. Seven pure helper tests pass;
no assistant-run real-data model execution or optimizer updates were performed.

Run this bounded diagnostic from the repository root:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/research_cpu.ps1 scripts/validation_parity.py --study-dir runs/bilstm_terminal_2026-11-02 --samples 64 --validation-batch-size 32 --model-id M3_draw42_n70_seed42 --diagnostic-batch-sizes 4 8 16 32 --output-dir reports/validation_parity_2026-11-15_m3_diagnostic
```

This reuses the exact same selection rule and source-validation cohort. It
loads the saved model and normalizer without fitting them; no test/ISRUC scoring
is involved. The new output directory must not already exist. Send the per-batch
lines and final summary for review. Overall `passed: false` and exit code 1 may
persist because the declared batch-32 gate remains in force.

Compare eager and compiled results at matched batch sizes before attributing
the difference to batching or compilation. Any change to the accepted execution
policy requires an explicit documented decision and a new complete parity check;
do not silently substitute a passing diagnostic result. Full-cohort parity and
end-to-end performance remain pending. Warm inference timings exclude loading
and tracing and do not establish training speedup or callback equivalence.

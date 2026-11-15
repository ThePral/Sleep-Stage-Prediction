# Matched-batch validation diagnostic — 15 November 2026

The user-run M3/n70 diagnostic confirms that the declared compiled path fails
the unchanged probability tolerance on the same 64 targets as the initial smoke
check. `inspect_diagnostic.py` verifies the retained evidence without running a
model. Current `src/` hashes still match the diagnostic policy.

| Batch | Eager vs saved: max difference | Eager gate | Compiled vs saved: max difference | Compiled gate |
|---|---:|---|---:|---|
| 4 | 1.048e-9 | PASS | 5.186e-6 | FAIL |
| 8 | 1.013e-6 | PASS | 5.186e-6 | FAIL |
| 16 | 2.086e-6 | FAIL | 4.768e-6 | FAIL |
| 32 | 1.609e-6 | FAIL | 5.484e-6 | FAIL |

The test uses `abs(candidate-reference) <= atol + rtol*abs(reference)` with
`atol=rtol=1e-6`. This explains why eager batch 8 can pass with a maximum
absolute difference slightly above 1e-6. No tolerance was changed.

Every comparison has zero class changes, identical accuracy/macro-F1 and equal
confusion matrices. All reported loss changes are within 1e-6. The largest
compiled discrepancy is N1 probability at `sleepedf:SC4321:001207` (selected
row 33). All compiled batch runs share one trace; the earlier retracing warning
does not explain this failure. At matched batch 4, compilation changes results;
eager-only batching also changes them at 16/32. The low-level numerical cause
has not been isolated. These outputs do not demonstrate data/model corruption,
but they do not satisfy the declared compiled equivalence criterion either.

Recommendation: keep `compiled_validation: false` and the original eager
validation batch size 4 for an eventual full study while its parity is checked.
Do not raise tolerances, alter TensorFlow/oneDNN environment flags, or select
batch 8 merely because this sample passes. The recovery pilot recipe currently
enables compiled validation and must not be executed as the conservative fallback.
No source, package, training configuration or historical artifact was changed
by this review. The script now offers an explicit `--eager-only` mode, which
does not construct or accept the compiled inference path. A passing eager-only
result will not turn the earlier compiled failure into a pass.

Next user-run check (one model, entire validation cohort):

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/research_cpu.ps1 scripts/validation_parity.py --study-dir runs/bilstm_terminal_2026-11-02 --samples all --eager-only --model-id M3_draw42_n70_seed42 --output-dir reports/validation_parity_2026-11-15_m3_eager_full
```

This is 65,133 inference targets with no optimizer updates, training or test/ISRUC
scoring. It prints progress every 2,048 targets and at completion. It is longer
than the 64-target diagnostic. The output directory must be new. This checks
the known failing model first; it does not establish full-family parity,
callback equivalence during training or end-to-end performance. Review its
result before any further full-family or optimizer-based work.

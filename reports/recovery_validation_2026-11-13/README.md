# User-run synthetic recovery gate passed — 13 November 2026

The user ran `tests.test_supervised_recovery_training` and reported one passing
test in 42.455 seconds. Artifacts are preserved at
`runs/recovery_gate/3149ea07f23440278975a4e95008ac5a/`. The assistant inspected
them with `inspect_gate.py` without importing TensorFlow, executing model
inference or performing optimizer updates. `verification.json` records the
independent checks and worker-log hashes.

The uninterrupted fit and the fresh-process restart after sealed epoch three
both finished at epoch four with best epoch two. All 145 numeric state arrays
match, including eight integer arrays checked exactly. The maximum floating
state-array difference and exported probability difference are both zero.
Histories, Python/NumPy RNG states and early-stop/LR callback counters match.
Both runs finish with twelve optimizer iterations and learning rate
approximately 0.0005, demonstrating the controlled test's LR reduction and
early-stop behavior. Checkpoint seals, archive CRCs, source hashes and installed
package versions also verify.

This establishes recovery equivalence for the tested tiny synthetic gated
BiLSTM, dropout configuration and fixed native CPU environment. Controlled
validation metrics in this test deliberately exercise callbacks; they are not
sleep-staging research scores. This does not establish equivalence to the old
implicit-generator trainer, all model/hardware combinations, real-data parity,
an end-to-end speedup or resilience to every possible hardware failure.

The oneDNN, CPU feature, deprecation and native-Windows GPU messages in the
user output did not cause a test failure. The intended route is native CPU;
no framework/environment change is needed merely to suppress these messages.

Next, the user runs the full regression suite from the project root, retaining
visible output and a log:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/research_cpu.ps1 -m unittest discover -s tests -v 2>&1 | Tee-Object -FilePath reports/recovery_validation_2026-11-13/full_regression_01.log
```

Use a new log filename for repeats. This suite performs optimizer updates on
test fixtures and remains user-owned. It is not a new research training matrix.
Full regression, full-cohort real-data validation parity, measured performance
and a new frozen research study remain pending. No research campaign was
launched in this review and no `src/` or package changes were made.

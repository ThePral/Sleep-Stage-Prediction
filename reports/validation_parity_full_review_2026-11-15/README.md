# Full-cohort M3 eager parity review — 15 November 2026

The user completed the full source-validation replay for
`M3_draw42_n70_seed42`, using the original eager inference path at batch size 4.
It passed the unchanged `atol=rtol=1e-6` criteria on all 65,133 targets.

- Maximum probability difference from the saved export: 1.110223e-16.
- Zero predicted-class changes; identical accuracy, macro-F1 and confusion matrix.
- Loss delta: -1.054712e-15.
- Warm inference pass: 1,763.755 seconds (29.40 minutes), including gathering/scaling.
- Total reported execution: 1,774.098 seconds (29.57 minutes).

`inspect_full.py` verifies the retained report policy, current source/script
hashes, preserved completed-job artifact seal, saved prediction export hash,
and all target IDs, labels and ordering. It loads CSV/JSON only; it neither
loads a model nor imports TensorFlow. The probability comparison itself was
performed by the user's replay; the reviewer checks its recorded result, not
an independent second model execution. No optimizer updates occurred.

The original compiled failure remains recorded and unaccepted. This pass covers
one model and one execution path. It does not establish parity for the remaining
13 models or equivalence of training callbacks, and the timings do not establish
an end-to-end training speed improvement. Keep compiled validation disabled;
do not change the tolerance or environment in response to this result.

## Next user-run check

The generated `remaining_models.txt` lists the other 13 completed classifiers
from the same frozen study, excluding M3/n70 so its full replay is not repeated.
Run both lines from the repository root in PowerShell:

```powershell
$remainingModels = @(Get-Content reports/validation_parity_full_review_2026-11-15/remaining_models.txt)
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/research_cpu.ps1 scripts/validation_parity.py --study-dir runs/bilstm_terminal_2026-11-02 --samples all --eager-only --model-id $remainingModels --output-dir reports/validation_parity_2026-11-15_remaining_eager_full
```

This performs inference only. Each model uses all 65,133 source-validation targets
with eager batch 4; the total is 846,729 model-target evaluations. Progress is
printed every 2,048 targets and results are retained after each completed model
in `progress.json`. The final summary should report 13 models. The output directory
must be new. This is a longer CPU job and may take several hours; the single M3
timing is not a measured duration for other architectures.

Full-family acceptance requires reviewing this result together with the already
verified M3 result. No new training campaign is authorized or launched here.
Performance measurements and any recovery-enabled integration pilot remain
separate user-run steps. The existing compiled-enabled pilot recipe is not the
eager fallback and remains unsuitable for that purpose.

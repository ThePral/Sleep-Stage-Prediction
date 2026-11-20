# Full model family validation parity review

All fourteen completed BiLSTM classifiers reproduce their saved source-validation
outputs on all 65,133 targets using eager inference at batch size 4. The two
user-run replays cover 911,862 model-target evaluations, with a maximum probability
difference of 1.110223e-16, zero predicted-class changes, identical accuracy and
macro-F1, and equal confusion matrices. The maximum absolute loss delta is
1.054712e-15, within the unchanged 1e-6 tolerance.

M3/n70 completed on 15 November; the remaining thirteen completed on 17 November.
The latter execution reported 21,934.810 seconds, or 6.09 hours. The combined
review on 20 November verifies exact model-family coverage, target IDs, labels
and order, preserved model artifact seals, prediction export hashes, and source
and parity-script hashes. `inspect_family.py` reproduces these checks from the
retained CSV/JSON evidence without model execution. Its probability summaries
come from the user's recorded replays.

Compiled validation remains unaccepted because of its earlier numerical failure.
These eager results establish inference parity for the tested saved models; they
do not establish training callback equivalence or a training speed improvement.
The historical completed study remains unchanged.

## Recovery integration pilot

The next step checks recovery-enabled training and artifact publication on real
source data using the accepted eager batch size. A new study is planned at
`runs/bilstm_recovery_eager_pilot_2026-11-20`, with no jobs started. Its recipe is
`configs/sleepedf/experiments/experiment_bilstm_recovery_eager_pilot.json`.

The pilot has eighteen jobs: waveform statistics, two budget normalizers, one
source-only SSL job and fourteen supervised jobs covering M0 through M6 at both
budgets. It uses two training participants and one validation participant, one
SSL epoch, two supervised epochs and at most two batches per phase. Recovery is
enabled; compiled validation is disabled; training and validation batch sizes
are both four. Matching batch sizes also preserves the capped validation target
selection rule. The old compiled-enabled pilot recipe is preserved separately.

`pilot_plan_verification.json` records the frozen fingerprint, source/package
and data-scope checks, empty ledger and configuration. This is a capped
integration exercise; its scores must not be reported as research performance
or convergence. Passing will permit inspection of real-data recovery states
and best-model exports; an interruption/resume test and a representative
end-to-end runtime measurement remain separate steps before a long campaign.

Launch it from the repository root in your PowerShell terminal:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/research_cpu.ps1 -m src.experiments.run execute --study-dir runs/bilstm_recovery_eager_pilot_2026-11-20 --live
```

Worker logs and checkpoints remain under the new study on D:. Keep `src/` and
the environment unchanged during execution. Send the final summary and any
exception before starting further experiments. Successful completion should
report eighteen planned and completed jobs with `limited_run: true`.

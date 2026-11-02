# Delegated source training execution — 1 November 2026

The user explicitly delegated execution after the earlier manual handoff.
`AGENTS.md` records the latest authorization. The source campaign uses only
Sleep-EDF training/validation data; it schedules no real source-test or ISRUC
predictions and provisions no paid compute.

The latest complete Windows regression suite passes **110 tests** in 113.994 seconds.
`windows_tests_summary.json` records the exact source hashes. Each experiment
worker now rechecks those study-specific code/package constraints immediately
before its job. An edit during a long campaign causes the next worker to fail
rather than silently mixing implementations.

Live execution status is `runs/training_execution_2026-11-01/status.json`.
The staged runner waits for package installation and a passed regression snapshot,
then requires a GPU training-kernel preflight, a five-job capped source benchmark,
and a disposable source-training GPU profile before the 18-job full matrix.
Any failed stage preserves logs and stops later stages. Status `running` does
not mean those checks or the full campaign have completed.

The benchmark is capped and cannot establish convergence. The full matrix uses
seed 42, seven conditions and two labeled budgets (18/70 participants), with a
fixed 15-participant validation scope. A completed matrix still requires review
of learning curves, repeated-seed evidence and source-selected protocol decisions
before frozen external evaluation.

Update, 2 November 2026: the final GPU attempt was under
`runs/training_execution_2026-11-01_gpu`; it stopped during the bounded SSL job
after the user raised C: disk pressure. No full GPU matrix started. The native
CPU transition and fresh source campaign are documented separately in
`reports/cpu_training_2026-11-02`. Earlier failure reports and attempts remain
historical evidence; use the new CPU ledger/status for current progress.

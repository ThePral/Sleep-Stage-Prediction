# User-run training handoff — 1 November 2026

The user requested ownership of training. The previous automatic GPU-check
continuation was stopped before it launched an optimizer workload. Setup now
installs dependencies only. GPU training readiness remains unverified until the
user runs the preflight in `TRAINING_RUNBOOK.md`.

The PowerShell launcher was checked against the actual WSL interpreter using a
print-only Python command, including an argument containing spaces. Parser and
CLI help checks passed. No model was constructed or optimized for those checks.

Eleven targeted tests passed in 4.010 seconds: six mocked GPU/CLI handoff checks
and five source-capsule roundtrip/integrity checks. These execute no optimizer
updates. The capsule fixture now includes the runbook, launcher and ownership
instructions, while continuing to exclude private and held-out epoch payloads.

`execute --require-gpu` fails before study reads or worker scheduling when no GPU
is visible. It does not establish kernel compatibility, memory fit or convergence.
The user still runs preflight before the capped GPU benchmark and full matrix.

The previous 103-test suite and CPU benchmark remain historical evidence under
`reports/compute_2026-11-01`; their files were not rewritten. The full suite
includes optimizer-based checks and was not rerun in this handoff. No new
research training, pretraining, fine-tuning, optimizer profile or held-out scoring
was launched. New study directories are required after the source changes.

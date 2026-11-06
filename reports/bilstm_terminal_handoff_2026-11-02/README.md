# User-run BiLSTM terminal handoff - 2 November 2026

The user now owns research training and wants to watch it in their own terminal.
The assistant stopped the verified background controller and its current
waveform-statistics worker. `stop.json` preserves the previous pipeline state
and stopped process IDs. The scheduler released its lock normally. Completed
pilot/benchmark artifacts and the source-training profile remain preserved.
No full-matrix optimizer training had started.

The old pilot has 18 completed jobs. The full-source capped M3/M6 benchmark has
five completed jobs and the disposable source-training profile completed with
saved artifacts unchanged. These checks establish readiness, not convergence.

`execute --live` streams complete worker output to the terminal and attempt log.
SSL and supervised training report the first/every-100th/final batch, metrics,
elapsed time and an approximate phase ETA. Ctrl+C cleans up that worker tree;
the scheduler preserves the interrupted attempt and releases its lock.
Trainer mathematical operations, model settings and sampler policy are unchanged.

Fourteen non-optimizer tests cover output/exit-code preservation, progress
cadence, interruption cleanup, CLI forwarding, worker logging configuration and
the existing mocked ownership guards. The child process in the tee test prints
strings only. No model was constructed, no optimizer updates were performed,
and the full optimizer-based regression suite was not rerun for this handoff.

A fresh frozen study is planned under `runs/bilstm_terminal_2026-11-02`, because
source logging code changed after the old freeze. It is uncapped and has 18
planned jobs. It has not been executed by the assistant. The user command is
in `docs/operations/TRAINING_RUNBOOK.md`. All managed runtime files remain on D:;
no package installation/update or real ISRUC/source-test prediction occurred.

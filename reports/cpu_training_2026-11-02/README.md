# Native CPU training with project runtime storage on D: - 2 November 2026

The user raised C: disk pressure after delegating training execution. The
WSL/GPU pipeline is stopped; its interrupted attempt is preserved. A process
inspection found no remaining project jobs and the optional RAM cache was
already absent. No global WSL shutdown, C: cleanup, image compaction or system
configuration change was performed.

`scripts/research_cpu.ps1` uses the existing native Windows environment on D:.
`runtime_validation.json` verifies its executable, temporary file creation,
cache variables, disabled bytecode writes, and inherited child-process temp
directory. Dependency consistency passes with no new installation. The launcher
restores its caller's environment after the child exits. Worker subprocesses
inherit the D: runtime paths. Explicitly supply D: paths for any custom outputs.
Windows paging/system writes can still use C:; this is not a system-wide disk
write sandbox and does not promise to reclaim previously allocated WSL storage.

Both synthetic CPU preflights pass. At batch 4, supervised updates took a median
0.03479 seconds; SSL batch 32 took 0.04204 seconds. At batch 32/SSL 128 those
medians were 0.29702/0.17950 seconds. Each measurement has two warm-up and five
timed steps; it excludes real data loading and cannot predict convergence or a
precise campaign duration. The original smaller batches are retained to limit
memory use. Host paging remains controlled by Windows.

The fresh uncapped source campaign is
`runs/training_execution_2026-11-02_cpu/source_matrix`, using
`configs/experiment_matrix.json`. Its outer `status.json`, study `ledger.json`,
and per-attempt logs establish actual progress. The pre-launch gate verifies
that every source file still matches the previously passed 110-test snapshot;
no source module or package was changed for the CPU transition. Every worker
also verifies the new study's frozen source/package constraints. Do not edit
source files or the environment while it runs.

The matrix schedules 18 jobs, seven conditions, training seed 42, labeled
budgets of 18/70 participants, and 15 validation participants. It retains ten
SSL epochs and up to 30 supervised epochs with patience five. Its earlier
five-job native CPU integration benchmark completed, but was capped and does
not establish convergence. This campaign can take days; it is not complete
merely because its scheduler started. No real ISRUC or new source-test
predictive scoring is scheduled. See `TRAINING_RUNBOOK.md` for manual commands
and recovery instructions.

Continuation: `progress_01.json` captures three completed preparation jobs with
all their artifact seals verified, followed by the running SSL job. Its scheduler
identity is checked against the process command line. During a five-second
sample its process tree used 21.34 CPU seconds and about 629 MiB resident memory;
this is a point-in-time observation, not a peak. Available host memory was about
3.06 GiB, with 129.8 GiB free on D: and 10.18 GiB on C:. Source hashes and package
versions still matched the study. No SSL epoch checkpoint had yet been published.
The new `scripts/training_status.py` reproduces these read-only checks without
importing TensorFlow or modifying the running campaign. No training result or
convergence claim follows from this activity sample.

`progress_02.json` additionally records CPU-only worker devices and another
successful source/package/artifact check. `monitor_validation.json` confirms
that a missing scheduler lock or an unrelated live PID is not reported as a
live scheduler. `monitor_cli_validation.json` records rejection of report
overwrites and reports inside the frozen study, including exit codes. These
checks perform no optimizer updates. The original regression suite was not
rerun while the campaign was active; no source modules or packages changed.

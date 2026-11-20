# Recovery runtime benchmark review

The user completed the runtime-only M6 benchmark on 20 November. The measured
training call took 3,568.364 seconds (59.47 minutes); total worker time was
3,572.083 seconds (59.53 minutes). Peak sampled host process RSS was 664.42 MiB,
305.70 MiB above the sampling-start RSS. These measurements cover native Windows
CPU execution and do not measure GPU VRAM, whole-system memory or temperature.

The run completed one capped epoch with 64 optimizer updates, full epoch-end
validation on 65,133 targets, recovery/best-model saving, reload and another full
validation pass for the final prediction export. Source packages/hashes, data
scope, initialization artifacts, model bundle seals, epoch-zero/one recovery
seals, optimizer counts and exported target identities all verify. The saved
probabilities reproduce the exported summary metrics. `inspect_runtime.py`
records these checks in `verification.json` without TensorFlow imports or model
execution.

The epoch-zero state was captured at 45.718 seconds and epoch one at 1,811.751
seconds. The completed training summary records another 1,755.850 seconds after
that capture. These boundaries group several operations and do not separately
time the optimizer or validation. The 59.47-minute result is not an uncapped
epoch duration, an SSL benchmark or evidence of a speed improvement.

The PowerShell transcript contains its header/footer but did not capture the
child process's native terminal output. The earlier claim that this transcript
would retain the terminal output was incorrect. JSON, recovery, history and
prediction evidence remain intact, so the benchmark need not be repeated for
this review. Future matrix execution uses `execute --live`, whose worker logs
capture stdout/stderr directly; the incomplete transcript is preserved.

## Replication scope and compute decision

Correctness checks now support the tested CPU eager path: all fourteen saved
models passed full-cohort inference parity, the eighteen-job recovery pilot
completed, and the capped real-data M6 replay matched uninterrupted state and
predictions exactly. Compiled validation remains unaccepted. The next decision
is compute allocation for replication, rather than another identical benchmark.

The [draft protocol](../../docs/REPLICATION_PROTOCOL_DRAFT.md) and
`configs/sleepedf/experiments/experiment_bilstm_replication_draft.json` retain
the full seven-model family, both label budgets and draw seed 42, with optimization
seeds 17, 42 and 123. Seed 42 is rerun under the same recovery-enabled training
procedure as the other seeds. The old seed-42 results remain separate exploratory
evidence; inference parity and checkpoint recovery do not establish equivalence
between the old implicit sampler and new explicit epoch sampler.

This scope gives 48 jobs: 42 supervised fits, three SSL fits, two budget
normalizers and one waveform-statistics job. `prepare_replication_draft.py`
reproduces the recipe and counts without TensorFlow imports, study creation or
training. The recipe is CPU-compatible and remains a draft pending the user's
CPU/GPU resource preference. No replication study directory has been frozen.

The historical single-seed matrix recorded 118.02 hours across classifier and
SSL summaries. Multiplying that by three gives 354.07 hours (14.75 days of
continuous execution), but this is arithmetic context, not a new-version runtime
forecast. Seed-specific stopping, code paths, thermal conditions and hardware
can change it. `replication_draft.json` also contains illustrative validation-only
capacity scenarios using the measured M3 eager pass; those scenarios assume
homogeneous model costs and exclude training/SSL/preparation. They are neither
guarantees nor lower bounds.

A cloud/GPU route would require its own environment and kernel/recovery/parity
checks before a fresh study freeze. CPU host RSS does not establish that a GPU
fits within 4 GiB, and CPU recovery states do not migrate as exact GPU resumes.
No upload, account setup, purchase, WSL change or training was performed here.

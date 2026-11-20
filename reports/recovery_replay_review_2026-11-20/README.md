# Real data M6 recovery replay review

The M6 replay from the pilot's sealed epoch-one checkpoint matches the
uninterrupted two-epoch pilot exactly. All 161 model, optimizer, best-weight and
RNG arrays have zero difference. Callback state, Python/NumPy RNG state and
training history are identical. Both runs finish at four optimizer steps,
select epoch one as the best model and export exactly equal validation
probabilities on the same eight targets. The selected model files also have
identical SHA256 hashes.

`inspect_replay.py` verifies the original job seal, data/source/package scope,
recovery contracts and retained outputs without model execution. It uses the
previously declared synthetic-gate float tolerances of `atol=rtol=1e-7`, exact
integer comparisons and identical non-timing state fields. Observed differences
are zero, stricter than the float limits. Wall-clock fields are excluded.
`verification.json` records individual array comparisons and replay file hashes.

This passes epoch-boundary recovery for the tested SSL-initialized M6 BiLSTM,
seed 42, two capped real-data epochs on the same native Windows CPU environment.
It does not establish full-schedule or cross-hardware recovery. The earlier
synthetic test separately exercised LR and early-stop events. The original
pilot, full seed-42 research study and all inference evidence remain preserved.
Compiled validation remains unaccepted.

The terminal's epoch-two validation loss can differ from the summary's exported
loss: export uses the best model selected by validation macro-F1. M6's macro-F1
ties across the two epochs, so epoch one remains selected even though epoch
two's validation loss is lower. This is the declared checkpoint policy.

## Runtime benchmark before replication

The remaining compute question is how long the recovery-enabled eager pipeline
takes when it includes full source validation and checkpoint publication.
`prepare_runtime.py` freezes `runtime_spec.json` using the completed original
full-budget normalizer and ten-epoch SSL encoder. It verifies their artifact
seals and the 70-training/15-validation participant scope. Historical checkpoints
supply initialization only; old optimizer or recovery state is not resumed.

The new benchmark uses one fresh M6 run, eager validation at batch four, epoch
recovery, 64 capped training batches and all 65,133 source-validation targets.
It performs 64 optimizer updates. The trainer evaluates validation once at the
epoch boundary and again when exporting predictions from the best model: two
full passes, or 130,266 validation evaluations. Based on the earlier 29.4-minute
eager pass, these may take roughly an hour; this is an estimate, not a benchmark
result. The training pass is capped and this does not time SSL or a complete
uncapped supervised epoch.

Run this command in PowerShell from the repository root:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File reports/recovery_replay_review_2026-11-20/run_runtime_benchmark.ps1
```

The launcher uses `scripts/research_cpu.ps1`, retains a terminal transcript and
writes a fresh output at `runs/bilstm_recovery_runtime_2026-11-20`. It prints
training progress every sixteen batches and elapsed-time/RSS updates every
thirty seconds. `benchmark.json` records training-call and total worker time,
plus sampled host RSS. Host memory sampling begins after TensorFlow import and
includes model construction through export; it is not whole-system memory.
All checkpoint and cache files managed by the project remain on D:.

Only `-InspectOnly` was run by the assistant. That mode verifies frozen inputs,
packages and source hashes without importing TensorFlow or creating a training
output. No runtime result or speed improvement is claimed yet. The run is for
compute measurement, and its capped training scores are not research results.
After it finishes, send the final benchmark JSON before freezing an uncapped
multi-seed replication schedule. Full-family eager inference parity and this
M6 recovery result do not justify pooling old and new training paths as identical.

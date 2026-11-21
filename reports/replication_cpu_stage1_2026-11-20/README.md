# CPU replication first stage review

The user completed the first four of forty-eight frozen jobs: waveform
statistics, both budget normalizers and full SSL pretraining for seed 17.
All four completed-job seals, current source/package and source-data checks,
frozen protocol hashes and the full installed-package snapshot verify.

SSL completed ten epochs, reporting 101,060 optimizer updates across 10,106
batches per epoch. Its reported duration was 5,280.679 seconds (88.01 minutes).
The encoder reload difference reported by the worker is zero. The completed
pretraining bundle and final epoch-ten checkpoint file hashes verify, and the
ten-row history is finite and consistent with the final summary. Binary
optimizer checkpoint tensors were not independently decoded in this review.

The SSL manifest contains exactly the frozen source-training pool: seventy
participants, 138 recordings and 323,423 quality epochs. No validation or held-out
recording enters that pool. Both labeled-budget normalizers use the same full
source waveform scaling, with spectral fits restricted to their planned labeled
participants. No supervised jobs have started, and no source-test or ISRUC
predictive scoring was opened. SSL retrieval/embedding diagnostics do not
establish downstream sleep-stage performance or superiority.

`inspect_stage.py` reproduces the metadata, seal, scope, normalization and history
checks without TensorFlow imports, model execution or optimizer updates.
`verification.json` records hashes and the proposed next stage. The native
worker log contains epoch-ten progress and completion output, unlike the earlier
runtime-only PowerShell transcript.

## Next user run

Run this once in PowerShell from the repository root:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/research_cpu.ps1 -m src.experiments.run execute --study-dir runs/bilstm_replication_cpu_2026-11-20 --live --max-jobs 2
```

The runner verifies/skips the four completed jobs, trains fresh SSL seeds 42 and
123 for ten epochs each, and stops before supervised jobs. It does not reuse the
historical seed-42 encoder. Successful completion should report forty-eight
planned, six completed and two executed in this call, with `limited_run: false`.
Report that summary before advancing further; repeating the command schedules
two additional jobs each time.

Keep source, packages, data and frozen manifests unchanged. Live worker logs
remain under this study on D:. If either SSL job is interrupted, preserve its
attempt/checkpoints and request inspection before retrying; the supervised
recovery flag does not provide scheduler-level SSL resume.

# CPU replication SSL completion review

All six preparation and SSL jobs have completed. On 22 November the review
verified all six job seals, the source/package and data-scope freeze, the
archived protocol hashes, both budget normalizers and all three completed
SSL bundles and final checkpoint file hashes. No supervised fit has started.

| SSL seed | Epochs | Recorded optimizer steps | Reported minutes | Reported encoder reload difference |
|---|---:|---:|---:|---:|
| 17 | 10 | 101,060 | 88.01 | 0 |
| 42 | 10 | 101,060 | 98.05 | 0 |
| 123 | 10 | 101,060 | 89.68 | 0 |

Each fresh fit uses the same frozen seventy-participant, 138-recording training
pool with 323,423 quality epochs and the same waveform normalization. The
resolved optimization seed, model, augmentation and uncapped schedule match
the study. Histories contain ten finite, ordered rows and match their final
summaries. Native worker logs contain epoch-ten completion output. The seed-42
encoder is a new fit in this study, not a substituted historical artifact.

`inspect_stage.py` reproduces these checks without TensorFlow imports, model
execution or optimizer updates. `verification.json` records seals, source scope,
seed-specific encoder hashes and the next supervised block. Binary optimizer
checkpoint tensors were not independently decoded; counts above are the sealed
producer summaries, consistent with ten epochs of 10,106 batches. SSL diagnostic
loss/retrieval values do not establish downstream staging accuracy or superiority.
Source-test and ISRUC predictive scoring remain unopened.

## First supervised block

Run this once from the repository root in PowerShell:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/research_cpu.ps1 -m src.experiments.run execute --study-dir runs/bilstm_replication_cpu_2026-11-20 --live --max-jobs 7
```

The runner verifies/skips the completed jobs and trains M0 through M6 for
`draw42_n18`, optimization seed 17. The scratch and SSL conditions share the
same normalizer and full source-validation targets; M4/M5/M6 initialize from
this study's completed SSL seed 17. Each fit follows the frozen uncapped schedule
of up to thirty epochs with early stopping, epoch recovery and eager batch-four
validation. This command stops before seed-42 supervised fits.

Successful completion should report forty-eight planned jobs, thirteen completed,
seven executed in this call and `limited_run: false`. Send the summary for
inspection before repeating or advancing. Each invocation's `--max-jobs` limit
counts additional jobs; it is not a lifetime total.

This is a long block. The historical seven-model 18-participant/seed-42 block
recorded 33.39 hours under the older procedure. Seed-specific stopping, current
runtime and hardware conditions can change the new duration; that historical
number is not a forecast. Keep source, environment, data and frozen manifests
unchanged. If interrupted, preserve the lock, attempts, logs and recovery states
and request inspection before the explicit supervised retry/resume procedure.
Do not delete locks or modify seals. The assistant has not started this block.

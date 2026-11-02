# Configuration index

| Directory | Purpose |
|---|---|
| `sleepedf/models/` | Existing raw, concatenation, gated, spectral and linear-probe models |
| `sleepedf/pretraining/` | Raw-EEG contrastive SSL |
| `sleepedf/experiments/` | Earlier source-only pilot, benchmark and full experiment matrices |
| `sleepedf/evaluation/` | Source-validation evaluator integration specification |
| `compute/gpu/` | Retained GPU batch/benchmark recipes |
| `deferred/cross_dataset/` | ISRUC evaluation policy for the later cross-dataset phase |

Configuration contents were preserved during relocation. The existing fusion
recipes still use GRU heads; no BiLSTM setting is implied by the new location.
The stopped study keeps its own resolved configuration and old path history.
New CLI calls should use these organized paths, for example:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/research_cpu.ps1 -m src.experiments.run plan --config configs/sleepedf/experiments/experiment_pilot.json --study-dir runs/my_new_source_pilot
```

This plans the existing GRU pilot. The future BiLSTM configurations must be added
and validated explicitly before a new paper campaign. Run commands from the
checkout root and use fresh study/output directories.

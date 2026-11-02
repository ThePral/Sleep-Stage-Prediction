# Configuration index

| Directory | Purpose |
|---|---|
| `sleepedf/models/` | Existing raw, concatenation, gated, spectral and linear-probe models |
| `sleepedf/pretraining/` | Raw-EEG contrastive SSL |
| `sleepedf/experiments/` | Earlier source-only pilot, benchmark and full experiment matrices |
| `sleepedf/evaluation/` | Source-validation evaluator integration specification |
| `compute/gpu/` | Retained GPU batch/benchmark recipes |
| `deferred/cross_dataset/` | ISRUC evaluation policy for the later cross-dataset phase |

Configuration contents were preserved during relocation. New `*_bilstm.json`
model recipes and `experiment_bilstm_*.json` studies explicitly select BiLSTM.
The earlier recipes retain their legacy heads.
The stopped study keeps its own resolved configuration and old path history.
New CLI calls should use these organized paths, for example:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/research_cpu.ps1 -m src.experiments.run plan --config configs/sleepedf/experiments/experiment_pilot.json --study-dir runs/my_new_source_pilot
```

This plans the historical GRU pilot. Use
`configs/sleepedf/experiments/experiment_bilstm_pilot.json` for the new BiLSTM
integration workflow, followed by its source benchmark and full matrix. See
[the implementation guide](../docs/SLEEPEDF_BILSTM.md). Run commands from the
checkout root and use fresh study/output directories.

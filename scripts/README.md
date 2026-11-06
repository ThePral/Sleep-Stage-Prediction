# Operational scripts

Run scripts from the checkout root.

| Script | Purpose |
|---|---|
| `research_cpu.ps1` | Native Windows CPU Python with project-managed runtime files on D: |
| `training_status.py` | Read-only frozen-study/process/checkpoint monitor |
| `repair_sleepedf_access.ps1` | Back up prepared Sleep-EDF access rules and grant the checkout owner read access; inspection only unless `-Apply` is supplied |
| `research.ps1` | Retained WSL launcher for historical GPU reproduction |
| `configure_wsl_cuda.py` | Retained virtual-environment CUDA library discovery helper |
| `cache_training_waveforms.py` | Optional historical Linux shared-memory cache helper; not used for the current CPU route |
| `audits/cross_dataset/audit_sleepedf_isruc_v3.py` | Latest standalone dataset-compatibility audit |
| `audits/cross_dataset/audit_sleep_dataset_compatibility*.py` | Earlier audit versions, preserved for provenance |

The audit tools accept project-root-relative input paths as before. For CLI
usage without opening data:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/research_cpu.ps1 scripts/audits/cross_dataset/audit_sleepedf_isruc_v3.py --help
```

The cross-dataset phase is deferred. These tools are retained for later review;
organization did not execute real ISRUC predictive scoring.

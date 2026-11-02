# Environment specifications

| File | Purpose |
|---|---|
| `requirements-research-common.txt` | Pinned direct research dependencies shared by the validated environments |
| `requirements-windows-py313.lock.txt` | Full snapshot of the validated native Windows/Python 3.13 environment |
| `requirements-linux-gpu.txt` | Linux/WSL NVIDIA framework recipe; includes the neighboring common requirements and constraints |
| `constraints-cuda125-pascal.txt` | CUDA dependency constraints for the earlier Pascal GPU setup |

Only specification files moved. Their contents are unchanged. The installed
Windows `.venv` stays on D: at the checkout root; the earlier Linux environment
stays under its recorded `runs/compute_2026-10-31` path. No packages were installed
or updated. The root `requirements.txt` remains the original broad dependency list.

Commands are run from the checkout root, for example:

```powershell
python -m pip install -r environments/requirements-linux-gpu.txt
```

That example describes the retained Linux recipe, not the current Windows CPU
environment. Use the CPU launcher and the
[compute guide](../docs/operations/COMPUTE_GUIDE.md) for the intended platform.
Do not reinstall dependencies during a frozen campaign. Source-only capsules
retain legacy member-name support for archives created before this move.

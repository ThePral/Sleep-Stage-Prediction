# Forward Python arguments into the existing project-local WSL environment.
# Examples: .\scripts\research.ps1 -m pip check
#           .\scripts\research.ps1 -m src.experiments.run --help
$ErrorActionPreference = 'Stop'
if ($args.Count -eq 0) {
    throw 'Supply Python arguments, for example: .\scripts\research.ps1 -m pip check'
}
$projectRoot = Split-Path -Parent $PSScriptRoot
# Windows PowerShell can turn redirected native stderr into terminating errors.
# Keep WSL diagnostics visible, and use its exit code as the process result.
$ErrorActionPreference = 'Continue'
$wslRoot = (& wsl.exe --distribution Ubuntu-24.04 --exec wslpath -a -u $projectRoot | Out-String).Trim()
$pathExitCode = $LASTEXITCODE
$ErrorActionPreference = 'Stop'
if ($pathExitCode -ne 0 -or -not $wslRoot.StartsWith('/')) {
    throw 'Cannot resolve this checkout inside Ubuntu-24.04. Check WSL before continuing.'
}
$runRoot = "$wslRoot/runs/compute_2026-10-31"
$pythonLinux = "$runRoot/linux_env/bin/python"
$venvLib = Join-Path $projectRoot 'runs/compute_2026-10-31/linux_env/lib'
$cudaLibraries = @()
if (Test-Path -LiteralPath $venvLib) {
    $pythonLibraries = @(Get-ChildItem -LiteralPath $venvLib -Directory | Where-Object { $_.Name -match '^python\d+\.\d+$' })
    if ($pythonLibraries.Count -gt 1) {
        throw 'The WSL environment has multiple Python library layouts. Use a fresh environment.'
    }
    if ($pythonLibraries.Count -eq 1) {
        $nvidiaRoot = Join-Path $pythonLibraries[0].FullName 'site-packages/nvidia'
        if (Test-Path -LiteralPath $nvidiaRoot) {
            foreach ($package in Get-ChildItem -LiteralPath $nvidiaRoot -Directory | Sort-Object Name) {
                if (Test-Path -LiteralPath (Join-Path $package.FullName 'lib')) {
                    $cudaLibraries += "$runRoot/linux_env/lib/$($pythonLibraries[0].Name)/site-packages/nvidia/$($package.Name)/lib"
                }
            }
        }
    }
}
# Process-local library discovery; no ldconfig, driver or system-path changes.
$cudaLibraryPath = ($cudaLibraries + @('/usr/lib/wsl/lib')) -join ':'
$ErrorActionPreference = 'Continue'
& wsl.exe --distribution Ubuntu-24.04 --cd $wslRoot --exec env `
    "LD_LIBRARY_PATH=$cudaLibraryPath" `
    "PYTHONDONTWRITEBYTECODE=1" "TF_FORCE_GPU_ALLOW_GROWTH=true" `
    "MPLCONFIGDIR=$runRoot/matplotlib-cache" "XDG_CACHE_HOME=$runRoot/xdg-cache" `
    "KERAS_HOME=$runRoot/keras-cache" "CUDA_CACHE_PATH=$runRoot/cuda-cache" `
    "TRITON_CACHE_DIR=$runRoot/triton-cache" `
    $pythonLinux -u -X utf8 @args
exit $LASTEXITCODE

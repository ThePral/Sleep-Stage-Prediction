# Native Windows CPU training with project-managed runtime files on D:.
# Example: .\scripts\research_cpu.ps1 -m src.experiments.run --help
$ErrorActionPreference = 'Stop'
if ($args.Count -eq 0) {
    throw 'Supply Python arguments, for example: .\scripts\research_cpu.ps1 -m pip check'
}
$projectRoot = Split-Path -Parent $PSScriptRoot
if ([System.IO.Path]::GetPathRoot($projectRoot) -ine 'D:\') {
    throw 'This CPU launcher requires the checkout and its runtime directories on D:.'
}
$pythonWindows = Join-Path $projectRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonWindows -PathType Leaf)) {
    throw 'The existing project-local Windows Python environment was not found.'
}
$runtimeRoot = Join-Path $projectRoot 'runs/cpu_runtime'
$settings = [ordered]@{
    CUDA_VISIBLE_DEVICES = '-1'
    PYTHONDONTWRITEBYTECODE = '1'
    TEMP = (Join-Path $runtimeRoot 'temp')
    TMP = (Join-Path $runtimeRoot 'temp')
    TMPDIR = (Join-Path $runtimeRoot 'temp')
    PIP_CACHE_DIR = (Join-Path $runtimeRoot 'pip-cache')
    KERAS_HOME = (Join-Path $runtimeRoot 'keras-cache')
    MPLCONFIGDIR = (Join-Path $runtimeRoot 'matplotlib-cache')
    XDG_CACHE_HOME = (Join-Path $runtimeRoot 'xdg-cache')
    CUDA_CACHE_PATH = (Join-Path $runtimeRoot 'cuda-cache')
    TRITON_CACHE_DIR = (Join-Path $runtimeRoot 'triton-cache')
    JOBLIB_TEMP_FOLDER = (Join-Path $runtimeRoot 'joblib-temp')
    NUMBA_CACHE_DIR = (Join-Path $runtimeRoot 'numba-cache')
}
$previous = @{}
$pythonExitCode = 1
foreach ($name in $settings.Keys) {
    $previous[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
}
Push-Location -LiteralPath $projectRoot
try {
    foreach ($name in $settings.Keys) {
        $value = $settings[$name]
        if ($name -notin @('CUDA_VISIBLE_DEVICES', 'PYTHONDONTWRITEBYTECODE')) {
            [System.IO.Directory]::CreateDirectory($value) | Out-Null
        }
        [Environment]::SetEnvironmentVariable($name, $value, 'Process')
    }
    # Native stderr is diagnostic output, not a PowerShell exception.
    $ErrorActionPreference = 'Continue'
    & $pythonWindows -u -X utf8 @args
    $pythonExitCode = $LASTEXITCODE
}
finally {
    foreach ($name in $settings.Keys) {
        [Environment]::SetEnvironmentVariable($name, $previous[$name], 'Process')
    }
    Pop-Location
}
exit $pythonExitCode

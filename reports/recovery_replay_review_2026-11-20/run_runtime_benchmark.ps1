# USER-RUN: runtime-only training benchmark; retains terminal output on D:.
[CmdletBinding()]
param([switch]$InspectOnly)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$pythonArguments = @((Join-Path $PSScriptRoot 'run_runtime_benchmark.py'))
if ($InspectOnly) { $pythonArguments += '--inspect-only' }
$transcriptStarted = $false
$benchmarkExitCode = 1
try {
    if (-not $InspectOnly) {
        Start-Transcript -LiteralPath (Join-Path $PSScriptRoot 'runtime_terminal.log') -NoClobber | Out-Null
        $transcriptStarted = $true
    }
    $ErrorActionPreference = 'Continue'
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $projectRoot 'scripts/research_cpu.ps1') @pythonArguments
    $benchmarkExitCode = $LASTEXITCODE
}
finally {
    if ($transcriptStarted) { Stop-Transcript | Out-Null }
}
exit $benchmarkExitCode

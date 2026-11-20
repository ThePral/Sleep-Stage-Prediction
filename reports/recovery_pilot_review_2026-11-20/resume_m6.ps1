# USER-RUN: two optimizer updates, resuming M6's preserved epoch-one generation.
[CmdletBinding()]
param([switch]$InspectOnly)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$handoff = Get-Content -LiteralPath (Join-Path $PSScriptRoot 'resume_handoff.json') -Raw | ConvertFrom-Json
if (Test-Path -LiteralPath $handoff.resume_output) {
    throw 'The replay output already exists; preserve it and review before preparing a new replay.'
}
if (-not (Test-Path -LiteralPath (Join-Path $handoff.epoch_one_generation 'seal.json') -PathType Leaf)) {
    throw 'The sealed epoch-one source is missing.'
}
$pythonArguments = @(
    '-m', 'src.training.supervised',
    '--config', $handoff.config,
    '--run-dir', $handoff.resume_output,
    '--data-root', $handoff.data_root,
    '--split-file', $handoff.split_file,
    '--max-train-batches', '2', '--max-validation-batches', '2',
    '--normalizer', $handoff.normalizer,
    '--pretrained', $handoff.pretrained,
    '--budget-plan', $handoff.budget_plan,
    '--budget-id', $handoff.budget_id,
    '--resume-from', $handoff.epoch_one_generation
)
if ($InspectOnly) {
    $pythonArguments | ConvertTo-Json
    return
}
$transcript = Join-Path $PSScriptRoot 'm6_resume_terminal.log'
Start-Transcript -LiteralPath $transcript -NoClobber | Out-Null
$replayExitCode = 1
try {
    # A separate PowerShell process retains the production CPU launcher's setup.
    # Its native stderr remains visible diagnostics rather than an exception.
    $ErrorActionPreference = 'Continue'
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $projectRoot 'scripts/research_cpu.ps1') @pythonArguments
    $replayExitCode = $LASTEXITCODE
}
finally {
    Stop-Transcript | Out-Null
}
exit $replayExitCode

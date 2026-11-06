# Repair read access for the checkout owner without changing prepared data bytes.
[CmdletBinding()]
param(
    [switch]$Apply,
    [string]$Account,
    [string]$ReportDirectory
)

$ErrorActionPreference = 'Stop'
$projectRoot = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$storeRoot = [IO.Path]::GetFullPath((Join-Path $projectRoot 'data/processed/research_v1/sleepedf'))
if (-not $storeRoot.StartsWith($projectRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'The prepared Sleep-EDF directory must stay inside this checkout.'
}
if ([string]::IsNullOrWhiteSpace($Account)) {
    $Account = (Get-Acl -LiteralPath $projectRoot).Owner
}
if ($Account -match 'CodexSandbox') {
    throw 'Specify the normal terminal account, not a sandbox account.'
}
$identity = New-Object Security.Principal.NTAccount($Account)
$accountSid = $identity.Translate([Security.Principal.SecurityIdentifier])
$nodes = @(Get-Item -LiteralPath $storeRoot -Force) + @(Get-ChildItem -LiteralPath $storeRoot -Force -Recurse)
foreach ($node in $nodes) {
    if (($node.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
        throw "Refusing to change permissions through a link: $($node.FullName)"
    }
}
Write-Output "Prepared Sleep-EDF scope: $($nodes.Count) objects; read account: $Account ($accountSid)"
if (-not $Apply) {
    Write-Output 'Inspection only. Add -Apply to back up access rules and grant read access.'
    return
}
if ([string]::IsNullOrWhiteSpace($ReportDirectory)) {
    $ReportDirectory = Join-Path $projectRoot ('reports/sleepedf_access_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
}
$reportRoot = [IO.Path]::GetFullPath($ReportDirectory)
if (-not $reportRoot.StartsWith($projectRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase) -or
    $reportRoot.StartsWith($storeRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase) -or
    $reportRoot -eq $storeRoot) {
    throw 'Save the access backup inside the checkout and outside the prepared stores.'
}
if (Test-Path -LiteralPath $reportRoot) { throw 'Use a new report directory to preserve previous evidence.' }
[IO.Directory]::CreateDirectory($reportRoot) | Out-Null
$before = @(
    foreach ($node in $nodes) {
        $acl = Get-Acl -LiteralPath $node.FullName
        [ordered]@{
            path = $node.FullName
            owner = $acl.Owner
            access_sddl = $acl.GetSecurityDescriptorSddlForm([Security.AccessControl.AccessControlSections]::Access)
            is_directory = [bool]$node.PSIsContainer
            length = $(if ($node.PSIsContainer) { $null } else { $node.Length })
            last_write_utc_ticks = $node.LastWriteTimeUtc.Ticks
        }
    }
)
$before | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $reportRoot 'access_before.json') -Encoding UTF8
Write-Output "Access-rule backup saved. Granting read access on $($nodes.Count) objects."
$changed = 0
foreach ($node in $nodes) {
    # icacls updates only the DACL. Set-Acl can also request security privileges
    # when round-tripping these protected security descriptors.
    $rights = if ($node.PSIsContainer) { 'RX' } else { 'R' }
    & "$env:SystemRoot/System32/icacls.exe" $node.FullName /grant "*$($accountSid.Value):($rights)" | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Read-access grant failed: $($node.FullName)" }
    $changed++
}
$verified = 0
foreach ($item in $before) {
    $node = Get-Item -LiteralPath $item.path -Force
    if (-not $item.is_directory -and ($node.Length -ne $item.length -or $node.LastWriteTimeUtc.Ticks -ne $item.last_write_utc_ticks)) {
        throw "File size or content-write time changed: $($item.path)"
    }
    $acl = Get-Acl -LiteralPath $item.path
    $rights = if ($item.is_directory) { [Security.AccessControl.FileSystemRights]::ReadAndExecute } else { [Security.AccessControl.FileSystemRights]::Read }
    $allowed = @($acl.GetAccessRules($true, $false, [Security.Principal.SecurityIdentifier]) | Where-Object {
        $_.IdentityReference -eq $accountSid -and $_.AccessControlType -eq [Security.AccessControl.AccessControlType]::Allow -and
        ($_.FileSystemRights -band $rights) -eq $rights
    })
    if ($allowed.Count -eq 0 -or $acl.Owner -ne $item.owner) { throw "Access or owner verification failed: $($item.path)" }
    $verified++
}
[ordered]@{
    account = $Account
    account_sid = $accountSid.Value
    store_root = $storeRoot
    objects_updated = $changed
    objects_verified = $verified
    owners_preserved = $true
    file_sizes_and_content_write_times_preserved = $true
    permission_scope = 'Explicit read for files; read and traverse for directories; existing access rules retained.'
    training_launched = $false
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $reportRoot 'repair.json') -Encoding UTF8
Write-Output "Verified read access for $Account on $verified objects. No training started."

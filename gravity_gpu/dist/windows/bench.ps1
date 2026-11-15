<#
.SYNOPSIS
    Runs the gsim_app GPU benchmark sweep on Windows and archives the results.
.DESCRIPTION
    Invokes gsim_app.exe --bench with the given flags (full ladder: legacy
    citations, autotune, CPU-parity, naive + tiled rungs, N-scaling sweep),
    then archives the JSON summary as telemetry/windows_<timestamp>.json.

    The .exe itself is produced by Stage 4 (cargo-xwin cross-compilation to
    x86_64-pc-windows-msvc); this script is complete and only needs the
    binary next to it (or at -Exe). Release binaries are timestamped
    (gsim_app_yyyyMMdd_HHmmss.exe) so browsers never serve a cached stale
    copy; when -Exe is not given explicitly, the newest timestamped binary
    in the script folder is picked automatically. Copy the resulting
    telemetry/windows_<timestamp>.json back for ingestion into the project
    telemetry.
.EXAMPLE
    .\bench.ps1
    .\bench.ps1 -Backend dx12 -Adapter 0 -N 10000 -Steps 150
    .\bench.ps1 -Exe C:\gsim\gsim_app.exe -Sweep "1000,5000,10000,20000" -SweepSteps 30
#>
param(
    [string]$Exe = (Join-Path $PSScriptRoot "gsim_app.exe"),
    [int]$Seed = 12345,
    [int]$N = 10000,
    [ValidateSet("vulkan", "dx12")][string]$Backend = "dx12",
    [int]$Adapter = 0,
    [int]$Steps = 150,
    [string]$Sweep = "1000,2000,5000,10000,20000,50000,100000,200000",
    [int]$SweepSteps = 60,
    [ValidateSet("naive", "tiled")][string]$Mode = "tiled",
    [ValidateSet("parallel", "serial")][string]$MergeMode = "parallel",
    [string]$OutDir = (Join-Path $PSScriptRoot "telemetry"),
    [string]$Telemetry = ""
)

$ErrorActionPreference = "Stop"

# Release binaries are timestamped (gsim_app_yyyyMMdd_HHmmss.exe) so a fresh
# download never collides with a browser-cached copy. Unless -Exe was given
# explicitly, prefer the newest timestamped binary next to this script.
if (-not $PSBoundParameters.ContainsKey('Exe')) {
    $stamped = Get-ChildItem -Path $PSScriptRoot -Filter "gsim_app_*.exe" -File |
        Sort-Object Name -Descending | Select-Object -First 1
    if ($stamped) { $Exe = $stamped.FullName }
}

if (-not (Test-Path $Exe)) {
    throw "gsim_app binary not found at '$Exe'. Copy the newest gsim_app_<timestamp>.exe next to bench.ps1, or pass -Exe <path>."
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$tmpJson = Join-Path $OutDir "bench_tmp_$stamp.json"
$finalJson = Join-Path $OutDir "windows_$stamp.json"

$exeArgs = @(
    "--bench",
    "--seed", "$Seed",
    "--n", "$N",
    "--backend", $Backend,
    "--adapter", "$Adapter",
    "--steps", "$Steps",
    "--sweep", $Sweep,
    "--sweep-steps", "$SweepSteps",
    "--mode", $Mode,
    "--merge-mode", $MergeMode,
    "--dump", $tmpJson
)
if ($Telemetry -ne "") {
    $exeArgs += @("--telemetry", $Telemetry)
}

Write-Host "== gsim Windows benchmark =="
Write-Host "exe:      $Exe"
Write-Host "backend:  $Backend   adapter: $Adapter"
Write-Host "N:        $N   steps: $Steps   seed: $Seed"
Write-Host "sweep:    $Sweep (steps/point: $SweepSteps)"
Write-Host "mode:     $Mode / $MergeMode"
Write-Host ""

$sw = [System.Diagnostics.Stopwatch]::StartNew()
& $Exe @exeArgs
$code = $LASTEXITCODE
$sw.Stop()
if ($code -ne 0) {
    throw "gsim_app.exe exited with code $code"
}
Write-Host ""
Write-Host "benchmark finished in $($sw.Elapsed.ToString())"

if (-not (Test-Path $tmpJson)) {
    throw "expected summary JSON at '$tmpJson' was not produced"
}
$summary = Get-Content $tmpJson -Raw | ConvertFrom-Json

# Sanity: the summary must carry adapter identity + all stages.
foreach ($field in @("adapter", "parity_ok", "rungs", "sweep")) {
    if ($null -eq $summary.$field) { throw "summary JSON is missing '$field'" }
}
if ($null -eq $summary.adapter.name) { throw "summary JSON is missing adapter.name" }

$summary | Add-Member -NotePropertyName "collected_by" -NotePropertyValue "dist/windows/bench.ps1" -Force
$summary | Add-Member -NotePropertyName "collected_at" -NotePropertyValue $stamp -Force
$summary | Add-Member -NotePropertyName "host" -NotePropertyValue $env:COMPUTERNAME -Force
$summary | Add-Member -NotePropertyName "os" -NotePropertyValue ([System.Environment]::OSVersion.VersionString) -Force

$summary | ConvertTo-Json -Depth 12 | Set-Content $finalJson -Encoding utf8
Remove-Item $tmpJson -Force

Write-Host ""
Write-Host "== results =="
Write-Host ("adapter : {0} [{1}/{2}] driver: {3}" -f $summary.adapter.name, $summary.adapter.type, $summary.adapter.backend, $summary.adapter.driver)
Write-Host ("parity  : {0}" -f $summary.parity_ok)
foreach ($rung in $summary.rungs) {
    Write-Host ("rung {0,-6}: {1,10:N2} steps/s  (N={2}, steps={3}, final={4})" -f $rung.mode, $rung.steps_per_sec, $rung.n, $rung.steps, $rung.final_count)
}
Write-Host ("sweep max N @60fps: 1x={0}  4x={1}" -f $summary.max_n_at_60fps_1x, $summary.max_n_at_60fps_4x)
Write-Host ""
Write-Host "archived: $finalJson"

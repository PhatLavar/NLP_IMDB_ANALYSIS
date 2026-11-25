param([ValidateSet('tectonic', 'xelatex')][string]$Engine = 'tectonic')
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    python tools/prepare_report.py
    if ($LASTEXITCODE -ne 0) { throw 'Report evidence checks failed.' }
    $buildDir = Join-Path $PSScriptRoot 'build'
    New-Item -ItemType Directory -Force -Path $buildDir | Out-Null
    if ($Engine -eq 'tectonic') {
        tectonic main.tex --outdir $buildDir --keep-logs --keep-intermediates
        if ($LASTEXITCODE -ne 0) { throw 'Tectonic failed.' }
    } else {
        1..3 | ForEach-Object {
            xelatex --disable-installer -no-shell-escape -interaction=nonstopmode -halt-on-error -output-directory=$buildDir main.tex
            if ($LASTEXITCODE -ne 0) { throw 'XeLaTeX failed.' }
        }
    }
    $reportDir = Join-Path (Split-Path $PSScriptRoot -Parent) '.report'
    New-Item -ItemType Directory -Force -Path $reportDir | Out-Null
    $destination = Join-Path $reportDir 'NLP_CC01_Group04_Report.pdf'
    Copy-Item -LiteralPath (Join-Path $buildDir 'main.pdf') -Destination $destination -Force
    Write-Output "Built: $destination"
} finally {
    Pop-Location
}

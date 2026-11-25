param([string]$Video)

$ErrorActionPreference = 'Stop'
$trackerPython = Join-Path $PSScriptRoot '.venv-app\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $trackerPython)) {
    $trackerPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
}
if (-not (Test-Path -LiteralPath $trackerPython)) {
    throw 'Install the Python environment first using uv sync (see README.md).'
}
Push-Location -LiteralPath $PSScriptRoot
try {
    if ($Video) {
        & $trackerPython -m mktracker.main $Video
    } else {
        & $trackerPython -m mktracker.main
    }
} finally {
    Pop-Location
}

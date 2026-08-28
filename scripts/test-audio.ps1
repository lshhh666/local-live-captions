$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $PSScriptRoot
$command = Join-Path $projectRoot '.venv\Scripts\live-caption.exe'
& $command --capture-test 5
exit $LASTEXITCODE

param(
    [ValidateSet('en', 'ru', 'auto')]
    [string]$Language = 'en',

    [switch]$Cpu,

    [switch]$NoTranslation,

    [switch]$ConsoleOnly
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$command = Join-Path $projectRoot '.venv\Scripts\live-caption.exe'
$modelRoot = Join-Path $projectRoot 'models'
$llamaServer = Join-Path $projectRoot 'runtime\llama.cpp\llama-server.exe'
$translationModel = Join-Path $projectRoot 'models\qwen3-1.7b\Qwen3-1.7B-Q4_K_M.gguf'
$arguments = @(
    '--model', 'large-v3-turbo',
    '--model-dir', $modelRoot,
    '--language', $Language
)

if ($Cpu) {
    $arguments += @('--cpu', '--compute', 'int8')
}

if (-not $NoTranslation) {
    $arguments += @(
        '--translate', 'llamacpp',
        '--llama-server', $llamaServer,
        '--translation-model', $translationModel
    )
}

if (-not $ConsoleOnly) {
    $arguments += '--overlay'
}

& $command @arguments
exit $LASTEXITCODE

param(
    [switch]$SkipTests,
    [string]$OutputRoot
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
$outputRoot = if ($OutputRoot) { $OutputRoot } else { Join-Path $projectRoot 'dist' }
$bundle = Join-Path $outputRoot '本地实时字幕-便携版'
$lockFile = Join-Path $PSScriptRoot 'build-requirements.lock'

$pythonVersion = & $python -c 'import platform; print(platform.python_version())'
if ($pythonVersion.Trim() -ne '3.13.13') {
    throw "发布构建需要 Python 3.13.13，当前为 $pythonVersion"
}
$pipVersion = & $python -m pip --version
if ($pipVersion -notmatch '^pip 25\.3 ') {
    throw "发布构建需要 pip 25.3，当前为 $pipVersion"
}
$lockCheck = (& $python -m pip install --dry-run --require-hashes -r $lockFile 2>&1) -join "`n"
if ($LASTEXITCODE -ne 0) {
    throw '当前环境不符合 packaging\build-requirements.lock。请按 README 重新安装锁定依赖。'
}
if ($lockCheck -match '(?m)^Would install ') {
    throw '当前环境中的依赖版本与哈希锁不一致。请按 README 同步锁定依赖后再构建。'
}

if (-not $SkipTests) {
    & $python -m unittest discover -s (Join-Path $projectRoot 'tests_py') -v
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

& $python -m PyInstaller --version *> $null
if ($LASTEXITCODE -ne 0) {
    throw '未安装打包依赖。请按 README 安装 gpu-windows、packaging-windows 和构建约束。'
}

& $python -m PyInstaller --noconfirm --clean `
    --distpath $outputRoot `
    --workpath (Join-Path $projectRoot 'build\pyinstaller') `
    (Join-Path $PSScriptRoot 'local_live_captions.spec')
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$whisperTarget = Join-Path $bundle 'models\large-v3-turbo'
$translationTarget = Join-Path $bundle 'models\qwen3-1.7b'
$runtimeTarget = Join-Path $bundle 'runtime\llama.cpp'
New-Item -ItemType Directory -Force $whisperTarget, $translationTarget, $runtimeTarget | Out-Null

Copy-Item (Join-Path $projectRoot 'models\large-v3-turbo\config.json') $whisperTarget
Copy-Item (Join-Path $projectRoot 'models\large-v3-turbo\model.bin') $whisperTarget
Copy-Item (Join-Path $projectRoot 'models\large-v3-turbo\preprocessor_config.json') $whisperTarget
Copy-Item (Join-Path $projectRoot 'models\large-v3-turbo\tokenizer.json') $whisperTarget
Copy-Item (Join-Path $projectRoot 'models\large-v3-turbo\vocabulary.json') $whisperTarget
Copy-Item (Join-Path $projectRoot 'models\qwen3-1.7b\Qwen3-1.7B-Q4_K_M.gguf') $translationTarget
$llamaRoot = Join-Path $projectRoot 'runtime\llama.cpp'
$llamaFiles = @(
    'ggml-base.dll',
    'ggml-cpu-alderlake.dll',
    'ggml-cpu-cannonlake.dll',
    'ggml-cpu-cascadelake.dll',
    'ggml-cpu-cooperlake.dll',
    'ggml-cpu-haswell.dll',
    'ggml-cpu-icelake.dll',
    'ggml-cpu-ivybridge.dll',
    'ggml-cpu-piledriver.dll',
    'ggml-cpu-sandybridge.dll',
    'ggml-cpu-sapphirerapids.dll',
    'ggml-cpu-skylakex.dll',
    'ggml-cpu-sse42.dll',
    'ggml-cpu-x64.dll',
    'ggml-cpu-zen4.dll',
    'ggml-vulkan.dll',
    'ggml.dll',
    'libomp.dll',
    'LICENSE-LLVM-OpenMP',
    'llama-common.dll',
    'llama-server-impl.dll',
    'llama-server.exe',
    'llama.dll',
    'mtmd.dll'
)
foreach ($name in $llamaFiles) {
    $source = Join-Path $llamaRoot $name
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
        throw "缺少 llama.cpp 运行文件：$source"
    }
    Copy-Item -LiteralPath $source $runtimeTarget
}
Copy-Item (Join-Path $PSScriptRoot '便携版说明.txt') $bundle

$criticalFiles = @(
    (Join-Path $bundle '本地实时字幕.exe'),
    (Join-Path $bundle 'caption-worker.exe'),
    (Join-Path $whisperTarget 'model.bin'),
    (Join-Path $translationTarget 'Qwen3-1.7B-Q4_K_M.gguf'),
    (Join-Path $runtimeTarget 'llama-server.exe')
)
$hashLines = foreach ($file in $criticalFiles) {
    $hash = Get-FileHash -LiteralPath $file -Algorithm SHA256
    $relative = [IO.Path]::GetRelativePath($bundle, $file)
    '{0} *{1}' -f $hash.Hash, $relative
}
$hashLines | Set-Content -LiteralPath (Join-Path $bundle 'SHA256SUMS.txt') -Encoding utf8

$size = (Get-ChildItem $bundle -Recurse -File | Measure-Object Length -Sum).Sum
Write-Host ('便携版构建完成：{0}' -f $bundle)
Write-Host ('总大小：{0:N2} GB' -f ($size / 1GB))

$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $PSScriptRoot
$runtimeDirectory = Join-Path $projectRoot 'runtime\llama.cpp'
$downloadDirectory = Join-Path $projectRoot 'runtime\downloads'
$archive = Join-Path $downloadDirectory 'llama-b10595-bin-win-vulkan-x64.zip'
$partial = "$archive.partial"
$url = 'https://github.com/ggml-org/llama.cpp/releases/download/b10595/llama-b10595-bin-win-vulkan-x64.zip'
$expectedSha256 = 'DB842C568BCC6FE04383F73AF5B12D095646854A773D7C5B4EBDBEA36D6DCED9'

New-Item -ItemType Directory -Force -Path $downloadDirectory, $runtimeDirectory | Out-Null
if (-not (Test-Path -LiteralPath $archive)) {
    Write-Host '正在下载 llama.cpp b10595 Windows Vulkan 运行库。'
    & curl.exe -L --fail --retry 5 -C - -o $partial $url
    if ($LASTEXITCODE -ne 0) {
        throw "下载失败，部分文件已保留：$partial"
    }
    $actualHash = (Get-FileHash -LiteralPath $partial -Algorithm SHA256).Hash
    if ($actualHash -ne $expectedSha256) {
        throw "SHA-256 校验失败，不会解压：$partial"
    }
    Move-Item -LiteralPath $partial -Destination $archive
}

$archiveHash = (Get-FileHash -LiteralPath $archive -Algorithm SHA256).Hash
if ($archiveHash -ne $expectedSha256) {
    throw "现有压缩包校验失败，请移走后重试：$archive"
}

$temporary = Join-Path $downloadDirectory 'llama-b10595-extracted'
if (Test-Path -LiteralPath $temporary) {
    $resolvedTemporary = (Resolve-Path -LiteralPath $temporary).Path
    if (-not $resolvedTemporary.StartsWith($downloadDirectory, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "临时目录超出预期范围：$resolvedTemporary"
    }
    Remove-Item -LiteralPath $resolvedTemporary -Recurse -Force
}
Expand-Archive -LiteralPath $archive -DestinationPath $temporary

$required = @(
    'ggml-base.dll',
    'ggml-cpu-x64.dll',
    'ggml-vulkan.dll',
    'ggml.dll',
    'libomp.dll',
    'LICENSE-LLVM-OpenMP',
    'llama-common.dll',
    'llama-server-impl.dll',
    'llama-server.exe',
    'llama.dll'
)
foreach ($name in $required) {
    $source = Get-ChildItem -LiteralPath $temporary -Recurse -File -Filter $name | Select-Object -First 1
    if ($null -eq $source) {
        throw "压缩包缺少运行文件：$name"
    }
    Copy-Item -LiteralPath $source.FullName -Destination (Join-Path $runtimeDirectory $name) -Force
}

$resolvedTemporary = (Resolve-Path -LiteralPath $temporary).Path
if (-not $resolvedTemporary.StartsWith($downloadDirectory, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "临时目录超出预期范围：$resolvedTemporary"
}
Remove-Item -LiteralPath $resolvedTemporary -Recurse -Force
Write-Host "llama.cpp 运行库已经准备完成：$runtimeDirectory"

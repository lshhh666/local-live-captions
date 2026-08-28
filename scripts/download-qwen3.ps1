$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $PSScriptRoot
$modelDirectory = Join-Path $projectRoot 'models\qwen3-1.7b'
$target = Join-Path $modelDirectory 'Qwen3-1.7B-Q4_K_M.gguf'
$partial = "$target.partial"
$revision = 'daeb8e2d528a760970442092f6bf1e55c3b659eb'
$url = "https://huggingface.co/ggml-org/Qwen3-1.7B-GGUF/resolve/$revision/Qwen3-1.7B-Q4_K_M.gguf?download=true"
$expectedSize = 1282439264
$expectedSha256 = 'D2387CA2DBFEE2FFABCE7120D3770DADCA0B293052BC2F0E138FDC940D9BC7B5'

New-Item -ItemType Directory -Force -Path $modelDirectory | Out-Null

if (Test-Path -LiteralPath $target) {
    $existingHash = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash
    if ($existingHash -eq $expectedSha256) {
        Write-Host "Qwen 模型已经存在并通过校验：$target"
        exit 0
    }
    throw "现有模型校验失败，请移走后重试：$target"
}

Write-Host '正在下载 Qwen3-1.7B Q4_K_M；可重新运行此脚本断点续传。'
& curl.exe -L --fail --retry 5 -C - -o $partial $url
if ($LASTEXITCODE -ne 0) {
    throw "下载失败，部分文件已保留：$partial"
}
if ((Get-Item -LiteralPath $partial).Length -ne $expectedSize) {
    throw "文件大小不正确，不会启用模型：$partial"
}
$actualHash = (Get-FileHash -LiteralPath $partial -Algorithm SHA256).Hash
if ($actualHash -ne $expectedSha256) {
    throw "SHA-256 校验失败，不会启用模型：$partial"
}
Move-Item -LiteralPath $partial -Destination $target
Write-Host "Qwen 模型已经下载并通过校验：$target"

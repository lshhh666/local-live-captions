$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $PSScriptRoot
$modelDirectory = Join-Path $projectRoot 'models\large-v3-turbo'
$revision = '0a363e9161cbc7ed1431c9597a8ceaf0c4f78fcf'
$baseUrl = "https://huggingface.co/mobiuslabsgmbh/faster-whisper-large-v3-turbo/resolve/$revision"
$files = @(
    @{ Name = 'config.json'; Sha256 = 'B0253EA6C0D3BEA6B1E19E91A02ACFD3B53F4467362EFCB5A3E6B16C9B3A9B7E' },
    @{ Name = 'model.bin'; Sha256 = 'E76620F83D5F5B69EFD3D87E3DC180C1BD21DF9FBEBACFD4335E5E1EFCC018DA' },
    @{ Name = 'preprocessor_config.json'; Sha256 = '7CCC62C6F2765AF1F3B46C00C9B5894426835A05021C8B9C01EECB6DFB542711' },
    @{ Name = 'tokenizer.json'; Sha256 = '297B13372AC43916285644FB9687ADD3CC62EE2A1ADB60DA3DC25CC94C1871FD' },
    @{ Name = 'vocabulary.json'; Sha256 = 'C69260F2AB26D659B7C398F9A2B2B48ED0DF16C3B47D7326782FD9CBA71690C1' }
)

New-Item -ItemType Directory -Force -Path $modelDirectory | Out-Null

foreach ($file in $files) {
    $target = Join-Path $modelDirectory $file.Name
    if (Test-Path -LiteralPath $target) {
        $existingHash = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash
        if ($existingHash -eq $file.Sha256) {
            Write-Host "已经存在并通过校验：$($file.Name)"
            continue
        }
        throw "现有文件校验失败，请移走后重试：$target"
    }

    $partial = "$target.partial"
    Write-Host "正在下载：$($file.Name)"
    & curl.exe -L --fail --retry 5 -C - -o $partial "$baseUrl/$($file.Name)?download=true"
    if ($LASTEXITCODE -ne 0) {
        throw "下载失败，部分文件已保留，可重新运行脚本续传：$partial"
    }
    $actualHash = (Get-FileHash -LiteralPath $partial -Algorithm SHA256).Hash
    if ($actualHash -ne $file.Sha256) {
        throw "SHA-256 校验失败，不会启用文件：$partial"
    }
    Move-Item -LiteralPath $partial -Destination $target
}

Write-Host "Whisper 模型已经下载并通过校验：$modelDirectory"

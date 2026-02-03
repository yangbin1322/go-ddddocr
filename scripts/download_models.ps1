# PowerShell 模型文件下载脚本 (Windows)
# 使用方法: powershell -ExecutionPolicy Bypass -File scripts\download_models.ps1

$ErrorActionPreference = "Stop"

# 从 GitHub Release 下载 URL
$GITHUB_REPO = "yangbin1322/go-ddddocr"
$RELEASE_TAG = "v1.0.0"
$BASE_URL = "https://github.com/$GITHUB_REPO/releases/download/$RELEASE_TAG"

# 模型文件列表
$FILES = @{
    "common.onnx" = "52M"
    "common_det.onnx" = "20M"
    "common_old.onnx" = "13M"
    "onnxruntime.dll" = "14M"
    "charsets_beta.json" = "56K"
    "charsets_old.json" = "56K"
}

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "go-ddddocr 模型文件下载工具" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

$PROJECT_ROOT = Split-Path -Parent $PSScriptRoot
Set-Location $PROJECT_ROOT

foreach ($file in $FILES.Keys) {
    if (Test-Path $file) {
        Write-Host "✓ $file 已存在,跳过" -ForegroundColor Green
    } else {
        Write-Host "⬇ 正在下载 $file ($($FILES[$file]))..." -ForegroundColor Yellow

        $url = "$BASE_URL/$file"
        try {
            # 使用 .NET WebClient 下载
            $webClient = New-Object System.Net.WebClient
            $webClient.DownloadFile($url, $file)
            Write-Host "✓ $file 下载完成" -ForegroundColor Green
        } catch {
            Write-Host "❌ $file 下载失败: $_" -ForegroundColor Red
            exit 1
        }
    }
    Write-Host ""
}

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "✓ 所有模型文件已准备就绪!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Cyan
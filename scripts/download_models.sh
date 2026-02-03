#!/bin/bash
# 模型文件下载脚本
# 使用方法: bash scripts/download_models.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 从 GitHub Release 下载 URL
GITHUB_REPO="yangbin1322/go-ddddocr"
RELEASE_TAG="v1.0.0"
BASE_URL="https://github.com/${GITHUB_REPO}/releases/download/${RELEASE_TAG}"

# 模型文件列表
declare -A FILES=(
    ["common.onnx"]="52M"
    ["common_det.onnx"]="20M"
    ["common_old.onnx"]="13M"
    ["onnxruntime.dll"]="14M"
    ["charsets_beta.json"]="56K"
    ["charsets_old.json"]="56K"
)

echo "=========================================="
echo "go-ddddocr 模型文件下载工具"
echo "=========================================="
echo ""

cd "$PROJECT_ROOT"

for file in "${!FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ $file 已存在,跳过"
    else
        echo "⬇ 正在下载 $file (${FILES[$file]})..."

        # 尝试使用 wget 或 curl
        if command -v wget &> /dev/null; then
            wget -q --show-progress "${BASE_URL}/${file}" -O "$file"
        elif command -v curl &> /dev/null; then
            curl -L --progress-bar "${BASE_URL}/${file}" -o "$file"
        else
            echo "❌ 错误: 需要安装 wget 或 curl"
            exit 1
        fi

        if [ $? -eq 0 ]; then
            echo "✓ $file 下载完成"
        else
            echo "❌ $file 下载失败"
            exit 1
        fi
    fi
    echo ""
done

echo "=========================================="
echo "✓ 所有模型文件已准备就绪!"
echo "=========================================="

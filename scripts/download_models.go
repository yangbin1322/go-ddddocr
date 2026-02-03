package main

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
)

const (
	githubRepo = "yangbin1322/go-ddddocr"
	releaseTag = "v1.0.1"
	baseURL    = "https://github.com/" + githubRepo + "/releases/download/" + releaseTag
	targetDir  = "./models" // 文件将统一下载到当前执行目录下的 models 文件夹
)

var files = map[string]string{
	"common.onnx":        "52M",
	"common_det.onnx":    "20M",
	"common_old.onnx":    "13M",
	"onnxruntime.dll":    "14M",
	"charsets_beta.json": "56K",
	"charsets_old.json":  "56K",
}

func main() {
	fmt.Println("==========================================")
	fmt.Println("🚀 go-ddddocr 模型文件自动下载工具")
	fmt.Println("==========================================")

	// 1. 确保目标目录存在
	if err := os.MkdirAll(targetDir, 0755); err != nil {
		fmt.Printf("❌ 无法创建目录: %v\n", err)
		return
	}

	for file, size := range files {
		destPath := filepath.Join(targetDir, file)

		// 2. 检查文件是否已存在
		if _, err := os.Stat(destPath); err == nil {
			fmt.Printf("✅ %s 已存在，跳过\n", file)
			continue
		}

		fmt.Printf("⬇️ 正在下载 %s (大小约 %s)... \n", file, size)

		url := fmt.Sprintf("%s/%s", baseURL, file)
		if err := downloadFile(destPath, url); err != nil {
			fmt.Printf("❌ %s 下载失败: %v\n", file, err)
			continue // 继续下载下一个
		}
		fmt.Printf("✨ %s 下载完成!\n\n", file)
	}

	fmt.Println("==========================================")
	fmt.Printf("🎉 所有文件已准备就绪！存放在: %s\n", targetDir)
	fmt.Println("==========================================")
}

func downloadFile(destPath string, url string) error {
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("HTTP 状态码错误: %d", resp.StatusCode)
	}

	// 创建临时文件下载，防止下载一半中断导致文件损坏
	out, err := os.Create(destPath)
	if err != nil {
		return err
	}
	defer out.Close()

	_, err = io.Copy(out, resp.Body)
	return err
}

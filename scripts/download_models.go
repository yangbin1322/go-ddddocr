// go run scripts/download_models.go
package main

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
)

const (
	githubRepo  = "yangbin1322/go-ddddocr"
	releaseTag  = "v1.0.0"
	baseURL     = "https://github.com/" + githubRepo + "/releases/download/" + releaseTag
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
	fmt.Println("go-ddddocr 模型文件下载工具")
	fmt.Println("==========================================")
	fmt.Println()

	// 切换到项目根目录
	if err := os.Chdir(filepath.Join("..")); err != nil {
		fmt.Printf("❌ 错误: %v\n", err)
		os.Exit(1)
	}

	for file, size := range files {
		if _, err := os.Stat(file); err == nil {
			fmt.Printf("✓ %s 已存在,跳过\n", file)
			continue
		}

		fmt.Printf("⬇ 正在下载 %s (%s)...\n", file, size)

		url := fmt.Sprintf("%s/%s", baseURL, file)
		if err := downloadFile(file, url); err != nil {
			fmt.Printf("❌ %s 下载失败: %v\n", file, err)
			os.Exit(1)
		}

		fmt.Printf("✓ %s 下载完成\n", file)
		fmt.Println()
	}

	fmt.Println("==========================================")
	fmt.Println("✓ 所有模型文件已准备就绪!")
	fmt.Println("==========================================")
}

func downloadFile(filepath string, url string) error {
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("HTTP %d", resp.StatusCode)
	}

	out, err := os.Create(filepath)
	if err != nil {
		return err
	}
	defer out.Close()

	_, err = io.Copy(out, resp.Body)
	return err
}
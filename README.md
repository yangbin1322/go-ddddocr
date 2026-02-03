# Go-DdddOcr

Go 语言实现的 ddddocr 验证码识别库，完整移植自 [sml2h3/ddddocr](https://github.com/sml2h3/ddddocr)。

## 功能特性

| 功能 | 描述 | 状态 |
|------|------|:----:|
| OCR 文字识别 | 识别图片中的验证码文字 | ✅ |
| 目标检测 | 检测图片中文字/目标的位置 | ✅ |
| 滑块匹配 | 边缘检测匹配滑块缺口位置 | ✅ |
| 滑块比较 | 图像差异检测缺口位置 | ✅ |
| 字符范围限制 | 限定识别的字符集 | ✅ |
| 概率输出 | 获取识别结果的概率分布 | ✅ |
| 透明 PNG 处理 | 处理透明背景验证码 | ✅ |
| 自定义模型 | 支持导入自定义训练模型 | ✅ |
| 并发安全 | 支持实例池高并发 | ✅ |

## 安装

### 1. 安装包

```bash
go get github.com/yangbin1322/go-ddddocr
```

### 2. 下载模型文件

模型文件已上传到 GitHub Release，使用以下任一方式下载：

#### 方式一：使用下载脚本（推荐）

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy Bypass -File scripts/download_models.ps1
```

**Linux/macOS:**
```bash
bash scripts/download_models.sh
```

**使用 Go:**
```bash
go run scripts/download_models.go
```

#### 方式二：手动下载

从 [GitHub Releases](https://github.com/yangbin1322/go-ddddocr/releases) 下载以下文件到项目根目录：

| 文件 | 大小 | 说明 |
|------|:----:|------|
| `common.onnx` | 52MB | Beta OCR 模型 |
| `common_det.onnx` | 20MB | 目标检测模型 |
| `common_old.onnx` | 13MB | 默认 OCR 模型 |
| `onnxruntime.dll` | 14MB | ONNX Runtime 动态库 (Windows) |
| `charsets_beta.json` | 56KB | Beta 模型字符集 |
| `charsets_old.json` | 56KB | 默认模型字符集 |

**注意:** Linux/macOS 用户需要额外下载对应平台的 ONNX Runtime 库：

| 平台 | 文件 | 下载地址 |
|------|------|----------|
| Linux | `libonnxruntime.so` | [ONNX Runtime Releases](https://github.com/microsoft/onnxruntime/releases) |
| macOS | `libonnxruntime.dylib` | [ONNX Runtime Releases](https://github.com/microsoft/onnxruntime/releases) |

### 3. 依赖

Go 依赖会自动安装：

- [onnxruntime_go](https://github.com/yalue/onnxruntime_go) - ONNX Runtime Go 绑定
- [resize](https://github.com/nfnt/resize) - 图像缩放

## 快速开始

### 基础 OCR 识别

```go
package main

import (
    "fmt"
    "os"
    "github.com/yangbin1322/go-ddddocr/ddddocr"
)

func main() {
    // 设置 ONNX Runtime 路径
    ddddocr.SetOnnxRuntimePath("onnxruntime.dll")
    
    // 创建识别器
    ocr, err := ddddocr.New(ddddocr.DefaultOptions())
    if err != nil {
        panic(err)
    }
    defer ocr.Close()
    
    // 读取并识别图片
    imageData, _ := os.ReadFile("captcha.jpg")
    result, _ := ocr.Classification(imageData)
    fmt.Println("识别结果:", result)
}
```

### 设置字符范围

```go
// 预设范围
ocr.SetRanges(ddddocr.RangeDigit)           // 0 - 仅数字
ocr.SetRanges(ddddocr.RangeLowercase)       // 1 - 仅小写字母
ocr.SetRanges(ddddocr.RangeUppercase)       // 2 - 仅大写字母
ocr.SetRanges(ddddocr.RangeLowerUpper)      // 3 - 大小写字母
ocr.SetRanges(ddddocr.RangeLowerDigit)      // 4 - 小写+数字
ocr.SetRanges(ddddocr.RangeUpperDigit)      // 5 - 大写+数字
ocr.SetRanges(ddddocr.RangeLowerUpperDigit) // 6 - 大小写+数字

// 自定义范围
ocr.SetRanges("0123456789+-x/=")

// 清除范围限制
ocr.ClearRanges()
```

### 目标检测

```go
opts := ddddocr.DefaultOptions()
opts.Ocr = false
opts.Det = true

det, _ := ddddocr.New(opts)
defer det.Close()

imageData, _ := os.ReadFile("image.jpg")
bboxes, _ := det.Detection(imageData)

for _, bbox := range bboxes {
    fmt.Printf("目标位置: [%d, %d, %d, %d]\n", bbox.X1, bbox.Y1, bbox.X2, bbox.Y2)
}
```

### 滑块验证码

```go
// 创建滑块识别器（不需要加载模型）
opts := ddddocr.DefaultOptions()
opts.Ocr = false
opts.Det = false
slide, _ := ddddocr.New(opts)
defer slide.Close()

targetBytes, _ := os.ReadFile("slider.png")      // 滑块图片
bgBytes, _ := os.ReadFile("background.png")       // 背景图片

// 方法 1: 边缘匹配（适用于有透明背景的滑块）
result, _ := slide.SlideMatch(targetBytes, bgBytes, false)
fmt.Printf("滑块位置: %v\n", result.Target) // [x1, y1, x2, y2]

// 方法 2: 图像差异比较（适用于有/无缺口的两张图对比）
gapImage, _ := os.ReadFile("with_gap.jpg")   // 带缺口的图
fullImage, _ := os.ReadFile("full.jpg")       // 完整的图
result2, _ := slide.SlideComparison(gapImage, fullImage)
fmt.Printf("缺口位置: (%d, %d)\n", result2.Target[0], result2.Target[1])
```

### 使用 Beta 模型

```go
opts := ddddocr.DefaultOptions()
opts.Beta = true
ocr, _ := ddddocr.New(opts)
```

### 自定义模型

```go
opts := ddddocr.Options{
    ImportOnnxPath: "custom_model.onnx",
    CharsetsPath:   "custom_charsets.json",
}
ocr, _ := ddddocr.New(opts)
```

自定义字符集 JSON 格式：

```json
{
  "charset": ["", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
  "word": false,
  "image": [-1, 64],
  "channel": 1
}
```

| 字段 | 说明 |
|------|------|
| `charset` | 字符列表，第一个通常为空字符串（CTC blank） |
| `word` | 是否为单字模式 |
| `image` | 图像尺寸 `[width, height]`，width 为 -1 表示自适应宽度 |
| `channel` | 通道数，1 为灰度，3 为 RGB |

## 高级用法

### 处理透明 PNG

```go
result, _ := ocr.ClassificationWithOptions(imageData, ddddocr.ClassificationOptions{
    PngFix: true,
})
```

### 获取概率输出

```go
probResult, _ := ocr.ClassificationProbability(imageData)
fmt.Printf("识别文本: %s\n", probResult.Text)
fmt.Printf("字符集: %v\n", probResult.Charsets)
fmt.Printf("概率矩阵: %v\n", probResult.Probability)
```

## 并发使用

### 方案 1: 单实例 + 互斥锁

```go
var mu sync.Mutex
ocr, _ := ddddocr.New(ddddocr.DefaultOptions())

// 在协程中使用
mu.Lock()
result, _ := ocr.Classification(imageData)
mu.Unlock()
```

### 方案 2: 实例池（推荐）

```go
type OcrPool struct {
    pool chan *ddddocr.DdddOcr
}

func NewOcrPool(size int) (*OcrPool, error) {
    p := &OcrPool{pool: make(chan *ddddocr.DdddOcr, size)}
    for i := 0; i < size; i++ {
        ocr, err := ddddocr.New(ddddocr.DefaultOptions())
        if err != nil {
            return nil, err
        }
        p.pool <- ocr
    }
    return p, nil
}

func (p *OcrPool) Get() *ddddocr.DdddOcr  { return <-p.pool }
func (p *OcrPool) Put(ocr *ddddocr.DdddOcr) { p.pool <- ocr }

// 使用
pool, _ := NewOcrPool(runtime.NumCPU())
ocr := pool.Get()
result, _ := ocr.Classification(imageData)
pool.Put(ocr)
```

### 性能测试结果

| 模式 | 总请求 | 耗时 | QPS |
|------|--------|------|-----|
| 单实例+锁 | 1000 | 6.83s | 146 |
| 多实例 | 500 | 1.13s | 443 |
| 实例池 | 1000 | 1.77s | **566** |

> 测试环境: Windows 11, CPU 模式

## API 参考

### 初始化选项

```go
type Options struct {
    Ocr            bool   // 启用 OCR 模式（默认 true）
    Det            bool   // 启用目标检测模式
    Beta           bool   // 使用 Beta 模型
    UseGPU         bool   // 使用 GPU 加速
    DeviceID       int    // GPU 设备 ID
    ImportOnnxPath string // 自定义模型路径
    CharsetsPath   string // 自定义字符集路径
    ModelDir       string // 模型目录（默认当前目录）
}
```

### DdddOcr 方法

| 方法 | 说明 |
|------|------|
| `New(opts Options) (*DdddOcr, error)` | 创建识别器 |
| `Classification(imageData []byte) (string, error)` | OCR 识别 |
| `ClassificationWithOptions(imageData, opts) (string, error)` | 带选项的 OCR 识别 |
| `ClassificationProbability(imageData []byte) (*ClassificationResult, error)` | 获取概率输出 |
| `SetRanges(val interface{})` | 设置字符范围（int 或 string） |
| `ClearRanges()` | 清除字符范围限制 |
| `Detection(imageData []byte) ([]BBox, error)` | 目标检测 |
| `SlideMatch(target, bg []byte, simple bool) (*SlideMatchResult, error)` | 滑块边缘匹配 |
| `SlideComparison(target, bg []byte) (*SlideComparisonResult, error)` | 滑块图像差异比较 |
| `Close() error` | 关闭并释放资源 |

### 字符范围常量

| 常量 | 值 | 说明 |
|------|:--:|------|
| `RangeDigit` | 0 | 纯数字 0-9 |
| `RangeLowercase` | 1 | 纯小写 a-z |
| `RangeUppercase` | 2 | 纯大写 A-Z |
| `RangeLowerUpper` | 3 | 大小写字母 |
| `RangeLowerDigit` | 4 | 小写+数字 |
| `RangeUpperDigit` | 5 | 大写+数字 |
| `RangeLowerUpperDigit` | 6 | 大小写+数字 |
| `RangeNonAlphaNum` | 7 | 非字母数字字符 |

## 与 Python 版本对比

| 功能 | Python | Go |
|------|:------:|:--:|
| OCR 识别 | ✅ | ✅ |
| 目标检测 | ✅ | ✅ |
| 滑块匹配 | ✅ | ✅ |
| 滑块比较 | ✅ | ✅ |
| GPU 加速 | ✅ | ✅ |
| 自定义模型 | ✅ | ✅ |
| 颜色过滤 | ✅ | ⚠️ 基础支持 |

> 注：滑块匹配结果可能与 Python 版本有几个像素的差异，这是由于 Go 标准库与 OpenCV 实现差异导致的，不影响实际使用。

## 常见问题

### 1. 找不到 ONNX Runtime 库

确保已正确设置库路径：

```go
ddddocr.SetOnnxRuntimePath("/path/to/onnxruntime.dll")
```

### 2. 找不到模型文件

确保模型文件在正确位置，或通过 `ModelDir` 指定：

```go
opts := ddddocr.DefaultOptions()
opts.ModelDir = "/path/to/models"
```

### 3. 并发时出现错误

单个 `DdddOcr` 实例不是线程安全的，请使用互斥锁或实例池。

### 4. 滑块匹配结果不准确

- 确保图片质量清晰
- 尝试使用 `simpleTarget=true` 模式
- 检查滑块图片是否有透明背景

## 许可证

MIT License

## 致谢

- [sml2h3/ddddocr](https://github.com/sml2h3/ddddocr) - 原版 Python 实现
- [yalue/onnxruntime_go](https://github.com/yalue/onnxruntime_go) - ONNX Runtime Go 绑定

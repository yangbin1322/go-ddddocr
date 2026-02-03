package ddddocr

import (
	"bytes"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	_ "image/gif"
	_ "image/jpeg"
	"image/png"
	"io"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"

	"github.com/nfnt/resize"
	ort "github.com/yalue/onnxruntime_go"
)

// ============================================================================
// 常量定义
// ============================================================================

const (
	// 字符范围预设
	RangeDigit           = 0 // 纯数字 0-9
	RangeLowercase       = 1 // 纯小写 a-z
	RangeUppercase       = 2 // 纯大写 A-Z
	RangeLowerUpper      = 3 // 小写+大写
	RangeLowerDigit      = 4 // 小写+数字
	RangeUpperDigit      = 5 // 大写+数字
	RangeLowerUpperDigit = 6 // 小写+大写+数字
	RangeNonAlphaNum     = 7 // 除去字母数字的其他字符
)

// 模型下载配置
const (
	GitHubRepo = "yangbin1322/go-ddddocr"
	ReleaseTag = "v1.0.1"
	BaseURL    = "https://github.com/" + GitHubRepo + "/releases/download/" + ReleaseTag
)

// 默认模型目录
var DefaultModelDir = "./models"

// 模型文件列表
var ModelFiles = map[string]string{
	"common.onnx":        "52M",
	"common_det.onnx":    "20M",
	"common_old.onnx":    "13M",
	"charsets_beta.json": "56K",
	"charsets_old.json":  "56K",
}

// ONNX Runtime 文件
var OnnxRuntimeFile = map[string]string{
	"windows": "onnxruntime.dll",
	"linux":   "libonnxruntime.so",
	"darwin":  "libonnxruntime.dylib",
}

// HSV 颜色范围定义
type HSVRange struct {
	LowH, LowS, LowV    uint8
	HighH, HighS, HighV uint8
}

// 预定义颜色范围 (HSV 空间)
var DefaultColorRanges = map[string]HSVRange{
	"red":    {0, 100, 100, 10, 255, 255},    // 红色 (需要两段)
	"red2":   {160, 100, 100, 180, 255, 255}, // 红色第二段
	"green":  {35, 100, 100, 85, 255, 255},   // 绿色
	"blue":   {100, 100, 100, 130, 255, 255}, // 蓝色
	"yellow": {20, 100, 100, 35, 255, 255},   // 黄色
	"orange": {10, 100, 100, 20, 255, 255},   // 橙色
	"purple": {130, 100, 100, 160, 255, 255}, // 紫色
	"pink":   {140, 50, 100, 170, 255, 255},  // 粉色
	"brown":  {10, 100, 50, 20, 255, 150},    // 棕色
}

// ============================================================================
// 模型下载功能
// ============================================================================

// DownloadModels 下载所有模型文件到指定目录
func DownloadModels(targetDir string) error {
	fmt.Println("==========================================")
	fmt.Println("🚀 go-ddddocr 模型文件自动下载")
	fmt.Println("==========================================")

	if targetDir == "" {
		targetDir = DefaultModelDir
	}

	// 确保目录存在
	if err := os.MkdirAll(targetDir, 0755); err != nil {
		return fmt.Errorf("无法创建目录: %w", err)
	}

	// 下载模型文件
	for file, size := range ModelFiles {
		destPath := filepath.Join(targetDir, file)

		if _, err := os.Stat(destPath); err == nil {
			fmt.Printf("✅ %s 已存在，跳过\n", file)
			continue
		}

		fmt.Printf("⬇️ 正在下载 %s (大小约 %s)...\n", file, size)

		url := fmt.Sprintf("%s/%s", BaseURL, file)
		if err := downloadFile(destPath, url); err != nil {
			fmt.Printf("❌ %s 下载失败: %v\n", file, err)
			continue
		}
		fmt.Printf("✨ %s 下载完成!\n", file)
	}

	// 下载 ONNX Runtime
	onnxFile := getOnnxRuntimeFileName()
	onnxPath := filepath.Join(targetDir, onnxFile)

	if _, err := os.Stat(onnxPath); err == nil {
		fmt.Printf("✅ %s 已存在，跳过\n", onnxFile)
	} else {
		fmt.Printf("⬇️ 正在下载 %s (大小约 14M)...\n", onnxFile)
		url := fmt.Sprintf("%s/%s", BaseURL, onnxFile)
		if err := downloadFile(onnxPath, url); err != nil {
			fmt.Printf("❌ %s 下载失败: %v\n", onnxFile, err)
		} else {
			fmt.Printf("✨ %s 下载完成!\n", onnxFile)
		}
	}

	fmt.Println("==========================================")
	fmt.Printf("🎉 所有文件已准备就绪！存放在: %s\n", targetDir)
	fmt.Println("==========================================")

	return nil
}

// EnsureModels 确保模型文件存在，不存在则自动下载
func EnsureModels(targetDir string) error {
	if targetDir == "" {
		targetDir = DefaultModelDir
	}

	// 检查必要文件是否存在
	requiredFiles := []string{"common_old.onnx", "charsets_old.json"}
	allExist := true

	for _, file := range requiredFiles {
		if _, err := os.Stat(filepath.Join(targetDir, file)); os.IsNotExist(err) {
			allExist = false
			break
		}
	}

	// 检查 ONNX Runtime
	onnxFile := getOnnxRuntimeFileName()
	if _, err := os.Stat(filepath.Join(targetDir, onnxFile)); os.IsNotExist(err) {
		allExist = false
	}

	if !allExist {
		return DownloadModels(targetDir)
	}

	return nil
}

// getOnnxRuntimeFileName 获取当前平台的 ONNX Runtime 文件名
func getOnnxRuntimeFileName() string {
	if name, ok := OnnxRuntimeFile[runtime.GOOS]; ok {
		return name
	}
	return "onnxruntime.dll"
}

// downloadFile 下载文件
func downloadFile(destPath string, url string) error {
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("HTTP 状态码错误: %d", resp.StatusCode)
	}

	out, err := os.Create(destPath)
	if err != nil {
		return err
	}
	defer out.Close()

	_, err = io.Copy(out, resp.Body)
	return err
}

// ============================================================================
// 核心结构体
// ============================================================================

// DdddOcr 核心识别器结构体
type DdddOcr struct {
	modelPath  string
	charsets   []string
	charsetLen int
	inputName  string
	outputName string
	session    *ort.DynamicAdvancedSession
	options    *ort.SessionOptions

	// 字符映射
	charIndexMap   map[string]int
	allowedIndices []int

	// 模式标识
	isDetMode bool
	isOcrMode bool

	// 自定义模型配置
	useImportOnnx bool
	word          bool
	resizeConfig  []int // [width, height], -1 表示自适应
	channel       int

	// 检测模型
	detSession *ort.DynamicAdvancedSession
	detOptions *ort.SessionOptions
}

// ClassificationResult 概率输出结果
type ClassificationResult struct {
	Text        string
	Charsets    []string
	Probability [][]float32
}

// BBox 边界框
type BBox struct {
	X1, Y1, X2, Y2 int
}

// SlideMatchResult 滑块匹配结果
type SlideMatchResult struct {
	TargetX int   `json:"target_x"`
	TargetY int   `json:"target_y"`
	Target  []int `json:"target"` // [x1, y1, x2, y2]
}

// SlideComparisonResult 滑块比较结果
type SlideComparisonResult struct {
	Target []int `json:"target"` // [x, y]
}

// ============================================================================
// 初始化
// ============================================================================

var (
	initOnce        sync.Once
	onnxRuntimePath string
)

// SetOnnxRuntimePath 设置 ONNX Runtime 库路径
func SetOnnxRuntimePath(path string) {
	onnxRuntimePath = path
}

func initOnnxRuntime() error {
	var initErr error
	initOnce.Do(func() {
		if onnxRuntimePath == "" {
			// 默认使用 models 目录下的库
			onnxRuntimePath = filepath.Join(DefaultModelDir, getOnnxRuntimeFileName())
		}
		ort.SetSharedLibraryPath(onnxRuntimePath)
		initErr = ort.InitializeEnvironment()
	})
	return initErr
}

// Options 初始化选项
type Options struct {
	Ocr            bool   // 是否启用 OCR
	Det            bool   // 是否启用目标检测
	Beta           bool   // 是否使用 Beta 模型
	UseGPU         bool   // 是否使用 GPU
	DeviceID       int    // GPU 设备 ID
	ImportOnnxPath string // 自定义模型路径
	CharsetsPath   string // 自定义字符集路径
	ModelDir       string // 模型目录
	AutoDownload   bool   // 自动下载模型（默认 true）
}

// DefaultOptions 默认选项
func DefaultOptions() Options {
	return Options{
		Ocr:          true,
		Det:          false,
		Beta:         false,
		UseGPU:       false,
		DeviceID:     0,
		ModelDir:     DefaultModelDir,
		AutoDownload: true,
	}
}

// New 创建识别器
func New(opts Options) (*DdddOcr, error) {
	// 设置默认模型目录
	if opts.ModelDir == "" {
		opts.ModelDir = DefaultModelDir
	}

	// 自动下载模型
	if opts.AutoDownload && opts.ImportOnnxPath == "" {
		if err := EnsureModels(opts.ModelDir); err != nil {
			return nil, fmt.Errorf("模型下载失败: %w", err)
		}
	}

	// 设置 ONNX Runtime 路径
	if onnxRuntimePath == "" {
		SetOnnxRuntimePath(filepath.Join(opts.ModelDir, getOnnxRuntimeFileName()))
	}

	if err := initOnnxRuntime(); err != nil {
		return nil, fmt.Errorf("初始化 ONNX Runtime 失败: %w", err)
	}

	d := &DdddOcr{
		charIndexMap: make(map[string]int),
		channel:      1,
	}

	// 自定义模型模式
	if opts.ImportOnnxPath != "" {
		return d.initCustomModel(opts)
	}

	// 目标检测模式
	if opts.Det {
		return d.initDetectionMode(opts)
	}

	// OCR 模式
	if opts.Ocr {
		return d.initOcrMode(opts)
	}

	// 滑块模式 (无需加载模型)
	d.isOcrMode = false
	d.isDetMode = false
	return d, nil
}

// initOcrMode 初始化 OCR 模式
func (d *DdddOcr) initOcrMode(opts Options) (*DdddOcr, error) {
	d.isOcrMode = true
	d.isDetMode = false

	// 选择模型
	var modelName string
	if opts.Beta {
		modelName = "common.onnx"
	} else {
		modelName = "common_old.onnx"
	}
	d.modelPath = filepath.Join(opts.ModelDir, modelName)

	// 加载模型信息
	inputs, outputs, err := ort.GetInputOutputInfo(d.modelPath)
	if err != nil {
		return nil, fmt.Errorf("获取模型信息失败: %w", err)
	}

	// 创建会话选项
	d.options, err = ort.NewSessionOptions()
	if err != nil {
		return nil, err
	}

	_ = d.options.SetIntraOpNumThreads(1)
	_ = d.options.SetInterOpNumThreads(1)
	_ = d.options.SetGraphOptimizationLevel(ort.GraphOptimizationLevel(99))

	// 创建会话
	d.session, err = ort.NewDynamicAdvancedSession(
		d.modelPath,
		[]string{inputs[0].Name},
		[]string{outputs[0].Name},
		d.options,
	)
	if err != nil {
		d.options.Destroy()
		return nil, err
	}

	d.inputName = inputs[0].Name
	d.outputName = outputs[0].Name

	// 加载字符集
	d.charsets = d.loadCharsetsForModel(opts.ModelDir, opts.Beta)
	d.charsetLen = len(d.charsets)

	// 构建反向映射
	for i, c := range d.charsets {
		d.charIndexMap[c] = i
	}

	return d, nil
}

// initDetectionMode 初始化检测模式
func (d *DdddOcr) initDetectionMode(opts Options) (*DdddOcr, error) {
	d.isOcrMode = false
	d.isDetMode = true

	modelPath := filepath.Join(opts.ModelDir, "common_det.onnx")

	inputs, outputs, err := ort.GetInputOutputInfo(modelPath)
	if err != nil {
		return nil, fmt.Errorf("获取检测模型信息失败: %w", err)
	}

	d.detOptions, err = ort.NewSessionOptions()
	if err != nil {
		return nil, err
	}

	_ = d.detOptions.SetIntraOpNumThreads(1)
	_ = d.detOptions.SetInterOpNumThreads(1)
	_ = d.detOptions.SetGraphOptimizationLevel(ort.GraphOptimizationLevel(99))

	d.detSession, err = ort.NewDynamicAdvancedSession(
		modelPath,
		[]string{inputs[0].Name},
		[]string{outputs[0].Name},
		d.detOptions,
	)
	if err != nil {
		d.detOptions.Destroy()
		return nil, err
	}

	return d, nil
}

// initCustomModel 初始化自定义模型
func (d *DdddOcr) initCustomModel(opts Options) (*DdddOcr, error) {
	d.useImportOnnx = true
	d.isOcrMode = true
	d.isDetMode = false
	d.modelPath = opts.ImportOnnxPath

	// 加载字符集配置
	if opts.CharsetsPath == "" {
		return nil, fmt.Errorf("自定义模型需要提供字符集路径")
	}

	data, err := os.ReadFile(opts.CharsetsPath)
	if err != nil {
		return nil, fmt.Errorf("读取字符集文件失败: %w", err)
	}

	var config struct {
		Charset []string `json:"charset"`
		Word    bool     `json:"word"`
		Image   []int    `json:"image"`
		Channel int      `json:"channel"`
	}
	if err := json.Unmarshal(data, &config); err != nil {
		return nil, fmt.Errorf("解析字符集配置失败: %w", err)
	}

	d.charsets = config.Charset
	d.charsetLen = len(d.charsets)
	d.word = config.Word
	d.resizeConfig = config.Image
	d.channel = config.Channel

	// 构建映射
	for i, c := range d.charsets {
		d.charIndexMap[c] = i
	}

	// 加载模型
	inputs, outputs, err := ort.GetInputOutputInfo(d.modelPath)
	if err != nil {
		return nil, fmt.Errorf("获取模型信息失败: %w", err)
	}

	d.options, err = ort.NewSessionOptions()
	if err != nil {
		return nil, err
	}

	_ = d.options.SetIntraOpNumThreads(1)
	_ = d.options.SetInterOpNumThreads(1)
	_ = d.options.SetGraphOptimizationLevel(ort.GraphOptimizationLevel(99))

	d.session, err = ort.NewDynamicAdvancedSession(
		d.modelPath,
		[]string{inputs[0].Name},
		[]string{outputs[0].Name},
		d.options,
	)
	if err != nil {
		d.options.Destroy()
		return nil, err
	}

	d.inputName = inputs[0].Name
	d.outputName = outputs[0].Name

	return d, nil
}

// loadCharsetsForModel 加载字符集
func (d *DdddOcr) loadCharsetsForModel(modelDir string, beta bool) []string {
	var charsetFile string
	if beta {
		charsetFile = "charsets_beta.json"
	} else {
		charsetFile = "charsets_old.json"
	}

	charsets := loadCharsets(filepath.Join(modelDir, charsetFile))
	if charsets == nil {
		charsets = loadCharsets(filepath.Join(modelDir, "charsets.json"))
	}
	if charsets == nil {
		// 默认字符集
		charsets = []string{
			"", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
			"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
			"n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
		}
	}
	return charsets
}

func loadCharsets(path string) []string {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	var charsets []string
	if err := json.Unmarshal(data, &charsets); err != nil {
		return nil
	}
	return charsets
}

// ============================================================================
// OCR 功能
// ============================================================================

// SetRanges 设置识别范围
func (d *DdddOcr) SetRanges(val interface{}) {
	var chars string

	switch v := val.(type) {
	case string:
		chars = v
	case int:
		switch v {
		case RangeDigit:
			chars = "0123456789"
		case RangeLowercase:
			chars = "abcdefghijklmnopqrstuvwxyz"
		case RangeUppercase:
			chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
		case RangeLowerUpper:
			chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
		case RangeLowerDigit:
			chars = "abcdefghijklmnopqrstuvwxyz0123456789"
		case RangeUpperDigit:
			chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
		case RangeLowerUpperDigit:
			chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
		case RangeNonAlphaNum:
			// 除去字母数字
			alphaNum := "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
			for _, c := range d.charsets {
				if !strings.Contains(alphaNum, c) && c != "" {
					chars += c
				}
			}
		default:
			d.allowedIndices = nil
			return
		}
	default:
		d.allowedIndices = nil
		return
	}

	// 构建索引列表
	indices := make([]int, 0, len(chars)+1)
	indices = append(indices, 0) // 保留空白占位符

	for _, char := range strings.Split(chars, "") {
		if idx, ok := d.charIndexMap[char]; ok {
			indices = append(indices, idx)
		}
	}
	d.allowedIndices = indices
}

// ClearRanges 清除字符范围限制
func (d *DdddOcr) ClearRanges() {
	d.allowedIndices = nil
}

// ClassificationOptions 识别选项
type ClassificationOptions struct {
	PngFix      bool                // 处理透明 PNG
	Probability bool                // 返回概率
	Colors      []string            // 颜色过滤
	ColorRanges map[string]HSVRange // 自定义颜色范围
}

// Classification 识别图片
func (d *DdddOcr) Classification(imageData []byte) (string, error) {
	return d.ClassificationWithOptions(imageData, ClassificationOptions{})
}

// ClassificationWithOptions 带选项的识别
func (d *DdddOcr) ClassificationWithOptions(imageData []byte, opts ClassificationOptions) (string, error) {
	if d.isDetMode {
		return "", fmt.Errorf("当前为目标检测模式，请使用 Detection 方法")
	}

	// 解码图片
	img, _, err := image.Decode(bytes.NewReader(imageData))
	if err != nil {
		return "", fmt.Errorf("解码失败: %w", err)
	}

	// PNG 透明背景处理
	if opts.PngFix {
		img = pngRgbaBlackPreprocess(img)
	}

	// 颜色过滤
	if len(opts.Colors) > 0 {
		img = filterByColors(img, opts.Colors, opts.ColorRanges)
	}

	// 计算缩放尺寸
	var width, height int
	bounds := img.Bounds()
	dx := bounds.Dx()
	dy := bounds.Dy()

	if d.useImportOnnx && len(d.resizeConfig) >= 2 {
		if d.resizeConfig[0] == -1 {
			if d.word {
				width = d.resizeConfig[1]
				height = d.resizeConfig[1]
			} else {
				width = int(float64(dx) * (float64(d.resizeConfig[1]) / float64(dy)))
				height = d.resizeConfig[1]
			}
		} else {
			width = d.resizeConfig[0]
			height = d.resizeConfig[1]
		}
	} else {
		width = int(float64(dx) * (64.0 / float64(dy)))
		height = 64
	}

	if width < 1 {
		width = 1
	}

	// 图片缩放
	resizedImage := resize.Resize(uint(width), uint(height), img, resize.Bilinear)

	// 转为灰度并归一化
	var inputData []float32
	if d.channel == 1 {
		inputData = imageToGrayFloat32(resizedImage, width, height, d.useImportOnnx)
	} else {
		inputData = imageToRGBFloat32(resizedImage, width, height)
	}

	// 构建输入张量
	var inputShape ort.Shape
	if d.channel == 1 {
		inputShape = ort.NewShape(1, 1, int64(height), int64(width))
	} else {
		inputShape = ort.NewShape(1, int64(d.channel), int64(height), int64(width))
	}

	inputTensor, err := ort.NewTensor(inputShape, inputData)
	if err != nil {
		return "", err
	}
	defer inputTensor.Destroy()

	outputs := []ort.Value{nil}
	err = d.session.Run([]ort.Value{inputTensor}, outputs)
	if err != nil {
		return "", err
	}

	if outputs[0] == nil {
		return "", fmt.Errorf("无输出")
	}
	defer outputs[0].Destroy()

	// 解码结果
	if outputTensor, ok := outputs[0].(*ort.Tensor[float32]); ok {
		return decodeOutputFloatFast(
			outputTensor.GetData(),
			outputTensor.GetShape(),
			d.charsets,
			d.charsetLen,
			d.allowedIndices,
		), nil
	}

	return "", fmt.Errorf("不支持的输出类型")
}

// ClassificationProbability 获取概率输出
func (d *DdddOcr) ClassificationProbability(imageData []byte) (*ClassificationResult, error) {
	if d.isDetMode {
		return nil, fmt.Errorf("当前为目标检测模式")
	}

	img, _, err := image.Decode(bytes.NewReader(imageData))
	if err != nil {
		return nil, fmt.Errorf("解码失败: %w", err)
	}

	bounds := img.Bounds()
	dx := bounds.Dx()
	dy := bounds.Dy()
	width := int(float64(dx) * (64.0 / float64(dy)))
	if width < 1 {
		width = 1
	}

	resizedImage := resize.Resize(uint(width), 64, img, resize.Bilinear)
	inputData := imageToGrayFloat32(resizedImage, width, 64, false)

	inputShape := ort.NewShape(1, 1, 64, int64(width))
	inputTensor, err := ort.NewTensor(inputShape, inputData)
	if err != nil {
		return nil, err
	}
	defer inputTensor.Destroy()

	outputs := []ort.Value{nil}
	err = d.session.Run([]ort.Value{inputTensor}, outputs)
	if err != nil {
		return nil, err
	}

	if outputs[0] == nil {
		return nil, fmt.Errorf("无输出")
	}
	defer outputs[0].Destroy()

	outputTensor, ok := outputs[0].(*ort.Tensor[float32])
	if !ok {
		return nil, fmt.Errorf("不支持的输出类型")
	}

	data := outputTensor.GetData()
	shape := outputTensor.GetShape()

	// 计算 softmax 概率
	timesteps := int(shape[0])
	numClasses := int(shape[2])

	probability := make([][]float32, timesteps)
	for t := 0; t < timesteps; t++ {
		offset := t * numClasses
		probs := softmax(data[offset : offset+numClasses])
		probability[t] = probs
	}

	// 解码文本
	text := decodeOutputFloatFast(data, shape, d.charsets, d.charsetLen, d.allowedIndices)

	// 根据是否有范围限制返回不同的字符集
	var charsets []string
	if len(d.allowedIndices) > 0 {
		charsets = make([]string, len(d.allowedIndices))
		for i, idx := range d.allowedIndices {
			if idx < len(d.charsets) {
				charsets[i] = d.charsets[idx]
			}
		}
	} else {
		charsets = d.charsets
	}

	return &ClassificationResult{
		Text:        text,
		Charsets:    charsets,
		Probability: probability,
	}, nil
}

// ============================================================================
// 目标检测功能
// ============================================================================

// Detection 目标检测
func (d *DdddOcr) Detection(imageData []byte) ([]BBox, error) {
	if !d.isDetMode {
		return nil, fmt.Errorf("当前不是目标检测模式，请使用 Det=true 初始化")
	}

	// 解码图片
	img, err := decodeImageCV(imageData)
	if err != nil {
		return nil, fmt.Errorf("解码图片失败: %w", err)
	}

	origH := img.Bounds().Dy()
	origW := img.Bounds().Dx()

	// 预处理：缩放到 416x416
	inputSize := [2]int{416, 416}
	preprocessed, ratio := preproc(img, inputSize)

	// 创建输入张量
	inputShape := ort.NewShape(1, 3, int64(inputSize[0]), int64(inputSize[1]))
	inputTensor, err := ort.NewTensor(inputShape, preprocessed)
	if err != nil {
		return nil, err
	}
	defer inputTensor.Destroy()

	outputs := []ort.Value{nil}
	err = d.detSession.Run([]ort.Value{inputTensor}, outputs)
	if err != nil {
		return nil, err
	}

	if outputs[0] == nil {
		return nil, fmt.Errorf("无输出")
	}
	defer outputs[0].Destroy()

	outputTensor, ok := outputs[0].(*ort.Tensor[float32])
	if !ok {
		return nil, fmt.Errorf("不支持的输出类型")
	}

	// 后处理
	predictions := demoPostprocess(outputTensor.GetData(), inputSize)
	bboxes := multiclassNMS(predictions, ratio, origW, origH, 0.45, 0.1)

	return bboxes, nil
}

// ============================================================================
// 滑块验证码功能
// ============================================================================

// SlideMatch 滑块匹配（边缘检测算法）
func (d *DdddOcr) SlideMatch(targetBytes, backgroundBytes []byte, simpleTarget bool) (*SlideMatchResult, error) {
	var target image.Image
	var targetX, targetY int
	var err error

	if !simpleTarget {
		// 提取透明区域
		target, targetX, targetY, err = getTarget(targetBytes)
		if err != nil {
			// 回退到简单模式
			return d.SlideMatch(targetBytes, backgroundBytes, true)
		}
	} else {
		target, _, err = image.Decode(bytes.NewReader(targetBytes))
		if err != nil {
			return nil, fmt.Errorf("解码滑块图失败: %w", err)
		}
		targetX, targetY = 0, 0
	}

	background, _, err := image.Decode(bytes.NewReader(backgroundBytes))
	if err != nil {
		return nil, fmt.Errorf("解码背景图失败: %w", err)
	}

	// 1. Canny 边缘检测（返回灰度图）
	targetEdge := cannyEdgeDetect(target, 100, 200)
	bgEdge := cannyEdgeDetect(background, 100, 200)

	// 2. 灰度转 RGB（与 Python 的 cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) 对应）
	targetRGB := grayToRGB(targetEdge)
	bgRGB := grayToRGB(bgEdge)

	// 3. 模板匹配 (TM_CCOEFF_NORMED) - 在 RGB 图像上进行
	maxLoc := templateMatchRGB(bgRGB, targetRGB)

	h := target.Bounds().Dy()
	w := target.Bounds().Dx()

	return &SlideMatchResult{
		TargetX: targetX,
		TargetY: targetY,
		Target:  []int{maxLoc.X, maxLoc.Y, maxLoc.X + w, maxLoc.Y + h},
	}, nil
}

// cannyEdgeDetect Canny 边缘检测（模拟 cv2.Canny）
func cannyEdgeDetect(img image.Image, lowThreshold, highThreshold int) *image.Gray {
	bounds := img.Bounds()
	w := bounds.Dx()
	h := bounds.Dy()

	// 转灰度
	gray := make([][]int, h)
	for y := 0; y < h; y++ {
		gray[y] = make([]int, w)
		for x := 0; x < w; x++ {
			r, g, b, _ := img.At(x+bounds.Min.X, y+bounds.Min.Y).RGBA()
			gray[y][x] = int(0.299*float64(r>>8) + 0.587*float64(g>>8) + 0.114*float64(b>>8))
		}
	}

	// Sobel 算子计算梯度
	gradX := make([][]int, h)
	gradY := make([][]int, h)
	magnitude := make([][]int, h)
	for y := 0; y < h; y++ {
		gradX[y] = make([]int, w)
		gradY[y] = make([]int, w)
		magnitude[y] = make([]int, w)
	}

	for y := 1; y < h-1; y++ {
		for x := 1; x < w-1; x++ {
			gx := -gray[y-1][x-1] - 2*gray[y][x-1] - gray[y+1][x-1] +
				gray[y-1][x+1] + 2*gray[y][x+1] + gray[y+1][x+1]
			gy := -gray[y-1][x-1] - 2*gray[y-1][x] - gray[y-1][x+1] +
				gray[y+1][x-1] + 2*gray[y+1][x] + gray[y+1][x+1]

			gradX[y][x] = gx
			gradY[y][x] = gy
			magnitude[y][x] = int(math.Sqrt(float64(gx*gx + gy*gy)))
		}
	}

	// 非极大值抑制
	suppressed := make([][]int, h)
	for y := 0; y < h; y++ {
		suppressed[y] = make([]int, w)
	}

	for y := 1; y < h-1; y++ {
		for x := 1; x < w-1; x++ {
			mag := magnitude[y][x]
			if mag == 0 {
				continue
			}

			gx := float64(gradX[y][x])
			gy := float64(gradY[y][x])
			angle := math.Atan2(gy, gx) * 180 / math.Pi
			if angle < 0 {
				angle += 180
			}

			var q, r int
			if (angle >= 0 && angle < 22.5) || (angle >= 157.5 && angle <= 180) {
				q = magnitude[y][x+1]
				r = magnitude[y][x-1]
			} else if angle >= 22.5 && angle < 67.5 {
				q = magnitude[y-1][x+1]
				r = magnitude[y+1][x-1]
			} else if angle >= 67.5 && angle < 112.5 {
				q = magnitude[y-1][x]
				r = magnitude[y+1][x]
			} else {
				q = magnitude[y-1][x-1]
				r = magnitude[y+1][x+1]
			}

			if mag >= q && mag >= r {
				suppressed[y][x] = mag
			}
		}
	}

	// 双阈值 + 滞后阈值
	result := image.NewGray(bounds)
	edge := make([][]int, h)
	for y := 0; y < h; y++ {
		edge[y] = make([]int, w)
	}

	for y := 1; y < h-1; y++ {
		for x := 1; x < w-1; x++ {
			if suppressed[y][x] >= highThreshold {
				edge[y][x] = 2
				result.SetGray(x+bounds.Min.X, y+bounds.Min.Y, color.Gray{255})
			} else if suppressed[y][x] >= lowThreshold {
				edge[y][x] = 1
			}
		}
	}

	// 边缘连接
	changed := true
	for changed {
		changed = false
		for y := 1; y < h-1; y++ {
			for x := 1; x < w-1; x++ {
				if edge[y][x] == 1 {
					for dy := -1; dy <= 1; dy++ {
						for dx := -1; dx <= 1; dx++ {
							if edge[y+dy][x+dx] == 2 {
								edge[y][x] = 2
								result.SetGray(x+bounds.Min.X, y+bounds.Min.Y, color.Gray{255})
								changed = true
								break
							}
						}
						if edge[y][x] == 2 {
							break
						}
					}
				}
			}
		}
	}

	return result
}

// grayToRGB 灰度图转 RGB
func grayToRGB(gray *image.Gray) *image.RGBA {
	bounds := gray.Bounds()
	rgb := image.NewRGBA(bounds)

	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			g := gray.GrayAt(x, y).Y
			rgb.Set(x, y, color.RGBA{g, g, g, 255})
		}
	}

	return rgb
}

// templateMatchRGB 在 RGB 图像上进行模板匹配 (TM_CCOEFF_NORMED)
func templateMatchRGB(background, template *image.RGBA) image.Point {
	bgBounds := background.Bounds()
	tplBounds := template.Bounds()

	bgW := bgBounds.Dx()
	bgH := bgBounds.Dy()
	tplW := tplBounds.Dx()
	tplH := tplBounds.Dy()

	if tplW > bgW || tplH > bgH {
		return image.Point{}
	}

	tplSize := tplW * tplH

	tplR := make([]float64, tplSize)
	tplG := make([]float64, tplSize)
	tplB := make([]float64, tplSize)
	var sumR, sumG, sumB float64

	for y := 0; y < tplH; y++ {
		for x := 0; x < tplW; x++ {
			c := template.RGBAAt(x+tplBounds.Min.X, y+tplBounds.Min.Y)
			idx := y*tplW + x
			tplR[idx] = float64(c.R)
			tplG[idx] = float64(c.G)
			tplB[idx] = float64(c.B)
			sumR += float64(c.R)
			sumG += float64(c.G)
			sumB += float64(c.B)
		}
	}

	n := float64(tplSize)
	meanR := sumR / n
	meanG := sumG / n
	meanB := sumB / n

	var tplVarR, tplVarG, tplVarB float64
	for i := 0; i < tplSize; i++ {
		tplR[i] -= meanR
		tplG[i] -= meanG
		tplB[i] -= meanB
		tplVarR += tplR[i] * tplR[i]
		tplVarG += tplG[i] * tplG[i]
		tplVarB += tplB[i] * tplB[i]
	}
	tplStd := math.Sqrt(tplVarR + tplVarG + tplVarB)

	if tplStd < 1.0 {
		return image.Point{}
	}

	maxVal := float64(-2)
	maxLoc := image.Point{}

	for y := 0; y <= bgH-tplH; y++ {
		for x := 0; x <= bgW-tplW; x++ {
			var wSumR, wSumG, wSumB float64
			for ty := 0; ty < tplH; ty++ {
				for tx := 0; tx < tplW; tx++ {
					c := background.RGBAAt(x+tx+bgBounds.Min.X, y+ty+bgBounds.Min.Y)
					wSumR += float64(c.R)
					wSumG += float64(c.G)
					wSumB += float64(c.B)
				}
			}
			wMeanR := wSumR / n
			wMeanG := wSumG / n
			wMeanB := wSumB / n

			var crossSum, winVar float64
			for ty := 0; ty < tplH; ty++ {
				for tx := 0; tx < tplW; tx++ {
					c := background.RGBAAt(x+tx+bgBounds.Min.X, y+ty+bgBounds.Min.Y)
					idx := ty*tplW + tx

					drW := float64(c.R) - wMeanR
					dgW := float64(c.G) - wMeanG
					dbW := float64(c.B) - wMeanB

					crossSum += drW*tplR[idx] + dgW*tplG[idx] + dbW*tplB[idx]
					winVar += drW*drW + dgW*dgW + dbW*dbW
				}
			}

			winStd := math.Sqrt(winVar)
			if winStd < 1.0 {
				continue
			}

			ncc := crossSum / (winStd * tplStd)

			if ncc > maxVal {
				maxVal = ncc
				maxLoc = image.Point{X: x, Y: y}
			}
		}
	}

	return maxLoc
}

// SlideComparison 滑块比较（图像差异算法）
func (d *DdddOcr) SlideComparison(targetBytes, backgroundBytes []byte) (*SlideComparisonResult, error) {
	target, _, err := image.Decode(bytes.NewReader(targetBytes))
	if err != nil {
		return nil, fmt.Errorf("解码目标图失败: %w", err)
	}

	background, _, err := image.Decode(bytes.NewReader(backgroundBytes))
	if err != nil {
		return nil, fmt.Errorf("解码背景图失败: %w", err)
	}

	targetBounds := target.Bounds()
	bgBounds := background.Bounds()

	if targetBounds.Dx() != bgBounds.Dx() || targetBounds.Dy() != bgBounds.Dy() {
		background = resize.Resize(
			uint(targetBounds.Dx()),
			uint(targetBounds.Dy()),
			background,
			resize.Bilinear,
		)
	}

	targetRGB := toRGB(target)
	backgroundRGB := toRGB(background)

	diff := imageDifferenceRGB(backgroundRGB, targetRGB)

	threshold := uint8(80)
	binaryImg := binarizeRGB(diff, threshold)

	startX, startY := findGapPython(binaryImg, 5)

	return &SlideComparisonResult{
		Target: []int{startX, startY},
	}, nil
}

// toRGB 转换图像为 RGB 格式
func toRGB(img image.Image) *image.RGBA {
	bounds := img.Bounds()
	rgba := image.NewRGBA(bounds)
	draw.Draw(rgba, bounds, img, bounds.Min, draw.Src)
	return rgba
}

// imageDifferenceRGB 计算两张 RGB 图像的差异
func imageDifferenceRGB(img1, img2 image.Image) *image.RGBA {
	bounds := img1.Bounds()
	bounds2 := img2.Bounds()

	w := bounds.Dx()
	h := bounds.Dy()
	if bounds2.Dx() < w {
		w = bounds2.Dx()
	}
	if bounds2.Dy() < h {
		h = bounds2.Dy()
	}

	result := image.NewRGBA(image.Rect(0, 0, w, h))

	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			r1, g1, b1, _ := img1.At(x+bounds.Min.X, y+bounds.Min.Y).RGBA()
			r2, g2, b2, _ := img2.At(x+bounds2.Min.X, y+bounds2.Min.Y).RGBA()

			dr := absDiff(uint8(r1>>8), uint8(r2>>8))
			dg := absDiff(uint8(g1>>8), uint8(g2>>8))
			db := absDiff(uint8(b1>>8), uint8(b2>>8))

			result.Set(x, y, color.RGBA{dr, dg, db, 255})
		}
	}

	return result
}

func absDiff(a, b uint8) uint8 {
	if a > b {
		return a - b
	}
	return b - a
}

// binarizeRGB 对 RGB 图像进行二值化
func binarizeRGB(img *image.RGBA, threshold uint8) *image.RGBA {
	bounds := img.Bounds()
	result := image.NewRGBA(bounds)

	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			r8, g8, b8 := uint8(r>>8), uint8(g>>8), uint8(b>>8)

			if r8 > threshold {
				r8 = 255
			} else {
				r8 = 0
			}
			if g8 > threshold {
				g8 = 255
			} else {
				g8 = 0
			}
			if b8 > threshold {
				b8 = 255
			} else {
				b8 = 0
			}

			result.Set(x, y, color.RGBA{r8, g8, b8, 255})
		}
	}

	return result
}

// findGapPython 查找缺口位置
func findGapPython(img image.Image, minCount int) (int, int) {
	bounds := img.Bounds()
	w := bounds.Dx()
	h := bounds.Dy()

	startX := 0
	startY := 0

	for x := 0; x < w; x++ {
		count := 0
		for y := 0; y < h; y++ {
			r, g, b, _ := img.At(x+bounds.Min.X, y+bounds.Min.Y).RGBA()
			r8, g8, b8 := uint8(r>>8), uint8(g>>8), uint8(b>>8)

			if r8 != 0 || g8 != 0 || b8 != 0 {
				count++
				if count == 1 && startY == 0 {
					startY = y
				}
			}
		}

		if count >= minCount {
			startX = x + 2
			break
		}
	}

	return startX, startY
}

// ============================================================================
// 图像处理辅助函数
// ============================================================================

// pngRgbaBlackPreprocess 处理透明 PNG
func pngRgbaBlackPreprocess(img image.Image) image.Image {
	bounds := img.Bounds()
	result := image.NewRGBA(bounds)

	draw.Draw(result, bounds, &image.Uniform{color.White}, image.Point{}, draw.Src)
	draw.Draw(result, bounds, img, bounds.Min, draw.Over)

	return result
}

// filterByColors 颜色过滤
func filterByColors(img image.Image, colors []string, customRanges map[string]HSVRange) image.Image {
	bounds := img.Bounds()
	result := image.NewRGBA(bounds)

	// 白色背景
	draw.Draw(result, bounds, &image.Uniform{color.White}, image.Point{}, draw.Src)

	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			r8, g8, b8 := uint8(r>>8), uint8(g>>8), uint8(b>>8)

			h, s, v := rgbToHSV(r8, g8, b8)

			match := false
			for _, colorName := range colors {
				var hsvRange HSVRange
				var ok bool

				if customRanges != nil {
					hsvRange, ok = customRanges[colorName]
				}
				if !ok {
					hsvRange, ok = DefaultColorRanges[colorName]
				}
				if !ok {
					continue
				}

				if h >= hsvRange.LowH && h <= hsvRange.HighH &&
					s >= hsvRange.LowS && s <= hsvRange.HighS &&
					v >= hsvRange.LowV && v <= hsvRange.HighV {
					match = true
					break
				}

				if colorName == "red" {
					if hsvRange2, ok := DefaultColorRanges["red2"]; ok {
						if h >= hsvRange2.LowH && h <= hsvRange2.HighH &&
							s >= hsvRange2.LowS && s <= hsvRange2.HighS &&
							v >= hsvRange2.LowV && v <= hsvRange2.HighV {
							match = true
							break
						}
					}
				}
			}

			if match {
				result.Set(x, y, color.RGBA{r8, g8, b8, 255})
			}
		}
	}

	return result
}

// rgbToHSV RGB 转 HSV
func rgbToHSV(r, g, b uint8) (uint8, uint8, uint8) {
	rf := float64(r) / 255
	gf := float64(g) / 255
	bf := float64(b) / 255

	maxVal := math.Max(rf, math.Max(gf, bf))
	minVal := math.Min(rf, math.Min(gf, bf))
	delta := maxVal - minVal

	var h, s, v float64
	v = maxVal

	if maxVal == 0 {
		s = 0
	} else {
		s = delta / maxVal
	}

	if delta == 0 {
		h = 0
	} else if maxVal == rf {
		h = 60 * math.Mod((gf-bf)/delta, 6)
	} else if maxVal == gf {
		h = 60 * ((bf-rf)/delta + 2)
	} else {
		h = 60 * ((rf-gf)/delta + 4)
	}

	if h < 0 {
		h += 360
	}

	return uint8(h / 2), uint8(s * 255), uint8(v * 255)
}

// imageToGrayFloat32 图像转灰度浮点数组
func imageToGrayFloat32(img image.Image, width, height int, normalize bool) []float32 {
	data := make([]float32, width*height)
	bounds := img.Bounds()

	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			r, g, b, _ := img.At(x+bounds.Min.X, y+bounds.Min.Y).RGBA()
			gray := 0.299*float64(r>>8) + 0.587*float64(g>>8) + 0.114*float64(b>>8)

			if normalize {
				data[y*width+x] = float32(gray / 255.0)
			} else {
				data[y*width+x] = float32((gray/255.0 - 0.5) / 0.5)
			}
		}
	}
	return data
}

// imageToRGBFloat32 图像转 RGB 浮点数组
func imageToRGBFloat32(img image.Image, width, height int) []float32 {
	data := make([]float32, 3*width*height)
	bounds := img.Bounds()

	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			r, g, b, _ := img.At(x+bounds.Min.X, y+bounds.Min.Y).RGBA()
			idx := y*width + x
			data[idx] = float32((float64(r>>8)/255.0 - 0.5) / 0.5)
			data[width*height+idx] = float32((float64(g>>8)/255.0 - 0.5) / 0.5)
			data[2*width*height+idx] = float32((float64(b>>8)/255.0 - 0.5) / 0.5)
		}
	}
	return data
}

// decodeOutputFloatFast 解码输出
func decodeOutputFloatFast(output []float32, shape ort.Shape, charsets []string, charsetLen int, allowedIndices []int) string {
	var sb strings.Builder
	sb.Grow(16)

	var timesteps, batchSize, numClasses int
	if len(shape) == 3 {
		timesteps = int(shape[0])
		batchSize = int(shape[1])
		numClasses = int(shape[2])
	} else if len(shape) == 2 {
		timesteps = int(shape[0])
		numClasses = int(shape[1])
		batchSize = 1
	} else {
		return ""
	}

	lastIdx := -1

	for t := 0; t < timesteps; t++ {
		offset := t * batchSize * numClasses

		maxIdx := 0
		maxVal := float32(-1e9)

		if len(allowedIndices) > 0 {
			for _, c := range allowedIndices {
				if c >= numClasses {
					continue
				}
				val := output[offset+c]
				if val > maxVal {
					maxVal = val
					maxIdx = c
				}
			}
		} else {
			maxVal = output[offset]
			for c := 1; c < numClasses; c++ {
				if output[offset+c] > maxVal {
					maxVal = output[offset+c]
					maxIdx = c
				}
			}
		}

		if maxIdx != lastIdx && maxIdx != 0 && maxIdx < charsetLen {
			sb.WriteString(charsets[maxIdx])
		}
		lastIdx = maxIdx
	}
	return sb.String()
}

// softmax 计算 softmax
func softmax(input []float32) []float32 {
	output := make([]float32, len(input))
	var maxVal float32 = input[0]
	for _, v := range input[1:] {
		if v > maxVal {
			maxVal = v
		}
	}

	var sum float32
	for i, v := range input {
		output[i] = float32(math.Exp(float64(v - maxVal)))
		sum += output[i]
	}

	for i := range output {
		output[i] /= sum
	}
	return output
}

// ============================================================================
// 目标检测辅助函数
// ============================================================================

func decodeImageCV(data []byte) (image.Image, error) {
	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, err
	}

	bounds := img.Bounds()
	rgba := image.NewRGBA(bounds)
	draw.Draw(rgba, bounds, img, bounds.Min, draw.Src)
	return rgba, nil
}

func preproc(img image.Image, inputSize [2]int) ([]float32, float64) {
	bounds := img.Bounds()
	origH := bounds.Dy()
	origW := bounds.Dx()

	r := math.Min(float64(inputSize[0])/float64(origH), float64(inputSize[1])/float64(origW))

	newH := int(float64(origH) * r)
	newW := int(float64(origW) * r)

	resized := resize.Resize(uint(newW), uint(newH), img, resize.Bilinear)

	padded := make([]float32, 3*inputSize[0]*inputSize[1])
	for i := range padded {
		padded[i] = 114.0
	}

	resizedBounds := resized.Bounds()
	for y := 0; y < newH; y++ {
		for x := 0; x < newW; x++ {
			r, g, b, _ := resized.At(x+resizedBounds.Min.X, y+resizedBounds.Min.Y).RGBA()
			idx := y*inputSize[1] + x
			padded[idx] = float32(r >> 8)
			padded[inputSize[0]*inputSize[1]+idx] = float32(g >> 8)
			padded[2*inputSize[0]*inputSize[1]+idx] = float32(b >> 8)
		}
	}

	return padded, r
}

func demoPostprocess(outputs []float32, imgSize [2]int) [][]float32 {
	strides := []int{8, 16, 32}

	var grids [][]float32
	var expandedStrides []float32

	for _, stride := range strides {
		hsize := imgSize[0] / stride
		wsize := imgSize[1] / stride

		for y := 0; y < hsize; y++ {
			for x := 0; x < wsize; x++ {
				grids = append(grids, []float32{float32(x), float32(y)})
				expandedStrides = append(expandedStrides, float32(stride))
			}
		}
	}

	numAnchors := len(grids)
	numClasses := len(outputs)/numAnchors - 5

	predictions := make([][]float32, numAnchors)
	for i := 0; i < numAnchors; i++ {
		pred := make([]float32, 5+numClasses)

		pred[0] = (outputs[i*(5+numClasses)+0] + grids[i][0]) * expandedStrides[i]
		pred[1] = (outputs[i*(5+numClasses)+1] + grids[i][1]) * expandedStrides[i]

		pred[2] = float32(math.Exp(float64(outputs[i*(5+numClasses)+2]))) * expandedStrides[i]
		pred[3] = float32(math.Exp(float64(outputs[i*(5+numClasses)+3]))) * expandedStrides[i]

		for j := 4; j < 5+numClasses; j++ {
			pred[j] = outputs[i*(5+numClasses)+j]
		}

		predictions[i] = pred
	}

	return predictions
}

func multiclassNMS(predictions [][]float32, ratio float64, origW, origH int, nmsThr, scoreThr float32) []BBox {
	type detection struct {
		box   BBox
		score float32
	}

	var dets []detection

	for _, pred := range predictions {
		objScore := pred[4]
		maxClassScore := float32(0)
		for j := 5; j < len(pred); j++ {
			if pred[j] > maxClassScore {
				maxClassScore = pred[j]
			}
		}
		score := objScore * maxClassScore

		if score < scoreThr {
			continue
		}

		cx, cy, w, h := pred[0], pred[1], pred[2], pred[3]
		x1 := int((cx - w/2) / float32(ratio))
		y1 := int((cy - h/2) / float32(ratio))
		x2 := int((cx + w/2) / float32(ratio))
		y2 := int((cy + h/2) / float32(ratio))

		if x1 < 0 {
			x1 = 0
		}
		if y1 < 0 {
			y1 = 0
		}
		if x2 > origW {
			x2 = origW
		}
		if y2 > origH {
			y2 = origH
		}

		dets = append(dets, detection{
			box:   BBox{X1: x1, Y1: y1, X2: x2, Y2: y2},
			score: score,
		})
	}

	sort.Slice(dets, func(i, j int) bool {
		return dets[i].score > dets[j].score
	})

	var result []BBox
	used := make([]bool, len(dets))

	for i := 0; i < len(dets); i++ {
		if used[i] {
			continue
		}
		result = append(result, dets[i].box)

		for j := i + 1; j < len(dets); j++ {
			if used[j] {
				continue
			}
			iou := computeIoU(dets[i].box, dets[j].box)
			if iou > nmsThr {
				used[j] = true
			}
		}
	}

	return result
}

func computeIoU(a, b BBox) float32 {
	x1 := max(a.X1, b.X1)
	y1 := max(a.Y1, b.Y1)
	x2 := min(a.X2, b.X2)
	y2 := min(a.Y2, b.Y2)

	if x2 <= x1 || y2 <= y1 {
		return 0
	}

	inter := float32((x2 - x1) * (y2 - y1))
	areaA := float32((a.X2 - a.X1) * (a.Y2 - a.Y1))
	areaB := float32((b.X2 - b.X1) * (b.Y2 - b.Y1))

	return inter / (areaA + areaB - inter)
}

// ============================================================================
// 滑块验证码辅助函数
// ============================================================================

func getTarget(imgBytes []byte) (image.Image, int, int, error) {
	img, _, err := image.Decode(bytes.NewReader(imgBytes))
	if err != nil {
		return nil, 0, 0, err
	}

	bounds := img.Bounds()
	w := bounds.Dx()
	h := bounds.Dy()

	var startX, startY, endX, endY int
	startX = w
	startY = h

	nrgba, isNRGBA := img.(*image.NRGBA)
	rgba, isRGBA := img.(*image.RGBA)

	for x := bounds.Min.X; x < bounds.Max.X; x++ {
		for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
			var alpha uint8
			if isNRGBA {
				alpha = nrgba.NRGBAAt(x, y).A
			} else if isRGBA {
				alpha = rgba.RGBAAt(x, y).A
			} else {
				_, _, _, a := img.At(x, y).RGBA()
				alpha = uint8(a >> 8)
			}

			if alpha > 0 {
				localX := x - bounds.Min.X
				localY := y - bounds.Min.Y

				if localX < startX {
					startX = localX
				}
				if localY < startY {
					startY = localY
				}
				if localX > endX {
					endX = localX
				}
				if localY > endY {
					endY = localY
				}
			}
		}
	}

	if startX >= endX || startY >= endY {
		return nil, 0, 0, fmt.Errorf("无法提取有效区域")
	}

	cropped := image.NewRGBA(image.Rect(0, 0, endX-startX, endY-startY))
	for y := startY; y < endY; y++ {
		for x := startX; x < endX; x++ {
			cropped.Set(x-startX, y-startY, img.At(x+bounds.Min.X, y+bounds.Min.Y))
		}
	}

	return cropped, startX, startY, nil
}

// ============================================================================
// 工具函数
// ============================================================================

// SaveImage 保存图片
func SaveImage(img image.Image, path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	return png.Encode(f, img)
}

// Close 关闭识别器
func (d *DdddOcr) Close() error {
	if d.session != nil {
		d.session.Destroy()
	}
	if d.options != nil {
		d.options.Destroy()
	}
	if d.detSession != nil {
		d.detSession.Destroy()
	}
	if d.detOptions != nil {
		d.detOptions.Destroy()
	}
	return nil
}

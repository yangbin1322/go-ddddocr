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
	"math"
	"os"
	"path/filepath"
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
			// 默认路径，根据系统调整
			onnxRuntimePath = "onnxruntime.dll"
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
}

// DefaultOptions 默认选项
func DefaultOptions() Options {
	return Options{
		Ocr:      true,
		Det:      false,
		Beta:     false,
		UseGPU:   false,
		DeviceID: 0,
		ModelDir: ".",
	}
}

// New 创建识别器
func New(opts Options) (*DdddOcr, error) {
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
	// 这一步很重要！Python 是在 3 通道图像上做模板匹配
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
			// OpenCV 使用 BGR，但灰度转换公式相同
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
			// Sobel X
			gx := -gray[y-1][x-1] - 2*gray[y][x-1] - gray[y+1][x-1] +
				gray[y-1][x+1] + 2*gray[y][x+1] + gray[y+1][x+1]

			// Sobel Y
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

			// 计算梯度方向
			gx := float64(gradX[y][x])
			gy := float64(gradY[y][x])
			angle := math.Atan2(gy, gx) * 180 / math.Pi
			if angle < 0 {
				angle += 180
			}

			var q, r int
			// 根据方向选择比较的邻居
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
	edge := make([][]int, h) // 0: none, 1: weak, 2: strong
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
					// 检查8邻域是否有强边缘
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

// grayToRGB 灰度图转 RGB（模拟 cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)）
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

	// 预计算模板数据（3通道）
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

	// 计算模板的归一化数据和标准差
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

	// 滑动窗口匹配
	for y := 0; y <= bgH-tplH; y++ {
		for x := 0; x <= bgW-tplW; x++ {
			// 计算窗口均值
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

			// 计算相关系数
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

			// TM_CCOEFF_NORMED
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
// 参数: targetBytes - 带缺口阴影的图片, backgroundBytes - 完整图片
// 与 Python 版本行为一致：计算 background - target 的差异
func (d *DdddOcr) SlideComparison(targetBytes, backgroundBytes []byte) (*SlideComparisonResult, error) {
	// target: 带缺口阴影的图片（第一个参数）
	target, _, err := image.Decode(bytes.NewReader(targetBytes))
	if err != nil {
		return nil, fmt.Errorf("解码目标图失败: %w", err)
	}

	// background: 完整图片（第二个参数）
	background, _, err := image.Decode(bytes.NewReader(backgroundBytes))
	if err != nil {
		return nil, fmt.Errorf("解码背景图失败: %w", err)
	}

	// 获取尺寸
	targetBounds := target.Bounds()
	bgBounds := background.Bounds()

	// 如果尺寸不一致，需要调整（Python 的 PIL 会自动处理）
	// 通常带缺口的图较大，完整图可能需要缩放
	if targetBounds.Dx() != bgBounds.Dx() || targetBounds.Dy() != bgBounds.Dy() {
		// 将完整图缩放到与带缺口图相同的尺寸
		background = resize.Resize(
			uint(targetBounds.Dx()),
			uint(targetBounds.Dy()),
			background,
			resize.Bilinear,
		)
	}

	// 转换为 RGB（与 Python 的 convert("RGB") 对应）
	targetRGB := toRGB(target)
	backgroundRGB := toRGB(background)

	// 计算图像差异（Python: ImageChops.difference(background, target)）
	// 注意：Python 是 background - target，所以这里也要保持一致
	diff := imageDifferenceRGB(backgroundRGB, targetRGB)

	// 二值化处理（point(lambda x: 255 if x > 80 else 0)）
	threshold := uint8(80)
	binaryImg := binarizeRGB(diff, threshold)

	// 查找缺口位置（使用 Python 完全一致的逻辑）
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

// imageDifferenceRGB 计算两张 RGB 图像的差异（对应 ImageChops.difference）
func imageDifferenceRGB(img1, img2 image.Image) *image.RGBA {
	bounds := img1.Bounds()
	bounds2 := img2.Bounds()

	// 使用较小的尺寸
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

			// 计算绝对差值
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

// binarizeRGB 对 RGB 图像进行二值化（每个通道独立处理）
func binarizeRGB(img *image.RGBA, threshold uint8) *image.RGBA {
	bounds := img.Bounds()
	result := image.NewRGBA(bounds)

	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			r, g, b, _ := img.At(x, y).RGBA()
			r8, g8, b8 := uint8(r>>8), uint8(g>>8), uint8(b>>8)

			// 每个通道独立二值化
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

// findGapPython 完全按照 Python 逻辑查找缺口位置
func findGapPython(img image.Image, minCount int) (int, int) {
	bounds := img.Bounds()
	w := bounds.Dx()
	h := bounds.Dy()

	startX := 0
	startY := 0

	// 按列扫描（与 Python 的 for i in range(0, image.width) 对应）
	for x := 0; x < w; x++ {
		count := 0
		for y := 0; y < h; y++ {
			r, g, b, _ := img.At(x+bounds.Min.X, y+bounds.Min.Y).RGBA()
			r8, g8, b8 := uint8(r>>8), uint8(g>>8), uint8(b>>8)

			// 检查像素是否非黑色（与 Python 的 pixel != (0, 0, 0) 对应）
			if r8 != 0 || g8 != 0 || b8 != 0 {
				count++
				// 记录第一个非黑色像素的 Y 坐标
				if count == 1 && startY == 0 {
					startY = y
				}
			}
		}

		// 如果这一列有足够多的非黑色像素，记录 X 坐标
		if count >= minCount {
			startX = x + 2 // Python 中是 start_x = i + 2
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

	// 填充白色背景
	draw.Draw(result, bounds, &image.Uniform{color.White}, image.Point{}, draw.Src)

	// 绘制原图
	draw.Draw(result, bounds, img, bounds.Min, draw.Over)

	return result
}

// filterByColors 颜色过滤
func filterByColors(img image.Image, colors []string, customRanges map[string]HSVRange) image.Image {
	bounds := img.Bounds()
	result := image.NewRGBA(bounds)

	// 合并颜色范围
	ranges := make(map[string]HSVRange)
	for k, v := range DefaultColorRanges {
		ranges[k] = v
	}
	for k, v := range customRanges {
		ranges[k] = v
	}

	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			r, g, b, a := img.At(x, y).RGBA()
			r8, g8, b8 := uint8(r>>8), uint8(g>>8), uint8(b>>8)

			h, s, v := rgbToHSV(r8, g8, b8)

			keep := false
			for _, colorName := range colors {
				if rng, ok := ranges[colorName]; ok {
					if isInHSVRange(h, s, v, rng) {
						keep = true
						break
					}
				}
				// 处理红色的两段
				if colorName == "red" {
					if rng, ok := ranges["red2"]; ok {
						if isInHSVRange(h, s, v, rng) {
							keep = true
							break
						}
					}
				}
			}

			if keep {
				result.Set(x, y, color.RGBA{r8, g8, b8, uint8(a >> 8)})
			} else {
				result.Set(x, y, color.White)
			}
		}
	}

	return result
}

// rgbToHSV RGB 转 HSV
func rgbToHSV(r, g, b uint8) (h, s, v uint8) {
	rf := float64(r) / 255.0
	gf := float64(g) / 255.0
	bf := float64(b) / 255.0

	max := math.Max(rf, math.Max(gf, bf))
	min := math.Min(rf, math.Min(gf, bf))
	delta := max - min

	// V
	v = uint8(max * 255)

	// S
	if max == 0 {
		s = 0
	} else {
		s = uint8((delta / max) * 255)
	}

	// H
	var hf float64
	if delta == 0 {
		hf = 0
	} else if max == rf {
		hf = 60 * math.Mod((gf-bf)/delta, 6)
	} else if max == gf {
		hf = 60 * ((bf-rf)/delta + 2)
	} else {
		hf = 60 * ((rf-gf)/delta + 4)
	}
	if hf < 0 {
		hf += 360
	}
	h = uint8(hf / 2) // OpenCV 的 H 范围是 0-180

	return
}

// isInHSVRange 检查是否在 HSV 范围内
func isInHSVRange(h, s, v uint8, rng HSVRange) bool {
	return h >= rng.LowH && h <= rng.HighH &&
		s >= rng.LowS && s <= rng.HighS &&
		v >= rng.LowV && v <= rng.HighV
}

// imageToGrayFloat32 图像转灰度 float32
func imageToGrayFloat32(img image.Image, width, height int, useImportOnnx bool) []float32 {
	out := make([]float32, height*width)
	bounds := img.Bounds()

	idx := 0
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			r, g, b, _ := img.At(x+bounds.Min.X, y+bounds.Min.Y).RGBA()
			// 灰度计算
			gray := float32(0.299*float64(r)+0.587*float64(g)+0.114*float64(b)) / 65535.0

			// 归一化
			if useImportOnnx {
				out[idx] = (gray - 0.456) / 0.224
			} else {
				out[idx] = (gray - 0.5) / 0.5
			}
			idx++
		}
	}
	return out
}

// imageToRGBFloat32 图像转 RGB float32 (CHW 格式)
func imageToRGBFloat32(img image.Image, width, height int) []float32 {
	out := make([]float32, 3*height*width)
	bounds := img.Bounds()

	mean := []float64{0.485, 0.456, 0.406}
	std := []float64{0.229, 0.224, 0.225}

	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			r, g, b, _ := img.At(x+bounds.Min.X, y+bounds.Min.Y).RGBA()
			rf := float64(r) / 65535.0
			gf := float64(g) / 65535.0
			bf := float64(b) / 65535.0

			idx := y*width + x
			out[idx] = float32((rf - mean[0]) / std[0])
			out[height*width+idx] = float32((gf - mean[1]) / std[1])
			out[2*height*width+idx] = float32((bf - mean[2]) / std[2])
		}
	}
	return out
}

// decodeOutputFloatFast 解码输出
func decodeOutputFloatFast(output []float32, shape []int64, charsets []string, charsetLen int, allowedIndices []int) string {
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

// decodeImageCV 解码图片为 RGBA
func decodeImageCV(data []byte) (image.Image, error) {
	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, err
	}

	// 转换为 RGBA
	bounds := img.Bounds()
	rgba := image.NewRGBA(bounds)
	draw.Draw(rgba, bounds, img, bounds.Min, draw.Src)
	return rgba, nil
}

// preproc 预处理图像
func preproc(img image.Image, inputSize [2]int) ([]float32, float64) {
	bounds := img.Bounds()
	origH := bounds.Dy()
	origW := bounds.Dx()

	// 计算缩放比例
	r := math.Min(float64(inputSize[0])/float64(origH), float64(inputSize[1])/float64(origW))

	newH := int(float64(origH) * r)
	newW := int(float64(origW) * r)

	// 缩放图像
	resized := resize.Resize(uint(newW), uint(newH), img, resize.Bilinear)

	// 创建填充图像 (114 灰色背景)
	padded := make([]float32, 3*inputSize[0]*inputSize[1])
	for i := range padded {
		padded[i] = 114.0
	}

	// 填充数据 (CHW 格式)
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

// demoPostprocess 后处理检测输出
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

	// 解析输出
	numAnchors := len(grids)
	numClasses := len(outputs)/numAnchors - 5

	predictions := make([][]float32, numAnchors)
	for i := 0; i < numAnchors; i++ {
		pred := make([]float32, 5+numClasses)

		// 中心点
		pred[0] = (outputs[i*(5+numClasses)+0] + grids[i][0]) * expandedStrides[i]
		pred[1] = (outputs[i*(5+numClasses)+1] + grids[i][1]) * expandedStrides[i]

		// 宽高
		pred[2] = float32(math.Exp(float64(outputs[i*(5+numClasses)+2]))) * expandedStrides[i]
		pred[3] = float32(math.Exp(float64(outputs[i*(5+numClasses)+3]))) * expandedStrides[i]

		// 置信度和类别
		for j := 4; j < 5+numClasses; j++ {
			pred[j] = outputs[i*(5+numClasses)+j]
		}

		predictions[i] = pred
	}

	return predictions
}

// multiclassNMS 多类别 NMS
func multiclassNMS(predictions [][]float32, ratio float64, origW, origH int, nmsThr, scoreThr float32) []BBox {
	type detection struct {
		box   BBox
		score float32
	}

	var dets []detection

	for _, pred := range predictions {
		// 计算 objectness * class_score
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

		// 转换为 xyxy 格式
		cx, cy, w, h := pred[0], pred[1], pred[2], pred[3]
		x1 := int((cx - w/2) / float32(ratio))
		y1 := int((cy - h/2) / float32(ratio))
		x2 := int((cx + w/2) / float32(ratio))
		y2 := int((cy + h/2) / float32(ratio))

		// 边界裁剪
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

	// 按分数排序
	sort.Slice(dets, func(i, j int) bool {
		return dets[i].score > dets[j].score
	})

	// NMS
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

// computeIoU 计算 IoU
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

// getTarget 提取滑块透明区域
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

	// 检测 NRGBA 类型以获取 alpha 通道
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

	// 裁剪
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

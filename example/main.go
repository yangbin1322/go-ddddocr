package main

import (
	"fmt"
	"go-ddddocr/ddddocr"
	"os"
)

func main() {
	// ========================================================================
	// 示例 1: 基础 OCR 识别
	// ========================================================================
	fmt.Println("=== 示例 1: 基础 OCR 识别 ===")

	// 设置 ONNX Runtime 路径
	ddddocr.SetOnnxRuntimePath("E:\\BaiduSyncdisk\\golang\\src\\go-ddddocr\\onnxruntime.dll")

	// 创建 OCR 识别器
	ocr, err := ddddocr.New(ddddocr.DefaultOptions())
	if err != nil {
		fmt.Printf("创建 OCR 失败: %v\n", err)
		return
	}
	defer ocr.Close()

	// 读取图片
	imageData, err := os.ReadFile("1.png")
	if err != nil {
		fmt.Printf("读取图片失败: %v\n", err)
		return
	}

	// 识别
	result, err := ocr.Classification(imageData)
	if err != nil {
		fmt.Printf("识别失败: %v\n", err)
		return
	}
	fmt.Printf("识别结果: %s\n\n", result)

	// ========================================================================
	// 示例 2: 设置字符范围
	// ========================================================================
	fmt.Println("=== 示例 2: 设置字符范围 ===")

	// 只识别数字
	ocr.SetRanges(ddddocr.RangeDigit)
	result, _ = ocr.Classification(imageData)
	fmt.Printf("仅数字: %s\n", result)

	// 只识别小写字母和数字
	ocr.SetRanges(ddddocr.RangeLowerDigit)
	result, _ = ocr.Classification(imageData)
	fmt.Printf("小写+数字: %s\n", result)

	// 自定义字符范围
	ocr.SetRanges("0123456789+-x/=")
	result, _ = ocr.Classification(imageData)
	fmt.Printf("自定义范围: %s\n", result)

	// 清除范围限制
	ocr.ClearRanges()
	fmt.Println()

	// ========================================================================
	// 示例 3: 处理透明 PNG
	// ========================================================================
	fmt.Println("=== 示例 3: 处理透明 PNG ===")

	pngData, _ := os.ReadFile("1.png")
	result, _ = ocr.ClassificationWithOptions(pngData, ddddocr.ClassificationOptions{
		PngFix: true,
	})
	fmt.Printf("透明PNG识别: %s\n\n", result)

	// ========================================================================
	// 示例 4: 颜色过滤
	// ========================================================================
	fmt.Println("=== 示例 4: 颜色过滤 ===")

	colorImage, _ := os.ReadFile("1.png")
	result, _ = ocr.ClassificationWithOptions(colorImage, ddddocr.ClassificationOptions{
		Colors: []string{"black"},
	})
	fmt.Printf("颜色过滤识别: %s\n", result)

	// 自定义颜色范围
	result, _ = ocr.ClassificationWithOptions(colorImage, ddddocr.ClassificationOptions{
		Colors: []string{"light_blue"},
		ColorRanges: map[string]ddddocr.HSVRange{
			"light_blue": {90, 30, 30, 110, 255, 255},
		},
	})
	fmt.Printf("自定义颜色过滤: %s\n\n", result)

	// ========================================================================
	// 示例 5: 获取概率输出
	// ========================================================================
	fmt.Println("=== 示例 5: 概率输出 ===")

	probResult, err := ocr.ClassificationProbability(imageData)
	if err != nil {
		fmt.Printf("获取概率失败: %v\n", err)
	} else {
		fmt.Printf("识别文本: %s\n", probResult.Text)
		fmt.Printf("字符集大小: %d\n", len(probResult.Charsets))
		fmt.Printf("时间步数: %d\n\n", len(probResult.Probability))
	}

	// ========================================================================
	// 示例 6: 使用 Beta 模型
	// ========================================================================
	fmt.Println("=== 示例 6: Beta 模型 ===")

	betaOpts := ddddocr.DefaultOptions()
	betaOpts.Beta = true
	betaOcr, err := ddddocr.New(betaOpts)
	if err != nil {
		fmt.Printf("创建 Beta OCR 失败: %v\n", err)
	} else {
		defer betaOcr.Close()
		result, _ = betaOcr.Classification(imageData)
		fmt.Printf("Beta模型识别: %s\n\n", result)
	}

	// ========================================================================
	// 示例 7: 目标检测
	// ========================================================================
	fmt.Println("=== 示例 7: 目标检测 ===")

	detOpts := ddddocr.DefaultOptions()
	detOpts.Ocr = false
	detOpts.Det = true
	det, err := ddddocr.New(detOpts)
	if err != nil {
		fmt.Printf("创建检测器失败: %v\n", err)
	} else {
		defer det.Close()

		detImage, _ := os.ReadFile("dx1.jpg")
		bboxes, err := det.Detection(detImage)
		if err != nil {
			fmt.Printf("检测失败: %v\n", err)
		} else {
			fmt.Printf("检测到 %d 个目标:\n", len(bboxes))
			for i, bbox := range bboxes {
				fmt.Printf("  目标 %d: [%d, %d, %d, %d]\n", i+1, bbox.X1, bbox.Y1, bbox.X2, bbox.Y2)
			}
		}
	}
	fmt.Println()

	// ========================================================================
	// 示例 8: 滑块验证码 - 边缘匹配
	// ========================================================================
	fmt.Println("=== 示例 8: 滑块验证码 (边缘匹配) ===")

	slideOpts := ddddocr.DefaultOptions()
	slideOpts.Ocr = false
	slideOpts.Det = false
	slide, err := ddddocr.New(slideOpts)
	if err != nil {
		fmt.Printf("创建滑块识别器失败: %v\n", err)
	} else {
		defer slide.Close()

		targetBytes, _ := os.ReadFile("slideImage4.png")
		bgBytes, _ := os.ReadFile("bgImage4.png")

		// 有透明背景的滑块
		matchResult, err := slide.SlideMatch(targetBytes, bgBytes, false)
		if err != nil {
			fmt.Printf("滑块匹配失败: %v\n", err)
		} else {
			fmt.Printf("滑块位置: %v\n", matchResult.Target)
			fmt.Printf("滑块原始偏移: (%d, %d)\n", matchResult.TargetX, matchResult.TargetY)
		}

		// 无透明背景的滑块
		matchResult2, err := slide.SlideMatch(targetBytes, bgBytes, true)
		if err != nil {
			fmt.Printf("简单匹配失败: %v\n", err)
		} else {
			fmt.Printf("简单匹配位置: %v\n", matchResult2.Target)
		}
	}
	fmt.Println()

	// ========================================================================
	// 示例 9: 滑块验证码 - 图像差异比较
	// ========================================================================
	fmt.Println("=== 示例 9: 滑块验证码 (图像差异) ===")

	if slide != nil {
		gapImage, _ := os.ReadFile("bgImage2.png")
		fullImage, _ := os.ReadFile("滑块完整背景图3.jpg")
		fmt.Println(len(gapImage), len(fullImage))
		compResult, err := slide.SlideComparison(gapImage, fullImage)
		if err != nil {
			fmt.Printf("图像比较失败: %v\n", err)
		} else {
			fmt.Printf("缺口位置: (%d, %d)\n", compResult.Target[0], compResult.Target[1])
		}
	}
	fmt.Println()

	// ========================================================================
	// 示例 10: 自定义模型
	// ========================================================================
	fmt.Println("=== 示例 10: 自定义模型 ===")

	customOpts := ddddocr.Options{
		ImportOnnxPath: "custom_model.onnx",
		CharsetsPath:   "custom_charsets.json",
	}
	customOcr, err := ddddocr.New(customOpts)
	if err != nil {
		fmt.Printf("加载自定义模型失败: %v\n", err)
	} else {
		defer customOcr.Close()
		result, _ = customOcr.Classification(imageData)
		fmt.Printf("自定义模型识别: %s\n", result)
	}
}

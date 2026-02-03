package main

import (
	"fmt"
	"go-ddddocr/ddddocr"
	"os"
	"sync"
	"sync/atomic"
	"time"
)

func main() {
	// 设置 ONNX Runtime 路径
	ddddocr.SetOnnxRuntimePath("E:\\BaiduSyncdisk\\golang\\src\\go-ddddocr\\onnxruntime.dll")

	fmt.Println("========================================")
	fmt.Println("       Go-DdddOcr 并发测试")
	fmt.Println("========================================")
	fmt.Println()

	// 测试 1: 单实例加锁并发
	testSingleInstanceWithLock()

	// 测试 2: 多实例并发（每个协程独立实例）
	testMultipleInstances()

	// 测试 3: 实例池并发
	testInstancePool()

	// 测试 4: 滑块识别并发
	testSlideMatchConcurrent()
}

// ============================================================================
// 测试 1: 单实例 + 互斥锁
// ============================================================================
func testSingleInstanceWithLock() {
	fmt.Println("=== 测试 1: 单实例 + 互斥锁 ===")

	ocr, err := ddddocr.New(ddddocr.DefaultOptions())
	if err != nil {
		fmt.Printf("创建 OCR 失败: %v\n", err)
		return
	}
	defer ocr.Close()

	imageData, err := os.ReadFile("1.png")
	if err != nil {
		fmt.Printf("读取图片失败: %v\n", err)
		return
	}

	var mu sync.Mutex
	var wg sync.WaitGroup
	var successCount, errorCount int64

	concurrency := 10
	iterations := 100

	start := time.Now()

	for i := 0; i < concurrency; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for j := 0; j < iterations; j++ {
				mu.Lock()
				result, err := ocr.Classification(imageData)
				mu.Unlock()

				if err != nil {
					atomic.AddInt64(&errorCount, 1)
				} else {
					atomic.AddInt64(&successCount, 1)
					if j == 0 && id == 0 {
						fmt.Printf("  示例结果: %s\n", result)
					}
				}
			}
		}(i)
	}

	wg.Wait()
	elapsed := time.Since(start)

	total := concurrency * iterations
	qps := float64(total) / elapsed.Seconds()

	fmt.Printf("  并发数: %d, 每协程迭代: %d, 总请求: %d\n", concurrency, iterations, total)
	fmt.Printf("  成功: %d, 失败: %d\n", successCount, errorCount)
	fmt.Printf("  耗时: %v, QPS: %.2f\n\n", elapsed, qps)
}

// ============================================================================
// 测试 2: 多实例并发（每个协程独立实例）
// ============================================================================
func testMultipleInstances() {
	fmt.Println("=== 测试 2: 多实例并发 ===")

	imageData, err := os.ReadFile("1.png")
	if err != nil {
		fmt.Printf("读取图片失败: %v\n", err)
		return
	}

	var wg sync.WaitGroup
	var successCount, errorCount int64

	concurrency := 5
	iterations := 100

	start := time.Now()

	for i := 0; i < concurrency; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()

			// 每个协程创建自己的实例
			ocr, err := ddddocr.New(ddddocr.DefaultOptions())
			if err != nil {
				fmt.Printf("  协程 %d 创建 OCR 失败: %v\n", id, err)
				atomic.AddInt64(&errorCount, int64(iterations))
				return
			}
			defer ocr.Close()

			for j := 0; j < iterations; j++ {
				result, err := ocr.Classification(imageData)
				if err != nil {
					atomic.AddInt64(&errorCount, 1)
				} else {
					atomic.AddInt64(&successCount, 1)
					if j == 0 && id == 0 {
						fmt.Printf("  示例结果: %s\n", result)
					}
				}
			}
		}(i)
	}

	wg.Wait()
	elapsed := time.Since(start)

	total := concurrency * iterations
	qps := float64(total) / elapsed.Seconds()

	fmt.Printf("  并发数: %d, 每协程迭代: %d, 总请求: %d\n", concurrency, iterations, total)
	fmt.Printf("  成功: %d, 失败: %d\n", successCount, errorCount)
	fmt.Printf("  耗时: %v, QPS: %.2f\n\n", elapsed, qps)
}

// ============================================================================
// 测试 3: 实例池
// ============================================================================
type OcrPool struct {
	pool chan *ddddocr.DdddOcr
	size int
}

func NewOcrPool(size int) (*OcrPool, error) {
	pool := &OcrPool{
		pool: make(chan *ddddocr.DdddOcr, size),
		size: size,
	}

	for i := 0; i < size; i++ {
		ocr, err := ddddocr.New(ddddocr.DefaultOptions())
		if err != nil {
			pool.Close()
			return nil, err
		}
		pool.pool <- ocr
	}

	return pool, nil
}

func (p *OcrPool) Get() *ddddocr.DdddOcr {
	return <-p.pool
}

func (p *OcrPool) Put(ocr *ddddocr.DdddOcr) {
	p.pool <- ocr
}

func (p *OcrPool) Close() {
	close(p.pool)
	for ocr := range p.pool {
		ocr.Close()
	}
}

func testInstancePool() {
	fmt.Println("=== 测试 3: 实例池并发 ===")

	poolSize := 5
	pool, err := NewOcrPool(poolSize)
	if err != nil {
		fmt.Printf("创建实例池失败: %v\n", err)
		return
	}
	defer pool.Close()

	imageData, err := os.ReadFile("1.png")
	if err != nil {
		fmt.Printf("读取图片失败: %v\n", err)
		return
	}

	var wg sync.WaitGroup
	var successCount, errorCount int64

	concurrency := 20
	iterations := 50

	start := time.Now()

	for i := 0; i < concurrency; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()

			for j := 0; j < iterations; j++ {
				// 从池中获取实例
				ocr := pool.Get()

				result, err := ocr.Classification(imageData)

				// 归还实例
				pool.Put(ocr)

				if err != nil {
					atomic.AddInt64(&errorCount, 1)
				} else {
					atomic.AddInt64(&successCount, 1)
					if j == 0 && id == 0 {
						fmt.Printf("  示例结果: %s\n", result)
					}
				}
			}
		}(i)
	}

	wg.Wait()
	elapsed := time.Since(start)

	total := concurrency * iterations
	qps := float64(total) / elapsed.Seconds()

	fmt.Printf("  池大小: %d, 并发数: %d, 每协程迭代: %d, 总请求: %d\n", poolSize, concurrency, iterations, total)
	fmt.Printf("  成功: %d, 失败: %d\n", successCount, errorCount)
	fmt.Printf("  耗时: %v, QPS: %.2f\n\n", elapsed, qps)
}

// ============================================================================
// 测试 4: 滑块识别并发
// ============================================================================
func testSlideMatchConcurrent() {
	fmt.Println("=== 测试 4: 滑块识别并发 ===")

	targetBytes, err := os.ReadFile("slideImage4.png")
	if err != nil {
		fmt.Printf("读取滑块图失败: %v\n", err)
		return
	}

	bgBytes, err := os.ReadFile("bgImage4.png")
	if err != nil {
		fmt.Printf("读取背景图失败: %v\n", err)
		return
	}

	var wg sync.WaitGroup
	var successCount, errorCount int64

	concurrency := 5
	iterations := 20

	start := time.Now()

	for i := 0; i < concurrency; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()

			// 滑块识别不需要加载模型，每个协程可以独立创建
			opts := ddddocr.DefaultOptions()
			opts.Ocr = false
			opts.Det = false
			slide, err := ddddocr.New(opts)
			if err != nil {
				fmt.Printf("  协程 %d 创建滑块识别器失败: %v\n", id, err)
				atomic.AddInt64(&errorCount, int64(iterations))
				return
			}
			defer slide.Close()

			for j := 0; j < iterations; j++ {
				result, err := slide.SlideMatch(targetBytes, bgBytes, false)
				if err != nil {
					atomic.AddInt64(&errorCount, 1)
				} else {
					atomic.AddInt64(&successCount, 1)
					if j == 0 && id == 0 {
						fmt.Printf("  示例结果: %v\n", result.Target)
					}
				}
			}
		}(i)
	}

	wg.Wait()
	elapsed := time.Since(start)

	total := concurrency * iterations
	qps := float64(total) / elapsed.Seconds()

	fmt.Printf("  并发数: %d, 每协程迭代: %d, 总请求: %d\n", concurrency, iterations, total)
	fmt.Printf("  成功: %d, 失败: %d\n", successCount, errorCount)
	fmt.Printf("  耗时: %v, QPS: %.2f\n\n", elapsed, qps)
}

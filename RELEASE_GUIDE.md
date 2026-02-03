# GitHub 发布指南

本文档说明如何将 go-ddddocr 项目上传到 GitHub 并管理大文件资源。

## 方案说明

由于模型文件较大（总计约 99MB），我们使用 **GitHub Releases** 来托管这些文件，而不是直接提交到仓库。这样可以：

1. 保持仓库体积小巧
2. 加快克隆速度
3. 方便用户按需下载
4. 避免 Git LFS 的额外费用

## 上传步骤

### 1. 初始化 Git 仓库

```bash
# 初始化仓库
git init

# 添加所有文件（模型文件已在 .gitignore 中排除）
git add .

# 创建首次提交
git commit -m "Initial commit: Go-DdddOcr implementation

- Complete OCR classification implementation
- Target detection support
- Slide captcha matching
- Concurrent pool support
- Download scripts for model files"
```

### 2. 创建 GitHub 仓库

1. 访问 [GitHub](https://github.com/new)
2. 创建新仓库，例如 `go-ddddocr`
3. 不要初始化 README、.gitignore 或 LICENSE（我们已经有了）

### 3. 关联远程仓库

```bash
# 关联远程仓库（替换为你的用户名）
git remote add origin https://github.com/your-username/go-ddddocr.git

# 推送代码
git branch -M main
git push -u origin main
```

### 4. 创建 GitHub Release

#### 方式一：使用 GitHub 网页界面（推荐）

1. 访问你的仓库页面
2. 点击右侧 "Releases" → "Create a new release"
3. 填写信息：
   - **Tag version**: `v1.0.0`
   - **Release title**: `v1.0.0 - 初始发布`
   - **Description**: 复制下面的模板

```markdown
## Go-DdddOcr v1.0.0

Go 语言实现的 ddddocr 验证码识别库首次发布。

### 功能特性

- ✅ OCR 文字识别
- ✅ 目标检测
- ✅ 滑块验证码匹配
- ✅ 字符范围限制
- ✅ 概率输出
- ✅ 自定义模型支持
- ✅ 并发安全实例池

### 安装说明

1. 安装 Go 包:
   ```bash
   go get github.com/your-username/go-ddddocr
   ```

2. 下载模型文件（从下方 Assets 中下载）

3. 或使用自动下载脚本:
   ```bash
   # Windows
   powershell -ExecutionPolicy Bypass -File scripts/download_models.ps1

   # Linux/macOS
   bash scripts/download_models.sh
   ```

### 模型文件

本次发布包含以下模型文件（请从 Assets 下载）：

| 文件 | 大小 | 说明 |
|------|:----:|------|
| common.onnx | 52MB | Beta OCR 模型 |
| common_det.onnx | 20MB | 目标检测模型 |
| common_old.onnx | 13MB | 默认 OCR 模型 |
| onnxruntime.dll | 14MB | ONNX Runtime 库 (Windows) |
| charsets_beta.json | 56KB | Beta 模型字符集 |
| charsets_old.json | 56KB | 默认模型字符集 |

### 文档

完整文档请查看 [README.md](https://github.com/your-username/go-ddddocr/blob/main/README.md)
```

4. 拖拽上传以下文件到 "Attach binaries" 区域：
   - `common.onnx`
   - `common_det.onnx`
   - `common_old.onnx`
   - `onnxruntime.dll`
   - `charsets_beta.json`
   - `charsets_old.json`

5. 点击 "Publish release"

#### 方式二：使用 GitHub CLI

```bash
# 安装 gh (如果还没安装)
# Windows: winget install GitHub.cli
# macOS: brew install gh
# Linux: 参考 https://cli.github.com/

# 登录
gh auth login

# 创建 Release 并上传文件
gh release create v1.0.0 \
  --title "v1.0.0 - 初始发布" \
  --notes-file RELEASE_NOTES.md \
  common.onnx \
  common_det.onnx \
  common_old.onnx \
  onnxruntime.dll \
  charsets_beta.json \
  charsets_old.json
```

### 5. 更新下载脚本

在 `scripts/download_models.sh` 和 `scripts/download_models.ps1` 中，将：

```bash
GITHUB_REPO="your-username/go-ddddocr"
```

替换为你的实际用户名，例如：

```bash
GITHUB_REPO="zhangsan/go-ddddocr"
```

然后提交更新：

```bash
git add scripts/
git commit -m "Update download scripts with actual repository URL"
git push
```

## 用户使用流程

用户克隆仓库后，有两种方式获取模型文件：

### 方式一：自动下载（推荐）

```bash
# Windows
powershell -ExecutionPolicy Bypass -File scripts/download_models.ps1

# Linux/macOS
bash scripts/download_models.sh

# 或使用 Go
go run scripts/download_models.go
```

### 方式二：手动下载

从 [Releases 页面](https://github.com/your-username/go-ddddocr/releases) 下载所需文件。

## 文件说明

### 提交到仓库的文件

- `ddddocr/` - 源代码
- `example/` - 示例代码
- `scripts/` - 模型下载脚本
- `README.md` - 项目文档
- `.gitignore` - Git 忽略规则（排除模型文件）
- `.gitattributes` - Git LFS 配置（备用方案）
- `go.mod` / `go.sum` - Go 依赖

### 不提交的文件（通过 Release 分发）

- `*.onnx` - ONNX 模型文件
- `onnxruntime.dll` - ONNX Runtime 库
- `charsets*.json` - 字符集配置（这些文件实际很小，也可以提交到仓库）

## 后续版本更新

### 更新代码

```bash
git add .
git commit -m "描述你的更新"
git push
```

### 创建新的 Release

如果模型文件有更新：

```bash
git tag v1.1.0
git push --tags

# 使用 GitHub 网页或 gh CLI 创建新 Release，并上传更新的模型文件
gh release create v1.1.0 --generate-notes new_model.onnx
```

## 替代方案：Git LFS

如果你希望模型文件也能通过 `git clone` 直接获取，可以使用 Git LFS：

### 安装 Git LFS

```bash
# Windows
git lfs install

# macOS
brew install git-lfs
git lfs install

# Linux
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git lfs install
```

### 使用 Git LFS

```bash
# 删除 .gitignore 中对 .onnx 和 .dll 的排除

# 追踪大文件（.gitattributes 已配置）
git lfs track "*.onnx"
git lfs track "*.dll"

# 提交文件
git add .
git commit -m "Add model files via Git LFS"
git push
```

**注意:** Git LFS 免费账户有存储和带宽限制，请根据实际情况选择。

## 常见问题

### Q: 为什么不直接提交模型文件？

A: GitHub 建议单个文件不超过 50MB，仓库总大小不超过 1GB。我们的模型文件总计约 99MB，使用 Release 分发更合适。

### Q: 用户如何验证下载的文件是否完整？

A: 可以在 Release 说明中提供文件的 SHA256 校验和：

```bash
# 生成校验和
sha256sum *.onnx *.dll *.json > checksums.txt

# 用户验证
sha256sum -c checksums.txt
```

### Q: 如何支持不同平台的 ONNX Runtime 库？

A: 可以在 Release 中上传多个平台的库文件：
- `onnxruntime-windows.dll`
- `onnxruntime-linux.so`
- `onnxruntime-macos.dylib`

并在下载脚本中根据平台自动选择。

## 许可证声明

确保模型文件的使用符合原始项目 [sml2h3/ddddocr](https://github.com/sml2h3/ddddocr) 的许可证要求。
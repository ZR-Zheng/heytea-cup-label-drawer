# Models

此目录用于统一管理应用使用的模型文件。

- `anime2sketch/`：存放 Anime2Sketch 的 `.pth` 或 `.bin` 权重。
- `anilines/`：存放 AniLines Detail 的 `detail.pth` 权重。
- `informative_drawings/`：存放 Informative Drawings 的 `model.onnx` 权重。
- 模型权重通常体积较大，已通过 `.gitignore` 排除，不会提交到 Git。
- 可通过应用界面中的“导入”按钮将模型复制到对应目录。

模型来源：

- AniLines Detail：https://github.com/zhenglinpan/AniLines-Anime-Lineart-Extractor
- Informative Drawings ONNX：https://huggingface.co/rocca/informative-drawings-line-art-onnx

# Heytea Cup Label Drawer

喜茶杯贴图片自动手绘工具。程序会把图片转成中心线、动漫线稿、边缘线稿或逐行扫描路径，并通过鼠标在已标定的画布区域中绘制。

## 安装

```powershell
pip install pyautogui opencv-python pillow numpy
```

如需使用“动漫线稿(Anime2Sketch)”模式，需要额外安装 PyTorch：

```powershell
pip install -e ".[anime]"
```

然后通过界面的“导入”按钮添加 Anime2Sketch 的 `netG.pth` 或 `improved.bin` 权重文件。程序会将模型统一保存至 `models/anime2sketch/`。

## 运行

```powershell
python -m heytea_cup_label_drawer
```

旧入口仍可用：

```powershell
python heytea_cup_label_drawer_gui.py
```

## 目录结构

- `models/`：统一管理模型文件；不同模型类型使用独立子目录。模型权重不会提交到 Git。
- `heytea_cup_label_drawer/config.py`：配置模型和默认配置文件路径。
- `heytea_cup_label_drawer/anime2sketch.py`：Anime2Sketch 模型推理，使用 PyTorch 懒加载。
- `heytea_cup_label_drawer/processing.py`：图片预处理、Anime2Sketch 线稿提取、中心线追踪、轮廓路径和逐行扫描算法。中心线模式采用细化骨架、方向桥接、短毛刺剪枝和角度感知路径追踪。
- `heytea_cup_label_drawer/automation.py`：鼠标移动、落笔、抬笔、坐标映射和绘制节奏。
- `heytea_cup_label_drawer/gui.py`：Tkinter 界面、配置读写、预览和绘制任务调度。
- `heytea_cup_label_drawer/main.py`：应用入口。
- `heytea_cup_label_drawer_gui.py`：兼容旧运行方式的薄入口。

## 测试

```powershell
python -m unittest discover -s tests
```

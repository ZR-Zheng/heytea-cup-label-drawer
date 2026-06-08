# Heytea Cup Label Drawer

喜茶杯贴图片自动手绘工具。程序会把图片转成中心线、边缘线稿或逐行扫描路径，并通过鼠标在已标定的画布区域中绘制。

## 安装

```powershell
pip install pyautogui opencv-python pillow numpy
```

## 运行

```powershell
python -m heytea_cup_label_drawer
```

旧入口仍可用：

```powershell
python heytea_cup_label_drawer_gui.py
```

## 目录结构

- `heytea_cup_label_drawer/config.py`：配置模型和默认配置文件路径。
- `heytea_cup_label_drawer/processing.py`：图片预处理、中心线追踪、轮廓路径和逐行扫描算法。中心线模式采用细化骨架、方向桥接、短毛刺剪枝和角度感知路径追踪。
- `heytea_cup_label_drawer/automation.py`：鼠标移动、落笔、抬笔、坐标映射和绘制节奏。
- `heytea_cup_label_drawer/gui.py`：Tkinter 界面、配置读写、预览和绘制任务调度。
- `heytea_cup_label_drawer/main.py`：应用入口。
- `heytea_cup_label_drawer_gui.py`：兼容旧运行方式的薄入口。

## 测试

```powershell
python -m unittest discover -s tests
```

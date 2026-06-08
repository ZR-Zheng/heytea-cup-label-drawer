"""
喜茶杯贴图片自动手绘工具（电脑版微信小程序鼠标版）

用途：
- 你手动在电脑版微信中打开喜茶 GO 小程序，并进入杯贴/喜贴的手绘画布。
- 本程序把图片转成中心线/线稿/逐行扫描路径，然后控制鼠标在画布区域自动拖动绘制。

安装依赖：
    pip install pyautogui opencv-python pillow numpy

运行：
    python heytea_cup_label_drawer_gui_centerline.py

紧急停止：
- 把鼠标迅速移动到屏幕左上角，PyAutoGUI 会触发 FailSafe 停止。
- 或点击 GUI 里的“停止绘制”。

注意：
- 本工具不负责登录、下单、提交订单或绕过任何平台限制。
- 使用前请先手动进入杯贴画布，并确认鼠标拖动能在画布上画出笔迹。
"""

import json
import os
import threading
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np
import pyautogui
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageOps, ImageTk

# Windows 高 DPI 下让屏幕坐标更接近真实像素，避免 125%/150% 缩放导致偏移。
try:
    import ctypes
    ctypes.windll.user32.SetProcessDPIAware()
except Exception:
    pass

# 不把 PyAutoGUI 的 MINIMUM_DURATION 强行设为 0。
# 某些 PyAutoGUI 版本在 duration 很小且 MINIMUM_DURATION=0 时会出现 float division by zero。
# 下面绘制时改用 _safe_move_to() 处理小 duration。
pyautogui.PAUSE = 0.002
pyautogui.FAILSAFE = True

APP_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = APP_DIR / "heytea_draw_config_centerline.json"


@dataclass
class DrawConfig:
    canvas_x: int = 400
    canvas_y: int = 250
    canvas_w: int = 500
    canvas_h: int = 500
    padding: int = 10

    # 黑色线稿优先使用中心线追踪：先把黑线细化成单像素骨架，再拟合成折线绘制。
    # 这样画的是“线的主干”，而不是描黑线外轮廓。
    method: str = "中心线追踪(线稿)"
    canny_low: int = 80
    canny_high: int = 160
    threshold: int = 200
    blur: int = 0
    dark_as_line: bool = True
    keep_aspect: bool = True
    # 中心线模式：对黑色线稿先做轻微闭合，帮助 1~2px 的断点连起来，再细化成骨架。
    # 0 = 不连接；1 = 轻微连接；2 以上会更容易粘连，谨慎使用。
    centerline_bridge_px: int = 1

    epsilon: float = 0.8
    min_path_len: float = 4.0
    max_paths: int = 30000
    point_step: int = 1

    # 逐行扫描模式参数：把二值图中需要绘制的像素按行压缩成水平线段。
    # 最佳效果核心：不要生成大量 1 像素短线，而是合并、延长、慢速、重叠。
    raster_row_step: int = 2
    raster_min_run: int = 4
    raster_gap_tolerance: int = 2
    raster_extend_px: int = 2
    raster_serpentine: bool = True

    # 鼠标绘制稳定性参数。
    # move_duration 是每个插值点之间的最小等待；真正长线会按距离自动分成多个鼠标 move 事件。
    move_duration: float = 0.006
    mouse_step_px: int = 2
    min_stroke_duration: float = 0.035
    stroke_duration_per_100px: float = 0.08
    pre_down_pause: float = 0.010
    pen_down_pause: float = 0.020
    pen_up_pause: float = 0.015
    between_strokes_pause: float = 0.008
    raster_backtrack: bool = False

    start_delay: int = 3
    test_padding: int = 20
    minimize_when_drawing: bool = True


class HeyTeaCupLabelDrawerGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("喜茶杯贴自动手绘工具 - 鼠标版")
        self.root.geometry("1120x760")
        self.root.minsize(980, 680)

        self.config = DrawConfig()
        self.image_path: str | None = None
        self.original_image: Image.Image | None = None
        self.preview_photo: ImageTk.PhotoImage | None = None
        self.processed_photo: ImageTk.PhotoImage | None = None
        self.last_paths: list[np.ndarray] = []
        self.last_debug_image: np.ndarray | None = None

        self.stop_event = threading.Event()
        self.worker_thread: threading.Thread | None = None
        self.calib_top_left: tuple[int, int] | None = None

        self._build_ui()
        self.load_config(silent=True)
        self._sync_config_to_vars()
        self.method_var.trace_add("write", lambda *_: self._update_param_visibility())
        self._update_param_visibility()
        self._log("先在电脑版微信中手动进入杯贴画布，再选择图片、标定画布、测试框、开始绘制。")
        self._log("紧急停止：把鼠标移到屏幕左上角，或点击“停止绘制”。")
        self._log("当前版本已优化断笔：横线延长、断点合并、落笔/抬笔停顿、鼠标插值移动。")
        self._log("黑色线稿推荐使用“中心线追踪(线稿)”：新版默认更保细节，并可轻微连接断线。")
        self._log("如果中心线追踪仍断笔或需要填充感，可以切换到“逐行扫描(横向)”。")

    # ---------- UI ----------
    def _build_ui(self):
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        # 左侧功能区内容较多，使用 Canvas + Scrollbar 做可滚动面板。
        left_outer = ttk.Frame(main)
        left_outer.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        left_canvas = tk.Canvas(left_outer, width=340, highlightthickness=0)
        left_scrollbar = ttk.Scrollbar(left_outer, orient=tk.VERTICAL, command=left_canvas.yview)
        left = ttk.Frame(left_canvas)

        left_window_id = left_canvas.create_window((0, 0), window=left, anchor="nw")
        left_canvas.configure(yscrollcommand=left_scrollbar.set)

        left_canvas.pack(side=tk.LEFT, fill=tk.Y, expand=False)
        left_scrollbar.pack(side=tk.LEFT, fill=tk.Y)

        def _update_left_scroll_region(_event=None):
            left_canvas.configure(scrollregion=left_canvas.bbox("all"))

        def _sync_left_width(event):
            # 让内部 Frame 宽度跟随 Canvas，避免控件被裁切。
            left_canvas.itemconfigure(left_window_id, width=event.width)

        def _on_left_mousewheel(event):
            # Windows/macOS: event.delta；Linux: Button-4/5。
            if getattr(event, "num", None) == 4:
                left_canvas.yview_scroll(-1, "units")
            elif getattr(event, "num", None) == 5:
                left_canvas.yview_scroll(1, "units")
            else:
                delta = int(-1 * (event.delta / 120)) if event.delta else 0
                left_canvas.yview_scroll(delta, "units")

        def _bind_left_mousewheel(_event=None):
            left_canvas.bind_all("<MouseWheel>", _on_left_mousewheel)
            left_canvas.bind_all("<Button-4>", _on_left_mousewheel)
            left_canvas.bind_all("<Button-5>", _on_left_mousewheel)

        def _unbind_left_mousewheel(_event=None):
            left_canvas.unbind_all("<MouseWheel>")
            left_canvas.unbind_all("<Button-4>")
            left_canvas.unbind_all("<Button-5>")

        left.bind("<Configure>", _update_left_scroll_region)
        left_canvas.bind("<Configure>", _sync_left_width)
        left_canvas.bind("<Enter>", _bind_left_mousewheel)
        left_canvas.bind("<Leave>", _unbind_left_mousewheel)
        left.bind("<Enter>", _bind_left_mousewheel)
        left.bind("<Leave>", _unbind_left_mousewheel)

        right = ttk.Frame(main)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 图片区域
        img_group = ttk.LabelFrame(left, text="1. 图片")
        img_group.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(img_group, text="选择图片", command=self.choose_image).pack(fill=tk.X, padx=8, pady=6)
        ttk.Button(img_group, text="刷新预览", command=self.refresh_preview).pack(fill=tk.X, padx=8, pady=6)

        self.image_label_var = tk.StringVar(value="未选择图片")
        ttk.Label(img_group, textvariable=self.image_label_var, wraplength=300).pack(fill=tk.X, padx=8, pady=4)

        # 画布标定
        canvas_group = ttk.LabelFrame(left, text="2. 画布标定")
        canvas_group.pack(fill=tk.X, pady=(0, 10))

        self.canvas_x_var = tk.StringVar()
        self.canvas_y_var = tk.StringVar()
        self.canvas_w_var = tk.StringVar()
        self.canvas_h_var = tk.StringVar()
        self.padding_var = tk.StringVar()

        self._add_labeled_entry(canvas_group, "画布 X", self.canvas_x_var)
        self._add_labeled_entry(canvas_group, "画布 Y", self.canvas_y_var)
        self._add_labeled_entry(canvas_group, "画布宽 W", self.canvas_w_var)
        self._add_labeled_entry(canvas_group, "画布高 H", self.canvas_h_var)
        self._add_labeled_entry(canvas_group, "内边距", self.padding_var)

        ttk.Button(canvas_group, text="3秒后记录左上角", command=lambda: self.start_calibration("tl")).pack(fill=tk.X, padx=8, pady=4)
        ttk.Button(canvas_group, text="3秒后记录右下角", command=lambda: self.start_calibration("br")).pack(fill=tk.X, padx=8, pady=4)
        ttk.Button(canvas_group, text="测试画布：画一个框", command=self.start_test_rectangle).pack(fill=tk.X, padx=8, pady=4)

        # 参数
        param_group = ttk.LabelFrame(left, text="3. 线稿与绘制参数")
        param_group.pack(fill=tk.X, pady=(0, 10))
        self.param_group = param_group

        self.method_var = tk.StringVar()
        self.canny_low_var = tk.StringVar()
        self.canny_high_var = tk.StringVar()
        self.threshold_var = tk.StringVar()
        self.blur_var = tk.StringVar()
        self.centerline_bridge_px_var = tk.StringVar()
        self.dark_as_line_var = tk.BooleanVar()
        self.keep_aspect_var = tk.BooleanVar()
        self.epsilon_var = tk.StringVar()
        self.min_path_len_var = tk.StringVar()
        self.max_paths_var = tk.StringVar()
        self.point_step_var = tk.StringVar()
        self.raster_row_step_var = tk.StringVar()
        self.raster_min_run_var = tk.StringVar()
        self.raster_gap_tolerance_var = tk.StringVar()
        self.raster_extend_px_var = tk.StringVar()
        self.raster_serpentine_var = tk.BooleanVar()
        self.mouse_step_px_var = tk.StringVar()
        self.min_stroke_duration_var = tk.StringVar()
        self.stroke_duration_per_100px_var = tk.StringVar()
        self.pre_down_pause_var = tk.StringVar()
        self.pen_down_pause_var = tk.StringVar()
        self.pen_up_pause_var = tk.StringVar()
        self.between_strokes_pause_var = tk.StringVar()
        self.raster_backtrack_var = tk.BooleanVar()
        self.move_duration_var = tk.StringVar()
        self.start_delay_var = tk.StringVar()
        self.test_padding_var = tk.StringVar()
        self.minimize_var = tk.BooleanVar()

        ttk.Label(param_group, text="处理方式").pack(anchor="w", padx=8, pady=(6, 2))
        self.method_combo = ttk.Combobox(
            param_group,
            textvariable=self.method_var,
            state="readonly",
            values=("中心线追踪(线稿)", "逐行扫描(横向)", "边缘线稿(Canny)", "黑白轮廓(阈值)"),
        )
        self.method_combo.pack(fill=tk.X, padx=8, pady=2)
        self.method_combo.bind("<<ComboboxSelected>>", lambda _e: self._update_param_visibility())

        ttk.Button(param_group, text="恢复当前模式默认参数", command=self.reset_current_mode_defaults).pack(fill=tk.X, padx=8, pady=(6, 4))

        self.method_hint_var = tk.StringVar(value="")
        ttk.Label(param_group, textvariable=self.method_hint_var, foreground="#555555", wraplength=300).pack(anchor="w", padx=8, pady=(2, 6))

        self.param_sections: dict[str, ttk.LabelFrame] = {}

        self.section_image_common = self._create_param_section(param_group, "通用图片参数")
        self._add_labeled_entry(self.section_image_common, "模糊强度 0/1/3/5", self.blur_var)
        ttk.Checkbutton(self.section_image_common, text="保持图片比例并居中", variable=self.keep_aspect_var).pack(anchor="w", padx=8, pady=2)

        self.section_binary = self._create_param_section(param_group, "黑白线稿提取参数")
        self._add_labeled_entry(self.section_binary, "黑白阈值", self.threshold_var)
        ttk.Checkbutton(self.section_binary, text="深色区域作为线条", variable=self.dark_as_line_var).pack(anchor="w", padx=8, pady=2)

        self.section_centerline = self._create_param_section(param_group, "中心线追踪参数")
        self._add_labeled_entry(self.section_centerline, "断线连接像素", self.centerline_bridge_px_var)
        ttk.Label(self.section_centerline, text="建议 0~2。值越大越容易连接断线，但也可能把相邻线条粘在一起。", foreground="#666666", wraplength=300).pack(anchor="w", padx=8, pady=(2, 6))

        self.section_canny = self._create_param_section(param_group, "Canny 边缘参数")
        self._add_labeled_entry(self.section_canny, "Canny 低阈值", self.canny_low_var)
        self._add_labeled_entry(self.section_canny, "Canny 高阈值", self.canny_high_var)

        self.section_path = self._create_param_section(param_group, "折线路径参数")
        self._add_labeled_entry(self.section_path, "路径简化 epsilon", self.epsilon_var)
        self._add_labeled_entry(self.section_path, "最短路径长度", self.min_path_len_var)
        self._add_labeled_entry(self.section_path, "最多路径数", self.max_paths_var)
        self._add_labeled_entry(self.section_path, "点采样步长", self.point_step_var)

        self.section_raster = self._create_param_section(param_group, "逐行扫描参数")
        self._add_labeled_entry(self.section_raster, "扫描行距", self.raster_row_step_var)
        self._add_labeled_entry(self.section_raster, "最短横线长度", self.raster_min_run_var)
        self._add_labeled_entry(self.section_raster, "断点合并容差", self.raster_gap_tolerance_var)
        self._add_labeled_entry(self.section_raster, "横线左右延长像素", self.raster_extend_px_var)
        self._add_labeled_entry(self.section_raster, "最多横线数", self.max_paths_var)
        ttk.Checkbutton(self.section_raster, text="蛇形顺序减少空移", variable=self.raster_serpentine_var).pack(anchor="w", padx=8, pady=2)
        ttk.Checkbutton(self.section_raster, text="横线往返补笔（更黑更慢）", variable=self.raster_backtrack_var).pack(anchor="w", padx=8, pady=2)

        self.section_stability = self._create_param_section(param_group, "断笔优化参数")
        self._add_labeled_entry(self.section_stability, "鼠标插值步距px", self.mouse_step_px_var)
        self._add_labeled_entry(self.section_stability, "插值点等待秒", self.move_duration_var)
        self._add_labeled_entry(self.section_stability, "每笔最短秒数", self.min_stroke_duration_var)
        self._add_labeled_entry(self.section_stability, "每100px绘制秒数", self.stroke_duration_per_100px_var)
        self._add_labeled_entry(self.section_stability, "落笔前停顿秒", self.pre_down_pause_var)
        self._add_labeled_entry(self.section_stability, "按下后停顿秒", self.pen_down_pause_var)
        self._add_labeled_entry(self.section_stability, "抬笔前停顿秒", self.pen_up_pause_var)
        self._add_labeled_entry(self.section_stability, "笔画间隔秒", self.between_strokes_pause_var)

        self.section_misc = self._create_param_section(param_group, "运行参数")
        self._add_labeled_entry(self.section_misc, "开始前倒计时秒", self.start_delay_var)
        self._add_labeled_entry(self.section_misc, "测试框内边距", self.test_padding_var)
        ttk.Checkbutton(self.section_misc, text="绘制时最小化本窗口", variable=self.minimize_var).pack(anchor="w", padx=8, pady=2)

        # 操作按钮
        action_group = ttk.LabelFrame(left, text="4. 操作")
        action_group.pack(fill=tk.X)

        ttk.Button(action_group, text="保存配置", command=self.save_config).pack(fill=tk.X, padx=8, pady=5)
        ttk.Button(action_group, text="加载配置", command=lambda: self.load_config(silent=False)).pack(fill=tk.X, padx=8, pady=5)
        ttk.Button(action_group, text="开始绘制", command=self.start_drawing).pack(fill=tk.X, padx=8, pady=5)
        ttk.Button(action_group, text="停止绘制", command=self.stop_drawing).pack(fill=tk.X, padx=8, pady=5)

        # 右侧预览
        preview_group = ttk.LabelFrame(right, text="预览")
        preview_group.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        previews = ttk.Frame(preview_group)
        previews.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        self.original_preview_label = ttk.Label(previews, text="原图预览", anchor="center")
        self.original_preview_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))

        self.processed_preview_label = ttk.Label(previews, text="线稿预览", anchor="center")
        self.processed_preview_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0))

        # 日志
        log_group = ttk.LabelFrame(right, text="日志")
        log_group.pack(fill=tk.BOTH, expand=False)

        self.log_text = tk.Text(log_group, height=11, wrap="word")
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0), pady=8)

        log_scroll = ttk.Scrollbar(log_group, command=self.log_text.yview)
        log_scroll.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8), pady=8)
        self.log_text.configure(yscrollcommand=log_scroll.set)

    def _add_labeled_entry(self, parent, label: str, variable: tk.StringVar):
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, padx=8, pady=2)
        ttk.Label(row, text=label, width=14).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=variable, width=14).pack(side=tk.LEFT, fill=tk.X, expand=True)
        return row

    def _create_param_section(self, parent, title: str) -> ttk.LabelFrame:
        section = ttk.LabelFrame(parent, text=title)
        self.param_sections[title] = section
        return section

    def _show_param_sections(self, sections: list[ttk.LabelFrame]):
        if not hasattr(self, "param_sections"):
            return
        for section in self.param_sections.values():
            section.pack_forget()
        for section in sections:
            section.pack(fill=tk.X, padx=8, pady=(0, 8))

    def _update_param_visibility(self):
        """只显示当前处理方式真正会用到的参数，避免左侧功能区过长、误调无效参数。"""
        if not hasattr(self, "section_image_common"):
            return
        method = self.method_var.get() or "中心线追踪(线稿)"

        if method == "中心线追踪(线稿)":
            self.method_hint_var.set("适合黑色线稿：提取黑线→细化为单像素中心线→按主干折线绘制。不会描粗线外轮廓。")
            sections = [
                self.section_image_common,
                self.section_binary,
                self.section_centerline,
                self.section_path,
                self.section_stability,
                self.section_misc,
            ]
        elif method == "逐行扫描(横向)":
            self.method_hint_var.set("适合填充感和抗断笔：二值化后从上到下画水平短线，轨迹规则但会有横纹。")
            sections = [
                self.section_image_common,
                self.section_binary,
                self.section_raster,
                self.section_stability,
                self.section_misc,
            ]
        elif method == "边缘线稿(Canny)":
            self.method_hint_var.set("适合照片/复杂图：识别明暗边缘。黑色线稿通常不如中心线模式自然。")
            sections = [
                self.section_image_common,
                self.section_canny,
                self.section_path,
                self.section_stability,
                self.section_misc,
            ]
        else:
            self.method_hint_var.set("适合 Logo/黑白图：提取黑白区域外轮廓。粗线会被描成外边缘，不适合作为中心线。")
            sections = [
                self.section_image_common,
                self.section_binary,
                self.section_path,
                self.section_stability,
                self.section_misc,
            ]

        self._show_param_sections(sections)

    def reset_current_mode_defaults(self):
        """恢复当前处理方式的推荐默认值，不改变画布坐标和已选图片。"""
        method = self.method_var.get() or "中心线追踪(线稿)"
        defaults = {
            "中心线追踪(线稿)": {
                "threshold": 205,
                "blur": 0,
                "dark_as_line": True,
                "keep_aspect": True,
                "centerline_bridge_px": 1,
                "epsilon": 0.6,
                "min_path_len": 3.0,
                "max_paths": 50000,
                "point_step": 1,
                "mouse_step_px": 2,
                "move_duration": 0.005,
                "min_stroke_duration": 0.035,
                "stroke_duration_per_100px": 0.09,
                "pre_down_pause": 0.012,
                "pen_down_pause": 0.025,
                "pen_up_pause": 0.018,
                "between_strokes_pause": 0.006,
            },
            "逐行扫描(横向)": {
                "threshold": 150,
                "blur": 1,
                "dark_as_line": True,
                "keep_aspect": True,
                "raster_row_step": 2,
                "raster_min_run": 4,
                "raster_gap_tolerance": 2,
                "raster_extend_px": 2,
                "raster_serpentine": True,
                "raster_backtrack": False,
                "max_paths": 30000,
                "mouse_step_px": 3,
                "move_duration": 0.006,
                "min_stroke_duration": 0.035,
                "stroke_duration_per_100px": 0.08,
                "pre_down_pause": 0.010,
                "pen_down_pause": 0.020,
                "pen_up_pause": 0.015,
                "between_strokes_pause": 0.008,
            },
            "边缘线稿(Canny)": {
                "canny_low": 80,
                "canny_high": 160,
                "blur": 3,
                "keep_aspect": True,
                "epsilon": 1.2,
                "min_path_len": 10.0,
                "max_paths": 30000,
                "point_step": 1,
                "mouse_step_px": 3,
                "move_duration": 0.006,
                "min_stroke_duration": 0.035,
                "stroke_duration_per_100px": 0.08,
                "pre_down_pause": 0.010,
                "pen_down_pause": 0.020,
                "pen_up_pause": 0.015,
                "between_strokes_pause": 0.008,
            },
            "黑白轮廓(阈值)": {
                "threshold": 150,
                "blur": 1,
                "dark_as_line": True,
                "keep_aspect": True,
                "epsilon": 1.2,
                "min_path_len": 8.0,
                "max_paths": 30000,
                "point_step": 1,
                "mouse_step_px": 3,
                "move_duration": 0.006,
                "min_stroke_duration": 0.035,
                "stroke_duration_per_100px": 0.08,
                "pre_down_pause": 0.010,
                "pen_down_pause": 0.020,
                "pen_up_pause": 0.015,
                "between_strokes_pause": 0.008,
            },
        }.get(method, {})

        # 先把当前界面值同步到 config，保留画布坐标等非绘图参数；若界面里有非法数字，则仍尽力只重置参数。
        try:
            self._sync_vars_to_config()
        except Exception:
            pass

        for key, value in defaults.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        self.config.method = method
        self._sync_config_to_vars()
        self._update_param_visibility()
        self._log(f"已恢复“{method}”的推荐默认参数。")
        if self.original_image is not None:
            try:
                self.refresh_preview()
            except Exception:
                pass

    # ---------- 配置 ----------
    def _sync_config_to_vars(self):
        c = self.config
        self.canvas_x_var.set(str(c.canvas_x))
        self.canvas_y_var.set(str(c.canvas_y))
        self.canvas_w_var.set(str(c.canvas_w))
        self.canvas_h_var.set(str(c.canvas_h))
        self.padding_var.set(str(c.padding))

        self.method_var.set(c.method)
        self.canny_low_var.set(str(c.canny_low))
        self.canny_high_var.set(str(c.canny_high))
        self.threshold_var.set(str(c.threshold))
        self.blur_var.set(str(c.blur))
        self.centerline_bridge_px_var.set(str(c.centerline_bridge_px))
        self.dark_as_line_var.set(c.dark_as_line)
        self.keep_aspect_var.set(c.keep_aspect)
        self.epsilon_var.set(str(c.epsilon))
        self.min_path_len_var.set(str(c.min_path_len))
        self.max_paths_var.set(str(c.max_paths))
        self.point_step_var.set(str(c.point_step))
        self.raster_row_step_var.set(str(c.raster_row_step))
        self.raster_min_run_var.set(str(c.raster_min_run))
        self.raster_gap_tolerance_var.set(str(c.raster_gap_tolerance))
        self.raster_extend_px_var.set(str(c.raster_extend_px))
        self.raster_serpentine_var.set(c.raster_serpentine)
        self.mouse_step_px_var.set(str(c.mouse_step_px))
        self.min_stroke_duration_var.set(str(c.min_stroke_duration))
        self.stroke_duration_per_100px_var.set(str(c.stroke_duration_per_100px))
        self.pre_down_pause_var.set(str(c.pre_down_pause))
        self.pen_down_pause_var.set(str(c.pen_down_pause))
        self.pen_up_pause_var.set(str(c.pen_up_pause))
        self.between_strokes_pause_var.set(str(c.between_strokes_pause))
        self.raster_backtrack_var.set(c.raster_backtrack)
        self.move_duration_var.set(str(c.move_duration))
        self.start_delay_var.set(str(c.start_delay))
        self.test_padding_var.set(str(c.test_padding))
        self.minimize_var.set(c.minimize_when_drawing)

    def _sync_vars_to_config(self) -> DrawConfig:
        try:
            c = DrawConfig(
                canvas_x=int(float(self.canvas_x_var.get())),
                canvas_y=int(float(self.canvas_y_var.get())),
                canvas_w=max(20, int(float(self.canvas_w_var.get()))),
                canvas_h=max(20, int(float(self.canvas_h_var.get()))),
                padding=max(0, int(float(self.padding_var.get()))),

                method=self.method_var.get(),
                canny_low=int(float(self.canny_low_var.get())),
                canny_high=int(float(self.canny_high_var.get())),
                threshold=int(float(self.threshold_var.get())),
                blur=int(float(self.blur_var.get())),
                centerline_bridge_px=max(0, int(float(self.centerline_bridge_px_var.get()))),
                dark_as_line=bool(self.dark_as_line_var.get()),
                keep_aspect=bool(self.keep_aspect_var.get()),

                epsilon=max(0.0, float(self.epsilon_var.get())),
                min_path_len=max(0.0, float(self.min_path_len_var.get())),
                max_paths=max(1, int(float(self.max_paths_var.get()))),
                point_step=max(1, int(float(self.point_step_var.get()))),
                raster_row_step=max(1, int(float(self.raster_row_step_var.get()))),
                raster_min_run=max(1, int(float(self.raster_min_run_var.get()))),
                raster_gap_tolerance=max(0, int(float(self.raster_gap_tolerance_var.get()))),
                raster_extend_px=max(0, int(float(self.raster_extend_px_var.get()))),
                raster_serpentine=bool(self.raster_serpentine_var.get()),

                move_duration=max(0.0, float(self.move_duration_var.get())),
                mouse_step_px=max(1, int(float(self.mouse_step_px_var.get()))),
                min_stroke_duration=max(0.0, float(self.min_stroke_duration_var.get())),
                stroke_duration_per_100px=max(0.0, float(self.stroke_duration_per_100px_var.get())),
                pre_down_pause=max(0.0, float(self.pre_down_pause_var.get())),
                pen_down_pause=max(0.0, float(self.pen_down_pause_var.get())),
                pen_up_pause=max(0.0, float(self.pen_up_pause_var.get())),
                between_strokes_pause=max(0.0, float(self.between_strokes_pause_var.get())),
                raster_backtrack=bool(self.raster_backtrack_var.get()),

                start_delay=max(0, int(float(self.start_delay_var.get()))),
                test_padding=max(0, int(float(self.test_padding_var.get()))),
                minimize_when_drawing=bool(self.minimize_var.get()),
            )
        except ValueError as e:
            raise ValueError("参数里有非数字内容，请检查画布坐标和绘制参数。") from e

        if c.padding * 2 >= c.canvas_w or c.padding * 2 >= c.canvas_h:
            raise ValueError("内边距过大，已经超过画布尺寸。")
        if c.canny_low >= c.canny_high:
            raise ValueError("Canny 低阈值必须小于高阈值。")
        if c.blur not in (0, 1, 3, 5, 7, 9):
            raise ValueError("模糊强度建议填写 0、1、3、5、7 或 9。")
        if not (0 <= c.threshold <= 255):
            raise ValueError("黑白阈值应在 0 到 255 之间。")
        if c.centerline_bridge_px > 5:
            raise ValueError("断线连接像素建议在 0 到 5 之间，过大容易把相邻线条粘连。")

        self.config = c
        return c

    def save_config(self):
        try:
            c = self._sync_vars_to_config()
            payload = asdict(c)
            payload["image_path"] = self.image_path
            with open(DEFAULT_CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            self._log(f"配置已保存：{DEFAULT_CONFIG_PATH}")
        except Exception as e:
            messagebox.showerror("保存失败", str(e))

    def load_config(self, silent=False):
        if not DEFAULT_CONFIG_PATH.exists():
            if not silent:
                messagebox.showinfo("提示", "还没有配置文件。")
            return
        try:
            with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
                payload = json.load(f)

            known = {k: v for k, v in payload.items() if k in DrawConfig.__dataclass_fields__}
            self.config = DrawConfig(**known)
            self._sync_config_to_vars()

            image_path = payload.get("image_path")
            if image_path and os.path.exists(image_path):
                self.load_image(image_path)

            if not silent:
                self._log(f"配置已加载：{DEFAULT_CONFIG_PATH}")
        except Exception as e:
            if not silent:
                messagebox.showerror("加载失败", str(e))

    # ---------- 图片与预览 ----------
    def choose_image(self):
        path = filedialog.askopenfilename(
            title="选择要绘制的图片",
            filetypes=[
                ("图片文件", "*.png *.jpg *.jpeg *.bmp *.webp"),
                ("所有文件", "*.*"),
            ],
        )
        if not path:
            return
        self.load_image(path)
        self.refresh_preview()

    def load_image(self, path: str):
        try:
            img = Image.open(path)
            self.original_image = self._flatten_image(img)
            self.image_path = path
            self.image_label_var.set(os.path.basename(path))
            self._log(f"已选择图片：{path}")
        except Exception as e:
            messagebox.showerror("图片读取失败", str(e))

    @staticmethod
    def _flatten_image(img: Image.Image) -> Image.Image:
        """把透明图层铺到白底，统一转成 RGB。"""
        img = ImageOps.exif_transpose(img)
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            rgba = img.convert("RGBA")
            bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            bg.alpha_composite(rgba)
            return bg.convert("RGB")
        return img.convert("RGB")

    def refresh_preview(self):
        if self.original_image is None:
            messagebox.showinfo("提示", "请先选择图片。")
            return
        try:
            c = self._sync_vars_to_config()
            paths, debug_img = self.make_paths(c)
            self.last_paths = paths
            self.last_debug_image = debug_img

            self._show_original_preview()
            self._show_processed_preview(debug_img)
            if c.method == "逐行扫描(横向)":
                self._log(f"扫描路径已生成：{len(paths)} 条横线。断笔优先调慢：每100px绘制秒数/每笔最短秒数；太慢再增大扫描行距。")
            elif c.method == "中心线追踪(线稿)":
                self._log(f"中心线已生成：{len(paths)} 条主干路径。若断笔，优先降低 epsilon 到 1.0、调慢绘制速度；若路径太碎，提高最短路径长度。")
            else:
                self._log(f"线稿已生成：{len(paths)} 条路径。路径太多时可提高 epsilon、最短路径长度或降低最多路径数。")
        except Exception as e:
            messagebox.showerror("预览失败", str(e))

    def _show_original_preview(self):
        if self.original_image is None:
            return
        img = self.original_image.copy()
        img.thumbnail((390, 430), Image.Resampling.LANCZOS)
        self.preview_photo = ImageTk.PhotoImage(img)
        self.original_preview_label.configure(image=self.preview_photo, text="")

    def _show_processed_preview(self, debug_img: np.ndarray):
        # debug_img 是 0/255 灰度图：黑色线条、白色背景。
        pil = Image.fromarray(debug_img).convert("RGB")
        pil.thumbnail((390, 430), Image.Resampling.NEAREST)
        self.processed_photo = ImageTk.PhotoImage(pil)
        self.processed_preview_label.configure(image=self.processed_photo, text="")

    def _make_work_image(self, c: DrawConfig) -> Image.Image:
        if self.original_image is None:
            raise RuntimeError("请先选择图片。")

        inner_w = max(10, c.canvas_w - c.padding * 2)
        inner_h = max(10, c.canvas_h - c.padding * 2)

        src = self.original_image.copy()
        canvas = Image.new("RGB", (inner_w, inner_h), "white")

        if c.keep_aspect:
            fitted = ImageOps.contain(src, (inner_w, inner_h), Image.Resampling.LANCZOS)
            ox = (inner_w - fitted.width) // 2
            oy = (inner_h - fitted.height) // 2
            canvas.paste(fitted, (ox, oy))
        else:
            canvas = src.resize((inner_w, inner_h), Image.Resampling.LANCZOS)

        return canvas

    def make_paths(self, c: DrawConfig) -> tuple[list[np.ndarray], np.ndarray]:
        work_img = self._make_work_image(c)
        rgb = np.array(work_img)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

        if c.blur > 1:
            k = c.blur if c.blur % 2 == 1 else c.blur + 1
            gray = cv2.GaussianBlur(gray, (k, k), 0)

        if c.method == "中心线追踪(线稿)":
            return self._make_centerline_paths(gray, c)

        if c.method == "逐行扫描(横向)":
            return self._make_raster_paths(gray, c)

        if c.method == "黑白轮廓(阈值)":
            thresh_type = cv2.THRESH_BINARY_INV if c.dark_as_line else cv2.THRESH_BINARY
            _, binary = cv2.threshold(gray, c.threshold, 255, thresh_type)
            contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            debug = 255 - binary  # 预览中用黑线白底显示
        else:
            edges = cv2.Canny(gray, c.canny_low, c.canny_high)
            contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            debug = 255 - edges  # 黑线白底

        paths: list[np.ndarray] = []
        scored: list[tuple[float, np.ndarray]] = []

        for contour in contours:
            if contour is None or len(contour) < 2:
                continue

            arc = cv2.arcLength(contour, closed=False)
            if arc < c.min_path_len:
                continue

            approx = cv2.approxPolyDP(contour, c.epsilon, closed=False)
            pts = approx.reshape(-1, 2)

            if c.point_step > 1:
                pts = pts[::c.point_step]

            if len(pts) < 2:
                continue

            scored.append((arc, pts.astype(np.int32)))

        # 先画较长路径，减少碎线优先级。
        scored.sort(key=lambda item: item[0], reverse=True)
        for _, pts in scored[: c.max_paths]:
            paths.append(pts)

        return paths, debug

    def _make_centerline_paths(self, gray: np.ndarray, c: DrawConfig) -> tuple[list[np.ndarray], np.ndarray]:
        """中心线追踪：适合黑色线稿。

        和“黑白轮廓”不同，这里不是描黑色笔画的外边缘，而是：
        1. 按阈值把黑色线条提取成 mask；
        2. 用 Zhang-Suen thinning 把粗线细化为单像素骨架；
        3. 把骨架像素追踪成连续折线；
        4. 用 epsilon 做折线简化，减少鼠标轨迹复杂度。

        对黑色线稿来说，这通常比 Canny/轮廓更接近“照着线画一遍”。
        """
        if c.dark_as_line:
            mask = gray < c.threshold
        else:
            mask = gray > c.threshold

        # 中心线模式默认更偏“保细节”：不做开运算去噪，避免把细小线条删掉。
        # 只在用户设置“断线连接像素”时做轻微闭合，用来连接 1~2px 的小断点。
        mask_u8 = mask.astype(np.uint8) * 255
        if c.centerline_bridge_px > 0:
            k = int(c.centerline_bridge_px) * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)

        skeleton = self._zhang_suen_thinning(mask_u8 > 0)
        raw_paths = self._trace_skeleton_paths(skeleton)

        scored: list[tuple[float, np.ndarray]] = []
        for pts in raw_paths:
            if pts is None or len(pts) < 2:
                continue

            length = self._polyline_length(pts)
            if length < c.min_path_len:
                continue

            # 对中心线进行折线拟合：epsilon 越小越贴合，越大越省笔画。
            contour = pts.reshape(-1, 1, 2).astype(np.int32)
            approx = cv2.approxPolyDP(contour, c.epsilon, closed=False).reshape(-1, 2)

            if c.point_step > 1:
                approx = approx[::c.point_step]

            if len(approx) < 2:
                continue

            scored.append((length, approx.astype(np.int32)))

        scored.sort(key=lambda item: item[0], reverse=True)
        paths = [pts for _, pts in scored[: c.max_paths]]

        h, w = gray.shape[:2]
        debug = np.full((h, w), 255, dtype=np.uint8)
        for pts in paths:
            cv2.polylines(debug, [pts.reshape(-1, 1, 2)], isClosed=False, color=0, thickness=1)

        return paths, debug

    @staticmethod
    def _zhang_suen_thinning(binary: np.ndarray, max_iter: int = 120) -> np.ndarray:
        """Zhang-Suen 二值图细化算法，返回 0/1 骨架。"""
        img = binary.astype(np.uint8)
        if img.ndim != 2:
            raise ValueError("细化算法只接受二维二值图。")

        # 四周补白，避免边界判断复杂化。
        img = np.pad(img, ((1, 1), (1, 1)), mode="constant", constant_values=0)

        def neighbors_and_metrics(src: np.ndarray):
            p2 = src[:-2, 1:-1]
            p3 = src[:-2, 2:]
            p4 = src[1:-1, 2:]
            p5 = src[2:, 2:]
            p6 = src[2:, 1:-1]
            p7 = src[2:, :-2]
            p8 = src[1:-1, :-2]
            p9 = src[:-2, :-2]
            ps = [p2, p3, p4, p5, p6, p7, p8, p9]
            b = sum(ps)
            a = np.zeros_like(b, dtype=np.uint8)
            cyclic = ps + [p2]
            for cur, nxt in zip(cyclic[:-1], cyclic[1:]):
                a += ((cur == 0) & (nxt == 1)).astype(np.uint8)
            return p2, p3, p4, p5, p6, p7, p8, p9, a, b

        for _ in range(max_iter):
            changed = False

            p2, p3, p4, p5, p6, p7, p8, p9, a, b = neighbors_and_metrics(img)
            center = img[1:-1, 1:-1]
            marker = (
                (center == 1)
                & (b >= 2) & (b <= 6)
                & (a == 1)
                & ((p2 * p4 * p6) == 0)
                & ((p4 * p6 * p8) == 0)
            )
            if np.any(marker):
                center[marker] = 0
                changed = True

            p2, p3, p4, p5, p6, p7, p8, p9, a, b = neighbors_and_metrics(img)
            center = img[1:-1, 1:-1]
            marker = (
                (center == 1)
                & (b >= 2) & (b <= 6)
                & (a == 1)
                & ((p2 * p4 * p8) == 0)
                & ((p2 * p6 * p8) == 0)
            )
            if np.any(marker):
                center[marker] = 0
                changed = True

            if not changed:
                break

        return img[1:-1, 1:-1].astype(bool)

    @staticmethod
    def _trace_skeleton_paths(skeleton: np.ndarray) -> list[np.ndarray]:
        """把单像素骨架追踪成多条折线。

        骨架像素形成一个 8 邻接图：
        - 端点/分叉点作为路径端点；
        - 度数为 2 的像素作为路径中间点；
        - 闭环没有端点时，从任意未访问边开始绕一圈。
        """
        h, w = skeleton.shape[:2]
        ys, xs = np.nonzero(skeleton)
        pixels = set(zip(ys.tolist(), xs.tolist()))
        if not pixels:
            return []

        offsets = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]

        def nbs(p: tuple[int, int]) -> list[tuple[int, int]]:
            y, x = p
            out = []
            for dy, dx in offsets:
                q = (y + dy, x + dx)
                if 0 <= q[0] < h and 0 <= q[1] < w and q in pixels:
                    out.append(q)
            # 为了路径更稳定，邻居排序固定，减少每次运行的偶然性。
            out.sort()
            return out

        degree = {p: len(nbs(p)) for p in pixels}
        nodes = [p for p, d in degree.items() if d != 2]
        visited_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
        paths_yx: list[list[tuple[int, int]]] = []

        def edge_key(a: tuple[int, int], b: tuple[int, int]):
            return (a, b) if a <= b else (b, a)

        def trace_from(start: tuple[int, int], nxt: tuple[int, int]) -> list[tuple[int, int]]:
            path = [start]
            prev = start
            cur = nxt
            while True:
                ek = edge_key(prev, cur)
                if ek in visited_edges:
                    break
                visited_edges.add(ek)
                path.append(cur)

                # 到端点/分叉点就结束这条 path。
                if cur != start and degree.get(cur, 0) != 2:
                    break

                candidates = [q for q in nbs(cur) if q != prev and edge_key(cur, q) not in visited_edges]
                if not candidates:
                    break
                prev, cur = cur, candidates[0]
            return path

        # 先从端点/分叉点出发，能把大多数笔画拆成自然笔段。
        for p in nodes:
            for q in nbs(p):
                if edge_key(p, q) not in visited_edges:
                    path = trace_from(p, q)
                    if len(path) >= 2:
                        paths_yx.append(path)

        # 再处理闭环：闭环所有点 degree=2，没有端点。
        for p in pixels:
            for q in nbs(p):
                if edge_key(p, q) not in visited_edges:
                    path = trace_from(p, q)
                    if len(path) >= 2:
                        paths_yx.append(path)

        paths: list[np.ndarray] = []
        for path in paths_yx:
            # yx -> xy
            arr = np.array([[x, y] for y, x in path], dtype=np.int32)
            paths.append(arr)
        return paths

    @staticmethod
    def _polyline_length(pts: np.ndarray) -> float:
        if pts is None or len(pts) < 2:
            return 0.0
        diffs = np.diff(pts.astype(np.float32), axis=0)
        return float(np.sqrt((diffs * diffs).sum(axis=1)).sum())

    def _make_raster_paths(self, gray: np.ndarray, c: DrawConfig) -> tuple[list[np.ndarray], np.ndarray]:
        """逐行扫描：把二值图中需要绘制的区域转成水平短线。

        这个模式不追求拟合轮廓，而是像扫描仪一样从上到下处理图像。
        每一行里连续的“线条像素”会被压缩成一段鼠标横拖路径。
        由于鼠标轨迹只有水平线，通常比复杂轮廓更稳定，也更不容易被画布丢笔。
        """
        if c.dark_as_line:
            mask = gray < c.threshold
        else:
            mask = gray > c.threshold

        h, w = mask.shape[:2]
        debug = np.full((h, w), 255, dtype=np.uint8)
        paths: list[np.ndarray] = []

        for y in range(0, h, c.raster_row_step):
            if self.stop_event.is_set():
                break

            row = mask[y]
            runs = self._row_runs(row, c.raster_min_run, c.raster_gap_tolerance)
            if not runs:
                continue

            # 先做左右延长，提升覆盖率，减少视觉断笔；同时不要越过图片边界。
            expanded_runs: list[tuple[int, int]] = []
            for x1, x2 in runs:
                x1e = max(0, int(x1) - c.raster_extend_px)
                x2e = min(w - 1, int(x2) + c.raster_extend_px)
                if x2e >= x1e:
                    expanded_runs.append((x1e, x2e))
            runs = expanded_runs

            # 蛇形顺序只改变“这一行的绘制顺序”，不会在空白处按下鼠标。
            if c.raster_serpentine and ((y // c.raster_row_step) % 2 == 1):
                runs = list(reversed(runs))
                run_points = [
                    np.array([[x2, y], [x1, y]], dtype=np.int32)
                    for x1, x2 in runs
                ]
            else:
                run_points = [
                    np.array([[x1, y], [x2, y]], dtype=np.int32)
                    for x1, x2 in runs
                ]

            for pts in run_points:
                if len(paths) >= c.max_paths:
                    return paths, debug
                x_start, x_end = int(min(pts[0][0], pts[1][0])), int(max(pts[0][0], pts[1][0]))
                debug[y, x_start : x_end + 1] = 0
                paths.append(pts)

        return paths, debug

    @staticmethod
    def _row_runs(row: np.ndarray, min_run: int, gap_tolerance: int) -> list[tuple[int, int]]:
        """返回一行中 True 区域的连续区间，并可合并很短的断点。"""
        xs = np.flatnonzero(row)
        if xs.size == 0:
            return []

        raw_runs: list[tuple[int, int]] = []
        start = int(xs[0])
        prev = int(xs[0])
        for x_raw in xs[1:]:
            x = int(x_raw)
            if x == prev + 1:
                prev = x
            else:
                raw_runs.append((start, prev))
                start = prev = x
        raw_runs.append((start, prev))

        if gap_tolerance > 0 and len(raw_runs) > 1:
            merged: list[tuple[int, int]] = []
            cur_start, cur_end = raw_runs[0]
            for run_start, run_end in raw_runs[1:]:
                gap = run_start - cur_end - 1
                if gap <= gap_tolerance:
                    cur_end = run_end
                else:
                    merged.append((cur_start, cur_end))
                    cur_start, cur_end = run_start, run_end
            merged.append((cur_start, cur_end))
            raw_runs = merged

        return [(a, b) for a, b in raw_runs if (b - a + 1) >= min_run]

    # ---------- 标定 ----------
    def start_calibration(self, which: str):
        def countdown(n: int):
            if n > 0:
                target = "左上角" if which == "tl" else "右下角"
                self._log(f"请把鼠标移动到画布{target}，{n} 秒后读取坐标。")
                self.root.after(1000, lambda: countdown(n - 1))
            else:
                pos = pyautogui.position()
                x, y = int(pos.x), int(pos.y)
                if which == "tl":
                    self.calib_top_left = (x, y)
                    self.canvas_x_var.set(str(x))
                    self.canvas_y_var.set(str(y))
                    self._log(f"已记录画布左上角：({x}, {y})")
                else:
                    if self.calib_top_left is not None:
                        x1, y1 = self.calib_top_left
                    else:
                        try:
                            x1 = int(float(self.canvas_x_var.get()))
                            y1 = int(float(self.canvas_y_var.get()))
                        except Exception:
                            messagebox.showerror("标定失败", "请先记录左上角，或手动填写画布 X/Y。")
                            return
                    left = min(x1, x)
                    top = min(y1, y)
                    right = max(x1, x)
                    bottom = max(y1, y)
                    self.canvas_x_var.set(str(left))
                    self.canvas_y_var.set(str(top))
                    self.canvas_w_var.set(str(right - left))
                    self.canvas_h_var.set(str(bottom - top))
                    self._log(f"已记录画布右下角：({x}, {y})，画布=({left}, {top}, {right - left}, {bottom - top})")
                    self.save_config()

        countdown(3)

    # ---------- 绘制 ----------
    def start_test_rectangle(self):
        if self._is_worker_running():
            messagebox.showinfo("提示", "当前正在绘制，请先停止或等待完成。")
            return
        try:
            c = self._sync_vars_to_config()
        except Exception as e:
            messagebox.showerror("参数错误", str(e))
            return

        self.stop_event.clear()
        self.worker_thread = threading.Thread(target=self._draw_test_rectangle_worker, args=(c,), daemon=True)
        self.worker_thread.start()

    def _draw_test_rectangle_worker(self, c: DrawConfig):
        try:
            self._countdown_in_worker(c.start_delay)
            if self.stop_event.is_set():
                self._log("测试框已取消。")
                return

            p = c.test_padding
            x1 = c.canvas_x + p
            y1 = c.canvas_y + p
            x2 = c.canvas_x + c.canvas_w - p
            y2 = c.canvas_y + c.canvas_h - p

            self._log("开始绘制测试框。")
            self._draw_screen_polyline(c, [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)], is_raster_line=False)
            self._log("测试框完成。若框偏离画布，请重新标定。")
        except pyautogui.FailSafeException:
            self._safe_mouse_up()
            self._log("已触发 PyAutoGUI FailSafe：鼠标到达屏幕左上角，绘制停止。")
        except Exception as e:
            self._safe_mouse_up()
            self._log(f"测试框失败：{e}")

    def start_drawing(self):
        if self._is_worker_running():
            messagebox.showinfo("提示", "当前正在绘制，请先停止或等待完成。")
            return
        if self.original_image is None:
            messagebox.showinfo("提示", "请先选择图片。")
            return

        try:
            c = self._sync_vars_to_config()
            paths, debug_img = self.make_paths(c)
            if not paths:
                messagebox.showwarning("没有可绘制路径", "当前参数没有生成路径，请换图或调整阈值/Canny/epsilon/扫描行距。")
                return
            self.last_paths = paths
            self.last_debug_image = debug_img
            self._show_processed_preview(debug_img)
        except Exception as e:
            messagebox.showerror("准备失败", str(e))
            return

        if not messagebox.askyesno(
            "确认开始",
            f"即将绘制 {len(paths)} 条路径/横线。\n\n请确认：\n1. 微信小程序已经停在杯贴画布；\n2. 画布位置已标定；\n3. 鼠标移到屏幕左上角可紧急停止。\n\n是否开始？",
        ):
            return

        self.stop_event.clear()
        if c.minimize_when_drawing:
            self.root.iconify()

        self.worker_thread = threading.Thread(target=self._draw_paths_worker, args=(c, paths), daemon=True)
        self.worker_thread.start()

    def _draw_paths_worker(self, c: DrawConfig, paths: list[np.ndarray]):
        drawn = 0
        try:
            self._countdown_in_worker(c.start_delay)
            if self.stop_event.is_set():
                self._log("绘制已取消。")
                return

            self._log(f"开始绘制，共 {len(paths)} 条路径/横线。")
            for idx, pts in enumerate(paths, start=1):
                if self.stop_event.is_set():
                    self._log("收到停止指令，正在结束。")
                    break

                # 把处理图坐标映射到屏幕坐标。处理图尺寸就是画布扣除 padding 后的尺寸。
                screen_points = [self._map_point_to_screen(c, p) for p in pts]
                is_raster_line = (c.method == "逐行扫描(横向)" and len(screen_points) == 2)
                self._draw_screen_polyline(c, screen_points, is_raster_line=is_raster_line)

                drawn += 1
                if idx % 50 == 0:
                    self._log(f"进度：{idx}/{len(paths)} 条路径/横线。")

            self._log(f"绘制结束：已绘制 {drawn}/{len(paths)} 条路径/横线。")
        except pyautogui.FailSafeException:
            self._safe_mouse_up()
            self._log("已触发 PyAutoGUI FailSafe：鼠标到达屏幕左上角，绘制停止。")
        except Exception as e:
            self._safe_mouse_up()
            self._log(f"绘制失败：{e}")
        finally:
            if c.minimize_when_drawing:
                self.root.after(0, self.root.deiconify)

    def stop_drawing(self):
        self.stop_event.set()
        self._safe_mouse_up()
        self._log("已请求停止。若鼠标仍在移动，请把鼠标移到屏幕左上角触发紧急停止。")

    def _draw_screen_polyline(self, c: DrawConfig, points: list[tuple[int, int]], is_raster_line: bool = False):
        """按下鼠标绘制一条屏幕坐标折线。

        优化点：
        1. 落笔前/按下后/抬笔前加入短暂停顿，避免小程序只收到点按而没收到拖动。
        2. 每个线段按距离切成多个小鼠标 move 事件，不再依赖 pyautogui 的极短 duration。
        3. 逐行扫描横线可选往返补笔：同一笔里先正向再反向，增强连续性和黑度。
        """
        if len(points) < 2:
            return

        start_x, start_y = points[0]
        self._safe_move_to(start_x, start_y, duration=0)
        self._sleep_interruptible(c.pre_down_pause)
        if self.stop_event.is_set():
            return

        pyautogui.mouseDown(button="left")
        try:
            self._sleep_interruptible(c.pen_down_pause)
            if self.stop_event.is_set():
                return

            for x2, y2 in points[1:]:
                if self.stop_event.is_set():
                    break
                cur = pyautogui.position()
                self._draw_smooth_segment(c, int(cur.x), int(cur.y), int(x2), int(y2), is_raster_line=is_raster_line)

            # 横线往返补笔：仅对逐行扫描横线生效。同一笔不抬笔返回，通常比重复落笔更稳。
            if is_raster_line and c.raster_backtrack and not self.stop_event.is_set():
                x_back, y_back = points[0]
                cur = pyautogui.position()
                self._draw_smooth_segment(c, int(cur.x), int(cur.y), int(x_back), int(y_back), is_raster_line=True)

            self._sleep_interruptible(c.pen_up_pause)
        finally:
            pyautogui.mouseUp(button="left")

        self._sleep_interruptible(c.between_strokes_pause)

    def _draw_smooth_segment(self, c: DrawConfig, x1: int, y1: int, x2: int, y2: int, is_raster_line: bool = False):
        """把一个线段拆成很多小 move，确保微信小程序 canvas 收到足够多的鼠标移动事件。"""
        dx = x2 - x1
        dy = y2 - y1
        dist = float((dx * dx + dy * dy) ** 0.5)
        if dist <= 0:
            return

        # 每隔 mouse_step_px 像素发一个 move 事件；步距越小越细，但越慢。
        steps = max(1, int(np.ceil(dist / max(1, c.mouse_step_px))))

        # 长线按距离增加持续时间；短横线也保证最短持续时间，避免被当成一次无效点击。
        duration_by_distance = dist / 100.0 * c.stroke_duration_per_100px
        if is_raster_line:
            total_duration = max(c.min_stroke_duration, duration_by_distance)
        else:
            total_duration = max(c.move_duration, duration_by_distance)

        # 继续保留 move_duration 作为每个插值点的最小等待，避免事件过密被吞。
        per_step_sleep = max(c.move_duration, total_duration / steps if steps else total_duration)

        for i in range(1, steps + 1):
            if self.stop_event.is_set():
                break
            t = i / steps
            x = int(round(x1 + dx * t))
            y = int(round(y1 + dy * t))
            pyautogui.moveTo(x, y, duration=0)
            self._sleep_interruptible(per_step_sleep)

    def _sleep_interruptible(self, seconds: float):
        seconds = max(0.0, float(seconds or 0.0))
        if seconds <= 0:
            return
        end = time.perf_counter() + seconds
        while not self.stop_event.is_set():
            remaining = end - time.perf_counter()
            if remaining <= 0:
                return
            time.sleep(min(0.02, remaining))

    def _map_point_to_screen(self, c: DrawConfig, p: np.ndarray | tuple[int, int]) -> tuple[int, int]:
        # p 已经在 inner_w/inner_h 坐标内，所以只要加画布起点和 padding。
        px = int(p[0])
        py = int(p[1])
        return c.canvas_x + c.padding + px, c.canvas_y + c.padding + py

    def _countdown_in_worker(self, seconds: int):
        for n in range(seconds, 0, -1):
            if self.stop_event.is_set():
                return
            self._log(f"{n} 秒后开始，请切到微信杯贴画布。")
            time.sleep(1)

    @staticmethod
    def _safe_move_to(x: int, y: int, duration: float = 0.0):
        """
        安全移动鼠标。

        原因：PyAutoGUI 在 Windows 上如果把 MINIMUM_DURATION 设为 0，
        并传入 0.001、0.002 这类很小的 duration，部分版本内部会把
        tween 步数算成 0，从而触发 float division by zero。

        处理方式：
        - duration <= 0：立即移动。
        - duration < 0.05：立即移动后手动 sleep，避免 PyAutoGUI 内部插值除零。
        - duration >= 0.05：交给 PyAutoGUI 做平滑移动。
        """
        duration = max(0.0, float(duration or 0.0))
        if duration <= 0:
            pyautogui.moveTo(int(x), int(y), duration=0)
        elif duration < 0.05:
            pyautogui.moveTo(int(x), int(y), duration=0)
            time.sleep(duration)
        else:
            pyautogui.moveTo(int(x), int(y), duration=duration)

    @staticmethod
    def _safe_mouse_up():
        try:
            pyautogui.mouseUp(button="left")
        except Exception:
            pass

    def _is_worker_running(self) -> bool:
        return self.worker_thread is not None and self.worker_thread.is_alive()

    # ---------- 日志 ----------
    def _log(self, msg: str):
        timestamp = time.strftime("%H:%M:%S")
        line = f"[{timestamp}] {msg}\n"

        def append():
            self.log_text.insert(tk.END, line)
            self.log_text.see(tk.END)

        try:
            self.root.after(0, append)
        except RuntimeError:
            pass


def main():
    root = tk.Tk()
    try:
        # Windows 默认主题更像原生应用。
        style = ttk.Style(root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
    except Exception:
        pass
    app = HeyTeaCupLabelDrawerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

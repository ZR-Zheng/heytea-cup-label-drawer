"""Tkinter GUI 与用户交互调度。"""

import json
import os
import shutil
import threading
import time
from dataclasses import asdict

import numpy as np
import pyautogui
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox, ttk

from .automation import countdown_in_worker, draw_screen_polyline, map_point_to_screen, safe_mouse_up
from .config import (
    ANILINES_MODELS_DIR,
    ANIME2SKETCH_MODELS_DIR,
    DEFAULT_CONFIG_PATH,
    INFORMATIVE_DRAWINGS_MODELS_DIR,
    MODELS_DIR,
    DrawConfig,
)
from .processing import flatten_image, make_paths


PARAMETER_HELP = {
    "左上角 X": "画布左上角相对于屏幕左侧的像素坐标。建议使用“记录左上角”自动填写。",
    "左上角 Y": "画布左上角相对于屏幕顶部的像素坐标。建议使用“记录左上角”自动填写。",
    "宽度 W": "可绘制画布的宽度，单位为屏幕像素。记录右下角时会自动计算。",
    "高度 H": "可绘制画布的高度，单位为屏幕像素。记录右下角时会自动计算。",
    "绘图内边距": "在画布四周保留的空白像素。增大可避免笔画贴近边缘或超出画布。",
    "模糊强度": "预处理时使用的模糊核大小。可填 0、1、3、5、7 或 9；数值越大，细节越少、线条越平滑。",
    "黑白阈值": "区分线条与背景的灰度界线，范围 0~255。值越大，更多较浅区域会被识别为深色线条。",
    "断线连接像素": "尝试连接相距较近的断点。建议 1~2；过大可能把相邻线条错误连接。",
    "模型权重文件": "当前线稿处理方式使用的模型文件。导入后会按模型类型保存到 models 下的独立文件夹。",
    "模型输入尺寸": "送入线稿模型的正方形边长。推荐 512；越大细节越多，但处理更慢、占用内存更多。",
    "运行设备": "模型推理设备。auto 自动选择，cpu 使用处理器，cuda 优先使用 NVIDIA 显卡。",
    "低阈值": "Canny 边缘检测的低阈值。降低会保留更多弱边缘，也可能引入噪点。",
    "高阈值": "Canny 边缘检测的高阈值，必须高于低阈值。提高会只保留更明显的边缘。",
    "简化 epsilon": "折线路径简化强度。值越大，路径点越少、绘制更快，但曲线细节也会减少。",
    "最短路径长度": "短于该长度的碎线将被忽略。提高可减少杂点和无意义短笔画。",
    "最多路径数": "限制最终生成的路径数量，防止复杂图片产生过多笔画和过长绘制时间。",
    "点采样步长": "沿路径每隔多少个点取一个绘制点。提高可加快绘制，但线条可能不够平滑。",
    "扫描行距": "逐行扫描时相邻横线的垂直间距。值越小填充越密、耗时越长。",
    "最短横线长度": "短于该长度的扫描横线会被忽略，用于减少零碎小笔画。",
    "断点合并容差": "同一扫描行内，间隔不超过该像素数的线段会被合并。",
    "左右延长像素": "将每条扫描横线向左右延长的像素数，可减少填充边缘出现空隙。",
    "最多横线数": "限制逐行扫描生成的横线数量，防止绘制耗时过长。",
    "鼠标插值步距 px": "鼠标移动时相邻插值点的最大像素距离。值越小越平滑稳定，但绘制更慢。",
    "插值点等待秒": "鼠标移动到每个插值点后的最短等待时间。增大可提高稳定性，但会减慢绘制。",
    "每笔最短秒数": "每一笔至少持续的时间。适当增大可降低短笔画未被画布识别的概率。",
    "每 100px 秒数": "每绘制 100 屏幕像素使用的时间。增大可减少长线断笔，但整体更慢。",
    "落笔前停顿秒": "移动到笔画起点后、按下鼠标前等待的时间。",
    "按下后停顿秒": "按下鼠标后、开始移动前等待的时间。增大可减少起笔丢失。",
    "落笔预热像素": "按下鼠标后在起点附近往返移动的距离，用于帮助画布识别落笔。",
    "抬笔前停顿秒": "到达笔画终点后、释放鼠标前等待的时间。增大可减少末端缺失。",
    "笔画间隔秒": "完成一笔后到开始下一笔之间的等待时间。",
    "开始前倒计时秒": "点击开始绘制或测试框后预留的切换窗口时间。",
    "测试框内边距": "测试框与已标定画布边缘之间保留的像素距离。",
    "保持图片比例并居中": "缩放图片时保持原始宽高比，并将内容放在画布中央，避免图像被拉伸。",
    "深色区域作为线条": "启用后将比阈值更暗的区域识别为线条；关闭后则识别较亮区域。",
    "反向补笔（防空笔，更慢）": "中心线路径完成后沿原路反向再画一次，可降低空笔概率，但绘制时间接近翻倍。",
    "蛇形顺序减少空移": "让相邻扫描行交替从左向右和从右向左绘制，减少鼠标在行间空移。",
    "横线往返补笔（更黑更慢）": "每条扫描横线完成后沿原路返回，可让填充更深，但绘制时间更长。",
    "绘制时最小化本窗口": "开始正式绘制时自动最小化工具窗口，结束后恢复，避免遮挡目标画布。",
}

MODEL_METHODS = {
    "动漫线稿(Anime2Sketch)": {
        "name": "Anime2Sketch",
        "dir": ANIME2SKETCH_MODELS_DIR,
        "field": "anime2sketch_model_path",
        "extensions": (".pth", ".bin"),
        "pattern": "*.pth *.bin",
    },
    "动漫精细线稿(AniLines)": {
        "name": "AniLines Detail",
        "dir": ANILINES_MODELS_DIR,
        "field": "anilines_model_path",
        "extensions": (".pth",),
        "pattern": "*.pth",
    },
    "通用语义线稿(Informative Drawings)": {
        "name": "Informative Drawings",
        "dir": INFORMATIVE_DRAWINGS_MODELS_DIR,
        "field": "informative_drawings_model_path",
        "extensions": (".onnx",),
        "pattern": "*.onnx",
    },
}


class HoverTip:
    """在鼠标悬浮于控件时显示简短说明。"""

    def __init__(self, widget: tk.Widget, text: str, delay_ms: int = 450):
        self.widget = widget
        self.text = text
        self.delay_ms = delay_ms
        self.after_id: str | None = None
        self.tip_window: tk.Toplevel | None = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self.hide, add="+")
        widget.bind("<ButtonPress>", self.hide, add="+")
        widget.bind("<Destroy>", self.hide, add="+")

    def _schedule(self, _event=None):
        self.hide()
        self.after_id = self.widget.after(self.delay_ms, self.show)

    def show(self):
        self.after_id = None
        if self.tip_window is not None or not self.widget.winfo_exists():
            return

        tip = tk.Toplevel(self.widget)
        tip.wm_overrideredirect(True)
        tip.wm_attributes("-topmost", True)
        tk.Label(
            tip,
            text=self.text,
            justify=tk.LEFT,
            wraplength=320,
            bg="#172033",
            fg="#f8fafc",
            relief=tk.SOLID,
            borderwidth=1,
            font=("Microsoft YaHei UI", 9),
            padx=10,
            pady=7,
        ).pack()
        tip.update_idletasks()
        x = self.widget.winfo_rootx() + 12
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        x = min(x, self.widget.winfo_screenwidth() - tip.winfo_reqwidth() - 8)
        y = min(y, self.widget.winfo_screenheight() - tip.winfo_reqheight() - 8)
        tip.wm_geometry(f"+{max(8, x)}+{max(8, y)}")
        self.tip_window = tip

    def hide(self, _event=None):
        if self.after_id is not None:
            self.widget.after_cancel(self.after_id)
            self.after_id = None
        if self.tip_window is not None:
            self.tip_window.destroy()
            self.tip_window = None


# Windows 高 DPI 下让屏幕坐标更接近真实像素，避免 125%/150% 缩放导致偏移。
try:
    import ctypes
    ctypes.windll.user32.SetProcessDPIAware()
except Exception:
    pass


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
        self.available_anime_models: dict[str, str] = {}

        self.stop_event = threading.Event()
        self.worker_thread: threading.Thread | None = None
        self.calib_top_left: tuple[int, int] | None = None
        self.parameter_tips: list[HoverTip] = []
        for spec in MODEL_METHODS.values():
            spec["dir"].mkdir(parents=True, exist_ok=True)

        self._build_ui()
        self.load_config(silent=True)
        self._sync_config_to_vars()
        self.refresh_anime2sketch_models(silent=True)
        self.method_var.trace_add("write", lambda *_: self._update_param_visibility())
        self._update_param_visibility()
        self._log("准备就绪。先选择图片并标定画布，再刷新预览。")
        self._log("紧急停止：把鼠标移到屏幕左上角，或点击顶部“停止”。")

    # ---------- UI ----------
    def _build_ui(self):
        self._init_ui_vars()
        self._configure_style()
        self.root.configure(bg="#f3f5f8")

        app = ttk.Frame(self.root, style="App.TFrame", padding=16)
        app.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(app, style="Header.TFrame", padding=(18, 14))
        header.pack(fill=tk.X, pady=(0, 12))

        title_block = ttk.Frame(header, style="Header.TFrame")
        title_block.pack(side=tk.LEFT, fill=tk.Y)
        ttk.Label(title_block, text="喜茶杯贴自动手绘", style="HeaderTitle.TLabel").pack(anchor="w")
        ttk.Label(title_block, text="选择图片  ·  标定画布  ·  调整线稿  ·  开始绘制", style="HeaderSubtle.TLabel").pack(anchor="w", pady=(3, 0))

        header_actions = ttk.Frame(header, style="Header.TFrame")
        header_actions.pack(side=tk.RIGHT)
        self._add_buttons(
            header_actions,
            [
                ("选择图片", self.choose_image, "Toolbar.TButton"),
                ("刷新预览", self.refresh_preview, "Toolbar.TButton"),
                ("开始绘制", self.start_drawing, "Primary.TButton"),
                ("停止", self.stop_drawing, "Danger.TButton"),
            ],
            horizontal=True,
        )

        body = ttk.Frame(app, style="App.TFrame")
        body.pack(fill=tk.BOTH, expand=True)

        shell = ttk.Frame(body, style="Card.TFrame")
        shell.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 12))

        nav = ttk.Frame(shell, style="Nav.TFrame", padding=(10, 12))
        nav.pack(side=tk.LEFT, fill=tk.Y)
        settings_host = ttk.Frame(shell, style="Card.TFrame", padding=12, width=390)
        settings_host.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        settings_host.pack_propagate(False)

        right = ttk.Frame(body, style="App.TFrame")
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.settings_pages: dict[str, ttk.Frame] = {}
        self.nav_buttons: dict[str, ttk.Button] = {}
        self.param_sections: dict[str, ttk.LabelFrame] = {}

        prepare_tab = self._create_scroll_page(settings_host, "准备")
        lineart_tab = self._create_scroll_page(settings_host, "线稿")
        draw_tab = self._create_scroll_page(settings_host, "绘制")
        advanced_tab = self._create_scroll_page(settings_host, "高级")
        for key in ("准备", "线稿", "绘制", "高级"):
            self._add_nav_button(nav, key)
        self._show_settings_page("准备")

        image_group = ttk.LabelFrame(prepare_tab, text="当前图片")
        image_group.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(image_group, textvariable=self.image_label_var, wraplength=340, style="Value.TLabel").pack(fill=tk.X, padx=10, pady=10)

        canvas_group = ttk.LabelFrame(prepare_tab, text="画布区域")
        canvas_group.pack(fill=tk.X, pady=(0, 10))
        self._add_entry_specs(
            canvas_group,
            [
                ("左上角 X", self.canvas_x_var),
                ("左上角 Y", self.canvas_y_var),
                ("宽度 W", self.canvas_w_var),
                ("高度 H", self.canvas_h_var),
                ("绘图内边距", self.padding_var),
            ],
        )

        calibrate_group = ttk.LabelFrame(prepare_tab, text="快速标定")
        calibrate_group.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(
            calibrate_group,
            text="依次记录画布左上角和右下角，随后用测试框确认位置。",
            style="Hint.TLabel",
            wraplength=340,
        ).pack(anchor="w", padx=10, pady=(8, 4))
        self._add_buttons(
            calibrate_group,
            [
                ("记录左上角", lambda: self.start_calibration("tl"), None),
                ("记录右下角", lambda: self.start_calibration("br"), None),
                ("绘制测试框", self.start_test_rectangle, "Accent.TButton"),
            ],
        )

        self.section_image_common = self._create_param_section(prepare_tab, "图片适配")
        self.section_image_common.pack(fill=tk.X, pady=(0, 10))
        self._add_labeled_entry(self.section_image_common, "模糊强度", self.blur_var)
        self._add_checkbutton(self.section_image_common, "保持图片比例并居中", self.keep_aspect_var, pady=(3, 9))

        method_group = ttk.LabelFrame(lineart_tab, text="处理方式")
        method_group.pack(fill=tk.X, pady=(0, 10))
        self.method_combo = ttk.Combobox(
            method_group,
            textvariable=self.method_var,
            state="readonly",
            values=(
                "中心线追踪(线稿)",
                "动漫精细线稿(AniLines)",
                "动漫线稿(Anime2Sketch)",
                "通用语义线稿(Informative Drawings)",
                "逐行扫描(横向)",
                "边缘线稿(Canny)",
                "黑白轮廓(阈值)",
            ),
        )
        self.method_combo.pack(fill=tk.X, padx=10, pady=(9, 5))
        self.method_combo.bind("<<ComboboxSelected>>", lambda _e: self._update_param_visibility())
        ttk.Button(method_group, text="恢复当前模式推荐参数", command=self.reset_current_mode_defaults).pack(fill=tk.X, padx=10, pady=(0, 5))
        self.method_hint_var = tk.StringVar(value="")
        ttk.Label(method_group, textvariable=self.method_hint_var, style="Hint.TLabel", wraplength=340).pack(anchor="w", padx=10, pady=(2, 9))

        self.dynamic_param_frame = ttk.Frame(lineart_tab)
        self.dynamic_param_frame.pack(fill=tk.X)

        self.section_binary = self._create_param_section(self.dynamic_param_frame, "黑白提取")
        self._add_labeled_entry(self.section_binary, "黑白阈值", self.threshold_var)
        self._add_checkbutton(self.section_binary, "深色区域作为线条", self.dark_as_line_var, pady=(3, 9))

        self.section_centerline = self._create_param_section(self.dynamic_param_frame, "中心线追踪")
        self._add_labeled_entry(self.section_centerline, "断线连接像素", self.centerline_bridge_px_var)
        ttk.Label(self.section_centerline, text="建议 1~2；数值越大越容易连接相邻线条。", style="Hint.TLabel", wraplength=340).pack(anchor="w", padx=10, pady=(2, 5))
        self._add_checkbutton(self.section_centerline, "反向补笔（防空笔，更慢）", self.centerline_retrace_var, pady=(2, 9))

        self.section_anime2sketch = self._create_param_section(self.dynamic_param_frame, "线稿模型")
        self._add_model_path_entry(self.section_anime2sketch)
        self._add_entry_specs(
            self.section_anime2sketch,
            [
                ("模型输入尺寸", self.anime2sketch_input_size_var),
                ("运行设备", self.anime2sketch_device_var),
            ],
        )
        ttk.Label(self.section_anime2sketch, text="模型会按类型独立管理；设备可填写 auto、cpu 或 cuda。", style="Hint.TLabel", wraplength=340).pack(anchor="w", padx=10, pady=(2, 9))
        ttk.Button(self.section_anime2sketch, text="打开模型文件夹", command=self.open_models_folder).pack(fill=tk.X, padx=10, pady=(0, 9))

        self.section_canny = self._create_param_section(self.dynamic_param_frame, "Canny 边缘")
        self._add_entry_specs(
            self.section_canny,
            [
                ("低阈值", self.canny_low_var),
                ("高阈值", self.canny_high_var),
            ],
        )

        self.section_path = self._create_param_section(self.dynamic_param_frame, "路径优化")
        self._add_entry_specs(
            self.section_path,
            [
                ("简化 epsilon", self.epsilon_var),
                ("最短路径长度", self.min_path_len_var),
                ("最多路径数", self.max_paths_var),
                ("点采样步长", self.point_step_var),
            ],
        )

        self.section_raster = self._create_param_section(self.dynamic_param_frame, "逐行扫描")
        self._add_entry_specs(
            self.section_raster,
            [
                ("扫描行距", self.raster_row_step_var),
                ("最短横线长度", self.raster_min_run_var),
                ("断点合并容差", self.raster_gap_tolerance_var),
                ("左右延长像素", self.raster_extend_px_var),
                ("最多横线数", self.max_paths_var),
            ],
        )
        self._add_checkbutton(self.section_raster, "蛇形顺序减少空移", self.raster_serpentine_var, pady=2)
        self._add_checkbutton(self.section_raster, "横线往返补笔（更黑更慢）", self.raster_backtrack_var, pady=(2, 9))

        self.section_stability = self._create_param_section(draw_tab, "绘制稳定性")
        self.section_stability.pack(fill=tk.X, pady=(0, 10))
        self._add_entry_specs(
            self.section_stability,
            [
                ("鼠标插值步距 px", self.mouse_step_px_var),
                ("插值点等待秒", self.move_duration_var),
                ("每笔最短秒数", self.min_stroke_duration_var),
                ("每 100px 秒数", self.stroke_duration_per_100px_var),
                ("落笔前停顿秒", self.pre_down_pause_var),
                ("按下后停顿秒", self.pen_down_pause_var),
                ("落笔预热像素", self.pen_down_nudge_px_var),
                ("抬笔前停顿秒", self.pen_up_pause_var),
                ("笔画间隔秒", self.between_strokes_pause_var),
            ],
        )

        self.section_misc = self._create_param_section(advanced_tab, "运行设置")
        self.section_misc.pack(fill=tk.X, pady=(0, 10))
        self._add_entry_specs(
            self.section_misc,
            [
                ("开始前倒计时秒", self.start_delay_var),
                ("测试框内边距", self.test_padding_var),
            ],
        )
        self._add_checkbutton(self.section_misc, "绘制时最小化本窗口", self.minimize_var, pady=(3, 9))

        config_group = ttk.LabelFrame(advanced_tab, text="配置文件")
        config_group.pack(fill=tk.X, pady=(0, 10))
        self._add_buttons(
            config_group,
            [
                ("保存当前配置", self.save_config, "Accent.TButton"),
                ("重新加载配置", lambda: self.load_config(silent=False), None),
            ],
        )

        preview_group = ttk.Frame(right, style="Card.TFrame", padding=12)
        preview_group.pack(fill=tk.BOTH, expand=True, pady=(0, 12))

        preview_header = ttk.Frame(preview_group, style="Card.TFrame")
        preview_header.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(preview_header, text="预览", style="SectionTitle.TLabel").pack(side=tk.LEFT)
        ttk.Label(preview_header, textvariable=self.path_count_var, style="Status.TLabel").pack(side=tk.RIGHT)

        previews = ttk.PanedWindow(preview_group, orient=tk.HORIZONTAL)
        previews.pack(fill=tk.BOTH, expand=True)
        self.original_preview_label = self._create_preview_card(previews, "原图", "选择图片后显示")
        self.processed_preview_label = self._create_preview_card(previews, "线稿", "刷新预览后显示")

        log_group = ttk.Frame(right, style="Card.TFrame", padding=12)
        log_group.pack(fill=tk.BOTH, expand=False)
        log_header = ttk.Frame(log_group, style="Card.TFrame")
        log_header.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(log_header, text="运行记录", style="SectionTitle.TLabel").pack(side=tk.LEFT)
        ttk.Button(log_header, text="清空", style="Link.TButton", command=self._clear_log).pack(side=tk.RIGHT)

        log_body = ttk.Frame(log_group, style="Card.TFrame")
        log_body.pack(fill=tk.BOTH, expand=True)
        self.log_text = tk.Text(log_body, height=7, wrap="word", bg="#111827", fg="#e5e7eb", insertbackground="#e5e7eb", relief=tk.FLAT, padx=10, pady=8)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        log_scroll = ttk.Scrollbar(log_body, command=self.log_text.yview)
        log_scroll.pack(side=tk.LEFT, fill=tk.Y)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        self.refresh_anime2sketch_models(silent=True)

    def _init_ui_vars(self):
        self.canvas_x_var = tk.StringVar()
        self.canvas_y_var = tk.StringVar()
        self.canvas_w_var = tk.StringVar()
        self.canvas_h_var = tk.StringVar()
        self.padding_var = tk.StringVar()
        self.method_var = tk.StringVar()
        self.canny_low_var = tk.StringVar()
        self.canny_high_var = tk.StringVar()
        self.threshold_var = tk.StringVar()
        self.blur_var = tk.StringVar()
        self.centerline_bridge_px_var = tk.StringVar()
        self.anime2sketch_model_path_var = tk.StringVar()
        self.anime2sketch_model_choice_var = tk.StringVar()
        self.anime2sketch_input_size_var = tk.StringVar()
        self.anime2sketch_device_var = tk.StringVar()
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
        self.pen_down_nudge_px_var = tk.StringVar()
        self.pen_up_pause_var = tk.StringVar()
        self.between_strokes_pause_var = tk.StringVar()
        self.centerline_retrace_var = tk.BooleanVar()
        self.raster_backtrack_var = tk.BooleanVar()
        self.move_duration_var = tk.StringVar()
        self.start_delay_var = tk.StringVar()
        self.test_padding_var = tk.StringVar()
        self.minimize_var = tk.BooleanVar()
        self.image_label_var = tk.StringVar(value="未选择图片")
        self.path_count_var = tk.StringVar(value="未生成路径")

    def _configure_style(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("App.TFrame", background="#f3f5f8")
        style.configure("Header.TFrame", background="#172033")
        style.configure("HeaderTitle.TLabel", background="#172033", foreground="#ffffff", font=("Microsoft YaHei UI", 15, "bold"))
        style.configure("HeaderSubtle.TLabel", background="#172033", foreground="#b6c2d9", font=("Microsoft YaHei UI", 9))
        style.configure("Card.TFrame", background="#ffffff", relief=tk.FLAT)
        style.configure("Nav.TFrame", background="#f7f8fa")
        style.configure("Preview.TFrame", background="#f7f8fa", relief=tk.FLAT)
        style.configure("Preview.TLabel", background="#f7f8fa", foreground="#7c8799", font=("Microsoft YaHei UI", 10))
        style.configure("SectionTitle.TLabel", background="#ffffff", foreground="#111827", font=("Microsoft YaHei UI", 11, "bold"))
        style.configure("Subtle.TLabel", background="#f7f8fa", foreground="#64748b", font=("Microsoft YaHei UI", 9, "bold"))
        style.configure("Hint.TLabel", background="#ffffff", foreground="#64748b", font=("Microsoft YaHei UI", 9))
        style.configure("Value.TLabel", background="#ffffff", foreground="#1d4ed8", font=("Microsoft YaHei UI", 9, "bold"))
        style.configure("Status.TLabel", background="#ffffff", foreground="#64748b", font=("Microsoft YaHei UI", 9))
        style.configure("Parameter.TLabel", background="#ffffff", foreground="#334155", font=("Microsoft YaHei UI", 9))
        style.configure("TLabel", background="#ffffff", foreground="#1f2937", font=("Microsoft YaHei UI", 9))
        style.configure("TLabelframe", background="#ffffff", bordercolor="#d8dee9")
        style.configure("TLabelframe.Label", background="#ffffff", foreground="#334155", font=("Microsoft YaHei UI", 9, "bold"))
        style.configure("TCheckbutton", background="#ffffff", foreground="#1f2937", font=("Microsoft YaHei UI", 9))
        style.configure("TEntry", padding=(5, 4))
        style.configure("TCombobox", padding=(5, 4))
        style.configure("TButton", padding=(10, 6), font=("Microsoft YaHei UI", 9))
        style.configure("Toolbar.TButton", padding=(12, 7), font=("Microsoft YaHei UI", 9))
        style.configure("Primary.TButton", padding=(14, 7), font=("Microsoft YaHei UI", 9, "bold"), foreground="#ffffff", background="#2563eb")
        style.map("Primary.TButton", background=[("active", "#1d4ed8"), ("pressed", "#1e40af")])
        style.configure("Danger.TButton", padding=(12, 7), font=("Microsoft YaHei UI", 9, "bold"), foreground="#ffffff", background="#dc2626")
        style.map("Danger.TButton", background=[("active", "#b91c1c"), ("pressed", "#991b1b")])
        style.configure("Accent.TButton", foreground="#1d4ed8", background="#eff6ff")
        style.map("Accent.TButton", background=[("active", "#dbeafe"), ("pressed", "#bfdbfe")])
        style.configure("Link.TButton", padding=(8, 3), foreground="#475569", background="#ffffff")
        style.map("Link.TButton", background=[("active", "#f1f5f9")])
        style.configure("Nav.TButton", padding=(14, 10), anchor="w", font=("Microsoft YaHei UI", 10), background="#f7f8fa")
        style.configure("NavActive.TButton", padding=(14, 9), anchor="w", font=("Microsoft YaHei UI", 10, "bold"), foreground="#1d4ed8", background="#dbeafe")

    def _add_buttons(self, parent, specs, horizontal=False):
        for index, (text, command, style) in enumerate(specs):
            button = ttk.Button(parent, text=text, command=command, style=style or "TButton")
            if horizontal:
                button.pack(side=tk.LEFT, padx=(0 if index == 0 else 6, 0))
            else:
                button.pack(fill=tk.X, padx=10, pady=(8 if index == 0 else 0, 5))

    def _add_entry_specs(self, parent, specs):
        for label, variable in specs:
            self._add_labeled_entry(parent, label, variable)

    def _add_checkbutton(self, parent, text: str, variable: tk.BooleanVar, pady=2):
        button = ttk.Checkbutton(parent, text=text, variable=variable, cursor="question_arrow")
        button.pack(anchor="w", padx=10, pady=pady)
        self._add_parameter_tip(button, text)
        return button

    def _create_preview_card(self, parent, title: str, placeholder: str) -> ttk.Label:
        card = ttk.Frame(parent, style="Preview.TFrame", padding=10)
        parent.add(card, weight=1)
        ttk.Label(card, text=title, style="Subtle.TLabel").pack(anchor="w", pady=(0, 8))
        label = ttk.Label(card, text=placeholder, anchor="center", style="Preview.TLabel")
        label.pack(fill=tk.BOTH, expand=True)
        return label

    def _create_scroll_page(self, parent: ttk.Frame, title: str) -> ttk.Frame:
        outer = ttk.Frame(parent, style="Card.TFrame")
        self.settings_pages[title] = outer

        canvas = tk.Canvas(outer, highlightthickness=0, bg="#ffffff", bd=0)
        scrollbar = ttk.Scrollbar(outer, orient=tk.VERTICAL, command=canvas.yview)
        content = ttk.Frame(canvas)
        window_id = canvas.create_window((0, 0), window=content, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.LEFT, fill=tk.Y)

        def _sync_scroll_region(_event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _sync_width(event):
            canvas.itemconfigure(window_id, width=event.width)

        def _on_mousewheel(event):
            if getattr(event, "num", None) == 4:
                canvas.yview_scroll(-1, "units")
            elif getattr(event, "num", None) == 5:
                canvas.yview_scroll(1, "units")
            else:
                delta = int(-1 * (event.delta / 120)) if event.delta else 0
                canvas.yview_scroll(delta, "units")

        def _bind_wheel(_event=None):
            canvas.bind_all("<MouseWheel>", _on_mousewheel)
            canvas.bind_all("<Button-4>", _on_mousewheel)
            canvas.bind_all("<Button-5>", _on_mousewheel)

        def _unbind_wheel(_event=None):
            canvas.unbind_all("<MouseWheel>")
            canvas.unbind_all("<Button-4>")
            canvas.unbind_all("<Button-5>")

        content.bind("<Configure>", _sync_scroll_region)
        canvas.bind("<Configure>", _sync_width)
        outer.bind("<Enter>", _bind_wheel)
        outer.bind("<Leave>", _unbind_wheel)
        canvas.bind("<Enter>", _bind_wheel)
        content.bind("<Enter>", _bind_wheel)
        return content

    def _add_nav_button(self, parent, title: str):
        btn = ttk.Button(parent, text=title, style="Nav.TButton", command=lambda t=title: self._show_settings_page(t))
        btn.pack(fill=tk.X, pady=(0, 6))
        self.nav_buttons[title] = btn
        return btn

    def _show_settings_page(self, title: str):
        for name, page in self.settings_pages.items():
            page.pack_forget()
            if name in self.nav_buttons:
                self.nav_buttons[name].configure(style="Nav.TButton")
        page = self.settings_pages.get(title)
        if page is not None:
            page.pack(fill=tk.BOTH, expand=True)
        if title in self.nav_buttons:
            self.nav_buttons[title].configure(style="NavActive.TButton")

    def _add_labeled_entry(self, parent, label: str, variable: tk.StringVar):
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, padx=10, pady=3)
        label_widget = ttk.Label(row, text=label, width=15, style="Parameter.TLabel", cursor="question_arrow")
        label_widget.pack(side=tk.LEFT)
        self._add_parameter_tip(label_widget, label)
        ttk.Entry(row, textvariable=variable, width=12).pack(side=tk.LEFT, fill=tk.X, expand=True)
        return row

    def _add_model_path_entry(self, parent):
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, padx=10, pady=3)
        label_widget = ttk.Label(row, text="模型权重文件", width=15, style="Parameter.TLabel", cursor="question_arrow")
        label_widget.pack(side=tk.LEFT)
        self._add_parameter_tip(label_widget, "模型权重文件")
        self.anime2sketch_model_combo = ttk.Combobox(row, textvariable=self.anime2sketch_model_choice_var, state="readonly", width=12)
        self.anime2sketch_model_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.anime2sketch_model_combo.bind("<<ComboboxSelected>>", lambda _e: self._select_anime2sketch_model())
        ttk.Button(row, text="刷新", command=self.refresh_anime2sketch_models).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Button(row, text="导入", command=self.choose_anime2sketch_model).pack(side=tk.LEFT, padx=(4, 0))
        return row

    def _add_parameter_tip(self, widget: tk.Widget, label: str):
        help_text = PARAMETER_HELP.get(label)
        if help_text:
            self.parameter_tips.append(HoverTip(widget, help_text))

    def _create_param_section(self, parent, title: str) -> ttk.LabelFrame:
        section = ttk.LabelFrame(parent, text=title)
        self.param_sections[title] = section
        return section

    def _show_param_sections(self, sections: list[ttk.LabelFrame]):
        if not hasattr(self, "section_binary"):
            return
        method_sections = [
            self.section_binary,
            self.section_centerline,
            self.section_anime2sketch,
            self.section_canny,
            self.section_path,
            self.section_raster,
        ]
        for section in method_sections:
            section.pack_forget()
        for section in sections:
            section.pack(fill=tk.X, pady=(0, 10))

    def _update_param_visibility(self):
        """只显示当前处理方式真正会用到的参数，避免左侧功能区过长、误调无效参数。"""
        if not hasattr(self, "section_image_common"):
            return
        method = self.method_var.get() or "中心线追踪(线稿)"
        if method in MODEL_METHODS:
            self._sync_model_controls_for_method(method)

        if method == "中心线追踪(线稿)":
            self.method_hint_var.set("适合黑色线稿：提取黑线→细化骨架→方向桥接断点→分叉处按角度延续主干。不会描粗线外轮廓。")
            sections = [
                self.section_binary,
                self.section_centerline,
                self.section_path,
            ]
        elif method == "动漫线稿(Anime2Sketch)":
            self.method_hint_var.set("适合彩色动漫/插画：先用 Anime2Sketch 抽取干净线稿，再转成中心线主干绘制。")
            sections = [
                self.section_anime2sketch,
                self.section_binary,
                self.section_centerline,
                self.section_path,
            ]
        elif method == "动漫精细线稿(AniLines)":
            self.method_hint_var.set("推荐用于动漫和平涂插画：比 Anime2Sketch 保留更多角色、背景和色块边缘细节。")
            sections = [
                self.section_anime2sketch,
                self.section_binary,
                self.section_centerline,
                self.section_path,
            ]
        elif method == "通用语义线稿(Informative Drawings)":
            self.method_hint_var.set("推荐用于照片和通用场景：保留物体轮廓、几何与语义结构，CPU 也可运行。")
            sections = [
                self.section_anime2sketch,
                self.section_binary,
                self.section_centerline,
                self.section_path,
            ]
        elif method == "逐行扫描(横向)":
            self.method_hint_var.set("适合填充感和抗断笔：二值化后从上到下画水平短线，轨迹规则但会有横纹。")
            sections = [
                self.section_binary,
                self.section_raster,
            ]
        elif method == "边缘线稿(Canny)":
            self.method_hint_var.set("适合照片/复杂图：识别明暗边缘。黑色线稿通常不如中心线模式自然。")
            sections = [
                self.section_canny,
                self.section_path,
            ]
        else:
            self.method_hint_var.set("适合 Logo/黑白图：提取黑白区域外轮廓。粗线会被描成外边缘，不适合作为中心线。")
            sections = [
                self.section_binary,
                self.section_path,
            ]

        self._show_param_sections(sections)

    def reset_current_mode_defaults(self):
        """恢复当前处理方式的推荐默认值，不改变画布坐标和已选图片。"""
        method = self.method_var.get() or "中心线追踪(线稿)"
        defaults = {
            "中心线追踪(线稿)": {
                "threshold": 210,
                "blur": 0,
                "dark_as_line": True,
                "keep_aspect": True,
                "centerline_bridge_px": 2,
                "epsilon": 0.7,
                "min_path_len": 6.0,
                "max_paths": 50000,
                "point_step": 1,
                "mouse_step_px": 2,
                "move_duration": 0.006,
                "min_stroke_duration": 0.035,
                "stroke_duration_per_100px": 0.10,
                "pre_down_pause": 0.012,
                "pen_down_pause": 0.040,
                "pen_down_nudge_px": 2,
                "pen_up_pause": 0.025,
                "between_strokes_pause": 0.012,
                "centerline_retrace": True,
            },
            "动漫线稿(Anime2Sketch)": {
                "threshold": 210,
                "blur": 0,
                "dark_as_line": True,
                "keep_aspect": True,
                "anime2sketch_input_size": 512,
                "anime2sketch_device": "auto",
                "centerline_bridge_px": 1,
                "epsilon": 0.8,
                "min_path_len": 6.0,
                "max_paths": 50000,
                "point_step": 1,
                "mouse_step_px": 2,
                "move_duration": 0.006,
                "min_stroke_duration": 0.035,
                "stroke_duration_per_100px": 0.10,
                "pre_down_pause": 0.012,
                "pen_down_pause": 0.040,
                "pen_down_nudge_px": 2,
                "pen_up_pause": 0.025,
                "between_strokes_pause": 0.012,
                "centerline_retrace": True,
            },
            "动漫精细线稿(AniLines)": {
                "threshold": 210,
                "blur": 0,
                "dark_as_line": True,
                "keep_aspect": True,
                "anime2sketch_input_size": 512,
                "anime2sketch_device": "auto",
                "centerline_bridge_px": 1,
                "epsilon": 0.8,
                "min_path_len": 6.0,
                "max_paths": 50000,
                "point_step": 1,
                "centerline_retrace": True,
            },
            "通用语义线稿(Informative Drawings)": {
                "threshold": 215,
                "blur": 0,
                "dark_as_line": True,
                "keep_aspect": True,
                "anime2sketch_input_size": 512,
                "anime2sketch_device": "auto",
                "centerline_bridge_px": 1,
                "epsilon": 1.0,
                "min_path_len": 8.0,
                "max_paths": 40000,
                "point_step": 1,
                "centerline_retrace": False,
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
                "pen_down_pause": 0.035,
                "pen_down_nudge_px": 2,
                "pen_up_pause": 0.020,
                "between_strokes_pause": 0.012,
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
                "pen_down_pause": 0.035,
                "pen_down_nudge_px": 2,
                "pen_up_pause": 0.020,
                "between_strokes_pause": 0.012,
                "centerline_retrace": False,
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
                "pen_down_pause": 0.035,
                "pen_down_nudge_px": 2,
                "pen_up_pause": 0.020,
                "between_strokes_pause": 0.012,
                "centerline_retrace": False,
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
        active_model_path = self._model_path_for_method(c.method)
        self.anime2sketch_model_path_var.set(active_model_path)
        self._set_anime2sketch_model_choice(active_model_path)
        self.anime2sketch_input_size_var.set(str(c.anime2sketch_input_size))
        self.anime2sketch_device_var.set(c.anime2sketch_device)
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
        self.pen_down_nudge_px_var.set(str(c.pen_down_nudge_px))
        self.pen_up_pause_var.set(str(c.pen_up_pause))
        self.between_strokes_pause_var.set(str(c.between_strokes_pause))
        self.centerline_retrace_var.set(c.centerline_retrace)
        self.raster_backtrack_var.set(c.raster_backtrack)
        self.move_duration_var.set(str(c.move_duration))
        self.start_delay_var.set(str(c.start_delay))
        self.test_padding_var.set(str(c.test_padding))
        self.minimize_var.set(c.minimize_when_drawing)

    def _sync_vars_to_config(self) -> DrawConfig:
        active_model_field = self._model_field_for_method(self.method_var.get())
        model_paths = {
            "anime2sketch_model_path": self.config.anime2sketch_model_path,
            "anilines_model_path": self.config.anilines_model_path,
            "informative_drawings_model_path": self.config.informative_drawings_model_path,
        }
        if active_model_field:
            model_paths[active_model_field] = self.anime2sketch_model_path_var.get().strip()
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
                anime2sketch_model_path=model_paths["anime2sketch_model_path"],
                anilines_model_path=model_paths["anilines_model_path"],
                informative_drawings_model_path=model_paths["informative_drawings_model_path"],
                anime2sketch_input_size=max(32, int(float(self.anime2sketch_input_size_var.get()))),
                anime2sketch_device=self.anime2sketch_device_var.get().strip() or "auto",
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
                pen_down_nudge_px=max(0, int(float(self.pen_down_nudge_px_var.get()))),
                pen_up_pause=max(0.0, float(self.pen_up_pause_var.get())),
                between_strokes_pause=max(0.0, float(self.between_strokes_pause_var.get())),
                centerline_retrace=bool(self.centerline_retrace_var.get()),
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
        if c.method in MODEL_METHODS:
            spec = MODEL_METHODS[c.method]
            model_path = getattr(c, spec["field"])
            if not model_path:
                raise ValueError(f"请先导入并选择 {spec['name']} 模型文件。")
            if not os.path.exists(model_path):
                raise ValueError(f"{spec['name']} 模型文件不存在：{model_path}")
            if c.anime2sketch_input_size < 128:
                raise ValueError("模型输入尺寸建议至少 128，推荐 512。")
            if c.anime2sketch_device.lower() not in ("auto", "cpu", "cuda"):
                raise ValueError("模型运行设备只能填写 auto、cpu 或 cuda。")
        if c.pen_down_nudge_px > 10:
            raise ValueError("落笔预热像素建议在 0 到 10 之间。")

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
            for method, spec in MODEL_METHODS.items():
                field = spec["field"]
                if field in known:
                    known[field] = self._resolve_model_path(known[field], method)
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
    def refresh_anime2sketch_models(self, silent=False):
        if not hasattr(self, "anime2sketch_model_combo"):
            return

        method = self.method_var.get()
        spec = MODEL_METHODS.get(method)
        if spec is None:
            return
        models = self._discover_anime2sketch_models()
        current_path = self.anime2sketch_model_path_var.get().strip()
        if current_path and os.path.exists(current_path):
            label = self._model_choice_label(current_path)
            models.setdefault(label, current_path)

        self.available_anime_models = models
        labels = list(models.keys())
        self.anime2sketch_model_combo.configure(values=labels)
        if current_path:
            self._set_anime2sketch_model_choice(current_path)
        elif labels:
            self.anime2sketch_model_choice_var.set(labels[0])
            self.anime2sketch_model_path_var.set(models[labels[0]])
        else:
            self.anime2sketch_model_choice_var.set("")

        if not silent:
            self._log(f"已刷新 {spec['name']} 模型列表：{len(labels)} 个。")

    def _discover_anime2sketch_models(self) -> dict[str, str]:
        models: dict[str, str] = {}
        spec = MODEL_METHODS.get(self.method_var.get())
        if spec is None:
            return models
        model_dir = spec["dir"]
        model_dir.mkdir(parents=True, exist_ok=True)
        for cur_dir, _dir_names, file_names in os.walk(model_dir):
            for file_name in file_names:
                if os.path.splitext(file_name)[1].lower() not in spec["extensions"]:
                    continue
                path = os.path.join(cur_dir, file_name)
                models[self._model_choice_label(path)] = path

        return dict(sorted(models.items(), key=lambda item: item[0].lower()))

    def _model_choice_label(self, path: str) -> str:
        p = os.path.abspath(path)
        spec = MODEL_METHODS.get(self.method_var.get())
        model_dir = spec["dir"] if spec else MODELS_DIR
        try:
            rel = os.path.relpath(p, model_dir)
            if not rel.startswith(".."):
                return rel
        except ValueError:
            pass
        return p

    def _set_anime2sketch_model_choice(self, path: str):
        if not hasattr(self, "anime2sketch_model_choice_var"):
            return
        if not path:
            self.anime2sketch_model_choice_var.set("")
            return
        label = self._model_choice_label(path)
        if label not in self.available_anime_models and os.path.exists(path):
            self.available_anime_models[label] = path
            if hasattr(self, "anime2sketch_model_combo"):
                self.anime2sketch_model_combo.configure(values=list(self.available_anime_models.keys()))
        self.anime2sketch_model_choice_var.set(label)

    def _select_anime2sketch_model(self):
        label = self.anime2sketch_model_choice_var.get()
        path = self.available_anime_models.get(label)
        if path:
            self.anime2sketch_model_path_var.set(path)
            method = self.method_var.get()
            field = self._model_field_for_method(method)
            if field:
                setattr(self.config, field, path)
            self._log(f"已选择 {MODEL_METHODS[method]['name']} 模型：{path}")

    def choose_anime2sketch_model(self):
        method = self.method_var.get()
        spec = MODEL_METHODS.get(method)
        if spec is None:
            return
        path = filedialog.askopenfilename(
            title=f"导入 {spec['name']} 模型",
            initialdir=str(spec["dir"]),
            filetypes=[
                (f"{spec['name']} 模型", spec["pattern"]),
                ("所有文件", "*.*"),
            ],
        )
        if not path:
            return
        try:
            imported_path = self._import_model_file(path)
        except Exception as e:
            messagebox.showerror("模型导入失败", str(e))
            return
        self.anime2sketch_model_path_var.set(imported_path)
        self.refresh_anime2sketch_models(silent=True)
        self._set_anime2sketch_model_choice(imported_path)
        setattr(self.config, spec["field"], imported_path)
        self._log(f"模型已导入：{imported_path}")

    def open_models_folder(self):
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(MODELS_DIR)
        except AttributeError:
            messagebox.showinfo("模型文件夹", str(MODELS_DIR))

    def _import_model_file(self, path: str) -> str:
        spec = MODEL_METHODS.get(self.method_var.get())
        if spec is None:
            raise ValueError("当前处理方式不使用模型。")
        source = os.path.abspath(path)
        if os.path.splitext(source)[1].lower() not in spec["extensions"]:
            raise ValueError(f"{spec['name']} 仅支持：{', '.join(spec['extensions'])}")

        spec["dir"].mkdir(parents=True, exist_ok=True)
        destination = os.path.abspath(spec["dir"] / os.path.basename(source))
        if os.path.normcase(source) == os.path.normcase(destination):
            return destination
        if os.path.exists(destination) and not messagebox.askyesno("覆盖模型", f"模型文件已存在：\n{destination}\n\n是否覆盖？"):
            return destination
        shutil.copy2(source, destination)
        return destination

    def _resolve_model_path(self, path: str, method: str) -> str:
        if not path:
            return ""
        expanded = os.path.abspath(os.path.expanduser(path))
        if os.path.exists(expanded):
            return expanded

        migrated = MODEL_METHODS[method]["dir"] / os.path.basename(path)
        if migrated.exists():
            return str(migrated.resolve())
        return path

    def _model_field_for_method(self, method: str) -> str | None:
        spec = MODEL_METHODS.get(method)
        return spec["field"] if spec else None

    def _model_path_for_method(self, method: str) -> str:
        field = self._model_field_for_method(method)
        return getattr(self.config, field, "") if field else ""

    def _sync_model_controls_for_method(self, method: str):
        path = self._model_path_for_method(method)
        self.anime2sketch_model_path_var.set(path)
        self.refresh_anime2sketch_models(silent=True)

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
            self.original_image = flatten_image(img)
            self.image_path = path
            self.image_label_var.set(os.path.basename(path))
            self._log(f"已选择图片：{path}")
        except Exception as e:
            messagebox.showerror("图片读取失败", str(e))

    def refresh_preview(self):
        if self.original_image is None:
            messagebox.showinfo("提示", "请先选择图片。")
            return
        try:
            c = self._sync_vars_to_config()
            paths, debug_img = make_paths(self.original_image, c, should_stop=self.stop_event.is_set)
            self.last_paths = paths
            self.last_debug_image = debug_img
            self.path_count_var.set(f"已生成 {len(paths)} 条路径/横线")

            self._show_original_preview()
            self._show_processed_preview(debug_img)
            if c.method == "逐行扫描(横向)":
                self._log(f"扫描路径已生成：{len(paths)} 条横线。断笔优先调慢：每100px绘制秒数/每笔最短秒数；太慢再增大扫描行距。")
            elif c.method in MODEL_METHODS:
                self._log(f"{MODEL_METHODS[c.method]['name']} 线稿已生成：{len(paths)} 条主干路径。细节过多时可提高最短路径长度或降低最多路径数。")
            elif c.method == "中心线追踪(线稿)":
                self._log(f"中心线已生成：{len(paths)} 条主干路径。若仍断笔，先把断线连接像素调到 2~3；若粘连，降到 1 或降低阈值。")
            else:
                self._log(f"线稿已生成：{len(paths)} 条路径。路径太多时可提高 epsilon、最短路径长度或降低最多路径数。")
        except Exception as e:
            messagebox.showerror("预览失败", str(e))

    def _show_original_preview(self):
        if self.original_image is None:
            return
        img = self.original_image.copy()
        img.thumbnail(self._preview_bounds(self.original_preview_label), Image.Resampling.LANCZOS)
        self.preview_photo = ImageTk.PhotoImage(img)
        self.original_preview_label.configure(image=self.preview_photo, text="")

    def _show_processed_preview(self, debug_img: np.ndarray):
        # debug_img 是 0/255 灰度图：黑色线条、白色背景。
        pil = Image.fromarray(debug_img).convert("RGB")
        pil.thumbnail(self._preview_bounds(self.processed_preview_label), Image.Resampling.NEAREST)
        self.processed_photo = ImageTk.PhotoImage(pil)
        self.processed_preview_label.configure(image=self.processed_photo, text="")

    def _preview_bounds(self, label: ttk.Label) -> tuple[int, int]:
        return (
            max(160, label.winfo_width() - 20),
            max(160, label.winfo_height() - 20),
        )

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
            countdown_in_worker(c.start_delay, self.stop_event, self._log)
            if self.stop_event.is_set():
                self._log("测试框已取消。")
                return

            p = c.test_padding
            x1 = c.canvas_x + p
            y1 = c.canvas_y + p
            x2 = c.canvas_x + c.canvas_w - p
            y2 = c.canvas_y + c.canvas_h - p

            self._log("开始绘制测试框。")
            draw_screen_polyline(c, [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)], self.stop_event, is_raster_line=False)
            self._log("测试框完成。若框偏离画布，请重新标定。")
        except pyautogui.FailSafeException:
            safe_mouse_up()
            self._log("已触发 PyAutoGUI FailSafe：鼠标到达屏幕左上角，绘制停止。")
        except Exception as e:
            safe_mouse_up()
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
            paths, debug_img = make_paths(self.original_image, c, should_stop=self.stop_event.is_set)
            if not paths:
                messagebox.showwarning("没有可绘制路径", "当前参数没有生成路径，请换图或调整阈值/Canny/epsilon/扫描行距。")
                return
            self.last_paths = paths
            self.last_debug_image = debug_img
            self.path_count_var.set(f"已生成 {len(paths)} 条路径/横线")
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
            countdown_in_worker(c.start_delay, self.stop_event, self._log)
            if self.stop_event.is_set():
                self._log("绘制已取消。")
                return

            self._log(f"开始绘制，共 {len(paths)} 条路径/横线。")
            for idx, pts in enumerate(paths, start=1):
                if self.stop_event.is_set():
                    self._log("收到停止指令，正在结束。")
                    break

                # 把处理图坐标映射到屏幕坐标。处理图尺寸就是画布扣除 padding 后的尺寸。
                screen_points = [map_point_to_screen(c, p) for p in pts]
                is_raster_line = (c.method == "逐行扫描(横向)" and len(screen_points) == 2)
                draw_screen_polyline(c, screen_points, self.stop_event, is_raster_line=is_raster_line)
                if c.method in ("中心线追踪(线稿)", *MODEL_METHODS) and c.centerline_retrace and not self.stop_event.is_set():
                    draw_screen_polyline(c, list(reversed(screen_points)), self.stop_event, is_raster_line=False)

                drawn += 1
                if idx % 50 == 0:
                    self._log(f"进度：{idx}/{len(paths)} 条路径/横线。")

            self._log(f"绘制结束：已绘制 {drawn}/{len(paths)} 条路径/横线。")
        except pyautogui.FailSafeException:
            safe_mouse_up()
            self._log("已触发 PyAutoGUI FailSafe：鼠标到达屏幕左上角，绘制停止。")
        except Exception as e:
            safe_mouse_up()
            self._log(f"绘制失败：{e}")
        finally:
            if c.minimize_when_drawing:
                self.root.after(0, self.root.deiconify)

    def stop_drawing(self):
        self.stop_event.set()
        safe_mouse_up()
        self._log("已请求停止。若鼠标仍在移动，请把鼠标移到屏幕左上角触发紧急停止。")

    def _is_worker_running(self) -> bool:
        return self.worker_thread is not None and self.worker_thread.is_alive()

    # ---------- 日志 ----------
    def _clear_log(self):
        self.log_text.delete("1.0", tk.END)

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

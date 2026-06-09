"""Tkinter GUI 与用户交互调度。"""

import json
import os
import threading
import time
from dataclasses import asdict

import numpy as np
import pyautogui
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox, ttk

from .automation import countdown_in_worker, draw_screen_polyline, map_point_to_screen, safe_mouse_up
from .config import APP_DIR, DEFAULT_CONFIG_PATH, DrawConfig
from .processing import flatten_image, make_paths


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

        self._build_ui()
        self.load_config(silent=True)
        self._sync_config_to_vars()
        self.method_var.trace_add("write", lambda *_: self._update_param_visibility())
        self._update_param_visibility()
        self._log("先在电脑版微信中手动进入杯贴画布，再选择图片、标定画布、测试框、开始绘制。")
        self._log("紧急停止：把鼠标移到屏幕左上角，或点击“停止绘制”。")
        self._log("当前版本已优化断笔：横线延长、方向桥接、角度感知追踪、落笔预热、反向补笔、鼠标插值移动。")
        self._log("黑色线稿推荐使用“中心线追踪(线稿)”：新版会在分叉点按笔画方向延续，减少碎路径。")
        self._log("彩色动漫/插画可切换到“动漫线稿(Anime2Sketch)”，并选择 netG.pth 或 improved.bin 权重。")
        self._log("如果中心线追踪仍断笔或需要填充感，可以切换到“逐行扫描(横向)”。")

    # ---------- UI ----------
    def _build_ui(self):
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

        self._configure_style()
        self.root.configure(bg="#eef2f7")

        app = ttk.Frame(self.root, style="App.TFrame", padding=14)
        app.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(app, style="Header.TFrame", padding=(16, 12))
        header.pack(fill=tk.X, pady=(0, 12))

        title_block = ttk.Frame(header, style="Header.TFrame")
        title_block.pack(side=tk.LEFT, fill=tk.Y)
        ttk.Label(title_block, text="喜茶杯贴自动手绘工具", style="HeaderTitle.TLabel").pack(anchor="w")
        ttk.Label(title_block, textvariable=self.path_count_var, style="HeaderSubtle.TLabel").pack(anchor="w", pady=(3, 0))

        header_actions = ttk.Frame(header, style="Header.TFrame")
        header_actions.pack(side=tk.RIGHT)
        ttk.Button(header_actions, text="选择图片", style="Toolbar.TButton", command=self.choose_image).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(header_actions, text="刷新预览", style="Toolbar.TButton", command=self.refresh_preview).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(header_actions, text="测试画布", style="Toolbar.TButton", command=self.start_test_rectangle).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(header_actions, text="开始绘制", style="Primary.TButton", command=self.start_drawing).pack(side=tk.LEFT, padx=(10, 6))
        ttk.Button(header_actions, text="停止", style="Danger.TButton", command=self.stop_drawing).pack(side=tk.LEFT)

        body = ttk.Frame(app, style="App.TFrame")
        body.pack(fill=tk.BOTH, expand=True)

        shell = ttk.Frame(body, style="Card.TFrame", padding=0)
        shell.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 12))

        nav = ttk.Frame(shell, style="Nav.TFrame", padding=(10, 12))
        nav.pack(side=tk.LEFT, fill=tk.Y)
        settings_host = ttk.Frame(shell, style="Card.TFrame", padding=(12, 12))
        settings_host.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        settings_host.configure(width=370)
        settings_host.pack_propagate(False)

        right = ttk.Frame(body, style="App.TFrame")
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.settings_pages: dict[str, ttk.Frame] = {}
        self.nav_buttons: dict[str, ttk.Button] = {}

        image_tab = self._create_scroll_page(settings_host, "图片")
        canvas_tab = self._create_scroll_page(settings_host, "画布")
        lineart_tab = self._create_scroll_page(settings_host, "线稿")
        draw_tab = self._create_scroll_page(settings_host, "绘制")
        run_tab = self._create_scroll_page(settings_host, "运行")
        for key in ("图片", "画布", "线稿", "绘制", "运行"):
            self._add_nav_button(nav, key)
        self._show_settings_page("线稿")
        self.param_sections: dict[str, ttk.LabelFrame] = {}

        img_group = ttk.LabelFrame(image_tab, text="图片")
        img_group.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(img_group, text="选择图片", command=self.choose_image).pack(fill=tk.X, padx=8, pady=(8, 4))
        ttk.Button(img_group, text="刷新预览", command=self.refresh_preview).pack(fill=tk.X, padx=8, pady=4)
        ttk.Label(img_group, textvariable=self.image_label_var, wraplength=330).pack(fill=tk.X, padx=8, pady=(4, 8))

        self.section_image_common = self._create_param_section(image_tab, "通用图片参数")
        self.section_image_common.pack(fill=tk.X, pady=(0, 10))
        self._add_labeled_entry(self.section_image_common, "模糊强度 0/1/3/5", self.blur_var)
        ttk.Checkbutton(self.section_image_common, text="保持图片比例并居中", variable=self.keep_aspect_var).pack(anchor="w", padx=8, pady=(2, 8))

        canvas_group = ttk.LabelFrame(canvas_tab, text="画布位置")
        canvas_group.pack(fill=tk.X, pady=(0, 10))
        self._add_labeled_entry(canvas_group, "画布 X", self.canvas_x_var)
        self._add_labeled_entry(canvas_group, "画布 Y", self.canvas_y_var)
        self._add_labeled_entry(canvas_group, "画布宽 W", self.canvas_w_var)
        self._add_labeled_entry(canvas_group, "画布高 H", self.canvas_h_var)
        self._add_labeled_entry(canvas_group, "内边距", self.padding_var)

        canvas_actions = ttk.LabelFrame(canvas_tab, text="标定")
        canvas_actions.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(canvas_actions, text="3秒后记录左上角", command=lambda: self.start_calibration("tl")).pack(fill=tk.X, padx=8, pady=(8, 4))
        ttk.Button(canvas_actions, text="3秒后记录右下角", command=lambda: self.start_calibration("br")).pack(fill=tk.X, padx=8, pady=4)
        ttk.Button(canvas_actions, text="测试画布：画一个框", command=self.start_test_rectangle).pack(fill=tk.X, padx=8, pady=(4, 8))

        method_group = ttk.LabelFrame(lineart_tab, text="处理方式")
        method_group.pack(fill=tk.X, pady=(0, 10))
        self.method_combo = ttk.Combobox(
            method_group,
            textvariable=self.method_var,
            state="readonly",
            values=("中心线追踪(线稿)", "动漫线稿(Anime2Sketch)", "逐行扫描(横向)", "边缘线稿(Canny)", "黑白轮廓(阈值)"),
        )
        self.method_combo.pack(fill=tk.X, padx=8, pady=(8, 4))
        self.method_combo.bind("<<ComboboxSelected>>", lambda _e: self._update_param_visibility())
        ttk.Button(method_group, text="恢复当前模式默认参数", command=self.reset_current_mode_defaults).pack(fill=tk.X, padx=8, pady=4)
        self.method_hint_var = tk.StringVar(value="")
        ttk.Label(method_group, textvariable=self.method_hint_var, foreground="#555555", wraplength=330).pack(anchor="w", padx=8, pady=(2, 8))

        self.dynamic_param_frame = ttk.Frame(lineart_tab)
        self.dynamic_param_frame.pack(fill=tk.X)

        self.section_binary = self._create_param_section(self.dynamic_param_frame, "黑白线稿提取参数")
        self._add_labeled_entry(self.section_binary, "黑白阈值", self.threshold_var)
        ttk.Checkbutton(self.section_binary, text="深色区域作为线条", variable=self.dark_as_line_var).pack(anchor="w", padx=8, pady=(2, 8))

        self.section_centerline = self._create_param_section(self.dynamic_param_frame, "中心线追踪参数")
        self._add_labeled_entry(self.section_centerline, "断线连接像素", self.centerline_bridge_px_var)
        ttk.Label(self.section_centerline, text="建议 1~2。新版会按端点方向桥接断线；值越大越积极，但也更可能粘连相邻线条。", foreground="#666666", wraplength=330).pack(anchor="w", padx=8, pady=(2, 6))
        ttk.Checkbutton(self.section_centerline, text="中心线反向补笔（防空笔，更慢）", variable=self.centerline_retrace_var).pack(anchor="w", padx=8, pady=(2, 8))

        self.section_anime2sketch = self._create_param_section(self.dynamic_param_frame, "Anime2Sketch 模型参数")
        self._add_model_path_entry(self.section_anime2sketch)
        self._add_labeled_entry(self.section_anime2sketch, "模型输入尺寸", self.anime2sketch_input_size_var)
        self._add_labeled_entry(self.section_anime2sketch, "运行设备", self.anime2sketch_device_var)
        ttk.Label(self.section_anime2sketch, text="设备填写 auto/cpu/cuda。默认权重用 netG.pth，improved.bin 也可用。", foreground="#666666", wraplength=330).pack(anchor="w", padx=8, pady=(2, 8))

        self.section_canny = self._create_param_section(self.dynamic_param_frame, "Canny 边缘参数")
        self._add_labeled_entry(self.section_canny, "Canny 低阈值", self.canny_low_var)
        self._add_labeled_entry(self.section_canny, "Canny 高阈值", self.canny_high_var)

        self.section_path = self._create_param_section(self.dynamic_param_frame, "折线路径参数")
        self._add_labeled_entry(self.section_path, "路径简化 epsilon", self.epsilon_var)
        self._add_labeled_entry(self.section_path, "最短路径长度", self.min_path_len_var)
        self._add_labeled_entry(self.section_path, "最多路径数", self.max_paths_var)
        self._add_labeled_entry(self.section_path, "点采样步长", self.point_step_var)

        self.section_raster = self._create_param_section(self.dynamic_param_frame, "逐行扫描参数")
        self._add_labeled_entry(self.section_raster, "扫描行距", self.raster_row_step_var)
        self._add_labeled_entry(self.section_raster, "最短横线长度", self.raster_min_run_var)
        self._add_labeled_entry(self.section_raster, "断点合并容差", self.raster_gap_tolerance_var)
        self._add_labeled_entry(self.section_raster, "横线左右延长像素", self.raster_extend_px_var)
        self._add_labeled_entry(self.section_raster, "最多横线数", self.max_paths_var)
        ttk.Checkbutton(self.section_raster, text="蛇形顺序减少空移", variable=self.raster_serpentine_var).pack(anchor="w", padx=8, pady=2)
        ttk.Checkbutton(self.section_raster, text="横线往返补笔（更黑更慢）", variable=self.raster_backtrack_var).pack(anchor="w", padx=8, pady=(2, 8))

        self.section_stability = self._create_param_section(draw_tab, "断笔优化参数")
        self.section_stability.pack(fill=tk.X, pady=(0, 10))
        self._add_labeled_entry(self.section_stability, "鼠标插值步距px", self.mouse_step_px_var)
        self._add_labeled_entry(self.section_stability, "插值点等待秒", self.move_duration_var)
        self._add_labeled_entry(self.section_stability, "每笔最短秒数", self.min_stroke_duration_var)
        self._add_labeled_entry(self.section_stability, "每100px绘制秒数", self.stroke_duration_per_100px_var)
        self._add_labeled_entry(self.section_stability, "落笔前停顿秒", self.pre_down_pause_var)
        self._add_labeled_entry(self.section_stability, "按下后停顿秒", self.pen_down_pause_var)
        self._add_labeled_entry(self.section_stability, "落笔预热像素", self.pen_down_nudge_px_var)
        self._add_labeled_entry(self.section_stability, "抬笔前停顿秒", self.pen_up_pause_var)
        self._add_labeled_entry(self.section_stability, "笔画间隔秒", self.between_strokes_pause_var)

        self.section_misc = self._create_param_section(run_tab, "运行参数")
        self.section_misc.pack(fill=tk.X, pady=(0, 10))
        self._add_labeled_entry(self.section_misc, "开始前倒计时秒", self.start_delay_var)
        self._add_labeled_entry(self.section_misc, "测试框内边距", self.test_padding_var)
        ttk.Checkbutton(self.section_misc, text="绘制时最小化本窗口", variable=self.minimize_var).pack(anchor="w", padx=8, pady=(2, 8))

        action_group = ttk.LabelFrame(run_tab, text="操作")
        action_group.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(action_group, text="保存配置", command=self.save_config).pack(fill=tk.X, padx=8, pady=(8, 4))
        ttk.Button(action_group, text="加载配置", command=lambda: self.load_config(silent=False)).pack(fill=tk.X, padx=8, pady=4)
        ttk.Button(action_group, text="刷新预览", command=self.refresh_preview).pack(fill=tk.X, padx=8, pady=4)
        ttk.Button(action_group, text="开始绘制", command=self.start_drawing).pack(fill=tk.X, padx=8, pady=4)
        ttk.Button(action_group, text="停止绘制", command=self.stop_drawing).pack(fill=tk.X, padx=8, pady=(4, 8))

        preview_group = ttk.Frame(right, style="Card.TFrame", padding=12)
        preview_group.pack(fill=tk.BOTH, expand=True, pady=(0, 12))

        ttk.Label(preview_group, text="预览工作台", style="SectionTitle.TLabel").pack(anchor="w", pady=(0, 10))
        previews = ttk.PanedWindow(preview_group, orient=tk.HORIZONTAL)
        previews.pack(fill=tk.BOTH, expand=True)

        original_card = ttk.Frame(previews, style="Preview.TFrame", padding=8)
        processed_card = ttk.Frame(previews, style="Preview.TFrame", padding=8)
        previews.add(original_card, weight=1)
        previews.add(processed_card, weight=1)

        ttk.Label(original_card, text="原图", style="Subtle.TLabel").pack(anchor="w", pady=(0, 6))
        self.original_preview_label = ttk.Label(original_card, text="原图预览", anchor="center", style="Preview.TLabel")
        self.original_preview_label.pack(fill=tk.BOTH, expand=True)

        ttk.Label(processed_card, text="线稿", style="Subtle.TLabel").pack(anchor="w", pady=(0, 6))
        self.processed_preview_label = ttk.Label(processed_card, text="线稿预览", anchor="center", style="Preview.TLabel")
        self.processed_preview_label.pack(fill=tk.BOTH, expand=True)

        log_group = ttk.Frame(right, style="Card.TFrame", padding=12)
        log_group.pack(fill=tk.BOTH, expand=False)
        ttk.Label(log_group, text="运行日志", style="SectionTitle.TLabel").pack(anchor="w", pady=(0, 8))

        log_body = ttk.Frame(log_group, style="Card.TFrame")
        log_body.pack(fill=tk.BOTH, expand=True)
        self.log_text = tk.Text(log_body, height=10, wrap="word", bg="#0f172a", fg="#e5e7eb", insertbackground="#e5e7eb", relief=tk.FLAT, padx=10, pady=8)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        log_scroll = ttk.Scrollbar(log_body, command=self.log_text.yview)
        log_scroll.pack(side=tk.LEFT, fill=tk.Y)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        self.refresh_anime2sketch_models(silent=True)

    def _configure_style(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("App.TFrame", background="#eef2f7")
        style.configure("Header.TFrame", background="#172033")
        style.configure("HeaderTitle.TLabel", background="#172033", foreground="#ffffff", font=("Microsoft YaHei UI", 15, "bold"))
        style.configure("HeaderSubtle.TLabel", background="#172033", foreground="#b6c2d9", font=("Microsoft YaHei UI", 9))
        style.configure("Card.TFrame", background="#ffffff", relief=tk.FLAT)
        style.configure("Nav.TFrame", background="#f8fafc")
        style.configure("Preview.TFrame", background="#f8fafc", relief=tk.FLAT)
        style.configure("Preview.TLabel", background="#f8fafc", foreground="#64748b", font=("Microsoft YaHei UI", 10))
        style.configure("SectionTitle.TLabel", background="#ffffff", foreground="#111827", font=("Microsoft YaHei UI", 11, "bold"))
        style.configure("Subtle.TLabel", background="#f8fafc", foreground="#64748b", font=("Microsoft YaHei UI", 9))
        style.configure("TLabel", background="#ffffff", foreground="#1f2937", font=("Microsoft YaHei UI", 9))
        style.configure("TLabelframe", background="#ffffff", bordercolor="#d8dee9")
        style.configure("TLabelframe.Label", background="#ffffff", foreground="#334155", font=("Microsoft YaHei UI", 9, "bold"))
        style.configure("TCheckbutton", background="#ffffff", foreground="#1f2937", font=("Microsoft YaHei UI", 9))
        style.configure("TEntry", padding=(4, 3))
        style.configure("TCombobox", padding=(4, 3))
        style.configure("Toolbar.TButton", padding=(12, 7), font=("Microsoft YaHei UI", 9))
        style.configure("Primary.TButton", padding=(14, 7), font=("Microsoft YaHei UI", 9, "bold"), foreground="#ffffff", background="#2563eb")
        style.map("Primary.TButton", background=[("active", "#1d4ed8"), ("pressed", "#1e40af")])
        style.configure("Danger.TButton", padding=(12, 7), font=("Microsoft YaHei UI", 9, "bold"), foreground="#ffffff", background="#dc2626")
        style.map("Danger.TButton", background=[("active", "#b91c1c"), ("pressed", "#991b1b")])
        style.configure("Nav.TButton", padding=(14, 9), anchor="w", font=("Microsoft YaHei UI", 10), background="#f8fafc")
        style.configure("NavActive.TButton", padding=(14, 9), anchor="w", font=("Microsoft YaHei UI", 10, "bold"), foreground="#1d4ed8", background="#dbeafe")

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
        row.pack(fill=tk.X, padx=8, pady=2)
        ttk.Label(row, text=label, width=14).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=variable, width=14).pack(side=tk.LEFT, fill=tk.X, expand=True)
        return row

    def _add_model_path_entry(self, parent):
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, padx=8, pady=2)
        ttk.Label(row, text="模型权重文件", width=14).pack(side=tk.LEFT)
        self.anime2sketch_model_combo = ttk.Combobox(row, textvariable=self.anime2sketch_model_choice_var, state="readonly", width=14)
        self.anime2sketch_model_combo.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.anime2sketch_model_combo.bind("<<ComboboxSelected>>", lambda _e: self._select_anime2sketch_model())
        ttk.Button(row, text="刷新", command=self.refresh_anime2sketch_models).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Button(row, text="选择文件", command=self.choose_anime2sketch_model).pack(side=tk.LEFT, padx=(4, 0))
        return row

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
        self.anime2sketch_model_path_var.set(c.anime2sketch_model_path)
        self._set_anime2sketch_model_choice(c.anime2sketch_model_path)
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
                anime2sketch_model_path=self.anime2sketch_model_path_var.get().strip(),
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
        if c.method == "动漫线稿(Anime2Sketch)":
            if not c.anime2sketch_model_path:
                raise ValueError("请先选择 Anime2Sketch 模型权重文件（netG.pth 或 improved.bin）。")
            if not os.path.exists(c.anime2sketch_model_path):
                raise ValueError(f"Anime2Sketch 模型文件不存在：{c.anime2sketch_model_path}")
            if c.anime2sketch_input_size < 128:
                raise ValueError("Anime2Sketch 模型输入尺寸建议至少 128，推荐 512。")
            if c.anime2sketch_device.lower() not in ("auto", "cpu", "cuda"):
                raise ValueError("Anime2Sketch 运行设备只能填写 auto、cpu 或 cuda。")
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
            self._log(f"已刷新 Anime2Sketch 模型列表：{len(labels)} 个。")

    def _discover_anime2sketch_models(self) -> dict[str, str]:
        models: dict[str, str] = {}
        ignored_dirs = {".git", "venv", "__pycache__", ".pytest_cache", "build", "dist"}
        for cur_dir, dir_names, file_names in os.walk(APP_DIR):
            dir_names[:] = [name for name in dir_names if name not in ignored_dirs]
            for file_name in file_names:
                if os.path.splitext(file_name)[1].lower() not in (".pth", ".bin"):
                    continue
                path = os.path.join(cur_dir, file_name)
                models[self._model_choice_label(path)] = path

        return dict(sorted(models.items(), key=lambda item: item[0].lower()))

    def _model_choice_label(self, path: str) -> str:
        p = os.path.abspath(path)
        try:
            rel = os.path.relpath(p, APP_DIR)
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
            self._log(f"已选择 Anime2Sketch 模型：{path}")

    def choose_anime2sketch_model(self):
        path = filedialog.askopenfilename(
            title="选择 Anime2Sketch 模型权重",
            filetypes=[
                ("Anime2Sketch 权重", "*.pth *.bin"),
                ("所有文件", "*.*"),
            ],
        )
        if not path:
            return
        self.anime2sketch_model_path_var.set(path)
        self.refresh_anime2sketch_models(silent=True)
        self._set_anime2sketch_model_choice(path)
        self._log(f"已选择 Anime2Sketch 模型：{path}")

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
            elif c.method == "动漫线稿(Anime2Sketch)":
                self._log(f"Anime2Sketch 线稿已生成：{len(paths)} 条主干路径。彩色图细节过多时可提高最短路径长度或降低最多路径数。")
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
        img.thumbnail((390, 430), Image.Resampling.LANCZOS)
        self.preview_photo = ImageTk.PhotoImage(img)
        self.original_preview_label.configure(image=self.preview_photo, text="")

    def _show_processed_preview(self, debug_img: np.ndarray):
        # debug_img 是 0/255 灰度图：黑色线条、白色背景。
        pil = Image.fromarray(debug_img).convert("RGB")
        pil.thumbnail((390, 430), Image.Resampling.NEAREST)
        self.processed_photo = ImageTk.PhotoImage(pil)
        self.processed_preview_label.configure(image=self.processed_photo, text="")
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
                if c.method in ("中心线追踪(线稿)", "动漫线稿(Anime2Sketch)") and c.centerline_retrace and not self.stop_event.is_set():
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

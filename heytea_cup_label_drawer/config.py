"""应用配置模型与默认路径。"""

from dataclasses import dataclass
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = APP_DIR / "heytea_draw_config_centerline.json"
MODELS_DIR = APP_DIR / "models"
ANIME2SKETCH_MODELS_DIR = MODELS_DIR / "anime2sketch"
ANILINES_MODELS_DIR = MODELS_DIR / "anilines"
INFORMATIVE_DRAWINGS_MODELS_DIR = MODELS_DIR / "informative_drawings"


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
    threshold: int = 210
    blur: int = 0
    dark_as_line: bool = True
    keep_aspect: bool = True
    # 中心线模式：先做轻微闭合，再按端点方向桥接小断点。
    # 0 = 不连接；1 = 保守连接；2 = 推荐；3 以上更激进。
    centerline_bridge_px: int = 2

    # Anime2Sketch 模式：用模型先把彩图/插画转成黑白线稿，再按中心线追踪绘制。
    anime2sketch_model_path: str = ""
    anilines_model_path: str = ""
    informative_drawings_model_path: str = ""
    anime2sketch_input_size: int = 512
    anime2sketch_device: str = "auto"

    epsilon: float = 0.7
    min_path_len: float = 6.0
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
    pen_down_pause: float = 0.035
    pen_down_nudge_px: int = 2
    pen_up_pause: float = 0.020
    between_strokes_pause: float = 0.012
    centerline_retrace: bool = True
    contour_retrace: bool = False
    raster_backtrack: bool = False

    start_delay: int = 3
    test_padding: int = 20
    minimize_when_drawing: bool = True

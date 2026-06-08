"""鼠标绘制与屏幕坐标相关的副作用逻辑。"""

import time

import numpy as np
import pyautogui

from .config import DrawConfig

# 不把 PyAutoGUI 的 MINIMUM_DURATION 强行设为 0。
# 某些 PyAutoGUI 版本在 duration 很小且 MINIMUM_DURATION=0 时会出现 float division by zero。
# 下面绘制时改用 safe_move_to() 处理小 duration。
pyautogui.PAUSE = 0.002
pyautogui.FAILSAFE = True


def draw_screen_polyline(c: DrawConfig, points: list[tuple[int, int]], stop_event, is_raster_line: bool = False):
    """按下鼠标绘制一条屏幕坐标折线。

    优化点：
    1. 落笔前/按下后/抬笔前加入短暂停顿，避免小程序只收到点按而没收到拖动。
    2. 按下后先沿首段做一个很小的预热拖动，让微信画布确认这是一笔拖动。
    3. 每个线段按距离切成多个小鼠标 move 事件，不再依赖 pyautogui 的极短 duration。
    4. 逐行扫描横线可选往返补笔：同一笔里先正向再反向，增强连续性和黑度。
    """
    if len(points) < 2:
        return

    start_x, start_y = points[0]
    safe_move_to(start_x, start_y, duration=0)
    sleep_interruptible(c.pre_down_pause, stop_event)
    if stop_event.is_set():
        return

    pyautogui.mouseDown(button="left")
    try:
        sleep_interruptible(effective_pen_down_pause(c, is_raster_line), stop_event)
        if stop_event.is_set():
            return

        prime_pen_down(c, points, stop_event)
        if stop_event.is_set():
            return

        for x2, y2 in points[1:]:
            if stop_event.is_set():
                break
            cur = pyautogui.position()
            draw_smooth_segment(c, int(cur.x), int(cur.y), int(x2), int(y2), stop_event, is_raster_line=is_raster_line)

        # 横线往返补笔：仅对逐行扫描横线生效。同一笔不抬笔返回，通常比重复落笔更稳。
        if is_raster_line and c.raster_backtrack and not stop_event.is_set():
            x_back, y_back = points[0]
            cur = pyautogui.position()
            draw_smooth_segment(c, int(cur.x), int(cur.y), int(x_back), int(y_back), stop_event, is_raster_line=True)

        sleep_interruptible(c.pen_up_pause, stop_event)
    finally:
        pyautogui.mouseUp(button="left")

    sleep_interruptible(c.between_strokes_pause, stop_event)


def prime_pen_down(c: DrawConfig, points: list[tuple[int, int]], stop_event):
    """按下后制造可见的微小拖动，避免整条轨迹只触发 hover 而没有笔画。"""
    if len(points) < 2 or c.pen_down_nudge_px <= 0:
        return

    x1, y1 = points[0]
    x2, y2 = points[1]
    dx = x2 - x1
    dy = y2 - y1
    dist = float((dx * dx + dy * dy) ** 0.5)
    if dist <= 0:
        return

    nudge = min(float(c.pen_down_nudge_px), dist)
    nx = int(round(x1 + dx / dist * nudge))
    ny = int(round(y1 + dy / dist * nudge))
    pyautogui.moveTo(nx, ny, duration=0)
    sleep_interruptible(max(c.move_duration, 0.012), stop_event)
    if not stop_event.is_set():
        pyautogui.moveTo(x1, y1, duration=0)
        sleep_interruptible(max(c.move_duration, 0.012), stop_event)


def effective_pen_down_pause(c: DrawConfig, is_raster_line: bool = False) -> float:
    """旧配置可能保存过很小的停顿；绘制时给落笔留一个可靠下限。"""
    minimum = 0.025 if is_raster_line else 0.035
    return max(float(c.pen_down_pause or 0.0), minimum)


def draw_smooth_segment(c: DrawConfig, x1: int, y1: int, x2: int, y2: int, stop_event, is_raster_line: bool = False):
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
        if stop_event.is_set():
            break
        t = i / steps
        x = int(round(x1 + dx * t))
        y = int(round(y1 + dy * t))
        pyautogui.moveTo(x, y, duration=0)
        sleep_interruptible(per_step_sleep, stop_event)

def sleep_interruptible(seconds: float, stop_event):
    seconds = max(0.0, float(seconds or 0.0))
    if seconds <= 0:
        return
    end = time.perf_counter() + seconds
    while not stop_event.is_set():
        remaining = end - time.perf_counter()
        if remaining <= 0:
            return
        time.sleep(min(0.02, remaining))

def map_point_to_screen(c: DrawConfig, p: np.ndarray | tuple[int, int]) -> tuple[int, int]:
    # p 已经在 inner_w/inner_h 坐标内，所以只要加画布起点和 padding。
    px = int(p[0])
    py = int(p[1])
    return c.canvas_x + c.padding + px, c.canvas_y + c.padding + py

def countdown_in_worker(seconds: int, stop_event, log):
    for n in range(seconds, 0, -1):
        if stop_event.is_set():
            return
        log(f"{n} 秒后开始，请切到微信杯贴画布。")
        time.sleep(1)

def safe_move_to(x: int, y: int, duration: float = 0.0):
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

def safe_mouse_up():
    try:
        pyautogui.mouseUp(button="left")
    except Exception:
        pass

"""图片预处理、路径生成与骨架追踪算法。"""

import cv2
import numpy as np
from PIL import Image, ImageOps

from .anilines import anilines_to_gray
from .anime2sketch import anime2sketch_to_gray
from .config import DrawConfig
from .informative_drawings import informative_drawings_to_gray


OFFSETS_8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def flatten_image(img: Image.Image) -> Image.Image:
    """把透明图层铺到白底，统一转成 RGB。"""
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        bg.alpha_composite(rgba)
        return bg.convert("RGB")
    return img.convert("RGB")

def make_work_image(original_image: Image.Image | None, c: DrawConfig) -> Image.Image:
    if original_image is None:
        raise RuntimeError("请先选择图片。")

    inner_w = max(10, c.canvas_w - c.padding * 2)
    inner_h = max(10, c.canvas_h - c.padding * 2)

    src = original_image.copy()
    canvas = Image.new("RGB", (inner_w, inner_h), "white")

    if c.keep_aspect:
        fitted = ImageOps.contain(src, (inner_w, inner_h), Image.Resampling.LANCZOS)
        ox = (inner_w - fitted.width) // 2
        oy = (inner_h - fitted.height) // 2
        canvas.paste(fitted, (ox, oy))
    else:
        canvas = src.resize((inner_w, inner_h), Image.Resampling.LANCZOS)

    return canvas

def make_paths(original_image: Image.Image | None, c: DrawConfig, should_stop=None) -> tuple[list[np.ndarray], np.ndarray]:
    work_img = make_work_image(original_image, c)
    rgb = np.array(work_img)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    if c.blur > 1:
        k = c.blur if c.blur % 2 == 1 else c.blur + 1
        gray = cv2.GaussianBlur(gray, (k, k), 0)

    if c.method == "中心线追踪(线稿)":
        return make_centerline_paths(gray, c)

    model_methods = {
        "动漫线稿(Anime2Sketch)": (anime2sketch_to_gray, c.anime2sketch_model_path),
        "动漫精细线稿(AniLines)": (anilines_to_gray, c.anilines_model_path),
        "通用语义线稿(Informative Drawings)": (informative_drawings_to_gray, c.informative_drawings_model_path),
    }
    if c.method in model_methods:
        converter, model_path = model_methods[c.method]
        lineart = converter(rgb, model_path, c.anime2sketch_input_size, c.anime2sketch_device)
        model_config = DrawConfig(**{**c.__dict__, "dark_as_line": True})
        return make_centerline_paths(lineart, model_config)

    if c.method == "逐行扫描(横向)":
        return make_raster_paths(gray, c, should_stop=should_stop)

    if c.method == "黑白轮廓(阈值)":
        thresh_type = cv2.THRESH_BINARY_INV if c.dark_as_line else cv2.THRESH_BINARY
        _, binary = cv2.threshold(gray, c.threshold, 255, thresh_type)
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        debug = 255 - binary  # 预览中用黑线白底显示
    else:
        edges = cv2.Canny(gray, c.canny_low, c.canny_high)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        debug = 255 - edges  # 黑线白底

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
    selected = [pts for _, pts in scored[: c.max_paths]]
    paths = order_paths_greedy(selected, retrace=c.method == "黑白轮廓(阈值)" and c.contour_retrace)

    return paths, debug

def make_centerline_paths(gray: np.ndarray, c: DrawConfig) -> tuple[list[np.ndarray], np.ndarray]:
    """中心线追踪：适合黑色线稿。

    这里使用“细化骨架 + 角度感知追踪”的中心线矢量化流程：
    1. 按阈值提取黑色线条 mask，并清掉极小噪点；
    2. 对 mask 做轻微闭合，连接小断点；
    3. 用 Zhang-Suen thinning 细化为单像素骨架；
    4. 在骨架端点之间按方向桥接小间隙，再剪掉短毛刺；
    5. 追踪骨架时在分叉点优先选择方向最顺的延续边，减少交叉处碎笔。

    对鼠标绘制来说，最后一步比“遇到分叉就切段”的传统骨架追踪更稳定。
    """
    if c.dark_as_line:
        mask = gray < c.threshold
    else:
        mask = gray > c.threshold

    mask_u8 = mask.astype(np.uint8) * 255
    mask_u8 = remove_small_components(mask_u8, min_area=3)
    if c.centerline_bridge_px > 0:
        k = int(c.centerline_bridge_px) * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)

    skeleton = zhang_suen_thinning(mask_u8 > 0)
    if c.centerline_bridge_px > 0:
        skeleton = bridge_skeleton_endpoints(
            skeleton,
            max_gap=2 + int(c.centerline_bridge_px) * 4,
            min_alignment=0.35,
        )
    skeleton = prune_skeleton_spurs(
        skeleton,
        max_spur_len=max(1, min(4, int(round(c.min_path_len / 2.0)))),
    )
    raw_paths = trace_skeleton_paths(skeleton)

    scored: list[tuple[float, np.ndarray]] = []
    for pts in raw_paths:
        if pts is None or len(pts) < 2:
            continue

        length = polyline_length(pts)
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
    selected = [pts for _, pts in scored[: c.max_paths]]
    paths = order_paths_greedy(selected, retrace=c.centerline_retrace)

    h, w = gray.shape[:2]
    debug = np.full((h, w), 255, dtype=np.uint8)
    for pts in paths:
        cv2.polylines(debug, [pts.reshape(-1, 1, 2)], isClosed=False, color=0, thickness=1)

    return paths, debug

def zhang_suen_thinning(binary: np.ndarray, max_iter: int = 120) -> np.ndarray:
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


def remove_small_components(mask_u8: np.ndarray, min_area: int = 3) -> np.ndarray:
    """移除极小黑点噪声，避免细化后生成独立碎路径。"""
    if min_area <= 1:
        return mask_u8
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask_u8 > 0).astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return mask_u8

    cleaned = np.zeros_like(mask_u8)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == label] = 255
    return cleaned


def skeleton_neighbors(pixels: set[tuple[int, int]], h: int, w: int, p: tuple[int, int]) -> list[tuple[int, int]]:
    y, x = p
    out = []
    for dy, dx in OFFSETS_8:
        q = (y + dy, x + dx)
        if 0 <= q[0] < h and 0 <= q[1] < w and q in pixels:
            out.append(q)
    out.sort()
    return out


def skeleton_degrees(skeleton: np.ndarray) -> tuple[set[tuple[int, int]], dict[tuple[int, int], int]]:
    h, w = skeleton.shape[:2]
    ys, xs = np.nonzero(skeleton)
    pixels = set(zip(ys.tolist(), xs.tolist()))
    degrees = {p: len(skeleton_neighbors(pixels, h, w, p)) for p in pixels}
    return pixels, degrees


def estimate_endpoint_outward(
    pixels: set[tuple[int, int]],
    h: int,
    w: int,
    endpoint: tuple[int, int],
    depth: int = 6,
) -> np.ndarray | None:
    """估计端点向外延伸方向，用于判断两个断点是否应该桥接。"""
    path = [endpoint]
    prev = None
    cur = endpoint

    for _ in range(depth):
        neighbors = skeleton_neighbors(pixels, h, w, cur)
        if prev is not None:
            neighbors = [q for q in neighbors if q != prev]
        if len(neighbors) != 1:
            break
        prev, cur = cur, neighbors[0]
        path.append(cur)

    if len(path) < 2:
        return None

    start = np.array(path[0], dtype=np.float32)
    inner = np.array(path[-1], dtype=np.float32)
    outward = start - inner
    norm = float(np.linalg.norm(outward))
    if norm <= 0:
        return None
    return outward / norm


def bridge_skeleton_endpoints(
    skeleton: np.ndarray,
    max_gap: int,
    min_alignment: float = 0.35,
) -> np.ndarray:
    """按端点方向连接小断点。

    只连接距离近且两端都朝向彼此的端点，避免把平行近线条误粘在一起。
    """
    if max_gap <= 1:
        return skeleton

    h, w = skeleton.shape[:2]
    pixels, degrees = skeleton_degrees(skeleton)
    endpoints = [p for p, degree in degrees.items() if degree == 1]
    if len(endpoints) < 2:
        return skeleton

    directions = {
        p: estimate_endpoint_outward(pixels, h, w, p)
        for p in endpoints
    }
    candidates: list[tuple[float, tuple[int, int], tuple[int, int]]] = []
    max_gap_sq = float(max_gap * max_gap)

    for i, p in enumerate(endpoints):
        dir_p = directions.get(p)
        if dir_p is None:
            continue
        for q in endpoints[i + 1:]:
            dir_q = directions.get(q)
            if dir_q is None:
                continue

            gap = np.array([q[0] - p[0], q[1] - p[1]], dtype=np.float32)
            dist_sq = float(gap.dot(gap))
            if dist_sq <= 1.0 or dist_sq > max_gap_sq:
                continue

            dist = dist_sq ** 0.5
            gap_dir = gap / dist
            align_p = float(dir_p.dot(gap_dir))
            align_q = float(dir_q.dot(-gap_dir))
            if align_p < min_alignment or align_q < min_alignment:
                continue

            # 距离优先，方向越一致越优先。
            score = dist - (align_p + align_q)
            candidates.append((score, p, q))

    if not candidates:
        return skeleton

    bridged = skeleton.astype(np.uint8).copy()
    used: set[tuple[int, int]] = set()
    for _, p, q in sorted(candidates, key=lambda item: item[0]):
        if p in used or q in used:
            continue
        cv2.line(bridged, (p[1], p[0]), (q[1], q[0]), color=1, thickness=1)
        used.add(p)
        used.add(q)

    return zhang_suen_thinning(bridged > 0)


def prune_skeleton_spurs(skeleton: np.ndarray, max_spur_len: int) -> np.ndarray:
    """剪掉端点到分叉点之间的极短毛刺。"""
    if max_spur_len <= 0:
        return skeleton

    pruned = skeleton.copy()
    h, w = pruned.shape[:2]

    for _ in range(max_spur_len):
        pixels, degrees = skeleton_degrees(pruned)
        endpoints = [p for p, degree in degrees.items() if degree == 1]
        to_remove: set[tuple[int, int]] = set()

        for endpoint in endpoints:
            path = [endpoint]
            prev = None
            cur = endpoint

            while len(path) <= max_spur_len + 1:
                neighbors = skeleton_neighbors(pixels, h, w, cur)
                if prev is not None:
                    neighbors = [q for q in neighbors if q != prev]
                if not neighbors:
                    break
                if len(neighbors) > 1:
                    break
                prev, cur = cur, neighbors[0]
                path.append(cur)
                if degrees.get(cur, 0) != 2:
                    break

            last_degree = degrees.get(path[-1], 0)
            if len(path) - 1 <= max_spur_len and last_degree >= 3:
                to_remove.update(path[:-1])

        if not to_remove:
            break
        for y, x in to_remove:
            pruned[y, x] = False

    return pruned


def trace_skeleton_paths(skeleton: np.ndarray) -> list[np.ndarray]:
    """把单像素骨架追踪成多条折线。

    遇到分叉点时不立刻断开，而是优先沿夹角最小的方向继续走。
    这样十字、T 字和轻微交叉的线稿会更像一笔画过，而不是被切成碎段。
    """
    h, w = skeleton.shape[:2]
    pixels, degree = skeleton_degrees(skeleton)
    if not pixels:
        return []

    def nbs(p: tuple[int, int]) -> list[tuple[int, int]]:
        return skeleton_neighbors(pixels, h, w, p)

    visited_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    paths_yx: list[list[tuple[int, int]]] = []

    def edge_key(a: tuple[int, int], b: tuple[int, int]):
        return (a, b) if a <= b else (b, a)

    def choose_next(
        prev: tuple[int, int],
        cur: tuple[int, int],
        candidates: list[tuple[int, int]],
    ) -> tuple[int, int] | None:
        if not candidates:
            return None

        prev_vec = np.array([cur[0] - prev[0], cur[1] - prev[1]], dtype=np.float32)
        prev_norm = float(np.linalg.norm(prev_vec))
        if prev_norm <= 0:
            return candidates[0]
        prev_vec /= prev_norm

        scored = []
        for q in candidates:
            next_vec = np.array([q[0] - cur[0], q[1] - cur[1]], dtype=np.float32)
            next_norm = float(np.linalg.norm(next_vec))
            if next_norm <= 0:
                continue
            scored.append((float(prev_vec.dot(next_vec / next_norm)), q))
        if not scored:
            return candidates[0]

        score, best = max(scored, key=lambda item: (item[0], item[1]))
        # 普通链路必须继续；分叉点只有方向足够顺才跨过去，避免 T 字竖笔硬拐。
        if degree.get(cur, 0) == 2 or score >= 0.15:
            return best
        return None

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

            candidates = [q for q in nbs(cur) if q != prev and edge_key(cur, q) not in visited_edges]
            next_point = choose_next(prev, cur, candidates)
            if next_point is None:
                break
            prev, cur = cur, next_point
        return path

    # 先从端点出发，能把自然笔画尽量连成长路径。
    endpoints = sorted([p for p, d in degree.items() if d == 1])
    junctions = sorted([p for p, d in degree.items() if d > 2])

    for p in endpoints:
        for q in nbs(p):
            if edge_key(p, q) not in visited_edges:
                path = trace_from(p, q)
                if len(path) >= 2:
                    paths_yx.append(path)

    # 再处理分叉点剩余边。
    for p in junctions:
        for q in nbs(p):
            if edge_key(p, q) not in visited_edges:
                path = trace_from(p, q)
                if len(path) >= 2:
                    paths_yx.append(path)

    # 最后处理闭环：闭环所有点 degree=2，没有端点。
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

def polyline_length(pts: np.ndarray) -> float:
    if pts is None or len(pts) < 2:
        return 0.0
    diffs = np.diff(pts.astype(np.float32), axis=0)
    return float(np.sqrt((diffs * diffs).sum(axis=1)).sum())


def order_paths_greedy(
    paths: list[np.ndarray],
    start: np.ndarray | tuple[int, int] = (0, 0),
    retrace: bool = False,
) -> list[np.ndarray]:
    """按最近端点排列笔画，并在反向更近时反转整条笔画。"""
    remaining = [np.asarray(path) for path in paths if path is not None and len(path) >= 2]
    if not remaining:
        return []

    cell_size = 8
    buckets: dict[tuple[int, int], dict[tuple[int, bool], None]] = {}

    def cell_of(point: np.ndarray) -> tuple[int, int]:
        return int(np.floor(float(point[0]) / cell_size)), int(np.floor(float(point[1]) / cell_size))

    for index, path in enumerate(remaining):
        for reverse, point in ((False, path[0]), (True, path[-1])):
            buckets.setdefault(cell_of(point), {})[(index, reverse)] = None

    def ring_cells(cx: int, cy: int, radius: int):
        if radius == 0:
            yield cx, cy
            return
        for x in range(cx - radius, cx + radius + 1):
            yield x, cy - radius
            yield x, cy + radius
        for y in range(cy - radius + 1, cy + radius):
            yield cx - radius, y
            yield cx + radius, y

    def nearest_endpoint(point: np.ndarray) -> tuple[int, bool]:
        cx, cy = cell_of(point)
        best: tuple[float, int, bool] | None = None
        radius = 0

        while best is None or radius * cell_size <= best[0] ** 0.5 + cell_size:
            for cell in ring_cells(cx, cy, radius):
                for index, reverse in buckets.get(cell, {}):
                    endpoint = remaining[index][-1 if reverse else 0].astype(np.float64)
                    delta = endpoint - point
                    candidate = (float(delta.dot(delta)), index, reverse)
                    if best is None or candidate < best:
                        best = candidate
            radius += 1

        if best is None:
            raise RuntimeError("路径端点索引为空。")
        return best[1], best[2]

    ordered: list[np.ndarray] = []
    current = np.asarray(start, dtype=np.float64)

    for _ in range(len(remaining)):
        best_index, best_reverse = nearest_endpoint(current)
        selected = remaining[best_index]
        for reverse, endpoint in ((False, selected[0]), (True, selected[-1])):
            buckets[cell_of(endpoint)].pop((best_index, reverse), None)

        if best_reverse:
            selected = selected[::-1].copy()

        ordered.append(selected)
        current = selected[0 if retrace else -1].astype(np.float64)

    return ordered


def make_raster_paths(gray: np.ndarray, c: DrawConfig, should_stop=None) -> tuple[list[np.ndarray], np.ndarray]:
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
        if should_stop is not None and should_stop():
            break

        row = mask[y]
        runs = row_runs(row, c.raster_min_run, c.raster_gap_tolerance)
        if not runs:
            continue

        # 先做左右延长，提升覆盖率，减少视觉断笔；同时不要越过图片边界。
        expanded_runs: list[tuple[int, int]] = []
        for x1, x2 in runs:
            x1e = max(0, int(x1) - c.raster_extend_px)
            x2e = min(w - 1, int(x2) + c.raster_extend_px)
            if x2e > x1e:
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

def row_runs(row: np.ndarray, min_run: int, gap_tolerance: int) -> list[tuple[int, int]]:
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

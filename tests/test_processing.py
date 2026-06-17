import unittest

import numpy as np
from PIL import Image, ImageDraw

from heytea_cup_label_drawer.config import ANILINES_MODELS_DIR, INFORMATIVE_DRAWINGS_MODELS_DIR, DrawConfig
from heytea_cup_label_drawer.processing import (
    make_paths,
    make_raster_paths,
    order_paths_greedy,
    polyline_length,
    trace_skeleton_paths,
    zhang_suen_thinning,
)


def blank_line_art(size=(100, 100)):
    image = Image.new("RGB", size, "white")
    return image, ImageDraw.Draw(image)


class CenterlineTracingTests(unittest.TestCase):
    def test_thinning_reduces_thick_line_to_single_pixel_centerline(self):
        mask = np.zeros((15, 25), dtype=bool)
        mask[5:10, 3:22] = True

        skeleton = zhang_suen_thinning(mask)

        self.assertTrue(skeleton.any())
        self.assertLessEqual(int(skeleton[:, 6:19].sum(axis=0).max()), 1)

    def test_disconnected_skeleton_is_traced_as_continuous_strokes(self):
        skeleton = np.zeros((12, 20), dtype=bool)
        skeleton[2, 2:8] = True
        skeleton[8, 11:18] = True

        paths = trace_skeleton_paths(skeleton)

        self.assertEqual(len(paths), 2)
        for path in paths:
            steps = np.abs(np.diff(path, axis=0))
            self.assertTrue(np.all(steps.max(axis=1) <= 1))

    def test_arc_traces_as_one_main_path(self):
        image, draw = blank_line_art()
        draw.arc((15, 15, 85, 85), start=20, end=330, fill="black", width=4)

        paths, _ = make_paths(image, DrawConfig(canvas_w=100, canvas_h=100, padding=0))

        self.assertEqual(len(paths), 1)
        self.assertGreater(polyline_length(paths[0]), 150)

    def test_cross_keeps_two_straight_main_strokes(self):
        image, draw = blank_line_art()
        draw.line((10, 50, 90, 50), fill="black", width=5)
        draw.line((50, 10, 50, 90), fill="black", width=5)

        paths, _ = make_paths(image, DrawConfig(canvas_w=100, canvas_h=100, padding=0))

        self.assertEqual(len(paths), 2)
        self.assertTrue(all(polyline_length(path) > 70 for path in paths))

    def test_aggressive_bridge_connects_collinear_gap(self):
        image, draw = blank_line_art()
        draw.line((10, 50, 45, 50), fill="black", width=4)
        draw.line((55, 50, 90, 50), fill="black", width=4)

        config = DrawConfig(canvas_w=100, canvas_h=100, padding=0, centerline_bridge_px=4)
        paths, _ = make_paths(image, config)

        self.assertEqual(len(paths), 1)
        self.assertGreater(polyline_length(paths[0]), 70)


class GreedyPathOrderingTests(unittest.TestCase):
    def test_selects_nearest_next_stroke_and_reverses_it(self):
        paths = [
            np.array([[10, 0], [20, 0]], dtype=np.int32),
            np.array([[40, 0], [30, 0]], dtype=np.int32),
            np.array([[2, 0], [4, 0]], dtype=np.int32),
        ]

        ordered = order_paths_greedy(paths)

        self.assertEqual([path.tolist() for path in ordered], [
            [[2, 0], [4, 0]],
            [[10, 0], [20, 0]],
            [[30, 0], [40, 0]],
        ])

    def test_retrace_orders_from_actual_returned_mouse_position(self):
        paths = [
            np.array([[1, 0], [100, 0]], dtype=np.int32),
            np.array([[4, 0], [5, 0]], dtype=np.int32),
            np.array([[90, 0], [95, 0]], dtype=np.int32),
        ]

        ordered = order_paths_greedy(paths, retrace=True)

        self.assertEqual([path[0].tolist() for path in ordered], [[1, 0], [4, 0], [90, 0]])


class BlackWhiteThresholdTests(unittest.TestCase):
    def test_black_white_threshold_reduces_thick_line_to_single_stroke(self):
        image, draw = blank_line_art()
        draw.line((10, 50, 90, 50), fill="black", width=11)
        config = DrawConfig(
            canvas_w=100,
            canvas_h=100,
            padding=0,
            method="黑白轮廓(阈值)",
            threshold=150,
            dark_as_line=True,
        )

        paths, preview = make_paths(image, config)

        self.assertEqual(len(paths), 1)
        ys = np.flatnonzero((preview == 0).any(axis=1))
        self.assertTrue(ys.size > 0)
        self.assertLessEqual(int(ys.max() - ys.min()), 2)

    def test_black_white_preview_shows_drawable_path_not_binary_area(self):
        image, draw = blank_line_art()
        draw.rectangle((20, 45, 80, 55), fill="black")
        config = DrawConfig(
            canvas_w=100,
            canvas_h=100,
            padding=0,
            method="黑白轮廓(阈值)",
            threshold=150,
            dark_as_line=True,
        )

        _paths, preview = make_paths(image, config)

        self.assertLess(int(np.count_nonzero(preview == 0)), 100)


class RasterPathTests(unittest.TestCase):
    def test_zero_length_raster_point_is_not_shown_as_drawable_path(self):
        gray = np.full((3, 5), 255, dtype=np.uint8)
        gray[1, 2] = 0
        config = DrawConfig(
            threshold=128,
            dark_as_line=True,
            raster_row_step=1,
            raster_min_run=1,
            raster_gap_tolerance=0,
            raster_extend_px=0,
        )

        paths, preview = make_raster_paths(gray, config)

        self.assertEqual(paths, [])
        self.assertTrue(np.all(preview == 255))


class ModelLineArtTests(unittest.TestCase):
    def test_anime2sketch_requires_model_path_before_torch(self):
        image, _ = blank_line_art()
        config = DrawConfig(canvas_w=100, canvas_h=100, padding=0, method="动漫线稿(Anime2Sketch)")

        with self.assertRaisesRegex(RuntimeError, "Anime2Sketch 模型权重文件"):
            make_paths(image, config)

    def test_anilines_requires_model_path_before_torch(self):
        image, _ = blank_line_art()
        config = DrawConfig(canvas_w=100, canvas_h=100, padding=0, method="动漫精细线稿(AniLines)")

        with self.assertRaisesRegex(RuntimeError, "AniLines Detail"):
            make_paths(image, config)

    def test_informative_drawings_requires_model_path_before_runtime(self):
        image, _ = blank_line_art()
        config = DrawConfig(canvas_w=100, canvas_h=100, padding=0, method="通用语义线稿(Informative Drawings)")

        with self.assertRaisesRegex(RuntimeError, "Informative Drawings ONNX"):
            make_paths(image, config)

    @unittest.skipUnless((ANILINES_MODELS_DIR / "detail.pth").exists(), "AniLines test model is not installed")
    def test_anilines_model_runs(self):
        image, _ = blank_line_art((128, 128))
        config = DrawConfig(
            canvas_w=128,
            canvas_h=128,
            padding=0,
            method="动漫精细线稿(AniLines)",
            anilines_model_path=str(ANILINES_MODELS_DIR / "detail.pth"),
            anime2sketch_input_size=128,
            anime2sketch_device="cpu",
        )

        paths, preview = make_paths(image, config)

        self.assertEqual(preview.shape, (128, 128))
        self.assertIsInstance(paths, list)

    @unittest.skipUnless((INFORMATIVE_DRAWINGS_MODELS_DIR / "model.onnx").exists(), "Informative Drawings test model is not installed")
    def test_informative_drawings_model_runs(self):
        image, _ = blank_line_art((128, 128))
        config = DrawConfig(
            canvas_w=128,
            canvas_h=128,
            padding=0,
            method="通用语义线稿(Informative Drawings)",
            informative_drawings_model_path=str(INFORMATIVE_DRAWINGS_MODELS_DIR / "model.onnx"),
            anime2sketch_input_size=128,
            anime2sketch_device="cpu",
        )

        paths, preview = make_paths(image, config)

        self.assertEqual(preview.shape, (128, 128))
        self.assertIsInstance(paths, list)


if __name__ == "__main__":
    unittest.main()

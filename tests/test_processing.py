import unittest

from PIL import Image, ImageDraw

from heytea_cup_label_drawer.config import ANILINES_MODELS_DIR, INFORMATIVE_DRAWINGS_MODELS_DIR, DrawConfig
from heytea_cup_label_drawer.processing import make_paths, polyline_length


def blank_line_art(size=(100, 100)):
    image = Image.new("RGB", size, "white")
    return image, ImageDraw.Draw(image)


class CenterlineTracingTests(unittest.TestCase):
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

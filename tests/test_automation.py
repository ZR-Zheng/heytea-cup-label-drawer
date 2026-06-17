import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from heytea_cup_label_drawer.automation import (
    draw_screen_polyline,
    draw_smooth_segment,
    effective_pen_down_pause,
    prime_pen_down,
    raster_line_needs_confirmation,
    should_retrace_path,
)
from heytea_cup_label_drawer.config import DrawConfig


class PenDownPrimingTests(unittest.TestCase):
    @patch("heytea_cup_label_drawer.automation.sleep_interruptible")
    @patch("heytea_cup_label_drawer.automation.pyautogui.moveTo")
    def test_prime_pen_down_moves_forward_and_back(self, move_to, _sleep):
        config = DrawConfig(pen_down_nudge_px=2)

        prime_pen_down(config, [(10, 20), (20, 20)], threading.Event())

        self.assertEqual(move_to.call_args_list[0].args[:2], (12, 20))
        self.assertEqual(move_to.call_args_list[1].args[:2], (10, 20))

    @patch("heytea_cup_label_drawer.automation.pyautogui.moveTo")
    def test_zero_nudge_does_not_move(self, move_to):
        config = DrawConfig(pen_down_nudge_px=0)

        prime_pen_down(config, [(10, 20), (20, 20)], threading.Event())

        move_to.assert_not_called()

    def test_old_config_pause_gets_reliable_runtime_minimum(self):
        config = DrawConfig(pen_down_pause=0.001)

        self.assertEqual(effective_pen_down_pause(config), 0.035)
        self.assertEqual(effective_pen_down_pause(config, is_raster_line=True), 0.025)


class PathRetraceTests(unittest.TestCase):
    def test_black_white_contour_has_independent_retrace_switch(self):
        enabled = DrawConfig(method="黑白轮廓(阈值)", contour_retrace=True, centerline_retrace=False)
        disabled = DrawConfig(method="黑白轮廓(阈值)", contour_retrace=False, centerline_retrace=True)

        self.assertTrue(should_retrace_path(enabled))
        self.assertFalse(should_retrace_path(disabled))

    def test_black_white_lineart_optimization_disables_retrace(self):
        config = DrawConfig(
            method="黑白轮廓(阈值)",
            contour_retrace=True,
            contour_lineart_optimize=True,
        )

        self.assertFalse(should_retrace_path(config))

    def test_canny_does_not_use_contour_retrace_switch(self):
        config = DrawConfig(method="边缘线稿(Canny)", contour_retrace=True)

        self.assertFalse(should_retrace_path(config))


class ReliableRasterDrawingTests(unittest.TestCase):
    def test_short_raster_line_is_confirmed_without_leaving_preview_segment(self):
        config = DrawConfig(raster_backtrack=False, mouse_step_px=3)
        self.assertTrue(raster_line_needs_confirmation(config, [(10, 20), (16, 20)]))
        self.assertFalse(raster_line_needs_confirmation(config, [(10, 20), (40, 20)]))

    @patch("heytea_cup_label_drawer.automation.sleep_interruptible")
    @patch("heytea_cup_label_drawer.automation.pyautogui.moveTo")
    def test_short_raster_segment_emits_multiple_move_events(self, move_to, _sleep):
        config = DrawConfig(mouse_step_px=3)

        draw_smooth_segment(config, 10, 20, 12, 20, threading.Event(), is_raster_line=True)

        self.assertGreaterEqual(move_to.call_count, 4)
        for call in move_to.call_args_list:
            x, y = call.args[:2]
            self.assertGreaterEqual(x, 10)
            self.assertLessEqual(x, 12)
            self.assertEqual(y, 20)

    @patch("heytea_cup_label_drawer.automation.sleep_interruptible")
    @patch("heytea_cup_label_drawer.automation.prime_pen_down")
    @patch("heytea_cup_label_drawer.automation.safe_move_to")
    @patch("heytea_cup_label_drawer.automation.draw_smooth_segment")
    @patch("heytea_cup_label_drawer.automation.pyautogui.mouseUp")
    @patch("heytea_cup_label_drawer.automation.pyautogui.mouseDown")
    @patch("heytea_cup_label_drawer.automation.pyautogui.position")
    def test_short_raster_line_automatically_backtracks(
        self,
        position,
        _mouse_down,
        _mouse_up,
        draw_segment,
        _safe_move,
        _prime,
        _sleep,
    ):
        config = DrawConfig(raster_backtrack=False, mouse_step_px=3)
        position.side_effect = [SimpleNamespace(x=10, y=20), SimpleNamespace(x=16, y=20)]

        draw_screen_polyline(config, [(10, 20), (16, 20)], threading.Event(), is_raster_line=True)

        self.assertEqual(draw_segment.call_count, 2)
        self.assertEqual(draw_segment.call_args_list[0].args[1:5], (10, 20, 16, 20))
        self.assertEqual(draw_segment.call_args_list[1].args[1:5], (16, 20, 10, 20))


if __name__ == "__main__":
    unittest.main()

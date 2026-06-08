import threading
import unittest
from unittest.mock import patch

from heytea_cup_label_drawer.automation import effective_pen_down_pause, prime_pen_down
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


if __name__ == "__main__":
    unittest.main()

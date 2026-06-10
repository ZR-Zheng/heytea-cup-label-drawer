import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from heytea_cup_label_drawer.main import configure_tcl_tk_paths


class TclTkPathTests(unittest.TestCase):
    def test_configures_paths_from_base_python_install(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            tcl_dir = root / "tcl" / "tcl8.6"
            tk_dir = root / "tcl" / "tk8.6"
            tcl_dir.mkdir(parents=True)
            tk_dir.mkdir(parents=True)
            (tcl_dir / "init.tcl").touch()
            (tk_dir / "tk.tcl").touch()

            with (
                patch("heytea_cup_label_drawer.main.sys.base_prefix", str(root)),
                patch("heytea_cup_label_drawer.main.sys.prefix", str(root / "venv")),
                patch("heytea_cup_label_drawer.main.sys.executable", str(root / "venv" / "python.exe")),
                patch.dict(os.environ, {}, clear=True),
            ):
                configure_tcl_tk_paths()
                self.assertEqual(os.environ["TCL_LIBRARY"], str(tcl_dir))
                self.assertEqual(os.environ["TK_LIBRARY"], str(tk_dir))

    def test_preserves_explicit_environment_paths(self):
        with patch.dict(os.environ, {"TCL_LIBRARY": "custom-tcl", "TK_LIBRARY": "custom-tk"}, clear=True):
            configure_tcl_tk_paths()
            self.assertEqual(os.environ["TCL_LIBRARY"], "custom-tcl")
            self.assertEqual(os.environ["TK_LIBRARY"], "custom-tk")


if __name__ == "__main__":
    unittest.main()

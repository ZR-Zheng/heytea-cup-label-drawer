"""应用入口。"""

import os
import sys
from pathlib import Path
import tkinter as tk
from tkinter import ttk

from .gui import HeyTeaCupLabelDrawerGUI


def configure_tcl_tk_paths():
    """修复部分 Windows 虚拟环境无法定位基础 Python Tcl/Tk 资源的问题。"""
    roots = {
        Path(sys.base_prefix),
        Path(sys.prefix),
        Path(sys.executable).resolve().parent,
    }
    for root in roots:
        tcl_dir = root / "tcl" / "tcl8.6"
        tk_dir = root / "tcl" / "tk8.6"
        if tcl_dir.joinpath("init.tcl").is_file() and tk_dir.joinpath("tk.tcl").is_file():
            os.environ.setdefault("TCL_LIBRARY", str(tcl_dir))
            os.environ.setdefault("TK_LIBRARY", str(tk_dir))
            return


def main():
    configure_tcl_tk_paths()
    root = tk.Tk()
    try:
        # Windows 默认主题更像原生应用。
        style = ttk.Style(root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
    except Exception:
        pass
    HeyTeaCupLabelDrawerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

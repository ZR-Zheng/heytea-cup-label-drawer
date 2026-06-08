"""应用入口。"""

import tkinter as tk
from tkinter import ttk

from .gui import HeyTeaCupLabelDrawerGUI


def main():
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

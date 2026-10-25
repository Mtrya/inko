"""Paper-and-ink plotting theme shared by the figure scripts."""

import matplotlib
from matplotlib import font_manager

PAPER = "#f4f0e8"
PANEL = "#fffcf7"
INK = "#322d28"
MUTED = "#766b60"
FAINT = "#94897b"
GRID = "#ded6c9"
VERMILION = "#a94f3c"
CYCLE = [
    "#a94f3c",
    "#44615a",
    "#5b6b8c",
    "#8a6d2f",
    "#7c4a63",
    "#3f6b6b",
    "#766b60",
]


def _inter_available():
    return any(entry.name == "Inter" for entry in font_manager.fontManager.ttflist)


def apply():
    rc = {
        "figure.facecolor": PAPER,
        "figure.edgecolor": PAPER,
        "savefig.facecolor": PAPER,
        "savefig.edgecolor": PAPER,
        "axes.facecolor": PANEL,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": MUTED,
        "ytick.labelcolor": MUTED,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "grid.alpha": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": matplotlib.cycler(color=CYCLE),
        "legend.edgecolor": GRID,
        "legend.labelcolor": INK,
    }
    if _inter_available():
        rc["font.family"] = ["Inter", "DejaVu Sans"]
    matplotlib.rcParams.update(rc)

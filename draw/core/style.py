"""
style.py
--------
CMS-style matplotlib configuration: global rcParams, axis labelling helpers,
and the CMS watermark label.
"""
from __future__ import annotations
from typing import Optional

import matplotlib.pyplot as plt

from core.config import (
    AXIS_TITLE_SIZE, AXIS_TICK_SIZE, PANEL_TITLE_SIZE,
)

try:
    import mplhep as hep
    HAS_MPLHEP = True
except ImportError:
    hep = None
    HAS_MPLHEP = False


def setup_cms_style() -> None:
    """Apply CMS-like plotting style globally."""
    if HAS_MPLHEP:
        plt.style.use(hep.style.CMS)
    else:
        plt.rcParams.update({
            "font.size": 14,
            "axes.labelsize": AXIS_TITLE_SIZE,
            "axes.titlesize": PANEL_TITLE_SIZE,
            "xtick.labelsize": AXIS_TICK_SIZE,
            "ytick.labelsize": AXIS_TICK_SIZE,
            "legend.fontsize": 11,
            "figure.titlesize": 18,
            "axes.linewidth": 1.2,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.size": 7,
            "ytick.major.size": 7,
            "xtick.minor.size": 4,
            "ytick.minor.size": 4,
        })


def add_cms_label(
    ax,
    label: str = "SPS2026June",
    extra: Optional[str] = None,
    fontsize: int = 16,
) -> None:
    """Draw a compact CMS-style watermark on an axis."""
    ax.text(
        0.00, 1.02, "CMS",
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=fontsize,
        fontweight="bold",
    )
    ax.text(
        0.14, 1.02, label,
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=max(fontsize - 2, 10),
        fontstyle="italic",
    )
    if extra:
        ax.text(
            1.00, 1.02, extra,
            transform=ax.transAxes,
            ha="right", va="bottom",
            fontsize=max(fontsize - 2, 10),
        )


def style_axis_labels(ax, xlabel: str, ylabel: str = "Entries") -> None:
    """Enlarge axis labels, right-align x title, top-align y title."""
    ax.set_xlabel(xlabel, fontsize=AXIS_TITLE_SIZE)
    ax.set_ylabel(ylabel, fontsize=AXIS_TITLE_SIZE)

    ax.xaxis.set_label_coords(1.0, -0.12)
    ax.xaxis.label.set_horizontalalignment("right")

    ax.yaxis.set_label_coords(-0.11, 1.02)
    ax.yaxis.label.set_verticalalignment("top")

    ax.tick_params(axis="both", which="major", labelsize=AXIS_TICK_SIZE)

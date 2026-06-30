"""
config.py
---------
Global constants: DUT labels, variable metadata, pair definitions, and plot style colors.
"""

# ── DUT Labels ──────────────────────────────────────────────────────────────
# DUT_LABELS = ["DUT1 (ch1)", "DUT2 (ch2)", "DUT3 (ch3)", "DUT4 (ch4)"]
# Batch1
DUT_LABELS = ["HPK 1 (ch1)", "HPK 3 (ch2)", "HPK 17 (ch3)", "HPK 18 (ch4)"]
DUT_SHORT  = ["DUT1", "DUT2", "DUT3", "DUT4"]

# ── Variable display info ────────────────────────────────────────────────────
VAR_INFO = {
    "pmax_fit": {"xlabel": "pmax_fit [mV]",             "positive": True,  "logy": True},
    "tmax_fit": {"xlabel": "tmax_fit [ns]",              "positive": False, "logy": True},
    "area_new": {"xlabel": r"area_new [mV$\cdot$ns]",   "positive": True,  "logy": False},
}

# ── DUT pair definitions ─────────────────────────────────────────────────────
# Each entry: (i, j, label)  →  ΔT = CFD(DUT_{i+1}) - CFD(DUT_{j+1})
PAIR_DEFS = [
    (1, 0, "21"),  # DUT2 - DUT1
    (2, 0, "31"),  # DUT3 - DUT1
    (3, 0, "41"),  # DUT4 - DUT1
    (2, 1, "32"),  # DUT3 - DUT2
    (3, 1, "42"),  # DUT4 - DUT2
    (3, 2, "43"),  # DUT4 - DUT3
]

# ── Plot style constants ─────────────────────────────────────────────────────
AXIS_TITLE_SIZE  = 17
AXIS_TICK_SIZE   = 13
PANEL_TITLE_SIZE = 17
STAT_TEXT_SIZE   = 15

FILL_COLOR = "#5DA5DA"   # blue fill
LINE_COLOR = "#4C78A8"   # dark blue edge
FIT_COLOR  = "#D62728"   # Gaussian / Langau fit red

PAIR_COLORS = {
    "21": "#4C78A8",  # blue
    "31": "#F58518",  # orange
    "41": "#54A24B",  # green
    "32": "#E45756",  # red
    "42": "#72B7B2",  # teal
    "43": "#B279A2",  # purple
}

# ── Area fit constants (Landau-Gaussian convolution) ────────────────────────
AREA_FIT_BINS      = 80
AREA_FIT_QUANTILES = (1.0, 97.0)
LANGAU_CONV_POINTS = 600

# ── Area fit constants (pure Landau / Moyal) ─────────────────────────────────
# Fit range: 1–99th percentile keeps both the rising edge and the long tail.
# Slightly wider than Langau because the pure Landau tail is heavier.
LANDAU_FIT_BINS      = 80
LANDAU_FIT_QUANTILES = (1.0, 99.0)

# ── Charge conversion ────────────────────────────────────────────────────────
CHARGE_CONVERSION  = 4.7   # collected charge [fC] = MPV / CHARGE_CONVERSION

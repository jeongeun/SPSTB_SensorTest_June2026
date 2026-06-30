import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep

# =========================
# CMS style
# =========================
plt.style.use(hep.style.CMS)

# =========================
# Input data: Beta test result
# =========================

# =========================
# FBK beta test data
# =========================

data_fbk = {
    "FBK240": {
        "full_name": "PRE_LF-FBK_QC-TS_W16_S13_LGAD-B",
        "BV":     [120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220],
        "TR":     [80.54, 75.30, 65.09, 57.98, 54.25, 49.31, 46.13, 40.47, 37.26, 30.83, 27.81],
        "TR_err": [2.39,  2.40,  2.27,  1.96,  1.87,  1.52,  0.88,  1.55,  0.87,  1.06,  1.03],
        "CQ":     [6.07, 6.88, 7.69, 8.85, 10.59, 12.27, 14.70, 18.40, 26.35, 36.02, 35.26],
        "CQ_err": [0.07, 0.09, 0.09, 0.10, 0.13, 0.12, 0.20, 0.20, 0.23, 0.30, 0.27],
    },

    "FBK242": {
        "full_name": "PRE_LF-FBK_QC-TS_W16_S52_LGAD-B",
        "BV":     [120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220],
        "TR":     [68.81, 60.36, 55.05, 51.79, 46.45, 45.93, 39.86, 37.18, 36.26, 31.34, 35.47],
        "TR_err": [1.65,  1.65,  1.59,  1.32,  1.09,  1.21,  1.26,  0.95,  1.04,  0.51,  0.81],
        "CQ":     [7.90, 8.87, 9.72, 11.22, 12.88, 14.96, 17.29, 20.61, 25.08, 31.01, 31.95],
        "CQ_err": [0.10, 0.09, 0.10, 0.10, 0.11, 0.12, 0.14, 0.16, 0.20, 0.25, 0.37],
    },
}

# =========================
# HPK beta test data
# =========================

data_hpk = {
    "HPK1": {
        "full_name": "PRE_HPK_QC-TS_W5_S1_LGAD-B",
        "BV":     [130, 140, 150, 160, 170, 180, 190, 200],
        "TR":     [49.30, 46.39, 42.31, 38.12, 35.29, 32.91, 29.30, 25.63],
        "TR_err": [0.70,  0.48,  0.62,  0.60,  0.48,  0.34,  0.60,  0.52],
        "CQ":     [9.20, 10.67, 12.37, 14.52, 17.21, 21.24, 27.29, 37.99],
        "CQ_err": [0.06, 0.06, 0.06, 0.07, 0.07, 0.09, 0.11, 0.18],
    },

    "HPK3": {
        "full_name": "PRE_HPK_QC-TS_W81_S1_LGAD-B",
        "BV":     [120, 130, 140, 150, 160, 170, 180, 190, 200],
        "TR":     [53.34, 47.53, 42.86, 39.10, 38.11, 33.56, 31.59, 28.60, 27.87],
        "TR_err": [0.71,  1.18,  0.97,  1.10,  0.68,  0.73,  0.38,  0.80,  0.44],
        "CQ":     [8.32, 9.56, 11.15, 12.71, 15.43, 18.42, 23.16, 29.84, 45.47],
        "CQ_err": [0.07, 0.07, 0.08, 0.08, 0.10, 0.12, 0.14, 0.20, 0.35],
    },

    "HPK10": {
        "full_name": "PRE_HPK_QC-TS_W5_S2_LGAD-B",
        "BV":     [120, 130, 140, 150, 160, 170, 180, 190, 200, 210],
        "TR":     [54.30, 50.53, 46.44, 43.49, 39.90, 36.99, 33.03, 30.99, 28.95, 27.11],
        "TR_err": [0.90,  1.09,  0.56,  0.77,  0.60,  0.70,  0.61,  0.58,  0.48,  0.61],
        "CQ":     [8.88, 10.04, 11.51, 13.14, 15.54, 18.37, 22.76, 27.96, 36.51, 54.37],
        "CQ_err": [0.07, 0.07, 0.08, 0.08, 0.09, 0.11, 0.14, 0.16, 0.19, 0.30],
    },

    "HPK16 (low stat.)": {
        "full_name": "PRE_HPK_QC-TS_W81_S2_LGAD-B",
        "BV":     [120, 130, 140, 150, 160, 170, 180, 190, 200],
        "TR":     [53.12, 49.11, 42.73, 39.40, 35.89, 35.10, 36.23, 32.15, 30.87],
        "TR_err": [3.68,  4.80,  2.68,  3.30,  3.56,  2.80,  3.21,  2.54,  2.36],
        "CQ":     [9.48, 10.33, 12.30, 13.48, 15.80, 19.49, 23.46, 33.87, 44.15],
        "CQ_err": [0.27, 0.29, 0.20, 0.55, 0.40, 0.49, 0.52, 0.11, 1.06],
    },
}

# =========================
# HPK TB data
# =========================

# =========================
# FBK TB data
# =========================


# =========================
# Common CMS label
# =========================

def add_cms_label(ax):
    # rlabel="" removes the default fb^{-1} / TeV label
    hep.cms.label(
        "ETL Preliminary",
        data=True,
        ax=ax,
        loc=0,
        fontsize=16,
        rlabel="",
    )

    ax.text(
        0.03, 0.95,
        "Sensor Beta test results ",
        transform=ax.transAxes,
        fontsize=17,
        ha="left",
        va="top",
    )

    ax.text(
        0.97, 1.010,
        r"Torino (Temp = $-25^{\circ}$C)",
        transform=ax.transAxes,
        fontsize=17,
        ha="right",
        #va="bottom",
    )

# =========================
# Utility for BV axis
# =========================

def get_bv_ticks(data):
    all_bv = []
    for sensor, values in data.items():
        all_bv.extend(values["BV"])
    all_bv = np.array(all_bv)

    bv_min = np.min(all_bv)
    bv_max = np.max(all_bv)

    tick_start = int(np.floor(bv_min / 10.0) * 10)
    tick_end   = int(np.ceil(bv_max / 10.0) * 10)

    # 20 V spacing looks clean
    ticks = np.arange(tick_start, tick_end + 1, 20)
    return bv_min, bv_max, ticks

# =========================
# BV vs quantity plot
# =========================
def make_cms_plot_vs_bv(
    data,
    left_text,
    right_text,
    y_key,
    yerr_key,
    ylabel,
    output_name,
    ylim=None,
    legend_loc="lower left",
    legend_ncol=2,
):
    fig, ax = plt.subplots(figsize=(8, 7))

    for sensor, values in data.items():
        bv = np.array(values["BV"])
        y = np.array(values[y_key])
        yerr = np.array(values[yerr_key])

        idx = np.argsort(bv)

        ax.errorbar(
            bv[idx],
            y[idx],
            yerr=yerr[idx],
            marker="o",
            markersize=7,
            linewidth=2,
            elinewidth=1.5,
            capsize=3,
            label=sensor,
        )

    bv_min, bv_max, ticks = get_bv_ticks(data)

    ax.set_xlabel("Bias voltage [V]")
    ax.set_ylabel(ylabel)
    ax.set_xlim(bv_min - 5, bv_max + 5)
    ax.set_xticks(ticks)

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.4)

    ax.legend(
        loc=legend_loc,
        ncol=legend_ncol,
        frameon=False,
        fontsize=13,
        columnspacing=1.0,
        handlelength=1.6,
        handletextpad=0.5,
    )

    add_cms_label(ax, left_text, right_text)

    fig.tight_layout()
    fig.savefig(output_name + ".pdf", bbox_inches="tight")
    fig.savefig(output_name + ".png", dpi=300, bbox_inches="tight")
    plt.show()

# =========================
# Collected charge vs Timing resolution plot
# =========================

def make_cms_plot_vs_bv(
    data,
    left_text,
    right_text,
    y_key,
    yerr_key,
    ylabel,
    output_name,
    ylim=None,
    legend_loc="lower left",
    legend_ncol=2,
):
    fig, ax = plt.subplots(figsize=(8, 7))

    for sensor, values in data.items():
        bv = np.array(values["BV"])
        y = np.array(values[y_key])
        yerr = np.array(values[yerr_key])

        idx = np.argsort(bv)

        ax.errorbar(
            bv[idx],
            y[idx],
            yerr=yerr[idx],
            marker="o",
            markersize=7,
            linewidth=2,
            elinewidth=1.5,
            capsize=3,
            label=sensor,
        )

    bv_min, bv_max, ticks = get_bv_ticks(data)

    ax.set_xlabel("Bias voltage [V]")
    ax.set_ylabel(ylabel)
    ax.set_xlim(bv_min - 5, bv_max + 5)
    ax.set_xticks(ticks)

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.4)

    ax.legend(
        loc=legend_loc,
        ncol=legend_ncol,
        frameon=False,
        fontsize=13,
        columnspacing=1.0,
        handlelength=1.6,
        handletextpad=0.5,
    )

    add_cms_label(ax, left_text, right_text)

    fig.tight_layout()
    fig.savefig(output_name + ".pdf", bbox_inches="tight")
    fig.savefig(output_name + ".png", dpi=300, bbox_inches="tight")
    plt.show()

# =========================
# Make plots
# =========================

# 1. Bias voltage vs collected charge
make_cms_plot_vs_bv(
    y_key="CQ",
    yerr_key="CQ_err",
    ylabel="Collected charge [fC]",
    output_name="BetaTest_FBK_Batch1_CollectedCharge_vs_BV",
    ylim=(0, 50),
    legend_loc="lower right",
)

# 2. Bias voltage vs timing resolution
make_cms_plot_vs_bv(
    y_key="TR",
    yerr_key="TR_err",
    ylabel="Timing resolution [ps]",
    output_name="BetaTest_FBK_Batch1_TimingResolution_vs_BV",
    ylim=(20, 90),
    legend_loc="lower left",
)

# 3. Collected charge vs timing resolution
make_cms_plot_charge_vs_timing(
    output_name="BetaTest_FBK_Batch1_TimingResolution_vs_CollectedCharge",
    legend_loc="lower left",
)

import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep

# =========================
# CMS style
# =========================
plt.style.use(hep.style.CMS)

# ============================================================
# Color map: same DUT -> same color
# ============================================================

color_map = {
    "FBK-LF240": "red",
    "FBK-LF242": "orange",
    "HPK1": "blue",
    "HPK3": "green",
    "HPK10": "violet",
    "HPK16": "darkblue",
    "HPK14": "brown",
    "HPK17": "yellow",
    "HPK18": "skyblue",
}

linestyle_map = {
    "beta": "--",
    "tb": "-",
}

# ============================================================
# Beta data : FBK-LF
# ============================================================

data_beta_fbk = {
    "FBK-LF240 Beta setup": {
        "sensor": "FBK-LF240",
        "setup": "beta",
        "full_name": "PRE_LF-FBK-LF_QC-TS_W16_S13_LGAD-B",
        "BV":     [120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220],
        "TR":     [80.54, 75.30, 65.09, 57.98, 54.25, 49.31, 46.13, 40.47, 37.26, 30.83, 27.81],
        "TR_err": [2.39,  2.40,  2.27,  1.96,  1.87,  1.52,  0.88,  1.55,  0.87,  1.06,  1.03],
        "CQ":     [6.07, 6.88, 7.69, 8.85, 10.59, 12.27, 14.70, 18.40, 26.35, 36.02, 35.26],
        "CQ_err": [0.07, 0.09, 0.09, 0.10, 0.13, 0.12, 0.20, 0.20, 0.23, 0.30, 0.27],
    },

    "FBK-LF242 Beta setup": {
        "sensor": "FBK-LF242",
        "setup": "beta",
        "full_name": "PRE_LF-FBK-LF_QC-TS_W16_S52_LGAD-B",
        "BV":     [120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220],
        "TR":     [68.81, 60.36, 55.05, 51.79, 46.45, 45.93, 39.86, 37.18, 36.26, 31.34, 35.47],
        "TR_err": [1.65,  1.65,  1.59,  1.32,  1.09,  1.21,  1.26,  0.95,  1.04,  0.51,  0.81],
        "CQ":     [7.90, 8.87, 9.72, 11.22, 12.88, 14.96, 17.29, 20.61, 25.08, 31.01, 31.95],
        "CQ_err": [0.10, 0.09, 0.10, 0.10, 0.11, 0.12, 0.14, 0.16, 0.20, 0.25, 0.37],
    },
}

# ============================================================
# Beta data : HPK
# ============================================================

data_beta_hpk = {
    "HPK1 Beta setup": {
        "sensor": "HPK1",
        "setup": "beta",
        "full_name": "PRE_HPK_QC-TS_W5_S1_LGAD-B",
        "BV":     [130, 140, 150, 160, 170, 180, 190, 200],
        "TR":     [49.30, 46.39, 42.31, 38.12, 35.29, 32.91, 29.30, 25.63],
        "TR_err": [0.70,  0.48,  0.62,  0.60,  0.48,  0.34,  0.60,  0.52],
        "CQ":     [9.20, 10.67, 12.37, 14.52, 17.21, 21.24, 27.29, 37.99],
        "CQ_err": [0.06, 0.06, 0.06, 0.07, 0.07, 0.09, 0.11, 0.18],
    },

    "HPK3 Beta setup": {
        "sensor": "HPK3",
        "setup": "beta",
        "full_name": "PRE_HPK_QC-TS_W81_S1_LGAD-B",
        "BV":     [120, 130, 140, 150, 160, 170, 180, 190, 200],
        "TR":     [53.34, 47.53, 42.86, 39.10, 38.11, 33.56, 31.59, 28.60, 27.87],
        "TR_err": [0.71,  1.18,  0.97,  1.10,  0.68,  0.73,  0.38,  0.80,  0.44],
        "CQ":     [8.32, 9.56, 11.15, 12.71, 15.43, 18.42, 23.16, 29.84, 45.47],
        "CQ_err": [0.07, 0.07, 0.08, 0.08, 0.10, 0.12, 0.14, 0.20, 0.35],
    },

    "HPK10 Beta setup": {
        "sensor": "HPK10",
        "setup": "beta",
        "full_name": "PRE_HPK_QC-TS_W5_S2_LGAD-B",
        "BV":     [120, 130, 140, 150, 160, 170, 180, 190, 200, 210],
        "TR":     [54.30, 50.53, 46.44, 43.49, 39.90, 36.99, 33.03, 30.99, 28.95, 27.11],
        "TR_err": [0.90,  1.09,  0.56,  0.77,  0.60,  0.70,  0.61,  0.58,  0.48,  0.61],
        "CQ":     [8.88, 10.04, 11.51, 13.14, 15.54, 18.37, 22.76, 27.96, 36.51, 54.37],
        "CQ_err": [0.07, 0.07, 0.08, 0.08, 0.09, 0.11, 0.14, 0.16, 0.19, 0.30],
    },

    "HPK16 Beta setup": {
        "sensor": "HPK16",
        "setup": "beta",
        "full_name": "PRE_HPK_QC-TS_W81_S2_LGAD-B",
        "BV":     [120, 130, 140, 150, 160, 170, 180, 190, 200],
        "TR":     [53.12, 49.11, 42.73, 39.40, 35.89, 35.10, 36.23, 32.15, 30.87],
        "TR_err": [3.68,  4.80,  2.68,  3.30,  3.56,  2.80,  3.21,  2.54,  2.36],
        "CQ":     [9.48, 10.33, 12.30, 13.48, 15.80, 19.49, 23.46, 33.87, 44.15],
        "CQ_err": [0.27, 0.29, 0.20, 0.55, 0.40, 0.49, 0.52, 0.11, 1.06],
    },
}

# ============================================================
# Test beam data : HPK batch1
# ============================================================

BV_batch1 = np.array([200, 195, 190, 180, 170, 160, 150, 140, 120])

data_tb_hpk_batch1 = {
    "HPK1 TB (SPS2026)": {
        "sensor": "HPK1",
        "setup": "tb",
        "BV":     BV_batch1.tolist(),
        "CQ":     [26.639, 23.626, 21.476, 17.763, 15.032, 13.008, 11.343, 9.983, 7.979],
        "CQ_err": [0.078, 0.068, 0.070, 0.060, 0.058, 0.054, 0.049, 0.043, 0.038],
        "TR":     [26.745, 26.056, 27.467, 30.928, 35.218, 32.550, 35.440, 40.887, 47.914],
        "TR_err": [1.193, 1.060, 1.123, 1.093, 1.138, 1.452, 1.590, 1.441, 1.795],
    },
    "HPK3 TB (SPS2026)": {
        "sensor": "HPK3",
        "setup": "tb",
        "BV":     BV_batch1.tolist(),
        "CQ":     [25.206, 22.622, 20.588, 17.319, 14.519, 12.713, 11.071, 9.744, 7.983],
        "CQ_err": [0.092, 0.084, 0.079, 0.074, 0.066, 0.066, 0.055, 0.052, 0.045],
        "TR":     [29.212, 28.441, 30.380, 32.724, 33.209, 39.559, 38.112, 39.872, 48.378],
        "TR_err": [1.043, 1.040, 1.060, 1.033, 1.145, 1.319, 1.468, 1.456, 1.682],
    },
    "HPK17 TB (SPS2026)": {
        "sensor": "HPK17",
        "setup": "tb",
        "BV":     BV_batch1.tolist(),
        "CQ":     [27.831, 24.830, 22.412, 18.507, 15.614, 13.384, 11.700, 10.178, 8.273],
        "CQ_err": [0.098, 0.085, 0.075, 0.071, 0.067, 0.061, 0.065, 0.050, 0.056],
        "TR":     [28.907, 29.927, 31.396, 33.597, 34.465, 37.213, 43.461, 46.912, 52.827],
        "TR_err": [0.960, 0.897, 0.921, 0.920, 1.031, 1.225, 1.242, 1.204, 1.485],
    },
    "HPK18 TB (SPS2026)": {
        "sensor": "HPK18",
        "setup": "tb",
        "BV":     BV_batch1.tolist(),
        "CQ":     [41.647, 30.940, 26.736, 21.641, 18.177, 15.529, 13.463, 11.863, 9.282],
        "CQ_err": [0.397, 0.122, 0.098, 0.089, 0.077, 0.074, 0.068, 0.060, 0.058],
        "TR":     [28.741, 28.510, 28.925, 29.071, 30.280, 36.888, 41.246, 37.681, 43.496],
        "TR_err": [1.264, 1.170, 1.230, 1.281, 1.414, 1.565, 1.602, 1.772, 2.176],
    },
}

# ============================================================
# Test beam data : HPK batch2
# ============================================================

BV_batch2 = np.array([200, 195, 190, 185, 180, 175, 170, 165,
                      160, 155, 150, 145, 140, 135, 130, 120])

data_tb_hpk_batch2 = {
    "HPK10 TB (SPS2026)": {
        "sensor": "HPK10",
        "setup": "tb",
        "BV":     BV_batch2.tolist(),
        "CQ":     [29.202, 26.360, 24.082, 21.987, 20.263, 18.708, 17.330, 16.138,
                   15.016, 14.053, 13.180, 12.447, 11.657, 10.980, 10.441, 9.366],
        "CQ_err": [0.033, 0.031, 0.030, 0.029, 0.027, 0.027, 0.025, 0.024,
                   0.023, 0.022, 0.047, 0.043, 0.020, 0.019, 0.037, 0.034],
        "TR":     [30.466, 31.571, 32.380, 33.865, 34.446, 35.584, 37.020, 38.281,
                   39.259, 41.374, 42.094, 43.643, 44.963, 47.006, 48.313, 53.351],
        "TR_err": [0.298, 0.301, 0.301, 0.312, 0.315, 0.324, 0.329, 0.341,
                   0.352, 0.358, 0.371, 0.386, 0.392, 0.418, 0.411, 0.461],
    },

    "HPK14 TB (SPS2026)": {
        "sensor": "HPK14",
        "setup": "tb",
        "BV":     BV_batch2.tolist(),
        "CQ":     [28.192, 24.992, 22.655, 20.627, 18.780, 17.306, 16.016, 14.924,
                   13.839, 12.969, 12.123, 11.362, 10.691, 10.020, 9.453, 8.484],
        "CQ_err": [0.090, 0.089, 0.082, 0.076, 0.075, 0.075, 0.069, 0.067,
                   0.065, 0.067, 0.059, 0.123, 0.056, 0.051, 0.109, 0.049],
        "TR":     [29.450, 29.911, 32.366, 33.714, 34.169, 34.254, 37.901, 38.816,
                   40.212, 40.145, 42.705, 45.201, 45.018, 49.362, 50.622, 53.626],
        "TR_err": [0.731, 0.749, 0.773, 0.818, 0.834, 0.816, 0.838, 0.848,
                   0.864, 0.959, 0.953, 0.968, 1.000, 1.051, 1.043, 1.184],
    },

    "HPK15 TB (SPS2026)": {
        "sensor": "HPK15",
        "setup": "tb",
        "BV":     BV_batch2.tolist(),
        "CQ":     [30.695, 26.512, 23.692, 21.479, 19.548, 17.955, 16.524, 15.210,
                   14.152, 13.133, 12.298, 11.450, 10.829, 10.118, 9.632, 8.620],
        "CQ_err": [0.035, 0.029, 0.028, 0.027, 0.026, 0.025, 0.024, 0.023,
                   0.022, 0.021, 0.020, 0.019, 0.041, 0.018, 0.040, 0.033],
        "TR":     [27.425, 27.955, 29.684, 31.012, 32.769, 33.031, 34.788, 35.412,
                   37.112, 38.441, 40.388, 41.111, 42.845, 44.561, 45.200, 48.342],
        "TR_err": [0.300, 0.304, 0.303, 0.311, 0.311, 0.324, 0.325, 0.338,
                   0.344, 0.352, 0.359, 0.369, 0.376, 0.398, 0.405, 0.447],
    },

    "HPK16 TB (SPS2026)": {
        "sensor": "HPK16",
        "setup": "tb",
        "BV":     BV_batch2.tolist(),
        "CQ":     [32.497, 28.963, 26.159, 23.751, 21.736, 20.061, 18.383, 17.152,
                   15.862, 14.777, 13.816, 13.130, 12.178, 11.513, 10.889, 9.631],
        "CQ_err": [0.043, 0.039, 0.037, 0.036, 0.035, 0.033, 0.031, 0.030,
                   0.029, 0.027, 0.026, 0.059, 0.057, 0.050, 0.049, 0.021],
        "TR":     [30.607, 31.612, 32.622, 34.111, 34.697, 36.382, 37.646, 39.424,
                   40.682, 42.089, 43.976, 44.983, 45.416, 48.254, 50.351, 54.849],
        "TR_err": [0.313, 0.312, 0.310, 0.320, 0.323, 0.327, 0.333, 0.341,
                   0.351, 0.361, 0.367, 0.384, 0.396, 0.416, 0.405, 0.456],
    },
}

# ============================================================
# Test beam data : FBK-LF
# ============================================================

BV_batch3 = np.array([220, 210, 205, 200, 195, 190, 185, 180,
                      175, 170, 165, 160, 155, 150, 145, 140])

data_tb_fbk = {
    "FBK-LF240 TB (SPS2026)": {
        "sensor": "FBK-LF240",
        "setup": "tb",
        "BV":     BV_batch3.tolist(),
        "CQ":     [33.27, 26.66, 24.50, 22.60, 20.82, 19.32, 18.01, 16.79,
                   15.73, 14.71, 13.79, 13.06, 12.30, 11.67, 11.10, 10.48],
        "CQ_err": [0.08, 0.07, 0.06, 0.06, 0.06, 0.05, 0.06, 0.05,
                   0.05, 0.05, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04],
        "TR":     [32.67, 36.67, 36.85, 38.60, 37.54, 40.73, 39.05, 42.68,
                   42.69, 43.98, 47.77, 45.41, 48.54, 47.97, 48.92, 54.13],
        "TR_err": [0.70, 0.66, 0.65, 0.68, 0.67, 0.71, 0.66, 0.67,
                   0.71, 0.69, 0.77, 0.71, 0.74, 0.76, 0.78, 0.83],
    },

    "FBK-LF242 TB (SPS2026)": {
        "sensor": "FBK-LF242",
        "setup": "tb",
        "BV":     BV_batch3.tolist(),
        "CQ":     [30.48, 24.88, 22.79, 21.02, 19.46, 13.79, 16.81, 15.84,
                   14.79, 13.83, 10.89, 9.40, 8.23, 11.02, 7.42, 7.08],
        "CQ_err": [0.06, 0.05, 0.05, 0.05, 0.04, 0.07, 0.04, 0.04,
                   0.04, 0.03, 0.03, 0.04, 0.02, 0.03, 0.02, 0.02],
        "TR":     [32.53, 33.14, 36.20, 37.53, 38.57, 44.14, 41.81, 41.80,
                   45.01, 45.86, 48.16, 54.99, 53.60, 52.11, 57.61, 59.87],
        "TR_err": [0.54, 0.56, 0.52, 0.54, 0.52, 0.54, 0.52, 0.55,
                   0.57, 0.55, 0.59, 0.54, 0.57, 0.59, 0.59, 0.63],
    },
}

# ============================================================
# Combine dictionaries
# ============================================================

data_all = {}
data_all.update(data_beta_fbk)
data_all.update(data_beta_hpk)
data_all.update(data_tb_hpk_batch1)
data_all.update(data_tb_hpk_batch2)
data_all.update(data_tb_fbk)

# ============================================================
# CMS label helper
# ============================================================

def add_cms_label(ax, left_text="", right_text=""):
    hep.cms.label(
        "ETL Preliminary",
        data=True,
        ax=ax,
        loc=0,
        fontsize=16,
        rlabel="",
    )

    if left_text:
        ax.text(
            0.03, 0.95,
            left_text,
            transform=ax.transAxes,
            fontsize=17,
            ha="left",
            va="top",
        )

    if right_text:
        ax.text(
            0.97, 1.010,
            right_text,
            transform=ax.transAxes,
            fontsize=17,
            ha="right",
            #va="top",
        )

# ============================================================
# Utility
# ============================================================

def filter_data_by_sensors(data, sensor_list):
    out = {}
    for key, values in data.items():
        if values["sensor"] in sensor_list:
            out[key] = values
    return out

def get_bv_range_and_ticks(data):
    all_bv = []
    for _, values in data.items():
        all_bv.extend(values["BV"])
    all_bv = np.array(all_bv)

    bv_min = np.min(all_bv)
    bv_max = np.max(all_bv)

    tick_start = int(np.floor(bv_min / 10.0) * 10)
    tick_end   = int(np.ceil(bv_max / 10.0) * 10)

    ticks = np.arange(tick_start, tick_end + 1, 20)
    return bv_min, bv_max, ticks

# ============================================================
# Generic plot: y vs BV
# ============================================================

def plot_vs_bv(data, y_key, yerr_key, ylabel, output_name,
               left_text="", right_text="",
               ylim=None, legend_loc="best", legend_ncol=2, legend_font=15):

    fig, ax = plt.subplots(figsize=(9, 7))

    for label, values in data.items():
        sensor = values["sensor"]
        setup  = values["setup"]

        bv = np.array(values["BV"])
        y = np.array(values[y_key])
        yerr = np.array(values[yerr_key])

        idx = np.argsort(bv)

        ax.errorbar(
            bv[idx],
            y[idx],
            yerr=yerr[idx],
            marker="o",
            markersize=6,
            linewidth=2.0,
            elinewidth=1.2,
            capsize=3,
            color=color_map[sensor],
            linestyle=linestyle_map[setup],
            label=label,
        )

    bv_min, bv_max, ticks = get_bv_range_and_ticks(data)

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
        fontsize=legend_font,
        columnspacing=1.0,
        handlelength=2.0,
        handletextpad=0.5,
    )

    add_cms_label(ax, left_text=left_text, right_text=right_text)

    fig.tight_layout()
    fig.savefig(output_name + ".pdf", bbox_inches="tight")
    fig.savefig(output_name + ".png", dpi=300, bbox_inches="tight")
    plt.show()

# ============================================================
# Generic plot: TR vs CQ
# ============================================================

def plot_tr_vs_cq(data, output_name,
                  left_text="", right_text="",
                  xlim=None, ylim=None,
                  legend_loc="best", legend_ncol=2, legend_font=15):

    fig, ax = plt.subplots(figsize=(9, 7))

    all_cq = []
    all_tr = []

    for label, values in data.items():
        sensor = values["sensor"]
        setup  = values["setup"]

        cq = np.array(values["CQ"])
        cq_err = np.array(values["CQ_err"])
        tr = np.array(values["TR"])
        tr_err = np.array(values["TR_err"])

        idx = np.argsort(cq)

        ax.errorbar(
            cq[idx],
            tr[idx],
            xerr=cq_err[idx],
            yerr=tr_err[idx],
            marker="o",
            markersize=6,
            linewidth=2.0,
            elinewidth=1.2,
            capsize=3,
            color=color_map[sensor],
            linestyle=linestyle_map[setup],
            label=label,
        )

        all_cq.extend(cq.tolist())
        all_tr.extend(tr.tolist())

    all_cq = np.array(all_cq)
    all_tr = np.array(all_tr)

    ax.set_xlabel("Collected charge [fC]")
    ax.set_ylabel("Timing resolution [ps]")

    if xlim is None:
        ax.set_xlim(max(0, np.min(all_cq) - 2), np.max(all_cq) + 4)
    else:
        ax.set_xlim(*xlim)

    if ylim is None:
        ax.set_ylim(0, np.max(all_tr) + 10)
    else:
        ax.set_ylim(*ylim)

    ax.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.4)

    ax.legend(
        loc=legend_loc,
        ncol=legend_ncol,
        frameon=False,
        fontsize=legend_font,
        columnspacing=1.0,
        handlelength=2.0,
        handletextpad=0.5,
    )

    add_cms_label(ax, left_text=left_text, right_text=right_text)

    fig.tight_layout()
    fig.savefig(output_name + ".pdf", bbox_inches="tight")
    fig.savefig(output_name + ".png", dpi=300, bbox_inches="tight")
    plt.show()

# ============================================================
# Sensor groups
# ============================================================

fbk_sensors = ["FBK-LF240", "FBK-LF242"]
hpk_sensors = ["HPK1", "HPK3", "HPK10", "HPK14", "HPK16", "HPK17", "HPK18"]
hpk_sensors_comp = ["HPK1", "HPK3", "HPK10", "HPK16"]
fbkhpk_sensors = ["FBK-LF240", "FBK-LF242", "HPK1", "HPK3", "HPK10", "HPK16"]

fbk_data = filter_data_by_sensors(data_all, fbk_sensors)
hpk_data = filter_data_by_sensors(data_all, hpk_sensors)
hpkcomp_data = filter_data_by_sensors(data_all, hpk_sensors_comp)

comp_data = filter_data_by_sensors(data_all, fbkhpk_sensors)
# ============================================================
# Plot 1
# Collected charge vs Bias Voltage (FBK-LF)
# ============================================================

plot_vs_bv(
    #data=fbk_data,
    data=comp_data,
    y_key="CQ",
    yerr_key="CQ_err",
    ylabel="Collected charge [fC]",
    output_name="Plot1_FBK-LF_CollectedCharge_vs_BV",
    left_text=r"Beta setup ($-25^{\circ}$C) vs. SPS H6 Sensor test beam ($-20.5^{\circ}$C)",
    right_text="FBK-LF Comparison (June 2026)",
    #ylim=(0, 40),
    ylim=(0, 50),
    #legend_loc="lower right",
    legend_loc="center left",
    legend_ncol=2,
    legend_font=12,
)

## ============================================================
## Plot 2
## Collected charge vs Bias Voltage (HPK)
## ============================================================
#
#plot_vs_bv(
#    #data=hpk_data,
#    data=hpkcomp_data,
#    y_key="CQ",
#    yerr_key="CQ_err",
#    ylabel="Collected charge [fC]",
#    output_name="Plot2_HPK_CollectedCharge_vs_BV",
#    left_text=r"Beta setup ($-25^{\circ}$C) vs. SPS H6 Sensor test beam ($-20.5^{\circ}$C)",
#    right_text="HPK Comparison (June 2026)",
#    ylim=(0, 70),
#    legend_loc="center left",
#    legend_ncol=2,
#    legend_font=14,
#)

# ============================================================
# Plot 3
# Timing resolution vs Bias Voltage (FBK-LF)
# ============================================================

plot_vs_bv(
    data=fbk_data,
    #data=comp_data,
    y_key="TR",
    yerr_key="TR_err",
    ylabel="Timing resolution [ps]",
    output_name="Plot3_FBK-LF_TimingResolution_vs_BV",
    left_text=r"Beta setup ($-25^{\circ}$C) vs. SPS H6 Sensor test beam ($-20.5^{\circ}$C)",
    right_text="FBK-LF Comparison (June 2026)",
    ylim=(0, 100),
    legend_loc="lower left",
    legend_ncol=2,
    legend_font=15,
    #legend_ncol=3,
    #legend_font=13,
)

## ============================================================
## Plot 4
## Timing resolution vs Bias Voltage (HPK)
## ============================================================
#
#plot_vs_bv(
#    #data=hpk_data,
#    data=hpkcomp_data,
#    y_key="TR",
#    yerr_key="TR_err",
#    ylabel="Timing resolution [ps]",
#    output_name="Plot4_HPK_TimingResolution_vs_BV",
#    left_text=r"Beta setup ($-25^{\circ}$C) vs. SPS H6 Sensor test beam ($-20.5^{\circ}$C)",
#    right_text="HPK Comparison (June 2026)",
#    ylim=(0, 70),
#    legend_loc="lower left",
#    legend_ncol=2,
#    legend_font=14,
#)

# ============================================================
# Plot 5
# Timing resolution vs Collected charge (FBK-LF)
# ============================================================

plot_tr_vs_cq(
    data=fbk_data,
    #data=comp_data,
    output_name="Plot5_FBK-LF_TimingResolution_vs_CollectedCharge",
    left_text=r"Beta setup ($-25^{\circ}$C) vs. SPS H6 Sensor test beam ($-20.5^{\circ}$C)",
    right_text="FBK-LF Comparison (June 2026)",
    xlim=(0, 40),
    ylim=(0, 100),
    legend_loc="lower left",
    legend_ncol=2,
    legend_font=15,
    #legend_ncol=3,
    #legend_font=12,
)

## ============================================================
## Plot 6
## Timing resolution vs Collected charge (HPK)
## ============================================================
#
#plot_tr_vs_cq(
#    #data=hpk_data,
#    data=hpkcomp_data,
#    output_name="Plot6_HPK_TimingResolution_vs_CollectedCharge",
#    left_text=r"Beta setup ($-25^{\circ}$C) vs. SPS H6 Sensor test beam ($-20.5^{\circ}$C)",
#    right_text="HPK Comparison (June 2026)",
#    xlim=(0, 60),
#    ylim=(0, 70),
#    legend_loc="lower left",
#    #legend_ncol=3,
#    #legend_font=13,
#    legend_ncol=2,
#    legend_font=13,
#)

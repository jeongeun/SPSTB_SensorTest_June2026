import numpy as np
import matplotlib.pyplot as plt
import mplhep as hep

# =========================
# CMS style
# =========================
plt.style.use(hep.style.CMS)

# =========================
# Input data_1
# =========================

BV_batch1 = np.array([200, 195, 190, 180, 170, 160, 150, 140, 120])

data_1 = {
    "HPK1": {
        "CQ":     [26.639, 23.626, 21.476, 17.763, 15.032, 13.008, 11.343, 9.983, 7.979],
        "CQ_err": [0.078, 0.068, 0.070, 0.060, 0.058, 0.054, 0.049, 0.043, 0.038],
        "TR":     [26.745, 26.056, 27.467, 30.928, 35.218, 32.550, 35.440, 40.887, 47.914],
        "TR_err": [1.193, 1.060, 1.123, 1.093, 1.138, 1.452, 1.590, 1.441, 1.795],
    },
    "HPK3": {
        "CQ":     [25.206, 22.622, 20.588, 17.319, 14.519, 12.713, 11.071, 9.744, 7.983],
        "CQ_err": [0.092, 0.084, 0.079, 0.074, 0.066, 0.066, 0.055, 0.052, 0.045],
        "TR":     [29.212, 28.441, 30.380, 32.724, 33.209, 39.559, 38.112, 39.872, 48.378],
        "TR_err": [1.043, 1.040, 1.060, 1.033, 1.145, 1.319, 1.468, 1.456, 1.682],
    },
    "HPK17": {
        "CQ":     [27.831, 24.830, 22.412, 18.507, 15.614, 13.384, 11.700, 10.178, 8.273],
        "CQ_err": [0.098, 0.085, 0.075, 0.071, 0.067, 0.061, 0.065, 0.050, 0.056],
        "TR":     [28.907, 29.927, 31.396, 33.597, 34.465, 37.213, 43.461, 46.912, 52.827],
        "TR_err": [0.960, 0.897, 0.921, 0.920, 1.031, 1.225, 1.242, 1.204, 1.485],
    },
    "HPK18": {
        "CQ":     [41.647, 30.940, 26.736, 21.641, 18.177, 15.529, 13.463, 11.863, 9.282],
        "CQ_err": [0.397, 0.122, 0.098, 0.089, 0.077, 0.074, 0.068, 0.060, 0.058],
        "TR":     [28.741, 28.510, 28.925, 29.071, 30.280, 36.888, 41.246, 37.681, 43.496],
        "TR_err": [1.264, 1.170, 1.230, 1.281, 1.414, 1.565, 1.602, 1.772, 2.176],
    },
}

# =========================
# Input data_2
# =========================

BV_batch2 = np.array([200, 195, 190, 185, 180, 175, 170, 165,
                      160, 155, 150, 145, 140, 135, 130, 120])

data_2 = {
    "HPK10": {
        "CQ":     [29.202, 26.360, 24.082, 21.987, 20.263, 18.708, 17.330, 16.138,
                   15.016, 14.053, 13.180, 12.447, 11.657, 10.980, 10.441, 9.366],
        "CQ_err": [0.033, 0.031, 0.030, 0.029, 0.027, 0.027, 0.025, 0.024,
                   0.023, 0.022, 0.047, 0.043, 0.020, 0.019, 0.037, 0.034],
        "TR":     [30.466, 31.571, 32.380, 33.865, 34.446, 35.584, 37.020, 38.281,
                   39.259, 41.374, 42.094, 43.643, 44.963, 47.006, 48.313, 53.351],
        "TR_err": [0.298, 0.301, 0.301, 0.312, 0.315, 0.324, 0.329, 0.341,
                   0.352, 0.358, 0.371, 0.386, 0.392, 0.418, 0.411, 0.461],
    },

    "HPK14": {
        "CQ":     [28.192, 24.992, 22.655, 20.627, 18.780, 17.306, 16.016, 14.924,
                   13.839, 12.969, 12.123, 11.362, 10.691, 10.020, 9.453, 8.484],
        "CQ_err": [0.090, 0.089, 0.082, 0.076, 0.075, 0.075, 0.069, 0.067,
                   0.065, 0.067, 0.059, 0.123, 0.056, 0.051, 0.109, 0.049],
        "TR":     [29.450, 29.911, 32.366, 33.714, 34.169, 34.254, 37.901, 38.816,
                   40.212, 40.145, 42.705, 45.201, 45.018, 49.362, 50.622, 53.626],
        "TR_err": [0.731, 0.749, 0.773, 0.818, 0.834, 0.816, 0.838, 0.848,
                   0.864, 0.959, 0.953, 0.968, 1.000, 1.051, 1.043, 1.184],
    },

    "HPK15": {
        "CQ":     [30.695, 26.512, 23.692, 21.479, 19.548, 17.955, 16.524, 15.210,
                   14.152, 13.133, 12.298, 11.450, 10.829, 10.118, 9.632, 8.620],
        "CQ_err": [0.035, 0.029, 0.028, 0.027, 0.026, 0.025, 0.024, 0.023,
                   0.022, 0.021, 0.020, 0.019, 0.041, 0.018, 0.040, 0.033],
        "TR":     [27.425, 27.955, 29.684, 31.012, 32.769, 33.031, 34.788, 35.412,
                   37.112, 38.441, 40.388, 41.111, 42.845, 44.561, 45.200, 48.342],
        "TR_err": [0.300, 0.304, 0.303, 0.311, 0.311, 0.324, 0.325, 0.338,
                   0.344, 0.352, 0.359, 0.369, 0.376, 0.398, 0.405, 0.447],
    },

    "HPK16": {
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

# =========================
# Input data_3
# =========================

BV_batch3 = np.array([220, 210, 205, 200, 195, 190, 185, 180,
                      175, 170, 165, 160, 155, 150, 145, 140])

data_3 = {
    "FBK240": {
        "CQ":     [33.27, 26.66, 24.50, 22.60, 20.82, 19.32, 18.01, 16.79,
                   15.73, 14.71, 13.79, 13.06, 12.30, 11.67, 11.10, 10.48],
        "CQ_err": [0.08, 0.07, 0.06, 0.06, 0.06, 0.05, 0.06, 0.05,
                   0.05, 0.05, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04],
        "TR":     [32.67, 36.67, 36.85, 38.60, 37.54, 40.73, 39.05, 42.68,
                   42.69, 43.98, 47.77, 45.41, 48.54, 47.97, 48.92, 54.13],
        "TR_err": [0.70, 0.66, 0.65, 0.68, 0.67, 0.71, 0.66, 0.67,
                   0.71, 0.69, 0.77, 0.71, 0.74, 0.76, 0.78, 0.83],
    },

    "FBK242": {
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

# =========================
# Attach BV to each DUT
# =========================

for dut in data_1:
    data_1[dut]["BV"] = BV_batch1

for dut in data_2:
    data_2[dut]["BV"] = BV_batch2

for dut in data_3:
    data_3[dut]["BV"] = BV_batch3

batches = {
    "batch1_HPK1_3_17_18": data_1,
    "batch2_HPK10_14_15_16": data_2,
    "batch3_FBK240_242": data_3,
}

# =========================
# Common CMS label
# =========================

def add_cms_label(ax, batch_title=""):
    hep.cms.label(
        "ETL Preliminary",
        #"Preliminary",
        data=True,
        ax=ax,
        loc=0,
        fontsize=16,
        rlabel="",
    )

    ax.text(
        0.03, 0.95,
        "LGAD Test Beam 120 GeV (June 2026)",
        transform=ax.transAxes,
        fontsize=17,
        ha="left",
        va="top",
    )

    ax.text(
        0.97, 1.010,
        r"CERN SPS H6 (Temp = $-20.5^{\circ}$C)",
        transform=ax.transAxes,
        fontsize=17,
        ha="right",
        #va="bottom",
    )
    if batch_title:
        ax.text(
            0.03, 0.88,
            batch_title,
            transform=ax.transAxes,
            fontsize=17,
            ha="left",
            va="top",
        )

# =========================
# Utility: x ticks for BV plot
# =========================

def get_bv_ticks(batch_data):
    all_bv = np.concatenate([np.array(values["BV"]) for values in batch_data.values()])
    bv_min = np.min(all_bv)
    bv_max = np.max(all_bv)

    tick_start = int(np.floor(bv_min / 20.0) * 20)
    tick_end   = int(np.ceil(bv_max / 20.0) * 20)

    ticks = np.arange(tick_start, tick_end + 1, 20)
    return bv_min, bv_max, ticks

# =========================
# BV vs quantity plot
# =========================

def make_cms_plot_vs_bv(
    batch_data,
    batch_title,
    y_key,
    yerr_key,
    ylabel,
    output_name,
    ylim=None,
    legend_loc="lower left",
):
    fig, ax = plt.subplots(figsize=(8, 7))

    for dut, values in batch_data.items():
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
            label=dut,
        )

    bv_min, bv_max, ticks = get_bv_ticks(batch_data)

    ax.set_xlabel("Bias voltage [V]")
    ax.set_ylabel(ylabel)
    ax.set_xlim(bv_min - 5, bv_max + 5)
    ax.set_xticks(ticks)

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.4)

    ax.legend(
        loc=legend_loc,
        frameon=False,
        fontsize=15,
        ncol=2,
    )

    add_cms_label(ax, batch_title=batch_title)

    fig.tight_layout()
    fig.savefig(output_name + ".pdf")
    fig.savefig(output_name + ".png", dpi=300)
    plt.show()

# =========================
# Charge collection vs Timing resolution plot
# =========================

def make_cms_plot_charge_vs_timing(
    batch_data,
    batch_title,
    output_name,
    legend_loc="lower left",
):
    fig, ax = plt.subplots(figsize=(8, 7))

    all_cq = []

    for dut, values in batch_data.items():
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
            markersize=7,
            linewidth=2,
            elinewidth=1.5,
            capsize=3,
            label=dut,
        )

        all_cq.extend(cq.tolist())

    all_cq = np.array(all_cq)

    ax.set_xlabel("Collected charge [fC]")
    ax.set_ylabel("Timing resolution [ps]")
    ax.set_xlim(max(0, np.min(all_cq) - 2), np.max(all_cq) + 3)
    ax.set_ylim(20, 70)

    ax.grid(True, which="major", linestyle="--", linewidth=0.7, alpha=0.4)

    ax.legend(
        loc=legend_loc,
        frameon=False,
        fontsize=15,
        ncol=2,
    )

    add_cms_label(ax, batch_title=batch_title)

    fig.tight_layout()
    fig.savefig(output_name + ".pdf")
    fig.savefig(output_name + ".png", dpi=300)
    plt.show()

# =========================
# Make plots for each batch
# =========================

for batch_name, batch_data in batches.items():

    if batch_name == "batch1_HPK1_3_17_18":
        batch_title = "Batch 1: HPK1, HPK3, HPK17, HPK18"
    elif batch_name == "batch2_HPK10_14_15_16":
        batch_title = "Batch 2: HPK10, HPK14, HPK15, HPK16"
    elif batch_name == "batch3_FBK240_242":
        batch_title = "Batch 3: FBK240, FBK242"
    else:
        batch_title = batch_name

    # 1. BV vs Collected charge
    make_cms_plot_vs_bv(
        batch_data=batch_data,
        batch_title=batch_title,
        y_key="CQ",
        yerr_key="CQ_err",
        ylabel="Collected charge [fC]",
        output_name=f"TB2026_{batch_name}_CollectedCharge_vs_BV",
        ylim=(0, 50),
        legend_loc="lower right",
    )

    # 2. BV vs Timing resolution
    make_cms_plot_vs_bv(
        batch_data=batch_data,
        batch_title=batch_title,
        y_key="TR",
        yerr_key="TR_err",
        ylabel="Timing resolution [ps]",
        output_name=f"TB2026_{batch_name}_TimingResolution_vs_BV",
        ylim=(20, 70),
        legend_loc="lower left",
    )

    # 3. Collected charge vs Timing resolution
    make_cms_plot_charge_vs_timing(
        batch_data=batch_data,
        batch_title=batch_title,
        output_name=f"TB2026_{batch_name}_TimingResolution_vs_CollectedCharge",
        legend_loc="lower left",
    )

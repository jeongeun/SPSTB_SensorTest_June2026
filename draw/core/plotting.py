"""
core/plotting.py
----------------
All matplotlib canvas / figure functions:
  - draw_three_var_canvas   : pmax / tmax / area per-DUT panel (with Langau fit)
  - plot_deltaT_overlay     : six DUT-pair ΔT distributions overlaid
  - plot_deltaT_fits_overlay: 2×3 panel of per-pair Gaussian fits
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from core.config import (
    DUT_LABELS, DUT_SHORT, VAR_INFO, PAIR_DEFS,
    FILL_COLOR, LINE_COLOR, FIT_COLOR, PAIR_COLORS,
    PANEL_TITLE_SIZE, STAT_TEXT_SIZE, AREA_FIT_BINS, CHARGE_CONVERSION,
)
from core.cuts import finite_values, get_stat
from core.fitting import (
    fit_area_landau_mpv, fit_area_langau_mpv, langau_counts_model,
    gaussian, fit_gaussian_deltaT,
)
from core.models import FitResult
from core.style import add_cms_label, style_axis_labels


# ── Per-DUT three-variable canvas ────────────────────────────────────────────

def draw_three_var_canvas(
    arrays: Dict[str, np.ndarray],
    dut_idx: int,
    mask: np.ndarray,
    edges: Dict[Tuple[int, str], np.ndarray],
    title: str,
    out_path: str,
    fit_method: str = "landau",
) -> List[Dict[str, object]]:
    """
    Draw pmax_fit / tmax_fit / area_new histograms for one DUT on a 1×3 canvas.

    fit_method: "landau"  — pure Landau (Moyal) fit on area_new   [default]
                "langau"  — Landau-Gaussian convolution fit on area_new

    Returns a list of one stat-row dict per variable (for CSV export).
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    stat_rows: List[Dict[str, object]] = []

    for ax, var in zip(axes, ["pmax_fit", "tmax_fit", "area_new"]):
        info   = VAR_INFO[var]
        arr    = arrays[var][:, dut_idx]
        use    = mask & np.isfinite(arr)
        if info["positive"]:
            use &= arr > 0
        values = arr[use]

        ax.hist(values, bins=edges[(dut_idx, var)],
                histtype="stepfilled", color=FILL_COLOR, edgecolor=LINE_COLOR, alpha=0.45)
        ax.hist(values, bins=edges[(dut_idx, var)],
                histtype="step", color=FILL_COLOR, edgecolor=LINE_COLOR, linewidth=1.5)
        style_axis_labels(ax, info["xlabel"], "Entries")
        if info["logy"]:
            ax.set_yscale("log")
        if var == "area_new":
            ax.set_xlim(0, 2000)

        st  = get_stat(values, positive=False)
        row = {
            "stage": title, "dut": DUT_SHORT[dut_idx], "variable": var,
            "entries": st.entries, "mean": st.mean, "std": st.std,
            "landau_mpv": np.nan, "landau_mpv_err": np.nan,
            "landau_sigma": np.nan, "landau_sigma_err": np.nan,
            "gauss_sigma": np.nan, "gauss_sigma_err": np.nan,
            "landau_chi2_ndf": np.nan,
            "collected_charge": np.nan, "collected_charge_err": np.nan,
        }

        stat_text = (
            f"Entries = {st.entries:,}\n"
            f"Mean = {st.mean:.5g}\n"
            f"Std = {st.std:.5g}"
        )

        if var == "area_new":
            if fit_method == "langau":
                stat_text = _overlay_langau(ax, values, edges[(dut_idx, var)], row, stat_text)
            else:
                stat_text = _overlay_landau(ax, values, edges[(dut_idx, var)], row, stat_text)

        stat_rows.append(row)
        ax.text(0.97, 0.95, stat_text, transform=ax.transAxes,
                ha="right", va="top", fontsize=STAT_TEXT_SIZE)
        ax.set_title(var, fontsize=PANEL_TITLE_SIZE)

    fig.suptitle(f"{DUT_LABELS[dut_idx]}  |  {title}", fontsize=18, fontweight="bold")
    for ax in axes:
        add_cms_label(ax, label="SPS2026June", extra=r"(T=-$20.5^{\circ}$)", fontsize=16)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return stat_rows


def _overlay_landau(
    ax,
    values: np.ndarray,
    bin_edges: np.ndarray,
    row: dict,
    stat_text: str,
) -> str:
    """Fit area_new with a pure Landau (Moyal) PDF, overlay the curve, return updated stat_text."""
    from scipy.stats import moyal as _moyal  # local to avoid circular import at module level

    area_fit = fit_area_landau_mpv(values)
    collected_charge     = area_fit.mpv     / CHARGE_CONVERSION
    collected_charge_err = area_fit.mpv_err / CHARGE_CONVERSION

    row.update({
        "landau_mpv":           area_fit.mpv,
        "landau_mpv_err":       area_fit.mpv_err,
        "landau_sigma":         area_fit.sigma,       # Moyal scale = Landau σ
        "landau_sigma_err":     area_fit.sigma_err,
        "gauss_sigma":          np.nan,               # not applicable for pure Landau
        "gauss_sigma_err":      np.nan,
        "landau_chi2_ndf":      area_fit.chi2_ndf,
        "collected_charge":     collected_charge,
        "collected_charge_err": collected_charge_err,
    })

    if np.isfinite(area_fit.mpv):
        x_fit     = np.linspace(area_fit.fit_lo, area_fit.fit_hi, 600)
        bin_width = float(bin_edges[1] - bin_edges[0])
        print(
            f"[AREA FIT landau] MPV = {area_fit.mpv:.3f} ± {area_fit.mpv_err:.3f},  "
            f"σ = {area_fit.sigma:.3f},  "
            f"Qcoll = {collected_charge:.3f} ± {collected_charge_err:.3f} fC"
        )
        y_fit = area_fit.amp * bin_width * _moyal.pdf(x_fit, loc=area_fit.mpv, scale=area_fit.sigma)
        ax.plot(x_fit, y_fit, color=FIT_COLOR, linewidth=1.5, label="Landau fit")
        stat_text += (
            "\n"
            rf"MPV = {area_fit.mpv:.2f} $\pm$ {area_fit.mpv_err:.2f}" + "\n"
            rf"$Q_{{coll}}$ = {collected_charge:.2f} $\pm$ {collected_charge_err:.2f} fC" + "\n"
            rf"$\sigma_L$ = {area_fit.sigma:.2f}" + "\n"
            rf"$\chi^2/ndf$ = {area_fit.chi2_ndf:.2f}"
        )
        ax.legend(fontsize=10, loc="lower right")
    else:
        stat_text += "\nLandau fit failed"

    return stat_text


def _overlay_langau(
    ax,
    values: np.ndarray,
    bin_edges: np.ndarray,
    row: dict,
    stat_text: str,
) -> str:
    """Fit area_new with Landau-Gaussian convolution, overlay the curve, return updated stat_text."""
    area_fit = fit_area_langau_mpv(values, n_bins=AREA_FIT_BINS)
    collected_charge     = area_fit.mpv     / CHARGE_CONVERSION
    collected_charge_err = area_fit.mpv_err / CHARGE_CONVERSION

    row.update({
        "landau_mpv":           area_fit.mpv,
        "landau_mpv_err":       area_fit.mpv_err,
        "landau_sigma":         area_fit.landau_sigma,
        "landau_sigma_err":     area_fit.landau_sigma_err,
        "gauss_sigma":          area_fit.gauss_sigma,
        "gauss_sigma_err":      area_fit.gauss_sigma_err,
        "landau_chi2_ndf":      area_fit.chi2_ndf,
        "collected_charge":     collected_charge,
        "collected_charge_err": collected_charge_err,
    })

    if np.isfinite(area_fit.mpv):
        x_fit     = np.linspace(area_fit.fit_lo, area_fit.fit_hi, 600)
        bin_width = float(bin_edges[1] - bin_edges[0])
        print(
            f"[AREA FIT langau] MPV = {area_fit.mpv:.3f} ± {area_fit.mpv_err:.3f},  "
            f"Qcoll = {collected_charge:.3f} ± {collected_charge_err:.3f} fC"
        )
        y_fit = langau_counts_model(
            x_fit, area_fit.mpv, area_fit.landau_sigma,
            area_fit.gauss_sigma, area_fit.norm_yield, bin_width,
        )
        ax.plot(x_fit, y_fit, color=FIT_COLOR, linewidth=1.5,
                label=r"Landau $\otimes$ Gaussian fit")
        stat_text += (
            "\n"
            rf"MPV = {area_fit.mpv:.2f} $\pm$ {area_fit.mpv_err:.2f}" + "\n"
            rf"$Q_{{coll}}$ = {collected_charge:.2f} $\pm$ {collected_charge_err:.2f} fC" + "\n"
            rf"$\sigma_L$ = {area_fit.landau_sigma:.2f}" + "\n"
            rf"$\sigma_G$ = {area_fit.gauss_sigma:.2f}" + "\n"
            rf"$\chi^2/ndf$ = {area_fit.chi2_ndf:.2f}"
        )
        ax.legend(fontsize=10, loc="lower right")
    else:
        stat_text += "\nLangau fit failed"

    return stat_text


# ── ΔT overlay (all pairs, no fits) ─────────────────────────────────────────

def plot_deltaT_overlay(
    dt_arrays: Dict[str, np.ndarray],
    out_path: str,
    bins: int,
    dt_range: Tuple[float, float],
    title: str,
) -> List[Dict[str, object]]:
    """Overlay all six DUT-pair ΔT distributions on one axis (step histograms)."""
    fig, ax = plt.subplots(figsize=(11, 7))
    stats_rows: List[Dict[str, object]] = []

    for label, arr in dt_arrays.items():
        v  = finite_values(arr)
        st = get_stat(v)
        ax.hist(v, bins=bins, range=dt_range, histtype="step", linewidth=1.8,
                color=PAIR_COLORS.get(label, "black"),
                label=rf"$\Delta T_{{{label}}}$  Nevt={len(v):,}")
        stats_rows.append({
            "pair": label, "entries": st.entries,
            "mean_ps": st.mean, "std_ps": st.std,
        })

    style_axis_labels(
        ax,
        xlabel=r"$\Delta T = \mathrm{CFD20}(DUT_i) - \mathrm{CFD20}(DUT_j) [ps]$",
        ylabel="Entries",
    )
    ax.set_title(title, fontsize=PANEL_TITLE_SIZE)
    ax.legend(fontsize=11, loc="upper left", ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return stats_rows


# ── ΔT Gaussian fits (2×3 panel) ─────────────────────────────────────────────

def plot_deltaT_fits_overlay(
    dt_arrays: Dict[str, np.ndarray],
    out_path: str,
    bins: int,
    dt_range: Tuple[float, float],
    fit_half_range_ps: float,
    title: str,
) -> Dict[str, FitResult]:
    """
    Draw six DUT-pair ΔT Gaussian fits in a 2×3 CMS-style canvas.
    Each panel uses median ± fit_half_range_ps as the fit and plot window.
    """
    fig, axes = plt.subplots(2, 3, figsize=(25, 15))
    axes_flat   = axes.flatten()
    fit_results: Dict[str, FitResult] = {}

    for ax, (i, j, label) in zip(axes_flat, PAIR_DEFS):
        v = finite_values(dt_arrays.get(label, np.array([])))

        if len(v) < 20:
            fit_results[label] = FitResult(
                label, len(v), np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
            )
            ax.text(0.50, 0.50, "Insufficient data",
                    transform=ax.transAxes, ha="center", va="center", fontsize=STAT_TEXT_SIZE)
            ax.set_title(rf"$\Delta T_{{{label}}}$=DUT{i+1}-DUT{j+1}", fontsize=PANEL_TITLE_SIZE)
            style_axis_labels(ax, xlabel=r"$\Delta T$ [ps]", ylabel="Entries")
            continue

        fit       = fit_gaussian_deltaT(v, bins=bins, fit_half_range_ps=fit_half_range_ps)
        fit.pair  = label
        fit_results[label] = fit

        med     = float(np.median(v))
        plot_lo = (fit.mu_ps if np.isfinite(fit.mu_ps) else med) - fit_half_range_ps
        plot_hi = (fit.mu_ps if np.isfinite(fit.mu_ps) else med) + fit_half_range_ps

        ax.hist(v, bins=bins, range=(plot_lo, plot_hi),
                histtype="stepfilled", color=FILL_COLOR, edgecolor="none", alpha=0.25, label="Data")
        ax.hist(v, bins=bins, range=(plot_lo, plot_hi),
                histtype="step", color=LINE_COLOR, linewidth=1.8)

        if np.isfinite(fit.sigma_ps):
            x_fit = np.linspace(plot_lo, plot_hi, 600)
            ax.plot(x_fit, gaussian(x_fit, fit.amp, fit.mu_ps, fit.sigma_ps),
                    linestyle="-", linewidth=2.0, color=FIT_COLOR, label="Gaussian fit")
            stat_txt = (
                f"Entries = {fit.entries:,}\n"
                rf"$\mu$ = {fit.mu_ps:.2f} ps" + "\n"
                rf"$\sigma$ = {fit.sigma_ps:.2f} $\pm$ {fit.sigma_err_ps:.2f} ps" + "\n"
                rf"$\chi^2/ndf$ = {fit.chi2_ndf:.2f}"
            )
        else:
            stat_txt = (
                f"Entries = {fit.entries:,}\n"
                f"Mean = {fit.mean_ps:.2f} ps\n"
                f"RMS = {fit.rms_ps:.2f} ps\n"
                "Fit failed"
            )

        ax.text(0.97, 0.94, stat_txt, transform=ax.transAxes,
                ha="right", va="top", fontsize=STAT_TEXT_SIZE)
        ax.set_title(rf"$\Delta T_{{{label}}}$ = DUT{i+1} - DUT{j+1}", fontsize=PANEL_TITLE_SIZE)
        style_axis_labels(ax, xlabel=r"$\Delta T$ [ps]", ylabel="Entries")
        ax.legend(fontsize=15, loc="upper left")

    for ax in axes_flat:
        add_cms_label(ax, label="SPS2026June", extra=r"(T=$-20.5^{\circ}$)", fontsize=16)

    fig.suptitle(title, fontsize=20, fontweight="bold", y=0.995)
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
    plt.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return fit_results

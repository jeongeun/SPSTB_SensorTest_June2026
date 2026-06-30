"""
core/io_utils.py
----------------
File I/O utilities:
  - ROOT file loading via uproot
  - CSV saving for per-DUT stats, deltaT stats, fit results, individual sigmas
  - Console printing helpers (stats tables, fit summaries)
"""
from __future__ import annotations

import csv
import os
from typing import Dict, List, Optional

import numpy as np
import uproot

from core.config import DUT_SHORT, PAIR_DEFS
from core.models import FitResult


# ── Directory helpers ────────────────────────────────────────────────────────

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ── ROOT loading ─────────────────────────────────────────────────────────────

def np_from_uproot(arr) -> np.ndarray:
    """Convert fixed-size or jagged uproot arrays to a dense float ndarray."""
    a = np.asarray(arr)
    if a.dtype == object:
        a = np.stack([np.asarray(x, dtype=float) for x in a])
    return np.asarray(a, dtype=float)


def load_analysis_arrays(file_path: str, tree_name: str) -> Dict[str, np.ndarray]:
    """
    Open a ROOT file and return numpy arrays for pmax_fit, tmax_fit,
    area_new, and cfd from the specified TTree.

    Raises KeyError / ValueError on missing tree, branches, or wrong shapes.
    """
    branches = ["pmax_fit", "tmax_fit", "area_new", "cfd"]
    with uproot.open(file_path) as f:
        if tree_name not in {k.split(";")[0] for k in f.keys()}:
            available = ", ".join(k.split(";")[0] for k in f.keys())
            raise KeyError(
                f"Tree '{tree_name}' not found. "
                f"Available trees/objects: {available}"
            )
        tree = f[tree_name]
        missing = [b for b in branches if b not in tree.keys()]
        if missing:
            raise KeyError(
                "Missing required branch(es): " + ", ".join(missing) +
                "\nAvailable branches include: " + ", ".join(list(tree.keys())[:80])
            )
        arrays = tree.arrays(branches, library="np")

    pmax = np_from_uproot(arrays["pmax_fit"])
    tmax = np_from_uproot(arrays["tmax_fit"])
    area = np_from_uproot(arrays["area_new"])
    cfd  = np_from_uproot(arrays["cfd"])

    if pmax.ndim != 2 or pmax.shape[1] < 6:
        raise ValueError(f"pmax_fit must have shape (N, >=6). Got: {pmax.shape}")
    if tmax.ndim != 2 or tmax.shape[1] < 6:
        raise ValueError(f"tmax_fit must have shape (N, >=6). Got: {tmax.shape}")
    if area.ndim != 2 or area.shape[1] < 6:
        raise ValueError(f"area_new must have shape (N, >=6). Got: {area.shape}")

    if cfd.ndim == 2:
        if cfd.shape[1] % 6 != 0:
            raise ValueError(f"Cannot reshape cfd with shape {cfd.shape} into (N, 6, nfrac)")
        cfd = cfd.reshape(cfd.shape[0], 6, cfd.shape[1] // 6)
    if cfd.ndim != 3 or cfd.shape[1] < 6:
        raise ValueError(f"cfd must have shape (N, >=6, nfrac). Got: {cfd.shape}")

    return {
        "pmax_fit": pmax[:, :6],
        "tmax_fit": tmax[:, :6],
        "area_new": area[:, :6],
        "cfd":      cfd[:, :6, :],
    }


# ── CSV savers ───────────────────────────────────────────────────────────────

def save_stats_csv(path: str, stats_rows: List[Dict[str, object]]) -> None:
    """Write per-DUT variable statistics (with optional Langau fit columns)."""
    fieldnames = [
        "stage", "dut", "variable", "entries", "mean", "std",
        "landau_mpv", "landau_mpv_err",
        "landau_sigma", "landau_sigma_err",
        "gauss_sigma", "gauss_sigma_err",
        "landau_chi2_ndf",
        "collected_charge", "collected_charge_err",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(stats_rows)


def save_delta_stats_csv(path: str, rows: List[Dict[str, object]]) -> None:
    """Write raw deltaT statistics (before Gaussian fit)."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["pair", "entries", "mean_ps", "std_ps"]
        )
        writer.writeheader()
        writer.writerows(rows)


def save_fit_results_csv(path: str, fit_results: Dict[str, FitResult]) -> None:
    """Write Gaussian fit results for each DUT pair."""
    fieldnames = [
        "pair", "entries", "mean_ps", "rms_ps",
        "fit_mu_ps", "fit_sigma_ps", "fit_sigma_err_ps", "chi2_ndf",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for label in [p[2] for p in PAIR_DEFS]:
            fit = fit_results[label]
            writer.writerow({
                "pair":             label,
                "entries":          fit.entries,
                "mean_ps":          fit.mean_ps,
                "rms_ps":           fit.rms_ps,
                "fit_mu_ps":        fit.mu_ps,
                "fit_sigma_ps":     fit.sigma_ps,
                "fit_sigma_err_ps": fit.sigma_err_ps,
                "chi2_ndf":         fit.chi2_ndf,
            })


def save_individual_sigmas_csv(
    path: str,
    sigmas: np.ndarray,
    variances: np.ndarray,
    sigma_errors: Optional[np.ndarray] = None,
    variance_errors: Optional[np.ndarray] = None,
) -> None:
    """Write per-DUT timing resolution (sigma, variance) with optional errors."""
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["DUT", "sigma_ps", "sigma_err_ps", "variance_ps2", "variance_err_ps2"],
        )
        writer.writeheader()
        for i in range(4):
            writer.writerow({
                "DUT":              DUT_SHORT[i],
                "sigma_ps":         sigmas[i],
                "sigma_err_ps":     sigma_errors[i]    if sigma_errors    is not None else np.nan,
                "variance_ps2":     variances[i],
                "variance_err_ps2": variance_errors[i] if variance_errors is not None else np.nan,
            })


# ── Console printers ─────────────────────────────────────────────────────────

def print_stats_table(title: str, stats_rows: List[Dict[str, object]]) -> None:
    print(f"\n=== {title} ===")
    print(f"{'DUT':<8} {'variable':<10} {'entries':>10} {'mean':>15} {'std':>15}")
    print("-" * 64)
    for row in stats_rows:
        print(
            f"{str(row['dut']):<8} {str(row['variable']):<10} "
            f"{int(row['entries']):>10d} {float(row['mean']):>15.6g} "
            f"{float(row['std']):>15.6g}"
        )


def print_delta_stats(rows: List[Dict[str, object]]) -> None:
    print("\n=== deltaT statistics before Gaussian fit ===")
    print(f"{'pair':<8} {'entries':>10} {'mean [ps]':>15} {'std [ps]':>15}")
    print("-" * 54)
    for r in rows:
        print(
            f"deltaT_{r['pair']:<2} {int(r['entries']):>10d} "
            f"{float(r['mean_ps']):>15.6g} {float(r['std_ps']):>15.6g}"
        )


def print_fit_results(fit_results: Dict[str, FitResult]) -> None:
    print("\n=== Gaussian fit results for deltaT ===")
    print(f"{'pair':<8} {'entries':>10} {'mu [ps]':>13} {'sigma [ps]':>15} {'err [ps]':>12} {'chi2/ndf':>10}")
    print("-" * 78)
    for label in [p[2] for p in PAIR_DEFS]:
        fit = fit_results[label]
        print(
            f"sigma{label:<2} {fit.entries:>10d} {fit.mu_ps:>13.4g} "
            f"{fit.sigma_ps:>15.4g} {fit.sigma_err_ps:>12.4g} {fit.chi2_ndf:>10.4g}"
        )


def print_individual_sigmas(
    sigmas: np.ndarray,
    variances: np.ndarray,
    residuals: np.ndarray,
    sigma_errors: Optional[np.ndarray] = None,
    variance_errors: Optional[np.ndarray] = None,
) -> None:
    print("\n=== Individual DUT timing resolutions from sigma_ij^2 = sigma_i^2 + sigma_j^2 ===")
    for i, sigma in enumerate(sigmas):
        s_err = sigma_errors[i]    if sigma_errors    is not None else None
        v_err = variance_errors[i] if variance_errors is not None else None
        base  = f"  sigma{i+1} ({DUT_SHORT[i]}) = {sigma:.3f}"
        base += f" ± {s_err:.3f} ps" if (s_err is not None and np.isfinite(s_err)) else " ps"
        base += f"   variance = {variances[i]:.3f}"
        base += f" ± {v_err:.3f} ps^2" if (v_err is not None and np.isfinite(v_err)) else " ps^2"
        print(base)

    if residuals.size:
        rms_res = float(np.sqrt(np.mean(residuals ** 2)))
        print(f"  LS residual RMS in variance space = {rms_res:.3f} ps^2")

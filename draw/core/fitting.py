"""
core/fitting.py
---------------
All curve-fitting routines:
  - Landau-Gaussian convolution (Langau) for area_new MPV extraction
  - Gaussian fit for per-pair ΔT distributions
  - Least-squares solve for individual DUT timing resolutions
  - Toy-MC error propagation for individual sigmas
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy.optimize import curve_fit, lsq_linear
from scipy.stats import moyal, norm

from core.config import (
    AREA_FIT_BINS, AREA_FIT_QUANTILES, LANGAU_CONV_POINTS,
    LANDAU_FIT_BINS, LANDAU_FIT_QUANTILES,
    PAIR_DEFS,
)
from core.cuts import finite_values
from core.models import AreaLandauResult, AreaLangauResult, FitResult


# ── Landau-Gaussian convolution ──────────────────────────────────────────────

def langau_pdf(
    x: np.ndarray,
    mpv: float,
    landau_sigma: float,
    gauss_sigma: float,
) -> np.ndarray:
    """Normalized Landau-Gaussian convolution PDF (Landau ≈ scipy.stats.moyal)."""
    x = np.asarray(x, dtype=float)
    landau_sigma = max(float(landau_sigma), 1.0e-9)
    gauss_sigma  = max(float(gauss_sigma),  1.0e-9)

    xmin, xmax = float(np.min(x)), float(np.max(x))
    width = max(xmax - xmin, 1.0)

    lo = xmin - 5.0 * gauss_sigma - 4.0 * landau_sigma
    hi = xmax + 5.0 * gauss_sigma + 8.0 * landau_sigma
    if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
        lo, hi = xmin - width, xmax + width

    xx = np.linspace(lo, hi, LANGAU_CONV_POINTS)
    landau_part = moyal.pdf(xx, loc=mpv, scale=landau_sigma)
    gauss_part  = norm.pdf(x[:, None], loc=xx[None, :], scale=gauss_sigma)
    conv = np.trapz(landau_part[None, :] * gauss_part, xx, axis=1)

    norm_factor = np.trapz(conv, x)
    if np.isfinite(norm_factor) and norm_factor > 0:
        conv = conv / norm_factor
    return conv


def langau_counts_model(
    x: np.ndarray,
    mpv: float,
    landau_sigma: float,
    gauss_sigma: float,
    norm_yield: float,
    bin_width: float,
) -> np.ndarray:
    """Scale normalized Langau PDF by bin_width × norm_yield to get bin counts."""
    return norm_yield * bin_width * langau_pdf(x, mpv, landau_sigma, gauss_sigma)


def fit_area_langau_mpv(
    values: np.ndarray,
    n_bins: int = AREA_FIT_BINS,
) -> AreaLangauResult:
    """
    Fit area_new with a Landau-Gaussian convolution and return the MPV.
    Tries a 3×3 seed grid and keeps the best chi2/ndf solution.
    """
    v = finite_values(values, positive=True)

    _empty = AreaLangauResult(
        entries=int(len(v)),
        mpv=np.nan, mpv_err=np.nan,
        landau_sigma=np.nan, landau_sigma_err=np.nan,
        gauss_sigma=np.nan,  gauss_sigma_err=np.nan,
        norm_yield=np.nan,   chi2_ndf=np.nan,
        fit_lo=np.nan,       fit_hi=np.nan,
    )
    if len(v) < 30:
        return _empty

    qlo, qhi = np.percentile(v, AREA_FIT_QUANTILES)
    if not (np.isfinite(qlo) and np.isfinite(qhi) and qlo < qhi):
        qlo, qhi = float(np.min(v)), float(np.max(v))

    probe_counts, probe_edges = np.histogram(v, bins=120, range=(qlo, qhi))
    probe_centers = 0.5 * (probe_edges[:-1] + probe_edges[1:])
    peak_idx = int(np.argmax(probe_counts))
    peak_x   = float(probe_centers[peak_idx])
    peak_y   = float(probe_counts[peak_idx])

    above = np.where(probe_counts >= 0.5 * peak_y)[0]
    fwhm  = (
        float(probe_centers[above[-1]] - probe_centers[above[0]])
        if len(above) >= 2
        else float(np.percentile(v, 75) - np.percentile(v, 25))
    )
    if not np.isfinite(fwhm) or fwhm <= 0:
        fwhm = max(0.25 * peak_x, 20.0)

    fit_lo = max(qlo, peak_x - 2.0 * fwhm)
    fit_hi = min(qhi, peak_x + 10.0 * fwhm)
    if not (np.isfinite(fit_lo) and np.isfinite(fit_hi) and fit_lo < fit_hi):
        fit_lo, fit_hi = qlo, qhi

    counts, edges = np.histogram(v, bins=n_bins, range=(fit_lo, fit_hi))
    centers       = 0.5 * (edges[:-1] + edges[1:])
    positive_bins = counts > 0

    if np.count_nonzero(positive_bins) < 8:
        return AreaLangauResult(**{**_empty.__dict__, "fit_lo": float(fit_lo), "fit_hi": float(fit_hi)})

    bin_width  = float(edges[1] - edges[0])
    mpv0       = peak_x
    norm0      = float(np.sum(counts))
    sigma_low  = max(0.2 * bin_width, 1.0e-6)
    sigma_high = max(3.0 * fwhm, 5.0 * bin_width)
    mpv_low    = max(fit_lo, peak_x - 1.0 * fwhm)
    mpv_high   = min(fit_hi, peak_x + 0.8 * fwhm)

    landau_seeds = [max(f * fwhm, bin_width) for f in (0.10, 0.18, 0.30)]
    gauss_seeds  = [max(f * fwhm, bin_width) for f in (0.05, 0.12, 0.25)]

    best, best_chi2_ndf, last_error = None, np.inf, None

    for ls0 in landau_seeds:
        for gs0 in gauss_seeds:
            try:
                popt, pcov = curve_fit(
                    lambda x, mpv, ls, gs, ny:
                        langau_counts_model(x, mpv, ls, gs, ny, bin_width),
                    centers[positive_bins],
                    counts[positive_bins].astype(float),
                    p0=[mpv0, ls0, gs0, norm0],
                    sigma=np.sqrt(np.maximum(counts[positive_bins], 1)),
                    absolute_sigma=True,
                    bounds=(
                        [mpv_low,  sigma_low,  sigma_low,  0.0],
                        [mpv_high, sigma_high, sigma_high, np.inf],
                    ),
                    maxfev=50000,
                )
                y_fit    = langau_counts_model(
                    centers[positive_bins], popt[0], popt[1], popt[2], popt[3], bin_width,
                )
                chi2     = float(np.sum(
                    (counts[positive_bins] - y_fit) ** 2 / np.maximum(counts[positive_bins], 1)
                ))
                ndf      = int(np.count_nonzero(positive_bins) - 4)
                chi2_ndf = chi2 / ndf if ndf > 0 else np.nan
                if np.isfinite(chi2_ndf) and chi2_ndf < best_chi2_ndf:
                    best_chi2_ndf = chi2_ndf
                    best = (popt, np.sqrt(np.diag(pcov)), chi2_ndf)
            except Exception as e:
                last_error = e

    if best is None:
        print(
            f"[WARNING] Langau fit failed. Last error: {last_error}\n"
            f"          entries={len(v)}, peak_x={peak_x:.3f}, "
            f"fwhm={fwhm:.3f}, fit_range=({fit_lo:.3f}, {fit_hi:.3f})"
        )
        return AreaLangauResult(**{**_empty.__dict__, "fit_lo": float(fit_lo), "fit_hi": float(fit_hi)})

    popt, perr, chi2_ndf = best
    return AreaLangauResult(
        entries=int(len(v)),
        mpv=float(popt[0]),               mpv_err=float(perr[0]),
        landau_sigma=float(abs(popt[1])), landau_sigma_err=float(perr[1]),
        gauss_sigma=float(abs(popt[2])),  gauss_sigma_err=float(perr[2]),
        norm_yield=float(popt[3]),
        chi2_ndf=float(chi2_ndf),
        fit_lo=float(fit_lo),             fit_hi=float(fit_hi),
    )


# ── Pure Landau (Moyal) MPV fit ──────────────────────────────────────────────

def fit_area_landau_mpv(
    values: np.ndarray,
    n_bins: int = LANDAU_FIT_BINS,
) -> AreaLandauResult:
    """
    Fit area_new with a pure Landau distribution (approximated by scipy.stats.moyal).

    The Moyal PDF has its mode (MPV) at x = loc, so the fitted `loc` parameter
    is directly the Most Probable Value.

    Returns an AreaLandauResult with mpv, mpv_err, sigma (Landau width = moyal scale),
    sigma_err, chi2/ndf, and the fit range used.
    """
    v = finite_values(values, positive=True)

    _empty = AreaLandauResult(
        entries=int(len(v)),
        mpv=np.nan, mpv_err=np.nan,
        sigma=np.nan, sigma_err=np.nan,
        amp=np.nan, chi2_ndf=np.nan,
        fit_lo=np.nan, fit_hi=np.nan,
    )
    if len(v) < 30:
        return _empty

    qlo, qhi = np.percentile(v, LANDAU_FIT_QUANTILES)
    if not (np.isfinite(qlo) and np.isfinite(qhi) and qlo < qhi):
        qlo, qhi = float(np.min(v)), float(np.max(v))

    # Probe histogram to find peak position and FWHM for seeding
    probe_counts, probe_edges = np.histogram(v, bins=120, range=(qlo, qhi))
    probe_centers = 0.5 * (probe_edges[:-1] + probe_edges[1:])
    peak_idx = int(np.argmax(probe_counts))
    peak_x   = float(probe_centers[peak_idx])
    peak_y   = float(probe_counts[peak_idx])

    above = np.where(probe_counts >= 0.5 * peak_y)[0]
    fwhm  = (
        float(probe_centers[above[-1]] - probe_centers[above[0]])
        if len(above) >= 2
        else float(np.percentile(v, 75) - np.percentile(v, 25))
    )
    if not np.isfinite(fwhm) or fwhm <= 0:
        fwhm = max(0.25 * peak_x, 20.0)

    # Fit range: from 2×fwhm below peak to upper quantile (keep the Landau tail)
    fit_lo = max(qlo, peak_x - 2.0 * fwhm)
    fit_hi = qhi
    if not (np.isfinite(fit_lo) and np.isfinite(fit_hi) and fit_lo < fit_hi):
        fit_lo, fit_hi = qlo, qhi

    counts, edges = np.histogram(v, bins=n_bins, range=(fit_lo, fit_hi))
    centers       = 0.5 * (edges[:-1] + edges[1:])
    positive_bins = counts > 0

    if np.count_nonzero(positive_bins) < 5:
        return AreaLandauResult(**{**_empty.__dict__, "fit_lo": float(fit_lo), "fit_hi": float(fit_hi)})

    bin_width = float(edges[1] - edges[0])
    norm0     = float(np.sum(counts))
    scale0    = max(0.25 * fwhm, bin_width)  # Moyal scale ≈ Landau sigma

    def _model(x, loc, scale, N):
        return N * bin_width * moyal.pdf(x, loc=loc, scale=scale)

    try:
        popt, pcov = curve_fit(
            _model,
            centers[positive_bins],
            counts[positive_bins].astype(float),
            p0=[peak_x, scale0, norm0],
            sigma=np.sqrt(np.maximum(counts[positive_bins], 1)),
            absolute_sigma=True,
            bounds=(
                [max(fit_lo, peak_x - 1.5 * fwhm), 1.0e-6, 0.0],
                [min(fit_hi, peak_x + 0.8 * fwhm), 3.0 * fwhm, np.inf],
            ),
            maxfev=50000,
        )
        y_fit = _model(centers[positive_bins], *popt)
        chi2  = float(np.sum(
            (counts[positive_bins] - y_fit) ** 2 / np.maximum(counts[positive_bins], 1)
        ))
        ndf   = int(np.count_nonzero(positive_bins) - 3)
        perr  = np.sqrt(np.diag(pcov))
        return AreaLandauResult(
            entries=int(len(v)),
            mpv=float(popt[0]),          mpv_err=float(perr[0]),
            sigma=float(abs(popt[1])),   sigma_err=float(perr[1]),
            amp=float(popt[2]),
            chi2_ndf=float(chi2 / ndf) if ndf > 0 else np.nan,
            fit_lo=float(fit_lo),        fit_hi=float(fit_hi),
        )
    except Exception as e:
        print(
            f"[WARNING] Pure Landau fit failed: {e}\n"
            f"          entries={len(v)}, peak_x={peak_x:.3f}, "
            f"fwhm={fwhm:.3f}, fit_range=({fit_lo:.3f}, {fit_hi:.3f})"
        )
        return AreaLandauResult(**{**_empty.__dict__, "fit_lo": float(fit_lo), "fit_hi": float(fit_hi)})


# ── Gaussian fit for ΔT ──────────────────────────────────────────────────────

def gaussian(x: np.ndarray, amp: float, mu: float, sigma: float) -> np.ndarray:
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def fit_gaussian_deltaT(
    arr: np.ndarray,
    bins: int,
    fit_half_range_ps: float,
) -> FitResult:
    """Fit a Gaussian to a ΔT distribution within median ± fit_half_range_ps."""
    v = finite_values(arr)
    if len(v) < 20:
        return FitResult("", len(v), np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)

    med        = float(np.median(v))
    hist_range = (med - fit_half_range_ps, med + fit_half_range_ps)
    counts, edges = np.histogram(v, bins=bins, range=hist_range)
    centers    = 0.5 * (edges[:-1] + edges[1:])
    positive   = counts > 0

    if np.count_nonzero(positive) < 5:
        return FitResult("", len(v), float(np.mean(v)), float(np.std(v, ddof=1)),
                         np.nan, np.nan, np.nan, np.nan, np.nan)

    sigma0 = float(np.std(v[(v > hist_range[0]) & (v < hist_range[1])], ddof=1))
    if not np.isfinite(sigma0) or sigma0 <= 0:
        sigma0 = fit_half_range_ps / 5.0
    peak_idx = int(np.argmax(counts))
    p0 = [float(max(counts[peak_idx], 1)), float(centers[peak_idx]), sigma0]

    try:
        popt, pcov = curve_fit(
            gaussian,
            centers[positive],
            counts[positive].astype(float),
            p0=p0,
            sigma=np.sqrt(np.maximum(counts[positive], 1)),
            absolute_sigma=True,
            bounds=([0.0, hist_range[0], 1.0e-9], [np.inf, hist_range[1], np.inf]),
            maxfev=10000,
        )
        y    = gaussian(centers[positive], *popt)
        chi2 = float(np.sum((counts[positive] - y) ** 2 / np.maximum(counts[positive], 1)))
        ndf  = int(np.count_nonzero(positive) - 3)
        perr = np.sqrt(np.diag(pcov))
        return FitResult(
            pair="", entries=int(len(v)),
            mean_ps=float(np.mean(v)), rms_ps=float(np.std(v, ddof=1)),
            amp=float(popt[0]), mu_ps=float(popt[1]),
            sigma_ps=float(abs(popt[2])), sigma_err_ps=float(perr[2]),
            chi2_ndf=float(chi2 / ndf) if ndf > 0 else np.nan,
        )
    except Exception:
        return FitResult(
            pair="", entries=int(len(v)),
            mean_ps=float(np.mean(v)), rms_ps=float(np.std(v, ddof=1)),
            amp=np.nan, mu_ps=np.nan, sigma_ps=np.nan,
            sigma_err_ps=np.nan, chi2_ndf=np.nan,
        )


# ── Individual sigma extraction ──────────────────────────────────────────────

def solve_individual_sigmas(
    fit_results: Dict[str, FitResult],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve σ_ij² = σ_i² + σ_j² for σ₁..σ₄ via non-negative least squares.
    Returns: (sigmas_ps, variances_ps2, residuals_ps2)
    """
    A_rows, b_vals = [], []
    for i, j, label in PAIR_DEFS:
        s = fit_results[label].sigma_ps
        if not np.isfinite(s):
            continue
        row = np.zeros(4)
        row[i], row[j] = 1.0, 1.0
        A_rows.append(row)
        b_vals.append(s ** 2)

    if len(A_rows) < 4:
        return np.full(4, np.nan), np.full(4, np.nan), np.array([])

    A         = np.vstack(A_rows)
    b         = np.asarray(b_vals)
    result    = lsq_linear(A, b, bounds=(0.0, np.inf))
    variances = result.x
    sigmas    = np.sqrt(np.maximum(variances, 0.0))
    residuals = A @ variances - b
    return sigmas, variances, residuals


def solve_individual_sigmas_toy_mc(
    fit_results: Dict[str, FitResult],
    n_toys: int = 2000,
    seed: int = 12345,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Propagate pairwise σ fit errors to individual DUT timing resolutions via toy MC.
    Returns: (central_sigmas, central_variances, sigma_errors, variance_errors)
    """
    rng = np.random.default_rng(seed)
    central_sigmas, central_variances, _ = solve_individual_sigmas(fit_results)
    pair_labels = [p[2] for p in PAIR_DEFS]

    toy_sigmas, toy_variances = [], []
    for _ in range(n_toys):
        toy: Dict[str, FitResult] = {}
        for label in pair_labels:
            fit = fit_results[label]
            if (np.isfinite(fit.sigma_ps) and np.isfinite(fit.sigma_err_ps)
                    and fit.sigma_ps > 0 and fit.sigma_err_ps > 0):
                s = rng.normal(fit.sigma_ps, fit.sigma_err_ps)
                s = s if s > 0 else fit.sigma_ps
            else:
                s = fit.sigma_ps
            toy[label] = FitResult(
                pair=fit.pair, entries=fit.entries,
                mean_ps=fit.mean_ps, rms_ps=fit.rms_ps,
                amp=fit.amp, mu_ps=fit.mu_ps,
                sigma_ps=s, sigma_err_ps=fit.sigma_err_ps,
                chi2_ndf=fit.chi2_ndf,
            )
        s_toy, v_toy, _ = solve_individual_sigmas(toy)
        if np.all(np.isfinite(s_toy)) and np.all(np.isfinite(v_toy)):
            toy_sigmas.append(s_toy)
            toy_variances.append(v_toy)

    if not toy_sigmas:
        return central_sigmas, central_variances, np.full(4, np.nan), np.full(4, np.nan)

    toy_sigmas    = np.asarray(toy_sigmas)
    toy_variances = np.asarray(toy_variances)
    return (
        central_sigmas, central_variances,
        np.std(toy_sigmas,    axis=0, ddof=1),
        np.std(toy_variances, axis=0, ddof=1),
    )

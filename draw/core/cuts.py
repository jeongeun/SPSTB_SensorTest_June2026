"""
core/cuts.py
------------
Event selection (cuts / masks) and array utility functions.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from core.config import PAIR_DEFS
from core.models import Stat


def finite_values(arr: np.ndarray, positive: bool = False) -> np.ndarray:
    """Return only finite (and optionally positive) elements of arr."""
    mask = np.isfinite(arr)
    if positive:
        mask &= arr > 0
    return arr[mask]


def get_stat(arr: np.ndarray, positive: bool = False) -> Stat:
    """Compute entries, mean, std of finite (optionally positive) values."""
    v = finite_values(arr, positive=positive)
    if len(v) == 0:
        return Stat(0, np.nan, np.nan)
    return Stat(
        int(len(v)),
        float(np.mean(v)),
        float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
    )


def make_edges(
    arr: np.ndarray,
    bins: int,
    positive: bool = False,
    user_range: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """Build histogram bin edges spanning the data range (or a user-specified range)."""
    v = finite_values(arr, positive=positive)
    if len(v) == 0:
        return np.linspace(0.0, 1.0, bins + 1)
    if user_range is not None:
        lo, hi = user_range
    else:
        lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
        if not np.isfinite(lo) or not np.isfinite(hi):
            lo, hi = 0.0, 1.0
        if lo == hi:
            width = abs(lo) * 0.05 if lo != 0 else 1.0
            lo, hi = lo - width, hi + width
    return np.linspace(lo, hi, bins + 1)


def pmax_cut_mask_for_dut(
    pmax: np.ndarray,
    tmax: np.ndarray,
    dut_idx: int,
    cuts: List[float],
    cut_highs: Optional[List[float]],
    abs_tmax_cuts: Optional[List[float]] = None,
) -> np.ndarray:
    """Boolean mask: events passing the pmax (and optional tmax) cuts for one DUT."""
    p = pmax[:, dut_idx]
    t = tmax[:, dut_idx]
    mask = np.isfinite(p) & (p >= cuts[dut_idx])
    if cut_highs is not None and cut_highs[dut_idx] > 0:
        mask &= p <= cut_highs[dut_idx]
    if abs_tmax_cuts is not None and abs_tmax_cuts[dut_idx] > 0:
        mask &= np.isfinite(t) & (np.abs(t) <= abs_tmax_cuts[dut_idx])
    return mask


def pair_mask(
    pmax: np.ndarray,
    tmax: np.ndarray,
    i: int,
    j: int,
    cuts: List[float],
    cut_highs: Optional[List[float]],
    abs_tmax_cuts: Optional[List[float]],
    require_pmax_cuts: bool,
) -> np.ndarray:
    """Boolean mask: events where both DUT i and DUT j pass their respective cuts."""
    mask = np.ones(pmax.shape[0], dtype=bool)
    if require_pmax_cuts:
        mask &= pmax_cut_mask_for_dut(pmax, tmax, i, cuts, cut_highs, abs_tmax_cuts)
        mask &= pmax_cut_mask_for_dut(pmax, tmax, j, cuts, cut_highs, abs_tmax_cuts)
    return mask


def get_delta_t_arrays(
    arrays: Dict[str, np.ndarray],
    cfd_index: int,
    cfd_unit: str,
    pmax_cuts: List[float],
    pmax_cut_highs: Optional[List[float]],
    abs_tmax_cuts: Optional[List[float]],
    require_pmax_cuts: bool = True,
) -> Dict[str, np.ndarray]:
    """
    For each pair in PAIR_DEFS compute ΔT = CFD(DUT_i) - CFD(DUT_j) in ps
    after applying pairwise selection cuts.
    """
    cfd = arrays["cfd"]
    if cfd_index < 0 or cfd_index >= cfd.shape[2]:
        raise IndexError(
            f"cfd-index {cfd_index} is outside cfd third dimension "
            f"with size {cfd.shape[2]}"
        )
    scale_to_ps = 1.0 if cfd_unit == "ps" else 1.0e3

    out: Dict[str, np.ndarray] = {}
    for i, j, label in PAIR_DEFS:
        mask = pair_mask(
            arrays["pmax_fit"], arrays["tmax_fit"],
            i, j, pmax_cuts, pmax_cut_highs, abs_tmax_cuts,
            require_pmax_cuts,
        )
        dt = (cfd[:, i, cfd_index] - cfd[:, j, cfd_index]) * scale_to_ps
        out[label] = dt[mask & np.isfinite(dt)]
    return out


def global_dt_range(
    dt_arrays: Dict[str, np.ndarray],
    explicit_range: Optional[Tuple[float, float]],
    margin_frac: float = 0.08,
) -> Tuple[float, float]:
    """Return a common x-range for all pair ΔT distributions."""
    if explicit_range is not None:
        return explicit_range
    all_values = np.concatenate(
        [v[np.isfinite(v)] for v in dt_arrays.values() if len(v) > 0]
    )
    if len(all_values) == 0:
        return -1000.0, 1000.0
    lo, hi = np.percentile(all_values, [0.5, 99.5])
    if lo == hi:
        lo -= 1.0
        hi += 1.0
    margin = (hi - lo) * margin_frac
    return float(lo - margin), float(hi + margin)

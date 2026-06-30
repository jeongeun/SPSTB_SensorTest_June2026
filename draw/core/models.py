"""
models.py
---------
Dataclasses for carrying fit / statistics results between analysis steps.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Stat:
    entries: int
    mean: float
    std: float


@dataclass
class FitResult:
    pair: str
    entries: int
    mean_ps: float
    rms_ps: float
    amp: float
    mu_ps: float
    sigma_ps: float
    sigma_err_ps: float
    chi2_ndf: float


@dataclass
class AreaLandauResult:
    entries: int
    mpv: float
    mpv_err: float
    sigma: float
    sigma_err: float
    amp: float
    chi2_ndf: float
    fit_lo: float
    fit_hi: float


@dataclass
class AreaLangauResult:
    entries: int
    mpv: float
    mpv_err: float
    landau_sigma: float
    landau_sigma_err: float
    gauss_sigma: float
    gauss_sigma_err: float
    norm_yield: float
    chi2_ndf: float
    fit_lo: float
    fit_hi: float

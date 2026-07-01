#!/usr/bin/env python3
"""
Analysis.py  —  6-channel oscilloscope timing analysis (main entry point)
=========================================================================
Reads a ROOT Analysis tree and produces:
  01  pmax / tmax / area histograms before any cut          (per DUT)
  02  pmax / tmax / area histograms after pmax_fit cuts     (per DUT)
  03  DUT-pair ΔT overlay (no per-pair fits)
  04  DUT-pair ΔT Gaussian fits (2×3 canvas)
  05  Individual DUT timing resolutions via σ_ij² = σ_i² + σ_j²

Module layout
-------------
  core/config.py   — constants, labels, colours
  core/models.py   — dataclasses (Stat, FitResult, AreaLangauResult …)
  core/style.py    — CMS matplotlib style helpers
  core/io_utils.py — ROOT loading, CSV writing, console printers
  core/cuts.py     — event selection masks and ΔT array construction
  core/fitting.py  — Langau / Gaussian fits and sigma extraction
  core/plotting.py — all figure-drawing functions

Example
-------
  python Analysis.py \\
      --file ../1stBatch/stats_Run2_Ch0-200V_Ch1-200V_Ch2-200V_Ch3-200V_trig180V.root \\
      --outdir results/BV200 \\
      --pmax-cuts 70 70 60 100 \\
      --cfd-index 1 \\
      --cfd-unit ns
"""
from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List, Tuple

import numpy as np

from core.config import DUT_SHORT, VAR_INFO
from core.cuts import (
    get_delta_t_arrays,
    global_dt_range,
    make_edges,
    pmax_cut_mask_for_dut,
)
from core.fitting import (
    solve_individual_sigmas,
    solve_individual_sigmas_toy_mc,
)
from core.io_utils import (
    ensure_dir,
    load_analysis_arrays,
    print_delta_stats,
    print_fit_results,
    print_individual_sigmas,
    print_stats_table,
    save_delta_stats_csv,
    save_fit_results_csv,
    save_individual_sigmas_csv,
    save_stats_csv,
)
from core.plotting import (
    draw_three_var_canvas,
    plot_deltaT_fits_overlay,
    plot_deltaT_overlay,
)
from core.style import setup_cms_style


# ── CLI argument parsing ─────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="6-channel DUT timing post-processing from ROOT Analysis tree"
    )
    p.add_argument("--file",    required=True, help="Input ROOT file")
    p.add_argument("--tree",    default="Analysis", help="TTree name (default: Analysis)")
    p.add_argument("--outdir",  default="plots",    help="Output directory")

    p.add_argument("--bins",    type=int, default=100, help="Bins for pmax/tmax/area histograms")
    p.add_argument("--dt-bins", type=int, default=200, help="Bins for deltaT histograms")
    p.add_argument("--dt-range", type=float, nargs=2, default=None,
                   metavar=("LO_PS", "HI_PS"), help="Global x-range for deltaT plots [ps]")
    p.add_argument("--fit-half-range", type=float, default=1000.0,
                   help="Fit window half-width around each deltaT median [ps]")

    p.add_argument("--cfd-index", type=int, default=1,
                   help="CFD fraction index (e.g. 1 → CFD20 for [10,20,30,…] ordering)")
    p.add_argument("--cfd-unit", choices=["ns", "ps"], default="ns",
                   help="Unit stored in the cfd branch (output is always ps)")

    p.add_argument("--pmax-cuts", type=float, nargs=4,
                   default=[0.0, 0.0, 0.0, 0.0],
                   metavar=("DUT1", "DUT2", "DUT3", "DUT4"),
                   help="Lower pmax_fit cut for each DUT [mV]")
    p.add_argument("--pmax-cut-highs", type=float, nargs=4, default=None,
                   metavar=("DUT1", "DUT2", "DUT3", "DUT4"),
                   help="Upper pmax_fit cut for each DUT [mV]; use 0 for no upper cut")
    p.add_argument("--abs-tmax-cuts", type=float, nargs=4,
                   default=[0.0, 0.0, 0.0, 0.0],
                   metavar=("DUT1", "DUT2", "DUT3", "DUT4"),
                   help="Upper cut on |tmax_fit| for each DUT [ns]")
    p.add_argument("--delta-no-pmax-cuts", action="store_true",
                   help="Skip pairwise pmax cuts when computing deltaT")
    p.add_argument(
        "--area-fit-method",
        choices=["landau", "langau"],
        default="landau",
        help=(
            "Method for fitting area_new MPV on each DUT. "
            "'landau'  = pure Landau (Moyal PDF) — faster, single peak [default]. "
            "'langau'  = Landau-Gaussian convolution — slower, more accurate for smeared data."
        ),
    )
    return p.parse_args()


# ── Main analysis pipeline ───────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    setup_cms_style()
    ensure_dir(args.outdir)

    arrays   = load_analysis_arrays(args.file, args.tree)
    n_events = arrays["pmax_fit"].shape[0]
    print(f"[INFO] Loaded {n_events:,} events from {args.file}:{args.tree}")
    print(f"[INFO] cfd shape = {arrays['cfd'].shape}; "
          f"using cfd-index {args.cfd_index}, cfd-unit {args.cfd_unit}")
    print(f"[INFO] area_new MPV fit method: {args.area_fit_method}")

    edges: Dict[Tuple[int, str], np.ndarray] = {
        (dut, var): make_edges(
            arrays[var][:, dut],
            bins=args.bins,
            positive=bool(VAR_INFO[var]["positive"]),
        )
        for dut in range(4)
        for var in ["pmax_fit", "tmax_fit", "area_new"]
    }

    all_stats: List[Dict] = []
    before_mask = np.ones(n_events, dtype=bool)

    fit_method = args.area_fit_method

    # (1) Before cuts
    before_rows: List[Dict] = []
    for dut in range(4):
        out = os.path.join(args.outdir, f"01_before_cut_{DUT_SHORT[dut]}.png")
        before_rows.extend(
            draw_three_var_canvas(
                arrays, dut, before_mask, edges, "Before pmax cut", out,
                fit_method=fit_method,
            )
        )
    print_stats_table("Before pmax cut", before_rows)
    all_stats.extend(before_rows)

    # (2) After individual pmax cuts
    after_rows:  List[Dict] = []
    event_rows:  List[Dict] = []
    print("\n=== Event counts before/after individual pmax_fit cuts ===")
    print(f"{'DUT':<8} {'pmax low':>12} {'pmax high':>12} {'before':>10} {'after':>10}")
    print("-" * 62)
    for dut in range(4):
        after_mask = pmax_cut_mask_for_dut(
            arrays["pmax_fit"], arrays["tmax_fit"],
            dut, args.pmax_cuts, args.pmax_cut_highs, args.abs_tmax_cuts,
        )
        hi     = args.pmax_cut_highs[dut] if args.pmax_cut_highs is not None else 0.0
        hi_str = f"{hi:.6g}" if hi > 0 else "none"
        print(f"{DUT_SHORT[dut]:<8} {args.pmax_cuts[dut]:>12.6g} {hi_str:>12} "
              f"{n_events:>10d} {int(after_mask.sum()):>10d}")
        event_rows.append({
            "DUT": DUT_SHORT[dut],
            "pmax_cut_low":   args.pmax_cuts[dut],
            "pmax_cut_high":  hi,
            "before_entries": n_events,
            "after_entries":  int(after_mask.sum()),
        })
        out = os.path.join(args.outdir, f"02_after_pmax_cut_{DUT_SHORT[dut]}.png")
        after_rows.extend(
            draw_three_var_canvas(
                arrays, dut, after_mask, edges, "After pmax cut", out,
                fit_method=fit_method,
            )
        )
    print_stats_table("After pmax cut", after_rows)
    all_stats.extend(after_rows)

    save_stats_csv(os.path.join(args.outdir, "dut_variable_stats.csv"), all_stats)
    with open(os.path.join(args.outdir, "event_counts_before_after_pmax_cut.csv"),
              "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["DUT", "pmax_cut_low", "pmax_cut_high",
                           "before_entries", "after_entries"]
        )
        writer.writeheader()
        writer.writerows(event_rows)

    # (3) ΔT overlay
    require_pair_cuts = not args.delta_no_pmax_cuts
    dt_arrays = get_delta_t_arrays(
        arrays,
        cfd_index=args.cfd_index,
        cfd_unit=args.cfd_unit,
        pmax_cuts=args.pmax_cuts,
        pmax_cut_highs=args.pmax_cut_highs,
        abs_tmax_cuts=args.abs_tmax_cuts,
        require_pmax_cuts=require_pair_cuts,
    )
    dt_rng   = global_dt_range(dt_arrays, args.dt_range)
    cut_note = "pmax_fit cuts applied" if require_pair_cuts else "no pmax cuts"

    delta_rows = plot_deltaT_overlay(
        dt_arrays,
        out_path=os.path.join(args.outdir, "03_deltaT_overlay_CFD20.png"),
        bins=args.dt_bins,
        dt_range=dt_rng,
        title=f"DUT-DUT deltaT distributions  |  CFD index {args.cfd_index}  |  {cut_note}",
    )
    print_delta_stats(delta_rows)
    save_delta_stats_csv(os.path.join(args.outdir, "deltaT_stats.csv"), delta_rows)

    # (4) ΔT Gaussian fits
    fit_results = plot_deltaT_fits_overlay(
        dt_arrays,
        out_path=os.path.join(args.outdir, "04_deltaT_gaussian_fits_CFD20.png"),
        bins=args.dt_bins,
        dt_range=dt_rng,
        fit_half_range_ps=args.fit_half_range,
        title=rf"Gaussian fits to $\Delta T$  |  CFD index {args.cfd_index}  |  {cut_note}",
    )
    print_fit_results(fit_results)
    save_fit_results_csv(
        os.path.join(args.outdir, "deltaT_gaussian_fit_sigmas.csv"), fit_results
    )

    # (5) Individual DUT timing resolutions
    sigmas, variances, residuals = solve_individual_sigmas(fit_results)
    sigmas_mc, variances_mc, sigma_errors_mc, variance_errors_mc = (
        solve_individual_sigmas_toy_mc(fit_results, n_toys=2000, seed=12345)
    )
    print_individual_sigmas(
        sigmas, variances, residuals,
        sigma_errors=sigma_errors_mc,
        variance_errors=variance_errors_mc,
    )
    save_individual_sigmas_csv(
        os.path.join(args.outdir, "individual_DUT_sigmas.csv"),
        sigmas, variances,
        sigma_errors=sigma_errors_mc,
        variance_errors=variance_errors_mc,
    )
    print(f"\n[DONE] Plots and CSV summaries saved in: {args.outdir}")


if __name__ == "__main__":
    main()

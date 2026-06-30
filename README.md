# SPS June 2026 — LGAD Timing Analysis

Post-processing and plotting code for the **CMS LGAD timing beam test** at CERN SPS (June 2026).  
A 6-channel Lecroy oscilloscope recorded signals from 4 DUT channels (HPK sensors) and 2 trigger channels.  
Pre-processed ROOT `Analysis` TTrees are read to extract pulse properties and DUT-pair timing resolutions.

---

## What this code does

| Step | Output |
|------|--------|
| Load ROOT file | `pmax_fit`, `tmax_fit`, `area_new`, `cfd` arrays |
| Before / after pmax cut histograms | `01_before_cut_*.png`, `02_after_pmax_cut_*.png` |
| Landau (or Landau⊗Gaussian) fit on area → MPV, Q_coll | selectable via `--area-fit-method` |
| ΔT overlay for all 6 DUT pairs | `03_deltaT_overlay_CFD20.png` |
| Gaussian fit on each ΔT distribution | `04_deltaT_gaussian_fits_CFD20.png` |
| Individual timing resolutions σ_i | via σ_ij² = σ_i² + σ_j², Toy-MC errors |
| CSV summaries | `dut_variable_stats.csv`, `deltaT_gaussian_fit_sigmas.csv`, `individual_DUT_sigmas.csv` |

---

## Repository Structure

```
SPS_June_2026/
│
├── README.md
│
├── draw/                          # Analysis scripts
│   ├── Analysis.py                # ★ Main entry point — run this
│   │
│   ├── core/                      # Shared module package
│   │   ├── config.py              # Constants, DUT labels, colours, pair definitions
│   │   ├── models.py              # Dataclasses (Stat, FitResult, AreaLangauResult …)
│   │   ├── style.py               # CMS matplotlib style helpers
│   │   ├── io_utils.py            # ROOT loading, CSV saving, console printers
│   │   ├── cuts.py                # Event selection masks, ΔT array construction
│   │   ├── fitting.py             # Langau / Gaussian fits, σ_i extraction + Toy-MC
│   │   └── plotting.py            # All figure-drawing functions
│   │
│   └── graph_draw_script/         # Summary comparison plots (bias-voltage scans)
│       ├── graph_tb.py            # Timing resolution vs bias voltage
│       ├── graph_beta.py          # Beta-source collected charge
│       ├── graph_comp.py          # Batch comparison
│       └── graph_comp_prototype.py
│
└── BetaDAQ_Federico/              # DAQ control scripts (Lecroy oscilloscope)
```

---

## Requirements

```bash
pip install numpy uproot matplotlib scipy mplhep
```

---

## Quick Start

Run from the `draw/` directory.

```bash
cd draw/

# Pure Landau fit (default — faster)
python Analysis.py \
    --file ../1stBatch/stats_Run2_Ch0-200V_Ch1-200V_Ch2-200V_Ch3-200V_trig180V.root \
    --tree Analysis \
    --outdir results/BV200 \
    --bins 100 \
    --dt-bins 200 \
    --pmax-cuts 70 70 60 100 \
    --cfd-index 1 \
    --cfd-unit ns

# Landau-Gaussian convolution fit (slower, accounts for detector smearing)
python Analysis.py \
    --file ../1stBatch/stats_Run2_Ch0-200V_Ch1-200V_Ch2-200V_Ch3-200V_trig180V.root \
    --outdir results/BV200_langau \
    --pmax-cuts 70 70 60 100 \
    --area-fit-method langau
```

Output plots and CSV files will appear in `results/BV200/`.

---

## CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--file` | *(required)* | Input ROOT file path |
| `--tree` | `Analysis` | TTree name inside the ROOT file |
| `--outdir` | `plots` | Output directory for PNGs and CSVs |
| `--bins` | `100` | Histogram bins for pmax / tmax / area |
| `--dt-bins` | `200` | Histogram bins for ΔT distributions |
| `--dt-range LO HI` | auto | Fix x-axis range of ΔT plots [ps] |
| `--fit-half-range` | `1000` | Gaussian fit window = median ± value [ps] |
| `--cfd-index` | `1` | CFD fraction index (1 = CFD20 for [10 %, 20 %, 30 %, …] ordering) |
| `--cfd-unit` | `ns` | Unit stored in the cfd branch (`ns` or `ps`; output always ps) |
| `--pmax-cuts D1 D2 D3 D4` | `0 0 0 0` | Lower pmax_fit cut per DUT [mV] |
| `--pmax-cut-highs D1 D2 D3 D4` | none | Upper pmax_fit cut per DUT [mV]; `0` = no upper cut |
| `--abs-tmax-cuts D1 D2 D3 D4` | `0 0 0 0` | Upper cut on \|tmax_fit\| per DUT [ns] |
| `--delta-no-pmax-cuts` | off | Skip pairwise pmax cuts when computing ΔT |
| `--area-fit-method` | `landau` | Area MPV fit method: `landau` or `langau` (see below) |

### Area fit method: `--area-fit-method`

| Method | Description | Speed | Use when |
|--------|-------------|-------|----------|
| `landau` *(default)* | Pure Landau fit using the Moyal PDF. The `loc` parameter of the Moyal distribution is directly the MPV. | Fast | Standard beam test analysis; clean Landau spectra |
| `langau` | Landau-Gaussian convolution fit. Fits the additional Gaussian smearing width σ_G separately from the Landau width σ_L. | Slower (~9× more evaluations) | Data with significant electronic noise smearing; when σ_G is physically meaningful |

**CSV output columns for each method:**

| Column | `landau` | `langau` |
|--------|----------|---------|
| `landau_mpv` / `landau_mpv_err` | MPV from Moyal `loc` | MPV from convolution peak |
| `landau_sigma` / `landau_sigma_err` | Moyal scale (= Landau σ) | Landau width σ_L |
| `gauss_sigma` / `gauss_sigma_err` | `NaN` (not applicable) | Gaussian smearing σ_G |
| `collected_charge` / `collected_charge_err` | MPV / 4.7 [fC] | MPV / 4.7 [fC] |

**Fit parameters** (bins, quantile range, convolution grid) are in `core/config.py`:
```python
LANDAU_FIT_BINS      = 80
LANDAU_FIT_QUANTILES = (1.0, 99.0)   # pure Landau

AREA_FIT_BINS        = 80
AREA_FIT_QUANTILES   = (1.0, 97.0)   # Langau (tighter upper bound)
LANGAU_CONV_POINTS   = 600           # convolution grid points
```

---

## Output Files

| File | Description |
|------|-------------|
| `01_before_cut_DUT{1-4}.png` | pmax / tmax / area histograms, no cuts |
| `02_after_pmax_cut_DUT{1-4}.png` | Same after pmax cut, with Langau fit on area |
| `03_deltaT_overlay_CFD20.png` | All 6 DUT-pair ΔT distributions overlaid |
| `04_deltaT_gaussian_fits_CFD20.png` | 2×3 canvas of per-pair Gaussian fits |
| `dut_variable_stats.csv` | Mean / std / Langau MPV / Q_coll per DUT |
| `event_counts_before_after_pmax_cut.csv` | Event counts before and after each cut |
| `deltaT_stats.csv` | Raw ΔT mean and RMS per pair |
| `deltaT_gaussian_fit_sigmas.csv` | Gaussian fit μ, σ, σ_err, χ²/ndf per pair |
| `individual_DUT_sigmas.csv` | σ_i per DUT with Toy-MC uncertainties |

---

## Module Reference (`draw/core/`)

| Module | Responsibility |
|--------|----------------|
| `config.py` | All constants: DUT labels, colours, `PAIR_DEFS`, Landau / Langau fit parameters |
| `models.py` | `Stat`, `FitResult`, `AreaLandauResult`, `AreaLangauResult` dataclasses |
| `style.py` | `setup_cms_style()`, `add_cms_label()`, `style_axis_labels()` |
| `io_utils.py` | `load_analysis_arrays()`, `save_*_csv()`, `print_*()` |
| `cuts.py` | `pmax_cut_mask_for_dut()`, `get_delta_t_arrays()`, `make_edges()` |
| `fitting.py` | `fit_area_landau_mpv()`, `fit_area_langau_mpv()`, `fit_gaussian_deltaT()`, `solve_individual_sigmas()`, Toy-MC |
| `plotting.py` | `draw_three_var_canvas()`, `plot_deltaT_overlay()`, `plot_deltaT_fits_overlay()` |

To change **DUT labels or sensor mapping**, edit `core/config.py`:

```python
DUT_LABELS = ["HPK 1 (ch1)", "HPK 3 (ch2)", "HPK 17 (ch3)", "HPK 18 (ch4)"]
```

To change **fit parameters**, edit `core/config.py`:

```python
# Pure Landau fit
LANDAU_FIT_BINS      = 80
LANDAU_FIT_QUANTILES = (1.0, 99.0)

# Landau-Gaussian convolution fit
AREA_FIT_BINS        = 80
AREA_FIT_QUANTILES   = (1.0, 97.0)
LANGAU_CONV_POINTS   = 600
```

---

## DUT Channel Mapping (Batch 1)

| Channel | Sensor |
|---------|--------|
| ch1 (DUT1) | HPK 1 |
| ch2 (DUT2) | HPK 3 |
| ch3 (DUT3) | HPK 17 |
| ch4 (DUT4) | HPK 18 |
| ch5 (TRG1) | Trigger 1 |
| ch6 (TRG2) | Trigger 2 |

---

## Example Commands — Batch 1 BV scan

```bash
cd draw/

python Analysis.py \
    --file ../1stBatch/stats_Run2_Ch0-180V_Ch1-180V_Ch2-180V_Ch3-180V_trig180V.root \
    --tree Analysis --outdir results/BV180 \
    --bins 100 --dt-bins 200 \
    --pmax-cuts 70 70 60 70 \
    --cfd-index 1 --cfd-unit ns

python Analysis.py \
    --file ../1stBatch/stats_Run2_Ch0-195V_Ch1-195V_Ch2-195V_Ch3-195V_trig180V.root \
    --tree Analysis --outdir results/BV195 \
    --bins 100 --dt-bins 150 \
    --pmax-cuts 70 70 60 100 \
    --cfd-index 1 --cfd-unit ns

python Analysis.py \
    --file ../1stBatch/stats_Run2_Ch0-200V_Ch1-200V_Ch2-200V_Ch3-200V_trig180V.root \
    --tree Analysis --outdir results/BV200 \
    --bins 100 --dt-bins 150 \
    --pmax-cuts 70 70 60 100 \
    --cfd-index 1 --cfd-unit ns
```

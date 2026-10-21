"""
================================================================================
 Elevation-aware compound drought-heat extremes in the Upper Blue Nile Basin:
 station-based climatology, nonstationarity, and explainable ML attribution
================================================================================

Full, reproducible analysis pipeline for 17 daily rainfall/temperature stations
(1981-2025) in the Upper Blue Nile River Basin (UBNB), Ethiopia. Implements the
Q1-journal research design (compound drought-heat extremes + explainable ML,
see uploaded "Research ideas for Q1 journals" brief) at a scope matched to the
data actually available (daily P, Tmax, Tmin per station; no station lat/lon,
elevation, ENSO/IOD series, or RH/wind were used as index inputs).

Pipeline stages
----------------
 1. LOAD & QC        merge 17 station CSVs, harmonize columns, run physical-
                      range / temporal-consistency checks
 2. INDICES           TX90p (calendar-day, 1981-2010 base), monthly P and PET
                      (Hargreaves), SPI-3 (gamma), SPEI-3 (log-logistic),
                      compound drought-heat event (CDHE) flag and continuous
                      severity index (CDHI)
 3. TRENDS            modified Mann-Kendall + Sen's slope per station, compared
                      against the uncorrected MK test to quantify the effect of
                      serial autocorrelation (lag-1 ACF); Benjamini-Hochberg FDR
                      correction across stations; Pettitt change-point test
                      (station-level and basin-mean); sensitivity of trend
                      estimates to the 2001-2005 rainfall/heat excursion; basin-
                      mean Tmax-rainfall and Tmax-CDHE interannual correlations
 4. ML ATTRIBUTION    one-month-ahead CDHE classifier (Logistic Regression /
                      Random Forest / XGBoost) using ONLY lagged predictors
                      (t-1..t-3) plus calendar and static station terms -- no
                      same-month leakage from the variables that define the
                      label -- time-blocked train/val/test split, SHAP global
                      importance + dependence
 5. FIGURES            all report figures, including the basin correlation
                      scatter plots (PNG, 140 dpi)

FIGURE STYLE
------------
 * Font family : Times New Roman (falls back to Liberation Serif / DejaVu Serif)
 * Font weight : bold for every text element (titles, labels, ticks, legends,
                 annotations, colorbars)
 * Font sizes  : set INDIVIDUALLY for each figure in the FS dictionary below.
                 Edit the numbers in FS to resize any figure independently.

Run with:  python3 compound_drought_heat_pipeline.py
Expects the 17 station CSVs in RAW_DIR (Date, Rain_mm, Tmax_C, Tmin_C, ...).
Outputs parquet/csv intermediates + figures/ to OUT_DIR.
================================================================================
"""
import os
import glob
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
import pymannkendall as mk
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import acf
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, brier_score_loss
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import shap

RAW_DIR = r"C:/Users/YOGA/Desktop/Water/NASA_POWER_Selected1"   # 17 station CSVs
OUT_DIR = os.path.join(RAW_DIR, "analysis_outputs")             # intermediate data products
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

BASE_START, BASE_END = 1981, 2010   # WMO-style 30-yr climatological base period
LAT_DEG = 10.5                      # representative UBNB latitude for Hargreaves PET
                                     # (no station coordinates were provided in the source files)

# ------------------------------------------------------------------------------
# GLOBAL FONT STYLE: Times New Roman + bold everywhere
# ------------------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 140, "savefig.dpi": 140,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "figure.facecolor": "white",
    # font family
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    # font weight (bold for all text)
    "font.weight": "bold",
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "figure.titleweight": "bold",
    # keep minus signs compatible with Times New Roman
    "axes.unicode_minus": False,
})
CB = ["#1b5e91", "#c0392b", "#2e8b57", "#e08214", "#7b3294", "#666666"]

# ------------------------------------------------------------------------------
# PER-FIGURE FONT SIZES  (edit these numbers to resize each figure individually)
#   title  : axes title          label : x/y axis labels
#   tick   : tick labels         legend: legend text
#   annot  : text drawn on plot (bar values, markers, etc.)
#   cbar   : colorbar label/ticks
# ------------------------------------------------------------------------------
FS = {
    "fig1_basin_trend":      dict(title=14, label=13, tick=12, legend=11, annot=11, cbar=11),
    "fig2_station_heatmap":  dict(title=14, label=12, tick=11, legend=10, annot=16, cbar=12),
    "fig3_cdhe_by_station":  dict(title=14, label=13, tick=12, legend=10, annot=11, cbar=11),
    "fig4_example_station":  dict(title=14, label=12, tick=11, legend=10, annot=10, cbar=11),
    "fig5_cdhe_changepoint": dict(title=13, label=13, tick=12, legend=10, annot=11, cbar=11),
    "fig_basin_correlations":dict(title=13, label=12, tick=11, legend=10, annot=10, cbar=11),
    "fig6_ml_roc_pr":        dict(title=13, label=12, tick=11, legend=11, annot=10, cbar=11),
    "fig7_shap_summary":     dict(title=13, label=12, tick=11, legend=10, annot=10, cbar=11),
    "fig8_shap_dependence":  dict(title=13, label=12, tick=11, legend=10, annot=10, cbar=11),
}


def style_axes(ax, f):
    """Apply one figure's font sizes to an axes (title, labels, ticks, legend).
    Family/weight come from rcParams; we also force bold on tick labels."""
    ax.title.set_fontsize(f["title"])
    ax.xaxis.label.set_fontsize(f["label"])
    ax.yaxis.label.set_fontsize(f["label"])
    ax.tick_params(axis="both", labelsize=f["tick"])
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")
        lbl.set_fontfamily("serif")
    leg = ax.get_legend()
    if leg is not None:
        for t in leg.get_texts():
            t.set_fontsize(f["legend"])
            t.set_fontweight("bold")


def style_colorbar(cb, f, label=None):
    if label is not None:
        cb.set_label(label, fontsize=f["cbar"], fontweight="bold")
    cb.ax.tick_params(labelsize=f["cbar"])
    for lbl in cb.ax.get_yticklabels():
        lbl.set_fontweight("bold")


# ==============================================================================
# STAGE 1: LOAD & QC
# ==============================================================================
def load_and_qc():
    files = sorted(glob.glob(os.path.join(RAW_DIR, "*.csv")))
    print(f"[1/5] Loading {len(files)} station files...")
    frames = []
    for f in files:
        station = os.path.splitext(os.path.basename(f))[0]
        df = pd.read_csv(f)
        df["Date"] = pd.to_datetime(df["Date"])
        df["Station"] = station
        df = df.rename(columns={"Rain_mm": "P", "Tmax_C": "Tmax", "Tmin_C": "Tmin"})
        df = df[["Date", "Station", "P", "Tmax", "Tmin"]]
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    data["Tmean"] = (data["Tmax"] + data["Tmin"]) / 2.0
    data["Year"] = data["Date"].dt.year
    data["Month"] = data["Date"].dt.month
    data["DOY"] = data["Date"].dt.dayofyear

    qc = {
        "negative_rain": int((data["P"] < 0).sum()),
        "tmax_lt_tmin": int((data["Tmax"] < data["Tmin"]).sum()),
        "tmax_out_of_range": int(((data["Tmax"] < 0) | (data["Tmax"] > 50)).sum()),
        "tmin_out_of_range": int(((data["Tmin"] < -10) | (data["Tmin"] > 35)).sum()),
        "missing_any": int(data[["P", "Tmax", "Tmin"]].isna().sum().sum()),
        "duplicate_station_dates": int(data.duplicated(subset=["Station", "Date"]).sum()),
    }
    print("  QC summary (violations found):", qc)
    print(f"  {data.Station.nunique()} stations, {data.Date.min().date()} to {data.Date.max().date()}, "
          f"{len(data):,} station-days")
    return data


# ==============================================================================
# STAGE 2: CLIMATE INDICES  (TX90p, SPI-3, PET/SPEI-3, CDHE, CDHI)
# ==============================================================================
def compute_tx90p_thresholds(df):
    """Calendar-day 90th percentile of Tmax using a +-5 day window, base period only."""
    base = df[(df.Year >= BASE_START) & (df.Year <= BASE_END)].copy()
    thresholds = {}
    for station, g in base.groupby("Station"):
        tmax_by_doy = g.groupby("DOY")["Tmax"].apply(list).to_dict()
        doy_vals = {}
        for doy in range(1, 367):
            window_days = [((doy - 1 + d) % 366) + 1 for d in range(-5, 6)]
            vals = []
            for wd in window_days:
                vals.extend(tmax_by_doy.get(wd, []))
            if vals:
                doy_vals[doy] = np.percentile(vals, 90)
        thresholds[station] = doy_vals
    return thresholds


def hargreaves_pet_daily(df, lat_rad):
    """Hargreaves-Samani reference ET (mm/day) from Tmax, Tmin, Tmean and day-of-year."""
    doy = df["DOY"].values
    dr = 1 + 0.033 * np.cos(2 * np.pi * doy / 365)
    delta = 0.409 * np.sin(2 * np.pi * doy / 365 - 1.39)
    ws = np.arccos(np.clip(-np.tan(lat_rad) * np.tan(delta), -1, 1))
    Ra = (24 * 60 / np.pi) * 0.0820 * dr * (
        ws * np.sin(lat_rad) * np.sin(delta) + np.cos(lat_rad) * np.cos(delta) * np.sin(ws)
    )
    Ra_mm = Ra * 0.408
    tmean = df["Tmean"].values
    trange = np.clip(df["Tmax"].values - df["Tmin"].values, 0, None)
    return np.clip(0.0023 * Ra_mm * (tmean + 17.8) * np.sqrt(trange), 0, None)


def compute_spi(monthly_df, scale=3):
    """SPI via gamma-distribution fit, per station x calendar month, zero-inflated."""
    out = []
    for station, g in monthly_df.groupby("Station"):
        g = g.sort_values("Date").copy()
        g[f"P_acc{scale}"] = g["P"].rolling(scale, min_periods=scale).sum()
        g["SPI"] = np.nan
        for m in range(1, 13):
            sub = g[g["Month"] == m]
            vals = sub[f"P_acc{scale}"].dropna()
            if len(vals) < 10:
                continue
            zero_frac = (vals == 0).mean()
            nonzero = vals[vals > 0]
            if len(nonzero) >= 5:
                a, loc, scale_ = stats.gamma.fit(nonzero, floc=0)
                cdf = zero_frac + (1 - zero_frac) * stats.gamma.cdf(vals, a, loc=0, scale=scale_)
                cdf = np.clip(cdf, 1e-6, 1 - 1e-6)
                g.loc[vals.index, "SPI"] = stats.norm.ppf(cdf)
        out.append(g)
    return pd.concat(out, ignore_index=True)


def compute_spei(monthly_df, scale=3):
    """SPEI via log-logistic fit on accumulated water balance D = P - PET."""
    out = []
    for station, g in monthly_df.groupby("Station"):
        g = g.sort_values("Date").copy()
        g[f"D_acc{scale}"] = g["D"].rolling(scale, min_periods=scale).sum()
        g["SPEI"] = np.nan
        for m in range(1, 13):
            sub = g[g["Month"] == m]
            vals = sub[f"D_acc{scale}"].dropna()
            if len(vals) < 10:
                continue
            shift = -vals.min() + 1.0
            params = stats.fisk.fit(vals + shift)
            cdf = stats.fisk.cdf(vals + shift, *params)
            cdf = np.clip(cdf, 1e-6, 1 - 1e-6)
            g.loc[vals.index, "SPEI"] = stats.norm.ppf(cdf)
        out.append(g)
    return pd.concat(out, ignore_index=True)


def build_indices(data):
    print("[2/5] Computing TX90p, SPI-3, Hargreaves PET, SPEI-3, CDHE/CDHI...")
    tx90_thresh = compute_tx90p_thresholds(data)
    data["TX90p_thresh"] = data.apply(lambda r: tx90_thresh[r["Station"]].get(r["DOY"], np.nan), axis=1)
    data["hot_day"] = (data["Tmax"] > data["TX90p_thresh"]).astype(int)

    monthly = data.groupby(["Station", "Year", "Month"]).agg(
        P=("P", "sum"), Tmax_mean=("Tmax", "mean"), Tmin_mean=("Tmin", "mean"),
        Tmean_mean=("Tmean", "mean"), hot_days=("hot_day", "sum"), n_days=("hot_day", "count"),
    ).reset_index()
    monthly["hot_frac"] = monthly["hot_days"] / monthly["n_days"]
    monthly["Date"] = pd.to_datetime(monthly[["Year", "Month"]].assign(Day=1))
    monthly = monthly.sort_values(["Station", "Date"]).reset_index(drop=True)

    monthly = compute_spi(monthly, scale=3).rename(columns={"SPI": "SPI3"})

    lat_rad = np.radians(LAT_DEG)
    data["PET"] = hargreaves_pet_daily(data, lat_rad)
    pet_monthly = data.groupby(["Station", "Year", "Month"])["PET"].sum().reset_index()
    monthly = monthly.merge(pet_monthly, on=["Station", "Year", "Month"], how="left")
    monthly["D"] = monthly["P"] - monthly["PET"]
    monthly = compute_spei(monthly, scale=3).rename(columns={"SPEI": "SPEI3"})

    # Compound Drought-Heat Event: moderate-or-worse drought (SPI3<=-1) co-occurring
    # with a hot month (>=1/3 of days exceeding the calendar-day TX90p threshold)
    monthly["drought_flag"] = (monthly["SPI3"] <= -1).astype(int)
    monthly["heat_flag"] = (monthly["hot_frac"] >= 1 / 3).astype(int)
    monthly["CDHE"] = (monthly["drought_flag"] & monthly["heat_flag"]).astype(int)
    monthly["heat_z"] = monthly.groupby(["Station", "Month"])["hot_frac"].transform(
        lambda x: (x - x.mean()) / (x.std(ddof=0) + 1e-9))
    monthly["drought_sev"] = (-monthly["SPI3"]).clip(lower=0)
    monthly["heat_sev"] = monthly["heat_z"].clip(lower=0)
    monthly["CDHI"] = 0.5 * monthly["drought_sev"] + 0.5 * monthly["heat_sev"]

    monthly.to_parquet(os.path.join(OUT_DIR, "panel_monthly.parquet"))
    print(f"  CDHE base rate: {monthly['CDHE'].mean():.4f} of all station-months")
    return data, monthly


# ==============================================================================
# STAGE 3: TRENDS, FDR CORRECTION, CHANGE-POINT DETECTION
# ==============================================================================
def pettitt_test(x):
    """Non-parametric Pettitt (1979) change-point test. Returns (index, p-value)."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    Ut = np.array([np.sum(np.sign(x[:k + 1, None] - x[None, k + 1:])) for k in range(n - 1)])
    K = np.max(np.abs(Ut))
    k_idx = int(np.argmax(np.abs(Ut)))
    p = min(2 * np.exp((-6 * K ** 2) / (n ** 3 + n ** 2)), 1.0)
    return k_idx, p


def trend_analysis(monthly):
    print("[3/5] Trend (mMK + Sen's slope), FDR correction, Pettitt change-point...")
    annual = monthly.groupby(["Station", "Year"]).agg(
        P_annual=("P", "sum"), Tmax_annual=("Tmax_mean", "mean"), Tmin_annual=("Tmin_mean", "mean"),
        hot_days_annual=("hot_days", "sum"), CDHE_annual=("CDHE", "sum"),
        CDHI_annual=("CDHI", "mean"), SPEI3_annual=("SPEI3", "mean"),
    ).reset_index()
    annual.to_parquet(os.path.join(OUT_DIR, "panel_annual.parquet"))

    variables = {
        "P_annual": "Annual rainfall (mm)", "Tmax_annual": "Mean annual Tmax (°C)",
        "hot_days_annual": "Warm days per year (TX90p count)",
        "CDHE_annual": "Compound drought-heat months per year",
    }
    records = []
    detrended_std = {}  # station -> residual std of annual Tmax after removing the Theil-Sen trend
    for station, g in annual.groupby("Station"):
        g = g.sort_values("Year")
        yrs_arr = g["Year"].values
        tmax_slope, tmax_intercept = stats.theilslopes(g["Tmax_annual"].values, yrs_arr)[:2]
        tmax_resid = g["Tmax_annual"].values - (tmax_intercept + tmax_slope * yrs_arr)
        detrended_std[station] = float(np.std(tmax_resid, ddof=1))
        for var, label in variables.items():
            series = g[var].values
            res = mk.hamed_rao_modification_test(series)
            p_classic = mk.original_test(series).p          # uncorrected MK, for transparency
            lag1 = acf(series, nlags=1, fft=False)[1]       # lag-1 autocorrelation
            cp_idx, cp_p = pettitt_test(series)
            records.append({
                "Station": station, "Variable": var, "Label": label, "n_years": len(series),
                "trend": res.trend, "sen_slope": res.slope, "p_value": res.p,
                "p_classic": p_classic, "lag1_acf": lag1,
                "pettitt_cp_year": g["Year"].values[cp_idx], "pettitt_p": cp_p,
            })
    trend_df = pd.DataFrame(records)
    trend_df["p_fdr"] = np.nan
    trend_df["significant_fdr"] = False
    for var in variables:
        mask = trend_df["Variable"] == var
        reject, p_adj, _, _ = multipletests(trend_df.loc[mask, "p_value"].values, alpha=0.05, method="fdr_bh")
        trend_df.loc[mask, "p_fdr"] = p_adj
        trend_df.loc[mask, "significant_fdr"] = reject
    trend_df["significant_fdr_classic"] = False
    for var in variables:
        mask = trend_df["Variable"] == var
        trend_df.loc[mask, "significant_fdr_classic"] = multipletests(
            trend_df.loc[mask, "p_classic"].values, alpha=0.05, method="fdr_bh")[0]
    trend_df.to_csv(os.path.join(OUT_DIR, "trend_results.csv"), index=False)

    print("  Serial dependence and effect of the Hamed-Rao correction (FDR-significant stations):")
    for var, label in variables.items():
        sub = trend_df[trend_df["Variable"] == var]
        print(f"    {label}: median lag-1 ACF = {sub['lag1_acf'].median():.2f}; "
              f"uncorrected MK {int(sub['significant_fdr_classic'].sum())}/{len(sub)} -> "
              f"modified MK {int(sub['significant_fdr'].sum())}/{len(sub)}")

    # Per-station table (Sen's slope per decade + FDR-adjusted p) used for the station-level trend table
    pmean = annual.groupby("Station")["P_annual"].mean()
    st_tab = trend_df.pivot(index="Station", columns="Variable", values=["sen_slope", "p_fdr"])
    st_out = pd.DataFrame({
        "Tmax_per_decade": st_tab["sen_slope"]["Tmax_annual"] * 10, "Tmax_q": st_tab["p_fdr"]["Tmax_annual"],
        "warmdays_per_decade": st_tab["sen_slope"]["hot_days_annual"] * 10, "warmdays_q": st_tab["p_fdr"]["hot_days_annual"],
        "rain_mm_per_decade": st_tab["sen_slope"]["P_annual"] * 10, "rain_q": st_tab["p_fdr"]["P_annual"],
    })
    st_out["rain_pct_of_mean_per_decade"] = st_out["rain_mm_per_decade"] / pmean * 100
    st_out.to_csv(os.path.join(OUT_DIR, "station_trend_table.csv"))

    # Sensitivity of trend estimates to the basin-wide 2001-2005 excursion (Theil-Sen slope with true years,
    # Kendall tau with BH-FDR; uncorrected test used so that the scenarios are compared like-for-like)
    sens_rows = []
    for label, excl in [("all years", None), ("excluding 2001-2005", (2001, 2005)), ("excluding 2000-2006", (2000, 2006))]:
        for var in ["Tmax_annual", "hot_days_annual", "P_annual"]:
            sl, ps = [], []
            for st, g in annual.groupby("Station"):
                g = g.sort_values("Year")
                if excl:
                    g = g[~g["Year"].between(*excl)]
                sl.append(stats.theilslopes(g[var].values, g["Year"].values)[0])
                ps.append(stats.kendalltau(g["Year"].values, g[var].values)[1])
            sens_rows.append({"scenario": label, "variable": var, "n_stations": len(sl),
                              "n_positive_slope": int(np.sum(np.array(sl) > 0)),
                              "n_significant_fdr": int(multipletests(ps, alpha=0.05, method="fdr_bh")[0].sum()),
                              "median_slope_per_decade": float(np.median(sl) * 10)})
    sens_df = pd.DataFrame(sens_rows)
    sens_df.to_csv(os.path.join(OUT_DIR, "trend_sensitivity.csv"), index=False)
    print("  Sensitivity of station trends to the 2001-2005 excursion:")
    print(sens_df.to_string(index=False))

    for var, label in variables.items():
        sub = trend_df[trend_df["Variable"] == var]
        sig = sub[sub["significant_fdr"]]
        print(f"  {label}: {len(sig)}/{len(sub)} stations significant after FDR correction")

    basin_annual = annual.groupby("Year").agg(
        P_annual=("P_annual", "mean"), Tmax_annual=("Tmax_annual", "mean"),
        CDHE_annual=("CDHE_annual", "mean"), SPEI3_annual=("SPEI3_annual", "mean"),
    ).reset_index()
    basin_annual.to_csv(os.path.join(OUT_DIR, "basin_annual.csv"), index=False)

    print("  Basin-wide (17-station mean):")
    for var, label in [("P_annual", "rainfall"), ("Tmax_annual", "Tmax"), ("CDHE_annual", "CDHE frequency")]:
        series = basin_annual[var].values
        res = mk.hamed_rao_modification_test(series)
        cp_idx, cp_p = pettitt_test(series)
        cp_year = basin_annual["Year"].values[cp_idx]
        print(f"    {label}: trend={res.trend}, Sen slope={res.slope:.4f}/yr, p={res.p:.4f} | "
              f"Pettitt change-point={cp_year}, p={cp_p:.4f}")

    corr = basin_correlations(basin_annual)
    return annual, trend_df, basin_annual, corr


def basin_correlations(basin_annual):
    """Interannual coupling between basin-mean Tmax, rainfall, and CDHE frequency
    (supports the 'nonstationarity' discussion of whether warm years are systematically
    dry years, and whether dry years are systematically the years that compound with heat)."""
    r_tp, p_tp = stats.pearsonr(basin_annual["Tmax_annual"], basin_annual["P_annual"])
    r_tc, p_tc = stats.pearsonr(basin_annual["Tmax_annual"], basin_annual["CDHE_annual"] * 100)
    print(f"  Basin-mean interannual correlations (Pearson r, n={len(basin_annual)} years):")
    print(f"    Tmax vs rainfall:       r = {r_tp:.3f}, p = {p_tp:.2e}")
    print(f"    Tmax vs CDHE frequency: r = {r_tc:.3f}, p = {p_tc:.2e}")
    corr = {"r_tmax_rain": r_tp, "p_tmax_rain": p_tp, "r_tmax_cdhe": r_tc, "p_tmax_cdhe": p_tc}
    pd.Series(corr).to_csv(os.path.join(OUT_DIR, "basin_correlations.csv"))
    return corr


# ==============================================================================
# STAGE 4: EXPLAINABLE ML ATTRIBUTION (one-month-ahead CDHE precursor model)
# ==============================================================================
def ml_attribution(monthly):
    """
    Target: CDHE(t), defined from SPI3(t) and hot_frac(t) (Stage 2).
    Predictors: ONLY lagged (t-1..t-3) climate variables plus calendar/static
    station terms -- never the same-month SPI3/hot_frac used to build the
    label, so the model answers "what precursor conditions raise next-month
    compound-event risk", not "can the label predict itself".
    """
    print("[4/5] Explainable ML attribution (Logistic / RF / XGBoost + SHAP)...")
    monthly = monthly.sort_values(["Station", "Date"]).reset_index(drop=True)
    g = monthly.groupby("Station")
    for lag in [1, 2, 3]:
        monthly[f"SPI3_lag{lag}"] = g["SPI3"].shift(lag)
        monthly[f"SPEI3_lag{lag}"] = g["SPEI3"].shift(lag)
        monthly[f"hot_frac_lag{lag}"] = g["hot_frac"].shift(lag)
        monthly[f"P_lag{lag}"] = g["P"].shift(lag)
        monthly[f"Tmax_lag{lag}"] = g["Tmax_mean"].shift(lag)
    monthly["month_sin"] = np.sin(2 * np.pi * monthly["Month"] / 12)
    monthly["month_cos"] = np.cos(2 * np.pi * monthly["Month"] / 12)

    # Station climatological Tmax/P as a static elevation proxy (no station
    # coordinates/elevation were supplied in the source files)
    station_clim = monthly.groupby("Station").agg(
        station_mean_Tmax=("Tmax_mean", "mean"), station_mean_P=("P", "mean")).reset_index()
    monthly = monthly.merge(station_clim, on="Station", how="left")

    feature_cols = (
        [f"SPI3_lag{l}" for l in [1, 2, 3]] + [f"SPEI3_lag{l}" for l in [1, 2, 3]]
        + [f"hot_frac_lag{l}" for l in [1, 2, 3]] + [f"P_lag{l}" for l in [1, 2, 3]]
        + [f"Tmax_lag{l}" for l in [1, 2, 3]]
        + ["month_sin", "month_cos", "station_mean_Tmax", "station_mean_P", "Year"]
    )
    ml_df = monthly.dropna(subset=feature_cols + ["CDHE"]).copy()

    train = ml_df[ml_df["Year"] <= 2007]
    val = ml_df[(ml_df["Year"] >= 2008) & (ml_df["Year"] <= 2014)]
    test = ml_df[ml_df["Year"] >= 2015]
    print(f"  Train {train.Year.min()}-{train.Year.max()} (n={len(train)}), "
          f"Val {val.Year.min()}-{val.Year.max()} (n={len(val)}), "
          f"Test {test.Year.min()}-{test.Year.max()} (n={len(test)})")

    Xtr, ytr = train[feature_cols], train["CDHE"]
    Xval, yval = val[feature_cols], val["CDHE"]
    Xte, yte = test[feature_cols], test["CDHE"]
    scaler = StandardScaler().fit(Xtr)

    results = {}
    logit = LogisticRegression(max_iter=2000, class_weight="balanced")
    logit.fit(scaler.transform(Xtr), ytr)
    results["LogisticRegression"] = logit.predict_proba(scaler.transform(Xte))[:, 1]

    rf = RandomForestClassifier(n_estimators=400, max_depth=6, min_samples_leaf=10,
                                 class_weight="balanced", random_state=42, n_jobs=-1)
    rf.fit(Xtr, ytr)
    results["RandomForest"] = rf.predict_proba(Xte)[:, 1]

    pos_weight = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
    xgb_model = xgb.XGBClassifier(
        n_estimators=600, max_depth=4, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=5, reg_lambda=1.0, scale_pos_weight=pos_weight, eval_metric="aucpr",
        early_stopping_rounds=40, random_state=42)
    xgb_model.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
    results["XGBoost"] = xgb_model.predict_proba(Xte)[:, 1]

    rows = []
    for name, p in results.items():
        rows.append({"Model": name, "ROC_AUC": roc_auc_score(yte, p),
                     "PR_AUC": average_precision_score(yte, p), "Brier": brier_score_loss(yte, p),
                     "F1@0.5": f1_score(yte, (p >= 0.5).astype(int))})
    eval_df = pd.DataFrame(rows).sort_values("PR_AUC", ascending=False)
    eval_df.to_csv(os.path.join(OUT_DIR, "ml_evaluation.csv"), index=False)
    print("  Independent test-period (2015-2025) performance:")
    print(eval_df.to_string(index=False))

    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(Xte)
    shap_importance = pd.DataFrame({
        "feature": feature_cols, "mean_abs_shap": np.abs(shap_values).mean(axis=0)
    }).sort_values("mean_abs_shap", ascending=False)
    shap_importance.to_csv(os.path.join(OUT_DIR, "shap_importance.csv"), index=False)
    print("  Top-5 SHAP drivers:", shap_importance.head(5)["feature"].tolist())

    art = {"xgb_model": xgb_model, "rf": rf, "logit": logit, "scaler": scaler,
           "feature_cols": feature_cols, "Xte": Xte, "yte": yte, "shap_values": shap_values,
           "results": results, "eval_df": eval_df, "shap_importance": shap_importance}
    joblib.dump(art, os.path.join(OUT_DIR, "ml_artifacts.joblib"))
    return art


# ==============================================================================
# STAGE 5: FIGURES  (Times New Roman, bold, per-figure font sizes from FS)
# ==============================================================================
def make_figures(monthly, annual, trend_df, basin_annual, art, corr):
    print("[5/5] Generating figures...")

    # ---------------- Fig 1: basin Tmax and rainfall trend ----------------
    f = FS["fig1_basin_trend"]
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    ax = axes[0]
    ax.plot(basin_annual.Year, basin_annual.Tmax_annual, "-o", color=CB[1], ms=3, lw=1.3)
    yrs = basin_annual.Year.values
    tres = mk.hamed_rao_modification_test(basin_annual.Tmax_annual.values)
    t_sl, t_ic = stats.theilslopes(basin_annual.Tmax_annual.values, yrs)[:2]
    ax.plot(yrs, t_ic + t_sl * yrs, "--", color="green", lw=1,
            label=f"Sen's slope = {t_sl:.3f} °C/yr (modified MK p = {tres.p:.3f})")
    ax.axvline(1999, color=CB[4], ls=":", lw=1.5, label="Pettitt change-point (1999)")
    ax.set_ylabel("Mean annual Tmax (°C)")
    ax.set_title("Basin-mean (17-station) annual maximum temperature and rainfall, 1981–2025")
    ax.legend(loc="upper left",  frameon=False)
    style_axes(ax, f)
    ax = axes[1]
    ax.bar(basin_annual.Year, basin_annual.P_annual, color=CB[0], alpha=0.75, width=0.8)
    pres = mk.hamed_rao_modification_test(basin_annual.P_annual.values)
    p_sl, p_ic = stats.theilslopes(basin_annual.P_annual.values, yrs)[:2]
    ax.plot(yrs, p_ic + p_sl * yrs, "--", color="green", lw=1,
            label=f"Sen's slope = {p_sl:.1f} mm/yr (n.s., modified MK p = {pres.p:.2f})")
    ax.axvline(2012, color=CB[4], ls=":", lw=1.5, label="Pettitt change-point (2012, n.s.)")
    ax.set_ylim(0, 2700)   # headroom so the legend does not overlap the bars
    ax.set_ylabel("Mean annual rainfall (mm)"); ax.set_xlabel("Year")
    ax.legend(loc="upper left",ncol=2, frameon=False)
    style_axes(ax, f)
    plt.tight_layout(); plt.savefig(f"{FIG_DIR}/fig1_basin_tmax_rainfall_trend.png"); plt.close()

    # ---------------- Fig 2: station trend heatmap ----------------
    # f = FS["fig2_station_heatmap"]
    # col_order = ["Annual rainfall (mm)", "Mean annual Tmax (°C)",
    #              "Warm days per year (TX90p count)", "Compound drought-heat months per year"]
    # pivot_slope = trend_df.pivot(index="Station", columns="Label", values="sen_slope")[col_order]
    # pivot_sig = trend_df.pivot(index="Station", columns="Label", values="significant_fdr")[col_order]
    # pivot_slope = pivot_slope.loc[pivot_slope["Mean annual Tmax (°C)"].sort_values(ascending=False).index]
    # pivot_sig = pivot_sig.loc[pivot_slope.index]
    # fig, ax = plt.subplots(figsize=(8.5, 7.2))
    # norm_data = pivot_slope.apply(lambda c: (c - c.mean()) / (c.std() + 1e-9))
    # im = ax.imshow(norm_data.values, cmap="RdBu_r", vmin=-2.5, vmax=2.5, aspect="auto")
    # ax.set_xticks(range(len(col_order)))
    # ax.set_xticklabels([c.replace(" (", "\n(") for c in col_order], rotation=0, ha="right")
    # ax.set_yticks(range(len(pivot_slope.index))); ax.set_yticklabels(pivot_slope.index)
    # for i in range(pivot_sig.shape[0]):
    #     for j in range(pivot_sig.shape[1]):
    #         if pivot_sig.values[i, j]:
    #             ax.text(j, i, "*", ha="center", va="center", color="black", fontsize=f["annot"])
    # ax.set_title("Station-wise Sen's slope (column-standardized)\n"
    #               "• = significant after Benjamini–Hochberg FDR (p<0.05)")
    # ax.grid(False)
    # style_axes(ax, f)
    # cb = plt.colorbar(im, ax=ax, shrink=0.95, pad=0.03)
    # style_colorbar(cb, f, label="z-score of Sen's slope within variable")
    # plt.subplots_adjust(bottom=0.22, right=0.98, left=0.16, top=0.88)
    # plt.savefig(f"{FIG_DIR}/fig2_station_trend_heatmap.png"); plt.close()
        # ---------------- Fig 2: station trend heatmap ----------------
    f = FS["fig2_station_heatmap"]
    col_order = ["Annual rainfall (mm)", "Mean annual Tmax (°C)",
                 "Warm days per year (TX90p count)", "Compound drought-heat months per year"]

    # Explicit two-line x-axis labels
    xlabels_2line = {
        "Annual rainfall (mm)":                  "Annual rainfall\n(mm)",
        "Mean annual Tmax (°C)":                 "Mean annual\nTmax (°C)",
        "Warm days per year (TX90p count)":      "Warm days/year\n(TX90p count)",
        "Compound drought-heat months per year": "Compound drought-heat\nmonths per year",
    }

    pivot_slope = trend_df.pivot(index="Station", columns="Label", values="sen_slope")[col_order]
    pivot_sig = trend_df.pivot(index="Station", columns="Label", values="significant_fdr")[col_order]
    pivot_slope = pivot_slope.loc[pivot_slope["Mean annual Tmax (°C)"].sort_values(ascending=False).index]
    pivot_sig = pivot_sig.loc[pivot_slope.index]

    fig, ax = plt.subplots(figsize=(10, 7.2))
    norm_data = pivot_slope.apply(lambda c: (c - c.mean()) / (c.std() + 1e-9))
    im = ax.imshow(norm_data.values, cmap="RdBu_r", vmin=-2.5, vmax=2.5, aspect="auto")

    ax.set_xticks(range(len(col_order)))
    ax.set_xticklabels([xlabels_2line[c] for c in col_order], rotation=0, ha="center")
    ax.set_yticks(range(len(pivot_slope.index)))
    ax.set_yticklabels(pivot_slope.index)

    for i in range(pivot_sig.shape[0]):
        for j in range(pivot_sig.shape[1]):
            if pivot_sig.values[i, j]:
                ax.text(j, i, "*", ha="center", va="center", color="black", fontsize=f["annot"])

    ax.set_title("Sen's slope (column-standardized) * = significant after Benjamini–Hochberg FDR (p<0.05)")
    ax.grid(False)
    style_axes(ax, f)   # keep after set_xticklabels so bold font and size apply

    cb = plt.colorbar(im, ax=ax,  pad=0.03) #\n" shrink=1,
    style_colorbar(cb, f, label="z-score of Sen's slope within variable")

    plt.subplots_adjust(bottom=0.22, right=0.98, left=0.16, top=0.88)
    plt.savefig(f"{FIG_DIR}/fig2_station_trend_heatmap.png")
    plt.close()

    # ---------------- Fig 3: CDHE frequency by station ----------------
    f = FS["fig3_cdhe_by_station"]
    cdhe_rank = monthly.groupby("Station")["CDHE"].mean().sort_values(ascending=False) * 100
    fig, ax = plt.subplots(figsize=(7.5, 4))
    bars = ax.barh(cdhe_rank.index[::-1], cdhe_rank.values[::-1], color=CB[1])
    ax.set_xlabel("Compound drought-heat months (% of all months, 1981-2025)")
    ax.set_title("")#Compound drought-heat event frequency by station
    for b, v in zip(bars, cdhe_rank.values[::-1]):
        ax.text(v + 0.05, b.get_y() + b.get_height() / 2, f"{v:.1f}%", va="center",
                fontsize=f["annot"], fontweight="bold")
    style_axes(ax, f)
    plt.tight_layout(); plt.savefig(f"{FIG_DIR}/fig3_cdhe_frequency_by_station.png"); plt.close()

    # ---------------- Fig 4: example station ----------------
    f = FS["fig4_example_station"]
    st = "Assosa"
    sub = monthly[monthly.Station == st].sort_values("Date")
    fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(sub.Date, sub.SPI3, color=CB[0], lw=0.9)
    axes[0].axhline(-1, color="grey", ls="--", lw=0.8)
    axes[0].fill_between(sub.Date, sub.SPI3, -1, where=(sub.SPI3 <= -1), color=CB[0], alpha=0.3)
    axes[0].set_ylabel("SPI-3")
    axes[0].set_title(f"{st}: SPI-3, warm-day fraction, and compound drought-heat events (1981–2025)")
    axes[1].plot(sub.Date, sub.hot_frac, color=CB[1], lw=0.9)
    axes[1].axhline(1 / 3, color="grey", ls="--", lw=0.8)
    axes[1].set_ylabel("Warm-day fraction (TX90p)")
    axes[2].scatter(sub.loc[sub.CDHE == 1, "Date"], [1] * (sub.CDHE == 1).sum(),
                     color=CB[3], marker="|", s=200)
    axes[2].set_ylim(0.5, 1.5); axes[2].set_yticks([]); axes[2].set_ylabel("CDHE month")
    axes[2].set_xlabel("Year")
    axes[2].xaxis.set_major_locator(mdates.YearLocator(5))
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for ax in axes:
        style_axes(ax, f)
    plt.tight_layout(); plt.savefig(f"{FIG_DIR}/fig4_example_station_{st}.png"); plt.close()

    # ---------------- Fig 5: basin CDHE change-point ----------------
    f = FS["fig5_cdhe_changepoint"]
    basin_cdhe = monthly.groupby("Year")["CDHE"].mean() * 100
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.bar(basin_cdhe.index, basin_cdhe.values, color=CB[0], alpha=0.8)
    pre, post = basin_cdhe[basin_cdhe.index <= 1997].mean(), basin_cdhe[basin_cdhe.index > 1997].mean()
    ax.axvline(1997.5, color=CB[4], ls=":", lw=1.6)
    ax.hlines(pre, basin_cdhe.index.min(), 1997.5, color="red", lw=1.5)
    ax.hlines(post, 1997.5, basin_cdhe.index.max(), color="black", lw=1.5)
    ax.set_ylabel("Basin-mean CDHE(% of station-months)"); ax.set_xlabel("Year")
    ax.set_title(f"Basin-wide compound drought-heat frequency: Pettitt change-point at 1997\n"
                 f"(pre-1997 mean = {pre:.1f}%, post-1997 mean = {post:.1f}%)")
    style_axes(ax, f)
    plt.tight_layout(); plt.savefig(f"{FIG_DIR}/fig5_basin_cdhe_changepoint.png"); plt.close()

    # ---------------- Basin correlations (Tmax vs rainfall / CDHE) ----------------
    f = FS["fig_basin_correlations"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    x = basin_annual["Tmax_annual"].values
    xs = np.linspace(x.min(), x.max(), 50)

    ax = axes[0]
    y = basin_annual["P_annual"].values
    ax.scatter(x, y, color=CB[0], s=28, alpha=0.85)
    b, a = np.polyfit(x, y, 1)
    ax.plot(xs, a + b * xs, "--", color="red", lw=1.3)
    ax.set_xlabel("Basin-mean annual Tmax (°C)")
    ax.set_ylabel("Basin-mean annual rainfall (mm)")
    ax.set_title(f"(a) Tmax vs. rainfall\nr = {corr['r_tmax_rain']:.2f}, "
                 f"p < 0.0001, n = {len(basin_annual)} years")
    style_axes(ax, f)

    ax = axes[1]
    y2 = basin_annual["CDHE_annual"].values * 100
    ax.scatter(x, y2, color=CB[1], s=28, alpha=0.85)
    b2, a2 = np.polyfit(x, y2, 1)
    ax.plot(xs, a2 + b2 * xs, "--", color="black", lw=1.3)
    ax.set_xlabel("Basin-mean annual Tmax (°C)")
    ax.set_ylabel("Basin-mean CDHE frequency\n(% of station-months)")
    ax.set_title(f"(b) Tmax vs. CDHE frequency\nr = {corr['r_tmax_cdhe']:.2f}, "
                 f"p < 0.0001, n = {len(basin_annual)} years")
    style_axes(ax, f)
    plt.tight_layout(); plt.savefig(f"{FIG_DIR}/fig_basin_correlations.png"); plt.close()

    # ---------------- Fig 6: ROC and PR curves ----------------
    f = FS["fig6_ml_roc_pr"]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    for i, (name, p) in enumerate(art["results"].items()):
        fpr, tpr, _ = roc_curve(art["yte"], p)
        prec, rec, _ = precision_recall_curve(art["yte"], p)
        axes[0].plot(fpr, tpr, color=CB[i], lw=1.6, label=name)
        axes[1].plot(rec, prec, color=CB[i], lw=1.6, label=name)
    axes[0].plot([0, 1], [0, 1], "--", color="grey", lw=0.8)
    axes[0].set_xlabel("False positive rate"); axes[0].set_ylabel("True positive rate")
    axes[0].set_title("ROC—held-out test period (2015–2025)"); axes[0].legend(frameon=False)
    base_rate = art["yte"].mean()
    axes[1].axhline(base_rate, color="grey", ls="--", lw=0.8, label=f"No-skill (base rate={base_rate:.3f})")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall—held-out test period"); axes[1].legend(frameon=False)
    for ax in axes:
        style_axes(ax, f)
    plt.tight_layout(); plt.savefig(f"{FIG_DIR}/fig6_ml_roc_pr_curves.png"); plt.close()

    # ---------------- Fig 7: SHAP summary ----------------
    f = FS["fig7_shap_summary"]
    fig = plt.figure(figsize=(10, 5.2))
    shap.summary_plot(art["shap_values"], art["Xte"], show=False, plot_size=None)
    plt.title("SHAP summary — one-month-ahead compound drought-heat prediction (XGBoost)",
              fontsize=f["title"], fontweight="bold")
    for ax in fig.axes:           # main axes + SHAP colorbar axes
        ax.tick_params(axis="both", labelsize=f["tick"])
        ax.xaxis.label.set_fontsize(f["label"]); ax.xaxis.label.set_fontweight("bold")
        ax.yaxis.label.set_fontsize(f["cbar"]); ax.yaxis.label.set_fontweight("bold")
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontweight("bold"); lbl.set_fontfamily("serif")
    plt.tight_layout(); plt.savefig(f"{FIG_DIR}/fig7_shap_summary.png"); plt.close()

    # ---------------- Fig 8: SHAP dependence ----------------
    f = FS["fig8_shap_dependence"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    top2 = art["shap_importance"]["feature"].values[:2]
    for ax, feat in zip(axes, top2):
        idx = art["feature_cols"].index(feat)
        sc = ax.scatter(art["Xte"][feat], art["shap_values"][:, idx],
                         c=art["Xte"]["hot_frac_lag1"], cmap="RdYlBu_r", s=10, alpha=0.7)
        ax.axhline(0, color="grey", lw=0.7); ax.set_xlabel(feat); ax.set_ylabel(f"SHAP value for {feat}")
        cb = plt.colorbar(sc, ax=ax, shrink=0.85)
        style_colorbar(cb, f, label="hot_frac_lag1")
    axes[0].set_title("SHAP dependence: drought persistence")
    axes[1].set_title("SHAP dependence: 1-month-lag SPI-3")
    for ax in axes:
        style_axes(ax, f)
    plt.tight_layout(); plt.savefig(f"{FIG_DIR}/fig8_shap_dependence.png"); plt.close()
    print(f"  Saved 9 figures to {FIG_DIR}")


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    data = load_and_qc()
    data, monthly = build_indices(data)
    annual, trend_df, basin_annual, corr = trend_analysis(monthly)
    art = ml_attribution(monthly)
    make_figures(monthly, annual, trend_df, basin_annual, art, corr)
    print("\nPipeline complete.")
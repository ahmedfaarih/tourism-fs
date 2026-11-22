#!/usr/bin/env python3
"""Prophet baseline for Maldives tourist arrivals, as an additional baseline
alongside the SARIMAX model in ../code/model_baseline_demand.py (per supervisor
feedback asking us to also try Prophet). Self-contained: reads the same shared
dataset but duplicates its own small scoring helper rather than importing
across directories, so this folder can be run/shared independently.

Same disclosed limitation as the SARIMAX baseline: flight/FX regressors use
realized values in the test window, not independently forecast ones -- not yet
a genuine ex-ante forecast.

Variants mirror the SARIMAX script's three regressor sets, though Prophet's
built-in yearly seasonality (Fourier terms) already covers month-level effects
on its own, so 'demand_parsimonious' here just adds a high/low-season regressor
on top rather than replacing monthly dummies (Prophet has none to replace) --
included for a like-for-like comparison across the two modeling approaches, not
because it resolves a collinearity issue the way it does for the SARIMAX model.

Runs both the single 12-month holdout and the year-by-year expanding-window
backtest, same as model_baseline_demand.py / model_baseline_yearly_backtest.py.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
from prophet import Prophet  # noqa: E402  (import after log-level set to quiet its own setup logging)

REPO_ROOT = Path(__file__).parent.parent
DATA = REPO_ROOT / "model_outputs" / "tourism_model_dataset.csv"
OUT_DIR = Path(__file__).parent / "model_outputs"
OUT_DIR.mkdir(exist_ok=True)

COVID_CLOSURE_MONTHS = {(2020, 4), (2020, 5), (2020, 6)}
HIGH_SEASON_MONTHS = {11, 12, 1, 2, 3, 4}  # Nov-Apr, standard Maldives tourism peak season
TEST_MONTHS = 12
FLIGHT_COL = "Total number of arrival flights"
FX_COL = "MVR per USD (average)"
BED_COL = "total_beds"
VARIANTS = ["demand_only", "demand_parsimonious", "supply_augmented"]
BACKTEST_YEARS = range(2017, 2026)  # 2026 excluded: partial year


def regressors_for(variant):
    base = ["covid_closure", FLIGHT_COL, FX_COL]
    if variant == "demand_only":
        return base
    if variant == "demand_parsimonious":
        return base + ["high_season"]
    if variant == "supply_augmented":
        return base + [BED_COL]
    raise ValueError(f"unknown variant: {variant!r}")


def load_series() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    df["ds"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=1))
    df = df.sort_values("ds").reset_index(drop=True)
    df["y"] = df["total_arrivals"]
    df["covid_closure"] = [
        1 if (y, m) in COVID_CLOSURE_MONTHS else 0
        for y, m in zip(df["year"], df["month"])
    ]
    df["high_season"] = df["month"].isin(HIGH_SEASON_MONTHS).astype(int)
    # Flight-movement data only exists from 2012-01 onward; truncate to that, same as the SARIMAX script.
    first_valid = df.loc[df[FLIGHT_COL].notna(), "ds"].min()
    df = df.loc[df["ds"] >= first_valid].reset_index(drop=True)
    return df


def score(actual, pred, label):
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    err = actual - pred
    nonzero = actual != 0  # MAPE undefined for zero-arrival months (e.g. COVID closure)
    mape = (np.abs(err[nonzero]) / actual[nonzero]).mean() * 100
    mae = np.abs(err).mean()
    rmse = np.sqrt((err ** 2).mean())
    note = "" if nonzero.all() else f"  (MAPE excludes {(~nonzero).sum()} zero-arrival month(s))"
    print(f"{label}: MAPE={mape:.2f}%  MAE={mae:.0f}  RMSE={rmse:.0f}{note}")
    return mape, mae, rmse


def fit_predict(train, test, variant):
    regs = regressors_for(variant)
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    for r in regs:
        m.add_regressor(r)
    m.fit(train[["ds", "y"] + regs])
    forecast = m.predict(test[["ds"] + regs])
    return forecast["yhat"].values


def main():
    df = load_series()
    train = df.iloc[:-TEST_MONTHS]
    test = df.iloc[-TEST_MONTHS:]
    print(f"Train: {train['ds'].iloc[0].date()} to {train['ds'].iloc[-1].date()} ({len(train)} months)")
    print(f"Test:  {test['ds'].iloc[0].date()} to {test['ds'].iloc[-1].date()} ({len(test)} months)")

    actual = test["y"].values
    naive = df["y"].shift(12).iloc[-TEST_MONTHS:].values

    print("\n=== Holdout evaluation (last 12 months), all variants ===")
    metrics_rows = []
    for variant in VARIANTS:
        pred = fit_predict(train, test, variant)
        mape, mae, rmse = score(actual, pred, f"Prophet[{variant}]")
        metrics_rows.append({"model": f"prophet_{variant}", "mape": mape, "mae": mae, "rmse": rmse})
    naive_mape, naive_mae, naive_rmse = score(actual, naive, "Seasonal-naive (same month last year)")
    metrics_rows.append({"model": "seasonal_naive", "mape": naive_mape, "mae": naive_mae, "rmse": naive_rmse})
    pd.DataFrame(metrics_rows).to_csv(OUT_DIR / "prophet_eval_metrics.csv", index=False)
    print(f"Wrote {OUT_DIR / 'prophet_eval_metrics.csv'}")

    print("\n=== Year-by-year expanding-window backtest, all variants ===")
    yearly_rows = []
    for year in BACKTEST_YEARS:
        fold_start = pd.Timestamp(year=year, month=1, day=1)
        fold_end = pd.Timestamp(year=year, month=12, day=1)
        fold_df = df[(df["ds"] >= fold_start) & (df["ds"] <= fold_end)]
        if len(fold_df) != 12:
            print(f"Skipping {year}: not enough data.")
            continue
        fold_train = df[df["ds"] < fold_start]
        if len(fold_train) < 36:
            print(f"Skipping {year}: train={len(fold_train)} months (too little data).")
            continue
        fold_actual = fold_df["y"].values
        fold_naive = df["y"].shift(12).loc[fold_df.index].values
        for variant in VARIANTS:
            pred = fit_predict(fold_train, fold_df, variant)
            mape, mae, rmse = score(fold_actual, pred, f"{year} [prophet_{variant}]")
            yearly_rows.append({"year": year, "model": f"prophet_{variant}", "mape": mape, "mae": mae, "rmse": rmse})
        naive_mape, naive_mae, naive_rmse = score(fold_actual, fold_naive, f"{year} [seasonal_naive]")
        yearly_rows.append({"year": year, "model": "seasonal_naive", "mape": naive_mape, "mae": naive_mae, "rmse": naive_rmse})
        print()

    yearly_result = pd.DataFrame(yearly_rows)
    print(yearly_result.pivot(index="year", columns="model", values="mape").round(2).to_string())
    out_path = OUT_DIR / "prophet_yearly_backtest.csv"
    yearly_result.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

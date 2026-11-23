#!/usr/bin/env python3
"""Actual vs. predicted, across the full 2017-2025 expanding-window backtest -- the
visual complement to the MAPE tables. Re-fits each fold the same way the backtest
scripts do (no saved predictions to reuse) but only for the two strongest performers
identified so far -- the season-only SARIMAX variant (demand_parsimonious) and the
original Prophet variant (prophet_demand_only) -- plus seasonal-naive, to keep the
chart readable. Stitches each fold's 12-month forecast into one continuous line so the
whole backtest period is visible in a single plot.
"""
import logging
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.statespace.sarimax import SARIMAX

from model_baseline_demand import FLIGHT_COL, exog_cols, grid_search_order, load_series

logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
from prophet import Prophet  # noqa: E402

ROOT = Path(__file__).parent.parent
OUT_DIR = ROOT / "model_outputs"
BACKTEST_YEARS = range(2017, 2026)

SARIMAX_VARIANT = "demand_parsimonious"  # season-only model -- most competitive SARIMAX variant
PROPHET_VARIANT_REGRESSORS = ["covid_closure", FLIGHT_COL, "MVR per USD (average)"]  # prophet_demand_only


def sarimax_predictions(df, d=0):
    """One fixed order (selected once on full history), re-fit per fold, forecast collected."""
    exog_full = df[exog_cols(SARIMAX_VARIANT)]
    table = grid_search_order(df["log_arrivals"], exog_full, d=d)
    order = (int(table.iloc[0]["p"]), d, int(table.iloc[0]["q"]))
    print(f"SARIMAX ({SARIMAX_VARIANT}) fixed order: {order}")

    preds = {}
    for year in BACKTEST_YEARS:
        fold_start = pd.Timestamp(year=year, month=1, day=1)
        fold_end = pd.Timestamp(year=year, month=12, day=1)
        if fold_end not in df.index:
            continue
        train = df.loc[df.index < fold_start]
        test = df.loc[(df.index >= fold_start) & (df.index <= fold_end)]
        if len(train) < 36 or len(test) != 12:
            continue
        fit = SARIMAX(
            train["log_arrivals"], exog=train[exog_cols(SARIMAX_VARIANT)], order=order,
            enforce_stationarity=False, enforce_invertibility=False,
        ).fit(disp=False)
        pred = np.exp(fit.get_forecast(steps=12, exog=test[exog_cols(SARIMAX_VARIANT)]).predicted_mean)
        for dt, val in pred.items():
            preds[dt] = val
    return pd.Series(preds).sort_index()


def prophet_predictions(df):
    pdf = df.reset_index().rename(columns={"date": "ds", "total_arrivals": "y"})
    preds = {}
    for year in BACKTEST_YEARS:
        fold_start = pd.Timestamp(year=year, month=1, day=1)
        fold_end = pd.Timestamp(year=year, month=12, day=1)
        fold_df = pdf[(pdf["ds"] >= fold_start) & (pdf["ds"] <= fold_end)]
        if len(fold_df) != 12:
            continue
        train = pdf[pdf["ds"] < fold_start]
        if len(train) < 36:
            continue
        m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
        for r in PROPHET_VARIANT_REGRESSORS:
            m.add_regressor(r)
        m.fit(train[["ds", "y"] + PROPHET_VARIANT_REGRESSORS])
        forecast = m.predict(fold_df[["ds"] + PROPHET_VARIANT_REGRESSORS])
        for dt, val in zip(fold_df["ds"], forecast["yhat"].values):
            preds[dt] = val
    return pd.Series(preds).sort_index()


def main():
    df = load_series()
    first_valid = df[FLIGHT_COL].first_valid_index()
    df = df.loc[df.index >= first_valid]

    sarimax_pred = sarimax_predictions(df)
    prophet_pred = prophet_predictions(df)
    naive = df["total_arrivals"].shift(12).loc[sarimax_pred.index]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df.index, df["total_arrivals"], color="black", linewidth=1.2, label="Actual arrivals")
    ax.plot(sarimax_pred.index, sarimax_pred, color="#1f77b4", linewidth=1.6,
            marker="o", markersize=3, label=f"Predicted -- Season-only (SARIMAX)")
    ax.plot(prophet_pred.index, prophet_pred, color="#d62728", linewidth=1.6,
            marker="o", markersize=3, label=f"Predicted -- Original (Prophet)")
    ax.plot(naive.index, naive, color="grey", linestyle="--", linewidth=1.2, label="Seasonal-naive")
    ax.axvspan(sarimax_pred.index[0], df.index[-1], color="grey", alpha=0.06)
    ax.set_title("Actual vs. predicted arrivals across the 2017-2025 expanding-window backtest\n"
                 "(shaded region = out-of-sample forecasts, one fold per year, stitched together)")
    ax.set_ylabel("Monthly arrivals")
    ax.legend(loc="upper left")
    fig.tight_layout()
    out_path = OUT_DIR / "actual_vs_predicted_backtest.png"
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

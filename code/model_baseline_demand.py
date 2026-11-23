#!/usr/bin/env python3
"""Demand-side ARIMA baseline for Maldives tourist arrivals, benchmarked
against the MMA/Shuaib (2021) model structure. Uses realized (not forecast)
flight/exchange-rate values, so it isn't yet usable for a forward forecast.

Runs three regressor variants side by side:
  demand_only         -- 11 monthly dummies + covid + flights + FX (original baseline)
  demand_parsimonious -- a single high/low-season dummy instead of the 11 monthly
                         dummies (can't just add it alongside them -- it's a linear
                         function of the month dummies, so together they'd be collinear)
  supply_augmented    -- demand_only regressors + total_beds (national bed capacity),
                         the actual supply-side regressor this thesis's RQ is about
"""
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Grid-search tries (p, q) combinations that are expected not to all converge; the
# resulting ConvergenceWarning is redundant noise since the printed grid already has
# its own `converged` column. Silencing it here, not globally, keeps a real warning
# elsewhere from going unnoticed.
warnings.filterwarnings("ignore", category=ConvergenceWarning)

ROOT = Path(__file__).parent.parent
DATA = ROOT / "model_outputs" / "tourism_model_dataset.csv"
OUT_DIR = ROOT / "model_outputs"

COVID_CLOSURE_MONTHS = {(2020, 4), (2020, 5), (2020, 6)}
HIGH_SEASON_MONTHS = {11, 12, 1, 2, 3, 4}  # Nov-Apr, standard Maldives tourism peak season
TEST_MONTHS = 12
FLIGHT_COL = "Total number of arrival flights"
FX_COL = "MVR per USD (average)"
BED_COL = "total_beds"
VARIANTS = ["demand_only", "demand_parsimonious", "supply_augmented"]


def load_series() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    df["date"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=1))
    df = df.sort_values("date").set_index("date").asfreq("MS")
    # log(0) is undefined for the 3 COVID zero-arrival months; treat as missing.
    with np.errstate(divide="ignore"):
        df["log_arrivals"] = np.log(df["total_arrivals"].replace(0, np.nan))
    df["covid_closure"] = [
        1 if (y, m) in COVID_CLOSURE_MONTHS else 0
        for y, m in zip(df["year"], df["month"])
    ]
    df["high_season"] = df["month"].isin(HIGH_SEASON_MONTHS).astype(int)
    for m in range(2, 13):
        df[f"month_{m}"] = (df["month"] == m).astype(int)
    return df


def run_adf(series: pd.Series, label: str):
    stat, pval, *_ = adfuller(series.dropna(), autolag="AIC", result_object=False)
    verdict = "stationary (reject unit root)" if pval < 0.05 else "non-stationary (unit root present)"
    print(f"ADF[{label}]: stat={stat:.4f}  p={pval:.4f}  -> {verdict}")
    return pval


def month_dummy_cols():
    return [f"month_{m}" for m in range(2, 13)]


def exog_cols(variant="demand_only"):
    if variant == "demand_only":
        return month_dummy_cols() + ["covid_closure", FLIGHT_COL, FX_COL]
    if variant == "demand_parsimonious":
        return ["high_season", "covid_closure", FLIGHT_COL, FX_COL]
    if variant == "supply_augmented":
        return month_dummy_cols() + ["covid_closure", FLIGHT_COL, FX_COL, BED_COL]
    raise ValueError(f"unknown variant: {variant!r}")


def grid_search_order(endog, exog, d, p_range=range(0, 4), q_range=range(0, 4)):
    results = []
    for p in p_range:
        for q in q_range:
            if p == 0 and q == 0:
                continue
            try:
                m = SARIMAX(
                    endog, exog=exog, order=(p, d, q),
                    enforce_stationarity=False, enforce_invertibility=False,
                ).fit(disp=False)
                results.append((p, q, m.aic, m.bic, m.mle_retvals.get("converged", None)))
            except Exception:
                results.append((p, q, np.nan, np.nan, False))
    table = pd.DataFrame(results, columns=["p", "q", "aic", "bic", "converged"])
    return table.sort_values("bic")


def score(actual, pred, label):
    err = actual - pred
    nonzero = actual != 0  # MAPE is undefined for zero-arrival months (e.g. COVID closure)
    mape = (err[nonzero].abs() / actual[nonzero]).mean() * 100
    mae = err.abs().mean()
    rmse = np.sqrt((err ** 2).mean())
    note = "" if nonzero.all() else f"  (MAPE excludes {(~nonzero).sum()} zero-arrival month(s))"
    print(f"{label}: MAPE={mape:.2f}%  MAE={mae:.0f}  RMSE={rmse:.0f}{note}")
    return mape, mae, rmse


def fit_and_forecast(train, test, variant, d):
    """Fit one SARIMAX variant (own BIC-selected order) and forecast the holdout."""
    exog_train = train[exog_cols(variant)]
    exog_test = test[exog_cols(variant)]

    print(f"\n=== Order search for variant={variant!r} (p, q), sorted by BIC ===")
    aic_table = grid_search_order(train["log_arrivals"], exog_train, d=d)
    print(aic_table.head(5).to_string(index=False))
    best_p, best_q = int(aic_table.iloc[0]["p"]), int(aic_table.iloc[0]["q"])
    print(f"-> selected order=({best_p},{d},{best_q}) by lowest BIC")

    fit = SARIMAX(
        train["log_arrivals"], exog=exog_train, order=(best_p, d, best_q),
        enforce_stationarity=False, enforce_invertibility=False,
    ).fit(disp=False)
    if not fit.mle_retvals.get("converged", True):
        print(f"-> WARNING: the selected ({best_p},{d},{best_q}) model itself did not converge "
              f"-- treat its forecast with caution")

    forecast = fit.get_forecast(steps=len(test), exog=exog_test)
    pred_mean = np.exp(forecast.predicted_mean)
    pred_ci = np.exp(forecast.conf_int(alpha=0.05))
    return fit, pred_mean, pred_ci, (best_p, d, best_q), aic_table


def main():
    df = load_series()
    # Flight-movement data only exists from 2012-01 onward; truncate to that.
    first_valid = df[FLIGHT_COL].first_valid_index()
    dropped = df.loc[df.index < first_valid]
    print(f"Truncating to {first_valid.date()} onward -- {len(dropped)} earlier months "
          f"dropped (no {FLIGHT_COL!r} data before this).")
    df = df.loc[df.index >= first_valid]

    train = df.iloc[:-TEST_MONTHS]
    test = df.iloc[-TEST_MONTHS:]
    print(f"Train: {train.index[0].date()} to {train.index[-1].date()} ({len(train)} months)")
    print(f"Test:  {test.index[0].date()} to {test.index[-1].date()} ({len(test)} months)")

    # Stationarity: decide d from the data instead of assuming it (shared across variants).
    print("\n=== Stationarity (ADF) ===")
    p_level = run_adf(train["log_arrivals"], "log(arrivals), levels")
    p_diff = run_adf(train["log_arrivals"].diff(), "log(arrivals), 1st diff")
    d = 0 if p_level < 0.05 else (1 if p_diff < 0.05 else 2)
    print(f"-> using d={d}")

    actual = test["total_arrivals"]
    naive = df["total_arrivals"].shift(12).iloc[-TEST_MONTHS:]  # same month last year

    results = {}
    for variant in VARIANTS:
        fit, pred_mean, pred_ci, order, aic_table = fit_and_forecast(train, test, variant, d)
        results[variant] = dict(fit=fit, pred_mean=pred_mean, pred_ci=pred_ci, order=order)
        aic_table.to_csv(OUT_DIR / f"baseline_aic_grid_{variant}.csv", index=False)

    print("\n=== Holdout evaluation (last 12 months), all variants ===")
    metrics_rows = []
    for variant in VARIANTS:
        mape, mae, rmse = score(actual, results[variant]["pred_mean"], f"Model[{variant}]")
        metrics_rows.append({"model": variant, "mape": mape, "mae": mae, "rmse": rmse})
    naive_mape, naive_mae, naive_rmse = score(actual, naive, "Seasonal-naive (same month last year)")
    metrics_rows.append({"model": "seasonal_naive", "mape": naive_mape, "mae": naive_mae, "rmse": naive_rmse})

    best_variant = min(VARIANTS, key=lambda v: next(r["mape"] for r in metrics_rows if r["model"] == v))
    best_mape = next(r["mape"] for r in metrics_rows if r["model"] == best_variant)
    if best_mape < naive_mape:
        print(f"-> best variant ({best_variant}) beats seasonal-naive "
              f"({best_mape:.2f}% < {naive_mape:.2f}% MAPE)")
    else:
        print(f"-> no variant beats seasonal-naive (best={best_variant} at {best_mape:.2f}% "
              f">= {naive_mape:.2f}% MAPE) -- added complexity isn't earning its keep on this holdout")

    # Plot the primary demand_only variant for readability (others are compared in the metrics table).
    plot_variant = "demand_only"
    pred_mean = results[plot_variant]["pred_mean"]
    pred_ci = results[plot_variant]["pred_ci"]
    best_p, d_, best_q = results[plot_variant]["order"]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train.index[-36:], train["total_arrivals"].iloc[-36:], label="train (actual)")
    ax.plot(test.index, actual, label="test (actual)", color="black", marker="o")
    ax.plot(test.index, pred_mean, label="model forecast", color="red", marker="x")
    ax.plot(test.index, naive, label="seasonal-naive", color="grey", linestyle="--", marker=".")
    ax.fill_between(test.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color="red", alpha=0.15)
    ax.set_title(f"Demand-side baseline (ARIMA({best_p},{d_},{best_q}) + dummies/regressors) vs. seasonal-naive")
    ax.set_ylabel("Monthly arrivals")
    ax.legend()
    fig.tight_layout()
    out_path = OUT_DIR / "baseline_demand_forecast.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nWrote {out_path}")

    pd.DataFrame(metrics_rows).to_csv(OUT_DIR / "baseline_eval_metrics.csv", index=False)
    print(f"Wrote per-variant AIC grids and {OUT_DIR / 'baseline_eval_metrics.csv'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Demand-side forecasting baseline for Maldives tourist arrivals, rebuilt to
match the structure of the MMA/Shuaib (2021) benchmark paper after reading it
in full (see logs/DEV_LOG.md, 2026-11-08 entries) rather than defaulting to a
generic SARIMA.

Matches the benchmark's approach:
  - ADF stationarity test decides differencing (not assumed)
  - log(arrivals), not raw levels
  - seasonality via monthly dummy regressors, not seasonal (SARIMA) differencing
  - economically-motivated regressors: flight movements, USD/MVR exchange rate
  - ARIMA order chosen by AIC grid search, not asserted
  - residual diagnostics reported (Ljung-Box / heteroskedasticity / normality),
    not discarded

Deliberate departure from the benchmark, flagged explicitly rather than
silently matched or silently diverged (see docs/supervisor_questions_2026-11-08.txt,
question 2 -- not yet answered by the supervisor): COVID is handled with a
dummy variable covering the zero-arrival closure months, rather than MMA's
full scenario-based abandonment of the ARIMA model for 2020-2023. The dataset
here runs through mid-2026, three-plus years past the acute crisis MMA was
writing into, which is the main reason a dummy is more defensible here than
it would have been in 2021.

Known limitation of this backtest (stated, not hidden): flight movements and
the exchange rate are used as regressors using their *actual realized* values
in the test window, not independently forecast values. That's fine for
evaluating whether these regressors explain arrivals in-sample/out-of-sample,
but this script as written could not be used for a genuine forward forecast
without first forecasting those two inputs themselves.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX

ROOT = Path(__file__).parent.parent
DATA = ROOT / "model_outputs" / "tourism_model_dataset.csv"
OUT_DIR = ROOT / "model_outputs"

COVID_CLOSURE_MONTHS = {(2020, 4), (2020, 5), (2020, 6)}
TEST_MONTHS = 12
FLIGHT_COL = "Total number of arrival flights"
FX_COL = "MVR per USD (average)"


def load_series() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    df["date"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=1))
    df = df.sort_values("date").set_index("date").asfreq("MS")
    # log(0) is undefined for the 3 total-closure months (border shut, literal
    # zero arrivals) -- treat as missing rather than imputing a fake value;
    # SARIMAX's state-space form handles missing endog natively (Kalman filter
    # interpolation), which is the right tool here, not a fabricated number.
    with np.errstate(divide="ignore"):
        df["log_arrivals"] = np.log(df["total_arrivals"].replace(0, np.nan))
    df["covid_closure"] = [
        1 if (y, m) in COVID_CLOSURE_MONTHS else 0
        for y, m in zip(df["year"], df["month"])
    ]
    for m in range(2, 13):
        df[f"month_{m}"] = (df["month"] == m).astype(int)
    return df


def run_adf(series: pd.Series, label: str):
    stat, pval, *_ = adfuller(series.dropna(), autolag="AIC")
    verdict = "stationary (reject unit root)" if pval < 0.05 else "non-stationary (unit root present)"
    print(f"ADF[{label}]: stat={stat:.4f}  p={pval:.4f}  -> {verdict}")
    return pval


def month_dummy_cols():
    return [f"month_{m}" for m in range(2, 13)]


def exog_cols():
    return month_dummy_cols() + ["covid_closure", FLIGHT_COL, FX_COL]


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


def main():
    df = load_series()
    # Flight movements (a regressor, matching MMA's use of flight data) only
    # exist from 2012-01 onward in the MMA series -- truncate to that start
    # rather than fabricate pre-2012 values or silently drop the regressor.
    first_valid = df[FLIGHT_COL].first_valid_index()
    dropped = df.loc[df.index < first_valid]
    print(f"Truncating to {first_valid.date()} onward -- {len(dropped)} earlier months "
          f"dropped (no {FLIGHT_COL!r} data before this).")
    df = df.loc[df.index >= first_valid]

    train = df.iloc[:-TEST_MONTHS]
    test = df.iloc[-TEST_MONTHS:]
    print(f"Train: {train.index[0].date()} to {train.index[-1].date()} ({len(train)} months)")
    print(f"Test:  {test.index[0].date()} to {test.index[-1].date()} ({len(test)} months)")

    # --- Step 1: stationarity, decide d (don't assume it) ---
    print("\n=== Stationarity (ADF) ===")
    p_level = run_adf(train["log_arrivals"], "log(arrivals), levels")
    p_diff = run_adf(train["log_arrivals"].diff(), "log(arrivals), 1st diff")
    d = 0 if p_level < 0.05 else (1 if p_diff < 0.05 else 2)
    print(f"-> using d={d} (matches MMA's own finding of I(1) at 5% for their series)")

    exog_train = train[exog_cols()]
    exog_test = test[exog_cols()]

    # --- Step 2: order selection by BIC (not AIC), not asserted ---
    # BIC penalizes extra parameters more heavily than AIC -- the right call
    # here given 163 training months and a model that already carries 14
    # regressors (11 month dummies + covid + 2 economic vars); AIC selection
    # picked ARIMA(3,0,2) in an earlier run of this script, which produced a
    # singular covariance matrix and a non-converged fit -- a textbook
    # overfitting symptom on a short, collinear series. BIC is the standard
    # fix and is reported alongside AIC below for transparency.
    print("\n=== Order search (p, q), sorted by BIC ===")
    aic_table = grid_search_order(train["log_arrivals"], exog_train, d=d)
    print(aic_table.head(10).to_string(index=False))
    best_p, best_q = int(aic_table.iloc[0]["p"]), int(aic_table.iloc[0]["q"])
    print(f"-> selected order=({best_p},{d},{best_q}) by lowest BIC")

    # --- Step 3: fit final model ---
    model = SARIMAX(
        train["log_arrivals"], exog=exog_train, order=(best_p, d, best_q),
        enforce_stationarity=False, enforce_invertibility=False,
    )
    fit = model.fit(disp=False)
    print("\n" + str(fit.summary()))  # includes Ljung-Box, heteroskedasticity, JB normality

    # --- Step 4: forecast + evaluate against actual + seasonal-naive ---
    forecast = fit.get_forecast(steps=TEST_MONTHS, exog=exog_test)
    pred_log = forecast.predicted_mean
    pred_ci_log = forecast.conf_int(alpha=0.05)
    pred_mean = np.exp(pred_log)
    pred_ci = np.exp(pred_ci_log)

    actual = test["total_arrivals"]
    naive = df["total_arrivals"].shift(12).iloc[-TEST_MONTHS:]  # same month last year

    def score(pred, label):
        err = actual - pred
        mape = (err.abs() / actual).mean() * 100
        rmse = np.sqrt((err ** 2).mean())
        print(f"{label}: MAPE={mape:.2f}%  RMSE={rmse:.0f}")
        return mape, rmse

    print("\n=== Holdout evaluation (last 12 months) ===")
    for d_, a, p, n in zip(test.index, actual, pred_mean, naive):
        print(f"{d_.date()}  actual={a:>8.0f}  model={p:>8.0f}  seasonal_naive={n:>8.0f}")
    print()
    model_mape, model_rmse = score(pred_mean, "Model (ARIMA + dummies + regressors)")
    naive_mape, naive_rmse = score(naive, "Seasonal-naive (same month last year)")
    if model_mape < naive_mape:
        print(f"-> model beats seasonal-naive ({model_mape:.2f}% < {naive_mape:.2f}% MAPE)")
    else:
        print(f"-> model does NOT beat seasonal-naive ({model_mape:.2f}% >= {naive_mape:.2f}% MAPE) "
              f"-- its added complexity isn't earning its keep on this holdout")

    # --- Step 5: plot ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train.index[-36:], train["total_arrivals"].iloc[-36:], label="train (actual)")
    ax.plot(test.index, actual, label="test (actual)", color="black", marker="o")
    ax.plot(test.index, pred_mean, label="model forecast", color="red", marker="x")
    ax.plot(test.index, naive, label="seasonal-naive", color="grey", linestyle="--", marker=".")
    ax.fill_between(test.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1], color="red", alpha=0.15)
    ax.set_title(f"Demand-side baseline (ARIMA({best_p},{d},{best_q}) + dummies/regressors) vs. seasonal-naive")
    ax.set_ylabel("Monthly arrivals")
    ax.legend()
    fig.tight_layout()
    out_path = OUT_DIR / "baseline_demand_forecast.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nWrote {out_path}")

    # Save AIC grid + metrics for the record.
    aic_table.to_csv(OUT_DIR / "baseline_aic_grid.csv", index=False)
    pd.DataFrame({
        "model": ["arima_dummies", "seasonal_naive"],
        "mape": [model_mape, naive_mape],
        "rmse": [model_rmse, naive_rmse],
    }).to_csv(OUT_DIR / "baseline_eval_metrics.csv", index=False)
    print(f"Wrote {OUT_DIR / 'baseline_aic_grid.csv'} and {OUT_DIR / 'baseline_eval_metrics.csv'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Demand-side ARIMA baseline for Maldives tourist arrivals, benchmarked
against the MMA/Shuaib (2021) model structure. Uses realized (not forecast)
flight/exchange-rate values, so it isn't yet usable for a forward forecast."""
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
    # log(0) is undefined for the 3 COVID zero-arrival months; treat as missing.
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

    # Stationarity: decide d from the data instead of assuming it.
    print("\n=== Stationarity (ADF) ===")
    p_level = run_adf(train["log_arrivals"], "log(arrivals), levels")
    p_diff = run_adf(train["log_arrivals"].diff(), "log(arrivals), 1st diff")
    d = 0 if p_level < 0.05 else (1 if p_diff < 0.05 else 2)
    print(f"-> using d={d}")

    exog_train = train[exog_cols()]
    exog_test = test[exog_cols()]

    # Order selection by BIC (AIC overfit to a singular, non-converged model).
    print("\n=== Order search (p, q), sorted by BIC ===")
    aic_table = grid_search_order(train["log_arrivals"], exog_train, d=d)
    print(aic_table.head(10).to_string(index=False))
    best_p, best_q = int(aic_table.iloc[0]["p"]), int(aic_table.iloc[0]["q"])
    print(f"-> selected order=({best_p},{d},{best_q}) by lowest BIC")

    model = SARIMAX(
        train["log_arrivals"], exog=exog_train, order=(best_p, d, best_q),
        enforce_stationarity=False, enforce_invertibility=False,
    )
    fit = model.fit(disp=False)
    print("\n" + str(fit.summary()))  # includes residual diagnostics

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

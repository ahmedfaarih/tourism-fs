#!/usr/bin/env python3
"""Year-by-year expanding-window backtest for the demand-side baseline, answering
the supervisor's "quantify this error by year" / "the single 12-month holdout is
fragile" feedback. For each calendar year Y with a full 12 months of data and
enough prior history, trains on everything strictly before Jan of Y and forecasts
Jan-Dec of Y, for each of the three regressor variants in model_baseline_demand.py.

Each variant's ARIMA order is selected ONCE (by BIC, on the full available history)
rather than re-searched per fold -- keeps folds comparable and avoids an expensive/
noisy re-search every year; this mirrors the choice already documented in the main
script.

Same disclosed limitation as the main script: flight/FX regressors use realized
values in every fold, not independently forecast ones -- fine for evaluating
in-sample structure, not yet a genuine forward (ex-ante) forecast.
"""
import pandas as pd

from model_baseline_demand import (
    OUT_DIR, FLIGHT_COL, VARIANTS, exog_cols, grid_search_order, load_series, score,
)
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np

BACKTEST_YEARS = range(2017, 2026)  # 2026 excluded: partial year


def select_orders(df, d=0):
    """Pick one fixed (p, d, q) per variant by BIC, using all available history."""
    orders = {}
    for variant in VARIANTS:
        exog = df[exog_cols(variant)]
        table = grid_search_order(df["log_arrivals"], exog, d=d)
        best_p, best_q = int(table.iloc[0]["p"]), int(table.iloc[0]["q"])
        orders[variant] = (best_p, d, best_q)
        print(f"variant={variant!r} -> fixed order {orders[variant]} (selected on full history)")
    return orders


def main():
    df = load_series()
    first_valid = df[FLIGHT_COL].first_valid_index()
    df = df.loc[df.index >= first_valid]

    orders = select_orders(df)

    rows = []
    for year in BACKTEST_YEARS:
        fold_start = pd.Timestamp(year=year, month=1, day=1)
        fold_end = pd.Timestamp(year=year, month=12, day=1)
        if fold_end not in df.index:
            print(f"Skipping {year}: not enough data (fold end {fold_end.date()} missing).")
            continue
        train = df.loc[df.index < fold_start]
        test = df.loc[(df.index >= fold_start) & (df.index <= fold_end)]
        if len(train) < 36 or len(test) != 12:
            print(f"Skipping {year}: train={len(train)} months, test={len(test)} months (too little data).")
            continue

        actual = test["total_arrivals"]
        naive = df["total_arrivals"].shift(12).loc[test.index]

        for variant in VARIANTS:
            order = orders[variant]
            exog_train = train[exog_cols(variant)]
            exog_test = test[exog_cols(variant)]
            fit = SARIMAX(
                train["log_arrivals"], exog=exog_train, order=order,
                enforce_stationarity=False, enforce_invertibility=False,
            ).fit(disp=False)
            if not fit.mle_retvals.get("converged", True):
                print(f"-> WARNING: {year} [{variant}] fit did not converge -- treat with caution")
            pred = np.exp(fit.get_forecast(steps=len(test), exog=exog_test).predicted_mean)
            mape, mae, rmse = score(actual, pred, f"{year} [{variant}]")
            rows.append({"year": year, "model": variant, "mape": mape, "mae": mae, "rmse": rmse})

        naive_mape, naive_mae, naive_rmse = score(actual, naive, f"{year} [seasonal_naive]")
        rows.append({"year": year, "model": "seasonal_naive", "mape": naive_mape, "mae": naive_mae, "rmse": naive_rmse})
        print()

    result = pd.DataFrame(rows)
    print("\n=== Year-by-year backtest, all variants ===")
    for metric, unit in [("mape", "%"), ("mae", "arrivals"), ("rmse", "arrivals")]:
        print(f"\n--- {metric.upper()} ({unit}) by year/model ---")
        print(result.pivot(index="year", columns="model", values=metric).round(2).to_string())

    out_path = OUT_DIR / "baseline_yearly_backtest.csv"
    result.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Visualizes how train/test splitting works in this project, for anyone (supervisor
included) who wants to see it rather than just read date ranges in a log:
  1. The single 12-month holdout used by model_baseline_demand.py -- full arrivals
     series colored by train (orange) vs. test (blue), same idea as the classic
     train/test-split chart style.
  2. The expanding-window backtest used by model_baseline_yearly_backtest.py -- one
     horizontal bar per backtested year, showing how the training window grows while
     the test window stays a fixed 12 months, always strictly after training ends.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).parent.parent
DATA = ROOT / "model_outputs" / "tourism_model_dataset.csv"
OUT_DIR = ROOT / "model_outputs"

TEST_MONTHS = 12
FLIGHT_COL = "Total number of arrival flights"
BACKTEST_YEARS = range(2017, 2026)


def load_series():
    df = pd.read_csv(DATA)
    df["date"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=1))
    df = df.sort_values("date").set_index("date").asfreq("MS")
    first_valid = df[FLIGHT_COL].first_valid_index()
    return df.loc[df.index >= first_valid]


def plot_single_holdout(ax, df):
    train = df.iloc[:-TEST_MONTHS]
    test = df.iloc[-TEST_MONTHS:]
    ax.plot(train.index, train["total_arrivals"], ".", color="#d4752b", ms=4, label="TRAINING SET")
    ax.plot(test.index, test["total_arrivals"], ".", color="#1f77b4", ms=4, label="TEST SET")
    ax.axvline(test.index[0], color="black", linewidth=0.8, linestyle=":")
    ax.set_title(f"Single 12-month holdout -- train {train.index[0].date()} to "
                 f"{train.index[-1].date()}, test {test.index[0].date()} to {test.index[-1].date()}")
    ax.set_ylabel("Monthly arrivals")
    ax.legend(loc="upper left")


def plot_expanding_backtest(ax, df):
    rows = []
    for year in BACKTEST_YEARS:
        fold_start = pd.Timestamp(year=year, month=1, day=1)
        fold_end = pd.Timestamp(year=year, month=12, day=1)
        if fold_end not in df.index:
            continue
        train = df.loc[df.index < fold_start]
        if len(train) < 36:
            continue
        rows.append((year, train.index[0], train.index[-1], fold_start, fold_end))

    for i, (year, train_start, train_end, test_start, test_end) in enumerate(rows):
        ax.barh(i, (train_end - train_start).days, left=train_start, height=0.6,
                color="#d4752b", label="TRAINING SET" if i == 0 else None)
        ax.barh(i, (test_end - test_start).days + 31, left=test_start, height=0.6,
                color="#1f77b4", label="TEST SET" if i == 0 else None)

    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{year} fold" for year, *_ in rows])
    ax.invert_yaxis()
    ax.set_title("Expanding-window backtest -- training window grows, test window is always the next full year")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left")


def main():
    df = load_series()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    plot_single_holdout(ax1, df)
    plot_expanding_backtest(ax2, df)
    fig.suptitle("How train/test splitting works in this project", fontsize=13)
    fig.tight_layout()
    out_path = OUT_DIR / "train_test_split.png"
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

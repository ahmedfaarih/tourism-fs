#!/usr/bin/env python3
"""Combined comparison chart across all models (SARIMAX x3 variants, Prophet x3 variants,
seasonal-naive), reading the already-written eval CSVs rather than refitting anything.
Two panels: the single 12-month holdout (bar chart) and the year-by-year backtest
(line chart, log-scale y-axis so the extreme 2020 COVID year doesn't squash every other
year flat). Plain-language labels throughout, matching the reply email's terminology.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).parent.parent
OUT_DIR = ROOT / "model_outputs"

LABELS = {
    "demand_only": "Original (SARIMAX)",
    "demand_parsimonious": "Season-only (SARIMAX)",
    "supply_augmented": "Supply-side (SARIMAX)",
    "prophet_demand_only": "Original (Prophet)",
    "prophet_demand_parsimonious": "Season-only (Prophet)",
    "prophet_supply_augmented": "Supply-side (Prophet)",
    "seasonal_naive": "Seasonal-naive (benchmark)",
}
COLORS = {
    "demand_only": "#1f77b4",
    "demand_parsimonious": "#4b9fd1",
    "supply_augmented": "#7fc3e8",
    "prophet_demand_only": "#d62728",
    "prophet_demand_parsimonious": "#e8716f",
    "supply_augmented_prophet": "#f3a6a4",
    "prophet_supply_augmented": "#f3a6a4",
    "seasonal_naive": "#333333",
}
ORDER = [
    "seasonal_naive", "demand_only", "demand_parsimonious", "supply_augmented",
    "prophet_demand_only", "prophet_demand_parsimonious", "prophet_supply_augmented",
]


def load_holdout():
    sarimax = pd.read_csv(OUT_DIR / "baseline_eval_metrics.csv")
    prophet = pd.read_csv(ROOT / "prophet_model" / "model_outputs" / "prophet_eval_metrics.csv")
    prophet = prophet[prophet["model"] != "seasonal_naive"]  # avoid duplicate naive row
    return pd.concat([sarimax, prophet], ignore_index=True)


def load_yearly():
    sarimax = pd.read_csv(OUT_DIR / "baseline_yearly_backtest.csv")
    prophet = pd.read_csv(ROOT / "prophet_model" / "model_outputs" / "prophet_yearly_backtest.csv")
    prophet = prophet[prophet["model"] != "seasonal_naive"]
    return pd.concat([sarimax, prophet], ignore_index=True)


def main():
    holdout = load_holdout()
    yearly = load_yearly()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Panel 1: single 12-month holdout, MAPE by model.
    holdout = holdout.set_index("model")
    present = [m for m in ORDER if m in holdout.index]
    holdout = holdout.loc[present]
    bars = ax1.bar(
        [LABELS[m] for m in holdout.index], holdout["mape"],
        color=[COLORS[m] for m in holdout.index],
    )
    ax1.bar_label(bars, fmt="%.1f%%")
    ax1.set_ylabel("MAPE (%)")
    ax1.set_title("Single 12-month holdout (2025-08 to 2026-07)")
    ax1.tick_params(axis="x", rotation=40)
    for label in ax1.get_xticklabels():
        label.set_ha("right")

    # Panel 2: year-by-year backtest, MAPE by model, log-scale y so 2020 doesn't flatten the rest.
    pivot = yearly.pivot(index="year", columns="model", values="mape")
    for m in ORDER:
        if m not in pivot.columns:
            continue
        style = "--" if m == "seasonal_naive" else "-"
        width = 2.5 if m == "seasonal_naive" else 1.8
        ax2.plot(pivot.index, pivot[m], style, marker="o", label=LABELS[m], color=COLORS[m], linewidth=width)
    ax2.set_yscale("log")
    ax2.set_ylabel("MAPE (%), log scale")
    ax2.set_xlabel("Backtest year")
    ax2.set_title("Year-by-year expanding-window backtest (2017-2025)")
    ax2.legend(fontsize=8, loc="upper left")
    ax2.grid(True, which="both", alpha=0.3)

    fig.suptitle("Maldives tourist-arrivals forecasting: model comparison", fontsize=13)
    fig.tight_layout()
    out_path = OUT_DIR / "model_comparison.png"
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

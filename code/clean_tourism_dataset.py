#!/usr/bin/env python3
"""Merges raw ministry + supplementary data into tourism_model_dataset.csv."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
EXTRACTED = ROOT / "raw_data" / "tourism_dataset_extracted"
SUPP_MMA = ROOT / "raw_data" / "supplementary_data" / "mma"
SUPP_WEATHER = ROOT / "raw_data" / "supplementary_data" / "weather"
OUT_DIR = ROOT / "model_outputs"

# Raw region key -> stable key, since category names shift across years.
REGION_CROSSWALK = {
    "OCEANIA": "ASIA & THE PACIFIC",
    "ASIA &amp; THE PACIFIC": "ASIA & THE PACIFIC",
    "ASIA & THE PACIFIC": "ASIA & THE PACIFIC",
    "EUROPE": "EUROPE",
    "AMERICAS": "AMERICAS",
    "MIDDLE EAST": "MIDDLE EAST",
    "AFRICA": "AFRICA",
    "UN PASSPORT HOLDERS": "UN PASSPORT HOLDERS",
    "OTHERS": "OTHERS",
    "OTHERS / NOT STATED": "OTHERS",
}
STABLE_REGIONS = [
    "EUROPE", "ASIA & THE PACIFIC", "AMERICAS", "MIDDLE EAST", "AFRICA",
    "OTHERS", "UN PASSPORT HOLDERS",
]

# Countries that drop out of the taxonomy entirely (flagged, not zeroed).
DISCONTINUED_COUNTRIES = {"Taiwan": 2023, "Israel": 2025, "Turkey": 2025}

# Months with a known-broken region breakdown; nulled rather than trusted.
BROKEN_REGION_MONTHS = {(2026, 6), (2026, 7)}


def remap_region_dict(raw: dict) -> dict:
    out = {k: 0 for k in STABLE_REGIONS}
    for raw_key, val in raw.items():
        stable = REGION_CROSSWALK.get(raw_key)
        if stable is None:
            raise ValueError(f"Unmapped region key: {raw_key!r} -- add it to REGION_CROSSWALK")
        out[stable] += val
    return out


def load_monthly_rows() -> list[dict]:
    rows = []
    for year_dir in sorted(EXTRACTED.iterdir()):
        if not year_dir.is_dir():
            continue
        year = int(year_dir.name)
        data = json.loads((year_dir / "tourism.json").read_text())
        monthly = data.get("monthly")
        if not monthly:
            # 2007/2008: no arrivals data at all, only atoll_bed_capacity. Skip.
            continue
        for m in monthly:
            row = {
                "year": year,
                "month": m["month"],
                "total_arrivals": m.get("total_arrivals"),
                "arrivals_by_air": m.get("arrivals_by_air"),
                "arrivals_by_sea": m.get("arrivals_by_sea"),
                "total_beds": m.get("total_beds"),
                "bed_nights": m.get("bed_nights"),
                "occupancy_rate_pct": m.get("occupancy_rate_pct"),
                "avg_stay_days": m.get("avg_stay_days"),
            }

            # Fix: 2009-01 avg_stay_days contains the year (2009), not a real value.
            if year == 2009 and m["month"] == 1 and row["avg_stay_days"] == 2009:
                row["avg_stay_days"] = np.nan

            # Region breakdown, with crosswalk; null out the two broken 2026 months.
            region_raw = m.get("arrivals_by_region", {})
            if (year, m["month"]) in BROKEN_REGION_MONTHS:
                for r in STABLE_REGIONS:
                    row[f"region_{r}"] = np.nan
                row["region_data_flagged_broken"] = True
            else:
                remapped = remap_region_dict(region_raw)
                for r, v in remapped.items():
                    row[f"region_{r}"] = v
                row["region_data_flagged_broken"] = False

            rows.append(row)
    return rows


def check_discontinued_countries(country_frames: dict[int, dict]) -> list[str]:
    """Sanity-check that DISCONTINUED_COUNTRIES actually disappear where expected."""
    notes = []
    for country, drop_year in DISCONTINUED_COUNTRIES.items():
        present_before = any(
            country in country_frames.get(y, {}) for y in range(drop_year - 3, drop_year)
        )
        present_after = any(
            country in country_frames.get(y, {}) for y in range(drop_year, drop_year + 3)
        )
        notes.append(
            f"{country}: present before {drop_year} = {present_before}, "
            f"present from {drop_year} onward = {present_after}"
        )
    return notes


def load_country_presence() -> dict[int, dict]:
    """year -> {country: total arrivals that year}, for the discontinuation sanity check."""
    out = {}
    for year_dir in sorted(EXTRACTED.iterdir()):
        if not year_dir.is_dir():
            continue
        year = int(year_dir.name)
        data = json.loads((year_dir / "tourism.json").read_text())
        countries = data.get("annual_summary", {}).get("arrivals_by_country", {})
        out[year] = countries
    return out


def load_mma_series(filename: str, series_id_name: dict[int, str]) -> pd.DataFrame:
    """Load one mma json file, return wide monthly dataframe indexed by (year, month)."""
    data = json.loads((SUPP_MMA / filename).read_text())
    frames = []
    for series in data:
        name = series_id_name.get(series["id"], series["name"])
        df = pd.DataFrame(series["data"])
        df["date"] = pd.to_datetime(df["date"])
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df = df.rename(columns={"amount": name})[["year", "month", name]]
        frames.append(df)
    out = frames[0]
    for f in frames[1:]:
        out = out.merge(f, on=["year", "month"], how="outer")
    return out


def load_weather_monthly() -> pd.DataFrame:
    data = json.loads((SUPP_WEATHER / "male_monthly_2009_2026.json").read_text())
    df = pd.DataFrame(data)
    return df[[
        "year", "month", "avg_temp_mean_c", "total_precipitation_mm",
        "rainy_days", "avg_max_windgust_kmh",
    ]]


def main():
    monthly_rows = load_monthly_rows()
    df = pd.DataFrame(monthly_rows).sort_values(["year", "month"]).reset_index(drop=True)

    # Sanity check: annual_summary total must equal sum(monthly totals).
    for year_dir in sorted(EXTRACTED.iterdir()):
        if not year_dir.is_dir():
            continue
        year = int(year_dir.name)
        data = json.loads((year_dir / "tourism.json").read_text())
        if not data.get("monthly"):
            continue
        annual_total = data.get("annual_summary", {}).get("total_arrivals")
        computed = df.loc[df["year"] == year, "total_arrivals"].sum()
        if annual_total is not None and annual_total != computed:
            print(f"WARNING: {year} annual_summary total ({annual_total}) != "
                  f"sum of monthly ({computed})")

    # Discontinued-country sanity check (printed only, not merged into output).
    country_frames = load_country_presence()
    for note in check_discontinued_countries(country_frames):
        print("NOTE:", note)

    # Merge supplementary MMA series.
    flight = load_mma_series("flight_movements.json", {})
    df = df.merge(flight, on=["year", "month"], how="left")

    bed_occ = load_mma_series("bed_capacity_occupancy.json", {})
    df = df.merge(bed_occ, on=["year", "month"], how="left")

    cpi = load_mma_series("cpi_prices.json", {})
    df = df.merge(cpi, on=["year", "month"], how="left")

    gdp = load_mma_series("tourism_gdp.json", {})
    df = df.merge(gdp, on=["year", "month"], how="left")

    fx = load_mma_series("exchange_rates.json", {})
    df = df.merge(fx, on=["year", "month"], how="left")

    # Weather runs longer than tourism data; left-merge truncates it to match.
    weather = load_weather_monthly()
    df = df.merge(weather, on=["year", "month"], how="left")

    out_path = OUT_DIR / "tourism_model_dataset.csv"
    df.to_csv(out_path, index=False)

    # Coverage report.
    print("\n=== Coverage ===")
    print(f"Rows: {len(df)} (months {df['year'].min()}-{df['month'][df['year']==df['year'].min()].min():02d} "
          f"to {df['year'].max()}-{df['month'][df['year']==df['year'].max()].max():02d})")
    print(f"Years covered: {sorted(df['year'].unique())}")
    print(f"avg_stay_days nulls: {df['avg_stay_days'].isna().sum()} "
          f"(expected 1, the fixed 2009-01 bug)")
    print(f"region_data_flagged_broken=True rows: {df['region_data_flagged_broken'].sum()} "
          f"(expected 2, 2026-06 and 2026-07)")
    for col in ["total_arrivals", "total_beds", "occupancy_rate_pct"]:
        print(f"{col} nulls: {df[col].isna().sum()}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

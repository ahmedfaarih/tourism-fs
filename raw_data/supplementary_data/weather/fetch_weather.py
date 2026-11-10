#!/usr/bin/env python3
"""Pull daily historical weather for Male/Hulhule from the Open-Meteo archive API
(free, no key) and aggregate it to monthly. No third-party deps.
"""
import json
import urllib.request
from pathlib import Path

LAT, LON = 4.1755, 73.5093
START, END = "2009-01-01", "2026-09-30"
DAILY_FIELDS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_sum",
    "rain_sum",
    "windspeed_10m_max",
    "windgusts_10m_max",
]
OUT_DIR = Path(__file__).parent


def fetch_daily() -> dict:
    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={LAT}&longitude={LON}&start_date={START}&end_date={END}"
        f"&daily={','.join(DAILY_FIELDS)}&timezone=Indian%2FMaldives"
    )
    with urllib.request.urlopen(url) as resp:
        return json.load(resp)


def to_monthly(daily: dict) -> list[dict]:
    dates = daily["daily"]["time"]
    by_month: dict[tuple[int, int], list[int]] = {}
    for i, d in enumerate(dates):
        y, m = int(d[:4]), int(d[5:7])
        by_month.setdefault((y, m), []).append(i)

    monthly = []
    for (y, m), idxs in sorted(by_month.items()):
        daily_data = daily["daily"]
        precip = [daily_data["precipitation_sum"][i] for i in idxs]
        monthly.append({
            "year": y,
            "month": m,
            "avg_temp_max_c": round(sum(daily_data["temperature_2m_max"][i] for i in idxs) / len(idxs), 2),
            "avg_temp_min_c": round(sum(daily_data["temperature_2m_min"][i] for i in idxs) / len(idxs), 2),
            "avg_temp_mean_c": round(sum(daily_data["temperature_2m_mean"][i] for i in idxs) / len(idxs), 2),
            "total_precipitation_mm": round(sum(precip), 2),
            "rainy_days": sum(1 for p in precip if p > 1.0),
            "avg_max_windgust_kmh": round(sum(daily_data["windgusts_10m_max"][i] for i in idxs) / len(idxs), 2),
            "days_in_month": len(idxs),
        })
    return monthly


def main():
    daily = fetch_daily()
    (OUT_DIR / "male_daily_2009_2026.json").write_text(json.dumps(daily, indent=2))

    monthly = to_monthly(daily)
    (OUT_DIR / "male_monthly_2009_2026.json").write_text(json.dumps(monthly, indent=2))

    print(f"Daily points: {len(daily['daily']['time'])}, monthly points: {len(monthly)}")


if __name__ == "__main__":
    main()

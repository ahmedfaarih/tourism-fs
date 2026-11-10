#!/usr/bin/env python3
"""Pull confirmed tourism-relevant series from the MMA statistics database API.

Reads the bearer token from the MMA_API_TOKEN environment variable.
Writes one raw JSON file per group plus a combined manifest.csv-style report.
Local-only output -- never touches tourism_dataset_msb/ or git.
"""
import json
import os
import sys
import urllib.request
from pathlib import Path

API_URL = "https://database.mma.gov.mv/api/series"
OUT_DIR = Path(__file__).parent

GROUPS = {
    # 104/105/151/179/186/197/205/210 (total arrivals, region split, bednights,
    # avg stay) and 211/216/226 (aggregate bed capacity/occupancy) deliberately
    # excluded -- already covered monthly 2007-2026 by the Ministry/yearbook
    # dataset, 2007+ is enough range for this thesis.
    "bed_capacity_occupancy": [i for i in range(211, 241) if i not in (211, 216, 226)],
    "flight_movements": [242, 243, 244],
    "travel_receipts": [246, 3409, 3412],
    "exchange_rates": list(range(4039, 4060)) + [4061, 4062, 4063],
    "cpi_prices": [280, 520],
    "tourism_gdp": [4943, 4955],
}


def fetch(ids: list[int]) -> list[dict]:
    url = f"{API_URL}?ids={','.join(str(i) for i in ids)}"
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {TOKEN}",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req) as resp:
        body = json.load(resp)
    return body.get("data", [])


def main():
    global TOKEN
    TOKEN = os.environ.get("MMA_API_TOKEN")
    if not TOKEN:
        print("Set MMA_API_TOKEN environment variable first.", file=sys.stderr)
        sys.exit(1)

    manifest = []
    for group_name, ids in GROUPS.items():
        series = fetch(ids)
        out_file = OUT_DIR / f"{group_name}.json"
        out_file.write_text(json.dumps(series, indent=2))

        found_ids = {s["id"] for s in series}
        missing = [i for i in ids if i not in found_ids]

        for s in series:
            dates = [p["date"] for p in s["data"]]
            manifest.append({
                "group": group_name,
                "id": s["id"],
                "name": s["name"],
                "frequency": s["frequency"],
                "unit": s["unit"],
                "points": len(s["data"]),
                "first": dates[0] if dates else None,
                "last": dates[-1] if dates else None,
            })
        if missing:
            print(f"WARNING [{group_name}]: no data returned for ids {missing}", file=sys.stderr)

    manifest_file = OUT_DIR / "manifest.json"
    manifest_file.write_text(json.dumps(manifest, indent=2))

    print(f"{'group':<24}{'id':<6}{'name':<45}{'freq':<10}{'unit':<12}{'pts':<6}{'first':<12}{'last':<12}")
    for m in manifest:
        print(f"{m['group']:<24}{m['id']:<6}{m['name'][:44]:<45}{m['frequency']:<10}{m['unit']:<12}{m['points']:<6}{str(m['first']):<12}{str(m['last']):<12}")
    print(f"\n{len(manifest)} series written across {len(GROUPS)} groups -> {OUT_DIR}")


if __name__ == "__main__":
    main()

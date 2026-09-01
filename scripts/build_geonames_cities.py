"""Build a global city dataset from GeoNames' CC BY 4.0 gazetteer.

The generated file is intentionally ignored by Git. Run this before
scripts/ingest_cities.py to add broad global coverage to Pinecone.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import urllib.request
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
CITY_ARCHIVE_URL = "https://download.geonames.org/export/dump/cities15000.zip"
COUNTRY_INFO_URL = "https://download.geonames.org/export/dump/countryInfo.txt"
OUTPUT_PATH = DATA_DIR / "cities.geonames.json"


def download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "Wanderlust-AI city dataset builder"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def country_names(raw_text: str) -> dict[str, str]:
    names: dict[str, str] = {}
    for row in csv.reader(io.StringIO(raw_text), delimiter="\t"):
        if not row or row[0].startswith("#") or len(row) < 5:
            continue
        names[row[0]] = row[4]
    return names


def travel_tags(population: int) -> tuple[list[str], list[str], str]:
    if population >= 1_000_000:
        return ["big city energy", "culture", "food scene"], ["Culture", "Foodie", "City walk", "Shopping"], "medium"
    if population >= 250_000:
        return ["local culture", "walkable districts", "food scene"], ["Culture", "Foodie", "City walk"], "medium"
    return ["local character", "slow exploration", "hidden gems"], ["Culture", "City walk", "Staycation"], "budget"


def build_city(row: list[str], countries: dict[str, str]) -> dict:
    population = int(row[14] or 0)
    vibes, styles, budget_level = travel_tags(population)
    country_code = row[8]
    country = countries.get(country_code, country_code)
    city = row[1]
    return {
        "id": f"geonames-{row[0]}",
        "city": city,
        "country": country,
        "country_code": country_code,
        "region": row[10] or "",
        "latitude": float(row[4]),
        "longitude": float(row[5]),
        "population": population,
        "timezone": row[17],
        "description": f"{city} is a city in {country} with a population of about {population:,}.",
        "vibes": vibes,
        "styles": styles,
        "budget_level": budget_level,
        "source": "GeoNames",
        "source_license": "CC BY 4.0",
        "geoname_id": row[0],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a GeoNames city dataset for Wanderlust AI.")
    parser.add_argument("--limit", type=int, default=5000, help="Maximum number of cities to retain (default: 5000).")
    args = parser.parse_args()
    if args.limit < 1:
        raise SystemExit("--limit must be greater than zero.")

    print("Downloading GeoNames cities15000 and country metadata...")
    countries = country_names(download(COUNTRY_INFO_URL).decode("utf-8"))
    archive = zipfile.ZipFile(io.BytesIO(download(CITY_ARCHIVE_URL)))
    city_file = next(name for name in archive.namelist() if name.endswith(".txt"))

    rows = csv.reader(io.TextIOWrapper(archive.open(city_file), encoding="utf-8"), delimiter="\t")
    candidates = [row for row in rows if len(row) >= 19 and row[6] == "P" and row[14].isdigit()]
    candidates.sort(key=lambda row: int(row[14]), reverse=True)

    cities = []
    seen: set[tuple[str, str]] = set()
    for row in candidates:
        key = (row[1].casefold(), row[8])
        if key in seen:
            continue
        seen.add(key)
        cities.append(build_city(row, countries))
        if len(cities) == args.limit:
            break

    DATA_DIR.mkdir(exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(cities, ensure_ascii=False, indent=2) + "\n")
    print(f"Wrote {len(cities)} GeoNames city profiles to {OUTPUT_PATH}.")
    print("Attribution required: GeoNames, CC BY 4.0 (https://www.geonames.org/).")


if __name__ == "__main__":
    main()

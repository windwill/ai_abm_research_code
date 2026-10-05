"""Public-data retrieval and transformation utilities for empirical model targets."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import date
from io import StringIO
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import numpy as np
import pandas as pd


FRED_SERIES = {
    "UNRATE": {
        "label": "Civilian unemployment rate",
        "source": "Federal Reserve Bank of St. Louis FRED, U.S. Bureau of Labor Statistics",
        "url": "https://fred.stlouisfed.org/series/UNRATE",
        "units": "percent",
    },
    "GDPC1": {
        "label": "Real gross domestic product",
        "source": "Federal Reserve Bank of St. Louis FRED, U.S. Bureau of Economic Analysis",
        "url": "https://fred.stlouisfed.org/series/GDPC1",
        "units": "billions of chained 2017 dollars",
    },
    "GFDEGDQ188S": {
        "label": "Federal debt held by the public as percent of GDP",
        "source": "Federal Reserve Bank of St. Louis FRED",
        "url": "https://fred.stlouisfed.org/series/GFDEGDQ188S",
        "units": "percent of GDP",
    },
    "CES0500000003": {
        "label": "Average hourly earnings of all private employees",
        "source": "Federal Reserve Bank of St. Louis FRED, U.S. Bureau of Labor Statistics",
        "url": "https://fred.stlouisfed.org/series/CES0500000003",
        "units": "dollars per hour",
    },
    "CPIAUCSL": {
        "label": "Consumer price index for all urban consumers",
        "source": "Federal Reserve Bank of St. Louis FRED, U.S. Bureau of Labor Statistics",
        "url": "https://fred.stlouisfed.org/series/CPIAUCSL",
        "units": "index 1982-1984=100",
    },
}


@dataclass(frozen=True)
class EmpiricalSource:
    name: str
    url: str
    notes: str
    status: str = "used"


def _read_url_text(url: str, timeout: int = 30) -> str:
    with urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8")


def fetch_fred_series(series_id: str) -> pd.DataFrame:
    """Download one FRED series in a normalized long format."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    text = _read_url_text(url)
    raw = pd.read_csv(StringIO(text))
    if "observation_date" not in raw.columns or series_id not in raw.columns:
        raise ValueError(f"Unexpected FRED response for {series_id}")
    raw["date"] = pd.to_datetime(raw["observation_date"], errors="coerce")
    raw["value"] = pd.to_numeric(raw[series_id], errors="coerce")
    out = raw.loc[raw["date"].notna() & raw["value"].notna(), ["date", "value"]].copy()
    out["series_id"] = series_id
    out["variable"] = series_id
    out["source_url"] = url
    return out[["date", "variable", "series_id", "value", "source_url"]]


def fetch_acs_gini(years: Iterable[int]) -> tuple[pd.DataFrame, int] | tuple[None, None]:
    """Fetch the latest available national ACS Gini coefficient."""
    for year in years:
        url = f"https://api.census.gov/data/{year}/acs/acs1?get=NAME,B19083_001E&for=us:1"
        try:
            data = json.loads(_read_url_text(url))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
            continue
        if len(data) < 2:
            continue
        df = pd.DataFrame(data[1:], columns=data[0])
        df["date"] = pd.to_datetime(f"{year}-12-31")
        df["value"] = pd.to_numeric(df["B19083_001E"], errors="coerce")
        df["variable"] = "ACS_GINI"
        df["series_id"] = "B19083_001E"
        df["source_url"] = url
        return df[["date", "variable", "series_id", "value", "source_url"]], year
    return None, None


def fetch_bds_firm_exits(years: Iterable[int]) -> pd.DataFrame:
    """Fetch national BDS firm exits and firm counts where the API is available."""
    rows = []
    for year in years:
        url = (
            "https://api.census.gov/data/timeseries/bds"
            f"?get=NAME,FIRM,FIRMDEATH_FIRMS&for=us:*&time={year}"
        )
        try:
            data = json.loads(_read_url_text(url))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError):
            continue
        if len(data) < 2:
            continue
        df = pd.DataFrame(data[1:], columns=data[0])
        if "FIRM" not in df.columns or "FIRMDEATH_FIRMS" not in df.columns:
            continue
        firms = pd.to_numeric(df["FIRM"], errors="coerce").iloc[0]
        deaths = pd.to_numeric(df["FIRMDEATH_FIRMS"], errors="coerce").iloc[0]
        if pd.notna(firms) and pd.notna(deaths) and firms > 0:
            rows.append({
                "date": pd.to_datetime(f"{year}-12-31"),
                "variable": "BDS_FIRM_EXIT_RATE",
                "series_id": "FIRMDEATH_FIRMS/FIRM",
                "value": float(deaths / firms),
                "source_url": url,
            })
    return pd.DataFrame(rows, columns=["date", "variable", "series_id", "value", "source_url"])


def annualize_fred(raw: pd.DataFrame, start_year: int, end_year: int) -> pd.DataFrame:
    """Convert downloaded FRED observations to annual model-compatible inputs."""
    data = raw.copy()
    data["year"] = pd.to_datetime(data["date"]).dt.year.astype(int)
    data = data[(data["year"] >= start_year) & (data["year"] <= end_year)].copy()
    rows = []
    for (series_id, year), group in data.sort_values("date").groupby(["series_id", "year"]):
        values = pd.to_numeric(group["value"], errors="coerce").dropna()
        if values.empty:
            continue
        method = "mean"
        value = float(values.mean())
        if series_id in {"GDPC1", "GFDEGDQ188S"}:
            method = "last"
            value = float(values.iloc[-1])
        rows.append({
            "year": int(year),
            "variable": series_id,
            "value": value,
            "aggregation": method,
            "source_url": str(group["source_url"].iloc[0]),
        })
    annual = pd.DataFrame(rows)
    if annual.empty:
        return annual

    gdp = annual[annual["variable"] == "GDPC1"].sort_values("year").copy()
    if not gdp.empty:
        gdp["value"] = np.log(gdp["value"]).diff()
        gdp["variable"] = "REAL_GDP_LOG_GROWTH"
        gdp["aggregation"] = "annual log difference"
        annual = pd.concat([annual, gdp[gdp["value"].notna()]], ignore_index=True)

    wages = annual.pivot_table(index="year", columns="variable", values="value", aggfunc="first")
    if {"CES0500000003", "CPIAUCSL"}.issubset(wages.columns):
        real_wage = wages["CES0500000003"] / wages["CPIAUCSL"]
        if real_wage.notna().any():
            normalized = real_wage / real_wage.dropna().iloc[0]
            wage_rows = pd.DataFrame({
                "year": normalized.index.astype(int),
                "variable": "REAL_WAGE_INDEX",
                "value": normalized.to_numpy(dtype=float),
                "aggregation": "nominal wage divided by CPI, normalized to first available year",
                "source_url": "FRED CES0500000003 and CPIAUCSL",
            }).dropna(subset=["value"])
            annual = pd.concat([annual, wage_rows], ignore_index=True)
    return annual.sort_values(["variable", "year"]).reset_index(drop=True)


def build_targets(
    annual: pd.DataFrame,
    *,
    start_year: int,
    end_year: int,
    acs_gini: pd.DataFrame | None = None,
    bds_exits: pd.DataFrame | None = None,
    ai_adoption: float = 0.18,
) -> pd.DataFrame:
    """Create a calibration target file compatible with `load_targets`."""
    rows = []

    def values(variable: str) -> pd.Series:
        return pd.to_numeric(annual.loc[annual["variable"] == variable, "value"], errors="coerce").dropna()

    def add(name: str, target: float, weight: float, group: str, units: str, source: str, series_id: str,
            transformation: str, notes: str) -> None:
        if not np.isfinite(target):
            return
        rows.append({
            "name": name,
            "target": float(target),
            "weight": float(weight),
            "group": group,
            "units": units,
            "frequency": "annual",
            "geography": "United States",
            "source": source,
            "series_id": series_id,
            "transformation": transformation,
            "anchor_period": f"{start_year}-{end_year}",
            "source_notes": notes,
        })

    u = values("UNRATE")
    if not u.empty:
        add(
            "U_mean", u.mean() / 100.0, 10.0, "calibration", "rate",
            FRED_SERIES["UNRATE"]["source"], "UNRATE",
            "Annual mean unemployment rate divided by 100, averaged over anchor period.",
            "Direct empirical anchor for the model unemployment moment.",
        )

    growth = values("REAL_GDP_LOG_GROWTH")
    if not growth.empty:
        add(
            "Y_growth", growth.mean(), 8.0, "calibration", "log growth",
            FRED_SERIES["GDPC1"]["source"], "GDPC1",
            "Annual log difference in real GDP, averaged over anchor period.",
            "Direct empirical anchor for real output growth.",
        )

    debt = values("GFDEGDQ188S")
    if not debt.empty:
        add(
            "Debt_to_GDP_final", debt.iloc[-1] / 100.0, 2.0, "calibration", "ratio",
            FRED_SERIES["GFDEGDQ188S"]["source"], "GFDEGDQ188S",
            "Final available debt-to-GDP observation divided by 100.",
            "Preferred fiscal calibration target because the model reports debt on a ratio scale.",
        )

    if acs_gini is not None and not acs_gini.empty:
        gini = pd.to_numeric(acs_gini["value"], errors="coerce").dropna()
        if not gini.empty:
            add(
                "Ineq_mean", gini.iloc[-1], 3.0, "calibration", "Gini coefficient",
                "U.S. Census Bureau ACS 1-year estimates", "B19083_001E",
                "Latest available national ACS Gini coefficient.",
                "Income Gini is used as an empirical proxy for the model inequality moment.",
            )

    add(
        "Adopt_final", ai_adoption, 4.0, "calibration", "share",
        "Federal Reserve FEDS Notes summary of Census BTOS", "BTOS_AI_ADOPTION_2025",
        "Year-end 2025 firm-weighted AI adoption share reported as about 18 percent.",
        "Publicly documented point estimate, not directly downloaded from the BTOS microdata.",
    )

    wage = values("REAL_WAGE_INDEX")
    if not wage.empty:
        add(
            "W_mean", wage.mean(), 1.0, "validation", "index",
            "FRED CES0500000003 and CPIAUCSL", "CES0500000003/CPIAUCSL",
            "Real hourly earnings index normalized to the first available anchor year.",
            "Hold-out validation target because model wages are normalized units.",
        )

    if bds_exits is not None and not bds_exits.empty:
        exits = pd.to_numeric(bds_exits["value"], errors="coerce").dropna()
        if not exits.empty:
            add(
                "Firm_exits_mean", exits.mean(), 1.0, "validation", "rate",
                "U.S. Census Bureau Business Dynamics Statistics", "FIRMDEATH_FIRMS/FIRM",
                "Firm exits divided by firm count, averaged over available anchor years.",
                "Hold-out validation target because model exits depend on simulation scale.",
            )

    return pd.DataFrame(rows)


def write_source_document(path: str, sources: list[EmpiricalSource], targets_path: str, annual_path: str) -> None:
    lines = [
        "# Empirical Calibration Data Sources",
        "",
        f"Access date: {date.today().isoformat()}",
        "",
        "This file documents the empirical inputs used to produce the generated calibration target file.",
        "The target CSV remains editable, but each included value should retain a source note before reported calibration results are produced.",
        "",
        "Generated outputs:",
        "",
        f"- `{annual_path}`",
        f"- `{targets_path}`",
        "",
        "## Sources",
        "",
    ]
    for src in sources:
        lines.extend([
            f"### {src.name}",
            "",
            f"- URL: {src.url}",
            f"- Status: {src.status}",
            f"- Notes: {src.notes}",
            "",
        ])
    lines.extend([
        "## Processing Notes",
        "",
        "- FRED unemployment is converted from percent to a rate for `U_mean`.",
        "- Real GDP is converted to annual log growth for `Y_growth`.",
        "- Federal debt is converted from percent of GDP to a ratio for `Debt_to_GDP_final`.",
        "- Average hourly earnings are divided by CPI and normalized to an index for validation rather than direct calibration.",
        "- ACS income Gini is treated as a proxy for the model inequality statistic.",
        "- The AI adoption target is a documented point estimate from a public Federal Reserve summary of Census BTOS rather than a direct API pull.",
    ])
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def fetch_and_build(outdir: str, start_year: int, end_year: int, ai_adoption: float) -> tuple[str, str, str]:
    os.makedirs(outdir, exist_ok=True)
    raw_dir = os.path.join(outdir, "raw")
    processed_dir = os.path.join(outdir, "processed")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    sources: list[EmpiricalSource] = []
    fred_frames = []
    for series_id, meta in FRED_SERIES.items():
        raw = fetch_fred_series(series_id)
        raw.to_csv(os.path.join(raw_dir, f"fred_{series_id}.csv"), index=False)
        fred_frames.append(raw)
        sources.append(EmpiricalSource(meta["label"], meta["url"], f"Downloaded as FRED CSV series {series_id}."))
    fred_raw = pd.concat(fred_frames, ignore_index=True)
    annual = annualize_fred(fred_raw, start_year, end_year)

    acs, acs_year = fetch_acs_gini(range(end_year, start_year - 1, -1))
    if acs is not None:
        acs.to_csv(os.path.join(raw_dir, f"census_acs_gini_{acs_year}.csv"), index=False)
        annual = pd.concat([annual, acs.assign(year=pd.to_datetime(acs["date"]).dt.year)[["year", "variable", "value", "source_url"]]], ignore_index=True)
        sources.append(EmpiricalSource("ACS national income Gini", str(acs["source_url"].iloc[0]), "Latest available ACS 1-year estimate."))
    else:
        sources.append(EmpiricalSource("ACS national income Gini", "https://api.census.gov/data.html", "No usable ACS Gini response was returned.", "not used"))

    bds = fetch_bds_firm_exits(range(start_year, min(end_year, 2023) + 1))
    if not bds.empty:
        bds.to_csv(os.path.join(raw_dir, "census_bds_firm_exits.csv"), index=False)
        annual = pd.concat([annual, bds.assign(year=pd.to_datetime(bds["date"]).dt.year)[["year", "variable", "value", "source_url"]]], ignore_index=True)
        sources.append(EmpiricalSource("BDS firm exits", "https://api.census.gov/data/timeseries/bds", "Firm deaths divided by firm counts."))
    else:
        sources.append(EmpiricalSource("BDS firm exits", "https://api.census.gov/data/timeseries/bds", "No usable BDS response was returned.", "not used"))

    sources.append(EmpiricalSource(
        "AI adoption point estimate",
        "https://www.federalreserve.gov/econres/notes/feds-notes/monitoring-ai-adoption-in-the-u-s-economy-20260403.html",
        "Federal Reserve note reports Census BTOS firm-level AI adoption at about 18 percent at year-end 2025.",
    ))

    annual_path = os.path.join(processed_dir, "empirical_annual_series.csv")
    targets_path = os.path.join(processed_dir, "calibration_targets_empirical.csv")
    docs_path = os.path.join(outdir, "empirical_data_sources.md")
    annual.sort_values(["variable", "year"]).to_csv(annual_path, index=False)
    targets = build_targets(annual, start_year=start_year, end_year=end_year, acs_gini=acs, bds_exits=bds, ai_adoption=ai_adoption)
    targets.to_csv(targets_path, index=False)
    write_source_document(docs_path, sources, targets_path, annual_path)
    return annual_path, targets_path, docs_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch public empirical calibration data and build model target files")
    parser.add_argument("--outdir", default="data/empirical", help="Output directory for raw, processed, and source files")
    parser.add_argument("--start-year", type=int, default=2015, help="First anchor year for annual targets")
    parser.add_argument("--end-year", type=int, default=2024, help="Last anchor year for annual targets")
    parser.add_argument("--ai-adoption", type=float, default=0.18, help="Documented AI adoption point estimate used for Adopt_final")
    args = parser.parse_args()
    annual_path, targets_path, docs_path = fetch_and_build(args.outdir, args.start_year, args.end_year, args.ai_adoption)
    print(f"Wrote annual series: {annual_path}")
    print(f"Wrote calibration targets: {targets_path}")
    print(f"Wrote source documentation: {docs_path}")


if __name__ == "__main__":
    main()

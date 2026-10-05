"""Utilities for aggregating heterogeneous empirical observations into annual targets."""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, Iterable

import numpy as np
import pandas as pd


def _aggregate(values: pd.Series, method: str) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if values.empty:
        return float("nan")
    method = method.lower()
    if method == "mean":
        return float(values.mean())
    if method == "sum":
        return float(values.sum())
    if method == "last":
        return float(values.iloc[-1])
    if method == "first":
        return float(values.iloc[0])
    raise ValueError(f"Unknown aggregation method: {method}")


def harmonize_long_data(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    variable_col: str = "variable",
    value_col: str = "value",
    aggregation: str = "mean",
    price_index_col: str | None = None,
    base_year: int | None = None,
    nominal_variables: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Convert long empirical data to annual, optionally real, values.

    Expected input columns are date, variable, and value. If `price_index_col`,
    `base_year`, and `nominal_variables` are supplied, selected variables are
    deflated to base-year price levels using:

        real_value = nominal_value * price_index_base / price_index_t
    """
    required = {date_col, variable_col, value_col}
    if not required.issubset(df.columns):
        raise ValueError(f"Input data must contain columns: {sorted(required)}")
    data = df.copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data = data[data[date_col].notna()].copy()
    data["year"] = data[date_col].dt.year.astype(int)
    data[value_col] = pd.to_numeric(data[value_col], errors="coerce")
    data = data[data[value_col].notna()].copy()
    value_for_aggregation = value_col
    if price_index_col and base_year is not None and nominal_variables:
        if price_index_col not in data.columns:
            raise ValueError(f"Missing price index column: {price_index_col}")
        data[price_index_col] = pd.to_numeric(data[price_index_col], errors="coerce")
        base_index_values = data.loc[data["year"] == int(base_year), price_index_col].dropna()
        if base_index_values.empty:
            raise ValueError(f"No price index observations found for base_year={base_year}")
        base_index = float(base_index_values.mean())
        nominal = {str(x) for x in nominal_variables}
        data["real_value"] = data[value_col]
        mask = data[variable_col].astype(str).isin(nominal)
        data.loc[mask, "real_value"] = data.loc[mask, value_col] * base_index / data.loc[mask, price_index_col]
        value_for_aggregation = "real_value"
    rows = []
    for (variable, year), group in data.sort_values(date_col).groupby([variable_col, "year"]):
        rows.append({
            "variable": str(variable),
            "year": int(year),
            "value": _aggregate(group[value_for_aggregation], aggregation),
            "aggregation": aggregation,
            "real_base_year": int(base_year) if value_for_aggregation == "real_value" else np.nan,
        })
    return pd.DataFrame(rows).sort_values(["variable", "year"]).reset_index(drop=True)


def annual_targets_from_harmonized(
    annual_df: pd.DataFrame,
    target_map: Dict[str, str],
    *,
    weight: float = 1.0,
    group: str = "calibration",
) -> pd.DataFrame:
    """Create a loadable target table from harmonized annual data.

    `target_map` maps target moment names to harmonized variable names. The
    target value is the mean annual value for that variable.
    """
    rows = []
    for target_name, variable in target_map.items():
        values = annual_df.loc[annual_df["variable"] == variable, "value"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        rows.append({
            "name": target_name,
            "target": float(np.mean(values)) if len(values) else float("nan"),
            "weight": float(weight),
            "group": group,
            "units": "harmonized annual",
            "frequency": "annual",
            "source": "harmonized empirical input",
            "transformation": f"annual {variable} mean",
            "source_notes": "Review and replace with final source documentation before reporting.",
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Harmonize empirical long-format data to annual targets")
    parser.add_argument("--input", required=True, help="Input CSV with date,variable,value columns")
    parser.add_argument("--outdir", default="outputs/harmonized_data", help="Output directory")
    parser.add_argument("--date-col", default="date")
    parser.add_argument("--variable-col", default="variable")
    parser.add_argument("--value-col", default="value")
    parser.add_argument("--aggregation", default="mean", choices=["mean", "sum", "last", "first"])
    parser.add_argument("--price-index-col", default=None)
    parser.add_argument("--base-year", type=int, default=None)
    parser.add_argument("--nominal-variable", action="append", default=[], help="Variable to deflate; repeat as needed")
    parser.add_argument("--target-map", default=None, help="JSON mapping target moment names to harmonized variable names")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    raw = pd.read_csv(args.input)
    annual = harmonize_long_data(
        raw,
        date_col=args.date_col,
        variable_col=args.variable_col,
        value_col=args.value_col,
        aggregation=args.aggregation,
        price_index_col=args.price_index_col,
        base_year=args.base_year,
        nominal_variables=args.nominal_variable,
    )
    annual.to_csv(os.path.join(args.outdir, "harmonized_annual.csv"), index=False)
    if args.target_map:
        with open(args.target_map, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        annual_targets_from_harmonized(annual, mapping).to_csv(
            os.path.join(args.outdir, "harmonized_targets_template.csv"),
            index=False,
        )


if __name__ == "__main__":
    main()

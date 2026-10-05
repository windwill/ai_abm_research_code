"""Standalone charts for calibration, scenario, debt, and occupation results."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CALIBRATION_MOMENTS = ["U_mean", "Y_growth", "Adopt_final", "Debt_to_GDP_final", "Ineq_mean"]
SCENARIO_METRICS = ["U_mean", "Y_growth", "Adopt_final", "Debt_to_GDP_final", "Ineq_mean", "W_mean"]


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _save(fig: plt.Figure, outpath: Path) -> None:
    _ensure_dir(outpath)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def load_targets_csv(path: str | Path) -> pd.Series:
    """Load target values from the standard target CSV format."""
    df = pd.read_csv(path)
    if not {"name", "target"}.issubset(df.columns):
        raise ValueError("Target CSV must contain `name` and `target` columns")
    return df.set_index("name")["target"].astype(float)


def plot_calibration_moments(
    baseline_path: str | Path,
    calibrated_path: str | Path,
    targets_path: str | Path,
    outpath: str | Path,
    *,
    moments: Iterable[str] = CALIBRATION_MOMENTS,
) -> None:
    """Plot empirical targets against baseline and calibrated simulated moments."""
    baseline = pd.read_csv(baseline_path).iloc[0]
    calibrated = pd.read_csv(calibrated_path).iloc[0]
    targets = load_targets_csv(targets_path)
    names = [m for m in moments if m in baseline.index and m in calibrated.index and m in targets.index]
    x = np.arange(len(names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(x - width, [float(baseline[m]) for m in names], width, label="Baseline")
    ax.bar(x, [float(calibrated[m]) for m in names], width, label="GA candidate")
    ax.scatter(x + width, [float(targets[m]) for m in names], marker="D", color="black", label="Empirical target", zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_title("Calibration Moments: Target, Baseline, and GA Candidate")
    ax.set_ylabel("Moment value")
    ax.legend()
    _save(fig, Path(outpath))


def plot_loss_trace(history_path: str | Path, outpath: str | Path) -> None:
    """Plot optimizer best, mean, and median loss traces when available."""
    hist = pd.read_csv(history_path)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(hist["iteration"], hist["best_loss"], marker="o", label="Best loss")
    for col, label in [("mean_loss", "Mean loss"), ("median_loss", "Median loss")]:
        if col in hist.columns:
            ax.plot(hist["iteration"], hist[col], marker=".", label=label)
    ax.set_title("Calibration Loss Trace")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("MSM loss")
    ax.legend()
    _save(fig, Path(outpath))


def plot_scenario_heatmap(
    comparison_path: str | Path,
    metric: str,
    outpath: str | Path,
) -> None:
    """Plot an A x L x P scenario heatmap for one metric.

    Scenario identifiers are expected to use labels such as A1+L2+P3.
    Policy scenarios become columns, and automation-labour combinations
    become rows.
    """
    df = pd.read_csv(comparison_path)
    if metric not in df.columns:
        raise ValueError(f"Metric `{metric}` not found in {comparison_path}")
    parsed = df["scenario_id"].str.extract(r"(?P<A>A\d)\+(?P<L>L\d)\+(?P<P>P\d)")
    work = df.join(parsed)
    work = work.dropna(subset=["A", "L", "P"])
    work["row"] = work["A"] + "+" + work["L"]
    pivot = work.pivot(index="row", columns="P", values=metric).sort_index()
    fig, ax = plt.subplots(figsize=(6.5, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title(f"Scenario Heatmap: {metric}")
    ax.set_xlabel("Policy scenario")
    ax.set_ylabel("Automation + labour scenario")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="white" if val > np.nanmean(pivot.values) else "black", fontsize=8)
    fig.colorbar(im, ax=ax, label=metric)
    _save(fig, Path(outpath))


def plot_debt_cap(comparison_path: str | Path, outpath: str | Path, *, cap: float = 5.0) -> None:
    """Plot final debt ratio by scenario with a horizontal cap line."""
    df = pd.read_csv(comparison_path).sort_values("scenario_id")
    metric = "Debt_to_GDP_final"
    fig, ax = plt.subplots(figsize=(11, 4.8))
    colors = ["#b33" if float(v) >= cap - 1e-9 else "#4c78a8" for v in df[metric]]
    ax.bar(df["scenario_id"], df[metric], color=colors)
    ax.axhline(cap, color="black", linestyle="--", linewidth=1.2, label=f"Debt cap = {cap:g}")
    ax.set_title("Debt-Cap Diagnostic by Scenario")
    ax.set_ylabel("Debt-to-output ratio")
    ax.tick_params(axis="x", rotation=60)
    ax.legend()
    _save(fig, Path(outpath))


def plot_occupation_heatmap(occupation_table_path: str | Path, value_col: str, outpath: str | Path) -> None:
    """Plot occupation by scenario heatmap for unemployment, wage, or separations."""
    df = pd.read_csv(occupation_table_path) if str(occupation_table_path).endswith(".csv") else _read_markdown_table(occupation_table_path)
    if value_col not in df.columns:
        raise ValueError(f"Column `{value_col}` not found in {occupation_table_path}")
    pivot = df.pivot(index="occupation", columns="scenario", values=value_col)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="magma")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title(f"Occupation Scenario Heatmap: {value_col}")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.iloc[i, j]:.2f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=ax, label=value_col)
    _save(fig, Path(outpath))


def plot_occupation_grouped_bars(occupation_table_path: str | Path, value_col: str, outpath: str | Path) -> None:
    """Plot grouped bars for an occupation metric across scenarios."""
    df = pd.read_csv(occupation_table_path) if str(occupation_table_path).endswith(".csv") else _read_markdown_table(occupation_table_path)
    pivot = df.pivot(index="occupation", columns="scenario", values=value_col)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(pivot.index))
    width = 0.8 / max(1, len(pivot.columns))
    for k, scenario in enumerate(pivot.columns):
        ax.bar(x - 0.4 + width / 2 + k * width, pivot[scenario], width, label=scenario)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=30, ha="right")
    ax.set_title(f"Occupation Impacts: {value_col}")
    ax.set_ylabel(value_col)
    ax.legend(ncol=min(4, len(pivot.columns)))
    _save(fig, Path(outpath))


def write_occupation_table_csv(markdown_path: str | Path, outpath: str | Path) -> None:
    """Extract the main occupation diagnostics table from the generated markdown report."""
    df = _read_markdown_table(markdown_path)
    _ensure_dir(Path(outpath))
    df.to_csv(outpath, index=False)


def _read_markdown_table(path: str | Path) -> pd.DataFrame:
    """Read the first markdown pipe table in a generated report table file."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    table_lines = []
    in_table = False
    for line in lines:
        if line.startswith("|"):
            in_table = True
            table_lines.append(line)
        elif in_table:
            break
    if len(table_lines) < 3:
        raise ValueError(f"No markdown table found in {path}")
    header = [x.strip() for x in table_lines[0].strip("|").split("|")]
    rows = []
    for line in table_lines[2:]:
        rows.append([x.strip() for x in line.strip("|").split("|")])
    df = pd.DataFrame(rows, columns=header)
    for col in df.columns:
        if col not in {"scenario", "occupation"}:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

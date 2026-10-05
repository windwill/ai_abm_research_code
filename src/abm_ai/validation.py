"""Calibration error summaries and empirical distribution comparison tests."""

from __future__ import annotations

from typing import Dict, Tuple

import math
import numpy as np
import pandas as pd


def validation_errors(
    moments: Dict[str, float],
    targets: Dict[str, Tuple[float, float]],
    *,
    label: str = "baseline",
    eps: float = 1e-9,
) -> pd.DataFrame:
    """Return per-moment validation errors against target moments.

    The normalized error divides by ``max(abs(target), eps)`` so moments with
    different scales can be summarized in a single NRMSE statistic.
    """
    rows = []
    for moment, (target, weight) in targets.items():
        simulated = float(moments.get(moment, np.nan))
        target = float(target)
        error = simulated - target if math.isfinite(simulated) else np.nan
        scale = max(abs(target), eps)
        normalized_error = error / scale if math.isfinite(error) else np.nan
        rows.append({
            "label": label,
            "moment": moment,
            "simulated": simulated,
            "target": target,
            "weight": float(weight),
            "error": error,
            "abs_error": abs(error) if math.isfinite(error) else np.nan,
            "squared_error": error * error if math.isfinite(error) else np.nan,
            "normalized_error": normalized_error,
            "normalized_squared_error": normalized_error * normalized_error if math.isfinite(normalized_error) else np.nan,
        })
    return pd.DataFrame(rows)


def nrmse(validation_df: pd.DataFrame) -> float:
    """Compute normalized root mean squared error from validation rows."""
    values = validation_df["normalized_squared_error"].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan")
    return float(np.sqrt(np.mean(values)))


def mape(validation_df: pd.DataFrame) -> float:
    """Compute mean absolute percentage error from validation rows."""
    values = np.abs(validation_df["normalized_error"].to_numpy(dtype=float))
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan")
    return float(np.mean(values) * 100.0)


def validation_summary(validation_df: pd.DataFrame, *, label: str | None = None) -> pd.DataFrame:
    """Return aggregate validation metrics for one validation table."""
    if label is None:
        labels = validation_df["label"].dropna().unique()
        label = str(labels[0]) if len(labels) == 1 else "combined"
    finite = validation_df[np.isfinite(validation_df["normalized_squared_error"].to_numpy(dtype=float))]
    return pd.DataFrame([{
        "label": label,
        "n_moments": int(len(validation_df)),
        "n_finite": int(len(finite)),
        "nrmse": nrmse(validation_df),
        "mape": mape(validation_df),
        "mean_abs_error": float(finite["abs_error"].mean()) if len(finite) else float("nan"),
        "max_abs_error": float(finite["abs_error"].max()) if len(finite) else float("nan"),
        "weighted_loss": float((finite["weight"] * finite["squared_error"]).sum()) if len(finite) else float("nan"),
    }])


def validate_moments(
    moments: Dict[str, float],
    targets: Dict[str, Tuple[float, float]],
    *,
    label: str = "baseline",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-moment errors and aggregate validation summary."""
    errors = validation_errors(moments, targets, label=label)
    return errors, validation_summary(errors, label=label)


def ks_2sample_statistic(x, y) -> float:
    """Return the two-sample Kolmogorov-Smirnov statistic."""
    x_arr = np.sort(np.asarray(x, dtype=float))
    y_arr = np.sort(np.asarray(y, dtype=float))
    x_arr = x_arr[np.isfinite(x_arr)]
    y_arr = y_arr[np.isfinite(y_arr)]
    if len(x_arr) == 0 or len(y_arr) == 0:
        return float("nan")
    pooled = np.sort(np.concatenate([x_arr, y_arr]))
    Fx = np.searchsorted(x_arr, pooled, side="right") / len(x_arr)
    Fy = np.searchsorted(y_arr, pooled, side="right") / len(y_arr)
    return float(np.max(np.abs(Fx - Fy)))


def cvm_2sample_statistic(x, y) -> float:
    """Return a two-sample Cramer-von Mises style ECDF distance."""
    x_arr = np.sort(np.asarray(x, dtype=float))
    y_arr = np.sort(np.asarray(y, dtype=float))
    x_arr = x_arr[np.isfinite(x_arr)]
    y_arr = y_arr[np.isfinite(y_arr)]
    if len(x_arr) == 0 or len(y_arr) == 0:
        return float("nan")
    pooled = np.sort(np.concatenate([x_arr, y_arr]))
    Fx = np.searchsorted(x_arr, pooled, side="right") / len(x_arr)
    Fy = np.searchsorted(y_arr, pooled, side="right") / len(y_arr)
    scale = len(x_arr) * len(y_arr) / ((len(x_arr) + len(y_arr)) ** 2)
    return float(scale * np.mean((Fx - Fy) ** 2))


def load_empirical_distributions(path: str) -> Dict[str, np.ndarray]:
    """Load empirical benchmark distributions from a long CSV file.

    The file must contain columns `variable` and `value`; variable names should
    match simulated history columns.
    """
    df = pd.read_csv(path)
    required = {"variable", "value"}
    if not required.issubset(df.columns):
        raise ValueError("Distribution benchmark CSV must have columns: variable,value")
    distributions: Dict[str, np.ndarray] = {}
    for variable, group in df.groupby("variable"):
        values = group["value"].to_numpy(dtype=float)
        distributions[str(variable)] = values[np.isfinite(values)]
    return distributions


def distributional_tests(
    simulated_history: pd.DataFrame,
    empirical_distributions: Dict[str, np.ndarray],
    *,
    label: str = "baseline",
) -> pd.DataFrame:
    """Compare simulated history distributions with empirical benchmarks."""
    rows = []
    for variable, empirical_values in empirical_distributions.items():
        if variable not in simulated_history.columns:
            simulated_values = np.array([], dtype=float)
            status = "missing_simulated_column"
        else:
            simulated_values = simulated_history[variable].to_numpy(dtype=float)
            status = "ok"
        simulated_values = simulated_values[np.isfinite(simulated_values)]
        empirical_values = np.asarray(empirical_values, dtype=float)
        empirical_values = empirical_values[np.isfinite(empirical_values)]
        if len(simulated_values) == 0 or len(empirical_values) == 0:
            status = "insufficient_data" if status == "ok" else status
        rows.append({
            "label": label,
            "variable": variable,
            "simulated_n": int(len(simulated_values)),
            "empirical_n": int(len(empirical_values)),
            "ks_stat": ks_2sample_statistic(simulated_values, empirical_values),
            "cvm_stat": cvm_2sample_statistic(simulated_values, empirical_values),
            "simulated_mean": float(np.mean(simulated_values)) if len(simulated_values) else float("nan"),
            "empirical_mean": float(np.mean(empirical_values)) if len(empirical_values) else float("nan"),
            "simulated_std": float(np.std(simulated_values, ddof=1)) if len(simulated_values) > 1 else float("nan"),
            "empirical_std": float(np.std(empirical_values, ddof=1)) if len(empirical_values) > 1 else float("nan"),
            "status": status,
        })
    return pd.DataFrame(rows)

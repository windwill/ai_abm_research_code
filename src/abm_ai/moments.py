"""Simulation moment summaries and simulated method of moments loss functions."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

def _safe_corr(x, y, *, min_obs: int = 3) -> float:
    """Return a finite-sample correlation, or NaN when the series cannot identify it."""
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    if len(x_arr) < min_obs:
        return float("nan")
    if float(np.std(x_arr)) <= 1e-12 or float(np.std(y_arr)) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def dynamic_diagnostic_moments(df: pd.DataFrame, *, window: int = 50) -> Dict[str, float]:
    """Return dynamic validation moments for macro relationships.

    The diagnostics are intended for validation and reporting rather than as
    default calibration targets. They are returned as NaN when the simulated
    path is too short or has too little variation to identify a correlation.
    """
    tail = df.tail(max(3, int(window) + 1))
    y = np.asarray(tail["Y"], dtype=float)
    u = np.asarray(tail["U"], dtype=float)
    w = np.asarray(tail["W_mean"], dtype=float)

    y_growth = np.diff(np.log(np.maximum(y, 1e-6)))
    u_change = np.diff(u)
    wage_growth = np.diff(np.log(np.maximum(w, 1e-6)))

    diagnostics = {
        "Okun_corr": _safe_corr(y_growth, u_change),
        "Phillips_corr": _safe_corr(u[1:], wage_growth),
    }
    if "Vacancies" in tail.columns:
        diagnostics["Beveridge_corr"] = _safe_corr(tail["U"], tail["Vacancies"])
    else:
        diagnostics["Beveridge_corr"] = float("nan")
    return diagnostics


def summarize_moments(df: pd.DataFrame) -> Dict[str, float]:
    """Return the MSM moment vector used for calibration and diagnostics."""
    y = np.asarray(df["Y"], dtype=float)
    y_growth = float(np.mean(np.diff(np.log(np.maximum(y, 1e-6)))[-50:])) if len(y) > 1 else 0.0
    firm_exit_series = df["Firm_exit_rate"] if "Firm_exit_rate" in df.columns else df["Firm_exits"]
    moments = {
        "U_mean": float(np.mean(df["U"][-50:])),
        "Y_growth": y_growth,
        "Adopt_final": float(df["Adopt_mean"].iloc[-1]),
        "Debt_final": float(df["Debt"].iloc[-1]),
        "Debt_to_GDP_final": float(df["Debt_to_GDP"].iloc[-1]) if "Debt_to_GDP" in df.columns else float("nan"),
        "Ineq_mean": float(np.mean(df["Ineq"][-50:])),
        "Ineq_all_workers_mean": float(np.mean(df["Ineq_all_workers"][-50:])) if "Ineq_all_workers" in df.columns else float("nan"),
        "W_mean": float(np.mean(df["W_mean"][-50:])),
        "Firm_exits_mean": float(np.mean(firm_exit_series[-50:])),
    }
    if "Resource_constraint" in df.columns:
        moments["Resource_constraint_mean"] = float(np.mean(df["Resource_constraint"][-50:]))
        moments["Resource_constraint_final"] = float(df["Resource_constraint"].iloc[-1])
    if "Exit_prob_mean" in df.columns:
        moments["Exit_prob_mean"] = float(np.mean(df["Exit_prob_mean"][-50:]))
    if "Exit_profit_shortfall_mean" in df.columns:
        moments["Exit_profit_shortfall_mean"] = float(np.mean(df["Exit_profit_shortfall_mean"][-50:]))
    if "Exit_below_profit_cutoff_share" in df.columns:
        moments["Exit_below_profit_cutoff_share_mean"] = float(np.mean(df["Exit_below_profit_cutoff_share"][-50:]))
    if "Exit_below_A_min_share" in df.columns:
        moments["Exit_below_A_min_share_mean"] = float(np.mean(df["Exit_below_A_min_share"][-50:]))
    if "Firm_profit_mean" in df.columns:
        moments["Firm_profit_mean"] = float(np.mean(df["Firm_profit_mean"][-50:]))
    if "Firm_profit_p10" in df.columns:
        moments["Firm_profit_p10_mean"] = float(np.mean(df["Firm_profit_p10"][-50:]))
    if "Firm_profit_margin_mean" in df.columns:
        moments["Firm_profit_margin_mean"] = float(np.mean(df["Firm_profit_margin_mean"][-50:]))
    if "Entry_vacancies" in df.columns:
        moments["Entry_vacancies_mean"] = float(np.mean(df["Entry_vacancies"][-50:]))
    if "Exit_layoffs" in df.columns:
        moments["Exit_layoffs_mean"] = float(np.mean(df["Exit_layoffs"][-50:]))
    if "Exit_reallocated_workers" in df.columns:
        moments["Exit_reallocated_workers_mean"] = float(np.mean(df["Exit_reallocated_workers"][-50:]))
    if "Exit_unabsorbed_workers" in df.columns:
        moments["Exit_unabsorbed_workers_mean"] = float(np.mean(df["Exit_unabsorbed_workers"][-50:]))
    if "Resource_demand" in df.columns:
        moments["Resource_demand_mean"] = float(np.mean(df["Resource_demand"][-50:]))
    if "MA_events" in df.columns:
        moments["MA_events_mean"] = float(np.mean(df["MA_events"][-50:]))
    if "MA_layoffs" in df.columns:
        moments["MA_layoffs_mean"] = float(np.mean(df["MA_layoffs"][-50:]))
    if "AI_cost" in df.columns:
        moments["AI_cost_mean"] = float(np.mean(df["AI_cost"][-50:]))
    if "AI_relative_cost_mean" in df.columns:
        moments["AI_relative_cost_mean"] = float(np.mean(df["AI_relative_cost_mean"][-50:]))
    if "AI_operating_cost" in df.columns:
        moments["AI_operating_cost_mean"] = float(np.mean(df["AI_operating_cost"][-50:]))
    if "Compute_demand" in df.columns:
        moments["Compute_demand_mean"] = float(np.mean(df["Compute_demand"][-50:]))
    if "Compute_unit_cost" in df.columns:
        moments["Compute_unit_cost_mean"] = float(np.mean(df["Compute_unit_cost"][-50:]))
    if "Compute_operating_cost" in df.columns:
        moments["Compute_operating_cost_mean"] = float(np.mean(df["Compute_operating_cost"][-50:]))
    if "Tech_share_top" in df.columns:
        moments["Tech_share_top_final"] = float(df["Tech_share_top"].iloc[-1])
    if "Tech_share_hhi" in df.columns:
        moments["Tech_share_hhi_mean"] = float(np.mean(df["Tech_share_hhi"][-50:]))
        moments["Tech_share_hhi_final"] = float(df["Tech_share_hhi"].iloc[-1])
    if "Tech_rnd_advantage" in df.columns:
        moments["Tech_rnd_advantage_mean"] = float(np.mean(df["Tech_rnd_advantage"][-50:]))
    if "Tech_markup_advantage" in df.columns:
        moments["Tech_markup_advantage_mean"] = float(np.mean(df["Tech_markup_advantage"][-50:]))
    if "Pilot_share" in df.columns:
        moments["Pilot_share_mean"] = float(np.mean(df["Pilot_share"][-50:]))
    if "Production_share" in df.columns:
        moments["Production_share_final"] = float(df["Production_share"].iloc[-1])
    if "Scaled_share" in df.columns:
        moments["Scaled_share_final"] = float(df["Scaled_share"].iloc[-1])
    if "Failed_pilots" in df.columns:
        moments["Failed_pilots_mean"] = float(np.mean(df["Failed_pilots"][-50:]))
    if "Data_readiness_mean" in df.columns:
        moments["Data_readiness_mean"] = float(np.mean(df["Data_readiness_mean"][-50:]))
    if "AI_error_events" in df.columns:
        moments["AI_error_events_mean"] = float(np.mean(df["AI_error_events"][-50:]))
    if "AI_operational_loss" in df.columns:
        moments["AI_operational_loss_mean"] = float(np.mean(df["AI_operational_loss"][-50:]))
    if "AI_error_rate_mean" in df.columns:
        moments["AI_error_rate_mean"] = float(np.mean(df["AI_error_rate_mean"][-50:]))
    if "Cyber_privacy_incidents" in df.columns:
        moments["Cyber_privacy_incidents_mean"] = float(np.mean(df["Cyber_privacy_incidents"][-50:]))
    if "Cyber_privacy_loss" in df.columns:
        moments["Cyber_privacy_loss_mean"] = float(np.mean(df["Cyber_privacy_loss"][-50:]))
    if "Cyber_privacy_risk_rate_mean" in df.columns:
        moments["Cyber_privacy_risk_rate_mean"] = float(np.mean(df["Cyber_privacy_risk_rate_mean"][-50:]))
    if "Cyber_privacy_control_cost" in df.columns:
        moments["Cyber_privacy_control_cost_mean"] = float(np.mean(df["Cyber_privacy_control_cost"][-50:]))
    if "AI_supervision_vacancies" in df.columns:
        moments["AI_supervision_vacancies_mean"] = float(np.mean(df["AI_supervision_vacancies"][-50:]))
    if "AI_complementary_hires" in df.columns:
        moments["AI_complementary_hires_mean"] = float(np.mean(df["AI_complementary_hires"][-50:]))
    if "AI_complementarity_premium" in df.columns:
        moments["AI_complementarity_premium_mean"] = float(np.mean(df["AI_complementarity_premium"][-50:]))
    if "New_task_share" in df.columns:
        moments["New_task_share_mean"] = float(np.mean(df["New_task_share"][-50:]))
    for col in df.columns:
        if col.startswith("Occ_U_"):
            moments[f"{col}_mean"] = float(np.mean(df[col][-50:]))
        elif col.startswith("Occ_W_"):
            moments[f"{col}_mean"] = float(np.mean(df[col][-50:]))
        elif col.startswith("Occ_Sep_"):
            moments[f"{col}_mean"] = float(np.mean(df[col][-50:]))
    moments.update(dynamic_diagnostic_moments(df))
    return moments

def get_default_targets() -> Dict[str, Tuple[float, float]]:
    """Default target moments and diagonal MSM weights."""
    return {
        "U_mean": (0.055, 10.0),
        "Y_growth": (0.018, 8.0),
        "Adopt_final": (0.20, 4.0),
        "Ineq_mean": (0.35, 3.0),
        "Debt_to_GDP_final": (0.9, 2.0),
    }

def covariance_msm_loss(
    moments: Dict[str, float],
    targets: Dict[str, Tuple[float, float]],
    covariance: Tuple[List[str], np.ndarray],
    *,
    ridge: float = 1e-8,
) -> float:
    """Mahalanobis-style MSM loss using a supplied moment covariance matrix."""
    names, matrix = covariance
    active = [
        (i, name)
        for i, name in enumerate(names)
        if name in targets and name in moments and np.isfinite(moments[name])
    ]
    if not active:
        return float("inf")
    idx = [i for i, _ in active]
    active_names = [name for _, name in active]
    residuals = np.array([float(moments[name]) - float(targets[name][0]) for name in active_names], dtype=float)
    cov = np.asarray(matrix, dtype=float)[np.ix_(idx, idx)]
    cov = cov + np.eye(len(active_names)) * float(ridge)
    inv_cov = np.linalg.pinv(cov)
    return float(residuals.T @ inv_cov @ residuals)


def msm_loss(
    moments: Dict[str, float],
    targets: Dict[str, Tuple[float, float]],
    covariance: Tuple[List[str], np.ndarray] | None = None,
) -> float:
    """Method-of-simulated-moments loss.

    By default, the loss uses diagonal scalar weights from the target table.
    If a covariance matrix is supplied, the loss switches to a Mahalanobis-style
    quadratic form over the moments covered by that matrix.
    """
    if covariance is not None:
        return covariance_msm_loss(moments, targets, covariance)
    L=0.0
    for k,(tgt,w) in targets.items():
        if k in moments and np.isfinite(moments[k]): L += w * (moments[k]-tgt)**2
    return float(L)

def loss(
    moments: Dict[str, float],
    targets: Dict[str, Tuple[float, float]],
    covariance: Tuple[List[str], np.ndarray] | None = None,
) -> float:
    """Backward-compatible alias for the MSM loss."""
    return msm_loss(moments, targets, covariance=covariance)

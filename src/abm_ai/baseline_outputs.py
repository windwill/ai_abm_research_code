"""Reference-simulation runner and numerical sanity checks."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from typing import Dict, Iterable

import numpy as np
import pandas as pd

from .config import load_config
from .moments import summarize_moments
from .reporting import export_figset_pdf, make_plots
from .simulation import Economy


OUTPUTS_OF_INTEREST = [
    "U",
    "Y",
    "Shock_productivity",
    "Shock_demand",
    "Shock_rate",
    "W_mean",
    "Adopt_mean",
    "Routine_mean",
    "Debt",
    "Ineq",
    "Firm_exits",
    "Separations",
    "Hires",
    "Vacancies",
    "Credit",
    "Tech_demand",
    "Tech_mu",
    "Tech_share_top",
    "Tech_share_hhi",
    "Tech_rnd_advantage",
    "Tech_markup_advantage",
    "Resource_capacity",
    "Resource_demand",
    "Resource_constraint",
    "MA_events",
    "MA_layoffs",
    "AI_cost",
    "AI_relative_cost_mean",
    "AI_operating_cost",
    "Compute_demand",
    "Compute_unit_cost",
    "Compute_operating_cost",
    "Pilot_share",
    "Production_share",
    "Scaled_share",
    "Failed_pilots",
    "Data_readiness_mean",
    "AI_error_events",
    "AI_operational_loss",
    "AI_error_rate_mean",
    "Cyber_privacy_incidents",
    "Cyber_privacy_loss",
    "Cyber_privacy_risk_rate_mean",
    "Cyber_privacy_control_cost",
    "AI_supervision_vacancies",
    "AI_complementary_hires",
    "AI_complementarity_premium",
    "New_task_share",
]


def _finite_series(df: pd.DataFrame, columns: Iterable[str]) -> Dict[str, bool]:
    return {col: bool(np.isfinite(df[col].to_numpy(dtype=float)).all()) for col in columns}


def simulation_sanity_checks(df: pd.DataFrame) -> Dict[str, object]:
    checks: Dict[str, object] = {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "finite_outputs": _finite_series(df, OUTPUTS_OF_INTEREST),
        "bounded_outputs": {
            "U": bool(((df["U"] >= 0.0) & (df["U"] <= 1.0)).all()),
            "Adopt_mean": bool(((df["Adopt_mean"] >= 0.0) & (df["Adopt_mean"] <= 1.0)).all()),
            "Routine_mean": bool(((df["Routine_mean"] >= 0.0) & (df["Routine_mean"] <= 1.0)).all()),
            "Ineq": bool(((df["Ineq"] >= 0.0) & (df["Ineq"] <= 1.0)).all()),
            "Credit": bool(((df["Credit"] >= 0.0) & (df["Credit"] <= 1.0)).all()),
            "Tech_share_top": bool(((df["Tech_share_top"] >= 0.0) & (df["Tech_share_top"] <= 1.0)).all()),
            "Tech_share_hhi": bool(((df["Tech_share_hhi"] >= 0.0) & (df["Tech_share_hhi"] <= 1.0)).all()),
            "Resource_constraint": bool(((df["Resource_constraint"] >= 0.0) & (df["Resource_constraint"] <= 1.0)).all()),
            "Pilot_share": bool(((df["Pilot_share"] >= 0.0) & (df["Pilot_share"] <= 1.0)).all()),
            "Production_share": bool(((df["Production_share"] >= 0.0) & (df["Production_share"] <= 1.0)).all()),
            "Scaled_share": bool(((df["Scaled_share"] >= 0.0) & (df["Scaled_share"] <= 1.0)).all()),
            "Data_readiness_mean": bool(((df["Data_readiness_mean"] >= 0.0) & (df["Data_readiness_mean"] <= 1.0)).all()),
            "AI_error_rate_mean": bool(((df["AI_error_rate_mean"] >= 0.0) & (df["AI_error_rate_mean"] <= 1.0)).all()),
            "Cyber_privacy_risk_rate_mean": bool(((df["Cyber_privacy_risk_rate_mean"] >= 0.0) & (df["Cyber_privacy_risk_rate_mean"] <= 1.0)).all()),
            "AI_complementarity_premium": bool(((df["AI_complementarity_premium"] >= 0.0) & (df["AI_complementarity_premium"] <= 1.0)).all()),
            "New_task_share": bool(((df["New_task_share"] >= 0.0) & (df["New_task_share"] <= 1.0)).all()),
        },
        "nonnegative_outputs": {
            "Y": bool((df["Y"] >= 0.0).all()),
            "Firm_exits": bool((df["Firm_exits"] >= 0.0).all()),
            "Separations": bool((df["Separations"] >= 0.0).all()),
            "Hires": bool((df["Hires"] >= 0.0).all()),
            "Vacancies": bool((df["Vacancies"] >= 0.0).all()),
            "Tech_mu": bool((df["Tech_mu"] >= 0.0).all()),
            "Tech_rnd_advantage": bool((df["Tech_rnd_advantage"] >= 0.0).all()),
            "Tech_markup_advantage": bool((df["Tech_markup_advantage"] >= 0.0).all()),
            "Resource_capacity": bool((df["Resource_capacity"] >= 0.0).all()),
            "Resource_demand": bool((df["Resource_demand"] >= 0.0).all()),
            "MA_events": bool((df["MA_events"] >= 0.0).all()),
            "MA_layoffs": bool((df["MA_layoffs"] >= 0.0).all()),
            "AI_cost": bool((df["AI_cost"] >= 0.0).all()),
            "AI_relative_cost_mean": bool((df["AI_relative_cost_mean"] >= 0.0).all()),
            "AI_operating_cost": bool((df["AI_operating_cost"] >= 0.0).all()),
            "Compute_demand": bool((df["Compute_demand"] >= 0.0).all()),
            "Compute_unit_cost": bool((df["Compute_unit_cost"] >= 0.0).all()),
            "Compute_operating_cost": bool((df["Compute_operating_cost"] >= 0.0).all()),
            "Failed_pilots": bool((df["Failed_pilots"] >= 0.0).all()),
            "AI_error_events": bool((df["AI_error_events"] >= 0.0).all()),
            "AI_operational_loss": bool((df["AI_operational_loss"] >= 0.0).all()),
            "Cyber_privacy_incidents": bool((df["Cyber_privacy_incidents"] >= 0.0).all()),
            "Cyber_privacy_loss": bool((df["Cyber_privacy_loss"] >= 0.0).all()),
            "Cyber_privacy_control_cost": bool((df["Cyber_privacy_control_cost"] >= 0.0).all()),
            "AI_supervision_vacancies": bool((df["AI_supervision_vacancies"] >= 0.0).all()),
            "AI_complementary_hires": bool((df["AI_complementary_hires"] >= 0.0).all()),
        },
    }
    checks["passed"] = bool(
        all(checks["finite_outputs"].values())
        and all(checks["bounded_outputs"].values())
        and all(checks["nonnegative_outputs"].values())
    )
    return checks


def generate_baseline_outputs(config: str | None, outdir: str, seed: int | None = None, T: int | None = None) -> Dict[str, object]:
    lp, tp, pp, sp = load_config(config)
    if seed is not None:
        sp.seed = int(seed)
    if T is not None:
        sp.T = int(T)

    os.makedirs(outdir, exist_ok=True)
    econ = Economy(lp, tp, pp, sp, outdir=outdir)
    history = econ.simulate()
    moments = summarize_moments(history)
    sanity = simulation_sanity_checks(history)

    history.to_csv(os.path.join(outdir, "section2_simulation_history.csv"), index=False)
    history[OUTPUTS_OF_INTEREST].to_csv(os.path.join(outdir, "section2_outputs_of_interest.csv"), index=False)
    pd.DataFrame([moments]).to_csv(os.path.join(outdir, "section2_moment_summary.csv"), index=False)
    with open(os.path.join(outdir, "section2_sanity_checks.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "sanity_checks": sanity,
                "parameters": {
                    "LaborParams": asdict(lp),
                    "TechParams": asdict(tp),
                    "PolicyParams": asdict(pp),
                    "SimulationParams": asdict(sp),
                },
            },
            f,
            indent=2,
        )
    make_plots(history, outdir)
    export_figset_pdf(history, os.path.join(outdir, "section2_baseline_figures.pdf"))
    return {"moments": moments, "sanity_checks": sanity, "outdir": outdir}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate baseline Section 2 ABM simulation outputs")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML or JSON config file")
    parser.add_argument("--outdir", type=str, default="outputs/baseline", help="Output directory")
    parser.add_argument("--seed", type=int, default=None, help="Override seed")
    parser.add_argument("--T", type=int, default=None, help="Override simulation length")
    args = parser.parse_args()

    result = generate_baseline_outputs(args.config, args.outdir, seed=args.seed, T=args.T)
    print("Section 2 baseline outputs generated.")
    print(f"Sanity checks passed: {result['sanity_checks']['passed']}")
    print(f"Outputs written to: {args.outdir}")


if __name__ == "__main__":
    main()

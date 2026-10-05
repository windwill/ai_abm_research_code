from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OCC_PREFIX_U = "Occ_U_"
OCC_PREFIX_W = "Occ_W_"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _clip_series(values: pd.Series | np.ndarray, floor: float | None = None, cap: float | None = None) -> pd.Series:
    s = pd.Series(values, dtype=float)
    if floor is not None:
        s = s.clip(lower=floor)
    if cap is not None:
        s = s.clip(upper=cap)
    return s


def _safe_div(num: pd.Series | np.ndarray | float, den: pd.Series | np.ndarray | float, eps: float = 1e-9) -> pd.Series:
    denominator = pd.Series(den, dtype=float).copy()
    denominator = denominator.mask(denominator.abs() < eps, eps)
    return pd.Series(num, dtype=float) / denominator


def _period_growth(y: pd.Series) -> pd.Series:
    y_safe = y.astype(float).clip(lower=1e-9)
    g = np.log(y_safe / y_safe.shift(1))
    return g.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _scenario_from_history(path: Path) -> tuple[str, int]:
    m = re.match(r"(.+)_history_rep(\d+)\.csv$", path.name)
    if not m:
        raise ValueError(f"Unexpected history file name: {path.name}")
    return m.group(1), int(m.group(2))


def load_histories(result_dir: Path) -> dict[str, list[pd.DataFrame]]:
    histories: dict[str, list[tuple[int, pd.DataFrame]]] = {}
    for path in sorted(result_dir.glob("*_history_rep*.csv")):
        scenario, rep = _scenario_from_history(path)
        df = pd.read_csv(path)
        df.insert(0, "period", np.arange(len(df)))
        df.insert(0, "replication", rep)
        df.insert(0, "scenario_id", scenario)
        histories.setdefault(scenario, []).append((rep, df))
    if "baseline" not in histories:
        raise FileNotFoundError(f"No baseline history files found in {result_dir}")
    return {k: [df for _, df in sorted(v, key=lambda x: x[0])] for k, v in histories.items()}


def mean_path(frames: list[pd.DataFrame]) -> pd.DataFrame:
    df = pd.concat(frames, ignore_index=True)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    keep = [c for c in numeric_cols if c not in {"replication"}]
    out = df.groupby("period", as_index=False)[keep].mean()
    return out


def load_occupation_weights(config_path: Path | None, histories: dict[str, list[pd.DataFrame]]) -> dict[str, float]:
    occs = sorted(
        c[len(OCC_PREFIX_U) :]
        for c in histories["baseline"][0].columns
        if c.startswith(OCC_PREFIX_U)
    )
    if not occs:
        return {}
    weights = {occ: 1.0 / len(occs) for occ in occs}
    if not config_path or not config_path.exists():
        return weights
    cfg = _read_json(config_path)
    sim = cfg.get("SimulationParams", cfg)
    specs = sim.get("occupation_specs", [])
    if not isinstance(specs, list):
        return weights
    parsed = {}
    for spec in specs:
        occ = str(spec.get("id", ""))
        if occ:
            parsed[occ] = float(spec.get("share", 0.0))
    total = sum(v for k, v in parsed.items() if k in occs)
    if total <= 0:
        return weights
    return {occ: parsed.get(occ, 0.0) / total for occ in occs}


def _col(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name in df.columns:
        return df[name].astype(float)
    return pd.Series(default, index=df.index, dtype=float)


def compute_translation_paths(
    scenario_path: pd.DataFrame,
    baseline_path: pd.DataFrame,
    assumptions: dict[str, Any],
    occ_weights: dict[str, float],
) -> pd.DataFrame:
    df = scenario_path.copy()
    base = baseline_path.copy()
    out = pd.DataFrame({"period": df["period"].astype(int)})

    u = _col(df, "U")
    u0 = _col(base, "U")
    w = _col(df, "W_mean")
    w0 = _col(base, "W_mean")
    y_growth = _period_growth(_col(df, "Y"))
    y_growth0 = _period_growth(_col(base, "Y"))
    debt = _col(df, "Debt_to_GDP")
    debt0 = _col(base, "Debt_to_GDP")
    ineq = _col(df, "Ineq_all_workers", _col(df, "Ineq").mean())
    ineq0 = _col(base, "Ineq_all_workers", _col(base, "Ineq").mean())

    life = assumptions.get("life", {})
    life_floor = float(life.get("index_floor", 0.25))
    wage_shortfall = ((w0 - w) / w0.replace(0.0, np.nan)).clip(lower=0.0).fillna(0.0)
    out["Life_lapse_index"] = _clip_series(
        1.0
        + float(life.get("beta_unemployment_lapse", 0.0)) * (u - u0)
        + float(life.get("beta_wage_shortfall_lapse", 0.0)) * wage_shortfall
        + float(life.get("beta_inequality_lapse", 0.0)) * (ineq - ineq0),
        floor=life_floor,
    )
    out["Life_new_business_index"] = _clip_series(
        1.0
        + float(life.get("beta_output_new_business", 0.0)) * (y_growth - y_growth0)
        - float(life.get("beta_unemployment_new_business", 0.0)) * (u - u0),
        floor=life_floor,
    )
    out["Life_income_stress_index"] = _clip_series(
        1.0 + 2.0 * (u - u0) + 0.5 * (ineq - ineq0) + wage_shortfall,
        floor=life_floor,
    )

    wc = assumptions.get("workers_comp", {})
    rates = wc.get("occupation_rates", {})
    freqs = wc.get("occupation_frequency", {})
    severities = wc.get("occupation_severity", {})
    gamma_u = float(wc.get("gamma_unemployment_frequency", 0.0))
    gamma_w = float(wc.get("gamma_wage_severity", 1.0))
    wc_floor = float(wc.get("index_floor", 0.05))
    payroll = pd.Series(0.0, index=df.index)
    payroll0 = pd.Series(0.0, index=df.index)
    premium = pd.Series(0.0, index=df.index)
    premium0 = pd.Series(0.0, index=df.index)
    claims = pd.Series(0.0, index=df.index)
    claims0 = pd.Series(0.0, index=df.index)
    occ_claim_deltas: dict[str, float] = {}
    for occ, weight in occ_weights.items():
        occ_u = _col(df, f"{OCC_PREFIX_U}{occ}")
        occ_u0 = _col(base, f"{OCC_PREFIX_U}{occ}")
        occ_w = _col(df, f"{OCC_PREFIX_W}{occ}", 1.0)
        occ_w0 = _col(base, f"{OCC_PREFIX_W}{occ}", 1.0)
        occ_payroll = weight * (1.0 - occ_u).clip(lower=0.0) * occ_w
        occ_payroll0 = weight * (1.0 - occ_u0).clip(lower=0.0) * occ_w0
        rate = float(rates.get(occ, 1.0))
        freq0 = float(freqs.get(occ, 1.0))
        sev0 = float(severities.get(occ, 1.0))
        freq = freq0 * (1.0 + gamma_u * (occ_u - occ_u0).clip(lower=0.0))
        severity = sev0 * _clip_series(_safe_div(occ_w, occ_w0), floor=wc_floor) ** gamma_w
        severity0 = pd.Series(sev0, index=df.index)
        occ_claim = occ_payroll * freq * severity
        occ_claim0 = occ_payroll0 * freq0 * severity0
        payroll += occ_payroll
        payroll0 += occ_payroll0
        premium += occ_payroll * rate
        premium0 += occ_payroll0 * rate
        claims += occ_claim
        claims0 += occ_claim0
        occ_claim_deltas[occ] = float((occ_claim - occ_claim0).mean())

    out["WC_payroll_index"] = _clip_series(_safe_div(payroll, payroll0), floor=wc_floor)
    out["WC_premium_index"] = _clip_series(_safe_div(premium, premium0), floor=wc_floor)
    out["WC_claim_cost_index"] = _clip_series(_safe_div(claims, claims0), floor=wc_floor)
    out["WC_loss_ratio_proxy"] = _clip_series(_safe_div(claims, premium), floor=wc_floor)

    pension = assumptions.get("pension", {})
    contribution_index = _clip_series(_safe_div((1.0 - u).clip(lower=0.0) * w, (1.0 - u0).clip(lower=0.0) * w0), floor=0.01)
    wage_growth = _period_growth(w)
    ma = _col(df, "MA_layoffs") / 100.0
    asset_return = (
        float(pension.get("base_asset_return", 0.055))
        + float(pension.get("alpha_growth_asset", 0.0)) * (y_growth - y_growth0)
        - float(pension.get("alpha_unemployment_asset", 0.0)) * (u - u0)
        - float(pension.get("alpha_debt_asset", 0.0)) * (debt - debt0).clip(lower=0.0)
    )
    asset_return = _clip_series(
        asset_return,
        floor=float(pension.get("return_floor", -0.3)),
        cap=float(pension.get("return_cap", 0.3)),
    )
    liability_growth = _clip_series(
        float(pension.get("base_liability_growth", 0.04))
        + float(pension.get("salary_weight", 0.5)) * wage_growth,
        floor=float(pension.get("liability_growth_floor", -0.1)),
        cap=float(pension.get("liability_growth_cap", 0.2)),
    )
    normal_cost = float(pension.get("normal_cost_rate", 0.08))
    benefit_rate = float(pension.get("benefit_payment_rate", 0.04))
    funded_ratio_raw = _roll_funded_ratio(
        asset_return,
        liability_growth,
        contribution_index,
        initial_funded_ratio=float(pension.get("initial_funded_ratio", 1.0)),
        normal_cost=normal_cost,
        benefit_rate=benefit_rate,
    )
    baseline_liability_growth = _clip_series(
        float(pension.get("base_liability_growth", 0.04))
        + float(pension.get("salary_weight", 0.5)) * _period_growth(w0),
        floor=float(pension.get("liability_growth_floor", -0.1)),
        cap=float(pension.get("liability_growth_cap", 0.2)),
    )
    baseline_asset_return = pd.Series(float(pension.get("base_asset_return", 0.055)), index=df.index)
    baseline_funded_ratio_raw = _roll_funded_ratio(
        baseline_asset_return,
        baseline_liability_growth,
        pd.Series(1.0, index=df.index),
        initial_funded_ratio=float(pension.get("initial_funded_ratio", 1.0)),
        normal_cost=normal_cost,
        benefit_rate=benefit_rate,
    )
    funded_ratio = _safe_div(funded_ratio_raw, baseline_funded_ratio_raw, eps=1e-30)
    if "funded_ratio_cap" in pension:
        funded_ratio = funded_ratio.clip(upper=float(pension.get("funded_ratio_cap", 3.0)))
    out["Pension_contribution_base_index"] = contribution_index
    out["Pension_funded_ratio_proxy"] = funded_ratio
    out["Pension_sponsor_stress_index"] = _clip_series(
        1.0
        + float(pension.get("sponsor_unemployment_weight", 1.0)) * (u - u0)
        - float(pension.get("sponsor_growth_weight", 1.0)) * (y_growth - y_growth0)
        + float(pension.get("sponsor_debt_weight", 0.2)) * (debt - debt0).clip(lower=0.0)
        + float(pension.get("sponsor_ma_weight", 0.02)) * ma,
        floor=0.0,
    )

    inv = assumptions.get("investment", {})
    equity_return = (
        float(inv.get("base_equity_return", 0.06))
        + float(inv.get("alpha_growth_equity", 0.0)) * (y_growth - y_growth0)
        - float(inv.get("alpha_unemployment_equity", 0.0)) * (u - u0)
    )
    equity_return = _clip_series(
        equity_return,
        floor=float(inv.get("return_floor", -0.4)),
        cap=float(inv.get("return_cap", 0.35)),
    )
    spread = _clip_series(
        float(inv.get("base_credit_spread", 0.015))
        + float(inv.get("alpha_unemployment_credit_spread", 0.0)) * (u - u0).clip(lower=0.0)
        + float(inv.get("alpha_debt_credit_spread", 0.0)) * (debt - debt0).clip(lower=0.0),
        floor=float(inv.get("spread_floor", 0.0)),
    )
    spread0 = float(inv.get("base_credit_spread", 0.015))
    base_bond_return = float(inv.get("base_bond_return", 0.035))
    core_bond_weight = float(inv.get("bond_weight", 0.45))
    credit_weight = float(inv.get("credit_weight", 0.20))
    fixed_income_weight = core_bond_weight + credit_weight
    credit_share_within_bonds = (
        credit_weight / fixed_income_weight if fixed_income_weight > 0 else 0.0
    )
    bond_return = base_bond_return + credit_share_within_bonds * (
        spread - float(inv.get("credit_duration", 4.0)) * (spread - spread0)
    )
    portfolio_return = (
        float(inv.get("equity_weight", 0.35)) * equity_return
        + fixed_income_weight * bond_return
    )
    out["Investment_equity_return_proxy"] = equity_return
    out["Investment_bond_return_proxy"] = bond_return
    out["Investment_credit_spread_proxy"] = spread
    out["Investment_portfolio_return_proxy"] = portfolio_return
    # Screening proxy only. A downstream ALM model can supply a time-varying
    # required return derived from its liability portfolio.
    required_return = pd.Series(float(inv.get("required_return", 0.05)), index=df.index)
    out["ALM_stress_index"] = _clip_series(1.0 + (required_return - portfolio_return).clip(lower=0.0), floor=0.0)

    op = assumptions.get("operational", {})
    y_scale = _col(df, "Y").replace(0.0, np.nan).clip(lower=1e-9)
    ai_loss = _col(df, "AI_operational_loss") / y_scale
    cyber_loss = _col(df, "Cyber_privacy_loss") / y_scale
    compute_cost = _col(df, "Compute_operating_cost") / y_scale
    control_cost = _col(df, "Cyber_privacy_control_cost") / y_scale
    out["Operational_loss_stress_index"] = _clip_series(
        1.0
        + float(op.get("ai_loss_weight", 1.0)) * ai_loss
        + float(op.get("cyber_loss_weight", 1.0)) * cyber_loss
        + float(op.get("compute_cost_weight", 0.5)) * compute_cost,
        floor=0.0,
    )
    out["Expense_pressure_index"] = _clip_series(
        1.0
        + float(op.get("compute_cost_weight", 0.5)) * compute_cost
        + float(op.get("control_cost_weight", 0.5)) * control_cost,
        floor=0.0,
    )
    out["Governance_need_index"] = _clip_series(
        float(op.get("governance_data_readiness_weight", 1.0)) * (1.0 - _col(df, "Data_readiness_mean", 1.0))
        + float(op.get("governance_ai_error_weight", 2.0)) * _col(df, "AI_error_rate_mean")
        + float(op.get("governance_cyber_weight", 2.0)) * _col(df, "Cyber_privacy_risk_rate_mean")
        + float(op.get("governance_supplier_hhi_weight", 0.5)) * _col(df, "Tech_share_hhi"),
        floor=0.0,
    )

    out.attrs["occ_claim_deltas"] = occ_claim_deltas
    return out


def _roll_funded_ratio(
    asset_return: pd.Series,
    liability_growth: pd.Series,
    contribution_index: pd.Series,
    *,
    initial_funded_ratio: float,
    normal_cost: float,
    benefit_rate: float,
) -> pd.Series:
    assets = [float(initial_funded_ratio)]
    liabilities = [1.0]
    for i in range(1, len(asset_return)):
        contrib = normal_cost * float(contribution_index.iloc[i])
        benefits = benefit_rate * liabilities[-1]
        accruals = normal_cost * float(contribution_index.iloc[i])
        assets.append(max(1e-9, assets[-1] * (1.0 + float(asset_return.iloc[i])) + contrib - benefits))
        liabilities.append(max(1e-9, liabilities[-1] * (1.0 + float(liability_growth.iloc[i])) + accruals - benefits))
    return pd.Series(assets, index=asset_return.index) / pd.Series(liabilities, index=asset_return.index)


def summarize_path(scenario_id: str, path: pd.DataFrame) -> dict[str, float | str]:
    row: dict[str, float | str] = {"scenario_id": scenario_id}
    metrics = [c for c in path.columns if c != "period"]
    for col in metrics:
        s = path[col].astype(float)
        row[f"{col}_mean"] = float(s.mean())
        row[f"{col}_final"] = float(s.iloc[-1])
        if "funded_ratio" in col or "return" in col:
            row[f"{col}_min"] = float(s.min())
        if "stress" in col or "lapse" in col or "loss_ratio" in col:
            row[f"{col}_max"] = float(s.max())
    return row


def write_markdown(
    summary: pd.DataFrame,
    outpath: Path,
    source_dir: Path,
    assumptions_path: Path,
    assumptions: dict[str, Any],
) -> None:
    selected_cols = [
        "scenario_id",
        "Life_lapse_index_mean",
        "Life_new_business_index_mean",
        "WC_payroll_index_mean",
        "WC_claim_cost_index_mean",
        "WC_loss_ratio_proxy_mean",
        "Pension_contribution_base_index_mean",
        "Pension_funded_ratio_proxy_final",
        "Pension_sponsor_stress_index_mean",
        "Investment_portfolio_return_proxy_mean",
        "Investment_credit_spread_proxy_mean",
        "Operational_loss_stress_index_mean",
        "Expense_pressure_index_mean",
        "Governance_need_index_mean",
    ]
    cols = [c for c in selected_cols if c in summary.columns]
    table = summary[cols].copy()
    for c in table.columns:
        if c != "scenario_id":
            table[c] = table[c].map(lambda x: f"{x:.4f}")
    table_md = _markdown_table(table)
    lines = [
        "# Actuarial Translation Results",
        "",
        f"Source scenario folder: `{source_dir.as_posix()}`",
        "",
        f"Assumption file: `{assumptions_path.as_posix()}`",
        "",
        str(
            assumptions.get(
                "description",
                "The following metrics translate ABM scenario paths into actuarial stress indicators. They are not company-specific pricing, reserving, pension valuation, ALM, or capital model outputs.",
            )
        ),
        "",
        "## Selected Scenario-Level Stress Indicators",
        "",
        table_md,
        "",
        "## Interpretation Notes",
        "",
        "- `Life_lapse_index_mean` above 1 indicates higher lapse-pressure stress than the selected baseline path.",
        "- `Life_new_business_index_mean` below 1 indicates weaker new-business or savings-capacity pressure.",
        "- `WC_payroll_index_mean` and `WC_claim_cost_index_mean` translate occupation employment and wage paths into workers' compensation exposure and claim-cost proxies.",
        "- `Pension_funded_ratio_proxy_final` is an illustrative roll-forward proxy, not a pension valuation result. It is measured relative to the selected baseline roll-forward. Values above 1 can occur in adverse scenarios if salary-linked liabilities compress faster than assets; it should therefore be read together with `Pension_contribution_base_index_mean` and `Pension_sponsor_stress_index_mean`.",
        "- `Investment_portfolio_return_proxy_mean` and `Investment_credit_spread_proxy_mean` are simplified ALM stress proxies.",
        "- `Operational_loss_stress_index_mean`, `Expense_pressure_index_mean`, and `Governance_need_index_mean` translate compute, cyber/privacy, AI error, data readiness, and supplier concentration diagnostics into operational-risk indicators.",
    ]
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _markdown_table(df: pd.DataFrame) -> str:
    headers = [str(c) for c in df.columns]
    rows = [[str(v) for v in row] for row in df.to_numpy()]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def plot_bar(summary: pd.DataFrame, metric: str, outdir: Path, title: str) -> None:
    if metric not in summary.columns:
        return
    df = summary[["scenario_id", metric]].sort_values(metric, ascending=False)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(df["scenario_id"], df[metric], color="#2f6f9f")
    ax.set_title(title)
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=65)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{metric}.png", dpi=180)
    plt.close(fig)


def plot_heatmap(summary: pd.DataFrame, metrics: list[str], outdir: Path) -> None:
    cols = [c for c in metrics if c in summary.columns]
    if not cols:
        return
    df = summary.set_index("scenario_id")[cols].copy()
    normalized = df.copy()
    for c in cols:
        s = df[c].astype(float)
        span = s.max() - s.min()
        normalized[c] = 0.0 if span == 0 else (s - s.min()) / span
    fig, ax = plt.subplots(figsize=(11, 7))
    im = ax.imshow(normalized.values, aspect="auto", cmap="YlOrRd")
    ax.set_yticks(range(len(normalized.index)))
    ax.set_yticklabels(normalized.index)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c.replace("_mean", "").replace("_proxy", "") for c in cols], rotation=45, ha="right")
    ax.set_title("Actuarial translation stress heatmap (column-normalized)")
    fig.colorbar(im, ax=ax, label="Relative stress within metric")
    fig.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / "actuarial_stress_heatmap.png", dpi=180)
    plt.close(fig)


def build(
    result_dir: Path,
    assumptions_path: Path,
    outdir: Path,
    figures_dir: Path,
    config_path: Path | None,
    report_md: Path,
) -> None:
    assumptions = _read_json(assumptions_path)
    histories = load_histories(result_dir)
    baseline_path = mean_path(histories["baseline"])
    occ_weights = load_occupation_weights(config_path, histories)

    long_rows = []
    summary_rows = []
    occ_delta_rows = []
    for scenario_id, frames in sorted(histories.items()):
        scenario_mean = mean_path(frames)
        path = compute_translation_paths(scenario_mean, baseline_path, assumptions, occ_weights)
        path.insert(0, "scenario_id", scenario_id)
        long_rows.append(path)
        summary_rows.append(summarize_path(scenario_id, path.drop(columns=["scenario_id"])))
        for occ, delta in path.attrs.get("occ_claim_deltas", {}).items():
            occ_delta_rows.append({"scenario_id": scenario_id, "occupation": occ, "mean_claim_cost_delta": delta})

    outdir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(summary_rows).sort_values("scenario_id")
    long = pd.concat(long_rows, ignore_index=True)
    occ_delta = pd.DataFrame(occ_delta_rows)
    summary.to_csv(outdir / "actuarial_translation_summary.csv", index=False)
    long.to_csv(outdir / "actuarial_translation_long.csv", index=False)
    occ_delta.to_csv(outdir / "actuarial_translation_occupation_claim_deltas.csv", index=False)

    write_markdown(summary, report_md, result_dir, assumptions_path, assumptions)

    plot_bar(summary, "Life_lapse_index_mean", figures_dir, "Life lapse-pressure index by scenario")
    plot_bar(summary, "WC_claim_cost_index_mean", figures_dir, "Workers' compensation claim-cost index by scenario")
    plot_bar(summary, "Investment_portfolio_return_proxy_mean", figures_dir, "Portfolio return proxy by scenario")
    plot_bar(summary, "Operational_loss_stress_index_mean", figures_dir, "Operational loss stress index by scenario")
    plot_heatmap(
        summary,
        [
            "Life_lapse_index_mean",
            "WC_claim_cost_index_mean",
            "Pension_sponsor_stress_index_mean",
            "Investment_credit_spread_proxy_mean",
            "Operational_loss_stress_index_mean",
            "Governance_need_index_mean",
        ],
        figures_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Translate ABM scenario histories into actuarial stress indicators.")
    parser.add_argument("result_dir", type=Path, help="Scenario output folder containing *_history_rep*.csv files.")
    parser.add_argument("--assumptions", type=Path, default=Path("configs/actuarial_translation_assumptions.json"))
    parser.add_argument("--outdir", type=Path, default=Path("outputs/actuarial_translation"))
    parser.add_argument("--figures-dir", type=Path, default=Path("docs/figures/actuarial_translation"))
    parser.add_argument("--config", type=Path, default=Path("configs/selected_baseline.json"))
    parser.add_argument("--report-md", type=Path, default=Path("docs/actuarial_translation_results.md"))
    args = parser.parse_args()
    build(args.result_dir, args.assumptions, args.outdir, args.figures_dir, args.config, args.report_md)
    print(f"Actuarial translation outputs written to: {args.outdir}")
    print(f"Figures written to: {args.figures_dir}")
    print(f"Report table written to: {args.report_md}")


if __name__ == "__main__":
    main()

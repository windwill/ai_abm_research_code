"""Scenario-overlay composition, replicated execution, and comparison output utilities."""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .calibration import PARAM_KEYS, params_and_simulation_from_vector
from .config import HAS_YAML, load_config, load_target_groups
from .moments import summarize_moments
from .parameters import LaborParams, PolicyParams, SimulationParams, TechParams
from .reporting import make_plots
from .simulation import Economy
from .utils import rng
from .validation import validate_moments

if HAS_YAML:
    import yaml  # type: ignore


PARAM_GROUPS = {
    "LaborParams": LaborParams,
    "TechParams": TechParams,
    "PolicyParams": PolicyParams,
    "SimulationParams": SimulationParams,
}


def _log(message: str, *, quiet: bool = False) -> None:
    if not quiet:
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def load_scenario_overlay(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load one scenario overlay from JSON or YAML."""
    path_obj = Path(path)
    ext = path_obj.suffix.lower()
    with open(path_obj, "r", encoding="utf-8") as f:
        if ext in {".yml", ".yaml"}:
            if not HAS_YAML:
                raise ImportError("YAML scenario overlays require installing the optional `yaml` extra.")
            data = yaml.safe_load(f)  # type: ignore[name-defined]
        else:
            data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Scenario overlay must be a mapping: {path_obj}")
    return data


def _load_mapping_file(path: str | os.PathLike[str]) -> dict[str, Any]:
    path_obj = Path(path)
    ext = path_obj.suffix.lower()
    with open(path_obj, "r", encoding="utf-8") as f:
        if ext in {".yml", ".yaml"}:
            if not HAS_YAML:
                raise ImportError("YAML files require installing the optional `yaml` extra.")
            data = yaml.safe_load(f)  # type: ignore[name-defined]
        else:
            data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a mapping in {path_obj}")
    return data


def _resolve_matrix_path(path: str, base_dir: Path) -> str:
    path_obj = Path(path)
    if path_obj.is_absolute():
        return str(path_obj)
    direct = Path(path)
    if direct.exists():
        return str(direct)
    return str(base_dir / path_obj)


def load_scenario_matrix(path: str | os.PathLike[str]) -> list[list[str]]:
    """Load and expand a batch scenario matrix.

    Supported JSON/YAML forms:
    - {"dimensions": {"automation": ["A1.json", "A2.json"], "policy": ["P1.json"]}}
      expands to the Cartesian product in insertion order.
    - {"scenarios": [["A2.json", "L2.json"], ["A3.json", "P3.json"]]}
      uses explicit overlay stacks.
    Relative overlay paths are resolved first from the current working directory,
    then relative to the matrix file.
    """
    path_obj = Path(path)
    data = _load_mapping_file(path_obj)
    base_dir = path_obj.parent
    if "scenarios" in data:
        stacks = data["scenarios"]
        if not isinstance(stacks, list):
            raise ValueError("`scenarios` must be a list of overlay stacks")
        out = []
        for stack in stacks:
            if isinstance(stack, str):
                stack = [stack]
            if not isinstance(stack, list):
                raise ValueError("Each scenario stack must be a list or string")
            out.append([_resolve_matrix_path(str(item), base_dir) for item in stack])
        return out
    dimensions = data.get("dimensions")
    if not isinstance(dimensions, dict) or not dimensions:
        raise ValueError("Scenario matrix must contain `dimensions` or `scenarios`")
    axes = []
    for name, values in dimensions.items():
        if isinstance(values, str):
            values = [values]
        if not isinstance(values, list) or not values:
            raise ValueError(f"Scenario matrix dimension `{name}` must be a non-empty list")
        axes.append([_resolve_matrix_path(str(item), base_dir) for item in values])
    return [list(combo) for combo in itertools.product(*axes)]


def _clone_params(
    lp: LaborParams,
    tp: TechParams,
    pp: PolicyParams,
    sp: SimulationParams,
) -> tuple[LaborParams, TechParams, PolicyParams, SimulationParams]:
    return (
        LaborParams(**asdict(lp)),
        TechParams(**asdict(tp)),
        PolicyParams(**asdict(pp)),
        SimulationParams(**asdict(sp)),
    )


def _set_param(obj: object, group_name: str, field_name: str, value: Any) -> tuple[Any, Any]:
    if not hasattr(obj, field_name):
        raise ValueError(f"Unknown scenario field `{group_name}.{field_name}`")
    before = getattr(obj, field_name)
    setattr(obj, field_name, value)
    return before, value


def apply_overlay(
    lp: LaborParams,
    tp: TechParams,
    pp: PolicyParams,
    sp: SimulationParams,
    overlay: dict[str, Any],
    *,
    source: str = "",
) -> list[dict[str, Any]]:
    """Apply one overlay to parameter dataclasses and return an audit trail."""
    groups = {
        "LaborParams": lp,
        "TechParams": tp,
        "PolicyParams": pp,
        "SimulationParams": sp,
    }
    changes: list[dict[str, Any]] = []
    for group_name, group_obj in groups.items():
        updates = overlay.get(group_name, {})
        if updates is None:
            continue
        if not isinstance(updates, dict):
            raise ValueError(f"`{group_name}` in scenario overlay must be a mapping")
        for field_name, value in updates.items():
            before, after = _set_param(group_obj, group_name, str(field_name), value)
            changes.append({
                "source": source,
                "group": group_name,
                "parameter": str(field_name),
                "before": before,
                "after": after,
            })
    return changes


def compose_scenario(
    lp: LaborParams,
    tp: TechParams,
    pp: PolicyParams,
    sp: SimulationParams,
    overlay_paths: list[str],
) -> tuple[LaborParams, TechParams, PolicyParams, SimulationParams, dict[str, Any]]:
    """Apply stacked scenario overlays to cloned baseline parameters."""
    scenario_lp, scenario_tp, scenario_pp, scenario_sp = _clone_params(lp, tp, pp, sp)
    overlay_meta = []
    changes: list[dict[str, Any]] = []
    for path in overlay_paths:
        overlay = load_scenario_overlay(path)
        meta = copy.deepcopy(overlay.get("Scenario", {}))
        if not isinstance(meta, dict):
            raise ValueError(f"`Scenario` metadata must be a mapping in {path}")
        if "id" not in meta:
            meta["id"] = Path(path).stem
        meta["path"] = str(path)
        overlay_meta.append(meta)
        changes.extend(apply_overlay(scenario_lp, scenario_tp, scenario_pp, scenario_sp, overlay, source=str(path)))
    scenario_id = "+".join(str(m["id"]) for m in overlay_meta) if overlay_meta else "baseline"
    manifest = {
        "scenario_id": scenario_id,
        "overlays": overlay_meta,
        "changes": changes,
        "parameters": {
            "LaborParams": asdict(scenario_lp),
            "TechParams": asdict(scenario_tp),
            "PolicyParams": asdict(scenario_pp),
            "SimulationParams": asdict(scenario_sp),
        },
    }
    return scenario_lp, scenario_tp, scenario_pp, scenario_sp, manifest


def run_scenario_replications(
    lp: LaborParams,
    tp: TechParams,
    pp: PolicyParams,
    sp: SimulationParams,
    *,
    scenario_id: str,
    replications: int = 1,
    outdir: str = "outputs/scenarios",
    targets: dict[str, tuple[float, float]] | None = None,
    save_history: bool = True,
    plot: bool = False,
    progress=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run repeated simulations for a scenario with deterministic seed advancement."""
    os.makedirs(outdir, exist_ok=True)
    replications = max(1, int(replications))
    seed_gen = rng(sp.seed)
    moment_rows = []
    validation_rows = []
    for rep in range(replications):
        sp_rep = SimulationParams(**asdict(sp))
        sp_rep.seed = int(seed_gen.integers(0, 1_000_000))
        if progress:
            progress(f"{scenario_id}: starting replication {rep + 1}/{replications} with seed={sp_rep.seed}")
        econ = Economy(
            LaborParams(**asdict(lp)),
            TechParams(**asdict(tp)),
            PolicyParams(**asdict(pp)),
            sp_rep,
            outdir=outdir,
        )
        history = econ.simulate()
        moments = summarize_moments(history)
        if progress:
            progress(
                f"{scenario_id}: completed replication {rep + 1}/{replications}; "
                f"U_mean={moments.get('U_mean', float('nan')):.4f}, "
                f"Y_growth={moments.get('Y_growth', float('nan')):.4f}, "
                f"Adopt_final={moments.get('Adopt_final', float('nan')):.4f}"
            )
        moment_row = {
            "scenario_id": scenario_id,
            "replication": rep + 1,
            "seed": sp_rep.seed,
            **moments,
        }
        moment_rows.append(moment_row)
        if save_history:
            history_path = os.path.join(outdir, f"{scenario_id}_history_rep{rep + 1}.csv")
            history.to_csv(history_path, index=False)
        if plot and rep == 0:
            plot_dir = os.path.join(outdir, f"{scenario_id}_plots")
            make_plots(history, plot_dir)
        if targets:
            val, _ = validate_moments(moments, targets, label=f"{scenario_id}_rep{rep + 1}")
            val.insert(0, "seed", sp_rep.seed)
            val.insert(0, "replication", rep + 1)
            val.insert(0, "scenario_id", scenario_id)
            validation_rows.append(val)

    moments_df = pd.DataFrame(moment_rows)
    summary_rows = []
    for col in moments_df.columns:
        if col in {"scenario_id", "replication", "seed"}:
            continue
        values = pd.to_numeric(moments_df[col], errors="coerce").dropna()
        if values.empty:
            continue
        summary_rows.append({
            "scenario_id": scenario_id,
            "moment": col,
            "replications": int(len(values)),
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "p05": float(np.percentile(values, 5)),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
        })
    summary_df = pd.DataFrame(summary_rows)
    moments_df.to_csv(os.path.join(outdir, f"{scenario_id}_moments.csv"), index=False)
    summary_df.to_csv(os.path.join(outdir, f"{scenario_id}_moment_summary.csv"), index=False)
    if progress:
        progress(f"{scenario_id}: wrote moments and summary outputs")
    if validation_rows:
        pd.concat(validation_rows, ignore_index=True).to_csv(
            os.path.join(outdir, f"{scenario_id}_validation.csv"),
            index=False,
        )
    return moments_df, summary_df


def _summarize_moment_rows(df: pd.DataFrame, *, scenario_id: str, q_low: float = 0.025, q_high: float = 0.975) -> pd.DataFrame:
    summary_rows = []
    for col in df.columns:
        if col in {"scenario_id", "replication", "posterior_draw", "sample_index", "seed"}:
            continue
        values = pd.to_numeric(df[col], errors="coerce").dropna()
        if values.empty:
            continue
        summary_rows.append({
            "scenario_id": scenario_id,
            "moment": col,
            "draws": int(len(values)),
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "p2.5": float(values.quantile(q_low)),
            "p50": float(values.quantile(0.50)),
            "p97.5": float(values.quantile(q_high)),
        })
    return pd.DataFrame(summary_rows)


def load_posterior_samples(path: str | os.PathLike[str]) -> np.ndarray:
    """Load posterior samples from CSV in PARAM_KEYS order."""
    df = pd.read_csv(path)
    if set(PARAM_KEYS).issubset(df.columns):
        samples = df[PARAM_KEYS].to_numpy(dtype=float)
    elif df.shape[1] >= len(PARAM_KEYS):
        samples = df.iloc[:, :len(PARAM_KEYS)].to_numpy(dtype=float)
    else:
        raise ValueError(f"Posterior sample file must contain {len(PARAM_KEYS)} parameter columns")
    if samples.ndim != 2 or samples.shape[1] != len(PARAM_KEYS) or samples.shape[0] == 0:
        raise ValueError("Posterior sample file contains no usable draws")
    return samples


def scenario_posterior_predictive(
    base_lp: LaborParams,
    base_tp: TechParams,
    base_pp: PolicyParams,
    base_sp: SimulationParams,
    overlay_paths: list[str],
    samples: np.ndarray,
    *,
    scenario_id: str | None = None,
    draws: int = 100,
    outdir: str = "outputs/scenarios",
    progress=None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Run posterior predictive simulations under one scenario overlay stack.

    Posterior draws first set calibrated parameters. Scenario overlays are then
    applied so counterfactual scenario settings take precedence over posterior
    draw values for any overlapping fields.
    """
    os.makedirs(outdir, exist_ok=True)
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2 or samples.shape[1] != len(PARAM_KEYS):
        raise ValueError("samples must have shape (draws, len(PARAM_KEYS))")
    _, _, _, _, manifest = compose_scenario(base_lp, base_tp, base_pp, base_sp, overlay_paths)
    scenario_name = scenario_id or str(manifest["scenario_id"])
    n_draws = min(max(1, int(draws)), len(samples))
    idx = np.linspace(0, len(samples) - 1, num=n_draws, dtype=int)
    seed_gen = rng(base_sp.seed)
    rows = []
    overlays = [(path, load_scenario_overlay(path)) for path in overlay_paths]
    for out_idx, sample_idx in enumerate(idx, start=1):
        if progress:
            progress(f"{scenario_name}: starting posterior draw {out_idx}/{n_draws} from sample_index={int(sample_idx)}")
        lp_draw, tp_draw, pp_draw, sp_draw = params_and_simulation_from_vector(base_lp, base_tp, base_pp, base_sp, samples[int(sample_idx)])
        for path, overlay in overlays:
            apply_overlay(lp_draw, tp_draw, pp_draw, sp_draw, overlay, source=str(path))
        sp_draw.seed = int(seed_gen.integers(0, 1_000_000))
        history = Economy(lp_draw, tp_draw, pp_draw, sp_draw, outdir=outdir).simulate()
        rows.append({
            "scenario_id": scenario_name,
            "posterior_draw": out_idx,
            "sample_index": int(sample_idx),
            "seed": sp_draw.seed,
            **summarize_moments(history),
        })
        if progress:
            progress(f"{scenario_name}: completed posterior draw {out_idx}/{n_draws}")
    raw = pd.DataFrame(rows)
    summary = _summarize_moment_rows(raw, scenario_id=scenario_name)
    raw.to_csv(os.path.join(outdir, f"{scenario_name}_scenario_ppc_raw.csv"), index=False)
    summary.to_csv(os.path.join(outdir, f"{scenario_name}_scenario_ppc_summary.csv"), index=False)
    manifest = copy.deepcopy(manifest)
    manifest["scenario_id"] = scenario_name
    manifest["posterior_predictive"] = {
        "draws_requested": int(draws),
        "draws_used": int(n_draws),
        "sample_indices": [int(i) for i in idx],
    }
    write_manifest(outdir, manifest)
    if progress:
        progress(f"{scenario_name}: wrote posterior predictive outputs")
    return raw, summary, manifest


def write_manifest(outdir: str, manifest: dict[str, Any]) -> str:
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{manifest['scenario_id']}_manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return path


def write_comparison_outputs(outdir: str, summaries: list[pd.DataFrame], moments: list[pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Write long and wide cross-scenario comparison tables."""
    os.makedirs(outdir, exist_ok=True)
    summary_all = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    moments_all = pd.concat(moments, ignore_index=True) if moments else pd.DataFrame()
    if not summary_all.empty:
        summary_all.to_csv(os.path.join(outdir, "scenario_comparison_moments_long.csv"), index=False)
        wide_mean = summary_all.pivot(index="scenario_id", columns="moment", values="mean").reset_index()
        wide_mean.to_csv(os.path.join(outdir, "scenario_comparison_moments_mean.csv"), index=False)
    else:
        wide_mean = pd.DataFrame()
    if not moments_all.empty:
        moments_all.to_csv(os.path.join(outdir, "scenario_replication_moments_all.csv"), index=False)
    return moments_all, summary_all, wide_mean


def run_scenario_batch(
    lp: LaborParams,
    tp: TechParams,
    pp: PolicyParams,
    sp: SimulationParams,
    scenario_stacks: list[list[str]],
    *,
    outdir: str = "outputs/scenarios",
    replications: int = 1,
    targets: dict[str, tuple[float, float]] | None = None,
    save_history: bool = True,
    plot: bool = False,
    progress=None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run a batch of scenario stacks and write combined comparison tables."""
    all_moments = []
    all_summaries = []
    seen: dict[str, int] = {}
    batch_manifest = []
    total = len(scenario_stacks)
    for idx, stack in enumerate(scenario_stacks, start=1):
        scenario_lp, scenario_tp, scenario_pp, scenario_sp, manifest = compose_scenario(lp, tp, pp, sp, stack)
        scenario_id = str(manifest["scenario_id"])
        seen[scenario_id] = seen.get(scenario_id, 0) + 1
        if seen[scenario_id] > 1:
            scenario_id = f"{scenario_id}_{seen[scenario_id]}"
            manifest["scenario_id"] = scenario_id
        if progress:
            overlay_names = ", ".join(Path(path).stem for path in stack) if stack else "baseline"
            progress(f"Starting scenario {idx}/{total}: {scenario_id} ({overlay_names})")
        write_manifest(outdir, manifest)
        moments_df, summary_df = run_scenario_replications(
            scenario_lp,
            scenario_tp,
            scenario_pp,
            scenario_sp,
            scenario_id=scenario_id,
            replications=replications,
            outdir=outdir,
            targets=targets,
            save_history=save_history,
            plot=plot,
            progress=progress,
        )
        all_moments.append(moments_df)
        all_summaries.append(summary_df)
        batch_manifest.append({
            "scenario_id": scenario_id,
            "overlays": manifest["overlays"],
            "changes": manifest["changes"],
        })
        if progress:
            progress(f"Completed scenario {idx}/{total}: {scenario_id}")
    with open(os.path.join(outdir, "scenario_batch_manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"scenarios": batch_manifest}, f, indent=2)
    if progress:
        progress("Writing scenario comparison outputs")
    return write_comparison_outputs(outdir, all_summaries, all_moments)


def run_scenario_posterior_batch(
    lp: LaborParams,
    tp: TechParams,
    pp: PolicyParams,
    sp: SimulationParams,
    scenario_stacks: list[list[str]],
    samples: np.ndarray,
    *,
    outdir: str = "outputs/scenarios",
    draws: int = 100,
    progress=None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run posterior predictive simulations for a batch of scenario stacks."""
    raws = []
    summaries = []
    seen: dict[str, int] = {}
    batch_manifest = []
    total = len(scenario_stacks)
    for idx, stack in enumerate(scenario_stacks, start=1):
        _, _, _, _, manifest = compose_scenario(lp, tp, pp, sp, stack)
        scenario_id = str(manifest["scenario_id"])
        seen[scenario_id] = seen.get(scenario_id, 0) + 1
        if seen[scenario_id] > 1:
            scenario_id = f"{scenario_id}_{seen[scenario_id]}"
        if progress:
            overlay_names = ", ".join(Path(path).stem for path in stack) if stack else "baseline"
            progress(f"Starting scenario posterior predictive {idx}/{total}: {scenario_id} ({overlay_names})")
        raw, summary, ppc_manifest = scenario_posterior_predictive(
            lp,
            tp,
            pp,
            sp,
            stack,
            samples,
            scenario_id=scenario_id,
            draws=draws,
            outdir=outdir,
            progress=progress,
        )
        raws.append(raw)
        summaries.append(summary)
        batch_manifest.append({
            "scenario_id": scenario_id,
            "overlays": ppc_manifest["overlays"],
            "posterior_predictive": ppc_manifest["posterior_predictive"],
        })
        if progress:
            progress(f"Completed scenario posterior predictive {idx}/{total}: {scenario_id}")
    raw_all = pd.concat(raws, ignore_index=True) if raws else pd.DataFrame()
    summary_all = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    if not raw_all.empty:
        raw_all.to_csv(os.path.join(outdir, "scenario_ppc_raw_all.csv"), index=False)
    if not summary_all.empty:
        summary_all.to_csv(os.path.join(outdir, "scenario_ppc_summary_all.csv"), index=False)
        wide = summary_all.pivot(index="scenario_id", columns="moment", values="mean").reset_index()
        wide.to_csv(os.path.join(outdir, "scenario_ppc_mean.csv"), index=False)
    else:
        wide = pd.DataFrame()
    with open(os.path.join(outdir, "scenario_ppc_batch_manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"scenarios": batch_manifest}, f, indent=2)
    if progress:
        progress("Writing scenario posterior predictive comparison outputs")
    return raw_all, summary_all, wide


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ABM scenario overlays with Monte Carlo replications")
    parser.add_argument("--config", default=None, help="Baseline JSON/YAML config")
    parser.add_argument("--scenario", action="append", default=[], help="Scenario overlay JSON/YAML file; repeat to stack overlays")
    parser.add_argument("--matrix", default=None, help="Optional JSON/YAML scenario matrix for batch runs")
    parser.add_argument("--targets", default=None, help="Optional target file for scenario validation output")
    parser.add_argument("--outdir", default="outputs/scenarios")
    parser.add_argument("--replications", type=int, default=1)
    parser.add_argument("--posterior-samples", default=None, help="CSV posterior samples in PARAM_KEYS order or with PARAM_KEYS columns")
    parser.add_argument("--ppc-draws", type=int, default=100, help="Posterior predictive draws per scenario")
    parser.add_argument("--T", type=int, default=None, help="Override simulation horizon")
    parser.add_argument("--seed", type=int, default=None, help="Override baseline scenario seed")
    parser.add_argument("--no-history", action="store_true", help="Do not write per-replication history files")
    parser.add_argument("--plot", action="store_true", help="Plot the first replication")
    parser.add_argument("--quiet", action="store_true", help="Suppress timestamped progress messages")
    args = parser.parse_args()

    progress = None if args.quiet else lambda message: _log(message, quiet=False)

    _log("Loading scenario configuration", quiet=args.quiet)
    lp, tp, pp, sp = load_config(args.config)
    if args.T is not None:
        sp.T = int(args.T)
    if args.seed is not None:
        sp.seed = int(args.seed)
    targets, _ = load_target_groups(args.targets) if args.targets else ({}, {})
    posterior_samples = load_posterior_samples(args.posterior_samples) if args.posterior_samples else None
    if args.matrix:
        stacks = load_scenario_matrix(args.matrix)
        _log(
            f"Loaded scenario matrix with {len(stacks)} scenario(s); "
            f"T={sp.T}, replications={args.replications}, outdir={args.outdir}",
            quiet=args.quiet,
        )
        if posterior_samples is not None:
            _, summary_all, _ = run_scenario_posterior_batch(
                lp,
                tp,
                pp,
                sp,
                stacks,
                posterior_samples,
                outdir=args.outdir,
                draws=args.ppc_draws,
                progress=progress,
            )
            print(f"Scenario posterior predictive batch: {len(stacks)} scenario(s)")
            print(f"Outputs written to: {args.outdir}")
            if not summary_all.empty:
                print("Comparison table: scenario_ppc_mean.csv")
            return
        _, summary_all, _ = run_scenario_batch(
            lp,
            tp,
            pp,
            sp,
            stacks,
            outdir=args.outdir,
            replications=args.replications,
            targets=targets,
            save_history=not args.no_history,
            plot=args.plot,
            progress=progress,
        )
        print(f"Scenario batch: {len(stacks)} scenario(s)")
        print(f"Outputs written to: {args.outdir}")
        if not summary_all.empty:
            print("Comparison table: scenario_comparison_moments_mean.csv")
        return
    if posterior_samples is not None:
        _log(
            f"Starting scenario posterior predictive; T={sp.T}, draws={args.ppc_draws}, outdir={args.outdir}",
            quiet=args.quiet,
        )
        _, summary, manifest = scenario_posterior_predictive(
            lp,
            tp,
            pp,
            sp,
            args.scenario,
            posterior_samples,
            draws=args.ppc_draws,
            outdir=args.outdir,
            progress=progress,
        )
        print(f"Scenario posterior predictive: {manifest['scenario_id']}")
        print(f"Outputs written to: {args.outdir}")
        if not summary.empty:
            for _, row in summary.iterrows():
                print(f"{row['moment']}: mean={row['mean']:.4f}, p2.5={row['p2.5']:.4f}, p97.5={row['p97.5']:.4f}")
        return
    scenario_lp, scenario_tp, scenario_pp, scenario_sp, manifest = compose_scenario(lp, tp, pp, sp, args.scenario)
    _log(
        f"Starting scenario {manifest['scenario_id']}; T={scenario_sp.T}, replications={args.replications}, outdir={args.outdir}",
        quiet=args.quiet,
    )
    write_manifest(args.outdir, manifest)
    _, summary = run_scenario_replications(
        scenario_lp,
        scenario_tp,
        scenario_pp,
        scenario_sp,
        scenario_id=manifest["scenario_id"],
        replications=args.replications,
        outdir=args.outdir,
        targets=targets,
        save_history=not args.no_history,
        plot=args.plot,
        progress=progress,
    )
    print(f"Scenario: {manifest['scenario_id']}")
    print(f"Outputs written to: {args.outdir}")
    if not summary.empty:
        for _, row in summary.iterrows():
            print(f"{row['moment']}: mean={row['mean']:.4f}, p05={row['p05']:.4f}, p95={row['p95']:.4f}")


if __name__ == "__main__":
    main()

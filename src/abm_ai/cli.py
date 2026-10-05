"""Command-line orchestration for simulation, calibration, diagnostics, validation, and reporting."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, fields
from datetime import datetime

import pandas as pd

from .calibration import (
    PARAM_KEYS,
    PARAM_BOUNDS,
    apply_param_bounds_overrides,
    average_msm_moments,
    bootstrap_moment_intervals,
    ofat_sensitivity,
    params_and_simulation_from_vector,
    run_cmaes,
    run_ga,
    run_simulated_annealing,
    sample_vector,
    save_sobol_tables,
    simulate_msm_replications,
    sobol_sensitivity,
    summarize_moment_replications,
)
from .config import load_calibration_bounds, load_config, load_moment_covariance, load_target_groups, load_targets
from .diagnostics import (
    build_diagnostics_report,
    plot_acf_bars,
    plot_ppc,
    plot_traces,
    posterior_predictive,
    run_pymc_importance_sampler,
    rhat_gelman_rubin,
    run_mcmc_chains,
    run_smc_sampler,
    summarize_posterior,
)
from .moments import dynamic_diagnostic_moments, summarize_moments
from .parameters import LaborParams, PolicyParams, SimulationParams, TechParams
from .reporting import build_combined_report, compare_figs, compare_moments, export_figset_pdf, generate_report, make_plots
from .simulation import Economy
from .utils import rng
from .validation import distributional_tests, load_empirical_distributions, validate_moments

def _known_dataclass_items(cls, values: dict) -> dict:
    """Drop obsolete saved-result keys before constructing dataclasses."""
    allowed = {field.name for field in fields(cls)}
    return {key: value for key, value in values.items() if key in allowed}

def _vector_from_params_json(path: str):
    """Load a calibrated parameter JSON file into the active PARAM_BOUNDS order."""
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    vec = []
    for key in PARAM_KEYS:
        lo, hi, group = PARAM_BOUNDS[key]
        name = key.split(".")[1]
        value = payload.get(group, {}).get(name, (lo + hi) / 2.0)
        vec.append(float(max(lo, min(hi, value))))
    return vec

def _params_from_params_json(path: str, base_lp, base_tp, base_pp, base_sp):
    """Load a saved best-parameter JSON file as dataclasses without re-optimizing."""
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    lp_data = asdict(base_lp)
    lp_data.update(_known_dataclass_items(LaborParams, payload.get("LaborParams", {})))
    tp_data = asdict(base_tp)
    tp_data.update(_known_dataclass_items(TechParams, payload.get("TechParams", {})))
    pp_data = asdict(base_pp)
    pp_data.update(_known_dataclass_items(PolicyParams, payload.get("PolicyParams", {})))
    sp_data = asdict(base_sp)
    sp_data.update(_known_dataclass_items(SimulationParams, payload.get("SimulationParams", {})))
    return (
        LaborParams(**lp_data),
        TechParams(**tp_data),
        PolicyParams(**pp_data),
        SimulationParams(**sp_data),
    )

def _simulate_saved_param_replications(lp, tp, pp, sp):
    """Evaluate already-loaded parameter dataclasses without vector re-clipping."""
    eval_R = max(1, int(getattr(sp, "monte_carlo_R", 1)))
    seed_gen = rng(sp.seed)
    rows = []
    for _ in range(eval_R):
        sp_local = SimulationParams(**asdict(sp))
        sp_local.seed = int(seed_gen.integers(0, 1_000_000))
        rows.append(summarize_moments(Economy(lp, tp, pp, sp_local).simulate()))
    return rows

def main():
    parser = argparse.ArgumentParser(description="ABM Automation Simulation + Optimizers + Sensitivity")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML or JSON config file")
    parser.add_argument("--targets", type=str, default=None, help="Path to targets file (CSV/JSON/YAML)")
    parser.add_argument("--covariance", type=str, default=None, help="Optional moment covariance matrix for Mahalanobis MSM loss")
    parser.add_argument("--validation-targets", type=str, default=None, help="Optional hold-out validation targets file")
    parser.add_argument("--empirical-distributions", type=str, default=None, help="Optional long CSV with variable,value columns for KS/CvM tests")
    parser.add_argument("--plot", action="store_true", help="Produce charts for key series")
    parser.add_argument("--figset", action="store_true", help="Export a compact PDF figure set with captions")
    parser.add_argument("--report", action="store_true", help="Generate a combined report PDF")
    parser.add_argument("--outdir", type=str, default="/mnt/data", help="Output directory")
    parser.add_argument("--seed", type=int, default=None, help="Override seed")
    parser.add_argument("--T", type=int, default=None, help="Override simulation length")
    parser.add_argument("--monte-carlo-R", type=int, default=None, help="Replications per MSM fitness evaluation")
    parser.add_argument("--moment-ci-R", type=int, default=0, help="Replications for baseline moment CI output")
    parser.add_argument("--bootstrap-ci", action="store_true", help="Add percentile bootstrap intervals to --moment-ci-R output")
    parser.add_argument("--bootstrap-draws", type=int, default=1000, help="Bootstrap resamples for moment interval output")
    parser.add_argument("--opt-ga", action="store_true", help="Run GA optimization")
    parser.add_argument("--ga-pop", type=int, default=24)
    parser.add_argument("--ga-gens", type=int, default=20)
    parser.add_argument("--ga-init-params", type=str, default=None, help="Optional *_best_params.json file used to seed the GA population locally")
    parser.add_argument("--ga-init-scale", type=float, default=0.08, help="Local GA initialization noise as a fraction of each calibrated parameter range")
    parser.add_argument("--post-calibration-R", type=int, default=0, help="Fresh replications for post-optimization validation of calibrated candidates")
    parser.add_argument("--eval-params", type=str, default=None, help="Evaluate a saved *_best_params.json file without running a new optimizer")
    parser.add_argument("--opt-cmaes", action="store_true", help="Run CMA-ES optimization")
    parser.add_argument("--cma-iters", type=int, default=30)
    parser.add_argument("--opt-sa", action="store_true", help="Run simulated annealing optimization")
    parser.add_argument("--sa-iters", type=int, default=100)
    parser.add_argument("--sa-start-temp", type=float, default=1.0)
    parser.add_argument("--sa-end-temp", type=float, default=0.01)
    parser.add_argument("--sa-step-scale", type=float, default=0.10)
    parser.add_argument("--mcmc", action="store_true", help="Run Bayesian MCMC parameter refinement")
    parser.add_argument("--mcmc-draws", type=int, default=1500)
    parser.add_argument("--mcmc-burn", type=int, default=300)
    parser.add_argument("--mcmc-thin", type=int, default=2)
    parser.add_argument("--mcmc-step", type=float, default=0.08)
    parser.add_argument("--mcmc-chains", type=int, default=4)
    parser.add_argument("--mcmc-init-params", type=str, default=None, help="Optional *_best_params.json file used to initialize MCMC chains near a calibrated vector")
    parser.add_argument("--mcmc-log-every", type=int, default=None, help="Log MCMC progress every N iterations per chain; default is about 20 messages per chain")
    parser.add_argument("--ppc-draws", type=int, default=100)
    parser.add_argument("--smc", action="store_true", help="Run dependency-free SMC sampler")
    parser.add_argument("--smc-particles", type=int, default=64)
    parser.add_argument("--smc-levels", type=int, default=6)
    parser.add_argument("--smc-rejuvenation", type=int, default=1)
    parser.add_argument("--pymc", action="store_true", help="Run optional PyMC-assisted prior importance sampler")
    parser.add_argument("--pymc-draws", type=int, default=128)
    parser.add_argument("--diag-report", action="store_true", help="Build diagnostics_mcmc.pdf")
    parser.add_argument("--sensitivity", action="store_true", help="Run Sobol-style global sensitivity")
    parser.add_argument("--ofat-sensitivity", action="store_true", help="Run local one-factor-at-a-time sensitivity")
    parser.add_argument("--ofat-step", type=float, default=0.10, help="Fraction of each calibrated parameter range used for OFAT perturbations")
    parser.add_argument("--sens-output", type=str, default="Y_growth", help="Moment for sensitivity analysis")
    parser.add_argument("--sens-N", type=int, default=128, help="Sobol base sample size")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress logging")

    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    def log(message: str) -> None:
        if not args.quiet:
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{now}] {message}", flush=True)

    progress = None if args.quiet else log

    log("Loading configuration and targets")
    lp,tp,pp,sp = load_config(args.config)
    bound_overrides = load_calibration_bounds(args.config)
    if bound_overrides:
        apply_param_bounds_overrides(bound_overrides)
        log(f"Applied config-local calibration bounds for {len(bound_overrides)} parameter(s)")
    if args.seed is not None:
        sp.seed = args.seed
    if args.T is not None:
        sp.T = args.T
    if args.monte_carlo_R is not None:
        sp.monte_carlo_R = max(1, int(args.monte_carlo_R))
    targets, validation_targets = load_target_groups(args.targets)
    covariance = load_moment_covariance(args.covariance)
    empirical_distributions = load_empirical_distributions(args.empirical_distributions) if args.empirical_distributions else None
    if args.validation_targets:
        validation_targets = load_targets(args.validation_targets)

    log(f"Starting baseline simulation: T={sp.T}, workers={sp.N_workers}, firms={sp.N_firms}, seed={sp.seed}")
    econ = Economy(lp,tp,pp,sp, outdir=args.outdir)
    df = econ.simulate()
    log("Baseline simulation complete; summarizing moments")
    moments = summarize_moments(df)
    df.to_csv(os.path.join(args.outdir, "abm_simulation_history.csv"), index=False)
    pd.DataFrame([moments]).to_csv(os.path.join(args.outdir, "abm_moments_baseline.csv"), index=False)
    pd.DataFrame([dynamic_diagnostic_moments(df)]).to_csv(os.path.join(args.outdir, "dynamic_diagnostics_baseline.csv"), index=False)
    validation_tables = []
    validation_summaries = []
    val_base, val_base_summary = validate_moments(moments, targets, label="baseline")
    val_base.to_csv(os.path.join(args.outdir, "validation_baseline.csv"), index=False)
    val_base_summary.to_csv(os.path.join(args.outdir, "validation_summary_baseline.csv"), index=False)
    validation_tables.append(val_base)
    validation_summaries.append(val_base_summary)
    if validation_targets:
        val_holdout, val_holdout_summary = validate_moments(moments, validation_targets, label="baseline_holdout")
        val_holdout.to_csv(os.path.join(args.outdir, "validation_holdout_baseline.csv"), index=False)
        val_holdout_summary.to_csv(os.path.join(args.outdir, "validation_summary_holdout_baseline.csv"), index=False)
        validation_tables.append(val_holdout)
        validation_summaries.append(val_holdout_summary)

    if args.eval_params:
        eval_R = int(args.post_calibration_R) if args.post_calibration_R and args.post_calibration_R > 0 else max(1, int(sp.monte_carlo_R))
        log(f"Evaluating saved parameter file with fresh replications: {args.eval_params}; R={eval_R}")
        eval_lp, eval_tp, eval_pp, eval_sp = _params_from_params_json(args.eval_params, lp, tp, pp, sp)
        eval_sp.seed = sp.seed + 40_000
        eval_sp.T = sp.T
        eval_sp.monte_carlo_R = eval_R
        eval_rows = _simulate_saved_param_replications(eval_lp, eval_tp, eval_pp, eval_sp)
        eval_mom = average_msm_moments(eval_rows)
        pd.DataFrame(eval_rows).to_csv(os.path.join(args.outdir, "abm_moment_replications_eval_params.csv"), index=False)
        pd.DataFrame([eval_mom]).to_csv(os.path.join(args.outdir, "abm_moments_eval_params.csv"), index=False)
        val_eval, val_eval_summary = validate_moments(eval_mom, targets, label="eval_params")
        val_eval.to_csv(os.path.join(args.outdir, "validation_eval_params.csv"), index=False)
        val_eval_summary.to_csv(os.path.join(args.outdir, "validation_summary_eval_params.csv"), index=False)
        validation_tables.append(val_eval)
        validation_summaries.append(val_eval_summary)
        if validation_targets:
            val_eval_holdout, val_eval_holdout_summary = validate_moments(eval_mom, validation_targets, label="eval_params_holdout")
            val_eval_holdout.to_csv(os.path.join(args.outdir, "validation_holdout_eval_params.csv"), index=False)
            val_eval_holdout_summary.to_csv(os.path.join(args.outdir, "validation_summary_holdout_eval_params.csv"), index=False)
            validation_tables.append(val_eval_holdout)
            validation_summaries.append(val_eval_holdout_summary)

    if args.moment_ci_R and args.moment_ci_R > 0:
        log(f"Computing replicated moment intervals: R={args.moment_ci_R}")
        sp_ci = SimulationParams(**asdict(sp))
        sp_ci.monte_carlo_R = int(args.moment_ci_R)
        baseline_vec = sample_vector(rng(sp.seed))
        # Use the midpoint of the calibrated parameter bounds as the neutral vector
        # for CI reporting unless a calibrated vector is supplied in a later stage.
        for i, key in enumerate(PARAM_KEYS):
            lo, hi, _ = PARAM_BOUNDS[key]
            baseline_vec[i] = (lo + hi) / 2.0
        rows = simulate_msm_replications(baseline_vec, lp, tp, pp, sp_ci)
        pd.DataFrame(rows).to_csv(os.path.join(args.outdir, "abm_moment_replications.csv"), index=False)
        summarize_moment_replications(rows).to_csv(os.path.join(args.outdir, "abm_moment_ci.csv"), index=False)
        if args.bootstrap_ci:
            log(f"Computing bootstrap moment intervals: draws={args.bootstrap_draws}")
            bootstrap_moment_intervals(rows, draws=args.bootstrap_draws, seed=sp.seed + 404).to_csv(
                os.path.join(args.outdir, "abm_moment_bootstrap_ci.csv"),
                index=False,
            )

    if args.plot:
        log("Writing baseline plots")
        make_plots(df, args.outdir)
    if args.figset:
        log("Writing baseline figure set PDF")
        export_figset_pdf(df, os.path.join(args.outdir, "abm_figures.pdf"))
    if empirical_distributions:
        log("Running baseline distributional tests")
        distributional_tests(df, empirical_distributions, label="baseline").to_csv(
            os.path.join(args.outdir, "distribution_tests_baseline.csv"),
            index=False,
        )

    df_ga = None
    mom_ga = None
    if args.opt_ga:
        log(f"Starting GA calibration: pop={args.ga_pop}, gens={args.ga_gens}")
        ga_init_vec = None
        if args.ga_init_params:
            log(f"Loading GA initial parameter vector from {args.ga_init_params}")
            ga_init_vec = _vector_from_params_json(args.ga_init_params)
        best_vec, best_score, ga_history = run_ga(
            lp,tp,pp,sp, targets,
            seed=2025,
            pop=args.ga_pop,
            gens=args.ga_gens,
            return_history=True,
            covariance=covariance,
            progress=progress,
            init_vec=ga_init_vec,
            init_scale=args.ga_init_scale,
        )
        log(f"GA calibration complete: best_loss={best_score:.6g}")
        ga_history.to_csv(os.path.join(args.outdir, "ga_history.csv"), index=False)
        best_lp,best_tp,best_pp,best_sp = params_and_simulation_from_vector(lp,tp,pp,sp,best_vec)
        with open(os.path.join(args.outdir, "ga_best_params.json"), "w", encoding="utf-8") as f:
            json.dump({"LaborParams":asdict(best_lp),"TechParams":asdict(best_tp),"PolicyParams":asdict(best_pp),"SimulationParams":asdict(best_sp),"loss":best_score}, f, indent=2)
        sp_ga = SimulationParams(**asdict(best_sp)); sp_ga.seed = sp.seed + 100
        log("Evaluating GA best parameter set")
        df_ga = Economy(best_lp, best_tp, best_pp, sp_ga, outdir=args.outdir).simulate()
        mom_ga = summarize_moments(df_ga)
        df_ga.to_csv(os.path.join(args.outdir, "abm_history_ga.csv"), index=False)
        pd.DataFrame([mom_ga]).to_csv(os.path.join(args.outdir, "abm_moments_ga.csv"), index=False)
        pd.DataFrame([dynamic_diagnostic_moments(df_ga)]).to_csv(os.path.join(args.outdir, "dynamic_diagnostics_ga.csv"), index=False)
        val_ga, val_ga_summary = validate_moments(mom_ga, targets, label="ga")
        val_ga.to_csv(os.path.join(args.outdir, "validation_ga.csv"), index=False)
        val_ga_summary.to_csv(os.path.join(args.outdir, "validation_summary_ga.csv"), index=False)
        validation_tables.append(val_ga)
        validation_summaries.append(val_ga_summary)
        if validation_targets:
            val_ga_holdout, val_ga_holdout_summary = validate_moments(mom_ga, validation_targets, label="ga_holdout")
            val_ga_holdout.to_csv(os.path.join(args.outdir, "validation_holdout_ga.csv"), index=False)
            val_ga_holdout_summary.to_csv(os.path.join(args.outdir, "validation_summary_holdout_ga.csv"), index=False)
            validation_tables.append(val_ga_holdout)
            validation_summaries.append(val_ga_holdout_summary)
        if args.post_calibration_R and args.post_calibration_R > 0:
            log(f"Evaluating GA best parameter set with fresh replications: R={args.post_calibration_R}")
            sp_ga_rep = SimulationParams(**asdict(best_sp))
            sp_ga_rep.seed = sp.seed + 10_000
            sp_ga_rep.monte_carlo_R = int(args.post_calibration_R)
            ga_rep_rows = simulate_msm_replications(best_vec, lp, tp, pp, sp_ga_rep)
            mom_ga_rep = average_msm_moments(ga_rep_rows)
            pd.DataFrame(ga_rep_rows).to_csv(os.path.join(args.outdir, "abm_moment_replications_ga.csv"), index=False)
            pd.DataFrame([mom_ga_rep]).to_csv(os.path.join(args.outdir, "abm_moments_ga_replicated.csv"), index=False)
            val_ga_rep, val_ga_rep_summary = validate_moments(mom_ga_rep, targets, label="ga_replicated")
            val_ga_rep.to_csv(os.path.join(args.outdir, "validation_ga_replicated.csv"), index=False)
            val_ga_rep_summary.to_csv(os.path.join(args.outdir, "validation_summary_ga_replicated.csv"), index=False)
            validation_tables.append(val_ga_rep)
            validation_summaries.append(val_ga_rep_summary)
            if validation_targets:
                val_ga_rep_holdout, val_ga_rep_holdout_summary = validate_moments(mom_ga_rep, validation_targets, label="ga_replicated_holdout")
                val_ga_rep_holdout.to_csv(os.path.join(args.outdir, "validation_holdout_ga_replicated.csv"), index=False)
                val_ga_rep_holdout_summary.to_csv(os.path.join(args.outdir, "validation_summary_holdout_ga_replicated.csv"), index=False)
                validation_tables.append(val_ga_rep_holdout)
                validation_summaries.append(val_ga_rep_holdout_summary)
        if empirical_distributions:
            distributional_tests(df_ga, empirical_distributions, label="ga").to_csv(
                os.path.join(args.outdir, "distribution_tests_ga.csv"),
                index=False,
            )
        compare_figs(df, df_ga, os.path.join(args.outdir, "compare_ga.pdf"), tag="GA")
        compare_moments(moments, mom_ga, targets, os.path.join(args.outdir, "compare_ga_moments.png"))

    df_cma = None
    mom_cma = None
    if args.opt_cmaes:
        log(f"Starting CMA-ES calibration: iters={args.cma_iters}")
        cma_vec, cma_score, cma_history = run_cmaes(lp,tp,pp,sp, targets, seed=2026, iters=args.cma_iters, return_history=True, covariance=covariance, progress=progress)
        log(f"CMA-ES calibration complete: loss={cma_score:.6g}")
        cma_history.to_csv(os.path.join(args.outdir, "cmaes_history.csv"), index=False)
        cma_lp,cma_tp,cma_pp,cma_sp = params_and_simulation_from_vector(lp,tp,pp,sp,cma_vec)
        with open(os.path.join(args.outdir, "cmaes_best_params.json"), "w", encoding="utf-8") as f:
            json.dump({"LaborParams":asdict(cma_lp),"TechParams":asdict(cma_tp),"PolicyParams":asdict(cma_pp),"SimulationParams":asdict(cma_sp),"loss":cma_score}, f, indent=2)
        sp_cma = SimulationParams(**asdict(cma_sp)); sp_cma.seed = sp.seed + 200
        log("Evaluating CMA-ES best parameter set")
        df_cma = Economy(cma_lp, cma_tp, cma_pp, sp_cma, outdir=args.outdir).simulate()
        mom_cma = summarize_moments(df_cma)
        df_cma.to_csv(os.path.join(args.outdir, "abm_history_cmaes.csv"), index=False)
        pd.DataFrame([mom_cma]).to_csv(os.path.join(args.outdir, "abm_moments_cmaes.csv"), index=False)
        pd.DataFrame([dynamic_diagnostic_moments(df_cma)]).to_csv(os.path.join(args.outdir, "dynamic_diagnostics_cmaes.csv"), index=False)
        val_cma, val_cma_summary = validate_moments(mom_cma, targets, label="cmaes")
        val_cma.to_csv(os.path.join(args.outdir, "validation_cmaes.csv"), index=False)
        val_cma_summary.to_csv(os.path.join(args.outdir, "validation_summary_cmaes.csv"), index=False)
        validation_tables.append(val_cma)
        validation_summaries.append(val_cma_summary)
        if validation_targets:
            val_cma_holdout, val_cma_holdout_summary = validate_moments(mom_cma, validation_targets, label="cmaes_holdout")
            val_cma_holdout.to_csv(os.path.join(args.outdir, "validation_holdout_cmaes.csv"), index=False)
            val_cma_holdout_summary.to_csv(os.path.join(args.outdir, "validation_summary_holdout_cmaes.csv"), index=False)
            validation_tables.append(val_cma_holdout)
            validation_summaries.append(val_cma_holdout_summary)
        if args.post_calibration_R and args.post_calibration_R > 0:
            log(f"Evaluating CMA-ES best parameter set with fresh replications: R={args.post_calibration_R}")
            sp_cma_rep = SimulationParams(**asdict(cma_sp))
            sp_cma_rep.seed = sp.seed + 20_000
            sp_cma_rep.monte_carlo_R = int(args.post_calibration_R)
            cma_rep_rows = simulate_msm_replications(cma_vec, lp, tp, pp, sp_cma_rep)
            mom_cma_rep = average_msm_moments(cma_rep_rows)
            pd.DataFrame(cma_rep_rows).to_csv(os.path.join(args.outdir, "abm_moment_replications_cmaes.csv"), index=False)
            pd.DataFrame([mom_cma_rep]).to_csv(os.path.join(args.outdir, "abm_moments_cmaes_replicated.csv"), index=False)
            val_cma_rep, val_cma_rep_summary = validate_moments(mom_cma_rep, targets, label="cmaes_replicated")
            val_cma_rep.to_csv(os.path.join(args.outdir, "validation_cmaes_replicated.csv"), index=False)
            val_cma_rep_summary.to_csv(os.path.join(args.outdir, "validation_summary_cmaes_replicated.csv"), index=False)
            validation_tables.append(val_cma_rep)
            validation_summaries.append(val_cma_rep_summary)
            if validation_targets:
                val_cma_rep_holdout, val_cma_rep_holdout_summary = validate_moments(mom_cma_rep, validation_targets, label="cmaes_replicated_holdout")
                val_cma_rep_holdout.to_csv(os.path.join(args.outdir, "validation_holdout_cmaes_replicated.csv"), index=False)
                val_cma_rep_holdout_summary.to_csv(os.path.join(args.outdir, "validation_summary_holdout_cmaes_replicated.csv"), index=False)
                validation_tables.append(val_cma_rep_holdout)
                validation_summaries.append(val_cma_rep_holdout_summary)
        if empirical_distributions:
            distributional_tests(df_cma, empirical_distributions, label="cmaes").to_csv(
                os.path.join(args.outdir, "distribution_tests_cmaes.csv"),
                index=False,
            )
        compare_figs(df, df_cma, os.path.join(args.outdir, "compare_cmaes.pdf"), tag="CMAES")
        compare_moments(moments, mom_cma, targets, os.path.join(args.outdir, "compare_cmaes_moments.png"))

    df_sa = None
    mom_sa = None
    if args.opt_sa:
        log(f"Starting simulated annealing calibration: iters={args.sa_iters}")
        sa_vec, sa_score, sa_history = run_simulated_annealing(
            lp,tp,pp,sp,targets,
            seed=2028,
            iters=args.sa_iters,
            start_temp=args.sa_start_temp,
            end_temp=args.sa_end_temp,
            step_scale=args.sa_step_scale,
            return_history=True,
            covariance=covariance,
            progress=progress,
        )
        log(f"Simulated annealing complete: best_loss={sa_score:.6g}")
        sa_history.to_csv(os.path.join(args.outdir, "sa_history.csv"), index=False)
        sa_lp,sa_tp,sa_pp,sa_sp = params_and_simulation_from_vector(lp,tp,pp,sp,sa_vec)
        with open(os.path.join(args.outdir, "sa_best_params.json"), "w", encoding="utf-8") as f:
            json.dump({"LaborParams":asdict(sa_lp),"TechParams":asdict(sa_tp),"PolicyParams":asdict(sa_pp),"SimulationParams":asdict(sa_sp),"loss":sa_score}, f, indent=2)
        sp_sa = SimulationParams(**asdict(sa_sp)); sp_sa.seed = sp.seed + 300
        log("Evaluating simulated annealing best parameter set")
        df_sa = Economy(sa_lp, sa_tp, sa_pp, sp_sa, outdir=args.outdir).simulate()
        mom_sa = summarize_moments(df_sa)
        df_sa.to_csv(os.path.join(args.outdir, "abm_history_sa.csv"), index=False)
        pd.DataFrame([mom_sa]).to_csv(os.path.join(args.outdir, "abm_moments_sa.csv"), index=False)
        pd.DataFrame([dynamic_diagnostic_moments(df_sa)]).to_csv(os.path.join(args.outdir, "dynamic_diagnostics_sa.csv"), index=False)
        val_sa, val_sa_summary = validate_moments(mom_sa, targets, label="sa")
        val_sa.to_csv(os.path.join(args.outdir, "validation_sa.csv"), index=False)
        val_sa_summary.to_csv(os.path.join(args.outdir, "validation_summary_sa.csv"), index=False)
        validation_tables.append(val_sa)
        validation_summaries.append(val_sa_summary)
        if validation_targets:
            val_sa_holdout, val_sa_holdout_summary = validate_moments(mom_sa, validation_targets, label="sa_holdout")
            val_sa_holdout.to_csv(os.path.join(args.outdir, "validation_holdout_sa.csv"), index=False)
            val_sa_holdout_summary.to_csv(os.path.join(args.outdir, "validation_summary_holdout_sa.csv"), index=False)
            validation_tables.append(val_sa_holdout)
            validation_summaries.append(val_sa_holdout_summary)
        if args.post_calibration_R and args.post_calibration_R > 0:
            log(f"Evaluating simulated annealing best parameter set with fresh replications: R={args.post_calibration_R}")
            sp_sa_rep = SimulationParams(**asdict(sa_sp))
            sp_sa_rep.seed = sp.seed + 30_000
            sp_sa_rep.monte_carlo_R = int(args.post_calibration_R)
            sa_rep_rows = simulate_msm_replications(sa_vec, lp, tp, pp, sp_sa_rep)
            mom_sa_rep = average_msm_moments(sa_rep_rows)
            pd.DataFrame(sa_rep_rows).to_csv(os.path.join(args.outdir, "abm_moment_replications_sa.csv"), index=False)
            pd.DataFrame([mom_sa_rep]).to_csv(os.path.join(args.outdir, "abm_moments_sa_replicated.csv"), index=False)
            val_sa_rep, val_sa_rep_summary = validate_moments(mom_sa_rep, targets, label="sa_replicated")
            val_sa_rep.to_csv(os.path.join(args.outdir, "validation_sa_replicated.csv"), index=False)
            val_sa_rep_summary.to_csv(os.path.join(args.outdir, "validation_summary_sa_replicated.csv"), index=False)
            validation_tables.append(val_sa_rep)
            validation_summaries.append(val_sa_rep_summary)
            if validation_targets:
                val_sa_rep_holdout, val_sa_rep_holdout_summary = validate_moments(mom_sa_rep, validation_targets, label="sa_replicated_holdout")
                val_sa_rep_holdout.to_csv(os.path.join(args.outdir, "validation_holdout_sa_replicated.csv"), index=False)
                val_sa_rep_holdout_summary.to_csv(os.path.join(args.outdir, "validation_summary_holdout_sa_replicated.csv"), index=False)
                validation_tables.append(val_sa_rep_holdout)
                validation_summaries.append(val_sa_rep_holdout_summary)
        if empirical_distributions:
            distributional_tests(df_sa, empirical_distributions, label="sa").to_csv(
                os.path.join(args.outdir, "distribution_tests_sa.csv"),
                index=False,
            )
        compare_figs(df, df_sa, os.path.join(args.outdir, "compare_sa.pdf"), tag="SA")
        compare_moments(moments, mom_sa, targets, os.path.join(args.outdir, "compare_sa_moments.png"))

    if args.mcmc:
        log(f"Starting MCMC: chains={args.mcmc_chains}, draws={args.mcmc_draws}, burn={args.mcmc_burn}, thin={args.mcmc_thin}")
        mcmc_init_vec = None
        if args.mcmc_init_params:
            log(f"Loading MCMC initial parameter vector from {args.mcmc_init_params}")
            mcmc_init_vec = _vector_from_params_json(args.mcmc_init_params)
        chains = run_mcmc_chains(
            lp,tp,pp,sp,targets,
            chains=args.mcmc_chains,
            draws=args.mcmc_draws,
            burn=args.mcmc_burn,
            thin=args.mcmc_thin,
            step=args.mcmc_step,
            covariance=covariance,
            progress=progress,
            init_vec=mcmc_init_vec,
            log_every=args.mcmc_log_every,
        )
        log("MCMC complete; writing posterior samples and diagnostics")
        for i in range(chains.shape[0]):
            pd.DataFrame(chains[i], columns=PARAM_KEYS).to_csv(os.path.join(args.outdir, f"posterior_samples_chain{i+1}.csv"), index=False)
        rhat = rhat_gelman_rubin(chains)
        pd.DataFrame({"parameter": PARAM_KEYS, "rhat": rhat}).to_csv(os.path.join(args.outdir, "posterior_rhat.csv"), index=False)
        plot_traces(chains, args.outdir, tag="mcmc")
        plot_acf_bars(chains, args.outdir, tag="mcmc")
        flat_samples = chains.reshape(-1, chains.shape[-1])
        posterior_summary, posterior_draws = summarize_posterior(flat_samples)
        posterior_summary.to_csv(os.path.join(args.outdir, "posterior_summary.csv"), index=False)
        posterior_draws.to_csv(os.path.join(args.outdir, "posterior_samples.csv"), index=False)
        log(f"Running posterior predictive checks: draws={args.ppc_draws}")
        ppc_summary, ppc_raw = posterior_predictive(lp,tp,pp,sp,targets, flat_samples, draws=args.ppc_draws, progress=progress)
        ppc_summary.to_csv(os.path.join(args.outdir, "ppc_moments.csv"), index=False)
        ppc_raw.to_csv(os.path.join(args.outdir, "ppc_raw_moments.csv"), index=False)
        plot_ppc(ppc_raw, moments, targets, args.outdir)

    if args.smc:
        log(f"Starting SMC sampler: particles={args.smc_particles}, levels={args.smc_levels}")
        smc_particles, smc_history = run_smc_sampler(
            lp,tp,pp,sp,targets,
            particles=args.smc_particles,
            levels=args.smc_levels,
            rejuvenation_steps=args.smc_rejuvenation,
            step=args.mcmc_step,
            seed=808,
            covariance=covariance,
            progress=progress,
        )
        log("SMC sampler complete")
        pd.DataFrame(smc_particles, columns=PARAM_KEYS).to_csv(os.path.join(args.outdir, "smc_particles.csv"), index=False)
        smc_history.to_csv(os.path.join(args.outdir, "smc_history.csv"), index=False)
        smc_summary, smc_draws = summarize_posterior(smc_particles)
        smc_summary.to_csv(os.path.join(args.outdir, "smc_posterior_summary.csv"), index=False)
        smc_draws.to_csv(os.path.join(args.outdir, "smc_posterior_particles.csv"), index=False)

    if args.pymc:
        log(f"Starting PyMC-assisted importance sampler: draws={args.pymc_draws}")
        try:
            pymc_particles, pymc_weights, pymc_summary = run_pymc_importance_sampler(
                lp,tp,pp,sp,targets,
                draws=args.pymc_draws,
                seed=909,
                covariance=covariance,
            )
        except ImportError as exc:
            parser.error(str(exc))
        pymc_particles.to_csv(os.path.join(args.outdir, "pymc_prior_particles.csv"), index=False)
        pymc_weights.to_csv(os.path.join(args.outdir, "pymc_importance_weights.csv"), index=False)
        pymc_summary.to_csv(os.path.join(args.outdir, "pymc_importance_summary.csv"), index=False)
        log("PyMC-assisted importance sampler complete")

    if args.sensitivity:
        log(f"Starting Sobol-style sensitivity: output={args.sens_output}, N={args.sens_N}")
        S, ST = sobol_sensitivity(lp,tp,pp,sp, outname=args.sens_output, N=args.sens_N, seed=2027, progress=progress)
        save_sobol_tables(S, ST, args.outdir, args.sens_output)
        log("Sobol-style sensitivity complete")

    if args.ofat_sensitivity:
        log(f"Starting OFAT sensitivity: step={args.ofat_step}")
        ofat = ofat_sensitivity(lp, tp, pp, sp, step_fraction=args.ofat_step)
        ofat.to_csv(os.path.join(args.outdir, "ofat_sensitivity.csv"), index=False)
        log("OFAT sensitivity complete")

    if validation_tables:
        pd.concat(validation_tables, ignore_index=True).to_csv(os.path.join(args.outdir, "validation_all.csv"), index=False)
    if validation_summaries:
        pd.concat(validation_summaries, ignore_index=True).to_csv(os.path.join(args.outdir, "validation_summary_all.csv"), index=False)

    if args.report:
        log("Building PDF reports")
        generate_report(os.path.join(args.outdir, "abm_report.pdf"), df, moments, targets,
                        df_ga, mom_ga, df_cma, mom_cma)
        build_combined_report(args.outdir)

    if args.diag_report:
        log("Building diagnostics PDF")
        build_diagnostics_report(args.outdir, tag="mcmc")

    log("Run complete")
    print("Baseline simulated moments:")
    for k,v in moments.items():
        print(f"{k}: {v:.4f}")
    print(f"Outputs written to: {args.outdir}")

if __name__ == "__main__":
    main()

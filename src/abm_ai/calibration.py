"""Calibration optimizers, parameter mappings, and local and global sensitivity screens."""

from __future__ import annotations

import os
from dataclasses import asdict
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .moments import msm_loss, summarize_moments
from .parameters import LaborParams, PolicyParams, SimulationParams, TechParams
from .simulation import Economy
from .utils import rng

_BASE_PARAM_BOUNDS = {
    "LaborParams.alpha": (0.15, 0.30, "LaborParams"),
    "LaborParams.eta_w": (0.15, 0.45, "LaborParams"),
    "LaborParams.zeta": (0.30, 1.20, "LaborParams"),
    "LaborParams.theta_s": (0.30, 1.00, "LaborParams"),
    "LaborParams.lambda_w": (0.05, 0.60, "LaborParams"),
    "TechParams.alpha_R": (0.01, 0.03, "TechParams"),
    "TechParams.beta_I": (0.005, 0.055, "TechParams"),
    "TechParams.beta_P": (0.005, 0.04, "TechParams"),
    "TechParams.bass_q": (0.00, 0.025, "TechParams"),
    "TechParams.depreciation_K": (0.02, 0.10, "TechParams"),
    "TechParams.Theta_star": (1.20, 3.50, "TechParams"),
    "TechParams.phi_A": (0.005, 0.03, "TechParams"),
    "PolicyParams.eta_G": (-0.60, 0.0, "PolicyParams"),
    "PolicyParams.tax_rate": (0.10, 0.40, "PolicyParams"),
    "PolicyParams.r_debt": (0.00, 0.05, "PolicyParams"),
    "SimulationParams.inv_gamma_base": (0.15, 0.50, "SimulationParams"),
}

_OPTIONAL_PARAM_BOUNDS = {
    "SimulationParams.firm_exit_prob_base": (0.005, 0.08, "SimulationParams"),
    "SimulationParams.firm_exit_profit_cutoff": (0.00, 0.10, "SimulationParams"),
    "SimulationParams.firm_exit_margin_cutoff": (0.01, 0.30, "SimulationParams"),
    "SimulationParams.firm_exit_slope": (0.05, 1.50, "SimulationParams"),
    "SimulationParams.firm_exit_A_min": (0.10, 0.60, "SimulationParams"),
    "SimulationParams.entry_reallocation_share": (0.00, 0.80, "SimulationParams"),
    "SimulationParams.entry_vacancy_multiplier": (1.00, 3.00, "SimulationParams"),
}
PARAM_BOUNDS = dict(_BASE_PARAM_BOUNDS)
PARAM_KEYS = list(PARAM_BOUNDS.keys())


def apply_param_bounds_overrides(overrides: Dict[str, Tuple[float, float]] | None) -> None:
    """Apply config-local calibration search bounds.

    The default calibrated parameter vector order is fixed by PARAM_KEYS.
    Overrides may narrow or widen default bounds. A small set of experimental
    exit/reallocation parameters may be activated only when the config supplies
    explicit bounds for them, keeping baseline calibration runs comparable with
    the Section 3 target mapping.
    """
    if not overrides:
        return
    for key, bounds in overrides.items():
        if key not in PARAM_BOUNDS:
            if key not in _OPTIONAL_PARAM_BOUNDS:
                raise ValueError(f"Unknown calibration bound override: {key}")
            PARAM_BOUNDS[key] = _OPTIONAL_PARAM_BOUNDS[key]
            PARAM_KEYS.append(key)
        if len(bounds) != 2:
            raise ValueError(f"Calibration bound override for {key} must contain [low, high]")
        lo, hi = float(bounds[0]), float(bounds[1])
        if not lo < hi:
            raise ValueError(f"Calibration bound override for {key} must satisfy low < high")
        _, _, group = PARAM_BOUNDS[key]
        PARAM_BOUNDS[key] = (lo, hi, group)


def params_from_vector(base_lp, base_tp, base_pp, vec: np.ndarray):
    lp, tp, pp = LaborParams(**asdict(base_lp)), TechParams(**asdict(base_tp)), PolicyParams(**asdict(base_pp))
    for val, key in zip(vec, PARAM_KEYS):
        lo, hi, grp = PARAM_BOUNDS[key]
        v = float(np.clip(val, lo, hi))
        name = key.split(".")[1]
        if grp=="LaborParams": setattr(lp, name, v)
        elif grp=="TechParams": setattr(tp, name, v)
        elif grp=="PolicyParams": setattr(pp, name, v)
    return lp, tp, pp

def simulation_params_from_vector(base_sp: SimulationParams, vec: np.ndarray) -> SimulationParams:
    """Return simulation parameters implied by calibrated vector entries."""
    sp = SimulationParams(**asdict(base_sp))
    for val, key in zip(vec, PARAM_KEYS):
        lo, hi, grp = PARAM_BOUNDS[key]
        if grp != "SimulationParams":
            continue
        v = float(np.clip(val, lo, hi))
        setattr(sp, key.split(".")[1], v)
    return sp

def params_and_simulation_from_vector(base_lp, base_tp, base_pp, base_sp: SimulationParams, vec: np.ndarray):
    """Return behavioral and simulation parameters implied by a vector."""
    lp, tp, pp = params_from_vector(base_lp, base_tp, base_pp, vec)
    sp = simulation_params_from_vector(base_sp, vec)
    return lp, tp, pp, sp

def vector_from_params(lp: LaborParams, tp: TechParams, pp: PolicyParams, sp: SimulationParams | None = None) -> np.ndarray:
    """Return the calibrated parameter vector implied by parameter dataclasses."""
    sp = SimulationParams() if sp is None else sp
    vec = []
    groups = {
        "LaborParams": lp,
        "TechParams": tp,
        "PolicyParams": pp,
        "SimulationParams": sp,
    }
    for key in PARAM_KEYS:
        lo, hi, grp = PARAM_BOUNDS[key]
        name = key.split(".")[1]
        vec.append(float(np.clip(getattr(groups[grp], name), lo, hi)))
    return np.array(vec, dtype=float)

def sample_vector(r: np.random.Generator):
    vec=[]
    for key in PARAM_KEYS:
        lo, hi, _ = PARAM_BOUNDS[key]
        vec.append(r.uniform(lo, hi))
    return np.array(vec, dtype=float)

def simulate_msm_replications(vec: np.ndarray, base_lp, base_tp, base_pp, sp: SimulationParams) -> List[Dict[str, float]]:
    """Simulate one parameter vector and return replication-level MSM moments."""
    lp,tp,pp,sp_candidate = params_and_simulation_from_vector(base_lp, base_tp, base_pp, sp, vec)
    R = max(1, int(getattr(sp, "monte_carlo_R", 1)))
    seed_gen = rng(sp.seed)
    rows = []
    for _ in range(R):
        sp_local = SimulationParams(**asdict(sp_candidate))
        sp_local.seed = int(seed_gen.integers(0, 1_000_000))
        econ = Economy(lp,tp,pp,sp_local)
        rows.append(summarize_moments(econ.simulate()))
    return rows

def average_msm_moments(rows: List[Dict[str, float]]) -> Dict[str, float]:
    """Average replication-level MSM moments."""
    keys = rows[0].keys()
    return {k: float(np.mean([row[k] for row in rows])) for k in keys}

def simulate_msm_moments(vec: np.ndarray, base_lp, base_tp, base_pp, sp: SimulationParams) -> Dict[str, float]:
    """Simulate one parameter vector and return Monte Carlo averaged MSM moments."""
    return average_msm_moments(simulate_msm_replications(vec, base_lp, base_tp, base_pp, sp))

def summarize_moment_replications(rows: List[Dict[str, float]], ci_level: float = 0.95) -> pd.DataFrame:
    """Summarize replication-level moments with normal-approximation confidence intervals."""
    df = pd.DataFrame(rows)
    z = 1.959963984540054 if abs(ci_level - 0.95) < 1e-12 else 1.959963984540054
    out = []
    n = len(df)
    for col in df.columns:
        values = df[col].to_numpy(dtype=float)
        mean = float(np.mean(values))
        sd = float(np.std(values, ddof=1)) if n > 1 else 0.0
        se = sd / float(np.sqrt(max(1, n)))
        out.append({
            "moment": col,
            "replications": int(n),
            "mean": mean,
            "std": sd,
            "se": se,
            "ci_level": float(ci_level),
            "ci_low": mean - z * se,
            "ci_high": mean + z * se,
        })
    return pd.DataFrame(out)

def bootstrap_moment_intervals(
    rows: List[Dict[str, float]],
    *,
    draws: int = 1000,
    ci_level: float = 0.95,
    seed: int = 12345,
) -> pd.DataFrame:
    """Summarize replication-level moments with percentile bootstrap intervals.

    The bootstrap resamples replication rows with replacement and computes the
    mean moment for each resample. Non-finite values are ignored moment by
    moment. If a moment has no finite replication values, interval fields are
    returned as NaN.
    """
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=[
            "moment",
            "replications",
            "finite_replications",
            "draws",
            "ci_level",
            "mean",
            "ci_low",
            "ci_high",
        ])
    r = rng(seed)
    alpha = max(0.0, min(1.0, 1.0 - float(ci_level)))
    q_low = 100.0 * alpha / 2.0
    q_high = 100.0 * (1.0 - alpha / 2.0)
    n = len(df)
    out = []
    B = max(1, int(draws))
    for col in df.columns:
        values = df[col].to_numpy(dtype=float)
        finite_values = values[np.isfinite(values)]
        if len(finite_values) == 0:
            mean = ci_low = ci_high = float("nan")
        else:
            sample_idx = r.integers(0, len(finite_values), size=(B, len(finite_values)))
            boot_means = finite_values[sample_idx].mean(axis=1)
            mean = float(np.mean(finite_values))
            ci_low = float(np.percentile(boot_means, q_low))
            ci_high = float(np.percentile(boot_means, q_high))
        out.append({
            "moment": col,
            "replications": int(n),
            "finite_replications": int(len(finite_values)),
            "draws": int(B),
            "ci_level": float(ci_level),
            "mean": mean,
            "ci_low": ci_low,
            "ci_high": ci_high,
        })
    return pd.DataFrame(out)

def fitness(
    vec: np.ndarray,
    base_lp,
    base_tp,
    base_pp,
    sp: SimulationParams,
    targets: Dict[str, Tuple[float,float]],
    covariance: Tuple[List[str], np.ndarray] | None = None,
) -> float:
    """Evaluate one parameter vector by simulation and MSM loss."""
    mom = simulate_msm_moments(vec, base_lp, base_tp, base_pp, sp)
    return msm_loss(mom, targets, covariance=covariance)

def _optimizer_history_row(
    method: str,
    iteration: int,
    scores: List[float],
    best_vec: np.ndarray,
    *,
    best_loss: float | None = None,
) -> Dict[str, float | str | int]:
    row: Dict[str, float | str | int] = {
        "method": method,
        "iteration": int(iteration),
        "best_loss": float(np.min(scores) if best_loss is None else best_loss),
        "mean_loss": float(np.mean(scores)),
        "median_loss": float(np.median(scores)),
        "worst_loss": float(np.max(scores)),
    }
    for key, value in zip(PARAM_KEYS, best_vec):
        row[key] = float(value)
    return row

ProgressFn = Callable[[str], None]


def _progress(progress: ProgressFn | None, message: str) -> None:
    if progress is not None:
        progress(message)


def _evaluate_population(
    population: List[np.ndarray] | np.ndarray,
    base_lp,
    base_tp,
    base_pp,
    sp: SimulationParams,
    targets,
    *,
    covariance=None,
    progress: ProgressFn | None = None,
    label: str = "population",
) -> List[float]:
    scores = []
    total = len(population)
    for idx, candidate in enumerate(population, start=1):
        _progress(progress, f"{label}: evaluating candidate {idx}/{total}")
        scores.append(fitness(candidate, base_lp, base_tp, base_pp, sp, targets, covariance=covariance))
    return scores


def run_ga(
    base_lp,
    base_tp,
    base_pp,
    sp: SimulationParams,
    targets,
    seed=123,
    pop=24,
    gens=20,
    p_cx=0.8,
    p_mut=0.2,
    mut_sigma=0.1,
    return_history: bool = False,
    covariance=None,
    progress: ProgressFn | None = None,
    init_vec: np.ndarray | None = None,
    init_scale: float = 0.08,
):
    r = rng(seed)
    # init population. When init_vec is supplied, the first candidate is the
    # saved vector and the rest are local perturbations around it.
    if init_vec is None:
        P = [sample_vector(r) for _ in range(pop)]
        init_msg = "random"
    else:
        lo = np.array([PARAM_BOUNDS[k][0] for k in PARAM_KEYS], dtype=float)
        hi = np.array([PARAM_BOUNDS[k][1] for k in PARAM_KEYS], dtype=float)
        center = np.clip(np.asarray(init_vec, dtype=float), lo, hi)
        scale = max(0.0, float(init_scale)) * (hi - lo)
        P = [center.copy()]
        while len(P) < pop:
            P.append(np.clip(center + r.normal(0.0, scale, size=len(PARAM_KEYS)), lo, hi))
        init_msg = f"local around supplied vector, init_scale={float(init_scale):.4g}"
    _progress(progress, f"GA: initial population size={pop}, generations={gens}, init={init_msg}")
    scores = _evaluate_population(P, base_lp, base_tp, base_pp, sp, targets, covariance=covariance, progress=progress, label="GA generation 0")
    best_idx = int(np.argmin(scores))
    global_best_vec = P[best_idx].copy()
    global_best_score = float(scores[best_idx])
    history = [_optimizer_history_row("GA", 0, scores, global_best_vec, best_loss=global_best_score)]
    _progress(progress, f"GA generation 0 complete: best_loss={float(np.min(scores)):.6g}, mean_loss={float(np.mean(scores)):.6g}")
    for g in range(gens):
        # tournament selection
        def select():
            i, j = r.integers(0, pop), r.integers(0, pop)
            return P[i] if scores[i] < scores[j] else P[j]
        newP=[]
        while len(newP) < pop:
            p1, p2 = select(), select()
            c1, c2 = p1.copy(), p2.copy()
            if r.random() < p_cx:
                pt = r.integers(1, len(PARAM_KEYS))
                c1[:pt], c2[:pt] = p2[:pt].copy(), p1[:pt].copy()
            if r.random() < p_mut:
                idx = r.integers(0, len(PARAM_KEYS))
                lo,hi,_ = PARAM_BOUNDS[PARAM_KEYS[idx]]
                c1[idx] = np.clip(c1[idx] + r.normal(0, mut_sigma*(hi-lo)), lo, hi)
            if r.random() < p_mut:
                idx = r.integers(0, len(PARAM_KEYS))
                lo,hi,_ = PARAM_BOUNDS[PARAM_KEYS[idx]]
                c2[idx] = np.clip(c2[idx] + r.normal(0, mut_sigma*(hi-lo)), lo, hi)
            newP += [c1, c2]
        P = newP[:pop]
        scores = _evaluate_population(P, base_lp, base_tp, base_pp, sp, targets, covariance=covariance, progress=progress, label=f"GA generation {g + 1}")
        best_idx = int(np.argmin(scores))
        generation_best_score = float(scores[best_idx])
        if generation_best_score < global_best_score:
            global_best_score = generation_best_score
            global_best_vec = P[best_idx].copy()
        history.append(_optimizer_history_row("GA", g + 1, scores, global_best_vec, best_loss=global_best_score))
        _progress(progress, f"GA generation {g + 1}/{gens} complete: best_loss={global_best_score:.6g}, mean_loss={float(np.mean(scores)):.6g}")
    best_vec = global_best_vec
    best_score = global_best_score
    if return_history:
        return best_vec, best_score, pd.DataFrame(history)
    return best_vec, best_score

def run_cmaes(base_lp, base_tp, base_pp, sp: SimulationParams, targets, seed=321, iters=30, mu=6, return_history: bool = False, covariance=None, progress: ProgressFn | None = None):
    # Minimal diagonal CMA-ES
    r = rng(seed)
    dim = len(PARAM_KEYS)
    # init mean at midpoint of bounds
    lo = np.array([PARAM_BOUNDS[k][0] for k in PARAM_KEYS], dtype=float)
    hi = np.array([PARAM_BOUNDS[k][1] for k in PARAM_KEYS], dtype=float)
    m = (lo + hi)/2.0
    sigma = 0.25 * (hi - lo)  # step-size per-dimension (diagonal)
    lam = 2*mu
    weights = np.log(mu + 0.5) - np.log(np.arange(1, mu+1))
    weights /= weights.sum()
    history = []
    global_best_vec = m.copy()
    global_best_score = float("inf")
    _progress(progress, f"CMA-ES: iterations={iters}, lambda={lam}, mu={mu}")
    for t in range(iters):
        # sample population
        Z = r.normal(0,1,size=(lam, dim))
        X = m + Z * sigma
        # clip to bounds
        X = np.clip(X, lo, hi)
        # evaluate
        scores = _evaluate_population(X, base_lp, base_tp, base_pp, sp, targets, covariance=covariance, progress=progress, label=f"CMA-ES iteration {t + 1}")
        idx = np.argsort(scores)[:mu]
        generation_best_idx = int(np.argmin(scores))
        generation_best_score = float(scores[generation_best_idx])
        if generation_best_score < global_best_score:
            global_best_score = generation_best_score
            global_best_vec = X[generation_best_idx].copy()
        history.append(_optimizer_history_row("CMA-ES", t + 1, scores, global_best_vec, best_loss=global_best_score))
        _progress(progress, f"CMA-ES iteration {t + 1}/{iters} complete: best_loss={global_best_score:.6g}, mean_loss={float(np.mean(scores)):.6g}")
        elites = X[idx]; m = np.average(elites, axis=0, weights=weights)
        # update sigma (simple): rank-μ based spread of elites
        spread = np.sqrt(np.average((elites - m)**2, axis=0, weights=weights))
        sigma = 0.9*sigma + 0.1*spread
    # final evaluation
    _progress(progress, "CMA-ES: evaluating final mean candidate")
    final_score = fitness(m, base_lp, base_tp, base_pp, sp, targets, covariance=covariance)
    if float(final_score) < global_best_score:
        global_best_score = float(final_score)
        global_best_vec = m.copy()
    _progress(progress, f"CMA-ES complete: best_loss={global_best_score:.6g}, final_mean_loss={float(final_score):.6g}")
    if return_history:
        return global_best_vec, float(global_best_score), pd.DataFrame(history)
    return global_best_vec, float(global_best_score)

def run_simulated_annealing(
    base_lp,
    base_tp,
    base_pp,
    sp: SimulationParams,
    targets,
    seed=111,
    iters=100,
    start_temp=1.0,
    end_temp=0.01,
    step_scale=0.10,
    return_history: bool = False,
    covariance=None,
    progress: ProgressFn | None = None,
):
    """Run a bounded simulated annealing optimizer for MSM calibration."""
    r = rng(seed)
    dim = len(PARAM_KEYS)
    lo = np.array([PARAM_BOUNDS[k][0] for k in PARAM_KEYS], dtype=float)
    hi = np.array([PARAM_BOUNDS[k][1] for k in PARAM_KEYS], dtype=float)
    cur = (lo + hi) / 2.0
    _progress(progress, f"SA: iterations={iters}, start_temp={start_temp}, end_temp={end_temp}")
    cur_score = fitness(cur, base_lp, base_tp, base_pp, sp, targets, covariance=covariance)
    best = cur.copy()
    best_score = float(cur_score)
    n_iters = max(1, int(iters))
    t0 = max(float(start_temp), 1e-12)
    t1 = max(float(end_temp), 1e-12)
    prop_scale = max(float(step_scale), 1e-12) * (hi - lo)
    history = []
    accepted = 0
    for i in range(1, n_iters + 1):
        frac = (i - 1) / max(1, n_iters - 1)
        temp = t0 * ((t1 / t0) ** frac)
        proposal = np.clip(cur + r.normal(0.0, prop_scale, size=dim), lo, hi)
        proposal_score = fitness(proposal, base_lp, base_tp, base_pp, sp, targets, covariance=covariance)
        delta = float(proposal_score - cur_score)
        accept = delta <= 0.0 or np.log(r.random()) < (-delta / max(temp, 1e-12))
        if accept:
            cur = proposal
            cur_score = float(proposal_score)
            accepted += 1
            if cur_score < best_score:
                best = cur.copy()
                best_score = float(cur_score)
        row = _optimizer_history_row("SA", i, [best_score, cur_score], best)
        row["current_loss"] = float(cur_score)
        row["temperature"] = float(temp)
        row["accepted"] = int(accepted)
        row["acceptance_rate"] = float(accepted / i)
        history.append(row)
        _progress(progress, f"SA iteration {i}/{n_iters} complete: current_loss={cur_score:.6g}, best_loss={best_score:.6g}, acceptance_rate={accepted / i:.3f}")
    if return_history:
        return best, float(best_score), pd.DataFrame(history)
    return best, float(best_score)

# ---------------------------
# Sobol-style global sensitivity
# ---------------------------
def evaluate_output(vec, base_lp, base_tp, base_pp, sp, outname="Y_growth"):
    mom = simulate_msm_moments(vec, base_lp, base_tp, base_pp, sp)
    return float(mom[outname])

def sobol_sensitivity(base_lp, base_tp, base_pp, sp, outname="Y_growth", N=128, seed=777, progress: ProgressFn | None = None):
    r = rng(seed)
    d = len(PARAM_KEYS)
    lo = np.array([PARAM_BOUNDS[k][0] for k in PARAM_KEYS], dtype=float)
    hi = np.array([PARAM_BOUNDS[k][1] for k in PARAM_KEYS], dtype=float)
    # Sample A and B
    A = r.uniform(lo, hi, size=(N, d))
    B = r.uniform(lo, hi, size=(N, d))
    _progress(progress, f"Sobol: evaluating A matrix for {outname}, N={N}")
    YA = np.array([evaluate_output(A[i], base_lp, base_tp, base_pp, sp, outname) for i in range(N)])
    _progress(progress, f"Sobol: evaluating B matrix for {outname}, N={N}")
    YB = np.array([evaluate_output(B[i], base_lp, base_tp, base_pp, sp, outname) for i in range(N)])
    VY = np.var(np.concatenate([YA, YB]), ddof=1)
    S = np.zeros(d); ST = np.zeros(d)
    for i in range(d):
        _progress(progress, f"Sobol: evaluating hybrid matrix {i + 1}/{d} ({PARAM_KEYS[i]})")
        ABi = A.copy(); ABi[:, i] = B[:, i]
        YABi = np.array([evaluate_output(ABi[j], base_lp, base_tp, base_pp, sp, outname) for j in range(N)])
        # First-order index (Jansen):
        S[i] = 1.0 - np.mean((YB - YABi)**2) / (2*VY + 1e-12)
        # Total-order index (Jansen):
        ST[i] = np.mean((YA - YABi)**2) / (2*VY + 1e-12)
    return S, ST

def save_sobol_tables(S, ST, outdir, outname):
    os.makedirs(outdir, exist_ok=True)
    df = pd.DataFrame({"parameter": PARAM_KEYS, "S_first": S, "S_total": ST})
    df.to_csv(os.path.join(outdir, f"sobol_{outname}.csv"), index=False)
    # simple bar charts (one per figure)
    plt.figure(); plt.bar(np.arange(len(S)), S); plt.xticks(np.arange(len(S)), [k.replace("Params.","\n") for k in PARAM_KEYS], rotation=45, ha='right')
    plt.title(f"Sobol First-Order Indices: {outname}"); plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"sobol_first_{outname}.png")); plt.close()
    plt.figure(); plt.bar(np.arange(len(ST)), ST); plt.xticks(np.arange(len(ST)), [k.replace("Params.","\n") for k in PARAM_KEYS], rotation=45, ha='right')
    plt.title(f"Sobol Total-Order Indices: {outname}"); plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"sobol_total_{outname}.png")); plt.close()


def ofat_sensitivity(
    base_lp: LaborParams,
    base_tp: TechParams,
    base_pp: PolicyParams,
    sp: SimulationParams,
    *,
    step_fraction: float = 0.10,
    moment_names: List[str] | None = None,
) -> pd.DataFrame:
    """Compute local one-factor-at-a-time sensitivity around current parameters.

    Each calibrated parameter is perturbed down and up by a fixed fraction of
    its calibration range, clipped to the admissible bounds. The same simulation
    seed and Monte Carlo replication scheme are reused for each perturbation.
    """
    base_vec = vector_from_params(base_lp, base_tp, base_pp, sp)
    baseline = simulate_msm_moments(base_vec, base_lp, base_tp, base_pp, sp)
    selected = list(moment_names) if moment_names is not None else list(baseline.keys())
    rows = []
    frac = max(0.0, float(step_fraction))
    eps = 1e-12
    for idx, key in enumerate(PARAM_KEYS):
        lo, hi, _ = PARAM_BOUNDS[key]
        base_value = float(base_vec[idx])
        step = frac * float(hi - lo)
        for direction, candidate_value in [("down", base_value - step), ("up", base_value + step)]:
            perturbed_value = float(np.clip(candidate_value, lo, hi))
            if abs(perturbed_value - base_value) <= eps:
                continue
            vec = base_vec.copy()
            vec[idx] = perturbed_value
            perturbed = simulate_msm_moments(vec, base_lp, base_tp, base_pp, sp)
            param_change = perturbed_value - base_value
            param_pct_change = param_change / max(abs(base_value), eps)
            for moment in selected:
                base_moment = float(baseline.get(moment, np.nan))
                perturbed_moment = float(perturbed.get(moment, np.nan))
                delta = perturbed_moment - base_moment if np.isfinite(base_moment) and np.isfinite(perturbed_moment) else float("nan")
                pct_delta = delta / max(abs(base_moment), eps) if np.isfinite(delta) else float("nan")
                elasticity = pct_delta / param_pct_change if np.isfinite(pct_delta) and abs(param_pct_change) > eps else float("nan")
                rows.append({
                    "parameter": key,
                    "direction": direction,
                    "base_value": base_value,
                    "perturbed_value": perturbed_value,
                    "parameter_change": param_change,
                    "parameter_pct_change": param_pct_change,
                    "moment": moment,
                    "baseline_moment": base_moment,
                    "perturbed_moment": perturbed_moment,
                    "delta": delta,
                    "pct_delta": pct_delta,
                    "elasticity": elasticity,
                })
    return pd.DataFrame(rows)

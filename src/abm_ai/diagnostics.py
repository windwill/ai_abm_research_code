"""Exploratory MCMC, SMC, posterior predictive, and convergence diagnostics."""

from __future__ import annotations

import glob
import json
import math
import os
from dataclasses import asdict
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .calibration import PARAM_BOUNDS, PARAM_KEYS, fitness, params_and_simulation_from_vector
from .moments import summarize_moments
from .parameters import SimulationParams
from .simulation import Economy
from .utils import rng

ProgressFn = Callable[[str], None]


def _progress(progress: ProgressFn | None, message: str) -> None:
    if progress is not None:
        progress(message)


def _should_log_index(index: int, total: int) -> bool:
    """Return True for useful progress milestones in long particle loops."""
    if total <= 20:
        return True
    step = max(1, total // 20)
    return index == 1 or index == total or index % step == 0


def _ordered_interval_errors(df: pd.DataFrame) -> tuple[pd.Series, np.ndarray, pd.DataFrame]:
    """Return nonnegative lower/upper plotting errors for interval summaries."""
    work = df.copy()
    for col in ["mean", "p2.5", "p97.5"]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    finite = np.isfinite(work[["mean", "p2.5", "p97.5"]]).all(axis=1)
    work = work.loc[finite].copy()
    lower = np.minimum(work["p2.5"].to_numpy(dtype=float), work["p97.5"].to_numpy(dtype=float))
    upper = np.maximum(work["p2.5"].to_numpy(dtype=float), work["p97.5"].to_numpy(dtype=float))
    mean = work["mean"].to_numpy(dtype=float)
    xerr = np.vstack([
        np.maximum(0.0, mean - lower),
        np.maximum(0.0, upper - mean),
    ])
    return work["mean"], xerr, work

def pymc_available() -> bool:
    """Return True when optional PyMC dependencies are importable."""
    try:
        import pymc  # noqa: F401
    except Exception:
        return False
    return True

def save_empirical_prior(samples: np.ndarray, outpath: str):
    # samples shape: [n_samples, dim] or [n_chains, n_samples, dim]
    if samples.ndim == 3:
        flat = samples.reshape(-1, samples.shape[-1])
    else:
        flat = samples
    means = flat.mean(axis=0)
    stds = flat.std(axis=0, ddof=1) + 1e-9
    obj = {k: {"mean": float(m), "std": float(s)} for k, m, s in zip(PARAM_KEYS, means, stds)}
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def load_empirical_prior(path: str):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    means = []; stds = []
    for k in PARAM_KEYS:
        d = obj.get(k, {"mean": None, "std": None})
        means.append(d["mean"]); stds.append(d["std"])
    return np.array(means, dtype=float), np.array(stds, dtype=float)

def rhat_gelman_rubin(chains: np.ndarray):
    """
    chains: shape (n_chains, n_samples, dim). Returns array of R-hat per dim.
    """
    m, n, d = chains.shape
    # per-parameter stats
    chain_means = chains.mean(axis=1)         # (m, d)
    chain_vars = chains.var(axis=1, ddof=1)   # (m, d)
    B = n * chain_means.var(axis=0, ddof=1)   # (d,)
    W = chain_vars.mean(axis=0)               # (d,)
    var_hat = (n - 1)/n * W + B/n
    Rhat = np.sqrt(np.maximum(var_hat / (W + 1e-12), 1e-12))
    return Rhat

def plot_traces(chains: np.ndarray, outdir: str, tag: str = "mcmc"):
    os.makedirs(outdir, exist_ok=True)
    # chains: (n_chains, n_samples, dim)
    m, n, d = chains.shape
    for j in range(d):
        plt.figure()
        for i in range(m):
            plt.plot(np.arange(n), chains[i, :, j], alpha=0.8, label=f"chain {i+1}")
        plt.title(f"Trace: {PARAM_KEYS[j]} ({tag})")
        plt.xlabel("iteration"); plt.ylabel(PARAM_KEYS[j]); plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"trace_{tag}_{j:02d}.png"))
        plt.close()

# ---------------------------
# Bayesian MCMC (Metropolis-Hastings, likelihood-free)
# posterior(theta) ∝ exp(-0.5 * loss(theta))
# ---------------------------
def _reflect(x, lo, hi):
    # Reflect x into [lo, hi]
    rng = hi - lo
    if rng <= 0: return np.clip(x, lo, hi)
    y = (x - lo) % (2*rng)
    return lo + (2*rng - y if y > rng else y)

def _prior_logpdf(vec: np.ndarray) -> float:
    # Uniform inside bounds. Proposals are reflected into bounds before evaluation,
    # so the prior contributes a constant to the pseudo-posterior.
    return 0.0

def _logpost(vec: np.ndarray, base_lp, base_tp, base_pp, sp, targets, covariance=None) -> float:
    L = fitness(vec, base_lp, base_tp, base_pp, sp, targets, covariance=covariance)  # uses simulation
    return _prior_logpdf(vec) - 0.5 * L  # energy-based pseudo-likelihood

def run_mcmc(base_lp, base_tp, base_pp, sp: SimulationParams, targets,
             draws=1000, burn=200, thin=1, step=0.1, seed=4242, covariance=None,
             progress: ProgressFn | None = None, init_vec: np.ndarray | None = None,
             log_every: int | None = None):
    r = rng(seed)
    dim = len(PARAM_KEYS)
    lo = np.array([PARAM_BOUNDS[k][0] for k in PARAM_KEYS], dtype=float)
    hi = np.array([PARAM_BOUNDS[k][1] for k in PARAM_KEYS], dtype=float)
    # Initialize at midpoint unless a calibrated vector is supplied.
    cur = (lo + hi)/2.0 if init_vec is None else np.clip(np.asarray(init_vec, dtype=float), lo, hi)
    cur_logp = _logpost(cur, base_lp, base_tp, base_pp, sp, targets, covariance=covariance)

    samples = []
    acc = 0
    # per-dimension proposal scales
    prop_scale = step * (hi - lo)
    # Adaptation params
    target_acc = 0.25
    window = max(20, burn // 10)
    adapt_eta = 0.1  # learning rate

    accepted_window = 0
    total_iters = draws + burn
    log_every = max(1, int(log_every)) if log_every is not None and int(log_every) > 0 else max(1, total_iters // 20)
    init_label = "midpoint" if init_vec is None else "supplied calibrated vector"
    _progress(progress, f"MCMC chain seed={seed}: burn={burn}, draws={draws}, thin={thin}, init={init_label}")
    for t in range(draws + burn):
        prop = cur + r.normal(0, prop_scale, size=dim)
        # reflect into bounds
        for i in range(dim):
            rng_ = hi[i] - lo[i]
            y = (prop[i] - lo[i]) % (2*rng_)
            prop[i] = lo[i] + (2*rng_ - y if y > rng_ else y)
        prop_logp = _logpost(prop, base_lp, base_tp, base_pp, sp, targets, covariance=covariance)
        accept = False
        if np.log(r.random()) < (prop_logp - cur_logp):
            cur, cur_logp = prop, prop_logp
            accept = True
            if t >= burn:
                acc += 1
        # Adapt during burn-in
        if t < burn:
            accepted_window += 1 if accept else 0
            if (t + 1) % window == 0:
                acc_rate_w = accepted_window / window
                # multiplicative update; keep scales within [1e-6, (hi-lo)]
                adj = math.exp(adapt_eta * (acc_rate_w - target_acc))
                prop_scale = np.clip(prop_scale * adj, 1e-6, (hi - lo))
                accepted_window = 0
        # Save draws after burn-in
        if t >= burn and ((t - burn) % thin == 0):
            samples.append(cur.copy())
        if (t + 1) % log_every == 0 or (t + 1) == total_iters:
            kept = len(samples)
            _progress(progress, f"MCMC chain seed={seed}: iteration {t + 1}/{total_iters}, kept={kept}, accepted_post_burn={acc}")

    post_burn_iters = max(1, total_iters - burn)
    acc_rate = acc / post_burn_iters
    _progress(progress, f"MCMC chain seed={seed} complete: kept={len(samples)}, acceptance_rate={acc_rate:.3f}")
    return np.array(samples), float(acc_rate)

def plot_ppc(ppc_raw: pd.DataFrame, baseline_moments: Dict[str, float], targets: Dict[str, Tuple[float,float]], outdir: str):
    os.makedirs(outdir, exist_ok=True)
    for col in ppc_raw.columns:
        plt.figure()
        # violin plot for 1D: matplotlib violinplot expects a sequence of arrays
        plt.violinplot([ppc_raw[col].values], showmeans=True, showextrema=True)
        plt.xticks([1], [col])
        # overlay target and baseline markers
        tgt = targets.get(col, (np.nan, 1.0))[0]
        base = baseline_moments.get(col, np.nan)
        if np.isfinite(tgt):
            plt.axhline(tgt, linestyle="--", linewidth=1)
        if np.isfinite(base):
            plt.axhline(base, linestyle=":", linewidth=1)
        plt.title(f"PPC for {col}")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"ppc_{col}.png"))
        plt.close()

def summarize_posterior(samples: np.ndarray):
    # Return mean and 95% credible intervals per parameter key
    df = pd.DataFrame(samples, columns=PARAM_KEYS)
    summary = []
    for k in PARAM_KEYS:
        s = df[k]
        summary.append({
            "parameter": k,
            "mean": float(s.mean()),
            "p2.5": float(s.quantile(0.025)),
            "p97.5": float(s.quantile(0.975))
        })
    return pd.DataFrame(summary), df

def posterior_predictive(base_lp, base_tp, base_pp, sp, targets, samples: np.ndarray, draws: int = 100, progress: ProgressFn | None = None):
    # Evaluate moments for a subset of posterior samples
    idx = np.linspace(0, len(samples)-1, num=min(draws, len(samples)), dtype=int)
    mom_rows = []
    total = len(idx)
    for draw_no, i in enumerate(idx, start=1):
        _progress(progress, f"PPC: simulating draw {draw_no}/{total}")
        vec = samples[i]
        lp,tp,pp,sp_candidate = params_and_simulation_from_vector(base_lp, base_tp, base_pp, sp, vec)
        sp_local = SimulationParams(**asdict(sp_candidate)); sp_local.seed = int(rng(sp.seed).integers(0, 1_000_000))
        econ = Economy(lp,tp,pp,sp_local); df = econ.simulate(); m = summarize_moments(df)
        mom_rows.append(m)
    mdf = pd.DataFrame(mom_rows)
    # Summaries
    out = []
    for col in mdf.columns:
        out.append({
            "moment": col,
            "mean": float(mdf[col].mean()),
            "p2.5": float(mdf[col].quantile(0.025)),
            "p97.5": float(mdf[col].quantile(0.975))
        })
    return pd.DataFrame(out), mdf



def run_mcmc_chains(base_lp, base_tp, base_pp, sp, targets, chains=4, covariance=None,
                    progress: ProgressFn | None = None, init_vec: np.ndarray | None = None,
                    log_every: int | None = None, **kwargs):
    # Run multiple independent MH chains for R-hat diagnostics
    all_samps = []
    lo = np.array([PARAM_BOUNDS[k][0] for k in PARAM_KEYS], dtype=float)
    hi = np.array([PARAM_BOUNDS[k][1] for k in PARAM_KEYS], dtype=float)
    for c in range(chains):
        _progress(progress, f"MCMC: starting chain {c + 1}/{chains}")
        chain_init = None
        if init_vec is not None:
            # Light chain-specific jitter avoids identical starting states while
            # keeping chains close to the selected calibrated candidate.
            jitter_rng = rng(9000 + c)
            chain_init = np.clip(
                np.asarray(init_vec, dtype=float) + jitter_rng.normal(0.0, 0.01 * (hi - lo), size=len(PARAM_KEYS)),
                lo,
                hi,
            )
        s, acc = run_mcmc(
            base_lp,
            base_tp,
            base_pp,
            sp,
            targets,
            seed=4242 + 17*c,
            covariance=covariance,
            progress=progress,
            init_vec=chain_init,
            log_every=log_every,
            **kwargs,
        )
        _progress(progress, f"MCMC: chain {c + 1}/{chains} complete, acceptance_rate={acc:.3f}")
        all_samps.append(s)
    # Pad to same length if necessary
    minlen = min(s.shape[0] for s in all_samps)
    all_samps = np.stack([s[:minlen] for s in all_samps], axis=0)  # (chains, minlen, dim)
    return all_samps

def _effective_sample_size(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=float)
    total = float(np.sum(weights))
    if total <= 0.0:
        return 0.0
    w = weights / total
    return float(1.0 / np.sum(w ** 2))


def run_smc_sampler(
    base_lp,
    base_tp,
    base_pp,
    sp,
    targets,
    *,
    particles: int = 64,
    levels: int = 6,
    rejuvenation_steps: int = 1,
    step: float = 0.08,
    seed: int = 707,
    covariance=None,
    progress: ProgressFn | None = None,
):
    """Run a dependency-free SMC sampler over the MSM pseudo-posterior."""
    r = rng(seed)
    dim = len(PARAM_KEYS)
    n = max(2, int(particles))
    L = max(2, int(levels))
    lo = np.array([PARAM_BOUNDS[k][0] for k in PARAM_KEYS], dtype=float)
    hi = np.array([PARAM_BOUNDS[k][1] for k in PARAM_KEYS], dtype=float)
    _progress(progress, f"SMC: initializing particles={n}, levels={L}, rejuvenation_steps={rejuvenation_steps}")
    particles_arr = r.uniform(lo, hi, size=(n, dim))
    _progress(progress, "SMC: evaluating initial particles")
    initial_losses = []
    for j, particle in enumerate(particles_arr, start=1):
        if _should_log_index(j, n):
            _progress(progress, f"SMC initial particle {j}/{n}")
        initial_losses.append(fitness(particle, base_lp, base_tp, base_pp, sp, targets, covariance=covariance))
    losses = np.array(initial_losses, dtype=float)
    _progress(progress, f"SMC initial particle evaluation complete: best_loss={float(np.min(losses)):.6g}, mean_loss={float(np.mean(losses)):.6g}")
    betas = np.linspace(0.0, 1.0, L)
    prop_scale = max(float(step), 1e-12) * (hi - lo)
    history = []
    prev_beta = 0.0
    total_accept = 0
    total_moves = 0
    for level_idx, beta in enumerate(betas[1:], start=1):
        _progress(progress, f"SMC: level {level_idx}/{L - 1}, beta={float(beta):.3f}")
        delta_beta = float(beta - prev_beta)
        logw = -0.5 * delta_beta * losses
        logw -= float(np.max(logw))
        raw_weights = np.exp(logw)
        weights = raw_weights / max(float(np.sum(raw_weights)), 1e-300)
        ess_before = _effective_sample_size(weights)
        idx = r.choice(np.arange(n), size=n, replace=True, p=weights)
        particles_arr = particles_arr[idx]
        losses = losses[idx]
        accepted = 0
        moves = 0
        n_rejuvenation = max(0, int(rejuvenation_steps))
        for rejuvenation_idx in range(n_rejuvenation):
            _progress(progress, f"SMC level {level_idx}/{L - 1}: rejuvenation pass {rejuvenation_idx + 1}/{n_rejuvenation}")
            for j in range(n):
                if _should_log_index(j + 1, n):
                    _progress(progress, f"SMC level {level_idx}/{L - 1}: rejuvenation {rejuvenation_idx + 1}/{n_rejuvenation}, particle {j + 1}/{n}")
                cur = particles_arr[j]
                cur_loss = float(losses[j])
                proposal = np.clip(cur + r.normal(0.0, prop_scale, size=dim), lo, hi)
                proposal_loss = fitness(proposal, base_lp, base_tp, base_pp, sp, targets, covariance=covariance)
                log_alpha = -0.5 * float(beta) * (float(proposal_loss) - cur_loss)
                moves += 1
                if np.log(r.random()) < log_alpha:
                    particles_arr[j] = proposal
                    losses[j] = float(proposal_loss)
                    accepted += 1
        total_accept += accepted
        total_moves += moves
        best_idx = int(np.argmin(losses))
        row = {
            "level": int(level_idx),
            "beta": float(beta),
            "ess_before_resample": float(ess_before),
            "mean_loss": float(np.mean(losses)),
            "best_loss": float(np.min(losses)),
            "acceptance_rate_level": float(accepted / moves) if moves else float("nan"),
            "acceptance_rate_total": float(total_accept / total_moves) if total_moves else float("nan"),
        }
        for key, value in zip(PARAM_KEYS, particles_arr[best_idx]):
            row[key] = float(value)
        history.append(row)
        _progress(progress, f"SMC level {level_idx}/{L - 1} complete: best_loss={row['best_loss']:.6g}, mean_loss={row['mean_loss']:.6g}, ess={ess_before:.3f}")
        prev_beta = float(beta)
    return particles_arr, pd.DataFrame(history)


def _safe_pymc_name(key: str) -> str:
    return key.replace(".", "__")


def run_pymc_importance_sampler(
    base_lp,
    base_tp,
    base_pp,
    sp,
    targets,
    *,
    draws: int = 128,
    seed: int = 909,
    covariance=None,
):
    """Run an optional PyMC-assisted prior importance sampler.

    PyMC is used to define and sample bounded prior variables. The ABM remains
    outside PyMC's differentiable graph, so MSM pseudo-likelihood weights are
    evaluated after prior sampling and normalized for posterior-style summaries.
    """
    try:
        import pymc as pm
    except Exception as exc:
        raise ImportError("PyMC is optional. Install with `pip install -e .[pymc]` to use --pymc.") from exc

    n = max(2, int(draws))
    lo = np.array([PARAM_BOUNDS[k][0] for k in PARAM_KEYS], dtype=float)
    hi = np.array([PARAM_BOUNDS[k][1] for k in PARAM_KEYS], dtype=float)
    name_map = {_safe_pymc_name(k): k for k in PARAM_KEYS}
    with pm.Model():
        for safe_name, key in name_map.items():
            idx = PARAM_KEYS.index(key)
            pm.Uniform(safe_name, lower=float(lo[idx]), upper=float(hi[idx]))
        prior = pm.sample_prior_predictive(samples=n, random_seed=seed, return_inferencedata=False)

    particles = np.column_stack([
        np.asarray(prior[_safe_pymc_name(key)], dtype=float).reshape(-1)[:n]
        for key in PARAM_KEYS
    ])
    losses = np.array([
        fitness(vec, base_lp, base_tp, base_pp, sp, targets, covariance=covariance)
        for vec in particles
    ], dtype=float)
    logw = -0.5 * losses
    logw -= float(np.max(logw))
    raw = np.exp(logw)
    weights = raw / max(float(np.sum(raw)), 1e-300)
    ess = _effective_sample_size(weights)
    weights_df = pd.DataFrame({
        "draw": np.arange(n, dtype=int),
        "loss": losses,
        "weight": weights,
        "ess": float(ess),
    })
    particles_df = pd.DataFrame(particles, columns=PARAM_KEYS)
    summary = []
    for key in PARAM_KEYS:
        values = particles_df[key].to_numpy(dtype=float)
        order = np.argsort(values)
        sorted_values = values[order]
        sorted_weights = weights[order]
        cdf = np.cumsum(sorted_weights)
        summary.append({
            "parameter": key,
            "mean": float(np.sum(values * weights)),
            "p2.5": float(sorted_values[np.searchsorted(cdf, 0.025, side="left")]),
            "p97.5": float(sorted_values[min(len(sorted_values) - 1, np.searchsorted(cdf, 0.975, side="left"))]),
        })
    return particles_df, weights_df, pd.DataFrame(summary)



# ---------------------------
# Autocorrelation, ESS, and Diagnostics Report
# ---------------------------
def acf_1d(x: np.ndarray, max_lag: int = 100) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    n = len(x)
    max_lag = min(max_lag, n - 2) if n > 2 else 1
    if n < 3:
        return np.zeros(max_lag+1, dtype=float)
    denom = np.sum(x * x)
    ac = np.empty(max_lag + 1, dtype=float)
    ac[0] = 1.0
    for k in range(1, max_lag + 1):
        ac[k] = np.dot(x[:-k], x[k:]) / (denom + 1e-12)
    return ac

def ess_concatenated(chains: np.ndarray, max_lag: int = 100) -> np.ndarray:
    """
    chains: (n_chains, n_samples, dim)
    Returns ESS per parameter (dim,)
    Uses simple initial-negative truncation on concatenated, chain-centered draws.
    """
    m, n, d = chains.shape
    ess = np.zeros(d, dtype=float)
    for j in range(d):
        Y = chains[:, :, j]
        # center each chain, then concatenate
        Yc = (Y - Y.mean(axis=1, keepdims=True)).reshape(-1)
        ac = acf_1d(Yc, max_lag=max_lag)
        # Geyer's initial monotone + nonnegative truncation (simple approx):
        # truncate at first k where ac[k] < 0
        pos_ac = [ac[0]]
        for k in range(1, len(ac)):
            if ac[k] <= 0:
                break
            pos_ac.append(ac[k])
        tau = 1 + 2 * np.sum(pos_ac[1:]) if len(pos_ac) > 1 else 1.0
        N = Yc.shape[0]
        ess[j] = max(1.0, N / tau)
    return ess

def plot_acf_bars(chains: np.ndarray, outdir: str, max_lag: int = 60, tag: str = "mcmc"):
    os.makedirs(outdir, exist_ok=True)
    m, n, d = chains.shape
    for j in range(d):
        Y = chains[:, :, j]
        Yc = (Y - Y.mean(axis=1, keepdims=True)).reshape(-1)
        ac = acf_1d(Yc, max_lag=max_lag)
        plt.figure()
        plt.bar(np.arange(len(ac)), ac)
        plt.title(f"ACF: {PARAM_KEYS[j]} ({tag})")
        plt.xlabel("lag"); plt.ylabel("autocorrelation")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"acf_{tag}_{j:02d}.png"))
        plt.close()

def build_diagnostics_report(outdir: str, tag: str = "mcmc"):
    """
    Create a merged diagnostics PDF that includes:
    - R-hat bar chart (if posterior_rhat.csv present)
    - ESS bar chart (if posterior_samples_chain*.csv present)
    - A selection of trace plots and ACF plots (first 6 params)
    - PPC summary (if ppc_moments.csv present) + small table
    """
    from matplotlib.backends.backend_pdf import PdfPages
    import glob
    report_path = os.path.join(outdir, f"diagnostics_{tag}.pdf")
    with PdfPages(report_path) as pdf:
        # Cover
        fig = plt.figure()
        fig.text(0.5, 0.9, f"Diagnostics Report ({tag})", ha="center", va="center", fontsize=14)
        fig.text(0.1, 0.8, f"Output directory: {outdir}", fontsize=9)
        fig.text(0.1, 0.76, "Includes R-hat, ESS, trace/ACF samples, and PPC summary if available.", fontsize=9)
        pdf.savefig(fig); plt.close(fig)

        # R-hat
        rhat_path = os.path.join(outdir, "posterior_rhat.csv")
        if os.path.exists(rhat_path):
            df_r = pd.read_csv(rhat_path)
            fig = plt.figure()
            plt.bar(np.arange(len(df_r)), df_r["rhat"].values)
            plt.xticks(np.arange(len(df_r)), df_r["parameter"].values, rotation=45, ha="right")
            plt.axhline(1.01, linestyle="--", linewidth=1)
            plt.title("R-hat by parameter")
            plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # ESS
        # Load chains back
        chain_files = sorted(glob.glob(os.path.join(outdir, "posterior_samples_chain*.csv")))
        if chain_files:
            chains = [pd.read_csv(f).values for f in chain_files]
            # align length
            L = min(x.shape[0] for x in chains)
            chains = np.stack([x[:L, :] for x in chains], axis=0)  # (C, L, D)
            ess = ess_concatenated(chains, max_lag=60)
            df_e = pd.DataFrame({"parameter": PARAM_KEYS, "ess": ess})
            df_e.to_csv(os.path.join(outdir, "posterior_ess.csv"), index=False)
            fig = plt.figure()
            plt.bar(np.arange(len(df_e)), df_e["ess"].values)
            plt.xticks(np.arange(len(df_e)), df_e["parameter"].values, rotation=45, ha="right")
            plt.title("ESS (concatenated, chain-centered)")
            plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

            # Include first few traces and ACFs
            for j in range(min(6, chains.shape[2])):
                # trace fig
                fig = plt.figure()
                for i in range(chains.shape[0]):
                    plt.plot(np.arange(chains.shape[1]), chains[i, :, j], alpha=0.8, label=f"chain {i+1}")
                plt.title(f"Trace: {PARAM_KEYS[j]}")
                plt.xlabel("iteration"); plt.ylabel(PARAM_KEYS[j]); plt.legend()
                plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

                # acf fig
                Y = chains[:, :, j]
                Yc = (Y - Y.mean(axis=1, keepdims=True)).reshape(-1)
                ac = acf_1d(Yc, max_lag=60)
                fig = plt.figure()
                plt.bar(np.arange(len(ac)), ac)
                plt.title(f"ACF: {PARAM_KEYS[j]}")
                plt.xlabel("lag"); plt.ylabel("autocorrelation")
                plt.tight_layout(); pdf.savefig(fig); plt.close(fig)

        # PPC summary
        ppc_path = os.path.join(outdir, "ppc_moments.csv")
        if os.path.exists(ppc_path):
            df_p = pd.read_csv(ppc_path)
            means, xerr, df_plot = _ordered_interval_errors(df_p)
            if not df_plot.empty:
                fig = plt.figure()
                y = np.arange(len(df_plot))
                plt.errorbar(means, y, xerr=xerr, fmt='o', capsize=3)
                plt.yticks(y, df_plot["moment"].values)
                plt.title("Posterior Predictive Moments (mean and 95% CI)")
                plt.tight_layout(); pdf.savefig(fig); plt.close(fig)
    return report_path



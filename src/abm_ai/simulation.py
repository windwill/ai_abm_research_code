"""Economy coordinator for the period-by-period agent based simulation."""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .agents import Firm, Government, TechFirm, Worker
from .parameters import LaborParams, PolicyParams, SimulationParams, TechParams
from .utils import rng

# ---------------------------
class Economy:
    def __init__(self, lp: LaborParams, tp: TechParams, pp: PolicyParams, sp: SimulationParams, outdir: str = "."):
        # Simulations update some policy parameters endogenously during a run.
        # Copy dataclasses here so calibration replications and validation runs
        # do not inherit evolved state from previous Economy instances.
        self.lp=LaborParams(**asdict(lp))
        self.tp=TechParams(**asdict(tp))
        self.pp=PolicyParams(**asdict(pp))
        self.sp=SimulationParams(**asdict(sp))
        self.rnd=rng(self.sp.seed); self.outdir=outdir
        self.workers: List[Worker]=[]; self.firms: List[Firm]=[]; self.techs: List[TechFirm]=[]
        self.gov = Government(G=0.20, tau=pp.tax_rate, S=0.05, B=0.8)
        # Realized tech demand index (lag) used for pricing (scalar and per-supplier)
        self.last_tech_demand: float = float(tp.tech_demand_target)
        self.last_tech_demand_vec: List[float] = []
        self.last_tech_share_hhi: float = 0.0
        self.tech_market_shares: List[float] = []
        self.last_tech_rnd_advantage: float = 0.0
        self.last_tech_markup_advantage: float = 0.0
        self.resource_capacity: float = float(getattr(sp, "resource_capacity_initial", 1.0))
        self.last_resource_demand: float = 0.0
        self.last_resource_constraint: float = 1.0
        self.current_ai_cost: float = 0.0
        self.last_ai_effective_cost: float = 0.0
        self.last_ai_relative_cost_mean: float = 0.0
        self.last_ai_operating_cost: float = 0.0
        self.current_compute_unit_cost: float = 0.0
        self.last_compute_demand: float = 0.0
        self.last_compute_unit_cost: float = 0.0
        self.last_compute_operating_cost: float = 0.0
        self.last_pilot_share: float = 0.0
        self.last_production_share: float = 0.0
        self.last_scaled_share: float = 0.0
        self.last_failed_pilots: int = 0
        self.last_data_readiness_mean: float = 0.0
        self.last_ai_error_events: int = 0
        self.last_ai_operational_loss: float = 0.0
        self.last_ai_error_rate_mean: float = 0.0
        self.last_cyber_privacy_incidents: int = 0
        self.last_cyber_privacy_loss: float = 0.0
        self.last_cyber_privacy_risk_rate_mean: float = 0.0
        self.last_cyber_privacy_control_cost: float = 0.0
        self.last_ai_supervision_vacancies: int = 0
        self.last_ai_complementary_hires: int = 0
        self.last_ai_complementarity_premium: float = 0.0
        self.last_new_task_share: float = 0.0
        self.last_exit_prob_mean: float = 0.0
        self.last_exit_profit_shortfall_mean: float = 0.0
        self.last_exit_below_profit_cutoff_share: float = 0.0
        self.last_exit_below_A_min_share: float = 0.0
        self.last_firm_profit_mean: float = 0.0
        self.last_firm_profit_p10: float = 0.0
        self.last_firm_profit_margin_mean: float = 0.0
        self.last_entry_vacancies: int = 0
        self.last_exit_layoffs: int = 0
        self.last_exit_reallocated_workers: int = 0
        self.last_exit_unabsorbed_workers: int = 0
        # Aggregate macro shock state
        self.macro_z: float = float(sp.macro_mu)
        self.active_productivity_shock: float = 0.0
        self.active_demand_shock: float = 0.0
        self.active_rate_shock: float = 0.0
        self.history: Dict[str, List[float]] = {"MacroZ": [], "Shock_productivity": [], "Shock_demand": [], "Shock_rate": [], "U": [], "Y": [], "W_mean": [], "Adopt_mean": [], "Routine_mean": [], "Debt": [], "G": [], "S": [], "Separations": [], "Hires": [], "Vacancies": [], "Ineq": [], "Ineq_all_workers": [], "Firm_exits": [], "Firm_exit_rate": [], "Exit_prob_mean": [], "Exit_profit_shortfall_mean": [], "Exit_below_profit_cutoff_share": [], "Exit_below_A_min_share": [], "Firm_profit_mean": [], "Firm_profit_p10": [], "Firm_profit_margin_mean": [], "Entry_vacancies": [], "Exit_layoffs": [], "Exit_reallocated_workers": [], "Exit_unabsorbed_workers": [], "eta_G": [], "eta_S": [], "tax_rate": [], "r_debt": [], "Debt_to_GDP": [], "Credit": [], "Tech_demand": [], "Tech_mu": [], "Tech_share_top": [], "Tech_share_hhi": [], "Tech_rnd_advantage": [], "Tech_markup_advantage": [], "Resource_capacity": [], "Resource_demand": [], "Resource_constraint": [], "MA_events": [], "MA_layoffs": [], "AI_cost": [], "AI_relative_cost_mean": [], "AI_operating_cost": [], "Compute_demand": [], "Compute_unit_cost": [], "Compute_operating_cost": [], "Pilot_share": [], "Production_share": [], "Scaled_share": [], "Failed_pilots": [], "Data_readiness_mean": [], "AI_error_events": [], "AI_operational_loss": [], "AI_error_rate_mean": [], "Cyber_privacy_incidents": [], "Cyber_privacy_loss": [], "Cyber_privacy_risk_rate_mean": [], "Cyber_privacy_control_cost": [], "AI_supervision_vacancies": [], "AI_complementary_hires": [], "AI_complementarity_premium": [], "New_task_share": []}
        if getattr(self.sp, "enable_occupation_layer", False):
            for occ_id in self.occupation_ids():
                self.history[f"Occ_U_{occ_id}"] = []
                self.history[f"Occ_W_{occ_id}"] = []
                self.history[f"Occ_Sep_{occ_id}"] = []
        self.last_separations: int = 0
        self.last_hires: int = 0
        self.last_vacancies: int = 0
        self.last_ineq: float = 0.0
        self.last_ineq_all_workers: float = 0.0
        self.last_occ_separations: Dict[str, int] = {}
        self.last_ma_events: int = 0
        self.last_ma_layoffs: int = 0

    def occupation_specs(self) -> list[dict]:
        specs = getattr(self.sp, "occupation_specs", []) or []
        return [s for s in specs if isinstance(s, dict) and s.get("id")]

    def occupation_ids(self) -> list[str]:
        return [str(s["id"]) for s in self.occupation_specs()]

    def occupation_spec_map(self) -> Dict[str, dict]:
        return {str(s["id"]): s for s in self.occupation_specs()}

    def _draw_occupation(self) -> dict:
        specs = self.occupation_specs()
        if not specs:
            return {"id": "general", "skill_mean": 0.55, "skill_sd": 0.15, "wage_multiplier": 1.0, "skill_vector": {}}
        # Appendix B, Eq. (B.1): draw o_i using occupation shares pi_o.
        weights = np.array([max(0.0, float(s.get("share", 0.0))) for s in specs], dtype=float)
        if weights.sum() <= 0:
            weights = np.ones(len(specs), dtype=float)
        probs = weights / weights.sum()
        return specs[int(self.rnd.choice(np.arange(len(specs)), p=probs))]

    def _occupation_skill_vector(self, spec: dict, scalar_skill: float) -> Dict[str, float]:
        """Draw the five skill components around mu_o in Eqs. (B.1) and (B.3)."""
        vec = spec.get("skill_vector", {}) if isinstance(spec, dict) else {}
        sd = max(0.0, float(spec.get("skill_sd", 0.10))) if isinstance(spec, dict) else 0.10
        out = {}
        for key in ("cog", "soc", "tech", "man", "manual"):
            center = float(vec.get(key, scalar_skill)) if isinstance(vec, dict) else scalar_skill
            out[key] = float(np.clip(self.rnd.normal(center, sd * 0.5), 0.0, 1.0))
        return out

    def _firm_required_skill_vector(self, req: float, routine: float) -> Dict[str, float]:
        """Construct the firm requirement vector q_j in Appendix B, Eq. (B.4)."""
        return {
            "cog": float(np.clip(req + 0.10 * (1.0 - routine), 0.0, 1.0)),
            "soc": float(np.clip(req + 0.05 * (1.0 - routine), 0.0, 1.0)),
            "tech": float(np.clip(req + 0.12 * routine, 0.0, 1.0)),
            "man": float(np.clip(req + 0.08 * (1.0 - routine), 0.0, 1.0)),
            "manual": float(np.clip(req + 0.10 * routine - 0.05, 0.0, 1.0)),
        }

    def _occupation_spec_for_worker(self, worker: Worker) -> dict:
        return self.occupation_spec_map().get(getattr(worker, "occupation", "general"), {})

    def _occupation_stats(self) -> Dict[str, tuple[float, float]]:
        stats: Dict[str, tuple[float, float]] = {}
        for occ_id in self.occupation_ids():
            group = [w for w in self.workers if getattr(w, "occupation", "") == occ_id]
            if not group:
                stats[occ_id] = (0.0, 0.0)
                continue
            # Appendix B, Eq. (B.8): occupation-specific unemployment rate U_o,t.
            u_rate = sum(1 for w in group if not w.employed) / len(group)
            # Appendix B, Eq. (B.9): mean realized wage among employed workers.
            wages = [w.w for w in group if w.employed and w.firm_id is not None]
            w_mean = float(np.mean(wages)) if wages else 0.0
            stats[occ_id] = (float(u_rate), w_mean)
        return stats

    @staticmethod
    def gini_nonnegative(values: list[float] | np.ndarray) -> float:
        """Return the Gini coefficient for a nonnegative vector.

        This implements the Appendix A inequality statistic. The caller chooses
        the income concept: employed-worker wages for `Ineq`, or all-worker
        current labor income, with unemployed workers assigned zero income, for
        `Ineq_all_workers`.
        """
        arr = np.sort(np.asarray(values, dtype=float))
        arr = np.clip(arr[np.isfinite(arr)], 0.0, None)
        if len(arr) < 2:
            return 0.0
        total = float(arr.sum())
        if total <= 1e-12:
            return 0.0
        idx = np.arange(1, len(arr) + 1, dtype=float)
        gini = (2.0 * float((idx * arr).sum())) / (len(arr) * total) - (len(arr) + 1.0) / len(arr)
        return float(np.clip(gini, 0.0, 1.0))

    def worker_firm_compatibility(self, worker: Worker, firm: Firm) -> float:
        """Return scalar and occupation-sensitive compatibility from Eq. (B.5)."""
        scalar_fit = math.exp(-self.lp.compat_psi * abs(float(worker.skill) - float(firm.req_skill)))
        if not getattr(self.sp, "enable_occupation_layer", False):
            return float(scalar_fit)
        vec_dist = float(np.linalg.norm(worker.skill_vector - firm.req_skill_vector) / math.sqrt(5.0))
        psi_o = float(self.sp.occupation_match_weight) * float(self.lp.compat_psi)
        occupation_fit = math.exp(-psi_o * vec_dist)
        return float(scalar_fit * occupation_fit)

    def effective_training_cost(self, w: Worker) -> float:
        """Compute effective training cost for a worker given government subsidy S.

        Training subsidy channel.

        The baseline training cost is lp.C_T. The government subsidy intensity S_t
        is mapped to a bounded fraction in [0,1) via S_t/(S_t+1). The subsidy reduces
        the effective training cost for eligible workers, subject to a floor.
        Optionally, the subsidy can be targeted by unemployment duration.
        """
        base = float(self.lp.C_T)
        if not getattr(self.sp, "enable_training_subsidy", True):
            return base
        # Only unemployed workers are eligible in the implemented policy channel
        if getattr(w, "employed", False):
            return base
        S = float(getattr(self.gov, "S", 0.0))
        # Eq. (A.4): Subsidy normalization
        s_tilde = S / (S + 1.0) if S > 0.0 else 0.0
        weight = 1.0
        if getattr(self.sp, "training_subsidy_use_duration", False):
            scale = max(1e-6, float(getattr(self.sp, "training_subsidy_dur_scale", 20.0)))
            # Eq. (A.6): Duration-targeted eligibility weight
            weight = min(1.0, float(getattr(w, "u_dur", 0)) / scale)
        eta_T = float(getattr(self.sp, "training_subsidy_effect", 0.60))
        floor_ratio = float(getattr(self.sp, "training_cost_floor_ratio", 0.05))
        floor = max(0.0, floor_ratio * base)
        # Eq. (A.5): Effective training cost with floor
        C_eff = base * (1.0 - eta_T * s_tilde * weight)
        return float(max(floor, C_eff))
        
    def update_macro_shock(self) -> float:
        """AR(1) macro shock process.

        z_t follows: z_t = mu + rho (z_{t-1} - mu) + eps, eps ~ N(0, sigma^2),
        clipped to [macro_min, macro_max] for numerical stability. The shock scales
        realized output/revenue in the production block.
        """
        sp = self.sp
        if not getattr(sp, "enable_macro_shocks", True):
            self.macro_z = float(sp.macro_mu)
            return self.macro_z
        # Eq. (A.64) macro shock update
        eps = float(self.rnd.normal(0.0, float(sp.macro_sigma)))
        z = float(sp.macro_mu) + float(sp.macro_rho) * (float(self.macro_z) - float(sp.macro_mu)) + eps
        self.macro_z = float(max(float(sp.macro_min), min(float(sp.macro_max), z)))
        return self.macro_z

    def apply_scheduled_shocks(self, t: int) -> None:
        """Resolve configured scenario shocks active in period t.

        Entries in `SimulationParams.shock_schedule` are dictionaries with
        `type`, `start`, `duration`, and `magnitude`. Supported types are:
        productivity, demand, and rate. Magnitudes are additive wedges, so a
        demand magnitude of -0.10 reduces realized output by 10 percent in the
        active period.
        """
        self.active_productivity_shock = 0.0
        self.active_demand_shock = 0.0
        self.active_rate_shock = 0.0
        # Appendix B, Eq. (B.56): each mapping is one schedule element
        # (k_m, t_m, d_m, xi_m).
        for item in getattr(self.sp, "shock_schedule", []) or []:
            if not isinstance(item, dict):
                raise ValueError("Each scheduled shock must be a mapping")
            start = int(item.get("start", 0))
            duration = max(1, int(item.get("duration", 1)))
            if int(t) < start or int(t) >= start + duration:
                continue
            magnitude = float(item.get("magnitude", 0.0))
            shock_type = str(item.get("type", "")).strip().lower()
            # Appendix B, Eq. (B.57): sum all active wedges by shock type.
            if shock_type in {"productivity", "productivity_shock", "tech"}:
                self.active_productivity_shock += magnitude
            elif shock_type in {"demand", "demand_shock"}:
                self.active_demand_shock += magnitude
            elif shock_type in {"rate", "credit", "credit_tightening", "interest_rate"}:
                self.active_rate_shock += magnitude
            else:
                raise ValueError(f"Unknown scheduled shock type: {shock_type}")
        if self.active_rate_shock:
            # Appendix B, Eq. (B.60): apply the additive rate wedge
            # immediately so current-period credit and fiscal channels see it.
            self.pp.r_debt = float(np.clip(
                max(float(self.pp.r_debt), float(getattr(self.sp, "r_base", 0.02))) + self.active_rate_shock,
                float(getattr(self.sp, "r_min", 0.0)),
                float(getattr(self.sp, "r_max", 0.10)),
            ))


    def initialize_agents(self):
        s = np.clip(self.rnd.beta(2,2,size=self.sp.N_workers),0,1)
        a = np.clip(self.rnd.beta(1.5,2,size=self.sp.N_workers),0,1)
        w_r = self.lp.w0 * (0.6 + 0.8*s)
        employed = self.rnd.random(self.sp.N_workers) < 0.9
        initial_firm_size_targets: Dict[int, int] = {}
        for i in range(self.sp.N_workers):
            firm_id = int(self.rnd.integers(0,self.sp.N_firms)) if employed[i] else None
            if getattr(self.sp, "enable_occupation_layer", False):
                occ = self._draw_occupation()
                skill_mean = float(occ.get("skill_mean", s[i]))
                skill_sd = max(1e-6, float(occ.get("skill_sd", 0.10)))
                # Appendix B, Eq. (B.2): occupation-specific clipped skill draw.
                s_i = float(np.clip(self.rnd.normal(skill_mean, skill_sd), 0.0, 1.0))
                wage_mult = float(occ.get("wage_multiplier", 1.0))
                w_r_i = self.lp.w0 * wage_mult * (0.6 + 0.8 * s_i)
                skill_vec = self._occupation_skill_vector(occ, s_i)
                occupation = str(occ.get("id", "general"))
            else:
                s_i = float(s[i])
                w_r_i = float(w_r[i])
                skill_vec = None
                occupation = "general"
            self.workers.append(Worker(s_i, a[i], w_r_i, bool(employed[i]), firm_id, u_dur=(0 if employed[i] else 1), occupation=occupation, skill_vector=skill_vec))
        for j in range(self.sp.N_firms):
            A0=float(self.rnd.uniform(0.5,1.5)); L0=int(max(5,self.rnd.poisson(20)))
            initial_firm_size_targets[j] = L0
            kappa0=float(self.rnd.lognormal(mean=0.0, sigma=0.25)); rho0=float(np.clip(self.rnd.normal(0.08,0.03),0,0.3))
            req=float(np.clip(self.rnd.normal(self.sp.firm_req_skill_mean, self.sp.firm_req_skill_sd),0,1))
            routine=float(np.clip(self.rnd.normal(self.sp.firm_routine_mean, self.sp.firm_routine_sd),0,1))
            req_vec = self._firm_required_skill_vector(req, routine) if getattr(self.sp, "enable_occupation_layer", False) else None
            data_ready = float(np.clip(self.rnd.normal(getattr(self.sp, "data_readiness_mean", 0.65), getattr(self.sp, "data_readiness_sd", 0.15)), 0.0, 1.0))
            firm = Firm(j,A0,L0,kappa0,rho0,req,routine,req_vec,data_ready)
            if getattr(self.sp, "enable_deployment_friction", False):
                firm.deployment_stage = 0
                firm.rho_adopt = 0.0
            self.firms.append(firm)
        self.sync_firm_employment_counts()
        for f in self.firms:
            f.desired_L = max(int(f.L), int(initial_firm_size_targets.get(int(f.id), f.L)))
        for k in range(self.sp.N_tech):
            pi0=float(self.rnd.lognormal(mean=0.0, sigma=0.1)); price0=float(self.rnd.uniform(0.8,1.2)); rho_R0=float(self.rnd.uniform(0.05,0.15))
            self.techs.append(TechFirm(pi0, price0, rho_R0, self.tp.mu_markup))

        # initialize per-supplier lagged demand (used for demand-sensitive pricing)
        self.last_tech_demand_vec = [float(self.tp.tech_demand_target) for _ in range(len(self.techs))]
        self.tech_market_shares = [1.0 / max(1, len(self.techs)) for _ in range(len(self.techs))]

    def sync_firm_employment_counts(self) -> None:
        """Synchronize firm employment stocks with actual worker assignments.

        Firm.L is the employment stock L_j,t used by the labor, production, cost,
        and restructuring equations. Worker assignments are the source of truth
        after initialization, layoffs, exits, and M&A transfers.
        """
        counts = {int(f.id): 0 for f in self.firms}
        valid_ids = set(counts)
        for w in self.workers:
            if w.employed and w.firm_id in valid_ids:
                counts[int(w.firm_id)] += 1
        for f in self.firms:
            f.L = int(counts.get(int(f.id), 0))

    def normalized_tech_market_shares(self) -> np.ndarray:
        """Return lagged smoothed shares s_k,t-1 used in Eqs. (B.29)-(B.34)."""
        K = len(self.techs)
        if K <= 0:
            return np.array([], dtype=float)
        floor = max(0.0, float(getattr(self.tp, "tech_share_floor", 1e-6)))
        shares = np.asarray(self.tech_market_shares if self.tech_market_shares else [1.0 / K] * K, dtype=float)
        if len(shares) != K:
            shares = np.ones(K, dtype=float) / float(K)
        shares = np.clip(shares, floor, None)
        total = float(np.sum(shares))
        if total <= 0.0:
            return np.ones(K, dtype=float) / float(K)
        return shares / total

    def update_tech_market_shares(self, current_shares: np.ndarray) -> None:
        """Apply the supplier-share memory equation in Appendix B, Eq. (B.31)."""
        K = len(self.techs)
        if K <= 0:
            self.tech_market_shares = []
            return
        cur = np.asarray(current_shares, dtype=float)
        if len(cur) != K or float(np.sum(cur)) <= 0.0:
            cur = np.ones(K, dtype=float) / float(K)
        cur = cur / max(1e-12, float(np.sum(cur)))
        prev = self.normalized_tech_market_shares()
        memory = float(np.clip(getattr(self.tp, "tech_share_memory", 0.70), 0.0, 1.0))
        # Appendix B, Eq. (B.31): s_k,t = omega_s s_k,t-1
        # + (1 - omega_s) tilde_s_k,t.
        smoothed = memory * prev + (1.0 - memory) * cur
        smoothed = smoothed / max(1e-12, float(np.sum(smoothed)))
        self.tech_market_shares = [float(x) for x in smoothed]

    
    def update_interest_rate(self, Y: float):
        """Endogenous government borrowing rate as a function of debt-to-GDP.

        The borrowing rate used in debt dynamics is updated each period using a simple spread rule:
            debt_ratio = B / max(eps, Y)
            r_t = clip(r_base + r_spread_slope * (debt_ratio - debt_gdp_target), r_min, r_max)

        This provides a parsimonious finance block that links fiscal stress to borrowing costs.
        """
        if not getattr(self.sp, "enable_endogenous_rates", False):
            return
        # Eq. (A.45) debt ratio definition. B is maintained as a debt-to-output
        # ratio in the fiscal block, which avoids mixing report-scale population
        # counts with ratio-style calibration targets.
        debt_ratio = float(self.gov.B)
        # Eq. (A.46): Endogenous borrowing rate (debt-dependent spread)
        r_t = float(self.sp.r_base + self.sp.r_spread_slope * (debt_ratio - self.sp.debt_gdp_target))
        r_t = float(np.clip(r_t, self.sp.r_min, self.sp.r_max))
        if getattr(self, "active_rate_shock", 0.0):
            r_t = max(r_t, float(getattr(self.sp, "r_base", 0.02)))
        # Appendix B, Eq. (B.60): the active additive rate wedge is applied
        # to the freshly calculated endogenous baseline rate and then clipped.
        r_t = float(np.clip(r_t + float(getattr(self, "active_rate_shock", 0.0)), self.sp.r_min, self.sp.r_max))
        self.pp.r_debt = r_t
        # store for diagnostics
        self._last_debt_ratio = debt_ratio

    def gamma_I_effective(self) -> float:
        """Make firm investment responsiveness decrease with borrowing costs.

        We use an exponential damping around the baseline rate:
            gamma_I = inv_gamma_base * exp(-inv_rate_sensitivity * max(0, r_debt - r_base))
        """
        # Eq. (A.47) effective investment responsiveness
        base = float(getattr(self.sp, "inv_gamma_base", 0.5))
        if not getattr(self.sp, "enable_endogenous_rates", False):
            return base
        r0 = float(getattr(self.sp, "r_base", 0.02))
        r = float(self.pp.r_debt)
        phi = float(getattr(self.sp, "inv_rate_sensitivity", 8.0))
        damp = math.exp(-phi * max(0.0, r - r0))
        return float(base * damp)

    def credit_factor(self) -> float:
        """Financial tightening index in (0, 1].

        Map borrowing costs and debt stress into a smooth credit
        availability factor that can damp hiring and adoption.

        The factor decreases when r_debt exceeds the baseline rate and when the
        debt-to-GDP ratio exceeds its target. Bounds ensure numerical stability.
        """
        if (not getattr(self.sp, "enable_endogenous_rates", True)) or (not getattr(self.sp, "enable_credit_channel", True)):
            return 1.0
        r_spread = max(0.0, float(self.pp.r_debt) - float(self.sp.r_base))
        debt_gap = max(0.0, float(getattr(self, "_last_debt_ratio", 0.0)) - float(self.sp.debt_gdp_target))
        # Eq. (A.48): Credit conditions factor
        base = math.exp(-float(self.sp.credit_rate_sensitivity) * r_spread) * math.exp(-float(self.sp.credit_debt_sensitivity) * debt_gap)
        return float(np.clip(base, float(self.sp.credit_min), 1.0))

    def update_resource_capacity(self) -> None:
        """Update the AI-supporting capacity index used in constrained diffusion.

        The capacity state is a reduced-form index for compute, data-center,
        power, cooling, and related infrastructure availability. It is not a
        legal or regulatory variable.
        """
        if not getattr(self.sp, "enable_resource_constraint", False):
            self.resource_capacity = float(getattr(self.sp, "resource_capacity_initial", 1.0))
            self.last_resource_constraint = 1.0
            self.last_resource_demand = 0.0
            return
        growth = float(getattr(self.sp, "resource_capacity_growth", 0.02))
        cap_min = max(1e-12, float(getattr(self.sp, "resource_capacity_min", 1e-6)))
        # Appendix B, Eq. (B.10): C_(t+1) = max(C_min, (1 + g_C) C_t).
        self.resource_capacity = float(max(cap_min, self.resource_capacity * (1.0 + growth)))

    def resource_constraint_factor(self, desired_resource_demand: float) -> float:
        """Return the bounded capacity factor in Appendix B, Eq. (B.12)."""
        demand = max(0.0, float(desired_resource_demand))
        self.last_resource_demand = demand
        if (not getattr(self.sp, "enable_resource_constraint", False)) or demand <= 1e-12:
            # Appendix B, Eq. (B.12): kappa_C,t = 1 when D_t = 0
            # or when the optional resource extension is disabled.
            self.last_resource_constraint = 1.0
            return 1.0
        floor = float(np.clip(getattr(self.sp, "resource_constraint_floor", 0.05), 0.0, 1.0))
        factor = float(self.resource_capacity) / demand
        # Appendix B, Eq. (B.12): clip(C_t / D_t, kappa_C,min, 1).
        self.last_resource_constraint = float(np.clip(factor, floor, 1.0))
        return self.last_resource_constraint

    def update_ai_cost(self, t: int) -> None:
        """Update the effective AI automation cost index for period t."""
        if not getattr(self.sp, "enable_ai_relative_cost", False):
            self.current_ai_cost = 0.0
            self.last_ai_effective_cost = 0.0
            self.last_ai_relative_cost_mean = 0.0
            self.last_ai_operating_cost = 0.0
            return
        base = float(getattr(self.sp, "ai_cost_base", 1.0))
        decline = float(getattr(self.sp, "ai_cost_decline", 0.02))
        floor = float(getattr(self.sp, "ai_cost_floor", 0.05))
        fixed = float(getattr(self.sp, "ai_cost_implementation", 0.0)) + float(getattr(self.sp, "ai_cost_monitoring", 0.0))
        # Appendix B, Eq. (B.14): baseline effective AI cost path.
        self.current_ai_cost = float(max(floor, base * math.exp(-decline * max(0, int(t))) + fixed))
        self.last_ai_effective_cost = self.current_ai_cost

    def update_compute_unit_cost(self, t: int) -> None:
        """Update the unit cost of AI computation for training/inference workloads."""
        if not getattr(self.sp, "enable_compute_cost", False):
            self.current_compute_unit_cost = 0.0
            self.last_compute_unit_cost = 0.0
            self.last_compute_demand = 0.0
            self.last_compute_operating_cost = 0.0
            return
        base = float(getattr(self.sp, "compute_unit_cost_base", 0.25))
        decline = float(getattr(self.sp, "compute_unit_cost_decline", 0.02))
        floor = float(getattr(self.sp, "compute_unit_cost_floor", 0.02))
        # Appendix B, Eq. (B.20): baseline compute unit-cost path.
        self.current_compute_unit_cost = float(max(floor, base * math.exp(-decline * max(0, int(t)))))
        self.last_compute_unit_cost = self.current_compute_unit_cost

    def effective_compute_unit_cost(self, compute_demand: float | None = None) -> float:
        """Return computation unit cost including optional resource-scarcity markup."""
        if not getattr(self.sp, "enable_compute_cost", False):
            return 0.0
        cost = float(self.current_compute_unit_cost)
        sensitivity = float(getattr(self.sp, "compute_resource_sensitivity", 0.0))
        if sensitivity > 0.0 and compute_demand is not None:
            capacity = max(float(getattr(self, "resource_capacity", 1.0)), float(getattr(self.sp, "resource_capacity_min", 1e-6)))
            scarcity = max(0.0, float(compute_demand) / capacity - 1.0)
            # Appendix B, Eq. (B.21): firm-specific resource-sensitive
            # compute unit cost because scarcity depends on D_cmp,j,t.
            cost *= (1.0 + sensitivity * scarcity)
        self.last_compute_unit_cost = float(max(float(getattr(self.sp, "compute_unit_cost_floor", 0.02)), cost))
        return self.last_compute_unit_cost

    def firm_compute_demand(self, f: Firm, *, include_training: bool = True) -> float:
        """Return reduced-form compute workload for a firm."""
        if not getattr(self.sp, "enable_compute_cost", False):
            return 0.0
        stage = int(getattr(f, "deployment_stage", 2))
        if getattr(self.sp, "enable_deployment_friction", False):
            stage_factor = 0.0 if stage == 0 else (0.35 if stage == 1 else (1.0 if stage == 2 else 1.25))
        else:
            stage_factor = 1.0
        training_share = float(getattr(self.sp, "compute_training_share", 0.20)) if include_training else 0.0
        inference_scale = float(getattr(self.sp, "compute_inference_scale", 1.0))
        intensity = float(getattr(self.sp, "compute_intensity_per_adoption", 1.0))
        # Appendix B, Eq. (B.19): firm-level training and inference workload.
        workload = intensity * float(f.rho_adopt) * max(0.0, float(f.L)) * stage_factor
        return float(max(0.0, workload * (inference_scale + training_share * max(0.0, 1.0 - float(f.rho_adopt)))))

    def effective_ai_cost(self, desired_resource_demand: float | None = None) -> float:
        """Return AI cost including optional resource-scarcity markup."""
        if not getattr(self.sp, "enable_ai_relative_cost", False):
            return 0.0
        if desired_resource_demand is None:
            return float(max(float(getattr(self.sp, "ai_cost_floor", 0.05)), self.last_ai_effective_cost))
        cost = float(self.current_ai_cost)
        sensitivity = float(getattr(self.sp, "ai_cost_resource_sensitivity", 0.0))
        if sensitivity > 0.0 and desired_resource_demand is not None:
            capacity = max(float(getattr(self, "resource_capacity", 1.0)), float(getattr(self.sp, "resource_capacity_min", 1e-6)))
            scarcity = max(0.0, float(desired_resource_demand) / capacity - 1.0)
            # Appendix B, Eq. (B.15): scarcity markup on baseline AI cost.
            cost *= 1.0 + sensitivity * scarcity
        self.last_ai_effective_cost = float(max(float(getattr(self.sp, "ai_cost_floor", 0.05)), cost))
        return self.last_ai_effective_cost

    def firm_mean_wage(self, firm_id: int) -> float:
        wages = [w.w for w in self.workers if w.employed and w.firm_id == firm_id and w.w > 0.0]
        if wages:
            return float(np.mean(wages))
        employed_wages = [w.w for w in self.workers if w.employed and w.w > 0.0]
        return float(np.mean(employed_wages)) if employed_wages else float(self.lp.w0)

    def firm_ai_cost_multiplier(self, firm_id: int) -> float:
        multipliers = getattr(self.sp, "occupation_ai_cost_multipliers", {}) or {}
        if not getattr(self.sp, "enable_occupation_layer", False) or not isinstance(multipliers, dict) or not multipliers:
            return 1.0
        firm_workers = [w for w in self.workers if w.employed and w.firm_id == firm_id]
        if not firm_workers:
            return 1.0
        vals = [float(multipliers.get(getattr(w, "occupation", "general"), 1.0)) for w in firm_workers]
        return float(np.mean(vals)) if vals else 1.0

    def update_deployment_stage(self, f: Firm, adoption_hazard: float) -> tuple[float, bool]:
        """Update z_j,t and return the deployment factor in Eqs. (B.24)-(B.27)."""
        if not getattr(self.sp, "enable_deployment_friction", False):
            f.deployment_stage = 3 if f.rho_adopt >= float(getattr(self.sp, "scaled_deployment_threshold", 0.75)) else (2 if f.rho_adopt > 0.0 else 0)
            return 1.0, False

        failed = False
        data = float(np.clip(getattr(f, "data_readiness", 1.0), 0.0, 1.0))
        stage = int(getattr(f, "deployment_stage", 0))
        # Appendix B, Eq. (B.24): probability of moving from no deployment
        # to a pilot, using the adjusted firm adoption hazard.
        if stage == 0 and self.rnd.random() < float(np.clip(getattr(self.sp, "pilot_start_scale", 0.60) * adoption_hazard * data, 0.0, 1.0)):
            stage = 1
        elif stage == 1:
            # Appendix B, Eq. (B.25): pilot-to-production probability.
            # f.profit is the most recently realized profit under the model's
            # within-period order, so it is lagged information at this stage.
            p_success = float(getattr(self.sp, "pilot_success_base", 0.25))
            p_success += float(getattr(self.sp, "pilot_success_data_weight", 0.55)) * data
            p_success += float(getattr(self.sp, "pilot_success_profit_weight", 0.20)) * max(0.0, float(f.profit))
            p_success = float(np.clip(p_success, 0.0, 1.0))
            # Appendix B, Eq. (B.26): pilot failure probability.
            p_fail = float(getattr(self.sp, "pilot_failure_base", 0.08))
            p_fail += float(getattr(self.sp, "pilot_failure_data_weight", 0.30)) * (1.0 - data)
            p_fail = float(np.clip(p_fail, 0.0, 1.0))
            draw = float(self.rnd.random())
            if draw < p_success:
                stage = 2
            elif draw < p_success + p_fail:
                stage = 0
                f.rho_adopt = max(0.0, float(f.rho_adopt) * 0.50)
                failed = True
        elif stage == 2 and f.rho_adopt >= float(getattr(self.sp, "scaled_deployment_threshold", 0.75)):
            stage = 3

        f.deployment_stage = int(stage)
        # Appendix B, Eq. (B.27): stage-dependent productivity deployment factor.
        if stage == 0:
            return 0.0, failed
        if stage == 1:
            return 0.20 * data, failed
        if stage == 2:
            return float(np.clip(data + float(getattr(self.sp, "production_scale_rate", 0.20)) * float(f.rho_adopt), 0.0, 1.0)), failed
        return 1.0, failed

    def deployment_stats(self) -> None:
        if not self.firms:
            self.last_pilot_share = self.last_production_share = self.last_scaled_share = 0.0
            self.last_data_readiness_mean = 0.0
            return
        n = float(len(self.firms))
        stages = [int(getattr(f, "deployment_stage", 0)) for f in self.firms]
        self.last_pilot_share = float(sum(1 for s in stages if s == 1) / n)
        self.last_production_share = float(sum(1 for s in stages if s >= 2) / n)
        self.last_scaled_share = float(sum(1 for s in stages if s >= 3) / n)
        self.last_data_readiness_mean = float(np.mean([float(getattr(f, "data_readiness", 1.0)) for f in self.firms]))

    def ai_operational_loss(self, f: Firm, output: float) -> tuple[float, bool, float]:
        """Draw an optional AI error event and return its operational loss."""
        if not getattr(self.sp, "enable_ai_operational_risk", False):
            return 0.0, False, 0.0
        if float(getattr(f, "rho_adopt", 0.0)) <= 0.0:
            return 0.0, False, 0.0
        data = float(np.clip(getattr(f, "data_readiness", 1.0), 0.0, 1.0))
        stage = int(getattr(f, "deployment_stage", 0))
        # Appendix B, Eq. (B.39): AI-error deployment-stage risk factor
        # psi_err^z = (0, 0.7, 1.0, 0.8) for stages z = (0, 1, 2, 3).
        stage_factor = 0.0 if stage == 0 else (0.7 if stage == 1 else (1.0 if stage == 2 else 0.8))
        # Appendix B, Eq. (B.39): bounded AI-error event probability.
        p_error = float(getattr(self.sp, "ai_error_base", 0.005))
        p_error += float(getattr(self.sp, "ai_error_data_sensitivity", 0.05)) * (1.0 - data)
        p_error += float(getattr(self.sp, "ai_error_stage_sensitivity", 0.02)) * stage_factor
        p_error += float(getattr(self.sp, "ai_error_adoption_sensitivity", 0.03)) * float(f.rho_adopt)
        p_error = float(np.clip(p_error, 0.0, 1.0))
        # Appendix B, Eq. (B.40): Bernoulli draw for e_j,t.
        if self.rnd.random() >= p_error:
            return 0.0, False, p_error
        severity = float(getattr(self.sp, "ai_loss_severity", 0.03))
        noise = float(getattr(self.sp, "ai_loss_noise", 0.0))
        if noise > 0.0:
            # epsilon_j,t in Eq. (B.41) is lognormal with log-mean zero.
            severity *= float(self.rnd.lognormal(mean=0.0, sigma=noise))
        cap = max(0.0, float(getattr(self.sp, "ai_loss_cap_share", 0.25))) * max(0.0, float(output))
        # Appendix B, Eq. (B.41): capped AI operational loss.
        loss = min(cap, max(0.0, severity * stage_factor * float(f.rho_adopt) * max(0.0, float(output))))
        return float(loss), True, p_error

    def cyber_privacy_risk_rate(self, f: Firm) -> float:
        """Return the period incident probability for cyber/privacy operational risk."""
        if not getattr(self.sp, "enable_cyber_privacy_risk", False):
            return 0.0
        if float(getattr(f, "rho_adopt", 0.0)) <= 0.0:
            return 0.0
        data = float(np.clip(getattr(f, "data_readiness", 1.0), 0.0, 1.0))
        stage = int(getattr(f, "deployment_stage", 0))
        # Appendix B, Eq. (B.42): cyber deployment-stage risk factor
        # psi_cyb^z = (0, 0.5, 1.0, 1.2) for stages z = (0, 1, 2, 3).
        stage_factor = 0.0 if stage == 0 else (0.5 if stage == 1 else (1.0 if stage == 2 else 1.2))
        control = float(np.clip(getattr(self.sp, "cyber_control_strength", 0.50), 0.0, 1.0))
        concentration = float(np.clip(getattr(self, "last_tech_share_hhi", 0.0), 0.0, 1.0))
        p_incident = float(getattr(self.sp, "cyber_incident_base", 0.002))
        p_incident += float(getattr(self.sp, "cyber_incident_data_sensitivity", 0.040)) * (1.0 - data)
        p_incident += float(getattr(self.sp, "cyber_incident_adoption_sensitivity", 0.030)) * float(f.rho_adopt)
        p_incident += float(getattr(self.sp, "cyber_incident_stage_sensitivity", 0.010)) * stage_factor
        p_incident += float(getattr(self.sp, "cyber_incident_concentration_sensitivity", 0.000)) * concentration
        # Appendix B, Eq. (B.42): controls proportionally reduce incident risk.
        p_incident *= (1.0 - control)
        return float(np.clip(p_incident, 0.0, 1.0))

    def cyber_privacy_control_cost(self, f: Firm) -> float:
        """Return recurring cyber/privacy control cost for an adopting firm."""
        if not getattr(self.sp, "enable_cyber_privacy_risk", False):
            return 0.0
        control = float(np.clip(getattr(self.sp, "cyber_control_strength", 0.50), 0.0, 1.0))
        scale = float(getattr(self.sp, "cyber_control_cost_scale", 0.010))
        deployment = 1.0
        if getattr(self.sp, "enable_deployment_friction", False):
            stage = int(getattr(f, "deployment_stage", 0))
            deployment = 0.0 if stage == 0 else (0.5 if stage == 1 else 1.0)
        # Appendix B, Eq. (B.44): recurring cyber/privacy control cost.
        return float(max(0.0, scale * control * deployment * float(f.rho_adopt) * max(0.0, float(f.L))))

    def cyber_privacy_operational_loss(self, f: Firm, output: float) -> tuple[float, bool, float]:
        """Draw an optional cyber/privacy incident and return its operational loss."""
        p_incident = self.cyber_privacy_risk_rate(f)
        # Appendix B, Eq. (B.43): Bernoulli draw for c_j,t.
        if p_incident <= 0.0 or self.rnd.random() >= p_incident:
            return 0.0, False, p_incident
        severity = float(getattr(self.sp, "cyber_loss_severity", 0.040))
        noise = float(getattr(self.sp, "cyber_loss_noise", 0.0))
        if noise > 0.0:
            # epsilon_cyb,j,t in Eq. (B.45) is lognormal with log-mean zero.
            severity *= float(self.rnd.lognormal(mean=0.0, sigma=noise))
        cap = max(0.0, float(getattr(self.sp, "cyber_loss_cap_share", 0.30))) * max(0.0, float(output))
        # Appendix B, Eq. (B.45): capped cyber/privacy incident loss.
        loss = min(cap, max(0.0, severity * float(f.rho_adopt) * max(0.0, float(output))))
        return float(loss), True, p_incident

    def worker_complementarity_score(self, w: Worker) -> float:
        """Return the bounded worker complementarity score in Eq. (B.51)."""
        adapt_w = float(getattr(self.sp, "ai_complementarity_adapt_weight", 0.60))
        skill_w = float(getattr(self.sp, "ai_complementarity_skill_weight", 0.40))
        denom = max(1e-9, adapt_w + skill_w)
        return float(np.clip((adapt_w * float(w.a) + skill_w * float(w.skill)) / denom, 0.0, 1.0))

    def complementarity_match_multiplier(self, f: Firm, w: Worker) -> float:
        """Return the multiplicative matching adjustment in Eq. (B.52)."""
        if not getattr(self.sp, "enable_ai_complementarity", False):
            return 1.0
        beta_c = float(getattr(self.sp, "ai_complementarity_match_bonus", 0.50))
        return float(1.0 + beta_c * self.firm_new_task_intensity(f) * self.worker_complementarity_score(w))

    def complementarity_wage_premium_rate(self, w: Worker) -> float:
        """Return the proportional complementary-role wage premium in Eq. (B.53)."""
        omega_c = float(getattr(self.sp, "ai_complementarity_wage_premium", 0.05))
        return float(omega_c * self.worker_complementarity_score(w))

    def ai_supervision_vacancies(self, f: Firm) -> int:
        """Return the additional AI-supervision vacancies in Eq. (B.50)."""
        if not getattr(self.sp, "enable_ai_complementarity", False):
            return 0
        lambda_n = float(getattr(self.sp, "ai_supervision_demand_scale", 0.10))
        return int(round(lambda_n * self.firm_new_task_intensity(f) * max(1.0, float(f.L))))

    def firm_new_task_intensity(self, f: Firm) -> float:
        """Return firm-level complementary new-task intensity from Eq. (B.49)."""
        if not getattr(self.sp, "enable_ai_complementarity", False):
            return 0.0
        threshold = float(getattr(self.sp, "ai_new_task_threshold", 0.30))
        deploy = 1.0 if not getattr(self.sp, "enable_deployment_friction", False) else (
            0.0 if int(getattr(f, "deployment_stage", 0)) == 0 else
            0.25 if int(getattr(f, "deployment_stage", 0)) == 1 else
            0.75 if int(getattr(f, "deployment_stage", 0)) == 2 else 1.0
        )
        adoption_term = max(0.0, float(f.rho_adopt) - threshold) / max(1e-9, 1.0 - threshold)
        data = float(getattr(f, "data_readiness", 1.0))
        data_weight = float(np.clip(getattr(self.sp, "ai_new_task_data_weight", 0.50), 0.0, 1.0))
        readiness = (1.0 - data_weight) + data_weight * data
        # Appendix B, Eq. (B.49): adoption gap, deployment, and readiness
        # jointly determine the bounded new-task intensity.
        return float(np.clip(adoption_term * deploy * readiness, 0.0, 1.0))



    def labor_market(self) -> Tuple[int,float,float]:
        # Effective vacancies may tighten under credit conditions
        credit_hire = self.credit_factor() ** self.sp.credit_hiring_power
        firm_vac_eff: List[Tuple[int,int]] = []
        complementary_slots: Dict[int, int] = {}
        V_t = 0
        for fid, f in enumerate(self.firms):
            # Eq. (A.49): raw vacancies
            vac_raw = max(0, f.desired_L - f.L)
            if getattr(self.sp, "enable_ai_complementarity", False):
                # Appendix B, Eq. (B.50): additional AI-supervision vacancies.
                comp_vac = self.ai_supervision_vacancies(f)
                if comp_vac > 0:
                    complementary_slots[fid] = comp_vac
                    vac_raw += comp_vac
            # Eq. (A.50): Credit-adjusted effective vacancies
            vac_eff = int(round(vac_raw * credit_hire))  # Eq. (A.40b): effective vacancies
            if vac_eff > 0:
                firm_vac_eff.append((fid, vac_eff))
                # Eq. (A.54) Aggregate vacancy
                V_t += vac_eff
        self.last_vacancies = int(V_t)
        self.last_ai_supervision_vacancies = int(sum(complementary_slots.values()))

        unemployed_ids = [i for i,w in enumerate(self.workers) if not w.employed]
        N_u = len(unemployed_ids)
        N = len(self.workers)

        # Search effort affects matching intensity and selection
        if N_u > 0:
            # Eq. (A.51) Unemployment duration
            dur = np.array([min(self.workers[i].u_dur, self.lp.search_cap_dur) for i in unemployed_ids], dtype=float)
            # Eq. (A.52) Search effort
            effort = np.clip(self.lp.search_base + self.lp.search_slope_dur * dur, self.lp.search_min, self.lp.search_max)
            # Eq. (A.53) Aggregate effective search mass
            N_eff = float(effort.sum())
        else:
            effort = np.array([], dtype=float)
            N_eff = 0.0

        # Matching count uses effective search mass
        # Eq. (A.55) Aggregate matching
        M_t = int(round(self.lp.zeta * (V_t ** self.lp.alpha) * ((N_eff + self.lp.eps_match) ** (1 - self.lp.alpha))))
        M_t = min(M_t, V_t, N_u)
        # Eq. (A.59): Labor market tightness
        tight = (V_t + self.lp.eps_match) / (N_u + self.lp.eps_match)

        hires = 0
        if M_t > 0 and firm_vac_eff and N_u > 0:
            # -------- Explicit application stage --------
            # Stage 1: each unemployed worker submits K applications to firms based on
            # compatibility (skill fit) and (optionally) vacancy intensity.
            applicants: Dict[int, List[int]] = {fid: [] for fid,_ in firm_vac_eff}

            firm_ids = np.array([fid for fid,_ in firm_vac_eff], dtype=int)
            vac_arr = np.array([vac for _,vac in firm_vac_eff], dtype=float)

            for idx_pos, wid in enumerate(unemployed_ids):
                w = self.workers[wid]
                e_i = float(effort[idx_pos]) if len(effort) else self.lp.search_base
                # Applications proportional to effort, bounded
                # Eq. (A.56) Number of job applications
                K = int(round(self.lp.applications_base * e_i))
                K = max(1, min(self.lp.applications_max, K, len(firm_ids)))

                # Eq. (A.9) baseline scalar fit; Appendix B, Eq. (B.5)
                # augments it with occupation-sensitive vector distance.
                compat = np.array(
                    [self.worker_firm_compatibility(w, self.firms[fid]) for fid in firm_ids],
                    dtype=float,
                )
                if getattr(self.sp, "enable_ai_complementarity", False):
                    # Appendix B, Eq. (B.52): multiply baseline compatibility
                    # before vacancy weighting and probability normalization.
                    compat = compat * np.array(
                        [self.complementarity_match_multiplier(self.firms[fid], w) for fid in firm_ids],
                        dtype=float,
                    )
                compat = np.maximum(self.lp.compat_floor, compat)

                # Optional vacancy weighting to reflect where jobs are posted
                # Eq. (A.57) weights to apply jobs among firms
                if self.lp.application_vacancy_weight > 0:
                    wv = (vac_arr / (vac_arr.max() + self.lp.eps_vac_norm)) ** self.lp.application_vacancy_weight
                    weights = compat * wv
                else:
                    weights = compat

                if weights.sum() <= 0:
                    continue
                # Eq. (A.58) sampling probabilities of job applications
                p = weights / weights.sum()
                chosen = self.rnd.choice(firm_ids, size=K, replace=False, p=p)
                for fid in chosen:
                    applicants[int(fid)].append(wid)

            # Stage 2: firms fill vacancy slots from their applicant pools, prioritizing match quality.
            # Randomize firm processing order to avoid systematic bias.
            firm_order = firm_ids.copy()
            self.rnd.shuffle(firm_order)

            # Maintain a set of available unemployed workers (no multiple offers in the same period)
            available = set(unemployed_ids)
            total_comp_hires = 0
            total_comp_premiums: List[float] = []

            for fid in firm_order:
                f = self.firms[int(fid)]
                vac_eff = next(v for (ff,v) in firm_vac_eff if ff==int(fid))
                if vac_eff <= 0:
                    continue

                pool = [wid for wid in applicants.get(int(fid), []) if wid in available]
                if not pool:
                    continue

                # Appendix B, Eq. (B.5): the same compatibility measure used
                # for applications also governs firm-side applicant ranking.
                def ranking_compatibility(wid: int) -> float:
                    worker = self.workers[wid]
                    score = self.worker_firm_compatibility(worker, f)
                    if getattr(self.sp, "enable_ai_complementarity", False):
                        # Appendix B, Eq. (B.52): firms use the same augmented
                        # compatibility multiplier when ranking applicants.
                        score *= self.complementarity_match_multiplier(f, worker)
                    return float(score)

                pool_sorted = sorted(pool, key=lambda wid: -ranking_compatibility(wid))

                comp_remaining = int(complementary_slots.get(int(fid), 0))
                comp_hires = 0
                comp_premiums: List[float] = []
                for wid in pool_sorted[:vac_eff]:
                    if wid not in available:
                        continue
                    w = self.workers[wid]

                    # Baseline wage schedule
                    # Eq. (A.60): baseline wage
                    occ_spec = self._occupation_spec_for_worker(w) if getattr(self.sp, "enable_occupation_layer", False) else {}
                    wage_mult = float(occ_spec.get("wage_multiplier", 1.0))
                    w_base = self.lp.w0 * wage_mult * (1.0 + float(self.rnd.normal(0, self.lp.wage_shock_std))) * (max(self.lp.wage_skill_floor, w.skill) ** self.lp.theta_s) * (tight ** self.lp.eta_w)
                    premium = 0.0
                    is_comp_hire = False
                    if getattr(self.sp, "enable_ai_complementarity", False) and comp_remaining > 0:
                        score = self.worker_complementarity_score(w)
                        if score >= float(getattr(self.sp, "ai_supervision_skill_threshold", 0.55)):
                            # Appendix B, Eq. (B.53): omega_C c_i is a
                            # proportional premium rate applied to the baseline
                            # wage offer before the bargaining cap is formed.
                            premium = self.complementarity_wage_premium_rate(w)
                            w_base *= (1.0 + premium)
                            is_comp_hire = True

                    # Bargaining via willingness-to-pay cap
                    w_cap = w_base * (1.0 + self.lp.bargaining_firm_premium)  # Eq. (A.61): cap
                    # Eq. (A.3): Job acceptance condition
                    if w_cap >= w.w_r:
                        w_neg = w.w_r + self.lp.bargaining_beta * (w_cap - w.w_r)  # Eq. (A.62): negotiated wage
                        w.employed = True
                        w.firm_id = int(fid)
                        w.w = float(w_neg)
                        f.L += 1
                        hires += 1
                        if is_comp_hire:
                            comp_remaining -= 1
                            comp_hires += 1
                            comp_premiums.append(float(premium))
                            total_comp_hires += 1
                            total_comp_premiums.append(float(premium))
                        available.remove(wid)
                    else:
                        # Bargaining fails -> possible reskilling
                        w.reskilling_decision(
                            # Eq. (A.63): proxy expected post-training wage used in the reskilling decision
                            expected_post_wage=w_base * self.lp.expected_post_wage_uplift,
                            lp=self.lp,
                            rnd=self.rnd,
                            training_cost_eff=self.effective_training_cost(w),
                        )
        # Eq. (A.66): realized unemployment after matching and hiring.
        # This must be measured after hires, not from the pre-matching
        # unemployment pool N_u.
        U = float(sum(1 for worker in self.workers if not worker.employed)) / max(1, len(self.workers))
        employed_wages = [w.w for w in self.workers if w.employed and w.firm_id is not None]
        W_mean = float(np.mean(employed_wages)) if employed_wages else self.lp.w0
        self.last_hires = int(hires)
        self.last_ai_complementary_hires = int(total_comp_hires) if "total_comp_hires" in locals() else 0
        self.last_ai_complementarity_premium = float(np.mean(total_comp_premiums)) if "total_comp_premiums" in locals() and total_comp_premiums else 0.0
        return hires, U, W_mean
    
    
    def production_and_profits(self) -> float:
        firm_workers: Dict[int, List[int]] = {f.id: [] for f in self.firms}
        for i,w in enumerate(self.workers):
            if w.employed and w.firm_id is not None: firm_workers[w.firm_id].append(i)
        Y=0.0
        ai_cost_total = 0.0
        compute_cost_total = 0.0
        compute_demand_total = 0.0
        ai_loss_total = 0.0
        ai_error_events = 0
        ai_error_rates: List[float] = []
        cyber_loss_total = 0.0
        cyber_control_cost_total = 0.0
        cyber_incidents = 0
        cyber_risk_rates: List[float] = []
        ai_cost = self.effective_ai_cost()
        for f in self.firms:
            skills=[self.workers[i].s for i in firm_workers.get(f.id, [])]; avg_skill=float(np.mean(skills)) if skills else 0.5
            y=f.ces_output(avg_skill, self.tp)
            z = float(getattr(self, "macro_z", 1.0))
            # Appendix B, Eq. (B.59): demand shock is a proportional output
            # and revenue wedge applied alongside aggregate demand state Z_t.
            demand = max(0.0, 1.0 + float(getattr(self, "active_demand_shock", 0.0)))
            y *= z * demand
            # Eq. (A.20): Wage bill aggregation
            wage_bill=sum(self.workers[i].w for i in firm_workers.get(f.id, []))
            ai_operating_cost = 0.0
            if getattr(self.sp, "enable_ai_relative_cost", False):
                stage = int(getattr(f, "deployment_stage", 2))
                deployment_cost_factor = 0.0
                if stage == 1:
                    deployment_cost_factor = float(getattr(self.sp, "pilot_cost", 0.02))
                elif stage >= 2:
                    deployment_cost_factor = float(f.rho_adopt)
                ai_operating_cost = (
                    ai_cost
                    * self.firm_ai_cost_multiplier(int(f.id))
                    * float(getattr(self.sp, "ai_operating_cost_scale", 0.05))
                    * deployment_cost_factor
                    * max(0.0, float(f.L))
                )
            compute_demand = self.firm_compute_demand(f)
            compute_unit_cost = self.effective_compute_unit_cost(compute_demand)
            # Appendix B, Eq. (B.23): realized compute operating cost is
            # capped at compute_cost_cap_share times firm output.
            compute_operating_cost = compute_unit_cost * compute_demand
            if compute_operating_cost > 0.0:
                cap = max(0.0, float(getattr(self.sp, "compute_cost_cap_share", 0.50))) * max(0.0, float(y))
                compute_operating_cost = min(cap, compute_operating_cost)
            ai_cost_total += float(ai_operating_cost)
            compute_cost_total += float(compute_operating_cost)
            compute_demand_total += float(compute_demand)
            ai_loss, ai_error, ai_error_rate = self.ai_operational_loss(f, y)
            ai_loss_total += float(ai_loss)
            ai_error_events += int(ai_error)
            if getattr(self.sp, "enable_ai_operational_risk", False) and float(getattr(f, "rho_adopt", 0.0)) > 0.0:
                ai_error_rates.append(float(ai_error_rate))
            cyber_control_cost = self.cyber_privacy_control_cost(f)
            cyber_loss, cyber_incident, cyber_rate = self.cyber_privacy_operational_loss(f, y)
            cyber_control_cost_total += float(cyber_control_cost)
            cyber_loss_total += float(cyber_loss)
            cyber_incidents += int(cyber_incident)
            if getattr(self.sp, "enable_cyber_privacy_risk", False) and float(getattr(f, "rho_adopt", 0.0)) > 0.0:
                cyber_risk_rates.append(float(cyber_rate))
            # Appendix B, Eq. (B.48): extended nonnegative firm profit.
            # The implemented capital-cost coefficient is r_K = 0.02.
            profit=max(0.0, y - wage_bill - 0.02 * f.A - ai_operating_cost - compute_operating_cost - ai_loss - cyber_control_cost - cyber_loss)
            f.last_output = float(y)
            f.last_wage_bill = float(wage_bill)
            f.profit=profit; Y += y
        self.last_ai_operating_cost = float(ai_cost_total)
        self.last_compute_operating_cost = float(compute_cost_total)
        self.last_compute_demand = float(compute_demand_total)
        self.last_ai_operational_loss = float(ai_loss_total)
        self.last_ai_error_events = int(ai_error_events)
        self.last_ai_error_rate_mean = float(np.mean(ai_error_rates)) if ai_error_rates else 0.0
        self.last_cyber_privacy_incidents = int(cyber_incidents)
        self.last_cyber_privacy_loss = float(cyber_loss_total)
        self.last_cyber_privacy_control_cost = float(cyber_control_cost_total)
        self.last_cyber_privacy_risk_rate_mean = float(np.mean(cyber_risk_rates)) if cyber_risk_rates else 0.0
        return Y

    def evolutionary_selection(self) -> int:
        """Evolutionary firm selection via exit and replacement.

        Firms with persistently low profitability (or very low productivity A) face a higher
        probability of exit. When a firm exits, its employed workers become unemployed
        immediately, and the firm is replaced by a new entrant (keeping the number of firms
        constant for comparability across scenarios).

        Returns
        -------
        int
            Number of exiting firms in the current period.
        """
        exits: List[Firm] = []
        exit_probs: List[float] = []
        profit_shortfalls: List[float] = []
        below_profit_cutoff = 0
        below_A_min = 0
        profits = [float(f.profit) for f in self.firms]
        margins = [
            float(f.profit) / max(1e-9, float(getattr(f, "last_output", 0.0)))
            for f in self.firms
            if float(getattr(f, "last_output", 0.0)) > 1e-9
        ]
        use_margin_exit = bool(getattr(self.sp, "firm_exit_use_margin", False))
        for f in self.firms:
            # Eq. (A.27): Profit shortfall. By default this uses raw profit,
            # preserving the baseline report equation. In the optional
            # margin-based variant, the shortfall is measured against profit
            # margin profit_j / max(Y_j, epsilon), which is scale-stable across
            # firms and calibration runs.
            if use_margin_exit:
                profit_metric = float(f.profit) / max(1e-9, float(getattr(f, "last_output", 0.0)))
                cutoff = float(getattr(self.sp, "firm_exit_margin_cutoff", 0.05))
            else:
                profit_metric = float(f.profit)
                cutoff = float(self.sp.firm_exit_profit_cutoff)
            profit_shortfall = max(0.0, cutoff - profit_metric)
            profit_shortfalls.append(float(profit_shortfall))
            if profit_shortfall > 0.0:
                below_profit_cutoff += 1
            # Eq. (A.28): Exit probability
            p_exit = float(self.sp.firm_exit_prob_base + self.sp.firm_exit_slope * profit_shortfall)
            # Eq. (A.29): Exit probability adjustment
            if float(f.A) < float(self.sp.firm_exit_A_min):
                below_A_min += 1
                p_exit = max(p_exit, float(self.sp.firm_exit_prob_floor_lowA))
            p_exit = float(np.clip(p_exit, 0.0, 1.0))
            exit_probs.append(float(p_exit))
            # Eq. (A.30): Bernoulli exit draw
            if self.rnd.random() < p_exit:
                exits.append(f)

        n_firms = max(1, len(self.firms))
        self.last_exit_prob_mean = float(np.mean(exit_probs)) if exit_probs else 0.0
        self.last_exit_profit_shortfall_mean = float(np.mean(profit_shortfalls)) if profit_shortfalls else 0.0
        self.last_exit_below_profit_cutoff_share = float(below_profit_cutoff / n_firms)
        self.last_exit_below_A_min_share = float(below_A_min / n_firms)
        self.last_firm_profit_mean = float(np.mean(profits)) if profits else 0.0
        self.last_firm_profit_p10 = float(np.percentile(profits, 10)) if profits else 0.0
        self.last_firm_profit_margin_mean = float(np.mean(margins)) if margins else 0.0

        if not exits:
            self.last_entry_vacancies = 0
            self.last_exit_layoffs = 0
            self.last_exit_reallocated_workers = 0
            self.last_exit_unabsorbed_workers = 0
            return 0

        # Appendix A exit/entry block: workers attached to exiting firms are
        # displaced before entrant reallocation is optionally applied.
        exiting_ids = {f.id for f in exits}
        displaced_workers: List[Worker] = []
        for w in self.workers:
            if w.employed and (w.firm_id in exiting_ids):
                w.employed = False
                w.firm_id = None
                w.w = 0.0
                # Initialize the within-period unemployment spell. Appendix A,
                # Eq. (A.65) assigns duration one at the period-end update.
                w.u_dur = 0
                displaced_workers.append(w)

        entrant_capacity: Dict[int, int] = {}
        for f in exits:
            # Eq. (A.31): Entrant productivity draw
            f.A = float(self.rnd.uniform(self.sp.firm_entry_A_low, self.sp.firm_entry_A_high))
            entry_target_L = int(max(self.sp.firm_entry_L_min, self.rnd.poisson(self.sp.firm_entry_L_mean)))
            entry_target_L = int(round(entry_target_L * max(1.0, float(getattr(self.sp, "entry_vacancy_multiplier", 1.0)))))
            f.L = 0
            # Eq. (A.32): Entrant automation initialization
            f.kappa = float(self.rnd.lognormal(self.sp.firm_entry_kappa_logn_mu, self.sp.firm_entry_kappa_logn_sigma))
            f.rho_adopt = float(np.clip(self.rnd.normal(self.sp.firm_entry_rho_mean, self.sp.firm_entry_rho_sd), 0.0, self.sp.firm_entry_rho_max))
            f.deployment_stage = 2 if f.rho_adopt > 0.0 else 0
            if getattr(self.sp, "enable_deployment_friction", False):
                f.deployment_stage = 0
                f.rho_adopt = 0.0
            f.data_readiness = float(np.clip(self.rnd.normal(getattr(self.sp, "data_readiness_mean", 0.65), getattr(self.sp, "data_readiness_sd", 0.15)), 0.0, 1.0))
            f.req_skill = float(np.clip(self.rnd.normal(self.sp.firm_req_skill_mean, self.sp.firm_req_skill_sd), 0.0, 1.0))
            f.routine_intensity = float(np.clip(self.rnd.normal(self.sp.firm_routine_mean, self.sp.firm_routine_sd), 0.0, 1.0))
            if getattr(self.sp, "enable_occupation_layer", False):
                req_vec = self._firm_required_skill_vector(f.req_skill, f.routine_intensity)
                f.req_cog = req_vec["cog"]
                f.req_soc = req_vec["soc"]
                f.req_tech = req_vec["tech"]
                f.req_man = req_vec["man"]
                f.req_manual = req_vec["manual"]
            f.profit = 0.0
            f.pi_exp = 0.0
            f.desired_L = max(0, int(entry_target_L) - int(f.L))
            entrant_capacity[int(f.id)] = int(f.desired_L)

        self.last_exit_layoffs = int(len(displaced_workers))
        self.last_entry_vacancies = int(sum(entrant_capacity.values()))
        self.last_exit_reallocated_workers = 0
        if getattr(self.sp, "enable_entry_reallocation", False) and displaced_workers and entrant_capacity:
            share = float(np.clip(getattr(self.sp, "entry_reallocation_share", 0.0), 0.0, 1.0))
            n_to_reallocate = min(int(round(share * len(displaced_workers))), sum(entrant_capacity.values()))
            if n_to_reallocate > 0:
                self.rnd.shuffle(displaced_workers)
                entrants_by_id = {int(f.id): f for f in exits}
                capacity_items = [(fid, cap) for fid, cap in entrant_capacity.items() if cap > 0]
                cap_idx = 0
                for w in displaced_workers[:n_to_reallocate]:
                    while cap_idx < len(capacity_items) and capacity_items[cap_idx][1] <= 0:
                        cap_idx += 1
                    if cap_idx >= len(capacity_items):
                        break
                    firm_id, cap = capacity_items[cap_idx]
                    firm = entrants_by_id[firm_id]
                    occ_spec = self._occupation_spec_for_worker(w)
                    wage_mult = float(occ_spec.get("wage_multiplier", 1.0))
                    wage_floor = self.lp.w0 * wage_mult * (max(self.lp.wage_skill_floor, w.skill) ** self.lp.theta_s)
                    w.employed = True
                    w.firm_id = firm_id
                    w.w = float(max(w.w_r, wage_floor))
                    w.u_dur = 0
                    firm.L += 1
                    firm.desired_L = max(0, int(firm.desired_L) - 1)
                    capacity_items[cap_idx] = (firm_id, cap - 1)
                    self.last_exit_reallocated_workers += 1
        self.last_exit_unabsorbed_workers = int(self.last_exit_layoffs - self.last_exit_reallocated_workers)

        return len(exits)

    def merger_acquisition_dynamics(self) -> tuple[int, int]:
        """Optional M&A restructuring channel with employment consequences.

        Distressed or AI-lagging firms may be acquired by stronger firms. A
        fraction of acquired workers is separated as redundant labor, while the
        remainder transfers to the acquiring firm. The acquired firm slot is
        then reinitialized so the aggregate number of firms remains comparable
        across scenarios.
        """
        self.last_ma_events = 0
        self.last_ma_layoffs = 0
        if not getattr(self.sp, "enable_ma_dynamics", False) or len(self.firms) < 2:
            return 0, 0

        avg_rho = float(np.mean([f.rho_adopt for f in self.firms])) if self.firms else 0.0
        avg_L = float(np.mean([max(0, f.L) for f in self.firms])) if self.firms else 1.0
        candidate_ids: list[int] = []
        for f in self.firms:
            # Appendix B, Eq. (B.35): acquisition probability. The profit
            # benchmark pi* is firm_exit_profit_cutoff. Because M&A is resolved
            # before current-period production, f.profit is the most recently
            # realized, effectively lagged, profit.
            profit_shortfall = max(0.0, self.sp.firm_exit_profit_cutoff - float(f.profit))
            ai_gap = max(0.0, avg_rho - float(f.rho_adopt))
            # max(1, avg_L) is a numerical denominator safeguard for B.35.
            size_gap = max(0.0, avg_L - float(max(0, f.L))) / max(1.0, avg_L)
            p_ma = float(getattr(self.sp, "ma_base_probability", 0.0))
            p_ma += float(getattr(self.sp, "ma_profit_sensitivity", 0.10)) * profit_shortfall
            p_ma += float(getattr(self.sp, "ma_ai_gap_sensitivity", 0.10)) * ai_gap
            p_ma += float(getattr(self.sp, "ma_size_sensitivity", 0.0)) * size_gap
            p_ma = float(np.clip(p_ma, 0.0, 1.0))
            if self.rnd.random() < p_ma:
                candidate_ids.append(int(f.id))

        if not candidate_ids:
            return 0, 0

        events = 0
        layoffs = 0
        occ_sep: Dict[str, int] = {occ_id: 0 for occ_id in self.occupation_ids()} if getattr(self.sp, "enable_occupation_layer", False) else {}
        for target_id in candidate_ids:
            target = self.firms[target_id]
            acquirer_pool = [f for f in self.firms if f.id != target_id and f.id not in candidate_ids]
            if not acquirer_pool:
                acquirer_pool = [f for f in self.firms if f.id != target_id]
            if not acquirer_pool:
                continue
            # Implementation detail following Eqs. (B.35)-(B.38): choose the
            # acquirer lexicographically by profit, productivity, then adoption.
            acquirer = max(acquirer_pool, key=lambda f: (float(f.profit), float(f.A), float(f.rho_adopt)))
            target_workers = [w for w in self.workers if w.employed and w.firm_id == target_id]
            # A candidate acquisition is completed only when the target has
            # employed workers to retain or release through Eq. (B.36).
            if not target_workers:
                continue

            redundancy = float(np.clip(getattr(self.sp, "ma_redundancy_rate", 0.15), 0.0, 1.0))
            # Appendix B, Eq. (B.36): each target worker is independently
            # retained with probability 1 - delta_M.
            retain_prob = 1.0 - redundancy
            retained = 0
            for w in target_workers:
                if self.rnd.random() < retain_prob:
                    w.firm_id = int(acquirer.id)
                    w.w = float(w.w) * float(getattr(self.sp, "ma_wage_retention_factor", 1.0))
                    retained += 1
                else:
                    w.employed = False
                    w.firm_id = None
                    w.w = 0.0
                    # Initialize the within-period unemployment spell. Appendix
                    # A, Eq. (A.65a) is applied after matching.
                    w.u_dur = 0
                    layoffs += 1
                    if getattr(self.sp, "enable_occupation_layer", False):
                        occ = getattr(w, "occupation", "general")
                        occ_sep[occ] = occ_sep.get(occ, 0) + 1

            acquirer.L = int(max(0, acquirer.L) + retained)
            # Appendix B, Eq. (B.37): partial transfer of target productivity.
            acquirer.A = float(max(acquirer.A, acquirer.A + float(getattr(self.sp, "ma_productivity_transfer", 0.30)) * max(0.0, target.A - acquirer.A)))
            # Appendix B, Eq. (B.38): partial transfer of target adoption stock.
            acquirer.rho_adopt = float(np.clip(
                acquirer.rho_adopt + float(getattr(self.sp, "ma_adoption_transfer", 0.30)) * max(0.0, target.rho_adopt - acquirer.rho_adopt),
                0.0,
                1.0,
            ))

            # The acquired firm slot is reinitialized after the transaction so
            # the simulated number of firms remains constant across scenarios.
            target.A = float(self.rnd.uniform(self.sp.firm_entry_A_low, self.sp.firm_entry_A_high))
            entry_target_L = int(max(self.sp.firm_entry_L_min, self.rnd.poisson(self.sp.firm_entry_L_mean)))
            target.L = 0
            target.kappa = float(self.rnd.lognormal(self.sp.firm_entry_kappa_logn_mu, self.sp.firm_entry_kappa_logn_sigma))
            target.rho_adopt = float(np.clip(self.rnd.normal(self.sp.firm_entry_rho_mean, self.sp.firm_entry_rho_sd), 0.0, self.sp.firm_entry_rho_max))
            target.deployment_stage = 2 if target.rho_adopt > 0.0 else 0
            if getattr(self.sp, "enable_deployment_friction", False):
                target.deployment_stage = 0
                target.rho_adopt = 0.0
            target.data_readiness = float(np.clip(self.rnd.normal(getattr(self.sp, "data_readiness_mean", 0.65), getattr(self.sp, "data_readiness_sd", 0.15)), 0.0, 1.0))
            target.req_skill = float(np.clip(self.rnd.normal(self.sp.firm_req_skill_mean, self.sp.firm_req_skill_sd), 0.0, 1.0))
            target.routine_intensity = float(np.clip(self.rnd.normal(self.sp.firm_routine_mean, self.sp.firm_routine_sd), 0.0, 1.0))
            if getattr(self.sp, "enable_occupation_layer", False):
                req_vec = self._firm_required_skill_vector(target.req_skill, target.routine_intensity)
                target.req_cog = req_vec["cog"]
                target.req_soc = req_vec["soc"]
                target.req_tech = req_vec["tech"]
                target.req_man = req_vec["man"]
                target.req_manual = req_vec["manual"]
            target.profit = 0.0
            target.pi_exp = 0.0
            target.desired_L = entry_target_L
            events += 1

        self.last_ma_events = int(events)
        self.last_ma_layoffs = int(layoffs)
        self.last_separations = int(getattr(self, "last_separations", 0)) + int(layoffs)
        if getattr(self.sp, "enable_occupation_layer", False):
            merged = dict(getattr(self, "last_occ_separations", {}) or {})
            for occ, count in occ_sep.items():
                merged[occ] = merged.get(occ, 0) + int(count)
            self.last_occ_separations = merged
        return events, layoffs

    def technology_updates(self):
        """Update technology suppliers (innovation + pricing) and compute supplier-specific price-performance.

        Tech pricing adapts to lagged realized demand. With supplier choice,
        demand is tracked per supplier and markups update with supplier-specific demand gaps.
        """
        adopt_mean = float(np.mean([f.rho_adopt for f in self.firms])) if self.firms else 0.0

        # Ensure demand vector exists
        if not self.last_tech_demand_vec or len(self.last_tech_demand_vec) != len(self.techs):
            self.last_tech_demand_vec = [float(self.tp.tech_demand_target) for _ in range(len(self.techs))]
        shares = self.normalized_tech_market_shares()
        K = max(1, len(self.techs))
        share_benchmark = 1.0 / float(K)
        # Appendix B, Eq. (B.32): positive lagged-share advantage.
        share_advantage = np.maximum(0.0, shares - share_benchmark)
        rnd_advantage = 0.0
        markup_advantage = 0.0

        # Update tech performance and prices using per-supplier demand gaps (lagged)
        for k, tech in enumerate(self.techs):
            # Eq. (A.15): Demand gap for markup adjustment
            demand_gap_k = float(self.last_tech_demand_vec[k] - self.tp.tech_demand_target)
            if getattr(self.tp, "enable_winner_take_more", False):
                # Appendix B, Eq. (B.34): lagged demand gap plus share-based
                # markup advantage.
                demand_gap_k += float(getattr(self.tp, "tech_markup_share_sensitivity", 0.10)) * float(share_advantage[k])
                original_alpha = float(self.tp.alpha_R)
                # Appendix B, Eq. (B.33): share-adjusted innovation drift.
                self.tp.alpha_R = original_alpha * (1.0 + float(getattr(self.tp, "tech_rnd_share_sensitivity", 0.25)) * float(share_advantage[k]))
                rnd_advantage += float(self.tp.alpha_R - original_alpha)
                markup_advantage += float(getattr(self.tp, "tech_markup_share_sensitivity", 0.10)) * float(share_advantage[k])
                tech.innovate(demand_index=adopt_mean, tp=self.tp, rnd=self.rnd)
                self.tp.alpha_R = original_alpha
            else:
                tech.innovate(demand_index=adopt_mean, tp=self.tp, rnd=self.rnd)
            if self.active_productivity_shock:
                # Appendix B, Eq. (B.58): proportional shock to technology
                # supplier performance pi_k after its baseline innovation update.
                tech.pi_k = max(1e-4, float(tech.pi_k) * (1.0 + self.active_productivity_shock))
            # Eq. (A.14): Unit cost from tech performance
            cost = 1.0 / max(1e-6, tech.pi_k)
            tech.set_price(cost=cost, demand_gap=demand_gap_k, tp=self.tp, rnd=self.rnd)
        self.last_tech_rnd_advantage = float(rnd_advantage / float(K))
        self.last_tech_markup_advantage = float(markup_advantage / float(K))

        # Supplier-specific price-performance signals
        # Eq. (A.67) Price performance signal
        pp_vec = [float(t.pi_k) / max(1e-6, float(t.price)) for t in self.techs]
        return pp_vec

    def tech_choice_probs(self, pp_vec: List[float]) -> np.ndarray:
        """Compute choice probabilities over tech suppliers given price-performance signals.

        Firms choose suppliers via a multinomial logit rule with optional idiosyncratic taste noise.
        """
        pp = np.asarray(pp_vec, dtype=float)
        pp = np.clip(pp, 1e-8, None)
        # Appendix B, Eq. (B.29): utility from centered price-performance.
        lam = float(self.tp.tech_choice_lambda)
        util = lam * (pp - float(np.mean(pp)))
        if getattr(self.tp, "enable_winner_take_more", False):
            shares = self.normalized_tech_market_shares()
            if len(shares) == len(pp):
                # Appendix B, Eq. (B.29): lagged-share feedback component.
                share_util = float(getattr(self.tp, "tech_share_feedback", 1.00)) * (shares - float(np.mean(shares)))
                util = util + share_util
        # Appendix B, Eq. (B.30): multinomial-logit supplier probabilities.
        # Clipping utilities before exponentiation is only a numerical safeguard.
        ex = np.exp(np.clip(util, -50, 50))
        probs = ex / max(1e-12, float(np.sum(ex)))
        return probs

    def adoption_updates(self, pp_vec: List[float]):
        """Update firm automation adoption and compute realized tech demand and market shares.

        Firms choose among tech suppliers; adoption hazard depends on the effective
        price-performance of the chosen supplier. Demand is tracked (expected hazard mass) per supplier
        and used as lagged input into tech pricing next period.
        """
        if not self.firms:
            self.last_tech_demand = float(self.tp.tech_demand_target)
            self.last_tech_demand_vec = [float(self.tp.tech_demand_target) for _ in range(len(self.techs))]
            self.last_resource_demand = 0.0
            self.last_resource_constraint = 1.0
            self.last_ai_relative_cost_mean = 0.0
            self.last_compute_demand = 0.0
            self.last_compute_unit_cost = 0.0
            self.last_failed_pilots = 0
            self.deployment_stats()
            self.last_tech_share_hhi = 0.0
            self.update_tech_market_shares(np.array([], dtype=float))
            self.history["Tech_demand"].append(float(self.last_tech_demand))
            self.history["Tech_mu"].append(float(np.mean([t.mu_markup for t in self.techs])) if self.techs else 0.0)
            self.history["Tech_share_top"].append(0.0)
            self.history["Tech_share_hhi"].append(0.0)
            self.history["Tech_rnd_advantage"].append(0.0)
            self.history["Tech_markup_advantage"].append(0.0)
            return

        adopt_mean = float(np.mean([f.rho_adopt for f in self.firms]))
        credit_scale = float(self.credit_factor()) if self.sp.enable_credit_channel else 1.0
        # Appendix B, Eq. (B.11): omega_C times the mean remaining adoption
        # gap (1 - rho_j,t) across all J firms.
        desired_resource_demand = (
            float(getattr(self.sp, "resource_demand_per_adopter", 1.0))
            * float(np.mean([max(0.0, 1.0 - float(f.rho_adopt)) for f in self.firms]))
        )
        resource_scale = self.resource_constraint_factor(desired_resource_demand)
        ai_cost = self.effective_ai_cost(desired_resource_demand)

        # Base supplier choice probabilities from price-performance
        base_probs = self.tech_choice_probs(pp_vec)
        K = len(pp_vec)
        demand_accum = np.zeros(K, dtype=float)
        relative_costs: List[float] = []
        failed_pilots = 0

        # Adoption updates per firm
        for f in self.firms:
            expected_delta_prod = 0.02 * (1.0 - f.rho_adopt)
            relative_ai_penalty = 0.0
            compute_penalty = 0.0
            cyber_penalty = 0.0
            if getattr(self.sp, "enable_ai_relative_cost", False):
                wage_ref = max(1e-6, self.firm_mean_wage(int(f.id)))
                # Appendix B, Eq. (B.16): wage-relative AI cost ratio q_j,t.
                relative_cost = (ai_cost * self.firm_ai_cost_multiplier(int(f.id))) / wage_ref
                relative_costs.append(float(relative_cost))
                # Appendix B, Eq. (B.17): relative-cost adoption penalty.
                relative_ai_penalty = float(getattr(self.sp, "ai_relative_cost_sensitivity", 1.0)) * max(0.0, relative_cost - 1.0)
            if getattr(self.sp, "enable_compute_cost", False):
                wage_ref = max(1e-6, self.firm_mean_wage(int(f.id)))
                compute_demand = self.firm_compute_demand(f, include_training=True)
                compute_unit_cost = self.effective_compute_unit_cost(compute_demand)
                relative_compute_cost = (compute_unit_cost * compute_demand) / max(wage_ref, 1e-6)
                # Appendix B, Eq. (B.22): compute-cost adoption penalty.
                compute_penalty = float(getattr(self.sp, "compute_adoption_cost_sensitivity", 0.50)) * max(0.0, relative_compute_cost - 1.0)
            if getattr(self.sp, "enable_cyber_privacy_risk", False):
                # Appendix B, Eq. (B.46): anticipated cyber-risk penalty.
                cyber_penalty = float(getattr(self.sp, "cyber_adoption_friction", 0.50)) * self.cyber_privacy_risk_rate(f)

            # Appendix B, Eq. (B.47): total penalty supplied to the adoption
            # decision. Disabled extension components remain equal to zero.
            ai_cost_penalty = relative_ai_penalty + compute_penalty + cyber_penalty

            # Firm-specific taste shock (optional): perturb utilities then softmax
            if self.tp.tech_choice_noise_std > 0:
                noise = self.rnd.normal(0.0, float(self.tp.tech_choice_noise_std), size=K)
                util = np.log(np.clip(base_probs, 1e-12, None)) + noise
                ex = np.exp(np.clip(util - float(np.max(util)), -50, 50))
                probs = ex / max(1e-12, float(np.sum(ex)))
            else:
                probs = base_probs

            # Effective price-performance faced by the firm (expected under choice probabilities)
            # Eq. (A.69) Effective price performance signal
            pp_eff = float(np.dot(probs, np.asarray(pp_vec, dtype=float)))
            # Resolve transitions for firms already in pilot/production and
            # obtain psi_j,t for the current productivity update, Eqs. (B.25)-(B.28).
            deployment_factor, failed = self.update_deployment_stage(f, 0.0)
            if failed:
                failed_pilots += 1

            # Appendix B, Eq. (B.13): resource_scale throttles the baseline
            # hazard. Credit remains a separate baseline multiplicative factor.
            h = f.decide_adoption(
                self.tp, pp_eff, expected_delta_prod,
                rho_bar=adopt_mean, rnd=self.rnd, credit_factor=credit_scale * resource_scale, ai_cost_penalty=ai_cost_penalty, deployment_factor=deployment_factor
            )
            # Appendix B, Eq. (B.24): the newly calculated hazard can initiate
            # a pilot. Its productivity effect begins in the next update.
            deployment_factor, failed = self.update_deployment_stage(f, h)
            if failed:
                failed_pilots += 1

            # Expected demand mass allocated across suppliers (hazard-weighted)
            demand_accum += float(h) * probs

        # Normalize demand to [0,1]-like scale per supplier (per-firm average hazard mass)
        J = max(1, len(self.firms))
        demand_vec = (demand_accum / float(J)).tolist()
        self.last_tech_demand_vec = [float(x) for x in demand_vec]
        self.last_tech_demand = float(np.mean(self.last_tech_demand_vec)) if self.last_tech_demand_vec else float(self.tp.tech_demand_target)
        self.last_ai_relative_cost_mean = float(np.mean(relative_costs)) if relative_costs else 0.0
        self.last_failed_pilots = int(failed_pilots)
        self.deployment_stats()

        # Current realized demand shares tilde_s_k,t in Appendix B, Eq. (B.31)
        # are hazard-weighted supplier-choice probabilities.
        total = float(np.sum(demand_accum))
        if total > 0:
            shares = demand_accum / total
            top_share = float(np.max(shares))
            hhi = float(np.sum(shares**2))
        else:
            shares = np.ones(len(self.techs), dtype=float) / max(1, len(self.techs))
            top_share, hhi = 0.0, 0.0
        self.last_tech_share_hhi = float(hhi)
        self.update_tech_market_shares(shares)

        # Record diagnostics
        self.history["Tech_demand"].append(float(self.last_tech_demand))
        self.history["Tech_mu"].append(float(np.mean([t.mu_markup for t in self.techs])) if self.techs else 0.0)
        self.history["Tech_share_top"].append(top_share)
        self.history["Tech_share_hhi"].append(hhi)
        self.history["Tech_rnd_advantage"].append(float(getattr(self, "last_tech_rnd_advantage", 0.0)))
        self.history["Tech_markup_advantage"].append(float(getattr(self, "last_tech_markup_advantage", 0.0)))

    def investment_and_planning(self):
        # Smooth profit expectations and use them for planning
        # Peer imitation in investment and (optionally) labor planning
        lam = float(getattr(self.sp, "profit_exp_lambda", 0.25))
        noise_std = float(getattr(self.sp, "profit_exp_noise_std", 0.0))
        xi = float(getattr(self.sp, "invest_imitation_xi", 0.20))
        top_frac = float(getattr(self.sp, "invest_imitation_top_frac", 0.10))
        xi = float(np.clip(xi, 0.0, 1.0))
        top_frac = float(np.clip(top_frac, 0.01, 1.0))

        # First pass: update expected profits and compute "self" plans
        expected_list: List[float] = []
        # Eq. (A.30): Self plans (investment/labor)
        I_self: List[float] = []
        L_self: List[int] = []
        gamma_I_eff = self.gamma_I_effective()
        # Eq. (A.24) and Eq. (A.26): profit hurdle for investment and desired labor.
        # The default value, 0.20, preserves the original hard-coded baseline.
        profit_target = float(np.clip(getattr(self.sp, "invest_profit_target", 0.20), 0.0, 1.0))

        for f in self.firms:
            # Eq. (A.22): Expected profit update using the baseline
            # level-profit planning rule.
            f.pi_exp = (1.0 - lam) * f.pi_exp + lam * float(f.profit)
            # Eq. (A.23): Expected profit with a noise term            
            expected = float(f.pi_exp + (self.rnd.normal(0, noise_std) if noise_std > 0 else 0.0))
            expected_list.append(expected)

            # Eq. (A.24): Investment rule
            I = max(0.0, gamma_I_eff * (expected - profit_target))
            I_self.append(float(I))

            # Eq. (A.26): Desired labor update. For a newly replaced firm,
            # f.L is zero but desired_L contains the entrant's initial hiring
            # target from Eq. (A.31). Preserve that target until matching has
            # an opportunity to fill the posted vacancies.
            entry_target = int(f.desired_L) if int(f.L) == 0 and int(f.desired_L) > 0 else 0
            Ld = max(entry_target, int(round(f.L + (expected - profit_target))), 0)
            L_self.append(int(Ld))

        # Peer benchmark from top-performing firms (by expected profit)
        J = len(self.firms)
        top_n = max(1, int(round(top_frac * J)))
        top_idx = np.argsort(np.array(expected_list))[-top_n:]
        # Eq. (A.33): Peer-average plans
        I_peer = float(np.mean([I_self[k] for k in top_idx])) if top_n > 0 else 0.0
        L_peer = int(round(float(np.mean([L_self[k] for k in top_idx])))) if top_n > 0 else 0

        # Second pass: apply imitation blend and update states
        for j, f in enumerate(self.firms):
            # Eq. (A.34): Blended investment and labor plan
            I_blend = (1.0 - xi) * I_self[j] + xi * I_peer
            f.A += float(I_blend)

            Ld_blend = int(round((1.0 - xi) * L_self[j] + xi * L_peer))
            # Complementary vacancies are added once in labor_market() under
            # Eq. (B.50), rather than being duplicated in desired labour.
            f.desired_L = max(0, int(Ld_blend))

        # Depreciation
        for f in self.firms:
            # Eq. (A.25): Productivity update
            f.A = max(0.0, (1.0 - self.tp.depreciation_K) * f.A)
        if getattr(self.sp, "enable_ai_complementarity", False) and self.firms:
            intensities = [self.firm_new_task_intensity(f) for f in self.firms]
            self.last_new_task_share = float(np.mean([1.0 if x > 0.0 else 0.0 for x in intensities]))
        else:
            self.last_new_task_share = 0.0

    def policy_feedback(self, Y: float | None = None):
        wages=[];
        all_worker_labor_income=[]
        for w in self.workers:
            if w.employed and w.firm_id is not None:
                wages.append(w.w)
                all_worker_labor_income.append(w.w)
            else:
                all_worker_labor_income.append(0.0)
        # Eq. (A.36): Inequality measure (Gini) on employed workers' realized wages.
        I_ratio = self.gini_nonnegative(wages)
        # Diagnostic alternative to Eq. (A.36): all-worker current labor-income
        # Gini includes unemployed workers with zero current labor income. This
        # is closer to an income-distribution target than an employed-only wage
        # Gini, but it remains a labor-income proxy rather than household income.
        self.last_ineq_all_workers = self.gini_nonnegative(all_worker_labor_income)
        self.last_ineq = float(I_ratio)
        U = self.unemployment_rate()
        self.gov.update_policy(U=U, I_ratio=I_ratio, pp=self.pp)
        self.gov.G = float(np.clip(self.gov.G, self.sp.policy_G_min, self.sp.policy_G_max))
        self.gov.S = float(np.clip(self.gov.S, 0.0, self.sp.policy_S_max))
        y_t = max(1e-6, float(Y) if Y is not None else (self.history["Y"][-1] if self.history["Y"] else 1.0))
        y_prev = max(1e-6, self.history["Y"][-1] if self.history["Y"] else y_t)
        growth = float((y_t - y_prev) / y_prev)
        subsidy_share = (self.sp.subsidy_spend_scale * self.gov.S * U) if getattr(self.sp, 'enable_training_subsidy', False) else 0.0
        # Appendix B, Eq. (B.61): primary deficit share. The subsidy term is
        # zero when the optional training-subsidy channel is disabled.
        primary_deficit_share = self.gov.G + subsidy_share - self.gov.tau
        # Appendix B, Eq. (B.62), extending Eq. (A.39): debt-ratio transition.
        # max(epsilon_g, 1 + growth) prevents division by a nonpositive value.
        self.gov.B = ((1.0 + self.pp.r_debt) / max(1e-6, 1.0 + growth)) * self.gov.B + primary_deficit_share
        self.gov.B = float(np.clip(self.gov.B, self.sp.debt_ratio_min, self.sp.debt_ratio_max))
        # Adapt policy rule coefficients after observing outcomes and updating debt
        self.gov.adapt_rule_coefficients(U=U, I_ratio=I_ratio, pp=self.pp, sp=self.sp)
        return I_ratio


    def worker_separations_only(self) -> None:
        """Apply separations/layoffs before matching (Option A timing).

        Separations use the most recently realized profits (from the end of period t−1),
        plus credit and macro conditions.
        """
        credit = float(self.credit_factor())
        z = float(getattr(self, "macro_z", 1.0))
        n_sep = 0
        occ_sep: Dict[str, int] = {occ_id: 0 for occ_id in self.occupation_ids()} if getattr(self.sp, "enable_occupation_layer", False) else {}
        for w in self.workers:
            if (not w.employed) or (w.firm_id is None):
                continue
            f = self.firms[w.firm_id]
            # Eq. (A.10): Profit stress calculation
            profit_gap = max(0.0, -float(f.profit)) / max(self.lp.eps_vac_norm, float(self.sp.sep_profit_scale))
            # Eq. (A.11): Separation probability components (baseline + profit/credit/macro stress)
            p = float(self.sp.sep_base)
            p += float(self.sp.sep_profit_sensitivity) * profit_gap
            p += float(self.sp.sep_credit_sensitivity) * max(0.0, 1.0 - credit)
            p += float(self.sp.sep_macro_sensitivity) * max(0.0, 1.0 - z)
            if getattr(self.sp, "enable_occupation_layer", False):
                occ = self._occupation_spec_for_worker(w)
                occ_id = getattr(w, "occupation", "general")
                multipliers = getattr(self.sp, "occupation_substitution_multipliers", {}) or {}
                sub_mult = float(multipliers.get(occ_id, 1.0)) if isinstance(multipliers, dict) else 1.0
                # Appendix B, Eq. (B.6): occupation-specific AI separation increment.
                ai_pressure = float(f.rho_adopt) * float(occ.get("ai_exposure", 0.0)) * float(occ.get("substitution", 0.0)) * sub_mult
                ai_buffer = float(f.rho_adopt) * float(occ.get("complementarity", 0.0)) * max(0.0, float(w.a))
                p += float(self.sp.occupation_ai_sep_scale) * max(0.0, ai_pressure - 0.5 * ai_buffer)
            # Appendix B, Eq. (B.7): bound total baseline-plus-AI separation probability.
            p = float(np.clip(p, 0.0, float(self.sp.sep_max)))

            if self.rnd.random() < p:
                f.L = max(0, int(f.L) - 1)
                w.employed = False  # Eq. (A.65): status set to be unemployed
                w.firm_id = None
                w.w = 0.0   # Eq. (A.65): realized wage is reset to 0
                # Start the within-period spell at zero; Eq. (A.65a) records
                # duration one at period end if the worker is not rehired.
                w.u_dur = 0
                n_sep += 1
                if getattr(self.sp, "enable_occupation_layer", False):
                    occ_sep[getattr(w, "occupation", "general")] = occ_sep.get(getattr(w, "occupation", "general"), 0) + 1
        self.last_separations = int(n_sep)
        self.last_occ_separations = occ_sep

    def worker_state_updates(self, avg_w: float) -> None:
        """End-of-period worker state updates (reservation wage, skill, duration)."""
        for w in self.workers:
            w.update_reservation_wage(float(avg_w), self.lp)
            w.update_skill(self.lp, self.rnd)
            # Appendix A, Eq. (A.65a): end-of-period unemployment-duration
            # transition. A new spell enters this step with u_dur = 0.
            if w.employed:
                w.u_dur = 0
            else:
                w.u_dur = int(w.u_dur) + 1

    def unemployment_rate(self) -> float:
        """Share of workers unemployed at the current point in the period."""
        return float(sum(1 for worker in self.workers if not worker.employed)) / max(1, len(self.workers))

    def employed_mean_wage(self) -> float:
        """Mean wage among currently employed workers."""
        wages = [float(worker.w) for worker in self.workers if worker.employed and worker.firm_id is not None]
        return float(np.mean(wages)) if wages else float(self.lp.w0)

    def worker_updates(self):
        """Backward-compatible wrapper. Prefer explicit stage calls."""
        avg_w = self.history["W_mean"][-1] if self.history["W_mean"] else self.lp.w0
        self.worker_separations_only()
        self.worker_state_updates(avg_w)


    def simulate(self) -> pd.DataFrame:
        self.initialize_agents()
        for t in range(self.sp.T):
            # Stage 0: realize macro shock
            self.update_macro_shock()
            self.apply_scheduled_shocks(t)
            self.update_resource_capacity()
            self.update_ai_cost(t)
            self.update_compute_unit_cost(t)

            # Stage 1: separations/layoffs based on most recently realized profits (t−1),
            # so job losses enter the unemployment pool before matching (Option A timing).
            self.worker_separations_only()
            self.merger_acquisition_dynamics()

            # Stage 2: technology and adoption updates
            pp_vec = self.technology_updates()
            self.adoption_updates(pp_vec)

            # Stage 3: investment and labor planning (sets desired_L for vacancy posting)
            self.investment_and_planning()

            # Stage 4: labor market (vacancies -> matching -> bargaining -> hiring)
            _, U, W_mean = self.labor_market()

            # Stage 5: production and profits (uses updated employment after hiring)
            Y = self.production_and_profits()

            # Stage 6: firm selection/exit
            n_exit = self.evolutionary_selection()
            # End-of-period unemployment includes layoffs caused by firm exit
            # before policy feedback and history recording. Recompute W_mean
            # from the same end-of-period worker state so wage diagnostics and
            # reservation-wage updates are aligned with exit/reallocation.
            U = self.unemployment_rate()
            W_mean = self.employed_mean_wage()

            # Stage 7: policy and interest rate updates
            self.policy_feedback(Y)
            self.update_interest_rate(Y)

            # Stage 8: end-of-period worker state updates (reservation wage, skills, durations)
            self.worker_state_updates(avg_w=W_mean)

            adopt_mean = float(np.mean([f.rho_adopt for f in self.firms]))
            routine_mean = float(np.mean([f.routine_intensity for f in self.firms])) if self.firms else 0.0
            self.history["U"].append(U)
            self.history["Y"].append(Y)
            self.history["W_mean"].append(W_mean)
            self.history["MacroZ"].append(float(self.macro_z))
            self.history["Shock_productivity"].append(float(self.active_productivity_shock))
            self.history["Shock_demand"].append(float(self.active_demand_shock))
            self.history["Shock_rate"].append(float(self.active_rate_shock))
            self.history["Adopt_mean"].append(adopt_mean)
            self.history["Routine_mean"].append(routine_mean)
            self.history["Debt"].append(float(self.gov.B))
            self.history["G"].append(float(self.gov.G))
            self.history["S"].append(float(self.gov.S))
            self.history["Ineq"].append(float(getattr(self, "last_ineq", 0.0)))
            self.history["Ineq_all_workers"].append(float(getattr(self, "last_ineq_all_workers", 0.0)))
            self.history["Firm_exits"].append(float(n_exit))
            # Raw exit count remains in Firm_exits. This rate aligns the
            # diagnostic moment with empirical firm-exit rates.
            self.history["Firm_exit_rate"].append(float(n_exit) / max(1, len(self.firms)))
            self.history["Exit_prob_mean"].append(float(getattr(self, "last_exit_prob_mean", 0.0)))
            self.history["Exit_profit_shortfall_mean"].append(float(getattr(self, "last_exit_profit_shortfall_mean", 0.0)))
            self.history["Exit_below_profit_cutoff_share"].append(float(getattr(self, "last_exit_below_profit_cutoff_share", 0.0)))
            self.history["Exit_below_A_min_share"].append(float(getattr(self, "last_exit_below_A_min_share", 0.0)))
            self.history["Firm_profit_mean"].append(float(getattr(self, "last_firm_profit_mean", 0.0)))
            self.history["Firm_profit_p10"].append(float(getattr(self, "last_firm_profit_p10", 0.0)))
            self.history["Firm_profit_margin_mean"].append(float(getattr(self, "last_firm_profit_margin_mean", 0.0)))
            self.history["Entry_vacancies"].append(float(getattr(self, "last_entry_vacancies", 0)))
            self.history["Exit_layoffs"].append(float(getattr(self, "last_exit_layoffs", 0)))
            self.history["Exit_reallocated_workers"].append(float(getattr(self, "last_exit_reallocated_workers", 0)))
            self.history["Exit_unabsorbed_workers"].append(float(getattr(self, "last_exit_unabsorbed_workers", 0)))
            self.history["Separations"].append(float(getattr(self, "last_separations", 0)))
            self.history["Hires"].append(float(getattr(self, "last_hires", 0)))
            self.history["Vacancies"].append(float(getattr(self, "last_vacancies", 0)))
            self.history["eta_G"].append(float(self.pp.eta_G))
            self.history["eta_S"].append(float(self.pp.eta_S))
            self.history["tax_rate"].append(float(self.pp.tax_rate))
            self.history["r_debt"].append(float(self.pp.r_debt))
            self.history["Debt_to_GDP"].append(float(getattr(self, "_last_debt_ratio", self.gov.B)))
            self.history["Credit"].append(float(self.credit_factor()))
            self.history["Resource_capacity"].append(float(self.resource_capacity))
            self.history["Resource_demand"].append(float(getattr(self, "last_resource_demand", 0.0)))
            self.history["Resource_constraint"].append(float(getattr(self, "last_resource_constraint", 1.0)))
            self.history["MA_events"].append(float(getattr(self, "last_ma_events", 0)))
            self.history["MA_layoffs"].append(float(getattr(self, "last_ma_layoffs", 0)))
            self.history["AI_cost"].append(float(getattr(self, "last_ai_effective_cost", 0.0)))
            self.history["AI_relative_cost_mean"].append(float(getattr(self, "last_ai_relative_cost_mean", 0.0)))
            self.history["AI_operating_cost"].append(float(getattr(self, "last_ai_operating_cost", 0.0)))
            self.history["Compute_demand"].append(float(getattr(self, "last_compute_demand", 0.0)))
            self.history["Compute_unit_cost"].append(float(getattr(self, "last_compute_unit_cost", 0.0)))
            self.history["Compute_operating_cost"].append(float(getattr(self, "last_compute_operating_cost", 0.0)))
            self.history["Pilot_share"].append(float(getattr(self, "last_pilot_share", 0.0)))
            self.history["Production_share"].append(float(getattr(self, "last_production_share", 0.0)))
            self.history["Scaled_share"].append(float(getattr(self, "last_scaled_share", 0.0)))
            self.history["Failed_pilots"].append(float(getattr(self, "last_failed_pilots", 0)))
            self.history["Data_readiness_mean"].append(float(getattr(self, "last_data_readiness_mean", 0.0)))
            self.history["AI_error_events"].append(float(getattr(self, "last_ai_error_events", 0)))
            self.history["AI_operational_loss"].append(float(getattr(self, "last_ai_operational_loss", 0.0)))
            self.history["AI_error_rate_mean"].append(float(getattr(self, "last_ai_error_rate_mean", 0.0)))
            self.history["Cyber_privacy_incidents"].append(float(getattr(self, "last_cyber_privacy_incidents", 0)))
            self.history["Cyber_privacy_loss"].append(float(getattr(self, "last_cyber_privacy_loss", 0.0)))
            self.history["Cyber_privacy_risk_rate_mean"].append(float(getattr(self, "last_cyber_privacy_risk_rate_mean", 0.0)))
            self.history["Cyber_privacy_control_cost"].append(float(getattr(self, "last_cyber_privacy_control_cost", 0.0)))
            self.history["AI_supervision_vacancies"].append(float(getattr(self, "last_ai_supervision_vacancies", 0)))
            self.history["AI_complementary_hires"].append(float(getattr(self, "last_ai_complementary_hires", 0)))
            self.history["AI_complementarity_premium"].append(float(getattr(self, "last_ai_complementarity_premium", 0.0)))
            self.history["New_task_share"].append(float(getattr(self, "last_new_task_share", 0.0)))
            if getattr(self.sp, "enable_occupation_layer", False):
                occ_stats = self._occupation_stats()
                for occ_id in self.occupation_ids():
                    u_occ, w_occ = occ_stats.get(occ_id, (0.0, 0.0))
                    # Appendix B, Eqs. (B.8)-(B.9): occupation unemployment
                    # and employed-worker mean wage diagnostics.
                    self.history[f"Occ_U_{occ_id}"].append(float(u_occ))
                    self.history[f"Occ_W_{occ_id}"].append(float(w_occ))
                    self.history[f"Occ_Sep_{occ_id}"].append(float(self.last_occ_separations.get(occ_id, 0)))
        return pd.DataFrame(self.history)

    # ---------------------------

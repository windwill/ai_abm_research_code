"""Agent state and behavioral rules for workers, firms, technology suppliers, and government."""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from .parameters import LaborParams, PolicyParams, SimulationParams, TechParams
from .utils import logistic

# ---------------------------
# Eq. (A.1): Worker state vector / attributes
class Worker:
    __slots__ = ("s","a","w_r","w","employed","firm_id","in_training","u_dur","occupation","s_cog","s_soc","s_tech","s_man","s_manual")
    def __init__(
        self,
        s: float,
        a: float,
        w_r: float,
        employed: bool,
        firm_id: Optional[int] = None,
        u_dur: int = 0,
        w: float = 0.0,
        occupation: str = "general",
        skill_vector: Optional[dict | list | tuple | np.ndarray] = None,
    ):
        self.s = float(np.clip(s,0,1)); self.a = float(np.clip(a,0,1)); self.w_r = float(w_r); self.w = float(w)
        self.employed = bool(employed); self.firm_id = firm_id; self.in_training = False; self.u_dur = int(u_dur)
        self.occupation = str(occupation)
        # Appendix B, Eq. (B.3): worker multidimensional skill vector s_i.
        if isinstance(skill_vector, dict):
            vals = [
                skill_vector.get("cog", self.s),
                skill_vector.get("soc", self.s),
                skill_vector.get("tech", self.s),
                skill_vector.get("man", self.s),
                skill_vector.get("manual", self.s),
            ]
        elif skill_vector is None:
            vals = [self.s] * 5
        else:
            vals = list(skill_vector)
            vals = (vals + [self.s] * 5)[:5]
        self.s_cog, self.s_soc, self.s_tech, self.s_man, self.s_manual = [float(np.clip(v, 0, 1)) for v in vals]

    @property
    def skill(self) -> float:
        """Alias for skill level s (kept for backward compatibility)."""
        return self.s

    @skill.setter
    def skill(self, value: float) -> None:
        self.s = float(np.clip(value, 0, 1))

    @property
    def skill_vector(self) -> np.ndarray:
        """Five-dimensional skill vector used by the optional occupation layer."""
        return np.array([self.s_cog, self.s_soc, self.s_tech, self.s_man, self.s_manual], dtype=float)

    def update_reservation_wage(self, observed_peer_wage: float, lp: LaborParams):
        # Eq. (A.2): Reservation wage updating
        self.w_r = self.w_r + lp.lambda_w * (observed_peer_wage - self.w_r)
    def reskilling_decision(self, expected_post_wage: float, lp: LaborParams, rnd: np.random.Generator, training_cost_eff: Optional[float] = None) -> bool:
        C_eff = lp.C_T if training_cost_eff is None else float(training_cost_eff)
        # Eq. (A.7): Training probability (logit)
        x = lp.gamma_T * (expected_post_wage - self.w_r - C_eff); p = logistic(x)
        self.in_training = rnd.random() < p; return self.in_training
    def update_skill(self, lp: LaborParams, rnd: np.random.Generator, sigma_s: float = 0.01):
        if self.in_training:
            # Eq. (A.8): Skill accumulation update
            ds = self.a * (1.0 - self.s) + rnd.normal(0.0, sigma_s)
            self.s = float(np.clip(self.s + ds, 0.0, 1.0)); self.in_training = False

# Eq. (A.12): Technology firm state vector / attributes
class TechFirm:
    __slots__=("pi_k","price","rho_R","mu_markup")
    def __init__(self, initial_pi: float, price: float, rho_R: float, mu_markup: float):
        self.pi_k=initial_pi; self.price=price; self.rho_R=rho_R; self.mu_markup=mu_markup
    # Eq. (A.13): Innovation dynamics
    def innovate(self, demand_index: float, tp: TechParams, rnd: np.random.Generator):
        shock = rnd.normal(0.0, tp.sigma_R); self.pi_k = max(1e-4, self.pi_k * (1.0 + tp.alpha_R * self.rho_R + shock))
    def set_price(self, cost: float, demand_gap: float, tp: TechParams, rnd: np.random.Generator):
        """Set technology price using demand-sensitive markup adjustment.

        Markups adapt to realized demand relative to a target.
        The firm updates its own markup state and prices at (1+mu)*cost.
        """
        # adaptive markup update (bounded)
        # Eq. (A.16): Markup update (clipped)
        shock = rnd.normal(0.0, tp.sigma_mu);
        self.mu_markup = float(np.clip(self.mu_markup + tp.mu_lr * demand_gap + shock, tp.mu_min, tp.mu_max))
        self.price = (1.0 + self.mu_markup) * max(1e-4, cost)

# Eq. (A.17): Adopting firm state vector / attributes
class Firm:
    __slots__=("A","L","kappa","rho_adopt","profit","last_output","last_wage_bill","desired_L","id","req_skill","routine_intensity","pi_exp","req_cog","req_soc","req_tech","req_man","req_manual","deployment_stage","data_readiness")
    def __init__(self, firm_id: int, A: float, L: int, kappa: float, rho_adopt: float, req_skill: float, routine_intensity: float = 0.5, req_skill_vector: Optional[dict | list | tuple | np.ndarray] = None, data_readiness: float = 1.0):
        # rho_adopt is the continuous AI adoption stock rho_j,t in [0, 1]:
        # 0 denotes no adoption and 1 denotes saturation of the modeled scope.
        self.id=firm_id; self.A=A; self.L=L; self.kappa=kappa; self.rho_adopt=rho_adopt; self.req_skill=float(req_skill); self.routine_intensity=float(np.clip(routine_intensity,0,1)); self.profit=0.0; self.last_output=0.0; self.last_wage_bill=0.0; self.pi_exp=0.0; self.desired_L=L
        # Appendix B: z_j,t is the discrete deployment stage and d_j is the
        # bounded firm-level data-readiness state used in Eqs. (B.24)-(B.27).
        self.deployment_stage = 3 if self.rho_adopt >= 0.75 else (2 if self.rho_adopt > 0.0 else 0)
        self.data_readiness = float(np.clip(data_readiness, 0.0, 1.0))
        # Appendix B, Eq. (B.4): firm required skill vector q_j.
        if isinstance(req_skill_vector, dict):
            vals = [
                req_skill_vector.get("cog", self.req_skill),
                req_skill_vector.get("soc", self.req_skill),
                req_skill_vector.get("tech", self.req_skill),
                req_skill_vector.get("man", self.req_skill),
                req_skill_vector.get("manual", self.req_skill),
            ]
        elif req_skill_vector is None:
            vals = [self.req_skill] * 5
        else:
            vals = list(req_skill_vector)
            vals = (vals + [self.req_skill] * 5)[:5]
        self.req_cog, self.req_soc, self.req_tech, self.req_man, self.req_manual = [float(np.clip(v, 0, 1)) for v in vals]
    @property
    def req_skill_vector(self) -> np.ndarray:
        """Five-dimensional skill requirement used by the optional occupation layer."""
        return np.array([self.req_cog, self.req_soc, self.req_tech, self.req_man, self.req_manual], dtype=float)
    def ces_output(self, avg_skill: float, tp: TechParams, sigma: float = 0.5, alpha_L: float = 0.6, alpha_A: float = 0.4, nu: float = 0.0) -> float:
        # Eq. (A.18): Effective labor definition in production
        L_eff = max(0.0, self.L) * max(0.2, avg_skill)
        term = alpha_L * (L_eff ** sigma) + alpha_A * (self.kappa * self.A ** sigma)
        # Eq. (A.19): CES production function
        y = (self.A ** nu) * (term ** (1.0 / sigma))
        return float(max(0.0, y))
    def adoption_probability(self, expected_delta_prod: float, price_perf: float, Theta_star: float, delta_buy: float) -> float:
        # Eq. (A.70) Innovation driven adoption propensity
        x = float(np.clip(delta_buy * (price_perf - Theta_star + expected_delta_prod), -50.0, 50.0)); return logistic(x)
    def decide_adoption(self, tp: TechParams, price_perf: float, expected_delta_prod: float, rho_bar: float, rnd: np.random.Generator, credit_factor: float = 1.0, ai_cost_penalty: float = 0.0, deployment_factor: float = 1.0):
        """Update the firm's automation adoption stock.

        Bass-style diffusion:
        - 'Innovation' component: a logit probability driven by price-performance and expected gains.
        - 'Imitation' component: adoption pressure increases with economy-wide adoption (rho_bar).

        The combined adoption probability is:
            h = clip(p_innov + tp.bass_q * rho_bar, 0, 1)

        Adoption stock is updated incrementally and capped to [0, 1].
        """
        routine = float(np.clip(getattr(self, "routine_intensity", 0.5), 0.0, 1.0))
        # Appendix B, Eq. (B.54): routine-task multiplier on the expected
        # adoption gain before the total cost/risk penalty is subtracted.
        routine_gain = 1.0 + float(getattr(tp, "routine_adoption_wedge", 0.0)) * routine
        # Appendix B, Eq. (B.18), as revised by Eq. (B.47): subtract the
        # total AI, compute, and cyber adoption penalty assembled by Economy.
        expected_delta_eff = expected_delta_prod * routine_gain - max(0.0, float(ai_cost_penalty))
        p_innov = self.adoption_probability(expected_delta_eff, price_perf, tp.Theta_star, tp.delta_buy)
        # Eq. (A.71): Adjusted adoption probability
        h = float(np.clip((p_innov + tp.bass_q * rho_bar) * float(credit_factor), 0.0, 1.0))
        # Eq. (A.72) Adoption increment
        increment = tp.beta_I * h + tp.beta_P * h * (1.0 - self.rho_adopt)
        # Eq. (A.73) Adoption update
        self.rho_adopt = float(np.clip(self.rho_adopt + increment * (1.0 - self.rho_adopt), 0.0, 1.0))
        # automation augments firm productivity in a reduced-form manner
        # Eq. (A.74) Firm productivity update
        # Appendix B, Eq. (B.55): routine-biased productivity multiplier.
        prod_gain = 1.0 + float(getattr(tp, "routine_productivity_wedge", 0.0)) * routine
        # Appendix B, Eq. (B.28): deployment-adjusted productivity gain.
        # rho_adopt has just been updated above; deployment_factor reflects
        # the stage resolved before this adoption update.
        self.A += self.rho_adopt * tp.phi_A * prod_gain * float(np.clip(deployment_factor, 0.0, 1.0))
        return h

# Eq. (A.35): Government state vector / attributes
class Government:
    __slots__=("G","tau","S","B")
    def __init__(self, G: float, tau: float, S: float, B: float):
        self.G=G; self.tau=tau; self.S=S; self.B=B
    def update_policy(self, U: float, I_ratio: float, pp: PolicyParams):
        # Eq. (A.37): Fiscal spending feedback rule
        self.G = self.G + pp.eta_G * (U - pp.U_star) + pp.eta_I * (I_ratio - pp.I_star)
        # Eq. (A.38): Reskilling subsidy feedback rule
        self.S = max(0.0, self.S + pp.eta_S * max(0.0, U - pp.U_star))


    def adapt_rule_coefficients(self, U: float, I_ratio: float, pp: PolicyParams, sp: SimulationParams):
        """Simple adaptive policy learning.

        The government gradually adjusts the responsiveness coefficients (eta_G, eta_I, eta_S)
        and the tax rate (tau) in response to persistent deviations in unemployment, inequality,
        and debt from targets.

        This is intentionally conservative and bounded to preserve numerical stability.
        """
        if not getattr(sp, "enable_policy_learning", False):
            return

        # Eq. (A.40): Gaps
        u_gap = float(U - pp.U_star)
        i_gap = float(I_ratio - pp.I_star)
        debt_gap = float(self.B - sp.policy_debt_target)

        # Increase stabilization strength when unemployment/inequality are above targets,
        # but penalize aggressive responses when debt is above target.
        # Eq. (A.41) Spending responsiveness parameter update
        pp.eta_G = float(np.clip(pp.eta_G + sp.policy_lr_G * u_gap - sp.policy_lr_debt_penalty * max(0.0, debt_gap),
                                 sp.policy_eta_G_min, sp.policy_eta_G_max))
        # Eq. (A.42) Inequality responsiveness parameter update
        pp.eta_I = float(np.clip(pp.eta_I + sp.policy_lr_I * i_gap - 0.5 * sp.policy_lr_debt_penalty * max(0.0, debt_gap),
                                 sp.policy_eta_I_min, sp.policy_eta_I_max))
        # Eq. (A.43) Reskilling responsiveness parameter update
        pp.eta_S = float(np.clip(pp.eta_S + sp.policy_lr_S * max(0.0, u_gap) - sp.policy_lr_debt_penalty * max(0.0, debt_gap),
                                 sp.policy_eta_S_min, sp.policy_eta_S_max))

        # Adjust tax rate to steer debt back toward target (slowly).
        # Eq. (A.44) Tax rate adjustment        
        pp.tax_rate = float(np.clip(pp.tax_rate + sp.policy_lr_tau * debt_gap,
                                    sp.policy_tau_min, sp.policy_tau_max))
        # Keep government object's tau synchronized
        self.tau = pp.tax_rate


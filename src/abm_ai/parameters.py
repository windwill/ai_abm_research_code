"""Dataclass definitions for labor, technology, policy, and simulation parameters."""

from __future__ import annotations

from dataclasses import dataclass, field

@dataclass
class LaborParams:
    zeta: float = 0.6
    alpha: float = 0.2
    eta_w: float = 0.3
    theta_s: float = 0.6
    w0: float = 1.0
    omega_L: float = 0.15
    lambda_w: float = 0.3
    gamma_T: float = 2.0
    C_T: float = 0.2
    # Search effort (job search intensity)
    search_min: float = 0.10  # minimum effort for unemployed workers
    search_base: float = 0.40  # baseline effort
    search_slope_dur: float = 0.05  # effort increase per period of unemployment
    search_max: float = 1.00  # maximum effort
    search_cap_dur: int = 20  # cap unemployment duration in effort function
    # Compatibility-based matching
    compat_psi: float = 6.0  # higher => stronger preference for skill-task fit
    compat_floor: float = 0.05  # floor to avoid zero weights


    # Explicit application stage (worker applies to firms)
    applications_base: int = 3  # baseline number of applications per unemployed worker per period
    applications_max: int = 8   # cap on applications per worker
    application_vacancy_weight: float = 0.5  # weight vacancies in application probabilities (0=ignore vacancies)

    # Nash-style wage bargaining (implemented as surplus split)
    bargaining_beta: float = 0.5  # worker bargaining power in [0,1]
    bargaining_firm_premium: float = 0.10  # firm willingness-to-pay premium over baseline offer

    # Numerical stabilizers (avoid division by zero)
    eps_match: float = 1e-6
    eps_vac_norm: float = 1e-9

    # Wage formation (parameterize previously hardcoded constants)
    wage_shock_std: float = 0.02
    wage_skill_floor: float = 0.20
    expected_post_wage_uplift: float = 1.10

@dataclass
class TechParams:
    alpha_R: float = 0.02
    sigma_R: float = 0.004
    mu_markup: float = 0.2
    sigma_mu: float = 0.004
    delta_price: float = 0.15  # legacy inactive field; Eq. (A.70) uses delta_buy
    Theta_star: float = 0.9
    delta_buy: float = 4.0
    beta_I: float = 0.15
    beta_P: float = 0.20
    bass_q: float = 0.25  # imitation strength for Bass-style diffusion
    phi_A: float = 0.05 # productivity augmentation coefficient
    # Appendix B, Eqs. (B.54)-(B.55): task-biased adoption and productivity wedges.
    routine_adoption_wedge: float = 0.0  # omega_r
    routine_productivity_wedge: float = 0.0  # chi_r
    # Demand-sensitive tech pricing (markup adapts to realized demand)
    tech_demand_target: float = 0.25  # target (normalized) demand for the technology
    mu_lr: float = 0.15  # learning rate for markup adjustment
    mu_min: float = 0.05  # minimum markup (fraction)
    mu_max: float = 0.60  # maximum markup (fraction)
    tech_choice_lambda: float = 6.0  # intensity of firm choice among tech suppliers
    tech_choice_noise_std: float = 0.0  # idiosyncratic taste shock std in supplier choice
    depreciation_K: float = 0.05
    # Appendix B, Eqs. (B.29)-(B.34): supplier choice utility and softmax,
    # smoothed shares, positive share advantage, R&D scaling, and markup gap.
    # Optional winner-take-more technology supplier dynamics.
    enable_winner_take_more: bool = False
    tech_share_memory: float = 0.70  # persistence of supplier share state
    tech_share_feedback: float = 1.00  # choice-utility advantage from lagged market share
    tech_rnd_share_sensitivity: float = 0.25  # R&D performance gain from lagged share
    tech_markup_share_sensitivity: float = 0.10  # markup gain from lagged share
    tech_share_floor: float = 1e-6  # numerical floor for share normalization

@dataclass
class PolicyParams:
    U_star: float = 0.055
    I_star: float = 0.35
    eta_G: float = -0.3
    eta_I: float = 0.0
    eta_S: float = 0.05
    tax_rate: float = 0.22
    r_debt: float = 0.02

@dataclass
class SimulationParams:
    T: int = 100
    N_workers: int = 5000
    N_firms: int = 200
    N_tech: int = 5

    # Firm heterogeneity (used in compatibility-based matching)
    firm_req_skill_mean: float = 0.50
    firm_req_skill_sd: float = 0.15
    firm_routine_mean: float = 0.50
    firm_routine_sd: float = 0.15

    # Appendix B, Eq. (B.1): occupation parameter vector Omega_o.
    # Each occupation spec supplies share (pi_o), scalar skill moments
    # (s_bar_o, sigma_s,o), wage multiplier, AI exposure, substitution and
    # complementarity intensities, and the skill-vector center mu_o.
    # Optional occupation layer for task- and occupation-specific AI exposure
    enable_occupation_layer: bool = False
    occupation_match_weight: float = 0.35
    occupation_ai_sep_scale: float = 0.04
    occupation_substitution_multipliers: dict = field(default_factory=dict)
    occupation_specs: list[dict] = field(default_factory=lambda: [
        {
            "id": "senior_management",
            "label": "Senior management",
            "share": 0.06,
            "skill_mean": 0.82,
            "skill_sd": 0.08,
            "wage_multiplier": 2.40,
            "routine_intensity": 0.30,
            "ai_exposure": 0.70,
            "substitution": 0.35,
            "complementarity": 0.55,
            "skill_vector": {"cog": 0.85, "soc": 0.80, "tech": 0.55, "man": 0.90, "manual": 0.10},
        },
        {
            "id": "professional_technical",
            "label": "Professional and technical",
            "share": 0.22,
            "skill_mean": 0.76,
            "skill_sd": 0.10,
            "wage_multiplier": 1.65,
            "routine_intensity": 0.35,
            "ai_exposure": 0.65,
            "substitution": 0.20,
            "complementarity": 0.65,
            "skill_vector": {"cog": 0.80, "soc": 0.55, "tech": 0.80, "man": 0.45, "manual": 0.15},
        },
        {
            "id": "clerical_admin",
            "label": "Clerical and administrative",
            "share": 0.18,
            "skill_mean": 0.55,
            "skill_sd": 0.12,
            "wage_multiplier": 1.00,
            "routine_intensity": 0.80,
            "ai_exposure": 0.75,
            "substitution": 0.65,
            "complementarity": 0.25,
            "skill_vector": {"cog": 0.55, "soc": 0.45, "tech": 0.45, "man": 0.25, "manual": 0.20},
        },
        {
            "id": "sales_service",
            "label": "Sales and service",
            "share": 0.24,
            "skill_mean": 0.50,
            "skill_sd": 0.14,
            "wage_multiplier": 0.90,
            "routine_intensity": 0.45,
            "ai_exposure": 0.45,
            "substitution": 0.25,
            "complementarity": 0.30,
            "skill_vector": {"cog": 0.45, "soc": 0.70, "tech": 0.35, "man": 0.25, "manual": 0.35},
        },
        {
            "id": "production_operations",
            "label": "Production and operations",
            "share": 0.20,
            "skill_mean": 0.48,
            "skill_sd": 0.13,
            "wage_multiplier": 0.95,
            "routine_intensity": 0.65,
            "ai_exposure": 0.40,
            "substitution": 0.35,
            "complementarity": 0.25,
            "skill_vector": {"cog": 0.40, "soc": 0.35, "tech": 0.45, "man": 0.20, "manual": 0.70},
        },
        {
            "id": "transport_logistics",
            "label": "Transportation and logistics",
            "share": 0.10,
            "skill_mean": 0.45,
            "skill_sd": 0.12,
            "wage_multiplier": 0.88,
            "routine_intensity": 0.55,
            "ai_exposure": 0.35,
            "substitution": 0.30,
            "complementarity": 0.20,
            "skill_vector": {"cog": 0.38, "soc": 0.35, "tech": 0.35, "man": 0.15, "manual": 0.75},
        },
    ])

    # Endogenous separations / layoffs
    sep_base: float = 0.005
    sep_max: float = 0.20
    sep_profit_sensitivity: float = 0.08
    sep_profit_scale: float = 1.0
    sep_credit_sensitivity: float = 0.10
    sep_macro_sensitivity: float = 0.05


    # Evolutionary selection (exit/entry)
    firm_exit_prob_base: float = 0.01
    firm_exit_profit_cutoff: float = 0.02
    firm_exit_use_margin: bool = False
    firm_exit_margin_cutoff: float = 0.05
    firm_exit_slope: float = 0.25
    firm_exit_A_min: float = 0.25
    firm_exit_prob_floor_lowA: float = 0.50
    firm_entry_A_low: float = 0.50
    firm_entry_A_high: float = 1.50
    firm_entry_L_min: int = 5
    firm_entry_L_mean: int = 20
    firm_entry_kappa_logn_mu: float = 0.0
    firm_entry_kappa_logn_sigma: float = 0.25
    firm_entry_rho_mean: float = 0.08
    firm_entry_rho_sd: float = 0.03
    enable_entry_reallocation: bool = False
    entry_reallocation_share: float = 0.30
    entry_vacancy_multiplier: float = 1.50


    firm_entry_rho_max: float = 0.3
    # Profit expectation smoothing (firms form persistent profit expectations)
    profit_exp_lambda: float = 0.25   # smoothing weight λ_π in π_exp ← (1-λ)π_exp + λπ
    profit_exp_noise_std: float = 0.0 # optional noise added to expected profit used in planning

    # Peer imitation in investment and labor planning (social learning)
    invest_imitation_xi: float = 0.20      # ξ in I ← (1-ξ)I_self + ξ I_peer
    invest_imitation_top_frac: float = 0.10 # fraction of top firms used to form peer benchmark
    invest_profit_target: float = 0.20      # profit hurdle in Eq. (A.24) and Eq. (A.26)


    
    # Adaptive policy learning (updating policy rule coefficients over time)
    enable_policy_learning: bool = True
    policy_debt_target: float = 1.0
    policy_lr_G: float = 0.05
    policy_lr_I: float = 0.02
    policy_lr_S: float = 0.02
    policy_lr_tau: float = 0.01
    policy_lr_debt_penalty: float = 0.02
    policy_eta_G_min: float = -1.5
    policy_eta_G_max: float = 0.0
    policy_eta_I_min: float = -0.5
    policy_eta_I_max: float = 0.5
    policy_eta_S_min: float = 0.0
    policy_eta_S_max: float = 0.5
    policy_tau_min: float = 0.05
    policy_tau_max: float = 0.40
    policy_G_min: float = 0.0
    policy_G_max: float = 0.35
    policy_S_max: float = 1.00
    debt_ratio_min: float = -1.00
    debt_ratio_max: float = 5.00



    # Endogenous borrowing rate and investment sensitivity (capital/finance block)
    enable_endogenous_rates: bool = True
    debt_gdp_target: float = 1.0
    r_base: float = 0.02
    r_spread_slope: float = 0.05
    r_min: float = 0.00
    r_max: float = 0.10
    y_denom_floor: float = 1.0
    inv_gamma_base: float = 0.50
    inv_rate_sensitivity: float = 8.0


    # Credit conditions / financial tightening feedback
    enable_credit_channel: bool = True
    credit_min: float = 0.30
    credit_rate_sensitivity: float = 6.0
    credit_debt_sensitivity: float = 0.75
    credit_hiring_power: float = 1.0
    credit_adopt_power: float = 1.0


    # Training subsidy channel (links government reskilling subsidy S_t to workers' effective training costs)
    enable_training_subsidy: bool = True
    training_subsidy_effect: float = 0.60   # fraction of training cost reduced at full subsidy intensity
    training_cost_floor_ratio: float = 0.05 # minimum fraction of baseline C_T that remains after subsidy
    training_subsidy_use_duration: bool = False  # if True, subsidy is stronger for longer unemployment duration
    training_subsidy_dur_scale: float = 20.0     # duration scale for targeting (periods) if enabled

    # Fiscal accounting: cost of reskilling subsidy outlays in the government budget
    subsidy_spend_scale: float = 1.0  # scales S_t * unemployment into the debt-ratio budget constraint


    seed: int = 42
    monte_carlo_R: int = 1
    # --- Macro shock process (Ext16) ---
    enable_macro_shocks: bool = True
    macro_mu: float = 1.0
    macro_rho: float = 0.85
    macro_sigma: float = 0.03
    macro_min: float = 0.7
    macro_max: float = 1.3
    # Appendix B, Eqs. (B.56)-(B.60): scheduled productivity, demand,
    # and interest-rate shock overlays.
    # Each entry may include type, start, duration, and magnitude.
    shock_schedule: list[dict] = field(default_factory=list)

    # Appendix B, Eqs. (B.10)-(B.13): AI-supporting resource capacity,
    # desired demand, the bounded capacity factor, and its adoption throttle.
    # This is framed as a physical/economic capacity limit, not a legal rule.
    enable_resource_constraint: bool = False
    resource_capacity_initial: float = 1.0
    resource_capacity_growth: float = 0.02
    resource_capacity_min: float = 1e-6
    resource_demand_per_adopter: float = 1.0
    resource_constraint_floor: float = 0.05

    # Appendix B, Eqs. (B.35)-(B.38): acquisition probability, worker
    # retention, and partial productivity and AI-adoption transfers.
    # Optional merger-and-acquisition restructuring channel.
    enable_ma_dynamics: bool = False
    ma_base_probability: float = 0.0
    ma_profit_sensitivity: float = 0.10
    ma_ai_gap_sensitivity: float = 0.10
    ma_size_sensitivity: float = 0.00
    ma_redundancy_rate: float = 0.15
    ma_productivity_transfer: float = 0.30
    ma_adoption_transfer: float = 0.30
    ma_wage_retention_factor: float = 1.0

    # Appendix B, Eqs. (B.14)-(B.18): AI cost path, scarcity markup,
    # wage-relative cost ratio, adoption penalty, and net expected gain.
    # This allows AI automation to be more expensive than human labour.
    enable_ai_relative_cost: bool = False
    ai_cost_base: float = 1.0
    ai_cost_decline: float = 0.02
    ai_cost_floor: float = 0.05
    ai_cost_implementation: float = 0.0
    ai_cost_monitoring: float = 0.0
    ai_cost_resource_sensitivity: float = 0.0
    ai_relative_cost_sensitivity: float = 1.0
    ai_operating_cost_scale: float = 0.05
    occupation_ai_cost_multipliers: dict = field(default_factory=dict)

    # Appendix B, Eqs. (B.19)-(B.23): compute workload, unit-cost path,
    # resource-sensitive unit cost, adoption penalty, and capped operating cost.
    # Optional computation-cost subchannel for AI training/inference workloads.
    enable_compute_cost: bool = False
    compute_unit_cost_base: float = 0.25
    compute_unit_cost_decline: float = 0.02
    compute_unit_cost_floor: float = 0.02
    compute_resource_sensitivity: float = 0.0
    compute_intensity_per_adoption: float = 1.0
    compute_training_share: float = 0.20
    compute_inference_scale: float = 1.0
    compute_adoption_cost_sensitivity: float = 0.50
    compute_cost_cap_share: float = 0.50

    # Appendix B, Eqs. (B.24)-(B.28): pilot entry, pilot success/failure,
    # the stage-dependent deployment factor, and realized productivity gain.
    # Optional pilot-to-production deployment friction and data readiness.
    enable_deployment_friction: bool = False
    data_readiness_mean: float = 0.65
    data_readiness_sd: float = 0.15
    pilot_start_scale: float = 0.60
    pilot_success_base: float = 0.25
    pilot_success_data_weight: float = 0.55
    pilot_success_profit_weight: float = 0.20
    pilot_failure_base: float = 0.08
    pilot_failure_data_weight: float = 0.30
    pilot_cost: float = 0.02
    production_scale_rate: float = 0.20
    scaled_deployment_threshold: float = 0.75

    # Appendix B, Eqs. (B.39)-(B.41): AI-error probability, Bernoulli
    # occurrence, and capped operational-loss severity.
    # Optional AI error and operational loss channel.
    enable_ai_operational_risk: bool = False
    ai_error_base: float = 0.005
    ai_error_data_sensitivity: float = 0.05
    ai_error_stage_sensitivity: float = 0.02
    ai_error_adoption_sensitivity: float = 0.03
    ai_loss_severity: float = 0.03
    ai_loss_noise: float = 0.00
    ai_loss_cap_share: float = 0.25

    # Appendix B, Eqs. (B.42)-(B.48): cyber/privacy incident probability,
    # occurrence, controls, capped loss, anticipated adoption friction,
    # total adoption penalty, and the extended profit equation.
    # Optional cybersecurity and data privacy operational-risk channel.
    # This is framed as operational resilience and data-governance risk, not legal liability.
    enable_cyber_privacy_risk: bool = False
    cyber_incident_base: float = 0.002
    cyber_incident_data_sensitivity: float = 0.040
    cyber_incident_adoption_sensitivity: float = 0.030
    cyber_incident_stage_sensitivity: float = 0.010
    cyber_incident_concentration_sensitivity: float = 0.000
    cyber_control_strength: float = 0.50
    cyber_control_cost_scale: float = 0.010
    cyber_adoption_friction: float = 0.50  # eta_cyb in Eq. (B.46)
    cyber_loss_severity: float = 0.040
    cyber_loss_noise: float = 0.00
    cyber_loss_cap_share: float = 0.30

    # Appendix B, Eqs. (B.49)-(B.53): complementary new-task intensity,
    # supervision vacancies, worker score, match bonus, and wage premium.
    # Optional human-AI complementarity and new-task creation channel.
    enable_ai_complementarity: bool = False
    ai_supervision_demand_scale: float = 0.10
    ai_new_task_threshold: float = 0.30
    ai_new_task_data_weight: float = 0.50
    ai_complementarity_match_bonus: float = 0.50
    ai_complementarity_wage_premium: float = 0.05
    ai_complementarity_adapt_weight: float = 0.60
    ai_complementarity_skill_weight: float = 0.40
    ai_supervision_skill_threshold: float = 0.55


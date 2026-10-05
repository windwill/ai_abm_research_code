# Data Dictionary

## Inputs

- `configs/selected_baseline.json`: calibrated parameters and simulation settings
- `data/calibration_targets.csv`: calibration targets, weights, diagnostic targets, units, transformations, and sources
- `configs/actuarial_translation_assumptions.json`: assumptions used for the report level actuarial indicators
- `scenarios/scenario_matrix.json`: combinations used in the report scenario analysis
- `data/empirical/raw`: public empirical series retained for traceability

## Core Outputs

- `U`: aggregate unemployment rate
- `Y`: output index
- `Y_growth`: output growth over the evaluation window
- `Debt_to_GDP`: public debt to output ratio
- `Adopt_mean`: mean AI adoption share
- `W_mean`: mean wage index
- `Ineq_mean`: employed worker wage inequality
- `Ineq_all_workers`: current labor income inequality including unemployed workers at zero current labor income

## Scenario Extensions

- `Compute_operating_cost`: modeled compute expenditure
- `AI_operational_loss`: modeled loss from AI errors and failures
- `Cyber_privacy_loss`: modeled cyber and privacy loss
- `Tech_share_hhi`: technology supplier concentration measured using the Herfindahl Hirschman Index
- `AI_supervision_vacancies`: vacancies created for AI supervision and complementary work
- `AI_complementary_hires`: completed hires into complementary work
- `MA_layoffs`: modeled layoffs associated with restructuring
- `Occ_U_*`: unemployment rate by modeled occupation group

The model uses normalized units for several financial quantities. Consult the report before interpreting a level as a monetary amount.

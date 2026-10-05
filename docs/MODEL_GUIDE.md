# Model Guide

The model contains workers, adopting firms, technology suppliers, and a government sector. Workers differ in skills, wages, unemployment duration, occupation, and adaptability. Firms make hiring, production, investment, and AI adoption decisions. Technology suppliers invest in technology quality and set markups. Fiscal policy responds to unemployment, inequality, output, and public debt.

AI adoption can change productivity, labor demand, task composition, operating cost, and exposure to operational and cyber losses. Optional modules represent resource constraints, deployment readiness, supplier concentration, restructuring, human and AI complementarity, occupation specific automation, and scheduled macroeconomic stress.

The selected configuration in `configs/selected_baseline.json` provides the common reference environment discussed in the report. The principal calibration moments are unemployment, output growth, debt to output, all worker inequality, and final AI adoption. Wage level, employed worker inequality, and firm exits are retained as diagnostics rather than principal fit criteria.

Important aggregate outputs include `U`, `Y`, `Y_growth`, `Debt_to_GDP`, `Adopt_mean`, `W_mean`, and `Ineq_all_workers`. Extension outputs include occupation unemployment, compute operating cost, AI operational loss, cyber and privacy loss, supplier concentration, complementary hiring, and restructuring layoffs.

One model period is an abstract simulation period. Results are most useful as relative paths and scenario contrasts rather than as calendar dated forecasts.

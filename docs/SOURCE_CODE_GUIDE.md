# Source Code Guide

## Program Flow

The package follows a small number of recurring steps:

1. `config.py` reads a JSON or YAML configuration into the parameter dataclasses in `parameters.py`.
2. `simulation.py` creates an `Economy`, initializes the agents from `agents.py`, advances the system one period at a time, and returns a pandas DataFrame containing the simulated history.
3. `moments.py` converts that history into aggregate calibration and diagnostic measures.
4. `scenarios.py` applies one or more JSON overlays, runs independent replications, and writes scenario summaries.
5. Optional modules perform calibration, uncertainty diagnostics, validation, visualization, and report assembly.

The main command `python -m abm_ai` enters through `__main__.py` and `cli.py`. Scenario commands enter through `python -m abm_ai.scenarios`.

## Core Modules

| Module | Main responsibility | Useful entry points |
| --- | --- | --- |
| `parameters.py` | Defines labor, technology, policy, and simulation settings as dataclasses. | `LaborParams`, `TechParams`, `PolicyParams`, `SimulationParams` |
| `agents.py` | Defines worker, adopting firm, technology supplier, and government state and behavior. | `Worker`, `Firm`, `TechFirm`, `Government` |
| `simulation.py` | Coordinates initialization, matching, production, adoption, investment, fiscal updates, extension mechanisms, and period output. | `Economy(...).simulate()` |
| `config.py` | Loads configurations, calibration targets, covariance matrices, and target groups. | `load_config`, `load_targets` |
| `moments.py` | Summarizes histories and evaluates simulated method of moments loss. | `summarize_moments`, `msm_loss`, `dynamic_diagnostic_moments` |
| `scenarios.py` | Loads, composes, and applies overlays and manages replicated scenario batches. | `load_scenario_overlay`, `compose_scenario`, `run_scenario_replications`, `run_scenario_batch` |

## Analytical Modules

| Module | Main responsibility |
| --- | --- |
| `calibration.py` | Parameter-vector conversion, replicated objective evaluation, genetic algorithm, CMA-ES, simulated annealing, OFAT, and Sobol-style screening. |
| `diagnostics.py` | MCMC and SMC uncertainty screens, posterior predictive simulation, convergence summaries, and diagnostic plots. |
| `validation.py` | Target error tables, aggregate validation measures, and distributional tests. |
| `empirical_data.py` | Downloads and transforms supported public series into annual data and calibration targets. Internet access is required when refreshing sources. |
| `harmonization.py` | Aggregates long-format empirical observations and constructs annual target files. |
| `baseline_outputs.py` | Runs a configured reference simulation and performs basic numerical sanity checks. |
| `reporting.py` | Produces standard plots and PDF summaries from simulation results. |
| `visualization.py` | Produces calibration, scenario, debt, and occupation visualizations from CSV inputs. |
| `target_files.py` | Combines compatible calibration-target files. |
| `utils.py` | Supplies random-number and numerical utility functions. |

## Working With the Model in Python

```python
from dataclasses import replace

from abm_ai.config import load_config
from abm_ai.moments import summarize_moments
from abm_ai.simulation import Economy

labor, technology, policy, simulation = load_config("configs/selected_baseline.json")
simulation = replace(simulation, T=20, seed=123)

history = Economy(labor, technology, policy, simulation).simulate()
moments = summarize_moments(history)
print(moments["U_mean"])
```

`Economy.simulate()` returns one row per period. The columns contain aggregate macroeconomic outcomes and any active extension diagnostics. `summarize_moments()` converts the path into scalar measures. Users who need a custom output should generally derive it from the history DataFrame rather than modify the simulation loop.

## Parameters and Overlays

The dataclasses in `parameters.py` are the authoritative list of supported settings. A configuration may specify any of those fields under `LaborParams`, `TechParams`, `PolicyParams`, and `SimulationParams`. Unknown saved fields are ignored when configurations are loaded, which permits older result files to be read without changing the active model.

Scenario overlays use the same group and field names. `compose_scenario()` deep copies the reference parameters before applying changes, so the input configuration is not mutated. Combined scenarios are applied in the order supplied. When two overlays change the same field, the later overlay takes precedence.

## Randomness and Replications

All stochastic behavior is generated from the simulation seed. Scenario batches derive a deterministic sequence of replication seeds from the requested master seed. This supports like-for-like comparisons, but it does not make a short run statistically representative. Use multiple replications for substantive comparisons and retain the replication-level results when assessing variation.

## Extending the Code

Add a structural parameter to the appropriate dataclass before using it in an agent or in `Economy`. Add its configuration value to the JSON file and, if it is calibrated, add the parameter to the bounds and vector mappings in `calibration.py`. New output series should be recorded in the period row assembled by `Economy`, then summarized in `moments.py` if they are needed as scalar results.

New scenario mechanisms should normally be implemented as dormant parameters whose reference values leave behavior unchanged. A scenario JSON file can then activate the mechanism without creating a separate simulation branch. Tests should cover configuration loading, a short finite simulation, and overlay application.

## Numerical Interpretation

The model uses an abstract period and normalized monetary quantities. Some outputs are indexes, ratios, or model units rather than directly observed currency amounts. Consult `docs/DATA_DICTIONARY.md` before interpreting a field. Nonlinear thresholds can produce large changes once a constraint becomes binding, so the simulated path and the activated parameters should be inspected together.

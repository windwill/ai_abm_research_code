# AI Transition Agent Based Model

This repository accompanies the research report *Evaluating the Effects of AI Automation on Employment and Investment with Agent-Based Models*.

It provides an educational implementation of the agent based model described in the report. Readers can inspect how workers, firms, technology suppliers, government policy, AI adoption, labor adjustment, deployment costs, operational risk, and market structure interact in the model.

The repository is designed for learning, inspection, and exploratory simulation. Published numerical results remain part of the accompanying report and its analytical record.

## Installation

Python 3.13 is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

On macOS or Linux, activate the environment with `source .venv/bin/activate`.

## Start Here

Run a short example to confirm that the model is installed correctly:

```powershell
python run_scripts\run_quick_example.py
```

The example uses a reduced horizon and writes results to `outputs/quick_example`. It is intended to demonstrate the program with modest computing requirements.

To explore one scenario overlay:

```powershell
python -m abm_ai.scenarios --config configs/selected_baseline.json --scenario scenarios/definitions/A2_rapid_automation.json --replications 1 --T 30 --outdir outputs/rapid_automation_example
```

To explore the complete scenario set with lightweight settings:

```powershell
python run_scripts\run_scenario_matrix.py --replications 1 --T 30
```

Because the model is stochastic, individual histories can differ materially. Exploratory comparisons should use common horizons, consistent seed schedules, and multiple replications where runtime permits.

## Repository Guide

- `src/abm_ai`: model implementation and analytical utilities
- `configs`: selected reference configuration and actuarial translation assumptions
- `data`: calibration targets, empirical series, and source notes
- `scenarios`: scenario matrix and parameter overlays
- `run_scripts`: short examples and optional analytical workflows
- `docs`: model, source code, data, and scenario guides
- `tests`: small installation and interface checks

Begin with `docs/MODEL_GUIDE.md` for the economic structure and `docs/SOURCE_CODE_GUIDE.md` for the Python implementation.

## Appropriate Use

The selected configuration provides a common reference point for comparative experimentation. It is not a forecast. The actuarial translation is an illustrative screening layer and does not replace organization specific pricing, reserving, capital, pension valuation, economic scenario generation, or asset and liability models.

# Scenario Guide

Scenario files are parameter overlays applied to `configs/selected_baseline.json`. Each overlay changes a limited set of assumptions so readers can examine a particular mechanism while leaving the rest of the model unchanged.

The scenario families cover automation speed and task exposure, labor adjustment and training, adoption feasibility, compute cost, operational and cyber risk, supplier concentration, restructuring, human and AI complementarity, occupation exposure, and macroeconomic stress. `scenarios/scenario_matrix.json` provides a set of illustrative combinations discussed in the research.

Run one overlay with `python -m abm_ai.scenarios --scenario <file>`. Repeat `--scenario` to combine overlays. The wrapper `run_scripts/run_scenario_matrix.py` runs the supplied matrix with a user selected horizon and replication count.

Each replication uses a different deterministic seed. More replications usually provide more stable averages, but they also increase runtime approximately in proportion to the replication count. The supplied wrapper defaults to lightweight educational settings. A single run should be read as an example path rather than as an estimate of the expected outcome.

To create a scenario, copy an existing JSON overlay, assign a descriptive identifier and name, and change only the parameters needed for the mechanism under study. Compare scenarios using the same horizon, population size, replication count, and seed schedule. Document the economic interpretation of each change, especially when an overlay activates a threshold, scheduled shock, or nonlinear cost channel.

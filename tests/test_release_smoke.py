import json
from pathlib import Path

from abm_ai.config import load_config
from abm_ai.simulation import Economy


ROOT = Path(__file__).resolve().parents[1]


def test_selected_baseline_loads():
    lp, tp, pp, sp = load_config(str(ROOT / "configs" / "selected_baseline.json"))
    assert sp.N_workers == 10000
    assert sp.N_firms == 200
    assert sp.N_tech == 5


def test_short_simulation_runs():
    lp, tp, pp, sp = load_config(str(ROOT / "configs" / "selected_baseline.json"))
    sp.T = 2
    sp.N_workers = 100
    sp.N_firms = 10
    sp.N_tech = 2
    sp.seed = 123
    frame = Economy(lp, tp, pp, sp).simulate()
    assert len(frame) == 2
    assert {"U", "Y", "Debt_to_GDP", "Adopt_mean"}.issubset(frame.columns)


def test_scenario_matrix_paths_exist():
    matrix = json.loads((ROOT / "scenarios" / "scenario_matrix.json").read_text(encoding="utf-8"))
    for stack in matrix["scenarios"]:
        for item in stack:
            assert (ROOT / item).exists()

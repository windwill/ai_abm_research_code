from pathlib import Path
import argparse
import subprocess
import sys


parser = argparse.ArgumentParser(description="Explore the supplied scenario matrix.")
parser.add_argument("--replications", type=int, default=1)
parser.add_argument("--T", type=int, default=30)
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--outdir", type=Path, default=Path("outputs/scenario_matrix_example"))
parser.add_argument("--no-history", action="store_true")
args = parser.parse_args()

root = Path(__file__).resolve().parents[1]
outdir = args.outdir if args.outdir.is_absolute() else root / args.outdir
command = [
    sys.executable,
    "-m",
    "abm_ai.scenarios",
    "--config",
    str(root / "configs" / "selected_baseline.json"),
    "--matrix",
    str(root / "scenarios" / "scenario_matrix.json"),
    "--targets",
    str(root / "data" / "calibration_targets.csv"),
    "--replications",
    str(args.replications),
    "--T",
    str(args.T),
    "--seed",
    str(args.seed),
    "--outdir",
    str(outdir),
]
if args.no_history:
    command.append("--no-history")
subprocess.run(command, check=True, cwd=root)
print(f"Scenario outputs written to {outdir}")

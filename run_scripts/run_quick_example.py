from pathlib import Path
import subprocess
import sys


root = Path(__file__).resolve().parents[1]
outdir = root / "outputs" / "quick_example"
command = [
    sys.executable,
    "-m",
    "abm_ai",
    "--config",
    str(root / "configs" / "selected_baseline.json"),
    "--targets",
    str(root / "data" / "calibration_targets.csv"),
    "--outdir",
    str(outdir),
    "--T",
    "20",
    "--seed",
    "123",
    "--quiet",
]
subprocess.run(command, check=True, cwd=root)
print(f"Quick example outputs written to {outdir}")

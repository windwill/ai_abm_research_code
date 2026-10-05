"""Command-line utility for merging calibration target files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def merge_targets(inputs: list[Path], output: Path) -> None:
    """Merge target CSV files, letting later files override duplicate moments."""
    frames = []
    for path in inputs:
        df = pd.read_csv(path)
        if not {"name", "target", "weight"}.issubset(df.columns):
            raise ValueError(f"{path} must contain name,target,weight columns")
        df["_source_file"] = str(path)
        df["_order"] = range(len(df))
        frames.append(df)
    if not frames:
        raise ValueError("At least one target file is required")
    merged = pd.concat(frames, ignore_index=True)
    merged["_file_order"] = merged["_source_file"].map({str(path): i for i, path in enumerate(inputs)})
    merged = merged.sort_values(["name", "_file_order", "_order"]).drop_duplicates("name", keep="last")
    merged = merged.sort_values(["_file_order", "_order"]).drop(columns=["_source_file", "_order", "_file_order"])
    output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge ABM calibration target CSV files.")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    merge_targets(args.inputs, args.output)


if __name__ == "__main__":
    main()

"""Configuration, target, covariance, and target-group loaders."""

from __future__ import annotations

import json
import os
from dataclasses import fields
from typing import Dict, Optional, Tuple

import pandas as pd
import numpy as np

from .moments import get_default_targets
from .parameters import LaborParams, PolicyParams, SimulationParams, TechParams

try:
    import yaml  # type: ignore
    HAS_YAML = True
except Exception:
    HAS_YAML = False

def _known_dataclass_items(cls, values: Dict) -> Dict:
    """Drop obsolete config keys so older saved JSON files remain loadable."""
    allowed = {field.name for field in fields(cls)}
    return {key: value for key, value in values.items() if key in allowed}

def load_config(path: Optional[str]) -> Tuple[LaborParams, TechParams, PolicyParams, SimulationParams]:
    if not path: return LaborParams(), TechParams(), PolicyParams(), SimulationParams()
    ext=os.path.splitext(path)[1].lower()
    with open(path,"r",encoding="utf-8") as f:
        data = yaml.safe_load(f) if ext in (".yml",".yaml") and HAS_YAML else json.load(f)
    return (
        LaborParams(**_known_dataclass_items(LaborParams, data.get("LaborParams", {}))),
        TechParams(**_known_dataclass_items(TechParams, data.get("TechParams", {}))),
        PolicyParams(**_known_dataclass_items(PolicyParams, data.get("PolicyParams", {}))),
        SimulationParams(**_known_dataclass_items(SimulationParams, data.get("SimulationParams", {}))),
    )

def load_calibration_bounds(path: Optional[str]) -> Dict[str, Tuple[float, float]]:
    """Load optional config-local bounds for calibrated parameters."""
    if not path:
        return {}
    ext = os.path.splitext(path)[1].lower()
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) if ext in (".yml", ".yaml") and HAS_YAML else json.load(f)
    raw = data.get("CalibrationBounds", {}) if isinstance(data, dict) else {}
    out: Dict[str, Tuple[float, float]] = {}
    for key, bounds in raw.items():
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError(f"CalibrationBounds entry for {key} must be [low, high]")
        out[str(key)] = (float(bounds[0]), float(bounds[1]))
    return out

def _target_tuple(value) -> Tuple[float, float]:
    if isinstance(value, dict):
        return float(value["target"]), float(value.get("weight", 1.0))
    return float(value[0]), float(value[1])

def load_targets(path: Optional[str]) -> Dict[str, Tuple[float, float]]:
    calibration_targets, _ = load_target_groups(path)
    return calibration_targets

def load_moment_covariance(path: Optional[str]) -> Tuple[list[str], np.ndarray] | None:
    """Load an optional empirical moment covariance matrix.

    Supported formats:
    - Square CSV with a first column named `moment` or `name`.
    - Long CSV with `moment_i,moment_j,covariance`.
    - JSON/YAML with `{"moments": [...], "covariance": [[...], ...]}`.
    """
    if not path:
        return None
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(path)
        lower_cols = {str(c).strip().lower(): c for c in df.columns}
        if {"moment_i", "moment_j", "covariance"}.issubset(lower_cols):
            ci = lower_cols["moment_i"]
            cj = lower_cols["moment_j"]
            cv = lower_cols["covariance"]
            names = sorted(set(df[ci].astype(str)).union(set(df[cj].astype(str))))
            pos = {name: i for i, name in enumerate(names)}
            matrix = np.zeros((len(names), len(names)), dtype=float)
            for _, row in df.iterrows():
                i = pos[str(row[ci])]
                j = pos[str(row[cj])]
                matrix[i, j] = float(row[cv])
                matrix[j, i] = float(row[cv])
            return names, matrix
        first_col = str(df.columns[0]).strip().lower()
        if first_col in {"moment", "name"}:
            names = [str(x) for x in df.iloc[:, 0].tolist()]
            matrix = df.iloc[:, 1:].to_numpy(dtype=float)
            col_names = [str(c) for c in df.columns[1:]]
            if col_names != names:
                raise ValueError("Square covariance CSV column names must match moment/name rows")
            if matrix.shape != (len(names), len(names)):
                raise ValueError("Covariance matrix must be square")
            return names, matrix
        names = [str(c) for c in df.columns]
        matrix = df.to_numpy(dtype=float)
        if matrix.shape != (len(names), len(names)):
            raise ValueError("Covariance matrix must be square")
        return names, matrix
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) if ext in (".yml", ".yaml") and HAS_YAML else json.load(f)
    if not isinstance(data, dict) or "moments" not in data or "covariance" not in data:
        raise ValueError("Covariance JSON/YAML must contain `moments` and `covariance`")
    names = [str(x) for x in data["moments"]]
    matrix = np.asarray(data["covariance"], dtype=float)
    if matrix.shape != (len(names), len(names)):
        raise ValueError("Covariance matrix must be square and match moment count")
    return names, matrix

def load_target_groups(path: Optional[str]) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, Tuple[float, float]]]:
    """Load calibration and optional hold-out validation target groups.

    Backward-compatible formats:
    - CSV with columns `name,target,weight` loads all rows as calibration targets.
    - JSON/YAML mapping `{name: [target, weight]}` loads all entries as calibration targets.

    Grouped formats:
    - CSV may include `group` with values `calibration`, `validation`,
      or forward-looking calibration aliases such as `forward` or `forecast`.
    - JSON/YAML may use top-level `calibration` and `validation` mappings.
    """
    if not path:
        return get_default_targets(), {}
    ext=os.path.splitext(path)[1].lower()
    if ext==".csv":
        df=pd.read_csv(path); req={"name","target","weight"}
        if not req.issubset(df.columns): raise ValueError("CSV targets must have columns: name,target,weight")
        if "group" not in df.columns:
            return {r["name"]:(float(r["target"]), float(r["weight"])) for _,r in df.iterrows()}, {}
        calibration: Dict[str, Tuple[float, float]] = {}
        validation: Dict[str, Tuple[float, float]] = {}
        for _, r in df.iterrows():
            group = str(r.get("group", "calibration")).strip().lower()
            item = (float(r["target"]), float(r["weight"]))
            if group in {"validation", "holdout", "hold-out", "hold_out"}:
                validation[str(r["name"])] = item
            elif group in {"calibration", "target", "train", "forward", "forecast", "scenario", "view", "industry_view"}:
                calibration[str(r["name"])] = item
            else:
                raise ValueError(f"Unknown target group: {group}")
        return calibration, validation
    with open(path,"r",encoding="utf-8") as f:
        data = yaml.safe_load(f) if ext in (".yml",".yaml") and HAS_YAML else json.load(f)
    if isinstance(data, dict) and ("calibration" in data or "validation" in data):
        calibration = {k: _target_tuple(v) for k,v in (data.get("calibration") or {}).items()}
        validation = {k: _target_tuple(v) for k,v in (data.get("validation") or {}).items()}
        return calibration, validation
    return {k: _target_tuple(v) for k,v in data.items()}, {}

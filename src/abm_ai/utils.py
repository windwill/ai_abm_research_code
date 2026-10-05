"""Random-number, numerical, and lightweight document utility functions."""

from __future__ import annotations

from contextlib import contextmanager
import math
import numpy as np

def rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))

def logistic(x: float) -> float:
    """Numerically stable scalar logistic function."""
    z = float(x)
    if z >= 709.0:
        return 1.0
    if z <= -709.0:
        return 0.0
    if z >= 0.0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)

@contextmanager
def pdf_pages(path):
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(path) as pdf:
        yield pdf

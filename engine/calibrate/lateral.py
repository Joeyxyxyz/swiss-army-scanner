"""
XY calibration: camera px -> mm on the base plane, from known W / L.

Least-squares scale through the origin over every (px, mm) pair.
Build step 6 refines this (base-footprint extents for tall parts, §11).
"""

from __future__ import annotations

import numpy as np


def fit(px, mm) -> dict:
    p = np.asarray(px, float)
    m = np.asarray(mm, float)
    if p.size == 0:
        raise ValueError("no lateral samples (need scans with known W and/or L)")
    k = float(np.sum(p * m) / np.sum(m * m))
    resid = p - k * m
    return {"lateral_px_per_mm": k, "rel_rms": float(np.sqrt(np.mean((resid / p) ** 2))),
            "n_pairs": int(p.size)}

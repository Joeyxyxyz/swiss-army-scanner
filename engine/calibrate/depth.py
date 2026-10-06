"""
Z calibration: relief px -> mm of height (calibrate_depth.py's fit).

height_mm = mm_per_px * relief_px + intercept_mm. With >= 2 samples this
is calibrate_depth's least-squares line with R^2; with a single sample
the line is forced through the origin (relief 0 is the base plane by
construction) and R^2 is undefined. Valid only at the geometry the
samples came from.
"""

from __future__ import annotations

import numpy as np


def fit(relief_px, height_mm) -> dict:
    x = np.asarray(relief_px, float)
    y = np.asarray(height_mm, float)
    if x.size == 0:
        raise ValueError("no depth samples (need scans with known H)")
    if np.any(np.abs(x) < 1e-6):
        raise ValueError("a depth sample has ~0 px relief — segmentation found no "
                         "object (the '1.63 px' failure mode); re-scan it")
    if x.size == 1:
        return {"mm_per_px": float(y[0] / x[0]), "intercept_mm": 0.0, "r2": None,
                "n_samples": 1, "note": "single sample: fit through origin"}
    slope, intercept = np.polyfit(x, y, 1)
    yhat = slope * x + intercept
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) + 1e-12
    return {"mm_per_px": float(slope), "intercept_mm": float(intercept),
            "r2": 1.0 - ss_res / ss_tot, "n_samples": int(x.size)}

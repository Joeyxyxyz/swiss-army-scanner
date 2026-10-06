"""Relief = object correspondence − base correspondence (§4), per axis.

Datum handling is first_scan_v10's: pixels valid in BOTH shots only, then
subtract the median so the base plane (the field majority) sits at 0.
"""

from __future__ import annotations

import numpy as np

from .. import rules


def compute(obj, base, axis: str = "corr_x", datum: bool = True):
    """-> (relief px float32 with NaN outside, mask, datum_px).

    datum=False keeps the raw object - base difference (trusting that
    nothing moved). Zone scans need that: only the zone is lit, so when the
    part fills the zone the median of the lit pixels IS the part top and
    subtracting it would erase the height."""
    rules.require_same_geometry(obj.geometry_id, base.geometry_id, "relief")
    a, b = getattr(obj, axis), getattr(base, axis)
    if a is None or b is None:
        raise ValueError(f"relief needs {axis} in both object and base fields")
    both = obj.mask & base.mask & np.isfinite(a) & np.isfinite(b)
    if not both.any():
        return np.full(obj.shape, np.nan, np.float32), both, 0.0
    relief = np.where(both, a.astype(np.float64) - b, np.nan)
    d = float(np.nanmedian(relief)) if datum else 0.0
    return (relief - d).astype(np.float32), both, d

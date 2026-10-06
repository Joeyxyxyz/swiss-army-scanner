"""
Fringe-order (period-jump) correction.

Gray code supplies the fringe order k; the phase supplies the position
inside the fringe. When the finest gray-code stripes are at or below the
camera's resolution, bits get misread and k is off by ±1 — the decoded
column jumps by exactly one period (±24 px) while the fine phase is still
right. Bench scan 2026-09-22 (vertical stripes): 83% of non-zero relief
values were such jumps.

Most of those were fixed at the source: the polarity-aware phase (the
2026-09-22 jumps) and coarse_bits (decode only the gray planes the camera
resolves). What is left — isolated spikes where fine planes are still
misread, e.g. the full 10-bit code on the y axis at ~0.75 camera px per
DMD row — is what fix_period_jumps removes. It is OPT-IN: set
"fix_period_jumps": true on the preset's graycode (or hybrid) spec and
HybridBasis applies it to every decode of that preset (scans, bases and
both axes of a projector map), recording period_jumps_fixed_pct in the
field meta.

The correction: the column map of a surface is smooth, so each pixel's k
is snapped to agree with a local median of its neighbours:
    column -= period * round((column - local_median) / period)
Real depth steps smaller than period/2 are untouched; pixels right at a
larger depth edge can be pulled to the wrong side (they are few) — the
reason it is not on by default.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def _fill_invalid(a: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Replace invalid pixels by their nearest valid value (for filtering)."""
    idx = ndimage.distance_transform_edt(~mask, return_distances=False, return_indices=True)
    return a[tuple(idx)]


def fix_period_jumps(column: np.ndarray, mask: np.ndarray, period: float,
                     size: int = 9, iterations: int = 2, step: int = 2) -> np.ndarray:
    """-> corrected column (same shape; values outside mask untouched).

    The local median is computed on a `step`-subsampled grid for speed.
    """
    if not mask.any():
        return column
    col = column.astype(np.float64).copy()
    for _ in range(iterations):
        filled = _fill_invalid(col, mask)
        small = filled[::step, ::step]
        med = ndimage.median_filter(small, size=max(3, size // step | 1))
        med = np.repeat(np.repeat(med, step, 0), step, 1)[:col.shape[0], :col.shape[1]]
        k = np.round((col - med) / period)
        col = np.where(mask, col - period * k, col)
    return col

"""
Scan-to-golden comparison (§2 compare). Scan-to-CAD, and the port of
swiss_army_sensor.golden's tolerance machinery, land in build step 7.

Deviation = scan relief − golden relief over pixels valid in both. A
FAIL needs a connected region of at least `min_defect_px` pixels beyond
tolerance, so isolated noisy pixels don't fail a part. The deviation is
first box-averaged over `smooth_px` (NaN-aware): the per-pixel difference
of two scans carries both scans' correspondence noise, which alone can
exceed a sub-mm band.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from .. import rules
from ..results import TIER_DEPTH, TIER_RELIEF, ToleranceBand, Verdict
from .segment import BRIDGE_PX


def compare_relief(scan, golden, tol: ToleranceBand, calibration=None) -> Verdict:
    rules.require_same_geometry(scan.geometry_id, golden.geometry_id, "compare")
    warnings = []
    both = scan.relief_mask & golden.relief_mask
    dev = np.where(both, scan.relief.astype(np.float64) - golden.relief, np.nan)
    dev = _smooth(dev, both, tol.smooth_px)

    if calibration is not None and not rules.calibration_matches(calibration, scan.geometry_id):
        calibration = None
    if calibration is not None and calibration.has_z:
        dev = dev * calibration.mm_per_px     # intercept cancels in a difference
        units, tier, thr = "mm", TIER_DEPTH, tol.fail_mm
    else:
        units, tier, thr = "px", TIER_RELIEF, tol.fail_px
        warnings.append(f"no depth calibration for geometry {scan.geometry_id}; "
                        "deviation in px (rule 6)")

    n = int(both.sum())
    if n == 0:
        return Verdict("INDETERMINATE", units, tier, None, None, 0, 0, thr, dev,
                       warnings + ["scan and golden share no valid pixels"])
    a = np.abs(dev[both])
    max_dev, p99 = float(a.max()), float(np.percentile(a, 99))
    if thr is None:
        return Verdict("INDETERMINATE", units, tier, round(max_dev, 4), round(p99, 4),
                       0, n, None, dev,
                       warnings + [f"tolerance has no {units} band; cannot judge"])
    over = both & (np.abs(np.nan_to_num(dev)) > thr)
    # a region split only by thin lines of px invalid in either scan (raised
    # surfaces decode with 1-2 px invalid lines) is still one region
    link = over | (~both & ndimage.binary_dilation(over, iterations=BRIDGE_PX))
    lab, k = ndimage.label(link)
    sizes = ndimage.sum(over, lab, range(1, k + 1)) if k else np.array([])
    n_fail = int(sizes[sizes >= tol.min_defect_px].sum())
    return Verdict("FAIL" if n_fail else "PASS", units, tier, round(max_dev, 4),
                   round(p99, 4), n_fail, n, thr, dev, warnings)


def _smooth(dev, mask, size: int):
    if size <= 1:
        return dev
    w = ndimage.uniform_filter(mask.astype(np.float64), size)
    s = ndimage.uniform_filter(np.where(mask, dev, 0.0), size)
    return np.where(mask, s / np.maximum(w, 1e-12), np.nan)

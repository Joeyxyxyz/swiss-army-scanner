"""
H/W/L of the object in a scan (§2 measure).

  H: median relief over the object blob (segment.object_blob), converted
     with the depth calibration when one matches the scan's geometry.
  W/L: extents of the blob along its principal axes (so rotation doesn't
     matter), converted with the lateral calibration. W = major, L = minor.

Rule 6: without a calibration for the scan's geometry_id, values stay in
px and a warning says so — never a guessed mm.

Zone scans keep the raw object - base relief (no median datum), so the blob
is found around the base level 0 instead (segment.scan_blob); when the part
fills the zone there is no base in view and H is the zone's median relief
while W/L are only the zone's extent (a warning says so).

Step-1 scope: the blob is the object's top surface. On the real rig a
tall part's top sits closer to the camera and images larger, so §11
prefers W/L from the base footprint; that refinement comes with the
lateral calibration port (build step 6).
"""

from __future__ import annotations

import numpy as np

from .. import rules
from ..results import TIER_DEPTH, TIER_LATERAL, TIER_RELIEF, Dimensions
from .segment import scan_blob


def principal_extents_px(blob: np.ndarray) -> tuple[float, float]:
    v, u = np.nonzero(blob)
    pts = np.column_stack([u, v]).astype(np.float64)
    pts -= pts.mean(axis=0)
    _, vecs = np.linalg.eigh(np.cov(pts.T))
    proj = pts @ vecs                     # columns: minor, major
    ext = proj.max(axis=0) - proj.min(axis=0) + 1.0
    return float(ext[1]), float(ext[0])


def measure(scan, calibration=None) -> Dimensions:
    warnings = []
    if calibration is not None and not rules.calibration_matches(calibration, scan.geometry_id):
        warnings.append(f"calibration {calibration.id} is for geometry "
                        f"{calibration.geometry_id}, not {scan.geometry_id}; ignored (rule 6)")
        calibration = None
    blob, h_px, info = scan_blob(scan)
    warnings += info.get("warnings", [])
    w_px, l_px = principal_extents_px(blob)

    units = {"H": "px", "W": "px", "L": "px"}
    H, W, L = h_px, w_px, l_px
    if calibration is not None and calibration.has_z:
        H = calibration.mm_per_px * h_px + calibration.intercept_mm
        units["H"] = "mm"
    if calibration is not None and calibration.has_xy:
        W, L = w_px / calibration.lateral_px_per_mm, l_px / calibration.lateral_px_per_mm
        units["W"] = units["L"] = "mm"
    if calibration is None:
        warnings.append(f"no calibration for geometry {scan.geometry_id}; "
                        "reporting px, not mm (rule 6)")
    elif units["H"] != "mm" or units["W"] != "mm":
        warnings.append("calibration is partial; uncalibrated dimensions are in px")

    all_units = set(units.values())
    overall = all_units.pop() if len(all_units) == 1 else "mixed"
    tier = (TIER_LATERAL if units["W"] == "mm" and units["H"] == "mm" else
            TIER_DEPTH if units["H"] == "mm" else TIER_RELIEF)

    err = err_pct = None
    known = scan.known_dims or {}
    if known:
        vals = {"H": H, "W": W, "L": L}
        err, err_pct = {}, {}
        for k, true in known.items():
            if units[k] == "mm":
                err[k] = round(vals[k] - true, 3)
                err_pct[k] = round(100.0 * (vals[k] - true) / true, 2)
    r = lambda x: round(float(x), 3)
    return Dimensions(H=r(H), W=r(W), L=r(L), units=overall, unit_by_dim=units,
                      tier=tier, error=err, error_pct=err_pct, n_px=int(blob.sum()),
                      relief_px=r(h_px), warnings=warnings)


def raw_sample(scan) -> dict:
    """Pixel-space measurements used to fit a calibration.

    A zone scan whose part fills (or reaches the edge of) the zone still gives
    a good height sample, but its W/L are the zone's: when the scan carries a
    known W/L that would fit a wrong lateral calibration, so it is refused."""
    blob, h_px, info = scan_blob(scan)
    w_px, l_px = principal_extents_px(blob)
    wl_ok = bool(info.get("base_in_view", True) and not info.get("clipped_by_zone", False))
    known = scan.known_dims or {}
    if not wl_ok and ("W" in known or "L" in known):
        raise ValueError(
            f"scan {scan.id}: the part fills or reaches the edge of zone "
            f"{(scan.meta or {}).get('zone')!r}, so its W/L are the zone's, not the part's - "
            "calibrate from a scan with a larger zone (or no zone), or one with only a known H")
    return {"scan_id": scan.id, "relief_px": h_px, "W_px": w_px, "L_px": l_px,
            "n_px": int(blob.sum()), "wl_ok": wl_ok, **info}

"""
Cylindrical cells: fit, unwrap to a (theta, z) dent map, find dents.
(docs/UV_MAPPING_FOR_STRUCTURED_LIGHT.md section 1.)

Geometry (frontal, wall-facing rig): the cell stands with its axis along
camera y; relief is height toward the camera; x runs across the can. A
cross-section at one image row is a circle arc

    z(x) = a + b * sqrt(R^2 - (x - x0)^2)

x0, R in lateral units (mm with a lateral calibration, else camera px), a
(axis height over the base) in depth units (mm with a depth calibration,
else relief px). b = 1 when both axes are mm (a true circle); otherwise b
is fitted (the ellipse aspect = depth units per lateral unit) and the
circle is expressed in DEPTH units by scaling x by b. So:
  * both calibrated      -> everything in mm (rule 6 honoured)
  * depth only           -> mm, the lateral scale is INFERRED from the can's
                            curvature (warning; pass radius_mm to fix R)
  * uncalibrated         -> relief px, no mm anywhere, no verdict

Unwrap: each valid can pixel -> theta = atan2(b(x-x0), z-a), r = hypot(...),
deviation r - R; binned onto a regular (theta, z) grid of arc step `step`
(default 1 mm, or 3 camera px uncalibrated) so each bin averages ~9 px.
The valid mask is ONLY the seen arc (bins that received pixels, plus gaps of
<= 2 bins inside the seen band); unseen angles stay invalid and the arc in
degrees is reported. A frontal view sees < 180 degrees of a can.

find_dents runs the inspect.py blob logic on dent_map (deviation minus a
large-window median) with the same honest verdict: FAIL on a dent over the
trigger, PASS only when a trigger-deep dent would have been seen
(threshold + DETECT_MARGIN_SIGMA sigma <= trigger), else INDETERMINATE.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import ndimage, optimize

from .. import rules
from ..decode.unwrap import _fill_invalid
from .inspect import DETECT_MARGIN_SIGMA, TRIGGER_FLOOR_FRAC, residual
from .segment import SegmentationError, scan_blob, support

ROW_VALID_FRAC = 0.3        # a grid row counts toward cell height with this share of the arc
COL_VALID_FRAC = 0.5        # a theta column counts toward the arc with this share of the rows


@dataclass
class CylinderFit:
    x0: float                   # axis x (lateral units: mm or camera px)
    axis_z: float               # axis height over the base (depth units)
    radius: float               # in depth units (== mm when units == "mm")
    aspect: float               # b: depth units per lateral unit (1.0 when both mm)
    units: str                  # "mm" | "px"
    lateral_units: str          # "mm" | "px"
    rms: float                  # column-profile fit residual (depth units)
    n_px: int
    v_range: tuple              # (v_top, v_bottom) camera rows of the can
    lateral_px_per_mm: Optional[float]
    warnings: list = field(default_factory=list)


@dataclass
class CellUnwrap:
    theta_deg: np.ndarray       # (n_theta,) bin centres, 0 = facing the camera, + toward +x
    z: np.ndarray               # (n_z,) along the axis from the cell bottom
    deviation: np.ndarray       # (n_z, n_theta) r - R (units), NaN where not seen
    valid: np.ndarray           # (n_z, n_theta) bool, seen bins only
    dent_map: np.ndarray        # deviation minus its smooth background, NaN where not seen
    radius: float
    arc_deg: float              # seen arc (columns with >= COL_VALID_FRAC valid)
    cell_height: float          # z extent of the seen can (units)
    step: float                 # grid step (arc length and z), units
    units: str
    fit: CylinderFit
    stats: dict
    scan_id: Optional[str] = None
    label: Optional[str] = None
    geometry_id: str = ""
    calibration_id: str = "uncalibrated"
    zone: Optional[str] = None
    atlas: Optional[object] = None
    warnings: list = field(default_factory=list)


@dataclass
class Dent:
    theta_deg: float
    z: float
    depth: float                # > 0 = inward (dent), < 0 = outward (bump)
    mean_depth: float
    area_bins: int
    area: float                 # units^2 (arc length x z)
    snr: float
    units: str
    bbox: list                  # [i_theta0, i_z0, i_theta1, i_z1] grid bins
    exceeds_trigger: Optional[bool] = None


@dataclass
class DentReport:
    verdict: str                # PASS | FAIL | INDETERMINATE
    dents: list
    units: str
    noise_sigma: float
    threshold: float
    trigger: Optional[float]
    arc_deg: float
    warnings: list = field(default_factory=list)


def _scales(scan, calibration):
    warnings = []
    if calibration is not None and not rules.calibration_matches(calibration, scan.geometry_id):
        warnings.append(f"calibration {calibration.id} is for {calibration.geometry_id}, not "
                        f"{scan.geometry_id}; ignored (rule 6)")
        calibration = None
    zc = calibration if (calibration is not None and calibration.has_z) else None
    xy = calibration.lateral_px_per_mm if (calibration is not None and calibration.has_xy) else None
    if zc is None:
        warnings.append(f"no depth calibration for geometry {scan.geometry_id}; cylinder in "
                        "relief px, no mm and no verdict (rule 6)")
    return calibration, zc, xy, warnings


def fit_cylinder(scan, calibration=None, radius_mm: Optional[float] = None) -> CylinderFit:
    """Vertical-axis cylinder fit to the scan's part blob (see module doc)."""
    cal, zc, xy, warnings = _scales(scan, calibration)
    relief = scan.relief.astype(np.float64)
    valid = scan.relief_mask & np.isfinite(relief)
    try:
        blob, _, info = scan_blob(scan)
        warnings += info.get("warnings", [])
    except SegmentationError as e:
        raise ValueError(f"no cell found in scan {getattr(scan, 'id', '')}: {e}") from e
    region = support(blob) & valid
    region = ndimage.binary_erosion(region, iterations=2) & valid   # drop edge px
    if region.sum() < 200:
        raise ValueError("cell region too small to fit a cylinder")
    h, w = relief.shape
    Z = zc.mm_per_px * relief + zc.intercept_mm if zc else relief
    units = "mm" if zc else "px"
    lat_units = "mm" if xy else "px"
    k = xy if xy else 1.0

    # column profile: the median over the can's rows is robust to dents
    vs, us = np.nonzero(region)
    cols = np.unique(us)
    prof = np.array([np.median(Z[region[:, c], c]) for c in cols])
    n_c = np.array([region[:, c].sum() for c in cols])
    ok = n_c >= max(5, 0.3 * n_c.max())
    xs = (cols[ok] - w / 2.0) / k
    zs = prof[ok]
    if xs.size < 8:
        raise ValueError("too few columns on the cell to fit a cylinder")

    fixed_b = bool(zc and xy)
    fixed_R = None
    if radius_mm is not None:
        if fixed_b:
            fixed_R = float(radius_mm)
        elif zc:
            warnings.append("radius_mm given without a lateral calibration: used to fix the "
                            "radius in mm, the lateral scale is inferred from it")
            fixed_R = float(radius_mm)
        else:
            warnings.append("radius_mm ignored: no depth calibration, the fit is in px")

    half = 0.5 * (xs.max() - xs.min())
    xc = 0.5 * (xs.max() + xs.min())
    zmin = float(zs.min())

    def model(p):
        x0, a, R_lat, b = p
        return a + b * np.sqrt(np.clip(R_lat ** 2 - (xs - x0) ** 2, 0, None))

    def resid(q):
        x0, a, r_lat, b = q
        if fixed_b:
            b = 1.0
            if fixed_R is not None:
                r_lat = fixed_R
        elif fixed_R is not None:
            r_lat = fixed_R / b                     # radius in depth units = b * R_lat
        return model((x0, a, r_lat, b)) - zs

    # R_lat must exceed the seen half width; start at half / sin(60 deg)
    r0 = half / math.sin(math.radians(60))
    b0 = 1.0 if fixed_b else max(float(zs.max() - zmin), 1e-6) / (r0 * (1 - math.cos(math.radians(60))))
    a0 = float(zs.max()) - b0 * r0
    lo = [xs.min(), -np.inf, half * 1.0001, 1e-6]
    hi = [xs.max(), np.inf, 50 * half + 1, 1e6]
    sol = optimize.least_squares(resid, [xc, a0, r0, b0], bounds=(lo, hi), loss="soft_l1",
                                 f_scale=max(float(np.std(zs)) * 0.05, 1e-6))
    x0, a, r_lat, b = sol.x
    if fixed_b:
        b = 1.0
        if fixed_R is not None:
            r_lat = fixed_R
    elif fixed_R is not None:
        r_lat = fixed_R / b
    rms = float(np.sqrt(np.mean(resid(sol.x) ** 2)))
    radius = float(b * r_lat)
    if not fixed_b and zc:
        warnings.append(f"no lateral calibration: lateral scale inferred from the can's "
                        f"curvature ({1.0 / b:.3f} camera px per mm); calibrate W/L to confirm")
    if not zc:
        warnings.append("uncalibrated: the aspect (relief px per camera px) is fitted, the "
                        "radius is in relief px")
    if rms > 0.05 * radius:
        warnings.append(f"poor cylinder fit: profile rms {rms:.3g} {units} vs radius "
                        f"{radius:.3g} {units}; is this a vertical cylinder?")
    return CylinderFit(x0=float(x0), axis_z=float(a), radius=radius, aspect=float(b),
                       units=units, lateral_units=lat_units, rms=rms, n_px=int(region.sum()),
                       v_range=(int(vs.min()), int(vs.max())),
                       lateral_px_per_mm=xy if xy else (1.0 / b if zc else None),
                       warnings=warnings)


def unwrap_cell(scan, calibration=None, radius_mm: Optional[float] = None,
                step: Optional[float] = None, bg_window: float = 20.0,
                fit: Optional[CylinderFit] = None) -> CellUnwrap:
    """Scan -> CellUnwrap on a regular (theta, z) grid. step = arc/z bin size in
    the result units (default 1 mm; 3 camera-px-equivalents uncalibrated).
    bg_window = dent_map background window (units; ~2x the largest dent)."""
    cal, zc, xy, _ = _scales(scan, calibration)
    fit = fit or fit_cylinder(scan, calibration, radius_mm)
    warnings = list(fit.warnings)
    relief = scan.relief.astype(np.float64)
    valid = scan.relief_mask & np.isfinite(relief)
    h, w = relief.shape
    Z = zc.mm_per_px * relief + zc.intercept_mm if zc else relief
    k_lat = xy if xy else 1.0
    b = fit.aspect
    # pixel -> depth units along x (b * lateral) and along the axis (same scale)
    blob, _, _ = scan_blob(scan)
    region = support(blob) & valid
    vs, us = np.nonzero(region)
    X = b * ((us - w / 2.0) / k_lat - fit.x0)
    D = Z[vs, us] - fit.axis_z
    r = np.hypot(X, D)
    th = np.arctan2(X, D)
    # along-axis scale: the same lateral px size (camera px are square)
    zz = b * (fit.v_range[1] - vs) / k_lat
    good = (D > 0) & (np.abs(r - fit.radius) < 0.5 * fit.radius)
    th, zz, dev = th[good], zz[good], (r - fit.radius)[good]
    if th.size < 50:
        raise ValueError("too few cell pixels on the fitted cylinder")
    px_size = b / k_lat                              # one camera px in result units
    if step is None:
        step = 1.0 if fit.units == "mm" else 3.0 * px_size
    if step < 1.5 * px_size:
        warnings.append(f"grid step {step:.3g} is under 1.5 camera px: many bins will be empty")
    dth = step / fit.radius
    t0, t1 = th.min(), th.max()
    n_t = max(2, int(math.ceil((t1 - t0) / dth)))
    n_z = max(2, int(math.ceil((zz.max() + 1e-9) / step)))
    it = np.clip(((th - t0) / dth).astype(int), 0, n_t - 1)
    iz = np.clip((zz / step).astype(int), 0, n_z - 1)
    s = np.zeros((n_z, n_t))
    n = np.zeros((n_z, n_t))
    np.add.at(s, (iz, it), dev)
    np.add.at(n, (iz, it), 1.0)
    seen = n > 0
    mean = np.where(seen, s / np.maximum(n, 1), 0.0)
    # close <= 2-bin gaps, only between seen bins on the same row AND column
    rows = np.maximum.accumulate(seen, 1) & np.maximum.accumulate(seen[:, ::-1], 1)[:, ::-1]
    cols = np.maximum.accumulate(seen, 0) & np.maximum.accumulate(seen[::-1], 0)[::-1]
    closed = ndimage.binary_closing(seen, structure=np.ones((3, 3)), iterations=2)
    vmask = seen | (closed & rows & cols)
    dev_g = np.where(vmask, _fill_invalid(mean, seen), np.nan)

    theta_deg = np.degrees(t0 + (np.arange(n_t) + 0.5) * dth)
    zc_grid = (np.arange(n_z) + 0.5) * step
    colf = vmask.mean(0)
    arc_cols = colf >= COL_VALID_FRAC * colf.max()
    arc_deg = float(arc_cols.sum() * math.degrees(dth))
    rowf = vmask.mean(1)
    rows_ok = np.nonzero(rowf >= ROW_VALID_FRAC * rowf.max())[0]
    cell_height = float((rows_ok.max() - rows_ok.min() + 1) * step) if rows_ok.size else float("nan")
    if arc_deg >= 180:
        warnings.append(f"seen arc {arc_deg:.0f} deg >= 180: a single frontal view cannot see "
                        "that much of a can - check the fit")

    win = max(5, int(round(bg_window / step)) | 1)
    dent = residual(np.where(vmask, dev_g, 0.0), vmask, bg_window=win, step=1)
    v = dev_g[vmask]
    stats = {"n_bins": int(vmask.sum()), "n_px": int(th.size),
             "deviation_median": float(np.median(v)), "deviation_p1": float(np.percentile(v, 1)),
             "deviation_p99": float(np.percentile(v, 99)),
             "dent_sigma": float(1.4826 * np.median(np.abs(dent[vmask] - np.median(dent[vmask])))),
             "fit_rms": fit.rms, "aspect": fit.aspect,
             "theta_range_deg": [float(theta_deg[0]), float(theta_deg[-1])]}
    m = getattr(scan, "meta", None) or {}
    return CellUnwrap(theta_deg=theta_deg, z=zc_grid, deviation=dev_g, valid=vmask,
                      dent_map=np.where(vmask, dent, np.nan), radius=fit.radius, arc_deg=arc_deg,
                      cell_height=cell_height, step=float(step), units=fit.units, fit=fit,
                      stats=stats, scan_id=getattr(scan, "id", None),
                      label=getattr(scan, "label", None), geometry_id=scan.geometry_id,
                      calibration_id=cal.id if (cal is not None and zc) else "uncalibrated",
                      zone=m.get("zone"), atlas=m.get("atlas"), warnings=warnings)


def find_dents(unwrap: CellUnwrap, depth_trigger_mm: Optional[float] = None,
               n_sigma: float = 5.0, min_area: int = 6, min_thr: Optional[float] = None,
               include_bumps: bool = False) -> DentReport:
    """Blobs on the dent map. depth > 0 = inward. Verdict only in mm with a trigger."""
    warnings = []
    mm = unwrap.units == "mm"
    trig = depth_trigger_mm if mm else None
    if not mm:
        warnings.append("dent map in px (no depth calibration): no mm verdict (rule 6)")
    elif not depth_trigger_mm:
        warnings.append("no depth trigger: dents are listed but there is no PASS/FAIL verdict")
    valid = unwrap.valid
    if valid.sum() < 50:
        return DentReport("INDETERMINATE", [], unwrap.units, float("nan"), float("nan"), trig,
                          unwrap.arc_deg, warnings + ["unwrap too small to inspect"])
    dm = np.where(valid, unwrap.dent_map, 0.0)
    v = dm[valid]
    sigma = 1.4826 * float(np.median(np.abs(v - np.median(v))))
    floor = min_thr if min_thr is not None else (TRIGGER_FLOOR_FRAC * trig if trig else 0.0)
    thr = max(n_sigma * sigma, floor)
    inward = -dm
    cand = valid & ((np.abs(inward) if include_bumps else inward) > thr)
    cand = ndimage.binary_opening(cand, structure=np.ones((2, 2)))
    lab, n = ndimage.label(cand)
    sm = ndimage.gaussian_filter(_fill_invalid(dm, valid), 0.7)
    dents = []
    for i in range(1, n + 1):
        m = lab == i
        area = int(m.sum())
        if area < min_area:
            continue
        zi, ti = np.nonzero(m)
        wgt = np.abs(dm[m])
        sign = 1.0 if np.median(inward[m]) > 0 else -1.0
        depth = float(sign * np.max(sign * -sm[m]))
        dents.append(Dent(
            theta_deg=round(float(np.average(unwrap.theta_deg[ti], weights=wgt)), 2),
            z=round(float(np.average(unwrap.z[zi], weights=wgt)), 3),
            depth=round(depth, 4), mean_depth=round(float(np.mean(inward[m])), 4),
            area_bins=area, area=round(area * unwrap.step ** 2, 3),
            snr=round(abs(depth) / max(sigma, 1e-9), 1), units=unwrap.units,
            bbox=[int(ti.min()), int(zi.min()), int(ti.max()), int(zi.max())],
            exceeds_trigger=(abs(depth) >= trig) if trig else None))
    dents.sort(key=lambda d: -abs(d.depth))
    if trig:
        reach = thr + DETECT_MARGIN_SIGMA * sigma
        if any(d.exceeds_trigger for d in dents):
            verdict = "FAIL"
        elif reach <= trig:
            verdict = "PASS"
        else:
            verdict = "INDETERMINATE"
            warnings.append(
                f"cannot PASS: the dent threshold {thr:.3f} mm (+{DETECT_MARGIN_SIGMA:g} sigma "
                f"margin = {reach:.3f} mm; noise sigma {sigma:.3f} mm, n_sigma {n_sigma:g}) is not "
                f"below the {trig:g} mm trigger, so a trigger-deep dent could go unseen. Lower the "
                "noise (more camera.average, a coarser unwrap step) or lower n_sigma")
    else:
        verdict = "INDETERMINATE"
    warnings.append(f"inspected the seen arc only ({unwrap.arc_deg:.0f} deg); the rest of the "
                    "can is unseen - rotate and rescan for full coverage")
    return DentReport(verdict=verdict, dents=dents, units=unwrap.units,
                      noise_sigma=round(sigma, 5), threshold=round(thr, 5), trigger=trig,
                      arc_deg=unwrap.arc_deg, warnings=warnings)


# ------------------------------------------------------------------ outputs

def save_unwrap(path, unwrap: CellUnwrap) -> None:
    """npz of the grids + scalars (new path only)."""
    rules.ensure_new_path(path)
    with open(path, "xb") as f:
        np.savez_compressed(f, theta_deg=unwrap.theta_deg, z=unwrap.z,
                            deviation=unwrap.deviation.astype(np.float32),
                            valid=unwrap.valid, dent_map=unwrap.dent_map.astype(np.float32),
                            radius=unwrap.radius, arc_deg=unwrap.arc_deg,
                            cell_height=unwrap.cell_height, step=unwrap.step,
                            units=unwrap.units, geometry_id=unwrap.geometry_id,
                            calibration_id=unwrap.calibration_id)


def dent_png(unwrap: CellUnwrap, report: Optional[DentReport] = None, scale: int = 4):
    """PNG bytes: dent map heatmap (theta across, z UP), unseen black, dents boxed."""
    try:
        import cv2
    except ImportError:
        return None
    dm = np.where(unwrap.valid, unwrap.dent_map, 0.0)
    lim = max(float(np.percentile(np.abs(dm[unwrap.valid]), 99.5)), 1e-9) if unwrap.valid.any() else 1.0
    if report is not None and report.dents:
        lim = max(lim, max(abs(d.depth) for d in report.dents))
    img = np.clip(0.5 + 0.5 * dm / lim, 0, 1)
    vis = cv2.applyColorMap((img * 255).astype(np.uint8), cv2.COLORMAP_JET)
    vis[~unwrap.valid] = 0
    vis = vis[::-1]                                              # z up
    vis = cv2.resize(vis, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    nz = unwrap.valid.shape[0]
    for i, d in enumerate((report.dents if report else [])[:10]):
        t0, z0, t1, z1 = d.bbox
        p0 = (t0 * scale, (nz - 1 - z1) * scale)
        p1 = ((t1 + 1) * scale, (nz - z0) * scale)
        cv2.rectangle(vis, p0, p1, (255, 255, 255), 1)
        cv2.putText(vis, f"#{i + 1} {d.depth:.2f}{d.units}", (p0[0], max(12, p0[1] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(vis, f"arc {unwrap.arc_deg:.0f} deg  +/-{lim:.2f} {unwrap.units}", (4, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    ok, buf = cv2.imencode(".png", vis)
    return buf.tobytes() if ok else None


def report_dict(unwrap: CellUnwrap, report: DentReport) -> dict:
    return {"scan": unwrap.scan_id, "label": unwrap.label, "geometry_id": unwrap.geometry_id,
            "calibration_id": unwrap.calibration_id, "units": unwrap.units,
            "radius": unwrap.radius, "arc_deg": unwrap.arc_deg, "cell_height": unwrap.cell_height,
            "step": unwrap.step, "fit": {k: v for k, v in vars(unwrap.fit).items()},
            "stats": unwrap.stats, "verdict": report.verdict,
            "dents": [vars(d) for d in report.dents], "noise_sigma": report.noise_sigma,
            "threshold": report.threshold, "trigger": report.trigger,
            "warnings": list(dict.fromkeys(unwrap.warnings + report.warnings))}

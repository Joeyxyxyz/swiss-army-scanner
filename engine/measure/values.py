"""
Measure mode's measurements: numbers read off ONE metric cloud (mm, camera frame: x right,
y down, z forward). No tolerance, no verdict: each function answers a list of VALUES, and
what to draw on the overlay.

Thin wrappers over the engine's own measures, nothing new is fitted here:

    flatness    robust plane (measure/endface/profile.robust_plane, as the recipe `flatness`
                check): RMS, peak-to-valley between the 2nd and 98th percentile, tilt
    step        plane to plane: the height of region B above region A's plane (as the recipe
                `height` check with ref zone), parallelism
    distance    two picked camera px -> their 3-D points -> the distance; a picked px to the
                plane of a region
    cylinder    measure/metric.cylinder_deviation: radius / diameter, axis, seen arc
    dims        measure/metric.foreground: the part standing on the reference (dominant)
                plane -> length x width (its minimum-area rectangle in that plane) x height
    deviation   reconstruct/metric_scan.compare_metric's signed deviation map vs a golden
                scan -> RMS, mean, extremes (values only)
    endfaces    measure/station's end faces (a cylinder's circular end face toward the
                camera: the rim circle fitted to the flood outline in the face plane) ->
                per face its diameter and flatness

A value: {"name", "label", "value", "units" ("mm" | "deg"), "n_points", "sensitivity", ...}.
`sensitivity` says what its uncertainty comes from (engine/modes/measure.py adds it):
"depth" (the depth noise sigma_z), "lateral" (the camera px size at that distance), "both"
or "angle" (depth noise across `lever_mm`). What only describes the measurement (the arc of a
cylinder the camera sees, the plane a height was taken from) is in `details`, not a value.

MeasureError = this scan cannot give that number (too few points, nothing decoded at the
picked px, no part on the plane): Measure mode reports NOT_MEASURABLE with the reason; it
never guesses, and a camera px is never a mm. The same when the number would not be what its
name says: a "cylinder" that is flat within the noise or that the points do not lie on, a
part cut by the image border or the roi (its bounding size is then the cut's), a patch of
the part too far from the golden to be compared (its largest deviation is then not the
largest), a picked px that straddles an edge (two surfaces, one point).

Every value carries the depth noise of the px it rests on (`noise_mm`, measured_noise()
there; None when too few neighbouring px are decoded): its uncertainty is the noise THERE,
not the scan's (a dark part in a quiet scan is noisier than the scan).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from . import metric
from .endface.profile import robust_plane

MIN_POINTS = 100                # fewer decoded points in a region: not measurable
FIT_POINTS = 200000             # a plane is fitted on at most this many of a region's points
PICK_RADIUS_PX = 3              # a picked point = the median of the decoded px this close
PICK_MIN_PX = 5                 # ... at least this many of them
TRIM_PCT = 2.0                  # flatness peak-to-valley: between p2 and p98
AXES = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0), "auto": None}
WEAK_ARC_DEG = 60.0             # a cylinder seen over less: its radius is weakly determined
TOP_WINDOW_PX = 5               # dims: the height is read off heights averaged over this window
BACKGROUND_SHARE = 0.4          # cylinder: a background plane holds at least this share of the
                                # scan's points (else the region itself is the cylinder)
NOT_PARALLEL_DEG = 1.0          # a step between faces further apart than this: say so
SURFACE_JUMP_PITCH = 5.0        # neighbouring px further apart in depth than this x the px
                                # size are on different surfaces (a face steeper than ~79 deg
                                # to the view is not decoded anyway)
CYL_MIN_SAGITTA_NOISE = 10.0    # cylinder: the seen arc must bulge at least this x the depth
                                # noise out of its chord, else it is flat: no radius
CYL_FIT_NOISE = 5.0             # cylinder: the points' robust spread about the fit may be at
CYL_FIT_REL = 0.01              # most this x the depth noise, this share of the radius, or
CYL_FIT_MIN_MM = 0.1            # this (whichever is largest): else they are not one cylinder
CYL_OFF_SHARE = 0.2             # ... and say so when more than this share lies off the fit
BORDER_PX = 2                   # dims: a part this close to the image border is cut by it
EDGE_GAP_PITCH = 4.0            # a pick: depths this x the px size apart within its window
                                # are two surfaces (an edge): not one point
PICK_FULL_SHARE = 0.8           # a pick: fewer decoded px than this share of its window is
                                # the rim of the decoded surface (its position is less sure)
NOISE_RADIUS_PX = 12            # a pick: the depth noise is read this close around it
MIN_PATCH_PX = 3                # deviation: a patch this wide and high that was not compared
                                # is a part of the part, not stray edge px


class MeasureError(ValueError):
    """This scan cannot give the number asked for. code: the doubt (engine/disposition
    reason: too_few_points, incomplete, invalid, golden ...)."""

    def __init__(self, message: str, code: str = "too_few_points"):
        super().__init__(message)
        self.code = code


def value(name: str, val, units: str, n_points: int, sensitivity: str, label: str,
          **extra) -> dict:
    v = float(val)
    if not math.isfinite(v):
        raise MeasureError(f"{label}: the measurement gave no finite number", "invalid")
    return {"name": name, "label": label, "value": v, "units": units,
            "n_points": int(n_points), "sensitivity": sensitivity, "notes": [], **extra}


def region_noise(cloud, *masks) -> Optional[float]:
    """The depth noise (mm, 1 sigma) of the px values rest on: measured_noise() in each of
    the masks, the largest of them (the noisier face sets the limit). None when none of
    them has enough neighbouring decoded px to tell."""
    got = [measured_noise(cloud, m) for m in masks if m is not None]
    got = [v for v in got if v is not None]
    return max(got) if got else None


def with_noise(vals: list, noise: Optional[float]) -> list:
    for v in vals:
        v["noise_mm"] = noise
    return vals


def roi_mask(shape, roi) -> np.ndarray:
    """[x0, y0, x1, y1] camera px (x1, y1 exclusive) -> bool (H, W); None = everything."""
    m = np.zeros(shape, bool)
    if roi is None:
        m[:] = True
        return m
    x0, y0, x1, y1 = (int(v) for v in roi)
    H, W = shape
    if x1 <= x0 or y1 <= y0 or x0 < 0 or y0 < 0 or x0 >= W or y0 >= H:
        raise ValueError(f"roi {list(roi)} is outside the {W} x {H} px camera image "
                         "(x0, y0, x1, y1; x1 > x0, y1 > y0)")
    m[y0:min(y1, H), x0:min(x1, W)] = True
    return m


def _points(cloud, region: Optional[np.ndarray], what: str) -> tuple:
    m = cloud.mask if region is None else (cloud.mask & region)
    P = np.asarray(cloud.xyz[m], np.float64)
    if len(P) < MIN_POINTS:
        n_px = int(m.size if region is None else region.sum())
        raise MeasureError(f"only {len(P)} decoded points in {what} ({n_px} camera px): too "
                           f"few to measure (need {MIN_POINTS}; check the roi / zone, the part "
                           "and the exposure)")
    return P, m


def _plane(P: np.ndarray) -> tuple:
    """Robust plane of the points -> (centroid, normal toward the camera, every point's
    residual about their median). Fitted on an even subsample of a large region; the
    residuals are of all points."""
    S = P if len(P) <= FIT_POINTS else P[:: len(P) // FIT_POINTS + 1]
    c, n = robust_plane(S)
    r = (P - c) @ n
    return c, n, r - np.median(r)


def _span(P: np.ndarray, c: np.ndarray, n: np.ndarray) -> float:
    """The points' smaller extent in their plane (mm, p2..p98): the lever an angle rests on."""
    e1 = np.cross(n, [0.0, 1.0, 0.0])
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.cross(n, [1.0, 0.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    S = P if len(P) <= 50000 else P[:: len(P) // 50000 + 1]
    uv = np.c_[(S - c) @ e1, (S - c) @ e2]
    uv -= uv.mean(0)
    _, vecs = np.linalg.eigh(np.cov(uv.T))
    q = uv @ vecs
    spans = np.percentile(q, 98, axis=0) - np.percentile(q, 2, axis=0)
    return float(max(min(spans), 1e-6))


def _tilt_deg(n: np.ndarray) -> float:
    """Angle between a plane's normal and the camera's optical axis."""
    return math.degrees(math.acos(min(1.0, abs(float(n[2])) / max(np.linalg.norm(n), 1e-12))))


def _pitch(cloud, m: np.ndarray) -> Optional[float]:
    """The camera px size on the surface (mm): the median spacing of horizontally
    neighbouring decoded px of m. None when m has no such neighbours."""
    pair = m[:, 1:] & m[:, :-1]
    if int(pair.sum()) < 10:
        return None
    d = (np.asarray(cloud.xyz[:, 1:][pair], np.float64)
         - np.asarray(cloud.xyz[:, :-1][pair], np.float64))
    v = float(np.median(np.hypot(d[:, 0], d[:, 1])))
    return v if math.isfinite(v) and v > 0 else None


def _largest_surface(cloud, m: np.ndarray) -> tuple:
    """(the largest depth-continuous piece of the decoded px of m, how many pieces there
    are). Neighbouring px more than SURFACE_JUMP_PITCH x the px size apart in depth are on
    different surfaces (the far one of each such pair is the cut)."""
    from scipy import ndimage
    pitch = _pitch(cloud, m)
    if pitch is None:
        return m, 1
    z = np.where(m, cloud.xyz[..., 2], np.nan).astype(np.float64)
    thr = SURFACE_JUMP_PITCH * pitch
    cut = np.zeros(m.shape, bool)
    with np.errstate(invalid="ignore"):
        dz = z[:, 1:] - z[:, :-1]
        cut[:, 1:] |= dz > thr                  # the right px is the far one
        cut[:, :-1] |= dz < -thr
        dz = z[1:] - z[:-1]
        cut[1:] |= dz > thr
        cut[:-1] |= dz < -thr
    lab, n = ndimage.label(m & ~cut)
    if n == 0:
        return m, 1
    sizes = np.bincount(lab.ravel())[1:]
    return lab == (int(np.argmax(sizes)) + 1), int(n)


# ------------------------------------------------------------------ flatness

def flatness(cloud, region: Optional[np.ndarray] = None) -> dict:
    """Plane flatness of a region: RMS and p2-p98 of the residuals to its robust plane (mm),
    the plane's tilt vs the camera's optical axis (deg) and its distance (mm)."""
    P, m = _points(cloud, region, "the region")
    c, n, r = _plane(P)
    lo, hi = np.percentile(r, [TRIM_PCT, 100.0 - TRIM_PCT])
    n_pts = len(P)
    lever = _span(P, c, n)
    return {"values": with_noise([
        value("flatness_rms", math.sqrt(float(np.mean(r * r))), "mm", n_pts, "depth",
              "Flatness (RMS)"),
        value("flatness_p2_p98", hi - lo, "mm", n_pts, "depth",
              "Flatness (peak-to-valley, p2-p98)", trim_pct=TRIM_PCT),
        value("tilt", _tilt_deg(n), "deg", n_pts, "angle", "Tilt vs the optical axis",
              lever_mm=lever),
        value("plane_distance", abs(float(n @ c)), "mm", n_pts, "depth",
              "Plane distance from the camera")], region_noise(cloud, m)),
        "z_mm": float(np.median(P[:, 2])), "mask": m,
        "details": {"plane_point_mm": c.tolist(), "plane_normal": n.tolist(),
                    "max_abs_dev_mm": float(np.abs(r).max()), "extent_mm": lever}}


# ------------------------------------------------------------------ step

def step(cloud, region_a: np.ndarray, region_b: np.ndarray) -> dict:
    """Step height, plane to plane: the median height of region B's points above region A's
    plane (mm, + toward the camera), and how parallel the two faces are (deg)."""
    Pa, ma = _points(cloud, region_a, "region A")
    Pb, mb = _points(cloud, region_b, "region B")
    ca, na, ra = _plane(Pa)
    cb, nb, rb = _plane(Pb)
    h = float(np.median((Pb - ca) @ na))            # B above A's plane
    h_rev = float(-np.median((Pa - cb) @ nb))       # the same step, seen from B's plane
    par = math.degrees(math.acos(min(1.0, abs(float(na @ nb)))))
    n_pts = min(len(Pa), len(Pb))
    v = value("step_height", h, "mm", n_pts, "depth", "Step height (B above A)",
              n_points_a=len(Pa), n_points_b=len(Pb))
    if par > NOT_PARALLEL_DEG:
        v["notes"].append(f"the two faces are {par:.2f} deg apart: the step depends on where "
                          f"it is taken (B above A's plane {h:.3f} mm, A below B's plane "
                          f"{h_rev:.3f} mm)")
    lever = min(_span(Pa, ca, na), _span(Pb, cb, nb))
    return {"values": with_noise([v, value("parallelism", par, "deg", n_pts, "angle",
                                       "Angle between the two faces", lever_mm=lever)],
                             region_noise(cloud, ma, mb)),   # the noisier face sets the limit
            "z_mm": float(np.median(np.r_[Pa[:, 2], Pb[:, 2]])), "mask": ma | mb,
            "details": {"height_b_above_a_mm": h, "height_a_below_b_mm": h_rev,
                        "rms_a_mm": float(np.sqrt(np.mean(ra * ra))),
                        "rms_b_mm": float(np.sqrt(np.mean(rb * rb)))}}


# ------------------------------------------------------------------ distance

def pick(cloud, px, radius: int = PICK_RADIUS_PX) -> tuple:
    """The 3-D point seen at a camera px: the median of the decoded points within `radius`
    px of it -> (xyz mm, number of px used). Never a guess: a px with no decoded surface
    around it, or one that straddles an edge (two surfaces within its window), is a
    MeasureError."""
    got = _pick(cloud, px, radius)
    return got["xyz"], got["n"]


def _pick(cloud, px, radius: int = PICK_RADIUS_PX) -> dict:
    """pick() with what it rests on: {"xyz", "n", "px" (the whole px), "partial" (the window
    is the rim of the decoded surface: fewer than PICK_FULL_SHARE of its px are decoded),
    "noise_mask" (the decoded px within NOISE_RADIUS_PX)}."""
    H, W = cloud.mask.shape
    try:
        x, y = (float(v) for v in px)
    except (TypeError, ValueError):
        raise ValueError(f"a picked point is [x, y] in camera px, got {px!r}") from None
    if not (math.isfinite(x) and math.isfinite(y)) or not (0 <= x < W and 0 <= y < H):
        raise ValueError(f"picked point ({x:g}, {y:g}) is outside the {W} x {H} px camera image")
    xi, yi = int(round(x)), int(round(y))
    y0, y1 = max(yi - radius, 0), min(yi + radius + 1, H)
    x0, x1 = max(xi - radius, 0), min(xi + radius + 1, W)
    m = cloud.mask[y0:y1, x0:x1]
    if int(m.sum()) < PICK_MIN_PX:
        raise MeasureError(f"no decoded surface at camera px ({xi}, {yi}) (only {int(m.sum())} "
                           f"decoded px within {radius} px): pick a point on the measured "
                           "surface")
    Q = np.asarray(cloud.xyz[y0:y1, x0:x1][m], np.float64)
    # one point is on one surface: an edge inside the window shows as a gap in its depths
    zs = np.sort(Q[:, 2])
    gap = float(np.diff(zs).max()) if len(zs) > 1 else 0.0
    near = np.zeros((H, W), bool)
    near[max(yi - NOISE_RADIUS_PX, 0):yi + NOISE_RADIUS_PX + 1,
         max(xi - NOISE_RADIUS_PX, 0):xi + NOISE_RADIUS_PX + 1] = True
    near &= cloud.mask
    pitch = _pitch(cloud, near)
    limit = EDGE_GAP_PITCH * pitch if pitch is not None else 1.0
    if gap > limit:
        raise MeasureError(f"camera px ({xi}, {yi}) is on an edge: the surface within {radius} "
                           f"px of it lies at two depths {gap:.2f} mm apart, so it is not one "
                           "point - pick a point on one face, a few px away from the edge",
                           "invalid")
    full = (2 * radius + 1) ** 2
    return {"xyz": np.median(Q, axis=0), "n": int(m.sum()), "px": (xi, yi),
            "partial": int(m.sum()) < PICK_FULL_SHARE * full, "noise_mask": near}


def distance(cloud, p1, p2=None, region: Optional[np.ndarray] = None) -> dict:
    """The distance between two picked camera px (their 3-D points), and / or the distance
    of the picked point(s) to the plane of a region (+ toward the camera)."""
    if p2 is None and region is None:
        raise ValueError("distance needs a second point (p2) or a region whose plane the "
                         "point is measured to (roi / zone)")
    g1 = _pick(cloud, p1)
    X1, n1 = g1["xyz"], g1["n"]
    vals, details = [], {"p1_xyz_mm": X1.tolist()}
    zs = [float(X1[2])]
    mask = np.zeros(cloud.mask.shape, bool)
    near = [g1["noise_mask"]]
    X2 = None
    if p2 is not None:
        g2 = _pick(cloud, p2)
        X2, n2 = g2["xyz"], g2["n"]
        d = X2 - X1
        v = value("distance", float(np.linalg.norm(d)), "mm", n1 + n2, "both",
                  "Distance P1 - P2")
        rim = [f"P{i} ({g['px'][0]}, {g['px'][1]})" for i, g in ((1, g1), (2, g2))
               if g["partial"]]
        if rim:                         # the median of a partly decoded window sits off the px
            v["lateral_gain"] = float(PICK_RADIUS_PX)
            v["notes"].append(f"{' and '.join(rim)}: at the rim of the decoded surface (fewer "
                              f"than {PICK_FULL_SHARE:.0%} of the px within {PICK_RADIUS_PX} px "
                              "are decoded): the point is the median of those, up to "
                              f"{PICK_RADIUS_PX} px off the picked px")
        near.append(g2["noise_mask"])
        vals += with_noise([v], region_noise(cloud, *near))
        details.update(p2_xyz_mm=X2.tolist(), delta_xyz_mm=d.tolist())
        zs.append(float(X2[2]))
    if region is not None:
        P, mask = _points(cloud, region, "the plane's region")
        c, n, _ = _plane(P)
        vals += with_noise([value("point_to_plane", float((X1 - c) @ n), "mm", len(P), "depth",
                              "P1 to the plane")], region_noise(cloud, near[0], mask))
        if X2 is not None:
            vals += with_noise([value("point2_to_plane", float((X2 - c) @ n), "mm", len(P),
                                  "depth", "P2 to the plane")],
                           region_noise(cloud, near[1], mask))
        details.update(plane_point_mm=c.tolist(), plane_normal=n.tolist())
    return {"values": vals, "z_mm": float(np.median(zs)), "mask": mask, "details": details}


# ------------------------------------------------------------------ cylinder

def cylinder(cloud, region: Optional[np.ndarray] = None, radius_mm: Optional[float] = None,
             axis: str = "auto") -> dict:
    """Cylinder fit (measure/metric.cylinder_deviation) on the part in the region: radius and
    diameter (unless the radius is fixed with radius_mm), the fit's RMS, the axis and the arc
    the camera sees.

    The part: what stands out of the scan's background plane when the scan shows one
    (BACKGROUND_SHARE), else the largest depth-continuous surface in the region (a cylinder
    that fills the view is not fitted together with the wall behind it).
    Not a cylinder, not a number (MeasureError `invalid`): the seen arc bulges less than
    CYL_MIN_SAGITTA_NOISE x the depth noise out of its chord (a flat face fits a "cylinder"
    of any huge radius), or the points' robust spread about the fit is beyond the CYL_FIT_*
    limit (two surfaces, another shape). A weak arc widens the radius' uncertainty
    (`depth_gain` = radius / sagitta: what a depth error does to the radius)."""
    if axis not in AXES:
        raise ValueError(f"axis must be one of {', '.join(AXES)}, got {axis!r}")
    if radius_mm is not None and not (math.isfinite(radius_mm) and radius_mm > 0):
        raise ValueError("radius_mm must be a positive, finite number (mm)")
    full = cloud.mask if region is None else (cloud.mask & region)
    part = None
    n_all = int(cloud.mask.sum())
    if n_all >= MIN_POINTS:
        try:                                # the part standing out of the background plane
            fg, bg = metric.foreground(cloud, region=region)
            if bg.n >= BACKGROUND_SHARE * n_all:    # ... when the scan shows such a plane
                part = fg
        except Exception:                   # noqa: BLE001 - no usable plane: the region itself
            part = None
    picked = None
    if part is not None and int(part.sum()) >= 2 * MIN_POINTS:
        use = part
    else:                                   # no background plane to stand out of
        use, pieces = _largest_surface(cloud, full)
        if pieces > 1:
            picked = (f"fitted on the largest continuous surface in the region "
                      f"({int(use.sum())} of {int(full.sum())} decoded points; {pieces} "
                      "separate surfaces are in it): give an roi on the cylinder to be sure")
    if int(use.sum()) < 2 * MIN_POINTS:
        raise MeasureError(f"only {int(use.sum())} decoded points on the cylinder (need "
                           f"{2 * MIN_POINTS}): check the roi and the part")
    try:
        dev = metric.cylinder_deviation(cloud, use, axis_hint=AXES[axis], radius_mm=radius_mm)
    except ValueError as e:
        raise MeasureError(str(e)) from None
    f = dev.fit
    n_pts = int(dev.mask.sum())
    P = np.asarray(cloud.xyz[dev.mask], np.float64)
    along = (P - f.point) @ f.axis
    length = float(np.percentile(along, 98) - np.percentile(along, 2))
    # is it a cylinder at all? the seen arc's bulge out of its chord (the sagitta) against
    # the depth noise, and the points' robust spread about the fit
    e0, e1 = f.frame()
    th = np.arctan2((P - f.point) @ e1, (P - f.point) @ e0)
    arc = float(np.percentile(th, 99.5) - np.percentile(th, 0.5))
    R = float(f.radius)
    sag = R * (1.0 - math.cos(min(arc, math.pi) / 2.0))
    noise = measured_noise(cloud, dev.mask)
    spread = float(dev.noise_sigma_mm)
    if noise is None:
        noise = spread
    if not (math.isfinite(R) and R > 0 and math.isfinite(sag) and math.isfinite(spread)):
        raise MeasureError("the cylinder fit gave no finite radius", "invalid")
    if radius_mm is None and sag < CYL_MIN_SAGITTA_NOISE * noise:
        raise MeasureError(
            f"no cylinder in the region: the surface is flat within the noise (the fitted arc "
            f"of {math.degrees(arc):.2f} deg bulges {sag:.4f} mm, the depth noise is "
            f"{noise:.4f} mm; a radius of {R:.4g} mm would be a guess) - give an roi on the "
            "cylinder", "invalid")
    limit = max(CYL_FIT_NOISE * noise, CYL_FIT_REL * R, CYL_FIT_MIN_MM)
    if spread > limit:
        raise MeasureError(
            f"the points in the region do not lie on one cylinder: their spread about the "
            f"best fit is {spread:.3f} mm (limit {limit:.3f} mm at a depth noise of "
            f"{noise:.4f} mm; fit RMS {f.rms:.3f} mm) - give an roi on the cylinder alone "
            "(a wall, a second part or another shape is in it)", "invalid")
    res = dev.residual[dev.mask]
    off = float(np.mean(np.abs(res - np.median(res)) > max(limit, 3.0 * spread)))
    gain = max(1.0, R / max(sag, 1e-9))     # what a depth error does to the radius
    vals = []
    if radius_mm is None:
        r = value("cylinder_radius", f.radius, "mm", n_pts, "both", "Cylinder radius",
                  depth_gain=gain)
        d = value("cylinder_diameter", f.diameter, "mm", n_pts, "both", "Cylinder diameter",
                  depth_gain=2.0 * gain, lateral_gain=2.0)
        for v in (r, d):
            if dev.arc_deg is not None and dev.arc_deg < WEAK_ARC_DEG:
                v["notes"].append(f"only {dev.arc_deg:.0f} deg of the cylinder is seen: its "
                                  "radius is weakly determined (a depth error counts "
                                  f"{gain:.1f} x in it)")
            if picked:
                v["notes"].append(picked)
            if off > CYL_OFF_SHARE:
                v["notes"].append(f"{off:.0%} of the fitted points lie off the cylinder "
                                  "(dents, a label, a second surface): the radius is that of "
                                  "the rest")
        vals += [r, d]
    vals += [value("cylinder_fit_rms", f.rms, "mm", n_pts, "depth", "Cylinder fit RMS"),
             value("cylinder_axis_tilt", math.degrees(math.asin(min(1.0, abs(float(f.axis[2]))))),
                   "deg", n_pts, "angle", "Axis vs the image plane", lever_mm=max(length, 1e-6))]
    return {"values": with_noise(vals, region_noise(cloud, dev.mask)),
            "z_mm": float(np.median(P[:, 2])), "mask": dev.mask,
            "details": {"axis": f.axis.tolist(), "axis_point_mm": f.point.tolist(),
                        "radius_fixed_mm": radius_mm, "seen_length_mm": length,
                        "noise_sigma_mm": dev.noise_sigma_mm, "arc_deg": dev.arc_deg,
                        "sagitta_mm": sag, "off_fit_share": off,
                        "fitted_on": picked or ("the part above the background plane"
                                                if use is part else "the region")},
            "axis_line": (f.point + f.axis * float(np.percentile(along, 2)),
                          f.point + f.axis * float(np.percentile(along, 98)))}


# ------------------------------------------------------------------ bounding dimensions

def dims(cloud, region: Optional[np.ndarray] = None, min_height_mm: float = 2.0) -> dict:
    """Bounding dimensions of the part standing on the reference plane (the dominant plane of
    the scan): length x width = its minimum-area rectangle in that plane (mm, length the
    longer side; + one camera px footprint: the outermost px centres sit half a px inside
    the outline on each side, as measure/dimensions.py counts an extent), height = its
    highest surface above the plane (mm: the heights averaged over TOP_WINDOW_PX x
    TOP_WINDOW_PX camera px, then their 99.9th percentile - a stray point is not the part,
    and the noise of single px does not lift a flat top).

    The WHOLE part or nothing: a part that reaches the image border (BORDER_PX), or that
    goes on outside the roi / zone, is cut - its bounding size would be the cut's, so it is
    not measurable (MeasureError `incomplete`)."""
    import cv2
    from scipy import ndimage
    if not (math.isfinite(min_height_mm) and min_height_mm > 0):
        raise ValueError("min_height_mm must be a positive, finite number (mm)")
    if int(cloud.mask.sum()) < MIN_POINTS:
        raise MeasureError(f"only {int(cloud.mask.sum())} decoded points in the scan: no "
                           "reference plane to stand a part on")
    fg, bg = metric.foreground(cloud, min_height_mm=float(min_height_mm), region=region)
    if int(fg.sum()) < MIN_POINTS:
        raise MeasureError(f"no part found: only {int(fg.sum())} points stand more than "
                           f"{min_height_mm:g} mm above the reference plane"
                           + (" in the roi" if region is not None else ""))
    H_, W_ = fg.shape
    sides = [name for name, hit in (("left", fg[:, :BORDER_PX].any()),
                                    ("right", fg[:, W_ - BORDER_PX:].any()),
                                    ("top", fg[:BORDER_PX].any()),
                                    ("bottom", fg[H_ - BORDER_PX:].any())) if hit]
    if sides:
        raise MeasureError(f"the part is cut by the image border (it reaches the "
                           f"{' and '.join(sides)} edge of the camera image): its length and "
                           "width can't be read - move the whole part into the field",
                           "incomplete")
    if region is not None:                  # does the part go on outside the roi / zone?
        ha = np.full(fg.shape, -np.inf)
        ha[cloud.mask] = bg.distance(np.asarray(cloud.xyz[cloud.mask], np.float64))
        beyond = ndimage.binary_dilation(fg) & (ha > min_height_mm) & ~region
        if beyond.any():
            raise MeasureError(f"the roi cuts the part: it goes on outside the roi "
                               f"({int(beyond.sum())} px of it touch the roi's border from "
                               "outside), so its length and width can't be read - widen the "
                               "roi to hold the whole part (or leave it out)", "incomplete")
    P = np.asarray(cloud.xyz[fg], np.float64)
    h = bg.distance(P)
    n = bg.normal
    e1 = np.cross(n, [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    uv = np.c_[(P - bg.point) @ e1, (P - bg.point) @ e2].astype(np.float32)
    (cu, cv_), (w, ht), ang = cv2.minAreaRect(uv)
    # the camera's sampling of the part: the in-plane spacing of neighbouring part px
    pair = fg[:, 1:] & fg[:, :-1]
    step_ = np.asarray(cloud.xyz[:, 1:][pair], np.float64) - np.asarray(cloud.xyz[:, :-1][pair],
                                                                         np.float64)
    pitch = float(np.median(np.hypot(step_ @ e1, step_ @ e2))) if len(step_) else 0.0
    w, ht = w + pitch, ht + pitch
    length, width = (w, ht) if w >= ht else (ht, w)
    top = _top_height(fg, h)
    box = cv2.boxPoints(((cu, cv_), (w, ht), ang)).astype(np.float64)
    corners = [bg.point + u * e1 + v * e2 + top * n for u, v in box]
    n_pts = len(P)
    return {"values": with_noise([
        value("length", length, "mm", n_pts, "lateral", "Length (longer side)"),
        value("width", width, "mm", n_pts, "lateral", "Width"),
        value("height", top, "mm", n_pts, "depth", "Height above the reference plane")],
        region_noise(cloud, fg)),
        "z_mm": float(np.median(P[:, 2])), "mask": fg, "box": corners,
        "details": {"reference_plane_point_mm": bg.point.tolist(),
                    "reference_plane_normal": bg.normal.tolist(),
                    "reference_plane_rms_mm": bg.rms, "min_height_mm": float(min_height_mm),
                    "height_median_mm": float(np.median(h)), "px_pitch_mm": pitch,
                    "seen": "the surface the camera sees: undercuts and hidden walls are "
                            "not in the scan"}}


def _top_height(fg: np.ndarray, h: np.ndarray) -> float:
    """The part's highest surface: the 99.9th percentile of its heights after a TOP_WINDOW_PX
    box average over the part's own px (windows at least half full)."""
    from scipy import ndimage
    hm = np.zeros(fg.shape, np.float64)
    hm[fg] = h
    w = ndimage.uniform_filter(fg.astype(np.float64), TOP_WINDOW_PX, mode="constant")
    sm = ndimage.uniform_filter(hm, TOP_WINDOW_PX, mode="constant")
    ok = fg & (w >= 0.5)
    if int(ok.sum()) < MIN_POINTS:          # a sliver of a part: its own percentile
        return float(np.percentile(h, 99.5))
    return float(np.percentile(sm[ok] / w[ok], 99.9))


# ------------------------------------------------------------------ end faces

def endfaces(cells: list) -> dict:
    """The end faces a station measurement found (engine/measure/station: a cylinder's
    circular end face toward the camera; the rim circle fitted, free radius, to the flood
    outline in the face plane) -> per face its diameter and its flatness RMS."""
    vals, circles, zs = [], [], []
    for c in cells:
        name = str(c.get("name") or f"C{len(circles) + 1}")
        d = c.get("outline_diameter_mm")
        if d is None:
            raise MeasureError(f"end face {name}: no rim circle could be fitted to its outline "
                               "in the flood image (a fast scan keeps the flood pair; check "
                               "the exposure and that the whole face is lit)", "incomplete")
        n = int(c.get("n_pts") or 0)
        vals.append(value(f"{name.lower()}_diameter", d, "mm", n, "lateral",
                          f"{name} end-face diameter",
                          outline_rms_mm=c.get("outline_rms_mm")))
        vals.append(value(f"{name.lower()}_flatness_rms", c.get("flat_rms_mm"), "mm", n,
                          "depth", f"{name} end-face flatness (RMS)"))
        circles.append((c.get("camera_px"), c.get("r_px")))
        zs.append(float((c.get("centre_mm") or [0, 0, 0])[2]))
    if not vals:
        raise MeasureError("no end face found in the scan")
    return {"values": vals, "z_mm": float(np.median(zs)), "mask": None, "circles": circles,
            "details": {"faces": [{k: c.get(k) for k in ("name", "camera_px", "r_px",
                                                          "centre_mm", "normal", "coverage",
                                                          "outline_rms_mm", "centre_method")}
                                  for c in cells]}}


# ------------------------------------------------------------------ deviation vs a golden

def deviation(dev: np.ndarray, report: dict, region: Optional[np.ndarray] = None,
              part: Optional[np.ndarray] = None, max_dist_mm: float = 5.0) -> dict:
    """Signed deviation map vs a golden scan (reconstruct/metric_scan.compare_metric: mm,
    + toward the camera, NaN = not compared) -> its RMS, mean, p2-p98 and extremes. Values
    only: nothing is judged.

    part: the px of the part that was placed on the golden (compare_metric's largest
    foreground blob). Its points further than max_dist_mm from the golden have no deviation
    (NaN): when they form a patch (MIN_PATCH_PX wide and high) the largest deviation is NOT
    in the numbers - a boss, a missing wall, another part - so nothing is reported
    (MeasureError `incomplete`); single stray px (edges) are counted in a note."""
    from scipy import ndimage
    icp = report.get("icp") or {}
    if not icp.get("converged"):
        raise MeasureError("placing the scan on the golden (ICP) did not converge: every "
                           "deviation depends on it (check the placement and the golden)",
                           "incomplete")
    ok = np.isfinite(dev)
    if region is not None:
        ok &= region
    d = dev[ok]
    if d.size < MIN_POINTS:
        raise MeasureError(f"only {d.size} points could be compared with the golden: wrong "
                           "part, placement or roi?")
    n_pts = int(d.size)
    missed, stray = 0, None
    if part is not None:
        lost = np.asarray(part, bool) & ~np.isfinite(dev)
        if region is not None:
            lost &= region
        missed = int(lost.sum())
        n_part = n_pts + missed
        if missed and ndimage.binary_opening(
                lost, structure=np.ones((MIN_PATCH_PX, MIN_PATCH_PX), bool)).any():
            raise MeasureError(
                f"{missed} of the part's {n_part} points ({100.0 * missed / n_part:.1f} %) lie "
                f"more than {max_dist_mm:g} mm from the golden surface and were not compared: "
                f"the largest deviation is beyond {max_dist_mm:g} mm and is not in the numbers "
                "(a gross defect, another part, or the part is not where the golden was)",
                "incomplete")
        if missed:
            stray = (f"{missed} single points ({100.0 * missed / n_part:.2f} % of the part, "
                     f"stray px at its edges) lie more than {max_dist_mm:g} mm from the golden "
                     "and are not in this number")
    lo, hi = np.percentile(d, [TRIM_PCT, 100.0 - TRIM_PCT])
    out = _deviation_values(d, lo, hi, n_pts, ok, icp, report, missed, max_dist_mm)
    if stray:
        for v in out["values"]:
            v["notes"].append(stray)
    return out


def _deviation_values(d, lo, hi, n_pts, ok, icp, report, missed, max_dist_mm) -> dict:
    return {"values": [
        value("deviation_rms", math.sqrt(float(np.mean(d * d))), "mm", n_pts, "depth",
              "Deviation vs golden (RMS)"),
        value("deviation_mean", float(np.mean(d)), "mm", n_pts, "depth",
              "Deviation vs golden (mean)"),
        value("deviation_p2_p98", hi - lo, "mm", n_pts, "depth",
              "Deviation vs golden (p2-p98)"),
        value("deviation_max", float(d.max()), "mm", n_pts, "depth",
              "Largest deviation above the golden"),
        value("deviation_min", float(d.min()), "mm", n_pts, "depth",
              "Largest deviation below the golden")],
        "z_mm": None, "mask": ok,
        "details": {"icp_rms_mm": icp.get("rms_mm"), "icp_fitness": icp.get("fitness"),
                    "noise_sigma_mm": report.get("noise_sigma_mm"),
                    "golden": report.get("golden"), "max_dist_mm": max_dist_mm,
                    "n_not_compared": missed,
                    "compared": "the largest foreground part of each scan, test -> golden "
                                "(material the golden has and the scan lacks is not a "
                                "deviation of a scanned point: it is not in these numbers)"}}


# ------------------------------------------------------------------ noise

def measured_noise(cloud, region: Optional[np.ndarray] = None) -> Optional[float]:
    """The scan's own depth noise (mm, 1 sigma): a robust estimate from the second
    differences of z along the camera rows (z[i-1] - 2 z[i] + z[i+1] has variance 6 sigma^2
    for independent noise; a smooth surface adds next to nothing, an edge is an outlier the
    median ignores). None when too few neighbouring px are decoded."""
    m = cloud.mask if region is None else (cloud.mask & region)
    z = np.where(m, cloud.xyz[..., 2], np.nan)
    d2 = z[:, :-2] - 2.0 * z[:, 1:-1] + z[:, 2:]
    d2 = d2[np.isfinite(d2)]
    if d2.size < 200:
        return None
    mad = float(np.median(np.abs(d2 - np.median(d2))))
    return 1.4826 * mad / math.sqrt(6.0)

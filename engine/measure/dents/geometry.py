"""
Cell geometry for the dent detector, from the metric scans of the same pose.

A FIXED-radius cylinder (R = cell diameter / 2, a parameter per cell type)
is fitted to the metric cloud: per repeat the tall, depth-continuous
component is segmented (|dz| < 3 mm to each neighbour), the pooled points
are fitted, |residual| < 1.5 mm are kept and refitted. The fit starts with
the axis centre BEHIND the visible surface (along the viewing ray) and is
rejected if it converges in front of it (the concave solution).

Every crop pixel's camera ray is then cast onto the fitted cylinder:
  theta (deg, 0 = facing the camera, + toward camera +x), s (mm along the
  axis, + down the image), cosv (cos of view ray vs surface normal),
  the projector (col, row) the cylinder point maps to, and the projector
  shift per mm of depth along the camera ray (sens_ray, DMD col / row).
The analysed FACE is cosv > 0.3 (normal within ~72 deg of the ray), from
2.5 mm below the decoded top end to the decoded bottom end (never past it).
Metric radial deviation per repeat (mm, + = outward, NaN beyond +-3 mm or
undecoded) is kept for depth measurement.

Needs the stereo calibration the metric scans were triangulated with; the
caller (verbs.py) checks it is the current one for the geometry (rule 6).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares

from ...reconstruct.triangulate import camera_rays
from ._util import Quiet

FACE_COSV = 0.3
END_MARGIN_MM = 2.5
BOTTOM_EXTEND_MM = 0.0
TRIM_MM = 1.5
DEV_MAX_MM = 3.0


# ------------------------------------------------------------ metric data

def load_metric(dirs, x0: int, y0: int, shape: tuple) -> list:
    """Per metric scan folder -> dict(name, xyz (H, W, 3) mm, mask, corr_x, corr_y), crop coords."""
    H, W = shape
    out = []
    for d in dirs:
        d = Path(d)
        with np.load(d / "cloud.npz") as c:
            xyz = np.asarray(c["xyz"][y0:y0 + H, x0:x0 + W], np.float64)
            mask = np.asarray(c["mask"][y0:y0 + H, x0:x0 + W], bool)
        cx = cy = None
        if (d / "field.npz").is_file():
            with np.load(d / "field.npz") as f:
                fm = np.asarray(f["mask"][y0:y0 + H, x0:x0 + W], bool)
                if "corr_x" in f.files:
                    cx = np.where(fm, f["corr_x"][y0:y0 + H, x0:x0 + W], np.nan).astype(np.float64)
                if "corr_y" in f.files:
                    cy = np.where(fm, f["corr_y"][y0:y0 + H, x0:x0 + W], np.nan).astype(np.float64)
        if xyz.shape[:2] != (H, W):
            raise ValueError(f"metric scan {d.name} does not cover the dent set's crop")
        mask &= np.isfinite(xyz).all(-1)
        out.append({"name": d.name, "xyz": xyz, "mask": mask, "corr_x": cx, "corr_y": cy})
    return out


def segment(xyz: np.ndarray, mask: np.ndarray, min_px: int = 300) -> np.ndarray:
    """The tall depth-continuous component (the standing cell)."""
    Z = np.where(mask, xyz[..., 2], np.nan)
    Zf = np.nan_to_num(Z, nan=1e9)
    gy = np.abs(np.diff(Zf, axis=0, prepend=Zf[:1])) < 3
    gx = np.abs(np.diff(Zf, axis=1, prepend=Zf[:, :1])) < 3
    ok = mask & (Z > 0) & gy & gx
    lab, n = ndimage.label(ok)
    best, bs = 0, -1.0
    for k, sl in enumerate(ndimage.find_objects(lab), 1):
        m = lab[sl] == k
        npx = int(m.sum())
        if npx < min_px:
            continue
        hgt = sl[0].stop - sl[0].start
        wid = max(sl[1].stop - sl[1].start, 1)
        sc = hgt * min(npx, 5000) / wid
        if sc > bs:
            best, bs = k, sc
    return lab == best if best else np.zeros_like(mask)


# ------------------------------------------------------------ cylinder

def _unpack(q):
    a = np.array([q[0], 1.0, q[1]])
    a /= np.linalg.norm(a)
    return a, np.array([q[2], 0.0, q[3]])


def radial_dev(X, a, c, R):
    return np.linalg.norm(np.cross(X - c, a), axis=-1) - R


def fit_cylinder_fixed(P: np.ndarray, R: float, f_scale: float = 0.1, trim: float = 0.6):
    """Fixed-radius cylinder, axis roughly along camera y (a standing cell).
    -> (axis a, point c on the axis at y = 0, behind: centre farther than the surface)."""
    P = np.asarray(P, np.float64)
    m = np.median(P, 0)
    view = m / np.linalg.norm(m)
    # initial axis: principal direction of the points (a tall patch), y down
    _, _, vt = np.linalg.svd(P[:: max(1, len(P) // 20000)] - m, full_matrices=False)
    a0 = vt[0] if vt[0][1] >= 0 else -vt[0]
    if abs(a0[1]) < 0.5:
        a0 = np.array([0.0, 1.0, 0.0])
    a0 = a0 / a0[1]

    def res(q, PP):
        a, cc = _unpack(q)
        return np.linalg.norm(np.cross(PP - cc, a), axis=1) - R

    best = None
    for push in (1.0, 1.5, 2.0):
        c0 = m + push * R * view                         # behind the visible surface
        c0 = c0 - (c0[1] / a0[1]) * a0 if abs(a0[1]) > 1e-6 else c0
        q = least_squares(res, [a0[0], a0[2], c0[0], c0[2]], args=(P,), loss="soft_l1",
                          f_scale=f_scale, x_scale="jac").x
        for _ in range(2):
            d = res(q, P)
            keep = np.abs(d - np.median(d)) < trim
            if keep.sum() < 50:
                break
            q = least_squares(res, q, args=(P[keep],), loss="soft_l1", f_scale=f_scale / 2,
                              x_scale="jac").x
        a, cc = _unpack(q)
        cm = cc + ((m - cc) @ a) * a
        behind = bool(np.linalg.norm(cm) > np.linalg.norm(m))
        if behind:
            return a, cc, True
        best = best or (a, cc, False)
    return best


def surface_frame(X, a, c):
    """theta (deg, 0 = toward the camera, + toward camera +x) and s (mm along a)."""
    w = X - c
    s = w @ a
    rad = w - s[..., None] * a
    cam = -c
    e0 = cam - (cam @ a) * a
    e0 /= np.linalg.norm(e0)
    e1 = np.cross(a, e0)
    e1 = e1 if e1[0] >= 0 else -e1
    th = np.degrees(np.arctan2(rad @ e1, rad @ e0))
    return th, s


def project(cal, X: np.ndarray) -> np.ndarray:
    """Camera-frame points (..., 3) -> projector (col, row) (pinhole + skew)."""
    Kp = np.asarray(cal.K_proj, float)
    Q = X @ np.asarray(cal.R, float).T + np.ravel(np.asarray(cal.t, float))
    with Quiet():
        x = Q[..., 0] / Q[..., 2]
        y = Q[..., 1] / Q[..., 2]
    return np.stack([Kp[0, 0] * x + Kp[0, 1] * y + Kp[0, 2], Kp[1, 1] * y + Kp[1, 2]], -1)


# ------------------------------------------------------------ the cell

def cell_geometry(metric: list, cal, radius_mm: float, x0: int, y0: int, shape: tuple,
                  face_cosv: float = FACE_COSV, end_margin_mm: float = END_MARGIN_MM,
                  bottom_extend_mm: float = BOTTOM_EXTEND_MM) -> dict:
    """metric: load_metric() output of the same pose; cal: its StereoCalibration."""
    if not metric:
        raise ValueError("the dent detector needs at least one metric scan of the same pose")
    H, W = shape
    R = float(radius_mm)
    pts, segs, warnings = [], [], []
    for mm in metric:
        sg = segment(mm["xyz"], mm["mask"])
        segs.append(sg)
        pts.append(mm["xyz"][sg])
    P = np.concatenate(pts) if pts else np.zeros((0, 3))
    if len(P) < 300:
        raise ValueError(f"only {len(P)} metric points on the cell (need >= 300): is the cell in "
                         "the zone, and was the metric scan decoded?")
    a, c, behind = fit_cylinder_fixed(P, R)
    d = radial_dev(P, a, c, R)
    keep = np.abs(d) < TRIM_MM
    if keep.sum() >= 300:
        a, c, behind = fit_cylinder_fixed(P[keep], R)
    if not behind:
        raise ValueError("the fixed-radius cylinder fit converged with its axis IN FRONT of the "
                         "visible surface (wrong cell diameter, or the metric cloud is not a cell)")
    d = radial_dev(P, a, c, R)
    inl = np.abs(d) < TRIM_MM
    fit_rms = float(np.sqrt(np.mean(d[inl] ** 2))) if inl.any() else float("nan")
    # rays through every crop pixel (full-camera pixel centres)
    v, u = np.mgrid[0:H, 0:W].astype(np.float64)
    rays = camera_rays(cal, None, np.stack([u + x0, v + y0], -1))
    dperp = rays - (rays @ a)[..., None] * a
    cperp = -(c - (c @ a) * a)
    A = np.einsum("...i,...i", dperp, dperp)
    B = 2 * (dperp @ cperp)
    C = cperp @ cperp - R ** 2
    disc = B * B - 4 * A * C
    hit = disc > 0
    t = (-B - np.sqrt(np.maximum(disc, 0))) / (2 * A)
    X = rays * t[..., None]
    th, s = surface_frame(X, a, c)
    w = X - c
    nrm = w - (w @ a)[..., None] * a
    nrm /= np.maximum(np.linalg.norm(nrm, axis=-1, keepdims=True), 1e-12)
    ru = rays / np.linalg.norm(rays, axis=-1, keepdims=True)
    cosv = -np.einsum("...i,...i", nrm, ru)
    uv = project(cal, X)
    sens_ray = (project(cal, X + 0.05 * ru) - uv) / 0.05          # DMD (col, row) per mm deeper
    devs = []
    for mm in metric:
        dv = radial_dev(mm["xyz"], a, c, R)
        dv[~mm["mask"] | ~np.isfinite(dv) | (np.abs(dv) > DEV_MAX_MM)] = np.nan
        devs.append(dv)
    ss = np.concatenate([surface_frame(mm["xyz"][sg], a, c)[1] for mm, sg in zip(metric, segs)])
    s_range = (float(np.percentile(ss, 0.5)), float(np.percentile(ss, 99.5)))
    theta = np.where(hit, th, np.nan)
    s_map = np.where(hit, s, np.nan)
    cosv = np.where(hit, cosv, np.nan)
    with Quiet():
        face = (np.isfinite(cosv) & (cosv > face_cosv) & (s_map > s_range[0] + end_margin_mm)
                & (s_map < s_range[1] + bottom_extend_mm))
    if face.sum() < 200:
        raise ValueError(f"the analysed cell face has only {int(face.sum())} px: check the cell "
                         "diameter and that the metric scans are of this pose")
    # camera px per mm along the surface near the face centre (for sizes)
    zs = t[face] * np.linalg.norm(rays[face], axis=-1)
    f_px = float(np.asarray(cal.K_cam, float)[0, 0])
    mm_per_px = float(np.median(zs) / f_px)
    cam_c = -c
    toward = cam_c - (cam_c @ a) * a
    return {"axis": a, "centre": c, "radius_mm": R, "behind": behind, "fit_rms_mm": fit_rms,
            "n_points": int(len(P)), "n_inliers": int(inl.sum()), "hit": hit, "theta": theta,
            "s": s_map, "cosv": cosv, "X": X, "normal": nrm, "uv": uv, "sens_ray": sens_ray,
            "devs": np.stack(devs), "s_range": s_range, "face": face, "mm_per_px": mm_per_px,
            "segments": segs, "x0": int(x0), "y0": int(y0), "shape": (H, W),
            "axis_distance_mm": float(np.linalg.norm(toward)), "warnings": warnings}

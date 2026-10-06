"""
Metric measurement on a triangulated cloud (reconstruct/triangulate.py,
mm in the camera frame): shape fits, a cylinder radial-deviation map with
dent blobs, point-to-plane ICP to a golden cloud, and the known-object
accuracy check.

Everything here takes a MetricCloud, which only exists with a matching
stereo calibration (rule 6), so every number is mm. No Open3D: numpy +
scipy (cKDTree, least_squares).

Sign convention for deviations: residual = measured - nominal along the
outward surface normal, so a dent (material pushed in) is NEGATIVE.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import ndimage

SHAPES = ("plane", "step", "cylinder", "sphere")


# ------------------------------------------------------------------ fits

@dataclass
class PlaneFit:
    point: np.ndarray
    normal: np.ndarray           # unit, oriented toward the camera (normal . point < 0)
    rms: float
    n: int

    def distance(self, P: np.ndarray) -> np.ndarray:
        """Signed distance, + toward the camera."""
        return (P - self.point) @ self.normal


def fit_plane(P: np.ndarray) -> PlaneFit:
    P = np.asarray(P, np.float64)
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    n = vt[2]
    if n @ c > 0:
        n = -n
    d = (P - c) @ n
    return PlaneFit(c, n, float(np.sqrt(np.mean(d * d))), len(P))


def ransac_plane(P: np.ndarray, thresh_mm: float = 0.5, iters: int = 300,
                 sample: int = 20000, seed: int = 0) -> tuple:
    """Dominant plane -> (PlaneFit refined on its inliers, inlier mask over P)."""
    rng = np.random.default_rng(seed)
    P = np.asarray(P, np.float64)
    S = P if len(P) <= sample else P[rng.choice(len(P), sample, replace=False)]
    best, best_n = None, -1
    for _ in range(iters):
        a, b, c = S[rng.choice(len(S), 3, replace=False)]
        n = np.cross(b - a, c - a)
        ln = np.linalg.norm(n)
        if ln < 1e-9:
            continue
        n /= ln
        k = int((np.abs((S - a) @ n) < thresh_mm).sum())
        if k > best_n:
            best, best_n = (a, n), k
    a, n = best
    inl = np.abs((P - a) @ n) < thresh_mm
    fit = fit_plane(P[inl])
    inl = np.abs(fit.distance(P)) < thresh_mm
    return fit_plane(P[inl]), inl


@dataclass
class SphereFit:
    center: np.ndarray
    radius: float
    rms: float
    n: int

    @property
    def diameter(self) -> float:
        return 2 * self.radius


def fit_sphere(P: np.ndarray) -> SphereFit:
    from scipy.optimize import least_squares
    P = np.asarray(P, np.float64)
    A = np.c_[2 * P, np.ones(len(P))]
    sol, *_ = np.linalg.lstsq(A, (P * P).sum(1), rcond=None)
    c0 = sol[:3]
    r0 = math.sqrt(max(sol[3] + c0 @ c0, 1e-9))
    res = least_squares(lambda q: np.linalg.norm(P - q[:3], axis=1) - q[3],
                        np.r_[c0, r0], method="lm")
    c, r = res.x[:3], abs(res.x[3])
    e = np.linalg.norm(P - c, axis=1) - r
    return SphereFit(c, float(r), float(np.sqrt(np.mean(e * e))), len(P))


@dataclass
class CylinderFit:
    point: np.ndarray            # on the axis (nearest the data centroid)
    axis: np.ndarray             # unit direction
    radius: float
    rms: float                   # of all residuals (dents included)
    n: int

    @property
    def diameter(self) -> float:
        return 2 * self.radius

    def radial(self, P: np.ndarray) -> np.ndarray:
        w = P - self.point
        return np.linalg.norm(w - np.outer(w @ self.axis, self.axis), axis=1)

    def residual(self, P: np.ndarray) -> np.ndarray:
        """Radial deviation, mm (negative = inside the fitted cylinder = dent)."""
        return self.radial(P) - self.radius

    def frame(self, towards=np.zeros(3)):
        """(e0 toward `towards` (the camera) perpendicular to the axis, e1 = axis x e0)."""
        w = np.asarray(towards, float) - self.point
        e0 = w - (w @ self.axis) * self.axis
        e0 /= max(np.linalg.norm(e0), 1e-12)
        e1 = np.cross(self.axis, e0)
        return e0, (e1 if e1[0] >= 0 else -e1)          # + theta = toward camera +x


def _circle_kasa(x, y):
    A = np.c_[2 * x, 2 * y, np.ones(len(x))]
    sol, *_ = np.linalg.lstsq(A, x * x + y * y, rcond=None)
    return sol[0], sol[1], math.sqrt(max(sol[2] + sol[0] ** 2 + sol[1] ** 2, 1e-9))


def fit_cylinder(P: np.ndarray, axis_hint=None, radius: Optional[float] = None,
                 f_scale_mm: float = 0.05, max_points: int = 40000, seed: int = 0,
                 trim: bool = True) -> CylinderFit:
    """Least-squares cylinder, robust (soft-L1) so dents do not pull it.
    axis_hint: approximate axis direction (e.g. camera y for a standing
    cell); None tries the three principal directions. radius: fix it to a
    known value (else fitted). trim: refit on the inliers of the robust fit."""
    from scipy.optimize import least_squares
    P = np.asarray(P, np.float64)
    rng = np.random.default_rng(seed)
    S = P if len(P) <= max_points else P[rng.choice(len(P), max_points, replace=False)]
    cen = S.mean(0)
    if axis_hint is not None:
        cands = [np.asarray(axis_hint, float) / np.linalg.norm(axis_hint)]
    else:
        _, _, vt = np.linalg.svd(S - cen, full_matrices=False)
        cands = list(vt)
    best = None
    for a0 in cands:
        e1 = np.cross(a0, [1.0, 0, 0] if abs(a0[0]) < 0.9 else [0, 1.0, 0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(a0, e1)
        w = S - cen
        cx, cy, r0 = _circle_kasa(w @ e1, w @ e2)
        if radius is not None:
            r0 = float(radius)

        def unpack(q):
            a = a0 + q[0] * e1 + q[1] * e2
            a /= np.linalg.norm(a)
            p = cen + q[2] * e1 + q[3] * e2
            r = float(radius) if radius is not None else q[4]
            return p, a, r

        def resid(q):
            p, a, r = unpack(q)
            ww = S - p
            return np.linalg.norm(ww - np.outer(ww @ a, a), axis=1) - r

        q0 = np.array([0.0, 0.0, cx, cy] + ([] if radius is not None else [r0]))
        res = least_squares(resid, q0, loss="soft_l1", f_scale=f_scale_mm, x_scale="jac")
        if best is None or res.cost < best[0]:
            best = (res.cost, unpack(res.x))
    p, a, r = best[1]
    if trim:
        # second pass on the inliers only: dents / bumps deeper than
        # max(3 sigma, 10 um) no longer pull the surface at all
        w = S - p
        e = np.linalg.norm(w - np.outer(w @ a, a), axis=1) - r
        med = float(np.median(e))
        sig = 1.4826 * float(np.median(np.abs(e - med)))
        keep = np.abs(e - med) < max(3 * sig, 0.01)
        if keep.sum() >= 50 and not keep.all():
            e1 = np.cross(a, [1.0, 0, 0] if abs(a[0]) < 0.9 else [0, 1.0, 0])
            e1 /= np.linalg.norm(e1)
            e2 = np.cross(a, e1)
            Sk = S[keep]

            def resid2(q):
                aa = a + q[0] * e1 + q[1] * e2
                aa /= np.linalg.norm(aa)
                pp = p + q[2] * e1 + q[3] * e2
                rr = float(radius) if radius is not None else q[4]
                ww = Sk - pp
                return np.linalg.norm(ww - np.outer(ww @ aa, aa), axis=1) - rr

            q0 = np.zeros(4 + (0 if radius is not None else 1))
            if radius is None:
                q0[4] = r
            q = least_squares(resid2, q0, x_scale="jac").x
            a = a + q[0] * e1 + q[1] * e2
            a /= np.linalg.norm(a)
            p = p + q[2] * e1 + q[3] * e2
            r = float(radius) if radius is not None else q[4]
    ref = np.asarray(axis_hint, float) if axis_hint is not None else np.eye(3)[np.argmax(np.abs(a))]
    if a @ ref < 0:
        a = -a                                          # axis sign: along the hint / +major
    p = p + ((P.mean(0) - p) @ a) * a
    fit = CylinderFit(p, a, float(r), 0.0, len(P))
    e = fit.residual(P)
    fit.rms = float(np.sqrt(np.mean(e * e)))
    return fit


# ------------------------------------------------------- segmentation

def foreground(cloud, min_height_mm: float = 2.0, plane_thresh_mm: float = 0.5,
               region: Optional[np.ndarray] = None):
    """Object in front of the dominant (background) plane: -> (mask (H, W) of
    the largest connected foreground blob [inside region], background PlaneFit)."""
    m = cloud.mask
    P = cloud.xyz[m]
    bg, _ = ransac_plane(P, plane_thresh_mm)
    h = np.full(m.shape, np.nan)
    h[m] = bg.distance(P)
    fg = m & (np.nan_to_num(h, nan=-1e9) > min_height_mm)
    if region is not None:
        fg &= region
    lab, n = ndimage.label(fg)
    if n > 1:
        sizes = ndimage.sum(fg, lab, range(1, n + 1))
        fg = lab == (int(np.argmax(sizes)) + 1)
    return fg, bg


def pixel_area_mm2(xyz: np.ndarray) -> np.ndarray:
    """Surface area each camera pixel covers (mm^2), from grid neighbours."""
    du = np.gradient(xyz, axis=1)
    dv = np.gradient(xyz, axis=0)
    return np.linalg.norm(np.cross(du, dv), axis=2)


# ------------------------------------------------ cylinder deviation map

@dataclass
class MetricDent:
    depth_mm: float              # > 0: how far the surface is pushed in
    area_mm2: float              # surface area below the detection threshold
    n_px: int
    centroid_mm: list            # camera frame
    theta_deg: float             # about the axis, 0 = facing the camera, + toward camera +x
    z_mm: float                  # along the axis from the fitted axis point
    bbox_px: list                # [x0, y0, x1, y1]
    snr: float


@dataclass
class CylinderDeviation:
    fit: CylinderFit
    residual: np.ndarray         # (H, W) mm, NaN off the cylinder
    mask: np.ndarray
    noise_sigma_mm: float
    threshold_mm: float
    dents: list = field(default_factory=list)
    arc_deg: Optional[float] = None
    units: str = "mm"
    warnings: list = field(default_factory=list)
    files: dict = field(default_factory=dict)


def _bowl_depth(res: np.ndarray, det: np.ndarray, sl, blob: np.ndarray, half: int = 2) -> float:
    """Dent depth: a quadratic surface fitted to the RAW residuals in a
    (2 half + 1)^2 window around the smoothed minimum, evaluated at its own
    minimum (unbiased by the smoothing, averages the noise). Falls back to
    the smoothed minimum when the window is not bowl-shaped."""
    d = np.where(blob, det[sl], np.inf)
    iy, ix = np.unravel_index(int(np.argmin(d)), d.shape)
    y, x = iy + sl[0].start, ix + sl[1].start
    fallback = float(-d[iy, ix])
    H, W = res.shape
    if y < half or x < half or y + half >= H or x + half >= W:
        return fallback
    win = res[y - half:y + half + 1, x - half:x + half + 1]
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1]
    ok = np.isfinite(win)
    if ok.sum() < 10:
        return fallback
    X, Y, Z = xx[ok].astype(float), yy[ok].astype(float), win[ok]
    A = np.c_[np.ones_like(X), X, Y, X * X, X * Y, Y * Y]
    a, b, c, dd, e, f = np.linalg.lstsq(A, Z, rcond=None)[0]
    Hs = np.array([[2 * dd, e], [e, 2 * f]])
    if np.linalg.det(Hs) <= 0 or Hs[0, 0] <= 0:
        return fallback
    p = np.linalg.solve(Hs, [-b, -c])
    if np.any(np.abs(p) > half):
        return fallback
    zmin = a + b * p[0] + c * p[1] + dd * p[0] ** 2 + e * p[0] * p[1] + f * p[1] ** 2
    return float(-zmin)


def cylinder_deviation(cloud, region: Optional[np.ndarray] = None, axis_hint=(0.0, 1.0, 0.0),
                       radius_mm: Optional[float] = None, n_sigma: float = 4.0,
                       min_depth_mm: float = 0.05, min_area_mm2: float = 0.5,
                       smooth_px: float = 1.0) -> CylinderDeviation:
    """Cylinder fit on the cloud -> radial deviation map (mm) + dent blobs
    (connected negative residuals below -max(n_sigma * noise, min_depth))."""
    warnings = []
    if region is None:
        region, _ = foreground(cloud)
    m = region & cloud.mask
    if m.sum() < 200:
        raise ValueError(f"only {int(m.sum())} cylinder points; nothing to fit")
    P = cloud.xyz[m]
    fit = fit_cylinder(P, axis_hint=axis_hint, radius=radius_mm)
    res = np.full(m.shape, np.nan)
    res[m] = fit.residual(P)
    r = res[m]
    med = float(np.median(r))
    sigma = 1.4826 * float(np.median(np.abs(r - med)))
    if smooth_px > 0:
        w = ndimage.gaussian_filter(m.astype(float), smooth_px)
        sm = ndimage.gaussian_filter(np.where(m, res, 0.0), smooth_px)
        det = np.where(m & (w > 0.3), sm / np.maximum(w, 1e-9), np.nan)
    else:
        det = res
    thr = max(n_sigma * sigma, min_depth_mm)
    below = m & (np.nan_to_num(det, nan=0.0) < -thr)
    lab, n = ndimage.label(below)
    area = pixel_area_mm2(np.where(m[..., None], cloud.xyz, np.nan))
    e0, e1 = fit.frame()
    dents = []
    for k, sl in enumerate(ndimage.find_objects(lab), 1):
        blob = lab[sl] == k
        a = float(np.nansum(area[sl][blob]))
        if a < min_area_mm2:
            continue
        depth = _bowl_depth(res, det, sl, blob)
        pts = cloud.xyz[sl][blob]
        c = pts.mean(0)
        w = c - fit.point
        dents.append(MetricDent(
            depth_mm=round(depth, 4), area_mm2=round(a, 3), n_px=int(blob.sum()),
            centroid_mm=[round(float(x), 3) for x in c],
            theta_deg=round(math.degrees(math.atan2(w @ e1, w @ e0)), 2),
            z_mm=round(float(w @ fit.axis), 3),
            bbox_px=[sl[1].start, sl[0].start, sl[1].stop, sl[0].stop],
            snr=round(depth / sigma, 1) if sigma > 0 else float("inf")))
    dents.sort(key=lambda d: -d.depth_mm)
    th = np.arctan2((P - fit.point) @ e1, (P - fit.point) @ e0)
    arc = math.degrees(float(np.percentile(th, 99.5) - np.percentile(th, 0.5)))
    if sigma > 0.5 * min_depth_mm and n_sigma * sigma > min_depth_mm:
        warnings.append(f"noise sigma {sigma:.3f} mm: dents shallower than {thr:.3f} mm "
                        "can't be told from noise")
    return CylinderDeviation(fit=fit, residual=res, mask=m, noise_sigma_mm=sigma,
                             threshold_mm=thr, dents=dents, arc_deg=round(arc, 1),
                             warnings=warnings)


# ------------------------------------------------------------------- ICP

def estimate_normals(P: np.ndarray, k: int = 12) -> np.ndarray:
    from scipy.spatial import cKDTree
    _, nn = cKDTree(P).query(P, k=min(k, len(P)))
    Q = P[nn] - P[nn].mean(1, keepdims=True)
    C = np.einsum("nki,nkj->nij", Q, Q)
    _, vec = np.linalg.eigh(C)
    n = vec[:, :, 0]
    return n * np.where((n * P).sum(1) > 0, -1.0, 1.0)[:, None]    # toward the camera


def _rodrigues(w: np.ndarray) -> np.ndarray:
    th = np.linalg.norm(w)
    if th < 1e-15:
        return np.eye(3)
    k = w / th
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * Kx + (1 - math.cos(th)) * Kx @ Kx


@dataclass
class ICPResult:
    R: np.ndarray                # aligned = R @ src + t
    t: np.ndarray
    rms_mm: float                # point-to-plane over the final pairs
    fitness: float               # fraction of source points paired
    iterations: int
    converged: bool


def icp_point_to_plane(src: np.ndarray, dst: np.ndarray, dst_normals: Optional[np.ndarray] = None,
                       max_iter: int = 50, max_dist_mm: float = 5.0, tol: float = 1e-8,
                       max_points: int = 30000, seed: int = 0, R0=None, t0=None) -> ICPResult:
    """Point-to-plane ICP (linearised, Rodrigues update) aligning src onto dst."""
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(seed)
    src = np.asarray(src, np.float64)
    dst = np.asarray(dst, np.float64)
    if len(src) > max_points:
        src = src[rng.choice(len(src), max_points, replace=False)]
    if dst_normals is None:
        dst_normals = estimate_normals(dst)
    tree = cKDTree(dst)
    R = np.eye(3) if R0 is None else np.asarray(R0, float)
    t = np.zeros(3) if t0 is None else np.asarray(t0, float)
    converged, it, rms, fit = False, 0, float("nan"), 0.0
    for it in range(1, max_iter + 1):
        p = src @ R.T + t
        dist, j = tree.query(p, distance_upper_bound=max_dist_mm)
        ok = np.isfinite(dist)
        if ok.sum() < 6:
            break
        p, q, n = p[ok], dst[j[ok]], dst_normals[j[ok]]
        r = np.einsum("ij,ij->i", p - q, n)
        J = np.c_[np.cross(p, n), n]
        x = np.linalg.lstsq(J, -r, rcond=None)[0]
        dR = _rodrigues(x[:3])
        R, t = dR @ R, dR @ t + x[3:]
        rms, fit = float(np.sqrt(np.mean(r * r))), float(ok.mean())
        if np.linalg.norm(x) < tol:
            converged = True
            break
    p = src @ R.T + t
    dist, j = tree.query(p, distance_upper_bound=max_dist_mm)
    ok = np.isfinite(dist)
    if ok.any():
        r = np.einsum("ij,ij->i", p[ok] - dst[j[ok]], dst_normals[j[ok]])
        rms, fit = float(np.sqrt(np.mean(r * r))), float(ok.mean())
    return ICPResult(R, t, rms, fit, it, converged)


def deviation_to(cloud, golden, icp: ICPResult, max_dist_mm: float = 5.0) -> np.ndarray:
    """(H, W) signed deviation (mm, + toward the camera) of the aligned scan
    from the golden surface (point-to-plane at the nearest golden point)."""
    from scipy.spatial import cKDTree
    G = golden.xyz[golden.mask]
    nG = estimate_normals(G)
    P = cloud.xyz[cloud.mask] @ icp.R.T + icp.t
    dist, j = cKDTree(G).query(P, distance_upper_bound=max_dist_mm)
    ok = np.isfinite(dist)
    d = np.full(len(P), np.nan)
    d[ok] = np.einsum("ij,ij->i", P[ok] - G[j[ok]], nG[j[ok]])
    out = np.full(cloud.mask.shape, np.nan)
    out[cloud.mask] = d
    return out


# ------------------------------------------------------- accuracy check

def measure_known(cloud, kind: str, known: dict, region: Optional[np.ndarray] = None) -> dict:
    """One cloud, one known object -> {"measured": value (mm), "error": measured - known, ...}.
    plane: flatness RMS of the dominant plane (known flatness 0)
    step: height of the raised face above the dominant plane (known["height_mm"])
    cylinder / sphere: fitted diameter (known["diameter_mm"])"""
    if kind not in SHAPES:
        raise ValueError(f"kind must be one of {SHAPES}, got {kind!r}")
    m = cloud.mask
    if kind == "plane":
        pts = cloud.xyz[m & region] if region is not None else cloud.xyz[m]
        fit, inl = ransac_plane(pts, 0.5)
        d = fit.distance(pts[inl])
        return {"measured": fit.rms, "error": fit.rms, "flatness_max_mm": float(np.abs(d).max()),
                "inlier_pct": round(100.0 * float(inl.mean()), 2), "n": int(inl.sum()),
                "quantity": "flatness rms"}
    key = "height_mm" if kind == "step" else "diameter_mm"
    if key not in known:
        raise ValueError(f"{kind}: known[{key!r}] is required")
    nominal = float(known[key])
    min_h = min(2.0, 0.4 * nominal) if kind == "step" else 2.0
    fg, bg = foreground(cloud, min_height_mm=min_h, region=region)
    if fg.sum() < 100:
        raise ValueError(f"{kind}: no object found in front of the background plane")
    P = cloud.xyz[fg]
    if kind == "step":
        # the top face: a plane through the foreground, its height above the base
        top, inl = ransac_plane(P, 0.3)
        h = bg.distance(P[inl])
        val = float(np.median(h))
        extra = {"parallel_deg": round(math.degrees(math.acos(min(1.0, abs(top.normal @ bg.normal)))), 4),
                 "top_rms_mm": top.rms}
    elif kind == "cylinder":
        f = fit_cylinder(P, axis_hint=known.get("axis_hint", (0.0, 1.0, 0.0)))
        val, extra = f.diameter, {"fit_rms_mm": f.rms, "axis": f.axis.round(5).tolist()}
    else:
        f = fit_sphere(P)
        val, extra = f.diameter, {"fit_rms_mm": f.rms, "center_mm": f.center.round(3).tolist()}
    return {"measured": val, "error": val - nominal, "nominal": nominal, "n": int(fg.sum()),
            "quantity": "height" if kind == "step" else "diameter", **extra}


def summarize(rows: list, tol_mm: float = 0.1, repeat_tol_mm: float = 0.03) -> dict:
    """Per-repeat measure_known rows -> mean / max error, repeatability, verdict."""
    err = np.array([r["error"] for r in rows], float)
    val = np.array([r["measured"] for r in rows], float)
    rep = float(val.std(ddof=1)) if len(val) > 1 else None
    mean_err, max_err = float(np.mean(err)), float(np.max(np.abs(err)))
    ok = abs(mean_err) <= tol_mm and (rep is None or rep <= repeat_tol_mm)
    return {"n_repeats": len(rows), "mean_error_mm": mean_err, "max_abs_error_mm": max_err,
            "repeatability_sigma_mm": rep, "mean_measured_mm": float(val.mean()),
            "tol_mm": tol_mm, "repeat_tol_mm": repeat_tol_mm,
            "verdict": "GO" if ok else "NO-GO", "units": "mm"}

"""
Metric triangulation: decoded projector coordinates + a stereo calibration
-> a point cloud in mm (camera frame: x right, y down, z forward).

  ray-plane  (column axis only): the decoded column u is a plane through the
             projector centre, n_p = (f_u, s, c_u - u) in the projector
             frame (s = skew); the camera ray through the undistorted pixel
             meets it at one point.
  ray-ray    (both axes): the camera ray and the projector ray through
             (u, v) generally miss each other by a little; the point is the
             midpoint of their common perpendicular and the miss distance
             is the per-pixel CONSISTENCY ERROR (mm). It needs no ground
             truth: patches where it is large mean a calibration or a
             diamond-parity convention problem (work plan section 4).

With both axes the default ("auto") reports ray-ray points; ray-plane is
always available as `method="plane"`. Pixels are the camera pixel centres
the decode ran on. The projector is treated as distortion-free when its
calibration says so (dist_proj all zero); otherwise the column plane is
bent, and the ray-ray path (which undistorts (u, v)) is the correct one.

mm only with a stereo calibration at the field's geometry and fingerprint
(rule 6) — the caller (api) resolves it with calibrate.stereo.require().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ..capture.mock_stereo import undistort_normalized

METHODS = ("auto", "plane", "rays")


@dataclass
class MetricCloud:
    xyz: np.ndarray                    # (H, W, 3) mm, NaN where invalid
    mask: np.ndarray                   # (H, W) valid
    consistency: Optional[np.ndarray]  # (H, W) mm ray-ray miss distance (both axes), else None
    method: str
    geometry_id: str
    calibration_id: str
    units: str = "mm"
    meta: dict = field(default_factory=dict)

    @property
    def points(self) -> np.ndarray:
        return self.xyz[self.mask]

    @property
    def shape(self):
        return self.mask.shape

    def consistency_stats(self) -> Optional[dict]:
        if self.consistency is None or not self.mask.any():
            return None
        c = self.consistency[self.mask]
        return {"median_mm": float(np.median(c)), "p95_mm": float(np.percentile(c, 95)),
                "max_mm": float(c.max())}

    def save_npz(self, f) -> None:
        arrays = {"xyz": self.xyz.astype(np.float32), "mask": self.mask}
        if self.consistency is not None:
            arrays["consistency"] = self.consistency.astype(np.float32)
        np.savez_compressed(f, **arrays)

    @classmethod
    def load(cls, path: Path, meta: dict) -> "MetricCloud":
        with np.load(path) as z:
            return cls(xyz=z["xyz"].astype(np.float64), mask=z["mask"],
                       consistency=z["consistency"] if "consistency" in z.files else None,
                       method=meta.get("method", "?"), geometry_id=meta.get("geometry_id", ""),
                       calibration_id=meta.get("calibration_id", ""), meta=meta)


def camera_rays(cal, shape, pixels: Optional[np.ndarray] = None) -> np.ndarray:
    """Undistorted camera ray directions (z = 1) for every pixel (H, W, 3)
    or for the given (N, 2) pixel coordinates (N, 3)."""
    if pixels is None:
        H, W = shape
        v, u = np.mgrid[0:H, 0:W].astype(np.float64)
        pixels = np.stack([u, v], -1)
    xy = undistort_normalized(pixels, cal.K_cam, cal.dist_cam)
    return np.concatenate([xy, np.ones(xy.shape[:-1] + (1,))], -1)


def ray_plane(d: np.ndarray, u: np.ndarray, cal) -> np.ndarray:
    """Camera rays d (N, 3) through origin, decoded columns u (N,) -> X (N, 3)."""
    Kp = cal.K_proj
    n_p = np.stack([np.full_like(u, Kp[0, 0]), np.full_like(u, Kp[0, 1]), Kp[0, 2] - u], -1)
    n_c = n_p @ cal.R                           # R^T n_p, row-wise
    num = -(n_p @ np.ravel(cal.t))
    den = np.einsum("ij,ij->i", n_c, d)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = num / den
    s = np.where(np.abs(den) > 1e-12, s, np.nan)
    return d * s[:, None]


def ray_ray(d: np.ndarray, u: np.ndarray, v: np.ndarray, cal) -> tuple:
    """Camera rays d (N, 3) and projector (u, v) -> (midpoints (N, 3), miss distance (N,))."""
    uv = np.stack([u, v], -1)
    if np.any(np.ravel(cal.dist_proj)):
        xy = undistort_normalized(uv, cal.K_proj, cal.dist_proj)
    else:
        Kp = cal.K_proj
        y = (v - Kp[1, 2]) / Kp[1, 1]
        xy = np.stack([(u - Kp[0, 2] - Kp[0, 1] * y) / Kp[0, 0], y], -1)
    dp = np.concatenate([xy, np.ones((len(u), 1))], -1) @ cal.R      # R^T dp (camera frame)
    C = cal.projector_center
    # closest points of X = s d and Y = C + r dp
    a = np.einsum("ij,ij->i", d, d)
    b = np.einsum("ij,ij->i", d, dp)
    c = np.einsum("ij,ij->i", dp, dp)
    e = d @ C
    f = dp @ C
    den = a * c - b * b
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (c * e - b * f) / den
        r = (b * e - a * f) / den
    P = d * s[:, None]
    Q = C + dp * r[:, None]
    return 0.5 * (P + Q), np.linalg.norm(P - Q, axis=1)


def triangulate(fld, cal, method: str = "auto", max_consistency_mm: Optional[float] = None,
                z_range_mm: tuple = (50.0, 5000.0)) -> MetricCloud:
    """DeformationField (corr_x [+ corr_y], native DMD px) + StereoCalibration -> MetricCloud."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if fld.corr_x is None:
        raise ValueError("triangulation needs the column axis (corr_x): the baseline is "
                         "horizontal, so columns carry the depth")
    both = fld.corr_y is not None
    if method == "rays" and not both:
        raise ValueError("ray-ray triangulation needs both axes (decode with axis 'xy')")
    use = "rays" if (method == "rays" or (method == "auto" and both)) else "plane"
    H, W = fld.shape
    m = fld.mask & np.isfinite(fld.corr_x)
    if both:
        m &= np.isfinite(fld.corr_y)
    vv, uu = np.nonzero(m)
    d = camera_rays(cal, (H, W), np.stack([uu, vv], -1).astype(np.float64))
    u = fld.corr_x[m].astype(np.float64)
    cons = None
    if use == "plane":
        X = ray_plane(d, u, cal)
        if both:        # still report how far the row plane disagrees
            _, miss = ray_ray(d, u, fld.corr_y[m].astype(np.float64), cal)
            cons = miss
    else:
        X, cons = ray_ray(d, u, fld.corr_y[m].astype(np.float64), cal)
    ok = np.all(np.isfinite(X), axis=1) & (X[:, 2] > z_range_mm[0]) & (X[:, 2] < z_range_mm[1])
    if cons is not None and max_consistency_mm is not None:
        ok &= cons <= max_consistency_mm
    xyz = np.full((H, W, 3), np.nan)
    mask = np.zeros((H, W), bool)
    xyz[vv[ok], uu[ok]] = X[ok]
    mask[vv[ok], uu[ok]] = True
    cmap = None
    if cons is not None:
        cmap = np.full((H, W), np.nan)
        cmap[vv[ok], uu[ok]] = cons[ok]
    return MetricCloud(xyz=xyz, mask=mask, consistency=cmap, method=use,
                       geometry_id=fld.geometry_id, calibration_id=cal.id,
                       meta={"n_points": int(ok.sum()), "axes": fld.axes,
                             "max_consistency_mm": max_consistency_mm})


def write_ply(path: Path, cloud: MetricCloud, stride: int = 1) -> Path:
    """Binary little-endian PLY of the valid points (mm), exclusive create."""
    sub = cloud.mask[::stride, ::stride]
    pts = cloud.xyz[::stride, ::stride][sub].astype(np.float32)
    header = ("ply\nformat binary_little_endian 1.0\n"
              f"comment units mm, geometry {cloud.geometry_id}, calibration {cloud.calibration_id}\n"
              f"element vertex {len(pts)}\nproperty float x\nproperty float y\nproperty float z\n"
              "end_header\n").encode()
    with open(path, "xb") as f:
        f.write(header)
        f.write(pts.tobytes())
    return Path(path)

"""
Mock stereo bench: an explicit pinhole camera + pinhole DMD projector with
known K / distortion / R / t, ray-cast against simple metric shapes (mm).

The orthographic MockBench model (camera px -> base plane, relief =
relief_px_per_mm * h) is kept for everything that exists; this module is the
ground truth the metric calibration layer (engine/calibrate/, reconstruct/
triangulate.py) is validated against. MockBench switches to it when a
stereo shape (Plane, Sphere, MetricCylinder, Block, CharucoTarget, EndFace) is
placed, so the normal capture path (projector prepare, exposure, averaging,
camera noise) runs unchanged.

Frames (all mm):
  world = camera frame: x right, y down, z forward (OpenCV convention)
  projector frame: X_p = R @ X + t

Camera: K, Brown distortion (k1, k2, p1, p2, k3), 1440 x 1080 (Blackfly S).

Projector (DLP4500, 912 x 1140 diamond DMD), native (col, row) coordinates:
  u = f_u * X_p / Z_p + c_u,  v = f_v * Y_p / Z_p + c_v,  f_v ~= 2 f_u
Rows are half a column pitch apart vertically (hence f_v = 2 f_u), and
alternate rows are offset by half a column: the mirror at (row r, col c)
sits at native column c + 0.5 on rows with r % 2 == parity (the rig's
unknown, measured with calibrate/parity.py). A pattern that varies with
the column index only is therefore a zigzag, not a straight fringe; the
`multifreq` basis pre-compensates it once the parity is known. The
projector has no lens distortion in this model.

Default geometry = the bench: projector 279.4 mm right of the camera,
aimed at the wall 823 mm away (triangulation angle ~18.8 deg), throw ratio
~1.4 horizontally (912 / 1200 x 823 ~= 625 mm wide at the wall).

Rendering, per camera pixel: the camera ray hits the nearest surface at X;
X is lit when the projector sees it first (shadow test by a second ray
cast from the projector), and the pixel reads
    ambient + gain * albedo * cos(incidence) * pattern(u, v) / 255
with the pattern sampled bilinearly between rows and, inside a row,
bilinearly between the (parity-shifted) mirrors. The board texture is
pre-filtered to the camera footprint so corners are anti-aliased.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ------------------------------------------------------------------ models

def undistort_normalized(px: np.ndarray, K: np.ndarray, dist, iters: int = 25) -> np.ndarray:
    """Distorted pixel coords (..., 2) -> ideal normalized coords (..., 2).
    Fixed-point inversion of the Brown model, run to convergence (cv2's
    undistortPoints stops after 5 iterations, too coarse for ground truth)."""
    px = np.asarray(px, np.float64)
    k1, k2, p1, p2, k3 = (list(np.asarray(dist, np.float64).ravel()) + [0.0] * 5)[:5]
    xd = (px[..., 0] - K[0, 2]) / K[0, 0]
    yd = (px[..., 1] - K[1, 2]) / K[1, 1]
    x, y = xd.copy(), yd.copy()
    if not any((k1, k2, p1, p2, k3)):
        return np.stack([x, y], axis=-1)
    for _ in range(iters):
        r2 = x * x + y * y
        rad = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
        dx = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        dy = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        xn = (xd - dx) / rad
        yn = (yd - dy) / rad
        step = max(float(np.max(np.abs(xn - x), initial=0)), float(np.max(np.abs(yn - y), initial=0)))
        x, y = xn, yn
        if step < 1e-13:                # converged (normalized units: ~1e-10 px)
            break
    return np.stack([x, y], axis=-1)


def distort_project(Xc: np.ndarray, K: np.ndarray, dist) -> np.ndarray:
    """Points in a camera frame (..., 3) -> distorted pixels (..., 2)."""
    Xc = np.asarray(Xc, np.float64)
    k1, k2, p1, p2, k3 = (list(np.asarray(dist, np.float64).ravel()) + [0.0] * 5)[:5]
    x = Xc[..., 0] / Xc[..., 2]
    y = Xc[..., 1] / Xc[..., 2]
    r2 = x * x + y * y
    rad = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
    xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return np.stack([K[0, 0] * xd + K[0, 1] * yd + K[0, 2], K[1, 1] * yd + K[1, 2]], axis=-1)


@dataclass
class PinholeCamera:
    K: np.ndarray
    dist: np.ndarray
    size: tuple = (1440, 1080)          # (W, H)

    def rays(self) -> np.ndarray:
        """(H, W, 3) ray directions (z = 1) through every pixel centre (cached)."""
        key = (np.asarray(self.K).tobytes(), np.asarray(self.dist).tobytes(), tuple(self.size))
        if getattr(self, "_rays_key", None) != key:
            W, H = self.size
            v, u = np.mgrid[0:H, 0:W].astype(np.float64)
            xy = undistort_normalized(np.stack([u, v], -1), self.K, self.dist)
            self._rays = np.concatenate([xy, np.ones((H, W, 1))], axis=-1)
            self._rays_key = key
        return self._rays

    def project(self, X: np.ndarray) -> np.ndarray:
        return distort_project(X, self.K, self.dist)


@dataclass
class DiamondProjector:
    """Pinhole projector in native DMD (col, row) coordinates."""
    K: np.ndarray
    size: tuple = (912, 1140)           # (W cols, H rows)
    parity: Optional[int] = 1           # rows with r % 2 == parity sit +0.5 col; None = square grid

    def project(self, Xp: np.ndarray) -> np.ndarray:
        Xp = np.asarray(Xp, np.float64)
        z = Xp[..., 2]
        return np.stack([self.K[0, 0] * Xp[..., 0] / z + self.K[0, 1] * Xp[..., 1] / z
                         + self.K[0, 2], self.K[1, 1] * Xp[..., 1] / z + self.K[1, 2]], -1)

    def row_shift(self, r: np.ndarray) -> np.ndarray:
        if self.parity is None:
            return np.zeros(np.shape(r), np.float64)
        return np.where((r & 1) == self.parity, 0.5, 0.0)

    def taps(self, u: np.ndarray, v: np.ndarray) -> tuple:
        """(flat mirror indices (4, N), weights (4, N)) of the light leaving
        towards native (u, v): bilinear between rows, and inside each row
        between its (parity-shifted) mirrors. Precomputed once per scene."""
        W, H = self.size
        r0 = np.clip(np.floor(v).astype(np.intp), 0, H - 1)
        r1 = np.minimum(r0 + 1, H - 1)
        fy = np.clip(v - r0, 0, 1)
        idx, wts = [], []
        for r, wr in ((r0, 1 - fy), (r1, fy)):
            c = np.clip(u - self.row_shift(r), 0, W - 1)
            c0 = np.floor(c).astype(np.intp)
            c1 = np.minimum(c0 + 1, W - 1)
            fx = c - c0
            idx += [r * W + c0, r * W + c1]
            wts += [wr * (1 - fx), wr * fx]
        return np.stack(idx), np.stack(wts).astype(np.float32)

    def sample(self, pattern: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Drive level 0..255 of the light leaving towards native (u, v)."""
        idx, w = self.taps(u, v)
        return apply_taps(pattern, idx, w)


def apply_taps(pattern: np.ndarray, idx: np.ndarray, w: np.ndarray) -> np.ndarray:
    p = np.asarray(pattern, np.float32).reshape(-1)
    return (w[0] * p[idx[0]] + w[1] * p[idx[1]] + w[2] * p[idx[2]] + w[3] * p[idx[3]])


def look_at_rotation(position, target) -> np.ndarray:
    """World -> device rotation for a device at `position` whose optical axis
    points at `target`, rows stay level (device y = world y projected)."""
    z = np.asarray(target, float) - np.asarray(position, float)
    z /= np.linalg.norm(z)
    x = np.cross([0.0, 1.0, 0.0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z])


@dataclass
class StereoRig:
    camera: PinholeCamera
    projector: DiamondProjector
    R: np.ndarray                       # X_p = R X + t
    t: np.ndarray

    @classmethod
    def bench(cls, baseline_mm: float = 279.4, standoff_mm: float = 823.0,
              parity: Optional[int] = 1) -> "StereoRig":
        Kc = np.array([[2450.0, 0.0, 722.0], [0.0, 2450.0, 537.5], [0, 0, 1]])
        dist = np.array([-0.09, 0.12, 0.0004, -0.0003, 0.0])
        Kp = np.array([[1200.0, 0.0, 458.0], [0.0, 2400.0, 580.0], [0, 0, 1]])
        C = np.array([baseline_mm, 0.0, 0.0])
        R = look_at_rotation(C, [0.0, 0.0, standoff_mm])
        return cls(PinholeCamera(Kc, dist), DiamondProjector(Kp, parity=parity), R, -R @ C)

    @property
    def projector_center(self) -> np.ndarray:
        return -self.R.T @ self.t

    @property
    def triangulation_angle_deg(self) -> float:
        C = self.projector_center
        return math.degrees(math.atan2(abs(C[0]), 823.0))

    def truth(self) -> dict:
        return {"K_cam": self.camera.K.copy(), "dist_cam": self.camera.dist.copy(),
                "K_proj": self.projector.K.copy(), "dist_proj": np.zeros(5),
                "R": self.R.copy(), "t": self.t.copy(), "parity": self.projector.parity,
                "cam_size": tuple(self.camera.size), "proj_size": tuple(self.projector.size)}


# ------------------------------------------------------------------ shapes

class StereoShape:
    albedo: float = 0.7

    def intersect(self, o: np.ndarray, d: np.ndarray):
        """Rays (o: (3,) or (N, 3), d: (N, 3)) -> (t (N,), normal (N, 3)); t = inf on a miss."""
        raise NotImplementedError

    def albedo_at(self, X: np.ndarray) -> np.ndarray:
        return np.full(X.shape[0], self.albedo, np.float32)


@dataclass
class Plane(StereoShape):
    point: tuple = (0.0, 0.0, 823.0)
    normal: tuple = (0.0, 0.0, -1.0)
    albedo: float = 0.6

    def intersect(self, o, d):
        n = np.asarray(self.normal, float)
        n = n / np.linalg.norm(n)
        den = d @ n
        with np.errstate(divide="ignore", invalid="ignore"):
            t = ((np.asarray(self.point, float) - o) @ n) / den
        t = np.where((np.abs(den) > 1e-12) & (t > 1e-6), t, np.inf)
        return t, np.broadcast_to(n, d.shape)

    def signed_distance(self, X):
        n = np.asarray(self.normal, float)
        n = n / np.linalg.norm(n)
        return (X - np.asarray(self.point, float)) @ n


@dataclass
class Sphere(StereoShape):
    center: tuple = (0.0, 0.0, 780.0)
    radius: float = 20.0
    albedo: float = 0.7

    def intersect(self, o, d):
        c = np.asarray(self.center, float)
        w = o - c
        a = np.einsum("ij,ij->i", d, d)
        b = 2 * (np.einsum("ij,ij->i", w, d) if w.ndim == 2 else d @ w)
        cc = (np.einsum("ij,ij->i", w, w) if w.ndim == 2 else w @ w) - self.radius ** 2
        disc = b * b - 4 * a * cc
        sq = np.sqrt(np.maximum(disc, 0))
        t = (-b - sq) / (2 * a)
        t = np.where((disc >= 0) & (t > 1e-6), t, np.inf)
        X = o + t[:, None] * d
        nrm = (X - c) / self.radius
        return t, np.nan_to_num(nrm)


@dataclass
class MetricCylinder(StereoShape):
    """A finite cylinder with optional cosine-bowl radial dents.

    center: a point on the axis at mid-height; axis: direction (default
    camera y = a standing cell). dents: [{depth_mm, size_mm, angle_deg,
    z_mm}] with angle_deg about the axis from the camera-facing line
    (+ toward camera +x) and z_mm along the axis from mid-height
    (+ along `axis`). A dent pushes the surface inward by
    depth * 0.5 * (1 + cos(pi q)), q = (distance on the surface) / (size/2)."""
    center: tuple = (0.0, 0.0, 799.0)
    radius: float = 24.0
    height: float = 90.0
    axis: tuple = (0.0, 1.0, 0.0)
    dents: list = field(default_factory=list)
    albedo: float = 0.7

    def _frame(self):
        a = np.asarray(self.axis, float)
        a = a / np.linalg.norm(a)
        c = np.asarray(self.center, float)
        e0 = -c - ((-c) @ a) * a                  # toward the camera (world origin)
        if np.linalg.norm(e0) < 1e-9:
            e0 = np.array([0.0, 0.0, -1.0]) - a[2] * a
        e0 /= np.linalg.norm(e0)
        e1 = np.cross(a, e0)
        return c, a, e0, e1

    def dent_depth(self, theta: np.ndarray, z: np.ndarray) -> np.ndarray:
        dr = np.zeros(np.shape(theta), np.float64)
        for dn in self.dents:
            rad = float(dn["size_mm"]) / 2
            dth = np.angle(np.exp(1j * (theta - math.radians(float(dn.get("angle_deg", 0.0))))))
            q = np.hypot(self.radius * dth, z - float(dn.get("z_mm", 0.0))) / rad
            dr = np.maximum(dr, np.where(q < 1, float(dn["depth_mm"]) * 0.5 * (1 + np.cos(np.pi * q)), 0))
        return dr

    def local(self, X):
        """-> (rho, theta, z) of points about the axis."""
        c, a, e0, e1 = self._frame()
        w = X - c
        z = w @ a
        p = w - z[:, None] * a
        return np.linalg.norm(p, axis=1), np.arctan2(p @ e1, p @ e0), z

    def intersect(self, o, d):
        c, a, e0, e1 = self._frame()
        o2 = np.broadcast_to(o, d.shape)
        w = o2 - c
        dp = d - (d @ a)[:, None] * a
        wp = w - (np.einsum("ij,j->i", w, a))[:, None] * a
        A = np.einsum("ij,ij->i", dp, dp)
        B = 2 * np.einsum("ij,ij->i", wp, dp)
        Cq = np.einsum("ij,ij->i", wp, wp) - self.radius ** 2
        disc = B * B - 4 * A * Cq
        ok = (disc >= 0) & (A > 1e-12)
        sq = np.sqrt(np.maximum(disc, 0))
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(ok, (-B - sq) / (2 * A), np.inf)
        if self.dents:
            # Newton on f(t) = rho(t) - (R - dent(theta, z)) from the perfect hit
            live = np.isfinite(t)
            tt = t[live]
            ol, dl = o2[live], d[live]
            for _ in range(8):
                def f(tv):
                    rho, th, z = self.local(ol + tv[:, None] * dl)
                    return rho - (self.radius - self.dent_depth(th, z))
                f0 = f(tt)
                h = 1e-4
                df = (f(tt + h) - f0) / h
                step = np.where(np.abs(df) > 1e-9, f0 / np.where(np.abs(df) > 1e-9, df, 1), 0)
                tt = tt - np.clip(step, -5, 5)
            t = t.copy()
            t[live] = tt
        X = o2 + np.where(np.isfinite(t), t, 0)[:, None] * d
        zz = (X - c) @ a
        t = np.where(np.abs(zz) <= self.height / 2, t, np.inf)
        nrm = (X - c) - zz[:, None] * a
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
        return t, nrm

    def radial_truth(self, X):
        """True radial deviation (mm, negative = dent) at surface points X."""
        rho, th, z = self.local(X)
        return -self.dent_depth(th, z)


@dataclass
class Block(StereoShape):
    """Oriented box: center, half-extents along its axes (rows of `rot`)."""
    center: tuple = (0.0, 0.0, 818.0)
    half: tuple = (30.0, 20.0, 5.0)
    rot: Optional[np.ndarray] = None    # (3, 3) rows = box axes in world; None = aligned
    albedo: float = 0.7

    def intersect(self, o, d):
        Rb = np.eye(3) if self.rot is None else np.asarray(self.rot, float)
        o2 = np.broadcast_to(o, d.shape)
        lo = (o2 - np.asarray(self.center, float)) @ Rb.T
        ld = d @ Rb.T
        h = np.asarray(self.half, float)
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (-h - lo) / ld
            t2 = (h - lo) / ld
        tmin = np.nanmax(np.minimum(t1, t2), axis=1)
        tmax = np.nanmin(np.maximum(t1, t2), axis=1)
        hit = (tmax >= tmin) & (tmin > 1e-6)
        t = np.where(hit, tmin, np.inf)
        face = np.argmax(np.minimum(t1, t2), axis=1)
        sgn = -np.sign(ld[np.arange(len(ld)), face])
        nl = np.zeros_like(ld)
        nl[np.arange(len(ld)), face] = sgn
        return t, nl @ Rb


@dataclass
class CharucoTarget(StereoShape):
    """A printed ChArUco board (calibrate/charuco.BoardSpec) at a pose:
    X_world = R @ X_board + t (board mm, origin top-left, z = 0)."""
    spec: object = None
    R: np.ndarray = field(default_factory=lambda: np.eye(3))
    t: np.ndarray = field(default_factory=lambda: np.array([-125.0, -87.5, 800.0]))
    px_per_mm: float = 8.0
    dark: float = 0.08
    light: float = 0.85
    blur_px: float = 1.3                # texture px: ~ the camera footprint
    albedo: float = 0.85
    _tex: Optional[np.ndarray] = field(default=None, repr=False)

    def intersect(self, o, d):
        R, t = np.asarray(self.R, float), np.asarray(self.t, float)
        n = R[:, 2]
        o2 = np.broadcast_to(o, d.shape)
        den = d @ n
        with np.errstate(divide="ignore", invalid="ignore"):
            tt = ((t - o2) @ n) / den
        X = o2 + np.where(np.isfinite(tt), tt, 0)[:, None] * d
        b = (X - t) @ R
        w, h = self.spec.board_size_mm
        inside = (b[:, 0] >= 0) & (b[:, 0] <= w) & (b[:, 1] >= 0) & (b[:, 1] <= h)
        tt = np.where(inside & (np.abs(den) > 1e-12) & (tt > 1e-6), tt, np.inf)
        return tt, np.broadcast_to(-n if n[2] > 0 else n, d.shape)

    def texture(self) -> np.ndarray:
        if self._tex is None:
            import cv2
            img = self.spec.image(self.px_per_mm).astype(np.float32) / 255.0
            img = self.dark + (self.light - self.dark) * img
            self._tex = cv2.GaussianBlur(img, (0, 0), self.blur_px)
        return self._tex

    def albedo_at(self, X):
        from scipy.ndimage import map_coordinates
        b = (X - np.asarray(self.t, float)) @ np.asarray(self.R, float)
        mx = b[:, 0] * self.px_per_mm - 0.5
        my = b[:, 1] * self.px_per_mm - 0.5
        out = map_coordinates(self.texture(), [my, mx], order=1, mode="nearest")
        return out.astype(np.float32)

    def corners_world(self) -> np.ndarray:
        obj = self.spec.object_points()
        return obj @ np.asarray(self.R, float).T + np.asarray(self.t, float)


@dataclass
class EndFace(StereoShape):
    """The flat circular END FACE of a cylindrical cell lying with its axis toward the
    camera: a disc of `radius` at `center` with unit `normal` pointing toward the camera.

    In-face coordinates (the engine's face frame, engine/measure/endface):
        e1 = n x y_cam (~ camera +x), e2 = e1 x n (~ camera +y), (u, v) mm from `center`.
    dents: [{depth_mm, size_mm, u_mm, v_mm}] cosine bowls pushed INTO the face,
        depth * 0.5 * (1 + cos(pi q)), q = distance / (size / 2).
    chamfer_mm / chamfer_depth_mm: the outer chamfer_mm of the radius recede linearly to
        chamfer_depth_mm at the edge (a grazing rim, like the bench cells).
    Place a MetricCylinder behind it (center - n * L / 2, axis n, height L) for the body."""
    center: tuple = (0.0, 0.0, 780.0)
    normal: tuple = (0.0, 0.0, -1.0)
    radius: float = 10.525
    dents: list = field(default_factory=list)
    chamfer_mm: float = 0.0
    chamfer_depth_mm: float = 0.0
    albedo: float = 0.7

    def frame(self):
        n = np.asarray(self.normal, float)
        n = n / np.linalg.norm(n)
        e1 = np.cross(n, [0.0, 1.0, 0.0])
        e1 /= np.linalg.norm(e1)
        return np.asarray(self.center, float), n, e1, np.cross(e1, n)

    def recess(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Depth (mm, >= 0) the surface sits behind the nominal face plane at (u, v)."""
        out = np.zeros(np.shape(u), np.float64)
        for dn in self.dents:
            rad = float(dn["size_mm"]) / 2
            q = np.hypot(u - float(dn.get("u_mm", 0.0)), v - float(dn.get("v_mm", 0.0))) / rad
            out = np.maximum(out, np.where(q < 1, float(dn["depth_mm"]) * 0.5 * (1 + np.cos(np.pi * q)),
                                           0.0))
        if self.chamfer_mm > 0 and self.chamfer_depth_mm:
            r = np.hypot(u, v)
            w = float(self.chamfer_mm)
            out = out + np.clip((r - (self.radius - w)) / w, 0, 1) * float(self.chamfer_depth_mm)
        return out

    def truth_height(self, u, v) -> np.ndarray:
        """Nominal-plane height of the surface (mm, + toward the camera: a dent is negative)."""
        return -self.recess(np.asarray(u, float), np.asarray(v, float))

    def intersect(self, o, d):
        c, n, e1, e2 = self.frame()
        o2 = np.broadcast_to(o, d.shape)
        den = d @ n
        with np.errstate(divide="ignore", invalid="ignore"):
            t = ((c - o2) @ n) / den
            shaped = bool(self.dents) or (self.chamfer_mm > 0 and self.chamfer_depth_mm)
            for _ in range(8 if shaped else 0):     # fixed point: plane moved back by recess
                w = o2 + np.nan_to_num(t)[:, None] * d - c
                delta = self.recess(w @ e1, w @ e2)
                t = ((c - delta[:, None] * n - o2) @ n) / den
        w = o2 + np.where(np.isfinite(t), t, 0)[:, None] * d - c
        r = np.hypot(w @ e1, w @ e2)
        hit = (np.abs(den) > 1e-12) & np.isfinite(t) & (t > 1e-6) & (r <= self.radius)
        return np.where(hit, t, np.inf), np.broadcast_to(n, d.shape)


STEREO_SHAPES = (Plane, Sphere, MetricCylinder, Block, CharucoTarget, EndFace)


# ----------------------------------------------------------------- the scene

@dataclass
class StereoScene:
    rig: StereoRig
    shapes: list = field(default_factory=list)
    wall: Optional[Plane] = field(default_factory=Plane)
    gain: float = 380.0
    ambient: float = 12.0
    _geo: Optional[dict] = field(default=None, repr=False)
    _last: Optional[tuple] = field(default=None, repr=False)
    _rays: Optional[np.ndarray] = field(default=None, repr=False)

    def invalidate(self):
        self._geo = None
        self._last = None

    def all_shapes(self) -> list:
        return list(self.shapes) + ([self.wall] if self.wall is not None else [])

    def trace(self, o, d):
        """Nearest hit over all shapes -> (t, normal, index into all_shapes; -1 = miss)."""
        t_best = np.full(d.shape[0], np.inf)
        n_best = np.zeros_like(d)
        idx = np.full(d.shape[0], -1, np.int32)
        for i, s in enumerate(self.all_shapes()):
            t, n = s.intersect(o, d)
            better = t < t_best
            t_best = np.where(better, t, t_best)
            n_best[better] = n[better]
            idx[better] = i
        return t_best, n_best, idx

    def geometry(self) -> dict:
        """Per camera pixel: world point, albedo, shading, projector (u, v), valid."""
        if self._geo is not None:
            return self._geo
        cam, prj = self.rig.camera, self.rig.projector
        W, H = cam.size
        if self._rays is None:
            self._rays = cam.rays().reshape(-1, 3)
        d = self._rays
        t, nrm, idx = self.trace(np.zeros(3), d)
        hit = np.isfinite(t)
        X = np.where(hit[:, None], d * np.where(hit, t, 0)[:, None], np.nan)
        # projector: sees X first? (shadow) and where on the DMD
        Cp = self.rig.projector_center
        toX = np.where(hit[:, None], X - Cp, np.array([0.0, 0.0, 1.0]))
        L = np.linalg.norm(toX, axis=1)
        dirp = toX / L[:, None]
        tp, _, _ = self.trace(Cp, dirp)
        lit = hit & (tp >= L - 0.02)
        Xp = np.where(hit[:, None], X, 0) @ self.rig.R.T + self.rig.t
        uv = prj.project(np.where(hit[:, None], Xp, 1.0))
        u, v = uv[:, 0], uv[:, 1]
        Wp, Hp = prj.size
        lit &= (Xp[:, 2] > 0) & (u >= 0) & (u <= Wp - 1) & (v >= 0) & (v <= Hp - 1)
        cosi = np.abs(np.einsum("ij,ij->i", nrm, -dirp))
        alb = np.zeros(d.shape[0], np.float32)
        shapes = self.all_shapes()
        for i, s in enumerate(shapes):
            m = idx == i
            if m.any():
                alb[m] = s.albedo_at(X[m])
        self._geo = {"X": X.reshape(H, W, 3), "albedo": alb.reshape(H, W),
                     "shade": np.where(lit, cosi, 0).astype(np.float32).reshape(H, W),
                     "u": u.reshape(H, W), "v": v.reshape(H, W),
                     "lit": lit.reshape(H, W), "hit": hit.reshape(H, W),
                     "shape": idx.reshape(H, W)}
        return self._geo

    def render(self, pattern: Optional[np.ndarray], led_weights=None) -> np.ndarray:
        """Camera frame (float, pre-noise). Identical consecutive patterns
        reuse the last frame (the camera averages several grabs of one)."""
        g = self.geometry()
        W, H = self.rig.camera.size
        frame = np.full((H, W), self.ambient, np.float32)
        if pattern is None:
            return frame
        pat = np.asarray(pattern)
        if pat.ndim == 3:
            w = np.ones(3, np.float32) / 3 if led_weights is None else led_weights
            pat = pat[..., 2::-1].astype(np.float32) @ w
        key = hashlib.blake2b(np.ascontiguousarray(pat).tobytes(), digest_size=16).digest()
        if self._last is not None and self._last[0] == key:
            return self._last[1].copy()
        if "taps" not in g:
            lit = g["lit"]
            g["lit_idx"] = np.flatnonzero(lit)
            g["taps"] = self.rig.projector.taps(g["u"][lit], g["v"][lit])
            g["scale"] = (self.gain / 255.0 * g["albedo"] * g["shade"]).reshape(-1)[g["lit_idx"]]
        flat = frame.reshape(-1)
        flat[g["lit_idx"]] += g["scale"] * apply_taps(pat, *g["taps"])
        self._last = (key, frame)
        return frame.copy()

    def oracle(self) -> tuple:
        """Exact (corr_x, corr_y, mask): native projector (col, row) per camera px."""
        g = self.geometry()
        m = g["lit"] & (g["shade"] > 0.05)
        return (np.where(m, g["u"], np.nan).astype(np.float64),
                np.where(m, g["v"], np.nan).astype(np.float64), m)


# ---------------------------------------------------------- board poses

def board_poses(spec, n: int = 12, seed: int = 0, standoff: float = 823.0) -> list:
    """n ChArUco poses spread over the camera view: centre 40..180 mm in front
    of the wall (a tilted board must stay in front of it), tilts up to
    +-28 deg, the board centre moved around the field (what the operator
    does by hand on the bench)."""
    rng = np.random.default_rng(seed)
    w, h = spec.board_size_mm
    out = []
    grid = [(-1, -1), (1, -1), (-1, 1), (1, 1), (0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
    for i in range(n):
        gx, gy = grid[i % len(grid)]
        z = standoff - rng.uniform(40, 180)
        cx = gx * 0.13 * z + rng.uniform(-15, 15)
        cy = gy * 0.09 * z + rng.uniform(-10, 10)
        ax = math.radians(rng.uniform(-28, 28))
        ay = math.radians(rng.uniform(-28, 28))
        az = math.radians(rng.uniform(-12, 12))
        Rx = np.array([[1, 0, 0], [0, math.cos(ax), -math.sin(ax)], [0, math.sin(ax), math.cos(ax)]])
        Ry = np.array([[math.cos(ay), 0, math.sin(ay)], [0, 1, 0], [-math.sin(ay), 0, math.cos(ay)]])
        Rz = np.array([[math.cos(az), -math.sin(az), 0], [math.sin(az), math.cos(az), 0], [0, 0, 1]])
        R = Rz @ Ry @ Rx
        t = np.array([cx, cy, z]) - R @ np.array([w / 2, h / 2, 0.0])
        out.append((R, t))
    return out

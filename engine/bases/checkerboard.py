"""
Checkerboard basis — corner / feature detection, not dense correspondence
(ENGINE_ARCHITECTURE §3; UV_MAPPING_FOR_STRUCTURED_LIGHT §3).

Frames (generate):

  white, black                       polarity + normalisation (measured, never assumed)
  per shift (dx, dy):
    checker, inverse                 square_px-mirror checker and its inverse
    column-index bits, row-index bits gray code of the SQUARE index (not the
                                     mirror index): cheap, coarse, only used to
                                     name each corner

Decode (detect):

  1. d = (checker - inverse) / (white - black): +-1 inside squares, a clean
     saddle at every checker corner; albedo, ambient and polarity cancel.
  2. saddle response on a Gaussian-smoothed d:  gxy^2 - gxx*gyy  (> 0 only
     at X-junctions, ~0 on straight edges and inside squares), local maxima.
     Partial boards are fine: nothing assumes a full rectangle of corners
     (cv2.findChessboardCorners does).
  3. sub-pixel: least-squares quadratic fit of the smoothed d around the
     peak, saddle point = zero of its gradient, iterated (bilinear resample).
  4. naming: the square-index code is sampled on a ring around the corner;
     exactly two column codes (k-1, k) and two row codes (m-1, m) must show,
     and the corner is DMD (k*s - dx - 0.5, m*s - dy - 0.5).

DMD coordinates use the mirror-centre convention of the correspondence
maps (corr_x/corr_y): mirror k spans [k - 0.5, k + 0.5], so the edge
between mirrors k-1 and k is at k - 0.5.

local_jacobian(corners): per checker cell (4 named corners) the 2x2 map
d(camera px)/d(DMD mirror): magnification along DMD x (the baseline) and y,
skew, anisotropy, area; `stretch` = baseline magnification vs a reference
(a flat-base checker scan) or vs the median -> a slope proxy (§3: stretch
along the baseline ~ surface tilt). `contrast` per cell = checker
modulation |d| at the square centre -> focus per region. CheckerCorners
.pairs() exports (u, v, X, Y) for future stereo calibration
(calibrate/stereo.py).

Nothing here touches hardware: capture_checker() goes through
session.rig() / sync.shoot(raw=True) like every other capture.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
from scipy import ndimage

from ..decode.deformation import DeformationField
from .base import PatternBasis
from .registry import register

LIT_FRACTION = 0.10       # lit: |white - black| > this x the 99th percentile
MIN_LIT_COUNTS = 8.0      # ... and more than this many camera counts
RESP_FRACTION = 0.10      # saddle threshold: fraction of the ideal peak for the measured contrast
RING_RADII = (0.22, 0.32)  # code-sampling ring around a corner, x the camera square size
RING_ANGLES = 32
RING_MIN_SHARE = 0.12     # each of the two column / row codes must hold this share of the ring


def _gray(n):
    return n ^ (n >> 1)


def _gray_inverse(g: np.ndarray) -> np.ndarray:
    b = g.copy()
    s = g >> 1
    while np.any(s):
        b ^= s
        s >>= 1
    return b


def checker_image(shape, square_px: int, dx: int = 0, dy: int = 0) -> np.ndarray:
    """uint8 0/255 checker; square (r, c) = ((y+dy)//s, (x+dx)//s), (0, 0) on."""
    H, W = shape
    r = (np.arange(H) + dy) // square_px
    c = (np.arange(W) + dx) // square_px
    return np.where(((r[:, None] + c[None, :]) & 1) == 0, 255, 0).astype(np.uint8)


def _n_bits(extent: int, square_px: int, off: int) -> int:
    n = (extent - 1 + off) // square_px + 1          # square indices 0..n-1
    return max(1, int(math.ceil(math.log2(n + 1))))


@dataclass
class CheckerCorners:
    """Detected checker corners, camera px matched to DMD mirrors."""
    cam: np.ndarray             # (N, 2) camera (u, v), sub-pixel
    dmd: np.ndarray             # (N, 2) DMD (X, Y), mirror-centre convention
    index: np.ndarray           # (N, 2) int square-corner index (k, m)
    shift: np.ndarray           # (N, 2) int DMD shift (dx, dy) of the board it came from
    response: np.ndarray        # (N,) saddle strength (normalised units)
    square_px: int
    cam_square_px: float        # mean camera px per square side (sqrt of area)
    shape: tuple                # camera (H, W)
    contrast: Optional[np.ndarray] = field(default=None, repr=False)  # |d|, camera image
    lit: Optional[np.ndarray] = field(default=None, repr=False)
    warnings: list = field(default_factory=list)

    def __len__(self):
        return len(self.cam)

    def pairs(self) -> np.ndarray:
        """(N, 4) float64 [u, v, X, Y]: camera px <-> DMD mirror pairs
        (for stereo / projector-as-inverse-camera calibration)."""
        return np.hstack([self.cam, self.dmd]).astype(np.float64)


@register
class CheckerboardBasis(PatternBasis):
    name = "checkerboard"
    axes = {"x", "y"}

    def __init__(self, square_px: int = 32, shifts: int = 1, **params):
        super().__init__(square_px=square_px, shifts=shifts, **params)
        self.square_px = int(square_px)
        self.shifts = int(shifts)
        if self.square_px < 4:
            raise ValueError("checkerboard square_px must be >= 4 mirrors "
                             "(1-mirror squares keep ~1% contrast on the bench)")
        if self.shifts < 1:
            raise ValueError("checkerboard shifts must be >= 1")
        self.dmd_shape = tuple(params.get("dmd_shape", (1140, 912)))

    # ------------------------------------------------------------ patterns
    def offsets(self) -> list:
        s, n = self.square_px, self.shifts
        return [(int(round(t * s / n)) % s,) * 2 for t in range(n)]

    def _bits(self, shape) -> tuple:
        H, W = shape
        s = self.square_px
        return _n_bits(W, s, s), _n_bits(H, s, s)

    def n_frames(self) -> int:
        bx, by = self._bits(self.dmd_shape)
        return 2 + self.shifts * (2 + bx + by)

    def generate(self, dmd_shape: tuple = None) -> list:
        shape = tuple(dmd_shape or self.dmd_shape)
        self.dmd_shape = shape
        H, W = shape
        s = self.square_px
        bx, by = self._bits(shape)
        out = [np.full(shape, 255, np.uint8), np.zeros(shape, np.uint8)]
        for dx, dy in self.offsets():
            c = checker_image(shape, s, dx, dy)
            out += [c, 255 - c]
            col = _gray((np.arange(W) + dx) // s)
            row = _gray((np.arange(H) + dy) // s)
            for b in range(bx - 1, -1, -1):
                line = np.where((col >> b) & 1, 255, 0).astype(np.uint8)
                out.append(np.broadcast_to(line[None, :], shape).copy())
            for b in range(by - 1, -1, -1):
                line = np.where((row >> b) & 1, 255, 0).astype(np.uint8)
                out.append(np.broadcast_to(line[:, None], shape).copy())
        return out

    # ------------------------------------------------------------- detect
    def detect(self, frames: list, dmd_shape: tuple = None) -> CheckerCorners:
        shape = tuple(dmd_shape or self.dmd_shape)
        bx, by = self._bits(shape)
        per = 2 + bx + by
        if len(frames) != 2 + self.shifts * per:
            raise ValueError(f"checkerboard: expected {2 + self.shifts * per} frames, "
                             f"got {len(frames)}")
        f = [np.asarray(x, np.float32) for x in frames]
        amp = f[0] - f[1]                        # signed: polarity-invariant below
        mag = np.abs(amp)
        lit = mag > max(MIN_LIT_COUNTS, LIT_FRACTION * float(np.percentile(mag, 99)))
        warnings = []
        if float(np.mean(amp[lit])) < 0 if lit.any() else False:
            warnings.append("white frame darker than black: pattern path inverted "
                            "(compensated; decode is polarity-invariant)")
        safe = np.where(lit, amp, 1.0)
        cams, dmds, idxs, shs, resps = [], [], [], [], []
        contrast = np.zeros(amp.shape, np.float32)
        cam_sq = []
        s = self.square_px
        for si, (dx, dy) in enumerate(self.offsets()):
            base = 2 + si * per
            d = np.where(lit, (f[base] - f[base + 1]) / safe, 0.0).astype(np.float32)
            np.clip(d, -1.5, 1.5, out=d)
            contrast = np.maximum(contrast, np.abs(d))
            mid = 0.5 * (f[0] + f[1])

            def code(frames_):
                g = np.zeros(amp.shape, np.int64)
                ok = lit.copy()
                for fr in frames_:
                    t = (fr - mid) / safe
                    ok &= np.abs(t) > 0.15
                    g = (g << 1) | (t > 0)
                return _gray_inverse(g), ok

            col, okc = code(f[base + 2: base + 2 + bx])
            row, okr = code(f[base + 2 + bx: base + per])
            okcode = okc & okr
            # camera square size: lit area / number of distinct squares seen
            keys = (col[okcode] * 4096 + row[okcode])
            if keys.size == 0:
                warnings.append(f"shift {(dx, dy)}: no decodable square codes")
                continue
            uk, cnt = np.unique(keys, return_counts=True)
            big = cnt >= max(4, int(0.02 * np.median(cnt)))
            sq = math.sqrt(float(cnt[big].sum()) / max(int(big.sum()), 1))
            cam_sq.append(sq)
            got = self._saddles(d, lit, sq)
            if got is None:
                continue
            pts, resp = got
            k, m, ok = self._name(pts, col, row, okcode, sq)
            X = k * s - dx - 0.5
            Y = m * s - dy - 0.5
            ok &= (X >= -0.5) & (X <= shape[1] - 0.5) & (Y >= -0.5) & (Y <= shape[0] - 0.5)
            cams.append(pts[ok])
            dmds.append(np.stack([X[ok], Y[ok]], 1))
            idxs.append(np.stack([k[ok], m[ok]], 1))
            shs.append(np.tile([dx, dy], (int(ok.sum()), 1)))
            resps.append(resp[ok])
        if cams:
            cam, dmd = np.concatenate(cams), np.concatenate(dmds)
            idx, sh, rs = np.concatenate(idxs), np.concatenate(shs), np.concatenate(resps)
            # one corner per (shift, k, m): keep the strongest
            key = [(int(a), int(b), int(c), int(e)) for a, b, c, e in np.hstack([sh, idx])]
            best = {}
            for i, kk in enumerate(key):
                if kk not in best or rs[i] > rs[best[kk]]:
                    best[kk] = i
            sel = np.array(sorted(best.values()), np.intp)
            cam, dmd, idx, sh, rs = cam[sel], dmd[sel], idx[sel], sh[sel], rs[sel]
        else:
            cam = dmd = np.zeros((0, 2))
            idx = sh = np.zeros((0, 2), np.int64)
            rs = np.zeros(0)
        if len(cam) == 0:
            warnings.append("no checker corners found (projector off / out of focus / "
                            "square_px too small for the optics?)")
        return CheckerCorners(cam=cam.astype(np.float64), dmd=dmd.astype(np.float64),
                              index=idx.astype(np.int64), shift=sh.astype(np.int64),
                              response=rs.astype(np.float64), square_px=s,
                              cam_square_px=float(np.mean(cam_sq)) if cam_sq else float("nan"),
                              shape=tuple(amp.shape), contrast=contrast, lit=lit,
                              warnings=warnings)

    def _saddles(self, d, lit, sq):
        sigma = max(1.0, 0.08 * sq)
        g = ndimage.gaussian_filter(d, sigma)
        gxx = ndimage.gaussian_filter(d, sigma, order=(0, 2))
        gyy = ndimage.gaussian_filter(d, sigma, order=(2, 0))
        gxy = ndimage.gaussian_filter(d, sigma, order=(1, 1))
        resp = gxy * gxy - gxx * gyy
        inner = ndimage.binary_erosion(lit, iterations=max(2, int(0.3 * sq)))
        if not inner.any():
            return None
        c = float(np.median(np.abs(d[inner])))
        ideal = 4.0 / (math.pi ** 2 * sigma ** 4)       # sharp +-1 X-junction
        thr = RESP_FRACTION * ideal * c * c
        size = max(3, int(0.5 * sq) | 1)
        peak = (resp == ndimage.maximum_filter(resp, size=size)) & (resp > thr) & inner
        vy, vx = np.nonzero(peak)
        if vx.size == 0:
            return None
        pts = self._subpixel(g, np.stack([vx, vy], 1).astype(np.float64), sigma)
        good = np.all(np.isfinite(pts), 1)
        return pts[good], resp[vy, vx][good] / ideal

    @staticmethod
    def _subpixel(g, pts, sigma, iters: int = 4):
        """Saddle of a quadratic fitted to g around each point (vectorised)."""
        r = max(2, int(math.ceil(1.5 * sigma)))
        oy, ox = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float64)
        ox, oy = ox.ravel(), oy.ravel()
        A = np.stack([ox * ox, ox * oy, oy * oy, ox, oy, np.ones_like(ox)], 1)
        pinv = np.linalg.pinv(A)                                   # (6, P)
        p = pts.copy()
        start = pts.copy()
        H, W = g.shape
        for _ in range(iters):
            xs = p[:, 0:1] + ox[None, :]
            ys = p[:, 1:2] + oy[None, :]
            vals = ndimage.map_coordinates(g, [ys.ravel(), xs.ravel()], order=1,
                                           mode="nearest").reshape(xs.shape)
            a, b, c, dd, e, _ = pinv @ vals.T                       # each (N,)
            det = 4 * a * c - b * b
            with np.errstate(divide="ignore", invalid="ignore"):
                sx = (-2 * c * dd + b * e) / det
                sy = (b * dd - 2 * a * e) / det
            bad = ~(det < 0) | ~np.isfinite(sx) | ~np.isfinite(sy)
            step = np.stack([np.clip(sx, -1, 1), np.clip(sy, -1, 1)], 1)
            step[bad] = 0
            p = p + step
            p[bad] = np.nan
            if np.all(np.abs(step[~bad]) < 1e-3):
                break
        far = np.any(np.abs(p - start) > r, 1)
        out = (p[:, 0] < 0) | (p[:, 0] > W - 1) | (p[:, 1] < 0) | (p[:, 1] > H - 1)
        p[far | out] = np.nan
        return p

    @staticmethod
    def _name(pts, col, row, okcode, sq):
        n = len(pts)
        ang = np.linspace(0, 2 * np.pi, RING_ANGLES, endpoint=False)
        rads = np.array(RING_RADII) * sq
        ox = (rads[:, None] * np.cos(ang)[None, :]).ravel()
        oy = (rads[:, None] * np.sin(ang)[None, :]).ravel()
        H, W = col.shape
        xs = np.clip(np.rint(pts[:, 0:1] + ox), 0, W - 1).astype(np.intp)
        ys = np.clip(np.rint(pts[:, 1:2] + oy), 0, H - 1).astype(np.intp)
        cv, rv, okv = col[ys, xs], row[ys, xs], okcode[ys, xs]
        k = np.zeros(n, np.int64)
        m = np.zeros(n, np.int64)
        ok = np.zeros(n, bool)
        need = RING_MIN_SHARE * ox.size
        for i in range(n):
            o = okv[i]
            if o.sum() < 0.6 * ox.size:
                continue
            ck, cc = np.unique(cv[i][o], return_counts=True)
            rk, rc = np.unique(rv[i][o], return_counts=True)
            ck, rk = ck[cc >= need], rk[rc >= need]
            if len(ck) == 2 and len(rk) == 2 and ck[1] - ck[0] == 1 and rk[1] - rk[0] == 1:
                k[i], m[i], ok[i] = ck[1], rk[1], True
        return k, m, ok

    # ------------------------------------------------------------- decode
    def decode(self, frames, geometry_id: str, base_frames=None, **_ignored) -> DeformationField:
        """Corners -> a DeformationField: corr_x / corr_y linearly interpolated
        between named corners (inside their hull), conf = checker contrast.
        The corners themselves are in meta['corners'] ([u, v, X, Y] rows)."""
        from scipy.interpolate import griddata
        c = self.detect(frames)
        H, W = c.shape
        cx = np.full((H, W), np.nan, np.float32)
        cy = np.full((H, W), np.nan, np.float32)
        if len(c) >= 3:
            vv, uu = np.mgrid[0:H, 0:W]
            q = (uu.ravel(), vv.ravel())
            cx = griddata(c.cam, c.dmd[:, 0], q, method="linear").reshape(H, W).astype(np.float32)
            cy = griddata(c.cam, c.dmd[:, 1], q, method="linear").reshape(H, W).astype(np.float32)
        mask = np.isfinite(cx) & np.isfinite(cy) & c.lit
        return DeformationField(
            corr_x=cx, corr_y=cy, corr_diag=None, conf=c.contrast.astype(np.float32),
            mask=mask, geometry_id=geometry_id, basis_names=["checkerboard"],
            meta={"corners": c.pairs().round(4).tolist(), "n_corners": len(c),
                  "square_px": self.square_px, "shifts": self.shifts,
                  "cam_square_px": c.cam_square_px, "warnings": c.warnings,
                  "note": "corner basis: corr between corners is linear interpolation"})


# ---------------------------------------------------------------- Jacobian

JACOBIAN_UNITS = {
    "mag_x": "camera px per DMD mirror, along DMD x (the baseline)",
    "mag_y": "camera px per DMD mirror, along DMD y",
    "skew_deg": "degrees (90 - angle between the mapped DMD x and y axes)",
    "anisotropy": "mag_x / mag_y (ratio)",
    "area": "camera px^2 per DMD mirror^2 (|det J|)",
    "stretch": "mag_x / reference mag_x (ratio; 1 = like the reference / median)",
    "contrast": "checker modulation |checker - inverse| / |white - black| (0..1)",
    "center_cam": "camera px (u, v)", "center_dmd": "DMD mirrors (X, Y)",
    "J": "camera px per DMD mirror, [[du/dX, du/dY], [dv/dX, dv/dY]]",
}


@dataclass
class JacobianMap:
    """Per checker cell: local DMD -> camera Jacobian and derived values."""
    center_cam: np.ndarray      # (M, 2)
    center_dmd: np.ndarray      # (M, 2)
    key: np.ndarray             # (M, 4) int (dx, dy, k, m) of the cell's top-left corner
    J: np.ndarray               # (M, 2, 2)
    mag_x: np.ndarray
    mag_y: np.ndarray
    skew_deg: np.ndarray
    anisotropy: np.ndarray
    area: np.ndarray
    stretch: np.ndarray
    contrast: np.ndarray
    shape: tuple
    stretch_ref: str = "median"     # "median" | "reference"
    units: dict = field(default_factory=lambda: dict(JACOBIAN_UNITS))

    VALUES = ("mag_x", "mag_y", "skew_deg", "anisotropy", "area", "stretch", "contrast")

    def __len__(self):
        return len(self.center_cam)

    def image(self, name: str, shape: Optional[tuple] = None) -> np.ndarray:
        """Per-camera-pixel map of one value: linear interpolation between
        cell centres (NaN outside their hull)."""
        from scipy.interpolate import griddata
        if name not in self.VALUES:
            raise ValueError(f"unknown Jacobian value {name!r}; one of {self.VALUES}")
        H, W = shape or self.shape
        out = np.full((H, W), np.nan, np.float32)
        v = getattr(self, name)
        ok = np.isfinite(v)
        if ok.sum() < 3:
            return out
        vv, uu = np.mgrid[0:H, 0:W]
        return griddata(self.center_cam[ok], v[ok], (uu, vv), method="linear").astype(np.float32)

    def arrays(self) -> dict:
        return {n: getattr(self, n) for n in
                ("center_cam", "center_dmd", "key", "J") + self.VALUES}

    def summary(self) -> dict:
        def med(a):
            a = a[np.isfinite(a)]
            return round(float(np.median(a)), 5) if a.size else None
        return {"n_cells": len(self), "stretch_ref": self.stretch_ref,
                **{f"{n}_median": med(getattr(self, n)) for n in self.VALUES}}


def local_jacobian(corners: CheckerCorners, reference=None) -> JacobianMap:
    """Per checker cell (4 named corners) the 2x2 map DMD -> camera.

    reference: a JacobianMap (or CheckerCorners) of the flat base at the
    same geometry; stretch is then mag_x / the reference's mag_x for the
    same cell (else / the median mag_x of this map)."""
    s = corners.square_px
    pos = {}
    for (u, v), (dx, dy), (k, m) in zip(corners.cam, corners.shift, corners.index):
        pos[(int(dx), int(dy), int(k), int(m))] = (u, v)
    cc, cd, keys, Js = [], [], [], []
    for (dx, dy, k, m), p00 in pos.items():
        p10 = pos.get((dx, dy, k + 1, m))
        p01 = pos.get((dx, dy, k, m + 1))
        p11 = pos.get((dx, dy, k + 1, m + 1))
        if p10 is None or p01 is None or p11 is None:
            continue
        p00, p10, p01, p11 = (np.asarray(p) for p in (p00, p10, p01, p11))
        jx = ((p10 - p00) + (p11 - p01)) / (2.0 * s)
        jy = ((p01 - p00) + (p11 - p10)) / (2.0 * s)
        Js.append(np.stack([jx, jy], 1))
        cc.append((p00 + p10 + p01 + p11) / 4.0)
        cd.append(((k + 0.5) * s - dx - 0.5, (m + 0.5) * s - dy - 0.5))
        keys.append((dx, dy, k, m))
    M = len(Js)
    J = np.asarray(Js, np.float64).reshape(M, 2, 2)
    center_cam = np.asarray(cc, np.float64).reshape(M, 2)
    center_dmd = np.asarray(cd, np.float64).reshape(M, 2)
    key = np.asarray(keys, np.int64).reshape(M, 4)
    mag_x = np.linalg.norm(J[:, :, 0], axis=1)
    mag_y = np.linalg.norm(J[:, :, 1], axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cosang = np.sum(J[:, :, 0] * J[:, :, 1], axis=1) / (mag_x * mag_y)
        skew = 90.0 - np.degrees(np.arccos(np.clip(cosang, -1, 1)))
        aniso = mag_x / mag_y
    area = np.abs(np.linalg.det(J)) if M else np.zeros(0)
    if isinstance(reference, CheckerCorners):
        reference = local_jacobian(reference)
    stretch_ref = "median"
    if reference is not None and len(reference):
        ref = {tuple(int(x) for x in kk): mx for kk, mx in zip(reference.key, reference.mag_x)}
        refv = np.array([ref.get(tuple(int(x) for x in kk), np.nan) for kk in key])
        with np.errstate(divide="ignore", invalid="ignore"):
            stretch = mag_x / refv
        stretch_ref = "reference"
    else:
        med = float(np.median(mag_x)) if M else float("nan")
        stretch = mag_x / med if M else np.zeros(0)
    contrast = np.full(M, np.nan)
    if corners.contrast is not None and M:
        sm = ndimage.uniform_filter(corners.contrast.astype(np.float32), 3)
        contrast = ndimage.map_coordinates(sm, [center_cam[:, 1], center_cam[:, 0]], order=1)
    return JacobianMap(center_cam=center_cam, center_dmd=center_dmd, key=key, J=J,
                       mag_x=mag_x, mag_y=mag_y, skew_deg=skew, anisotropy=aniso, area=area,
                       stretch=np.asarray(stretch, np.float64),
                       contrast=np.asarray(contrast, np.float64), shape=corners.shape,
                       stretch_ref=stretch_ref)


# ---------------------------------------------------------------- capture

@dataclass
class CheckerScan:
    corners: CheckerCorners
    jacobian: JacobianMap
    label: str
    geometry_id: str
    npz_path: Optional[object] = None       # output/measurements/*.npz (pairs + cells)
    json_path: Optional[object] = None      # output/measurements/*.json summary
    png_paths: dict = field(default_factory=dict)   # output/previews: stretch, contrast
    warnings: list = field(default_factory=list)
    units: dict = field(default_factory=dict)


def _png_of(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    import cv2
    a = np.clip((np.nan_to_num(arr, nan=lo) - lo) / (hi - lo + 1e-12), 0, 1)
    img = cv2.applyColorMap((a * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    img[~np.isfinite(arr)] = 0
    return img


def _write_png(path, img) -> Optional[object]:
    try:
        import cv2
    except ImportError:
        return None
    from .. import rules
    rules.ensure_new_path(path)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        return None
    with open(path, "xb") as f:
        f.write(buf.tobytes())
    return path


def capture_checker(session, preset, square_px: int = 32, shifts: int = 1,
                    label: str = "checker", reference=None) -> CheckerScan:
    """Project the checker frames (raw, polarity measured by the white/black
    pair), detect corners, compute the Jacobian, save to the output bin.

    reference: a CheckerScan / JacobianMap of the flat base (same geometry)
    for stretch; otherwise stretch is vs the median cell."""
    from .. import rules
    from ..library.store import safe_label
    safe_label(label)
    rules.check_led_duration(preset.projection.led, None, "a checker scan")
    w, h = preset.projection.size
    shape = (int(h), int(w))
    basis = CheckerboardBasis(square_px=square_px, shifts=shifts, dmd_shape=shape)
    with session.rig(preset) as sync:
        frames = [sync.shoot(p, raw=True) for p in basis.generate(shape)]
    corners = basis.detect(frames, shape)
    if isinstance(reference, CheckerScan):
        reference = reference.jacobian
    jac = local_jacobian(corners, reference=reference)
    warnings = list(corners.warnings)
    if len(jac) == 0:
        warnings.append("no complete checker cells: no Jacobian")

    store = session.store
    gid = preset.geometry_id
    npz = store.output_path("measurements", f"{label}_checker", ".npz")
    store.write_npz(npz, pairs=corners.pairs(), index=corners.index, shift=corners.shift,
                    response=corners.response, **{f"cell_{k}": v for k, v in jac.arrays().items()})
    store.record_output("measurements", npz, label=label, geometry_id=gid)
    pngs = {}
    if len(jac) >= 3:
        for name, (lo, hi) in (("stretch", (0.8, 1.2)), ("contrast", (0.0, 1.0))):
            img = _png_of(jac.image(name), lo, hi)
            out = _write_png(store.output_path("previews", f"{label}_{name}", ".png"), img)
            if out:
                store.record_output("previews", out, label=label, geometry_id=gid)
                pngs[name] = out
    units = {"corners.pairs": "[u, v] camera px, [X, Y] DMD mirrors (mirror-centre convention)",
             "square_px": "DMD mirrors", "cam_square_px": "camera px", **JACOBIAN_UNITS}
    doc = {"kind": "checker_scan", "label": label, "geometry_id": gid,
           "preset": getattr(preset, "name", ""), "mock": bool(getattr(session, "mock", False)),
           "created": datetime.now().isoformat(timespec="seconds"),
           "square_px": int(square_px), "shifts": int(shifts), "n_corners": len(corners),
           "cam_square_px": corners.cam_square_px, "jacobian": jac.summary(),
           "npz": str(npz), "pngs": {k: str(v) for k, v in pngs.items()},
           "units": units, "warnings": warnings}
    js = store.output_path("measurements", f"{label}_checker", ".json")
    with open(js, "x", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, default=str)
    store.record_output("measurements", js, label=label, geometry_id=gid)
    return CheckerScan(corners=corners, jacobian=jac, label=label, geometry_id=gid,
                       npz_path=npz, json_path=js, png_paths=pngs, warnings=warnings,
                       units=units)

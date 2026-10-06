"""
Shared numerics of the dent finders: shifted stacks, NaN-aware filters,
robust statistics, and the shear that makes the cell axis vertical.

Coordinates: every map is in CROP pixels of the dent set (full camera =
crop + (x0, y0)). `Rect` is the axis-rectified window used by the finders
that model the surface along the axis: rect column c, row r <->
crop x = c + X0 - tilt * (y - yc), y = r + Y0, so "along the axis" is
"along a rect column".
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import ndimage

SAT = 250.0          # camera DN treated as saturated (8-bit frames, averaged)


class Quiet:
    """Context: NaN-heavy statistics are expected here (masked faces), so
    all-NaN slice / divide warnings are silenced inside it."""

    def __enter__(self):
        self._w = warnings.catch_warnings()
        self._w.__enter__()
        warnings.simplefilter("ignore")
        self._e = np.seterr(all="ignore")
        return self

    def __exit__(self, *exc):
        np.seterr(**self._e)
        return self._w.__exit__(*exc)


def nanmedian_axis(a: np.ndarray, axis: int = 0) -> np.ndarray:
    """NaN-ignoring median along one axis (sort-based; np.nanmedian goes through masked
    arrays and is ~10x slower on the finders' stacks). All-NaN -> NaN."""
    a = np.moveaxis(np.asarray(a, np.float64), axis, 0)
    s = np.sort(a, axis=0)                                  # NaN sorts last
    n = np.isfinite(a).sum(0)
    lo = np.clip((n - 1) // 2, 0, a.shape[0] - 1)[None]
    hi = np.clip(n // 2, 0, a.shape[0] - 1)[None]
    v = 0.5 * (np.take_along_axis(s, lo, 0)[0] + np.take_along_axis(s, hi, 0)[0])
    return np.where(n > 0, v, np.nan)


def shift_stack(a: np.ndarray, shifts, fill=np.nan) -> np.ndarray:
    """Stack of copies of `a` shifted along axis 0: out[i][r] = a[r + shifts[i]]."""
    out = np.full((len(shifts),) + a.shape, fill, dtype=a.dtype)
    n = a.shape[0]
    for i, s in enumerate(shifts):
        s = int(s)
        if s >= 0:
            out[i, :n - s] = a[s:]
        else:
            out[i, -s:] = a[:n + s]
    return out


def nan_gauss(a: np.ndarray, sigma, min_w: float = 0.25) -> np.ndarray:
    """Gaussian blur that ignores NaN (normalised by the blurred weight)."""
    ok = np.isfinite(a)
    num = ndimage.gaussian_filter(np.where(ok, a, 0.0), sigma)
    den = ndimage.gaussian_filter(ok.astype(float), sigma)
    with Quiet():
        out = num / np.maximum(den, 1e-6)
    out[den < min_w] = np.nan
    return out


def mad_sigma(v) -> float:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    return 1.4826 * float(np.median(np.abs(v - np.median(v))))


def robust_global_norm(S: np.ndarray, mask: np.ndarray):
    """(S - median) / (1.4826 MAD) over mask -> (z, median, mad)."""
    v = S[mask & np.isfinite(S)]
    if v.size == 0:
        return np.full(S.shape, np.nan), 0.0, 1.0
    med = float(np.median(v))
    mad = 1.4826 * float(np.median(np.abs(v - med)))
    mad = mad if mad > 1e-12 else 1.0
    return (S - med) / mad, med, mad


def disk(r: float) -> np.ndarray:
    k = int(np.ceil(r))
    yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
    return np.hypot(yy, xx) <= r + 0.01


class Rect:
    """Shear making the cell axis vertical, in crop coordinates.

    rect (c, r) <-> crop x = c + X0 - tilt * (y - yc), y = r + Y0."""

    def __init__(self, tilt: float, yc: float, X0: int, X1: int, Y0: int, Y1: int):
        self.tilt, self.yc = float(tilt), float(yc)
        self.X0, self.X1, self.Y0, self.Y1 = int(X0), int(X1), int(Y0), int(Y1)
        xr, yr = np.meshgrid(np.arange(self.X0, self.X1), np.arange(self.Y0, self.Y1))
        self.srcx = (xr - self.tilt * (yr - self.yc)).astype(np.float32)
        self.srcy = yr.astype(np.float32)
        self.shape = xr.shape

    @classmethod
    def from_face(cls, face: np.ndarray, pad_x: int = 10, pad_y: int = 22) -> "Rect":
        """Tilt from a robust line through the face's row centres."""
        rows = np.nonzero(face.any(1))[0]
        if len(rows) < 10:
            raise ValueError("the analysed cell face is too small (under 10 rows)")
        cen = np.array([np.nonzero(face[r])[0].mean() for r in rows])
        keep = np.ones(len(rows), bool)
        k = b = 0.0
        for _ in range(4):
            k, b = np.polyfit(rows[keep], cen[keep], 1)
            res = cen - (k * rows + b)
            s = 1.4826 * np.median(np.abs(res[keep])) + 0.3
            keep = np.abs(res) < 3 * s
        tilt = -float(k)
        yc = float(0.5 * (rows[0] + rows[-1]))
        ys, xs = np.nonzero(face)
        c = xs + tilt * (ys - yc)
        return cls(tilt, yc, int(np.floor(c.min())) - pad_x, int(np.ceil(c.max())) + pad_x + 1,
                   int(rows[0]) - pad_y, int(rows[-1]) + pad_y + 1)

    def fwd(self, a: np.ndarray, nearest: bool = False) -> np.ndarray:
        """crop -> rect (NaN outside the crop)."""
        import cv2
        interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
        return cv2.remap(np.ascontiguousarray(a, dtype=np.float32), self.srcx, self.srcy, interp,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=float("nan"))

    def fwd_bool(self, m: np.ndarray) -> np.ndarray:
        return np.nan_to_num(self.fwd(m.astype(np.float32), nearest=True), nan=0.0) > 0.5

    def back(self, a: np.ndarray, shape: tuple) -> np.ndarray:
        """rect -> crop (NaN outside the rect window)."""
        import cv2
        H, W = shape
        xc, yc_ = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
        c = xc + self.tilt * (yc_ - self.yc) - self.X0
        r = yc_ - self.Y0
        return cv2.remap(np.ascontiguousarray(a, dtype=np.float32), c.astype(np.float32),
                         r.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                         borderValue=float("nan"))

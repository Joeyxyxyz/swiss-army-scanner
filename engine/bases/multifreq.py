"""
Multi-frequency phase-shift basis: N steps (default 12) at each of several
fringe periods (default 8, 80, 1280 DMD px), unwrapped hierarchically in
time — coarse to fine, each level fixing the fringe order of the next.
Not heterodyne: every level is a real projected period, and the coarsest
period is longer than the coded axis, so its phase alone is absolute.

  frames: white, black, then per coded axis (x, then y), per period
          (fine -> coarse), N fringes I_k = 127.5 (1 + cos(2 pi pos / T + 2 pi k / N))

  decode: phi = atan2(-sum I_k sin d_k, sum I_k cos d_k), B = 2/N |.|
          pos_coarsest = wrap(phi) T / 2 pi                   (absolute)
          k_i = round((2 pi pos_{i+1} / T_i - phi_i) / 2 pi)   (next finer)
          pos_i = (phi_i + 2 pi k_i) T_i / 2 pi

A pixel is kept when the white-black contrast and the finest level's
modulation B clear the floors (the modulation-B mask), and every level
agrees with the one above it within a quarter of its period (else the
fringe order is a guess and the pixel is dropped, never kept). A coarse
error of a whole number of finer periods is self-consistent and can only
be caught by the axis range check; keep level ratios modest (<= 20).

axis: "x" (vertical fringes -> projector column, corr_x), "y" (horizontal
fringes -> projector row, corr_y) or "xy" (both, one capture: what the
projector calibration and ray-ray triangulation need).

diamond_parity: the DLP4500's rows are offset by half a column on
alternate rows. None = treat the DMD as a square grid (column index =
native column). 0 or 1 = rows with row % 2 == parity sit half a column to
the right (measure it with calibrate/parity.py): x fringes are then drawn
in native continuous column coordinates, so they are straight in space
and corr_x is the native column of the unshifted rows' lattice (the convention
the projector calibration uses).

The white/black frames measure polarity (the bench HDMI path inverts);
inverted fringes are cos(...) + pi and corrected by pi.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..decode.deformation import DeformationField
from .base import PatternBasis
from .registry import register

DMD_SHAPE = (1140, 912)                 # (rows, cols)
DEFAULT_PERIODS = (8, 80, 1280)
DEFAULT_STEPS = 12
MAX_LEVEL_RATIO = 20                    # finer/coarser period ratio the unwrap tolerates
_AXES = {"x": ("x",), "y": ("y",), "xy": ("x", "y")}


@register
class MultiFreqBasis(PatternBasis):
    name = "multifreq"
    axes = {"x", "y"}

    def __init__(self, steps: int = DEFAULT_STEPS, periods_px=DEFAULT_PERIODS,
                 axis: str = "x", diamond_parity: Optional[int] = None,
                 mod_min: Optional[float] = None):
        if axis not in _AXES:
            raise ValueError(f"multifreq: axis must be one of {sorted(_AXES)}, got {axis!r}")
        if isinstance(steps, bool) or int(steps) != steps or int(steps) < 3:
            raise ValueError(f"multifreq: steps must be an integer >= 3, got {steps!r}")
        periods = [float(p) for p in (periods_px or DEFAULT_PERIODS)]
        if len(periods) < 2:
            raise ValueError("multifreq: need at least 2 periods (fine ... absolute coarse)")
        if any(p < 3 for p in periods) or sorted(periods) != periods or len(set(periods)) != len(periods):
            raise ValueError(f"multifreq: periods_px must be increasing and >= 3, got {periods}")
        for a, b in zip(periods, periods[1:]):
            if b / a > MAX_LEVEL_RATIO:
                raise ValueError(f"multifreq: period ratio {b:g}/{a:g} > {MAX_LEVEL_RATIO}: the "
                                 "coarser level can't fix the finer fringe order reliably; add a "
                                 "level in between")
        self._coded = _AXES[axis]
        extent = {"x": DMD_SHAPE[1], "y": DMD_SHAPE[0]}
        for ax in self._coded:
            if periods[-1] < extent[ax]:
                raise ValueError(f"multifreq axis {ax}: the coarsest period {periods[-1]:g} px "
                                 f"must cover the whole axis ({extent[ax]} px) to be absolute")
        if diamond_parity not in (None, 0, 1):
            raise ValueError(f"multifreq: diamond_parity must be null, 0 or 1, got {diamond_parity!r}")
        super().__init__(steps=int(steps), periods_px=periods, axis=axis,
                         diamond_parity=diamond_parity, mod_min=mod_min)
        self.steps = int(steps)
        self.periods = periods
        self.axis = axis
        self.axes = set(self._coded)
        self.parity = diamond_parity
        self.mod_min = mod_min
        self.extent = extent

    def n_frames(self) -> int:
        return 2 + len(self._coded) * len(self.periods) * self.steps

    # ---------------------------------------------------------- patterns
    def _coord(self, ax: str, shape) -> np.ndarray:
        H, W = shape
        if ax == "y":
            return np.repeat(np.arange(H, dtype=np.float64)[:, None], W, axis=1)
        c = np.repeat(np.arange(W, dtype=np.float64)[None, :], H, axis=0)
        if self.parity is not None:
            c = c + 0.5 * ((np.arange(H)[:, None] & 1) == self.parity)
        return c

    def generate(self, dmd_shape=DMD_SHAPE) -> list[np.ndarray]:
        shape = tuple(dmd_shape or DMD_SHAPE)
        out = [np.full(shape, 255, np.uint8), np.zeros(shape, np.uint8)]
        for ax in self._coded:
            pos = self._coord(ax, shape)
            for T in self.periods:
                for k in range(self.steps):
                    ph = 2 * np.pi * pos / T + 2 * np.pi * k / self.steps
                    out.append(np.clip(np.rint(127.5 * (1 + np.cos(ph))), 0, 255).astype(np.uint8))
        return out

    # ------------------------------------------------------------ decode
    def _wrapped(self, frames):
        n = len(frames)
        d = 2 * np.pi * np.arange(n) / n
        S = sum(f.astype(np.float64) * np.sin(dk) for f, dk in zip(frames, d))
        C = sum(f.astype(np.float64) * np.cos(dk) for f, dk in zip(frames, d))
        return np.arctan2(-S, C), 2.0 / n * np.hypot(S, C)

    def _offset(self, frames):
        return sum(f.astype(np.float64) for f in frames) / len(frames)

    def decode(self, frames, geometry_id: str, base_frames=None,
               contrast_min: float = 8.0, conf_min: float = 4.0) -> DeformationField:
        if len(frames) != self.n_frames():
            raise ValueError(f"multifreq expects {self.n_frames()} frames (white, black, "
                             f"{len(self._coded)} axis x {len(self.periods)} periods x "
                             f"{self.steps} steps), got {len(frames)}")
        white, black = frames[0].astype(np.float64), frames[1].astype(np.float64)
        contrast = white - black
        flipped = float(np.nanmean(contrast)) < 0
        if flipped:
            contrast = -contrast
        mod_floor = conf_min if self.mod_min is None else float(self.mod_min)
        mask = contrast > contrast_min
        corr, meta_ax, conf = {}, {}, None
        i = 2
        for ax in self._coded:
            levels = []
            for _ in self.periods:
                ph, B = self._wrapped(frames[i:i + self.steps])
                if flipped:
                    ph = ph + np.pi
                levels.append((ph, B))
                i += self.steps
            T = self.periods[-1]
            pos = np.mod(levels[-1][0], 2 * np.pi) * T / (2 * np.pi)
            ext = self.extent[ax]
            pos = np.where(pos > T - (T - ext) / 2, pos - T, pos)   # just below 0, not near T
            ok = np.ones(pos.shape, bool)
            for (ph, _), Ti in zip(levels[-2::-1], self.periods[-2::-1]):
                k = np.round((2 * np.pi * pos / Ti - ph) / (2 * np.pi))
                new = (ph + 2 * np.pi * k) * Ti / (2 * np.pi)
                ok &= np.abs(new - pos) < Ti / 4
                pos = new
            B = levels[0][1]
            A = self._offset(frames[i - len(self.periods) * self.steps:
                                    i - (len(self.periods) - 1) * self.steps])
            good = mask & (B > mod_floor) & ok & (pos >= -0.5) & (pos <= ext - 0.5)
            lit = mask & (B > mod_floor)
            corr[ax] = np.where(good, pos, np.nan).astype(np.float32)
            conf = B if conf is None else np.minimum(conf, B)
            meta_ax[ax] = {"modulation_median": float(np.median(B[lit])) if lit.any() else 0.0,
                           "offset_median": float(np.median(A[lit])) if lit.any() else 0.0,
                           "unwrap_rejected_pct": round(100.0 * float((lit & ~ok).sum())
                                                        / max(int(lit.sum()), 1), 3)}
        valid = np.all([np.isfinite(c) for c in corr.values()], axis=0)
        return DeformationField(
            corr_x=corr.get("x"), corr_y=corr.get("y"), corr_diag=None,
            conf=conf.astype(np.float32), mask=valid, geometry_id=geometry_id,
            basis_names=["multifreq"],
            meta={"polarity_flipped": flipped, "steps": self.steps, "periods_px": self.periods,
                  "axis": self.axis, "diamond_parity": self.parity,
                  "units": "projector px (native col, row)",
                  "gc_conf": float(np.median(conf[valid])) if valid.any() else 0.0,
                  "contrast_mean": float(contrast.mean()), "mod_min": mod_floor,
                  "per_axis": meta_ax})


def from_spec(spec: Optional[dict] = None, **override) -> MultiFreqBasis:
    s = {k: v for k, v in (spec or {}).items() if k != "name"}
    s.update({k: v for k, v in override.items() if v is not None})
    return MultiFreqBasis(**s)

"""
Tile + phase basis (EXPERIMENTAL, method contributed by Grok (xAI)): a fast depth set for a SMALL
lit zone (the targeted end-face scan). Per coded axis one fine phase-shift level (N steps,
default 12, at T = 8 DMD px) plus a few Gray-code bit planes that give the fringe order -
instead of the multifreq basis's two coarser phase levels.

  frames: white, black, then per coded axis (x, then y): N fringes at period T, then
          `code_bits` binary planes of Gray(m), m = floor(2 q / T) mod 2 W  (q = pos + 0.5,
          so the phase wraps exactly on tile edges; W = 2 ** (code_bits - 1) tiles)
          2 + 2 (12 + 5) = 36 frames for xy, vs 2 + 2 x 3 x 12 = 74 for the default multifreq.

  decode: f = wrap(phi) T / 2 pi (position inside the tile, 0..T)
          k1 = m >> 1           tile index mod W, its edges at the phase wrap
          k2 = round(q / T)     mod W, its edges mid-fringe
          the fringe order is read from whichever code is far from its own edges
          (f in [T/4, 3T/4): k1, else k2), so no order is ever decided on a blurred code edge;
          where both codes are clean (f in [T/8, 3T/8) or [5T/8, 7T/8)) they must agree, else
          the pixel is dropped.

LIMITS (read before using it):
- The order is unique only inside a window of W x T DMD px per axis (5 bits: 128 px) placed on
  the lit zone. It is absolute only because nothing outside that window is lit: the caller
  must light only a zone that fits inside it (targeted_face_scan checks the face mask does).
  A full-field scan would alias.
- Binary code planes clip and blur unlike sinusoids: heavy defocus (> ~2 DMD px), steep tilt
  (the 8 px period compressed in the camera) or glints can flip a code bit.
- Only the k1 / k2 consistency check guards the order (no multi-level phase check): a tile
  error where both codes are wrong together is not caught.
- Motion during the code frames corrupts the order of the moved pixels (fewer frames make
  it less likely, not impossible).
- No flood pair beyond white / black and no polar set of its own (targeted_face_scan still
  appends the polar set / flood pair after it).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..decode.deformation import DeformationField
from .base import PatternBasis
from .multifreq import DMD_SHAPE

DEFAULT_PERIOD = 8.0
DEFAULT_STEPS = 12
DEFAULT_CODE_BITS = 5


def gray_to_bin(g: np.ndarray) -> np.ndarray:
    m = g.copy()
    s = 1
    while s < 64:
        m ^= m >> s
        s *= 2
    return m


class TilePhaseBasis(PatternBasis):
    name = "tilephase"
    axes = {"x", "y"}

    def __init__(self, centre, steps: int = DEFAULT_STEPS, period_px: float = DEFAULT_PERIOD,
                 code_bits: int = DEFAULT_CODE_BITS, diamond_parity: Optional[int] = None):
        """centre: (col, row) DMD px the unique window is centred on (the lit zone's centre)."""
        if isinstance(steps, bool) or int(steps) != steps or int(steps) < 3:
            raise ValueError(f"tilephase: steps must be an integer >= 3, got {steps!r}")
        if not 4 <= float(period_px) <= 64:
            raise ValueError(f"tilephase: period_px must be 4..64, got {period_px!r}")
        if isinstance(code_bits, bool) or int(code_bits) != code_bits or not 3 <= int(code_bits) <= 9:
            raise ValueError(f"tilephase: code_bits must be 3..9, got {code_bits!r}")
        if diamond_parity not in (None, 0, 1):
            raise ValueError(f"tilephase: diamond_parity must be null, 0 or 1, got {diamond_parity!r}")
        super().__init__(steps=int(steps), period_px=float(period_px), code_bits=int(code_bits),
                         diamond_parity=diamond_parity)
        self.steps = int(steps)
        self.T = float(period_px)
        self.bits = int(code_bits)
        self.win = 2 ** (self.bits - 1)               # tiles per axis in the unique window
        self.parity = diamond_parity
        self.extent = {"x": DMD_SHAPE[1], "y": DMD_SHAPE[0]}
        c = {"x": float(centre[0]), "y": float(centre[1])}
        self.K0 = {ax: int(np.floor((c[ax] + 0.5 - self.win * self.T / 2) / self.T))
                   for ax in ("x", "y")}

    @property
    def window_px(self) -> float:
        return self.win * self.T

    def window(self, ax: str) -> tuple:
        """[lo, hi) DMD px of the unique window on axis ax."""
        lo = self.K0[ax] * self.T - 0.5
        return lo, lo + self.window_px

    def fits(self, mask: np.ndarray) -> Optional[str]:
        """None when every lit DMD px of mask lies inside the unique window, else why not."""
        rows, cols = np.nonzero(mask)
        if not len(rows):
            return "the lit zone is empty"
        for ax, v in (("x", cols), ("y", rows)):
            lo, hi = self.window(ax)
            if v.min() < lo or v.max() + 1 > hi:
                return (f"the lit zone spans {ax} {v.min()}..{v.max()} DMD px, the {self.bits}-bit "
                        f"unique window is {lo + 0.5:.0f}..{hi - 0.5:.0f} ({self.window_px:.0f} px): "
                        "use more code bits")
        return None

    def n_frames(self) -> int:
        return 2 + 2 * (self.steps + self.bits)

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
        T = self.T
        for ax in ("x", "y"):
            q = self._coord(ax, shape) + 0.5
            for k in range(self.steps):
                ph = 2 * np.pi * q / T + 2 * np.pi * k / self.steps
                out.append(np.clip(np.rint(127.5 * (1 + np.cos(ph))), 0, 255).astype(np.uint8))
            m = np.mod(np.floor(2 * q / T).astype(np.int64), 2 * self.win)
            g = m ^ (m >> 1)
            for j in range(self.bits):
                out.append((((g >> j) & 1) * 255).astype(np.uint8))
        return out

    def decode(self, frames, geometry_id: str, base_frames=None,
               contrast_min: float = 8.0, conf_min: float = 4.0) -> DeformationField:
        if len(frames) != self.n_frames():
            raise ValueError(f"tilephase expects {self.n_frames()} frames (white, black, 2 axes x "
                             f"({self.steps} steps + {self.bits} code bits)), got {len(frames)}")
        T, N, W = self.T, self.steps, self.win
        white, black = frames[0].astype(np.float64), frames[1].astype(np.float64)
        con = white - black
        flipped = float(np.nanmean(con)) < 0
        den = np.where(np.abs(con) > 1e-6, con, np.nan)
        base = np.abs(con) > contrast_min
        d = 2 * np.pi * np.arange(N) / N
        corr, meta_ax, conf = {}, {}, None
        i = 2
        for ax in ("x", "y"):
            F = frames[i:i + N]
            S = sum(f.astype(np.float64) * np.sin(dk) for f, dk in zip(F, d))
            C = sum(f.astype(np.float64) * np.cos(dk) for f, dk in zip(F, d))
            ph, B = np.arctan2(-S, C), 2.0 / N * np.hypot(S, C)
            if flipped:
                ph = ph + np.pi
            f = np.mod(ph, 2 * np.pi) * T / (2 * np.pi)
            i += N
            g = np.zeros(white.shape, np.int64)
            margin = np.full(white.shape, np.inf)
            for j in range(self.bits):
                # (frame - black) / (white - black): 1 = lit, either polarity
                r = (frames[i + j].astype(np.float64) - black) / den
                g |= (np.nan_to_num(r, nan=0.0) > 0.5).astype(np.int64) << j
                margin = np.minimum(margin, np.abs(np.nan_to_num(r, nan=0.5) - 0.5))
            i += self.bits
            m = gray_to_bin(g)
            k1 = m >> 1
            k2 = ((m + 1) >> 1) % W
            nmod = np.where(f < T / 4, k2, np.where(f >= 3 * T / 4, (k2 - 1) % W, k1))
            zA = (f >= T / 8) & (f < 3 * T / 8)
            zB = (f >= 5 * T / 8) & (f < 7 * T / 8)
            cons = ~((zA & (k2 != k1)) | (zB & (k2 != (k1 + 1) % W)))
            K = self.K0[ax] + np.mod(nmod - self.K0[ax], W)
            pos = K * T + f - 0.5
            lit = base & (B > conf_min)
            ext = self.extent[ax]
            good = lit & cons & (pos >= -0.5) & (pos <= ext - 0.5)
            corr[ax] = np.where(good, pos, np.nan).astype(np.float32)
            conf = B if conf is None else np.minimum(conf, B)
            meta_ax[ax] = {"modulation_median": float(np.median(B[lit])) if lit.any() else 0.0,
                           "unwrap_rejected_pct": round(100.0 * float((lit & ~cons).sum())
                                                        / max(int(lit.sum()), 1), 3),
                           "code_margin_median": float(np.median(margin[lit])) if lit.any() else 0.0,
                           "window_start_tile": self.K0[ax]}
        valid = np.isfinite(corr["x"]) & np.isfinite(corr["y"])
        return DeformationField(
            corr_x=corr["x"], corr_y=corr["y"], corr_diag=None, conf=conf.astype(np.float32),
            mask=valid, geometry_id=geometry_id, basis_names=["tilephase"],
            meta={"polarity_flipped": flipped, "steps": N, "period_px": T,
                  "code_bits_per_axis": self.bits, "window_tiles": W, "axis": "xy",
                  "diamond_parity": self.parity, "units": "projector px (native col, row)",
                  "gc_conf": float(np.median(conf[valid])) if valid.any() else 0.0,
                  "contrast_mean": float(con.mean()), "mod_min": conf_min,
                  "experimental": True, "per_axis": meta_ax})

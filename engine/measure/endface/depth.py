"""
How deep and how wide a dent is, read off the deviation map (mm on the face grid).

The deepest single grid point of a dent over-reads: it is the dent PLUS the most negative
noise of the handful of cells at its bottom. Measured on the bench, the same dent (0.46 mm
deep, 1.33 mm wide by a depth gauge) read 0.57-0.65 mm at its deepest point on two rigs and
0.49 mm as the average of the 3 x 3 grid cells at its centre. So:

  depth_centre_mm      THE depth: the mean of the 3 x 3 grid cells (0.75 mm square on the
                       0.25 mm grid, about two camera px) centred where that mean is deepest
                       inside the dent, against the LOCAL background (the median of the ring
                       around the dent). + = a dent (below the surface), - = a bump.
  depth_min_mm         the deepest single grid point against the same background: secondary,
                       'single point, over-reads' (kept for comparison with older reports)
  width_half_depth_mm  the diameter of the region around the centre that is at least half
                       depth_centre_mm deep (the circle of the same area). It reads about one
                       camera px wide: the camera's pixel footprint blurs the dent's wall
                       (the 1.33 mm dent read 1.5-1.64 mm at 0.36 mm / px).

Thresholds (min_depth_mm, a recipe's depth_mm) apply to depth_centre_mm.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np

DEPTH_MIN_NOTE = "single point, over-reads"
MIN_BOX_CELLS = 5                   # of the 9 cells of the 3 x 3 average
MIN_RING_CELLS = 12                 # cells of the background ring; fewer: referenced to 0
RING = (0.75, 1.5)                  # background ring radii, x the dent diameter
UPSAMPLE = 4                        # half-depth area on a 4 x finer (bilinear) grid


def width_note(px_mm: Optional[float] = None) -> str:
    px = f" (~{px_mm:.2f} mm here)" if px_mm else ""
    return ("diameter of the region at least half as deep as depth_centre_mm; reads about one "
            f"camera px{px} wide: the pixel footprint blurs the dent's wall")


def box3(a: np.ndarray) -> np.ndarray:
    """NaN-aware 3 x 3 mean (NaN where fewer than MIN_BOX_CELLS of the 9 cells are measured)."""
    import cv2
    fin = np.isfinite(a)
    num = cv2.boxFilter(np.where(fin, a, 0.0), cv2.CV_64F, (3, 3), normalize=False,
                        borderType=cv2.BORDER_CONSTANT)
    den = cv2.boxFilter(fin.astype(np.float64), cv2.CV_64F, (3, 3), normalize=False,
                        borderType=cv2.BORDER_CONSTANT)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[den < MIN_BOX_CELLS] = np.nan
    return out


def _half_width(excess: np.ndarray, cy: int, cx: int, half: float, step: float) -> Optional[float]:
    """Equivalent diameter (mm) of the connected region around (cy, cx) where excess >=
    half, on a bilinear UPSAMPLE x finer grid."""
    import cv2
    if not half > 0:
        return None
    e = np.where(np.isfinite(excess), excess, 0.0).astype(np.float32)
    up = cv2.resize(e, None, fx=UPSAMPLE, fy=UPSAMPLE, interpolation=cv2.INTER_LINEAR)
    m = (up >= half).astype(np.uint8)
    # the upsampled cell centre of (cy, cx): cv2.resize maps centre to centre
    uy, ux = int((cy + 0.5) * UPSAMPLE), int((cx + 0.5) * UPSAMPLE)
    if not (0 <= uy < m.shape[0] and 0 <= ux < m.shape[1]) or not m[uy, ux]:
        return None
    _, lab = cv2.connectedComponents(m, connectivity=8)
    area = float((lab == lab[uy, ux]).sum()) * (step / UPSAMPLE) ** 2
    return 2.0 * float(np.sqrt(area / np.pi))


def dent_metrics(dev: np.ndarray, g, y: int, x: int, width_mm: float, sign: int = 1,
                 allowed: Optional[np.ndarray] = None, px_mm: Optional[float] = None,
                 dev3: Optional[np.ndarray] = None) -> dict:
    """The depth and width of the dent (sign +1) or bump (sign -1) found at grid cell (y, x)
    of the deviation map dev (mm, - = below the surface), whose diameter is about width_mm.
    allowed: cells that may be used (the fitted / inspected zone). dev3: box3(dev), when the
    caller has it. -> {depth_centre_mm, depth_min_mm, width_half_depth_mm, background_mm,
    centre (u_mm, v_mm), ...}; depth_centre_mm None when the 3 x 3 cells at the dent are not
    measured (the caller then keeps its own depth)."""
    g = np.asarray(g, np.float64)
    step = float(g[1] - g[0])
    fin = np.isfinite(dev) if allowed is None else (np.isfinite(dev) & allowed)
    d = np.where(fin, dev, np.nan)
    d3 = box3(d) if dev3 is None or allowed is not None else dev3
    out = {"depth_centre_mm": None, "depth_min_mm": None, "width_half_depth_mm": None,
           "background_mm": None, "depth_min_note": DEPTH_MIN_NOTE,
           "width_note": width_note(px_mm)}
    w = max(float(width_mm), 2 * step)
    # everything within 2.5 dent diameters of the cell, as a local window
    n = int(np.ceil(2.5 * w / step)) + 2
    y0, y1 = max(y - n, 0), min(y + n + 1, d.shape[0])
    x0, x1 = max(x - n, 0), min(x + n + 1, d.shape[1])
    L, L3 = d[y0:y1, x0:x1], d3[y0:y1, x0:x1]
    yy, xx = np.mgrid[y0:y1, x0:x1]
    rr = np.hypot((yy - y) * step, (xx - x) * step)
    foot = rr <= max(w / 2, 2 * step) + 1e-9
    s = 1.0 if sign >= 0 else -1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        c3 = np.where(foot, -s * L3, np.nan)                 # + = deeper (dent) / higher (bump)
        if not np.isfinite(c3).any():
            return out
        k = int(np.nanargmax(c3))
        cy, cx = np.unravel_index(k, c3.shape)
        bg, n_ring = 0.0, 0
        for _ in range(3):                                   # the ring follows the measured width
            rc = np.hypot((yy - (cy + y0)) * step, (xx - (cx + x0)) * step)
            ring = (rc >= RING[0] * w) & (rc <= RING[1] * w) & np.isfinite(L)
            n_ring = int(ring.sum())
            bg = float(np.median(L[ring])) if n_ring >= MIN_RING_CELLS else 0.0
            depth_c = float(s * (bg - L3[cy, cx]))
            wh = _half_width(s * (bg - L), cy, cx, 0.5 * depth_c, step)
            full = None if wh is None else float(np.sqrt(2.0) * wh)   # a paraboloid's rim
            if full is None or full <= w * 1.15 or RING[1] * full > 2.5 * w + step:
                break
            w = full
        pt = np.where(foot | (rc <= w / 2), -s * L, np.nan)
        dmin = float(s * (bg - L[np.unravel_index(int(np.nanargmax(pt)), pt.shape)])) \
            if np.isfinite(pt).any() else None
    # depth_c / dmin are magnitudes of the dent (or bump); signed: + = a dent, - = a bump
    out.update({"depth_centre_mm": round(s * depth_c, 4),
                "depth_min_mm": None if dmin is None else round(s * dmin, 4),
                "width_half_depth_mm": None if wh is None else round(wh, 3),
                "background_mm": round(bg, 4), "background_cells": n_ring,
                "centre_u_mm": round(float(g[cx + x0]), 3),
                "centre_v_mm": round(float(g[cy + y0]), 3)})
    return out

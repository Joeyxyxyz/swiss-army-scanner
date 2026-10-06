"""
swiss_army_sensor.golden
========================

Golden-model (CAD) comparison: scan vs. reference, with tolerance bands.

This is the inspection workflow fixed-function vendors sell as
"scan-to-CAD": load a reference geometry, rasterize it to the scanner's
viewpoint, subtract the measured surface, classify deviations against
per-axis tolerances, return a pass/fail report with a deviation map.

Deliberately simple v1 (no heavy deps):
  * STL only — the one mesh format every CAD package exports
    (binary and ASCII, parsed natively with numpy).
  * Top-down orthographic rasterization (z-buffer max) — matches a
    single-camera structured-light height map.
  * Z tolerance = signed surface deviation bands (warn / fail).
  * X/Y tolerance = footprint boundary offset, measured where the part
    silhouette edge lands vs. where the CAD says it should.

UNITS — read this before trusting numbers:
  The comparison is unit-agnostic: it trusts that `measured` is in the
  same units and on the same grid as the rasterized reference. Producing
  a *metric* height map from phase requires the calibration chain
  (gamma + camera-projector stereo + phase unwrapping), which is not in
  this package yet. Until then, use `scale` / fixture constants from a
  known artifact, and treat absolute numbers accordingly. Relative
  deviation (dents, bumps, missing material, glue beads) is meaningful
  today; certified metric tolerance verdicts are not.

Typical flow:
    verts, faces = load_stl("bracket.stl")
    ref = rasterize_heightmap(verts, faces, grid=(480, 640))
    dev = compare_to_golden(measured, ref,
                            tol_z=ToleranceBand(warn=0.1, fail=0.3))
    print(dev.passed, dev.pct_in_tolerance)
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


# --------------------------------------------------------------------- STL

def load_stl(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load an STL file (binary or ASCII).

    Returns (vertices, faces): vertices float64 (N,3), faces int64 (M,3).
    Vertices are deduplicated so faces share indices.
    """
    with open(path, "rb") as f:
        raw = f.read()
    if _is_binary_stl(raw):
        tris = _parse_binary_stl(raw)
    else:
        tris = _parse_ascii_stl(raw.decode("utf-8", errors="replace"))
    if tris.size == 0:
        raise ValueError(f"No triangles found in {path}")
    flat = tris.reshape(-1, 3)
    verts, inverse = np.unique(np.round(flat, 9), axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3).astype(np.int64)
    return verts.astype(np.float64), faces


def _is_binary_stl(raw: bytes) -> bool:
    if len(raw) < 84:
        return False
    (n,) = struct.unpack_from("<I", raw, 80)
    if len(raw) == 84 + 50 * n:
        return True
    # 'solid' header alone is not conclusive — some binary files start
    # with it; the size test above is authoritative when it matches.
    return not raw.lstrip()[:5].lower() == b"solid"


def _parse_binary_stl(raw: bytes) -> np.ndarray:
    (n,) = struct.unpack_from("<I", raw, 80)
    body = np.frombuffer(raw, dtype=np.uint8, count=50 * n, offset=84)
    rec = body.reshape(n, 50)
    floats = rec[:, :48].copy().view("<f4").reshape(n, 12)
    return floats[:, 3:12].astype(np.float64).reshape(n, 3, 3)


def _parse_ascii_stl(text: str) -> np.ndarray:
    pts = []
    for line in text.splitlines():
        s = line.strip().split()
        if len(s) == 4 and s[0].lower() == "vertex":
            pts.append([float(s[1]), float(s[2]), float(s[3])])
    arr = np.asarray(pts, dtype=np.float64)
    if arr.size and len(arr) % 3 == 0:
        return arr.reshape(-1, 3, 3)
    return np.empty((0, 3, 3))


def save_stl(path: str, verts: np.ndarray, faces: np.ndarray) -> None:
    """Write binary STL (useful for tests and synthetic golden parts)."""
    tris = verts[faces]                                  # (M,3,3)
    n = len(tris)
    e1 = tris[:, 1] - tris[:, 0]
    e2 = tris[:, 2] - tris[:, 0]
    nrm = np.cross(e1, e2)
    nn = np.linalg.norm(nrm, axis=1, keepdims=True)
    nrm = np.where(nn > 0, nrm / np.maximum(nn, 1e-12), 0.0)
    with open(path, "wb") as f:
        f.write(b"\x00" * 80)
        f.write(struct.pack("<I", n))
        for i in range(n):
            f.write(struct.pack("<3f", *nrm[i]))
            for v in tris[i]:
                f.write(struct.pack("<3f", *v))
            f.write(b"\x00\x00")


# ------------------------------------------------------------ rasterization

def rasterize_heightmap(verts: np.ndarray, faces: np.ndarray,
                        grid: Tuple[int, int] = (480, 640),
                        bounds: Optional[Tuple[float, float, float, float]] = None
                        ) -> "ReferenceSurface":
    """
    Orthographic top-down rasterization: z-buffer max over triangles.

    grid   = (rows, cols) of the output height map.
    bounds = (xmin, xmax, ymin, ymax) in CAD units; defaults to the mesh
             bounding box. Cells the mesh does not cover are NaN.
    """
    H, W = grid
    if bounds is None:
        xmin, ymin = verts[:, 0].min(), verts[:, 1].min()
        xmax, ymax = verts[:, 0].max(), verts[:, 1].max()
    else:
        xmin, xmax, ymin, ymax = bounds
    px = (xmax - xmin) / W
    py = (ymax - ymin) / H
    z = np.full((H, W), np.nan, dtype=np.float64)

    tris = verts[faces]                                  # (M,3,3)
    for t in tris:
        # triangle bbox in pixel coords
        txmin = int(np.floor((t[:, 0].min() - xmin) / px))
        txmax = int(np.ceil((t[:, 0].max() - xmin) / px))
        tymin = int(np.floor((t[:, 1].min() - ymin) / py))
        tymax = int(np.ceil((t[:, 1].max() - ymin) / py))
        txmin, txmax = max(txmin, 0), min(txmax, W - 1)
        tymin, tymax = max(tymin, 0), min(tymax, H - 1)
        if txmin > txmax or tymin > tymax:
            continue
        xs = xmin + (np.arange(txmin, txmax + 1) + 0.5) * px
        ys = ymin + (np.arange(tymin, tymax + 1) + 0.5) * py
        XX, YY = np.meshgrid(xs, ys)
        # barycentric coordinates
        x1, y1 = t[0, 0], t[0, 1]
        x2, y2 = t[1, 0], t[1, 1]
        x3, y3 = t[2, 0], t[2, 1]
        det = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
        if abs(det) < 1e-15:
            continue
        l1 = ((y2 - y3) * (XX - x3) + (x3 - x2) * (YY - y3)) / det
        l2 = ((y3 - y1) * (XX - x3) + (x1 - x3) * (YY - y3)) / det
        l3 = 1.0 - l1 - l2
        inside = (l1 >= -1e-9) & (l2 >= -1e-9) & (l3 >= -1e-9)
        if not inside.any():
            continue
        zz = l1 * t[0, 2] + l2 * t[1, 2] + l3 * t[2, 2]
        patch = z[tymin:tymax + 1, txmin:txmax + 1]
        upd = inside & (np.isnan(patch) | (zz > patch))
        patch[upd] = zz[upd]
        z[tymin:tymax + 1, txmin:txmax + 1] = patch

    return ReferenceSurface(height=z, bounds=(xmin, xmax, ymin, ymax),
                            pixel_size=(px, py))


@dataclass
class ReferenceSurface:
    height: np.ndarray                       # (H,W), NaN where uncovered
    bounds: Tuple[float, float, float, float]
    pixel_size: Tuple[float, float]          # (px, py) CAD units / pixel

    @property
    def footprint(self) -> np.ndarray:
        return ~np.isnan(self.height)


# ---------------------------------------------------------------- tolerance

@dataclass
class ToleranceBand:
    """Symmetric |deviation| thresholds, in the comparison's z units."""
    warn: float
    fail: float

    def classify(self, dev: np.ndarray) -> np.ndarray:
        """0 = in tolerance, 1 = warn, 2 = fail (NaN-safe)."""
        a = np.abs(dev)
        cls = np.zeros(dev.shape, dtype=np.uint8)
        cls[a > self.warn] = 1
        cls[a > self.fail] = 2
        cls[np.isnan(dev)] = 0
        return cls


@dataclass
class GoldenComparisonResult:
    deviation: np.ndarray            # signed measured - reference, NaN outside
    classes: np.ndarray              # 0 in-tol / 1 warn / 2 fail
    pct_in_tolerance: float
    pct_warn: float
    pct_fail: float
    max_over: float                  # most-positive deviation (extra material)
    max_under: float                 # most-negative deviation (missing material)
    worst_px: Tuple[int, int]        # location of the largest |deviation|
    xy_boundary_offset_px: float     # mean footprint edge offset (NaN if skipped)
    passed: bool
    notes: str = ""


def compare_to_golden(measured: np.ndarray,
                      reference: ReferenceSurface,
                      tol_z: ToleranceBand,
                      scale: float = 1.0,
                      offset: Optional[float] = None,
                      tol_xy_px: Optional[float] = None,
                      measured_footprint: Optional[np.ndarray] = None,
                      max_fail_px: int = 0) -> GoldenComparisonResult:
    """
    Compare a measured height map against a rasterized golden surface.

    measured  : (H,W) array on the SAME grid as reference.height.
                Apply your own resampling/alignment first; v1 assumes a
                fixtured part (known pose), which is the realistic
                bench/demo configuration. ICP alignment is a later layer.
    scale     : multiply measured by this to reach reference z units.
    offset    : z offset to subtract after scaling. Default: robust fit
                (median of measured-reference over the footprint), which
                removes the arbitrary phase/height datum.
    tol_xy_px : if given (and a measured footprint is available or
                derivable), check the part silhouette boundary against
                the CAD footprint and fail if the mean edge offset
                exceeds this many pixels.
    max_fail_px: number of 'fail'-class pixels tolerated before the
                overall verdict is False (0 = any fail pixel fails).
    """
    ref = reference.height
    if measured.shape != ref.shape:
        raise ValueError(f"measured {measured.shape} vs reference "
                         f"{ref.shape}: resample to the reference grid first")
    m = measured.astype(np.float64) * scale
    fp = reference.footprint
    dev = np.where(fp, m - ref, np.nan)
    if offset is None:
        offset = float(np.nanmedian(dev))
    dev = dev - offset

    classes = tol_z.classify(dev)
    n = int(fp.sum())
    n_warn = int(((classes == 1) & fp).sum())
    n_fail = int(((classes == 2) & fp).sum())
    finite = np.where(np.isnan(dev), 0.0, dev)
    worst_flat = int(np.nanargmax(np.abs(np.where(fp, dev, 0.0))))
    worst_px = np.unravel_index(worst_flat, dev.shape)

    xy_off = float("nan")
    xy_ok = True
    if tol_xy_px is not None:
        mfp = measured_footprint
        if mfp is None:
            mfp = np.isfinite(measured) & (measured > np.nanmin(measured))
        xy_off = _mean_boundary_offset(mfp, fp)
        xy_ok = xy_off <= tol_xy_px

    passed = (n_fail <= max_fail_px) and xy_ok
    return GoldenComparisonResult(
        deviation=dev,
        classes=classes,
        pct_in_tolerance=100.0 * (n - n_warn - n_fail) / max(n, 1),
        pct_warn=100.0 * n_warn / max(n, 1),
        pct_fail=100.0 * n_fail / max(n, 1),
        max_over=float(np.nanmax(dev)) if n else float("nan"),
        max_under=float(np.nanmin(dev)) if n else float("nan"),
        worst_px=(int(worst_px[0]), int(worst_px[1])),
        xy_boundary_offset_px=xy_off,
        passed=passed,
        notes=f"datum offset removed: {offset:.6g}; z units = CAD units "
              f"after scale={scale}",
    )


def _mean_boundary_offset(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Mean distance (px) from A's boundary pixels to B's boundary."""
    ea, eb = _edges(mask_a), _edges(mask_b)
    if not ea.any() or not eb.any():
        return float("inf")
    ay, ax = np.nonzero(ea)
    by, bx = np.nonzero(eb)
    # subsample for speed on big masks
    if len(ay) > 2000:
        sel = np.linspace(0, len(ay) - 1, 2000).astype(int)
        ay, ax = ay[sel], ax[sel]
    d2 = (ay[:, None] - by[None, :]) ** 2 + (ax[:, None] - bx[None, :]) ** 2
    return float(np.sqrt(d2.min(axis=1)).mean())


def _edges(mask: np.ndarray) -> np.ndarray:
    m = mask.astype(bool)
    interior = m.copy()
    interior[1:, :] &= m[:-1, :]; interior[:-1, :] &= m[1:, :]
    interior[:, 1:] &= m[:, :-1]; interior[:, :-1] &= m[:, 1:]
    return m & ~interior


# ------------------------------------------------------- synthetic goldens

def make_box_stl(path: str, size=(40.0, 30.0, 10.0)) -> None:
    """Axis-aligned box golden part for tests and dry runs."""
    sx, sy, sz = size
    v = np.array([[0, 0, 0], [sx, 0, 0], [sx, sy, 0], [0, sy, 0],
                  [0, 0, sz], [sx, 0, sz], [sx, sy, sz], [0, sy, sz]],
                 dtype=np.float64)
    f = np.array([
        [0, 2, 1], [0, 3, 2],            # bottom
        [4, 5, 6], [4, 6, 7],            # top
        [0, 1, 5], [0, 5, 4],            # sides
        [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6],
        [3, 0, 4], [3, 4, 7],
    ], dtype=np.int64)
    save_stl(path, v, f)

"""
Recipe geometry on one metric scan: the scan view, the part frame and the zones.

ScanView   the metric cloud (xyz mm, camera frame: x right, y down, z forward), its valid
           mask, the decoded DMD (col, row) per camera px, the flood pair, the camera
           intrinsics of the stereo calibration (to project mm back to camera px).
PartFrame  the reference plane (the fixture: the dominant plane of the scan, or a plane
           fitted in a datum zone), its normal n toward the camera, the origin (where the
           camera's optical axis, or origin_px, meets the plane) and in-plane axes
           e1 (~ camera right), e2 (~ camera down): part-frame mm (u, v) and the height h
           (+ toward the camera) of every measured point.
ZoneView   one zone resolved on the scan: its camera px (mask), the measured px in it, its
           outline in camera px (overlay, feedback) and, for projector-px zones, in DMD px;
           `cover`: the zone's whole extent in its own units and which of it was measured
           (the coverage a check needs before it may call something absent or defect-free).

Zone units (said in the recipe, never guessed):
  camera_px     the shape drawn in the camera image
  projector_px  native DMD (col, row) px: camera px whose decoded correspondence falls in it
                (needs both axes decoded); grid-cells are DMD cells too, and so is a zone
                painted on the part (shape "zoneset": engine/zones - the set painted at the
                scan's geometry, with the same geometry-bound rig settings, or a ZoneError)
  part_mm       mm in the part frame: measured points whose (u, v) falls in it (3-D: a part
                standing proud of the fixture is still in its zone)

Coverage: the zone's whole extent as cells in ITS units (camera px; native DMD px binned to
the camera's sampling; part-frame mm at half the camera's sampling) and the cells a measured
camera px of the zone falls in. Whatever of the zone is off the camera image, occluded, in
shadow, undecoded or simply not measured is a cell no measured px falls in (unseen),
wherever it is and at whatever depth: an unmeasured part never enters a part_mm or
projector_px zone's points, so only its unseen cells show that it may be there. Unknown
(None) when the camera's sampling of the zone cannot be estimated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

DMD_SHAPE = (1140, 912)                 # rows, cols (native)
CIRCLE_PTS = 96
MM_RASTER = 0.1                         # part_mm polygon raster, mm
PLANE_SAMPLE = 200000                   # points the reference plane is fitted on (random)
MAX_COVER_CELLS = 4_000_000             # coverage grid cells (a huge zone gets coarser cells)
MIN_STEP_PAIRS = 20                     # adjacent measured px pairs to estimate the sampling


class ZoneError(ValueError):
    pass


@dataclass
class ScanView:
    xyz: np.ndarray                      # (H, W, 3) mm, NaN where not measured
    valid: np.ndarray                    # (H, W) bool
    K: np.ndarray                        # camera intrinsics (stereo calibration)
    dist: np.ndarray
    corr_x: Optional[np.ndarray] = None  # (H, W) native DMD column per camera px (NaN = none)
    corr_y: Optional[np.ndarray] = None  # (H, W) native DMD row
    flood_white: Optional[np.ndarray] = None
    flood_black: Optional[np.ndarray] = None
    scan_id: str = ""
    calibration_id: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple:
        return self.valid.shape

    @property
    def flood(self) -> Optional[np.ndarray]:
        """white - black (the lit scene without ambient), or None."""
        if self.flood_white is None:
            return None
        b = 0.0 if self.flood_black is None else self.flood_black
        return np.asarray(self.flood_white, np.float32) - np.asarray(b, np.float32)

    @property
    def has_xy(self) -> bool:
        return self.corr_x is not None and self.corr_y is not None

    def project(self, X: np.ndarray) -> np.ndarray:
        """Camera-frame points (..., 3) mm -> distorted camera px (..., 2)."""
        from ..capture.mock_stereo import distort_project
        return distort_project(np.asarray(X, np.float64), self.K, self.dist)

    def ray(self, px) -> np.ndarray:
        """Camera px (x, y) -> unit-z ray direction (3,)."""
        from ..capture.mock_stereo import undistort_normalized
        xy = undistort_normalized(np.asarray(px, np.float64).reshape(1, 2), self.K, self.dist)[0]
        return np.array([xy[0], xy[1], 1.0])

    @classmethod
    def from_scan(cls, scan, cal) -> "ScanView":
        """A stored MetricScan (+ its stereo calibration) -> ScanView."""
        cloud = scan.cloud
        xyz = np.asarray(cloud.xyz, np.float64)
        valid = np.asarray(cloud.mask, bool) & np.isfinite(xyz).all(-1)
        cx = cy = None
        try:
            fld = scan.field
            cx = None if fld.corr_x is None else np.asarray(fld.corr_x, np.float64)
            cy = None if fld.corr_y is None else np.asarray(fld.corr_y, np.float64)
        except (OSError, ValueError, KeyError):
            pass
        white = black = None
        f = scan.path / "flood.npz" if scan.path is not None else None
        if f is not None and f.is_file():
            with np.load(f) as z:
                white = np.asarray(z["white"], np.float32)
                black = np.asarray(z["black"], np.float32) if "black" in z.files else None
        return cls(xyz=xyz, valid=valid, K=np.asarray(cal.K_cam, np.float64),
                   dist=np.asarray(cal.dist_cam, np.float64), corr_x=cx, corr_y=cy,
                   flood_white=white, flood_black=black, scan_id=scan.id,
                   calibration_id=scan.calibration_id, meta=dict(scan.meta or {}))


@dataclass
class PartFrame:
    origin: np.ndarray
    e1: np.ndarray
    e2: np.ndarray
    n: np.ndarray                        # toward the camera
    plane_rms_mm: float
    n_plane_points: int
    reference: str
    uvh: np.ndarray                      # (H, W, 3): u, v, h per camera px (NaN off the cloud)

    def to_uvh(self, X: np.ndarray) -> np.ndarray:
        W = np.asarray(X, np.float64) - self.origin
        return np.stack([W @ self.e1, W @ self.e2, W @ self.n], -1)

    def to_xyz(self, u, v, h=0.0) -> np.ndarray:
        u, v, h = np.broadcast_arrays(np.asarray(u, float), np.asarray(v, float),
                                      np.asarray(h, float))
        return (self.origin + u[..., None] * self.e1 + v[..., None] * self.e2
                + h[..., None] * self.n)

    def summary(self) -> dict:
        return {"reference": self.reference, "origin_mm": [round(float(x), 3) for x in self.origin],
                "normal": [round(float(x), 5) for x in self.n],
                "e1": [round(float(x), 5) for x in self.e1],
                "e2": [round(float(x), 5) for x in self.e2],
                "plane_rms_mm": round(self.plane_rms_mm, 4), "n_plane_points": self.n_plane_points,
                "units": "mm, camera frame; u along e1 (~ right), v along e2 (~ down), h along "
                         "n (+ toward the camera)"}


def face_axes(n: np.ndarray) -> tuple:
    from ..measure.endface.locate import face_axes as _fa
    return _fa(np.asarray(n, np.float64))


def part_frame(sv: ScanView, reference="dominant_plane", plane_band_mm: float = 1.0,
               origin_px=None, datum_mask: Optional[np.ndarray] = None) -> PartFrame:
    """The reference plane + frame. reference 'dominant_plane': RANSAC over the whole cloud
    (the fixture / background); a datum zone: a robust plane fitted on that zone's points."""
    from ..measure import metric
    from ..measure.endface.profile import robust_plane
    P = sv.xyz[sv.valid]
    if len(P) < 100:
        raise ZoneError(f"only {len(P)} measured points in the scan: no reference plane")
    if datum_mask is not None:
        Q = sv.xyz[sv.valid & datum_mask]
        if len(Q) < 50:
            raise ZoneError(f"the datum zone has only {len(Q)} measured points: it must see the "
                            "fixture surface")
        c, n = robust_plane(Q)
        d = (Q - c) @ n
        rms, npts, ref = float(np.sqrt(np.mean(d * d))), len(Q), "datum zone"
        point = c
    else:
        if len(P) > PLANE_SAMPLE:                # plenty for a plane; fast on a full frame
            P = P[np.random.default_rng(0).choice(len(P), PLANE_SAMPLE, replace=False)]
        fit, inl = metric.ransac_plane(P, float(plane_band_mm))
        point, n = np.asarray(fit.point), np.asarray(fit.normal)
        rms, npts, ref = float(fit.rms), int(inl.sum()), "dominant_plane"
    if n @ point > 0:                    # toward the camera (the camera is at the origin)
        n = -n
    ray = sv.ray(origin_px) if origin_px is not None else np.array([0.0, 0.0, 1.0])
    den = float(ray @ n)
    if abs(den) < 1e-6:
        raise ZoneError("the reference plane is seen edge-on: no part frame")
    origin = ray * float(point @ n) / den
    e1, e2 = face_axes(n)
    if e2[1] < 0:                         # keep v ~ camera down
        e2 = -e2
    W = sv.xyz - origin
    uvh = np.stack([W @ e1, W @ e2, W @ n], -1)
    uvh[~sv.valid] = np.nan
    return PartFrame(origin=origin, e1=e1, e2=e2, n=n, plane_rms_mm=rms, n_plane_points=npts,
                     reference=ref, uvh=uvh)


# ------------------------------------------------------------------ zones

@dataclass
class ZoneView:
    id: str
    shape: str
    units: str
    coords: object
    mask: np.ndarray                     # (H, W) camera px in the zone
    valid: np.ndarray                    # mask & measured
    outline_px: list                     # [(N, 2) camera px polygons]
    dmd_mask: Optional[np.ndarray] = None   # projector_px zones: the DMD region
    label: str = ""
    datum: bool = False
    feedback: bool = True
    warnings: list = field(default_factory=list)
    cover: Optional["Coverage"] = None     # the zone's extent and what of it was measured

    @property
    def n_px(self) -> int:
        return int(self.mask.sum())

    @property
    def n_valid(self) -> int:
        return int(self.valid.sum())

    @property
    def n_unseen(self) -> Optional[int]:
        """~Camera px the zone's unseen extent would cover (None: coverage unknown)."""
        if self.cover is None:
            return None
        return int(round(int(self.cover.unseen.sum()) * self.cover.px_per_cell ** 2))

    def coverage_pct(self, sel: Optional[np.ndarray] = None) -> Optional[float]:
        """% of the zone measured, over its whole extent (off-image, occluded, shadowed and
        undecoded parts count as unmeasured). sel: a (H, W) bool subset of the zone's
        measured px (e.g. those above min_height_mm): cells only the OTHER measured px fall
        in are dropped from the total (measured, but not what is inspected), so sel cells /
        (sel cells + unseen cells). None when the coverage is unknown (a check must then not
        call anything absent or clean)."""
        c = self.cover
        if c is None:
            return None
        q = int(c.seen.sum()) if sel is None else int(c.hits(sel & self.valid).sum())
        tot = q + int(c.unseen.sum())
        return 100.0 * q / tot if tot else 0.0

    def centroid_px(self) -> list:
        ys, xs = np.nonzero(self.valid if self.valid.any() else self.mask)
        if not len(xs):
            pts = np.concatenate(self.outline_px) if self.outline_px else np.zeros((1, 2))
            return [float(pts[:, 0].mean()), float(pts[:, 1].mean())]
        return [float(xs.mean()), float(ys.mean())]


def shape_points(shape: str, coords, n: int = CIRCLE_PTS) -> np.ndarray:
    """rect / circle / polygon -> (N, 2) closed outline points (in the zone's own units)."""
    if shape == "rect":
        x0, y0, x1, y1 = coords
        return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float64)
    if shape == "circle":
        (cx, cy), r = coords["center"], coords["radius"]
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.c_[cx + r * np.cos(t), cy + r * np.sin(t)]
    if shape == "polygon":
        return np.asarray(coords, np.float64)
    raise ZoneError(f"no outline for shape {shape}")


def _raster(poly: np.ndarray, shape: tuple, scale: float = 1.0, offset=(0.0, 0.0)) -> np.ndarray:
    """Polygon (N, 2) -> bool mask of `shape`, sub-pixel vertices (cv2 shift)."""
    import cv2
    m = np.zeros(shape, np.uint8)
    pts = np.round((np.asarray(poly) - np.asarray(offset)) * scale * 16).astype(np.int32)
    cv2.fillPoly(m, [pts], 1, lineType=cv2.LINE_8, shift=4)
    return m.astype(bool)


def grid_mask(cells: str) -> np.ndarray:
    """'F10:J12' / 'F10:J12@96' -> the DMD mask of those grid cells (default 96 px cells)."""
    from ..capture import zones as zones_mod
    text, spacing = zones_mod._split_spacing(cells)
    grid = zones_mod.grid_spec(spacing or zones_mod.DEFAULT_SPACING)
    return zones_mod.projector_mask(zones_mod.parse_zone(text, grid), grid)


def dmd_zone_mask(z, meta: Optional[dict] = None) -> np.ndarray:
    """The DMD mask of a projector_px zone. meta: the scan's meta (its geometry_id +
    geometry_fp), which a painted zone (shape "zoneset") is bound to."""
    if z.shape == "grid-cells":
        return grid_mask(z.coords)
    if z.shape == "zoneset":
        from ..zones import use as zone_use
        try:
            return zone_use.recipe_mask(z, meta)
        except ValueError as e:              # another geometry / rig settings, unbound, gone
            raise ZoneError(str(e)) from None
    return _raster(shape_points(z.shape, z.coords), DMD_SHAPE)


def _from_dmd(sv: ScanView, dm: np.ndarray) -> np.ndarray:
    if not sv.has_xy:
        raise ZoneError("a projector_px / grid-cells zone needs both DMD axes decoded (scan "
                        "axis xy): this scan decoded columns only")
    ok = np.isfinite(sv.corr_x) & np.isfinite(sv.corr_y)
    c = np.clip(np.round(np.where(ok, sv.corr_x, 0)), 0, DMD_SHAPE[1] - 1).astype(np.intp)
    r = np.clip(np.round(np.where(ok, sv.corr_y, 0)), 0, DMD_SHAPE[0] - 1).astype(np.intp)
    return ok & dm[r, c]


def _densify(poly: np.ndarray, step: float = 1.0, max_per_edge: int = 200) -> np.ndarray:
    """Closed polygon -> the same outline with a vertex every `step` (its units) along each
    edge, so a projection through lens distortion keeps the edges' true (curved) course."""
    poly = np.asarray(poly, np.float64)
    out = []
    for a, b in zip(poly, np.roll(poly, -1, 0)):
        n = int(min(max(np.ceil(np.hypot(*(b - a)) / step), 1), max_per_edge))
        t = np.arange(n)[:, None] / n
        out.append(a + (b - a) * t)
    return np.concatenate(out)


@dataclass
class Coverage:
    """A zone's whole extent as a grid of cells in its own units and the cells a measured
    camera px of the zone falls in (see the module doc)."""
    zmask: np.ndarray                    # (gh, gw) the zone's cells
    seen: np.ndarray                     # (gh, gw) zone cells a measured px of the zone hits
    cell: float                          # cell edge in the zone's units (px, DMD px, mm)
    px_per_cell: float                   # ~camera px per cell edge (sizes in mm, n_unseen)
    to_cell: object = None               # (ys, xs) camera px -> (gi, gj) cell indices
    close: bool = False                  # sampled grids: close the 1-cell sampling gaps

    @property
    def unseen(self) -> np.ndarray:
        return self.zmask & ~self.seen

    def hits(self, sel: np.ndarray) -> np.ndarray:
        """The zone cells the camera px of sel ((H, W) bool) fall in."""
        g = np.zeros(self.zmask.shape, np.uint8)
        ys, xs = np.nonzero(sel)
        if len(xs):
            gi, gj = self.to_cell(ys, xs)
            ok = (gi >= 0) & (gj >= 0) & (gi < g.shape[0]) & (gj < g.shape[1])
            g[gi[ok], gj[ok]] = 1
            if self.close:
                import cv2
                g = cv2.morphologyEx(g, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        return g.astype(bool) & self.zmask


def _steps(A: np.ndarray, ok: np.ndarray) -> Optional[tuple]:
    """Median step of A (H, W, k) between horizontally / vertically adjacent px that are both
    ok -> (along x, along y), or None with fewer than MIN_STEP_PAIRS pairs either way."""
    out = []
    for d, m in ((A[:, 1:] - A[:, :-1], ok[:, 1:] & ok[:, :-1]),
                 (A[1:] - A[:-1], ok[1:] & ok[:-1])):
        n = np.linalg.norm(d[m], axis=-1)
        n = n[np.isfinite(n)]
        if n.size < MIN_STEP_PAIRS:
            return None
        out.append(float(np.median(n)))
    return tuple(out) if min(out) > 0 else None


def _around(mask: np.ndarray, pad: int = 2):
    """The bbox slices of mask (padded), or None when it is empty."""
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    H, W = mask.shape
    return (slice(max(ys.min() - pad, 0), min(ys.max() + pad + 1, H)),
            slice(max(xs.min() - pad, 0), min(xs.max() + pad + 1, W)))


def _poly_area(poly: np.ndarray) -> float:
    x, y = np.asarray(poly, np.float64).T
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _poly_grid(poly: np.ndarray, cell: float, pad: int = 2) -> tuple:
    """A cell grid over a polygon (zone units) -> (zmask, origin, cell); cells get coarser
    for a zone that would need more than MAX_COVER_CELLS."""
    poly = np.asarray(poly, np.float64)
    lo, hi = poly.min(0), poly.max(0)
    span = np.maximum(hi - lo, cell) + cell
    cell = max(float(cell), float(np.sqrt(span[0] * span[1] / MAX_COVER_CELLS)))
    origin = (np.floor(lo / cell) - pad) * cell        # on the cell lattice (camera: whole px)
    shape = (int(np.ceil(span[1] / cell)) + 2 * pad + 1,
             int(np.ceil(span[0] / cell)) + 2 * pad + 1)
    return _raster(poly, shape, 1.0 / cell, origin), origin, cell


def _cover_camera(poly: np.ndarray, valid: np.ndarray) -> Coverage:
    """camera_px: cells are camera px (the zone's part off the image included); valid: the
    scan's measured px."""
    zmask, (ox, oy), cell = _poly_grid(poly, 1.0)

    def to_cell(ys, xs):
        return (np.round((ys - oy) / cell).astype(np.intp),
                np.round((xs - ox) / cell).astype(np.intp))
    c = Coverage(zmask=zmask, seen=zmask, cell=cell, px_per_cell=cell, to_cell=to_cell,
                 close=cell > 1.0)
    c.seen = c.hits(valid)
    return c


def _cover_part(poly: np.ndarray, sv: ScanView, frame: PartFrame, zone_valid: np.ndarray,
                hz: float) -> Optional[Coverage]:
    """part_mm: cells of half the camera's (u, v) sampling at the zone (the median step
    between adjacent measured px around the zone's; else the zone outline projected at
    height hz); a cell is seen when any measured point's (u, v) falls in it."""
    uv = frame.uvh[..., :2]
    ok = sv.valid & np.isfinite(uv).all(-1)
    st = None
    box = _around(zone_valid, 3)
    if box is not None:
        st = _steps(uv[box], ok[box])
    if st is None:                                   # too little measured: the outline
        dense = _densify(poly, 1.0)
        fpx = sv.project(frame.to_xyz(dense[:, 0], dense[:, 1], hz))
        a_px, a_mm = (_poly_area(fpx), _poly_area(poly)) if np.isfinite(fpx).all() else (0, 0)
        if not (a_px > 0 and a_mm > 0):
            return None
        s = float(np.sqrt(a_mm / a_px))
        st = (s, s)
    zmask, (ou, ov), cell = _poly_grid(poly, max(st) / 2.0)

    def to_cell(ys, xs):
        q = uv[ys, xs]
        return (np.round((q[:, 1] - ov) / cell).astype(np.intp),
                np.round((q[:, 0] - ou) / cell).astype(np.intp))
    c = Coverage(zmask=zmask, seen=zmask, cell=cell,
                 px_per_cell=cell / float(np.sqrt(st[0] * st[1])), to_cell=to_cell, close=True)
    c.seen = c.hits(ok)
    return c


def _cover_dmd(sv: ScanView, dm: np.ndarray) -> Optional[Coverage]:
    """projector_px: cells are native DMD px, binned so that adjacent decoded camera px are
    at most ~2 cells apart (the camera's sampling of the DMD near the zone); a cell is seen
    when a measured, decoded camera px falls in it."""
    rows, cols = np.nonzero(dm)
    if not len(rows):
        return None
    ok = np.isfinite(sv.corr_x) & np.isfinite(sv.corr_y)
    r0, r1, c0, c1 = int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())
    mr, mc = max(r1 - r0, 40), max(c1 - c0, 40)              # a zone's size around it
    near = ok & (sv.corr_y >= r0 - mr) & (sv.corr_y <= r1 + mr) & \
        (sv.corr_x >= c0 - mc) & (sv.corr_x <= c1 + mc)
    C = np.stack([sv.corr_x, sv.corr_y], -1)
    st = None
    for m in (near, ok):
        box = _around(m, 1)
        if box is not None:
            st = _steps(C[box], m[box])
        if st is not None:
            break
    if st is None:
        return None
    b = max(1, int(np.ceil(max(st) / 2.0)))
    while ((r1 - r0) / b + 1) * ((c1 - c0) / b + 1) > MAX_COVER_CELLS:
        b += 1
    pad = 2
    oy, ox = r0 - pad * b, c0 - pad * b
    gh, gw = (r1 - r0) // b + 2 * pad + 1, (c1 - c0) // b + 2 * pad + 1
    ry = oy + np.arange(gh) * b + b // 2                     # each cell's centre DMD px
    rx = ox + np.arange(gw) * b + b // 2
    iy, ix = (ry >= 0) & (ry < dm.shape[0]), (rx >= 0) & (rx < dm.shape[1])
    zmask = np.zeros((gh, gw), bool)
    zmask[np.ix_(iy, ix)] = dm[np.ix_(ry[iy], rx[ix])]

    def to_cell(ys, xs):
        return ((np.round(sv.corr_y[ys, xs]).astype(np.intp) - oy) // b,
                (np.round(sv.corr_x[ys, xs]).astype(np.intp) - ox) // b)
    c = Coverage(zmask=zmask, seen=zmask, cell=float(b),
                 px_per_cell=b / float(np.sqrt(st[0] * st[1])), to_cell=to_cell, close=True)
    c.seen = c.hits(sv.valid & ok)
    return c


def _contours(mask: np.ndarray) -> list:
    import cv2
    m = mask.astype(np.uint8)
    if not m.any():
        return []
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cs = [c for c in cs if cv2.contourArea(c) >= 0.02 * m.sum() and len(c) >= 3]
    return [c.reshape(-1, 2).astype(np.float64) for c in cs]


def resolve_zone(z, sv: ScanView, frame: PartFrame) -> ZoneView:
    """A ZoneSpec on this scan -> ZoneView (mask, measured px, camera outline)."""
    H, W = sv.shape
    dm = None
    warnings = []
    if z.units == "camera_px":
        poly = shape_points(z.shape, z.coords)
        mask = _raster(poly, (H, W))
        outline = [poly]
        cover = _cover_camera(poly, sv.valid)
    elif z.units == "projector_px":
        dm = dmd_zone_mask(z, sv.meta)
        if z.shape == "zoneset":
            from ..zones import use as zone_use
            warnings += zone_use.recipe_warnings(z, sv.meta)
        mask = _from_dmd(sv, dm)
        outline = _contours(mask)
        cover = _cover_dmd(sv, dm)
        if cover is None:
            warnings.append(f"zone {z.id}: too few decoded px around it to tell the camera's "
                            "sampling of its DMD px: its coverage is unknown")
    else:                                                    # part_mm
        poly = shape_points(z.shape, z.coords)
        uv = frame.uvh[..., :2]
        fin = sv.valid & np.isfinite(uv).all(-1)
        lo, hi = poly.min(0), poly.max(0)
        cand = fin & (uv[..., 0] >= lo[0]) & (uv[..., 0] <= hi[0]) & \
            (uv[..., 1] >= lo[1]) & (uv[..., 1] <= hi[1])
        mask = np.zeros((H, W), bool)
        if cand.any():
            ys, xs = np.nonzero(cand)
            q = uv[ys, xs]
            if z.shape == "rect":
                inside = np.ones(len(q), bool)
            elif z.shape == "circle":
                (cx, cy), r = z.coords["center"], z.coords["radius"]
                inside = np.hypot(q[:, 0] - cx, q[:, 1] - cy) <= r
            else:
                R = _raster(poly, (int(np.ceil((hi[1] - lo[1]) / MM_RASTER)) + 2,
                                   int(np.ceil((hi[0] - lo[0]) / MM_RASTER)) + 2),
                            1.0 / MM_RASTER, lo)
                iu = np.clip(((q[:, 0] - lo[0]) / MM_RASTER).astype(int), 0, R.shape[1] - 1)
                iv = np.clip(((q[:, 1] - lo[1]) / MM_RASTER).astype(int), 0, R.shape[0] - 1)
                inside = R[iv, iu]
            mask[ys[inside], xs[inside]] = True
        h = frame.uvh[..., 2][mask]
        hz = float(np.median(h)) if h.size else 0.0
        outline = [sv.project(frame.to_xyz(poly[:, 0], poly[:, 1], hz))]
        cover = _cover_part(poly, sv, frame, mask & sv.valid, hz)
        if cover is None:
            warnings.append(f"zone {z.id}: too little measured and its outline does not "
                            "project into the camera: its coverage is unknown")
        if not mask.any():
            warnings.append(f"zone {z.id}: no measured point falls in it (part-frame mm)")
    return ZoneView(id=z.id, shape=z.shape, units=z.units, coords=z.coords, mask=mask,
                    valid=mask & sv.valid, outline_px=outline, dmd_mask=dm, label=z.label,
                    datum=z.datum, feedback=z.feedback, warnings=warnings, cover=cover)


def datum_mask(recipe, sv: ScanView) -> Optional[np.ndarray]:
    """The camera mask of the reference zone (part.frame.reference {"zone": id}), or None."""
    ref = (recipe.part.get("frame") or {}).get("reference")
    if not isinstance(ref, dict):
        return None
    z = recipe.zone(ref["zone"])
    if z.units == "camera_px":
        return _raster(shape_points(z.shape, z.coords), sv.shape)
    return _from_dmd(sv, dmd_zone_mask(z, sv.meta))


def frame_for(recipe, sv: ScanView) -> PartFrame:
    f = recipe.part.get("frame") or {}
    return part_frame(sv, f.get("reference", "dominant_plane"), float(f.get("plane_band_mm", 1.0)),
                      f.get("origin_px"), datum_mask(recipe, sv))

"""
Projector map — full 2-axis camera<->projector correspondence.

The 3D path decodes one axis (projector columns, corr_x). Zones and grids
need both: capture() runs the hybrid basis twice — vertical stripes
(axis "x" -> projector column) and horizontal stripes (axis "y" ->
projector row) — on the current scene and stores a ProjectorMap. With it,
any camera pixel knows which DMD mirror lights it, so a projected grid
cell "C4" maps to an exact camera region, and a camera region maps back
to the DMD pixels that light it.

The map is of the SCENE, not just the rig: a part shifts the columns it
is lit by (that shift is the relief). Capture it with the part in place
when the camera-side cut has to follow the part exactly; a map of the
empty base is still right everywhere the part is not.

Coordinates are pattern space (what Projector.project receives, rows x
cols = dmd_shape), because that is what the decoded stripes encode —
rotate90 and HDMI polarity are handled by the projector chain / decode.

Storage: library/maps/<geometry_id>/<ts>_<label>/map.npz (corr_x, corr_y, mask), meta.json, preview.png
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from .. import rules
from ..library.store import safe_label

DMD_SHAPE = (1140, 912)                 # rows, cols (pattern space)
MAP_UNITS = {"corr_x": "projector px (DMD column, pattern space)",
             "corr_y": "projector px (DMD row, pattern space)",
             "valid_pct": "% of camera px mapped in BOTH axes"}
LOOKUP_RADIUS_PX = 1        # to_camera: camera px mapping within ~1 DMD px of the query
JUMP_PX = 3.0               # neighbours further apart than this straddle a depth edge
LOW_VALID_PCT = 5.0
PREVIEW_GRID_PX = 96        # DMD px between the reference lines drawn on preview.png


@dataclass
class ProjectorMap:
    id: str
    geometry_id: str
    path: Optional[Path]
    created: str
    corr_x: np.ndarray = field(repr=False)     # camera px -> projector column (NaN invalid)
    corr_y: np.ndarray = field(repr=False)     # camera px -> projector row (NaN invalid)
    mask: np.ndarray = field(repr=False)       # valid in both axes
    dmd_shape: tuple = (1140, 912)             # (rows, cols)
    warnings: list = field(default_factory=list)
    label: str = ""
    valid_pct: float = 0.0                     # % of camera px mapped in both axes
    units: dict = field(default_factory=lambda: dict(MAP_UNITS))
    meta: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self.corr_x = np.asarray(self.corr_x, np.float32)
        self.corr_y = np.asarray(self.corr_y, np.float32)
        shape = np.shape(self.mask)
        for name in ("corr_x", "corr_y"):
            if getattr(self, name).shape != shape:
                raise ValueError(f"ProjectorMap: {name} shape {getattr(self, name).shape} "
                                 f"!= mask shape {shape}")
        # a pixel is mapped only if BOTH coordinates are real numbers
        self.mask = (np.asarray(self.mask, bool) & np.isfinite(self.corr_x)
                     & np.isfinite(self.corr_y))
        self.dmd_shape = tuple(int(v) for v in self.dmd_shape)

    @property
    def camera_shape(self) -> tuple:
        return self.mask.shape

    # ------------------------------------------------ projector -> camera
    def camera_mask_for_projector_rect(self, x0, y0, x1, y1) -> np.ndarray:
        """Camera pixels lit by DMD rectangle [x0,x1) x [y0,y1) (projector px)."""
        x0, x1 = sorted((float(x0), float(x1)))
        y0, y1 = sorted((float(y0), float(y1)))
        cx, cy = self.corr_x, self.corr_y
        with np.errstate(invalid="ignore"):
            return self.mask & (cx >= x0) & (cx < x1) & (cy >= y0) & (cy < y1)

    def camera_mask_for_projector_mask(self, proj_mask: np.ndarray) -> np.ndarray:
        """Camera pixels lit by any DMD pixel of a bool dmd_shape mask. The
        union-of-rectangles version of camera_mask_for_projector_rect: a
        camera px belongs to the DMD pixel floor(corr) (the one whose
        [c, c+1) x [r, r+1) square holds its correspondence)."""
        pm = np.asarray(proj_mask, bool)
        if pm.shape != self.dmd_shape:
            raise ValueError(f"projector mask shape {pm.shape} != DMD shape {self.dmd_shape}")
        out = np.zeros(self.camera_shape, bool)
        rows, cols = self.dmd_index()
        out[self.mask] = pm[rows, cols]
        return out

    def dmd_index(self):
        """(rows, cols) of the DMD pixel lighting each mapped camera px, in
        camera-mask order (image[self.mask] lines up with them)."""
        H, W = self.dmd_shape
        cols = np.clip(np.floor(self.corr_x[self.mask]), 0, W - 1).astype(np.intp)
        rows = np.clip(np.floor(self.corr_y[self.mask]), 0, H - 1).astype(np.intp)
        return rows, cols

    # ------------------------------------------------ camera -> projector
    def projector_mask_for_camera_mask(self, cam_mask: np.ndarray, dilate_px: int = 2) -> np.ndarray:
        """DMD pixels (bool, dmd_shape) that light the given camera pixels.

        Each mapped camera pixel splats onto its nearest mirror; the result
        is dilated by `dilate_px` mirrors because the camera samples the DMD
        unevenly (on the mock bench one camera row spans ~1.3 DMD rows), so
        the raw splat has holes. Unmapped camera pixels contribute nothing."""
        cm = np.asarray(cam_mask, bool)
        if cm.shape != self.camera_shape:
            raise ValueError(f"camera mask shape {cm.shape} != map camera shape "
                             f"{self.camera_shape}")
        d = int(dilate_px)
        if d < 0:
            raise ValueError(f"dilate_px must be >= 0 DMD px, got {dilate_px}")
        H, W = self.dmd_shape
        out = np.zeros(self.dmd_shape, bool)
        sel = cm & self.mask
        if not sel.any():
            return out
        cols = np.clip(np.rint(self.corr_x[sel]), 0, W - 1).astype(np.intp)
        rows = np.clip(np.rint(self.corr_y[sel]), 0, H - 1).astype(np.intp)
        out[rows, cols] = True
        if d > 0:
            import cv2
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1))
            out = cv2.dilate(out.astype(np.uint8), k).astype(bool)
        return out

    def to_projector(self, points) -> np.ndarray:
        """[[x, y], ...] camera px -> [[col, row], ...] projector px (NaN if unknown)."""
        pts, single = _points(points)
        out = _sample(np.stack([self.corr_x, self.corr_y]), self.mask, pts)
        return out[0] if single else out

    def to_camera(self, points) -> np.ndarray:
        """[[col, row], ...] projector px -> [[x, y], ...] camera px (centroid of the
        camera pixels that see it; NaN if not seen)."""
        pts, single = _points(points)
        lut = self._camera_lut()
        out = _sample(lut, np.isfinite(lut[0]), pts)
        return out[0] if single else out

    def _camera_lut(self) -> np.ndarray:
        """(2, rows, cols) camera x, y per DMD pixel: centroid of the camera
        pixels whose correspondence rounds to within LOOKUP_RADIUS_PX of it.
        Built once per map (one bincount pass) and cached on the instance —
        not a dataclass field, so it never reaches asdict / the session log."""
        lut = self.__dict__.get("_lut")
        if lut is not None:
            return lut
        H, W = self.dmd_shape
        ys, xs = np.nonzero(self.mask)
        rows, cols = (np.clip(np.rint(self.corr_y[ys, xs]), 0, H - 1).astype(np.intp),
                      np.clip(np.rint(self.corr_x[ys, xs]), 0, W - 1).astype(np.intp))
        idx = rows * W + cols
        n = np.bincount(idx, minlength=H * W).reshape(H, W).astype(np.float64)
        su = np.bincount(idx, weights=xs.astype(np.float64), minlength=H * W).reshape(H, W)
        sv = np.bincount(idx, weights=ys.astype(np.float64), minlength=H * W).reshape(H, W)
        r = LOOKUP_RADIUS_PX
        n, su, sv = _box_sum(n, r), _box_sum(su, r), _box_sum(sv, r)
        with np.errstate(invalid="ignore", divide="ignore"):
            lut = np.stack([np.where(n > 0.5, su / n, np.nan),
                            np.where(n > 0.5, sv / n, np.nan)]).astype(np.float32)
        self.__dict__["_lut"] = lut
        return lut


# ------------------------------------------------------------- helpers

def _points(points):
    pts = np.asarray(points, dtype=np.float64)
    single = pts.ndim == 1
    pts = pts.reshape(-1, 2) if pts.size else pts.reshape(0, 2)
    if pts.shape[-1] != 2:
        raise ValueError(f"points must be [[x, y], ...] pairs, got shape {np.shape(points)}")
    return pts, single


def _sample(planes: np.ndarray, valid: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Sample a (2, H, W) map at float (x, y) points: bilinear when all four
    neighbours are valid and on one surface, else the nearest valid pixel,
    else NaN. Bilinear across a depth edge would invent a point between the
    part and the base, so a JUMP_PX spread falls back to nearest."""
    _, H, W = planes.shape
    out = np.full((len(pts), 2), np.nan)
    if not len(pts):
        return out
    x, y = pts[:, 0], pts[:, 1]
    inb = np.isfinite(x) & np.isfinite(y) & (x > -0.5) & (x < W - 0.5) & (y > -0.5) & (y < H - 0.5)
    if not inb.any():
        return out
    x, y = x[inb], y[inb]
    xn = np.clip(np.rint(x), 0, W - 1).astype(np.intp)
    yn = np.clip(np.rint(y), 0, H - 1).astype(np.intp)
    near = np.where(valid[yn, xn][:, None], planes[:, yn, xn].T, np.nan)

    x0 = np.clip(np.floor(x), 0, W - 1).astype(np.intp)
    y0 = np.clip(np.floor(y), 0, H - 1).astype(np.intp)
    x1, y1 = np.minimum(x0 + 1, W - 1), np.minimum(y0 + 1, H - 1)
    fx, fy = np.clip(x - x0, 0, 1), np.clip(y - y0, 0, 1)
    corners = ((y0, x0, (1 - fx) * (1 - fy)), (y0, x1, fx * (1 - fy)),
               (y1, x0, (1 - fx) * fy), (y1, x1, fx * fy))
    vals = np.stack([planes[:, r, c] for r, c, _ in corners])        # (4, 2, n)
    ok = np.stack([valid[r, c] for r, c, _ in corners]).all(axis=0)   # (n,)
    wts = np.stack([w for _, _, w in corners])                         # (4, n)
    with np.errstate(invalid="ignore"):
        # NaN (an invalid corner) propagates, so such points take `near`
        spread = (vals.max(axis=0) - vals.min(axis=0)).max(axis=0)
        bil = (vals * wts[:, None, :]).sum(axis=0).T                   # (n, 2)
        use = ok & (spread <= JUMP_PX)
    out[inb] = np.where(use[:, None], bil, near)
    return out


def _box_sum(a: np.ndarray, r: int) -> np.ndarray:
    """Sum over the (2r+1)^2 window around each pixel (zero outside)."""
    k = 2 * r + 1
    c = np.pad(np.pad(a, r).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    return c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]


def write_png(path: Path, img: np.ndarray) -> Optional[Path]:
    """Exclusive-create PNG write (rule 7) — shared by the map and zone previews."""
    import cv2
    rules.ensure_new_path(path)
    ok, buf = cv2.imencode(".png", np.ascontiguousarray(img))
    if not ok:
        return None
    with open(path, "xb") as f:
        f.write(buf.tobytes())
    return path


def dmd_shape_of(preset) -> tuple:
    """(rows, cols) of the preset's display mode — must be the DLP4500's."""
    w, h = preset.projection.size
    if (int(h), int(w)) != DMD_SHAPE:
        raise ValueError(f"projection.size {tuple(preset.projection.size)} is not the DLP4500 "
                         f"HDMI mode {(DMD_SHAPE[1], DMD_SHAPE[0])}; grids, zones and maps are "
                         "drawn in 912x1140 pattern space")
    return DMD_SHAPE


def _basis(preset, axis: str):
    """The preset's hybrid (gray code + phase shift) forced to one axis."""
    from ..bases import hybrid
    try:
        b = hybrid.from_preset_bases(preset.bases, axis=axis)
    except TypeError as e:            # a hybrid without the per-axis override
        raise NotImplementedError(
            "the projector map needs HybridBasis(axis='y') (horizontal stripes -> projector "
            f"rows) via from_preset_bases(..., axis=...): {e}") from None
    except ValueError as e:
        raise ValueError(f"projector map, axis {axis}: {e}") from None
    if getattr(b, "axis", axis) != axis:
        raise NotImplementedError(f"the hybrid basis ignored axis={axis!r}; the projector map "
                                  "needs both x (columns) and y (rows)")
    return b


def _preview(corr_x, corr_y, mask, dmd_shape) -> np.ndarray:
    """Camera-space picture of the map: red = DMD column, green = DMD row,
    white lines every PREVIEW_GRID_PX mirrors (the default grid), black =
    unmapped. Straight, evenly spaced lines = a healthy map."""
    H, W = dmd_shape
    img = np.zeros(mask.shape + (3,), np.uint8)
    cx = np.where(mask, corr_x, 0.0)
    cy = np.where(mask, corr_y, 0.0)
    img[..., 2] = np.where(mask, 40 + 215 * cx / max(W - 1, 1), 0).astype(np.uint8)
    img[..., 1] = np.where(mask, 40 + 215 * cy / max(H - 1, 1), 0).astype(np.uint8)
    img[..., 0] = np.where(mask, 60, 0).astype(np.uint8)
    lines = mask & (((cx % PREVIEW_GRID_PX) < 2.0) | ((cy % PREVIEW_GRID_PX) < 2.0))
    img[lines] = 255
    return img


# --------------------------------------------------------------- verbs

def capture(session, preset, label: str = "map") -> ProjectorMap:
    """Both axes of the hybrid on the current scene, in ONE rig session (the
    two decodes must see the same scene and the same projector state)."""
    safe_label(label)                     # a bad label fails before the rig comes up
    dmd = dmd_shape_of(preset)
    bases = {ax: _basis(preset, ax) for ax in ("x", "y")}
    t0 = time.perf_counter()
    frames = {}
    with session.rig(preset) as sync:
        for ax, b in bases.items():
            frames[ax] = [sync.shoot(p) for p in b.generate()]
    kw = {"contrast_min": preset.decode.contrast_min, "conf_min": preset.decode.conf_min}
    flds = {}
    for ax in ("x", "y"):
        flds[ax] = bases[ax].decode(frames.pop(ax), preset.geometry_id, **kw)
        if getattr(flds[ax], f"corr_{ax}") is None:
            raise RuntimeError(f"hybrid decode for axis {ax} returned no corr_{ax}")
    fx, fy = flds["x"], flds["y"]
    mask = (fx.mask & fy.mask & np.isfinite(fx.corr_x) & np.isfinite(fy.corr_y))
    corr_x = np.where(mask, fx.corr_x, np.nan).astype(np.float32)
    corr_y = np.where(mask, fy.corr_y, np.nan).astype(np.float32)

    valid_pct = round(100.0 * float(mask.mean()), 3)
    x_pct, y_pct = 100.0 * float(fx.mask.mean()), 100.0 * float(fy.mask.mean())
    warnings = []
    if valid_pct < LOW_VALID_PCT:
        warnings.append(f"only {valid_pct:.1f}% of camera px mapped in both axes — run "
                        "check_signal (aim, focus, exposure) before using grid zones")
    if min(x_pct, y_pct) < 0.5 * max(x_pct, y_pct):
        weak = "row (horizontal stripes)" if y_pct < x_pct else "column (vertical stripes)"
        warnings.append(f"the {weak} decode is much weaker ({x_pct:.1f}% columns vs "
                        f"{y_pct:.1f}% rows valid); the map covers only where both decode")
    for ax, name in (("x", "column (vertical stripes)"), ("y", "row (horizontal stripes)")):
        warnings.extend(f"{name} decode: {w}" for w in flds[ax].meta.get("warnings", []))

    store = session.store
    d = store.new_dir("maps", preset.geometry_id, label=label)
    store.write_npz(d / "map.npz", corr_x=corr_x, corr_y=corr_y, mask=mask)
    preview = _preview(corr_x, corr_y, mask, dmd)
    write_png(d / "preview.png", preview)
    created = datetime.now().isoformat(timespec="seconds")
    axis_meta = {ax: {k: flds[ax].meta.get(k) for k in
                      ("polarity_flipped", "gc_conf", "bits", "steps", "period_px",
                       "complementary", "coarse_bits", "sat_level", "saturated_pct",
                       "saturated_resolved_pct", "saturated_masked_pct",
                       "fix_period_jumps", "period_jumps_fixed_pct")}
                 for ax in ("x", "y")}
    meta = {"kind": "projector_map", "id": d.name, "label": label,
            "geometry_id": preset.geometry_id, "created": created, "preset": preset.name,
            "mock": bool(preset.mock), "dmd_shape": list(dmd),
            "camera_shape": list(mask.shape), "valid_pct": valid_pct,
            "valid_pct_x": round(x_pct, 3), "valid_pct_y": round(y_pct, 3),
            "decode": {"x": axis_meta["x"], "y": axis_meta["y"], **kw},
            "capture_s": round(time.perf_counter() - t0, 2),
            "units": {**MAP_UNITS, "valid_pct_x": "% of camera px",
                      "valid_pct_y": "% of camera px", "gc_conf": "camera counts (modulation)",
                      "sat_level": "camera counts (Mono8)",
                      "saturated_pct": "% of lit camera px",
                      "saturated_resolved_pct": "% of lit camera px",
                      "saturated_masked_pct": "% of lit camera px",
                      "period_jumps_fixed_pct": "% of valid camera px"},
            "note": "map of the scene at capture time; re-capture after moving the rig or "
                    "the part if zones must follow it exactly",
            "warnings": warnings}
    store.write_json(d / "meta.json", meta)
    out = store.output_path("previews", f"{label}_map", ".png")
    if write_png(out, preview):
        store.record_output("previews", out, label=label, geometry_id=preset.geometry_id,
                            source=d)
    return ProjectorMap(id=d.name, geometry_id=preset.geometry_id, path=d, created=created,
                        corr_x=corr_x, corr_y=corr_y, mask=mask, dmd_shape=dmd,
                        warnings=warnings, label=label, valid_pct=valid_pct, meta=meta)


def load(path: Path) -> ProjectorMap:
    """A stored map from its entry directory (or its map.npz)."""
    p = Path(path)
    d = p.parent if p.suffix == ".npz" else p
    f = d / "map.npz"
    if not f.is_file():
        raise FileNotFoundError(f"{d} is not a projector map entry (no map.npz)")
    mf = d / "meta.json"
    meta = json.loads(mf.read_text(encoding="utf-8")) if mf.is_file() else {}
    with np.load(f) as z:
        cx, cy, m = z["corr_x"], z["corr_y"], z["mask"]
    return ProjectorMap(id=meta.get("id", d.name), geometry_id=meta.get("geometry_id", d.parent.name),
                        path=d, created=meta.get("created", ""), corr_x=cx, corr_y=cy, mask=m,
                        dmd_shape=tuple(meta.get("dmd_shape", DMD_SHAPE)),
                        warnings=list(meta.get("warnings", [])), label=meta.get("label", ""),
                        valid_pct=float(meta.get("valid_pct", 100.0 * float(np.mean(m)))),
                        meta=meta)


def list_maps(session, geometry_id: Optional[str] = None) -> list:
    """Summaries (dicts, no arrays) of the stored maps, oldest first."""
    store = session.store
    geos = [geometry_id] if geometry_id else store.geometry_ids("maps")
    out = []
    for g in geos:
        for d in store.entries("maps", g):
            if not (d / "map.npz").is_file():
                continue
            mf = d / "meta.json"
            m = json.loads(mf.read_text(encoding="utf-8")) if mf.is_file() else {}
            out.append({"id": d.name, "label": m.get("label", ""),
                        "geometry_id": m.get("geometry_id", g), "created": m.get("created", ""),
                        "valid_pct": m.get("valid_pct"), "preset": m.get("preset"),
                        "mock": m.get("mock"), "path": d, "warnings": m.get("warnings", []),
                        "units": {"valid_pct": MAP_UNITS["valid_pct"]}})
    return sorted(out, key=lambda r: r["id"])


def latest(session, geometry_id: str) -> Optional[ProjectorMap]:
    """Newest stored map for the geometry, or None."""
    for d in reversed(session.store.entries("maps", geometry_id)):
        if (d / "map.npz").is_file():
            return load(d)
    return None

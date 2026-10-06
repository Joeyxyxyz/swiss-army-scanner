"""
Atlas — a frozen "held field" from the Surfer method (numbered grid ->
flood-verify -> freeze): named DMD rectangles such as the cell face
('P17-P22' -> DMD [720, 768, 768, 1056]) and the floor ('P23').

load_atlas() reads the Surfer JSON (read-only), cross-checks each region's
grid label against its DMD rect, and exposes every region both as an exact
DMD rect and as a zone spec zones.parse_zone understands ('P17:P22@48').
import_atlas() keeps an append-only copy in the library:
    library/atlases/<geometry_id>/<name>/<ts>.json
resolve_region(session, preset, 'atlas:<name>/<region>') returns the same
(projector_mask, camera_mask, cells) triple as zones.resolve, so a zone
scan can target an atlas region once zones.resolve / api dispatch on the
'atlas:' prefix.

Rects are (x0, y0, x1, y1) DMD px, [x0, x1) x [y0, y1), origin DMD (0, 0)
top-left of the 912 x 1140 pattern space (zones.DMD_SHAPE is rows, cols).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .. import rules
from ..decode import projmap
from ..library.store import LibraryStore, safe_label, timestamp
from . import zones as Z

log = logging.getLogger(__name__)

ATLAS_PREFIX = "atlas:"
ATLAS_KIND = "atlases"


@dataclass
class Region:
    name: str
    rect: tuple                  # (x0, y0, x1, y1) DMD px, half-open
    label: str = ""              # grid label as written in the atlas ('P17-P22')
    zone_spec: str = ""          # zones.parse_zone form with spacing ('P17:P22@48'); '' if off-grid

    def mask(self) -> np.ndarray:
        return rect_mask(self.rect)


@dataclass
class Atlas:
    name: str
    step_px: int
    dmd_size: tuple              # (W, H) DMD px
    origin: str
    regions: dict                # region name -> Region
    source: str = ""
    notes: dict = field(default_factory=dict)
    path: Optional[Path] = None
    doc: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    def region(self, name: str) -> Region:
        if name not in self.regions:
            raise ValueError(f"atlas {self.name!r} has no region {name!r}; "
                             f"regions: {sorted(self.regions)}")
        return self.regions[name]


# --------------------------------------------------------------- rects

def rect_mask(rect, shape: tuple = Z.DMD_SHAPE) -> np.ndarray:
    """bool DMD mask of a half-open (x0, y0, x1, y1) rect, clipped to the DMD."""
    x0, y0, x1, y1 = _check_rect(rect)
    m = np.zeros(shape, bool)
    m[max(y0, 0):min(y1, shape[0]), max(x0, 0):min(x1, shape[1])] = True
    return m


def _check_rect(rect) -> tuple:
    try:
        x0, y0, x1, y1 = (int(v) for v in rect)
    except (TypeError, ValueError):
        raise ValueError(f"rect must be [x0, y0, x1, y1] DMD px, got {rect!r}") from None
    if not (x1 > x0 and y1 > y0):
        raise ValueError(f"rect {rect!r} is empty (need x1 > x0 and y1 > y0)")
    return x0, y0, x1, y1


def cells_rect(cells, grid: Z.GridSpec) -> tuple:
    rs = [Z.cell_rect(c, grid) for c in cells]
    return (min(r[0] for r in rs), min(r[1] for r in rs),
            max(r[2] for r in rs), max(r[3] for r in rs))


def rect_cells(rect, grid: Z.GridSpec) -> list:
    """Grid cells overlapping a rect."""
    x0, y0, x1, y1 = _check_rect(rect)
    s = grid.spacing_px
    return [(c, r) for c in range(max(x0 // s, 0), min(-(-x1 // s), grid.n_cols))
            for r in range(max(y0 // s, 0), min(-(-y1 // s), grid.n_rows))]


def label_to_zone(label: str) -> str:
    """'P17-P22' -> 'P17:P22' (Surfer writes blocks with '-')."""
    return re.sub(r"(?<=\d)\s*-\s*(?=[A-Za-z])", ":", label.strip())


# ---------------------------------------------------------------- load

def load_atlas(path, name: Optional[str] = None) -> Atlas:
    """Read a Surfer atlas JSON (never written to). Regions come from
    dmd.held_<region> rects, labelled by grid_labels.<region>."""
    p = Path(path)
    doc = json.loads(p.read_text(encoding="utf-8-sig"))      # PowerShell writes a BOM
    return atlas_from_doc(doc, name or p.stem, path=p)


def atlas_from_doc(doc: dict, name: str, path: Optional[Path] = None) -> Atlas:
    if not isinstance(doc, dict):
        raise ValueError("atlas JSON must be an object")
    step = doc.get("step_px")
    grid = Z.grid_spec(step)                                  # validates the step
    dmd = doc.get("dmd") or {}
    size = tuple(int(v) for v in dmd.get("size", (Z.DMD_SHAPE[1], Z.DMD_SHAPE[0])))
    if size != (Z.DMD_SHAPE[1], Z.DMD_SHAPE[0]):
        raise ValueError(f"atlas DMD size {size} is not the engine's "
                         f"{Z.DMD_SHAPE[1]}x{Z.DMD_SHAPE[0]} pattern space")
    labels = doc.get("grid_labels") or {}
    warnings, regions = [], {}
    rects = {k[len("held_"):]: v for k, v in dmd.items() if k.startswith("held_")}
    rects.update(doc.get("regions") or {})                   # optional {name: rect} extension
    for rname, rect in rects.items():
        rect = _check_rect(rect)
        label = labels.get(rname, "") if isinstance(labels.get(rname), str) else ""
        zone_spec = ""
        if label:
            cells = Z.parse_zone(label_to_zone(label), grid)
            if cells_rect(cells, grid) != rect:
                warnings.append(f"region {rname!r}: label {label!r} spans "
                                f"{list(cells_rect(cells, grid))}, the rect is {list(rect)}; "
                                "the rect wins")
            else:
                zone_spec = f"{label_to_zone(label)}@{grid.spacing_px}"
        if not zone_spec:
            cells = rect_cells(rect, grid)
            if cells and cells_rect(cells, grid) == rect:
                a, b = cells[0], cells[-1]
                zone_spec = f"{Z.cell_name(a, grid)}:{Z.cell_name(b, grid)}@{grid.spacing_px}"
        regions[rname] = Region(rname, rect, label, zone_spec)
    if not regions:
        raise ValueError("atlas has no regions (expected dmd.held_<name> rects)")
    for w in warnings:
        log.warning("atlas %s: %s", name, w)
    return Atlas(name=name, step_px=grid.spacing_px, dmd_size=size,
                 origin=str(doc.get("origin", "")), regions=regions,
                 source=str(doc.get("source", "")),
                 notes={k: v for k, v in labels.items() if k not in regions},
                 path=path, doc=doc, warnings=warnings)


# ------------------------------------------------------------- library

def _atlas_dir(store: LibraryStore, geometry_id: str, name: str) -> Path:
    return store.root / ATLAS_KIND / safe_label(geometry_id) / safe_label(name)


def import_atlas(session, path, name: str, geometry_id: str) -> Path:
    """Copy a validated atlas into library/atlases/<geometry_id>/<name>/<ts>.json
    (append-only: always a new file; the source is only read)."""
    atlas = load_atlas(path, name)
    d = _atlas_dir(session.store, geometry_id, name)
    d.mkdir(parents=True, exist_ok=True)
    stem = timestamp()
    out = next((d / f"{stem}{'' if n == 0 else f'-{n}'}.json" for n in range(1000)
                if not (d / f"{stem}{'' if n == 0 else f'-{n}'}.json").exists()), None)
    if out is None:
        raise rules.OverwriteRefused(f"could not allocate a new atlas file under {d}")
    doc = dict(atlas.doc)
    doc["_import"] = {"from": str(Path(path)), "name": safe_label(name),
                      "geometry_id": geometry_id, "time": timestamp(),
                      "regions": {r.name: {"rect": list(r.rect), "zone": r.zone_spec}
                                  for r in atlas.regions.values()},
                      "warnings": atlas.warnings}
    return session.store.write_json(out, doc)


def list_atlases(session, geometry_id: Optional[str] = None) -> list:
    root = session.store.root / ATLAS_KIND
    rows = []
    if not root.is_dir():
        return rows
    for g in sorted(p for p in root.iterdir() if p.is_dir()):
        if geometry_id and g.name != geometry_id:
            continue
        for n in sorted(p for p in g.iterdir() if p.is_dir()):
            files = sorted(n.glob("*.json"))
            if files:
                rows.append({"geometry_id": g.name, "name": n.name, "latest": files[-1],
                             "versions": len(files)})
    return rows


def latest_atlas(session, geometry_id: str, name: str) -> Atlas:
    d = _atlas_dir(session.store, geometry_id, name)
    files = sorted(d.glob("*.json")) if d.is_dir() else []
    if not files:
        have = [r["name"] for r in list_atlases(session, geometry_id)]
        raise ValueError(f"no atlas {name!r} for {geometry_id} (import_atlas first); "
                         f"have: {have or 'none'}")
    return load_atlas(files[-1], name)


# ------------------------------------------------------------- resolve

def is_atlas_spec(zone_text) -> bool:
    return isinstance(zone_text, str) and zone_text.strip().lower().startswith(ATLAS_PREFIX)


def parse_spec(zone_text: str) -> tuple:
    """'atlas:cell/face' -> ('cell', 'face')."""
    body = zone_text.strip()[len(ATLAS_PREFIX):]
    name, sep, region = body.partition("/")
    if not sep or not name.strip() or not region.strip():
        raise ValueError(f"atlas zone {zone_text!r} must look like 'atlas:<name>/<region>'")
    return name.strip(), region.strip()


def resolve_region(session, preset, zone_text: str, pmap=None, atlas: Optional[Atlas] = None):
    """'atlas:<name>/<region>' -> (projector_mask, camera_mask or None, cells),
    exactly like zones.resolve. The projector mask is the region's exact
    rect; cells are the atlas-grid cells it covers."""
    name, rname = parse_spec(zone_text)
    if atlas is None:
        atlas = latest_atlas(session, preset.geometry_id, name)
    reg = atlas.region(rname)
    grid = Z.grid_spec(atlas.step_px)
    pm = reg.mask()
    cells = rect_cells(reg.rect, grid)
    if pmap is None:
        pmap = projmap.latest(session, preset.geometry_id)
    else:
        rules.require_same_geometry(pmap.geometry_id, preset.geometry_id, "zone map")
    if pmap is None:
        log.warning("zone %s: no projector map for %s - lit but results not cut to it",
                    zone_text, preset.geometry_id)
        return pm, None, cells
    cam = pmap.camera_mask_for_projector_mask(pm)
    if not cam.any():
        raise ValueError(f"atlas zone {zone_text!r} (DMD {list(reg.rect)}) is not seen by the "
                         f"camera in map {pmap.id}")
    return pm, cam, cells

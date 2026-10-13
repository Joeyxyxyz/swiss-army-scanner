"""
Grid & zones — a labelled grid projected onto the scene, so a person or an
AI agent can say "scan zone C4" (or "C4:D6", "B2,E5").

Grid: DMD (projector) space is divided into square cells of `spacing_px`
mirrors; columns lettered A, B, C... left->right, rows numbered 1, 2, 3...
top->bottom (as seen in the CAMERA image when the rig is set up per the
preset). project_grid() shows the grid with big cell labels over HDMI,
photographs it, and saves the photo (+ an overlay with cell boundaries
mapped into camera space when a ProjectorMap exists). That photo is what a
human or an agent reads to choose a zone.

The overlay is the image to hand an agent: its labels are drawn in CAMERA
space, upright, where each cell actually lands — so they stay readable
even when the optics mirror or rotate the projected text. Without a map
the raw photo still shows the projected labels, but nothing guarantees
their orientation.

Zone specs (parse_zone): "C4" one cell; "C4:D6" inclusive block; "B2,E5"
union; "all" whole DMD. resolve() turns a spec into (projector_mask,
camera_mask) using the latest ProjectorMap for the geometry. Zone scans
light ONLY the zone (less inter-reflection) and cut results to it. A spec
names cells of the grid last projected for the geometry (its spacing is
remembered in the output bin); "C4@64" names the cell of a 64 px grid.

Custom zones: Zone(name, polygon, space="camera"|"projector", pattern=...)
for machine-vision marking — project_zones() lights each zone with its
pattern (flood, outline, grid, stripes, checker) and photographs it.

Polarity: grids and zones are sent raw (no duty / invert chain) after a
raw black vs raw white measurement, so they are PHYSICALLY bright-on-dim
whichever way the HDMI path lands (the bench DMD shows raw as sent).

Holds: hold_s keeps the grid / zones up after the photo (aiming, marking),
as HOLD_SLICE_S slices of the window's pumped hold (never a sleep), so a UI
can stop it by setting a threading.Event passed as `cancel`. A hold is a
sustained projection: rule 3's LED duration check runs before the rig opens.

Outputs: library/output/previews/<ts>_grid*.png, <ts>_zones*.png
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from .. import rules
from ..decode import projmap
from ..library.store import safe_label

log = logging.getLogger(__name__)

DMD_SHAPE = (1140, 912)        # rows, cols (pattern space)
DEFAULT_SPACING = 96
MIN_SPACING = 16               # smaller cells can't carry a readable label

FILL_LEVEL = 40                # dim fill: the scene stays visible under the grid
LINE_PX = 3                    # bright cell boundaries
BORDER_PX = 8                  # thick frame around the whole DMD
PATTERNS = ("flood", "outline", "grid", "stripes", "checker")
SPACES = ("camera", "projector")
OUTLINE_PX = 3
GRID_PITCH, GRID_LINE = 16, 2
STRIPE_PX = 8
CHECKER_PX = 16

POLARITY_MARGIN = 2.0          # camera counts between raw white and raw black
VISIBLE_FRACTION = 0.25        # of a full cell's camera px for "the camera sees it"
MIN_VISIBLE_PX = 16
MAP_AGREEMENT_MIN = 0.5        # photo vs map-predicted grid correlation

GRID_JSON_EXT = ".grid.json"
ZONES_JSON_EXT = ".zones.json"
LABEL_BGR = (0, 255, 255)      # yellow
LINE_BGR = (255, 255, 0)       # cyan
PLATE_KEEP = 0.25              # label plates keep 25% of the photo behind them
OVERLAY_TONE = 0.8             # overlay photo toned down so the cyan / yellow marks pop

UNITS = {"spacing_px": "DMD px (mirrors)", "cell_centers": "camera px [x, y]",
         "polygon": "camera px [x, y] or DMD px [col, row] (per Zone.space)",
         "map_agreement": "correlation, photo vs map-predicted grid (1 = perfect)"}


@dataclass
class Zone:
    name: str
    polygon: list               # [[x, y], ...]
    space: str = "camera"       # "camera" | "projector"
    pattern: str = "flood"      # flood | outline | grid | stripes | checker
    level: int = 255


@dataclass
class GridSpec:
    spacing_px: int = DEFAULT_SPACING
    n_cols: int = 0             # derived from DMD_SHAPE
    n_rows: int = 0
    col_labels: list = field(default_factory=list)   # ["A", "B", ...]
    row_labels: list = field(default_factory=list)   # ["1", "2", ...]


@dataclass
class ZoneShot:
    kind: str                   # "grid" | "zones"
    grid: Optional[GridSpec]
    zones: list                 # names (grid cells or Zone.name)
    camera_png: Optional[Path]  # what the camera saw
    overlay_png: Optional[Path] # camera image + labels/boundaries (needs a ProjectorMap)
    dmd_png: Optional[Path]     # what was sent to the DMD
    cells_visible: list = field(default_factory=list)   # grid cells the camera can see (with a map)
    warnings: list = field(default_factory=list)
    json_path: Optional[Path] = None      # machine-readable record of this shot (output bin)
    cell_centers: dict = field(default_factory=dict)  # cell / zone name -> [x, y] camera px (map)
    polarity_inverted: Optional[bool] = None          # raw HDMI path measured inverted?
    map_id: Optional[str] = None                      # ProjectorMap used for the overlay
    units: dict = field(default_factory=lambda: dict(UNITS))
    held_s: float = 0.0                               # seconds the pattern was held after the photo
    hold_cancelled: bool = False                      # the hold was stopped via `cancel`


# ------------------------------------------------------------- the grid

def col_label(i: int) -> str:
    """0 -> A ... 25 -> Z, 26 -> AA (spreadsheet style)."""
    s, n = "", int(i) + 1
    while n:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s


def col_index(letters: str) -> int:
    n = 0
    for ch in letters.upper():
        n = n * 26 + (ord(ch) - 64)
    return n - 1


def grid_spec(spacing_px: int = DEFAULT_SPACING) -> GridSpec:
    if isinstance(spacing_px, bool) or not isinstance(spacing_px, (int, float, np.integer)) \
            or not float(spacing_px).is_integer():
        raise ValueError(f"spacing_px must be a whole number of DMD px, got {spacing_px!r}")
    s = int(spacing_px)
    rows, cols = DMD_SHAPE
    if not MIN_SPACING <= s <= max(DMD_SHAPE):
        raise ValueError(f"spacing_px must be {MIN_SPACING}..{max(DMD_SHAPE)} DMD px (a cell "
                         f"needs room for a readable label), got {spacing_px}")
    n_cols, n_rows = -(-cols // s), -(-rows // s)
    return GridSpec(spacing_px=s, n_cols=n_cols, n_rows=n_rows,
                    col_labels=[col_label(i) for i in range(n_cols)],
                    row_labels=[str(j + 1) for j in range(n_rows)])


def cell_name(cell, grid: GridSpec) -> str:
    c, r = cell
    return f"{grid.col_labels[c]}{grid.row_labels[r]}"


def cell_rect(cell, grid: GridSpec) -> tuple:
    """(x0, y0, x1, y1) DMD px, [x0, x1) x [y0, y1), clipped to the DMD."""
    c, r = cell
    s = grid.spacing_px
    return (c * s, r * s, min((c + 1) * s, DMD_SHAPE[1]), min((r + 1) * s, DMD_SHAPE[0]))


def all_cells(grid: GridSpec) -> list:
    return [(c, r) for c in range(grid.n_cols) for r in range(grid.n_rows)]


def grid_image(spec: GridSpec, labels: bool = True) -> np.ndarray:
    """uint8 DMD_SHAPE image: grid lines + big cell labels (pattern space)."""
    import cv2
    H, W = DMD_SHAPE
    s = spec.spacing_px
    img = np.full(DMD_SHAPE, FILL_LEVEL, np.uint8)
    h = LINE_PX // 2
    for k in range(1, spec.n_cols):
        img[:, max(k * s - h, 0):k * s - h + LINE_PX] = 255
    for k in range(1, spec.n_rows):
        img[max(k * s - h, 0):k * s - h + LINE_PX, :] = 255
    b = BORDER_PX
    img[:b, :] = img[-b:, :] = 255
    img[:, :b] = img[:, -b:] = 255
    if not labels:
        return img
    font = cv2.FONT_HERSHEY_SIMPLEX
    thick = max(2, s // 32)
    longest = max((cell_name(c, spec) for c in all_cells(spec)), key=len)
    (tw, th), base = cv2.getTextSize(longest, font, 1.0, thick)
    scale = min(0.72 * s / tw, 0.42 * s / (th + base))    # one size for every full cell
    for cell in all_cells(spec):
        x0, y0, x1, y1 = cell_rect(cell, spec)
        # the usable part of the cell: inside the lines and the frame
        ix0, iy0 = max(x0 + h + 1, b), max(y0 + h + 1, b)
        ix1, iy1 = min(x1 - h - 1, W - b), min(y1 - h - 1, H - b)
        text = cell_name(cell, spec)
        (w1, h1), b1 = cv2.getTextSize(text, font, 1.0, thick)
        sc = min(scale, 0.9 * (ix1 - ix0) / w1, 0.8 * (iy1 - iy0) / (h1 + b1))
        if sc < 0.3:
            continue                     # a sliver at the DMD edge: no room for a label
        (w2, h2), _ = cv2.getTextSize(text, font, sc, thick)
        org = (int(round((ix0 + ix1) / 2 - w2 / 2)), int(round((iy0 + iy1) / 2 + h2 / 2)))
        cv2.putText(img, text, org, font, sc, 255, thick, cv2.LINE_AA)
    return img


# ------------------------------------------------------------ zone specs

_CELL_RE = re.compile(r"^([A-Za-z]+)(\d+)$")


def _help(grid: GridSpec) -> str:
    last = cell_name((grid.n_cols - 1, grid.n_rows - 1), grid)
    return (f"valid cells are A1..{last} (columns A-{grid.col_labels[-1]}, rows "
            f"1-{grid.n_rows}; a {grid.n_cols} x {grid.n_rows} grid of {grid.spacing_px} DMD px "
            "cells). Examples: 'C4', 'C4:D6', 'B2,E5', 'all'")


def _cell(token: str, grid: GridSpec, text: str) -> tuple:
    m = _CELL_RE.match(token)
    if not m:
        raise ValueError(f"zone {text!r}: {token!r} is not a cell name (a column letter then a "
                         f"row number, like 'C4'); {_help(grid)}")
    letters, digits = m.group(1).upper(), m.group(2)
    c, r = col_index(letters), int(digits) - 1
    if not 0 <= c < grid.n_cols:
        raise ValueError(f"zone {text!r}: column {letters} is outside "
                         f"A-{grid.col_labels[-1]}; {_help(grid)}")
    if not 0 <= r < grid.n_rows:
        raise ValueError(f"zone {text!r}: row {int(digits)} is outside 1-{grid.n_rows}; "
                         f"{_help(grid)}")
    return (c, r)


def parse_zone(spec_text: str, grid: GridSpec) -> list:
    """'C4' | 'C4:D6' | 'B2,E5' | 'all' -> list of (col_idx, row_idx) cells."""
    if not isinstance(spec_text, str) or not spec_text.strip():
        raise ValueError(f"empty zone; {_help(grid)}")
    text = spec_text.strip()
    norm = re.sub(r"\s*:\s*", ":", text)
    parts = [q for q in re.split(r"[,;\s]+", norm) if q]      # stray separators are harmless
    if not parts:
        raise ValueError(f"empty zone {text!r}; {_help(grid)}")
    cells = set()
    for part in parts:
        if part.lower() == "all":
            cells.update(all_cells(grid))
        elif ":" in part:
            ends = part.split(":")
            if len(ends) != 2 or not all(ends):
                raise ValueError(f"zone {text!r}: {part!r} is not a block — use two corners "
                                 f"like 'C4:D6'; {_help(grid)}")
            (c0, r0), (c1, r1) = (_cell(e, grid, text) for e in ends)
            cells.update((c, r) for c in range(min(c0, c1), max(c0, c1) + 1)
                         for r in range(min(r0, r1), max(r0, r1) + 1))
        else:
            cells.add(_cell(part, grid, text))
    return sorted(cells)


def _check_cells(cells, grid: GridSpec) -> list:
    out = []
    for cell in cells:
        if isinstance(cell, str):
            out.extend(parse_zone(cell, grid))
            continue
        try:
            c, r = (int(v) for v in cell)
        except (TypeError, ValueError):
            raise ValueError(f"cell {cell!r} must be (col_idx, row_idx) or a name like 'C4'") \
                from None
        if not (0 <= c < grid.n_cols and 0 <= r < grid.n_rows):
            raise ValueError(f"cell {cell!r} is outside the grid; {_help(grid)}")
        out.append((c, r))
    return out


def projector_mask(cells: list, grid: GridSpec) -> np.ndarray:
    """bool DMD_SHAPE mask of the given cells."""
    m = np.zeros(DMD_SHAPE, bool)
    for cell in _check_cells(cells, grid):
        x0, y0, x1, y1 = cell_rect(cell, grid)
        m[y0:y1, x0:x1] = True
    return m


def _split_spacing(zone_text: str):
    """'C4@64' -> ('C4', 64); 'C4' -> ('C4', None)."""
    if not isinstance(zone_text, str):
        raise ValueError(f"zone must be text like 'C4', got {zone_text!r}")
    text, at, sp = zone_text.rpartition("@")
    if not at:
        return zone_text, None
    try:
        return text, int(sp.strip())
    except ValueError:
        raise ValueError(f"zone {zone_text!r}: after '@' comes the grid spacing in DMD px, "
                         "like 'C4@64'") from None


def last_grid_spacing(session, geometry_id: str) -> Optional[int]:
    """Spacing of the grid last projected for this geometry (from the output
    bin), so 'C4' means the cell the operator / agent actually saw."""
    store = session.store
    for row in reversed(store.outputs("previews")):
        if row.get("geometry_id") != geometry_id or not row.get("file", "").endswith(GRID_JSON_EXT):
            continue
        f = store.root / row["file"]
        try:
            return int(json.loads(f.read_text(encoding="utf-8"))["spacing_px"])
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return None


def resolve(session, preset, zone_text: str, pmap=None):
    """-> (projector_mask bool DMD_SHAPE, camera_mask bool camera shape or None,
    cells). camera_mask needs a ProjectorMap (latest for the geometry if pmap
    is None); without one it is None and a warning explains how to make one.
    'atlas:<name>/<region>' names a region of an imported Surfer atlas;
    'zoneset:<set>[:Z1,Z3]' the zones of a painted zone set (engine/zones: lit through
    their union; never a camera mask - their camera px are derived per scan)."""
    from . import atlas as _atlas
    if _atlas.is_atlas_spec(zone_text):
        return _atlas.resolve_region(session, preset, zone_text, pmap=pmap)
    from ..zones import use as _zoneset
    if _zoneset.is_spec(zone_text):
        return _zoneset.resolve(session, preset, zone_text)
    text, spacing = _split_spacing(zone_text)
    if spacing is None:
        spacing = last_grid_spacing(session, preset.geometry_id) or DEFAULT_SPACING
    grid = grid_spec(spacing)
    cells = parse_zone(text, grid)
    pm = projector_mask(cells, grid)
    if pmap is None:
        pmap = projmap.latest(session, preset.geometry_id)
    else:
        rules.require_same_geometry(pmap.geometry_id, preset.geometry_id, "zone map")
    if pmap is None:
        log.warning("zone %s: no projector map for %s - the zone is lit but results are not "
                    "cut to it (run capture_map)", zone_text, preset.geometry_id)
        return pm, None, cells
    cam = pmap.camera_mask_for_projector_mask(pm)
    if not cam.any():
        seen = cells_seen(pmap, grid)[0]
        raise ValueError(f"zone {zone_text!r} is not seen by the camera in map {pmap.id} "
                         f"(0 mapped camera px); cells the camera sees: "
                         f"{', '.join(seen[:30]) or 'none'}{' ...' if len(seen) > 30 else ''}")
    return pm, cam, cells


# ---------------------------------------------------------- map helpers

def _cell_ids(pmap, grid: GridSpec) -> np.ndarray:
    """int32 camera image: cell index (col * n_rows + row) per mapped px, -1 elsewhere."""
    s = grid.spacing_px
    ids = np.full(pmap.camera_shape, -1, np.int32)
    rows, cols = pmap.dmd_index()
    ids[pmap.mask] = (cols // s) * grid.n_rows + rows // s
    return ids


def cells_seen(pmap, grid: GridSpec, ids: Optional[np.ndarray] = None):
    """-> (visible names, counts per cell, centroids (n, 2) camera px).

    A cell is visible when at least VISIBLE_FRACTION of the camera px a full
    cell of its DMD area would cover are mapped (density from the best-seen
    cells, so edge slivers and shadowed cells are judged fairly)."""
    ids = _cell_ids(pmap, grid) if ids is None else ids
    n = grid.n_cols * grid.n_rows
    v, u = np.nonzero(ids >= 0)
    k = ids[v, u]
    counts = np.bincount(k, minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        cen = np.stack([np.bincount(k, weights=u, minlength=n) / counts,
                        np.bincount(k, weights=v, minlength=n) / counts], axis=1)
    cells = all_cells(grid)
    area = np.array([(x1 - x0) * (y1 - y0) for x0, y0, x1, y1
                     in (cell_rect(c, grid) for c in cells)], np.float64)
    seen = counts > 0
    names = []
    if seen.any():
        density = float(np.percentile(counts[seen] / area[seen], 75))
        vis = (counts >= VISIBLE_FRACTION * density * area) & (counts >= MIN_VISIBLE_PX)
        names = [cell_name(c, grid) for c, ok in zip(cells, vis) if ok]
    return names, counts, cen


def _cell_size_camera(pmap, spacing: int) -> tuple:
    """(w, h) camera px of one full cell, from the map's median Jacobian —
    works for any rotation/mirroring of the DMD in the camera image."""
    cx, cy, m = pmap.corr_x, pmap.corr_y, pmap.mask
    mx, my = m[:, 1:] & m[:, :-1], m[1:, :] & m[:-1, :]
    if not (mx.any() and my.any()):
        return float(spacing), float(spacing)
    J = np.array([[np.median((cx[:, 1:] - cx[:, :-1])[mx]), np.median((cx[1:] - cx[:-1])[my])],
                  [np.median((cy[:, 1:] - cy[:, :-1])[mx]), np.median((cy[1:] - cy[:-1])[my])]])
    if abs(np.linalg.det(J)) < 1e-6:
        return float(spacing), float(spacing)
    e = np.linalg.inv(J) @ np.array([[spacing, 0.0], [0.0, spacing]])   # columns: cell edges
    return float(abs(e[0, 0]) + abs(e[0, 1])), float(abs(e[1, 0]) + abs(e[1, 1]))


def _boundaries(ids: np.ndarray) -> np.ndarray:
    """Camera px where the cell index changes between two mapped neighbours."""
    import cv2
    edge = np.zeros(ids.shape, bool)
    a, b = ids[:, 1:], ids[:, :-1]
    edge[:, 1:] |= (a != b) & (a >= 0) & (b >= 0)
    a, b = ids[1:, :], ids[:-1, :]
    edge[1:, :] |= (a != b) & (a >= 0) & (b >= 0)
    return cv2.dilate(edge.astype(np.uint8), np.ones((2, 2), np.uint8)).astype(bool)


def _label_scale(texts, box_w: float, box_h: float, thick: int) -> float:
    import cv2
    font = cv2.FONT_HERSHEY_SIMPLEX
    longest = max(texts, key=len) if texts else "A1"
    (tw, th), base = cv2.getTextSize(longest, font, 1.0, thick)
    # small enough that most of the cell (the part under it) stays visible
    return float(np.clip(min(0.42 * box_w / tw, 0.3 * box_h / (th + base)), 0.35, 4.0))


def _put_label(img, text: str, x: float, y: float, scale: float, thick: int) -> None:
    """Upright text centred on (x, y) on a translucent dark plate: readable on
    any photo, and the scene still shows through behind it."""
    import cv2
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), base = cv2.getTextSize(text, font, scale, thick)
    x0, y0 = int(round(x - tw / 2)), int(round(y + th / 2))
    pad = max(2, thick)
    H, W = img.shape[:2]
    ya, yb = max(y0 - th - pad, 0), min(y0 + base + pad, H)
    xa, xb = max(x0 - pad, 0), min(x0 + tw + pad, W)
    if ya < yb and xa < xb:
        img[ya:yb, xa:xb] = (img[ya:yb, xa:xb] * PLATE_KEEP).astype(np.uint8)
    cv2.putText(img, text, (x0, y0), font, scale, LABEL_BGR, thick, cv2.LINE_AA)


def _map_agreement(pmap, physical: np.ndarray, photo: np.ndarray) -> Optional[float]:
    """Correlation of the photo with the image the map predicts (the physical
    DMD image looked up at each camera px's correspondence). Low = the map
    is stale: the rig, focus or scene changed since it was captured."""
    rows, cols = pmap.dmd_index()
    pred = physical[rows, cols].astype(np.float64)[::5]
    obs = np.asarray(photo, np.float64)[pmap.mask][::5]
    if pred.size < 100 or pred.std() == 0 or obs.std() == 0:
        return None
    return round(float(np.corrcoef(pred, obs)[0, 1]), 4)


# ------------------------------------------------------------ rig steps

HOLD_SLICE_S = 0.25            # a long hold runs in slices so a UI can cancel it


def _hold(hold_s, preset=None, what: str = "grid/zone hold") -> float:
    try:
        h = float(hold_s)
    except (TypeError, ValueError):
        raise ValueError(f"hold_s must be seconds (a number), got {hold_s!r}") from None
    if not 0.0 <= h <= 3600.0:
        raise ValueError(f"hold_s must be 0..3600 s, got {hold_s!r}")
    if preset is not None:
        # rule 3 over time: a long hold is a sustained projection, so 'burst'
        # LED currents above the running-average limit are refused before the
        # rig opens
        rules.check_led_duration(preset.projection.led, h, what)
    return h


def _check_cancel(cancel) -> None:
    if cancel is not None and not callable(getattr(cancel, "is_set", None)):
        raise ValueError("cancel must be a threading.Event (or anything with is_set())")


def _pumped_hold(window, seconds: float, cancel=None) -> float:
    """Keep the pattern up for `seconds` in HOLD_SLICE_S slices of the
    window's own pumped hold (never a sleep), stopping early once `cancel`
    (a threading.Event) is set. -> seconds actually held."""
    held = 0.0
    while seconds - held > 1e-9:
        if cancel is not None and cancel.is_set():
            break
        step = min(HOLD_SLICE_S, seconds - held)
        window.hold(step)
        held += step
    return held


def _measure_polarity(sync, warnings: list):
    """Raw black vs raw white: whichever is brighter is 'on' (measured, never
    assumed — the bench HDMI path has landed patterns inverted).
    -> (inverted, the physically full-on frame = the scene under a flood)."""
    f0 = sync.shoot(np.zeros(DMD_SHAPE, np.uint8), raw=True)
    f1 = sync.shoot(np.full(DMD_SHAPE, 255, np.uint8), raw=True)
    diff = float(f1.mean()) - float(f0.mean())
    if abs(diff) < POLARITY_MARGIN:
        warnings.append(f"raw white and raw black look the same to the camera ({diff:+.1f} "
                        "counts): is the projector on and aimed at the scene?")
        return False, f1
    if diff < 0:
        warnings.append("raw HDMI path measured INVERTED; the pattern was sent inverted so it "
                        "is physically bright-on-dim")
    return diff < 0, (f0 if diff < 0 else f1)


def _project_and_shoot(session, preset, physical: np.ndarray, hold: float, warnings: list,
                       cancel=None):
    """-> (photo of `physical`, image sent, inverted, flood-lit scene frame,
    seconds held)."""
    held = 0.0
    with session.rig(preset) as sync:
        inverted, lit = _measure_polarity(sync, warnings)
        sent = (255 - physical) if inverted else physical
        photo = sync.shoot(sent, raw=True)
        if hold > 0:
            # pumped hold in slices: the window stays painted, a UI can stop it
            held = _pumped_hold(sync.projector.window, hold, cancel)
            if held < hold - 1e-9:
                warnings.append(f"hold cancelled after {held:.2f} of {hold:g} s")
    return photo, sent, inverted, lit, held


def _photo_u8(photo: np.ndarray) -> np.ndarray:
    f = np.asarray(photo, np.float32)
    hi = max(float(np.percentile(f, 99.8)), 1.0)
    return np.clip(f * (255.0 / hi), 0, 255).astype(np.uint8)


def _save(store, preset, label: str, suffix: str, img: np.ndarray, source=None) -> Optional[Path]:
    out = projmap.write_png(store.output_path("previews", f"{label}_{suffix}", ".png"), img)
    if out:
        store.record_output("previews", out, label=label, geometry_id=preset.geometry_id,
                            source=source)
    return out


def _save_json(store, preset, label: str, ext: str, doc: dict, source=None) -> Path:
    out = store.output_path("previews", label, ext)
    store.write_json(out, doc)
    store.record_output("previews", out, label=label, geometry_id=preset.geometry_id,
                        source=source)
    return out


def _map_for(session, preset, pmap):
    if pmap is None:
        return projmap.latest(session, preset.geometry_id)
    rules.require_same_geometry(pmap.geometry_id, preset.geometry_id, "zone map")
    return pmap


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _finite_xy(p) -> Optional[list]:
    return [round(float(p[0]), 1), round(float(p[1]), 1)] if np.all(np.isfinite(p)) else None


# ------------------------------------------------------------ the verbs

def project_grid(session, preset, spacing_px: int = DEFAULT_SPACING, label: str = "grid",
                 hold_s: float = 0.0, cancel=None) -> ZoneShot:
    """Project the labelled grid, photograph it, save photo / DMD image (+ the
    camera-space overlay and the visible cells when a ProjectorMap exists).
    hold_s keeps the grid up afterwards (pumped, in slices); setting `cancel`
    (a threading.Event) ends the hold early."""
    safe_label(label)
    hold = _hold(hold_s, preset)
    _check_cancel(cancel)
    projmap.dmd_shape_of(preset)
    spec = grid_spec(spacing_px)
    img = grid_image(spec)
    pmap = _map_for(session, preset, None)
    warnings = []
    photo, sent, inverted, lit, held = _project_and_shoot(session, preset, img, hold, warnings,
                                                          cancel)

    store = session.store
    photo8 = _photo_u8(photo)
    src = pmap.path if pmap is not None else None
    camera_png = _save(store, preset, label, "camera", photo8)
    dmd_png = _save(store, preset, label, "dmd", sent)
    overlay_png, visible, centers, agreement = None, [], {}, None
    if pmap is None:
        warnings.append(f"no projector map for {preset.geometry_id}: the overlay (upright cell "
                        "labels in camera space) and cells_visible need one — run capture_map "
                        "with the scene in place, then project_grid again")
    elif pmap.camera_shape != photo.shape:
        warnings.append(f"projector map {pmap.id} is for a {pmap.camera_shape} camera image, "
                        f"the photo is {photo.shape}: re-run capture_map")
        pmap = src = None
    if pmap is not None:
        ids = _cell_ids(pmap, spec)
        visible, _counts, cen = cells_seen(pmap, spec, ids)
        cells = all_cells(spec)
        mids = np.array([[(x0 + x1) / 2.0, (y0 + y1) / 2.0]
                         for x0, y0, x1, y1 in (cell_rect(c, spec) for c in cells)])
        at = pmap.to_camera(mids)
        # drawn on the flood-lit scene, not the grid photo: the projected
        # labels (possibly mirrored by the optics) would clash with these
        over = np.repeat((_photo_u8(lit) * OVERLAY_TONE).astype(np.uint8)[..., None], 3, axis=2)
        over[_boundaries(ids)] = LINE_BGR
        w, h = _cell_size_camera(pmap, spec.spacing_px)
        thick = max(1, int(round(min(w, h) / 45)))
        scale = _label_scale(visible, w, h, thick)
        vis = set(visible)
        for i, c in enumerate(cells):
            name = cell_name(c, spec)
            if name not in vis:
                continue
            p = at[i] if np.all(np.isfinite(at[i])) else cen[i]
            centers[name] = _finite_xy(p)
            _put_label(over, name, p[0], p[1], scale, thick)
        overlay_png = _save(store, preset, label, "overlay", over, source=src)
        agreement = _map_agreement(pmap, img, photo)
        if agreement is not None and agreement < MAP_AGREEMENT_MIN:
            warnings.append(f"the photo matches projector map {pmap.id} poorly (agreement "
                            f"{agreement:.2f} < {MAP_AGREEMENT_MIN}): the rig, focus or scene "
                            "moved since the map — re-run capture_map before naming zones")
        if not visible:
            warnings.append("the camera sees no grid cell through the projector map")

    doc = {"kind": "grid", "label": label, "geometry_id": preset.geometry_id,
           "preset": preset.name, "mock": bool(preset.mock), "created": _now(),
           "spacing_px": spec.spacing_px, "n_cols": spec.n_cols, "n_rows": spec.n_rows,
           "col_labels": spec.col_labels, "row_labels": spec.row_labels,
           "polarity_inverted": inverted, "hold_s": hold, "held_s": round(held, 3),
           "hold_cancelled": held < hold - 1e-9,
           "camera_png": camera_png, "overlay_png": overlay_png, "dmd_png": dmd_png,
           "map_id": pmap.id if pmap is not None else None, "map_agreement": agreement,
           "cells_visible": visible, "cell_centers": centers,
           "how_to_use": "name cells from the overlay (or the photo): 'C4', 'C4:D6' (block), "
                         "'B2,E5' (union) -> capture_object(..., zone=...)",
           "units": UNITS, "warnings": warnings}
    json_path = _save_json(store, preset, label, GRID_JSON_EXT, doc, source=src)
    return ZoneShot(kind="grid", grid=spec, zones=[cell_name(c, spec) for c in all_cells(spec)],
                    camera_png=camera_png, overlay_png=overlay_png, dmd_png=dmd_png,
                    cells_visible=visible, warnings=warnings, json_path=json_path,
                    cell_centers=centers, polarity_inverted=inverted,
                    map_id=pmap.id if pmap is not None else None,
                    held_s=round(held, 3), hold_cancelled=held < hold - 1e-9)


def _as_zone(z, i: int) -> Zone:
    if isinstance(z, dict):
        names = {f.name for f in fields(Zone)}
        extra = set(z) - names
        if extra:
            raise ValueError(f"zone #{i}: unknown keys {sorted(extra)} (allowed {sorted(names)})")
        try:
            z = Zone(**z)
        except TypeError as e:
            raise ValueError(f"zone #{i}: {e}") from None
    if not isinstance(z, Zone):
        raise ValueError(f"zone #{i} must be a Zone or a dict, got {type(z).__name__}")
    if not isinstance(z.name, str) or not z.name.strip():
        raise ValueError(f"zone #{i}: name must be non-empty text")
    if z.space not in SPACES:
        raise ValueError(f"zone {z.name!r}: space must be one of {SPACES}, got {z.space!r}")
    if z.pattern not in PATTERNS:
        raise ValueError(f"zone {z.name!r}: pattern must be one of {PATTERNS}, got {z.pattern!r}")
    if isinstance(z.level, bool) or not isinstance(z.level, (int, np.integer)) \
            or not 0 <= int(z.level) <= 255:
        raise ValueError(f"zone {z.name!r}: level must be an int 0..255 (DMD grey), got {z.level!r}")
    try:
        pts = np.asarray(z.polygon, dtype=np.float64)
    except (TypeError, ValueError):
        pts = np.empty((0,))
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 2 or not np.isfinite(pts).all():
        raise ValueError(f"zone {z.name!r}: polygon must be >= 3 [x, y] points (or 2 opposite "
                         f"corners of a rectangle), got {z.polygon!r}")
    return z


def _polygon(z: Zone) -> np.ndarray:
    pts = np.asarray(z.polygon, dtype=np.float64)
    if len(pts) == 2:                                   # two corners -> rectangle
        (x0, y0), (x1, y1) = pts
        pts = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    return np.rint(pts).astype(np.int32)


def _raster(poly: np.ndarray, shape: tuple) -> np.ndarray:
    import cv2
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [poly.reshape(-1, 1, 2)], 1)
    return m.astype(bool)


def _pattern(name: str, level: int, mask: np.ndarray) -> np.ndarray:
    """Pattern-space image of one zone's pattern (0 outside the zone)."""
    yy, xx = np.indices(DMD_SHAPE)
    if name == "flood":
        on = mask
    elif name == "outline":
        import cv2
        k = np.ones((2 * OUTLINE_PX + 1, 2 * OUTLINE_PX + 1), np.uint8)
        inner = cv2.erode(mask.astype(np.uint8), k, borderType=cv2.BORDER_CONSTANT,
                          borderValue=0).astype(bool)
        on = mask & ~inner
    elif name == "grid":
        on = mask & (((xx % GRID_PITCH) < GRID_LINE) | ((yy % GRID_PITCH) < GRID_LINE))
    elif name == "stripes":
        on = mask & ((xx // STRIPE_PX) % 2 == 0)
    else:                                               # checker
        on = mask & (((xx // CHECKER_PX) + (yy // CHECKER_PX)) % 2 == 0)
    return np.where(on, np.uint8(level), np.uint8(0))


def project_zones(session, preset, zones: list, pmap=None, label: str = "zones",
                  hold_s: float = 0.0, cancel=None) -> ZoneShot:
    """Light each zone with its pattern (later zones paint over earlier ones),
    photograph it, save photo / DMD image (+ overlay with a map). hold_s
    keeps the zones lit afterwards (pumped, in slices); setting `cancel` (a
    threading.Event) ends the hold early."""
    safe_label(label)
    hold = _hold(hold_s, preset)
    _check_cancel(cancel)
    projmap.dmd_shape_of(preset)
    if not zones:
        raise ValueError("project_zones needs at least one zone")
    zs = [_as_zone(z, i) for i, z in enumerate(zones)]
    pmap = _map_for(session, preset, pmap)
    cam_zones = [z.name for z in zs if z.space == "camera"]
    if cam_zones and pmap is None:
        raise ValueError(f"zone(s) {', '.join(cam_zones)} are in camera space, which needs a "
                         f"projector map for {preset.geometry_id}: run capture_map first (with "
                         "the scene in place), or give the polygon in projector space")
    warnings = []
    physical = np.zeros(DMD_SHAPE, np.uint8)
    masks = {}
    for z in zs:
        poly = _polygon(z)
        if z.space == "camera":
            cm = _raster(poly, pmap.camera_shape)
            unmapped = int((cm & ~pmap.mask).sum())
            m = pmap.projector_mask_for_camera_mask(cm)
            if unmapped and m.any():
                warnings.append(f"zone {z.name!r}: {unmapped} of its camera px are not in the "
                                "projector map (shadow / outside the lit area) and stay dark")
        else:
            m = _raster(poly, DMD_SHAPE)
        if not m.any():
            warnings.append(f"zone {z.name!r} covers no DMD pixel (polygon outside the "
                            f"{'mapped camera area' if z.space == 'camera' else 'DMD'}); "
                            "nothing lit for it")
        masks[z.name] = m
        pat = _pattern(z.pattern, int(z.level), m)
        physical[m] = pat[m]
    photo, sent, inverted, _lit, held = _project_and_shoot(session, preset, physical, hold,
                                                           warnings, cancel)

    store = session.store
    photo8 = _photo_u8(photo)
    src = pmap.path if pmap is not None else None
    camera_png = _save(store, preset, label, "camera", photo8)
    dmd_png = _save(store, preset, label, "dmd", sent)
    overlay_png, centers = None, {}
    if pmap is not None and pmap.camera_shape == photo.shape:
        import cv2
        over = np.repeat((photo8 * OVERLAY_TONE).astype(np.uint8)[..., None], 3, axis=2)
        w, h = _cell_size_camera(pmap, DEFAULT_SPACING)
        thick = max(1, int(round(min(w, h) / 45)))
        scale = _label_scale([z.name for z in zs], w, h, thick)
        for z in zs:
            cm = pmap.camera_mask_for_projector_mask(masks[z.name])
            if not cm.any():
                continue
            cs, _ = cv2.findContours(cm.astype(np.uint8), cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(over, cs, -1, LINE_BGR, max(1, thick))
            v, u = np.nonzero(cm)
            centers[z.name] = [round(float(u.mean()), 1), round(float(v.mean()), 1)]
            _put_label(over, z.name, u.mean(), v.mean(), scale, thick)
        overlay_png = _save(store, preset, label, "overlay", over, source=src)
    elif pmap is None:
        warnings.append(f"no projector map for {preset.geometry_id}: no camera-space overlay")

    doc = {"kind": "zones", "label": label, "geometry_id": preset.geometry_id,
           "preset": preset.name, "mock": bool(preset.mock), "created": _now(),
           "zones": [{"name": z.name, "space": z.space, "pattern": z.pattern,
                      "level": int(z.level), "polygon": np.asarray(z.polygon).tolist(),
                      "dmd_px": int(masks[z.name].sum())} for z in zs],
           "polarity_inverted": inverted, "hold_s": hold, "held_s": round(held, 3),
           "hold_cancelled": held < hold - 1e-9, "camera_png": camera_png,
           "overlay_png": overlay_png, "dmd_png": dmd_png,
           "map_id": pmap.id if pmap is not None else None, "zone_centers": centers,
           "units": {**UNITS, "dmd_px": "DMD px lit per zone",
                     "zone_centers": "camera px [x, y]", "level": "DMD grey 0..255"},
           "warnings": warnings}
    json_path = _save_json(store, preset, label, ZONES_JSON_EXT, doc, source=src)
    return ZoneShot(kind="zones", grid=None, zones=[z.name for z in zs], camera_png=camera_png,
                    overlay_png=overlay_png, dmd_png=dmd_png, warnings=warnings,
                    json_path=json_path, cell_centers=centers, polarity_inverted=inverted,
                    map_id=pmap.id if pmap is not None else None,
                    held_s=round(held, 3), hold_cancelled=held < hold - 1e-9)

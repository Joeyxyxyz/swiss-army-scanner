"""
Virtual sensors — the DMD + camera standing in for industrial point sensors.

An industrial laser-spot sensor is a light spot plus a receiver. The rig has
both: the DMD draws the spot (any size, anywhere, no aiming bracket) and
the camera is the receiver. This module emulates

  * spot sizing        spot_px_for_mm: a spot of D mm at the working plane
                       -> DMD px, like choosing a sensor by its spot size
  * a spot             project_spot: bright spot on a dark field at a grid
                       cell ("H8"), a DMD point {"dmd": [col, row]} or a
                       camera point {"camera": [x, y]} (via the ProjectorMap);
                       photographed and measured (centroid, size, intensity)
  * part present       teach_presence / check_presence, two methods:
      photoelectric    returned light in the receiver window vs the taught
                       EMPTY reference; relative threshold (default 25 %).
                       Blind to a part that reflects like the background.
      bgs              background suppression (triangulation): a surface
                       closer than the taught background shifts the spot
                       along the baseline axis (parallax); the shift ->
                       relief (DMD px) -> height. Indifferent to colour /
                       albedo, so dark and bright parts read the same.
                       A zone spec ("G7:H9") lights a whole block of grid
                       cells instead of a spot (photoelectric only).
  * alignment          teach_alignment / check_alignment: pose (centroid,
                       principal-axis angle, extents) of the part blob in a
                       scan's relief vs a taught reference pose.

Spot size. mm per DMD px at the working plane is the operator's MEASURED
value when given (project a spot, measure it on the part with a rule or
calipers: mm_per_dmd_px = measured_mm / diameter_dmd_px), else NOMINAL
from the rig geometry: standoff_mm / throw_ratio / 912 (the image width
at the standoff over the 912 DMD columns; spot emulation in the seed's
spectral.emulate_spot). The DLP4500 is a DIAMOND array: its 1140 rows sit
at HALF the column pitch (active area 912 x 10.8 um by 1140 x 5.4 um), so
a physically round spot spans twice as many rows as columns. Pattern-space
aspect is not corrected when projection.rotate90 resamples the image (a
warning says so).

Polarity. Every pattern is sent raw (sync.shoot(..., raw=True)) after a
raw black vs raw white measurement: whichever is brighter is "on" — the
bench HDMI path has landed patterns inverted — so the spot is physically
bright on a dark field either way. The physically dark frame is also the
DARK reference: spot signal = photo - dark (ambient and the DMD's black
level cancel, like a modulated sensor's receiver).

Spot measurement. Half-maximum blob of the (lightly smoothed) signal; the
component carrying the most light is the spot. Centroid = signal-weighted;
diameter = equivalent-circle diameter of the half-max area (camera px;
mm only with a lateral calibration for the geometry, rule 6).

Photoelectric signal = sum of the signal over the receiver window / the
taught lit px count: the mean reflected intensity of the lit area, robust
to the spot sliding inside the window. The window is the taught lit area's
bounding box grown by a margin (default 3 spot diameters, >= 32 px).
Verdict: PRESENT if |current / reference - 1| > threshold, else ABSENT
(no hysteresis); INDETERMINATE if the projector does not light the scene,
or the window saturates and the change is still inside the threshold.

BGS. Teach measures the empty spot centroid AND the local Jacobian J
(camera px per DMD px) with two probe spots (+24 columns, +48 rows), so
no ProjectorMap is needed. Check: d = centroid - taught centroid (camera
px); J^-1 d = DMD offset; relief = -(offset along the DMD axis that moves
the camera's baseline axis most) — the engine's relief convention (object
corr - base corr), so a depth calibration applies directly: height_mm =
mm_per_px * relief (slope only: 0 relief is the taught background). No
depth calibration -> the rig's NOMINAL mm per relief px
(standoff^2 / (throw_ratio * 912 * baseline)), labelled "nominal" and
compared as |height| (the sign of a raised surface depends on the rig).
Neither -> relief px only and INDETERMINATE unless min_relief_px is given.
baseline axis = preset.rig.coded_axis (x for a left/right projector).

Alignment. segment.scan_blob on the scan (datum-aware, so zone scans work:
their relief has the base plane at 0) + PCA of the blob px:
centroid (camera px), major-axis angle (deg, image axes: x right, y down,
an undirected axis in (-90, 90]), principal extents. dx/dy/dtheta vs the
reference; mm only with a LATERAL calibration for the geometry (rule 6) —
otherwise px, said so, and the position check needs tol_px.

Physical steps are human-only: teach_presence records the EMPTY reference
and refuses unless the caller passes empty_confirmed=True after the
OPERATOR cleared the sensor's field (in mock mode the MockBench stands in).

Storage (append-only; latest version wins):
    library/sensors/<geometry_id>/<name>/<ts>.json        teach versions
    library/output/sensors/<ts>_<label>_*.png / .json     photos + reports
      (output kind "sensors"; "previews" until the store knows that kind)
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np

from .. import rules
from ..capture import zones as zones_mod
from ..decode import projmap
from ..library.store import safe_label, timestamp

DMD_SHAPE = (1140, 912)            # rows, cols (pattern space)
DMD_COLS = 912
ROW_PITCH = 0.5                    # DLP4500 diamond array: row pitch / column pitch

SHAPES = ("disk", "square")
METHODS = ("photoelectric", "bgs")
PRESENT, ABSENT, INDETERMINATE = "PRESENT", "ABSENT", "INDETERMINATE"
ALIGNED, NOT_ALIGNED = "ALIGNED", "NOT_ALIGNED"

DEFAULT_THRESHOLD = 0.25           # photoelectric: relative change that means "present"
DEFAULT_MIN_HEIGHT_MM = 3.0        # bgs: |height| above the taught background
POLARITY_MARGIN = 2.0              # camera counts (frame mean) between raw white and black
SAT_COUNTS = 250.0                 # 8-bit camera: at/above this a px is saturated
SAT_MAX_FRACTION = 0.02            # of the lit px count
SPOT_MIN_SNR = 8.0                 # spot peak / noise sigma
SPOT_MIN_COUNTS = 4.0
HALF_MAX = 0.5
SPLIT_FRACTION = 0.15              # a 2nd blob with >= this share of the light -> warn
NOISE_FLOOR = 0.3
MIN_REF_COUNTS = 8.0               # photoelectric reference floor (counts per lit px)
WINDOW_SPOT_FACTOR = 3.0           # receiver window margin, in spot diameters
WINDOW_MIN_PX = 32
PROBE_DMD_PX = 24                  # bgs Jacobian probes: +24 columns, +48 rows
OFF_AXIS_WARN = 0.35               # off-axis / along-axis spot shift
ANGLE_AXIS_RATIO_MIN = 1.15        # alignment: sqrt(eig ratio) below this = angle ill-defined
AREA_CHANGE_WARN = 0.25

TARGET_DECIMALS = 2                # DMD px; spot targets are stored (and re-drawn) at this

SENSORS_DIR = "sensors"
OUTPUT_KIND = "sensors"
FALLBACK_OUTPUT_KIND = "previews"

SPOT_BGR, EXPECT_BGR, WINDOW_BGR = (0, 255, 255), (255, 255, 0), (0, 200, 0)

SPEC_UNITS = {"diameter_mm": "mm at the working plane",
              "diameter_dmd_px": "DMD px (mirror columns)",
              "diameter_dmd_rows": "DMD rows (diamond array: 1/2 column pitch)",
              "mm_per_dmd_px": "mm per DMD column at the working plane"}
SPOT_UNITS = {"target_dmd": "DMD px [col, row] (pattern space)",
              "centroid_px": "camera px [x, y]", "expected_px": "camera px [x, y]",
              "offset_px": "camera px [x, y] (centroid - expected)",
              "diameter_px": "camera px (equivalent-circle diameter of the half-max area)",
              "extent_px": "camera px [w, h] (half-max bbox)", "area_px": "camera px",
              "diameter_mm": "mm (lateral calibration)",
              "peak_counts": "camera counts above the dark frame",
              "mean_counts": "camera counts above the dark frame (half-max area mean)",
              "snr": "peak / noise sigma", "level": "DMD grey 0..255"}
PRESENCE_UNITS = {"signal": "camera counts per lit px (mean reflected intensity)",
                  "reference_signal": "camera counts per lit px (taught, empty)",
                  "relative_change": "fraction (signal / reference - 1)",
                  "threshold": "fraction", "window": "camera px [x0, y0, x1, y1)",
                  "centroid_px": "camera px [x, y]", "reference_centroid_px": "camera px [x, y]",
                  "shift_px": "camera px along the baseline axis (rig.coded_axis)",
                  "off_axis_px": "camera px across the baseline axis",
                  "relief_px": "DMD px (projector columns/rows; engine relief convention)",
                  "height_mm": "mm (see height_source: calibrated | nominal)",
                  "mm_per_relief_px": "mm per DMD px of relief",
                  "min_height_mm": "mm", "jacobian": "camera px per DMD px "
                  "[[dx/dcol, dx/drow], [dy/dcol, dy/drow]]",
                  "saturated_fraction": "fraction of the lit px count"}
ALIGN_UNITS = {"centroid_px": "camera px [x, y]", "angle_deg": "deg (major axis, image axes, "
               "x right / y down, (-90, 90])", "extents_px": "camera px [major, minor]",
               "area_px": "camera px", "dx_px": "camera px", "dy_px": "camera px",
               "offset_px": "camera px", "dx_mm": "mm (lateral calibration)",
               "dy_mm": "mm (lateral calibration)", "offset_mm": "mm (lateral calibration)",
               "dtheta_deg": "deg", "tol_mm": "mm", "tol_deg": "deg", "tol_px": "camera px",
               "relief_px": "DMD px (median over the blob)"}


# ------------------------------------------------------------ dataclasses

@dataclass
class SpotSpec:
    diameter_mm: float
    diameter_dmd_px: float              # across DMD columns
    mm_per_dmd_px: float                # per DMD column, at the working plane
    source: str                         # "measured" | "nominal"
    diameter_dmd_rows: float = 0.0      # across DMD rows (twice the columns: diamond array)
    standoff_mm: Optional[float] = None
    throw_ratio: Optional[float] = None
    warnings: list = field(default_factory=list)
    units: dict = field(default_factory=lambda: dict(SPEC_UNITS))


@dataclass
class SpotReading:
    label: str
    at: dict                            # how the spot was placed (normalized)
    target_dmd: list                    # [col, row] DMD px
    spec: SpotSpec
    shape: str
    level: int
    found: bool
    centroid_px: Optional[list] = None
    expected_px: Optional[list] = None  # from the ProjectorMap (or the camera point asked for)
    offset_px: Optional[list] = None
    diameter_px: Optional[float] = None
    extent_px: Optional[list] = None
    area_px: int = 0
    diameter_mm: Optional[float] = None         # lateral calibration only (rule 6)
    peak_counts: Optional[float] = None
    mean_counts: Optional[float] = None
    snr: Optional[float] = None
    saturated_px: int = 0
    polarity_inverted: Optional[bool] = None
    map_id: Optional[str] = None
    calibration_id: Optional[str] = None
    photo_png: Optional[Path] = None
    overlay_png: Optional[Path] = None
    json_path: Optional[Path] = None
    geometry_id: str = ""
    mock: Optional[bool] = None
    created: str = ""
    warnings: list = field(default_factory=list)
    units: dict = field(default_factory=lambda: dict(SPOT_UNITS))


@dataclass
class PresenceTeach:
    name: str
    method: str                         # "photoelectric" | "bgs"
    mode: str                           # "spot" | "zone"
    at: dict
    geometry_id: str
    reference_signal: float             # counts per lit px, EMPTY scene
    n_px: int                           # lit camera px (spot half-max area / zone)
    window: list                        # receiver window [x0, y0, x1, y1) camera px
    threshold: float
    min_height_mm: float
    target_dmd: Optional[list] = None
    zone: Optional[str] = None
    cells: Optional[list] = None
    spacing_px: Optional[int] = None
    spot: Optional[SpotSpec] = None
    shape: str = "disk"
    level: int = 255
    reference_centroid_px: Optional[list] = None
    reference_saturated_px: int = 0
    jacobian: Optional[list] = None     # bgs: camera px per DMD px
    depth_axis: Optional[str] = None    # bgs: "col" | "row" (DMD axis carrying relief)
    baseline_axis: str = "x"            # camera axis of the parallax
    noise_counts: float = 0.0
    polarity_inverted: Optional[bool] = None
    photo_png: Optional[Path] = None
    path: Optional[Path] = None         # the sensor's folder (all versions)
    version_path: Optional[Path] = None
    created: str = ""
    mock: Optional[bool] = None
    warnings: list = field(default_factory=list)
    units: dict = field(default_factory=lambda: dict(PRESENCE_UNITS))


@dataclass
class PresenceResult:
    name: str
    method: str
    verdict: str                        # PRESENT | ABSENT | INDETERMINATE
    present: Optional[bool]
    signal: Optional[float] = None
    reference_signal: Optional[float] = None
    relative_change: Optional[float] = None
    threshold: Optional[float] = None
    centroid_px: Optional[list] = None
    reference_centroid_px: Optional[list] = None
    shift_px: Optional[float] = None
    off_axis_px: Optional[float] = None
    relief_px: Optional[float] = None
    height_mm: Optional[float] = None
    height_source: Optional[str] = None     # "calibrated" | "nominal" | None
    mm_per_relief_px: Optional[float] = None
    min_height_mm: Optional[float] = None
    saturated_fraction: float = 0.0
    polarity_inverted: Optional[bool] = None
    calibration_id: Optional[str] = None
    teach_version: Optional[Path] = None
    photo_png: Optional[Path] = None
    json_path: Optional[Path] = None
    geometry_id: str = ""
    created: str = ""
    warnings: list = field(default_factory=list)
    units: dict = field(default_factory=lambda: dict(PRESENCE_UNITS))


@dataclass
class AlignmentRef:
    name: str
    geometry_id: str
    scan_id: str
    centroid_px: list
    angle_deg: float
    extents_px: list
    area_px: int
    axis_ratio: float
    relief_px: float
    path: Optional[Path] = None
    version_path: Optional[Path] = None
    created: str = ""
    mock: Optional[bool] = None
    warnings: list = field(default_factory=list)
    units: dict = field(default_factory=lambda: dict(ALIGN_UNITS))


@dataclass
class AlignmentResult:
    name: str
    verdict: str                        # ALIGNED | NOT_ALIGNED | INDETERMINATE
    position_units: str                 # "mm" | "px"
    dx_px: Optional[float] = None
    dy_px: Optional[float] = None
    offset_px: Optional[float] = None
    dtheta_deg: Optional[float] = None
    dx_mm: Optional[float] = None
    dy_mm: Optional[float] = None
    offset_mm: Optional[float] = None
    tol_mm: Optional[float] = None
    tol_deg: Optional[float] = None
    tol_px: Optional[float] = None
    position_ok: Optional[bool] = None
    angle_ok: Optional[bool] = None
    centroid_px: Optional[list] = None
    angle_deg: Optional[float] = None
    ref_centroid_px: Optional[list] = None
    ref_angle_deg: Optional[float] = None
    area_ratio: Optional[float] = None
    lateral_px_per_mm: Optional[float] = None
    calibration_id: Optional[str] = None
    scan_id: str = ""
    ref_version: Optional[Path] = None
    overlay_png: Optional[Path] = None
    json_path: Optional[Path] = None
    geometry_id: str = ""
    created: str = ""
    warnings: list = field(default_factory=list)
    units: dict = field(default_factory=lambda: dict(ALIGN_UNITS))


# ------------------------------------------------------------ small helpers

def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _positive(v, what: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating)) \
            or not math.isfinite(float(v)) or float(v) <= 0:
        raise ValueError(f"{what} must be a positive number, got {v!r}")
    return float(v)


def _fraction(v, what: str) -> float:
    f = _positive(v, what)
    if f >= 10:
        raise ValueError(f"{what} is a fraction (0.25 = 25 %), got {v!r}")
    return f


def _level(level) -> int:
    if isinstance(level, bool) or not isinstance(level, (int, np.integer)) \
            or not 1 <= int(level) <= 255:
        raise ValueError(f"level must be an int 1..255 (DMD grey), got {level!r}")
    return int(level)


def _name(name) -> str:
    if not isinstance(name, str) or not name.strip():
        raise ValueError("sensor name must be non-empty text")
    safe_label(name)                        # fails on names with no usable character
    return name.strip()


def _xy(p) -> Optional[list]:
    if p is None:
        return None
    a = np.asarray(p, np.float64)
    return [round(float(a[0]), 2), round(float(a[1]), 2)] if np.all(np.isfinite(a)) else None


def _r(v, nd: int = 4) -> Optional[float]:
    return None if v is None or not math.isfinite(float(v)) else round(float(v), nd)


def _calibration(session, preset):
    """The calibration for the preset's geometry (library first, then a
    preset-supplied one) — rule 6: never one from another geometry."""
    from .. import config
    # same lookup as api.calibration_for: only a calibration fitted at the same
    # geometry-bound settings (fingerprint) may supply mm
    cal = session.store.latest_calibration(preset.geometry_id,
                                           config.geometry_fingerprint(preset))
    if cal is None and rules.calibration_matches(preset.calibration, preset.geometry_id) \
            and getattr(preset.calibration, "source", "") == "preset":
        cal = preset.calibration
    return cal


# ------------------------------------------------------------ spot sizing

def spot_px_for_mm(preset, diameter_mm, mm_per_dmd_px=None) -> SpotSpec:
    """A spot of `diameter_mm` at the working plane -> DMD px (columns; rows
    are twice that — diamond array). mm per DMD px: the operator's MEASURED
    value when given, else nominal standoff_mm / throw_ratio / 912."""
    d = _positive(diameter_mm, "diameter_mm")
    rig = preset.rig
    warnings = []
    if mm_per_dmd_px is not None:
        k = _positive(mm_per_dmd_px, "mm_per_dmd_px")
        source = "measured"
    else:
        if not rig.standoff_mm:
            raise ValueError(
                "rig.standoff_mm is not set in the preset, so there is no nominal spot scale: "
                "measure it (project a spot, measure it on the part: mm_per_dmd_px = "
                "measured_mm / diameter_dmd_px) and pass mm_per_dmd_px, or set rig.standoff_mm")
        k = float(rig.standoff_mm) / float(rig.throw_ratio) / DMD_COLS
        source = "nominal"
        warnings.append(f"nominal spot scale {k:.4f} mm per DMD px from rig.standoff_mm "
                        f"{rig.standoff_mm:g} / throw_ratio {rig.throw_ratio:g} / {DMD_COLS} — "
                        "not measured; measure the spot on the part and pass mm_per_dmd_px")
    cols = d / k
    rows = cols / ROW_PITCH
    if cols > DMD_COLS / 2:
        raise ValueError(f"a {d:g} mm spot is {cols:.0f} DMD px across — more than half the "
                         f"DMD ({DMD_COLS} px); use a zone instead")
    if cols < 1.0:
        warnings.append(f"{d:g} mm is {cols:.2f} DMD px: below one mirror column, so one column "
                        f"({k:.3f} mm) is lit and the optics blur it")
    if getattr(preset.projection, "rotate90", False):
        warnings.append("projection.rotate90 resamples the pattern before the DMD: the spot's "
                        "aspect is not corrected (diamond rows) — measure it")
    return SpotSpec(diameter_mm=d, diameter_dmd_px=round(cols, 4), mm_per_dmd_px=k,
                    source=source, diameter_dmd_rows=round(rows, 4),
                    standoff_mm=rig.standoff_mm, throw_ratio=rig.throw_ratio,
                    warnings=warnings)


def spot_image(target, spec: SpotSpec, shape: str = "disk", level: int = 255) -> np.ndarray:
    """uint8 DMD_SHAPE image (physical: bright = lit) of one spot centred on
    target (col, row) — pixel index convention: mirror i is centred at i."""
    if shape not in SHAPES:
        raise ValueError(f"shape must be one of {SHAPES}, got {shape!r}")
    lv = _level(level)
    H, W = DMD_SHAPE
    tc, tr = (float(v) for v in target)
    rx = max(float(spec.diameter_dmd_px) / 2.0, 0.5)
    ry = max(float(spec.diameter_dmd_rows or spec.diameter_dmd_px / ROW_PITCH) / 2.0, 0.5)
    c0, c1 = max(int(math.floor(tc - rx)), 0), min(int(math.ceil(tc + rx)) + 1, W)
    r0, r1 = max(int(math.floor(tr - ry)), 0), min(int(math.ceil(tr + ry)) + 1, H)
    img = np.zeros(DMD_SHAPE, np.uint8)
    if c0 < c1 and r0 < r1:
        cc, rr = np.meshgrid(np.arange(c0, c1, dtype=np.float64),
                             np.arange(r0, r1, dtype=np.float64))
        if shape == "disk":
            on = ((cc - tc) / rx) ** 2 + ((rr - tr) / ry) ** 2 <= 1.0 + 1e-9
        else:
            on = (np.abs(cc - tc) <= rx + 1e-9) & (np.abs(rr - tr) <= ry + 1e-9)
        img[r0:r1, c0:c1][on] = lv
    if not img.any():
        img[int(np.clip(round(tr), 0, H - 1)), int(np.clip(round(tc), 0, W - 1))] = lv
    return img


# ------------------------------------------------------------ where ("at")

def _is_zone_text(text: str, cells: list) -> bool:
    t = text.strip()
    return len(cells) > 1 or t.lower() == "all" or bool(re.search(r"[:,;\s]", t))


def _grid_for(session, preset, text: str):
    body, spacing = zones_mod._split_spacing(text)
    if spacing is None:
        spacing = zones_mod.last_grid_spacing(session, preset.geometry_id) \
            or zones_mod.DEFAULT_SPACING
    grid = zones_mod.grid_spec(spacing)
    return body, grid, zones_mod.parse_zone(body, grid)


def _resolve_at(session, preset, at, allow_zone: bool) -> dict:
    """-> {"mode": "spot", "target": (col, row), "at": {...}} or
          {"mode": "zone", "proj_mask", "cam_mask", "cells", "spacing_px", "zone", "at"}."""
    force = None
    if isinstance(at, dict):
        if len(at) != 1:
            raise ValueError(f"at must have exactly one key ('dmd', 'camera', 'cell' or 'zone'), "
                             f"got {sorted(at)}")
        (key, val), = at.items()
        if key in ("cell", "zone"):
            force, at = key, val
        elif key == "dmd":
            try:
                c, r = (float(v) for v in val)
            except (TypeError, ValueError):
                raise ValueError(f"at={{'dmd': ...}} needs [col, row] DMD px, got {val!r}") \
                    from None
            if not (0 <= c <= DMD_SHAPE[1] - 1 and 0 <= r <= DMD_SHAPE[0] - 1):
                raise ValueError(f"DMD point {[c, r]} is outside the DMD (cols 0..911, "
                                 "rows 0..1139)")
            # rounded as stored, so a check re-draws exactly the taught spot
            c, r = round(c, TARGET_DECIMALS), round(r, TARGET_DECIMALS)
            return {"mode": "spot", "target": (c, r), "at": {"dmd": [c, r]}}
        elif key == "camera":
            try:
                x, y = (float(v) for v in val)
            except (TypeError, ValueError):
                raise ValueError(f"at={{'camera': ...}} needs [x, y] camera px, got {val!r}") \
                    from None
            pmap = projmap.latest(session, preset.geometry_id)
            if pmap is None:
                raise ValueError(f"a camera point needs a projector map for {preset.geometry_id}: "
                                 "run capture_map (scene in place) first, or give a grid cell "
                                 "('H8') or a DMD point {'dmd': [col, row]}")
            H, W = pmap.camera_shape
            if not (0 <= x <= W - 1 and 0 <= y <= H - 1):
                raise ValueError(f"camera point {[x, y]} is outside the {W}x{H} camera image")
            p = pmap.to_projector([x, y])
            if not np.all(np.isfinite(p)):
                raise ValueError(f"camera point {[x, y]} is not in projector map {pmap.id} "
                                 "(shadow / outside the lit area): pick a lit point")
            c, r = round(float(p[0]), TARGET_DECIMALS), round(float(p[1]), TARGET_DECIMALS)
            return {"mode": "spot", "target": (c, r),
                    "at": {"camera": [x, y], "dmd": [c, r], "map_id": pmap.id},
                    "camera_point": [x, y], "pmap": pmap}
        else:
            raise ValueError(f"at key must be 'dmd', 'camera', 'cell' or 'zone', got {key!r}")
    if not isinstance(at, str):
        raise ValueError(f"at must be a grid cell ('H8'), a zone ('G7:H9'), {{'dmd': [col, row]}} "
                         f"or {{'camera': [x, y]}}, got {at!r}")
    body, grid, cells = _grid_for(session, preset, at)
    zone = force == "zone" or (force is None and _is_zone_text(body, cells))
    if not zone:
        if len(cells) != 1:
            raise ValueError(f"{at!r} names {len(cells)} cells; a spot needs one cell like 'H8'")
        x0, y0, x1, y1 = zones_mod.cell_rect(cells[0], grid)
        target = ((x0 + x1 - 1) / 2.0, (y0 + y1 - 1) / 2.0)       # centre mirror (index convention)
        return {"mode": "spot", "target": target,
                "at": {"cell": zones_mod.cell_name(cells[0], grid),
                       "spacing_px": grid.spacing_px}}
    if not allow_zone:
        raise ValueError(f"{at!r} is a zone; a spot needs one grid cell ('H8'), a DMD point or a "
                         "camera point (zones: photoelectric presence only)")
    text = f"{body}@{grid.spacing_px}"
    pm, cam, cells = zones_mod.resolve(session, preset, text)
    return {"mode": "zone", "proj_mask": pm, "cam_mask": cam, "cells": [list(c) for c in cells],
            "spacing_px": grid.spacing_px, "zone": body.strip(),
            "at": {"zone": body.strip(), "spacing_px": grid.spacing_px}}


# ------------------------------------------------------------ the rig

@dataclass
class _Shots:
    photos: list
    dark: np.ndarray            # physically dark frame (DMD all off)
    white: np.ndarray           # physically full-on frame
    inverted: bool
    lit_ok: bool
    warnings: list


def _shoot(session, preset, physicals: list, hold: float) -> _Shots:
    """Raw black + raw white (polarity MEASURED), then each physical image
    sent so it is physically bright where asked. One rig session."""
    warnings = []
    with session.rig(preset) as sync:
        f0 = sync.shoot(np.zeros(DMD_SHAPE, np.uint8), raw=True)
        f1 = sync.shoot(np.full(DMD_SHAPE, 255, np.uint8), raw=True)
        diff = float(f1.mean()) - float(f0.mean())
        lit_ok = abs(diff) >= POLARITY_MARGIN
        inverted = lit_ok and diff < 0
        photos = [np.asarray(sync.shoot((255 - p) if inverted else p, raw=True), np.float32)
                  for p in physicals]
        if hold > 0:
            sync.projector.window.hold(hold)        # pumped hold: the spot stays painted
    if not lit_ok:
        warnings.append(f"raw white and raw black look the same to the camera ({diff:+.1f} "
                        "counts): is the projector on and aimed at the scene?")
    elif inverted:
        warnings.append("raw HDMI path measured INVERTED; patterns were sent inverted so they "
                        "are physically bright-on-dark")
    dark, white = (f1, f0) if inverted else (f0, f1)
    return _Shots(photos, np.asarray(dark, np.float32), np.asarray(white, np.float32),
                  inverted, lit_ok, warnings)


def _noise(diff: np.ndarray) -> float:
    """Noise sigma of a difference image from its high-pass residual (robust
    to how much of the frame is lit)."""
    import cv2
    res = diff - cv2.blur(diff, (3, 3))
    s = res[::2, ::2].ravel()
    mad = float(np.median(np.abs(s - np.median(s)))) * 1.4826
    return max(mad / 0.943, NOISE_FLOOR)            # box-3 residual keeps sqrt(8/9) of sigma


def _measure_spot(diff: np.ndarray, photo: np.ndarray, noise: float) -> Optional[dict]:
    """Half-max blob carrying the most light -> centroid / size / intensity."""
    import cv2
    sm = cv2.blur(diff, (3, 3))
    peak = float(sm.max())
    if peak < max(SPOT_MIN_SNR * noise, SPOT_MIN_COUNTS):
        return None
    mask = (sm >= HALF_MAX * peak).astype(np.uint8)
    n, lab = cv2.connectedComponents(mask, connectivity=8)
    light = np.clip(diff, 0, None)
    sums = np.bincount(lab.ravel(), weights=light.ravel(), minlength=n)
    sums[0] = 0.0
    k = int(np.argmax(sums))
    comp = lab == k
    ys, xs = np.nonzero(comp)
    w = light[ys, xs].astype(np.float64)
    if w.sum() <= 0:
        return None
    cx, cy = float((xs * w).sum() / w.sum()), float((ys * w).sum() / w.sum())
    area = int(comp.sum())
    others = np.sort(sums[1:])[::-1]
    split = int((others[1:] >= SPLIT_FRACTION * sums[k]).sum()) if len(others) > 1 else 0
    vals = diff[comp]
    return {"centroid": [cx, cy], "area": area, "diameter": 2.0 * math.sqrt(area / math.pi),
            "bbox": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
            "peak": float(vals.max()), "mean": float(vals.mean()), "snr": peak / noise,
            "saturated": int((photo[comp] >= SAT_COUNTS).sum()), "split": split, "mask": comp}


def _lit_mask(diff: np.ndarray, noise: float) -> np.ndarray:
    """Pixels lit by a flood (zone without a map): >= half the typical lit level."""
    import cv2
    sm = cv2.blur(diff, (3, 3))
    sig = sm[sm > SPOT_MIN_SNR * noise]
    if sig.size < 16:
        return np.zeros(diff.shape, bool)
    return sm >= HALF_MAX * float(np.percentile(sig, 90))


def _grow(bbox, margin: int, shape) -> list:
    H, W = shape
    x0, y0, x1, y1 = bbox
    m = int(margin)
    return [max(int(x0) - m, 0), max(int(y0) - m, 0), min(int(x1) + m, W), min(int(y1) + m, H)]


def _window_signal(diff, photo, window, n_px: int) -> tuple:
    x0, y0, x1, y1 = window
    s = float(diff[y0:y1, x0:x1].sum()) / max(int(n_px), 1)
    sat = int((photo[y0:y1, x0:x1] >= SAT_COUNTS).sum())
    return s, sat / max(int(n_px), 1)


# ------------------------------------------------------------ output bin

def _output_path(store, label: str, ext: str):
    """A fresh path in output/sensors/ — or output/previews/ while the store
    does not know the "sensors" kind yet. -> (path, kind)."""
    safe_label(label)                   # a bad label must not look like an unknown kind
    try:
        return store.output_path(OUTPUT_KIND, label, ext), OUTPUT_KIND
    except ValueError:
        return store.output_path(FALLBACK_OUTPUT_KIND, label, ext), FALLBACK_OUTPUT_KIND


def _save_png(store, preset, label: str, img: np.ndarray, source=None) -> Optional[Path]:
    path, kind = _output_path(store, label, ".png")
    out = projmap.write_png(path, img)          # exclusive create (rule 7)
    if out:
        store.record_output(kind, out, label=label, geometry_id=preset.geometry_id, source=source)
    return out


def _save_json(store, preset, label: str, doc: dict, source=None) -> Path:
    path, kind = _output_path(store, label, ".json")
    store.write_json(path, doc)
    store.record_output(kind, path, label=label, geometry_id=preset.geometry_id, source=source)
    return path


def _photo_u8(photo: np.ndarray) -> np.ndarray:
    f = np.asarray(photo, np.float32)
    hi = float(f.max()) if f.size else 0.0
    if hi > 255.0:
        f = f * (255.0 / hi)
    return np.clip(np.rint(f), 0, 255).astype(np.uint8)


def _overlay(photo: np.ndarray, marks: list) -> np.ndarray:
    """BGR copy of the photo with marks: ("circle", xy, r, bgr), ("cross", xy, bgr),
    ("rect", [x0, y0, x1, y1], bgr), ("axis", xy, angle_deg, half_len, bgr)."""
    import cv2
    img = np.repeat(_photo_u8(photo)[..., None], 3, axis=2)
    for m in marks:
        kind = m[0]
        if kind == "circle" and m[1] is not None:
            cv2.circle(img, (int(round(m[1][0])), int(round(m[1][1]))), max(int(round(m[2])), 3),
                       m[3], 1, cv2.LINE_AA)
        elif kind == "cross" and m[1] is not None:
            x, y = int(round(m[1][0])), int(round(m[1][1]))
            cv2.line(img, (x - 8, y), (x + 8, y), m[2], 1, cv2.LINE_AA)
            cv2.line(img, (x, y - 8), (x, y + 8), m[2], 1, cv2.LINE_AA)
        elif kind == "rect" and m[1] is not None:
            x0, y0, x1, y1 = (int(v) for v in m[1])
            cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), m[2], 1)
        elif kind == "axis" and m[1] is not None:
            (x, y), a, h = m[1], math.radians(m[2]), m[3]
            p0 = (int(round(x - h * math.cos(a))), int(round(y - h * math.sin(a))))
            p1 = (int(round(x + h * math.cos(a))), int(round(y + h * math.sin(a))))
            cv2.line(img, p0, p1, m[4], 2, cv2.LINE_AA)
            cv2.circle(img, (int(round(x)), int(round(y))), 4, m[4], -1, cv2.LINE_AA)
    return img


# ------------------------------------------------------------ project_spot

def project_spot(session, preset, at, diameter_mm, mm_per_dmd_px=None, shape: str = "disk",
                 hold_s: float = 0.0, label: str = "spot", level: int = 255) -> SpotReading:
    """Project one bright spot of diameter_mm on a dark field, photograph it
    and measure it in the camera. `at`: grid cell "H8" (its centre; "H8@64"
    names a cell of a 64 px grid), {"dmd": [col, row]} or {"camera": [x, y]}
    (needs a ProjectorMap). hold_s keeps the spot painted afterwards (aiming)."""
    safe_label(label)
    hold = zones_mod._hold(hold_s, preset, "spot hold")     # + rule 3 duration check
    projmap.dmd_shape_of(preset)
    if shape not in SHAPES:
        raise ValueError(f"shape must be one of {SHAPES}, got {shape!r}")
    lv = _level(level)
    loc = _resolve_at(session, preset, at, allow_zone=False)
    spec = spot_px_for_mm(preset, diameter_mm, mm_per_dmd_px)
    physical = spot_image(loc["target"], spec, shape, lv)
    pmap = loc.get("pmap") or projmap.latest(session, preset.geometry_id)
    sh = _shoot(session, preset, [physical], hold)
    photo = sh.photos[0]
    warnings = list(spec.warnings) + sh.warnings
    diff = photo - sh.dark
    noise = _noise(diff)
    spot = _measure_spot(diff, photo, noise) if sh.lit_ok else None

    expected = None
    if "camera_point" in loc:
        expected = _xy(loc["camera_point"])
    elif pmap is not None and pmap.camera_shape == photo.shape:
        expected = _xy(pmap.to_camera(list(loc["target"])))
        if expected is None:
            warnings.append(f"projector map {pmap.id} does not see DMD point "
                            f"{_xy(loc['target'])}: no expected camera location")
    elif pmap is None:
        warnings.append(f"no projector map for {preset.geometry_id}: no expected camera location "
                        "(run capture_map to predict where the spot lands)")
    cal = _calibration(session, preset)
    reading = SpotReading(label=label, at=loc["at"], target_dmd=_xy(loc["target"]), spec=spec,
                          shape=shape, level=lv, found=spot is not None, expected_px=expected,
                          polarity_inverted=sh.inverted if sh.lit_ok else None,
                          map_id=pmap.id if pmap is not None else None,
                          geometry_id=preset.geometry_id, mock=bool(preset.mock), created=_now(),
                          warnings=warnings)
    if spot is None:
        if sh.lit_ok:
            warnings.append(f"no spot found (peak below {SPOT_MIN_SNR:g} x noise): the spot is "
                            "off the lit scene, on a black / specular surface, or in a shadow")
    else:
        reading.centroid_px = _xy(spot["centroid"])
        reading.diameter_px = _r(spot["diameter"], 3)
        x0, y0, x1, y1 = spot["bbox"]
        reading.extent_px = [x1 - x0, y1 - y0]
        reading.area_px = spot["area"]
        reading.peak_counts = _r(spot["peak"], 2)
        reading.mean_counts = _r(spot["mean"], 2)
        reading.snr = _r(spot["snr"], 1)
        reading.saturated_px = spot["saturated"]
        if expected is not None:
            reading.offset_px = _xy(np.subtract(spot["centroid"], expected))
        if cal is not None and cal.has_xy:
            reading.diameter_mm = _r(spot["diameter"] / cal.lateral_px_per_mm, 3)
            reading.calibration_id = cal.id
        else:
            warnings.append(f"no lateral calibration for {preset.geometry_id}: the measured spot "
                            "size is in camera px only (rule 6)")
        if spot["saturated"]:
            warnings.append(f"{spot['saturated']} spot px saturated: lower `level` or the "
                            "exposure for true intensities")
        if spot["split"]:
            warnings.append(f"the spot is split in {spot['split'] + 1} pieces (it straddles an "
                            "edge or a shadow); the brightest piece was measured")

    store = session.store
    src = pmap.path if pmap is not None else None
    reading.photo_png = _save_png(store, preset, f"{label}_photo", _photo_u8(photo))
    marks = [("cross", expected, EXPECT_BGR)]
    if spot is not None:
        marks.append(("circle", spot["centroid"], spot["diameter"] / 2.0, SPOT_BGR))
    reading.overlay_png = _save_png(store, preset, f"{label}_overlay", _overlay(photo, marks),
                                    source=src)
    doc = {"kind": "spot", "preset": preset.name, **_doc(reading)}
    reading.json_path = _save_json(store, preset, label, doc, source=src)
    return reading


def _doc(obj) -> dict:
    """JSON-ready dict of a result dataclass (Paths as text)."""
    d = asdict(obj)
    return json.loads(json.dumps(d, default=lambda o: str(o) if isinstance(o, Path) else
                                 (o.tolist() if isinstance(o, np.ndarray) else
                                  (o.item() if isinstance(o, np.generic) else repr(o)))))


# ------------------------------------------------------------ sensor store

def _sensors_root(session) -> Path:
    return session.store.root / SENSORS_DIR


def _sensor_dir(session, geometry_id: str, name: str) -> Path:
    return _sensors_root(session) / geometry_id / safe_label(name)


_VERSION_RE = re.compile(r"^\d{8}T\d{6}_\d{6}(-\d+)?\.json$")


def _version_key(p: Path):
    stem, _, n = p.stem.partition("-")
    return (stem, int(n) if n.isdigit() else 0)


def _versions(d: Path) -> list:
    if not d.is_dir():
        return []
    return sorted((p for p in d.iterdir() if p.is_file() and _VERSION_RE.match(p.name)),
                  key=_version_key)


def _write_version(session, geometry_id: str, name: str, doc: dict) -> Path:
    d = _sensor_dir(session, geometry_id, name)
    d.mkdir(parents=True, exist_ok=True)
    stem = timestamp()
    for n in range(1000):
        p = d / (f"{stem}.json" if n == 0 else f"{stem}-{n}.json")
        if not p.exists():
            session.store.write_json(p, doc)            # exclusive create ("x")
            return p
    raise rules.OverwriteRefused(f"could not allocate a new sensor version under {d}")


def _latest_doc(session, geometry_id: str, name: str, kind: Optional[str] = None):
    """-> (doc, version path). Clear errors for a wrong geometry / kind."""
    vs = _versions(_sensor_dir(session, geometry_id, name))
    if not vs:
        elsewhere = [g.name for g in _sensors_root(session).iterdir()
                     if g.is_dir() and _versions(g / safe_label(name))] \
            if _sensors_root(session).is_dir() else []
        if elsewhere:
            raise rules.GeometryMismatch(
                f"sensor {name!r} was taught at geometry {', '.join(elsewhere)}, not "
                f"{geometry_id}; a sensor is valid only at the rig geometry it was taught at — "
                "teach it again here")
        raise ValueError(f"no sensor named {name!r} for {geometry_id}; teach it first "
                         "(list_sensors shows the taught ones)")
    doc = json.loads(vs[-1].read_text(encoding="utf-8"))
    if kind is not None and doc.get("kind") != kind:
        raise ValueError(f"sensor {name!r} is a {doc.get('kind')} sensor, not {kind}")
    return doc, vs[-1]


def _guard_kind(session, geometry_id: str, name: str, kind: str) -> None:
    vs = _versions(_sensor_dir(session, geometry_id, name))
    if vs:
        old = json.loads(vs[-1].read_text(encoding="utf-8")).get("kind")
        if old != kind:
            raise ValueError(f"sensor {name!r} already exists as a {old} sensor at "
                             f"{geometry_id}; pick another name")


def list_sensors(session, geometry_id: Optional[str] = None) -> list:
    """Latest version of every taught sensor (summaries, oldest name first)."""
    root = _sensors_root(session)
    if not root.is_dir():
        return []
    geos = [geometry_id] if geometry_id else sorted(p.name for p in root.iterdir() if p.is_dir())
    out = []
    for g in geos:
        gdir = root / g
        if not gdir.is_dir():
            continue
        for d in sorted(p for p in gdir.iterdir() if p.is_dir()):
            vs = _versions(d)
            if not vs:
                continue
            doc = json.loads(vs[-1].read_text(encoding="utf-8"))
            row = {"name": doc.get("name", d.name), "kind": doc.get("kind"),
                   "method": doc.get("method"), "mode": doc.get("mode"),
                   "geometry_id": doc.get("geometry_id", g), "at": doc.get("at"),
                   "created": doc.get("created", ""), "versions": len(vs), "path": d,
                   "version_path": vs[-1], "mock": doc.get("mock"),
                   "warnings": doc.get("warnings", [])}
            if doc.get("kind") == "presence":
                row.update({"reference_signal": doc.get("reference_signal"),
                            "threshold": doc.get("threshold"),
                            "min_height_mm": doc.get("min_height_mm"),
                            "units": {"reference_signal": PRESENCE_UNITS["reference_signal"],
                                      "threshold": "fraction", "min_height_mm": "mm"}})
            else:
                row.update({"centroid_px": doc.get("centroid_px"),
                            "angle_deg": doc.get("angle_deg"),
                            "units": {"centroid_px": ALIGN_UNITS["centroid_px"],
                                      "angle_deg": ALIGN_UNITS["angle_deg"]}})
            out.append(row)
    return out


def load_sensor(session, preset, name: str):
    """Latest taught version -> PresenceTeach | AlignmentRef."""
    doc, vpath = _latest_doc(session, preset.geometry_id, _name(name))
    return _teach_from_doc(doc, vpath) if doc.get("kind") == "presence" \
        else _align_from_doc(doc, vpath)


def _teach_from_doc(doc: dict, vpath: Path) -> PresenceTeach:
    spot = SpotSpec(**doc["spot"]) if doc.get("spot") else None
    return PresenceTeach(
        name=doc["name"], method=doc["method"], mode=doc["mode"], at=doc["at"],
        geometry_id=doc["geometry_id"], reference_signal=doc["reference_signal"],
        n_px=doc["n_px"], window=doc["window"], threshold=doc["threshold"],
        min_height_mm=doc["min_height_mm"], target_dmd=doc.get("target_dmd"),
        zone=doc.get("zone"), cells=doc.get("cells"), spacing_px=doc.get("spacing_px"),
        spot=spot, shape=doc.get("shape", "disk"), level=doc.get("level", 255),
        reference_centroid_px=doc.get("reference_centroid_px"),
        reference_saturated_px=doc.get("reference_saturated_px", 0),
        jacobian=doc.get("jacobian"), depth_axis=doc.get("depth_axis"),
        baseline_axis=doc.get("baseline_axis", "x"), noise_counts=doc.get("noise_counts", 0.0),
        polarity_inverted=doc.get("polarity_inverted"),
        photo_png=Path(doc["photo_png"]) if doc.get("photo_png") else None,
        path=vpath.parent, version_path=vpath, created=doc.get("created", ""),
        mock=doc.get("mock"), warnings=list(doc.get("warnings", [])))


# ------------------------------------------------------------ presence

def _jacobian(c0, c1, c2, dcol: float, drow: float) -> np.ndarray:
    return np.array([[(c1[0] - c0[0]) / dcol, (c2[0] - c0[0]) / drow],
                     [(c1[1] - c0[1]) / dcol, (c2[1] - c0[1]) / drow]], np.float64)


def _probe_steps(target) -> tuple:
    """+24 columns / +48 rows (or minus, near the DMD edge)."""
    tc, tr = target
    dc = PROBE_DMD_PX if tc + PROBE_DMD_PX <= DMD_SHAPE[1] - 1 else -PROBE_DMD_PX
    rstep = int(round(PROBE_DMD_PX / ROW_PITCH))
    dr = rstep if tr + rstep <= DMD_SHAPE[0] - 1 else -rstep
    return dc, dr


def teach_presence(session, preset, name: str, at, diameter_mm=None,
                   method: str = "photoelectric", *, mm_per_dmd_px=None, shape: str = "disk",
                   level: int = 255, threshold: float = DEFAULT_THRESHOLD,
                   min_height_mm: float = DEFAULT_MIN_HEIGHT_MM, window_px=None,
                   hold_s: float = 0.0, empty_confirmed: bool = False) -> PresenceTeach:
    """Teach a part-present sensor on the EMPTY scene (a new version; latest
    wins). The operator clears the field — a physical, human-only step — and
    the caller passes empty_confirmed=True to say so. at: grid cell / DMD /
    camera point (spot), or a zone spec "G7:H9" / {"zone": "H8"} (photoelectric)."""
    if not empty_confirmed:
        raise ValueError("teach_presence records the EMPTY reference: have the operator clear "
                         "the sensor's field (a physical, human-only step), then call again "
                         "with empty_confirmed=True")
    name = _name(name)
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    thr = _fraction(threshold, "threshold")
    min_h = _positive(min_height_mm, "min_height_mm")
    lv = _level(level)
    if shape not in SHAPES:
        raise ValueError(f"shape must be one of {SHAPES}, got {shape!r}")
    if window_px is not None and (isinstance(window_px, bool) or
                                  not isinstance(window_px, (int, np.integer)) or window_px < 0):
        raise ValueError(f"window_px must be an int >= 0 (camera px margin), got {window_px!r}")
    hold = zones_mod._hold(hold_s, preset, "spot hold")     # + rule 3 duration check
    projmap.dmd_shape_of(preset)
    geo = preset.geometry_id
    _guard_kind(session, geo, name, "presence")
    loc = _resolve_at(session, preset, at, allow_zone=(method == "photoelectric"))
    if loc["mode"] == "zone" and method != "photoelectric":
        raise ValueError("a zone sensor is photoelectric only (bgs triangulates one spot)")

    spec, probes = None, None
    if loc["mode"] == "spot":
        if diameter_mm is None:
            raise ValueError("a spot sensor needs diameter_mm (the spot size at the part)")
        spec = spot_px_for_mm(preset, diameter_mm, mm_per_dmd_px)
        physicals = [spot_image(loc["target"], spec, shape, lv)]
        if method == "bgs":
            dc, dr = _probe_steps(loc["target"])
            tc, tr = loc["target"]
            probes = (dc, dr)
            physicals += [spot_image((tc + dc, tr), spec, shape, lv),
                          spot_image((tc, tr + dr), spec, shape, lv)]
    else:
        physicals = [np.where(loc["proj_mask"], np.uint8(lv), np.uint8(0))]
    sh = _shoot(session, preset, physicals, hold)
    warnings = (list(spec.warnings) if spec else []) + sh.warnings
    if not sh.lit_ok:
        raise ValueError("cannot teach: the projector does not light the scene (raw white == "
                         "raw black) — check that it is on and aimed; " + "; ".join(sh.warnings))
    photo = sh.photos[0]
    diff = photo - sh.dark
    noise = _noise(diff)

    centroid, jac, depth_axis = None, None, None
    if loc["mode"] == "spot":
        spot = _measure_spot(diff, photo, noise)
        if spot is None:
            raise ValueError("no spot found on the empty scene (peak below "
                             f"{SPOT_MIN_SNR:g} x noise): aim it at a lit, matte surface")
        centroid = spot["centroid"]
        n_px = spot["area"]
        margin = int(window_px) if window_px is not None else \
            max(int(round(WINDOW_SPOT_FACTOR * spot["diameter"])), WINDOW_MIN_PX)
        window = _grow(spot["bbox"], margin, photo.shape)
        if spot["split"]:
            warnings.append("the taught spot is split (edge / shadow under it): move it onto a "
                            "flat part of the background")
        if method == "bgs":
            cs = []
            for ph in sh.photos[1:]:
                s2 = _measure_spot(ph - sh.dark, ph, noise)
                if s2 is None:
                    raise ValueError("a triangulation probe spot was not found: teach on a flat, "
                                     "lit background away from edges")
                cs.append(s2["centroid"])
            jac = _jacobian(centroid, cs[0], cs[1], probes[0], probes[1])
            if abs(np.linalg.det(jac)) < 1e-6:
                raise ValueError("the probe spots did not move in the camera (degenerate "
                                 "camera <-> DMD geometry); cannot triangulate here")
            ax = 0 if preset.rig.coded_axis == "x" else 1
            depth_axis = "col" if abs(jac[ax, 0]) >= abs(jac[ax, 1]) else "row"
    else:
        cam = loc["cam_mask"]
        if cam is None or cam.shape != photo.shape:
            cam = _lit_mask(diff, noise)
            warnings.append(f"no projector map for {geo}: the zone's camera area is the lit area "
                            "of this photo (run capture_map for the exact cut)")
        n_px = int(cam.sum())
        if n_px < 16:
            raise ValueError(f"zone {loc['zone']!r} lights almost nothing the camera sees "
                             f"({n_px} px): pick cells in the camera's view")
        ys, xs = np.nonzero(cam)
        bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
        window = _grow(bbox, int(window_px) if window_px is not None else WINDOW_MIN_PX,
                       photo.shape)
    ref, sat_frac = _window_signal(diff, photo, window, n_px)
    if ref < max(MIN_REF_COUNTS, SPOT_MIN_SNR * noise):
        raise ValueError(f"the empty reference returns only {ref:.1f} counts per lit px: raise "
                         "`level` or the exposure, or aim at a brighter background")
    if sat_frac > SAT_MAX_FRACTION:
        warnings.append(f"{100 * sat_frac:.0f}% of the reference px are saturated: a BRIGHTER "
                        "part can't be told from empty — lower `level` or the exposure")

    store = session.store
    photo_png = _save_png(store, preset, f"{safe_label(name)}_teach",
                          _overlay(photo, [("rect", window, WINDOW_BGR),
                                           ("cross", centroid, SPOT_BGR)]))
    doc = {"kind": "presence", "name": name, "method": method, "mode": loc["mode"],
           "at": loc["at"], "geometry_id": geo, "preset": preset.name,
           "mock": bool(preset.mock), "created": _now(), "empty_confirmed": True,
           "target_dmd": _xy(loc["target"]) if loc["mode"] == "spot" else None,
           "zone": loc.get("zone"), "cells": loc.get("cells"),
           "spacing_px": loc.get("spacing_px"),
           "spot": _doc(spec) if spec else None, "shape": shape, "level": lv,
           "reference_signal": round(ref, 4), "n_px": int(n_px), "window": window,
           "threshold": thr, "min_height_mm": min_h,
           "reference_centroid_px": _xy(centroid),
           "reference_saturated_px": int(round(sat_frac * n_px)),
           "jacobian": jac.round(5).tolist() if jac is not None else None,
           "probe_dmd_px": list(probes) if probes else None, "depth_axis": depth_axis,
           "baseline_axis": preset.rig.coded_axis, "noise_counts": round(noise, 4),
           "polarity_inverted": sh.inverted,
           "camera": {"exposure_us": preset.camera.exposure_us,
                      "gain_db": preset.camera.gain_db, "average": preset.camera.average},
           "photo_png": photo_png, "units": PRESENCE_UNITS, "warnings": warnings}
    vpath = _write_version(session, geo, name, doc)
    return _teach_from_doc(json.loads(json.dumps(doc, default=str)), vpath)


def _physical_from_doc(doc: dict) -> np.ndarray:
    lv = int(doc.get("level", 255))
    if doc["mode"] == "zone":
        grid = zones_mod.grid_spec(int(doc["spacing_px"]))
        pm = zones_mod.projector_mask([tuple(c) for c in doc["cells"]], grid)
        return np.where(pm, np.uint8(lv), np.uint8(0))
    return spot_image(doc["target_dmd"], SpotSpec(**doc["spot"]), doc.get("shape", "disk"), lv)


def check_presence(session, preset, name: str, *, threshold=None, min_height_mm=None,
                   min_relief_px=None, hold_s: float = 0.0, label: Optional[str] = None,
                   save_photo: bool = True) -> PresenceResult:
    """Fire a taught sensor: project its spot / zone, photograph, decide
    PRESENT / ABSENT / INDETERMINATE. threshold / min_height_mm default to the
    taught values; min_relief_px (DMD px) decides bgs when no mm is available."""
    name = _name(name)
    geo = preset.geometry_id
    doc, vpath = _latest_doc(session, geo, name, kind="presence")
    rules.require_same_geometry(doc.get("geometry_id", geo), geo, f"sensor {name!r}")
    thr = _fraction(threshold, "threshold") if threshold is not None else float(doc["threshold"])
    min_h = _positive(min_height_mm, "min_height_mm") if min_height_mm is not None \
        else float(doc["min_height_mm"])
    min_rel = _positive(min_relief_px, "min_relief_px") if min_relief_px is not None else None
    lab = safe_label(label if label is not None else f"{name}_check")
    hold = zones_mod._hold(hold_s, preset, "spot hold")     # + rule 3 duration check
    projmap.dmd_shape_of(preset)
    method = doc["method"]

    sh = _shoot(session, preset, [_physical_from_doc(doc)], hold)
    photo = sh.photos[0]
    warnings = list(sh.warnings)
    cam_then = doc.get("camera") or {}
    cam_now = {"exposure_us": preset.camera.exposure_us, "gain_db": preset.camera.gain_db}
    changed = [k for k in cam_now if k in cam_then and cam_then[k] != cam_now[k]]
    if changed:
        warnings.append(f"camera {', '.join(changed)} changed since the teach "
                        f"({cam_then} -> {cam_now}): intensities are not comparable — re-teach")
    diff = photo - sh.dark
    noise = _noise(diff)
    window = [int(v) for v in doc["window"]]
    ref = float(doc["reference_signal"])
    signal, sat_frac = _window_signal(diff, photo, window, int(doc["n_px"]))
    rel = signal / ref - 1.0 if ref > 0 else None
    res = PresenceResult(name=name, method=method, verdict=INDETERMINATE, present=None,
                         signal=_r(signal), reference_signal=_r(ref), relative_change=_r(rel),
                         threshold=thr, reference_centroid_px=doc.get("reference_centroid_px"),
                         min_height_mm=min_h if method == "bgs" else None,
                         saturated_fraction=_r(sat_frac), teach_version=vpath,
                         polarity_inverted=sh.inverted if sh.lit_ok else None,
                         geometry_id=geo, created=_now(), warnings=warnings)

    centroid = None
    if not sh.lit_ok:
        warnings.append("INDETERMINATE: the projector does not light the scene")
    elif method == "photoelectric":
        _decide_photoelectric(res, rel, thr, sat_frac)
    else:
        spot = _measure_spot(diff, photo, noise)
        if spot is not None:
            centroid = spot["centroid"]
            res.centroid_px = _xy(centroid)
            if spot["split"]:
                warnings.append("the spot is split (it straddles a part edge): the brightest "
                                "piece was triangulated")
            if spot["saturated"]:
                warnings.append(f"{spot['saturated']} spot px saturated (centroid still valid)")
        _decide_bgs(session, preset, res, doc, centroid, min_h, min_rel)

    store = session.store
    sdir = _sensor_dir(session, geo, name)
    if save_photo:
        res.photo_png = _save_png(
            store, preset, f"{lab}_photo",
            _overlay(photo, [("rect", window, WINDOW_BGR),
                             ("cross", doc.get("reference_centroid_px"), EXPECT_BGR),
                             ("cross", centroid, SPOT_BGR)]), source=sdir)
    out = {"kind": "presence_check", "preset": preset.name, "mock": bool(preset.mock),
           "teach": {"at": doc.get("at"), "mode": doc.get("mode"), "created": doc.get("created")},
           **_doc(res)}
    res.json_path = _save_json(store, preset, lab, out, source=sdir)
    return res


def _decide_photoelectric(res: PresenceResult, rel, thr: float, sat_frac: float) -> None:
    if rel is None:
        res.warnings.append("INDETERMINATE: the taught reference is zero")
        return
    if sat_frac > SAT_MAX_FRACTION and abs(rel) <= thr:
        res.warnings.append(f"INDETERMINATE: {100 * sat_frac:.0f}% of the receiver px saturated "
                            "and the change is inside the threshold — lower `level` / exposure "
                            "and re-teach")
        return
    if sat_frac > SAT_MAX_FRACTION:
        res.warnings.append(f"{100 * sat_frac:.0f}% of the receiver px saturated: the change is "
                            "a lower bound (still beyond the threshold)")
    res.present = abs(rel) > thr
    res.verdict = PRESENT if res.present else ABSENT


def _decide_bgs(session, preset, res: PresenceResult, doc: dict, centroid, min_h: float,
                min_rel) -> None:
    ref_c = doc.get("reference_centroid_px")
    jac = doc.get("jacobian")
    if centroid is None:
        res.warnings.append("INDETERMINATE: no spot returned — a black / absorbing / specular "
                            "surface, or something blocks the light; can't triangulate")
        return
    if ref_c is None or jac is None:
        res.warnings.append("INDETERMINATE: the teach record has no reference spot / Jacobian")
        return
    J = np.asarray(jac, np.float64)
    d = np.subtract(centroid, ref_c).astype(np.float64)
    off = np.linalg.solve(J, d)                          # DMD px [col, row]
    k = 0 if doc.get("depth_axis", "col") == "col" else 1
    relief = -float(off[k])
    ax = 0 if doc.get("baseline_axis", "x") == "x" else 1
    res.shift_px = _r(d[ax], 3)
    res.off_axis_px = _r(d[1 - ax], 3)
    res.relief_px = _r(relief, 3)
    if abs(d[1 - ax]) > max(1.0, OFF_AXIS_WARN * abs(d[ax])):
        res.warnings.append(f"the spot also moved {d[1 - ax]:+.1f} px ACROSS the baseline axis: "
                            "the rig or the background moved, or the spot sits on a sloped / "
                            "edge surface — height is less certain")
    cal = _calibration(session, preset)
    if cal is not None and cal.has_z:
        res.mm_per_relief_px = _r(cal.mm_per_px, 6)
        res.height_source = "calibrated"
        res.calibration_id = cal.id
        res.height_mm = _r(cal.mm_per_px * relief, 3)
        if cal.intercept_mm:
            res.warnings.append("height uses the depth calibration's slope only (0 relief = the "
                                "taught background)")
    elif preset.rig.nominal_mm_per_relief_px:
        k_nom = preset.rig.nominal_mm_per_relief_px
        res.mm_per_relief_px = _r(k_nom, 6)
        res.height_source = "nominal"
        res.height_mm = _r(k_nom * relief, 3)
        res.warnings.append(f"NOMINAL height: no depth calibration for {preset.geometry_id}; "
                            f"{k_nom:.3f} mm per relief px from rig standoff / baseline / throw "
                            "ratio (small-angle estimate, sign depends on the rig) — compared as "
                            "|height|; calibrate for a measured height")
    if res.height_mm is not None:
        res.present = abs(res.height_mm) > min_h
        res.verdict = PRESENT if res.present else ABSENT
    elif min_rel is not None:
        res.present = abs(relief) > min_rel
        res.verdict = PRESENT if res.present else ABSENT
        res.warnings.append(f"no depth calibration and no rig baseline/standoff: decided on "
                            f"|relief| > {min_rel:g} DMD px (no mm)")
    else:
        res.warnings.append("INDETERMINATE: no mm scale (no depth calibration, no rig baseline / "
                            "standoff) — calibrate, set rig.baseline_mm + standoff_mm, or pass "
                            "min_relief_px")


# ------------------------------------------------------------ alignment

def _wrap90(a: float) -> float:
    """Undirected axis angle difference -> (-90, 90]."""
    a = (a + 90.0) % 180.0 - 90.0
    return 90.0 if a == -90.0 else a


def _pose(scan) -> dict:
    """Part blob pose in a scan. segment.scan_blob picks the reference level
    for the scan's datum: a zone scan's relief is datum-free (base plane at
    0), so segmenting around its median would take the part top for the
    base whenever the part fills most of the zone."""
    from ..measure import segment
    from ..measure.dimensions import principal_extents_px
    blob, h_px, info = segment.scan_blob(scan)
    v, u = np.nonzero(blob)
    pts = np.column_stack([u, v]).astype(np.float64)
    c = pts.mean(axis=0)
    evals, evecs = np.linalg.eigh(np.cov((pts - c).T))
    major = evecs[:, 1]
    angle = _wrap90(math.degrees(math.atan2(major[1], major[0])))
    ratio = math.sqrt(max(evals[1], 0.0) / max(evals[0], 1e-12))
    ext = principal_extents_px(blob)
    return {"centroid": [float(c[0]), float(c[1])], "angle": angle, "ratio": ratio,
            "extents": [float(ext[0]), float(ext[1])], "area": int(blob.sum()),
            "relief_px": float(h_px), "blob": blob,
            "warnings": list(info.get("warnings", []))}


def _align_from_doc(doc: dict, vpath: Path) -> AlignmentRef:
    return AlignmentRef(name=doc["name"], geometry_id=doc["geometry_id"], scan_id=doc["scan_id"],
                        centroid_px=doc["centroid_px"], angle_deg=doc["angle_deg"],
                        extents_px=doc["extents_px"], area_px=doc["area_px"],
                        axis_ratio=doc["axis_ratio"], relief_px=doc["relief_px"],
                        path=vpath.parent, version_path=vpath, created=doc.get("created", ""),
                        mock=doc.get("mock"), warnings=list(doc.get("warnings", [])))


def teach_alignment(session, preset, name: str, scan) -> AlignmentRef:
    """Store the reference pose of the part blob in `scan`'s relief (a new
    version; latest wins). A scan of a part judged REJECT or PULL is refused: a
    reference is taught on a good part only (engine/badparts.py)."""
    from .. import badparts
    name = _name(name)
    geo = preset.geometry_id
    rules.require_same_geometry(scan.geometry_id, geo, "teach_alignment")
    _guard_kind(session, geo, name, "alignment")
    badparts.require_good(session.store, f"teach_alignment {name!r}", [scan], sweep=True)
    from ..measure.segment import SegmentationError
    try:
        pose = _pose(scan)
    except SegmentationError as e:
        raise ValueError(f"scan {scan.id}: {e}") from None
    warnings = list(pose["warnings"])
    if pose["ratio"] < ANGLE_AXIS_RATIO_MIN:
        warnings.append(f"the part is nearly round / square in the image (axis ratio "
                        f"{pose['ratio']:.2f}): its angle is ill-defined, so check_alignment "
                        "can't judge rotation")
    doc = {"kind": "alignment", "name": name, "geometry_id": geo, "preset": preset.name,
           "mock": bool(preset.mock), "created": _now(), "scan_id": scan.id,
           "scan_path": str(scan.path), "centroid_px": _xy(pose["centroid"]),
           "angle_deg": round(pose["angle"], 4), "axis_ratio": round(pose["ratio"], 4),
           "extents_px": [round(v, 2) for v in pose["extents"]], "area_px": pose["area"],
           "relief_px": round(pose["relief_px"], 4), "units": ALIGN_UNITS, "warnings": warnings}
    vpath = _write_version(session, geo, name, doc)
    return _align_from_doc(doc, vpath)


def check_alignment(session, preset, name: str, scan, tol_mm: float = 1.0,
                    tol_deg: float = 2.0, tol_px=None) -> AlignmentResult:
    """dx, dy, dtheta of the part in `scan` vs the taught pose -> ALIGNED /
    NOT_ALIGNED / INDETERMINATE. Position in mm only with a lateral
    calibration for the geometry (rule 6); otherwise px, and the position
    check needs tol_px (camera px)."""
    name = _name(name)
    geo = preset.geometry_id
    tmm = _positive(tol_mm, "tol_mm")
    tdeg = _positive(tol_deg, "tol_deg")
    tpx = _positive(tol_px, "tol_px") if tol_px is not None else None
    rules.require_same_geometry(scan.geometry_id, geo, "check_alignment")
    doc, vpath = _latest_doc(session, geo, name, kind="alignment")
    rules.require_same_geometry(doc.get("geometry_id", geo), geo, f"alignment {name!r}")
    from .. import badparts                 # never against a part judged REJECT / PULL since
    taught = doc.get("scan_path")
    badparts.require_good(session.store, f"check_alignment (reference {name!r}): its scan",
                          [SimpleNamespace(id=doc.get("scan_id"), path=Path(taught))
                           if taught and Path(taught).is_dir() else doc.get("scan_id")],
                          how=f"teach {name!r} again on a good part")
    cal = _calibration(session, preset)
    k = cal.lateral_px_per_mm if cal is not None and cal.has_xy else None
    res = AlignmentResult(name=name, verdict=INDETERMINATE, position_units="mm" if k else "px",
                          tol_mm=tmm, tol_deg=tdeg, tol_px=tpx, ref_centroid_px=doc["centroid_px"],
                          ref_angle_deg=doc["angle_deg"], lateral_px_per_mm=_r(k, 6),
                          calibration_id=cal.id if k else None, scan_id=scan.id,
                          ref_version=vpath, geometry_id=geo, created=_now())
    w = res.warnings
    from ..measure.segment import SegmentationError
    pose = None
    try:
        pose = _pose(scan)
    except SegmentationError as e:
        w.append(f"INDETERMINATE: no part blob in scan {scan.id} ({e}) — is the part present?")
    if pose is not None:
        w.extend(pose["warnings"])
        dx = pose["centroid"][0] - doc["centroid_px"][0]
        dy = pose["centroid"][1] - doc["centroid_px"][1]
        dth = _wrap90(pose["angle"] - float(doc["angle_deg"]))
        res.centroid_px, res.angle_deg = _xy(pose["centroid"]), _r(pose["angle"], 4)
        res.dx_px, res.dy_px = _r(dx, 3), _r(dy, 3)
        res.offset_px = _r(math.hypot(dx, dy), 3)
        res.dtheta_deg = _r(dth, 4)
        res.area_ratio = _r(pose["area"] / max(int(doc["area_px"]), 1), 4)
        if abs(res.area_ratio - 1.0) > AREA_CHANGE_WARN:
            w.append(f"the part blob is {100 * (res.area_ratio - 1):+.0f}% in area vs the "
                     "reference: a different part, or partly hidden / shadowed")
        if k:
            res.dx_mm, res.dy_mm = _r(dx / k, 4), _r(dy / k, 4)
            res.offset_mm = _r(math.hypot(dx, dy) / k, 4)
            res.position_ok = res.offset_mm <= tmm
        else:
            w.append(f"no lateral calibration for {geo}: dx / dy are camera px only (rule 6)"
                     + ("" if tpx else "; pass tol_px or calibrate to judge position"))
            if tpx is not None:
                res.position_ok = res.offset_px <= tpx
        ratio = min(float(doc["axis_ratio"]), pose["ratio"])
        if ratio < ANGLE_AXIS_RATIO_MIN:
            w.append(f"the part is nearly round / square (axis ratio {ratio:.2f}): rotation "
                     "can't be judged from its outline")
        else:
            res.angle_ok = abs(dth) <= tdeg
        oks = (res.position_ok, res.angle_ok)
        if False in oks:
            res.verdict = NOT_ALIGNED
        elif None in oks:
            res.verdict = INDETERMINATE
        else:
            res.verdict = ALIGNED

    store = session.store
    sdir = _sensor_dir(session, geo, name)
    rel = np.asarray(scan.relief, np.float64)
    fin = np.isfinite(rel)
    img = np.zeros(rel.shape, np.float32)
    if fin.any():
        lo, hi = np.percentile(rel[fin], [1, 99])
        img[fin] = np.clip((rel[fin] - lo) / (hi - lo + 1e-12), 0, 1) * 255.0
    half = 0.5 * float(doc["extents_px"][0])
    marks = [("axis", doc["centroid_px"], float(doc["angle_deg"]), half, EXPECT_BGR)]
    if pose is not None:
        marks.append(("axis", pose["centroid"], pose["angle"], 0.5 * pose["extents"][0],
                      SPOT_BGR))
    res.overlay_png = _save_png(store, preset, f"{safe_label(name)}_align", _overlay(img, marks),
                                source=sdir)
    out = {"kind": "alignment_check", "preset": preset.name, "mock": bool(preset.mock),
           **_doc(res)}
    res.json_path = _save_json(store, preset, f"{safe_label(name)}_align", out, source=sdir)
    return res

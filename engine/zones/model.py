"""
Zone model: shapes drawn in PROJECTOR px, and the set they belong to.

    Zone     id ("Z1"), name (a label, may be empty), shape rect | ellipse | polygon, note
             rect / ellipse: center [col, row], size [width, height] along the shape's own
             axes, rotation_deg (clockwise in the DMD image: col right, row down)
             polygon: points [[col, row], ...] (>= 3, not self-crossing)
    ZoneSet  name, geometry_id + fingerprint (config.geometry_fingerprint: the rig settings
             the px mean something at), zones, created

Coordinates are native DMD px, origin the top-left corner of the 912 x 1140 pattern image
(DMD_SHAPE is rows, cols): mirror (col, row) covers [col, col + 1) x [row, row + 1), its
centre is (col + 0.5, row + 0.5). A mirror belongs to a zone when its CENTRE is inside the
shape, so the lit area of a zone matches the shape's analytic area to within its outline.

Everything here is pure geometry (numpy only): no display, no library, no rig.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

DMD_SHAPE = (1140, 912)                 # rows, cols (native pattern space)
SHAPES = ("rect", "ellipse", "polygon")
MIN_SIZE_PX = 8.0                       # smallest side / axis / polygon extent, DMD px
MIN_AREA_PX = 48.0                      # smallest polygon area, DMD px^2
MAX_ZONES = 64
MAX_POINTS = 256                        # polygon vertices
ELLIPSE_PTS = 96                        # outline points of an ellipse (drawing, recipe polygons)
SCHEMA = "zoneset-1"
UNITS = "projector px [col, row]: native DMD px, origin top-left of the 912 x 1140 image"
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,31}$")
_SET = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_NAME = re.compile(r"^[A-Za-z0-9_.\- ]{0,32}$")
_EPS = 1e-6


class ZoneError(ValueError):
    """A zone / zone set is missing, unreadable or invalid; `.problems` lists every problem."""

    def __init__(self, message: str, problems: Optional[list] = None):
        super().__init__(message)
        self.problems = list(problems or [message])


def valid_set_name(name) -> bool:
    """A zone set name: one path-safe segment (it is a folder name in the library)."""
    return isinstance(name, str) and bool(_SET.fullmatch(name)) and name not in (".", "..") \
        and not name.endswith(".")


def valid_zone_id(zid) -> bool:
    return isinstance(zid, str) and bool(_ID.fullmatch(zid))


def valid_zone_name(name) -> bool:
    """A zone's name (its label; may be empty): letters, digits, space, _ . - (max 32)."""
    return isinstance(name, str) and bool(_NAME.fullmatch(name))


def _fin(*vals) -> bool:
    try:
        return all(math.isfinite(float(v)) for v in vals)
    except (TypeError, ValueError):
        return False


def norm_deg(a: float) -> float:
    """An angle in (-180, 180]."""
    a = math.fmod(float(a), 360.0)
    if a > 180.0:
        a -= 360.0
    elif a <= -180.0:
        a += 360.0
    return a


def _cs(deg: float) -> tuple:
    t = math.radians(deg)
    return math.cos(t), math.sin(t)


# ------------------------------------------------------------------ one zone

@dataclass
class Zone:
    id: str
    shape: str
    name: str = ""
    note: str = ""
    center: Optional[tuple] = None          # rect / ellipse: (col, row)
    size: Optional[tuple] = None            # rect / ellipse: (width, height) on its own axes
    rotation_deg: float = 0.0               # rect / ellipse: clockwise in the DMD image
    points: Optional[list] = None           # polygon: [(col, row), ...]

    # -- constructors
    @classmethod
    def rect(cls, zid: str, x0: float, y0: float, x1: float, y1: float, rotation_deg: float = 0.0,
             name: str = "", note: str = "") -> "Zone":
        """A rectangle from two opposite corners (before rotation about its centre)."""
        return cls(id=zid, shape="rect", name=name, note=note,
                   center=((x0 + x1) / 2.0, (y0 + y1) / 2.0),
                   size=(abs(x1 - x0), abs(y1 - y0)), rotation_deg=norm_deg(rotation_deg))

    @classmethod
    def ellipse(cls, zid: str, x0: float, y0: float, x1: float, y1: float,
                rotation_deg: float = 0.0, name: str = "", note: str = "") -> "Zone":
        """The ellipse inscribed in the box of two opposite corners."""
        z = cls.rect(zid, x0, y0, x1, y1, rotation_deg, name, note)
        z.shape = "ellipse"
        return z

    @classmethod
    def polygon(cls, zid: str, points, name: str = "", note: str = "") -> "Zone":
        return cls(id=zid, shape="polygon", name=name, note=note,
                   points=[(float(p[0]), float(p[1])) for p in points])

    def copy(self) -> "Zone":
        return Zone(id=self.id, shape=self.shape, name=self.name, note=self.note,
                    center=None if self.center is None else tuple(self.center),
                    size=None if self.size is None else tuple(self.size),
                    rotation_deg=self.rotation_deg,
                    points=None if self.points is None else [tuple(p) for p in self.points])

    # -- geometry
    def corners(self) -> np.ndarray:
        """rect / ellipse: the 4 corners of its (rotated) box, clockwise from top-left."""
        cx, cy = self.center
        w, h = self.size
        c, s = _cs(self.rotation_deg)
        out = []
        for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            u, v = su * w / 2.0, sv * h / 2.0
            out.append((cx + u * c - v * s, cy + u * s + v * c))
        return np.array(out, np.float64)

    def outline(self, n: int = ELLIPSE_PTS) -> np.ndarray:
        """(N, 2) closed outline [col, row]: 4 corners, n ellipse points, or the polygon."""
        if self.shape == "polygon":
            return np.array(self.points, np.float64).reshape(-1, 2)
        if self.shape == "rect":
            return self.corners()
        cx, cy = self.center
        a, b = self.size[0] / 2.0, self.size[1] / 2.0
        c, s = _cs(self.rotation_deg)
        t = np.linspace(0.0, 2.0 * np.pi, int(n), endpoint=False)
        u, v = a * np.cos(t), b * np.sin(t)
        return np.c_[cx + u * c - v * s, cy + u * s + v * c]

    def bbox(self) -> tuple:
        """(x0, y0, x1, y1) of the shape, DMD px (continuous, not rounded)."""
        if self.shape == "ellipse":
            cx, cy = self.center
            a, b = self.size[0] / 2.0, self.size[1] / 2.0
            c, s = _cs(self.rotation_deg)
            hx, hy = math.hypot(a * c, b * s), math.hypot(a * s, b * c)
            return (cx - hx, cy - hy, cx + hx, cy + hy)
        p = self.outline()
        return (float(p[:, 0].min()), float(p[:, 1].min()),
                float(p[:, 0].max()), float(p[:, 1].max()))

    def area(self) -> float:
        """Analytic area, DMD px^2."""
        if self.shape == "rect":
            return float(self.size[0] * self.size[1])
        if self.shape == "ellipse":
            return float(math.pi * self.size[0] * self.size[1] / 4.0)
        return polygon_area(self.points)

    def centroid(self) -> tuple:
        if self.shape != "polygon":
            return (float(self.center[0]), float(self.center[1]))
        p = np.array(self.points, np.float64)
        x, y = p[:, 0], p[:, 1]
        xn, yn = np.roll(x, -1), np.roll(y, -1)
        cr = x * yn - xn * y
        a = cr.sum() / 2.0
        if abs(a) < 1e-9:
            return (float(x.mean()), float(y.mean()))
        return (float(((x + xn) * cr).sum() / (6.0 * a)), float(((y + yn) * cr).sum() / (6.0 * a)))

    def local(self, x: float, y: float) -> tuple:
        """rect / ellipse: a DMD point in the shape's own axes (u along its width)."""
        c, s = _cs(self.rotation_deg)
        dx, dy = x - self.center[0], y - self.center[1]
        return (dx * c + dy * s, -dx * s + dy * c)

    def world(self, u: float, v: float) -> tuple:
        c, s = _cs(self.rotation_deg)
        return (self.center[0] + u * c - v * s, self.center[1] + u * s + v * c)

    def contains(self, x: float, y: float) -> bool:
        if self.shape == "polygon":
            return point_in_polygon(x, y, self.points)
        u, v = self.local(x, y)
        a, b = self.size[0] / 2.0, self.size[1] / 2.0
        if self.shape == "rect":
            return abs(u) <= a and abs(v) <= b
        return a > 0 and b > 0 and (u / a) ** 2 + (v / b) ** 2 <= 1.0

    def mask(self, shape: tuple = DMD_SHAPE) -> np.ndarray:
        """bool (rows, cols): the mirrors whose centre is inside the shape."""
        H, W = shape
        m = np.zeros((H, W), bool)
        if self.shape == "polygon":
            return _raster_polygon(np.array(self.points, np.float64), (H, W))
        x0, y0, x1, y1 = self.bbox()
        c0, c1 = max(0, int(math.floor(x0))), min(W, int(math.ceil(x1)) + 1)
        r0, r1 = max(0, int(math.floor(y0))), min(H, int(math.ceil(y1)) + 1)
        if c1 <= c0 or r1 <= r0:
            return m
        c, s = _cs(self.rotation_deg)
        dx = (np.arange(c0, c1) + 0.5 - self.center[0])[None, :]
        dy = (np.arange(r0, r1) + 0.5 - self.center[1])[:, None]
        u, v = dx * c + dy * s, -dx * s + dy * c
        a, b = self.size[0] / 2.0, self.size[1] / 2.0
        if self.shape == "rect":
            inside = (np.abs(u) <= a + _EPS) & (np.abs(v) <= b + _EPS)
        else:
            inside = (u / max(a, _EPS)) ** 2 + (v / max(b, _EPS)) ** 2 <= 1.0 + _EPS
        m[r0:r1, c0:c1] = inside
        return m

    # -- transforms (new zones; nothing is changed in place)
    def moved(self, dx: float, dy: float) -> "Zone":
        z = self.copy()
        if z.shape == "polygon":
            z.points = [(x + dx, y + dy) for x, y in z.points]
        else:
            z.center = (z.center[0] + dx, z.center[1] + dy)
        return z

    def rotated(self, ddeg: float) -> "Zone":
        """Turned by ddeg about its centre (a polygon: about its centroid)."""
        z = self.copy()
        if z.shape == "polygon":
            cx, cy = self.centroid()
            c, s = _cs(ddeg)
            z.points = [(cx + (x - cx) * c - (y - cy) * s, cy + (x - cx) * s + (y - cy) * c)
                        for x, y in z.points]
        else:
            z.rotation_deg = norm_deg(z.rotation_deg + ddeg)
        return z

    # -- JSON
    def to_dict(self) -> dict:
        d = {"id": self.id, "name": self.name, "shape": self.shape}
        if self.shape == "polygon":
            d["points"] = [[round(float(x), 3), round(float(y), 3)] for x, y in self.points]
        else:
            d["center"] = [round(float(v), 3) for v in self.center]
            d["size"] = [round(float(v), 3) for v in self.size]
            d["rotation_deg"] = round(float(self.rotation_deg), 3)
        d["note"] = self.note
        return d

    @classmethod
    def from_dict(cls, d) -> "Zone":
        """A zone mapping -> Zone, or ZoneError (its form only: zone_problems() checks the
        rest: inside the DMD, sizes, a polygon that crosses itself)."""
        if not isinstance(d, dict):
            raise ZoneError(f"a zone is a mapping, got {type(d).__name__}")
        known = {"id", "name", "shape", "note", "center", "size", "rotation_deg", "points",
                 "_comment"}
        extra = sorted(set(d) - known)
        if extra:
            raise ZoneError(f"zone {d.get('id')!r}: unknown keys {extra} (known: "
                            f"{', '.join(sorted(known - {'_comment'}))})")
        shape = d.get("shape")
        if shape not in SHAPES:
            raise ZoneError(f"zone {d.get('id')!r}: shape {shape!r} must be one of "
                            f"{', '.join(SHAPES)}")
        zid, name, note = d.get("id"), d.get("name", ""), d.get("note", "")
        if shape == "polygon":
            pts = d.get("points")
            if not isinstance(pts, (list, tuple)) or not all(
                    isinstance(p, (list, tuple)) and len(p) == 2 and _fin(*p) for p in pts):
                raise ZoneError(f"zone {zid!r}: points must be [[col, row], ...] finite numbers")
            if any(d.get(k) is not None for k in ("center", "size")) or \
                    d.get("rotation_deg") not in (None, 0, 0.0):
                raise ZoneError(f"zone {zid!r}: a polygon has points only (no center / size / "
                                "rotation_deg)")
            return cls(id=zid, shape=shape, name=name, note=note,
                       points=[(float(p[0]), float(p[1])) for p in pts])
        c, s, r = d.get("center"), d.get("size"), d.get("rotation_deg", 0.0)
        if not (isinstance(c, (list, tuple)) and len(c) == 2 and _fin(*c)):
            raise ZoneError(f"zone {zid!r}: center must be [col, row] finite numbers, got {c!r}")
        if not (isinstance(s, (list, tuple)) and len(s) == 2 and _fin(*s)):
            raise ZoneError(f"zone {zid!r}: size must be [width, height] finite numbers, got {s!r}")
        if isinstance(r, bool) or not _fin(r):
            raise ZoneError(f"zone {zid!r}: rotation_deg must be a finite number, got {r!r}")
        if d.get("points") is not None:
            raise ZoneError(f"zone {zid!r}: a {shape} has center / size / rotation_deg (no points)")
        return cls(id=zid, shape=shape, name=name, note=note, center=(float(c[0]), float(c[1])),
                   size=(float(s[0]), float(s[1])), rotation_deg=norm_deg(float(r)))

    def describe(self) -> str:
        x0, y0, x1, y1 = self.bbox()
        rot = (f", {self.rotation_deg:g} deg"
               if self.shape != "polygon" and self.rotation_deg else "")
        return (f"{self.id}{' ' + self.name if self.name else ''}: {self.shape} "
                f"{x1 - x0:.0f} x {y1 - y0:.0f} px at ({x0:.0f}, {y0:.0f}){rot}")


# ------------------------------------------------------------------ polygons

def polygon_area(points) -> float:
    p = np.array(points, np.float64).reshape(-1, 2)
    x, y = p[:, 0], p[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def point_in_polygon(x: float, y: float, points) -> bool:
    """Even-odd rule (the rule the rasteriser uses)."""
    inside = False
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        if (y1 <= y < y2) or (y2 <= y < y1):
            if x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
                inside = not inside
    return inside


def _seg_cross(a, b, c, d) -> bool:
    """Do the open segments a-b and c-d properly cross?"""
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
    o1, o2, o3, o4 = orient(a, b, c), orient(a, b, d), orient(c, d, a), orient(c, d, b)
    return (o1 * o2 < 0) and (o3 * o4 < 0)


def self_crossing(points) -> bool:
    """True when two non-adjacent edges of the closed polygon cross (a bow-tie)."""
    n = len(points)
    for i in range(n):
        a, b = points[i], points[(i + 1) % n]
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue                                # the closing edge is adjacent to the first
            if _seg_cross(a, b, points[j], points[(j + 1) % n]):
                return True
    return False


def _raster_polygon(pts: np.ndarray, shape: tuple) -> np.ndarray:
    """Even-odd scanline fill at the mirror centres -> bool (rows, cols)."""
    H, W = shape
    if len(pts) < 3:
        return np.zeros((H, W), bool)
    x1, y1 = pts[:, 0], pts[:, 1]
    x2, y2 = np.roll(x1, -1), np.roll(y1, -1)
    r0 = max(0, int(math.floor(float(y1.min()) - 0.5)))
    r1 = min(H - 1, int(math.ceil(float(y1.max()))))
    if r1 < r0:
        return np.zeros((H, W), bool)
    rows = np.arange(r0, r1 + 1)
    yc = (rows + 0.5)[:, None]
    cross = ((y1 <= yc) & (yc < y2)) | ((y2 <= yc) & (yc < y1))
    with np.errstate(divide="ignore", invalid="ignore"):
        x = x1 + (yc - y1) * (x2 - x1) / (y2 - y1)
    ri, ei = np.nonzero(cross)
    toggles = np.zeros((H, W + 1), np.int32)
    col = np.clip(np.ceil(x[ri, ei] - 0.5), 0, W).astype(np.intp)   # first centre right of the edge
    np.add.at(toggles, (rows[ri], col), 1)
    return (np.cumsum(toggles[:, :W], axis=1) % 2) == 1


# ------------------------------------------------------------------ validation

def zone_problems(z: Zone, shape: tuple = DMD_SHAPE) -> list:
    """Every problem of one zone (empty: it is valid): finite numbers, inside the DMD, not
    smaller than MIN_SIZE_PX, a polygon that does not cross itself."""
    H, W = shape
    out = []
    who = f"zone {z.id!r}"
    if not valid_zone_id(z.id):
        out.append(f"{who}: id must be letters, digits, _ . - (max 32, e.g. Z1)")
    if not valid_zone_name(z.name):
        out.append(f"{who}: name must be text of letters, digits, space, _ . - (max 32)")
    if not isinstance(z.note, str) or len(z.note) > 200:
        out.append(f"{who}: note must be text (max 200 characters)")
    if z.shape not in SHAPES:
        return out + [f"{who}: shape {z.shape!r} must be one of {', '.join(SHAPES)}"]
    if z.shape == "polygon":
        pts = z.points or []
        if not all(isinstance(p, (list, tuple)) and len(p) == 2 and _fin(*p) for p in pts):
            return out + [f"{who}: polygon points must be finite [col, row] numbers (no NaN)"]
        if not 3 <= len(pts) <= MAX_POINTS:
            return out + [f"{who}: a polygon needs 3..{MAX_POINTS} points, got {len(pts)}"]
        if self_crossing(pts):
            out.append(f"{who}: the polygon crosses itself")
        if polygon_area(pts) < MIN_AREA_PX:
            out.append(f"{who}: polygon area {polygon_area(pts):.1f} px^2 is under "
                       f"{MIN_AREA_PX:g} px^2")
    else:
        if z.center is None or z.size is None or not _fin(*z.center, *z.size, z.rotation_deg):
            return out + [f"{who}: center, size and rotation_deg must be finite numbers (no NaN)"]
        if min(z.size) <= 0:
            return out + [f"{who}: size must be positive, got {tuple(z.size)}"]
    x0, y0, x1, y1 = z.bbox()
    small = (min(z.size) if z.shape != "polygon" else min(x1 - x0, y1 - y0))
    if small < MIN_SIZE_PX - _EPS:
        out.append(f"{who}: {small:.1f} px is smaller than the minimum {MIN_SIZE_PX:g} DMD px")
    if x0 < -_EPS or y0 < -_EPS or x1 > W + _EPS or y1 > H + _EPS:
        out.append(f"{who}: reaches outside the DMD (0..{W} cols, 0..{H} rows): its box is "
                   f"({x0:.1f}, {y0:.1f}) .. ({x1:.1f}, {y1:.1f})")
    elif not out and not z.mask(shape).any():
        out.append(f"{who}: covers no DMD mirror")
    return out


# ------------------------------------------------------------------ the set

@dataclass
class ZoneSet:
    name: str
    geometry_id: str
    zones: list = field(default_factory=list)       # [Zone]
    fingerprint: Optional[dict] = None              # config.geometry_fingerprint(preset)
    created: str = ""
    preset: str = ""
    note: str = ""
    mock: bool = False
    dmd_shape: tuple = DMD_SHAPE
    polarity_inverted: Optional[bool] = None        # as measured when it was painted
    source: Optional[str] = None                    # the set it was edited from (--from)
    path: Optional[Path] = None                     # its library folder (None: not saved)

    @property
    def ids(self) -> list:
        return [z.id for z in self.zones]

    def zone(self, zid: str) -> Zone:
        for z in self.zones:
            if z.id == zid:
                return z
        raise ZoneError(f"zone set {self.name!r} has no zone {zid!r} (it has "
                        f"{', '.join(self.ids) or 'none'})")

    def select(self, ids=None) -> list:
        """The zones named by `ids` (None / empty = all), in the set's order; an unknown or
        repeated id is refused."""
        if not ids:
            return list(self.zones)
        ids = [str(i).strip() for i in ids]
        if len(set(ids)) != len(ids):
            raise ZoneError(f"zone ids are repeated in {ids}")
        for i in ids:
            self.zone(i)
        return [z for z in self.zones if z.id in set(ids)]

    def mask(self, ids=None) -> np.ndarray:
        """bool (rows, cols): the union of the selected zones (the mask a scan is lit through)."""
        m = np.zeros(tuple(self.dmd_shape), bool)
        for z in self.select(ids):
            m |= z.mask(tuple(self.dmd_shape))
        return m

    def masks(self, ids=None) -> dict:
        """{zone id: bool (rows, cols)}"""
        return {z.id: z.mask(tuple(self.dmd_shape)) for z in self.select(ids)}

    def bboxes(self, ids=None) -> dict:
        """{zone id: [x0, y0, x1, y1]} DMD px, the box of the lit mirrors ([x0, x1) x [y0, y1))."""
        out = {}
        for zid, m in self.masks(ids).items():
            rows, cols = np.nonzero(m)
            out[zid] = ([int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1]
                        if len(rows) else None)
        return out

    def problems(self) -> list:
        out = []
        if not valid_set_name(self.name):
            out.append("name must be letters, digits, _ . - (max 64), e.g. lid_zones")
        if not isinstance(self.geometry_id, str) or not self.geometry_id:
            out.append("geometry_id is required (zones are bound to the rig geometry)")
        if tuple(self.dmd_shape) != DMD_SHAPE:
            out.append(f"dmd_shape {tuple(self.dmd_shape)} is not the DMD's {DMD_SHAPE} "
                       "(rows, cols)")
        if not self.zones:
            out.append("a zone set needs at least one zone")
        if len(self.zones) > MAX_ZONES:
            out.append(f"at most {MAX_ZONES} zones, got {len(self.zones)}")
        seen = set()
        for z in self.zones:
            if z.id in seen:
                out.append(f"duplicate zone id {z.id!r}")
            seen.add(z.id)
            out += zone_problems(z, DMD_SHAPE)
        return out

    def check(self) -> "ZoneSet":
        probs = self.problems()
        if probs:
            raise ZoneError(f"zone set {self.name!r}: {len(probs)} problem(s):\n  - "
                            + "\n  - ".join(probs), probs)
        return self

    def to_dict(self) -> dict:
        return {"schema": SCHEMA, "kind": "zoneset", "name": self.name,
                "geometry_id": self.geometry_id, "geometry_fp": self.fingerprint,
                "preset": self.preset, "mock": bool(self.mock), "created": self.created,
                "dmd_shape": [int(v) for v in self.dmd_shape], "units": UNITS,
                "polarity_inverted": self.polarity_inverted, "source": self.source,
                "note": self.note, "zones": [z.to_dict() for z in self.zones]}

    @classmethod
    def from_dict(cls, d, path: Optional[Path] = None) -> "ZoneSet":
        if not isinstance(d, dict) or d.get("kind") != "zoneset":
            raise ZoneError("not a zone set document (kind zoneset)")
        if d.get("schema") != SCHEMA:
            raise ZoneError(f"zone set schema {d.get('schema')!r} is not {SCHEMA!r}")
        zs = d.get("zones")
        if not isinstance(zs, list):
            raise ZoneError("zones must be a list")
        shape = d.get("dmd_shape", list(DMD_SHAPE))
        if not (isinstance(shape, (list, tuple)) and len(shape) == 2
                and all(isinstance(v, int) and not isinstance(v, bool) for v in shape)):
            raise ZoneError(f"dmd_shape must be [rows, cols] whole numbers, got {shape!r}")
        fp = d.get("geometry_fp")
        if fp is not None and not isinstance(fp, dict):
            raise ZoneError("geometry_fp must be a mapping or null")
        return cls(name=d.get("name"), geometry_id=d.get("geometry_id"),
                   zones=[Zone.from_dict(z) for z in zs], fingerprint=fp,
                   created=str(d.get("created") or ""), preset=str(d.get("preset") or ""),
                   note=str(d.get("note") or ""), mock=bool(d.get("mock")),
                   dmd_shape=(int(shape[0]), int(shape[1])),
                   polarity_inverted=d.get("polarity_inverted"), source=d.get("source"),
                   path=Path(path) if path is not None else None)

    def summary(self) -> dict:
        boxes = self.bboxes()
        return {"name": self.name, "geometry_id": self.geometry_id, "preset": self.preset,
                "created": self.created, "mock": bool(self.mock), "n_zones": len(self.zones),
                "zones": [{"id": z.id, "name": z.name, "shape": z.shape,
                           "size_px": _own_size(z),
                           "bbox_dmd_px": boxes.get(z.id), "area_px": round(z.area(), 1),
                           "rotation_deg": (round(z.rotation_deg, 3)
                                            if z.shape != "polygon" else 0.0),
                           "note": z.note} for z in self.zones],
                "path": self.path}


def _own_size(z: Zone) -> list:
    """[width, height] along the shape's own axes (a polygon: of its box), DMD px."""
    if z.shape != "polygon":
        return [round(float(z.size[0]), 1), round(float(z.size[1]), 1)]
    x0, y0, x1, y1 = z.bbox()
    return [round(x1 - x0, 1), round(y1 - y0, 1)]


def next_id(zones) -> str:
    """The next free Z<n>: one past the highest number in use (ids are never re-used while
    their zone exists, and never renumbered: a recipe names them)."""
    n = 0
    for z in zones:
        m = re.fullmatch(r"Z(\d+)", z.id if isinstance(z, Zone) else str(z))
        if m:
            n = max(n, int(m.group(1)))
    return f"Z{n + 1}"

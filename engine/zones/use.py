"""
Using painted zone sets: scan masks, per-zone point counts, recipe zones, regions.

A zone set is named where a lit zone is named anywhere in the engine, as text:

    zoneset:<set>             every zone of the set (the union is lit)
    zoneset:<set>:Z1,Z3       only those zones
    zoneset:<set>/Z1          one zone (the form a recipe zone's `ref` uses)
    zoneset:<a>+<b>:Z1        several sets in one text: '+' joins terms (it is in no set
                              name and no zone id); the union of every term is lit

`scan --zone`, `scan-fast --zone / --zones`, `scan-metric --zone`, a measurement's zone, a
recipe's scan.zone and a station job's zone all take it (engine/capture/zones.resolve
dispatches on the prefix, as it does for atlas regions). Each set is looked up AT THE PRESET'S
GEOMETRY and its fingerprint is compared with the preset's (rules 5 / 6): zones are projector
px at one rig placement, so a set painted at another geometry or other rig settings is
refused, never silently used. The word `selection` (engine/zones/selection.py) stands for
the zone sets switched on at the preset's geometry; the verbs expand it into this text.

Zone ids (Z1, Z2 ...) repeat across sets, so wherever zones of a text are told apart they
carry a KEY "<set>/<id>" ("cells/Z1"): the form a recipe `ref` already uses.

No camera mask is returned for a zone set: where its mirrors land in the camera image
depends on the depth of what they hit (parallax), so camera px are derived PER SCAN from
that scan's own decoded (col, row) - never from a stored projector map of another scene.

    select_multi()     a zone text at a preset -> Picked (the sets, the zones, the keys)
    scan_fast()        lit through the union, + per-zone point counts (zones.json in the scan)
    count_points()     decoded points per zone of one scan (one set, keyed by the zone id)
    report()           ... of a whole text: points and coverage per zone, keyed "<set>/<id>"
    camera_region()    the camera px of a scan that decoded into the zones (their union)
    camera_regions()   ... one camera mask PER ZONE (what a per-zone measurement / verdict
                       rests on: a plane is never fitted across separate zones)
    recipe_zone(s)()   a zone -> a recipe zone in projector_px (rect | circle | polygon)
    recipe_binder()    resolves a recipe zone's ref "zoneset:<set>/<id>" (engine/recipes/schema)
    recipe_mask()      ... to its DMD mask on a given scan (geometry + fingerprint checked)
    users()            the recipes and station configurations that name a set
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from .. import config, rules
from ..library.store import json_safe
from . import store as zstore
from .model import DMD_SHAPE, ELLIPSE_PTS, Zone, ZoneError, ZoneSet, valid_set_name, valid_zone_id

PREFIX = "zoneset:"
JOIN = "+"                              # between the sets of one text (in no set name / zone id)
SCAN_FILE = "zones.json"                # per-zone counts, written into the metric scan's entry
COVERAGE_CELL_PX = 8                    # coverage of a zone: counted in cells of this many mirrors
MIN_ZONE_POINTS = 100                   # fewer decoded points in a zone: it was not inspected
MIN_ZONE_COVERAGE_PCT = 30.0            # ... or less than this share of it seen by the camera
_REF = re.compile(r"^zoneset:([A-Za-z0-9][A-Za-z0-9_.-]{0,63})/([A-Za-z0-9][A-Za-z0-9_.-]{0,31})$")


# ------------------------------------------------------------------ the text form

def is_spec(text) -> bool:
    return isinstance(text, str) and text.strip().lower().startswith(PREFIX)


def _body(text, bare: bool) -> str:
    """The text without its prefix (ZoneError when it has none and one is required)."""
    if not isinstance(text, str) or not text.strip():
        raise ZoneError("empty zone set; write <set> or <set>:Z1,Z3 (`scan zones` lists them)")
    t = text.strip()
    if t.lower().startswith(PREFIX):
        return t[len(PREFIX):]
    if not bare:
        raise ZoneError(f"{text!r} is not a zone set (zoneset:<set>[:Z1,Z3])")
    return t


def _term(t: str, text) -> tuple:
    """One term '<set>', '<set>:Z1,Z3' or '<set>/Z1' -> (set name, [ids] or None)."""
    name, sep, rest = re.split(r"([:/])", t, maxsplit=1) if re.search(r"[:/]", t) else (t, "", "")
    name = name.strip()
    if not valid_set_name(name):
        raise ZoneError(f"zone set {text!r}: {name!r} is not a set name (letters, digits, _ . -)")
    if not sep:
        return name, None
    ids = [i.strip() for i in rest.split(",")]
    if not rest.strip() or not all(valid_zone_id(i) for i in ids):
        raise ZoneError(f"zone set {text!r}: after '{sep}' come zone ids, like {name}:Z1,Z3")
    if len(set(ids)) != len(ids):
        raise ZoneError(f"zone set {text!r}: a zone id is repeated")
    return name, ids


def parse_spec(text, bare: bool = False) -> tuple:
    """'zoneset:lid', 'zoneset:lid:Z1,Z3', 'zoneset:lid/Z1' -> (set name, [ids] or None).
    bare: the prefix may be left out ('lid:Z1,Z3', the CLI's --zones). ONE set: a text that
    joins several ('a+b') is refused here (parse_multi takes it)."""
    t = _body(text, bare)
    if JOIN in t:
        raise ZoneError(f"zone set {text!r} names several sets: ONE set is needed here "
                        "(<set> or <set>:Z1,Z3)")
    return _term(t, text)


def parse_multi(text, bare: bool = False) -> list:
    """A zone text of one or several sets -> [(set name, [ids] or None), ...] in the order
    written: 'zoneset:a' / 'zoneset:a:Z1,Z3' / 'zoneset:a/Z1' (one set, as parse_spec) and
    'zoneset:a+b:Z1' (several). A set named twice and an empty term are refused."""
    t = _body(text, bare)
    out, seen = [], set()
    for term in t.split(JOIN):
        if not term.strip():
            raise ZoneError(f"empty zone set in {text!r}: between two '{JOIN}' comes <set> or "
                            "<set>:Z1,Z3")
        name, ids = _term(term.strip(), text)
        if name in seen:
            raise ZoneError(f"zone set {name!r} is named twice in {text!r}")
        seen.add(name)
        out.append((name, ids))
    return out


def spec_text(name: str, ids=None) -> str:
    return f"{PREFIX}{name}" + (":" + ",".join(ids) if ids else "")


def text_multi(terms) -> str:
    """[(set name, [ids] or None), ...] -> 'zoneset:a+b:Z1' (one set: exactly spec_text)."""
    return PREFIX + JOIN.join(n + (":" + ",".join(i) if i else "") for n, i in terms)


def join_terms(zones) -> str:
    """What `zones` arguments take - one text ('cells', 'cells+tube:Z1', 'zoneset:cells') or a
    list of terms (['cells', 'tube:Z1']) - as ONE zone text 'zoneset:cells+tube:Z1'."""
    if isinstance(zones, (list, tuple)):
        if not zones:
            raise ZoneError("empty zone set; write <set> or <set>:Z1,Z3 (`scan zones` lists them)")
        terms = []
        for z in zones:
            terms += parse_multi(z, bare=True)
        names = [n for n, _ in terms]
        twice = sorted({n for n in names if names.count(n) > 1})
        if twice:
            raise ZoneError(f"zone set {twice[0]!r} is named twice in {list(zones)!r}")
        return text_multi(terms)
    return text_multi(parse_multi(zones, bare=True))


# ------------------------------------------------------------------ bound to the rig

def for_preset(store, preset, name: str) -> ZoneSet:
    """The set `name` at the preset's geometry, its fingerprint checked against the preset's
    (GeometryMismatch when it was painted at another geometry / other rig settings)."""
    if not valid_set_name(name):
        raise ZoneError(f"zone set name must be letters, digits, _ . - (max 64), got {name!r}")
    geo = preset.geometry_id
    if not zstore.exists(store, geo, name):
        others = zstore.geometries_of(store, name)
        if others:
            raise rules.GeometryMismatch(
                f"zone set {name!r} was painted at geometry {', '.join(others)}; this preset "
                f"({preset.name}) is geometry {geo}. Zones are projector px at ONE rig "
                f"placement: paint it again with this preset (scan --preset {preset.name} "
                f"zone-paint --name {name})")
    zs = zstore.load(store, geo, name)
    rules.require_same_fingerprint(zs.fingerprint, config.geometry_fingerprint(preset),
                                   f"zone set {name}")
    return zs


def select(store, preset, text, bare: bool = False) -> tuple:
    """-> (ZoneSet, [ids] or None, canonical text). ONE set; unknown ids are refused here."""
    name, ids = parse_spec(text, bare=bare)
    zs = for_preset(store, preset, name)
    zs.select(ids)
    return zs, ids, spec_text(name, ids)


class Picked:
    """A zone text resolved at a preset: its sets (each one's geometry and fingerprint
    checked) with the zones chosen in each. `text` is the canonical text (what is lit and
    recorded); `zones()` lists the chosen zones with their keys "<set>/<id>"."""

    def __init__(self, terms: list):
        self.terms = list(terms)                    # [(ZoneSet, [ids] or None)]
        self.text = text_multi([(zs.name, ids) for zs, ids in self.terms])

    @property
    def sets(self) -> list:
        return [zs for zs, _ in self.terms]

    @property
    def several_sets(self) -> bool:
        return len(self.terms) > 1

    def zones(self) -> list:
        """[(key, ZoneSet, Zone)] in the order of the text, each set's zones in its own."""
        return [(f"{zs.name}/{z.id}", zs, z) for zs, ids in self.terms for z in zs.select(ids)]

    @property
    def keys(self) -> list:
        return [k for k, _, _ in self.zones()]

    def mask(self) -> np.ndarray:
        """bool (1140, 912): the union of every chosen zone (what a scan is lit through)."""
        m = np.zeros(DMD_SHAPE, bool)
        for zs, ids in self.terms:
            m |= zs.mask(ids)
        return m

    def masks(self) -> dict:
        """{key: bool (1140, 912)}"""
        return {k: z.mask(tuple(zs.dmd_shape)) for k, zs, z in self.zones()}

    def who(self, key: str) -> str:
        """How a zone is named in a message: its id when the text names one set (as before
        several sets could be named), its key otherwise."""
        return key if self.several_sets else key.split("/", 1)[1]

    def created(self) -> list:
        """[{"set", "created", "ids"}]: which painting of each set this is."""
        return [{"set": zs.name, "created": zs.created, "ids": [z.id for z in zs.select(ids)]}
                for zs, ids in self.terms]


def select_multi(store, preset, text, bare: bool = False) -> Picked:
    """A zone text of one or several sets (or a list of terms) at the preset -> Picked. Every
    set is checked against the preset's geometry and rig settings and every id against its
    set, so everything refusable is refused here, before the rig."""
    terms = parse_multi(join_terms(text) if (bare or isinstance(text, (list, tuple))) else text)
    out = []
    for name, ids in terms:
        zs = for_preset(store, preset, name)
        zs.select(ids)
        out.append((zs, ids))
    return Picked(out)


def same_zones(a: Picked, b: Picked) -> bool:
    """Do two resolved texts light the same zones (set / id pairs, whatever the order and
    whether a set was written with or without its ids)?"""
    return set(a.keys) == set(b.keys)


def _grid_cells(mask: np.ndarray) -> list:
    """The default grid's cells a DMD mask touches (what a base / scan records as zone_cells)."""
    from ..capture import zones as gz
    s = gz.DEFAULT_SPACING
    rows, cols = np.nonzero(mask)
    return sorted({(int(c), int(r)) for c, r in zip(cols // s, rows // s)})


def resolve(session, preset, text: str) -> tuple:
    """A zone-set text as engine/capture/zones.resolve returns a zone: (DMD mask bool
    (1140, 912), None, cells). The camera mask is always None (see the module doc)."""
    m = select_multi(session.store, preset, text).mask()
    if not m.any():
        raise ZoneError(f"zone set {text!r} covers no DMD mirror: nothing would be lit")
    return m, None, _grid_cells(m)


def scan_mask(zs: ZoneSet, ids=None) -> np.ndarray:
    """The DMD mask a scan is lit through: the union of the zones, bool (1140, 912)."""
    return zs.mask(ids)


def export_mask(zs, path, ids=None) -> Path:
    """Write the union mask as a .npy for `scan scan-fast --mask` (a NEW file, never
    replaced). zs: a ZoneSet (+ ids) or a Picked (several sets)."""
    p = Path(path)
    if p.suffix.lower() != ".npy":
        raise ZoneError(f"the mask file must end in .npy, got {p.name!r}")
    with open(p, "xb") as f:
        np.save(f, zs.mask() if isinstance(zs, Picked) else scan_mask(zs, ids))
    return p


# ------------------------------------------------------------------ points per zone

def _index(corr: np.ndarray, n: int) -> np.ndarray:
    return np.clip(np.rint(corr), 0, n - 1).astype(np.intp)


def count_points(zs: ZoneSet, ids, corr_x, corr_y, valid) -> dict:
    """How many of a scan's points decoded into each zone.

    corr_x / corr_y: the scan's decoded DMD column / row per camera px (corr_y None: a
    columns-only scan); valid: the camera px that carry a point. With both axes a point is in
    a zone when its decoded (col, row) is one of the zone's mirrors. With columns only a
    zone can be counted only when no other selected zone shares its columns; the others are
    None (unknown, said in warnings) - never a guess."""
    chosen = zs.select(ids)
    masks = {z.id: z.mask(tuple(zs.dmd_shape)) for z in chosen}
    H, W = tuple(zs.dmd_shape)
    union = np.zeros((H, W), bool)
    for m in masks.values():
        union |= m
    ok = np.asarray(valid, bool) & np.isfinite(corr_x)
    warnings = []
    counts: dict = {}
    if corr_y is not None:
        ok &= np.isfinite(corr_y)
        c, r = _index(np.asarray(corr_x)[ok], W), _index(np.asarray(corr_y)[ok], H)
        for zid, m in masks.items():
            counts[zid] = int(m[r, c].sum())
        in_zones = int(union[r, c].sum())
        axis = "xy"
    else:
        c = _index(np.asarray(corr_x)[ok], W)
        cols = {zid: m.any(axis=0) for zid, m in masks.items()}
        for zid, cs in cols.items():
            shared = [o for o, oc in cols.items() if o != zid and (oc & cs).any()]
            if shared:
                counts[zid] = None
                warnings.append(f"zone {zid}: this scan decoded DMD columns only and "
                                f"{', '.join(shared)} share(s) its columns, so its points can't "
                                "be told apart: scan with both axes (scan-fast --axis xy)")
            else:
                counts[zid] = int(cs[c].sum())
        in_zones = int(union.any(axis=0)[c].sum())
        axis = "x"
    total = int(ok.sum())
    for zid, n in counts.items():
        if n == 0:
            warnings.append(f"zone {zid}: no point decoded in it (off the part, in shadow, "
                            "outside the camera's view, or too dark)")
    return {"axis": axis, "counts": counts, "points": total, "in_zones": in_zones,
            "outside": total - in_zones, "dmd_px": {zid: int(m.sum()) for zid, m in masks.items()},
            "union_dmd_px": int(union.sum()), "warnings": warnings}


def _coverage(zmask: np.ndarray, r: np.ndarray, c: np.ndarray) -> float:
    """How much of a zone the camera saw, %: the share of the zone's COVERAGE_CELL_PX x
    COVERAGE_CELL_PX-mirror cells (those at least half inside the zone; a tiny zone: any it
    touches) in which at least one camera px decoded. It does not depend on how many camera
    px one mirror covers, and a part of the zone in shadow or off the part lowers it."""
    k = COVERAGE_CELL_PX
    H, W = zmask.shape
    hb, wb = -(-H // k), -(-W // k)
    rows, cols = np.nonzero(zmask)
    per = np.bincount((rows // k) * wb + cols // k, minlength=hb * wb)
    cells = per >= (k * k) // 2
    if not cells.any():
        cells = per > 0
    seen = np.zeros(hb * wb, bool)
    seen[(r // k) * wb + c // k] = True
    return 100.0 * float((seen & cells).sum()) / float(max(int(cells.sum()), 1))


def report(picked: Picked, corr_x, corr_y, valid, masks: bool = False) -> dict:
    """The zones of a resolved text on ONE scan, from the scan's own decode (any scan kind):

        by_zone   [{key "<set>/<id>", zoneset, id, name, points, dmd_px, coverage_pct}]
        points / in_zones / outside   the scan's points: all, in a zone, in none (stray
                  light, other lit areas: reported, never judged)
        camera    {key: bool camera mask} (masks=True): the px that decoded into each zone

    corr_y None (a columns-only scan): a zone is counted only when no other zone of the
    text shares its columns, else its points are None; coverage and camera masks are None
    (never a guess). Zones of different sets may overlap: a point there counts in each."""
    zones = picked.zones()
    zmasks = {k: z.mask(DMD_SHAPE) for k, _zs, z in zones}
    H, W = DMD_SHAPE
    union = picked.mask()
    cx = np.asarray(corr_x)
    fin = np.isfinite(cx)
    warnings, rows, cam = [], [], {}
    overlap = [f"{picked.who(a)} and {picked.who(b)}" for i, a in enumerate(zmasks)
               for b in list(zmasks)[i + 1:] if a.split("/")[0] != b.split("/")[0]
               and (zmasks[a] & zmasks[b]).any()]
    if overlap:
        warnings.append(f"zones overlap ({'; '.join(overlap[:4])}): they are lit once, and a "
                        "point there counts in each")
    if corr_y is not None:
        cy = np.asarray(corr_y)
        fin = fin & np.isfinite(cy)
        ok = fin & np.asarray(valid, bool)
        fc, fr = _index(cx[fin], W), _index(cy[fin], H)
        pts = np.asarray(valid, bool)[fin]              # of the decoded px: those with a point
        for key, zs, z in zones:
            zm = zmasks[key]
            x0, y0, x1, y1 = z.bbox()               # only the px that decoded into its box
            box = (fc >= x0 - 1) & (fc <= x1 + 1) & (fr >= y0 - 1) & (fr <= y1 + 1)
            inz = np.zeros(fc.shape, bool)
            inz[box] = zm[fr[box], fc[box]]
            got = inz & pts
            rows.append({"key": key, "zoneset": zs.name, "id": z.id, "name": z.name,
                         "points": int(got.sum()), "dmd_px": int(zm.sum()),
                         "coverage_pct": round(_coverage(zm, fr[got], fc[got]), 1)})
            if masks:
                m = np.zeros(cx.shape, bool)
                m[fin] = inz
                cam[key] = m
        c, r = _index(cx[ok], W), _index(cy[ok], H)
        in_zones = int(union[r, c].sum())
        axis = "xy"
    else:
        ok = fin & np.asarray(valid, bool)
        c = _index(cx[ok], W)
        cols = {k: m.any(axis=0) for k, m in zmasks.items()}
        for key, zs, z in zones:
            shared = [picked.who(o) for o, oc in cols.items() if o != key and (oc & cols[key]).any()]
            n = None if shared else int(cols[key][c].sum())
            if shared:
                warnings.append(f"zone {picked.who(key)}: this scan decoded DMD columns only and "
                                f"{', '.join(shared)} share(s) its columns, so its points can't "
                                "be told apart: scan with both axes (scan-fast --axis xy)")
            rows.append({"key": key, "zoneset": zs.name, "id": z.id, "name": z.name,
                         "points": n, "dmd_px": int(zmasks[key].sum()), "coverage_pct": None})
        in_zones = int(union.any(axis=0)[c].sum())
        axis = "x"
        cam = None
    total = int(ok.sum())
    for row in rows:
        if row["points"] == 0:
            warnings.append(f"zone {picked.who(row['key'])}: no point decoded in it (off the "
                            "part, in shadow, outside the camera's view, or too dark)")
    return {"axis": axis, "by_zone": rows, "points": total, "in_zones": in_zones,
            "outside": total - in_zones, "union_dmd_px": int(union.sum()),
            "warnings": warnings, **({"camera": cam} if masks else {})}


def zone_doubt(row: dict, min_coverage_pct: float = MIN_ZONE_COVERAGE_PCT,
               where: str = "") -> Optional[str]:
    """Why one zone of a report() was not inspected well enough to judge or measure it (None:
    it was): no / too few decoded points, or too little of it seen by the camera. where: what
    else cut it (" inside Region 10,10,200,200")."""
    n, cov = row.get("points"), row.get("coverage_pct")
    who = f"zone {row['key']}" + (f" ({row['name']})" if row.get("name") else "")
    if n is None:
        return (f"{who}: its points can't be told apart from another zone's (the scan "
                "decoded DMD columns only): scan with both axes")
    if n == 0:
        return (f"{who}: no point decoded in it{where} (off the part, in shadow, outside "
                "the camera's view, or too dark): it was not inspected")
    if n < MIN_ZONE_POINTS:
        return (f"{who}: only {n} points decoded in it{where} (need {MIN_ZONE_POINTS}): it "
                "was not inspected")
    if cov is not None and cov < min_coverage_pct:
        return (f"{who}: the camera saw only {cov:g} % of it{where} (need "
                f"{min_coverage_pct:g} %; the rest is off the part, in shadow or too dark): "
                "it was not inspected")
    return None


def scan_record(picked: Picked, rep: dict, scan_id: Optional[str] = None) -> dict:
    """The zones.json / meta["zones"] record of a scan lit through `picked`. ONE set: the
    shape it always had (counts, names, dmd_px keyed by the zone id; zoneset = its name) plus
    by_zone and sets. Several sets: the same fields keyed "<set>/<id>", zoneset null,
    zonesets = the names."""
    one = not picked.several_sets
    k = (lambda row: row["id"]) if one else (lambda row: row["key"])
    rows = rep["by_zone"]
    zs0 = picked.sets[0]
    return {"kind": "scan_zones", "zoneset": zs0.name if one else None,
            **({} if one else {"zonesets": [zs.name for zs in picked.sets]}),
            "spec": picked.text, "ids": [k(r) for r in rows], "names": {k(r): r["name"] for r in rows},
            "geometry_id": zs0.geometry_id, "zoneset_created": zs0.created if one else None,
            "sets": picked.created(), "scan_id": scan_id,
            "created": datetime.now().isoformat(timespec="seconds"),
            "units": {"counts": "metric points (mm) whose decoded DMD px is in the zone",
                      "dmd_px": "DMD mirrors of the zone", "outside": "points decoded outside "
                      "every selected zone (stray light, other lit areas)",
                      "coverage_pct": f"share of the zone's {COVERAGE_CELL_PX} x "
                      f"{COVERAGE_CELL_PX}-mirror cells a camera px decoded in"},
            "axis": rep["axis"], "counts": {k(r): r["points"] for r in rows},
            "points": rep["points"], "in_zones": rep["in_zones"], "outside": rep["outside"],
            "dmd_px": {k(r): r["dmd_px"] for r in rows}, "union_dmd_px": rep["union_dmd_px"],
            "by_zone": [dict(r) for r in rows], "warnings": list(rep["warnings"])}


def stale_warnings(store, picked: Picked) -> list:
    """One warning per set of the text whose rig-state record changed since it was painted
    (and last shown): zstore.stale_warning."""
    return [w for w in (zstore.stale_warning(store, zs) for zs in picked.sets) if w]


def scan_fast(s, preset, zones, *, zone=None, label: str = "fast", steps: int = 4, periods=None,
              average: int = 2, min_hold_s: float = 0.12, axis: str = "xy", faces=None,
              margin_dmd_px: float = 6.0):
    """The HDMI fast scan lit through the zones of one or several zone sets (their union;
    every other mirror dark) -> the metric scan, with the per-zone point counts in its
    meta["zones"] and in zones.json of its library entry. zones: "<set>", "<set>:Z1,Z3",
    "<a>+<b>:Z1" or a list of such terms. axis "xy" (the default here) decodes rows too, so
    every point is assigned to its zone exactly; "x" is the faster columns-only set."""
    from ..reconstruct import fast_scan
    if zone is not None and str(zone).strip():
        raise ValueError("give a zone (grid cells / mask) or a zone set (zones), not both")
    picked = select_multi(s.store, preset, zones, bare=True)        # refused before the rig
    ms = fast_scan.scan_fast(s, preset, picked.text, int(steps),
                             periods or list(fast_scan.FAST_PERIODS), average,
                             float(min_hold_s), axis, label, faces, float(margin_dmd_px))
    fld = ms.field
    rep = report(picked, fld.corr_x, fld.corr_y, ms.cloud.mask)
    rep["warnings"] += stale_warnings(s.store, picked)
    doc = scan_record(picked, rep, ms.id)
    s.store.write_json(Path(ms.path) / SCAN_FILE, json_safe(doc))
    if isinstance(ms.meta, dict):
        ms.meta["zones"] = doc
        ms.meta.setdefault("warnings", [])
        ms.meta["warnings"] = list(ms.meta["warnings"]) + rep["warnings"]
    return ms


def read_scan_record(ms) -> Optional[dict]:
    """The zones.json of a stored scan (which painting of which sets it was lit through),
    or None when it has none (a scan that was not lit through a zone set by scan_fast)."""
    import json
    try:
        doc = json.loads((Path(ms.path) / SCAN_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return doc if isinstance(doc, dict) else None


def golden_mismatch(s, preset, golden, picked: Optional[Picked]) -> Optional[str]:
    """Why a golden metric scan cannot be compared with a run lit through `picked` (None: the
    run lights the whole field / grid cells), or None when it can. A golden shows only what
    was lit when it was scanned: against a run lit through other zones its unlit rest would
    count as missing (or extra) material. So the two must be lit through the SAME zones (set
    / id pairs), and - where the golden recorded it (zones.json) - the same painting of
    each set."""
    lit = (golden.meta or {}).get("zone")
    said = lit if lit and str(lit) != "full field" else "the whole field"
    if picked is None:
        if is_spec(lit):
            return (f"the golden {golden.id} was scanned through {lit}, this run lights "
                    "other zones (or the whole field): scan a golden the way this run is lit")
        return None
    if not is_spec(lit):
        return (f"the golden {golden.id} was scanned through {said}, this run is lit through "
                f"{picked.text}: scan a golden through the same zones")
    try:
        then = select_multi(s.store, preset, lit)
    except (ZoneError, rules.GeometryMismatch) as e:
        return (f"the golden {golden.id} was scanned through {lit}, which does not resolve "
                f"now ({str(e).splitlines()[0]}): scan a golden through the zones of this run")
    if not same_zones(then, picked):
        return (f"the golden {golden.id} was scanned through {then.text}, this run is lit "
                f"through {picked.text}: scan a golden through the same zones")
    painted = {x.get("set"): x.get("created") for x in (read_scan_record(golden) or {}).get("sets")
               or [] if isinstance(x, dict)}
    for zs in picked.sets:
        if painted.get(zs.name) and painted[zs.name] != zs.created:
            return (f"zone set {zs.name} was painted again ({zs.created}) since the golden "
                    f"{golden.id} was scanned through it ({painted[zs.name]}): scan a golden "
                    "through the zones as they are now")
    return None


def camera_region(s, preset, text: str, ms, shape, warnings: list, lit_zone="?",
                  need: bool = False) -> Optional[np.ndarray]:
    """Measure mode: the camera px of the scan `ms` that belong to a zone-set region - those
    whose decoded (col, row) is one of the zones' mirrors (derived from THIS scan; the union
    of every zone of the text). A columns-only scan can't be cut: the region is then whatever
    it decoded, which is only the zones when the scan was lit through exactly them (else
    refused)."""
    picked = select_multi(s.store, preset, text)
    canon = picked.text
    for stale in stale_warnings(s.store, picked):
        if stale not in warnings:
            warnings.append(stale)
    fld = ms.field if ms is not None else None
    cx = getattr(fld, "corr_x", None)
    cy = getattr(fld, "corr_y", None)
    if cx is not None and cy is not None:
        cx, cy = np.asarray(cx), np.asarray(cy)
        if cx.shape != tuple(shape):
            raise ValueError(f"zone set {canon}: the scan's decode is {cx.shape}, its cloud "
                             f"{tuple(shape)}")
        ok = np.isfinite(cx) & np.isfinite(cy)
        H, W = DMD_SHAPE
        m = np.zeros(tuple(shape), bool)
        m[ok] = picked.mask()[_index(cy[ok], H), _index(cx[ok], W)]
        if not m.any():
            raise ValueError(f"zone set {canon}: no camera px of this scan decoded into its "
                             "zones (were they lit, and does the camera see them? `scan "
                             f"zone-show {canon[len(PREFIX):]}`)")
        return m
    same = "".join(str(lit_zone or "").split()).lower() == "".join(canon.split()).lower()
    if need or (lit_zone != "?" and not same):
        raise ValueError(f"zone set {canon} can't cut this scan: it decoded DMD columns only"
                         + ("" if lit_zone == "?" else
                            f" and was lit through {lit_zone or 'the whole field'}")
                         + " - scan with both axes (scan kind metric, or fast)")
    warnings.append(f"{canon}: the scan decoded DMD columns only, so the region is whatever "
                    "the lit zones decoded")
    return None


def camera_regions(s, preset, text: str, ms, shape, warnings: Optional[list] = None) -> tuple:
    """One camera mask PER ZONE of a zone text on the scan `ms`, from the scan's own decoded
    (col, row) -> (Picked, {key: bool camera mask}, report()). What a per-zone measurement or
    verdict rests on (a plane is fitted per zone, never across separate zones). Needs both
    DMD axes: a columns-only scan cannot tell zones apart (ValueError)."""
    picked = select_multi(s.store, preset, text)
    if warnings is not None:
        for stale in stale_warnings(s.store, picked):
            if stale not in warnings:
                warnings.append(stale)
    fld = ms.field if ms is not None else None
    cx, cy = getattr(fld, "corr_x", None), getattr(fld, "corr_y", None)
    if cx is None or cy is None:
        raise ValueError(f"zone set {picked.text} can't cut this scan zone by zone: it decoded "
                         "DMD columns only - scan with both axes (scan kind metric, or fast "
                         "with axis xy)")
    if np.asarray(cx).shape != tuple(shape):
        raise ValueError(f"zone set {picked.text}: the scan's decode is {np.asarray(cx).shape}, "
                         f"its cloud {tuple(shape)}")
    rep = report(picked, cx, cy, ms.cloud.mask, masks=True)
    cam = rep.pop("camera")
    return picked, cam, rep


# ------------------------------------------------------------------ recipes

def recipe_zone(z: Zone) -> dict:
    """One zone as a recipe zone's geometry, in projector px: an upright rectangle stays a
    rect, a round ellipse a circle, everything else (turned, oval, polygon) a polygon."""
    upright = z.shape != "polygon" and abs(z.rotation_deg) % 180.0 < 1e-9
    if z.shape == "rect" and upright:
        x0, y0, x1, y1 = z.bbox()
        return {"shape": "rect", "units": "projector_px",
                "coords": [round(x0, 3), round(y0, 3), round(x1, 3), round(y1, 3)]}
    if z.shape == "ellipse" and abs(z.size[0] - z.size[1]) < 1e-9:
        return {"shape": "circle", "units": "projector_px",
                "coords": {"center": [round(float(v), 3) for v in z.center],
                           "radius": round(z.size[0] / 2.0, 3)}}
    return {"shape": "polygon", "units": "projector_px",
            "coords": [[round(float(x), 3), round(float(y), 3)] for x, y in z.outline(ELLIPSE_PTS)]}


def recipe_zones(zs: ZoneSet, ids=None, checks=None, by_ref: bool = False) -> list:
    """The `zones` list of a recipe for these zones (ids and labels kept). by_ref: each zone
    refers to the set ("ref": "zoneset:<set>/<id>": a re-painted set updates the recipe)
    instead of copying its coordinates. checks: the checks every zone gets (default: a
    presence check, to be edited)."""
    checks = checks if checks is not None else [{"type": "presence", "params": {}}]
    out = []
    for z in zs.select(ids):
        geom = {"ref": f"{PREFIX}{zs.name}/{z.id}"} if by_ref else recipe_zone(z)
        out.append({"id": z.id, "label": z.name, **geom,
                    "checks": [dict(c) for c in checks]})
    return out


def parse_ref(ref) -> tuple:
    """A recipe zone's ref 'zoneset:<set>/<id>' -> (set, id), or ValueError."""
    m = _REF.fullmatch(ref.strip()) if isinstance(ref, str) else None
    if m is None:
        raise ValueError(f"ref {ref!r} must be \"zoneset:<set>/<zone id>\" (e.g. "
                         "zoneset:lid_zones/Z1; `scan zones` lists the sets)")
    return m.group(1), m.group(2)


def recipe_binder(store):
    """The resolver engine/recipes/schema.validate calls for a zone ref: (set, id) ->
    {geometry_id: {"zone": zone dict, "geometry_fp", "created", "preset"}} for every
    geometry the set is painted at (the run picks its scan's geometry: recipe_mask).
    LookupError when no such set / zone exists in this library."""
    def bind(set_name: str, zone_id: str) -> dict:
        out, errors = {}, []
        for geo in zstore.geometries_of(store, set_name):
            try:
                zs = zstore.load(store, geo, set_name)
                out[geo] = {"zone": zs.zone(zone_id).to_dict(), "geometry_fp": zs.fingerprint,
                            "created": zs.created, "preset": zs.preset,
                            "stale": zstore.stale_warning(store, zs)}
            except ZoneError as e:
                errors.append(str(e).splitlines()[0])
        if not out:
            raise LookupError(errors[0] if errors else
                              f"no zone set {set_name!r} in this library (`scan zones` lists "
                              f"them; paint it: scan zone-paint --name {set_name})")
        return out
    return bind


def recipe_mask(z, meta: Optional[dict]) -> np.ndarray:
    """The DMD mask of a recipe zone that refers to a zone set, on a scan whose meta is
    `meta` (its geometry_id + geometry_fp): the set painted AT THAT GEOMETRY, with the same
    geometry-bound rig settings. ZoneError otherwise (the run then cannot resolve the zone:
    its checks are INVALID, the disposition PULL - never a guess at another placement)."""
    ref = getattr(z, "ref", None) or {}
    spec, bound = ref.get("spec") or "zoneset:?", ref.get("bound")
    if not isinstance(bound, dict):
        raise ZoneError(f"zone {z.id}: {spec} was not resolved against a library (load the "
                        "recipe by name / file through the engine: scan recipe-run, "
                        "api.recipe_load)")
    geo = (meta or {}).get("geometry_id")
    b = bound.get(geo)
    if b is None:
        raise ZoneError(f"zone {z.id}: {spec} is not painted at this scan's geometry {geo!r} "
                        f"(painted at: {', '.join(sorted(bound)) or 'none'}). Zones are "
                        "projector px at one rig placement: paint the set with this preset "
                        f"(scan zone-paint --name {ref.get('set')})")
    diffs = rules.fingerprint_diffs(b.get("geometry_fp"), (meta or {}).get("geometry_fp"))
    if diffs:
        raise ZoneError(f"zone {z.id}: {spec} was painted at other rig settings "
                        f"({'; '.join(diffs)}): paint it again")
    m = Zone.from_dict(b["zone"]).mask(DMD_SHAPE)
    if not m.any():
        raise ZoneError(f"zone {z.id}: {spec} covers no DMD mirror")
    return m


def recipe_warnings(z, meta: Optional[dict]) -> list:
    """What a run should say about a recipe zone that refers to a zone set (after
    recipe_mask resolved it): the rig-state record changed since the set was painted."""
    b = ((getattr(z, "ref", None) or {}).get("bound") or {}).get((meta or {}).get("geometry_id"))
    return [f"zone {z.id}: {b['stale']}"] if b and b.get("stale") else []


def _names_set(name: str) -> "re.Pattern":
    """A text names the zone set `name`: 'zoneset:<name>' or a later term of a several-set
    text ('zoneset:a+<name>:Z1')."""
    return re.compile(rf"zoneset:(?:[A-Za-z0-9_.,:/-]+\+)*{re.escape(name)}(?![A-Za-z0-9_.-])",
                      re.I)


def recipe_refs(store, builtin_dir, name: str) -> list:
    """Recipes (built-in files and this library's user recipes) whose text names the zone set
    `name` (a zone ref or a scan.zone, also inside a several-set text): they stop working
    when it is deleted."""
    from ..recipes import library as recipe_lib
    rx = _names_set(name)
    files = dict(recipe_lib._builtin_files(Path(builtin_dir)))
    for n in recipe_lib.user_names(store):
        files[n] = recipe_lib.user_dir(store) / n / recipe_lib.RECIPE_FILE
    out = []
    for n, f in sorted(files.items()):
        try:
            if rx.search(Path(f).read_text(encoding="utf-8")):
                out.append(n)
        except OSError:
            continue
    return out


def station_refs(stations_dir, name: str) -> list:
    """Station configuration files (stations/*.json | *.yaml | *.yml) whose text names the
    zone set `name` (a job.zone or an inline scan zone): what they inspect changes when the
    set is painted over, and they PULL every part when it is deleted."""
    d = Path(stations_dir) if stations_dir else None
    if d is None or not d.is_dir():
        return []
    rx = _names_set(name)
    out = []
    for f in sorted(d.iterdir()):
        if f.suffix.lower() not in (".json", ".yaml", ".yml") or not f.is_file():
            continue
        try:
            if rx.search(f.read_text(encoding="utf-8")):
                out.append(f.name)
        except (OSError, UnicodeDecodeError):
            continue
    return out


def users(store, builtin_dir, stations_dir, name: str) -> dict:
    """Who names the zone set `name`: {"recipes": [...], "stations": [...]}."""
    return {"recipes": recipe_refs(store, builtin_dir, name),
            "stations": station_refs(stations_dir, name)}

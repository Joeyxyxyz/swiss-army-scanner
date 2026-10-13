"""
Using a painted zone set: scan masks, per-zone point counts, recipe zones, regions.

A zone set is named where a lit zone is named anywhere in the engine, as text:

    zoneset:<set>             every zone of the set (the union is lit)
    zoneset:<set>:Z1,Z3       only those zones
    zoneset:<set>/Z1          one zone (the form a recipe zone's `ref` uses)

`scan --zone`, `scan-fast --zone / --zones`, `scan-metric --zone`, a measurement's zone and a
recipe's scan.zone all take it (engine/capture/zones.resolve dispatches on the prefix, as it
does for atlas regions). The set is looked up AT THE PRESET'S GEOMETRY and its fingerprint
is compared with the preset's (rules 5 / 6): zones are projector px at one rig placement, so
a set painted at another geometry or other rig settings is refused, never silently used.

No camera mask is returned for a zone set: where its mirrors land in the camera image
depends on the depth of what they hit (parallax), so camera px are derived PER SCAN from
that scan's own decoded (col, row) - never from a stored projector map of another scene.

    scan_fast()        lit through the union, + per-zone point counts (zones.json in the scan)
    count_points()     decoded points per zone of one scan
    camera_region()    the camera px of a scan that decoded into the zones (Measure mode)
    recipe_zone(s)()   a zone -> a recipe zone in projector_px (rect | circle | polygon)
    recipe_binder()    resolves a recipe zone's ref "zoneset:<set>/<id>" (engine/recipes/schema)
    recipe_mask()      ... to its DMD mask on a given scan (geometry + fingerprint checked)
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
SCAN_FILE = "zones.json"                # per-zone counts, written into the metric scan's entry
_REF = re.compile(r"^zoneset:([A-Za-z0-9][A-Za-z0-9_.-]{0,63})/([A-Za-z0-9][A-Za-z0-9_.-]{0,31})$")


# ------------------------------------------------------------------ the text form

def is_spec(text) -> bool:
    return isinstance(text, str) and text.strip().lower().startswith(PREFIX)


def parse_spec(text, bare: bool = False) -> tuple:
    """'zoneset:lid', 'zoneset:lid:Z1,Z3', 'zoneset:lid/Z1' -> (set name, [ids] or None).
    bare: the prefix may be left out ('lid:Z1,Z3', the CLI's --zones)."""
    if not isinstance(text, str) or not text.strip():
        raise ZoneError("empty zone set; write <set> or <set>:Z1,Z3 (`scan zones` lists them)")
    t = text.strip()
    if t.lower().startswith(PREFIX):
        t = t[len(PREFIX):]
    elif not bare:
        raise ZoneError(f"{text!r} is not a zone set (zoneset:<set>[:Z1,Z3])")
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


def spec_text(name: str, ids=None) -> str:
    return f"{PREFIX}{name}" + (":" + ",".join(ids) if ids else "")


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
    """-> (ZoneSet, [ids] or None, canonical text). Unknown ids are refused here."""
    name, ids = parse_spec(text, bare=bare)
    zs = for_preset(store, preset, name)
    zs.select(ids)
    return zs, ids, spec_text(name, ids)


def _grid_cells(mask: np.ndarray) -> list:
    """The default grid's cells a DMD mask touches (what a base / scan records as zone_cells)."""
    from ..capture import zones as gz
    s = gz.DEFAULT_SPACING
    rows, cols = np.nonzero(mask)
    return sorted({(int(c), int(r)) for c, r in zip(cols // s, rows // s)})


def resolve(session, preset, text: str) -> tuple:
    """A zone-set spec as engine/capture/zones.resolve returns a zone: (DMD mask bool
    (1140, 912), None, cells). The camera mask is always None (see the module doc)."""
    zs, ids, _ = select(session.store, preset, text)
    m = zs.mask(ids)
    if not m.any():
        raise ZoneError(f"zone set {text!r} covers no DMD mirror: nothing would be lit")
    return m, None, _grid_cells(m)


def scan_mask(zs: ZoneSet, ids=None) -> np.ndarray:
    """The DMD mask a scan is lit through: the union of the zones, bool (1140, 912)."""
    return zs.mask(ids)


def export_mask(zs: ZoneSet, path, ids=None) -> Path:
    """Write the union mask as a .npy for `scan scan-fast --mask` (a NEW file, never replaced)."""
    p = Path(path)
    if p.suffix.lower() != ".npy":
        raise ZoneError(f"the mask file must end in .npy, got {p.name!r}")
    with open(p, "xb") as f:
        np.save(f, scan_mask(zs, ids))
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


def scan_fast(s, preset, zones, *, zone=None, label: str = "fast", steps: int = 4, periods=None,
              average: int = 2, min_hold_s: float = 0.12, axis: str = "xy", faces=None,
              margin_dmd_px: float = 6.0):
    """The HDMI fast scan lit through a zone set's zones (their union; every other mirror
    dark) -> the metric scan, with the per-zone point counts in its meta["zones"] and in
    zones.json of its library entry. axis "xy" (the default here) decodes rows too, so every
    point is assigned to its zone exactly; "x" is the faster columns-only set."""
    from ..reconstruct import fast_scan
    if zone is not None and str(zone).strip():
        raise ValueError("give a zone (grid cells / mask) or a zone set (zones), not both")
    zs, ids, text = select(s.store, preset, zones, bare=True)       # refused before the rig
    ms = fast_scan.scan_fast(s, preset, text, int(steps), periods or list(fast_scan.FAST_PERIODS),
                             average, float(min_hold_s), axis, label, faces, float(margin_dmd_px))
    fld = ms.field
    rep = count_points(zs, ids, fld.corr_x, fld.corr_y, ms.cloud.mask)
    stale = zstore.stale_warning(s.store, zs)
    if stale:
        rep["warnings"].append(stale)
    doc = {"kind": "scan_zones", "zoneset": zs.name, "spec": text,
           "ids": [z.id for z in zs.select(ids)],
           "names": {z.id: z.name for z in zs.select(ids)},
           "geometry_id": zs.geometry_id, "zoneset_created": zs.created, "scan_id": ms.id,
           "created": datetime.now().isoformat(timespec="seconds"),
           "units": {"counts": "metric points (mm) whose decoded DMD px is in the zone",
                     "dmd_px": "DMD mirrors of the zone", "outside": "points decoded outside "
                     "every selected zone (stray light, other lit areas)"}, **rep}
    s.store.write_json(Path(ms.path) / SCAN_FILE, json_safe(doc))
    if isinstance(ms.meta, dict):
        ms.meta["zones"] = doc
        ms.meta.setdefault("warnings", [])
        ms.meta["warnings"] = list(ms.meta["warnings"]) + rep["warnings"]
    return ms


def camera_region(s, preset, text: str, ms, shape, warnings: list, lit_zone="?",
                  need: bool = False) -> Optional[np.ndarray]:
    """Measure mode: the camera px of the scan `ms` that belong to a zone-set region - those
    whose decoded (col, row) is one of the zones' mirrors (derived from THIS scan). A
    columns-only scan can't be cut: the region is then whatever it decoded, which is only the
    zones when the scan was lit through exactly them (else refused)."""
    zs, ids, canon = select(s.store, preset, text)
    stale = zstore.stale_warning(s.store, zs)
    if stale and stale not in warnings:
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
        H, W = tuple(zs.dmd_shape)
        m = np.zeros(tuple(shape), bool)
        m[ok] = zs.mask(ids)[_index(cy[ok], H), _index(cx[ok], W)]
        if not m.any():
            raise ValueError(f"zone set {canon}: no camera px of this scan decoded into its "
                             "zones (were they lit, and does the camera see them? `scan "
                             f"zone-show {zs.name}`)")
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


def recipe_refs(store, builtin_dir, name: str) -> list:
    """Recipes (built-in files and this library's user recipes) whose text names the zone set
    `name` (a zone ref or a scan.zone): they stop working when it is deleted."""
    from ..recipes import library as recipe_lib
    rx = re.compile(rf"zoneset:{re.escape(name)}(?![A-Za-z0-9_.-])", re.I)
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

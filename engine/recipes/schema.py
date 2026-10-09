"""
Recipe schema (versioned data, JSON or YAML) and its validator.

A recipe is one work-state check, portable to any DMD projector + camera line:

    preset (the profile: rig geometry + capture settings)
      -> scan      how to capture (fast | metric | trigger), axis, lit zone
      -> part      the part frame (reference plane, origin) and the scan label
      -> zones     where to look: rect | circle | polygon | grid-cells, in camera px,
                   projector (DMD) px or part-frame mm (units always said)
           -> checks   what is recognised there (presence, position, height, flatness,
                       defect, edge, code, or a plugin), with their tolerances in mm / deg
      -> golden    optional reference scan (position / edge vs the good part): a golden
                   recorded by `scan recipe-golden` (by label, or 'latest' for this recipe)
                   or an exact metric scan id; never the scan being checked
      -> feedback  what the projector paints after the verdict (green outline = PASS,
                   red fill + cross at the fault = FAIL, amber = PULL)
      -> pull      the disposition on top of the verdict (engine/disposition.py): the guard
                   band (guard_band_pct of a tolerance next to its limit: PULL) and the doubt
                   reasons the line rejects instead of pulling
      -> outputs   run folder, result codes for the I/O registers, overlay image

validate(doc) -> Recipe, or RecipeError listing EVERY problem with its path
("zones[1].checks[0] (flatness): tol_mm must be > 0 (mm), got -1"). Hand-written: no
schema library is needed. Unknown keys are refused (a typo must not silently do nothing).
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .. import disposition as D

SCHEMA_VERSION = 1
SUPPORTED_VERSIONS = (1,)
SCAN_KINDS = ("fast", "metric", "trigger")
SHAPES = ("rect", "circle", "polygon", "grid-cells")
UNITS = ("camera_px", "projector_px", "part_mm")
COLOURS = {"red": (0, 0, 255), "green": (0, 255, 0), "blue": (255, 0, 0),
           "amber": (0, 191, 255), "yellow": (0, 255, 255), "white": (255, 255, 255),
           "cyan": (255, 255, 0), "magenta": (255, 0, 255)}      # BGR, as the HDMI window draws
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_GRID = re.compile(r"^[A-Za-z]+\d+(\s*:\s*[A-Za-z]+\d+)?(\s*,\s*[A-Za-z]+\d+(\s*:\s*[A-Za-z]+\d+)?)*"
                   r"(@\d+)?$")

TOP_KEYS = {"schema_version", "name", "description", "preset", "scan", "part", "golden",
            "zones", "feedback", "outputs", "pull", "_comment"}
SCAN_DEFAULTS = {"kind": "fast", "axis": "xy", "zone": None, "average": None, "steps": None,
                 "periods": None, "hold_s": None, "pack": None, "variant": None,
                 "codes": False, "rig_check": False}
PART_DEFAULTS = {"label": None, "frame": None, "mock_scene": None}
FRAME_DEFAULTS = {"reference": "dominant_plane", "plane_band_mm": 1.0, "origin_px": None}
FEEDBACK_DEFAULTS = {"enabled": True, "pass_colour": "green", "fail_colour": "red",
                     "invalid_colour": "amber", "pull_colour": "amber", "line_px": 4,
                     "fill_fail": True,
                     "fill_level": 0.6, "cross_px": 24, "text": True, "text_px": 28,
                     "hold_s": 5.0, "max_fit_px": 2.0, "photo": True}
OUTPUT_DEFAULTS = {"runs_dir": None, "overlay": True,
                   "result_codes": {"PASS": 1, "FAIL": 2, "INVALID": 3, "ABORTED": 4, "PULL": 5}}
# the disposition on top of the verdict (engine/disposition.py): guard band in % of a
# tolerance (0 = off) and the doubt reasons this line rejects instead of pulling ([] = none)
PULL_DEFAULTS = {"guard_band_pct": 0.0, "invalid_reasons_to_reject": []}
ZONE_KEYS = {"id", "label", "shape", "units", "coords", "checks", "datum", "feedback"}
CHECK_KEYS = {"type", "id", "params", "enabled"}
MAX_HOLD_S = 300.0


class RecipeError(ValueError):
    """The recipe is missing, unreadable or invalid; `.problems` lists every problem."""

    def __init__(self, message: str, problems: Optional[list] = None):
        super().__init__(message)
        self.problems = list(problems or [message])


@dataclass
class CheckSpec:
    type: str
    id: str
    params: dict
    check: object                       # the Check instance (validated params)

    def to_dict(self) -> dict:
        return {"type": self.type, "id": self.id, "params": dict(self.check.p)}


@dataclass
class ZoneSpec:
    id: str
    shape: str
    units: str
    coords: object
    checks: list = field(default_factory=list)       # [CheckSpec]
    label: str = ""
    datum: bool = False
    feedback: bool = True

    def to_dict(self) -> dict:
        return {"id": self.id, "label": self.label, "shape": self.shape, "units": self.units,
                "coords": self.coords, "datum": self.datum, "feedback": self.feedback,
                "checks": [c.to_dict() for c in self.checks]}


@dataclass
class Recipe:
    name: str
    preset: str
    zones: list                          # [ZoneSpec]
    schema_version: int = SCHEMA_VERSION
    description: str = ""
    scan: dict = field(default_factory=lambda: dict(SCAN_DEFAULTS))
    part: dict = field(default_factory=dict)
    golden: Optional[dict] = None
    feedback: dict = field(default_factory=lambda: dict(FEEDBACK_DEFAULTS))
    outputs: dict = field(default_factory=lambda: copy.deepcopy(OUTPUT_DEFAULTS))
    pull: dict = field(default_factory=lambda: copy.deepcopy(PULL_DEFAULTS))
    source: Optional[Path] = None
    raw: dict = field(default_factory=dict, repr=False)

    def zone(self, zid: str) -> ZoneSpec:
        for z in self.zones:
            if z.id == zid:
                return z
        raise KeyError(zid)

    @property
    def needs(self) -> set:
        out = set()
        for z in self.zones:
            for c in z.checks:
                out |= c.check.needs_for()
        return out

    def summary(self) -> dict:
        return {"name": self.name, "schema_version": self.schema_version,
                "description": self.description, "preset": self.preset,
                "source": str(self.source) if self.source else None,
                "scan": dict(self.scan), "golden": self.golden,
                "pull": copy.deepcopy(self.pull),
                "zones": [{"id": z.id, "shape": z.shape, "units": z.units,
                           "checks": [c.type for c in z.checks]} for z in self.zones]}

    def to_dict(self) -> dict:
        """The normalised recipe (every default filled in)."""
        return {"schema_version": self.schema_version, "name": self.name,
                "description": self.description, "preset": self.preset,
                "scan": dict(self.scan), "part": copy.deepcopy(self.part),
                "golden": copy.deepcopy(self.golden), "zones": [z.to_dict() for z in self.zones],
                "feedback": dict(self.feedback), "outputs": copy.deepcopy(self.outputs),
                "pull": copy.deepcopy(self.pull)}


# ------------------------------------------------------------------ reading

def read_doc(path) -> dict:
    """A recipe file (.json, or .yaml / .yml with PyYAML) -> the raw mapping."""
    path = Path(path)
    if not path.is_file():
        raise RecipeError(f"recipe file not found: {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml  # optional: PyYAML
        except ImportError as e:
            raise RecipeError(f"{path.name}: YAML recipes need PyYAML (pip install pyyaml), or "
                              "use the .json form") from e
        try:
            doc = yaml.safe_load(text)
        except yaml.YAMLError as e:
            raise RecipeError(f"{path.name}: not valid YAML ({e})") from e
    else:
        try:
            doc = json.loads(text)
        except json.JSONDecodeError as e:
            raise RecipeError(f"{path.name}: not valid JSON ({e})") from e
    if not isinstance(doc, dict):
        raise RecipeError(f"{path.name}: the top level must be a mapping")
    return doc


def load(path) -> Recipe:
    path = Path(path)
    r = validate(read_doc(path), source=path)
    return r


# ------------------------------------------------------------------ validation

class _Errors:
    def __init__(self):
        self.items: list = []

    def add(self, where: str, msg: str) -> None:
        self.items.append(f"{where}: {msg}" if where else msg)


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _colour(v, where, E: _Errors):
    if isinstance(v, str):
        if v.lower() not in COLOURS:
            E.add(where, f"colour {v!r} is not one of {', '.join(COLOURS)} (or [b, g, r] 0-255)")
            return None
        return list(COLOURS[v.lower()])
    if isinstance(v, (list, tuple)) and len(v) == 3 and all(_num(x) and 0 <= x <= 255 for x in v):
        return [int(x) for x in v]
    E.add(where, f"colour must be a name ({', '.join(COLOURS)}) or [b, g, r] 0-255, got {v!r}")
    return None


def _section(doc: dict, key: str, defaults: dict, E: _Errors) -> dict:
    v = doc.get(key)
    if v is None:
        return copy.deepcopy(defaults)
    if not isinstance(v, dict):
        E.add(key, "must be a mapping")
        return copy.deepcopy(defaults)
    unknown = sorted(set(v) - set(defaults) - {"_comment"})
    if unknown:
        E.add(key, f"unknown keys {unknown} (known: {', '.join(sorted(defaults))})")
    out = copy.deepcopy(defaults)
    out.update({k: v[k] for k in v if k in defaults})
    return out


def validate(doc: dict, source: Optional[Path] = None) -> Recipe:
    """Raw mapping -> Recipe (defaults filled in), or RecipeError listing every problem."""
    from . import checks as registry
    E = _Errors()
    if not isinstance(doc, dict):
        raise RecipeError("a recipe is a mapping (JSON object)")
    unknown = sorted(set(doc) - TOP_KEYS)
    if unknown:
        E.add("", f"unknown top-level keys {unknown} (known: {', '.join(sorted(TOP_KEYS - {'_comment'}))})")
    ver = doc.get("schema_version")
    if ver is None:
        E.add("schema_version", f"is required (this engine reads {SUPPORTED_VERSIONS})")
    elif isinstance(ver, bool) or not isinstance(ver, int) or ver not in SUPPORTED_VERSIONS:
        E.add("schema_version", f"{ver!r} is not supported (this engine reads {SUPPORTED_VERSIONS})")
    name = doc.get("name")
    if not isinstance(name, str) or not _ID.match(name):
        E.add("name", "is required: letters, digits, _ . - (max 64), e.g. box_lid_check")
    preset = doc.get("preset")
    if not isinstance(preset, str) or not _ID.match(preset):
        E.add("preset", "is required: the preset (profile) this recipe pins, e.g. bench_9in_metric")
    desc = doc.get("description", "")
    if not isinstance(desc, str):
        E.add("description", "must be text")
        desc = ""

    scan = _section(doc, "scan", SCAN_DEFAULTS, E)
    _scan_rules(scan, E)
    part = _section(doc, "part", PART_DEFAULTS, E)
    part["frame"] = _frame(part.get("frame"), E)
    if part["label"] is not None and (not isinstance(part["label"], str)
                                      or not _ID.match(part["label"])):
        E.add("part.label", "must be a label [A-Za-z0-9_.-] or null (null: the recipe name)")
    if part["mock_scene"] is not None and not isinstance(part["mock_scene"], str):
        E.add("part.mock_scene", "must be text (a mock scene, see `scan --help`), or null")

    golden = doc.get("golden")
    if golden is not None:
        if not isinstance(golden, dict) or set(golden) - {"scan", "_comment"} \
                or not isinstance(golden.get("scan"), str) or not golden["scan"].strip():
            E.add("golden", "must be {\"scan\": \"<golden label | latest | exact metric scan "
                  "id>\"} or null (`scan recipe-golden` records a golden)")
            golden = None
        else:
            golden = {"scan": golden["scan"].strip()}

    fb = _section(doc, "feedback", FEEDBACK_DEFAULTS, E)
    _feedback_rules(fb, E)
    outs = _section(doc, "outputs", OUTPUT_DEFAULTS, E)
    _output_rules(outs, E)
    pull = _section(doc, "pull", PULL_DEFAULTS, E)
    _pull_rules(pull, E)

    zones = _zones(doc.get("zones"), scan, E, registry)
    ref = part["frame"].get("reference")
    if isinstance(ref, dict):
        z = next((z for z in zones if z.id == ref.get("zone")), None)
        if z is None:
            E.add("part.frame.reference.zone", f"no zone {ref.get('zone')!r} in zones")
        elif z.units == "part_mm":
            E.add("part.frame.reference.zone", f"zone {z.id} is in part_mm: the reference zone "
                  "DEFINES the part frame, so give it in camera_px or projector_px")
    needs = set()
    for z in zones:
        for c in z.checks:
            needs |= c.check.needs_for()
            for ref_zone in getattr(c.check, "zone_refs", lambda: [])():
                if not any(o.id == ref_zone for o in zones):
                    E.add(f"zones[{z.id}].checks[{c.id}]", f"refers to zone {ref_zone!r}, which "
                          "is not in this recipe")
    if "golden" in needs and golden is None:
        E.add("golden", "a check compares with the good part (position / edge position): set "
              "golden: {\"scan\": \"<label of a good-part scan>\"} (scan recipe-golden makes one)")
    if any(z.units == "projector_px" for z in zones) and scan.get("axis") != "xy":
        E.add("scan.axis", "projector_px / grid-cells zones need both DMD axes decoded: axis xy")
    if fb.get("enabled") and scan.get("axis") != "xy":
        E.add("scan.axis", "projected feedback maps camera px to the DMD from the scan's decoded "
              "rows AND columns: axis xy (or feedback.enabled false)")
    if E.items:
        who = f"recipe {name!r}" if isinstance(name, str) else "recipe"
        raise RecipeError(f"{who}: {len(E.items)} problem(s):\n  - " + "\n  - ".join(E.items),
                          E.items)
    return Recipe(name=name, preset=preset, zones=zones, schema_version=ver, description=desc,
                  scan=scan, part=part, golden=golden, feedback=fb, outputs=outs, pull=pull,
                  source=Path(source) if source else None, raw=copy.deepcopy(doc))


def _scan_rules(scan: dict, E: _Errors) -> None:
    if scan["kind"] not in SCAN_KINDS:
        E.add("scan.kind", f"{scan['kind']!r} must be one of {', '.join(SCAN_KINDS)}")
    if scan["axis"] not in ("x", "xy"):
        E.add("scan.axis", f"{scan['axis']!r} must be 'xy' (both DMD axes) or 'x' (columns only)")
    if scan["zone"] is not None and (not isinstance(scan["zone"], str) or not scan["zone"].strip()):
        E.add("scan.zone", "must be grid cells ('F10:J12', 'F10:J12@96'), 'atlas:<name>/<region>' "
              "or null (whole field)")
    for k, lo in (("average", 1), ("steps", 3)):
        v = scan[k]
        if v is not None and (isinstance(v, bool) or not isinstance(v, int) or v < lo):
            E.add(f"scan.{k}", f"must be a whole number >= {lo} or null")
    p = scan["periods"]
    if p is not None and (not isinstance(p, list) or not p or not all(_num(v) and v > 0 for v in p)):
        E.add("scan.periods", "must be a list of positive numbers (DMD px) or null")
    h = scan["hold_s"]
    if h is not None and (not _num(h) or not 0 <= h <= 5):
        E.add("scan.hold_s", "must be 0..5 s (HDMI hold per pattern) or null")
    for k in ("codes", "rig_check"):
        if not isinstance(scan[k], bool):
            E.add(f"scan.{k}", "must be true or false")
    if scan["kind"] == "trigger":
        if not isinstance(scan["pack"], str) or not scan["pack"].strip():
            E.add("scan.pack", "a trigger scan needs the flash pack folder (scan.pack)")
    elif scan["pack"] is not None:
        E.add("scan.pack", "only for scan.kind trigger")


def _frame(v, E: _Errors) -> dict:
    out = dict(FRAME_DEFAULTS)
    if v is None:
        return out
    if not isinstance(v, dict):
        E.add("part.frame", "must be a mapping")
        return out
    unknown = sorted(set(v) - set(FRAME_DEFAULTS) - {"_comment"})
    if unknown:
        E.add("part.frame", f"unknown keys {unknown} (known: {', '.join(FRAME_DEFAULTS)})")
    out.update({k: v[k] for k in v if k in FRAME_DEFAULTS})
    ref = out["reference"]
    if isinstance(ref, dict):
        if set(ref) != {"zone"} or not isinstance(ref.get("zone"), str):
            E.add("part.frame.reference", "must be 'dominant_plane' or {\"zone\": \"<zone id>\"}")
    elif ref != "dominant_plane":
        E.add("part.frame.reference", "must be 'dominant_plane' or {\"zone\": \"<zone id>\"}")
    b = out["plane_band_mm"]
    if not _num(b) or not 0 < b <= 20:
        E.add("part.frame.plane_band_mm", "must be > 0 and <= 20 (mm)")
    o = out["origin_px"]
    if o is not None and (not isinstance(o, list) or len(o) != 2 or not all(_num(x) for x in o)):
        E.add("part.frame.origin_px", "must be [x, y] camera px or null (null: the optical axis)")
    return out


def _feedback_rules(fb: dict, E: _Errors) -> None:
    if not isinstance(fb["enabled"], bool):
        E.add("feedback.enabled", "must be true or false")
    for k in ("pass_colour", "fail_colour", "invalid_colour", "pull_colour"):
        c = _colour(fb[k], f"feedback.{k}", E)
        if c is not None:
            fb[k] = c
    for k, lo, hi in (("line_px", 1, 40), ("cross_px", 4, 300), ("text_px", 8, 200)):
        v = fb[k]
        if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi:
            E.add(f"feedback.{k}", f"must be a whole number {lo}..{hi} ({'DMD px' if k != 'text_px' else 'camera px'})")
    if not _num(fb["fill_level"]) or not 0 <= fb["fill_level"] <= 1:
        E.add("feedback.fill_level", "must be 0..1 (fraction of the fail colour for the fill)")
    if not _num(fb["hold_s"]) or not 0 <= fb["hold_s"] <= MAX_HOLD_S:
        E.add("feedback.hold_s", f"must be 0..{MAX_HOLD_S:g} s")
    if not _num(fb["max_fit_px"]) or not fb["max_fit_px"] > 0:
        E.add("feedback.max_fit_px", "must be > 0 (DMD px: the camera -> DMD fit's p95 residual)")
    for k in ("fill_fail", "text", "photo"):
        if not isinstance(fb[k], bool):
            E.add(f"feedback.{k}", "must be true or false")


def _output_rules(outs: dict, E: _Errors) -> None:
    if outs["runs_dir"] is not None and not isinstance(outs["runs_dir"], str):
        E.add("outputs.runs_dir", "must be a folder path or null")
    if not isinstance(outs["overlay"], bool):
        E.add("outputs.overlay", "must be true or false")
    rc = outs["result_codes"]
    want = {"PASS", "FAIL", "INVALID", "ABORTED"}
    if not isinstance(rc, dict) or set(rc) - {"PULL"} != want or not all(
            isinstance(v, int) and not isinstance(v, bool) and 0 <= v <= 65535 for v in rc.values()):
        E.add("outputs.result_codes", "must map PASS, FAIL, INVALID, ABORTED (and optionally PULL, "
              "default 5 or the next code they leave free) to register values 0..65535")
        return
    if "PULL" not in rc:        # older recipes: PULL gets its own code, one they leave free
        used = set(rc.values())
        rc = outs["result_codes"] = {**rc, "PULL": next(
            c for c in range(OUTPUT_DEFAULTS["result_codes"]["PULL"], 65536) if c not in used)}
    # fail closed: a doubt or a reject never reads as the PASS code on the register
    same = [k for k in ("FAIL", "INVALID", "ABORTED", "PULL") if rc[k] == rc["PASS"]]
    if same:
        E.add("outputs.result_codes", f"{', '.join(same)} must not share the PASS code "
              f"{rc['PASS']} (a PLC would read them as a pass)")
    other = [k for k in ("FAIL", "INVALID", "ABORTED") if rc[k] == rc["PULL"]]
    if rc["PULL"] == 0 or (other and rc["PULL"] != rc["PASS"]):
        E.add("outputs.result_codes", f"PULL {rc['PULL']} must differ from 0 (no result yet) "
              f"and from {', '.join(f'{k} {rc[k]}' for k in other) or 'every other code'} (a "
              "PLC must tell a pulled part from a rejected or an unmeasured one): give PULL "
              "its own code")


def _pull_rules(pull: dict, E: _Errors) -> None:
    """pull.guard_band_pct: 0 <= pct < 100, finite; pull.invalid_reasons_to_reject: reasons
    from engine/disposition.REJECTABLE (each message names its key)."""
    try:
        pull["guard_band_pct"] = D.check_pct(pull["guard_band_pct"], "pull.guard_band_pct")
    except ValueError as e:
        E.add("", str(e))
    try:
        pull["invalid_reasons_to_reject"] = D.check_reject_list(
            pull["invalid_reasons_to_reject"], "pull.invalid_reasons_to_reject")
    except ValueError as e:
        E.add("", str(e))


def _zones(v, scan: dict, E: _Errors, registry) -> list:
    if not isinstance(v, list) or not v:
        E.add("zones", "at least one zone is required: [{id, shape, units, coords, checks}]")
        return []
    out, seen = [], set()
    for i, z in enumerate(v):
        where = f"zones[{i}]"
        if not isinstance(z, dict):
            E.add(where, "must be a mapping")
            continue
        zid = z.get("id")
        if isinstance(zid, str) and _ID.match(zid):
            where = f"zones[{i}] ({zid})"
            if zid in seen:
                E.add(where, f"duplicate zone id {zid!r}")
            seen.add(zid)
        else:
            E.add(where, "id is required: letters, digits, _ . -")
            zid = f"zone{i}"
        unknown = sorted(set(z) - ZONE_KEYS - {"_comment"})
        if unknown:
            E.add(where, f"unknown keys {unknown} (known: {', '.join(sorted(ZONE_KEYS))})")
        shape = z.get("shape")
        if shape not in SHAPES:
            E.add(where, f"shape {shape!r} must be one of {', '.join(SHAPES)}")
            continue
        units = z.get("units", "projector_px" if shape == "grid-cells" else None)
        if units not in UNITS:
            E.add(where, f"units {units!r} must be one of {', '.join(UNITS)} (say which: camera "
                  "px, native DMD px, or mm in the part frame)")
            continue
        if shape == "grid-cells" and units != "projector_px":
            E.add(where, "grid-cells are projector (DMD) cells: units projector_px")
        coords = _coords(shape, z.get("coords"), where, E)
        datum = z.get("datum", False)
        if not isinstance(datum, bool):
            E.add(where, "datum must be true or false")
            datum = False
        zfb = z.get("feedback", True)
        if not isinstance(zfb, bool):
            E.add(where, "feedback must be true or false")
            zfb = True
        label = z.get("label", "")
        if not isinstance(label, str):
            E.add(where, "label must be text")
            label = ""
        checks = _checks(z.get("checks"), where, shape, units, datum, E, registry)
        out.append(ZoneSpec(id=zid, shape=shape, units=units, coords=coords, checks=checks,
                            label=label, datum=datum, feedback=zfb))
    return out


def _pt(p) -> bool:
    return isinstance(p, (list, tuple)) and len(p) == 2 and all(_num(x) for x in p)


def _coords(shape: str, c, where: str, E: _Errors):
    if shape == "rect":
        if (isinstance(c, (list, tuple)) and len(c) == 4 and all(_num(x) for x in c)
                and c[2] > c[0] and c[3] > c[1]):
            return [float(x) for x in c]
        E.add(where, f"rect coords must be [x0, y0, x1, y1] with x1 > x0, y1 > y0, got {c!r}")
    elif shape == "circle":
        if isinstance(c, dict) and set(c) == {"center", "radius"} and _pt(c["center"]) \
                and _num(c["radius"]) and c["radius"] > 0:
            return {"center": [float(x) for x in c["center"]], "radius": float(c["radius"])}
        E.add(where, f"circle coords must be {{\"center\": [x, y], \"radius\": r > 0}}, got {c!r}")
    elif shape == "polygon":
        if isinstance(c, (list, tuple)) and len(c) >= 3 and all(_pt(p) for p in c):
            return [[float(x), float(y)] for x, y in c]
        E.add(where, f"polygon coords must be >= 3 points [[x, y], ...], got {c!r}")
    elif shape == "grid-cells":
        if isinstance(c, str) and _GRID.match(c.strip()):
            return c.strip()
        E.add(where, f"grid-cells coords must be cells like 'F10:J12' or 'F10:J12@96', got {c!r}")
    return None


def _checks(v, where: str, shape: str, units: str, datum: bool, E: _Errors, registry) -> list:
    if v is None:
        v = []
    if not isinstance(v, list):
        E.add(where, "checks must be a list")
        return []
    if not v and not datum:
        E.add(where, "checks: at least one check (or datum: true for a reference-only zone)")
    out, seen = [], set()
    for j, c in enumerate(v):
        cw = f"{where}.checks[{j}]"
        if not isinstance(c, dict):
            E.add(cw, "must be a mapping {type, params}")
            continue
        unknown = sorted(set(c) - CHECK_KEYS - {"_comment"})
        if unknown:
            E.add(cw, f"unknown keys {unknown} (known: type, id, params, enabled; the check's "
                  "settings go under params)")
        ctype = c.get("type")
        try:
            cls = registry.get(ctype) if isinstance(ctype, str) else None
        except KeyError as e:
            E.add(cw, str(e.args[0]))
            continue
        if cls is None:
            E.add(cw, "type is required (a check id: see `scan recipe-checks`)")
            continue
        cid = c.get("id", ctype)
        if not isinstance(cid, str) or not _ID.match(cid):
            E.add(cw, "id must be a name [A-Za-z0-9_.-]")
            cid = ctype
        if cid in seen:
            E.add(cw, f"duplicate check id {cid!r} in this zone (give each an id)")
        seen.add(cid)
        cw = f"{where}.checks[{j}] ({ctype})"
        if c.get("enabled", True) is False:
            continue
        if cls.shapes and shape not in cls.shapes:
            E.add(cw, f"works on {', '.join(cls.shapes)} zones, not {shape}")
        try:
            inst = cls(c.get("params") or {}, check_id=cid)
        except Exception as e:              # noqa: BLE001 - every param problem, with its path
            E.add(cw, str(e))
            continue
        out.append(CheckSpec(type=ctype, id=cid, params=dict(inst.p), check=inst))
    return out

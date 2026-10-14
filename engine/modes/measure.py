"""
MEASURE mode: the scanner as an instrument. Scan, report numbers, decide nothing.

    run()        one measurement in a NEW run folder:
                     result.json    the document (schema MEASURE_SCHEMA, below)
                     values.csv     one row per value: value, units, n points, uncertainty
                     overlay.png    the camera view with what was measured drawn on it
                                    (camera resolution: 1 image px = 1 camera px)
                     scan.ply       the scan's point cloud, mm (the first repeat)
                     spec.json      the measurement spec that was run (run it again with
                                    `scan measure-run --spec`, promote it with to_check())
                     run.log.jsonl  one JSON event per line
    spec_from()  a measurement spec (a mapping / JSON) -> Spec, every problem listed
    to_check()   a spec + tolerances -> a check the Quick check and Station modes can run
                 (a recipe, or a single-job station config)

What is measured (`what`; engine/measure/values.py, thin wrappers over the engine's fits):
flatness (RMS, p2-p98, tilt) of a region; step height between two regions (plane to
plane); distance between two picked camera px (their 3-D points) and point to plane;
cylinder radius / axis; end-face diameter (a cylinder's circular end face toward the camera:
the station measurement's rim circle); bounding dimensions L x W x H of the part above the
reference plane; deviation vs a golden scan (values only). A region is an `roi` ([x0, y0,
x1, y1] camera px) or a grid `zone`.

Every value carries its units, the number of points behind it and an UNCERTAINTY:
    repeats = 1   the predicted depth precision at that distance (measure/precision.py: the
                  phase-noise model with this scan's fringe offset / modulation) or the
                  measured depth noise of the px THE VALUE RESTS ON (its region, the picked
                  points' surroundings - not the whole scan's: a dark part in a quiet scan
                  is noisier than the scan), whichever is larger; for a lateral quantity (a
                  length, a width, a distance between picked px) the camera px size at that
                  distance (an outline is only as sharp as a px). Where a depth error counts
                  more than once in the value (the radius of a weak arc) it is multiplied
                  accordingly. Noise only: a systematic calibration error is not in it
                  (`scan accuracy-block` proves the mm).
    repeats > 1   the sample standard deviation of the value over the repeat scans
                  (method "repeatability"), but never less than what one scan resolves /
                  sqrt(repeats) (method "single_scan_noise/sqrt(n)": two scans that happen
                  to agree are not a measurement of the spread); a lateral quantity keeps
                  the camera px size with it (the part does not move between the repeats, so
                  they cannot average the px sampling away). The repeats are DIFFERENT
                  scans: the same stored scan named twice is refused.
and `near_resolution_limit` + a note when the value itself is within 3 x that depth noise or
the px size (NEAR_LIMIT): the number is then mostly the instrument, not the part.

The validity statement replaces the verdict. status MEASURED, or NOT_MEASURABLE with the
`reason`, a message and how to fix it - never a px presented as mm, never a guess:
    uncalibrated       no stereo calibration at the preset's geometry and settings
    stale_calibration  a stored scan made with another calibration / at other rig settings
    rig_unverified     a MOBILE rig (preset rig.mount) not verified in this session (also
                       after its rig check said MOVED)
    rig_check          a FIXED rig whose rig check said MOVED / UNKNOWN, unanswered; or a
                       stored scan made while that stood, or just before it
    too_few_points     too few decoded points in the region / at the picked px
    incomplete         the number would not be the whole of what it names: the part is cut
                       by the image border or the roi (dims), a patch of it is too far from
                       the golden to be compared (deviation), no rim circle (endface)
    size_mismatch      endface: a face reads more than 4 % off the diameter given (its flood
                       outline): the calibration or the lens is suspect, so no mm of this
                       scan is reported (nothing is rescaled)
    invalid            it is not what was asked for: no cylinder in the region (flat within
                       the noise, or the points do not lie on one), a picked px on an edge
    golden, aborted, refused, error
(engine/rigstate.py decides the two rig reasons; engine/measure/values.py the rest).

A measurement is not a run of a check: `result` is null and `disposition` is "NONE"
(engine/disposition.NO_DISPOSITION), on every path. A flow, a PLC gateway or the command
bus that reads this document as a run gets a PULL, never a CONTINUE.

Exit codes (`scan measure-run`): 0 MEASURED, 5 NOT_MEASURABLE; 1 aborted, 2 refused, 4
crashed (all three NOT_MEASURABLE in the document).
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .. import config, eventlog, rigstate, rules
from .. import disposition as D
from .. import runs as R
from ..measure import values as V

MEASURE_SCHEMA = "measure-0.1"
MODE = "measure"
WHATS = ("flatness", "step", "distance", "cylinder", "endface", "dims", "deviation")
MAX_FACES = 8
STATUSES = ("MEASURED", "NOT_MEASURABLE")
SPEC_KIND = "measurement_spec"
SPEC_VERSION = 1
SCAN_KINDS = ("metric", "fast")         # api.scan_metric (both axes, the preset's set) | scan_fast
MAX_REPEATS = 50
NEAR_LIMIT = 3.0                        # a value within this x the resolution: say so
COMPARE_MAX_DIST_MM = 5.0               # deviation: a point further from the golden is not compared
RESULT_FILE, VALUES_FILE, OVERLAY_FILE = "result.json", "values.csv", "overlay.png"
PLY_FILE, SPEC_FILE, LOG_FILE = "scan.ply", "spec.json", "run.log.jsonl"
EXIT_CODES = {"MEASURED": 0, "NOT_MEASURABLE": 5}
EXIT_BY_REASON = {"aborted": 1, "refused": 2, "error": 4}
CSV_FIELDS = ("name", "label", "value", "units", "n_points", "uncertainty", "uncertainty_units",
              "uncertainty_method", "n_repeats", "near_resolution_limit", "notes")
HOW_TO_FIX = {
    "uncalibrated": ("calibrate it first, with this preset: `scan --preset {preset} charuco` "
                     "(print at 100 %, caliper a square), `scan --preset {preset} calib-pose` "
                     "per board pose (12-20, tilted +-30 deg), then `scan --preset {preset} "
                     "calibrate-stereo --square MM` (mock: `scan --mock --preset {preset} "
                     "calibrate-stereo --mock-poses 12`)"),
    "stale_calibration": ("scan again with the calibration in use (a stored scan: `scan "
                          "triangulate <scan>` re-triangulates it)"),
    "too_few_points": ("check the part, the roi / zone and the exposure (`scan signal`), then "
                       "measure again"),
    "golden": ("name a golden metric scan made at this geometry with the calibration in use "
               "(`scan metric` lists them, `scan scan-metric <label>` makes one)"),
    "incomplete": "check the part's placement against the golden, then measure again",
    "size_mismatch": ("run `scan rig-check` and recalibrate if the lens (zoom / focus) or the "
                      "rig moved since the calibration; or give the diameter of the bright "
                      "flat the flood shows (face_diameter_mm), not the cylinder's"),
    "invalid": "measure again; `scan signal` checks the setup",
    "aborted": "measure again",
    "refused": "fix the input the message names",
    "error": "see the message (the CLI's --debug prints the trace)",
}
_SPEC_KEYS = {"schema_version", "kind", "name", "preset", "what", "roi", "roi2", "zone",
              "zone2", "p1", "p2", "golden", "radius_mm", "axis", "min_height_mm",
              "face_diameter_mm", "n_faces", "repeats", "scan", "_comment"}
_SCAN_KEYS = {"kind", "steps", "periods", "average"}


# ------------------------------------------------------------------ the spec

@dataclass
class Spec:
    """What to measure, where and how (the document `scan measure-run --spec` reads)."""
    what: str
    preset: Optional[str] = None            # the triangulation preset it was made for
    name: str = ""
    roi: Optional[list] = None              # region A: [x0, y0, x1, y1] camera px
    roi2: Optional[list] = None             # region B (step)
    zone: Optional[str] = None              # region A as grid cells (also lights only them)
    zone2: Optional[str] = None             # region B as grid cells
    p1: Optional[list] = None               # picked camera px [x, y]
    p2: Optional[list] = None
    golden: Optional[str] = None            # deviation: the golden metric scan (id / label)
    radius_mm: Optional[float] = None       # cylinder: fix the radius (it is then not measured)
    axis: str = "auto"                      # cylinder: axis hint x | y | z | auto
    min_height_mm: float = 2.0              # dims: the part stands at least this far out
    face_diameter_mm: Optional[float] = None    # endface: the expected diameter (to find it)
    n_faces: int = 1                        # endface: how many end faces are in view
    repeats: int = 1
    scan: dict = field(default_factory=lambda: {"kind": "metric", "steps": None,
                                                "periods": None, "average": None})

    def to_dict(self) -> dict:
        return {"schema_version": SPEC_VERSION, "kind": SPEC_KIND, **asdict(self)}


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _whole(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def spec_from(d) -> Spec:
    """A measurement spec mapping -> Spec, or ValueError listing EVERY problem. Unknown keys
    are problems (a typo never silently measures something else)."""
    if isinstance(d, Spec):
        d = d.to_dict()
    if not isinstance(d, dict):
        raise ValueError("a measurement spec is a mapping (JSON object) with at least `what`")
    bad = []
    extra = sorted(set(d) - _SPEC_KEYS)
    if extra:
        bad.append(f"unknown keys {extra}")
    if d.get("kind") not in (None, SPEC_KIND):
        bad.append(f"kind {d.get('kind')!r} is not {SPEC_KIND!r}")
    if d.get("schema_version") not in (None, SPEC_VERSION):
        bad.append(f"schema_version {d.get('schema_version')!r} is not {SPEC_VERSION}")
    what = d.get("what")
    if isinstance(what, str):
        what = what.strip().lower().replace("-", "_")
    if what not in WHATS:
        bad.append(f"what {d.get('what')!r} is not one of {', '.join(WHATS)}")
    out = {"what": what}
    for k in ("roi", "roi2"):
        v = d.get(k)
        if v is not None:
            if (not isinstance(v, (list, tuple)) or len(v) != 4 or not all(_whole(x) for x in v)
                    or min(v) < 0 or v[2] <= v[0] or v[3] <= v[1]):
                bad.append(f"{k} must be [x0, y0, x1, y1] camera px (whole numbers >= 0, x1 > "
                           f"x0, y1 > y0), got {v!r}")
            else:
                out[k] = [int(x) for x in v]
    for k in ("p1", "p2"):
        v = d.get(k)
        if v is not None:
            if not isinstance(v, (list, tuple)) or len(v) != 2 or not all(
                    _num(x) and x >= 0 for x in v):
                bad.append(f"{k} must be [x, y] camera px (>= 0), got {v!r}")
            else:
                out[k] = [float(x) for x in v]
    for k in ("zone", "zone2", "golden", "preset", "name"):
        v = d.get(k)
        if v is not None and v != "":
            if not isinstance(v, str) or not v.strip():
                bad.append(f"{k} must be text, got {v!r}")
            else:
                out[k] = v.strip()
    r = d.get("radius_mm")
    if r is not None:
        if not _num(r) or r <= 0:
            bad.append(f"radius_mm must be a positive, finite number (mm), got {r!r}")
        else:
            out["radius_mm"] = float(r)
    ax = d.get("axis", "auto")
    if ax not in V.AXES:
        bad.append(f"axis must be one of {', '.join(V.AXES)}, got {ax!r}")
    else:
        out["axis"] = ax
    mh = d.get("min_height_mm", 2.0)
    if not _num(mh) or mh <= 0:
        bad.append(f"min_height_mm must be a positive, finite number (mm), got {mh!r}")
    else:
        out["min_height_mm"] = float(mh)
    fd = d.get("face_diameter_mm")
    if fd is not None:
        if not _num(fd) or fd <= 0:
            bad.append(f"face_diameter_mm must be a positive, finite number (mm), got {fd!r}")
        else:
            out["face_diameter_mm"] = float(fd)
    nf = d.get("n_faces", 1)
    if not _whole(nf) or not 1 <= nf <= MAX_FACES:
        bad.append(f"n_faces must be a whole number 1..{MAX_FACES}, got {nf!r}")
    else:
        out["n_faces"] = int(nf)
    n = d.get("repeats", 1)
    if not _whole(n) or not 1 <= n <= MAX_REPEATS:
        bad.append(f"repeats must be a whole number 1..{MAX_REPEATS}, got {n!r}")
    else:
        out["repeats"] = int(n)
    sc = d.get("scan") or {}
    scan = {"kind": "metric", "steps": None, "periods": None, "average": None}
    if not isinstance(sc, dict):
        bad.append("scan must be a mapping {kind, steps, periods, average}")
    else:
        if set(sc) - _SCAN_KEYS:
            bad.append(f"scan: unknown keys {sorted(set(sc) - _SCAN_KEYS)}")
        kind = sc.get("kind") or "metric"
        if kind not in SCAN_KINDS:
            bad.append(f"scan.kind must be one of {', '.join(SCAN_KINDS)}, got {kind!r}")
        steps, avg, per = sc.get("steps"), sc.get("average"), sc.get("periods")
        if steps is not None and (not _whole(steps) or steps < 3):
            bad.append("scan.steps must be a whole number >= 3 or null")
        if avg is not None and (not _whole(avg) or avg < 1):
            bad.append("scan.average must be a whole number >= 1 or null")
        if per is not None and (not isinstance(per, (list, tuple)) or not per or not all(
                _num(x) and x > 0 for x in per)):
            bad.append("scan.periods must be a list of positive numbers (DMD px) or null")
        scan = {"kind": kind, "steps": steps, "average": avg,
                "periods": [float(x) for x in per] if isinstance(per, (list, tuple)) and per
                and all(_num(x) for x in per) else None}
    out["scan"] = scan
    a = out.get("roi") or out.get("zone")
    b = out.get("roi2") or out.get("zone2")
    if what == "step" and not (a and b):
        bad.append("step needs two regions: roi (or zone) and roi2 (or zone2)")
    if what == "distance":
        if not out.get("p1"):
            bad.append("distance needs p1 ([x, y] camera px)")
        if not out.get("p2") and not a:
            bad.append("distance needs p2 (a second point) or a region (roi / zone) whose "
                       "plane p1 is measured to")
    if what == "deviation" and not out.get("golden"):
        bad.append("deviation needs golden (the id or label of a golden metric scan)")
    if what == "endface" and not out.get("face_diameter_mm"):
        bad.append("endface needs face_diameter_mm: the expected diameter (mm; the face is "
                   "searched for within +-25 % of it, its diameter is then measured)")
    for k, only in (("roi2", ("step",)), ("zone2", ("step",)), ("p1", ("distance",)),
                    ("p2", ("distance",)), ("golden", ("deviation",)),
                    ("radius_mm", ("cylinder",)), ("face_diameter_mm", ("endface",))):
        if out.get(k) is not None and what in WHATS and what not in only:
            bad.append(f"{k} is for {' / '.join(only)}, not for {what}")
    if bad:
        raise ValueError(f"measurement spec: {len(bad)} problem(s): " + "; ".join(bad))
    return Spec(**out)


def load_spec(path) -> Spec:
    """A measurement spec file (.json; the spec.json of a measure run) -> Spec."""
    p = Path(path)
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except OSError as e:
        raise ValueError(f"measurement spec {p}: {e.strerror or e}") from None
    except ValueError as e:
        raise ValueError(f"measurement spec {p.name}: not valid JSON ({e})") from None
    return spec_from(doc)


# ------------------------------------------------------------------ the document

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def empty_doc(*, run_id: str, run_folder: str, mock: bool, preset, spec: Spec,
              label: str) -> dict:
    return {
        "schema_version": MEASURE_SCHEMA,
        "operating_mode": MODE,
        "run_id": run_id,
        "run_folder": run_folder,
        "mode": "MOCK" if mock else "REAL",
        # the validity statement: MEASURED, or NOT_MEASURABLE + why + how to fix it
        "status": "NOT_MEASURABLE",
        "reason": "refused",
        "message": "not run",
        "how_to_fix": "",
        "validity": "NOT MEASURABLE: not run",
        "what": spec.what,
        "label": label,
        "spec": spec.to_dict(),
        "preset": {"name": preset.name, "geometry_id": preset.geometry_id,
                   "mount": preset.rig.mount},
        "calibration": {"calibration_id": None, "created": None, "rehomed": None},
        "rig": None,
        "scans": [],
        "repeats": spec.repeats,
        "values": [],
        "precision": None,
        "details": {},
        # a measurement decides nothing: never a verdict, never a disposition of a part
        "result": None,
        "disposition": D.NO_DISPOSITION,
        "warnings": [],
        "outputs": {"values_csv": VALUES_FILE, "overlay": None, "ply": None, "spec": SPEC_FILE,
                    "log": LOG_FILE},
        "software": R.software_info(),
        "started": _now(),
        "finished": None,
    }


def exit_code(doc: dict) -> int:
    """0 MEASURED, 5 NOT_MEASURABLE (1 aborted, 2 refused, 4 crashed); anything else 5."""
    if doc.get("status") == "MEASURED":
        return 0
    return EXIT_BY_REASON.get(doc.get("reason"), EXIT_CODES["NOT_MEASURABLE"])


def _refuse(doc: dict, code: str, message: str, how: str = "") -> None:
    msg = " ".join(str(message).split())
    doc.update(status="NOT_MEASURABLE", reason=code, message=msg,
               how_to_fix=how or HOW_TO_FIX.get(code, ""), values=[], result=None,
               disposition=D.NO_DISPOSITION)
    doc["how_to_fix"] = doc["how_to_fix"].format(preset=doc["preset"]["name"])
    doc["validity"] = f"NOT MEASURABLE ({code}): {msg}" + (
        f" | fix: {doc['how_to_fix']}" if doc["how_to_fix"] else "")


class _Stop(Exception):
    """The measurement ends NOT_MEASURABLE (the document already says why)."""


# ------------------------------------------------------------------ uncertainty

def predicted_sigma_z(ms, cal, preset, z_mm: Optional[float]) -> tuple:
    """The precision model (measure/precision.py) at distance z with this scan's own fringe
    offset / modulation -> (sigma_z mm or None, why not)."""
    from ..measure import precision
    meta = ms.meta or {}
    dec = meta.get("decode") or {}
    ax = (dec.get("per_axis") or {}).get("x") or {}
    A, B = ax.get("offset_median"), ax.get("modulation_median")
    periods = dec.get("periods_px") or meta.get("periods_px")
    steps = dec.get("steps") or meta.get("steps")
    if not z_mm:
        return None, "no distance to predict at"
    if not (_num(A) and _num(B) and B > 0 and periods and steps):
        return None, ("the scan records no fringe offset / modulation (a synthetic or imported "
                      "scan): the model has nothing to predict from")
    try:
        r = precision.predict_sigma_z(float(z_mm), cal.baseline_mm, float(cal.K_proj[0][0]),
                                      float(min(periods)), float(A), float(B), int(steps),
                                      int(meta.get("average") or preset.camera.average),
                                      float(preset.camera.gain_db))
    except (ValueError, TypeError) as e:
        return None, f"the precision model refused its inputs ({e})"
    return float(r["sigma_z_mm"]), ""


def precision_of(ms, cal, preset, z_mm: Optional[float]) -> dict:
    """What one scan resolves at distance z: the predicted depth noise, the WHOLE scan's
    measured depth noise, the larger of the two (`sigma_z_mm`) and the camera px size
    (`lateral_px_mm`). A value's own uncertainty uses the noise of the px it rests on
    (its `noise_mm`, engine/measure/values.py) in place of the whole scan's: sigma_z_of()."""
    cloud = ms.cloud
    if not z_mm:
        z = cloud.xyz[..., 2][cloud.mask]
        z_mm = float(np.median(z)) if z.size else None
    pred, why = predicted_sigma_z(ms, cal, preset, z_mm)
    meas = V.measured_noise(cloud)
    use = max([v for v in (pred, meas) if v is not None], default=None)
    basis = ("predicted_precision" if (pred is not None and use == pred) else "measured_noise")
    return {"z_mm": round(z_mm, 2) if z_mm else None,
            "sigma_z_mm": use, "method": basis if use is not None else None,
            "predicted_sigma_z_mm": pred, "predicted_note": why or None,
            "measured_noise_mm": meas,
            "lateral_px_mm": (z_mm / float(cal.K_cam[0][0])) if z_mm else None,
            "model": "Lv & Kemao phase noise (engine/measure/precision.py); measured = "
                     "second differences of z along the camera rows; noise only",
            "note": "sigma_z_mm / measured_noise_mm are the whole scan's; each value's "
                    "uncertainty uses the noise of the px it rests on (values[].noise_mm)"}


def sigma_z_of(v: dict, prec: dict) -> tuple:
    """(the depth noise a value's uncertainty rests on, mm; what it is): the prediction or
    the measured noise of the value's own px (`noise_mm`), whichever is larger; the whole
    scan's figure only when the value's px are too few to tell their own."""
    pred, own = prec.get("predicted_sigma_z_mm"), v.get("noise_mm")
    if own is None:
        return prec.get("sigma_z_mm"), prec.get("method")
    if pred is not None and pred >= own:
        return float(pred), "predicted_precision"
    return float(own), "measured_noise"


def _parts(v: dict, prec: dict) -> tuple:
    """(depth part, lateral part) of one scan's uncertainty of a value, each None when
    unknown / not involved: the depth noise x `depth_gain`, the px size x `lateral_gain`."""
    sz, _ = sigma_z_of(v, prec)
    px = prec.get("lateral_px_mm")
    if sz is not None:
        sz = sz * float(v.get("depth_gain") or 1.0)
    if px is not None:
        px = px * float(v.get("lateral_gain") or 1.0)
    return sz, px


def _depth_only(v: dict, prec: dict) -> Optional[float]:
    """The part of one scan's uncertainty that repeat scans average down (the depth noise,
    in the value's units); None for a purely lateral value."""
    sz, _ = _parts(v, prec)
    sens = v.get("sensitivity")
    if sens in ("depth", "both"):
        return sz
    if sens == "angle":
        lever = v.get("lever_mm")
        if sz is None or not lever:
            return None
        return math.degrees(math.atan2(2.0 * sz, float(lever)))
    return None


def _single(v: dict, prec: dict) -> Optional[float]:
    """One scan's uncertainty of a value, by what the value is sensitive to."""
    sz, px = _parts(v, prec)
    sens = v.get("sensitivity")
    if sens == "lateral":
        return px
    if sens == "both":
        return math.hypot(sz, px) if (sz is not None and px is not None) else None
    return _depth_only(v, prec)


def _basis_text(v: dict, prec: dict) -> str:
    sens = v.get("sensitivity")
    z = prec.get("z_mm")
    sz, how = sigma_z_of(v, prec)
    where = "of the measured px" if v.get("noise_mm") is not None else "of the whole scan"
    gain, lgain = float(v.get("depth_gain") or 1.0), float(v.get("lateral_gain") or 1.0)
    depth = (f"depth noise {sz:.4f} mm {where} at {z:g} mm ({how})"
             + (f" x {gain:.2f} (what a depth error does to this value)" if gain != 1.0 else "")
             if sz is not None else "depth noise unknown")
    px = (f"camera px size {prec['lateral_px_mm']:.4f} mm at {z:g} mm"
          + (f" x {lgain:g}" if lgain != 1.0 else "")
          if prec.get("lateral_px_mm") is not None else "px size unknown")
    return {"depth": depth, "lateral": px, "both": f"{depth} and {px}",
            "angle": f"{depth} across {float(v.get('lever_mm') or 0):.1f} mm"}.get(sens, "")


def finish_values(rows: list, prec: dict) -> list:
    """Per-scan value lists (same names, same order) -> the reported values: the mean over
    the repeats with its uncertainty (module docstring) and the resolution note."""
    out = []
    n_rep = len(rows)
    for i, first in enumerate(rows[0]):
        same = [r[i] for r in rows]
        if any(x["name"] != first["name"] for x in same):
            raise V.MeasureError("the repeat scans did not give the same values: "
                                 "they can't be combined", "invalid")
        vals = np.array([x["value"] for x in same], float)
        v = {k: first[k] for k in ("name", "label", "units", "sensitivity")}
        v["value"] = float(vals.mean())
        v["n_points"] = int(min(x["n_points"] for x in same))
        notes = []
        for x in same:
            notes += [n for n in x.get("notes") or [] if n not in notes]
        # the noisiest repeat and the weakest geometry speak for the value
        own = [x["noise_mm"] for x in same if x.get("noise_mm") is not None]
        first = {**first, "noise_mm": max(own) if own else None}
        for k in ("depth_gain", "lateral_gain"):
            if any(k in x for x in same):
                first[k] = max(float(x.get(k) or 1.0) for x in same)
        v["noise_mm"] = first["noise_mm"]
        single = _single(first, prec)
        if n_rep > 1:
            sd = float(vals.std(ddof=1))
            _, px = _parts(first, prec)
            lateral = first["sensitivity"] in ("lateral", "both") and px is not None
            # two scans that happen to agree are not a measurement of the spread: the
            # repeatability is never taken below what one scan resolves / sqrt(n)
            one = _depth_only(first, prec)
            floor = one / math.sqrt(n_rep) if one is not None else None
            floored = floor is not None and sd < floor
            rep = floor if floored else sd
            if not lateral and not rep > 0:
                raise V.MeasureError(
                    f"{first['label']}: the {n_rep} repeat scans gave the same value and the "
                    "scan gives no depth-noise estimate: no uncertainty, so no measurement "
                    "(are the repeats different scans?)", "invalid")
            if floored:
                notes.append(f"the {n_rep} repeats agree to {sd:.2g} {v['units']}, closer than "
                             f"one scan resolves ({one:.4g} {v['units']}): the uncertainty is "
                             f"that of one scan / sqrt({n_rep}), not their spread")
            v["uncertainty"] = {
                "value": math.hypot(rep, px) if lateral else rep, "units": v["units"],
                "method": ("single_scan_noise/sqrt(n)" if floored else "repeatability")
                          + ("+lateral_px_size" if lateral else ""),
                "n_repeats": n_rep, "coverage": "1 sigma", "repeatability": sd,
                "floor": floor,
                "basis": f"sample standard deviation over {n_rep} repeat scans"
                         + (f", not below one scan's noise / sqrt({n_rep})"
                            if floor is not None else "")
                         + (f", and the camera px size {px:.4f} mm (not averaged away by "
                            "repeats)" if lateral else ""),
                "single_scan": single}
            v["repeat_values"] = [float(x) for x in vals]
        else:
            if single is None:
                raise V.MeasureError(
                    f"{first['label']}: no uncertainty estimate (the scan gives neither a "
                    "predicted nor a measured depth noise): a number without one is not a "
                    "measurement - measure with repeats", "invalid")
            noise = sigma_z_of(first, prec)[1]
            v["uncertainty"] = {"value": float(single), "units": v["units"],
                                "method": {"lateral": "lateral_px_size",
                                           "both": f"{noise}+lateral_px_size"}.get(
                                               first["sensitivity"], noise),
                                "n_repeats": 1,
                                "coverage": ("the px sampling bound" if first["sensitivity"]
                                             == "lateral" else "1 sigma"),
                                "basis": _basis_text(first, prec)}
        near, why = _near_limit(v, prec)
        v["near_resolution_limit"] = near
        if near:
            notes.append(why)
        v["notes"] = notes
        for k in ("lever_mm", "trim_pct", "n_points_a", "n_points_b", "outline_rms_mm",
                  "depth_gain", "lateral_gain"):
            if k in first:
                v[k] = first[k]
        out.append(v)
    return out


def _near_limit(v: dict, prec: dict) -> tuple:
    """(near, note): a mm value within NEAR_LIMIT x the depth noise (depth values) or the
    camera px size (lateral values)."""
    if v["units"] != "mm":
        return False, ""
    a, sens = abs(v["value"]), v["sensitivity"]
    hits = []
    sz, px = sigma_z_of(v, prec)[0], prec.get("lateral_px_mm")
    if sens in ("depth", "both") and sz is not None and a < NEAR_LIMIT * sz:
        hits.append(f"the depth noise ({sz:.4f} mm)")
    if sens in ("lateral", "both") and px is not None and a < NEAR_LIMIT * px:
        hits.append(f"the camera px size ({px:.4f} mm)")
    if not hits:
        return False, ""
    return True, (f"near the resolution limit: {a:.4f} mm is within {NEAR_LIMIT:g} x "
                  + " and ".join(hits) + " - mostly the instrument, not the part")


# ------------------------------------------------------------------ one scan

def _regions(s, preset, spec: Spec, shape, warnings: list, lit_zone="?", ms=None) -> tuple:
    """(region A, region B) as camera masks (None = the whole field): the roi, cut to the
    grid zone's camera mask when the zone has one (a projector map). lit_zone: the zone the
    scan was LIT through (its meta; "?" = this run's own scan, lit through spec.zone).
    Without a projector map a zone is only the region when the scan was lit through exactly
    that zone - on any other scan (a stored full-field one) it would silently be the whole
    field, so it is refused. A painted zone set ("zoneset:<set>[:ids]", engine/zones) is cut
    from the scan `ms` itself: the camera px whose decoded (col, row) is in its zones."""
    def same(a, b) -> bool:
        return "".join(str(a or "").split()).upper() == "".join(str(b or "").split()).upper()

    def zone_mask(text: str, need: bool) -> Optional[np.ndarray]:
        from ..capture import zones as zones_mod
        from ..zones import use as zone_use
        if zone_use.is_spec(text):
            return zone_use.camera_region(s, preset, text, ms, shape, warnings, lit_zone, need)
        _, cam, _ = zones_mod.resolve(s, preset, text)
        if cam is None:
            if need:
                raise ValueError(f"zone {text!r} has no camera mask: capture a projector map "
                                 "first (`scan map`), or give the region as an roi")
            if lit_zone != "?" and not same(lit_zone, text):
                raise ValueError(
                    f"zone {text!r} can't cut this stored scan: there is no projector map at "
                    f"this geometry, and the scan was lit through "
                    f"{lit_zone or 'the whole field'}, not through that zone - the region "
                    "would be the whole scan. Capture a projector map (`scan map`), or give "
                    "the region as an roi")
            warnings.append(f"zone {text}: no projector map, so the region is whatever the "
                            "lit zone decoded (`scan map` cuts it exactly)")
            return None
        cam = np.asarray(cam, bool)
        if cam.shape != tuple(shape):
            raise ValueError(f"zone {text!r}: its camera mask is {cam.shape}, the scan "
                             f"{tuple(shape)}")
        return cam

    def region(roi, zone, need_zone: bool) -> Optional[np.ndarray]:
        m = V.roi_mask(shape, roi) if roi is not None else None
        if zone:
            z = zone_mask(zone, need_zone and roi is None)
            if z is not None:
                m = z if m is None else (m & z)
        return m

    return region(spec.roi, spec.zone, bool(spec.zone2)), region(spec.roi2, spec.zone2, True)


def measure_scan(s, preset, spec: Spec, ms, golden=None, warnings: Optional[list] = None,
                 stored: bool = False) -> dict:
    """The values of ONE metric scan -> {"values", "z_mm", "mask", "details", ...}
    (engine/measure/values.py); MeasureError when this scan cannot give them. stored: the
    scan was not captured by this run (a zone then only cuts it by a projector map, or when
    the scan itself was lit through that zone)."""
    from ..measure import metric
    from ..reconstruct import metric_scan
    warnings = [] if warnings is None else warnings
    cloud = ms.cloud
    a, b = _regions(s, preset, spec, cloud.mask.shape, warnings,
                    (ms.meta or {}).get("zone") if stored else "?", ms)
    if spec.what == "flatness":
        return V.flatness(cloud, a)
    if spec.what == "step":
        return V.step(cloud, a, b)
    if spec.what == "distance":
        has_region = spec.roi is not None or bool(spec.zone)
        return V.distance(cloud, spec.p1, spec.p2, a if has_region else None)
    if spec.what == "cylinder":
        return V.cylinder(cloud, a, spec.radius_mm, spec.axis)
    if spec.what == "dims":
        return V.dims(cloud, a, spec.min_height_mm)
    if spec.what == "endface":
        from ..measure.station import verbs as station
        from ..measure.station.geometry import StationError
        try:
            m, _ = station.station_measure(
                s, ms, None, None, {"n_cells": spec.n_faces,
                                    "face_diameter_mm": spec.face_diameter_mm}, write=False)
        except StationError as e:
            # a face of known size that reads the wrong size: its own reason (the
            # calibration or the lens, not the part's placement)
            code = "size_mismatch" if "size mismatch" in str(e) else "incomplete"
            raise V.MeasureError(f"end face: {e}{_calibration_hint(s, ms, str(e))}",
                                 code) from None
        warnings += [w for w in (str(x) for x in m.warnings if "tube" not in str(x).lower())
                     if w not in warnings]
        out = V.endfaces(m.cells)
        yy, xx = np.ogrid[:cloud.mask.shape[0], :cloud.mask.shape[1]]
        for i, (c, r) in enumerate(out["circles"]):     # each face: the noise inside its rim
            disc = ((xx - c[0]) ** 2 + (yy - c[1]) ** 2 <= float(r) ** 2) if (c and r) else None
            V.with_noise(out["values"][2 * i:2 * i + 2],
                         V.region_noise(cloud, disc & cloud.mask) if disc is not None else None)
        return out
    rep, dev = metric_scan.compare_metric(s, ms, golden, max_dist_mm=COMPARE_MAX_DIST_MM)
    # the part compare_metric placed on the golden (its largest foreground blob; the same
    # seeded plane fit): its px with no deviation were too far from the golden to compare
    out = V.deviation(dev, rep, a, part=metric.foreground(cloud)[0],
                      max_dist_mm=COMPARE_MAX_DIST_MM)
    V.with_noise(out["values"], V.region_noise(cloud, out["mask"]))
    out["deviation_map"] = dev
    z = cloud.xyz[..., 2][out["mask"]]
    out["z_mm"] = float(np.median(z)) if z.size else None
    return out


def _calibration_hint(s, ms, why: str) -> str:
    """End faces set aside because too little of the disc was measured, on a scan whose
    camera and projector rays miss each other far more than its calibration did
    (metric_scan.consistency_of): the likelier cause is the calibration, not the part -> the
    hint for the NOT_MEASURABLE reason; "" otherwise."""
    from ..reconstruct import metric_scan
    if "of the disc measured" not in why and "too few measured points" not in why:
        return ""
    chk = metric_scan.consistency_of(s, ms)
    if chk is None or chk.get("fits"):
        return ""
    seen = (f"{chk['median_mm']:.2f} mm (median)" if chk.get("median_mm") is not None else
            "an amount that could not be computed")
    return (f" - the scan's camera and projector rays miss each other by {seen}, limit "
            f"{chk['limit_mm']:.2f} mm: {metric_scan.CALIBRATION_MAY_NOT_FIT}")


def _live_scan(s, preset, spec: Spec, label: str):
    from .. import api
    sc = spec.scan
    zone = None if spec.zone2 else spec.zone        # two zones: the whole field is lit
    if sc["kind"] == "fast":
        return api.scan_fast(zone, preset, label, sc["steps"] or 4, sc["periods"],
                             sc["average"] if sc["average"] is not None else 2, axis="xy")
    p = preset
    if sc["average"] is not None:                   # this run only: the preset is untouched
        from dataclasses import replace
        p = replace(preset, camera=replace(preset.camera, average=int(sc["average"])))
    return api.scan_metric(p, label, steps=sc["steps"], periods=sc["periods"], zone=zone)


# ------------------------------------------------------------------ outputs

def _base_image(ms) -> np.ndarray:
    """The camera view the overlay is drawn on (uint8, camera resolution): the scan's flood
    when it kept one (a fast scan), else its depth map."""
    cloud = ms.cloud
    img = None
    f = Path(ms.path) / "flood.npz"
    if f.is_file():
        try:
            with np.load(f) as z:
                img = np.asarray(z["white"], np.float64)
        except (OSError, ValueError, KeyError):
            img = None
    if img is None or img.shape != cloud.mask.shape:
        z = cloud.xyz[..., 2]
        img = np.where(cloud.mask, -z, np.nan)
    ok = np.isfinite(img)
    if not ok.any():
        return np.zeros(cloud.mask.shape, np.uint8)
    lo, hi = np.percentile(img[ok], [1, 99])
    g = np.clip((np.nan_to_num(img, nan=lo) - lo) / max(hi - lo, 1e-9), 0, 1)
    g8 = (40 + 200 * g).astype(np.uint8)
    g8[~ok] = 0
    return g8


def _px(cal, X) -> tuple:
    from ..capture.mock_stereo import distort_project
    p = distort_project(np.asarray(X, np.float64).reshape(1, 3), np.asarray(cal.K_cam, float),
                        cal.dist_cam)[0]
    return int(round(float(p[0]))), int(round(float(p[1])))


def overlay(ms, cal, spec: Spec, one: dict, title: str) -> np.ndarray:
    """BGR uint8, camera resolution: the camera view, the points used tinted, the regions,
    the picked points, the bounding box / cylinder axis, the deviation map."""
    import cv2
    img = cv2.cvtColor(_base_image(ms), cv2.COLOR_GRAY2BGR)
    m = one.get("mask")
    dev = one.get("deviation_map")
    if dev is not None:
        ok = np.isfinite(dev)
        lim = max(float(np.percentile(np.abs(dev[ok]), 99)) if ok.any() else 1.0, 1e-6)
        d = np.clip(np.nan_to_num(dev) / lim, -1, 1)
        col = np.zeros(img.shape, np.float64)
        col[..., 2] = 255 * np.clip(d, 0, 1)            # above the golden: red
        col[..., 0] = 255 * np.clip(-d, 0, 1)           # below: blue
        col[..., 1] = 255 * (1 - np.abs(d)) * 0.6
        img[ok] = (0.45 * img[ok] + 0.55 * col[ok]).astype(np.uint8)
    elif m is not None and m.any():
        tint = np.array([90, 200, 90], np.float64)
        img[m] = (0.7 * img[m] + 0.3 * tint).astype(np.uint8)
    H, W = img.shape[:2]
    for tag, roi, colour in (("A", spec.roi, (255, 255, 0)), ("B", spec.roi2, (255, 0, 255))):
        if roi is not None:
            x0, y0, x1, y1 = roi
            cv2.rectangle(img, (x0, y0), (min(x1, W) - 1, min(y1, H) - 1), colour, 2)
            cv2.putText(img, tag, (x0 + 6, y0 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2)
    for tag, p in (("P1", spec.p1), ("P2", spec.p2)):
        if p is not None:
            x, y = int(round(p[0])), int(round(p[1]))
            cv2.drawMarker(img, (x, y), (0, 255, 255), cv2.MARKER_CROSS, 28, 2)
            cv2.putText(img, tag, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    if spec.p1 is not None and spec.p2 is not None:
        cv2.line(img, tuple(int(round(v)) for v in spec.p1), tuple(int(round(v)) for v in spec.p2),
                 (0, 255, 255), 1)
    try:
        if one.get("box"):
            pts = np.array([_px(cal, X) for X in one["box"]], np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], True, (0, 255, 255), 2)
        if one.get("axis_line"):
            a, b = (_px(cal, X) for X in one["axis_line"])
            cv2.line(img, a, b, (0, 255, 255), 2)
        for c, r in one.get("circles") or []:
            if c and r:
                cv2.circle(img, (int(round(c[0])), int(round(c[1]))), int(round(r)),
                           (0, 255, 255), 2)
    except (ValueError, OverflowError, FloatingPointError):
        pass                                # a drawing never changes a number
    cv2.rectangle(img, (0, 0), (W - 1, 34), (0, 0, 0), -1)
    cv2.putText(img, title[:110], (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    return img


def _write_png(path: Path, img: np.ndarray) -> Optional[str]:
    import cv2
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        return None
    with open(path, "xb") as f:             # a run folder is new: never overwritten
        f.write(buf.tobytes())
    return path.name


def write_values_csv(path: Path, values: list) -> Path:
    """values.csv: one row per value (the header alone when nothing was measured)."""
    with open(path, "x", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_FIELDS)
        for v in values:
            u = v.get("uncertainty") or {}
            w.writerow([v["name"], v["label"], f"{v['value']:.6g}", v["units"], v["n_points"],
                        "" if u.get("value") is None else f"{u['value']:.4g}",
                        u.get("units", ""), u.get("method", ""), u.get("n_repeats", ""),
                        "yes" if v.get("near_resolution_limit") else "no",
                        " | ".join(v.get("notes") or [])])
    return path


def _write_doc(run_dir: Path, doc: dict) -> Path:
    doc["finished"] = _now()
    # fail closed, whatever happened above: a measurement never carries a verdict
    doc["result"], doc["disposition"] = None, D.NO_DISPOSITION
    if doc.get("status") not in STATUSES:
        doc["status"] = "NOT_MEASURABLE"
    if doc["status"] != "MEASURED":
        doc["values"] = []
    out = run_dir / RESULT_FILE
    with open(out, "x", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, default=str)
        f.write("\n")
    eventlog.record(run_dir)                # its row in the inspection event log (never raises)
    return out


def _round(v):
    return round(float(v), 6) if isinstance(v, (int, float)) and not isinstance(v, bool) else v


def _clean(o):
    """numpy scalars / arrays -> plain JSON types; NaN / inf -> null."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, np.generic):
        return _clean(o.item())
    if isinstance(o, float):
        return _round(o) if math.isfinite(o) else None
    return o


# ------------------------------------------------------------------ the run

def run(s, preset, spec: Spec, *, runs_dir, scans: Optional[list] = None,
        mock_place: Optional[Callable] = None, label: Optional[str] = None) -> dict:
    """One measurement in a NEW run folder -> {run_id, run_dir, result (the document),
    exit_code}. scans: stored metric scans (ids / labels / handles) measured instead of
    capturing (their count is the number of repeats; nothing touches the rig). mock_place
    (MOCK only): the simulated human places the part, once, before the first scan."""
    from ..calibrate import stereo
    from ..recipes import runner
    from ..reconstruct import metric_scan
    from ..reconstruct.triangulate import write_ply
    label = label or spec.name or f"measure_{spec.what}"
    run_id, run_dir = R.new_run(runs_dir, f"measure_{spec.what}")
    doc = empty_doc(run_id=run_id, run_folder=str(run_dir), mock=s.mock, preset=preset,
                    spec=spec, label=label)
    log = R.RunLog(Path(run_dir) / LOG_FILE, run_id)
    used, one, cal = [], None, None
    try:
        log.event("run_start", operating_mode=MODE, mode=doc["mode"], what=spec.what,
                  preset=preset.name, mount=preset.rig.mount, repeats=spec.repeats,
                  stored_scans=bool(scans))
        fp = config.geometry_fingerprint(preset)
        cal = stereo.latest(s.store, preset.geometry_id, fp)
        if cal is None:
            _refuse(doc, "uncalibrated",
                    f"no stereo calibration for geometry {preset.geometry_id} at preset "
                    f"{preset.name}'s rig settings: no mm (rule 6), and a camera px is never "
                    "reported as one")
            raise _Stop
        doc["calibration"] = {"calibration_id": cal.id, "created": cal.created,
                              "rehomed": (cal.rehomed or {}).get("note")}
        stored = []
        for key in scans or []:
            try:
                stored.append(runner._find_metric(s, key, preset.geometry_id))
            except LookupError as e:
                raise ValueError(f"scan {key!r}: {e}") from None
        ids = [m.id for m in stored]
        twice = sorted({i for i in ids if ids.count(i) > 1})
        if twice:                           # the same scan twice has no spread: +- 0 is a lie
            raise ValueError(f"scan {twice[0]} is named more than once: repeats are DIFFERENT "
                             "scans of the part (one scan measured twice gives the same "
                             "number twice, and a repeatability of 0)")
        pdir = getattr(s, "presets_dir", None)
        gate = rigstate.gate(s.store, preset, strict=True, presets_dir=pdir)
        for ms in stored:                   # a stored scan: made while the rig could be trusted
            if gate["ok"]:
                gate = rigstate.gate(s.store, preset, strict=True, scan=ms, presets_dir=pdir)
        doc["rig"] = gate["state"]
        log.event("rig", ok=gate["ok"], mount=gate["mount"], code=gate["code"],
                  message=gate["message"])
        if not gate["ok"]:
            _refuse(doc, gate["code"], gate["message"], gate["how_to_fix"])
            raise _Stop
        golden = None
        if spec.what == "deviation":
            try:
                golden = runner._find_metric(s, spec.golden, preset.geometry_id)
                metric_scan.check_cal(s, golden, "the golden scan")
                runner._same_setup(golden, preset, fp, cal, "the golden scan")
            except (LookupError, rules.RuleViolation) as e:
                _refuse(doc, "golden", f"golden scan {spec.golden!r}: {e}")
                raise _Stop from None
        n = len(stored) if stored else spec.repeats
        doc["repeats"] = n
        rows = []
        for i in range(n):
            s.check_cancel()
            if stored:
                ms = stored[i]
                rules.require_same_geometry(ms.geometry_id, preset.geometry_id, "measure")
                metric_scan.check_cal(s, ms, "measure")             # rule 6 at use time
                runner._same_setup(ms, preset, fp, cal, "measure")
            else:
                if mock_place is not None and i == 0:
                    mock_place()
                ms = _live_scan(s, preset, spec, label if n == 1 else f"{label}_{i + 1}")
                doc["warnings"] += [str(w) for w in (ms.meta or {}).get("warnings") or []]
            if golden is not None and ms.id == golden.id:
                _refuse(doc, "golden", f"the golden {golden.id} is the scan being measured: a "
                                       "part is never compared with itself")
                raise _Stop
            used.append(ms)
            doc["scans"].append(ms.id)
            log.event("scan", scan_id=ms.id, kind="stored" if stored else spec.scan["kind"],
                      calibration_id=ms.calibration_id, n_points=(ms.meta or {}).get("n_points"))
            s.check_cancel()
            got = measure_scan(s, preset, spec, ms, golden, doc["warnings"],
                               stored=bool(stored))
            rows.append(got["values"])
            if one is None:
                one = got
        prec = precision_of(used[0], cal, preset, one.get("z_mm"))
        doc["precision"] = prec
        doc["values"] = finish_values(rows, prec)
        doc["details"] = one.get("details") or {}
        if n > 1:
            doc["details"]["repeat_scans"] = [m.id for m in used]
        if prec.get("predicted_sigma_z_mm") and prec.get("measured_noise_mm") and \
                prec["measured_noise_mm"] > 2.0 * prec["predicted_sigma_z_mm"]:
            doc["warnings"].append(
                f"the scan's measured depth noise ({prec['measured_noise_mm']:.4f} mm) is more "
                f"than twice the prediction ({prec['predicted_sigma_z_mm']:.4f} mm): something "
                "systematic (timing / flicker, interreflection, the calibration)")
        if stored:
            doc["warnings"].append("offline: measured on stored scan(s) "
                                   + ", ".join(m.id for m in used) + " (nothing was captured)")
        st = gate["state"]
        rig = (f"mobile rig verified by {st.get('verified_by')}" if st["mount"] == "mobile"
               else "fixed rig")
        doc.update(status="MEASURED", reason=None, how_to_fix="",
                   message=f"{len(doc['values'])} value(s) from {n} scan(s)")
        doc["validity"] = (f"MEASURED: mm from stereo calibration {cal.id}"
                           + (f" ({doc['calibration']['rehomed']})" if doc["calibration"]["rehomed"]
                              else "") + f", {rig}, {n} scan(s); uncertainty = "
                           + (f"repeatability over the repeats, never below one scan's noise / "
                              f"sqrt({n})" if n > 1 else
                              f"depth noise of the measured px (or the prediction) / px size "
                              f"at {prec['z_mm']} mm")
                           + " (noise only, 1 sigma)")
        log.event("measure", what=spec.what, n_values=len(doc["values"]),
                  values={v["name"]: _round(v["value"]) for v in doc["values"]})
    except _Stop:
        log.event("not_measurable", level="warning", reason=doc["reason"], message=doc["message"])
    except V.MeasureError as e:
        _refuse(doc, e.code, str(e))
        log.event("not_measurable", level="warning", reason=e.code, message=str(e))
    except rules.Cancelled as e:
        _refuse(doc, "aborted", f"aborted: {e}")
        log.event("aborted", level="warning", message=str(e))
    except rules.GeometryMismatch as e:     # another calibration / other rig settings
        code = D.classify(str(e), "stale_calibration")
        _refuse(doc, code if code in ("uncalibrated", "stale_calibration") else
                "stale_calibration", str(e))
        log.event("not_measurable", level="warning", reason=doc["reason"], message=str(e))
    except Exception as e:                  # noqa: BLE001 - refused / crashed: never a number
        refused = isinstance(e, (rules.RuleViolation, config.ConfigError, ValueError,
                                 LookupError))
        _refuse(doc, "refused" if refused else "error", f"{type(e).__name__}: {e}")
        log.event("error", level="error", type=type(e).__name__, message=str(e))
    code = exit_code(doc)
    try:
        if doc["status"] == "MEASURED":
            try:
                title = f"MEASURED {spec.what}  {run_id}"
                doc["outputs"]["overlay"] = _write_png(Path(run_dir) / OVERLAY_FILE,
                                                       overlay(used[0], cal, spec, one, title))
            except Exception as e:          # noqa: BLE001 - a picture never changes a number
                doc["warnings"].append(f"overlay image: {type(e).__name__}: {e}")
            try:
                write_ply(Path(run_dir) / PLY_FILE, used[0].cloud)
                doc["outputs"]["ply"] = PLY_FILE
            except Exception as e:          # noqa: BLE001
                doc["warnings"].append(f"scan.ply: {type(e).__name__}: {e}")
        doc = _clean(doc)
        with open(Path(run_dir) / SPEC_FILE, "x", encoding="utf-8") as f:
            json.dump(spec.to_dict(), f, indent=2)
            f.write("\n")
        write_values_csv(Path(run_dir) / VALUES_FILE, doc["values"])
        out = _write_doc(Path(run_dir), doc)
        log.event("run_end", status=doc["status"], reason=doc["reason"], exit_code=code,
                  result_json=out.name, disposition=doc["disposition"])
    finally:
        log.close()
    return {"run_id": run_id, "run_dir": run_dir, "result": doc, "exit_code": code}


# ------------------------------------------------------------------ spec -> check

def _limits(name: str, tol) -> tuple:
    """A tolerance: [lo, hi] (either may be null), {"lo", "hi"}, or a number (an upper
    limit) -> (lo, hi)."""
    if _num(tol):
        lo, hi = None, float(tol)
    elif isinstance(tol, dict):
        lo, hi = tol.get("lo"), tol.get("hi")
    elif isinstance(tol, (list, tuple)) and len(tol) == 2:
        lo, hi = tol
    else:
        raise ValueError(f"tolerance {name}: give [lo, hi], {{lo, hi}} or one number (an "
                         f"upper limit), got {tol!r}")
    for k, v in (("lo", lo), ("hi", hi)):
        if v is not None and not _num(v):
            raise ValueError(f"tolerance {name}: {k} must be a finite number or null, got {v!r}")
    if lo is None and hi is None:
        raise ValueError(f"tolerance {name}: give lo, hi or both")
    if lo is not None and hi is not None and lo > hi:
        raise ValueError(f"tolerance {name}: lo {lo} > hi {hi}")
    return (None if lo is None else float(lo)), (None if hi is None else float(hi))


def to_check(spec, tolerances: dict, name: Optional[str] = None,
             guard_band_pct: float = 0.0) -> dict:
    """Promote a measurement spec to a CHECK by adding tolerances: the same regions, judged
    by the engine's existing checks (nothing new decides a part).

        flatness    -> a recipe: zone = the roi, `flatness` check (tolerances
                       flatness_p2_p98 -> tol_mm, optional flatness_rms -> rms_tol_mm)
        step        -> a recipe: region A = the datum zone (the reference plane), `height`
                       check on region B (tolerance step_height [lo, hi] -> band_mm)
        deviation   -> a single-job station config: metric golden compare (tolerance
                       deviation_max, an upper limit on |deviation| -> threshold_mm)

    -> {"kind": "recipe" | "station_config", "document", "run_with"}. Run a recipe with
    `scan quick-check <preset> --check recipe --recipe <file>` (or name it in a profile of a
    station configuration); a station config with `scan run <file>`. The other measurements
    (distance, cylinder, dims) have no one-to-one check: write a recipe (docs/CHECKS.md:
    `height`, `position`, `edge`, or a check plugin). ValueError says what is missing."""
    from . import quick
    sp = spec_from(spec)
    if not isinstance(tolerances, dict) or not tolerances:
        raise ValueError("give at least one tolerance: {value name: [lo, hi]} (the names are "
                         "the measurement's value names, e.g. flatness_p2_p98, step_height)")
    gb = D.check_pct(guard_band_pct)
    if not sp.preset:
        raise ValueError("the spec names no preset: a check is bound to its triangulation "
                         "preset (the spec.json of a measure run names it)")
    name = name or sp.name or f"{sp.what}_check"
    tol = {str(k).strip().lower(): _limits(str(k), v) for k, v in tolerances.items()}

    def only(*names):
        extra = sorted(set(tol) - set(names))
        if extra:
            raise ValueError(f"{sp.what}: no check for tolerance(s) {', '.join(extra)} (it "
                             f"takes {', '.join(names)})")

    def rect(roi, what):
        if roi is None:
            raise ValueError(f"{sp.what}: {what} must be an roi ([x0, y0, x1, y1] camera px): "
                             "a check needs its zone")
        x0, y0, x1, y1 = roi
        return [x0, y0, x1 - 1, y1 - 1]         # a recipe rect is inclusive

    if sp.what == "deviation":
        only("deviation_max")
        lo, hi = tol.get("deviation_max", (None, None))
        if hi is None or hi <= 0:
            raise ValueError("deviation: give deviation_max, the largest |deviation| from the "
                             "golden allowed (mm, an upper limit > 0)")
        doc = quick.build_config("golden", sp.preset, threshold_mm=hi, golden=sp.golden,
                                 roi=sp.roi, zone=sp.zone, guard_band_pct=gb, station_id=name)
        return {"kind": "station_config", "document": doc,
                "run_with": "scan run <file> (or scan quick-check --check golden)"}
    recipe = {"schema_version": 1, "name": name,
              "description": f"promoted from the measurement spec {sp.name or sp.what} "
                             "(engine/modes/measure.to_check)",
              "preset": sp.preset,
              "scan": {"kind": sp.scan["kind"], "axis": "xy", "zone": None,
                       "average": sp.scan["average"], "steps": sp.scan["steps"],
                       "periods": sp.scan["periods"]},
              "feedback": {"enabled": True}, "pull": {"guard_band_pct": gb}}
    if sp.what == "flatness":
        only("flatness_p2_p98", "flatness_rms")
        _, hi = tol.get("flatness_p2_p98", (None, None))
        if hi is None or hi <= 0:
            raise ValueError("flatness: give flatness_p2_p98, the peak-to-valley limit (mm, an "
                             "upper limit > 0); flatness_rms is optional")
        params = {"tol_mm": hi, "trim_pct": V.TRIM_PCT}
        if "flatness_rms" in tol:
            if tol["flatness_rms"][1] is None or tol["flatness_rms"][1] <= 0:
                raise ValueError("flatness: flatness_rms is an upper limit > 0 (mm)")
            params["rms_tol_mm"] = tol["flatness_rms"][1]
        recipe["zones"] = [{"id": "a", "label": "region A", "shape": "rect",
                            "units": "camera_px", "coords": rect(sp.roi, "the region"),
                            "checks": [{"type": "flatness", "params": params}]}]
    elif sp.what == "step":
        only("step_height")
        lo, hi = tol.get("step_height", (None, None))
        if lo is None or hi is None:
            raise ValueError("step: give step_height as a band [lo, hi] (mm, B above A)")
        recipe["part"] = {"frame": {"reference": {"zone": "a"}}}
        recipe["zones"] = [
            {"id": "a", "label": "region A (the reference plane)", "shape": "rect",
             "units": "camera_px", "coords": rect(sp.roi, "region A"), "datum": True,
             "checks": []},
            {"id": "b", "label": "region B", "shape": "rect", "units": "camera_px",
             "coords": rect(sp.roi2, "region B"),
             "checks": [{"type": "height", "params": {"ref": "plane", "stat": "median",
                                                      "band_mm": [lo, hi]}}]}]
    else:
        raise ValueError(f"{sp.what}: no one-to-one check to promote to - write a recipe "
                         "(docs/CHECKS.md: height, position, edge, or a check plugin) with the "
                         "regions of this spec")
    return {"kind": "recipe", "document": recipe,
            "run_with": "scan quick-check <preset> --check recipe --recipe <file>"}

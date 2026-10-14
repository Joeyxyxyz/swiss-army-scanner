"""
Recipe runner: preset -> scan -> zones -> checks -> verdict -> run folder -> projected feedback.

    execute()     scan per the recipe (or take a stored scan), resolve the part frame and the
                  zones, run every check, aggregate, write the camera overlay and the DMD
                  feedback image into the run folder, fill the result document (never writes
                  result.json itself)
    finish()      result.json (once), then the projected feedback (pumped hold, ABORT clears
                  it), run_end in the log
    run_recipe()  the whole thing in a NEW run folder (api.recipe_run)

Verdict: any check FAIL -> FAIL (a trusted measurement found a bad part); else every check
PASS -> PASS; anything else -> INVALID (never a pass: no stereo calibration at the preset's
geometry, a scan made under another calibration or other rig settings, a zone with too few
points, no golden scan, ...). A check that raises, or answers anything but a CheckResult
with PASS / FAIL / INVALID, is INVALID with its error (a bug is never a verdict).

Disposition (engine/disposition.py, the recipe's `pull` section), on top of the verdict:
CONTINUE (PASS, no doubt), REJECT (a definite measured FAIL, even when another zone is
INVALID), PULL for quality inspection (any doubt: INVALID of any kind, a crashed check, an
unreadable code, ABORTED, the rig check MOVED / UNKNOWN, a PASS value in the guard band).
It decides the exit code (0 / 1 / 5), the result code and what the projector paints.

Golden (the good part position / edge compare with): only a GOLDEN entry, made by
scan_golden (`scan recipe-golden`, api.recipe_golden) and recorded in
<library>/recipe_golden/<geometry_id>/, or a metric scan named by its exact id. golden.scan
'latest' = the newest golden recorded for THIS recipe at this geometry; a label = the
newest golden entry with that label. Never the scan being checked, never a plain metric scan
picked by label or 'latest' (an earlier run's own test scan carries the same label), never
one made at other geometry-bound settings or with another stereo calibration.

result.json = the run schema (engine/runs.py) + `recipe` (name, preset, frame, pull config,
result code), `zones` [{id, verdict, checks: [{id, type, verdict, values (units in the
names), key, limit, tolerances, pull, message, fault_points}]}], `feedback` (what the DMD
showed: zones drawn / refused with the camera -> DMD fit residuals, crosses, text, image
file) and `disposition` / `pull_reasons`.
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .. import badparts, config, eventlog, rigstate, rules
from .. import disposition as D
from .. import runs as R
from . import feedback as FB
from .checks.base import VERDICTS, CheckResult
from .context import CheckContext, GoldenView
from .geometry import ScanView, ZoneError, frame_for, resolve_zone

CALIBRATE_STEREO_HOWTO = ("calibrate it first, with the preset this run uses: `scan "
                          "--preset {preset} charuco` (print at 100 %, caliper a square), `scan "
                          "--preset {preset} calib-pose` per board pose (12-20, tilted +-30 "
                          "deg), then `scan --preset {preset} calibrate-stereo --square MM` "
                          "(mock: `scan --mock --preset {preset} calibrate-stereo --mock-poses "
                          "12`)")
FEEDBACK_IMAGE = "feedback_dmd.png"
FEEDBACK_PHOTO = "feedback_photo.png"
OVERLAY_IMAGE = "overlay.png"
ORDER = {"PASS": 0, "INVALID": 1, "FAIL": 2}
GOLDEN_KIND = "recipe_golden"           # <library>/recipe_golden/<geometry_id>/<ts>_<recipe>/


def aggregate(verdicts) -> str:
    """FAIL if any is exactly FAIL; PASS only when every one is exactly PASS; anything else
    (INVALID, nothing, an unknown or misspelt verdict) is INVALID: fail closed."""
    v = list(verdicts)
    if "FAIL" in v:
        return "FAIL"
    if not v or any(x != "PASS" for x in v):
        return "INVALID"
    return "PASS"


def _find_metric(s, key: str, geometry_id: str):
    """A metric scan of this geometry by id, label (newest wins) or 'latest'."""
    from ..reconstruct import metric_scan
    from ..results import MetricScan
    if isinstance(key, MetricScan):
        return key
    items = metric_scan.list_metric(s.store, geometry_id)
    if not items:
        raise LookupError(f"no metric scans at geometry {geometry_id}")
    k = str(key).strip()
    if k.lower() in ("latest", "last"):
        return items[-1]
    for test in (lambda x: x.id == k, lambda x: x.label == k):
        hits = [x for x in items if test(x)]
        if hits:
            return hits[-1]
    hits = [x for x in items if k in x.id]
    if len(hits) == 1:
        return hits[0]
    raise LookupError(f"no metric scan {k!r} at geometry {geometry_id}"
                      + (f" ({len(hits)} ids contain it: give more of the id)" if hits else ""))


# ------------------------------------------------------------------ golden scans

def record_golden(s, recipe, preset, ms) -> Path:
    """Record the metric scan `ms` (a GOOD part) as a golden of `recipe`: an append-only
    entry recipe_golden/<geometry_id>/<ts>_<recipe>/meta.json naming the scan, its label,
    rig fingerprint and stereo calibration. Only recorded goldens (or an exact scan id) can
    be a recipe's golden. A scan of a part judged REJECT or PULL is refused (engine/
    badparts.py: references are made from good parts only). -> the entry folder."""
    from ..library.store import json_safe
    rules.require_same_geometry(ms.geometry_id, preset.geometry_id, "recipe golden")
    badparts.require_good(s.store, f"recipe golden of {recipe.name}", [ms], sweep=True)
    d = s.store.new_dir(GOLDEN_KIND, ms.geometry_id, label=recipe.name)
    meta = {"kind": GOLDEN_KIND, "recipe": recipe.name, "scan_id": ms.id, "label": ms.label,
            "geometry_id": ms.geometry_id,
            "geometry_fp": (ms.meta or {}).get("geometry_fp"),
            "calibration_id": ms.calibration_id, "preset": preset.name,
            "created": datetime.now().isoformat(timespec="seconds")}
    with s.store.filling(d):
        s.store.write_json(d / "meta.json", json_safe(meta))
    return d


def scan_golden(s, recipe, preset, label: str):
    """Scan a GOOD part with the recipe's scan settings and record it as the recipe's golden
    (record_golden). -> MetricScan"""
    ms = _scan(s, recipe, preset, label, None, {"capture": {"average": None}}, False)
    record_golden(s, recipe, preset, ms)
    return ms


def list_goldens(store, geometry_id: str) -> list:
    """The golden entries at this geometry, oldest first: [meta (+ "entry": folder)]."""
    from ..library.store import read_meta
    out = []
    for d in store.entries(GOLDEN_KIND, geometry_id):
        m = read_meta(d)
        if m is None or m.get("kind") != GOLDEN_KIND or not m.get("scan_id"):
            continue                        # unreadable / half-written: never a golden
        out.append({**m, "entry": d})
    return out


def find_golden(s, recipe, geometry_id: str, key=None, exclude=()):
    """The recipe's golden scan -> MetricScan, or LookupError saying why.

    key (default golden.scan): 'latest' / 'last' = the newest golden recorded for this
    recipe at this geometry; else the newest golden entry whose label (or scan id) is key;
    else the metric scan whose id is exactly key. A golden entry whose scan was trashed is
    skipped. Refused when the scan is one of `exclude` (the scan being checked: a part is
    never compared with itself), and when it is a part judged REJECT or PULL (BadPartReference:
    nothing is ever compared with a bad part, engine/badparts.py)."""
    from ..reconstruct import metric_scan
    key = (recipe.golden or {}).get("scan") if key is None else key
    k = str(key or "").strip()
    if not k:
        raise LookupError("the recipe names no golden scan (golden.scan)")
    scans = {x.id: x for x in metric_scan.list_metric(s.store, geometry_id)}
    gold = [g for g in list_goldens(s.store, geometry_id) if g["scan_id"] in scans]
    if k.lower() in ("latest", "last"):
        hits = [g for g in gold if g.get("recipe") == recipe.name]
        if not hits:
            raise LookupError(f"no golden recorded for recipe {recipe.name} at geometry "
                              f"{geometry_id}")
        ms = scans[hits[-1]["scan_id"]]
    else:
        hits = [g for g in gold if g.get("label") == k or g["scan_id"] == k]
        if hits:
            ms = scans[hits[-1]["scan_id"]]
        elif k in scans:
            ms = scans[k]                   # an explicit scan id: the integrator's choice
        else:
            plain = [x for x in scans.values() if x.label == k]
            raise LookupError(
                f"no golden {k!r} at geometry {geometry_id}"
                + (f" ({len(plain)} metric scan(s) carry that label, but none was recorded as "
                   "a golden: a test scan is never a golden; name one by its full id)"
                   if plain else ""))
    if ms.id in set(exclude):
        raise LookupError(f"the golden {ms.id} is the scan being checked: a part is never "
                          "compared with itself")
    badparts.require_good(s.store, f"the golden of recipe {recipe.name}", [ms],
                          how=badparts.HOW_USE)
    return ms


def _same_setup(ms, preset, fp, cal, what: str) -> None:
    """A stored metric scan must come from THIS rig setup: the preset's geometry-bound
    settings (fingerprint) and the run's stereo calibration (GeometryMismatch otherwise)."""
    diffs = rules.fingerprint_diffs((ms.meta or {}).get("geometry_fp"), fp)
    if diffs:
        raise rules.GeometryMismatch(
            f"{what}: scan {ms.id} was made at other rig settings than preset {preset.name} "
            f"({'; '.join(diffs)}): re-scan it at these settings")
    if ms.calibration_id != cal.id:
        raise rules.GeometryMismatch(
            f"{what}: scan {ms.id} was triangulated with stereo calibration "
            f"{ms.calibration_id}, this run uses {cal.id}: re-scan it")


def _scan(s, recipe, preset, label: str, log, res, pattern_mode_ack: bool):
    from .. import api
    sc = recipe.scan
    kind = sc["kind"]
    if kind == "metric":
        ms = api.scan_metric(preset, label, steps=sc["steps"], periods=sc["periods"],
                             zone=sc["zone"])
        pats = "scan_metric (both axes)"
    elif kind == "trigger":
        from ..capture.patternmode import ACK_PHRASE
        r = api.scan_trigger(sc["pack"], sc["variant"], label, preset,
                             acknowledge=ACK_PHRASE if pattern_mode_ack else None)
        ms = r["scan"]
        pats = f"scan_trigger {Path(str(sc['pack'])).name}"
    else:
        kw = {}
        if sc["hold_s"] is not None:
            kw["min_hold_s"] = sc["hold_s"]
        ms = api.scan_fast(sc["zone"], preset, label, sc["steps"] or 4, sc["periods"],
                           sc["average"] if sc["average"] is not None else 2, axis=sc["axis"],
                           **kw)
        pats = f"scan_fast axis {sc['axis']}"
    m = ms.meta or {}
    res["capture"].update(patterns=f"multifreq ({pats})", average=m.get("average",
                                                                         res["capture"]["average"]))
    return ms


def execute(s, recipe, preset, *, res: dict, log, run_dir: Path, scan=None,
            mock_place: Optional[Callable] = None, pattern_mode_ack: bool = False,
            label: Optional[str] = None) -> dict:
    """Fill `res` (the run's result document) for one recipe run. -> state for finish()."""
    from .. import api
    from ..calibrate import stereo
    from ..reconstruct import metric_scan
    state = {"img": None, "fb_info": None}
    fp = config.geometry_fingerprint(preset)
    bad_part_output(res, recipe)
    res["recipe"] = {**recipe.summary(), "tier": None, "result_code": None, "frame": None}
    res["capture"].update(preset=preset.name, geometry_id=preset.geometry_id,
                          exposure_us=preset.camera.exposure_us,
                          average=preset.camera.average, zone=recipe.scan["zone"])
    if preset.name != recipe.preset:
        res["warnings"].append(f"recipe {recipe.name} pins preset {recipe.preset}; this run "
                               f"used {preset.name}")
    cal = stereo.latest(s.store, preset.geometry_id, fp)
    if cal is None:
        res["calibration"]["same_geometry"] = False
        res["invalid_reason"] = (f"no stereo calibration for geometry {preset.geometry_id} (these "
                                 f"rig settings), so no mm (rule 6): "
                                 + CALIBRATE_STEREO_HOWTO.format(preset=preset.name))
        D.add(res, "uncalibrated", res["invalid_reason"])
        log.event("refused", level="warning", reason=res["invalid_reason"])
        return state
    res["calibration"].update(calibration_id=cal.id, same_geometry=True)
    s.check_cancel()
    if recipe.scan["rig_check"] and scan is None:
        rc = api.rig_check(preset)
        log.event("rig_check", status=rc["status"], message=rc.get("message"))
        if rc["status"] in ("MOVED", "UNKNOWN"):
            res["invalid_reason"] = f"rig check {rc['status']}: {rc.get('message')} (HOME / recalibrate)"
            D.add(res, "rig_check", res["invalid_reason"])
            return state
        if rc["status"] == "REFERENCE_SET":
            res["warnings"].append(f"rig check: {rc.get('message') or 'reference stored'}")
    if scan is None and _rig_refused(s, preset, res, log):
        return state
    codes = None
    if recipe.scan["codes"] and scan is None:
        rd = api.read_codes(preset, label=f"{label or recipe.name}_codes")
        codes = list(rd.codes)
        log.event("codes", n=len(codes), data=[c.data for c in codes])
    s.check_cancel()
    t0 = time.perf_counter()
    if scan is None:
        if mock_place is not None:
            mock_place()
        ms = _scan(s, recipe, preset, label or recipe.part.get("label") or recipe.name, log, res,
                   pattern_mode_ack)
    else:
        ms = _find_metric(s, scan, preset.geometry_id)
        rules.require_same_geometry(ms.geometry_id, preset.geometry_id, "recipe_run scan")
        metric_scan.check_cal(s, ms, "recipe_run")               # rule 6 at use time
        if _rig_refused(s, preset, res, log, ms):
            return state
        res["capture"]["patterns"] = f"stored metric scan ({(ms.meta or {}).get('method', '?')})"
        res["warnings"].append(f"offline: checks run on the stored scan {ms.id} (nothing captured)")
    _same_setup(ms, preset, fp, cal, "recipe_run scan")      # this preset's settings + calibration
    res["part"]["scan_id"] = ms.id
    res["calibration"]["calibration_id"] = ms.calibration_id
    res["warnings"] += [str(w) for w in (ms.meta or {}).get("warnings") or []]
    log.event("scan", scan_id=ms.id, kind=recipe.scan["kind"] if scan is None else "stored",
              calibration_id=ms.calibration_id, n_points=(ms.meta or {}).get("n_points"),
              capture_s=round(time.perf_counter() - t0, 2))
    s.check_cancel()
    sv = ScanView.from_scan(ms, cal)
    try:
        frame = frame_for(recipe, sv)
    except ZoneError as e:
        res["invalid_reason"] = f"part frame: {e}"
        D.add(res, D.classify(str(e)), res["invalid_reason"])
        log.event("refused", level="warning", reason=res["invalid_reason"])
        return state
    res["recipe"]["frame"] = frame.summary()
    views = {}
    for z in recipe.zones:
        try:
            views[z.id] = resolve_zone(z, sv, frame)
        except (ZoneError, ValueError) as e:
            views[z.id] = None
            res["warnings"].append(f"zone {z.id}: {e}")
    golden, gerr = None, None
    if "golden" in recipe.needs:
        golden, gerr = _golden(s, recipe, preset, cal, exclude=(ms.id,))
        if gerr:
            res["warnings"].append(f"golden: {gerr}")
    zres = []
    sections = {}                           # height lines a check gridded: files, never results
    for z in recipe.zones:
        zv = views[z.id]
        entry = {"id": z.id, "label": z.label, "shape": z.shape, "units": z.units,
                 "datum": z.datum, "verdict": None, "n_px": None, "n_points": None,
                 "checks": [], "warnings": []}
        if zv is None:
            entry["checks"] = [{"id": c.id, "type": c.type, "verdict": "INVALID",
                                "values": {}, "units": {}, "key": None, "limit": None,
                                "message": "the zone could not be resolved on this scan",
                                "fault_points": [], "warnings": [], "details": {},
                                "tolerances": None, "pull": "zone_unresolved"}
                               for c in z.checks]
        else:
            entry.update(n_px=zv.n_px, n_points=zv.n_valid, warnings=list(zv.warnings))
            ctx = CheckContext(scan=sv, frame=frame, zone=zv,
                               zones={k: v for k, v in views.items() if v is not None},
                               golden=golden, golden_error=gerr, codes=codes,
                               recipe_name=recipe.name, sections=sections)
            for c in z.checks:
                s.check_cancel()
                entry["checks"].append(_run_check(c, ctx))
        entry["verdict"] = (aggregate(ch["verdict"] for ch in entry["checks"])
                            if entry["checks"] else ("PASS" if zv is not None else "INVALID"))
        zres.append(entry)
        log.event("zone", zone=z.id, verdict=entry["verdict"], n_points=entry["n_points"],
                  checks={ch["id"]: ch["verdict"] for ch in entry["checks"]})
    eventlog.write_profiles(run_dir, sections)      # profile_x.csv / profile_y.csv (never raises)
    checked = [z for z in zres if z["checks"]]
    verdict = aggregate(z["verdict"] for z in checked)
    res["zones"] = zres
    res["result"] = res["engine_verdict"] = verdict
    res["invalid_reason"] = None if verdict != "INVALID" else _invalid_reason(checked)
    # the disposition first: the pictures and the result code show it
    settle(res, recipe)
    disp, pulls = res["disposition"], res["pull_reasons"]
    _headline(res, zres, ms)
    # camera overlay + the DMD feedback image (files in the run folder)
    fb = recipe.feedback
    if recipe.outputs.get("overlay", True):
        try:
            png = FB.write_png(Path(run_dir) / OVERLAY_IMAGE,
                               FB.overlay(sv, frame, {k: v for k, v in views.items() if v},
                                          zres, fb, verdict, disposition=disp,
                                          pull_reasons=pulls))
            res["outputs"]["deviation_map"] = png
        except Exception as e:              # noqa: BLE001 - a picture never changes a verdict
            res["warnings"].append(f"overlay image: {type(e).__name__}: {e}")
    info = {"enabled": bool(fb["enabled"]), "rendered": False, "image": None,
            "hold_s": fb["hold_s"], "projected": None}
    if fb["enabled"]:
        try:
            img, rinfo = FB.render({k: v for k, v in views.items() if v}, zres, sv, fb, verdict,
                                   disposition=disp, pull_reasons=pulls)
            info.update(rinfo)
            if img is not None:
                info["image"] = FB.write_png(Path(run_dir) / FEEDBACK_IMAGE, img)
                info["rendered"] = True
                state["img"] = img
        except Exception as e:              # noqa: BLE001
            info["refused"] = f"{type(e).__name__}: {e}"
        if info.get("refused"):
            res["warnings"].append(f"feedback not projected: {info['refused']}")
        elif info.get("frame"):
            res["warnings"].append(f"feedback: {info['frame']['why']}: the whole DMD field is "
                                   f"framed {info['frame']['verdict']}")
    res["feedback"] = info
    log.event("verdict", result=verdict, disposition=disp, pull_reasons=pulls,
              zones={z["id"]: z["verdict"] for z in zres}, feedback_rendered=info["rendered"])
    return state


def bad_part_output(res: dict, recipe) -> None:
    """The recipe's outputs.bad_part_output (relative to the recipe file) into the run
    document - unless the station configuration put its own there first: the station's wins
    (engine/badparts.py; the command line and the environment are settled when result.json
    is written). Never raises: it is set before the run's own error handling."""
    outs = res.setdefault("outputs", {})
    mine = recipe_bad_part_output(recipe)
    if outs.get(badparts.KEY) is None and mine is not None:
        outs[badparts.KEY] = mine


def recipe_bad_part_output(recipe) -> Optional[str]:
    """A recipe's own outputs.bad_part_output: a full folder path (relative to the recipe
    file), "off", or None (it says nothing; a value that is no folder path, which loading the
    recipe refuses). Text only: the folder is not touched."""
    mine = (getattr(recipe, "outputs", None) or {}).get(badparts.KEY)
    src = getattr(recipe, "source", None)
    try:
        return badparts.configured(mine, os.path.dirname(os.path.abspath(str(src)))
                                   if src else None)
    except ValueError:
        return None


def _rig_refused(s, preset, res: dict, log, scan=None) -> bool:
    """A MOBILE rig (preset rig.mount; or another preset at its geometry says so) is assumed
    moved since its last use: no mm until it was verified in this session, and a stored scan
    must have been made after that (engine/rigstate.py). -> True when the run stops here:
    INVALID, a doubt (PULL; `rig_unverified`, which no recipe can turn into a reject). A
    fixed rig is never gated here (unchanged: its calibration is trusted until a rig check
    says MOVED / UNKNOWN - scan.rig_check, the command bus's HOME)."""
    gate = rigstate.gate(s.store, preset, strict=False, scan=scan,
                         presets_dir=getattr(s, "presets_dir", None))
    if gate["ok"]:
        return False
    res["invalid_reason"] = gate["message"]
    D.add(res, gate["code"], gate["message"])
    log.event("refused", level="warning", reason=gate["message"], mount=gate["mount"],
              how_to_fix=gate["how_to_fix"])
    return True


def settle(res: dict, recipe) -> dict:
    """res["disposition"] / res["pull_reasons"] with the recipe's pull config, and the
    result code of that disposition (engine/disposition.register_key), on every path (a run
    refused before the recipe block was filled gets it here). Call it again whenever the
    verdict changes after it (ABORTED, refused, crashed): a pulled cycle never keeps the
    PASS code an earlier settle wrote. Idempotent."""
    pull = getattr(recipe, "pull", None) or {}
    D.settle(res, guard_band_pct=pull.get("guard_band_pct") or 0.0,
             reject=pull.get("invalid_reasons_to_reject") or ())
    if res.get("recipe") is None:
        res["recipe"] = {**recipe.summary(), "tier": None, "result_code": None, "frame": None}
    key = D.register_key(res["result"], res["disposition"])
    res["recipe"]["result_code"] = recipe.outputs["result_codes"].get(key)
    return res


def _invalid_check(message: str, pull: str = "invalid") -> dict:
    return {"verdict": "INVALID", "values": {}, "units": {}, "key": None, "limit": None,
            "message": message, "fault_points": [], "warnings": [], "details": {},
            "tolerances": None, "pull": pull}


def _run_check(c, ctx) -> dict:
    t0 = time.perf_counter()
    try:
        r = c.check.run(ctx)
        if not isinstance(r, CheckResult):
            raise TypeError(f"run() returned {type(r).__name__}, not a CheckResult")
        d = r.to_dict()
        if d.get("verdict") not in VERDICTS:        # e.g. set after construction
            raise ValueError(f"verdict {d.get('verdict')!r} is not one of {', '.join(VERDICTS)}")
        d["tolerances"] = (None if d.get("tolerances") is None       # malformed: a bug
                           else D.check_tolerances(d["tolerances"]))
        warn = []
        if d.get("pull") is not None and d["pull"] not in D.REASONS:
            warn.append(f"pull {d['pull']!r} is not a known reason: recorded as 'invalid'")
            d["pull"] = "invalid"                   # still a doubt, never dropped
        d["fault_points"], fwarn = _fault_px(d.get("fault_points") or [], ctx)
        d["warnings"] = list(d.get("warnings") or []) + warn + fwarn
    except Exception as e:                  # noqa: BLE001 - a crashing check is never a verdict
        d = _invalid_check(f"the check failed: {type(e).__name__}: {e}", "check_crashed")
    d = {"id": c.id, "type": c.type, **d, "params": dict(c.check.p),
         "duration_s": round(time.perf_counter() - t0, 3)}
    return _clean(d)


def _fault_px(faults: list, ctx) -> tuple:
    """Every fault point gets its camera px (the projector's cross and the overlay mark
    need one): {"xyz_mm": [x, y, z]} (camera frame) is projected with the calibration's
    intrinsics. -> (fault points, warnings for those that could not be placed)."""
    out, warn = [], []
    for fp in faults:
        fp = dict(fp)
        if fp.get("px") is None and fp.get("xyz_mm") is not None:
            try:
                px = ctx.px_of(fp["xyz_mm"])
                if not all(np.isfinite(px)):
                    raise ValueError("projects to no camera px")
                fp["px"] = px
            except (TypeError, ValueError) as e:
                warn.append(f"fault point {fp.get('label', '')!r} at xyz_mm {fp['xyz_mm']}: "
                            f"{e}: not marked")
        elif fp.get("px") is None:
            warn.append(f"fault point {fp.get('label', '')!r} has neither px nor xyz_mm: "
                        "not marked")
        out.append(fp)
    return out, warn


def _clean(o):
    """numpy scalars / arrays -> plain JSON types (result.json, the web UI)."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, np.generic):
        return _clean(o.item())
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def _invalid_reason(zres: list) -> str:
    bits = []
    for z in zres:
        for ch in z["checks"]:
            if ch["verdict"] == "INVALID":
                bits.append(f"{z['id']}/{ch['id']}: {ch['message']}")
    return "; ".join(bits[:4]) + (f" (+{len(bits) - 4} more)" if len(bits) > 4 else "") \
        if bits else "no check could run"


def _headline(res: dict, zres: list, ms) -> None:
    """deviation_mm / threshold_mm: a check's key value in deviation semantics (a deviation
    and its tolerance: |value| > limit is out). A FAILing check always outranks a passing
    one; among them the largest |value| / limit. A FAIL run whose failing checks carry no
    mm value names the failing check with no number (never a passing check's)."""
    best, failed = None, None
    for z in zres:
        for ch in z["checks"]:
            if ch.get("verdict") not in ("PASS", "FAIL"):
                continue
            if ch["verdict"] == "FAIL" and failed is None:
                failed = f"{z['id']}/{ch['id']}"
            k, lim = ch.get("key"), ch.get("limit")
            v = (ch.get("values") or {}).get(k) if k else None
            if v is None or lim is None or not str(k).endswith("_mm"):
                continue
            try:
                v, lim = float(v), float(lim)
            except (TypeError, ValueError):
                continue
            if not (np.isfinite(v) and np.isfinite(lim)) or lim <= 0:
                continue
            rank = (ch["verdict"] == "FAIL", abs(v) / lim)
            if best is None or rank > best[0]:
                best = (rank, v, lim, f"{z['id']}/{ch['id']} {k}")
    res["deviation_units"] = "mm"
    n = int((ms.meta or {}).get("n_points") or 0) or None
    if best is not None and (failed is None or best[0][0]):
        res["deviation_mm"].update(max=round(best[1], 5), method=f"recipe {best[3]}", n_points=n)
        res["threshold_mm"] = best[2]
    elif failed is not None:
        res["deviation_mm"].update(method=f"recipe {failed} (FAIL, no mm value)", n_points=n)
    else:
        res["deviation_mm"].update(method="recipe (no mm check)", n_points=n)


def _golden(s, recipe, preset, cal, exclude=()) -> tuple:
    """-> (GoldenView, None) or (None, why). The golden: find_golden (never one of
    `exclude`, the run's own scan), at this preset's geometry-bound settings and made with
    the run's stereo calibration `cal`."""
    from ..reconstruct import metric_scan
    key = (recipe.golden or {}).get("scan")
    if not key:
        return None, "the recipe names no golden scan (golden.scan)"
    try:
        g = find_golden(s, recipe, preset.geometry_id, key, exclude=exclude)
        metric_scan.check_cal(s, g, "recipe golden")
        _same_setup(g, preset, config.geometry_fingerprint(preset), cal, "recipe golden")
    except (LookupError, rules.RuleViolation) as e:
        return None, (f"golden scan {key!r}: {e} - scan a GOOD part with `scan recipe-golden "
                      f"{recipe.name}`")
    try:
        sv = ScanView.from_scan(g, cal)
        fr = frame_for(recipe, sv)
        zones = {}
        for z in recipe.zones:
            try:
                zones[z.id] = resolve_zone(z, sv, fr)
            except (ZoneError, ValueError):
                continue
    except ZoneError as e:
        return None, f"golden scan {g.id}: {e}"
    return GoldenView(scan=sv, frame=fr, zones=zones, scan_id=g.id), None


# ------------------------------------------------------------------ finish: result + feedback

def finish(s, preset, recipe, run_dir: Path, res: dict, log, state: dict,
           project_feedback: bool, hold_s: Optional[float] = None, cancel=None,
           exit_code: Optional[int] = None) -> Path:
    """result.json (once), then the projected feedback (if rendered and wanted), run_end."""
    out = R.write_result(run_dir, res)
    img = state.get("img") if state else None
    fbinfo = res.get("feedback") or {}
    if project_feedback and img is not None and res["result"] in ("PASS", "FAIL", "INVALID"):
        hold = float(fbinfo.get("hold_s") if hold_s is None else hold_s)
        try:
            pr = FB.project(s, preset, img, hold, cancel=cancel,
                            photo=bool(recipe.feedback.get("photo", True)))
            photo = None
            if pr["photo"] is not None:
                from ..capture.zones import _photo_u8
                photo = FB.write_png(Path(run_dir) / FEEDBACK_PHOTO, _photo_u8(pr["photo"]))
            log.event("feedback_shown", hold_s=pr["hold_s"], held_s=pr["held_s"],
                      polarity_inverted=pr["inverted"], photo=photo, warnings=pr["warnings"])
            if pr["cancelled"]:
                log.event("feedback_cleared", level="info", held_s=pr["held_s"])
        except rules.Cancelled as e:            # ABORT while it came up: cleared, verdict stands
            log.event("feedback_cleared", level="info", message=str(e))
        except KeyboardInterrupt:               # Ctrl+C on the command line: the same
            log.event("feedback_cleared", level="info", message="interrupted (Ctrl+C)")
        except Exception as e:                  # noqa: BLE001 - the verdict is already written
            log.event("feedback_error", level="warning", type=type(e).__name__, message=str(e))
    elif fbinfo.get("rendered"):
        log.event("feedback_skipped", reason="project_feedback off (the caller shows it)")
    log.event("run_end", result=res["result"], disposition=res.get("disposition"),
              pull_reasons=res.get("pull_reasons"), exit_code=exit_code, result_json=out.name,
              invalid_reason=res["invalid_reason"])
    log.close()
    return out


def run_recipe(s, recipe, preset, *, runs_dir, operator_confirmed: bool = False,
               project_feedback: bool = True, scan=None, hold_s: Optional[float] = None,
               run_name: Optional[str] = None, mock_place: Optional[Callable] = None,
               pattern_mode_ack: bool = False, label: Optional[str] = None,
               tier: Optional[str] = None, station: Optional[str] = None,
               context: Optional[dict] = None) -> dict:
    """One recipe run in a NEW run folder -> {run_id, run_dir, result, exit_code}. On the real
    rig it needs operator_confirmed, except an offline re-check: a stored `scan` with no
    projected feedback captures and projects nothing (the rig is never opened). exit_code:
    the disposition's (CONTINUE 0, REJECT 1, PULL 5); refused 2, crashed 4, ABORTED 1 keep
    theirs (disposition PULL). station / context: a work state's check (engine/flow/
    verify.py) names its station and adds {station_config, profile, job: {config,
    work_state}} to the result; context bad_part_output: the station configuration's bad-part
    output (engine/badparts.py), which wins over the recipe's outputs.bad_part_output."""
    ctx = dict(context or {})
    run_id, run_dir = R.new_run(runs_dir, run_name or recipe.name)
    res = R.empty_result(run_id=run_id, run_folder=str(run_dir), mock=s.mock,
                         operator_confirmed=operator_confirmed,
                         station=station or run_name or recipe.name,
                         job={"type": "recipe", "recipe": recipe.name, "base": None,
                              "zone": recipe.scan["zone"], "config": None,
                              **(ctx.pop("job", None) or {})},
                         part={"role": "test", "label": label or recipe.part.get("label")
                               or recipe.name}, zone=recipe.scan["zone"],
                         station_config=ctx.pop("station_config", None),
                         profile=ctx.pop("profile", None))
    res["outputs"][badparts.KEY] = ctx.pop(badparts.KEY, None)     # the station's, if it says
    bad_part_output(res, recipe)
    log = R.RunLog(Path(run_dir) / "run.log.jsonl", run_id)
    state = {}
    code = 5
    try:
        log.event("run_start", station=res["station"], mode=res["mode"], job="recipe",
                  recipe=recipe.name, preset=preset.name, operator_confirmed=operator_confirmed)
        offline = scan is not None and not project_feedback
        if not s.mock and not operator_confirmed and not offline:
            raise rules.RuleViolation("the real rig needs a human to confirm the physical setup "
                                      "first (part placed, rig untouched): operator_confirmed "
                                      "(a stored scan re-checked without feedback needs none)")
        state = execute(s, recipe, preset, res=res, log=log, run_dir=run_dir, scan=scan,
                        mock_place=mock_place, pattern_mode_ack=pattern_mode_ack, label=label)
        if res.get("recipe") is not None and tier:
            res["recipe"]["tier"] = tier
        settle(res, recipe)
        code = D.exit_code(res)
        s.check_cancel()                # an ABORT up to here: the cycle is ABORTED, not judged
    except rules.Cancelled as e:
        res["result"], res["invalid_reason"] = "ABORTED", None
        res["warnings"].append(f"aborted: {e}")
        D.add(res, "aborted", f"aborted: {e}")
        log.event("aborted", level="warning", message=str(e))
        state, code = {}, 1
    except Exception as e:                      # noqa: BLE001 - refused / crashed: INVALID
        res["result"] = "INVALID"
        res["invalid_reason"] = f"{type(e).__name__}: {e}"
        log.event("error", level="error", type=type(e).__name__, message=str(e))
        state = {}
        code = 2 if isinstance(e, (rules.RuleViolation, config.ConfigError, ValueError,
                                   LookupError)) else 4
        # recorded as what happened, never keyword-classified: a crash text is not engine
        # wording, and a run that never finished is never a reject (engine/disposition.py)
        D.add(res, "refused" if code == 2 else "error", res["invalid_reason"])
    settle(res, recipe)                         # every path: ABORTED and refused are PULL
    finish(s, preset, recipe, run_dir, res, log, state, project_feedback, hold_s,
           cancel=s.cancel, exit_code=code)
    return {"run_id": run_id, "run_dir": run_dir, "result": res, "exit_code": code}

"""
Recipe runner: preset -> scan -> zones -> checks -> verdict -> run folder -> projected feedback.

    execute()     scan per the recipe (or take a stored scan), resolve the part frame and the
                  zones, run every check, aggregate, write the camera overlay and the DMD
                  feedback image into the run folder, fill the result document (never writes
                  result.json itself)
    finish()      result.json (once), then the projected feedback (pumped hold, ABORT clears
                  it), run_end in the log
    run_recipe()  the whole thing in a NEW run folder (api.recipe_run)

Verdict: any check FAIL -> FAIL (a trusted measurement found a bad part); else any INVALID ->
INVALID (never a pass: no stereo calibration at the preset's geometry, a scan made under
another calibration, a zone with too few points, no golden scan, ...); else PASS. A check that
raises is INVALID with its error (a bug is never a verdict).

result.json = the run schema (engine/runs.py) + `recipe` (name, preset, frame, result code),
`zones` [{id, verdict, checks: [{id, type, verdict, values (units in the names), key, limit,
message, fault_points}]}] and `feedback` (what the DMD showed: zones drawn / refused with the
camera -> DMD fit residuals, crosses, text, image file).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .. import config, rules
from .. import runs as R
from . import feedback as FB
from .context import CheckContext, GoldenView
from .geometry import ScanView, ZoneError, frame_for, resolve_zone

CALIBRATE_STEREO_HOWTO = ("calibrate it first: `scan charuco` (print at 100 %, caliper a "
                          "square), `scan calib-pose` per board pose (12-20, tilted +-30 deg), "
                          "then `scan calibrate-stereo --square MM` (mock: `scan --mock "
                          "calibrate-stereo --mock-poses 12`)")
FEEDBACK_IMAGE = "feedback_dmd.png"
FEEDBACK_PHOTO = "feedback_photo.png"
OVERLAY_IMAGE = "overlay.png"
ORDER = {"PASS": 0, "INVALID": 1, "FAIL": 2}


def aggregate(verdicts) -> str:
    v = list(verdicts)
    if "FAIL" in v:
        return "FAIL"
    if "INVALID" in v or not v:
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
                                 f"rig settings), so no mm (rule 6): {CALIBRATE_STEREO_HOWTO}")
        log.event("refused", level="warning", reason=res["invalid_reason"])
        return state
    res["calibration"].update(calibration_id=cal.id, same_geometry=True)
    s.check_cancel()
    if recipe.scan["rig_check"] and scan is None:
        rc = api.rig_check(preset)
        log.event("rig_check", status=rc["status"], message=rc.get("message"))
        if rc["status"] in ("MOVED", "UNKNOWN"):
            res["invalid_reason"] = f"rig check {rc['status']}: {rc.get('message')} (HOME / recalibrate)"
            return state
        if rc["status"] == "REFERENCE_SET":
            res["warnings"].append("rig check: first check at this geometry, reference stored")
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
        res["capture"]["patterns"] = f"stored metric scan ({(ms.meta or {}).get('method', '?')})"
        res["warnings"].append(f"offline: checks run on the stored scan {ms.id} (nothing captured)")
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
        golden, gerr = _golden(s, recipe, preset, cal)
        if gerr:
            res["warnings"].append(f"golden: {gerr}")
    zres = []
    for z in recipe.zones:
        zv = views[z.id]
        entry = {"id": z.id, "label": z.label, "shape": z.shape, "units": z.units,
                 "datum": z.datum, "verdict": None, "n_px": None, "n_points": None,
                 "checks": [], "warnings": []}
        if zv is None:
            entry["checks"] = [{"id": c.id, "type": c.type, "verdict": "INVALID",
                                "values": {}, "units": {}, "key": None, "limit": None,
                                "message": "the zone could not be resolved on this scan",
                                "fault_points": [], "warnings": [], "details": {}}
                               for c in z.checks]
        else:
            entry.update(n_px=zv.n_px, n_points=zv.n_valid, warnings=list(zv.warnings))
            ctx = CheckContext(scan=sv, frame=frame, zone=zv,
                               zones={k: v for k, v in views.items() if v is not None},
                               golden=golden, golden_error=gerr, codes=codes,
                               recipe_name=recipe.name)
            for c in z.checks:
                s.check_cancel()
                entry["checks"].append(_run_check(c, ctx))
        entry["verdict"] = (aggregate(ch["verdict"] for ch in entry["checks"])
                            if entry["checks"] else ("PASS" if zv is not None else "INVALID"))
        zres.append(entry)
        log.event("zone", zone=z.id, verdict=entry["verdict"], n_points=entry["n_points"],
                  checks={ch["id"]: ch["verdict"] for ch in entry["checks"]})
    checked = [z for z in zres if z["checks"]]
    verdict = aggregate(z["verdict"] for z in checked)
    res["zones"] = zres
    res["result"] = res["engine_verdict"] = verdict
    res["recipe"]["result_code"] = recipe.outputs["result_codes"].get(verdict)
    res["invalid_reason"] = None if verdict != "INVALID" else _invalid_reason(checked)
    _headline(res, zres, ms)
    # camera overlay + the DMD feedback image (files in the run folder)
    fb = recipe.feedback
    if recipe.outputs.get("overlay", True):
        try:
            png = FB.write_png(Path(run_dir) / OVERLAY_IMAGE,
                               FB.overlay(sv, frame, {k: v for k, v in views.items() if v},
                                          zres, fb, verdict))
            res["outputs"]["deviation_map"] = png
        except Exception as e:              # noqa: BLE001 - a picture never changes a verdict
            res["warnings"].append(f"overlay image: {type(e).__name__}: {e}")
    info = {"enabled": bool(fb["enabled"]), "rendered": False, "image": None,
            "hold_s": fb["hold_s"], "projected": None}
    if fb["enabled"]:
        try:
            img, rinfo = FB.render({k: v for k, v in views.items() if v}, zres, sv, fb, verdict)
            info.update(rinfo)
            if img is not None:
                info["image"] = FB.write_png(Path(run_dir) / FEEDBACK_IMAGE, img)
                info["rendered"] = True
                state["img"] = img
        except Exception as e:              # noqa: BLE001
            info["refused"] = f"{type(e).__name__}: {e}"
        if info.get("refused"):
            res["warnings"].append(f"feedback not projected: {info['refused']}")
    res["feedback"] = info
    log.event("verdict", result=verdict, zones={z["id"]: z["verdict"] for z in zres},
              feedback_rendered=info["rendered"])
    return state


def _run_check(c, ctx) -> dict:
    t0 = time.perf_counter()
    try:
        r = c.check.run(ctx)
        d = r.to_dict()
    except Exception as e:                  # noqa: BLE001 - a crashing check is never a verdict
        d = {"verdict": "INVALID", "values": {}, "units": {}, "key": None, "limit": None,
             "message": f"the check failed: {type(e).__name__}: {e}", "fault_points": [],
             "warnings": [], "details": {}}
    d = {"id": c.id, "type": c.type, **d, "params": dict(c.check.p),
         "duration_s": round(time.perf_counter() - t0, 3)}
    return _clean(d)


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
    """deviation_mm / threshold_mm: the mm value closest to (or furthest past) its limit."""
    best = None
    for z in zres:
        for ch in z["checks"]:
            k, lim = ch.get("key"), ch.get("limit")
            v = (ch.get("values") or {}).get(k) if k else None
            if v is None or lim in (None, 0) or not str(k).endswith("_mm"):
                continue
            ratio = abs(float(v)) / abs(float(lim))
            if best is None or ratio > best[0]:
                best = (ratio, float(v), float(lim), f"{z['id']}/{ch['id']} {k}")
    res["deviation_units"] = "mm"
    n = int((ms.meta or {}).get("n_points") or 0) or None
    if best is not None:
        res["deviation_mm"].update(max=round(best[1], 5), method=f"recipe {best[3]}", n_points=n)
        res["threshold_mm"] = best[2]
    else:
        res["deviation_mm"].update(method="recipe (no mm check)", n_points=n)


def _golden(s, recipe, preset, cal) -> tuple:
    """-> (GoldenView, None) or (None, why)."""
    from ..reconstruct import metric_scan
    key = (recipe.golden or {}).get("scan")
    if not key:
        return None, "the recipe names no golden scan (golden.scan)"
    try:
        g = _find_metric(s, key, preset.geometry_id)
        metric_scan.check_cal(s, g, "recipe golden")
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
    log.event("run_end", result=res["result"], exit_code=exit_code, result_json=out.name,
              invalid_reason=res["invalid_reason"])
    log.close()
    return out


def run_recipe(s, recipe, preset, *, runs_dir, operator_confirmed: bool = False,
               project_feedback: bool = True, scan=None, hold_s: Optional[float] = None,
               run_name: Optional[str] = None, mock_place: Optional[Callable] = None,
               pattern_mode_ack: bool = False, label: Optional[str] = None,
               tier: Optional[str] = None) -> dict:
    """One recipe run in a NEW run folder -> {run_id, run_dir, result, exit_code}."""
    run_id, run_dir = R.new_run(runs_dir, run_name or recipe.name)
    res = R.empty_result(run_id=run_id, run_folder=str(run_dir), mock=s.mock,
                         operator_confirmed=operator_confirmed, station=run_name or recipe.name,
                         job={"type": "recipe", "recipe": recipe.name, "base": None,
                              "zone": recipe.scan["zone"], "config": None},
                         part={"role": "test", "label": label or recipe.part.get("label")
                               or recipe.name}, zone=recipe.scan["zone"])
    log = R.RunLog(Path(run_dir) / "run.log.jsonl", run_id)
    state = {}
    code = 5
    try:
        log.event("run_start", station=res["station"], mode=res["mode"], job="recipe",
                  recipe=recipe.name, preset=preset.name, operator_confirmed=operator_confirmed)
        if not s.mock and not operator_confirmed:
            raise rules.RuleViolation("the real rig needs a human to confirm the physical setup "
                                      "first (part placed, rig untouched): operator_confirmed")
        state = execute(s, recipe, preset, res=res, log=log, run_dir=run_dir, scan=scan,
                        mock_place=mock_place, pattern_mode_ack=pattern_mode_ack, label=label)
        if res.get("recipe") is not None and tier:
            res["recipe"]["tier"] = tier
        code = {"PASS": 0, "FAIL": 1}.get(res["result"], 5)
    except rules.Cancelled as e:
        res["result"], res["invalid_reason"] = "ABORTED", None
        res["warnings"].append(f"aborted: {e}")
        log.event("aborted", level="warning", message=str(e))
        state, code = {}, 1
    except Exception as e:                      # noqa: BLE001 - refused / crashed: INVALID
        res["result"] = "INVALID"
        res["invalid_reason"] = f"{type(e).__name__}: {e}"
        log.event("error", level="error", type=type(e).__name__, message=str(e))
        state = {}
        code = 2 if isinstance(e, (rules.RuleViolation, config.ConfigError, ValueError,
                                   LookupError)) else 4
    if res.get("recipe") is not None:
        res["recipe"]["result_code"] = recipe.outputs["result_codes"].get(res["result"])
    finish(s, preset, recipe, run_dir, res, log, state, project_feedback, hold_s,
           cancel=s.cancel, exit_code=code)
    return {"run_id": run_id, "run_dir": run_dir, "result": res, "exit_code": code}

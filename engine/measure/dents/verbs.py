"""
Dent verbs behind engine/api.py: capture a dent set, calibrate the finders
on a known-good cell, detect dents.

Library (append-only, rule 7):
  dentsets/<geometry_id>/<ts>_<label>/        frames_long.npy, frames_short.npy (float32 camera
                                              DN, cropped), cam_zone_mask.npy, dmd_zone_mask.npy,
                                              preview.png, meta.json (written last)
  calibrations/<geometry_id>/<ts>_dentnull_<label>/  dent_null.json (the null calibration)
  output/defects/<ts>_<label>_dents.{json,png,npz}   detections, overlay, maps

Rules: both exposures (and the metric scan) use presets of ONE geometry
(same geometry_id and fingerprint, rule 5); the two exposure presets may
differ ONLY in camera exposure (the HDR composite scales the short pass by
the exposure ratio). Everything in mm comes from the metric scans' stereo
calibration, which must still be the library's current one for the
geometry (rule 6) - checked before the rig opens (capture) or before any
analysis (calibrate / detect). A null calibration is bound to the geometry
and fingerprint, the cell diameter, the fixture (the zone unless named),
the pattern set and the stereo calibration it was made under: detect_dents
auto-picks only a null matching all of them (the newest; without a cell
diameter the matching nulls must agree on one), refuses an explicitly
named null from another geometry, fingerprint, diameter, pattern set or
stereo calibration, and only warns when an explicitly named null is from
another fixture (the operator's choice).
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from ... import config, rules
from ...library.store import json_safe, read_meta, safe_label
from ...results import DentReport, DentSet, MetricScan
from . import dentset as ds_mod
from . import fusion, overlay

NULL_FILE = "dent_null.json"
DIAMETER_TOL_MM = 0.01
CROP_MARGIN_PX = 10


# ------------------------------------------------------------ helpers

def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _save_npy(path: Path, arr: np.ndarray) -> Path:
    rules.ensure_new_path(path)
    with open(path, "xb") as f:
        np.save(f, arr)
    return path


def with_exposure(preset, exposure_us: int):
    """A copy of `preset` with another camera exposure (the file is untouched). Exposure
    is not geometry (config.geometry_fingerprint), so rules 5 / 6 hold across the two."""
    if not isinstance(exposure_us, (int, float)) or exposure_us <= 0:
        raise ValueError(f"exposure_us must be > 0, got {exposure_us!r}")
    cam = replace(preset.camera, exposure_us=int(round(exposure_us)))
    warns = list(preset.warnings)
    if not preset.mock and not config.frame_locked(cam.exposure_us):
        warns.append(f"exposure {cam.exposure_us} us is not a whole number of 60 Hz video frames "
                      "(frames flicker); prefer 16667 / 33333 / 66667 us")
    return replace(preset, camera=cam, warnings=warns)


def radiometric_diffs(a, b) -> list:
    """Settings other than the exposure that change camera DN per unit of light between two
    presets (the HDR composite scales the short pass by the exposure ratio ONLY)."""
    pairs = (("camera.gain_db", a.camera.gain_db, b.camera.gain_db),
             ("camera.mono", a.camera.mono, b.camera.mono),
             ("projection.duty", a.projection.duty, b.projection.duty),
             ("projection.gamma", a.projection.gamma, b.projection.gamma),
             ("projection.invert", a.projection.invert, b.projection.invert),
             ("projection.scheme", a.projection.scheme, b.projection.scheme),
             ("projection.led", a.projection.led, b.projection.led))
    return [f"{k} {va!r} vs {vb!r}" for k, va, vb in pairs if va != vb]


def _flood_index(specs: list, level: int) -> int:
    return next(i for i, q in enumerate(specs) if q["kind"] == "flood" and q["level"] == level)


def _crop_box(cam_mask, on: np.ndarray, off: np.ndarray, shape) -> tuple:
    """(x0, y0, x1, y1): the zone's camera mask bbox, else where the projector lights the
    scene (|on - off| well above the noise), + a margin; full frame as a last resort."""
    H, W = shape
    m = None
    if cam_mask is not None and np.asarray(cam_mask, bool).any():
        m = np.asarray(cam_mask, bool)
    else:
        d = np.abs(np.asarray(on, float) - np.asarray(off, float))
        thr = max(8.0, 0.1 * float(np.percentile(d, 99.5)))
        lit = d > thr
        if lit.sum() > 500:
            m = lit
    if m is None:
        return 0, 0, W, H
    ys, xs = np.nonzero(m)
    return (max(int(xs.min()) - CROP_MARGIN_PX, 0), max(int(ys.min()) - CROP_MARGIN_PX, 0),
            min(int(xs.max()) + CROP_MARGIN_PX + 1, W), min(int(ys.max()) + CROP_MARGIN_PX + 1, H))


# ------------------------------------------------------------ capture

def capture_dent_set(s, long_preset, short_preset, label: str = "dentset",
                     zone: Optional[str] = None, lines_v: bool = True, metric: bool = True,
                     metric_preset=None, steps=None, periods=None,
                     pattern_set: str = "production") -> DentSet:
    """One pose: long pass, short pass (two rig sessions) + one metric scan (same zone)."""
    from ...calibrate import flow, stereo
    safe_label(label)
    if pattern_set not in ds_mod.PATTERN_SETS:
        raise ValueError(f"pattern_set must be one of {ds_mod.PATTERN_SETS}, got {pattern_set!r}")
    metric_preset = metric_preset or long_preset
    fp = config.geometry_fingerprint(long_preset)
    for other, what in ((short_preset, "short-exposure preset"), (metric_preset, "metric preset")):
        rules.require_same_geometry(other.geometry_id, long_preset.geometry_id,
                                    f"capture_dent_set ({what})")
        rules.require_same_fingerprint(config.geometry_fingerprint(other), fp,
                                       f"capture_dent_set ({what})")
    if float(long_preset.camera.exposure_us) <= float(short_preset.camera.exposure_us):
        raise ValueError(f"the long pass ({long_preset.camera.exposure_us} us) must expose longer "
                         f"than the short pass ({short_preset.camera.exposure_us} us)")
    diffs = radiometric_diffs(long_preset, short_preset)
    if diffs:
        raise ValueError(f"the short-exposure preset {short_preset.name} differs from "
                         f"{long_preset.name} in more than the exposure ({'; '.join(diffs)}): the "
                         "HDR composite scales the short pass by the exposure ratio only - use "
                         "two presets that differ only in camera.exposure_us")
    cal = None
    if metric:                           # rule 6, before the rig is touched
        cal = stereo.require(s.store, long_preset.geometry_id, fp, "capture_dent_set (metric scan)")
    proj_mask = cam_mask = None
    cells = []
    if zone:
        from ...capture import zones as zones_mod
        proj_mask, cam_mask, cells = zones_mod.resolve(s, long_preset, zone)
    warnings = []
    if zone and cam_mask is None:
        warnings.append("no projector map for this geometry: the frames are cropped to where the "
                        "projector lights the scene, not to the zone (run capture_map)")
    specs = {"long": ds_mod.production_specs("long", pattern_set=pattern_set),
             "short": ds_mod.production_specs("short", lines_v=lines_v, pattern_set=pattern_set)}
    frames, secs = {}, {}
    for which, p in (("long", long_preset), ("short", short_preset)):
        t0 = time.perf_counter()
        pats = [ds_mod.pattern_image(q) for q in specs[which]]
        frames[which] = flow.capture_patterns(s, p, pats, f"the dent set ({which} exposure)",
                                              zone_mask=proj_mask)
        secs[which] = round(time.perf_counter() - t0, 2)
    L = frames["long"]
    i_on = _flood_index(specs["long"], 0)
    i_off = _flood_index(specs["long"], 255)
    shape = np.asarray(L[0]).shape
    x0, y0, x1, y1 = _crop_box(cam_mask, L[i_on], L[i_off], shape)
    stacks = {w: np.stack([np.asarray(f, np.float32)[y0:y1, x0:x1] for f in frames[w]])
              for w in frames}
    # the metric scan runs last; if it fails the captured pattern frames are still saved
    # (no re-capture of the pose), with metric_scans [] and a warning
    ms = interrupted = None
    metric_error = None
    if metric:
        from ...reconstruct import metric_scan
        try:
            ms = metric_scan.scan_metric(s, metric_preset, f"{label}_metric", steps=steps,
                                         periods=periods, zone=zone)
        except Exception as e:           # noqa: BLE001 - keep the frames, report the failure
            metric_error = f"{type(e).__name__}: {e}"
        except KeyboardInterrupt as e:   # save the frames, then stop as asked
            interrupted, metric_error = e, "interrupted"
        if metric_error:
            warnings.append(f"the metric scan failed ({metric_error[:300]}); the pattern frames "
                            "are saved WITHOUT one: scan_metric the same zone without moving the "
                            "cell and pass it to detect_dents / calibrate_dent_null as "
                            "metric_scans")
    d = s.store.new_dir("dentsets", long_preset.geometry_id, label=label)
    with s.store.filling(d):
        _save_npy(d / "frames_long.npy", stacks["long"])
        _save_npy(d / "frames_short.npy", stacks["short"])
        if cam_mask is not None:
            _save_npy(d / "cam_zone_mask.npy", np.asarray(cam_mask, bool)[y0:y1, x0:x1])
        if proj_mask is not None:
            _save_npy(d / "dmd_zone_mask.npy", np.asarray(proj_mask, bool))
        img = stacks["short"][_flood_index(specs["short"], 0)]
        s.store.write_preview(d / "preview.png", img, None)
        created = _now()
        passes = []
        for which, p in (("long", long_preset), ("short", short_preset)):
            passes.append({"name": which, "preset": p.name, "exposure_us": p.camera.exposure_us,
                           "average": p.camera.average, "gain_db": p.camera.gain_db,
                           "duty": p.projection.duty, "gamma": p.projection.gamma,
                           "file": f"frames_{which}.npy",
                           "crop_x0y0x1y1": [x0, y0, x1, y1], "capture_s": secs[which],
                           "patterns": specs[which]})
        meta = {"kind": "dentset", "format": ds_mod.FORMAT, "id": d.name, "label": label,
                "geometry_id": long_preset.geometry_id, "geometry_fp": fp, "created": created,
                "zone": zone, "zone_cells": [list(map(int, c)) for c in cells] if zone else None,
                "crop_x0y0x1y1": [x0, y0, x1, y1], "mock": s.mock, "passes": passes,
                "pattern_set": pattern_set + ("" if lines_v else "_nolines"),
                "dmd_coords": "sinusoids: X = column, Y = row / 2 (diamond rows)",
                "floods": "levels as SENT; which lands bright (projection.invert) is measured "
                          "from the floods at load time",
                "metric_scans": [ms.id] if ms is not None else [],
                "metric_error": metric_error,
                "calibration_id": cal.id if cal is not None else None,
                "units": "camera DN (averaged frames), crop offset x0, y0 in camera px",
                "warnings": warnings}
        s.store.write_json(d / "meta.json", json_safe(meta))
    if interrupted is not None:
        raise interrupted
    return DentSet(id=d.name, label=label, geometry_id=long_preset.geometry_id, path=d,
                   created=created, zone=zone, metric_scan_ids=meta["metric_scans"], meta=meta,
                   warnings=warnings)


# ------------------------------------------------------------ listings

def _handle(path: Path) -> DentSet:
    m = ds_mod.read_meta(path)
    legacy = bool(m.get("legacy"))
    return DentSet(id=m.get("id") or Path(path).name, label=m.get("label") or Path(path).name,
                   geometry_id=m.get("geometry_id") or "", path=Path(path),
                   created=m.get("created", ""), zone=m.get("zone"),
                   metric_scan_ids=list(m.get("metric_scans") or []), legacy=legacy, meta=m)


def list_dentsets(store, geometry_id: Optional[str] = None) -> list:
    geos = [geometry_id] if geometry_id else store.geometry_ids("dentsets")
    out = []
    for g in geos:
        for d in store.entries("dentsets", g):
            m = read_meta(d)
            if m is None or m.get("kind") != "dentset":
                continue
            out.append(_handle(d))
    return out


def as_dentset(s, dentset) -> DentSet:
    """DentSet handle, a folder path (engine entry or legacy dataset.json), or an id."""
    if isinstance(dentset, DentSet):
        return dentset
    p = Path(str(dentset))
    if p.is_dir():
        return _handle(p)
    for d in list_dentsets(s.store):
        if d.id == str(dentset) or d.label == str(dentset):
            return d
    raise ValueError(f"no dent set {dentset!r} (a dentset id / label or a folder with "
                     "meta.json or dataset.json)")


def as_metric_scan(s, scan) -> MetricScan:
    if isinstance(scan, MetricScan):
        return scan
    p = Path(str(scan))
    if p.is_dir():
        m = read_meta(p)
        if m is None or "geometry_id" not in m:
            raise ValueError(f"{p} is not a metric scan (no meta.json)")
        return MetricScan(id=p.name, label=m.get("label", p.name), geometry_id=m["geometry_id"],
                          path=p, created=m.get("created", ""),
                          calibration_id=m.get("calibration_id", ""), meta=m)
    from ...reconstruct import metric_scan
    for ms in metric_scan.list_metric(s.store):
        if ms.id == str(scan):
            return ms
    raise ValueError(f"no metric scan {scan!r} in the library")


def _metric_for(s, dset: DentSet, metric_scans) -> tuple:
    """-> (metric scans, stereo calibration, geometry_id, fingerprint, warnings). Rules 5 + 6."""
    from ...reconstruct import metric_scan
    warnings = []
    if metric_scans:
        scans = [as_metric_scan(s, m) for m in metric_scans]
    elif dset.metric_scan_ids:
        scans = [as_metric_scan(s, i) for i in dset.metric_scan_ids]
    else:
        raise ValueError(f"dent set {dset.id} names no metric scan: pass metric_scans (the metric "
                         "scans of the same pose)")
    geo_id = scans[0].geometry_id
    fp = (scans[0].meta or {}).get("geometry_fp")
    cal = None
    for ms in scans:
        rules.require_same_geometry(ms.geometry_id, geo_id, "dent detection (metric scans)")
        rules.require_same_fingerprint((ms.meta or {}).get("geometry_fp"), fp,
                                       "dent detection (metric scans)")
        c = metric_scan.check_cal(s, ms, "dent detection")          # rule 6: mm or nothing
        if cal is not None and c.id != cal.id:
            raise rules.GeometryMismatch("the metric scans were triangulated with different "
                                         "stereo calibrations")
        cal = c
    if dset.geometry_id:
        rules.require_same_geometry(dset.geometry_id, geo_id,
                                    "dent detection (dent set vs metric scan)")
        rules.require_same_fingerprint(dset.meta.get("geometry_fp"), fp,
                                       "dent detection (dent set vs metric scan)")
    else:
        warnings.append(f"legacy dent set {dset.id}: its geometry is taken from its metric scans "
                        f"({geo_id})")
    return scans, cal, geo_id, fp, warnings


def _fixture(dset: DentSet) -> Optional[str]:
    return dset.zone or dset.meta.get("zone")


def _geometry_summary(an: dict) -> dict:
    g = an["geo"]
    return {"axis": g["axis"].tolist(), "axis_point_mm": g["centre"].tolist(),
            "radius_mm": g["radius_mm"], "fit_rms_mm": g["fit_rms_mm"], "n_points": g["n_points"],
            "s_range_mm": list(g["s_range"]), "face_px": int(g["face"].sum()),
            "mm_per_px": g["mm_per_px"], "crop_xy": [an["data"].x0, an["data"].y0],
            "units": "mm (camera frame), px"}


# ------------------------------------------------------------ null calibration

def null_calibrations(store, geometry_id: Optional[str] = None) -> list:
    geos = [geometry_id] if geometry_id else store.geometry_ids("calibrations")
    out = []
    for g in geos:
        for d in store.entries("calibrations", g):
            f = d / NULL_FILE
            if not f.is_file():
                continue
            try:
                c = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if isinstance(c, dict) and c.get("kind") == "dent_null":
                out.append({**c, "path": d})
    return out


def null_mismatch(c: dict, fp, diameter_mm: Optional[float], fixture: Optional[str],
                  signature: dict, stereo_id: Optional[str]) -> Optional[str]:
    """Why null calibration c does not fit (None = it fits). diameter_mm / fixture /
    stereo_id None = not checked."""
    if rules.fingerprint_diffs(c.get("geometry_fp"), fp):
        return "another geometry fingerprint"
    d = float(c.get("cell_diameter_mm", -1))
    if diameter_mm is not None and abs(d - float(diameter_mm)) > DIAMETER_TOL_MM:
        return f"a {d:g} mm cell"
    if fixture and c.get("fixture") and c["fixture"] != fixture:
        return f"fixture {c['fixture']}"
    if c.get("pattern_signature") != signature:             # what the finders saw
        return "another pattern set"
    if stereo_id and c.get("stereo_calibration_id") != stereo_id:
        return f"stereo calibration {c.get('stereo_calibration_id')} (not the current one)"
    return None


def latest_null(store, geometry_id: str, fp, diameter_mm: Optional[float],
                fixture: Optional[str], signature: dict,
                stereo_id: Optional[str] = None) -> tuple:
    """The newest null calibration matching geometry + fingerprint, cell diameter (None =
    any, but the matching nulls must then agree on one), fixture, pattern signature and
    stereo calibration -> (null or None, warnings about NEWER nulls that were skipped for
    anything but another cell diameter)."""
    ok, newer = [], []
    for c in reversed(null_calibrations(store, geometry_id)):          # newest first
        why = null_mismatch(c, fp, diameter_mm, fixture, signature, stereo_id)
        if why is None:
            ok.append(c)
        elif not ok and not why.endswith("mm cell"):
            newer.append(f"{c.get('id')} ({why})")
    if not ok:
        return None, []
    if diameter_mm is None:
        ds = sorted({round(float(c.get("cell_diameter_mm", -1)), 3) for c in ok})
        if len(ds) > 1:
            raise ValueError("null calibrations for several cell diameters match ("
                             + ", ".join(f"{d:g}" for d in ds) + " mm): pass cell_diameter_mm")
    warns = [f"newer dent null calibration(s) skipped: {'; '.join(newer)}; using "
             f"{ok[0].get('id')}"] if newer else []
    return ok[0], warns


def _load_null(s, null) -> dict:
    if isinstance(null, dict):
        return null
    p = Path(str(null))
    f = p / NULL_FILE if p.is_dir() else p
    if f.is_file():
        c = json.loads(f.read_text(encoding="utf-8"))
        return {**c, "path": f.parent}
    for c in null_calibrations(s.store):
        if c.get("id") == str(null) or Path(c["path"]).name == str(null):
            return c
    raise ValueError(f"no dent null calibration {null!r}")


def calibrate_dent_null(s, dentsets: list, cell_diameter_mm: float, label: str = "null",
                        metric_scans: Optional[list] = None, fixture: Optional[str] = None,
                        subset: str = "all", params: Optional[dict] = None) -> dict:
    """Run the finders on known-good cell(s) of one type in one fixture -> T per finder
    and T_F, stored as calibrations/<geometry>/<ts>_dentnull_<label>/dent_null.json."""
    safe_label(label)
    if not dentsets:
        raise ValueError("calibrate_dent_null needs at least one dent set of a KNOWN-GOOD cell")
    if not (isinstance(cell_diameter_mm, (int, float)) and cell_diameter_mm > 0):
        raise ValueError("cell_diameter_mm must be a positive number of mm")
    if metric_scans is not None and len(dentsets) != 1:
        raise ValueError("metric_scans can only be given with ONE dent set (otherwise each "
                         "dent set's own are used)")
    ans, sources, warnings = [], [], []
    geo_id = fp = cal = None
    sig = None
    fixtures = set()
    for ds in dentsets:
        dset = as_dentset(s, ds)
        scans, c, g, f, w = _metric_for(s, dset, metric_scans)
        warnings += w
        if geo_id is None:
            geo_id, fp, cal = g, f, c
        else:
            rules.require_same_geometry(g, geo_id, "calibrate_dent_null (dent sets)")
            rules.require_same_fingerprint(f, fp, "calibrate_dent_null (dent sets)")
        data = ds_mod.load(dset.path, subset=subset)
        if sig is None:
            sig = data.pattern_signature()
        elif data.pattern_signature() != sig:
            raise ValueError("the null dent sets were captured with different pattern sets")
        an = fusion.analyse(data, [m.path for m in scans], c, float(cell_diameter_mm) / 2, params)
        ans.append(an)
        fixtures.add(_fixture(dset))
        sources.append({"dentset": dset.id, "path": str(dset.path), "legacy": dset.legacy,
                        "metric_scans": [m.id for m in scans], "finder_seconds": an["seconds"],
                        "geometry": _geometry_summary(an)})
    null = fusion.calibrate_null(ans, params)
    warnings += null.pop("warnings", [])
    fixtures.discard(None)
    if fixture is None and len(fixtures) == 1:
        fixture = next(iter(fixtures))
    elif fixture is None and len(fixtures) > 1:
        warnings.append(f"the null dent sets come from different zones {sorted(fixtures)}: the "
                        "calibration is not bound to one fixture")
    d = s.store.new_dir("calibrations", geo_id, label=f"dentnull_{label}")
    rec = {"id": f"{geo_id}/{d.name}", "kind": "dent_null", "label": label, "geometry_id": geo_id,
           "geometry_fp": fp, "cell_diameter_mm": float(cell_diameter_mm), "fixture": fixture,
           "subset": subset, "pattern_signature": sig, "stereo_calibration_id": cal.id,
           **null, "sources": sources, "created": _now(), "mock": s.mock,
           "units": {"T": "each finder's own score units (its maximum on the good face)",
                     "T_F": "weighted sum of calibrated evidence (its maximum >= 3 px inside the "
                            "good face); detect_dents divides by it"},
           "note": "negative control: 1.0 = strongest response on the known-good cell(s); valid "
                   "only for this geometry, cell diameter, fixture, pattern set and stereo "
                   "calibration",
           "warnings": warnings}
    s.store.write_json(d / NULL_FILE, json_safe(rec))
    return {**rec, "path": d}


# ------------------------------------------------------------ detection

def detect_dents(s, dentset, metric_scans=None, cell_diameter_mm: Optional[float] = None,
                 null_calibration=None, label: Optional[str] = None, subset: str = "all",
                 fixture: Optional[str] = None, params: Optional[dict] = None) -> DentReport:
    dset = as_dentset(s, dentset)
    scans, cal, geo_id, fp, warnings = _metric_for(s, dset, metric_scans)
    data = ds_mod.load(dset.path, subset=subset)
    sig = data.pattern_signature()
    fixture = fixture or _fixture(dset)
    if null_calibration is not None:
        null = _load_null(s, null_calibration)
    else:
        null, w = latest_null(s.store, geo_id, fp, cell_diameter_mm, fixture, sig, cal.id)
        warnings += w
        if null is None:
            cell = "" if cell_diameter_mm is None else f" for a {cell_diameter_mm:g} mm cell"
            raise ValueError(
                f"no dent null calibration{cell} at geometry {geo_id}"
                + (f", fixture {fixture}" if fixture else "")
                + f", this pattern set ({subset}) and stereo calibration {cal.id}: "
                "capture a KNOWN-GOOD cell of the same type in the same fixture and run "
                "calibrate_dent_null first")
    if null.get("kind") != "dent_null":
        raise ValueError("that file is not a dent null calibration")
    rules.require_same_geometry(null["geometry_id"], geo_id, "detect_dents (null calibration)")
    rules.require_same_fingerprint(null.get("geometry_fp"), fp, "detect_dents (null calibration)")
    if null.get("stereo_calibration_id") != cal.id:                 # rule 6
        raise rules.GeometryMismatch(
            f"rule 6: the null calibration was made under stereo calibration "
            f"{null.get('stereo_calibration_id')}, the metric scans use {cal.id} (the face, the "
            "projector sensitivity and mm / px came from it): run calibrate_dent_null again on "
            "the good cell")
    if cell_diameter_mm is None:
        cell_diameter_mm = float(null["cell_diameter_mm"])
    elif abs(float(null["cell_diameter_mm"]) - float(cell_diameter_mm)) > DIAMETER_TOL_MM:
        raise ValueError(f"the null calibration is for a {null['cell_diameter_mm']:g} mm cell, not "
                         f"{cell_diameter_mm:g} mm")
    if null.get("pattern_signature") != sig:
        raise ValueError("the null calibration was made with another pattern set: calibrate on a "
                         "good cell captured (or subset) the same way")
    if fixture and null.get("fixture") and null["fixture"] != fixture:
        warnings.append(f"the null calibration was made in fixture {null['fixture']}, this dent "
                        f"set is from {fixture} (an explicitly chosen null: used anyway)")
    an = fusion.analyse(data, [m.path for m in scans], cal, float(cell_diameter_mm) / 2, params)
    det = fusion.detect(an, null, params)
    warnings += det["warnings"]
    cands = det["candidates"]
    n_det = sum(c["level"] == "detection" for c in cands)
    lab = label or f"{dset.label}_dents"
    out = s.store.output_path("defects", lab, ".json")
    source = dset.path if _inside(s.store.root, dset.path) else None
    title = (f"{dset.id} | {n_det} detection(s) | cell {cell_diameter_mm:g} mm | patterns {subset} "
             f"| null {str(null.get('id', '?')).split('/')[-1]}")
    files = {"json": out}
    try:
        png = out.with_suffix(".png")
        rules.ensure_new_path(png)
        with open(png, "xb") as f:
            f.write(overlay.render(an, det, title))
        files["png"] = png
    except Exception as e:                      # noqa: BLE001 - the report must still be written
        warnings.append(f"overlay image failed: {type(e).__name__}: {e}")
    npz = out.with_suffix(".npz")
    s.store.write_npz(npz, fused=det["F"], face=det["face"],
                      **{f"finder_{k}": v for k, v in an["maps"].items()},
                      **{f"evidence_{k}": v.astype(np.float32) for k, v in det["E"].items()},
                      crop_xy=np.array([data.x0, data.y0]))
    files["npz"] = npz
    rep = DentReport(verdict="FLAG" if n_det else "CLEAR", candidates=cands,
                     n_detections=int(n_det), max_score=det["max_score"],
                     units=dict(fusion.UNITS), dentset_id=dset.id,
                     metric_scan_ids=[m.id for m in scans], null_calibration_id=str(null.get("id")),
                     cell_diameter_mm=float(cell_diameter_mm), subset=subset,
                     geometry={**_geometry_summary(an),
                               "depth_noise_p90_mm": det["depth_noise_p90_mm"],
                               "depth_significant_above_mm": det["depth_significant_above_mm"]},
                     finder_seconds=an["seconds"], files=files, warnings=warnings,
                     finders=list(det["finders"]),
                     thresholds={k: det["params"][k] for k in ("thr", "watch", "edge_px",
                                                               "depth_sig_factor")},
                     fixture=fixture)
    keys = ("verdict", "n_detections", "max_score", "units", "dentset_id", "metric_scan_ids",
            "null_calibration_id", "cell_diameter_mm", "subset", "fixture", "finders",
            "thresholds", "geometry", "finder_seconds", "warnings", "candidates")
    doc = {k: getattr(rep, k) for k in keys}
    doc.update({"dentset_path": str(dset.path), "stereo_calibration_id": cal.id,
                "pattern_signature": sig, "null": {k: null.get(k) for k in ("T", "T_F", "weights")},
                "created": _now(), "mock": s.mock})
    s.store.write_json(out, json_safe(doc))
    for f in files.values():
        s.store.record_output("defects", f, label=lab, geometry_id=geo_id, source=source)
    return rep


def _inside(root: Path, p: Path) -> bool:
    try:
        Path(p).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False

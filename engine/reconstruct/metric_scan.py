"""
Metric scans behind the api verbs: capture both axes (multifreq), triangulate
with the stereo calibration of the SAME geometry + fingerprint (rule 6,
checked before the rig is touched), store, and the measurements on top
(cylinder deviation, golden compare, accuracy check, metric mesh).

    metric/<geometry_id>/<ts>_<label>/  field.npz, cloud.npz, preview.png, meta.json (last)

The projector display (rule 6): a capture is made in video display (every scan
that projects through the HDMI window) or in pattern-sequence display (the two
scans a flash pack plays: meta mode flash_trigger / flash_softsync). _save, the
one function that writes mm, stamps the capture's display (meta "display_mode")
and refuses a calibration of the other display before any file exists; check_cal
and triangulate_scan look the calibration up by the stored scan's display, so
re-triangulating never changes which display a scan counts as.

The rig check, the re-home and the rig gate (engine/rigstate) speak for the video
calibration. So a pattern-display calibration counts only while no video calibration
at its geometry is newer (_overtaken_by), and a mobile rig that is verified for one
calibration gives no mm with another (_require_rig).
"""

from __future__ import annotations

import time
from dataclasses import asdict
from datetime import datetime
from typing import Optional

import numpy as np

from .. import config, rules
from ..calibrate import flow, stereo
from ..library.store import json_safe, read_meta, safe_label
from ..measure import metric
from ..results import MetricScan
from .triangulate import MetricCloud, triangulate, write_ply


# meta "mode" of the scans captured in pattern-sequence display (trigger_scan.MODE,
# flash_scan.MODE): how such a scan is known when it was stored before the stamp existed
PATTERN_MODES = ("flash_trigger", "flash_softsync")


def _both_displays(s, geometry_id: str) -> bool:
    """True only on the simulated bench, whose projector has no offset between the two
    displays: a MOCK session's calibration counts for both. Decided from the session and
    from where mock data lives (geometry mock-<id>), never from a record or an argument:
    a real session never gets this answer, and neither does real data opened in a mock
    session."""
    return getattr(s, "mock", False) is True and \
        str(geometry_id).startswith(config.MOCK_PREFIX)


def _overtaken_by(s, cal, geometry_id: str, fp: Optional[dict]):
    """The video calibration that is NEWER than this pattern-display calibration, or None.

    A newer calibration at the geometry (solved from new poses, or a re-home) says the rig
    is not where the older one was made. The rig check and the re-home work on the video
    calibration only, so they never renew a pattern-display one: it counts until a video
    calibration is newer, and is then made again. (Before a calibration carried its
    display, the newest record was the rig's calibration for every scan: the same rule.)"""
    if cal is None or (getattr(cal, "display_mode", None) or "video") != "pattern":
        return None
    vid = stereo.latest(s.store, geometry_id, fp, "video")
    return vid if vid is not None and vid.path.name > cal.path.name else None


def _find_cal(s, geometry_id: str, fp: Optional[dict], display_mode: str = "video"):
    """The stereo calibration a capture in `display_mode` gives mm with, or None: the
    newest at this geometry + fingerprint made in that display (a pattern-display one only
    while no video calibration is newer, _overtaken_by). A video lookup is therefore the
    record stereo.latest() gives every other caller. On the simulated bench
    (_both_displays), and only when there is none of that display: the other display's."""
    rules.check_display(display_mode)
    cal = stereo.latest(s.store, geometry_id, fp, display_mode)
    if _overtaken_by(s, cal, geometry_id, fp) is not None:
        cal = None
    if cal is None and _both_displays(s, geometry_id):
        other = next(d for d in rules.DISPLAYS if d != display_mode)
        cal = stereo.latest(s.store, geometry_id, fp, other)
    return cal


def _require_cal(s, geometry_id: str, fp: Optional[dict], what: str,
                 display_mode: str = "video"):
    """Rule 6, before the rig: the stereo calibration for a capture made in projector
    display `display_mode`, else refuse (GeometryMismatch; DisplayMismatch when the
    display is what is missing, or when the pattern-display calibration is older than
    the rig's newest video calibration)."""
    cal = _find_cal(s, geometry_id, fp, display_mode)
    if cal is not None:
        return cal
    old = stereo.latest(s.store, geometry_id, fp, display_mode)
    newer = _overtaken_by(s, old, geometry_id, fp)
    if newer is not None:
        raise rules.DisplayMismatch(
            f"{what}: the stereo calibration made in pattern-sequence display ({old.id}) is "
            f"older than the rig's calibration {newer.id}: the rig was calibrated again or "
            "re-homed since, and neither renews a calibration of the other display, so mm "
            "are refused (rule 6). Make the pattern-display calibration again, from poses "
            "captured now.")
    # raises, naming what is missing (the mock bench: none of either display)
    return stereo.require(s.store, geometry_id, fp, what,
                          "video" if _both_displays(s, geometry_id) else display_mode)


def _check_display(s, cal, display_mode: str, geometry_id: str, what: str) -> None:
    """Refuse (DisplayMismatch) a calibration made in the other projector display."""
    rules.check_display(display_mode)
    made_in = getattr(cal, "display_mode", None) or "video"
    if made_in != display_mode and not _both_displays(s, geometry_id):
        raise rules.DisplayMismatch(
            f"{what}: the capture was made in {rules.display_text(display_mode)}, the stereo "
            f"calibration {cal.id} in {rules.display_text(made_in)}, so mm are refused "
            "(rule 6): a calibration counts only for the projector display it was made in.")


def _require_rig(s, preset, what: str, scan=None, cal=None) -> Optional[dict]:
    """The rig gate of a capture that gives mm (engine/rigstate.require): a MOBILE rig that
    is not verified in this session is refused (rules.RuleViolation) before the rig is
    touched; a fixed rig never is. -> what the scan records of it (meta "rig": the
    verification it was captured under; None for a fixed rig).
    cal: the calibration the capture is triangulated with. The gate speaks for the video
    calibration of the geometry; a mobile rig verified for that one is not verified for
    another (a pattern-display calibration): refused too."""
    from .. import rigstate
    g = rigstate.require(s, preset, what, scan=scan)
    spoke_for = (g.get("state") or {}).get("calibration_id")
    if cal is not None and g.get("mount") == "mobile" and spoke_for != cal.id:
        made_in = rules.display_text(getattr(cal, "display_mode", None) or "video")
        raise rules.RuleViolation(
            f"{what}: {rigstate.UNVERIFIED} (the rig is verified for calibration {spoke_for}; "
            f"this capture is triangulated with {cal.id}, made in {made_in}, and a "
            "verification counts only for the calibration it was made with)")
    return rigstate.stamp(g)


def scan_metric(s, preset, label: str = "metric", method: str = "auto", steps=None,
                periods=None, max_consistency_mm: Optional[float] = None,
                zone: Optional[str] = None) -> MetricScan:
    """zone: grid cells ("F11:F12", "atlas:cell/cell") - light only those, with a black
    surround (less stray light on the part), and keep only the points inside them."""
    safe_label(label)
    fp = config.geometry_fingerprint(preset)
    cal = _require_cal(s, preset.geometry_id, fp, "scan_metric")       # before the rig
    rig = _require_rig(s, preset, "scan_metric", cal=cal)              # a mobile rig: verified
    basis = flow.multifreq_basis(preset, cal.diamond_parity, steps, periods)
    t0 = time.perf_counter()
    proj_mask = cam_mask = cells = None
    if zone:
        from ..capture import zones as zones_mod
        proj_mask, cam_mask, cells = zones_mod.resolve(s, preset, zone)
    frames = flow.capture_patterns(s, preset, basis.generate(), "a metric scan", zone_mask=proj_mask)
    fld = basis.decode(frames, preset.geometry_id, contrast_min=preset.decode.contrast_min,
                       conf_min=preset.decode.conf_min)
    capture_s = round(time.perf_counter() - t0, 2)
    if cam_mask is not None:
        fld.mask = fld.mask & np.asarray(cam_mask, bool)
    cloud = triangulate(fld, cal, method, max_consistency_mm)
    fld.meta.update({"preset": preset.name, "label": label, "mock": s.mock})
    return _save(s, fld, cloud, label, cal, fp, preset.name,
                 {"capture_s": capture_s, "steps": basis.steps, "periods_px": basis.periods,
                  "diamond_parity": cal.diamond_parity, "zone": zone, "rig": rig})


def _save(s, fld, cloud: MetricCloud, label: str, cal, fp, preset_name, extra: dict,
          arrays: Optional[dict] = None, display_mode: str = "video") -> MetricScan:
    """arrays: {file name: {array name: ndarray}} extra .npz files of the entry (e.g. the
    flood pair of a fast scan), written before meta.json.
    display_mode: the projector display the frames were captured in (rules.DISPLAYS),
    stamped on the entry. A calibration made in the other display is refused
    (rules.DisplayMismatch) before the entry's folder exists: no file is left."""
    _check_display(s, cal, display_mode, fld.geometry_id, f"metric scan {label!r}")
    d = s.store.new_dir("metric", fld.geometry_id, label=label)
    with s.store.filling(d):
        s.store.write_field(d / "field.npz", fld)
        with open(d / "cloud.npz", "xb") as f:
            cloud.save_npz(f)
        for name, arrs in (arrays or {}).items():
            s.store.write_npz(d / name, **arrs)
        z = np.where(cloud.mask, cloud.xyz[..., 2], np.nan)
        s.store.write_preview(d / "preview.png", -z, cloud.mask)
        created = datetime.now().isoformat(timespec="seconds")
        meta = {"id": d.name, "label": label, "geometry_id": fld.geometry_id, "geometry_fp": fp,
                "created": created, "calibration_id": cal.id, "method": cloud.method,
                "units": "mm", "n_points": int(cloud.mask.sum()),
                "valid_pct": round(100.0 * float(cloud.mask.mean()), 2),
                "consistency": cloud.consistency_stats(), "preset": preset_name,
                "mock": s.mock, "decode": fld.meta, **extra, "display_mode": display_mode}
        s.store.write_json(d / "meta.json", json_safe(meta))
    cloud.meta = meta
    ply = s.store.output_path("pointclouds", label, ".ply")
    write_ply(ply, cloud)
    s.store.record_output("pointclouds", ply, label=label, geometry_id=fld.geometry_id, source=d)
    return MetricScan(id=d.name, label=label, geometry_id=fld.geometry_id, path=d,
                      created=created, calibration_id=cal.id, meta=meta, _cloud=cloud,
                      _field=fld)


def _require_rig_of(s, scan, what: str, cal=None) -> Optional[dict]:
    """_require_rig for a STORED capture (no preset in hand): the preset it was captured
    with says how the rig is mounted. A capture of a mobile rig gives mm only when it was
    made in the session the rig is verified in now. A capture whose preset is gone is
    refused when any preset at its geometry says mobile (it can't be shown to be covered).
    cal: the calibration it is triangulated with (_require_rig)."""
    from .. import rigstate
    name = (scan.meta or {}).get("preset")
    preset = None
    try:
        if name:
            preset = config.load_preset(name, presets_dir=s.presets_dir, store=s.store,
                                        mock=s.mock)
    except config.ConfigError:
        preset = None
    if preset is not None and preset.geometry_id == scan.geometry_id:
        return _require_rig(s, preset, what, scan=scan, cal=cal)
    geo = scan.geometry_id
    if s.mock and geo.startswith(config.MOCK_PREFIX):
        geo = geo[len(config.MOCK_PREFIX):]
    builtin = s.presets_dir or config.PRESETS_DIR
    by = rigstate._mobile_presets(s.store, builtin).get(geo)
    if by:
        raise rules.RuleViolation(
            f"{what}: {rigstate.UNVERIFIED} (preset {by[0]} declares the rig at geometry {geo} "
            f"mobile, and the preset {name or '?'} this capture was made with can't be loaded: "
            "it can't be shown to come from a verified session - scan again)")
    return None


def list_metric(store, geometry_id: Optional[str] = None) -> list:
    geos = [geometry_id] if geometry_id else store.geometry_ids("metric")
    out = []
    for g in geos:
        for d in store.entries("metric", g):
            m = read_meta(d)
            if m is None or "geometry_id" not in m:
                continue
            out.append(MetricScan(id=d.name, label=m.get("label", d.name),
                                  geometry_id=m["geometry_id"], path=d,
                                  created=m.get("created", ""),
                                  calibration_id=m.get("calibration_id", ""), meta=m))
    return out


def _source_meta(s, scan_id: str, geometry_id: str) -> Optional[dict]:
    """meta.json of the stored scan a metric entry was triangulated from (a metric entry
    at the same geometry, or an object scan), or None when it is not in the library."""
    for parts in (("metric", geometry_id), ("objects",)):
        try:
            dirs = s.store.entries(*parts)
        except ValueError:
            continue
        for d in dirs:
            if d.name == scan_id:
                return read_meta(d)
    return None


def scan_display(s, scan) -> str:
    """The projector display a stored scan was captured in: "video" or "pattern".

    - its stamp (meta "display_mode", written by _save);
    - pattern, whatever else it says, when its mode is one of PATTERN_MODES or its decode
      block has the "softsync" key the one-bit flash decode writes;
    - an entry without a stamp that was triangulated from another scan (meta
      "source_scan"; such entries were stored without their mode) is its source's display,
      and is REFUSED (rules.DisplayMismatch) when the source is not in the library any
      more: its display cannot be shown;
    - else video (every other scan stored before the stamp existed projected through the
      HDMI window)."""
    meta, sid, seen = scan.meta or {}, str(getattr(scan, "id", "?")), set()
    first = sid
    while True:
        stamp = meta.get("display_mode")
        if stamp is not None and stamp not in rules.DISPLAYS:
            raise rules.DisplayMismatch(
                f"scan {sid}: its record names an unknown projector display ({stamp!r}), so "
                "mm are refused (rule 6): scan again")
        dec = meta.get("decode")
        if stamp == "pattern" or meta.get("mode") in PATTERN_MODES or \
                (isinstance(dec, dict) and "softsync" in dec):
            return "pattern"
        src = meta.get("source_scan")
        if stamp == "video" or not src:
            return "video"
        seen.add(sid)
        nxt = None if str(src) in seen else _source_meta(s, str(src), scan.geometry_id)
        if nxt is None:
            raise rules.DisplayMismatch(
                f"scan {first} was triangulated from {src}, which is not in the library any "
                "more: it cannot be shown which projector display it was captured in, so mm "
                "are refused (rule 6). Scan again")
        meta, sid = nxt, str(src)


def check_cal(s, scan, what: str):
    """Rule 6 at use time: the calibration that made a stored metric scan must
    still be the one in the library for its geometry (a trashed or replaced
    calibration stops the mm), and it must have been made in the projector
    display the scan was captured in."""
    cal = _require_cal(s, scan.geometry_id, (scan.meta or {}).get("geometry_fp"), what,
                       display_mode=scan_display(s, scan))
    if isinstance(scan, MetricScan) and cal.id != scan.calibration_id:
        raise rules.GeometryMismatch(
            f"{what}: scan {scan.id} was triangulated with calibration {scan.calibration_id}, but "
            f"the current one for {scan.geometry_id} is {cal.id}; re-triangulate it "
            "(triangulate) or re-scan")
    return cal


def triangulate_scan(s, scan, method: str = "auto", label: Optional[str] = None) -> dict:
    """An existing scan (ObjectScan from capture_object, or a MetricScan) ->
    a new metric cloud with the current stereo calibration + .ply. The calibration is the
    one of the display the scan was captured in, and the new entry carries that display."""
    fp = (scan.meta or {}).get("geometry_fp")
    display = scan_display(s, scan)
    cal = _require_cal(s, scan.geometry_id, fp, "triangulate", display_mode=display)
    rig = _require_rig_of(s, scan, "triangulate", cal=cal)
    fld = scan.field
    warnings = []
    basis = (scan.meta or {}).get("basis_names") or fld.basis_names
    if cal.diamond_parity is not None and "multifreq" not in (basis or []):
        warnings.append("this scan's columns were decoded without the diamond row parity "
                        "(gray code + phase): expect a +-0.25 column zigzag; use scan_metric")
    cloud = triangulate(fld, cal, method)
    ms = _save(s, fld, cloud, label or f"{scan.label}_tri", cal, fp,
               (scan.meta or {}).get("preset"), {"source_scan": scan.id, "warnings": warnings,
                                                 "rig": rig}, display_mode=display)
    return {"scan": ms, "calibration_id": cal.id, "method": cloud.method,
            "n_points": int(cloud.mask.sum()), "consistency": cloud.consistency_stats(),
            "units": "mm", "warnings": warnings}


# ------------------------------------------------------------- outputs

def _deviation_png(path, res: np.ndarray, mask: np.ndarray, lim_mm: float):
    import cv2
    x = np.clip(np.nan_to_num(res) / max(lim_mm, 1e-9), -1, 1)
    img8 = ((x + 1) * 127.5).astype(np.uint8)
    col = cv2.applyColorMap(img8, cv2.COLORMAP_JET)
    col[~mask] = 0
    rules.ensure_new_path(path)
    ok, buf = cv2.imencode(".png", col)
    if ok:
        with open(path, "xb") as f:
            f.write(buf.tobytes())
        return path
    return None


def cylinder_deviation(s, scan: MetricScan, radius_mm=None, n_sigma: float = 4.0,
                       min_depth_mm: float = 0.05, axis_hint=(0.0, 1.0, 0.0), roi=None):
    check_cal(s, scan, "cylinder_deviation")
    cloud = scan.cloud
    region = None
    if roi is not None:
        x0, y0, x1, y1 = (int(v) for v in roi)
        region = np.zeros(cloud.shape, bool)
        region[max(y0, 0):y1, max(x0, 0):x1] = True
        region, _ = metric.foreground(cloud, region=region)
    dev = metric.cylinder_deviation(cloud, region, axis_hint=axis_hint, radius_mm=radius_mm,
                                    n_sigma=n_sigma, min_depth_mm=min_depth_mm)
    rep = {"scan": scan.id, "calibration_id": scan.calibration_id, "units": "mm",
           "diameter_mm": dev.fit.diameter, "radius_mm": dev.fit.radius,
           "axis": dev.fit.axis.tolist(), "axis_point_mm": dev.fit.point.tolist(),
           "fit_rms_mm": dev.fit.rms, "noise_sigma_mm": dev.noise_sigma_mm,
           "threshold_mm": dev.threshold_mm, "arc_deg": dev.arc_deg,
           "dents": [asdict(d) for d in dev.dents], "warnings": dev.warnings}
    out = s.store.output_path("defects", f"{scan.label}_cyldev", ".json")
    s.store.write_json(out, json_safe(rep))
    s.store.record_output("defects", out, label=scan.label, geometry_id=scan.geometry_id,
                          source=scan.path)
    lim = max(3 * dev.threshold_mm, max((d.depth_mm for d in dev.dents), default=0.0))
    png = _deviation_png(out.with_suffix(".png"), dev.residual, dev.mask, lim)
    if png:
        s.store.record_output("defects", png, label=scan.label, geometry_id=scan.geometry_id,
                              source=scan.path)
    npz = out.with_suffix(".npz")
    s.store.write_npz(npz, residual_mm=dev.residual.astype(np.float32), mask=dev.mask)
    return dev, {"json": out, "png": png, "npz": npz}


def _masked(cloud: MetricCloud, keep: np.ndarray) -> MetricCloud:
    from dataclasses import replace
    m = cloud.mask & keep
    return replace(cloud, mask=m, xyz=np.where(m[..., None], cloud.xyz, np.nan))


def compare_metric(s, scan: MetricScan, golden: MetricScan, max_dist_mm: float = 5.0,
                   n_sigma: float = 4.0, min_depth_mm: float = 0.05, min_area_mm2: float = 0.5,
                   foreground_only: bool = True):
    """ICP-align scan onto golden (point-to-plane), then the signed deviation map
    (mm, negative = below the golden surface) and its negative blobs. Informational, no
    verdict: with foreground_only only each scan's LARGEST foreground blob is kept, and only
    the test -> golden direction is measured (points beyond max_dist_mm are NaN), so missing
    material and a second part or component are not reported. A verdict comes from a station
    metric job with measure: "golden" (every component, both directions)."""
    from scipy import ndimage
    check_cal(s, scan, "compare_metric")
    check_cal(s, golden, "compare_metric (golden)")
    rules.require_same_geometry(scan.geometry_id, golden.geometry_id, "compare_metric")
    a, g = scan.cloud, golden.cloud
    if foreground_only:
        # the part, not the wall: a re-placed part moves against the background,
        # so one rigid ICP over both would split the difference
        a, g = _masked(a, metric.foreground(a)[0]), _masked(g, metric.foreground(g)[0])
    icp = metric.icp_point_to_plane(a.points, g.points, max_dist_mm=max_dist_mm)
    dev = metric.deviation_to(a, g, icp, max_dist_mm)
    m = np.isfinite(dev)
    r = dev[m]
    sigma = 1.4826 * float(np.median(np.abs(r - np.median(r)))) if r.size else float("nan")
    thr = max(n_sigma * sigma, min_depth_mm)
    lab, n = ndimage.label(m & (np.nan_to_num(dev) < -thr))
    area = metric.pixel_area_mm2(np.where(a.mask[..., None], a.xyz, np.nan))
    blobs = []
    for k, sl in enumerate(ndimage.find_objects(lab), 1):
        b = lab[sl] == k
        ar = float(np.nansum(area[sl][b]))
        if ar >= min_area_mm2:
            blobs.append({"depth_mm": round(float(-np.nanmin(dev[sl][b])), 4),
                          "area_mm2": round(ar, 3), "n_px": int(b.sum()),
                          "bbox_px": [sl[1].start, sl[0].start, sl[1].stop, sl[0].stop]})
    blobs.sort(key=lambda d: -d["depth_mm"])
    rep = {"scan": scan.id, "golden": golden.id, "units": "mm", "foreground_only": foreground_only,
           "icp": {"rms_mm": icp.rms_mm, "fitness": icp.fitness, "iterations": icp.iterations,
                   "converged": icp.converged, "R": icp.R.tolist(), "t": icp.t.tolist()},
           "noise_sigma_mm": sigma, "threshold_mm": thr, "blobs": blobs,
           "max_below_mm": float(-np.nanmin(dev)) if r.size else None}
    out = s.store.output_path("comparisons", f"{scan.label}_vs_{golden.label}_metric", ".json")
    s.store.write_json(out, json_safe(rep))
    s.store.record_output("comparisons", out, label=scan.label, geometry_id=scan.geometry_id,
                          source=scan.path)
    return rep, dev


def accuracy_check(s, preset, kind: str, known: dict, scans: Optional[list] = None,
                   repeats: int = 3, tol_mm: float = 0.1, repeat_tol_mm: float = 0.03,
                   label: str = "accuracy", steps=None, periods=None) -> dict:
    """Known reference object -> per-repeat measured values, mean / max error,
    repeatability; GO when |mean error| <= tol and repeat sigma <= repeat_tol."""
    if kind not in metric.SHAPES:
        raise ValueError(f"kind must be one of {metric.SHAPES}")
    if scans is None:
        if repeats < 1:
            raise ValueError("repeats must be >= 1")
        scans = [scan_metric(s, preset, f"{label}_{i + 1}", steps=steps, periods=periods)
                 for i in range(int(repeats))]
    rows = []
    for sc in scans:
        check_cal(s, sc, "accuracy_check")
        rows.append({"scan": sc.id, **metric.measure_known(sc.cloud, kind, known)})
    summ = metric.summarize(rows, tol_mm, repeat_tol_mm)
    rep = {"kind": kind, "known": known, **summ, "rows": rows,
           "geometry_id": scans[0].geometry_id, "calibration_id": scans[0].calibration_id}
    out = s.store.output_path("measurements", f"{label}_{kind}", ".json")
    s.store.write_json(out, json_safe(rep))
    s.store.record_output("measurements", out, label=label, geometry_id=rep["geometry_id"])
    rep["file"] = out
    return rep


def accuracy_block(s, preset, block, measured=None, scans: Optional[list] = None,
                   repeats: int = 1, tol_height_mm=None, tol_diameter_mm=None,
                   flatness_max_mm=None, label: str = "block", steps=None, periods=None) -> dict:
    """The check block (measure/checkblock.py) over new metric scans (or the given ones)
    -> the report (rows, verdict, disposition) in output/measurements."""
    from ..measure import checkblock
    spec = checkblock.load(block)
    meas = checkblock.load_measured(measured, spec)        # bad input refused before the rig
    if scans is None:
        if repeats < 1:
            raise ValueError("repeats must be >= 1")
        scans = [scan_metric(s, preset, f"{label}_{i + 1}", steps=steps, periods=periods)
                 for i in range(int(repeats))]
    for sc in scans:
        check_cal(s, sc, "accuracy_block")
    rep = checkblock.analyze([sc.cloud for sc in scans], spec, meas, tol_height_mm,
                             tol_diameter_mm, flatness_max_mm)
    rep.update(scans=[sc.id for sc in scans], geometry_id=scans[0].geometry_id,
               calibration_id=scans[0].calibration_id, measured_given=sorted(meas))
    out = s.store.output_path("measurements", f"{label}_checkblock", ".json")
    s.store.write_json(out, json_safe(rep))
    s.store.record_output("measurements", out, label=label, geometry_id=rep["geometry_id"])
    rep["file"] = out
    return rep


def export_metric_mesh(s, scan: MetricScan, fmt: str = "stl", stride: int = 1,
                       max_edge_mm: Optional[float] = None, foreground_only: bool = False):
    from . import mesh as mesh_mod
    check_cal(s, scan, "export_mesh")
    region = metric.foreground(scan.cloud)[0] if foreground_only else None
    out = s.store.output_path("pointclouds", f"{scan.label}_metric", f".{fmt}")
    p, mp, meta = mesh_mod.cloud_mesh(scan.cloud, out, fmt, stride, max_edge_mm, region,
                                      {"scan": scan.id, "label": scan.label, "mock": s.mock})
    s.store.record_output("pointclouds", p, label=scan.label, geometry_id=scan.geometry_id,
                          source=scan.path)
    return p, mp, meta

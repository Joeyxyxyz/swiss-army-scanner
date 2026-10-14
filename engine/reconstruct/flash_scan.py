"""
Flash soft-sync fast scan behind the api verbs (docs/FLASH_SOFTSYNC.md).

The DLPC350 plays the fast-scan pack's 16 one-bit patterns from ONE 24-bit
flash image on its internal timer, on repeat. A HUMAN loads the pack with the
firmware link of the EKB 4500 MKII GUI and plays the sequence in its Pattern
Sequence tab. This module only opens the CAMERA: it free-runs faster than
the pattern rate (ROI, fixed exposure, OldestFirst buffers), the frames are
sorted by time and content (capture/softsync.py) and the sorted images go
through the unchanged multifreq decode and 'plane' triangulation into a
normal metric scan (meta mode 'flash_softsync').

    fast_scan_flash()        record -> sort -> decode -> triangulate -> metric scan
    sort_flash_recording()   offline: a saved recording -> per-pattern images (+ decode)

Rule 1: nothing here imports or sends a DLPC350 USB / HID command; the
projector window (HDMI) is never opened either. Rule 6: the stereo
calibration is checked before the camera is touched.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from typing import Optional

import numpy as np

from .. import config, rules
from ..capture import softsync
from ..library.store import json_safe, safe_label
from . import metric_scan
from .triangulate import triangulate

MODE = "flash_softsync"
METHODS = ("plane", "auto")          # one coded axis: the column plane (auto = plane)
MIN_EXPECTED_CLEAN = 1.5             # refuse a plan with fewer expected clean frames / pattern
PROCESS_ESTIMATE_S = {"configure": 0.15, "sort_roi": 0.5, "sort_full": 1.5, "decode": 0.3,
                      "save": 0.3}   # bench-typical, for the dry-run estimate only


# ---------------------------------------------------------------- the loop

def load_layout(pack=None, variant: Optional[str] = None) -> tuple:
    """(FlashLayout, manifest or None, pack dir or None, warnings)."""
    warnings = []
    if variant is not None and variant not in ("full", "zone"):
        raise ValueError(f"variant must be 'full' or 'zone', got {variant!r}")
    if pack is None:
        lay = softsync.default_layout(variant)
        warnings.append("no pack given: the fast-scan pack defaults are assumed (16 one-bit "
                        f"slots, {lay.pattern_period_us:g} us per slot); pass the pack folder "
                        "the GUI plays so its timing is checked")
        return lay, None, None, warnings
    lay, manifest, pdir = softsync.load_pack_layout(pack, variant)
    return lay, manifest, pdir, warnings


def _camera_zone(s, preset, zone: Optional[str], roi, layout, pack_dir, full_shape,
                 manifest: Optional[dict] = None) -> tuple:
    """-> (camera mask or None, zone label, DMD zone mask for the mock or None, warnings)."""
    warnings = []
    H, W = full_shape
    dmd_zone = (softsync.pack_zone_mask(pack_dir, manifest) if layout.variant == "zone"
                else None)
    if layout.variant == "zone" and pack_dir is not None and dmd_zone is None:
        raise ValueError(f"pack {pack_dir} has no zone_mask.png for the zone variant")
    if roi is not None:
        x0, y0, x1, y1 = (int(v) for v in roi)
        if not (0 <= x0 < x1 <= W and 0 <= y0 < y1 <= H):
            raise ValueError(f"roi {list(roi)} is not inside the {W}x{H} camera frame")
        m = np.zeros((H, W), bool)
        m[y0:y1, x0:x1] = True
        return m, f"roi:{x0},{y0},{x1},{y1}", dmd_zone, warnings
    if zone:
        from ..capture import zones as zones_mod
        _, cam, _ = zones_mod.resolve(s, preset, zone)
        if cam is None:
            warnings.append(f"zone {zone}: no projector map, so the camera window stays full "
                            "frame and the points are not cut to the zone (run capture_map)")
        return (None if cam is None else np.asarray(cam, bool)), zone, dmd_zone, warnings
    if dmd_zone is not None:
        from ..decode import projmap
        pmap = projmap.latest(s, preset.geometry_id)
        if pmap is not None:
            cam = pmap.camera_mask_for_projector_mask(dmd_zone)
            if cam.any():
                import cv2
                cam = cv2.dilate(cam.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
                return cam, "pack:zone", dmd_zone, warnings
        warnings.append("zone variant without a projector map: the camera records the full "
                        "frame (slower, ~1.5 s sorting); capture_map (or roi / zone) gives a "
                        "small window and a faster frame rate")
    return None, None, dmd_zone, warnings


@contextmanager
def freerun_camera(s, preset, layout, dmd_zone=None, pack_dir=None):
    """The free-running camera, and nothing else of the rig.

    mock: the simulated human has loaded the pack and pressed Play (mock_flashloop;
    set `api.mock_bench().flash_gui = {...}` to simulate a GUI mistake, e.g.
    {"running": False}). real: the rig lock, then the Blackfly S opened by the
    engine's RealCamera and switched to free-run soft-sync settings
    (PySpinFreeRunCamera); on the way out the HDMI scan settings are restored and
    everything is released. The projector (HDMI window, USB) is never opened."""
    rules.check_scan_path_usb_free()
    if s.mock:
        from ..capture import mock_flashloop
        bench = s.bench
        seed = int(bench._rng.integers(0, 2**31 - 1))
        pats = (mock_flashloop.pack_patterns(pack_dir, layout.variant)
                if pack_dir is not None else None)
        cam = mock_flashloop.mock_camera(bench, layout, zone_mask=dmd_zone,
                                         gui=getattr(bench, "flash_gui", None),
                                         camera=getattr(bench, "flash_camera", None), seed=seed,
                                         patterns=pats)
        yield cam
        return
    from ..capture.camera import open_camera
    from ..capture.riglock import RigLock
    from ..session import _release_rig, default_holder
    lock = RigLock(holder=getattr(s, "holder", "") or default_holder()).acquire()
    real = fr = None
    primary = None
    try:
        real = open_camera(preset.camera, mock=False)
        fr = softsync.PySpinFreeRunCamera(real._drv._cam)
        yield fr
    except BaseException as e:
        primary = e
        raise
    finally:
        if fr is not None:
            try:
                fr.restore()
            except Exception:           # noqa: BLE001 - the close below resets the camera
                pass
        _release_rig(real, None, lock, primary)


FINE_PREFILTER_MAX_PX = 3.0


def _set_fine_prefilter(layout, px: Optional[float]) -> None:
    """The optional fine-level camera low-pass (onebit FINE_PREFILTER_PX = 1 px is the
    pack's recommendation for a sharp projector; None keeps the layout's, off)."""
    if px is None:
        return
    px = float(px)
    if not 0.0 <= px <= FINE_PREFILTER_MAX_PX:
        raise ValueError(f"fine_prefilter_px must be 0..{FINE_PREFILTER_MAX_PX:g} camera px, "
                         f"got {px:g}")
    layout.fine_prefilter_px = px or None


def _plan(layout, roi, exposure_us, fps, loops) -> dict:
    return softsync.plan_timing(layout.n, roi, exposure_us=exposure_us or layout.camera_exposure_us,
                                fps=fps, pattern_exposure_us=layout.pattern_exposure_us,
                                pattern_period_us=layout.pattern_period_us, loops=loops,
                                guard_us=layout.guard_us)


def _check_plan(plan: dict) -> None:
    if plan["dlp_timing_errors"]:
        raise ValueError("the pack's pattern timing is not legal for 1-bit patterns: "
                         + "; ".join(plan["dlp_timing_errors"]))
    if plan["clean_frames_per_pattern"] < MIN_EXPECTED_CLEAN:
        raise ValueError(
            f"only {plan['clean_frames_per_pattern']:.1f} clean frames per pattern expected "
            f"(pattern exposure {plan['pattern_exposure_us']:g} us, camera exposure "
            f"{plan['camera_exposure_us']:g} us at {plan['fps']:g} fps, guard "
            f"{plan['guard_us']:g} us): shorten the camera exposure, use a smaller camera window "
            "(roi / zone) for a faster frame rate, or record more loops")


def _estimate(plan: dict) -> dict:
    H, W = softsync.SENSOR_SHAPE
    full = plan["roi"][2] * plan["roi"][3] > 0.5 * H * W
    e = PROCESS_ESTIMATE_S
    parts = {"configure_s": e["configure"], "capture_s": round(plan["capture_ms"] / 1e3, 3),
             "sort_s": e["sort_full"] if full else e["sort_roi"], "decode_s": e["decode"],
             "save_s": e["save"]}
    parts["total_s"] = round(sum(parts.values()), 2)
    return parts


# ---------------------------------------------------------------- the verb

def fast_scan_flash(s, preset, pack=None, variant: Optional[str] = None, label: str = "flash",
                    zone: Optional[str] = None, roi=None, loops: Optional[int] = None,
                    exposure_us: Optional[float] = None, fps: Optional[float] = None,
                    gain_db: Optional[float] = None, method: str = "plane",
                    keep_recording: bool = True, dry_run: bool = False,
                    fine_prefilter_px: Optional[float] = None) -> dict:
    safe_label(label)
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS} (one coded axis: the column plane)")
    rules.check_scan_path_usb_free()
    layout, manifest, pack_dir, warnings = load_layout(pack, variant)
    _set_fine_prefilter(layout, fine_prefilter_px)
    pack_geo = (manifest or {}).get("geometry_id")
    if pack_geo and pack_geo != preset.geometry_id:
        warnings.append(f"the pack was made for geometry {pack_geo!r}, the preset is "
                        f"{preset.geometry_id!r}: its zone image lights the zones of that rig "
                        "position (rebuild the pack's zones at this geometry)")
    fp = config.geometry_fingerprint(preset)
    cal = rig = None
    if not dry_run:
        cal = metric_scan._require_cal(s, preset.geometry_id, fp, "fast_scan_flash",   # rule 6
                                       display_mode="pattern")
        rig = metric_scan._require_rig(s, preset, "fast_scan_flash", cal=cal)   # mobile: verified
    full_shape = tuple(s.bench.cam_shape) if s.mock else softsync.SENSOR_SHAPE
    cam_mask, zone_label, dmd_zone, w2 = _camera_zone(s, preset, zone, roi, layout, pack_dir,
                                                      full_shape, manifest)
    warnings += w2
    roi_t = softsync.roi_from_mask(cam_mask) if cam_mask is not None else None
    plan = _plan(layout, roi_t, exposure_us, fps, loops)
    _check_plan(plan)
    if cal is not None and layout.diamond_parity != cal.diamond_parity:
        warnings.append(f"the pack's fringes are drawn for diamond parity {layout.diamond_parity}, "
                        f"the stereo calibration measured {cal.diamond_parity}: expect a +-0.25 "
                        "column zigzag (rebuild the pack with this calibration)")
    pack_info = {"pack": str(pack_dir) if pack_dir is not None else None,
                 "pack_name": (manifest or {}).get("name"), "variant": layout.variant,
                 "layout_source": layout.source, "n_patterns": layout.n,
                 "pattern_exposure_us": layout.pattern_exposure_us,
                 "pattern_period_us": layout.pattern_period_us,
                 "fine_prefilter_px": layout.fine_prefilter_px,
                 "led": (manifest or {}).get("led")}
    if dry_run:
        return {"dry_run": True, "mode": MODE, "plan": plan, "estimate_s": _estimate(plan),
                "layout": pack_info, "zone": zone_label, "warnings": warnings,
                "calibration": (lambda c: c.id if c is not None else None)(
                    metric_scan._find_cal(s, preset.geometry_id, fp, "pattern"))}
    gain = float(preset.camera.gain_db if gain_db is None else gain_db)
    times = {}
    t_all = time.perf_counter()
    with freerun_camera(s, preset, layout, dmd_zone, pack_dir) as cam:
        t0 = time.perf_counter()
        if not s.mock and roi_t is not None:            # the camera's own ROI increments
            inc = cam.increments()
            x, y, w, h = roi_t
            plan = _plan(layout, softsync.align_roi((x, y, x + w, y + h), step=inc, margin=0),
                         exposure_us, fps, loops)
            _check_plan(plan)
        meta = {"layout": layout.to_json(), "plan": plan, "label": label, "mock": s.mock,
                "geometry_id": preset.geometry_id, "preset": preset.name, **pack_info}
        rec = softsync.FlashFreeRunSync(cam).record(
            plan["n_frames"], plan["roi"], plan["camera_exposure_us"], plan["fps"], gain_db=gain,
            full_shape=full_shape, meta=meta, min_duration_s=plan["capture_ms"] / 1e3)
        times["record_s"] = round(time.perf_counter() - t0, 3)
    rec_path = None
    if keep_recording:
        rec_path = s.store.output_path("flashscans", label, ".npz")
        rec.save(rec_path)
        s.store.record_output("flashscans", rec_path, label=label, geometry_id=preset.geometry_id)
    t0 = time.perf_counter()
    try:
        res = softsync.sort_recording(rec, layout, lit_mask=cam_mask)
    except softsync.SequenceError as e:
        if rec_path is not None and hasattr(e, "add_note"):
            e.add_note(f"recording kept for offline sorting: {rec_path}")
        raise
    times["sort_s"] = round(time.perf_counter() - t0, 3)
    t0 = time.perf_counter()
    fld = softsync.decode_sorted(res, layout, preset.geometry_id,
                                 contrast_min=preset.decode.contrast_min,
                                 conf_min=preset.decode.conf_min)
    if cam_mask is not None:
        fld.mask = fld.mask & cam_mask
    times["decode_s"] = round(time.perf_counter() - t0, 3)
    t0 = time.perf_counter()
    cloud = triangulate(fld, cal, "plane")
    fld.meta.update({"preset": preset.name, "label": label, "mock": s.mock})
    r = res.report
    frame_stats = {"n_frames": r["n_frames"], "frames_dropped": r["frames_dropped"],
                   "fps_measured": r["fps_measured"], "status_counts": r["status_counts"],
                   "per_pattern": r["counts"], "loops_seen": r["loops_seen"],
                   "min_count": r["min_count"], "confidence": res.confidence,
                   "signal_p99_dn": (r.get("checks") or {}).get("white_minus_black_p99_dn"),
                   "zone_edge": fld.meta.get("zone_edge"),
                   "prefilter_edge": fld.meta.get("prefilter_edge")}
    flash = {**pack_info, "variant_detected": r.get("variant_detected"),
             "loop_ms": r["loop_period_ms"], "pattern_period_ms": r["pattern_period_ms"],
             "camera": {"roi": list(rec.roi), "exposure_us": rec.exposure_us, "fps": rec.fps,
                        "gain_db": gain, "applied": rec.meta.get("camera")},
             "frame_stats": frame_stats, "timing_fit": r.get("timing_fit"),
             "anchor": r.get("anchor"), "checks": r.get("checks"),
             "recording": str(rec_path) if rec_path else None, "plan": plan,
             "sort_warnings": r.get("warnings", [])}
    ms = metric_scan._save(s, fld, cloud, label, cal, fp, preset.name, {
        "mode": MODE, "capture_s": times["record_s"], "steps": layout.steps,
        "periods_px": list(layout.periods), "diamond_parity": layout.diamond_parity,
        "zone": zone_label, "fast": True, "flash": json_safe(flash), "rig": rig},
        display_mode="pattern")
    times["triangulate_save_s"] = round(time.perf_counter() - t0, 3)
    times["total_s"] = round(time.perf_counter() - t_all, 3)
    # the entry's own first: a fixed rig in doubt (the rig check's words), then the capture's
    warnings = list(dict.fromkeys([*(ms.meta.get("warnings") or []), *warnings,
                                   *r.get("warnings", [])]))
    return {"scan": ms, "id": ms.id, "mode": MODE, "n_points": int(cloud.mask.sum()),
            "valid_pct": ms.meta.get("valid_pct"), "calibration_id": cal.id,
            "confidence": res.confidence, "frame_stats": frame_stats,
            "loop_ms": r["loop_period_ms"], "variant_detected": r.get("variant_detected"),
            "plan": plan, "times_s": times, "recording": rec_path, "zone": zone_label,
            "warnings": warnings, "rig": rig}


# ------------------------------------------------------------- offline sort

def sort_flash_recording(s, recording, pack=None, variant: Optional[str] = None,
                         decode: bool = False, label: Optional[str] = None,
                         geometry_id: Optional[str] = None,
                         fine_prefilter_px: Optional[float] = None) -> dict:
    """A saved recording -> per-pattern images (+ report) in output/flashscans/; decode:
    also the projector-column map (no mm; fine_prefilter_px: the optional fine-level
    low-pass). Nothing touches the rig."""
    from pathlib import Path
    rec_path = Path(recording)
    if not rec_path.is_file():
        raise ValueError(f"no recording {recording}")
    rec = softsync.Recording.load(rec_path)
    layout = softsync.recording_layout(rec, pack, variant)
    _set_fine_prefilter(layout, fine_prefilter_px)
    t0 = time.perf_counter()
    res = softsync.sort_recording(rec, layout)
    sort_s = round(time.perf_counter() - t0, 3)
    lab = label or f"{(rec.meta or {}).get('label') or rec_path.stem}_sorted"
    safe_label(lab)
    geo = geometry_id or (rec.meta or {}).get("geometry_id") or ""
    arrays = {"images": np.stack(res.images).astype(np.float32),
              "names": np.array(layout.names()), "counts": np.asarray(res.counts, np.int64),
              "labels": np.asarray(res.labels, np.int64), "status": np.asarray(res.status),
              "roi": np.asarray(res.roi, np.int64),
              "report_json": np.array(json.dumps(json_safe(res.report)))}
    fld = None
    if decode:
        fld = softsync.decode_sorted(res, layout, geo or "offline")
        arrays["corr_x"] = fld.corr_x
        arrays["mask"] = fld.mask
    out = s.store.output_path("flashscans", lab, ".npz")
    with open(out, "xb") as f:
        np.savez_compressed(f, **arrays)
    s.store.record_output("flashscans", out, label=lab, geometry_id=geo, source=None)
    r = res.report
    doc = {"sorted": out, "recording": rec_path, "confidence": res.confidence,
           "loop_ms": r["loop_period_ms"], "pattern_period_ms": r["pattern_period_ms"],
           "status_counts": r["status_counts"], "counts": r["counts"],
           "variant_detected": r.get("variant_detected"), "sort_s": sort_s,
           "anchor": r.get("anchor"), "warnings": r.get("warnings", []),
           "layout_source": layout.source}
    if fld is not None:
        doc["decoded_px"] = int(fld.mask.sum())
        doc["fine_prefilter_px"] = fld.meta.get("fine_prefilter_px")
        doc["zone_edge"] = fld.meta.get("zone_edge")
        doc["corr_x_median"] = (float(np.nanmedian(fld.corr_x[fld.mask])) if fld.mask.any()
                                else None)
    return doc


def expected_times(periods_us=(10000, 5000, 2500), exposures_us=(3000, 1500, 750),
                   rois=((1440, 1080), (480, 352)), loops=None) -> list:
    """The timing table of docs/FLASH_SOFTSYNC.md: plan_timing over pattern period P,
    camera exposure E and camera window (full frame / a zone window) -> rows."""
    rows = []
    for P, E in zip(periods_us, exposures_us):
        for w, h in rois:
            try:
                p = softsync.plan_timing(16, (0, 0, w, h), exposure_us=E, pattern_period_us=P,
                                         pattern_exposure_us=P, loops=loops,
                                         guard_us=softsync.FASTSCAN_GUARD_US if P >= 5000
                                         else softsync.DEFAULT_GUARD_US)
            except ValueError as e:
                rows.append({"P_us": P, "E_us": E, "roi": [w, h], "error": str(e)})
                continue
            rows.append({"P_us": P, "E_us": E, "roi": [w, h], "fps": p["fps"],
                         "fps_limit": p["fps_limit"], "loops": p["loops"],
                         "clean_per_pattern": p["clean_frames_per_pattern"],
                         "capture_ms": p["capture_ms"], "estimate_s": _estimate(p)})
    return rows

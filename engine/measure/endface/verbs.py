"""
End-face verbs behind engine/api.py: locate a face, aim / check the ring, set the intensity,
the targeted scan, the profile, the golden and the check.

Library (append-only, rule 7; meta.json written LAST = the completeness marker):
  faces/<geometry_id>/<ts>_<label>/        face.npz (camera face mask crop, rim px, camera
                                           circle, DMD ring), preview.png, meta.json
  facescans/<geometry_id>/<ts>_<label>/    masks.npz (DMD face mask, camera face mask crop),
                                           polar_frames_r<k>.npy (float32 camera DN, crop),
                                           preview.png, meta.json (names the repeat metric scans,
                                           each a normal metric/<geometry_id>/ entry)
  face_golden/<geometry_id>/<ts>_<label>/  golden.npz (grid; MF: radial profile, 2-D median, the
                                           reflectance FIR's 49 coefficients; band: golden, band,
                                           sigma; the placements' height maps), preview.png,
                                           meta.json (params: the default detector + every
                                           setting, leave-one-out, synthetic set)
Output bin: previews/ (locate overlay, ring photo + overlay + json), tuning/ (auto_intensity),
profiles/ (face PNG, radial + section CSV, JSON, NPZ: height map + per-pixel repeat data + the
flood pair), pointclouds/ (face STL + meta), defects/ (check_face JSON, overlay PNG, maps NPZ).

Rules: every mm is from the stereo calibration the metric scans were triangulated with, which
must still be the library's current one for the geometry (rule 6: checked before the rig
opens and before any analysis; a golden made under another calibration is refused). Face
targets, face scans, profiles and goldens carry the geometry_id + fingerprint and are only
combined when both match (rule 5). The rig is driven only through session.rig /
flow.capture_patterns (HDMI, pumped holds; the LED currents are never touched).
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
from ...results import FaceCheck, FaceGolden, FaceProfile, FaceScan, FaceTarget, MetricScan, RingShot
from . import detectors as det_mod
from . import golden as golden_mod
from . import locate as locate_mod
from . import mfcfar
from . import patterns
from . import profile as profile_mod

CROP_MARGIN_PX = 24                 # camera crop around the face (masks, polar frames)
DIAMETER_TOL_MM = 0.01
RING_TOL_PX = 1.5                   # check_ring: centre offset still ON_FACE (camera px)
RING_SIZE_TOL = 0.15
RING_MIN_LIT_PCT = 90.0             # check_ring: the face flood lights this much of the face
RING_MAX_SPILL_PCT = 10.0           # ... and at most this much outside it
TARGET_DN = 200.0
MIN_DUTY = 0.05
SATURATED_DN = 254.0
REPEAT_WARN_MM = 0.05
DEPTH_MODES = ("multifreq", "tilephase")    # targeted_face_scan depth sets (tilephase: experimental)
UNITS_FACE = {"centre_mm": "mm, camera frame (x right, y down, z forward)",
              "normal": "unit vector toward the camera", "camera": "camera px (full frame)",
              "dmd": "native DMD px (col, row)", "tilt_deg": "deg", "rim_rms_mm": "mm"}
UNITS_MAP = {"u_mm": "mm, face frame ~ camera right", "v_mm": "mm, face frame ~ camera down",
             "h_mm": "mm, + toward the camera (a dent is negative)", "theta_deg":
             "deg, 0 = camera right, 90 = camera down", "area_mm2": "mm^2"}


# ------------------------------------------------------------ helpers

def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _save_npy(path: Path, arr: np.ndarray) -> Path:
    rules.ensure_new_path(path)
    with open(path, "xb") as f:
        np.save(f, arr)
    return path


def _write_bytes(path: Path, data: Optional[bytes]) -> Optional[Path]:
    if not data:
        return None
    rules.ensure_new_path(path)
    with open(path, "xb") as f:
        f.write(data)
    return path


def _write_text(path: Path, text: str) -> Path:
    rules.ensure_new_path(path)
    with open(path, "x", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return path


def _crop_box(mask: np.ndarray, margin: int = CROP_MARGIN_PX) -> tuple:
    H, W = mask.shape
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError("empty face mask")
    return (max(int(xs.min()) - margin, 0), max(int(ys.min()) - margin, 0),
            min(int(xs.max()) + margin + 1, W), min(int(ys.max()) + margin + 1, H))


def as_metric_scan(s, scan) -> MetricScan:
    from ..dents.verbs import as_metric_scan as _as
    return _as(s, scan)


def _scans_and_cal(s, scans: list, what: str) -> tuple:
    """Metric scans of one geometry + fingerprint, triangulated with the CURRENT stereo
    calibration (rules 5 + 6). -> (scans, calibration, geometry_id, fingerprint)."""
    from ...reconstruct import metric_scan
    if not scans:
        raise ValueError(f"{what}: no metric scans")
    out = [as_metric_scan(s, m) for m in scans]
    geo, fp = out[0].geometry_id, (out[0].meta or {}).get("geometry_fp")
    cal = None
    for ms in out:
        rules.require_same_geometry(ms.geometry_id, geo, f"{what} (metric scans)")
        rules.require_same_fingerprint((ms.meta or {}).get("geometry_fp"), fp, f"{what} (metric scans)")
        c = metric_scan.check_cal(s, ms, what)
        if cal is not None and c.id != cal.id:
            raise rules.GeometryMismatch(f"{what}: the metric scans were triangulated with different "
                                         "stereo calibrations")
        cal = c
    return out, cal, geo, fp


# ------------------------------------------------------------ face targets

def _face_handle(d: Path) -> Optional[FaceTarget]:
    m = read_meta(d)
    if m is None or m.get("kind") != "face_target":
        return None
    return FaceTarget(id=m["id"], label=m.get("label", d.name), geometry_id=m["geometry_id"],
                      path=Path(d), created=m.get("created", ""),
                      face_diameter_mm=float(m["face_diameter_mm"]),
                      calibration_id=m.get("calibration_id", ""), source_scan=m.get("source_scan", ""),
                      centre_mm=m["centre_mm"], normal=m["normal"], tilt_deg=m["tilt_deg"],
                      rim_rms_mm=m["rim_rms_mm"], n_rim_points=m["n_rim_points"],
                      camera_ellipse=m["camera_ellipse"], dmd_centre=m["dmd_centre"],
                      dmd_ellipse=m["dmd_ellipse"], affine=m["affine"],
                      mm_per_camera_px=m.get("mm_per_camera_px"), units=m.get("units") or {},
                      meta=m, warnings=list(m.get("warnings") or []))


def list_faces(store, geometry_id: Optional[str] = None) -> list:
    geos = [geometry_id] if geometry_id else store.geometry_ids("faces")
    out = []
    for g in geos:
        for d in store.entries("faces", g):
            h = _face_handle(d)
            if h is not None:
                out.append(h)
    return out


def as_face(s, face, preset=None, what: str = "end face") -> FaceTarget:
    """A FaceTarget handle, an id / label / folder, or None = the newest face target at the
    preset's geometry + fingerprint. With a preset, the target must match it (rule 5)."""
    if isinstance(face, FaceTarget):
        t = face
    elif face is None:
        if preset is None:
            raise ValueError(f"{what}: no face target given (and no preset to find the newest)")
        fp = config.geometry_fingerprint(preset)
        cands = [f for f in list_faces(s.store, preset.geometry_id)
                 if not rules.fingerprint_diffs(f.meta.get("geometry_fp"), fp)]
        if not cands:
            raise ValueError(f"{what}: no face target at geometry {preset.geometry_id}: run "
                             "locate_face on a metric scan of the face first")
        t = cands[-1]
    else:
        p = Path(str(face))
        t = _face_handle(p) if p.is_dir() else None
        if t is None:
            hits = [f for f in list_faces(s.store) if f.id == str(face) or f.label == str(face)]
            if not hits:
                raise ValueError(f"no face target {face!r} in the library (see list_library('faces'))")
            t = hits[-1]
    if preset is not None:
        rules.require_same_geometry(t.geometry_id, preset.geometry_id, f"{what}: face target {t.id}")
        rules.require_same_fingerprint(t.meta.get("geometry_fp"), config.geometry_fingerprint(preset),
                                       f"{what}: face target {t.id}")
    return t


def face_arrays(t: FaceTarget) -> dict:
    with np.load(Path(t.path) / "face.npz") as z:
        return {k: z[k] for k in z.files}


def face_cam_mask(t: FaceTarget, shape=None) -> tuple:
    """-> (full-frame bool mask, crop box, crop mask)."""
    a = face_arrays(t)
    x0, y0, x1, y1 = (int(v) for v in a["crop"])
    H, W = shape or tuple(int(v) for v in a["cam_shape"])
    full = np.zeros((H, W), bool)
    full[y0:y1, x0:x1] = a["cam_mask"].astype(bool)
    return full, (x0, y0, x1, y1), a["cam_mask"].astype(bool)


def _locate_png(xyz, box, r: dict) -> Optional[bytes]:
    try:
        import cv2
    except ImportError:
        return None
    x0, y0, x1, y1 = box
    Z = xyz[y0:y1, x0:x1, 2]
    ok = np.isfinite(Z)
    if not ok.any():
        return None
    lo, hi = np.percentile(Z[ok], [1, 99])
    g8 = (np.clip((np.nan_to_num(Z, nan=hi) - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
    img = cv2.cvtColor(255 - g8, cv2.COLOR_GRAY2BGR)
    img[~ok] = (40, 0, 40)
    sc = 4
    img = cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_NEAREST)
    circ = (r["circle_cam_px"] - [x0, y0]) * sc
    cv2.polylines(img, [np.round(circ).astype(np.int32).reshape(-1, 1, 2)], True, (0, 220, 0), 1)
    for p in r["rim_px"]:
        cv2.circle(img, (int((p[0] - x0) * sc + sc / 2), int((p[1] - y0) * sc + sc / 2)), 1,
                   (0, 0, 255), -1)
    e = r["cam_ellipse"]
    cv2.drawMarker(img, (int((e["cx"] - x0) * sc), int((e["cy"] - y0) * sc)), (0, 220, 0),
                   cv2.MARKER_CROSS, 12, 1)
    cv2.putText(img, f"face D {r['face_diameter_mm']:g} mm, rim rms {r['rim_rms_mm']:.3f} mm, "
                     f"tilt {r['tilt_deg']:.1f} deg", (6, 18), 0, 0.5, (255, 255, 255), 1)
    cv2.putText(img, "red: rim points (far background), green: fitted circle", (6, 36), 0, 0.45,
                (255, 255, 255), 1)
    ok2, buf = cv2.imencode(".png", img)
    return buf.tobytes() if ok2 else None


def locate_face(s, scan, face_diameter_mm: float = locate_mod.FACE_DIAMETER_MM,
                label: str = "face", roi=None, params: Optional[dict] = None,
                flood=None) -> FaceTarget:
    """Metric scan of the face (both axes decoded) -> face target in faces/<geometry>/.
    flood: for the flood-circle fallback (None = the scan's own flood pair, else its decode
    modulation; or an array / file / another metric scan, as station_measure takes it)."""
    safe_label(label)
    known = set(locate_mod.LocateParams().to_dict())
    bad = sorted(set(params or {}) - known)
    if bad:
        raise ValueError(f"locate_face: unknown params {bad}; known: {sorted(known)}")
    prm = locate_mod.LocateParams(**(params or {}))
    [ms], cal, geo, fp = _scans_and_cal(s, [scan], "locate_face")          # rule 6
    fld, cloud = ms.field, ms.cloud
    if fld.corr_x is None or fld.corr_y is None:
        raise ValueError(f"locate_face: metric scan {ms.id} has only one decoded axis; the camera -> "
                         "DMD map needs both: rescan with `scan-fast --axis xy` (or scan_metric, "
                         "which decodes x and y)")
    flood_d, flood_src, fwarn = None, None, []
    if prm.flood_fallback:
        from ..station import verbs as station_verbs
        flood_d, flood_src, fwarn = station_verbs.resolve_flood(s, ms, flood, geo, fp)
    r = locate_mod.locate(cloud.xyz, cloud.mask, fld.corr_x, fld.corr_y, fld.mask, cal.K_cam,
                          cal.dist_cam, float(face_diameter_mm), roi, prm, flood_d)
    warnings = list(r["warnings"])
    if r["rim_source"] == "flood outline":
        warnings += [f"flood: {w}" for w in fwarn]
    cons = (ms.meta or {}).get("consistency") or {}
    if (ms.meta or {}).get("method") == "rays" and (cons.get("median_mm") or 0) > 1.0:
        warnings.append(f"the locate scan was triangulated ray-ray, but its row and column rays miss "
                        f"by {cons['median_mm']:.1f} mm (median): the depths average in the projector "
                        "ROW axis. Prefer scan_metric(method='plane') for the locate scan")
    off = patterns.off_dmd_pct(r["dmd_ellipse"])
    if off > 0.5:
        warnings.append(f"{off:.0f}% of the face maps outside the DMD: the projector cannot light "
                        "that part (never measured); move the part toward the middle of the zone")
    if r["rejected_components"]:
        warnings.append(f"{len(r['rejected_components'])} nearer non-face surface(s) set aside "
                        f"(e.g. {r['rejected_components'][0]['why']} at z "
                        f"{r['rejected_components'][0]['z_near_mm']} mm)")
    box = _crop_box(r["cam_mask"])
    x0, y0, x1, y1 = box
    created = _now()
    d = s.store.new_dir("faces", geo, label=label)
    with s.store.filling(d):
        s.store.write_npz(d / "face.npz", cam_mask=r["cam_mask"][y0:y1, x0:x1].astype(np.uint8),
                          crop=np.array(box), cam_shape=np.array(r["cam_mask"].shape),
                          circle_cam_px=r["circle_cam_px"], dmd_ring=r["dmd_ring"],
                          rim_px=r["rim_px"], region=np.array(r["region"]),
                          depth_face=r["depth_face"].astype(np.uint8))
        png = _locate_png(cloud.xyz, r["region"], r)
        _write_bytes(d / "preview.png", png)
        meta = {"kind": "face_target", "id": d.name, "label": label, "geometry_id": geo,
                "geometry_fp": fp, "created": created, "calibration_id": cal.id,
                "source_scan": ms.id, "source_method": (ms.meta or {}).get("method"),
                "face_diameter_mm": float(face_diameter_mm), "radius_mm": r["radius_mm"],
                "centre_mm": r["centre_mm"], "normal": r["normal"], "e1": r["e1"], "e2": r["e2"],
                "tilt_deg": r["tilt_deg"], "rim_rms_mm": r["rim_rms_mm"],
                "n_rim_points": r["n_rim"], "n_contour_points": r["n_contour"],
                "rim_arc_deg": r["rim_arc_deg"], "z_near_mm": r["z_near_mm"],
                "rim_source": r["rim_source"],
                "flood_source": flood_src if r["rim_source"] == "flood outline" else None,
                "flood_outline_diameter_mm": r.get("flood_outline_diameter_mm"),
                "depth_face_px": r["depth_face_px"], "search_region_px": list(r["region"]),
                "rejected_components": r["rejected_components"],
                "camera_ellipse": r["cam_ellipse"], "camera_crop_x0y0x1y1": list(box),
                "face_cam_px": int(r["cam_mask"].sum()), "mm_per_camera_px": r["mm_per_camera_px"],
                "affine": r["affine"], "dmd_centre": r["dmd_centre"], "dmd_ellipse": r["dmd_ellipse"],
                "params": r["params"], "units": UNITS_FACE, "mock": s.mock, "warnings": warnings}
        s.store.write_json(d / "meta.json", json_safe(meta))
    if png:
        out = _write_bytes(s.store.output_path("previews", f"{label}_locate", ".png"), png)
        s.store.record_output("previews", out, label=label, geometry_id=geo, source=d)
    return _face_handle(d)


# ------------------------------------------------------------ aim / check ring

def _shoot_polarity(sync, warnings: list) -> tuple:
    """Raw black vs raw white (measured, never assumed) -> (inverted, dark frame, lit frame)."""
    from ...capture import zones
    shape = patterns.DMD_SHAPE
    f0 = sync.shoot(np.zeros(shape, np.uint8), raw=True)
    f1 = sync.shoot(np.full(shape, 255, np.uint8), raw=True)
    diff = float(f1.mean()) - float(f0.mean())
    if abs(diff) < zones.POLARITY_MARGIN:
        warnings.append(f"raw white and raw black look the same to the camera ({diff:+.1f} counts): "
                        "is the projector on and aimed at the scene?")
        return False, f0, f1
    inv = diff < 0
    if inv:
        warnings.append("raw HDMI path measured INVERTED; the ring was sent inverted so it is "
                        "physically bright-on-dark")
    return inv, (f1 if inv else f0), (f0 if inv else f1)


def _ring_overlay(photo, t: FaceTarget, meas: Optional[dict], box) -> Optional[bytes]:
    try:
        import cv2
    except ImportError:
        return None
    x0, y0, x1, y1 = box
    f = np.asarray(photo, np.float32)[y0:y1, x0:x1]
    hi = max(float(np.percentile(f, 99.8)), 1.0)
    vis = cv2.cvtColor(np.clip(f * (255.0 / hi), 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    sc = 4
    vis = cv2.resize(vis, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
    e = t.camera_ellipse
    cv2.ellipse(vis, (((e["cx"] - x0) * sc, (e["cy"] - y0) * sc), (e["a"] * sc, e["b"] * sc),
                      e["angle_deg"]), (0, 200, 255), 1)
    if meas and meas.get("ring_ellipse"):
        m = meas["ring_ellipse"]
        cv2.ellipse(vis, (((m["cx"] - x0) * sc, (m["cy"] - y0) * sc), (m["a"] * sc, m["b"] * sc),
                          m["angle_deg"]), (0, 220, 0), 1)
    if meas and meas.get("cross_centre_px"):
        cx, cy = meas["cross_centre_px"]
        cv2.drawMarker(vis, (int((cx - x0) * sc), int((cy - y0) * sc)), (0, 220, 0),
                       cv2.MARKER_CROSS, 10, 1)
    cv2.drawMarker(vis, (int((e["cx"] - x0) * sc), int((e["cy"] - y0) * sc)), (0, 200, 255),
                   cv2.MARKER_TILTED_CROSS, 10, 1)
    cv2.putText(vis, "orange: located face  green: ring / cross as photographed", (4, 14), 0, 0.4,
                (255, 255, 255), 1)
    ok, buf = cv2.imencode(".png", vis)
    return buf.tobytes() if ok else None


def project_ring(s, preset, face=None, mode: str = "aim", hold_s: float = 0.0,
                 label: Optional[str] = None, offset_dmd=None, tol_px: float = RING_TOL_PX,
                 cancel=None) -> RingShot:
    """Ring + cross for the face target -> photo (+ check: where it landed); optional pumped
    hold (the operator's placement guide), cancellable via `cancel` (threading.Event)."""
    from ...capture import zones
    from ...decode import projmap
    if mode not in ("aim", "check"):
        raise ValueError("mode must be 'aim' or 'check'")
    label = label or ("aim" if mode == "aim" else "ring_check")
    safe_label(label)
    t = as_face(s, face, preset, f"{mode}_ring")
    hold = zones._hold(hold_s, preset, "end-face ring hold")          # rule 3 before the rig
    zones._check_cancel(cancel)
    off = [0.0, 0.0] if offset_dmd is None else [float(v) for v in offset_dmd]
    if len(off) != 2:
        raise ValueError("offset_dmd must be [cols, rows]")
    arr = face_arrays(t)
    img = patterns.ring_image(arr["dmd_ring"], t.dmd_centre, off)
    warnings, held, flood = [], 0.0, None
    # the offset in mm uses the face target's mm per camera px (from its metric scan): only
    # while that scan's stereo calibration is still the current one here (rule 6)
    from ...calibrate import stereo
    cur = stereo.latest(s.store, preset.geometry_id, config.geometry_fingerprint(preset))
    mm_ok = cur is not None and cur.id == t.calibration_id
    if mode == "check" and not mm_ok:
        warnings.append(f"face target {t.id} was located under stereo calibration "
                        f"{t.calibration_id or '?'}, the current one is "
                        f"{cur.id if cur else 'none'}: the offset is reported in camera px only")
    with s.rig(preset) as sync:
        inverted, dark, _lit = _shoot_polarity(sync, warnings)
        if mode == "check":
            fl = patterns.face_mask(t.dmd_ellipse, 0.0).astype(np.uint8) * 255
            flood = sync.shoot((255 - fl) if inverted else fl, raw=True)
        photo = sync.shoot((255 - img) if inverted else img, raw=True)
        if hold > 0:
            held = zones._pumped_hold(sync.projector.window, hold, cancel)
            if held < hold - 1e-9:
                warnings.append(f"hold cancelled after {held:.2f} of {hold:g} s")
    full = face_cam_mask(t, photo.shape)[0]
    box = _crop_box(full, 3 * CROP_MARGIN_PX)
    meas, verdict, off_px, off_mm, ratio, lit = {}, None, None, None, None, {}
    if mode == "check":
        lit = patterns.face_lit(flood, dark, full)
        meas = patterns.measure_ring(photo, dark, t.camera_ellipse)
        meas["face_lit"] = lit
        if not lit.get("found") or not meas.get("found"):
            verdict = "NOT_FOUND"
            warnings.append(f"face / ring not found: {lit.get('why') or meas.get('why', '?')}")
        else:
            off_px = meas["centre_offset_px"]
            ratio = meas.get("size_ratio", 1.0)
            dist = float(np.hypot(*off_px))
            off_mm = dist * t.mm_per_camera_px if t.mm_per_camera_px and mm_ok else None
            why = []
            if lit["lit_pct"] < RING_MIN_LIT_PCT:
                why.append(f"only {lit['lit_pct']:.0f}% of the located face is lit by the face flood")
            if lit["spill_pct"] > RING_MAX_SPILL_PCT:
                why.append(f"{lit['spill_pct']:.0f}% of the face flood lands outside the located face")
            if dist > tol_px or abs(ratio - 1) > RING_SIZE_TOL:
                why.append(f"the {meas.get('centre_from', 'ring')} landed {dist:.1f} camera px "
                           f"from the located centre (ring size ratio {ratio:.2f})")
            verdict = "OFF" if why else "ON_FACE"
            if why:
                warnings.append("; ".join(why) + ": the part moved since locate_face (or the camera "
                                "-> DMD map is off) - re-locate before scanning")
    photo_p = projmap.write_png(s.store.output_path("previews", f"{label}_photo", ".png"),
                                   zones._photo_u8(photo))
    over_p = _write_bytes(s.store.output_path("previews", f"{label}_overlay", ".png"),
                          _ring_overlay(photo, t, meas, box))
    shot = RingShot(face_id=t.id, mode=mode, photo=photo_p, overlay=over_p,
                    dmd_centre=[float(v) for v in np.asarray(t.dmd_centre) + off], offset_dmd=off,
                    hold_s=hold, held_s=round(held, 3), hold_cancelled=held < hold - 1e-9,
                    verdict=verdict, lit_pct=lit.get("lit_pct"), spill_pct=lit.get("spill_pct"),
                    centre_offset_px=off_px, centre_offset_mm=off_mm,
                    size_ratio=ratio, tol_px=tol_px if mode == "check" else None,
                    measurement=meas, polarity_inverted=inverted,
                    units={"offset_dmd": "native DMD px (cols, rows)", "centre_offset_px": "camera px",
                           "centre_offset_mm": "mm on the face (face mm per camera px)"},
                    warnings=warnings)
    js = s.store.output_path("previews", f"{label}_ring", ".json")
    s.store.write_json(js, json_safe({k: v for k, v in vars(shot).items()}))
    for f in (photo_p, over_p, js):
        if f:
            s.store.record_output("previews", f, label=label, geometry_id=t.geometry_id, source=t.path)
    return shot


# ------------------------------------------------------------ auto intensity

def auto_intensity(s, preset, face=None, target_dn: float = TARGET_DN, tol_dn: float = 10.0,
                   max_shots: int = 4, margin_dmd_px: float = 6.0,
                   label: str = "auto_intensity") -> dict:
    """projection.duty so that the face's p99 is ~target_dn at the preset's (whole-frame)
    exposure, from flood shots lit through the face mask. Never edits the preset and never
    touches the LED currents (no USB): pass the duty to targeted_face_scan."""
    safe_label(label)
    if preset.projection.invert:
        raise ValueError("auto_intensity needs a preset with projection.invert false: with invert "
                         "the duty scales the DARK level, not the face's brightness; use a "
                         "*_bright preset (projection.invert false)")
    if not 20 <= float(target_dn) <= 250:
        raise ValueError(f"target_dn must be 20..250 camera DN, got {target_dn!r}")
    if int(max_shots) < 1:
        raise ValueError("max_shots must be >= 1")
    t = as_face(s, face, preset, "auto_intensity")
    rules.check_led_duration(preset.projection.led, None, "auto_intensity")
    full, _, _ = face_cam_mask(t)
    mask = patterns.face_mask(t.dmd_ellipse, margin_dmd_px)
    shape = patterns.DMD_SHAPE
    warnings, shots = [], []
    d0 = float(preset.projection.duty)
    with s.rig(preset, zone_mask=mask) as sync:
        cfg0 = sync.projector.cfg
        try:
            dark = sync.shoot(np.zeros(shape, np.uint8))
            if dark.shape != full.shape:
                full, _, _ = face_cam_mask(t, dark.shape)
            dark_dn = float(np.median(dark[full]))
            d = d0
            for _ in range(int(max_shots)):
                sync.projector.cfg = replace(cfg0, duty=d)
                fr = sync.shoot(np.full(shape, 255, np.uint8))
                v = fr[full]
                p99 = float(np.percentile(v, 99))
                sat = float(100.0 * np.mean(v >= SATURATED_DN))
                shots.append({"duty": round(d, 4), "p99_dn": round(p99, 2),
                              "p50_dn": round(float(np.median(v)), 2), "saturated_pct": round(sat, 3)})
                if abs(p99 - target_dn) <= tol_dn and sat < 1.0:
                    break
                if p99 >= SATURATED_DN - 4 or sat >= 1.0:
                    nd = d * 0.5
                else:
                    nd = d * (target_dn - dark_dn) / max(p99 - dark_dn, 1.0)
                nd = float(np.clip(nd, MIN_DUTY, 1.0))
                if abs(nd - d) < 1e-3:
                    break
                d = nd
        finally:
            sync.projector.cfg = cfg0
    last = shots[-1]
    ok = [x for x in shots if x["saturated_pct"] < 1.0 and x["p99_dn"] < SATURATED_DN - 4]
    verified = abs(last["p99_dn"] - target_dn) <= tol_dn and last["saturated_pct"] < 1.0
    if verified:
        duty = last["duty"]
    elif ok:
        b = min(ok, key=lambda x: abs(x["p99_dn"] - target_dn))
        duty = float(np.clip(b["duty"] * (target_dn - dark_dn) / max(b["p99_dn"] - dark_dn, 1.0),
                             MIN_DUTY, 1.0))
        warnings.append(f"duty {duty:.3f} is predicted (linear in duty from the {b['duty']:g} shot), "
                        "not verified by a shot")
    else:
        duty = MIN_DUTY
        warnings.append("every shot saturated the face: lower the exposure (whole video frames)")
    if duty >= 1.0 - 1e-6 and last["p99_dn"] < target_dn - tol_dn:
        warnings.append(f"at duty 1 the face's p99 is only {last['p99_dn']:.0f} DN: use a longer "
                        "whole-frame exposure (33 333 us) - never the LED currents")
    exp = preset.camera.exposure_us
    locked = config.frame_locked(exp)
    if not locked and not s.mock:
        warnings.append(f"exposure {exp} us is not a whole number of 60 Hz video frames")
    rep = {"duty": round(float(duty), 4), "preset_duty": d0, "target_dn": float(target_dn),
           "tol_dn": float(tol_dn), "verified": bool(verified), "dark_dn": round(dark_dn, 2),
           "p99_dn": last["p99_dn"], "shots": shots, "exposure_us": exp, "frame_locked": locked,
           "face_id": t.id, "preset": preset.name, "led": "unchanged (never set by the engine)",
           "use": "targeted_face_scan(duty=...) (the preset file is not edited)",
           "units": {"p99_dn": "camera DN (8-bit, averaged frames)", "duty": "fraction of full "
                     "DMD drive"}, "warnings": warnings}
    out = s.store.output_path("tuning", label, ".json")
    s.store.write_json(out, json_safe(rep))
    s.store.record_output("tuning", out, label=label, geometry_id=preset.geometry_id, source=t.path)
    rep["file"] = out
    return rep


# ------------------------------------------------------------ targeted scan

def _facescan_handle(d: Path) -> Optional[FaceScan]:
    m = read_meta(d)
    if m is None or m.get("kind") != "facescan":
        return None
    return FaceScan(id=m["id"], label=m.get("label", d.name), geometry_id=m["geometry_id"],
                    path=Path(d), created=m.get("created", ""), face_id=m.get("face_target", ""),
                    metric_scan_ids=list(m.get("metric_scans") or []), repeats=list(m.get("repeats") or []),
                    polar=bool(m.get("polar", m.get("polar_patterns"))), meta=m,
                    warnings=list(m.get("warnings") or []))


def list_facescans(store, geometry_id: Optional[str] = None) -> list:
    geos = [geometry_id] if geometry_id else store.geometry_ids("facescans")
    out = []
    for g in geos:
        for d in store.entries("facescans", g):
            h = _facescan_handle(d)
            if h is not None:
                out.append(h)
    return out


def as_facescan(s, fs) -> FaceScan:
    if isinstance(fs, FaceScan):
        return fs
    p = Path(str(fs))
    h = _facescan_handle(p) if p.is_dir() else None
    if h is not None:
        return h
    hits = [f for f in list_facescans(s.store) if f.id == str(fs) or f.label == str(fs)]
    if not hits:
        raise ValueError(f"no face scan {fs!r} in the library (see list_library('facescans'))")
    return hits[-1]


def targeted_face_scan(s, preset, face=None, label: str = "face", repeats: int = 3,
                       margin_dmd_px: float = 6.0, polar: bool = True, duty: Optional[float] = None,
                       steps=None, periods=None, polar_steps: int = patterns.POLAR_STEPS,
                       ring_periods=patterns.RING_PERIODS_R, spokes=patterns.SPOKES,
                       mode: str = "multifreq", code_bits: int = 5) -> FaceScan:
    """N repeats of: depth set (multifreq xy, ray-plane triangulation) + optional polar set,
    lit only through the face ellipse (+ margin). Rule 6 before the rig.
    mode="tilephase" (EXPERIMENTAL, bases/tilephase.py): one fine phase level (steps, default
    12, at periods[0], default 8 px) + code_bits Gray tile planes per axis instead of the
    multifreq ladder (5 bits: 36 depth frames vs 74); refused when the lit face (+ margin) does
    not fit the code's unique window (2 ** (code_bits - 1) x period DMD px per axis)."""
    from ...calibrate import flow, stereo
    from ...reconstruct import metric_scan
    from ...reconstruct.triangulate import triangulate
    safe_label(label)
    if int(repeats) < 1 or int(repeats) > 20:
        raise ValueError(f"repeats must be 1..20, got {repeats!r}")
    if not 0 <= float(margin_dmd_px) <= 60:
        raise ValueError(f"margin_dmd_px must be 0..60 DMD px, got {margin_dmd_px!r}")
    p = preset
    if duty is not None:
        if not 0 < float(duty) <= 1:
            raise ValueError(f"duty must be in (0, 1], got {duty!r}")
        p = replace(p, projection=replace(p.projection, duty=float(duty)))
    fp = config.geometry_fingerprint(p)
    cal = stereo.require(s.store, p.geometry_id, fp, "targeted_face_scan")    # rule 6, pre-rig
    t = as_face(s, face, p, "targeted_face_scan")                               # rule 5
    warnings = []
    if t.calibration_id != cal.id:
        warnings.append(f"face target {t.id} was located under stereo calibration {t.calibration_id}; "
                        f"the current one is {cal.id} (the camera mask and DMD ring are px, still "
                        "valid if the rig did not move)")
    if mode not in DEPTH_MODES:
        raise ValueError(f"mode must be one of {list(DEPTH_MODES)}, got {mode!r}")
    mask = patterns.face_mask(t.dmd_ellipse, margin_dmd_px)
    if mode == "tilephase":
        from ...bases.tilephase import DEFAULT_PERIOD, DEFAULT_STEPS, TilePhaseBasis
        if periods is not None and len(list(periods)) != 1:
            raise ValueError("mode tilephase takes one period (the fine fringe), e.g. [8]")
        basis = TilePhaseBasis((t.dmd_ellipse["cx"], t.dmd_ellipse["cy"]),
                               steps if steps is not None else DEFAULT_STEPS,
                               float(list(periods)[0]) if periods is not None else DEFAULT_PERIOD,
                               code_bits, cal.diamond_parity)
        why = basis.fits(mask)
        if why:
            raise ValueError(f"tilephase: {why}")
        set_meta = {"basis": "tilephase", "axis": "xy", "steps": basis.steps,
                    "periods_px": [basis.T], "code_bits_per_axis": basis.bits,
                    "window_px": basis.window_px, "diamond_parity": cal.diamond_parity,
                    "experimental": True}
    else:
        basis = flow.multifreq_basis(p, cal.diamond_parity, steps, periods)
        set_meta = {"basis": "multifreq", "axis": "xy", "steps": basis.steps,
                    "periods_px": basis.periods, "diamond_parity": cal.diamond_parity}
    depth_pats = basis.generate()
    nb = len(depth_pats)
    off = patterns.off_dmd_pct(t.dmd_ellipse)
    if off > 0.5:
        warnings.append(f"{off:.0f}% of the face maps outside the DMD: that part is never lit "
                        "(the check will call the face INCOMPLETE); re-place the part")
    # polar=False drops the rings / spokes (no added sensitivity in the research) but keeps
    # the flood pair: the MF detector's reflectance FIR needs it
    specs = patterns.polar_specs(ring_periods if polar else (), spokes if polar else (),
                                 polar_steps)
    pol = patterns.polar_patterns(t.dmd_ellipse, specs)
    full, _, _ = face_cam_mask(t)
    x0, y0, x1, y1 = box = _crop_box(full)
    created = _now()
    d = s.store.new_dir("facescans", p.geometry_id, label=label)
    reps, ids = [], []
    with s.store.filling(d):
        s.store.write_npz(d / "masks.npz", dmd_mask=mask, face_cam_mask=full[y0:y1, x0:x1],
                          crop=np.array(box))
        flood = None
        for k in range(int(repeats)):
            t0 = time.perf_counter()
            fr = flow.capture_patterns(s, p, depth_pats + pol, "a targeted end-face scan",
                                       zone_mask=mask)
            fld = basis.decode(fr[:nb], p.geometry_id, contrast_min=p.decode.contrast_min,
                               conf_min=p.decode.conf_min)
            cap = round(time.perf_counter() - t0, 2)
            cloud = triangulate(fld, cal, "plane")
            lab = f"{label}_r{k + 1}"
            fld.meta.update({"preset": p.name, "label": lab, "mock": s.mock})
            ms = metric_scan._save(s, fld, cloud, lab, cal, fp, p.name,
                                   {"capture_s": cap, "steps": basis.steps,
                                    "periods_px": set_meta["periods_px"], "depth_mode": mode,
                                    "diamond_parity": cal.diamond_parity, "zone": "custom:face_ellipse",
                                    "targeted": True, "face_target": t.id, "facescan": d.name,
                                    "repeat": k + 1, "duty": p.projection.duty})
            if specs:
                _save_npy(d / f"polar_frames_r{k + 1}.npy",
                          np.stack([np.asarray(f, np.float32)[y0:y1, x0:x1] for f in fr[nb:]]))
            flood = np.asarray(fr[0], np.float32)
            ids.append(ms.id)
            n_face = int((cloud.mask & full).sum())
            reps.append({"metric_scan": ms.id, "capture_s": cap, "n_points": int(cloud.mask.sum()),
                         "face_points": n_face, "face_decoded_pct": round(100.0 * n_face / max(int(full.sum()), 1), 2),
                         "consistency": cloud.consistency_stats()})
            if n_face < 0.5 * full.sum():
                warnings.append(f"repeat {k + 1}: only {reps[-1]['face_decoded_pct']:.0f}% of the face "
                                "decoded (dark or glinting face? run auto_intensity; moved part? "
                                "check_ring)")
        if flood is not None:
            s.store.write_preview(d / "preview.png", flood[y0:y1, x0:x1])
        meta = {"kind": "facescan", "id": d.name, "label": label, "geometry_id": p.geometry_id,
                "geometry_fp": fp, "created": created, "face_target": t.id,
                "calibration_id": cal.id, "preset": p.name, "duty": p.projection.duty,
                "invert": p.projection.invert, "exposure_us": p.camera.exposure_us,
                "average": p.camera.average, "margin_dmd_px": float(margin_dmd_px),
                "dmd_ellipse": t.dmd_ellipse, "method": "plane",
                "depth_set": set_meta, "depth_frames": nb,
                "polar": bool(polar), "polar_patterns": specs, "crop_x0y0x1y1": list(box),
                "metric_scans": ids, "repeats": reps, "mock": s.mock,
                "note": "polar coords: R = 1 at the face rim (the DMD face ellipse normalised); "
                        "polar frames are float32 camera DN of the crop, in polar_patterns order; "
                        "pattern 255 lands bright when invert is false",
                "units": {"polar_frames": "camera DN (averaged)", "crop": "camera px"},
                "warnings": warnings}
        s.store.write_json(d / "meta.json", json_safe(meta))
    return _facescan_handle(d)


# ------------------------------------------------------------ profile

def _profile_inputs(s, scans, face) -> tuple:
    """-> (metric scans, calibration, geometry, fp, face target, facescan id | None, warnings)."""
    warnings, fs = [], None
    if isinstance(scans, FaceScan):
        fs = scans
    elif isinstance(scans, (str, Path)):
        try:
            fs = as_facescan(s, scans)
        except ValueError:
            scans = [scans]                         # a metric scan id / folder
    elif isinstance(scans, MetricScan):
        scans = [scans]
    if fs is not None:
        scans = fs.metric_scan_ids
    if not scans:
        raise ValueError("face_profile: give a face scan (targeted_face_scan) or its metric scans")
    ms, cal, geo, fp = _scans_and_cal(s, list(scans), "face_profile")
    if face is None and fs is not None:
        face = fs.face_id
    if face is None:
        face_ids = {(m.meta or {}).get("face_target") for m in ms} - {None}
        if len(face_ids) == 1:
            face = face_ids.pop()
    if face is None:
        cands = [f for f in list_faces(s.store, geo)
                 if not rules.fingerprint_diffs(f.meta.get("geometry_fp"), fp)]
        if not cands:
            raise ValueError(f"face_profile: no face target at geometry {geo}: pass face=")
        face = cands[-1]
        warnings.append(f"no face target named: using the newest at {geo}, {face.id}")
    t = as_face(s, face, None, "face_profile")
    rules.require_same_geometry(t.geometry_id, geo, f"face_profile: face target {t.id}")
    rules.require_same_fingerprint(t.meta.get("geometry_fp"), fp, f"face_profile: face target {t.id}")
    return ms, cal, geo, fp, t, (fs.id if fs is not None else None), warnings


def _face_mesh(res: dict, fm: np.ndarray, path: Path, meta: dict) -> tuple:
    """Open surface of the face (median of the repeats) in the face frame: x = u (camera
    right), y = -v (camera up), z = h (toward the camera) - right-handed, mm."""
    from ...reconstruct import mesh as mesh_mod
    M, C, n, e1, e2 = res["median_xyz"], res["centre"], res["normal"], res["e1"], res["e2"]
    W = M - C
    U, V, Hh = W @ e1, -(W @ e2), W @ n
    keep = fm & np.all(np.isfinite(M), -1)
    verts, faces = mesh_mod.grid_mesh(U, V, Hh, keep)
    if len(faces) == 0:
        raise ValueError("no complete face cells to mesh")
    e = np.stack([np.linalg.norm(verts[faces[:, i]] - verts[faces[:, (i + 1) % 3]], axis=1)
                  for i in range(3)], 1)
    max_edge = 4.0 * float(np.median(e))
    faces = faces[(e <= max_edge).all(1)]
    used = np.unique(faces)
    remap = np.full(len(verts), -1, np.int64)
    remap[used] = np.arange(len(used))
    meta = {"kind": "end_face", "units": {"xyz": "mm"}, "to_scale": True,
            "frame": "face: x = u (~camera right), y = -v (~camera up), z = h (+ toward the camera)",
            "face_centre_mm_camera": C.tolist(), "e1": e1.tolist(), "e2": e2.tolist(),
            "normal": n.tolist(), "max_edge_mm": round(max_edge, 4),
            "note": "open surface as measured (median of the repeats); nothing closed or invented",
            **meta}
    return mesh_mod._write(path, "stl", verts[used], remap[faces], meta)


def _floods(fs: Optional[FaceScan], ms_ids: list, box: tuple, warnings: list) -> dict:
    """The targeted scan's flood pair (255 / 0) for the profile crop `box`: mean over the
    repeats in ms_ids, saturation = any repeat's flood at / above FIR sat_dn. {} without."""
    if fs is None:
        return {}
    specs = fs.meta.get("polar_patterns") or []
    i_on = next((i for i, p in enumerate(specs) if p.get("kind") == "flood"
                 and int(p.get("level", -1)) == 255), None)
    i_off = next((i for i, p in enumerate(specs) if p.get("kind") == "flood"
                  and int(p.get("level", -1)) == 0), None)
    crop = fs.meta.get("crop_x0y0x1y1")
    if i_on is None or i_off is None or not crop:
        return {}
    fx0, fy0, fx1, fy1 = (int(v) for v in crop)
    x0, y0, x1, y1 = box
    ix0, iy0, ix1, iy1 = max(x0, fx0), max(y0, fy0), min(x1, fx1), min(y1, fy1)
    if ix1 <= ix0 or iy1 <= iy0:
        warnings.append("the targeted scan's polar frames do not cover the face crop: no flood pair")
        return {}
    ons, offs = [], []
    for k, mid in enumerate(fs.metric_scan_ids, 1):
        if mid not in ms_ids:
            continue
        f = Path(fs.path) / f"polar_frames_r{k}.npy"
        if not f.is_file():
            continue
        F = np.load(f, mmap_mode="r")
        if F.shape[0] != len(specs) or F.shape[1:] != (fy1 - fy0, fx1 - fx0):
            warnings.append(f"{f.name}: {F.shape} does not match the scan's polar patterns / crop: "
                            "not used")
            continue
        a, b = (np.full((y1 - y0, x1 - x0), np.nan, np.float64) for _ in range(2))
        a[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0] = F[i_on, iy0 - fy0:iy1 - fy0, ix0 - fx0:ix1 - fx0]
        b[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0] = F[i_off, iy0 - fy0:iy1 - fy0, ix0 - fx0:ix1 - fx0]
        ons.append(a)
        offs.append(b)
    if not ons:
        return {}
    on, off = np.mean(ons, 0), np.mean(offs, 0)
    if np.nanmean(on) < np.nanmean(off):          # measured, never assumed (an invert preset)
        on, off, ons = off, on, offs
        warnings.append("flood 255 landed darker than flood 0 (invert preset): the pair is swapped")
    sat = np.nan_to_num(np.max(ons, 0), nan=0.0) >= mfcfar.FIR_DEFAULTS["sat_dn"]
    return {"flood_on": on, "flood_off": off, "flood_sat": sat, "n_repeats": len(ons)}


def face_profile(s, scans=None, face=None, label: Optional[str] = None,
                 grid_mm: float = profile_mod.GRID_MM, r_flat_mm: float = profile_mod.R_FLAT_MM,
                 stl: bool = True) -> FaceProfile:
    ms, cal, geo, fp, t, fs_id, warnings = _profile_inputs(s, scans, face)
    label = label or (ms[0].label.rsplit("_r", 1)[0] if ms[0].label else "face")
    safe_label(label)
    if not (0.05 <= float(grid_mm) <= 2.0):
        raise ValueError(f"grid_mm must be 0.05..2 mm, got {grid_mm!r}")
    full, (x0, y0, x1, y1), cm = face_cam_mask(t)
    shape = ms[0].cloud.mask.shape
    if shape != full.shape:
        raise ValueError(f"face target {t.id} is for a {full.shape} camera, the scans are {shape}")
    X = np.stack([m.cloud.xyz[y0:y1, x0:x1] for m in ms]).astype(np.float64)
    K = np.stack([m.cloud.mask[y0:y1, x0:x1] for m in ms])
    radius = t.face_diameter_mm / 2
    clipped = bool(full[0].any() or full[-1].any() or full[:, 0].any() or full[:, -1].any())
    if clipped:
        warnings.append("the face leaves the camera frame: the profile centre comes from the "
                        "located circle, the part outside is not measured")
    e = t.camera_ellipse
    res = profile_mod.analyse(X, K, cm, (x0, y0), cal, radius, grid_mm, r_flat_mm,
                              centre_px=(e["cx"] - x0, e["cy"] - y0) if clipped else None)
    st = res["stats"]
    fs = as_facescan(s, fs_id) if fs_id else None
    fl = _floods(fs, [m.id for m in ms], (x0, y0, x1, y1), warnings)
    pxd = res["pixels"]
    pixels = mfcfar.Pixels(iy=pxd["iy"], ix=pxd["ix"], U=pxd["U"], V=pxd["V"], H=pxd["H"],
                           cam_mask=cm, flood_on=fl.get("flood_on"), flood_off=fl.get("flood_off"),
                           flood_sat=fl.get("flood_sat"))
    st["flood_pair"] = bool(fl)
    st["flood_saturated_pct"] = mfcfar.saturated_pct(pixels)
    if not fl:
        warnings.append("no flood pair (a profile from metric scans, or a targeted scan without its "
                        "polar set): the MF detector runs without the reflectance FIR")
    elif st["flood_saturated_pct"] and st["flood_saturated_pct"] > 1.0:
        warnings.append(f"{st['flood_saturated_pct']:.0f}% of the face is saturated in the flood: "
                        "lower the duty (auto_intensity, ~200 DN); a saturated face is noisier and "
                        f"above {mfcfar.FIR_DEFAULTS['max_sat_pct']:g}% it is checked without the "
                        "reflectance FIR")
    if len(ms) < 2:
        warnings.append("one scan: no repeatability (targeted_face_scan repeats >= 2)")
    elif st["repeatability_median_mm"] is not None and st["repeatability_median_mm"] > REPEAT_WARN_MM:
        warnings.append(f"per-point repeatability {st['repeatability_median_mm']:.3f} mm (median) is "
                        f"above {REPEAT_WARN_MM:g} mm: dim / glinting face or a moving part?")
    if st["grid_coverage_pct"] is not None and st["grid_coverage_pct"] < 90:
        warnings.append(f"only {st['grid_coverage_pct']:.0f}% of the face grid is measured")
    if t.calibration_id != cal.id:
        warnings.append(f"face target {t.id} was located under calibration {t.calibration_id}, the "
                        f"scans under {cal.id}")
    created = _now()
    pmeta = {"kind": "face_profile", "label": label, "geometry_id": geo, "geometry_fp": fp,
             "calibration_id": cal.id, "face_id": t.id, "facescan_id": fs_id,
             "metric_scan_ids": [m.id for m in ms], "face_diameter_mm": t.face_diameter_mm,
             "grid_mm": float(grid_mm), "created": created, "mock": s.mock}
    src = Path(s.store.root / "facescans" / geo / fs_id) if fs_id else ms[0].path
    files = {}
    png = profile_mod.profile_png(res, radius, f"{label}: {len(ms)} repeat(s), face {t.id}")
    files["png"] = _write_bytes(s.store.output_path("profiles", f"{label}_face", ".png"), png)
    files["csv_radial"] = _write_text(s.store.output_path("profiles", f"{label}_face_radial", ".csv"),
                                      profile_mod.radial_csv(res["radial"]))
    files["csv_sections"] = _write_text(s.store.output_path("profiles", f"{label}_face_sections",
                                                            ".csv"), profile_mod.sections_csv(res["sections"]))
    npz = s.store.output_path("profiles", f"{label}_face", ".npz")
    s.store.write_npz(npz, grid_mm=res["grid_mm"], height_mm=res["height_mm"].astype(np.float32),
                      u=res["u"].astype(np.float32), v=res["v"].astype(np.float32),
                      h=res["h"].astype(np.float32), centre=res["centre"], normal=res["normal"],
                      e1=res["e1"], e2=res["e2"], crop=np.array([x0, y0, x1, y1]),
                      meta=np.array(json.dumps(json_safe(pmeta))), **pixels.arrays())
    files["npz"] = npz
    if stl:
        try:
            p_, mp_, _ = _face_mesh(res, cm, s.store.output_path("pointclouds", f"{label}_face", ".stl"),
                                    {"geometry_id": geo, "calibration_id": cal.id, "face_id": t.id,
                                     "metric_scans": pmeta["metric_scan_ids"], "label": label})
            files["stl"], files["stl_meta"] = p_, mp_
        except ValueError as e:
            warnings.append(f"no STL: {e}")
    doc = {**pmeta, "stats": st, "radial": res["radial"],
           "sections": {"pos_mm": res["grid_mm"], "horizontal_mm": res["sections"]["horizontal_mm"],
                        "vertical_mm": res["sections"]["vertical_mm"]},
           "centre_mm": res["centre"], "normal": res["normal"], "e1": res["e1"], "e2": res["e2"],
           "units": {**UNITS_MAP, "stats": "mm (tilt deg, coverage %)"}, "warnings": warnings,
           "files": {k: v for k, v in files.items() if v is not None}}
    files["json"] = s.store.output_path("profiles", f"{label}_face", ".json")
    s.store.write_json(files["json"], json_safe(doc))
    for k, f in files.items():
        if f is not None:
            s.store.record_output("pointclouds" if k.startswith("stl") else "profiles", f,
                                  label=label, geometry_id=geo, source=src)
    return FaceProfile(label=label, geometry_id=geo, calibration_id=cal.id, face_id=t.id,
                       facescan_id=fs_id, metric_scan_ids=pmeta["metric_scan_ids"],
                       face_diameter_mm=t.face_diameter_mm, grid_mm=float(grid_mm), stats=st,
                       radial=res["radial"], centre_mm=res["centre"].tolist(),
                       normal=res["normal"].tolist(), units={**UNITS_MAP},
                       files={k: v for k, v in files.items() if v is not None}, warnings=warnings,
                       geometry_fp=fp, _height=res["height_mm"], _grid=res["grid_mm"],
                       _pixels=pixels)


def load_profile(s, item) -> FaceProfile:
    """A FaceProfile, a face-profile .npz (output bin), or a face scan (handle / id: its
    profile is computed now)."""
    if isinstance(item, FaceProfile):
        if item._height is None:
            return load_profile(s, item.files.get("npz"))
        return item
    p = Path(str(item))
    if p.suffix.lower() == ".npz" and p.is_file():
        with np.load(p) as z:
            m = json.loads(str(z["meta"]))
            if m.get("kind") != "face_profile":
                raise ValueError(f"{p} is not a face profile")
            return FaceProfile(label=m["label"], geometry_id=m["geometry_id"],
                               calibration_id=m["calibration_id"], face_id=m["face_id"],
                               facescan_id=m.get("facescan_id"), metric_scan_ids=m["metric_scan_ids"],
                               face_diameter_mm=float(m["face_diameter_mm"]), grid_mm=float(m["grid_mm"]),
                               stats={}, radial=[], centre_mm=z["centre"].tolist(),
                               normal=z["normal"].tolist(), files={"npz": p}, geometry_fp=m.get("geometry_fp"),
                               _height=z["height_mm"].astype(np.float64), _grid=z["grid_mm"].astype(np.float64),
                               _pixels=mfcfar.Pixels.from_arrays(z))
    return face_profile(s, as_facescan(s, item))


def _check_same(profiles: list, what: str) -> dict:
    a = profiles[0]
    for q in profiles[1:]:
        rules.require_same_geometry(q.geometry_id, a.geometry_id, what)
        rules.require_same_fingerprint(q.geometry_fp, a.geometry_fp, what)
        if q.calibration_id != a.calibration_id:
            raise rules.GeometryMismatch(f"{what}: placements measured under different stereo "
                                         f"calibrations ({a.calibration_id} vs {q.calibration_id})")
        if abs(q.face_diameter_mm - a.face_diameter_mm) > DIAMETER_TOL_MM:
            raise ValueError(f"{what}: face diameters differ ({a.face_diameter_mm:g} vs "
                             f"{q.face_diameter_mm:g} mm): one golden per face type")
        if q._grid.shape != a._grid.shape or not np.allclose(q._grid, a._grid):
            raise ValueError(f"{what}: height maps on different grids ({a.grid_mm:g} vs {q.grid_mm:g} mm)")
    return {"geometry_id": a.geometry_id, "geometry_fp": a.geometry_fp,
            "calibration_id": a.calibration_id, "face_diameter_mm": a.face_diameter_mm,
            "grid_mm": a.grid_mm}


def _require_current_cal(s, geo, fp, cal_id, what):
    from ...calibrate import stereo
    cal = stereo.require(s.store, geo, fp, what)
    if cal.id != cal_id:
        raise rules.GeometryMismatch(
            f"{what}: made under stereo calibration {cal_id}, but the current one for {geo} is "
            f"{cal.id}: rebuild it from placements measured (or re-triangulated) with the current "
            "calibration (rule 6)")
    return cal


# ------------------------------------------------------------ golden

def _golden_handle(d: Path) -> Optional[FaceGolden]:
    m = read_meta(d)
    if m is None or m.get("kind") != "face_golden":
        return None
    return FaceGolden(id=m["id"], label=m.get("label", d.name), geometry_id=m["geometry_id"],
                      path=Path(d), created=m.get("created", ""),
                      face_diameter_mm=float(m["face_diameter_mm"]),
                      calibration_id=m.get("calibration_id", ""), n_placements=int(m["n_placements"]),
                      params=m.get("params") or {}, leave_one_out=list(m.get("leave_one_out") or []),
                      sensitivity=m.get("sensitivity") or {}, units=m.get("units") or {},
                      files={"npz": Path(d) / "golden.npz", "png": Path(d) / "preview.png"},
                      meta=m, warnings=list(m.get("warnings") or []))


def list_goldens(store, geometry_id: Optional[str] = None) -> list:
    geos = [geometry_id] if geometry_id else store.geometry_ids("face_golden")
    out = []
    for g in geos:
        for d in store.entries("face_golden", g):
            h = _golden_handle(d)
            if h is not None:
                out.append(h)
    return out


def _band_params(d: Optional[dict]) -> golden_mod.BandParams:
    keys = set(golden_mod.BandParams().to_dict())
    return golden_mod.BandParams(**{k: v for k, v in (d or {}).items() if k in keys})


def _trig_params(d: Optional[dict]) -> golden_mod.TriggerParams:
    keys = set(golden_mod.TriggerParams().to_dict())
    return golden_mod.TriggerParams(**{k: v for k, v in (d or {}).items() if k in keys})


def _placement(p: FaceProfile) -> det_mod.Placement:
    return det_mod.Placement(label=p.label, height=np.asarray(p._height, np.float64),
                             pixels=p._pixels)


def _stored_settings(gh: FaceGolden, g: np.ndarray, over: Optional[dict] = None) -> tuple:
    """(Settings, default detector) from a golden's params (+ per-check trigger overrides)."""
    prm = gh.params or {}
    nested = isinstance(prm.get("band"), dict)
    over = dict(over or {})
    band = _band_params(prm["band"] if nested else prm)
    trig = _trig_params({**((prm.get("trigger") or {}) if nested else prm),
                         **{k: v for k, v in over.items() if k == "min_area_mm2"}})
    mfo = {k: v for k, v in (prm.get("mf") or {}).items() if k in mfcfar.MF_DEFAULTS}
    mfo.update({k: v for k, v in over.items() if k in ("z_thresh", "min_depth_mm", "min_area_mm2")})
    st = det_mod.Settings(g=g, radius_mm=gh.face_diameter_mm / 2, band=band, trig=trig,
                          mf=mfcfar.mf_params(mfo),
                          fir=mfcfar.fir_params({k: v for k, v in (prm.get("fir") or {}).items()
                                                 if k in mfcfar.FIR_DEFAULTS}))
    return st, (prm.get("detector") or "band")


def face_golden(s, placements: list, label: str = "golden", params: Optional[dict] = None,
                synthetic=True, detector: str = det_mod.DEFAULT_DETECTOR, fir: bool = True,
                mf_params: Optional[dict] = None) -> FaceGolden:
    """Golden face from >= 2 (3+ for leave-one-out) GOOD placements: every detector's model
    (MF radial profile + reflectance FIR, band median + tolerance band), the default detector
    and all settings in the record; leave-one-out false triggers (every detector) and the
    synthetic-dent set (the default detector)."""
    safe_label(label)
    if callable(detector):
        raise ValueError("face_golden: detector must be a name "
                         f"{det_mod.DETECTORS}; a trigger function is given to check_face")
    det = det_mod.check_name(detector)
    bad = set(params or {}) - set(golden_mod.BandParams().to_dict()) - set(golden_mod.TriggerParams().to_dict())
    if bad:
        raise ValueError(f"face_golden: unknown params {sorted(bad)}")
    if not placements or len(placements) < 2:
        raise ValueError("face_golden needs at least 2 GOOD placements (3+ for leave-one-out)")
    bat = det_mod.battery_spec(synthetic)
    mprm = mfcfar.mf_params(mf_params)
    profs = [load_profile(s, p) for p in placements]
    b = _check_same(profs, "face_golden")
    _require_current_cal(s, b["geometry_id"], b["geometry_fp"], b["calibration_id"], "face_golden")
    bp, tp = _band_params(params), _trig_params(params)
    g = profs[0]._grid
    radius = b["face_diameter_mm"] / 2
    st = det_mod.Settings(g=g, radius_mm=radius, band=bp, trig=tp, mf=mprm, fir=mfcfar.fir_params())
    pls = [_placement(p) for p in profs]
    names = [p.label for p in profs]
    warnings = []
    no_px = [p.label for p in pls if p.pixels is None]
    no_fl = [p.label for p in pls if p.pixels is None or not p.pixels.has_flood]
    use_fir = bool(fir) and not no_fl
    if no_px:
        warnings.append(f"no per-pixel data in {no_px} (profiles saved before the MF detector): the MF "
                        "uses their gridded median maps - re-run face_profile on their face scans")
    if fir and no_fl:
        warnings.append(f"no flood pair in {no_fl}: the golden has no reflectance FIR (the MF runs "
                        "without it, ~10-20 % noisier)")
    model = det_mod.quiet(det_mod.build)(pls, st, use_fir)
    if use_fir and model["fir_coef"] is None:
        warnings.append("the reflectance FIR could not be fitted "
                        f"({(model['fir_info'] or {}).get('error', '?')}): the golden has none")
        use_fir = False
    loo, sens = det_mod.quiet(det_mod.leave_one_out)(pls, st, det, use_fir, battery=bat)
    spread = golden_mod.placement_spread(np.stack([p.height for p in pls]), g, bp.rim_r_mm)
    if len(profs) < 3:
        warnings.append("fewer than 3 placements: no leave-one-out check (false triggers) and no "
                        "synthetic set")
    n_false = sum(r["n_face"] + (r["n_rim"] if det != "band" else 0) for r in loo)
    if loo and n_false:
        warnings.append(f"leave-one-out: {n_false} trigger(s) on GOOD placements with '{det}' - "
                        + ("add placements, or raise k_sigma / floor_mm" if det == "band" else
                           "check those placements (saturated flood? moved part?) or raise z_thresh"))
    if loo and det != "band":
        zs = [r["max_z"] for r in loo if r.get("max_z") is not None]
        if zs and max(zs) * 1.1 > mprm["z_thresh"]:
            warnings.append(f"leave-one-out: a good placement reached dent z {max(zs):.2f}, within "
                            f"10 % of z_thresh {mprm['z_thresh']:g}: a thin margin")
    sat = {p.label: mfcfar.saturated_pct(p.pixels) for p in pls if p.pixels is not None}
    hot = {k: v for k, v in sat.items() if v and v > 1.0}
    if hot:
        warnings.append(f"flood saturated on {hot} (% of the face): those placements are noisier "
                        f"and above {st.fir['max_sat_pct']:g}% are checked without the FIR; keep "
                        "the face below ~230 DN (auto_intensity)")
    loo0 = loo[0].pop("_maps", None) if loo else None
    for r in loo:
        r.pop("_maps", None)
    compare = {d: {"face": sum(r[d]["n_face"] for r in loo), "rim": sum(r[d]["n_rim"] for r in loo)}
               for d in det_mod.DETECTORS if loo and d in loo[0]}
    created = _now()
    d = s.store.new_dir("face_golden", b["geometry_id"], label=label)
    fir_d = {**st.fir, "used": use_fir,
             "coef": model["fir_coef"].tolist() if model["fir_coef"] is not None else None,
             "fit": model["fir_info"]}
    params_rec = {**st.to_dict(det, use_fir), "fir": fir_d}
    with s.store.filling(d):
        arrays = {"grid_mm": g, "golden_mm": model["band"]["golden"].astype(np.float32),
                  "band_mm": model["band"]["band"].astype(np.float32),
                  "sigma_mm": model["band"]["sigma"].astype(np.float32),
                  "heights_mm": np.stack([p.height for p in pls]).astype(np.float32),
                  "placements": np.array(names),
                  "mf_radial_r_mm": model["mf"]["radial_r_mm"],
                  "mf_radial_h_mm": model["mf"]["radial_h_mm"],
                  "mf_map_mm": model["mf"]["map_mm"].astype(np.float32)}
        if model["fir_coef"] is not None:
            arrays["fir_coef"] = model["fir_coef"]
        s.store.write_npz(d / "golden.npz", **arrays)
        png = _write_bytes(d / "preview.png", det_mod.golden_png(model, st, loo0))
        meta = {"kind": "face_golden", "id": d.name, "label": label, **b, "created": created,
                "n_placements": len(profs), "detector": det,
                "placements": [{"label": p.label, "face_id": p.face_id, "facescan_id": p.facescan_id,
                                "metric_scans": p.metric_scan_ids,
                                "flood_saturated_pct": sat.get(p.label)} for p in profs],
                "params": params_rec, "placement_spread": spread,
                "leave_one_out": loo, "leave_one_out_false_triggers": compare,
                "sensitivity": sens or {}, "mock": s.mock,
                "units": {"golden_mm": "mm, face frame (+ toward the camera)", "band_mm": "mm",
                          "mf_radial_h_mm": "mm at mf_radial_r_mm (mm)",
                          "fir_coef": "mm per unit high-passed log flood (7x7 camera px)",
                          "z": "dent significance (CFAR sigma units)", **UNITS_MAP},
                "note": "detector mf_fir / mf: MF-CFAR (mfcfar.py) against the median radial "
                        "profile of the placements (+ the reflectance FIR fitted on them); band: "
                        "median map + max(k_sigma x sigma, floor), rim r > rim_r_mm x rim_factor. "
                        "Bound to geometry + fingerprint, face diameter, grid and the stereo "
                        "calibration",
                "warnings": warnings}
        s.store.write_json(d / "meta.json", json_safe(meta))
    if png:
        s.store.record_output("previews", png, label=label, geometry_id=b["geometry_id"], source=d)
    return _golden_handle(d)


def as_golden(s, golden, prof: FaceProfile) -> FaceGolden:
    if isinstance(golden, FaceGolden):
        return golden
    if golden is not None:
        p = Path(str(golden))
        h = _golden_handle(p) if p.is_dir() else None
        if h is None:
            hits = [x for x in list_goldens(s.store) if x.id == str(golden) or x.label == str(golden)]
            if not hits:
                raise ValueError(f"no golden face {golden!r} in the library (see "
                                 "list_library('face_golden'))")
            h = hits[-1]
        return h
    cands = [x for x in list_goldens(s.store, prof.geometry_id)
             if not rules.fingerprint_diffs(x.meta.get("geometry_fp"), prof.geometry_fp)
             and abs(x.face_diameter_mm - prof.face_diameter_mm) <= DIAMETER_TOL_MM
             and x.calibration_id == prof.calibration_id
             and abs(float(x.meta.get("grid_mm", 0)) - prof.grid_mm) < 1e-9]
    if not cands:
        raise ValueError(f"no golden face for geometry {prof.geometry_id}, face {prof.face_diameter_mm:g} "
                         f"mm, grid {prof.grid_mm:g} mm, calibration {prof.calibration_id}: build one "
                         "with face_golden from good placements")
    return cands[-1]


def _golden_model(gh: FaceGolden) -> tuple:
    with np.load(Path(gh.path) / "golden.npz") as z:
        g = z["grid_mm"].astype(np.float64)
        model = {"band": {"golden": z["golden_mm"].astype(np.float64),
                          "band": z["band_mm"].astype(np.float64)},
                 "mf": ({"radial_r_mm": z["mf_radial_r_mm"].astype(np.float64),
                         "radial_h_mm": z["mf_radial_h_mm"].astype(np.float64),
                         "map_mm": z["mf_map_mm"].astype(np.float64)}
                        if "mf_radial_h_mm" in z.files else None),
                 "fir_coef": z["fir_coef"].astype(np.float64) if "fir_coef" in z.files else None}
    return g, model


def check_face(s, placement, golden=None, label: Optional[str] = None,
               params: Optional[dict] = None, detector=None) -> FaceCheck:
    """Placement vs golden -> triggers, PASS / FAIL / INCOMPLETE, overlay. detector: None =
    the golden's, a name in detectors.DETECTORS, or a callable(dev_map, band, params) trigger
    function (the band detector with that function)."""
    prof = load_profile(s, placement)
    gh = as_golden(s, golden, prof)
    what = f"check_face vs golden {gh.id}"
    rules.require_same_geometry(prof.geometry_id, gh.geometry_id, what)
    rules.require_same_fingerprint(prof.geometry_fp, gh.meta.get("geometry_fp"), what)
    if abs(prof.face_diameter_mm - gh.face_diameter_mm) > DIAMETER_TOL_MM:
        raise ValueError(f"{what}: the golden is for a {gh.face_diameter_mm:g} mm face, the placement "
                         f"{prof.face_diameter_mm:g} mm")
    if prof.calibration_id != gh.calibration_id:
        raise rules.GeometryMismatch(f"{what}: the golden was made under stereo calibration "
                                     f"{gh.calibration_id}, the placement under {prof.calibration_id} "
                                     "(rule 6)")
    _require_current_cal(s, gh.geometry_id, gh.meta.get("geometry_fp"), gh.calibration_id, what)
    g, model = _golden_model(gh)
    if g.shape != prof._grid.shape or not np.allclose(g, prof._grid):
        raise ValueError(f"{what}: the placement's grid ({prof.grid_mm:g} mm) is not the golden's")
    over = {k: v for k, v in (params or {}).items() if v is not None}
    extra = set(over) - {"min_area_mm2", "z_thresh", "min_depth_mm"}
    if extra:
        raise ValueError(f"check_face: unknown params {sorted(extra)} (the other settings belong "
                         "to the golden)")
    st, default_det = _stored_settings(gh, g, over)
    det = detector if detector is not None else default_det
    name = det_mod.check_name(det)
    if not callable(det) and det != "band" and model["mf"] is None:
        raise ValueError(f"{what}: golden {gh.id} was built before the MF detector (band only): "
                         "check with detector='band' or build a new golden")
    res = det_mod.quiet(det_mod.check)(det, _placement(prof), model, st)
    trig, stt = res["triggers"], res["stats"]
    verdict = det_mod.verdict(trig, stt)
    label = label or prof.label
    safe_label(label)
    warnings = list(res["warnings"])
    if verdict == "INCOMPLETE":
        warnings.append(f"only {stt['compared_pct'] or 0:.0f}% of the inspected face measured "
                        f"(< {det_mod.MIN_COMPARED_PCT:g}%): no dent found in what was measured, "
                        "but the face is NOT passed (re-scan: auto_intensity, check_ring)")
    elif stt["compared_pct"] is not None and stt["compared_pct"] < 99:
        warnings.append(f"{100 - stt['compared_pct']:.1f}% of the inspected face unmeasured "
                        "(holes cannot trigger)")
    placement_d = {"label": prof.label, "face_id": prof.face_id, "facescan_id": prof.facescan_id,
                   "metric_scans": prof.metric_scan_ids, "profile_npz": prof.files.get("npz")}
    files = {}
    out = s.store.output_path("defects", f"{label}_face_check", ".json")
    files["png"] = _write_bytes(out.with_suffix(".png"), det_mod.check_png(res, st, verdict))
    files["npz"] = out.with_suffix(".npz")
    maps = {k: np.asarray(v, np.float32) for k, v in res["maps"].items()}
    s.store.write_npz(files["npz"], grid_mm=g, dev_mm=res["dev"].astype(np.float32), **maps)
    files["json"] = out
    mf_used = name in ("mf", "mf_fir")
    params_used = {"detector": name, **({"mf": {k: (list(v) if isinstance(v, tuple) else v)
                                               for k, v in st.mf.items()},
                                         "fir_applied": stt.get("fir_applied")} if mf_used else
                                        {**st.band.to_dict(), **st.trig.to_dict()}),
                   "min_compared_pct": det_mod.MIN_COMPARED_PCT}
    units = {**UNITS_MAP, "dev_mm": "mm, placement - golden (- = below: a dent)"}
    units.update({"depth_mm": "mm, matched-filter dent depth (+ = below the surface)",
                  "width_mm": "mm, best dent diameter of the filter bank",
                  "z": "dent significance (CFAR sigma units)",
                  "noise_floor_mm": "mm, 1-sigma dent depth per diameter"} if mf_used
                 else {"band_mm": "mm"})
    rep = FaceCheck(verdict=verdict, triggers=trig, n_triggers=len(trig), golden_id=gh.id,
                    placement=placement_d, stats=stt, params=params_used, detector=name,
                    units=units, files=files, warnings=warnings)
    s.store.write_json(out, json_safe({k: v for k, v in vars(rep).items() if k != "files"}
                                      | {"files": {k: v for k, v in files.items() if k != "json"}}))
    src = Path(s.store.root / "facescans" / prof.geometry_id / prof.facescan_id) \
        if prof.facescan_id else None
    for f in (files["json"], files["png"], files["npz"]):
        if f is not None:
            s.store.record_output("defects", f, label=label, geometry_id=prof.geometry_id,
                                  source=src if src is not None and src.is_dir() else None)
    return rep

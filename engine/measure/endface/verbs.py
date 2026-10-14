"""
End-face verbs behind engine/api.py: locate a face, aim / check the ring, set the intensity,
the targeted scan, the profile, the golden and the check.

Library (append-only, rule 7; meta.json written LAST = the completeness marker):
  faces/<geometry_id>/<ts>_<label>/        face.npz (camera face mask crop, rim px, camera
                                           circle, DMD ring), preview.png, meta.json (+ the
                                           known-size check of its flood outline: size_check)
  facescans/<geometry_id>/<ts>_<label>/    masks.npz (DMD face mask, camera face mask crop),
                                           polar_frames_r<k>.npy (float32 camera DN, crop),
                                           preview.png, meta.json (names the repeat metric scans,
                                           each a normal metric/<geometry_id>/ entry; the camera
                                           exposure used and how it was chosen: auto_exposure)
  face_golden/<geometry_id>/<ts>_<label>/  golden.npz (grid; MF: radial profile, 2-D median, the
                                           reflectance FIR's 49 coefficients; band: golden, band,
                                           sigma; the placements' height maps; the rim outline's
                                           per-angle median + spread), preview.png, meta.json
                                           (params: the default detector + every setting, the
                                           resolved radii, leave-one-out, synthetic set)
Output bin: previews/ (locate overlay, ring photo + overlay + json), tuning/ (auto_intensity,
auto exposure), profiles/ (face PNG, radial + section CSV, JSON, NPZ: height map + per-pixel
repeat data + the flood pair + the rim outline), pointclouds/ (face STL + meta), defects/
(check_face JSON, overlay PNG, maps NPZ).

Fixes from the bench of 2026-10-12 (a 21 mm class cylinder whose flat end face is ~19.5 mm
across inside a rounded rim): every detector radius derives from the face diameter (radii.py);
the targeted scan picks a frame-locked camera exposure per face (exposure.py); the flood
outline's size is checked against the face diameter wherever it is known (outline.py:
size_mismatch = PULL, nothing is rescaled); the outline's roundness is checked for edge dents
(zone 'rim_outline'); a dent's depth is the 3 x 3 centre average, not its deepest point
(depth.py). The face diameter is the bright flat's: when the rim around it returns depth
points the depth silhouette is the part's larger outer circle, so locate fits its rim circle
at the silhouette's own radius (locate.py) and face_profile centres the inspected disc on
the flood outline of the flat itself (stats 'centre').

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

from ... import badparts, config, rules
from ...library.store import json_safe, read_meta, safe_label
from ...results import FaceCheck, FaceGolden, FaceProfile, FaceScan, FaceTarget, MetricScan, RingShot
from . import detectors as det_mod
from . import exposure as exposure_mod
from . import golden as golden_mod
from . import locate as locate_mod
from . import mfcfar
from . import outline as outline_mod
from . import patterns
from . import profile as profile_mod
from . import radii as radii_mod

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
RECENTRE_MIN_MM = 0.05              # face_profile: a flood outline centred further than this
RECENTRE_MAX_FRAC = 0.25            # from the located centre (and within this x the face
                                    # radius of it) re-centres the profile on the outline
RIM_KEYS = ("tol_mm", "min_arc_mm", "guard_band_pct", "k_sigma")
RIM_DEFAULTS = {"tol_mm": outline_mod.RIM_TOL_MM, "min_arc_mm": outline_mod.MIN_ARC_MM,
                "guard_band_pct": outline_mod.GUARD_BAND_PCT, "k_sigma": outline_mod.K_SIGMA}
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


def _size_tol(v) -> float:
    """The known-size tolerance in % of the nominal diameter (0 = the check is off)."""
    if v is None:
        return float(outline_mod.SIZE_TOL_PCT)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) \
            or not 0 <= v < 50:
        raise ValueError(f"size_tol_pct must be 0 (off) .. 50 % of the nominal diameter, got {v!r}")
    return float(v)


def _rim_params(d: Optional[dict], base: Optional[dict] = None) -> dict:
    """The rim outline (roundness) settings: RIM_DEFAULTS < base (a golden's) < d."""
    bad = sorted(set(d or {}) - set(RIM_KEYS))
    if bad:
        raise ValueError(f"unknown rim outline settings {bad} (known: {list(RIM_KEYS)})")
    out = {**RIM_DEFAULTS, **{k: v for k, v in (base or {}).items() if k in RIM_KEYS},
           **{k: v for k, v in (d or {}).items() if v is not None}}
    for k, v in out.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or v < 0:
            raise ValueError(f"rim outline {k} must be a number >= 0, got {v!r}")
    if not out["tol_mm"] > 0 or not out["min_arc_mm"] > 0:
        raise ValueError("rim outline tol_mm and min_arc_mm must be > 0 mm")
    if not out["guard_band_pct"] < 100:
        raise ValueError("rim outline guard_band_pct must be < 100 % of the tolerance")
    return {k: float(v) for k, v in out.items()}


def _outline_of(flood_d, cal, centre, e1, e2, radius_mm: float, origin=(0.0, 0.0),
                sat=None, behind=None) -> dict:
    """The rim outline (outline.measure) of a face in a flood difference image (lit - dark,
    its pixel (0, 0) = camera px `origin`), on the face plane through `centre` (camera
    frame, mm) spanned by e1 / e2, through the stereo calibration's camera model. sat: the
    flood's saturated pixels (rays through them are not measured); behind: the pixels the
    scan measured behind the face plane (dark to the outline)."""
    to_px = outline_mod.ray_plane_map(cal.K_cam, cal.dist_cam, np.asarray(centre, np.float64),
                                      np.asarray(e1, np.float64), np.asarray(e2, np.float64),
                                      origin)
    return outline_mod.measure(np.asarray(flood_d, np.float32), to_px, float(radius_mm),
                               sat=sat, exclude=behind)


def _capture(s, preset, pats: list, what: str, zone_mask=None, exposure_us=None) -> list:
    """flow.capture_patterns, with a per-capture camera exposure when exposure_us differs
    from the preset's: set on the open camera (camera.set_exposure_us), so the preset is not
    edited and the next rig session starts at the preset's exposure again."""
    from ...calibrate import flow
    if exposure_us is None or float(exposure_us) == float(preset.camera.exposure_us):
        return flow.capture_patterns(s, preset, pats, what, zone_mask=zone_mask)
    rules.check_led_duration(preset.projection.led, None, what)
    with s.rig(preset, zone_mask=zone_mask) as sync:
        sync.camera.set_exposure_us(float(exposure_us))
        return [sync.shoot(q) for q in pats]


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


def face_size_check(flood_d, flood_src: Optional[str], cal, r: dict, face_diameter_mm: float,
                    size_tol_pct: float, behind=None) -> tuple:
    """The known-size check of a located face (locate.locate's result r): its outline in the
    flood, measured on its plane, against the face diameter it was located with. A decode
    modulation standing in for the flood is not a size reference (a saturated face is dim in
    it): not checked then. behind: the scan's pixels measured behind the face plane.
    -> (outline | None, size_check)."""
    o, why = None, ""
    if flood_d is None or flood_src == "decode":
        why = ("the scan kept no flood pair - a decode modulation is not a size reference; "
               "scan_fast and face-locate --zone keep one, or pass flood=")
    else:
        o = _outline_of(flood_d, cal, r["centre_mm"], r["e1"], r["e2"], r["radius_mm"],
                        behind=behind)
        why = o["why"]
    sc = outline_mod.outline_size(o, float(face_diameter_mm), size_tol_pct,
                                  "face's flood outline", source=flood_src or "", why=why)
    return o, sc


def _locate(s, scan, face_diameter_mm: float, roi, params: Optional[dict], flood,
            size_tol_pct) -> dict:
    """locate_face without the library side: the scan, its calibration, the located face
    (locate.locate), the size check and the warnings."""
    known = set(locate_mod.LocateParams().to_dict())
    bad = sorted(set(params or {}) - known)
    if bad:
        raise ValueError(f"locate_face: unknown params {bad}; known: {sorted(known)}")
    prm = locate_mod.LocateParams(**(params or {}))
    tol = _size_tol(size_tol_pct)
    [ms], cal, geo, fp = _scans_and_cal(s, [scan], "locate_face")          # rule 6
    fld, cloud = ms.field, ms.cloud
    if fld.corr_x is None or fld.corr_y is None:
        raise ValueError(f"locate_face: metric scan {ms.id} has only one decoded axis; the camera -> "
                         "DMD map needs both: rescan with `scan-fast --axis xy` (or scan_metric, "
                         "which decodes x and y)")
    from ..station import verbs as station_verbs
    flood_d, flood_src, fwarn = station_verbs.resolve_flood(s, ms, flood, geo, fp)
    r = locate_mod.locate(cloud.xyz, cloud.mask, fld.corr_x, fld.corr_y, fld.mask, cal.K_cam,
                          cal.dist_cam, float(face_diameter_mm), roi, prm,
                          flood_d if prm.flood_fallback else None)
    warnings = list(r["warnings"])
    if r["rim_source"] == "flood outline":
        warnings += [f"flood: {w}" for w in fwarn]
    with np.errstate(invalid="ignore"):
        behind = np.asarray(cloud.mask, bool) & (
            (np.asarray(cloud.xyz, np.float64) - r["centre_mm"]) @ r["normal"]
            < -profile_mod.BEHIND_MM)
    o, sc = face_size_check(flood_d, flood_src, cal, r, face_diameter_mm, tol, behind)
    if sc["status"] == "mismatch":
        warnings.append(sc["message"])
    elif tol <= 0:
        warnings.append("the known-size check is OFF (size tolerance 0): a scan that reads "
                        "the wrong size is not noticed")
    # where the bright flat's outline is centred against where the face was located
    cc = outline_mod.centre_check(o, r["radius_mm"])
    if cc["status"] == "off":
        warnings.append(cc["message"] + " - the face's profile is centred on the outline "
                        "(face_profile); if the aim ring is off the flat, give the part's "
                        "outer diameter (silhouette_diameter_mm / --outer-diameter) and "
                        "locate again")
    return {"ms": ms, "cal": cal, "geo": geo, "fp": fp, "r": r, "warnings": warnings,
            "flood_src": flood_src, "flood_d": flood_d, "outline": o, "size_check": sc,
            "centre_check": cc}


def locate_face(s, scan, face_diameter_mm: float = locate_mod.FACE_DIAMETER_MM,
                label: str = "face", roi=None, params: Optional[dict] = None,
                flood=None, size_tol_pct: Optional[float] = None,
                extra: Optional[dict] = None) -> FaceTarget:
    """Metric scan of the face (both axes decoded) -> face target in faces/<geometry>/.
    flood: for the flood-circle fallback and the size check (None = the scan's own flood
    pair, else its decode modulation; or an array / file / another metric scan, as
    station_measure takes it). Known-size check: the face's outline in the flood pair,
    measured on its plane, against face_diameter_mm - more than size_tol_pct (default 4 %)
    off is a size mismatch: a warning, meta size_check / disposition PULL / pull_reasons
    (size_mismatch), and every later verb on this face says so too; nothing is rescaled.
    extra: recorded with the face target (meta 'locate_scan': how its scan was made)."""
    safe_label(label)
    return _store_face(s, _locate(s, scan, face_diameter_mm, roi, params, flood, size_tol_pct),
                       face_diameter_mm, label, extra)


def _store_face(s, L: dict, face_diameter_mm: float, label: str,
                extra: Optional[dict] = None) -> FaceTarget:
    """A located face (_locate) -> its face target in faces/<geometry>/ (+ the overlay)."""
    from ...reconstruct import metric_scan
    ms, cal, geo, fp, r = (L[k] for k in ("ms", "cal", "geo", "fp", "r"))
    # what the locate scan rests on, first (meta "rig"): a fixed rig in doubt when it was
    # made, a calibration that may not fit
    rig, doubts = metric_scan.stored_doubts(s, [ms])
    warnings = doubts + [w for w in L["warnings"] if w not in doubts]
    cloud = ms.cloud
    flood_src, sc = L["flood_src"], L["size_check"]
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
                "silhouette_diameter_mm": r.get("silhouette_diameter_mm"),
                "silhouette_source": r.get("silhouette_source"),
                "flood_source": flood_src if r["rim_source"] == "flood outline" else None,
                "flood_outline_diameter_mm": r.get("flood_outline_diameter_mm"),
                "depth_face_px": r["depth_face_px"], "search_region_px": list(r["region"]),
                "rejected_components": r["rejected_components"],
                "camera_ellipse": r["cam_ellipse"], "camera_crop_x0y0x1y1": list(box),
                "face_cam_px": int(r["cam_mask"].sum()), "mm_per_camera_px": r["mm_per_camera_px"],
                "affine": r["affine"], "dmd_centre": r["dmd_centre"], "dmd_ellipse": r["dmd_ellipse"],
                "params": r["params"], "units": UNITS_FACE, "mock": s.mock, "warnings": warnings,
                "rig": rig, "size_check": sc, "centre_check": L.get("centre_check"),
                "rim_outline": outline_mod.public(L["outline"]),
                "locate_scan": dict(extra or {})}
        if sc["status"] == "mismatch":               # never a silent pass: a PULL, said here
            meta.update(disposition="PULL", pull_reasons=[outline_mod.pull_reason(sc)])
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


# ------------------------------------------------------------ auto exposure

def _exposure_masks(full: np.ndarray, cam_ellipse: dict) -> tuple:
    """A located face's camera mask -> (the face inside 0.96 x its radius: where clipping
    is counted, its inner part inside 0.8 x the radius: where the p95 is taken)."""
    rn = patterns._norm_radius(cam_ellipse, full.shape)
    return full & (rn < exposure_mod.FACE_R), full & (rn < exposure_mod.INNER_R)


def _face_masks(t: FaceTarget, shape=None) -> tuple:
    return _exposure_masks(face_cam_mask(t, shape)[0], t.camera_ellipse)


def face_exposure(s, preset, face=None, target_dn: float = exposure_mod.TARGET_DN,
                  exposure_range_us=exposure_mod.RANGE_US,
                  max_clipped_pct: float = exposure_mod.MAX_CLIPPED_PCT,
                  margin_dmd_px: float = 6.0, label: str = "auto_exposure",
                  duty: Optional[float] = None) -> dict:
    """The frame-locked camera exposure for THIS seating of the face (exposure.py): flood
    shots lit only through the face's DMD mask, the p95 inside 0.8 x the face radius brought
    near target_dn with less than max_clipped_pct of the face clipped, among the whole
    video-frame exposures inside exposure_range_us. -> the report {ok, exposure_us, p95_dn,
    clipped_pct, shots, why, ...} (+ a JSON in output/tuning); ok False (never an
    exception) when no exposure fits: `why` says which of too dark / clipped / mirror.
    The camera exposure is set on the open camera for these shots only; the preset is not
    edited and the LED currents are never touched. Works with projection.invert presets:
    the lit flood is whichever of the 255 / 0 patterns is brighter on the face (measured)."""
    safe_label(label)
    if not 20 <= float(target_dn) <= 235:
        raise ValueError(f"target_dn must be 20..235 camera DN, got {target_dn!r}")
    if not 0 < float(max_clipped_pct) <= 20:
        raise ValueError(f"max_clipped_pct must be in (0, 20] %, got {max_clipped_pct!r}")
    cands = exposure_mod.candidates(exposure_range_us)
    p = preset
    if duty is not None:
        if not 0 < float(duty) <= 1:
            raise ValueError(f"duty must be in (0, 1], got {duty!r}")
        p = replace(p, projection=replace(p.projection, duty=float(duty)))
    t = as_face(s, face, p, "face_exposure")
    rep_ = _exposure_search(s, p, lambda shape=None: _face_masks(t, shape), t.dmd_ellipse,
                            cands, float(target_dn), float(max_clipped_pct), margin_dmd_px)
    rep_["face_id"] = t.id
    out = s.store.output_path("tuning", label, ".json")
    s.store.write_json(out, json_safe(rep_))
    s.store.record_output("tuning", out, label=label, geometry_id=p.geometry_id, source=t.path)
    rep_["file"] = out
    return rep_


def _exposure_search(s, p, masks_for, dmd_ellipse: dict, cands: list, target_dn: float,
                     max_clipped_pct: float, margin_dmd_px: float = 6.0) -> dict:
    """face_exposure's rig part: masks_for(shape | None) -> _exposure_masks of the face;
    the flood is lit through the face's DMD ellipse (+ margin)."""
    rules.check_led_duration(p.projection.led, None, "auto exposure")
    full, inner = masks_for()
    mask = patterns.face_mask(dmd_ellipse, margin_dmd_px)
    shape = patterns.DMD_SHAPE
    e0 = float(p.camera.exposure_us)
    start = min(cands, key=lambda c: abs(np.log(c / e0)))
    warnings, first, state = [], {}, {"lit": 255, "dark_dn": None}
    with s.rig(p, zone_mask=mask) as sync:
        cam = sync.camera
        if not hasattr(cam, "set_exposure_us"):
            raise RuntimeError("this camera driver cannot change its exposure: no "
                               "set_exposure_us (auto exposure needs it)")
        try:
            cam.set_exposure_us(start)
            a = sync.shoot(np.full(shape, 255, np.uint8))
            b = sync.shoot(np.zeros(shape, np.uint8))
            if a.shape != full.shape:
                full, inner = masks_for(a.shape)
            sa = exposure_mod.face_stats(a, inner, full)
            sb = exposure_mod.face_stats(b, inner, full)
            lit, dark = (sa, sb) if sa["p50_dn"] >= sb["p50_dn"] else (sb, sa)
            state.update(lit=255 if lit is sa else 0, dark_dn=dark["p50_dn"])
            first[start] = lit
            if lit["p50_dn"] - dark["p50_dn"] < 3.0:
                warnings.append(f"the face flood is not visible on the located face "
                                f"({lit['p50_dn']:.1f} vs {dark['p50_dn']:.1f} DN at "
                                f"{start / 1000:.1f} ms): is the projector on and the face "
                                "still where it was located (check_ring)?")

            def measure(e):
                if e in first:
                    return first[e]
                cam.set_exposure_us(e)
                fr = sync.shoot(np.full(shape, state["lit"], np.uint8))
                return exposure_mod.face_stats(fr, inner, full)
            rep_ = exposure_mod.choose(measure, cands, start, float(target_dn),
                                       float(max_clipped_pct))
        finally:
            cam.set_exposure_us(e0)
    rep_.update({"preset_exposure_us": p.camera.exposure_us, "range_us": [cands[0], cands[-1]],
                 "frame_us": round(config.VIDEO_FRAME_US, 3), "lit_pattern": state["lit"],
                 "dark_dn": state["dark_dn"], "inverted": bool(p.projection.invert),
                 "duty": p.projection.duty, "average": p.camera.average,
                 "preset": p.name, "inner_r": exposure_mod.INNER_R,
                 "led": "unchanged (never set by the engine)",
                 "use": "a per-capture camera exposure (the preset file is not edited)",
                 "units": {"exposure_us": "us (whole 60 Hz video frames)",
                           "p95_dn": "camera DN (8-bit, averaged frames), inside 0.8 x the "
                                     "face radius", "clipped_pct": "% of the located face at "
                                                                   "/ above 254 DN"},
                 "warnings": warnings})
    if not rep_["ok"]:
        rep_["disposition"] = "PULL"
        rep_["message"] = (f"no frame-locked exposure in {cands[0] / 1000:.1f}-"
                           f"{cands[-1] / 1000:.1f} ms gives the face a p95 of "
                           f"{rep_['p95_band_dn'][0]:.0f}-{rep_['p95_band_dn'][1]:.0f} DN with "
                           f"less than {float(max_clipped_pct):g} % clipped: {rep_['why']}")
    return rep_


def _scan_exposure(s, p, t: FaceTarget, auto_exposure, exposure_us, label: str,
                   margin_dmd_px: float, warnings: list, what: str, **kw) -> tuple:
    """The camera exposure of a targeted capture -> (exposure_us | None = the preset's,
    source 'forced' | 'auto' | 'preset', report | None). auto_exposure None = the default:
    on when the preset allows it (exposure.allowed), else off with a warning; a forced
    exposure_us wins. ExposureNotFound when auto exposure finds none (the face is not
    scanned)."""
    if exposure_us is not None:
        if auto_exposure:
            raise ValueError(f"{what}: give exposure_us (forced) or auto_exposure, not both")
        e = exposure_mod.check_forced(exposure_us)
        if not config.frame_locked(e) and not s.mock:
            warnings.append(f"exposure {e:g} us is not a whole number of 60 Hz video frames: "
                            "gray levels flicker frame to frame")
        return e, "forced", None
    ok, why_not = exposure_mod.allowed(p)
    if auto_exposure is None and not ok:
        warnings.append(f"auto exposure is off: {why_not}; scanned at the preset's exposure "
                        "(auto_exposure=True turns it on, exposure_us forces one)")
        return None, "preset", None
    if auto_exposure is not None and not auto_exposure:
        return None, "preset", None
    rep_ = face_exposure(s, p, t, label=f"{label}_exposure", margin_dmd_px=margin_dmd_px, **kw)
    if not rep_["ok"]:
        raise exposure_mod.ExposureNotFound(
            f"{what}: auto exposure - {rep_['message']} - the face is NOT scanned: PULL the "
            "part (quality inspection) / re-seat it, or force an exposure (exposure_us)", rep_)
    warnings += [f"auto exposure: {w}" for w in rep_["warnings"]]
    return float(rep_["exposure_us"]), "auto", rep_


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


def _low_repeats(reps: list) -> list:
    """The repeats of one placement that kept far fewer face points than the others
    (profile.low_repeats) -> [{repeat, metric_scan, points, others_median, pct_of_others,
    capture_s, others_capture_s}]. reps: [{metric_scan, face_points, capture_s}]."""
    out = []
    for lo in profile_mod.low_repeats([r.get("face_points") or 0 for r in reps]):
        k = lo["repeat"]
        caps = [float(r["capture_s"]) for i, r in enumerate(reps, 1)
                if i != k and r.get("capture_s") is not None]
        out.append({**lo, "metric_scan": reps[k - 1].get("metric_scan"),
                    "capture_s": reps[k - 1].get("capture_s"),
                    "others_capture_s": round(float(np.median(caps)), 2) if caps else None})
    return out


def _low_repeat_warning(lo: dict, did: str, tail: str = "") -> str:
    """One line naming the repeat and the two counts: its face points, the others' median."""
    took = ""
    if lo.get("capture_s") is not None and lo.get("others_capture_s") is not None:
        took = (f"; its capture took {float(lo['capture_s']):.1f} s, the others "
                f"{float(lo['others_capture_s']):.1f} s")
    return (f"repeat {lo['repeat']} ({lo['metric_scan']}) {did} {lo['points']} face points, "
            f"far fewer than the other repeats ({lo['others_median']:.0f}, their median: "
            f"{lo['pct_of_others']:.0f} % of theirs){took} - something disturbed it (a glint, a "
            "moved part, a busy PC); the profile uses it only where it measured" + tail
            + ": scan again if in doubt")


def targeted_face_scan(s, preset, face=None, label: str = "face", repeats: int = 3,
                       margin_dmd_px: float = 6.0, polar: bool = True, duty: Optional[float] = None,
                       steps=None, periods=None, polar_steps: int = patterns.POLAR_STEPS,
                       ring_periods=patterns.RING_PERIODS_R, spokes=patterns.SPOKES,
                       mode: str = "multifreq", code_bits: int = 5,
                       auto_exposure: Optional[bool] = None,
                       exposure_us: Optional[float] = None,
                       exposure_target_dn: float = exposure_mod.TARGET_DN,
                       exposure_range_us=exposure_mod.RANGE_US,
                       max_clipped_pct: float = exposure_mod.MAX_CLIPPED_PCT) -> FaceScan:
    """N repeats of: depth set (multifreq xy, ray-plane triangulation) + optional polar set,
    lit only through the face ellipse (+ margin). Rule 6 before the rig.
    Camera exposure (per capture, the preset is not edited): auto_exposure None = the
    default, ON when the preset's exposure is a whole number of video frames: the face's
    flood p95 (inside 0.8 x its radius) is brought near exposure_target_dn with less than
    max_clipped_pct clipped by a frame-locked exposure inside exposure_range_us
    (face_exposure; works with inverted presets); when none fits the face is NOT scanned
    (exposure.ExposureNotFound: PULL, re-seat the part). exposure_us forces one. The
    exposure used and how it was chosen are in the face scan's meta.
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
    from ...reconstruct import metric_scan as _ms
    rig_stamp = _ms._require_rig(s, p, "targeted_face_scan")                   # mobile: verified
    t = as_face(s, face, p, "targeted_face_scan")                               # rule 5
    # a fixed rig in doubt still scans (rigstate.doubt), and says so first
    warnings = [rig_stamp["warning"]] if (rig_stamp or {}).get("warning") else []
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
    sz = (t.meta or {}).get("size_check") or {}
    if sz.get("status") == "mismatch":
        warnings.append(f"face target {t.id}: {sz.get('message')}")
    # polar=False drops the rings / spokes (no added sensitivity in the research) but keeps
    # the flood pair: the MF detector's reflectance FIR needs it
    specs = patterns.polar_specs(ring_periods if polar else (), spokes if polar else (),
                                 polar_steps)
    pol = patterns.polar_patterns(t.dmd_ellipse, specs)
    full, _, _ = face_cam_mask(t)
    x0, y0, x1, y1 = box = _crop_box(full)
    # the exposure for THIS seating (flood shots through the face mask), after every refusal
    # above and before anything is written: a face no exposure fits is not scanned
    exp_us, exp_src, exp_rep = _scan_exposure(
        s, p, t, auto_exposure, exposure_us, label, float(margin_dmd_px), warnings,
        "targeted_face_scan", target_dn=float(exposure_target_dn),
        exposure_range_us=exposure_range_us, max_clipped_pct=float(max_clipped_pct))
    used_us = float(exp_us) if exp_us is not None else float(p.camera.exposure_us)
    created = _now()
    d = s.store.new_dir("facescans", p.geometry_id, label=label)
    reps, ids = [], []
    with s.store.filling(d):
        s.store.write_npz(d / "masks.npz", dmd_mask=mask, face_cam_mask=full[y0:y1, x0:x1],
                          crop=np.array(box))
        flood = None
        for k in range(int(repeats)):
            t0 = time.perf_counter()
            fr = _capture(s, p, depth_pats + pol, "a targeted end-face scan", mask, exp_us)
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
                                    "repeat": k + 1, "duty": p.projection.duty,
                                    "exposure_us": used_us, "exposure_source": exp_src,
                                    "rig": rig_stamp})
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
                                "decoded (dark or glinting face? auto exposure / auto_intensity; "
                                "moved part? check_ring)")
            reps[-1]["consistency_check"] = ms.meta.get("consistency_check")
        miss = [(k + 1, r["consistency_check"]) for k, r in enumerate(reps)
                if (r.get("consistency_check") or {}).get("warning")]
        if miss:                            # the calibration may not fit: once, the worst repeat
            worst = max(miss, key=lambda kc: kc[1].get("median_mm") or float("inf"))[1]
            warnings.append(f"repeat{'s' if len(miss) > 1 else ''} "
                            f"{', '.join(str(k) for k, _ in miss)}: {worst['warning']}")
        # a repeat that kept far fewer face points than its siblings (the bench: one of three
        # kept half and took longer, and nothing said so) is named, with both counts
        low = _low_repeats(reps)
        warnings += [_low_repeat_warning(lo, "kept") for lo in low]
        if flood is not None:
            s.store.write_preview(d / "preview.png", flood[y0:y1, x0:x1])
        meta = {"kind": "facescan", "id": d.name, "label": label, "geometry_id": p.geometry_id,
                "geometry_fp": fp, "created": created, "face_target": t.id,
                "calibration_id": cal.id, "preset": p.name, "duty": p.projection.duty,
                "invert": p.projection.invert, "exposure_us": used_us,
                "preset_exposure_us": p.camera.exposure_us, "exposure_source": exp_src,
                "auto_exposure": ({k_: v for k_, v in exp_rep.items() if k_ != "file"}
                                  if exp_rep is not None else None),
                "face_size_check": sz or None,
                "average": p.camera.average, "margin_dmd_px": float(margin_dmd_px),
                "dmd_ellipse": t.dmd_ellipse, "method": "plane",
                "depth_set": set_meta, "depth_frames": nb,
                "polar": bool(polar), "polar_patterns": specs, "crop_x0y0x1y1": list(box),
                "metric_scans": ids, "repeats": reps, "low_repeats": low,
                "mock": s.mock, "rig": rig_stamp,
                "note": "polar coords: R = 1 at the face rim (the DMD face ellipse normalised); "
                        "polar frames are float32 camera DN of the crop, in polar_patterns order; "
                        "pattern 255 lands bright when invert is false",
                "units": {"polar_frames": "camera DN (averaged)", "crop": "camera px",
                          "exposure_us": "us, the camera exposure of every frame of this scan "
                                         "(exposure_source: auto | forced | preset)"},
                "warnings": warnings}
        s.store.write_json(d / "meta.json", json_safe(meta))
    return _facescan_handle(d)


# ------------------------------------------------------------ locate scan (live)

def locate_scan(s, preset, zone: Optional[str] = None, label: str = "face_locate", steps=None,
                periods=None, exposure_us: Optional[float] = None) -> MetricScan:
    """The scan locate_face wants: the both-axis multifreq set, ray-plane triangulated, lit
    through `zone` (grid cells / an atlas region; None = the whole field) - plus the flood
    pair (flood.npz, as a fast scan keeps it), so the face's outline can be measured and its
    size checked. exposure_us: a per-capture camera exposure (the preset is not edited).
    Rule 6 before the rig."""
    from ...calibrate import flow, stereo
    from ...reconstruct import metric_scan
    from ...reconstruct.triangulate import triangulate
    safe_label(label)
    fp = config.geometry_fingerprint(preset)
    cal = stereo.require(s.store, preset.geometry_id, fp, "locate_scan")       # rule 6, pre-rig
    rig_stamp = metric_scan._require_rig(s, preset, "locate_scan")
    basis = flow.multifreq_basis(preset, cal.diamond_parity, steps, periods)
    proj_mask = cam_mask = None
    if zone:
        from ...capture import zones as zones_mod
        proj_mask, cam_mask, _cells = zones_mod.resolve(s, preset, zone)
    pats = basis.generate()
    shape = patterns.DMD_SHAPE
    floods = [np.full(shape, 255, np.uint8), np.zeros(shape, np.uint8)]
    if exposure_us is not None:
        exposure_us = exposure_mod.check_forced(exposure_us)
    t0 = time.perf_counter()
    fr = _capture(s, preset, pats + floods, "a face locate scan", proj_mask, exposure_us)
    fld = basis.decode(fr[:len(pats)], preset.geometry_id,
                       contrast_min=preset.decode.contrast_min, conf_min=preset.decode.conf_min)
    cap = round(time.perf_counter() - t0, 2)
    if cam_mask is not None:
        fld.mask = fld.mask & np.asarray(cam_mask, bool)
    cloud = triangulate(fld, cal, "plane")
    fld.meta.update({"preset": preset.name, "label": label, "mock": s.mock})
    a, b = (np.asarray(f, np.float32) for f in fr[len(pats):])
    white, black = (a, b) if float(a.mean()) >= float(b.mean()) else (b, a)    # measured
    used = float(exposure_us) if exposure_us is not None else float(preset.camera.exposure_us)
    return metric_scan._save(s, fld, cloud, label, cal, fp, preset.name,
                             {"capture_s": cap, "steps": basis.steps,
                              "periods_px": basis.periods, "diamond_parity": cal.diamond_parity,
                              "zone": zone, "rig": rig_stamp, "locate_scan": True,
                              "exposure_us": used,
                              "preset_exposure_us": preset.camera.exposure_us,
                              "flood": "white / black at the scan's exposure (flood.npz)"},
                             arrays={"flood.npz": {"white": white, "black": black}})


def _merge_locate_scans(s, preset, scans: list, label: str) -> MetricScan:
    """One metric scan from the locate scans of ONE seating made at different exposures (the
    one whose flood is kept last): per camera pixel the LONGEST exposure that did not
    saturate it (its white flood frame below SATURATED_DN) decides - the point it decoded
    there, or none: a pixel a longer exposure saw unsaturated and could not decode is not
    taken from a shorter, noisier scan (a dim face at a marginal exposure can decode with a
    wrong fringe order: a ghost patch beside the face). A dim part in front of a bright
    surround decodes at 100 ms, where the surround is saturated, and the surround at 16.7 ms,
    where the face is dark: only together do they show the face AND what it stands in front
    of (its depth rim, the surround the outline is measured against). The flood pair kept is
    the last scan's - the exposure that shows the face best. Nothing is captured."""
    from ...calibrate import stereo
    from ...decode.deformation import DeformationField
    from ...reconstruct import metric_scan
    from ...reconstruct.triangulate import MetricCloud
    safe_label(label)
    new = scans[-1]
    order = sorted(scans, key=lambda m: float(m.meta["exposure_us"]))
    f0, c0 = new.field, new.cloud
    shape = c0.mask.shape
    xyz = np.full(shape + (3,), np.nan, np.float64)
    cx = np.full(shape, np.nan, np.asarray(f0.corr_x).dtype)
    cy = np.full(shape, np.nan, np.asarray(f0.corr_y).dtype)
    conf = np.zeros(shape, np.float32)
    src = np.full(shape, -1, np.int8)
    nw = nb = None
    decided = np.zeros(shape, bool)
    for i in range(len(order) - 1, -1, -1):         # the longest exposure first
        m = order[i]
        with np.load(Path(m.path) / "flood.npz") as z:
            white, black = np.asarray(z["white"], np.float32), np.asarray(z["black"], np.float32)
        fld, cl = m.field, m.cloud
        if cl.mask.shape != shape or fld.corr_x is None or fld.corr_y is None:
            raise ValueError(f"locate scans {new.id} and {m.id} do not go together")
        mine = (white < SATURATED_DN) & ~decided    # unsaturated here, saturated at every longer
        ok = mine & np.asarray(cl.mask, bool) & np.asarray(fld.mask, bool)
        xyz[ok], cx[ok], cy[ok] = cl.xyz[ok], fld.corr_x[ok], fld.corr_y[ok]
        conf[ok] = np.asarray(fld.conf, np.float32)[ok]
        src[ok] = i
        decided |= mine
        if m is new:
            nw, nb = white, black
    mask = src >= 0
    fld = DeformationField(corr_x=cx, corr_y=cy, corr_diag=None, conf=conf, mask=mask,
                           geometry_id=f0.geometry_id, basis_names=list(f0.basis_names),
                           meta={**(f0.meta or {}), "label": label})
    cloud = MetricCloud(xyz=xyz, mask=mask, consistency=None, method=c0.method,
                        geometry_id=c0.geometry_id, calibration_id=c0.calibration_id)
    fp = config.geometry_fingerprint(preset)
    cal = stereo.require(s.store, preset.geometry_id, fp, "locate_scan")           # rule 6
    extra = {k: new.meta.get(k) for k in ("steps", "periods_px", "diamond_parity", "zone",
                                          "rig", "exposure_us", "preset_exposure_us")}
    extra.update({
        "capture_s": round(sum(float(m.meta.get("capture_s") or 0.0) for m in scans), 2),
        "locate_scan": True,
        "flood": "white / black of the newest scan (flood.npz): the exposure chosen for the face",
        "merged": {"scans": [m.id for m in order],
                   "exposures_us": [float(m.meta["exposure_us"]) for m in order],
                   "points": [int((src == i).sum()) for i in range(len(order))],
                   "rule": "per camera px the longest exposure that did not saturate it "
                           "decides (hdr.npz: source = index into scans, -1 = no point)"}})
    return metric_scan._save(s, fld, cloud, label, cal, fp, preset.name, extra,
                             arrays={"flood.npz": {"white": nw, "black": nb},
                                     "hdr.npz": {"source": src}})


def _scan_face_stats(ms: MetricScan, r: dict) -> Optional[dict]:
    """The located face's brightness in its locate scan's own flood (white frame)."""
    f = Path(ms.path) / "flood.npz"
    if not f.is_file():
        return None
    with np.load(f) as z:
        white = np.asarray(z["white"], np.float32)
    full = np.asarray(r["cam_mask"], bool)
    if white.shape != full.shape or not full.any():
        return None
    face, inner = _exposure_masks(full, r["cam_ellipse"])
    if not inner.any():
        return None
    return exposure_mod.face_stats(white, inner, face)


MAX_LOCATE_SCANS = 3                # locate_face_live: scans of one seating, at most


def _locate_is_sound(r: dict) -> bool:
    """A located face nothing speaks against: its rim from the depth step (not the flood
    outline stand-in) and the face one flat decoded patch (the camera -> DMD affine within
    locate.AFFINE_WARN_PX; a bad unwrap of a dim face is a patch that is not)."""
    aff = r.get("affine") or {}
    worst = max(float(aff.get("rms_col_px") or 0.0), float(aff.get("rms_row_px") or 0.0))
    return r.get("rim_source") != "flood outline" and worst <= locate_mod.AFFINE_WARN_PX


def _next_locate_exposure(cands: list, tried: list, stats: list, target_dn: float,
                          max_clipped_pct: float) -> Optional[float]:
    """The next locate-scan exposure from what the scans so far show of the located face
    (stats: each scan's face_stats, None = unknown): the brightest unclipped one scaled to
    the target (the flood is linear in the exposure), half the shortest when every scan
    clipped the face - the frame-locked candidate nearest that, not within 20 % of one
    already scanned. None when there is none left."""
    seen = [(e, st) for e, st in zip(tried, stats) if st is not None]
    clean = [(e, st) for e, st in seen if st["clipped_pct"] < max_clipped_pct
             and st["p95_dn"] < exposure_mod.CLIP_DN]
    if clean:
        e, st = max(clean, key=lambda t: t[1]["p95_dn"])
        est = e * float(target_dn) / max(float(st["p95_dn"]), 1.0)
    elif seen:
        est = min(tried) / 2.0
    else:
        return None
    free = [float(c) for c in cands if all(abs(c - t) > 0.2 * t for t in tried)]
    if est < min(tried):                            # never the wrong way
        free = [c for c in free if c < min(tried)]
    elif est > max(tried):
        free = [c for c in free if c > max(tried)]
    return min(free, key=lambda c: abs(c - est)) if free else None


def locate_face_live(s, preset, zone: Optional[str] = None,
                     face_diameter_mm: float = locate_mod.FACE_DIAMETER_MM, label: str = "face",
                     steps=None, periods=None, roi=None, params: Optional[dict] = None,
                     size_tol_pct: Optional[float] = None,
                     auto_exposure: Optional[bool] = None, exposure_us: Optional[float] = None,
                     exposure_target_dn: float = exposure_mod.TARGET_DN,
                     exposure_range_us=exposure_mod.RANGE_US,
                     max_clipped_pct: float = exposure_mod.MAX_CLIPPED_PCT) -> FaceTarget:
    """Scan (locate_scan) and locate the face in one go, at the exposure the face needs.
    The scan is made at the preset's exposure (or exposure_us: forced, one scan). With auto
    exposure (the default when the preset's exposure is frame-locked), when no face is found
    or the located face is too dark / clipped in the scan's own flood (p95 inside 0.8 R
    against the target, the clipped share of the face), the zone is scanned again at another
    frame-locked exposure - the one the flood measured says (linear in the exposure), a
    longer one when no face showed - and the face located on ALL the scans of this seating
    together (_merge_locate_scans: per pixel the longest exposure that did not saturate it;
    a dim part's surround saturates at the exposure its face needs, and the face is found
    against that surround). At most MAX_LOCATE_SCANS scans. A doubtful face (on its flood
    outline only, or not one flat decoded patch) found after a scan that showed none is
    scanned once more at the longest exposure (it may not be the face). The face target
    records the scans, the exposure at which its face is exposed best and every scan's
    reading (meta locate_scan). A face no exposure fits is still located (it is a position,
    not a verdict) with a warning: its targeted scan may refuse."""
    safe_label(label)
    tol = _size_tol(size_tol_pct)
    target, max_clip = float(exposure_target_dn), float(max_clipped_pct)
    notes, used, source = [], None, "preset"
    if exposure_us is not None:
        if auto_exposure:
            raise ValueError("locate_face_live: give exposure_us (forced) or auto_exposure, "
                             "not both")
        auto, used, source = False, exposure_mod.check_forced(exposure_us), "forced"
    else:
        ok, why_not = exposure_mod.allowed(preset)
        auto = ok if auto_exposure is None else bool(auto_exposure)
        if auto_exposure is None and not ok:
            notes.append(f"auto exposure is off: {why_not}")
    cands = [float(c) for c in exposure_mod.candidates(exposure_range_us)] if auto else []
    scans, tried, ms, L, stats, whys, best = [], [], None, None, [], [], None
    for attempt in range(MAX_LOCATE_SCANS):
        lab = f"{label}_locate" + ("" if attempt == 0 else str(attempt + 1))
        scans.append(locate_scan(s, preset, zone, lab, steps, periods, used))
        cur = float(scans[-1].meta["exposure_us"])
        tried.append(cur)
        last = attempt == MAX_LOCATE_SCANS - 1
        # from the second scan on: every scan of this seating together, so a longer
        # exposure for a dark face does not cost the surround the face is found against
        ms = scans[0] if len(scans) == 1 else _merge_locate_scans(
            s, preset, scans, f"{label}_locate_merged{len(scans)}")
        try:
            L = _locate(s, ms, face_diameter_mm, roi, params, None, tol)
        except locate_mod.FaceNotFound as e:
            longer = [c for c in cands if c > max(tried) * 1.2]
            if not longer or last:
                raise
            used = longer[-1] if attempt else min(longer, key=lambda c: abs(c - 2 * cur))
            source = "auto"
            notes.append(f"no face at {cur / 1000:.1f} ms ({str(e)[:80]}...): scanned again at "
                         f"{used / 1000:.1f} ms")
            continue
        # the located face in each scan's own flood: which exposure shows it best
        stats = [_scan_face_stats(m, L["r"]) for m in scans]
        whys = [None if st is None else exposure_mod.judge(st, target, max_clip) for st in stats]
        fit = [i for i, w in enumerate(whys) if w == ""]
        best = min(fit, key=lambda i: abs(stats[i]["p95_dn"] - target)) if fit else None
        if not auto or last or stats[-1] is None:
            break
        if best is not None and (len(scans) == 1 or _locate_is_sound(L["r"])):
            break
        if best is not None:
            # a doubtful face (on its flood outline only, or not one flat decoded patch)
            # after a scan that showed none: once more at the longest exposure - the face
            # then decodes cleanly and has its depth rim, or it was not the face
            nxt = cands[-1] if cands and cands[-1] > max(tried) * 1.2 else None
            why = "the face found is doubtful (flood outline only, or not one flat patch)"
        else:
            nxt = _next_locate_exposure(cands, tried, stats, target, max_clip)
            why = f"the face was {whys[-1]}"
        if nxt is None:
            break
        used, source = float(nxt), "auto"
        notes.append(f"{why} at {cur / 1000:.1f} ms: scanned again at {used / 1000:.1f} ms")
    j = best if best is not None else len(scans) - 1
    if len(scans) > 1:
        if scans[j] is not scans[-1]:
            # the flood kept with the merged scan is the scan's that exposes the face best
            # (the size check and the outline read it); nothing is captured
            try:
                ms2 = _merge_locate_scans(s, preset, [m for m in scans if m is not scans[j]]
                                          + [scans[j]], f"{label}_locate_merged")
                L, ms = _locate(s, ms2, face_diameter_mm, roi, params, None, tol), ms2
            except locate_mod.FaceNotFound:
                pass                                # the locate above stands, with its flood
        notes.append(f"located on {len(scans)} locate scans of this seating merged per pixel ("
                     + ", ".join(f"{t / 1000:.1f}" for t in tried)
                     + " ms: the longest exposure that did not saturate the pixel)")
    face_us = float(tried[j])
    if auto and best is None and whys and whys[j]:
        notes.append(f"the face is still {whys[j]} at {face_us / 1000:.1f} ms: no exposure "
                     "scanned shows it well - its targeted scan may refuse this seating")
    L["warnings"] = list(L["warnings"]) + [n for n in notes if "still" in n]
    shots = [{"exposure_us": float(e), "scan": m.id,
              **({k: st[k] for k in ("p95_dn", "p50_dn", "clipped_pct")} if st else {}),
              "ok": w == "", "why": w if w is not None else "the face's flood was not read"}
             for e, m, st, w in zip(tried, scans, stats or [None] * len(scans),
                                    whys or [None] * len(scans))]
    lo, hi = exposure_mod.band(target)
    return _store_face(s, L, face_diameter_mm, label,
                       {"scan": ms.id, "zone": zone, "scans": [m.id for m in scans],
                        "exposure_us": face_us,
                        "preset_exposure_us": preset.camera.exposure_us,
                        "exposure_source": source, "auto_exposure_on": bool(auto),
                        "auto_exposure": ({"ok": best is not None, "exposure_us": face_us,
                                           "shots": shots, "target_dn": target,
                                           "p95_band_dn": [round(lo, 1), round(hi, 1)],
                                           "max_clipped_pct": max_clip,
                                           "candidates_us": cands,
                                           "read_from": "the locate scans' own flood frames "
                                                        "(p95 inside 0.8 x the face radius)"}
                                          if auto else None),
                        "notes": notes})


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


def _profile_outline(prof: FaceProfile, cal) -> Optional[dict]:
    """The rim outline of a profile's placement in its flood pair (measured once, kept with
    its per-pixel data): None without a flood pair / the crop's place in the camera frame."""
    px = prof._pixels
    if px is None or not px.has_flood:
        return None
    if px.outline is None:
        if px.origin is None:
            return None
        n = np.asarray(prof.normal, np.float64)
        e1, e2 = locate_mod.face_axes(n)
        px.outline = _outline_of(px.flood_diff(), cal, np.asarray(prof.centre_mm, np.float64),
                                 e1, e2, prof.face_diameter_mm / 2, px.origin, px.flood_sat,
                                 px.behind)
    return px.outline


def _why_no_outline(prof: FaceProfile) -> str:
    px = prof._pixels
    if px is None:
        return "a profile without per-pixel data"
    if not px.has_flood:
        return "no flood pair (a profile from metric scans, or made before its targeted scan)"
    if px.origin is None:
        return "a profile stored without its crop's place in the camera frame"
    return (px.outline or {}).get("why") or "not measured"


def face_profile(s, scans=None, face=None, label: Optional[str] = None,
                 grid_mm: float = profile_mod.GRID_MM,
                 r_flat_mm: Optional[float] = profile_mod.R_FLAT_MM,
                 stl: bool = True, size_tol_pct: Optional[float] = None) -> FaceProfile:
    """r_flat_mm: the flatness radius; None = 0.855 x the face radius (radii.py: 9.0 mm on a
    21.05 mm face). A value given is used as given (it must lie on the face: refused
    otherwise, never clamped). The placement's rim outline is measured in the targeted
    scan's flood pair (stats rim_outline; the per-angle residuals are in the .npz) and its
    diameter checked against the face diameter (stats size_check; more than size_tol_pct
    off: a warning - check_face PULLs it).
    The CENTRE of the face frame is the flood outline's when the whole outline shows (the
    bright flat itself: stats 'centre' says how far that is from the located circle's
    centre, the profile is computed about it), else the located circle's."""
    from ...reconstruct import metric_scan
    tol = _size_tol(size_tol_pct)
    ms, cal, geo, fp, t, fs_id, warnings = _profile_inputs(s, scans, face)
    # what the scans rest on, first: a fixed rig in doubt when they were made, a calibration
    # that may not fit (stats "rig"); the profile is made as before
    rig, doubts = metric_scan.stored_doubts(s, ms)
    warnings[:0] = [w for w in doubts if w not in warnings]
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
    fs = as_facescan(s, fs_id) if fs_id else None
    fl = _floods(fs, [m.id for m in ms], (x0, y0, x1, y1), warnings)

    def analyse(centre_px):
        """The profile about a centre (crop px; None = the face mask's), its per-pixel
        data and its rim outline in the flood pair, on this placement's own plane."""
        res_ = profile_mod.analyse(X, K, cm, (x0, y0), cal, radius, grid_mm, r_flat_mm,
                                   centre_px=centre_px)
        pxd = res_["pixels"]
        px_ = mfcfar.Pixels(iy=pxd["iy"], ix=pxd["ix"], U=pxd["U"], V=pxd["V"], H=pxd["H"],
                            cam_mask=cm, flood_on=fl.get("flood_on"),
                            flood_off=fl.get("flood_off"), flood_sat=fl.get("flood_sat"),
                            origin=(x0, y0), behind=res_["behind"])
        o_ = None
        if fl:
            o_ = px_.outline = _outline_of(px_.flood_diff(), cal, res_["centre"], res_["e1"],
                                           res_["e2"], radius, (x0, y0), px_.flood_sat,
                                           px_.behind)
        return res_, px_, o_

    res, pixels, o = analyse((e["cx"] - x0, e["cy"] - y0) if clipped else None)
    # the centre: the bright flat's own outline, when the whole of it shows. The located
    # circle is the depth silhouette's (a rim that returns depth points makes it the part's
    # outer circle, and a fixed-radius fit slides): the inspected disc belongs on the flat
    cc = outline_mod.centre_check(o, radius)
    centre = {"source": "located circle", "located_offset_mm": cc["offset_mm"],
              "located_offset_uv_mm": cc["offset_uv_mm"], "tol_mm": cc["tol_mm"],
              "status": cc["status"], "message": cc["message"]}
    if cc["offset_mm"] is not None and cc["offset_mm"] > RECENTRE_MIN_MM:
        if clipped or cc["offset_mm"] > RECENTRE_MAX_FRAC * radius:
            centre["message"] = cc["message"] + (
                " - NOT re-centred (the face leaves the camera frame)" if clipped else
                f" - NOT re-centred (more than {RECENTRE_MAX_FRAC:g} x the face radius: not "
                "the face that was located?)")
        else:
            from ...capture.mock_stereo import distort_project
            du, dv = cc["offset_uv_mm"]
            C1 = np.asarray(res["centre"], np.float64) + du * np.asarray(res["e1"]) \
                + dv * np.asarray(res["e2"])
            p1 = distort_project(C1[None], np.asarray(cal.K_cam, np.float64), cal.dist_cam)[0]
            res, pixels, o = analyse((float(p1[0]) - x0, float(p1[1]) - y0))
            left = outline_mod.centre_check(o, radius, "profile centre")
            centre.update(source="flood outline", status=(
                "recentred" if left["status"] == "ok" else left["status"]),
                residual_offset_mm=left["offset_mm"],
                message=(f"centred on the face's flood outline: {cc['offset_mm']:.2f} mm from "
                         "the located circle's centre"))
            if cc["status"] == "off":
                warnings.append(cc["message"] + " - the profile is centred on the outline; "
                                "locate the face again if the aim ring is off the flat (a rim "
                                "that returns depth points: give the part's outer diameter, "
                                "silhouette_diameter_mm / --outer-diameter)")
    if centre["status"] == "off":
        warnings.append(centre["message"] + ": the inspected zone is not centred on the flat "
                        "(check_face PULLs it) - locate the face again")
    st = res["stats"]
    st["centre"] = centre
    st["rig"] = rig
    st["flood_pair"] = bool(fl)
    st["flood_saturated_pct"] = mfcfar.saturated_pct(pixels)
    # the rim outline: its size (the one dimension known beforehand) and its roundness (an
    # edge dent moves the outline)
    st["rim_outline"] = outline_mod.public(o)
    lsz = (t.meta or {}).get("size_check") or {}
    if (o is None or not o["ok"]) and lsz.get("measured_mm") is not None and tol > 0:
        # no outline in this placement's flood (a face located off its place is lit only in
        # part): the size its face target read at locate still stands
        sc = outline_mod.size_check(lsz["measured_mm"], t.face_diameter_mm, tol,
                                    "face's flood outline",
                                    source=f"face target {t.id} (its locate scan)",
                                    axes=lsz.get("axes_mm"))
    else:
        sc = outline_mod.outline_size(
            o, t.face_diameter_mm, tol, "face's flood outline",
            source="the targeted scan's flood pair" if fl else "",
            why=(o["why"] if o is not None else "no flood pair"))
    st["size_check"] = sc
    if sc["status"] == "mismatch":
        warnings.append(sc["message"])
    elif tol <= 0:
        warnings.append("the known-size check is OFF (size tolerance 0): a scan that reads "
                        "the wrong size is not noticed")
    if lsz.get("status") == "mismatch" and sc["status"] != "mismatch":
        warnings.append(f"face target {t.id}: {lsz.get('message')}")
    if o is not None and (o.get("saturated_pct") or 0) > 1.0:
        warnings.append(f"the flood is saturated on {o['saturated_pct']:.0f} % of the rim: the "
                        "outline is not measured there (auto exposure keeps the face below "
                        "clipping)")
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
    # what each repeat gave the profile: one that measured far fewer face points than the
    # others is named with both counts (whatever its face scan said: an older scan, or plain
    # metric scans, said nothing)
    st["low_repeats"] = _low_repeats([
        {"metric_scan": m.id, "face_points": n, "capture_s": (m.meta or {}).get("capture_s")}
        for m, n in zip(ms, st["repeat_face_points"])])
    for lo in st["low_repeats"]:
        warnings.append(_low_repeat_warning(
            lo, "gave the profile",
            f" ({st['repeats_used']} of {len(ms)} repeats measured the face; "
            f"{st['all_repeats_pct'] or 0:.0f} % of the face points were measured by all "
            f"{len(ms)})"))
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
    rim_arrays = {}
    if o is not None and o["ok"]:
        rim_arrays = {"rim_theta_deg": np.asarray(o["theta_deg"], np.float32),
                      "rim_rho_mm": np.asarray(o["rho_mm"], np.float32),
                      "rim_residual_mm": np.asarray(o["residual_mm"], np.float32)}
    s.store.write_npz(npz, grid_mm=res["grid_mm"], height_mm=res["height_mm"].astype(np.float32),
                      u=res["u"].astype(np.float32), v=res["v"].astype(np.float32),
                      h=res["h"].astype(np.float32), centre=res["centre"], normal=res["normal"],
                      e1=res["e1"], e2=res["e2"], crop=np.array([x0, y0, x1, y1]),
                      meta=np.array(json.dumps(json_safe(pmeta))), **rim_arrays,
                      **pixels.arrays())
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
           "units": {**UNITS_MAP, "stats": "mm (tilt deg, coverage %)",
                     "rim_outline": "mm; residual = radius - fitted outline at theta (- = "
                                    "inside: a notch)"}, "warnings": warnings,
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


def _sources(s, geometry_id: str, facescan_id, metric_scan_ids, files=()) -> list:
    """What a placement rests on, for engine/badparts.require_good: its face scan's entry
    (check_face records its result there), its repeat metric scans, its own files."""
    fs = s.store.root / "facescans" / str(geometry_id) / str(facescan_id or "")
    out = [fs] if facescan_id and fs.is_dir() else []
    out += [m for m in (metric_scan_ids or []) if isinstance(m, str)]
    return out + [Path(f) for f in files if f]


def _stored_settings(gh: FaceGolden, g: np.ndarray, over: Optional[dict] = None) -> tuple:
    """(Settings, default detector) from a golden's params (+ per-check trigger overrides).
    The radii are the golden's: its stored resolved set (params['radii']); a golden made
    before the radii derived from the face keeps the radii its detectors stored (they are
    its settings), and whatever it did not store derives from its face diameter."""
    prm = gh.params or {}
    nested = isinstance(prm.get("band"), dict)
    over = dict(over or {})
    R = gh.face_diameter_mm / 2
    band = _band_params(prm["band"] if nested else prm)
    trig = _trig_params({**((prm.get("trigger") or {}) if nested else prm),
                         **{k: v for k, v in over.items() if k == "min_area_mm2"}})
    mfo = {k: v for k, v in (prm.get("mf") or {}).items() if k in mfcfar.MF_DEFAULTS}
    mfo.update({k: v for k, v in over.items() if k in ("z_thresh", "min_depth_mm", "min_area_mm2")})
    firo = {k: v for k, v in (prm.get("fir") or {}).items() if k in mfcfar.FIR_DEFAULTS}
    if prm.get("fir") and "edge_px" not in firo:
        firo["edge_px"] = 0     # its coefficients were fitted with the mask's border pixels in
    stored = prm.get("radii") if isinstance(prm.get("radii"), dict) else None
    rr = None
    if stored is not None and all(k in stored for k in radii_mod.KEYS):
        rr = radii_mod.face_radii(R, {k: stored[k] for k in radii_mod.KEYS})
        rr["given"] = list(stored.get("given") or [])
    derived = rr if rr is not None else radii_mod.face_radii(
        R, {k: mfo[k] for k in mfcfar.MF_RADII if mfo.get(k) is not None})
    st = det_mod.Settings(g=g, radius_mm=R, band=golden_mod.resolved(band, derived), trig=trig,
                          mf=mfcfar.mf_params(mfo, radii=derived),
                          fir=mfcfar.fir_params(firo, radii=derived), radii=rr)
    return st, (prm.get("detector") or "band")


def face_golden(s, placements: list, label: str = "golden", params: Optional[dict] = None,
                synthetic=True, detector: str = det_mod.DEFAULT_DETECTOR, fir: bool = True,
                mf_params: Optional[dict] = None, radii: Optional[dict] = None,
                rim: Optional[dict] = None, size_tol_pct: Optional[float] = None) -> FaceGolden:
    """Golden face from >= 2 (3+ for leave-one-out) GOOD placements: every detector's model
    (MF radial profile + reflectance FIR, band median + tolerance band), the default detector
    and all settings in the record; leave-one-out false triggers (every detector) and the
    synthetic-dent set (the default detector).
    radii: explicit radii (radii.KEYS: inspect_r_mm, rim_r_mm, plane_r_mm, ...); every one
    not given is a fraction of the face radius, and rim_r_mm is ONE value for every detector
    (params['rim_r_mm'], the band's former own rim radius, sets the same one). A radius
    given is used as given - also 9.75 / 9.0, the former fixed defaults - or refused when it
    does not fit the face (off the face, a rim radius outside the inspected zone): never
    dropped, never clamped. The resolved set is stored (params['radii']).
    rim: the rim outline (roundness) settings {tol_mm 0.4, min_arc_mm 1.5, guard_band_pct 20,
    k_sigma 4}; the golden learns the outline's per-angle median and spread from the
    placements whose flood pair shows it. size_tol_pct (4): a placement whose outline is
    further off the face diameter is refused (a golden is never built on mis-scaled data).
    A placement of a part judged REJECT or PULL - by an inspection, by check_face, or one that
    lives in a bad-part output - is refused (engine/badparts.py: references are made from
    good parts only)."""
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
    mfcfar.check_mf_keys(mf_params)
    bad = sorted(set(radii or {}) - set(radii_mod.KEYS))
    if bad:
        raise ValueError(f"face_golden: unknown radii {bad} (known: {list(radii_mod.KEYS)})")
    rimp = _rim_params(rim)
    tol = _size_tol(size_tol_pct)
    # good parts only: what is given (a face scan, a profile file), then what each rests on
    badparts.require_good(s.store, "face_golden placement",
                          [p for p in placements if not isinstance(p, FaceProfile)])
    profs = [load_profile(s, p) for p in placements]
    badparts.require_good(s.store, "face_golden placement",
                          [x for p in profs for x in _sources(s, p.geometry_id, p.facescan_id,
                                                              p.metric_scan_ids,
                                                              [p.files.get("npz")])], sweep=True)
    b = _check_same(profs, "face_golden")
    cal = _require_current_cal(s, b["geometry_id"], b["geometry_fp"], b["calibration_id"],
                               "face_golden")
    tp = _trig_params(params)
    g = profs[0]._grid
    radius = b["face_diameter_mm"] / 2
    # ONE rim radius for every detector: radii['rim_r_mm'], else the band's rim_r_mm (a
    # value given is a value given), else 0.855 x the face radius
    rover = {k: v for k, v in (radii or {}).items() if v is not None}
    radii_notes = []
    band_rim = (params or {}).get("rim_r_mm")
    if band_rim is not None:
        if "rim_r_mm" in rover and float(rover["rim_r_mm"]) != float(band_rim):
            raise ValueError(f"face_golden: rim_r_mm given twice ({band_rim!r} in the band "
                             f"params, {rover['rim_r_mm']!r} in the radii): there is ONE rim "
                             "radius for every detector")
        rover["rim_r_mm"] = float(band_rim)
        radii_notes.append("rim_r_mm from the band params: the one rim radius of every detector")
    bp = replace(_band_params(params), rim_r_mm=None)
    st = det_mod.settings(g, radius, bp, tp, mf_params, None, rover)
    mprm = st.mf
    pls = [_placement(p) for p in profs]
    names = [p.label for p in profs]
    warnings = []
    # the known size: a golden is never built on placements that read the wrong size
    outlines = [_profile_outline(p, cal) for p in profs]
    sizes = [outline_mod.outline_size(o, b["face_diameter_mm"], tol, "face's flood outline",
                                      source="the targeted scan's flood pair",
                                      why=_why_no_outline(p)) for p, o in zip(profs, outlines)]
    wrong = [(p.label, z) for p, z in zip(profs, sizes) if z["status"] == "mismatch"]
    if wrong:
        raise ValueError(f"face_golden: placement {wrong[0][0]}: {wrong[0][1]['message']} - "
                         "not a GOOD placement for a golden"
                         + (f" (and {len(wrong) - 1} more)" if len(wrong) > 1 else ""))
    unchecked = [p.label for p, z in zip(profs, sizes) if z["status"] == "not_checked"]
    if unchecked and tol > 0:
        warnings.append(f"size not checked on {unchecked} ({sizes[names.index(unchecked[0])]['message']})")
    if tol <= 0:
        warnings.append("the known-size check is OFF (size tolerance 0, stored in this "
                        "golden): placements and checks that read the wrong size are not "
                        "noticed")
    # ... nor on a placement inspected about a centre that is not the flat's
    off = [(p.label, z) for p, z in ((p, outline_mod.centre_check(o, radius, "profile centre"))
                                     for p, o in zip(profs, outlines)) if z["status"] == "off"]
    if off:
        raise ValueError(f"face_golden: placement {off[0][0]}: {off[0][1]['message']} - run "
                         "face_profile on its face scan again (it centres the profile on "
                         "the outline) or locate the face again"
                         + (f" (and {len(off) - 1} more)" if len(off) > 1 else ""))
    # the rim outline: the per-angle median + spread of the good placements
    rim_model = outline_mod.golden_model(outlines)
    rim_loo = []
    n_round = sum(1 for o in outlines if o is not None and o["round_ok"])
    if rim_model is None:
        warnings.append(f"no rim outline in the golden ({n_round} of {len(profs)} placements "
                        "show the whole outline in a flood pair; 2 needed): checks judge a "
                        "placement's own outline when it has one, edge dents are not checked "
                        "otherwise")
    else:
        if n_round < len(profs):
            warnings.append(f"rim outline learned from {n_round} of {len(profs)} placements "
                            "(the others show no whole outline in a flood pair)")
        good_o = [(p.label, o) for p, o in zip(profs, outlines)
                  if o is not None and o["round_ok"]]
        for i, (lab, o) in enumerate(good_o):
            others = outline_mod.golden_model([x for j, (_, x) in enumerate(good_o) if j != i]) \
                if len(good_o) >= 3 else None
            j = outline_mod.judge(o, rimp["tol_mm"], rimp["min_arc_mm"], rimp["guard_band_pct"],
                                  others, rimp["k_sigma"])
            rim_loo.append({"placement": lab, "status": j["status"], "why": j.get("why") or "",
                            "unjudged": j.get("unjudged") or [],
                            "rms_mm": round(float(o["rms_mm"]), 4),
                            "max_abs_mm": round(float(o["max_abs_mm"]), 4),
                            "vs_others_max_abs_mm": None if j["max_abs_mm"] is None
                            else round(float(j["max_abs_mm"]), 4),
                            "diameter_mm": round(float(o["diameter_mm"]), 4),
                            "findings": j["findings"], "marginal": j["marginal"]})
        bad_rim = [r["placement"] for r in rim_loo if r["status"] != "ok"]
        if bad_rim:
            warnings.append(f"rim outline: {bad_rim} are not round within {rimp['tol_mm']:g} mm "
                            f"over {rimp['min_arc_mm']:g} mm of arc among the GOOD placements "
                            "(or part of their rim could not be judged) - check those parts "
                            "(an edge dent? a glint on the rim?) or raise the rim tolerance")
        wide = float(np.max(rimp["k_sigma"] * rim_model["spread_mm"]))
        if wide > rimp["tol_mm"]:
            widest = min(wide, outline_mod.MAX_TOL_FACTOR * rimp["tol_mm"])
            warnings.append(f"rim outline: the good placements differ by up to "
                            f"{float(np.max(rim_model['spread_mm'])):.3f} mm (1 sigma) at one "
                            f"angle: the tolerance there is {widest:.2f} mm instead of "
                            f"{rimp['tol_mm']:g}")
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
    spread = golden_mod.placement_spread(np.stack([p.height for p in pls]), g,
                                         st.band.rim_r_mm)
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
    params_rec = {**st.to_dict(det, use_fir), "fir": fir_d, "rim": rimp, "size_tol_pct": tol}
    params_rec["radii"]["notes"] = radii_notes
    rim_rec = None
    if rim_model is not None:
        rim_rec = {"n_placements": rim_model["n_placements"],
                   "diameter_mm": round(rim_model["diameter_mm"], 4),
                   "diameter_spread_mm": round(rim_model["diameter_spread_mm"], 4),
                   "rms_mm": rim_model["rms_mm"], "max_abs_mm": rim_model["max_abs_mm"],
                   "spread_median_mm": round(float(np.median(rim_model["spread_mm"])), 4),
                   "spread_max_mm": round(float(np.max(rim_model["spread_mm"])), 4),
                   "leave_one_out": rim_loo, **rimp,
                   "note": "per-angle median + spread of the good placements' outline "
                           "residuals (golden.npz rim_*): a check takes its residual against "
                           "the median and a tolerance of at least k_sigma x the spread"}
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
        if rim_model is not None:
            arrays.update(rim_theta_deg=rim_model["theta_deg"].astype(np.float32),
                          rim_median_mm=rim_model["median_mm"].astype(np.float32),
                          rim_spread_mm=rim_model["spread_mm"].astype(np.float32))
        s.store.write_npz(d / "golden.npz", **arrays)
        png = _write_bytes(d / "preview.png", det_mod.golden_png(model, st, loo0))
        meta = {"kind": "face_golden", "id": d.name, "label": label, **b, "created": created,
                "n_placements": len(profs), "detector": det,
                "placements": [{"label": p.label, "face_id": p.face_id, "facescan_id": p.facescan_id,
                                "metric_scans": p.metric_scan_ids,
                                "flood_saturated_pct": sat.get(p.label)} for p in profs],
                "params": params_rec, "placement_spread": spread, "rim_outline": rim_rec,
                "size_check": [{"placement": p.label, **{k: z[k] for k in (
                    "status", "measured_mm", "deviation_pct")}} for p, z in zip(profs, sizes)],
                "leave_one_out": loo, "leave_one_out_false_triggers": compare,
                "sensitivity": sens or {}, "mock": s.mock,
                "units": {"golden_mm": "mm, face frame (+ toward the camera)", "band_mm": "mm",
                          "mf_radial_h_mm": "mm at mf_radial_r_mm (mm)",
                          "fir_coef": "mm per unit high-passed log flood (7x7 camera px)",
                          "z": "dent significance (CFAR sigma units)",
                          "radii": "mm from the face centre (params.radii: the resolved set)",
                          "rim_median_mm": "mm at rim_theta_deg (deg, 0 = camera right, 90 = "
                                           "camera down)", **UNITS_MAP},
                "note": "detector mf_fir / mf: MF-CFAR (mfcfar.py) against the median radial "
                        "profile of the placements (+ the reflectance FIR fitted on them); band: "
                        "median map + max(k_sigma x sigma, floor), rim r > rim_r_mm x rim_factor. "
                        "Every radius is a fraction of the face radius unless given "
                        "(params.radii). Bound to geometry + fingerprint, face diameter, grid "
                        "and the stereo calibration",
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
                 "fir_coef": z["fir_coef"].astype(np.float64) if "fir_coef" in z.files else None,
                 "rim": ({"theta_deg": z["rim_theta_deg"].astype(np.float64),
                          "median_mm": z["rim_median_mm"].astype(np.float64),
                          "spread_mm": z["rim_spread_mm"].astype(np.float64)}
                         if "rim_median_mm" in z.files else None)}
    return g, model


CHECK_OVERRIDES = ("min_area_mm2", "z_thresh", "min_depth_mm", "rim_tol_mm", "rim_min_arc_mm",
                   "rim_guard_band_pct", "size_tol_pct")


def _rim_triggers(j: dict) -> list:
    """A judged outline's notches / bulges as check_face triggers (zone 'rim_outline')."""
    out = []
    for f in j["findings"]:
        out.append({"zone": outline_mod.RIM_ZONE, "kind": f["kind"], "r_mm": f["r_mm"],
                    "theta_deg": f["theta_deg"], "u_mm": f["u_mm"], "v_mm": f["v_mm"],
                    # + = the outline moved IN (a notch: material missing, like a dent)
                    "depth_mm": round(-f["depth_mm"], 3), "dev_mm": f["depth_mm"],
                    "arc_mm": f["arc_mm"], "width_mm": f["arc_mm"], "tol_mm": f["tol_mm"],
                    "z": None, "area_mm2": None})
    return out


def check_face(s, placement, golden=None, label: Optional[str] = None,
               params: Optional[dict] = None, detector=None) -> FaceCheck:
    """Placement vs golden -> triggers, PASS / FAIL / INCOMPLETE, overlay. detector: None =
    the golden's, a name in detectors.DETECTORS, or a callable(dev_map, band, params) trigger
    function (the band detector with that function).
    Besides the dents on the inspected flat (zone face / rim):
      * the rim outline (zone 'rim_outline', outline.py): a notch / bulge of the flood
        outline beyond the rim tolerance over min_arc_mm of rim is a trigger (FAIL); an arc
        within the guard band of the tolerance, an outline that shows but cannot be judged
        all the way round (a stretch of rim with no edge: a chip deep enough leaves none,
        a glint hides one), or an outline the golden holds but this placement does not
        show, is never passed (INCOMPLETE);
      * the centre: a placement whose profile centre is not the centre of its flood
        outline (a profile made before face_profile centred on the outline, a face that
        leaves the frame) is inspected off the flat: never passed (INCOMPLETE);
      * the known size: the outline's diameter against the golden's face diameter - more
        than size_tol_pct off is a SIZE MISMATCH: no verdict of this check can be trusted
        (not its FAILs either), so it is INCOMPLETE with disposition PULL, reason
        size_mismatch; nothing is rescaled.
    The disposition (CONTINUE / REJECT / PULL) and its pull_reasons are in stats (and at the
    top of the JSON); stats['radii'] holds the radii the check used. params: per-check
    overrides (CHECK_OVERRIDES), the rest belongs to the golden."""
    prof = load_profile(s, placement)
    gh = as_golden(s, golden, prof)
    what = f"check_face vs golden {gh.id}"
    # never against a golden whose placements were judged REJECT / PULL since (engine/badparts)
    badparts.require_good(s.store, f"{what}: its placement",
                          [gh.path] + [x for p in (gh.meta or {}).get("placements") or []
                                       if isinstance(p, dict)
                                       for x in _sources(s, gh.geometry_id, p.get("facescan_id"),
                                                         p.get("metric_scans"))],
                          how=badparts.HOW_USE)
    rules.require_same_geometry(prof.geometry_id, gh.geometry_id, what)
    rules.require_same_fingerprint(prof.geometry_fp, gh.meta.get("geometry_fp"), what)
    if abs(prof.face_diameter_mm - gh.face_diameter_mm) > DIAMETER_TOL_MM:
        raise ValueError(f"{what}: the golden is for a {gh.face_diameter_mm:g} mm face, the placement "
                         f"{prof.face_diameter_mm:g} mm")
    if prof.calibration_id != gh.calibration_id:
        raise rules.GeometryMismatch(f"{what}: the golden was made under stereo calibration "
                                     f"{gh.calibration_id}, the placement under {prof.calibration_id} "
                                     "(rule 6)")
    cal = _require_current_cal(s, gh.geometry_id, gh.meta.get("geometry_fp"), gh.calibration_id,
                               what)
    g, model = _golden_model(gh)
    if g.shape != prof._grid.shape or not np.allclose(g, prof._grid):
        raise ValueError(f"{what}: the placement's grid ({prof.grid_mm:g} mm) is not the golden's")
    over = {k: v for k, v in (params or {}).items() if v is not None}
    extra = set(over) - set(CHECK_OVERRIDES)
    if extra:
        raise ValueError(f"check_face: unknown params {sorted(extra)} (the other settings belong "
                         "to the golden)")
    gprm = gh.params or {}
    rimp = _rim_params({k: over.get(f"rim_{k}") for k in ("tol_mm", "min_arc_mm",
                                                          "guard_band_pct")}, gprm.get("rim"))
    tol = _size_tol(over["size_tol_pct"] if "size_tol_pct" in over else gprm.get("size_tol_pct"))
    st, default_det = _stored_settings(gh, g, over)
    det = detector if detector is not None else default_det
    name = det_mod.check_name(det)
    if not callable(det) and det != "band" and model["mf"] is None:
        raise ValueError(f"{what}: golden {gh.id} was built before the MF detector (band only): "
                         "check with detector='band' or build a new golden")
    res = det_mod.quiet(det_mod.check)(det, _placement(prof), model, st)
    trig, stt = list(res["triggers"]), res["stats"]
    label = label or prof.label
    safe_label(label)
    warnings = list(res["warnings"])
    pulls = []                                          # [{reason, message}]: the doubts
    # ---- the rim outline (edge dents) and the known size
    o = _profile_outline(prof, cal)
    j = outline_mod.judge(o, rimp["tol_mm"], rimp["min_arc_mm"], rimp["guard_band_pct"],
                          model.get("rim"), rimp["k_sigma"])
    rim_trig = _rim_triggers(j)
    face_meta = {}
    try:
        face_meta = as_face(s, prof.face_id, None, what).meta or {}
    except (ValueError, OSError):
        pass                                            # a profile whose face target is gone
    lsz = face_meta.get("size_check") or {}
    if o is not None and o["ok"]:
        sc = outline_mod.outline_size(o, gh.face_diameter_mm, tol, "face's flood outline",
                                      source="the placement's flood outline")
    elif lsz.get("measured_mm") is not None and tol > 0:
        # no outline in this placement's flood: the size its face target read at locate
        sc = outline_mod.size_check(lsz["measured_mm"], gh.face_diameter_mm, tol,
                                    "face's flood outline",
                                    source=f"face target {prof.face_id} (its locate scan)",
                                    axes=lsz.get("axes_mm"))
    else:
        sc = outline_mod.outline_size(None, gh.face_diameter_mm, tol, "face's flood outline",
                                      why=_why_no_outline(prof))
    if sc["status"] == "ok" and lsz.get("status") == "mismatch":
        warnings.append(f"face target {prof.face_id} read a size mismatch at locate "
                        f"({lsz.get('deviation_pct')} %); this placement's own outline is "
                        "within tolerance - locate the face again")
    stt.update({"radii": st.radii_public(), "size_check": sc,
                "rim_outline": outline_mod.public(o, j) if o is not None else
                {"measured": False, "roundness_measured": False, "status": j["status"],
                 "why": _why_no_outline(prof), "findings": [], "marginal": [],
                 **{k: j[k] for k in ("tol_mm", "min_arc_mm", "guard_band_pct", "golden")}},
                "n_rim_outline": len(rim_trig)})
    all_trig = trig + rim_trig
    verdict = det_mod.verdict(all_trig, stt)
    if verdict == "INCOMPLETE":
        msg = (f"only {stt['compared_pct'] or 0:.0f}% of the inspected face measured "
               f"(< {det_mod.MIN_COMPARED_PCT:g}%): no dent found in what was measured, "
               "but the face is NOT passed (re-scan: auto exposure, check_ring)")
        warnings.append(msg)
        pulls.append({"reason": "incomplete", "message": msg})
    elif stt["compared_pct"] is not None and stt["compared_pct"] < 99:
        warnings.append(f"{100 - stt['compared_pct']:.1f}% of the inspected face unmeasured "
                        "(holes cannot trigger)")
    has_model = model.get("rim") is not None
    if j["status"] == "not_measured":
        why = _why_no_outline(prof) if o is None or not o["ok"] else (o.get("why") or "")
        if has_model:
            msg = (f"the rim outline could not be judged ({why}): the golden checks it, so "
                   "the face is NOT passed (edge dents unchecked)")
            warnings.append(msg)
            pulls.append({"reason": "incomplete", "message": msg})
        else:
            warnings.append(f"rim outline not checked ({why}): edge dents are not looked for")
    elif j["status"] == "incomplete":
        # the flood shows the outline, but not all of it could be judged: a chip deep
        # enough leaves no edge, a glint hides one - never passed, golden outline or not
        msg = (f"the rim outline could not be judged all the way round "
               f"({j.get('why') or (o or {}).get('why') or '?'}): the face is NOT passed "
               "(edge dents unchecked there; re-seat / re-scan the part)")
        warnings.append(msg)
        pulls.append({"reason": "incomplete", "message": msg})
    elif j["status"] == "marginal":
        m0 = j["marginal"][0]
        msg = (f"rim outline: a {m0['kind']} of {abs(m0['depth_mm']):.2f} mm over {m0['arc_mm']:g} "
               f"mm of rim at {m0['theta_deg']:g} deg is within {rimp['guard_band_pct']:g} % of "
               f"the tolerance ({m0['tol_mm']:g} mm): not passed")
        warnings.append(msg)
        pulls.append({"reason": "guard_band", "message": msg})
    elif not has_model and j["status"] in ("ok", "fail"):
        warnings.append("rim outline judged against its own fit (the golden holds no rim "
                        "outline: built from placements without a flood pair)")
    # the centre the placement was inspected about against its flood outline's own
    cc = outline_mod.centre_check(o, gh.face_diameter_mm / 2, "profile centre")
    stt["centre_check"] = cc
    if cc["status"] == "off":
        msg = (cc["message"] + ": the inspected zone is not centred on the flat (it reaches "
               "the rim on one side) - run face_profile on the face scan again (it centres "
               "the profile on the outline) or locate the face again; NOT passed")
        warnings.append(msg)
        pulls.append({"reason": "invalid", "message": msg})
    if sc["status"] == "mismatch":
        # nothing this check says can be trusted: not its PASS, not its FAIL
        warnings.append(sc["message"])
        pulls.insert(0, outline_mod.pull_reason(sc))
        if all_trig:
            warnings.append(f"{len(all_trig)} trigger(s) found, but on a mis-scaled measurement: "
                            "not a definite FAIL (PULL)")
    elif cc["status"] == "off" and all_trig:
        warnings.append(f"{len(all_trig)} trigger(s) found, but about a centre that is not "
                        "the flat's: not a definite FAIL (PULL)")
    if sc["status"] == "mismatch":
        pass
    elif sc["status"] == "not_checked" and tol > 0:
        warnings.append(sc["message"])
    elif tol <= 0:
        warnings.append("the known-size check is OFF (size tolerance 0"
                        + ("" if "size_tol_pct" in over else ", stored in the golden")
                        + "): a scan that reads the wrong size is not noticed")
    # a size mismatch or a profile inspected off the flat: its triggers are not definite
    # either (the rim inside the inspected zone triggers on a good part)
    if sc["status"] == "mismatch" or cc["status"] == "off" or (verdict == "PASS" and pulls):
        verdict = "INCOMPLETE"
    # a definite FAIL (a trigger on a trusted measurement) is a REJECT whatever else is in
    # doubt; any doubt without one is a PULL; CONTINUE only for a PASS with no doubt at all
    disposition = {"PASS": "CONTINUE", "FAIL": "REJECT"}.get(verdict, "PULL")
    stt["disposition"] = disposition
    stt["pull_reasons"] = pulls if disposition == "PULL" else []
    placement_d = {"label": prof.label, "face_id": prof.face_id, "facescan_id": prof.facescan_id,
                   "metric_scans": prof.metric_scan_ids, "profile_npz": prof.files.get("npz")}
    files = {}
    out = s.store.output_path("defects", f"{label}_face_check", ".json")
    files["png"] = _write_bytes(out.with_suffix(".png"),
                                det_mod.check_png({**res, "triggers": all_trig}, st, verdict))
    files["npz"] = out.with_suffix(".npz")
    maps = {k: np.asarray(v, np.float32) for k, v in res["maps"].items()}
    if o is not None and o["ok"]:
        maps.update(rim_theta_deg=np.asarray(o["theta_deg"], np.float32),
                    rim_residual_mm=np.asarray(o["residual_mm"], np.float32))
    s.store.write_npz(files["npz"], grid_mm=g, dev_mm=res["dev"].astype(np.float32), **maps)
    files["json"] = out
    mf_used = name in ("mf", "mf_fir")
    params_used = {"detector": name, **({"mf": {k: (list(v) if isinstance(v, tuple) else v)
                                               for k, v in st.mf.items()},
                                         "fir_applied": stt.get("fir_applied")} if mf_used else
                                        {**st.band.to_dict(), **st.trig.to_dict()}),
                   "min_compared_pct": det_mod.MIN_COMPARED_PCT, "radii": st.radii_public(),
                   "rim": rimp, "size_tol_pct": tol}
    units = {**UNITS_MAP, "dev_mm": "mm, placement - golden (- = below: a dent)"}
    units.update({"depth_mm": "mm, THE dent depth = depth_centre_mm (+ = below the surface)",
                  "depth_centre_mm": "mm, mean of the 3 x 3 grid cells at the dent's deepest "
                                     "place vs the local background (thresholds apply to it)",
                  "depth_min_mm": "mm, deepest single grid point (single point, over-reads)",
                  "width_half_depth_mm": "mm, diameter at half depth (reads ~1 camera px wide)",
                  "depth_fit_mm": "mm, the matched filter's paraboloid amplitude",
                  "width_mm": "mm, best dent diameter of the filter bank (rim_outline: the arc)",
                  "z": "dent significance (CFAR sigma units)",
                  "noise_floor_mm": "mm, 1-sigma dent depth per diameter"} if mf_used
                 else {"band_mm": "mm", "depth_centre_mm": "mm, mean of the 3 x 3 grid cells at "
                                                           "the peak vs the local background",
                       "depth_min_mm": "mm, deepest single grid point (over-reads)"})
    units["rim_outline"] = ("zone rim_outline: depth_mm = how far the flood outline moved in "
                            "(+, a notch) or out (-, a bulge), mm; arc_mm along the rim")
    rep = FaceCheck(verdict=verdict, triggers=all_trig, n_triggers=len(all_trig), golden_id=gh.id,
                    placement=placement_d, stats=stt, params=params_used, detector=name,
                    units=units, files=files, warnings=warnings, disposition=disposition,
                    pull_reasons=list(stt["pull_reasons"]))
    s.store.write_json(out, json_safe({k: v for k, v in vars(rep).items() if k != "files"}
                                      | {"files": {k: v for k, v in files.items() if k != "json"}}))
    src = Path(s.store.root / "facescans" / prof.geometry_id / prof.facescan_id) \
        if prof.facescan_id else None
    for f in (files["json"], files["png"], files["npz"]):
        if f is not None:
            s.store.record_output("defects", f, label=label, geometry_id=prof.geometry_id,
                                  source=src if src is not None and src.is_dir() else None)
    return rep

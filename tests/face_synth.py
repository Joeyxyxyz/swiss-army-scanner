"""Synthetic end-face placements for the end-face tests (no rig, no capture path).

A cell lying with its circular END FACE toward the camera (engine.capture.mock_stereo.EndFace:
a disc with an optional rim chamfer and cosine-bowl dents, + its cylindrical body behind it)
in front of the wall, seen by the mock bench's pinhole camera + projector (StereoRig.bench,
square DMD grid). Only a camera crop around the face is ray-cast (fast): the true surface
points + noise along the camera ray, and the true projector (col, row) + noise, written
exactly as the library writes metric scans (metric/<geo>/<entry>: cloud.npz, field.npz,
meta.json) next to the rig's own K / R / t as the geometry's stereo calibration
(calibrations/<geo>/<entry>/stereo_calibration.json).

Optionally a REFLECTANCE texture (refl_texture): the flood pair of the face scan shows it, and
each metric scan carries the height error it causes (k x the camera-x gradient of the log
reflectance: the phase decode is pulled toward the brighter side), identical in every repeat -
what the MF detector's reflectance FIR learns and removes.

Other faces: shapes / placement take radius= (a 16, 19.5 or 30 mm face) and rim_notches=
(edge dents: the outline moved in / out); flood_image() renders the targeted scan's flood of
a face (bright disc, dark surround, anti-aliased) for write_facescan(flood=...), so the rim
outline (size, roundness) is in the flood pair as on the rig.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from engine.calibrate.stereo import StereoCalibration
from engine.capture.mock_stereo import EndFace, MetricCylinder, Plane, StereoRig, undistort_normalized

GEO = "synthetic_face_rig"
FP = {"rotate90": False, "size": [912, 1140], "basis_axis": "x", "projector_side": "right",
      "baseline_mm": 279.4, "standoff_mm": 823.0}
RADIUS = 10.525
DIAMETER = 2 * RADIUS
CAM = (1080, 1440)


def rig():
    return StereoRig.bench(parity=None)


def write_calibration(lib: Path, rg, name: str = "20260101T000000_000000_stereo") -> StereoCalibration:
    d = Path(lib) / "calibrations" / GEO / name
    d.mkdir(parents=True)
    tr = rg.truth()
    cal = StereoCalibration(id=f"{GEO}/{name}", geometry_id=GEO, K_cam=tr["K_cam"],
                            dist_cam=tr["dist_cam"], K_proj=tr["K_proj"], dist_proj=np.zeros(5),
                            R=tr["R"], t=np.ravel(tr["t"]), diamond_parity=None,
                            fingerprint=dict(FP), path=d,
                            created=datetime.now().isoformat(timespec="seconds"))
    (d / "stereo_calibration.json").write_text(json.dumps(cal.to_json()))
    return cal


def normal_from_tilt(tilt_x_deg: float, tilt_y_deg: float) -> np.ndarray:
    """Face normal toward the camera, tilted about the camera y (x-lean) and x (y-lean) axes."""
    n = np.array([np.tan(np.radians(tilt_x_deg)), np.tan(np.radians(tilt_y_deg)), -1.0])
    return n / np.linalg.norm(n)


def shapes(center, normal, dents=(), chamfer=(0.6, 0.15), body_mm: float = 30.0,
           radius: float = RADIUS, rim_notches=()) -> list:
    c = np.asarray(center, float)
    n = np.asarray(normal, float)
    face = EndFace(center=tuple(c), normal=tuple(n), radius=radius, dents=list(dents),
                   chamfer_mm=chamfer[0], chamfer_depth_mm=chamfer[1],
                   rim_notches=list(rim_notches))
    body = MetricCylinder(center=tuple(c - n * body_mm / 2), radius=radius, height=body_mm,
                          axis=tuple(n))
    return [face, body, Plane()]


def crop_box(rg, center, margin_mm: float = 28.0) -> tuple:
    c = np.asarray(center, float)
    pts = np.array([rg.camera.project(c + np.array([dx, dy, 0.0]))
                    for dx in (-margin_mm, margin_mm) for dy in (-margin_mm, margin_mm)])
    x0, y0 = np.floor(pts.min(0)).astype(int)
    x1, y1 = np.ceil(pts.max(0)).astype(int)
    return max(int(x0), 0), max(int(y0), 0), min(int(x1), CAM[1]), min(int(y1), CAM[0])


def _trace(objs, o, d):
    t_best = np.full(d.shape[0], np.inf)
    idx = np.full(d.shape[0], -1)
    for i, s in enumerate(objs):
        t, _ = s.intersect(o, d)
        better = t < t_best
        t_best = np.where(better, t, t_best)
        idx[better] = i
    return t_best, idx


def render(rg, objs, box):
    """-> (X (h, w, 3) mm, uv (h, w, 2) native DMD, ok (h, w), which shape)."""
    x0, y0, x1, y1 = box
    v, u = np.mgrid[y0:y1, x0:x1].astype(np.float64)
    xy = undistort_normalized(np.stack([u, v], -1), rg.camera.K, rg.camera.dist)
    d = np.concatenate([xy, np.ones(xy.shape[:-1] + (1,))], -1).reshape(-1, 3)
    t, idx = _trace(objs, np.zeros(3), d)
    hit = np.isfinite(t)
    X = d * np.where(hit, t, 0)[:, None]
    Cp = rg.projector_center
    toX = np.where(hit[:, None], X - Cp, [0.0, 0.0, 1.0])
    L = np.linalg.norm(toX, axis=1)
    tp, _ = _trace(objs, Cp, toX / L[:, None])
    lit = hit & (tp >= L - 0.02)                                  # projector sees it first
    uv = rg.projector.project(np.where(hit[:, None], X, 1.0) @ rg.R.T + rg.t)
    h, w = y1 - y0, x1 - x0
    return X.reshape(h, w, 3), uv.reshape(h, w, 2), lit.reshape(h, w), idx.reshape(h, w)


def refl_texture(seed: int, sigma_px: float = 1.2, amp: float = 0.35) -> np.ndarray:
    """A full-frame camera reflectance texture (~1, log-normal, correlated over sigma_px)."""
    import cv2
    z = cv2.GaussianBlur(np.random.default_rng(seed).normal(0, 1, CAM), (0, 0), sigma_px)
    return np.exp(amp * z / z.std())


def face_refl(rg, refl: np.ndarray, center, normal, box, radius: float = RADIUS,
              keep: float = 0.85, fade_px: float = 2.5) -> np.ndarray:
    """refl with its texture kept on the face and faded out toward its rim (half of it at
    keep x radius, ~5 % at the rim, none beyond): the test texture is far stronger than a
    real face's (the FIR needs something to learn), and near the rim it would notch the
    flood outline (as it is here the outline of a good face is as rough as on the bench:
    0.06-0.10 mm rms). -> a full-frame texture, 1 outside the crop `box`."""
    x0, y0, x1, y1 = box
    w = flood_image(rg, center, normal, box, radius=keep * radius, level_dn=1.0, dark_dn=0.0,
                    blur_px=fade_px, noise_dn=0.0)
    out = np.ones_like(refl)
    out[y0:y1, x0:x1] = np.exp(w * np.log(refl[y0:y1, x0:x1]))
    return out


def refl_error(refl: np.ndarray, box, k_mm: float) -> np.ndarray:
    """Height error (mm, + toward the camera) in the crop: k_mm x d log(refl) / d x_cam."""
    x0, y0, x1, y1 = box
    return k_mm * np.gradient(np.log(refl[y0:y1, x0:x1]), axis=1)


def write_metric(lib: Path, rg, objs, box, name: str, seed: int, cal, noise_mm: float = 0.025,
                 corr_noise_px: float = 0.03, refl=None, refl_k_mm: float = 0.0) -> str:
    """One metric scan (library layout) of the crop: truth + noise along the camera ray
    (+ the reflectance-induced height error when refl is given)."""
    rng = np.random.default_rng(seed)
    x0, y0, x1, y1 = box
    X, uv, ok, _ = render(rg, objs, box)
    ray = X / np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), 1e-9)
    err = rng.normal(0, noise_mm, ok.shape)
    if refl is not None and refl_k_mm:
        err = err - refl_error(refl, box, refl_k_mm)          # + toward the camera = shorter ray
    Xn = X + ray * err[..., None]
    xyz = np.full(CAM + (3,), np.nan, np.float32)
    mask = np.zeros(CAM, bool)
    cx = np.full(CAM, np.nan, np.float32)
    cy = np.full(CAM, np.nan, np.float32)
    xyz[y0:y1, x0:x1] = np.where(ok[..., None], Xn, np.nan)
    mask[y0:y1, x0:x1] = ok
    cx[y0:y1, x0:x1] = np.where(ok, uv[..., 0] + rng.normal(0, corr_noise_px, ok.shape), np.nan)
    cy[y0:y1, x0:x1] = np.where(ok, uv[..., 1] + rng.normal(0, corr_noise_px, ok.shape), np.nan)
    d = Path(lib) / "metric" / GEO / name
    d.mkdir(parents=True)
    np.savez_compressed(d / "cloud.npz", xyz=xyz, mask=mask)
    hdr = json.dumps({"geometry_id": GEO, "basis_names": ["multifreq"], "meta": {}})
    np.savez_compressed(d / "field.npz", corr_x=cx, corr_y=cy, mask=mask,
                        conf=mask.astype(np.float32), _header=np.array(hdr))
    meta = {"id": name, "label": name, "geometry_id": GEO, "geometry_fp": dict(FP),
            "created": datetime.now().isoformat(timespec="seconds"), "calibration_id": cal.id,
            "method": "plane", "units": "mm", "n_points": int(mask.sum()), "zone": "synthetic"}
    (d / "meta.json").write_text(json.dumps(meta))
    return name


def flood_image(rg, center, normal, box, radius: float = RADIUS, rim_notches=(),
                level_dn: float = 170.0, dark_dn: float = 8.0, blur_px: float = 0.7,
                ss: int = 4, noise_dn: float = 1.0, seed: int = 0, glint=None) -> np.ndarray:
    """The targeted scan's flood of the camera crop `box`: the face disc (radius, with
    rim_notches as mock_stereo.EndFace takes them) bright at level_dn on a dark surround
    (only the face is lit), anti-aliased (ss x ss rays per camera pixel through the camera
    model, so the outline is true to a fraction of a pixel), blurred and noisy.
    glint: (u_mm, v_mm, radius_mm) a saturated spot on the face."""
    import cv2
    x0, y0, x1, y1 = box
    face = EndFace(center=tuple(center), normal=tuple(normal), radius=radius,
                   rim_notches=list(rim_notches))
    c, n, e1, e2 = face.frame()
    o = (np.arange(ss) + 0.5) / ss - 0.5
    v, u = np.mgrid[y0:y1, x0:x1].astype(np.float64)
    cover = np.zeros(u.shape)
    spot = np.zeros(u.shape)
    for dy in o:
        for dx in o:
            xy = undistort_normalized(np.stack([u + dx, v + dy], -1), rg.camera.K, rg.camera.dist)
            d = np.concatenate([xy, np.ones(xy.shape[:-1] + (1,))], -1)
            t = (c @ n) / (d @ n)
            w = d * t[..., None] - c
            wu, wv = w @ e1, w @ e2
            inside = np.hypot(wu, wv) <= face.rim_radius(np.arctan2(wv, wu))
            cover += inside
            if glint is not None:
                spot += inside & (np.hypot(wu - glint[0], wv - glint[1]) <= glint[2])
    img = dark_dn + (level_dn - dark_dn) * cover / ss ** 2 + 400.0 * spot / ss ** 2
    if blur_px:
        img = cv2.GaussianBlur(img, (0, 0), blur_px)
    rng = np.random.default_rng(seed)
    return np.clip(img + rng.normal(0, noise_dn, img.shape), 0, 255)


def placement(lib: Path, rg, cal, name: str, center, normal, dents=(), repeats: int = 2,
              seed: int = 0, refl=None, refl_k_mm: float = 0.0, **kw) -> tuple:
    """A locate scan + `repeats` repeat scans of one placement -> (locate id, [repeat ids]).
    margin_mm: the camera crop around the centre (default 28: a face up to ~30 mm)."""
    margin = kw.pop("margin_mm", 28.0)
    objs = shapes(center, normal, dents, **kw)
    box = crop_box(rg, center, margin)
    loc = write_metric(lib, rg, objs, box, f"20260101T0100{seed % 100:02d}_000000_{name}_locate",
                       seed * 10, cal)
    reps = [write_metric(lib, rg, objs, box, f"20260101T0200{seed % 100:02d}_{k:06d}_{name}_r{k + 1}",
                         seed * 10 + k + 1, cal, refl=refl, refl_k_mm=refl_k_mm)
            for k in range(repeats)]
    return loc, reps


def write_facescan(lib: Path, name: str, face_id: str, metric_ids: list, refl=None, box=None,
                   flood_dn: float = 90.0, dark_dn: float = 6.0, flood=None) -> str:
    """A face-scan entry (as targeted_face_scan writes it) naming the repeat metric scans of
    one placement and its face target; with refl + box, polar frames holding only the flood
    pair (255: flood_dn x the reflectance, 0: dark_dn) of that camera crop; with flood + box
    (flood_image()'s picture of the crop: the face's outline) the 255 flood is that image -
    times the reflectance when refl is given too (the lit face with its texture on a dark
    surround: what a targeted scan shows; refl alone lights the whole crop, the shadow of
    the part included)."""
    d = Path(lib) / "facescans" / GEO / name
    d.mkdir(parents=True)
    np.savez_compressed(d / "masks.npz", dmd_mask=np.zeros((1140, 912), bool))
    specs, crop = [], None
    if refl is not None or flood is not None:
        specs = [{"kind": "flood", "level": 255}, {"kind": "flood", "level": 0}]
        x0, y0, x1, y1 = crop = [int(v) for v in box]
        rng = np.random.default_rng(len(name))
        for k in range(len(metric_ids)):
            if flood is not None:
                on = np.asarray(flood, np.float64)
                if refl is not None:
                    on = on * refl[y0:y1, x0:x1] / float(np.mean(refl[y0:y1, x0:x1]))
            else:
                on = flood_dn * refl[y0:y1, x0:x1] / float(np.mean(refl[y0:y1, x0:x1]))
            on = np.clip(on + rng.normal(0, 0.5, on.shape), 0, 255)
            off = dark_dn + rng.normal(0, 0.5, on.shape)
            np.save(d / f"polar_frames_r{k + 1}.npy", np.stack([on, off]).astype(np.float32))
    meta = {"kind": "facescan", "id": name, "label": name, "geometry_id": GEO,
            "geometry_fp": dict(FP), "created": datetime.now().isoformat(timespec="seconds"),
            "face_target": face_id, "metric_scans": list(metric_ids),
            "repeats": [{"metric_scan": m} for m in metric_ids], "polar_patterns": specs,
            "crop_x0y0x1y1": crop, "method": "plane", "mock": True}
    (d / "meta.json").write_text(json.dumps(meta))
    return name

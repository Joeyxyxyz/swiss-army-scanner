"""
Stereo (camera + projector) calibration: camera intrinsics, projector
intrinsics, then cv2.stereoCalibrate with both fixed -> R, t. Stored as a
geometry-bound calibration (rule 6) in the library:

    calibrations/<geometry_id>/<ts>_stereo/
        report.json               per-pose + per-axis reprojection, truth diff (mock)
        stereo_calibration.json   K_cam, dist_cam, K_proj, dist_proj, R, t, parity,
                                  geometry_fp — written LAST (the completeness marker)

Frames: X_proj = R @ X_cam + t, mm. Projector pixels are native DMD
(col, row) with the diamond row parity the poses were decoded with.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .. import rules
from .camera import calibrate_camera
from .charuco import BoardSpec, detect
from .projector import calibrate_projector, corners_to_projector

STEREO_FILE = "stereo_calibration.json"


@dataclass
class StereoCalibration:
    id: str
    geometry_id: str
    K_cam: np.ndarray
    dist_cam: np.ndarray
    K_proj: np.ndarray
    dist_proj: np.ndarray
    R: np.ndarray                    # X_proj = R X_cam + t (mm)
    t: np.ndarray
    cam_size: tuple = (1440, 1080)
    proj_size: tuple = (912, 1140)
    diamond_parity: Optional[int] = None
    rms: dict = field(default_factory=dict)
    board: dict = field(default_factory=dict)
    n_poses: int = 0
    fingerprint: Optional[dict] = None
    path: Optional[Path] = None
    created: str = ""

    @property
    def baseline_mm(self) -> float:
        return float(np.linalg.norm(self.R.T @ self.t))

    @property
    def projector_center(self) -> np.ndarray:
        return -self.R.T @ self.t

    @property
    def angle_deg(self) -> float:
        """Angle between the optical axes."""
        zc = np.array([0.0, 0.0, 1.0])
        zp = self.R.T @ zc
        return math.degrees(math.acos(np.clip(zc @ zp, -1, 1)))

    def to_json(self) -> dict:
        return {"id": self.id, "geometry_id": self.geometry_id, "kind": "stereo",
                "K_cam": self.K_cam.tolist(), "dist_cam": np.ravel(self.dist_cam).tolist(),
                "K_proj": self.K_proj.tolist(), "dist_proj": np.ravel(self.dist_proj).tolist(),
                "R": self.R.tolist(), "t": np.ravel(self.t).tolist(),
                "cam_size": list(self.cam_size), "proj_size": list(self.proj_size),
                "diamond_parity": self.diamond_parity, "rms": self.rms, "board": self.board,
                "n_poses": self.n_poses, "geometry_fp": self.fingerprint,
                "baseline_mm": self.baseline_mm, "angle_deg": self.angle_deg,
                "created": self.created, "units": "mm; projector px = native DMD (col, row)",
                "note": "X_proj = R X_cam + t; valid only at this geometry_id + fingerprint"}

    @classmethod
    def from_json(cls, d: dict, path: Optional[Path] = None) -> "StereoCalibration":
        a = lambda k: np.asarray(d[k], np.float64)          # noqa: E731
        return cls(id=d["id"], geometry_id=d["geometry_id"], K_cam=a("K_cam"),
                   dist_cam=a("dist_cam"), K_proj=a("K_proj"), dist_proj=a("dist_proj"),
                   R=a("R"), t=a("t"), cam_size=tuple(d.get("cam_size", (1440, 1080))),
                   proj_size=tuple(d.get("proj_size", (912, 1140))),
                   diamond_parity=d.get("diamond_parity"), rms=d.get("rms") or {},
                   board=d.get("board") or {}, n_poses=d.get("n_poses", 0),
                   fingerprint=d.get("geometry_fp"), path=path, created=d.get("created", ""))


def calibrate(poses: list, spec: BoardSpec, cam_size=(1440, 1080), proj_size=(912, 1140),
              projector_distortion: bool = False, half_win: int = 18) -> dict:
    """poses: [{"image": flood uint8, "corr_x", "corr_y"}] (one per board pose)
    -> {"cam": ..., "proj": ..., "R", "t", "rms_stereo", "per_pose": [...]}"""
    import cv2
    det = [detect(p["image"], spec) for p in poses]
    used = [i for i, d in enumerate(det) if d is not None]
    if len(used) < 3:
        raise ValueError(f"only {len(used)} of {len(poses)} poses show the ChArUco board "
                         f"({spec.squares_x}x{spec.squares_y}, {spec.dictionary}); need >= 3")
    cam = calibrate_camera([det[i] for i in used], cam_size)
    det, n_outliers = _drop_corner_outliers(det, used, cam)
    if n_outliers:
        cam = calibrate_camera([det[i] for i in used], cam_size)
    cam["corner_outliers_dropped"] = n_outliers
    obj_v, cam_v, prj_v, per_pose = [], [], [], []
    for k, i in enumerate(used):
        corners, ids, obj = det[i]
        pp, ok, hrms = corners_to_projector(corners, poses[i]["corr_x"], poses[i]["corr_y"],
                                            cam["K"], cam["dist"], half_win=half_win)
        per_pose.append({"pose": i, "corners": int(len(ids)), "projector_corners": int(ok.sum()),
                         "camera_rms_px": cam["per_pose_rms"][k],
                         "homography_rms_px": float(np.nanmedian(hrms)) if ok.any() else None})
        if ok.sum() >= 6:
            obj_v.append(obj[ok])
            cam_v.append(corners[ok])
            prj_v.append(pp[ok])
    proj = calibrate_projector(obj_v, prj_v, proj_size, distortion=projector_distortion)
    flags = cv2.CALIB_FIX_INTRINSIC
    rms, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
        [np.asarray(o, np.float32).reshape(-1, 1, 3) for o in obj_v],
        [np.asarray(c, np.float32).reshape(-1, 1, 2) for c in cam_v],
        [np.asarray(p, np.float32).reshape(-1, 1, 2) for p in prj_v],
        cam["K"], cam["dist"], proj["K"], proj["dist"], tuple(cam_size), flags=flags)
    return {"cam": cam, "proj": proj, "R": R, "t": T.reshape(3), "rms_stereo": float(rms),
            "per_pose": per_pose, "n_poses": len(obj_v), "n_detected": len(used),
            "corner_outliers_dropped": cam["corner_outliers_dropped"]}


OUTLIER_MIN_PX, OUTLIER_MEDIAN_X = 1.0, 4.0


def _drop_corner_outliers(det: list, used: list, cam: dict) -> tuple:
    """Drop single ChArUco corners whose reprojection error is > max(1 px,
    4 x the median corner error): a corner misread under glare or a partly
    hidden marker (bench 2026-09-28: 5 px misses on otherwise 0.3 px boards)
    must not drag K. Views keep >= 6 corners or are left as they were."""
    import cv2
    errs = {}
    for k, i in enumerate(used):
        corners, ids, obj = det[i]
        pr, _ = cv2.projectPoints(np.asarray(obj, np.float32).reshape(-1, 1, 3),
                                  cam["rvecs"][k], cam["tvecs"][k], cam["K"], cam["dist"])
        errs[i] = np.linalg.norm(pr.reshape(-1, 2) - np.asarray(corners).reshape(-1, 2), axis=1)
    thr = max(OUTLIER_MIN_PX, OUTLIER_MEDIAN_X * float(np.median(np.concatenate(list(errs.values())))))
    out, dropped = list(det), 0
    for i, e in errs.items():
        keep = e <= thr
        if keep.all() or keep.sum() < 6:
            continue
        corners, ids, obj = det[i]
        out[i] = (np.asarray(corners)[keep], np.asarray(ids)[keep], np.asarray(obj)[keep])
        dropped += int((~keep).sum())
    return out, dropped


def compare_truth(cal: StereoCalibration, truth: dict) -> dict:
    """Mock only: how far the fit is from the bench's true K / R / t."""
    dR = cal.R @ np.asarray(truth["R"]).T
    ang = math.degrees(math.acos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
    kc, kp = np.asarray(truth["K_cam"]), np.asarray(truth["K_proj"])
    return {"cam_f_px": [float(cal.K_cam[0, 0] - kc[0, 0]), float(cal.K_cam[1, 1] - kc[1, 1])],
            "cam_c_px": [float(cal.K_cam[0, 2] - kc[0, 2]), float(cal.K_cam[1, 2] - kc[1, 2])],
            "proj_f_px": [float(cal.K_proj[0, 0] - kp[0, 0]), float(cal.K_proj[1, 1] - kp[1, 1])],
            "proj_c_px": [float(cal.K_proj[0, 2] - kp[0, 2]), float(cal.K_proj[1, 2] - kp[1, 2])],
            "R_deg": ang, "t_mm": float(np.linalg.norm(cal.t - np.asarray(truth["t"]))),
            "units": "fit - truth"}


# ----------------------------------------------------------------- library

def save(store, geometry_id: str, fit: dict, *, fingerprint: Optional[dict], spec: BoardSpec,
         parity: Optional[int], created: str, extra_report: Optional[dict] = None,
         truth: Optional[dict] = None) -> StereoCalibration:
    from ..library.store import json_safe
    d = store.new_dir("calibrations", geometry_id, label="stereo")
    cal = StereoCalibration(
        id=f"{geometry_id}/{d.name}", geometry_id=geometry_id, K_cam=fit["cam"]["K"],
        dist_cam=fit["cam"]["dist"], K_proj=fit["proj"]["K"], dist_proj=fit["proj"]["dist"],
        R=np.asarray(fit["R"], float), t=np.asarray(fit["t"], float),
        cam_size=tuple(fit["cam"]["image_size"]), proj_size=tuple(fit["proj"]["image_size"]),
        diamond_parity=parity,
        rms={"camera_px": fit["cam"]["rms"], "projector_px": fit["proj"]["rms"],
             "projector_u_px": fit["proj"]["rms_u"], "projector_v_px": fit["proj"]["rms_v"],
             "stereo_px": fit["rms_stereo"], "camera_ok": fit["cam"]["ok"],
             "projector_ok": fit["proj"]["ok"]},
        board=spec.to_dict(), n_poses=fit["n_poses"], fingerprint=fingerprint, path=d,
        created=created)
    report = {"per_pose": fit["per_pose"], "camera_per_pose_rms": fit["cam"]["per_pose_rms"],
              "projector_per_pose_rms": fit["proj"]["per_pose_rms"],
              "fv_over_fu": fit["proj"]["fv_over_fu"], **(extra_report or {})}
    if truth is not None:
        report["vs_truth"] = compare_truth(cal, truth)
    store.write_json(d / "report.json", json_safe(report))
    store.write_json(d / STEREO_FILE, json_safe(cal.to_json()))       # LAST: marker
    return cal


def latest(store, geometry_id: str, fingerprint: Optional[dict] = None) -> Optional[StereoCalibration]:
    """Newest complete stereo calibration at this geometry (and fingerprint)."""
    for d in reversed(store.entries("calibrations", geometry_id)):
        f = d / STEREO_FILE
        if not f.is_file():
            continue
        try:
            cal = StereoCalibration.from_json(json.loads(f.read_text(encoding="utf-8")), d)
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if cal.geometry_id == geometry_id and not rules.fingerprint_diffs(cal.fingerprint, fingerprint):
            return cal
    return None


def require(store, geometry_id: str, fingerprint: Optional[dict], what: str) -> StereoCalibration:
    """Rule 6 for metric (mm) output: a stereo calibration at this geometry_id
    AND fingerprint, read from the library now — else refuse."""
    cal = latest(store, geometry_id, fingerprint)
    if cal is None:
        raise rules.GeometryMismatch(
            f"{what}: no stereo calibration for geometry '{geometry_id}' at these geometry-bound "
            "settings, so mm are refused (rule 6). Run calibrate_stereo at this geometry first.")
    rules.require_same_geometry(cal.geometry_id, geometry_id, what)
    rules.require_same_fingerprint(cal.fingerprint, fingerprint, what)
    return cal

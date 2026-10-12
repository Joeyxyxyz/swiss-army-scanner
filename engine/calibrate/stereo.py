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

Fail closed: calibrate() raises CalibrationRejected (nothing is saved) when the
fit is not a usable calibration - too few poses gave projector points, the
projector intrinsics are degenerate (a focal length <= 0 or absurd, f_v / f_u
far from the DMD's), or the stereo RMS is beyond any sane bound. Seen on the
rig: fringes too fine for a printed board mapped 0-11 of ~50 corners per pose,
the projector fit collapsed (f_u ~ 0, stereo RMS 164 px) and it was reported
as a calibration.

A re-homed calibration (rehome()) is a NEW entry: the same intrinsics and t,
R1 = R0 @ Rd for a small camera rotation Rd fitted by the rig check, flagged
`rehomed` with the calibration it came from. The original is never changed.
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

# what makes a fit a calibration at all (calibrate() refuses anything else)
MIN_PROJECTOR_CORNERS = 6           # per pose: corners with a decoded projector position
MIN_PROJECTOR_POSES = 3             # poses that contribute projector points, at least
MIN_PROJECTOR_POSE_FRACTION = 0.5   # ... and this share of the poses that show the board
MAX_STEREO_RMS_PX = 2.0             # stereo reprojection RMS beyond this is not a fit
# projector f_v / f_u: ~2 on a diamond DMD (rows half a column pitch apart), ~1 on a
# square-pixel device; anything else is not a projector the engine can drive
FV_OVER_FU_EXPECTED = (1.0, 2.0)
FV_OVER_FU_TOL = 0.25               # relative, around either: 0.75 .. 1.25 or 1.5 .. 2.5
FOCAL_OVER_WIDTH = (0.25, 20.0)     # f_u / DMD width: a throw ratio outside this is absurd
# how a diamond row parity came to be known
PARITY_SOURCES = ("row_parity", "preset", "argument", "none")


class CalibrationRejected(ValueError):
    """The fit is not a usable calibration: nothing was saved. reasons: why; per_pose: one
    row per pose that showed the board (corners, projector corners, contributed)."""

    def __init__(self, reasons, per_pose=None, numbers=None):
        self.reasons = [str(r) for r in reasons]
        self.per_pose = list(per_pose or [])
        self.numbers = dict(numbers or {})
        self.report_path = None
        super().__init__("stereo calibration REJECTED, nothing saved: " + "; ".join(self.reasons)
                         + self._poses_text())

    def _poses_text(self) -> str:
        if not self.per_pose:
            return ""
        yes = [r for r in self.per_pose if r.get("contributed")]
        return (f" | projector points from {len(yes)} of {len(self.per_pose)} poses: "
                + ", ".join(f"{r.get('id', r.get('view', r.get('pose')))} "
                            f"{r.get('projector_corners')}/{r.get('corners')}"
                            for r in self.per_pose))


def parity_text(parity, source) -> str:
    """The diamond row parity of a calibration, in words: measured (also when the
    measurement found NO half-column offset), set, or not measured at all."""
    val = {1: "1 (odd rows sit half a column right)", 0: "0 (even rows sit half a column right)",
           None: "none (no half-column offset: square grid)"}.get(parity, str(parity))
    if source == "row_parity":
        return f"{val} - measured"
    if source in ("preset", "argument"):
        return f"{val} - set by the {source}"
    if source == "none":
        return "NOT measured (columns decoded as a square grid)"
    return val + (" - not recorded whether it was measured" if parity is None else "")


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
    # how diamond_parity is known: "row_parity" (measured - also when it found no offset:
    # parity None), "preset" / "argument" (set), "none" (not measured), None (not recorded)
    parity_source: Optional[str] = None
    parity_info: dict = field(default_factory=dict)     # the measurement (shift, note, id)
    rehomed: Optional[dict] = None      # set on a re-homed copy: {"from": calibration id, ...}

    @property
    def parity_measured(self) -> Optional[bool]:
        """True: the row parity was measured (row_parity); False: it was not; None: an
        older calibration that did not record it."""
        if self.parity_source is None:
            return None
        return self.parity_source == "row_parity"

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
                "diamond_parity": self.diamond_parity,
                "diamond_parity_source": self.parity_source,
                "diamond_parity_measured": self.parity_measured,
                "diamond_parity_info": self.parity_info or None,
                "rms": self.rms, "board": self.board,
                "n_poses": self.n_poses, "geometry_fp": self.fingerprint,
                "baseline_mm": self.baseline_mm, "angle_deg": self.angle_deg,
                "created": self.created, "units": "mm; projector px = native DMD (col, row)",
                "rehomed": self.rehomed,
                "note": "X_proj = R X_cam + t; valid only at this geometry_id + fingerprint"
                        + (f"; {self.rehomed.get('note')}" if self.rehomed else "")}

    @classmethod
    def from_json(cls, d: dict, path: Optional[Path] = None) -> "StereoCalibration":
        a = lambda k: np.asarray(d[k], np.float64)          # noqa: E731
        return cls(id=d["id"], geometry_id=d["geometry_id"], K_cam=a("K_cam"),
                   dist_cam=a("dist_cam"), K_proj=a("K_proj"), dist_proj=a("dist_proj"),
                   R=a("R"), t=a("t"), cam_size=tuple(d.get("cam_size", (1440, 1080))),
                   proj_size=tuple(d.get("proj_size", (912, 1140))),
                   diamond_parity=d.get("diamond_parity"), rms=d.get("rms") or {},
                   board=d.get("board") or {}, n_poses=d.get("n_poses", 0),
                   fingerprint=d.get("geometry_fp"), path=path, created=d.get("created", ""),
                   parity_source=d.get("diamond_parity_source"),
                   parity_info=d.get("diamond_parity_info") or {},
                   rehomed=d.get("rehomed") or None)


def calibrate(poses: list, spec: BoardSpec, cam_size=(1440, 1080), proj_size=(912, 1140),
              projector_distortion: bool = False, half_win: int = 18) -> dict:
    """poses: [{"image": flood uint8, "corr_x", "corr_y"}] (one per board view; a view may
    carry its own "spec": one face of the calibration fixture, several views sharing one
    capture) -> {"cam": ..., "proj": ..., "R", "t", "rms_stereo", "per_pose": [...],
    "used": view indices in camera-view order}"""
    import cv2
    det = [detect(p["image"], p.get("spec") or spec) for p in poses]
    used = [i for i, d in enumerate(det) if d is not None]
    if len(used) < 3:
        raise ValueError(f"only {len(used)} of {len(poses)} views show their ChArUco board "
                         f"({spec.squares_x}x{spec.squares_y}, {spec.dictionary}); need >= 3")
    cam = calibrate_camera([det[i] for i in used], cam_size)
    det, n_outliers = _drop_corner_outliers(det, used, cam)
    if n_outliers:
        cam = calibrate_camera([det[i] for i in used], cam_size)
    cam["corner_outliers_dropped"] = n_outliers
    obj_v, cam_v, prj_v, per_pose, obs = [], [], [], [], []
    for k, i in enumerate(used):
        corners, ids, obj = det[i]
        pp, ok, hrms = corners_to_projector(corners, poses[i]["corr_x"], poses[i]["corr_y"],
                                            cam["K"], cam["dist"], half_win=half_win)
        obs.append({"view": i, "obj": np.asarray(obj, float),
                    "cam_px": np.asarray(corners, float),
                    "proj_px": np.where(ok[:, None], pp, np.nan)})
        gives = bool(ok.sum() >= MIN_PROJECTOR_CORNERS)
        per_pose.append({"pose": i, "corners": int(len(ids)), "projector_corners": int(ok.sum()),
                         "projector_corner_pct": round(100.0 * float(ok.mean()), 1),
                         "contributed": gives,
                         "camera_rms_px": cam["per_pose_rms"][k],
                         "homography_rms_px": float(np.nanmedian(hrms)) if ok.any() else None,
                         **({"view": poses[i]["view"]} if "view" in poses[i] else {})})
        if gives:
            obj_v.append(obj[ok])
            cam_v.append(corners[ok])
            prj_v.append(pp[ok])
    numbers = {"camera_rms_px": cam["rms"], "n_detected": len(used), "n_projector": len(obj_v)}
    need = max(MIN_PROJECTOR_POSES, int(math.ceil(MIN_PROJECTOR_POSE_FRACTION * len(used))))
    if len(obj_v) < need:
        raise CalibrationRejected(
            [f"only {len(obj_v)} of the {len(used)} poses that show the board gave projector "
             f"points (>= {MIN_PROJECTOR_CORNERS} corners with a decoded projector position); "
             f"{need} are needed. The fringes are not resolved on the board: capture the set "
             "again with coarser fringe periods (16,160,1280 or 24,192,1280 DMD px), check "
             "the projector focus and that the board sits inside the projected light"],
            per_pose, numbers)
    proj = calibrate_projector(obj_v, prj_v, proj_size, distortion=projector_distortion)
    numbers.update(projector_fu_px=float(proj["K"][0, 0]), projector_fv_px=float(proj["K"][1, 1]),
                   fv_over_fu=proj["fv_over_fu"], projector_rms_u_px=proj["rms_u"],
                   projector_rms_v_px=proj["rms_v"])
    bad = intrinsics_problems(cam["K"], cam_size, "camera") + \
        intrinsics_problems(proj["K"], proj_size, "projector", FV_OVER_FU_EXPECTED)
    if bad:
        raise CalibrationRejected(bad, per_pose, numbers)
    flags = cv2.CALIB_FIX_INTRINSIC
    rms, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
        [np.asarray(o, np.float32).reshape(-1, 1, 3) for o in obj_v],
        [np.asarray(c, np.float32).reshape(-1, 1, 2) for c in cam_v],
        [np.asarray(p, np.float32).reshape(-1, 1, 2) for p in prj_v],
        cam["K"], cam["dist"], proj["K"], proj["dist"], tuple(cam_size), flags=flags)
    numbers["stereo_rms_px"] = float(rms)
    if not (math.isfinite(float(rms)) and float(rms) <= MAX_STEREO_RMS_PX
            and np.all(np.isfinite(R)) and np.all(np.isfinite(T))):
        raise CalibrationRejected(
            [f"stereo reprojection RMS {float(rms):.2f} px > {MAX_STEREO_RMS_PX:g} px: the "
             "camera and projector views do not describe one rig (projector positions badly "
             "decoded, the board moved during a capture, or mixed fringe periods)"],
            per_pose, numbers)
    return {"cam": cam, "proj": proj, "R": R, "t": T.reshape(3), "rms_stereo": float(rms),
            "per_pose": per_pose, "n_poses": len(obj_v), "n_detected": len(used),
            "used": used, "observations": obs,
            "corner_outliers_dropped": cam["corner_outliers_dropped"]}


def intrinsics_problems(K, size, what: str, fv_over_fu=None) -> list:
    """Why intrinsics K (for an image of `size` = (W, H)) can't be a real lens: a focal length
    that is not finite, <= 0 or absurd (f_u outside FOCAL_OVER_WIDTH x the width), or f_v / f_u
    further than FV_OVER_FU_TOL from every expected ratio (fv_over_fu: one or several; default
    1, a camera's square pixels; the projector: FV_OVER_FU_EXPECTED). [] = plausible."""
    K = np.asarray(K, np.float64)
    fu, fv = float(K[0, 0]), float(K[1, 1])
    W = float(size[0])
    if not (np.all(np.isfinite(K)) and fu > 0 and fv > 0):
        return [f"degenerate {what} intrinsics: f_u {fu:.3g} px, f_v {fv:.3g} px (a focal "
                "length must be finite and > 0)"]
    out = []
    lo, hi = FOCAL_OVER_WIDTH
    if not lo * W <= fu <= hi * W:
        out.append(f"absurd {what} focal length: f_u {fu:.1f} px for a {W:.0f} px wide image "
                   f"(plausible: {lo * W:.0f} .. {hi * W:.0f} px)")
    want = [float(v) for v in np.atleast_1d(1.0 if fv_over_fu is None else fv_over_fu)]
    if all(abs(fv / fu / w - 1.0) > FV_OVER_FU_TOL for w in want):
        out.append(f"{what} f_v / f_u = {fv / fu:.3f}, expected ~"
                   f"{' or ~'.join(f'{w:g}' for w in want)} (+-{100 * FV_OVER_FU_TOL:.0f} %)")
    return out


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

def fit_problems(fit: dict) -> list:
    """The gates of calibrate() on a finished fit dict (also after the rigid-fixture
    refinement replaced its values). [] = a usable calibration."""
    bad = intrinsics_problems(fit["cam"]["K"], fit["cam"]["image_size"], "camera") + \
        intrinsics_problems(fit["proj"]["K"], fit["proj"]["image_size"], "projector",
                            FV_OVER_FU_EXPECTED)
    rms = float(fit["rms_stereo"])
    if not (math.isfinite(rms) and rms <= MAX_STEREO_RMS_PX):
        bad.append(f"stereo reprojection RMS {rms:.2f} px > {MAX_STEREO_RMS_PX:g} px")
    return bad


def save(store, geometry_id: str, fit: dict, *, fingerprint: Optional[dict], spec: BoardSpec,
         parity: Optional[int], created: str, extra_report: Optional[dict] = None,
         truth: Optional[dict] = None, board: Optional[dict] = None,
         parity_source: Optional[str] = None,
         parity_info: Optional[dict] = None) -> StereoCalibration:
    """board: what the calibration records as its target (default spec.to_dict(); the
    calibration fixture records its faces). parity_source / parity_info: how the diamond
    row parity is known (PARITY_SOURCES) and its measurement. A fit that fails
    fit_problems() is refused (CalibrationRejected): never saved as a calibration."""
    from ..library.store import json_safe
    bad = fit_problems(fit)
    if bad:
        raise CalibrationRejected(bad, fit.get("per_pose"))
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
        board=board or spec.to_dict(), n_poses=fit["n_poses"], fingerprint=fingerprint, path=d,
        created=created, parity_source=parity_source, parity_info=dict(parity_info or {}))
    report = {"per_pose": fit["per_pose"], "camera_per_pose_rms": fit["cam"]["per_pose_rms"],
              "projector_per_pose_rms": fit["proj"]["per_pose_rms"],
              "fv_over_fu": fit["proj"]["fv_over_fu"],
              "diamond_parity": parity, "diamond_parity_source": parity_source,
              "diamond_parity_measured": cal.parity_measured,
              "diamond_parity_text": parity_text(parity, parity_source),
              **(extra_report or {})}
    if truth is not None:
        report["vs_truth"] = compare_truth(cal, truth)
    store.write_json(d / "report.json", json_safe(report))
    store.write_json(d / STEREO_FILE, json_safe(cal.to_json()))       # LAST: marker
    return cal


REHOME_MAX_ROTATION_DEG = 1.0       # a re-home is a nudge: anything larger is a recalibration
REHOME_NOTE = "re-homed (rotation only) from {id}"


def rehome(store, cal: StereoCalibration, Rd, *, created: str,
           details: Optional[dict] = None) -> StereoCalibration:
    """A NEW stereo calibration entry: `cal` corrected for a small camera rotation.

        R1 = R0 @ Rd,   t1 = t0,   intrinsics unchanged

    Rd is the camera's rotation as rigcheck.camera_rotation returns it ("R"): a point at
    X_new in the camera frame after the move was at X_old = Rd @ X_new before it, so
    X_proj = R0 @ X_old + t0 = (R0 @ Rd) @ X_new + t0 (the projector did not move). The
    composition and its sign are locked by tests/test_rehome_mock.py, which turns the
    simulated camera by a known rotation; they were first verified on a real rig.
    The entry is flagged rehomed = {"from": cal.id, "note": "re-homed (rotation only) from
    <id>", ...details}; `cal`'s own entry is never touched. Rotation only: a camera
    translation is not corrected, so the absolute depth can keep a small offset - a full
    calibration is the reference."""
    from ..library.store import json_safe
    Rd = np.asarray(Rd, np.float64).reshape(3, 3)
    if not (np.all(np.isfinite(Rd)) and np.allclose(Rd @ Rd.T, np.eye(3), atol=1e-6)
            and np.linalg.det(Rd) > 0):
        raise ValueError("re-home: Rd is not a rotation matrix")
    ang = math.degrees(math.acos(float(np.clip((np.trace(Rd) - 1.0) / 2.0, -1.0, 1.0))))
    if ang > REHOME_MAX_ROTATION_DEG:
        raise ValueError(f"re-home: a rotation of {ang:.3f} deg is not a nudge (limit "
                         f"{REHOME_MAX_ROTATION_DEG:g} deg): recalibrate")
    U, _, Vt = np.linalg.svd(np.asarray(cal.R, np.float64) @ Rd)    # stay orthonormal
    R1 = U @ Vt
    prev = cal.rehomed or {}
    d = store.new_dir("calibrations", cal.geometry_id, label="stereo_rehomed")
    info = {"from": cal.id, "root": prev.get("root") or cal.id,
            "generation": int(prev.get("generation") or 0) + 1,
            "note": REHOME_NOTE.format(id=cal.id), "rotation_deg": round(ang, 5),
            "Rd": Rd.tolist(), "t_unchanged": True, "created": created,
            "caveat": "rotation only: a camera translation is not corrected, the absolute "
                      "depth may keep a small offset; a full calibration is the reference",
            **(details or {})}
    new = StereoCalibration(
        id=f"{cal.geometry_id}/{d.name}", geometry_id=cal.geometry_id, K_cam=cal.K_cam,
        dist_cam=cal.dist_cam, K_proj=cal.K_proj, dist_proj=cal.dist_proj, R=R1,
        t=np.asarray(cal.t, np.float64).reshape(3), cam_size=tuple(cal.cam_size),
        proj_size=tuple(cal.proj_size), diamond_parity=cal.diamond_parity, rms=dict(cal.rms),
        board=dict(cal.board), n_poses=cal.n_poses, fingerprint=cal.fingerprint, path=d,
        created=created, parity_source=cal.parity_source, parity_info=dict(cal.parity_info),
        rehomed=info)
    store.write_json(d / "report.json", json_safe({"rehomed": info, "R_before": cal.R.tolist(),
                                                   "R_after": R1.tolist(),
                                                   "t": np.ravel(cal.t).tolist()}))
    store.write_json(d / STEREO_FILE, json_safe(new.to_json()))       # LAST: marker
    return new


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

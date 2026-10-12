"""
Projector as an inverse camera.

Per board pose the projector's (col, row) is decoded on both axes over the
board (multifreq axis "xy"). The board is planar, so on it the map from
UNDISTORTED camera pixels to projector pixels is a homography; around each
ChArUco corner a local homography is fitted to the decoded pixels in a
window (RANSAC, then least squares on the inliers) and evaluated at the
corner. That gives the corner in DMD coordinates without ever detecting
anything in the projector image. Then the projector is calibrated like a
camera from (board mm, DMD px) pairs.

Coordinates: native DMD (col, row), as decoded by bases/multifreq with the
measured diamond_parity — so f_v comes out ~2 f_u (rows are half a column
pitch apart). Reprojection is reported per axis. Target: < 0.5 px each.
The projector lens is modelled without distortion by default (the mock has
none; enable `distortion=True` on a real lens if the per-axis RMS says so).
"""

from __future__ import annotations

import numpy as np

PROJECTOR_RMS_TARGET_PX = 0.5


def corners_to_projector(corners: np.ndarray, corr_x: np.ndarray, corr_y: np.ndarray,
                         K: np.ndarray, dist, half_win: int = 18,
                         min_valid: float = 0.4) -> tuple:
    """Camera-px corners (N, 2) -> (projector (col, row) (N, 2), ok (N,), inlier rms (N,))."""
    import cv2
    H, W = corr_x.shape
    out = np.full((len(corners), 2), np.nan)
    ok = np.zeros(len(corners), bool)
    rms = np.full(len(corners), np.nan)
    good = np.isfinite(corr_x) & np.isfinite(corr_y)
    for i, (x, y) in enumerate(np.asarray(corners, float)):
        x0, x1 = int(round(x)) - half_win, int(round(x)) + half_win + 1
        y0, y1 = int(round(y)) - half_win, int(round(y)) + half_win + 1
        if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
            continue
        m = good[y0:y1, x0:x1]
        if m.mean() < min_valid:
            continue
        vv, uu = np.nonzero(m)
        src = np.stack([uu + x0, vv + y0], 1).astype(np.float64)
        dst = np.stack([corr_x[y0:y1, x0:x1][m], corr_y[y0:y1, x0:x1][m]], 1).astype(np.float64)
        pts = np.vstack([src, [[x, y]]]).reshape(-1, 1, 2)
        und = cv2.undistortPoints(pts, K, np.asarray(dist, np.float64), P=K).reshape(-1, 2)
        Hm, inl = cv2.findHomography(und[:-1], dst, cv2.RANSAC, 1.0)
        if Hm is None:
            continue
        inl = inl.ravel().astype(bool)
        if inl.sum() < 12:
            continue
        Hm, _ = cv2.findHomography(und[:-1][inl], dst[inl], 0)
        p = Hm @ np.array([und[-1, 0], und[-1, 1], 1.0])
        out[i] = p[:2] / p[2]
        q = (np.c_[und[:-1][inl], np.ones(inl.sum())] @ Hm.T)
        e = q[:, :2] / q[:, 2:3] - dst[inl]
        rms[i] = float(np.sqrt(np.mean(np.sum(e * e, 1))))
        ok[i] = True
    return out, ok, rms


def projector_corner_count(corners: np.ndarray, corr_x: np.ndarray, corr_y: np.ndarray,
                           half_win: int = 18) -> int:
    """How many of these board corners (camera px, (N, 2)) get a projector position from
    this decode: the count calibrate_stereo will find. No camera calibration is needed for
    the COUNT (the lens distortion is locally affine over the window, so the local
    homography is fitted on raw pixels): a pose can be judged the moment it is captured."""
    if corr_x is None or corr_y is None or len(corners) == 0:
        return 0
    _, ok, _ = corners_to_projector(np.asarray(corners, np.float64).reshape(-1, 2), corr_x,
                                    corr_y, np.eye(3), np.zeros(5), half_win=half_win)
    return int(ok.sum())


def calibrate_projector(obj_views: list, proj_views: list, size=(912, 1140),
                        distortion: bool = False) -> dict:
    """obj_views / proj_views: per pose (N, 3) board mm and (N, 2) DMD px.
    -> {K, dist, rms, rms_u, rms_v, per_pose_rms, rvecs, tvecs, ...}"""
    import cv2
    pairs = [(o, p) for o, p in zip(obj_views, proj_views) if len(o) >= 6]
    if len(pairs) < 3:
        raise ValueError(f"projector calibration needs >= 3 poses with >= 6 decoded corners, "
                         f"got {len(pairs)}")
    obj = [np.asarray(o, np.float32).reshape(-1, 1, 3) for o, _ in pairs]
    img = [np.asarray(p, np.float32).reshape(-1, 1, 2) for _, p in pairs]
    flags = cv2.CALIB_FIX_K3
    if not distortion:
        flags |= (cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2)
    # the planar init assumes square pixels: start from the diamond DMD's
    # f_v = 2 f_u instead (throw ratio ~1.4) so LM starts near the truth
    W, H = int(size[0]), int(size[1])
    K0 = np.array([[1.4 * W, 0, W / 2], [0, 2.8 * W, H / 2], [0, 0, 1]], np.float64)
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj, img, (W, H), K0, np.zeros(5), flags=flags | cv2.CALIB_USE_INTRINSIC_GUESS)
    eu, ev, per_pose = [], [], []
    for o, i, rv, tv in zip(obj, img, rvecs, tvecs):
        proj, _ = cv2.projectPoints(o, rv, tv, K, dist)
        e = (proj - i).reshape(-1, 2)
        eu.append(e[:, 0])
        ev.append(e[:, 1])
        per_pose.append(float(np.sqrt(np.mean(np.sum(e * e, 1)))))
    eu, ev = np.concatenate(eu), np.concatenate(ev)
    rms_u, rms_v = float(np.sqrt(np.mean(eu ** 2))), float(np.sqrt(np.mean(ev ** 2)))
    return {"K": K, "dist": dist.reshape(-1), "rms": float(rms), "rms_u": rms_u, "rms_v": rms_v,
            "per_pose_rms": per_pose, "rvecs": [r.reshape(3) for r in rvecs],
            "tvecs": [t.reshape(3) for t in tvecs], "n_views": len(pairs),
            "fv_over_fu": float(K[1, 1] / K[0, 0]), "image_size": (W, H),
            "ok": rms_u < PROJECTOR_RMS_TARGET_PX and rms_v < PROJECTOR_RMS_TARGET_PX}

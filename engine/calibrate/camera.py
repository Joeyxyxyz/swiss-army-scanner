"""
Camera intrinsics from ChArUco views: cv2.calibrateCamera + a per-pose
reprojection report (so one bad pose — motion blur, a curled board — is
visible and can be dropped instead of silently dragging K).

Target on the bench: overall RMS < 0.3 px.
"""

from __future__ import annotations

import numpy as np

CAMERA_RMS_TARGET_PX = 0.3


def calibrate_camera(views: list, image_size: tuple, fix_k3: bool = True) -> dict:
    """views: [(corners (N, 2) px, ids (N,), object points (N, 3) mm), ...];
    image_size (W, H). -> {K, dist, rms, per_pose_rms, rvecs, tvecs, ...}"""
    import cv2
    views = [v for v in views if v is not None and len(v[0]) >= 6]
    if len(views) < 3:
        raise ValueError(f"camera calibration needs >= 3 board views with >= 6 corners, "
                         f"got {len(views)}")
    obj = [np.asarray(v[2], np.float32).reshape(-1, 1, 3) for v in views]
    img = [np.asarray(v[0], np.float32).reshape(-1, 1, 2) for v in views]
    flags = cv2.CALIB_FIX_K3 if fix_k3 else 0
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj, img, tuple(int(s) for s in image_size),
                                                     None, None, flags=flags)
    per_pose = []
    for o, i, rv, tv in zip(obj, img, rvecs, tvecs):
        proj, _ = cv2.projectPoints(o, rv, tv, K, dist)
        e = (proj - i).reshape(-1, 2)
        per_pose.append(float(np.sqrt(np.mean(np.sum(e * e, axis=1)))))
    return {"K": K, "dist": dist.reshape(-1), "rms": float(rms), "per_pose_rms": per_pose,
            "rvecs": [r.reshape(3) for r in rvecs], "tvecs": [t.reshape(3) for t in tvecs],
            "n_views": len(views), "n_points": int(sum(len(v[0]) for v in views)),
            "image_size": tuple(int(s) for s in image_size),
            "ok": float(rms) < CAMERA_RMS_TARGET_PX}

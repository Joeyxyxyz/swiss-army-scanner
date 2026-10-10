"""
Rigid-fixture refinement for a calibration-fixture pose set.

Each face of the fixture is first calibrated as its own planar view (camera.py,
projector.py, stereoCalibrate: the unchanged path). A ~100 mm face at ~700 mm is a weak
view on its own: its image is nearly affine, so the focal length trades against distance
(weak perspective) and the 3-placement fit lands several px off (mock: up to ~15 px).

The fixture is rigid, though: whatever the print made of its faces, face k sits at the SAME
pose relative to face 0 in every placement. This step estimates those face-to-face poses
as unknowns shared by all placements, jointly with the camera intrinsics + distortion, the
projector intrinsics and the stereo R, t (one bundle adjustment, scipy least_squares). That
turns the fixture into a 3D target without ever assuming the PRINTED angles: the spec's face
poses are only the starting point (and the reference of the face-angle check afterwards).

Gauge: the fixture frame is pinned to one reference face at its spec pose.
Fails closed: if the joint fit is clearly worse than the independent views (the fixture
moved / flexed between placements), the caller keeps the independent result and warns.

MOCK-TESTED ONLY: not yet used on a real rig.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


def _rodrigues(r: np.ndarray) -> np.ndarray:
    """(n, 3) rotation vectors -> (n, 3, 3)."""
    r = np.atleast_2d(np.asarray(r, float))
    th = np.linalg.norm(r, axis=1)
    k = r / np.where(th > 1e-15, th, 1.0)[:, None]
    K = np.zeros((len(r), 3, 3))
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 2] = -k[:, 2], k[:, 1], -k[:, 0]
    K[:, 1, 0], K[:, 2, 0], K[:, 2, 1] = k[:, 2], -k[:, 1], k[:, 0]
    s, c = np.sin(th)[:, None, None], np.cos(th)[:, None, None]
    return np.eye(3)[None] + s * K + (1 - c) * (K @ K)


def _rvec(R: np.ndarray) -> np.ndarray:
    import cv2
    return cv2.Rodrigues(np.asarray(R, np.float64))[0].reshape(3)


def _project(X: np.ndarray, fx, fy, cx, cy, dist) -> np.ndarray:
    """Camera-frame points (n, 3) -> pixels with Brown distortion (k1, k2, p1, p2, k3)."""
    k1, k2, p1, p2, k3 = (list(dist) + [0.0] * 5)[:5]
    x, y = X[:, 0] / X[:, 2], X[:, 1] / X[:, 2]
    r2 = x * x + y * y
    rad = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
    xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return np.stack([fx * xd + cx, fy * yd + cy], 1)


def refine(views: list, cam: dict, proj: dict, R: np.ndarray, t: np.ndarray,
           face_init: dict, projector_distortion: bool = False, max_nfev: int = 300) -> dict:
    """views: [{"placement", "face", "obj" (n, 3) board mm, "cam_px" (n, 2),
    "proj_px" (n, 2) (NaN = not decoded)}]; cam / proj: {"K", "dist"} (first pass);
    R, t: stereo (X_proj = R X_cam + t); face_init: {face: (R_bF, t_bF)} (spec poses).
    -> {"cam": {K, dist}, "proj": {K, dist}, "R", "t", "faces": {face: (R, t)},
        "rms_cam", "rms_proj_u", "rms_proj_v", "per_view_cam_rms", "per_view_proj_rms",
        "success", "nfev", "ref_face"}"""
    from scipy.optimize import least_squares
    pls = sorted({int(v["placement"]) for v in views})
    faces = sorted({int(v["face"]) for v in views})
    count = {f: sum(1 for v in views if int(v["face"]) == f) for f in faces}
    ref = max(faces, key=lambda f: (count[f], -f))
    free = [f for f in faces if f != ref]
    Kc, dc = np.asarray(cam["K"], float), np.asarray(cam["dist"], float).ravel()
    Kp, dp = np.asarray(proj["K"], float), np.asarray(proj["dist"], float).ravel()
    dc = np.r_[dc, np.zeros(5)][:5]
    dp = np.r_[dp, np.zeros(5)][:5]
    k3c = dc[4]

    # placement poses (fixture -> camera) from one view each, at the spec face pose
    import cv2
    pl_pose = {}
    for v in views:
        p = int(v["placement"])
        if p in pl_pose and int(v["face"]) != ref:
            continue
        ok, rv, tv = cv2.solvePnP(np.asarray(v["obj"], np.float64),
                                  np.asarray(v["cam_px"], np.float64), Kc, dc)
        if not ok:
            continue
        Rcb = cv2.Rodrigues(rv)[0]
        Rf, tf = (np.asarray(a, float) for a in face_init[int(v["face"])])
        Rcf = Rcb @ Rf.T                                  # fixture -> camera
        pl_pose[p] = (Rcf, tv.reshape(3) - Rcf @ tf)
    if set(pl_pose) != set(pls):
        raise ValueError("rigid refinement: a placement could not be posed")

    n_pd = 4 if projector_distortion else 0
    x0 = [Kc[0, 0], Kc[1, 1], Kc[0, 2], Kc[1, 2], *dc[:4], Kp[0, 0], Kp[1, 1], Kp[0, 2],
          Kp[1, 2], *(dp[:4] if projector_distortion else []), *_rvec(R), *np.ravel(t)]
    for p in pls:
        x0 += [*_rvec(pl_pose[p][0]), *pl_pose[p][1]]
    for f in free:
        Rf, tf = face_init[f]
        x0 += [*_rvec(Rf), *np.ravel(tf)]
    x0 = np.asarray(x0, float)
    o_st = 12 + n_pd
    o_pl = o_st + 6
    o_fc = o_pl + 6 * len(pls)
    pl_ix = {p: i for i, p in enumerate(pls)}
    fc_ix = {f: i for i, f in enumerate(free)}
    Rref, tref = (np.asarray(a, float) for a in face_init[ref])

    # flatten the observations once
    obj = np.concatenate([np.asarray(v["obj"], float) for v in views])
    cpx = np.concatenate([np.asarray(v["cam_px"], float) for v in views])
    ppx = np.concatenate([np.asarray(v["proj_px"], float) for v in views])
    vi = np.concatenate([np.full(len(v["obj"]), i) for i, v in enumerate(views)])
    v_pl = np.array([pl_ix[int(v["placement"])] for v in views])
    v_fc = np.array([fc_ix.get(int(v["face"]), -1) for v in views])
    pok = np.isfinite(ppx).all(1)

    def unpack(x):
        fx, fy, cx, cy, k1, k2, p1, p2 = x[:8]
        fu, fv, cu, cv = x[8:12]
        pd = list(x[12:12 + n_pd]) + [0.0] * (5 - n_pd) if n_pd else [0.0] * 5
        Rs = _rodrigues(x[o_st:o_st + 3])[0]
        ts = x[o_st + 3:o_st + 6]
        P = x[o_pl:o_fc].reshape(-1, 6)
        F = x[o_fc:].reshape(-1, 6)
        return (fx, fy, cx, cy, [k1, k2, p1, p2, k3c]), (fu, fv, cu, cv, pd), Rs, ts, P, F

    def to_camera(x):
        c, pj, Rs, ts, P, F = unpack(x)
        Rp, tp = _rodrigues(P[:, :3]), P[:, 3:]
        Rf = np.concatenate([_rodrigues(F[:, :3]), Rref[None]]) if len(F) else Rref[None]
        tf = np.concatenate([F[:, 3:], tref[None]]) if len(F) else tref[None]
        fsel = np.where(v_fc >= 0, v_fc, len(F))            # the reference face last
        Rv = Rp[v_pl] @ Rf[fsel]                              # board -> camera, per view
        tv = np.einsum("vij,vj->vi", Rp[v_pl], tf[fsel]) + tp[v_pl]
        Xc = np.einsum("nij,nj->ni", Rv[vi], obj) + tv[vi]
        return Xc, c, pj, Rs, ts

    def resid(x):
        Xc, c, pj, Rs, ts = to_camera(x)
        rc = _project(Xc, *c) - cpx
        Xp = Xc[pok] @ Rs.T + ts
        rp = _project(Xp, *pj) - ppx[pok]
        return np.concatenate([rc.ravel(), rp.ravel()])

    res = least_squares(resid, x0, method="trf", x_scale="jac", max_nfev=max_nfev)
    x = res.x
    Xc, c, pj, Rs, ts = to_camera(x)
    ec = _project(Xc, *c) - cpx
    ep = _project(Xc[pok] @ Rs.T + ts, *pj) - ppx[pok]
    fx, fy, cx, cy, dist_c = c
    fu, fv, cu, cv, dist_p = pj
    F = x[o_fc:].reshape(-1, 6)
    faces_out = {ref: (Rref, tref)}
    for f, i in fc_ix.items():
        faces_out[f] = (_rodrigues(F[i, :3])[0], F[i, 3:].copy())
    per_c, per_p = [], []
    pv = vi[pok]
    for i in range(len(views)):
        e = ec[vi == i]
        per_c.append(float(np.sqrt(np.mean(np.sum(e * e, 1)))))
        q = ep[pv == i]
        per_p.append(float(np.sqrt(np.mean(np.sum(q * q, 1)))) if len(q) else None)
    return {"cam": {"K": np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]]),
                    "dist": np.asarray(dist_c, float)},
            "proj": {"K": np.array([[fu, 0, cu], [0, fv, cv], [0, 0, 1.0]]),
                     "dist": np.asarray(dist_p, float)},
            "R": Rs, "t": np.asarray(ts, float), "faces": faces_out, "ref_face": ref,
            "rms_cam": float(np.sqrt(np.mean(np.sum(ec * ec, 1)))),
            "rms_proj": float(np.sqrt(np.mean(np.sum(ep * ep, 1)))),
            "rms_proj_u": float(np.sqrt(np.mean(ep[:, 0] ** 2))),
            "rms_proj_v": float(np.sqrt(np.mean(ep[:, 1] ** 2))),
            "per_view_cam_rms": per_c, "per_view_proj_rms": per_p,
            "success": bool(res.success), "nfev": int(res.nfev),
            "n_placements": len(pls), "n_faces": len(faces)}


def face_normal_angles(faces: dict) -> dict:
    """{(i, j): angle between the estimated face normals, deg}."""
    ks = sorted(faces)
    out = {}
    for a in range(len(ks)):
        for b in range(a + 1, len(ks)):
            ni = np.asarray(faces[ks[a]][0])[:, 2]
            nj = np.asarray(faces[ks[b]][0])[:, 2]
            out[(ks[a], ks[b])] = math.degrees(math.acos(float(np.clip(ni @ nj, -1, 1))))
    return out

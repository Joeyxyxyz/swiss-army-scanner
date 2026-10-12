"""Synthetic metric scans for the recipe tests (no capture path): the mock stereo bench's
pinhole camera + projector ray-cast against a named mock scene (engine/capture/mock_scenes.py)
- the true surface points + noise along the camera ray, the true projector (col, row) + noise
as the decoded correspondences, and the flood pair (white / black) the fast scan keeps - saved
exactly as the library saves a metric scan (reconstruct/metric_scan._save), next to the rig's
TRUE stereo calibration at the preset's geometry. A full mock capture (api.scan_fast) is the
same picture, ~30 s slower."""

from __future__ import annotations

import numpy as np

from engine import config
from engine.calibrate.stereo import StereoCalibration
from engine.capture import mock_scenes
from engine.capture.mock_stereo import Plane, StereoScene
from engine.decode.deformation import DeformationField
from engine.reconstruct import metric_scan
from engine.reconstruct.triangulate import MetricCloud


def truth_calibration(s, preset) -> StereoCalibration:
    """The bench's TRUE stereo calibration, stored at the preset's geometry + fingerprint."""
    from engine.calibrate import stereo
    have = stereo.latest(s.store, preset.geometry_id, config.geometry_fingerprint(preset))
    if have is not None:
        return have
    tr = s.bench.stereo_truth()
    d = s.store.new_dir("calibrations", preset.geometry_id, label="stereo")
    cal = StereoCalibration(id=f"{preset.geometry_id}/{d.name}", geometry_id=preset.geometry_id,
                            K_cam=tr["K_cam"], dist_cam=tr["dist_cam"], K_proj=tr["K_proj"],
                            dist_proj=np.zeros(5), R=tr["R"], t=np.ravel(tr["t"]),
                            diamond_parity=tr["parity"],
                            fingerprint=config.geometry_fingerprint(preset), path=d,
                            created="2026-01-01T00:00:00")
    s.store.write_json(d / "stereo_calibration.json", cal.to_json())
    return cal


def calibrated_now(s, preset, n_poses: int = 3, now=None) -> StereoCalibration:
    """truth_calibration() plus what a real calibration leaves on the geometry's rig-state
    record (engine/rigstate.py): its poses were captured and it was solved NOW (`now`: the
    time written on the records). That is the calibration that verifies a MOBILE rig for the
    session and answers an earlier rig check MOVED; a calibration with no poses on the
    record (truth_calibration alone) does neither."""
    from engine import rigstate
    cal = truth_calibration(s, preset)
    poses = [f"{cal.path.name}_pose{i + 1}" for i in range(n_poses)]
    for pose in poses:
        rigstate.record_pose(s.store, preset, pose, "synthetic", now)
    rigstate.record_calibration(s.store, preset, cal.id, poses, "synthetic", now)
    return cal


_WALL: dict = {}                    # the wall-only picture, per rig (traced once)


def _crop(rig, shapes, margin_px: int = 70) -> tuple:
    """Camera box around the shapes (their bounding boxes projected) + margin (shadows)."""
    W, H = rig.camera.size
    pts = []
    for sh in shapes:
        c = np.asarray(getattr(sh, "center", (0.0, 0.0, 800.0)), float)
        r = float(max(np.max(getattr(sh, "half", (0.0,))), getattr(sh, "radius", 0.0),
                      getattr(sh, "height", 0.0) / 2))
        for dx in (-r, r):
            for dy in (-r, r):
                for dz in (-r, r):
                    pts.append(rig.camera.project(c + [dx, dy, dz]))
    if not pts:
        return 0, 0, 0, 0
    pts = np.array(pts)
    x0, y0 = np.floor(pts.min(0)).astype(int) - margin_px
    x1, y1 = np.ceil(pts.max(0)).astype(int) + margin_px
    return max(int(x0), 0), max(int(y0), 0), min(int(x1), W), min(int(y1), H)


def geometry(rig, shapes, albedo_wall: float = 0.6) -> dict:
    """StereoScene.geometry, but the shapes are traced only inside their camera box (the rest
    of the frame sees the plain wall): the same picture, several times faster."""
    wall = Plane(albedo=albedo_wall)
    key = (id(rig), float(albedo_wall))
    if key not in _WALL:
        _WALL.clear()
        _WALL[key] = StereoScene(rig, [], wall=wall).geometry()
    g = {k: np.array(v, copy=True) for k, v in _WALL[key].items()}
    x0, y0, x1, y1 = _crop(rig, shapes)
    if x1 <= x0 or y1 <= y0:
        return g
    sub = StereoScene(rig, list(shapes), wall=wall)
    d = rig.camera.rays()[y0:y1, x0:x1].reshape(-1, 3)
    t, nrm, idx = sub.trace(np.zeros(3), d)
    hit = np.isfinite(t)
    X = np.where(hit[:, None], d * np.where(hit, t, 0)[:, None], np.nan)
    Cp = rig.projector_center
    toX = np.where(hit[:, None], X - Cp, np.array([0.0, 0.0, 1.0]))
    L = np.linalg.norm(toX, axis=1)
    dirp = toX / L[:, None]
    tp, _, _ = sub.trace(Cp, dirp)
    lit = hit & (tp >= L - 0.02)
    Xp = np.where(hit[:, None], X, 0) @ rig.R.T + rig.t
    uv = rig.projector.project(np.where(hit[:, None], Xp, 1.0))
    Wp, Hp = rig.projector.size
    lit &= (Xp[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] <= Wp - 1) & (uv[:, 1] >= 0) & \
        (uv[:, 1] <= Hp - 1)
    cosi = np.abs(np.einsum("ij,ij->i", nrm, -dirp))
    alb = np.zeros(len(d), np.float32)
    allsh = sub.all_shapes()
    for i, sh in enumerate(allsh):
        m = idx == i
        if m.any():
            alb[m] = sh.albedo_at(X[m])
    h, w = y1 - y0, x1 - x0
    g["X"][y0:y1, x0:x1] = X.reshape(h, w, 3)
    g["u"][y0:y1, x0:x1] = uv[:, 0].reshape(h, w)
    g["v"][y0:y1, x0:x1] = uv[:, 1].reshape(h, w)
    g["lit"][y0:y1, x0:x1] = lit.reshape(h, w)
    g["hit"][y0:y1, x0:x1] = hit.reshape(h, w)
    g["albedo"][y0:y1, x0:x1] = alb.reshape(h, w)
    g["shade"][y0:y1, x0:x1] = np.where(lit, cosi, 0).astype(np.float32).reshape(h, w)
    return g


def oracle_scan(s, preset, spec, label: str, noise_mm: float = 0.02, corr_noise_px: float = 0.03,
                seed: int = 0, flood: bool = True, axis: str = "xy"):
    """A metric scan of mock scene `spec` ('scene:tube:misaligned', or a list of stereo shapes)
    -> MetricScan (library entry)."""
    cal = truth_calibration(s, preset)
    shapes = mock_scenes.shapes(spec) if isinstance(spec, str) else list(spec)
    rig = s.bench.rig_model()
    g = geometry(rig, shapes, s.bench.base_albedo)
    rng = np.random.default_rng(seed)
    ok = g["lit"] & (g["shade"] > 0.05)
    X = g["X"]
    ray = X / np.maximum(np.linalg.norm(np.nan_to_num(X), axis=-1, keepdims=True), 1e-9)
    xyz = np.where(ok[..., None], X + ray * rng.normal(0, noise_mm, ok.shape)[..., None], np.nan)
    cx = np.where(ok, g["u"] + rng.normal(0, corr_noise_px, ok.shape), np.nan).astype(np.float32)
    cy = np.where(ok, g["v"] + rng.normal(0, corr_noise_px, ok.shape), np.nan).astype(np.float32)
    fld = DeformationField(corr_x=cx, corr_y=cy if axis == "xy" else None, corr_diag=None,
                           conf=ok.astype(np.float32), mask=ok, geometry_id=preset.geometry_id,
                           basis_names=["multifreq"],
                           meta={"synthetic": True, "axis": axis, "polarity_flipped": False})
    cloud = MetricCloud(xyz=xyz.astype(np.float32), mask=ok, consistency=None, method="plane",
                        geometry_id=preset.geometry_id, calibration_id=cal.id)
    arrays = None
    if flood:
        amb, gain = s.bench.ambient, s.bench.gain
        white = amb + gain * g["albedo"] * g["shade"] + rng.normal(0, 1.5, ok.shape)
        black = amb + rng.normal(0, 1.5, ok.shape)
        arrays = {"flood.npz": {"white": white.astype(np.float32),
                                "black": black.astype(np.float32)}}
    return metric_scan._save(s, fld, cloud, label, cal, config.geometry_fingerprint(preset),
                             preset.name, {"synthetic": True, "scene": str(spec)[:80],
                                           "axis": axis}, arrays=arrays)

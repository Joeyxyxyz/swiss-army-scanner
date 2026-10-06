"""Metric calibration layer against the mock stereo bench (noise-free
rendering, no session): ChArUco camera calibration, projector as an
inverse camera, stereoCalibrate, then triangulation of planes / cylinder /
sphere / step and the dent depths. The calibration is computed once per
module (~40 s); every check reuses it."""

import numpy as np
import pytest

from engine.bases.multifreq import MultiFreqBasis
from engine.calibrate import stereo
from engine.calibrate.charuco import BoardSpec
from engine.capture.mock_stereo import (Block, CharucoTarget, MetricCylinder, Sphere,
                                        StereoRig, StereoScene, board_poses, distort_project)
from engine.measure import metric
from engine.reconstruct.triangulate import triangulate

DENTS = [{"depth_mm": 1.0, "size_mm": 8.0, "angle_deg": -15, "z_mm": -20},
         {"depth_mm": 0.3, "size_mm": 6.0, "angle_deg": 15, "z_mm": 20}]


def _cal(fit_or_truth, gid="mock-g"):
    t = fit_or_truth
    if "cam" in t:
        return stereo.StereoCalibration(id="fit", geometry_id=gid, K_cam=t["cam"]["K"],
                                        dist_cam=t["cam"]["dist"], K_proj=t["proj"]["K"],
                                        dist_proj=t["proj"]["dist"], R=t["R"], t=t["t"],
                                        diamond_parity=1)
    return stereo.StereoCalibration(id="truth", geometry_id=gid, K_cam=t["K_cam"],
                                    dist_cam=t["dist_cam"], K_proj=t["K_proj"],
                                    dist_proj=t["dist_proj"], R=t["R"], t=t["t"],
                                    diamond_parity=t["parity"])


@pytest.fixture(scope="module")
def rig():
    return StereoRig.bench()


@pytest.fixture(scope="module")
def fit(rig):
    spec = BoardSpec()
    basis = MultiFreqBasis(steps=4, axis="xy", diamond_parity=1)
    pats = basis.generate()
    poses = []
    for R, t in board_poses(spec, 8, seed=0):
        sc = StereoScene(rig, [CharucoTarget(spec=spec, R=R, t=t)], gain=250)
        frames = [sc.render(p) for p in pats]
        fld = basis.decode(frames, "mock-g")
        poses.append({"image": np.clip(frames[0], 0, 255).astype(np.uint8),
                      "corr_x": fld.corr_x, "corr_y": fld.corr_y})
    return stereo.calibrate(poses, spec)


@pytest.fixture(scope="module")
def scene(rig):
    cyl = MetricCylinder(center=(0, 0, 799), radius=24.0, height=90, dents=DENTS)
    sph = Sphere(center=(-120, 40, 790), radius=15.0)
    blk = Block(center=(120, -60, 818), half=(30, 20, 5))       # a 10 mm step on the wall
    sc = StereoScene(rig, [cyl, sph, blk], gain=250)
    basis = MultiFreqBasis(steps=12, axis="xy", diamond_parity=1)
    fld = basis.decode([sc.render(p) for p in basis.generate()], "mock-g")
    return sc, cyl, fld


def test_camera_calibration_recovers_K(fit, rig):
    cam = fit["cam"]
    assert cam["rms"] < 0.3 and cam["ok"]
    assert max(cam["per_pose_rms"]) < 0.3
    K = rig.camera.K
    assert abs(cam["K"][0, 0] - K[0, 0]) < 2.0 and abs(cam["K"][1, 1] - K[1, 1]) < 2.0
    assert abs(cam["K"][0, 2] - K[0, 2]) < 1.5 and abs(cam["K"][1, 2] - K[1, 2]) < 1.5
    assert np.allclose(cam["dist"][:2], rig.camera.dist[:2], atol=0.02)


def test_projector_calibration_is_diamond_aware(fit, rig):
    pr = fit["proj"]
    assert pr["rms_u"] < 0.5 and pr["rms_v"] < 0.5 and pr["ok"]
    assert pr["fv_over_fu"] == pytest.approx(2.0, abs=0.01)
    Kp = rig.projector.K
    assert abs(pr["K"][0, 0] - Kp[0, 0]) < 3 and abs(pr["K"][1, 1] - Kp[1, 1]) < 6
    assert abs(pr["K"][0, 2] - Kp[0, 2]) < 2 and abs(pr["K"][1, 2] - Kp[1, 2]) < 3


def test_stereo_extrinsics_match_the_rig(fit, rig):
    d = stereo.compare_truth(_cal(fit), rig.truth())
    assert d["R_deg"] < 0.05 and d["t_mm"] < 1.0
    cal = _cal(fit)
    assert cal.baseline_mm == pytest.approx(279.4, abs=1.0)
    assert cal.angle_deg == pytest.approx(18.75, abs=0.1)


def test_triangulation_is_exact_with_true_geometry(rig, scene):
    """Oracle correspondences + the true calibration: the math itself (both methods)."""
    sc, _, _ = scene
    cx, cy, m = sc.oracle()
    from engine.decode.deformation import DeformationField
    fld = DeformationField(corr_x=cx, corr_y=cy, corr_diag=None, conf=m.astype(np.float32),
                           mask=m, geometry_id="mock-g", basis_names=["oracle"])
    g = sc.geometry()
    for method in ("plane", "rays"):
        mc = triangulate(fld, _cal(rig.truth()), method)
        err = np.linalg.norm(mc.xyz[mc.mask] - g["X"][mc.mask], axis=1)
        assert err.max() < 1e-6, method
        if method == "rays":
            assert np.nanmax(mc.consistency) < 1e-6
    # and the camera model agrees with cv2's projection convention
    X = np.ascontiguousarray(g["X"][m][::5000])
    uv = distort_project(X, rig.camera.K, rig.camera.dist)
    import cv2
    ref, _ = cv2.projectPoints(X, np.zeros(3), np.zeros(3), rig.camera.K, rig.camera.dist)
    assert np.abs(uv - ref.reshape(-1, 2)).max() < 1e-6


def test_triangulated_shapes_within_20_um(fit, scene):
    """Decoded 12-step multifreq + the FITTED calibration, no noise: plane,
    cylinder, sphere and step shape errors < 0.02 mm; both dents within 0.05 mm."""
    sc, cyl, fld = scene
    g = sc.geometry()
    mc = triangulate(fld, _cal(fit))
    wall = mc.mask & (g["shape"] == 3)
    assert metric.fit_plane(mc.xyz[wall]).rms < 0.02
    dev = metric.cylinder_deviation(mc, region=mc.mask & (g["shape"] == 0))
    assert abs(dev.fit.radius - 24.0) < 0.02
    healthy = dev.mask & (np.abs(dev.residual) < 0.2)       # off the dents
    assert np.sqrt(np.nanmean(dev.residual[healthy] ** 2)) < 0.02
    sph = metric.fit_sphere(mc.xyz[mc.mask & (g["shape"] == 1)])
    assert abs(sph.radius - 15.0) < 0.02
    step = metric.measure_known(mc, "step", {"height_mm": 10.0}, region=(g["shape"] == 2))
    assert abs(step["error"]) < 0.02
    depths = sorted(d.depth_mm for d in dev.dents)
    assert len(depths) == 2
    assert depths[0] == pytest.approx(0.3, abs=0.05) and depths[1] == pytest.approx(1.0, abs=0.05)
    assert mc.consistency_stats()["median_mm"] < 0.02


def test_consistency_map_flags_a_wrong_calibration(fit, scene):
    _, _, fld = scene
    good = triangulate(fld, _cal(fit)).consistency_stats()["median_mm"]
    bad_fit = dict(fit)
    bad_fit["t"] = np.asarray(fit["t"]) + [0, 3.0, 0]           # projector 3 mm off vertically
    bad = triangulate(fld, _cal(bad_fit)).consistency_stats()["median_mm"]
    assert bad > 10 * good and bad > 1.0


def test_wrong_row_parity_shows_as_a_zigzag(rig):
    """Decoding a diamond DMD as a square grid leaves a +-0.25 column zigzag:
    visible as depth noise on a flat wall (the reason for row_parity)."""
    sc = StereoScene(rig, [], gain=250)
    g = sc.geometry()
    out = {}
    for par in (1, None):
        b = MultiFreqBasis(steps=3, axis="x", diamond_parity=par)
        fld = b.decode([sc.render(p) for p in b.generate()], "mock-g")
        mc = triangulate(fld, _cal(rig.truth()), "plane")
        out[par] = metric.fit_plane(mc.xyz[mc.mask & g["lit"]]).rms
    assert out[1] < 0.02 and out[None] > 5 * out[1]


def test_mock_bench_switches_between_models():
    """Stereo shapes switch MockBench to the pinhole model; orthographic
    objects switch it back, so every existing mock scene is unchanged."""
    from engine.capture.mock_bench import Box, MockBench
    b = MockBench(seed=0)
    pat = np.full((1140, 912), 200, np.uint8)
    legacy = b.render(pat)
    b.place(Sphere(center=(0, 0, 790), radius=15))
    assert b.stereo_mode
    st = b.render(pat)
    assert st.shape == legacy.shape and not np.allclose(st, legacy)
    fld = b.oracle_field("mock-g")
    assert fld.corr_y is not None and fld.meta["stereo"]
    assert b.stereo_truth()["K_proj"][1, 1] == pytest.approx(2 * b.stereo_truth()["K_proj"][0, 0])
    b.clear()
    assert b.stereo_mode and not b.stereo_shapes                 # empty stereo wall
    b.place(Box(h_mm=10, w_mm=20, l_mm=20))
    assert not b.stereo_mode
    b.clear()
    assert np.array_equal(b.render(pat), legacy)

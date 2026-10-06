"""Metric measurement primitives on synthetic clouds (no rendering): shape
fits, cylinder deviation + dents, point-to-plane ICP, sigma_z predictor,
metric mesh export, ChArUco board files, row-parity helper."""

import math
import struct

import numpy as np
import pytest

from engine.calibrate import parity
from engine.calibrate.charuco import BoardSpec, detect, write_pdf, write_png
from engine.measure import metric, precision
from engine.reconstruct import mesh
from engine.reconstruct.triangulate import MetricCloud


def _cloud(xyz, mask=None):
    m = np.all(np.isfinite(xyz), axis=2) if mask is None else mask
    return MetricCloud(xyz=np.where(m[..., None], xyz, np.nan), mask=m, consistency=None,
                       method="rays", geometry_id="g", calibration_id="c")


def _cylinder_grid(R=24.0, zc=799.0, dents=(), n=(300, 240), noise=0.0, seed=0):
    """A standing cylinder (axis = y) in front of a wall at z = 823 on a camera-like
    grid: rows = y, columns = theta over +-60 deg. dents: (depth, size, theta_deg, y)."""
    rng = np.random.default_rng(seed)
    y = np.linspace(-40, 40, n[0])[:, None] * np.ones((1, n[1]))
    th = np.radians(np.linspace(-60, 60, n[1]))[None, :] * np.ones((n[0], 1))
    r = np.full(y.shape, R)
    for depth, size, tdeg, yc in dents:
        q = np.hypot(R * (th - math.radians(tdeg)), y - yc) / (size / 2)
        r -= np.where(q < 1, depth * 0.5 * (1 + np.cos(np.pi * q)), 0)
    r += rng.normal(0, noise, r.shape) if noise else 0
    xyz = np.stack([r * np.sin(th), y, zc - r * np.cos(th)], -1)
    return xyz


def test_plane_sphere_fits():
    rng = np.random.default_rng(1)
    P = rng.uniform(-50, 50, (5000, 3))
    P[:, 2] = 800 + 0.1 * P[:, 0] - 0.05 * P[:, 1]
    f = metric.fit_plane(P)
    assert f.rms < 1e-9 and f.normal[2] < 0
    Q = rng.normal(size=(4000, 3))
    Q = Q / np.linalg.norm(Q, axis=1, keepdims=True)
    Q = Q[Q[:, 2] < 0.2] * 15.0 + [10, -5, 780]            # a cap, as a camera sees it
    s = metric.fit_sphere(Q)
    assert s.diameter == pytest.approx(30.0, abs=1e-6)
    assert np.allclose(s.center, [10, -5, 780], atol=1e-6)


def test_cylinder_fit_ignores_dents_and_reports_them():
    xyz = _cylinder_grid(dents=[(1.0, 8.0, -15, -20), (0.3, 6.0, 15, 20)])
    cl = _cloud(xyz)
    dev = metric.cylinder_deviation(cl, region=cl.mask, axis_hint=(0, 1, 0))
    assert dev.fit.radius == pytest.approx(24.0, abs=2e-3)
    assert abs(dev.fit.axis[1]) == pytest.approx(1.0, abs=1e-5)
    assert len(dev.dents) == 2
    big, small = dev.dents
    assert big.depth_mm == pytest.approx(1.0, abs=0.02)
    assert small.depth_mm == pytest.approx(0.3, abs=0.02)
    assert big.theta_deg == pytest.approx(-15, abs=1.0) and big.z_mm == pytest.approx(-20, abs=1)
    assert big.area_mm2 > small.area_mm2 > 0
    assert np.nanmax(dev.residual) < 0.02                       # nothing sticks out


def test_cylinder_deviation_with_noise_keeps_dent_depth():
    xyz = _cylinder_grid(dents=[(1.0, 8.0, 0, 0)], noise=0.01, seed=3)
    cl = _cloud(xyz)
    dev = metric.cylinder_deviation(cl, region=cl.mask)
    assert dev.noise_sigma_mm == pytest.approx(0.01, rel=0.3)
    assert len(dev.dents) == 1 and dev.dents[0].depth_mm == pytest.approx(1.0, abs=0.03)


def test_foreground_and_known_measurements():
    H, W = 200, 300
    v, u = np.mgrid[0:H, 0:W].astype(float)
    xyz = np.stack([(u - W / 2) * 0.4, (v - H / 2) * 0.4, np.full((H, W), 823.0)], -1)
    top = (u > 100) & (u < 200) & (v > 50) & (v < 150)
    xyz[..., 2] = np.where(top, 823.0 - 5.0, 823.0)
    cl = _cloud(xyz)
    r = metric.measure_known(cl, "step", {"height_mm": 5.0})
    assert r["error"] == pytest.approx(0.0, abs=1e-6)
    assert metric.measure_known(cl, "plane", {})["measured"] < 1e-9
    rows = [{"measured": 5.01, "error": 0.01}, {"measured": 4.99, "error": -0.01},
            {"measured": 5.03, "error": 0.03}]
    s = metric.summarize(rows)
    assert s["verdict"] == "GO" and s["max_abs_error_mm"] == pytest.approx(0.03)
    assert s["repeatability_sigma_mm"] == pytest.approx(0.02)
    assert metric.summarize(rows, tol_mm=0.005)["verdict"] == "NO-GO"


def test_icp_point_to_plane_recovers_a_rigid_motion():
    xyz = _cylinder_grid(n=(120, 120))
    wall = np.stack(np.meshgrid(np.linspace(-60, 60, 80), np.linspace(-50, 50, 80)), -1)
    ph, th = np.meshgrid(np.linspace(0, 2 * np.pi, 60), np.linspace(0.05, 1.5, 30))
    ball = np.c_[(10 * np.sin(th) * np.cos(ph)).ravel() - 40, (10 * np.sin(th) * np.sin(ph)).ravel() + 30,
                 823.0 - 10 * np.cos(th).ravel()]                # breaks the slide along the axis
    G = np.vstack([xyz.reshape(-1, 3), np.c_[wall.reshape(-1, 2), np.full(6400, 823.0)], ball])
    a = math.radians(0.8)
    R = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
    t = np.array([0.6, -0.4, 0.5])
    S = (G - G.mean(0)) @ R.T + G.mean(0) + t                   # a moved copy
    res = metric.icp_point_to_plane(S, G, max_dist_mm=5.0)
    back = S @ res.R.T + res.t
    assert res.converged and res.rms_mm < 1e-3
    assert np.abs(back - G).max() < 5e-3


def test_sigma_z_predictor():
    r = precision.predict_sigma_z(823, 279.4, 1200, 8, A_dn=120, B_dn=100, steps=12,
                                  K_dn_per_e=0.0, read_noise_dn=1.5, quantization=False)
    sphi = math.sqrt(2 * 1.5 ** 2 / (12 * 100 ** 2))
    assert r["sigma_phi_rad"] == pytest.approx(sphi)
    assert r["sigma_z_mm"] == pytest.approx(8 / (2 * math.pi) * 823 ** 2 / (1200 * 279.4) * sphi)
    # sigma_z ~ T, ~ 1/B, ~ 1/sqrt(N), ~ z^2, ~ 1/sqrt(frames)
    base = r["sigma_z_mm"]
    kw = dict(K_dn_per_e=0.0, read_noise_dn=1.5, quantization=False)
    assert precision.predict_sigma_z(823, 279.4, 1200, 16, 120, 100, 12, **kw)["sigma_z_mm"] == pytest.approx(2 * base)
    assert precision.predict_sigma_z(823, 279.4, 1200, 8, 120, 200, 12, **kw)["sigma_z_mm"] == pytest.approx(base / 2)
    assert precision.predict_sigma_z(823, 279.4, 1200, 8, 120, 100, 48, **kw)["sigma_z_mm"] == pytest.approx(base / 2)
    assert precision.predict_sigma_z(823, 279.4, 1200, 8, 120, 100, 12, 4, **kw)["sigma_z_mm"] == pytest.approx(base / 2)
    shot = precision.predict_sigma_z(823, 279.4, 1200, 8, 120, 100, 12)
    assert shot["sigma_z_mm"] > 0 and shot["units"] == "mm"
    with pytest.raises(ValueError):
        precision.predict_sigma_z(823, 0, 1200, 8, 120, 100)


def test_metric_mesh_is_open_and_does_not_bridge_depth_jumps(tmp_path):
    xyz = _cylinder_grid(n=(60, 80))
    xyz[:, 40:] += [0, 0, 30.0]                                  # a 30 mm depth jump
    xyz[20:25, 10:15] = np.nan                                   # a hole stays a hole
    cl = _cloud(xyz)
    p, mp, meta = mesh.cloud_mesh(cl, tmp_path / "m.stl", "stl")
    hdr = mesh.read_stl_header(p)
    assert hdr["size_ok"] and hdr["finite"] and meta["units"]["xyz"] == "mm"
    assert hdr["n_triangles"] == meta["n_triangles"]
    full = 2 * 59 * 79
    lost_to_jump = 2 * 59
    lost_to_hole = 2 * 6 * 6
    assert meta["n_triangles"] == full - lost_to_jump - lost_to_hole
    p2, _, _ = mesh.cloud_mesh(cl, tmp_path / "m.obj", "obj", stride=2)
    assert p2.read_text().count("\nf ") > 0
    with pytest.raises(Exception):
        mesh.cloud_mesh(cl, tmp_path / "m.stl", "stl")           # never overwritten


def test_charuco_files_and_detection(tmp_path):
    spec = BoardSpec(squares_x=6, squares_y=5, square_mm=20, marker_mm=15)
    png = write_png(tmp_path / "b.png", spec, dpi=150)
    data = png.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n" and b"pHYs" in data
    ppm = struct.unpack(">I", data[data.find(b"pHYs") + 4:][:4])[0]
    assert ppm == round(150 / 0.0254)
    pdf = write_pdf(tmp_path / "b.pdf", spec, dpi=150, margin_mm=10)
    txt = pdf.read_bytes()
    assert txt.startswith(b"%PDF-1.4") and txt.rstrip().endswith(b"%%EOF")
    w_pt = (6 * 20 + 20) / 25.4 * 72
    assert f"/MediaBox [0 0 {w_pt:.4f}".encode() in txt
    img = spec.image(8.0, margin_mm=10)
    det = detect(img, spec)
    assert det is not None and len(det[1]) == spec.n_corners
    # corners land where the board mm say (8 px / mm, 10 mm margin, pixel centres)
    assert np.abs(det[0] - (det[2][:, :2] * 8 + 80 - 0.5)).max() < 0.2
    assert spec.scaled(20.2).square_mm == 20.2 and spec.scaled(20.2).marker_mm == pytest.approx(15.15)
    with pytest.raises(ValueError):
        BoardSpec(square_mm=10, marker_mm=12)


def test_parity_helper_on_ideal_frames():
    """Synthetic camera: 2.5 px per DMD column, odd rows shifted half a column right."""
    pats = parity.parity_patterns()

    def camera(pat, shift_odd):
        x = np.arange(1200) / 2.5                                # camera x -> DMD column
        rows = []
        for r in (100, 101):                                     # one even, one odd row
            sh = shift_odd if r % 2 else 0.0
            c = x - sh
            c0 = np.clip(np.floor(c).astype(int), 0, 910)
            f = c - c0
            rows.append((1 - f) * pat[r, c0] + f * pat[r, c0 + 1])
        return np.tile(np.mean(rows, 0), (50, 1))

    for sh, want in ((0.5, 1), (-0.5, 0), (0.0, None)):
        rep = parity.measure_parity([camera(p, sh) for p in pats])
        assert rep["ok"] and rep["parity"] == want, rep
        assert rep["shift_cols"] == pytest.approx(sh, abs=0.05)
    inv = parity.measure_parity([255 - camera(p, 0.5) for p in pats])     # inverted HDMI path
    assert inv["parity"] == 1

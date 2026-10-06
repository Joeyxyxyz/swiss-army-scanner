"""Checkerboard corner basis + baked map stack, mock bench only."""

import numpy as np
import pytest
from scipy import ndimage

from engine import api
from engine.bases.checkerboard import (CheckerboardBasis, capture_checker, local_jacobian)
from engine.capture.mock_bench import Box
from engine.measure import maps as maps_mod

from conftest import PRESET

BOX = Box(h_mm=40.0, w_mm=60.0, l_mm=50.0, center_mm=(0.0, 0.0))


@pytest.fixture
def fast(eng):
    p = api.load_preset(PRESET)
    p.camera.average = 1
    return p


def _truth_cam(bench, dmd):
    """Flat base: DMD (X, Y) -> camera (u, v) by inverting MockBench's corr_x / corr_y."""
    H, W = bench.cam_shape
    pw, ph = bench.lit_fraction * W, bench.lit_fraction * H
    dh, dw = bench.dmd_shape
    return np.stack([dmd[:, 0] * pw / dw + W / 2 - pw / 2,
                     dmd[:, 1] * ph / dh + H / 2 - ph / 2], 1)


def test_checker_frames_and_polarity_invariance():
    b = CheckerboardBasis(square_px=32, shifts=2)
    frames = b.generate((1140, 912))
    assert len(frames) == b.n_frames()
    assert all(f.shape == (1140, 912) and f.dtype == np.uint8 for f in frames)
    with pytest.raises(ValueError):
        CheckerboardBasis(square_px=2)


def test_checker_corners_subpixel_vs_truth(eng, fast):
    bench = api.mock_bench()
    cs = capture_checker(eng, fast, square_px=32, label="flat")
    c = cs.corners
    assert len(c) > 500                                   # ~28 x 35 corners on the patch
    err = np.hypot(*(c.cam - _truth_cam(bench, c.dmd)).T)
    # x is continuous in the mock; y is quantised to whole mirror rows (rint), so the
    # edge itself sits anywhere within a camera px: <= 0.5 px, median well under
    assert np.median(err) < 0.4
    assert err.max() < 0.75
    dx = c.cam[:, 0] - _truth_cam(bench, c.dmd)[:, 0]
    assert abs(np.median(dx)) < 0.05 and np.percentile(np.abs(dx), 95) < 0.2
    j = cs.jacobian
    H, W = bench.cam_shape
    assert np.median(j.mag_x) == pytest.approx(0.8 * W / 912, rel=0.01)
    # rows are rint-quantised in the mock: cell heights are 24 or 25 px, so average
    assert np.mean(j.mag_y) == pytest.approx(0.8 * H / 1140, rel=0.01)
    assert np.nanmax(np.abs(j.skew_deg)) < 1.0
    assert np.nanmax(np.abs(j.stretch - 1)) < 0.03
    assert np.nanmedian(j.contrast) > 0.9                 # the mock is in focus everywhere
    assert c.pairs().shape == (len(c), 4)
    # outputs recorded in the bin
    kinds = {r["kind"] for r in eng.store.outputs()}
    assert {"measurements", "previews"} <= kinds
    assert cs.npz_path.exists() and cs.json_path.exists() and "stretch" in cs.png_paths


def test_jacobian_raised_box_differs_from_flat(eng, fast):
    bench = api.mock_bench()
    flat = capture_checker(eng, fast, square_px=32, label="flat")
    bench.place(BOX)
    obj = capture_checker(eng, fast, square_px=32, label="box", reference=flat)
    j = obj.jacobian
    assert j.stretch_ref == "reference"
    # footprint in camera px
    H, W = bench.cam_shape
    k = bench.cam_px_per_mm
    near = ((np.abs(j.center_cam[:, 0] - W / 2) < (BOX.w_mm / 2) * k + 40)
            & (np.abs(j.center_cam[:, 1] - H / 2) < (BOX.l_mm / 2) * k + 40))
    far = ~((np.abs(j.center_cam[:, 0] - W / 2) < (BOX.w_mm / 2) * k + 120)
            & (np.abs(j.center_cam[:, 1] - H / 2) < (BOX.l_mm / 2) * k + 120))
    dev = np.abs(j.stretch - 1)
    assert np.nanmax(dev[near]) > 0.2                     # cells straddling the box edge
    assert np.nanmax(dev[far]) < 0.02                     # the flat base is unchanged
    # interpolated image has the camera shape
    img = j.image("stretch")
    assert img.shape == bench.cam_shape


def test_local_jacobian_synthetic_tilt():
    """A pure affine map DMD -> camera gives exactly its matrix per cell."""
    from engine.bases.checkerboard import CheckerCorners
    s = 20
    A = np.array([[1.5, 0.2], [0.1, 0.8]])
    k, m = np.meshgrid(np.arange(1, 8), np.arange(1, 6))
    idx = np.stack([k.ravel(), m.ravel()], 1)
    dmd = idx * s - 0.5
    cam = dmd @ A.T + 100.0
    c = CheckerCorners(cam=cam, dmd=dmd.astype(float), index=idx, shift=np.zeros_like(idx),
                       response=np.ones(len(idx)), square_px=s, cam_square_px=30.0,
                       shape=(400, 400))
    j = local_jacobian(c)
    assert len(j) == 6 * 4
    assert np.allclose(j.J, A)
    assert np.allclose(j.mag_x, np.hypot(1.5, 0.1))


def test_bake_maps_box_scan(eng, fast):
    bench = api.mock_bench()
    base = api.capture_base(fast, "base")
    bench.place(BOX)
    scan = api.capture_object(fast, base, "box")
    ms = maps_mod.bake_maps(eng, scan, calibration=None)

    valid = scan.relief_mask & np.isfinite(scan.relief)
    assert np.array_equal(ms.validity, valid)
    assert np.all(np.isfinite(ms.normal[valid]))
    assert np.allclose(np.linalg.norm(ms.normal[valid], axis=-1), 1.0, atol=1e-4)
    assert np.all(np.isnan(ms.height[~valid]))
    assert set(ms.units) >= {"height", "cavity", "normal", "validity", "density"}
    assert "rule 6" in ms.units["height"]

    # cavity highlights the box outline, not its flat top or the flat base
    H, W = bench.cam_shape
    fp = bench.height_map() > 0
    edge = (ndimage.binary_dilation(fp, iterations=4) & ~ndimage.binary_erosion(fp, iterations=4))
    top = ndimage.binary_erosion(fp, iterations=20)
    vv, uu = np.mgrid[0:H, 0:W]
    flat = valid & (np.abs(uu - W / 2) > 300) & (np.abs(uu - W / 2) < 500)
    cav = np.abs(np.nan_to_num(ms.cavity))
    e = np.percentile(cav[edge & valid], 90)
    assert e > 10 * np.percentile(cav[top & valid], 90)
    assert e > 10 * np.percentile(cav[flat], 90)
    # top edge is convex (< 0) where the base is lower on the outside... the sign flips
    # across the edge: both signs present in the edge band
    ce = ms.cavity[edge & valid]
    assert (ce > 0).any() and (ce < 0).any()

    # outputs: npz + png per map in previews, json in measurements, recorded vs the scan
    assert set(ms.npz_paths) == set(ms.png_paths) == {"height", "cavity", "normal",
                                                      "validity", "density"}
    assert ms.json_path.exists()
    rows = eng.store.outputs()
    files = {r["file"] for r in rows}
    for p in list(ms.npz_paths.values()) + list(ms.png_paths.values()) + [ms.json_path]:
        assert eng.store._rel(p) in files
    assert all(r["source"] for r in rows if r["label"] == "box_maps")
    with np.load(ms.npz_paths["cavity"]) as z:
        assert str(z["units"])


def test_bake_maps_albedo_subsurface_dust(eng, fast):
    bench = api.mock_bench()
    base = api.capture_base(fast, "base")
    golden = api.spectral_capture(fast, "golden")        # empty scene reflectance
    bench.place(Box(h_mm=20.0, w_mm=40.0, l_mm=40.0, albedo=0.3))
    scan = api.capture_object(fast, base, "dark_box")
    reading = api.spectral_capture(fast, "obj")
    light = api.separate_light(fast, "obj_light")
    ms = maps_mod.bake_maps(eng, scan, spectral=reading, light=light, golden=golden)
    for n in ("albedo", "subsurface", "dust"):
        assert getattr(ms, n) is not None and n in ms.units and n in ms.png_paths
    fp = ndimage.binary_erosion(bench.height_map() > 0, iterations=10)
    # the box is darker than the base it covers: dust (albedo - golden) is negative there
    assert np.nanmedian(ms.dust[fp]) < -0.2
    assert np.nanmedian(np.abs(ms.subsurface)) < 0.2     # no global light in this bench


def test_decode_returns_interpolated_field():
    """decode() via the PatternBasis interface, frames rendered straight from the
    bench (no rig): corr_x / corr_y between corners match the bench's truth."""
    from engine.bases import registry
    from engine.capture.mock_bench import MockBench
    bench = MockBench(seed=3)
    b = CheckerboardBasis(square_px=32)
    frames = [bench.render(p) for p in b.generate(bench.dmd_shape)]
    fld = b.decode(frames, "g")
    assert registry.is_implemented("checkerboard")
    assert fld.meta["n_corners"] > 500 and fld.mask.mean() > 0.4
    _, _, valid, cx, cy = bench._scene()
    m = fld.mask & valid
    assert np.median(np.abs(fld.corr_x[m] - cx[m])) < 0.3
    assert np.median(np.abs(fld.corr_y[m] - cy[m])) < 0.6

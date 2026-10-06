"""
Golden-model comparison against known ground truth.

A synthetic box STL is rasterized to a reference height map; the
'measured' surface is the same map with a Gaussian dent of known depth
at a known location. The comparison must (1) pass a clean part,
(2) fail the dented part, (3) report the dent's depth and location.
"""
import sys; sys.path.insert(0, ".")
import os
import tempfile
import numpy as np

from swiss_army_sensor.golden import (
    load_stl, save_stl, make_box_stl, rasterize_heightmap,
    ToleranceBand, compare_to_golden,
)


def _box_reference(grid=(120, 160)):
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "box.stl")
        make_box_stl(p, size=(40.0, 30.0, 10.0))
        v, f = load_stl(p)
        assert len(f) == 12 and len(v) == 8, "STL round-trip broken"
        ref = rasterize_heightmap(v, f, grid=grid)
    return ref


def test_stl_roundtrip_binary():
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "box.stl")
        make_box_stl(p)
        v, f = load_stl(p)
        assert v.shape == (8, 3) and f.shape == (12, 3)
    print("[OK] binary STL write/read round-trips (8 verts, 12 tris)")


def test_rasterized_box_is_flat_top():
    ref = _box_reference()
    top = ref.height[20:100, 20:140]
    assert np.nanmax(np.abs(top - 10.0)) < 1e-9, "box top should be z=10"
    assert ref.footprint.mean() > 0.95, "box should cover ~full grid"
    print("[OK] rasterizer: box top is exactly z=10 over the footprint")


def test_clean_part_passes_and_dent_fails():
    ref = _box_reference()
    H, W = ref.height.shape
    tol = ToleranceBand(warn=0.10, fail=0.30)
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 0.01, (H, W))

    clean = ref.height + noise
    r1 = compare_to_golden(clean, ref, tol)
    assert r1.passed, f"clean part failed: {r1.pct_fail:.2f}% fail px"
    assert r1.pct_in_tolerance > 99.0

    # dent: 0.8 deep Gaussian at (40, 110), sigma 4 px
    yy, xx = np.mgrid[0:H, 0:W]
    dent_depth, cy, cx = 0.8, 40, 110
    dent = dent_depth * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 4.0 ** 2))
    dented = ref.height + noise - dent
    r2 = compare_to_golden(dented, ref, tol)
    assert not r2.passed, "dented part passed"
    assert abs(abs(r2.max_under) - dent_depth) < 0.1, (
        f"dent depth misreported: {r2.max_under}")
    wy, wx = r2.worst_px
    assert abs(wy - cy) <= 3 and abs(wx - cx) <= 3, (
        f"dent located at {(wy, wx)}, truth {(cy, cx)}")
    print(f"[OK] clean part passes ({r1.pct_in_tolerance:.1f}% in tol); "
          f"0.8-deep dent fails, depth {-r2.max_under:.2f} at {r2.worst_px} "
          f"(truth 0.80 at ({cy}, {cx}))")


def test_xy_footprint_shift_detected():
    # pad the bounds so the part has a real silhouette inside the grid
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "box.stl")
        make_box_stl(p, size=(40.0, 30.0, 10.0))
        v, f = load_stl(p)
        ref = rasterize_heightmap(v, f, grid=(120, 160),
                                  bounds=(-10.0, 50.0, -10.0, 40.0))
    assert 0.3 < ref.footprint.mean() < 0.8, "box should be an island"
    tol = ToleranceBand(warn=0.10, fail=0.30)
    shifted = np.roll(np.nan_to_num(ref.height, nan=np.nan), 8, axis=1)
    shifted_fp = np.isfinite(shifted)
    r = compare_to_golden(np.nan_to_num(shifted, nan=0.0), ref, tol,
                          tol_xy_px=3.0, measured_footprint=shifted_fp,
                          max_fail_px=10_000_000)      # isolate the XY check
    assert np.isfinite(r.xy_boundary_offset_px), "degenerate: no silhouette"
    assert r.xy_boundary_offset_px > 3.0
    assert not r.passed, "8-px footprint shift passed a 3-px XY tolerance"

    aligned_fp = ref.footprint
    r_ok = compare_to_golden(np.nan_to_num(ref.height, nan=0.0), ref, tol,
                             tol_xy_px=3.0, measured_footprint=aligned_fp)
    assert r_ok.passed and r_ok.xy_boundary_offset_px < 0.5
    print(f"[OK] silhouette check: aligned part offset "
          f"{r_ok.xy_boundary_offset_px:.2f} px (passes); 8-px shift gives "
          f"{r.xy_boundary_offset_px:.1f} px (fails 3-px tolerance)")


if __name__ == "__main__":
    test_stl_roundtrip_binary()
    test_rasterized_box_is_flat_top()
    test_clean_part_passes_and_dent_fails()
    test_xy_footprint_shift_detected()

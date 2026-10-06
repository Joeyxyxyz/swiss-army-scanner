"""Atlas (Surfer held field) + Surfer verbs, mock only."""

import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from engine import api
from engine.capture import atlas as A
from engine.capture import surfer as S
from engine.capture import zones as Z

from conftest import PRESET

SRC = Path(os.environ.get("SCANNER_TEST_ATLAS", ""))   # a real held-cell atlas.json; skipped if unset
FACE = (720, 768, 768, 1056)
FLOOR = (720, 1056, 768, 1104)


@pytest.fixture
def atlas_file(tmp_path):
    if not SRC.is_file():
        pytest.skip("real atlas file not present")
    dst = tmp_path / "cell_held_atlas.json"
    shutil.copyfile(SRC, dst)                 # the source is read-only for us
    return dst


@pytest.fixture
def fast(eng):
    p = api.load_preset(PRESET)
    p.camera.average = 1
    return p


def expected_cam_rect(bench, rect):
    """Mock bench: DMD rect -> camera rect (x0, y0, x1, y1), ignoring relief."""
    H, W = bench.cam_shape
    dh, dw = bench.dmd_shape
    pw, ph = bench.lit_fraction * W, bench.lit_fraction * H
    u0, v0 = W / 2 - pw / 2, H / 2 - ph / 2
    x0, y0, x1, y1 = rect
    return (u0 + x0 * pw / dw, v0 + y0 * ph / dh, u0 + x1 * pw / dw, v0 + y1 * ph / dh)


def test_load_real_atlas(atlas_file):
    a = A.load_atlas(atlas_file, "cell")
    assert a.step_px == 48 and a.dmd_size == (912, 1140) and not a.warnings
    face, floor = a.region("cell"), a.region("floor")
    assert face.rect == FACE and floor.rect == FLOOR
    assert face.zone_spec == "P17:P22@48" and floor.zone_spec == "P23@48"
    g = Z.grid_spec(48)
    for reg in (face, floor):
        text, sp = Z._split_spacing(reg.zone_spec)
        assert (Z.projector_mask(Z.parse_zone(text, g), Z.grid_spec(sp)) == reg.mask()).all()
    assert face.mask().sum() == 48 * 288
    with pytest.raises(ValueError, match="no region"):
        a.region("Q")


def test_label_rect_mismatch_warns(tmp_path):
    doc = {"step_px": 48, "dmd": {"size": [912, 1140], "held_cell": [720, 768, 816, 1056]},
           "grid_labels": {"cell": "P17-P22"}}
    a = A.atlas_from_doc(doc, "x")
    assert a.warnings and a.region("cell").rect == (720, 768, 816, 1056)
    assert a.region("cell").zone_spec == "P17:Q22@48"          # rebuilt from the rect


def test_import_list_resolve(eng, fast, atlas_file):
    before = atlas_file.read_bytes()
    p1 = A.import_atlas(eng, atlas_file, "cell", fast.geometry_id)
    p2 = A.import_atlas(eng, atlas_file, "cell", fast.geometry_id)
    assert p1 != p2 and p1.exists() and p2.exists()           # append-only
    assert atlas_file.read_bytes() == before
    root = eng.store.root.resolve()
    assert root in p1.resolve().parents
    rows = A.list_atlases(eng, fast.geometry_id)
    assert rows[0]["name"] == "cell" and rows[0]["versions"] == 2
    pm, cam, cells = A.resolve_region(eng, fast, "atlas:cell/cell")
    assert (pm == A.rect_mask(FACE)).all() and cam is None     # no map yet
    assert len(cells) == 6 and cells[0] == (15, 16)
    with pytest.raises(ValueError, match="atlas:<name>/<region>"):
        A.resolve_region(eng, fast, "atlas:cell")
    with pytest.raises(ValueError, match="no atlas"):
        A.resolve_region(eng, fast, "atlas:nope/cell")
    api.capture_map(fast, "m")
    pm, cam, _ = A.resolve_region(eng, fast, "atlas:cell/cell")
    assert cam is not None and cam.any()


def test_flood_verify_lands_in_expected_area(eng, fast, atlas_file):
    A.import_atlas(eng, atlas_file, "cell", fast.geometry_id)
    r = S.flood_verify(eng, fast, "atlas:cell/cell")
    assert r.lands, r.verdict
    ex = expected_cam_rect(eng.bench, FACE)
    x0, y0, x1, y1 = r.lit_bbox
    tol = 6
    assert x0 >= ex[0] - tol and y0 >= ex[1] - tol and x1 <= ex[2] + tol and y1 <= ex[3] + tol
    assert r.lit_area_px > 0.7 * (ex[2] - ex[0]) * (ex[3] - ex[1])
    assert abs(r.outside_mean_counts) < 1.0                   # the surround stays dark
    assert r.json_path.exists() and r.camera_png is not None
    # a region off the lit part does not land
    off = S.flood_verify(eng, fast, (0, 0, 16, 16), label="off")
    assert off.lit_area_px < r.lit_area_px


def test_flood_verify_inverted_path_keeps_surround_black(eng, fast):
    fast.projection.invert = False
    from engine.capture import projector as P
    orig = P.MockWindow.__init__

    def inv(self, *a, **k):
        orig(self, *a, **k)
        self.hdmi_inverts = True
    P.MockWindow.__init__ = inv
    try:
        r = S.flood_verify(eng, fast, FACE)
    finally:
        P.MockWindow.__init__ = orig
    assert r.polarity_inverted and r.lands
    assert abs(r.outside_mean_counts) < 1.0


def test_anti_glare_sweep_locks(eng, fast):
    r = S.anti_glare_sweep(eng, fast, FACE, carriers=[(0, 0), (0, -0.5), (0.3, -0.5, 0.7)])
    assert r.lock is not None and r.lock != (0.0, 0.0, 0.0)    # a flat flood has no modulation
    assert len(r.scores) == 3 and "not optical wavefront" in r.wording
    assert r.json_path.exists()


def test_multi_orientation_scan_finite(eng, fast):
    r = S.multi_orientation_scan(eng, fast, (672, 720, 816, 1056), steps=4, period_px=24)
    assert set(r.phase) == {"v", "h", "d45", "d135"}
    assert r.mask.any()
    assert np.isfinite(r.fused_phase[r.mask]).all()
    assert np.isfinite(r.fused_modulation).all() and np.isfinite(r.quality).all()
    assert (r.quality[r.mask] <= 1.0001).all()
    assert r.stats["median_fused_modulation"] > S.MOD_MIN
    assert "NOT" in r.note and r.npz_path.exists()


def test_nothing_written_outside_library(eng, fast, atlas_file, tmp_path):
    before = {p for p in tmp_path.rglob("*")}
    A.import_atlas(eng, atlas_file, "cell", fast.geometry_id)
    S.flood_verify(eng, fast, FACE)
    lib = eng.store.root.resolve()
    new = {p for p in tmp_path.rglob("*")} - before
    assert new and all(lib == p.resolve() or lib in p.resolve().parents for p in new)

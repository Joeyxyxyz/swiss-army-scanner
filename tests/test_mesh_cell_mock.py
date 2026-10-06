"""Mock: cylinder fit / unwrap / dents, and STL/OBJ meshes (engine/measure/cell.py,
engine/reconstruct/mesh.py). Captures are shared per module (each costs ~30 s)."""

import json

import numpy as np
import pytest

from engine import api
from engine.capture.mock_bench import Box, Cylinder, MockBench
from engine.measure import cell
from engine.reconstruct import mesh

from conftest import PRESET

R, HGT, CENTER = 24.0, 48.0, (10.0, 5.0)
DENT = dict(depth_mm=3.0, size_mm=10.0, angle_deg=20.0, z_mm=30.0)


@pytest.fixture(scope="module")
def rig(tmp_path_factory):
    root = tmp_path_factory.mktemp("meshcell")
    api.configure(root / "library", mock=True, bench=MockBench(seed=1))
    p = api.load_preset(PRESET)
    base = api.capture_base(p, "empty")
    bench = api.mock_bench()
    bench.place(Box(40.0, 50.0, 30.0, center_mm=(-60, 0)))
    box = api.capture_object(p, base, "box", known_dims={"H": 40.0, "W": 50.0, "L": 30.0})
    raw = {}
    bench.clear()
    bench.place(Cylinder(R, HGT, center_mm=CENTER, dents=[DENT]))
    raw["dented"] = api.capture_object(p, base, "cell_dented")
    uncal_unwrap = cell.unwrap_cell(raw["dented"], None)         # before any calibration
    cal = api.calibrate(p, [box])
    bench.clear()
    bench.place(Cylinder(R, HGT, center_mm=CENTER))
    raw["good"] = api.capture_object(p, base, "cell_good")
    yield {"root": root, "cal": cal, "base": base, "uncal": uncal_unwrap, **raw}
    api._session = None


def test_mock_cylinder_is_an_open_arc():
    b = MockBench(seed=0)
    b.place(Cylinder(R, HGT))
    h, _, valid, _, _ = b._scene()
    assert h.max() == pytest.approx(2 * R, abs=0.5)           # back touches the wall
    row = valid[b.cam_shape[0] // 2]
    seen_px = int(row[np.abs(np.arange(row.size) - b.cam_shape[1] / 2) < R * b.cam_px_per_mm].sum())
    assert seen_px == pytest.approx(2 * R * np.sin(np.radians(60)) * b.cam_px_per_mm, abs=4)


def test_fit_unwrap_and_dent_found(rig):
    u = cell.unwrap_cell(rig["dented"], rig["cal"])
    assert u.units == "mm" and u.calibration_id == rig["cal"].id
    assert u.radius == pytest.approx(R, rel=0.05)
    assert u.cell_height == pytest.approx(HGT, abs=3.0)
    assert 90 < u.arc_deg < 180                                 # honest open arc
    assert np.abs(u.theta_deg).max() < 65                      # grid holds only seen angles
    rep = cell.find_dents(u, depth_trigger_mm=0.7)
    assert rep.verdict == "FAIL" and len(rep.dents) == 1
    d = rep.dents[0]
    assert d.theta_deg == pytest.approx(DENT["angle_deg"], abs=4)
    assert d.z == pytest.approx(DENT["z_mm"], abs=3)
    assert 1.5 < d.depth < 4.0 and d.exceeds_trigger


def test_no_dent_no_detection(rig):
    u = cell.unwrap_cell(rig["good"], rig["cal"])
    rep = cell.find_dents(u, depth_trigger_mm=0.7)
    assert rep.dents == []
    assert rep.verdict in ("PASS", "INDETERMINATE")
    if rep.verdict == "INDETERMINATE":
        assert any("cannot PASS" in w for w in rep.warnings)
    # a trigger below the noise can never PASS
    tiny = cell.find_dents(u, depth_trigger_mm=0.01)
    assert tiny.verdict == "INDETERMINATE" and any("cannot PASS" in w for w in tiny.warnings)


def test_uncalibrated_stays_px_no_verdict(rig):
    u = rig["uncal"]
    assert u.units == "px" and u.calibration_id == "uncalibrated"
    assert any("rule 6" in w for w in u.warnings)
    rep = cell.find_dents(u, depth_trigger_mm=0.7)
    assert rep.verdict == "INDETERMINATE" and rep.trigger is None


def _check_stl(path, meta_path):
    info = mesh.read_stl_header(path)
    assert info["size_ok"] and info["n_triangles"] > 100 and info["finite"]
    meta = json.loads(meta_path.read_text())
    for k in ("geometry_id", "units", "calibration_id", "finite_vertices",
              "n_vertices", "n_triangles", "zone"):
        assert k in meta
    assert meta["finite_vertices"] is True and meta["n_triangles"] == info["n_triangles"]
    return meta


def test_meshes_open_finite_with_meta(rig, tmp_path):
    sc, cal = rig["dented"], rig["cal"]
    p, mp, _ = mesh.heightmap_to_stl(sc, cal, tmp_path / "relief.stl", stride=4)
    meta = _check_stl(p, mp)
    assert meta["units"] == {"xy": "mm", "z": "mm"} and meta["calibration_id"] == cal.id
    p, mp, _ = mesh.heightmap_to_stl(sc, None, tmp_path / "relief_px.stl", stride=4,
                                     roi=(500, 400, 900, 700))
    meta = _check_stl(p, mp)
    assert meta["calibration_id"] == "uncalibrated" and meta["to_scale"] is False
    p, mp, _ = mesh.base_to_stl(rig["base"], tmp_path / "base.stl", stride=6)
    _check_stl(p, mp)
    u = cell.unwrap_cell(sc, cal)
    p, mp, _ = mesh.cylinder_arc_mesh(u, tmp_path / "cyl.stl")
    meta = _check_stl(p, mp)
    assert meta["open_arc_only"] and meta["arc_deg"] < 180
    p, mp, meta = mesh.cylinder_arc_mesh(u, tmp_path / "cyl.obj", fmt="obj")
    txt = p.read_text()
    assert txt.count("\nv ") + txt.startswith("v ") == meta["n_vertices"]
    assert "nan" not in txt.lower()
    with pytest.raises(Exception):                            # never overwrite
        mesh.cylinder_arc_mesh(u, tmp_path / "cyl.stl")


def test_unwrap_outputs(rig, tmp_path):
    u = cell.unwrap_cell(rig["dented"], rig["cal"])
    cell.save_unwrap(tmp_path / "u.npz", u)
    z = np.load(tmp_path / "u.npz")
    assert z["dent_map"].shape == u.valid.shape and float(z["radius"]) == pytest.approx(u.radius)
    png = cell.dent_png(u, cell.find_dents(u, 0.7))
    assert png is None or png[:4] == b"\x89PNG"

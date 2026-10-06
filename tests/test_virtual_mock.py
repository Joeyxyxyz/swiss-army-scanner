"""Virtual sensors (engine/sense/virtual.py) end to end in mock mode:
Laser-spot-sensor-style spot sizing, projected spots, part-present (photoelectric and
background suppression) and alignment. Nothing here touches hardware."""

import json
import sys

import numpy as np
import pytest

from engine import api, rules
from engine.capture import camera as camera_mod
from engine.capture import projector as projector_mod
from engine.capture.mock_bench import Box, MockBench
from engine.capture.projector import MockWindow
from engine.decode import projmap
from engine.library import store as store_mod
from engine.sense import virtual as V

from conftest import PRESET

BENCH = MockBench()                          # default geometry, for ground truth only
RELIEF_PX_PER_MM = BENCH.relief_px_per_mm    # 0.45 DMD px of relief per mm of height
CAM_PER_COL = 1152 / 912                     # mock: camera px per DMD column (lit patch 0.8 x 1440)
CAM_PER_ROW = 864 / 1140                     # mock: camera px per DMD row
SPOT_AT = {"dmd": [456, 570]}                # lands at camera (720, 540) = base-plane (0, 0) mm
NOMINAL_MM_PER_DMD_PX = 823.0 / 1.4 / 912    # the preset's rig: standoff / throw ratio / 912


def cam_of_dmd(col, row):
    """Mock bench, empty scene: camera px [x, y] that sees DMD mirror (col, row)."""
    return np.array([144 + col * CAM_PER_COL, 108 + row * CAM_PER_ROW])


@pytest.fixture(autouse=True)
def no_hardware(monkeypatch):
    """Any attempt to open the real HDMI window or the real camera fails the test."""
    def boom(*a, **k):
        raise AssertionError("real hardware touched in a mock test")
    monkeypatch.setattr(projector_mod.HDMIWindow, "__init__", boom)
    monkeypatch.setattr(camera_mod.RealCamera, "__init__", boom)
    yield
    assert not [m for m in rules.FORBIDDEN_SCAN_MODULES if m in sys.modules]


@pytest.fixture
def fast(eng):
    p = api.load_preset(PRESET)
    p.camera.average = 1
    return p


def outputs():
    return api.list_library("outputs")


def out_kind():
    return "sensors" if "sensors" in store_mod.OUTPUT_KINDS else "previews"


def recorded(path) -> bool:
    rows = outputs()
    return any(r["file"].endswith(path.name) and r["kind"] == out_kind() for r in rows)


# ------------------------------------------------------------ spot sizing

def test_spot_sizing_nominal_from_rig_geometry(fast):
    s = V.spot_px_for_mm(fast, 6.5)
    assert s.source == "nominal"
    assert s.mm_per_dmd_px == pytest.approx(NOMINAL_MM_PER_DMD_PX)      # 0.6446 mm / DMD px
    assert s.diameter_dmd_px == pytest.approx(6.5 / NOMINAL_MM_PER_DMD_PX, abs=1e-3)   # 10.08
    assert s.diameter_dmd_rows == pytest.approx(2 * s.diameter_dmd_px, abs=1e-3)   # diamond rows
    assert any("nominal" in w for w in s.warnings)
    assert s.units["diameter_dmd_px"].startswith("DMD px")
    # the seed's laser-spot reference numbers: 6.5 mm at 300 mm working distance ~ 14 DMD px radius
    fast.rig.standoff_mm = 300.0
    assert V.spot_px_for_mm(fast, 6.5).diameter_dmd_px / 2 == pytest.approx(13.8, abs=0.1)


def test_spot_sizing_measured_override(fast):
    s = V.spot_px_for_mm(fast, 6.5, mm_per_dmd_px=0.25)
    assert (s.source, s.mm_per_dmd_px, s.diameter_dmd_px) == ("measured", 0.25, 26.0)
    assert not any("nominal" in w for w in s.warnings)
    small = V.spot_px_for_mm(fast, 0.1, mm_per_dmd_px=0.25)
    assert any("below one mirror column" in w for w in small.warnings)


def test_spot_sizing_rejects_bad_input(fast):
    for bad in (0, -1, "6.5", True, float("nan")):
        with pytest.raises(ValueError, match="diameter_mm"):
            V.spot_px_for_mm(fast, bad)
    with pytest.raises(ValueError, match="mm_per_dmd_px"):
        V.spot_px_for_mm(fast, 6.5, mm_per_dmd_px=0)
    with pytest.raises(ValueError, match="half the"):
        V.spot_px_for_mm(fast, 500)
    fast.rig.standoff_mm = None
    with pytest.raises(ValueError, match="standoff_mm"):
        V.spot_px_for_mm(fast, 6.5)
    assert V.spot_px_for_mm(fast, 6.5, mm_per_dmd_px=0.5).source == "measured"


def test_spot_image_is_round_on_the_diamond_array(fast):
    spec = V.spot_px_for_mm(fast, 6.5, mm_per_dmd_px=0.25)       # 26 columns, 52 rows
    img = V.spot_image((456, 570), spec)
    assert img.shape == (1140, 912) and img.dtype == np.uint8 and set(np.unique(img)) == {0, 255}
    rows, cols = np.nonzero(img)
    assert cols.min() == 456 - 13 and cols.max() == 456 + 13
    assert rows.min() == 570 - 26 and rows.max() == 570 + 26
    assert (cols.mean(), rows.mean()) == pytest.approx((456, 570))
    sq = V.spot_image((456, 570), spec, shape="square", level=100)
    assert sq.max() == 100 and (sq > 0).sum() > (img > 0).sum()
    tiny = V.spot_image((10, 10), V.spot_px_for_mm(fast, 0.01, mm_per_dmd_px=0.25))
    assert (tiny > 0).sum() >= 1                                  # never an empty pattern
    with pytest.raises(ValueError, match="shape"):
        V.spot_image((10, 10), spec, shape="star")
    with pytest.raises(ValueError, match="level"):
        V.spot_image((10, 10), spec, level=300)


def test_where_specs(fast):
    s = api._engine()
    assert V._resolve_at(s, fast, "H8", False)["target"] == (719.5, 719.5)    # cell centre mirror
    assert V._resolve_at(s, fast, "C4@64", False)["target"] == (159.5, 223.5)
    assert V._resolve_at(s, fast, {"cell": "c4"}, False)["at"] == {"cell": "C4", "spacing_px": 96}
    assert V._resolve_at(s, fast, {"dmd": [10, 20.123]}, False)["target"] == (10.0, 20.12)
    z = V._resolve_at(s, fast, "G7:H9", True)
    assert z["mode"] == "zone" and len(z["cells"]) == 6 and z["cam_mask"] is None
    assert V._resolve_at(s, fast, {"zone": "H8"}, True)["mode"] == "zone"
    with pytest.raises(ValueError, match="is a zone"):
        V._resolve_at(s, fast, "G7:H9", False)
    with pytest.raises(ValueError, match="capture_map"):
        V._resolve_at(s, fast, {"camera": [700, 500]}, False)
    with pytest.raises(ValueError, match="outside the DMD"):
        V._resolve_at(s, fast, {"dmd": [1000, 20]}, False)
    with pytest.raises(ValueError, match="column K"):
        V._resolve_at(s, fast, "K1", False)
    with pytest.raises(ValueError, match="one key"):
        V._resolve_at(s, fast, {"dmd": [1, 2], "camera": [3, 4]}, False)


# ------------------------------------------------------------ project_spot

def test_project_spot_at_a_grid_cell(fast):
    r = V.project_spot(api._engine(), fast, "H8", 6.5, label="h8")
    assert r.found and r.target_dmd == [719.5, 719.5] and r.polarity_inverted is False
    assert np.hypot(*(np.array(r.centroid_px) - cam_of_dmd(719.5, 719.5))) < 1.5
    assert r.spec.source == "nominal" and r.snr > 20 and r.peak_counts > 100
    assert r.diameter_mm is None and any("rule 6" in w for w in r.warnings)
    assert r.expected_px is None and any("capture_map" in w for w in r.warnings)
    for p in (r.photo_png, r.overlay_png, r.json_path):
        assert p.is_file() and recorded(p)
    doc = json.loads(r.json_path.read_text(encoding="utf-8"))
    assert doc["kind"] == "spot" and doc["centroid_px"] == r.centroid_px
    assert doc["spec"]["source"] == "nominal" and "units" in doc and "warnings" in doc
    assert api._engine().bench.objects == []                       # nothing placed, nothing moved


def test_measured_spot_diameter_scales_with_diameter_mm(fast):
    s = api._engine()
    d = {mm: V.project_spot(s, fast, SPOT_AT, mm, label=f"d{mm}").diameter_px for mm in (4, 8, 16)}
    assert d[4] < d[8] < d[16]
    assert d[16] / d[8] == pytest.approx(2.0, rel=0.12)
    assert d[8] / d[4] == pytest.approx(2.0, rel=0.2)
    # a round DMD spot of D mm: D / 0.6446 columns x 1.263 cam px, 2x rows x 0.758 cam px
    cols = 8 / NOMINAL_MM_PER_DMD_PX
    assert d[8] == pytest.approx(np.sqrt(cols * CAM_PER_COL * 2 * cols * CAM_PER_ROW), rel=0.15)


def test_project_spot_at_a_camera_point_via_the_projector_map(fast):
    s = api._engine()
    m = projmap.capture(s, fast, "map")
    r = V.project_spot(s, fast, {"camera": [700, 500]}, 6.5, label="cam")
    assert r.found and r.map_id == m.id and r.expected_px == [700.0, 500.0]
    assert np.hypot(*np.subtract(r.centroid_px, [700, 500])) < 1.5
    assert r.at["camera"] == [700.0, 500.0] and r.at["map_id"] == m.id
    cell = V.project_spot(s, fast, "C4", 3.0, label="c4")          # map predicts where it lands
    assert cell.expected_px is not None and np.hypot(*cell.offset_px) < 1.5
    zone = V.teach_presence(s, fast, "tray", "E6:F7", empty_confirmed=True)   # map's camera cut
    cut = m.camera_mask_for_projector_rect(4 * 96, 5 * 96, 6 * 96, 7 * 96)
    assert zone.n_px == int(cut.sum()) and not any("capture_map" in w for w in zone.warnings)


def test_project_spot_measures_an_inverting_hdmi_path(fast, monkeypatch):
    real_init = MockWindow.__init__

    def inverting(self, hdmi_inverts=False, **kw):
        real_init(self, hdmi_inverts=True, **kw)
    monkeypatch.setattr(MockWindow, "__init__", inverting)
    r = V.project_spot(api._engine(), fast, SPOT_AT, 6.5, label="inv")
    assert r.polarity_inverted is True and any("INVERTED" in w for w in r.warnings)
    assert r.found and np.hypot(*np.subtract(r.centroid_px, cam_of_dmd(456, 570))) < 1.5


# ------------------------------------------------------------ presence

def test_teach_refuses_without_the_human_confirming_empty(fast):
    with pytest.raises(ValueError, match="empty_confirmed=True"):
        V.teach_presence(api._engine(), fast, "pe", SPOT_AT, 6.5)
    assert V.list_sensors(api._engine()) == []


def test_photoelectric_presence(fast):
    s, bench = api._engine(), api.mock_bench()
    t = V.teach_presence(s, fast, "pe", SPOT_AT, 6.5, empty_confirmed=True)
    assert t.method == "photoelectric" and t.mode == "spot" and t.reference_signal > 150
    assert t.version_path.is_file() and t.version_path.parent == t.path
    assert t.path == s.store.root / "sensors" / fast.geometry_id / "pe"
    doc = json.loads(t.version_path.read_text(encoding="utf-8"))
    assert doc["empty_confirmed"] is True and doc["kind"] == "presence" and "units" in doc
    assert t.photo_png.is_file() and recorded(t.photo_png)

    bench.place(Box(h_mm=10, w_mm=40, l_mm=40, albedo=0.2))       # darker than the 0.6 base
    r = V.check_presence(s, fast, "pe")
    assert r.verdict == "PRESENT" and r.present is True
    assert r.relative_change == pytest.approx(0.2 / 0.6 - 1, abs=0.06)
    assert r.json_path.is_file() and recorded(r.json_path) and r.photo_png.is_file()
    assert r.units["signal"].startswith("camera counts") and r.teach_version == t.version_path

    bench.clear()
    r = V.check_presence(s, fast, "pe", label="after")
    assert r.verdict == "ABSENT" and r.present is False and abs(r.relative_change) < 0.05
    assert json.loads(r.json_path.read_text(encoding="utf-8"))["verdict"] == "ABSENT"

    # a white part saturates the receiver: the clipped change stays inside the threshold
    bench.place(Box(h_mm=10, w_mm=40, l_mm=40, albedo=0.95))
    r = V.check_presence(s, fast, "pe")
    assert r.verdict == "INDETERMINATE" and r.present is None and r.saturated_fraction > 0.5
    assert any("saturated" in w for w in r.warnings)
    bench.clear()

    bench.gain = 0.0                                               # projector dark / not aimed
    r = V.check_presence(s, fast, "pe")
    assert r.verdict == "INDETERMINATE" and any("projector" in w for w in r.warnings)


def test_bgs_sees_a_part_photoelectric_cannot(fast):
    """Same albedo as the base: no intensity change, but the spot moves."""
    s, bench = api._engine(), api.mock_bench()
    V.teach_presence(s, fast, "pe", SPOT_AT, 6.5, empty_confirmed=True)
    t = V.teach_presence(s, fast, "bgs", SPOT_AT, 6.5, method="bgs", empty_confirmed=True)
    assert t.depth_axis == "col" and t.baseline_axis == "x"
    np.testing.assert_allclose(t.jacobian, [[CAM_PER_COL, 0], [0, CAM_PER_ROW]], atol=0.02)

    h = 25.0
    bench.place(Box(h_mm=h, w_mm=40, l_mm=40, albedo=bench.base_albedo))
    pe, bgs = V.check_presence(s, fast, "pe"), V.check_presence(s, fast, "bgs")
    assert pe.verdict == "ABSENT"                                  # photoelectric is blind to it
    assert bgs.verdict == "PRESENT" and bgs.present is True
    assert bgs.shift_px == pytest.approx(-RELIEF_PX_PER_MM * h * CAM_PER_COL, abs=0.5)
    assert bgs.relief_px == pytest.approx(RELIEF_PX_PER_MM * h, abs=0.3)
    assert bgs.height_source == "nominal" and any("NOMINAL" in w for w in bgs.warnings)
    assert bgs.mm_per_relief_px == pytest.approx(fast.rig.nominal_mm_per_relief_px)
    assert abs(bgs.height_mm) == pytest.approx(h, rel=0.3)
    assert abs(bgs.off_axis_px) < 0.5 and bgs.units["height_mm"].startswith("mm")

    bench.clear()                                                  # a dark part: same verdict
    bench.place(Box(h_mm=h, w_mm=40, l_mm=40, albedo=0.1))
    dark = V.check_presence(s, fast, "bgs")
    assert dark.verdict == "PRESENT" and dark.relief_px == pytest.approx(bgs.relief_px, abs=0.3)

    bench.clear()                                                  # a black hole: can't triangulate
    bench.place(Box(h_mm=h, w_mm=40, l_mm=40, albedo=0.0))
    black = V.check_presence(s, fast, "bgs")
    assert black.verdict == "INDETERMINATE" and black.present is None
    assert any("no spot" in w for w in black.warnings)

    bench.clear()
    gone = V.check_presence(s, fast, "bgs")
    assert gone.verdict == "ABSENT" and abs(gone.height_mm) < 1.0
    thin = V.check_presence(s, fast, "bgs", min_height_mm=100.0)   # override the taught value
    assert thin.min_height_mm == 100.0


def test_zone_presence_and_sensor_bookkeeping(fast):
    s, bench = api._engine(), api.mock_bench()
    t = V.teach_presence(s, fast, "tray", "E6:F7", empty_confirmed=True)
    assert t.mode == "zone" and t.cells and t.spot is None and t.n_px > 10000
    assert any("capture_map" in w for w in t.warnings)
    with pytest.raises(ValueError, match="photoelectric"):
        V.teach_presence(s, fast, "tray_bgs", "E6:F7", method="bgs", empty_confirmed=True)
    bench.place(Box(h_mm=10, w_mm=100, l_mm=70, albedo=0.2))       # covers the whole zone
    assert V.check_presence(s, fast, "tray").verdict == "PRESENT"
    bench.clear()
    assert V.check_presence(s, fast, "tray").verdict == "ABSENT"

    # re-teach = a new version (append-only); latest wins
    t2 = V.teach_presence(s, fast, "tray", "E6:F7", threshold=0.4, empty_confirmed=True)
    assert t2.version_path != t.version_path and t.version_path.is_file()
    V.teach_presence(s, fast, "spot1", SPOT_AT, 3.0, empty_confirmed=True)
    rows = {r["name"]: r for r in V.list_sensors(s)}
    assert set(rows) == {"tray", "spot1"}
    assert rows["tray"]["versions"] == 2 and rows["tray"]["threshold"] == 0.4
    assert rows["spot1"]["mode"] == "spot" and V.list_sensors(s, "elsewhere") == []
    assert isinstance(V.load_sensor(s, fast, "tray"), V.PresenceTeach)

    with pytest.raises(ValueError, match="no sensor named"):
        V.check_presence(s, fast, "nope")
    fast.geometry_id = "some_other_rig"                            # the rig was moved
    with pytest.raises(rules.GeometryMismatch):
        V.check_presence(s, fast, "tray")


def test_outputs_go_to_the_sensors_kind_or_previews(eng, monkeypatch):
    store = eng.store
    monkeypatch.setattr(store_mod, "OUTPUT_KINDS", tuple(k for k in store_mod.OUTPUT_KINDS
                                                         if k != "sensors"))
    p, kind = V._output_path(store, "x", ".json")
    assert kind == "previews" and p.parent.name == "previews"
    monkeypatch.setattr(store_mod, "OUTPUT_KINDS", store_mod.OUTPUT_KINDS + ("sensors",))
    p, kind = V._output_path(store, "x", ".json")
    assert kind == "sensors" and p.parent.name == "sensors"
    with pytest.raises(ValueError, match="label"):
        V._output_path(store, "///", ".json")


# ------------------------------------------------------------ alignment

def test_alignment(fast):
    s, bench = api._engine(), api.mock_bench()
    base = api.capture_base(fast, "empty")
    # a bgs sensor on the empty fixture, fired later against a depth calibration
    V.teach_presence(s, fast, "bgs", SPOT_AT, 6.5, method="bgs", empty_confirmed=True)

    part = dict(h_mm=20.0, w_mm=60.0, l_mm=25.0, albedo=0.5)
    bench.place(Box(**part))
    ref = api.capture_object(fast, base, "ref", known_dims={"H": 20.0, "W": 60.0, "L": 25.0})
    a = V.teach_alignment(s, fast, "fixture", ref)
    assert a.version_path.is_file() and a.area_px > 5000 and a.axis_ratio > 2
    assert a.angle_deg == pytest.approx(0.0, abs=0.3)
    assert np.allclose(a.centroid_px, [720, 540], atol=1.0)

    # no lateral calibration yet: px only, position can't be judged in mm (rule 6)
    r = V.check_alignment(s, fast, "fixture", ref)
    assert r.position_units == "px" and r.dx_mm is None and r.verdict == "INDETERMINATE"
    assert any("rule 6" in w for w in r.warnings)
    assert V.check_alignment(s, fast, "fixture", ref, tol_px=3).verdict == "ALIGNED"

    cal = api.calibrate(fast, [ref])
    assert cal.has_xy and cal.has_z
    bgs = V.check_presence(s, fast, "bgs")                         # the box sits under the spot
    assert bgs.verdict == "PRESENT" and bgs.height_source == "calibrated"
    assert bgs.height_mm == pytest.approx(20.0, rel=0.05) and bgs.calibration_id == cal.id

    def scan(label, **kw):
        bench.clear()
        bench.place(Box(**{**part, **kw}))
        return api.capture_object(fast, base, label)

    same = V.check_alignment(s, fast, "fixture", scan("same"))
    assert same.verdict == "ALIGNED" and same.position_units == "mm"
    assert same.offset_mm < 0.3 and abs(same.dtheta_deg) < 0.3

    moved = V.check_alignment(s, fast, "fixture", scan("moved", center_mm=(5.0, 0.0)))
    assert moved.verdict == "NOT_ALIGNED" and moved.position_ok is False and moved.angle_ok
    assert moved.dx_mm == pytest.approx(5.0, abs=0.3) and abs(moved.dy_mm) < 0.3
    assert moved.dx_px == pytest.approx(5.0 * BENCH.cam_px_per_mm, abs=1.0)

    turned = V.check_alignment(s, fast, "fixture", scan("turned", angle_deg=5.0))
    assert turned.verdict == "NOT_ALIGNED" and turned.angle_ok is False and turned.position_ok
    assert turned.dtheta_deg == pytest.approx(5.0, abs=0.3)

    for res in (same, moved, turned):
        assert res.json_path.is_file() and recorded(res.json_path)
        assert res.overlay_png.is_file() and res.units["dx_mm"].startswith("mm")
    doc = json.loads(moved.json_path.read_text(encoding="utf-8"))
    assert doc["verdict"] == "NOT_ALIGNED" and doc["kind"] == "alignment_check"

    rows = {r["name"]: r for r in V.list_sensors(s, fast.geometry_id)}
    assert rows["fixture"]["kind"] == "alignment" and rows["bgs"]["method"] == "bgs"
    with pytest.raises(ValueError, match="alignment sensor"):
        V.teach_presence(s, fast, "fixture", SPOT_AT, 6.5, empty_confirmed=True)
    with pytest.raises(ValueError, match="presence sensor"):
        V.check_alignment(s, fast, "bgs", ref)


def test_alignment_on_a_zone_scan(fast):
    """A zone scan's relief is datum-free (base plane at 0). With the part
    covering most of the zone, segmenting around the median (the old
    object_blob call) took the part top for the base and returned the base
    ring as the 'part' (~86 px off). scan_blob uses the base plane."""
    s, bench = api._engine(), api.mock_bench()
    base = api.capture_base(fast, "empty")
    api.capture_map(fast)                                  # the zone's camera-side cut
    k = BENCH.cam_px_per_mm
    part = dict(h_mm=15.0, w_mm=130.0, l_mm=80.0, albedo=0.5)
    bench.place(Box(**part, center_mm=(0.0, -22.0)))
    ref = api.capture_object(fast, base, "zref", zone="D4:G7")
    assert ref.meta["zone"] and np.isfinite(ref.relief).sum() > 1.3 * 130 * 80 * k * k
    a = V.teach_alignment(s, fast, "zfix", ref)
    assert np.allclose(a.centroid_px, [720.0, 540.0 - 22.0 * k], atol=1.5)
    assert a.area_px == pytest.approx(130 * 80 * k * k, rel=0.03)
    assert a.angle_deg == pytest.approx(0.0, abs=0.3) and not a.warnings

    bench.clear()
    bench.place(Box(**part, center_mm=(4.0, -22.0)))
    moved = api.capture_object(fast, base, "zmoved", zone="D4:G7")
    r = V.check_alignment(s, fast, "zfix", moved, tol_px=3)
    assert r.dx_px == pytest.approx(4.0 * k, abs=1.5) and abs(r.dy_px) < 1.5
    assert r.verdict == "NOT_ALIGNED" and r.position_ok is False and r.angle_ok

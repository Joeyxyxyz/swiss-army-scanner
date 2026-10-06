"""Projector map, labelled grid and zone scans ("scan zone C4"), end to end in mock mode."""

import json

import numpy as np
import pytest

from engine import api, rules
from engine.capture import zones as Z
from engine.capture.mock_bench import Box
from engine.capture.projector import MockWindow
from engine.decode import projmap

from conftest import PRESET


@pytest.fixture
def fast(eng):
    p = api.load_preset(PRESET)
    p.camera.average = 1          # speed; the map decodes fine without averaging
    return p


def truth():
    """Mock-bench ground truth: (valid, corr_x, corr_y) per camera px."""
    _, _, valid, cx, cy = api.mock_bench()._scene()
    return valid, cx, cy


def to_mm(bench, uv):
    """Camera px -> base-plane mm (the bench's centred orthographic camera)."""
    H, W = bench.cam_shape
    return ((uv[0] - W / 2) / bench.cam_px_per_mm, (uv[1] - H / 2) / bench.cam_px_per_mm)


def outputs(kind="previews"):
    return api.list_library("outputs", kind)


def read_png(path):
    import cv2
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_UNCHANGED)


# ------------------------------------------------------------ grid & specs

def test_grid_spec_labels():
    g = Z.grid_spec(96)
    assert (g.n_cols, g.n_rows, g.spacing_px) == (10, 12, 96)      # ceil(912/96), ceil(1140/96)
    assert g.col_labels == list("ABCDEFGHIJ")
    assert g.row_labels == [str(i) for i in range(1, 13)]
    fine = Z.grid_spec(16)
    assert fine.n_cols == 57 and fine.col_labels[25:28] == ["Z", "AA", "AB"]
    assert fine.col_labels[-1] == "BE" and Z.col_index("BE") == 56
    for bad in (8, 2000, 12.5, "96", True):
        with pytest.raises(ValueError, match="spacing_px"):
            Z.grid_spec(bad)


def test_grid_image_is_bright_lines_on_dim_with_labels():
    g = Z.grid_spec(96)
    img = Z.grid_image(g)
    assert img.shape == (1140, 912) and img.dtype == np.uint8
    assert img[96 * 3 + 10, 96 * 4 + 10] == Z.FILL_LEVEL          # dim fill, scene stays visible
    assert (img[:, 96 * 4] == 255).all() and (img[96 * 3, :] == 255).all()   # cell lines
    assert (img[:Z.BORDER_PX] == 255).all() and (img[:, -Z.BORDER_PX:] == 255).all()
    x0, y0, x1, y1 = Z.cell_rect((2, 3), g)
    label = Z.grid_image(g)[y0 + 20:y1 - 20, x0 + 20:x1 - 20]
    assert (label > 200).mean() > 0.1                              # "C4" is drawn in its cell
    plain = Z.grid_image(g, labels=False)[y0 + 20:y1 - 20, x0 + 20:x1 - 20]
    assert (plain == Z.FILL_LEVEL).all()


@pytest.mark.parametrize("text, cells", [
    ("C4", [(2, 3)]),
    ("c4", [(2, 3)]),
    (" C4 ", [(2, 3)]),
    ("C4:D6", [(2, 3), (2, 4), (2, 5), (3, 3), (3, 4), (3, 5)]),
    ("D6:C4", [(2, 3), (2, 4), (2, 5), (3, 3), (3, 4), (3, 5)]),     # any corner order
    ("D4 : C6", [(2, 3), (2, 4), (2, 5), (3, 3), (3, 4), (3, 5)]),
    ("B2,E5", [(1, 1), (4, 4)]),
    ("B2, e5; B2", [(1, 1), (4, 4)]),                                # union, de-duplicated
    ("A1:B2,J12", [(0, 0), (0, 1), (1, 0), (1, 1), (9, 11)]),        # blocks and cells mixed
    ("C4,,D5,", [(2, 3), (3, 4)]),                                   # stray separators
])
def test_parse_zone_forms(text, cells):
    assert Z.parse_zone(text, Z.grid_spec(96)) == cells


def test_parse_zone_all():
    g = Z.grid_spec(96)
    assert len(Z.parse_zone("all", g)) == 120 == len(Z.parse_zone("ALL", g))
    assert Z.projector_mask(Z.parse_zone("all", g), g).all()


@pytest.mark.parametrize("text, msg", [
    ("", "empty zone"),
    ("   ", "empty zone"),
    ("K1", r"column K is outside A-J"),
    ("C0", r"row 0 is outside 1-12"),
    ("C13", r"row 13 is outside 1-12"),
    ("4C", r"not a cell name"),
    ("C4:", r"not a block"),
    ("C4:D5:E6", r"not a block"),
    (" , ;", "empty zone"),
    ("zone seven", r"not a cell name"),
])
def test_parse_zone_errors_name_the_valid_range(text, msg):
    with pytest.raises(ValueError, match=msg) as e:
        Z.parse_zone(text, Z.grid_spec(96))
    if text.strip():
        assert "A1..J12" in str(e.value) and "rows 1-12" in str(e.value)


def test_projector_mask_cells():
    g = Z.grid_spec(96)
    m = Z.projector_mask([(2, 3), "J12"], g)
    assert m.shape == (1140, 912) and m.dtype == bool
    assert m[3 * 96:4 * 96, 2 * 96:3 * 96].all()
    assert m[11 * 96:, 9 * 96:].all()                              # edge cell clipped to the DMD
    assert m.sum() == 96 * 96 + (1140 - 11 * 96) * (912 - 9 * 96)
    with pytest.raises(ValueError, match="outside the grid"):
        Z.projector_mask([(10, 0)], g)


# ------------------------------------------------ ProjectorMap on a known map

def synthetic_map(shape=(60, 80)):
    """cam (u, v) -> DMD col = 2u + 5, row = 3v + 7; a hole is unmapped."""
    v, u = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float32)
    mask = np.ones(shape, bool)
    mask[20:30, 40:50] = False
    return projmap.ProjectorMap(id="syn", geometry_id="g", path=None, created="",
                                corr_x=np.where(mask, 2 * u + 5, np.nan),
                                corr_y=np.where(mask, 3 * v + 7, np.nan), mask=mask,
                                dmd_shape=(200, 180))


def test_projector_map_methods_on_a_known_map():
    m = synthetic_map()
    # rect: cols [25, 45) -> u 10..19, rows [37, 67) -> v 10..19
    cm = m.camera_mask_for_projector_rect(25, 37, 45, 67)
    v, u = np.nonzero(cm)
    assert (u.min(), u.max(), v.min(), v.max()) == (10, 19, 10, 19) and cm.sum() == 100
    assert (m.camera_mask_for_projector_rect(45, 67, 25, 37) == cm).all()   # corner order
    pm = np.zeros((200, 180), bool)
    pm[37:67, 25:45] = True
    assert (m.camera_mask_for_projector_mask(pm) == cm).all()
    back = m.projector_mask_for_camera_mask(cm, dilate_px=2)
    assert back[37:67, 25:45].sum() >= 0.9 * pm.sum()             # holes filled by dilation
    assert not back[:30].any() and not back[75:].any()
    assert not m.projector_mask_for_camera_mask(~m.mask).any()     # unmapped px light nothing

    p = m.to_projector([[10.5, 12.25], [45, 25], [-3, 0], [np.nan, 1]])
    assert np.allclose(p[0], [26.0, 43.75])
    assert np.isnan(p[1:]).all()                                   # hole, outside, NaN
    assert np.allclose(m.to_projector([10, 10]), [25, 37])         # a single point stays 1-D
    c = m.to_camera([[25, 37], [26, 40], [95, 70], [500, 10]])
    assert np.allclose(c[0], [10, 10], atol=0.35) and np.allclose(c[1], [10.5, 11], atol=0.35)
    assert np.isnan(c[2:]).all()                                   # unseen (hole) / off the DMD
    with pytest.raises(ValueError, match="camera mask shape"):
        m.projector_mask_for_camera_mask(np.ones((5, 5), bool))


# ------------------------------------------------------------ capture_map

def test_capture_map_matches_truth_and_round_trips(fast):
    m = api.capture_map(fast, "empty")
    valid, cx, cy = truth()
    assert m.geometry_id == fast.geometry_id and m.dmd_shape == (1140, 912)
    assert (m.mask & valid).sum() >= 0.99 * valid.sum()            # most of the lit area
    assert not (m.mask & ~valid).any()
    assert abs(m.valid_pct - 100 * valid.mean()) < 1.0 and not m.warnings
    ex, ey = np.abs(m.corr_x - cx)[m.mask], np.abs(m.corr_y - cy)[m.mask]
    assert np.percentile(ex, 99) < 0.5 and np.percentile(ey, 99) < 1.0

    rng = np.random.default_rng(0)
    ys, xs = np.nonzero(m.mask)
    i = rng.choice(len(xs), 3000, replace=False)
    cam = np.stack([xs[i], ys[i]], 1).astype(float)
    rt = m.to_camera(m.to_projector(cam))
    assert np.hypot(*(rt - cam).T).max() < 2.0                     # camera -> DMD -> camera
    dmd = rng.uniform([2, 2], [909, 1137], (3000, 2))
    rt = m.to_projector(m.to_camera(dmd))
    assert np.hypot(*(rt - dmd).T).max() < 2.0                     # DMD -> camera -> DMD
    x0, y0, x1, y1 = Z.cell_rect((2, 3), Z.grid_spec(96))
    cell = m.camera_mask_for_projector_rect(x0, y0, x1, y1)
    pm = m.projector_mask_for_camera_mask(cell)
    rect = np.zeros((1140, 912), bool)
    rect[y0:y1, x0:x1] = True
    assert (pm & rect).sum() >= 0.99 * rect.sum()                  # camera region -> its cell,
    grown = np.zeros_like(rect)                                    # + the 2 px dilation margin
    grown[y0 - 3:y1 + 3, x0 - 3:x1 + 3] = True
    assert not (pm & ~grown).any()

    # stored in the library, listed, reloadable, and the preview is in the output bin
    for f in ("map.npz", "meta.json", "preview.png"):
        assert (m.path / f).is_file()
    assert m.path.parent.name == fast.geometry_id and m.path.parent.parent.name == "maps"
    meta = json.loads((m.path / "meta.json").read_text())
    assert meta["units"]["corr_x"].startswith("projector px") and meta["valid_pct"] == m.valid_pct
    listed = api.list_library("maps")
    assert [r["id"] for r in listed] == [m.id] and listed[0]["path"] == m.path
    again = projmap.latest(api._engine(), fast.geometry_id)
    assert again.id == m.id and np.array_equal(again.mask, m.mask)
    assert np.allclose(again.corr_x[again.mask], m.corr_x[m.mask])
    assert projmap.load(m.path / "map.npz").id == m.id
    assert any(r["file"].endswith("_empty_map.png") for r in outputs())
    assert projmap.latest(api._engine(), "some-other-geometry") is None

    api.delete(m, "test")                                          # 'delete' = move to trash
    assert projmap.latest(api._engine(), fast.geometry_id) is None
    assert api.list_library("maps") == []


# ------------------------------------------------------------ project_grid

def test_project_grid_with_map_gives_an_agent_readable_overlay(fast, monkeypatch):
    holds = []
    real_hold = MockWindow.hold
    monkeypatch.setattr(MockWindow, "hold",
                        lambda self, s: (holds.append(s), real_hold(self, s))[1])
    m = api.capture_map(fast)
    n0 = len(holds)
    shot = api.project_grid(fast, 96, "grid", hold_s=1.5)
    assert shot.kind == "grid" and shot.grid.n_cols == 10 and len(shot.zones) == 120
    # pumped hold, not a sleep - in 0.25 s slices so a UI can cancel it
    assert holds[-6:] == [Z.HOLD_SLICE_S] * 6 and sum(holds[-6:]) == pytest.approx(1.5)
    assert len(holds) - n0 > 6 and shot.held_s == 1.5 and not shot.hold_cancelled
    for p in (shot.camera_png, shot.overlay_png, shot.dmd_png, shot.json_path):
        assert p is not None and p.is_file()
    assert shot.map_id == m.id and shot.polarity_inverted is False and not shot.warnings
    assert len(shot.cells_visible) == 120                          # the camera sees the whole DMD
    assert "C4" in shot.cells_visible and "A1" in shot.cells_visible

    # the label for C4 sits in C4's camera region
    x0, y0, x1, y1 = Z.cell_rect((2, 3), shot.grid)
    u, v = shot.cell_centers["C4"]
    assert m.camera_mask_for_projector_rect(x0, y0, x1, y1)[int(round(v)), int(round(u))]

    # what went to the DMD is the grid image (bench shows raw as sent)
    assert np.array_equal(read_png(shot.dmd_png), Z.grid_image(shot.grid))
    # the photo shows bright lines on a dim fill where the map says they land
    photo = read_png(shot.camera_png).astype(float)
    on = m.to_camera([[4 * 96, 3 * 96 + 12]])[0]
    off = m.to_camera([[4 * 96 + 12, 3 * 96 + 12]])[0]
    assert photo[int(round(on[1])), int(round(on[0]))] > photo[int(round(off[1])),
                                                               int(round(off[0]))] + 100
    overlay = read_png(shot.overlay_png)
    assert overlay.shape == (1080, 1440, 3)
    assert (np.all(overlay == Z.LABEL_BGR, axis=2)).sum() > 1000   # upright labels drawn
    assert (np.all(overlay == Z.LINE_BGR, axis=2)).sum() > 10000   # cell boundaries drawn

    doc = json.loads(shot.json_path.read_text())
    assert doc["spacing_px"] == 96 and doc["cells_visible"] == shot.cells_visible
    assert doc["map_agreement"] > 0.8 and doc["units"]["cell_centers"] == "camera px [x, y]"
    files = {r["file"].rsplit("/", 1)[-1] for r in outputs()}
    for p in (shot.camera_png, shot.overlay_png, shot.dmd_png, shot.json_path):
        assert p.name in files


def test_project_grid_without_map_still_photographs_the_labels(fast):
    shot = api.project_grid(fast)
    assert shot.camera_png.is_file() and shot.dmd_png.is_file()
    assert shot.overlay_png is None and shot.cells_visible == [] and shot.map_id is None
    assert any("capture_map" in w for w in shot.warnings)


def test_project_grid_measures_an_inverting_hdmi_path(fast, monkeypatch):
    """If the raw path lands inverted, the grid is sent inverted so it is
    still PHYSICALLY bright lines on a dim fill."""
    real_init = MockWindow.__init__

    def inverting(self, hdmi_inverts=False, **kw):
        real_init(self, hdmi_inverts=True, **kw)
    monkeypatch.setattr(MockWindow, "__init__", inverting)
    shot = api.project_grid(fast, label="inv")
    assert shot.polarity_inverted is True
    assert any("INVERTED" in w for w in shot.warnings)
    assert np.array_equal(read_png(shot.dmd_png), 255 - Z.grid_image(shot.grid))
    photo = read_png(shot.camera_png).astype(float)
    # mock bench, empty scene: camera u = 144 + col * 1152/912, v = 108 + row * 864/1140
    u = lambda col: int(round(144 + col * 1152 / 912))            # noqa: E731
    v = lambda row: int(round(108 + row * 864 / 1140))            # noqa: E731
    assert photo[v(300), u(4 * 96)] > photo[v(300), u(4 * 96 + 12)] + 100


# ------------------------------------------------------------ zone scans

def test_zone_scan_lights_and_cuts_to_the_cell(fast):
    base = api.capture_base(fast, "empty")
    empty = api.capture_map(fast, "empty")
    bench = api.mock_bench()
    g = Z.grid_spec(96)
    x0, y0, x1, y1 = Z.cell_rect((3, 3), g)                        # D4
    centre = empty.to_camera([(x0 + x1) / 2, (y0 + y1) / 2])
    bench.place(Box(20.0, 15.0, 10.0, center_mm=to_mm(bench, centre)))   # human step (mock)

    m = api.capture_map(fast, "with part")                        # map of the scene as it is
    shot = api.project_grid(fast)
    assert "D4" in shot.cells_visible
    u, v = centre       # label drawn where D4's centre mirror lands: on the box top, which
    # parallax shifts ~9 DMD cols (~11 camera px) from the empty-scene position
    assert shot.cell_centers["D4"] == pytest.approx([u, v], abs=20)

    scan = api.capture_object(fast, base, "zoned", zone="D4")
    # the map cut the relief (no "no projector map" warning); capture_object
    # may still note the fully-lit base / decode saturation (api.py's own warnings)
    assert scan.meta["zone"] == "D4"
    assert not any("projector map" in w for w in scan.meta["warnings"])
    cell = m.camera_mask_for_projector_rect(x0, y0, x1, y1)
    valid = scan.relief_mask
    assert valid.any() and not (valid & ~cell).any()              # nothing outside the cell
    assert valid.sum() >= 0.9 * cell.sum()
    h = bench.height_map()
    on_box = valid & (h > 0)
    assert on_box.sum() > 1000
    assert np.median(scan.relief[on_box]) == pytest.approx(0.45 * 20.0, abs=0.5)
    assert abs(np.median(scan.relief[valid & (h == 0)])) < 0.3
    # only the zone was lit: the decoded field is valid only where D4's mirrors land
    _, tx, ty = truth()
    fm = scan.field.mask
    assert fm.any()
    assert (tx[fm] > x0 - 1.5).all() and (tx[fm] < x1 + 1.5).all()
    assert (ty[fm] > y0 - 1.5).all() and (ty[fm] < y1 + 1.5).all()
    assert any(r["label"] == "zoned" for r in outputs())          # relief preview in the bin

    # union / block specs resolve to the matching camera regions, no scan needed
    pm, cm, cells = Z.resolve(api._engine(), fast, "B2,E5")
    assert cells == [(1, 1), (4, 4)] and pm.sum() == 2 * 96 * 96
    assert (cm == m.camera_mask_for_projector_mask(pm)).all()
    with pytest.raises(ValueError, match="column K"):
        api.capture_object(fast, base, "bad", zone="K4")


def test_zone_scan_without_a_map_warns_and_still_lights_only_the_zone(fast):
    base = api.capture_base(fast, "empty")
    api.mock_bench().place(Box(20.0, 15.0, 10.0, center_mm=(-50.0, -59.0)))
    scan = api.capture_object(fast, base, "zoned", zone="D4")
    assert any("no projector map" in w and "capture_map" in w for w in scan.meta["warnings"])
    _, tx, ty = truth()
    x0, y0, x1, y1 = Z.cell_rect((3, 3), Z.grid_spec(96))
    valid = scan.relief_mask
    assert valid.sum() > 5000
    assert (tx[valid] > x0 - 1.5).all() and (tx[valid] < x1 + 1.5).all()
    assert (ty[valid] > y0 - 1.5).all() and (ty[valid] < y1 + 1.5).all()


def test_zone_names_follow_the_last_projected_grid(fast):
    api.project_grid(fast, spacing_px=64, label="fine")
    s = api._engine()
    pm, cm, cells = Z.resolve(s, fast, "C4")
    assert cm is None and cells == [(2, 3)]                        # no map yet
    assert pm[3 * 64:4 * 64, 2 * 64:3 * 64].all() and pm.sum() == 64 * 64
    pm96, _, _ = Z.resolve(s, fast, "C4@96")                       # explicit spacing wins
    assert pm96.sum() == 96 * 96 and pm96[3 * 96, 2 * 96]
    with pytest.raises(ValueError, match="grid spacing"):
        Z.resolve(s, fast, "C4@big")
    with pytest.raises(ValueError, match="rows 1-18"):
        Z.resolve(s, fast, "C40")                                  # 64 px grid: 18 rows


def test_zone_refuses_a_map_from_another_geometry(fast):
    m = synthetic_map((1080, 1440))
    m.geometry_id = "some_other_rig"
    with pytest.raises(rules.GeometryMismatch):
        Z.resolve(api._engine(), fast, "C4", pmap=m)


# ------------------------------------------------------------ project_zones

def test_project_zones_camera_rect_and_projector_polygon(fast):
    m = api.capture_map(fast)
    rect = [[500, 400], [760, 400], [760, 600], [500, 600]]        # camera px
    tri = [[100, 700], [300, 700], [200, 900]]                     # DMD px (col, row)
    shot = api.project_zones(
        [Z.Zone("part", rect, space="camera", pattern="flood"),
         {"name": "tri", "polygon": tri, "space": "projector", "pattern": "checker",
          "level": 200}], fast, label="marks")
    assert shot.kind == "zones" and shot.zones == ["part", "tri"] and shot.grid is None
    assert not shot.warnings and shot.map_id == m.id
    for p in (shot.camera_png, shot.overlay_png, shot.dmd_png, shot.json_path):
        assert p.is_file()
    files = {r["file"].rsplit("/", 1)[-1] for r in outputs()}
    assert {shot.camera_png.name, shot.overlay_png.name, shot.dmd_png.name} <= files

    dmd = read_png(shot.dmd_png)
    lit = dmd == 255
    # the camera rectangle was mapped through the map: its DMD pixels light that rectangle
    cam_rect = np.zeros((1080, 1440), bool)
    cam_rect[400:601, 500:761] = True
    lit_cam = m.camera_mask_for_projector_mask(lit)
    assert (lit_cam & cam_rect).sum() >= 0.97 * cam_rect.sum()
    grown = np.zeros_like(cam_rect)
    grown[396:605, 496:765] = True
    assert not (lit_cam & ~grown).any()
    # the projector triangle carries a 16 px checker at level 200, drawn straight on the DMD
    assert dmd[750, 196] == 200 and dmd[750, 212] == 0 and dmd[766, 212] == 200
    assert set(np.unique(dmd)) <= {0, 200, 255}
    # and the photo is bright where the rectangle is, dark away from all zones
    photo = read_png(shot.camera_png).astype(float)
    assert photo[500, 630] > photo[900, 1300] + 100
    doc = json.loads(shot.json_path.read_text())
    assert [z["name"] for z in doc["zones"]] == ["part", "tri"] and doc["zones"][1]["dmd_px"] > 0
    assert set(shot.cell_centers) == {"part", "tri"}


def test_zone_patterns():
    m = np.zeros((1140, 912), bool)
    m[100:200, 100:300] = True
    out = Z._pattern("outline", 255, m)
    assert out[100, 150] == 255 and out[102, 150] == 255 and out[103, 150] == 0
    assert out[150, 150] == 0 and out[50, 50] == 0
    st = Z._pattern("stripes", 255, m)
    assert st[150, 100] == 255 and st[150, 108] == 0 and st[150, 116] == 255   # 8 px stripes
    gr = Z._pattern("grid", 99, m)
    assert gr[150, 112] == 99 and gr[112, 120] == 99                # 16 px pitch lines
    assert gr[150, 115] == 0 and gr[115, 120] == 0
    assert (Z._pattern("flood", 7, m) == np.where(m, 7, 0)).all()


def test_camera_space_zones_need_a_map(fast):
    rect = [[500, 400], [760, 600]]                                # 2 corners = rectangle
    with pytest.raises(ValueError, match="capture_map"):
        api.project_zones([Z.Zone("part", rect)], fast)
    shot = api.project_zones([{"name": "p", "polygon": [[10, 10], [200, 200]],
                               "space": "projector", "pattern": "outline"}], fast)
    assert shot.overlay_png is None and any("no projector map" in w for w in shot.warnings)
    assert shot.camera_png.is_file()


@pytest.mark.parametrize("zone, msg", [
    ({"name": "a", "polygon": [[0, 0], [9, 9]], "pattern": "plaid"}, "pattern must be"),
    ({"name": "a", "polygon": [[0, 0], [9, 9]], "space": "world"}, "space must be"),
    ({"name": "a", "polygon": [[0, 0], [9, 9]], "level": 300}, "level must be"),
    ({"name": "a", "polygon": [[0, 0]]}, "polygon must be"),
    ({"name": "a", "polygon": "box"}, "polygon must be"),
    ({"name": "", "polygon": [[0, 0], [9, 9]]}, "name must be"),
    ({"name": "a", "polygon": [[0, 0], [9, 9]], "colour": "red"}, "unknown keys"),
])
def test_bad_zones_are_refused_before_the_rig(fast, zone, msg):
    with pytest.raises(ValueError, match=msg):
        api.project_zones([zone], fast)


# ------------------------------------------- holds: sliced, cancellable, rule 3

def test_long_hold_runs_in_slices_and_can_be_cancelled(fast, monkeypatch):
    import threading
    holds = []
    stop = threading.Event()
    real_hold = MockWindow.hold

    def hold(self, s):
        holds.append(s)
        real_hold(self, s)
        if len(holds) >= 4 and s == Z.HOLD_SLICE_S:
            stop.set()                                  # the UI's Stop button, mid-hold
    monkeypatch.setattr(MockWindow, "hold", hold)
    shot = api.project_grid(fast, 96, "g", hold_s=0.6)
    assert holds[-3:] == pytest.approx([0.25, 0.25, 0.1])   # the last slice is the rest
    assert shot.held_s == pytest.approx(0.6) and not shot.hold_cancelled
    holds.clear()
    stop.clear()
    shot = Z.project_grid(api._engine(), fast, 96, "g2", hold_s=600.0, cancel=stop)
    sliced = [h for h in holds if h == Z.HOLD_SLICE_S]
    assert sliced == [Z.HOLD_SLICE_S]                   # stopped after the first slice, not 600 s
    assert shot.hold_cancelled and shot.held_s == Z.HOLD_SLICE_S
    assert any("hold cancelled" in w for w in shot.warnings)
    doc = json.loads(shot.json_path.read_text())
    assert doc["hold_cancelled"] is True and doc["hold_s"] == 600.0

    stop.clear()
    holds.clear()
    shot = Z.project_zones(api._engine(), fast,
                           [{"name": "z", "polygon": [[100, 100], [300, 300]],
                             "space": "projector"}], hold_s=1.0, cancel=stop)
    assert shot.hold_cancelled                          # stop set again after 4 holds
    with pytest.raises(ValueError, match="threading.Event"):
        Z.project_grid(api._engine(), fast, 96, "g3", hold_s=1.0, cancel=True)


def test_long_hold_with_burst_led_currents_is_refused_before_projecting(fast, monkeypatch):
    shots = []
    real_show = MockWindow.show
    monkeypatch.setattr(MockWindow, "show", lambda self, p: (shots.append(1), real_show(self, p)))
    fast.projection.led = {"mode": "burst", "r": 180, "g": 200, "b": 150}
    with pytest.raises(rules.SafetyViolation, match="grid/zone hold"):
        api.project_grid(fast, 96, "hot", hold_s=rules.LED_BURST_WINDOW_S + 5)
    with pytest.raises(rules.SafetyViolation, match="grid/zone hold"):
        api.project_zones([{"name": "z", "polygon": [[0, 0], [100, 100]],
                            "space": "projector"}], fast, hold_s=60.0)
    assert not shots                                    # nothing was projected
    shot = api.project_grid(fast, 96, "short", hold_s=2.0)   # inside the burst window: fine
    assert shot.held_s == 2.0
    fast.projection.led = {"mode": "continuous", "r": 120, "g": 180, "b": 160}
    assert api.project_grid(fast, 96, "cont", hold_s=30.0).held_s == 30.0

"""Spectral mode (R/G/B LED reflectance -> materials) and direct/global light
separation, end to end in mock mode."""

import json

import numpy as np
import pytest

from engine import api
from engine.capture import projector as projector_mod
from engine.capture.mock_bench import Box, Label, MockBench
from engine.sense import spectral

from conftest import PRESET

RED = (0.85, 0.20, 0.15)       # red-ish paint
BLUE = (0.15, 0.30, 0.85)      # blue-ish paint
RED_C, BLUE_C = (-80.0, 0.0), (80.0, 0.0)
BOX_W, BOX_L = 60.0, 40.0


@pytest.fixture
def fast(eng):
    p = api.load_preset(PRESET)
    p.camera.average = 1          # speed; averaging is not the subject here
    return p


def _flood(bgr):
    img = np.zeros((1140, 912, 3), np.uint8)
    img[...] = bgr
    return img


def _px(bench, center_mm):
    H, W = bench.cam_shape
    return W / 2 + center_mm[0] * bench.cam_px_per_mm, H / 2 + center_mm[1] * bench.cam_px_per_mm


def _roi(bench, center_mm, half_mm=10.0):
    cx, cy = _px(bench, center_mm)
    h = half_mm * bench.cam_px_per_mm
    return (cx - h, cy - h, cx + h, cy + h)


def _footprint(bench, center_mm):
    """Camera px of a box footprint (its top), from the bench ground truth."""
    H, W = bench.cam_shape
    v, u = np.mgrid[0:H, 0:W]
    cx, cy = _px(bench, center_mm)
    k = bench.cam_px_per_mm
    return (np.abs(u - cx) < BOX_W / 2 * k - 1) & (np.abs(v - cy) < BOX_L / 2 * k - 1)


def _two_boxes(bench):
    bench.place(Box(30.0, BOX_W, BOX_L, center_mm=RED_C, rgb=RED))
    bench.place(Box(30.0, BOX_W, BOX_L, center_mm=BLUE_C, rgb=BLUE))


# ------------------------------------------------------------ mock bench colour model

def test_mock_bench_colour_model():
    b = MockBench(seed=0)
    b.place(Box(30.0, BOX_W, BOX_L, center_mm=RED_C, rgb=RED))
    b.place(Box(20.0, 30.0, 30.0, center_mm=(0.0, -100.0), albedo=0.5))     # neutral box
    amb = b.ambient
    red, green, blue = (b.render(_flood(c)) for c in ((0, 0, 255), (0, 255, 0), (255, 0, 0)))
    white3 = b.render(_flood((255, 255, 255)))
    white2 = b.render(np.full((1140, 912), 255, np.uint8))
    # white = the three LED sub-frames; equal-channel BGR == the 2-D gray pattern
    np.testing.assert_allclose(red + green + blue - 3 * amb, white3 - amb, atol=1e-3)
    np.testing.assert_allclose(white3, white2, atol=1e-3)
    w = b.led_weights()
    assert w.sum() == pytest.approx(1.0) and w[2] < w[0]            # blue LED weaker
    cx, cy = (int(v) for v in _px(b, RED_C))
    assert red[cy, cx] - amb == pytest.approx(b.gain * w[0] * RED[0], rel=1e-4)
    assert blue[cy, cx] - amb == pytest.approx(b.gain * w[2] * RED[2], rel=1e-4)
    # 2-D patterns see the LED-weighted gray albedo of a coloured surface
    assert white2[cy, cx] - amb == pytest.approx(b.gain * b.gray_albedo(RED), rel=1e-4)
    # a neutral (rgb=None) surface keeps its albedo in every band
    nx, ny = (int(v) for v in _px(b, (0.0, -100.0)))
    assert red[ny, nx] - amb == pytest.approx(b.gain * w[0] * 0.5, rel=1e-4)
    assert np.allclose(b.reflectance()[ny, nx], 0.5)


def test_mock_label_red_ink_vanishes_under_red_light():
    b = MockBench(seed=0)
    b.place(Label("SAS", center_mm=(0.0, 0.0), size_mm=45, rgb=(0.85, 0.85, 0.85),
                  ink_rgb=(0.80, 0.10, 0.08)))
    refl = b.reflectance()
    ink = np.isclose(refl[..., 1], 0.10)
    paper = np.isclose(refl[..., 1], 0.85)                           # label paper, not the base
    assert ink.any() and paper.any()
    for bgr, lo, hi in (((0, 0, 255), 0.0, 0.1), ((255, 0, 0), 0.8, 1.0)):
        f = b.render(_flood(bgr)) - b.ambient
        contrast = 1 - np.median(f[ink]) / np.median(f[paper])
        assert lo <= contrast <= hi, (bgr, contrast)


# ------------------------------------------------------------ capture / learn / classify

def test_spectral_capture_learn_classify(fast):
    bench = api.mock_bench()
    _two_boxes(bench)
    r = api.spectral_capture(fast, "two_boxes")
    assert isinstance(r, spectral.SpectralReading)
    assert r.bands == ["red", "green", "blue"] and r.mock and not r.polarity_inverted
    assert r.geometry_id == fast.geometry_id and r.valid_pct > 50 and r.units
    for f in ("cube.npz", "meta.json", "preview.png"):
        assert (r.path / f).exists(), f
    assert r.colour_png is not None and r.colour_png.exists()
    meta = json.loads((r.path / "meta.json").read_text())
    assert meta["band_sum_vs_white"] == pytest.approx(1.0, abs=0.02)   # R+G+B == white
    with np.load(r.path / "cube.npz") as z:
        bands, valid = z["bands"], z["valid"]
    assert bands.shape == bench.cam_shape + (3,) and bands.dtype == np.float32
    red_fp, blue_fp = _footprint(bench, RED_C), _footprint(bench, BLUE_C)
    assert valid[red_fp].mean() > 0.99 and valid[blue_fp].mean() > 0.99
    assert np.median(bands[red_fp][:, 0]) > 3 * np.median(bands[red_fp][:, 2])
    assert np.median(bands[blue_fp][:, 2]) > 3 * np.median(bands[blue_fp][:, 0])
    assert not valid[:, :100].any()                                  # outside the lit patch

    m_red = api.learn_material(r, "red_box", _roi(bench, RED_C))
    m_blue = api.learn_material(r, "blue_box", _roi(bench, BLUE_C))
    for m in (m_red, m_blue):
        assert sum(m.signature) == pytest.approx(1.0, abs=1e-4)
        assert min(m.spread) >= spectral.SPREAD_FLOOR and m.n_px > 3000
        assert m.version_path.exists() and m.path.is_dir() and 0 < m.brightness < 2
    assert m_red.signature[0] > 0.6 and m_blue.signature[2] > 0.5

    c = api.classify(r)
    assert isinstance(c, spectral.Classification)
    assert set(c.materials) == {"red_box", "blue_box"}
    with np.load(c.labels_npz) as z:
        labels, names = z["labels"], list(z["names"])
    lab = {n: i for i, n in enumerate(names)}                       # 0 = unknown
    assert (labels[red_fp] == lab["red_box"]).mean() > 0.8
    assert (labels[blue_fp] == lab["blue_box"]).mean() > 0.8
    base = valid & (bench.height_map() == 0)
    base_labels = labels[base]
    assert (base_labels == lab["unknown"]).mean() > 0.95             # gray base: not learned
    assert c.coverage_pct["unknown"] > 50
    assert sum(c.coverage_pct.values()) == pytest.approx(100.0, abs=0.01)

    # blobs locate the boxes (camera px)
    for name, center, fp in (("red_box", RED_C, red_fp), ("blue_box", BLUE_C, blue_fp)):
        top = max((b for b in c.blobs if b["name"] == name), key=lambda b: b["area_px"])
        cx, cy = _px(bench, center)
        assert abs(top["centroid"][0] - cx) < 5 and abs(top["centroid"][1] - cy) < 5
        ys, xs = np.nonzero(fp)
        x0, y0, x1, y1 = top["bbox"]
        assert abs(x0 - xs.min()) < 6 and abs(x1 - xs.max()) < 6
        assert abs(y0 - ys.min()) < 6 and abs(y1 - ys.max()) < 6
        assert top["area_px"] > 0.9 * fp.sum() and top["units"] == "px"

    # output bin: map PNG + JSON + label map, recorded in the index
    assert c.map_png.exists() and c.json_path.exists() and c.labels_npz.exists()
    assert c.map_png.parent.name == "spectral" and c.map_png.parent.parent.name == "output"
    rows = api.list_library("outputs", "spectral")
    files = {row["file"].rsplit("/", 1)[-1] for row in rows}
    assert {c.map_png.name, c.json_path.name, c.labels_npz.name} <= files
    doc = json.loads(c.json_path.read_text())
    assert doc["coverage_pct"] == c.coverage_pct and doc["units"]["bbox"].startswith("camera px")
    exports = (r.path / "exports.jsonl").read_text().splitlines()
    assert len(exports) >= 4                                         # colour preview + 3 outputs

    # library browsing: readings and latest material versions
    assert [x.id for x in api.list_library("spectral")] == [r.id]
    assert {m.name for m in api.list_library("materials")} == {"red_box", "blue_box"}
    m_red2 = api.learn_material(r, "red_box", _roi(bench, RED_C, half_mm=6.0))
    latest = {m.name: m for m in api.list_library("materials")}
    assert latest["red_box"].version_path == m_red2.version_path
    assert len(list(m_red.path.glob("*.json"))) == 2                 # append-only versions
    only_red = api.classify(r, ["red_box"])
    assert only_red.materials == ["red_box"] and only_red.coverage_pct["red_box"] > 0


def test_brightness_separates_black_from_white(fast):
    fast.camera.average = 4                  # the dark box needs a little SNR
    bench = api.mock_bench()
    bench.gain = 250.0                       # keep the white box below saturation
    bench.place(Box(30.0, BOX_W, BOX_L, center_mm=RED_C, albedo=0.85))    # white, neutral
    bench.place(Box(30.0, BOX_W, BOX_L, center_mm=BLUE_C, albedo=0.12))   # black, neutral
    r = api.spectral_capture(fast, "bw")
    wm = api.learn_material(r, "white", _roi(bench, RED_C))
    bm = api.learn_material(r, "black", _roi(bench, BLUE_C))
    assert wm.brightness > 4 * bm.brightness
    # same hue: the signatures alone cannot tell them apart
    assert np.allclose(wm.signature, bm.signature, atol=0.02)
    wfp, bfp = _footprint(bench, RED_C), _footprint(bench, BLUE_C)

    def fraction(c, fp, name):
        with np.load(c.labels_npz) as z:
            labels, names = z["labels"], list(z["names"])
        return (labels[fp] == names.index(name)).mean()

    c = spectral.classify(api._engine(), r)
    assert fraction(c, wfp, "white") > 0.8 and fraction(c, bfp, "black") > 0.8
    # with only 'white' known: hue alone accepts the black box as white,
    # brightness rejects it (0.12 vs 0.85 albedo is ~5.6 brightness sigmas)
    hue_only = spectral.classify(api._engine(), r, ["white"], brightness_weight=0.0)
    assert fraction(hue_only, bfp, "white") > 0.8
    with_b = spectral.classify(api._engine(), r, ["white"])
    assert fraction(with_b, bfp, "white") < 0.05 and fraction(with_b, wfp, "white") > 0.8


def test_learn_and_classify_refuse_bad_input(fast):
    bench = api.mock_bench()
    _two_boxes(bench)
    r = api.spectral_capture(fast, "bad")
    with pytest.raises(ValueError, match="no usable materials"):
        api.classify(r)
    with pytest.raises(ValueError, match="valid pixels"):
        api.learn_material(r, "outside", (0, 0, 60, 60))           # outside the lit patch
    with pytest.raises(ValueError, match="outside"):
        api.learn_material(r, "off_frame", (5000, 5000, 5100, 5100))
    with pytest.raises(ValueError, match="roi"):
        api.learn_material(r, "junk", "C4")
    with pytest.raises(ValueError, match="reserved"):
        api.learn_material(r, "unknown", _roi(bench, RED_C))
    api.learn_material(r, "red_box", _roi(bench, RED_C))
    with pytest.raises(ValueError, match="unknown material"):
        api.classify(r, ["no_such_thing"])
    # every failure was logged, nothing was half-written
    assert not (api.paths()["spectral"] / "materials" / "outside").exists()


def test_inverted_raw_path_is_measured_not_assumed(fast, monkeypatch):
    class InvertingWindow(projector_mod.MockWindow):
        def __init__(self, **kw):
            super().__init__(hdmi_inverts=True, **kw)

    monkeypatch.setattr(projector_mod, "MockWindow", InvertingWindow)
    bench = api.mock_bench()
    _two_boxes(bench)
    r = api.spectral_capture(fast, "inverted")
    assert r.polarity_inverted and any("INVERTED" in w for w in r.warnings)
    m = api.learn_material(r, "red_box", _roi(bench, RED_C))
    assert m.signature[0] > 0.6                                      # still the red LED


# ------------------------------------------------------------ direct / global separation

def test_separate_light_direct_and_global(fast):
    bench = api.mock_bench()
    bench.gain = 200.0                       # keep Ld + Lg below saturation
    bench.place(Box(30.0, BOX_W, BOX_L, center_mm=RED_C))
    s = api.separate_light(fast, "clean")
    assert isinstance(s, spectral.LightSeparation)
    assert 0.0 <= s.global_fraction_median <= 0.06, s
    assert s.square_px == 16 and s.shifts == 8 and len(s.offsets) == 8
    assert s.lit_pct > 40 and s.checker_contrast > 0.9 and not s.warnings
    for p in (s.direct_png, s.global_png):
        assert p.exists() and p.parent.name == "previews"
    rows = {row["file"].rsplit("/", 1)[-1] for row in api.list_library("outputs", "previews")}
    assert {s.direct_png.name, s.global_png.name} <= rows
    assert s.json_path.exists() and (s.path / "separation.npz").exists()

    bench.global_light = 0.25                # uniform scatter: expect 0.25 / 1.25 = 0.2
    fast.camera.average = 2                  # max/min over noisy frames bias global low
    g = api.separate_light(fast, "scatter", shifts=4)
    assert 0.0 <= g.global_fraction_median <= 1.0 and len(g.offsets) == 4
    assert g.global_fraction_median == pytest.approx(0.2, abs=0.04)
    with np.load(g.path / "separation.npz") as z:
        glob, lit = z["global"], z["lit"]
    assert np.median(glob[lit]) == pytest.approx(0.25 * bench.gain * bench.base_albedo, rel=0.1)
    # separation readings are not listed as spectral readings
    assert api.list_library("spectral") == []


def test_separate_light_warns_on_fine_squares(fast):
    bench = api.mock_bench()
    bench.gain = 200.0
    s = api.separate_light(fast, "fine", square_px=4, shifts=8)
    assert any("square_px=4" in w for w in s.warnings)
    assert len(s.offsets) <= 8 and 0.0 <= s.global_fraction_median <= 1.0
    with pytest.raises(ValueError):
        api.separate_light(fast, "bad", square_px=0)


def test_checker_offsets_cover_both_axes():
    offs = spectral.checker_offsets(16, 8)
    assert len(offs) == 8 and len({dx for dx, _ in offs}) > 1 and len({dy for _, dy in offs}) > 1
    assert all(0 <= dx < 16 and 0 <= dy < 16 for dx, dy in offs)
    # every pixel phase gets clear of the square edges in some frame, better than a
    # purely diagonal shift (which fails where a vertical and a horizontal edge meet)
    diagonal = [(2 * k, 2 * k) for k in range(8)]
    assert spectral.checker_margin(offs, 16) > 0.3 > spectral.checker_margin(diagonal, 16)
    p = spectral.checkerboard((1140, 912), 16, *offs[3])
    assert p.dtype == np.uint8 and set(np.unique(p)) == {0, 255}
    assert abs(p.mean() / 255 - 0.5) < 0.02


# ------------------------------------------- materials: mock never shadows real

def test_material_versions_are_picked_by_mode(fast):
    """A signature learned on the simulated bench must never stand in for
    the real one under the same name (or the reverse): listings and name
    resolution take the newest version whose JSON "mock" matches the mode."""
    bench = api.mock_bench()
    _two_boxes(bench)
    r = api.spectral_capture(fast, "modes")
    m = api.learn_material(r, "red_box", _roi(bench, RED_C))
    assert m.mock is True
    doc = json.loads(m.version_path.read_text())

    def add_version(name, stamp, mock, sig):
        d = api.paths()["spectral"] / "materials" / name
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{stamp}.json").write_text(json.dumps(
            {**doc, "name": name, "mock": mock, "signature": sig}))
        return d / f"{stamp}.json"

    # a NEWER version of the same name, learned on the real rig (other mode)
    real_v = add_version("red_box", "29990101T000000_000000", False, [0.1, 0.1, 0.8])
    # a material that only exists on the real rig
    add_version("real_only", "29990101T000000_000000", False, [0.3, 0.3, 0.4])

    listed = {x.name: x for x in api.list_library("materials")}      # mock session
    assert set(listed) == {"red_box"}
    assert listed["red_box"].version_path == m.version_path and listed["red_box"].mock is True
    s = api._engine()
    real = {x.name: x for x in spectral.list_materials(s, mock=False)}
    assert set(real) == {"red_box", "real_only"} and real["red_box"].version_path == real_v

    c = api.classify(r, ["red_box"])                  # resolves to the mock signature
    assert c.materials == ["red_box"] and c.coverage_pct["red_box"] > 0
    assert not any("real mode" in w for w in c.warnings)
    with pytest.raises(ValueError, match="unknown material"):
        api.classify(r, ["real_only"])

    # a version without the key (older library) counts for either mode
    legacy = add_version("legacy", "20000101T000000_000000", None, [0.4, 0.3, 0.3])
    assert {x.name for x in api.list_library("materials")} == {"red_box", "legacy"}
    assert {x.name for x in spectral.list_materials(s, mock=False)} >= {"legacy"}
    # a half-written newest version is skipped, the one before it is used
    (m.path / "29990202T000000_000000.json").write_text('{"name": "red_box", "sig')
    assert {x.name: x for x in api.list_library("materials")}["red_box"].version_path \
        == m.version_path
    assert legacy.exists()

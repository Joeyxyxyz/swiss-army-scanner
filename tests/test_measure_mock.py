"""Measure-area regressions: zone-scan segmentation, the inspect verdict vs its
detection threshold, and the good-bank model (floor without a base plane,
bounded memory). Synthetic reliefs where the mock bench would be slow; one
mock zone scan end to end."""

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from engine import api
from engine.capture import zones as Z
from engine.capture.mock_bench import Box
from engine.measure import bank as bank_mod
from engine.measure import dimensions, segment
from engine.measure import inspect as inspect_mod
from engine.measure import profile as profile_mod
from engine.results import Calibration

from conftest import PRESET


@dataclasses.dataclass
class FakeScan:
    """The ObjectScan fields the measure modules read, over a synthetic relief."""
    id: str
    label: str
    geometry_id: str
    path: Path
    relief: np.ndarray
    relief_mask: np.ndarray
    meta: dict = dataclasses.field(default_factory=dict)
    known_dims: dict = None


GEO = "synthetic_geo"
ZONE = (slice(40, 200), slice(60, 260))            # the lit zone: 160 x 200 px


def zone_relief(part=(slice(60, 180), slice(80, 240)), height=9.0, noise=0.03,
                nan_grid=False, seed=0):
    """A datum-free zone relief: base plane at 0, part top at `height`, NaN
    outside the zone, a thin opposite-sign shadow strip along the part."""
    rng = np.random.default_rng(seed)
    rel = np.full((240, 320), np.nan)
    rel[ZONE] = rng.normal(0.0, noise, (160, 200))
    rel[part] += height
    top = np.zeros(rel.shape, bool)
    top[part] = True
    top &= np.isfinite(rel)
    if nan_grid:                                   # thin invalid lines across the part top
        grid = np.zeros(rel.shape, bool)
        grid[::7, :] = True
        grid[:, ::7] = True
        rel[grid & top] = np.nan
    ys, xs = np.nonzero(top)
    x1 = xs.max() + 1
    if x1 + 3 <= ZONE[1].stop:                     # shadow on the right of the part
        rel[ys.min():ys.max() + 1, x1:x1 + 3] = -3.0
    return rel.astype(np.float32), top & np.isfinite(rel)


def fake(tmp_path, rel, **meta):
    d = tmp_path / "objects" / "fake"
    d.mkdir(parents=True, exist_ok=True)
    return FakeScan("fake", "fake", GEO, d, rel, np.isfinite(rel), dict(meta))


# ------------------------------------------------------------- segmentation

def test_datum_free_blob_is_the_part_when_it_fills_most_of_the_zone():
    rel, top = zone_relief()                       # part = 60% of the zone
    assert top.sum() > 0.5 * np.isfinite(rel).sum()
    # the old median reference lands on the part top and takes the base ring,
    # which is now at least called out
    ring, h_old, info_old = segment.object_blob(rel)
    assert abs(h_old) < 0.2 and not (ring & top).any()
    assert any("within noise of the base plane" in w for w in info_old["warnings"])
    # around the base level 0 the part is found
    blob, h, info = segment.object_blob(rel, ref=0.0)
    assert h == pytest.approx(9.0, abs=0.02)
    assert (blob & top).sum() >= 0.95 * top.sum() and not (blob & ~top).any()
    assert info["base_in_view"] and abs(info["base_level_px"]) < 0.01
    assert not info["warnings"]


def test_datum_free_blob_when_the_part_fills_the_whole_zone():
    rel, top = zone_relief(part=ZONE)
    blob, h, info = segment.object_blob(rel, ref=0.0)
    assert h == pytest.approx(9.0, abs=0.02)
    assert blob.sum() >= 0.95 * top.sum()
    assert info["base_in_view"] is False
    assert any("no base plane in view" in w for w in info["warnings"])


def test_part_top_split_by_thin_nan_lines_stays_one_blob():
    rel, top = zone_relief(nan_grid=True)
    blob, h, _ = segment.object_blob(rel, ref=0.0)
    assert h == pytest.approx(9.0, abs=0.02)
    assert blob.sum() >= 0.9 * top.sum()           # not one 6x6 cell of the grid
    # the datum'd (normal) path gets the same robustness (part a minority)
    rel2, top2 = zone_relief(part=(slice(80, 140), slice(100, 180)), nan_grid=True)
    blob2, h2, _ = segment.object_blob(rel2)
    assert h2 == pytest.approx(9.0, abs=0.02) and blob2.sum() >= 0.9 * top2.sum()


def test_zone_scan_measure_inspect_profile_use_the_part(tmp_path, eng):
    rel, top = zone_relief()
    scan = fake(tmp_path, rel, zone="D4", datum_px=0.0, overlap_px=int(np.isfinite(rel).sum()))
    assert segment.is_datum_free(scan)
    d = dimensions.measure(scan)
    assert d.H == pytest.approx(9.0, abs=0.02) and d.units == "px"
    assert d.W == pytest.approx(160, abs=3) and d.L == pytest.approx(120, abs=3)
    assert not any("within noise" in w or "base plane" in w for w in d.warnings)

    insp = inspect_mod.inspect(scan)
    assert insp.region_px > 5000 and "no part found" not in " ".join(insp.warnings)

    pr = profile_mod.profile(api._engine(), scan, None, axis="x")
    assert pr.stats["step_height"] == pytest.approx(9.0, abs=0.05)

    # the part filling the zone: H stays right, W/L are flagged as the zone's,
    # and a calibration sample with a known W/L is refused
    full, _ = zone_relief(part=ZONE)
    filled = fake(tmp_path, full, zone="D4", datum_px=0.0, overlap_px=1)
    d2 = dimensions.measure(filled)
    assert d2.H == pytest.approx(9.0, abs=0.02)
    assert any("no base plane in view" in w for w in d2.warnings)
    assert dimensions.raw_sample(filled)["relief_px"] == pytest.approx(9.0, abs=0.02)
    filled.known_dims = {"H": 20.0, "W": 80.0}
    with pytest.raises(ValueError, match="W/L are the zone's"):
        dimensions.raw_sample(filled)
    filled.known_dims = {"H": 20.0}
    assert dimensions.raw_sample(filled)["wl_ok"] is False
    pr2 = profile_mod.profile(api._engine(), filled, None, axis="auto")
    assert pr2.stats["median"] == pytest.approx(9.0, abs=0.05)
    assert any("no base plane in view" in w for w in pr2.warnings)


def test_zone_scan_part_filling_most_of_the_zone_mock(eng):
    """The reported case: zone D4, a box covering most of it, then one larger
    than the zone. Before: H ~ 0 / W,L = the zone (base ring), then errors."""
    p = api.load_preset(PRESET)
    p.camera.average = 1
    base = api.capture_base(p, "empty")
    empty = api.capture_map(p, "empty")
    bench = api.mock_bench()
    x0, y0, x1, y1 = Z.cell_rect((3, 3), Z.grid_spec(96))
    u, v = empty.to_camera([(x0 + x1) / 2, (y0 + y1) / 2])
    H, W = bench.cam_shape
    centre = ((u - W / 2) / bench.cam_px_per_mm, (v - H / 2) / bench.cam_px_per_mm)

    bench.place(Box(20.0, 30.0, 20.0, center_mm=centre))
    api.capture_map(p, "with part")
    scan = api.capture_object(p, base, "zoned", zone="D4")
    on_box = scan.relief_mask & (bench.height_map() > 0)
    assert on_box.sum() > 0.5 * scan.relief_mask.sum()          # the part is the majority
    d = api.measure(scan)
    assert d.H == pytest.approx(0.45 * 20.0, abs=0.3), d
    k = bench.cam_px_per_mm
    assert d.W == pytest.approx(30 * k, rel=0.1) and d.L == pytest.approx(20 * k, rel=0.1), d
    blob, _, _ = segment.scan_blob(scan)
    assert (blob & on_box).sum() >= 0.95 * blob.sum()
    insp = api.inspect(scan)
    assert insp.region_px > 200 and "inspection region too small" not in insp.warnings
    pr = api.profile(scan, axis="auto")
    assert pr.stats["step_height"] == pytest.approx(9.0, abs=0.3)

    bench.clear()
    bench.place(Box(20.0, 80.0, 80.0, center_mm=centre))        # bigger than the zone
    api.capture_map(p, "big part")
    big = api.capture_object(p, base, "zoned_big", zone="D4")
    d2 = api.measure(big)
    assert d2.H == pytest.approx(9.0, abs=0.3)
    assert any("no base plane in view" in w for w in d2.warnings)
    pr2 = api.profile(big, axis="auto")
    assert pr2.stats["median"] == pytest.approx(9.0, abs=0.3)


# ---------------------------------------------------------------- inspect

def plate_scan(tmp_path, noise, bump_px=0.0, seed=1):
    """Base 0 with a 9 px plate (a minority of the field) and an optional
    40x40 px bump on it. Datum'd relief, like a normal scan."""
    rng = np.random.default_rng(seed)
    rel = rng.normal(0.0, noise, (400, 600))
    rel[100:300, 150:450] += 9.0
    rel[180:220, 280:320] += bump_px
    return fake(tmp_path, rel.astype(np.float32))


CAL = Calibration(id=f"{GEO}/c", geometry_id=GEO, mm_per_px=2.222, lateral_px_per_mm=3.0)


def test_inspect_passes_only_when_the_threshold_is_safely_below_the_trigger(tmp_path):
    # quiet scan: threshold (the calibrated floor, trigger/2) is well below the trigger
    good = inspect_mod.inspect(plate_scan(tmp_path, 0.005), CAL, trigger_mm=0.7)
    assert good.verdict == "PASS", good
    assert good.threshold + 2 * good.noise_sigma <= 0.7
    bad = inspect_mod.inspect(plate_scan(tmp_path, 0.005, bump_px=1.5 / 2.222), CAL,
                              trigger_mm=0.7)
    assert bad.verdict == "FAIL" and bad.defects[0].height == pytest.approx(1.5, abs=0.05)

    # noisy scan (sigma 0.27 mm -> threshold ~1.6 mm): a 1.0 mm bump is over the
    # 0.7 mm trigger but under the threshold, so it is not seen. That used to be
    # a confident PASS; now it cannot pass.
    noisy = inspect_mod.inspect(plate_scan(tmp_path, 0.12, bump_px=1.0 / 2.222), CAL,
                                trigger_mm=0.7)
    assert noisy.threshold > 0.7 and not any(d.exceeds_trigger for d in noisy.defects)
    assert noisy.verdict == "INDETERMINATE", noisy
    msg = [w for w in noisy.warnings if "cannot PASS" in w]
    assert msg and "0.7 mm trigger" in msg[0] and "camera.average" in msg[0]
    # a defect over the trigger that IS seen still fails, whatever the threshold
    big = inspect_mod.inspect(plate_scan(tmp_path, 0.12, bump_px=4.0 / 2.222), CAL,
                              trigger_mm=0.7)
    assert big.verdict == "FAIL"
    # uncalibrated / no trigger: never a verdict
    assert inspect_mod.inspect(plate_scan(tmp_path, 0.005)).verdict == "INDETERMINATE"
    nt = inspect_mod.inspect(plate_scan(tmp_path, 0.005), CAL, trigger_mm=None)
    assert nt.verdict == "INDETERMINATE" and any("no trigger" in w for w in nt.warnings)


# ------------------------------------------------------------ memory bank

def _reference_model(reliefs):
    """The pre-chunking model: full stack, NaN median, MAD (what build_model must equal)."""
    R = np.stack(reliefs).astype(np.float32)
    ref = bank_mod._nan_median0(R)
    sig = np.nan_to_num(1.4826 * bank_mod._nan_median0(np.abs(R - ref[None])), nan=0.0)
    from scipy import ndimage
    return ref, ndimage.maximum_filter(sig, size=3), np.isfinite(R).sum(axis=0)


def test_bank_model_is_chunked_cropped_and_unchanged(monkeypatch):
    rng = np.random.default_rng(3)
    reliefs = []
    for i in range(4):
        r = np.full((120, 160), np.nan, np.float32)
        r[10 + i:100, 20:150 - i] = rng.normal(0, 0.1, (90 - i, 130 - i))
        r[40:80, 50:110] += 9.0
        reliefs.append(r)
    monkeypatch.setattr(bank_mod, "CHUNK_ELEMS", 2000)           # many row chunks
    m = bank_mod.build_model(iter(reliefs))                      # any iterable
    ref, sig, nv = _reference_model(reliefs)
    assert np.array_equal(np.isnan(m["reference"]), np.isnan(ref))
    assert np.allclose(np.nan_to_num(m["reference"]), np.nan_to_num(ref), atol=1e-6)
    assert np.allclose(m["sigma"], sig, atol=1e-6) and np.array_equal(m["n_valid"], nv)
    assert str(m["floor_source"]) == "base_plane" and int(m["n_members"]) == 4
    with pytest.raises(ValueError, match="relief is"):
        bank_mod.build_model([reliefs[0], reliefs[0][:100]])


def test_bank_floor_without_a_base_plane_comes_from_member_differences():
    rng = np.random.default_rng(4)
    members = []
    for i in range(2):                              # zone scans of a part filling the zone
        r = np.full((200, 300), np.nan, np.float32)
        r[40:160, 60:240] = 9.0 + rng.normal(0, 0.15, (120, 180))
        members.append(r)
    assert all(bank_mod.base_plane_sigma(r) is None for r in members)
    m = bank_mod.build_model(members)
    assert str(m["floor_source"]) == "member_diff"
    assert float(m["noise_floor"]) == pytest.approx(0.15, rel=0.1)   # not the 0.05 minimum
    one = bank_mod.build_model(members[:1])
    assert str(one["floor_source"]) == "minimum"
    assert float(one["noise_floor"]) == bank_mod.MIN_FLOOR_PX


def _member(root, name, rel, zone="D4"):
    d = root / "objects" / name
    d.mkdir(parents=True)
    np.savez_compressed(d / "relief.npz", relief=rel, mask=np.isfinite(rel))
    return FakeScan(name, name, GEO, d, rel, np.isfinite(rel), {"zone": zone})


def test_bank_check_warns_about_its_floor_and_caps_the_members(eng, monkeypatch):
    s = api._engine()
    rng = np.random.default_rng(5)

    def part(i):
        r = np.full((200, 300), np.nan, np.float32)
        r[40:160, 60:240] = 9.0 + rng.normal(0, 0.15, (120, 180))
        return r
    goods = [_member(s.store.root, f"g{i}", part(i)) for i in range(4)]
    monkeypatch.setattr(bank_mod, "MAX_MODEL_MEMBERS", 3)
    for g in goods:
        bank_mod.add(s, g, "cans")
    v = bank_mod.check(s, _member(s.store.root, "new", part(9)), "cans")
    assert v.result == "PASS", v
    assert v.n_ref == 4 and v.n_compared_px == 120 * 180    # judged with a 3-member model
    assert any("model uses the newest 3" in w for w in v.warnings)
    assert any("estimated from differences" in w for w in v.warnings)
    assert v.noise_floor == pytest.approx(0.15, rel=0.15)


def test_golden_compare_region_split_by_thin_invalid_lines_still_fails(tmp_path):
    from engine.measure import golden
    from engine.results import ToleranceBand
    flat = np.zeros((100, 100), np.float32)
    bumped = flat.copy()
    bumped[40:48, 40:50] = 2.0                     # an 8x10 px defect ...
    bumped[:, 43::4] = np.nan                      # ... cut into 3 px strips (< 25 px each)
    g = fake(tmp_path, flat)
    s = fake(tmp_path, bumped)
    v = golden.compare_relief(s, g, ToleranceBand(fail_px=1.0, min_defect_px=25, smooth_px=1))
    assert v.result == "FAIL" and v.n_fail_px >= 60, v

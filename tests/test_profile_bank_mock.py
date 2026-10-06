"""2D profiles and the good memory bank, end to end in mock mode."""

import csv
import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest

from engine import api, rules
from engine.measure import bank as bank_mod
from engine.measure import profile as profile_mod

from conftest import PRESET, Box, VALIDATION_OBJECT

V = {"H": 106.95, "W": 56.50, "L": 21.90}           # §11 validation object
PART = Box(30.0, 90.0, 70.0)                        # flat part, 30 mm tall
BUMP = Box(34.0, 12.0, 12.0, center_mm=(15, 10))    # 4 mm bump on its top
JITTER = [(0.0, 0.0), (0.25, -0.2), (-0.2, 0.25)]   # sub-mm re-placement of good parts


@pytest.fixture
def fast(eng):
    p = api.load_preset(PRESET)
    p.camera.average = 1          # speed; the mock's relief is already clean at 1
    return p


def shoot(preset, base, label, *objs, known=None):
    bench = api.mock_bench()
    bench.clear()
    for o in objs:
        bench.place(o)
    return api.capture_object(preset, base, label, known_dims=known)


def moved(box, dx, dy):
    return dataclasses.replace(box, center_mm=(box.center_mm[0] + dx, box.center_mm[1] + dy))


def exports(scan, kind):
    """Output-bin pointers the scan's entry got (exports.jsonl), of one kind."""
    lines = (Path(scan.path) / "exports.jsonl").read_text(encoding="utf-8").splitlines()
    return [r for r in map(json.loads, lines) if r["kind"] == kind]


def contains(bbox, inner, slack=2):
    x0, y0, x1, y1 = bbox
    a0, b0, a1, b1 = inner
    return x0 <= a0 + slack and y0 <= b0 + slack and x1 >= a1 - slack and y1 >= b1 - slack


# ------------------------------------------------------------- profiles

@dataclasses.dataclass
class FakeScan:
    """Just the ObjectScan fields profile() reads — exact synthetic relief."""
    id: str
    label: str
    geometry_id: str
    path: Path
    relief: np.ndarray
    relief_mask: np.ndarray
    meta: dict = dataclasses.field(default_factory=dict)


def test_profile_samples_exact_synthetic_step(eng):
    rel = np.zeros((200, 400), np.float32)
    rel[:, 200:] = 10.0
    mask = np.ones(rel.shape, bool)
    mask[:, 250:261] = False                           # a shadow band: must be NaN, not 0
    rel[~mask] = np.nan
    d = eng.store.root / "objects" / "fake"
    d.mkdir(parents=True)
    scan = FakeScan("fake", "fake", "synthetic_geo", d, rel, mask)

    pr = api.profile(scan, p0=[100, 50], p1=[300, 50], width_px=5)
    assert pr.axis == "points" and pr.p0 == [100, 50] and pr.p1 == [300, 50]
    dist, h = np.array(pr.distance), np.array(pr.height)
    assert len(dist) == 201 and dist[-1] == pytest.approx(200.0)
    assert np.allclose(h[dist < 99], 0.0) and np.allclose(h[(dist > 101) & (dist < 149)], 10.0)
    assert np.isnan(h[(dist > 151) & (dist < 160)]).all()      # the band, 1 px bilinear reach
    assert pr.n_samples[0] == 5
    assert pr.units_height == "px" and pr.units_distance == "px" and pr.tier == "relief"
    assert any("rule 6" in w for w in pr.warnings)
    assert pr.stats["step_height"] == pytest.approx(10.0, abs=1e-6)
    assert pr.stats["min"] == 0.0 and pr.stats["max"] == 10.0

    rows = list(csv.DictReader(open(pr.csv_path, newline="", encoding="utf-8")))
    assert list(rows[0]) == ["distance_px", "height_px", "n_samples"] and len(rows) == 201
    assert rows[155]["height_px"] == "nan" and rows[155]["n_samples"] == "0"
    assert pr.png_path.exists() and pr.png_path.read_bytes()[:4] == b"\x89PNG"
    assert pr.csv_path.parent == api.paths()["output"] / "profiles"

    with pytest.raises(ValueError, match="both p0 and p1"):
        api.profile(scan, p0=[1, 1])
    with pytest.raises(ValueError, match="axis"):
        api.profile(scan, axis="diagonal")
    with pytest.raises(ValueError, match="1 px apart"):
        api.profile(scan, p0=[5, 5], p1=[5.2, 5])

    # an empty scene: no object to put an auto line on -> say so, points still work
    empty = FakeScan("empty", "empty", "synthetic_geo", d, np.zeros_like(rel), np.ones_like(mask))
    with pytest.raises(ValueError, match="pass p0 and p1"):
        api.profile(empty)
    flat = api.profile(empty, p0=[10, 100], p1=[390, 100])
    assert flat.stats["step_height"] is None and flat.stats["median"] == 0.0


def test_profile_stats_only_report_real_steps():
    rng = np.random.default_rng(0)
    flat = rng.normal(0, 0.1, 400)
    ramp = np.linspace(0, 10, 400) + rng.normal(0, 0.05, 400)
    step = np.r_[rng.normal(0, 0.1, 150), rng.normal(13.5, 0.1, 250)]
    assert profile_mod.profile_stats(flat, "px")["step_height"] is None
    assert profile_mod.profile_stats(ramp, "px")["step_height"] is None
    s = profile_mod.profile_stats(np.r_[step, [np.nan] * 20], "px")
    assert s["step_height"] == pytest.approx(13.5, abs=0.05) and s["n_valid"] == 400
    assert profile_mod.profile_stats([np.nan] * 5, "mm")["median"] is None


def test_profile_auto_across_box_px_then_mm(fast):
    base = api.capture_base(fast, "empty")
    box = Box(20.0, 60.0, 25.0, center_mm=(15, -10), angle_deg=30)
    part = shoot(fast, base, "block20", box)

    raw = api.profile(part)                            # uncalibrated: px + rule 6 warning
    assert raw.axis == "auto"
    assert raw.units_height == "px" and raw.units_distance == "px" and raw.tier == "relief"
    assert any("rule 6" in w for w in raw.warnings)
    assert raw.stats["step_height"] == pytest.approx(20.0 * 0.45, abs=0.45)
    (x0, y0), (x1, y1) = raw.p0, raw.p1
    assert math.degrees(math.atan2(y1 - y0, x1 - x0)) == pytest.approx(30.0, abs=3.0)
    assert math.hypot(x1 - x0, y1 - y0) > 60 * 3 + 2 * 40      # long axis + base both ends
    h = np.array(raw.height)
    assert abs(h[np.isfinite(h)][0]) < 0.3 and abs(h[np.isfinite(h)][-1]) < 0.3   # base ends

    cal_scan = shoot(fast, base, "validation", VALIDATION_OBJECT, known=V)
    api.calibrate(fast, [cal_scan])
    pr = api.profile(part)
    assert pr.units_height == "mm" and pr.units_distance == "mm"
    assert pr.tier == "lateral_calibrated"
    assert pr.stats["step_height"] == pytest.approx(20.0, abs=1.0)
    assert pr.stats["step_low"] == pytest.approx(0.0, abs=1.0)
    assert pr.distance[-1] == pytest.approx(math.hypot(x1 - x0, y1 - y0) / 3.0, rel=0.03)
    assert not any("rule 6" in w for w in pr.warnings)

    for axis in ("x", "y"):
        pa = api.profile(part, axis=axis)
        assert pa.axis == axis
        assert (pa.p0[1] == pa.p1[1]) if axis == "x" else (pa.p0[0] == pa.p1[0])
        assert pa.stats["step_height"] == pytest.approx(20.0, abs=1.0)

    cx, cy = 720 + 15 * 3, 540 - 10 * 3                   # box centre in camera px
    pe = api.profile(part, p0=[cx - 150, cy], p1=[cx + 150, cy], width_px=9)
    assert pe.p0 == [cx - 150, cy] and pe.width_px == 9 and max(pe.n_samples) == 9
    assert pe.distance[-1] == pytest.approx(300 / 3.0, rel=0.03)
    assert pe.stats["step_height"] == pytest.approx(20.0, abs=1.0)
    rows = list(csv.DictReader(open(pe.csv_path, newline="", encoding="utf-8")))
    assert list(rows[0]) == ["distance_mm", "height_mm", "n_samples"]

    outs = api.list_library("outputs", "profiles")
    assert len(outs) == 2 * 5                             # 5 profiles, csv + png each
    assert {Path(o["file"]).suffix for o in outs} == {".csv", ".png"}
    assert all(o["geometry_id"] == fast.geometry_id for o in outs)
    assert all((api.paths()["library"] / o["file"]).exists() for o in outs)
    assert len(exports(part, "profiles")) == 10


# ---------------------------------------------------------- memory bank

def test_bank_pass_flag_and_membership(fast):
    base = api.capture_base(fast, "empty")
    goods = [shoot(fast, base, f"good{i}", moved(PART, dx, dy))
             for i, (dx, dy) in enumerate(JITTER)]

    empty = api.bank_check(goods[0], "cells")
    assert empty.result == "INDETERMINATE" and empty.n_ref == 0
    assert json.loads(empty.json_path.read_text())["max_z"] is None     # strict JSON, no NaN

    infos = [api.bank_add(g, "cells") for g in goods]
    assert any("at least 3" in w for w in infos[0].warnings)
    info = infos[-1]
    assert info.n_scans == 3 and info.scan_ids == [g.id for g in goods]
    assert info.geometry_id == fast.geometry_id and info.path.name == "cells"
    assert info.path.parent.name == fast.geometry_id and info.path.parent.parent.name == "banks"
    lines = (info.path / "members.jsonl").read_text().splitlines()
    assert [json.loads(l)["scan_id"] for l in lines] == [g.id for g in goods]
    again = api.bank_add(goods[0], "cells")               # idempotent, never double-counted
    assert again.n_scans == 3 and any("already" in w for w in again.warnings)

    good = shoot(fast, base, "good_new", moved(PART, 0.1, 0.15))
    ok = api.bank_check(good, "cells")
    assert ok.result == "PASS", ok
    assert ok.n_ref == 3 and ok.units == "px" and ok.anomalies == []
    assert ok.max_z < ok.k_sigma <= ok.max_z_pixel     # edge slivers exist but don't count
    assert any("rule 6" in w for w in ok.warnings)
    assert ok.overlap_pct > 99 and ok.n_compared_px > 500_000
    # the mock camera now draws fresh noise per capture (it used to repeat the
    # same noise, which pinned the floor at the minimum): the floor is the
    # members' measured base-plane noise, never below the minimum
    assert bank_mod.MIN_FLOOR_PX <= ok.noise_floor < 0.3
    models = list(info.path.glob("model_3_*.npz"))
    assert len(models) == 1                               # built once, cached

    bad = shoot(fast, base, "bumped", PART, BUMP)
    hmap = api.mock_bench().height_map()
    ys, xs = np.nonzero(hmap > PART.h_mm + 1)
    bump_bbox = [xs.min(), ys.min(), xs.max(), ys.max()]
    fl = api.bank_check(bad, "cells")
    assert fl.result == "FLAG" and fl.max_z > 10
    top = fl.anomalies[0]
    assert contains(top["bbox"], bump_bbox), (top, bump_bbox)
    assert top["area_px"] > 800 and top["units"] == "px"
    assert top["mean_dev"] == pytest.approx(4.0 * 0.45, abs=0.3)
    assert len(list(info.path.glob("model_*.npz"))) == 1  # same members -> same model

    # rule 6: once calibrated, deviations come back in mm
    cal_scan = shoot(fast, base, "validation", VALIDATION_OBJECT, known=V)
    api.calibrate(fast, [cal_scan])
    fl_mm = api.bank_check(bad, "cells")
    assert fl_mm.result == "FLAG" and fl_mm.units == "mm" and fl_mm.tier == "depth_calibrated"
    assert fl_mm.anomalies[0]["mean_dev"] == pytest.approx(4.0, abs=0.5)
    assert fl_mm.anomalies[0]["area_mm2"] == pytest.approx(144, rel=0.25)

    # outputs: JSON (+ PNG when judged) in the output bin, pointed at from the scan
    outs = api.list_library("outputs", "anomalies")
    assert len([o for o in outs if o["file"].endswith(".json")]) == 4
    assert len([o for o in outs if o["file"].endswith(".png")]) == 3
    doc = json.loads(fl_mm.json_path.read_text())
    assert doc["result"] == "FLAG" and doc["units"] == "mm" and doc["scan"] == bad.id
    assert fl_mm.png_path.read_bytes()[:4] == b"\x89PNG"
    assert len(exports(bad, "anomalies")) == 4             # 2 checks x (json + png)

    banks = api.list_library("banks")
    assert [(b.name, b.n_scans) for b in banks] == [("cells", 3)]

    # membership is append-only: removing / trashing a member never rewrites anything
    after = bank_mod.remove(api._engine(), goods[2], "cells", reason="test")
    assert after.n_scans == 2 and goods[2].id not in after.scan_ids
    api.delete(goods[1], reason="test")
    two = api.bank_check(good, "cells")
    assert two.n_ref == 1 and two.result == "PASS"
    assert any("missing from the library" in w for w in two.warnings)
    assert any("noise floor" in w for w in two.warnings)
    assert len((info.path / "members.jsonl").read_text().splitlines()) == 4
    assert len(list(info.path.glob("model_1_*.npz"))) == 1 and len(models) == 1
    assert models[0].exists()


def test_bank_refuses_other_geometry(fast):
    base = api.capture_base(fast, "empty")
    good = shoot(fast, base, "good", PART)
    info = api.bank_add(good, "cells")
    foreign = dataclasses.replace(good, id="moved", geometry_id="some_other_rig")

    with pytest.raises(rules.GeometryMismatch, match="some_other_rig"):
        api.bank_add(foreign, "cells")
    with pytest.raises(rules.GeometryMismatch):
        api.bank_check(foreign, "cells")
    assert [b.geometry_id for b in api.list_library("banks")] == [fast.geometry_id]
    assert not (api.paths()["library"] / "banks" / "some_other_rig").exists()

    # the ways forward after moving the rig: a new bank name, or trash the old bank
    assert api.bank_add(foreign, "cells_rig2").geometry_id == "some_other_rig"
    api.delete(info, reason="rig moved")
    assert api.bank_add(foreign, "cells").n_scans == 1
    assert {(b.name, b.geometry_id) for b in api.list_library("banks")} == \
        {("cells", "some_other_rig"), ("cells_rig2", "some_other_rig")}
    assert (api.paths()["trash"] / "banks" / fast.geometry_id / "cells" / "TRASHED.json").exists()

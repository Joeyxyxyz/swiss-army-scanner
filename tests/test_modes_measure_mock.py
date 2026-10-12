"""MEASURE mode (engine/modes/measure.py, engine/measure/values.py), MOCK only: an instrument
that reports numbers and decides nothing.

Every measurement type on scenes with known truth - first as pure functions on synthetic
clouds (a pinhole camera over a wall: fast), then through api.measure_run on synthetic metric
scans of the mock stereo bench (tests/recipe_synth.py: saved as the library saves a scan,
nothing captured): values within tolerance, units, n points, an uncertainty on every value,
the near-resolution-limit note, repeats. The run folder (result.json, values.csv, overlay,
PLY, spec). NOT_MEASURABLE - never a number, never a px as mm - without a calibration, with a
stale one, on a rig whose check said MOVED (the mobile-rig paths are in
test_modes_rig_mock.py), and whenever the number would not be what its name says: a
"cylinder" that is flat or that the points do not lie on, a part cut by the image border or
the roi, a patch too far from the golden to compare, a picked px on an edge, a zone that can't
cut a stored scan, the same scan given twice as "repeats". Each value's uncertainty is the
noise of the px it rests on, and repeats never report less than one scan resolves / sqrt(n).
A measurement never reads as a verdict: result null, disposition
NONE, a PULL (never a CONTINUE, never a permission) for whatever takes it for a run. The
CLI's exit codes (0 MEASURED, 5 NOT MEASURABLE, 2 refused) and one measurement through the
full mock capture path."""

import contextlib
import csv
import io
import json
import math

from types import SimpleNamespace

import numpy as np
import pytest

from engine import api, config
from engine import disposition as D
from engine import runs as R
from engine.capture.mock_bench import MockBench
from engine.capture.mock_stereo import Block, MetricCylinder
from engine.control import commands as C
from engine.flow import cycle as FC
from engine.flow import schema as FS
from engine.measure import values as V
from engine.modes import measure as MM
from engine.reconstruct.triangulate import MetricCloud
from ui import cli

import recipe_synth as RS

PRESET = "bench_9in_metric"
F, Z0, H, W = 1600.0, 500.0, 300, 400        # 0.3125 mm / px on a wall 500 mm away
PX = Z0 / F


@pytest.fixture(autouse=True)
def _fresh_engine():
    yield
    api._session = None


# ================================================================== synthetic clouds (pure)

def cloud(h_fn=lambda u, v: 0.0, noise=0.01, seed=0, valid=None) -> MetricCloud:
    """A wall at Z0 with heights h_fn(u, v) (mm toward the camera) seen by a pinhole camera;
    (u, v) = wall mm. Heights move a point along z only (as the recipe tests' clouds)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    u, v = (xx - W / 2) / F * Z0, (yy - H / 2) / F * Z0
    h = np.asarray(h_fn(u, v), float) + np.zeros_like(u)
    ok = np.isfinite(h) if valid is None else (np.isfinite(h) & valid(u, v))
    xyz = np.stack([u, v, Z0 - np.nan_to_num(h) + rng.normal(0, noise, u.shape)], -1)
    xyz[~ok] = np.nan
    return MetricCloud(xyz=xyz, mask=ok, consistency=None, method="plane",
                       geometry_id="synthetic", calibration_id="synthetic")


def box(u0, v0, u1, v1, h):
    return lambda u, v: np.where((u >= u0) & (u <= u1) & (v >= v0) & (v <= v1), h, 0.0)


def px_of(u, v):
    return [u / PX + W / 2, v / PX + H / 2]


def roi_of(u0, v0, u1, v1):
    (x0, y0), (x1, y1) = px_of(u0, v0), px_of(u1, v1)
    return [int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))]


def by_name(out) -> dict:
    return {v["name"]: v for v in out["values"]}


def test_flatness_of_a_flat_and_of_a_tilted_region():
    c = cloud(noise=0.01)
    f = by_name(V.flatness(c))
    assert f["flatness_rms"]["value"] == pytest.approx(0.01, abs=0.002)
    assert f["flatness_rms"]["units"] == "mm" and f["flatness_rms"]["n_points"] == H * W
    assert 0.03 < f["flatness_p2_p98"]["value"] < 0.05           # ~4.1 sigma of the noise
    assert f["tilt"]["units"] == "deg" and f["tilt"]["value"] < 0.02
    assert f["plane_distance"]["value"] == pytest.approx(Z0, abs=0.01)
    # a plane tilted 3 deg about the camera's y axis: flat, and its tilt is read
    tilt = cloud(lambda u, v: math.tan(math.radians(3.0)) * u)
    t = by_name(V.flatness(tilt, V.roi_mask((H, W), [100, 50, 300, 250])))
    assert t["tilt"]["value"] == pytest.approx(3.0, abs=0.03)
    assert t["flatness_rms"]["value"] < 0.02 and t["tilt"]["n_points"] == 200 * 200
    assert t["tilt"]["lever_mm"] == pytest.approx(0.96 * 200 * PX, rel=0.05)   # p2..p98 of 62 mm
    # a warped face is not flat: a 0.5 mm bow shows in both numbers
    bow = by_name(V.flatness(cloud(lambda u, v: 0.5 * np.cos(u / 60.0 * math.pi / 2))))
    assert bow["flatness_p2_p98"]["value"] > 0.3 and bow["flatness_rms"]["value"] > 0.1


def test_step_height_plane_to_plane_and_its_parallelism_note():
    c = cloud(box(-20, -15, 20, 15, 5.0))
    a, b = V.roi_mask((H, W), roi_of(-55, -15, -30, 15)), V.roi_mask((H, W), roi_of(-15, -10, 15, 10))
    s = by_name(V.step(c, a, b))
    assert s["step_height"]["value"] == pytest.approx(5.0, abs=0.005)
    assert s["step_height"]["units"] == "mm" and s["step_height"]["notes"] == []
    assert s["parallelism"]["value"] < 0.1 and s["parallelism"]["units"] == "deg"
    assert s["step_height"]["n_points"] == min(s["step_height"]["n_points_a"],
                                               s["step_height"]["n_points_b"])
    down = by_name(V.step(c, b, a))                 # the other way round: the wall is below
    assert down["step_height"]["value"] == pytest.approx(-5.0, abs=0.005)
    # faces 2 deg apart: the value stands, and says that it depends on where it is taken
    tilted = cloud(lambda u, v: np.where(np.abs(u) < 20, 5.0 + math.tan(math.radians(2.0)) * u,
                                         0.0))
    t = by_name(V.step(tilted, a, V.roi_mask((H, W), roi_of(-15, -10, 15, 10))))
    assert t["parallelism"]["value"] == pytest.approx(2.0, abs=0.1)
    assert any("deg apart" in n for n in t["step_height"]["notes"])


def test_distance_between_picked_px_and_point_to_plane():
    c = cloud(box(-20, -15, 20, 15, 5.0))
    p1, p2 = px_of(-10, 0), px_of(10, 0)            # both on the top face, 5 mm proud
    out = V.distance(c, p1, p2, V.roi_mask((H, W), roi_of(-55, -15, -30, 15)))
    d = by_name(out)
    assert d["distance"]["value"] == pytest.approx(20.0, abs=PX)
    assert d["distance"]["sensitivity"] == "both" and d["distance"]["units"] == "mm"
    assert d["point_to_plane"]["value"] == pytest.approx(5.0, abs=0.01)
    assert d["point2_to_plane"]["value"] == pytest.approx(5.0, abs=0.01)
    assert d["point_to_plane"]["sensitivity"] == "depth"
    # across the step: the 3-D distance, not the px distance times a scale
    x = by_name(V.distance(c, px_of(-40, 0), px_of(0, 0)))
    assert x["distance"]["value"] == pytest.approx(math.hypot(40.0, 5.0), abs=PX)
    with pytest.raises(ValueError, match="second point"):
        V.distance(c, p1)
    with pytest.raises(ValueError, match="outside"):
        V.pick(c, [W + 5, 10])
    # nothing decoded at the picked px: not measurable, never a guess
    hole = cloud(valid=lambda u, v: np.hypot(u, v) > 15)
    with pytest.raises(V.MeasureError, match="no decoded surface"):
        V.pick(hole, px_of(0, 0))


def _cylinder_cloud(radius=20.0, zc=480.0, noise=0.01, seed=0, half_len=40.0) -> MetricCloud:
    """A cylinder (axis along the camera's y) in front of the wall, by ray casting."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    dx, dy = (xx - W / 2) / F, (yy - H / 2) / F
    a, b, c0 = dx * dx + 1.0, -2.0 * zc, zc * zc - radius * radius
    disc = b * b - 4 * a * c0
    t = np.where(disc > 0, (-b - np.sqrt(np.maximum(disc, 0))) / (2 * a), Z0)
    t = np.where(np.abs(dy * t) < half_len, t, Z0)  # 80 mm long, then the wall
    t = t + rng.normal(0, noise, t.shape)
    xyz = np.stack([dx * t, dy * t, t], -1)
    return MetricCloud(xyz=xyz, mask=np.ones(t.shape, bool), consistency=None, method="plane",
                       geometry_id="synthetic", calibration_id="synthetic")


def test_cylinder_radius_axis_and_a_fixed_radius():
    c = _cylinder_cloud()
    out = V.cylinder(c)
    v = by_name(out)
    assert v["cylinder_radius"]["value"] == pytest.approx(20.0, abs=0.02)
    assert v["cylinder_diameter"]["value"] == pytest.approx(40.0, abs=0.04)
    assert v["cylinder_fit_rms"]["value"] < 0.02
    assert abs(out["details"]["axis"][1]) > 0.999          # along the camera's y
    assert v["cylinder_axis_tilt"]["value"] < 0.1 and out["details"]["arc_deg"] > 120
    assert V.cylinder(c, axis="y")["values"][0]["value"] == pytest.approx(20.0, abs=0.02)
    fixed = by_name(V.cylinder(c, radius_mm=20.0))          # fixed: no longer a measurement
    assert "cylinder_radius" not in fixed and fixed["cylinder_fit_rms"]["value"] < 0.02
    with pytest.raises(ValueError, match="axis"):
        V.cylinder(c, axis="w")
    # a narrow strip of it: the radius is weakly determined, and the value says so - and a
    # depth error counts many times in it (its uncertainty is that much wider)
    strip = V.cylinder(c, V.roi_mask((H, W), [185, 60, 215, 240]))
    assert any("weakly determined" in n for n in strip["values"][0]["notes"])
    assert strip["values"][0]["depth_gain"] > 10 * v["cylinder_radius"]["depth_gain"] >= 10
    assert strip["values"][1]["depth_gain"] == 2 * strip["values"][0]["depth_gain"]   # diameter
    assert v["cylinder_radius"]["noise_mm"] == pytest.approx(0.01, rel=0.5)


def test_a_cylinder_that_is_not_one_is_not_measured():
    """A cylinder fit always returns a radius. A flat face "fits" any huge one; two surfaces
    fitted as one give a wrong one: neither is a measurement."""
    for flat in (cloud(noise=0.01), cloud(box(-30, -12, 30, 12, 8.0))):   # a wall; a box's top
        with pytest.raises(V.MeasureError, match="flat within the noise") as e:
            V.cylinder(flat)
        assert e.value.code == "invalid" and "would be a guess" in str(e.value)
    # a roof (two planes): it bulges, but the points do not lie on a cylinder
    roof = cloud(lambda u, v: np.where(np.abs(u) < 30, 15.0 - 0.5 * np.abs(u), 0.0))
    with pytest.raises(V.MeasureError, match="do not lie on one cylinder") as e:
        V.cylinder(roof)
    assert e.value.code == "invalid"
    # ... nor on one of a radius it does not have
    with pytest.raises(V.MeasureError, match="do not lie on one cylinder"):
        V.cylinder(_cylinder_cloud(), radius_mm=35.0)
    # a cylinder that fills most of the view: the wall is no "background plane" any more, and
    # it must not be fitted together with it (that read 61 mm for this 45 mm one)
    big = _cylinder_cloud(radius=45.0, zc=455.0, half_len=100.0)
    assert float((big.xyz[..., 2] < Z0 - 5).mean()) > 0.7
    out = V.cylinder(big)
    r = by_name(out)["cylinder_radius"]
    assert r["value"] == pytest.approx(45.0, abs=0.05) and out["details"]["arc_deg"] > 120
    assert any("largest continuous surface" in n for n in r["notes"])
    assert by_name(out)["cylinder_fit_rms"]["value"] < 0.03


def test_bounding_dimensions_of_the_part_above_the_reference_plane():
    c = cloud(box(-30, -12, 30, 12, 8.0))
    out = V.dims(c)
    v = by_name(out)
    assert v["length"]["value"] == pytest.approx(60.0, abs=PX)
    assert v["width"]["value"] == pytest.approx(24.0, abs=PX)
    assert v["height"]["value"] == pytest.approx(8.0, abs=0.01)
    assert [v[k]["sensitivity"] for k in ("length", "width", "height")] == [
        "lateral", "lateral", "depth"]
    assert out["details"]["px_pitch_mm"] == pytest.approx(PX, rel=0.01)    # the px footprint
    assert len(out["box"]) == 4
    # turned 30 deg in the plane: still its own length x width (the minimum-area rectangle)
    a = math.radians(30.0)

    def turned(u, v):
        p, q = u * math.cos(a) + v * math.sin(a), -u * math.sin(a) + v * math.cos(a)
        return np.where((np.abs(p) <= 30) & (np.abs(q) <= 12), 8.0, 0.0)
    t = by_name(V.dims(cloud(turned)))
    assert t["length"]["value"] == pytest.approx(60.0, abs=2 * PX)
    assert t["width"]["value"] == pytest.approx(24.0, abs=2 * PX)
    with pytest.raises(V.MeasureError, match="no part found"):
        V.dims(cloud())                             # the empty wall: nothing stands on it
    with pytest.raises(V.MeasureError, match="no part found"):
        V.dims(c, min_height_mm=20.0)
    assert v["height"]["noise_mm"] == pytest.approx(0.01, rel=0.5)


def test_the_bounding_size_of_a_cut_part_is_not_reported():
    """A part that leaves the camera image, or that the roi cuts: what is seen of it has a
    bounding box, but that is not the part's length and width."""
    edge = (W / 2) / F * Z0                         # the frame's right edge, wall mm
    cut = cloud(box(edge - 40, -12, edge + 20, 12, 8.0))      # 20 mm of it are out of view
    with pytest.raises(V.MeasureError, match="cut by the image border") as e:
        V.dims(cut)
    assert e.value.code == "incomplete" and "right edge" in str(e.value)
    top = cloud(box(-30, -(H / 2) / F * Z0 - 5, 30, -20, 8.0))
    with pytest.raises(V.MeasureError, match="top edge"):
        V.dims(top)
    c = cloud(box(-30, -12, 30, 12, 8.0))
    with pytest.raises(V.MeasureError, match="the roi cuts the part") as e:
        V.dims(c, V.roi_mask((H, W), roi_of(0, -30, 60, 30)))             # half of it
    assert e.value.code == "incomplete"
    whole = by_name(V.dims(c, V.roi_mask((H, W), roi_of(-45, -25, 45, 25))))   # an roi that holds it
    assert whole["length"]["value"] == pytest.approx(60.0, abs=PX)
    assert whole["width"]["value"] == pytest.approx(24.0, abs=PX)


def test_deviation_values_come_from_the_map_and_never_from_an_unconverged_placement():
    rng = np.random.default_rng(0)
    dev = rng.normal(0.0, 0.02, (H, W))
    dev[100:110, 100:110] = -0.6                    # a dent
    dev[:20] = np.nan
    rep = {"icp": {"converged": True, "rms_mm": 0.02, "fitness": 1.0}, "golden": "g"}
    v = by_name(V.deviation(dev, rep))
    assert v["deviation_min"]["value"] == pytest.approx(-0.6) and v["deviation_max"]["value"] < 0.15
    assert v["deviation_rms"]["value"] == pytest.approx(0.02, abs=0.02)
    assert v["deviation_rms"]["n_points"] == (H - 20) * W
    with pytest.raises(V.MeasureError, match="did not converge") as e:
        V.deviation(dev, {"icp": {"converged": False}})
    assert e.value.code == "incomplete"
    with pytest.raises(V.MeasureError, match="could be compared"):
        V.deviation(np.full((H, W), np.nan), rep)


def test_a_patch_too_far_from_the_golden_is_not_silently_dropped():
    """compare_metric gives no deviation to a point further than 5 mm from the golden. A
    patch of such points (a boss, a wrong part) is where the largest deviation is: the
    extremes of the rest are not the part's."""
    rng = np.random.default_rng(1)
    dev = rng.normal(0.0, 0.02, (H, W))
    part = np.ones((H, W), bool)
    rep = {"icp": {"converged": True, "rms_mm": 0.02, "fitness": 1.0}, "golden": "g"}
    whole = V.deviation(dev, rep, part=part)
    assert all(x["notes"] == [] for x in whole["values"]) and whole["details"]["n_not_compared"] == 0
    far = dev.copy()
    far[100:112, 100:112] = np.nan                  # a 12 x 12 px boss, > 5 mm proud
    with pytest.raises(V.MeasureError, match="were not compared") as e:
        V.deviation(far, rep, part=part)
    assert e.value.code == "incomplete" and "144 of the part's" in str(e.value)
    assert "largest deviation is beyond 5 mm" in str(e.value)
    # an roi that does not hold the patch: its own numbers are whole
    assert V.deviation(far, rep, V.roi_mask((H, W), [200, 0, 400, 300]), part=part)["values"]
    # px that are not the part's (the wall behind it) are not missing
    assert V.deviation(far, rep, part=np.isfinite(far))["details"]["n_not_compared"] == 0
    # single stray px (edges): the numbers stand, and say how many are not in them
    stray = dev.copy()
    stray[50, 10:60:7] = np.nan
    out = V.deviation(stray, rep, part=part)
    assert out["details"]["n_not_compared"] == 8
    assert all(any("8 single points" in n for n in x["notes"]) for x in out["values"])


def test_a_picked_px_on_an_edge_is_not_one_point():
    c = cloud(box(-20, -15, 20, 15, 5.0))
    x_edge = px_of(-20, 0)[0]                       # the box's left edge
    with pytest.raises(V.MeasureError, match="is on an edge") as e:
        V.pick(c, [x_edge, H / 2])
    assert e.value.code == "invalid" and "at two depths" in str(e.value)
    with pytest.raises(V.MeasureError, match="is on an edge"):         # corner to corner
        V.distance(c, px_of(-20, -15), px_of(20, 15))
    X, n = V.pick(c, [x_edge + 6, H / 2])           # a few px inside the face: one surface
    assert X[2] == pytest.approx(Z0 - 5.0, abs=0.02) and n == 49
    steep = cloud(lambda u, v: math.tan(math.radians(50.0)) * u)       # a tilted face is no edge
    assert V.pick(steep, px_of(0, 0))[1] == 49
    # at the rim of the decoded surface the point is the median of what is decoded: up to
    # the pick radius off the picked px, and the distance's uncertainty says so
    half = cloud(valid=lambda u, v: u < 0.0)
    d = by_name(V.distance(half, px_of(-0.5, 0), px_of(-30, 0)))["distance"]
    assert d["lateral_gain"] == V.PICK_RADIUS_PX
    assert any("rim of the decoded surface" in n for n in d["notes"])
    inside = by_name(V.distance(half, px_of(-10, 0), px_of(-30, 0)))["distance"]
    assert "lateral_gain" not in inside and inside["notes"] == []


def test_the_noise_is_that_of_the_px_the_value_rests_on():
    """A dark part in a quiet scan: 15 x the wall's depth noise in one region. Its flatness
    was reported +- the whole scan's noise, and not flagged as being at the limit."""
    c = cloud(noise=0.01)
    rng = np.random.default_rng(5)
    c.xyz[100:200, 150:250, 2] += rng.normal(0.0, 0.15, (100, 100))
    noisy, quiet = V.roi_mask((H, W), [150, 100, 250, 200]), V.roi_mask((H, W), [10, 10, 110, 90])
    out = V.flatness(c, noisy)
    assert by_name(out)["flatness_rms"]["noise_mm"] == pytest.approx(0.15, rel=0.1)
    assert by_name(V.flatness(c, quiet))["flatness_rms"]["noise_mm"] == pytest.approx(0.01, rel=0.2)
    assert V.measured_noise(c) < 0.02 and V.region_noise(c, quiet, noisy) > 0.13
    assert by_name(V.step(c, quiet, noisy))["step_height"]["noise_mm"] > 0.13   # the noisier face
    ms = SimpleNamespace(cloud=c, meta={}, path=".")
    cal = SimpleNamespace(K_cam=np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1.0]]),
                          baseline_mm=200.0, K_proj=np.eye(3) * 1000.0)
    prec = MM.precision_of(ms, cal, None, out["z_mm"])
    assert prec["sigma_z_mm"] < 0.02                # the whole scan's: not what it rests on
    fin = {x["name"]: x for x in MM.finish_values([out["values"]], prec)}
    assert fin["flatness_rms"]["uncertainty"]["value"] == pytest.approx(0.15, rel=0.1)
    assert fin["flatness_rms"]["near_resolution_limit"] is True
    assert "of the measured px" in fin["flatness_rms"]["uncertainty"]["basis"]
    assert fin["plane_distance"]["uncertainty"]["value"] == pytest.approx(0.15, rel=0.1)


def test_the_uncertainty_of_repeats_has_a_floor():
    prec = {"sigma_z_mm": 0.01, "method": "measured_noise", "predicted_sigma_z_mm": None,
            "lateral_px_mm": 0.3, "z_mm": 500.0}

    def row(val, noise=0.02, sens="depth", **kw):
        return [{"name": "h", "label": "H", "value": val, "units": "mm", "n_points": 100,
                 "sensitivity": sens, "notes": [], "noise_mm": noise, **kw}]

    def unc(rows, pr=prec):
        return MM.finish_values(rows, pr)[0]["uncertainty"]

    # one scan: the value's own noise; the whole scan's only when its px can't tell theirs;
    # the prediction when that is larger
    assert unc([row(5.0, noise=0.15)])["value"] == pytest.approx(0.15)
    assert unc([row(5.0, noise=None)])["value"] == pytest.approx(0.01)
    u = unc([row(5.0)], {**prec, "predicted_sigma_z_mm": 0.05})
    assert u["value"] == pytest.approx(0.05) and u["method"] == "predicted_precision"
    assert MM.finish_values([row(0.3, noise=0.15)], prec)[0]["near_resolution_limit"] is True
    assert MM.finish_values([row(0.3, noise=0.02)], prec)[0]["near_resolution_limit"] is False
    # repeats that scatter more than one scan's noise: their spread
    u = unc([row(5.0), row(5.2), row(4.8)])
    assert u["method"] == "repeatability" and u["value"] == pytest.approx(0.2)
    assert u["floor"] == pytest.approx(0.02 / math.sqrt(3)) and u["n_repeats"] == 3
    # repeats that agree: never less than one scan resolves / sqrt(n) - and never 0
    same = MM.finish_values([row(5.0), row(5.0)], prec)[0]
    u = same["uncertainty"]
    assert u["method"] == "single_scan_noise/sqrt(n)" and u["repeatability"] == 0.0
    assert u["value"] == pytest.approx(0.02 / math.sqrt(2)) and u["value"] > 0
    assert any("closer than one scan resolves" in n for n in same["notes"])
    assert unc([row(5.0, noise=0.02), row(5.0, noise=0.2)])["value"] == pytest.approx(
        0.2 / math.sqrt(2))                         # the noisiest repeat speaks
    # ... and with nothing to say what one scan resolves, identical repeats are no measurement
    with pytest.raises(V.MeasureError, match="same value") as e:
        unc([row(5.0, noise=None), row(5.0, noise=None)],
            {**prec, "sigma_z_mm": None, "method": None})
    assert e.value.code == "invalid"
    # a lateral value keeps the px size whatever the repeats say
    u = unc([row(80.0, sens="lateral"), row(80.0, sens="lateral")])
    assert u["value"] == pytest.approx(0.3) and u["method"] == "repeatability+lateral_px_size"
    # a depth error that counts more than once in the value (the radius of a weak arc)
    g = MM.finish_values([row(20.0, sens="both", depth_gain=30.0)], prec)[0]
    assert g["uncertainty"]["value"] == pytest.approx(math.hypot(30.0 * 0.02, 0.3))
    assert g["depth_gain"] == 30.0 and "x 30.00" in g["uncertainty"]["basis"]


def test_too_few_points_and_the_measured_noise():
    c = cloud(noise=0.02)
    assert V.measured_noise(c) == pytest.approx(0.02, rel=0.05)
    assert V.measured_noise(cloud(box(-20, -15, 20, 15, 5.0), noise=0.02)) == pytest.approx(
        0.02, rel=0.05)                             # an edge is an outlier the median ignores
    assert V.measured_noise(cloud(valid=lambda u, v: u > 1e9)) is None
    with pytest.raises(V.MeasureError, match="too few") as e:
        V.flatness(c, V.roi_mask((H, W), [10, 10, 15, 15]))
    assert e.value.code == "too_few_points"
    with pytest.raises(ValueError, match="outside"):
        V.roi_mask((H, W), [500, 10, 600, 20])


# ================================================================== the spec

def test_spec_lists_every_problem_and_round_trips():
    sp = MM.spec_from({"what": "step", "roi": [0, 0, 10, 10], "roi2": [20, 0, 30, 10],
                       "repeats": 3, "scan": {"kind": "fast"}})
    again = MM.spec_from(json.loads(json.dumps(sp.to_dict())))
    assert again == sp and sp.to_dict()["kind"] == MM.SPEC_KIND
    with pytest.raises(ValueError) as e:
        MM.spec_from({"what": "step", "roi": [0, 0, 10, 10], "colour": "red", "repeats": 0,
                      "p1": [1, 2], "scan": {"kind": "slow"}})
    msg = str(e.value)
    for needle in ("unknown keys ['colour']", "step needs two regions", "repeats must be",
                   "p1 is for distance", "scan.kind"):
        assert needle in msg, (needle, msg)
    for bad, needle in (({"what": "teleport"}, "is not one of"),
                        ({"what": "distance"}, "needs p1"),
                        ({"what": "distance", "p1": [1, 2]}, "needs p2"),
                        ({"what": "deviation"}, "needs golden"),
                        ({"what": "endface"}, "face_diameter_mm"),
                        ({"what": "flatness", "roi": [5, 5, 5, 9]}, "roi must be"),
                        ({"what": "cylinder", "radius_mm": float("nan")}, "radius_mm"),
                        ({"what": "flatness", "roi": [0, 0, True, 4]}, "roi must be")):
        with pytest.raises(ValueError, match=needle):
            MM.spec_from(bad)
    assert MM.spec_from({"what": "distance", "p1": [3, 4], "roi": [0, 0, 9, 9]}).p2 is None
    assert MM.spec_from({"what": "DIMS"}).what == "dims"


# ================================================================== engine: synthetic scans

@pytest.fixture(scope="module")
def lab(tmp_path_factory):
    """A mock library with the bench's TRUE stereo calibration and synthetic metric scans of
    known scenes (nothing is captured)."""
    tmp = tmp_path_factory.mktemp("measure")
    s = api.configure(tmp / "library", mock=True, bench=MockBench(seed=1))
    p = api.load_preset(PRESET)
    s.bench.use_stereo()
    RS.truth_calibration(s, p)
    blk = [Block(center=(0.0, 0.0, 813.0), half=(40.0, 25.0, 10.0))]       # 80 x 50 x 20 mm
    out = {"s": s, "p": p, "tmp": tmp, "runs": tmp / "runs",
           "wall": RS.oracle_scan(s, p, [], "wall"),
           "box": RS.oracle_scan(s, p, blk, "box", seed=1),
           "box_b": RS.oracle_scan(s, p, blk, "box_b", seed=2),
           "box_c": RS.oracle_scan(s, p, blk, "box_c", seed=3),
           "cyl": RS.oracle_scan(s, p, [MetricCylinder(center=(0, 0, 799), radius=24.0,
                                                       height=90)], "cyl", seed=4),
           "faces": RS.oracle_scan(s, p, "scene:endfaces", "faces", seed=5),
           "plate": RS.oracle_scan(s, p, "scene:plate", "plate", seed=6),
           "bump": RS.oracle_scan(s, p, "scene:plate:bump", "bump", seed=7),
           "ajar": RS.oracle_scan(s, p, "scene:box_lid:ajar", "ajar", seed=8)}
    m = out["box"].cloud.mask & (out["box"].cloud.xyz[..., 2] < 810)
    ys, xs = np.nonzero(m)
    out["top"] = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    yield out
    api._session = None


def _use(lab):
    api._session = lab["s"]
    return lab["p"]


def _regions(lab) -> tuple:
    """(a patch of the wall left of the box, a patch of its top face, the top's centre px)."""
    x0, y0, x1, y1 = lab["top"]
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    return ([x0 - 150, cy - 30, x0 - 60, cy + 30], [cx - 40, cy - 30, cx + 40, cy + 30],
            (cx, cy))


def _measure(lab, what, scan="wall", **kw) -> dict:
    p = _use(lab)
    scans = [lab[k] for k in ([scan] if isinstance(scan, str) else scan)]
    return api.measure_run(what, p, scans=scans, runs_dir=lab["runs"], **kw)


def _values(out) -> dict:
    assert out["result"]["status"] == "MEASURED", out["result"]["validity"]
    return {v["name"]: v for v in out["result"]["values"]}


def _never_a_verdict(doc: dict) -> None:
    """A measurement decides nothing, on every path."""
    assert doc["operating_mode"] == "measure" and doc["schema_version"] == MM.MEASURE_SCHEMA
    assert doc["result"] is None and doc["disposition"] == D.NO_DISPOSITION == "NONE"
    assert "pull_reasons" not in doc and "permissions" not in doc and "steps" not in doc
    assert doc["disposition"] not in D.DISPOSITIONS
    for v in doc["values"]:
        assert v["units"] in ("mm", "deg") and isinstance(v["n_points"], int) and v["n_points"] > 0
        u = v["uncertainty"]
        assert u["value"] is not None and u["value"] >= 0 and u["units"] == v["units"]
        assert u["method"] and isinstance(v["near_resolution_limit"], bool)
        assert math.isfinite(v["value"])


def test_flatness_run_writes_the_whole_run_folder(lab):
    out = _measure(lab, "flatness", roi=[500, 400, 900, 700])
    doc, d = out["result"], out["run_dir"]
    assert out["exit_code"] == 0 and doc["status"] == "MEASURED" and doc["reason"] is None
    _never_a_verdict(doc)
    v = _values(out)
    assert v["flatness_rms"]["value"] == pytest.approx(0.02, abs=0.004)       # the scan's noise
    assert v["flatness_rms"]["n_points"] == 400 * 300
    assert v["tilt"]["value"] < 0.05 and v["plane_distance"]["value"] == pytest.approx(823.0, abs=0.05)
    # a flat wall reads as its own noise: the value is the instrument's, and says so
    assert v["flatness_rms"]["near_resolution_limit"] is True
    assert "near the resolution limit" in v["flatness_rms"]["notes"][0]
    assert v["plane_distance"]["near_resolution_limit"] is False
    # a synthetic scan records no fringe levels: the uncertainty is its own measured noise
    prec = doc["precision"]
    assert prec["predicted_sigma_z_mm"] is None and prec["method"] == "measured_noise"
    assert prec["sigma_z_mm"] == pytest.approx(0.02, rel=0.1)
    assert prec["lateral_px_mm"] == pytest.approx(823.0 / 2450.0, rel=0.01)
    # ... of the px the value rests on (its roi), not the whole scan's
    assert v["flatness_rms"]["noise_mm"] == pytest.approx(0.02, rel=0.1)
    assert v["flatness_rms"]["uncertainty"]["value"] == pytest.approx(v["flatness_rms"]["noise_mm"])
    assert v["flatness_rms"]["uncertainty"]["method"] == "measured_noise"
    assert v["tilt"]["uncertainty"]["units"] == "deg"
    assert "MEASURED" in doc["validity"] and doc["calibration"]["calibration_id"] in doc["validity"]
    assert doc["preset"] == {"name": PRESET, "geometry_id": lab["p"].geometry_id,
                             "mount": "fixed"}
    # the run folder: result.json, values.csv, overlay.png, scan.ply, spec.json, the log
    assert sorted(f.name for f in d.iterdir()) == ["overlay.png", "result.json",
                                                   "run.log.jsonl", "scan.ply", "spec.json",
                                                   "values.csv"]
    assert json.loads((d / "result.json").read_text(encoding="utf-8")) == json.loads(
        json.dumps(doc))
    rows = list(csv.DictReader(io.StringIO((d / "values.csv").read_text(encoding="utf-8"))))
    assert [r["name"] for r in rows] == [x["name"] for x in doc["values"]]
    assert rows[0]["units"] == "mm" and float(rows[0]["uncertainty"]) > 0
    assert rows[0]["near_resolution_limit"] == "yes" and rows[0]["n_points"] == "120000"
    import cv2
    img = cv2.imdecode(np.frombuffer((d / "overlay.png").read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    assert img.shape == (1080, 1440, 3)             # 1 image px = 1 camera px (the page's clicks)
    ply = (d / "scan.ply").read_bytes()
    assert ply.startswith(b"ply\n") and b"comment units mm" in ply[:200]
    assert f"element vertex {lab['wall'].meta['n_points']}".encode() in ply[:300]
    spec = json.loads((d / "spec.json").read_text(encoding="utf-8"))
    assert spec["what"] == "flatness" and spec["preset"] == PRESET and spec["roi"] == [500, 400, 900, 700]
    events = [json.loads(x) for x in (d / "run.log.jsonl").read_text().splitlines()]
    assert events[0]["event"] == "run_start" and events[-1]["event"] == "run_end"
    assert events[-1]["status"] == "MEASURED" and events[-1]["disposition"] == "NONE"
    assert all(e["run_id"] == doc["run_id"] for e in events)
    assert "offline" in doc["warnings"][-1]


def test_tilt_of_a_lid_tilted_three_degrees(lab):
    c = lab["ajar"].cloud
    m = c.mask & (c.xyz[..., 2] < 790)              # the lid, ~40 mm proud of the wall
    ys, xs = np.nonzero(m)
    cx, cy = int(np.median(xs)), int(np.median(ys))
    v = _values(_measure(lab, "flatness", "ajar", roi=[cx - 60, cy - 40, cx + 60, cy + 40]))
    assert v["tilt"]["value"] == pytest.approx(3.0, abs=0.1)
    assert v["flatness_rms"]["value"] < 0.03 and v["tilt"]["uncertainty"]["value"] < 0.2


def test_step_dims_and_distance_on_the_block(lab):
    wall_roi, top_roi, (cx, cy) = _regions(lab)
    s = _values(_measure(lab, "step", "box", roi=wall_roi, roi2=top_roi))
    assert s["step_height"]["value"] == pytest.approx(20.0, abs=0.02)
    assert s["step_height"]["uncertainty"]["value"] == pytest.approx(0.02, rel=0.15)
    assert s["step_height"]["near_resolution_limit"] is False
    out = _measure(lab, "dims", "box")
    v = _values(out)
    px = out["result"]["precision"]["lateral_px_mm"]
    assert px == pytest.approx(803.0 / 2450.0, rel=0.01)
    assert v["length"]["value"] == pytest.approx(80.0, abs=px + 0.02)
    assert v["width"]["value"] == pytest.approx(50.0, abs=px + 0.02)
    assert v["height"]["value"] == pytest.approx(20.0, abs=0.03)
    assert v["length"]["uncertainty"] == {**v["length"]["uncertainty"], "value": pytest.approx(px),
                                          "method": "lateral_px_size", "units": "mm"}
    assert v["height"]["uncertainty"]["method"] == "measured_noise"
    assert not any(x["near_resolution_limit"] for x in v.values())
    d = _values(_measure(lab, "distance", "box", p1=[cx - 60, cy], p2=[cx + 60, cy],
                         roi=wall_roi))
    assert d["distance"]["value"] == pytest.approx(120 * 803.0 / 2450.0, abs=0.35)
    assert d["distance"]["uncertainty"]["method"] == "measured_noise+lateral_px_size"
    assert d["point_to_plane"]["value"] == pytest.approx(20.0, abs=0.03)
    # two patches of one wall: a step of nothing, within what the scan resolves
    z = _values(_measure(lab, "step", roi=[300, 400, 500, 600], roi2=[900, 400, 1100, 600]))
    assert abs(z["step_height"]["value"]) < 0.01 and z["step_height"]["near_resolution_limit"]


def test_cylinder_and_end_face_diameters(lab):
    out = _measure(lab, "cylinder", "cyl")
    v = _values(out)
    assert v["cylinder_radius"]["value"] == pytest.approx(24.0, abs=0.02)
    assert v["cylinder_diameter"]["value"] == pytest.approx(48.0, abs=0.04)
    assert v["cylinder_diameter"]["units"] == "mm" and v["cylinder_radius"]["n_points"] > 10000
    assert abs(out["result"]["details"]["axis"][1]) > 0.999 and out["result"]["details"]["arc_deg"] > 100
    f = _measure(lab, "endface", "faces", face_diameter_mm=21.0, n_faces=2)
    fv = _values(f)
    px = f["result"]["precision"]["lateral_px_mm"]
    for name in ("c1", "c2"):                       # the scene's end faces: 21.0 mm across
        assert fv[f"{name}_diameter"]["value"] == pytest.approx(21.0, abs=px)
        assert fv[f"{name}_diameter"]["uncertainty"]["method"] == "lateral_px_size"
        assert fv[f"{name}_flatness_rms"]["value"] < 0.05
    assert all(x["noise_mm"] is not None for x in fv.values())  # the noise inside each rim
    # the wall shows no end face: not measurable, with the reason
    none = _measure(lab, "endface", "wall", face_diameter_mm=21.0)
    assert none["result"]["status"] == "NOT_MEASURABLE" and none["exit_code"] == 5
    assert none["result"]["reason"] == "incomplete" and none["result"]["values"] == []
    # ... and neither the wall nor a box shows a cylinder: no radius of 3.9e6 mm +- 0.3
    for scan in ("wall", "box"):
        flat = _measure(lab, "cylinder", scan)
        assert flat["exit_code"] == 5 and flat["result"]["status"] == "NOT_MEASURABLE"
        assert flat["result"]["reason"] == "invalid" and flat["result"]["values"] == []
        assert "flat within the noise" in flat["result"]["message"]


def test_deviation_vs_a_golden_scan_gives_values_only(lab):
    same = _measure(lab, "deviation", "box_b", golden=lab["box"].id)
    v = _values(same)
    _never_a_verdict(same["result"])
    assert v["deviation_rms"]["value"] == pytest.approx(0.028, abs=0.012)     # two scans' noise
    assert abs(v["deviation_mean"]["value"]) < 0.01 and v["deviation_max"]["value"] < 0.25
    bump = _values(_measure(lab, "deviation", "bump", golden="plate"))        # by its label
    assert bump["deviation_max"]["value"] == pytest.approx(0.8, abs=0.12)     # the 0.8 mm bump
    assert bump["deviation_max"]["near_resolution_limit"] is False
    assert (same["run_dir"] / "overlay.png").is_file()
    for golden, needle in (("no_such_scan", "no metric scan"), (lab["box_b"].id, "itself")):
        bad = _measure(lab, "deviation", "box_b", golden=golden)
        assert bad["result"]["status"] == "NOT_MEASURABLE" and bad["result"]["reason"] == "golden"
        assert needle in bad["result"]["message"] and bad["exit_code"] == 5
    assert same["result"]["details"]["n_not_compared"] == 0
    # the block with an 8 mm boss on it against the plain block: the boss is further than
    # 5 mm from the golden, so it has no deviation - "deviation_max 0.12 mm" would be a lie
    p = _use(lab)
    boss = RS.oracle_scan(lab["s"], p, [Block(center=(0.0, 0.0, 813.0), half=(40.0, 25.0, 10.0)),
                                        Block(center=(10.0, 5.0, 799.0), half=(6.0, 6.0, 4.0))],
                          "boss", seed=2)
    out = api.measure_run("deviation", p, scans=[boss], golden=lab["box"].id,
                          runs_dir=lab["runs"])
    doc = out["result"]
    assert out["exit_code"] == 5 and doc["status"] == "NOT_MEASURABLE" and doc["values"] == []
    assert doc["reason"] == "incomplete" and "were not compared" in doc["message"]
    _never_a_verdict(doc)


def test_repeats_give_the_repeatability(lab):
    out = _measure(lab, "dims", ["box", "box_b", "box_c"])
    doc = out["result"]
    v = _values(out)
    assert doc["repeats"] == 3 and len(doc["scans"]) == 3
    h = v["height"]
    u = h["uncertainty"]
    assert u["n_repeats"] == 3 and len(h["repeat_values"]) == 3
    assert h["value"] == pytest.approx(np.mean(h["repeat_values"]))
    assert u["repeatability"] == pytest.approx(np.std(h["repeat_values"], ddof=1), abs=2e-6)
    # three scans that agree to a few um are not a measurement of the spread: the
    # uncertainty is never below what one scan resolves / sqrt(3)
    assert u["repeatability"] < 0.01 and u["single_scan"] > 0.015
    assert u["method"] == "single_scan_noise/sqrt(n)"
    assert u["value"] == u["floor"] == pytest.approx(u["single_scan"] / math.sqrt(3), abs=2e-6)
    assert any("closer than one scan resolves" in n for n in h["notes"])
    # a lateral value keeps the px size: repeats of a part that did not move can't average it
    w = v["width"]["uncertainty"]
    assert w["method"] == "repeatability+lateral_px_size"
    assert w["value"] >= doc["precision"]["lateral_px_mm"] > 10 * w["repeatability"]
    assert "repeatability" in doc["validity"]
    rows = list(csv.DictReader(io.StringIO((out["run_dir"] / "values.csv").read_text())))
    assert {r["n_repeats"] for r in rows} == {"3"}
    # the same scan twice is not a repeat: refused - never "+- 0.0000 repeatability x2"
    twice = _measure(lab, "flatness", ["wall", "wall"], roi=[500, 400, 900, 700])
    assert twice["exit_code"] == 2 and twice["result"]["reason"] == "refused"
    assert twice["result"]["status"] == "NOT_MEASURABLE" and twice["result"]["values"] == []
    assert "named more than once" in twice["result"]["message"]
    code, text = _cli(["--mock", "--library", str(lab["s"].store.root), "measure-run", PRESET,
                       "--what", "flatness", "--scan", "latest", "latest", "--runs-dir",
                       str(lab["runs"])])
    assert code == 2 and "+-" not in text
    api._session = lab["s"]


def test_not_measurable_never_gives_a_number(lab, tmp_path):
    # no part where one is asked for
    none = _measure(lab, "dims", "wall")
    doc = none["result"]
    assert none["exit_code"] == 5 and doc["status"] == "NOT_MEASURABLE"
    assert doc["reason"] == "too_few_points" and "no part found" in doc["message"]
    assert doc["values"] == [] and doc["how_to_fix"] and "NOT MEASURABLE" in doc["validity"]
    _never_a_verdict(doc)
    d = none["run_dir"]
    assert sorted(f.name for f in d.iterdir()) == ["result.json", "run.log.jsonl", "spec.json",
                                                   "values.csv"]       # no picture, no cloud
    assert (d / "values.csv").read_text().strip() == ",".join(MM.CSV_FIELDS)
    assert doc["outputs"]["overlay"] is None and doc["outputs"]["ply"] is None
    # a picked px in the projector's shadow beside the block: nothing decoded there
    from scipy import ndimage
    x0, y0, x1, y1 = lab["top"]
    dark = np.argwhere(~ndimage.binary_dilation(lab["box"].cloud.mask, iterations=5))
    assert len(dark) > 20
    yy, xx = dark[len(dark) // 2]
    gone = _measure(lab, "distance", "box", p1=[int(xx), int(yy)], p2=[x0 + 5, y0 + 5])
    assert gone["result"]["status"] == "NOT_MEASURABLE" and gone["exit_code"] == 5
    assert "no decoded surface" in gone["result"]["message"] and gone["result"]["values"] == []
    # an roi off the image: refused (bad input), still no number
    off = _measure(lab, "flatness", roi=[5000, 10, 5100, 20])
    assert off["exit_code"] == 2 and off["result"]["reason"] == "refused"
    assert off["result"]["status"] == "NOT_MEASURABLE" and off["result"]["values"] == []
    # a grid zone on a stored full-field scan, with no projector map to cut it by: the
    # "region" would be the whole scan (that read flatness 20 mm on the block's scan)
    for kw in ({"zone": "A1:B2"}, {"zone": "H8:I11", "roi": [500, 400, 900, 700]}):
        zoned = _measure(lab, "flatness", "box", **kw)
        assert zoned["exit_code"] == 2 and zoned["result"]["reason"] == "refused", kw
        assert "can't cut this stored scan" in zoned["result"]["message"]
        assert zoned["result"]["values"] == []
    # a picked px on the block's edge: two surfaces, not one point
    edge = _measure(lab, "distance", "box", p1=[x0, (y0 + y1) // 2], p2=[x1, (y0 + y1) // 2])
    assert edge["exit_code"] == 5 and edge["result"]["reason"] == "invalid"
    assert "is on an edge" in edge["result"]["message"] and edge["result"]["values"] == []
    # no stereo calibration at all: uncalibrated, with the commands that fix it
    s = api.configure(tmp_path / "lib", mock=True, bench=MockBench(seed=2))
    try:
        p = api.load_preset(PRESET)
        un = api.measure_run("flatness", p, runs_dir=tmp_path / "runs")
        doc = un["result"]
        assert un["exit_code"] == 5 and doc["status"] == "NOT_MEASURABLE"
        assert doc["reason"] == "uncalibrated" and "never" in doc["message"]
        assert "calibrate-stereo" in doc["how_to_fix"] and PRESET in doc["how_to_fix"]
        assert doc["values"] == [] and doc["scans"] == [] and doc["calibration"]["calibration_id"] is None
        _never_a_verdict(doc)
        assert not list((s.store.root / "metric").glob("*/*"))        # nothing was scanned
    finally:
        api._session = None


def test_stale_calibration_and_a_moved_rig_are_not_measurable(tmp_path):
    """A stored scan made with a calibration that is no longer the one in use; a FIXED rig
    whose newest rig check said MOVED: no number until it is put right."""
    from engine.calibrate import stereo
    s = api.configure(tmp_path / "lib", mock=True, bench=MockBench(seed=3))
    p = api.load_preset(PRESET)
    s.bench.use_stereo()
    cal = RS.truth_calibration(s, p)
    old = RS.oracle_scan(s, p, [], "old")
    runs = tmp_path / "runs"
    assert api.measure_run("flatness", p, scans=[old], runs_dir=runs)["result"]["status"] == "MEASURED"
    # recalibrated since: the stored scan's mm belong to the calibration it was made with
    d = s.store.new_dir("calibrations", p.geometry_id, label="stereo")
    new = stereo.StereoCalibration.from_json({**cal.to_json(), "id": f"{p.geometry_id}/{d.name}"}, d)
    s.store.write_json(d / "stereo_calibration.json", new.to_json())
    stale = api.measure_run("flatness", p, scans=[old], runs_dir=runs)
    assert stale["exit_code"] == 5 and stale["result"]["status"] == "NOT_MEASURABLE"
    assert stale["result"]["reason"] == "stale_calibration" and old.id in stale["result"]["message"]
    assert "scan again" in stale["result"]["how_to_fix"] and stale["result"]["values"] == []
    # a fixed rig: trusted, until a rig check says MOVED
    fresh = RS.oracle_scan(s, p, [], "fresh")
    assert api.rig_check(p)["status"] == "REFERENCE_SET"
    assert api.measure_run("flatness", p, scans=[fresh], runs_dir=runs)["result"]["status"] == "MEASURED"
    assert api.rig_check(p, mock_shift_px=[2.5, 0.0])["status"] == "MOVED"
    moved = api.measure_run("flatness", p, scans=[fresh], runs_dir=runs)
    doc = moved["result"]
    assert moved["exit_code"] == 5 and doc["status"] == "NOT_MEASURABLE" and doc["reason"] == "rig_check"
    assert "MOVED" in doc["message"] and "rig-check" in doc["how_to_fix"] and doc["values"] == []
    assert doc["rig"]["objection"]["status"] == "MOVED" and doc["rig"]["mount"] == "fixed"
    assert api.rig_state(p)["ok"] is False and api.rig_state(p, strict=False)["ok"] is True
    during = RS.oracle_scan(s, p, [], "during")     # scanned while the objection stood
    assert api.rig_check(p)["status"] == "OK"       # put right: measurable again
    assert api.rig_state(p)["ok"] is True
    again = RS.oracle_scan(s, p, [], "again")
    assert api.measure_run("flatness", p, scans=[again], runs_dir=runs)["result"]["status"] == "MEASURED"
    # ... but a scan made while the rig check objected, or just before it did (the rig moved
    # at some time before that check), stays in doubt after the OK: scan again
    for scan, needle in ((during, "stood"), (fresh, "the next rig check")):
        doc = api.measure_run("flatness", p, scans=[scan], runs_dir=runs)["result"]
        assert doc["status"] == "NOT_MEASURABLE" and doc["reason"] == "rig_check", doc["message"]
        assert "in doubt" in doc["message"] and needle in doc["message"] and doc["values"] == []


# ================================================================== never a verdict

def test_a_measurement_is_never_a_continue_a_pass_or_a_permission(lab, tmp_path):
    ok = _measure(lab, "flatness")["result"]
    bad = _measure(lab, "dims", "wall")["result"]
    for doc in (ok, bad):
        _never_a_verdict(doc)
        assert D.of_result(doc["result"], doc["disposition"]) == "PULL"
        assert D.exit_code(doc) == 5                # whatever reads it as a run: never 0
        assert D.ensure(dict(doc, pull_reasons=[]))["disposition"] == "PULL"
        with pytest.raises(ValueError):             # it is not a run result: never written as one
            R.write_result(tmp_path, dict(doc))
        # a work state "verified" by it is a DOUBT: no permission
        c = FC.classify(FS.Step(id="s", permit="GO"), {}, doc)
        assert c["outcome"] == "DOUBT"
        # the command bus, handed it as the result of a run: never PASS, never a permission
        for result in (doc["result"], doc["status"], "PASS"):
            bus = C.CommandBus(configs=["a.json"])
            out = C.SimulatedOutput()
            bus.outputs.append(out)
            assert bus.command("START").accepted and bus.state == "RUNNING"
            bus.job_finished("run", result, disposition=doc["disposition"], step=doc)
            assert bus.state == "READY" and not out.current["PASS"]
            assert bus.last_disposition == "PULL" and out.current["PULL"]
            assert not any(v for k, v in out.current.items() if k.startswith("PERMIT_"))
            assert bus.permissions() == {} and out.registers["result_code"] != C.RESULT_CODES["PASS"]
    assert MM.exit_code(ok) == 0 and MM.exit_code(bad) == 5
    assert MM.exit_code({"status": "PASS"}) == 5 and MM.exit_code({}) == 5


# ================================================================== spec -> check

def test_a_spec_is_promoted_to_a_check_by_adding_tolerances(lab):
    p = _use(lab)
    wall_roi, top_roi, _ = _regions(lab)
    run = _measure(lab, "step", "box", roi=wall_roi, roi2=top_roi)
    spec = run["run_dir"] / "spec.json"             # the spec a run leaves behind
    good = api.measure_to_check(spec, {"step_height": [19.5, 20.5]}, name="step_ok",
                                guard_band_pct=10)
    assert good["kind"] == "recipe" and good["document"]["preset"] == PRESET
    zones = {z["id"]: z for z in good["document"]["zones"]}
    assert zones["a"]["datum"] is True and zones["b"]["checks"][0]["type"] == "height"
    assert zones["b"]["checks"][0]["params"]["band_mm"] == [19.5, 20.5]
    assert good["document"]["pull"]["guard_band_pct"] == 10
    # the engine's own check judges it: the same block passes its band, fails another
    res = api.recipe_run(good["document"], p, scan=lab["box"], project_feedback=False,
                         runs_dir=lab["runs"])["result"]
    assert res["result"] == "PASS" and res["disposition"] == "CONTINUE"
    h = res["zones"][1]["checks"][0]["values"]["height_mm"]
    assert h == pytest.approx(_values(run)["step_height"]["value"], abs=0.02)
    bad = api.measure_to_check(spec, {"step_height": [25.0, 26.0]})
    res = api.recipe_run(bad["document"], p, scan=lab["box"], project_feedback=False,
                         runs_dir=lab["runs"])["result"]
    assert res["result"] == "FAIL" and res["disposition"] == "REJECT"
    # flatness -> the recipe flatness check on the roi (the same statistic: p2-p98)
    flat = api.measure_to_check({"what": "flatness", "preset": PRESET, "roi": top_roi},
                                {"flatness_p2_p98": 0.3, "flatness_rms": 0.1})
    chk = flat["document"]["zones"][0]["checks"][0]
    assert chk == {"type": "flatness", "params": {"tol_mm": 0.3, "trim_pct": 2.0,
                                                  "rms_tol_mm": 0.1}}
    assert flat["document"]["zones"][0]["coords"] == [top_roi[0], top_roi[1], top_roi[2] - 1,
                                                      top_roi[3] - 1]
    res = api.recipe_run(flat["document"], p, scan=lab["box"], project_feedback=False,
                         runs_dir=lab["runs"])["result"]
    assert res["result"] == "PASS", res["invalid_reason"]
    # deviation -> a single-job station config (the quick check's golden compare)
    dev = api.measure_to_check({"what": "deviation", "preset": PRESET, "golden": "plate"},
                               {"deviation_max": 0.5})
    assert dev["kind"] == "station_config" and dev["document"]["job"]["measure"] == "golden"
    assert dev["document"]["threshold_mm"] == 0.5 and dev["document"]["job"]["golden"] == "plate"
    for spec_, tol, needle in (
            ({"what": "dims", "preset": PRESET}, {"length": [1, 2]}, "no one-to-one check"),
            ({"what": "flatness", "preset": PRESET}, {"flatness_p2_p98": 0.3}, "must be an roi"),
            ({"what": "flatness", "roi": top_roi}, {"flatness_p2_p98": 0.3}, "names no preset"),
            ({"what": "flatness", "preset": PRESET, "roi": top_roi}, {"tilt": 1}, "no check for"),
            ({"what": "flatness", "preset": PRESET, "roi": top_roi}, {}, "at least one tolerance"),
            ({"what": "step", "preset": PRESET, "roi": wall_roi, "roi2": top_roi},
             {"step_height": 20.0}, "as a band")):
        with pytest.raises(ValueError, match=needle):
            api.measure_to_check(spec_, tol)


# ================================================================== CLI

def _cli(argv) -> tuple:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = cli.main(argv)
    return code, buf.getvalue()


def test_cli_exit_codes_measured_0_not_measurable_5_refused_2(lab, tmp_path):
    lib = str(lab["s"].store.root)
    base = ["--mock", "--library", lib]
    runs = tmp_path / "runs"
    spec = tmp_path / "flat.json"
    code, out = _cli([*base, "measure-run", PRESET, "--what", "flatness", "--scan",
                      lab["wall"].id, "--roi", "500,400,900,700", "--runs-dir", str(runs),
                      "--save-spec", str(spec)])
    assert code == 0 and "MEASURED" in out and "Flatness (RMS)" in out and "+-" in out
    assert "validity: MEASURED" in out and "values.csv" in out and "NEAR LIMIT" in out
    saved = json.loads(spec.read_text())
    assert saved["roi"] == [500, 400, 900, 700] and saved["preset"] == PRESET
    # the saved spec runs again as it is (its preset too), --json for scripts
    code, out = _cli([*base, "--json", "measure-run", "--spec", str(spec), "--scan",
                      lab["wall"].id, "--runs-dir", str(runs)])
    doc = json.loads(out)
    assert code == 0 and doc["ok"] and doc["status"] == "MEASURED" and doc["preset"] == PRESET
    assert doc["result"]["disposition"] == "NONE" and "disposition" not in doc
    assert {v["name"] for v in doc["values"]} >= {"flatness_rms", "tilt"}
    # not measurable: 5, the reason and how to fix it, no table
    code, out = _cli([*base, "measure-run", PRESET, "--what", "dims", "--scan", lab["wall"].id,
                      "--runs-dir", str(runs)])
    assert code == 5 and "NOT MEASURABLE (too_few_points)" in out and "fix:" in out
    assert "+-" not in out
    code, out = _cli(["--mock", "--library", str(tmp_path / "empty"), "--json", "measure-run",
                      PRESET, "--what", "flatness", "--runs-dir", str(runs)])
    doc = json.loads(out)
    assert code == 5 and not doc["ok"] and doc["reason"] == "uncalibrated" and doc["values"] == []
    # bad input: refused (2) before anything runs - no run folder
    n = len(list(runs.iterdir()))
    for argv in (["--what", "step", "--roi", "1,2,30,40"], ["--what", "distance"],
                 ["--what", "flatness", "--roi", "1,2,3"], []):
        code, _ = _cli([*base, "measure-run", PRESET, *argv, "--runs-dir", str(runs)])
        assert code == cli.EXIT_REFUSED, argv
    assert len(list(runs.iterdir())) == n
    # measure-to-check writes the promoted check, never over a file
    wall_roi, top_roi, _ = _regions(lab)
    step = tmp_path / "step.json"
    step.write_text(json.dumps({"what": "step", "preset": PRESET, "roi": wall_roi,
                                "roi2": top_roi}))
    out_file = tmp_path / "step_check.json"
    code, out = _cli([*base, "measure-to-check", str(step), "--tol", "step_height=19.5,20.5",
                      "--out", str(out_file)])
    assert code == 0 and "quick-check" in out
    assert json.loads(out_file.read_text())["zones"][1]["checks"][0]["params"]["band_mm"] == [19.5, 20.5]
    assert _cli([*base, "measure-to-check", str(step), "--tol", "step_height=19.5,20.5",
                 "--out", str(out_file)])[0] != 0
    assert _cli([*base, "measure-to-check", str(step), "--tol", "nonsense"])[0] == cli.EXIT_REFUSED


def test_measure_through_the_full_mock_capture_path(tmp_path):
    """One live measurement (the mock bench, camera noise, the normal fast-scan path): the
    predicted precision at that distance is there and in line with the scan's own noise."""
    lib = tmp_path / "lib"
    s = api.configure(lib, mock=True)
    RS.truth_calibration(s, api.load_preset(PRESET))
    api._session = None
    code, out = _cli(["--mock", "--library", str(lib), "--mock-part", "80,50,20", "--json",
                      "measure-run", PRESET, "--what", "dims", "--scan-kind", "fast",
                      "--runs-dir", str(tmp_path / "runs")])
    doc = json.loads(out)
    assert code == 0 and doc["status"] == "MEASURED", doc
    v = {x["name"]: x for x in doc["values"]}
    prec = doc["result"]["precision"]
    assert 0.005 < prec["predicted_sigma_z_mm"] < 0.1 and prec["measured_noise_mm"] > 0
    assert 0.3 < prec["predicted_sigma_z_mm"] / prec["measured_noise_mm"] < 3.0
    assert prec["sigma_z_mm"] == max(prec["predicted_sigma_z_mm"], prec["measured_noise_mm"])
    px = prec["lateral_px_mm"]
    assert v["length"]["value"] == pytest.approx(80.0, abs=2 * px)
    assert v["width"]["value"] == pytest.approx(50.0, abs=2 * px)
    assert v["height"]["value"] == pytest.approx(20.0, abs=0.1)
    assert v["height"]["uncertainty"]["method"] in ("predicted_precision", "measured_noise")
    run = next((tmp_path / "runs").iterdir())
    assert (run / "scan.ply").is_file() and (run / "overlay.png").is_file()
    assert doc["result"]["scans"] and config.MOCK_PREFIX in doc["result"]["preset"]["geometry_id"]

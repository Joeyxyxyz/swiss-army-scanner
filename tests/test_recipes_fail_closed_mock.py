"""Recipes fail closed, MOCK only: the golden is never the scan being checked (nor an earlier
run's test scan, nor a scan from other rig settings), an unmeasured zone is never "absent" or
"clean" in any zone unit, edge lines compare undirected and every side asked for is measured,
the run verdict passes only on exact PASSes, the projected feedback never shows a green-only
picture of a run that is not a PASS (and never draws an unchecked sparse decode), fault points
in mm get their cross, the check registry keeps its built-ins, and the headline number is a
deviation from the failing check."""

import copy
import json
import math
import sys

import numpy as np
import pytest

from engine import api
from engine.recipes import checks as registry
from engine.recipes import feedback as FB
from engine.recipes import runner, schema
from engine.recipes.checks.base import Check, CheckResult, Param
from engine.recipes.context import CheckContext
from engine.recipes.geometry import part_frame, resolve_zone

import recipe_synth as RS
from conftest import ROOT
from test_recipes_mock import F, H, W, Z0, _bump, box, doc, one, view, zone

RECIPES = ROOT / "recipes"
BENCH = "bench_9in_metric"


@pytest.fixture(autouse=True)
def _fresh_engine():
    yield
    api._session = None


def _px(u, v):
    """Part-frame mm on the synthetic wall -> camera px of view()."""
    return u / Z0 * F + W / 2, v / Z0 * F + H / 2


def _cam_rect(u0, v0, u1, v1):
    x0, y0 = _px(u0, v0)
    x1, y1 = _px(u1, v1)
    return (x0, y0, x1, y1)


def _dmd_rect(u0, v0, u1, v1):
    """The DMD (col, row) rect view() decodes for a part-frame rect at height 0."""
    x0, y0, x1, y1 = _cam_rect(u0, v0, u1, v1)
    return (0.5 * x0 + 100, y0 + 200, 0.5 * x1 + 100, y1 + 200)


# ================================================================== presence: unseen != absent

UNMEASURED_PART = lambda u, v: np.where((abs(u) <= 20) & (abs(v) <= 15), np.nan, 0.0)  # noqa: E731


@pytest.mark.parametrize("units", ["part_mm", "camera_px", "projector_px"])
def test_presence_absent_never_passes_an_unmeasured_part(units):
    coords = {"part_mm": (-30, -20, 30, 20), "camera_px": _cam_rect(-30, -20, 30, 20),
              "projector_px": _dmd_rect(-30, -20, 30, 20)}[units]
    d = doc([zone("z", [{"type": "presence", "params": {"expect": "absent"}}],
                  coords=coords, units=units)])
    r = one(d, view(UNMEASURED_PART))
    assert r.verdict == "INVALID" and "unseen" in r.message, (r.verdict, r.message, r.values)
    assert r.values["unseen_pct"] > 40
    # the same zone, all measured: empty -> PASS; a measured part -> FAIL
    assert one(d, view(lambda u, v: 0.0)).verdict == "PASS"
    full = one(d, view(lambda u, v: np.where((abs(u) <= 20) & (abs(v) <= 15), 10.0, 0.0)))
    assert full.verdict == "FAIL" and full.values["part_pct"] > 40


def test_presence_missing_part_needs_the_zone_seen_too():
    d = doc([zone("z", [{"type": "presence", "params": {"min_height_mm": 5}}],
                  coords=(-30, -20, 30, 20))])
    r = one(d, view(UNMEASURED_PART))                     # dark part there, not decoded
    assert r.verdict == "INVALID" and "unseen" in r.message
    # only a 0.6 mm sliver of the zone measured (the fixture round it is): never "absent"
    def sliver(u, v):
        inside = (abs(u) <= 30) & (abs(v) <= 20)
        return np.where(inside & (abs(v + 19.7) >= 0.3), np.nan, 0.0)
    r = one(doc([zone("z", [{"type": "presence", "params": {"expect": "absent"}}],
                      coords=(-30, -20, 30, 20))]), view(sliver))
    assert r.verdict == "INVALID" and r.values["valid_pct"] < 10, r.values


def test_presence_needs_min_points():
    d = doc([zone("z", [{"type": "presence", "params": {"min_points": 5000}}],
                  coords=(-10, -10, 10, 10))])
    r = one(d, view(box(-20, -20, 20, 20, 10)))
    assert r.verdict == "INVALID" and "measured points" in r.message


def test_projector_zone_footprint_covers_its_unmeasured_px():
    sv = view(UNMEASURED_PART)
    r = schema.validate(doc([zone("z", [{"type": "presence"}], units="projector_px",
                                  coords=_dmd_rect(-30, -20, 30, 20))]))
    zv = resolve_zone(r.zones[0], sv, part_frame(sv))
    x0, y0, x1, y1 = _cam_rect(-30, -20, 30, 20)
    true = (x1 - x0) * (y1 - y0)
    assert zv.n_valid + zv.n_unseen == pytest.approx(true, rel=0.03)
    assert zv.n_unseen == pytest.approx(40 * 30 * (F / Z0) ** 2, rel=0.05)    # the dark part


ABSENT = [{"type": "presence", "params": {"expect": "absent"}}]


@pytest.mark.parametrize("units", ["part_mm", "camera_px", "projector_px"])
def test_presence_absent_zone_partly_off_the_camera_image_is_invalid(units):
    """The camera sees u up to 62.5 mm; the zone runs to 110 mm. Its visible half is measured
    fixture: the half off the image was never seen (it used to read 'absent', 100 % valid)."""
    coords = {"part_mm": (30, -20, 110, 20), "camera_px": _cam_rect(30, -20, 110, 20),
              "projector_px": _dmd_rect(30, -20, 110, 20)}[units]
    r = one(doc([zone("z", ABSENT, coords=coords, units=units)]), view(lambda u, v: 0.0))
    assert r.verdict == "INVALID" and r.values["unseen_pct"] > 40, (r.message, r.values)


def test_presence_absent_zone_hidden_by_a_measured_object_outside_it_is_invalid():
    """80 % of a part_mm zone's camera px see a tall measured object whose (u, v) is outside
    the zone: the zone's own extent there was never seen (it read 100 % valid)."""
    sv = view(lambda u, v: 0.0)
    xyz = sv.xyz.copy()
    x0, y0, x1, y1 = [int(round(c)) for c in _cam_rect(-30, -20, 30, 20)]
    ys, xs = slice(y0, y1), slice(x0, x0 + int(0.8 * (x1 - x0)))
    yy, xx = np.mgrid[ys, xs].astype(float)
    Z = Z0 - 300.0
    xyz[ys, xs, 0] = (xx - W / 2) / F * Z + 80.0
    xyz[ys, xs, 1] = (yy - H / 2) / F * Z
    xyz[ys, xs, 2] = Z
    sv = type(sv)(xyz=xyz, valid=sv.valid, K=sv.K, dist=sv.dist, corr_x=sv.corr_x,
                  corr_y=sv.corr_y, flood_white=sv.flood_white, flood_black=sv.flood_black,
                  scan_id="s", calibration_id="c")
    r = one(doc([zone("z", ABSENT, coords=(-30, -20, 30, 20))]), sv)
    assert r.verdict == "INVALID", (r.message, r.values)


@pytest.mark.parametrize("dark", [(-10, -2), (2, 10)])
def test_presence_absent_projector_zone_off_the_datum_depth_is_invalid(dark):
    """A boss 40 mm proud, its top the datum; the zone is the DMD region lighting the boss top
    (parallax: deeper -> +x); a dark, undecoded object on 40 % of it. A camera footprint
    placed with the surroundings' depth missed half of it on one side (PASS 'absent'); the
    zone's own DMD extent shows it whichever side it is on."""
    from engine.recipes.geometry import frame_for
    a, b = dark

    def f(u, v):
        boss = (abs(u) <= 15) & (abs(v) <= 25)
        shadow = (u > 15) & (u <= 21.25) & (abs(v) <= 25)
        dk = (u >= a) & (u <= b) & (abs(v) <= 20)
        return np.where(dk | shadow, np.nan, np.where(boss, 40.0, 0.0))
    sv = view(f)
    x0, y0, x1, y1 = _cam_rect(-10, -20, 10, 20)
    d = doc([zone("datum", [], coords=_cam_rect(11, -24, 14, 24), units="camera_px", datum=True),
             zone("z", ABSENT, units="projector_px",
                  coords=(0.5 * x0 + 110, y0 + 200, 0.5 * x1 + 110, y1 + 200))],
            part={"frame": {"reference": {"zone": "datum"}}})
    r = schema.validate(d)
    fr = frame_for(r, sv)
    views = {z.id: resolve_zone(z, sv, fr) for z in r.zones}
    ctx = CheckContext(scan=sv, frame=fr, zone=views["z"], zones=views)
    res = r.zones[1].checks[0].check.run(ctx)
    assert res.verdict == "INVALID" and res.values["unseen_pct"] > 30, (res.message, res.values)


# ================================================================== defect local: coverage

PLATE = box(-30, -20, 30, 20, 8)
DENTED = lambda u, v: PLATE(u, v) - _bump(5, 3, 4, 0.6)(u, v)  # noqa: E731
DEFECT = [{"type": "defect", "params": {"depth_mm": 0.3, "min_area_mm2": 2, "mode": "local",
                                        "min_height_mm": 5}}]


def test_defect_local_full_coverage_pass_and_fail():
    d = doc([zone("z", DEFECT, coords=(-25, -15, 25, 15))])
    clean = one(d, view(PLATE))
    assert clean.verdict == "PASS" and clean.values["valid_pct"] > 95, clean.values
    assert clean.values["max_depth_mm"] < 0.3 and clean.key == "max_depth_mm"
    bad = one(d, view(DENTED))
    assert bad.verdict == "FAIL" and bad.values["max_depth_mm"] >= 0.3


def test_defect_local_strips_only_is_invalid():
    def strips(u, v):
        h = DENTED(u, v)
        keep = ((v >= -14) & (v <= -10)) | ((v >= 10) & (v <= 14))
        return np.where(keep | (h < 1), h, np.nan)
    r = one(doc([zone("z", DEFECT, coords=(-25, -15, 25, 15))]), view(strips))
    assert r.verdict == "INVALID" and "measured" in r.message, (r.message, r.values)
    assert r.values["valid_pct"] < 50


def test_flatness_needs_the_zone_measured():
    d = doc([zone("z", [{"type": "flatness", "params": {"tol_mm": 0.3, "min_height_mm": 5}}],
                  coords=(-25, -15, 25, 15))])
    assert one(d, view(PLATE)).verdict == "PASS"
    warped = lambda u, v: PLATE(u, v) + np.where(PLATE(u, v) > 1, 0.002 * u * u, 0)  # noqa: E731
    assert one(d, view(warped)).verdict == "FAIL"

    def strips(u, v):                       # the warped face, two flat-reading strips seen
        h = warped(u, v)
        return np.where((abs(u) <= 4) | (h < 1), h, np.nan)
    r = one(d, view(strips))
    assert r.verdict == "INVALID" and r.values["valid_pct"] < 70, (r.message, r.values)


def test_defect_local_sparse_camera_zone_is_invalid():
    def patch(u, v):
        h = DENTED(u, v)
        return np.where((u > -24) & (u < -18) & (v > -14) & (v < -8), h, np.nan)
    d = doc([zone("z", [{"type": "defect", "params": {"depth_mm": 0.3, "min_area_mm2": 2,
                                                       "mode": "local"}}],
                  coords=_cam_rect(-25, -15, 25, 15), units="camera_px")])
    r = one(d, view(patch))
    assert r.verdict == "INVALID", (r.message, r.values)


def test_defect_local_dropout_over_the_dent_is_invalid():
    hole = lambda u, v: np.where(np.hypot(u - 5, v - 3) < 5, np.nan, DENTED(u, v))  # noqa: E731
    r = one(doc([zone("z", DEFECT, coords=(-25, -15, 25, 15))]), view(hole))
    assert r.verdict == "INVALID" and "gap" in r.message, (r.message, r.values)
    assert r.values["largest_gap_mm2"] > 50
    # a pin-hole far below min_area_mm2 cannot hide a defect: still a clean PASS
    pin = lambda u, v: np.where(np.hypot(u + 10, v) < 0.3, np.nan, PLATE(u, v))  # noqa: E731
    ok = one(doc([zone("z", DEFECT, coords=(-25, -15, 25, 15))]), view(pin))
    assert ok.verdict == "PASS" and 0 < ok.values["largest_gap_mm2"] < 2, ok.values


HOLE = lambda u, v: np.where(np.hypot(u - 5, v - 3) < 5, np.nan, DENTED(u, v))  # noqa: E731


@pytest.mark.parametrize("case", ["px_line", "glint_0.35", "glint_0.7", "glint_1.5", "border",
                                  "camera_part_edge"])
def test_defect_local_dropout_open_to_an_edge_is_invalid(case):
    """The dent's dropout used to count as a gap only when measured px enclosed it: one
    unmeasured px line to the zone border, a glint streak through it, a dent at the zone
    border or at the part edge (camera zone wider than the part) left it 'open' - PASS with
    the whole 0.6 mm dent unmeasured. Any unseen region of the zone that can hold the
    defect now counts, enclosed or not."""
    d = doc([zone("z", DEFECT, coords=(-25, -15, 25, 15))])
    if case == "px_line":
        f = lambda u, v: np.where((np.abs(v - 3) < 0.2) & (u > 5), np.nan, HOLE(u, v))  # noqa
    elif case.startswith("glint"):
        w = float(case.split("_")[1])
        f = lambda u, v: np.where(np.abs(v - 3) < w / 2, np.nan, HOLE(u, v))  # noqa: E731
    elif case == "border":
        dent = lambda u, v: PLATE(u, v) - _bump(20, 3, 4, 0.6)(u, v)  # noqa: E731
        assert one(d, view(dent)).verdict == "FAIL"
        f = lambda u, v: np.where(np.hypot(u - 22, v - 3) < 5, np.nan, dent(u, v))  # noqa
    else:
        small = box(-20, -12, 20, 12, 8)
        dent = lambda u, v: small(u, v) - _bump(16, 3, 4, 0.6)(u, v)  # noqa: E731
        d = doc([zone("z", DEFECT, coords=_cam_rect(-28, -18, 28, 18), units="camera_px")])
        assert one(d, view(dent)).verdict == "FAIL"
        f = lambda u, v: np.where(np.hypot(u - 18, v - 3) < 5, np.nan, dent(u, v))  # noqa
    r = one(d, view(f))
    assert r.verdict == "INVALID" and "gap" in r.message, (r.message, r.values)


def test_defect_local_a_thin_streak_alone_cannot_hide_a_defect():
    """A clean plate with a glint streak 0.35 mm wide across the zone: no 2 mm2 defect fits
    in it - a clean PASS, not INVALID."""
    f = lambda u, v: np.where(np.abs(v - 3) < 0.175, np.nan, PLATE(u, v))  # noqa: E731
    r = one(doc([zone("z", DEFECT, coords=(-25, -15, 25, 15))]), view(f))
    assert r.verdict == "PASS", (r.message, r.values)


def test_defect_dent_sign_headline_ignores_bumps():
    bumped = lambda u, v: PLATE(u, v) + _bump(5, 3, 4, 0.6)(u, v)  # noqa: E731
    d = doc([zone("z", [{"type": "defect", "params": {"depth_mm": 0.3, "min_area_mm2": 2,
                                                       "mode": "local", "sign": "dent",
                                                       "min_height_mm": 5}}],
                  coords=(-25, -15, 25, 15))])
    r = one(d, view(bumped))
    assert r.verdict == "PASS" and r.values["max_depth_mm"] < 0.3, r.values
    assert r.values["max_residual_mm"] < 0.3                  # a bump is no dent


# ================================================================== edge

def _rot_plate(deg, hw=15.0, hl=40.0, h=6.0):
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    return lambda u, v: np.where((abs(c * u + s * v) <= hw) & (abs(-s * u + c * v) <= hl), h, 0.0)


@pytest.mark.parametrize("side", ["top", "both"])
@pytest.mark.parametrize("gdeg, sdeg", [(0.2, -0.2), (-0.3, 0.3), (0.0, 0.0)])
def test_edge_angle_of_a_top_bottom_part_is_undirected(side, gdeg, sdeg):
    d = doc([zone("e", [{"type": "edge", "params": {"side": side, "angle_tol_deg": 1.0,
                                                     "position_tol_mm": 1.0,
                                                     "min_height_mm": 3}}],
                  coords=(-30, -46, 30, 46))], golden={"scan": "g"})
    r = one(d, view(_rot_plate(sdeg), seed=5), view(_rot_plate(gdeg), seed=6))
    assert r.verdict == "PASS", (r.message, r.values)
    for k, v in r.values.items():
        if k.endswith("_angle_deg"):
            assert abs(v) < 1.0, (k, v)


def test_edge_angle_still_fails_a_turned_part():
    d = doc([zone("e", [{"type": "edge", "params": {"side": "top", "angle_tol_deg": 1.0,
                                                     "min_height_mm": 3}}],
                  coords=(-30, -46, 30, 46))], golden={"scan": "g"})
    r = one(d, view(_rot_plate(3.0), seed=5), view(_rot_plate(0.0), seed=6))
    assert r.verdict == "FAIL" and 2.0 < abs(r.values["top_angle_deg"]) < 4.0, r.values


def _sparse_top_edge(plate, v_edge=-15.0, band=8.0, step=5):
    """plate with only a sparse px grid measured in a band just inside its top edge."""
    def f(u, v):
        h = plate(u, v)
        xx = np.round(u / Z0 * F + W / 2).astype(int)
        yy = np.round(v / Z0 * F + H / 2).astype(int)
        band_px = (v >= v_edge) & (v <= v_edge + band)
        grid = (xx % step == 0) & (yy % step == 0)
        return np.where(band_px & ~grid, np.nan, h)
    return f


def test_edge_both_sides_asked_both_measured():
    plate = box(-40, -15, 40, 15, 6)
    sparse = _sparse_top_edge(plate)
    for side in ("top", "both"):
        d = doc([zone("e", [{"type": "edge", "params": {"side": side, "straightness_tol_mm": 0.4,
                                                         "min_height_mm": 3}}],
                      coords=(-45, -25, 45, 25))])
        assert one(d, view(plate)).verdict == "PASS"
        r = one(d, view(sparse))
        assert r.verdict == "INVALID" and "top" in r.message, (side, r.message, r.values)
    # the golden must have every side compared too
    dg = doc([zone("e", [{"type": "edge", "params": {"side": "both", "position_tol_mm": 1.0,
                                                      "min_height_mm": 3}}],
                   coords=(-45, -25, 45, 25))], golden={"scan": "g"})
    r = one(dg, view(plate, seed=2), view(sparse))
    assert r.verdict == "INVALID" and r.message.startswith("golden"), r.message


# ================================================================== aggregate + check results

def test_aggregate_fails_closed():
    assert runner.aggregate(["PASS", "PASS"]) == "PASS"
    assert runner.aggregate(["FAIL", "INVALID"]) == "FAIL"
    for bad in (["fail"], ["PASS", "Fail"], [None], ["FAILED"], ["PASS", "ERROR"], []):
        assert runner.aggregate(bad) == "INVALID", bad


class _Spec:
    def __init__(self, check):
        self.id, self.type, self.check = check.id, check.id, check


def _ctx():
    sv = view(box(-10, -10, 10, 10, 10))
    fr = part_frame(sv)
    r = schema.validate(doc([zone("z", [{"type": "presence"}])]))
    zv = resolve_zone(r.zones[0], sv, fr)
    return CheckContext(scan=sv, frame=fr, zone=zv, zones={"z": zv})


def test_a_check_answering_no_verdict_is_invalid():
    class Mutated(Check):
        id = "mutated"

        def run(self, ctx):
            r = CheckResult("PASS")
            r.verdict = "fail"                   # set after construction: never validated
            return r

    class NotAResult(Check):
        id = "not_a_result"

        def run(self, ctx):
            return {"verdict": "PASS"}

    ctx = _ctx()
    for cls in (Mutated, NotAResult):
        d = runner._run_check(_Spec(cls()), ctx)
        assert d["verdict"] == "INVALID" and "check failed" in d["message"], d
    assert runner.aggregate([runner._run_check(_Spec(Mutated()), ctx)["verdict"]]) == "INVALID"


def test_xyz_mm_fault_points_get_their_camera_px():
    class MmFault(Check):
        id = "mm_fault"

        def run(self, ctx):
            ys, xs = ctx.index()
            i = 0                                   # the zone's corner: on the wall (on its ray)
            return CheckResult("FAIL", fault_points=[{"xyz_mm": ctx.scan.xyz[ys[i], xs[i]].tolist(),
                                                      "label": "here"},
                                                     {"label": "nowhere"}],
                               details={"px": [float(xs[i]), float(ys[i])]})

    ctx = _ctx()
    d = runner._run_check(_Spec(MmFault()), ctx)
    assert d["verdict"] == "FAIL"
    fp = d["fault_points"][0]
    assert fp["px"] == pytest.approx(d["details"]["px"], abs=0.05)
    assert any("nowhere" in w for w in d["warnings"])       # never dropped silently


# ================================================================== feedback

def _fb():
    f = schema.validate(doc([zone("z", [{"type": "presence"}])],
                            feedback={"enabled": True, "text": False})).feedback
    return f


def _zres(zid, verdict, faults=()):
    return {"id": zid, "verdict": verdict,
            "checks": [{"id": "c", "type": "c", "verdict": verdict,
                        "fault_points": [dict(f) for f in faults]}]}


def _views(sv, zones):
    r = schema.validate(doc(zones))
    fr = part_frame(sv)
    return {z.id: resolve_zone(z, sv, fr) for z in r.zones}


def _red(img):
    return (img[..., 2] > 200) & (img[..., 1] < 60) & (img[..., 0] < 60)


def _green(img):
    return (img[..., 1] > 200) & (img[..., 2] < 60) & (img[..., 0] < 60)


TWO = [zone("a", [{"type": "presence"}], coords=_cam_rect(-30, -15, -5, 15), units="camera_px"),
       zone("b", [{"type": "presence"}], coords=_cam_rect(5, -15, 30, 15), units="camera_px")]


def test_feedback_unmappable_fail_zone_frames_the_field_red():
    sv = view(lambda u, v: 0.0)
    views = _views(sv, TWO)
    ok = views["a"].mask & np.isfinite(sv.corr_x)
    ys, xs = np.nonzero(ok)
    keep = np.zeros_like(ok)
    keep[ys[:5], xs[:5]] = True                         # only 5 decoded px in the FAIL zone
    sv.corr_x[ok & ~keep] = np.nan
    sv.corr_y[ok & ~keep] = np.nan
    zr = [_zres("a", "FAIL", [{"px": [float(xs[0]), float(ys[0])]}]), _zres("b", "PASS")]
    img, info = FB.render(views, zr, sv, _fb(), "FAIL")
    assert img is not None and info["frame"]["verdict"] == "FAIL" and "a" in info["frame"]["why"]
    k = FB.FRAME_PX
    assert _red(img[:k]).all() and _red(img[-k:]).all() and _red(img[:, :k]).all()
    assert _green(img).sum() > 100                      # the passing zone is still outlined
    # an unresolved INVALID zone frames it amber
    img, info = FB.render({"b": views["b"]}, [_zres("a", "INVALID"), _zres("b", "PASS")], sv,
                          _fb(), "INVALID")
    assert info["frame"]["verdict"] == "INVALID"
    assert tuple(img[0, 0]) == tuple(FB.colour_for("INVALID", _fb()))
    # a drawn FAIL zone needs no frame
    sv2 = view(lambda u, v: 0.0)
    img, info = FB.render(_views(sv2, TWO), [_zres("a", "FAIL"), _zres("b", "PASS")], sv2,
                          _fb(), "FAIL")
    assert "frame" not in info and not _red(img[:4]).any()


@pytest.mark.parametrize("truthful, drawn", [(False, False), (True, True)])
def test_feedback_sparse_decode_is_checked_whole_zone(truthful, drawn):
    sv = view(lambda u, v: 0.0)
    views = _views(sv, TWO[:1])
    m = views["a"].mask
    rng = np.random.default_rng(3)
    sparse = m & (rng.random(m.shape) < 0.04)
    sv.corr_x[m & ~sparse] = np.nan
    sv.corr_y[m & ~sparse] = np.nan
    if not truthful:                                    # wrong correspondences, 4 % density
        sv.corr_x[sparse] = rng.uniform(0, 911, sparse.sum())
        sv.corr_y[sparse] = rng.uniform(0, 1139, sparse.sum())
    zm = FB.zone_map(sv, m)
    assert zm["n_tiles"] == 0 or zm["tile_pct"] < FB.MIN_TILE_PCT
    img, info = FB.render(views, [_zres("a", "FAIL")], sv, _fb(), "FAIL")
    (rec,) = info["zones"]
    assert rec["drawn"] is drawn, rec
    if not drawn:
        assert img is None and "sparse" in rec["refused"]


def test_feedback_xyz_mm_fault_gets_a_cross():
    sv = view(lambda u, v: 0.0)
    views = _views(sv, TWO[:1])
    ys, xs = np.nonzero(views["a"].valid)
    i = len(xs) // 2
    fault = {"xyz_mm": sv.xyz[ys[i], xs[i]].tolist(), "label": "dent"}
    img, info = FB.render(views, [_zres("a", "FAIL", [fault])], sv, _fb(), "FAIL")
    (cr,) = info["crosses"]
    assert cr["px"] == pytest.approx([xs[i], ys[i]], abs=0.05)
    vis = FB.overlay(sv, part_frame(sv), views, [_zres("a", "FAIL", [fault])], _fb(), "FAIL")
    assert vis is not None


# ================================================================== check registry

def _fresh_registry(monkeypatch):
    """The registry as a new process has it (nothing read yet); restored afterwards."""
    registry.available()                    # the real one loaded, so teardown restores it
    monkeypatch.setattr(registry, "REGISTRY", {})
    monkeypatch.setitem(registry._LOADED, "builtin", False)
    monkeypatch.setitem(registry._LOADED, "entry_points", True)
    monkeypatch.delitem(sys.modules, registry.BUILTIN_MODULE)
    monkeypatch.delattr(registry, "builtin", raising=False)


def test_a_plugin_never_shadows_a_built_in(monkeypatch):
    _fresh_registry(monkeypatch)

    class MyFlatness(Check):
        id = "flatness"
        params = {"tol_mm": Param("number", 1.0, "mm")}

    with pytest.raises(ValueError, match="built-in"):
        registry.register(MyFlatness)                   # before the registry was ever read
    assert registry.get("flatness") is not MyFlatness
    assert {"presence", "position", "height", "flatness", "defect", "edge",
            "code"} <= set(registry.available())


def test_a_failed_builtin_import_is_retried_whole(monkeypatch):
    _fresh_registry(monkeypatch)
    monkeypatch.setitem(sys.modules, registry.BUILTIN_MODULE, None)     # the import fails
    with pytest.raises(ImportError):
        registry.get("defect")
    assert registry._LOADED["builtin"] is False and not registry.REGISTRY
    monkeypatch.delitem(sys.modules, registry.BUILTIN_MODULE)
    assert registry.get("defect").id == "defect" and "code" in registry.available()


def test_a_direct_builtin_import_that_failed_is_never_taken_for_loaded(monkeypatch):
    """Plugin code may import builtin.py directly (docs/CHECKS.md points at edge_points). A
    failure partway through that import used to leave presence / position / height
    registered with the 'loaded' flag set: a plugin could then take 'flatness' and 'defect'
    was gone. The built-in ids are reserved, and the next read re-imports the module whole."""
    import importlib
    _fresh_registry(monkeypatch)
    real = registry.register
    state = {"fail": True}

    def flaky(cls):
        if cls.id == "flatness" and state["fail"]:
            state["fail"] = False
            raise RuntimeError("transient failure at flatness")
        return real(cls)
    monkeypatch.setattr(registry, "register", flaky)
    with pytest.raises(RuntimeError, match="transient"):
        importlib.import_module(registry.BUILTIN_MODULE)
    monkeypatch.setattr(registry, "register", real)
    assert registry._LOADED["builtin"] is False

    class MyFlatness(Check):
        id = "flatness"
    with pytest.raises(ValueError, match="built-in"):
        registry.register(MyFlatness)
    assert registry.get("flatness").__module__ == registry.BUILTIN_MODULE
    assert set(registry.BUILTIN_IDS) <= set(registry.available())
    assert all(registry.REGISTRY[k].__module__ == registry.BUILTIN_MODULE
               for k in registry.BUILTIN_IDS)


def test_the_registry_holds_no_lock_over_an_import(monkeypatch):
    """Thread B imports builtin.py directly (slowly) while thread A reads the registry for the
    first time: the registry lock used to be held over the import, against Python's own
    module import lock - both threads hung, and every later read with them."""
    import importlib
    import importlib.util
    import threading
    import time
    _fresh_registry(monkeypatch)
    spec0 = importlib.util.find_spec(registry.BUILTIN_MODULE)

    class SlowLoader(type(spec0.loader)):
        def exec_module(self, module):
            time.sleep(0.5)
            return super().exec_module(module)

    class Finder:
        def find_spec(self, name, path=None, target=None):
            if name == registry.BUILTIN_MODULE and threading.current_thread().name == "B":
                return importlib.util.spec_from_file_location(
                    name, spec0.origin, loader=SlowLoader(name, spec0.origin))
            return None
    monkeypatch.setattr(sys, "meta_path", [Finder()] + sys.meta_path)
    res = {}
    tb = threading.Thread(target=lambda: res.update(
        B=importlib.import_module(registry.BUILTIN_MODULE)), name="B", daemon=True)
    tb.start()
    time.sleep(0.15)
    ta = threading.Thread(target=lambda: res.update(A=registry.get("defect")), name="A",
                          daemon=True)
    ta.start()
    ta.join(20)
    tb.join(20)
    assert not ta.is_alive() and not tb.is_alive(), "registry read and builtin import hung"
    assert res["A"].id == "defect" and set(registry.BUILTIN_IDS) <= set(registry.available())


# ================================================================== docs

def test_checks_doc_reuses_only_what_the_repo_has():
    """docs/CHECKS.md's built-in table points readers at code they can open: every path in
    its 'reuses' column exists in the repo, and every row names a path or a repo concept."""
    text = (ROOT / "docs" / "CHECKS.md").read_text(encoding="utf-8")
    table = text.split("## Built-in checks", 1)[1].split("\n## ", 1)[0]
    rows = [r for r in table.splitlines() if r.startswith("| `")]
    assert {r.split("`")[1] for r in rows} == {"presence", "position", "height", "flatness",
                                               "defect", "edge", "code"}
    for r in rows:
        reuses = r.rstrip().rstrip("|").rsplit("|", 1)[1].strip()
        paths = [t for t in reuses.split("`")[1::2] if "/" in t]
        for t in paths:
            stem = t if t.endswith(".py") else t.split(".")[0] + ".py"
            assert (ROOT / stem).is_file(), (r.split("`")[1], t)
        assert paths or reuses in ("", "the part frame", "the recipe's golden scan"), reuses


# ================================================================== headline

def _ch(cid, verdict, key, value, limit):
    return {"id": cid, "type": cid, "verdict": verdict, "key": key, "limit": limit,
            "values": {key: value} if key else {}}


class _Ms:
    meta = {"n_points": 1000}


def _head(checks):
    res = {"deviation_mm": {"max": None, "method": None, "n_points": None},
           "threshold_mm": None}
    runner._headline(res, [{"id": "z", "checks": checks}], _Ms())
    return res


def test_headline_comes_from_the_failing_check():
    res = _head([_ch("low", "FAIL", "deviation_mm", -2.1, 1.5),
                 _ch("high", "PASS", "deviation_mm", 1.4, 1.5)])
    assert res["deviation_mm"]["max"] == -2.1 and res["threshold_mm"] == 1.5
    assert "z/low" in res["deviation_mm"]["method"]
    res = _head([_ch("small_fail", "FAIL", "offset_mm", 0.5, 2.0),
                 _ch("near_pass", "PASS", "flatness_mm", 0.29, 0.3)])
    assert "small_fail" in res["deviation_mm"]["method"]
    res = _head([_ch("code", "FAIL", None, None, None),
                 _ch("near_pass", "PASS", "flatness_mm", 0.29, 0.3)])
    assert res["deviation_mm"]["max"] is None and "z/code (FAIL" in res["deviation_mm"]["method"]
    res = _head([_ch("inv", "INVALID", "offset_mm", 9.0, 1.0),
                 _ch("ok", "PASS", "offset_mm", 0.2, 1.0)])
    assert res["deviation_mm"]["max"] == 0.2


def test_height_reports_a_deviation_not_an_absolute_height():
    d = doc([zone("z", [{"type": "height", "params": {"nominal_mm": 20, "tol_mm": 0.5,
                                                       "min_height_mm": 5}}],
                  coords=(-10, -10, 10, 10))])
    r = one(d, view(box(-20, -20, 20, 20, 20)))
    assert r.verdict == "PASS" and r.key == "deviation_mm" and r.limit == pytest.approx(0.5)
    assert abs(r.values["deviation_mm"]) < 0.1
    assert r.values["height_mm"] == pytest.approx(20, abs=0.1)
    lo = one(d, view(box(-20, -20, 20, 20, 19.0)))
    assert lo.verdict == "FAIL" and lo.values["deviation_mm"] == pytest.approx(-1.0, abs=0.1)
    assert abs(lo.values["deviation_mm"]) > lo.limit
    band = one(doc([zone("z", [{"type": "height", "params": {"band_mm": [19, 23],
                                                              "min_height_mm": 5}}],
                         coords=(-10, -10, 10, 10))]), view(box(-20, -20, 20, 20, 20)))
    assert band.verdict == "PASS" and band.values["deviation_mm"] == pytest.approx(-1.0, abs=0.1)
    assert band.limit == pytest.approx(2.0)


# ================================================================== runs on stored scans

@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    """A mock library: the bench's true stereo calibration, a GOOD tube recorded as the
    golden (of the built-in recipe and of a 'latest' copy), a misaligned tube labelled like
    the part ('tube': what every run's own scan is called) and a good box lid."""
    root = tmp_path_factory.mktemp("fail_closed") / "library"
    s = api.configure(root, mock=True)
    try:
        p = api.load_preset(BENCH)
        g = RS.oracle_scan(s, p, "scene:tube", "ribbed_tube_alignment_golden", seed=9)
        runner.record_golden(s, api.recipe_load("ribbed_tube_alignment"), p, g)
        runner.record_golden(s, schema.validate(_latest_doc()), p, g)
        bad = RS.oracle_scan(s, p, "scene:tube:misaligned", "tube", seed=1)
        lid = RS.oracle_scan(s, p, "scene:box_lid", "box", seed=2)
    finally:
        api._session = None
    return root, {"golden": g.id, "misaligned": bad.id, "lid": lid.id}


@pytest.fixture
def eng_lib(lib):
    s = api.configure(lib[0], mock=True)
    yield s, lib[1]
    api._session = None


def _latest_doc(name="tube_latest", golden="latest"):
    d = json.loads((RECIPES / "ribbed_tube_alignment.json").read_text(encoding="utf-8"))
    d["name"], d["golden"] = name, {"scan": golden}
    d["feedback"]["enabled"] = False
    return d


def _run(recipe, scan, tmp_path, **kw):
    out = api.recipe_run(recipe, scan=scan, runs_dir=tmp_path / "runs", project_feedback=False,
                         **kw)
    return out["result"], out


def _zone(res, zid):
    return next(z for z in res["zones"] if z["id"] == zid)


def test_golden_latest_is_the_recorded_golden_not_the_newest_scan(eng_lib, tmp_path):
    s, sc = eng_lib
    res, _ = _run(_latest_doc(), sc["misaligned"], tmp_path)  # the newest scan is the test one
    assert res["result"] == "FAIL" and _zone(res, "right")["verdict"] == "FAIL", res["warnings"]
    assert not [w for w in res["warnings"] if w.startswith("golden")]
    pos = next(c for c in _zone(res, "right")["checks"] if c["type"] == "position")
    assert pos["values"]["offset_mm"] > 2.0


def test_golden_is_never_the_scan_being_checked(eng_lib, tmp_path):
    s, sc = eng_lib
    for recipe in ("ribbed_tube_alignment", _latest_doc(), _latest_doc("by_id", sc["golden"])):
        res, out = _run(recipe, sc["golden"], tmp_path)
        assert res["result"] == "INVALID" and out["exit_code"] == 5, res["result"]
        assert "being checked" in res["invalid_reason"], res["invalid_reason"]
        assert all(c["verdict"] == "INVALID" for z in res["zones"] for c in z["checks"])


def test_golden_label_never_matches_a_plain_test_scan(eng_lib, tmp_path):
    s, sc = eng_lib
    res, _ = _run(_latest_doc("tube_label", "tube"), sc["misaligned"], tmp_path)
    assert res["result"] == "INVALID"
    assert "none was recorded as a golden" in res["invalid_reason"], res["invalid_reason"]
    with pytest.raises(LookupError, match="no golden recorded for recipe nobody"):
        runner.find_golden(s, schema.validate(_latest_doc("nobody")),
                           api.load_preset(BENCH).geometry_id)
    # an exact metric scan id is the integrator's explicit choice
    res, _ = _run(_latest_doc("tube_by_id", sc["golden"]), sc["misaligned"], tmp_path)
    assert res["result"] == "FAIL"


def test_box_lid_headline_is_a_deviation(eng_lib, tmp_path):
    s, sc = eng_lib
    res, _ = _run("box_lid", sc["lid"], tmp_path)
    assert res["result"] == "PASS"
    dev = res["deviation_mm"]
    assert abs(dev["max"]) < res["threshold_mm"] <= 1.5, (dev, res["threshold_mm"])   # not 40 mm
    assert "height_mm" not in dev["method"]
    h = next(c for c in _zone(res, "lid_left")["checks"] if c["type"] == "height")
    assert h["key"] == "deviation_mm" and h["limit"] == pytest.approx(1.5)
    assert abs(h["values"]["deviation_mm"]) < 0.3 and h["values"]["height_mm"] == \
        pytest.approx(40.0, abs=0.3)


def test_golden_and_scan_at_other_rig_settings_are_invalid(tmp_path):
    s = api.configure(tmp_path / "library", mock=True)
    p_old = api.load_preset(BENCH)
    g = RS.oracle_scan(s, p_old, "scene:tube", "ribbed_tube_alignment_golden", seed=9)
    runner.record_golden(s, api.recipe_load("ribbed_tube_alignment"), p_old, g)
    old_test = RS.oracle_scan(s, p_old, "scene:tube", "tube", seed=3)
    p_new = copy.deepcopy(p_old)                    # the preset edited in place: same id,
    p_new.rig.standoff_mm = (p_old.rig.standoff_mm or 600.0) + 50.0      # new fingerprint
    cal_new = RS.truth_calibration(s, p_new)        # recalibrated at the new settings
    assert cal_new.id != g.calibration_id and p_new.geometry_id == p_old.geometry_id
    t = RS.oracle_scan(s, p_new, "scene:tube", "tube", seed=4)
    res, _ = _run("ribbed_tube_alignment", t.id, tmp_path, preset=p_new)
    assert res["result"] == "INVALID", res["result"]
    assert "other rig settings" in res["invalid_reason"], res["invalid_reason"]
    res, out = _run("ribbed_tube_alignment", old_test.id, tmp_path, preset=p_new)
    assert res["result"] == "INVALID" and out["exit_code"] == 2
    assert "other rig settings" in res["invalid_reason"], res["invalid_reason"]


# ================================================================== live mock capture

def test_scan_golden_records_what_it_scanned(tmp_path, monkeypatch):
    s = api.configure(tmp_path / "library", mock=True)
    p = api.load_preset(BENCH)
    r = schema.validate(_latest_doc())
    good = RS.oracle_scan(s, p, "scene:tube", "tube_latest_golden", seed=9)
    seen = {}

    def fake_scan(s_, recipe, preset, label, log, res, ack):     # the capture, already done
        seen["label"] = label
        return good
    monkeypatch.setattr(runner, "_scan", fake_scan)
    g = runner.scan_golden(s, r, p, "tube_latest_golden")
    assert g is good and seen["label"] == "tube_latest_golden"
    (rec,) = runner.list_goldens(s.store, p.geometry_id)
    assert rec["scan_id"] == good.id and rec["recipe"] == "tube_latest"
    assert rec["calibration_id"] == good.calibration_id and rec["geometry_fp"]
    assert runner.find_golden(s, r, p.geometry_id).id == good.id


def test_api_recipe_golden_records_the_golden(tmp_path, monkeypatch):
    """`scan recipe-golden` / the web verb go through api.recipe_golden: it used to scan
    without recording, so every run naming the golden by label or 'latest' was INVALID
    ('none was recorded as a golden'). The entry is a library entry (listable, trashable)."""
    s = api.configure(tmp_path / "library", mock=True)
    p = api.load_preset(BENCH)
    good = RS.oracle_scan(s, p, "scene:tube", "tube_latest_golden", seed=9)
    monkeypatch.setattr(runner, "_scan", lambda *a: good)          # the capture, already done
    assert api.recipe_golden(_latest_doc(), p) is good
    (rec,) = api.list_library("recipe_golden", p.geometry_id)
    assert rec["scan_id"] == good.id and rec["recipe"] == "tube_latest"
    assert runner.find_golden(s, schema.validate(_latest_doc()), p.geometry_id).id == good.id
    from engine.library.store import ENTRY_KINDS
    assert "recipe_golden" in ENTRY_KINDS
    api.delete(rec["entry"], reason="test")
    assert api.list_library("recipe_golden", p.geometry_id) == []


def test_live_run_with_golden_latest_never_compares_its_own_scan(tmp_path):
    """The full mock capture path: a run on a misaligned tube with golden 'latest' compares
    with the recorded GOOD tube (FAIL), not with the scan it has just taken (the newest metric
    scan: every offset 0, a PASS)."""
    s = api.configure(tmp_path / "library", mock=True)
    p = api.load_preset(BENCH)
    g = RS.oracle_scan(s, p, "scene:tube", "tube_latest_golden", seed=9)
    runner.record_golden(s, schema.validate(_latest_doc()), p, g)
    out = api.recipe_run(_latest_doc(), mock_scene="scene:tube:misaligned",
                         runs_dir=tmp_path / "runs", project_feedback=False)
    res = out["result"]
    assert res["part"]["scan_id"] != g.id
    assert res["result"] == "FAIL" and _zone(res, "right")["verdict"] == "FAIL", res

"""Dispositions on top of the verdict (engine/disposition.py), MOCK only: CONTINUE (PASS),
REJECT (a definite measured FAIL), PULL for quality inspection (any doubt).

Every PULL reason (INVALID of each kind, a crashed check, an unreadable code, the rig check
MOVED / UNKNOWN, ABORTED, the guard band), the guard band just inside / outside both limits
of a two-sided tolerance (and one-sided ones), the precedence (a definite FAIL rejects even
with an INVALID zone; doubt pulls), the exit codes (0 / 1 / 5; refused 2, crashed 4 and
ABORTED 1 kept), result.json + run.log.jsonl fields, the command bus PULL output and result
code, the /quick lamp (node, fake DOM), the projected feedback on the mock scene and the
config validation. Recipe runs check synthetic metric scans offline (tests/recipe_synth.py):
no capture, no hardware."""

import contextlib
import io
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest

from engine import api, recipes, rules
from engine import disposition as D
from engine.control import commands as C
from engine.recipes import feedback as FB
from engine.recipes import schema
from engine.recipes.checks.base import Check
from ui import cli
from ui import run_job as RJ

import recipe_synth as RS
from conftest import ROOT

RECIPES = ROOT / "recipes"
STATIC = ROOT / "ui" / "web" / "static"
AMBER = tuple(schema.COLOURS["amber"])          # BGR


@pytest.fixture(autouse=True)
def _fresh_engine():
    yield
    api._session = None


# ================================================================== the rules (pure)

def _hits(tols, pct=10.0):
    return [h["side"] for h in D.guard_band(D.check_tolerances(tols), pct)]


def test_guard_band_just_inside_and_outside_every_kind_of_limit():
    T = D.tolerance
    # one-sided upper (flatness <= 0.5 mm): band 10 % of 0.5 = 0.05 mm
    assert _hits([T("flatness_mm", 0.46, hi=0.5)]) == ["hi"]
    assert _hits([T("flatness_mm", 0.44, hi=0.5)]) == []
    assert _hits([T("flatness_mm", 0.5, hi=0.5)]) == ["hi"]          # on the limit: inside
    assert _hits([T("flatness_mm", 0.51, hi=0.5)]) == []             # outside: the check FAILs
    # one-sided lower (part present >= 60 %): band 6 %
    assert _hits([T("part_pct", 65.0, lo=60.0)]) == ["lo"]
    assert _hits([T("part_pct", 67.0, lo=60.0)]) == []
    assert _hits([T("part_pct", 59.0, lo=60.0)]) == []
    # two-sided [37.1, 40.1] (nominal 38.6 +- 1.5): band 10 % of the half width = 0.15 mm
    band = [T("height_mm", None, 37.1, 40.1)]
    for v, want in ((40.0, ["hi"]), (39.9, []), (37.2, ["lo"]), (37.3, []), (38.6, [])):
        band[0]["value"] = v
        assert _hits(band) == want, v
    # a plugin without tolerances: its key / limit in deviation semantics (|value| <= limit)
    ch = {"key": "x_mm", "limit": 1.0, "values": {"x_mm": -0.95}}
    assert [h["side"] for h in D.guard_band(D.tolerances_of(ch), 10)] == ["lo"]
    ch["values"]["x_mm"] = 0.5
    assert D.guard_band(D.tolerances_of(ch), 10) == []
    # 0 % = off, even on the limit; no value: nothing to judge
    assert _hits([T("flatness_mm", 0.5, hi=0.5)], 0) == []
    assert _hits([T("rotation_deg", None, -2, 2)]) == []
    for bad in ([{"name": "x", "value": 1}], [{"name": "x", "lo": 2, "hi": 1}], "x",
                [{"name": "x", "lo": float("nan")}]):
        with pytest.raises(ValueError):
            D.check_tolerances(bad)


@pytest.mark.parametrize("message, reason", [
    ("the check failed: RuntimeError: kaboom", "check_crashed"),
    ("rig check MOVED: camera moved ~2.3 px (HOME / recalibrate)", "rig_check"),
    ("golden scan 'g': no golden 'g' at geometry x - scan a GOOD part", "golden"),
    ("deviation in px, not mm: no calibration at geometry g (rule 6)", "px_not_mm"),
    ("no stereo calibration for geometry g (these rig settings), so no mm (rule 6)",
     "uncalibrated"),
    ("GeometryMismatch: recipe_run: scan s was triangulated with calibration a, but the "
     "current one for g is b", "stale_calibration"),
    ("ambiguous end faces: the discs at camera px (1, 2) pass every gate", "ambiguous_faces"),
    ("the zone could not be resolved on this scan", "zone_unresolved"),
    ("only 12 points in the zone (need 30)", "too_few_points"),
    ("only 40 % of the zone measured (< 70 %): a partly measured face is never passed flat",
     "too_few_points"),
    ("placing the part on the golden (ICP) did not converge: can't be trusted", "incomplete"),
    ("INCOMPLETE: not every golden item was measured", "incomplete"),
    ("something nobody planned for", "invalid"),
])
def test_invalid_messages_are_classified(message, reason):
    assert D.classify(message) == reason


def _pass_run(tols=None, pct=0.0, **check):
    """A one-check PASS run with these tolerances (or a plugin's key / limit) -> settled."""
    ch = {"id": "c", "verdict": "PASS", "message": "ok", "pull": None, "tolerances": tols,
          "values": {}, "key": None, "limit": None, **check}
    res = {"result": "PASS", "invalid_reason": None, "pull_reasons": [],
           "zones": [{"id": "z", "verdict": "PASS", "checks": [ch]}]}
    return D.settle(res, guard_band_pct=pct)


def test_a_pass_its_own_numbers_do_not_hold_up_is_doubt():
    """A PASS whose value is missing, not a finite number or outside its own limits (the
    verdict and the number disagree: a bug, a rounded copy) is PULLed, guard band or not."""
    T = D.tolerance
    for tols in ([T("gap_mm", None, hi=0.5)], [T("gap_mm", float("nan"), hi=0.5)],
                 [T("gap_mm", "0.2", hi=0.5)], [T("gap_mm", 0.7, hi=0.5)],
                 [T("height_mm", 37.0, 37.1, 40.1)]):
        for pct in (0.0, 10.0):
            r = _pass_run(tols, pct)
            assert r["disposition"] == "PULL", (tols, pct)
            assert [q["reason"] for q in r["pull_reasons"]] == ["invalid"], r["pull_reasons"]
            assert "disagree" in r["pull_reasons"][0]["message"]
    r = _pass_run([{"name": "x", "value": 1}])                     # malformed: never ignored
    assert r["disposition"] == "PULL" and "malformed" in r["pull_reasons"][0]["message"]
    # a plugin without tolerances: its key / limit (|value| <= limit) NaN, missing or past it
    for v in (float("nan"), None, 0.7, -0.7, "x"):
        r = _pass_run(None, 10.0, key="gap_mm", limit=0.5, values={"gap_mm": v})
        assert r["disposition"] == "PULL", v
    assert _pass_run(None, 10.0, key="gap_mm", limit=0.5,
                     values={"gap_mm": 0.1})["disposition"] == "CONTINUE"
    # nominal 0.7 +- 0.2: hi is 0.8999999999999999 in float. The value judged (0.89996)
    # PASSes inside the guard band; its 4-decimal copy (0.9) reads past the limit: either
    # way a PULL, never a CONTINUE
    lo, hi = 0.7 - 0.2, 0.7 + 0.2
    r = _pass_run([T("height_mm", 0.89996, lo, hi)], 10.0)
    assert [q["reason"] for q in r["pull_reasons"]] == ["guard_band"]
    assert _pass_run([T("height_mm", round(0.89996, 4), lo, hi)], 10.0)["disposition"] == "PULL"
    # run-level tolerances (a station metric job): the same rules
    res = {"result": "PASS", "invalid_reason": None, "zones": [], "pull_reasons": []}
    D.settle(res, 10.0, tolerances=[T("deviation_mm.max", None, hi=0.5)])
    assert res["disposition"] == "PULL" and res["pull_reasons"][0]["reason"] == "invalid"


def test_guard_band_size_bounded_shares_and_limits_at_zero():
    T = D.tolerance
    # part present >= 92 % of a share that can't pass 100 %: 10 % of the room above the
    # limit (span 8 -> band 0.8), not 10 % of 92 (9.2: every good part pulled)
    assert _hits([T("part_pct", 98.91, lo=92.0, span=8.0)]) == []
    assert _hits([T("part_pct", 92.7, lo=92.0, span=8.0)]) == ["lo"]
    assert _hits([T("part_pct", 98.91, lo=92.0)]) == ["lo"]        # no span: |limit|
    # a band that comes out 0 clears nothing: on a limit at 0, a zero-width tolerance, a
    # one-sided limit at 0 with no span
    for t in (T("x_mm", 0.0, hi=0.0), T("x_mm", 1.5, 1.5, 1.5), T("x_mm", -3.0, hi=0.0),
              T("x_pct", 4.0, lo=0.0)):
        assert len(D.guard_band(D.check_tolerances([t]), 10)) == 1, t
    (h,) = D.guard_band(D.check_tolerances([T("x_mm", -3.0, hi=0.0)]), 10)
    assert h["unsized"] and "no size" in D._gb_reason(h, 10)["message"]
    assert _hits([T("x_mm", -3.0, hi=0.0, span=5.0)]) == []         # sized by its span
    assert _hits([T("x_mm", -0.3, hi=0.0, span=5.0)]) == ["hi"]
    assert _hits([T("x_mm", 0.0, hi=0.0)], 0) == []                 # 0 % is still off
    for bad in (-1, float("nan"), "5"):
        with pytest.raises(ValueError, match="span"):
            D.check_tolerances([T("x", 1.0, hi=2.0, span=bad)])
    # presence judged on 33.3333 % (exactly a third) against 33.333: the value judged is in
    # the band; its 2-decimal copy 33.33 would read below the limit - either way not CONTINUE
    third = 100.0 / 3.0
    assert _hits([T("part_pct", third, lo=33.333, span=100 - 33.333)]) == ["lo"]
    assert _pass_run([T("part_pct", round(third, 2), lo=33.333)], 10.0)["disposition"] == "PULL"


def test_a_refused_or_crashed_run_is_never_rejected():
    """refused / error are recorded as such (a crash text is never keyword-classified) and
    a run that never finished is never a REJECT, whatever the line rejects."""
    for code in ("refused", "error"):
        for msg in ("TypeError: only integer scalar arrays can be converted to a scalar index",
                    "RuntimeError: partial write"):
            res = {"result": "INVALID", "invalid_reason": msg, "zones": [], "pull_reasons": []}
            D.add(res, code, msg)
            assert D.settle(res, reject=D.REJECTABLE)["disposition"] == "PULL", (code, msg)
    # crashed after a zone's unreadable code: still PULL for a line rejecting code_unreadable
    res = {"result": "INVALID", "invalid_reason": "RuntimeError: partial write",
           "zones": [_z("a", ("c", "FAIL", "0 of 1", "code_unreadable", []))],
           "pull_reasons": []}
    D.add(res, "error", "RuntimeError: partial write")
    assert D.settle(res, reject=("code_unreadable", "incomplete"))["disposition"] == "PULL"
    # a run-level FAIL with a recorded doubt (a metric FAIL on an unconverged placement)
    res = {"result": "FAIL", "invalid_reason": None, "zones": [], "pull_reasons": []}
    D.add(res, "incomplete", "FAIL on numbers that can't be trusted: ICP did not converge")
    assert D.settle(res)["disposition"] == "PULL"
    assert D.settle(json.loads(json.dumps(res)))["disposition"] == "PULL"     # idempotent
    res = {"result": "FAIL", "invalid_reason": None, "zones": [], "pull_reasons": []}
    assert D.settle(res)["disposition"] == "REJECT" and res["pull_reasons"] == []


def _z(zid, *checks):
    return {"id": zid, "verdict": "x", "checks": [
        {"id": cid, "verdict": v, "message": m, "pull": pull, "tolerances": tols,
         "values": {}, "key": None, "limit": None}
        for cid, v, m, pull, tols in checks]}


def test_precedence_reject_beats_doubt_and_doubt_never_continues():
    fail = ("h", "FAIL", "height out", None, [])
    inval = ("f", "INVALID", "only 3 points in the zone", None, None)
    unread = ("c", "FAIL", "0 of 1 matching code(s)", "code_unreadable", [])
    near = ("h", "PASS", "ok", None, [D.tolerance("height_mm", 40.0, 37.1, 40.1)])
    ok = ("h", "PASS", "ok", None, [D.tolerance("height_mm", 38.6, 37.1, 40.1)])

    def run(result, zones, pct=10.0, reject=()):
        res = {"result": result, "invalid_reason": None, "zones": zones, "pull_reasons": []}
        return D.settle(res, guard_band_pct=pct, reject=reject)

    r = run("FAIL", [_z("a", fail), _z("b", inval)])          # a definite FAIL wins
    assert r["disposition"] == "REJECT"
    assert [(p["zone"], p["check"], p["reason"]) for p in r["pull_reasons"]] == [
        ("b", "f", "too_few_points")]                            # the doubt is still listed
    assert run("INVALID", [_z("a", ok), _z("b", inval)])["disposition"] == "PULL"
    assert run("FAIL", [_z("a", unread)])["disposition"] == "PULL"       # unreadable != wrong
    assert run("FAIL", [_z("a", unread), _z("b", fail)])["disposition"] == "REJECT"
    assert run("PASS", [_z("a", near)])["disposition"] == "PULL"         # guard band
    assert run("PASS", [_z("a", near)], pct=0)["disposition"] == "CONTINUE"
    assert run("PASS", [_z("a", ok)])["disposition"] == "CONTINUE"
    r = run("INVALID", [_z("a", ok), _z("b", inval)], reject=("too_few_points",))
    assert r["disposition"] == "REJECT"                        # the line rejects that doubt
    assert run("INVALID", [_z("b", inval)], reject=("golden",))["disposition"] == "PULL"
    ab = {"result": "ABORTED", "invalid_reason": None, "zones": [], "pull_reasons": []}
    assert D.settle(ab, reject=D.REJECTABLE)["disposition"] == "PULL"
    assert ab["pull_reasons"][0]["reason"] == "aborted"
    # idempotent; and a result is never written as CONTINUE unless it is a clean PASS
    again = json.loads(json.dumps(r))
    assert D.settle(again, guard_band_pct=10, reject=("too_few_points",)) == r
    for bad in ({"result": "INVALID", "disposition": "CONTINUE", "invalid_reason": "only 3 "
                 "points"}, {"result": "PASS", "disposition": "CONTINUE",
                             "pull_reasons": [D.reason("guard_band")]},
                {"result": "PASS", "disposition": "MAYBE"}, {"result": "FAIL"}):
        d = D.ensure(dict(bad))
        assert d["disposition"] in ("PULL", "REJECT") and d["disposition"] != "CONTINUE"
    # who reads the register / the exit code
    assert D.register_key("PASS", "CONTINUE") == "PASS"
    assert D.register_key("PASS", "PULL") == "PULL" and D.register_key("FAIL", "PULL") == "PULL"
    assert D.register_key("INVALID", "PULL") == "INVALID"
    assert D.register_key("ABORTED", "PULL") == "ABORTED"
    assert D.register_key("INVALID", "REJECT") == "FAIL" and D.register_key(None, None) is None
    assert D.register_key("PASS", None) == "PULL"              # no disposition: never PASS
    assert [D.exit_code({"disposition": d}) for d in ("CONTINUE", "REJECT", "PULL", None)] == \
        [0, 1, 5, 5]
    assert D.of_result("PASS") == "CONTINUE" and D.of_result("FAIL") == "REJECT"
    assert D.of_result("PASS", "PULL") == "PULL" and D.of_result("FAIL", "CONTINUE") == "PULL"
    assert D.of_result("PASS", "bogus") == "PULL" and D.of_result("PASS", error=True) == "PULL"
    assert D.of_result("ABORTED", "CONTINUE") == "PULL" and D.of_result("INVALID") == "PULL"


# ================================================================== config validation

def _recipe(**top):
    d = json.loads((RECIPES / "box_lid.json").read_text(encoding="utf-8"))
    d.update(top)
    return d


@pytest.mark.parametrize("pull, msg", [
    ({"guard_band_pct": -1}, "guard_band_pct"),
    ({"guard_band_pct": 100}, "guard_band_pct"),
    ({"guard_band_pct": 150}, "guard_band_pct"),
    ({"guard_band_pct": float("nan")}, "guard_band_pct"),
    ({"guard_band_pct": float("inf")}, "guard_band_pct"),
    ({"guard_band_pct": "10"}, "guard_band_pct"),
    ({"guard_band_pct": True}, "guard_band_pct"),
    ({"invalid_reasons_to_reject": ["aborted"]}, "never reject"),
    ({"invalid_reasons_to_reject": ["guard_band"]}, "never reject"),
    ({"invalid_reasons_to_reject": ["nope"]}, "invalid_reasons_to_reject"),
    ({"invalid_reasons_to_reject": "too_few_points"}, "invalid_reasons_to_reject"),
    ({"invalid_reasons_to_reject": [1]}, "invalid_reasons_to_reject"),
    ({"guard_band": 10}, "unknown keys"),
    ("ten", "pull: must be a mapping"),
])
def test_recipe_pull_config_is_validated(pull, msg):
    with pytest.raises(schema.RecipeError, match=msg):
        schema.validate(_recipe(pull=pull))


@pytest.mark.parametrize("codes, msg", [
    ({"PASS": 1, "FAIL": 2, "INVALID": 3, "ABORTED": 4, "PULL": 1}, "PASS code"),
    ({"PASS": 1, "FAIL": 2, "INVALID": 1, "ABORTED": 4}, "PASS code"),
    ({"PASS": 1, "FAIL": 2, "INVALID": 3, "ABORTED": 4, "PULL": 2}, "PULL 2 must differ"),
    ({"PASS": 1, "FAIL": 2, "INVALID": 3, "ABORTED": 4, "PULL": 3}, "PULL 3 must differ"),
    ({"PASS": 1, "FAIL": 2, "INVALID": 3, "ABORTED": 4, "PULL": 0}, "PULL 0 must differ"),
    ({"PASS": 1, "FAIL": 2, "INVALID": 3, "ABORTED": 4, "PULL": "5"}, "result_codes"),
])
def test_result_codes_never_give_a_doubt_the_pass_code(codes, msg):
    with pytest.raises(schema.RecipeError, match=msg):
        schema.validate(_recipe(outputs={"result_codes": codes}))


def test_recipe_pull_defaults_and_the_builtins_guard_band():
    d = _recipe()
    d.pop("pull")
    r = schema.validate(d)
    assert r.pull == {"guard_band_pct": 0.0, "invalid_reasons_to_reject": []}
    r = schema.validate(_recipe(outputs={"result_codes": {"PASS": 10, "FAIL": 20, "INVALID": 30,
                                                          "ABORTED": 40}}))
    assert r.outputs["result_codes"]["PULL"] == 5                  # older codes: the default
    for codes, pull in (({"PASS": 1, "FAIL": 5, "INVALID": 3, "ABORTED": 4}, 6),
                        ({"PASS": 5, "FAIL": 6, "INVALID": 7, "ABORTED": 8}, 9),
                        ({"PASS": 1, "FAIL": 2, "INVALID": 5, "ABORTED": 4}, 6)):
        r = schema.validate(_recipe(outputs={"result_codes": codes}))  # valid before PULL was
        assert r.outputs["result_codes"]["PULL"] == pull, codes       # a code of its own
    r = schema.validate(_recipe(pull={"guard_band_pct": 2.5, "invalid_reasons_to_reject": [
        "code_unreadable", "too_few_points", "code_unreadable"]}))
    assert r.pull == {"guard_band_pct": 2.5,
                      "invalid_reasons_to_reject": ["code_unreadable", "too_few_points"]}
    assert schema.validate(r.to_dict()).to_dict() == r.to_dict()  # round-trips with `pull`
    assert r.feedback["pull_colour"] == list(AMBER)
    for p in sorted(RECIPES.glob("*.json")):
        b = schema.load(p)
        assert b.pull["guard_band_pct"] == 10.0, p.name
        assert b.pull["invalid_reasons_to_reject"] == [] and b.outputs["result_codes"]["PULL"] == 5


def _station(tmp_path, job_over=None, **top) -> Path:
    doc = json.loads((ROOT / "stations" / "example_metric_check.json").read_text())
    doc.update(runs_dir=str(tmp_path / "runs"), mock_library=None, **top)
    doc["job"].update(job_over or {})
    f = tmp_path / f"st_{len(list(tmp_path.glob('st_*.json')))}.json"
    f.write_text(json.dumps(doc), encoding="utf-8")
    return f


def test_station_guard_band_is_validated(tmp_path):
    assert RJ.load_station_config(_station(tmp_path)).guard_band_pct == 0.0
    assert RJ.load_station_config(_station(tmp_path, {"guard_band_pct": 12.5})).guard_band_pct \
        == 12.5
    for bad in (-1, 100, 250, "10", True, float("nan"), float("inf")):
        with pytest.raises(RJ.StationConfigError, match="job.guard_band_pct"):
            RJ.load_station_config(_station(tmp_path, {"guard_band_pct": bad}))
    # a guard band that would do nothing is refused, never ignored
    rec = json.loads((ROOT / "stations" / "example_recipe_box_lid.json").read_text())
    rec.update(mock_library=None, runs_dir=str(tmp_path / "runs"))
    rec["job"]["guard_band_pct"] = 10
    f = tmp_path / "rec.json"
    f.write_text(json.dumps(rec), encoding="utf-8")
    with pytest.raises(RJ.StationConfigError, match="pull.guard_band_pct"):
        RJ.load_station_config(f)
    rec["job"]["guard_band_pct"] = None
    f.write_text(json.dumps(rec), encoding="utf-8")
    assert RJ.load_station_config(f).guard_band_pct == 0.0


# ================================================================== recipe runs (synthetic scans)

@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    """A mock library with the bench's true stereo calibration at bench_9in_metric and
    synthetic scans of the box lid (good, ajar)."""
    root = tmp_path_factory.mktemp("disposition") / "library"
    s = api.configure(root, mock=True)
    try:
        p = api.load_preset("bench_9in_metric")
        scans = {"good": RS.oracle_scan(s, p, "scene:box_lid", "lid_good").id,
                 "ajar": RS.oracle_scan(s, p, "scene:box_lid:ajar", "lid_ajar", seed=1).id}
    finally:
        api._session = None
    return root, scans


@pytest.fixture
def eng(lib):
    s = api.configure(lib[0], mock=True)
    yield s, lib[1]
    api._session = None


def _lid(nominal=40.0, text=True, **top) -> dict:
    """box_lid (guard band 10 %) with lid_right's height nominal moved: the measured 40.0 mm
    sits that far from the band's limits (half width 1.5 mm: guard band 0.15 mm)."""
    d = _recipe(name="lid_t", **top)
    d["zones"][1]["checks"][1]["params"]["nominal_mm"] = nominal
    d["feedback"]["text"] = text
    return d


def _run(recipe, scan, tmp_path, **kw):
    kw.setdefault("project_feedback", False)
    out = api.recipe_run(recipe, scan=scan, runs_dir=tmp_path / "runs", **kw)
    run_dir = Path(out["run_dir"])
    res = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    ev = [json.loads(x) for x in (run_dir / "run.log.jsonl").read_text(encoding="utf-8")
          .splitlines()]
    assert res == json.loads(json.dumps(out["result"], default=str))
    end = ev[-1]
    assert end["event"] == "run_end" and end["disposition"] == res["disposition"]
    assert end["pull_reasons"] == res["pull_reasons"] and end["exit_code"] == out["exit_code"]
    return out, res, run_dir, ev


def test_continue_and_reject_with_their_codes_and_fields(eng, tmp_path):
    s, scans = eng
    out, res, _, ev = _run("box_lid", scans["good"], tmp_path)
    assert res["schema_version"] == "0.3" and res["result"] == "PASS"
    assert res["disposition"] == "CONTINUE" and res["pull_reasons"] == [] and out["exit_code"] == 0
    assert res["recipe"]["result_code"] == 1 and res["recipe"]["pull"]["guard_band_pct"] == 10
    verdict = next(e for e in ev if e["event"] == "verdict")
    assert verdict["disposition"] == "CONTINUE"
    h = res["zones"][0]["checks"][1]
    (t,) = h["tolerances"]                  # the value judged, not its rounded copy
    assert (t["name"], t["lo"], t["hi"]) == ("height_mm", 38.5, 41.5) and h["pull"] is None
    assert t["value"] == pytest.approx(h["values"]["height_mm"], abs=5e-5)
    out, res, _, _ = _run("box_lid", scans["ajar"], tmp_path)
    assert res["result"] == "FAIL" and res["disposition"] == "REJECT" and out["exit_code"] == 1
    assert res["recipe"]["result_code"] == 2 and res["feedback"]["text"].startswith("FAIL")
    assert res["feedback"]["crosses"]


@pytest.mark.parametrize("nominal, disp, side", [
    (38.63, "PULL", "upper"),           # hi 40.13: 0.13 mm from the limit, inside 0.15
    (38.67, "CONTINUE", None),          # hi 40.17: 0.17 mm, just outside the guard band
    (41.37, "PULL", "lower"),           # lo 39.87: 0.13 mm
    (41.33, "CONTINUE", None),          # lo 39.83: 0.17 mm
    (38.40, "REJECT", None),            # hi 39.90: past the limit, the check FAILs
])
def test_guard_band_just_inside_and_outside_both_limits(eng, tmp_path, nominal, disp, side):
    s, scans = eng
    out, res, _, _ = _run(_lid(nominal), scans["good"], tmp_path)
    assert res["disposition"] == disp, res["pull_reasons"]
    assert out["exit_code"] == D.EXIT_CODES[disp]
    if disp == "PULL":
        assert res["result"] == "PASS"                       # the verdict is unchanged
        (p,) = res["pull_reasons"]
        assert (p["zone"], p["check"], p["reason"]) == ("lid_right", "height", "guard_band")
        assert side in p["message"] and "10 %" in p["message"]
        assert res["recipe"]["result_code"] == 5             # the PULL code, never PASS 1
    elif disp == "CONTINUE":
        assert res["result"] == "PASS" and res["pull_reasons"] == []
    else:
        assert res["result"] == "FAIL" and res["recipe"]["result_code"] == 2


def _amber(img):
    return (img[..., 2] > 200) & (img[..., 1] > 150) & (img[..., 1] < 230) & (img[..., 0] < 60)


def _red(img):
    return (img[..., 2] > 120) & (img[..., 1] < 60)


def _green(img):
    return (img[..., 1] > 120) & (img[..., 2] < 60)


def test_pull_paints_an_amber_part_frame_and_the_zone_that_caused_it(eng, tmp_path):
    s, scans = eng
    out, res, run_dir, ev = _run(_lid(38.63), scans["good"], tmp_path, project_feedback=True,
                                 hold_s=0.1)
    fb = res["feedback"]
    assert res["disposition"] == "PULL" and fb["rendered"] and fb["disposition"] == "PULL"
    assert fb["text"] == "PULL lid_right" and fb["crosses"] == []
    zones = {z["id"]: z for z in fb["zones"]}
    assert zones["lid_right"]["drawn"] and zones["lid_right"]["shown"] == "PULL"
    assert zones["lid_left"]["drawn"] and "shown" not in zones["lid_left"]
    img = FB.read_png(run_dir / "feedback_dmd.png")
    c0, r0, c1, r1 = fb["part_frame"]["dmd"]
    assert tuple(img[r0, (c0 + c1) // 2]) == AMBER and tuple(img[r1, (c0 + c1) // 2]) == AMBER
    assert tuple(img[(r0 + r1) // 2, c0]) == AMBER and tuple(img[(r0 + r1) // 2, c1]) == AMBER
    assert not _red(img).any()                               # a doubt is not a defect
    assert _green(img).sum() > 200 and _amber(img).sum() > 2000
    # on the part: what the mock scene's surfaces see of the DMD image
    from engine.capture import mock_scenes
    g = RS.geometry(s.bench.rig_model(), mock_scenes.shapes("scene:box_lid"))
    r = np.clip(np.round(np.nan_to_num(g["v"])), 0, 1139).astype(int)
    c = np.clip(np.round(np.nan_to_num(g["u"])), 0, 911).astype(int)
    seen = img[r, c]
    seen[~g["lit"]] = 0
    assert _amber(seen).sum() > 200 and not _red(seen).any()
    shown = [e for e in ev if e["event"] == "feedback_shown"]
    assert shown and shown[0]["held_s"] == pytest.approx(0.1)   # projected (mock), raw colour
    # the camera overlay says it too; the bus would show PULL, never PASS
    assert (run_dir / "overlay.png").is_file()


def test_an_invalid_zone_pulls_unless_a_definite_fail_rejects(eng, tmp_path):
    s, scans = eng
    ghost = {"id": "ghost", "shape": "rect", "units": "part_mm", "coords": [400, 300, 420, 320],
             "checks": [{"type": "flatness", "params": {"tol_mm": 0.5}}]}
    d = _lid(text=False)
    d["zones"].append(ghost)
    out, res, run_dir, _ = _run(d, scans["good"], tmp_path)
    assert res["result"] == "INVALID" and res["disposition"] == "PULL" and out["exit_code"] == 5
    (p,) = res["pull_reasons"]
    assert p["zone"] == "ghost" and p["check"] == "flatness"
    assert p["reason"] in ("zone_unresolved", "too_few_points"), p
    assert res["recipe"]["result_code"] == 3                     # INVALID's own code
    fb = res["feedback"]
    assert fb["frame"]["verdict"] == "PULL" and fb["part_frame"]["dmd"]   # ghost not drawn
    img = FB.read_png(run_dir / "feedback_dmd.png")
    assert tuple(img[0, 0]) == AMBER and not _red(img).any()
    # the same doubt next to a definite FAIL: REJECT (and the doubt is still listed)
    out, res, _, _ = _run(d, scans["ajar"], tmp_path)
    assert res["result"] == "FAIL" and res["disposition"] == "REJECT" and out["exit_code"] == 1
    assert any(q["zone"] == "ghost" for q in res["pull_reasons"])
    # a line that rejects that doubt: REJECT, the FAIL code on the register
    d["pull"] = {"guard_band_pct": 10, "invalid_reasons_to_reject": [p["reason"]]}
    out, res, run_dir, _ = _run(d, scans["good"], tmp_path)
    assert res["result"] == "INVALID" and res["disposition"] == "REJECT"
    assert out["exit_code"] == 1 and res["recipe"]["result_code"] == 2
    assert res["feedback"]["frame"]["verdict"] == "FAIL"        # never an amber-only REJECT
    img = FB.read_png(run_dir / "feedback_dmd.png")
    assert tuple(img[0, 0]) == tuple(schema.COLOURS["red"])


def test_a_crashed_check_and_an_unreadable_code_pull(eng, tmp_path):
    s, scans = eng

    class Boom(Check):
        id = "boom_pull"

        def run(self, ctx):
            raise RuntimeError("kaboom")
    recipes.register(Boom)
    try:
        d = _lid(text=False)
        d["zones"][0]["checks"].append({"type": "boom_pull"})
        out, res, _, _ = _run(d, scans["good"], tmp_path)
        assert res["result"] == "INVALID" and res["disposition"] == "PULL"
        assert out["exit_code"] == 5
        (p,) = res["pull_reasons"]
        assert (p["zone"], p["check"], p["reason"]) == ("lid_left", "boom_pull", "check_crashed")
    finally:
        recipes.unregister("boom_pull")
    # an expected code that is not there to read: FAIL, but PULL (unreadable is not wrong)
    d = _lid(text=False)
    d["zones"][0]["checks"].append({"type": "code", "params": {"expect": "LOT-7"}})
    out, res, run_dir, _ = _run(d, scans["good"], tmp_path)
    code = next(c for c in res["zones"][0]["checks"] if c["type"] == "code")
    assert code["verdict"] == "FAIL" and code["pull"] == "code_unreadable"
    assert res["result"] == "FAIL" and res["disposition"] == "PULL" and out["exit_code"] == 5
    assert [p["reason"] for p in res["pull_reasons"]] == ["code_unreadable"]
    assert res["recipe"]["result_code"] == 5 and res["feedback"]["crosses"] == []
    zones = {z["id"]: z for z in res["feedback"]["zones"]}
    assert zones["lid_left"]["shown"] == "PULL"
    assert not _red(FB.read_png(run_dir / "feedback_dmd.png")).any()


def test_code_check_wrong_text_is_definite_unreadable_is_doubt():
    import cv2
    from test_recipes_mock import doc, one, view, zone
    qr = cv2.QRCodeEncoder.create().encode("PART-42")
    qr = cv2.resize(qr, None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST).astype(np.float32)
    sv = view(lambda u, v: 0.0)
    q = np.pad(qr, 24, constant_values=255)
    sv.flood_white[40:40 + q.shape[0], 120:120 + q.shape[1]] = q
    z = (110, 30, 130 + q.shape[1], 50 + q.shape[0])
    wrong = one(doc([zone("c", [{"type": "code", "params": {"expect": "PART-43", "kind": "QR"}}],
                          units="camera_px", coords=z)]), sv)
    assert wrong.verdict == "FAIL" and wrong.pull is None             # a wrong code: REJECT
    other = one(doc([zone("c", [{"type": "code", "params": {"kind": "DATA_MATRIX"}}],
                          units="camera_px", coords=z)]), sv)
    assert other.verdict == "FAIL" and other.pull == "code_unreadable"   # none of that kind read
    blank = view(lambda u, v: 0.0)
    none = one(doc([zone("c", [{"type": "code"}], units="camera_px", coords=z)]), blank)
    assert none.verdict == "FAIL" and none.pull == "code_unreadable"
    assert "unreadable" in none.message and none.fault_points[0]["label"] == "code unreadable"


def test_uncalibrated_golden_stale_rig_check_and_aborted_are_pull(tmp_path, monkeypatch):
    import threading as th
    from engine import config
    from engine.calibrate.stereo import StereoCalibration
    s = api.configure(tmp_path / "lib", mock=True)
    out, res, _, _ = _run("box_lid", None, tmp_path)                  # no calibration
    assert res["disposition"] == "PULL" and out["exit_code"] == 5
    assert [p["reason"] for p in res["pull_reasons"]] == ["uncalibrated"]
    p = api.load_preset("bench_9in_metric")
    ms = RS.oracle_scan(s, p, "scene:box_lid", "lid_old")
    tube = RS.oracle_scan(s, p, "scene:tube", "tube_nogold")
    out, res, _, _ = _run("ribbed_tube_alignment", tube.id, tmp_path)     # no golden
    assert res["result"] == "INVALID" and res["disposition"] == "PULL" and out["exit_code"] == 5
    assert {q["reason"] for q in res["pull_reasons"]} == {"golden"}
    # the rig check before the cycle says MOVED / UNKNOWN: never scanned, PULL
    for status in ("MOVED", "UNKNOWN"):
        monkeypatch.setattr(api, "rig_check", lambda preset, st=status: {
            "status": st, "message": "camera moved ~2.3 px"})
        out, res, _, _ = _run(_lid(scan={"kind": "fast", "axis": "xy", "rig_check": True}),
                              None, tmp_path)
        assert res["disposition"] == "PULL" and out["exit_code"] == 5
        assert res["part"]["scan_id"] is None
        assert [q["reason"] for q in res["pull_reasons"]] == ["rig_check"]
    # ABORTED mid-cycle: exit 1 as documented, disposition PULL, the ABORTED register code
    s.cancel = th.Event()
    s.cancel.set()
    try:
        out, res, _, _ = _run("box_lid", ms.id, tmp_path)
    finally:
        s.cancel = None
    assert res["result"] == "ABORTED" and res["disposition"] == "PULL" and out["exit_code"] == 1
    assert res["pull_reasons"][0]["reason"] == "aborted" and res["recipe"]["result_code"] == 4
    # a newer stereo calibration: the stored scan's mm are stale -> refused (2), PULL
    tr = s.bench.stereo_truth()
    d = s.store.new_dir("calibrations", p.geometry_id, label="stereo")
    cal = StereoCalibration(id=f"{p.geometry_id}/{d.name}", geometry_id=p.geometry_id,
                            K_cam=tr["K_cam"], dist_cam=tr["dist_cam"], K_proj=tr["K_proj"],
                            dist_proj=np.zeros(5), R=tr["R"], t=np.ravel(tr["t"]),
                            diamond_parity=tr["parity"],
                            fingerprint=config.geometry_fingerprint(p), path=d,
                            created="2026-01-02T00:00:00")
    s.store.write_json(d / "stereo_calibration.json", cal.to_json())
    out, res, _, _ = _run("box_lid", ms.id, tmp_path)
    assert res["result"] == "INVALID" and out["exit_code"] == 2, res["invalid_reason"]
    assert res["disposition"] == "PULL"            # refused: recorded as such, never a reject
    assert [q["reason"] for q in res["pull_reasons"]] == ["refused"]
    assert "calibration" in res["pull_reasons"][0]["message"]


def test_presence_guard_band_is_a_share_of_the_room_above_its_limit(eng, tmp_path):
    """part present >= min_area_pct of a share that can't pass 100 %: a good lid (~99 %)
    with min_area_pct 92 is no longer pulled (10 % of 92 = 9.2 put every good part in the
    band); one just over its limit still is."""
    s, scans = eng

    def lid(min_area):
        d = _lid(text=False)
        for z in d["zones"]:
            for c in z["checks"]:
                if c["type"] == "presence":
                    c["params"]["min_area_pct"] = min_area
        return d
    out, res, _, _ = _run(lid(92.0), scans["good"], tmp_path)
    assert res["result"] == "PASS" and res["disposition"] == "CONTINUE", res["pull_reasons"]
    pres = [c for z in res["zones"] for c in z["checks"] if c["type"] == "presence"]
    (t,) = pres[0]["tolerances"]
    assert t["lo"] == 92.0 and t["span"] == pytest.approx(8.0) and t["value"] > 95
    near = round(min(c["tolerances"][0]["value"] for c in pres) - 0.05, 3)
    out, res, _, _ = _run(lid(near), scans["good"], tmp_path)
    assert res["result"] == "PASS" and res["disposition"] == "PULL" and out["exit_code"] == 5
    assert {q["reason"] for q in res["pull_reasons"]} == {"guard_band"}


def test_a_recipe_run_crashing_after_its_verdict_never_keeps_the_pass_code(eng, tmp_path,
                                                                           monkeypatch):
    """The runner settles before its pictures (result code 1 for a clean PASS); a crash
    after that is INVALID with INVALID's code, PULL with reason `error` - never classified
    from the crash text, never a reject even for a line that rejects that doubt."""
    from engine.recipes import runner
    s, scans = eng

    def boom(*a, **k):
        raise RuntimeError("partial write: only 3 points written")
    monkeypatch.setattr(runner, "_headline", boom)
    d = _lid(text=False)
    d["pull"] = {"guard_band_pct": 10, "invalid_reasons_to_reject": ["incomplete",
                                                                     "too_few_points"]}
    out, res, _, _ = _run(d, scans["good"], tmp_path)
    assert out["exit_code"] == 4 and res["result"] == "INVALID"
    assert res["disposition"] == "PULL" and res["recipe"]["result_code"] == 3
    assert [q["reason"] for q in res["pull_reasons"]] == ["error"]


def _recipe_station(tmp_path) -> Path:
    rec = json.loads((ROOT / "stations" / "example_recipe_box_lid.json").read_text())
    rec.update(mock_library=None, runs_dir=str(tmp_path / "runs"), mock_part=None)
    f = tmp_path / "rec_station.json"
    f.write_text(json.dumps(rec), encoding="utf-8")
    return f


def _only_run(tmp_path) -> tuple:
    (run_dir,) = [d for d in (tmp_path / "runs").iterdir() if d.is_dir()]
    res = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    ev = [json.loads(x) for x in (run_dir / "run.log.jsonl").read_text(encoding="utf-8")
          .splitlines()]
    return run_dir, res, ev


@pytest.mark.parametrize("what", ["crash", "abort"])
def test_station_recipe_run_after_its_verdict_never_keeps_the_pass_code(tmp_path, monkeypatch,
                                                                        what):
    """`scan run` (and the operator page's run_station) on a recipe work state: the
    runner's own settle already wrote the PASS code (1); a crash after it (a failed output
    listing) is INVALID / PULL with INVALID's code (3), an ABORT up to result.json is
    ABORTED / PULL with ABORTED's code (4). The --json error output names the disposition."""
    import threading as th
    from engine.recipes import runner

    def fake_execute(s, recipe, preset, *, res, log, run_dir, **kw):
        res["recipe"] = {**recipe.summary(), "tier": None, "result_code": None, "frame": None}
        res["result"] = res["engine_verdict"] = "PASS"
        res["invalid_reason"] = None
        runner.settle(res, recipe)                  # what execute does before its pictures
        assert res["disposition"] == "CONTINUE" and res["recipe"]["result_code"] == 1
        if what == "abort":                         # ABORT pressed after the verdict
            s.cancel = th.Event()
            s.cancel.set()
        return {"img": None}
    monkeypatch.setattr(runner, "execute", fake_execute)
    if what == "crash":
        def gone(self):
            raise OSError("the output bin is gone")
        monkeypatch.setattr(cli.Ctx, "new_outputs", gone)
    f = _recipe_station(tmp_path)
    code, out = _cli(["--mock", "--library", str(tmp_path / "lib"), "--json", "run", str(f)])
    run_dir, res, ev = _only_run(tmp_path)
    want = {"crash": ("INVALID", 4, 3, "error"), "abort": ("ABORTED", 1, 4, "aborted")}[what]
    assert (res["result"], code, res["recipe"]["result_code"]) == want[:3], out
    assert res["disposition"] == "PULL" and res["pull_reasons"][0]["reason"] == want[3]
    assert ev[-1]["event"] == "run_end" and ev[-1]["disposition"] == "PULL"
    doc = json.loads(out)
    assert doc["exit_code"] == code and doc.get("disposition") == "PULL", doc


def test_json_error_output_names_the_disposition(tmp_path, monkeypatch):
    """A refused / crashed run, station-check or face-check prints its disposition (PULL)
    in --json too; a crashed metric job's crash text is never classified."""
    lib = ["--mock", "--library", str(tmp_path / "lib"), "--json"]
    monkeypatch.setattr(cli, "_run_metric", _fake_metric(None, raise_=TypeError(
        "only integer scalar arrays can be converted to a scalar index")))
    code, out = _cli([*lib, "run", str(_station(tmp_path))])
    doc = json.loads(out)
    assert code == 4 and doc["disposition"] == "PULL" and doc["run_id"]
    assert [p["reason"] for p in doc["pull_reasons"]] == ["error"]
    from engine.measure.station.geometry import StationError
    monkeypatch.setattr(cli, "_station_item", lambda key: key)

    def no_face(*a):
        raise StationError("end face C2 not found in the zone")
    monkeypatch.setattr(api, "station_check", no_face)
    code, out = _cli([*lib, "station-check", "st7"])
    doc = json.loads(out)
    assert code == cli.EXIT_REFUSED and doc["disposition"] == "PULL"
    assert [p["reason"] for p in doc["pull_reasons"]] == ["refused"]
    code, out = _cli([*lib, "face-check", "no_such_placement"])
    doc = json.loads(out)
    assert code != 0 and doc["disposition"] == "PULL", doc


@pytest.fixture(scope="module")
def endfaces(tmp_path_factory):
    """A synthetic metric scan of the end-face scene with one dent (no capture)."""
    root = tmp_path_factory.mktemp("disposition_faces") / "library"
    s = api.configure(root, mock=True)
    try:
        p = api.load_preset("bench_9in_metric_bright")
        scan = RS.oracle_scan(s, p, "scene:endfaces:dent", "ef_dent", seed=3).id
    finally:
        api._session = None
    return root, scan


def test_face_mode_defect_reports_how_close_a_pass_came(endfaces, tmp_path):
    """face mode (MF-CFAR): a PASS reports the deepest significant dent under depth_mm,
    never a flat 0, so the guard band sees a dent just under the limit (it used to read
    max_depth_mm 0.0 and CONTINUE)."""
    api.configure(endfaces[0], mock=True)
    doc = json.loads((RECIPES / "cylinder_end_faces.json").read_text(encoding="utf-8"))
    doc["name"] = "ef_t"
    assert doc["pull"]["guard_band_pct"] == 10

    def run(depth=None):
        d = json.loads(json.dumps(doc))
        for z in d["zones"]:
            for c in z["checks"]:
                if c["type"] == "defect" and depth is not None:
                    c["params"]["depth_mm"] = depth
        return _run(d, endfaces[1], tmp_path)
    out, res, _, _ = run()
    dents = [abs(t["depth_mm"]) for z in res["zones"] for c in z["checks"]
             if c["type"] == "defect" for t in (c.get("details") or {}).get("triggers", [])]
    assert res["disposition"] == "REJECT" and dents, res["pull_reasons"]
    dent = max(dents)
    out, res, _, _ = run(round(dent * 1.03, 3))          # 3 % under the limit: in the band
    assert res["result"] == "PASS" and res["disposition"] == "PULL" and out["exit_code"] == 5
    (q,) = res["pull_reasons"]
    assert q["reason"] == "guard_band" and q["check"] == "defect"
    ch = next(c for z in res["zones"] if z["id"] == q["zone"] for c in z["checks"]
              if c["type"] == "defect")
    assert ch["values"]["max_depth_mm"] == pytest.approx(dent, abs=0.01)
    assert ch["tolerances"][0]["value"] == pytest.approx(dent, abs=0.01)
    out, res, _, _ = run(round(dent * 1.2, 3))           # well under the limit: CONTINUE
    assert res["disposition"] == "CONTINUE" and out["exit_code"] == 0, res["pull_reasons"]


def _cli(argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = cli.main(argv)
    api._session = None
    return code, buf.getvalue()


def test_recipe_run_cli_prints_the_disposition_and_exits_with_it(eng, tmp_path):
    s, scans = eng
    lib = str(s.store.root)
    api._session = None
    f = tmp_path / "lid_near.json"
    f.write_text(json.dumps(_lid(38.63)), encoding="utf-8")
    L = ["--mock", "--library", lib]
    for recipe, scan, want, word in (("box_lid", "good", 0, "disposition: CONTINUE"),
                                     ("box_lid", "ajar", 1, "disposition: REJECT"),
                                     (str(f), "good", 5, "PULL (quality inspection): "
                                                         "lid_right/height: guard_band")):
        code, out = _cli([*L, "recipe-run", recipe, "--scan", scans[scan], "--no-feedback",
                          "--runs-dir", str(tmp_path / "runs")])
        assert code == want and word in out, out
    code, out = _cli([*L, "--json", "recipe-run", str(f), "--scan", scans["good"],
                      "--no-feedback", "--runs-dir", str(tmp_path / "runs")])
    doc = json.loads(out)
    assert code == 5 and doc["exit_code"] == 5 and not doc["ok"]
    assert doc["result"]["result"] == "PASS" and doc["result"]["disposition"] == "PULL"


# ================================================================== station metric jobs

def _fake_metric(result, max_mm=None, reason=None, raise_=None, doubt=None):
    def fake(ctx, cfg, p, log, res, run_dir):
        if raise_ is not None:
            raise raise_
        if result in ("PASS", "FAIL"):
            res["deviation_units"] = "mm"
            res["deviation_mm"].update(max=max_mm, method="plane_fit")
            res["threshold_mm"] = cfg.threshold_mm
            res["result"] = res["engine_verdict"] = result
            res["invalid_reason"] = None
            if doubt:                               # what _run_metric records on such a FAIL
                D.add(res, "incomplete", f"FAIL on numbers that can't be trusted: {doubt}")
        else:
            res["invalid_reason"] = reason
    return fake


@pytest.mark.parametrize("fake, code, disp, reason", [
    (_fake_metric("PASS", 0.46), 5, "PULL", "guard_band"),        # inside 10 % of 0.5 mm
    (_fake_metric("PASS", 0.44), 0, "CONTINUE", None),             # just outside the band
    (_fake_metric("FAIL", 0.71), 1, "REJECT", None),
    (_fake_metric("INVALID", reason="only 12 points (0.1 % of the roi) decoded, below "
                                    "job.min_valid_pct 30 %"), 5, "PULL", "too_few_points"),
    (_fake_metric(None, raise_=rules.Cancelled("aborted by the operator")), 1, "PULL",
     "aborted"),
    (_fake_metric(None, raise_=rules.RuleViolation("refused for a test")), 2, "PULL",
     "refused"),
    (_fake_metric(None, raise_=RuntimeError("simulated crash")), 4, "PULL", "error"),
    (_fake_metric(None, raise_=RuntimeError("partial write: only 3 points")), 4, "PULL",
     "error"),                                     # a crash text is never classified
    (_fake_metric("FAIL", 0.9, doubt="ICP did not converge"), 5, "PULL", "incomplete"),
])
def test_station_metric_job_dispositions_and_exit_codes(tmp_path, monkeypatch, fake, code, disp,
                                                        reason):
    monkeypatch.setattr(cli, "_run_metric", fake)
    cfg = _station(tmp_path, {"guard_band_pct": 10})
    got, out = _cli(["--mock", "--library", str(tmp_path / "lib"), "run", str(cfg)])
    assert got == code, out
    (run_dir,) = [d for d in (tmp_path / "runs").iterdir() if d.is_dir()]
    res = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    assert res["schema_version"] == "0.3" and res["disposition"] == disp
    assert [p["reason"] for p in res["pull_reasons"]] == ([reason] if reason else [])
    end = json.loads((run_dir / "run.log.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert end["event"] == "run_end" and end["disposition"] == disp and end["exit_code"] == code
    assert f"disposition: {'PULL (quality inspection)' if disp == 'PULL' else disp}" in out
    if reason == "guard_band":
        assert res["result"] == "PASS" and "deviation_mm.max 0.46" in res["pull_reasons"][0][
            "message"]


# ================================================================== face / station checks

def test_face_and_station_checks_pull_incomplete_and_ambiguous_faces(tmp_path, monkeypatch):
    """face-check / station-check: INCOMPLETE and ambiguous end faces are PULL (exit 5), not
    FAIL (1); PASS 0 and FAIL 1 as before."""
    from types import SimpleNamespace
    from engine.measure.station.geometry import StationError
    lib = ["--mock", "--library", str(tmp_path / "lib"), "--json"]
    monkeypatch.setattr(cli, "_placement", lambda key: None)
    monkeypatch.setattr(cli, "_find_golden", lambda key: None)

    def face(verdict):
        return SimpleNamespace(placement={"label": "c7"}, verdict=verdict, n_triggers=0,
                               golden_id="g", detector="band", triggers=[], files={},
                               warnings=[], stats={"dev_rms_mm": 0.01, "max_abs_dev_mm": 0.02,
                                                   "dev_r_mm": 9.0, "compared_pct": 41.0})
    for verdict, code, disp in (("PASS", 0, "CONTINUE"), ("FAIL", 1, "REJECT"),
                                ("INCOMPLETE", 5, "PULL")):
        monkeypatch.setattr(api, "check_face", lambda *a, v=verdict: face(v))
        got, out = _cli([*lib, "face-check", "c7"])
        doc = json.loads(out)
        assert got == code and doc["disposition"] == disp, out
        if disp == "PULL":
            assert doc["pull_reasons"][0]["reason"] == "incomplete"
            assert "41.0 % compared" in doc["pull_reasons"][0]["message"]
    monkeypatch.setattr(cli, "_station_item", lambda key: key)

    def station(verdict, items):
        return SimpleNamespace(measurement={"scan_id": "st7"}, verdict=verdict, n_triggers=0,
                               golden_id="g", items=items, files={}, warnings=[])
    it_ok = {"item": "C2-C1", "value_mm": 0.1, "nominal_mm": 0.1, "band_mm": [0, 0.2],
             "status": "ok"}
    it_missing = {"item": "T3", "value_mm": None, "nominal_mm": -4.5, "band_mm": [-5, -4],
                  "status": "missing"}
    monkeypatch.setattr(api, "station_check",
                        lambda *a: station("INCOMPLETE", [it_ok, it_missing]))
    got, out = _cli([*lib, "station-check", "st7"])
    doc = json.loads(out)
    assert got == 5 and doc["disposition"] == "PULL" and "T3" in doc["pull_reasons"][0]["message"]
    monkeypatch.setattr(api, "station_check", lambda *a: station("PASS", [it_ok]))
    assert _cli([*lib, "station-check", "st7"])[0] == 0

    def ambiguous(*a):
        raise StationError("ambiguous end faces: the discs at camera px (10, 20) and (30, 20) "
                           "both pass every gate: which is the face?")
    monkeypatch.setattr(api, "station_check", ambiguous)
    got, out = _cli([*lib, "station-check", "st7"])
    doc = json.loads(out)
    assert got == 5 and doc["disposition"] == "PULL" and doc["check"] is None
    assert [p["reason"] for p in doc["pull_reasons"]] == ["ambiguous_faces"]

    def no_face(*a):
        raise StationError("end face C2 not found in the zone")
    monkeypatch.setattr(api, "station_check", no_face)
    assert _cli([*lib, "station-check", "st7"])[0] == cli.EXIT_REFUSED     # unchanged


# ================================================================== command bus

def _bus(info=None):
    bus = C.CommandBus(C.Handlers(), ["a.json", "b.json"])
    out = C.SimulatedOutput({"PULL": "DO5"})
    bus.outputs.append(out)
    if info:
        bus.update_workstates(["a.json", "b.json"], info=info)
    return bus, out


def _cycle(bus, result, disposition=None, error=None):
    assert bus.command("START").accepted
    bus.job_finished("run", result, error, disposition=disposition)
    return bus.signals(), bus.registers()["result_code"]


def test_bus_pull_output_and_result_code():
    bus, out = _bus()
    assert "PULL" in C.SIGNALS and C.RESULT_CODES["PULL"] == 5
    sig, code = _cycle(bus, "PASS", "CONTINUE")
    assert sig["PASS"] and not sig["PULL"] and code == 1
    sig, code = _cycle(bus, "PASS", "PULL")                    # guard band: never PASS
    assert sig["PULL"] and not sig["PASS"] and not sig["FAIL"] and code == 5
    assert out.current["DO5"] and not out.current["PASS"] and out.registers["result_code"] == 5
    assert bus.snapshot()["last_disposition"] == "PULL"
    sig, code = _cycle(bus, "FAIL", "PULL")                    # unreadable code: not a reject
    assert sig["PULL"] and not sig["FAIL"] and code == 5
    sig, code = _cycle(bus, "FAIL", "REJECT")
    assert sig["FAIL"] and not sig["PULL"] and code == 2
    sig, code = _cycle(bus, "INVALID", "PULL")
    assert sig["PULL"] and sig["ALARM"] and not sig["PASS"] and code == 3
    sig, code = _cycle(bus, "INVALID", "REJECT")               # a doubt the line rejects
    assert sig["FAIL"] and sig["ALARM"] and not sig["PULL"] and code == 2
    sig, code = _cycle(bus, "PASS", "bogus")                   # unknown: doubt
    assert sig["PULL"] and not sig["PASS"] and code == 5
    sig, code = _cycle(bus, "FAIL", "CONTINUE")                # does not fit the verdict
    assert sig["PULL"] and not sig["PASS"] and code == 5
    sig, code = _cycle(bus, None, error="crashed")
    assert sig["PULL"] and sig["ALARM"] and code == 3
    assert bus.command("START").accepted and bus.command("ABORT").accepted
    bus.job_finished("run", "PASS", disposition="CONTINUE")    # aborted: PULL whatever it says
    assert bus.last_result == "ABORTED" and bus.signals()["PULL"] and not bus.signals()["PASS"]
    assert bus.registers()["result_code"] == 4
    # while busy every result output is off; a never-started cycle (unusable work state) pulls
    bus.command("START")
    assert not any(bus.signals()[k] for k in ("PASS", "FAIL", "PULL"))
    bus.job_finished("run", "PASS")                            # no disposition: as the verdict
    assert bus.signals()["PASS"] and bus.registers()["result_code"] == 1
    bus.update_workstates(["a.json", "b.json"], unusable={"a.json": "config invalid"})
    assert not bus.command("START").accepted
    assert bus.signals()["PULL"] and bus.last_disposition == "PULL"


def test_bus_recipe_codes_renumbered_and_io_example_maps_pull():
    codes = {"PASS": 10, "FAIL": 20, "INVALID": 30, "ABORTED": 40, "PULL": 50}
    bus, _ = _bus({"a.json": {"recipe": "r", "result_codes": codes}})
    assert _cycle(bus, "PASS", "PULL")[1] == 50 and _cycle(bus, "PASS", "CONTINUE")[1] == 10
    old = {"PASS": 10, "FAIL": 20, "INVALID": 30, "ABORTED": 40}      # a code set without PULL
    bus, _ = _bus({"a.json": {"recipe": "r", "result_codes": old}})
    assert _cycle(bus, "PASS", "PULL")[1] == 30                       # never the PASS code
    doc = json.loads((ROOT / "stations" / "io_example.json").read_text(encoding="utf-8"))
    outs = {o["kind"]: o for o in doc["outputs"]}
    for kind in ("digital_io", "modbus_tcp", "opc_ua", "ethernet_ip"):
        assert "PULL" in outs[kind]["map"], kind
    assert "PULL" in doc["_comment"]
    for o in doc["outputs"]:
        o["enabled"] = o["kind"] == "simulated"
    o = next(o for o in doc["outputs"] if o["kind"] == "simulated")
    o["map"] = {"PULL": "lane_qi"}
    _, (sim,) = C.load_io_doc({"inputs": [], "outputs": [o]})
    bus = C.CommandBus(C.Handlers(), ["a.json"])
    bus.outputs.append(sim)
    _cycle(bus, "INVALID", "PULL")
    assert sim.current["lane_qi"] is True
    with pytest.raises(ValueError, match="unknown signals"):
        C.load_io_doc({"outputs": [{"kind": "simulated", "map": {"QI": "x"}}]})


def test_bus_home_ends_a_pass_and_a_failed_start_pulls():
    """A PASS belongs to the cycle that just ended: HOME (OK, or MOVED -> FAULT -> RESET)
    never hands its PASS output or PASS code back. FAIL / PULL stay. A START whose handler
    fails is INVALID / PULL (its code on the register; PULL once RESET)."""
    for home in ("OK", "MOVED"):
        bus, out = _bus()
        seen = []
        bus.outputs.append(type("Spy", (C.OutputBackend,), {
            "publish": lambda self, sig: seen.append(sig["PASS"]),
            "publish_registers": lambda self, regs: None})())
        _cycle(bus, "PASS", "CONTINUE")
        assert bus.signals()["PASS"] and bus.registers()["result_code"] == 1
        assert bus.command("HOME").accepted
        bus.job_finished("home", home)
        if home == "MOVED":
            assert bus.state == "FAULT" and bus.command("RESET").accepted
        assert bus.state == "READY" and not bus.signals()["PASS"], home
        assert bus.registers()["result_code"] == 0 and bus.last_result is None
        assert seen[-1] is False and seen.count(True) == 1, seen     # never on again
    bus, _ = _bus()
    _cycle(bus, "FAIL", "REJECT")
    bus.command("HOME")
    bus.job_finished("home", "OK")
    assert bus.signals()["FAIL"] and bus.registers()["result_code"] == 2   # releases nothing

    def refuse(ws, ok):
        raise RuntimeError("the job queue is closed")
    bus = C.CommandBus(C.Handlers(start=refuse), ["a.json"])
    r = bus.command("START")
    assert not r.accepted and bus.state == "FAULT"
    assert (bus.last_result, bus.last_disposition) == ("INVALID", "PULL")
    assert bus.registers()["result_code"] == 3 and not bus.signals()["PASS"]
    assert bus.command("RESET").accepted and bus.signals()["PULL"]


def _web_app(tmp_path):
    from ui.web import server as ui_server
    stations = tmp_path / "stations"
    stations.mkdir()
    (stations / "a.json").write_text(_station(tmp_path).read_text(), encoding="utf-8")
    return ui_server.App(mock=True, library=tmp_path / "lib", stations_dir=stations).start()


def _wait(cond, what):
    t0 = time.monotonic()
    while not cond():
        assert time.monotonic() - t0 < 30, what
        time.sleep(0.02)


def test_web_abort_after_the_runs_verdict_pulls_everywhere(tmp_path, monkeypatch):
    """ABORT pressed after the run wrote its verdict but before its job ended: the bus
    counts the cycle ABORTED (PULL, ABORTED's code); the job's result says PULL with an
    `aborted` reason, the run log records it, and no verdict is projected on the part."""
    import threading as th
    app = _web_app(tmp_path)
    go = th.Event()
    run_dir = tmp_path / "runs" / "r1"
    run_dir.mkdir(parents=True)
    (run_dir / "run.log.jsonl").write_text("", encoding="utf-8")

    def fake_cmd_run(ctx):                    # the run is past its last cancel check
        assert go.wait(30)
        return 0, {"run_dir": str(run_dir), "result": {
            "run_id": "r1", "result": "PASS", "disposition": "CONTINUE", "pull_reasons": [],
            "feedback": {"rendered": True, "enabled": True}, "warnings": []}}
    monkeypatch.setattr(cli, "cmd_run", fake_cmd_run)
    try:
        assert app.bus.command("START").accepted
        (jid,) = app.bus_jobs()
        _wait(lambda: app._jobs[jid].status == "running", "the run never started")
        assert app.bus.command("ABORT").accepted and app.bus.state == "ABORTING"
        go.set()
        _wait(lambda: app.bus.state == "READY", "the bus never settled")
        snap = app.cmd_state()
        assert snap["last_result"] == "ABORTED" and snap["last_disposition"] == "PULL"
        assert snap["signals"]["PULL"] and not snap["signals"]["PASS"]
        assert snap["registers"]["result_code"] == 4
        job = app._jobs[jid]
        _wait(lambda: (job.result or {}).get("disposition") == "PULL", "job result not PULL")
        assert job.result["line_result"] == "ABORTED" and "feedback_job" not in job.result
        assert job.result["pull_reasons"][-1]["reason"] == "aborted"
        assert not [j for j in app._jobs.values() if j.verb == "recipe_feedback"]
        ev = [json.loads(x) for x in (run_dir / "run.log.jsonl").read_text(
            encoding="utf-8").splitlines()]
        assert ev[-1]["event"] == "aborted_after_verdict" and ev[-1]["disposition"] == "PULL"
    finally:
        go.set()
        app.close()
        api._session = None


def test_web_bus_follows_the_runs_disposition(tmp_path):
    """The operator page's bus hears the run's own disposition (job result), not only its
    verdict: a guard-band PASS lights PULL and writes the PULL code, never PASS."""
    from ui.web import server as ui_server
    stations = tmp_path / "stations"
    stations.mkdir()
    (stations / "a.json").write_text(_station(tmp_path).read_text(), encoding="utf-8")
    app = ui_server.App(mock=True, library=tmp_path / "lib", stations_dir=stations).start()
    try:
        replies = iter([{"result": "PASS", "disposition": "PULL", "pull_reasons": [
                            D.reason("guard_band", "near the limit")]},
                        {"result": "FAIL", "disposition": "REJECT", "pull_reasons": []},
                        {"result": "PASS", "disposition": "CONTINUE", "pull_reasons": []}])
        app._v_run_station = lambda a: next(replies)
        for want, sig in (("PULL", "PULL"), ("REJECT", "FAIL"), ("CONTINUE", "PASS")):
            assert app.bus.command("START").accepted
            t0 = time.monotonic()
            while app.bus.state != "READY":
                assert time.monotonic() - t0 < 30
                time.sleep(0.02)
            snap = app.cmd_state()
            assert snap["last_disposition"] == want and snap["signals"][sig], snap
            others = {"PASS", "FAIL", "PULL"} - {sig}
            assert not any(snap["signals"][k] for k in others), snap["signals"]
            assert snap["registers"]["result_code"] == {"PULL": 5, "REJECT": 2,
                                                        "CONTINUE": 1}[want]
    finally:
        app.close()
        api._session = None


# ================================================================== /quick (node, fake DOM)

_LAMP_HARNESS = r"""
const fs = require('fs'), vm = require('vm');
class El {
  constructor(tag, id) { this.tagName = String(tag).toUpperCase(); this.id = id; this.children = [];
    this.attrs = {}; this.dataset = {}; this.className = ''; this._text = ''; this.hidden = false;
    this.disabled = false; this.checked = false; this.href = ''; this.src = '';
    this.classList = { toggle() {}, add() {}, remove() {} }; }
  set textContent(t) { this._text = String(t); this.children = []; }
  get textContent() { return this._text + this.children.map((c) => c.textContent).join(''); }
  appendChild(c) { this.children.push(c); return c; }
  addEventListener() {}
}
const byId = {};
const el = (id) => byId[id] || (byId[id] = new El('div', id));
const sandbox = { console, setTimeout, clearTimeout, setInterval, clearInterval,
  document: { addEventListener() {}, querySelector: (s) => el(s.slice(1)),
              querySelectorAll: () => [], createElement: (t) => new El(t),
              createTextNode: (t) => { const n = new El('#text'); n.textContent = t; return n; } },
  fetch: async () => ({ ok: true, status: 200, json: async () => ({}) }) };
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(process.argv[2], 'utf8'), sandbox, { filename: 'quick.js' });
const show = vm.runInContext('showRun', sandbox);
const out = {};
const cases = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
for (const [name, job] of Object.entries(cases)) {
  show(job);
  out[name] = { lamp: el('qLamp').textContent, cls: el('qLamp').className,
    disp: el('qDisposition').textContent, pull: el('qPull').textContent,
    pullHidden: el('qPull').hidden, verdict: el('qVerdict').textContent,
    zones: el('qZoneList').children.map((c) => c.className) };
}
process.stdout.write(JSON.stringify(out));
"""


def test_quick_lamp_shows_pull_in_its_own_amber_with_the_reasons(tmp_path):
    html = (STATIC / "quick.html").read_text(encoding="utf-8")
    css = (STATIC / "quick.css").read_text(encoding="utf-8")
    assert 'id="qPull"' in html and 'id="qDisposition"' in html and 'data-sig="PULL"' in html
    assert ".lamp-pull" in css and "#e8890c" in css               # amber, not the INVALID lamp
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed; quick.js logic not checked")
    pull = {"zone": "lid_right", "check": "height", "reason": "guard_band",
            "message": "height_mm 40 is within 10 % of the tolerance of its upper limit"}
    cases = {
        "pull": {"status": "done", "result": {"result": "PASS", "disposition": "PULL",
                                              "pull_reasons": [pull],
                                              "zones": [{"id": "lid_right", "verdict": "PASS",
                                                         "checks": []},
                                                        {"id": "lid_left", "verdict": "PASS",
                                                         "checks": []}]}},
        "reject": {"status": "done", "result": {"result": "FAIL", "disposition": "REJECT",
                                                "pull_reasons": []}},
        "continue": {"status": "done", "result": {"result": "PASS", "disposition": "CONTINUE",
                                                  "pull_reasons": []}},
        "no_disposition": {"status": "done", "result": {"result": "PASS"}},
        "aborted": {"status": "error", "error": "Cancelled: aborted by the operator",
                    "error_type": "Cancelled", "result": None},
        "invalid": {"status": "done", "result": {"result": "INVALID", "disposition": "PULL",
                                                 "invalid_reason": "no stereo calibration",
                                                 "pull_reasons": [{"reason": "uncalibrated",
                                                                   "message": "no mm"}]}},
    }
    f = tmp_path / "cases.json"
    f.write_text(json.dumps(cases), encoding="utf-8")
    h = tmp_path / "lamp_harness.js"
    h.write_text(_LAMP_HARNESS, encoding="utf-8")
    r = subprocess.run([node, str(h), str(STATIC / "quick.js"), str(f)], capture_output=True,
                       text=True, encoding="utf-8", timeout=60)
    assert r.returncode == 0, r.stderr
    o = json.loads(r.stdout)
    p = o["pull"]
    assert p["lamp"] == "PULL – quality inspection" and p["cls"] == "qlamp lamp-pull"
    assert p["disp"] == "PULL (quality inspection)" and p["pullHidden"] is False
    assert "lid_right/height: guard_band" in p["pull"] and p["verdict"] == "PASS"
    assert p["zones"] == ["z-pull", "z-pass"]
    assert o["reject"]["lamp"] == "FAIL" and o["reject"]["cls"] == "qlamp lamp-fail"
    assert o["reject"]["pullHidden"] is True
    assert o["continue"]["lamp"] == "PASS" and o["continue"]["cls"] == "qlamp lamp-pass"
    for name in ("no_disposition", "aborted", "invalid"):          # doubt never lights PASS
        assert o[name]["cls"] == "qlamp lamp-pull" and o[name]["pullHidden"] is False, name
    assert o["aborted"]["verdict"] == "ABORTED" and "aborted" in o["aborted"]["pull"]
    assert "uncalibrated" in o["invalid"]["pull"]


_BUS_HARNESS = r"""
const fs = require('fs'), vm = require('vm');
class El {
  constructor(tag, id) { this.tagName = String(tag).toUpperCase(); this.id = id; this.children = [];
    this.attrs = {}; this.dataset = {}; this.className = ''; this._text = ''; this.hidden = false;
    this.disabled = false; this.checked = false; this.href = ''; this.src = ''; this.value = '';
    this.classList = { toggle() {}, add() {}, remove() {} }; }
  set textContent(t) { this._text = String(t); this.children = []; }
  get textContent() { return this._text + this.children.map((c) => c.textContent).join(''); }
  appendChild(c) { this.children.push(c); return c; }
  addEventListener() {}
}
const byId = {};
const el = (id) => byId[id] || (byId[id] = new El('div', id));
const script = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
let step = null;
const sandbox = { console, setTimeout, clearTimeout, setInterval, clearInterval,
  document: { addEventListener() {}, querySelector: (s) => el(s.slice(1)),
              querySelectorAll: () => [], createElement: (t) => new El(t),
              createTextNode: (t) => { const n = new El('#text'); n.textContent = t; return n; } },
  fetch: async (path, opt) => {
    const body = path === '/api/cmd/state' ? step.bus
      : path.startsWith('/api/jobs/') ? step.job
      : path.startsWith('/api/cmd/') ? { accepted: step.accepted, command: 'START',
                                         message: step.message || '', bus: step.bus, jobs: {} }
      : {};
    return { ok: true, status: 200, json: async () => body };
  } };
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(process.argv[2], 'utf8'), sandbox, { filename: 'quick.js' });
const pollOnce = vm.runInContext('pollOnce', sandbox);
const send = vm.runInContext('send', sandbox);
const show = vm.runInContext('showRun', sandbox);
(async () => {
  const out = [];
  for (const st of script) {
    step = st;
    if (st.show) show(st.show);
    if (st.send) await send(st.send);
    else if (st.bus) await pollOnce();
    out.push({ lamp: el('qLamp').textContent, cls: el('qLamp').className,
               disp: el('qDisposition').textContent, pull: el('qPull').textContent,
               pullHidden: el('qPull').hidden });
  }
  process.stdout.write(JSON.stringify(out));
})().catch((e) => { console.error(e); process.exit(1); });
"""


def _bus_state(state, result=None, disp=None, jobs=None, history=None, sig=None):
    return {"state": state, "last_result": result, "last_disposition": disp,
            "jobs": jobs or {}, "workstates": [], "workstate": None, "unusable": {},
            "signals": sig or {}, "history": history or [], "warnings": []}


def test_quick_lamp_follows_the_bus_and_never_shows_a_pass_it_does_not_give(tmp_path):
    """The lamp reads the same signals as the line: a run's CONTINUE is shown only once the
    bus has heard it end, and a bus PULL (ABORT after the verdict, a START refused on an
    unusable work state) replaces a green PASS on the lamp."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed; quick.js logic not checked")
    cont = {"id": "j1", "status": "done", "result": {"result": "PASS", "disposition":
                                                     "CONTINUE", "pull_reasons": []}}
    script = [
        # the run's job is done, the bus still ABORTING: no PASS on the lamp yet
        {"bus": _bus_state("ABORTING", jobs={"j1": "run"}), "job": cont},
        # the bus settled: ABORTED / PULL, whatever the job said
        {"bus": _bus_state("READY", "ABORTED", "PULL", sig={"PULL": True}), "job": cont},
        # a fresh CONTINUE cycle (shown), then a START refused on an unusable work state
        {"show": cont, "bus": _bus_state("READY", "PASS", "CONTINUE", sig={"PASS": True}),
         "job": cont},
        {"send": "start", "accepted": False, "message": "work state a.json is not usable: "
                                                        "bad JSON",
         "bus": _bus_state("READY", "INVALID", "PULL", sig={"PULL": True, "ALARM": True},
                           history=[["t", "quick", "START", False,
                                     "work state a.json is not usable: bad JSON"]])},
        # a bus CONTINUE never turns a run shown as PULL green
        {"show": {"id": "j2", "status": "done", "result": {"result": "PASS",
                                                           "disposition": "PULL"}},
         "bus": _bus_state("READY", "PASS", "CONTINUE", sig={"PASS": True}),
         "job": {"id": "j2", "status": "done", "result": {"result": "PASS",
                                                         "disposition": "PULL"}}},
    ]
    f = tmp_path / "script.json"
    f.write_text(json.dumps(script), encoding="utf-8")
    h = tmp_path / "bus_harness.js"
    h.write_text(_BUS_HARNESS, encoding="utf-8")
    r = subprocess.run([node, str(h), str(STATIC / "quick.js"), str(f)], capture_output=True,
                       text=True, encoding="utf-8", timeout=60)
    assert r.returncode == 0, r.stderr
    o = json.loads(r.stdout)
    assert o[0]["lamp"] == "RUNNING"                               # waits for the bus
    assert o[1]["cls"] == "qlamp lamp-pull" and "ABORTED" in o[1]["pull"], o[1]
    assert o[2]["cls"] == "qlamp lamp-pass"
    assert o[3]["cls"] == "qlamp lamp-pull" and "START refused" in o[3]["pull"], o[3]
    assert o[3]["disp"] == "PULL (quality inspection)" and o[3]["pullHidden"] is False
    assert o[4]["cls"] == "qlamp lamp-pull"                       # never PASS over a doubt


# ================================================================== docs

def test_docs_name_the_dispositions():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "Dispositions: CONTINUE / REJECT / PULL (quality inspection)" in readme
    porting = (ROOT / "docs" / "PORTING.md").read_text(encoding="utf-8")
    assert "PULL" in porting and "quality-inspection lane" in porting
    assert "guard_band_pct" in (ROOT / "recipes" / "README.md").read_text(encoding="utf-8")
    assert "invalid_reasons_to_reject" in (ROOT / "recipes" / "README.md").read_text(
        encoding="utf-8")
    assert "guard_band_pct" in (ROOT / "stations" / "README.md").read_text(encoding="utf-8")

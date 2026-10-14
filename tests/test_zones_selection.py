"""The zone selection (engine/zones/selection.py) and several zone sets in one scan,
measurement and quick check, in MOCK mode: which sets and zones are switched on is stored per
rig placement and never repaired silently; the union of what is on is lit; per-zone results
are keyed "<set>/<id>"; a plane is fitted PER ZONE, never across separate zones; a selection
that cannot be resolved is refused (Measure: not measurable; Quick check: INVALID -> PULL),
never the whole field in its place. Judging runs on synthetic scans of the mock wall
(recipe_synth.oracle_scan: the true surface + the true projector px as the decode); the two
real mock captures of the module check what is lit."""

import contextlib
import io
import json
import re
import threading
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from engine import api, badparts, config, eventlog, rules
from engine.calibrate import flow
from engine.measure import values as V
from engine.zones import ZoneError, ZoneSet, Zone
from engine.zones import painter as P
from engine.zones import selection as sel
from engine.zones import session as S
from engine.zones import store as zstore
from engine.zones import use
from ui import cli

import recipe_synth as RS

PRESET = "bench_9in_metric"
GEO = "mock-bench_9in_v1"


def _zs(p, name, zones, **kw) -> ZoneSet:
    return ZoneSet(name=name, geometry_id=p.geometry_id, preset=p.name, mock=True,
                   fingerprint=config.geometry_fingerprint(p), zones=zones, **kw)


def _cells(p, **kw) -> ZoneSet:
    """Two ellipses on two parts (the first painted set of the bench)."""
    return _zs(p, "cells", [Zone.ellipse("Z1", 200, 280, 400, 460, name="part"),
                            Zone.ellipse("Z2", 540, 300, 740, 440, name="part")], **kw)


def _tube(p, **kw) -> ZoneSet:
    """Three rectangles; its ids repeat those of `cells`."""
    return _zs(p, "tube", [Zone.rect("Z1", 240, 620, 460, 800, name="tube"),
                           Zone.rect("Z2", 560, 640, 740, 820),
                           Zone.rect("Z3", 600, 480, 720, 580)], **kw)


@pytest.fixture(scope="module")
def lab(tmp_path_factory):
    """One mock library: the bench's true stereo calibration, the sets `cells` (2 zones) and
    `tube` (3 zones), a synthetic full-field scan of the flat wall (what the rig would have
    made), and TWO real mock fast scans with a spy on what was lit: one through the stored
    selection (cells on with Z2 switched off, tube off), one through two sets at once."""
    lib = tmp_path_factory.mktemp("zsel") / "library"
    s = api.configure(lib, mock=True)
    seen = []
    real = flow.capture_patterns

    def spy(sess, preset, patterns, what, zone_mask=None):
        seen.append(None if zone_mask is None else np.array(zone_mask, bool))
        return real(sess, preset, patterns, what, zone_mask=zone_mask)

    flow.capture_patterns = spy
    try:
        p = api.load_preset(PRESET)
        s.bench.use_stereo()
        RS.truth_calibration(s, p)
        painted = {"cells": zstore.save(s.store, _cells(p)).created,
                   "tube": zstore.save(s.store, _tube(p)).created}
        wall = RS.oracle_scan(s, p, [], "wall")
        wall.cloud
        api.zone_select(sets=[{"set": "cells", "on": True, "ids": ["Z1"]},
                              {"set": "tube", "on": False, "ids": None}], preset=p, by="test")
        one = api.scan_fast(None, p, "sel_scan", average=2, axis="xy", zones="selection")
        two = api.scan_fast(None, p, "two_sets", average=2, axis="xy", zones="cells+tube:Z1")
        api.zone_select(clear=True, preset=p, by="test")
    finally:
        flow.capture_patterns = real
        api._session = None
    return {"lib": lib, "wall": wall, "one": one.meta, "two": two.meta, "two_id": two.id,
            "lit": [m for m in seen if m is not None], "painted": painted}


@pytest.fixture
def rig(lab):
    """The module's library as the module made it: the two sets as they were painted (a test
    may have painted over or deleted one), nothing else, nothing switched on."""
    s = api.configure(lab["lib"], mock=True)
    p = api.load_preset(PRESET)
    _reset(s, p, lab)
    yield s, p, lab
    _reset(s, p, lab)
    api._session = None


def _reset(s, p, lab) -> None:
    f = sel.file_of(s.store, p.geometry_id)
    if f.exists():
        f.unlink()                          # (a test's own clean slate: the engine never does)
    # no inspection judged the module's scans: a test's REJECT / PULL marks the scan it judged
    # (its synthetic scans are copies of the stored wall), and a part judged bad is never a
    # golden (engine/badparts.py) - a later test that names the wall as its golden would test
    # that rule instead of its own
    for entry in (Path(s.store.root) / "metric").glob("*/*"):
        for name in (badparts.STAMP, badparts.SWEPT):
            (entry / name).unlink(missing_ok=True)
    for extra in set(zstore.names(s.store, GEO)) - {"cells", "tube"}:
        zstore.delete(s.store, GEO, extra, "test cleanup")
    for name, make in (("cells", _cells), ("tube", _tube)):
        want = make(p, created=lab["painted"][name])
        try:
            have = zstore.load(s.store, GEO, name)
        except ZoneError:
            have = None
        if have is None or have.created != want.created or \
                [z.to_dict() for z in have.zones] != [z.to_dict() for z in want.zones]:
            zstore.save(s.store, want, replace=have is not None)


def _with_cloud(scan, xyz=None, mask=None):
    """A copy of a stored scan with its cloud altered (the stored scan is untouched)."""
    c = scan.cloud
    xyz = c.xyz.copy() if xyz is None else xyz
    mask = c.mask.copy() if mask is None else mask
    xyz[~mask] = np.nan
    return replace(scan, _cloud=replace(c, xyz=xyz, mask=mask))


def _regions(s, p, scan, text="zoneset:cells+tube") -> dict:
    return use.camera_regions(s, p, text, scan, scan.cloud.mask.shape)[1]


def _planes(s, p, scan, moves: dict):
    """`scan` with zones moved onto other planes: {key: (offset mm, slope along x, slope
    along y)} - each zone stays flat, at its own height and tilt."""
    cam = _regions(s, p, scan)
    xyz = scan.cloud.xyz.copy()
    for key, (dz, ax, ay) in moves.items():
        m = cam[key] & scan.cloud.mask
        x, y = xyz[..., 0][m], xyz[..., 1][m]
        xyz[..., 2][m] += dz + ax * (x - x.mean()) + ay * (y - y.mean())
    return _with_cloud(scan, xyz=xyz)


def _bump(s, p, scan, key=None, at=None, mm=2.0, half=12):
    """`scan` with a square bump: in the middle of zone `key`, or at camera px `at`."""
    xyz = scan.cloud.xyz.copy()
    if key is not None:
        ys, xs = np.nonzero(_regions(s, p, scan)[key])
        at = (int(xs.mean()), int(ys.mean()))
    x, y = at
    xyz[y - half:y + half, x - half:x + half, 2] -= mm
    return _with_cloud(scan, xyz=xyz)


def run_cli(lib, *argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(["--mock", "--library", str(lib), "--preset", PRESET, *argv])
    assert "Traceback" not in out.getvalue() + err.getvalue(), err.getvalue()
    return code, out.getvalue(), err.getvalue()


def _quick(lab, monkeypatch, runs, scan, *argv) -> tuple:
    """`scan quick-check ...` with the scan the rig would have made -> (exit code, the run's
    result.json or None, how often the rig was asked to scan, stdout)."""
    asked = []

    def fake_scan_fast(zone, p, label, steps, periods, average, **kw):
        asked.append(zone)
        return scan
    before = set(runs.iterdir()) if runs.is_dir() else set()
    with monkeypatch.context() as m:        # (only these two: a test's own patches stay)
        m.setattr(api, "scan_fast", fake_scan_fast)
        m.setattr(cli, "_mock_place_metric", lambda ctx: None)
        code, out, err = run_cli(lab["lib"], "quick-check", PRESET, *argv, "--runs-dir",
                                 str(runs))
    api.configure(lab["lib"], mock=True)        # (cli.main ends its own session)
    api.load_preset(PRESET)
    new = sorted(set(runs.iterdir()) - before) if runs.is_dir() else []
    res = json.loads((new[0] / "result.json").read_text(encoding="utf-8")) if new else None
    return code, res, asked, out + err


# ------------------------------------------------------------------ the stored selection

def test_selection_file_is_append_only_and_an_unreadable_newest_line_is_unknown(rig):
    s, p, _ = rig
    f = sel.file_of(s.store, GEO)
    assert f == Path(s.store.root) / "zones" / GEO / "_selection.jsonl" and not f.exists()
    st = api.zone_selection(p)
    assert st["state"] == "ok" and st["ok"] and st["empty"] and st["label"] == "whole field"
    assert st["text"] == "" and st["time"] is None and [r["set"] for r in st["sets"]] == ["cells", "tube"]
    a = api.zone_select(on="cells", preset=p, by="tester")
    b = api.zone_select(on="tube:Z1,Z2", preset=p, by="tester")
    lines = [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2 and lines[0]["text"] == "zoneset:cells"         # the history stays
    assert lines[1] == {"schema": "zone-selection-1", "geometry_id": GEO, "time": lines[1]["time"],
                        "by": "tester", "text": "zoneset:cells+tube:Z1,Z2",
                        "sets": [{"set": "cells", "on": True, "ids": None},
                                 {"set": "tube", "on": True, "ids": ["Z1", "Z2"]}]}
    assert a["label"] == "cells (2)" and b["label"] == "cells (2), tube (2 of 3)"
    assert b["text"] == "zoneset:cells+tube:Z1,Z2" and b["by"] == "tester" and a["stamp"] != b["stamp"]
    assert b["n_sets_on"] == 2 and b["n_zones_on"] == 4
    tube = next(r for r in b["sets"] if r["set"] == "tube")
    assert [z["on"] for z in tube["zones"]] == [True, True, False] and tube["n_on"] == 2
    # the file is not a zone set: never listed, never in the way of one
    assert zstore.names(s.store, GEO) == ["cells", "tube"]
    assert [i["name"] for i in api.zone_sets(p)] == ["cells", "tube"]
    # a torn newest line: the selection is NOT KNOWN - the line before it is never used in its
    # place (that would bring back a selection somebody changed) - and runs are refused
    with open(f, "a", encoding="utf-8") as fh:
        fh.write('{"schema": "zone-selection-1", "geometry_id": "' + GEO + '", "sets": [{"se')
    st = api.zone_selection(p)
    assert st["state"] == "unknown" and not st["ok"] and st["label"] == "NOT KNOWN"
    assert st["text"] == "" and not any(r["on"] for r in st["sets"])
    assert st["problems"][0]["reason"] == "unreadable" and "newest line" in st["problems"][0]["message"]
    with pytest.raises(ZoneError, match="is not known"):
        api.scan_fast(None, p, "x", zones="selection")
    # setting it again is the repair (a new line; the torn one stays in the history)
    again = api.zone_select(zones="cells", preset=p)
    assert again["state"] == "ok" and again["label"] == "cells (2)"
    assert len(f.read_text(encoding="utf-8").splitlines()) == 4


def test_several_sets_text_parses_and_one_set_text_is_unchanged(rig):
    s, p, _ = rig
    assert use.parse_multi("zoneset:a") == [("a", None)]
    assert use.parse_multi("zoneset:a:Z1,Z3") == [("a", ["Z1", "Z3"])]
    assert use.parse_multi("zoneset:a/Z1") == [("a", ["Z1"])]
    assert use.parse_multi("zoneset:a+b:Z1") == [("a", None), ("b", ["Z1"])]
    assert use.parse_multi(" a + b/Z2 ", bare=True) == [("a", None), ("b", ["Z2"])]
    assert use.text_multi([("a", None)]) == use.spec_text("a") == "zoneset:a"
    assert use.text_multi([("a", ["Z1", "Z3"]), ("b", None)]) == "zoneset:a:Z1,Z3+b"
    assert use.join_terms(["cells", "tube:Z1"]) == use.join_terms("cells+tube:Z1") \
        == "zoneset:cells+tube:Z1"
    for bad, msg in (("zoneset:a+a", "named twice"), ("zoneset:a+", "empty zone set"),
                     ("zoneset:+a", "empty zone set"), ("zoneset:a+b:", "come zone ids"),
                     ("zoneset:a+b:Z1,Z1", "repeated"), ("a+b", "is not a zone set")):
        with pytest.raises(ZoneError, match=msg):
            use.parse_multi(bad)
    with pytest.raises(ZoneError, match="named twice"):
        use.join_terms(["cells", "cells:Z1"])
    # where ONE set is needed, several are refused (never the first of them)
    with pytest.raises(ZoneError, match="names several sets"):
        use.parse_spec("zoneset:cells+tube")
    with pytest.raises(ZoneError, match="names several sets"):
        api.zone_set("cells+tube", p)
    # resolved at the preset: the sets, the keys, the union
    picked = use.select_multi(s.store, p, "zoneset:cells+tube:Z1")
    assert picked.text == "zoneset:cells+tube:Z1" and picked.several_sets
    assert picked.keys == ["cells/Z1", "cells/Z2", "tube/Z1"]
    m = picked.mask()
    assert np.array_equal(m, _cells(p).mask() | _tube(p).mask(["Z1"]))
    assert np.array_equal(api.zone_mask("cells+tube:Z1", p), m)
    assert np.array_equal(api.zone_mask(["cells", "tube:Z1"], p), m)
    assert np.array_equal(use.resolve(s, p, "zoneset:cells+tube:Z1")[0], m)
    one = use.select_multi(s.store, p, "cells:Z2", bare=True)
    assert one.text == "zoneset:cells:Z2" and not one.several_sets and one.who("cells/Z2") == "Z2"
    for bad, msg in (("zoneset:cells+nope", "no zone set 'nope'"),
                     ("zoneset:cells+tube:Z9", "no zone 'Z9'")):
        with pytest.raises(ZoneError, match=msg):
            use.select_multi(s.store, p, bad)


# ------------------------------------------------------------------ what is lit

def test_the_selection_lights_only_what_is_on_and_two_sets_light_their_union(rig):
    s, p, lab = rig
    cells, tube = _cells(p), _tube(p)
    sel_mask, two_mask = lab["lit"]
    # the stored selection was: cells on with Z2 switched off, tube off
    assert np.array_equal(sel_mask, cells.mask(["Z1"]))
    assert not sel_mask[cells.zone("Z2").mask()].any() and not sel_mask[tube.mask()].any()
    m = lab["one"]
    assert m["zone"] == "zoneset:cells:Z1"                   # the word, expanded once: recorded
    z = m["zones"]
    assert z["zoneset"] == "cells" and z["ids"] == ["Z1"] and list(z["counts"]) == ["Z1"]
    assert z["counts"]["Z1"] > 5000 and z["outside"] < 0.02 * z["points"]
    assert [b["key"] for b in z["by_zone"]] == ["cells/Z1"] and z["by_zone"][0]["coverage_pct"] > 90
    # two sets in one scan: the union is lit, and nothing else
    assert np.array_equal(two_mask, cells.mask() | tube.mask(["Z1"]))
    assert not two_mask[tube.zone("Z2").mask()].any()


def test_per_zone_results_are_keyed_set_slash_id_when_ids_repeat(rig):
    s, p, lab = rig
    m = lab["two"]
    assert m["zone"] == "zoneset:cells+tube:Z1"
    z = m["zones"]
    # both sets have a Z1: the keys tell them apart
    assert z["zoneset"] is None and z["zonesets"] == ["cells", "tube"]
    assert z["ids"] == ["cells/Z1", "cells/Z2", "tube/Z1"] == list(z["counts"]) == list(z["dmd_px"])
    assert z["names"] == {"cells/Z1": "part", "cells/Z2": "part", "tube/Z1": "tube"}
    assert all(n > 3000 for n in z["counts"].values())
    assert z["counts"]["cells/Z1"] != z["counts"]["tube/Z1"]
    assert z["in_zones"] == sum(z["counts"].values()) and z["outside"] < 0.02 * z["points"]
    assert [(b["key"], b["zoneset"], b["id"]) for b in z["by_zone"]] == [
        ("cells/Z1", "cells", "Z1"), ("cells/Z2", "cells", "Z2"), ("tube/Z1", "tube", "Z1")]
    assert [x["set"] for x in z["sets"]] == ["cells", "tube"] and z["sets"][1]["ids"] == ["Z1"]
    doc = json.loads((Path(s.store.root) / "metric" / GEO / lab["two_id"].split("/")[-1]
                      / use.SCAN_FILE).read_text(encoding="utf-8"))
    assert doc["counts"] == z["counts"] and doc["spec"] == "zoneset:cells+tube:Z1"
    # ONE set keeps the record it always had (keyed by the zone id), plus by_zone / sets
    one = lab["one"]["zones"]
    legacy = {"kind", "zoneset", "spec", "ids", "names", "geometry_id", "zoneset_created",
              "scan_id", "created", "units", "axis", "counts", "points", "in_zones", "outside",
              "dmd_px", "union_dmd_px", "warnings"}
    assert set(one) == legacy | {"by_zone", "sets"}
    assert one["names"] == {"Z1": "part"} and one["dmd_px"] == {"Z1": int(_cells(p).zone("Z1").mask().sum())}
    # the report of any scan (here the synthetic wall, both axes): points + coverage per zone
    wall = lab["wall"]
    picked = use.select_multi(s.store, p, "zoneset:cells+tube")
    rep = use.report(picked, wall.field.corr_x, wall.field.corr_y, wall.cloud.mask)
    assert [r["key"] for r in rep["by_zone"]] == ["cells/Z1", "cells/Z2", "tube/Z1", "tube/Z2", "tube/Z3"]
    assert all(r["points"] > 3000 and r["coverage_pct"] > 95 for r in rep["by_zone"])
    assert rep["outside"] == rep["points"] - rep["in_zones"] > 0     # a full-field scan
    # columns only: zones that share columns cannot be told apart - unknown, never a guess
    x = use.report(picked, wall.field.corr_x, None, wall.cloud.mask)
    by = {r["key"]: r for r in x["by_zone"]}
    assert by["cells/Z1"]["points"] is None and by["cells/Z1"]["coverage_pct"] is None
    assert any("cells/Z1" in w and "columns only" in w for w in x["warnings"])
    assert "can't be told apart" in use.zone_doubt(by["cells/Z1"])


# ------------------------------------------------------------------ never the whole field instead

def test_selection_naming_a_deleted_set_is_refused_never_the_whole_field(rig, tmp_path, monkeypatch):
    s, p, lab = rig
    api.zone_select(zones="cells+tube", preset=p)
    # deleting a set that is switched on needs force (as a set a recipe names does)
    with pytest.raises(ZoneError, match="switched on in the zone selection.*--off tube"):
        api.zone_delete("tube", "test", p)
    out = api.zone_delete("tube", "fixture changed", p, force=True)
    assert out["was_on"] and out["forced"] and not zstore.exists(s.store, GEO, "tube")
    # the selection is not repaired: it names something gone, and says so
    st = api.zone_selection(p)
    assert st["state"] == "problem" and not st["ok"] and st["label"] == "cells (2), tube (NOT FOUND)"
    (pr,) = st["problems"]
    assert pr["set"] == "tube" and pr["reason"] == "missing" and "deleted or renamed" in pr["message"]
    gone = next(r for r in st["sets"] if r["set"] == "tube")
    assert gone["on"] and not gone["found"] and not gone["usable"]
    # every way of asking for the selection is refused BEFORE the rig: nothing is scanned

    def no_rig(*a, **k):
        raise AssertionError("the rig was opened")
    monkeypatch.setattr(type(s), "rig", no_rig)
    for call in (lambda: api.scan_fast(None, p, "x", zones="selection"),
                 lambda: api.scan_fast("selection", p, "x"),
                 lambda: api.scan_metric(p, "x", zone="selection"),
                 lambda: api.zone_show("selection", p),
                 lambda: api.zone_mask("selection", p)):
        with pytest.raises(ZoneError, match="cannot be used.*tube.*zone-select --off tube"):
            call()
    doc = api.measure_run("flatness", p, zone="selection", runs_dir=tmp_path / "runs")
    r = doc["result"]
    assert r["status"] == "NOT_MEASURABLE" and r["reason"] == "refused" and doc["exit_code"] == 2
    assert "cannot be used" in r["message"] and "tube" in r["message"] and r["scans"] == []
    # another change is refused too, until the set that is gone is switched off
    with pytest.raises(ZoneError, match="must be usable at this rig placement: switch tube off"):
        api.zone_select(on="cells:Z1", preset=p)
    fixed = api.zone_select(off="tube", preset=p)
    assert fixed["ok"] and fixed["label"] == "cells (2)" and [r["set"] for r in fixed["sets"]] == ["cells"]


def test_a_set_painted_at_another_geometry_is_refused_as_that(rig, tmp_path):
    """Not as a stale calibration (what Measure said before the zone text had its own gate)."""
    s, p, lab = rig
    other = replace(p, name="elsewhere", geometry_id="mock-elsewhere_v1")
    zstore.save(s.store, _zs(other, "far", [Zone.rect("Z1", 100, 100, 300, 300)]))
    with pytest.raises(rules.GeometryMismatch, match="painted at geometry mock-elsewhere_v1"):
        api.zone_select(on="far", preset=p)
    assert api.zone_selection(p)["empty"]                       # nothing was written
    doc = api.measure_run("flatness", p, zone="zoneset:cells+far", scans=[lab["wall"]],
                          runs_dir=tmp_path / "runs")["result"]
    assert doc["status"] == "NOT_MEASURABLE" and doc["reason"] == "refused"
    assert "painted at geometry mock-elsewhere_v1" in doc["message"]
    zstore.delete(s.store, "mock-elsewhere_v1", "far", "test cleanup")


def test_a_zone_id_that_is_gone_refuses_the_run_and_a_repaint_resets_the_switches(rig, tmp_path):
    s, p, lab = rig
    api.zone_select(zones="tube:Z1,Z3", preset=p)
    # the set changes behind the selection's back (another tool, a hand edit): Z3 is gone
    zstore.save(s.store, _zs(p, "tube", _tube(p).zones[:2]), replace=True)
    st = api.zone_selection(p)
    assert st["state"] == "problem" and st["label"] == "tube (ZONE MISSING)"
    assert st["problems"][0]["reason"] == "zone_missing" and st["problems"][0]["zone"] == "Z3"
    with pytest.raises(ZoneError, match="has no zone 'Z3'"):
        api.scan_fast(None, p, "x", zones="selection")
    # painted over through the painter: the per-zone switches start again at "all zones" (the
    # painter re-issues ids: Z3 may be another drawing now) and the result says so
    out = api.zone_paint("tube", p, source="tube", replace=True, polarity="normal", select=True,
                         events=[{"type": "drag", "from": [60, 60], "to": [160, 140]},
                                 {"type": "key", "key": "space"}])
    assert out["saved"] and out["outcome"] == "saved" and out["on"] is True
    assert out["switches_reset"] is True and "per-zone switches were reset" in out["selection_note"]
    assert out["replaced"].startswith(f"trash/zones/{GEO}/tube")
    st = api.zone_selection(p)
    assert st["ok"] and st["label"] == "tube (3)" and st["text"] == "zoneset:tube"
    assert [z["id"] for z in st["sets"][-1]["zones"]] == ["Z1", "Z2", "Z3"]
    # an empty id list is never "every zone": refused where it is written ...
    with pytest.raises(ZoneError, match="no zone is switched on in it"):
        api.zone_select(sets=[{"set": "tube", "on": True, "ids": []}], preset=p)
    assert api.zone_selection(p)["label"] == "tube (3)"
    # ... and a full list is stored as "every zone"
    full = api.zone_select(sets=[{"set": "tube", "on": True, "ids": ["Z3", "Z1", "Z2"]}], preset=p)
    assert next(r for r in full["sets"] if r["set"] == "tube")["ids"] is None


def test_an_empty_selection_asked_for_by_name_is_refused_and_a_stamp_must_match(rig, tmp_path):
    s, p, lab = rig
    wall = lab["wall"]
    empty = api.zone_selection(p)
    assert empty["empty"]
    # the bare word (a script, a station file): "the selected zones" is never the whole field
    with pytest.raises(ZoneError, match="is empty.*switch a zone set on first"):
        api.scan_fast(None, p, "x", zones="selection")
    with pytest.raises(ZoneError, match="is empty"):
        sel.expand(s.store, p, "selection")
    # a stamp that only travels WITH the bare word (a station START) compares and no more:
    # it never turns "nothing switched on" into the whole field
    with pytest.raises(ZoneError, match="is empty"):
        sel.expand(s.store, p, "selection", expect=empty["stamp"])
    with pytest.raises(sel.SelectionChanged, match="changed since it was shown"):
        sel.expand(s.store, p, "selection", expect="0123456789ab")
    # with the stamp the sender was shown IN the word, "nothing on" IS the whole field
    assert sel.expand(s.store, p, f"selection@{empty['stamp']}")[0] is None
    doc = api.measure_run("flatness", p, zone=f"selection@{empty['stamp']}", scans=[wall],
                          runs_dir=tmp_path / "runs")["result"]
    assert doc["status"] == "MEASURED" and doc["spec"]["zone"] is None
    assert doc["zone_selection"]["text"] == "" and doc["zone_selection"]["source"] == "selection"
    assert [v["name"] for v in doc["values"]][0] == "flatness_rms"         # one plane: one field
    # the selection changes under the sender: the stamp it holds is refused (never run)
    now = api.zone_select(on="cells", preset=p)
    assert now["stamp"] != empty["stamp"]
    with pytest.raises(sel.SelectionChanged, match="changed since it was shown.*cells \\(2\\)"):
        sel.expand(s.store, p, f"selection@{empty['stamp']}")
    doc = api.measure_run("flatness", p, zone=f"selection@{empty['stamp']}", scans=[wall],
                          runs_dir=tmp_path / "runs")["result"]
    assert doc["status"] == "NOT_MEASURABLE" and doc["reason"] == "refused" and doc["values"] == []
    assert "changed since it was shown" in doc["message"]
    # a set painted again keeps its text and changes the stamp
    zstore.save(s.store, _cells(p, created="2031-01-01T00:00:00"), replace=True)
    again = api.zone_selection(p)
    assert again["text"] == now["text"] == "zoneset:cells" and again["stamp"] != now["stamp"]
    # write with `expect`: refused when the stored selection is another one by now
    with pytest.raises(sel.SelectionChanged, match="changed meanwhile"):
        api.zone_select(on="tube", preset=p, expect=now["stamp"])
    assert api.zone_select(on="tube", preset=p, expect=again["stamp"])["label"] == "cells (2), tube (3)"


def test_switching_keeps_the_zone_switches_of_a_set_that_is_off(rig):
    s, p, _ = rig
    api.zone_select(zones="cells+tube:Z1,Z2", preset=p)
    off = api.zone_select(off="tube", preset=p)
    assert off["label"] == "cells (2)" and off["text"] == "zoneset:cells"
    tube = next(r for r in off["sets"] if r["set"] == "tube")
    assert not tube["on"] and tube["ids"] == ["Z1", "Z2"] and tube["n_on"] == 2      # remembered
    assert api.zone_select(on="tube", preset=p)["label"] == "cells (2), tube (2 of 3)"
    assert api.zone_select(on="tube:Z3", preset=p)["text"] == "zoneset:cells+tube:Z3"
    assert api.zone_select(zones="tube", preset=p)["label"] == "tube (3)"       # exactly this
    assert api.zone_select(clear=True, preset=p)["label"] == "whole field"
    for bad, msg in (({"on": "tube:Z9"}, "no zone 'Z9'"), ({"on": "nope"}, "no zone set 'nope'"),
                     ({"off": "nope"}, "no zone set 'nope'"),
                     ({"on": "tube", "off": "tube"}, "on and off in one change"),
                     ({"zones": "tube", "sets": []}, "not both")):
        with pytest.raises(ZoneError, match=msg):
            api.zone_select(preset=p, **bad)
    assert api.zone_selection(p)["empty"]                       # every refusal wrote nothing
    # a set deleted while it is off leaves the selection; a new set of that name starts clean
    api.zone_select(sets=[{"set": "tube", "on": False, "ids": ["Z1"]}], preset=p)
    api.zone_delete("tube", "test", p)
    assert sel.read(s.store, GEO)["record"]["sets"] == []
    out = api.zone_paint("tube", p, events=list(P.DEMO_EVENTS), polarity="normal")
    assert out["saved"] and out["on"] is False and out["selection_note"] == "It is NOT switched on."
    assert api.zone_selection(p)["empty"]


# ------------------------------------------------------------------ one plane per zone

def test_flatness_of_two_zones_on_two_planes_is_measured_per_zone(rig, tmp_path):
    """The rig finding: one plane through two flat, separate faces read 0.8 mm RMS. Two
    zones at different heights and tilts are two planes: each is flat, and no single
    "flatness" across them is reported."""
    s, p, lab = rig
    two = _planes(s, p, lab["wall"], {"cells/Z1": (3.0, 0.05, 0.0), "cells/Z2": (-2.0, 0.0, -0.10)})
    out = api.measure_run("flatness", p, zone="zoneset:cells", scans=[two],
                          runs_dir=tmp_path / "runs")
    doc = out["result"]
    assert doc["status"] == "MEASURED" and out["exit_code"] == 0, doc
    by = {v["name"]: v for v in doc["values"]}
    assert list(by) == [f"cells/{z}:{k}" for z in ("Z1", "Z2")
                        for k in ("flatness_rms", "flatness_p2_p98", "tilt", "plane_distance")]
    assert by["cells/Z1:flatness_rms"]["label"] == "cells/Z1 Flatness (RMS)"
    assert by["cells/Z1:flatness_rms"]["zone"] == "cells/Z1" and by["cells/Z2:tilt"]["zone"] == "cells/Z2"
    for z in ("Z1", "Z2"):                                     # each face is flat
        assert by[f"cells/{z}:flatness_rms"]["value"] < 0.06, by[f"cells/{z}:flatness_rms"]
        assert by[f"cells/{z}:flatness_p2_p98"]["value"] < 0.25
        assert by[f"cells/{z}:flatness_rms"]["uncertainty"]["value"] > 0
    # ... at its own height and tilt
    assert abs(by["cells/Z1:plane_distance"]["value"] - by["cells/Z2:plane_distance"]["value"]) > 2.0
    assert abs(by["cells/Z1:tilt"]["value"] - by["cells/Z2:tilt"]["value"]) > 1.5
    # one plane across both would have called two flat faces 1 mm rough
    cam = _regions(s, p, two, "zoneset:cells")
    across = V.flatness(two.cloud, cam["cells/Z1"] | cam["cells/Z2"])["values"][0]["value"]
    assert across > 1.0
    assert doc["details"]["per_zone"] and set(doc["details"]["by_zone"]) == {"cells/Z1", "cells/Z2"}
    zsel = doc["zone_selection"]
    assert zsel["text"] == "zoneset:cells" and [z["key"] for z in zsel["by_zone"]] == ["cells/Z1", "cells/Z2"]
    csv = (Path(out["run_dir"]) / "values.csv").read_text(encoding="utf-8")
    assert "cells/Z1:flatness_rms,cells/Z1 Flatness (RMS)" in csv
    # one zone alone keeps the names it always had
    one = api.measure_run("flatness", p, zone="zoneset:cells:Z2", scans=[two],
                          runs_dir=tmp_path / "runs")["result"]
    assert [v["name"] for v in one["values"]] == ["flatness_rms", "flatness_p2_p98", "tilt",
                                                 "plane_distance"]
    assert one["values"][0]["value"] == pytest.approx(by["cells/Z2:flatness_rms"]["value"], abs=1e-6)
    # zones of two sets: named by their set, however the ids repeat
    both = api.measure_run("flatness", p, zone="zoneset:cells:Z1+tube:Z1", scans=[two],
                           runs_dir=tmp_path / "runs")["result"]
    assert [v["name"] for v in both["values"]][::4] == ["cells/Z1:flatness_rms", "tube/Z1:flatness_rms"]


def test_measurements_with_no_meaning_across_zones_are_refused_before_the_scan(rig, tmp_path,
                                                                             monkeypatch):
    s, p, lab = rig

    def no_scan(*a, **k):
        raise AssertionError("a scan was made")
    monkeypatch.setattr(api, "scan_metric", no_scan)
    monkeypatch.setattr(api, "scan_fast", no_scan)
    runs = tmp_path / "runs"
    for kw, msg in (
            (dict(what="step", roi2=[10, 10, 60, 60]), "name ONE zone as the zone .region A."),
            (dict(what="distance", p1=[100, 100]), "the plane of which one"),
            (dict(what="dims"), "ONE part"),
            (dict(what="deviation", golden="wall"), "ONE compare")):
        r = api.measure_run(preset=p, zone="zoneset:cells", runs_dir=runs, **kw)["result"]
        assert r["status"] == "NOT_MEASURABLE" and r["reason"] == "refused", (kw, r["message"])
        assert re.search(msg, r["message"]), r["message"]
        assert r["scans"] == [] and r["values"] == []
    r = api.measure_run("step", p, zone="zoneset:cells/Z1", zone2="zoneset:tube", runs_dir=runs)
    assert r["result"]["reason"] == "refused" and "zone2 (region B) must name ONE zone" in r["result"]["message"]
    # the end-face measurement finds the rim on the fully lit part: no zone set, no selection
    for zone in ("zoneset:cells", "selection"):
        with pytest.raises(ValueError, match="end-face measurement uses the whole field"):
            api.measure_run("endface", p, zone=zone, face_diameter_mm=21.0, runs_dir=runs)
    monkeypatch.undo()
    code, out, err = run_cli(lab["lib"], "measure-run", PRESET, "--what", "endface", "--diameter",
                             "21", "--zones", "selection", "--runs-dir", str(runs))
    assert code == cli.EXIT_REFUSED and "end-face measurement uses the whole field" in err
    api.configure(lab["lib"], mock=True)
    p = api.load_preset(PRESET)
    # two points need no region: measured, and no plane of "the zones" is made up
    wall = lab["wall"]
    cam = _regions(s, p, wall, "zoneset:cells")
    pts = []
    for key in ("cells/Z1", "cells/Z2"):
        ys, xs = np.nonzero(cam[key])
        pts.append([float(xs.mean()), float(ys.mean())])
    r = api.measure_run("distance", p, zone="zoneset:cells", p1=pts[0], p2=pts[1], scans=[wall],
                        runs_dir=runs)["result"]
    assert r["status"] == "MEASURED" and [v["name"] for v in r["values"]] == ["distance"]
    one = api.measure_run("distance", p, zone="zoneset:cells:Z1", p1=pts[0], scans=[wall],
                          runs_dir=runs)["result"]
    assert [v["name"] for v in one["values"]] == ["point_to_plane"]


def test_a_selected_zone_with_no_points_is_not_measurable_and_pulled(rig, tmp_path, monkeypatch):
    s, p, lab = rig
    wall = lab["wall"]
    cam = _regions(s, p, wall)
    blind = _with_cloud(wall, mask=wall.cloud.mask & ~cam["cells/Z2"])      # Z2: nothing decoded
    doc = api.measure_run("flatness", p, zone="zoneset:cells", scans=[blind],
                          runs_dir=tmp_path / "runs")["result"]
    assert doc["status"] == "NOT_MEASURABLE" and doc["reason"] == "too_few_points" and doc["values"] == []
    assert "zone cells/Z2 (part): no point decoded in it" in doc["message"]
    # half a zone in shadow: seen too little (its coverage, against the zone itself)
    ys, xs = np.nonzero(cam["cells/Z1"])
    shadow = wall.cloud.mask.copy()
    shadow[:, :int(np.percentile(xs, 80))] &= ~cam["cells/Z1"][:, :int(np.percentile(xs, 80))]
    doc = api.measure_run("flatness", p, zone="zoneset:cells", scans=[_with_cloud(wall, mask=shadow)],
                          runs_dir=tmp_path / "runs")["result"]
    assert doc["reason"] == "too_few_points" and "the camera saw only" in doc["message"]
    assert "cells/Z1" in doc["message"]
    # Quick check: the run is INVALID -> PULL, and names the zone - never a PASS on the other
    code, res, asked, out = _quick(lab, monkeypatch, tmp_path / "q", blind, "--check", "flatness",
                                   "--threshold", "0.5", "--zones", "cells")
    assert code == 5 and res["result"] == "INVALID" and res["disposition"] == "PULL", out
    assert "zone cells/Z2 (part): no point decoded in it" in res["invalid_reason"]
    assert res["pull_reasons"][0]["reason"] == "too_few_points" and res["pull_reasons"][0]["zone"] == "cells/Z2"
    zsel = res["zone_selection"]
    assert zsel["invalid"] == ["cells/Z2"] and zsel["failed"] == []
    assert {z["key"]: z["verdict"] for z in zsel["by_zone"]} == {"cells/Z1": "PASS", "cells/Z2": "INVALID"}
    assert asked == ["zoneset:cells"]
    # a region typed next to the zones cuts them: a zone outside it is not judged, and said
    x0, x1 = int(xs.min()), int(xs.max())
    code, res, _, out = _quick(lab, monkeypatch, tmp_path / "q", wall, "--check", "flatness",
                               "--threshold", "0.5", "--zones", "cells",
                               "--roi", f"{x0},{int(ys.min())},{x1 + 1},{int(ys.max()) + 1}")
    assert code == 5 and res["disposition"] == "PULL", out
    assert "zone cells/Z2" in res["invalid_reason"] and "inside job.roi" in res["invalid_reason"]
    assert zsel["by_zone"][0]["coverage_pct"] > 95


# ------------------------------------------------------------------ Quick check through zones

def test_quick_check_judges_each_zone_on_its_own_px_and_not_the_frame(rig, tmp_path, monkeypatch):
    """Lit through two small zones a flatness quick check used to be INVALID for every part
    (coverage counted against the whole camera image), and one plane was fitted across both."""
    s, p, lab = rig
    runs = tmp_path / "q"
    args = ("--check", "flatness", "--threshold", "0.5")
    two = _planes(s, p, lab["wall"], {"cells/Z1": (3.0, 0.05, 0.0), "cells/Z2": (-2.0, 0.0, -0.10)})
    code, res, asked, out = _quick(lab, monkeypatch, runs, two, *args, "--zones", "cells")
    assert code == 0 and res["result"] == "PASS" and res["disposition"] == "CONTINUE", out
    assert res["deviation_mm"]["method"] == "plane_fit_per_zone" and res["deviation_mm"]["max"] < 0.3
    zsel = res["zone_selection"]
    assert zsel["text"] == "zoneset:cells" and zsel["source"] == "argument" and zsel["selected"] is None
    assert [(z["key"], z["verdict"]) for z in zsel["by_zone"]] == [("cells/Z1", "PASS"), ("cells/Z2", "PASS")]
    assert all(z["coverage_pct"] > 95 and z["max_mm"] < 0.3 and z["points"] > 3000 for z in zsel["by_zone"])
    assert zsel["outside"] > 0 and "cells/Z1 PASS" in zsel["summary"] and zsel["failed"] == []
    assert res["capture"]["zone"] == "zoneset:cells" and res["job"]["zone"] == "zoneset:cells"
    assert asked == ["zoneset:cells"] and "cells/Z1" in out and "seen %" in out
    # the same scan judged as a whole field: two good parts at two heights FAIL
    code, whole, _, _ = _quick(lab, monkeypatch, runs, two, *args)
    assert whole["result"] == "FAIL" and "zone_selection" not in whole
    # a defect OUTSIDE every zone changes nothing; one INSIDE a zone rejects, and names it
    ys, xs = np.nonzero(_regions(s, p, two)["tube/Z2"])
    far = _bump(s, p, two, at=(int(xs.mean()), int(ys.mean())))
    code, res, _, out = _quick(lab, monkeypatch, runs, far, *args, "--zones", "cells")
    assert code == 0 and res["result"] == "PASS", out
    bad = _bump(s, p, two, key="cells/Z2")
    code, res, _, out = _quick(lab, monkeypatch, runs, bad, *args, "--zones", "cells")
    assert code == 1 and res["result"] == "FAIL" and res["disposition"] == "REJECT", out
    zsel = res["zone_selection"]
    assert zsel["failed"] == ["cells/Z2"] and zsel["summary"].startswith("cells/Z2 FAIL (max ")
    assert {z["key"]: z["verdict"] for z in zsel["by_zone"]} == {"cells/Z1": "PASS", "cells/Z2": "FAIL"}
    assert res["deviation_mm"]["largest_defect"]["zone"] == "cells/Z2"
    assert res["deviation_mm"]["max"] == pytest.approx(2.0, abs=0.3)
    assert "cells/Z2 FAIL" in out
    # the selection by name, several sets, and the selection with a switched-off zone
    api.zone_select(zones="cells+tube:Z1,Z3", preset=p, by="tester")
    code, res, asked, out = _quick(lab, monkeypatch, runs, bad, *args, "--zones", "selection")
    assert code == 1 and asked == ["zoneset:cells+tube:Z1,Z3"], out
    zsel = res["zone_selection"]
    assert zsel["source"] == "selection" and zsel["selected"]["by"] == "tester"
    assert [z["key"] for z in zsel["by_zone"]] == ["cells/Z1", "cells/Z2", "tube/Z1", "tube/Z3"]
    assert res["job"]["zone"] == "selection" and res["capture"]["zone"] == "zoneset:cells+tube:Z1,Z3"
    code, res, asked, out = _quick(lab, monkeypatch, runs, bad, *args, "--zones", "cells:Z1",
                                   "--zones", "tube:Z2")
    assert code == 0 and asked == ["zoneset:cells:Z1+tube:Z2"], out        # Z2 of cells is left out
    # --zones and --zone: one or the other; a columns-only job cannot tell zones apart
    code, out, err = run_cli(lab["lib"], "quick-check", PRESET, *args, "--zones", "cells",
                             "--zone", "F10")
    assert code == cli.EXIT_REFUSED and "not both" in err


def test_quick_check_with_a_selection_that_cannot_be_used_is_invalid_and_pulled(rig, tmp_path,
                                                                             monkeypatch):
    s, p, lab = rig
    runs = tmp_path / "q"
    args = ("--check", "flatness", "--threshold", "0.5", "--zones", "selection")
    # nothing switched on: "the selected zones" is never the whole field
    code, res, asked, out = _quick(lab, monkeypatch, runs, lab["wall"], *args)
    assert code == 5 and res["result"] == "INVALID" and res["disposition"] == "PULL", out
    assert res["pull_reasons"][0]["reason"] == "zone_unresolved" and "is empty" in res["invalid_reason"]
    assert asked == [] and res["part"]["scan_id"] is None and "zone_selection" not in res
    # a set that is gone
    api.zone_select(zones="cells+tube", preset=p)
    api.zone_delete("tube", "test", p, force=True)
    code, res, asked, out = _quick(lab, monkeypatch, runs, lab["wall"], *args)
    assert code == 5 and res["disposition"] == "PULL" and asked == [], out
    assert res["pull_reasons"][0]["reason"] == "zone_unresolved"
    assert "cannot be used" in res["invalid_reason"] and "tube" in res["invalid_reason"]
    # a stamp that is not the stored selection's: refused, nothing scanned
    api.zone_select(off="tube", preset=p)
    code, res, asked, out = _quick(lab, monkeypatch, runs, lab["wall"], "--check", "flatness",
                                   "--threshold", "0.5", "--zone", "selection@0000000000aa")
    assert code == 5 and asked == [] and "changed since it was shown" in res["invalid_reason"]
    # an unknown set named directly, and a columns-only scan
    code, res, asked, out = _quick(lab, monkeypatch, runs, lab["wall"], "--check", "flatness",
                                   "--threshold", "0.5", "--zones", "cells+nope")
    assert code == 5 and asked == [] and "no zone set 'nope'" in res["invalid_reason"]


def test_a_golden_must_have_been_scanned_through_the_same_zones(rig, tmp_path, monkeypatch):
    """A golden lit on the whole field against a run lit through zones: the unlit rest of
    the golden read as missing material - a REJECT of a good part."""
    s, p, lab = rig
    runs = tmp_path / "q"
    code, res, asked, out = _quick(lab, monkeypatch, runs, lab["wall"], "--check", "golden",
                                   "--golden", "wall", "--threshold", "0.5", "--zones", "cells")
    assert code == 5 and res["result"] == "INVALID" and res["disposition"] == "PULL", out
    assert res["pull_reasons"][0]["reason"] == "golden" and asked == []
    assert "was scanned through the whole field, this run is lit through zoneset:cells" in res["invalid_reason"]
    assert "scan a golden through the same zones" in res["invalid_reason"]
    ms = next(m for m in api.list_library("metric") if m.label == "two_sets")
    picked = use.select_multi(s.store, p, "zoneset:cells+tube:Z1")
    assert use.golden_mismatch(s, p, ms, picked) is None             # the same zones: comparable
    assert use.golden_mismatch(s, p, ms, use.select_multi(s.store, p, "zoneset:tube:Z1+cells")) is None
    assert "this run is lit through zoneset:cells" in use.golden_mismatch(
        s, p, ms, use.select_multi(s.store, p, "zoneset:cells"))
    assert "lights other zones (or the whole field)" in use.golden_mismatch(s, p, ms, None)
    assert use.golden_mismatch(s, p, lab["wall"], None) is None
    # the set was painted again since the golden: not the same zones any more
    zstore.save(s.store, _cells(p, created="2032-02-02T00:00:00"), replace=True)
    assert "was painted again (2032-02-02T00:00:00) since the golden" in use.golden_mismatch(
        s, p, ms, use.select_multi(s.store, p, "zoneset:cells+tube:Z1"))
    # Measure: a deviation vs a golden lit otherwise is not measurable (reason golden)
    doc = api.measure_run("deviation", p, zone="zoneset:cells:Z1", golden="wall",
                          runs_dir=tmp_path / "runs", mock_scene="none")["result"]
    assert doc["status"] == "NOT_MEASURABLE" and doc["reason"] == "golden" and doc["scans"] == []


def test_a_rig_change_since_a_set_was_painted_pulls_until_it_was_shown(rig, tmp_path, monkeypatch):
    """A set painted two hours ago, a new rig session one hour ago (the rig was set down
    again): the engine cannot see where the projector points now. The row and the selection
    say CHECK; a judged run PULLs until the zones were shown on the part again."""
    from datetime import datetime, timedelta
    from engine import rigstate
    s, p, lab = rig
    then = datetime.now().replace(microsecond=0) - timedelta(hours=2)
    old = zstore.save(s.store, _zs(p, "old", _cells(p).zones, created=then.isoformat()))
    rigstate.append(s.store, GEO, {"event": "session", "reason": "set down again"},
                    now=then + timedelta(hours=1))
    api.zone_select(zones="old+tube", preset=p)
    st = api.zone_selection(p)
    assert st["ok"] and st["label"] == "old (2, CHECK), tube (3)"         # tube: painted after it
    assert st["check"] == [f"old: a new rig session ({(then + timedelta(hours=1)).isoformat()})"]
    assert next(r for r in st["sets"] if r["set"] == "old")["rig_changes"]
    assert not next(r for r in st["sets"] if r["set"] == "tube")["rig_changes"]
    args = ("--check", "flatness", "--threshold", "0.5", "--zones", "selection")
    # a judged run through the set is a PULL: its zones may no longer sit on the part
    code, res, asked, out = _quick(lab, monkeypatch, tmp_path / "q", lab["wall"], *args)
    assert code == 5 and res["disposition"] == "PULL" and asked == [], out
    assert res["pull_reasons"][0]["reason"] == "zone_unresolved"
    assert "a new rig session" in res["invalid_reason"] and "Show selected zones" in res["invalid_reason"]
    # Measure decides nothing: it measures, and carries the warning
    doc = api.measure_run("flatness", p, zone="zoneset:old", scans=[lab["wall"]],
                          runs_dir=tmp_path / "runs")["result"]
    assert doc["status"] == "MEASURED" and any("a new rig session" in w for w in doc["warnings"])
    # a show too short to look at is not an answer
    short = api.zone_show("old", p, hold_s=0)
    assert short["all_seen"] and short["recorded_as_shown"] == [] and zstore.rig_changes(s.store, old)
    # the zones held on the part, every one seen: recorded, and the answer to what came before
    shown = api.zone_show("selection", p, hold_s=1.0)
    assert shown["all_seen"] and shown["recorded_as_shown"] == ["old", "tube"]
    recs = [json.loads(x) for x in (Path(s.store.root) / "zones" / GEO / zstore.SHOWN_FILE)
            .read_text(encoding="utf-8").splitlines()]
    assert recs[-2]["set"] == "old" and recs[-2]["created"] == old.created and recs[-2]["ids"] == ["Z1", "Z2"]
    assert zstore.rig_changes(s.store, old) == [] and api.zone_selection(p)["label"] == "old (2), tube (3)"
    code, res, asked, out = _quick(lab, monkeypatch, tmp_path / "q", lab["wall"], *args)
    assert code == 0 and res["result"] == "PASS", out
    # painted again: another painting, which that show does not speak for
    again = zstore.save(s.store, _zs(p, "old", _cells(p).zones,
                                     created=(then + timedelta(minutes=30)).isoformat()), replace=True)
    assert zstore.last_shown(s.store, again) is None and zstore.rig_changes(s.store, again)


# ------------------------------------------------------------------ show, users, the painter

def test_zone_show_projects_several_sets_with_set_slash_id_labels(rig):
    s, p, _ = rig
    api.zone_select(zones="cells:Z1+tube:Z1,Z2", preset=p)
    r = api.zone_show("selection", p, hold_s=0)
    assert r["zoneset"] is None and r["zonesets"] == ["cells", "tube"] and r["all_seen"]
    assert r["spec"] == "zoneset:cells:Z1+tube:Z1,Z2"
    assert [(z["key"], z["zoneset"], z["id"]) for z in r["zones"]] == [
        ("cells/Z1", "cells", "Z1"), ("tube/Z1", "tube", "Z1"), ("tube/Z2", "tube", "Z2")]
    assert all(z["seen"] and z["camera_px"] > 1000 for z in r["zones"])
    doc = json.loads(Path(r["json_path"]).read_text(encoding="utf-8"))
    assert [z["key"] for z in doc["zones"]] == ["cells/Z1", "tube/Z1", "tube/Z2"]
    # one set: the result it always had, plus the keys
    one = api.zone_show("tube:Z2", p, hold_s=0)
    assert one["zoneset"] == "tube" and one["zones"][0]["id"] == "Z2" and one["zones"][0]["key"] == "tube/Z2"
    assert api.zone_show(["cells:Z1", "tube:Z3"], p, hold_s=0)["zonesets"] == ["cells", "tube"]
    empty = api.zone_select(clear=True, preset=p)
    with pytest.raises(ZoneError, match="is empty: no zone set is switched on"):
        api.zone_show(f"selection@{empty['stamp']}", p, hold_s=0)


def test_users_find_a_set_inside_a_several_set_text_and_in_station_files(rig, tmp_path):
    s, p, _ = rig
    st = tmp_path / "stations"
    st.mkdir()
    (st / "a.json").write_text(json.dumps({"station_id": "a", "job": {
        "type": "metric", "zone": "zoneset:cells:Z1+tube"}}), encoding="utf-8")
    (st / "b.json").write_text(json.dumps({"station_id": "b", "job": {"type": "metric",
                                                                    "zone": "zoneset:mytube"}}),
                               encoding="utf-8")
    (st / "c.yaml").write_text("station_id: c\njob:\n  zone: zoneset:tube/Z2\n", encoding="utf-8")
    assert use.station_refs(st, "tube") == ["a.json", "c.yaml"]
    assert use.station_refs(st, "cells") == ["a.json"] and use.station_refs(st, "tub") == []
    assert api.zone_users("tube", st) == {"recipes": [], "stations": ["a.json", "c.yaml"]}
    recipe = {"schema_version": 1, "name": "uses_both", "preset": PRESET,
              "scan": {"kind": "fast", "axis": "xy", "zone": "zoneset:cells+tube:Z1"},
              "zones": [{"id": "a", "shape": "rect", "units": "camera_px", "coords": [1, 1, 50, 50],
                         "checks": [{"type": "presence"}]}], "feedback": {"enabled": False}}
    api.recipe_add("uses_both", recipe=recipe)
    try:
        assert use.recipe_refs(s.store, api.paths()["recipes"], "tube") == ["uses_both"]
        assert use.recipe_refs(s.store, api.paths()["recipes"], "cells") == ["uses_both"]
        # named by a recipe or a station configuration: delete needs force
        with pytest.raises(ZoneError, match="used by recipe.s. uses_both and station "
                                            "configuration.s. a.json, c.yaml"):
            api.zone_delete("tube", "test", p, stations_dir=st)
    finally:
        api.recipe_delete("uses_both", "test cleanup", force=True)
    with pytest.raises(ZoneError, match="used by station configuration.s. a.json: change"):
        api.zone_delete("cells", "test", p, stations_dir=st)      # the station file alone
    assert zstore.names(s.store, GEO) == ["cells", "tube"]
    # a recipe never takes the operator's selection: it names its own zones
    v = api.recipe_validate({**recipe, "scan": {**recipe["scan"], "zone": "selection"}})
    assert not v["valid"] and any("not for a recipe" in x for x in v["problems"])


def test_painter_result_says_how_the_session_ended(rig):
    s, p, _ = rig
    draw = {"type": "drag", "from": [100, 100], "to": [300, 300]}
    # saved: a NEW set is switched on when asked (and only then)
    out = api.zone_paint("fresh", p, polarity="normal", select=True,
                         events=[draw, {"type": "key", "key": "space"}])
    assert (out["saved"], out["outcome"], out["on"], out["replaced"]) == (True, "saved", True, None)
    assert out["selection_note"] == "It is switched ON (all 1 zones)." and not out["switches_reset"]
    assert api.zone_selection(p)["label"] == "fresh (1)"
    # painted over: its on / off is kept
    over = api.zone_paint("fresh", p, polarity="normal", source="fresh", replace=True, select=True,
                          events=[{"type": "drag", "from": [400, 400], "to": [600, 600]},
                                  {"type": "key", "key": "space"}])
    assert over["outcome"] == "saved" and over["on"] is True and over["n_zones"] == 2
    assert over["selection_note"] == "It stays switched ON (all 2 zones)."
    assert over["replaced"] == f"trash/zones/{GEO}/fresh"
    api.zone_select(off="fresh", preset=p)
    off = api.zone_paint("fresh", p, polarity="normal", source="fresh", replace=True, select=True,
                         events=[{"type": "key", "key": "space"}])
    assert off["saved"] and off["on"] is False and off["selection_note"] == "It stays switched OFF, as before."
    # cancelled (Esc), the script ended without Space, aborted: nothing saved, each named
    esc = api.zone_paint("other", p, polarity="normal", events=[draw, {"type": "key", "key": "escape"}])
    assert (esc["saved"], esc["outcome"], esc["on"]) == (False, "cancelled", None)
    end = api.zone_paint("other", p, polarity="normal", events=[draw])
    assert (end["saved"], end["outcome"]) == (False, "script_ended")
    ev = threading.Event()
    ev.set()
    ab = api.zone_paint("other", p, polarity="normal", events=[draw], cancel=ev)
    assert (ab["saved"], ab["outcome"]) == (False, "aborted") and "aborted" in ab["reason"]
    assert not zstore.exists(s.store, GEO, "other")
    # the idle timeout (an injected clock): the rig is given back, nothing saved

    class Quiet:
        def show(self, img):
            pass

        def poll(self, ms):
            return []
    ticks = iter(range(0, 10_000, 30))
    m = P.PainterModel(name="idle")
    assert S.run_painter(m, Quiet(), mock=True, idle_timeout_s=20.0, clock=lambda: next(ticks)) == "cancel"
    assert m.ended == "idle_timeout" and set(S.OUTCOMES) >= {m.ended, "saved", "cancelled", "aborted"}
    # the word is not a set name
    with pytest.raises(ZoneError, match="cannot be named 'selection'"):
        api.zone_paint("selection", p, polarity="normal", events=[draw])


def test_a_saved_set_stays_saved_when_giving_the_rig_back_fails(rig, monkeypatch):
    """The painter saves while it still holds the rig. A camera / projector close error after
    that is a warning on a SAVED result: an error would be told as "not saved", and a set that
    was painted over would keep per-zone switches that now point at other drawings."""
    from engine import session as engine_session
    s, p, _ = rig
    api.zone_select(zones="tube:Z2", preset=p)
    real, fails = engine_session._release_rig, []

    def release(camera, projector, lock, primary):
        real(camera, projector, lock, primary)
        if primary is None:
            fails.append(1)
            raise RuntimeError("camera close failed")
    monkeypatch.setattr(engine_session, "_release_rig", release)
    out = api.zone_paint("tube", p, source="tube", replace=True, polarity="normal", select=True,
                         events=[{"type": "drag", "from": [60, 60], "to": [200, 180]},
                                 {"type": "key", "key": "space"}])
    assert fails == [1]
    assert out["saved"] and out["outcome"] == "saved" and out["n_zones"] == 4
    assert out["replaced"].startswith(f"trash/zones/{GEO}/tube")
    (w,) = [x for x in out["warnings"] if "rig was not given back cleanly" in x]
    assert "RuntimeError: camera close failed" in w and "check the camera and the projector" in w
    assert out["rig_error"] == "RuntimeError: camera close failed"
    # ... and the selection followed the save: every zone on, not "Z2" of another drawing
    assert out["on"] is True and out["switches_reset"] is True
    st = api.zone_selection(p)
    assert st["ok"] and st["label"] == "tube (4)" and st["text"] == "zoneset:tube"
    # nothing saved (Esc): the close error is the answer, as for every other rig job
    with pytest.raises(RuntimeError, match="camera close failed"):
        api.zone_paint("other", p, polarity="normal", events=[{"type": "key", "key": "escape"}])
    assert not zstore.exists(s.store, GEO, "other") and len(fails) == 2


# ------------------------------------------------------------------ the event log, bad parts, the rig

def _event_row(lib, run_id) -> dict:
    """The inspection event log's row of one run (engine/eventlog.py), as text cells."""
    rows = [r for f in sorted((Path(lib) / "events").glob(f"{eventlog.INSPECTIONS}*.csv"))
            for r in eventlog.read_rows(f) if r["inspection_id"] == run_id]
    assert len(rows) == 1, rows
    return rows[0]


def _event_values(lib, run_id) -> list:
    return [r for f in sorted((Path(lib) / "events").glob(f"{eventlog.VALUES}*.csv"))
            for r in eventlog.read_rows(f) if r["inspection_id"] == run_id]


def _bad_index(root, run_id) -> dict:
    (row,) = [r for r in eventlog.read_rows(Path(root) / badparts.INDEX)
              if r["inspection_id"] == run_id]
    return row


def test_runs_through_zones_write_their_zones_to_the_event_log_and_bad_parts_out(
        rig, tmp_path, monkeypatch, eventlog_on):
    """Where the zones meet the inspection event log and the bad-part output: a measurement
    through two zones names the zone of every value; a quick check judged zone by zone
    writes one value row per zone ("cells/Z1"), and its REJECT names the zone that failed;
    a selection that cannot be resolved is a PULL with its reason - in the log and in the
    bad-part output alike."""
    s, p, lab = rig
    lib, bad, runs = lab["lib"], tmp_path / "bad", tmp_path / "q"
    two = _planes(s, p, lab["wall"], {"cells/Z1": (3.0, 0.05, 0.0), "cells/Z2": (-2.0, 0.0, -0.10)})
    # Measure: every value named by its zone, no limits, no judgment
    out = api.measure_run("flatness", p, zone="zoneset:cells", scans=[two],
                          runs_dir=tmp_path / "runs")
    assert out["result"]["status"] == "MEASURED"
    row, vals = _event_row(lib, out["run_id"]), _event_values(lib, out["run_id"])
    assert row["judgment"] == eventlog.NO_JUDGMENT and row["n_values"] == "8"
    assert [(v["zone"], v["name"]) for v in vals][::4] == [("cells/Z1", "cells/Z1:flatness_rms"),
                                                          ("cells/Z2", "cells/Z2:flatness_rms")]
    assert {v["zone"] for v in vals} == {"cells/Z1", "cells/Z2"}
    assert all(v["result"] == eventlog.NO_JUDGMENT for v in vals)
    # Quick check, cells/Z2 out of its limit: REJECT - one row per zone, the zone named
    args = ("--check", "flatness", "--threshold", "0.5", "--bad-part-output", str(bad))
    code, res, _, said = _quick(lab, monkeypatch, runs, _bump(s, p, two, key="cells/Z2"), *args,
                                "--zones", "cells")
    assert code == 1 and res["disposition"] == "REJECT", said
    assert res["outputs"]["bad_part_output"] == str(bad.resolve())
    vals = {v["zone"]: v for v in _event_values(lib, res["run_id"])}
    assert set(vals) == {"cells/Z1", "cells/Z2"}
    assert (vals["cells/Z1"]["result"], vals["cells/Z2"]["result"]) == ("OK", "OUT")
    assert vals["cells/Z2"]["name"] == "deviation_mm.max" and vals["cells/Z2"]["check"] == "metric"
    assert float(vals["cells/Z2"]["value"]) == pytest.approx(2.0, abs=0.3)
    assert float(vals["cells/Z2"]["upper_limit"]) == 0.5
    row = _event_row(lib, res["run_id"])
    assert row["judgment"] == "FAIL" and row["n_values"] == "2" and row["n_out_of_limit"] == "1"
    assert row["reason"].startswith("cells/Z2: deviation_mm.max ") and "outside its limits" in row["reason"]
    idx = _bad_index(bad, res["run_id"])
    assert idx["disposition"] == "REJECT" and idx["out_of_limit"].startswith("cells/Z2/metric: ")
    assert (bad / idx["folder"] / "result.json").is_file()
    # a zone that cannot be judged: PULL, the zone named, and its row INVALID
    blind = _with_cloud(two, mask=two.cloud.mask & ~_regions(s, p, two)["cells/Z2"])
    code, res, _, said = _quick(lab, monkeypatch, runs, blind, *args, "--zones", "cells")
    assert code == 5 and res["disposition"] == "PULL", said
    vals = {v["zone"]: v["result"] for v in _event_values(lib, res["run_id"])}
    assert vals == {"cells/Z1": "OK", "cells/Z2": "INVALID"}
    row = _event_row(lib, res["run_id"])
    assert row["judgment"] == "PULL" and row["reason"].startswith("too_few_points: cells/Z2: ")
    assert _bad_index(bad, res["run_id"])["disposition"] == "PULL"
    # the selection asked for and nothing switched on: a PULL before anything is lit, and
    # its reason where the line reads it
    code, res, asked, said = _quick(lab, monkeypatch, runs, two, *args, "--zones", "selection")
    assert code == 5 and asked == [] and res["pull_reasons"][0]["reason"] == "zone_unresolved", said
    row = _event_row(lib, res["run_id"])
    assert row["judgment"] == "PULL" and row["reason"].startswith("zone_unresolved: ")
    assert "is empty" in row["reason"] and _event_values(lib, res["run_id"]) == []
    idx = _bad_index(bad, res["run_id"])
    assert idx["disposition"] == "PULL" and idx["reason"] == row["reason"]
    assert json.loads((bad / idx["folder"] / "result.json").read_text(encoding="utf-8"))[
        "pull_reasons"][0]["reason"] == "zone_unresolved"


def test_a_part_judged_bad_through_zones_is_never_a_golden_through_them(rig, tmp_path,
                                                                        monkeypatch):
    """References are made from good parts only (engine/badparts.py), also through zones: a
    scan judged REJECT zone by zone is refused as the golden of a run lit through the very
    same zones (which the zone rule alone would allow), in Quick check and in Measure."""
    s, p, lab = rig
    runs = tmp_path / "q"
    golden = next(m for m in api.list_library("metric") if m.label == "two_sets")
    zones = "cells+tube:Z1"
    assert use.golden_mismatch(s, p, golden, use.select_multi(s.store, p, f"zoneset:{zones}")) is None
    ys, xs = np.nonzero(_regions(s, p, golden, f"zoneset:{zones}")["cells/Z2"])
    dented = _bump(s, p, golden, at=(int(xs.mean()), int(ys.mean())))
    code, res, _, said = _quick(lab, monkeypatch, runs, dented, "--check", "flatness",
                                "--threshold", "0.5", "--zones", zones)
    assert res["disposition"] in ("REJECT", "PULL") and res["part"]["scan_id"] == golden.id, said
    judged = res["run_id"]
    code, res, asked, said = _quick(lab, monkeypatch, runs, golden, "--check", "golden",
                                    "--golden", "two_sets", "--threshold", "0.5", "--zones", zones)
    assert code == cli.EXIT_REFUSED and res["disposition"] == "PULL" and asked == [], said
    assert "BadPartReference" in res["invalid_reason"] and judged in res["invalid_reason"]
    assert "GOOD parts only" in res["invalid_reason"]
    doc = api.measure_run("deviation", p, zone="zoneset:cells:Z1", golden="two_sets",
                          runs_dir=tmp_path / "runs", mock_scene="none")["result"]
    assert doc["status"] == "NOT_MEASURABLE" and doc["reason"] == "golden" and doc["scans"] == []
    assert "GOOD parts only" in doc["message"] and judged in doc["message"]


def test_a_zone_lit_scan_on_a_fixed_rig_in_doubt_says_so(tmp_path):
    """A fixed rig whose newest rig check said MOVED still scans, and every scan says it
    (engine/rigstate.py): a scan lit through painted zones too - the rig check's words first,
    the rig stamp trusted false, and the zones' own warning that the set was painted before
    that change."""
    from datetime import datetime, timedelta
    from engine import rigstate
    s = api.configure(tmp_path / "library", mock=True)
    try:
        p = api.load_preset(PRESET)
        s.bench.use_stereo()
        cal = RS.truth_calibration(s, p)
        hour_ago = (datetime.now() - timedelta(hours=1)).isoformat(timespec="seconds")
        zstore.save(s.store, _cells(p, created=hour_ago))
        words = "projector (or its picture) moved ~18.6 px against the scene"
        rigstate.record_check(s.store, p, {
            "status": "MOVED", "tol_px": 0.5, "calibration_id": cal.id, "reference_id": "ref0",
            "reference_calibration_id": cal.id, "reference_trust": {"mobile": True, "fixed": True},
            "message": words, "moved": "projector"})
        ms = api.scan_fast(None, p, "zones_after_moved", average=2, axis="xy", zones="cells")
        w = ms.meta["warnings"]
        assert w[0].startswith(rigstate.IN_DOUBT) and words in w[0], w
        assert ms.meta["rig"]["mount"] == "fixed" and ms.meta["rig"]["trusted"] is False
        assert any(x.startswith("zone set cells was painted") and "rig check MOVED" in x for x in w)
        assert [z["key"] for z in ms.meta["zones"]["by_zone"]] == ["cells/Z1", "cells/Z2"]
        stored = json.loads((Path(ms.path) / "meta.json").read_text(encoding="utf-8"))
        assert stored["rig"]["trusted"] is False and stored["warnings"][0] == w[0]
    finally:
        api._session = None

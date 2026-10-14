"""Station mode and the zone selection, in MOCK mode. A station configuration is the
definition of an inspection: its run inspects what its file says, always. The zone sets
switched on at the operator page are used only when the file's own job says "zone":
"selection" - then an empty or unusable selection is INVALID -> PULL with nothing captured,
the selection cannot change while the station is ARMED or in a cycle, and a selection that
changed under the page (or since ARM) starts nothing. The command bus asks its owner's gate
before every cycle start; the runs here judge a synthetic scan of the mock wall."""

import contextlib
import io
import json
import threading
import time

import pytest

from engine import api, badparts, config, eventlog
from engine.control.commands import CommandBus, Handlers
from engine.zones import Zone, ZoneSet
from engine.zones import selection as sel
from engine.zones import store as zstore
from conftest import ROOT
from test_run_job_mock import RESULT_KEYS
from test_ui_server import Client
from ui import cli
from ui import run_job as RJ
from ui.web import server as ui_server

import recipe_synth as RS

PRESET = "bench_9in_metric"
GEO = "mock-bench_9in_v1"


def _zs(p, name, zones) -> ZoneSet:
    return ZoneSet(name=name, geometry_id=p.geometry_id, preset=p.name, mock=True,
                   fingerprint=config.geometry_fingerprint(p), zones=zones)


def _station_doc(tmp, zone, **job) -> dict:
    doc = json.loads((ROOT / "stations" / "example_metric_check.json").read_text(encoding="utf-8"))
    doc.update(station_id="zstation", preset=PRESET, runs_dir=str(tmp / "runs"), mock_library=None)
    doc["job"] = {**doc["job"], "zone": zone, **job}
    return doc


@pytest.fixture(scope="module")
def lab(tmp_path_factory):
    """A mock library with the bench's true stereo calibration, the sets `cells` (2 zones)
    and `tube` (1 zone) and a synthetic full-field scan of the flat wall."""
    tmp = tmp_path_factory.mktemp("zstation")
    s = api.configure(tmp / "library", mock=True)
    try:
        p = api.load_preset(PRESET)
        s.bench.use_stereo()
        RS.truth_calibration(s, p)
        zstore.save(s.store, _zs(p, "cells", [Zone.ellipse("Z1", 200, 280, 400, 460, name="part"),
                                             Zone.ellipse("Z2", 540, 300, 740, 440, name="part")]))
        zstore.save(s.store, _zs(p, "tube", [Zone.rect("Z1", 240, 620, 460, 800, name="tube")]))
        wall = RS.oracle_scan(s, p, [], "wall")
        wall.cloud
    finally:
        api._session = None
    return {"lib": tmp / "library", "tmp": tmp, "wall": wall}


@pytest.fixture
def rig(lab, monkeypatch):
    """The library with nothing switched on, and the rig's scan replaced by the wall scan
    (`asked` lists the zone each scan was lit through)."""
    s = api.configure(lab["lib"], mock=True)
    p = api.load_preset(PRESET)
    f = sel.file_of(s.store, GEO)
    f.unlink(missing_ok=True)
    asked = []

    def fake_scan_fast(zone, preset, label, steps, periods, average, **kw):
        asked.append(zone)
        return lab["wall"]
    monkeypatch.setattr(api, "scan_fast", fake_scan_fast)
    monkeypatch.setattr(cli, "_mock_place_metric", lambda ctx: None)
    yield s, p, lab, asked
    f.unlink(missing_ok=True)
    api._session = None


def _run(lab, tmp_path, doc) -> tuple:
    """`scan run <config>` -> (exit code, result.json, stdout)."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    cfg = tmp_path / "station.json"
    cfg.write_text(json.dumps({**doc, "runs_dir": str(tmp_path / "runs")}), encoding="utf-8")
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
        code = cli.main(["--mock", "--library", str(lab["lib"]), "run", str(cfg)])
    (run_dir,) = sorted((tmp_path / "runs").iterdir())
    res = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    api.configure(lab["lib"], mock=True)            # (cli.main ends its own session)
    api.load_preset(PRESET)
    return code, res, out.getvalue()


def _run_handed(lab, tmp_path, doc, stamp) -> tuple:
    """`scan run <config>` as the web server runs a station job: handed the stamp of the
    selection the bus accepted -> (exit code, result.json)."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    cfg = tmp_path / "station.json"
    cfg.write_text(json.dumps({**doc, "runs_dir": str(tmp_path / "runs")}), encoding="utf-8")
    args = cli.build_parser().parse_args(["--mock", "--library", str(lab["lib"]), "--json",
                                          "run", str(cfg)])
    args.preset_given, args.mock_part_given, args.zone_expect = False, False, stamp
    ctx = cli.Ctx(args)
    ctx.session = api._engine()
    ctx._outputs_before = len(ctx.session.store.outputs())
    with contextlib.redirect_stdout(io.StringIO()):
        code, data = cli.cmd_run(ctx)
    return code, data["result"]


# ------------------------------------------------------------------ the station rule (scan run)

def test_a_station_run_ignores_the_selection_unless_its_configuration_says_so(rig, tmp_path):
    s, p, lab, asked = rig
    api.zone_select(zones="cells", preset=p)                 # switched on at the operator page
    code, res, out = _run(lab, tmp_path, _station_doc(tmp_path, None))
    assert code == 0 and res["result"] == "PASS", out
    assert asked == [None]                                   # the whole field was lit
    assert set(res) == RESULT_KEYS and res["capture"]["zone"] is None and res["job"]["zone"] is None
    assert res["deviation_mm"]["method"] == "plane_fit"
    assert RJ.uses_selection("selection") and RJ.uses_selection("Selection@ab12")
    assert not RJ.uses_selection(None) and not RJ.uses_selection("zoneset:cells")


def test_a_configuration_with_zone_selection_runs_and_records_the_selection(rig, tmp_path):
    s, p, lab, asked = rig
    st = api.zone_select(zones="cells+tube", preset=p, by="the operator")
    code, res, out = _run(lab, tmp_path, _station_doc(tmp_path, "selection"))
    assert code == 0 and res["result"] == "PASS" and res["disposition"] == "CONTINUE", out
    assert asked == ["zoneset:cells+tube"]                   # the word, expanded once
    assert set(res) == RESULT_KEYS | {"zone_selection"}
    assert res["job"]["zone"] == "selection"                 # the configuration's own word
    assert res["capture"]["zone"] == "zoneset:cells+tube"    # what was lit
    z = res["zone_selection"]
    assert z["text"] == "zoneset:cells+tube" and z["source"] == "selection"
    assert z["selected"] == {"time": st["time"], "by": "the operator"}
    assert [(x["key"], x["verdict"]) for x in z["by_zone"]] == [
        ("cells/Z1", "PASS"), ("cells/Z2", "PASS"), ("tube/Z1", "PASS")]
    assert [x["set"] for x in z["sets"]] == ["cells", "tube"]
    assert res["deviation_mm"]["method"] == "plane_fit_per_zone" and "cells/Z1" in out
    # the stamp the bus accepted travels with the run: another selection by then is a PULL
    api.zone_select(off="tube", preset=p)
    code, res = _run_handed(lab, tmp_path / "again", _station_doc(tmp_path, "selection"),
                            st["stamp"])
    assert code == 5 and res["result"] == "INVALID" and res["disposition"] == "PULL"
    assert "changed since it was shown" in res["invalid_reason"] and len(asked) == 1
    # ... and the selection as it is now, with its own stamp, runs
    now = api.zone_selection(p)
    code, res = _run_handed(lab, tmp_path / "now", _station_doc(tmp_path, "selection"),
                            now["stamp"])
    assert code == 0 and res["result"] == "PASS" and asked[1:] == ["zoneset:cells"]


def test_zone_selection_empty_or_unusable_is_invalid_and_pulled(rig, tmp_path):
    s, p, lab, asked = rig
    doc = _station_doc(tmp_path, "selection")
    # nothing switched on: never the whole field in its place
    code, res, out = _run(lab, tmp_path / "a", doc)
    assert code == 5 and res["result"] == "INVALID" and res["disposition"] == "PULL", out
    assert res["pull_reasons"][0]["reason"] == "zone_unresolved" and "is empty" in res["invalid_reason"]
    assert asked == [] and res["part"]["scan_id"] is None and set(res) == RESULT_KEYS
    # the same when the run is handed the stamp of that empty selection (a job sent to the
    # server by hand, past the bus): a stamp beside the run compares, it never makes
    # "nothing switched on" the whole field
    empty = api.zone_selection(p)
    assert empty["empty"]
    code, res = _run_handed(lab, tmp_path / "a2", doc, empty["stamp"])
    assert code == 5 and res["result"] == "INVALID" and res["disposition"] == "PULL", res
    assert res["pull_reasons"][0]["reason"] == "zone_unresolved" and "is empty" in res["invalid_reason"]
    assert asked == [] and res["part"]["scan_id"] is None and "zone_selection" not in res
    # a set that is switched on and gone
    api.zone_select(zones="cells+tube", preset=p)
    api.zone_delete("tube", "test", p, force=True)
    try:
        code, res, out = _run(lab, tmp_path / "b", doc)
        assert code == 5 and res["disposition"] == "PULL" and asked == [], out
        assert "cannot be used" in res["invalid_reason"] and "tube" in res["invalid_reason"]
    finally:
        zstore.save(s.store, _zs(p, "tube", [Zone.rect("Z1", 240, 620, 460, 800, name="tube")]))
    # a set named in the file that does not exist: the same doubt, never another zone
    code, res, out = _run(lab, tmp_path / "c", _station_doc(tmp_path, "zoneset:nope"))
    assert code == 5 and res["pull_reasons"][0]["reason"] == "zone_unresolved" and asked == []
    assert "no zone set 'nope'" in res["invalid_reason"]
    # a named set that exists: judged zone by zone, whatever the page has switched on
    api.zone_select(clear=True, preset=p)
    code, res, out = _run(lab, tmp_path / "d", _station_doc(tmp_path, "zoneset:cells:Z2"))
    assert code == 0 and asked == ["zoneset:cells:Z2"] and res["zone_selection"]["source"] == "argument"
    assert [x["key"] for x in res["zone_selection"]["by_zone"]] == ["cells/Z2"]


def test_zone_selection_and_a_bad_part_output_in_one_configuration(rig, tmp_path):
    """Both settings of a station configuration at once: "zone": "selection" and its
    bad_part_output. A selection that cannot be used is a PULL with nothing captured, copied
    to the configuration's bad-part output with its reason; a good part writes nothing
    there."""
    s, p, lab, asked = rig
    doc = {**_station_doc(tmp_path, "selection"), "bad_part_output": "bad"}
    cfg = RJ.config_from_doc(doc, tmp_path)
    assert cfg.zone == "selection" and cfg.bad_part_output == str((tmp_path / "bad").resolve())
    code, res, out = _run(lab, tmp_path / "a", doc)
    assert code == 5 and res["disposition"] == "PULL" and asked == [], out
    assert res["pull_reasons"][0]["reason"] == "zone_unresolved"
    bad = tmp_path / "a" / "bad"
    assert res["outputs"]["bad_part_output"] == str(bad.resolve())
    (row,) = eventlog.read_rows(bad / badparts.INDEX)
    assert row["inspection_id"] == res["run_id"] and row["disposition"] == "PULL"
    assert row["reason"].startswith("zone_unresolved: ") and "is empty" in row["reason"]
    assert (bad / row["folder"] / "result.json").is_file()
    # switched on: judged zone by zone, a good part - nothing written there
    api.zone_select(zones="cells", preset=p)
    code, res, out = _run(lab, tmp_path / "b", doc)
    assert code == 0 and res["disposition"] == "CONTINUE" and asked == ["zoneset:cells"], out
    assert res["outputs"]["bad_part_output"] == str((tmp_path / "b" / "bad").resolve())
    assert not (tmp_path / "b" / "bad").exists()


def test_zone_selection_is_refused_for_recipe_jobs_and_inside_recipes(tmp_path):
    ok = RJ.config_from_doc(_station_doc(tmp_path, "selection"), tmp_path)
    assert ok.zone == "selection" and ok.job_type == "metric"
    for job, msg in ((dict(type="inspect"), "job.type inspect judges only what its base"),
                     (dict(type="compare", golden="g1"), "job.type compare judges only"),
                     (dict(type="recipe", recipe="flat_plate"), "names its zones in its recipe"),
                     (dict(axis="x"), "needs job.axis xy")):
        doc = _station_doc(tmp_path, "selection", **job)
        if job.get("type", "metric") != "metric":
            doc["job"].pop("guard_band_pct", None)
        with pytest.raises(RJ.StationConfigError, match=msg):
            RJ.config_from_doc(doc, tmp_path)
    with pytest.raises(RJ.StationConfigError, match="job.zone must be text"):
        RJ.config_from_doc(_station_doc(tmp_path, ["cells"]), tmp_path)
    # the stamped word ("nothing on" = the whole field) is a quick check's own record: a
    # station configuration may not say it
    with pytest.raises(RJ.StationConfigError, match="a station configuration says \"selection\""):
        RJ.config_from_doc(_station_doc(tmp_path, "selection@0123456789ab"), tmp_path)
    quick = {**_station_doc(tmp_path, "selection@0123456789ab"), "operating_mode": "quick_check"}
    assert RJ.config_from_doc(quick, tmp_path).zone == "selection@0123456789ab"
    # a configuration that names its own zone set is not affected
    assert RJ.config_from_doc(_station_doc(tmp_path, "zoneset:cells+tube", type="inspect",
                                           guard_band_pct=None), tmp_path).zone == "zoneset:cells+tube"


# ------------------------------------------------------------------ the bus asks its owner's gate

def _bus(gate) -> tuple:
    started = []
    bus = CommandBus(Handlers(start=lambda cfg, ok: started.append(cfg), gate=gate),
                     configs=["a.json"])
    return bus, started


def test_the_bus_gate_refuses_a_cycle_without_or_with_a_pull():
    say = {}
    asked = []

    def gate(cmd, arg, armed):
        asked.append((cmd, (arg or {}).get("zones") if isinstance(arg, dict) else None, armed))
        return say.get("now")
    bus, started = _bus(gate)
    # nothing to say: the cycle starts, and ends PASS
    assert bus.command("START").accepted and started == ["a.json"]
    bus.job_finished("run", "PASS", None, disposition="CONTINUE")
    assert bus.signals()["PASS"] and bus.last_disposition == "CONTINUE"
    # refused without a pull: nothing starts, and the last PASS no longer speaks for the part
    say["now"] = {"message": "the zone painter holds the rig", "pull": False}
    r = bus.command("START", arg={"config": "a.json", "zones": "abc"})
    assert not r.accepted and r.message == "the zone painter holds the rig" and bus.state == "READY"
    assert started == ["a.json"] and bus.last_result is None and not bus.signals()["PASS"]
    assert asked[-1] == ("START", "abc", None)
    # refused with a pull: the cycle never started - INVALID, the part PULLed
    say["now"] = {"message": "none is switched on", "pull": True}
    r = bus.command("START")
    assert not r.accepted and bus.last_result == "INVALID" and bus.last_disposition == "PULL"
    assert bus.signals()["PULL"] and bus.signals()["KICKOUT"] and started == ["a.json"]
    # ARM: refused (no part was announced: no PULL of its own); armed: the token is kept ...
    r = bus.command("ARM")
    assert not r.accepted and r.message == "none is switched on" and bus.state == "READY"
    say["now"] = {"token": "s1"}
    assert bus.command("ARM").accepted and bus.state == "ARMED" and bus.armed_token == "s1"
    # ... and handed to the gate at every TRIGGER; a gate that then pulls disarms
    assert bus.command("TRIGGER").accepted and asked[-1] == ("TRIGGER", None, "s1")
    bus.job_finished("run", "PASS", None, disposition="CONTINUE")
    assert bus.state == "ARMED"
    say["now"] = {"message": "the zones changed since ARM", "pull": True}
    r = bus.command("TRIGGER")
    assert not r.accepted and "the zones changed since ARM - disarmed" in r.message
    assert bus.state == "READY" and bus.last_disposition == "PULL" and len(started) == 2
    # a gate that cannot answer never lets a cycle start
    say["now"] = None

    def broken(cmd, arg, armed):
        raise OSError("disk gone")
    bus.h.gate = broken
    r = bus.command("START")
    assert not r.accepted and "the start check failed: OSError" in r.message
    assert bus.last_disposition == "PULL" and len(started) == 2


# ------------------------------------------------------------------ over HTTP

@pytest.fixture(scope="module")
def web(lab):
    """A MOCK server on the module's library with two station configurations: own.json
    (the whole field, as written) and sel.json ("zone": "selection"). Its scans are the
    synthetic wall scan."""
    stations = lab["tmp"] / "stations"
    stations.mkdir()
    (stations / "own.json").write_text(json.dumps(_station_doc(lab["tmp"], None)), encoding="utf-8")
    (stations / "sel.json").write_text(json.dumps(_station_doc(lab["tmp"], "selection")),
                                       encoding="utf-8")
    httpd, app = ui_server.make_server(mock=True, library=lab["lib"], port=0,
                                       stations_dir=stations, preset=PRESET)
    t = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
    t.start()
    asked = []
    real_scan, real_place = api.scan_fast, cli._mock_place_metric

    def fake_scan_fast(zone, preset, label, steps, periods, average, **kw):
        asked.append(zone)
        return lab["wall"]
    api.scan_fast, cli._mock_place_metric = fake_scan_fast, lambda ctx: None
    try:
        yield Client(app.port), app, asked
    finally:
        api.scan_fast, cli._mock_place_metric = real_scan, real_place
        httpd.shutdown()
        httpd.server_close()
        app.close()
        api._session = None


def _clear(c, app):
    """READY, the given configuration's preset loaded, nothing switched on."""
    c.post("/api/cmd/abort", {})
    c.post("/api/cmd/disarm", {})
    c.post("/api/cmd/reset", {})
    _wait_state(c, ("READY",))
    sel.file_of(app.session.store, GEO).unlink(missing_ok=True)
    app._zone_cache.clear()


def _wait_state(c, want, timeout=120.0):
    t0 = time.monotonic()
    while True:
        _, b = c.get("/api/cmd/state")
        if b["state"] in want:
            return b
        assert time.monotonic() - t0 < timeout, b
        time.sleep(0.1)


def _job(c, jid, timeout=120.0):
    t0 = time.monotonic()
    while True:
        _, j = c.get(f"/api/jobs/{jid}")
        if j["status"] in ("done", "error"):
            return j
        assert time.monotonic() - t0 < timeout, j
        time.sleep(0.1)


def _select(c, sets, was=None):
    return c.post("/api/zones/selection", {"sets": sets, "was": was, "source": "quick"})


def test_start_runs_the_configuration_as_written_whatever_is_switched_on(web):
    c, app, asked = web
    _clear(c, app)
    _, d = c.get("/api/stations")
    by = {s["config"]: s for s in d["stations"]}
    assert by["own.json"]["zone_from_selection"] is False and by["own.json"]["zone"] is None
    assert by["sel.json"]["zone_from_selection"] is True and by["sel.json"]["zone"] == "selection"
    assert c.post("/api/cmd/config", {"config": "own.json"})[0] == 200
    status, r = _select(c, [{"set": "cells", "on": True, "ids": None}])
    assert status == 200 and r["accepted"] and r["selection"]["label"] == "cells (2)"
    n = len(asked)
    status, r = c.post("/api/cmd/start", {"source": "test", "zones": "not-the-stamp"})
    assert status == 200 and r["accepted"], r             # its own zone: no stamp is asked
    j = _job(c, next(k for k, v in r["jobs"].items() if v == "run"))
    assert j["status"] == "done" and j["result"]["result"] == "PASS", j
    assert asked[n:] == [None] and "zone_selection" not in j["result"]
    assert "zones_expect" not in j["args"]
    _wait_state(c, ("READY",))


def test_start_with_zone_selection_needs_a_usable_selection_and_the_one_the_page_showed(web):
    c, app, asked = web
    _clear(c, app)
    assert c.post("/api/cmd/config", {"config": "sel.json"})[0] == 200
    _wait_state(c, ("READY",))
    n = len(asked)
    # nothing switched on: refused in the bus, the part PULLed, nothing captured
    status, r = c.post("/api/cmd/start", {"source": "test"})
    assert status == 409 and not r["accepted"], r
    assert "inspects the selected zones and none is switched on" in r["message"]
    b = r["bus"]
    assert b["state"] == "READY" and b["last_result"] == "INVALID" and b["last_disposition"] == "PULL"
    assert b["signals"]["PULL"] and asked[n:] == [] and r["jobs"] == {}
    # a run job sent to the server by hand, past the bus, with the stamp of that empty
    # selection: the run refuses it too - INVALID, PULL, never the whole field
    _, d = c.get("/api/zones/selection")
    assert d["selection"]["empty"]
    status, j = c.post("/api/jobs", {"verb": "run_station", "args": {
        "config": "sel.json", "zones_expect": d["selection"]["stamp"]}})
    assert status == 200, j
    res = _job(c, j["id"])["result"]
    assert res["result"] == "INVALID" and res["disposition"] == "PULL", res
    assert "is empty" in res["invalid_reason"] and asked[n:] == []
    # the same from a controller's TRIGGER / ARM: ARM is refused
    status, r = c.post("/api/cmd/arm", {"source": "plc"})
    assert status == 409 and "none is switched on" in r["message"] and r["bus"]["state"] == "READY"
    # a selection: START carries the stamp the page shows; another stamp starts nothing
    status, r = _select(c, [{"set": "cells", "on": True, "ids": ["Z1"]}])
    stamp = r["selection"]["stamp"]
    assert status == 200 and r["selection"]["label"] == "cells (1 of 2)"
    c.post("/api/cmd/config", {"config": "own.json"})      # (forget the PULL of the last START)
    assert c.post("/api/cmd/config", {"config": "sel.json"})[1]["bus"]["last_result"] is None
    status, r = c.post("/api/cmd/start", {"source": "quick", "zones": "0123456789ab"})
    assert status == 409 and "the zone selection changed (it is now: cells (1 of 2))" in r["message"]
    assert r["bus"]["last_result"] is None and asked[n:] == []          # refused: not a PULL
    status, r = c.post("/api/cmd/start", {"source": "quick", "zones": stamp})
    assert status == 200 and r["accepted"], r
    j = _job(c, next(k for k, v in r["jobs"].items() if v == "run"))
    res = j["result"]
    assert j["status"] == "done" and res["result"] == "PASS" and res["disposition"] == "CONTINUE", j
    assert j["args"]["zones_expect"] == stamp and asked[n:] == ["zoneset:cells:Z1"]
    assert res["zone_selection"]["source"] == "selection" and res["job"]["zone"] == "selection"
    assert res["zone_selection"]["selected"]["by"] == "operator page"
    assert [z["key"] for z in res["zone_selection"]["by_zone"]] == ["cells/Z1"]
    b = _wait_state(c, ("READY",))
    assert b["last_disposition"] == "CONTINUE" and b["signals"]["PASS"]
    # another preset is loaded than the run uses: the page shows another rig placement's sets
    other = next(n_ for n_ in app.preset_names() if n_ != PRESET)
    assert c.post(f"/api/presets/{other}/load", {})[0] == 200
    try:
        status, r = c.post("/api/cmd/start", {"source": "test"})
        assert status == 409 and f"with preset {PRESET}, and preset {other} is loaded" in r["message"]
        assert r["bus"]["last_disposition"] == "PULL"
    finally:
        assert c.post(f"/api/presets/{PRESET}/load", {})[0] == 200


def test_the_selection_is_fixed_while_the_station_is_armed(web):
    c, app, asked = web
    _clear(c, app)
    assert c.post("/api/cmd/config", {"config": "sel.json"})[0] == 200
    _wait_state(c, ("READY",))
    status, r = _select(c, [{"set": "cells", "on": True, "ids": None}])
    stamp = r["selection"]["stamp"]
    status, r = c.post("/api/cmd/arm", {"source": "quick", "zones": stamp})
    assert status == 200 and r["bus"]["state"] == "ARMED"
    # the switches, the painter and a delete are refused while armed
    status, r = _select(c, [{"set": "cells", "on": False, "ids": None}], was=stamp)
    assert status == 409 and not r["accepted"] and "refused while the station is ARMED" in r["message"]
    assert r["selection"]["label"] == "cells (2)" and r["selection"]["stamp"] == stamp     # as stored
    for verb, args in (("zone_paint", {"name": "cells", "source": "cells", "replace": True}),
                       ("zone_delete", {"name": "tube", "reason": "x", "confirm": True}),
                       ("zone_select", {"clear": True})):
        status, r = c.post("/api/jobs", {"verb": verb, "args": args})
        assert status == 409 and "refused while the station is ARMED" in r["error"], (verb, r)
    # a TRIGGER runs the armed selection
    n = len(asked)
    status, r = c.post("/api/cmd/trigger", {"source": "plc"})
    assert status == 200 and r["accepted"], r
    j = _job(c, next(k for k, v in r["jobs"].items() if v == "run"))
    assert j["result"]["result"] == "PASS" and j["args"]["zones_expect"] == stamp
    assert _wait_state(c, ("ARMED",))["state"] == "ARMED" and asked[n:] == ["zoneset:cells"]
    # the selection changes underneath (a terminal: `scan zone-select`), not through this
    # server: the next TRIGGER starts nothing, PULLs the part and disarms
    api.zone_select(on="tube", preset=app.session.preset, by="a terminal")
    status, r = c.post("/api/cmd/trigger", {"source": "plc"})
    assert status == 409 and "the zones changed since ARM" in r["message"], r
    assert "cells (2), tube (1)" in r["message"] and "disarmed" in r["message"]
    b = r["bus"]
    assert b["state"] == "READY" and b["last_disposition"] == "PULL" and asked[n:] == ["zoneset:cells"]


def test_start_is_refused_while_the_painter_holds_the_rig(web):
    c, app, asked = web
    _clear(c, app)
    assert c.post("/api/cmd/config", {"config": "own.json"})[0] == 200
    _wait_state(c, ("READY",))
    hold, entered = threading.Event(), threading.Event()
    real = app._zone_paint

    def slow_paint(a):
        entered.set()
        assert hold.wait(60), "the fake painter was never released"
        return real(a)
    app._zone_paint = slow_paint
    try:
        status, j = c.post("/api/jobs", {"verb": "zone_paint", "args": {
            "name": "held", "polarity": "normal", "select": True,
            "events": [{"type": "drag", "from": [100, 100], "to": [300, 300]},
                       {"type": "key", "key": "space"}]}})
        assert status == 200 and entered.wait(30)
        _, b = c.get("/api/cmd/state")
        assert b["painter"]["running"] and b["painter"]["verb"] == "zone_paint"
        assert b["painter"]["name"] == "held" and b["painter"]["job"] == j["id"]
        n = len(asked)
        for name in ("start", "arm"):
            status, r = c.post(f"/api/cmd/{name}", {"source": "test"})
            assert status == 409 and "the zone painter holds the rig" in r["message"], (name, r)
            assert r["bus"]["state"] == "READY" and r["bus"]["last_result"] is None and r["jobs"] == {}
        assert c.post("/api/cmd/mode", {"mode": "measure"})[0] == 200
        status, r = c.post("/api/measure", {"what": "flatness"})
        assert status == 409 and "the zone painter holds the rig" in r["error"]
        status, r = _select(c, [{"set": "cells", "on": True, "ids": None}])
        assert status == 409 and "still queued or running" in r["message"]
        assert asked[n:] == []
    finally:
        hold.set()
        app._zone_paint = real
        c.post("/api/cmd/mode", {"mode": "station"})
    done = _job(c, j["id"])
    assert done["status"] == "done" and done["result"]["saved"] and done["result"]["on"] is True
    _, b = c.get("/api/cmd/state")
    assert b["painter"] is None and b["last_paint"]["name"] == "held"
    assert b["last_paint"]["outcome"] == "saved" and b["last_paint"]["on"] is True
    assert b["zone_selection"]["label"] == "held (1)"
    _select(c, [{"set": "held", "on": False, "ids": None}])
    c.done("zone_delete", {"name": "held", "reason": "test cleanup", "confirm": True})

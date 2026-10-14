"""The operating modes on the command bus, over HTTP and on the operator page, MOCK only.

The bus holds the mode (SELECT_MODE): outside Station mode it starts no cycle and keeps READY,
PASS and every permission off, so a measurement or a quick check never reaches the line's
outputs. Over HTTP: POST /api/cmd/mode, GET /api/modes, POST /api/measure (numbers, a neutral
MEASURED / NOT_MEASURABLE, the run's files) and POST /api/quick-check (CONTINUE / REJECT
through the full mock capture path) - each refused outside its mode; a MOBILE rig is not
verified by a server start and HOME verifies it - also a mobile preset the server first sees
after its start; the bus snapshot does not read the rig-state record on every poll. The page:
the mode switch is there, and the real quick.js (node, fake DOM) shows the right panes, posts
the right verbs, fills camera px from clicks on the picture, never shows a verdict for a
measurement, keeps the station's keys to Station mode, follows the loaded preset and drops the
operator's confirmation on a mode or preset change."""

import json
import re
import shutil
import subprocess
import threading
import time

import pytest

from engine import api, rigstate
from engine import modes as M
from engine.control import commands as C

import recipe_synth as RS
from conftest import ROOT
from test_flow_mock import GOOD, OK, _answer, _flow_bus
from test_ui_server import Client

from ui.web import server as ui_server

STATIC = ROOT / "ui" / "web" / "static"
PRESET = "bench_9in_metric"
PHRASE = "mobile rig not verified: run rig-check / re-home / calibrate"


# ================================================================== the bus (pure)

def test_modes_are_three_and_station_is_the_default():
    assert M.MODES == ("station", "quick_check", "measure") and M.DEFAULT_MODE == "station"
    assert [M.check_mode(v) for v in ("Measure", " quick-check ", "QUICK", "station")] == [
        "measure", "quick_check", "quick_check", "station"]
    for bad in ("", "auto", None, 1, True, ["measure"]):
        with pytest.raises(ValueError, match="operating mode"):
            M.check_mode(bad)
    assert "SELECT_MODE" in C.COMMANDS
    bus = C.CommandBus(configs=["a.json"])
    snap = bus.snapshot()
    assert bus.mode == "station" and snap["operating_mode"] == "station"
    assert snap["operating_modes"] == list(M.MODES) and bus.signals()["READY"]
    ins, _ = C.load_io_doc({"inputs": [{"kind": "simulated", "map": {"DI5": "SELECT_MODE"}}]})
    ins[0].start(bus)                               # a hardware input can select the mode
    assert ins[0].inject("DI5", arg="measure").accepted and bus.mode == "measure"


def test_outside_station_mode_the_bus_starts_no_cycle_and_shows_no_pass():
    bus, calls, out = _flow_bus()
    bus.set_inputs({"part_present": True})
    assert bus.command("START").accepted            # one whole cycle: PASS, every permission
    for answer in (OK, GOOD, GOOD):
        assert bus.command("STEP").accepted
        _answer(bus, answer)
    assert out.current["PASS"] and out.current["READY"] and out.current["PERMIT_TRANSFER"]
    assert out.registers["result_code"] == C.RESULT_CODES["PASS"]
    # Measure mode: the last cycle is forgotten with its PASS and its permissions
    r = bus.command("SELECT_MODE", arg="measure", source="quick")
    assert r.accepted and r.message == "Measure mode" and bus.mode == "measure"
    assert bus.state == "READY" and bus.last_result is None and bus.cycle is None
    assert not out.current["PASS"] and not out.current["READY"] and not out.current["BUSY"]
    assert not any(v for k, v in out.current.items() if k.startswith("PERMIT_"))
    assert out.registers["result_code"] == 0 and bus.snapshot()["operating_mode"] == "measure"
    n = len(calls)
    for cmd in ("START", "ARM", "TRIGGER", "STEP", "ACK"):
        r = bus.command(cmd)
        assert not r.accepted, cmd
    assert "Measure mode" in bus.command("START").message
    assert "select Station mode" in bus.command("ARM").message
    assert len(calls) == n and bus.state == "READY"            # the owner was asked nothing
    # the rig check and ABORT work in every mode
    assert bus.command("HOME").accepted and bus.state == "HOMING" and out.current["BUSY"]
    bus.job_finished("home", "OK")
    assert bus.state == "READY" and not out.current["READY"]   # still not ready for cycles
    assert bus.command("ABORT").accepted
    # the same mode again changes nothing; an unknown one is refused
    assert bus.command("SELECT_MODE", arg={"mode": "measure"}).accepted
    r = bus.command("SELECT_MODE", arg="turbo")
    assert not r.accepted and "is not one of" in r.message and bus.mode == "measure"
    assert not bus.command("SELECT_MODE").accepted
    # quick check: the same hold
    assert bus.command("SELECT_MODE", arg="quick").accepted and bus.mode == "quick_check"
    assert not bus.command("START").accepted and "Quick check mode" in bus.command("START").message
    # back to Station: ready again, and a cycle starts
    assert bus.command("SELECT_MODE", arg="station").accepted and out.current["READY"]
    assert not out.current["PASS"] and bus.command("START").accepted and bus.state == "WAITING"


def test_the_mode_cannot_change_under_a_cycle_and_changing_it_disarms():
    bus, _, out = _flow_bus()
    bus.set_inputs({"part_present": True})
    assert bus.command("START").accepted and bus.state == "WAITING"
    r = bus.command("SELECT_MODE", arg="measure")
    assert not r.accepted and "WAITING" in r.message and bus.mode == "station"
    assert bus.command("STEP").accepted and bus.state == "RUNNING"
    assert not bus.command("SELECT_MODE", arg="measure").accepted
    _answer(bus, OK)
    assert out.current["PERMIT_LOAD_DONE"] and bus.mode == "station"
    assert bus.command("ABORT").accepted and bus.state == "READY"
    # ARMED for cycles: leaving Station mode disarms
    assert bus.command("ARM").accepted and bus.state == "ARMED"
    r = bus.command("SELECT_MODE", arg="quick_check")
    assert r.accepted and "disarmed" in r.message and bus.state == "READY"
    assert not bus.command("TRIGGER").accepted and bus.last_result is None
    # a FAULT (the rig check said MOVED) does not stop the switch, and stays a fault
    bus.command("HOME")
    bus.job_finished("home", "MOVED")
    assert bus.state == "FAULT" and bus.command("SELECT_MODE", arg="measure").accepted
    assert bus.state == "FAULT" and out.current["ALARM"] and not out.current["READY"]


# ================================================================== over HTTP

@pytest.fixture(scope="module")
def srv(tmp_path_factory):
    """A MOCK server on a library with the true stereo calibration, two synthetic metric
    scans and a rig reference, and a MOBILE rig (`cart`: its own geometry - a geometry_id
    names one rig) that was calibrated (its poses on the rig-state record), given its
    reference and scanned once, all BEFORE the server starts."""
    tmp = tmp_path_factory.mktemp("modes_ui")
    lib, stations = tmp / "library", tmp / "stations"
    stations.mkdir()
    doc = json.loads((ROOT / "stations" / "example_metric_check.json").read_text())
    doc.update(preset=PRESET, runs_dir=str(tmp / "runs"), mock_library=None)
    (stations / "flat_check.json").write_text(json.dumps(doc))
    s = api.configure(lib, mock=True)
    try:
        p = api.load_preset(PRESET)
        s.bench.use_stereo()
        RS.truth_calibration(s, p)
        from engine.capture.mock_stereo import Block
        scans = {"wall": RS.oracle_scan(s, p, [], "wall").id,
                 "box": RS.oracle_scan(s, p, [Block(center=(0.0, 0.0, 813.0),
                                                    half=(40.0, 25.0, 10.0))], "box").id}
        assert api.rig_check(p)["status"] == "REFERENCE_SET"       # right after the calibration
        api.preset_add("cart", PRESET, {"rig.mount": "mobile"}, geometry_id="cart_rig_v1")
        cart = api.load_preset("cart")
        RS.calibrated_now(s, cart)
        scans["cart_wall"] = RS.oracle_scan(s, cart, [], "cart_wall").id
        r = api.rig_check(cart)                     # stored while the rig was verified
        assert r["status"] == "REFERENCE_SET" and r["reference_trust"]["mobile"] is True
    finally:
        api._session = None
    httpd, app = ui_server.make_server(mock=True, library=lib, port=0, stations_dir=stations,
                                       preset=PRESET)
    t = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
    t.start()
    try:
        yield Client(app.port), app, scans
    finally:
        httpd.shutdown()
        httpd.server_close()
        app.close()
        api._session = None


def _job(c, jid, timeout=240.0):
    t0 = time.monotonic()
    while True:
        _, j = c.get(f"/api/jobs/{jid}")
        if j["status"] in ("done", "error"):
            return j
        assert time.monotonic() - t0 < timeout, j
        time.sleep(0.2)


def _state(c) -> dict:
    return c.get("/api/cmd/state")[1]


def _no_line_outputs(b: dict) -> None:
    """Nothing of a measurement / quick check is on the bus: no PASS, FAIL, PULL, permission."""
    sig = b["signals"]
    assert not any(sig[k] for k in ("PASS", "FAIL", "PULL", "KICKOUT", "READY"))
    assert not any(v for k, v in sig.items() if k.startswith("PERMIT_"))
    assert b["last_result"] is None and b["last_disposition"] is None
    assert b["registers"]["result_code"] == 0 and b["cycle"] is None


def test_the_page_has_the_mode_switch_and_the_modes_are_listed(srv):
    c, _, _ = srv
    status, raw, _ = c.request("GET", "/quick")
    html = raw.decode()
    assert status == 200
    for i in ("qModeMeasure", "qModeQuick", "qModeStation", "qPaneMeasure", "qPaneQuick",
              "qPaneStation", "qMeasure", "qValues", "qMeasImg", "qRigPreset"):
        assert f'id="{i}"' in html, i
    assert html.index('id="qModeMeasure"') < html.index('id="qModeQuick"') < html.index(
        'id="qModeStation"') < html.index("<main")                # the switch, at the top
    assert not re.search(r"\son[a-z]+\s*=", html), "inline handler (the CSP forbids them)"
    status, d = c.get("/api/modes")
    assert status == 200 and d["mode"] == "station"
    assert [m["name"] for m in d["modes"]] == ["station", "quick_check", "measure"]
    assert [m["title"] for m in d["modes"]] == ["Station", "Quick check", "Measure"]
    assert d["measure"]["whats"] == ["flatness", "step", "distance", "cylinder", "endface",
                                     "dims", "deviation"]
    assert d["quick_check"]["checks"] == ["flatness", "golden", "recipe"]
    by = {p["name"]: p for p in d["presets"]}
    assert by["cart"]["mount"] == "mobile" and by["cart"]["calibrated"] is True
    assert by[PRESET]["mount"] == "fixed" and d["loaded_preset"] == PRESET
    assert "flat_plate" in d["recipes"] and d["rig"]["ok"] is True
    b = _state(c)
    assert b["operating_mode"] == "station" and b["mode_jobs"] == [] and b["rig"]["mount"] == "fixed"
    verbs = {v["verb"]: v for v in c.get("/api/verbs")[1]["verbs"]}
    assert {"measure_run", "quick_check", "rig_session"} <= set(verbs)


def test_measure_and_quick_check_are_refused_outside_their_mode(srv):
    c, _, scans = srv
    for path, body in (("/api/measure", {"what": "flatness", "metric_ids": [scans["wall"]]}),
                       ("/api/quick-check", {"check": "flatness", "threshold_mm": 0.5})):
        status, r = c.post(path, body)
        assert status == 409 and "mode" in r["error"] and "POST /api/cmd/mode" in r["error"], r
    status, r = c.post("/api/jobs", {"verb": "measure_run", "args": {"what": "flatness"}})
    assert status == 409 and "Measure mode" in r["error"]
    assert c.post("/api/cmd/mode", {"mode": "turbo"})[0] == 409
    assert _state(c)["operating_mode"] == "station"


def test_measure_mode_over_http_gives_numbers_and_nothing_on_the_line(srv):
    c, app, scans = srv
    status, r = c.post("/api/cmd/mode", {"mode": "measure", "source": "test"})
    assert status == 200 and r["accepted"] and r["bus"]["operating_mode"] == "measure"
    _no_line_outputs(r["bus"])
    for cmd in ("start", "arm", "trigger"):
        status, r = c.post(f"/api/cmd/{cmd}", {})
        assert status == 409 and not r["accepted"], cmd
    assert "Measure mode" in c.post("/api/cmd/start", {})[1]["message"]
    assert c.post("/api/quick-check", {"check": "flatness", "threshold_mm": 0.5})[0] == 409
    # one measurement on a stored scan: the values, their uncertainty, the run's files
    status, j = c.post("/api/measure", {"what": "dims", "metric_ids": [scans["box"]],
                                        "source": "quick"})
    assert status == 200 and j["verb"] == "measure_run" and j["bus"]["operating_mode"] == "measure"
    j = _job(c, j["id"])
    assert j["status"] == "done", j
    res = j["result"]
    assert res["status"] == "MEASURED" and res["exit_code"] == 0 and res["operating_mode"] == "measure"
    assert res["result"] is None and res["disposition"] == "NONE"
    v = {x["name"]: x for x in res["values"]}
    assert v["length"]["value"] == pytest.approx(80.0, abs=0.4)
    assert v["height"]["value"] == pytest.approx(20.0, abs=0.05)
    assert all(x["uncertainty"]["value"] > 0 and x["units"] == "mm" for x in v.values())
    links = j["links"]
    for key, ctype in (("result_json", "application/json"), ("values_csv", "text/csv"),
                       ("preview", "image/png"), ("spec", "application/json"),
                       ("ply", "application/octet-stream")):
        status, raw, hdrs = c.request("GET", links[key])
        assert status == 200 and hdrs["Content-Type"].startswith(ctype) and raw, key
    assert links["preview"].endswith("/overlay.png")
    assert json.loads(c.request("GET", links["result_json"])[1])["run_id"] == res["run_id"]
    _no_line_outputs(_state(c))                     # the bus heard nothing of it
    assert app.bus_jobs() == {}                     # ... and it was never a bus job
    # not measurable: the job is done, its answer is NOT_MEASURABLE - no values, no picture
    j = _job(c, c.post("/api/measure", {"what": "dims", "metric_ids": [scans["wall"]]})[1]["id"])
    assert j["status"] == "done" and j["result"]["status"] == "NOT_MEASURABLE"
    assert j["result"]["reason"] == "too_few_points" and j["result"]["values"] == []
    assert j["result"]["exit_code"] == 5 and "preview" not in j["links"]
    # a malformed measurement: the job fails with every problem named, nothing ran
    j = _job(c, c.post("/api/measure", {"what": "step", "roi": [1, 2, 30, 40]})[1]["id"])
    assert j["status"] == "error" and "step needs two regions" in j["error"]
    assert c.post("/api/measure", {"what": "flatness", "colour": "red"})[0] == 400
    _no_line_outputs(_state(c))


def test_the_bus_snapshot_does_not_read_the_rig_record_on_every_poll(srv, monkeypatch):
    """/quick polls GET /api/cmd/state twice a second: the rig's state in it is cached while
    the rig-state record and the calibration folder are unchanged, and a new line is seen at
    once. In Station mode a fixed rig is not asked strictly (nothing is read for it)."""
    c, app, _ = srv
    asked = []
    real = rigstate.gate
    monkeypatch.setattr(rigstate, "gate",
                        lambda *a, **k: (asked.append(k.get("strict")), real(*a, **k))[1])
    monkeypatch.setattr(ui_server, "RIG_GATE_TTL_S", 600.0)    # (a slow machine: no expiry here)
    app._rig_cache.clear()
    assert _state(c)["operating_mode"] == "measure"
    for _ in range(20):
        assert _state(c)["rig"] == {"ok": True, "code": None, "message": "", "how_to_fix": "",
                                    "mount": "fixed"}
    assert asked == [True]                          # strict, once: the other 20 were cached
    p = app.session.preset
    assert p.name == PRESET
    rigstate.append(app.session.store, p.geometry_id, {
        "event": "session", "reason": "a line more", "by": "test", "preset": p.name,
        "mount": "fixed"})
    _state(c)
    assert asked == [True, True]                    # the record grew: read again, at once
    _state(c)
    assert len(asked) == 2


def test_a_mobile_rig_over_http_is_unverified_until_home(srv):
    """The server's start is a new rig session: the mobile preset gives no mm until HOME (the
    rig check against its reference, stored while it was verified) says OK. Then one
    measurement through the full mock capture path."""
    c, app, scans = srv
    assert _state(c)["operating_mode"] == "measure"
    j = _job(c, c.post("/api/measure", {"what": "flatness", "preset": "cart",
                                        "metric_ids": [scans["cart_wall"]]})[1]["id"])
    res = j["result"]
    assert j["status"] == "done" and res["status"] == "NOT_MEASURABLE", res
    assert res["reason"] == "rig_unverified" and res["message"].startswith(PHRASE)
    assert res["preset"] == {"name": "cart", "geometry_id": res["preset"]["geometry_id"],
                             "mount": "mobile"} and res["values"] == []
    b = _state(c)
    assert b["active_preset"] == "cart" and b["rig"]["mount"] == "mobile"
    assert b["rig"]["ok"] is False and b["rig"]["message"].startswith(PHRASE)
    assert c.get("/api/modes")[1]["rig"]["ok"] is False
    # HOME = the rig check, in every mode (the simulated bench as the reference saw it: the
    # empty wall of the stereo bench)
    app.call(lambda: (api.mock_bench().clear(), api.mock_bench().use_stereo()))
    status, r = c.post("/api/cmd/home", {})
    assert status == 200 and r["bus"]["state"] == "HOMING"
    home = _job(c, next(k for k, v in r["jobs"].items() if v == "home"))
    assert home["result"]["status"] == "OK", home["result"]
    t0 = time.monotonic()
    while _state(c)["state"] != "READY":
        assert time.monotonic() - t0 < 30
        time.sleep(0.1)
    b = _state(c)
    assert b["rig"]["ok"] is True and b["operating_mode"] == "measure"
    _no_line_outputs(b)
    # the stored scan was made before the verification: still not covered by it
    j = _job(c, c.post("/api/measure", {"what": "flatness",
                                        "metric_ids": [scans["cart_wall"]]})[1]["id"])
    assert j["result"]["reason"] == "rig_unverified" and "stored scan" in j["result"]["message"]
    # a live measurement (the simulated bench, the fast scan): MEASURED, with a prediction
    j = _job(c, c.post("/api/measure", {"what": "flatness", "scan_kind": "fast",
                                        "roi": [500, 400, 900, 700]})[1]["id"])
    res = j["result"]
    assert j["status"] == "done" and res["status"] == "MEASURED", res
    assert "mobile rig verified by the rig check OK" in res["validity"]
    assert res["precision"]["predicted_sigma_z_mm"] > 0 and len(res["scans"]) == 1
    assert {x["name"] for x in res["values"]} >= {"flatness_rms", "flatness_p2_p98", "tilt"}
    _no_line_outputs(_state(c))
    # the scan records the verification it was captured under
    meta = json.loads((app.root / "metric" / res["preset"]["geometry_id"] / res["scans"][0]
                       / "meta.json").read_text(encoding="utf-8"))
    assert meta["rig"]["mount"] == "mobile" and "rig check OK" in meta["rig"]["verified_by"]


def test_quick_check_mode_over_http_continue_and_reject(srv):
    c, app, scans = srv
    status, r = c.post("/api/cmd/mode", {"mode": "quick_check"})
    assert status == 200 and r["bus"]["operating_mode"] == "quick_check"
    assert c.post("/api/measure", {"what": "flatness"})[0] == 409
    assert c.post("/api/cmd/start", {})[0] == 409
    # the flat wall within 0.5 mm: CONTINUE (the full mock capture path)
    status, j = c.post("/api/quick-check", {"check": "flatness", "threshold_mm": 0.5,
                                            "preset": PRESET, "mock_part": "none"})
    assert status == 200 and j["verb"] == "quick_check"
    j = _job(c, j["id"])
    assert j["status"] == "done", j
    res = j["result"]
    assert res["result"] == "PASS" and res["disposition"] == "CONTINUE" and res["exit_code"] == 0
    assert res["operating_mode"] == "quick_check" and res["station"] == "quick_check"
    assert res["permissions"] == {} and res["steps"][0]["expects"] == {}
    assert res["capture"]["preset"] == PRESET and res["job"]["config"] == "quick_check.json"
    st, raw, _ = c.request("GET", f"/runs/{res['run_id']}/quick_check.json")
    assert st == 200 and json.loads(raw)["threshold_mm"] == 0.5
    assert c.request("GET", j["links"]["result_json"])[0] == 200
    _no_line_outputs(_state(c))                     # the lamp is the page's own: not the line's
    # a block on the wall: REJECT
    j = _job(c, c.post("/api/quick-check", {"check": "flatness", "threshold_mm": 0.5,
                                            "mock_part": "80,50,20"})[1]["id"])
    res = j["result"]
    assert res["result"] == "FAIL" and res["disposition"] == "REJECT" and res["exit_code"] == 1
    _no_line_outputs(_state(c))
    # a bad quick check: the job reports the refusal, nothing ran
    j = _job(c, c.post("/api/quick-check", {"check": "golden", "threshold_mm": 0.5})[1]["id"])
    assert j["status"] == "error" and "golden" in j["error"]
    # back to Station: ready for cycles again, the mode verbs refused again
    status, r = c.post("/api/cmd/mode", {"mode": "station"})
    assert status == 200 and r["bus"]["signals"]["READY"] and not r["bus"]["signals"]["PASS"]
    assert c.post("/api/quick-check", {"check": "flatness", "threshold_mm": 0.5})[0] == 409


def test_a_mobile_preset_first_seen_after_the_server_started_is_held_by_its_session(tmp_path):
    """A mobile preset the server did not know when it started gets no session marker at
    start-up - but the server's own start is the floor of every session: what was calibrated
    before it verifies nothing, in Station mode and in Quick check, the first time and the
    second. HOME then verifies it, and it stays verified (no marker is written after a job)."""
    lib, stations = tmp_path / "library", tmp_path / "stations"
    stations.mkdir()
    doc = json.loads((ROOT / "stations" / "example_metric_check.json").read_text())
    doc.update(preset="late", runs_dir=str(tmp_path / "runs"), mock_library=None)
    (stations / "late_check.json").write_text(json.dumps(doc))
    s = api.configure(lib, mock=True)
    try:
        p = api.load_preset(PRESET)
        s.bench.use_stereo()
        RS.calibrated_now(s, p)                     # calibrated, and its reference stored,
        assert api.rig_check(p)["reference_trust"]["mobile"] is True   # before the server starts
        geo = p.geometry_id
    finally:
        api._session = None
    httpd, app = ui_server.make_server(mock=True, library=lib, port=0, stations_dir=stations,
                                       preset=PRESET)
    threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True).start()
    c = Client(app.port)

    def events() -> list:
        return [r["event"] for r in rigstate.read(app.session.store, geo)]

    def quick() -> dict:
        j = _job(c, c.post("/api/quick-check", {"check": "flatness", "threshold_mm": 0.5,
                                                "preset": "late", "mock_part": "none"})[1]["id"])
        assert j["status"] == "done", j
        return j["result"]
    try:
        assert "session" not in events()            # no mobile preset at start-up: no marker
        j = c.job("preset_add", {"name": "late", "base": PRESET,
                                 "overrides": {"rig.mount": "mobile"}})
        assert j["status"] == "done", j
        # Station mode: a single-job config that names the late preset
        j = c.job("run_station", {"config": "late_check.json", "mock_part": "none"}, timeout=300)
        res = j["result"]
        assert res["result"] == "INVALID" and res["disposition"] == "PULL" and res["exit_code"] == 5
        assert [r["reason"] for r in res["pull_reasons"]] == ["rig_unverified"]
        assert "this program started" in res["pull_reasons"][0]["message"]
        # Quick check, twice, nothing changed in between: the same answer
        assert c.post("/api/cmd/mode", {"mode": "quick_check"})[0] == 200
        for _ in (1, 2):
            res = quick()
            assert res["disposition"] == "PULL" and res["capture"]["preset"] == "late"
            assert [r["reason"] for r in res["pull_reasons"]] == ["rig_unverified"]
        assert events().count("session") == 1       # written BEFORE the first quick check
        # HOME: the rig check against the reference stored while the rig was verified
        app.call(lambda: (api.mock_bench().clear(), api.mock_bench().use_stereo()))
        status, r = c.post("/api/cmd/home", {})
        home = _job(c, next(k for k, v in r["jobs"].items() if v == "home"))
        assert home["result"]["status"] == "OK", home["result"]
        t0 = time.monotonic()
        while _state(c)["state"] != "READY":
            assert time.monotonic() - t0 < 30
            time.sleep(0.1)
        assert _state(c)["rig"]["ok"] is True and _state(c)["active_preset"] == "late"
        res = quick()                               # the full mock capture path
        assert res["result"] == "PASS" and res["disposition"] == "CONTINUE", res
        assert events().count("session") == 1 and _state(c)["rig"]["ok"] is True
        # the fixed preset it was copied from is the same rig: gated as mobile too
        by = {i["name"]: i for i in c.get("/api/modes")[1]["presets"]}
        assert by[PRESET]["mount"] == "mobile" and by[PRESET]["mount_declared_by"] == "late"
    finally:
        httpd.shutdown()
        httpd.server_close()
        app.close()
        api._session = None
    assert str(lib.resolve()) not in rigstate._PROCESS_SESSIONS     # its session ended with it


# ================================================================== quick.js (node, fake DOM)

_MODES_HARNESS = r"""
const fs = require('fs'), vm = require('vm');
class El {
  constructor(tag, id) { this.tagName = String(tag).toUpperCase(); this.id = id; this.children = [];
    this.attrs = {}; this.dataset = {}; this.className = ''; this._text = ''; this._value = '';
    this.checked = false; this.disabled = false; this.hidden = false; this._sel = undefined;
    this.parent = null; this.isContentEditable = false; this.href = ''; this.src = '';
    this.on = {}; this.classes = {};
    this.classList = { toggle: (c, v) => { this.classes[c] = !!v; }, add() {}, remove() {} }; }
  set textContent(t) { this._text = String(t); this.children = []; }
  get textContent() { return this._text + this.children.map((c) => c.textContent).join(''); }
  appendChild(c) { this.children.push(c); c.parent = this; return c; }
  get options() { return this.children.filter((c) => c.tagName === 'OPTION'); }
  get value() {
    if (this.tagName !== 'SELECT') return this._value;
    const o = this.options;
    if (this._sel === null) return '';
    if (this._sel && o.includes(this._sel)) return this._sel.value;
    return o.length ? o[0].value : '';
  }
  set value(v) {
    if (this.tagName !== 'SELECT') { this._value = String(v); return; }
    this._sel = this.options.find((x) => x.value === String(v)) || null;
  }
  addEventListener(ev, fn) { this.on[ev] = fn; }
  matches(sel) { return sel.split(',').map((s) => s.trim()).some((s) => this.tagName === s.toUpperCase()); }
  closest(sel) { for (let e = this; e; e = e.parent) if (e.matches(sel)) return e; return null; }
}
const TAGS = { qStation: 'select', qProfile: 'select', qRigPreset: 'select', qQuickRecipe: 'select',
  qConfirm: 'input', qStart: 'button', qMeasure: 'button', qAbort: 'button', qHome: 'button',
  qModeMeasure: 'button', qModeQuick: 'button', qModeStation: 'button', qMeasImg: 'img' };
const byId = {};
const el = (id) => byId[id] || (byId[id] = new El(TAGS[id] || 'div', id));
const sent = [];
const srv = { mode: 'mock', jobs: {}, n: 6,
  bus: { state: 'READY', operating_mode: 'station', station_config: 'a.json',
         station_configs: ['a.json'], unusable: {}, signals: {}, jobs: {}, mode_jobs: [],
         active_preset: 'bench', run_preset: 'bench',
         // the zone selection the server sends with every bus state: nothing switched on
         zone_selection: { state: 'ok', ok: true, empty: true, text: '', label: 'whole field',
                           stamp: 's0', problems: [], sets: [], check: [], n_zones_on: 0,
                           preset: 'bench', geometry_id: 'g' },
         painter: null, last_paint: null,
         rig: { ok: true, mount: 'fixed', message: '' } } };
const sandbox = {
  console, setTimeout, clearTimeout, setInterval, clearInterval,
  document: { addEventListener() {}, querySelector: (s) => el(s.slice(1)), querySelectorAll: () => [],
              createElement: (t) => new El(t), body: new El('body') },
  fetch: async (path, opt) => {
    const body = opt && opt.body ? JSON.parse(opt.body) : undefined;
    sent.push({ path, body });
    let data = {}, status = 200;
    if (path === '/api/cmd/state') data = { ...srv.bus, mode: srv.mode };
    else if (path === '/api/stations') data = { mode: srv.mode, stations: [{ config: 'a.json',
      station_id: 'a', part_label: 'p', job_type: 'metric', preset: null }] };
    else if (path === '/api/modes') data = { mode: srv.bus.operating_mode,
      loaded_preset: srv.bus.active_preset,
      presets: [{ name: 'bench', mount: 'fixed', calibrated: true },
                { name: 'cart', mount: 'mobile', calibrated: true },
                { name: 'raw', mount: 'fixed', calibrated: false },
                { name: 'broken', valid: false }],
      recipes: ['flat_plate', 'box_lid'] };
    else if (path === '/api/cmd/mode') {
      if (srv.refuseMode) { status = 409; data = { accepted: false, command: 'SELECT_MODE',
        message: 'refused while RUNNING', bus: { ...srv.bus } }; }
      else { srv.bus = { ...srv.bus, operating_mode: body.mode };
        data = { accepted: true, command: 'SELECT_MODE', message: 'ok', bus: { ...srv.bus }, jobs: {} }; }
    } else if (path.startsWith('/api/cmd/')) data = { accepted: true,
      command: path.split('/').pop().toUpperCase(), message: 'ok', bus: { ...srv.bus }, jobs: {} };
    else if (path === '/api/measure' || path === '/api/quick-check') {
      if (srv.refuseJob) { status = 409; data = { error: 'select the mode first' }; }
      else { srv.n += 1; data = { id: `j${srv.n}`, status: 'queued', bus: { ...srv.bus } }; }
    } else if (path.startsWith('/api/jobs/')) data = srv.jobs[path.split('/').pop()] || { status: 'running' };
    else if (path.startsWith('/api/presets/')) {
      srv.bus = { ...srv.bus, active_preset: decodeURIComponent(path.split('/')[3]) };
    }
    return { ok: status < 400, status, json: async () => data };
  },
};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(process.argv[2], 'utf8'), sandbox, { filename: 'quick.js' });
const run = (code) => vm.runInContext(code, sandbox);
const out = {};
const sentTo = (p) => sent.filter((s) => s.path === p);
const key = (k) => {
  const n = sent.length;
  let prevented = false;
  run('onKey')({ key: k, target: sandbox.document.body, ctrlKey: false, altKey: false,
                 metaKey: false, preventDefault() { prevented = true; } });
  return { prevented, n };
};
const settle = () => new Promise((r) => setTimeout(r, 20));
const panes = () => ({ station: !el('qPaneStation').hidden, extras: !el('qStationExtras').hidden,
  rig: !el('qPaneRig').hidden, measure: !el('qPaneMeasure').hidden, quick: !el('qPaneQuick').hidden,
  measureBtn: !el('qMeasure').hidden, startBtn: !el('qStart').hidden,
  on: ['qModeMeasure', 'qModeQuick', 'qModeStation'].filter((i) => el(i).classes.on) });
const lampNow = () => ({ text: el('qLamp').textContent, cls: el('qLamp').className });
const measured = (over) => ({ id: 'x', verb: 'measure_run', status: 'done',
  links: { result_json: '/runs/r1/result.json', values_csv: '/runs/r1/values.csv',
           preview: '/runs/r1/overlay.png', ply: '/runs/r1/scan.ply' },
  result: { status: 'MEASURED', what: 'flatness', repeats: 1, run_id: 'r1', reason: null,
            validity: 'MEASURED: mm from stereo calibration c1, fixed rig, 1 scan(s)',
            values: [{ name: 'flatness_rms', label: 'Flatness (RMS)', value: 0.0241, units: 'mm',
                       n_points: 120000, near_resolution_limit: true,
                       notes: ['near the resolution limit: mostly the instrument'],
                       uncertainty: { value: 0.0262, units: 'mm', method: 'measured_noise', n_repeats: 1 } },
                     { name: 'tilt', label: 'Tilt', value: 0.02, units: 'deg', n_points: 120000,
                       near_resolution_limit: false, notes: [],
                       uncertainty: { value: 0.03, units: 'deg', method: 'repeatability', n_repeats: 3 } }],
            result: null, disposition: 'NONE', ...over } });

(async () => {
  await run('loadStations()');
  out.station = panes();
  out.stationInfo = el('qModeInfo').textContent;
  // ---- the switch: Measure
  await run("selectMode('measure')");
  await settle();
  out.modePost = sentTo('/api/cmd/mode').pop().body;
  out.measure = panes();
  out.presets = el('qRigPreset').options.map((o) => o.textContent);
  out.presetShown = el('qRigPreset').value;
  out.askedModes = sentTo('/api/modes').length;
  out.lampAfterSwitch = lampNow();
  // a refused switch: the page keeps showing the bus's mode, and says why
  srv.refuseMode = true;
  await run("selectMode('station')");
  out.refusedSwitch = { panes: panes(), notice: el('qNotice').textContent };
  srv.refuseMode = false;
  // ---- clicks on the picture: camera px (the picture is drawn at half size)
  el('qMeasImg').naturalWidth = 1440; el('qMeasImg').naturalHeight = 1080;
  el('qMeasImg').clientWidth = 720; el('qMeasImg').clientHeight = 540;
  el('qMeasWhat').value = 'distance';
  run('measFields()');
  out.distanceRows = { p1: !el('qMeasP1Row').hidden, roi2: !el('qMeasRoi2Row').hidden,
                       golden: !el('qMeasGoldenRow').hidden };
  run('onPreviewClick')({ offsetX: 100, offsetY: 50 });
  run('onPreviewClick')({ offsetX: 300.4, offsetY: 50 });
  out.points = [el('qMeasP1').value, el('qMeasP2').value];
  await run('sendMeasure()');
  out.distancePost = sentTo('/api/measure').pop().body;
  srv.jobs[`j${srv.n}`] = measured({ what: 'distance' });
  await run('pollOnce()');
  el('qMeasWhat').value = 'step';
  run('measFields()');
  for (const [x, y] of [[50, 40], [10, 20], [200, 100], [260, 160]]) run('onPreviewClick')({ offsetX: x, offsetY: y });
  out.regions = [el('qMeasRoi').value, el('qMeasRoi2').value];
  out.hint = el('qMeasHint').textContent;
  el('qMeasRepeats').value = '3';
  el('qMeasScan').value = 'fast';
  el('qMockPart').value = 'scene:plate';
  // ---- M = MEASURE
  const k = key('m');
  await settle();
  out.mKey = { prevented: k.prevented, sent: sent.slice(k.n).map((s) => s.path) };
  out.stepPost = sentTo('/api/measure').pop().body;
  out.lampBusy = lampNow();
  out.busyButtons = { measure: el('qMeasure').disabled, modes: el('qModeStation').disabled };
  // the job ends MEASURED: a neutral lamp, the values with their uncertainty - never a verdict,
  // whatever the document claims
  srv.jobs[`j${srv.n}`] = measured({ result: 'PASS', disposition: 'CONTINUE' });
  await run('pollOnce()');
  out.measuredLamp = lampNow();
  out.table = { hidden: el('qValues').hidden,
                rows: el('qValuesBody').children.map((r) => r.children.map((c) => c.textContent)),
                near: el('qValuesBody').children.map((r) => r.className) };
  out.measuredPanel = { verdict: el('qVerdict').textContent, disp: el('qDisposition').textContent,
                        pull: el('qPull').hidden, validity: el('qValidity').textContent,
                        img: el('qMeasImg').src.split('?')[0], imgHidden: el('qMeasImg').hidden,
                        csv: el('qValuesLink').href, ply: el('qPlyLink').href };
  await run('pollOnce()');                         // the polls never turn it into the bus's lamp
  out.lampKept = lampNow();
  out.buttonsAfter = { measure: el('qMeasure').disabled, modes: el('qModeStation').disabled };
  // Space is the mode's own action; a malformed input is refused on the page, nothing is sent
  const sp = key(' ');
  await settle();
  out.spaceMeasure = sent.slice(sp.n).map((s) => s.path);
  srv.jobs[`j${srv.n}`] = { id: 'y', verb: 'measure_run', status: 'done', links: {},
    result: { status: 'NOT_MEASURABLE', reason: 'rig_unverified', values: [], result: null,
              disposition: 'NONE', message: 'mobile rig not verified: run rig-check / re-home / calibrate',
              how_to_fix: 'scan rig-check' } };
  await run('pollOnce()');
  out.notMeasurable = { lamp: lampNow(), reason: el('qReason').textContent,
                        table: el('qValues').hidden, verdict: el('qVerdict').textContent };
  el('qMeasRoi').value = '1,2,three';
  const nBad = sentTo('/api/measure').length;
  await run('sendMeasure()');
  out.badInput = { sent: sentTo('/api/measure').length - nBad, notice: el('qNotice').textContent };
  el('qMeasRoi').value = '';
  // a mobile rig that is not verified: the page says so, with what to press
  srv.bus = { ...srv.bus, active_preset: 'cart',
              rig: { ok: false, mount: 'mobile', message: 'mobile rig not verified: run rig-check / re-home / calibrate' } };
  await run('pollOnce()');
  out.rigInfo = { text: el('qRigInfo').textContent, warn: el('qRigInfo').classes.qwarn };
  // ---- Quick check
  await run("selectMode('quick_check')");
  await settle();
  out.quick = panes();
  out.quickLamp = lampNow();
  out.recipes = el('qQuickRecipe').options.map((o) => o.value);
  const q0 = key(' ');                             // no threshold yet: nothing is sent
  await settle();
  out.quickNoThreshold = { sent: sent.slice(q0.n).map((s) => s.path), notice: el('qNotice').textContent };
  el('qQuickThreshold').value = '0.5';
  el('qQuickRoi').value = '10,20,300,400';
  el('qQuickGuard').value = '10';
  const q1 = key('Enter');
  await settle();
  out.quickSent = sent.slice(q1.n).map((s) => s.path);
  out.quickPost = sentTo('/api/quick-check').pop().body;
  const quick = (res) => ({ id: 'q', verb: 'quick_check', status: 'done',
    links: { result_json: '/runs/q1/result.json' },
    result: { run_id: 'q1', deviation_mm: { max: 0.1, method: 'plane_fit' }, deviation_units: 'mm',
              zones: [], warnings: [], pull_reasons: [], ...res } });
  srv.jobs[`j${srv.n}`] = quick({ result: 'PASS', disposition: 'CONTINUE' });
  await run('pollOnce()');
  out.quickPass = { lamp: lampNow(), disp: el('qDisposition').textContent };
  await run('pollOnce()');
  out.quickPassKept = lampNow();
  await run('sendQuick()');
  srv.jobs[`j${srv.n}`] = quick({ result: 'FAIL', disposition: 'REJECT' });
  await run('pollOnce()');
  out.quickFail = lampNow();
  await run('sendQuick()');
  srv.jobs[`j${srv.n}`] = quick({ result: 'PASS', disposition: 'PULL',
    pull_reasons: [{ reason: 'guard_band', message: 'within 10 % of its limit' }] });
  await run('pollOnce()');
  out.quickPull = { lamp: lampNow(), reasons: el('qPull').textContent };
  el('qQuickCheck').value = 'recipe';
  run('quickFields()');
  el('qQuickRecipe').value = 'box_lid';
  await run('sendQuick()');
  out.recipePost = sentTo('/api/quick-check').pop().body;
  out.recipeRows = { threshold: !el('qQuickThresholdRow').hidden, recipe: !el('qQuickRecipeRow').hidden };
  srv.jobs[`j${srv.n}`] = quick({ result: 'PASS', disposition: 'CONTINUE' });
  await run('pollOnce()');
  // the M key means nothing outside Measure mode
  const m2 = key('m');
  out.mOutside = { prevented: m2.prevented, sent: sent.length - m2.n };
  // the server refuses the job (another client changed the mode): said, nothing shown as run
  srv.refuseJob = true;
  await run('sendQuick()');
  out.refusedJob = el('qNotice').textContent;
  srv.refuseJob = false;
  // ---- back to Station: everything as before
  await run("selectMode('station')");
  await settle();
  out.back = panes();
  out.backLamp = lampNow();
  const s2 = key(' ');
  await settle();
  out.spaceStation = sent.slice(s2.n).map((s) => s.path);
  // the real rig: a quick check needs the operator's confirmation, a measurement does not
  srv.mode = 'real';
  await run('loadStations()');
  await run("selectMode('quick_check')");
  await settle();
  el('qQuickCheck').value = 'flatness';
  run('refreshButtons()');
  out.realQuick = { start: el('qStart').disabled, confirmRow: !el('qConfirmRow').hidden };
  el('qConfirm').checked = true;
  run('refreshButtons()');
  await run('sendQuick()');
  out.realQuickPost = sentTo('/api/quick-check').pop().body;
  out.confirmCleared = el('qConfirm').checked;
  srv.jobs[`j${srv.n}`] = quick({ result: 'PASS', disposition: 'CONTINUE' });
  await run('pollOnce()');
  await run("selectMode('measure')");
  await settle();
  run('refreshButtons()');
  out.realMeasure = { measure: el('qMeasure').disabled, confirmRow: !el('qConfirmRow').hidden };
  await settle();
  // ---- the station's keys work in Station mode only: its selectors are hidden here, and a
  // digit typed after a click on the measurement picture must not select a configuration
  srv.mode = 'mock';
  await run('loadStations()');
  out.stationKeys = {};
  for (const k of ['2', 'p', 't', 'a', 'n', 'k']) {
    const r = key(k);
    await settle();
    out.stationKeys[k] = { prevented: r.prevented, sent: sent.slice(r.n).map((s) => s.path) };
  }
  const kh = key('h');                             // HOME (the rig check) works in every mode
  await settle();
  out.homeKey = sent.slice(kh.n).map((s) => s.path);
  // ---- the picker follows the LOADED preset, and another preset drops the confirmation
  srv.mode = 'real';
  await run('loadStations()');
  await run("selectMode('quick_check')");
  await settle();
  el('qConfirm').checked = true;
  srv.bus = { ...srv.bus, active_preset: 'raw' };  // another client / a PLC loaded another one
  await run('pollOnce()');
  out.followed = { picker: el('qRigPreset').value, confirm: el('qConfirm').checked,
                   info: el('qRigInfo').textContent };
  el('qConfirm').checked = true;
  el('qRigPreset').value = 'bench';
  await run("selectRigPreset('bench')");
  await settle();
  out.picked = { confirm: el('qConfirm').checked, loaded: srv.bus.active_preset,
                 picker: el('qRigPreset').value, start: el('qStart').disabled };
  // ... and so does a mode change, in both directions
  el('qConfirm').checked = true;
  await run("selectMode('station')");
  await settle();
  out.confirmAfterMode = el('qConfirm').checked;
  const s3 = key(' ');
  await settle();
  out.spaceUnconfirmed = sent.slice(s3.n).map((s) => s.path);
  out.labelInStation = el('qNumLabel').textContent;
  // ---- a measurement that ends after the mode was left (another client) shows nothing
  srv.mode = 'mock';
  await run('loadStations()');
  await run("selectMode('measure')");
  await settle();
  out.labelInMeasure = el('qNumLabel').textContent;
  await run('sendMeasure()');
  srv.jobs[`j${srv.n}`] = measured();
  srv.bus = { ...srv.bus, operating_mode: 'station' };
  await run('pollOnce()');
  out.leftMode = { lamp: lampNow(), img: !el('qMeasImg').hidden, values: !el('qValues').hidden,
                   label: el('qNumLabel').textContent, verdict: el('qVerdict').textContent };
  // ---- a mode switch during a FAULT keeps the fault's lamp
  srv.bus = { ...srv.bus, state: 'FAULT' };
  await run("selectMode('measure')");
  await settle();
  out.faultLamp = lampNow();
  process.stdout.write(JSON.stringify(out));
})().catch((e) => { console.error(e); process.exit(1); });
"""


def test_quick_js_mode_switch_posts_the_right_verbs_and_a_measurement_shows_no_verdict(tmp_path):
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed; quick.js logic not checked")
    h = tmp_path / "modes_harness.js"
    h.write_text(_MODES_HARNESS, encoding="utf-8")
    r = subprocess.run([node, str(h), str(STATIC / "quick.js")], capture_output=True, text=True,
                       encoding="utf-8", timeout=120)
    assert r.returncode == 0, r.stderr
    o = json.loads(r.stdout)
    station = {"station": True, "extras": True, "rig": False, "measure": False, "quick": False,
               "measureBtn": False, "startBtn": True, "on": ["qModeStation"]}
    assert o["station"] == station and "work states" in o["stationInfo"]
    # ---- Measure: the switch posts SELECT_MODE, the panes follow the bus
    assert o["modePost"] == {"mode": "measure", "source": "quick"}
    assert o["measure"] == {"station": False, "extras": False, "rig": True, "measure": True,
                            "quick": False, "measureBtn": True, "startBtn": False,
                            "on": ["qModeMeasure"]}
    assert o["presets"] == ["bench", "cart [mobile]", "raw (no stereo calibration)"]
    assert o["presetShown"] == "bench" and o["askedModes"] >= 1
    assert o["lampAfterSwitch"]["cls"] == "qlamp lamp-ready"
    assert o["refusedSwitch"]["panes"] == o["measure"]
    assert "refused while RUNNING" in o["refusedSwitch"]["notice"]
    # ---- clicks -> camera px; the measurement posts what it reads, and only that
    assert o["distanceRows"] == {"p1": True, "roi2": False, "golden": False}
    assert o["points"] == ["200,100", "601,100"]
    # (zone: the zone selection exactly as the page showed it - here: nothing switched on)
    assert o["distancePost"] == {"what": "distance", "source": "quick", "scan_kind": "metric",
                                 "preset": "bench", "p1": [200, 100], "p2": [601, 100],
                                 "zone": "selection@s0"}
    assert o["regions"] == ["20,40,101,81", "400,200,521,321"]
    assert "region B" in o["hint"]
    assert o["mKey"] == {"prevented": True, "sent": ["/api/measure"]}
    assert o["stepPost"] == {"what": "step", "source": "quick", "scan_kind": "fast",
                             "preset": "bench", "roi": [20, 40, 101, 81],
                             "roi2": [400, 200, 521, 321], "repeats": 3,
                             "zone": "selection@s0", "mock_scene": "scene:plate"}
    assert o["lampBusy"] == {"text": "MEASURING", "cls": "qlamp lamp-running"}
    assert o["busyButtons"] == {"measure": True, "modes": True}
    # ---- MEASURED: neutral, a table with uncertainty, never a PASS / CONTINUE
    assert o["measuredLamp"] == {"text": "MEASURED", "cls": "qlamp lamp-measured"}
    assert o["lampKept"] == o["measuredLamp"]
    t = o["table"]
    assert t["hidden"] is False and t["near"] == ["near", "note", ""]
    assert t["rows"][0] == ["Flatness (RMS)", "0.0241 mm", "± 0.0262 mm", "measured_noise",
                            "120000"]
    assert t["rows"][1] == ["near the resolution limit: mostly the instrument"]
    assert t["rows"][2] == ["Tilt", "0.020 deg", "± 0.030 deg", "repeatability ×3", "120000"]
    p = o["measuredPanel"]
    assert "none" in p["verdict"] and p["disp"] == "-" and p["pull"] is True
    assert "PASS" not in json.dumps([o["measuredLamp"], p]) and "CONTINUE" not in json.dumps(p)
    assert p["validity"].startswith("MEASURED: mm") and p["img"] == "/runs/r1/overlay.png"
    assert p["imgHidden"] is False and p["csv"] == "/runs/r1/values.csv"
    assert p["ply"] == "/runs/r1/scan.ply"
    assert o["buttonsAfter"] == {"measure": False, "modes": False}
    assert o["spaceMeasure"] == ["/api/measure"]
    nm = o["notMeasurable"]
    assert nm["lamp"] == {"text": "NOT MEASURABLE", "cls": "qlamp lamp-notmeasurable"}
    assert nm["reason"].startswith("rig_unverified: mobile rig not verified") and "fix: scan rig-check" in nm["reason"]
    assert nm["table"] is True and "none" in nm["verdict"]
    assert o["badInput"]["sent"] == 0 and "region" in o["badInput"]["notice"]
    assert "cart" in o["rigInfo"]["text"] and "NO mm now" in o["rigInfo"]["text"]
    assert "HOME" in o["rigInfo"]["text"] and o["rigInfo"]["warn"] is True
    # ---- Quick check: START posts the one check, the lamp shows its disposition
    assert o["quick"] == {"station": False, "extras": False, "rig": True, "measure": False,
                          "quick": True, "measureBtn": False, "startBtn": True,
                          "on": ["qModeQuick"]}
    assert o["quickLamp"]["cls"] == "qlamp lamp-ready" and o["recipes"] == ["flat_plate", "box_lid"]
    assert o["quickNoThreshold"]["sent"] == [] and "threshold" in o["quickNoThreshold"]["notice"]
    assert o["quickSent"] == ["/api/quick-check"]
    assert o["quickPost"] == {"check": "flatness", "source": "quick", "operator_confirmed": True,
                              "threshold_mm": 0.5, "roi": [10, 20, 300, 400],
                              "guard_band_pct": 10, "preset": "cart",       # the loaded one
                              "zone": "selection@s0", "mock_part": "scene:plate"}
    assert o["quickPass"] == {"lamp": {"text": "PASS", "cls": "qlamp lamp-pass"},
                              "disp": "CONTINUE"}
    assert o["quickPassKept"] == o["quickPass"]["lamp"]
    assert o["quickFail"] == {"text": "FAIL", "cls": "qlamp lamp-fail"}
    assert o["quickPull"]["lamp"]["cls"] == "qlamp lamp-pull" and "guard_band" in o["quickPull"]["reasons"]
    assert o["recipePost"] == {"check": "recipe", "source": "quick", "operator_confirmed": True,
                               "recipe": "box_lid", "mock_part": "scene:plate"}
    assert o["recipeRows"] == {"threshold": False, "recipe": True}
    assert o["mOutside"] == {"prevented": False, "sent": 0}
    assert "select the mode first" in o["refusedJob"]
    # ---- Station again: as before
    assert o["back"] == station and o["backLamp"]["cls"] == "qlamp lamp-ready"
    assert o["spaceStation"] == ["/api/cmd/start"]
    # ---- the real rig
    assert o["realQuick"] == {"start": True, "confirmRow": True}
    assert o["realQuickPost"]["operator_confirmed"] is True and "mock_part" not in o["realQuickPost"]
    assert o["confirmCleared"] is False
    assert o["realMeasure"] == {"measure": False, "confirmRow": False}
    # ---- the station's keys: Station mode only; HOME everywhere
    assert o["stationKeys"] == {k: {"prevented": False, "sent": []} for k in "2ptank"}
    assert o["homeKey"] == ["/api/cmd/home"]
    # ---- the picker shows the loaded preset; the confirmation is for one setup
    assert o["followed"]["picker"] == "raw" and o["followed"]["confirm"] is False
    assert "Loaded: raw" in o["followed"]["info"]
    assert o["picked"] == {"confirm": False, "loaded": "bench", "picker": "bench", "start": True}
    assert o["confirmAfterMode"] is False and o["spaceUnconfirmed"] == []
    assert o["labelInStation"] == "Max deviation" and o["labelInMeasure"] == "Measured"
    # ---- nothing of a mode outlives it; a FAULT looks like one in every mode
    assert o["leftMode"] == {"lamp": {"text": "READY", "cls": "qlamp lamp-ready"}, "img": False,
                             "values": False, "label": "Max deviation", "verdict": "-"}
    assert o["faultLamp"] == {"text": "FAULT", "cls": "qlamp lamp-invalid"}


def test_quick_js_and_css_have_the_mode_pieces():
    js = (STATIC / "quick.js").read_text(encoding="utf-8")
    css = (STATIC / "quick.css").read_text(encoding="utf-8")
    html = (STATIC / "quick.html").read_text(encoding="utf-8")
    for needle in ("'/api/cmd/mode'", "'/api/measure'", "'/api/quick-check'", "'/api/modes'"):
        assert needle in js, needle
    for cls in (".lamp-measured", ".lamp-notmeasurable", ".qmode.on", ".qvalues"):
        assert cls in css, cls
    used = set(re.findall(r"\$\('#([\w-]+)", js)) | set(re.findall(r"'#(q[\w-]+)'", js))
    defined = set(re.findall(r'\bid="([\w-]+)"', html))
    assert not (used - defined), sorted(used - defined)
    assert "<kbd>M</kbd>" in html and "in Station mode only" in html

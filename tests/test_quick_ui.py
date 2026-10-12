"""Operator page (/quick) + the command bus over HTTP, in MOCK mode: the page and its
links, the station configurations, START -> a PASS run, ABORT mid-run -> ABORTED,
HOME -> the rig check (reference, OK, a simulated nudge -> MOVED -> FAULT), and the
gateway rules (TRIGGER only when armed, HOME refused while running). The bus plumbing
tests (every bus job reports back, strict confirmation, ABORT in any state, stable
configuration numbers) drive fake station runs / rig checks instead of captures; the page
logic runs in
node on the real quick.js with a tiny fake DOM."""

import json
import re
import shutil
import subprocess
import threading
import time
from collections import deque

import numpy as np
import pytest

from engine import api, config, rules
from conftest import ROOT
from test_ui_server import Client

from ui.web import server as ui_server

STATIC = ROOT / "ui" / "web" / "static"


@pytest.fixture(scope="module")
def quick(tmp_path_factory):
    from engine.calibrate.stereo import StereoCalibration
    tmp = tmp_path_factory.mktemp("quick")
    lib, stations = tmp / "library", tmp / "stations"
    stations.mkdir()
    doc = json.loads((ROOT / "stations" / "example_metric_check.json").read_text())
    doc.update(runs_dir=str(tmp / "runs"), mock_library=None)
    (stations / "flat_check.json").write_text(json.dumps(doc))
    (stations / "broken.json").write_text("{ not json")
    shutil.copy(ROOT / "stations" / "io_example.json", stations / "io_example.json")
    s = api.configure(lib, mock=True)                 # the bench's TRUE stereo calibration
    try:
        p = api.load_preset(doc["preset"])
        tr = s.bench.stereo_truth()
        d = s.store.new_dir("calibrations", p.geometry_id, label="stereo")
        cal = StereoCalibration(id=f"{p.geometry_id}/{d.name}", geometry_id=p.geometry_id,
                                K_cam=tr["K_cam"], dist_cam=tr["dist_cam"], K_proj=tr["K_proj"],
                                dist_proj=np.zeros(5), R=tr["R"], t=np.ravel(tr["t"]),
                                diamond_parity=tr["parity"],
                                fingerprint=config.geometry_fingerprint(p), path=d,
                                created="2026-01-01T00:00:00")
        s.store.write_json(d / "stereo_calibration.json", cal.to_json())
    finally:
        api._session = None
    httpd, app = ui_server.make_server(mock=True, library=lib, port=0, stations_dir=stations,
                                       preset=doc["preset"])
    t = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
    t.start()
    try:
        yield Client(app.port), app
    finally:
        httpd.shutdown()
        httpd.server_close()
        app.close()
        api._session = None


def _wait_state(c, want, timeout=180.0):
    t0 = time.monotonic()
    while True:
        _, b = c.get("/api/cmd/state")
        if b["state"] in want:
            return b
        assert time.monotonic() - t0 < timeout, b
        time.sleep(0.2)


def _job(c, jid, timeout=180.0):
    t0 = time.monotonic()
    while True:
        _, j = c.get(f"/api/jobs/{jid}")
        if j["status"] in ("done", "error"):
            return j
        assert time.monotonic() - t0 < timeout, j
        time.sleep(0.2)


def _jid(reply, kind):
    return next(k for k, v in reply["jobs"].items() if v == kind)


def test_quick_page_is_served_and_linked(quick):
    c, _ = quick
    status, raw, hdrs = c.request("GET", "/quick")
    html = raw.decode()
    assert status == 200 and hdrs["Content-Type"].startswith("text/html")
    assert '<script src="/static/quick.js"' in html and 'href="/"' in html
    assert hdrs.get("X-Frame-Options") == "DENY"
    for name in ("quick.js", "quick.css"):
        assert c.request("GET", f"/static/{name}")[0] == 200
    assert 'href="/quick"' in (STATIC / "index.html").read_text(encoding="utf-8")
    assert not re.search(r"\son[a-z]+\s*=", html), "inline handler (the CSP forbids them)"
    js = (STATIC / "quick.js").read_text(encoding="utf-8")
    used = set(re.findall(r"\$\('#([\w-]+)", js))
    defined = set(re.findall(r'\bid="([\w-]+)"', html))
    assert used and not (used - defined), sorted(used - defined)
    assert "cdn" not in html.lower() and "https://" not in html
    node = shutil.which("node")
    if node:
        r = subprocess.run([node, "--check", str(STATIC / "quick.js")], capture_output=True,
                           text=True)
        assert r.returncode == 0, r.stderr


def test_station_configurations_are_listed(quick):
    c, _ = quick
    _, d = c.get("/api/stations")
    by = {s["config"]: s for s in d["stations"]}
    assert d["mode"] == "mock" and set(by) == {"flat_check.json", "broken.json"}   # no io_*
    assert by["flat_check.json"]["job_type"] == "metric" and "error" in by["broken.json"]
    _, b = c.get("/api/cmd/state")
    # every config keeps its number (an invalid one is listed and marked unusable, so it
    # never shifts the others); the first usable one is selected at start-up
    assert b["station_configs"] == ["broken.json", "flat_check.json"] and b["state"] == "READY"
    assert b["station_config"] == "flat_check.json" and list(b["unusable"]) == ["broken.json"]
    status, r = c.post("/api/cmd/config", {"config": "broken.json"})
    assert status == 409 and not r["accepted"] and "not usable" in r["message"]
    assert c.post("/api/cmd/config", {"config": 1})[0] == 409
    status, r = c.post("/api/cmd/config", {"config": 2})
    assert status == 200 and r["bus"]["station_config"] == "flat_check.json"
    assert c.post("/api/cmd/teleport", {})[0] == 404


def test_start_runs_the_station_to_pass(quick):
    c, app = quick
    status, r = c.post("/api/cmd/start", {"source": "test"})
    assert status == 200 and r["accepted"] and r["bus"]["state"] == "RUNNING", r
    j = _job(c, _jid(r, "run"))
    assert j["status"] == "done", j
    res = j["result"]
    assert res["result"] == "PASS" and res["job"]["type"] == "metric" and res["exit_code"] == 0
    assert res["deviation_units"] == "mm" and res["deviation_mm"]["max"] < 0.5
    b = _wait_state(c, {"READY"})
    assert b["signals"]["PASS"] and b["signals"]["READY"] and not b["signals"]["BUSY"]
    st, raw, _ = c.request("GET", j["links"]["result_json"])
    assert st == 200 and json.loads(raw)["run_id"] == res["run_id"]
    if "preview" in j["links"]:
        assert c.request("GET", j["links"]["preview"])[0] == 200
    assert c.request("GET", f"/runs/{res['run_id']}/secret.txt")[0] == 404
    assert c.request("GET", "/runs/not_a_run/result.json")[0] == 404


def test_a_crashed_run_reports_the_exit_code_its_log_records(quick, monkeypatch):
    """run_station answered exit_code 2 (refused) for every exception, while run.log.jsonl's
    run_end (and `scan run`) record 4 for a crash."""
    from ui import cli
    c, app = quick

    def boom(*a, **k):
        raise RuntimeError("simulated crash in the measurement")
    monkeypatch.setattr(cli, "_run_metric", boom)
    j = c.job("run_station", {"config": "flat_check.json"})
    assert j["status"] == "done", j
    res = j["result"]
    assert res["result"] == "INVALID" and res["exit_code"] == cli.EXIT_ERROR, res
    st, raw, _ = c.request("GET", j["links"]["log"])
    end = [json.loads(x) for x in raw.decode("utf-8").splitlines() if '"run_end"' in x]
    assert st == 200 and end and end[-1]["exit_code"] == res["exit_code"], end


def test_abort_mid_run_is_recorded_and_wins(quick):
    c, app = quick
    _, r = c.post("/api/cmd/start", {})
    jid = _jid(r, "run")
    t0 = time.monotonic()
    while c.get(f"/api/jobs/{jid}")[1]["status"] != "running":
        assert time.monotonic() - t0 < 60
        time.sleep(0.05)
    time.sleep(1.0)                                  # mid capture
    assert c.post("/api/cmd/home", {})[0] == 409     # HOME refused while RUNNING
    assert c.post("/api/cmd/trigger", {})[0] == 409
    status, a = c.post("/api/cmd/abort", {"source": "plc"})
    assert status == 200 and a["bus"]["state"] == "ABORTING"
    j = _job(c, jid, 120)
    assert j["status"] == "done" and j["result"]["result"] == "ABORTED", j
    assert j["result"]["exit_code"] == 1 and j["result"]["invalid_reason"] is None
    b = _wait_state(c, {"READY"})
    assert not b["signals"]["PASS"] and app.session.cancel is None
    run_dir = app._runs[j["result"]["run_id"]]
    events = [json.loads(x)["event"] for x in (run_dir / "run.log.jsonl").read_text().splitlines()]
    assert "aborted" in events and events[-1] == "run_end"


def test_trigger_needs_arm(quick):
    c, _ = quick
    assert c.post("/api/cmd/trigger", {})[0] == 409
    status, r = c.post("/api/cmd/arm", {})
    assert status == 200 and r["bus"]["state"] == "ARMED"
    status, r = c.post("/api/cmd/config", {"config": "flat_check.json"})   # the same one
    assert status == 200 and r["bus"]["state"] == "ARMED"
    status, r = c.post("/api/cmd/disarm", {})
    assert status == 200 and r["bus"]["state"] == "READY"


def test_home_rig_check_reference_ok_moved_fault(quick):
    c, _ = quick
    status, r = c.post("/api/cmd/home", {})
    assert status == 200 and r["bus"]["state"] == "HOMING"
    j = _job(c, _jid(r, "home"))
    assert j["status"] == "done" and j["result"]["status"] == "REFERENCE_SET", j
    _wait_state(c, {"READY"})
    _, r = c.post("/api/cmd/home", {})
    j = _job(c, _jid(r, "home"))
    assert j["result"]["status"] == "OK", j["result"]
    _wait_state(c, {"READY"})
    _, r = c.post("/api/cmd/home", {"mock_shift_px": [2.5, 0.0]})
    j = _job(c, _jid(r, "home"))
    assert j["result"]["status"] == "MOVED" and j["result"]["shift_px"] > 1.5, j["result"]
    b = _wait_state(c, {"FAULT"})
    assert b["signals"]["ALARM"] and "MOVED" in b["fault"]
    assert c.post("/api/cmd/start", {})[0] == 409
    _, r = c.post("/api/cmd/home", {})
    assert _job(c, _jid(r, "home"))["result"]["status"] == "OK"
    _wait_state(c, {"READY"})


def test_real_rig_run_station_refused_without_confirmation(tmp_path):
    """A REAL-mode App refuses an unconfirmed run before anything touches the rig."""
    import sys
    app = ui_server.App(mock=False, library=tmp_path / "lib").start()
    try:
        assert app.bus.require_confirmation
        r = app.bus.command("START")
        assert not r.accepted and "confirmation" in r.message
        job = app.submit("run_station", {"config": "example_metric_check.json"})
        job.done.wait(60)
        assert job.status == "error" and "confirmation" in job.error
        assert "PySpin" not in sys.modules
    finally:
        app.close()
        api._session = None


# ------------------------------------------------ bus plumbing (fake runs, no capture)

class FakeRig:
    """Stands in for run_station / rig_check on the worker: each call waits until the test
    releases the gate, or until its job is cancelled (ABORT) - unless ignore_cancel holds it
    there (a capture that can't stop yet). No camera, no projector, no engine call."""

    def __init__(self, app):
        self.app = app
        self.gate = threading.Event()
        self.ignore_cancel = False
        self.fail_run = None
        self.runs, self.homes = [], []
        app._v_run_station = self._run              # instance attributes win over the verbs
        app._v_rig_check = self._home

    def _wait(self) -> bool:
        me = self.app._current
        cancel = me.cancel if me is not None else threading.Event()
        t0 = time.monotonic()
        while True:
            if cancel.is_set() and not self.ignore_cancel:
                return True
            if self.gate.is_set():
                return cancel.is_set()
            assert time.monotonic() - t0 < 60, "fake rig job never released"
            time.sleep(0.01)

    def _run(self, a):
        self.runs.append(dict(a))
        if self.fail_run:
            raise ui_server.ArgError(self.fail_run)
        return {"result": "ABORTED" if self._wait() else "PASS"}

    def _home(self, a):
        self.homes.append(dict(a))
        if self._wait():
            raise rules.Cancelled("aborted by the operator")
        return {"status": "OK"}


@pytest.fixture
def station(tmp_path):
    """A fresh MOCK server with three station configurations a / b / c and a FakeRig."""
    stations = tmp_path / "stations"
    stations.mkdir()
    doc = json.loads((ROOT / "stations" / "example_metric_check.json").read_text())
    doc.update(runs_dir=str(tmp_path / "runs"), mock_library=None)
    for name in ("a.json", "b.json", "c.json"):
        (stations / name).write_text(json.dumps(doc))
    httpd, app = ui_server.make_server(mock=True, library=tmp_path / "lib", port=0,
                                       stations_dir=stations)
    t = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
    t.start()
    fake = FakeRig(app)
    try:
        yield Client(app.port), app, stations, fake
    finally:
        fake.ignore_cancel = False
        fake.gate.set()
        httpd.shutdown()
        httpd.server_close()
        app.close()
        api._session = None


def _until(cond, timeout=30.0):
    t0 = time.monotonic()
    while not cond():
        assert time.monotonic() - t0 < timeout, "condition never became true"
        time.sleep(0.01)


def _status(c, jid):
    return c.get(f"/api/jobs/{jid}")[1]["status"]


def _hold_worker(app):
    """Occupy the worker with a short read so new jobs stay queued. -> release event."""
    hold = threading.Event()
    th = threading.Thread(target=app.call, args=(lambda: hold.wait(30), 60), daemon=True)
    th.start()
    _until(lambda: not app._quick)
    return hold


def test_cancelled_queued_bus_job_reports_back(station):
    """Cancelling a queued START / HOME job from the job list used to leave the bus RUNNING
    / HOMING for ever (ABORT then stuck in ABORTING; START, RESET refused): the job never
    ran, so nobody told the bus."""
    c, app, _, fake = station
    fake.gate.set()
    hold = _hold_worker(app)
    try:
        status, r = c.post("/api/cmd/start", {})
        jid = _jid(r, "run")
        assert status == 200 and _status(c, jid) == "queued"
        status, j = c.post(f"/api/jobs/{jid}/cancel", {})
        assert status == 200 and j["status"] == "error" and j["error_type"] == "Cancelled"
        b = c.get("/api/cmd/state")[1]
        assert b["state"] == "READY" and b["last_result"] == "ABORTED" and b["jobs"] == {}
        assert not any(h[2] == "JOB_LOST" for h in app.bus.history)    # reported, not rescued
        # a queued rig check, cancelled: it never ran, so the rig is unchecked -> FAULT
        _, r = c.post("/api/cmd/home", {})
        c.post(f"/api/jobs/{_jid(r, 'home')}/cancel", {})
        b = c.get("/api/cmd/state")[1]
        assert b["state"] == "FAULT" and "cancelled" in b["fault"]
        assert c.post("/api/cmd/reset", {})[0] == 200
    finally:
        hold.set()
    _, r = c.post("/api/cmd/start", {})                 # the station still works
    assert _job(c, _jid(r, "run"))["result"]["result"] == "PASS"
    assert _wait_state(c, {"READY"}, 30)["last_result"] == "PASS"
    assert fake.runs == [{"config": "a.json", "operator_confirmed": False}] and fake.homes == []


def test_bus_jobs_are_registered_before_the_worker_can_see_them(station):
    """The id used to be recorded after submit() returned: a job that failed at once could
    finish first, and its end never reached the bus (same wedge as the cancel bug)."""
    c, app, _, fake = station
    seen = []

    class Spy(deque):
        def append(self, job):
            seen.append((job.verb, job.id in app._bus_jobs))
            super().append(job)

    with app._cv:
        app._queue = Spy(app._queue)
    fake.fail_run = "station config vanished"
    status, r = c.post("/api/cmd/start", {})
    assert status == 200 and r["accepted"]
    b = _wait_state(c, {"READY"}, 30)
    assert b["last_result"] == "INVALID" and b["jobs"] == {}
    fake.fail_run = None
    fake.gate.set()
    _, r = c.post("/api/cmd/home", {})
    _wait_state(c, {"READY"}, 30)
    assert seen == [("run_station", True), ("rig_check", True)]
    assert not any(h[2] == "JOB_LOST" for h in app.bus.history)


def test_reset_and_a_good_home_clear_alarm_over_http(station):
    """Seen on the rig's operator page: an INVALID cycle (PULL + ALARM), ABORT during HOME ->
    FAULT, RESET, HOME OK -> READY with the ALARM output still on. RESET clears ALARM; the
    stale PULL (and its code) stays until the next cycle; PASS never shows."""
    c, app, _, fake = station
    fake.fail_run = "no stereo calibration at this geometry"
    assert c.post("/api/cmd/start", {})[0] == 200
    b = _wait_state(c, {"READY"}, 30)
    assert b["last_result"] == "INVALID" and b["signals"]["ALARM"] and b["signals"]["PULL"]
    fake.fail_run = None
    _, r = c.post("/api/cmd/home", {})                  # the rig check, aborted while running
    jid = _jid(r, "home")
    _until(lambda: _status(c, jid) == "running")
    assert c.post("/api/cmd/abort", {})[0] == 200
    b = _wait_state(c, {"FAULT"}, 30)
    assert b["signals"]["ALARM"] and not b["signals"]["READY"]
    status, r = c.post("/api/cmd/reset", {})
    assert status == 200 and "PULL output stays" in r["message"]
    b = r["bus"]
    assert b["state"] == "READY" and not b["signals"]["ALARM"] and b["signals"]["READY"]
    assert b["signals"]["PULL"] and not b["signals"]["PASS"] and b["alarm_acknowledged"]
    assert b["registers"]["result_code"] == 3           # the stale INVALID code, documented
    fake.gate.set()
    _, r = c.post("/api/cmd/home", {})
    assert _job(c, _jid(r, "home"))["result"]["status"] == "OK"
    b = _wait_state(c, {"READY"}, 30)
    assert not b["signals"]["ALARM"] and b["signals"]["PULL"] and not b["signals"]["PASS"]
    assert c.post("/api/cmd/reset", {})[0] == 409       # nothing left to acknowledge
    _, r = c.post("/api/cmd/start", {})                 # the next cycle: a clean PASS
    assert _job(c, _jid(r, "run"))["result"]["result"] == "PASS"
    b = _wait_state(c, {"READY"}, 30)
    assert b["signals"]["PASS"] and not b["signals"]["ALARM"] and not b["signals"]["PULL"]


def _start_held_abort(c, app, fake, home=None):
    """START, then ABORT while the capture can't stop yet (ABORTING), optionally a HOME
    queued behind it, then lose the run's report (the old wedge). -> the run's job id."""
    _, r = c.post("/api/cmd/start", {})
    jid = _jid(r, "run")
    _until(lambda: _status(c, jid) == "running")
    fake.ignore_cancel = True
    assert c.post("/api/cmd/abort", {})[1]["bus"]["state"] == "ABORTING"
    if home is not None:
        status, r = c.post("/api/cmd/home", home)
        assert status == 200 and "after the abort" in r["message"]
    with app._cv:
        app._bus_jobs.pop(jid)
    return jid


def test_aborting_with_no_job_left_settles_instead_of_wedging(station):
    """Recovery: a busy state with no tracked bus job (a report that got lost) is settled
    by the next state read or command instead of wedging the station; a HOME pressed while
    aborting still runs with its own options - unless a further ABORT cancels it."""
    c, app, _, fake = station
    jid = _start_held_abort(c, app, fake, {"tol_px": 0.7})
    b = c.get("/api/cmd/state")[1]                     # a poll settles it
    assert b["state"] == "HOMING" and b["last_result"] == "ABORTED", b
    assert any(h[2] == "JOB_LOST" for h in app.bus.history)
    fake.ignore_cancel = False
    fake.gate.set()
    assert _wait_state(c, {"READY"}, 30)["fault"] is None
    assert fake.homes == [{"tol_px": 0.7}] and _job(c, jid)["result"]["result"] == "ABORTED"
    # the same wedge, then ABORT again: it settles at once and wins over the queued HOME
    fake.gate.clear()
    jid = _start_held_abort(c, app, fake, {"set_reference": True})
    status, r = c.post("/api/cmd/abort", {})
    assert status == 200 and r["bus"]["state"] == "READY", r
    assert r["bus"]["last_result"] == "ABORTED" and "queued HOME is cancelled" in r["message"]
    fake.ignore_cancel = False
    _job(c, jid)
    assert fake.homes == [{"tol_px": 0.7}] and app.bus.home_args == {}
    # RUNNING whose job vanished: settled as INVALID, and START works again
    _, r = c.post("/api/cmd/start", {})
    jid = _jid(r, "run")
    _until(lambda: _status(c, jid) == "running")
    with app._cv:
        app._bus_jobs.pop(jid)
    fake.gate.set()
    _job(c, jid)
    status, r = c.post("/api/cmd/start", {})
    assert status == 200 and r["accepted"], r
    assert app.bus.history[-2][2:4] == ("JOB_LOST", False)
    assert _wait_state(c, {"READY"}, 30)["last_result"] == "PASS"


def test_operator_confirmed_must_be_a_json_boolean(station):
    """bool("false") is True: a gateway sending the string "false" used to count as the
    real rig's operator confirmation (and an ARM with it confirmed every later TRIGGER)."""
    c, app, _, fake = station
    app.bus.require_confirmation = True                # what a REAL-mode App sets
    for bad in ("false", "true", "0", "no", 1, 0, [], {}):
        for cmd in ("arm", "start", "trigger"):
            status, r = c.post(f"/api/cmd/{cmd}", {"operator_confirmed": bad})
            assert status == 400 and "JSON boolean" in r["error"], (cmd, bad, r)
    assert app.bus.state == "READY" and not app.bus.armed_confirmed and fake.runs == []
    assert c.post("/api/cmd/arm", {"operator_confirmed": False})[0] == 409
    assert c.post("/api/cmd/arm", {"operator_confirmed": None})[0] == 409
    status, r = c.post("/api/cmd/arm", {"operator_confirmed": True})
    assert status == 200 and app.bus.armed_confirmed
    assert c.post("/api/cmd/disarm", {})[0] == 200
    # the verbs behind POST /api/jobs refuse a non-boolean confirmation too
    with pytest.raises(ui_server.ArgError, match="JSON boolean"):
        ui_server.App._v_run_station(app, {"config": "a.json", "operator_confirmed": "false"})
    j = c.job("recipe_run", {"recipe": "box_lid", "operator_confirmed": "yes"})
    assert j["status"] == "error" and "JSON boolean" in j["error"], j
    j = c.job("preset_delete", {"name": "x", "reason": "r", "confirm": "true"})
    assert j["status"] == "error" and "JSON boolean" in j["error"], j


def test_abort_and_disarm_ignore_a_malformed_operator_confirmed(station):
    """Only START / TRIGGER / ARM read operator_confirmed. A gateway that sends booleans as
    strings must still stop the rig: ABORT and DISARM with {"operator_confirmed": "false"}
    used to get a 400 (the run went on, the bus stayed ARMED with its confirmation)."""
    c, app, _, fake = station
    app.bus.require_confirmation = True                # what a REAL-mode App sets
    assert c.post("/api/cmd/arm", {"operator_confirmed": True})[0] == 200
    status, r = c.post("/api/cmd/disarm", {"operator_confirmed": "false"})
    assert status == 200 and r["accepted"], r
    assert app.bus.state == "READY" and not app.bus.armed_confirmed
    status, r = c.post("/api/cmd/start", {"operator_confirmed": True})
    assert status == 200 and app.bus.state == "RUNNING", r
    _until(lambda: len(fake.runs) == 1)
    status, r = c.post("/api/cmd/abort", {"operator_confirmed": "false"})
    assert status == 200 and r["accepted"], r
    _until(lambda: app.bus.state == "READY")
    assert app.bus.last_result == "ABORTED"
    for cmd in ("home", "reset"):                     # never a 400 for the ignored field
        status, r = c.post(f"/api/cmd/{cmd}", {"operator_confirmed": "no"})
        assert status in (200, 409), (cmd, status, r)
    fake.gate.set()


def test_abort_when_idle_stops_direct_runs_the_verdict_and_the_live_view(station):
    """ABORT used to do nothing unless the bus itself was RUNNING / HOMING: a station run a
    flow submitted through POST /api/jobs, a projected verdict and the live view went on."""
    c, app, _, fake = station
    _, run = c.post("/api/jobs", {"verb": "run_station", "args": {"config": "a.json"}})
    _until(lambda: _status(c, run["id"]) == "running")
    _, fb = c.post("/api/jobs", {"verb": "recipe_feedback", "args": {"run_id": "r1"}})
    _, live = c.post("/api/live/start", {})
    assert app.bus.state == "READY" and _status(c, fb["id"]) == "queued"
    status, a = c.post("/api/cmd/abort", {"source": "key"})
    assert status == 200 and a["accepted"] and a["state"] == "READY", a
    assert run["id"] in a["message"] and fb["id"] in a["message"] and "live view" in a["message"]
    assert _job(c, run["id"])["result"]["result"] == "ABORTED"
    for j in (fb, live):
        j = _job(c, j["id"])
        assert j["status"] == "error" and j["error_type"] == "Cancelled", j
    # ARMED: ABORT disarms and still reaches the owner
    assert c.post("/api/cmd/arm", {})[0] == 200
    _, run = c.post("/api/jobs", {"verb": "run_station", "args": {"config": "a.json"}})
    _until(lambda: _status(c, run["id"]) == "running")
    status, a = c.post("/api/cmd/abort", {})
    assert a["state"] == "READY" and "disarmed" in a["message"] and run["id"] in a["message"]
    assert _job(c, run["id"])["result"]["result"] == "ABORTED"


def test_station_list_refresh_never_switches_the_selected_configuration(station):
    """GET /api/stations (any page load) used to replace an invalid selected configuration with
    the first valid one - without loading its preset, while ARMED with the operator's
    confirmation - and renumbered the rest, so a PLC's 'configuration 2' became another file."""
    c, app, stations, fake = station
    app.bus.require_confirmation = True
    _, b = c.get("/api/cmd/state")
    assert b["station_configs"] == ["a.json", "b.json", "c.json"] and b["station_config"] == "a.json"
    assert c.post("/api/cmd/config", {"config": 2})[1]["bus"]["station_config"] == "b.json"
    assert c.post("/api/cmd/arm", {"operator_confirmed": True})[0] == 200
    good = (stations / "b.json").read_text()
    (stations / "b.json").write_text("{ broken")      # mid-save in an editor, say
    assert c.get("/api/stations")[0] == 200
    b = c.get("/api/cmd/state")[1]
    assert b["state"] == "READY" and b["station_config"] == "b.json" and "b.json" in b["unusable"]
    assert b["station_configs"] == ["a.json", "b.json", "c.json"] and b["registers"]["station_config"] == 2
    assert not app.bus.armed_confirmed
    assert c.post("/api/cmd/trigger", {})[0] == 409
    status, r = c.post("/api/cmd/start", {"operator_confirmed": True})
    assert status == 409 and "not usable" in r["message"], r
    assert r["bus"]["last_result"] == "INVALID" and r["bus"]["signals"]["ALARM"]
    assert c.post("/api/cmd/config", {"config": 2})[0] == 409
    assert c.post("/api/cmd/config", {"config": 3})[1]["bus"]["station_config"] == "c.json"
    (stations / "b.json").write_text(good)
    c.get("/api/stations")
    assert c.post("/api/cmd/config", {"config": 2})[1]["bus"]["station_config"] == "b.json"
    assert fake.runs == []
    # while a cycle runs the list and its numbers stay as they are
    _, r = c.post("/api/cmd/start", {"operator_confirmed": True})
    _until(lambda: _status(c, _jid(r, "run")) == "running")
    (stations / "a.json").unlink()
    (stations / "d.json").write_text(good)
    c.get("/api/stations")
    b = c.get("/api/cmd/state")[1]
    assert b["state"] == "RUNNING" and b["station_config"] == "b.json"
    assert b["station_configs"] == ["a.json", "b.json", "c.json"] and b["registers"]["station_config"] == 2
    fake.gate.set()
    _wait_state(c, {"READY"}, 30)
    c.get("/api/stations")                             # idle again: the new list
    b = c.get("/api/cmd/state")[1]
    assert b["station_configs"] == ["b.json", "c.json", "d.json"] and b["station_config"] == "b.json"
    assert fake.runs == [{"config": "b.json", "operator_confirmed": True}]


def test_changing_the_station_configuration_while_armed_disarms(station):
    """The real rig's ARM confirmation used to carry over a configuration change: TRIGGER then
    ran the new configuration with nobody confirming its setup."""
    c, app, _, fake = station
    app.bus.require_confirmation = True
    assert c.post("/api/cmd/arm", {"operator_confirmed": True})[0] == 200
    status, r = c.post("/api/cmd/config", {"config": "b.json", "source": "plc"})
    assert status == 200 and "disarmed" in r["message"]
    assert r["bus"]["state"] == "READY" and not app.bus.armed_confirmed
    assert c.post("/api/cmd/trigger", {"source": "plc"})[0] == 409
    assert c.post("/api/cmd/start", {})[0] == 409            # unconfirmed
    # START / TRIGGER / ARM naming the configuration the page shows: refused on a mismatch
    status, r = c.post("/api/cmd/start", {"operator_confirmed": True, "config": "a.json"})
    assert status == 409 and "b.json" in r["message"] and fake.runs == []
    fake.gate.set()
    status, r = c.post("/api/cmd/start", {"operator_confirmed": True, "config": "b.json"})
    assert status == 200
    _wait_state(c, {"READY"}, 30)
    assert fake.runs == [{"config": "b.json", "operator_confirmed": True}]


def test_home_options_travel_only_with_an_accepted_home(station):
    """A HOME pressed while aborting a rig check was lost, and its stale flag later fired an
    unrequested rig check with stale options (e.g. set_reference) after an unrelated abort."""
    c, app, _, fake = station
    _, r = c.post("/api/cmd/start", {})
    _until(lambda: _status(c, _jid(r, "run")) == "running")
    assert c.post("/api/cmd/home", {"set_reference": True})[0] == 409     # refused: forgotten
    fake.ignore_cancel = True
    assert c.post("/api/cmd/abort", {})[1]["bus"]["state"] == "ABORTING"
    assert c.post("/api/cmd/home", {"tol_px": 0.7})[0] == 200
    fake.ignore_cancel = False
    _wait_state(c, {"HOMING"}, 30)
    fake.gate.set()
    _wait_state(c, {"READY"}, 30)
    assert fake.homes == [{"tol_px": 0.7}]
    # HOME pressed while a rig check is being aborted: it runs when that check ends
    fake.gate.clear()
    _, r = c.post("/api/cmd/home", {})
    _until(lambda: _status(c, _jid(r, "home")) == "running")
    fake.ignore_cancel = True
    assert c.post("/api/cmd/abort", {})[1]["bus"]["state"] == "ABORTING"
    assert c.post("/api/cmd/home", {"mock_shift_px": [1.0, 0.0]})[0] == 200
    fake.ignore_cancel = False
    _until(lambda: len(fake.homes) == 3)
    assert fake.homes[-1] == {"mock_shift_px": [1.0, 0.0]} and app.bus.state == "HOMING"
    fake.gate.set()
    assert _wait_state(c, {"READY"}, 30)["fault"] is None
    # nothing stale left: a later aborted run does not start a rig check nobody asked for
    fake.gate.clear()
    _, r = c.post("/api/cmd/start", {})
    _until(lambda: _status(c, _jid(r, "run")) == "running")
    c.post("/api/cmd/abort", {})
    _wait_state(c, {"READY"}, 30)
    assert len(fake.homes) == 3 and app.bus.home_args == {} and not app.bus.home_after_abort


def test_station_without_a_preset_runs_the_loaded_preset(tmp_path):
    """A config naming no preset used to run the library's selected preset or, with none
    selected, the CLI's default one - not the loaded preset /quick shows (and the run then
    replaced the server's loaded preset)."""
    stations = tmp_path / "stations"
    stations.mkdir()
    doc = {"station_id": "nopreset", "preset": None, "library": None, "mock_library": None,
           "runs_dir": str(tmp_path / "runs"), "require_mm": False, "threshold_mm": None,
           "job": {"type": "inspect", "base": "capture"}, "part": {"role": "test", "label": "p"}}
    (stations / "nopreset.json").write_text(json.dumps(doc))
    loaded = "cell_wall_2p7ft"
    assert loaded != ui_server.DEFAULT_PRESET
    app = ui_server.App(mock=True, library=tmp_path / "lib", stations_dir=stations,
                        preset_name=loaded).start()
    try:
        assert app.call(api.preset_active) is None                 # nothing selected
        st = app.cmd_state()
        assert st["active_preset"] == loaded and st["run_preset"] == loaded
        job = app.submit("run_station", {"config": "nopreset.json"})
        assert job.done.wait(300) and job.status == "done", job.error
        assert job.result["capture"]["preset"] == loaded
        assert app.session.preset.name == loaded
        # nothing loaded: refused, never the CLI default
        p, app.session.preset = app.session.preset, None
        try:
            with pytest.raises(ui_server.ArgError, match="names no preset and none is loaded"):
                ui_server.App._v_run_station(app, {"config": "nopreset.json"})
        finally:
            app.session.preset = p
    finally:
        app.close()
        api._session = None


# ------------------------------------------------ quick.js logic (node, fake DOM)

_QUICK_HARNESS = r"""
const fs = require('fs'), vm = require('vm');
class El {
  constructor(tag, id) { this.tagName = String(tag).toUpperCase(); this.id = id; this.children = [];
    this.attrs = {}; this.dataset = {}; this.className = ''; this._text = ''; this._value = '';
    this.checked = false; this.disabled = false; this.hidden = false; this._sel = undefined;
    this.parent = null; this.isContentEditable = false; this.href = '';
    this.classList = { toggle() {}, add() {}, remove() {} }; }
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
  addEventListener() {}
  matches(sel) {
    return sel.split(',').map((s) => s.trim()).some((s) => {
      if (s === 'a[href]') return this.tagName === 'A' && 'href' in this.attrs;
      if (s === '[role="button"]') return this.attrs.role === 'button';
      if (s === '[tabindex]') return 'tabindex' in this.attrs;
      return this.tagName === s.toUpperCase();
    });
  }
  closest(sel) { for (let e = this; e; e = e.parent) if (e.matches(sel)) return e; return null; }
}
const TAGS = { qStation: 'select', qConfirm: 'input', qStart: 'button', qAbort: 'button',
  qHome: 'button', qArm: 'button', qTrigger: 'button', qGrid: 'button', qZones: 'button',
  qLive: 'button', qSignal: 'button', qResultLink: 'a', qFeedbackLink: 'a', qPreview: 'img' };
const byId = {};
const el = (id) => byId[id] || (byId[id] = new El(TAGS[id] || 'div', id));
const sent = [];
const stations = (names) => names.map((n) => ({ config: n, station_id: n.replace('.json', ''),
  part_label: 'part ' + n, job_type: 'metric', preset: null }));
const srv = { stations: stations(['a.json', 'b.json']), refuse: false,
  bus: { state: 'READY', station_config: 'a.json', station_configs: ['a.json', 'b.json'], unusable: {},
         signals: {}, jobs: {}, active_preset: 'loaded_p', run_preset: 'loaded_p' } };
const sandbox = {
  console, setTimeout, clearTimeout, setInterval, clearInterval,
  document: { addEventListener() {}, querySelector: (s) => el(s.slice(1)), querySelectorAll: () => [],
              createElement: (t) => new El(t), body: new El('body') },
  fetch: async (path, opt) => {
    const body = opt && opt.body ? JSON.parse(opt.body) : undefined;
    sent.push({ path, body });
    let data = {}, status = 200;
    if (path === '/api/cmd/state') data = { ...srv.bus };
    else if (path === '/api/stations') data = { mode: 'real', stations: srv.stations };
    else if (path.startsWith('/api/cmd/')) {
      const ok = !srv.refuse;
      status = ok ? 200 : 409;
      data = { accepted: ok, command: path.split('/').pop().toUpperCase(),
               message: ok ? 'ok' : 'refused', bus: { ...srv.bus }, jobs: {} };
    } else if (path.startsWith('/api/jobs/')) data = { status: 'queued' };
    return { ok: status < 400, status, json: async () => data };
  },
};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(process.argv[2], 'utf8'), sandbox, { filename: 'quick.js' });
const run = (code) => vm.runInContext(code, sandbox);
const out = {};
const sentTo = (p) => sent.filter((s) => s.path === p);
const key = (k, target) => {
  let prevented = false;
  const n = sent.length;
  run('onKey')({ key: k, target, ctrlKey: false, altKey: false, metaKey: false,
                 preventDefault() { prevented = true; } });
  return { prevented, sent: sent.slice(n).map((s) => s.path) };
};

(async () => {
  await run('loadStations()');
  out.first = { shown: el('qStation').value, preset: el('qPreset').textContent,
                start: el('qStart').disabled };
  el('qConfirm').checked = true;
  run('refreshButtons()');
  out.confirmedStart = el('qStart').disabled;
  // a PLC selects b.json: the page follows it and the confirmation (for a.json) is cleared
  srv.bus = { ...srv.bus, station_config: 'b.json', run_preset: 'b_preset' };
  await run('pollOnce()');
  out.afterPlc = { shown: el('qStation').value, confirm: el('qConfirm').checked,
                   start: el('qStart').disabled, info: el('qStationInfo').textContent,
                   preset: el('qPreset').textContent, loaded: el('qPresetLoaded').textContent };
  // START sends the station configuration it shows
  el('qConfirm').checked = true;
  run('refreshButtons()');
  await run("send('start')");
  out.startBody = sentTo('/api/cmd/start').pop().body;
  // a refused select puts the bus's configuration back in the selector
  srv.refuse = true;
  el('qStation').value = 'a.json';
  await run("selectConfig('a.json')");
  out.refused = { shown: el('qStation').value, reason: el('qReason').textContent };
  srv.refuse = false;
  // the bus marks the shown configuration unusable: START off, the reason shown
  srv.bus = { ...srv.bus, unusable: { 'b.json': 'config invalid' } };
  await run('pollOnce()');
  out.unusable = { start: el('qStart').disabled, arm: el('qArm').disabled,
                   info: el('qStationInfo').textContent };
  srv.bus = { ...srv.bus, unusable: {} };
  // a config added while idle: the list is re-read
  srv.stations = stations(['a.json', 'b.json', 'c.json']);
  srv.bus = { ...srv.bus, station_configs: ['a.json', 'b.json', 'c.json'] };
  await run('pollOnce()');
  out.options = el('qStation').options.map((o) => o.value);
  out.shownAfterReload = el('qStation').value;
  // keys: Space / Enter on a focused control press that control, never START
  el('qConfirm').checked = true;
  run('refreshButtons()');
  out.startEnabled = !el('qStart').disabled;
  const summary = new El('summary');
  const inner = summary.appendChild(new El('b'));
  const link = new El('a');
  link.attrs.href = '/runs/x/result.json';
  out.keys = { abortSpace: key(' ', el('qAbort')), homeEnter: key('Enter', el('qHome')),
               summaryEnter: key('Enter', summary), innerSpace: key(' ', inner),
               linkEnter: key('Enter', link), escOnButton: key('Escape', el('qAbort')),
               bodySpace: key(' ', sandbox.document.body) };
  await new Promise((r) => setTimeout(r, 50));
  process.stdout.write(JSON.stringify(out));
})().catch((e) => { console.error(e); process.exit(1); });
"""


def test_quick_js_follows_the_bus_configuration_and_leaves_focused_buttons_alone(tmp_path):
    """Regressions on the real quick.js (node, fake DOM):
    - the station-configuration selector follows the bus (a PLC / another client changed it),
      clears the operator's confirmation on a change, and START names the configuration it
      shows; a refused select shows the bus's configuration again; the preset shown is the one the
      run uses (run_preset);
    - Space / Enter on a focused button, summary or link press that control: they used to
      be swallowed (a keyboard ABORT during a run) or start a cycle."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed; quick.js logic not checked")
    h = tmp_path / "quick_harness.js"
    h.write_text(_QUICK_HARNESS, encoding="utf-8")
    r = subprocess.run([node, str(h), str(STATIC / "quick.js")], capture_output=True, text=True,
                       timeout=60)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    assert out["first"] == {"shown": "a.json", "preset": "loaded_p", "start": True}
    assert out["confirmedStart"] is False
    a = out["afterPlc"]
    assert a["shown"] == "b.json" and a["confirm"] is False and a["start"] is True, a
    assert "part b.json" in a["info"] and a["preset"] == "b_preset" and "loaded_p" in a["loaded"]
    assert out["startBody"] == {"source": "quick", "operator_confirmed": True,
                                "config": "b.json"}
    assert out["refused"]["shown"] == "b.json" and "refused" in out["refused"]["reason"]
    u = out["unusable"]
    assert u["start"] is True and u["arm"] is True and "config invalid" in u["info"], u
    assert out["options"] == ["a.json", "b.json", "c.json"] and out["shownAfterReload"] == "b.json"
    assert out["startEnabled"] is True
    k = out["keys"]
    for name in ("abortSpace", "homeEnter", "summaryEnter", "innerSpace", "linkEnter"):
        assert k[name] == {"prevented": False, "sent": []}, (name, k[name])
    assert k["escOnButton"] == {"prevented": True, "sent": ["/api/cmd/abort"]}
    assert k["bodySpace"] == {"prevented": True, "sent": ["/api/cmd/start"]}

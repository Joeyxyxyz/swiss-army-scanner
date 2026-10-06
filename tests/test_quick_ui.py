"""Operator page (/quick) + the command bus over HTTP, in MOCK mode: the page and its
links, the work states (station configs), START -> a PASS run, ABORT mid-run -> ABORTED,
HOME -> the rig check (reference, OK, a simulated nudge -> MOVED -> FAULT), and the
gateway rules (TRIGGER only when armed, HOME refused while running)."""

import json
import re
import shutil
import subprocess
import threading
import time

import numpy as np
import pytest

from engine import api, config
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


def test_work_states_are_the_station_configs(quick):
    c, _ = quick
    _, d = c.get("/api/stations")
    by = {s["config"]: s for s in d["stations"]}
    assert d["mode"] == "mock" and set(by) == {"flat_check.json", "broken.json"}   # no io_*
    assert by["flat_check.json"]["job_type"] == "metric" and "error" in by["broken.json"]
    _, b = c.get("/api/cmd/state")
    assert b["workstates"] == ["flat_check.json"] and b["state"] == "READY"
    status, r = c.post("/api/cmd/workstate", {"workstate": "broken.json"})
    assert status == 409 and not r["accepted"]
    status, r = c.post("/api/cmd/workstate", {"workstate": 1})
    assert status == 200 and r["bus"]["workstate"] == "flat_check.json"
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
    assert c.post("/api/cmd/workstate", {"workstate": 1})[0] == 200
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

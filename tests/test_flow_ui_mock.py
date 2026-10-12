"""The operator page (/quick) and the command bus over HTTP for a station configuration with
profiles and work states, MOCK only: the page's top (station configuration + profile), the
step flow (sensor says / DLP says / permission), the prompt banner + ACK, the sensor claims
(POST /api/io/inputs), the permissions, the cycle folder, the deprecated `workstate` route,
the simulated line (mock cases), an old single-job config next to it - and one cycle through
the full mock capture path. The bus plumbing tests answer each DLP check from a script
instead of capturing; the page logic runs in node on the real quick.js with a tiny fake DOM."""

import json
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

import pytest

from engine import api
from engine import disposition as D
from engine.recipes import runner

import recipe_synth as RS
from conftest import ROOT
from test_ui_server import Client

from ui.web import server as ui_server

STATIC = ROOT / "ui" / "web" / "static"
EXAMPLE = ROOT / "stations" / "example_tube_line.json"
FLOW = ROOT / "flows" / "example_tube_line.json"
PRESET = "bench_9in_metric_bright"
PERMITS = ("LOAD_DONE", "CLAMP", "TRANSFER")


def answer(result="PASS", disposition="CONTINUE", **observed) -> dict:
    return {"result": result, "disposition": disposition, "observed": dict(observed),
            "feedback_rendered": True, "message": "" if result == "PASS" else "seat/check: failed",
            "pull_reasons": [] if disposition != "PULL" else [D.reason("invalid", "no points")]}


OK = answer(part_present=True)
ABSENT = answer("FAIL", "REJECT", part_present=False)
GOOD = answer(part_present=True)                        # the tube is there, every check passes
FAILS = answer("FAIL", "REJECT", part_present=True)     # the tube is there, a check fails


class FakeDLP:
    """Stands in for the DLP check of a work state (the flow_verify job) and for the fault
    projection on the worker: each check is answered from `answers` (step id -> an answer,
    or a list used in turn), optionally held until the test releases the gate or the job is
    cancelled. No camera, no projector, no engine call."""

    def __init__(self, app):
        self.app = app
        self.answers = {}
        self.calls, self.projected, self.runs = [], [], []
        self.gate = threading.Event()
        self.gate.set()
        app._v_flow_verify = self._verify           # instance attributes win over the verbs
        app._v_recipe_feedback = self._feedback
        app._v_run_station = self._run

    def _held(self) -> bool:
        me = self.app._current
        t0 = time.monotonic()
        while not self.gate.is_set():
            if me is not None and me.cancel.is_set():
                return True
            assert time.monotonic() - t0 < 60, "fake DLP check never released"
            time.sleep(0.01)
        return me is not None and me.cancel.is_set()

    def _verify(self, a):
        self.calls.append(dict(a))
        if self._held():
            return {"result": "ABORTED", "disposition": "PULL", "run_id": None, "dlp": None}
        ans = self.answers.get(a["step"], GOOD)
        if isinstance(ans, list):
            ans = ans.pop(0) if len(ans) > 1 else ans[0]
        if isinstance(ans, Exception):
            raise ans
        rid = f"fake_{len(self.calls)}"
        self.app._runs[rid] = Path(self.app.root)
        return {"result": ans["result"], "disposition": ans["disposition"], "run_id": rid,
                "zones": [], "dlp": {**ans, "run_id": rid, "run_folder": rid, "step": a["step"],
                                     "attempt": a.get("attempt"), "cycle_id": a.get("cycle_id")}}

    def _feedback(self, a):
        self.projected.append(a["run_id"])
        return {"cancelled": False, "run_id": a["run_id"]}

    def _run(self, a):
        self.runs.append(dict(a))
        return {"result": "PASS", "disposition": "CONTINUE"}


def _stations(tmp_path, **step_over) -> Path:
    """A stations folder with the example tube line (+ its flow) and one single-job config."""
    stations, flows = tmp_path / "stations", tmp_path / "flows"
    stations.mkdir()
    flows.mkdir()
    doc = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    doc["parameters"].update(runs_dir=str(tmp_path / "runs"), mock_library=None)
    (stations / "example_tube_line.json").write_text(json.dumps(doc), encoding="utf-8")
    flow = json.loads(FLOW.read_text(encoding="utf-8"))
    for sid, over in step_over.items():
        next(s for s in flow["steps"] if s["id"] == sid).update(over)
    (flows / FLOW.name).write_text(json.dumps(flow), encoding="utf-8")
    old = json.loads((ROOT / "stations" / "example_metric_check.json").read_text())
    old.update(runs_dir=str(tmp_path / "runs"), mock_library=None)
    (stations / "old_single_job.json").write_text(json.dumps(old), encoding="utf-8")
    shutil.copy(ROOT / "stations" / "io_example.json", stations / "io_example.json")
    return stations


def _serve(tmp_path, stations):
    httpd, app = ui_server.make_server(mock=True, library=tmp_path / "lib", port=0,
                                       stations_dir=stations)
    threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.1},
                     daemon=True).start()
    return httpd, app


@pytest.fixture
def line(tmp_path):
    """A fresh MOCK server on the example tube line with a FakeDLP."""
    httpd, app = _serve(tmp_path, _stations(tmp_path))
    fake = FakeDLP(app)
    try:
        yield Client(app.port), app, fake
    finally:
        fake.gate.set()
        httpd.shutdown()
        httpd.server_close()
        app.close()
        api._session = None


def _state(c, want=None, timeout=30.0) -> dict:
    """The bus snapshot, once it is in one of `want` states (and no DLP check is running)."""
    t0 = time.monotonic()
    while True:
        b = c.get("/api/cmd/state")[1]
        if want is None or b["state"] in want:
            return b
        assert time.monotonic() - t0 < timeout, (b["state"], b.get("cycle"))
        time.sleep(0.02)


def _step(c, expect=200) -> dict:
    status, r = c.post("/api/cmd/step", {})
    assert status == expect, r
    return _state(c, {"WAITING", "READY", "FAULT", "ARMED"}) if status == 200 else r


def _rows(b) -> list:
    return [(r["id"], r["outcome"], r["permission_granted"]) for r in b["cycle"]["steps"]]


# ================================================================== the page

def test_quick_page_has_the_configuration_the_profile_and_the_step_flow():
    html = (STATIC / "quick.html").read_text(encoding="utf-8")
    js = (STATIC / "quick.js").read_text(encoding="utf-8")
    css = (STATIC / "quick.css").read_text(encoding="utf-8")
    for el in ("qStation", "qProfile", "qFlow", "qPrompt", "qPromptMsg", "qAck", "qStep",
               "qInputs", "qPermits", "qMock", "qMockCase", "qLamp", "qLights", "qCamera",
               "qStart", "qAbort", "qHome", "qArm", "qTrigger", "qNotice"):
        assert f'id="{el}"' in html, el
    assert "Station configuration" in html and "Profile" in html and "Work states" in html
    # top = configuration + profile, middle = the flow, then the Lights / Camera / Action panels
    order = [html.index(f'id="{el}"') for el in ("qStation", "qProfile", "qFlow", "qLights",
                                                 "qCamera", "qStart")]
    assert order == sorted(order)
    assert html.index('id="qPrompt"') < html.index('id="qStation"')     # the banner on top
    for sig in ("KICKOUT", "PROMPT", "SENSOR_MISMATCH", "PASS", "PULL"):
        assert f'data-sig="{sig}"' in html, sig
    # the old wording is gone: a station config is not a work state
    assert "Work state (station config)" not in html and "work-state list" not in html
    assert not re.search(r"\son[a-z]+\s*=", html), "inline handler (the CSP forbids them)"
    assert "cdn" not in html.lower() and "https://" not in html and "innerHTML" not in js
    used = set(re.findall(r"\$\('#([\w-]+)", js))
    defined = set(re.findall(r'\bid="([\w-]+)"', html))
    assert used and not (used - defined), sorted(used - defined)
    for needle in ("sensor says", "DLP says", "permission", "/api/cmd/config", "/api/cmd/profile",
                   "/api/io/inputs", "/api/mock/case", "send('step')", "send('ack')",
                   "selectConfig", "selectProfile"):
        assert needle in js, needle
    assert "selectWorkstate" not in js and "/api/cmd/workstate" not in js
    for key in ("<kbd>N</kbd> STEP", "<kbd>K</kbd> ACK", "<kbd>P</kbd> next profile",
                "<kbd>1</kbd>-<kbd>9</kbd> station configuration"):
        assert key in html, key
    for cls in (".qflow", ".qws.current", ".qchip", ".qprompt", ".lamp-wait", ".qnotice"):
        assert cls in css, cls
    # the prompt banner (with its ACK button) stays in view wherever the page is scrolled
    banner = css[css.index(".qprompt {"):css.index("}", css.index(".qprompt {"))]
    assert "position: sticky" in banner and "top: 0" in banner
    node = shutil.which("node")
    if node:
        r = subprocess.run([node, "--check", str(STATIC / "quick.js")], capture_output=True,
                           text=True)
        assert r.returncode == 0, r.stderr


def test_stations_list_configurations_with_profiles_and_work_states(line):
    c, app, fake = line
    status, raw, hdrs = c.request("GET", "/quick")
    assert status == 200 and b'id="qProfile"' in raw and hdrs.get("X-Frame-Options") == "DENY"
    _, d = c.get("/api/stations")
    by = {s["config"]: s for s in d["stations"]}
    assert list(by) == ["example_tube_line.json", "old_single_job.json"]      # no io_*
    t = by["example_tube_line.json"]
    assert t["kind"] == "flow" and t["job_type"] == "flow" and "error" not in t
    assert [p["name"] for p in t["profiles"]] == ["long_tube", "short_tube"]
    assert t["profiles"][0]["recipes"]["presence"] == "tube_long_presence"
    assert [(s["id"], s["title"], s["permit"]) for s in t["flow"]] == [
        ("loaded", "Tube loaded?", "LOAD_DONE"), ("aligned", "Tube aligned?", "CLAMP"),
        ("end_face", "End face OK?", "TRANSFER")]
    assert t["flow"][0]["expects"] == {"part_present": True} and t["signals"] == ["part_present"]
    assert t["permissions"] == list(PERMITS) and t["preset"] == PRESET
    assert {m["name"] for m in t["mock_cases"]} >= {"all_good", "dented", "misaligned"}
    o = by["old_single_job.json"]
    assert o["kind"] == "single job" and o["job_type"] == "metric"
    assert [p["name"] for p in o["profiles"]] == ["default"] and [s["id"] for s in o["flow"]] == ["run"]
    b = _state(c)
    assert b["station_config"] == "example_tube_line.json" and b["profile"] == "long_tube"
    assert b["station_configs"] == ["example_tube_line.json", "old_single_job.json"]
    assert [p["name"] for p in b["profiles"]] == ["long_tube", "short_tube"] and b["legacy"] is False
    assert [s["id"] for s in b["flow"]] == ["loaded", "aligned", "end_face"]
    assert b["permissions"] == dict.fromkeys(PERMITS, False) and b["cycle"] is None
    # MOCK: the configuration starts with its simulated line's default case (scene + sensors)
    assert b["inputs"] == {"part_present": True} and b["mock"]["case"] == "all_good"
    assert b["signal_names"] == ["part_present"] and b["prompt"] is None
    assert b["run_preset"] == PRESET and b["mode"] == "mock"
    assert b["registers"] == {"result_code": 0, "state_code": 0, "station_config": 1,
                              "profile": 1, "work_state": 0, "work_state_code": 0}
    assert "workstate" not in b and "workstates" not in b
    assert len(b["mock"]["cases"]) == 5
    # a broken flow file makes the configuration unusable (listed, numbered, refused)
    flows = app.stations_dir.parent / "flows" / FLOW.name
    good = flows.read_text(encoding="utf-8")
    bad = json.loads(good)
    bad["steps"][2]["verify"] = {"recipe": "polish"}
    flows.write_text(json.dumps(bad), encoding="utf-8")
    _, d = c.get("/api/stations")
    assert "verify.recipe 'polish'" in d["stations"][0]["error"]
    status, r = c.post("/api/cmd/start", {})
    assert status == 409 and "not usable" in r["message"] and r["bus"]["signals"]["PULL"]
    flows.write_text(good, encoding="utf-8")
    c.get("/api/stations")
    assert c.post("/api/cmd/reset", {})[0] == 200           # the refused START's ALARM


# ================================================================== a cycle over HTTP

def test_sensor_claims_over_http_are_strict_booleans(line):
    c, app, fake = line
    status, r = c.post("/api/io/inputs", {"part_present": True})
    assert status == 200 and r["inputs"] == {"part_present": True}
    assert r["bus"]["inputs"] == {"part_present": True}
    status, r = c.post("/api/io/inputs", {"inputs": {"part_present": False}, "source": "plc"})
    assert status == 200 and r["inputs"] == {"part_present": False}
    assert any(h[1] == "plc" and h[2] == "INPUTS" for h in app.bus.history)
    for bad in ({"part_present": "true"}, {"part_present": 1}, {"part_present": "false"},
                {"part present": True}, {"inputs": {"part_present": [True]}},
                # a signal no station configuration declares (a typo is never a silent sensor)
                {"part_presnt": True}, {"inputs": {"door_closed": False}, "source": "plc"},
                {"part_present": True, "door_closed": True},
                # both forms at once: the flat key would be silently ignored
                {"inputs": {"part_present": True}, "part_present": False},
                {"inputs": [True]}, {"inputs": None, "part_present": True}):
        status, r = c.post("/api/io/inputs", bad)
        assert status == 400 and "error" in r, (bad, r)
        assert app.bus.inputs == {"part_present": False}, bad       # nothing of it applied
    status, r = c.post("/api/io/inputs", {"part_present": None})                # forgotten
    assert status == 200 and r["inputs"] == {}
    status, r = c.post("/api/io/inputs", {"part_present": True})
    status, d = c.get("/api/io/inputs")
    assert status == 200 and d["inputs"] == {"part_present": True}
    assert d["signals"] == ["part_present"] and d["permissions"] == dict.fromkeys(PERMITS, False)
    assert d["outputs"]["READY"] is True and "PERMIT_CLAMP" in d["outputs"]
    assert d["registers"]["station_config"] == 1
    # a cross-origin page can't write the line's sensors either
    status, raw, _ = c.request("POST", "/api/io/inputs", {"part_present": False},
                               {"Origin": "http://evil.example"})
    assert status == 403 and app.bus.inputs == {"part_present": True}


def test_a_cycle_over_http_grants_each_permission_and_writes_its_folder(line):
    c, app, fake = line
    fake.answers = {"loaded": OK}
    c.post("/api/io/inputs", {"part_present": True})
    assert c.post("/api/cmd/step", {})[0] == 409            # no cycle yet
    status, r = c.post("/api/cmd/start", {"source": "plc", "config": 1, "profile": "long_tube"})
    assert status == 200 and r["bus"]["state"] == "WAITING" and "waits for STEP" in r["message"]
    cid = r["bus"]["cycle"]["cycle_id"]
    assert r["bus"]["cycle"]["active"] and r["bus"]["cycle"]["waiting"] == "step"
    assert r["bus"]["registers"]["state_code"] == 6 and r["bus"]["registers"]["work_state"] == 1
    for cmd, body in (("start", {}), ("home", {}), ("arm", {}), ("profile", {"profile": 2}),
                      ("config", {"config": 2}), ("ack", {}), ("skip", {}),
                      ("step", {"step": "aligned"})):
        status, r = c.post(f"/api/cmd/{cmd}", body)
        assert status == 409 and not r["accepted"], (cmd, r)
    for verb in ("load_preset", "preset_select"):           # no preset change in a cycle
        status, r = c.post("/api/jobs", {"verb": verb, "args": {"name": PRESET}})
        assert status == 409 and "WAITING" in r["error"], (verb, r)
    b = _step(c)
    assert b["state"] == "WAITING" and _rows(b)[0] == ("loaded", "CONFIRMED", True)
    assert b["permissions"] == {"LOAD_DONE": True, "CLAMP": False, "TRANSFER": False}
    assert b["signals"]["PERMIT_LOAD_DONE"] and not b["signals"]["PASS"]
    assert b["cycle"]["steps"][0]["sensor_inputs"] == {"part_present": True}
    assert b["registers"]["work_state"] == 2
    assert fake.calls[0]["step"] == "loaded" and fake.calls[0]["profile"] == "long_tube"
    assert fake.calls[0]["cycle_id"] == cid and fake.calls[0]["operator_confirmed"] is False
    b = _step(c)
    assert b["permissions"]["CLAMP"] and b["state"] == "WAITING"
    b = _step(c)
    assert b["state"] == "READY" and b["last_result"] == "PASS" and b["last_disposition"] == "CONTINUE"
    assert b["signals"]["PASS"] and all(b["signals"][f"PERMIT_{p}"] for p in PERMITS)
    assert not b["cycle"]["active"] and b["registers"]["result_code"] == 1
    assert b["last_cycle"]["cycle_id"] == cid and b["last_cycle"]["disposition"] == "CONTINUE"
    assert [x["step"] for x in fake.calls] == ["loaded", "aligned", "end_face"]
    # the cycle folder: result.json (served) + log; schema 0.4
    st, raw, _ = c.request("GET", b["last_cycle"]["links"]["result_json"])
    doc = json.loads(raw)
    assert st == 200 and doc["run_id"] == cid and doc["schema_version"] == "0.4"
    assert doc["station_config"] == "example_tube_line.json" and doc["profile"] == "long_tube"
    assert doc["disposition"] == "CONTINUE" and doc["permissions"] == dict.fromkeys(PERMITS, True)
    assert [(s["id"], s["outcome"], s["permission_granted"]) for s in doc["steps"]] == [
        ("loaded", "CONFIRMED", True), ("aligned", "CONFIRMED", True),
        ("end_face", "CONFIRMED", True)]
    assert doc["steps"][0]["expects"] == {"part_present": True}
    assert doc["steps"][0]["dlp_result"]["observed"] == {"part_present": True}
    st, raw, _ = c.request("GET", b["last_cycle"]["links"]["log"])
    events = [json.loads(x)["event"] for x in raw.decode().splitlines()]
    assert st == 200 and events[0] == "run_start" and events[-1] == "run_end"
    assert events.count("work_state") == 3 and events.count("permission") == 3
    # the next START drops every permission; a profile change too
    assert c.post("/api/cmd/start", {})[0] == 200
    assert not any(_state(c)["permissions"].values())
    assert c.post("/api/cmd/abort", {})[1]["bus"]["last_result"] == "ABORTED"
    status, r = c.post("/api/cmd/profile", {"profile": "short_tube", "source": "plc"})
    assert status == 200 and r["bus"]["profile"] == "short_tube" and r["bus"]["cycle"] is None
    assert r["bus"]["registers"]["profile"] == 2
    status, r = c.post("/api/cmd/start", {"profile": "long_tube"})      # the page shows another
    assert status == 409 and "the selected profile is short_tube" in r["message"]
    c.post("/api/cmd/start", {"profile": 2})
    _step(c)
    assert fake.calls[-1]["profile"] == "short_tube"
    assert c.post("/api/cmd/profile", {"profile": "medium"})[0] == 409


def test_corrected_prompts_and_ack_holds_then_the_dent_is_kicked_out(line):
    c, app, fake = line
    fake.answers = {"loaded": [ABSENT, OK], "end_face": FAILS}
    c.post("/api/io/inputs", {"part_present": True})
    c.post("/api/cmd/start", {})
    b = _step(c)
    row = b["cycle"]["steps"][0]
    assert b["state"] == "WAITING" and row["outcome"] == "CORRECTED" and not row["permission_granted"]
    assert row["sensor_mismatch"] == [{"signal": "part_present", "sensor": True, "dlp": False,
                                       "expected": True}]
    assert b["signals"]["PROMPT"] and b["signals"]["SENSOR_MISMATCH"]
    assert not any(b["signals"][f"PERMIT_{p}"] for p in PERMITS)
    assert b["prompt"]["message"] == "no tube found: load or clear the sensor"
    assert b["prompt"]["title"] == "Tube loaded?" and b["cycle"]["waiting"] == "ack"
    assert b["registers"]["work_state_code"] == 2           # CORRECTED
    t0 = time.monotonic()
    while not fake.projected:                               # project_fault: its own job
        assert time.monotonic() - t0 < 30
        time.sleep(0.02)
    assert fake.projected == [row["dlp_result"]["run_id"]]
    assert c.post("/api/cmd/step", {})[0] == 409            # a prompt is open
    status, r = c.post("/api/cmd/ack", {"source": "operator"})
    assert status == 200 and "acknowledged" in r["message"]
    b = _state(c, {"WAITING"})
    assert _rows(b)[0] == ("loaded", "CONFIRMED", True) and not b["signals"]["PROMPT"]
    assert not b["signals"]["SENSOR_MISMATCH"] and b["cycle"]["steps"][0]["attempts"] == 2
    assert b["cycle"]["steps"][0]["earlier"][0]["actions"][1]["acknowledged_by"] == "operator"
    _step(c)
    b = _step(c)
    assert b["state"] == "READY" and b["last_result"] == "FAIL" and b["last_disposition"] == "REJECT"
    assert b["signals"]["FAIL"] and b["signals"]["KICKOUT"] and not b["signals"]["PASS"]
    assert not any(b["permissions"].values()) and b["registers"]["result_code"] == 2
    assert b["cycle"]["kickout"] == {"step": "end_face", "disposition": "REJECT",
                                     "reason": "end face not OK", "detail": "seat/check: failed",
                                     "by": "kickout"}
    doc = json.loads(c.request("GET", b["last_cycle"]["links"]["result_json"])[1])
    assert doc["result"] == "FAIL" and doc["disposition"] == "REJECT"
    assert doc["steps"][2]["outcome"] == "DEFECT" and doc["cycle"]["ended_by"] == "kickout"
    assert doc["steps"][0]["earlier"][0]["outcome"] == "CORRECTED"
    assert not any(doc["permissions"].values())


def test_doubt_error_and_abort_never_grant_over_http(line):
    c, app, fake = line
    c.post("/api/io/inputs", {"part_present": True})
    for ans in (answer("INVALID", "PULL", part_present=None), answer("PASS", "PULL", part_present=True),
                ui_server.ArgError("no stereo calibration at this geometry")):
        fake.answers = {"loaded": ans}
        c.post("/api/io/inputs", {"part_present": True})    # for THIS part
        c.post("/api/cmd/start", {})
        b = _step(c)
        assert b["cycle"]["steps"][0]["sensor_inputs"] == {"part_present": True}
        assert b["state"] == "READY" and b["last_disposition"] == "PULL", ans
        assert b["last_result"] == "INVALID" and b["signals"]["PULL"] and b["signals"]["ALARM"]
        assert not any(b["signals"][f"PERMIT_{p}"] for p in PERMITS) and not b["signals"]["PASS"]
        assert b["cycle"]["steps"][0]["outcome"] == "DOUBT"
        doc = json.loads(c.request("GET", b["last_cycle"]["links"]["result_json"])[1])
        assert doc["disposition"] == "PULL" and not any(doc["permissions"].values())
        assert doc["steps"][0]["permission_granted"] is False and doc["pull_reasons"]
        assert c.post("/api/cmd/reset", {})[0] == 200
    # ABORT while the DLP verifies: the cycle is ABORTED whatever the check says afterwards
    fake.answers = {"loaded": OK}
    fake.gate.clear()
    c.post("/api/cmd/start", {})
    assert c.post("/api/cmd/step", {})[0] == 200
    t0 = time.monotonic()
    while len(fake.calls) < 4:
        assert time.monotonic() - t0 < 30
        time.sleep(0.02)
    status, r = c.post("/api/cmd/abort", {})
    assert status == 200 and r["bus"]["state"] in ("ABORTING", "READY")
    b = _state(c, {"READY"})
    assert b["last_result"] == "ABORTED" and b["last_disposition"] == "PULL"
    assert b["cycle"]["steps"][0]["outcome"] == "ABORTED" and not any(b["permissions"].values())
    fake.gate.set()
    # a sensor nobody reported: DOUBT (the DLP has nothing to confirm or correct)
    c.post("/api/io/inputs", {"part_present": None})
    c.post("/api/cmd/start", {})
    b = _step(c)
    assert b["last_disposition"] == "PULL" and "has not reported" in b["cycle"]["steps"][0]["message"]


def test_the_deprecated_workstate_route_still_selects_a_configuration(line):
    """/api/cmd/workstate and the `workstate` body key are the old names of /api/cmd/config
    and `config`: they selected a station configuration, never a work state."""
    c, app, fake = line
    status, r = c.post("/api/cmd/workstate", {"workstate": 2})
    assert status == 200 and r["command"] == "SELECT_CONFIG"
    assert r["bus"]["station_config"] == "old_single_job.json" and "deprecated" in r["message"]
    assert r["bus"]["profile"] == "default" and r["bus"]["legacy"] is True
    status, r = c.post("/api/cmd/start", {"workstate": "example_tube_line.json"})
    assert status == 409 and "the selected station configuration is old_single_job.json" in r["message"]
    # the single job still runs as before: START -> its job -> PASS, no STEP, no permission
    status, r = c.post("/api/cmd/start", {"workstate": "old_single_job.json"})
    assert status == 200 and r["bus"]["state"] == "RUNNING"
    assert r["message"] == ("started old_single_job.json [the `workstate` key is deprecated: "
                            "use `config`]")
    b = _state(c, {"READY"})
    assert b["last_result"] == "PASS" and b["signals"]["PASS"] and b["permissions"] == {}
    assert fake.runs == [{"config": "old_single_job.json", "operator_confirmed": False}]
    assert fake.calls == [] and b["cycle"]["legacy"] and b["cycle"]["steps"][0]["outcome"] == "CONFIRMED"
    assert c.post("/api/cmd/step", {})[0] == 409
    status, r = c.post("/api/cmd/config", {"config": "example_tube_line.json"})
    assert status == 200 and "deprecated" not in r["message"]
    assert r["bus"]["profile"] == "long_tube" and r["bus"]["permissions"] == dict.fromkeys(PERMITS, False)
    # another configuration: nothing of the single job's PASS is left on the line
    assert r["bus"]["last_result"] is None and not r["bus"]["signals"]["PASS"]
    assert r["bus"]["registers"]["result_code"] == 0 and r["bus"]["cycle"] is None
    # the old body key on the NEW route is deprecated too, and says so
    status, r = c.post("/api/cmd/config", {"workstate": 2})
    assert status == 200 and r["bus"]["station_config"] == "old_single_job.json"
    assert "the `workstate` key is deprecated: use `config`" in r["message"]


def test_the_simulated_line_is_mock_only(line):
    c, app, fake = line
    assert app.mock_case == "all_good" and app.bus.inputs == {"part_present": True}
    status, r = c.post("/api/mock/case", {"case": "sensor_false_part_present"})
    assert status == 200 and r["case"] == "sensor_false_part_present"
    assert r["inputs"] == {"part_present": False} and r["bus"]["mock"]["case"] == r["case"]
    cfg = app.bus.flow_of("example_tube_line.json")
    assert app._mock_scene_now(cfg, "long_tube") == "scene:tube_long"
    c.post("/api/mock/case", {"case": "dented"})
    assert app._mock_scene_now(cfg, "short_tube") == "scene:tube_short:dent"
    assert app.bus.inputs == {"part_present": True}
    status, r = c.post("/api/mock/case", {"scene": "none", "inputs": {"part_present": True}})
    assert status == 200 and app._mock_scene_now(cfg, "long_tube") == "none"
    assert c.post("/api/mock/case", {})[1]["case"] is None
    assert app._mock_scene_now(cfg, "long_tube") == "scene:tube_long"           # the default case
    for bad in ({"case": "nope"}, {"scene": "scene:nope"}, {"inputs": {"part_present": "yes"}}):
        assert c.post("/api/mock/case", bad)[0] == 400, bad
    # another configuration, and back: the simulated line is the configuration's own again
    c.post("/api/mock/case", {"case": "dented"})
    c.post("/api/io/inputs", {"part_present": False})
    assert c.post("/api/cmd/config", {"config": 2})[1]["bus"]["mock"]["case"] is None
    r = c.post("/api/cmd/config", {"config": 1})[1]
    assert r["bus"]["mock"]["case"] == "all_good" and r["bus"]["inputs"] == {"part_present": True}
    c.post("/api/mock/case", {"case": "dented"})            # a profile change keeps the case
    assert c.post("/api/cmd/profile", {"profile": 2})[1]["bus"]["mock"]["case"] == "dented"
    app.mock = False                    # what a REAL-mode server answers (nothing is opened)
    try:
        status, r = c.post("/api/mock/case", {"case": "dented"})
        assert status == 400 and "MOCK ONLY" in r["error"]
        assert app._mock_scene_now(cfg, "long_tube") is None
    finally:
        app.mock = True


def test_a_work_state_that_waits_too_long_ends_its_cycle_pull(tmp_path):
    """timeout_s: nobody sends STEP (or ACK). The server's own tick ends the cycle PULL, even
    when no page polls."""
    httpd, app = _serve(tmp_path, _stations(tmp_path, loaded={"timeout_s": 0.3}))
    FakeDLP(app)
    c = Client(app.port)
    try:
        assert c.post("/api/cmd/start", {})[0] == 200
        t0 = time.monotonic()
        while app.bus.state != "READY":                     # no HTTP poll: the ticker thread
            assert time.monotonic() - t0 < 30
            time.sleep(0.05)
        b = _state(c)
        assert b["last_result"] == "INVALID" and b["last_disposition"] == "PULL"
        assert b["cycle"]["ended_by"] == "timeout" and b["signals"]["ALARM"]
        doc = json.loads(c.request("GET", b["last_cycle"]["links"]["result_json"])[1])
        assert doc["pull_reasons"][0]["reason"] == "timeout" and doc["steps"][0]["outcome"] == "DOUBT"
    finally:
        httpd.shutdown()
        httpd.server_close()
        app.close()
        api._session = None


def test_flow_verbs_are_exposed_and_refuse_what_does_not_fit(line):
    c, app, fake = line
    for v in ("flow_verify", "flow_run"):
        assert v in ui_server.VERBS and v in ui_server.CANCELLABLE_VERBS
        assert v not in ui_server.RIG_FREE_VERBS
    del app._v_flow_verify                                  # the real verb (it refuses first)
    j = c.job("flow_verify", {"config": "old_single_job.json", "step": "run"})
    assert j["status"] == "error" and "single-job station config" in j["error"]
    j = c.job("flow_verify", {"config": "example_tube_line.json", "step": "loaded",
                              "cycle_id": "nope"})
    assert j["status"] == "error" and "no open cycle" in j["error"]
    j = c.job("flow_verify", {"config": "example_tube_line.json", "step": "loaded",
                              "operator_confirmed": "yes"})
    assert j["status"] == "error" and "JSON boolean" in j["error"]
    j = c.job("flow_run", {"config": "example_tube_line.json", "inputs": {"part_present": "true"}})
    assert j["status"] == "error" and "inputs" in j["error"]
    j = c.job("flow_run", {"config": "example_tube_line.json", "mock_case": "nope"})
    assert j["status"] == "error" and "no mock case" in j["error"]


# ================================================================== quick.js (node, fake DOM)

_FLOW_HARNESS = r"""
const fs = require('fs'), vm = require('vm');
class El {
  constructor(tag, id) { this.tagName = String(tag).toUpperCase(); this.id = id; this.children = [];
    this.attrs = {}; this.dataset = {}; this.className = ''; this._text = ''; this._value = '';
    this.checked = false; this.disabled = false; this.hidden = false; this._sel = undefined;
    this.parent = null; this.isContentEditable = false; this.href = ''; this.title = '';
    this.classList = { toggle() {}, add() {}, remove() {} }; }
  set textContent(t) { this._text = String(t); this.children = []; }
  get textContent() { return this._text + this.children.map((c) => c.textContent).join(' '); }
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
  matches() { return false; }
  closest(sel) { for (let e = this; e; e = e.parent) if (sel === '[data-signal]' && e.dataset.signal) return e; return null; }
}
const TAGS = { qStation: 'select', qProfile: 'select', qMockCase: 'select', qConfirm: 'input',
  qStart: 'button', qAbort: 'button', qHome: 'button', qArm: 'button', qTrigger: 'button',
  qStep: 'button', qAck: 'button' };
const byId = {};
const el = (id) => byId[id] || (byId[id] = new El(TAGS[id] || 'div', id));
const script = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
const sent = [];
const srv = { bus: script.states[0], job: { status: 'queued' } };
const sandbox = { console, setTimeout, clearTimeout, setInterval, clearInterval,
  document: { addEventListener() {}, querySelector: (s) => el(s.slice(1)), querySelectorAll: () => [],
              createElement: (t) => new El(t), body: new El('body'),
              createTextNode: (t) => { const n = new El('#text'); n.textContent = t; return n; } },
  fetch: async (path, opt) => {
    const body = opt && opt.body ? JSON.parse(opt.body) : undefined;
    sent.push({ path, body });
    let data = {};
    if (path === '/api/cmd/state') data = srv.bus;
    else if (path === '/api/stations') data = { mode: 'mock', stations: script.stations };
    else if (path.startsWith('/api/cmd/')) data = srv.refuse
      ? { accepted: false, command: 'SELECT_PROFILE', message: srv.refuse, bus: srv.bus, jobs: {} }
      : { accepted: true, command: 'X', message: 'ok', bus: srv.bus, jobs: {} };
    else if (path.startsWith('/api/jobs/')) data = srv.job;
    else data = { bus: srv.bus };
    return { ok: true, status: 200, json: async () => data };
  } };
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(process.argv[2], 'utf8'), sandbox, { filename: 'quick.js' });
const run = (code) => vm.runInContext(code, sandbox);
const key = (k) => { const n = sent.length;
  run('onKey')({ key: k, target: sandbox.document.body, ctrlKey: false, altKey: false, metaKey: false, preventDefault() {} });
  return sent.slice(n).map((s) => s.path); };
const snap = () => ({
  lamp: el('qLamp').textContent, lampCls: el('qLamp').className,
  config: el('qStation').value, profile: el('qProfile').value,
  profiles: el('qProfile').options.map((o) => o.textContent),
  configs: el('qStation').options.map((o) => o.textContent),
  info: el('qStationInfo').textContent,
  rows: el('qFlow').children.map((li) => ({ cls: li.className, text: li.textContent })),
  inputs: el('qInputs').children.map((c) => c.textContent),
  permits: el('qPermits').children.map((c) => [c.textContent, c.className]),
  promptHidden: el('qPrompt').hidden, prompt: el('qPromptTitle').textContent + ' ' + el('qPromptMsg').textContent,
  promptDetail: el('qPromptDetail').textContent,
  step: el('qStep').disabled, ack: el('qAck').disabled, start: el('qStart').disabled,
  trigger: el('qTrigger').disabled, profileOff: el('qProfile').disabled, stationOff: el('qStation').disabled,
  mockHidden: el('qMock').hidden, mockCases: el('qMockCase').options.map((o) => o.value),
  verdict: el('qVerdict').textContent, disp: el('qDisposition').textContent,
  number: el('qNumber').textContent, pull: el('qPull').textContent, pullHidden: el('qPull').hidden,
  feedback: el('qFeedback').textContent, reason: el('qReason').textContent,
  run: el('qRunId').textContent,
  notice: el('qNotice').hidden ? null : el('qNotice').textContent,
  resultLink: el('qResultLink').hidden ? null : el('qResultLink').href });
(async () => {
  const out = { shots: [] };
  await run('loadStations()');
  out.shots.push(snap());
  out.keysIdle = { n: key('n'), k: key('k'), p: key('P') };
  await new Promise((r) => setTimeout(r, 20));
  for (let i = 1; i < script.states.length; i += 1) {
    srv.bus = script.states[i];
    srv.job = script.jobs[i] || { status: 'queued' };
    await run('pollOnce()');
    out.shots.push(snap());
    if (i === 1) { out.keysWaitStep = { n: key('N'), k: key('K') }; await new Promise((r) => setTimeout(r, 20)); }
    if (i === 3) {
      out.keysPrompt = { n: key('n'), k: key('k') }; await new Promise((r) => setTimeout(r, 20));
      // P pressed in a cycle: refused - the refusal stays on the page through the next polls
      srv.refuse = 'refused while WAITING: a cycle is running';
      key('p');
      await new Promise((r) => setTimeout(r, 20));
      await run('pollOnce()');
      out.refused = snap();
      srv.refuse = null;
    }
  }
  // START names the configuration and the profile the page shows
  srv.bus = script.states[0];
  await run('pollOnce()');
  await run("send('start')");
  out.startBody = sent.filter((s) => s.path === '/api/cmd/start').pop().body;
  await run("selectProfile('short_tube')");
  out.profileBody = sent.filter((s) => s.path === '/api/cmd/profile').pop().body;
  // MOCK: a sensor chip toggles the simulated sensor, the case selector sets the line
  out.noticeAfterAccepted = el('qNotice').hidden;
  run('onInputClick')({ target: el('qInputs').children[0] });
  await new Promise((r) => setTimeout(r, 20));
  out.inputBody = sent.filter((s) => s.path === '/api/mock/case').pop().body;
  out.inputsToLine = sent.filter((s) => s.path === '/api/io/inputs').length;
  await run("setMockCase('dented')");
  out.mockBody = sent.filter((s) => s.path === '/api/mock/case').pop().body;
  out.digits = key('2');
  await new Promise((r) => setTimeout(r, 20));
  out.configBody = sent.filter((s) => s.path === '/api/cmd/config').pop().body;
  process.stdout.write(JSON.stringify(out));
})().catch((e) => { console.error(e); process.exit(1); });
"""


def _page_states() -> tuple:
    """Bus snapshots as the server gives them, built with the real bus (no server)."""
    from engine.control import commands as C
    from engine.flow import schema as FS
    cfg = FS.load(EXAMPLE)
    bus = C.CommandBus(C.Handlers(cycle_begin=lambda info: {"cycle_id": "cyc_1"}))
    bus.update_configs(["example_tube_line.json", "old.json"], flows={"example_tube_line.json": cfg})
    bus.set_inputs({"part_present": True})

    def snap(**extra):
        return {**json.loads(json.dumps(bus.snapshot())), "jobs": {}, "active_preset": PRESET,
                "run_preset": PRESET, "warnings": [], "mode": "mock", "last_cycle": None,
                "mock": {"case": None, "scene": None, "inputs": {"part_present": True},
                         "cases": cfg.public()["mock_cases"]},
                **extra}

    def fin(a):
        row = bus.cycle.row
        bus.job_finished("run", a["result"], None, disposition=a["disposition"],
                         step={**a, "run_id": "r", "step": row["id"], "attempt": row["attempts"],
                               "cycle_id": "cyc_1"})

    states, jobs = [snap()], {}
    bus.command("START")
    states.append(snap())                                   # 1: waits for STEP at work state 1
    bus.command("STEP")
    states.append(snap(jobs={"j7": "run"}))                 # 2: the DLP verifies it
    fin(ABSENT)
    states.append(snap())                                   # 3: CORRECTED, a prompt waits
    # the step's own job says FAIL / REJECT: the lamp must not turn red for it
    jobs[3] = {"id": "j7", "status": "done", "links": {},
               "result": {"result": "FAIL", "disposition": "REJECT", "zones": [],
                          "job": {"work_state": "loaded"}, "feedback": {"rendered": True,
                                                                        "hold_s": 5}}}
    bus.command("ACK")
    fin(OK)
    states.append(snap(jobs={"j8": "run"}))                 # 4: confirmed, LOAD_DONE granted
    jobs[4] = {"id": "j8", "status": "done", "links": {},
               "result": {"result": "PASS", "disposition": "CONTINUE", "zones": [],
                          "job": {"work_state": "loaded"}, "feedback": {"rendered": True,
                                                                        "hold_s": 5}}}
    for a in (GOOD, GOOD):
        bus.command("STEP")
        fin(a)
    states.append(snap(last_cycle={"cycle_id": "cyc_1", "links": {
        "result_json": "/runs/cyc_1/result.json"}}))        # 5: every work state CONFIRMED
    assert bus.command("SELECT_PROFILE", arg="short_tube").accepted
    states.append(snap())                                   # 6: another profile: forgotten
    stations = [{"config": "example_tube_line.json", "station_id": "example_tube_line",
                 "job_type": "flow", "kind": "flow", "profiles": cfg.public()["profiles"],
                 "flow": cfg.public()["flow"], "description": cfg.description,
                 "part_label": "long_tube"},
                {"config": "old.json", "station_id": "old", "part_label": "panel",
                 "job_type": "metric", "preset": None}]
    return states, jobs, stations


def test_quick_js_renders_the_configuration_the_profile_and_the_work_states(tmp_path):
    """The real quick.js (node, fake DOM) on real bus snapshots: the selectors, one row per
    work state with its sensor / DLP / permission chips, the prompt banner + ACK, the lamp
    following the CYCLE (never one work state's own verdict), the keys N / K / P."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed; quick.js logic not checked")
    states, jobs, stations = _page_states()
    f = tmp_path / "script.json"
    f.write_text(json.dumps({"states": states, "jobs": jobs, "stations": stations}))
    h = tmp_path / "flow_harness.js"
    h.write_text(_FLOW_HARNESS, encoding="utf-8")
    r = subprocess.run([node, str(h), str(STATIC / "quick.js"), str(f)], capture_output=True,
                       text=True, encoding="utf-8", timeout=60)
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    idle, wait, verifying, prompt, granted, done, forgot = out["shots"]
    # top: the station configuration and the profile of its profile library set
    assert idle["config"] == "example_tube_line.json" and idle["profile"] == "long_tube"
    assert idle["configs"] == ["1: example_tube_line - 3 work states", "2: old - panel"]
    assert idle["profiles"] == ["1: Long tube (60 mm)", "2: Short tube (38 mm)"]
    assert "2 profile(s) (long_tube, short_tube), 3 work state(s)" in idle["info"]
    # middle: the flow, nothing verified yet: the sensor chip shows the live claim
    assert [r["cls"] for r in idle["rows"]] == ["qws o-pending"] * 3
    assert "Tube loaded?" in idle["rows"][0]["text"] and "End face OK?" in idle["rows"][2]["text"]
    assert "sensor says part_present true" in idle["rows"][0]["text"]
    assert "permission LOAD_DONE withheld" in idle["rows"][0]["text"]
    assert idle["inputs"] == ["part_present true"] and idle["promptHidden"] is True
    assert idle["permits"] == [[f"PERMIT_{p}", ""] for p in PERMITS]
    assert idle["step"] is True and idle["ack"] is True and idle["start"] is False
    assert idle["mockHidden"] is False and "dented" in idle["mockCases"]
    assert out["keysIdle"] == {"n": [], "k": [], "p": ["/api/cmd/profile"]}    # N / K: nothing waits
    # a cycle waits for STEP at work state 1: highlighted, STEP on, selectors locked
    assert wait["lamp"] == "STEP 1/3" and wait["lampCls"] == "qlamp lamp-wait"
    assert wait["rows"][0]["cls"] == "qws o-pending current" and "waits for STEP" in wait["rows"][0]["text"]
    assert wait["step"] is False and wait["trigger"] is False and wait["start"] is True
    assert wait["profileOff"] is True and wait["stationOff"] is True
    assert out["keysWaitStep"] == {"n": ["/api/cmd/step"], "k": []}
    assert verifying["lamp"] == "VERIFYING 1/3" and "VERIFYING" in verifying["rows"][0]["text"]
    assert verifying["step"] is True
    # CORRECTED: both chips flagged, the prompt banner with ACK; the step's own FAIL / REJECT
    # never turns the lamp red
    assert prompt["lamp"] == "ACK NEEDED" and prompt["lampCls"] == "qlamp lamp-invalid"
    row = prompt["rows"][0]
    assert row["cls"] == "qws o-corrected current" and "CORRECTED" in row["text"]
    assert "sensor says part_present true" in row["text"] and "DLP says part_present false" in row["text"]
    assert "permission LOAD_DONE withheld" in row["text"] and "prompt (waits for ACK)" in row["text"]
    assert prompt["promptHidden"] is False and prompt["ack"] is False and prompt["step"] is True
    assert prompt["prompt"] == "1. Tube loaded?: no tube found: load or clear the sensor"
    assert "sensor part_present says true, the DLP says false" in prompt["promptDetail"]
    assert out["keysPrompt"] == {"n": [], "k": ["/api/cmd/ack"]}
    # what was painted is the cycle's project_fault action - never claimed for a check that
    # merely rendered a picture
    assert prompt["feedback"].startswith("Fault sent to the projector") and granted["feedback"] == ""
    # confirmed after the ACK: the permission chip and the PERMIT output are on
    assert "CONFIRMED" in granted["rows"][0]["text"]
    assert "permission LOAD_DONE GRANTED" in granted["rows"][0]["text"]
    assert granted["permits"][0] == ["PERMIT_LOAD_DONE", "on"] and granted["permits"][1][1] == ""
    assert granted["rows"][1]["cls"] == "qws o-pending current" and granted["lamp"] == "STEP 2/3"
    assert granted["promptHidden"] is True
    # the cycle ended CONTINUE: PASS comes from the bus outputs
    assert done["lamp"] == "PASS" and done["lampCls"] == "qlamp lamp-pass"
    assert [r["cls"] for r in done["rows"]] == ["qws o-confirmed"] * 3
    assert all(c == "on" for _, c in done["permits"]) and done["disp"] == "CONTINUE"
    assert done["verdict"] == "PASS" and "3/3 End face OK?: CONFIRMED" in done["number"]
    assert done["resultLink"] == "/runs/cyc_1/result.json" and done["pullHidden"] is True
    # the claims were made for that cycle: afterwards the chip shows the SIMULATED sensor (MOCK)
    assert done["inputs"] == ["part_present true"]
    # another profile selected: the long tube's PASS, its work states, its run and its link
    # are gone from the lamp and the panel (the bus has forgotten that cycle)
    assert forgot["lamp"] == "READY" and forgot["lampCls"] == "qlamp lamp-ready"
    assert forgot["profile"] == "short_tube" and forgot["verdict"] == "-" and forgot["disp"] == "-"
    assert forgot["number"] == "-" and forgot["run"] == "-" and forgot["resultLink"] is None
    assert [r["cls"] for r in forgot["rows"]] == ["qws o-pending"] * 3
    assert all(c == "" for _, c in forgot["permits"])
    # a command the bus refused in a cycle: the text stays through the polls (the cycle's own
    # line is redrawn every time), until a command is accepted
    assert out["refused"]["notice"] == "SELECT_PROFILE: refused while WAITING: a cycle is running"
    assert out["refused"]["lamp"] == "ACK NEEDED" and out["noticeAfterAccepted"] is True
    # what the page sends
    assert out["startBody"] == {"source": "quick", "operator_confirmed": True,
                                "config": "example_tube_line.json", "profile": "long_tube"}
    assert out["profileBody"] == {"profile": "short_tube", "source": "quick"}
    # MOCK: a sensor chip sets the SIMULATED line's sensor (it reports for every cycle); the
    # page never posts as the line itself
    assert out["inputBody"] == {"inputs": {"part_present": False}} and out["inputsToLine"] == 0
    assert out["mockBody"] == {"case": "dented"}
    assert out["digits"] == ["/api/cmd/config"] and out["configBody"] == {"config": 2,
                                                                          "source": "quick"}


# ================================================================== the full mock capture path

@pytest.fixture(scope="module")
def captured(tmp_path_factory):
    """A MOCK server whose library has the bench's TRUE stereo calibration and a golden per
    profile: the DLP checks really capture (the simulated bench), nothing is faked."""
    tmp = tmp_path_factory.mktemp("flow_capture")
    s = api.configure(tmp / "lib", mock=True)
    try:
        p = api.load_preset(PRESET)
        RS.truth_calibration(s, p)
        for prof in ("long", "short"):
            g = RS.oracle_scan(s, p, f"scene:tube_{prof}", f"tube_{prof}_alignment_golden", seed=9)
            runner.record_golden(s, api.recipe_load(f"tube_{prof}_alignment"), p, g)
    finally:
        api._session = None
    httpd, app = _serve(tmp, _stations(tmp))
    try:
        yield Client(app.port), app
    finally:
        httpd.shutdown()
        httpd.server_close()
        app.close()
        api._session = None


def test_a_cycle_through_the_mock_capture_path_confirms_or_corrects_the_sensor(captured):
    """START / STEP on the simulated line, every DLP check a real mock capture. Cycle 1: the
    simulated sensor says a tube is there and the seat is EMPTY - the DLP corrects it, paints
    the missing-part zone, prompts; the simulated operator loads a tube, ACK - verified again:
    CONFIRMED. Then aligned, end face: CONTINUE, every permission granted. Cycle 2: the short
    tube with a dent - kicked out REJECT at "End face OK?", nothing released."""
    c, app = captured
    wait = {"WAITING", "READY", "FAULT"}
    assert c.post("/api/mock/case", {"case": "sensor_true_part_absent"})[0] == 200
    assert c.post("/api/cmd/start", {})[0] == 200
    assert c.post("/api/cmd/step", {})[0] == 200
    b = _state(c, wait, 240)
    row = b["cycle"]["steps"][0]
    assert row["outcome"] == "CORRECTED" and row["sensor_inputs"] == {"part_present": True}
    assert row["dlp_result"]["observed"] == {"part_present": False}
    assert row["dlp_result"]["recipe"] == "tube_long_presence" and row["dlp_result"]["scan_id"]
    assert b["signals"]["PROMPT"] and b["signals"]["SENSOR_MISMATCH"]
    assert b["prompt"]["message"] == "no tube found: load or clear the sensor"
    assert row["actions"][0] == {"action": "project_fault", "requested": True}
    jobs = c.get("/api/jobs")[1]["jobs"]
    assert [j["verb"] for j in jobs[:2]] == ["recipe_feedback", "flow_verify"]   # the fault, painted
    step_job = c.get(f"/api/jobs/{jobs[1]['id']}")[1]
    assert step_job["result"]["dlp"]["observed"] == {"part_present": False}
    assert c.request("GET", step_job["links"]["feedback"])[0] == 200
    assert c.request("GET", step_job["links"]["preview"])[0] == 200
    c.post("/api/mock/case", {"case": "all_good"})          # the operator loads a tube
    assert c.post("/api/cmd/ack", {})[0] == 200
    b = _state(c, wait, 240)
    assert _rows(b)[0] == ("loaded", "CONFIRMED", True) and b["signals"]["PERMIT_LOAD_DONE"]
    for n in (2, 3):
        assert c.post("/api/cmd/step", {"step": n})[0] == 200
        b = _state(c, wait, 240)
    assert b["state"] == "READY" and b["last_disposition"] == "CONTINUE" and b["signals"]["PASS"]
    assert _rows(b) == [("loaded", "CONFIRMED", True), ("aligned", "CONFIRMED", True),
                        ("end_face", "CONFIRMED", True)]
    assert all(b["signals"][f"PERMIT_{p}"] for p in PERMITS)
    doc = json.loads(c.request("GET", b["last_cycle"]["links"]["result_json"])[1])
    assert doc["disposition"] == "CONTINUE" and doc["mode"] == "MOCK"
    assert doc["steps"][0]["attempts"] == 2 and doc["steps"][0]["earlier"][0]["outcome"] == "CORRECTED"
    assert {s["dlp_result"]["recipe"] for s in doc["steps"]} == {
        "tube_long_presence", "tube_long_alignment", "tube_long_end_face"}
    assert len(list((Path(doc["run_folder"]) / "steps").iterdir())) == 4
    # ---- the other profile: a short tube with a dent
    assert c.post("/api/cmd/profile", {"profile": "short_tube"})[0] == 200
    assert not any(_state(c)["permissions"].values())
    c.post("/api/mock/case", {"case": "dented"})
    c.post("/api/cmd/start", {})
    for _ in range(3):
        assert c.post("/api/cmd/step", {})[0] == 200
        b = _state(c, wait, 240)
    assert b["state"] == "READY" and b["last_result"] == "FAIL" and b["last_disposition"] == "REJECT"
    # (the rows say what stands now: the permissions granted on the way dropped with the REJECT)
    assert _rows(b) == [("loaded", "CONFIRMED", False), ("aligned", "CONFIRMED", False),
                        ("end_face", "DEFECT", False)]
    assert b["cycle"]["steps"][0]["permission_note"] == "dropped: the cycle ended REJECT"
    assert b["signals"]["FAIL"] and b["signals"]["KICKOUT"] and not any(b["permissions"].values())
    assert b["cycle"]["steps"][2]["dlp_result"]["recipe"] == "tube_short_end_face"
    assert "defect" in b["cycle"]["steps"][2]["message"]
    k = b["cycle"]["kickout"]
    assert k["reason"] == "end face not OK" and k["detail"].startswith("end_face/defect")
    assert c.post("/api/cmd/abort", {})[0] == 200           # clears the projected fault
    # ---- a tube taken away after "Tube aligned?": at "End face OK?" the sensor still says
    # present, the seat is empty. That is a sensor / DLP disagreement (is that tube there or
    # not?) - never a dent, never a REJECT
    c.post("/api/mock/case", {"case": "all_good"})
    assert c.post("/api/cmd/start", {})[0] == 200
    for _ in range(2):
        assert c.post("/api/cmd/step", {})[0] == 200
        b = _state(c, wait, 240)
    assert _rows(b)[:2] == [("loaded", "CONFIRMED", True), ("aligned", "CONFIRMED", True)]
    assert c.post("/api/mock/case", {"scene": "none"})[0] == 200
    assert c.post("/api/cmd/step", {})[0] == 200
    b = _state(c, wait, 240)
    row = b["cycle"]["steps"][2]
    assert b["state"] == "WAITING" and row["outcome"] == "CORRECTED", row
    assert row["sensor_inputs"] == {"part_present": True}
    assert row["dlp_result"]["observed"] == {"part_present": False}
    assert row["sensor_mismatch"][0]["signal"] == "part_present"
    assert b["prompt"]["message"] == "no tube found: load or clear the sensor"
    assert b["signals"]["SENSOR_MISMATCH"] and b["signals"]["PROMPT"]
    assert not b["signals"]["FAIL"] and not b["signals"]["KICKOUT"]
    assert not b["signals"]["PERMIT_TRANSFER"] and b["cycle"]["kickout"] is None
    # ABORT while the mismatch prompt is open: the record keeps the mismatch
    assert c.post("/api/cmd/abort", {})[0] == 200
    b = _state(c, {"READY"}, 240)
    assert b["last_result"] == "ABORTED" and b["signals"]["PULL"] and b["signals"]["KICKOUT"]
    doc = json.loads(c.request("GET", b["last_cycle"]["links"]["result_json"])[1])
    assert doc["cycle"]["sensor_mismatch"] is True and doc["cycle"]["kickout"]["by"] == "abort"
    assert doc["steps"][2]["outcome"] == "ABORTED"
    assert doc["steps"][2]["earlier"][-1]["outcome"] == "CORRECTED"
    assert [r["reason"] for r in doc["pull_reasons"]] == ["aborted", "sensor_mismatch"]
    assert not any(doc["permissions"].values())
    assert not any(s["permission_granted"] for s in doc["steps"])


def test_an_uncalibrated_station_is_a_doubt_on_every_path_never_a_permission(line):
    """The real verbs (no fake) on a library with no stereo calibration: no mm, so the DLP
    cannot tell. `run_station` / `flow_run` (a whole cycle in line) and START / STEP on the
    bus all end PULL with the first work state in DOUBT - nothing is granted, nothing passes."""
    c, app, fake = line
    del app._v_flow_verify, app._v_run_station              # the real verbs
    for verb, args in (("run_station", {"config": "example_tube_line.json"}),
                       ("flow_run", {"config": "example_tube_line.json", "profile": "short_tube",
                                     "inputs": {"part_present": True}, "acknowledge": True})):
        j = c.job(verb, args)
        assert j["status"] == "done", j
        res = j["result"]
        assert (res["result"], res["disposition"], res["exit_code"]) == ("INVALID", "PULL", 5), verb
        assert res["job"]["type"] == "flow" and res["schema_version"] == "0.4"
        first = res["steps"][0]
        assert first["outcome"] == "DOUBT" and first["permission_granted"] is False
        assert "stereo calibration" in first["message"] or "stereo calibration" in str(
            first["dlp_result"])
        assert [s["outcome"] for s in res["steps"][1:]] == ["PENDING", "PENDING"]
        assert not any(res["permissions"].values()) and res["pull_reasons"]
        assert c.request("GET", j["links"]["result_json"])[0] == 200
    assert j["result"]["profile"] == "short_tube"
    c.post("/api/io/inputs", {"part_present": True})
    assert c.post("/api/cmd/start", {})[0] == 200
    b = _step(c)
    assert b["state"] == "READY" and b["last_result"] == "INVALID" and b["last_disposition"] == "PULL"
    assert b["signals"]["PULL"] and b["signals"]["ALARM"] and not b["signals"]["PASS"]
    assert not any(b["signals"][f"PERMIT_{p}"] for p in PERMITS)
    assert b["cycle"]["steps"][0]["outcome"] == "DOUBT" and b["cycle"]["kickout"]["disposition"] == "PULL"
    assert {p["reason"] for p in b["cycle"]["pull_reasons"]} == {"uncalibrated"}


# ================================================================== fail closed over HTTP

def _good_cycle(c) -> dict:
    """One cycle that ends CONTINUE (the line reports its sensor for this part first)."""
    c.post("/api/io/inputs", {"part_present": True})
    assert c.post("/api/cmd/start", {})[0] == 200
    for _ in range(3):
        b = _step(c)
    assert b["state"] == "READY" and b["signals"]["PASS"] and all(b["permissions"].values())
    return b


def _released(b) -> None:
    assert not any(v for k, v in b["signals"].items() if k.startswith("PERMIT_")), b["signals"]
    assert not b["signals"]["PASS"] and b["last_result"] is None and b["cycle"] is None
    assert (b["registers"]["result_code"], b["registers"]["work_state"],
            b["registers"]["work_state_code"]) == (0, 0, 0)


def test_a_refused_trigger_never_leaves_the_last_parts_permits_for_the_plc(line):
    """The documented handshake: STEP / TRIGGER -> BUSY -> read work_state_code and the
    PERMIT bit. After a TRIGGER the bus did not accept, nothing of the part before is left
    to be read as the new part's."""
    c, app, fake = line
    _good_cycle(c)
    status, r = c.post("/api/cmd/trigger", {"source": "plc"})
    assert status == 409 and "ignored: not armed" in r["message"]
    _released(r["bus"])
    _released(_state(c))
    # armed line: DISARM, then a TRIGGER
    assert c.post("/api/cmd/arm", {})[0] == 200
    c.post("/api/io/inputs", {"part_present": True})
    assert c.post("/api/cmd/trigger", {})[0] == 200
    b = _state(c, {"WAITING"})
    for _ in range(2):
        assert c.post("/api/cmd/trigger", {})[0] == 200
        b = _state(c, {"WAITING", "ARMED"})
    assert b["state"] == "ARMED" and b["signals"]["PASS"] and all(b["permissions"].values())
    assert c.post("/api/cmd/disarm", {})[0] == 200
    _released(_state(c))
    assert c.post("/api/cmd/trigger", {})[0] == 409
    _released(_state(c))
    # START naming another profile than the selected one
    _good_cycle(c)
    status, r = c.post("/api/cmd/start", {"profile": "short_tube"})
    assert status == 409
    _released(r["bus"])


def test_a_sensor_claim_posted_once_is_not_reused_for_the_next_part(line):
    c, app, fake = line
    b = _good_cycle(c)
    assert b["inputs"] == {}                                # forgotten with the cycle
    n = sum(1 for h in app.bus.history if h[2] == "INPUTS" and "forgotten" not in h[4])
    for _ in range(2):                                      # the gateway says nothing more
        assert c.post("/api/cmd/start", {})[0] == 200
        b = _step(c)
        assert b["state"] == "READY" and b["last_disposition"] == "PULL"
        row = b["cycle"]["steps"][0]
        assert row["outcome"] == "DOUBT" and row["sensor_inputs"] == {"part_present": None}
        assert b["cycle"]["pull_reasons"][0]["reason"] == "sensor_unknown"
        assert not any(b["signals"][f"PERMIT_{p}"] for p in PERMITS)
        assert c.post("/api/cmd/reset", {})[0] == 200
    assert sum(1 for h in app.bus.history if h[2] == "INPUTS" and "forgotten" not in h[4]) == n
    # MOCK: a signal the line posted itself is the line's: the simulated sensor is quiet
    assert app.mock_inputs == {} and _state(c)["mock"]["inputs"] == {}
    # the simulated line (a mock case) reports for every cycle, as a line's gateway must
    assert c.post("/api/mock/case", {"case": "all_good"})[0] == 200
    for _ in range(2):
        assert c.post("/api/cmd/start", {})[0] == 200
        b = _step(c)
        assert _rows(b)[0] == ("loaded", "CONFIRMED", True)
        assert c.post("/api/cmd/abort", {})[0] == 200
    assert [h[1] for h in app.bus.history if h[2] == "INPUTS"][-2] == "mock line"
    # ... until its simulated sensor is told to stop reporting
    assert c.post("/api/mock/case", {"inputs": {"part_present": None}})[0] == 200
    assert c.post("/api/cmd/start", {})[0] == 200
    assert _step(c)["cycle"]["steps"][0]["outcome"] == "DOUBT"


def test_pass_does_not_survive_a_profile_change_and_digit_profiles_select_by_name(tmp_path):
    stations = _stations(tmp_path)
    doc = json.loads((stations / "example_tube_line.json").read_text(encoding="utf-8"))
    doc["profiles"] = {"60": doc["profiles"]["long_tube"], "38": doc["profiles"]["short_tube"]}
    doc["parameters"]["default_profile"] = "60"
    (stations / "example_tube_line.json").write_text(json.dumps(doc), encoding="utf-8")
    httpd, app = _serve(tmp_path, stations)
    FakeDLP(app)
    c = Client(app.port)
    try:
        b = _good_cycle(c)
        assert b["profile"] == "60" and b["registers"]["result_code"] == 1
        status, r = c.post("/api/cmd/profile", {"profile": "38", "source": "quick"})
        assert status == 200 and r["bus"]["profile"] == "38"      # the NAME, not position 38
        _released(r["bus"])
        assert r["bus"]["registers"]["profile"] == 2
        # START as the page sends it (the profile by name)
        c.post("/api/io/inputs", {"part_present": True})
        status, r = c.post("/api/cmd/start", {"config": "example_tube_line.json", "profile": "38"})
        assert status == 200, r
        assert c.post("/api/cmd/abort", {})[0] == 200
        assert c.post("/api/cmd/profile", {"profile": 1})[1]["bus"]["profile"] == "60"
    finally:
        httpd.shutdown()
        httpd.server_close()
        app.close()
        api._session = None


def test_a_signal_no_check_was_named_for_is_refused_not_confirmed_by_presence(line):
    """`expects {door_closed: true}` with a recipe that only has a part-presence check: the
    configuration is refused (listed as unusable, START refused) - a door switch is never
    "confirmed" or "corrected" by looking at the seat."""
    c, app, fake = line
    f = app.stations_dir / "example_tube_line.json"
    flow = app.stations_dir.parent / "flows" / FLOW.name
    good = flow.read_text(encoding="utf-8")
    doc = json.loads(f.read_text(encoding="utf-8"))
    doc["parameters"]["io"]["inputs"]["door_closed"] = "the cell door switch"
    f.write_text(json.dumps(doc), encoding="utf-8")
    bad = json.loads(good)
    bad["steps"][0]["expects"] = {"part_present": True, "door_closed": True}
    flow.write_text(json.dumps(bad), encoding="utf-8")
    t = c.get("/api/stations")[1]["stations"][0]
    assert "expects 'door_closed' but no check observes it" in t["error"]
    status, r = c.post("/api/cmd/start", {})
    assert status == 409 and "not usable" in r["message"] and not r["bus"]["signals"]["PASS"]
    # named explicitly it is the author's statement (here: both read the seat's presence)
    bad["steps"][0]["verify"]["observe"]["door_closed"] = "seat/presence"
    flow.write_text(json.dumps(bad), encoding="utf-8")
    t = c.get("/api/stations")[1]["stations"][0]
    assert "error" not in t and t["flow"][0]["observe"] == {
        "part_present": "seat/presence", "door_closed": "seat/presence"}
    flow.write_text(good, encoding="utf-8")


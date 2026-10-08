"""Recipes through the full MOCK capture path: `scan run` with job.type recipe (a station
config whose work state IS a recipe), `scan recipe-*` on the command line, and the operator
page / command bus: the work state selects the recipe (which pins its preset), START runs it,
PASS / FAIL / ALARM + the result-code register go to the outputs, the verdict is painted on
the part by a separate job that the next command (HOME here) clears."""

import contextlib
import io
import json
import shutil
import threading
import time
from pathlib import Path

import pytest

from engine import api
from ui import cli
from ui import run_job as RJ
from ui.web import server as ui_server

import recipe_synth as RS
from conftest import ROOT
from test_ui_server import Client

EXAMPLE = ROOT / "stations" / "example_recipe_box_lid.json"


def _cli(argv) -> tuple:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = cli.main(argv)
    api._session = None
    return code, buf.getvalue()


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    """A mock library with the bench's TRUE stereo calibration at the recipes' geometry."""
    root = tmp_path_factory.mktemp("recipe_run") / "library"
    s = api.configure(root, mock=True)
    try:
        RS.truth_calibration(s, api.load_preset("bench_9in_metric"))
    finally:
        api._session = None
    return root


def _station(tmp_path, **over) -> Path:
    doc = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    doc.update({"runs_dir": str(tmp_path / "runs"), "mock_library": None})
    for k, v in over.items():
        doc[k] = {**doc[k], **v} if isinstance(v, dict) and isinstance(doc.get(k), dict) else v
    p = tmp_path / "recipe_station.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    return p


def test_station_config_recipe_keys(tmp_path):
    cfg = RJ.load_station_config(EXAMPLE)
    assert cfg.job_type == "recipe" and cfg.recipe == "box_lid" and cfg.preset is None
    assert cfg.mock_part == "scene:box_lid:ajar" and cfg.feedback is None
    with pytest.raises(RJ.StationConfigError, match="job.recipe"):
        RJ.load_station_config(_station(tmp_path, job={"recipe": None}))
    with pytest.raises(RJ.StationConfigError, match="job.feedback"):
        RJ.load_station_config(_station(tmp_path, job={"feedback": "yes"}))


def test_unknown_recipe_refused_without_a_run_folder(tmp_path, lib):
    code, out = _cli(["--mock", "--library", str(lib), "run",
                      str(_station(tmp_path, job={"recipe": "no_such_recipe"}))])
    assert code == cli.EXIT_REFUSED, out
    assert not (tmp_path / "runs").exists()


def test_run_job_recipe_fails_the_ajar_lid_and_projects_red(tmp_path, lib):
    code, out = _cli(["--mock", "--library", str(lib), "--json", "run",
                      str(_station(tmp_path))])
    doc = json.loads(out)
    (run_dir,) = [d for d in (tmp_path / "runs").iterdir() if d.is_dir()]
    res = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    assert doc["result"] == res and code == cli.EXIT_NOT_OK, res
    assert res["result"] == "FAIL" and res["job"]["type"] == "recipe"
    assert res["job"]["recipe"] == "box_lid" and res["station"] == "example_box_lid"
    assert res["capture"]["preset"] == "bench_9in_metric"            # the recipe pins it
    z = {z["id"]: z["verdict"] for z in res["zones"]}
    assert z == {"lid_left": "PASS", "lid_right": "FAIL"}
    assert res["recipe"]["result_code"] == 2 and res["feedback"]["rendered"]
    assert res["part"]["scan_id"] and res["part"]["label"] == "box"
    events = [json.loads(x) for x in (run_dir / "run.log.jsonl").read_text().splitlines()]
    names = [e["event"] for e in events]
    assert names[0] == "run_start" and names[-1] == "run_end"
    assert names.index("verdict") < names.index("feedback_shown") < names.index("run_end")
    assert (run_dir / "feedback_photo.png").is_file() and (run_dir / "overlay.png").is_file()

    # the same scan checked again offline from the command line: the same verdict, no capture
    code2, out2 = _cli(["--mock", "--library", str(lib), "--json", "recipe-run", "box_lid",
                        "--scan", res["part"]["scan_id"], "--no-feedback",
                        "--runs-dir", str(tmp_path / "again")])
    d2 = json.loads(out2)
    assert code2 == cli.EXIT_NOT_OK and d2["result"]["result"] == "FAIL", out2
    assert d2["result"]["part"]["scan_id"] == res["part"]["scan_id"]


def test_cli_recipe_library_commands(tmp_path, lib):
    L = ["--mock", "--library", str(tmp_path / "lib")]
    code, out = _cli([*L, "--json", "recipe-list"])
    names = {r["name"] for r in json.loads(out)["recipes"]}
    assert code == 0 and {"box_lid", "flat_plate", "cylinder_end_faces",
                          "ribbed_tube_alignment"} <= names
    code, out = _cli([*L, "recipe-add", "lid2", "--from", "box_lid",
                      "--set", "feedback.hold_s=9", "--desc", "line 2"])
    assert code == 0 and "lid2" in out, out
    code, out = _cli([*L, "--json", "recipe-show", "lid2"])
    assert code == 0 and json.loads(out)["recipe"]["recipe"]["feedback"]["hold_s"] == 9
    assert _cli([*L, "recipe-select", "lid2"])[0] == 0
    code, _ = _cli([*L, "recipe-delete", "lid2", "--reason", "x"])
    assert code == cli.EXIT_REFUSED                          # the selected recipe: refused
    assert _cli([*L, "recipe-select", "box_lid"])[0] == 0
    code, out = _cli([*L, "recipe-delete", "lid2", "--reason", "retired"])
    assert code == 0 and (tmp_path / "lib" / "trash" / "recipes" / "lid2").is_dir()
    bad = tmp_path / "bad.json"
    d = json.loads((ROOT / "recipes" / "box_lid.json").read_text())
    d["zones"][0]["checks"][0]["params"]["min_area_pct"] = 150
    bad.write_text(json.dumps(d))
    code, out = _cli([*L, "recipe-validate", str(bad)])
    assert code == cli.EXIT_REFUSED and "min_area_pct must be <= 100" in out
    code, out = _cli([*L, "recipe-checks"])
    assert code == 0 and "flatness" in out and "tol_mm" in out
    code, _ = _cli([*L, "--mock-part", "scene:nope", "recipe-run", "box_lid"])
    assert code == cli.EXIT_REFUSED


# ------------------------------------------------------------------ operator page + bus

@pytest.fixture(scope="module")
def quick(tmp_path_factory, lib):
    tmp = tmp_path_factory.mktemp("quick_recipe")
    stations = tmp / "stations"
    stations.mkdir()
    doc = json.loads(EXAMPLE.read_text())
    doc.update(runs_dir=str(tmp / "runs"), mock_library=None)
    (stations / "lid_check.json").write_text(json.dumps(doc))
    shutil.copy(ROOT / "stations" / "io_example.json", stations / "io_example.json")
    httpd, app = ui_server.make_server(mock=True, library=lib, port=0, stations_dir=stations)
    t = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
    t.start()
    try:
        yield Client(app.port), app
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


def _wait_state(c, want, timeout=120.0):
    t0 = time.monotonic()
    while True:
        _, b = c.get("/api/cmd/state")
        if b["state"] in want:
            return b
        assert time.monotonic() - t0 < timeout, b
        time.sleep(0.2)


def test_quick_lists_the_recipe_work_state(quick):
    c, app = quick
    _, d = c.get("/api/stations")
    (st,) = d["stations"]
    assert st["job_type"] == "recipe" and st["recipe"] == "box_lid"
    assert st["preset"] == "bench_9in_metric" and st["zones"] == ["lid_left", "lid_right"]
    assert set(st["checks"]) == {"presence", "height", "flatness"} and st["feedback"]
    _, b = c.get("/api/cmd/state")
    assert b["workstate"] == "lid_check.json" and b["workstate_recipe"] == "box_lid"
    status, r = c.post("/api/cmd/workstate", {"workstate": 1})
    assert status == 200 and "bench_9in_metric" in r["message"]
    _, d = c.get("/api/recipes")
    assert {"box_lid", "flat_plate"} <= {x["name"] for x in d["recipes"]}
    assert "flatness" in d["checks"]
    status, one = c.get("/api/recipes/box_lid")
    assert status == 200 and one["preset"] == "bench_9in_metric"
    assert c.get("/api/recipes/nope")[0] == 404
    js = (ROOT / "ui" / "web" / "static" / "quick.js").read_text(encoding="utf-8")
    assert "st.job_type === 'recipe'" in js and "showZones" in js


def test_start_runs_the_recipe_outputs_and_the_projected_verdict(quick):
    from engine.control.commands import SimulatedOutput
    c, app = quick
    out = SimulatedOutput()
    app.bus.outputs.append(out)
    status, r = c.post("/api/cmd/start", {"source": "test"})
    assert status == 200 and r["accepted"], r
    jid = next(k for k, v in r["jobs"].items() if v == "run")
    j = _job(c, jid)
    assert j["status"] == "done", j
    res = j["result"]
    assert res["result"] == "FAIL" and res["exit_code"] == 1 and res["job"]["type"] == "recipe"
    assert {z["id"]: z["verdict"] for z in res["zones"]} == {"lid_left": "PASS",
                                                             "lid_right": "FAIL"}
    b = _wait_state(c, {"READY"})
    assert b["signals"]["FAIL"] and not b["signals"]["PASS"]
    assert b["registers"]["result_code"] == 2 and b["registers"]["workstate"] == 1
    assert out.current["FAIL"] and out.registers["result_code"] == 2
    for k in ("result_json", "preview", "feedback"):
        assert c.request("GET", j["links"][k])[0] == 200, k
    # the verdict on the part: its own job, cleared by the next command (HOME)
    fj = res["feedback_job"]
    t0 = time.monotonic()
    while c.get(f"/api/jobs/{fj}")[1]["status"] == "queued":
        assert time.monotonic() - t0 < 60
        time.sleep(0.05)
    status, h = c.post("/api/cmd/home", {})
    assert status == 200 and h["accepted"], h
    f = _job(c, fj)
    assert (f["status"] == "error" and "Cancelled" in (f.get("error_type") or "")) or \
        (f["status"] == "done" and f["result"]["cancelled"]), f
    hj = next(k for k, v in h["jobs"].items() if v == "home")
    assert _job(c, hj)["status"] == "done"
    _wait_state(c, {"READY"})


def test_recipe_run_verb_offline_on_a_stored_scan(quick):
    c, app = quick
    scans = api.list_library("metric")
    j = c.job("recipe_run", {"recipe": "box_lid", "scan_id": scans[-1].id,
                             "project_feedback": False})
    assert j["status"] == "done", j
    assert j["result"]["result"] == "FAIL" and j["links"]["result_json"]
    j = c.job("recipe_run", {"recipe": "nope"})
    assert j["status"] == "error" and "no recipe" in j["error"]

"""Preset library (part recipes), MOCK only: built-in presets are read-only, user
presets live in the library and are added / selected / deleted (to trash/) through
api, the CLI and the web server; work states pin their own preset."""

import json
import threading
import time

import pytest

from engine import api, config
from engine.config import ConfigError
from engine.control.commands import CommandBus, Handlers
from ui import cli
from ui.web import server as ui_server

from conftest import FIXTURE_PRESETS, PRESET, ROOT
from test_ui_server import Client

BUILTIN = "cell_wall_2p7ft_vstripe_fullrange"


@pytest.fixture(autouse=True)
def _fresh_engine():
    yield
    api._session = None


def _builtin_bytes():
    return {p.name: p.read_bytes() for p in FIXTURE_PRESETS.glob("*.json")}


# ------------------------------------------------------------------ api

def test_add_from_builtin_overrides_and_loads(eng):
    before = _builtin_bytes()
    r = api.preset_add("part_a", BUILTIN, {"camera.exposure_us": "33333",
                                           "projection": {"duty": 0.8}},
                       description="box part, line 1")
    assert r["tier"] == "user" and r["note"] is None
    assert r["path"] == eng.store.root / "presets" / "part_a" / "preset.json"
    meta = json.loads((eng.store.root / "presets" / "part_a" / "meta.json").read_text())
    assert meta["from"] == BUILTIN and meta["description"] == "box part, line 1"
    assert meta["created"] and "author" in meta
    p = api.load_preset("part_a")
    assert p.camera.exposure_us == 33333 and p.projection.duty == 0.8
    assert p.rig_geometry_id == json.loads((FIXTURE_PRESETS / f"{BUILTIN}.json").read_text())[
        "geometry_id"]
    assert "part_a" in api.list_presets()
    assert _builtin_bytes() == before                     # built-ins never written


def test_new_geometry_starts_uncalibrated(eng):
    r = api.preset_add("rig_b", BUILTIN, geometry_id="rig_b_v1")
    assert "UNCALIBRATED" in r["note"]
    raw = json.loads(r["path"].read_text())
    assert raw["geometry_id"] == "rig_b_v1" and raw["calibration"]["mm_per_px"] is None
    assert not list((eng.store.root / "calibrations").glob("*rig_b_v1*"))
    item = next(i for i in api.preset_list() if i["name"] == "rig_b")
    assert item["calibrated"] is False and item["geometry_id"] == "rig_b_v1"


@pytest.mark.parametrize("ov, msg", [
    ({"camera.bogus": 1}, "unknown keys"),
    ({"projection.duty": 1.5}, "duty"),
    ({"camera.gain_db": 99}, "gain_db"),
    ({"name": "x"}, "set by the command"),
    ({"projection.rotate90": True}, "geometry_id"),       # geometry-bound change, same id
])
def test_override_validation_errors(eng, ov, msg):
    with pytest.raises(ConfigError, match=msg):
        api.preset_add("bad", BUILTIN, ov)
    assert "bad" not in api.list_presets()


def test_clash_refusals_and_replace(eng):
    with pytest.raises(ConfigError, match="built-in"):
        api.preset_add(BUILTIN, BUILTIN)
    api.preset_add("part_a", BUILTIN, {"camera.exposure_us": 33333})
    with pytest.raises(ConfigError, match="exists"):
        api.preset_add("part_a", BUILTIN)
    r = api.preset_add("part_a", BUILTIN, {"camera.exposure_us": 16667}, replace=True)
    assert r["replaced"] is not None and (eng.store.root / "trash" / "presets" / "part_a").is_dir()
    assert api.load_preset("part_a").camera.exposure_us == 16667
    with pytest.raises(ConfigError, match="no preset"):
        api.preset_add("x", "nosuch")


def test_select_sets_the_default(eng):
    assert api.preset_active() is None
    api.preset_add("part_a", BUILTIN)
    api.preset_select("part_a")
    assert api.preset_active() == "part_a"
    api.preset_select(PRESET)                             # append-only: last line wins
    assert api.preset_active() == PRESET
    assert len((eng.store.root / "presets" / "active.jsonl").read_text().splitlines()) == 2
    with pytest.raises(ConfigError):
        api.preset_select("nosuch")


def test_delete_moves_to_trash_and_refusals(eng, tmp_path):
    stations = tmp_path / "stations"
    stations.mkdir()
    for n in ("part_a", "part_b", "part_c"):
        api.preset_add(n, BUILTIN)
    (stations / "line1.json").write_text(json.dumps({"station_id": "l1", "preset": "part_b"}))
    (stations / "line2.yaml").write_text("station_id: l2\npreset: part_b\n")
    with pytest.raises(ConfigError, match="built-in"):
        api.preset_delete(BUILTIN, "x", stations_dir=stations)
    api.preset_select("part_a")
    with pytest.raises(ConfigError, match="active"):
        api.preset_delete("part_a", "x", stations_dir=stations)
    with pytest.raises(ConfigError, match="line1.json, line2.yaml"):
        api.preset_delete("part_b", "x", stations_dir=stations)
    with pytest.raises(ConfigError, match="reason"):
        api.preset_delete("part_c", " ", stations_dir=stations)
    r = api.preset_delete("part_c", "recipe retired", stations_dir=stations)
    t = eng.store.root / "trash" / "presets" / "part_c"
    assert r["trashed"] == t and (t / "preset.json").is_file()
    assert json.loads((t / "TRASHED.json").read_text())["reason"] == "recipe retired"
    assert "part_c" not in api.list_presets()
    r = api.preset_delete("part_b", "line changed", force=True, stations_dir=stations)
    assert r["forced"] and r["stations"] == ["line1.json", "line2.yaml"]


def test_list_flags_both_tiers(eng):
    api.preset_add("part_a", BUILTIN, description="d1")
    api.preset_select("part_a")
    items = {i["name"]: i for i in api.preset_list()}
    assert items[BUILTIN]["tier"] == "builtin" and items["part_a"]["tier"] == "user"
    assert items["part_a"]["active"] and not items[BUILTIN]["active"]
    assert items["part_a"]["description"] == "d1" and items["part_a"]["base"] == BUILTIN
    assert all("calibrated" in i and "geometry_id" in i for i in items.values())
    # a stereo calibration at the geometry flips the flag
    geo = config.MOCK_PREFIX + items["part_a"]["geometry_id"]
    d = eng.store.new_dir("calibrations", geo, label="stereo")
    (d / "stereo_calibration.json").write_text("{}")
    items = {i["name"]: i for i in api.preset_list()}
    assert items["part_a"]["calibrated"] and items[BUILTIN]["calibrated"]
    assert api.preset_show("part_a")["preset"]["name"] == "part_a"


# ------------------------------------------------------------------ CLI

def test_cli_add_select_show_delete(tmp_path, capsys):
    lib = str(tmp_path / "lib")
    run = lambda *a: (cli.main(["--mock", "--library", lib, *a]), capsys.readouterr())
    code, (out, err) = run("preset-add", "part_a", "--from", PRESET,
                           "--set", "camera.exposure_us=8000", "--desc", "part A")
    assert code == 0 and "part_a added" in out, err
    code, (out, err) = run("preset-add", "part_x", "--from", PRESET, "--set", "camera.nope=1")
    assert code == cli.EXIT_REFUSED and "unknown keys" in err
    code, (out, err) = run("preset-select", "part_a")
    assert code == 0
    code, (out, err) = run("--json", "presets")
    doc = json.loads(out)
    assert doc["active"] == "part_a" and doc["selected"] == "part_a"   # default follows selection
    row = next(p for p in doc["presets"] if p["name"] == "part_a")
    assert row["tier"] == "user" and row["camera"]["exposure_us"] == 8000
    code, (out, err) = run("--preset", PRESET, "--json", "presets")
    assert json.loads(out)["active"] == PRESET                         # --preset wins
    code, (out, err) = run("preset-show", "part_a")
    assert code == 0 and "user" in out and "part A" in out
    code, (out, err) = run("preset-delete", PRESET, "--reason", "x")
    assert code == cli.EXIT_REFUSED and "built-in" in err
    run("preset-select", PRESET)
    code, (out, err) = run("preset-delete", "part_a", "--reason", "done",
                           "--stations", str(tmp_path))
    assert code == 0 and "trash" in out


# ------------------------------------------------------------------ bus

def test_select_workstate_reports_its_preset():
    seen = []
    bus = CommandBus(Handlers(select=lambda ws, p: seen.append((ws, p))),
                     workstates=["a.json", "b.json"])
    bus.workstate_presets = {"a.json": "part_a", "b.json": None}
    r = bus.command("SELECT_WORKSTATE", arg="a.json")
    assert r.accepted and "part_a" in r.message and seen == [("a.json", "part_a")]
    assert bus.snapshot()["workstate_preset"] == "part_a"
    bus.state = "RUNNING"
    assert not bus.command("SELECT_WORKSTATE", arg="b.json").accepted
    assert seen == [("a.json", "part_a")]


# ------------------------------------------------------------------ web

@pytest.fixture
def web(tmp_path):
    presets, stations = tmp_path / "presets", tmp_path / "stations"
    presets.mkdir()
    stations.mkdir()
    for p in FIXTURE_PRESETS.glob("*.json"):
        (presets / p.name).write_bytes(p.read_bytes())
    doc = json.loads((ROOT / "stations" / "example_metric_check.json").read_text())
    doc.update(runs_dir=str(tmp_path / "runs"), mock_library=None, preset=BUILTIN)
    (stations / "line_a.json").write_text(json.dumps(doc))
    httpd, app = ui_server.make_server(mock=True, library=tmp_path / "library",
                                       presets_dir=presets, port=0, stations_dir=stations)
    t = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
    t.start()
    try:
        yield Client(app.port), app, presets
    finally:
        httpd.shutdown()
        httpd.server_close()
        app.close()
        api._session = None


def test_server_preset_endpoints(web):
    c, app, presets = web
    before = {p.name: p.read_bytes() for p in presets.glob("*.json")}
    j = c.done("preset_add", {"name": "part_a", "base": BUILTIN,
                              "overrides": {"camera.exposure_us": 33333},
                              "description": "line recipe"})
    assert j["result"]["name"] == "part_a"
    _, r = c.get("/api/presets")
    assert "part_a" in r["presets"] and "user_dir" in r
    tiers = {t["name"]: t for t in r["tiers"]}
    assert tiers["part_a"]["tier"] == "user" and tiers[BUILTIN]["tier"] == "builtin"
    _, d = c.get("/api/presets/part_a")
    assert d["tier"] == "user" and d["preset"]["camera"]["exposure_us"] == 33333
    j = c.done("preset_select", {"name": "part_a"})
    assert j["result"]["loaded"] == "part_a"
    _, r = c.get("/api/presets")
    assert r["selected"] == "part_a" and r["active"] == "part_a"
    j = c.job("preset_delete", {"name": "part_a", "reason": "x", "confirm": True})
    assert j["status"] == "error" and "loaded" in j["error"]
    c.done("preset_add", {"name": "part_b", "base": BUILTIN})
    j = c.job("preset_delete", {"name": "part_b", "reason": "x"})      # no confirm
    assert j["status"] == "error" and "confirm" in j["error"]
    j = c.job("preset_delete", {"name": BUILTIN, "reason": "x", "confirm": True})
    assert j["status"] == "error" and "built-in" in j["error"]
    c.done("preset_delete", {"name": "part_b", "reason": "retired", "confirm": True})
    assert "part_b" not in c.get("/api/presets")[1]["presets"]
    assert {p.name: p.read_bytes() for p in presets.glob("*.json")} == before
    # cross-origin POST is still refused
    status, _ = c.post("/api/jobs", {"verb": "preset_add", "args": {}},
                       headers={"Origin": "http://evil.example"})
    assert status == 403


def test_quick_page_shows_active_preset_and_workstate_selects_it(web):
    c, app, _ = web
    status, raw, _ = c.request("GET", "/quick")
    assert status == 200 and 'id="qPreset"' in raw.decode()
    _, st = c.get("/api/cmd/state")
    assert st["active_preset"] == PRESET
    c.get("/api/stations")
    status, r = c.post("/api/cmd/workstate", {"workstate": "line_a.json"})
    assert status == 200 and BUILTIN in r["message"]
    assert r["bus"]["workstate_preset"] == BUILTIN
    t0 = time.monotonic()
    while c.get("/api/cmd/state")[1]["active_preset"] != BUILTIN:
        assert time.monotonic() - t0 < 60
        time.sleep(0.1)
    app.bus.state = "RUNNING"                                    # preset changes refused
    try:
        status, r = c.post("/api/jobs", {"verb": "preset_select", "args": {"name": PRESET}})
        assert status == 409
    finally:
        app.bus.state = "READY"

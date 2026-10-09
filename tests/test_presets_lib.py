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


# ------------------------------------------------------------------ review regressions

@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity", "1e999", "nan", "inf"])
def test_non_finite_override_refused(eng, value):
    """A NaN / Infinity FAIL trigger never FAILs anything: refused at preset-add."""
    with pytest.raises(ConfigError, match="finite"):
        api.preset_add("tol_bad", BUILTIN, {"tolerance.ch_fail_mm": value})
    with pytest.raises(ConfigError, match="finite"):
        api.preset_add("tol_bad", BUILTIN, {"tolerance.ch_fail_mm": float(value)})
    assert "tol_bad" not in api.list_presets()


def test_non_finite_preset_file_refused():
    raw = json.loads((FIXTURE_PRESETS / f"{BUILTIN}.json").read_text())
    for key, val in (("ch_fail_mm", float("nan")), ("ch_fail_mm", float("inf")),
                     ("dent_fail_um", float("nan")), ("ch_fail_mm", True)):
        bad = {**raw, "tolerance": {**raw["tolerance"], key: val}}
        with pytest.raises(ConfigError, match=key):
            config.preset_from_dict(bad)
    bad = {**raw, "calibration": {**(raw.get("calibration") or {}), "mm_per_px": float("nan")}}
    with pytest.raises(ConfigError, match="mm_per_px"):
        config.preset_from_dict(bad)


def test_name_differing_from_a_builtin_only_by_case_refused(eng):
    with pytest.raises(ConfigError, match="only by case"):
        api.preset_add(BUILTIN.upper(), BUILTIN, {"camera.exposure_us": 8000})
    api.preset_add("Part_A", BUILTIN)
    with pytest.raises(ConfigError, match="only by case"):
        api.preset_add("part_a", BUILTIN)
    # names resolve exactly, on every platform: never the built-in file by case folding
    with pytest.raises(ConfigError, match="not found"):
        api.load_preset(BUILTIN.upper())
    assert api.load_preset("Part_A").name == "Part_A"


def test_a_builtin_added_later_never_silently_replaces_a_user_preset(tmp_path):
    """A repo update adds a built-in with EXACTLY the name of an existing user preset: the
    built-in used to win silently (loaded in the user preset's place, and the user one could
    not be deleted by name). Both are refused by name now, and preset-delete takes the user
    copy (built-ins are never deleted), after which the built-in loads."""
    from engine.capture.mock_bench import MockBench
    presets = tmp_path / "presets"
    presets.mkdir()
    for p in FIXTURE_PRESETS.glob("*.json"):
        (presets / p.name).write_bytes(p.read_bytes())
    api.configure(tmp_path / "library", mock=True, bench=MockBench(seed=1), presets_dir=presets)
    api.preset_add("part_x", BUILTIN, {"camera.exposure_us": 8000})
    raw = json.loads((presets / f"{BUILTIN}.json").read_text(encoding="utf-8"))
    (presets / "part_x.json").write_text(json.dumps({**raw, "name": "part_x"}), encoding="utf-8")
    with pytest.raises(ConfigError, match="both a built-in"):
        api.load_preset("part_x")
    with pytest.raises(ConfigError, match="both a built-in"):
        api.preset_select("part_x")
    row = [r for r in api.preset_list() if r["name"] == "part_x"]
    assert row and all(not r["valid"] and "both a built-in" in r["error"] for r in row)
    r = api.preset_delete("part_x", reason="shadowed by a new built-in")
    assert r["name"] == "part_x" and (presets / "part_x.json").is_file()   # built-in untouched
    assert api.load_preset("part_x").name == "part_x"
    with pytest.raises(ConfigError, match="read-only"):
        api.preset_delete("part_x", reason="x")


@pytest.mark.parametrize("name", ["CON", "nul", "COM1", "lpt9", "con.txt", "foo.", "...",
                                  "a\x00b", "x.json", "y.YAML", "bad\n", ".", ".."])
def test_unportable_or_file_like_names_refused(eng, name):
    """Windows strips trailing dots, CON / NUL / COMn are devices, a .json name reads as a
    file: each is refused up front (ConfigError, exit 2), never a half-made folder."""
    with pytest.raises(ConfigError):
        api.preset_add(name, BUILTIN)
    folder = eng.store.root / "presets"
    assert not folder.exists() or list(folder.iterdir()) == []


def test_cli_unportable_name_exit_2(tmp_path, capsys):
    lib = str(tmp_path / "lib")
    for name in ("CON", "foo.", "NUL", "..."):
        assert cli.main(["--mock", "--library", lib, "preset-add", name, "--from", PRESET]) == \
            cli.EXIT_REFUSED
        assert "unexpected" not in capsys.readouterr().err


def test_preset_names_never_resolve_against_the_cwd(eng, tmp_path, monkeypatch):
    """A user preset named like a folder (or a device) in the current folder loads the user
    preset, never that path."""
    api.preset_add("docs", BUILTIN, {"camera.exposure_us": 8000})
    (tmp_path / "docs").mkdir()
    (tmp_path / "other").write_text("{}")
    monkeypatch.chdir(tmp_path)
    assert api.load_preset("docs").camera.exposure_us == 8000
    with pytest.raises(ConfigError, match="not found"):
        api.load_preset("other")                    # a file in the CWD is not a preset name
    with pytest.raises(ConfigError):
        api.load_preset("CON")                      # never opens the console device
    p = tmp_path / "mine.json"                      # an explicit .json path is still a file
    p.write_text((FIXTURE_PRESETS / f"{BUILTIN}.json").read_text())
    assert config.load_preset(str(p), mock=True).name == BUILTIN


def test_replace_keeps_the_geometry_reuse_rule(eng):
    """--replace with a geometry-bound change under the same geometry_id is refused (as a
    new name is): the old calibration would be reused at another geometry."""
    api.preset_add("p", BUILTIN, geometry_id="g1")
    api.preset_add("p", "p", {"calibration.mm_per_px": "0.05"}, replace=True)
    with pytest.raises(ConfigError, match="geometry_id g1"):
        api.preset_add("p2", "p", {"projection.rotate90": "true"})
    with pytest.raises(ConfigError, match="geometry_id g1"):
        api.preset_add("p", "p", {"projection.rotate90": "true"}, replace=True)
    raw = json.loads((eng.store.root / "presets" / "p" / "preset.json").read_text())
    assert raw["projection"]["rotate90"] is False and raw["calibration"]["mm_per_px"] == 0.05
    # with a new geometry_id the change is fine, and the calibration starts empty
    r = api.preset_add("p", "p", {"projection.rotate90": "true"}, geometry_id="g2",
                       replace=True)
    raw = json.loads(r["path"].read_text())
    assert raw["geometry_id"] == "g2" and raw["calibration"]["mm_per_px"] is None


def test_failed_replace_keeps_the_old_preset(eng, monkeypatch):
    """The replacement is written before the old one is trashed; a failure puts it back."""
    from engine.library.store import LibraryStore
    api.preset_add("line_a", BUILTIN, {"camera.exposure_us": 33333})
    api.preset_select("line_a")

    def full(path, obj):
        raise OSError(28, "No space left on device")
    monkeypatch.setattr(LibraryStore, "write_json", staticmethod(full))
    with pytest.raises(OSError):
        api.preset_add("line_a", BUILTIN, {"camera.exposure_us": 16667}, replace=True)
    monkeypatch.undo()
    assert api.preset_active() == "line_a"
    assert api.load_preset("line_a").camera.exposure_us == 33333
    assert [p.name for p in (eng.store.root / "presets").iterdir() if p.name[0] == "~"] == []


def test_delete_refuses_a_preset_pinned_by_recipes(eng, tmp_path):
    stations = tmp_path / "stations"
    stations.mkdir()
    api.preset_add("pa", BUILTIN)
    api.preset_add("pb", BUILTIN)
    api.recipe_add("ra", "flat_plate", {"preset": "pa"})
    api.recipe_add("rb", "flat_plate", {"preset": "pb"})
    api.recipe_select("ra")
    (stations / "line_b.json").write_text(json.dumps(
        {"station_id": "lb", "job": {"type": "recipe", "recipe": "rb"}}))
    with pytest.raises(ConfigError, match="selected recipe 'ra'"):
        api.preset_delete("pa", "x", stations_dir=stations)
    with pytest.raises(ConfigError, match="selected recipe"):
        api.preset_delete("pa", "x", force=True, stations_dir=stations)
    with pytest.raises(ConfigError, match="rb.*line_b.json"):
        api.preset_delete("pb", "x", stations_dir=stations)
    assert api.recipe_validate("ra")["valid"] and api.recipe_validate("rb")["valid"]
    r = api.preset_delete("pb", "x", force=True, stations_dir=stations)
    assert r["forced"] and r["recipes"] == ["rb"] and r["recipe_stations"] == ["line_b.json"]


def test_delete_sees_a_recipe_file_named_with_an_env_var(eng, tmp_path, monkeypatch):
    """A station config's job.recipe '$VAR/rf.json' is expanded by the run job (run_job): the
    delete check resolved it literally, missed the pin and deleted the preset the station
    runs."""
    stations, elsewhere = tmp_path / "st", tmp_path / "elsewhere"
    stations.mkdir()
    elsewhere.mkdir()
    api.preset_add("pa", BUILTIN)
    doc = json.loads((ROOT / "recipes" / "flat_plate.json").read_text(encoding="utf-8"))
    doc.update(name="rf", preset="pa")
    (elsewhere / "rf.json").write_text(json.dumps(doc), encoding="utf-8")
    monkeypatch.setenv("SAS_TEST_RECIPES", str(elsewhere))
    (stations / "line.json").write_text(json.dumps(
        {"station_id": "line", "job": {"type": "recipe", "recipe": "$SAS_TEST_RECIPES/rf.json"}}))
    with pytest.raises(ConfigError, match="line.json"):
        api.preset_delete("pa", "x", stations_dir=stations)


def test_unreadable_selection_lines_are_skipped(tmp_path, capsys):
    """A non-UTF-8 byte (or a UTF-16 BOM) in active.jsonl never stops the CLI: the line is
    skipped with a warning, and preset-select can still append a good one."""
    lib = tmp_path / "lib"
    run = lambda *a: (cli.main(["--mock", "--library", str(lib), *a]), capsys.readouterr())
    assert run("preset-select", BUILTIN)[0] == 0
    with open(lib / "presets" / "active.jsonl", "ab") as fh:
        fh.write(b'{"name": "Jos\xe9"}\n\xff\xfe{\x00}\x00')
    code, (out, err) = run("--json", "presets")
    assert code == 0 and json.loads(out)["selected"] == BUILTIN, err
    code, (out, err) = run("presets")
    assert code == 0 and "unreadable" in out
    assert run("paths")[0] == 0
    assert run("preset-select", PRESET)[0] == 0
    code, (out, err) = run("--json", "presets")
    assert json.loads(out)["selected"] == PRESET


def test_selected_preset_gone_is_a_clear_error_not_the_default(tmp_path, capsys):
    import shutil
    lib = tmp_path / "lib"
    run = lambda *a: (cli.main(["--mock", "--average", "1", "--library", str(lib), *a]),
                      capsys.readouterr())
    assert run("preset-add", "line_a", "--from", PRESET)[0] == 0
    assert run("preset-select", "line_a")[0] == 0
    shutil.rmtree(lib / "presets" / "line_a")               # gone behind the library's back
    code, (out, err) = run("signal")
    assert code == cli.EXIT_REFUSED and "line_a" in err and "no longer exists" in err
    code, (out, err) = run("--json", "presets")             # the listing works, and says so
    doc = json.loads(out)
    assert code == 0 and doc["selected_missing"] == "line_a" and doc["selected"] is None
    assert run("--preset", PRESET, "signal")[0] == 0        # an explicit preset still works
    assert run("preset-select", PRESET)[0] == 0             # and the operator can repair it
    assert run("signal")[0] == 0


@pytest.mark.parametrize("spelling", [["--pres", BUILTIN], ["--pre=" + BUILTIN],
                                      ["--prese", BUILTIN]])
def test_abbreviated_preset_is_refused_not_ignored(tmp_path, capsys, spelling):
    """argparse would take --pres for --preset while main() only sees --preset: refused."""
    lib = str(tmp_path / "lib")
    assert cli.main(["--mock", "--library", lib, "preset-add", "part_a", "--from", PRESET]) == 0
    assert cli.main(["--mock", "--library", lib, "preset-select", "part_a"]) == 0
    capsys.readouterr()
    assert cli.main(["--mock", "--library", lib, *spelling, "--json", "presets"]) == \
        cli.EXIT_REFUSED
    assert cli.main(["--mock", "--library", lib, "presets", *spelling]) == cli.EXIT_REFUSED
    assert cli.main(["--mock", "--library", lib, "--mock-p", "none", "presets"]) == \
        cli.EXIT_REFUSED
    capsys.readouterr()
    assert cli.main(["--mock", "--library", lib, "--preset", BUILTIN, "--json", "presets"]) == 0
    assert json.loads(capsys.readouterr().out)["active"] == BUILTIN

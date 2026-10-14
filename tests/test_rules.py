"""§9 safety rules — enforced, not advisory."""

import ast
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from engine import api, rules
from engine.capture import riglock
from engine.capture.projector import MockWindow, Projector
from engine.config import ProjectionCfg
from engine.library.store import LibraryStore

import conftest
from conftest import ROOT, VALIDATION_OBJECT

ENGINE = ROOT / "engine"


def _imports(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield a.name
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.module


# ------------------------------------------------------------ rule 1: HDMI

# The only two engine files that may load a USB module, and nothing else:
USB_TOOL = Path("engine/capture/dmd_control.py")    # the standalone diagnostics tool
SEQ_HELPER = Path("engine/seqctl/helper.py")        # the sequence-control helper process
HELPER_MODULE = "engine.seqctl.helper"
IMPORT_CALLS = ("import_module", "__import__", "run_module")


def _usb_import_offenders(root: Path) -> list:
    """Engine files under `root` that import the seed package or hidapi and may
    not. dmd_control.py may import both. The sequence-control helper may import
    hid only: it carries its own framing, never the seed driver (which parks the
    mirrors when it closes)."""
    offenders = []
    for py in (root / "engine").rglob("*.py"):
        rel = py.relative_to(root)
        if rel == USB_TOOL:
            continue
        for mod in _imports(py):
            if mod.startswith("swiss_army_sensor") or (mod == "hid" and rel != SEQ_HELPER):
                offenders.append(f"{rel.as_posix()}: {mod}")
    return offenders


def _named(py: Path, rel: Path):
    """("import", module) for every module a file imports, relative imports made
    absolute ('from . import x' names the package and package.x); ("call", text)
    for a string handed to an import call; ("text", text) for every string."""
    for node in ast.walk(ast.parse(py.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield "import", a.name
        elif isinstance(node, ast.ImportFrom):
            pkg = list(rel.with_suffix("").parts[:-1]) if node.level else []
            pkg = pkg[:max(0, len(pkg) - node.level + 1)]
            mod = ".".join(pkg + ([node.module] if node.module else []))
            yield "import", mod
            for a in node.names:
                yield "import", f"{mod}.{a.name}"
        elif isinstance(node, ast.Call) and getattr(
                node.func, "attr", getattr(node.func, "id", None)) in IMPORT_CALLS:
            for a in node.args:
                if isinstance(a, ast.Constant) and isinstance(a.value, str):
                    yield "call", a.value
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield "text", node.value


def _names_the_tool(kind: str, name: str) -> bool:
    if kind == "text":                  # importlib.import_module("...dmd_control")
        return name.endswith("dmd_control") and "." in name
    return name.split(".")[-1] == "dmd_control"


def _imports_the_helper(kind: str, name: str) -> bool:
    if kind == "import":
        return name == HELPER_MODULE or name.startswith(HELPER_MODULE + ".")
    return kind == "call" and name.split(".")[-1] == "helper" and (
        "seqctl" in name or name.startswith("."))


def _usb_module_importers(root: Path) -> list:
    """Files under `root` that import one of the two USB modules. Both run as
    their own process only. dmd_control: never imported by the engine, nor named
    in a string there. The helper: never imported by the engine or the UI; its
    name for `python -m` is how it is started and is allowed, an import call
    with that name is not. Neither may import the other."""
    offenders = set()
    for top in ("engine", "ui"):
        for py in (root / top).rglob("*.py"):
            rel = py.relative_to(root)
            for kind, name in _named(py, rel):
                if (top == "engine" and rel != USB_TOOL and _names_the_tool(kind, name)) \
                        or (rel != SEQ_HELPER and _imports_the_helper(kind, name)):
                    offenders.add(rel.as_posix())
    return sorted(offenders)


def test_engine_never_imports_dlp_usb_driver():
    """No engine module imports swiss_army_sensor (whose __init__ pulls in
    the DLPC350 driver) or hidapi. Seed modules come via engine/_seed.py.
    Only the standalone diagnostics tool dmd_control.py may, and the
    sequence-control helper engine/seqctl/helper.py may import hid."""
    offenders = _usb_import_offenders(ROOT)
    assert not offenders, offenders


def test_nothing_imports_the_usb_tool():
    """dmd_control and the sequence-control helper are run as their own
    process, never imported by the engine (the helper: nor by the UI)."""
    offenders = _usb_module_importers(ROOT)
    assert not offenders, offenders


def _tree(root: Path, files: dict) -> Path:
    for rel, text in files.items():
        f = root / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text, encoding="utf-8")
    return root


# a made-up source tree holding what is allowed: the two USB modules, and the
# helper started by name as a child process
ALLOWED_TREE = {
    "engine/capture/dmd_control.py": "import hid\nfrom swiss_army_sensor.drivers import x\n",
    "engine/seqctl/helper.py": "def open_device():\n    import hid\n    return hid\n",
    "engine/seqctl/client.py":
        "import sys\nARGS = [sys.executable, '-m', 'engine.seqctl.helper']\n",
    "ui/web/projector.py": "from engine.seqctl import client\n",
}


def test_usb_checks_accept_the_two_usb_modules(tmp_path):
    root = _tree(tmp_path, ALLOWED_TREE)
    assert _usb_import_offenders(root) == [] and _usb_module_importers(root) == []


@pytest.mark.parametrize("rel,text,offender", [
    ("engine/seqctl/link.py", "import hid\n", "engine/seqctl/link.py: hid"),
    ("engine/capture/other.py", "def f():\n    import hid\n", "engine/capture/other.py: hid"),
    ("engine/seqctl/helper.py", "import hid\nfrom swiss_army_sensor.drivers import dlp_driver\n",
     "engine/seqctl/helper.py: swiss_army_sensor.drivers"),
])
def test_usb_checks_refuse_any_other_usb_import(tmp_path, rel, text, offender):
    root = _tree(tmp_path, {**ALLOWED_TREE, rel: text})
    assert _usb_import_offenders(root) == [offender]


@pytest.mark.parametrize("rel,text", [
    ("engine/seqctl/client.py", "from . import helper\n"),
    ("engine/seqctl/client.py", "from .helper import main\n"),
    ("engine/seqctl/__init__.py", "from .helper import main\n"),
    ("engine/capture/other.py", "from ..seqctl import helper\n"),
    ("engine/api.py", "def f():\n    import engine.seqctl.helper as h\n"),
    ("engine/api.py", "import importlib\nh = importlib.import_module('engine.seqctl.helper')\n"),
    ("engine/api.py", "import runpy\nrunpy.run_module('engine.seqctl.helper')\n"),
    ("ui/web/server.py", "from engine.seqctl import helper\n"),
    ("engine/capture/other.py", "from . import dmd_control\n"),
    ("engine/api.py", "TOOL = 'engine.capture.dmd_control'\n"),
    ("engine/seqctl/helper.py", "from ..capture import dmd_control\n"),
    ("engine/capture/dmd_control.py", "from engine.seqctl.helper import main\n"),
])
def test_usb_checks_refuse_an_import_of_a_usb_module(tmp_path, rel, text):
    root = _tree(tmp_path, {**ALLOWED_TREE, rel: text})
    assert _usb_module_importers(root) == [rel]


def test_projector_stops_if_usb_driver_loaded_mid_scan(monkeypatch):
    proj = Projector(ProjectionCfg(), mock=True)
    proj.project(np.zeros((1140, 912), np.uint8))
    monkeypatch.setitem(sys.modules, "swiss_army_sensor.drivers.dlp_driver", object())
    with pytest.raises(rules.SafetyViolation):
        proj.project(np.zeros((1140, 912), np.uint8))


def test_seed_loader_excludes_usb_driver():
    from engine import _seed
    assert not any("dlp" in p for p in _seed.SEED_MODULES.values())


def test_full_mock_run_never_loads_usb_driver(tmp_path):
    """End to end in a clean interpreter, then inspect sys.modules."""
    code = f"""
import sys; sys.path.insert(0, {str(ROOT)!r})
from engine import api
from engine.capture.mock_bench import VALIDATION_OBJECT
api.configure({str(tmp_path)!r}, mock=True)
p = api.load_preset("cell_wall_2p7ft_vstripe")
api.check_signal()
b = api.capture_base(p, "empty")
api.mock_bench().place(VALIDATION_OBJECT)
o = api.capture_object(p, b, "obj", known_dims={{"H": 106.95}})
api.calibrate(p, [o]); api.measure(o)
bad = [m for m in ("swiss_army_sensor.drivers.dlp_driver", "swiss_army_sensor.drivers",
                   "swiss_army_sensor", "hid") if m in sys.modules]
print("BAD", bad)
"""
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         timeout=300)
    assert out.returncode == 0, out.stderr
    assert "BAD []" in out.stdout, out.stdout


def test_runtime_guard_trips_if_usb_driver_loaded():
    rules.check_scan_path_usb_free({})
    with pytest.raises(rules.SafetyViolation, match="HDMI-only"):
        rules.check_scan_path_usb_free({"swiss_army_sensor.drivers.dlp_driver": object()})
    with pytest.raises(rules.SafetyViolation):
        rules.check_scan_path_usb_free({"hid": object()})


def test_projector_refuses_usb_display_commands():
    proj = Projector(ProjectionCfg(), mock=True)
    for cmd in ("set_mode", "set_display_mode", "set_pattern_source", "load_pattern_sequence"):
        with pytest.raises(rules.SafetyViolation, match="HDMI-only"):
            getattr(proj, cmd)("video")
    with pytest.raises(AttributeError):
        proj.not_a_thing


def test_dlp_driver_rejected_as_transport():
    class MockDLPDriver:            # same shape as swiss_army_sensor's
        def show(self, p): ...
        def hold(self, s): ...
    with pytest.raises(rules.SafetyViolation, match="DLPC350"):
        rules.check_transport(MockDLPDriver())

    class Sneaky(MockWindow):
        def set_mode(self, m): ...
    with pytest.raises(rules.SafetyViolation, match="set_mode"):
        rules.check_transport(Sneaky())


def test_projection_chain_matches_v10():
    """gamma=1, duty 0.55, inverted — v10's _project, pushed twice."""
    cfg = ProjectionCfg(duty=0.55, invert=True, push_twice=True, min_hold_s=0.35)
    proj = Projector(cfg, mock=True)
    pat = np.full((1140, 912), 255, np.uint8)
    proj.project(pat)
    sent = proj.window.last
    assert sent[0, 0] == 255 - int(255 * 0.55)
    assert proj.window.physical[0, 0] == 255 - int(255 * 0.55)   # bench DMD shows it as sent
    assert proj.window.count == 2
    assert proj.window.held_s == pytest.approx(0.35)


# ------------------------------------------------ rule 2: pumped holds

def test_projector_has_no_bare_sleeps():
    tree = ast.parse((ENGINE / "capture" / "projector.py").read_text(encoding="utf-8"))
    sleeps = [n.lineno for n in ast.walk(tree)
              if isinstance(n, ast.Call) and (
                  (isinstance(n.func, ast.Attribute) and n.func.attr == "sleep") or
                  (isinstance(n.func, ast.Name) and n.func.id == "sleep"))]
    assert not sleeps, f"bare sleep in projector.py at lines {sleeps}"


# ------------------------------------------- rules 5/6: geometry binding

def test_capture_object_refuses_base_from_other_geometry(eng, preset):
    base = api.capture_base(preset, "empty")
    base.geometry_id = "some_other_rig"
    with pytest.raises(rules.GeometryMismatch):
        api.capture_object(preset, base, "obj")


def test_real_preset_refused_in_mock_engine(eng):
    from engine import config
    real = config.load_preset("cell_wall_2p7ft")
    with pytest.raises(rules.RuleViolation, match="mock"):
        api.capture_base(real, "empty")


def test_calibrate_refuses_samples_from_other_geometry(eng, preset):
    base = api.capture_base(preset, "empty")
    api.mock_bench().place(VALIDATION_OBJECT)
    obj = api.capture_object(preset, base, "obj", known_dims={"H": 106.95})
    obj.geometry_id = "elsewhere"
    with pytest.raises(rules.GeometryMismatch):
        api.calibrate(preset, [obj])


# --------------------------------------------------- rule 7: no overwrite

def test_store_never_overwrites(tmp_path):
    s = LibraryStore(tmp_path)
    d = s.new_dir("objects", label="x")
    s.write_json(d / "meta.json", {"a": 1})
    with pytest.raises(rules.OverwriteRefused):
        s.write_json(d / "meta.json", {"a": 2})
    assert (d / "meta.json").read_text().count('"a": 1') == 1
    dirs = {s.new_dir("objects", label="x") for _ in range(20)}
    assert len(dirs) == 20


def test_repeat_captures_append(eng, preset):
    a = api.capture_base(preset, "empty")
    b = api.capture_base(preset, "empty")
    assert a.path != b.path and a.path.exists() and b.path.exists()


# ---------------------------------------- rule 3 over time (review fix)

def test_burst_led_currents_only_for_short_projections():
    burst = {"mode": "burst", "r": 180, "g": 200, "b": 150}
    rules.validate_led(burst)                                   # a valid burst setting
    rules.check_led_duration(burst, 5.0, "check_signal")        # inside the burst window
    for secs in (None, rules.LED_BURST_WINDOW_S + 1, 1800.0):   # scan / long hold / live view
        with pytest.raises(rules.SafetyViolation, match="R=180.*G=200"):
            rules.check_led_duration(burst, secs, "live view")
    rules.check_led_duration({"mode": "burst", "r": 120, "g": 180}, None, "scan")   # avg-safe
    rules.check_led_duration({"mode": "continuous", "r": 120}, None, "scan")
    rules.check_led_duration(None, None, "scan")                # unrecorded: config warns


def test_full_scan_refuses_burst_currents(eng, preset):
    preset.projection.led = {"mode": "burst", "r": 180}
    with pytest.raises(rules.SafetyViolation, match="full scan"):
        api.capture_base(preset, "hot")
    assert api.list_library("bases") == []


def test_fingerprint_diffs():
    a = {"rotate90": False, "size": [912, 1140], "basis_axis": "x", "baseline_mm": 279.4}
    assert rules.fingerprint_diffs(a, dict(a)) == []
    assert rules.fingerprint_diffs(a, {**a, "size": (912, 1140)}) == []      # list == tuple
    assert rules.fingerprint_diffs(None, a) == [] and rules.fingerprint_diffs(a, {}) == []
    assert rules.fingerprint_diffs(a, {**a, "baseline_mm": None}) == []      # unknown
    assert rules.fingerprint_diffs(a, {**a, "rotate90": True}) == ["rotate90 False -> True"]
    with pytest.raises(rules.GeometryMismatch, match="basis_axis"):
        rules.require_same_fingerprint(a, {**a, "basis_axis": "y"}, "capture_object")


# ------------------------------------- the test harness (tests/conftest.py)

def test_tests_start_without_hardware_and_without_machine_files():
    """Every test process: no real device, a projector rig file path that is not
    the machine's and does not exist (control off), the event log off."""
    assert os.environ.get(conftest.NO_HARDWARE_ENV) == "1", \
        "the tests run with SAS_NO_HARDWARE=1 only"
    path = Path(os.environ[conftest.RIG_FILE_ENV])
    assert path.name == conftest.RIG_FILE_NAME and not path.exists()
    beside_lock = riglock.lock_candidates()[0].with_name(conftest.RIG_FILE_NAME)
    assert conftest._forms(beside_lock) <= conftest._Guard.machine
    assert not conftest._forms(path) & conftest._Guard.machine
    assert os.environ[conftest.EVENTLOG_ENV] == "off"


def test_machine_rig_file_is_refused_and_noted(tmp_path, monkeypatch, rig_file):
    """The guard, shown on a stand-in for the machine's file (the real one is
    never addressed): every way of opening it is refused and noted, so the test
    fails even when the code under test swallows the refusal."""
    machine = tmp_path / "machine" / conftest.RIG_FILE_NAME
    monkeypatch.setattr(conftest._Guard, "machine", conftest._forms(machine))
    for attempt in (lambda: open(machine, encoding="utf-8"),
                    lambda: machine.read_text(encoding="utf-8"),
                    lambda: os.open(str(machine), os.O_RDONLY)):
        with pytest.raises(conftest.IsolationError, match="rig file"):
            attempt()
    assert len(conftest.take_isolation_notes()) == 3
    # the test's own rig file is the one the variable names, and the test's to write
    assert Path(os.environ[conftest.RIG_FILE_ENV]) == rig_file and not rig_file.exists()
    rig_file.write_text('{"control": "off"}', encoding="utf-8")
    assert rig_file.read_text(encoding="utf-8")
    assert conftest.take_isolation_notes() == []


def test_event_log_files_are_refused_without_the_fixture(tmp_path):
    events = tmp_path / "library" / "events"
    (events / "pending").mkdir(parents=True)
    rows = events / "inspections_v1_2026-01.csv"
    for attempt in (lambda: open(rows, "a", encoding="utf-8"),
                    lambda: rows.write_text("time\n", encoding="utf-8"),
                    lambda: os.replace(tmp_path / "rows.tmp", rows),
                    lambda: open(events / "inspection_values_v1_2026-01.csv", "w"),
                    lambda: open(events / "pending" / "row.json", "w"),
                    lambda: open(events / "img" / "a.png", "wb"),
                    lambda: open(events / "last_v1.csv", "w"),
                    lambda: open(events / "eventlog.lock", "a+b"),
                    lambda: open(events / "pending.lock", "a+b"),
                    lambda: open(events / "rebuild_20260101T000000Z" / "rows.csv", "w"),
                    lambda: os.replace(rows, events / "replaced_20260101T000000Z" / rows.name)):
        with pytest.raises(conftest.IsolationError, match="eventlog_on"):
            attempt()
    assert len(conftest.take_isolation_notes()) == 11
    assert [p for p in events.rglob("*") if p.is_file()] == []
    with pytest.raises(FileNotFoundError):                   # reading is never refused
        open(rows, encoding="utf-8")
    (tmp_path / "library" / "events.txt").write_text("not an event-log file")
    assert conftest.take_isolation_notes() == []


def test_event_log_fixture_allows_rows_in_a_temporary_folder_only(tmp_path, monkeypatch,
                                                                  eventlog_on):
    assert os.environ[conftest.EVENTLOG_ENV] == "on"
    rows = tmp_path / "library" / "events" / "inspection_values_v1_2026-01.csv"
    rows.parent.mkdir(parents=True)
    with open(rows, "a", encoding="utf-8") as fh:
        fh.write("time\n")
    os.replace(rows, rows.with_name("inspections_v1_2026-01.csv"))
    assert conftest.take_isolation_notes() == []
    # the same write where no temporary folder is (shown by narrowing what counts as one)
    monkeypatch.setattr(conftest._Guard, "temp",
                        {f + os.sep for f in conftest._forms(tmp_path / "temp")})
    with pytest.raises(conftest.IsolationError, match="temporary folder"):
        open(rows, "a", encoding="utf-8")
    assert len(conftest.take_isolation_notes()) == 1


def test_the_zone_selection_is_written_in_a_temporary_folder_only(tmp_path, monkeypatch):
    """The stored zone selection and zone-show record of a rig placement (engine/zones)
    decide what a run lights: a test writes them in its own library under tmp_path only."""
    geo = tmp_path / "library" / "zones" / "geo_1"
    geo.mkdir(parents=True)
    for name in ("_selection.jsonl", "_shown.jsonl"):
        with open(geo / name, "a", encoding="utf-8") as fh:     # in a temporary folder: fine
            fh.write("{}\n")
    assert conftest.take_isolation_notes() == []
    monkeypatch.setattr(conftest._Guard, "temp",
                        {f + os.sep for f in conftest._forms(tmp_path / "temp")})
    for attempt in (lambda: open(geo / "_selection.jsonl", "a", encoding="utf-8"),
                    lambda: open(geo / "_shown.jsonl", "ab"),
                    lambda: os.replace(tmp_path / "sel.tmp", geo / "_selection.jsonl")):
        with pytest.raises(conftest.IsolationError, match="zone selection"):
            attempt()
    assert len(conftest.take_isolation_notes()) == 3
    with open(geo / "_selection.jsonl", encoding="utf-8") as fh:  # reading: never refused
        assert fh.read() == "{}\n"
    open(geo.parent / "_selection.jsonl", "w").close()             # not a rig placement's file
    assert conftest.take_isolation_notes() == []

"""§9 safety rules — enforced, not advisory."""

import ast
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from engine import api, rules
from engine.capture.projector import MockWindow, Projector
from engine.config import ProjectionCfg
from engine.library.store import LibraryStore

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

USB_TOOL = ENGINE / "capture" / "dmd_control.py"   # the one sanctioned USB module


def test_engine_never_imports_dlp_usb_driver():
    """No engine module imports swiss_army_sensor (whose __init__ pulls in
    the DLPC350 driver) or hidapi. Seed modules come via engine/_seed.py.
    Only the standalone diagnostics tool dmd_control.py may."""
    offenders = []
    for py in ENGINE.rglob("*.py"):
        if py == USB_TOOL:
            continue
        for mod in _imports(py):
            if mod == "hid" or mod.startswith("swiss_army_sensor"):
                offenders.append(f"{py.relative_to(ROOT)}: {mod}")
    assert not offenders, offenders


def test_nothing_imports_the_usb_tool():
    """dmd_control is run as its own process, never imported by the engine."""
    offenders = []
    for py in ENGINE.rglob("*.py"):
        if py == USB_TOOL:
            continue
        for node in ast.walk(ast.parse(py.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.ImportFrom):
                names = [node.module or ""] + [a.name for a in node.names]
            elif isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.Constant) and isinstance(node.value, str) \
                    and node.value.endswith("dmd_control") and "." in node.value:
                names = [node.value]            # importlib.import_module("...dmd_control")
            if any(n.split(".")[-1] == "dmd_control" for n in names):
                offenders.append(str(py.relative_to(ROOT)))
    assert not offenders, offenders


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

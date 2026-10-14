"""Every path that could write the projector's display register, closed (mock only).

- The seed driver (swiss_army_sensor/drivers/dlp_driver.py) checks every command on
  the BUILT packet before the first byte goes to a real device: a read passes, a write
  passes only inside `driver.writes_allowed(reason)`, and a write of Display Mode
  Select never passes.
- SensorRig.create(mock=False, usb=True) is refused before anything is opened.
- A search of every text file of the repo for the display command by value fails on a
  hit outside the allow-list below.

The "real device" here is the real driver class with a fake object in place of hidapi:
nothing in this file imports hid or opens a device. The seed package loads the USB
driver module, which (by design) blocks the scan path in the same process, so it is
imported inside a fixture and removed from sys.modules afterwards.
"""

import ast
import importlib
import inspect
import re
import sys
import types

import numpy as np
import pytest

from engine import rules
from engine.capture.projector import Projector
from engine.config import ProjectionCfg

from conftest import ROOT

DISPLAY = (0x1A, 0x1B)              # Display Mode Select as (CMD2, CMD3)


@pytest.fixture
def seed():
    before = set(sys.modules)
    drv = importlib.import_module("swiss_army_sensor.drivers.dlp_driver")
    core = importlib.import_module("swiss_army_sensor.core")
    yield types.SimpleNamespace(drv=drv, core=core)
    for k in set(sys.modules) - before:
        del sys.modules[k]


class FakeHID:
    """Stands in for the opened USB device: keeps what is written to it."""

    def __init__(self):
        self.written, self.closed = [], False

    def write(self, pkt):
        self.written.append(bytes(pkt))
        return len(pkt)

    def read(self, n, timeout_ms=0):
        return [0x40, 0, 1, 0, 0] + [0] * 59         # one data byte: 0x00

    def close(self):
        self.closed = True

    @property
    def writes(self):
        """The commands of the write packets on the device (read bit clear)."""
        return [(p[6], p[5]) for p in self.written if not p[1] & 0x80]


def _real(seed):
    """The real driver class on a fake device, as open() would leave it."""
    d = seed.drv.DLPDriver()
    d._hid = FakeHID()
    return d, d._hid


@pytest.fixture
def no_hidapi(monkeypatch):
    """`import hid` gives a stand-in that fails the test when a device is asked for."""
    fake = types.ModuleType("hid")

    def device():
        raise AssertionError("a test asked hidapi for a device")
    fake.device = device
    monkeypatch.setitem(sys.modules, "hid", fake)


# one call of every public method of the driver
CALLS = {
    "open": (), "close": (), "writes_allowed": ("a test",), "software_reset": (),
    "park_dmd": (True,), "set_mode": ("pattern",), "set_pattern_source": ("stream",),
    "set_pattern_trigger_mode": (0,), "set_trigger_out1": (), "set_leds": (True, False, False),
    "set_led_current": (10, 10, 10), "set_exposure": (15000, 16667), "main_status": (),
    "validate": (), "load_pattern_sequence": ([None],), "start_sequence": (),
    "pause_sequence": (), "stop_sequence": (), "project_static": (np.zeros((2, 2), np.uint8),),
}
READS = {"main_status", "validate"}          # status reads: the only traffic without permission


def _public(cls) -> set:
    return {n for n, f in inspect.getmembers(cls, callable) if not n.startswith("_")}


def test_the_call_table_names_every_public_method(seed):
    """A method added to the driver later cannot be forgotten by the test below."""
    assert _public(seed.drv.DLPDriver) == set(CALLS)


@pytest.mark.parametrize("name", sorted(CALLS))
def test_no_public_method_writes_to_a_real_device(seed, no_hidapi, monkeypatch, name):
    """Every public method, called once on a real device without writes_allowed: no
    write reaches the device. Only the two status reads put anything on it at all."""
    monkeypatch.setattr(seed.drv.time, "sleep", lambda s: None)
    d, hid = _real(seed)
    try:
        r = getattr(d, name)(*CALLS[name])
        if name == "writes_allowed":                 # entering it alone sends nothing
            with r:
                pass
    except (seed.drv.WriteRefused, RuntimeError):
        pass
    assert hid.writes == []
    if name in READS:
        assert hid.written and all(p[1] & 0x80 for p in hid.written)
    else:
        assert hid.written == []


def test_open_is_refused_without_hardware(seed, no_hidapi):
    """SAS_NO_HARDWARE (set for every test): open() refuses before hidapi is asked."""
    d = seed.drv.DLPDriver()
    with pytest.raises(RuntimeError, match="SAS_NO_HARDWARE"):
        d.open()
    with pytest.raises(RuntimeError, match="SAS_NO_HARDWARE"):
        with seed.drv.DLPDriver():
            pass
    assert d._hid is None


def test_close_does_not_park_the_mirrors(seed):
    d, hid = _real(seed)
    d.close()                                        # `with DLPDriver() as d: pass` ends here
    assert hid.closed and hid.written == [] and d._hid is None


def test_writes_pass_only_inside_writes_allowed(seed):
    d, hid = _real(seed)
    with pytest.raises(seed.drv.WriteRefused, match="writes_allowed"):
        d.set_pattern_source("stream")
    assert hid.written == [] and d.cfg.pattern_source == "stream"     # the default, untouched
    with pytest.raises(ValueError, match="reason"):
        with d.writes_allowed(""):
            pass
    with d.writes_allowed("a test: outer"):
        d.set_pattern_trigger_mode(0)
        with d.writes_allowed("a test: inner"):
            d.set_pattern_source("flash")
        d.stop_sequence()                            # still allowed after the inner block
    assert hid.writes == [(0x1A, 0x23), (0x1A, 0x22), (0x1A, 0x24)]
    n = len(hid.written)
    with pytest.raises(seed.drv.WriteRefused):
        d.stop_sequence()                            # and not after the outer one
    with pytest.raises(seed.drv.WriteRefused):       # an error inside closes it too
        with d.writes_allowed("a test: fails"):
            raise seed.drv.WriteRefused("x")
    with pytest.raises(seed.drv.WriteRefused):
        d.start_sequence()
    assert len(hid.written) == n


def test_a_long_write_is_refused_before_its_first_packet(seed):
    d, hid = _real(seed)
    data = bytes(range(200))                         # four packets
    with pytest.raises(seed.drv.WriteRefused):
        d._send(seed.drv.CMD.LUT_DATA, data)
    assert hid.written == []
    with d.writes_allowed("a test"):
        d._send(seed.drv.CMD.LUT_DATA, data)
    assert len(hid.written) == 4


DISPLAY_WRITES = [
    lambda d, m: d.set_mode("pattern"),
    lambda d, m: d.set_mode("video"),
    lambda d, m: d._send(m.CMD.DISPLAY_MODE, b"\x01"),
    lambda d, m: d._send((0x1A, 0x1B), b"\x00"),
    lambda d, m: d._send([0x1A, 0x1B], b"\x01"),             # a list is not the tuple
    lambda d, m: d._send(np.array([0x1A, 0x1B]), bytearray(b"\x01")),
    lambda d, m: d._send(iter((26, 27)), b"\x01"),
    lambda d, m: d._send(DISPLAY, bytes(range(200))),        # nor in a long write
]


@pytest.mark.parametrize("i", range(len(DISPLAY_WRITES)))
@pytest.mark.parametrize("allowed", [False, True])
def test_the_display_register_is_never_written(seed, i, allowed):
    """Display Mode Select is refused on the built packet, however the command is
    spelled, with and without writes_allowed, and the mode the driver believes in is
    not changed."""
    d, hid = _real(seed)
    before = d.cfg.mode
    with pytest.raises(seed.drv.WriteRefused, match="switched by a person"):
        if allowed:
            with d.writes_allowed("a test"):
                DISPLAY_WRITES[i](d, seed.drv)
        else:
            DISPLAY_WRITES[i](d, seed.drv)
    assert hid.written == [] and d.cfg.mode == before


def test_reading_the_display_register_is_a_read(seed):
    d, hid = _real(seed)
    d._send(seed.drv.CMD.DISPLAY_MODE, read=True)
    assert len(hid.written) == 1 and hid.written[0][1] & 0x80 and hid.writes == []


def test_the_guard_reads_the_packet_as_the_wire_format_builds_it(seed):
    """Bytes 5-6 are CMD3, CMD2 and bit 7 of byte 1 is "read": what the guard looks at."""
    w = seed.drv.build_packet(DISPLAY, b"\x01", seq=3)[0]
    r = seed.drv.build_packet(DISPLAY, read=True, seq=3)[0]
    assert (w[6], w[5]) == DISPLAY == (r[6], r[5]) == seed.drv.CMD.DISPLAY_MODE
    assert not w[1] & 0x80 and r[1] & 0x80


def test_the_mock_driver_is_untouched(seed):
    """No hardware behind it: the simulator records every command, as before."""
    m = seed.drv.MockDLPDriver().open()
    m.set_mode("pattern")
    m.set_pattern_source("stream")
    m.park_dmd(True)
    assert m.sent == [(DISPLAY, b"\x01"), ((0x1A, 0x22), b"\x00"), ((0x06, 0x09), b"\x01")]
    assert m.validate()["valid"]


def test_sensor_rig_with_usb_is_refused_before_anything_opens(seed, no_hidapi, monkeypatch):
    opened = []
    from swiss_army_sensor.drivers import camera_driver, pattern_transport
    monkeypatch.setattr(seed.drv.DLPDriver, "open", lambda self: opened.append("usb"))
    monkeypatch.setattr(pattern_transport.HDMITransport, "__init__",
                        lambda self, *a, **k: opened.append("window"))
    monkeypatch.setattr(camera_driver.CameraDriver, "open", lambda self, *a: opened.append("camera"))
    for call in (lambda: seed.core.SensorRig.create(mock=False, usb=True),
                 lambda: seed.core.SensorRig.create(False, "plane", None, (1920, 0), True),
                 lambda: seed.core.SensorRig.create(mock=False, usb=1)):
        with pytest.raises(seed.drv.WriteRefused, match="refused"):
            call()
    assert opened == []
    for fn in (seed.core.SensorRig.create, seed.drv.DLPDriver.set_mode):     # no flag opens it
        assert not [p for p in inspect.signature(fn).parameters if "allow" in p]
    rig = seed.core.SensorRig.create(mock=True, usb=True)        # the simulator: no device
    assert isinstance(rig.dlp, seed.drv.MockDLPDriver) and rig.dlp.sent == []


def test_scan_path_clean_again_after_the_seed_fixture():
    rules.check_scan_path_usb_free()
    Projector(ProjectionCfg(), mock=True).project(np.zeros((1140, 912), np.uint8))


# ----------------------------------------------------------- engine rule names

def test_switch_display_and_the_helper_are_named_in_the_rules():
    assert "switch_display" in rules.USB_DISPLAY_COMMANDS
    assert "engine.seqctl.helper" in rules.FORBIDDEN_SCAN_MODULES
    proj = Projector(ProjectionCfg(), mock=True)
    with pytest.raises(rules.SafetyViolation, match="HDMI-only"):
        proj.switch_display("pattern")
    with pytest.raises(rules.SafetyViolation, match="engine.seqctl.helper"):
        rules.check_scan_path_usb_free({"engine.seqctl.helper": object()})

    class Sneaky:
        def show(self, p): ...
        def hold(self, s): ...
        def switch_display(self, target): ...
    with pytest.raises(rules.SafetyViolation, match="switch_display"):
        rules.check_transport(Sneaky())


# ------------------------------------------------- who may allow a write

def _writes_allowed_callers(root) -> list:
    """(file, enclosing function) of every call of writes_allowed in the source."""
    out = []
    for top in ("engine", "ui", "tools", "scripts", "examples", "swiss_army_sensor"):
        for py in sorted((root / top).rglob("*.py")):
            tree = ast.parse(py.read_text(encoding="utf-8-sig"))
            for fn in [n for n in ast.walk(tree)
                       if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Module))]:
                for n in ast.iter_child_nodes(fn):
                    out += [(py.relative_to(root).as_posix(), getattr(fn, "name", "<module>"))
                            for c in _calls_below(n) if _name(c.func) == "writes_allowed"]
    return out


def _calls_below(node):
    """Call nodes under `node` that are not inside a nested function."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return
    if isinstance(node, ast.Call):
        yield node
    for c in ast.iter_child_nodes(node):
        yield from _calls_below(c)


def _name(func):
    return getattr(func, "attr", getattr(func, "id", None))


def test_only_the_diagnostics_tool_allows_writes():
    """writes_allowed is entered in one place: main() of dmd_control, which gets there
    only with --yes (tests/test_dmd_control.py::test_write_commands_need_yes)."""
    assert _writes_allowed_callers(ROOT) == [("engine/capture/dmd_control.py", "main")]


# ------------------------------------------- the display command, by value

# What the search looks for, in every text file of the repo:
PATTERNS = {
    # the command as (CMD2, CMD3), and as it goes on the wire (CMD3 first)
    "bytes": re.compile(r"0x1a\s*,\s*0x1b|0x1b\s*,\s*0x1a|\\x1a\\x1b|\\x1b\\x1a", re.I),
    # the command as one word: 0x1A1B, 1A1B (not inside a longer hex string)
    "word": re.compile(r"(?<![0-9a-f])(?:0x)?1a1b(?![0-9a-f])", re.I),
    # its name in the seed driver
    "name": re.compile(r"DISPLAY_MODE"),
    # the driver call (SensorRig.set_mode(SensorMode...) picks a sensor, not a display)
    "call": re.compile(r"\bset_mode\((?!\s*SensorMode\b)"),
}
TEXT = {".py", ".md", ".txt", ".json", ".toml", ".bat", ".sh", ".html", ".js", ".css", ".cfg",
        ".ini", ".yml", ".yaml", ".csv", ".ps1", ""}
# not source: caches and run folders anywhere, the runtime data folders at the top
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", "node_modules", "runs"}
SKIP_TOP = {"library", "library_mock", "scan_out", "build", "dist", ".venv", "venv"}
THIS = "tests/test_display_write_guard.py"

# The allow-list: file -> the most lines that may match, per pattern. A hit in any other
# file, or one more hit in a file listed here, fails the test: look at it, then decide.
ALLOWED = {
    # the seed driver: the command table, the guard that refuses the write on a real
    # device, and set_mode(), which only the mock driver completes
    "swiss_army_sensor/drivers/dlp_driver.py": {"bytes": 1, "name": 3, "call": 1},
    # SensorRig.set_mode(mode: SensorMode): the sensor mode, not the display
    "swiss_army_sensor/core.py": {"call": 1},
    "swiss_army_sensor/README.md": {"call": 1},
    # the wire-format test of the seed package (no device)
    "tests_sensor/test_driver_packets.py": {"bytes": 1},
    # warnings against the call, in the rule's own words (the last two are local files
    # that are not part of the published repo)
    "engine/rules.py": {"call": 1},
    "docs/SCANNER_BRIEFING.md": {"call": 2},
    "scripts/first_scan_v10.py": {"call": 1},
    # a made-up transport that the rule must refuse
    "tests/test_rules.py": {"call": 1},
    # the assert that the diagnostics tool's constant for the command is gone
    "tests/test_dmd_control.py": {"name": 1},
    # this file
    THIS: None,
    # the one sender, built in the next package
    "engine/seqctl/link.py": None,
}


def _display_command_hits(root) -> dict:
    """{file: {pattern: matching lines}} over the text files under `root`."""
    hits = {}
    stack = [root]
    while stack:
        for p in sorted(stack.pop().iterdir()):
            if p.is_dir():
                skip = p.name in SKIP_DIRS or p.name.endswith(".egg-info") or (
                    p.parent == root and p.name in SKIP_TOP)
                if not skip:
                    stack.append(p)
                continue
            if p.suffix.lower() not in TEXT or p.stat().st_size > 4_000_000:
                continue
            try:
                lines = p.read_text(encoding="utf-8-sig").splitlines()
            except (UnicodeDecodeError, OSError):
                continue
            found = {k: sum(1 for ln in lines if rx.search(ln)) for k, rx in PATTERNS.items()}
            found = {k: n for k, n in found.items() if n}
            if found:
                hits[p.relative_to(root).as_posix()] = found
    return hits


def _offenders(hits: dict, allowed: dict) -> list:
    out = []
    for f, found in sorted(hits.items()):
        if f not in allowed:
            out.append(f"{f}: {found}")
        elif allowed[f] is not None:
            over = {k: n for k, n in found.items() if n > allowed[f].get(k, 0)}
            if over:
                out.append(f"{f}: {over} (allowed: {allowed[f]})")
    return out


@pytest.fixture(scope="module")
def repo_hits():
    return _display_command_hits(ROOT)


def test_the_display_command_occurs_only_where_it_is_allowed(repo_hits):
    offenders = _offenders(repo_hits, ALLOWED)
    assert not offenders, (
        "the projector's display command (or its driver call) is named outside the "
        f"allow-list of {THIS}: " + "; ".join(offenders))


def test_the_search_sees_the_files_it_is_about(repo_hits):
    """The walk reaches the seed package and its tests, the engine and the tests."""
    for f in ("swiss_army_sensor/drivers/dlp_driver.py", "swiss_army_sensor/README.md",
              "tests_sensor/test_driver_packets.py", "engine/rules.py", THIS):
        assert f in repo_hits, f


@pytest.mark.parametrize("rel,text,kind", [
    ("engine/capture/other.py", "CMD = (0x1A, 0x1B)\n", "bytes"),
    ("engine/capture/other.py", "cmd = [0x1a,0x1b]\n", "bytes"),
    ("tools/poke.py", "dev.write(bytes([0, 0x40, 1, 3, 0, 0x1B, 0x1A, 1]))\n", "bytes"),
    ("tools/poke.py", "PKT = b'\\x00\\x40\\x01\\x03\\x00\\x1b\\x1a\\x01'\n", "bytes"),
    ("ui/web/thing.py", "link.write(0x1A1B, 1)\n", "word"),
    ("presets/odd.json", '{"command": "1A1B", "data": 1}\n', "word"),
    ("docs/NOTE.md", "send 0x1a1b with data 01 to switch\n", "word"),
    ("engine/capture/other.py", "DISPLAY_MODE = 1\n", "name"),
    ("scripts/try.py", "dlp.set_mode('pattern')\n", "call"),
    ("scripts/try.py", "d.set_mode(m)\n", "call"),
    ("engine/seqctl/client.py", "WRITE = (0x1A, 0x1B)\n", "bytes"),
    ("engine/library/store.py", "WRITE = (0x1A, 0x1B)\n", "bytes"),      # source, not data
    ("swiss_army_sensor/core.py", "def set_mode(self, m): ...\ndlp.set_mode('pattern')\n", "call"),
])
def test_the_search_finds_a_new_mention(tmp_path, rel, text, kind):
    f = tmp_path / rel
    f.parent.mkdir(parents=True)
    f.write_text(text, encoding="utf-8")
    hits = _display_command_hits(tmp_path)
    assert kind in hits[rel]
    assert len(_offenders(hits, ALLOWED)) == 1


def test_the_search_leaves_other_text_alone(tmp_path):
    files = {
        "engine/reconstruct/x.py": "meta['display_mode'] = display_mode = 'video'\n",
        "examples/e.py": "rig.set_mode(SensorMode.EDGE)\nrig.set_mode( SensorMode('x'))\n",
        "presets/p.json": '{"sha": "9f1a1b77c0", "id": "a1a1b2"}\n',
        "engine/capture/y.py": "CMD_INPUT_SOURCE = (0x1A, 0x00)\nSTATUS = (0x1A, 0x0C)\n",
        "engine/seqctl/link.py": "DISPLAY = (0x1A, 0x1B)\n",         # the one sender
        "library/metric/g/e/meta.json": '{"note": "0x1A1B"}\n',       # data, not source
    }
    for rel, text in files.items():
        f = tmp_path / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text, encoding="utf-8")
    hits = _display_command_hits(tmp_path)
    assert set(hits) == {"engine/seqctl/link.py"}
    assert _offenders(hits, ALLOWED) == []

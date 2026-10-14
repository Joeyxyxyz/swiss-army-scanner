"""dmd_control — the standalone DLPC350 USB diagnostics tool (mock only).

The module loads the USB driver, which (by design) blocks the scan path in
the same process, so it is imported inside a fixture and removed from
sys.modules afterwards.
"""

import importlib
import sys

import numpy as np
import pytest


@pytest.fixture
def dc():
    before = set(sys.modules)
    mod = importlib.import_module("engine.capture.dmd_control")
    yield mod
    for k in set(sys.modules) - before:
        del sys.modules[k]


def lock_reply(code=0x02, h=912, v=1140, total_px=1072, total_lines=1180,
               active_px=912, active_lines=1140, first_px=144, first_line=37,
               vfreq_raw=6000, hpol=0, vpol=1, pclk=759, hfreq=708):
    le16 = lambda x: x.to_bytes(2, "little")
    return (bytes([code]) + le16(h) + le16(v) + b"\x00" + bytes([hpol, vpol])
            + pclk.to_bytes(4, "little") + le16(hfreq) + le16(vfreq_raw)
            + le16(total_px) + le16(total_lines) + le16(active_px) + le16(active_lines)
            + le16(first_px) + le16(first_line))


def scripted(dc, replies):
    class Scripted(dc.MockDLPDriver):
        def __init__(self):
            super().__init__()
            self.calls, self._replies = [], list(replies)

        def _send(self, cmd, data=b"", read=False):
            self.calls.append((cmd, bytes(data), read))
            if read:
                return self._replies.pop(0) if len(self._replies) > 1 else self._replies[0]
            return b""
    return Scripted()


def test_lock_status_opcode_and_wire_bytes(dc):
    from swiss_army_sensor.drivers.dlp_driver import build_packet
    assert dc.CMD_VIDEO_SIGNAL_STATUS == (0x07, 0x1C)     # DLPU010G §2.1.5 / p.68
    pkt = build_packet(dc.CMD_VIDEO_SIGNAL_STATUS, b"", read=True, seq=1)[0]
    assert pkt[1] == 0xC0                                  # read + reply
    assert pkt[3:5] == b"\x02\x00"                         # payload = CMD3 + CMD2
    assert pkt[5:7] == b"\x1c\x07"                         # CMD3 first, then CMD2


def test_select_source_and_bit_depth_encoding(dc):
    drv = scripted(dc, [lock_reply()])
    d = dc.DMDControl(driver=drv)
    d.select_source(0, bit_depth=24)      # bits 2:0 = 0, bits 5:3 = 1 (24-bit)
    d.select_source(1)                    # internal test pattern
    d.select_source(0, bit_depth=30)
    assert drv.calls == [((0x1A, 0x00), b"\x08", False),
                         ((0x1A, 0x00), b"\x09", False),
                         ((0x1A, 0x00), b"\x00", False)]
    with pytest.raises(ValueError):
        d.select_source(4)
    with pytest.raises(ValueError):
        d.select_source(0, bit_depth=12)


def test_tool_has_no_display_command(dc):
    """The tool cannot switch the projector display (video / pattern sequence): the
    method and its command constant are gone, and no flag reaches one."""
    assert not hasattr(dc.DMDControl, "select_display_mode")
    assert not hasattr(dc, "CMD_DISPLAY_MODE")
    names = [n for n in dir(dc.DMDControl) if "display" in n.lower() or "mode" in n.lower()]
    assert names == []
    assert dc.main(["--mock", "--status", "--yes"]) == 2      # a known command line (mock: zeros)
    for flag in ("--display", "--mode", "--display-mode", "--pattern", "--video"):
        with pytest.raises(SystemExit):                       # the same with an unknown flag
            dc.main(["--mock", "--status", "--yes", flag, "1"])


def test_parse_full_timing_and_margins(dc):
    st = dc.parse_video_lock(lock_reply())
    assert st.detected and st.state == "Detected" and st.resolution == "912x1140"
    assert st.v_blank_lines == 40 and st.h_blank_pixels == 160
    assert st.v_backporch_lines == 37 and st.v_freq_hz == pytest.approx(60.0)
    assert st.vsync_positive and not st.hsync_positive
    assert st.implied_pixel_clock_mhz == pytest.approx(1072 * 1180 * 60 / 1e6)
    t = st.table()
    for key in ("VERTICAL BLANKING", "40 lines", "first line", "912x1140", "60.00 Hz"):
        assert key in t
    with pytest.raises(ValueError, match="firmware"):
        dc.parse_video_lock(b"")


def test_safe_relock_polls_until_detected(dc):
    drv = scripted(dc, [lock_reply(code=0x01), lock_reply(code=0x01), lock_reply(code=0x02)])
    sleeps = []
    st = dc.DMDControl(driver=drv).safe_relock(timeout_s=5, sleep=sleeps.append)
    assert st.detected
    assert drv.calls[0] == ((0x1A, 0x00), b"\x08", False)      # source 0 @ 24-bit first
    assert [c[0] for c in drv.calls[1:]] == [(0x07, 0x1C)] * 3
    assert sleeps == [pytest.approx(0.2)] * 2


def test_safe_relock_reports_lock_failed(dc):
    drv = scripted(dc, [lock_reply(code=0x03, v=1139, total_lines=1150, first_line=6)])
    st = dc.DMDControl(driver=drv).safe_relock(timeout_s=0.05, poll_s=0.01)
    assert st.code == 0x03 and not st.detected
    msg = st.describe()
    assert "LOCK FAILED" in msg and "912x1139" in msg and "first line 6" in msg


def test_close_never_parks_dmd(dc):
    class FakeHID:
        closed = False
        def close(self):
            self.closed = True
    drv = scripted(dc, [lock_reply()])
    hid = FakeHID()
    drv._hid = hid
    dc.DMDControl(driver=drv).close()
    assert hid.closed and drv._hid is None
    assert not any(c[0] == (0x06, 0x09) for c in drv.calls)    # DMD park command


def test_cli_mock_status_prints_table(dc, capsys):
    rc = dc.main(["--mock", "--status"])
    out = capsys.readouterr().out
    assert rc == 2 and "Stopped" in out and "VERTICAL BLANKING" in out   # mock reads zeros
    with pytest.raises(SystemExit):
        dc.main(["--mock"])


def test_cli_bitdepth_alone_asserts_source0(dc, monkeypatch):
    sent = []
    orig = dc.DMDControl.select_source
    monkeypatch.setattr(dc.DMDControl, "select_source",
                        lambda self, k, bit_depth=24: sent.append((k, bit_depth)) or orig(self, k, bit_depth))
    monkeypatch.setattr(dc.time, "sleep", lambda s: None)
    dc.main(["--mock", "--bitdepth", "24", "--yes"])
    assert sent == [(0, 24)]


def test_loading_tool_blocks_scan_path(dc):
    """The whole point of the separation: with the USB driver loaded, the
    projector refuses to run in this process."""
    from engine import rules
    from engine.capture.projector import Projector
    from engine.config import ProjectionCfg
    with pytest.raises(rules.SafetyViolation):
        Projector(ProjectionCfg(), mock=True)


def test_scan_path_clean_again_after_fixture():
    from engine.capture.projector import Projector
    from engine.config import ProjectionCfg
    Projector(ProjectionCfg(), mock=True).project(np.zeros((1140, 912), np.uint8))


# ------------------------------------------- rig lock around write commands

@pytest.fixture
def lockfile(dc, tmp_path, monkeypatch):
    """Point the tool's rig lock at a private file (no machine-wide lock)."""
    path = tmp_path / "rig.lock"
    real = dc.RigLock
    monkeypatch.setattr(dc, "RigLock", lambda holder="": real(path, holder))
    return path, real


def _fake_real_driver(dc, on_send):
    """Stands in for the real DLPDriver (no USB): records each command."""
    class FakeReal(dc.MockDLPDriver):
        def _send(self, cmd, data=b"", read=False):
            on_send(cmd, read)
            return lock_reply() if read else b""
    return FakeReal


WRITES = [["--relock", "--timeout", "0.01"], ["--source", "0"], ["--bitdepth", "24"],
          ["--source", "1"]]


@pytest.mark.parametrize("argv", WRITES)
def test_write_commands_hold_the_rig_lock(dc, lockfile, monkeypatch, argv):
    path, real = lockfile
    seen = []

    def on_send(cmd, read):
        # while the tool talks to the controller, nobody else can take the rig
        with pytest.raises(dc.RigBusy, match="dmd_control"):
            real(path, "scan").acquire()
        seen.append(cmd)
    monkeypatch.setattr(dc, "DLPDriver", _fake_real_driver(dc, on_send))
    monkeypatch.setattr(dc.time, "sleep", lambda s: None)
    assert dc.main(argv + ["--yes"]) in (0, 2)
    assert (0x1A, 0x00) in seen                          # the write went out, under the lock
    real(path, "scan").acquire().release()              # ...and the lock is released after


def test_write_command_refused_while_a_scan_holds_the_rig(dc, lockfile, monkeypatch, capsys):
    path, real = lockfile
    opened = []

    class Untouchable(dc.MockDLPDriver):
        def open(self):
            opened.append(1)
            raise AssertionError("USB opened while the rig is busy")
    monkeypatch.setattr(dc, "DLPDriver", Untouchable)
    scan = real(path, "web UI scan").acquire()
    try:
        for argv in (["--relock"], ["--source", "0"], ["--bitdepth", "24"]):
            assert dc.main(argv + ["--yes"]) == 2
            err = capsys.readouterr().err.strip().splitlines()
            assert len(err) == 1 and "nothing sent" in err[0] and "web UI scan" in err[0]
    finally:
        scan.release()
    assert not opened                                    # refused before touching the device


@pytest.mark.parametrize("argv", WRITES)
@pytest.mark.parametrize("mock", [[], ["--mock"]])
def test_write_commands_need_yes(dc, lockfile, monkeypatch, capsys, argv, mock):
    """--relock, --source and --bitdepth make the projector re-acquire its video input.
    Without --yes: exit 2, one line, the device not opened, the rig lock not taken, and
    the command never waits for input."""
    path, real = lockfile
    touched = []

    class Untouchable(dc.MockDLPDriver):
        def open(self):
            touched.append("open")
            return self

        def _send(self, cmd, data=b"", read=False):
            touched.append(cmd)
            return b""
    monkeypatch.setattr(dc, "DLPDriver", Untouchable)
    monkeypatch.setattr(dc, "MockDLPDriver", Untouchable)
    monkeypatch.setattr("builtins.input", lambda *a: pytest.fail("the tool asked a question"))
    scan = real(path, "a scan").acquire()                # would be RigBusy if the lock were tried
    try:
        assert dc.main(mock + argv) == 2
    finally:
        scan.release()
    cap = capsys.readouterr()
    err = cap.err.strip().splitlines()
    assert len(err) == 1 and "nothing sent" in err[0] and "--yes" in err[0]
    assert "a scan" not in err[0] and cap.out == ""
    assert touched == []


def test_status_takes_the_rig_lock_and_is_refused_while_a_scan_holds_it(dc, lockfile,
                                                                         monkeypatch, capsys):
    """A scan may be playing a pattern sequence, when nothing may be sent to the
    controller, not even a read: --status takes the rig lock like every other command."""
    path, real = lockfile
    seen = []

    def on_send(cmd, read):
        assert read                                      # --status only reads
        with pytest.raises(dc.RigBusy, match="dmd_control"):
            real(path, "scan").acquire()                 # held during the read
        seen.append(cmd)
    monkeypatch.setattr(dc, "DLPDriver", _fake_real_driver(dc, on_send))
    assert dc.main(["--status"]) == 0                    # needs no --yes
    assert seen == [(0x07, 0x1C)] and "Detected" in capsys.readouterr().out
    real(path, "scan").acquire().release()              # released after
    opened = []

    class Untouchable(dc.MockDLPDriver):
        def open(self):
            opened.append(1)
            raise AssertionError("USB opened while the rig is busy")
    monkeypatch.setattr(dc, "DLPDriver", Untouchable)
    scan = real(path, "web UI scan").acquire()
    try:
        assert dc.main(["--status"]) == 2
    finally:
        scan.release()
    cap = capsys.readouterr()
    err = cap.err.strip().splitlines()
    assert len(err) == 1 and "nothing sent" in err[0] and "web UI scan" in err[0]
    assert not opened and cap.out == ""


def test_mock_takes_no_lock(dc, lockfile, monkeypatch):
    path, real = lockfile
    monkeypatch.setattr(dc.time, "sleep", lambda s: None)
    scan = real(path, "scan").acquire()
    try:
        assert dc.main(["--mock", "--source", "1", "--yes"]) == 0
        assert dc.main(["--mock", "--status"]) == 2      # the mock reads zeros: not Detected
    finally:
        scan.release()


# ------------------------------------- the driver's write guard under the tool

class FakeHID:
    """Stands in for the opened USB device: keeps what is written to it."""

    def __init__(self, reply=None):
        self.written, self.closed, self.reply = [], False, reply

    def write(self, pkt):
        self.written.append(bytes(pkt))
        return len(pkt)

    def read(self, n, timeout_ms=0):
        data = lock_reply() if self.reply is None else self.reply
        return list(bytes([0x40, 0, len(data) & 0xFF, len(data) >> 8]) + data)[:n]

    def close(self):
        self.closed = True


def _real_tool(dc):
    """DMDControl on the REAL driver class with a fake device in place of hidapi."""
    drv = dc.DLPDriver()
    drv._hid = FakeHID()
    return dc.DMDControl(driver=drv), drv._hid


def test_tool_writes_as_python_calls_are_refused(dc):
    """select_source / safe_relock called from other code, on a real device: refused by
    the driver with nothing sent. Only main() allows writes, after --yes."""
    from swiss_army_sensor.drivers.dlp_driver import WriteRefused
    dmd, hid = _real_tool(dc)
    with pytest.raises(WriteRefused, match="Nothing was sent"):
        dmd.select_source(0, bit_depth=24)
    with pytest.raises(WriteRefused):
        dmd.select_source(1)
    with pytest.raises(WriteRefused):
        dmd.safe_relock(timeout_s=0.01)
    assert hid.written == []
    assert dmd.read_video_lock().detected                # a read needs no permission
    assert len(hid.written) == 1 and hid.written[0][1] & 0x80
    assert hid.written[0][5:7] == b"\x1c\x07"
    dmd.close()
    assert hid.closed and len(hid.written) == 1          # closed, and never parked


@pytest.mark.parametrize("argv,writes", [(["--status"], 0), (["--source", "0", "--yes"], 1),
                                         (["--relock", "--yes", "--timeout", "0.01"], 1),
                                         (["--bitdepth", "24", "--yes"], 1)])
def test_cli_on_the_real_driver_class(dc, lockfile, monkeypatch, argv, writes):
    """main() on the real driver class (fake device): --status sends reads only; a write
    command with --yes sends its one Input Source Selection write and never a park, a
    reset or the display command."""
    hid = FakeHID()

    class Opened(dc.DLPDriver):
        def open(self):
            self._hid = hid
            return self
    monkeypatch.setattr(dc, "DLPDriver", Opened)
    monkeypatch.setattr(dc.time, "sleep", lambda s: None)
    assert dc.main(argv) == 0
    sent = [(p[1] & 0x80 == 0, (p[6], p[5])) for p in hid.written]
    assert [c for w, c in sent if w] == [(0x1A, 0x00)] * writes
    assert {c for w, c in sent if not w} == {(0x07, 0x1C)}
    assert hid.closed


def test_tool_loads_only_riglock_and_rules_from_the_engine():
    import ast
    from conftest import ROOT
    tree = ast.parse((ROOT / "engine" / "capture" / "dmd_control.py").read_text(encoding="utf-8"))
    rel = sorted({(n.level, n.module) for n in ast.walk(tree)
                  if isinstance(n, ast.ImportFrom) and n.level > 0})
    assert rel == [(1, "riglock"), (2, "rules")]

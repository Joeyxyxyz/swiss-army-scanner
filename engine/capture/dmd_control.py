"""
DMD control utility — DLPC350 over USB-HID. NOT PART OF THE SCAN PATH.

This module exists for diagnostics of the video input. It is one of the two
engine modules allowed to load USB code (the sequence-control helper,
engine/seqctl/helper.py, is the other), the only one that loads the seed
DLPC350 driver, and loading it poisons the process for scanning on purpose:
`rules.check_scan_path_usb_free()` refuses to open or drive the projector
once swiss_army_sensor.drivers.dlp_driver is imported. Run it as its own
process:

    python -m engine.capture.dmd_control --status
    python -m engine.capture.dmd_control --relock --yes
    python -m engine.capture.dmd_control --source 0 --yes

Every command takes the machine-wide rig lock (capture/riglock.py, holder
"dmd_control") before the USB device is opened, and exits 2 with one line if
a scan holds it: nothing may be sent to the controller while a scan runs.

The write commands (--relock, --source, --bitdepth) make the controller
re-acquire its video input: the picture leaves the wall and may not come
back without a power cycle. They need --yes and are refused (exit 2, nothing
sent) without it; no command waits for input. --status only reads.

Writes reach the device only inside `driver.writes_allowed(reason)`, which
main() enters after --yes and nothing else in the repo enters: select_source
and safe_relock called from other code on a real device are refused by the
driver. This tool cannot switch the projector display (video / pattern
sequence): it has no such command, and the driver refuses that write always.

tests/test_rules.py fails the build if any other engine module imports it.

THE "BLUE LINES" FAILURE — root cause (working hypothesis)
    * DLPC350 video display mode is specified for streaming input "up to
      1280 x 800 up to 120 Hz" (DLPU010G §2.4.1 Table 2-49, §2.4.2); 912 x 1140
      is the pattern-display-mode native, pixel-accurate resolution (same
      table). The EVM's HDMI input also accepts 912 x 1140 in video mode
      (DLPU011E, connector 20), and it works when the EVM boots into it.
    * The 912 x 1140 custom mode on this laptop was created with CVT-RB
      (reduced blanking) -> small vertical back porch -> marginal lock.
      DLPU011E §3.2 ("Forced Swap" status): buffer-swap errors "can occur if
      the DLP LightCrafter 4500 is set to Video Mode and the vertical
      backporch timing is too small."
    * A USB display-mode / input-source command makes the controller
      re-acquire the source; with marginal timing the re-lock fails and the
      DMD falls back to its internal test pattern: the blue lines.
    * Fix (human, see docs/HDMI_TIMING_FIX.md): re-create 912x1140@60 with
      standard CVT timing, parallel RGB bit depth 24. This tool VERIFIES it:
      --status shows the timing the controller actually locked to, including
      vertical blanking and the first active line.
    DLPU010G §2.1.5 names the status read as the debugging aid for this
    ("detect if the controller reads the resolution correctly").

COMMANDS (DLPU010G; wire format = swiss_army_sensor.drivers.dlp_driver,
byte-verified against pycrafter4500 — CMD3 goes on the wire before CMD2)
    Input Source Selection   CMD2 0x1A  CMD3 0x00  write   §2.3.4.2, Table 2-19, p.20-21
        byte0 bits 2:0 source: 0 parallel RGB (the EVM's HDMI), 1 internal
        test pattern, 2 flash, 3 FPD-link; bits 5:3 parallel bit depth
        (0=30, 1=24 [reset], 2=20, 3=16, 4=10, 5=8).
    Display Mode Selection   CMD2 0x1A  CMD3 0x1B          §2.4.1, Table 2-49, p.37-38
        byte0 bit 0: 0 video, 1 pattern. NOT SENT by this tool (named here
        because it is the other command that makes the controller re-acquire
        its input).
    Input Video Signal       CMD2 0x07  CMD3 0x1C  read    §2.1.5, Table 2-5, p.13-14;
      Detection Status                                     appendix quick reference p.68
        28-byte reply, firmware >= 2.0.0. The controller monitors the signal
        for ~200 ms before answering. (§2.1.5's header also shows a stray
        "USB: 0x04"; the appendix lists only 0x071C / 28 bytes — treated as
        a typo.)

Closing: this utility closes the HID handle directly and never parks the
mirrors, so a status check does not black out the display. (The seed
DLPDriver.close() would park them where writes are allowed.)
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

from swiss_army_sensor.drivers.dlp_driver import DLPDriver, MockDLPDriver

# the only scan-path modules this tool loads: the rig lock (and the rules it
# raises from). Nothing that projects or captures.
from ..rules import RigBusy
from .riglock import RigLock

LOCK_HOLDER = "dmd_control"

CMD_INPUT_SOURCE = (0x1A, 0x00)
CMD_VIDEO_SIGNAL_STATUS = (0x07, 0x1C)

WRITE_FLAGS = "--relock, --source, --bitdepth"
NEEDS_YES = (f"dmd_control: nothing sent - {WRITE_FLAGS} write to the projector: it "
             "re-acquires its video input, the picture leaves the wall and may not come "
             "back without a power cycle. Add --yes to send (no command waits for input).")

SOURCES = {0: "parallel RGB (HDMI on the EVM)", 1: "internal test pattern",
           2: "flash", 3: "FPD-link"}
BIT_DEPTHS = {0: 30, 1: 24, 2: 20, 3: 16, 4: 10, 5: 8}
LOCK_STATES = {0x00: "Stopped", 0x01: "Processing", 0x02: "Detected", 0x03: "Lock Failed"}
DETECTED, LOCK_FAILED = 0x02, 0x03
NATIVE = (912, 1140)
STATUS_POLL_S = 0.2                    # the controller samples ~200 ms per read


@dataclass
class VideoLockStatus:
    code: int
    state: str
    h_res: int = 0
    v_res: int = 0
    hsync_positive: Optional[bool] = None
    vsync_positive: Optional[bool] = None
    pixel_clock_raw: int = 0           # Table 2-5 "(100 x kHz)" units — raw, unscaled
    h_freq_raw: int = 0
    v_freq_raw: int = 0                # Table 2-5 "(100 x Hz)" units — raw, unscaled
    total_pixels_per_line: int = 0
    total_lines_per_frame: int = 0
    active_pixels_per_line: int = 0
    active_lines_per_frame: int = 0
    first_pixel: int = 0
    first_line: int = 0
    raw: bytes = field(default=b"", repr=False)

    @property
    def detected(self) -> bool:
        return self.code == DETECTED

    @property
    def resolution(self) -> str:
        return f"{self.h_res}x{self.v_res}"

    @property
    def v_blank_lines(self) -> int:
        """Vertical blanking = total lines - active lines."""
        return self.total_lines_per_frame - self.active_lines_per_frame

    @property
    def h_blank_pixels(self) -> int:
        return self.total_pixels_per_line - self.active_pixels_per_line

    @property
    def v_backporch_lines(self) -> int:
        """Table 2-5 'First Line (beginning of active line in the frame)',
        counted from the start of VSYNC -> VSYNC width + back porch. The
        guide gives no separate VSYNC width, so this is the upper bound on
        the back porch; it is the number that moves when CVT-RB is replaced
        by standard CVT."""
        return self.first_line

    @property
    def v_freq_hz(self) -> float:
        return self.v_freq_raw / 100.0          # Table 2-5: units of 1/100 Hz

    @property
    def implied_pixel_clock_mhz(self) -> float:
        """total_pixels x total_lines x vfreq — a cross-check on pixel_clock_raw,
        whose unit wording in Table 2-5 is ambiguous."""
        return self.total_pixels_per_line * self.total_lines_per_frame * self.v_freq_hz / 1e6

    def table(self) -> str:
        rows = [
            ("signal status", f"{self.state} (0x{self.code:02X})"),
            ("resolution (H x V)", self.resolution),
            ("HSYNC / VSYNC polarity", f"{'+' if self.hsync_positive else '-'} / "
                                       f"{'+' if self.vsync_positive else '-'}"),
            ("vertical frequency", f"{self.v_freq_hz:.2f} Hz  (raw {self.v_freq_raw})"),
            ("horizontal frequency", f"raw {self.h_freq_raw}  (Table 2-5: 100 x kHz units)"),
            ("pixel clock", f"raw {self.pixel_clock_raw}  (implied from totals: "
                            f"{self.implied_pixel_clock_mhz:.3f} MHz)"),
            ("total pixels / line", self.total_pixels_per_line),
            ("active pixels / line", self.active_pixels_per_line),
            ("horizontal blanking", f"{self.h_blank_pixels} px"),
            ("first pixel", self.first_pixel),
            ("total lines / frame", self.total_lines_per_frame),
            ("active lines / frame", self.active_lines_per_frame),
            ("VERTICAL BLANKING", f"{self.v_blank_lines} lines"),
            ("first line (VSYNC + back porch)", f"{self.v_backporch_lines} lines"),
        ]
        w = max(len(k) for k, _ in rows)
        return "\n".join(f"  {k:<{w}}  {v}" for k, v in rows)

    def describe(self) -> str:
        s = f"video input: {self.state} (0x{self.code:02X}), resolution {self.resolution}"
        if self.v_freq_raw:
            s += f", vfreq raw {self.v_freq_raw}"
        if self.code == LOCK_FAILED:
            s += (f" — LOCK FAILED. The controller saw {self.resolution}, "
                  f"v-blank {self.v_blank_lines} lines, first line {self.first_line}; the DMD "
                  "will show its internal test pattern (blue lines). Check the "
                  "Windows display mode is 912x1140@60 and re-run --relock --yes.")
        elif self.detected and (self.h_res, self.v_res) != NATIVE:
            s += f" — note: not the DMD-native {NATIVE[0]}x{NATIVE[1]}"
        return s


def parse_video_lock(data: bytes) -> VideoLockStatus:
    """Decode the Table 2-5 reply (little-endian multi-byte fields)."""
    if not data:
        raise ValueError("empty reply to Input Video Signal Detection Status "
                         "(firmware older than 2.0.0, or no reply)")
    d = bytes(data).ljust(28, b"\x00")
    u16 = lambda i: d[i] | (d[i + 1] << 8)
    code = d[0]
    return VideoLockStatus(
        code=code, state=LOCK_STATES.get(code, f"unknown 0x{code:02X}"),
        h_res=u16(1), v_res=u16(3),
        hsync_positive=bool(d[6]), vsync_positive=bool(d[7]),
        pixel_clock_raw=int.from_bytes(d[8:12], "little"),
        h_freq_raw=u16(12), v_freq_raw=u16(14),
        total_pixels_per_line=u16(16), total_lines_per_frame=u16(18),
        active_pixels_per_line=u16(20), active_lines_per_frame=u16(22),
        first_pixel=u16(24), first_line=u16(26), raw=bytes(data))


class DMDControl:
    """Thin, guarded wrapper over the DLPC350 HID driver.

    `driver` may be a MockDLPDriver (tests) or a real, unopened DLPDriver.
    On a real driver the writes (select_source, safe_relock) pass only inside
    `driver.writes_allowed(reason)`, which main() enters after --yes: called
    from anywhere else they raise WriteRefused with nothing sent.
    """

    def __init__(self, driver: Optional[DLPDriver] = None, mock: bool = False):
        self.driver = driver or (MockDLPDriver() if mock else DLPDriver())
        self._open = False

    def open(self) -> "DMDControl":
        self.driver.open()
        self._open = True
        return self

    def close(self) -> None:
        """Release the HID handle WITHOUT parking the DMD."""
        hid = getattr(self.driver, "_hid", None)
        if hid is not None:
            try:
                hid.close()
            finally:
                self.driver._hid = None
        self._open = False

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------ commands
    def select_source(self, kind: int, bit_depth: int = 24) -> None:
        """Input Source Selection (0x1A00). kind: 0 parallel/HDMI, 1 test
        pattern, 2 flash, 3 FPD-link. bit_depth applies to the parallel port."""
        if kind not in SOURCES:
            raise ValueError(f"source must be one of {sorted(SOURCES)}")
        depth_code = {v: k for k, v in BIT_DEPTHS.items()}.get(bit_depth)
        if depth_code is None:
            raise ValueError(f"bit_depth must be one of {sorted(BIT_DEPTHS.values())}")
        self.driver._send(CMD_INPUT_SOURCE, bytes([(kind & 0x07) | (depth_code << 3)]))

    def read_video_lock(self) -> VideoLockStatus:
        """Input Video Signal Detection Status (0x071C read, 28 bytes)."""
        return parse_video_lock(self.driver._send(CMD_VIDEO_SIGNAL_STATUS, read=True))

    def safe_relock(self, timeout_s: float = 5.0, bit_depth: int = 24,
                    poll_s: float = STATUS_POLL_S, sleep=time.sleep) -> VideoLockStatus:
        """Re-assert the parallel (HDMI) source and poll until Detected.

        Returns the final status — Detected, Lock Failed (with the resolution
        the controller saw), or the last state seen at timeout. Never hides a
        failure: callers get the status object and the CLI exits non-zero."""
        self.select_source(0, bit_depth=bit_depth)
        deadline = time.monotonic() + timeout_s
        status = self.read_video_lock()
        while not status.detected and time.monotonic() < deadline:
            sleep(poll_s)
            status = self.read_video_lock()
        return status


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m engine.capture.dmd_control",
        description="DLPC350 video-input diagnostics over USB. NOT part of the scan path. "
                    "Every command takes the machine-wide rig lock, so it is refused (exit 2) "
                    "while a scan holds the rig. The write commands (--relock, --source, "
                    "--bitdepth) make the projector re-acquire its video input and need "
                    "--yes; --status only reads.")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--status", action="store_true",
                   help="read input video lock status (read-only; takes the rig lock for "
                        "the read)")
    g.add_argument("--relock", action="store_true",
                   help="re-assert HDMI/parallel source and poll until Detected (a write: "
                        "needs --yes)")
    g.add_argument("--source", type=int, choices=sorted(SOURCES),
                   help="Input Source Selection (a write: needs --yes): "
                        + ", ".join(f"{k}={v}" for k, v in SOURCES.items()))
    ap.add_argument("--bitdepth", "--bit-depth", dest="bit_depth", type=int, default=None,
                    choices=sorted(BIT_DEPTHS.values()),
                    help="parallel RGB bit depth (default 24). Alone: assert source 0 "
                         "at this depth, then read status (a write: needs --yes). With "
                         "--relock/--source: the depth sent with them.")
    ap.add_argument("--yes", action="store_true",
                    help=f"send the write ({WRITE_FLAGS}): the picture leaves the wall while "
                         "the projector re-acquires its input. Without --yes a write command "
                         "is refused (exit 2) with nothing sent")
    ap.add_argument("--timeout", type=float, default=5.0)
    ap.add_argument("--mock", action="store_true", help="MockDLPDriver, no hardware")
    a = ap.parse_args(argv)
    if not (a.status or a.relock or a.source is not None or a.bit_depth is not None):
        ap.error("give one of --status, --relock, --source N, --bitdepth N")
    set_depth_only = a.bit_depth is not None and not (a.status or a.relock or a.source is not None)
    a.bit_depth = 24 if a.bit_depth is None else a.bit_depth

    # every command but --status WRITES to the controller (Input Source
    # Selection makes it re-acquire the HDMI input): the answer is an
    # argument, never a prompt, and without it nothing is opened or sent.
    if not a.status and not a.yes:
        print(NEEDS_YES, file=sys.stderr)
        return 2
    # nothing may be sent to the controller while a scan runs (a write wrecks
    # it, and a scan may be playing a pattern sequence, when even a read is
    # too much): every command takes the same machine-wide lock the scan path
    # takes, BEFORE the USB device is opened. The mock touches no hardware.
    lock = None
    if not a.mock:
        try:
            lock = RigLock(holder=LOCK_HOLDER).acquire()
        except RigBusy as e:
            print(f"dmd_control: nothing sent - {e}", file=sys.stderr)
            return 2
    try:
        with DMDControl(mock=a.mock) as dmd:
            if a.status:
                return _run(a, dmd, set_depth_only)
            # the one place in the repo that allows writes to the real device
            what = "--relock" if a.relock else (
                f"--bitdepth {a.bit_depth}" if set_depth_only else f"--source {a.source}")
            with dmd.driver.writes_allowed(f"dmd_control {what} --yes"):
                return _run(a, dmd, set_depth_only)
    finally:
        if lock is not None:
            lock.release()


def _run(a, dmd: DMDControl, set_depth_only: bool) -> int:
    if set_depth_only:
        dmd.select_source(0, bit_depth=a.bit_depth)
        print(f"input source -> 0 ({SOURCES[0]}), parallel bit depth {a.bit_depth}")
        time.sleep(STATUS_POLL_S)
        st = dmd.read_video_lock()
    elif a.status:
        st = dmd.read_video_lock()
    elif a.relock:
        print(f"re-asserting source 0 ({SOURCES[0]}, {a.bit_depth}-bit), "
              f"polling up to {a.timeout:.1f}s ...")
        st = dmd.safe_relock(timeout_s=a.timeout, bit_depth=a.bit_depth)
    else:
        dmd.select_source(a.source, bit_depth=a.bit_depth)
        print(f"input source -> {a.source} ({SOURCES[a.source]})")
        if a.source != 0:
            return 0
        time.sleep(STATUS_POLL_S)
        st = dmd.read_video_lock()
    print(st.describe())
    print(st.table())
    return 0 if st.detected else 2


if __name__ == "__main__":
    sys.exit(main())

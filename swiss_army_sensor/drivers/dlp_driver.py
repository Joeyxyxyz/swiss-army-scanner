"""
DLP / DMD driver layer (DLPC350 / LightCrafter 4500).

VERIFICATION STATUS (v0.5):
  - Packet layout cross-checked byte-for-byte against pycrafter4500
    (the established, hardware-proven LightCrafter 4500 library) via
    tests/test_driver_packets.py. NOT yet verified on a live DLPC350.
  - Command codes traced to TI DLPU010G (May 2013, rev. July 2018).

WIRE FORMAT (DLPU010G §1.2 + pycrafter4500 ground truth):
  The DLPC350 sends/receives multi-byte fields least-significant-byte
  first. The 16-bit command word (CMD2<<8 | CMD3) therefore goes on the
  wire as CMD3 *then* CMD2. v0.4 of this driver had the order swapped —
  every command would have been silently malformed on hardware.

  Host->device HID report (65 bytes via hidapi: report-id + 64):
    byte0:    report_id      (0x00, hidapi convention)
    byte1:    flags          bit7=read, bit6=reply-required, bits2:0=0
    byte2:    sequence #
    byte3-4:  payload length (LE; counts bytes from CMD3 onward)
    byte5:    CMD3           <-- LSB first
    byte6:    CMD2
    byte7+:   data           (continuation packets carry raw data only)

  Device->host reply (64 bytes):
    byte0:    flags
    byte1:    sequence # (matches request)
    byte2-3:  data length (LE)
    byte4+:   data

  Like pycrafter4500, we set the reply-required bit on every command and
  drain the 64-byte reply, keeping the HID pipe clean and giving us the
  ack/NACK for free.

PATTERN DATA SOURCES (DLPU010G §2.4.3 — there is no USB frame push):
  Pattern display mode accepts 1-8 bit 912x1140 images either streamed
  through the 24-bit RGB / FPD-link interface or stored in flash. On the
  LightCrafter 4500 EVM the streaming path is the HDMI input. Arbitrary
  runtime patterns therefore require a *display transport* (see
  pattern_transport.HDMITransport), with Pattern Trigger Mode 0:
  VSYNC clocks the sequence, TRIG_OUT_1 frames each exposure for the
  camera. `project_static()` on the real driver delegates to the
  attached transport and raises a clear error if none is attached.

Command codes (TI DLPU010G):
  Display Mode Select          CMD2:0x1A CMD3:0x1B
  Pattern Display Data Source  CMD2:0x1A CMD3:0x22  (0=video port, 3=flash)
  Pattern Trigger Mode         CMD2:0x1A CMD3:0x23  (0=VSYNC ... 4)
  Start/Stop Pattern Sequence  CMD2:0x1A CMD3:0x24
  Validate Data                CMD2:0x1A CMD3:0x1A
  LUT Access Control           CMD2:0x1A CMD3:0x33
  LUT Data (Pattern Def)       CMD2:0x1A CMD3:0x34
  Trigger Out1 Control         CMD2:0x1A CMD3:0x1D
  Trigger Out2 Control         CMD2:0x1A CMD3:0x1E
  LED Enable Outputs           CMD2:0x1A CMD3:0x07
  Pattern Exposure/Period      CMD2:0x1A CMD3:0x29
  Software Reset               CMD2:0x08 CMD3:0x02
  DMD Park/Unpark              CMD2:0x06 CMD3:0x09

WRITE GUARD (real device only; the mock driver sends nothing):
  Every command passes one place, DLPDriver._send, and is checked there on
  the BUILT packet (bytes 5-6 are the command, bit 7 of byte 1 is "read"),
  before the first byte goes to the device:
    - a read always passes;
    - a write passes only inside `with driver.writes_allowed(reason)`;
    - a write of Display Mode Select never passes: the projector display is
      switched by a person, never through this driver.
  So no single call (set_mode, software_reset, set_led_current, start_sequence,
  a raw _send, ...) changes anything on the controller, and close() parks the
  mirrors only when writes are allowed. open() refuses when SAS_NO_HARDWARE is
  set (the test suite sets it).
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

DLP4500_VID = 0x0451
DLP4500_PID = 0x6401
DMD_WIDTH = 912
DMD_HEIGHT = 1140

_PKT = 65          # hidapi write size: report id + 64
_FIRST_DATA = 58   # data bytes that fit in the first packet (64 - 6 header)
_CONT_DATA = 64    # data bytes per continuation packet

# environment variable: set (to anything but "" or "0") for a process that must not
# open a real device
NO_HARDWARE_ENV = "SAS_NO_HARDWARE"


class WriteRefused(RuntimeError):
    """A write was about to go to a real DLPC350 and is not allowed. Nothing
    was sent."""


class CMD:
    DISPLAY_MODE       = (0x1A, 0x1B)
    PATTERN_SOURCE     = (0x1A, 0x22)
    PATTERN_TRIG_MODE  = (0x1A, 0x23)
    PATTERN_START_STOP = (0x1A, 0x24)
    VALIDATION         = (0x1A, 0x1A)
    LUT_ACCESS         = (0x1A, 0x33)
    LUT_DATA           = (0x1A, 0x34)
    LED_ENABLE         = (0x1A, 0x07)
    LED_PWM_POLARITY   = (0x1A, 0x05)
    LED_RED_PWM        = (0x0B, 0x01)   # see note in set_led_current
    LED_GREEN_PWM      = (0x0B, 0x02)
    LED_BLUE_PWM       = (0x0B, 0x03)
    LED_CURRENT        = (0x0B, 0x01)
    SOFTWARE_RESET     = (0x08, 0x02)
    DMD_PARK           = (0x06, 0x09)
    PATTERN_EXPOSURE   = (0x1A, 0x29)
    TRIGGER_OUT1       = (0x1A, 0x1D)
    TRIGGER_OUT2       = (0x1A, 0x1E)
    MAIN_STATUS        = (0x1A, 0x0C)


@dataclass
class DLPConfig:
    mode: str = "pattern"
    pattern_source: str = "stream"
    trigger_mode: int = 0          # 0 = VSYNC (streaming), 1/2 = flash modes
    led_red: bool = False
    led_green: bool = True
    led_blue: bool = False
    exposure_us: int = 15000
    period_us: int = 16667


def build_packet(cmd: Tuple[int, int], data: bytes = b"",
                 read: bool = False, reply: bool = True,
                 seq: int = 0) -> List[bytes]:
    """
    Build the HID packet(s) for one DLPC350 command.

    Pure function so the wire format is unit-testable without hardware
    (tests/test_driver_packets.py checks it against pycrafter4500).
    Returns a list of 65-byte packets (report id included).
    """
    cmd2, cmd3 = cmd
    flags = 0x00
    if reply:
        flags |= 0x40
    if read:
        flags |= 0xC0  # read implies reply
    payload_len = 2 + len(data)
    head = bytes([
        0x00,                      # hidapi report id
        flags,
        seq & 0xFF,
        payload_len & 0xFF, (payload_len >> 8) & 0xFF,
        cmd3, cmd2,                # LSB first — CMD3 before CMD2
    ])
    first = head + data[:_FIRST_DATA]
    packets = [first.ljust(_PKT, b"\x00")]
    rest = data[_FIRST_DATA:]
    while rest:
        chunk, rest = rest[:_CONT_DATA], rest[_CONT_DATA:]
        packets.append((b"\x00" + chunk).ljust(_PKT, b"\x00"))
    return packets


def parse_reply(raw: bytes) -> bytes:
    """
    Extract the data field from a 64-byte DLPC350 reply.

    Reply layout: byte0 flags, byte1 seq, byte2-3 length (LE), byte4+ data.
    v0.4 read raw[0] (the flags byte) as data — fixed.
    """
    if len(raw) < 4:
        return b""
    n = raw[2] | (raw[3] << 8)
    return bytes(raw[4:4 + n])


class DLPDriver:
    """Hardware driver for a DLP4500 LightCrafter (DLPC350-based)."""

    def __init__(self, vid: int = DLP4500_VID, pid: int = DLP4500_PID,
                 transport=None):
        self.vid = vid
        self.pid = pid
        self.cfg = DLPConfig()
        self.transport = transport   # PatternTransport for streamed patterns
        self._hid = None
        self._seq = 0
        self._writes_reason = None   # set inside writes_allowed() only

    # ------------------------------------------------------------ lifecycle

    def open(self):
        if os.environ.get(NO_HARDWARE_ENV, "").strip() not in ("", "0"):
            raise RuntimeError(
                f"{NO_HARDWARE_ENV} is set: this process must not open a real "
                "DLPC350 (use MockDLPDriver)")
        try:
            import hid
            self._hid = hid.device()
            self._hid.open(self.vid, self.pid)
            self._hid.set_nonblocking(0)
            log.info("Opened DLPC350: VID=0x%04x PID=0x%04x", self.vid, self.pid)
            return self
        except Exception as e:
            log.warning("Could not open real DLP (%s). Caller should use "
                        "MockDLPDriver.", e)
            raise

    def close(self):
        if self.transport is not None:
            try:
                self.transport.close()
            except Exception:
                pass
        if self._hid is not None:
            try:
                self.park_dmd(True)
            except WriteRefused:
                # parking is a write: outside writes_allowed() the handle is
                # closed and the mirrors stay as they are
                log.info("DLPC350 closed without parking the mirrors "
                         "(writes are not allowed)")
            finally:
                self._hid.close()
                self._hid = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------ transport

    @contextmanager
    def writes_allowed(self, reason: str):
        """Allow writes to the real device inside this `with` block, and only
        there. `reason` says who allowed them and how (it is logged). Reads
        need no permission; Display Mode Select is refused in here too (see
        WRITE GUARD in the module docstring)."""
        reason = str(reason or "").strip()
        if not reason:
            raise ValueError("writes_allowed needs a reason: who allows the "
                             "write, and how")
        before, self._writes_reason = self._writes_reason, reason
        log.warning("DLPC350 writes allowed: %s", reason)
        try:
            yield self
        finally:
            self._writes_reason = before

    def _check_packet(self, first: bytes) -> None:
        """The write guard, on the first packet of a built command (it carries
        the header; continuation packets are raw data). Raises WriteRefused
        before anything is written."""
        if first[1] & 0x80:
            return                              # a read
        cmd = (first[6], first[5])              # on the wire: CMD3, then CMD2
        name = f"0x{cmd[0]:02X}{cmd[1]:02X}"
        if cmd == CMD.DISPLAY_MODE:
            raise WriteRefused(
                f"Display Mode Select ({name}) is never written through this "
                "driver: the projector display (video / pattern sequence) is "
                "switched by a person. Nothing was sent.")
        if self._writes_reason is None:
            raise WriteRefused(
                f"write of command {name} refused: this driver writes to a "
                "real DLPC350 only inside `with driver.writes_allowed(reason)`"
                ". Nothing was sent.")

    def _send(self, cmd: Tuple[int, int], data: bytes = b"",
              read: bool = False) -> bytes:
        """Send one command; always drain and parse the 64-byte reply.
        A write that is not allowed raises WriteRefused with nothing sent."""
        self._seq = (self._seq + 1) & 0xFF
        packets = build_packet(cmd, data, read=read, reply=True,
                               seq=self._seq)
        self._check_packet(packets[0])
        for pkt in packets:
            self._hid.write(pkt)
        raw = bytes(self._hid.read(64, timeout_ms=2000))
        return parse_reply(raw)

    # ------------------------------------------------------------- commands

    def software_reset(self):
        self._send(CMD.SOFTWARE_RESET, bytes([0x01]))
        time.sleep(2.0)

    def park_dmd(self, park: bool = True):
        self._send(CMD.DMD_PARK, bytes([0x01 if park else 0x00]))

    def set_mode(self, mode: str):
        """Display Mode Select. On a real device this is refused, always
        (WriteRefused, nothing sent); the mock driver records it."""
        assert mode in ("video", "pattern")
        self._send(CMD.DISPLAY_MODE, bytes([0x01 if mode == "pattern" else 0x00]))
        self.cfg.mode = mode

    def set_pattern_source(self, source: str):
        """'stream' = 24-bit RGB/FPD port (HDMI on the EVM); 'flash' = stored."""
        assert source in ("stream", "flash")
        self._send(CMD.PATTERN_SOURCE, bytes([0x00 if source == "stream" else 0x03]))
        self.cfg.pattern_source = source

    def set_pattern_trigger_mode(self, mode: int = 0):
        """
        DLPU010G Table 2-53. Mode 0: VSYNC triggers the sequence (streaming).
        In Mode 0 pattern exposure must equal the pattern period.
        Stop the sequence first; validate before restarting.
        """
        assert 0 <= mode <= 4
        self._send(CMD.PATTERN_TRIG_MODE, bytes([mode]))
        self.cfg.trigger_mode = mode

    def set_trigger_out1(self, invert: bool = False,
                         rising_delay: int = 0xBB,
                         falling_delay: int = 0xBB):
        """
        DLPU010G Table 2-54. TRIG_OUT_1 frames each pattern exposure —
        this is the camera sync line. Delay bytes: 0xBB is the zero-delay
        default; each LSB is 107.2 ns, range -20.05 µs .. +2.787 µs.
        Stop sequence first; validate before restarting.
        """
        b0 = 0x02 if invert else 0x00
        self._send(CMD.TRIGGER_OUT1,
                   bytes([b0, rising_delay & 0xFF, falling_delay & 0xFF]))

    def set_leds(self, red: bool, green: bool, blue: bool, sequencer: bool = False):
        b = (int(red) & 1) | ((int(green) & 1) << 1) | ((int(blue) & 1) << 2)
        if sequencer:
            b |= 0x08
        self._send(CMD.LED_ENABLE, bytes([b]))
        self.cfg.led_red, self.cfg.led_green, self.cfg.led_blue = red, green, blue

    def set_led_current(self, red: int, green: int, blue: int):
        """
        LED current as 0-255 PWM values, all three in one command
        (DLPC350 LED current command takes R,G,B together).
        NOTE: values are inverted-polarity on stock LightCrafter 4500
        firmware (255 - x). First-light advice: start ~25%% drive.
        """
        self._send(CMD.LED_CURRENT,
                   bytes([red & 0xFF, green & 0xFF, blue & 0xFF]))

    def set_exposure(self, exposure_us: int, period_us: int):
        data = exposure_us.to_bytes(4, "little") + period_us.to_bytes(4, "little")
        self._send(CMD.PATTERN_EXPOSURE, data)
        self.cfg.exposure_us = exposure_us
        self.cfg.period_us = period_us

    def main_status(self) -> dict:
        d = self._send(CMD.MAIN_STATUS, read=True)
        b = d[0] if d else 0
        return {
            "dmd_parked":        bool(b & 0x01),
            "sequencer_running": bool(b & 0x02),
            "buffer_frozen":     bool(b & 0x04),
            "gamma_enabled":     bool(b & 0x08),
            "raw": hex(b),
        }

    def validate(self, poll_timeout_s: float = 2.0) -> dict:
        """
        Validate Data (DLPU010G Table 2-52). Bit 7 = busy; poll until it
        clears, then interpret bits 0-4 (1 = invalid/warning).
        """
        deadline = time.time() + poll_timeout_s
        b = 0x80
        while time.time() < deadline:
            d = self._send(CMD.VALIDATION, bytes([0x00]), read=True)
            b = d[0] if d else 0x80
            if not (b & 0x80):
                break
            time.sleep(0.05)
        return {
            "busy":              bool(b & 0x80),
            "valid":             not (b & 0x9F),
            "exposure_ok":       not (b & 0x01),
            "lut_patterns_ok":   not (b & 0x02),
            "trigger_out1_ok":   not (b & 0x04),
            "post_vector_ok":    not (b & 0x08),
            "period_margin_ok":  not (b & 0x10),
            "raw":               hex(b),
        }

    # --------------------------------------------------------- LUT sequence

    def load_pattern_sequence(self, patterns: list, bit_depth: int = 8,
                              led: str = "green",
                              trigger_type: int = 0) -> dict:
        """
        Write the pattern LUT for a streamed sequence.

        IMPORTANT: this defines *how* the sequencer interprets incoming
        bit-planes; the pattern images themselves arrive via the video
        port (use the attached PatternTransport). bit_depth now reflects
        the actual pattern data (v0.4 hardcoded 1-bit, silently
        binarizing 8-bit fringes).

        LUT entry (DLPU010G Table 2-65, 3 bytes):
          byte0: bits1:0 trigger type, bits7:2 pattern number
          byte1: bits3:0 bit depth, bits6:4 LED select
          byte2: flags (invert/insert-black/buffer-swap/trig-out-high)
        """
        assert 1 <= bit_depth <= 8
        led_sel = {"red": 0b001, "green": 0b010, "blue": 0b100}[led]
        self.stop_sequence()
        self._send(CMD.LUT_ACCESS, bytes([0x02]))      # open LUT for write
        for idx, _p in enumerate(patterns):
            byte0 = (trigger_type & 0x03) | ((idx & 0x3F) << 2)
            byte1 = (bit_depth & 0x0F) | (led_sel << 4)
            byte2 = 0b00000000
            self._send(CMD.LUT_DATA, bytes([byte0, byte1, byte2]))
        self._send(CMD.LUT_ACCESS, bytes([0x00]))      # close LUT
        return self.validate()

    def start_sequence(self):
        self._send(CMD.PATTERN_START_STOP, bytes([0x02]))

    def pause_sequence(self):
        self._send(CMD.PATTERN_START_STOP, bytes([0x01]))

    def stop_sequence(self):
        self._send(CMD.PATTERN_START_STOP, bytes([0x00]))

    # ------------------------------------------------------ pattern display

    def project_static(self, image: np.ndarray):
        """
        Display one arbitrary pattern. The DLPC350 has no USB frame push
        (DLPU010G §2.4.3) — patterns stream through the video port. This
        delegates to the attached PatternTransport (HDMI fullscreen
        renderer). See drivers/pattern_transport.py.
        """
        if self.transport is None:
            raise RuntimeError(
                "No pattern transport attached. The DLPC350 cannot receive "
                "frames over USB; arbitrary patterns must stream through "
                "the EVM's HDMI/RGB port. Create the rig with "
                "transport=HDMITransport(monitor=...) or pre-load patterns "
                "to flash with the TI LightCrafter GUI."
            )
        self.transport.show(image)


class MockDLPDriver(DLPDriver):
    """numpy-backed simulator. Use for CI tests and laptop development."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_pattern = np.zeros((DMD_HEIGHT, DMD_WIDTH), dtype=np.uint8)
        self._sequence = []
        self._seq_idx = 0
        self._running = False
        self.sent = []          # (cmd, data) log for tests

    def open(self):
        log.info("MockDLPDriver active — no hardware required.")
        return self

    def close(self):
        self._running = False

    def _send(self, cmd, data=b"", read=False):
        self.sent.append((cmd, bytes(data)))
        if cmd == CMD.VALIDATION:
            return b"\x00"      # not busy, all valid
        return b"\x00" * (8 if read else 0)

    def load_pattern_sequence(self, patterns, bit_depth: int = 8,
                              led: str = "green", trigger_type: int = 0):
        self._sequence = [p.copy() for p in patterns]
        self._seq_idx = 0
        if self._sequence:
            self.current_pattern = self._sequence[0]
        return {"busy": False, "valid": True, "raw": "mock"}

    def project_static(self, image):
        self.current_pattern = image.copy()
        if self.transport is not None:
            self.transport.show(image)

    def start_sequence(self):
        self._running = True

    def stop_sequence(self):
        self._running = False

    def advance(self):
        if not self._sequence:
            return None
        self._seq_idx = (self._seq_idx + 1) % len(self._sequence)
        self.current_pattern = self._sequence[self._seq_idx]
        return self.current_pattern

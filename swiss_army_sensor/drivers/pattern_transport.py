"""
Pattern transport â€” how pattern images physically reach the DMD.

The DLPC350 (LightCrafter 4500) accepts pattern data only from the
24-bit RGB / FPD-link video port or from flash (TI DLPU010G Â§2.4.3).
There is no USB frame push. On the EVM the video port is fed by the
HDMI receiver, so the runtime path for arbitrary patterns is:

    pattern array -> fullscreen window on the EVM's HDMI output
                  -> 24-bit RGB port -> sequencer
    VSYNC clocks the sequence       (Pattern Trigger Mode 0)
    TRIG_OUT_1 frames each exposure (wire to camera trigger input)

This module is deliberately controller-agnostic: any DLP EVM with a
video input works the same way, which is most of what makes the package
drop-in across DMD families (DLP4500 here; DLP3010/DLP471TP/DLP472TP
differ only in resolution, taken from hardware.DMD_PROFILES).

Usage (real hardware, Windows 10 host):
    from swiss_army_sensor.drivers.pattern_transport import HDMITransport
    t = HDMITransport(monitor_offset=(1920, 0), size=(912, 1140))
    dlp = DLPDriver(transport=t).open()
    dlp.set_mode("pattern"); dlp.set_pattern_source("stream")
    dlp.set_pattern_trigger_mode(0); dlp.set_trigger_out1()
    dlp.project_static(pattern)

`monitor_offset` is the top-left pixel of the EVM's display in the
Windows virtual desktop (e.g. (1920, 0) when the EVM is to the right of
a 1920-wide primary monitor). Set the EVM display to 912x1140 in
Windows display settings first; the LightCrafter advertises that mode.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Iterable, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


class PatternTransport(ABC):
    """Delivers pattern arrays to the DMD's video input."""

    @abstractmethod
    def show(self, pattern: np.ndarray) -> None:
        """Display one pattern (H x W uint8, grayscale or RGB)."""

    def show_sequence(self, patterns: Iterable[np.ndarray],
                      dwell_s: float = 0.05) -> None:
        for p in patterns:
            self.show(p)
            time.sleep(dwell_s)

    def close(self) -> None:  # pragma: no cover - trivial
        pass



def discover_evm_monitor(preferred_size: Tuple[int, int] = (912, 1140),
                         primary_is_origin: bool = True):
    """Find the LightCrafter HDMI display via EnumDisplayMonitors (Windows).

    Returns (offset_x, offset_y, width, height). Prefers a non-primary monitor
    matching preferred_size (default 912x1140). Falls back to the first
    non-primary monitor. This is the Surfer/Joey proven path that avoids
    projecting onto the wrong display (wallpaper / white≈black miss).
    """
    import sys
    if sys.platform != "win32":
        return (1920, 0, preferred_size[0], preferred_size[1])
    import ctypes

    class RECT(ctypes.Structure):
        _fields_ = [("l", ctypes.c_long), ("t", ctypes.c_long),
                    ("r", ctypes.c_long), ("b", ctypes.c_long)]

    mons = []
    PROC = ctypes.WINFUNCTYPE(
        ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong,
        ctypes.POINTER(RECT), ctypes.c_double)

    def _cb(h, hdc, rect, data):
        r = rect.contents
        mons.append((r.l, r.t, r.r - r.l, r.b - r.t))
        return 1

    ctypes.windll.user32.EnumDisplayMonitors(0, 0, PROC(_cb), 0)
    if not mons:
        return (1920, 0, preferred_size[0], preferred_size[1])

    pw, ph = preferred_size
    seconds = [m for m in mons if not (primary_is_origin and m[0] == 0 and m[1] == 0)]
    if not seconds:
        seconds = mons
    match = [m for m in seconds if abs(m[2] - pw) < 40 and abs(m[3] - ph) < 40]
    ox, oy, mw, mh = (match[0] if match else seconds[0])
    log.info("discover_evm_monitor -> offset (%s,%s) size %sx%s (mons=%s)",
             ox, oy, mw, mh, mons)
    return (ox, oy, mw, mh)

class HDMITransport(PatternTransport):
    """
    Fullscreen OpenCV window on the display output wired to the EVM's
    HDMI input. Requires `opencv-python` (install extra: hardware).

    Frame pacing: the DMD displays one video frame *behind* the stream
    (DLPU010G Â§2.4.2 double-buffering), so after show() allow at least
    two frame periods before treating the pattern as live. settle_s
    defaults to 2 frames at 60 Hz plus margin.
    """

    WINDOW = "swiss-army-sensor-dmd"

    def __init__(self, monitor_offset: Optional[Tuple[int, int]] = None,
                 size: Tuple[int, int] = (912, 1140),
                 settle_s: float = 0.040,
                 auto_discover: bool = True):
        try:
            import cv2  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "HDMITransport requires opencv-python: "
                "pip install opencv-python") from e
        self._cv2 = __import__("cv2")
        self.settle_s = settle_s
        if monitor_offset is None and auto_discover:
            ox, oy, mw, mh = discover_evm_monitor(preferred_size=size)
            monitor_offset = (ox, oy)
            # keep logical DMD size; window uses discovered monitor size
            self._window_size = (mw, mh)
        else:
            if monitor_offset is None:
                monitor_offset = (1920, 0)
            self._window_size = size
        self.monitor_offset = monitor_offset
        self.size = size            # (W, H) pattern / EVM mode
        cv2 = self._cv2
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
        cv2.moveWindow(self.WINDOW, int(monitor_offset[0]), int(monitor_offset[1]))
        ww, wh = self._window_size
        cv2.resizeWindow(self.WINDOW, int(ww), int(wh))
        cv2.setWindowProperty(self.WINDOW, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)
        log.info("HDMITransport fullscreen at offset %s size %s window %s",
                 monitor_offset, size, self._window_size)

    def show(self, pattern: np.ndarray) -> None:
        cv2 = self._cv2
        img = np.asarray(pattern)
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        w, h = self.size
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.imshow(self.WINDOW, img)
        cv2.waitKey(1)              # pump the event loop / flush the frame
        time.sleep(self.settle_s)   # double-buffer latency (see docstring)

    def close(self) -> None:
        try:
            self._cv2.destroyWindow(self.WINDOW)
        except Exception:  # pragma: no cover
            pass


class MockTransport(PatternTransport):
    """Records patterns; used by tests and by MockDLPDriver integration."""

    def __init__(self):
        self.last: Optional[np.ndarray] = None
        self.count = 0

    def show(self, pattern: np.ndarray) -> None:
        self.last = np.asarray(pattern).copy()
        self.count += 1


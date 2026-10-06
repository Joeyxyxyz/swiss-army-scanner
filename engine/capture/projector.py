"""
Projector — the ONLY way patterns reach the DMD, and it is HDMI-only (§9.1).

Patterns are drawn in a fullscreen window on the EVM's extended display
(912x1140 @ 60 Hz); the DLPC350 is never opened over USB. This module
must not import swiss_army_sensor.drivers or hid — tests/test_rules.py
checks that statically and at runtime.

HDMIWindow is ported from the bench-patched HDMITransport of
swiss_army_sensor 0.5.1 (the version the known-good scans ran on; the
in-repo swiss_army_sensor/ source does not carry these fixes):
  * rot90 for the DMD's portrait orientation (projection.rotate90)
  * the hold pumps cv2.waitKey continuously — a bare sleep lets the
    fullscreen window go stale and the camera grabs a blank (§9.2).
    There is deliberately no sleep call anywhere in this file.

The projection chain (gamma precomp -> duty -> polarity -> push twice)
is first_scan_v10's `_project`, with its constants moved into the preset.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import numpy as np

from .. import rules
from ..config import ProjectionCfg

log = logging.getLogger(__name__)


RECT_TOL_PX = 2                    # window rect vs the expected EVM display, per edge


class HDMIWindow:
    """Fullscreen OpenCV window on the EVM display.

    After going fullscreen (and again after the first pattern is painted,
    when the window manager has settled) the window's client rect is read
    with cv2.getWindowImageRect and compared with the expected display:
    `size` at `monitor_offset`. A mismatch (window on the wrong monitor,
    Windows display scaling, the EVM not at 912x1140) is logged as a
    warning and exposed as `rect` / `rect_ok` / `rect_warning` for check
    views — never raised: the camera-side checks (check_signal) judge the
    scan, this only says why it might fail."""

    WINDOW = "swiss-army-scanner-dmd"

    def __init__(self, monitor_offset=(1920, 0), size=(912, 1140),
                 rotate90: bool = False):
        try:
            import cv2
        except ImportError as e:  # pragma: no cover - opencv is a core dependency
            raise ImportError("HDMI projection needs opencv-python "
                              "(pip install opencv-python)") from e
        self._cv2 = cv2
        self.size = tuple(size)            # (W, H) of the EVM display mode
        self.monitor_offset = tuple(monitor_offset)
        self.rotate90 = rotate90
        self.rect: Optional[tuple] = None  # (x, y, w, h) as the window manager reports it
        self.rect_ok: Optional[bool] = None
        self.rect_warning: Optional[str] = None
        self._rect_settled = False
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
        cv2.moveWindow(self.WINDOW, *monitor_offset)
        cv2.setWindowProperty(self.WINDOW, cv2.WND_PROP_FULLSCREEN,
                              cv2.WINDOW_FULLSCREEN)
        cv2.waitKey(1)                     # let the window manager apply fullscreen
        self.check_rect(warn=False)        # may still be settling: re-checked after 1st paint
        log.info("HDMIWindow fullscreen at %s size %s (window rect %s)",
                 monitor_offset, size, self.rect)

    def check_rect(self, warn: bool = True) -> Optional[bool]:
        """Read the window's client rect; record whether it is the expected
        display (size at monitor_offset, within RECT_TOL_PX). Never raises."""
        try:
            x, y, w, h = (int(v) for v in self._cv2.getWindowImageRect(self.WINDOW))
        except Exception as e:             # noqa: BLE001 - old OpenCV / no window backend
            self.rect, self.rect_ok = None, None
            self.rect_warning = f"could not read the DMD window rect ({type(e).__name__}: {e})"
            if warn:
                log.warning("HDMIWindow: %s", self.rect_warning)
            return None
        self.rect = (x, y, w, h)
        (ox, oy), (ew, eh) = self.monitor_offset, self.size
        self.rect_ok = (abs(w - ew) <= RECT_TOL_PX and abs(h - eh) <= RECT_TOL_PX
                        and abs(x - ox) <= RECT_TOL_PX and abs(y - oy) <= RECT_TOL_PX)
        self.rect_warning = None if self.rect_ok else (
            f"DMD window is {w}x{h} at ({x}, {y}), expected {ew}x{eh} at ({ox}, {oy}): the "
            "patterns are scaled or on the wrong monitor. Check that the EVM display is "
            f"{ew}x{eh}@60 at desktop position ({ox}, {oy}) with 100% display scaling "
            "(projection.monitor_offset / size in the preset)")
        if warn and not self.rect_ok:
            log.warning("HDMIWindow: %s", self.rect_warning)
        return self.rect_ok

    def show(self, pattern: np.ndarray) -> None:
        cv2 = self._cv2
        img = np.asarray(pattern)
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        if self.rotate90:
            img = np.rot90(img)            # DMD portrait rotation (bench fix)
        w, h = self.size
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.imshow(self.WINDOW, img)
        cv2.waitKey(1)
        if not self._rect_settled:         # once: the first paint has settled the window
            self._rect_settled = True
            self.check_rect(warn=True)

    def hold(self, seconds: float) -> None:
        """Keep the window painted for `seconds` (pumped, never a bare sleep)."""
        end = time.monotonic() + seconds
        self._cv2.waitKey(1)
        while time.monotonic() < end:
            self._cv2.waitKey(15)

    def close(self) -> None:
        try:
            self._cv2.destroyWindow(self.WINDOW)
        except Exception:  # pragma: no cover
            pass


class MockWindow:
    """Records what would be sent over HDMI; models the bench's DMD.

    The bench DMD shows what is sent (measured 2026-09-22: a projection.invert
    pattern lands inverted, so decode sees flipped polarity). Pass
    hdmi_inverts=True only to model a path that inverts in hardware."""

    def __init__(self, hdmi_inverts: bool = False, rotate90: bool = False,
                 size=(912, 1140)):
        self.hdmi_inverts = hdmi_inverts
        self.rotate90 = rotate90           # modelled exactly like HDMIWindow.show
        self.size = tuple(size)
        self.last: Optional[np.ndarray] = None
        self.count = 0
        self.held_s = 0.0
        # HDMIWindow's window-rect check has no meaning without a real window
        self.rect: Optional[tuple] = None
        self.rect_ok: Optional[bool] = None
        self.rect_warning: Optional[str] = None

    def show(self, pattern: np.ndarray) -> None:
        self.last = np.asarray(pattern).copy()
        self.count += 1

    def hold(self, seconds: float) -> None:
        self.held_s += seconds

    @property
    def physical(self) -> Optional[np.ndarray]:
        """What the mirrors actually show."""
        if self.last is None:
            return None
        img = self.last
        if self.rotate90:
            import cv2
            w, h = self.size
            img = cv2.resize(np.ascontiguousarray(np.rot90(img)), (w, h),
                             interpolation=cv2.INTER_NEAREST)
        return 255 - img if self.hdmi_inverts else img

    def close(self) -> None:
        pass


class Projector:
    def __init__(self, cfg: ProjectionCfg, *, mock: bool):
        rules.check_scan_path_usb_free()
        rules.validate_projection(vars(cfg))
        self.cfg = cfg
        self.zone_mask: Optional[np.ndarray] = None   # DMD-space bool; outside = dark
        self.window = (MockWindow(rotate90=cfg.rotate90, size=cfg.size) if mock else
                       HDMIWindow(cfg.monitor_offset, cfg.size, cfg.rotate90))
        rules.check_transport(self.window)
        if not mock and cfg.warmup_s > 0:
            # the fullscreen window takes ~0.2 s to come up on the EVM; the
            # camera sees desktop / an unpainted window until then
            self.window.show(self.prepare(np.zeros((cfg.size[1], cfg.size[0]), np.uint8)))
            self.window.hold(cfg.warmup_s)

    def prepare(self, pattern: np.ndarray) -> np.ndarray:
        """Pattern as sent over HDMI: duty -> polarity -> gamma precomp.

        Gamma precompensation is applied LAST, to the value actually sent:
        the projector's response acts on the sent level, so correcting the
        level before inversion (the old order) was wrong for invert=True
        (the bench presets). gamma=1 is bit-identical to the old chain."""
        pat = np.asarray(pattern)
        if pat.dtype != np.uint8:
            pat = np.clip(pat, 0, 255).astype(np.uint8)
        c = self.cfg
        if c.duty != 1.0:
            pat = (pat.astype(np.float32) * c.duty).astype(np.uint8)
        sent = (255 - pat) if c.invert else pat
        if c.gamma != 1.0:
            sent = (255.0 * (sent.astype(np.float32) / 255.0) ** (1.0 / c.gamma)).astype(np.uint8)
        return sent

    def project(self, pattern: np.ndarray, raw: bool = False) -> None:
        """raw=True sends the image exactly as given (e.g. a BGR colour flood
        for code-reading illumination) — no gamma/duty/polarity chain."""
        # re-checked per pattern: if the USB driver (e.g. dmd_control) gets
        # loaded mid-scan in this process, the scan stops at the next frame
        rules.check_scan_path_usb_free()
        if raw:
            shown = np.clip(np.asarray(pattern), 0, 255).astype(np.uint8)
        else:
            shown = self.prepare(pattern)
        if self.zone_mask is not None:
            # applied AFTER the polarity chain so outside-zone mirrors are
            # physically dark on any bench (invert/duty would otherwise light them)
            m = self.zone_mask if shown.ndim == 2 else self.zone_mask[..., None]
            shown = np.where(m, shown, 0).astype(np.uint8)
        self.window.show(shown)
        if self.cfg.push_twice:            # DMD shows HDMI one frame behind
            self.window.show(shown)
        self.window.hold(max(self.cfg.settle_s, self.cfg.min_hold_s))

    def close(self) -> None:
        self.window.close()

    def __getattr__(self, name):
        if name in rules.USB_DISPLAY_COMMANDS:
            rules.refuse_usb_command(name)
        raise AttributeError(name)

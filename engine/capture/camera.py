"""
Camera: FLIR Blackfly S (PySpin) + mock.

The mock wraps swiss_army_sensor's MockCameraDriver (loaded file-by-file
via engine._seed, never through the package __init__), with the
MockBench providing the parallax-correct observation.
"""

from __future__ import annotations

import logging

import numpy as np

from .. import _seed
from ..config import VIDEO_FRAME_US, CameraCfg, frame_locked

log = logging.getLogger(__name__)


class _DMDTap:
    """Gives MockCameraDriver the `current_pattern` it expects from a DLP
    driver — read from the projector window, so no DLP object is needed."""

    def __init__(self, window):
        self._window = window

    @property
    def current_pattern(self):
        return self._window.physical


class MockCamera:
    def __init__(self, cfg: CameraCfg, bench, window, noise_sigma: float = 1.5):
        cd = _seed.load("camera_driver")
        self.cfg = cfg
        self._bench = bench
        self.exposure_us = float(cfg.exposure_us)
        self._drv = cd.MockCameraDriver(dlp=_DMDTap(window), obs=self._observe,
                                        noise_sigma=noise_sigma).open()
        # The seed driver reseeds to 42 on every construction, so repeated mock
        # captures were bit-identical (a repeatability test measured 0 noise and
        # a good-bank spread collapsed to 0). Draw each rig's noise stream from
        # the bench's generator instead: still reproducible per bench seed.
        rng = getattr(bench, "_rng", None)
        if rng is not None:
            self._drv._rng = np.random.default_rng(int(rng.integers(0, 2**63 - 1)))

    def _observe(self, pattern):
        """Exposure model: the bench's counts are calibrated at the preset's
        exposure; light (projector + ambient) integrates linearly with time.
        Scaling happens BEFORE the driver adds noise and clips to 0..255, so a
        long exposure saturates here exactly as the Blackfly does on the bench."""
        frame = self._bench.render(pattern)
        scale = self.exposure_us / float(self.cfg.exposure_us)
        return frame if scale == 1.0 else frame * np.float32(scale)

    def set_exposure_us(self, us) -> None:
        """Change integration time for the next grabs; never touches cfg
        (the preset), so a sweep can't leak into later scans."""
        us = float(us)
        if not us > 0:
            raise ValueError(f"exposure must be > 0 us, got {us!r}")
        self.exposure_us = us

    def flush(self) -> None:
        pass

    def grab(self) -> np.ndarray:
        return self._drv.capture()

    def close(self) -> None:
        self._drv.close()


GRAB_TIMEOUT_MS = 2000
INCOMPLETE_RETRIES = 3       # an incomplete frame (USB3 packet loss) is re-grabbed this often


def _set_max(node) -> None:
    node.SetValue(node.GetMax())


def _reset_steps(drv, PySpin) -> list:
    """(name, action) pairs that put the Blackfly back into the state every
    known-good scan ran in, whatever a previous program (SpinView, another
    script) left behind: full frame, 1x1 binning, unmirrored, linear
    response, free-running at its own maximum rate. Binning goes first (it
    caps Width/Height), then the offsets, then the size maxima.

    The actions look the camera up on the driver each time: PySpin refuses
    to release its system while a Python reference to the camera is alive,
    and a traceback frame holding one would keep it alive."""
    return [
        ("BinningHorizontal", lambda: drv._cam.BinningHorizontal.SetValue(1)),
        ("BinningVertical", lambda: drv._cam.BinningVertical.SetValue(1)),
        ("OffsetX", lambda: drv._cam.OffsetX.SetValue(0)),
        ("OffsetY", lambda: drv._cam.OffsetY.SetValue(0)),
        ("Width", lambda: _set_max(drv._cam.Width)),
        ("Height", lambda: _set_max(drv._cam.Height)),
        ("ReverseX", lambda: drv._cam.ReverseX.SetValue(False)),
        ("ReverseY", lambda: drv._cam.ReverseY.SetValue(False)),
        ("GammaEnable", lambda: drv._cam.GammaEnable.SetValue(False)),
        ("AcquisitionFrameRateEnable",
         lambda: drv._cam.AcquisitionFrameRateEnable.SetValue(False)),
        ("TriggerMode", lambda: drv._cam.TriggerMode.SetValue(PySpin.TriggerMode_Off)),
    ]


def _release_driver(drv) -> None:
    """Best-effort close of a half-set-up seed CameraDriver (error path only).
    The seed's close() ends acquisition and de-inits in ONE try, so a camera
    that was not acquiring would skip DeInit; here each step runs on its own
    and nothing raises (the error that got us here is the one to report)."""
    for step in ("EndAcquisition", "DeInit"):
        try:
            if getattr(drv, "_cam", None) is not None:
                getattr(drv._cam, step)()
        except Exception:               # noqa: BLE001 - not acquiring / already de-inited
            pass
    try:
        drv.close()
    except Exception as e:              # noqa: BLE001 - cleanup must not mask the cause
        log.warning("camera driver close failed: %s: %s", type(e).__name__, e)


PYSPIN_MISSING = ("real rig needs PySpin (FLIR Spinnaker SDK) in this Python ({exe}); set "
                  "SCAN_PYTHON to the env that has it, or use --mock")


def require_pyspin():
    """Import PySpin, or fail with the one line an operator can act on (the
    launchers run %SCAN_PYTHON% / the `python` on PATH, which may be an env
    without the vendor wheel)."""
    try:
        import PySpin
    except ImportError as e:
        import sys
        raise ModuleNotFoundError(PYSPIN_MISSING.format(exe=sys.executable),
                                  name="PySpin") from e
    return PySpin


class RealCamera:
    """FLIR Blackfly S in free-run (first_scan_v10's setup), via the
    bench-patched swiss_army_sensor CameraDriver. No trigger wiring.

    After open, camera state a previous program may have left (ROI, binning,
    mirroring, gamma, frame-rate cap, trigger) is reset one node at a time;
    `reset_state` records what was reset and what failed (a node the model
    lacks, or one that is locked). Any failure after the driver opened closes
    it before the error propagates, so a half-set-up camera is never left
    held by this process. Code paths here are PySpin-only (the bench)."""

    def __init__(self, cfg: CameraCfg):
        require_pyspin()
        cd = _seed.load("camera_driver")
        drv = cd.CameraDriver()
        drv.cfg.trigger_mode = "free"
        drv.cfg.exposure_us = cfg.exposure_us
        drv.cfg.gain_db = float(cfg.gain_db)
        self.cfg = cfg
        self.reset_state = {"reset": [], "failed": {}}
        self.incomplete_frames = 0          # incomplete frames re-grabbed since open
        drv.open()
        try:
            import PySpin
            drv._cam.EndAcquisition()
            for name, action in _reset_steps(drv, PySpin):
                try:
                    action()
                    self.reset_state["reset"].append(name)
                except Exception as e:      # noqa: BLE001 - best effort, recorded
                    self.reset_state["failed"][name] = f"{type(e).__name__}: {e}"
            if self.reset_state["failed"]:
                log.warning("camera reset after open: could not reset %s", "; ".join(
                    f"{k} ({v})" for k, v in self.reset_state["failed"].items()))
            # OldestFirst (the default) queues ~a dozen frames of the PREVIOUS
            # pattern during each projector hold; flushing a few isn't enough and
            # scan frames lag the pattern (bench-measured 2026-09-22). NewestOnly
            # makes the first grab after a hold a fresh frame.
            drv._cam.TLStream.StreamBufferHandlingMode.SetValue(
                PySpin.StreamBufferHandlingMode_NewestOnly)
            drv._cam.BeginAcquisition()
        except BaseException:
            _release_driver(drv)
            raise
        self._drv = drv

    def set_exposure_us(self, us) -> None:
        """Change integration time mid-acquisition (scan-assistant sweeps).
        Frames already in flight carry the old exposure; FreeRunSync flushes
        after every projection, so they are never used. cfg (the preset) is
        left alone — the next rig open re-applies the preset's exposure.
        An exposure that is not a whole number of HDMI video frames is
        allowed but logged: each frame then catches a different slice of the
        DMD's R/G/B + PWM bit-plane sequence (config.frame_locked)."""
        us = float(us)
        if not us > 0:
            raise ValueError(f"exposure must be > 0 us, got {us!r}")
        try:
            self._drv._cam.ExposureTime.SetValue(float(us))
        except Exception as e:
            raise ValueError(f"camera rejected exposure {us:g} us: {e}") from e
        if not frame_locked(us):
            log.warning("exposure %g us is not a whole number of %.0f us HDMI video frames "
                        "(e.g. %d or %d us): gray levels and colour floods flicker frame to "
                        "frame", us, VIDEO_FRAME_US, round(VIDEO_FRAME_US),
                        round(2 * VIDEO_FRAME_US))

    def _next_frame(self) -> np.ndarray:
        """One complete Mono8 frame; an incomplete one (USB3 packet loss) is
        released and re-grabbed up to INCOMPLETE_RETRIES times before raising."""
        status = None
        for _ in range(1 + INCOMPLETE_RETRIES):
            img = self._drv._cam.GetNextImage(GRAB_TIMEOUT_MS)
            try:
                if not img.IsIncomplete():
                    return img.GetNDArray().copy()
                self.incomplete_frames += 1
                try:
                    status = img.GetImageStatus()
                except Exception:           # noqa: BLE001 - status is only for the message
                    status = None
            finally:
                img.Release()
        raise RuntimeError(
            f"camera delivered {1 + INCOMPLETE_RETRIES} incomplete frames in a row"
            + (f" (image status {status})" if status is not None else "")
            + ": USB3 bandwidth / cable - use a USB3 port on its own controller, a shorter "
              "cable, and close other programs streaming from the camera")

    def flush(self) -> None:
        # DMD video lags a frame and the camera buffer is deeper than one grab
        self._next_frame()

    def grab(self) -> np.ndarray:
        return self._next_frame()

    def close(self) -> None:
        self._drv.close()


def open_camera(cfg: CameraCfg, *, mock: bool, bench=None, window=None):
    if mock:
        return MockCamera(cfg, bench, window)
    return RealCamera(cfg)

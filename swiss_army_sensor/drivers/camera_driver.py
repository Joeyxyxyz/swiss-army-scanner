"""
Camera driver.

Real implementation targets the FLIR Blackfly S BFS-U3-16S2M-CS (Sony
IMX273, 1440x1080, hardware-trigger via 6-pin Hirose HR10A) via PySpin.
A MockCameraDriver simulates a capture by integrating the projected DMD
pattern with a configurable scene image (or pure synthetic scenes).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Callable

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class CamConfig:
    width: int = 1440
    height: int = 1080
    exposure_us: int = 28000
    gain_db: float = 0.0
    trigger_mode: str = "hardware"
    pixel_format: str = "Mono8"


class CameraDriver:
    """FLIR Spinnaker / PySpin adapter (real hardware)."""

    def __init__(self):
        self.cfg = CamConfig()
        self._sys = None
        self._cam = None

    def open(self, serial: Optional[str] = None):
        import PySpin
        self._sys = PySpin.System.GetInstance()
        cams = self._sys.GetCameras()
        if cams.GetSize() == 0:
            cams.Clear()
            self._sys.ReleaseInstance()
            raise RuntimeError("No FLIR cameras found.")
        self._cam = cams[0] if serial is None else next(
            c for c in cams if c.GetUniqueID() == serial)
        self._cam.Init()
        self._configure(PySpin)
        return self

    def _configure(self, PySpin):
        if self._cam.PixelFormat.GetAccessMode() == PySpin.RW:
            self._cam.PixelFormat.SetValue(PySpin.PixelFormat_Mono8)
        elif int(self._cam.PixelFormat.GetValue()) != int(PySpin.PixelFormat_Mono8):
            raise RuntimeError('PixelFormat locked and not Mono8')
        self._cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
        self._cam.ExposureTime.SetValue(self.cfg.exposure_us)
        self._cam.GainAuto.SetValue(PySpin.GainAuto_Off)
        self._cam.Gain.SetValue(self.cfg.gain_db)
        if self.cfg.trigger_mode == "hardware":
            self._cam.TriggerMode.SetValue(PySpin.TriggerMode_Off)
            self._cam.TriggerSource.SetValue(PySpin.TriggerSource_Line0)
            self._cam.TriggerActivation.SetValue(PySpin.TriggerActivation_RisingEdge)
            self._cam.TriggerMode.SetValue(PySpin.TriggerMode_On)
        self._cam.BeginAcquisition()

    def capture(self, timeout_ms: int = 1000) -> np.ndarray:
        img = self._cam.GetNextImage(timeout_ms)
        if img.IsIncomplete():
            img.Release()
            raise RuntimeError("Incomplete frame.")
        arr = img.GetNDArray().copy()
        img.Release()
        return arr

    def close(self):
        if self._cam is not None:
            try:
                self._cam.EndAcquisition()
                self._cam.DeInit()
            except Exception:
                pass
            del self._cam
        if self._sys is not None:
            self._sys.ReleaseInstance()

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()


class MockCameraDriver(CameraDriver):
    """Multiplies the current DMD pattern by a scene and adds noise."""

    def __init__(
        self,
        scene: Optional[np.ndarray] = None,
        dlp=None,
        noise_sigma: float = 1.5,
        obs: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        super().__init__()
        self.scene = scene
        self.dlp = dlp
        self.noise_sigma = noise_sigma
        self.obs = obs
        self._rng = np.random.default_rng(42)

    def open(self, serial=None):
        log.info("MockCameraDriver active.")
        return self

    def close(self):
        pass

    def capture(self, timeout_ms: int = 1000) -> np.ndarray:
        if self.obs is not None:
            pat = self.dlp.current_pattern if self.dlp is not None else None
            img = self.obs(pat)
        else:
            img = self._default_obs()
        img = img.astype(np.float32)
        img += self._rng.normal(0, self.noise_sigma, img.shape)
        return np.clip(img, 0, 255).astype(np.uint8)

    def _default_obs(self):
        h, w = self.cfg.height, self.cfg.width
        if self.dlp is None:
            return np.zeros((h, w), dtype=np.float32)
        pat = self.dlp.current_pattern.astype(np.float32) / 255.0
        from .._utils import resample_nearest
        pat_cam = resample_nearest(pat, (h, w))
        scene = self.scene if self.scene is not None else np.full((h, w), 0.5, dtype=np.float32)
        return pat_cam * scene * 255.0

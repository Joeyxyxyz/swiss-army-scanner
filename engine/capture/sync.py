"""
Projector/camera synchronisation.

FreeRunSync is what runs today (first_scan_v10's `shoot`): project ->
pumped hold -> flush stale camera buffers -> average N grabs.

Hardware-trigger mode would need the DLPC350 in USB pattern mode. Per
§9.1 that is a separate, explicitly-entered mode with its own guard,
never the default — it is not available yet (build step 10).

No trigger at all: capture/softsync.py (FlashFreeRunSync) records the
flash loop a human plays in the 4500 MKII GUI with the camera free-running
and sorts the frames in software; it opens the camera only.
"""

from __future__ import annotations

import numpy as np


class FreeRunSync:
    def __init__(self, projector, camera, average: int = 1, flush: int = 4):
        self.projector = projector
        self.camera = camera
        self.average = average
        self.flush = flush

    def shoot(self, pattern: np.ndarray, raw: bool = False) -> np.ndarray:
        self.projector.project(pattern, raw=raw)
        for _ in range(self.flush):
            self.camera.flush()
        acc = self.camera.grab().astype(np.float32)
        for _ in range(self.average - 1):
            acc += self.camera.grab().astype(np.float32)
        return acc / self.average

    def grab(self, pump_s: float = 0.03) -> np.ndarray:
        """One fresh frame of whatever is projected now (live view): keep the
        window painted for pump_s (pumped hold, never a sleep), then grab."""
        self.projector.window.hold(pump_s)
        return self.camera.grab().astype(np.float32)

    def close(self) -> None:
        self.camera.close()
        self.projector.close()


class HardwareTriggerSync:  # pragma: no cover - build step 10
    def __init__(self, *a, **k):
        raise NotImplementedError(
            "hardware-trigger sync needs DLPC350 USB pattern mode; it will be a "
            "separate, explicitly-entered, guarded mode (build step 10). "
            "The scan path stays HDMI-only.")

"""
Mock of the flash soft-sync bench: the DLPC350 playing 1-bit patterns from
flash on its internal timer (MockFlashLoop), seen by a free-running camera
(MockFreeRunCamera) with its own clock, exposure, frame rate, phase, jitter
and noise.

The camera integrates whatever the DMD shows during [t, t + E]: a frame
inside one pattern exposure is that pattern; a frame across a pattern change
is a mixture of the two (and of the dark gap, when period > exposure). The
projector defocus is a Gaussian blur of the DMD image (sigma in columns,
twice that in rows: the diamond DMD's rows are half a column pitch apart),
which is what turns dithered stripes into sinusoids.

Used by the api in mock mode (the simulated human loaded the pack and set
the GUI up: the loop plays the pack's own unpacked bit-planes, or the
layout's patterns, at the pack's timing; `bench.flash_gui` / `bench.
flash_camera` dicts simulate a GUI mistake / camera trouble, see
mock_camera) and by the tests (loop not running, wrong order, a pattern
missing, stopped mid-way, dropped camera frames). Nothing here touches
hardware.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from . import softsync


@dataclass
class MockFlashLoop:
    """The DLP's internal-timer loop. `patterns` are DMD images in LAYOUT order;
    `order` is the play order (indices into patterns; default all, in order)."""
    patterns: list
    exposure_us: float
    period_us: float
    order: Optional[list] = None
    running: bool = True                 # False: one static image (static_index) instead
    static_index: int = 0
    start_us: Optional[float] = None     # DLP time at camera time 0 (None: random)
    gap_us: float = 0.0                  # dark time added at the end of every loop
    stop_after_us: Optional[float] = None  # the sequence stops (Play Once on): dark after
    clock_ppm: float = 0.0               # DLP clock rate error vs the camera clock
    defocus_px: float = 1.2              # Gaussian sigma, DMD columns (rows: x2)
    seed: int = 0

    def __post_init__(self):
        if self.period_us < self.exposure_us:
            raise ValueError("period must be >= exposure")
        if self.start_us is None:
            self.start_us = float(np.random.default_rng(self.seed).uniform(0, self.loop_us()))

    def play(self) -> list:
        return list(range(len(self.patterns))) if self.order is None else list(self.order)

    def loop_us(self) -> float:
        return len(self.play()) * float(self.period_us) + float(self.gap_us)

    def weights(self, t0_us: float, t1_us: float) -> dict:
        """{pattern index: fraction of the camera window [t0, t1] it was shown}."""
        if not self.running:
            return {int(self.static_index): 1.0}
        k = 1.0 + self.clock_ppm * 1e-6
        a, b = (t0_us + self.start_us) * k, (t1_us + self.start_us) * k
        span = b - a
        L = self.loop_us()
        play = self.play()
        P, X = float(self.period_us), float(self.exposure_us)
        out = {}
        stop = None if self.stop_after_us is None else (self.stop_after_us + self.start_us) * k
        for n in range(int(math.floor(a / L)), int(math.floor(b / L)) + 1):
            base = n * L
            for i, idx in enumerate(play):
                s = base + i * P
                e = s + X
                if stop is not None and s >= stop:
                    continue
                if stop is not None:
                    e = min(e, stop)
                ov = min(e, b) - max(s, a)
                if ov > 0:
                    out[idx] = out.get(idx, 0.0) + ov / span
        return out


def blurred(pattern: np.ndarray, defocus_px: float) -> np.ndarray:
    import cv2
    p = np.asarray(pattern, np.float32)
    if defocus_px <= 0:
        return p
    return cv2.GaussianBlur(p, (0, 0), sigmaX=float(defocus_px), sigmaY=2.0 * float(defocus_px),
                            borderType=cv2.BORDER_REPLICATE)


@dataclass
class MockFreeRunCamera:
    """Free-running camera over `render` (DMD image -> camera frame incl. ambient;
    render(None) = ambient only), e.g. MockBench.render. Brightness: the rendered
    counts are what a `ref_exposure_us` exposure collects; light and ambient
    integrate linearly with the exposure. Noise: read + shot, 8-bit, clipped."""
    render: Callable
    loop: MockFlashLoop
    full_shape: tuple = softsync.SENSOR_SHAPE
    ref_exposure_us: Optional[float] = None   # None: the configured exposure (no scaling)
    level: float = 0.75                       # scales the rendered signal (LED current)
    jitter_us: float = 2.0                    # device timestamp / exposure-start jitter
    read_noise_dn: float = 1.5
    shot_dn: float = 0.08                     # variance per DN of signal
    drop_frames: tuple = ()                   # frame indices the "USB" loses
    fps_max: Optional[float] = None           # None: softsync.estimate_max_fps for the ROI
    seed: int = 0
    _cache: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)
        self._k = 0
        self._fid = 0
        self.applied = {}
        self.started = False
        self.stopped = False

    # the PySpinFreeRunCamera interface --------------------------------------
    def configure(self, *, roi, exposure_us, fps, gain_db=None, buffers=64) -> dict:
        x, y, w, h = (int(v) for v in roi)
        H, W = self.full_shape
        if x < 0 or y < 0 or x + w > W or y + h > H:
            raise ValueError(f"ROI {roi} is off the {W}x{H} sensor")
        est = softsync.estimate_max_fps(w, h, exposure_us)["fps"]
        fmax = est if self.fps_max is None else min(est, float(self.fps_max))
        self.roi, self.exposure_us = (x, y, w, h), float(exposure_us)
        self.fps = min(float(fps), fmax)
        self.applied = {"roi": [x, y, w, h], "exposure_us": self.exposure_us, "fps": self.fps,
                        "fps_max": fmax, "gain_db": gain_db, "buffers": int(buffers),
                        "buffer_mode": "OldestFirst", "mock": True}
        return self.applied

    def start(self) -> None:
        self.started = True
        self._k = 0
        self._fid = 0

    def _signal(self, idx):
        if idx not in self._cache:
            x, y, w, h = self.roi
            amb = self._ambient()
            if idx is None:
                img = amb
            else:
                full = self.render(blurred(self.loop.patterns[idx], self.loop.defocus_px))
                img = np.asarray(full, np.float32)[y:y + h, x:x + w] - amb
            self._cache[idx] = img
        return self._cache[idx]

    def _ambient(self):
        if "amb" not in self._cache:
            x, y, w, h = self.roi
            self._cache["amb"] = np.asarray(self.render(None), np.float32)[y:y + h, x:x + w]
        return self._cache["amb"]

    def frame_at(self, t_us: float) -> np.ndarray:
        """Noise-free counts for an exposure starting at camera time t_us."""
        E = self.exposure_us
        scale = 1.0 if self.ref_exposure_us is None else E / float(self.ref_exposure_us)
        img = self._ambient() * scale
        for idx, wgt in self.loop.weights(t_us, t_us + E).items():
            img = img + (self.level * scale * wgt) * self._signal(idx)
        return img

    def next_frame(self, timeout_ms):
        while self._k in set(self.drop_frames):
            self._k += 1
            self._fid += 1
        k = self._k
        self._k += 1
        fid = self._fid
        self._fid += 1
        t_us = k * 1e6 / self.fps + float(self._rng.normal(0, self.jitter_us))
        img = self.frame_at(t_us)
        sd = np.sqrt(self.read_noise_dn ** 2 + self.shot_dn * np.clip(img, 0, None))
        img = img + self._rng.standard_normal(img.shape).astype(np.float32) * sd
        img = np.clip(np.rint(img), 0, 255).astype(np.uint8)
        dev_ns = int(round(t_us * 1e3))
        host_ns = dev_ns + int(abs(self._rng.normal(900e3, 300e3)))
        return img, fid, dev_ns, host_ns

    def stop(self) -> None:
        self.stopped = True

    def close(self) -> None:
        self.stop()

    # a direct (synchronous) capture of each pattern, for comparisons --------
    def direct(self, idx, n_avg: int = 1) -> np.ndarray:
        """The ROI image of pattern idx held on the DMD for the whole exposure, averaged
        over n_avg noisy frames (what an ideal hardware-triggered capture gets)."""
        E = self.exposure_us
        scale = 1.0 if self.ref_exposure_us is None else E / float(self.ref_exposure_us)
        clean = self._ambient() * scale + self.level * scale * self._signal(idx)
        acc = np.zeros(clean.shape, np.float64)
        for _ in range(int(n_avg)):
            sd = np.sqrt(self.read_noise_dn ** 2 + self.shot_dn * np.clip(clean, 0, None))
            f = clean + self._rng.standard_normal(clean.shape).astype(np.float32) * sd
            acc += np.clip(np.rint(f), 0, 255)
        return (acc / int(n_avg)).astype(np.float32)


def pack_patterns(pack_dir, variant: Optional[str] = None) -> list:
    """The DMD images a fast-scan pack's flash image holds, in play order: its BMP read
    back and its bit-planes unpacked with tools/make_flash_pack.py (the same code that
    packed them; the BMP hashes are checked). variant None: the first image."""
    import sys
    from pathlib import Path
    root = str(Path(__file__).resolve().parents[2])
    if root not in sys.path:
        sys.path.insert(0, root)
    from tools.make_flash_pack import load_pack, unpack
    m, images = load_pack(Path(pack_dir))
    seq = sorted(m["sequence"], key=lambda e: e.get("seq", 0))
    variants = list(m.get("variants") or [])
    want = variant or (variants[0] if variants else None)
    if want is not None:
        seq = [e for e in seq if e.get("variant") == want]
    return unpack(images, seq)


def mock_camera(bench, layout, *, zone_mask=None, gui: Optional[dict] = None,
                camera: Optional[dict] = None, seed: int = 0,
                patterns: Optional[list] = None) -> MockFreeRunCamera:
    """The simulated bench after the simulated human loaded the pack and pressed Play
    in the GUI: MockFlashLoop plays the layout's patterns (layout_patterns; zone_mask
    = the zone variant) at the layout's pattern exposure / period, seen by a
    MockFreeRunCamera over bench.render.

    gui: what the human did differently in the GUI, MockFlashLoop fields -
    {"running": False} (not playing), {"order": [...]} (patterns in another order /
    one missing), {"stop_after_us": t} (stopped mid-recording), {"period_us": ...,
    "exposure_us": ...} (other timing entered), {"gap_us": ...}, {"clock_ppm": ...},
    {"defocus_px": ...}. camera: MockFreeRunCamera fields (noise, jitter, drops).
    patterns: the DMD images to play (e.g. pack_patterns of the pack the human loaded);
    default layout_patterns(layout, zone_mask)."""
    gui = dict(gui or {})
    P = float(gui.pop("period_us", None) or layout.pattern_period_us or softsync.FASTSCAN_PERIOD_US)
    X = float(gui.pop("exposure_us", None) or layout.pattern_exposure_us or P)
    pats = list(patterns) if patterns is not None else softsync.layout_patterns(layout, zone_mask)
    if len(pats) != layout.n:
        raise ValueError(f"{len(pats)} patterns for a {layout.n}-pattern layout")
    rng = np.random.default_rng(seed)
    loop = MockFlashLoop(pats, min(X, P), P, seed=int(rng.integers(0, 2**31 - 1)), **gui)
    cam_kw = {"seed": int(rng.integers(0, 2**31 - 1)), **(camera or {})}
    return MockFreeRunCamera(bench.render, loop, full_shape=tuple(bench.cam_shape), **cam_kw)

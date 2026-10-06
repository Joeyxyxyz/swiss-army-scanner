"""Decoders — turn captured frames into structured sensor readings."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import math
import numpy as np


@dataclass
class BarcodeResult:
    decoded: list
    n_codes: int

def decode_barcode(frame: np.ndarray) -> BarcodeResult:
    decoded = []
    try:
        from pyzbar.pyzbar import decode as zbar_decode
        for d in zbar_decode(frame):
            decoded.append({"data": d.data.decode(errors="replace"),
                            "type": d.type,
                            "polygon": [(p.x, p.y) for p in d.polygon]})
    except ImportError:
        pass
    try:
        from pylibdmtx.pylibdmtx import decode as dmtx_decode
        for d in dmtx_decode(frame):
            decoded.append({"data": d.data.decode(errors="replace"),
                            "type": "DATAMATRIX", "polygon": []})
    except ImportError:
        pass
    return BarcodeResult(decoded=decoded, n_codes=len(decoded))


@dataclass
class PhotoelectricResult:
    intensity_mean: float
    intensity_max: float
    triggered: bool
    threshold: float

def decode_photoelectric(frame, threshold: float = 80.0, roi=None) -> PhotoelectricResult:
    if roi:
        y0, x0, y1, x1 = roi
        frame = frame[y0:y1, x0:x1]
    return PhotoelectricResult(
        intensity_mean=float(frame.mean()),
        intensity_max=float(frame.max()),
        triggered=bool(frame.mean() >= threshold),
        threshold=threshold,
    )


@dataclass
class DistanceResult:
    spot_x: float
    spot_y: float
    distance_mm: float
    confidence: float

def decode_distance(frame, baseline_mm: float = 50.0, focal_px: float = 1800.0,
                    reference_x: float = 720.0) -> DistanceResult:
    """Triangulation distance from a projected point's centroid displacement."""
    blurred = _box_blur(frame, 5)
    thr = blurred.max() * 0.6
    mask = blurred >= thr
    if mask.sum() == 0:
        return DistanceResult(0, 0, float("inf"), 0.0)
    ys, xs = np.where(mask)
    w = blurred[mask].astype(np.float32)
    cx = float((xs * w).sum() / w.sum())
    cy = float((ys * w).sum() / w.sum())
    disparity = cx - reference_x
    distance = baseline_mm * focal_px / abs(disparity) if abs(disparity) >= 0.5 else float("inf")
    conf = float(w.sum() / (mask.sum() + 1e-9) / 255.0)
    return DistanceResult(cx, cy, distance, conf)


@dataclass
class ProfilometryResult:
    phase: np.ndarray
    modulation: np.ndarray
    height_map: Optional[np.ndarray]

def decode_phase_shift(frames: list) -> ProfilometryResult:
    """N-step phase-shifting (Zhang 2018, Eq 4)."""
    N = len(frames)
    stack = np.stack([f.astype(np.float32) for f in frames], axis=0)
    k = np.arange(N).reshape(N, 1, 1)
    s = (stack * np.sin(2 * math.pi * k / N)).sum(0)
    c = (stack * np.cos(2 * math.pi * k / N)).sum(0)
    phase = np.arctan2(-s, c)
    modulation = 2.0 / N * np.sqrt(s * s + c * c)
    return ProfilometryResult(phase=phase, modulation=modulation, height_map=None)


@dataclass
class EdgeResult:
    edge_map: np.ndarray
    edge_strength: float

def decode_edges(frame_grad_x_pos, frame_grad_x_neg, frame_grad_y_pos, frame_grad_y_neg) -> EdgeResult:
    """Differential gradient capture: subtract complementary patterns."""
    gx = frame_grad_x_pos.astype(np.float32) - frame_grad_x_neg.astype(np.float32)
    gy = frame_grad_y_pos.astype(np.float32) - frame_grad_y_neg.astype(np.float32)
    mag = np.sqrt(gx * gx + gy * gy)
    return EdgeResult(edge_map=np.clip(mag, 0, 255).astype(np.uint8),
                      edge_strength=float(mag.mean()))


@dataclass
class ColorResult:
    r: float; g: float; b: float
    chromaticity: tuple
    nearest: str

_COLOR_REFS = {
    "red":     (220, 30, 30),  "green": (30, 200, 60),  "blue": (30, 60, 220),
    "yellow":  (230, 220, 40), "cyan":  (40, 220, 220), "magenta": (220, 40, 220),
    "white":   (240, 240, 240),"gray":  (128, 128, 128),"black": (20, 20, 20),
    "orange":  (240, 140, 30), "purple":(140, 60, 200), "brown": (130, 80, 40),
}

def decode_color(frame_r, frame_g, frame_b, roi=None) -> ColorResult:
    if roi:
        y0, x0, y1, x1 = roi
        frame_r, frame_g, frame_b = frame_r[y0:y1, x0:x1], frame_g[y0:y1, x0:x1], frame_b[y0:y1, x0:x1]
    r, g, b = float(frame_r.mean()), float(frame_g.mean()), float(frame_b.mean())
    s = r + g + b + 1e-9
    chrom = (r / s, g / s, b / s)
    nearest = min(_COLOR_REFS, key=lambda k: sum((meas - ref) ** 2
                  for meas, ref in zip((r, g, b), _COLOR_REFS[k])))
    return ColorResult(r=r, g=g, b=b, chromaticity=chrom, nearest=nearest)


@dataclass
class SinglePixelResult:
    image: np.ndarray
    snr_db: float

def decode_single_pixel(intensities: list, n_per_side: int) -> SinglePixelResult:
    """Inverse Walsh-Hadamard transform on bucket measurements.

    Binary (0/255) Hadamard patterns measure (H s H + S_total·J)/2 — the
    scene's total flux lands entirely in the [0,0] (DC) coefficient and,
    untreated, dominates the normalization so the image quantizes to
    nothing (v0.5 fix, caught by tests/test_reference_physics.py).
    The DC term is replaced with the median of its neighbors before
    display scaling; for exact DC recovery capture differential ± pairs
    (see nullification.HadamardNu / barcode.decode_hadamard_single_pixel).
    """
    N = n_per_side
    y = np.asarray(intensities, dtype=np.float32).reshape(N, N)
    from .patterns import _sylvester_hadamard
    H = _sylvester_hadamard(N).astype(np.float32)
    img = (H @ y @ H) / (N * N)
    img[0, 0] = float(np.median(img[:2, :2].ravel()[1:]))   # de-spike DC
    sig = img - np.median(img)
    snr = float(20 * math.log10(np.abs(sig).max() / (sig.std() + 1e-6) + 1e-9))
    a, b = img.min(), img.max()
    out = ((img - a) / (b - a + 1e-9) * 255).astype(np.uint8)
    return SinglePixelResult(image=out, snr_db=snr)


@dataclass
class VibrationResult:
    dominant_hz: float
    amplitude: float
    spectrum: np.ndarray

def decode_vibration(intensities, fps: float) -> VibrationResult:
    sig = np.asarray(intensities, dtype=np.float32)
    sig -= sig.mean()
    fft = np.fft.rfft(sig * np.hanning(len(sig)))
    mag = np.abs(fft)
    freqs = np.fft.rfftfreq(len(sig), 1.0 / fps)
    idx = int(np.argmax(mag[1:]) + 1)
    return VibrationResult(dominant_hz=float(freqs[idx]),
                           amplitude=float(mag[idx] / len(sig)),
                           spectrum=mag)


@dataclass
class DefectResult:
    diffuse: np.ndarray
    specular: np.ndarray
    defect_score: float

def decode_defect(frame_a, frame_b) -> DefectResult:
    """Nayar high-frequency separation: diffuse = min(A, B), specular = A+B - 2·min."""
    a = frame_a.astype(np.float32); b = frame_b.astype(np.float32)
    diffuse = np.minimum(a, b)
    specular = np.clip(a + b - 2 * diffuse, 0, 255)
    return DefectResult(diffuse=diffuse.astype(np.uint8),
                        specular=specular.astype(np.uint8),
                        defect_score=float(specular.std()))


@dataclass
class MarkResult:
    text: str
    confidence: float

def decode_mark(frame, expected: Optional[str] = None) -> MarkResult:
    try:
        import pytesseract
        text = pytesseract.image_to_string(frame).strip()
        conf = 1.0 if (expected is None or expected in text) else 0.0
        return MarkResult(text=text, confidence=conf)
    except ImportError:
        return MarkResult(text="(pytesseract not installed)", confidence=0.0)


def _box_blur(img: np.ndarray, k: int) -> np.ndarray:
    from numpy.lib.stride_tricks import sliding_window_view as swv
    pad = k // 2
    p = np.pad(img.astype(np.float32), pad, mode="edge")
    w = swv(p, (k, k))
    return w.mean(axis=(-1, -2)).astype(np.float32)

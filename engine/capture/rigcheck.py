"""
Rig check: has the camera or the projector been nudged since the reference?

Pure functions (no rig, no library): the image shift between a reference
frame and a fresh one by phase correlation, refined to sub-pixel with a
locally upsampled DFT (Guizar-Sicairos et al. 2008), and the OK / MOVED
assessment. engine.api.rig_check captures the frames and stores the
reference; this module only does the numbers, so it is unit-tested on
synthetic shifts.

Two frames are compared:
  flood     the scene under white light: it moves when the CAMERA moves
  pattern   a seeded random block pattern: it moves when the camera OR the
            projector moves (the pattern is normalised by the flood, so the
            scene's albedo does not dominate)
A pattern shift with no flood shift points at the projector.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

DEFAULT_TOL_PX = 0.5          # a shift above this is MOVED
MIN_PEAK = 0.35               # normalised correlation below this: no usable texture
HIGHPASS_SIGMA_PX = 8.0       # shading removed before matching (px of the downsampled image)
PATTERN_BLOCK_PX = 16         # DMD px per block of the random pattern
PATTERN_SEED = 20261006


def random_block_pattern(shape=(1140, 912), block: int = PATTERN_BLOCK_PX,
                         seed: int = PATTERN_SEED) -> np.ndarray:
    """A seeded random black / white block pattern (DMD rows x cols, uint8).
    Not periodic, so its phase correlation has one peak (a checker or a
    fringe would alias at its period)."""
    rng = np.random.default_rng(seed)
    h, w = shape
    cells = rng.random(((h + block - 1) // block, (w + block - 1) // block)) > 0.5
    img = np.kron(cells, np.ones((block, block), bool))[:h, :w]
    return (img * 255).astype(np.uint8)


def _prep(img: np.ndarray, max_side: int) -> tuple:
    """Float image, downsampled by an integer factor to max_side, high-passed
    (shading and vignetting removed: they are common to both frames whatever
    moved), Hann windowed -> (image, factor)."""
    from scipy import ndimage
    a = np.asarray(img, np.float64)
    if a.ndim == 3:
        a = a.mean(axis=2)
    a = np.nan_to_num(a, nan=float(np.nanmean(a)) if np.isfinite(a).any() else 0.0)
    f = max(1, int(np.ceil(max(a.shape) / float(max_side)))) if max_side else 1
    if f > 1:
        h, w = (a.shape[0] // f) * f, (a.shape[1] // f) * f
        a = a[:h, :w].reshape(h // f, f, w // f, f).mean(axis=(1, 3))
    a = a - ndimage.gaussian_filter(a, HIGHPASS_SIGMA_PX, mode="nearest")
    win = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    return a * win, f


def _upsampled_dft(data: np.ndarray, region: int, up: int, r0: float, c0: float) -> np.ndarray:
    """The inverse DFT of `data` on a region x region grid with 1/up px spacing,
    starting at (r0, c0) px (matrix-multiply DFT: only the patch is computed)."""
    nr, nc = data.shape
    kc = np.fft.ifftshift(np.arange(nc)) - nc // 2
    kr = np.fft.ifftshift(np.arange(nr)) - nr // 2
    pc = (np.arange(region) / up + c0)
    pr = (np.arange(region) / up + r0)
    col = np.exp(2j * np.pi * np.outer(kc, pc) / nc)          # (nc, region)
    row = np.exp(2j * np.pi * np.outer(pr, kr) / nr)          # (region, nr)
    return row @ data @ col


def phase_shift(ref: np.ndarray, img: np.ndarray, upsample: int = 50,
                max_side: int = 1024) -> dict:
    """Translation of `img` relative to `ref` (img(x) ~ ref(x - shift)), in px of
    the input images -> {"dx", "dy", "shift_px", "peak"}. The integer peak comes
    from the phase correlation (whitened: one sharp peak), the sub-pixel position
    from the plain cross-correlation upsampled around it (unbiased under noise).
    peak is the normalised cross-correlation there (1 = the same texture, ~0 =
    nothing in common: a lens cap, a dark projector, a featureless scene)."""
    a, f = _prep(ref, max_side)
    b, fb = _prep(img, max_side)
    if a.shape != b.shape or f != fb:
        raise ValueError(f"reference {np.shape(ref)} and image {np.shape(img)} differ in size")
    norm = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if norm <= 0:
        return {"dx": 0.0, "dy": 0.0, "shift_px": 0.0, "peak": 0.0}
    X = np.fft.fft2(b) * np.conj(np.fft.fft2(a))
    pc = np.abs(np.fft.ifft2(X / np.maximum(np.abs(X), 1e-12)))
    r, c = np.unravel_index(int(np.argmax(pc)), pc.shape)
    nr, nc = pc.shape
    dy = float(r if r <= nr // 2 else r - nr)
    dx = float(c if c <= nc // 2 else c - nc)
    region = int(np.ceil(upsample * 3))                     # +-1.5 px around the peak
    off = (region // 2) / upsample
    patch = np.real(_upsampled_dft(X, region, upsample, dy - off, dx - off))
    pr, pcl = np.unravel_index(int(np.argmax(patch)), patch.shape)
    dy, dx = dy - off + pr / upsample, dx - off + pcl / upsample
    peak = float(patch[pr, pcl]) / X.size / norm
    dx, dy = dx * f, dy * f
    return {"dx": round(float(dx), 3), "dy": round(float(dy), 3),
            "shift_px": round(float(np.hypot(dx, dy)), 3), "peak": round(max(peak, 0.0), 4)}


def normalised_pattern(pattern: np.ndarray, white: np.ndarray,
                       black: Optional[np.ndarray] = None) -> np.ndarray:
    """(pattern - black) / (white - black), clipped; 0 where the flood is dark."""
    p = np.asarray(pattern, np.float64)
    w = np.asarray(white, np.float64)
    k = np.zeros_like(w) if black is None else np.asarray(black, np.float64)
    den = w - k
    ok = den > max(4.0, 0.05 * float(np.percentile(den, 99)))
    out = np.zeros_like(p)
    out[ok] = np.clip((p - k)[ok] / den[ok], 0.0, 1.5)
    return out


def assess(flood: Optional[dict], pattern: Optional[dict], tol_px: float = DEFAULT_TOL_PX,
           mm_per_px: Optional[float] = None, distance_mm: Optional[float] = None) -> dict:
    """OK / MOVED / UNKNOWN from the flood and pattern shifts (phase_shift dicts,
    None = not measured). MOVED names what moved; mm only when mm_per_px is
    known (never a px number reported as mm)."""
    def usable(s):
        return s is not None and s["peak"] >= MIN_PEAK

    cam = flood["shift_px"] if usable(flood) else None
    rig = pattern["shift_px"] if usable(pattern) else None
    if cam is None and rig is None:
        return {"status": "UNKNOWN", "moved": None, "shift_px": None, "shift_mm": None,
                "message": "no usable texture in the frames (lens cap? projector off? an empty, "
                           "featureless scene?): rig position not checked"}
    worst = max(v for v in (cam, rig) if v is not None)
    mm = round(worst * mm_per_px, 3) if mm_per_px else None
    at = f" at {distance_mm:.0f} mm" if (mm is not None and distance_mm) else ""
    size = f"~{worst:.1f} px" + (f" (~{mm:.2f} mm{at})" if mm is not None else "")
    if worst <= tol_px:
        return {"status": "OK", "moved": None, "shift_px": round(worst, 3), "shift_mm": mm,
                "message": f"rig in place: image shift {size} <= {tol_px:g} px"}
    if cam is not None and cam > tol_px:
        what = "camera"
    elif cam is not None and rig is not None:
        what = "projector"
    else:
        what = "camera or projector"
    return {"status": "MOVED", "moved": what, "shift_px": round(worst, 3), "shift_mm": mm,
            "message": f"{what} moved {size}: re-home / recalibrate before measuring"}

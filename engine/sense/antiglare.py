"""
Anti-glare code reading (EXPERIMENTAL, opt-in: `codes --anti-glare`).

A glossy, curved label under a flood shows a saturated specular stripe that
wipes out the code under it. Three steps, each a pure function on arrays:

  1. HDR fusion of a DMD flood bracket at many levels. Frames are ordered by
     MEASURED brightness (the HDMI path can be optically inverted, as in
     codes.decode), each frame's gain relative to the brightest is measured on
     pixels unsaturated in both, and every pixel is the gain-normalised mean
     of its unsaturated samples. The stripe is recovered from the dim frames.
  2. Optional narrow DMD column-strip sweep over the code region: each camera
     pixel keeps only the energy of the strips that light it directly (its own
     strip +- a couple), which drops the bloom other strips spill onto it.
  3. De-veil: specular gloss is an ADDITIVE offset that varies slowly along
     the stripe, so subtracting a local lower envelope and dividing by the
     local range (min / max filters, ~9 px along the stripe x 3 px across)
     leaves the print contrast.

The result goes through the normal codes.decode_image. Capture is driven
through sync.shoot like codes.capture (shoot calls only; decode after the
rig is released).
"""

from __future__ import annotations

import numpy as np

from .. import _seed
from . import codes as codes_mod

LEVELS = (255, 192, 128, 96, 64, 48, 32, 24, 16, 12, 8, 4)   # DMD flood levels
SAT_DN = 250.0                 # camera DN at / above which a sample is saturated
DEVEIL_WINDOWS = ((9, 3), (5, 3), (7, 3), (11, 3), (21, 3), (31, 5))   # (rows, cols) px


# ------------------------------------------------------------ 1. HDR fusion

def hdr_fuse(frames, dark=None, sat: float = SAT_DN, floor: float = 4.0) -> np.ndarray:
    """Radiance map (in units of the brightest frame) from a flood bracket.

    `frames`: list of camera frames at different flood levels, any order and
    either polarity; `dark`: a DMD-black frame to subtract (optional). The
    brightest frame (by mean) is the reference; each other frame's gain is the
    median ratio over pixels unsaturated in both and above `floor`.
    Pixels saturated in every frame keep the dimmest frame's scaled value."""
    fr = [np.asarray(f, np.float32) for f in frames]
    if not fr:
        raise ValueError("hdr_fuse needs at least one frame")
    order = np.argsort([-float(f.mean()) for f in fr])
    fr = [fr[i] for i in order]
    unsat = [f < sat for f in fr]
    d = np.zeros_like(fr[0]) if dark is None else np.asarray(dark, np.float32)
    sig = [np.clip(f - d, 0, None) for f in fr]
    ref, ref_ok = sig[0], unsat[0]
    gains = [1.0]
    for s, ok in zip(sig[1:], unsat[1:]):
        m = ref_ok & ok & (s > floor) & (ref > floor)
        if m.sum() < 50:                   # nothing comparable with the reference:
            m = ok & (s > floor)           # chain via brightness ratio instead
            g = float(s[m].mean() / max(ref.mean(), 1e-6)) if m.any() else 0.0
        else:
            g = float(np.median(s[m] / ref[m]))
        gains.append(g)
    num = np.zeros_like(ref)
    den = np.zeros_like(ref)
    for s, ok, g in zip(sig, unsat, gains):
        if g <= 0:
            continue
        w = ok & (s > floor * 0.5)
        num += np.where(w, s, 0.0)                       # weight ~ signal: a gain-weighted
        den += np.where(w, g, 0.0)                       # mean, low SNR samples count less
    out = num / np.maximum(den, 1e-6)
    dimmest = next((i for i in range(len(sig) - 1, -1, -1) if gains[i] > 0), 0)
    never = den <= 0
    out[never] = sig[dimmest][never] / max(gains[dimmest], 1e-6)
    return out


# ------------------------------------------------------------ 2. strip sweep

def strip_patterns(c0: int, c1: int, width: int = 3, step: int = 1, *, bright: int = 255,
                   dark: int = 0, shape=(1140, 912)) -> list:
    """[(name, pattern)] narrow column strips stepped over DMD columns c0..c1.
    `bright` / `dark` are the DMD values that light / do not light (swap them
    on an optically inverted path)."""
    h, w = shape
    if not 0 <= c0 < c1 <= w:
        raise ValueError(f"strip columns {c0},{c1}: need 0 <= c0 < c1 <= {w}")
    out = []
    for c in range(c0, c1, max(1, int(step))):
        img = np.full((h, w), dark, np.uint8)
        img[:, c:min(c + int(width), w)] = bright
        out.append((f"strip_{c}", img))
    return out


def strip_own(stack, dark=None, half: int = 2) -> np.ndarray:
    """Per pixel, the energy of its own strip (the strip that lights it most)
    +- `half` neighbours, from a (n_strips, h, w) stack in sweep order. Light
    spilled from far strips (bloom, the specular veil of other columns) is
    dropped."""
    S = np.asarray(stack, np.float32)
    if S.ndim != 3 or len(S) == 0:
        raise ValueError("strip_own needs a (n_strips, h, w) stack")
    if dark is not None:
        S = np.clip(S - np.asarray(dark, np.float32)[None], 0, None)
    arg = S.argmax(0)
    k = np.arange(-int(half), int(half) + 1)[:, None, None]
    idx = np.clip(arg[None] + k, 0, len(S) - 1)
    return np.take_along_axis(S, idx, 0).sum(0)


# ------------------------------------------------------------ 3. de-veil

def deveil(img, win=(9, 3)) -> np.ndarray:
    """(img - local lower envelope) / local range, in 0..1. `win` = (rows,
    cols) px of the min / max filters: long along the stripe (vertical), short
    across it, so an additive gloss offset is removed and the print remains."""
    import cv2
    a = np.asarray(img, np.float32)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (int(win[1]), int(win[0])))
    lo = cv2.blur(cv2.erode(a, k), (int(win[1]), int(win[0])))
    hi = cv2.blur(cv2.dilate(a, k), (int(win[1]), int(win[0])))
    return np.clip((a - lo) / np.maximum(hi - lo, 1e-3), 0, 1)


def decode_radiance(img, illumination: str = "anti_glare", windows=DEVEIL_WINDOWS) -> list:
    """De-veil a radiance map at a few window sizes and decode; the first
    window that reads anything wins. The plain radiance is tried first."""
    found = codes_mod.decode_image(np.asarray(img, np.float32), illumination=f"{illumination}:hdr")
    if found:
        return found
    for win in windows:
        c = deveil(img, win)
        u8 = (c * 255).astype(np.uint8)
        found = codes_mod.decode_image(u8, illumination=f"{illumination}:deveil{win[0]}x{win[1]}")
        if found:
            return found
    return []


# ------------------------------------------------------------ capture + decode

def capture(sync, levels=LEVELS, strip_cols=None, strip_width: int = 3,
            strip_step: int = 1) -> dict:
    """Shoot the anti-glare set: DMD black, a flood per level, and (with
    `strip_cols` = (c0, c1) DMD columns over the code) a strip sweep. Only
    shoot calls, like codes.capture. -> frames {name: frame}; the strip frames
    are 'strip_<col>', the strip-sweep background 'strip_dark'."""
    if not levels:
        raise ValueError("levels must list at least one DMD flood level")
    P = _seed.load("patterns")
    frames = {"dmd_black": sync.shoot(P.black())}
    for lv in levels:
        frames[f"flood_{lv}"] = sync.shoot(P.flood(int(lv)))
    if strip_cols:
        # polarity is measured, not assumed: on an inverted path 'black' is the bright one
        inverted = bool(frames["dmd_black"].mean() > frames[f"flood_{max(levels)}"].mean())
        bright, dark = (0, 255) if inverted else (255, 0)
        shape = P.black().shape[:2]
        c0, c1 = (int(v) for v in strip_cols)
        frames["strip_dark"] = sync.shoot(np.full(shape, dark, np.uint8))
        for name, pat in strip_patterns(c0, c1, strip_width, strip_step, bright=bright,
                                        dark=dark, shape=shape):
            frames[name] = sync.shoot(pat)
    return frames


def decode(frames: dict) -> tuple:
    """-> (codes, radiance, used): HDR over every flood + the DMD black frame
    (both polarities), then the strip-sweep image if there is one; the first
    that decodes wins. No rig needed."""
    floods = [f for n, f in frames.items() if n.startswith("flood_") or n == "dmd_black"]
    if not floods:
        raise ValueError("frames lack the flood bracket: decode() takes what capture() returned")
    darkest = min(floods, key=lambda f: float(f.mean()))
    lit = [f for f in floods if f is not darkest]
    hdr = hdr_fuse(lit or floods, dark=darkest if lit else None)
    found = decode_radiance(hdr, "anti_glare_hdr")
    if found:
        return found, hdr, "hdr"
    strips = [n for n in frames if n.startswith("strip_") and n != "strip_dark"]
    if strips:
        strips.sort(key=lambda n: int(n.split("_", 1)[1]))
        own = strip_own(np.stack([frames[n] for n in strips]), frames.get("strip_dark"))
        found = decode_radiance(own, "anti_glare_strips")
        if found:
            return found, own, "strips"
    return [], hdr, "hdr"

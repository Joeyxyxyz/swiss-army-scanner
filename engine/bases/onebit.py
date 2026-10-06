"""
1-bit fringe set for DLPC350 pattern-sequence mode ("fastscan"): the
column-axis multifreq scan (4 steps x periods 8, 80, 1280 DMD columns +
white/black) as 1-bit images, plus two sync markers, for a sequence that
plays from ONE 24-bit flash image on the controller's internal timer while
the camera free-runs faster than the pattern rate (soft sync).

Why 1-bit: in pattern mode an 8-bit pattern is shown as PWM bit-planes
inside its exposure; an unsynchronised camera integrates a random slice of
it. A 1-bit pattern is constant for its whole exposure, so every camera
frame that lies inside one pattern slot sees exactly that pattern.

How the sinusoids are made 1-bit (chosen by offline simulation through a
Gaussian defocus PSF sigma 0.5..2 DMD columns + mirror aperture + camera
pixel sampling, `study()`; results in docs/FLASH_FASTSCAN_4500.md):

  fine, T = 8 (the level that sets depth precision) — COLUMN-EXACT ordered
    dither. Every column carries one level of an 8-sample profile
    FINE_LEVELS_Q4 = (4, 3, 2, 0, 0, 0, 2, 3) / 4, symmetric about column 0
    (so no phase bias against the 8-bit cosine) and chosen so DFT bin 3 —
    the only bin a 4-step decode of a period-8 profile is sensitive to
    (harmonics 3 and 5) — is 2.9 % of the fundamental. A level j/4 lights j
    of every 4 rows in bit-reversed order, odd columns on the complementary
    row phase: the texture is at most 4 rows (2 column units) in y, 2
    columns in x, and every step is an exact 2-column translate, so the
    x texture is common to all four steps (DC for the 4-step sum).
    A binary square wave is worse unless the projector is defocused
    (sigma >= 1.5 col): its 3rd harmonic leaks into the 4-step phase.
  mid / coarse, T = 80 / 1280 (they only fix fringe orders; tolerance
    T_finer / 4 = 2 / 20 columns) — Floyd-Steinberg error diffusion
    (serpentine) of ONE wide strip; step k is the strip window shifted by
    k T/4 columns, so the four steps are exact translates. At a sharply
    focused projector their texture is visible to the camera, so these two
    levels are decoded from frames low-passed in the camera
    (prefilter_frames, Gaussian 3 camera px). The fine level is not
    filtered by default; at a sharply focused projector an optional 1 px
    Gaussian (fine_px=FINE_PREFILTER_PX) removes most of its texture.

Diamond DMD: odd (or even) rows sit half a column right. The fine tile
lights different rows per level, so against the HDMI 8-bit fringes the
calibration was made with (every row of a column the same grey) it reads a
constant ~0.043 column offset whose sign is the row parity (study(),
fine_vs_hdmi). A diamond-neutral tile needs 2 x 2 row pairs per level and
doubles the texture error at sharp focus, so this tile is kept: the offset
is common to the whole field and cancels in relative heights.

Sequence (16 x 1-bit = bit-planes G0..G7, R0..R7 of one 24-bit image):

  slot 0  sync_white  every mirror on, full field (both variants)
  slot 1  sync_code   horizontal bars, 16 rows on / 16 off, full field: the
                      only frame whose structure runs along camera y
  slot 2  white       decode white   (zone variant: zones only)
  slot 3  black       decode black
  4..7    fine   T=8     steps 0..3   (zone variant: zones only)
  8..11   mid    T=80    steps 0..3
  12..15  coarse T=1280  steps 0..3

decode order (MultiFreqBasis(steps=4, periods_px=(8, 80, 1280), axis="x")):
slots 2, 3, 4..15 — see DECODE_SLOTS / decode().

Pure numpy / OpenCV. Nothing here talks to the projector or the camera.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

import numpy as np

from .multifreq import MultiFreqBasis

PATTERN_VERSION = 1                           # bump when any slot image changes
DMD_SHAPE = (1140, 912)                       # (rows, cols)
STEPS = 4
PERIODS = (8, 80, 1280)
FINE_LEVELS_Q4 = (4, 3, 2, 0, 0, 0, 2, 3)     # quarters, column 0..7 of a fine period
ROW_ORDER_Q4 = (0, 2, 1, 3)                   # bit-reversed thresholds for 4 rows
SYNC_BAR_ROWS = 16                            # sync_code: 16 rows on, 16 off
PREFILTER_PX = 3.0                            # camera px, mid + coarse levels only
FINE_PREFILTER_PX = 1.0                       # optional, fine level, sharp projector (study())
# prefiltered levels: px whose phasor coherence |G*z| / (G*|z|) is below this sit where the
# 3 px Gaussian mixed two surfaces (a depth edge): the mixed mid phase can pick the wrong
# fine order (an 8-column error). Mock e2e (tests/test_flash_e2e_mock.py): every such px
# < 0.83, smooth surfaces p0.1 > 0.85 (p50 0.99), sharp or defocused projector.
PREFILTER_COHERENCE_MIN = 0.86
FS_MARGIN = 16                                # strip margin (error-diffusion start-up)

SLOTS = ("sync_white", "sync_code", "white", "black",
         "fine_k0", "fine_k1", "fine_k2", "fine_k3",
         "mid_k0", "mid_k1", "mid_k2", "mid_k3",
         "coarse_k0", "coarse_k1", "coarse_k2", "coarse_k3")
DECODE_SLOTS = tuple(range(2, 16))            # white, black, fine x4, mid x4, coarse x4
SYNC_SLOTS = (0, 1)
_LEVEL = {"fine": 8, "mid": 80, "coarse": 1280}


def basis() -> MultiFreqBasis:
    """The decoder the captured frames go through (unchanged multifreq)."""
    return MultiFreqBasis(steps=STEPS, periods_px=PERIODS, axis="x", diamond_parity=None)


# ------------------------------------------------------------- fine level
def fine_profile() -> np.ndarray:
    """The 8 column levels (0..1) of one fine period, column 0 = the cosine peak."""
    return np.asarray(FINE_LEVELS_Q4, np.float64) / 4.0


def row_thresholds() -> np.ndarray:
    """(4, 2) thresholds: row r mod 4, column parity. A level j/4 lights
    exactly j of the 4 rows of every column."""
    M = np.zeros((4, 2), np.float64)
    for r in range(4):
        M[r, 0] = ROW_ORDER_Q4[r]
        M[r, 1] = ROW_ORDER_Q4[(r + 2) % 4]          # complementary row phase
    return (M + 0.5) / 4.0


def fine_fringe(k: int, shape=DMD_SHAPE) -> np.ndarray:
    """Fine (T = 8) step k as a bool image: column c shows level
    FINE_LEVELS_Q4[(c + 2k) mod 8] — the translate of step 0 by 2k columns,
    matching MultiFreqBasis (phase + 2 pi k / 4)."""
    H, W = shape
    c = np.arange(W)
    lv = fine_profile()[(c + 2 * k) % 8]
    t = row_thresholds()[(np.arange(H) % 4)[:, None], (c % 2)[None, :]]
    return lv[None, :] > t


def fine_harmonics() -> dict:
    """|F3| / |F1| of the fine profile (the 4-step phase-error source) and |F1|."""
    F = np.fft.fft(fine_profile())
    return {"f3_over_f1": float(abs(F[3]) / abs(F[1])), "f1": float(abs(F[1])),
            "f1_cosine": 2.0}


def best_fine_profile(q: int = 4) -> np.ndarray:
    """Brute-force search the FINE_LEVELS_Q4 choice: symmetric, monotone
    8-sample profiles in steps of 1/q with |F1| >= 85 % of a full cosine,
    minimising |F3|/|F1| (ties -> closest to the sampled cosine)."""
    import itertools
    ideal = 0.5 * (1 + np.cos(2 * np.pi * np.arange(8) / 8))
    best = None
    for f0, f1, f2, f3, f4 in itertools.product(range(q + 1), repeat=5):
        if not f0 >= f1 >= f2 >= f3 >= f4:
            continue
        f = np.array([f0, f1, f2, f3, f4, f3, f2, f1], np.float64) / q
        F = np.fft.fft(f)
        if F[1].real < 0.85 * 2.0:
            continue
        key = (round(float(abs(F[3]) / abs(F[1])), 6), float(np.sum((f - ideal) ** 2)))
        if best is None or key < best[0]:
            best = (key, f)
    return best[1]


# -------------------------------------------------- mid / coarse levels
def fs_dither(img: np.ndarray) -> np.ndarray:
    """Floyd-Steinberg error diffusion, serpentine, deterministic. 0..1 -> bool."""
    a = np.asarray(img, np.float64)
    H, W = a.shape
    out = np.zeros((H, W), bool)
    nxt = a[0].tolist()
    for r in range(H):
        row = nxt
        nxt = a[r + 1].tolist() if r + 1 < H else [0.0] * W
        o = [False] * W
        if r % 2 == 0:
            cols, d = range(W), 1
        else:
            cols, d = range(W - 1, -1, -1), -1
        for c in cols:
            v = row[c]
            b = v >= 0.5
            o[c] = b
            e = v - 1.0 if b else v
            c1 = c + d
            c0 = c - d
            if 0 <= c1 < W:
                row[c1] += e * 0.4375
                nxt[c1] += e * 0.0625
            if 0 <= c0 < W:
                nxt[c0] += e * 0.1875
            nxt[c] += e * 0.3125
        out[r] = o
    return out


@lru_cache(maxsize=8)
def _fs_strip(T: int, H: int, W: int) -> np.ndarray:
    """One FS-dithered cosine strip wide enough for all 4 step windows."""
    shift = (STEPS - 1) * T // STEPS
    c = np.arange(-FS_MARGIN, W + shift + FS_MARGIN, dtype=np.float64)
    prof = 0.5 * (1 + np.cos(2 * np.pi * c / T))
    strip = fs_dither(np.repeat(prof[None, :], H, 0))
    strip.setflags(write=False)
    return strip


def fs_fringe(T: int, k: int, shape=DMD_SHAPE) -> np.ndarray:
    """Mid / coarse step k (bool): the strip window shifted by k T/4 columns,
    i.e. column c shows the dithered cos(2 pi c / T + 2 pi k / 4)."""
    H, W = shape
    if (T * k) % STEPS:
        raise ValueError(f"T={T}: k T/4 must be a whole number of columns")
    s = FS_MARGIN + k * T // STEPS
    return _fs_strip(int(T), H, W)[:, s:s + W].copy()


# ------------------------------------------------------------------ sync
def sync_white(shape=DMD_SHAPE) -> np.ndarray:
    return np.ones(shape, bool)


def sync_code(shape=DMD_SHAPE, bar_rows: int = SYNC_BAR_ROWS) -> np.ndarray:
    """Horizontal bars (DMD rows), full field: its camera image varies along
    camera y, every decode frame (vertical fringes) along camera x."""
    H, W = shape
    return np.repeat(((np.arange(H) // bar_rows) % 2 == 0)[:, None], W, 1)


def sync_signature(frame: np.ndarray, roi=None) -> dict:
    """What a soft-sync sorter needs per camera frame: mean level and the
    along-y / along-x gradient energies. The sync_code slot is the one with
    the highest orientation = grad_y / grad_x (~19 in the simulated camera
    view vs < 1 for every fringe slot, ~0 for the floods); sync_white sits
    immediately before it and is the brightest slot (tied with slot 2, white,
    in the full image; the only full-field white in the zone image); black
    (slot 3) is the darkest. On a real scene compare slots with each other
    (scene edges add to every frame), not with fixed numbers."""
    f = np.asarray(frame, np.float32)
    if roi is not None:
        f = f[roi]
    gy = float(np.mean(np.abs(np.diff(f, axis=0)))) if f.shape[0] > 1 else 0.0
    gx = float(np.mean(np.abs(np.diff(f, axis=1)))) if f.shape[1] > 1 else 0.0
    return {"mean": float(f.mean()), "grad_y": gy, "grad_x": gx,
            "orientation": gy / max(gx, 1e-6)}


# ---------------------------------------------------------------- the set
def slot_meaning(slot: int) -> dict:
    name = SLOTS[slot]
    if slot == 0:
        return {"name": name, "role": "sync_white", "period_px": None, "step": None,
                "method": "all on", "decode_index": None,
                "meaning": "SYNC A: every mirror on, full field; the brightest slot"}
    if slot == 1:
        return {"name": name, "role": "sync_code", "period_px": None, "step": None,
                "method": f"horizontal bars {SYNC_BAR_ROWS} rows on / {SYNC_BAR_ROWS} off",
                "decode_index": None,
                "meaning": "SYNC B: horizontal bars, full field; the only slot whose "
                           "structure runs along camera y; follows SYNC A"}
    if slot in (2, 3):
        return {"name": name, "role": name, "period_px": None, "step": None,
                "method": "all on" if slot == 2 else "all off", "decode_index": slot - 2,
                "meaning": f"decode {name} (polarity / contrast mask)"}
    level = name.split("_")[0]
    k = int(name[-1])
    T = _LEVEL[level]
    method = ("column-exact ordered dither, levels (4,3,2,0,0,0,2,3)/4"
              if level == "fine" else "Floyd-Steinberg strip, exact k T/4 translate")
    return {"name": name, "role": "fringe", "period_px": T, "step": k, "method": method,
            "decode_index": slot - 2,
            "meaning": f"{level} fringe T={T} DMD columns, step {k} of 4 "
                       f"(phase + {90 * k} deg)"}


def slot_image(slot: int, shape=DMD_SHAPE) -> np.ndarray:
    name = SLOTS[slot]
    if name == "sync_white" or name == "white":
        return np.ones(shape, bool)
    if name == "sync_code":
        return sync_code(shape)
    if name == "black":
        return np.zeros(shape, bool)
    level, k = name.split("_")[0], int(name[-1])
    if level == "fine":
        return fine_fringe(k, shape)
    return fs_fringe(_LEVEL[level], k, shape)


def fastscan_patterns(zone_mask: Optional[np.ndarray] = None, shape=DMD_SHAPE) -> list:
    """The 16 slots in play order: dicts {slot, name, role, period_px, step,
    method, decode_index, meaning, zone, image (uint8 0/255)}.
    zone_mask (bool, DMD shape): decode frames (slots 2..15) lit only inside
    it; the sync markers stay full field so they are found in any ROI."""
    if zone_mask is not None:
        zone_mask = np.asarray(zone_mask, bool)
        if zone_mask.shape != tuple(shape):
            raise ValueError(f"zone mask is {zone_mask.shape}, DMD is {tuple(shape)}")
        if not zone_mask.any():
            raise ValueError("zone mask is empty")
    out = []
    for s in range(len(SLOTS)):
        img = slot_image(s, shape)
        zoned = zone_mask is not None and s not in SYNC_SLOTS
        if zoned:
            img = img & zone_mask
        d = slot_meaning(s)
        d.update(slot=s, zone=bool(zoned), image=img.astype(np.uint8) * 255,
                 fill=float(img.mean()))
        out.append(d)
    return out


# ---------------------------------------------------------------- decode
def decode_order(slot_frames: list) -> list:
    """16 slot frames (play order) -> the 14 frames MultiFreqBasis.decode takes."""
    if len(slot_frames) != len(SLOTS):
        raise ValueError(f"need {len(SLOTS)} slot frames (play order), got {len(slot_frames)}")
    return [slot_frames[s] for s in DECODE_SLOTS]


def prefilter_frames(frames: list, sigma_px: float = PREFILTER_PX,
                     fine_px: float = 0.0) -> list:
    """Decode-order frames -> the same with the mid and coarse fringe frames
    Gaussian-blurred (sigma_px camera px). White, black and the fine level
    are returned untouched unless fine_px > 0 (an optional light blur of the
    fine frames only: at a sharply focused projector it removes most of the
    1-bit dither texture, FINE_PREFILTER_PX, at the cost of a little lateral
    resolution; see study()). Linear, so it is a smoothing of the phasor
    field: phases stay unbiased, the dither texture and camera noise
    average out."""
    import cv2
    if len(frames) != 2 + len(PERIODS) * STEPS:
        raise ValueError(f"need {2 + len(PERIODS) * STEPS} decode-order frames, got {len(frames)}")
    out = [np.asarray(f, np.float32) for f in frames]
    if sigma_px and sigma_px > 0:
        for i in range(2 + STEPS, len(out)):
            out[i] = cv2.GaussianBlur(out[i], (0, 0), float(sigma_px),
                                      borderType=cv2.BORDER_REFLECT)
    if fine_px and fine_px > 0:
        for i in range(2, 2 + STEPS):
            out[i] = cv2.GaussianBlur(out[i], (0, 0), float(fine_px),
                                      borderType=cv2.BORDER_REFLECT)
    return out


def phasor_coherence(group: list, sigma_px: float) -> np.ndarray:
    """|G * z| / (G * |z|) for the phasor z of one N-step group (G: Gaussian sigma_px).
    1 where the phase is uniform under the kernel, a little less on a phase ramp (~0.99
    for a 120 camera px fringe period at 3 px), and well below where the kernel straddles
    two surfaces with different phases: there the low-passed phase is a mixture of both."""
    import cv2
    n = len(group)
    d = 2 * np.pi * np.arange(n) / n
    S = sum(np.asarray(f, np.float32) * np.float32(np.sin(dk)) for f, dk in zip(group, d))
    C = sum(np.asarray(f, np.float32) * np.float32(np.cos(dk)) for f, dk in zip(group, d))

    def g(a):
        return cv2.GaussianBlur(np.asarray(a, np.float32), (0, 0), float(sigma_px),
                                borderType=cv2.BORDER_REFLECT)
    return np.hypot(g(S), g(C)) / np.maximum(g(np.hypot(S, C)), 1e-6)


def prefilter_keep(frames: list, sigma_px: float = PREFILTER_PX,
                   coherence_min: float = PREFILTER_COHERENCE_MIN) -> np.ndarray:
    """Decode-order frames (unfiltered) -> bool: px where every prefiltered level (mid,
    coarse) keeps a coherent phase under the sigma_px Gaussian (phasor_coherence >=
    coherence_min). False near depth edges, where prefiltering mixes surfaces."""
    if len(frames) != 2 + len(PERIODS) * STEPS:
        raise ValueError(f"need {2 + len(PERIODS) * STEPS} decode-order frames, got {len(frames)}")
    keep = np.ones(np.shape(frames[0]), bool)
    if not sigma_px or sigma_px <= 0 or not coherence_min:
        return keep
    for i in range(2 + STEPS, len(frames), STEPS):
        keep &= phasor_coherence(frames[i:i + STEPS], sigma_px) >= float(coherence_min)
    return keep


def decode(frames: list, geometry_id: str, *, prefilter_px: float = PREFILTER_PX,
           fine_px: float = 0.0, contrast_min: float = 8.0, conf_min: float = 4.0,
           coherence_min: float = PREFILTER_COHERENCE_MIN):
    """Decode-order frames (14) -> DeformationField via the unchanged
    MultiFreqBasis, after prefilter_frames; px where the prefilter mixed surfaces
    (prefilter_keep) are left undecoded."""
    fld = basis().decode(prefilter_frames(frames, prefilter_px, fine_px), geometry_id,
                         contrast_min=contrast_min, conf_min=conf_min)
    keep = prefilter_keep(frames, prefilter_px, coherence_min)
    drop = fld.mask & ~keep
    fld.mask = fld.mask & keep
    fld.corr_x = np.where(drop, np.nan, fld.corr_x).astype(np.float32)
    fld.meta["prefilter_edge"] = {"min_coherence": coherence_min, "masked_px": int(drop.sum())}
    return fld


# ------------------------------------------------------------ simulation
DEFAULT_REGION = (880, 1140, 560, 820)        # DMD rows r0:r1, cols c0:c1 (the two faces)


def observe(pattern: np.ndarray, sigma_cols: float, *, cam_px_per_col: float = 1.47,
            rot_deg: float = 3.0, region=DEFAULT_REGION, margin_cols: float = 14.0,
            parity: Optional[int] = None):
    """Render a DMD pattern (bool / 0..255 / 0..1, full DMD shape) as the
    camera sees it: mirror box (1 column x 1 row; rows are 0.5 column units
    apart), Gaussian defocus sigma_cols (isotropic, column units), camera
    pixel box, sampled on a camera grid (cam_px_per_col px per column,
    rotated rot_deg). parity 0 / 1: the diamond DMD, rows with row % 2 ==
    parity sit half a column to the right (None: square grid).
    -> (values (ny, nx) float32 0..1, x (ny, nx) absolute DMD column
    coordinate of each camera sample)."""
    import cv2
    r0, r1, c0, c1 = region
    p = np.asarray(pattern)
    p = (p > 0).astype(np.float32) if p.dtype == bool or p.max() > 1 else p.astype(np.float32)
    p = p[r0:r1, c0:c1]
    up = np.repeat(np.repeat(p, 2, 0), 4, 1)           # 0.25 column-unit samples
    if parity is not None:
        if parity not in (0, 1):
            raise ValueError(f"parity must be None, 0 or 1, got {parity!r}")
        shifted = np.repeat((np.arange(r0, r1) % 2) == parity, 2)
        up[shifted] = np.roll(up[shifted], 2, axis=1)  # + 0.5 column (wrap is in the margin)
    cam = 1.0 / cam_px_per_col
    st = 4.0 * np.sqrt(sigma_cols ** 2 + cam ** 2 / 12.0)
    b = cv2.GaussianBlur(up, (0, 0), st, sigmaY=st, borderType=cv2.BORDER_REFLECT)
    H, W = p.shape
    nx = int((W - 2 * margin_cols) / cam)
    ny = int((H / 2 - margin_cols) / cam)
    ii, jj = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    th = np.deg2rad(rot_deg)
    x = margin_cols + cam * (jj * np.cos(th) - ii * np.sin(th)) + 0.37
    y = margin_cols / 2 + cam * (jj * np.sin(th) + ii * np.cos(th)) + 0.21
    x = np.clip(x, 0, W - 1)
    y = np.clip(y, 0, H / 2 - 0.5)
    val = cv2.remap(b, (4 * x + 1.5).astype(np.float32), (4 * y + 0.5).astype(np.float32),
                    cv2.INTER_LINEAR)
    return val, (x + c0).astype(np.float64)


def square_wave_error(sigma_cols: float, *, T: float = 8.0, cam_px_per_col: float = 1.47,
                      duty: float = 0.5, harmonics: int = 199) -> dict:
    """Closed-form 4-step phase error of a binary square wave (duty `duty`,
    all rows lit) seen through Gaussian defocus sigma_cols x mirror box (1
    column) x camera pixel box (1 / cam_px_per_col columns).

    The profile's cosine series a_h (h = 1, 2, ...) is weighted by the
    optics MTF(h / T); a 4-step decode returns
        arg( sum_{h = 1 mod 4} a_h e^{i h theta} + sum_{h = 3 mod 4} a_h e^{-i h theta} )
    so every harmonic h = 4m +- 1 (3, 5, 7, ...) leaks into a 4-theta
    ripple. -> peak / rms error in columns, and |a3| / |a1| after the optics."""
    h = np.arange(1, harmonics + 1, dtype=np.float64)
    a = 2.0 * np.sin(np.pi * h * duty) / (np.pi * h)          # cosine series, centred on the lit block
    f = h / float(T)                                          # cycles per column
    mtf = (np.exp(-2.0 * np.pi ** 2 * sigma_cols ** 2 * f ** 2) * np.sinc(f)
           * np.sinc(f / cam_px_per_col))
    a = a * mtf
    th = np.linspace(0.0, 2.0 * np.pi, 4096, endpoint=False)
    z = np.zeros(th.shape, np.complex128)
    for hi, ai in zip(h.astype(int), a):
        if hi % 4 == 1:
            z += ai * np.exp(1j * hi * th)
        elif hi % 4 == 3:
            z += ai * np.exp(-1j * hi * th)
    e = np.angle(z * np.exp(-1j * th))
    k = T / (2.0 * np.pi)
    return {"peak_cols": round(float(np.abs(e).max()) * k, 5),
            "rms_cols": round(float(np.sqrt(np.mean(e ** 2))) * k, 5),
            "a3_over_a1": round(float(abs(a[2]) / abs(a[0])), 5)}


def noise_rms_cols(modulation: float, T: float, *, white_dn: float = 200.0,
                   noise_dn: float = 1.5, frames: float = 4.0, steps: int = STEPS) -> float:
    """Random phase error (columns, rms per camera pixel) of an N-step decode:
    T / 2 pi * sqrt(2 / N) * sigma_I / (B * white), sigma_I = noise_dn /
    sqrt(frames averaged). modulation B in white = 1 units (a full-swing
    cosine has B = 0.5)."""
    if modulation <= 0:
        return float("inf")
    s = noise_dn / np.sqrt(max(frames, 1.0))
    return float(T / (2 * np.pi) * np.sqrt(2.0 / steps) * s / (modulation * white_dn))


def _phase_err(frames, x, T, keep):
    ph, B = basis()._wrapped(frames)
    err = np.angle(np.exp(1j * (ph - 2 * np.pi * x / T)))[keep]
    bias = float(np.angle(np.mean(np.exp(1j * err))))
    e = np.angle(np.exp(1j * (err - bias)))
    k = T / (2 * np.pi)
    return {"bias_cols": round(bias * k, 5), "rms_rad": round(float(np.std(e)), 5),
            "rms_cols": round(float(np.std(e)) * k, 5),
            "p99_cols": round(float(np.percentile(np.abs(e), 99)) * k, 5),
            "max_cols": round(float(np.abs(e).max()) * k, 5),
            "modulation": round(float(np.median(B[keep])), 4)}


NOISE_MODEL = {"white_dn": 200.0, "noise_dn": 1.5, "frames": 4.0}


def _vs_reference(fr, ref, T):
    """Phase of 1-bit frames minus the phase of the 8-bit reference frames seen
    through the same optics (what the HDMI-made calibration expects)."""
    b = basis()
    ph, B = b._wrapped(fr)
    pr, _ = b._wrapped(ref)
    e = np.angle(np.exp(1j * (ph - pr))).ravel()
    bias = float(np.angle(np.mean(np.exp(1j * e))))
    r = np.angle(np.exp(1j * (e - bias)))
    k = T / (2 * np.pi)
    return {"bias_cols": round(bias * k, 5), "rms_cols": round(float(np.std(r)) * k, 5),
            "p99_cols": round(float(np.percentile(np.abs(r), 99)) * k, 5),
            "modulation": round(float(np.median(B)), 4)}


def study(sigmas=(0.5, 1.0, 1.5, 2.0), *, cam_px_per_col: float = 1.47,
          mm_per_col: float = 1.87, prefilter_px: float = PREFILTER_PX,
          region=DEFAULT_REGION, parities=(None,), noise: Optional[dict] = None,
          fine_prefilter=(0.0,)) -> dict:
    """Noise-free phase error of every level, as decoded (4-step, multifreq
    formula) from the simulated camera view, vs the ideal phase 2 pi x / T;
    plus the binary square wave (4 on / 4 off) at T = 8 for comparison
    (simulated and closed form, square_wave_error) and an end-to-end decode
    (prefilter + MultiFreqBasis.decode, unwrap). Errors in DMD columns and
    mm (mm_per_col: range per column at the working distance, from the
    stereo calibration).

    fine_vs_hdmi[parity]: the fine level against the HDMI 8-bit fringe (what
    the calibration was made with: every row of a column the same grey) seen
    through the same optics, on the square grid (None) and on the diamond
    DMD with either row parity (0, 1) - the 1-bit tile lights different rows
    per level, so a half-column row offset shifts levels differently.
    noise_rms_*: random error per camera pixel from noise_rms_cols with
    NOISE_MODEL (or `noise`): white level, camera noise per frame, frames
    averaged per slot. fine_prefiltered[px]: the fine level again after the
    optional camera-side Gaussian (prefilter_frames fine_px), vs the ideal
    and vs the HDMI reference filtered the same way."""
    import cv2
    nm = dict(NOISE_MODEL, **(noise or {}))
    pats = fastscan_patterns()
    out = {"cam_px_per_col": cam_px_per_col, "mm_per_col": mm_per_col,
           "prefilter_px": prefilter_px, "region_dmd_rows_cols": list(region),
           "noise_model": nm, "parities": [p for p in parities],
           "fine_prefilter_px": [float(p) for p in fine_prefilter if p and p > 0], "sigmas": {}}
    H, W = DMD_SHAPE
    sq = []
    for k in range(STEPS):                       # square wave, centre at column 1.5
        c = np.arange(W)
        sq.append(np.repeat((((c + 2 * k) % 8) < 4)[None, :], H, 0))
    hdmi_fine = [f.astype(np.float32) / 255.0 for f in basis().generate()[2:2 + STEPS]]
    for sg in sigmas:
        obs = {p["slot"]: observe(p["image"], sg, cam_px_per_col=cam_px_per_col, region=region)
               for p in pats}
        x = obs[4][1]
        keep = np.ones(x.shape, bool)
        e = int(np.ceil(4 * prefilter_px))
        keep_f = np.zeros_like(keep)
        keep_f[e:-e, e:-e] = True
        row = {}
        for level, T in _LEVEL.items():
            slots = [SLOTS.index(f"{level}_k{k}") for k in range(STEPS)]
            fr = [obs[s][0] for s in slots]
            if level == "fine":
                r = _phase_err(fr, x, T, keep)
            else:
                fr = [cv2.GaussianBlur(f, (0, 0), prefilter_px, borderType=cv2.BORDER_REFLECT)
                      for f in fr]
                r = _phase_err(fr, x, T, keep_f)
                r["tolerance_cols"] = {"mid": 2.0, "coarse": 20.0}[level]
            r["rms_mm"] = round(r["rms_cols"] * mm_per_col, 4)
            r["p99_mm"] = round(r["p99_cols"] * mm_per_col, 4)
            nz = noise_rms_cols(r["modulation"], T, white_dn=nm["white_dn"],
                                noise_dn=nm["noise_dn"], frames=nm["frames"])
            r["noise_rms_cols"] = round(nz, 5)
            r["noise_rms_mm"] = round(nz * mm_per_col, 4)
            row[level] = r
        sqo = [observe(s, sg, cam_px_per_col=cam_px_per_col, region=region)[0] for s in sq]
        r = _phase_err(sqo, x - 1.5, 8, keep)
        r["rms_mm"] = round(r["rms_cols"] * mm_per_col, 4)
        r["p99_mm"] = round(r["p99_cols"] * mm_per_col, 4)
        r["closed_form"] = square_wave_error(sg, T=8, cam_px_per_col=cam_px_per_col)
        nz = noise_rms_cols(r["modulation"], 8, white_dn=nm["white_dn"],
                            noise_dn=nm["noise_dn"], frames=nm["frames"])
        r["noise_rms_cols"] = round(nz, 5)
        r["noise_rms_mm"] = round(nz * mm_per_col, 4)
        row["fine_square_wave"] = r
        # the fine level against the HDMI 8-bit fringe, square grid and diamond
        vs, pf = {}, {str(px): {"vs_hdmi": {}} for px in fine_prefilter if px and px > 0}
        fine_imgs = [pats[SLOTS.index(f"fine_k{k}")]["image"] for k in range(STEPS)]
        for par in parities:
            fr = [observe(im, sg, cam_px_per_col=cam_px_per_col, region=region,
                          parity=par)[0] for im in fine_imgs]
            ref = [observe(im, sg, cam_px_per_col=cam_px_per_col, region=region,
                           parity=par)[0] for im in hdmi_fine]
            v = _vs_reference(fr, ref, 8)
            v["bias_mm"] = round(v["bias_cols"] * mm_per_col, 4)
            v["rms_mm"] = round(v["rms_cols"] * mm_per_col, 4)
            vs["none" if par is None else str(par)] = v
            for px in pf:                          # the optional fine prefilter, both sides
                blur = [cv2.GaussianBlur(f, (0, 0), float(px), borderType=cv2.BORDER_REFLECT)
                        for f in fr + ref]
                ef = int(np.ceil(4 * float(px)))
                cut = (slice(ef, -ef), slice(ef, -ef))
                v = _vs_reference([b[cut] for b in blur[:STEPS]], [b[cut] for b in blur[STEPS:]], 8)
                v["bias_mm"] = round(v["bias_cols"] * mm_per_col, 4)
                v["rms_mm"] = round(v["rms_cols"] * mm_per_col, 4)
                pf[px]["vs_hdmi"]["none" if par is None else str(par)] = v
                if par is None:
                    r = _phase_err([b[cut] for b in blur[:STEPS]], x[cut], 8,
                                   np.ones(x[cut].shape, bool))
                    gain = min(1.0, 1.0 / (2.0 * np.sqrt(np.pi) * float(px)))
                    nz = noise_rms_cols(r["modulation"], 8, white_dn=nm["white_dn"],
                                        noise_dn=nm["noise_dn"], frames=nm["frames"]) * gain
                    r.update(rms_mm=round(r["rms_cols"] * mm_per_col, 4),
                             p99_mm=round(r["p99_cols"] * mm_per_col, 4),
                             noise_rms_cols=round(nz, 5), noise_rms_mm=round(nz * mm_per_col, 4),
                             lateral_blur_camera_px=float(px))
                    pf[px]["vs_ideal"] = r
        row["fine_vs_hdmi"] = vs
        if pf:
            row["fine_prefiltered"] = pf
        # end to end: 14 decode frames -> MultiFreqBasis
        frames = [obs[s][0] * 200.0 + 5.0 for s in DECODE_SLOTS]
        fld = decode(frames, "sim", prefilter_px=prefilter_px)
        cx = fld.corr_x
        ok = keep_f & np.isfinite(cx)
        err = (cx - x)[ok]
        row["end_to_end"] = {"valid_pct": round(100.0 * ok.sum() / keep_f.sum(), 3),
                             "unwrap_rejected_pct": fld.meta["per_axis"]["x"]["unwrap_rejected_pct"],
                             "median_err_cols": round(float(np.median(err)), 5) if err.size else None,
                             "rms_err_cols": round(float(np.std(err)), 5) if err.size else None,
                             "max_abs_err_cols": round(float(np.abs(err).max()), 4) if err.size else None}
        out["sigmas"][str(sg)] = row
    return out

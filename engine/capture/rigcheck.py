"""
Rig check: has the camera or the projector been nudged since the reference?

Pure functions (no rig, no library): the image shift between a reference
frame and a fresh one by phase correlation, refined to sub-pixel with a
locally upsampled DFT (Guizar-Sicairos et al. 2008), and the OK / MOVED
assessment. engine.api.rig_check captures the frames and stores the
reference; this module only does the numbers, so it is unit-tested on
synthetic shifts, rolls and zooms.

Two frames are compared:
  flood     the scene under white light: it moves when the CAMERA moves
  pattern   a seeded random block pattern: it moves when the camera OR the
            projector moves (the pattern is normalised by the flood, so the
            scene's albedo does not dominate)
A pattern shift with no flood shift points at the projector.

One global shift misses a roll or a zoom / focus nudge about the image centre
(the centre-weighted correlation averages it out). motion() therefore also
matches a grid of tiles and fits a similarity (shift + roll + scale) to their
local shifts; the motion that is judged is the largest image displacement over
the matched area (the fitted motion at its corners, or any tile's own shift).
OK needs both channels matched and the pattern's roll / zoom determined (the
pattern is the only channel that sees the projector): a channel that cannot be
matched is never read as "in place".
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

DEFAULT_TOL_PX = 0.5          # a shift above this is MOVED
MIN_PEAK = 0.35               # normalised correlation below this: no usable texture
HIGHPASS_SIGMA_PX = 8.0       # shading removed before matching (px of the downsampled image)
PATTERN_BLOCK_PX = 16         # DMD px per block of the random pattern
PATTERN_SEED = 20261006
TILE_GRID = 3                 # motion(): TILE_GRID x TILE_GRID tiles for the local shifts
TILE_UPSAMPLE = 20            # sub-pixel step of the tiles' shifts: 1/20 px (downsampled)
MIN_TILE_ISOTROPY = 0.1       # structure-tensor eigenvalue ratio: below it a tile holds one
                              # straight edge (its shift along the edge is not measured)
MIN_TILE_SPREAD = 0.25        # matched tile centres must lie this far apart (x the frame
                              # diagonal) for the roll / scale to be determined
TEXTURE_MIN_CONTRAST = 0.05   # high-passed rms / mean level: a frame with less shows no texture


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


def _highpass(img: np.ndarray, max_side: int) -> tuple:
    """Float image, downsampled by an integer factor to max_side, high-passed
    (shading and vignetting removed: they are common to both frames whatever
    moved) -> (image, factor, mean level before the high-pass, rms after it)."""
    from scipy import ndimage
    a = np.asarray(img, np.float64)
    if a.ndim == 3:
        a = a.mean(axis=2)
    a = np.nan_to_num(a, nan=float(np.nanmean(a)) if np.isfinite(a).any() else 0.0)
    f = max(1, int(np.ceil(max(a.shape) / float(max_side)))) if max_side else 1
    if f > 1:
        h, w = (a.shape[0] // f) * f, (a.shape[1] // f) * f
        a = a[:h, :w].reshape(h // f, f, w // f, f).mean(axis=(1, 3))
    level = float(np.mean(a))
    a = a - ndimage.gaussian_filter(a, HIGHPASS_SIGMA_PX, mode="nearest")
    return a, f, level, float(np.sqrt(np.mean(a * a)))


def _window(a: np.ndarray) -> np.ndarray:
    """Hann windowed (the image edges do not wrap around in the FFT)."""
    return a * np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))


def _prep(img: np.ndarray, max_side: int) -> tuple:
    """_highpass + _window -> (image, factor)."""
    a, f, _, _ = _highpass(img, max_side)
    return _window(a), f


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
    return _match(a, b, f, upsample)


def _match(a: np.ndarray, b: np.ndarray, f: int, upsample: int = 50) -> dict:
    """phase_shift on two _prep'ed images of the same size (f: their downsampling factor)."""
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


def _isotropy(a: np.ndarray) -> float:
    """Smaller / larger eigenvalue of the structure tensor of a (prepared) image: ~1 for a
    2-D texture, ~0 for one straight edge, whose shift along itself is not measured."""
    gy, gx = np.gradient(np.asarray(a, np.float64))
    jxx, jyy, jxy = float((gx * gx).sum()), float((gy * gy).sum()), float((gx * gy).sum())
    tr = jxx + jyy
    if tr <= 0:
        return 0.0
    disc = float(np.sqrt(max(0.25 * tr * tr - (jxx * jyy - jxy * jxy), 0.0)))
    hi, lo = 0.5 * tr + disc, 0.5 * tr - disc
    return float(max(lo, 0.0) / hi) if hi > 0 else 0.0


def _tile_boxes(shape, grid: int) -> list:
    """[(y0, y1, x0, x1)] of a grid x grid split of an image of this shape."""
    H, W = shape
    ys = np.linspace(0, H, grid + 1).astype(int)
    xs = np.linspace(0, W, grid + 1).astype(int)
    return [(int(ys[i]), int(ys[i + 1]), int(xs[j]), int(xs[j + 1]))
            for i in range(grid) for j in range(grid)]


def fit_similarity(pts, shifts, centre, weights=None) -> Optional[dict]:
    """Least-squares similarity motion d(p) = [[a, -b], [b, a]] (p - centre) + t from local
    shifts (N x 2, dx dy) at points pts (N x 2, x y) -> {a, b, tx, ty, rotation_deg, scale,
    rms_px}, or None with fewer than two points."""
    P = np.asarray(pts, np.float64).reshape(-1, 2) - np.asarray(centre, np.float64)
    D = np.asarray(shifts, np.float64).reshape(-1, 2)
    n = len(P)
    if n < 2:
        return None
    w = np.ones(n) if weights is None else np.asarray(weights, np.float64)
    A = np.zeros((2 * n, 4))
    A[0::2] = np.c_[P[:, 0], -P[:, 1], np.ones(n), np.zeros(n)]
    A[1::2] = np.c_[P[:, 1], P[:, 0], np.zeros(n), np.ones(n)]
    y = D.reshape(-1)
    sw = np.repeat(np.sqrt(np.maximum(w, 1e-9)), 2)
    sol = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)[0]
    a, b, tx, ty = (float(v) for v in sol)
    res = (A @ sol - y).reshape(-1, 2)
    return {"a": a, "b": b, "tx": tx, "ty": ty,
            "rotation_deg": float(np.degrees(np.arctan2(b, 1.0 + a))),
            "scale": float(np.hypot(1.0 + a, b)),
            "rms_px": float(np.sqrt(np.mean(np.sum(res * res, axis=1))))}


def similarity_at(sim: dict, pts, centre) -> np.ndarray:
    """The fitted similarity's displacement (N x 2, dx dy) at points pts (N x 2, x y)."""
    P = np.asarray(pts, np.float64).reshape(-1, 2) - np.asarray(centre, np.float64)
    a, b = sim["a"], sim["b"]
    return np.c_[a * P[:, 0] - b * P[:, 1] + sim["tx"], b * P[:, 0] + a * P[:, 1] + sim["ty"]]


def motion(ref: np.ndarray, img: np.ndarray, grid: int = TILE_GRID, upsample: int = 50,
           max_side: int = 1024) -> dict:
    """How `img` moved relative to `ref`, over the whole frame -> the global phase_shift
    keys {"dx", "dy", "shift_px", "peak"} plus
      matched       the frames could be matched (the global peak, or the tile similarity)
      similarity    roll + scale determined: >= 3 tiles matched, far enough apart
      rotation_deg, scale_pct, fit_rms_px   the similarity fitted to the tiles' shifts
      motion_px     the largest image displacement over the matched area: the fitted
                    motion at the matched tiles' outer corners, any matched tile's own
                    shift, the global shift (None when nothing was matched)
      texture       both frames show texture (a pattern that is visible but does not
                    match moved; one that is not visible was not checked)
      tiles_used / tiles, tile_shifts"""
    r = np.asarray(ref, np.float64)
    m = np.asarray(img, np.float64)
    if r.ndim == 3:
        r = r.mean(axis=2)
    if m.ndim == 3:
        m = m.mean(axis=2)
    if r.shape != m.shape:
        raise ValueError(f"reference {np.shape(ref)} and image {np.shape(img)} differ in size")
    hr, f, la, ea = _highpass(r, max_side)
    hm, _, lb, eb = _highpass(m, max_side)
    g = _match(_window(hr), _window(hm), f, upsample)
    contrast = [e / lv if lv > 1e-9 else 0.0 for e, lv in ((ea, la), (eb, lb))]
    texture = bool(min(contrast) >= TEXTURE_MIN_CONTRAST and ea > 0 and 0.5 <= eb / ea <= 2.0)
    H, W = r.shape
    centre = ((W - 1) / 2.0, (H - 1) / 2.0)
    tiles = []
    # tiles of the same high-passed (downsampled by f) images; tile px -> input px: (v + 0.5) f - 0.5
    for y0, y1, x0, x1 in _tile_boxes(hr.shape, int(grid)):
        if min(y1 - y0, x1 - x0) < 16:
            continue
        ta, tb = _window(hr[y0:y1, x0:x1]), _window(hm[y0:y1, x0:x1])
        t = _match(ta, tb, f, TILE_UPSAMPLE)
        iso = min(_isotropy(ta), _isotropy(tb))
        wgt = ta * ta
        s = float(wgt.sum())
        if s > 0:
            iy, ix = np.indices(wgt.shape)
            cy, cx = y0 + float((iy * wgt).sum()) / s, x0 + float((ix * wgt).sum()) / s
        else:
            cy, cx = (y0 + y1 - 1) / 2.0, (x0 + x1 - 1) / 2.0
        tiles.append({"box": [x0 * f, y0 * f, x1 * f, y1 * f],
                      "centre": [(cx + 0.5) * f - 0.5, (cy + 0.5) * f - 0.5],
                      "dx": t["dx"], "dy": t["dy"], "peak": t["peak"], "isotropy": round(iso, 3),
                      "used": bool(t["peak"] >= MIN_PEAK and iso >= MIN_TILE_ISOTROPY)})
    used = [t for t in tiles if t["used"]]
    sim = None
    if len(used) >= 3:
        P = np.array([t["centre"] for t in used])
        span = max(float(np.hypot(*(p - q))) for p in P for q in P)
        if span >= MIN_TILE_SPREAD * float(np.hypot(H, W)):
            sim = fit_similarity(P, [[t["dx"], t["dy"]] for t in used], centre,
                                 [t["peak"] for t in used])
    matched = bool(g["peak"] >= MIN_PEAK or sim is not None)
    cand = [float(np.hypot(t["dx"], t["dy"])) for t in used]
    if g["peak"] >= MIN_PEAK:
        cand.append(g["shift_px"])
    if sim is not None:
        x0 = min(t["box"][0] for t in used)
        y0 = min(t["box"][1] for t in used)
        x1 = max(t["box"][2] for t in used) - 1
        y1 = max(t["box"][3] for t in used) - 1
        cnr = similarity_at(sim, [[x0, y0], [x1, y0], [x0, y1], [x1, y1]], centre)
        cand.append(float(np.max(np.hypot(cnr[:, 0], cnr[:, 1]))))
    out = dict(g)
    out.update({"matched": matched, "similarity": sim is not None,
                "rotation_deg": round(sim["rotation_deg"], 4) if sim else None,
                "scale_pct": round(100.0 * (sim["scale"] - 1.0), 4) if sim else None,
                "fit_rms_px": round(sim["rms_px"], 3) if sim else None,
                "motion_px": round(max(cand), 3) if (matched and cand) else None,
                "texture": texture, "tiles_used": len(used), "tiles": len(tiles),
                "tile_shifts": [{"centre_px": [round(t["centre"][0], 1), round(t["centre"][1], 1)],
                                 "dx": t["dx"], "dy": t["dy"], "peak": t["peak"],
                                 "used": t["used"]} for t in tiles]})
    return out


def _matched(s: Optional[dict]) -> bool:
    return s is not None and bool(s.get("matched", s["peak"] >= MIN_PEAK))


def _motion_px(s: dict) -> float:
    v = s.get("motion_px")
    return float(s["shift_px"] if v is None else v)


def check_tol(tol_px) -> float:
    """The MOVED tolerance as a finite px number >= 0 (ValueError otherwise: a NaN or an
    infinite tolerance would read every motion as 'in place')."""
    try:
        t = float(tol_px)
    except (TypeError, ValueError):
        raise ValueError(f"rig check tolerance must be a number of px, got {tol_px!r}") from None
    if not math.isfinite(t) or t < 0:
        raise ValueError(f"rig check tolerance must be a finite number of px >= 0, got {tol_px!r}")
    return t


def assess(flood: Optional[dict], pattern: Optional[dict], tol_px: float = DEFAULT_TOL_PX,
           mm_per_px: Optional[float] = None, distance_mm: Optional[float] = None) -> dict:
    """OK / MOVED / UNKNOWN from the flood and pattern motions (motion() dicts; a bare
    phase_shift dict is a shift only: roll / zoom not determined; None = not measured).
    MOVED names what moved; mm only when mm_per_px is known (never a px number reported as
    mm). Fail closed: OK needs the pattern matched with its roll / zoom determined (the
    pattern is the only channel that sees the projector) and every matched channel within
    tol_px; a flood with no texture is fine (the pattern sees the camera too), a flood that
    shows texture but no longer matches is not. tol_px must be finite and >= 0
    (ValueError)."""
    tol_px = check_tol(tol_px)
    cam = _motion_px(flood) if _matched(flood) else None
    rig = _motion_px(pattern) if _matched(pattern) else None
    vals = [v for v in (cam, rig) if v is not None]
    worst = (float("nan") if any(math.isnan(v) for v in vals) else max(vals)) if vals else None
    mm = round(worst * mm_per_px, 3) if (mm_per_px and worst is not None) else None
    at = f" at {distance_mm:.0f} mm" if (mm is not None and distance_mm) else ""
    size = (f"~{worst:.1f} px" + (f" (~{mm:.2f} mm{at})" if mm is not None else "")
            if worst is not None else "")

    def rollzoom(s):
        if not s or not s.get("similarity"):
            return ""
        return f", roll {s['rotation_deg']:+.3f} deg, scale {s['scale_pct']:+.3f} %"

    def verdict(status, moved, message, px=worst, smm=mm):
        return {"status": status, "moved": moved,
                "shift_px": round(px, 3) if px is not None else None,
                "shift_mm": smm, "message": message}

    if worst is not None and not worst <= tol_px:          # NaN motion: MOVED, never OK
        if cam is not None and not cam <= tol_px:
            what, s = "camera", flood
        elif cam is not None and rig is not None:
            what, s = "projector", pattern
        else:
            what, s = "camera or projector", (pattern if rig is not None else flood)
        return verdict("MOVED", what, f"{what} moved {size} (largest image motion over the "
                                      f"frame{rollzoom(s)}): re-home / recalibrate before measuring")
    if rig is None:
        if pattern is not None and pattern.get("texture"):
            what = "projector" if cam is not None else "camera or projector"
            return verdict("MOVED", what,
                           f"{what} moved: the pattern is visible but no longer matches the "
                           f"reference (match {pattern['peak']:.2f} < {MIN_PEAK:g}): moved, "
                           "rolled or zoomed / refocused beyond what the check measures - "
                           "re-home / recalibrate before measuring", px=None, smm=None)
        if cam is None:
            return verdict("UNKNOWN", None, "no usable texture in the frames (lens cap? projector "
                                            "off? an empty, featureless scene?): rig position "
                                            "not checked", px=None, smm=None)
        return verdict("UNKNOWN", None,
                       f"projector not checked: no usable pattern in the frames (projector off, "
                       f"dark or blocked?); the camera alone is in place ({size}): rig position "
                       "not confirmed", px=None, smm=None)
    if not pattern.get("similarity"):
        return verdict("UNKNOWN", None,
                       f"roll / zoom not checked: the pattern was matched in "
                       f"{pattern.get('tiles_used', 0)} of {pattern.get('tiles', 0)} parts of the "
                       f"frame (shift {size}): light more of the scene, or take the reference "
                       "on a surface that fills the view", px=None, smm=None)
    if cam is None and flood is not None and flood.get("texture"):
        return verdict("UNKNOWN", None,
                       f"the flood shows texture but no longer matches the reference (match "
                       f"{flood['peak']:.2f}): the scene changed since the reference (station "
                       "not in its usual state?) or the camera moved beyond what the check "
                       "measures: rig position not confirmed", px=None, smm=None)
    note = "" if cam is not None else " (no flood texture: the camera is checked through the pattern)"
    return verdict("OK", None, f"rig in place: largest image motion {size} <= {tol_px:g} px"
                               f"{rollzoom(pattern)}{note}")

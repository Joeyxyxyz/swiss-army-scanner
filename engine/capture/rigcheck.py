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

Which device moved (who_moved): a camera move shifts the scene in its view and the
projected pattern with it (the pattern lies on the scene); a projector move - or a move of
its picture: focus / zoom, the picture the computer sends - shifts the pattern AGAINST the
scene and leaves the flood. So the camera's motion is the flood's, the projector's is the
pattern's motion relative to the flood, tile by tile. A tile that moves with neither (a part
placed since the reference changes the scene there) is set aside before either is judged.
MOVED then names the camera, the projector (or its picture), both, or "camera or projector"
when it can't tell (the flood is not matched, or neither moved beyond the tolerance on its own
and the frames do not show two small moves on the reference's scene: _mover), and says what to
do: a small camera turn can be re-homed, a projector move cannot.

One global shift misses a roll or a zoom / focus nudge about the image centre
(the centre-weighted correlation averages it out). motion() therefore also
matches a grid of tiles and fits a similarity (shift + roll + scale) to their
local shifts; the motion that is judged is the largest image displacement over
the matched area (the fitted motion at its corners, or any tile's own shift).
OK needs both channels matched and the pattern's roll / zoom determined (the
pattern is the only channel that sees the projector): a channel that cannot be
matched is never read as "in place".

Re-home (rehome_fit): a small CAMERA rotation about its own centre moves the
whole image by one homography, whatever the scene's depth. camera_rotation()
fits it from the flood's tile shifts on undistorted coordinates about the
principal point (shift -> pan / tilt, roll), and rehome_fit() says whether the
motion is one a rotation-only correction of the stereo calibration can take
out: camera-like (the pattern moves with the flood), small, well registered.
A camera TRANSLATION looks the same on a flat scene and is not corrected: the
absolute depth can keep a small offset; a full calibration is the reference.
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
REHOME_MAX_PX = 5.0           # re-home: a larger image motion is not a nudge - recalibrate
REHOME_GRID = 5               # re-home: REHOME_GRID x REHOME_GRID tiles (the check's own
                              # downsampling and high-pass: the same texture band)
REHOME_TILE_UPSAMPLE = 50     # re-home: sub-pixel step of the tiles' shifts, 1/50 px
REHOME_MIN_TILES = 6          # re-home: flood tiles matched, and agreeing on one rotation
REHOME_MIN_PEAK = 0.5         # re-home: median match of the flood tiles used
REHOME_INLIER_PX = 0.3        # re-home: a tile agrees when its shift is this close to the fit
REHOME_MIN_AGREE = 0.6        # re-home: share of the matched tiles (and of their spread over
                              # the frame) that must agree on the one rotation
REHOME_AGREE_PX = 0.4         # re-home: pattern tiles vs the flood's rotation, rms (or
REHOME_AGREE_FRACTION = 0.15  #          this fraction of the motion, whichever is larger)
WHO_INLIER_PX = 0.3           # who moved: a tile further than this (and 3 x the median) from the
                              # motion the other tiles agree on is set aside (a changed scene);
                              # the pattern moved WITH the scene (the camera alone moved) while
                              # the two stay within the tolerance, or within
                              # REHOME_AGREE_FRACTION of the camera's motion (the re-home's own
                              # rule: tile errors grow with a large move)
# what a MOVED says to do, per device it names
MOVED_FIX = {
    "camera": "a small camera turn can be re-homed, else recalibrate before measuring",
    "projector": ("a projector move can't be re-homed: put the projector and its picture back "
                  "(mount, focus / zoom, the computer's display settings) and check again, else "
                  "recalibrate before measuring"),
    "camera and projector": "recalibrate before measuring (a projector move can't be re-homed)",
    "camera or projector": ("put the rig right and check again, else recalibrate before "
                            "measuring")}


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
           max_side: int = 1024, tile_upsample: int = TILE_UPSAMPLE) -> dict:
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
      tiles_used / tiles, tile_shifts (each: centre_px, box_px [x0, y0, x1, y1] input px,
                    dx, dy, peak, used)"""
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
        t = _match(ta, tb, f, int(tile_upsample))
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
    out.update({"shape": [int(H), int(W)], "matched": matched, "similarity": sim is not None,
                "rotation_deg": round(sim["rotation_deg"], 4) if sim else None,
                "scale_pct": round(100.0 * (sim["scale"] - 1.0), 4) if sim else None,
                "fit_rms_px": round(sim["rms_px"], 3) if sim else None,
                "motion_px": round(max(cand), 3) if (matched and cand) else None,
                "texture": texture, "tiles_used": len(used), "tiles": len(tiles),
                "tile_shifts": [{"centre_px": [round(t["centre"][0], 1), round(t["centre"][1], 1)],
                                 "box_px": [int(v) for v in t["box"]],
                                 "dx": t["dx"], "dy": t["dy"], "peak": t["peak"],
                                 "used": t["used"]} for t in tiles]})
    return out


def _matched(s: Optional[dict]) -> bool:
    return s is not None and bool(s.get("matched", s["peak"] >= MIN_PEAK))


# ------------------------------------------------------------ re-home (camera rotation)

def _normalised(px, K, dist=None, iters: int = 25) -> np.ndarray:
    """Distorted pixels (N, 2) -> ideal normalised camera coordinates (N, 2): the Brown
    model (k1, k2, p1, p2, k3) inverted by fixed-point iteration, run to convergence."""
    px = np.asarray(px, np.float64).reshape(-1, 2)
    K = np.asarray(K, np.float64)
    coef = np.zeros(5) if dist is None else np.ravel(np.asarray(dist, np.float64))
    k1, k2, p1, p2, k3 = (list(coef) + [0.0] * 5)[:5]
    xd = (px[:, 0] - K[0, 2]) / K[0, 0]
    yd = (px[:, 1] - K[1, 2]) / K[1, 1]
    x, y = xd.copy(), yd.copy()
    if any((k1, k2, p1, p2, k3)):
        for _ in range(iters):
            r2 = x * x + y * y
            rad = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
            x, y = ((xd - (2 * p1 * x * y + p2 * (r2 + 2 * x * x))) / rad,
                    (yd - (p1 * (r2 + 2 * y * y) + 2 * p2 * x * y)) / rad)
    return np.stack([x, y], axis=1)


def _rotvec(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> rotation vector (axis x angle, rad)."""
    v = 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    sn = float(np.linalg.norm(v))
    ang = float(np.arctan2(sn, 0.5 * (np.trace(R) - 1.0)))
    return v * (ang / sn) if sn > 1e-12 else np.zeros(3)


def rotation_from_rotvec(w) -> np.ndarray:
    """Rotation vector (rad) -> rotation matrix (Rodrigues)."""
    w = np.asarray(w, np.float64).reshape(3)
    ang = float(np.linalg.norm(w))
    if ang < 1e-15:
        return np.eye(3)
    k = w / ang
    Kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(ang) * Kx + (1.0 - np.cos(ang)) * (Kx @ Kx)


def camera_rotation(pts, shifts, K, dist=None, weights=None) -> Optional[dict]:
    """The small camera rotation (about the camera's own centre) that explains local image
    shifts: pts (N, 2) px in the reference image, shifts (N, 2) px = where that texture is
    in the fresh image minus where it was. Fitted on UNDISTORTED coordinates about the
    principal point: the similarity (shift + roll + scale) there gives the numbers a person
    reads - pan = shift_x / fx, tilt = shift_y / fy, roll - and the rotation itself is the
    least-squares rotation between the two sets of viewing rays (exact at any field
    position; the similarity is its small-angle form).

    Convention (locked by the mock test that turns the simulated camera): with a point at
    X_old in the camera frame before the move and X_new after it,
        X_new = R_new_from_old @ X_old          X_old = R @ X_new        (R = its transpose)
    so a stereo calibration X_proj = R0 @ X_cam + t0 taken before the move becomes
        R1 = R0 @ R,   t1 = t0                  (the projector did not move).
    Image motion of a pure rotation w (rad, the rotation vector of R_new_from_old) at the
    principal point: dx = fx * w_y, dy = -fy * w_x; roll (x towards y) = w_z.
    -> {R, R_new_from_old, rvec_deg, rotation_deg, pan_deg, tilt_deg, roll_deg, shift_px,
        scale_pct, similarity_rms_px, residual_px (rms of the shifts vs the fitted pure
        rotation), residual_max_px, n}; None with fewer than 3 points."""
    P = np.asarray(pts, np.float64).reshape(-1, 2)
    D = np.asarray(shifts, np.float64).reshape(-1, 2)
    if len(P) < 3 or len(P) != len(D):
        return None
    K = np.asarray(K, np.float64)
    f = np.array([K[0, 0], K[1, 1]])
    c = np.array([K[0, 2], K[1, 2]])
    a, b = _normalised(P, K, dist), _normalised(P + D, K, dist)
    ua, ub = a * f + c, b * f + c                         # undistorted px
    w = np.ones(len(P)) if weights is None else np.maximum(np.asarray(weights, np.float64), 1e-9)
    sim = fit_similarity(ua, ub - ua, c, w)
    ra = np.c_[a, np.ones(len(a))]
    rb = np.c_[b, np.ones(len(b))]
    ra /= np.linalg.norm(ra, axis=1, keepdims=True)
    rb /= np.linalg.norm(rb, axis=1, keepdims=True)
    U, _, Vt = np.linalg.svd((ra * w[:, None]).T @ rb)    # rb ~ Rn ra (Kabsch)
    d = 1.0 if np.linalg.det(Vt.T @ U.T) >= 0 else -1.0
    Rn = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    pr = ra @ Rn.T
    res = pr[:, :2] / pr[:, 2:3] * f + c - ub
    err = np.hypot(res[:, 0], res[:, 1])
    rv = _rotvec(Rn)
    return {"R": Rn.T, "R_new_from_old": Rn,
            "rvec_deg": [float(v) for v in np.degrees(rv)],
            "rotation_deg": float(np.degrees(np.linalg.norm(rv))),
            "pan_deg": float(np.degrees(rv[1])), "tilt_deg": float(np.degrees(-rv[0])),
            "roll_deg": float(np.degrees(rv[2])),
            "shift_px": [float(sim["tx"]), float(sim["ty"])],
            "scale_pct": float(100.0 * (sim["scale"] - 1.0)),
            "similarity_rms_px": float(sim["rms_px"]),
            "residual_px": float(np.sqrt(np.average(err * err, weights=w))),
            "residual_max_px": float(err.max()), "n": int(len(P))}


def _used_tiles(m: Optional[dict]) -> tuple:
    """(centres (N, 2), shifts (N, 2), peaks (N,)) of the tiles a motion() used."""
    ts = [t for t in ((m or {}).get("tile_shifts") or []) if t.get("used")]
    return (np.array([t["centre_px"] for t in ts], np.float64).reshape(-1, 2),
            np.array([[t["dx"], t["dy"]] for t in ts], np.float64).reshape(-1, 2),
            np.array([t["peak"] for t in ts], np.float64))


def _rotation_errors(R_new_from_old, pts, shifts, K, dist=None) -> np.ndarray:
    """|measured - predicted| (undistorted px) of image shifts under a pure camera rotation."""
    K = np.asarray(K, np.float64)
    f = np.array([K[0, 0], K[1, 1]])
    c = np.array([K[0, 2], K[1, 2]])
    P = np.asarray(pts, np.float64).reshape(-1, 2)
    a = _normalised(P, K, dist)
    ra = np.c_[a, np.ones(len(a))] @ np.asarray(R_new_from_old, np.float64).T
    want = ra[:, :2] / ra[:, 2:3] * f + c
    got = _normalised(P + np.asarray(shifts, np.float64).reshape(-1, 2), K, dist) * f + c
    return np.hypot(*(got - want).T)


def _robust_rotation(P, D, w, K, dist=None) -> tuple:
    """camera_rotation on the tiles that agree on ONE rotation -> (fit or None, inliers).
    A tile whose texture does not pin its shift (one soft edge, a nearly flat patch) can
    match well at the wrong place; it must not drag the rotation, and it must not count as
    agreement either. Start from the median shift (a pure pan / tilt), then refit on the
    tiles within a shrinking distance of the fit, ending at REHOME_INLIER_PX."""
    K = np.asarray(K, np.float64)
    med = np.median(D, axis=0)
    R = rotation_from_rotvec([-med[1] / K[1, 1], med[0] / K[0, 0], 0.0])
    keep = np.ones(len(P), bool)
    fit = None
    for thr in (1.0, 0.6, 0.4, REHOME_INLIER_PX, REHOME_INLIER_PX):
        keep = _rotation_errors(R, P, D, K, dist) <= thr
        if keep.sum() < 3:
            return None, keep
        fit = camera_rotation(P[keep], D[keep], K, dist, w[keep])
        R = fit["R_new_from_old"]
    keep = _rotation_errors(R, P, D, K, dist) <= REHOME_INLIER_PX
    return fit, keep


def rehome_fit(flood: Optional[dict], pattern: Optional[dict], K, dist=None,
               max_px: float = REHOME_MAX_PX) -> dict:
    """Can this motion be re-homed (a rotation-only correction of the stereo calibration)?
    flood / pattern: motion() of the reference vs the fresh frames (on a finer grid than the
    check's: REHOME_GRID). -> {"applicable": bool, "why": the refusal (None when
    applicable), "motion_px" (the fitted rotation's largest image motion over the frame),
    "tiles_used", "tiles", "tiles_agree", "match", "pattern_rms_px", and the
    camera_rotation() numbers}. Applicable only when ALL of these hold (fail closed):
      - the flood is matched in >= REHOME_MIN_TILES tiles with a median match >=
        REHOME_MIN_PEAK
      - those tiles agree on ONE camera rotation: >= REHOME_MIN_TILES of them and >=
        REHOME_MIN_AGREE of the matched ones lie within REHOME_INLIER_PX of it, spread over
        the frame (a zoom, a refocus or a changed scene does not)
      - the motion is small: <= max_px
      - the pattern moves with the flood (the projector did not move): the pattern tiles'
        shifts fit the SAME rotation within max(REHOME_AGREE_PX, REHOME_AGREE_FRACTION x
        the motion), rms."""
    out = {"applicable": False, "why": None, "motion_px": None,
           "tiles_used": int((flood or {}).get("tiles_used") or 0),
           "tiles": int((flood or {}).get("tiles") or 0), "tiles_agree": None, "match": None,
           "pattern_rms_px": None, "max_px": float(max_px)}

    def no(why: str) -> dict:
        out["why"] = why
        return out

    if not _matched(flood) or not flood.get("similarity"):
        return no("the flood (the scene's own texture) could not be registered over the frame "
                  f"({out['tiles_used']} of {out['tiles']} tiles matched): no camera rotation "
                  "can be fitted")
    P, Dv, pk = _used_tiles(flood)
    out["match"] = round(float(np.median(pk)), 3) if len(pk) else None
    if len(P) < REHOME_MIN_TILES:
        return no(f"the registration is poor: the flood matched in {len(P)} of {out['tiles']} "
                  f"tiles, {REHOME_MIN_TILES} are needed")
    if out["match"] < REHOME_MIN_PEAK:
        return no(f"the registration is poor: flood match {out['match']:.2f} < "
                  f"{REHOME_MIN_PEAK:g}")
    rot, keep = _robust_rotation(P, Dv, pk, K, dist)
    n_in = int(keep.sum())
    out["tiles_agree"] = n_in
    if rot is None:
        return no(f"the registration is poor, or the motion is not a camera rotation: fewer "
                  f"than 3 of the {len(P)} matched flood tiles agree on one rotation")
    out.update(rot)
    # the size of the move: the fitted rotation over the whole frame (not one tile's shift)
    shape = flood.get("shape")
    if shape:
        H, W = int(shape[0]), int(shape[1])
        box = np.array([[0, 0], [W - 1, 0], [0, H - 1], [W - 1, H - 1]], np.float64)
    else:
        box = np.array([P.min(0), P.max(0), [P[:, 0].min(), P[:, 1].max()],
                        [P[:, 0].max(), P[:, 1].min()]])
    Kc = np.asarray(K, np.float64)
    pts = np.vstack([box, [[Kc[0, 2], Kc[1, 2]]]])
    mv = float(_rotation_errors(rot["R_new_from_old"], pts, np.zeros_like(pts), K, dist).max())
    out["motion_px"] = round(mv, 3)
    if not mv <= max_px:
        return no(f"the motion is too large for a re-home: {mv:.1f} px > {max_px:g} px - "
                  "recalibrate")

    def span(Q) -> float:
        return float(np.hypot(*(Q.max(0) - Q.min(0)))) if len(Q) else 0.0

    if (n_in < REHOME_MIN_TILES or n_in < REHOME_MIN_AGREE * len(P)
            or span(P[keep]) < REHOME_MIN_AGREE * span(P)):
        return no(f"the registration is poor, or the motion is not a camera rotation: only "
                  f"{n_in} of the {len(P)} matched flood tiles agree on one rotation within "
                  f"{REHOME_INLIER_PX:g} px (scale {rot['scale_pct']:+.3f} %; a zoom / "
                  "refocus, a changed scene or weak texture)")
    if not _matched(pattern) or not pattern.get("similarity"):
        return no("the pattern could not be registered over the frame, so the projector is "
                  "not known to be in place")
    Pp, Dp, _ = _used_tiles(pattern)
    if len(Pp) < 3:
        return no("the pattern matched in too few tiles to compare it with the flood")
    e = _rotation_errors(rot["R_new_from_old"], Pp, Dp, K, dist)
    out["pattern_rms_px"] = round(float(np.sqrt(np.mean(e * e))), 3)
    lim = max(REHOME_AGREE_PX, REHOME_AGREE_FRACTION * mv)
    if not out["pattern_rms_px"] <= lim:
        return no(f"the motion is not camera-like: the projected pattern does not move with "
                  f"the scene ({out['pattern_rms_px']:.2f} px rms apart, limit {lim:.2f} px) - "
                  "the projector moved: recalibrate")
    out["applicable"] = True
    return out


def _motion_px(s: dict) -> float:
    v = s.get("motion_px")
    return float(s["shift_px"] if v is None else v)


# ------------------------------------------------------------ who moved

def _field_motion(pts, shifts, weights, boxes=None, diag: Optional[float] = None) -> Optional[dict]:
    """The largest image displacement of ONE motion field over the area its tiles cover, with
    the tiles that do not move with the others set aside -> {"px", "tiles", "set_aside"}, None
    without a tile. Three or more tiles spread over the frame (MIN_TILE_SPREAD x diag): a
    similarity (shift + roll + scale) is fitted, and the worst tile is dropped while it lies
    further than WHO_INLIER_PX and 3 x the median from the fit of the rest (at least three
    stay); px = the fit at the corners of the kept tiles' area or a kept tile's own shift,
    the larger. Fewer tiles, or too close together: the median tile."""
    P = np.asarray(pts, np.float64).reshape(-1, 2)
    D = np.asarray(shifts, np.float64).reshape(-1, 2)
    W = np.asarray(weights, np.float64).reshape(-1)
    n = len(P)
    if n == 0:
        return None
    keep = np.ones(n, bool)
    sim, c = None, P.mean(axis=0)
    while keep.sum() >= 3:
        Q = P[keep]
        span = max(float(np.hypot(*(p - q))) for p in Q for q in Q)
        if diag is not None and span < MIN_TILE_SPREAD * float(diag):
            sim = None
            break
        sim = fit_similarity(Q, D[keep], c, W[keep])
        r = np.hypot(*(similarity_at(sim, P, c) - D).T)
        worst = int(np.argmax(np.where(keep, r, -1.0)))
        if keep.sum() > 3 and r[worst] > max(WHO_INLIER_PX, 3.0 * float(np.median(r[keep]))):
            keep[worst] = False
            continue
        break
    mag = np.hypot(*D[keep].T)
    if sim is None:
        px = float(np.median(np.hypot(*D.T)))
    else:
        if boxes is not None:
            B = np.asarray(boxes, np.float64).reshape(-1, 4)[keep]
            x0, y0, x1, y1 = B[:, 0].min(), B[:, 1].min(), B[:, 2].max() - 1, B[:, 3].max() - 1
        else:
            (x0, y0), (x1, y1) = P[keep].min(axis=0), P[keep].max(axis=0)
        cnr = similarity_at(sim, [[x0, y0], [x1, y0], [x0, y1], [x1, y1]], c)
        px = max(float(np.max(np.hypot(cnr[:, 0], cnr[:, 1]))), float(mag.max()))
    return {"px": round(px, 3), "tiles": int(keep.sum()), "set_aside": int(n - keep.sum())}


def _diag(m: Optional[dict]) -> Optional[float]:
    sh = (m or {}).get("shape")
    return float(np.hypot(*sh)) if sh else None


def _tiles(m: Optional[dict]) -> list:
    return [t for t in ((m or {}).get("tile_shifts") or []) if t.get("used")]


def _boxes(tiles: list) -> Optional[list]:
    return [t["box_px"] for t in tiles] if tiles and all(t.get("box_px") for t in tiles) else None


def who_moved(flood: Optional[dict], pattern: Optional[dict]) -> dict:
    """The camera's and the projector's own motion, from the two channels (module docstring)
    -> {"camera_motion_px": the scene's motion in the camera's view (the flood's tiles),
    "projector_motion_px": the projected picture's motion AGAINST the scene (pattern minus
    flood, tile by tile: the same tile of the same grid), "tiles_set_aside": tiles that moved
    with neither (a part placed since the reference, a glint)}; a number is None when its
    channel is not matched. A channel matched without tiles (a bare phase_shift) counts with
    its global shift."""
    out = {"camera_motion_px": None, "projector_motion_px": None, "tiles_set_aside": 0}
    if not _matched(flood):
        return out
    ft = _tiles(flood)
    cam = _field_motion([t["centre_px"] for t in ft], [[t["dx"], t["dy"]] for t in ft],
                        [t["peak"] for t in ft], _boxes(ft), _diag(flood))
    if cam is None:
        out["camera_motion_px"] = round(_motion_px(flood), 3)
    else:
        out["camera_motion_px"], out["tiles_set_aside"] = cam["px"], cam["set_aside"]
    if not _matched(pattern):
        return out
    fa, pa = (list((m or {}).get("tile_shifts") or []) for m in (flood, pattern))
    if all(t.get("box_px") for t in fa + pa):
        by_box = {tuple(t["box_px"]): t for t in pa}
        pairs = [(f, by_box.get(tuple(f["box_px"]))) for f in fa]
    else:                                   # no boxes given: the same place in the same grid
        pairs = list(zip(fa, pa)) if len(fa) == len(pa) else []
    pairs = [(f, p) for f, p in pairs if p is not None and f.get("used") and p.get("used")]
    rel = _field_motion([f["centre_px"] for f, _ in pairs],
                        [[p["dx"] - f["dx"], p["dy"] - f["dy"]] for f, p in pairs],
                        [min(f["peak"], p["peak"]) for f, p in pairs],
                        _boxes([f for f, _ in pairs]), _diag(flood))
    if rel is not None:
        out["projector_motion_px"] = rel["px"]
        out["tiles_set_aside"] = max(out["tiles_set_aside"], rel["set_aside"])
    elif flood["peak"] >= MIN_PEAK and pattern["peak"] >= MIN_PEAK:
        out["projector_motion_px"] = round(float(np.hypot(pattern["dx"] - flood["dx"],
                                                          pattern["dy"] - flood["dy"])), 3)
    return out


def _unchanged(flood: Optional[dict], pattern: Optional[dict], who: dict) -> bool:
    """The frames show the reference's scene: every tile of both channels matched, none set
    aside. A part placed since the reference shows as tiles that no longer match, or that moved
    with neither device; the tiles around it match in part and leak up to ~0.35 px into each
    device's motion (synthetic parts after an empty reference, nothing moved)."""
    if who.get("tiles_set_aside"):
        return False
    for m in (flood, pattern):
        n = (m or {}).get("tiles")
        if not n or (m or {}).get("tiles_used") != n:
            return False
    return True


def _mover(cam: Optional[float], who: dict, tol_px: float, worst: Optional[float] = None,
           unchanged: bool = False) -> str:
    """Which device a MOVED names: "camera" (the pattern moved with the scene), "projector"
    (the pattern moved against it, the camera's view within the tolerance), "camera and
    projector", or "camera or projector" (the camera's view can't be measured, the pattern
    can't be compared with it, or neither moved beyond the tolerance on its own). A device
    moved when its own motion is beyond tol_px. When neither is, two small moves are named
    against half of it only on the reference's scene (`unchanged`, _unchanged) and when they
    explain the frame's largest motion `worst` (it exceeds the sum of their motions by no more
    than half the tolerance): a part placed since the reference leaks a few tenths of a px into
    each device's motion, which alone named the camera (a re-home!) or both. The pattern moved
    with the scene while it stays within the limit, or within REHOME_AGREE_FRACTION of the
    camera's motion. NaN reads as moved, and never as explained."""
    cpx, ppx = who.get("camera_motion_px"), who.get("projector_motion_px")
    if cam is None or ppx is None:
        return "camera or projector"
    cpx = cam if cpx is None else cpx
    together = REHOME_AGREE_FRACTION * cpx if math.isfinite(cpx) else 0.0

    def named(lim: float) -> Optional[str]:
        proj, camera = not ppx <= max(lim, together), not cpx <= lim
        return ("camera and projector" if proj and camera else "projector" if proj else
                "camera" if camera else None)

    what = named(tol_px)
    half = 0.5 * tol_px
    if what is None and unchanged and worst is not None and worst - (cpx + ppx) <= half:
        what = named(half)
    return what or "camera or projector"


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
    MOVED names what moved ("moved": camera | projector | camera and projector | camera or
    projector, who_moved / _mover) and what to do about it (MOVED_FIX: a small camera turn can
    be re-homed, a projector move can't); every verdict carries who_moved()'s numbers
    (camera_motion_px, projector_motion_px, tiles_set_aside). mm only when mm_per_px is known
    (never a px number reported as mm). Fail closed: OK needs the pattern matched with its
    roll / zoom determined (the pattern is the only channel that sees the projector) and
    every matched channel within tol_px; a flood with no texture is fine (the pattern sees
    the camera too), a flood that shows texture but no longer matches is not. tol_px must be
    finite and >= 0 (ValueError)."""
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

    who = who_moved(flood, pattern)

    def verdict(status, moved, message, px=worst, smm=mm):
        return {"status": status, "moved": moved,
                "shift_px": round(px, 3) if px is not None else None,
                "shift_mm": smm, "message": message, **who}

    def pm(v):
        return f"~{v:.1f} px" + (f" (~{v * mm_per_px:.2f} mm)" if mm_per_px else "")

    if worst is not None and not worst <= tol_px:          # NaN motion: MOVED, never OK
        what = _mover(cam, who, tol_px, worst, _unchanged(flood, pattern, who))
        cpx, ppx = who["camera_motion_px"], who["projector_motion_px"]
        frame = f"largest image motion over the frame {size}"
        if what == "camera":
            s = flood if not cam <= tol_px else pattern
            text = (f"camera moved {size} (largest image motion over the frame{rollzoom(s)}; "
                    "the projected pattern moved with the scene)")
        elif what == "projector":
            text = (f"projector (or its picture) moved {pm(ppx)} against the scene, the "
                    f"camera's view {cpx:.1f} px ({frame}{rollzoom(pattern)})")
        elif what == "camera and projector":
            text = (f"camera and projector moved: the camera's view {pm(cpx)}, the projector's "
                    f"picture {pm(ppx)} against the scene ({frame}{rollzoom(pattern)})")
        else:
            why = ("the scene's own texture is not matched" if cam is None else
                   "the pattern can't be compared with the scene" if ppx is None else
                   f"neither the camera's view ({cpx:.1f} px) nor the projector's picture "
                   f"({ppx:.1f} px) moved that far on its own - two small moves together, or "
                   "has the scene changed since the reference?")
            text = (f"camera or projector moved {size} (largest image motion over the frame"
                    f"{rollzoom(pattern if rig is not None else flood)}; which one can't be "
                    f"told: {why})")
        return verdict("MOVED", what, f"{text}: {MOVED_FIX[what]}")
    if rig is None:
        if pattern is not None and pattern.get("texture"):
            what = "projector" if cam is not None else "camera or projector"
            name = "projector (or its picture)" if cam is not None else what
            return verdict("MOVED", what,
                           f"{name} moved: the pattern is visible but no longer matches the "
                           f"reference (match {pattern['peak']:.2f} < {MIN_PEAK:g}): moved, "
                           "rolled or zoomed / refocused beyond what the check measures - "
                           f"{MOVED_FIX[what]}", px=None, smm=None)
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

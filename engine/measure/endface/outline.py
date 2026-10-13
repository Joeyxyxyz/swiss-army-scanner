"""
The rim outline of an end face in its flood image: how large it is and how round.

The inspected flat stops short of the rim (the chamfer / rounded edge is never inspected for
dents), so a dent ON the edge does not show in the height map: it shows as the bright flat's
OUTLINE moving in (a notch) or out (a bulge). And the outline's size is the one dimension of
the part that is known beforehand (the face diameter given to locate / the golden / the
recipe): a part that reads 16 % small is not a small part, it is a lens that was zoomed
after the calibration (bench 2026-10-05: every lateral size 0.84 x, a 19.5 mm face read
16.3 mm and nothing warned).

  measure   sub-pixel radial edge of the flood (lit - dark) on N_ANGLES (360) rays from the
            located centre, sampled IN THE FACE PLANE (mm; through the camera model or a
            homography fitted on the measured points), so radius and angle are those of the
            face frame (theta 0 = camera right, 90 = camera down, as every trigger):
              * coarse: per ray the first fall below half-way between the face level
                (the angle-median radial profile near the centre) and the darkest ring,
                0.15 .. 1.6 x the nominal radius out; the circle half of those falls lie on
                gives the disc's centre and radius (a mis-scaled face, or a part a few mm
                off the centre it was looked for at, is still found; half a face is not
                taken for a small one)
              * per ray: the first place, going outward, where the Gaussian-smoothed (1 px)
                profile falls below half-way between that ray's face level (the 30th
                percentile just inside the rim: a glint does not raise it) and its dark
                level (linear sub-sample interpolation): a glint inside the face does not
                move it, a dark defect at the rim does. The face level of a ray is its own,
                but a ray under half of what the rays around it show (the median over
                +-45 deg, itself at least half the whole rim's) takes THEIR level: inside a
                chip the ray's own level is the light that leaks into it, and half of THAT
                would put the edge back on the rim the part no longer has (a chip wider
                than that window reads up to ~0.3 mm shallow). A ray that is already dark
                where the search starts (0.72 x the radius: a chip deeper than that) has
                its edge AT the start - a notch at least that deep ('inner_limit'): it is
                judged as that, and never part of the fit (a row of such rays is an arc
                of a smaller circle: the curve would follow it).
                A ray with SATURATED flood within 3 px inside its edge is not measured (a
                clipped glint at the rim reads as a bulge): saturation can leave the
                outline unjudged, never make a finding. Pixels the scan MEASURED behind the
                face plane (`exclude`: the background or the part's own side wall, lit by
                the margin of the face flood) are dark to the outline: a lit surface behind
                the rim is not a bulge of the face
              * harmonic fit r(theta) = r0 + a1 cos + b1 sin + a2 cos 2theta + b2 sin 2theta:
                r0 = the mean radius, the 1st harmonic = a centre offset (the rays are
                re-cast from the corrected centre), the 2nd = an ellipse (a residual tilt
                of the face plane, or an oval part: reported, not judged). The fit is the
                curve the best 60 % of the rays lie on (least trimmed squares, started
                from the fits that leave one arc of the rim out, then refitted on the rays
                within 2.5 sigma of it): a notch or a bulge is NOT part of the curve it is
                measured against, however much of the rim it takes (up to 40 %) - an
                ordinary trimmed fit bends into a wide notch and reports half its depth
              * residual radius vs angle (mm): rms, max, and per angle
              * an outline only when the edges lie on ONE closed curve (robust scatter about
                it <= 0.3 mm / 1 px, 80 % of them within 3 x that): a flood that fills the
                image (texture, no dark surround) has "edges" everywhere - not measured,
                never a size
            'seen' says the flood shows an outline at all: at least 35 % of the rays have
            their edge on ONE circle (the best-fitting one, least trimmed squares at 40 %),
            whatever the rest of the rim does - half a face cut by the lit zone is seen, a
            textured flood whose edges are everywhere is not. An outline that is seen but
            not measured all the way round can never be passed (judge)
  judge     notches / bulges: contiguous arcs where |residual| > tolerance over at least
            min_arc_mm of rim (bench baseline on good parts at 0.36 mm / px: rms 0.06-0.09
            mm, max 0.20-0.28 mm -> tolerance 0.4 mm over >= 1.5 mm of arc); with a golden
            model the residual is taken against the good placements' per-angle median and
            the tolerance is at least k_sigma x their per-angle spread (capped at 2 x the
            tolerance: a golden never opens the tolerance further). An arc within
            guard_band_pct of the tolerance is MARGINAL: never passed (PULL). Rays with no
            edge inside an arc do not split it (the middle of a chip has none); a stretch
            of rim of min_arc_mm or more with no edge at all (saturated, or dark with
            nothing beside it out of tolerance) is UNJUDGED: status 'incomplete', never
            passed. An outline that is seen but not round-measurable is 'incomplete' too;
            'not_measured' is left for a flood that shows no outline at all
  centre    the centre the outline was looked for at (the located face's, a profile's)
            against the outline's own: more than CENTRE_TOL_FRAC x the face radius apart
            (half the margin between the inspected zone and the flat's edge) the inspected
            disc is off the flat - face_profile re-centres on the outline, a check that
            cannot is a PULL
  size      the outline diameter (2 r0, and the fitted ellipse's two axes: a lens zoomed
            after the calibration also tilts the triangulated plane, so the outline comes out
            small along one axis only) against the nominal one: more than tol_pct (4 %) off
            is a SIZE MISMATCH - "part reads N % small / large: calibration or lens suspect":
            a warning in every result and a PULL, never a silent pass, never a rescale

Plain numpy / OpenCV; the callers (verbs.py, the station, the recipe checks) say what the
plane, the centre and the nominal size are.
"""

from __future__ import annotations

import warnings
from typing import Callable, Optional

import numpy as np

N_ANGLES = 360
COARSE_SPAN = (0.15, 1.60)          # x the nominal radius: where the outline is looked for
FINE_SPAN = (0.72, 1.28)            # x the coarse radius: the per-ray search
INNER_END = 0.90                    # per ray: the face level is taken inside this x radius
LEVEL_PCT = 30.0                    # ... as this percentile (a glint must not raise it)
LEVEL_WIN_DEG = 45.0                # ... but a ray under LEVEL_FLOOR x the median contrast of
LEVEL_FLOOR = 0.5                   # the rays within this angle (itself >= LEVEL_FLOOR x the
                                    # whole rim's) takes theirs: the light that leaks into a
                                    # chip is not a face level
SAT_DIFF_DN = 240.0                 # without a saturation mask: lit - dark at / above this
SAT_NEAR_PX = 3.0                   # saturation this close inside a ray's edge: not measured
LTS_KEEP = 0.60                     # the outline's curve: the one the best 60 % of the rays
LTS_WINDOWS_DEG = (60.0, 120.0, 160.0)  # lie on; starts = the fits without one arc this wide,
LTS_STEP_DEG = 15.0                 # every this many degrees round the rim
LTS_REFIT_SIGMA = 2.5               # then refitted on the rays within this x its scatter
NO_EDGE_OK, NO_EDGE_SAT, NO_EDGE_DARK = 0, 1, 2     # per ray: an edge | saturated | none
STEP_PX = 0.25                      # radial sampling, camera px
SMOOTH_PX = 1.0                     # Gaussian on the flood before sampling
MIN_CONTRAST_DN = 8.0               # face level - dark level a ray needs
MIN_CONTRAST_FRAC = 0.12            # ... and this fraction of its face level
SIZE_MIN_VALID_PCT = 50.0           # the diameter needs this many rays with an edge
SIZE_MAX_GAP_DEG = 120.0            # ... and no larger gap between them
SIZE_MAX_SCATTER_MM = 0.3           # ... lying on ONE closed curve: their robust scatter about
SIZE_MAX_SCATTER_PX = 1.0           # it at most the larger of these (mm / camera px), and
SIZE_MIN_INLIER_PCT = 80.0          # this share of them within 3 x that of it (a textured
                                    # flood without a dark surround has edges everywhere)
SEEN_ON_CURVE_PCT = 35.0            # 'seen': at least this share of ALL the rays have their
SEEN_KEEP = 0.40                    # edge within that scatter of one circle (the curve the
                                    # best 40 % of the edges lie on)
MIN_RING_RAYS = 0.5                 # coarse profile: a ring counts with this share of its rays
                                    # in the image
ROUND_MIN_VALID_PCT = 90.0          # the roundness (residuals) needs these
ROUND_MAX_GAP_DEG = 20.0
SIZE_TOL_PCT = 4.0                  # known-size check: outline vs nominal diameter
RIM_TOL_MM = 0.4                    # roundness: |residual| beyond this ...
MIN_ARC_MM = 1.5                    # ... over at least this much rim is a notch / bulge
GUARD_BAND_PCT = 20.0               # an arc within this % below the tolerance: marginal
K_SIGMA = 4.0                       # golden: tolerance >= k x the per-angle spread
MAX_TOL_FACTOR = 2.0                # ... but never more than this x the tolerance
CENTRE_TOL_FRAC = 0.035             # x the face radius: the located / profile centre vs the
                                    # outline's own (0.34 mm on a 19.5 mm face: half of what
                                    # lies between the inspected zone, 0.926 R, and the edge)
SIZE_REASON = "size_mismatch"
RIM_ZONE = "rim_outline"


# ------------------------------------------------------------------ plane -> image maps

def ray_plane_map(K, dist, centre, e1, e2, origin=(0.0, 0.0)) -> Callable:
    """(u, v) mm in the face plane (about `centre`, along e1 / e2) -> image px (x, y) of an
    image whose (0, 0) is camera px `origin`, through the camera model."""
    from ...capture.mock_stereo import distort_project
    K = np.asarray(K, np.float64)
    c, a, b = (np.asarray(x, np.float64) for x in (centre, e1, e2))
    o = np.asarray(origin, np.float64)

    def to_px(u, v):
        u, v = np.asarray(u, np.float64), np.asarray(v, np.float64)
        X = c + u[..., None] * a + v[..., None] * b
        return distort_project(X, K, dist) - o
    return to_px


def homography_map(px: np.ndarray, uv: np.ndarray, max_points: int = 6000) -> Optional[Callable]:
    """(u, v) mm -> image px from measured points: px (N, 2) and their in-plane (u, v) (N, 2)
    (a plane seen by a pinhole camera is a homography; fitted by least squares, once more
    without the points off by more than 3 x 1.4826 MAD). None with fewer than 12 points."""
    import cv2
    px, uv = np.asarray(px, np.float64), np.asarray(uv, np.float64)
    ok = np.isfinite(px).all(1) & np.isfinite(uv).all(1)
    px, uv = px[ok], uv[ok]
    if len(px) < 12:
        return None
    if len(px) > max_points:
        k = np.random.default_rng(0).choice(len(px), max_points, replace=False)
        px, uv = px[k], uv[k]
    H = None
    keep = np.ones(len(px), bool)
    for _ in range(2):
        H, _ = cv2.findHomography(uv[keep], px[keep], 0)
        if H is None:
            return None
        q = np.c_[uv, np.ones(len(uv))] @ H.T
        e = np.hypot(q[:, 0] / q[:, 2] - px[:, 0], q[:, 1] / q[:, 2] - px[:, 1])
        mad = float(np.median(np.abs(e - np.median(e))))
        new = e <= np.median(e) + max(3 * 1.4826 * mad, 0.25)
        if new.sum() < 12 or np.array_equal(new, keep):
            break
        keep = new

    def to_px(u, v):
        u, v = np.asarray(u, np.float64), np.asarray(v, np.float64)
        w = H[2, 0] * u + H[2, 1] * v + H[2, 2]
        return np.stack([(H[0, 0] * u + H[0, 1] * v + H[0, 2]) / w,
                         (H[1, 0] * u + H[1, 1] * v + H[1, 2]) / w], -1)
    return to_px


# ------------------------------------------------------------------ sampling

def _smooth(img: np.ndarray, sigma: float) -> np.ndarray:
    import cv2
    a = np.asarray(img, np.float32)
    fin = np.isfinite(a)
    if not sigma or sigma <= 0:
        return np.where(fin, a, np.nan)
    num = cv2.GaussianBlur(np.where(fin, a, 0.0).astype(np.float32), (0, 0), sigma)
    den = cv2.GaussianBlur(fin.astype(np.float32), (0, 0), sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[~fin] = np.nan
    return out


def _sample(img: np.ndarray, to_px: Callable, c, theta: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """img at the in-plane points c + rho (cos theta, sin theta) -> (n_theta, n_rho), NaN
    off the image / where the image is NaN."""
    import cv2
    u = c[0] + np.outer(np.cos(theta), rho)
    v = c[1] + np.outer(np.sin(theta), rho)
    p = to_px(u, v).astype(np.float32)
    return cv2.remap(img, p[..., 0], p[..., 1], cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=float("nan"))


def _first_fall(V: np.ndarray, thr: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Per ray: the radius where V first falls below thr going outward (linear between the
    two samples), NaN when the ray does not start above it or never falls below it."""
    with np.errstate(invalid="ignore"):
        above = V >= thr[:, None]
        cross = above[:, :-1] & np.isfinite(V[:, 1:]) & (V[:, 1:] < thr[:, None])
        # everything before the crossing must be measured and bright: the FIRST fall
        lead = np.cumsum(~above[:, :-1], axis=1) == 0
        cross &= lead
    has = cross.any(1)
    i = np.argmax(cross, axis=1)
    n = np.arange(len(V))
    a, b = V[n, i], V[n, np.minimum(i + 1, V.shape[1] - 1)]
    with np.errstate(invalid="ignore", divide="ignore"):
        f = np.clip((a - thr) / (a - b), 0.0, 1.0)
    step = float(rho[1] - rho[0])
    return np.where(has, rho[i] + f * step, np.nan)


def _gap_deg(valid: np.ndarray) -> float:
    """The largest run of rays without an edge, in degrees (circular)."""
    n = len(valid)
    if valid.all():
        return 0.0
    if not valid.any():
        return 360.0
    k = int(np.argmax(valid))                       # start at a valid ray
    v = np.roll(valid, -k)
    best = run = 0
    for x in v:
        run = 0 if x else run + 1
        best = max(best, run)
    return 360.0 * best / n


def _harmonics(theta: np.ndarray, rho: np.ndarray, order: int, iters: int = 4) -> tuple:
    """Robust least squares of rho(theta) on harmonics 0..order -> (coefficients [c0, a1,
    b1, a2, b2 ...], residual (n,) NaN where rho is NaN, the rays kept)."""
    ok = np.isfinite(rho)
    cols = [np.ones_like(theta)]
    for k in range(1, order + 1):
        cols += [np.cos(k * theta), np.sin(k * theta)]
    A = np.stack(cols, 1)
    keep = ok.copy()
    co = np.zeros(A.shape[1])
    for _ in range(max(iters, 1)):
        if keep.sum() < A.shape[1] + 2:
            break
        co = np.linalg.lstsq(A[keep], rho[keep], rcond=None)[0]
        res = rho - A @ co
        r = res[keep]
        mad = float(np.median(np.abs(r - np.median(r))))
        new = ok & (np.abs(res - np.median(r)) <= max(3 * 1.4826 * mad, 0.05))
        if new.sum() < A.shape[1] + 2 or np.array_equal(new, keep):
            break
        keep = new
    return co, np.where(ok, rho - A @ co, np.nan), keep


def _circ_median(a: np.ndarray, half: int) -> np.ndarray:
    """Circular running median over +-half samples, NaN ignored (NaN where a window is
    all NaN)."""
    from numpy.lib.stride_tricks import sliding_window_view
    half = max(0, min(int(half), (len(a) - 1) // 2))
    w = sliding_window_view(np.r_[a[len(a) - half:], a, a[:half]], 2 * half + 1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(w, axis=1)


def _robust_harmonics(theta: np.ndarray, rho: np.ndarray, order: int,
                      keep_frac: float = LTS_KEEP) -> tuple:
    """rho(theta) on harmonics 0..order by LEAST TRIMMED SQUARES: the coefficients whose
    best keep_frac of the rays have the smallest squared residuals, so up to 1 - keep_frac
    of the rim may be anything (a notch, a bulge, a chord) without moving the curve. Starts:
    the plain fit and the fits that leave one arc of the rim out (LTS_WINDOWS_DEG wide,
    every LTS_STEP_DEG: a defect is an arc), each concentrated (refit on its best rays until
    they stop changing); then one refit on the rays within LTS_REFIT_SIGMA x the robust
    scatter of the winner. -> (coefficients [c0, a1, b1, a2, b2 ...], residual (n,) NaN
    where rho is NaN, the rays the final fit used, the robust scatter)."""
    ok = np.isfinite(rho)
    cols = [np.ones_like(theta)]
    for k in range(1, order + 1):
        cols += [np.cos(k * theta), np.sin(k * theta)]
    A = np.stack(cols, 1)
    p = A.shape[1]
    n_ok = int(ok.sum())
    co = np.zeros(p)
    if n_ok < p + 2:
        return co, np.where(ok, rho, np.nan), ok.copy(), float("nan")
    h = max(p + 2, int(np.ceil(float(keep_frac) * n_ok)))
    y = np.where(ok, rho, 0.0)
    deg = np.degrees(theta) % 360.0
    starts = [ok]
    for wdeg in LTS_WINDOWS_DEG:
        for c0 in np.arange(0.0, 360.0, LTS_STEP_DEG):
            out = np.abs((deg - c0 + 180.0) % 360.0 - 180.0) <= wdeg / 2.0
            s = ok & ~out
            if s.sum() >= p + 2 and (ok & out).any():
                starts.append(s)
    def fit(keep):                                  # normal equations: p is 3 or 5
        Ak = A[keep]
        try:
            return np.linalg.solve(Ak.T @ Ak, Ak.T @ y[keep])
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(Ak, y[keep], rcond=None)[0]

    best = None
    for keep in starts:
        c = None
        for _ in range(8):
            c = fit(keep)
            r = np.where(ok, np.abs(y - A @ c), np.inf)
            new = np.zeros(len(y), bool)
            new[np.argpartition(r, h - 1)[:h]] = True
            if np.array_equal(new, keep):
                break
            keep = new
        r = np.where(ok, np.abs(y - A @ c), np.inf)
        obj = float(np.sum(np.partition(r, h - 1)[:h] ** 2))
        if best is None or obj < best[0] - 1e-12:
            best = (obj, c)
    co = best[1]
    r = y - A @ co
    sig = 1.4826 * float(np.median(np.abs(r[ok])))
    keep = ok & (np.abs(r) <= max(LTS_REFIT_SIGMA * sig, 0.05))
    if keep.sum() >= max(p + 2, h // 2):
        co = fit(keep)
    else:
        keep = ok.copy()
    return co, np.where(ok, rho - A @ co, np.nan), keep, sig


def _harmonic_values(theta: np.ndarray, co: np.ndarray) -> np.ndarray:
    """The curve of _robust_harmonics' coefficients [c0, a1, b1, a2, b2 ...] at theta."""
    v = np.full(theta.shape, float(co[0]))
    for k in range(1, (len(co) - 1) // 2 + 1):
        v = v + co[2 * k - 1] * np.cos(k * theta) + co[2 * k] * np.sin(k * theta)
    return v


def _seen(theta: np.ndarray, edge: np.ndarray, lim: float) -> bool:
    """Do the edges show an outline at all: SEEN_ON_CURVE_PCT of all the rays within lim of
    one circle (centre free: harmonics 0 + 1, fitted to the best SEEN_KEEP of the edges)?"""
    valid = np.isfinite(edge)
    if 100.0 * float(valid.mean()) < SEEN_ON_CURVE_PCT:
        return False
    res = _robust_harmonics(theta, edge, 1, keep_frac=SEEN_KEEP)[1]
    with np.errstate(invalid="ignore"):
        on = np.abs(res) <= lim
    return bool(100.0 * float(on.mean()) >= SEEN_ON_CURVE_PCT)


def measure(img: np.ndarray, to_px: Callable, radius_mm: float, n_angles: int = N_ANGLES,
            smooth_px: float = SMOOTH_PX, centre_passes: int = 3,
            sat: Optional[np.ndarray] = None, exclude: Optional[np.ndarray] = None) -> dict:
    """The rim outline of the bright disc around the in-plane origin of to_px in the flood
    image img (lit - dark, camera DN; NaN = unknown), about radius_mm in radius. sat: the
    flood's saturated pixels (bool, img's shape; None = img at / above SAT_DIFF_DN).
    exclude: pixels known NOT to be the face (bool, img's shape: measured behind its plane):
    they count as dark.
    -> {seen (a bright disc on a dark surround was found), ok (the diameter was measured),
    round_ok (the residuals too), why, diameter_mm, radius_mm (r0), centre_offset_mm [u, v],
    theta_deg (n,), rho_mm (n,) (NaN = no edge on that ray), residual_mm (n,), no_edge (n,)
    (NO_EDGE_OK | NO_EDGE_SAT | NO_EDGE_DARK), inner_limit (n,) bool (the ray is dark where
    the search starts: its edge is AT the start, a notch at least that deep), rms_mm,
    max_abs_mm, valid_pct, max_gap_deg, ellipse {...}, level_dn, contrast_dn, saturated_pct
    (rays not measured: saturated), inner_limit_pct, mm_per_px, scatter_mm / inlier_pct (how
    well the edges lie on one closed curve)}."""
    R = float(radius_mm)
    n = int(n_angles)
    if n < 180:
        raise ValueError(f"the rim outline needs at least 180 rays, got {n_angles!r}")
    out = {"seen": False, "ok": False, "round_ok": False, "why": "", "nominal_radius_mm": R,
           "n_angles": n, "diameter_mm": None, "radius_mm": None, "centre_offset_mm": None,
           "theta_deg": np.degrees(np.arange(n) * 2 * np.pi / n),
           "rho_mm": np.full(n, np.nan), "residual_mm": np.full(n, np.nan),
           "no_edge": np.full(n, NO_EDGE_DARK, np.int8), "inner_limit": np.zeros(n, bool),
           "rms_mm": None, "max_abs_mm": None, "valid_pct": 0.0, "max_gap_deg": 360.0,
           "ellipse": None, "level_dn": None, "contrast_dn": None, "saturated_pct": None,
           "inner_limit_pct": None, "mm_per_px": None}
    if img is None or not R > 0:
        out["why"] = "no flood image" if img is None else "no nominal radius"
        return out
    theta = np.arange(n) * 2 * np.pi / n
    p0 = to_px(np.zeros(1), np.zeros(1))[0]
    pu = to_px(np.array([1.0]), np.zeros(1))[0]
    pv = to_px(np.zeros(1), np.array([1.0]))[0]
    ppm = 0.5 * (float(np.hypot(*(pu - p0))) + float(np.hypot(*(pv - p0))))    # px per mm
    if not np.isfinite(ppm) or ppm <= 0:
        out["why"] = "the face plane does not map into the image"
        return out
    out["mm_per_px"] = round(1.0 / ppm, 5)
    step = STEP_PX / ppm
    if exclude is not None and np.asarray(exclude).any():
        a = np.array(img, np.float32, copy=True)
        fin = np.isfinite(a)
        # ... as dark as the darkest the image shows (not a low percentile: a long exposure
        # can leave most of the image a saturated surround)
        dark = float(np.percentile(a[fin], 0.1)) if fin.any() else 0.0
        a[np.asarray(exclude, bool) & fin] = dark
        img = a
    else:
        exclude = None
    g = _smooth(img, smooth_px)
    with np.errstate(invalid="ignore"):
        satb = (np.asarray(sat, bool) if sat is not None
                else np.nan_to_num(np.asarray(img, np.float32)) >= SAT_DIFF_DN)
    if exclude is not None:             # a saturated surround behind the plane is not a glint
        satb = satb & ~np.asarray(exclude, bool)
    satf = satb.astype(np.float32)
    c = np.zeros(2)
    # coarse: the angle-median radial profile
    rho = np.arange(COARSE_SPAN[0] * R, COARSE_SPAN[1] * R, step)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        V = _sample(g, to_px, c, theta, rho)
        prof = np.nanmedian(V, 0)
        # a ring mostly off the image is unknown, not dark (the median of its few rays left
        # is noise: a flood that fills the image would get a 'dark surround' from it)
        prof[np.isfinite(V).mean(0) < MIN_RING_RAYS] = np.nan
        core = prof[rho <= 0.5 * R]
        if not np.isfinite(core).any():
            out["why"] = "the flood does not cover the face centre"
            return out
        L0 = float(np.nanmedian(core))
        D0 = float(np.nanmin(prof[rho > 0.5 * R])) if np.isfinite(prof[rho > 0.5 * R]).any() \
            else float("nan")
    if not np.isfinite(D0) or L0 - D0 < max(MIN_CONTRAST_DN, MIN_CONTRAST_FRAC * L0):
        out["why"] = (f"no bright disc with a dark surround in the flood (face level {L0:.0f} "
                      f"DN, darkest ring {D0:.0f} DN)")
        return out
    thr0 = D0 + 0.5 * (L0 - D0)
    rc = _first_fall(prof[None], np.array([thr0]), rho)[0]
    e0 = _first_fall(V, np.full(n, thr0), rho)          # where the disc is: its own centre
    v0 = np.isfinite(e0)
    if v0.mean() >= 0.5 and _gap_deg(v0) <= SIZE_MAX_GAP_DEG:
        # (the circle half of these falls lie on: a face cut by the lit zone, or one with a
        # large chip, must not pull the centre into what is left of it)
        co0 = _robust_harmonics(theta, e0, 1, keep_frac=0.5)[0]
        if np.isfinite(co0).all() and COARSE_SPAN[0] * R <= co0[0] <= COARSE_SPAN[1] * R \
                and float(np.hypot(co0[1], co0[2])) <= R:
            c, rc = np.array(co0[1:3], np.float64), float(co0[0])
    if not np.isfinite(rc):
        out["why"] = "the flood's radial profile has no edge near the face radius"
        return out
    co = res = keep = None
    r_fit = float(rc)
    half = int(round(LEVEL_WIN_DEG * n / 360.0))
    for _ in range(max(int(centre_passes), 1)):
        rho = np.arange(FINE_SPAN[0] * r_fit, FINE_SPAN[1] * r_fit, step)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            V = _sample(g, to_px, c, theta, rho)
            inner = rho <= INNER_END * r_fit
            Lr = np.nanpercentile(V[:, inner], LEVEL_PCT, axis=1)
            fin_in = np.isfinite(V[:, inner]).mean(1) >= 0.5
            Dr = np.nanmin(np.where(np.isfinite(V[:, ~inner]), V[:, ~inner], np.inf), 1)
            Dr = np.where(np.isfinite(Dr), Dr, np.nan)
            con = Lr - Dr
            # the face level of a ray: its own, never under half its neighbours' (inside a
            # chip a ray's own level is the light that leaks in: not a face level)
            con_all = float(np.nanmedian(con)) if np.isfinite(con).any() else float("nan")
            ref = np.fmax(_circ_median(con, half), LEVEL_FLOOR * con_all)
        # (a ray under half its neighbours' level shows no face of its own: theirs is used)
        eff = np.where(con >= LEVEL_FLOOR * ref, con, ref)
        good = fin_in & np.isfinite(eff) & np.isfinite(Dr) & (
            eff >= np.maximum(MIN_CONTRAST_DN, MIN_CONTRAST_FRAC * (Dr + eff)))
        thr = Dr + 0.5 * eff
        edge = np.where(good, _first_fall(V, thr, rho), np.nan)
        # dark where the search starts (a chip deeper than the search span): the edge is at
        # the start - a notch at least that deep, never 'no finding'
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            v0 = np.nanmedian(V[:, :max(3, int(round(1.0 / STEP_PX)))], axis=1)
        deep = good & ~np.isfinite(edge) & np.isfinite(v0) & (v0 < thr)
        # saturated flood just inside (or at) a ray's edge: not measured
        Sr = np.nan_to_num(_sample(satf, to_px, c, theta, rho))
        e_at = np.where(np.isfinite(edge), edge, np.where(deep, rho[0], rho[-1]))[:, None]
        near = (rho[None, :] <= e_at + 2 * step) & (rho[None, :] >= e_at - SAT_NEAR_PX / ppm)
        hot = (np.where(near, Sr, 0.0) > 0.25).any(1)
        n_hot = int((hot & (np.isfinite(edge) | deep)).sum())
        edge = np.where(hot, np.nan, edge)            # the edges FOUND: what the curve is
        deep &= ~hot                                  # fitted to; `deep` rays are judged
        found = np.isfinite(edge)                     # against it, never part of it
        valid = found | deep
        vp, gap = 100.0 * float(valid.mean()), _gap_deg(valid)
        fp, fgap = 100.0 * float(found.mean()), _gap_deg(found)
        out.update(valid_pct=round(vp, 2), max_gap_deg=round(gap, 1),
                   rho_mm=np.where(deep, rho[0], edge), inner_limit=deep,
                   no_edge=np.where(valid, NO_EDGE_OK,
                                    np.where(hot, NO_EDGE_SAT, NO_EDGE_DARK)).astype(np.int8))
        if fp < SIZE_MIN_VALID_PCT or fgap > SIZE_MAX_GAP_DEG:
            out["why"] = (f"the outline shows on {fp:.0f} % of the rim only (largest gap "
                          f"{fgap:.0f} deg): shadowed, cut by the lit zone, or no dark surround")
            out["seen"] = _seen(theta, edge, max(SIZE_MAX_SCATTER_MM,
                                                 SIZE_MAX_SCATTER_PX / ppm))
            return out
        round_ok = vp >= ROUND_MIN_VALID_PCT and gap <= ROUND_MAX_GAP_DEG
        co, res, keep, sig = _robust_harmonics(theta, edge, 2 if round_ok else 1)
        # a ray dark where the search starts: its edge is at most there
        res = np.where(deep, rho[0] - _harmonic_values(theta, co), res)
        c = c + co[1:3]                               # the 1st harmonic: a centre offset
        r_fit = float(co[0])
    # one closed outline? The edges of a face lie on one smooth curve (a notch, a chord where
    # the lit zone cuts it: some rays off); edges scattered over the search span are a
    # textured flood with no dark surround - not a size, not a roundness
    r2 = _robust_harmonics(theta, edge, 2)[1][found]
    sig = 1.4826 * float(np.median(np.abs(r2)))
    lim = max(SIZE_MAX_SCATTER_MM, SIZE_MAX_SCATTER_PX / ppm)
    inl = 100.0 * float(np.sum(np.abs(r2) <= 3 * lim)) / max(int(valid.sum()), 1)
    out.update(scatter_mm=round(sig, 4), inlier_pct=round(inl, 1),
               seen=bool(sig <= lim) or _seen(theta, edge, lim))
    if sig > lim or inl < SIZE_MIN_INLIER_PCT:
        out["why"] = (f"the flood shows no closed outline: its edges scatter by {sig:.2f} mm "
                      f"({inl:.0f} % of them within {3 * lim:.2f} mm of one curve; needs "
                      f"{lim:.2f} mm, {SIZE_MIN_INLIER_PCT:g} %) - a large part of the rim is "
                      "off the curve (a large edge defect), no dark surround around the "
                      "face, or it is cut by the lit zone")
        return out
    out.update({"ok": True, "round_ok": bool(round_ok), "diameter_mm": 2.0 * float(co[0]),
                "radius_mm": float(co[0]),
                "centre_offset_mm": [float(c[0]), float(c[1])],
                "residual_mm": res,
                "level_dn": float(np.nanmedian(Lr[found])),
                "contrast_dn": float(np.nanmedian(con[found])),
                "saturated_pct": round(100.0 * n_hot / n, 2),
                "inner_limit_pct": round(100.0 * float(deep.mean()), 2),
                "fit_rays": int(keep.sum())})
    r = res[valid]
    out["rms_mm"] = float(np.sqrt(np.mean(r ** 2)))
    out["max_abs_mm"] = float(np.max(np.abs(r)))
    if round_ok:
        e = float(np.hypot(co[3], co[4]))
        major, minor = float(co[0] + e), float(co[0] - e)
        ratio = minor / major if major > 0 else 1.0
        out["ellipse"] = {"major_mm": 2 * major, "minor_mm": 2 * minor,
                          "angle_deg": float(np.degrees(0.5 * np.arctan2(co[4], co[3]))),
                          "axis_ratio": ratio,
                          "tilt_equiv_deg": float(np.degrees(np.arccos(min(max(ratio, -1.0), 1.0)))),
                          "note": "the 2nd harmonic of the outline: a residual tilt of the "
                                  "face plane or an oval part (its axes are held to the "
                                  "known size; the roundness residual is taken about it)"}
    else:
        out["why"] = (f"roundness not judged: an edge on {vp:.0f} % of the rays, largest gap "
                      f"{gap:.0f} deg (needs {ROUND_MIN_VALID_PCT:g} %, {ROUND_MAX_GAP_DEG:g} deg)")
    return out


# ------------------------------------------------------------------ size

def mismatch_text(dev_pct: float) -> str:
    return (f"part reads {abs(dev_pct):.1f} % {'small' if dev_pct < 0 else 'large'}: "
            "calibration or lens suspect - run rig-check / recalibrate")


def outline_size(o: Optional[dict], nominal_mm: Optional[float],
                 tol_pct: float = SIZE_TOL_PCT, what: str = "outline", source: str = "",
                 why: str = "") -> dict:
    """size_check of a measured outline (measure()'s result; None / not measured: not
    checked, with why): its mean diameter and, when the whole outline was measured, the two
    axes of its fitted ellipse."""
    if o is None or not o.get("ok"):
        return size_check(None, nominal_mm, tol_pct, what, source,
                          why or (o or {}).get("why", ""))
    e = o.get("ellipse")
    return size_check(o["diameter_mm"], nominal_mm, tol_pct, what, source,
                      axes=(e["major_mm"], e["minor_mm"]) if e else None)


def size_check(measured_mm: Optional[float], nominal_mm: Optional[float],
               tol_pct: float = SIZE_TOL_PCT, what: str = "outline", source: str = "",
               why: str = "", axes: Optional[tuple] = None,
               axis_tol_factor: float = 1.0) -> dict:
    """The known-size sanity check: a measured outline diameter against its nominal one.
    axes: (major, minor) of the outline's fitted ellipse, mm - each is held to the nominal
    too (within axis_tol_factor x tol_pct: a caller whose plane is fitted on few points gives
    an axis alone more room, a small tilt of that plane moves it), and the size furthest
    outside its tolerance is the deviation reported.
    -> {status 'ok' | 'mismatch' | 'not_checked', nominal_mm, measured_mm, axes_mm,
    deviation_pct, tol_pct, message, reason}. 'mismatch' (more than tol_pct off) is a PULL
    with reason size_mismatch: the measurement is never rescaled and never passed."""
    out = {"status": "not_checked", "nominal_mm": None if nominal_mm is None else float(nominal_mm),
           "measured_mm": None if measured_mm is None else round(float(measured_mm), 4),
           "axes_mm": None if axes is None else [round(float(a), 4) for a in axes],
           "deviation_pct": None, "tol_pct": float(tol_pct) if tol_pct is not None else None,
           "source": source, "message": "", "reason": None}
    if tol_pct is None or not float(tol_pct) > 0:
        out["message"] = "size check off (tolerance 0)"
        return out
    if nominal_mm is None or not float(nominal_mm) > 0:
        out["message"] = "size not checked: no nominal size is known"
        return out
    if measured_mm is None or not np.isfinite(measured_mm):
        out["message"] = ("size not checked: the " + what + " could not be measured"
                          + (f" ({why})" if why else ""))
        return out
    sizes = [float(measured_mm)] + ([float(a) for a in axes if np.isfinite(a)] if axes else [])
    devs = [100.0 * (v - float(nominal_mm)) / float(nominal_mm) for v in sizes]
    atol = float(tol_pct) * max(float(axis_tol_factor), 1.0)
    lims = [float(tol_pct)] + [atol] * (len(devs) - 1)
    # the size furthest outside (or nearest to) its tolerance
    dev, lim = max(zip(devs, lims), key=lambda t: abs(t[0]) / t[1])
    across = (f"{float(measured_mm):.2f} mm across" if not axes else
              f"{float(axes[1]):.2f} x {float(axes[0]):.2f} mm across (mean "
              f"{float(measured_mm):.2f})")
    tol_text = f"tolerance {float(tol_pct):g} %" + (
        f" (an axis alone {atol:g} %)" if len(devs) > 1 and atol != float(tol_pct) else "")
    out["deviation_pct"] = round(dev, 3)
    out["mean_deviation_pct"] = round(devs[0], 3)
    if abs(dev) > lim:
        out.update(status="mismatch", reason=SIZE_REASON,
                   message=(f"{mismatch_text(dev)} (the {what} is {across}, nominal "
                            f"{float(nominal_mm):g} mm, {tol_text}; nothing "
                            "is rescaled - or the nominal is not the diameter of what the "
                            "outline shows)"))
    else:
        out.update(status="ok", message=(f"size ok: the {what} is {across}, {dev:+.1f} % of "
                                         f"the nominal {float(nominal_mm):g} mm"))
    return out


def pull_reason(sc: dict, where: str = "") -> dict:
    """The pull reason {reason, message} of a size check that is a mismatch."""
    return {"reason": SIZE_REASON, "message": (f"{where}: " if where else "") + sc["message"]}


# ------------------------------------------------------------------ centre

def centre_check(o: Optional[dict], radius_mm: float, what: str = "located centre") -> dict:
    """The centre an outline was looked for at against the outline's own centre (measure()'s
    centre_offset_mm; a whole outline only: round_ok). -> {status 'ok' | 'off' |
    'not_checked', offset_mm, offset_uv_mm, tol_mm, message}. 'off': the two are more than
    CENTRE_TOL_FRAC x the face radius apart - a disc inspected about that centre reaches
    the rim on one side."""
    tol = CENTRE_TOL_FRAC * float(radius_mm)
    out = {"status": "not_checked", "offset_mm": None, "offset_uv_mm": None,
           "tol_mm": round(tol, 4), "message": ""}
    if o is None or not o.get("ok") or not o.get("round_ok") or o.get("centre_offset_mm") is None:
        out["message"] = ("centre not checked: no whole flood outline"
                          + (f" ({o.get('why')})" if o is not None and o.get("why") else ""))
        return out
    du, dv = (float(x) for x in o["centre_offset_mm"])
    d = float(np.hypot(du, dv))
    out.update(offset_mm=round(d, 4), offset_uv_mm=[round(du, 4), round(dv, 4)])
    if d > tol:
        out.update(status="off", message=(
            f"the {what} is {d:.2f} mm from the centre of the face's flood outline "
            f"(tolerance {tol:.2f} mm): the outline of the bright flat is not centred where "
            "the face was located"))
    else:
        out.update(status="ok", message=f"centre ok: the {what} is {d:.2f} mm from the flood "
                                        "outline's centre")
    return out


# ------------------------------------------------------------------ roundness

def _runs(flag: np.ndarray, sign: np.ndarray, bridge: int = 1,
          soft: Optional[np.ndarray] = None) -> list:
    """Circular runs of flag with one sign, gaps of up to `bridge` rays bridged -> [index
    arrays]. soft: rays that neither belong to a run nor end one (no edge was found there:
    the middle of a chip): a run goes on across them, and they are part of its arc when
    flagged rays lie on both sides. A run is the rays from its first to its last flagged
    one."""
    n = len(flag)
    soft = np.zeros(n, bool) if soft is None else np.asarray(soft, bool)
    out = []
    for s in (1, -1):
        f = flag & (sign == s)
        if not f.any():
            continue
        hard_gap = ~f & ~soft
        if not hard_gap.any():                        # flagged (or no edge) all the way round
            out.append(np.arange(n))
            continue
        k = int(np.argmax(hard_gap))                  # start at a ray outside every run
        idx = (np.arange(n) + k) % n
        cur, last, miss = [], 0, 0

        def close():
            if cur:
                out.append(np.array(cur[:last]))      # up to its last flagged ray

        for i in idx:
            if f[i]:
                cur.append(i)
                last, miss = len(cur), 0
            elif not cur:
                continue
            elif soft[i]:
                cur.append(i)
            else:
                miss += 1
                if miss > bridge:
                    close()
                    cur, last, miss = [], 0, 0
                else:
                    cur.append(i)
        close()
    return out


def _gaps(missing: np.ndarray) -> list:
    """Circular runs of True -> [index arrays]."""
    n = len(missing)
    if not missing.any():
        return []
    if missing.all():
        return [np.arange(n)]
    k = int(np.argmin(missing))
    idx = (np.arange(n) + k) % n
    out, cur = [], []
    for i in idx:
        if missing[i]:
            cur.append(i)
        elif cur:
            out.append(np.array(cur))
            cur = []
    if cur:
        out.append(np.array(cur))
    return out


def _sustained(res: np.ndarray, n_win: int) -> float:
    """The largest deviation (mm, >= 0) that n_win consecutive rays (circular) all exceed
    with one sign: the notch / bulge depth that holds over the minimum arc."""
    from numpy.lib.stride_tricks import sliding_window_view
    n_win = max(1, min(int(n_win), len(res)))
    best = 0.0
    for s in (1.0, -1.0):
        x = np.where(np.isfinite(res), s * res, -np.inf)
        m = sliding_window_view(np.r_[x, x[:n_win - 1]], n_win).min(1)
        best = max(best, float(m.max()))
    return best if np.isfinite(best) else 0.0


def golden_model(outlines: list) -> Optional[dict]:
    """The good placements' outlines (measure() results, round_ok) -> the per-angle model:
    median residual (the part of the outline's shape that repeats: lighting, camera angle)
    and spread (sample std; 1.4826 MAD from 4 placements on), the diameters. None with
    fewer than 2 usable outlines (or different ray counts)."""
    good = [o for o in outlines if o is not None and o.get("round_ok")]
    if len(good) < 2 or len({len(o["residual_mm"]) for o in good}) != 1:
        return None
    Rm = np.stack([np.asarray(o["residual_mm"], np.float64) for o in good])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(Rm, 0)
        if len(good) >= 4:
            spr = 1.4826 * np.nanmedian(np.abs(Rm - med), 0)
        else:
            spr = np.nanstd(Rm, 0, ddof=1)
    med = np.where(np.isfinite(med), med, 0.0)
    fin = np.isfinite(spr)
    spr = np.where(fin, spr, np.nanmax(spr[fin]) if fin.any() else 0.0)
    k = 5                                           # +-5 rays: few placements per angle
    pad = np.r_[spr[-k:], spr, spr[:k]]
    spr = np.array([np.median(pad[i:i + 2 * k + 1]) for i in range(len(spr))])
    d = np.array([o["diameter_mm"] for o in good], float)
    return {"theta_deg": np.asarray(good[0]["theta_deg"], np.float64), "median_mm": med,
            "spread_mm": spr, "n_placements": len(good),
            "diameter_mm": float(np.median(d)), "diameter_spread_mm": float(np.ptp(d)),
            "rms_mm": [round(float(o["rms_mm"]), 4) for o in good],
            "max_abs_mm": [round(float(o["max_abs_mm"]), 4) for o in good]}


def judge(o: dict, tol_mm: float = RIM_TOL_MM, min_arc_mm: float = MIN_ARC_MM,
          guard_band_pct: float = GUARD_BAND_PCT, model: Optional[dict] = None,
          k_sigma: float = K_SIGMA) -> dict:
    """Notches and bulges of a measured outline. -> {status 'ok' | 'fail' | 'marginal' |
    'incomplete' | 'not_measured', findings [{kind 'notch' | 'bulge', theta_deg, arc_mm,
    depth_mm (signed: - = the outline is inside, a notch), u_mm, v_mm, r_mm, tol_mm,
    at_least (the depth is a lower bound: the ray was dark where the search starts)}],
    marginal [same], unjudged [{theta_deg, arc_mm, why}] (rim of min_arc_mm or more with no
    edge), rms_mm, max_abs_mm (against the golden's median when there is a model),
    arc_peak_mm (the largest deviation held over min_arc_mm of rim: what is compared with
    the tolerance), tol_mm, min_arc_mm, guard_band_pct, golden (a model was used), why}.
    Only 'ok' may be passed: 'incomplete' = the flood shows an outline (o['seen']) but it
    could not be judged all the way round (not measured round, or a stretch of min_arc_mm
    or more without an edge: a chip deep enough leaves none); 'not_measured' = the flood
    shows no outline at all (no flood, no bright disc on a dark surround)."""
    out = {"status": "not_measured", "findings": [], "marginal": [], "unjudged": [],
           "rms_mm": None, "max_abs_mm": None, "arc_peak_mm": None,
           "tol_mm": float(tol_mm), "min_arc_mm": float(min_arc_mm),
           "guard_band_pct": float(guard_band_pct), "golden": False,
           "why": "" if o is None else o.get("why", "")}
    if o is None or not o.get("round_ok"):
        if o is None:
            out["why"] = "no outline"
        elif o.get("seen"):
            out["status"] = "incomplete"
        return out
    res = np.asarray(o["residual_mm"], np.float64)
    theta = np.asarray(o["theta_deg"], np.float64)
    n = len(res)
    tol = np.full(n, float(tol_mm))
    if model is not None and len(model["median_mm"]) == n:
        res = res - np.asarray(model["median_mm"], np.float64)
        tol = np.clip(np.maximum(tol, float(k_sigma) * np.asarray(model["spread_mm"], np.float64)),
                      float(tol_mm), MAX_TOL_FACTOR * float(tol_mm))
        out["golden"] = True
        out["tol_max_mm"] = round(float(tol.max()), 4)
    fin = np.isfinite(res)
    r0 = float(o["radius_mm"])
    ray_mm = 2 * np.pi * r0 / n
    sign = np.where(np.nan_to_num(res) >= 0, 1, -1)
    out["rms_mm"] = float(np.sqrt(np.mean(res[fin] ** 2))) if fin.any() else None
    out["max_abs_mm"] = float(np.max(np.abs(res[fin]))) if fin.any() else None
    out["arc_peak_mm"] = _sustained(res, int(np.ceil(float(min_arc_mm) / ray_mm)))
    no_edge = np.asarray(o.get("no_edge", np.where(fin, NO_EDGE_OK, NO_EDGE_DARK)))
    deep = np.asarray(o.get("inner_limit", np.zeros(n, bool)), bool)
    # no edge and not saturated: such a ray may lie inside a notch (it does not split one);
    # a saturated ray is never part of a finding
    dark = ~fin & (no_edge != NO_EDGE_SAT)

    def arcs(limit, across=dark):
        found = []
        for idx in _runs(fin & (np.abs(np.nan_to_num(res)) > limit), sign, soft=across):
            arc = len(idx) * ray_mm
            if arc < float(min_arc_mm):
                continue
            a = np.abs(np.nan_to_num(res[idx]))
            k = idx[int(np.argmax(a))]              # its deepest ray: the depth reported
            flat = idx[a >= 0.9 * a.max()]          # ... at the middle of its floor
            t = float(theta[flat[len(flat) // 2]])
            found.append({"kind": "notch" if res[k] < 0 else "bulge", "theta_deg": round(t, 1),
                          "arc_mm": round(float(arc), 2), "depth_mm": round(float(res[k]), 3),
                          "at_least": bool(deep[idx].any()),
                          "r_mm": round(r0, 3),
                          "u_mm": round(r0 * float(np.cos(np.radians(t))), 3),
                          "v_mm": round(r0 * float(np.sin(np.radians(t))), 3),
                          "tol_mm": round(float(tol[k]), 4), "_rays": idx})
        found.sort(key=lambda f: -abs(f["depth_mm"]))
        return found

    hard = arcs(tol)
    taken = np.zeros(n, bool)
    for f in hard:
        taken[f["_rays"]] = True
    soft = [f for f in arcs(tol * (1.0 - float(guard_band_pct) / 100.0))
            if not taken[f["_rays"]].any()] if guard_band_pct else []
    for f in soft:
        taken[f["_rays"]] = True
    # out of tolerance on BOTH sides of a saturated stretch: not a finding (a glint is never
    # part of one), but what is under the glint is not judged either
    low = tol * (1.0 - float(guard_band_pct) / 100.0)
    for f in arcs(low, across=~fin):
        if not taken[f["_rays"]].any():
            taken[f["_rays"]] = True
            out["unjudged"].append({"theta_deg": f["theta_deg"], "arc_mm": f["arc_mm"],
                                    "why": f"a {f['kind']} of {abs(f['depth_mm']):.2f} mm on "
                                           "both sides of saturated flood"})
    for f in hard + soft:
        f.pop("_rays")
    # rim with no edge that no finding covers: min_arc_mm of it or more is not judged
    for idx in _gaps(~fin & ~taken):
        arc = len(idx) * ray_mm
        if arc >= float(min_arc_mm):
            mid = idx[len(idx) // 2]
            out["unjudged"].append({
                "theta_deg": round(float(theta[mid]), 1), "arc_mm": round(float(arc), 2),
                "why": ("saturated flood" if (no_edge[idx] == NO_EDGE_SAT).mean() >= 0.5
                        else "no edge (dark, or no contrast)")})
    out["findings"], out["marginal"] = hard, soft
    if hard:
        out["status"] = "fail"
    elif out["unjudged"]:
        u0 = out["unjudged"][0]
        out["status"] = "incomplete"
        out["why"] = (f"{len(out['unjudged'])} stretch(es) of rim with no edge, e.g. "
                      f"{u0['arc_mm']:g} mm at {u0['theta_deg']:g} deg ({u0['why']}): not "
                      "judged there")
    else:
        out["status"] = "marginal" if soft else "ok"
    return out


def public(o: Optional[dict], j: Optional[dict] = None, arrays: bool = False) -> Optional[dict]:
    """A measured outline (+ its judgement) as stored in a result: numbers, no arrays unless
    asked."""
    if o is None:
        return None
    r4 = (lambda v: None if v is None else round(float(v), 4))            # noqa: E731
    d = {"seen": bool(o.get("seen", o["ok"])), "measured": bool(o["ok"]),
         "roundness_measured": bool(o["round_ok"]),
         "why": o.get("why") or "", "diameter_mm": r4(o["diameter_mm"]),
         "nominal_diameter_mm": r4(2 * o["nominal_radius_mm"]),
         "centre_offset_mm": None if o["centre_offset_mm"] is None
         else [r4(x) for x in o["centre_offset_mm"]],
         "rms_mm": r4(o["rms_mm"]), "max_abs_mm": r4(o["max_abs_mm"]),
         "valid_pct": o["valid_pct"], "max_gap_deg": o["max_gap_deg"], "n_angles": o["n_angles"],
         "ellipse": None if o["ellipse"] is None else
         {k: (r4(v) if not isinstance(v, str) else v) for k, v in o["ellipse"].items()},
         "level_dn": r4(o["level_dn"]), "contrast_dn": r4(o["contrast_dn"]),
         "saturated_pct": o["saturated_pct"], "inner_limit_pct": o.get("inner_limit_pct"),
         "mm_per_px": o["mm_per_px"],
         "units": {"diameter_mm": "mm, the outline's mean diameter in the face plane",
                   "residual_mm": "mm, radius - fitted outline (- = inside: a notch)",
                   "theta_deg": "deg, 0 = camera right, 90 = camera down"}}
    if j is not None:
        d.update({k: j[k] for k in ("status", "findings", "marginal", "tol_mm", "min_arc_mm",
                                    "guard_band_pct", "golden")})
        d["unjudged"] = j.get("unjudged") or []
        if j.get("why") and not d["why"]:
            d["why"] = j["why"]
        d["arc_peak_mm"] = r4(j.get("arc_peak_mm"))
        d["rms_mm"], d["max_abs_mm"] = r4(j["rms_mm"]), r4(j["max_abs_mm"])
        if "tol_max_mm" in j:
            d["tol_max_mm"] = j["tol_max_mm"]
    if arrays:
        d["theta_deg"] = [round(float(x), 2) for x in o["theta_deg"]]
        d["residual_mm"] = [None if not np.isfinite(x) else round(float(x), 4)
                            for x in o["residual_mm"]]
    return d

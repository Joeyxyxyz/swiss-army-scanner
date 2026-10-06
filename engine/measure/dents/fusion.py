"""
Dent detection proper: run the four finders on one pose, calibrate them on
a known-good cell (negative control), fuse, measure, classify.

CALIBRATION (null): on a known-good cell of the same type in the same
fixture, T_m = each finder's maximum over the good face and T_F = the
maximum of the fused sum (>= 3 px inside the face). 1.0 therefore means
"as strong as the strongest thing the good cell produced". A finder whose
good-face maximum is below min_T (1 robust sigma: every finder map is a
robust z) saw nothing to calibrate against - for the line finder: no
mirrored room line on the good face (lamp off, line outside the face) - and
is EXCLUDED (T = None, a warning in the null; detect_dents fuses the same
finders the null has) instead of being divided by a near-zero maximum.

FUSION:  e_m = clip(maxfilter_r3(S_m) / T_m, 0, 2.5)        (3 px location tolerance)
         F   = (1.0 e_warp + 0.8 e_photo + 0.6 e_phase + 0.8 e_line) / T_F
Peaks: detection F > thr = 1.25, watch 0.8..thr, edge = within 3 px of the
face boundary (reported, never a detection). The null is ONE cell's
maximum, so an independent clean face lands near 1.0 on either side of it
(bench good cell: a null made from one part of the face, evaluated on the
other part, reaches 1.13-1.14; five synthetic clean cells against one
synthetic good cell's null: 1.05-1.16): detection needs 25 % more than the
good cell's strongest response.
Greedy NMS 7 px, each peak then moved to where the uncapped, unfiltered
evidence peaks on its plateau (the 3 px tolerance filter and the 2.5 cap
make flat tops) without crossing the inner / edge boundary; score and
level are the NMS pick's. A weaker peak within 12 px on the same hill as
a stronger one (the saddle between them >= 0.7 x its score) is part of
that feature; peaks carried by the reflected line (line kink >= 1.0) along
one line within 16 px are one feature.

MEASURE: depth from the metric radial residual of the same pose (fixed-
radius cylinder), ring minus disk at two scales (disk r 2 / ring 5-9 px,
disk r 4 / ring 8-14 px), median over repeats, at the deepest position
within the 3 px location tolerance; the larger scale reported; + = pushed
in, mm. Unreliable (depth_reliable false) in the ambient / specular
zone (the mirrored room makes the decode a mirror artefact), in the dark
band (little projector light returns: decode noise) and where the metric
scans did not decode.

The same estimator at 200 random face positions gives the scan's depth
noise p90 (a robust level: a dent covers < 1 % of the face, so it cannot
lift the p90 the way it lifts a p99). depth_significant needs depth >
1.5 x p90 (on the bench good cell 1.5 x p90 is above the p99 and the
maximum of 400 random positions: p99 / p90 = 1.30 there); a depth below it
is reported but flagged not significant.

CLASSIFY: scratch = a short thin bright ridge along the cell axis in the
projector-only image (Hessian ridge test, see _streak) WITHOUT bowl-like
depth (phase-depth evidence < 1.0 and no significant reliable metric
depth: a groove is narrower than the depth estimators, while glints on a
dent's wall can also form short axial ridges); speck = a tiny
bright spot (characteristic LoG scale <= 1 px, i.e. glitter) without depth
support (phase depth and line kink < 0.5, no significant reliable metric
depth); else dent. Every candidate carries its per-finder evidence. kind
is advisory (tuned on a handful of features); level is the decision.

TUNING PROVENANCE: four choices were made while scoring the bench's
re-placed bad-cell pose ("pose 2", meant as held-out): the line finder's
3-row dip bridge (it alone restores pose 2's lower dent) and its wider
tracking band (cosv > 0.2), the photometric blob sigma 1.5 / 2.5 px and
sin90 T36 in the production pattern set (both for pose 2's small dent).
Results on pose 2 are therefore TUNED-ON, not held-out. DEV_PARAMS (with
pattern subset "minimal") restores the development finder's choices,
fixed before pose 2 was looked at. The detection threshold (1.25) and the
depth significance factor (1.5) were set from the good cell, synthetic
clean cells and pose 1 only.
"""

from __future__ import annotations

import json
import time
from typing import Optional

import numpy as np
from scipy import ndimage as ndi

from . import finder_line, finder_phase, finder_photo, finder_warp
from ._util import Quiet, Rect, disk
from .geometry import cell_geometry, load_metric

FINDERS = ("pattern_warp", "photometric", "phase_depth", "line_kink")
DEFAULTS = dict(
    weights=dict(pattern_warp=1.0, photometric=0.8, phase_depth=0.6, line_kink=0.8),
    tol_px=3, e_clip=2.5, thr=1.25, watch=0.8, nms_px=7, edge_px=3, line_merge_px=16,
    merge_px=12, merge_saddle=0.7, min_T=1.0,
    face_cosv=0.3, end_margin_mm=2.5, bottom_extend_mm=0.0, line_track_cosv=0.2,
    line_bridge_rows=3, photo_blob_sigma=(1.5, 2.5),
    depth_noise_n=200, depth_sig_factor=1.5,
    spec_ratio=2.0, dark_ratio=0.2, streak_len_px=5, streak_deg=20.0, speck_sigma_px=1.0,
)
# what the finder maps and the null's T / T_F depend on: recorded in the null, and a detection
# run with other values is warned about (its scores are not relative to that null)
ANALYSE_KEYS = ("face_cosv", "end_margin_mm", "bottom_extend_mm", "line_track_cosv",
                "line_bridge_rows", "photo_blob_sigma")
NULL_KEYS = ANALYSE_KEYS + ("tol_px", "edge_px", "e_clip", "weights")
# the development finder's choices, fixed before pose 2 was examined: the line
# followed on the analysed face only (line_track_cosv = face_cosv), no dip bridge,
# photometric blob sigma 1 / 2 px. Use with subset "minimal" for the pre-pose-2 pattern set.
DEV_PARAMS = dict(line_track_cosv=0.3, line_bridge_rows=0, photo_blob_sigma=(1.0, 2.0))
UNITS = {"x": "px (full camera)", "y": "px (full camera)", "theta_deg": "deg about the cell axis "
         "(0 = facing the camera, + toward camera +x)",
         "s_mm": "mm along the fitted cell axis (+ down the image); origin = the axis point level "
                 "with the camera's optical axis (camera-frame y = 0), NOT a cell end",
         "s_from_top_mm": "mm along the axis below the top end of the decoded cell surface of "
                          "this pose (geometry s_range_mm[0]; the cell's own top only when the "
                          "zone includes it)",
         "score": "fused evidence / good-cell maximum (1.0 = the known-good cell's strongest "
                  "response); level detection: score > thr (params, default 1.25), watch: "
                  "0.8..thr, edge: within 3 px of the face boundary",
         "depth_mm": "mm, + = pushed in (metric residual, ring minus disk)",
         "depth_significant": "depth > depth_sig_factor (1.5) x the face's depth noise p90",
         "size_mm": "mm, footprint of the anomaly evidence (>= half its peak), not a caliper size",
         "evidence": "per finder: clip(local max / good-cell max, 0, 2.5); finders the null "
                     "calibration excluded are absent"}


# ------------------------------------------------------------ one pose

def analyse(data, metric_dirs, cal, radius_mm: float, params: Optional[dict] = None) -> dict:
    """Geometry + the four finder maps of one pose (crop coords)."""
    p = dict(DEFAULTS)
    p.update(params or {})
    t0 = time.perf_counter()
    metric = load_metric(metric_dirs, data.x0, data.y0, data.shape)
    geo = cell_geometry(metric, cal, radius_mm, data.x0, data.y0, data.shape, p["face_cosv"],
                        p["end_margin_mm"], p["bottom_extend_mm"])
    with Quiet():
        cxs = [m["corr_x"] for m in metric if m["corr_x"] is not None]
        cys = [m["corr_y"] for m in metric if m["corr_y"] is not None]
        geo["corr_x"] = np.nanmedian(np.stack(cxs), 0) if cxs else None
        geo["corr_y"] = np.nanmedian(np.stack(cys), 0) if cys else None
    secs = {"geometry": round(time.perf_counter() - t0, 2)}
    R = Rect.from_face(geo["face"])
    maps, info = {}, {}
    t = time.perf_counter()
    maps["pattern_warp"], info["pattern_warp"] = finder_warp.score_map(data, geo, R)
    secs["pattern_warp"] = round(time.perf_counter() - t, 2)
    t = time.perf_counter()
    maps["photometric"], info["photometric"] = finder_photo.score_map(
        data, geo, R, {"blob_sigma": tuple(p["photo_blob_sigma"])})
    secs["photometric"] = round(time.perf_counter() - t, 2)
    t = time.perf_counter()
    maps["phase_depth"], info["phase_depth"] = finder_phase.score_map(data, geo)
    secs["phase_depth"] = round(time.perf_counter() - t, 2)
    t = time.perf_counter()
    amb = data.hdr(level_on=False)
    s0, s1 = geo["s_range"]
    with Quiet():
        track = (np.nan_to_num(geo["cosv"]) > p["line_track_cosv"]) \
            & (geo["s"] > s0 + p["end_margin_mm"]) \
            & (geo["s"] < s1 + p["bottom_extend_mm"])
    maps["line_kink"], info["line_kink"] = finder_line.score_map(
        amb, geo["face"], track | geo["face"], bridge=int(p["line_bridge_rows"]))
    secs["line_kink"] = round(time.perf_counter() - t, 2)
    secs["total"] = round(time.perf_counter() - t0, 2)
    return {"data": data, "geo": geo, "rect": R, "maps": maps, "info": info, "seconds": secs,
            "metric_dirs": [str(d) for d in metric_dirs], "params": p}


# ------------------------------------------------------------ evidence

def _inner(face, r):
    return ndi.distance_transform_edt(face) > r


def _tol_max(a, face, r):
    b = np.where(face & np.isfinite(a), a, -np.inf)
    m = ndi.maximum_filter(b, footprint=disk(r))
    return np.where(face & np.isfinite(m), m, np.nan)


def evidence(maps: dict, face, T: dict, p: dict) -> tuple:
    E = {}
    for n, a in maps.items():
        if T.get(n) is None:
            continue
        m = _tol_max(a, face, p["tol_px"])
        with Quiet():
            E[n] = np.clip(np.nan_to_num(m / T[n], nan=0.0), 0, p["e_clip"])
    F = np.zeros(face.shape)
    for n, e in E.items():
        F += p["weights"].get(n, 0.0) * e
    return E, np.where(face, F, np.nan)


def _norm(v):
    """JSON form (tuples -> lists) so a stored and a live parameter set compare equal."""
    return json.loads(json.dumps(v, sort_keys=True))


def calibrate_null(analyses: list, params: Optional[dict] = None) -> dict:
    """Negative control: T_m = max of each finder over every good face, T_F = max of the
    fused sum >= edge_px inside the face (over every good pose). A finder with no response
    on the good face (maximum < min_T, or the line finder found no mirrored line) is
    excluded (T None) with a warning: dividing by a near-zero maximum would flood every
    later cell with capped evidence."""
    p = dict(DEFAULTS)
    p.update(params or {})
    T, excluded = {}, {}
    for n in FINDERS:
        vals = []
        for an in analyses:
            a = an["maps"].get(n)
            if a is None:
                continue
            v = _tol_max(a, an["geo"]["face"], p["tol_px"])[an["geo"]["face"]]
            v = v[np.isfinite(v)]
            if v.size:
                vals.append(float(v.max()))
        top = max(vals) if vals else None
        if n == "line_kink" and not any((an.get("info") or {}).get("line_kink")
                                        for an in analyses):
            excluded[n] = ("no mirrored room line was found on the good face (lamp off, or the "
                           "line runs outside the face)")
        elif top is None:
            excluded[n] = "no response map on the good face"
        elif top < p["min_T"]:
            excluded[n] = (f"its maximum on the good face is {top:.3g}, below min_T "
                           f"{p['min_T']:g} (nothing to calibrate against)")
        T[n] = None if n in excluded else float(top)
    if all(v is None for v in T.values()):
        raise ValueError("no dent finder responded on the good cell: " +
                         "; ".join(f"{k}: {v}" for k, v in excluded.items()))
    warnings = [f"finder {n} is EXCLUDED from this null calibration: {why}; detection fuses "
                "the other finders only (recalibrate on a good cell where it responds to use it)"
                for n, why in excluded.items()]
    fused_max, p999, face_px = [], [], []
    for an in analyses:
        face = an["geo"]["face"]
        _, F = evidence(an["maps"], face, T, p)
        inner = _inner(face, p["edge_px"])
        fused_max.append(float(np.nanmax(F[inner])))
        p999.append(float(np.nanpercentile(F[inner], 99.9)))
        face_px.append(int(face.sum()))
    T_F = max(fused_max)
    ap = analyses[0].get("params") or {}                 # what the finder maps were made with
    fparams = {k: ap.get(k, p[k]) if k in ANALYSE_KEYS else p[k] for k in NULL_KEYS}
    return {"T": T, "T_F": T_F, "weights": dict(p["weights"]), "tol_px": p["tol_px"],
            "edge_px": p["edge_px"], "e_clip": p["e_clip"],
            "finders": [n for n in FINDERS if T[n] is not None], "excluded_finders": excluded,
            "min_T": p["min_T"], "finder_params": _norm(fparams),
            "null_fused_max": [f / T_F for f in fused_max],
            "null_fused_p999": [v / T_F for v in p999], "face_px": face_px,
            "warnings": warnings}


# ------------------------------------------------------------ measure

def _ring_minus_disk(devs, xc, yc, broad: bool) -> tuple:
    r_in, r1, r2 = (4, 8, 14) if broad else (2, 5, 9)
    H, W = devs.shape[1:]
    y0, y1 = max(yc - r2, 0), min(yc + r2 + 1, H)
    x0, x1 = max(xc - r2, 0), min(xc + r2 + 1, W)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d2 = (xx - xc) ** 2 + (yy - yc) ** 2
    dk, rg = d2 <= r_in * r_in, (d2 >= r1 * r1) & (d2 <= r2 * r2)
    vals = []
    for dv in devs:
        w = dv[y0:y1, x0:x1]
        a, b = w[dk], w[rg]
        if np.isfinite(a).sum() >= max(3, dk.sum() // 3) and np.isfinite(b).sum() >= 8:
            vals.append(float(np.nanmedian(b) - np.nanmedian(a)))
    return (float(np.median(vals)) if vals else None), len(vals)


def depth_at(devs, xc, yc, broad: bool, search_px: int = 0) -> tuple:
    """Ring minus disk of the metric radial residual (mm, + = pushed in), median over
    repeats, at the deepest position within search_px of (xc, yc) (the fused peak may sit
    up to the location tolerance off the dent centre) -> (depth or None, repeats, (dx, dy))."""
    best = (None, 0, (0, 0))
    for dy in range(-search_px, search_px + 1):
        for dx in range(-search_px, search_px + 1):
            if dx * dx + dy * dy > search_px * search_px:
                continue
            d, n = _ring_minus_disk(devs, xc + dx, yc + dy, broad)
            if d is not None and (best[0] is None or d > best[0]):
                best = (d, n, (dx, dy))
    return best


def depth_noise(devs, face, search_px: int, n: int = 200, seed: int = 0) -> Optional[float]:
    """p90 of the same depth estimator (both scales, the larger, same search) at random
    positions >= 6 px inside the face: the scale of the scan's depth noise (a significant
    depth must beat depth_sig_factor x this; see the module doc)."""
    inner = ndi.distance_transform_edt(face) > 6
    ys, xs = np.nonzero(inner)
    if len(ys) < 20:
        return None
    rng = np.random.default_rng(seed)
    vals = []
    for i in rng.choice(len(ys), min(n, len(ys)), replace=False):
        x, y = int(xs[i]), int(ys[i])
        dd = [d for d in (depth_at(devs, x, y, False, search_px)[0],
                          depth_at(devs, x, y, True, search_px)[0]) if d is not None]
        if dd:
            vals.append(max(dd))
    return float(np.percentile(vals, 90)) if len(vals) >= 10 else None


def spot_scale(img: np.ndarray, sigmas=(0.7, 1.0, 1.4, 2.0, 2.8)) -> tuple:
    """Scale-normalised LoG of the log projector image (bright blobs positive) at a few
    sigmas -> (stack, sigmas); the characteristic scale of a spot is its argmax."""
    la = np.log(np.maximum(img, 1.0))
    st = np.stack([-(s * s) * ndi.gaussian_laplace(la, s) for s in sigmas])
    return st, np.asarray(sigmas)


def _spot_sigma(stack, sigmas, xc, yc, r=3):
    H, W = stack.shape[1:]
    w = stack[:, max(yc - r, 0):min(yc + r + 1, H), max(xc - r, 0):min(xc + r + 1, W)]
    if w.size == 0:
        return None, 0.0
    k, iy, ix = np.unravel_index(int(np.argmax(w)), w.shape)
    return float(sigmas[k]), float(w[k, iy, ix])


def _footprint(G, face, xc, yc, mm_per_px, cosv, r=12):
    """Equivalent diameter (mm) of the evidence >= half its local peak around (xc, yc)."""
    H, W = G.shape
    y0, y1, x0, x1 = max(yc - r, 0), min(yc + r + 1, H), max(xc - r, 0), min(xc + r + 1, W)
    g = np.where(face[y0:y1, x0:x1], np.nan_to_num(G[y0:y1, x0:x1]), 0.0)
    cy, cx = yc - y0, xc - x0
    k = g[max(cy - 3, 0):cy + 4, max(cx - 3, 0):cx + 4]
    if k.size == 0 or k.max() <= 0:
        return None, 0
    pk = float(k.max())
    iy, ix = np.unravel_index(int(np.argmax(k)), k.shape)
    iy += max(cy - 3, 0)
    ix += max(cx - 3, 0)
    lab, _ = ndi.label(g >= 0.5 * pk)
    area = int((lab == lab[iy, ix]).sum()) if lab[iy, ix] else 1
    c = float(np.clip(np.nan_to_num(cosv, nan=1.0), 0.3, 1.0))
    return 2.0 * np.sqrt(area * mm_per_px ** 2 / c / np.pi), area


def _axis_dir(geo, xc, yc):
    """Unit image direction (dx, dy) of the cell axis at a pixel (increasing s)."""
    s = geo["s"]
    H, W = s.shape
    y0, y1, x0, x1 = max(yc - 3, 0), min(yc + 4, H), max(xc - 3, 0), min(xc + 4, W)
    with Quiet():
        gy, gx = np.gradient(s[y0:y1, x0:x1])
        g = np.array([np.nanmedian(gx), np.nanmedian(gy)])
    if not np.isfinite(g).all() or np.linalg.norm(g) < 1e-9:
        return np.array([0.0, 1.0])
    return g / np.linalg.norm(g)


def _streak(img, face, geo, xc, yc, p, r=12):
    """Scratch test: a SHORT thin bright ridge along the cell axis near the candidate, in
    the projector-only image (HDR on - off: the mirrored room cancels). Hessian at sigma 1
    of its log: ridge pixels have a strong negative curvature across (lambda_1 < -3 sigma
    of the face) and little along (|lambda_2| < 0.35 |lambda_1|), with the ridge running
    within streak_deg of the axis. The longest connected ridge passing within 3 px of the
    candidate decides: scratch when it is >= streak_len_px long, >= 2x longer than wide,
    and ends inside the +-r px window (a ridge running through it is a glint / room line
    along the whole cell, not a scratch). -> dict(length_px, width_px, angle_deg,
    contrast, scratch)."""
    la = np.log(np.maximum(img, 1.0))
    Hxx = ndi.gaussian_filter(la, 1.0, order=(0, 2))
    Hyy = ndi.gaussian_filter(la, 1.0, order=(2, 0))
    Hxy = ndi.gaussian_filter(la, 1.0, order=(1, 1))
    tr, dt = Hxx + Hyy, Hxx * Hyy - Hxy ** 2
    disc = np.sqrt(np.maximum(tr * tr / 4 - dt, 0))
    l1, l2 = tr / 2 - disc, tr / 2 + disc                   # l1 <= l2
    vx, vy = Hxy, l1 - Hxx                                  # eigenvector of l1 (across)
    nrm = np.hypot(vx, vy) + 1e-12
    rdx, rdy = -vy / nrm, vx / nrm                          # along the ridge
    ax = _axis_dir(geo, xc, yc)
    cosang = np.abs(rdx * ax[0] + rdy * ax[1])
    fvals = l1[face & np.isfinite(l1)]
    sig = 1.4826 * float(np.median(np.abs(fvals - np.median(fvals)))) if fvals.size else 1.0
    ridge = face & (l1 < -3.0 * sig) & (np.abs(l2) < 0.35 * np.abs(l1)) \
        & (cosang > np.cos(np.radians(p["streak_deg"])))
    H, W = img.shape
    y0, y1, x0, x1 = max(yc - r, 0), min(yc + r + 1, H), max(xc - r, 0), min(xc + r + 1, W)
    lab, n = ndi.label(ridge[y0:y1, x0:x1], structure=np.ones((3, 3)))
    best = {"length_px": 0.0, "width_px": 0.0, "angle_deg": None, "contrast": 0.0, "scratch": False}
    yy, xx = np.mgrid[y0:y1, x0:x1]
    for k in range(1, n + 1):
        m = lab == k
        if np.min((yy[m] - yc) ** 2 + (xx[m] - xc) ** 2) > 9:
            continue
        proj = (xx[m] - xc) * ax[0] + (yy[m] - yc) * ax[1]
        across = (xx[m] - xc) * -ax[1] + (yy[m] - yc) * ax[0]
        length = float(proj.max() - proj.min() + 1)
        width = float(across.max() - across.min() + 1)
        through = bool(proj.min() <= -(r - 1) or proj.max() >= r - 1)
        if length > best["length_px"]:
            ang = float(np.degrees(np.arccos(np.clip(np.median(cosang[y0:y1, x0:x1][m]), -1, 1))))
            best = {"length_px": length, "width_px": width, "angle_deg": round(ang, 1),
                    "contrast": round(float(-np.median(l1[y0:y1, x0:x1][m]) / max(sig, 1e-9)), 1),
                    "through_window": through,
                    "scratch": bool(length >= p["streak_len_px"] and length >= 2.0 * width
                                    and not through)}
    return best


# ------------------------------------------------------------ detect

def pick_peaks(F: np.ndarray, Gu: np.ndarray, inner: np.ndarray, p: dict) -> list:
    """Greedy NMS (nms_px) over F >= watch, strongest first. The tolerance max-filter (and
    the 2.5 cap) make plateaus, so each pick is then moved to where the uncapped, unfiltered
    evidence Gu peaks on its plateau (F >= 0.97 x the pick's F, within tol_px + 2), never
    across the inner / edge boundary. Score and edge status are the NMS pick's (as in the
    development fusion), so refinement cannot turn an edge peak into a detection.
    -> [(y, x, score, at_edge)] crop px."""
    Fz = np.nan_to_num(F, nan=-1.0)
    tie = Fz + 1e-3 * ndi.gaussian_filter(Fz, 2.0)
    ys_, xs_ = np.nonzero(Fz >= p["watch"])
    order = np.argsort(-tie[ys_, xs_])
    picked = []
    for i in order:
        yq, xq = int(ys_[i]), int(xs_[i])
        if all((yq - a) ** 2 + (xq - b) ** 2 > p["nms_px"] ** 2 for a, b in picked):
            picked.append((yq, xq))
    out = []
    H, W = Fz.shape
    r = p["tol_px"] + 2
    for yq, xq in picked:
        y0_, y1_, x0_, x1_ = max(yq - r, 0), min(yq + r + 1, H), max(xq - r, 0), min(xq + r + 1, W)
        same_side = inner[y0_:y1_, x0_:x1_] == inner[yq, xq]
        ok = (Fz[y0_:y1_, x0_:x1_] >= 0.97 * Fz[yq, xq]) & same_side
        win = np.where(ok, Gu[y0_:y1_, x0_:x1_], -np.inf)
        iy, ix = np.unravel_index(int(np.argmax(win)), win.shape)
        yr, xr = y0_ + iy, x0_ + ix
        if all((yr - a) ** 2 + (xr - b) ** 2 > 4 for a, b, _, _ in out):
            out.append((int(yr), int(xr), float(F[yq, xq]), not bool(inner[yq, xq])))
    return out


def detect(an: dict, null: dict, params: Optional[dict] = None) -> dict:
    """Fuse one analysed pose with a null calibration -> candidates + maps."""
    p = dict(DEFAULTS)
    p.update(params or {})
    p["weights"] = dict(DEFAULTS["weights"], **(null.get("weights") or {}))
    data, geo = an["data"], an["geo"]
    face = geo["face"]
    x0, y0 = data.x0, data.y0
    used = [n for n in FINDERS if null["T"].get(n) is not None]
    warnings = []
    skipped = [n for n in FINDERS if n not in used]
    if skipped:
        warnings.append(f"the null calibration excludes finder(s) {', '.join(skipped)} (no "
                        "response on its good cell): fused from " + ", ".join(used) + " only")
    fp_null = null.get("finder_params")
    if fp_null:
        ap = an.get("params") or {}
        live = _norm({k: ap.get(k, p[k]) if k in ANALYSE_KEYS else p[k] for k in NULL_KEYS})
        diff = sorted(k for k in NULL_KEYS if k in fp_null and fp_null[k] != live.get(k))
        if diff:
            warnings.append("the null calibration was made with other finder parameters ("
                            + ", ".join(f"{k} {fp_null[k]} vs {live[k]}" for k in diff)
                            + "): scores are not relative to it - recalibrate")
    if "line_kink" in used and not (an.get("info") or {}).get("line_kink"):
        warnings.append("the line finder found no mirrored room line on this cell (the null's "
                        "good cell had one): lamp off or moved? - slope anomalies of broad "
                        "shallow dents are not checked")
    E, Fraw = evidence(an["maps"], face, null["T"], p)
    F = Fraw / null["T_F"]
    G = np.zeros(face.shape)                                 # evidence without the tolerance filter
    Gu = np.zeros(face.shape)                                # ... and uncapped (location)
    for n, a in an["maps"].items():
        if null["T"].get(n):
            with Quiet():
                r = np.nan_to_num(a / null["T"][n])
            G += p["weights"].get(n, 0.0) * np.clip(r, 0, p["e_clip"])
            Gu += p["weights"].get(n, 0.0) * np.clip(r, 0, 10.0)
    G = np.where(face, G / null["T_F"], np.nan)
    Gu = ndi.gaussian_filter(np.where(face, Gu, 0.0), 1.0)
    on = data.hdr(level_on=True)
    off = data.hdr(level_on=False)
    proj = np.maximum(on - off, 1.0)
    with Quiet():
        amb_ratio = off / max(float(np.nanmedian(off[face])), 1.0)
        proj_ratio = proj / max(float(np.nanmedian(proj[face])), 1.0)
    inner = _inner(face, p["edge_px"])
    picked = pick_peaks(F, Gu, inner, p)
    cov = np.mean([np.isfinite(dv) for dv in geo["devs"]], 0)
    floor = depth_noise(geo["devs"], face, p["tol_px"], int(p["depth_noise_n"]))
    sig_mm = None if floor is None else p["depth_sig_factor"] * floor
    sstack, ssig = spot_scale(proj)
    cands = []
    for yc, xc, f, at_edge in picked:
        ev = {n: round(float(e[yc, xc]), 2) for n, e in E.items()}
        n_sup = int(sum(v >= 0.8 for v in ev.values()))
        d_s, n_s, o_s = depth_at(geo["devs"], xc, yc, False, p["tol_px"])
        d_b, n_b, o_b = depth_at(geo["devs"], xc, yc, True, p["tol_px"])
        dd = [d for d in (d_s, d_b) if d is not None]
        dmm = max(dd) if dd else None
        broad = d_b is not None and (d_s is None or d_b > d_s)
        win = (slice(max(yc - 2, 0), yc + 3), slice(max(xc - 2, 0), xc + 3))
        spec = bool(np.nanmax(amb_ratio[win]) > p["spec_ratio"])
        dark = bool(np.nanmin(proj_ratio[win]) < p["dark_ratio"])
        cov_c = float(np.nanmean(cov[max(yc - 4, 0):yc + 5, max(xc - 4, 0):xc + 5]))
        size_mm, area = _footprint(G, face, xc, yc, geo["mm_per_px"], geo["cosv"][yc, xc])
        st = _streak(proj, face, geo, xc, yc, p)
        flags = []
        if spec:
            flags.append("specular / ambient zone (mirrored room light): metric depth unreliable")
        if dark:
            flags.append("dark band (little projector light returns): metric depth unreliable")
        if dmm is None:
            flags.append("no metric depth (not decoded here)")
        elif cov_c < 0.5:
            flags.append(f"metric data sparse here ({cov_c:.0%} of the disk decoded)")
        reliable = bool(dmm is not None and not spec and not dark and cov_c >= 0.5)
        significant = bool(dmm is not None and sig_mm is not None and dmm > sig_mm)
        if dmm is not None and sig_mm is not None and not significant:
            flags.append(f"depth within the metric noise (<= {p['depth_sig_factor']:g} x the p90 "
                         f"{floor:.2f} mm of this face = {sig_mm:.2f} mm)")
        sig_spot, _ = _spot_sigma(sstack, ssig, xc, yc)
        bowl = ev.get("phase_depth", 0) >= 1.0 or (reliable and significant)
        if st["scratch"] and not bowl:
            kind = "scratch"
        elif (sig_spot is not None and sig_spot <= p["speck_sigma_px"]
              and ev.get("phase_depth", 0) < 0.5 and ev.get("line_kink", 0) < 0.5
              and not (reliable and significant)):
            kind = "speck"
        else:
            kind = "dent"
        level = "edge" if at_edge else ("detection" if f > p["thr"] else "watch")
        ranked = sorted(ev.items(), key=lambda t: -t[1])
        note = ["support: " + ", ".join(f"{n} {v}" for n, v in ranked),
                f"{n_sup} finder(s) >= 0.8 of their good-cell max"]
        if ev.get("line_kink", 0) >= 1.0:
            note.append("the mirrored room line kinks / splits here (slope anomaly)")
        if kind == "speck":
            note.append(f"tiny bright spot (scale {sig_spot:g} px) without depth support: "
                        "glitter / speck, not a dent")
        if kind == "scratch":
            note.append(f"thin bright ridge along the axis ({st['length_px']:.0f} px long, "
                        f"{st['angle_deg']} deg off the axis): scratch-like, not a round dent")
        cands.append({
            "x": int(xc + x0), "y": int(yc + y0),
            "theta_deg": round(float(geo["theta"][yc, xc]), 1),
            "s_mm": round(float(geo["s"][yc, xc]), 2),
            "s_from_top_mm": round(float(geo["s"][yc, xc] - geo["s_range"][0]), 2),
            "score": round(f, 3), "level": level, "kind": kind,
            "depth_mm": None if dmm is None else round(dmm, 3), "depth_reliable": reliable,
            "depth_scale": "broad" if broad else "small", "depth_repeats": n_b if broad else n_s,
            "depth_offset_px": list(o_b if broad else o_s), "depth_significant": significant,
            "spot_sigma_px": sig_spot,
            "depth_small_mm": None if d_s is None else round(d_s, 3),
            "depth_broad_mm": None if d_b is None else round(d_b, 3),
            "size_mm": None if size_mm is None else round(float(size_mm), 2),
            "evidence": ev, "n_support": n_sup, "streak": st,
            "specular_zone": spec, "dark_band": dark, "metric_coverage": round(cov_c, 2),
            "flags": flags, "note": "; ".join(note)})
    cands.sort(key=lambda c: -c["score"])
    merged = []
    for c in cands:
        # a secondary peak on the same hill as a stronger candidate (the saddle between them
        # stays >= merge_saddle x the weaker peak) is part of that feature (broad dents are
        # wider than the 7 px NMS)
        yc_, xc_ = c["y"] - y0, c["x"] - x0
        lab, _ = ndi.label(np.nan_to_num(F, nan=-1.0) >= p["merge_saddle"] * c["score"])
        host = next((m for m in merged
                     if (m["x"] - c["x"]) ** 2 + (m["y"] - c["y"]) ** 2 <= p["merge_px"] ** 2
                     and lab[m["y"] - y0, m["x"] - x0] == lab[yc_, xc_]), None)
        if host is not None and lab[yc_, xc_]:
            host.setdefault("merged_peaks", []).append([c["x"], c["y"], c["score"]])
            continue
        if c["evidence"].get("line_kink", 0) >= 1.0:
            host = next((m for m in merged if m["evidence"].get("line_kink", 0) >= 1.0
                         and (m["x"] - c["x"]) ** 2 + (m["y"] - c["y"]) ** 2
                         <= p["line_merge_px"] ** 2), None)
            if host is not None:
                lr = host.setdefault("line_rows", [host["y"], host["y"]])
                host["line_rows"] = [min(lr[0], c["y"]), max(lr[1], c["y"])]
                continue
        merged.append(c)
    for m in merged:
        if "line_rows" in m:
            m["note"] += f"; line anomaly spans rows {m['line_rows'][0]}-{m['line_rows'][1]}"
        if m.get("merged_peaks"):
            m["note"] += f"; {len(m['merged_peaks'])} weaker peak(s) on the same feature merged"
    inner_F = F[inner]
    return {"candidates": merged, "F": F.astype(np.float32), "E": E, "G": G, "face": face,
            "max_score": float(np.nanmax(inner_F)) if np.isfinite(inner_F).any() else None,
            "depth_noise_p90_mm": floor, "depth_significant_above_mm": sig_mm,
            "finders": used, "warnings": warnings,
            "params": {k: v for k, v in p.items()}}

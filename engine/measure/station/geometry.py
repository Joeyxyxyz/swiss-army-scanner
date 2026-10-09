"""
Station geometry: cylinder end faces + a ribbed/serpentine tube edge, from ONE metric scan (column-axis
ray-plane cloud, mm) and a flood image. Plain numpy / OpenCV / scipy on arrays; verbs.py does the
library side (rules, output bin, goldens).

  1. cells on the flood: Hough on (white - black) at the expected face size (from the scan's
     depth), brightest first, non-overlapping, each refined by an ellipse fit on the rim contour
     (contour points far from the expected circle - where a neighbour touches the rim -
     rejected; a leaking / broken disc keeps its Hough circle) and KEPT only if it is a face in
     the scan: its outline the expected size, its disc measured and flat, and proud of its
     surroundings (a flat patch of a wall is not a face). The flood may be the decode's fringe
     modulation (a stand-in: saturated faces are dim in it, the depth checks sort that out).
  2. face planes: robust plane (3 MAD) of the scan's points inside each disc (shrunk a few px);
     face centre ('rim_circle', default): sub-pixel bright -> dark rim edges on rays from the
     disc centre, back-projected onto that plane, robust circle fitted IN the plane (a dark dent
     spot or a bright neighbour at the rim does not move it); 'ellipse_ray' (the prototype): the
     camera ray through the rim ellipse centre on that plane.
  3. common face frame: n = normalised sum of the face normals (toward the camera), origin O =
     the mean of the face centres, e1 = y_cam x n (~ image right), e2 = n x e1 (~ image down).
     Heights along n: + = toward the camera (proud), - = recessed.
  4. cells: height of each face centre vs the mean face plane; pairwise differences Cj - Ci along
     n, centre spacing in the face plane, angle between the face normals, flatness.
  5. CT front edge: points a few mm behind the mean face plane (the height band: given, or found
     automatically: the 7 mm window, slid over -25..-0.5 mm, whose thin runs cover the most
     camera columns, re-centred on their median height), outside every cell body (3-D distance to the
     cell axis > R + margin, not in front of the face) and outside the face discs in the image;
     per camera column the THIN runs (the ribbon's edge seen end-on), then connected components
     wider than min_width_px.
  6. trapezoid wave: the edge centreline's lateral position w(s) in the face plane fitted by a
     trapezoid wave (pitch, phase vs the cylinder-pair centre, amplitude, plateau fraction, offset,
     slope) - part-fixed zones: one zone = one plateau + the two half-ramps beside it (half a
     pitch); zones are numbered T1..Tn left -> right over the zones kept (coverage, fit
     residual, optional end skips). Per zone: tube edge height vs the mean face plane (median,
     p5, p95, pixels).

Units: mm (camera frame: x right, y down, z forward), camera px, degrees.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np

from ...capture.mock_stereo import undistort_normalized

FACE_DIAMETER_MM = 21.05
OUTLINE_WARN_MM = 0.6       # warn when a face's flood outline misses its diameter by more (the
                            # bench faces' dark chamfer alone makes it ~0.3 mm small)


@dataclass
class StationParams:
    n_cells: int = 2
    face_diameter_mm: float = FACE_DIAMETER_MM
    order: str = "auto"                 # C1..Cn: 'rows' (top first), 'cols' (left first), 'auto'
    cell_px_hint: Optional[list] = None  # [[x, y], ...] camera px seeds (skip the Hough pick)
    hough_param1: float = 80.0
    hough_param2: float = 18.0
    hough_min_dist_px: float = 40.0
    r_px_range: Optional[list] = None   # Hough radius range, camera px; None = expected +-25 %
    min_brightness_frac: float = 0.5    # a disc must be this bright vs the brightest face
    min_face_coverage: float = 0.5      # a face disc must be at least this much measured ...
    max_face_rms_mm: float = 0.3        # ... and flat (robust plane rms): else not a face
    max_surround_same_frac: float = 0.5  # ... and proud: at most this much of the ring around
                                         # it (1.2-1.6 r) continues its plane (+-1.5 mm)
    face_disc_px: Optional[float] = None  # face-plane disc radius; None = the rim ellipse's
    face_shrink_px: float = 3.0         # the plane is fitted inside the disc shrunk by this
    centre_method: str = "rim_circle"   # 'rim_circle': sub-pixel flood rim edges back-projected
                                        # onto the face plane, circle fitted there; 'ellipse_ray':
                                        # the camera ray through the rim ellipse centre
    ct: bool = True                     # measure the tube edge
    ct_roi: Optional[list] = None       # [x0, y0, x1, y1] camera px; None = cell rows +- margin
    ct_rows_margin_r: float = 3.3       # default roi: the cells' rows +- this many face radii
    ct_band_mm: Optional[list] = None   # [lo, hi] edge height band vs the face plane; None = auto
    ct_band_search_mm: list = field(default_factory=lambda: [-25.0, -0.5])
    ct_band_halfwidth_mm: float = 3.5   # auto band = densest thin-run height +- this
    ct_run_px: list = field(default_factory=lambda: [2, 16])   # thin run per column, px
    ct_min_width_px: int = 40           # connected components narrower than this are dropped
    ct_body_margin_mm: float = 0.35     # cell body = distance to the cell axis < R + this
    ct_disc_margin_px: float = 1.5
    trap_pitch_range_mm: list = field(default_factory=lambda: [15.0, 40.0])
    zone_min_coverage: float = 0.8      # a zone must cover this much of its half pitch
    zone_max_fit_rms_mm: Optional[float] = None  # set: zones whose edge misfits the wave by more are out
    zone_skip: list = field(default_factory=lambda: [0, 0])   # drop zones at the (left, right) end
    zone_min_px: int = 20
    # quality gates against a background patch taken for a cell (a deep scene: a bright disc on
    # the wall behind can pass the flat / proud checks above); strict=False keeps only those.
    # The pair gates are judged on the whole set of candidate discs, not against whichever disc
    # happened to be accepted first (a bright background disc must not set aside the real
    # faces): the face set is chosen among the discs that pass every per-disc check, and a
    # validated disc left out of that set that conflicts with it in depth (or is brighter than
    # its faces, continues their row past max_pair_spacing_diam, or forms a group of its own)
    # makes the result ambiguous: refused, both named, never guessed
    strict: bool = True
    max_outline_rms_mm: float = 1.0     # flood rim circle rms on the face plane (good faces ~0.1-0.7)
    strict_min_coverage: float = 0.7    # measured fraction of the disc; below it (down to
                                        # min_face_coverage) only with a clean rim circle:
    strict_partial_outline_rms_mm: float = 0.3   # real faces measured 0.5-0.7 on the bench
                                        # (glint, zone edge) had rim rms 0.09-0.13 mm (one at
                                        # 0.699 on the decode-modulation flood: 0.49, refused);
                                        # background patches there measured 0.57-0.68 with no
                                        # rim circle or a 0.8 mm one (a tube rib), and one at
                                        # 0.565 had 0.62: a looser rim lets partly measured
                                        # background discs near the faces' depth in
    max_pair_dz_diam: float = 2.0       # |height difference| between any two faces, x diameter
    max_pair_spacing_diam: float = 6.0  # 3D centre spacing to the NEAREST other face, x diameter
                                        # (a row of faces may span more; each needs a neighbour,
                                        # and a disc that would continue the row is ambiguous)
    min_pair_spacing_diam: float = 0.9  # ... and to every other face at least this (no overlap)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "StationParams":
        d = dict(d or {})
        known = set(cls.__dataclass_fields__)
        bad = sorted(set(d) - known)
        if bad:
            raise ValueError(f"unknown station parameter(s) {bad}; known: {sorted(known)}")
        p = cls(**d)
        if isinstance(p.n_cells, bool) or not isinstance(p.n_cells, (int, np.integer)) or p.n_cells < 1:
            raise ValueError("n_cells must be a whole number >= 1")
        if p.order not in ("auto", "rows", "cols"):
            raise ValueError("order must be 'auto', 'rows' or 'cols'")
        if not p.face_diameter_mm or not np.isfinite(p.face_diameter_mm) or p.face_diameter_mm <= 0:
            raise ValueError("face_diameter_mm must be > 0")
        if p.centre_method not in ("rim_circle", "ellipse_ray"):
            raise ValueError("centre_method must be 'rim_circle' or 'ellipse_ray'")
        # the list parameters: a reversed / short one used to crash deep inside (or silently
        # find no CT); refuse it here with its name
        _pair(p.trap_pitch_range_mm, "trap_pitch_range_mm", lo_min=10.0, hi_max=80.0)
        _pair(p.ct_band_search_mm, "ct_band_search_mm")
        if p.ct_band_mm is not None:
            _pair(p.ct_band_mm, "ct_band_mm")
        if p.r_px_range is not None:
            _pair(p.r_px_range, "r_px_range", lo_min=1.0)
        lo, hi = _pair(p.ct_run_px, "ct_run_px", lo_min=1.0, equal_ok=True)
        if lo != int(lo) or hi != int(hi):
            raise ValueError(f"ct_run_px must be whole px [lo, hi], got {p.ct_run_px!r}")
        if p.ct_roi is not None:
            v = _numbers(p.ct_roi, 4, "ct_roi [x0, y0, x1, y1] camera px")
            if not (v[0] < v[2] and v[1] < v[3]):
                raise ValueError(f"ct_roi must be [x0, y0, x1, y1] with x0 < x1 and y0 < y1, got {p.ct_roi!r}")
        sk = _numbers(p.zone_skip, 2, "zone_skip [left, right]")
        if any(v < 0 or v != int(v) for v in sk):
            raise ValueError(f"zone_skip must be two whole numbers >= 0, got {p.zone_skip!r}")
        if p.cell_px_hint is not None:
            if not isinstance(p.cell_px_hint, (list, tuple)) or not p.cell_px_hint:
                raise ValueError("cell_px_hint must be [[x, y], ...] camera px")
            for h in p.cell_px_hint:
                _numbers(h, 2, "cell_px_hint entry [x, y]")
        if not (np.isfinite(p.ct_band_halfwidth_mm) and p.ct_band_halfwidth_mm > 0):
            raise ValueError("ct_band_halfwidth_mm must be > 0")
        if not 0 < p.zone_min_coverage <= 1:
            raise ValueError("zone_min_coverage must be in (0, 1]")
        if not 0 < p.strict_min_coverage <= 1:
            raise ValueError("strict_min_coverage must be in (0, 1]")
        for k in ("max_outline_rms_mm", "strict_partial_outline_rms_mm", "max_pair_dz_diam",
                  "max_pair_spacing_diam", "min_pair_spacing_diam"):
            v = getattr(p, k)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or v < 0:
                raise ValueError(f"{k} must be a number >= 0, got {v!r}")
        if p.min_pair_spacing_diam >= p.max_pair_spacing_diam:
            raise ValueError("min_pair_spacing_diam must be < max_pair_spacing_diam")
        return p


def _numbers(v, n: int, what: str) -> list:
    if not isinstance(v, (list, tuple, np.ndarray)) or len(v) != n:
        raise ValueError(f"{what}: {n} numbers expected, got {v!r}")
    try:
        out = [float(x) for x in v]
    except (TypeError, ValueError):
        raise ValueError(f"{what}: {n} numbers expected, got {v!r}") from None
    if not all(np.isfinite(out)):
        raise ValueError(f"{what}: finite numbers expected, got {v!r}")
    return out


def _pair(v, what: str, lo_min: Optional[float] = None, hi_max: Optional[float] = None,
          equal_ok: bool = False) -> tuple:
    lo, hi = _numbers(v, 2, f"{what} [lo, hi]")
    if lo > hi or (lo == hi and not equal_ok):
        raise ValueError(f"{what} must be [lo, hi] with lo < hi, got {v!r}")
    if (lo_min is not None and lo < lo_min) or (hi_max is not None and hi > hi_max):
        raise ValueError(f"{what} must lie within [{lo_min if lo_min is not None else '-inf'}, "
                         f"{hi_max if hi_max is not None else 'inf'}], got {v!r}")
    return lo, hi


class StationError(ValueError):
    """The scan / flood does not show what the station measurement needs."""


# ------------------------------------------------------------------ helpers

def flood_diff(flood) -> np.ndarray:
    """(2, H, W) [white, black] (either order) or (H, W) -> the lit-minus-dark image, float."""
    a = np.asarray(flood, np.float64)
    if a.ndim == 3:
        if a.shape[0] != 2:
            raise StationError(f"a flood pair is (2, H, W) [white, black], got {a.shape}")
        d = a[0] - a[1]
        if float(np.nanmean(d)) < 0:            # the HDMI path may invert: brighter first
            d = -d
        a = d
    if a.ndim != 2:
        raise StationError(f"flood must be (H, W) or (2, H, W), got {a.shape}")
    return np.clip(np.nan_to_num(a), 0, 255)


def conf_flood(conf: np.ndarray) -> np.ndarray:
    """The decode's fringe modulation as a flood stand-in (faces are bright discs in it too),
    stretched so its 99.5th percentile is 255."""
    c = np.nan_to_num(np.asarray(conf, np.float64))
    hi = float(np.percentile(c[c > 0], 99.5)) if (c > 0).any() else 1.0
    return np.clip(c * (255.0 / max(hi, 1e-6)), 0, 255)


def camera_rays(K, dist, px) -> np.ndarray:
    xy = undistort_normalized(np.asarray(px, np.float64).reshape(-1, 2), K, dist)
    return np.c_[xy, np.ones(len(xy))]


def fit_plane(P: np.ndarray, iters: int = 4, k: float = 3.0) -> tuple:
    """Robust plane: (centroid, normal toward the camera, inlier mask, finite points)."""
    P = P[np.all(np.isfinite(P), 1)]
    if len(P) < 10:
        raise StationError(f"only {len(P)} measured points for a plane")
    keep = np.ones(len(P), bool)
    for _ in range(iters):
        c = P[keep].mean(0)
        _, _, vt = np.linalg.svd(P[keep] - c, full_matrices=False)
        n = vt[2] * np.sign(-(vt[2] @ c))
        h = (P - c) @ n
        med = np.median(h[keep])
        mad = 1.4826 * np.median(np.abs(h[keep] - med))
        keep = np.abs(h - med) < k * max(mad, 1e-4)
    return c, n, keep, P


def angle_deg(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.degrees(np.arccos(np.clip(abs(a @ b) / np.linalg.norm(a) / np.linalg.norm(b),
                                              -1, 1))))


def expected_radius_px(X: np.ndarray, K, radius_mm: float) -> Optional[float]:
    """Face radius in camera px at the scan's typical depth (median z of the points)."""
    z = X[..., 2]
    z = z[np.isfinite(z)]
    if not len(z):
        return None
    f = 0.5 * (float(K[0][0]) + float(K[1][1]))
    return f * radius_mm / float(np.median(z))


# ------------------------------------------------------------------ 1. cells on the flood

def _refine(g: np.ndarray, cx: float, cy: float, r: float, r_win: Optional[float] = None) -> dict:
    """Ellipse fit on the rim contour of the bright disc around (cx, cy) (as the prototype),
    in a window of +- 1.68 x r_win (the expected face radius; default r) px."""
    import cv2
    H, W = g.shape
    half = int(round(1.68 * float(r_win or r)))
    x0, y0 = int(cx) - half, int(cy) - half
    x0c, y0c = max(x0, 0), max(y0, 0)
    x1c, y1c = min(x0 + 2 * half, W), min(y0 + 2 * half, H)
    win = g[y0c:y1c, x0c:x1c].astype(float)
    wy, wx = np.mgrid[y0c:y1c, x0c:x1c]
    rr = np.hypot(wx - cx, wy - cy)
    inner, outer = win[rr < 0.7 * r], win[rr > 1.3 * r]
    if not len(inner) or not len(outer):
        return {"px": [float(cx), float(cy)], "r_px": float(r), "ellipse_axes_px": [2 * r, 2 * r],
                "ellipse_angle_deg": 0.0, "rim_pts_used": 0, "rim_pts": 0, "refined": False}
    thr = 0.5 * (np.median(inner) + np.median(outer))
    _, lab = cv2.connectedComponents((win > thr).astype(np.uint8))
    iy, ix = int(cy) - y0c, int(cx) - x0c
    k0 = lab[iy, ix] if (0 <= iy < lab.shape[0] and 0 <= ix < lab.shape[1]) else 0
    if k0 == 0:                         # a dark spot (a dent) at the centre: the bright
        core = lab[rr < 0.5 * r]        # component that covers most of the inner disc
        core = core[core > 0]
        if not len(core):
            return {"px": [float(cx), float(cy)], "r_px": float(r),
                    "ellipse_axes_px": [2 * r, 2 * r], "ellipse_angle_deg": 0.0,
                    "rim_pts_used": 0, "rim_pts": 0, "refined": False}
        k0 = int(np.bincount(core).argmax())
    comp = (lab == k0).astype(np.uint8)
    cs, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pts = max(cs, key=cv2.contourArea)[:, 0, :].astype(float) + [x0c, y0c]
    ex, ey, a, b, ang = cx, cy, 2 * r, 2 * r, 0.0
    k = np.ones(len(pts), bool)
    for _ in range(3):
        rho = np.hypot(pts[:, 0] - ex, pts[:, 1] - ey)
        k = np.abs(rho - np.median(rho)) < 1.5
        if k.sum() < 6:
            break
        (ex, ey), (a, b), ang = cv2.fitEllipse(pts[k].astype(np.float32))
    rf = float((a + b) / 4)
    if np.hypot(ex - cx, ey - cy) > 0.3 * r or abs(rf - r) > 0.25 * r:
        # the thresholded disc leaked into a bright neighbour (or broke up): keep the Hough
        # circle; the rim edges (centre_method rim_circle) still refine the centre
        return {"px": [float(cx), float(cy)], "r_px": float(r), "ellipse_axes_px": [2 * r, 2 * r],
                "ellipse_angle_deg": 0.0, "rim_pts_used": int(k.sum()), "rim_pts": int(len(k)),
                "refined": False}
    return {"px": [float(ex), float(ey)], "r_px": rf,
            "ellipse_axes_px": [float(a), float(b)], "ellipse_angle_deg": float(ang),
            "rim_pts_used": int(k.sum()), "rim_pts": int(len(k)), "refined": True}


def rim_edges(g: np.ndarray, cx: float, cy: float, r: float, n_ang: int = 180,
              span=(0.7, 1.3)) -> np.ndarray:
    """Sub-pixel bright -> dark edge along rays from (cx, cy): per ray the steepest descent of
    the smoothed flood between span[0] r and span[1] r (parabola peak) -> (N, 3) [x, y, drop];
    rays without a clear descent are left out."""
    import cv2
    gf = cv2.GaussianBlur(np.asarray(g, np.float32), (0, 0), 1.0)
    rr = np.arange(span[0] * r, span[1] * r, 0.25)
    ang = np.linspace(0, 2 * np.pi, n_ang, endpoint=False)
    xs = (cx + np.outer(np.cos(ang), rr)).astype(np.float32)
    ys = (cy + np.outer(np.sin(ang), rr)).astype(np.float32)
    v = cv2.remap(gf, xs, ys, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    dv = np.diff(v, axis=1)                      # per 0.25 px
    out = []
    for a in range(n_ang):
        i = int(np.argmin(dv[a]))
        drop = -float(dv[a, i]) * 4.0            # DN per px
        lo, hi = float(v[a].max()), float(v[a].min())
        if drop <= 0 or lo - hi < 8.0:
            continue
        if 0 < i < dv.shape[1] - 1:
            y0_, y1_, y2_ = dv[a, i - 1], dv[a, i], dv[a, i + 1]
            den = y0_ - 2 * y1_ + y2_
            off = 0.5 * (y0_ - y2_) / den if den != 0 else 0.0
        else:
            off = 0.0
        rho = rr[0] + (i + 0.5 + float(np.clip(off, -1, 1))) * 0.25
        out.append((cx + rho * np.cos(ang[a]), cy + rho * np.sin(ang[a]), drop))
    return np.array(out, float).reshape(-1, 3)


def circle_in_plane(edge_px: np.ndarray, c, n, K, dist, iters: int = 6, k: float = 3.0):
    """Edge px back-projected onto the plane (c, n), robust algebraic circle there ->
    (centre 3-D, radius mm, residual rms mm, points used) or None."""
    if len(edge_px) < 12:
        return None
    d = camera_rays(K, dist, edge_px[:, :2])
    P = d * ((c @ n) / (d @ n))[:, None]
    e1 = np.cross(n, [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    xy = np.c_[(P - c) @ e1, (P - c) @ e2]
    keep = np.ones(len(xy), bool)
    for _ in range(iters):
        A = np.c_[2 * xy[keep], np.ones(keep.sum())]
        s = np.linalg.lstsq(A, (xy[keep] ** 2).sum(1), rcond=None)[0]
        q, rad = s[:2], np.sqrt(max(s[2] + s[0] ** 2 + s[1] ** 2, 0.0))
        res = np.hypot(xy[:, 0] - q[0], xy[:, 1] - q[1]) - rad
        mad = 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep])))
        new = np.abs(res - np.median(res[keep])) < max(k * mad, 0.05)
        if new.sum() < 12 or np.array_equal(new, keep):
            break
        keep = new
    return c + q[0] * e1 + q[1] * e2, float(rad), float(np.sqrt(np.mean(res[keep] ** 2))), int(keep.sum())


MAX_EXAMINED = 60          # strict selection: candidates validated first (brightest), before
                           # the rest are looked at only when no face set is found among them
MAX_SEARCH_STEPS = 200000  # strict selection: subset search budget (then refused, never guessed)


def _overlap(a: dict, b: dict) -> bool:
    (ax, ay, ar), (bx, by, br) = a["hough_px"], b["hough_px"]
    return bool(np.hypot(ax - bx, ay - by) <= ar + br)


def _select_cells(cands: list, g: np.ndarray, d: np.ndarray, prm: StationParams,
                  r_exp_px: Optional[float], validate, pair) -> tuple:
    """Strict selection: every candidate disc (brightest first) is validated on its own; the
    face set is then the brightest set of n_cells validated discs in which every pair passes
    pair(a, b) -> (why, near, depth_conflict), no two overlap, none is dimmer than
    min_brightness_frac x the set's brightest, and each face has a near neighbour - so no
    single disc (e.g. a bright background disc) is the reference the others are judged by.
    Fail closed: a validated disc left out of the chosen set that is incompatible with it (or
    has no face of it within max_pair_spacing_diam x D: a separate group) and is at least as
    bright as the set's dimmest face, or that conflicts with it in depth and either goes with
    another left-out disc there (a partial set: the part with a face missing?) or makes a full
    set of its own, makes the result ambiguous: StationError naming both. So does a disc that
    the pair gates let go with every face of the set but continues its row: within the bound
    of one face, past it of another - the set + it span more than one face set, and
    brightness alone (a bright background disc beyond the end of the row) must not decide
    which faces are the part: refused when it goes with every face, and when only the
    brightness rule keeps it out but it makes a full set with other discs; with n_cells > 1
    the same within the bound among the faces (a compact group of more discs than n_cells;
    n_cells = 1 takes the brightest disc of such a group). So does a disc at
    the set's depth with no face of it within the bound that goes with another such disc (a
    group of its own, partial or full). A dimmer lone disc is set aside (named in
    `rejected`, pair-gate conflicts first). -> (faces, rejected), faces shorter than n_cells
    when no set passes."""
    n = prm.n_cells
    hint = bool(prm.cell_px_hint)
    examined = []                                # [(cell, ok, why)] brightest first

    def examine(batch):
        for m, cx, cy, r in batch:
            c = _refine(g, cx, cy, r, r_exp_px)
            c.update({"hough_px": [cx, cy, r], "flood_level": m})
            c["edge_px"] = rim_edges(d, c["px"][0], c["px"][1], float(r_exp_px or c["r_px"]))
            ok, why = validate(c) if validate is not None else (True, "ok")
            examined.append((c, ok, why))

    rel_cache = {}

    def rel(i, j, V):
        """(why from V[i]'s side vs V[j], near, depth conflict, the pair gate's own why) for
        two validated discs; the depth conflict is the pair gate's whatever their brightness
        (a much brighter or much dimmer disc at another depth still makes the choice
        ambiguous); `why` adds the brightness rule to the pair gate's."""
        if (i, j) not in rel_cache:
            a, b = V[i], V[j]
            if _overlap(a, b):
                w = "overlaps a brighter disc" if j < i else ""
                rel_cache[(i, j)] = (w, True, False, w)
            else:
                why, near, depth = pair(a, b)
                gwhy = why
                if not why and not hint:
                    if a["flood_level"] <= prm.min_brightness_frac * b["flood_level"]:
                        why = f"dimmer than {prm.min_brightness_frac:g} x the brightest face"
                    elif b["flood_level"] <= prm.min_brightness_frac * a["flood_level"]:
                        why = (f"over {1 / max(prm.min_brightness_frac, 1e-9):.3g} x as bright "
                               "as the faces")
                rel_cache[(i, j)] = (why, near, depth, gwhy)
        return rel_cache[(i, j)]

    def compatible(i, j, V):
        return not _overlap(V[i], V[j]) and not rel(i, j, V)[0] and not rel(j, i, V)[0]

    def fits_with(i, j, V):
        """compatible but for the brightness rule: the pair gates pass, no overlap."""
        return not _overlap(V[i], V[j]) and not rel(i, j, V)[3] and not rel(j, i, V)[3]

    def connected(sel, V):
        if len(sel) < 2:
            return True
        seen, todo = {sel[0]}, [sel[0]]
        while todo:
            i = todo.pop()
            for j in sel:
                if j not in seen and rel(i, j, V)[1]:
                    seen.add(j)
                    todo.append(j)
        return len(seen) == len(sel)

    split = []                                   # pairwise fine, but no neighbour chain

    def search(V, forced=None):
        """The first (brightest) feasible set, and the largest pairwise-compatible partial
        (fewer than n_cells)."""
        order = [i for i in range(len(V)) if i != forced]
        best = [list([forced] if forced is not None else [])]
        steps = [0]

        def rec(start, sel):
            steps[0] += 1
            if steps[0] > MAX_SEARCH_STEPS:
                raise StationError(f"too many candidate end-face discs ({len(V)} pass the per-disc "
                                   "checks) to choose from: pass cell_px_hint or r_px_range, or "
                                   "scan a tighter zone")
            if len(best[0]) < len(sel) < n:
                best[0] = list(sel)
            if len(sel) == n:
                if connected(sel, V):
                    return list(sel)
                if not split:
                    split.append(list(sel))
                return None
            for p in range(start, len(order)):
                i = order[p]
                if all(compatible(i, j, V) for j in sel):
                    hit = rec(p + 1, sel + [i])
                    if hit is not None:
                        return hit
            return None
        return rec(0, [forced] if forced is not None else []), best[0]

    examine(cands[:MAX_EXAMINED])
    V = [c for c, ok, _ in examined if ok]
    chosen, partial = search(V)
    if chosen is None and len(cands) > MAX_EXAMINED:
        examine(cands[MAX_EXAMINED:])
        V = [c for c, ok, _ in examined if ok]
        rel_cache.clear()
        split.clear()
        chosen, partial = search(V)
    if chosen is None and split:
        D = prm.face_diameter_mm
        raise StationError(
            f"{n} end-face discs pass every check (at camera px "
            + ", ".join(f"({V[i]['hough_px'][0]:.0f}, {V[i]['hough_px'][1]:.0f})" for i in split[0])
            + f"), but they split into groups more than {prm.max_pair_spacing_diam:g} x the "
            f"diameter ({prm.max_pair_spacing_diam * D:.0f} mm) apart: raise "
            "max_pair_spacing_diam, or measure each group in its own scan zone")
    if chosen is not None:
        def at(s):
            return ", ".join(f"({V[i]['hough_px'][0]:.0f}, {V[i]['hough_px'][1]:.0f})"
                             for i in sorted(s))
        lo_set = min(V[c]["flood_level"] for c in chosen)
        out_ = [v for v in range(len(V))
                if v not in chosen and not any(_overlap(V[v], V[c]) for c in chosen)]
        deep = [v for v in out_ if any(rel(v, c, V)[2] for c in chosen)]
        D = prm.face_diameter_mm
        bound = (f"{prm.max_pair_spacing_diam:g} x the diameter "
                 f"({prm.max_pair_spacing_diam * D:.0f} mm)")
        # left-out discs that the pair gates let go with every face of the set but that have
        # no face of it within the bound: elsewhere, at the set's depth
        aloof = [v for v in out_
                 if all(fits_with(v, c, V) and not rel(v, c, V)[1] for c in chosen)]
        for v in out_:
            depth = [c for c in chosen if rel(v, c, V)[2]]
            other = [c for c in chosen if not compatible(v, c, V)]
            near = [c for c in chosen if rel(v, c, V)[1]]
            far = [c for c in chosen if not rel(v, c, V)[1]]
            bright = V[v]["flood_level"] >= lo_set and not hint
            if near and (far or n > 1) and all(fits_with(v, c, V) for c in chosen):
                # it goes with the set: it continues its row (within the bound of one face,
                # past it of another), or - n_cells > 1 - sits within the bound among its
                # faces. The set and it hold more than n_cells faces that go together, and
                # only brightness chose among them (a bright background disc beyond the end of
                # the row, or beside the faces, must not push a real face out silently). One
                # face (n_cells = 1) is the brightest disc of a compact group, as documented.
                where = (f"continues their row (within {bound} of the face at {at(near[:1])}, "
                         f"not of the one at {at(far[:1])})" if far else
                         f"goes with them, within {bound} of the face at {at(near[:1])}")
                if not other:
                    raise StationError(
                        f"ambiguous end faces: the discs at camera px {at(chosen)} pass every "
                        f"check, and so does the disc at {at([v])}, which {where}: more "
                        f"end-face discs together than n_cells = {n} - which are the part? Pass "
                        "cell_px_hint, or scan a zone that leaves the background out")
                # ... and only the brightness rule keeps it out (dimmer than half of one face
                # of the set, or the reverse): when it makes a full set with other discs, that
                # set is as plausible - brightness relative to one disc (a bright background
                # disc) must not decide which faces are the part
                alt = search(V, forced=v)[0]
                if alt is not None:
                    c = other[0]
                    dim = rel(v, c, V)[0] or rel(c, v, V)[0]
                    raise StationError(
                        f"ambiguous end faces: the discs at camera px {at(chosen)} and those at "
                        f"{at(alt)} each pass every check; the disc at {at([v])} {where} and is "
                        f"kept out of the first by brightness alone ({dim}, the disc at "
                        f"{at([c])}): which are the part? Pass cell_px_hint, or scan a zone that "
                        "leaves the background out")
            # a validated disc left out that goes with another left-out disc at its depth (a
            # partial set there: the part with a face missing?)
            group = depth and any(w != v and compatible(v, w, V) for w in deep)
            # ... or, at the set's depth but with no face of it within the bound, goes with
            # another such disc within the bound (a partial set elsewhere: the same question)
            mates = ([w for w in aloof if w != v and compatible(v, w, V) and rel(v, w, V)[1]]
                     if v in aloof else [])
            # no face of the set within the bound of it: a separate group (judged as a disc
            # that does not go with the set)
            apart = not other and not near
            if not ((depth and (bright or group)) or ((other or apart) and bright) or mates):
                alt = search(V, forced=v)[0] if depth else None
                if alt is None:
                    continue                     # dimmer, alone, no full set of its own
            else:
                alt = search(V, forced=v)[0]
            c = (depth or other or far)[0]
            why = (rel(v, c, V)[0] or rel(c, v, V)[0]
                   or f"no face of the set within {bound} of the disc at {at([v])}")
            what = (f"the discs at camera px {at(chosen)} and those at {at(alt)} each pass "
                    "every check" if alt is not None else
                    f"the discs at camera px {at(chosen)} pass every check, and so do the "
                    f"discs at {at([v] + mates)}, a group of their own" if mates else
                    f"the discs at camera px {at(chosen)} pass every check, and so does the "
                    f"disc at {at([v])}")
            raise StationError(
                f"ambiguous end faces: {what}, but they do not go together ({why}): which is "
                "the part - or is a face of it missing? Pass cell_px_hint, or scan a zone "
                "that leaves the background out")
    sel = chosen if chosen is not None else partial
    faces = [V[i] for i in sel]
    lo = min((c["flood_level"] for c in faces), default=np.inf)
    rejected, unfit = [], []                     # pair-gate conflicts first: they decide
    for c, ok, why in examined:
        if ok:
            i = next(k for k, v in enumerate(V) if v is c)
            if i in sel:
                continue
            bad = [rel(i, j, V) for j in sel if not compatible(i, j, V)]
            if any(_overlap(c, V[j]) for j in sel):
                continue                                 # another Hough circle on a face
            if not bad and sel and not any(rel(i, j, V)[1] for j in sel):
                D = prm.face_diameter_mm
                bad = [(f"no face within {prm.max_pair_spacing_diam:g} x the diameter "
                        f"({prm.max_pair_spacing_diam * D:.0f} mm) of it", False, False)]
            if not bad:
                continue                                 # a further face beyond n_cells
            if chosen is not None and c["flood_level"] < lo and not any(b[2] for b in bad) and not hint:
                continue                                 # dimmer than the faces: not needed
            rejected.append({"hough_px": c["hough_px"], "why": "; ".join(dict.fromkeys(
                b[0] for b in bad if b[0]))})
        else:
            if chosen is not None and not hint and (c["flood_level"] < lo
                                                    or any(_overlap(c, f) for f in faces)):
                continue                                 # never needed / a face's duplicate
            unfit.append({"hough_px": c["hough_px"], "why": why})
    return faces, rejected + unfit


def find_cells(d: np.ndarray, prm: StationParams, r_exp_px: Optional[float] = None,
               roi=None, validate=None, pair=None) -> tuple:
    """Face discs on the flood difference image d (0..255) -> ([{px, r_px, hough_px, edge_px,
    name, ...}] in C1..Cn order, [rejected candidates]). validate(cell) -> (ok, why) (the depth
    check: a flat face) sets aside bright discs that are not faces. Without pair (strict off):
    candidates are taken brightest first, non-overlapping, and a disc dimmer than
    min_brightness_frac x the brightest accepted one is not a face of the same kind. With
    pair(a, b) -> (why, near, depth_conflict) (the strict pair gates): the face set is chosen
    among all validated discs (_select_cells), never anchored on the first one accepted."""
    import cv2
    g = cv2.GaussianBlur(np.clip(d, 0, 255).astype(np.uint8), (5, 5), 1.5)
    H, W = g.shape
    if prm.r_px_range:
        rmin, rmax = (int(round(v)) for v in prm.r_px_range)
    elif r_exp_px:
        rmin, rmax = int(np.floor(0.75 * r_exp_px)), int(np.ceil(1.25 * r_exp_px))
    else:
        rmin, rmax = 22, 34
    cands = []
    if prm.cell_px_hint:
        for sx, sy in prm.cell_px_hint:
            cands.append((0.0, float(sx), float(sy), float(r_exp_px or (rmin + rmax) / 2)))
    else:
        cir = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1, minDist=prm.hough_min_dist_px,
                               param1=prm.hough_param1, param2=prm.hough_param2,
                               minRadius=rmin, maxRadius=rmax)
        for cx, cy, r in (cir[0] if cir is not None else []):
            if roi is not None and not (roi[0] <= cx <= roi[2] and roi[1] <= cy <= roi[3]):
                continue
            x0, x1 = max(int(cx - r), 0), min(int(cx + r) + 2, W)
            y0, y1 = max(int(cy - r), 0), min(int(cy + r) + 2, H)
            yy, xx = np.mgrid[y0:y1, x0:x1]
            ins = (xx - cx) ** 2 + (yy - cy) ** 2 < (0.8 * r) ** 2
            if ins.any():
                cands.append((float(d[y0:y1, x0:x1][ins].mean()), float(cx), float(cy), float(r)))
        cands.sort(reverse=True)
        if not cands:
            raise StationError(f"no face disc (radius {rmin}..{rmax} px) found on the flood")
    if pair is not None:
        out, rejected = _select_cells(cands, g, d, prm, r_exp_px, validate, pair)
    else:
        out, rejected, top = [], [], None
        for m, cx, cy, r in cands:
            if len(out) == prm.n_cells:
                break
            if any(np.hypot(cx - c["hough_px"][0], cy - c["hough_px"][1]) <= r + c["hough_px"][2]
                   for c in out):
                continue                                     # overlaps an accepted disc
            if top is not None and m <= prm.min_brightness_frac * top:
                rejected.append({"hough_px": [cx, cy, r], "why": f"dimmer than {prm.min_brightness_frac:g}"
                                                                  " x the brightest face"})
                continue
            c = _refine(g, cx, cy, r, r_exp_px)
            c.update({"hough_px": [cx, cy, r], "flood_level": m})
            c["edge_px"] = rim_edges(d, c["px"][0], c["px"][1], float(r_exp_px or c["r_px"]))
            if validate is not None:
                ok, why = validate(c)
                if not ok:
                    rejected.append({"hough_px": [cx, cy, r], "why": why})
                    continue
            if top is None and not prm.cell_px_hint:
                top = m
            out.append(c)
    if len(out) < prm.n_cells:
        why = "; ".join(f"({x['hough_px'][0]:.0f}, {x['hough_px'][1]:.0f}): {x['why']}"
                        for x in rejected[:5])
        if len(rejected) > 5:
            why += f"; and {len(rejected) - 5} more"
        raise StationError(f"found {len(out)} face disc(s), expected {prm.n_cells} (radius "
                           f"{rmin}..{rmax} px){'; set aside: ' + why if why else ''} - pass "
                           "cell_px_hint or r_px_range, or check the flood")
    order = prm.order
    if order == "auto":
        P = np.array([c["px"] for c in out])
        order = "rows" if len(out) < 2 or np.ptp(P[:, 1]) >= np.ptp(P[:, 0]) else "cols"
    key = (lambda c: (c["px"][1], c["px"][0])) if order == "rows" else (lambda c: (c["px"][0], c["px"][1]))
    out.sort(key=key)
    for i, c in enumerate(out):
        c["name"] = f"C{i + 1}"
    return out, rejected


# ------------------------------------------------------------------ 2.-4. faces and the frame

def face_fit(X: np.ndarray, cell: dict, K, dist, disc_px: Optional[float] = None,
             shrink_px: float = 3.0, centre_method: str = "rim_circle") -> dict:
    H, W = X.shape[:2]
    cx, cy = cell["px"]
    r = float(disc_px or cell["r_px"])
    x0, x1 = max(int(cx - r - 2), 0), min(int(cx + r + 3), W)
    y0, y1 = max(int(cy - r - 2), 0), min(int(cy + r + 3), H)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    ins = (xx - cx) ** 2 + (yy - cy) ** 2 < (r - shrink_px) ** 2
    Q = X[y0:y1, x0:x1][ins]
    n_disc = int(ins.sum())
    try:
        c, n, keep, P = fit_plane(Q)
    except StationError:
        raise StationError(f"end face {cell.get('name', '?')}: too few measured points on its face "
                           f"({int(np.all(np.isfinite(Q), 1).sum())} of {n_disc} disc px) - is "
                           "the face inside the scanned zone?") from None
    res = (P - c) @ n
    ray = camera_rays(K, dist, [[cx, cy]])[0]
    C = ray * ((c @ n) / (ray @ n))
    method, outline = "ellipse_ray", None
    if centre_method == "rim_circle":
        fit = circle_in_plane(np.asarray(cell.get("edge_px", np.zeros((0, 3)))), c, n, K, dist)
        if fit is not None and fit[3] >= 24 and abs(np.linalg.norm(fit[0] - C)) < 2.0:
            C, method = fit[0], "rim_circle"
            outline = {"diameter_mm": 2 * fit[1], "rms_mm": fit[2], "n_edge_px": fit[3]}
    elif centre_method != "ellipse_ray":
        raise ValueError("centre_method must be 'rim_circle' or 'ellipse_ray'")
    rr = np.linalg.norm((P - C) - np.outer((P - C) @ n, n), axis=1)
    return {"centre": C, "normal": n, "n_pts": int(len(P)), "n_disc_px": n_disc,
            "coverage": float(len(P) / max(n_disc, 1)),
            "flat_rms_mm": float(res[keep].std()), "flat_rms_all_mm": float(res.std()),
            "flat_p2_p98_mm": float(np.ptp(np.percentile(res, [2, 98]))),
            "flat_rms_r9_mm": float(res[rr < 9].std()) if (rr < 9).sum() > 3 else None,
            "disc_px": r, "centre_method": method, "outline": outline, "plane_c": c,
            "centre_ellipse_ray": C if method == "ellipse_ray" else ray * ((c @ n) / (ray @ n))}


def surround_same_frac(X: np.ndarray, cell: dict, f: dict, band=(1.2, 1.6),
                       tol_mm: float = 1.5) -> float:
    """Fraction of the ring around a face disc (camera px, band x r) whose points lie within
    tol_mm of the face plane: ~0 for a face standing proud of its surroundings, ~1 for a flat
    patch of a wall (undecoded px count as not continuing: shadows sit next to faces)."""
    H, W = X.shape[:2]
    cx, cy = cell["px"]
    r = float(cell["r_px"])
    R = band[1] * r
    x0, x1 = max(int(cx - R - 1), 0), min(int(cx + R + 2), W)
    y0, y1 = max(int(cy - R - 1), 0), min(int(cy + R + 2), H)
    yy, xx = np.mgrid[y0:y1, x0:x1]
    rho = np.hypot(xx - cx, yy - cy)
    ring = (rho >= band[0] * r) & (rho <= R)
    if not ring.any():
        return 0.0
    h = (X[y0:y1, x0:x1][ring] - f["plane_c"]) @ f["normal"]
    return float(np.sum(np.isfinite(h) & (np.abs(h) < tol_mm)) / ring.sum())


def _pct(frac: float) -> int:
    """A measured fraction as whole percent, rounded DOWN: a disc refused just below a limit
    never reads as being at it ('70 % (< 70 %)')."""
    return int(np.floor(100.0 * float(frac) + 1e-9))


def _over(v: float, lim: float) -> str:
    """v (> lim) with as many decimals as it takes not to read as lim ('0.30 > 0.3'); the
    exact value when no rounding does."""
    for nd in range(2, 10):
        if float(f"{v:.{nd}f}") > lim:
            return f"{v:.{nd}f}"
    return repr(float(v))


def strict_disc_gate(f: dict, prm: StationParams) -> list:
    """The strict per-disc reasons of a fitted face f ([] = passes): its flood outline is
    ragged, or too little of it is measured - a disc measured below strict_min_coverage passes
    only with a clean flood rim circle (rms <= strict_partial_outline_rms_mm): the rim is what
    shows an end face whose decode dropped out, a background patch has none (or a rough one)."""
    why = []
    rms = (f.get("outline") or {}).get("rms_mm")
    if rms is not None and rms > prm.max_outline_rms_mm:
        why.append(f"ragged outline (rim circle rms {_over(rms, prm.max_outline_rms_mm)} mm > "
                   f"{prm.max_outline_rms_mm:g})")
    if f["coverage"] < prm.strict_min_coverage and (
            rms is None or rms > prm.strict_partial_outline_rms_mm):
        rim = ("no rim circle" if rms is None else
               f"rim circle rms {_over(rms, prm.strict_partial_outline_rms_mm)} mm > "
               f"{prm.strict_partial_outline_rms_mm:g}")
        why.append(f"only {_pct(f['coverage'])} % of the disc measured "
                   f"(< {100 * prm.strict_min_coverage:g} %) and {rim}: not shown to be an "
                   "end face")
    return why


def strict_pair(f: dict, o: dict, prm: StationParams) -> tuple:
    """The strict pair gate between two fitted faces -> (why from f's side, '' = compatible;
    near: o within max_pair_spacing_diam x D; depth_conflict). The height difference is taken
    along both faces' normals, the larger counts (symmetric)."""
    D = prm.face_diameter_mm
    dv = np.asarray(f["centre"], float) - np.asarray(o["centre"], float)
    dz = max((float(dv @ o["normal"]), float(dv @ f["normal"])), key=abs)
    sp = float(np.linalg.norm(dv))
    near = sp <= prm.max_pair_spacing_diam * D
    if abs(dz) > prm.max_pair_dz_diam * D:
        return (f"{dz:+.1f} mm in height from an accepted face (> {prm.max_pair_dz_diam:g} "
                "x the diameter: background?)", near, True)
    if sp < prm.min_pair_spacing_diam * D:
        return (f"{sp:.1f} mm from an accepted face (plausible {prm.min_pair_spacing_diam:g}"
                f"..{prm.max_pair_spacing_diam:g} x the diameter)", near, False)
    return "", near, False


def strict_gate(f: dict, others: list, prm: StationParams) -> str:
    """'' when the fitted face f passes the strict quality gates against the faces `others`,
    else every reason (joined): its flood outline is ragged, too little of it is measured, it
    sits implausibly far in height from (or overlaps) one of them, or none of them is within
    max_pair_spacing_diam x the diameter (a face needs a neighbour, not every face near)."""
    why = strict_disc_gate(f, prm)
    near, dist = False, []
    for o in others:
        w, nr, _ = strict_pair(f, o, prm)
        if w:
            why.append(w)
        near |= nr
        dist.append(float(np.linalg.norm(np.asarray(f["centre"], float) - o["centre"])))
    if others and not near and not any("in height" in w for w in why):
        why.append(f"{min(dist):.1f} mm from the nearest accepted face (plausible "
                   f"{prm.min_pair_spacing_diam:g}..{prm.max_pair_spacing_diam:g} x the diameter)")
    return "; ".join(why)


def common_frame(faces: list) -> tuple:
    """(O, n, e1, e2): n = normalised sum of the face normals, O = mean face centre."""
    n = np.sum([f["normal"] for f in faces], axis=0)
    n = n / np.linalg.norm(n)
    O = np.mean([f["centre"] for f in faces], axis=0)
    e1 = np.cross([0.0, 1.0, 0.0], n)
    e1 /= np.linalg.norm(e1)
    e1 *= np.sign(e1[0]) or 1.0
    e2 = np.cross(n, e1)
    e2 *= np.sign(e2[1]) or 1.0
    return O, n, e1, e2


def cell_pairs(faces: list, names: list, O, n) -> list:
    out = []
    for i in range(len(faces)):
        for j in range(i + 1, len(faces)):
            dv = faces[j]["centre"] - faces[i]["centre"]
            dn = float(dv @ n)
            out.append({"pair": f"{names[j]}-{names[i]}", "a": names[i], "b": names[j],
                        "height_diff_mm": dn,
                        "spacing_inplane_mm": float(np.linalg.norm(dv - dn * n)),
                        "spacing_3d_mm": float(np.linalg.norm(dv)),
                        "face_angle_deg": angle_deg(faces[i]["normal"], faces[j]["normal"])})
    return out


# ------------------------------------------------------------------ 5. the tube edge

def default_ct_roi(cells: list, shape, prm: StationParams) -> tuple:
    H, W = shape
    if prm.ct_roi:
        x0, y0, x1, y1 = (int(round(v)) for v in prm.ct_roi)
        return max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    ys = [c["px"][1] for c in cells]
    m = prm.ct_rows_margin_r * float(np.median([c["r_px"] for c in cells]))
    return 0, max(int(min(ys) - m), 0), W, min(int(max(ys) + m) + 1, H)


def _thin_runs(cand: np.ndarray, x0: int, x1: int, run) -> np.ndarray:
    """Per camera column (x0..x1) the vertical runs of cand whose length is run[0]..run[1] px
    (vectorised: every column padded with a False pixel at both ends, runs found on the
    column-major flattening, so no run crosses a column)."""
    lo, hi = int(run[0]), int(run[1])
    sub = cand[:, x0:x1]
    H, W = sub.shape
    a = np.zeros((W, H + 2), bool)
    a[:, 1:-1] = sub.T
    f = a.ravel()
    d = np.diff(f.astype(np.int8))
    starts = np.nonzero(d == 1)[0] + 1
    ends = np.nonzero(d == -1)[0] + 1
    L = ends - starts
    keep = (L >= lo) & (L <= hi)
    mark = np.zeros(f.size + 1, np.int32)
    np.add.at(mark, starts[keep], 1)
    np.add.at(mark, ends[keep], -1)
    m = (np.cumsum(mark)[:-1] > 0).reshape(W, H + 2)[:, 1:-1].T
    out = np.zeros_like(cand)
    out[:, x0:x1] = m
    return out


def _wide_components(m: np.ndarray, minw: int) -> np.ndarray:
    import cv2
    n_, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    keep = np.zeros_like(m)
    for k in range(1, n_):
        if st[k, cv2.CC_STAT_WIDTH] > minw:
            keep |= lab == k
    return keep


def ct_candidates(X, cells, faces, O, n, prm: StationParams, roi) -> tuple:
    """-> (candidate mask outside the cells, h = height along n vs the mean face plane)."""
    H, W = X.shape[:2]
    h = (X - O) @ n
    x0, y0, x1, y1 = roi
    box = np.zeros((H, W), bool)
    box[y0:y1, x0:x1] = True
    ok = np.isfinite(h) & box
    yy, xx = np.mgrid[0:H, 0:W]
    R0 = prm.face_diameter_mm / 2
    for f, c in zip(faces, cells):
        rel = X - f["centre"]
        ax = rel @ f["normal"]
        rad = np.linalg.norm(rel - ax[..., None] * f["normal"], axis=-1)
        ok &= ~((rad < R0 + prm.ct_body_margin_mm) & (ax < 1.0))     # face + side wall
        ok &= ~((xx - c["px"][0]) ** 2 + (yy - c["px"][1]) ** 2
                < (c["r_px"] + prm.ct_disc_margin_px) ** 2)
    return ok, h


def auto_band(ok, h, prm: StationParams, roi) -> Optional[list]:
    """The tube edge height band: a window 2 x ct_band_halfwidth_mm wide slid over the search band
    (0.5 mm steps); the window whose thin runs (wider than min_width_px) cover the most camera
    columns wins (the ribbon's edge seen end-on: its visible side, further back, makes the runs
    too long in a wider window); the band is then centred on the median height of those runs."""
    lo, hi = (float(v) for v in prm.ct_band_search_mm)
    hw = float(prm.ct_band_halfwidth_mm)
    x0, y0, x1, y1 = roi
    okr, hr = ok[y0:y1], h[y0:y1]
    best, best_thin = 0, None
    for c in np.arange(lo + hw, max(hi - hw, lo + hw) + 1e-9, 0.5):
        with np.errstate(invalid="ignore"):
            cand = okr & (hr > c - hw) & (hr < c + hw)
        thin = _wide_components(_thin_runs(cand, x0, x1, prm.ct_run_px), prm.ct_min_width_px)
        cols = int(thin.any(0).sum())
        if cols > best:
            best, best_thin = cols, thin
    if best_thin is None or best_thin.sum() < 20:
        return None
    c0 = float(np.median(hr[best_thin]))
    return [c0 - hw, min(c0 + hw, hi)]


def ct_mask(X, cells, faces, O, n, prm: StationParams) -> dict:
    roi = default_ct_roi(cells, X.shape[:2], prm)
    ok, h = ct_candidates(X, cells, faces, O, n, prm, roi)
    band, band_src = prm.ct_band_mm, "given"
    if not band:
        band, band_src = auto_band(ok, h, prm, roi), "auto"
        if band is None:
            return {"mask": np.zeros(h.shape, bool), "h": h, "roi": roi, "band_mm": None,
                    "band_source": band_src}
    cand = ok & (h > band[0]) & (h < band[1])
    m = _wide_components(_thin_runs(cand, roi[0], roi[2], prm.ct_run_px), prm.ct_min_width_px)
    return {"mask": m, "h": h, "roi": roi, "band_mm": [float(band[0]), float(band[1])],
            "band_source": band_src}


# ------------------------------------------------------------------ 6. trapezoid wave + zones

def trap(s, p, ph, A, f, c, k):
    """Trapezoid wave: triangle -1..1 (x = 0 bottom plateau centre, x = 0.5 top) clipped at
    1 - f -> plateau fraction f; + offset c + slope k s."""
    x = ((s - ph) / p) % 1.0
    tri = np.where(x < 0.5, 4 * x - 1, 3 - 4 * x)
    lim = 1 - f
    return c + k * s + A * (np.clip(tri / lim, -1, 1) if lim > 1e-3 else np.sign(tri))


def centreline(X, ct, O, n, e1, e2) -> dict:
    """Per camera column of the CT mask: the median point -> s (along e1), w (along e2), h."""
    pts, cols = [], []
    for j in np.nonzero(ct.any(0))[0]:
        ys = np.nonzero(ct[:, j])[0]
        Q = X[ys, j]
        Q = Q[np.all(np.isfinite(Q), 1)]
        if len(Q):
            pts.append(np.median(Q, 0))
            cols.append((j, float(np.median(ys))))
    if not pts:
        return {"s": np.zeros(0), "w": np.zeros(0), "h": np.zeros(0), "P": np.zeros((0, 3)),
                "cols": np.zeros((0, 2))}
    Pc = np.array(pts)
    s, w, h = (Pc - O) @ e1, (Pc - O) @ e2, (Pc - O) @ n
    o = np.argsort(s)
    return {"s": s[o], "w": w[o], "h": h[o], "P": Pc[o], "cols": np.array(cols)[o]}


def _trap_grid(s, w, p_range, n_ph: int = 16, fracs=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
               step: float = 0.25, top: int = 8) -> list:
    """Linear least-squares scan over (pitch, phase, plateau fraction): for each, offset,
    slope and amplitude are linear -> the residual in closed form. -> the `top` best starts
    [p, ph, A, f, c, k], distinct in pitch / phase."""
    s = np.asarray(s, float)
    w = np.asarray(w, float)
    B = np.c_[np.ones_like(s), s - s.mean()]
    Q, _ = np.linalg.qr(B)
    wp = w - Q @ (Q.T @ w)
    fr = np.asarray(fracs, float)
    rows = []
    for p in np.arange(float(p_range[0]), float(p_range[1]) + 1e-9, step):
        ph = np.linspace(0, p, n_ph, endpoint=False)
        x = ((s[None, :] - ph[:, None]) / p) % 1.0                         # (n_ph, N)
        tri = np.where(x < 0.5, 4 * x - 1, 3 - 4 * x)
        lim = np.maximum(1 - fr, 1e-3)[:, None, None]
        T = np.clip(tri[None] / lim, -1, 1).reshape(-1, len(s))            # (n_f*n_ph, N)
        Tp = T - (T @ Q) @ Q.T
        tt = np.einsum("ij,ij->i", Tp, Tp)
        tw = Tp @ wp
        A = tw / np.maximum(tt, 1e-12)
        ss = wp @ wp - tw * A
        ss = np.where(A > 0.05, ss, np.inf)
        for i in np.argsort(ss)[:top]:
            fi, pi = divmod(int(i), n_ph)
            rows.append((float(ss[i]), p, float(ph[pi]), float(A[i]), float(fr[fi])))
    rows.sort()
    out = []
    for ss_, p, ph, A, f in rows:
        if all(abs(p - o[0]) > 0.6 or abs(((ph - o[1] + p / 2) % p) - p / 2) > p / 8 for o in out):
            c = float(np.mean(w - A * np.clip(np.where(((s - ph) / p) % 1 < 0.5, 4 * (((s - ph) / p) % 1) - 1,
                                                       3 - 4 * (((s - ph) / p) % 1)) / max(1 - f, 1e-3), -1, 1)))
            out.append([p, ph, A, min(max(f, 0.0), 0.9), c, 0.0])
        if len(out) == top:
            break
    return out


def fit_trap(s, w, p_range=(15.0, 40.0)) -> dict:
    """Trapezoid wave w(s): a closed-form linear scan over pitch / phase / plateau fraction,
    then soft-L1 least squares (f_scale 0.3 mm) from the best starts."""
    from scipy.optimize import least_squares
    if len(s) < 12:
        raise StationError(f"only {len(s)} tube edge columns: too few for the trapezoid wave")
    lo_b = [10, -100, 0.1, 0.0, -50, -1]
    hi_b = [80, 100, 15, 0.9, 50, 1]
    best = None
    for x0 in _trap_grid(s, w, p_range):
        x0 = np.clip(x0, np.array(lo_b) + 1e-6, np.array(hi_b) - 1e-6)
        r = least_squares(lambda q: trap(s, *q) - w, x0, bounds=(lo_b, hi_b),
                          loss="soft_l1", f_scale=0.3)
        if best is None or r.cost < best.cost:
            best = r
    if best is None:
        raise StationError(f"no trapezoid-wave start in the pitch range {list(p_range)} mm")
    p, ph, A, f, c0, k = best.x
    res = trap(s, *best.x) - w
    ph = ph % p
    return {"pitch_mm": float(p), "phase_mm": float(ph), "amp_mm": float(A),
            "plateau_frac": float(f), "offset_mm": float(c0), "slope": float(k),
            "fit_rms_mm": float(np.sqrt(np.mean(res ** 2))),
            "fit_rms_robust_mm": float(1.4826 * np.median(np.abs(res))), "n_cols": int(len(s)),
            "s_range_mm": [float(s.min()), float(s.max())]}


def zone_index(s, p, ph):
    """One zone = a plateau + the two half-ramps beside it (boundaries at mid-ramp)."""
    return np.floor(2 * (np.asarray(s) - ph) / p - 0.5).astype(int)


def zone_centre(k, p, ph):
    return ph + p * (k + 1) / 2.0


def zones(X, ct, O, n, e1, model: dict, line: dict, prm: StationParams,
          fixed: Optional[dict] = None) -> list:
    """Per part-fixed zone k: coverage, fit residual, camera position, edge height stats.
    fixed = {k: label} (a golden's numbering) keeps exactly those zones; else they are chosen
    (coverage, residual, end skips) and numbered T1..Tn left -> right."""
    p, ph = model["pitch_mm"], model["phase_mm"]
    v, u = np.nonzero(ct)
    P = X[v, u]
    okp = np.all(np.isfinite(P), 1)
    v, u, P = v[okp], u[okp], P[okp]
    s_px = (P - O) @ e1
    h_px = (P - O) @ n
    k_px = zone_index(s_px, p, ph)
    res = trap(line["s"], p, ph, model["amp_mm"], model["plateau_frac"], model["offset_mm"],
               model["slope"]) - line["w"] if len(line["s"]) else np.zeros(0)
    k_col = zone_index(line["s"], p, ph) if len(line["s"]) else np.zeros(0, int)
    out = []
    ks = sorted(set(int(k) for k in np.unique(k_px)) | set(int(k) for k in (fixed or {})))
    for k in ks:
        sel = k_px == k
        c0 = zone_centre(k, p, ph)
        half = p / 4
        lo, hi = c0 - half, c0 + half
        cov = (max(0.0, (min(hi, float(s_px[sel].max())) - max(lo, float(s_px[sel].min()))) / (hi - lo))
               if sel.any() else 0.0)
        z = {"k": int(k), "s_centre_mm": float(c0), "s_lo_mm": float(lo), "s_hi_mm": float(hi),
             "coverage": float(cov), "n_px": int(sel.sum()),
             "fit_res_rms_mm": (float(np.sqrt(np.mean(res[k_col == k] ** 2)))
                                if (k_col == k).any() else None),
             "cam_uv": ([float(np.median(u[sel])), float(np.median(v[sel]))] if sel.any() else None),
             "plateau": "down (image down)" if k % 2 == 0 else "up (image up)"}
        if sel.sum() >= max(prm.zone_min_px, 1):
            hh = h_px[sel]
            z.update({"median_mm": float(np.median(hh)), "p5_mm": float(np.percentile(hh, 5)),
                      "p95_mm": float(np.percentile(hh, 95))})
        out.append(z)
    if fixed is not None:
        keep = {int(k): lab for k, lab in fixed.items()}
        out = [z for z in out if z["k"] in keep]
        for z in out:
            z["label"] = keep[z["k"]]
            z["included"] = True
        return out
    cand = []
    for z in out:
        why = None
        if "median_mm" not in z:
            why = f"only {z['n_px']} edge px"
        elif z["coverage"] < prm.zone_min_coverage:
            why = f"partial (coverage {z['coverage']:.2f})"
        elif (prm.zone_max_fit_rms_mm is not None and z["fit_res_rms_mm"] is not None
              and z["fit_res_rms_mm"] > prm.zone_max_fit_rms_mm):
            why = (f"edge does not follow the wave (fit residual {z['fit_res_rms_mm']:.2f} mm > "
                   f"{prm.zone_max_fit_rms_mm:g}): a lead-in or another part")
        z["included"] = why is None
        if why:
            z["excluded_reason"] = why
        else:
            cand.append(z)
    a, b = (int(v) for v in (prm.zone_skip or [0, 0]))
    for z in cand[:a] + (cand[len(cand) - b:] if b else []):
        z["included"] = False
        z["excluded_reason"] = "skipped at the end (zone_skip)"
    num = 0
    for z in out:
        if z["included"]:
            num += 1
            z["label"] = f"T{num}"
    return out


# ------------------------------------------------------------------ the whole measurement

def measure(X: np.ndarray, flood_d: np.ndarray, K, dist, prm: StationParams,
            zone_model: Optional[dict] = None, zone_labels: Optional[dict] = None) -> dict:
    """X (H, W, 3) mm (NaN = no point), flood_d (H, W) 0..255 -> the station result (arrays
    included under 'arrays'). zone_model = {pitch_mm, phase_mm, ...} fixes the zones (a golden's);
    zone_labels = {k: label} its numbering."""
    if X.shape[:2] != flood_d.shape:
        raise StationError(f"flood {flood_d.shape} and scan {X.shape[:2]} differ in size")
    r_exp = expected_radius_px(X, K, prm.face_diameter_mm / 2)
    fits = {}

    def validate(c):
        if r_exp and abs(c["r_px"] - r_exp) > 0.25 * r_exp:
            return False, (f"its rim outline is {c['r_px']:.1f} px in radius, a "
                           f"{prm.face_diameter_mm:g} mm face is ~{r_exp:.1f} px here")
        try:
            f = face_fit(X, c, K, dist, prm.face_disc_px, prm.face_shrink_px, prm.centre_method)
        except StationError as e:
            return False, str(e)
        if f["coverage"] < prm.min_face_coverage:
            return False, f"only {_pct(f['coverage'])} % of the disc measured"
        if f["flat_rms_mm"] > prm.max_face_rms_mm:
            return False, f"not a flat face (plane rms {f['flat_rms_mm']:.2f} mm)"
        same = surround_same_frac(X, c, f)
        if same > prm.max_surround_same_frac:
            return False, (f"not a proud face: {100 * same:.0f} % of its surround continues its "
                           "plane (a flat patch of a larger surface)")
        if prm.strict:
            why = strict_disc_gate(f, prm)
            if why:
                return False, "; ".join(why)
        f["surround_same_frac"] = same
        fits[id(c)] = f
        return True, "ok"

    def pair(a, b):
        return strict_pair(fits[id(a)], fits[id(b)], prm)

    # strict: the pair gates are judged over every validated disc (find_cells chooses the
    # set), not against whichever disc was accepted first
    cells, rejected = find_cells(flood_d, prm, r_exp, validate=validate,
                                 pair=pair if prm.strict else None)
    faces = [fits[id(c)] for c in cells]
    names = [c["name"] for c in cells]
    warnings = []
    # one face too: the same frame (e1 ~ image right, e2 ~ image down), so s, the zone index
    # and the T1..Tn numbering (left -> right) mean the same thing for n_cells = 1
    O, n, e1, e2 = common_frame(faces)
    for c, f in zip(cells, faces):
        f["height_mm"] = float((f["centre"] - O) @ n)
        f["tilt_vs_common_deg"] = angle_deg(f["normal"], n)
        if f["coverage"] < 0.7:
            warnings.append(f"{c['name']}: only {_pct(f['coverage'])} % of its face disc is "
                            "measured")
        # a face cut by the zone edge (its flood dark past the cut) or a poor outline moves the
        # centre sideways: the height is unaffected (the centre stays on the face plane), the
        # spacing is not - say so
        if prm.centre_method == "rim_circle" and f["centre_method"] != "rim_circle":
            warnings.append(f"{c['name']}: its flood rim edges gave no circle near the rim "
                            "ellipse: centre from the rim ellipse (is the face cut by the zone "
                            "edge?) - its spacing is less certain, its height is not affected")
        od = (f["outline"] or {}).get("diameter_mm")
        if od is not None and abs(od - prm.face_diameter_mm) > OUTLINE_WARN_MM:
            warnings.append(f"{c['name']}: its flood outline is {od:.2f} mm across (face "
                            f"{prm.face_diameter_mm:g} mm): cut by the zone edge or a poor outline "
                            "- its spacing is less certain, its height is not affected")
    pairs = cell_pairs(faces, names, O, n)
    if rejected:
        warnings.append(f"{len(rejected)} bright disc(s) set aside (e.g. at camera px "
                        f"({rejected[0]['hough_px'][0]:.0f}, {rejected[0]['hough_px'][1]:.0f}): "
                        f"{rejected[0]['why']})")
    out = {"cells": cells, "faces": faces, "names": names, "frame": {"O": O, "n": n, "e1": e1, "e2": e2},
           "pairs": pairs, "r_expected_px": r_exp, "warnings": warnings, "ct": None,
           "zones": [], "arrays": {}, "rejected_discs": rejected,
           "set_aside": [{"camera_px": [float(x["hough_px"][0]), float(x["hough_px"][1])],
                          "why": x["why"]} for x in rejected]}
    if not prm.ct or len(faces) < 1:
        return out
    ct = ct_mask(X, cells, faces, O, n, prm)
    out["arrays"]["ct_mask"] = ct["mask"]
    ctr = {"roi_px": list(ct["roi"]), "band_mm": ct["band_mm"], "band_source": ct["band_source"],
           "n_px": int(ct["mask"].sum())}
    out["ct"] = ctr
    if ct["mask"].sum() < 20:
        warnings.append("no tube edge found (no thin run a few mm behind the faces "
                        f"in camera px {ct['roi']}): tube not measured")
        return out
    line = centreline(X, ct["mask"], O, n, e1, e2)
    out["arrays"]["line"] = line
    cols = ct["mask"].any(0)
    ctr["cam_cols"] = [int(np.nonzero(cols)[0].min()), int(np.nonzero(cols)[0].max())]
    hv = ct["h"][ct["mask"]]
    ctr["edge_height_all"] = {"median_mm": float(np.median(hv)), "p5_mm": float(np.percentile(hv, 5)),
                              "p95_mm": float(np.percentile(hv, 95)), "n_px": int(len(hv))}
    try:
        fit = fit_trap(line["s"], line["w"], prm.trap_pitch_range_mm)
    except StationError as e:
        warnings.append(f"tube: {e}")
        return out
    ctr["trap_fit"] = fit
    model = dict(fit)
    if zone_model:
        model.update({k: float(zone_model[k]) for k in ("pitch_mm", "phase_mm", "amp_mm",
                                                        "plateau_frac", "offset_mm", "slope")
                      if k in zone_model})
        ctr["zone_model_source"] = "given (the golden's pitch + phase)"
    else:
        ctr["zone_model_source"] = "this scan's fit"
    ctr["zone_model"] = {k: model[k] for k in ("pitch_mm", "phase_mm", "amp_mm", "plateau_frac",
                                                "offset_mm", "slope")}
    zs = zones(X, ct["mask"], O, n, e1, model, line, prm, zone_labels)
    out["zones"] = zs
    inc = [z for z in zs if z.get("included") and "median_mm" in z]
    if inc:
        sel = np.zeros(ct["mask"].shape, bool)
        ks = {z["k"] for z in inc}
        v, u = np.nonzero(ct["mask"])
        P = X[v, u]
        good = np.all(np.isfinite(P), 1)
        kk = zone_index((P[good] - O) @ e1, model["pitch_mm"], model["phase_mm"])
        hh = ((P[good] - O) @ n)[np.isin(kk, list(ks))]
        ctr["edge_height_zones"] = {"median_mm": float(np.median(hh)), "n_px": int(len(hh)),
                                    "n_zones": len(inc)}
    return out

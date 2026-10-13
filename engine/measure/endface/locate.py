"""
Locate a circular end face in a metric scan (the bench prototype ring_check2, productised).

  1. face by DEPTH: in the region (the scan's lit zone, or an roi), the nearest
     depth-continuous surface: pixels within near_band_mm of the nearest POPULATED depth
     (the first depth with >= min_face_px points within populated_span_mm behind it: the
     prototype's 1st depth percentile, but independent of how much of the region the face
     fills), opened 3x3, connected components, holes / undecoded bands filled. A component must be
     FACE-SHAPED (round enough, with a rim against far background and a good fixed-radius
     circle fit); one that is not (a fixture edge nearer than the face) is set aside and the
     next depth layer is searched.
  2. face plane: SVD of the depth face (the prototype's plain fit), normal toward the camera;
     after the first circle fit it is refitted (refine_iters) on the depth-face points inside
     the circle only, MAD-trimmed: the depth face also holds the cell's side wall.
  3. rim: contour points of the depth face whose outward neighbourhood (far_dilate_px) holds
     FAR surface (> the nearest depth + far_gap_mm): the true silhouette against the
     background, never the edge of an undecoded band or of the cell's own side wall.
     PLANE SPLIT (neighbour_split): a neighbour only a few mm behind the face (a ribbed
     tube, the next cell's body) falls inside near_band_mm and merges with the face into one
     component whose outline is not the face's. The component's planes are found by
     sequential RANSAC (a face and a parallel ribbon 3 mm behind it come out as two planes);
     per plane the face = its pixels within face_band_mm (opened, largest part), and a rim
     point needs surface at least near_gap_mm BEHIND THAT PLANE in its outward neighbourhood
     (plane-relative, so a tilted face's own far half is never "behind"; 2 mm is far above
     the noise and a rim chamfer); the best-fitting face-shaped candidate is kept. It
     replaces the plain depth result when that failed, when its rim is one-sided (<
     split_below_arc_deg) and the split fits better per covered arc, or when the split finds
     split_rim_gain x the rim points at no worse a fit (a neighbour hid the background).
     FLOOD CIRCLE (when the depth rim test fails in every component): the face outline in a
     flood image (the scan's flood pair, or the decode's fringe modulation): Hough disc of the
     expected size nearest the depth face, sub-pixel rim edges back-projected onto the depth
     face's plane, circle fitted there.
  4. fixed-radius circle (the known face diameter) in the face plane, soft-L1 least
     squares (f_scale 0.3 mm) on the rim points, at R - half a pixel footprint (rim points
     are pixel centres just inside the silhouette; the rim arc is one-sided because the
     projector shadows the far side, so an uncorrected fit slides ~0.15 mm).
     SILHOUETTE LARGER THAN THE FACE: the face diameter is the bright flat's; when the rim
     around it (rounded, chamfered) returns depth points, the depth silhouette is the
     part's OUTER circle (a 19.5 mm flat on a 21.05 mm cylinder), and a 19.5 mm circle
     fitted to it slides until it touches one side - the centre lands (D_outer - D) / 2
     off, 0.8 mm there. So the rim circle is fitted at the silhouette's own radius: the
     one given (silhouette_diameter_mm), else - when the rim points cover at least
     silhouette_min_arc_deg and lie on a clean circle 2 .. 20 % larger than the face - the
     free-radius circle through them. The face stays the diameter given: the camera mask,
     the DMD ring and every radius derive from it; only the CENTRE comes from the
     silhouette. (A silhouette smaller than the face, or far larger, is not this: the
     fixed fit stands and its misfit says so.)
  5. the 3-D circle projected into the camera (K_cam, dist_cam): the camera face mask and
     its ellipse.
  6. camera -> DMD AFFINE fitted on the decoded face pixels (corr_x / corr_y of the scan;
     least squares, then 2 refits without rim pixels that decoded the background behind
     the face), residual reported; the DMD ring = the camera ellipse through the affine. The stereo
     calibration's projector model is NOT used for aiming (its row axis is known to be
     inconsistent on the bench; the column axis is fine and triangulates the depth).

Everything is plain numpy / OpenCV / scipy on arrays; verbs.py does the library side.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

FACE_DIAMETER_MM = 21.05            # the bench cells
AFFINE_WARN_PX = 0.5                # warn when the camera -> DMD affine misfits by more


@dataclass
class LocateParams:
    near_band_mm: float = 12.0      # depth face: within this of the nearest populated depth
    far_gap_mm: float = 30.0        # far background: beyond the nearest depth + this
    neighbour_split: bool = True    # when that fails: split the face from a near neighbour ...
    face_band_mm: float = 1.0       # ... face = component px within this of its robust plane
    near_gap_mm: float = 2.0        # ... rim = surface this far BEHIND the face plane beside it
    split_below_arc_deg: float = 150.0  # a depth rim this one-sided: keep the split if it
                                    # scores better (rim rms per covered arc)
    split_rim_gain: float = 1.5     # or if it finds this many times the rim points, no worse fit
    flood_fallback: bool = True     # then: the face outline in the flood (verbs pass the flood)
    min_flood_edges: int = 40       # flood rim edge points needed (of 180 rays)
    open_px: int = 3                # morphological opening of the near mask
    far_dilate_px: int = 7          # outward neighbourhood a rim point must see far surface in
    min_face_px: int = 30           # smallest component considered
    populated_span_mm: float = 2.0  # nearest depth with >= min_face_px points within this
    min_rim_points: int = 12
    min_rim_arc_deg: float = 60.0   # rim points must span at least this arc around the centre
    silhouette_diameter_mm: Optional[float] = None  # the part's OUTER diameter when its depth
                                    # silhouette is larger than the face (a flat inside a
                                    # rounded rim that decodes); None = found from the rim
    silhouette_auto: bool = True    # ... points when they cover silhouette_min_arc_deg and
    silhouette_min_arc_deg: float = 150.0   # lie on a clean circle this much larger than the
    silhouette_max_grow: float = 0.20       # face at most (and at least 2 % / 0.25 mm)
    min_aspect: float = 0.35        # minor / major spread of a face component (round-ish)
    max_rim_rms_mm: float = 1.0     # a worse fixed-radius fit is not this face
    f_scale_mm: float = 0.3         # soft-L1 scale of the circle fit
    rim_inset_px: float = 0.5       # rim pixel centres sit ~half a pixel inside the silhouette
    refine_iters: int = 1           # face-only plane refit + circle refit (0 = the prototype)
    refine_margin_mm: float = 1.0   # the refit uses depth-face points this far inside the rim
    max_layers: int = 12            # depth layers searched past face-less near objects
    circle_points: int = 360
    ring_points: int = 720

    def to_dict(self) -> dict:
        return asdict(self)


class FaceNotFound(ValueError):
    """No face-shaped nearest surface in the scan / region."""


def _region(mask: np.ndarray, roi) -> tuple:
    H, W = mask.shape
    if roi is not None:
        x0, y0, x1, y1 = (int(round(float(v))) for v in roi)
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
        if x1 - x0 < 8 or y1 - y0 < 8:
            raise ValueError(f"roi {roi} is empty or outside the {W}x{H} camera frame")
        return x0, y0, x1, y1
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise FaceNotFound("the scan has no measured points")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def face_axes(n: np.ndarray) -> tuple:
    """(e1, e2) of the face frame: e1 = n x y_cam (~ camera right), e2 = e1 x n (~ down)."""
    e1 = np.cross(n, [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(e1, n)


def plane_svd(P: np.ndarray) -> tuple:
    """(centroid, unit normal toward the camera) of points (N, 3), camera at the origin."""
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    n = vt[2]
    if n @ c > 0:
        n = -n
    return c, n


def nearest_populated(z: np.ndarray, min_count: int, span_mm: float) -> Optional[float]:
    """The smallest depth with >= min_count values in [depth, depth + span_mm] (isolated
    outliers nearer than the first real surface are skipped)."""
    zs = np.sort(np.asarray(z, np.float64).ravel())
    if len(zs) < min_count:
        return None
    cnt = np.searchsorted(zs, zs + span_mm, side="right") - np.arange(len(zs))
    i = int(np.argmax(cnt >= min_count))
    return float(zs[i]) if cnt[i] >= min_count else None


def _fill(comp: np.ndarray) -> np.ndarray:
    """Component with its holes / undecoded bands filled (external contours drawn filled)."""
    import cv2
    cs, _ = cv2.findContours(comp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = np.zeros(comp.shape, np.uint8)
    cv2.drawContours(out, cs, -1, 1, -1)
    return out.astype(bool)


def _circle_fit(ub: np.ndarray, radius: float, f_scale: float) -> tuple:
    from scipy.optimize import least_squares
    q = least_squares(lambda q_: np.hypot(ub[:, 0] - q_[0], ub[:, 1] - q_[1]) - radius,
                      ub.mean(0), loss="soft_l1", f_scale=f_scale).x
    res = np.hypot(ub[:, 0] - q[0], ub[:, 1] - q[1]) - radius
    return q, res


SILHOUETTE_MIN_GROW = 0.02          # a silhouette this much larger than the face (and
SILHOUETTE_MIN_GROW_MM = 0.25       # this many mm in radius) is the part's outer circle
SILHOUETTE_MAX_RMS_MM = 0.3         # ... when its points lie on that circle this well


def _free_circle(ub: np.ndarray, f_scale: float) -> Optional[tuple]:
    """Free-radius circle through rim points (algebraic start, soft-L1 refinement) ->
    (centre (2,), radius, residuals) or None."""
    from scipy.optimize import least_squares
    if len(ub) < 12:
        return None
    A = np.c_[2 * ub, np.ones(len(ub))]
    s = np.linalg.lstsq(A, (ub ** 2).sum(1), rcond=None)[0]
    r0 = float(np.sqrt(max(s[2] + s[0] ** 2 + s[1] ** 2, 0.0)))
    if not np.isfinite(r0) or r0 <= 0:
        return None
    x = least_squares(lambda p: np.hypot(ub[:, 0] - p[0], ub[:, 1] - p[1]) - p[2],
                      [s[0], s[1], r0], loss="soft_l1", f_scale=f_scale).x
    return x[:2], float(x[2]), np.hypot(ub[:, 0] - x[0], ub[:, 1] - x[1]) - float(x[2])


def _silhouette_radius(ub: np.ndarray, q: np.ndarray, res: np.ndarray, radius: float,
                       inset: float, prm: "LocateParams") -> Optional[float]:
    """The radius of the depth silhouette when it is a clean circle LARGER than the face
    (the rim beyond the flat returns depth points), else None: the rim points cover enough
    arc, their free-radius circle is 2 .. silhouette_max_grow larger than the face, fits
    them well, and clearly better than the face's own radius does."""
    if not prm.silhouette_auto or _arc_deg(ub, q) < prm.silhouette_min_arc_deg:
        return None
    free = _free_circle(ub, prm.f_scale_mm)
    if free is None:
        return None
    qf, rf, resf = free
    r_sil = rf + inset
    grow = r_sil / radius - 1.0
    rms_f = float(np.sqrt(np.mean(resf ** 2)))
    rms_0 = float(np.sqrt(np.mean(res ** 2)))
    if not (max(SILHOUETTE_MIN_GROW, SILHOUETTE_MIN_GROW_MM / radius) < grow
            <= prm.silhouette_max_grow):
        return None
    if rms_f > SILHOUETTE_MAX_RMS_MM or rms_f > 0.6 * rms_0 \
            or _arc_deg(ub, qf) < prm.silhouette_min_arc_deg:
        return None
    return float(r_sil)


def _arc_deg(ub: np.ndarray, q: np.ndarray) -> float:
    """Arc (deg) around q covered by the rim points: 360 minus the largest angular gap."""
    a = np.sort(np.degrees(np.arctan2(ub[:, 1] - q[1], ub[:, 0] - q[0])))
    if len(a) < 2:
        return 0.0
    gaps = np.diff(np.concatenate([a, [a[0] + 360.0]]))
    return float(360.0 - gaps.max())


def _ransac_planes(P: np.ndarray, band: float, n_planes: int = 3, min_pts: int = 30,
                   iters: int = 200, seed: int = 0) -> list:
    """Sequential RANSAC: up to n_planes planes (centroid, normal toward the camera), each the
    3-point plane with the most points within `band`, refitted (SVD) on them, its points then
    removed. Two parallel surfaces a few mm apart (a face and a ribbon behind it) come out as two
    planes, where one robust fit would settle between them."""
    rng = np.random.default_rng(seed)
    out = []
    Q = P
    for _ in range(n_planes):
        if len(Q) < max(min_pts, 3):
            break
        best, best_n = None, 0
        for _ in range(iters):
            a3 = Q[rng.choice(len(Q), 3, replace=False)]
            nv = np.cross(a3[1] - a3[0], a3[2] - a3[0])
            nn = np.linalg.norm(nv)
            if nn < 1e-9:
                continue
            nv /= nn
            cnt = int(np.sum(np.abs((Q - a3[0]) @ nv) < band))
            if cnt > best_n:
                best, best_n = (a3[0], nv), cnt
        if best is None or best_n < min_pts:
            break
        inl = np.abs((Q - best[0]) @ best[1]) < band
        c, n = plane_svd(Q[inl])
        inl = np.abs((Q - c) @ n) < band
        c, n = plane_svd(Q[inl])
        out.append((c, n, int(inl.sum())))
        Q = Q[~inl]
    return out


def _split_faces(comp, X3, prm: LocateParams) -> tuple:
    """Neighbour split: -> ([(face mask, plane-relative far mask)], reason). The component's
    planes (sequential RANSAC, nearest to the camera first); per plane its pixels within
    face_band_mm, opened, the largest part; far = surface >= near_gap_mm behind that plane."""
    import cv2
    m = _fill(comp)
    P = X3[m]
    P = P[np.all(np.isfinite(P), 1)]
    if len(P) < prm.min_face_px:
        return [], f"{len(P)} measured points"
    planes = _ransac_planes(P, prm.face_band_mm, min_pts=prm.min_face_px)
    if not planes:
        return [], "no plane"
    planes.sort(key=lambda q: -(q[0] @ q[1]))           # distance from the camera along n
    k = max(int(prm.open_px), 1)
    out = []
    for c0, n0, _ in planes:
        Hr = (X3 - c0) @ n0                              # NaN where not measured
        with np.errstate(invalid="ignore"):
            face = m & (np.abs(Hr) < prm.face_band_mm)
            far = np.isfinite(Hr) & (Hr < -prm.near_gap_mm)
        face = cv2.morphologyEx(face.astype(np.uint8), cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
        n_, lab, st, _ = cv2.connectedComponentsWithStats(face)
        if n_ < 2:
            continue
        big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        out.append((lab == big, far))
    return out, ("ok" if out else "no face-plane part")


def _rim_score(c: dict) -> float:
    """Lower is better: rim misfit per covered fraction of the circle."""
    return c["rim_rms_mm"] * 360.0 / max(c["rim_arc_deg"], 1.0)


def _split_try(comp, X3, Z, zmin, radius, prm: LocateParams) -> tuple:
    """-> (the split candidate that is the best face (_rim_score) | None, reason)."""
    cands, why = _split_faces(comp, X3, prm)
    whys = [why] if not cands else []
    best = None
    for face, far in cands:
        cand, w = _try_component(face, X3, Z, zmin, radius, prm, far=far)
        if cand is None:
            whys.append(w)
        elif best is None or _rim_score(cand) < _rim_score(best):
            best = cand
    return (best, "ok") if best is not None else (None, " / ".join(whys[:3]))


def _try_component(comp, X3, Z, zmin, radius, prm: LocateParams, far=None) -> tuple:
    """-> (candidate dict | None, reason). far: a precomputed far mask (the neighbour split's,
    plane-relative); None = far background beyond zmin + far_gap_mm."""
    import cv2
    m = _fill(comp)
    P = X3[m]
    P = P[np.all(np.isfinite(P), 1)]
    if len(P) < prm.min_face_px:
        return None, f"{len(P)} measured points"
    c, n = plane_svd(P)
    e1, e2 = face_axes(n)
    pu, pv = (P - c) @ e1, (P - c) @ e2
    ev = np.sort(np.linalg.eigvalsh(np.cov(np.stack([pu, pv]))))
    aspect = float(np.sqrt(max(ev[0], 0) / max(ev[1], 1e-12)))
    if aspect < prm.min_aspect:
        return None, f"not round (aspect {aspect:.2f} < {prm.min_aspect:g})"
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cs, key=cv2.contourArea)[:, 0, :]
    if far is None:
        far = np.isfinite(Z) & (Z > zmin + prm.far_gap_mm)
    k = int(prm.far_dilate_px)
    far_d = cv2.dilate(far.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
    rim_px = np.array([p for p in cnt if far_d[p[1], p[0]]]).reshape(-1, 2)
    Rm = X3[rim_px[:, 1], rim_px[:, 0]] if len(rim_px) else np.zeros((0, 3))
    ok = np.all(np.isfinite(Rm), 1)
    rim_px, Rm = rim_px[ok], Rm[ok]
    if len(Rm) < prm.min_rim_points:
        return None, (f"{len(Rm)} rim points against far background (< {prm.min_rim_points}): "
                      "no silhouette")
    # rim points are pixel CENTRES, on average half a pixel inside the silhouette: fit the
    # fixed circle through them at R - inset (else its centre slides away from the rim arc,
    # which is one-sided: the projector shadows the far side)
    inset = float(prm.rim_inset_px) * _footprint_mm(X3, m)
    c0, n0 = c, n
    # the circle the rim points lie on: the face's (its diameter is known), unless the
    # depth silhouette is the part's larger outer circle (given, or found from the points)
    fit_r = radius if prm.silhouette_diameter_mm is None else float(prm.silhouette_diameter_mm) / 2
    sil_src = None if prm.silhouette_diameter_mm is None else "given"
    for attempt in range(2):
        c, n = c0, n0
        e1, e2 = face_axes(n)
        ub = np.c_[(Rm - c) @ e1, (Rm - c) @ e2]
        q, res = _circle_fit(ub, fit_r - inset, prm.f_scale_mm)
        for _ in range(int(prm.refine_iters)):
            # the depth face also holds the cell's side wall (within near_band_mm): refit
            # the plane on the face only (inside the circle, MAD-trimmed), then the circle
            C3 = c + q[0] * e1 + q[1] * e2
            w = P - C3
            inner = np.hypot(w @ e1, w @ e2) < radius - prm.refine_margin_mm
            if inner.sum() < max(prm.min_face_px, 10):
                break
            c, n = _robust_plane(P[inner])
            e1, e2 = face_axes(n)
            ub = np.c_[(Rm - c) @ e1, (Rm - c) @ e2]
            q, res = _circle_fit(ub, fit_r - inset, prm.f_scale_mm)
        if attempt or sil_src is not None:
            break
        r_sil = _silhouette_radius(ub, q, res, radius, inset, prm)
        if r_sil is None:
            break
        fit_r, sil_src = r_sil, "rim points"
    rms = float(np.sqrt(np.mean(res ** 2)))
    arc = _arc_deg(ub, q)
    if rms > prm.max_rim_rms_mm:
        return None, (f"rim misfits a {2 * fit_r:g} mm circle (rms {rms:.2f} mm)"
                      + _other_size(ub, fit_r, inset, arc))
    if arc < prm.min_rim_arc_deg:
        return None, f"rim covers only {arc:.0f} deg"
    return {"depth_face": m, "plane_c": c, "n": n, "e1": e1, "e2": e2, "q": q, "rim_px": rim_px,
            "rim_res": res, "rim_rms_mm": rms, "rim_arc_deg": arc, "aspect": aspect,
            "rim_inset_mm": inset, "n_contour": int(len(cnt)), "zmin": float(zmin),
            "silhouette_diameter_mm": None if sil_src is None else 2.0 * float(fit_r),
            "silhouette_source": sil_src}, "ok"


def _other_size(ub: np.ndarray, radius: float, inset: float, arc: float) -> str:
    """A rim that misfits the diameter given but is a clean circle of ANOTHER size: say so
    (a scan that reads the wrong size - a lens zoomed after the calibration makes every
    lateral size small - or simply not this face). '' when the rim is no circle either."""
    if len(ub) < 12 or arc < 120.0:
        return ""
    A = np.c_[2 * ub, np.ones(len(ub))]
    s = np.linalg.lstsq(A, (ub ** 2).sum(1), rcond=None)[0]
    r = float(np.sqrt(max(s[2] + s[0] ** 2 + s[1] ** 2, 0.0)))
    rms = float(np.sqrt(np.mean((np.hypot(ub[:, 0] - s[0], ub[:, 1] - s[1]) - r) ** 2)))
    d = 2.0 * (r + inset)
    pct = 100.0 * (d / (2.0 * radius) - 1.0)
    if rms > 0.3 or abs(pct) <= 4.0:
        return ""
    return (f"; it is a circle {d:.1f} mm across (rms {rms:.2f} mm): {abs(pct):.0f} % "
            f"{'smaller' if pct < 0 else 'larger'} than the diameter given - a size mismatch "
            "(calibration or lens suspect: run rig-check / recalibrate) or not this face")


def _footprint_mm(X3: np.ndarray, m: np.ndarray) -> float:
    """Median 3-D distance between neighbouring pixels of the component (mm per camera px)."""
    d = []
    for a, b in ((X3[:, 1:], X3[:, :-1]), (X3[1:], X3[:-1])):
        mm = (m[:, 1:] & m[:, :-1]) if a.shape[1] != X3.shape[1] else (m[1:] & m[:-1])
        v = np.linalg.norm(a - b, axis=-1)[mm]
        d.append(v[np.isfinite(v)])
    v = np.concatenate(d)
    return float(np.median(v)) if v.size else 0.0


def _robust_plane(P: np.ndarray, iters: int = 3, k: float = 3.0) -> tuple:
    c, n = plane_svd(P)
    for _ in range(iters):
        h = (P - c) @ n
        med = np.median(h)
        mad = np.median(np.abs(h - med))
        keep = np.abs(h - med) < k * 1.4826 * mad if mad > 0 else np.ones(len(P), bool)
        if keep.sum() < 10:
            break
        c, n = plane_svd(P[keep])
    return c, n


def find_face(xyz: np.ndarray, mask: np.ndarray, radius: float, roi=None,
              prm: Optional[LocateParams] = None) -> dict:
    """Depth face + rim + fixed-radius circle -> dict (crop coordinates + origin)."""
    import cv2
    prm = prm or LocateParams()
    x0, y0, x1, y1 = _region(mask, roi)
    X3 = np.where(mask[y0:y1, x0:x1, None], xyz[y0:y1, x0:x1], np.nan).astype(np.float64)
    Z = X3[..., 2]
    finite = np.isfinite(Z)
    excluded = np.zeros(Z.shape, bool)
    reasons = []
    k = max(int(prm.open_px), 1)
    for layer in range(int(prm.max_layers)):
        fin = finite & ~excluded
        if fin.sum() < prm.min_face_px:
            break
        zmin = nearest_populated(Z[fin], prm.min_face_px, prm.populated_span_mm)
        if zmin is None:
            break
        near_raw = fin & (Z < zmin + prm.near_band_mm)
        near = cv2.morphologyEx(near_raw.astype(np.uint8), cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(near)
        order = [int(i) + 1 for i in np.argsort(-st[1:, cv2.CC_STAT_AREA])
                 if st[int(i) + 1, cv2.CC_STAT_AREA] >= prm.min_face_px]
        for ci in order:
            cand, why = _try_component(lab == ci, X3, Z, zmin, radius, prm)
            mode = "depth"
            if prm.neighbour_split:
                alt, why2 = _split_try(lab == ci, X3, Z, zmin, radius, prm)
                if alt is not None and (
                        cand is None
                        or (cand["rim_arc_deg"] < prm.split_below_arc_deg
                            and _rim_score(alt) < _rim_score(cand))
                        # a neighbour hid much of the far background: the split sees the rim
                        # against it too, at no worse a fit
                        or (len(alt["rim_px"]) >= prm.split_rim_gain * len(cand["rim_px"])
                            and alt["rim_rms_mm"] <= cand["rim_rms_mm"])):
                    cand, mode = alt, "depth (plane split)"
                if cand is None:
                    why = f"{why}; split from a near neighbour: {why2}"
            if cand is not None:
                cand.update({"origin": (x0, y0), "region": (x0, y0, x1, y1), "layer": layer,
                             "rejected": reasons, "rim_source": mode})
                return cand
            reasons.append({"layer": layer, "z_near_mm": round(zmin, 1),
                            "px": int(st[ci, cv2.CC_STAT_AREA]), "why": why})
        excluded |= near_raw
    detail = "; ".join(f"z {r['z_near_mm']} mm, {r['px']} px: {r['why']}" for r in reasons[:6])
    raise FaceNotFound(
        f"no face-shaped nearest surface in camera px {x0},{y0}..{x1},{y1}"
        + (f" ({detail})" if detail else "") + ": is the face in the scan's zone and is the "
        "background at least far_gap_mm behind it (or a neighbour near_gap_mm behind the face "
        "plane)? Pass roi= to narrow the search")


def find_face_flood(xyz: np.ndarray, mask: np.ndarray, radius: float, roi, prm: LocateParams,
                    flood_d: np.ndarray, K_cam, dist_cam, why: str = "") -> dict:
    """Flood-circle fallback: the face disc in the flood image (Hough, the expected size at the
    nearest depth), the nearest flat one; its sub-pixel rim edges back-projected onto its depth
    plane and a circle fitted there. -> the find_face dict (crop coordinates + origin)."""
    import cv2
    from ..station import geometry as sg
    x0, y0, x1, y1 = _region(mask, roi)
    X3 = np.where(mask[y0:y1, x0:x1, None], xyz[y0:y1, x0:x1], np.nan).astype(np.float64)
    Z = X3[..., 2]
    fin = np.isfinite(Z)
    zmin = nearest_populated(Z[fin], prm.min_face_px, prm.populated_span_mm) if fin.any() else None
    if zmin is None:
        raise FaceNotFound(f"flood fallback: no measured surface in camera px {x0},{y0}..{x1},{y1}")
    f = 0.5 * (float(np.asarray(K_cam)[0, 0]) + float(np.asarray(K_cam)[1, 1]))
    r_px = f * radius / zmin
    d = np.clip(np.nan_to_num(np.asarray(flood_d, np.float64)[y0:y1, x0:x1]), 0, 255)
    g = cv2.GaussianBlur(d.astype(np.uint8), (5, 5), 1.5)
    cir = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1, minDist=max(r_px, 8.0), param1=80,
                           param2=18, minRadius=int(np.floor(0.75 * r_px)),
                           maxRadius=int(np.ceil(1.25 * r_px)))
    H, W = Z.shape
    yy, xx = np.mgrid[0:H, 0:W]
    best, notes = None, []
    for cx, cy, r in (cir[0] if cir is not None else []):
        disc = (xx - cx) ** 2 + (yy - cy) ** 2 < (0.8 * r) ** 2
        P = X3[disc]
        P = P[np.all(np.isfinite(P), 1)]
        if len(P) < max(prm.min_face_px, 0.5 * disc.sum()):
            notes.append(f"({cx + x0:.0f}, {cy + y0:.0f}): {len(P)} measured px")
            continue
        c, n = _robust_plane(P)
        h = (P - c) @ n
        rms = float(1.4826 * np.median(np.abs(h - np.median(h))))
        if rms > 0.3:
            notes.append(f"({cx + x0:.0f}, {cy + y0:.0f}): not flat ({rms:.2f} mm)")
            continue
        zc = float(np.median(P[:, 2]))
        if best is None or zc < best[0]:
            best = (zc, float(cx), float(cy), float(r), c, n, disc)
    if best is None:
        raise FaceNotFound(f"flood fallback: no flat face disc of ~{r_px:.0f} px in the flood"
                           + (f" ({'; '.join(notes[:4])})" if notes else "") + (f" [{why}]" if why else ""))
    _, cx, cy, r, c, n, disc = best
    edges = sg.rim_edges(d, cx, cy, r_px)
    full = edges[:, :2] + [x0, y0]
    fit = sg.circle_in_plane(full, c, n, K_cam, dist_cam)
    if fit is None or fit[3] < prm.min_flood_edges:
        raise FaceNotFound(f"flood fallback: only {0 if fit is None else fit[3]} rim edge points "
                           f"on the flood disc at ({cx + x0:.0f}, {cy + y0:.0f})")
    C, rad, rms, n_used = fit
    e1, e2 = face_axes(n)
    q = np.array([(C - c) @ e1, (C - c) @ e2])
    d3 = sg.camera_rays(K_cam, dist_cam, full)
    Pe = d3 * ((c @ n) / (d3 @ n))[:, None]
    ub = np.c_[(Pe - c) @ e1, (Pe - c) @ e2]
    res = np.hypot(ub[:, 0] - q[0], ub[:, 1] - q[1]) - rad
    comp = np.zeros(Z.shape, bool)
    comp[disc] = True
    return {"depth_face": comp, "plane_c": c, "n": n, "e1": e1, "e2": e2, "q": q,
            "rim_px": np.round(edges[:, :2]).astype(int), "rim_res": res, "rim_rms_mm": float(rms),
            "rim_arc_deg": _arc_deg(ub, q), "aspect": 1.0, "rim_inset_mm": 0.0,
            "n_contour": int(len(edges)), "zmin": float(zmin), "origin": (x0, y0),
            "region": (x0, y0, x1, y1), "layer": 0, "rejected": [],
            "rim_source": "flood outline", "flood_outline_diameter_mm": 2 * rad,
            "flood_edges_used": int(n_used), "depth_failure": why}


def camera_circle(C3, e1, e2, radius, K_cam, dist_cam, n_pts: int = 360) -> np.ndarray:
    """The 3-D face circle projected into the camera -> (n_pts, 2) px (full frame)."""
    import cv2
    tt = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    circ = C3 + radius * (np.cos(tt)[:, None] * e1 + np.sin(tt)[:, None] * e2)
    px, _ = cv2.projectPoints(circ.astype(np.float64), np.zeros(3), np.zeros(3),
                              np.asarray(K_cam, np.float64), np.asarray(dist_cam, np.float64))
    return px[:, 0, :]


def ellipse_dict(e) -> dict:
    (cx, cy), (a, b), ang = e
    return {"cx": float(cx), "cy": float(cy), "a": float(a), "b": float(b), "angle_deg": float(ang),
            "note": "cv2.fitEllipse: centre, FULL axes, angle"}


def ellipse_tuple(d: dict) -> tuple:
    return (d["cx"], d["cy"]), (d["a"], d["b"]), d["angle_deg"]


def ellipse_points(d: dict, n: int = 720) -> np.ndarray:
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    a, b = d["a"] / 2, d["b"] / 2
    ca, sa = np.cos(np.radians(d["angle_deg"])), np.sin(np.radians(d["angle_deg"]))
    return np.c_[d["cx"] + a * np.cos(t) * ca - b * np.sin(t) * sa,
                 d["cy"] + a * np.cos(t) * sa + b * np.sin(t) * ca]


def fit_affine(cam_mask: np.ndarray, corr_x, corr_y, field_mask, trim_iters: int = 2,
               trim_k: float = 4.0, trim_min_px: float = 0.5) -> dict:
    """Camera px -> native DMD (col, row) affine on the decoded pixels of cam_mask
    (full-frame arrays) -> {A (2x3), rms_col_px, rms_row_px, p95_abs_px, n_px, n_outliers}.
    Least squares (the prototype), then `trim_iters` refits without pixels off by more than
    max(trim_k x 1.4826 MAD, trim_min_px): rim pixels that see the background behind the face
    (parallax) decode another surface and must not bend the face's map."""
    ok = cam_mask & np.asarray(field_mask, bool) & np.isfinite(corr_x) & np.isfinite(corr_y)
    fy, fx = np.nonzero(ok)
    if len(fx) < 12:
        raise FaceNotFound(f"only {len(fx)} decoded pixels on the face: the camera -> DMD map "
                           "can't be fitted (the locate scan needs both axes decoded on the face)")
    A = np.c_[fx, fy, np.ones(len(fx))].astype(np.float64)
    U = corr_x[fy, fx].astype(np.float64)
    V = corr_y[fy, fx].astype(np.float64)
    keep = np.ones(len(fx), bool)
    for it in range(int(trim_iters) + 1):
        au = np.linalg.lstsq(A[keep], U[keep], rcond=None)[0]
        av = np.linalg.lstsq(A[keep], V[keep], rcond=None)[0]
        ru, rv = U - A @ au, V - A @ av
        if it == int(trim_iters):
            break
        e = np.hypot(ru, rv)
        med = float(np.median(e[keep]))
        mad = float(np.median(np.abs(e[keep] - med)))
        new = e <= med + max(trim_k * 1.4826 * mad, trim_min_px)
        if new.sum() < 12 or np.array_equal(new, keep):
            break
        keep = new
    return {"A": np.stack([au, av]), "rms_col_px": float(ru[keep].std()),
            "rms_row_px": float(rv[keep].std()),
            "p95_abs_px": float(np.percentile(np.hypot(ru[keep], rv[keep]), 95)),
            "n_px": int(keep.sum()), "n_outliers": int((~keep).sum()),
            "rms_all_px": float(np.sqrt(np.mean(ru ** 2 + rv ** 2))),
            "units": "native DMD px (col, row) per camera px"}


def apply_affine(A, pts) -> np.ndarray:
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    return np.c_[pts, np.ones(len(pts))] @ np.asarray(A, np.float64).T


def locate(xyz, mask, corr_x, corr_y, field_mask, K_cam, dist_cam,
           face_diameter_mm: float = FACE_DIAMETER_MM, roi=None,
           prm: Optional[LocateParams] = None, flood: Optional[np.ndarray] = None) -> dict:
    """The whole locate: face, circle, camera mask + ellipse, affine, DMD ring. -> dict
    (full-frame camera px; mm in the camera frame; DMD native px). flood: (H, W) lit-minus-
    dark image 0..255 for the flood-circle fallback (None = no fallback)."""
    import cv2
    prm = prm or LocateParams()
    if not (face_diameter_mm and face_diameter_mm > 0):
        raise ValueError(f"face_diameter_mm must be > 0, got {face_diameter_mm!r}")
    radius = float(face_diameter_mm) / 2
    sd = prm.silhouette_diameter_mm
    if sd is not None:
        if isinstance(sd, bool) or not isinstance(sd, (int, float)) or not np.isfinite(sd) \
                or not float(face_diameter_mm) <= sd <= 1.5 * float(face_diameter_mm):
            raise ValueError(f"silhouette_diameter_mm {sd!r} must be the part's outer diameter: "
                             f"between the face diameter {float(face_diameter_mm):g} mm and 1.5 x "
                             "it (the depth silhouette of a flat face inside a rounded rim)")
    if flood is not None and np.shape(flood) != np.shape(mask):
        raise ValueError(f"flood {np.shape(flood)} and scan {np.shape(mask)} differ in size: the "
                         "flood-outline fallback needs the scan's camera frame")
    try:
        f = find_face(xyz, mask, radius, roi, prm)
    except FaceNotFound as e:
        if flood is None or not prm.flood_fallback:
            raise
        f = find_face_flood(xyz, mask, radius, roi, prm, flood, K_cam, dist_cam, str(e))
    x0, y0 = f["origin"]
    c, n, e1, e2, q = f["plane_c"], f["n"], f["e1"], f["e2"], f["q"]
    C3 = c + q[0] * e1 + q[1] * e2
    H, W = mask.shape
    circ = camera_circle(C3, e1, e2, radius, K_cam, dist_cam, prm.circle_points)
    cam_mask = np.zeros((H, W), np.uint8)
    cv2.fillPoly(cam_mask, [np.round(circ).astype(np.int32).reshape(-1, 1, 2)], 1)
    cam_mask = cam_mask.astype(bool)
    cam_ell = ellipse_dict(cv2.fitEllipse(circ.astype(np.float32)))
    aff = fit_affine(cam_mask, corr_x, corr_y, field_mask)
    ring = apply_affine(aff["A"], ellipse_points(cam_ell, prm.ring_points))
    dmd_centre = apply_affine(aff["A"], [[cam_ell["cx"], cam_ell["cy"]]])[0]
    dmd_ell = ellipse_dict(cv2.fitEllipse(ring.astype(np.float32)))
    tilt = float(np.degrees(np.arccos(min(1.0, abs(n @ (-C3 / np.linalg.norm(C3)))))))
    warnings = []
    worst = max(aff["rms_col_px"], aff["rms_row_px"])
    if worst > AFFINE_WARN_PX:
        warnings.append(f"camera -> DMD affine residual {worst:.2f} DMD px > {AFFINE_WARN_PX:g}: "
                        "the face is not one flat decoded patch (partly decoded, a bad unwrap, or "
                        "not the face) - check the ring with check_ring before scanning")
    if not cam_mask[np.clip(np.round(circ[:, 1]).astype(int), 0, H - 1),
                    np.clip(np.round(circ[:, 0]).astype(int), 0, W - 1)].all():
        warnings.append("the face circle leaves the camera frame")
    mean_semi = (cam_ell["a"] + cam_ell["b"]) / 4
    src = f.get("rim_source", "depth")
    if src == "flood outline":
        warnings.append(f"no depth rim (the face merges with its surroundings in depth): the rim is "
                        f"the face outline in the flood (outline {f['flood_outline_diameter_mm']:.2f} mm "
                        f"across, {f['flood_edges_used']} edge points); check the ring with check_ring")
    if f.get("silhouette_source") == "rim points":
        warnings.append(f"the depth silhouette is {f['silhouette_diameter_mm']:.2f} mm across, "
                        f"larger than the {float(face_diameter_mm):g} mm face given (the rim "
                        "around the flat returns depth points - or the scan reads large: see "
                        "the size check): the face is centred on the silhouette and keeps the "
                        "diameter given (the part's outer diameter can be given: "
                        "silhouette_diameter_mm / --outer-diameter)")
    return {"radius_mm": radius, "face_diameter_mm": float(face_diameter_mm),
            "rim_source": src, "flood_outline_diameter_mm": f.get("flood_outline_diameter_mm"),
            "silhouette_diameter_mm": f.get("silhouette_diameter_mm"),
            "silhouette_source": f.get("silhouette_source"),
            "centre_mm": C3, "normal": n, "e1": e1, "e2": e2, "plane_point": c,
            "tilt_deg": tilt, "rim_rms_mm": f["rim_rms_mm"], "rim_inset_mm": f["rim_inset_mm"],
            "n_rim": int(len(f["rim_px"])),
            "n_contour": f["n_contour"], "rim_arc_deg": f["rim_arc_deg"],
            "rim_px": f["rim_px"] + [x0, y0], "depth_face": f["depth_face"],
            "depth_face_px": int(f["depth_face"].sum()), "z_near_mm": f["zmin"],
            "region": f["region"], "origin": (x0, y0), "layer": f["layer"],
            "rejected_components": f["rejected"],
            "circle_cam_px": circ, "cam_mask": cam_mask, "cam_ellipse": cam_ell,
            "affine": aff, "dmd_ring": ring, "dmd_centre": dmd_centre, "dmd_ellipse": dmd_ell,
            "mm_per_camera_px": radius / mean_semi if mean_semi > 0 else None,
            "params": prm.to_dict(), "warnings": warnings}

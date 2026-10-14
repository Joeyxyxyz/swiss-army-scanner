"""
Face profile from N repeat metric scans of one placement (the bench prototype
profile_quick, productised).

  * the per-pixel MEDIAN of the repeats (camera grid, mm, camera frame)
  * face plane: SVD on the face mask eroded 5x5 (~2 px inside the rim), then 3 robust
    refits (points within 3 x 1.4826 MAD of the median height), normal toward the camera
  * centre C: the camera ray through the face-mask ellipse centre, hit on that plane
  * face frame e1 = n x y_cam (~ camera right), e2 = e1 x n (~ camera down);
    u, v, h = (X - C).(e1, e2, n)  (h + toward the camera: a dent is NEGATIVE)
  * height map: linear interpolation (scipy griddata) of the face points onto a square
    grid (default 0.25 mm) over |r| <= the face radius
  * radial profile (0.5 mm bins: median height, spread, count), horizontal (v = 0) and
    vertical (u = 0) sections, flatness (RMS and p2-p98 inside r_flat_mm: 0.855 x the face
    radius unless given, radii.py - 9.0 mm on a 21.05 mm face), per-point repeatability (std
    across repeats at each face pixel), tilt vs the camera ray
  * what each repeat gave: its face points (repeat_face_points), the repeats that gave any
    (repeats_used) and the share of the face points every repeat measured (all_repeats_pct).
    A repeat that kept far fewer points than the others (low_repeats: under
    LOW_REPEAT_FRACTION of their median - a glint, a moved part, a busy PC) is named by the
    face scan and by the profile; it still counts where it measured (the median per pixel)
"""

from __future__ import annotations

import io
import warnings
from typing import Optional

import numpy as np

from . import radii as radii_mod
from .locate import face_axes

GRID_MM = 0.25
R_FLAT_MM = None                        # flatness radius: None = 0.855 x the face radius
BEHIND_MM = 2.0                         # a measured point this far behind the face plane is
                                        # not the face (background, the part's side wall)
RADIAL_BIN_MM = 0.5
MAP_RANGE_MM = 0.4                      # PNG colour range +-
SECTION_RANGE_MM = 0.5
# a repeat with fewer face points than this x the median of the other repeats is named (the
# bench: one of three repeats kept half its points and took longer, and nothing said so)
LOW_REPEAT_FRACTION = 0.75


def low_repeats(counts, fraction: float = LOW_REPEAT_FRACTION) -> list:
    """The repeats that kept far fewer points than their siblings: fewer than `fraction` x
    the median of the OTHER repeats' counts -> [{repeat (1-based), points, others_median,
    pct_of_others}]. One repeat, or others with no points: nothing to compare with."""
    c = [int(n) for n in counts]
    out = []
    for k, n in enumerate(c):
        others = c[:k] + c[k + 1:]
        med = float(np.median(others)) if others else 0.0
        if med > 0 and n < fraction * med:
            out.append({"repeat": k + 1, "points": n, "others_median": med,
                        "pct_of_others": round(100.0 * n / med, 1)})
    return out


def grid_axis(radius_mm: float, step_mm: float = GRID_MM) -> np.ndarray:
    """Symmetric grid axis covering the face (+ one cell): -half .. +half mm."""
    if not step_mm > 0:
        raise ValueError(f"grid step must be > 0 mm, got {step_mm!r}")
    n = int(np.ceil(radius_mm / step_mm)) + 1
    return np.arange(-n, n + 1) * float(step_mm)


def robust_plane(P: np.ndarray, iters: int = 3, k: float = 3.0) -> tuple:
    """(centroid, normal toward the camera) with `iters` MAD-trimmed refits."""
    P = np.asarray(P, np.float64)       # a float32 mean over ~1e6 points is mm off
    c = P.mean(0)
    _, _, vt = np.linalg.svd(P - c, full_matrices=False)
    n = vt[2] * np.sign(-vt[2] @ c)
    for _ in range(int(iters)):
        h = (P - c) @ n
        med = np.median(h)
        mad = np.median(np.abs(h - med))
        keep = np.abs(h - med) < k * 1.4826 * mad if mad > 0 else np.ones(len(P), bool)
        if keep.sum() < 10:
            break
        c = P[keep].mean(0)
        _, _, vt = np.linalg.svd(P[keep] - c, full_matrices=False)
        n = vt[2] * np.sign(-vt[2] @ c)
    return c, n


def mask_ellipse_centre(face_mask: np.ndarray) -> tuple:
    import cv2
    cs, _ = cv2.findContours(face_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        raise ValueError("empty face mask")
    (ex, ey), _, _ = cv2.fitEllipse(max(cs, key=cv2.contourArea))
    return float(ex), float(ey)


def flat_radius(radius_mm: float, r_flat_mm: Optional[float] = None) -> float:
    """The flatness radius: the one given (it must lie on the face), else 0.855 x the face
    radius."""
    return float(radii_mod.face_radii(radius_mm, {"r_flat_mm": r_flat_mm})["r_flat_mm"])


def analyse(X: np.ndarray, K: np.ndarray, face_mask: np.ndarray, origin, cal, radius_mm: float,
            grid_mm: float = GRID_MM, r_flat_mm: Optional[float] = R_FLAT_MM,
            erode_px: int = 5, off_face_mm: float = 2.0, centre_px=None) -> dict:
    """X (n, h, w, 3) repeat clouds (crop, mm), K (n, h, w) their masks, face_mask (h, w)
    = the located face in the same crop, origin = (x0, y0) of the crop in the camera.
    centre_px: the face centre in crop px (default: the face mask's ellipse centre; pass the
    located circle's when the face leaves the camera frame and the mask is clipped).
    -> the height map + stats, 'pixels' (iy, ix, per-repeat U, V, H of the face pixels) and
    'behind' ((h, w) bool: crop pixels measured more than BEHIND_MM behind the face plane)."""
    import cv2
    from scipy.interpolate import griddata
    from ...reconstruct.triangulate import camera_rays
    r_flat_mm = flat_radius(radius_mm, r_flat_mm)
    fm = np.asarray(face_mask, bool)
    X = np.where(K[..., None], X, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)          # all-NaN pixels
        M = np.nanmedian(X, 0)
    k = max(int(erode_px), 1)
    er = cv2.erode(fm.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
    P = M[er & np.isfinite(M[..., 2])]
    if len(P) < 30:
        raise ValueError(f"only {len(P)} measured face points (the repeats decoded almost nothing "
                         "inside the face mask: is the face still where it was located?)")
    c, n = robust_plane(P)
    ex, ey = mask_ellipse_centre(fm) if centre_px is None else (float(centre_px[0]),
                                                                  float(centre_px[1]))
    d = camera_rays(cal, None, np.array([[ex + origin[0], ey + origin[1]]], float))[0]
    C0 = d * ((c @ n) / (d @ n))
    e1, e2 = face_axes(n)
    sel = fm & np.isfinite(M[..., 2])
    W = M - C0
    hh = np.where(sel, W @ n, np.nan)
    med = float(np.nanmedian(hh[sel]))
    mad = float(np.nanmedian(np.abs(hh[sel] - med)))
    # not the face: mask-edge pixels that see the body / background behind the rim
    near = np.abs(np.nan_to_num(hh, nan=np.inf) - med) <= max(off_face_mm, 10 * 1.4826 * mad)
    sel &= near & (np.hypot(W @ e1, W @ e2) <= radius_mm + 1.0)
    Q = M[sel]
    u, v, h = (Q - C0) @ e1, (Q - C0) @ e2, (Q - C0) @ n
    r = np.hypot(u, v)
    # per camera pixel and per repeat (the MF detector: per-repeat planes, the reflectance
    # FIR in camera space, dents injected per pixel): u, v, h of every face-mask pixel, NaN
    # where that repeat did not decode it or saw something off the face
    iy, ix = np.nonzero(fm)
    Wr = X[:, iy, ix] - C0                                        # (K, N, 3)
    pu, pv, ph = Wr @ e1, Wr @ e2, Wr @ n
    good = np.isfinite(ph) & (np.abs(np.nan_to_num(ph, nan=np.inf) - med)
                              <= max(off_face_mm, 10 * 1.4826 * mad)) \
        & (np.hypot(pu, pv) <= radius_mm + 1.0)
    pu, pv, ph = (np.where(good, a, np.nan) for a in (pu, pv, ph))
    anyk = good.any(0)
    per_repeat = good.sum(1)                # the face points each repeat gave
    n_by_point = good.sum(0)[anyk]          # how many repeats measured each face point
    pixels = {"iy": iy[anyk], "ix": ix[anyk], "U": pu[:, anyk], "V": pv[:, anyk],
              "H": ph[:, anyk]}
    # per-point repeatability: std of the height across repeats at each face pixel
    Hs = np.stack([(X[i][sel] - C0) @ n for i in range(len(X))])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        rep = np.nanstd(Hs, 0) if len(X) > 1 else np.full(Hs.shape[1], np.nan)
    rep = rep[np.isfinite(rep) & (np.sum(np.isfinite(Hs), 0) >= 2)]
    g = grid_axis(radius_mm, grid_mm)
    GU, GV = np.meshgrid(g, g)
    Hg = griddata(np.c_[u, v], h, (GU, GV), method="linear")
    Hg[np.hypot(GU, GV) > radius_mm] = np.nan
    rb = np.arange(0.0, radius_mm + RADIAL_BIN_MM, RADIAL_BIN_MM)
    radial = []
    for lo in rb:
        s = (r >= lo) & (r < lo + RADIAL_BIN_MM)
        if s.sum() > 5:
            radial.append({"r_mm": round(float(lo + RADIAL_BIN_MM / 2), 3),
                           "h_median_mm": float(np.median(h[s])), "h_std_mm": float(np.std(h[s])),
                           "n": int(s.sum())})
    ic = len(g) // 2
    inner = h[r < r_flat_mm]
    cover = np.isfinite(Hg[np.hypot(GU, GV) <= radius_mm])
    tilt = float(np.degrees(np.arccos(min(1.0, abs(n @ (-C0 / np.linalg.norm(C0)))))))
    stats = {"n_points": int(len(Q)), "n_off_face": int((fm & np.isfinite(M[..., 2])).sum() - len(Q)),
             "n_repeats": int(len(X)),
             "repeat_face_points": [int(n) for n in per_repeat],
             "repeats_used": int((per_repeat > 0).sum()),
             "all_repeats_pct": (round(100.0 * float(np.mean(n_by_point == len(X))), 1)
                                 if n_by_point.size else None),
             "covered_radius_mm": float(r.max()) if len(r) else None,
             "grid_coverage_pct": round(100.0 * float(cover.mean()), 2) if cover.size else None,
             "r_flat_mm": float(r_flat_mm),
             "flatness_rms_mm": float(inner.std()) if inner.size > 5 else None,
             "flatness_p2_p98_mm": float(np.ptp(np.percentile(inner, [2, 98]))) if inner.size > 5 else None,
             "repeatability_median_mm": float(np.median(rep)) if rep.size else None,
             "repeatability_p90_mm": float(np.percentile(rep, 90)) if rep.size else None,
             "tilt_deg": tilt, "plane_points": int(len(P))}
    with np.errstate(invalid="ignore"):
        behind = np.isfinite(M[..., 2]) & (((M - C0) @ n) - med < -BEHIND_MM)
    return {"grid_mm": g, "height_mm": Hg, "u": u, "v": v, "h": h, "centre": C0, "normal": n,
            "e1": e1, "e2": e2, "median_xyz": M, "radial": radial, "behind": behind,
            "sections": {"u_mm": g, "horizontal_mm": Hg[ic, :], "v_mm": g, "vertical_mm": Hg[:, ic]},
            "stats": stats, "pixels": pixels}


# ------------------------------------------------------------------ files

def radial_csv(radial: list) -> str:
    lines = ["r_mm,h_median_mm,h_std_mm,n_points"]
    lines += [f"{d['r_mm']:.3f},{d['h_median_mm']:.5f},{d['h_std_mm']:.5f},{d['n']}" for d in radial]
    return "\n".join(lines) + "\n"


def sections_csv(sec: dict) -> str:
    lines = ["pos_mm,horizontal_h_mm (v=0, along u = camera right),vertical_h_mm (u=0, along v = "
             "camera down)"]
    f = lambda x: "" if not np.isfinite(x) else f"{x:.5f}"      # noqa: E731
    for p, a, b in zip(sec["u_mm"], sec["horizontal_mm"], sec["vertical_mm"]):
        lines.append(f"{p:.3f},{f(a)},{f(b)}")
    return "\n".join(lines) + "\n"


def colour_map(a: np.ndarray, lo: float, hi: float, valid: Optional[np.ndarray] = None,
               scale: int = 6) -> np.ndarray:
    import cv2
    x = np.clip((np.nan_to_num(a, nan=lo) - lo) / (hi - lo), 0, 1)
    col = cv2.applyColorMap((x * 255).astype(np.uint8), cv2.COLORMAP_JET)
    bad = ~np.isfinite(a) if valid is None else ~(valid & np.isfinite(a))
    col[bad] = 40
    return cv2.resize(col, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)


def profile_png(res: dict, radius_mm: float, title: str = "") -> Optional[bytes]:
    """Height map (+-0.4 mm, camera view: u right, v down) with the rim circle + sections."""
    try:
        import cv2
    except ImportError:
        return None
    g, Hg = res["grid_mm"], res["height_mm"]
    step = float(g[1] - g[0])
    sc = 6
    col = colour_map(Hg, -MAP_RANGE_MM, MAP_RANGE_MM, scale=sc)
    cx = cy = int(round((len(g) / 2) * sc))
    cv2.circle(col, (cx, cy), int(round(radius_mm / step * sc)), (255, 255, 255), 1)
    cv2.putText(col, f"end-face height (mm) -{MAP_RANGE_MM:g} blue .. +{MAP_RANGE_MM:g} red",
                (6, 18), 0, 0.45, (255, 255, 255), 1)
    cv2.putText(col, "camera view: u right, v down; + toward camera", (6, 36), 0, 0.4,
                (255, 255, 255), 1)
    Hh = col.shape[0]
    sec = np.full((Hh, 520, 3), 30, np.uint8)
    span = float(g[-1] - g[0])
    for j, (name, line, clr) in enumerate((("horizontal (v=0)", res["sections"]["horizontal_mm"],
                                             (0, 200, 255)),
                                            ("vertical (u=0)", res["sections"]["vertical_mm"],
                                             (255, 200, 0)))):
        ok = np.isfinite(line)
        if ok.sum() > 1:
            pts = np.c_[20 + (g[ok] - g[0]) / span * 480,
                        Hh / 2 - line[ok] / SECTION_RANGE_MM * (Hh / 2 - 30)].astype(np.int32)
            cv2.polylines(sec, [pts.reshape(-1, 1, 2)], False, clr, 2)
        cv2.putText(sec, name + " section", (20, 24 + 20 * j), 0, 0.5, clr, 1)
    cv2.line(sec, (20, Hh // 2), (500, Hh // 2), (90, 90, 90), 1)
    st = res["stats"]
    txt = (f"+-{SECTION_RANGE_MM:g} mm full scale, {g[0]:g}..{g[-1]:g} mm | flat rms "
           f"{st['flatness_rms_mm'] or float('nan'):.3f} mm (r<{st['r_flat_mm']:g})")
    cv2.putText(sec, txt, (20, Hh - 12), 0, 0.4, (200, 200, 200), 1)
    if title:
        cv2.putText(sec, title[:60], (20, 64), 0, 0.45, (220, 220, 220), 1)
    ok, buf = cv2.imencode(".png", np.hstack([col, sec]))
    return buf.tobytes() if ok else None


def npz_bytes(**arrays) -> bytes:
    b = io.BytesIO()
    np.savez_compressed(b, **arrays)
    return b.getvalue()

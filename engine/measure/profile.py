"""
2D profiles — height along a line across a scan (cell-height work).

A profile samples the relief along the segment p0 -> p1 (camera px),
averaging `width_px` pixels perpendicular to the line (noise ~1/sqrt(w)).
axis="auto" with no points: the line runs through the object blob's centre
along its principal (long) axis; axis="x"/"y": horizontal/vertical line
through the blob centre. Heights in mm with a matching depth calibration,
else px (rule 6); distance in mm with a lateral calibration, else px.
Zone scans are datum-free (base plane at 0), so their blob is found around
0, not the median; a part filling the zone gives a line across the whole
zone and a warning (no base plane in view, so no step - read the median).

Outputs: library/output/profiles/<ts>_<label>.csv (distance, height, n) and .png plot.

Sampling is bilinear and NaN-aware: a sample takes the weighted mean of
its valid neighbours and is NaN when less than half its weight is valid,
so shadows and unlit pixels never bleed into heights as zeros.

The auto line spans the blob extent plus a margin on both ends, so the
profile always holds base plane AND top — `stats["step_height"]` (a 2-level
split, the optimal 1-D 2-means cut, reporting the difference of the two
level medians) is then the part height. Medians, not means, because the
edge-transition samples and the width-averaged shadow fringes belong to
neither level.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .. import rules
from ..results import TIER_DEPTH, TIER_LATERAL, TIER_RELIEF
from .segment import SegmentationError, is_datum_free, object_blob

AXES = ("auto", "x", "y")
MARGIN_MIN_PX = 40.0        # base plane kept past each end of the blob (camera px)
MARGIN_FRAC = 0.3           # ... or this fraction of the blob extent, if larger
STEP_MIN_SEP = 6.0          # a step must exceed this many within-level robust sigmas
STEP_MIN_FRAC = 0.05        # each level needs >= 5% of the valid samples (and >= 3)


@dataclass
class Profile:
    scan_id: str
    p0: list                    # [x, y] camera px
    p1: list
    width_px: int
    distance: list              # along the line
    height: list                # NaN where invalid
    units_distance: str         # "mm" | "px"
    units_height: str           # "mm" | "px"
    tier: str
    stats: dict                 # min, max, range, median, step_height (if a step is found)
    csv_path: Optional[Path] = None
    png_path: Optional[Path] = None
    warnings: list = field(default_factory=list)
    n_samples: list = field(default_factory=list)   # valid perpendicular samples per point
    axis: str = ""              # "auto" | "x" | "y" | "points"


def profile(session, scan, calibration=None, p0=None, p1=None, width_px: int = 5,
            axis: str = "auto", label: Optional[str] = None) -> Profile:
    warnings = []
    width_px = int(width_px)
    if width_px < 1:
        raise ValueError("width_px must be >= 1 (pixels averaged across the line)")
    if (p0 is None) != (p1 is None):
        raise ValueError("give both p0 and p1 (camera px [x, y]), or neither to use axis=")
    if axis not in AXES:
        raise ValueError(f"axis must be one of {AXES} (it is used when p0/p1 are not given)")

    relief = np.asarray(scan.relief, dtype=np.float64)
    valid = np.asarray(scan.relief_mask, bool) & np.isfinite(relief)
    H, W = relief.shape

    if p0 is not None:
        a, b = _point(p0, "p0"), _point(p1, "p1")
        used_axis = "points"
    else:
        # zone scans are datum-free: the base plane is at 0, find the part around it
        a, b, seg_warn = _auto_line(relief, axis, (H, W),
                                    ref=0.0 if is_datum_free(scan) else None)
        warnings += seg_warn
        used_axis = axis

    # rule 6: mm only with a calibration bound to this scan's geometry
    cal = calibration
    if cal is not None and not rules.calibration_matches(cal, scan.geometry_id):
        warnings.append(f"calibration {cal.id} is for geometry {cal.geometry_id}, "
                        f"not {scan.geometry_id}; ignored (rule 6)")
        cal = None
    z_ok = cal is not None and cal.has_z
    xy_ok = cal is not None and cal.has_xy
    if cal is None:
        warnings.append(f"no calibration for geometry {scan.geometry_id}; height and "
                        "distance in px, not mm (rule 6)")
    else:
        if not z_ok:
            warnings.append("calibration has no depth (Z) fit; height in relief px (rule 6)")
        if not xy_ok:
            warnings.append("calibration has no lateral (XY) fit; distance in camera px")

    dist_px, h_px, n = sample_line(relief, valid, a, b, width_px)
    inside = _inside(a, b, dist_px, (H, W))
    if inside < 0.5:
        warnings.append(f"only {100 * inside:.0f}% of the line lies inside the "
                        f"{W}x{H} camera frame")
    finite = np.isfinite(h_px)
    if not finite.any():
        warnings.append("no valid relief along the line (shadow, unlit or off the part?)")
    elif finite.mean() < 0.5:
        warnings.append(f"only {100 * finite.mean():.0f}% of the line has valid relief")

    height = cal.mm_per_px * h_px + cal.intercept_mm if z_ok else h_px
    distance = dist_px / cal.lateral_px_per_mm if xy_ok else dist_px
    uh, ud = ("mm" if z_ok else "px"), ("mm" if xy_ok else "px")
    tier = TIER_LATERAL if (z_ok and xy_ok) else TIER_DEPTH if z_ok else TIER_RELIEF
    stats = profile_stats(height, uh)

    prof = Profile(scan_id=scan.id, p0=[round(float(a[0]), 2), round(float(a[1]), 2)],
                   p1=[round(float(b[0]), 2), round(float(b[1]), 2)], width_px=width_px,
                   distance=[round(float(d), 4) for d in distance],
                   height=[round(float(h), 5) if np.isfinite(h) else float("nan") for h in height],
                   units_distance=ud, units_height=uh, tier=tier, stats=stats,
                   warnings=warnings, n_samples=[int(k) for k in n], axis=used_axis)
    _write_outputs(session, scan, prof, relief, valid, label or f"{scan.label}_profile")
    return prof


# ------------------------------------------------------------- geometry

def _point(p, name) -> np.ndarray:
    try:
        v = np.asarray(p, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        v = np.array([])
    if v.size != 2 or not np.all(np.isfinite(v)):
        raise ValueError(f"{name} must be a camera pixel [x, y], got {p!r}")
    return v


def _auto_line(relief, axis: str, shape, ref=None):
    """Line through the object blob: principal axis ('auto') or x / y,
    spanning the blob extent plus a margin of base plane on both ends.
    ref: the base level of a datum-free (zone) relief, see segment.object_blob.
    -> (a, b, segmentation warnings)."""
    try:
        blob, _, info = object_blob(relief, ref=ref)
    except SegmentationError as e:
        raise ValueError(f"profile axis={axis!r} places the line on the object, but none "
                         f"was found ({e}); pass p0 and p1 (camera px) instead") from e
    v, u = np.nonzero(blob)
    c = np.array([u.mean(), v.mean()])
    if axis == "auto":
        pts = np.column_stack([u - c[0], v - c[1]])
        _, vecs = np.linalg.eigh(pts.T @ pts / len(pts))
        d = vecs[:, 1]                                   # largest eigenvalue = long axis
        if d[0] < 0 or (abs(d[0]) < 1e-12 and d[1] < 0):
            d = -d                                       # run left -> right (top -> bottom)
    else:
        d = np.array([1.0, 0.0]) if axis == "x" else np.array([0.0, 1.0])
    proj = (u - c[0]) * d[0] + (v - c[1]) * d[1]
    lo, hi = float(proj.min()), float(proj.max())
    margin = max(MARGIN_MIN_PX, MARGIN_FRAC * (hi - lo))
    a, b = c + (lo - margin) * d, c + (hi + margin) * d
    clipped = _clip_segment(a, b, shape)
    if clipped is None:                                  # cannot happen for an in-frame blob
        raise ValueError("profile line falls outside the camera frame")
    return clipped[0], clipped[1], list(info.get("warnings", []))


def _clip_segment(a, b, shape):
    """Clip a -> b to [0, W-1] x [0, H-1] (Liang-Barsky)."""
    H, W = shape
    d = b - a
    t0, t1 = 0.0, 1.0
    for k, lim in ((0, W - 1), (1, H - 1)):
        if abs(d[k]) < 1e-12:
            if a[k] < 0 or a[k] > lim:
                return None
            continue
        ta, tb = (0 - a[k]) / d[k], (lim - a[k]) / d[k]
        t0, t1 = max(t0, min(ta, tb)), min(t1, max(ta, tb))
    if t1 <= t0:
        return None
    return a + t0 * d, a + t1 * d


def _inside(a, b, dist_px, shape) -> float:
    H, W = shape
    L = max(float(np.hypot(*(b - a))), 1e-12)
    t = dist_px / L
    x, y = a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])
    return float(np.mean((x >= 0) & (x <= W - 1) & (y >= 0) & (y <= H - 1)))


# -------------------------------------------------------------- sampling

def sample_line(relief, valid, p0, p1, width_px: int = 5):
    """-> (distance px from p0, height px (NaN where invalid), n valid per point).

    Points every ~1 px along p0 -> p1; at each, `width_px` bilinear samples
    1 px apart across the line are averaged, ignoring invalid ones."""
    a, b = np.asarray(p0, np.float64), np.asarray(p1, np.float64)
    L = float(np.hypot(*(b - a)))
    if L < 1.0:
        raise ValueError(f"profile line is {L:.2f} px long; p0 and p1 must be >= 1 px apart")
    n = int(math.floor(L)) + 1
    t = np.linspace(0.0, L, n)
    ux, uy = (b - a) / L
    nx, ny = -uy, ux
    off = np.arange(width_px, dtype=np.float64) - (width_px - 1) / 2.0
    X = a[0] + t[:, None] * ux + off[None, :] * nx
    Y = a[1] + t[:, None] * uy + off[None, :] * ny
    s = bilinear_nan(relief, valid, X, Y)
    cnt = np.isfinite(s).sum(axis=1)
    with np.errstate(invalid="ignore"):
        h = np.where(cnt > 0, np.nansum(s, axis=1) / np.maximum(cnt, 1), np.nan)
    return t, h, cnt


def bilinear_nan(img, valid, X, Y):
    """Bilinear interpolation that renormalises over valid neighbours;
    NaN where less than half the interpolation weight is valid."""
    H, W = img.shape
    x0, y0 = np.floor(X), np.floor(Y)
    fx, fy = X - x0, Y - y0
    x0, y0 = x0.astype(np.intp), y0.astype(np.intp)
    acc = np.zeros(X.shape)
    wsum = np.zeros(X.shape)
    for dx, dy, w in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                      (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        xi, yi = x0 + dx, y0 + dy
        inb = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
        xc, yc = np.clip(xi, 0, W - 1), np.clip(yi, 0, H - 1)
        ok = inb & valid[yc, xc]
        acc += np.where(ok, w * np.where(ok, img[yc, xc], 0.0), 0.0)
        wsum += np.where(ok, w, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(wsum >= 0.5, acc / np.maximum(wsum, 1e-12), np.nan)


# ----------------------------------------------------------------- stats

def _rsigma(v) -> float:
    return 1.4826 * float(np.median(np.abs(v - np.median(v)))) if v.size else float("nan")


def two_level_split(values):
    """Optimal 1-D 2-means cut (Otsu on the sorted values) -> (low, high) arrays."""
    v = np.sort(np.asarray(values, np.float64))
    n = v.size
    c1 = np.cumsum(v)
    c2 = np.cumsum(v * v)
    k = np.arange(1, n)                              # low = v[:k], high = v[k:]
    sse_lo = c2[k - 1] - c1[k - 1] ** 2 / k
    s_hi, s2_hi = c1[-1] - c1[k - 1], c2[-1] - c2[k - 1]
    sse_hi = s2_hi - s_hi ** 2 / (n - k)
    cut = int(k[np.argmin(sse_lo + sse_hi)])
    return v[:cut], v[cut:]


def profile_stats(height, units: str) -> dict:
    """min/max/range/median of the valid heights, plus the 2-level step.

    step_height is None unless the split is a real step: both levels hold
    enough samples and their medians differ by more than STEP_MIN_SEP of the
    noisier level's robust sigma (a flat or ramped line splits too, but
    with a separation of only ~2-3 sigma)."""
    h = np.asarray(height, np.float64)
    v = h[np.isfinite(h)]
    r = lambda x: None if x is None or not np.isfinite(x) else round(float(x), 4)
    out = {"units": units, "n_valid": int(v.size), "n_points": int(h.size),
           "min": None, "max": None, "range": None, "median": None,
           "step_height": None, "step_low": None, "step_high": None,
           "step_n_low": 0, "step_n_high": 0, "step_sep_sigma": None}
    if v.size == 0:
        return out
    out.update(min=r(v.min()), max=r(v.max()), range=r(v.max() - v.min()),
               median=r(np.median(v)))
    if v.size < 6:
        return out
    lo, hi = two_level_split(v)
    need = max(3, int(math.ceil(STEP_MIN_FRAC * v.size)))
    if lo.size < need or hi.size < need:
        return out
    m_lo, m_hi = float(np.median(lo)), float(np.median(hi))
    noise = max(_rsigma(lo), _rsigma(hi), 1e-9)
    sep = (m_hi - m_lo) / noise
    out.update(step_n_low=int(lo.size), step_n_high=int(hi.size),
               step_sep_sigma=round(min(sep, 1e6), 1))
    if sep >= STEP_MIN_SEP:
        out.update(step_height=r(m_hi - m_lo), step_low=r(m_lo), step_high=r(m_hi))
    return out


# --------------------------------------------------------------- outputs

def _write_outputs(session, scan, prof: Profile, relief, valid, label: str) -> None:
    store = session.store
    out = store.output_path("profiles", label, ".csv")
    ud, uh = prof.units_distance, prof.units_height
    rules.ensure_new_path(out)
    with open(out, "x", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([f"distance_{ud}", f"height_{uh}", "n_samples"])
        for d, h, k in zip(prof.distance, prof.height, prof.n_samples):
            w.writerow([f"{d:.4f}", "nan" if not np.isfinite(h) else f"{h:.5f}", k])
    prof.csv_path = out
    store.record_output("profiles", out, label=label, geometry_id=scan.geometry_id,
                        source=scan.path)
    png = plot_png(prof, relief, valid, title=f"{scan.label}  ({scan.id})")
    if png is not None:
        p = out.with_suffix(".png")
        rules.ensure_new_path(p)
        with open(p, "xb") as f:
            f.write(png)
        prof.png_path = p
        store.record_output("profiles", p, label=label, geometry_id=scan.geometry_id,
                            source=scan.path)


def _nice_ticks(lo: float, hi: float, n: int = 6):
    span = hi - lo
    raw = span / max(n, 1)
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    start = math.ceil(lo / step - 1e-9) * step
    ticks = np.arange(start, hi + step * 1e-6, step)
    return ticks, step


def _fmt(v: float, step: float) -> str:
    """Just enough decimals to tell neighbouring ticks apart (2.5-steps need one more)."""
    e = math.floor(math.log10(step))
    dec = max(0, -e) + (1 if round(step / 10 ** e, 6) == 2.5 else 0)
    return f"{v:.{min(dec, 4)}f}"


def plot_png(prof: Profile, relief=None, valid=None, title: str = "") -> Optional[bytes]:
    """Height-vs-distance plot (cv2; matplotlib is not a dependency), with a
    thumbnail of the relief showing where the line runs. None without cv2."""
    try:
        import cv2
    except ImportError:
        return None
    Wc, Hc = 1140, 540
    L, T, R, B = 90, 50, 770, 470                 # plot box
    img = np.full((Hc, Wc, 3), 255, np.uint8)
    black, grid, curve, lvl = (0, 0, 0), (228, 228, 228), (190, 90, 20), (0, 140, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    d = np.asarray(prof.distance, np.float64)
    h = np.asarray(prof.height, np.float64)
    fin = np.isfinite(h)
    ud, uh = prof.units_distance, prof.units_height

    x_lo, x_hi = (float(d.min()), float(d.max())) if d.size else (0.0, 1.0)
    if x_hi <= x_lo:
        x_hi = x_lo + 1.0
    if fin.any():
        y_lo, y_hi = float(h[fin].min()), float(h[fin].max())
        pad = 0.08 * (y_hi - y_lo) if y_hi > y_lo else 0.5
        y_lo, y_hi = y_lo - pad, y_hi + pad
    else:
        y_lo, y_hi = -1.0, 1.0
    X = lambda x: int(round(L + (x - x_lo) / (x_hi - x_lo) * (R - L)))
    Y = lambda y: int(round(B - (y - y_lo) / (y_hi - y_lo) * (B - T)))

    xt, xs = _nice_ticks(x_lo, x_hi, 8)
    yt, ys = _nice_ticks(y_lo, y_hi, 6)
    for x in xt:
        cv2.line(img, (X(x), T), (X(x), B), grid, 1)
        s = _fmt(x, xs)
        (tw, _), _ = cv2.getTextSize(s, font, 0.42, 1)
        cv2.putText(img, s, (X(x) - tw // 2, B + 18), font, 0.42, black, 1, cv2.LINE_AA)
    for y in yt:
        cv2.line(img, (L, Y(y)), (R, Y(y)), grid, 1)
        s = _fmt(y, ys)
        (tw, th), _ = cv2.getTextSize(s, font, 0.42, 1)
        cv2.putText(img, s, (L - tw - 6, Y(y) + th // 2), font, 0.42, black, 1, cv2.LINE_AA)
    cv2.rectangle(img, (L, T), (R, B), black, 1)

    st = prof.stats
    if st.get("step_height") is not None:
        for yv in (st["step_low"], st["step_high"]):
            for x0 in range(L, R, 12):
                cv2.line(img, (x0, Y(yv)), (min(x0 + 6, R), Y(yv)), lvl, 1)
    # the curve, broken wherever the relief is invalid
    edges = np.flatnonzero(np.diff(np.r_[0, fin.astype(np.int8), 0]))
    for s0, s1 in zip(edges[::2], edges[1::2]):
        pts = np.array([[X(d[i]), Y(h[i])] for i in range(s0, s1)], np.int32)
        if len(pts) == 1:
            cv2.circle(img, tuple(int(v) for v in pts[0]), 1, curve, -1)
        else:
            cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, curve, 2, cv2.LINE_AA)
    if not fin.any():
        cv2.putText(img, "no valid relief along this line", (L + 20, (T + B) // 2), font,
                    0.7, (0, 0, 200), 2, cv2.LINE_AA)

    xl = f"distance along line ({ud})"
    (tw, _), _ = cv2.getTextSize(xl, font, 0.5, 1)
    cv2.putText(img, xl, ((L + R - tw) // 2, B + 42), font, 0.5, black, 1, cv2.LINE_AA)
    yl = f"height ({uh})" + ("" if uh == "mm" else " - relief, uncalibrated")
    (tw, th), bl = cv2.getTextSize(yl, font, 0.5, 1)
    lab = np.full((th + bl + 6, tw + 6, 3), 255, np.uint8)
    cv2.putText(lab, yl, (3, th + 3), font, 0.5, black, 1, cv2.LINE_AA)
    lab = cv2.rotate(lab, cv2.ROTATE_90_COUNTERCLOCKWISE)
    lh, lw = lab.shape[:2]
    y0 = max(0, (T + B) // 2 - lh // 2)
    lh = min(lh, Hc - y0)
    img[y0:y0 + lh, 8:8 + lw] = lab[:lh]
    cv2.putText(img, (title or prof.scan_id)[:90], (L, 22), font, 0.5, black, 1, cv2.LINE_AA)
    cv2.putText(img, f"profile, width {prof.width_px} px, tier {prof.tier}", (L, 40), font,
                0.42, (90, 90, 90), 1, cv2.LINE_AA)

    # side panel: where the line is, and the numbers
    px0, py0, tw_, th_ = 790, 50, 330, 248
    if relief is not None and relief.size:
        a = np.asarray(relief, np.float64)
        m = np.asarray(valid, bool) if valid is not None else np.isfinite(a)
        vals = a[m]
        lo, hi = (np.percentile(vals, 1), np.percentile(vals, 99)) if vals.size else (0, 1)
        g = np.clip((np.nan_to_num(a, nan=lo) - lo) / (hi - lo + 1e-12), 0, 1)
        g8 = (g * 255).astype(np.uint8)
        g8[~m] = 0
        sc = min(tw_ / a.shape[1], th_ / a.shape[0])
        tn = cv2.resize(g8, (int(a.shape[1] * sc), int(a.shape[0] * sc)),
                        interpolation=cv2.INTER_AREA)
        tn = cv2.cvtColor(tn, cv2.COLOR_GRAY2BGR)
        q0 = (int(round(prof.p0[0] * sc)), int(round(prof.p0[1] * sc)))
        q1 = (int(round(prof.p1[0] * sc)), int(round(prof.p1[1] * sc)))
        cv2.line(tn, q0, q1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.circle(tn, q0, 4, (0, 200, 0), -1)
        cv2.putText(tn, "p0", (q0[0] + 5, q0[1] - 5), font, 0.4, (0, 200, 0), 1, cv2.LINE_AA)
        img[py0:py0 + tn.shape[0], px0:px0 + tn.shape[1]] = tn
        cv2.rectangle(img, (px0 - 1, py0 - 1), (px0 + tn.shape[1], py0 + tn.shape[0]), black, 1)
        ty = py0 + tn.shape[0] + 24
    else:
        ty = py0
    f = lambda k: "n/a" if st.get(k) is None else f"{st[k]:.3f} {uh}"
    lines = [f"p0 {prof.p0}  p1 {prof.p1} (camera px)",
             f"min {f('min')}   max {f('max')}", f"median {f('median')}",
             f"range {f('range')}",
             "step " + (f"{st['step_height']:.3f} {uh}  ({st['step_low']:.3f} -> "
                        f"{st['step_high']:.3f})" if st.get("step_height") is not None
                        else "none found"),
             f"valid {st['n_valid']}/{st['n_points']} points"]
    for i, s in enumerate(lines):
        cv2.putText(img, s, (px0, ty + 20 * i), font, 0.42, black, 1, cv2.LINE_AA)
    for i, wmsg in enumerate(prof.warnings[:3]):
        cv2.putText(img, wmsg[:52], (px0, ty + 20 * (len(lines) + 1 + i)), font, 0.38,
                    (0, 0, 190), 1, cv2.LINE_AA)
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes() if ok else None

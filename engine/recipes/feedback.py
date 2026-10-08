"""
Projected feedback: after the verdict the DMD paints it ON the part, so the operator sees
where to act (a misaligned tube, a missing part, a dent) without reading a screen.

    PASS     the zones outlined in the pass colour (green)
    FAIL     failing zones filled (fill_level x the fail colour) and outlined red, a cross at
             every fault point; passing zones still outlined green
    INVALID  the zones that could not be measured outlined amber
    text     optional short text ("FAIL tube_right") next to the zones

Camera px -> native DMD (col, row), from the scan's OWN decoded correspondences (each camera
px knows which DMD row AND column lit it: scan axis xy), so the paint lands on the surfaces
the camera saw, at whatever depth (a tube 20 mm proud of its fixture and the fixture beside
it each get their own light; no single plane is assumed):
  zones    the zone's decoded camera px rasterised into the DMD (+ a 5 px close): the fill,
           and its contour the outline. Projector-px / grid-cell zones are drawn as given.
  check    a local affine (least squares, trimmed like the end-face aim's fit) on 16 px
           tiles of the zone: the median tile's p95 residual (DMD px) must stay under
           feedback.max_fit_px, else the decode there is not trusted and the zone is not
           drawn (refused, with the number); nothing is projected when no zone can be drawn.
           (Tiles across a depth step or a curved surface fit worse: the median, not the
           worst tile, judges the decode.)
  crosses  centred on the fault px's own decoded DMD point, arms drawn in camera space and
           mapped through the local affine at that point; text likewise (upright in the
           scene, whatever mirror / rotation the optics add, and the diamond DMD's 2:1 rows).

Projection goes through the normal rig: HDMI window only, raw colour (no duty / gamma
chain), polarity MEASURED first (raw black vs raw white; an inverting path gets the
inverted image, so the colours land as drawn), then a pumped hold (never a sleep) that an
ABORT (the run's cancel event) ends at once; the rig closes after it, which clears it.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

DMD_SHAPE = (1140, 912)                  # rows, cols
MIN_FIT_PX = 12
OVERLAY_W = 960                          # camera overlay width in the run folder, px


class FeedbackRefused(ValueError):
    pass


def _cv():
    import cv2
    return cv2


# ------------------------------------------------------------------ camera -> DMD

def fit_points(fx, fy, U, V, trim_iters: int = 2, trim_k: float = 4.0,
               trim_min_px: float = 0.5) -> Optional[dict]:
    """Trimmed least-squares affine camera px (fx, fy) -> DMD (U, V) -> {A (2x3), rms_px,
    p95_px (over the kept px), n_px, n_outliers} or None (fewer than MIN_FIT_PX px)."""
    if len(fx) < MIN_FIT_PX:
        return None
    A = np.c_[fx, fy, np.ones(len(fx))].astype(np.float64)
    U = np.asarray(U, np.float64)
    V = np.asarray(V, np.float64)
    keep = np.ones(len(fx), bool)
    for it in range(trim_iters + 1):
        au = np.linalg.lstsq(A[keep], U[keep], rcond=None)[0]
        av = np.linalg.lstsq(A[keep], V[keep], rcond=None)[0]
        e = np.hypot(U - A @ au, V - A @ av)
        if it == trim_iters:
            break
        med = float(np.median(e[keep]))
        mad = float(np.median(np.abs(e[keep] - med)))
        new = e <= med + max(trim_k * 1.4826 * mad, trim_min_px)
        if new.sum() < MIN_FIT_PX or np.array_equal(new, keep):
            break
        keep = new
    return {"A": np.stack([au, av]), "rms_px": float(np.sqrt(np.mean(e[keep] ** 2))),
            "p95_px": float(np.percentile(e[keep], 95)), "n_px": int(keep.sum()),
            "n_outliers": int((~keep).sum()), "units": "native DMD px per camera px"}


def fit_affine(sv, mask: np.ndarray, max_px: int = 40000, **kw) -> Optional[dict]:
    """Camera px -> DMD (col, row) affine on the decoded px of mask (fit_points)."""
    if not sv.has_xy:
        return None
    ok = mask & np.isfinite(sv.corr_x) & np.isfinite(sv.corr_y)
    fy, fx = np.nonzero(ok)
    if len(fx) > max_px:
        sel = np.random.default_rng(0).choice(len(fx), max_px, replace=False)
        fx, fy = fx[sel], fy[sel]
    return fit_points(fx, fy, sv.corr_x[fy, fx], sv.corr_y[fy, fx], **kw)


def apply(A, pts) -> np.ndarray:
    pts = np.asarray(pts, np.float64).reshape(-1, 2)
    return np.c_[pts, np.ones(len(pts))] @ np.asarray(A, np.float64).T


def _fit_public(f: Optional[dict]) -> Optional[dict]:
    if f is None:
        return None
    return {"p95_px": round(f["p95_px"], 3), "rms_px": round(f["rms_px"], 3), "n_px": f["n_px"],
            "n_outliers": f["n_outliers"], "A": np.round(f["A"], 6).tolist(),
            "units": "native DMD px (col, row); A maps camera px [x, y, 1]"}


def _decoded(sv, mask) -> np.ndarray:
    return mask & np.isfinite(sv.corr_x) & np.isfinite(sv.corr_y)


def local_fit(sv, mask: np.ndarray, centre, radius: int = 12) -> Optional[dict]:
    """The trimmed affine on the decoded px of mask within `radius` camera px of centre."""
    if not sv.has_xy:
        return None
    H, W = mask.shape
    cx, cy = int(round(centre[0])), int(round(centre[1]))
    x0, x1 = max(cx - radius, 0), min(cx + radius + 1, W)
    y0, y1 = max(cy - radius, 0), min(cy + radius + 1, H)
    if x1 <= x0 or y1 <= y0:
        return None
    sl = (slice(y0, y1), slice(x0, x1))
    ok = mask[sl] & np.isfinite(sv.corr_x[sl]) & np.isfinite(sv.corr_y[sl])
    fy, fx = np.nonzero(ok)
    return fit_points(fx + x0, fy + y0, sv.corr_x[sl][fy, fx], sv.corr_y[sl][fy, fx])


def zone_map(sv, mask: np.ndarray, tile: int = 16, max_tiles: int = 64) -> dict:
    """A zone's decoded px -> {dmd_mask (DMD rows x cols), local_p95_px (the median over
    tiles of each tile's trimmed local-affine p95 residual), local_worst_px, n_tiles, n_px, fit
    (one affine over the zone, for reference)}; dmd_mask None when fewer than MIN_FIT_PX px
    decoded."""
    cv2 = _cv()
    dec = _decoded(sv, mask)
    n = int(dec.sum())
    out = {"dmd_mask": None, "local_p95_px": None, "local_worst_px": None, "n_tiles": 0,
           "n_px": n, "fit": None}
    if n < MIN_FIT_PX:
        return out
    ys, xs = np.nonzero(dec)
    c = np.clip(np.round(sv.corr_x[ys, xs]), 0, DMD_SHAPE[1] - 1).astype(np.intp)
    r = np.clip(np.round(sv.corr_y[ys, xs]), 0, DMD_SHAPE[0] - 1).astype(np.intp)
    dm = np.zeros(DMD_SHAPE, np.uint8)
    dm[r, c] = 1
    dm = cv2.morphologyEx(dm, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    out["dmd_mask"] = dm.astype(bool)
    U, V = sv.corr_x[ys, xs], sv.corr_y[ys, xs]
    key = (ys // tile) * 100000 + (xs // tile)
    order = np.argsort(key, kind="stable")
    uk, start, counts = np.unique(key[order], return_index=True, return_counts=True)
    groups = [order[a:a + n] for a, n in zip(start, counts) if n >= MIN_FIT_PX * 2]
    if len(groups) > max_tiles:
        groups = [groups[i] for i in np.random.default_rng(0).choice(len(groups), max_tiles,
                                                                     replace=False)]
    p95 = []
    for g in groups:
        f = fit_points(xs[g], ys[g], U[g], V[g])
        if f is not None:
            p95.append(f["p95_px"])
    out["n_tiles"] = len(p95)
    out["local_p95_px"] = float(np.median(p95)) if p95 else None
    out["local_worst_px"] = float(np.max(p95)) if p95 else None
    out["fit"] = fit_affine(sv, dec)
    return out


def fault_dmd(sv, px, zone_mask) -> tuple:
    """A fault px -> (its DMD point, the local affine for the arms) or (None, None). The
    point is the mean decoded correspondence of the zone's px within 2 px (the surface the
    camera saw there), else the local affine's image of the px."""
    H, W = zone_mask.shape
    x, y = float(px[0]), float(px[1])
    ix, iy = int(round(x)), int(round(y))
    dec_all = np.isfinite(sv.corr_x) & np.isfinite(sv.corr_y)
    f = local_fit(sv, zone_mask & dec_all, (x, y), 12) or local_fit(sv, dec_all, (x, y), 16)
    if f is None:
        return None, None
    y0, y1, x0, x1 = max(iy - 2, 0), min(iy + 3, H), max(ix - 2, 0), min(ix + 3, W)
    near = (zone_mask & dec_all)[y0:y1, x0:x1]
    if near.any():
        cx = float(np.mean(sv.corr_x[y0:y1, x0:x1][near]))
        cy = float(np.mean(sv.corr_y[y0:y1, x0:x1][near]))
        return np.array([cx, cy]), f
    return apply(f["A"], [[x, y]])[0], f


# ------------------------------------------------------------------ rendering

def _poly(img, pts, colour, line_px: int, fill: Optional[tuple] = None) -> None:
    cv2 = _cv()
    q = np.round(np.asarray(pts) * 16).astype(np.int32)
    if fill is not None:
        cv2.fillPoly(img, [q], fill, lineType=cv2.LINE_AA, shift=4)
    cv2.polylines(img, [q], True, colour, int(line_px), lineType=cv2.LINE_AA, shift=4)


def _cross_at(img, f, centre_dmd, colour, cross_px: int, line_px: int) -> None:
    """Cross centred on centre_dmd, arms along camera x / y mapped by f's linear part."""
    cv2 = _cv()
    A = np.asarray(f["A"], float)
    sx = float(np.hypot(A[0][0], A[1][0])) or 1.0           # DMD px per camera px (x arm)
    L = cross_px / sx
    ax_x, ax_y = A[:, 0] * L, A[:, 1] * L
    c = np.asarray(centre_dmd, float)
    for a, b in ((c - ax_x, c + ax_x), (c - ax_y, c + ax_y)):
        cv2.line(img, tuple(int(v) for v in np.round(a * 16)), tuple(int(v) for v in np.round(b * 16)),
                 colour, int(line_px), lineType=cv2.LINE_AA, shift=4)


def _text(img, A, anchor_px, text: str, colour, text_px: int) -> bool:
    """Text rendered upright in CAMERA space at anchor (bottom-left), warped to the DMD."""
    cv2 = _cv()
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = text_px / 22.0
    thick = max(2, int(round(text_px / 10)))
    (tw, th), base = cv2.getTextSize(text, font, scale, thick)
    pad = 4
    tile = np.zeros((th + base + 2 * pad, tw + 2 * pad), np.uint8)
    cv2.putText(tile, text, (pad, pad + th), font, scale, 255, thick, cv2.LINE_AA)
    x0, y1 = float(anchor_px[0]), float(anchor_px[1])
    y0 = y1 - tile.shape[0]
    T = np.array([[1.0, 0.0, x0], [0.0, 1.0, y0], [0.0, 0.0, 1.0]])   # tile px -> camera px
    M = (np.vstack([np.asarray(A, float), [0, 0, 1]]) @ T)[:2]        # -> DMD
    warped = cv2.warpAffine(tile, M, (DMD_SHAPE[1], DMD_SHAPE[0]), flags=cv2.INTER_LINEAR)
    m = warped > 100
    if not m.any():
        return False
    img[m] = colour
    return True


def colour_for(verdict: str, fb: dict) -> tuple:
    return tuple(int(c) for c in {"PASS": fb["pass_colour"], "FAIL": fb["fail_colour"]}.get(
        verdict, fb["invalid_colour"]))


def render(zone_views: dict, zone_results: list, sv, fb: dict, verdict: str) -> tuple:
    """-> (DMD image (1140, 912, 3) uint8 BGR or None, info dict). zone_results: the
    result's zones list (id, verdict, checks with fault_points in camera px)."""
    cv2 = _cv()
    img = np.zeros(DMD_SHAPE + (3,), np.uint8)
    info = {"verdict": verdict, "zones": [], "crosses": [], "text": None, "refused": None,
            "warnings": [], "units": {"dmd": "native DMD px (col, row)",
                                      "local_p95_px": "DMD px"}}
    if not sv.has_xy:
        info["refused"] = ("the scan decoded columns only: camera -> DMD needs rows and columns "
                           "(scan axis xy)")
        return None, info
    line, maxfit = int(fb["line_px"]), float(fb["max_fit_px"])
    usable, outline_cam = {}, []
    union = np.zeros(sv.shape, bool)
    order = sorted(zone_results, key=lambda r: {"PASS": 0, "INVALID": 1, "FAIL": 2}.get(
        r["verdict"], 1))                                    # FAIL drawn last (on top)
    for zr in order:
        zv = zone_views.get(zr["id"])
        if zv is None or not zv.feedback or (zv.datum and not zr.get("checks")):
            continue
        col = colour_for(zr["verdict"], fb)
        fill = None
        if zr["verdict"] == "FAIL" and fb["fill_fail"]:
            fill = tuple(int(round(c * float(fb["fill_level"]))) for c in col)
        rec = {"id": zv.id, "verdict": zr["verdict"], "drawn": False, "refused": None,
               "local_p95_px": None, "n_tiles": 0, "n_px": 0, "fit": None, "dmd_px": 0}
        dm = None
        if zv.units == "projector_px" and zv.dmd_mask is not None:
            dm = zv.dmd_mask
        else:
            zm = zone_map(sv, zv.mask)
            rec.update(local_p95_px=None if zm["local_p95_px"] is None
                       else round(zm["local_p95_px"], 3),
                       local_worst_px=None if zm["local_worst_px"] is None
                       else round(zm["local_worst_px"], 3), n_tiles=zm["n_tiles"],
                       n_px=zm["n_px"], fit=_fit_public(zm["fit"]))
            if zm["dmd_mask"] is None:
                rec["refused"] = (f"only {zm['n_px']} decoded px in the zone: no camera -> DMD "
                                  "mapping")
            elif zm["local_p95_px"] is not None and zm["local_p95_px"] > maxfit:
                rec["refused"] = (f"the zone's camera -> DMD map is not consistent: local affine "
                                  f"residual p95 {zm['local_p95_px']:.2f} DMD px > {maxfit:g} "
                                  "(feedback.max_fit_px) - not drawn")
            else:
                dm = zm["dmd_mask"]
        if dm is not None and dm.any():
            if fill is not None:
                img[dm] = fill
            cs, _ = cv2.findContours(dm.astype(np.uint8), cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cs, -1, col, line, lineType=cv2.LINE_AA)
            rec.update(drawn=True, dmd_px=int(dm.sum()))
            usable[zv.id] = zv
            union |= zv.mask
            outline_cam += [np.asarray(p) for p in zv.outline_px]
        if rec["refused"]:
            info["warnings"].append(f"zone {zv.id}: {rec['refused']}")
        info["zones"].append(rec)
    col = colour_for("FAIL", fb)
    for zr in zone_results:                                  # crosses on top of the fills
        zv = usable.get(zr["id"])
        if zv is None:
            continue
        for ch in zr.get("checks", []):
            if ch["verdict"] != "FAIL":
                continue
            for fp in ch.get("fault_points", []):
                if fp.get("px") is None:
                    continue
                c, f = fault_dmd(sv, fp["px"], zv.mask)
                if c is None:
                    info["warnings"].append(f"zone {zv.id}: fault at {fp['px']} has no decoded "
                                            "px nearby: no cross")
                    continue
                _cross_at(img, f, c, col, int(fb["cross_px"]), line)
                info["crosses"].append({"zone": zr["id"], "check": ch["id"],
                                        "px": [round(float(v), 2) for v in fp["px"]],
                                        "dmd": [round(float(v), 2) for v in c],
                                        "label": fp.get("label", "")})
    if not usable:
        info["refused"] = ("no zone could be mapped onto the DMD: " + "; ".join(info["warnings"])
                           if info["warnings"] else "nothing to draw")
        return None, info
    if fb["text"] and outline_cam:
        pts = np.concatenate(outline_cam)
        anchor = [float(pts[:, 0].min()), float(pts[:, 1].min()) - 6.0]
        dec = np.isfinite(sv.corr_x) & np.isfinite(sv.corr_y)
        g = fit_affine(sv, union) or local_fit(sv, dec, (anchor[0] + 40, anchor[1] - 10), 40)
        if g is not None:
            fail_z = [z["id"] for z in zone_results if z["verdict"] == "FAIL"]
            txt = verdict + ((" " + ", ".join(fail_z[:3])) if fail_z else "")
            if _text(img, g["A"], anchor, txt, colour_for(verdict, fb), int(fb["text_px"])):
                info["text"] = txt
    return img, info


# ------------------------------------------------------------------ camera overlay (run folder)

def overlay(sv, frame, zone_views: dict, zone_results: list, fb: dict, verdict: str) -> np.ndarray:
    """Camera-space picture of the verdict: the flood (or the height map), zones outlined in
    their verdict colour, failing zones tinted, crosses at the fault points, zone labels."""
    cv2 = _cv()
    base = sv.flood
    if base is None:
        base = frame.uvh[..., 2]
    b = np.asarray(base, np.float32)
    fin = np.isfinite(b)
    lo, hi = (np.percentile(b[fin], [1, 99.5]) if fin.any() else (0.0, 1.0))
    g = np.clip((np.nan_to_num(b, nan=lo) - lo) / max(hi - lo, 1e-6), 0, 1)
    vis = cv2.cvtColor((g * 200).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    tint = vis.copy()
    for zr in zone_results:
        zv = zone_views.get(zr["id"])
        if zv is not None and zr["verdict"] == "FAIL":
            tint[zv.mask] = colour_for("FAIL", fb)
    vis = cv2.addWeighted(vis, 0.7, tint, 0.3, 0)
    for zr in zone_results:
        zv = zone_views.get(zr["id"])
        if zv is None:
            continue
        col = colour_for(zr["verdict"], fb)
        for poly in zv.outline_px:
            _poly(vis, poly, col, 2)
        if zv.outline_px:
            p = np.concatenate(zv.outline_px)
            cv2.putText(vis, f"{zv.id} {zr['verdict']}",
                        (int(p[:, 0].min()), max(int(p[:, 1].min()) - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
        for ch in zr.get("checks", []):
            for fp in ch.get("fault_points", []) if ch["verdict"] == "FAIL" else []:
                if fp.get("px") is not None:
                    x, y = (int(round(v)) for v in fp["px"])
                    cv2.drawMarker(vis, (x, y), colour_for("FAIL", fb), cv2.MARKER_CROSS, 28, 2)
    cv2.putText(vis, verdict, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                colour_for(verdict, fb), 3, cv2.LINE_AA)
    h, w = vis.shape[:2]
    if w > OVERLAY_W:                            # a picture for people, not a measurement
        vis = cv2.resize(vis, (OVERLAY_W, int(round(h * OVERLAY_W / w))),
                         interpolation=cv2.INTER_AREA)
    return vis


def write_png(path, img) -> Optional[str]:
    cv2 = _cv()
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        return None
    with open(path, "xb") as f:                  # a run folder is new: never overwrite
        f.write(buf.tobytes())
    return str(path)


def read_png(path) -> np.ndarray:
    cv2 = _cv()
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None or img.shape[:2] != DMD_SHAPE:
        raise FeedbackRefused(f"{path} is not a {DMD_SHAPE[1]} x {DMD_SHAPE[0]} DMD image")
    return img


# ------------------------------------------------------------------ projection (the rig)

def project(s, preset, img: np.ndarray, hold_s: float, cancel=None, photo: bool = True) -> dict:
    """Show the feedback image through the normal rig: measured polarity, raw colour,
    photographed (photo), then a pumped hold of hold_s that `cancel` (the run's ABORT event)
    ends early. -> {inverted, hold_s, held_s, cancelled, photo (camera frame or None),
    warnings}."""
    from ..capture import zones as Z
    img = np.asarray(img)
    if img.shape[:2] != DMD_SHAPE or img.dtype != np.uint8:
        raise FeedbackRefused(f"the feedback image must be uint8 {DMD_SHAPE} (+ 3 colours)")
    hold = Z._hold(hold_s, preset, "recipe feedback hold")          # rule 3 before the rig
    Z._check_cancel(cancel)
    warnings: list = []
    held = 0.0
    with s.rig(preset) as sync:
        inverted, _lit = Z._measure_polarity(sync, warnings)
        sent = (255 - img) if inverted else img
        frame = sync.shoot(sent, raw=True) if photo else None
        if not photo:
            sync.projector.project(sent, raw=True)
        if hold > 0:
            held = Z._pumped_hold(sync.projector.window, hold, cancel)
    cancelled = held < hold - 1e-9
    if cancelled:
        warnings.append(f"feedback cleared after {held:.2f} of {hold:g} s")
    return {"inverted": bool(inverted), "hold_s": hold, "held_s": round(held, 3),
            "cancelled": cancelled, "photo": frame, "warnings": warnings}

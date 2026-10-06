"""
DMD images for an end face (native DMD (col, row); rows are half a column pitch tall, so a
round face is an ellipse ~2x taller in rows than in columns) and the ring-check measurement.

  ring_image     the aim ring (the face rim through the camera -> DMD affine) + a centre
                 cross (6 cols x 12 rows: about square on the part)
  face_mask      the face ellipse + margin: the targeted scan lights only this
  polar_specs /  concentric rings (period in rim radii: 0.08 ~ 0.84 mm on a 21 mm face,
  polar_image    0.25) and spokes (24 and 6 per turn), N steps each, + floods 255 / 0, in
                 face-normalised polar coordinates (the DMD ellipse mapped to the unit circle,
                 R = 1 at the rim)
  measure_ring   where the projected ring landed in the photo vs where the face is
"""

from __future__ import annotations

from typing import Optional

import numpy as np

DMD_SHAPE = (1140, 912)                 # rows, cols
RING_PX = 3                             # ring line width, DMD px
CROSS = (6, 12)                         # cross half-lengths (cols, rows)
CROSS_PX = 2
RING_PERIODS_R = (0.08, 0.25)           # concentric ring periods, rim radii
SPOKES = (24, 6)                        # spokes per turn
POLAR_STEPS = 8


def ring_image(ring: np.ndarray, centre, offset=(0.0, 0.0), ring_px: int = RING_PX,
               cross=CROSS, cross_px: int = CROSS_PX, level: int = 255) -> np.ndarray:
    """uint8 DMD image: the ring polyline + centre cross, shifted by offset (cols, rows)."""
    import cv2
    img = np.zeros(DMD_SHAPE, np.uint8)
    off = np.asarray(offset if offset is not None else (0.0, 0.0), np.float64).reshape(2)
    pts = np.round(np.asarray(ring, np.float64) + off).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(img, [pts], True, int(level), int(ring_px))
    cc = np.asarray(centre, np.float64) + off
    if cross:
        for v in ((cross[0], 0), (0, cross[1])):
            a = tuple(int(x) for x in np.round(cc - v))
            b = tuple(int(x) for x in np.round(cc + v))
            cv2.line(img, a, b, int(level), int(cross_px))
    return img


def off_dmd_pct(dmd_ellipse: dict) -> float:
    """% of the face ellipse's area that falls outside the DMD (a face the projector can
    only partly light: that part is never measured)."""
    import cv2
    a, b = float(dmd_ellipse["a"]), float(dmd_ellipse["b"])
    total = np.pi * a * b / 4
    if not total > 0:
        return 100.0
    pad = int(np.ceil(max(a, b))) + 2
    H, W = DMD_SHAPE
    m = np.zeros((H + 2 * pad, W + 2 * pad), np.uint8)
    cv2.ellipse(m, ((dmd_ellipse["cx"] + pad, dmd_ellipse["cy"] + pad), (a, b),
                    dmd_ellipse["angle_deg"]), 1, -1)
    inside = float(m[pad:pad + H, pad:pad + W].sum())
    return round(100.0 * max(0.0, 1.0 - inside / max(float(m.sum()), 1.0)), 2)


def face_mask(dmd_ellipse: dict, margin_px: float = 6.0) -> np.ndarray:
    """bool DMD mask: the face ellipse grown by margin_px on each side."""
    import cv2
    m = np.zeros(DMD_SHAPE, np.uint8)
    cv2.ellipse(m, ((dmd_ellipse["cx"], dmd_ellipse["cy"]),
                    (dmd_ellipse["a"] + 2 * margin_px, dmd_ellipse["b"] + 2 * margin_px),
                    dmd_ellipse["angle_deg"]), 1, -1)
    return m.astype(bool)


def polar_coords(dmd_ellipse: dict) -> tuple:
    """(R, TH) over the DMD: the face ellipse mapped to the unit circle (R = 1 at the rim)."""
    H, W = DMD_SHAPE
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    c, s = np.cos(np.radians(dmd_ellipse["angle_deg"])), np.sin(np.radians(dmd_ellipse["angle_deg"]))
    dx, dy = xx - dmd_ellipse["cx"], yy - dmd_ellipse["cy"]
    a1 = (dx * c + dy * s) / (dmd_ellipse["a"] / 2)
    a2 = (-dx * s + dy * c) / (dmd_ellipse["b"] / 2)
    return np.hypot(a1, a2), np.arctan2(a2, a1)


def polar_specs(ring_periods=RING_PERIODS_R, spokes=SPOKES, steps: int = POLAR_STEPS,
                floods: bool = True) -> list:
    if int(steps) < 3:
        raise ValueError(f"polar steps must be >= 3, got {steps!r}")
    out = [{"kind": "rings", "period_r": float(T), "step": k, "steps": int(steps)}
           for T in (ring_periods or ()) for k in range(int(steps))]
    out += [{"kind": "spokes", "n": int(n), "step": k, "steps": int(steps)}
            for n in (spokes or ()) for k in range(int(steps))]
    if floods:
        out += [{"kind": "flood", "level": 255}, {"kind": "flood", "level": 0}]
    return out


def polar_image(spec: dict, R: np.ndarray, TH: np.ndarray) -> np.ndarray:
    k = spec["kind"]
    if k == "flood":
        return np.full(DMD_SHAPE, int(spec["level"]), np.uint8)
    ph = 2 * np.pi * spec["step"] / spec["steps"]
    if k == "rings":
        arg = 2 * np.pi * R / spec["period_r"] - ph
    elif k == "spokes":
        arg = spec["n"] * TH - ph
    else:
        raise ValueError(f"unknown polar pattern kind {k!r}")
    return np.clip(127.5 + 127.5 * np.cos(arg), 0, 255).astype(np.uint8)


def polar_patterns(dmd_ellipse: dict, specs: list) -> list:
    R, TH = polar_coords(dmd_ellipse)
    return [polar_image(sp, R, TH) for sp in specs]


# ---------------------------------------------------------------- ring check

def _norm_radius(cam_ellipse: dict, shape) -> np.ndarray:
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    c, s = np.cos(np.radians(cam_ellipse["angle_deg"])), np.sin(np.radians(cam_ellipse["angle_deg"]))
    dx, dy = xx - cam_ellipse["cx"], yy - cam_ellipse["cy"]
    return np.hypot((dx * c + dy * s) / (cam_ellipse["a"] / 2), (-dx * s + dy * c) / (cam_ellipse["b"] / 2))


def face_lit(flood: np.ndarray, dark: Optional[np.ndarray], cam_mask: np.ndarray,
             erode_px: int = 2) -> dict:
    """The face flood (light through the DMD face ellipse only) vs the located camera face
    mask. A face still in place is lit over the whole mask; where it moved away the camera
    sees background lit through other mirrors (parallax) - dark - and light spills outside
    the mask where the face moved into it. -> lit_pct (of the eroded mask), spill_pct (lit
    pixels outside the mask, % of the mask area)."""
    import cv2
    img = np.asarray(flood, np.float64) - (np.asarray(dark, np.float64) if dark is not None else 0)
    m = np.asarray(cam_mask, bool)
    k = max(int(erode_px), 1)
    inner = cv2.erode(m.astype(np.uint8), np.ones((2 * k + 1, 2 * k + 1), np.uint8)).astype(bool)
    if inner.sum() < 20:
        return {"found": False, "why": "face mask too small"}
    ref = float(np.percentile(img[inner], 90))
    if ref < 8:
        return {"found": False, "why": f"the face flood is not visible ({ref:.1f} DN over dark)"}
    thr = 0.4 * ref
    lit = img > thr
    ys, xs = np.nonzero(m)
    r = int(max(np.ptp(xs), np.ptp(ys)))
    x0, y0 = max(int(xs.min()) - r, 0), max(int(ys.min()) - r, 0)
    x1, y1 = min(int(xs.max()) + r + 1, m.shape[1]), min(int(ys.max()) + r + 1, m.shape[0])
    around = np.zeros_like(m)
    around[y0:y1, x0:x1] = True
    near = cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    return {"found": True, "lit_pct": round(100.0 * float(lit[inner].mean()), 2),
            "spill_pct": round(100.0 * float((lit & around & ~near).sum()) / max(int(m.sum()), 1), 2),
            "flood_p90_dn": round(ref, 2), "threshold_dn": round(thr, 2)}


def measure_ring(photo: np.ndarray, dark: Optional[np.ndarray], cam_ellipse: dict,
                 band=(0.6, 1.1), min_px: int = 20) -> dict:
    """Where the projected ring + cross landed (camera px, full frame) vs the face ellipse it
    was drawn for: bright ring pixels in a band of normalised radius around the expected rim
    (inside it: the outer half of the line falls off the face) -> ellipse fit (size ratio);
    cross = bright pixels inside 0.45 of the radius -> centroid = the centre offset (the ring
    when no cross is seen). A face that moved toward / away from the camera shifts both
    sideways (parallax); a face that slid in its own plane does not - face_lit catches that."""
    import cv2
    img = np.asarray(photo, np.float64) - (np.asarray(dark, np.float64) if dark is not None else 0)
    rn = _norm_radius(cam_ellipse, img.shape)
    zone = rn < band[1] + 0.4
    vals = img[zone]
    if vals.size < min_px:
        return {"found": False, "why": "the face is outside the photo"}
    lo, hi = np.percentile(vals, 5), np.percentile(vals, 99.5)
    if hi - lo < 8:
        return {"found": False, "why": f"no projected ring visible (contrast {hi - lo:.1f} DN)"}
    thr = lo + 0.5 * (hi - lo)
    bright = zone & (img > thr)
    ring = bright & (rn >= band[0]) & (rn <= band[1])
    ys, xs = np.nonzero(ring)
    out = {"found": False, "threshold_dn": float(thr), "contrast_dn": float(hi - lo),
           "ring_px": int(len(xs))}
    if len(xs) >= max(min_px, 5):
        (cx, cy), (a, b), ang = cv2.fitEllipse(np.c_[xs, ys].astype(np.float32))
        out.update({"found": True, "ring_ellipse": {"cx": float(cx), "cy": float(cy), "a": float(a),
                                                     "b": float(b), "angle_deg": float(ang)},
                    "centre_offset_px": [float(cx - cam_ellipse["cx"]), float(cy - cam_ellipse["cy"])],
                    "size_ratio": float((a + b) / (cam_ellipse["a"] + cam_ellipse["b"]))})
    else:
        out["why"] = f"only {len(xs)} ring pixels near the expected rim"
    cy_, cx_ = np.nonzero(bright & (rn < 0.45))
    if len(cx_) >= 5:
        out["cross_centre_px"] = [float(cx_.mean()), float(cy_.mean())]
        out["cross_offset_px"] = [float(cx_.mean() - cam_ellipse["cx"]),
                                  float(cy_.mean() - cam_ellipse["cy"])]
        # the cross lands wholly on the face: its centroid tracks the parallax shift
        # (~0.9 camera px per mm of depth on the bench) without the rim's clipping
        out["centre_offset_px"] = out["cross_offset_px"]
        out["centre_from"] = "cross"
        out["found"] = True
    elif out["found"]:
        out["centre_from"] = "ring"
    return out

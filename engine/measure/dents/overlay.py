"""
Overlay PNG of a dent detection (cv2 only), cropped around the analysed face:

  left   projector-lit image (HDR, log) with the candidates
  middle fused score (0..2 x the good-cell maximum, inferno) + face outline
  right  metric radial deviation (median of the repeats, +-0.4 mm; red = pushed in)

Rings: green = detection, orange = watch, grey = edge. Labels: rank, score and
kind (D dent, S scratch, K speck). Header: the strongest candidates in words.
"""

from __future__ import annotations

import numpy as np

from ._util import Quiet

COLOURS = {"detection": (0, 220, 0), "watch": (0, 170, 255), "edge": (160, 160, 160)}
KIND = {"dent": "D", "scratch": "S", "speck": "K"}


def _gray(a, lo=None, hi=None):
    import cv2
    with Quiet():
        lo = np.nanpercentile(a, 1) if lo is None else lo
        hi = np.nanpercentile(a, 99.5) if hi is None else hi
    v = np.clip((np.nan_to_num(a, nan=lo) - lo) / (hi - lo + 1e-9), 0, 1) ** 0.7
    return cv2.cvtColor((255 * v).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def render(an: dict, det: dict, title: str, scale: int = 4, pad: int = 14) -> bytes:
    """-> PNG bytes."""
    import cv2
    data, geo = an["data"], an["geo"]
    face = geo["face"]
    ys, xs = np.nonzero(face)
    H, W = face.shape
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, H)
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, W)
    sl = (slice(y0, y1), slice(x0, x1))
    on = np.log(np.maximum(data.hdr(level_on=True), 1.0))
    p1 = _gray(on[sl])
    F = det["F"][sl]
    v = np.clip(np.nan_to_num(F, nan=0.0) / 2.0, 0, 1)
    p2 = cv2.applyColorMap((255 * v).astype(np.uint8), cv2.COLORMAP_INFERNO)
    p2[~np.isfinite(F)] = (35, 35, 35)
    with Quiet():
        dev = np.nanmedian(geo["devs"], 0)[sl]
    dv = np.clip(0.5 - 0.5 * np.nan_to_num(dev, nan=0.0) / 0.4, 0, 1)
    p3 = cv2.applyColorMap((255 * dv).astype(np.uint8), cv2.COLORMAP_JET)
    p3[~np.isfinite(dev)] = (35, 35, 35)
    panels = [cv2.resize(p, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
              for p in (p1, p2, p3)]
    fm = cv2.resize(face[sl].astype(np.uint8), None, fx=scale, fy=scale,
                    interpolation=cv2.INTER_NEAREST)
    cnts, _ = cv2.findContours(fm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(panels[1], cnts, -1, (0, 190, 0), 1)
    ox, oy = data.x0 + x0, data.y0 + y0
    for im in panels:
        for i, c in enumerate(det["candidates"][:20]):
            cx = int((c["x"] - ox + 0.5) * scale)
            cy = int((c["y"] - oy + 0.5) * scale)
            col = COLOURS.get(c["level"], (255, 255, 255))
            cv2.circle(im, (cx, cy), 5 * scale, col, 2 if c["level"] == "detection" else 1)
            cv2.putText(im, f"{i + 1}:{c['score']:.2f}{KIND.get(c['kind'], '?')}",
                        (cx + 5 * scale + 2, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
        for yy in range((oy // 20 + 1) * 20, oy + (y1 - y0), 20):
            py = int((yy - oy + 0.5) * scale)
            cv2.line(im, (0, py), (8, py), (255, 255, 0), 1)
            cv2.putText(im, str(yy), (10, py + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255, 255, 0), 1)
    names = ["projector-lit (HDR, log)", "fused score 0..2 x good-cell max",
             "metric residual +-0.4 mm"]
    for im, nm in zip(panels, names):
        cv2.putText(im, nm, (4, im.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (255, 255, 255), 1)
    gap = np.full((panels[0].shape[0], 6, 3), 255, np.uint8)
    body = np.concatenate([panels[0], gap, panels[1], gap, panels[2]], 1)
    lines = [title]
    for i, c in enumerate(det["candidates"][:10]):
        dep = "-" if c["depth_mm"] is None else \
            f"{c['depth_mm']:.2f} mm" + ("" if c["depth_reliable"] else " (unreliable)")
        lines.append(f"{i + 1}: {c['level']} {c['kind']} ({c['x']},{c['y']}) "
                     f"score {c['score']:.2f} "
                     f"theta {c['theta_deg']:.0f} deg s {c['s_mm']:.1f} mm depth {dep}")
    hdr = np.zeros((16 * len(lines) + 6, body.shape[1], 3), np.uint8)
    for i, t in enumerate(lines):
        cv2.putText(hdr, t[:150], (4, 14 + 16 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1)
    ok, buf = cv2.imencode(".png", np.vstack([hdr, body]))
    if not ok:
        raise RuntimeError("could not encode the dent overlay PNG")
    return buf.tobytes()

"""Station figure (cv2 only; matplotlib is not a dependency): the flood crop with the end-face rims,
the tube edge pixels coloured by zone (triggered zones red) and the zone labels, and
below it the tube edge height per zone (median, p5-p95) with the golden bands of a check."""

from __future__ import annotations

from typing import Optional

import numpy as np

GREEN, RED, GREY, WHITE, BLACK = (0, 200, 0), (0, 0, 255), (150, 150, 150), (255, 255, 255), (0, 0, 0)
ZONE_COLOURS = ((255, 170, 0), (0, 200, 255))          # BGR, alternating zones
BAND = (200, 240, 200)


def _crop(report: dict, shape, ct_mask: Optional[np.ndarray]) -> tuple:
    H, W = shape
    xs, ys = [], []
    for c in report["cells"]:
        (x, y), r = c["camera_px"], c["r_px"]
        xs += [x - 2.2 * r, x + 2.2 * r]
        ys += [y - 2.2 * r, y + 2.2 * r]
    if ct_mask is not None and ct_mask.any():
        yy, xx = np.nonzero(ct_mask)
        xs += [xx.min() - 20, xx.max() + 20]
        ys += [yy.min() - 20, yy.max() + 20]
    x0, x1 = max(int(min(xs)), 0), min(int(max(xs)) + 1, W)
    y0, y1 = max(int(min(ys)), 0), min(int(max(ys)) + 1, H)
    return x0, y0, x1, y1


def station_png(report: dict, flood_d: np.ndarray, ct_mask: Optional[np.ndarray],
                zone_px: Optional[np.ndarray] = None, check: Optional[dict] = None,
                title: str = "") -> Optional[bytes]:
    """zone_px: int image of the part-fixed zone index k per CT pixel (else no colours by zone).
    check: {'items': [...], 'verdict'} of a station check (bands + triggers drawn)."""
    try:
        import cv2
    except ImportError:
        return None
    font = cv2.FONT_HERSHEY_SIMPLEX
    x0, y0, x1, y1 = _crop(report, flood_d.shape, ct_mask)
    g = np.clip(flood_d[y0:y1, x0:x1], 0, 255)
    hi = float(np.percentile(g, 99.5)) if g.size else 1.0
    g8 = np.clip(g * (230.0 / max(hi, 1.0)), 0, 255).astype(np.uint8)
    img = cv2.cvtColor(g8, cv2.COLOR_GRAY2BGR)
    trig = {it["item"] for it in (check or {}).get("items", []) if it.get("status") == "trigger"}
    zones = report.get("zones") or []
    lab_by_k = {z["k"]: z.get("label") for z in zones if z.get("included")}
    if ct_mask is not None and zone_px is not None:
        m = ct_mask[y0:y1, x0:x1]
        kk = zone_px[y0:y1, x0:x1]
        for k in np.unique(kk[m]):
            sel = m & (kk == k)
            lab = lab_by_k.get(int(k))
            col = GREY if lab is None else (RED if lab in trig else ZONE_COLOURS[int(k) % 2])
            img[sel] = col
    sc = max(1.0, min(2.0, 1400.0 / max(x1 - x0, 1)))
    img = cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_NEAREST)
    for c in report["cells"]:
        (x, y), r = c["camera_px"], c["r_px"]
        col = RED if any(t.startswith(c["name"] + "-") or t.endswith("-" + c["name"])
                         for t in trig) else GREEN
        cv2.circle(img, (int((x - x0) * sc), int((y - y0) * sc)), int(r * sc), col, 1)
        cv2.putText(img, f"{c['name']} {c['height_mm']:+.3f}", (int((x - x0 + r) * sc) + 4,
                    int((y - y0) * sc)), font, 0.5, col, 1)
    for i, z in enumerate(zones):
        if not z.get("included") or not z.get("cam_uv"):
            continue
        u, v = z["cam_uv"]
        dy = -14 if i % 2 else 22
        cv2.putText(img, z["label"], (int((u - x0) * sc) - 8, int((v - y0) * sc) + dy), font, 0.4,
                    RED if z["label"] in trig else WHITE, 1)
    top = img
    # ---- the zone chart
    inc = [z for z in zones if z.get("included")]
    Wc = max(top.shape[1], 900)
    Hc = 300
    ch = np.full((Hc, Wc, 3), 255, np.uint8)
    L, T, R, B = 70, 40, Wc - 20, Hc - 45
    items = {it["item"]: it for it in (check or {}).get("items", [])}
    vals = []
    for z in inc:
        for key in ("median_mm", "p5_mm", "p95_mm"):
            if z.get(key) is not None:
                vals.append(z[key])
        it = items.get(z["label"])
        if it and it.get("band_mm"):
            vals += list(it["band_mm"])
    if inc and vals:
        lo, hi = min(vals), max(vals)
        pad = max(0.1, 0.08 * (hi - lo))
        lo, hi = lo - pad, hi + pad
        ymap = lambda h: int(B - (h - lo) / (hi - lo) * (B - T))
        n = len(inc)
        xmap = lambda i: int(L + (i + 0.5) * (R - L) / n)
        cv2.rectangle(ch, (L, T), (R, B), (200, 200, 200), 1)
        for t in np.linspace(lo, hi, 5):
            cv2.line(ch, (L, ymap(t)), (R, ymap(t)), (235, 235, 235), 1)
            cv2.putText(ch, f"{t:.2f}", (4, ymap(t) + 4), font, 0.4, BLACK, 1)
        for i, z in enumerate(inc):
            x = xmap(i)
            it = items.get(z["label"])
            if it and it.get("band_mm"):
                b0, b1 = it["band_mm"]
                cv2.rectangle(ch, (x - 9, ymap(b1)), (x + 9, ymap(b0)), BAND, -1)
                if it.get("nominal_mm") is not None:
                    cv2.line(ch, (x - 9, ymap(it["nominal_mm"])), (x + 9, ymap(it["nominal_mm"])),
                             (0, 120, 0), 1)
            if z.get("median_mm") is None or (it and it.get("status") == "missing"):
                cv2.putText(ch, "?", (x - 4, B - 6), font, 0.5, RED, 1)
                if z.get("median_mm") is not None:
                    cv2.circle(ch, (x, ymap(z["median_mm"])), 4, GREY, 1)
            else:
                col = RED if z["label"] in trig else (160, 60, 0)
                cv2.line(ch, (x, ymap(z["p5_mm"])), (x, ymap(z["p95_mm"])), col, 1)
                cv2.circle(ch, (x, ymap(z["median_mm"])), 4, col, -1)
            cv2.putText(ch, z["label"], (x - 10, B + 18), font, 0.4, BLACK, 1)
        cv2.putText(ch, "tube edge height vs the mean end-face plane (mm; median, p5-p95"
                        + ("; green: golden band" if items else "") + ")", (L, 24), font, 0.5,
                    BLACK, 1)
    else:
        cv2.putText(ch, "no tube zones measured", (L, Hc // 2), font, 0.6, BLACK, 1)
    # ---- header
    hdr = np.full((70, Wc, 3), 255, np.uint8)
    pairs = "   ".join(f"{p['pair']} {p['height_diff_mm']:+.3f} mm, spacing "
                       f"{p['spacing_inplane_mm']:.3f} mm" for p in report.get("pairs", [])[:3])
    cv2.putText(hdr, title[:110], (10, 24), font, 0.55, BLACK, 1)
    line2 = pairs
    if check:
        line2 = f"{check['verdict']} ({len(trig)} trigger(s))   " + pairs
    cv2.putText(hdr, line2[:140], (10, 52), font, 0.5, RED if trig else BLACK, 1)
    if top.shape[1] < Wc:
        top = cv2.copyMakeBorder(top, 0, 0, 0, Wc - top.shape[1], cv2.BORDER_CONSTANT, value=WHITE)
    out = np.vstack([hdr, top, ch])
    ok, buf = cv2.imencode(".png", out)
    return buf.tobytes() if ok else None

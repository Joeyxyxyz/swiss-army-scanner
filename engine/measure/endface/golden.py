"""
The BAND detector (detectors.py 'band'): the bench prototype golden.py + the sweep of
2026-09-28, productised; every setting is a parameter. MF-CFAR (mfcfar.py) replaced it as the
default: on the bench data the band gave 5-19 false triggers over 4 good placements (MF-CFAR 0)
for no better sensitivity.

  golden   per grid cell: the median of the placements' height maps (face frame, mm)
  band     max(k_sigma x sigma_placement, floor_mm), sigma = the placements' std at the cell
           (x sqrt(n / (n - 1))), median-smoothed over smooth_cells x smooth_cells (few
           samples per cell); the RIM ZONE r > rim_r_mm (a grazing chamfer on the bench
           cells) gets max(band x rim_factor, rim_min_mm)
  check    dev = placement - golden; TRIGGERS = connected regions with |dev| > band of at
           least min_area_mm2 (detect_face_triggers; check_face(detector=fn) swaps it for any
           callable(dev_map, band, params))

Defaults = the bench's best band trade-off: k_sigma 4, floor 0.10 mm, height maps
Gaussian-smoothed 0.25 mm, sigma median 5 x 5 cells, rim zone r > 9.75 mm at x2 (min 0.3 mm),
triggers >= 0.25 mm^2. Leave-one-out, the synthetic battery and the images: detectors.py.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
import numpy as np


@dataclass
class BandParams:
    k_sigma: float = 4.0
    floor_mm: float = 0.10
    smooth_cells: int = 5           # median filter of sigma (cells; 5 x 0.25 mm = 1.25 mm)
    rim_r_mm: float = 9.75          # rim zone: r > this
    rim_factor: float = 2.0
    rim_min_mm: float = 0.3
    height_smooth_mm: float = 0.25  # NaN-aware Gaussian sigma on every height map (0 = off)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TriggerParams:
    min_area_mm2: float = 0.25      # smallest trigger region (4 cells of 0.25 mm)

    def to_dict(self) -> dict:
        return asdict(self)


def grid_coords(g: np.ndarray) -> tuple:
    GU, GV = np.meshgrid(g, g)
    return GU, GV, np.hypot(GU, GV), np.degrees(np.arctan2(GV, GU))


def smooth_nan(a: np.ndarray, sigma_cells: float) -> np.ndarray:
    """NaN-aware Gaussian (normalised convolution); NaN cells stay NaN."""
    if not sigma_cells or sigma_cells <= 0:
        return a
    from scipy.ndimage import gaussian_filter
    ok = np.isfinite(a)
    num = gaussian_filter(np.where(ok, a, 0.0), sigma_cells, mode="constant")
    den = gaussian_filter(ok.astype(float), sigma_cells, mode="constant")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    return np.where(ok, out, np.nan)


def prepare(H: np.ndarray, g: np.ndarray, prm: BandParams) -> np.ndarray:
    """The height map(s) as compared: NaN-aware Gaussian smoothing when height_smooth_mm > 0."""
    if not (prm.height_smooth_mm and prm.height_smooth_mm > 0):
        return H
    s = prm.height_smooth_mm / float(g[1] - g[0])
    if H.ndim == 3:
        return np.stack([smooth_nan(h, s) for h in H])
    return smooth_nan(H, s)


def build_golden(H: np.ndarray, g: np.ndarray, radius_mm: float, prm: BandParams) -> dict:
    """H (n, G, G) placement height maps -> {golden, band, sigma, n_valid}."""
    from scipy.ndimage import median_filter
    H = prepare(np.asarray(H, np.float64), g, prm)
    n = len(H)
    if n < 2:
        raise ValueError("a golden face needs at least 2 good placements (3+ recommended)")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        gold = np.nanmedian(H, 0)
        nv = np.sum(np.isfinite(H), 0)
        sig = np.nanstd(H, 0) * np.sqrt(np.maximum(nv, 2) / np.maximum(nv - 1, 1))
    sig = np.where(nv >= 2, sig, np.nan)
    k = int(prm.smooth_cells)
    if k > 1:
        s2 = median_filter(np.nan_to_num(sig, nan=0.0), size=k, mode="nearest")
        s2 = np.where(np.isnan(sig), np.nan, s2)
    else:
        s2 = sig
    band = np.maximum(prm.k_sigma * s2, prm.floor_mm)
    _, _, R, _ = grid_coords(g)
    rim = R > prm.rim_r_mm
    band = np.where(rim, np.maximum(band * prm.rim_factor, prm.rim_min_mm), band)
    face = R <= radius_mm
    band = np.where(face & np.isfinite(gold), band, np.nan)
    gold = np.where(face, gold, np.nan)
    return {"golden": gold, "band": band, "sigma": sig, "n_valid": nv}


def detect_face_triggers(dev_map: np.ndarray, band: np.ndarray, params: dict) -> list:
    """THE trigger detector: connected regions where |dev| > band.

    params: grid_mm (1-D axis, mm), radius_mm, rim_r_mm, min_area_mm2.
    -> triggers, largest |dev| first: {zone 'face' | 'rim', r_mm, theta_deg, u_mm, v_mm,
    dev_mm (signed peak: - = below the golden, a dent), area_mm2, band_mm (at the peak),
    excess_mm (|dev| - band at the peak), n_cells}."""
    import cv2
    g = np.asarray(params["grid_mm"], np.float64)
    step = float(g[1] - g[0])
    GU, GV, R, TH = grid_coords(g)
    face = R <= float(params["radius_mm"])
    ok = face & np.isfinite(dev_map) & np.isfinite(band)
    out_m = ok & (np.abs(np.nan_to_num(dev_map)) > np.nan_to_num(band, nan=np.inf))
    n, lab, _, _ = cv2.connectedComponentsWithStats(out_m.astype(np.uint8))
    trig = []
    for i in range(1, n):
        m = lab == i
        area = float(m.sum()) * step * step
        if area < float(params.get("min_area_mm2", 0.25)):
            continue
        k = int(np.nanargmax(np.where(m, np.abs(dev_map), -1.0)))
        y, x = np.unravel_index(k, dev_map.shape)
        r = float(R[y, x])
        trig.append({"zone": "rim" if r > float(params.get("rim_r_mm", np.inf)) else "face",
                     "r_mm": round(r, 2), "theta_deg": round(float(TH[y, x]), 1),
                     "u_mm": round(float(GU[y, x]), 2), "v_mm": round(float(GV[y, x]), 2),
                     "dev_mm": round(float(dev_map[y, x]), 3), "area_mm2": round(area, 2),
                     "band_mm": round(float(band[y, x]), 3),
                     "excess_mm": round(float(abs(dev_map[y, x]) - band[y, x]), 3),
                     "n_cells": int(m.sum())})
    trig.sort(key=lambda t: -abs(t["dev_mm"]))
    return trig


def trigger_params(g, radius_mm: float, band_prm: BandParams, trig_prm: TriggerParams) -> dict:
    return {"grid_mm": np.asarray(g), "radius_mm": float(radius_mm),
            "rim_r_mm": float(band_prm.rim_r_mm), **trig_prm.to_dict()}


def check(h: np.ndarray, gold: dict, g, radius_mm: float, band_prm: BandParams,
          trig_prm: TriggerParams, detector=None) -> tuple:
    """-> (dev map, triggers, stats)."""
    hh = prepare(np.asarray(h, np.float64), np.asarray(g), band_prm)
    dev = hh - gold["golden"]
    det = detector or detect_face_triggers
    trig = det(dev, gold["band"], trigger_params(g, radius_mm, band_prm, trig_prm))
    _, _, R, _ = grid_coords(np.asarray(g))
    inner = dev[(R < band_prm.rim_r_mm) & np.isfinite(dev)]
    stats = {"dev_rms_mm": float(np.std(inner)) if inner.size else None,
             "max_abs_dev_mm": float(np.max(np.abs(inner))) if inner.size else None,
             "compared_pct": round(100.0 * float(np.isfinite(dev[R <= radius_mm]).mean()), 2),
             "n_face": sum(t["zone"] == "face" for t in trig),
             "n_rim": sum(t["zone"] == "rim" for t in trig), "rim_r_mm": band_prm.rim_r_mm}
    return dev, trig, stats


def placement_spread(H: np.ndarray, g, r_max_mm: float) -> dict:
    """How much good placements differ (raw height maps, r < r_max_mm): RMS of the per-cell
    std across placements, and the RMS of every pair's difference map."""
    H = np.asarray(H, np.float64)
    _, _, R, _ = grid_coords(np.asarray(g))
    inner = R < r_max_mm
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sig = np.nanstd(H, 0, ddof=1) if len(H) > 1 else np.full(R.shape, np.nan)
        pairs = [float(np.sqrt(np.nanmean(((H[i] - H[j])[inner]) ** 2)))
                 for i in range(len(H)) for j in range(i + 1, len(H))]
        s_rms = float(np.sqrt(np.nanmean(sig[inner] ** 2))) if np.isfinite(sig[inner]).any() else None
    pairs = [x for x in pairs if np.isfinite(x)]
    return {"sigma_rms_mm": s_rms, "pair_rms_mm": [round(x, 4) for x in pairs],
            "pair_rms_median_mm": float(np.median(pairs)) if pairs else None,
            "r_max_mm": float(r_max_mm), "note": "raw (unsmoothed) height maps"}


def bowl(g, depth_mm: float, width_mm: float, u_mm: float, v_mm: float) -> np.ndarray:
    """The prototype's synthetic bowl dent: -depth x max(1 - (d / (width / 2))^2, 0)."""
    GU, GV, _, _ = grid_coords(np.asarray(g))
    rr = np.hypot(GU - u_mm, GV - v_mm)
    return -depth_mm * np.clip(1 - (rr / (width_mm / 2)) ** 2, 0, None)

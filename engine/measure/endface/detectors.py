"""
The end-face detector hook: which detector turns a placement + a golden into triggers.

  'mf_fir'  (default) MF-CFAR with the reflectance FIR pre-step (mfcfar.py): radial golden,
            dent-shaped matched filter bank, z against this face's own robust noise, the
            flood-predicted reflectance height error removed first. Needs the per-pixel data
            of face_profile and (for the FIR) the targeted scan's flood pair; without a flood
            pair it runs as 'mf' (warned).
  'mf'      the same without the FIR.
  'band'    the bench prototype: golden = median height map, tolerance band = max(k sigma,
            floor), rim zone x2 (golden.py); `trigger_fn` (callable(dev_map, band, params))
            swaps its trigger function (the original one-function hook).

A golden face carries the models of ALL detectors (band: golden + band maps; MF: radial
profile + 2-D median; the FIR's 49 coefficients when every placement had a flood pair), so a
check can use any detector; params['detector'] is the default. Every setting is stored with
the golden; the trigger settings (z_thresh, min_depth_mm, min_area_mm2) can be overridden
per check.

Leave-one-out (golden, FIR and noise from the OTHER placements, the held-out one checked) and
the synthetic-dent set (paraboloid dents injected per camera pixel, integrated over the
pixel footprint, into every repeat, re-gridded through the same operator: the research's
protocol) measure a golden's false triggers and sensitivity on good data.

Research comparison on the bench data (4 placements of 2 cells, leave-one-placement-out):
band 19 false triggers (16 on the saturated placement); mf_fir 0 at every z >= 4.5, and more
sensitive (T 5.5: 49 % at 0.15 x 1.5 mm vs 48 % for the band, which is partly chance with
~5 noise triggers per face).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np

from . import golden as band_mod
from . import mfcfar
from .profile import colour_map

DETECTORS = ("mf_fir", "mf", "band")
DEFAULT_DETECTOR = "mf_fir"
MIN_COMPARED_PCT = 90.0             # a check measuring less of the face is INCOMPLETE


@dataclass
class Placement:
    """One placement as the detectors see it: the profile's height map (band) and its
    per-pixel data (MF; None for a profile saved without it)."""
    label: str
    height: np.ndarray
    pixels: Optional[mfcfar.Pixels] = None


@dataclass
class Settings:
    g: np.ndarray
    radius_mm: float
    band: band_mod.BandParams
    trig: band_mod.TriggerParams
    mf: dict
    fir: dict

    def to_dict(self, detector: str, use_fir: bool) -> dict:
        return {"detector": detector, "band": self.band.to_dict(), "trigger": self.trig.to_dict(),
                "mf": {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.mf.items()},
                "fir": {**self.fir, "used": bool(use_fir)}}


def check_name(detector) -> str:
    if callable(detector):
        return getattr(detector, "__name__", "custom")
    if detector not in DETECTORS:
        raise ValueError(f"detector must be one of {DETECTORS} (or a callable(dev_map, band, "
                         f"params) trigger function for the band), got {detector!r}")
    return detector


# ------------------------------------------------------------------ models

def _fir(placements: list, st: Settings, use_fir: bool) -> tuple:
    if not use_fir:
        return None, None
    if not all(p.pixels is not None and p.pixels.has_flood for p in placements):
        return None, None
    try:
        return mfcfar.fit_fir([p.pixels for p in placements], st.fir)
    except (ValueError, np.linalg.LinAlgError) as e:       # too few usable face pixels
        return None, {"error": str(e)}


def mf_stack(pl: Placement, st: Settings, coef=None, extra_px=None, extra_grid=None,
             gridder=None) -> tuple:
    """-> ((K, G, G) maps for the MF, gridder | None)."""
    if pl.pixels is None:
        h = np.asarray(pl.height, np.float64)
        return (h if extra_grid is None else h + extra_grid)[None], None
    maps, gr = mfcfar.placement_maps(pl.pixels, st.g, st.radius_mm, st.mf, coef, st.fir,
                                     extra_px, gridder)
    if extra_grid is not None:
        maps = maps + np.asarray(extra_grid, np.float64)[None]
    return maps, gr


def build(placements: list, st: Settings, use_fir: bool = True) -> dict:
    """GOOD placements -> every detector's model: {'band': {golden, band, sigma, n_valid},
    'mf': {radial_r_mm, radial_h_mm, map_mm}, 'fir_coef': (49,) | None, 'fir_info'}."""
    H = np.stack([np.asarray(p.height, np.float64) for p in placements])
    coef, info = _fir(placements, st, use_fir)
    maps = [mfcfar.align_and_average(
        mf_stack(p, st, coef if mfcfar.fir_usable(p.pixels, st.fir)[0] else None)[0], st.g, st.mf)
        for p in placements]
    return {"band": band_mod.build_golden(H, st.g, st.radius_mm, st.band),
            "mf": mfcfar.build_model(maps, st.g, st.mf), "fir_coef": coef, "fir_info": info}


# ------------------------------------------------------------------ one check

def check(detector: Union[str, Callable], pl: Placement, model: dict, st: Settings,
          extra_px=None, extra_grid=None, gridder=None) -> dict:
    """-> {detector, dev, triggers, stats, maps, warnings}. extra_px (n,) / extra_grid (G, G):
    heights added to the placement (a synthetic dent) before detection."""
    name = check_name(detector)
    warn = []
    if callable(detector) or detector == "band":
        h = np.asarray(pl.height, np.float64)
        if extra_grid is not None:
            h = h + extra_grid
        elif extra_px is not None and gridder is not None:
            h = h + np.nan_to_num(gridder.grid(extra_px))
        dev, trig, stats = band_mod.check(h, model["band"], st.g, st.radius_mm, st.band, st.trig,
                                          detector if callable(detector) else None)
        stats["dev_r_mm"] = st.band.rim_r_mm
        return {"detector": name, "dev": dev, "triggers": trig, "stats": stats, "warnings": warn,
                "maps": {"height_mm": band_mod.prepare(h, st.g, st.band), "band_mm": model["band"]["band"]}}
    coef = None
    fir_applied = False
    if detector == "mf_fir":
        usable, why = mfcfar.fir_usable(pl.pixels, st.fir)
        if model.get("fir_coef") is None:
            warn.append("the golden has no reflectance FIR (a placement without a flood pair, or "
                        "fir=False): checked as 'mf'")
        elif not usable:
            warn.append(f"{why} in this placement: checked without the reflectance FIR ('mf', "
                        "~10-20 % noisier)" + (" - keep the face below ~230 DN (auto_intensity)"
                                              if "saturates" in why else
                                              " (a profile made without its targeted scan's "
                                              "polar set?)"))
        else:
            coef, fir_applied = model["fir_coef"], True
    if pl.pixels is None:
        warn.append("no per-pixel data in this profile (made before the MF detector): the MF runs "
                    "on the gridded median map (no per-repeat plane removal, no FIR)")
    stack, gr = mf_stack(pl, st, coef, extra_px, extra_grid, gridder)
    res = mfcfar.detect(stack, model["mf"], st.g, st.mf)
    dev = res["deviation"]
    GU, GV, R = mfcfar.grid_mesh(st.g)
    inner = dev[(R <= st.mf["inspect_r_mm"]) & np.isfinite(dev)]
    trig = res["triggers"]
    stats = {"dev_rms_mm": float(np.std(inner)) if inner.size else None,
             "max_abs_dev_mm": float(np.max(np.abs(inner))) if inner.size else None,
             "compared_pct": res["coverage_pct"], "dev_r_mm": st.mf["inspect_r_mm"],
             "n_face": sum(t["zone"] == "face" for t in trig),
             "n_rim": sum(t["zone"] == "rim" for t in trig), "rim_r_mm": st.mf["rim_r_mm"],
             "inspect_r_mm": st.mf["inspect_r_mm"], "max_z": res["max_z"],
             "z_thresh": st.mf["z_thresh"], "noise_floor_mm": res["noise_floor_mm"],
             "fir_applied": fir_applied, "n_repeats": int(stack.shape[0]),
             "flood_saturated_pct": mfcfar.saturated_pct(pl.pixels) if pl.pixels is not None else None}
    sat = stats["flood_saturated_pct"]
    if sat is not None and 1.0 < sat <= st.fir["max_sat_pct"] and fir_applied:
        warn.append(f"{sat:.0f}% of the face is saturated in the flood (keep it below ~230 DN: "
                    "auto_intensity): the FIR is off there")
    return {"detector": name, "dev": dev, "triggers": trig, "stats": stats, "warnings": warn,
            "maps": {"height_mm": res["height"], "z_map": res["z_map"],
                     "depth_map_mm": res["depth_map"], "scale_map_mm": res["scale_map"]},
            "gridder": gr}


def verdict(triggers: list, stats: dict, min_compared_pct: float = MIN_COMPARED_PCT) -> str:
    if triggers:
        return "FAIL"
    cp = stats.get("compared_pct")
    if cp is None or cp < min_compared_pct:
        return "INCOMPLETE"
    return "PASS"


# ------------------------------------------------------------------ leave-one-out + battery

def battery_spec(synthetic) -> Optional[dict]:
    """synthetic: False | True (the default battery) | {depths_mm, widths_mm, positions, seed}."""
    if not synthetic:
        return None
    d = {"depths_mm": mfcfar.BATTERY_DEPTHS_MM, "widths_mm": mfcfar.BATTERY_WIDTHS_MM,
         "positions": 6, "seed": 1000, "tol_mm": mfcfar.BATTERY_TOL_MM, "r_mm": mfcfar.BATTERY_R_MM}
    if isinstance(synthetic, dict):
        bad = set(synthetic) - set(d)
        if bad:
            raise ValueError(f"synthetic: unknown keys {sorted(bad)} (known: {sorted(d)})")
        d.update(synthetic)
    d["depths_mm"] = tuple(float(x) for x in d["depths_mm"])
    d["widths_mm"] = tuple(float(x) for x in d["widths_mm"])
    d["positions"] = int(d["positions"])
    if not d["depths_mm"] or not d["widths_mm"] or min(d["depths_mm"]) <= 0 \
            or min(d["widths_mm"]) <= 0 or not 1 <= d["positions"] <= 200:
        raise ValueError("synthetic: depths_mm / widths_mm must be > 0 mm and positions 1..200")
    return d


def leave_one_out(placements: list, st: Settings, detector: str, use_fir: bool,
                  compare: tuple = DETECTORS, battery: Optional[dict] = None) -> tuple:
    """Each placement vs the golden (+ FIR) of the others -> (rows, sensitivity). rows: per
    held-out placement and per detector in `compare`: triggers on this GOOD face (false
    triggers), dev rms, max z. sensitivity: the battery for `detector` (None without)."""
    n = len(placements)
    rows, sens_rows = [], []
    if n < 3:
        return rows, None
    for i in range(n):
        others = [placements[j] for j in range(n) if j != i]
        model = build(others, st, use_fir)
        pl = placements[i]
        row = {"placement": pl.label}
        base = None
        for det in compare:
            r = check(det, pl, model, st)
            s = r["stats"]
            row[det] = {"n_face": s["n_face"], "n_rim": s["n_rim"],
                        "dev_rms_mm": s["dev_rms_mm"], "max_z": s.get("max_z"),
                        "triggers": r["triggers"][:10]}
            if det == detector:
                base = r
        if base is None:
            base = check(detector, pl, model, st)
        s = base["stats"]
        row.update({"n_face": s["n_face"], "n_rim": s["n_rim"], "dev_rms_mm": s["dev_rms_mm"],
                    "max_z": s.get("max_z"), "triggers": base["triggers"][:10]})
        if i == 0:
            row["_maps"] = {"dev": base["dev"], "z_map": base["maps"].get("z_map")}
        rows.append(row)
        if battery:
            sens_rows += _battery_fold(pl, model, st, detector, battery, i)
    return rows, (_battery_summary(sens_rows, battery, detector, st) if battery else None)


def _battery_fold(pl: Placement, model: dict, st: Settings, detector: str, bat: dict,
                  fold: int) -> list:
    coef = model.get("fir_coef") if detector == "mf_fir" \
        and mfcfar.fir_usable(pl.pixels, st.fir)[0] else None
    gr = None
    if pl.pixels is not None:
        gr = mfcfar.Gridder(pl.pixels, st.g, st.radius_mm, st.mf["max_edge_mm"])
        J = gr.jacobian() if gr.n >= 3 else None
    out = []
    is_mf = detector in ("mf", "mf_fir")
    base_maps = None
    if is_mf and gr is not None and gr.n >= 3:
        base_maps, _ = mfcfar.placement_maps(pl.pixels, st.g, st.radius_mm, st.mf, coef, st.fir,
                                             gridder=gr)
    for di, d in enumerate(bat["depths_mm"]):
        for wi, w in enumerate(bat["widths_mm"]):
            C = mfcfar.battery_centres(bat["positions"], [int(bat["seed"]) + fold, di, wi],
                                       bat["r_mm"])
            for cx, cy in C:
                if gr is not None and gr.n >= 3:
                    px = gr.dent_pixels(cx, cy, d, w, J=J)
                    if is_mf:
                        stack = base_maps + np.nan_to_num(gr.grid(px))[None]
                        res = mfcfar.detect(stack, model["mf"], st.g, st.mf)
                        zn = mfcfar.z_near(res["z_map"], st.g, cx, cy, w, st.mf, bat["tol_mm"])
                        out.append({"depth_mm": d, "width_mm": w, "u_mm": round(float(cx), 2),
                                    "v_mm": round(float(cy), 2), "z": zn})
                        continue
                    r = check(detector, pl, model, st, extra_px=px, gridder=gr)
                else:
                    r = check(detector, pl, model, st,
                              extra_grid=band_mod.bowl(st.g, d, w, cx, cy))
                    if is_mf:
                        zn = mfcfar.z_near(r["maps"]["z_map"], st.g, cx, cy, w, st.mf, bat["tol_mm"])
                        out.append({"depth_mm": d, "width_mm": w, "u_mm": round(float(cx), 2),
                                    "v_mm": round(float(cy), 2), "z": zn})
                        continue
                hit = [t for t in r["triggers"] if (t.get("dev_mm") or 0) < 0
                       and np.hypot(t["u_mm"] - cx, t["v_mm"] - cy) <= w / 2 + bat["tol_mm"]]
                out.append({"depth_mm": d, "width_mm": w, "u_mm": round(float(cx), 2),
                            "v_mm": round(float(cy), 2), "detected": bool(hit)})
    return out


def _battery_summary(rows: list, bat: dict, detector: str, st: Settings) -> dict:
    is_mf = detector in ("mf", "mf_fir")
    T = float(st.mf["z_thresh"])
    by = []
    for d in bat["depths_mm"]:
        for w in bat["widths_mm"]:
            sel = [r for r in rows if r["depth_mm"] == d and r["width_mm"] == w]
            if is_mf:
                z = np.array([r["z"] for r in sel], float)
                hit = int(np.sum(np.nan_to_num(z, nan=-np.inf) >= T))
                e = {"depth_mm": d, "width_mm": w, "of": len(sel), "detected": hit,
                     "pct": round(100.0 * hit / max(len(sel), 1), 1),
                     "pct_at_z": {f"{t:g}": round(100.0 * float(np.mean(np.nan_to_num(z, nan=-np.inf) >= t)), 1)
                                  for t in mfcfar.Z_REPORT} if len(sel) else {}}
            else:
                hit = int(sum(r["detected"] for r in sel))
                e = {"depth_mm": d, "width_mm": w, "of": len(sel), "detected": hit,
                     "pct": round(100.0 * hit / max(len(sel), 1), 1)}
            by.append(e)
    return {"detector": detector, "z_thresh": T if is_mf else None,
            "detected": int(sum(b["detected"] for b in by)), "total": int(sum(b["of"] for b in by)),
            "by_size": by, "depths_mm": list(bat["depths_mm"]), "widths_mm": list(bat["widths_mm"]),
            "positions_per_size_per_fold": bat["positions"], "seed": bat["seed"],
            "tol_mm": bat["tol_mm"], "r_mm": bat["r_mm"],
            "criterion": ("dent z >= z_thresh within width / 2 + tol_mm of the centre" if is_mf
                          else "a trigger below the golden (dev < 0) peaking within width / 2 + "
                               "tol_mm of the centre"),
            "injection": "paraboloid dents per camera pixel (4 x 4 footprint) into every repeat, "
                         "leave-one-out (golden, noise, FIR from the other placements)"}


# ------------------------------------------------------------------ images

def _tile(a, lo, hi, g, radius_mm, circles, title, sc=5):
    import cv2
    t = colour_map(a, lo, hi, scale=sc)
    step = float(g[1] - g[0])
    c = int(round(len(g) / 2 * sc))
    for r, col in circles:
        if r is not None:
            cv2.circle(t, (c, c), int(round(r / step * sc)), col, 1)
    cv2.putText(t, title, (5, 16), 0, 0.42, (255, 255, 255), 1)
    return t


def golden_png(model: dict, st: Settings, loo0: Optional[dict] = None) -> Optional[bytes]:
    try:
        import cv2
    except ImportError:
        return None
    circ = [(st.radius_mm, (255, 255, 255)), (st.mf["inspect_r_mm"], (160, 160, 160))]
    tiles = [_tile(model["mf"]["map_mm"], -0.2, 0.2, st.g, st.radius_mm, circ, "golden face (mm, +-0.2)"),
             _tile(model["band"]["band"], 0.0, 0.4, st.g, st.radius_mm, circ, "band detector (0..0.4 mm)")]
    if loo0 is not None:
        tiles.append(_tile(loo0["dev"], -0.2, 0.2, st.g, st.radius_mm, circ,
                           "placement 1 vs the others (+-0.2)"))
        if loo0.get("z_map") is not None:
            tiles.append(_tile(loo0["z_map"], 0.0, float(st.mf["z_thresh"]), st.g, st.radius_mm,
                               circ, f"its dent z (0..{st.mf['z_thresh']:g})"))
    ok, buf = cv2.imencode(".png", np.hstack(tiles))
    return buf.tobytes() if ok else None


def check_png(res: dict, st: Settings, verdict_: str) -> Optional[bytes]:
    try:
        import cv2
    except ImportError:
        return None
    sc = 6
    step = float(st.g[1] - st.g[0])
    mf = "z_map" in res["maps"]
    circ = [(st.radius_mm, (255, 255, 255)),
            ((st.mf["inspect_r_mm"] if mf else st.band.rim_r_mm), (160, 160, 160))]
    rng = 0.2 if mf else 0.4
    tiles = [_tile(res["maps"]["height_mm"], -rng, rng, st.g, st.radius_mm, circ,
                   f"placement height (+-{rng:g} mm)", sc),
             _tile(res["dev"], -0.2 if mf else -0.3, 0.2 if mf else 0.3, st.g, st.radius_mm, circ,
                   f"deviation vs golden (+-{0.2 if mf else 0.3:g} mm)", sc)]
    if mf:
        tiles.append(_tile(res["maps"]["z_map"], 0.0, float(st.mf["z_thresh"]), st.g, st.radius_mm,
                           circ, f"dent z (0..{st.mf['z_thresh']:g} = trigger)", sc))
    else:
        tiles.append(_tile(res["maps"]["band_mm"], 0.0, 0.4, st.g, st.radius_mm, circ,
                           "band (0..0.4 mm)", sc))
    c0 = len(st.g) / 2 * sc
    for i, t in enumerate(res["triggers"], 1):
        x = int(round(c0 + t["u_mm"] / step * sc))
        y = int(round(c0 + t["v_mm"] / step * sc))
        rr = max(8, int(round(np.sqrt(t["area_mm2"] / np.pi) / step * sc)) + 6)
        txt = f"{i}:{t['depth_mm']:.2f}mm z{t['z']:.1f}" if "z" in t else f"{i}:{t['dev_mm']:+.2f}"
        cv2.circle(tiles[1], (x, y), rr, (255, 255, 255), 2)
        cv2.putText(tiles[1], txt, (x + rr, y), 0, 0.4, (255, 255, 255), 1)
    col = (0, 200, 0) if verdict_ == "PASS" else (0, 0, 255)
    cv2.putText(tiles[1], f"{verdict_} ({len(res['triggers'])} trigger(s), {res['detector']})",
                (5, tiles[1].shape[0] - 10), 0, 0.5, col, 2)
    ok, buf = cv2.imencode(".png", np.hstack(tiles))
    return buf.tobytes() if ok else None


def quiet(fn):
    """Run fn with numpy's all-NaN RuntimeWarnings silenced."""
    def w(*a, **k):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return fn(*a, **k)
    return w

"""
Baked maps — a per-scan inspection stack, all aligned per camera pixel
(UV_MAPPING_FOR_STRUCTURED_LIGHT §4).

    height      relief (mm with a depth calibration, else projector px - rule 6)
    cavity      Laplacian of the smoothed height, signed: > 0 concave (pits,
                valley floors, the foot of a wall), < 0 convex (ridges, top edges)
    normal      unit normal from the height gradient, (H, W, 3) float in
                [-1, 1] + the usual RGB normal-map PNG ((n + 1) / 2)
    validity    decode mask (bool)
    density     decode modulation (scan.field.conf, camera counts; 0 where invalid)
    albedo      white-flood reflectance of a spectral reading / its white_ref (optional)
    subsurface  global / (direct + global) from separate_light (optional)
    dust        albedo - golden albedo (optional; needs albedo and a golden reading)

Every map has units (MapStack.units). Each map is saved as one .npz and one
.png in output/previews/, and a JSON summary in output/measurements/, all
recorded against the scan entry (LibraryStore.output_path / record_output).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import ndimage

from .. import rules
from ..decode.unwrap import _fill_invalid

MAP_NAMES = ("height", "cavity", "normal", "validity", "density", "albedo", "subsurface", "dust")


@dataclass
class MapStack:
    height: np.ndarray                  # (H, W) float32, NaN where invalid
    cavity: np.ndarray                  # (H, W) float32, NaN where invalid
    normal: np.ndarray                  # (H, W, 3) float32 unit vectors (nx, ny, nz), NaN invalid
    validity: np.ndarray                # (H, W) bool
    density: np.ndarray                 # (H, W) float32
    albedo: Optional[np.ndarray] = None
    subsurface: Optional[np.ndarray] = None
    dust: Optional[np.ndarray] = None
    units: dict = field(default_factory=dict)
    calibrated: bool = False            # height in mm (depth calibration applied)
    lateral_mm: bool = False            # gradients per mm (lateral calibration applied)
    npz_paths: dict = field(default_factory=dict)
    png_paths: dict = field(default_factory=dict)
    json_path: Optional[Path] = None
    stats: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    scan_id: str = ""

    def maps(self) -> dict:
        return {n: getattr(self, n) for n in MAP_NAMES if getattr(self, n) is not None}


# ------------------------------------------------------------------ inputs

def _spectral_albedo(reading, shape, geometry_id: str, what: str):
    """-> (albedo = white / white_ref, valid) of a SpectralReading (or its path)."""
    from ..sense import spectral
    c = spectral._cube(reading)
    gid = c["meta"].get("geometry_id", "")
    rules.require_same_geometry(gid, geometry_id, what)              # rule 5
    white = np.asarray(c["white"], np.float32)
    if white.shape != tuple(shape):
        raise ValueError(f"{what}: reading is {white.shape}, scan is {tuple(shape)}")
    ref = spectral._white_ref(c)
    valid = np.asarray(c["valid"], bool)
    return np.where(valid, white / ref, np.nan).astype(np.float32), valid


def _light_subsurface(light, shape):
    d = Path(getattr(light, "path", light))
    f = d / "separation.npz"
    if not f.exists():
        raise FileNotFoundError(f"{d} is not a light separation (no separation.npz); "
                                "pass the LightSeparation from separate_light")
    with np.load(f) as z:
        direct, glob, lit = z["direct"], z["global"], z["lit"].astype(bool)
    if direct.shape != tuple(shape):
        raise ValueError(f"light separation is {direct.shape}, scan is {tuple(shape)}")
    tot = direct + glob
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(lit & (tot > 0), glob / tot, np.nan)
    return frac.astype(np.float32), lit


def _light_geometry(light) -> Optional[str]:
    d = Path(getattr(light, "path", light))
    try:
        return json.loads((d / "meta.json").read_text(encoding="utf-8")).get("geometry_id")
    except (OSError, ValueError):
        return None


# ------------------------------------------------------------------- bake

def bake(relief, mask, conf, calibration=None, *, geometry_id: str = "",
         sigma_px: float = 2.0) -> MapStack:
    """Geometry maps from a relief array (no I/O)."""
    relief = np.asarray(relief, np.float64)
    valid = np.asarray(mask, bool) & np.isfinite(relief)
    warnings = []
    if calibration is not None and not rules.calibration_matches(calibration, geometry_id):
        warnings.append(f"calibration {calibration.id} is for {calibration.geometry_id}, "
                        f"not {geometry_id}: ignored")
        calibration = None
    z = calibration is not None and calibration.has_z
    xy = calibration is not None and calibration.has_xy
    if z:
        h = calibration.mm_per_px * relief + calibration.intercept_mm
        hu = "mm"
    else:
        h = relief
        hu = "projector px (relief; no depth calibration - rule 6)"
        warnings.append(f"no depth calibration for geometry {geometry_id}: height, cavity and "
                        "normal are in projector px, not mm (rule 6)")
    # lateral step of one camera px, in the height's lateral unit
    step = (1.0 / calibration.lateral_px_per_mm) if xy else 1.0
    lu = "mm" if xy else "camera px"
    if z and not xy:
        warnings.append("no lateral calibration: gradients are mm per camera px, so the normal "
                        "map's tilt is not a true angle")
    if not z:
        warnings.append("normal map from projector px per camera px: direction is indicative, "
                        "tilt angles are not physical")

    height = np.where(valid, h, np.nan).astype(np.float32)
    if valid.any():
        filled = _fill_invalid(np.where(valid, h, 0.0), valid)
        sm = ndimage.gaussian_filter(filled, sigma_px)
        gy, gx = np.gradient(sm, step)
        lap = ndimage.laplace(sm) / (step * step)
    else:
        sm = gx = gy = lap = np.zeros(relief.shape)
    cavity = np.where(valid, lap, np.nan).astype(np.float32)
    n = np.stack([-gx, -gy, np.ones_like(gx)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    normal = np.where(valid[..., None], n, np.nan).astype(np.float32)
    density = np.where(valid, np.asarray(conf, np.float32), 0.0).astype(np.float32)
    units = {"height": hu,
             "cavity": f"{hu.split(' ')[0]} per {lu}^2 (Laplacian of height smoothed "
                       f"sigma {sigma_px:g} camera px; > 0 concave, < 0 convex)",
             "normal": "unit vector (nx, ny, nz), camera x right / y down / z up; "
                       "PNG = (n + 1) / 2 as RGB",
             "validity": "bool (decode mask)",
             "density": "camera counts (decode modulation, scan.field.conf)"}
    return MapStack(height=height, cavity=cavity, normal=normal, validity=valid,
                    density=density, units=units, calibrated=z, lateral_mm=xy,
                    warnings=warnings)


def _stretch_u8(a: np.ndarray, valid: np.ndarray, symmetric: bool = False) -> np.ndarray:
    v = a[valid & np.isfinite(a)]
    if v.size == 0:
        return np.zeros(a.shape, np.uint8)
    if symmetric:
        m = float(np.percentile(np.abs(v), 99)) or 1.0
        lo, hi = -m, m
    else:
        lo, hi = float(np.percentile(v, 1)), float(np.percentile(v, 99))
    out = np.clip((np.nan_to_num(a, nan=lo) - lo) / (hi - lo + 1e-12), 0, 1)
    img = (out * 255).astype(np.uint8)
    img[~(valid & np.isfinite(a))] = 0
    return img


def _png(name: str, arr: np.ndarray, valid: np.ndarray) -> np.ndarray:
    import cv2
    if name == "normal":
        rgb = np.nan_to_num((arr + 1.0) * 0.5 * 255.0, nan=0.0)
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        rgb[~valid] = 0
        return rgb[..., ::-1].copy()                        # RGB -> BGR for cv2
    if name == "validity":
        return (arr.astype(np.uint8) * 255)
    if name in ("cavity", "dust"):
        g = _stretch_u8(arr, valid, symmetric=True)
        img = cv2.applyColorMap(g, cv2.COLORMAP_JET)        # blue < 0 < red
        img[~(valid & np.isfinite(arr))] = 0
        return img
    return _stretch_u8(arr, valid)


def _write_png(path: Path, img: np.ndarray) -> Optional[Path]:
    try:
        import cv2
    except ImportError:
        return None
    rules.ensure_new_path(path)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        return None
    with open(path, "xb") as f:
        f.write(buf.tobytes())
    return path


def _stats(a: np.ndarray, valid: np.ndarray) -> dict:
    v = a[valid & np.isfinite(a)] if a.ndim == 2 else None
    if v is None or v.size == 0:
        return {}
    return {"median": round(float(np.median(v)), 6), "p1": round(float(np.percentile(v, 1)), 6),
            "p99": round(float(np.percentile(v, 99)), 6), "n_px": int(v.size)}


def bake_maps(session, scan, spectral=None, light=None, golden=None, *,
              calibration=None, sigma_px: float = 2.0, label: Optional[str] = None,
              save: bool = True) -> MapStack:
    """Bake the aligned map stack of one ObjectScan and save it to the output bin.

    spectral: SpectralReading of the same scene -> albedo
    light:    LightSeparation of the same scene -> subsurface
    golden:   SpectralReading (or MapStack with albedo) of the golden part -> dust
    calibration: the scan's Calibration (api passes _scan_calibration(scan))."""
    label = label or f"{scan.label}_maps"
    relief = scan.relief
    fld = scan.field
    ms = bake(relief, scan.relief_mask, fld.conf, calibration,
              geometry_id=scan.geometry_id, sigma_px=sigma_px)
    ms.scan_id = scan.id
    shape = relief.shape
    if spectral is not None:
        ms.albedo, _ = _spectral_albedo(spectral, shape, scan.geometry_id, "bake_maps albedo")
        ms.units["albedo"] = "fraction of the reading's white reference (white flood / white_ref)"
    if light is not None:
        lg = _light_geometry(light)
        if lg:
            rules.require_same_geometry(lg, scan.geometry_id, "bake_maps subsurface")
        ms.subsurface, _ = _light_subsurface(light, shape)
        ms.units["subsurface"] = "fraction: global / (direct + global) (Nayar 2006)"
    if golden is not None:
        if ms.albedo is None:
            ms.warnings.append("golden given without a spectral reading: no dust map")
        else:
            if isinstance(golden, MapStack):
                g = golden.albedo
                if g is None:
                    raise ValueError("golden MapStack has no albedo map")
            else:
                g, _ = _spectral_albedo(golden, shape, scan.geometry_id, "bake_maps golden")
            ms.dust = (ms.albedo - g).astype(np.float32)
            ms.units["dust"] = ("albedo - golden albedo (fraction of white reference; "
                                "< 0 darker than golden, > 0 brighter)")
    valid = ms.validity
    for name, arr in ms.maps().items():
        if name == "normal":
            continue
        ms.stats[name] = _stats(arr.astype(np.float32) if arr.dtype == bool else arr,
                                np.ones(shape, bool) if name == "validity" else
                                (np.isfinite(arr) if name in ("albedo", "subsurface", "dust")
                                 else valid))
    ms.stats["validity"] = {"valid_pct": round(100.0 * float(valid.mean()), 3)}
    if not save:
        return ms

    store = session.store
    gid = scan.geometry_id
    for name, arr in ms.maps().items():
        npz = store.output_path("previews", f"{label}_{name}", ".npz")
        store.write_npz(npz, **{name: arr, "units": np.array(ms.units.get(name, ""))})
        store.record_output("previews", npz, label=label, geometry_id=gid, source=scan.path)
        ms.npz_paths[name] = npz
        mvalid = valid if name in ("height", "cavity", "normal", "density", "validity") \
            else np.isfinite(arr)
        out = _write_png(store.output_path("previews", f"{label}_{name}", ".png"),
                         _png(name, arr, mvalid))
        if out:
            store.record_output("previews", out, label=label, geometry_id=gid, source=scan.path)
            ms.png_paths[name] = out
    doc = {"kind": "baked_maps", "label": label, "scan_id": scan.id, "geometry_id": gid,
           "created": datetime.now().isoformat(timespec="seconds"),
           "maps": list(ms.maps()), "calibrated": ms.calibrated, "lateral_mm": ms.lateral_mm,
           "calibration_id": getattr(calibration, "id", None), "sigma_px": sigma_px,
           "inputs": {"spectral": str(getattr(spectral, "path", spectral) or "") or None,
                      "light": str(getattr(light, "path", light) or "") or None,
                      "golden": (None if golden is None else
                                 "MapStack" if isinstance(golden, MapStack)
                                 else str(getattr(golden, "path", golden)))},
           "npz": {k: str(v) for k, v in ms.npz_paths.items()},
           "png": {k: str(v) for k, v in ms.png_paths.items()},
           "stats": ms.stats, "units": ms.units, "warnings": ms.warnings}
    js = store.output_path("measurements", label, ".json")
    store.write_json(js, doc)
    store.record_output("measurements", js, label=label, geometry_id=gid, source=scan.path)
    ms.json_path = js
    return ms

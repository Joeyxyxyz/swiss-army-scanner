"""
Spectral mode — 3-band (R/G/B LED) reflectance for material / object ID,
plus direct / global light separation.

The LightCrafter 4500 has separate red (~625 nm), green (~520 nm) and blue
(~460 nm) LEDs (swiss_army_sensor.spectral.STOCK_LEDS). Over HDMI video
mode a pure-red image lights only the red LED — no USB (§9.1). With the
mono camera: dark frame + one frame per band -> per-pixel signal in 3
bands -> normalised signature. Learn signatures from ROIs ("this is
aluminium", "this is the label"), then classify every pixel by nearest
signature with a distance cut-off ("unknown" beyond it). The blobs of a
classification (bbox / centroid, camera px) are what a person or an agent
uses to say "the red box is here" and pick the grid zone to scan.

Capture. Every flood is sent raw (sync.shoot(img, raw=True)), so the
preset's gamma / duty / invert chain cannot tint or dim one band. The raw
path's polarity is MEASURED, never assumed: raw black and raw white go
first; if "black" comes out brighter the path inverts and every flood is
sent inverted. That pair also gives the white reference. Then red, green,
blue, and a second dark, whose difference from the first measures the
camera noise and any ambient drift. Per band: signal = frame - dark.

Signature. Per pixel, the band signals divided by their sum: independent
of exposure, standoff and illumination fall-off, so a material learned in
the centre of the field is recognised at the edge. Brightness (total
signal / the reading's white reference) is kept separately, compared in
log space with a generous spread, because it is the only thing that
separates black from white of the same hue but it IS affected by fall-off.

Valid pixels: total R+G+B signal above k x noise (and the preset's
decode.contrast_min floor), no band saturated, no band significantly
negative. Deliberately not "every band above noise": a saturated-colour
material (red anodising under the blue LED) legitimately returns ~0 in a
band, and that near-zero IS its signature.

Light separation (Nayar et al. 2006). A high-frequency binary pattern
lighting half the scene gives, at a lit pixel, L+ = Ld + Lg/2 and at an
unlit one L- = Lg/2 (global light is low-frequency, so it sees the mean).
Over shifted checkerboards + inverses: direct = Lmax - Lmin,
global = 2 (Lmin - dark). Caveat: blur looks exactly like global light —
a checkerboard the optics do not resolve reads as "global". Bench: a
1-mirror checkerboard keeps ~1 % contrast through the optics, 16 mirrors
~35 % median; below 8 mirrors the split is unreliable (warned). Lmax / Lmin
over N noisy frames are biased by ~1.5 sigma each (direct up, global down);
average more frames (camera.average) when global light is weak.

Storage:  library/spectral/<ts>_<label>/cube.npz   bands (HxWx3 signal), dark, white, valid
                                         meta.json, preview.png (white)
          library/spectral/materials/<name>/<ts>.json   signature versions; the latest
                                                        whose "mock" matches the mode wins
          library/spectral/light/<ts>_<label>/separation.npz, meta.json
          library/output/spectral/<ts>_<label>_classes.png/.json/.npz   classification
          library/output/spectral/<ts>_<label>_light.json               separation numbers
          library/output/previews/<ts>_<label>_colour.png               R/G/B composite
          library/output/previews/<ts>_<label>_direct.png / _global.png
"""

from __future__ import annotations

import json
import math
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from .. import rules
from ..library.store import safe_label, timestamp

BANDS = ("red", "green", "blue")
BAND_NM = {"red": 625.0, "green": 520.0, "blue": 460.0}        # STOCK_LEDS centres
FLOOD_BGR = {"red": (0, 0, 255), "green": (0, 255, 0), "blue": (255, 0, 0)}  # as drawn over HDMI
BLACK, WHITE = (0, 0, 0), (255, 255, 255)

VALID_K_SIGMA = 6.0          # validity threshold, in noise sigmas
SAT_FRACTION = 0.98          # >= this fraction of full scale = saturated
POLARITY_MARGIN = 0.5        # counts (frame mean) between raw black and raw white
SPREAD_FLOOR = 0.02          # signature spread floor (fraction of the band sum)
BRIGHTNESS_LOG_FLOOR = 0.35  # brightness spread floor, ln units (~ +-40 % fall-off)
MIXED_SPREAD = 0.08          # ROI spread above this -> "ROI straddles two materials"
MIN_LEARN_PX = 25
RESERVED_NAMES = ("unknown", "invalid")
BLOB_MIN_PX = 50
MAX_BLOBS = 5                # per material
# map colours (BGR), a categorical palette; material order = legend order
PALETTE = ((180, 119, 31), (14, 127, 255), (44, 160, 44), (40, 39, 214), (189, 103, 148),
           (75, 86, 140), (194, 119, 227), (34, 189, 188), (207, 190, 23), (127, 127, 127))
UNKNOWN_BGR = (70, 70, 70)


@dataclass
class SpectralReading:
    id: str
    label: str
    path: Path
    bands: list                 # band names in cube order
    geometry_id: str
    mock: bool
    created: str
    warnings: list = field(default_factory=list)
    valid_pct: float = 0.0              # % of camera px with a usable signature
    polarity_inverted: bool = False     # raw HDMI path measured inverted (compensated)
    noise_sigma: float = 0.0            # camera counts, per averaged frame
    white_ref: Optional[float] = None   # camera counts: 99th pct of the white-flood signal
    colour_png: Optional[Path] = None   # R/G/B composite in the output bin
    units: dict = field(default_factory=dict)


@dataclass
class Material:
    name: str
    signature: list             # per-band normalised reflectance (sums to 1)
    spread: list                # per-band std of the normalised reflectance
    brightness: float           # mean total reflectance (helps separate black/white)
    n_px: int
    bands: list
    path: Optional[Path] = None             # the material's folder (all versions)
    brightness_spread: float = BRIGHTNESS_LOG_FLOOR   # std of ln(brightness), floored
    version_path: Optional[Path] = None     # this version's JSON
    geometry_id: str = ""
    mock: Optional[bool] = None
    created: str = ""
    roi: Optional[list] = None              # [x0, y0, x1, y1] camera px
    reading_id: str = ""
    warnings: list = field(default_factory=list)
    units: dict = field(default_factory=dict)


@dataclass
class Classification:
    reading_id: str
    materials: list             # names considered
    coverage_pct: dict          # name -> % of valid pixels ("unknown" included)
    blobs: list                 # [{name, area_px, bbox, centroid}] largest regions per material
    map_png: Optional[Path]
    json_path: Optional[Path]
    warnings: list = field(default_factory=list)
    labels_npz: Optional[Path] = None   # per-pixel label map (-1 invalid, 0 unknown, k material k-1)
    max_distance: float = 3.0
    valid_pct: float = 0.0
    units: dict = field(default_factory=dict)


@dataclass
class LightSeparation:
    """Direct (single-bounce) vs global (inter-reflection/scatter) light,
    from shifted checkerboards + inverses (Nayar et al. 2006)."""
    direct_png: Optional[Path]
    global_png: Optional[Path]
    global_fraction_median: float
    square_px: int
    shifts: int
    path: Optional[Path] = None
    warnings: list = field(default_factory=list)
    json_path: Optional[Path] = None
    lit_pct: float = 0.0
    direct_median: float = 0.0          # camera counts
    global_median: float = 0.0          # camera counts
    checker_contrast: float = 0.0       # median direct / white-flood signal
    offsets: list = field(default_factory=list)   # [(dx, dy)] mirror shifts used
    polarity_inverted: bool = False
    units: dict = field(default_factory=dict)


# ------------------------------------------------------------ raw frames

def _dmd_shape(preset) -> tuple:
    w, h = preset.projection.size
    return (int(h), int(w))


def _flood(bgr, shape, inverted: bool = False) -> np.ndarray:
    img = np.empty(tuple(shape) + (3,), np.uint8)
    img[...] = bgr
    return 255 - img if inverted else img


@dataclass
class _Raw:
    inverted: bool
    dark: np.ndarray            # physically dark frame
    white: np.ndarray           # physically full-on (all LEDs) frame


def _measure_polarity(sync, shape, colour: bool = True) -> _Raw:
    """Shoot raw black and raw white; whichever is brighter is 'on'."""
    if colour:
        f0 = sync.shoot(_flood(BLACK, shape), raw=True)
        f1 = sync.shoot(_flood(WHITE, shape), raw=True)
    else:
        f0 = sync.shoot(np.zeros(shape, np.uint8), raw=True)
        f1 = sync.shoot(np.full(shape, 255, np.uint8), raw=True)
    inverted = float(f0.mean()) - float(f1.mean()) > POLARITY_MARGIN
    return _Raw(inverted, f1 if inverted else f0, f0 if inverted else f1)


def _noise_sigma(d0: np.ndarray, d1: np.ndarray) -> float:
    """Per-frame noise from two dark frames: 5-sigma-clipped std of their
    difference / sqrt(2). Not the MAD: on integer camera counts the MAD is
    quantized and under-reads sigma ~1.5 by a third (-> false 'valid' px)."""
    diff = (np.asarray(d1, np.float64) - np.asarray(d0, np.float64))[::2, ::2].ravel()
    diff = diff - np.median(diff)
    s = float(diff.std())
    for _ in range(3):
        kept = diff[np.abs(diff) <= 5 * s + 1e-9]
        if kept.size == 0:
            break
        s = float(kept.std())
    return max(s / math.sqrt(2.0), 0.1)


def _saturation_level(*frames) -> float:
    top = max(float(np.max(f)) for f in frames)
    for full in (255.0, 1023.0, 4095.0, 65535.0):
        if top <= full + 0.5:
            return full
    return top


def _write_png(path: Path, img: np.ndarray) -> Optional[Path]:
    import cv2
    rules.ensure_new_path(path)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        return None
    with open(path, "xb") as f:
        f.write(buf.tobytes())
    return path


def _stretch(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    vals = arr[mask] if mask.any() else arr.ravel()
    lo, hi = np.percentile(vals, 1), np.percentile(vals, 99)
    return (np.clip((arr - lo) / (hi - lo + 1e-9), 0, 1) * 255).astype(np.uint8)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _finite_or_none(v) -> Optional[float]:
    return float(v) if v is not None and np.isfinite(v) else None


# ------------------------------------------------------------ capture

def capture(session, preset, label: str = "spectral") -> SpectralReading:
    safe_label(label)                       # a bad label fails before the rig comes up
    shape = _dmd_shape(preset)
    warnings = []
    if not preset.camera.mono:
        warnings.append("camera.mono is false: spectral mode assumes a mono camera that sums "
                        "the LED sub-frames; colour-camera frames are not modelled")
    with session.rig(preset) as sync:
        pol = _measure_polarity(sync, shape)
        raw = {b: sync.shoot(_flood(FLOOD_BGR[b], shape, pol.inverted), raw=True) for b in BANDS}
        dark_end = sync.shoot(_flood(BLACK, shape, pol.inverted), raw=True)

    dark = 0.5 * (pol.dark + dark_end)
    sigma = _noise_sigma(pol.dark, dark_end)
    bands = np.stack([raw[b] - dark for b in BANDS], axis=-1).astype(np.float32)
    white = (pol.white - dark).astype(np.float32)
    total = bands.sum(axis=-1)
    # total = sum of 3 frames - 3 x mean of 2 darks: var = (3 + 9/2) sigma^2
    thr = max(VALID_K_SIGMA * sigma * math.sqrt(7.5), float(preset.decode.contrast_min))
    sat = _saturation_level(pol.white, *raw.values())
    band_sat = np.zeros(total.shape, bool)
    for f in raw.values():
        band_sat |= f >= SAT_FRACTION * sat
    white_sat = pol.white >= SAT_FRACTION * sat
    lit = total > thr
    valid = lit & ~band_sat & np.all(bands > -VALID_K_SIGMA * sigma * math.sqrt(1.5), axis=-1)

    n_valid, n_lit = int(valid.sum()), int(lit.sum())
    valid_pct = 100.0 * n_valid / valid.size
    white_ref = float(np.percentile(white[valid], 99)) if n_valid else float("nan")
    ok_w = valid & ~white_sat & (white > thr)
    band_sum_vs_white = (float(np.median(total[ok_w] / white[ok_w])) if ok_w.sum() > 100
                         else float("nan"))
    drift = float(np.mean(dark_end) - np.mean(pol.dark))
    sat_pct = 100.0 * int((band_sat & lit).sum()) / max(n_lit, 1)
    white_sat_pct = 100.0 * int((white_sat & valid).sum()) / max(n_valid, 1)

    if pol.inverted:
        warnings.append("raw HDMI path measured INVERTED (raw black was brighter than raw white); "
                        "floods were sent inverted to compensate")
    if not n_valid:
        warnings.append(f"no pixel has a usable signature (threshold {thr:.1f} counts): is the "
                        "projector on and aimed at the scene, and the exposure long enough?")
    elif valid_pct < 5.0:
        warnings.append(f"only {valid_pct:.1f}% of the frame is valid — raise exposure or check "
                        "the lit patch covers the objects")
    if sat_pct > 0.5:
        warnings.append(f"{sat_pct:.1f}% of lit pixels saturate in a colour band and were dropped "
                        "— lower camera.exposure_us")
    if white_sat_pct > 1.0:
        warnings.append(f"white reference saturates on {white_sat_pct:.1f}% of valid pixels — "
                        "brightness of the brightest surfaces is under-estimated")
    if n_valid and abs(drift) > max(1.0, 0.02 * white_ref):
        warnings.append(f"ambient light changed during capture (dark drift {drift:+.1f} counts) "
                        "— shade the scene and re-capture")
    if np.isfinite(band_sum_vs_white) and not 0.8 <= band_sum_vs_white <= 1.25:
        warnings.append(f"R+G+B floods sum to {band_sum_vs_white:.2f} x the white flood — the "
                        "video path may add colour processing (white segment / BrilliantColor); "
                        "signatures stay comparable, brightness less so")

    store = session.store
    d = store.new_dir("spectral", label=label)
    store.write_npz(d / "cube.npz", bands=bands, dark=dark.astype(np.float32), white=white,
                    valid=valid, band_names=np.array(BANDS), noise_sigma=np.float32(sigma),
                    white_ref=np.float32(white_ref))
    store.write_preview(d / "preview.png", white)
    created = _now()
    colour_png = None
    comp = _colour_composite(bands, valid)
    if comp is not None:
        out = store.output_path("previews", f"{label}_colour", ".png")
        if _write_png(out, comp):
            store.record_output("previews", out, label=label, geometry_id=preset.geometry_id,
                                source=d)
            colour_png = out
    units = {"valid_pct": "% of camera px", "noise_sigma": "camera counts",
             "white_ref": "camera counts (99th pct of the white-flood signal over valid px)",
             "cube.bands": "camera counts (frame - dark), H x W x band"}
    meta = {"kind": "spectral_reading", "id": d.name, "label": label, "bands": list(BANDS),
            "band_nm": BAND_NM, "flood_bgr": {b: list(FLOOD_BGR[b]) for b in BANDS},
            "geometry_id": preset.geometry_id, "preset": preset.name, "mock": bool(preset.mock),
            "created": created, "camera": vars(preset.camera),
            "polarity_inverted": pol.inverted, "noise_sigma": round(sigma, 4),
            "valid_threshold": round(thr, 3), "valid_pct": round(valid_pct, 3),
            "white_ref": _finite_or_none(white_ref),
            "band_sum_vs_white": _finite_or_none(band_sum_vs_white),
            "saturated_pct": round(sat_pct, 3), "white_saturated_pct": round(white_sat_pct, 3),
            "dark_drift": round(drift, 3), "colour_png": colour_png,
            "units": {**units, "valid_threshold": "camera counts (R+G+B signal)",
                      "dark_drift": "camera counts", "saturated_pct": "% of lit px",
                      "white_saturated_pct": "% of valid px"},
            "warnings": warnings}
    store.write_json(d / "meta.json", meta)
    return SpectralReading(id=d.name, label=label, path=d, bands=list(BANDS),
                           geometry_id=preset.geometry_id, mock=bool(preset.mock), created=created,
                           warnings=warnings, valid_pct=round(valid_pct, 3),
                           polarity_inverted=pol.inverted, noise_sigma=round(sigma, 4),
                           white_ref=_finite_or_none(white_ref), colour_png=colour_png,
                           units=units)


def _colour_composite(bands: np.ndarray, valid: np.ndarray) -> Optional[np.ndarray]:
    """A colour photo from the mono camera: R/G/B band signals as BGR,
    gray-world balanced (the base / wall dominates most scenes)."""
    if valid.sum() < 100:
        return None
    med = np.array([np.median(bands[..., c][valid]) for c in range(3)], np.float64)
    if np.any(med <= 0):
        return None
    img = bands * (med.mean() / med).astype(np.float32)
    top = float(np.percentile(img[valid], 99))
    out = np.clip(img / (top + 1e-9), 0, 1) * 255
    return np.ascontiguousarray(out[..., ::-1]).astype(np.uint8)


# ------------------------------------------------------------ reading data

_CUBES: "OrderedDict[str, dict]" = OrderedDict()


def _reading_dir(reading) -> Path:
    return Path(getattr(reading, "path", reading))


def _cube(reading) -> dict:
    """cube.npz + meta.json of a reading (small LRU: learn/classify reuse it)."""
    d = _reading_dir(reading)
    key = str(d.resolve())
    if key in _CUBES:
        _CUBES.move_to_end(key)
        return _CUBES[key]
    f = d / "cube.npz"
    if not f.exists():
        raise FileNotFoundError(f"{d} is not a spectral reading (no cube.npz); "
                                "use a SpectralReading from spectral_capture / list_library('spectral')")
    with np.load(f) as z:
        c = {k: z[k] for k in z.files}
    c["meta"] = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    _CUBES[key] = c
    while len(_CUBES) > 2:
        _CUBES.popitem(last=False)
    return c


def _white_ref(c: dict) -> float:
    w = float(c["white_ref"])
    if not np.isfinite(w) or w <= 0:
        raise ValueError("this reading has no white reference (no lit pixels) — re-capture "
                         "with the projector aimed at the scene")
    return w


def _roi(roi, shape) -> tuple:
    try:
        x0, y0, x1, y1 = (int(round(float(v))) for v in roi)
    except (TypeError, ValueError):
        raise ValueError(f"roi must be (x0, y0, x1, y1) in camera px, got {roi!r}") from None
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    H, W = shape
    cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    if cx1 <= cx0 or cy1 <= cy0:
        raise ValueError(f"roi {roi!r} is empty or outside the {W}x{H} camera frame")
    return cx0, cy0, cx1, cy1


# ------------------------------------------------------------ materials

def _materials_root(session) -> Path:
    return session.store.root / "spectral" / "materials"


def _version_key(p: Path):
    stem, _, n = p.stem.partition("-")
    return (stem, int(n) if n.isdigit() else 0)


def _new_version_path(d: Path) -> Path:
    stem = timestamp()
    for n in range(1000):
        p = d / (f"{stem}.json" if n == 0 else f"{stem}-{n}.json")
        if not p.exists():
            return p
    raise rules.OverwriteRefused(f"could not allocate a new material version under {d}")


def _material_from_doc(doc: dict, vpath: Path) -> Material:
    return Material(name=doc["name"], signature=list(doc["signature"]),
                    spread=list(doc["spread"]), brightness=float(doc["brightness"]),
                    n_px=int(doc["n_px"]), bands=list(doc["bands"]), path=vpath.parent,
                    brightness_spread=float(doc.get("brightness_log_spread",
                                                    BRIGHTNESS_LOG_FLOOR)),
                    version_path=vpath, geometry_id=doc.get("geometry_id", ""),
                    mock=doc.get("mock"), created=doc.get("created", ""), roi=doc.get("roi"),
                    reading_id=doc.get("reading", ""), warnings=list(doc.get("warnings", [])),
                    units=dict(doc.get("units", {})))


def learn(session, reading: SpectralReading, name: str, roi) -> Material:
    """roi = (x0, y0, x1, y1) in camera px."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("material name must be a non-empty string")
    if name.strip().lower() in RESERVED_NAMES:
        raise ValueError(f"{name!r} is reserved for classification maps; pick another name")
    safe_label(name)                        # fail before any work on an unusable name
    c = _cube(reading)
    meta = c["meta"]
    valid = c["valid"]
    x0, y0, x1, y1 = _roi(roi, valid.shape)
    v = valid[y0:y1, x0:x1]
    n = int(v.sum())
    if n < MIN_LEARN_PX:
        raise ValueError(
            f"ROI {[x0, y0, x1, y1]} has only {n} valid pixels (need >= {MIN_LEARN_PX}): draw it "
            "inside a lit, unshadowed, unsaturated part of the object")
    s = c["bands"][y0:y1, x0:x1][v].astype(np.float64)
    tot = s.sum(axis=1)
    x = s / tot[:, None]
    raw_spread = x.std(axis=0)
    sig = x.mean(axis=0)
    spread = np.maximum(raw_spread, SPREAD_FLOOR)
    b = tot / _white_ref(c)
    brightness = float(b.mean())
    b_spread = max(float(np.log(b).std()), BRIGHTNESS_LOG_FLOOR)
    warnings = []
    if n < 0.5 * v.size:
        warnings.append(f"only {100.0 * n / v.size:.0f}% of the ROI is valid (shadow / unlit / "
                        "saturated); the rest was ignored")
    if float(raw_spread.max()) > MIXED_SPREAD:
        warnings.append(f"signature spread {float(raw_spread.max()):.3f} is large — the ROI may "
                        "straddle two materials or an edge; draw it inside one surface")

    mdir = _materials_root(session) / safe_label(name)
    mdir.mkdir(parents=True, exist_ok=True)
    vpath = _new_version_path(mdir)
    doc = {"kind": "spectral_material", "name": name.strip(),
           "signature": [round(float(a), 6) for a in sig],
           "spread": [round(float(a), 6) for a in spread],
           "spread_floor": SPREAD_FLOOR, "brightness": round(brightness, 6),
           "brightness_log_spread": round(b_spread, 6), "n_px": n,
           "bands": list(meta.get("bands", BANDS)), "roi": [x0, y0, x1, y1],
           "reading": meta.get("id", _reading_dir(reading).name),
           "reading_path": str(_reading_dir(reading)), "geometry_id": meta.get("geometry_id", ""),
           "mock": meta.get("mock"), "created": _now(),
           "units": {"signature": "fraction of the R+G+B band signal",
                     "spread": "fraction of the R+G+B band signal (std, floored)",
                     "brightness": "x the reading's white reference",
                     "brightness_log_spread": "ln units", "n_px": "camera px",
                     "roi": "camera px [x0, y0, x1, y1)"},
           "warnings": warnings}
    session.store.write_json(vpath, doc)
    return _material_from_doc(doc, vpath)


def _mock_matches(doc_mock, mock: bool) -> bool:
    """A version's JSON "mock" vs the wanted mode. A version without the key
    (learned before it was recorded) is taken as matching either mode."""
    return doc_mock is None or bool(doc_mock) == bool(mock)


def list_materials(session, mock: Optional[bool] = None) -> list:
    """Newest version of every learned material (Material handles, by name)
    whose JSON "mock" matches `mock` (default: the session's mode): a
    signature learned on the simulated bench never shadows the real one
    learned under the same name, and vice versa. A material with no version
    in this mode is not listed."""
    want = session.mock if mock is None else mock
    root = _materials_root(session)
    if not root.is_dir():
        return []
    out = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        for v in reversed(sorted(d.glob("*.json"), key=_version_key)):
            try:
                doc = json.loads(v.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue                    # unreadable / half-written: never used
            if isinstance(doc, dict) and _mock_matches(doc.get("mock"), want):
                out.append(_material_from_doc(doc, v))
                break
    return out


def list_readings(session) -> list:
    """Every spectral reading in the library, oldest first (handles with path)."""
    out = []
    for d in session.store.entries("spectral"):
        f = d / "meta.json"
        if not f.exists():
            continue                    # materials/, light/
        m = json.loads(f.read_text(encoding="utf-8"))
        if m.get("kind") != "spectral_reading":
            continue
        out.append(SpectralReading(
            id=m["id"], label=m["label"], path=d, bands=list(m["bands"]),
            geometry_id=m["geometry_id"], mock=bool(m["mock"]), created=m["created"],
            warnings=list(m.get("warnings", [])), valid_pct=m.get("valid_pct", 0.0),
            polarity_inverted=bool(m.get("polarity_inverted", False)),
            noise_sigma=m.get("noise_sigma", 0.0), white_ref=m.get("white_ref"),
            colour_png=Path(m["colour_png"]) if m.get("colour_png") else None,
            units={k: v for k, v in m.get("units", {}).items()
                   if k in ("valid_pct", "noise_sigma", "white_ref", "cube.bands")}))
    return out


def _resolve_materials(session, materials, meta: dict) -> tuple:
    # names resolve to the newest version learned in the READING's mode (the
    # session's when the reading does not say): mock never shadows real
    mode = meta.get("mock")
    latest = list_materials(session, mock=None if mode is None else bool(mode))
    by_name = {m.name: m for m in latest}
    by_name.update({safe_label(m.name): m for m in latest if safe_label(m.name) not in by_name})
    if materials is None:
        chosen = latest
    else:
        chosen = []
        for item in materials:
            if isinstance(item, Material):
                chosen.append(item)
            elif isinstance(item, str):
                m = by_name.get(item) or by_name.get(safe_label(item))
                if m is None:
                    raise ValueError(f"unknown material {item!r}; learned: "
                                     f"{sorted(x.name for x in latest)}")
                chosen.append(m)
            else:
                raise TypeError(f"materials must be names or Material handles, got {type(item)}")
    out, seen, warnings = [], set(), []
    bands = list(meta.get("bands", BANDS))
    for m in chosen:
        if m.name in seen:
            continue
        seen.add(m.name)
        if list(m.bands) != bands:
            warnings.append(f"material {m.name!r} has bands {m.bands}, reading has {bands}; skipped")
        elif m.mock is not None and bool(m.mock) != bool(meta.get("mock")):
            warnings.append(f"material {m.name!r} was learned in {'mock' if m.mock else 'real'} "
                            "mode; skipped for this reading")
        else:
            if m.geometry_id and m.geometry_id != meta.get("geometry_id"):
                warnings.append(f"material {m.name!r} was learned at geometry {m.geometry_id!r}; "
                                "signatures carry over but brightness may not")
            out.append(m)
    return out, warnings


# ------------------------------------------------------------ classify

def classify(session, reading: SpectralReading, materials=None,
             max_distance: float = 3.0, *, brightness_weight: float = 1.0) -> Classification:
    """Nearest learned signature per valid pixel; 'unknown' beyond max_distance.

    distance = sqrt(sum(((x - signature) / spread)^2)
                    + (brightness_weight * ln(b / brightness) / brightness_spread)^2)
    in spread units (sigmas). brightness_weight=0 compares hue only."""
    if not max_distance > 0:
        raise ValueError("max_distance must be > 0 (spread-normalised sigmas)")
    c = _cube(reading)
    meta = c["meta"]
    mats, warnings = _resolve_materials(session, materials, meta)
    if not mats:
        raise ValueError("no usable materials — teach some first: learn_material(reading, name, "
                         "(x0, y0, x1, y1))" + (f" ({'; '.join(warnings)})" if warnings else ""))
    valid = c["valid"]
    n_valid = int(valid.sum())
    if n_valid == 0:
        raise ValueError("this reading has no valid pixels — nothing to classify; re-capture")
    S = c["bands"][valid].astype(np.float64)
    tot = S.sum(axis=1)
    X = S / tot[:, None]
    lnb = np.log(tot / _white_ref(c)) if brightness_weight > 0 else None
    D = np.empty((n_valid, len(mats)), np.float32)
    for k, m in enumerate(mats):
        z = (X - np.asarray(m.signature)) / np.asarray(m.spread)
        d2 = (z * z).sum(axis=1)
        if lnb is not None and m.brightness > 0:
            zb = brightness_weight * (lnb - math.log(m.brightness)) / max(m.brightness_spread, 1e-6)
            d2 += zb * zb
        D[:, k] = np.sqrt(d2)
    best = D.argmin(axis=1)
    dmin = D[np.arange(n_valid), best]
    lab_v = np.where(dmin <= max_distance, best + 1, 0).astype(np.int16)
    labels = np.full(valid.shape, -1, np.int16)
    labels[valid] = lab_v
    dist = np.full(valid.shape, np.nan, np.float32)
    dist[valid] = dmin

    names = [m.name for m in mats]
    counts = np.bincount(lab_v, minlength=len(mats) + 1)
    coverage = {n: round(100.0 * float(counts[k + 1]) / n_valid, 3) for k, n in enumerate(names)}
    coverage["unknown"] = round(100.0 * float(counts[0]) / n_valid, 3)
    blobs = _blobs(labels, dist, names)
    for m in mats:
        if coverage[m.name] == 0.0:
            warnings.append(f"material {m.name!r} matched no pixel within {max_distance} sigma")
    valid_pct = round(100.0 * n_valid / valid.size, 3)

    store = session.store
    rid = meta.get("id", _reading_dir(reading).name)
    label = meta.get("label", "spectral")
    geo = meta.get("geometry_id", "")
    src = _reading_dir(reading)
    out_json = store.output_path("spectral", f"{label}_classes", ".json")
    png = _write_png(out_json.with_suffix(".png"),
                     _class_map(c["white"], valid, labels, names, coverage, blobs, max_distance))
    npz = store.write_npz(out_json.with_suffix(".npz"), labels=labels,
                          names=np.array(["unknown"] + names))
    units = {"coverage_pct": "% of valid camera px", "area_px": "camera px",
             "bbox": "camera px [x0, y0, x1, y1)", "centroid": "camera px [x, y]",
             "mean_distance": "sigma (spread-normalised)", "max_distance": "sigma",
             "valid_pct": "% of camera px"}
    doc = {"reading": rid, "reading_path": str(src), "geometry_id": geo, "mock": meta.get("mock"),
           "created": _now(), "max_distance": max_distance, "brightness_weight": brightness_weight,
           "materials": [{"name": m.name, "signature": m.signature, "spread": m.spread,
                          "brightness": m.brightness,
                          "version": str(m.version_path) if m.version_path else None}
                         for m in mats],
           "valid_px": n_valid, "valid_pct": valid_pct, "coverage_pct": coverage, "blobs": blobs,
           "map_png": str(png) if png else None, "labels_npz": str(npz),
           "labels_legend": {"-1": "invalid", "0": "unknown",
                             **{str(k + 1): n for k, n in enumerate(names)}},
           "units": units, "warnings": warnings}
    store.write_json(out_json, doc)
    for p in (out_json, png, npz):
        if p is not None:
            store.record_output("spectral", p, label=label, geometry_id=geo, source=src)
    return Classification(reading_id=rid, materials=names, coverage_pct=coverage, blobs=blobs,
                          map_png=png, json_path=out_json, warnings=warnings, labels_npz=npz,
                          max_distance=max_distance, valid_pct=valid_pct, units=units)


def _blobs(labels: np.ndarray, dist: np.ndarray, names: list) -> list:
    """Largest connected regions per material (3x3 opening drops speckle)."""
    import cv2
    out = []
    k3 = np.ones((3, 3), np.uint8)
    for k, name in enumerate(names, start=1):
        m = (labels == k).astype(np.uint8)
        if not m.any():
            continue
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k3)
        n, lab, stats, cents = cv2.connectedComponentsWithStats(m, connectivity=8)
        if n <= 1:
            continue
        areas = stats[1:, cv2.CC_STAT_AREA]
        for i in np.argsort(-areas)[:MAX_BLOBS]:
            a = int(areas[i])
            if a < BLOB_MIN_PX:
                break
            x, y, w, h = (int(v) for v in stats[i + 1, :4])
            out.append({"name": name, "area_px": a, "bbox": [x, y, x + w, y + h],
                        "centroid": [round(float(cents[i + 1][0]), 1),
                                     round(float(cents[i + 1][1]), 1)],
                        "fill": round(a / float(w * h), 3),
                        "mean_distance": round(float(np.nanmean(dist[lab == i + 1])), 3),
                        "units": "px"})
    return out


def _class_map(white, valid, labels, names, coverage, blobs, max_distance) -> np.ndarray:
    """Colour-coded label map over the white frame, with blob boxes and a legend."""
    import cv2
    gray = _stretch(white, valid).astype(np.float32)[..., None]
    img = np.repeat(gray, 3, axis=2) * 0.3                   # invalid: dimmed photo
    unk = labels == 0
    img[unk] = 0.45 * gray[unk] + 0.55 * np.array(UNKNOWN_BGR, np.float32)
    for k in range(len(names)):
        sel = labels == k + 1
        col = np.array(PALETTE[k % len(PALETTE)], np.float32)
        img[sel] = 0.4 * gray[sel] + 0.6 * col
    img = np.clip(img, 0, 255).astype(np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for b in blobs:
        k = names.index(b["name"])
        col = tuple(int(v) for v in PALETTE[k % len(PALETTE)])
        x0, y0, x1, y1 = b["bbox"]
        cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), (255, 255, 255), 3)
        cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), col, 1)
        cv2.putText(img, b["name"], (x0 + 2, max(y0 - 6, 16)), font, 0.7, (255, 255, 255), 2,
                    cv2.LINE_AA)
    H = img.shape[0]
    panel = np.full((H, 440, 3), 28, np.uint8)
    rows = [(PALETTE[k % len(PALETTE)], f"{n}  {coverage[n]:.1f}%") for k, n in enumerate(names)]
    rows += [(UNKNOWN_BGR, f"unknown  {coverage['unknown']:.1f}%"), ((20, 20, 20), "invalid")]
    cv2.putText(panel, "materials", (16, 36), font, 0.9, (235, 235, 235), 2, cv2.LINE_AA)
    cv2.putText(panel, f"max distance {max_distance:g} sigma", (16, 64), font, 0.55,
                (180, 180, 180), 1, cv2.LINE_AA)
    for i, (col, text) in enumerate(rows):
        y = 100 + 40 * i
        if y + 24 > H:
            break
        cv2.rectangle(panel, (16, y), (44, y + 24), tuple(int(v) for v in col), -1)
        cv2.rectangle(panel, (16, y), (44, y + 24), (200, 200, 200), 1)
        cv2.putText(panel, text[:28], (56, y + 19), font, 0.65, (235, 235, 235), 1, cv2.LINE_AA)
    return np.hstack([img, panel])


# ------------------------------------------------------------ light separation

def checker_margin(offsets: list, square_px: int, grid: int = 32) -> float:
    """Worst case, over every sub-square pixel position, of the best edge
    clearance any of the shifted checkerboards gives it (fraction of a
    square; 0.5 = centred). What the shift set must maximise: Nayar's
    max/min only see pure lit / unlit values at pixels that sit clear of
    square edges in at least one frame."""
    g = (np.arange(grid) + 0.5) / grid
    fx, fy = np.meshgrid(g, g)
    best = np.zeros_like(fx)
    for dx, dy in offsets:
        ux, uy = (fx + dx / square_px) % 1.0, (fy + dy / square_px) % 1.0
        clear = np.minimum(np.minimum(ux, 1 - ux), np.minimum(uy, 1 - uy))
        best = np.maximum(best, clear)
    return float(best.min())


def checker_offsets(square_px: int, shifts: int) -> list:
    """(dx, dy) mirror shifts spread over ONE square in x and y.

    One square, not the full 2-square period: a shift of one square is the
    inverse, which is shot anyway. Candidates are rank-1 lattices
    (k, k*m mod n)/n and a x b product grids; the set with the best
    checker_margin wins, so pixels near a corner (where a purely diagonal
    shift fails) still land inside a square — lit and unlit — in some frame."""
    n = int(shifts)
    cands = []
    for m in range(1, max(n, 2)):
        if math.gcd(m, n) == 1:
            cands.append([(k / n, (k * m % n) / n) for k in range(n)])
    for a in range(1, n + 1):
        if n % a == 0:
            cands.append([(i / a, j / (n // a)) for i in range(a) for j in range(n // a)])
    best, best_score = [(0, 0)], -1.0
    for pts in cands:
        offs = []
        for fx, fy in pts:
            o = (int(round(fx * square_px)) % square_px, int(round(fy * square_px)) % square_px)
            if o not in offs:
                offs.append(o)
        score = checker_margin(offs, square_px)
        if score > best_score + 1e-9:
            best, best_score = offs, score
    return best


def checkerboard(shape, square_px: int, dx: int = 0, dy: int = 0) -> np.ndarray:
    """uint8 0/255 checkerboard of square_px mirrors, shifted by (dx, dy)."""
    H, W = shape
    r = (np.arange(H) + dy) // square_px
    c = (np.arange(W) + dx) // square_px
    return np.where(((r[:, None] + c[None, :]) & 1) == 0, 255, 0).astype(np.uint8)


def separate_light(session, preset, label: str = "light", square_px: int = 16,
                   shifts: int = 8) -> LightSeparation:
    try:
        square_px, shifts = int(square_px), int(shifts)
    except (TypeError, ValueError):
        raise ValueError("square_px and shifts must be integers") from None
    safe_label(label)
    shape = _dmd_shape(preset)
    if square_px < 1 or shifts < 1:
        raise ValueError("square_px and shifts must be >= 1")
    if square_px > min(shape) // 4:
        raise ValueError(f"square_px={square_px} is too coarse: the checkerboard must be high-"
                         f"frequency (<= {min(shape) // 4} mirrors) for global light to see its mean")
    warnings = []
    if square_px < 8:
        warnings.append(f"square_px={square_px} < 8: the bench optics keep only ~1% contrast for "
                        "1-mirror squares (~35% median at 16), so blur will read as global light")
    offsets = checker_offsets(square_px, shifts)
    if len(offsets) < shifts:
        warnings.append(f"only {len(offsets)} distinct phases fit in a {square_px}-mirror square "
                        f"(asked for {shifts})")

    with session.rig(preset) as sync:
        pol = _measure_polarity(sync, shape, colour=False)
        lmax = lmin = None
        for dx, dy in offsets:
            pat = checkerboard(shape, square_px, dx, dy)
            for p in (pat, 255 - pat):
                f = sync.shoot(p, raw=True)
                lmax = f.copy() if lmax is None else np.maximum(lmax, f)
                lmin = f.copy() if lmin is None else np.minimum(lmin, f)
        dark_end = sync.shoot(np.full(shape, 255 if pol.inverted else 0, np.uint8), raw=True)

    dark = 0.5 * (pol.dark + dark_end)
    sigma = _noise_sigma(pol.dark, dark_end)
    white = (pol.white - dark).astype(np.float32)
    direct = (lmax - lmin).astype(np.float32)
    glob = np.clip(2.0 * (lmin - dark), 0, None).astype(np.float32)
    sat = _saturation_level(pol.white, lmax)
    thr = max(VALID_K_SIGMA * sigma * math.sqrt(1.5), float(preset.decode.contrast_min))
    lit = (white > thr) & (lmax < SAT_FRACTION * sat) & (direct + glob > thr)
    n_lit = int(lit.sum())
    lit_pct = 100.0 * n_lit / lit.size
    if n_lit:
        frac = glob[lit] / (direct[lit] + glob[lit])
        gfm = float(np.median(frac))
        d_med, g_med = float(np.median(direct[lit])), float(np.median(glob[lit]))
        contrast = float(np.median(direct[lit] / white[lit]))
        pair = float(np.median((lmax + lmin - 2 * dark)[lit] / white[lit]))
    else:
        gfm = d_med = g_med = contrast = pair = float("nan")
        warnings.append("no lit pixels — is the projector on and aimed at the scene?")
    sat_pct = 100.0 * int(((lmax >= SAT_FRACTION * sat) & (white > thr)).sum()) / lit.size
    if pol.inverted:
        warnings.append("raw HDMI path measured INVERTED; the physically dark frame was used")
    if sat_pct > 0.5:
        warnings.append(f"{sat_pct:.1f}% of the frame saturates under the checkerboard and was "
                        "dropped — lower camera.exposure_us")
    if n_lit and contrast < 0.5:
        warnings.append(f"the checkerboard modulates only {contrast:.0%} of the flood signal: "
                        "strong global light, or the pattern is not resolved (focus / square_px) "
                        "— blur is counted as global")
    if n_lit and not 0.8 <= pair <= 1.2:
        warnings.append(f"Lmax + Lmin is {pair:.2f} x the white flood (expect ~1): nonlinear "
                        "response or the pattern is not half-on at the scene; split is approximate")

    store = session.store
    d = store.new_dir("spectral", "light", label=label)
    store.write_npz(d / "separation.npz", direct=direct, lit=lit, dark=dark.astype(np.float32),
                    **{"global": glob})
    scale = float(np.percentile((direct + glob)[lit], 99)) if n_lit else 1.0
    pngs = {}
    for kind, arr in (("direct", direct), ("global", glob)):
        img = (np.clip(arr / (scale + 1e-9), 0, 1) * 255).astype(np.uint8)
        img[~lit] = 0
        out = _write_png(store.output_path("previews", f"{label}_{kind}", ".png"), img)
        if out:
            store.record_output("previews", out, label=label, geometry_id=preset.geometry_id,
                                source=d)
        pngs[kind] = out
    units = {"global_fraction_median": "fraction of (direct + global), over lit px",
             "lit_pct": "% of camera px", "direct_median": "camera counts",
             "global_median": "camera counts", "checker_contrast": "fraction of the flood signal",
             "square_px": "DMD mirrors", "offsets": "DMD mirrors (dx, dy)"}
    doc = {"kind": "light_separation", "id": d.name, "label": label,
           "geometry_id": preset.geometry_id, "preset": preset.name, "mock": bool(preset.mock),
           "created": _now(), "square_px": square_px, "shifts": shifts,
           "offsets": [list(o) for o in offsets], "polarity_inverted": pol.inverted,
           "global_fraction_median": _finite_or_none(gfm), "lit_pct": round(lit_pct, 3),
           "direct_median": _finite_or_none(d_med), "global_median": _finite_or_none(g_med),
           "checker_contrast": _finite_or_none(contrast), "lmax_plus_lmin_vs_white":
           _finite_or_none(pair), "noise_sigma": round(sigma, 4),
           "direct_png": pngs["direct"], "global_png": pngs["global"],
           "method": "Nayar et al. 2006: direct = Lmax - Lmin, global = 2 (Lmin - dark)",
           "units": units, "warnings": warnings}
    store.write_json(d / "meta.json", doc)
    out_json = store.output_path("spectral", f"{label}_light", ".json")
    store.write_json(out_json, {**doc, "entry": str(d)})
    store.record_output("spectral", out_json, label=label, geometry_id=preset.geometry_id,
                        source=d)
    return LightSeparation(direct_png=pngs["direct"], global_png=pngs["global"],
                           global_fraction_median=gfm, square_px=square_px, shifts=shifts,
                           path=d, warnings=warnings, json_path=out_json,
                           lit_pct=round(lit_pct, 3),
                           direct_median=d_med, global_median=g_med, checker_contrast=contrast,
                           offsets=[tuple(o) for o in offsets], polarity_inverted=pol.inverted,
                           units=units)

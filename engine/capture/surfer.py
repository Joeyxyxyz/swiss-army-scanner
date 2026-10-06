"""
Surfer verbs — targeted work inside a held region, everything else
OPTICALLY BLACK.

    flood_verify           light only the region, photograph, say where it lands
    anti_glare_sweep       tilted / curved carrier fringes in the region; pick the
                           carrier with the best modulation and least clipping
    multi_orientation_scan V / H / +45 / -45 N-step phase in the region, fused by a
                           modulation-weighted circular mean

Black surround: patterns are built in PHYSICAL terms (0 = dark outside the
region) and the whole image is inverted when the raw HDMI path measures
inverted (zones._measure_polarity: raw black vs raw white). This avoids
Surfer aim.window_pattern's bug (outside forced to sent-0, which is
physically WHITE on an inverted path). Everything is sent raw (no duty /
gamma / invert chain) and projector.zone_mask is NOT used, since it forces
sent-0 which is only dark on a non-inverting path.

The LED light engine is incoherent: the 'wavefront' carriers here are a
pattern-modulation choice that moves specular hot spots / lowers clipping
in the camera, not optical wavefront shaping. A lock is "the carrier that
scored best here, now", not a measured reflectance change.

Region: 'atlas:<name>/<region>', a zone spec ('P17:P22@48'), a DMD rect
(x0, y0, x1, y1), or a bool DMD mask.
Outputs: library/output/previews/<ts>_<label>_*.png/.json, tuning/*.npz.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..decode import projmap
from ..library.store import safe_label
from . import atlas as A
from . import zones as Z

log = logging.getLogger(__name__)

SAT_LEVEL = 250.0              # camera counts: >= this is clipped (8-bit mono)
LIT_FRACTION_OF_PEAK = 0.35    # lit camera px: diff >= this x the 99.5th pct diff
LIT_MIN_COUNTS = 8.0           # ... and at least this many counts over the dark frame
MIN_LIT_PX = 50
LEAK_WARN = 0.15               # lit px outside the map-predicted area (fraction)
MOD_MIN = 3.0                  # counts: phase px below this modulation are masked

ORIENTATIONS = {               # unit direction of the fringe phase gradient (DMD x, y)
    "v": (1.0, 0.0),           # vertical lines: phase ~ DMD column
    "h": (0.0, 1.0),           # horizontal lines: phase ~ DMD row
    "d45": (np.sqrt(0.5), np.sqrt(0.5)),
    "d135": (np.sqrt(0.5), -np.sqrt(0.5)),
}
DEFAULT_CARRIERS = [(0.0, 0.0), (0.0, -0.5), (0.3, -0.5), (0.5, 0.0), (-0.3, 0.5)]


# ------------------------------------------------------------ helpers

def region_mask(session, preset, region):
    """-> (DMD bool mask, camera mask or None, description)."""
    if isinstance(region, np.ndarray):
        m = np.asarray(region, bool)
        if m.shape != Z.DMD_SHAPE:
            raise ValueError(f"region mask must be {Z.DMD_SHAPE}, got {m.shape}")
        return m, _cam_for(session, preset, m), "mask"
    if isinstance(region, (tuple, list)) and len(region) == 4 \
            and all(isinstance(v, (int, float, np.integer)) for v in region):
        m = A.rect_mask(region)
        return m, _cam_for(session, preset, m), f"rect {list(map(int, region))}"
    if A.is_atlas_spec(region):
        pm, cam, _ = A.resolve_region(session, preset, region)
        return pm, cam, region
    if isinstance(region, str):
        pm, cam, _ = Z.resolve(session, preset, region)
        return pm, cam, region
    raise ValueError(f"region must be 'atlas:<name>/<region>', a zone spec, a DMD rect or a "
                     f"DMD mask; got {region!r}")


def _cam_for(session, preset, pm):
    pmap = projmap.latest(session, preset.geometry_id)
    if pmap is None:
        return None
    cam = pmap.camera_mask_for_projector_mask(pm)
    return cam if cam.any() else None


def _check_level(level):
    if isinstance(level, bool) or not isinstance(level, (int, np.integer)) \
            or not 0 < int(level) <= 255:
        raise ValueError(f"level must be an int 1..255 (DMD grey), got {level!r}")
    return int(level)


def _send(sync, physical: np.ndarray, inverted: bool) -> np.ndarray:
    """Shoot a PHYSICAL image (0 = dark) raw, inverting the whole frame when
    the HDMI path is inverted, so the surround stays optically black."""
    img = np.clip(physical, 0, 255).astype(np.uint8)
    return sync.shoot((255 - img) if inverted else img, raw=True)


def _polarity(sync, warnings):
    inverted, _lit = Z._measure_polarity(sync, warnings)
    return inverted


def _bbox(mask: np.ndarray):
    v, u = np.nonzero(mask)
    if not v.size:
        return None
    return [int(u.min()), int(v.min()), int(u.max()) + 1, int(v.max()) + 1]


def _lit_mask(diff: np.ndarray) -> np.ndarray:
    peak = float(np.percentile(diff, 99.5))
    thr = max(LIT_FRACTION_OF_PEAK * peak, LIT_MIN_COUNTS)
    return diff >= thr


def _save_png(store, preset, label, suffix, img):
    return Z._save(store, preset, label, suffix, img)


def _save_doc(store, preset, label, ext, doc):
    return Z._save_json(store, preset, label, ext, doc)


def _rig_check(preset):
    projmap.dmd_shape_of(preset)


# -------------------------------------------------------- flood_verify

@dataclass
class FloodVerify:
    region: str
    level: int
    lands: bool                          # the lit region lands on something the camera sees
    verdict: str
    lit_bbox: Optional[list]             # camera px [x0, y0, x1, y1]
    lit_area_px: int
    lit_mean_counts: float               # mean (lit - dark) over lit px
    dark_mean_counts: float
    outside_mean_counts: float           # mean (lit - dark) outside the lit px
    expected_overlap: Optional[float]    # lit px inside the map-predicted area (with a map)
    polarity_inverted: bool
    dmd_px: int
    camera_png: Optional[object] = None
    diff_png: Optional[object] = None
    dmd_png: Optional[object] = None
    json_path: Optional[object] = None
    warnings: list = field(default_factory=list)


def flood_verify(session, preset, region, level: int = 255,
                 label: str = "flood_verify") -> FloodVerify:
    """Light ONLY the region (raw, measured polarity, black elsewhere),
    photograph it against a dark frame, and report where it lands."""
    safe_label(label)
    level = _check_level(level)
    _rig_check(preset)
    pm, cam, desc = region_mask(session, preset, region)
    if not pm.any():
        raise ValueError(f"region {desc!r} covers no DMD px")
    warnings = []
    physical = np.where(pm, level, 0).astype(np.uint8)
    with session.rig(preset) as sync:
        inverted = _polarity(sync, warnings)
        dark = _send(sync, np.zeros(Z.DMD_SHAPE, np.uint8), inverted)
        lit = _send(sync, physical, inverted)
    diff = lit.astype(np.float32) - dark.astype(np.float32)
    lm = _lit_mask(diff)
    area = int(lm.sum())
    lit_mean = float(diff[lm].mean()) if area else 0.0
    out_mean = float(diff[~lm].mean()) if (~lm).any() else 0.0
    overlap = None
    if cam is not None and area:
        overlap = round(float((lm & cam).sum()) / area, 4)
        if 1.0 - overlap > LEAK_WARN:
            warnings.append(f"{100 * (1 - overlap):.0f}% of the lit camera px fall outside the "
                            "map-predicted area: map stale, or light scatters onto the scene")
    lands = area >= MIN_LIT_PX and lit_mean >= LIT_MIN_COUNTS
    if lands:
        verdict = (f"region {desc} lands on a lit camera patch of {area} px "
                   f"(+{lit_mean:.1f} counts over dark)")
    else:
        verdict = (f"region {desc} does NOT visibly land on anything ({area} lit px, "
                   f"+{lit_mean:.1f} counts): off the part, blocked, or too dim")
    store = session.store
    cam_png = _save_png(store, preset, label, "camera", Z._photo_u8(lit))
    diff_png = _save_png(store, preset, label, "diff", Z._photo_u8(np.clip(diff, 0, None)))
    dmd_png = _save_png(store, preset, label, "dmd", physical)
    res = FloodVerify(region=desc, level=level, lands=lands, verdict=verdict,
                      lit_bbox=_bbox(lm), lit_area_px=area, lit_mean_counts=round(lit_mean, 2),
                      dark_mean_counts=round(float(dark.mean()), 2),
                      outside_mean_counts=round(out_mean, 3), expected_overlap=overlap,
                      polarity_inverted=bool(inverted), dmd_px=int(pm.sum()),
                      camera_png=cam_png, diff_png=diff_png, dmd_png=dmd_png,
                      warnings=warnings)
    doc = {k: v for k, v in vars(res).items() if k != "json_path"}
    doc.update({"kind": "flood_verify", "geometry_id": preset.geometry_id,
                "mock": bool(preset.mock), "created": Z._now(),
                "units": {"lit_bbox": "camera px [x0, y0, x1, y1]",
                          "*_counts": "camera counts (lit - dark)"}})
    res.json_path = _save_doc(store, preset, label, ".flood.json", doc)
    return res


# ---------------------------------------------------- anti_glare_sweep

def carrier_pattern(pm: np.ndarray, fx: float, fy: float, z: float = 0.0,
                    shift: float = 0.0, cycles: float = 6.0, level: int = 255) -> np.ndarray:
    """Physical carrier fringe inside the region (0 outside): a tilt (fx, fy,
    in units of `cycles` across the region) plus a Fresnel-like quadratic
    term z, shifted by `shift` rad. fx = fy = z = 0 is a flat flood."""
    ys, xs = np.nonzero(pm)
    if not ys.size:
        return np.zeros(Z.DMD_SHAPE, np.uint8)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    u = 2 * (xx - x0) / max(x1 - x0, 1) - 1
    v = 2 * (yy - y0) / max(y1 - y0, 1) - 1
    ph = 2 * np.pi * cycles * (fx * u + fy * v) + np.pi * cycles * z * (u * u + v * v) + shift
    out = np.zeros(Z.DMD_SHAPE, np.float32)
    out[y0:y1, x0:x1] = level * (0.5 + 0.5 * np.cos(ph))
    out[~pm] = 0
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


@dataclass
class GlareLock:
    region: str
    lock: Optional[tuple]                # (fx, fy, z) that scored best
    scores: list                         # per carrier dict
    baseline_clip_frac: float            # flat flood clipping in the region
    wording: str
    polarity_inverted: bool
    json_path: Optional[object] = None
    warnings: list = field(default_factory=list)


def _norm_carrier(c):
    t = tuple(float(v) for v in c)
    if len(t) == 2:
        t = t + (0.0,)
    if len(t) != 3 or not all(np.isfinite(t)):
        raise ValueError(f"carrier must be (fx, fy) or (fx, fy, z), got {c!r}")
    return t


def anti_glare_sweep(session, preset, region, carriers=None, steps: int = 4,
                     cycles: float = 6.0, level: int = 255,
                     label: str = "anti_glare") -> GlareLock:
    """For each carrier, project `steps` phase shifts in the region (black
    elsewhere) and score it in the lit camera area: median modulation x
    (1 - clipped fraction). The best-scoring carrier is the lock."""
    safe_label(label)
    level = _check_level(level)
    if not isinstance(steps, int) or steps < 3:
        raise ValueError("steps must be an int >= 3")
    cs = [_norm_carrier(c) for c in (carriers or DEFAULT_CARRIERS)]
    _rig_check(preset)
    pm, cam, desc = region_mask(session, preset, region)
    warnings, scores = [], []
    deltas = 2 * np.pi * np.arange(steps) / steps
    with session.rig(preset) as sync:
        inverted = _polarity(sync, warnings)
        dark = _send(sync, np.zeros(Z.DMD_SHAPE, np.uint8), inverted)
        flood = _send(sync, np.where(pm, level, 0).astype(np.uint8), inverted)
        lm = _lit_mask(flood - dark)
        if cam is not None:
            lm = lm & cam if (lm & cam).sum() >= MIN_LIT_PX else lm
        if lm.sum() < MIN_LIT_PX:
            warnings.append("the region barely lights the camera; scores are unreliable")
            lm = cam if cam is not None else np.ones(flood.shape, bool)
        base_clip = float((flood[lm] >= SAT_LEVEL).mean())
        for fx, fy, z in cs:
            frames = [_send(sync, carrier_pattern(pm, fx, fy, z, d, cycles, level), inverted)
                      for d in deltas]
            st = np.stack(frames).astype(np.float32)
            s = np.tensordot(np.sin(deltas), st, 1)
            c = np.tensordot(np.cos(deltas), st, 1)
            mod = (2.0 / steps) * np.hypot(s, c)
            clip = float((st[:, lm] >= SAT_LEVEL).any(axis=0).mean())
            med = float(np.median(mod[lm]))
            scores.append({"fx": fx, "fy": fy, "z": z, "median_modulation": round(med, 3),
                           "clip_frac": round(clip, 4),
                           "score": round(med * (1.0 - clip), 3)})
    best = max(scores, key=lambda r: r["score"]) if scores else None
    lock = (best["fx"], best["fy"], best["z"]) if best and best["score"] > 0 else None
    if lock is None:
        wording = ("no carrier gave usable modulation in this region: no lock "
                   "(check flood_verify first)")
    else:
        wording = (f"lock fx={lock[0]:g}, fy={lock[1]:g}, z={lock[2]:g}: best modulation x "
                   f"(1 - clipping) of {len(cs)} carriers in {desc} under current lighting "
                   f"(clipped {100 * best['clip_frac']:.1f}% vs flat flood "
                   f"{100 * base_clip:.1f}%). Incoherent LED: a pattern choice that dodges "
                   "glare, not optical wavefront shaping; re-check if the part or light moves")
    res = GlareLock(region=desc, lock=lock, scores=scores,
                    baseline_clip_frac=round(base_clip, 4), wording=wording,
                    polarity_inverted=bool(inverted), warnings=warnings)
    doc = {k: v for k, v in vars(res).items() if k != "json_path"}
    doc.update({"kind": "anti_glare_sweep", "geometry_id": preset.geometry_id,
                "mock": bool(preset.mock), "created": Z._now(), "steps": steps,
                "cycles": cycles, "sat_level": SAT_LEVEL})
    res.json_path = _save_doc(session.store, preset, label, ".glare.json", doc)
    return res


# ------------------------------------------------ multi_orientation_scan

def fringe_pattern(pm: np.ndarray, orientation: str, shift: float, period_px: float,
                   level: int = 255) -> np.ndarray:
    """Physical sinusoidal fringe of one orientation inside the region."""
    if orientation not in ORIENTATIONS:
        raise ValueError(f"orientation must be one of {tuple(ORIENTATIONS)}, got {orientation!r}")
    dx, dy = ORIENTATIONS[orientation]
    yy, xx = np.indices(Z.DMD_SHAPE, dtype=np.float32)
    s = dx * xx + dy * yy
    out = level * (0.5 + 0.5 * np.cos(2 * np.pi * s / period_px + shift))
    return np.where(pm, np.rint(out), 0).astype(np.uint8)


@dataclass
class MultiOrientation:
    region: str
    orientations: tuple
    phase: dict                  # orientation -> wrapped phase (rad), camera image, NaN = masked
    modulation: dict             # orientation -> modulation (counts)
    fused_phase: np.ndarray      # modulation-weighted circular mean (rad), NaN = masked
    fused_modulation: np.ndarray # sum(m^2) / sum(m): modulation-weighted modulation (counts)
    quality: np.ndarray          # |sum m e^{i phi}| / sum m, 0..1 (orientation agreement)
    mask: np.ndarray             # camera px usable (region lit, modulation >= MOD_MIN)
    stats: dict
    note: str
    polarity_inverted: bool
    npz_path: Optional[object] = None
    json_path: Optional[object] = None
    warnings: list = field(default_factory=list)


FUSION_NOTE = (
    "Only 'v' phase is (wrapped) DMD column / period and only 'h' phase is (wrapped) DMD "
    "row / period; d45 / d135 phases are (col +/- row) / (sqrt2 * period). The fused phase "
    "is a modulation-weighted circular mean across orientations (Surfer method): it is NOT "
    "a projector coordinate. Use per-axis 'v' / 'h' phases for correspondence and "
    "fused_modulation / quality as the per-px confidence / hole-fill weight map.")


def multi_orientation_scan(session, preset, region, orientations=("v", "h", "d45", "d135"),
                           steps: int = 4, period_px: float = 24, level: int = 255,
                           label: str = "multi_orient") -> MultiOrientation:
    """N-step phase for each orientation inside the region (black elsewhere),
    then a modulation-weighted fusion. See FUSION_NOTE."""
    safe_label(label)
    level = _check_level(level)
    if not isinstance(steps, int) or steps < 3:
        raise ValueError("steps must be an int >= 3")
    if not float(period_px) >= 4:
        raise ValueError("period_px must be >= 4 DMD px")
    ors = tuple(orientations)
    for o in ors:
        if o not in ORIENTATIONS:
            raise ValueError(f"orientation must be in {tuple(ORIENTATIONS)}, got {o!r}")
    if not ors:
        raise ValueError("need at least one orientation")
    _rig_check(preset)
    pm, cam, desc = region_mask(session, preset, region)
    warnings = []
    deltas = 2 * np.pi * np.arange(steps) / steps
    phase, modul = {}, {}
    with session.rig(preset) as sync:
        inverted = _polarity(sync, warnings)
        dark = _send(sync, np.zeros(Z.DMD_SHAPE, np.uint8), inverted)
        flood = _send(sync, np.where(pm, level, 0).astype(np.uint8), inverted)
        for o in ors:
            st = np.stack([_send(sync, fringe_pattern(pm, o, d, period_px, level), inverted)
                           for d in deltas]).astype(np.float32)
            s = np.tensordot(np.sin(deltas), st, 1)
            c = np.tensordot(np.cos(deltas), st, 1)
            # I_k = A + B cos(phi + d_k): sum I sin = -B N/2 sin phi, sum I cos = B N/2 cos phi
            phase[o] = np.arctan2(-s, c).astype(np.float32)
            modul[o] = ((2.0 / steps) * np.hypot(s, c)).astype(np.float32)
    lit = _lit_mask(flood - dark)
    if cam is not None:
        lit &= cam
    M = np.stack([modul[o] for o in ors])
    P = np.stack([phase[o] for o in ors])
    wsum = M.sum(0)
    z = (M * np.exp(1j * P)).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        fmod = np.where(wsum > 0, (M * M).sum(0) / wsum, 0.0).astype(np.float32)
        qual = np.where(wsum > 0, np.abs(z) / wsum, 0.0).astype(np.float32)
    mask = lit & (fmod >= MOD_MIN)
    fphase = np.where(mask, np.angle(z), np.nan).astype(np.float32)
    for o in ors:
        phase[o] = np.where(lit & (modul[o] >= MOD_MIN), phase[o], np.nan).astype(np.float32)
    fmod = np.where(lit, fmod, 0.0).astype(np.float32)
    qual = np.where(lit, qual, 0.0).astype(np.float32)
    stats = {"lit_px": int(lit.sum()), "usable_px": int(mask.sum()),
             "median_modulation": {o: round(float(np.median(modul[o][lit])), 3) if lit.any()
                                   else 0.0 for o in ors},
             "median_fused_modulation": round(float(np.median(fmod[mask])), 3)
             if mask.any() else 0.0,
             "median_quality": round(float(np.median(qual[mask])), 4) if mask.any() else 0.0}
    if not mask.any():
        warnings.append("no camera px reached the modulation floor: region too dim / too thin "
                        "(Surfer bench: a 48 px column gives weak SNR)")
    res = MultiOrientation(region=desc, orientations=ors, phase=phase, modulation=modul,
                           fused_phase=fphase, fused_modulation=fmod, quality=qual, mask=mask,
                           stats=stats, note=FUSION_NOTE, polarity_inverted=bool(inverted),
                           warnings=warnings)
    store = session.store
    npz = store.output_path("tuning", label, ".npz")
    store.write_npz(npz, fused_phase=fphase, fused_modulation=fmod, quality=qual, mask=mask,
                    **{f"phase_{o}": phase[o] for o in ors},
                    **{f"modulation_{o}": modul[o] for o in ors})
    store.record_output("tuning", npz, label=label, geometry_id=preset.geometry_id)
    res.npz_path = npz
    _save_png(store, preset, label, "fused_modulation", Z._photo_u8(fmod))
    doc = {"kind": "multi_orientation_scan", "region": desc, "orientations": list(ors),
           "steps": steps, "period_px": period_px, "geometry_id": preset.geometry_id,
           "mock": bool(preset.mock), "created": Z._now(), "stats": stats, "note": FUSION_NOTE,
           "npz": npz, "polarity_inverted": bool(inverted), "warnings": warnings}
    res.json_path = _save_doc(store, preset, label, ".multi.json", doc)
    return res

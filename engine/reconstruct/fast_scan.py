"""
HDMI fast zone scan behind api.scan_fast: the column-axis multifreq set only (default 4 steps x
periods 8 / 80 / 1280 DMD px + the white / black floods = 14 patterns), lit through a zone
(grid cells, an atlas region, a DMD mask, or located end faces + a margin) and captured with
per-run camera averaging and HDMI hold overrides (the preset file is never edited), then
ray-plane triangulated (the column axis carries the depth; the calibration's row axis is not
used) -> a normal metric scan entry, plus the flood pair it captured (flood.npz: white, black,
float32 camera DN) for the station measurement's cell outlines.

    metric/<geometry_id>/<ts>_<label>/  field.npz, cloud.npz, flood.npz, preview.png, meta.json

Bench (HDMI video mode, bench_9in_v1): 4 steps x 3 periods + floods, average 2, hold 0.12 s =
5.3 s for the whole station scene, repeating the slow reference scan's heights; average 1 is
not enough (the free-running camera catches the DLPC350's frame sequence differently every
frame: HDMI flicker), so average >= 2 is enforced.
"""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from typing import Optional

import numpy as np

from .. import config
from ..calibrate import flow
from ..library.store import safe_label
from . import metric_scan
from .triangulate import triangulate

FAST_STEPS = 4
FAST_PERIODS = (8.0, 80.0, 1280.0)
FAST_AVERAGE = 2
MIN_AVERAGE = 2
FAST_HOLD_S = 0.12
DMD_SHAPE = (1140, 912)
FLICKER_WARNING = ("camera average {n} is below {m}: over HDMI the free-running camera catches a "
                   "different slice of the DLPC350's R/G/B + PWM frame sequence every frame, so a "
                   "single frame per pattern flickers (the bench's 1-frame fast scans were not "
                   "repeatable); average raised to {m}")


def _dmd_mask(zone) -> Optional[np.ndarray]:
    """A DMD mask given as an array or a .npy file -> bool (1140, 912); None if not a mask."""
    if isinstance(zone, np.ndarray):
        m = zone
    elif isinstance(zone, (str, Path)) and str(zone).lower().endswith(".npy") and Path(zone).is_file():
        m = np.load(Path(zone))
    else:
        return None
    m = np.asarray(m).astype(bool)
    if m.shape != DMD_SHAPE:
        raise ValueError(f"a DMD zone mask is {DMD_SHAPE[0]} rows x {DMD_SHAPE[1]} cols (native "
                         f"DMD px), got {m.shape}")
    if not m.any():
        raise ValueError("the DMD zone mask is empty: nothing would be lit")
    return m


def resolve_zone(s, preset, zone=None, faces=None, margin_dmd_px: float = 6.0) -> tuple:
    """-> (DMD mask or None = full field, camera mask or None, description)."""
    proj = cam = None
    desc = []
    if zone is not None and not (isinstance(zone, str) and not zone.strip()):
        m = _dmd_mask(zone)
        if m is not None:
            proj = m
            desc.append("dmd mask" + (f" {Path(zone).name}" if not isinstance(zone, np.ndarray) else ""))
        else:
            from ..capture import zones as zones_mod
            proj, cam, _ = zones_mod.resolve(s, preset, str(zone))
            proj = np.asarray(proj, bool)
            desc.append(str(zone))
    if faces:
        from ..measure.endface import patterns
        from ..measure.endface import verbs as face_verbs
        fm = np.zeros(DMD_SHAPE, bool)
        names = []
        for f in faces:
            t = face_verbs.as_face(s, f, preset, "scan_fast")          # rule 5
            fm |= np.asarray(patterns.face_mask(t.dmd_ellipse, float(margin_dmd_px)), bool)
            names.append(t.label)
        proj = fm if proj is None else (proj | fm)
        cam = None                        # a face ellipse has no grid-cell camera cut
        desc.append("faces " + ",".join(names) + f" +{margin_dmd_px:g} DMD px")
    return proj, cam, (" + ".join(desc) if desc else "full field")


def scan_fast(s, preset, zone=None, steps: int = FAST_STEPS, periods=FAST_PERIODS,
              average: int = FAST_AVERAGE, min_hold_s: float = FAST_HOLD_S, axis: str = "x",
              label: str = "fast", faces=None, margin_dmd_px: float = 6.0):
    safe_label(label)
    if axis not in ("x", "xy"):
        raise ValueError("axis must be 'x' (columns: the depth axis) or 'xy'; the column axis "
                         "carries the depth on a left / right rig")
    steps = int(steps)
    if steps < 3:
        raise ValueError("steps must be >= 3 (phase shifting)")
    periods = [float(p) for p in (periods or FAST_PERIODS)]
    if not periods or any(p <= 0 for p in periods):
        raise ValueError("periods must be positive DMD px")
    if not isinstance(average, (int, np.integer)) or isinstance(average, bool) or average < 1:
        raise ValueError("average must be a whole number >= 1")
    if min_hold_s is None or not (0.0 <= float(min_hold_s) <= 5.0):
        raise ValueError("min_hold_s must be 0..5 s")
    warnings = []
    if average < MIN_AVERAGE:
        warnings.append(FLICKER_WARNING.format(n=int(average), m=MIN_AVERAGE))
        average = MIN_AVERAGE
    fp = config.geometry_fingerprint(preset)
    cal = metric_scan._require_cal(s, preset.geometry_id, fp, "scan_fast")    # rule 6, before the rig
    rig = metric_scan._require_rig(s, preset, "scan_fast")        # a mobile rig: verified
    run = replace(preset, camera=replace(preset.camera, average=int(average)),
                  projection=replace(preset.projection, min_hold_s=float(min_hold_s)))
    basis = flow.multifreq_basis(run, cal.diamond_parity, steps, periods, axis=axis)
    proj_mask, cam_mask, zdesc = resolve_zone(s, run, zone, faces, margin_dmd_px)
    pats = basis.generate()
    t0 = time.perf_counter()
    frames = flow.capture_patterns(s, run, pats, "a fast scan", zone_mask=proj_mask)
    capture_s = time.perf_counter() - t0
    fld = basis.decode(frames, preset.geometry_id, contrast_min=preset.decode.contrast_min,
                       conf_min=preset.decode.conf_min)
    if cam_mask is not None:
        fld.mask = fld.mask & np.asarray(cam_mask, bool)
    cloud = triangulate(fld, cal, "plane")
    fld.meta.update({"preset": preset.name, "label": label, "mock": s.mock})
    a, b = (np.asarray(f, np.float32) for f in frames[:2])
    white, black = (a, b) if float(a.mean()) >= float(b.mean()) else (b, a)
    extra = {"fast": True, "zone": zdesc, "steps": basis.steps, "periods_px": basis.periods,
             "axis": axis, "average": int(average), "min_hold_s": float(min_hold_s),
             "n_patterns": len(pats), "capture_s": round(capture_s, 2),
             "per_pattern_s": round(capture_s / max(len(pats), 1), 3),
             "diamond_parity": cal.diamond_parity, "flood": "flood.npz (white, black)",
             "warnings": warnings, "rig": rig}
    return metric_scan._save(s, fld, cloud, label, cal, fp, preset.name, extra,
                             arrays={"flood.npz": {"white": white, "black": black}})

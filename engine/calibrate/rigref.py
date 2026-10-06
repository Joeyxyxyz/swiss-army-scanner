"""
Rig check behind api.rig_check: a fresh flood + random-block pattern vs the
reference stored for this geometry, to catch a nudged camera or projector
before it silently skews every measurement.

Library layout (append-only, rule 7):
    calibrations/<geometry_id>/<ts>_rigref/   rigref.npz (white, black, pattern),
                                              flood.png, rigref.json (last)

The reference is bound to the geometry fingerprint and records the newest
base and stereo calibration it was taken with. The first check at a geometry
(or set_reference=True) stores a new reference: take it right after base /
calibration, with the station in its usual empty state. A later check then
compares (numbers in engine/capture/rigcheck.py, pure and unit-tested).
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

import numpy as np

from .. import config, rules
from ..capture import rigcheck
from ..library.store import json_safe
from . import flow, stereo

REF_FILE = "rigref.json"


def latest_reference(store, geometry_id: str, fp: Optional[dict]) -> Optional[dict]:
    for d in reversed(store.entries("calibrations", geometry_id)):
        f = d / REF_FILE
        if not f.is_file():
            continue
        try:
            meta = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if rules.fingerprint_diffs(meta.get("geometry_fp"), fp):
            continue
        with np.load(d / "rigref.npz") as z:
            meta["frames"] = {k: z[k] for k in ("white", "black", "pattern")}
        meta["path"] = d
        return meta
    return None


def _working_distance(store, geometry_id: str, cal) -> Optional[float]:
    """Median camera-frame z (mm) of the newest metric scan made with this calibration."""
    if cal is None:
        return None
    from ..reconstruct import metric_scan
    for ms in reversed(metric_scan.list_metric(store, geometry_id)):
        if ms.calibration_id != cal.id:
            continue
        try:
            z = ms.cloud.points[:, 2]
        except (OSError, ValueError, KeyError):
            continue
        z = z[np.isfinite(z)]
        if z.size:
            return float(np.median(z))
    return None


def _mock_shift(frames: list, shift) -> list:
    from scipy import ndimage
    dx, dy = (float(v) for v in shift)
    return [ndimage.shift(np.asarray(f, np.float64), (dy, dx), order=3, mode="nearest")
            for f in frames]


def rig_check(s, preset, set_reference: bool = False, tol_px: float = rigcheck.DEFAULT_TOL_PX,
              mock_shift_px=None, label: str = "rigref") -> dict:
    if mock_shift_px is not None and not s.mock:
        raise rules.RuleViolation("mock_shift_px is mock-only (it fakes a nudged camera)")
    fp = config.geometry_fingerprint(preset)
    h, w = preset.projection.size[1], preset.projection.size[0]
    pats = [np.full((h, w), 255, np.uint8), rigcheck.random_block_pattern((h, w)),
            np.zeros((h, w), np.uint8)]                 # black last: the DMD ends dark
    white, pattern, black = flow.capture_patterns(s, preset, pats, "the rig check")
    if float(np.mean(black)) > float(np.mean(white)):   # an inverting HDMI path (projection.invert)
        white, black = black, white
        pattern = white + black - pattern
    if mock_shift_px is not None:
        white, pattern, black = _mock_shift([white, pattern, black], mock_shift_px)
    from .. import api
    base = api.latest_base(preset)
    cal = stereo.latest(s.store, preset.geometry_id, fp)
    ref = None if set_reference else latest_reference(s.store, preset.geometry_id, fp)
    out = {"geometry_id": preset.geometry_id, "preset": preset.name, "tol_px": tol_px,
           "base_id": base.id if base is not None else None,
           "calibration_id": cal.id if cal is not None else None, "mock": s.mock}
    if ref is None:
        d = s.store.new_dir("calibrations", preset.geometry_id, label=label)
        with s.store.filling(d):
            s.store.write_npz(d / "rigref.npz", white=np.asarray(white, np.float32),
                              black=np.asarray(black, np.float32),
                              pattern=np.asarray(pattern, np.float32))
            s.store.write_preview(d / "flood.png", np.asarray(white, np.float64))
            meta = {**out, "kind": "rigref", "geometry_fp": fp,
                    "created": datetime.now().isoformat(timespec="seconds")}
            s.store.write_json(d / REF_FILE, json_safe(meta))
        return {**out, "status": "REFERENCE_SET", "reference_id": d.name, "path": d,
                "flood": None, "pattern": None, "shift_px": None, "shift_mm": None,
                "message": "no rig reference at this geometry yet: stored this one (take it "
                           "right after base / calibration, station in its usual empty state)"}
    fr = ref["frames"]
    fl = rigcheck.phase_shift(fr["white"], white)
    pt = rigcheck.phase_shift(rigcheck.normalised_pattern(fr["pattern"], fr["white"], fr["black"]),
                              rigcheck.normalised_pattern(pattern, white, black))
    dist = _working_distance(s.store, preset.geometry_id, cal)
    mm_per_px = (dist / float(cal.K_cam[0][0])) if (dist and cal is not None) else None
    verdict = rigcheck.assess(fl, pt, tol_px, mm_per_px, dist)
    warnings = []
    if ref.get("calibration_id") != out["calibration_id"]:
        warnings.append(f"the reference was taken with calibration {ref.get('calibration_id')}, "
                        f"the current one is {out['calibration_id']}: set a new reference "
                        "(set_reference) once the rig is confirmed good")
    if mm_per_px is None:
        warnings.append("shift in camera px only: mm need a stereo calibration and a metric "
                        "scan at this geometry (the working distance)")
    return {**out, **verdict, "reference_id": ref["path"].name, "flood": fl, "pattern": pt,
            "working_distance_mm": round(dist, 1) if dist else None,
            "mm_per_px": round(mm_per_px, 5) if mm_per_px else None, "warnings": warnings}

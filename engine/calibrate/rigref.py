"""
Rig check behind api.rig_check: a fresh flood + random-block pattern vs the
reference stored for this geometry, to catch a nudged camera or projector
before it silently skews every measurement.

Library layout (append-only, rule 7):
    calibrations/<geometry_id>/<ts>_rigref/   rigref.npz (white, black, pattern),
                                              flood.png, rigref.json (last)
    calibrations/<geometry_id>/rig_state.jsonl   every check's result, one line each
                                              (engine/rigstate.py)

The reference is bound to the geometry fingerprint and records the newest
base and stereo calibration it was taken with. The first check at a geometry
(or set_reference=True) stores a new reference: take it right after base /
calibration, with the station in its usual empty state. A later check then
compares (numbers in engine/capture/rigcheck.py, pure and unit-tested).
On a MOBILE rig that was just calibrated where it stands (verified by that calibration), a
check that finds only another calibration's reference stores this calibration's own instead
of comparing with where the rig stood before.
A reference also records what it can be trusted for (`rig_trust`, engine/rigstate.py):
whether the rig was verified (the mobile rule) and whether no rig check objection stood
(the fixed rule; or the operator asked for it) at the moment it was stored. Every check
made against it carries that with it: an OK against a reference taken while the rig was in
doubt verifies nothing. Only an OK at the default tolerance or tighter counts for a mm
result; a looser one is reported and says so.
The REFERENCE_SET message says why a reference was stored: the first one at
this geometry, a new calibration (the previous reference belonged to another
one), on request, or after a re-home.

Re-home (rehome=True; `scan rig-check --rehome`): when the check finds a small
CAMERA move (the flood and the projected pattern moved together), the camera
rotation is fitted from the reference flood vs the fresh flood (tile shifts on
undistorted coordinates about the principal point; rigcheck.camera_rotation)
and written as a NEW stereo calibration R1 = R0 @ Rd, t unchanged
(stereo.rehome), flagged "re-homed (rotation only) from <calibration id>";
the fresh frames become the new calibration's reference. The original
calibration is never changed. Refused (nothing written) when the motion is not
camera-like (the projector moved), too large, or badly registered. Rotation
only: a camera translation is not corrected, so the absolute depth may keep a
small offset - a full calibration is the reference.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
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
    """Mock only: the fresh frames moved as by a nudged camera - (dx, dy) px, optionally
    (dx, dy, roll_deg) or (dx, dy, roll_deg, scale) about the image centre."""
    from scipy import ndimage
    v = [float(x) for x in shift]
    if not 2 <= len(v) <= 4 or not all(np.isfinite(v)):
        raise ValueError(f"mock_shift_px must be [dx, dy] (+ roll_deg, + scale), got {shift!r}")
    dx, dy = v[0], v[1]
    roll = v[2] if len(v) > 2 else 0.0
    scale = v[3] if len(v) > 3 else 1.0
    if roll == 0.0 and scale == 1.0:
        return [ndimage.shift(np.asarray(f, np.float64), (dy, dx), order=3, mode="nearest")
                for f in frames]
    if not scale > 0:
        raise ValueError(f"mock scale must be > 0, got {scale!r}")
    out = []
    for f in frames:
        a = np.asarray(f, np.float64)
        H, W = a.shape[:2]
        c = np.array([(H - 1) / 2.0, (W - 1) / 2.0])
        t = np.radians(roll)
        # output (row, col) q <- input p = M^-1 (q - c - d) + c, M = scale x rotation by roll
        # (x right, y down: the inverse of [[cos, sin], [-sin, cos]] in (row, col) order)
        R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]) / scale
        off = c - R @ (c + np.array([dy, dx]))
        out.append(ndimage.affine_transform(a, R, offset=off, order=3, mode="nearest"))
    return out


REHOME_HOWTO = "scan rig-check --rehome"
REHOME_CAVEAT = ("rotation only: a camera translation is not corrected, so the absolute depth "
                 "may keep a small offset - a full calibration (calib-pose + calibrate-stereo) "
                 "is the reference")
TAKE_IT = "take it right after base / calibration, station in its usual empty state"


def _store_reference(s, preset, out: dict, fp, white, pattern, black, label: str,
                     why: str, trust: Optional[dict] = None) -> Path:
    """A new reference entry (append-only) from these frames; `why` and what it can be
    trusted for (`trust`: rigstate.trust_now at this moment) are recorded with it."""
    d = s.store.new_dir("calibrations", preset.geometry_id, label=label)
    with s.store.filling(d):
        s.store.write_npz(d / "rigref.npz", white=np.asarray(white, np.float32),
                          black=np.asarray(black, np.float32),
                          pattern=np.asarray(pattern, np.float32))
        s.store.write_preview(d / "flood.png", np.asarray(white, np.float64))
        meta = {**out, "kind": "rigref", "geometry_fp": fp, "reason": why,
                "rig_trust": trust, "mount": preset.rig.mount,
                "created": datetime.now().isoformat(timespec="seconds")}
        s.store.write_json(d / REF_FILE, json_safe(meta))
    return d


def _reference_message(prev: Optional[dict], cal, requested: bool) -> tuple:
    """(reason, message) of a REFERENCE_SET: why this reference was stored. prev: the
    reference that was the newest one until now (None: none at this geometry)."""
    cid = cal.id if cal is not None else None
    if prev is None:
        if cid is None:
            return "first", (f"no rig reference at this geometry yet: stored this one "
                             f"({TAKE_IT})")
        return "new_calibration", (f"new calibration {cid} (no reference was taken with it "
                                   f"yet) -> new reference stored, the first at this "
                                   f"geometry ({TAKE_IT})")
    old = f"the previous reference {prev['path'].name}"
    if prev.get("calibration_id") != cid and cid is not None:
        was = prev.get("calibration_id") or "no stereo calibration"
        return "new_calibration", (f"new calibration {cid} -> new reference stored ({old} "
                                   f"belonged to {was}; it stays in the library, unused; "
                                   f"{TAKE_IT})")
    return "requested", (f"new reference stored on request ({old} stays in the library, "
                         f"unused; {TAKE_IT})")


def _mobile_verified(s, preset) -> bool:
    """Is this a mobile rig (engine/rigstate.mount_of) that is verified at this moment?"""
    from .. import rigstate
    pdir = getattr(s, "presets_dir", None)
    if rigstate.mount_of(s.store, preset, pdir)[0] != "mobile":
        return False
    return rigstate.state(s.store, preset, presets_dir=pdir)["verified"] is True


def _camera_like(verdict: dict) -> bool:
    return verdict.get("status") == "MOVED" and verdict.get("moved") == "camera"


def _rehome_fit(ref: dict, white, pattern, black, cal) -> dict:
    """rigcheck.rehome_fit on a fine registration of the reference vs the fresh frames."""
    fr = ref["frames"]
    kw = {"grid": rigcheck.REHOME_GRID, "tile_upsample": rigcheck.REHOME_TILE_UPSAMPLE}
    fl = rigcheck.motion(fr["white"], white, **kw)
    pt = rigcheck.motion(rigcheck.normalised_pattern(fr["pattern"], fr["white"], fr["black"]),
                         rigcheck.normalised_pattern(pattern, white, black), **kw)
    return rigcheck.rehome_fit(fl, pt, cal.K_cam, cal.dist_cam)


def _rehome_public(fit: dict) -> dict:
    """The fit's numbers for the result (matrices as lists, rounded)."""
    out = {}
    for k, v in fit.items():
        if k in ("R", "R_new_from_old"):
            continue
        if isinstance(v, float):
            v = round(v, 5)
        elif isinstance(v, list):
            v = [round(float(x), 5) for x in v]
        out[k] = v
    return out


def rig_check(s, preset, set_reference: bool = False, tol_px: float = rigcheck.DEFAULT_TOL_PX,
              mock_shift_px=None, label: str = "rigref", rehome: bool = False) -> dict:
    """The rig check (module docstring); its result is also appended to the geometry's
    rig-state record (engine/rigstate.py: what a fixed rig is trusted until, what verifies a
    mobile one)."""
    from .. import rigstate
    res = _rig_check(s, preset, set_reference, tol_px, mock_shift_px, label, rehome)
    try:
        rigstate.record_check(s.store, preset, res)
    except OSError as e:                    # the check stands; the record says nothing new
        res["warnings"] = [*(res.get("warnings") or []),
                           f"the rig-state record was not updated ({type(e).__name__}: {e}): "
                           "a mobile rig stays unverified"]
    return res


def _rig_check(s, preset, set_reference: bool = False, tol_px: float = rigcheck.DEFAULT_TOL_PX,
               mock_shift_px=None, label: str = "rigref", rehome: bool = False) -> dict:
    if mock_shift_px is not None and not s.mock:
        raise rules.RuleViolation("mock_shift_px is mock-only (it fakes a nudged camera)")
    tol_px = rigcheck.check_tol(tol_px)                 # before any frame is captured
    fp = config.geometry_fingerprint(preset)
    if rehome:                                          # refused before any frame is captured
        if set_reference:
            raise ValueError("rehome and set_reference exclude each other: a re-home is fitted "
                             "against the stored reference (and stores the new one itself)")
        cal0 = stereo.latest(s.store, preset.geometry_id, fp)
        ref0 = latest_reference(s.store, preset.geometry_id, fp)
        if cal0 is None:
            raise ValueError(f"re-home: no stereo calibration at geometry {preset.geometry_id} "
                             "(these settings) to re-home: calibrate first")
        if ref0 is None:
            raise ValueError("re-home: no rig reference at this geometry: the camera rotation "
                             "is fitted against the reference taken right after the "
                             "calibration, and none was stored - recalibrate (or, with the "
                             "rig known good, store one with set_reference)")
        if ref0.get("calibration_id") != cal0.id:
            raise ValueError(f"re-home: the rig reference {ref0['path'].name} belongs to "
                             f"{ref0.get('calibration_id') or 'no stereo calibration'}, not to "
                             f"the calibration in use ({cal0.id}): there is no reference of "
                             "this calibration to measure the camera's move against - "
                             "recalibrate")
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
    prev = latest_reference(s.store, preset.geometry_id, fp)
    ref = None if set_reference else prev
    if ref is not None and cal is not None and ref.get("calibration_id") != cal.id \
            and not rehome and _mobile_verified(s, preset):
        # a MOBILE rig calibrated where it stands now: the newest reference is another
        # calibration's, taken where the rig stood THEN. Comparing with it says nothing
        # about this calibration (and a MOVED would undo what the calibration just
        # verified): this check stores the calibration's own reference instead
        ref = None
    out = {"geometry_id": preset.geometry_id, "preset": preset.name, "tol_px": tol_px,
           "base_id": base.id if base is not None else None,
           "calibration_id": cal.id if cal is not None else None, "mock": s.mock}
    if ref is None:
        from .. import rigstate
        reason, message = _reference_message(prev, cal, bool(set_reference))
        # what this reference can verify later: the rig's state at this moment
        pdir = getattr(s, "presets_dir", None)
        trust = rigstate.trust_now(s.store, preset, requested=bool(set_reference),
                                   presets_dir=pdir)
        d = _store_reference(s, preset, out, fp, white, pattern, black, label, reason, trust)
        warnings = []
        if rigstate.mount_of(s.store, preset, pdir)[0] == "mobile":
            if not trust["mobile"]:
                warnings.append("this reference was stored while the mobile rig was NOT "
                                "verified: a rig check against it verifies nothing (calibrate "
                                "from poses captured now, then store the reference: "
                                "set_reference)")
        elif not trust["fixed"]:
            warnings.append("this reference was stored while a rig check's MOVED / UNKNOWN "
                            "stood: a rig check against it does not clear that (with the rig "
                            "known good: set_reference)")
        return {**out, "status": "REFERENCE_SET", "reference_id": d.name, "path": d,
                "flood": None, "pattern": None, "shift_px": None, "shift_mm": None,
                "reference_reason": reason, "requested": bool(set_reference),
                "reference_calibration_id": out["calibration_id"], "reference_trust": trust,
                "previous_reference_id": prev["path"].name if prev is not None else None,
                "message": message, "warnings": warnings}
    fr = ref["frames"]
    # shift + roll + scale over the frame (a global shift alone misses a roll or a zoom /
    # focus nudge about the image centre)
    fl = rigcheck.motion(fr["white"], white)
    pt = rigcheck.motion(rigcheck.normalised_pattern(fr["pattern"], fr["white"], fr["black"]),
                         rigcheck.normalised_pattern(pattern, white, black))
    dist = _working_distance(s.store, preset.geometry_id, cal)
    mm_per_px = (dist / float(cal.K_cam[0][0])) if (dist and cal is not None) else None
    verdict = rigcheck.assess(fl, pt, tol_px, mm_per_px, dist)
    warnings = []
    bound = ref.get("calibration_id") == out["calibration_id"]
    if not bound:
        warnings.append(f"the reference was taken with calibration {ref.get('calibration_id')}, "
                        f"the current one is {out['calibration_id']}: set a new reference "
                        "(set_reference) once the rig is confirmed good")
    if mm_per_px is None:
        warnings.append("shift in camera px only: mm need a stereo calibration and a metric "
                        "scan at this geometry (the working distance)")
    if verdict.get("status") == "OK" and tol_px > rigcheck.DEFAULT_TOL_PX:
        warnings.append(f"OK at a tolerance of {tol_px:g} px, looser than the default "
                        f"{rigcheck.DEFAULT_TOL_PX:g} px: this check does not verify the rig "
                        "for a mm result (a mobile rig stays unverified, an earlier MOVED "
                        "stands)")
    res = {**out, **verdict, "reference_id": ref["path"].name, "flood": fl, "pattern": pt,
           "working_distance_mm": round(dist, 1) if dist else None,
           "mm_per_px": round(mm_per_px, 5) if mm_per_px else None, "warnings": warnings,
           "reference_calibration_id": ref.get("calibration_id"),
           "reference_trust": ref.get("rig_trust"), "rehome": None}
    if not rehome and not (_camera_like(verdict) and cal is not None and bound):
        return res
    if not _camera_like(verdict):                       # rehome asked, nothing to re-home
        why = ("the rig is in place: nothing to re-home" if verdict["status"] == "OK" else
               f"the check is {verdict['status']}"
               + (f" ({verdict['moved']} moved)" if verdict.get("moved") else "")
               + ": only a small CAMERA move (the flood and the pattern moving together) "
                 "can be re-homed - recalibrate")
        res["rehome"] = {"applicable": False, "done": False, "why": why}
        if verdict["status"] != "OK":
            res["message"] += f" | re-home REFUSED: {why}"
        return res
    if cal is None or not bound:        # the library changed while the frames were captured
        why = ("the calibration or its reference changed during the check: nothing to "
               "re-home against - run the check again")
        res["rehome"] = {"applicable": False, "done": False, "why": why}
        res["message"] += f" | re-home REFUSED: {why}"
        return res
    fit = _rehome_fit(ref, white, pattern, black, cal)
    res["rehome"] = {**_rehome_public(fit), "done": False, "from": cal.id}
    if not fit["applicable"]:
        if rehome:
            res["message"] += f" | re-home REFUSED: {fit['why']}"
        return res
    if not rehome:                                      # say that it could be re-homed
        res["message"] += (f" | a camera rotation of {fit['rotation_deg']:.4f} deg explains "
                           f"it (residual {fit['residual_px']:.2f} px): `{REHOME_HOWTO}` "
                           f"writes a re-homed calibration ({REHOME_CAVEAT})")
        return res
    created = datetime.now().isoformat(timespec="seconds")
    new = stereo.rehome(s.store, cal, fit["R"], created=created, details={
        "reference_id": ref["path"].name, "motion_px": fit["motion_px"],
        "shift_px": [round(v, 4) for v in fit["shift_px"]],
        "pan_deg": round(fit["pan_deg"], 5), "tilt_deg": round(fit["tilt_deg"], 5),
        "roll_deg": round(fit["roll_deg"], 5), "scale_pct": round(fit["scale_pct"], 4),
        "residual_px": round(fit["residual_px"], 4), "match": fit["match"],
        "pattern_rms_px": fit["pattern_rms_px"], "tiles_agree": fit["tiles_agree"],
        "tiles_used": fit["tiles_used"], "mock": s.mock})
    out2 = {**out, "calibration_id": new.id}
    # the re-homed calibration's reference is trusted as far as the one it was fitted against
    d = _store_reference(s, preset, out2, fp, white, pattern, black, label, "rehome",
                         ref.get("rig_trust"))
    res.update(out2)
    res.update(status="REHOMED", moved="camera", path=new.path, reference_id=d.name,
               previous_reference_id=ref["path"].name, reference_reason="rehome",
               rehome_from=cal.id,
               message=(f"camera re-homed: {stereo.REHOME_NOTE.format(id=cal.id)} -> "
                        f"{new.id} (rotation {fit['rotation_deg']:.4f} deg: pan "
                        f"{fit['pan_deg']:+.4f}, tilt {fit['tilt_deg']:+.4f}, roll "
                        f"{fit['roll_deg']:+.4f} deg; image motion {fit['motion_px']:.2f} px, "
                        f"residual {fit['residual_px']:.2f} px); new calibration -> new "
                        f"reference stored ({d.name})"))
    res["rehome"].update(done=True, calibration_id=new.id, note=new.rehomed["note"],
                         generation=new.rehomed["generation"])
    res["warnings"] = [w for w in warnings if "px only" in w] + [
        REHOME_CAVEAT,
        f"goldens, null calibrations and face targets recorded with {cal.id} are bound to "
        "that calibration: record them again with the re-homed one (as after a "
        "recalibration)",
        f"the original calibration {cal.id} is unchanged: `scan delete {new.id}` (to the "
        "library trash) makes it the one in use again"]
    if new.rehomed["generation"] > 1:
        res["warnings"].append(f"this is re-home number {new.rehomed['generation']} since the "
                               f"full calibration {new.rehomed['root']}: the errors add up - "
                               "recalibrate soon")
    return res

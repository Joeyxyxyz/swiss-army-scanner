"""
Auto exposure per end face: the camera exposure a bare-metal face needs depends on how it
is seated.

Bench (2026-10-12, one preset, the same part): 33 / 50 / 66.7 / 83 ms depending on the
seating - at 33 ms a face tilted 6-10 deg from the mirror direction was too dark (1 % of it
decoded), at 83 ms a face that mirrors the projector was saturated white, and 2 % clipped
pixels were enough for a false rim trigger. A fixed preset exposure cannot be right for
both, and the duty-based auto_intensity only turns the light DOWN (and refuses inverted
presets: there the duty scales the dark level).

  what is measured   the face's FLOOD (the brighter of the 255 / 0 patterns, measured, never
                     assumed - an inverted preset lights the face with pattern 0), lit only
                     through the face's DMD mask: p95 of the camera DN inside 0.8 x the face
                     radius, and the share of the located face (inside 0.96 x its radius: the
                     mask's border pixels see the background) at / above 254 DN
  what is chosen     a FRAME-LOCKED exposure: a whole number of 60 Hz video frames (16.667 ms;
                     anything else catches a different slice of the DMD's bit-plane sequence
                     every frame) inside the range (default 16.7-100 ms), with the p95 nearest
                     the target (default 170 of 255 DN) among those whose p95 lies inside
                     [target / 1.8, the smaller of 1.4 x target and 240] with less than 0.5 %
                     of the face clipped
  how                a per-capture CAMERA exposure override (camera.set_exposure_us after the
                     rig opens): the preset is not edited, the LED currents are never touched
  when it fails      no exposure satisfies both (a dim face that glints: it clips before it
                     is bright enough; a face that mirrors the projector: saturated at the
                     shortest exposure; a face too dark at the longest): ExposureNotFound with
                     every shot and the reason - the face is not scanned (PULL: re-seat it)

choose() is the search (a few flood shots: start at the preset's exposure, jump to the one
the shot predicts, step shorter while it clips); verbs.py drives the rig with it.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np

from ...config import VIDEO_FRAME_US, frame_locked

RANGE_US = (16667, 100000)          # 1 .. 6 video frames
TARGET_DN = 170.0
MAX_CLIPPED_PCT = 0.5
CLIP_DN = 254.0                     # camera DN at / above: clipped (averaged frames)
INNER_R = 0.8                       # the p95 is taken inside this x the face radius
FACE_R = 0.96                       # clipping is counted inside this x the face radius (the
                                    # located mask's border pixels are partly background)
BAND = (1.0 / 1.8, 1.4)             # accepted p95, x the target ...
MAX_DN = 240.0                      # ... and never above this


class ExposureNotFound(ValueError):
    """No frame-locked exposure in the range gives the face a usable brightness without
    clipping. report: the search (every shot)."""

    def __init__(self, message: str, report: Optional[dict] = None):
        super().__init__(message)
        self.report = report or {}


def candidates(range_us=RANGE_US) -> list:
    """The frame-locked exposures (whole us) inside range_us [lo, hi] (inclusive, +-0.5 %)."""
    try:
        lo, hi = (float(v) for v in range_us)
    except (TypeError, ValueError):
        raise ValueError(f"exposure range must be [lo, hi] us, got {range_us!r}") from None
    if not (math.isfinite(lo) and math.isfinite(hi)) or lo <= 0 or hi < lo:
        raise ValueError(f"exposure range must be [lo, hi] us with 0 < lo <= hi, got {range_us!r}")
    n0 = max(1, int(math.ceil(lo * (1 - 0.005) / VIDEO_FRAME_US - 1e-9)))
    out = []
    n = n0
    while n * VIDEO_FRAME_US <= hi * (1 + 0.005):
        out.append(int(round(n * VIDEO_FRAME_US)))
        n += 1
    if not out:
        raise ValueError(f"no whole number of {VIDEO_FRAME_US:.0f} us video frames inside the "
                         f"exposure range {lo:g}..{hi:g} us")
    return out


def band(target_dn: float) -> tuple:
    """(min, max) p95 DN accepted for a target."""
    return float(target_dn) * BAND[0], min(float(target_dn) * BAND[1], MAX_DN)


def face_stats(frame: np.ndarray, inner: np.ndarray, face: np.ndarray) -> dict:
    """One flood frame -> the face's p95 / p50 (inside `inner`: 0.8 x the radius) and the
    clipped share of `face` (the located face inside 0.96 x its radius)."""
    f = np.asarray(frame, np.float64)
    v = f[inner]
    w = f[face]
    if not v.size or not w.size:
        raise ValueError("the face mask is empty on this camera frame")
    return {"p95_dn": round(float(np.percentile(v, 95)), 2),
            "p50_dn": round(float(np.median(v)), 2),
            "clipped_pct": round(100.0 * float(np.mean(w >= CLIP_DN)), 3),
            "n_face_px": int(w.size)}


def judge(st: dict, target_dn: float, max_clipped_pct: float) -> str:
    """'' when the shot is usable, else why not."""
    lo, hi = band(target_dn)
    why = []
    if st["clipped_pct"] >= max_clipped_pct:
        why.append(f"{st['clipped_pct']:.2f} % of the face clipped (limit {max_clipped_pct:g} %)")
    if st["p95_dn"] < lo:
        why.append(f"too dark (p95 {st['p95_dn']:.0f} DN < {lo:.0f})")
    elif st["p95_dn"] > hi:
        why.append(f"too bright (p95 {st['p95_dn']:.0f} DN > {hi:.0f})")
    return "; ".join(why)


def choose(measure: Callable, cands: list, start_us: float, target_dn: float = TARGET_DN,
           max_clipped_pct: float = MAX_CLIPPED_PCT) -> dict:
    """measure(exposure_us) -> face_stats of a flood shot at that exposure. -> {ok,
    exposure_us, p95_dn, clipped_pct, shots [{exposure_us, p95_dn, p50_dn, clipped_pct, ok,
    why}], why}: the usable shot nearest the target, searched from the candidate nearest
    start_us (never more shots than candidates)."""
    cands = sorted(int(c) for c in cands)
    lo, hi = band(target_dn)
    shots = {}
    e = min(cands, key=lambda c: abs(math.log(c / float(start_us))))
    while e is not None and e not in shots:
        st = dict(measure(e))
        st["why"] = judge(st, target_dn, max_clipped_pct)
        st["ok"] = not st["why"]
        shots[e] = st
        nxt = None
        if st["clipped_pct"] >= max_clipped_pct or st["p95_dn"] >= CLIP_DN - 4:
            shorter = [c for c in cands if c < e and c not in shots]     # it only gets worse
            nxt = max(shorter) if shorter else None                     # with more light
        else:
            pred = {c: st["p95_dn"] * c / e for c in cands}             # linear in exposure
            fit = [c for c in cands if pred[c] <= hi] or [cands[0]]
            best = min(fit, key=lambda c: abs(math.log(max(pred[c], 1e-6) / target_dn)))
            if best not in shots:
                nxt = best
        e = nxt
    rows = [{"exposure_us": k, **{x: v[x] for x in ("p95_dn", "p50_dn", "clipped_pct", "ok",
                                                    "why")}} for k, v in sorted(shots.items())]
    good = [k for k, v in shots.items() if v["ok"]]
    out = {"ok": bool(good), "exposure_us": None, "p95_dn": None, "clipped_pct": None,
           "shots": rows, "why": "", "target_dn": float(target_dn),
           "p95_band_dn": [round(lo, 1), round(hi, 1)],
           "max_clipped_pct": float(max_clipped_pct), "candidates_us": cands}
    if good:
        k = min(good, key=lambda c: abs(math.log(shots[c]["p95_dn"] / target_dn)))
        out.update(exposure_us=k, p95_dn=shots[k]["p95_dn"], clipped_pct=shots[k]["clipped_pct"])
        return out
    out["why"] = explain(rows, cands, target_dn, max_clipped_pct)
    return out


def explain(rows: list, cands: list, target_dn: float, max_clipped_pct: float) -> str:
    """Why no exposure fits, in the operator's words."""
    lo, hi = band(target_dn)
    shortest, longest = rows[0], rows[-1]
    clip = [r for r in rows if r["clipped_pct"] >= max_clipped_pct]
    if clip and shortest["exposure_us"] == cands[0] and shortest["clipped_pct"] >= max_clipped_pct:
        if shortest["p95_dn"] >= lo:
            return (f"saturated at the shortest exposure ({shortest['exposure_us'] / 1000:.1f} ms: "
                    f"{shortest['clipped_pct']:.1f} % of the face clipped, p95 "
                    f"{shortest['p95_dn']:.0f} DN): the face mirrors the projector into the "
                    "camera - turn / re-seat the part or lower the light (duty)")
        return (f"the face glints before it is bright enough ({shortest['exposure_us'] / 1000:.1f}"
                f" ms: {shortest['clipped_pct']:.1f} % clipped while p95 is only "
                f"{shortest['p95_dn']:.0f} DN): a specular patch mirrors the projector - turn / "
                "re-seat the part")
    if clip:
        ok_dark = [r for r in rows if r["clipped_pct"] < max_clipped_pct]
        d = max(ok_dark, key=lambda r: r["exposure_us"]) if ok_dark else shortest
        c = min(clip, key=lambda r: r["exposure_us"])
        return (f"too dark without clipping ({d['exposure_us'] / 1000:.1f} ms: p95 "
                f"{d['p95_dn']:.0f} DN) and clipped when bright enough "
                f"({c['exposure_us'] / 1000:.1f} ms: {c['clipped_pct']:.1f} % clipped): a "
                "specular patch mirrors the projector - turn / re-seat the part")
    if longest["exposure_us"] == cands[-1] and longest["p95_dn"] < lo:
        return (f"too dark at the longest exposure ({longest['exposure_us'] / 1000:.1f} ms: p95 "
                f"{longest['p95_dn']:.0f} DN < {lo:.0f}): raise the light (duty / a brighter "
                "preset) or widen the exposure range")
    return "; ".join(f"{r['exposure_us'] / 1000:.1f} ms: {r['why']}" for r in rows)


def allowed(preset) -> tuple:
    """(auto exposure is the default for this preset, why not). It is when the preset's own
    exposure is a whole number of video frames: the preset is set up for HDMI video timing,
    and a frame-locked exposure near it is the same kind of capture. A preset whose
    exposure is not (a sub-frame exposure, a mock fixture) is left as it is unless auto
    exposure is asked for."""
    e = preset.camera.exposure_us
    if frame_locked(e):
        return True, ""
    return False, (f"the preset's exposure ({e:g} us) is not a whole number of "
                   f"{VIDEO_FRAME_US:.0f} us video frames")


def check_forced(exposure_us) -> float:
    if isinstance(exposure_us, bool) or not isinstance(exposure_us, (int, float)) \
            or not math.isfinite(exposure_us) or not exposure_us > 0:
        raise ValueError(f"exposure_us must be > 0 us, got {exposure_us!r}")
    return float(exposure_us)

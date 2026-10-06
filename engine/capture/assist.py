"""
Scan assistant — helps pick good parameters before scanning.

tune(): sweeps camera exposure (and, only when asked, DMD duty), running
the check_signal probe at each step, and recommends the setting with the
best fringe modulation (gc_conf) and valid % while keeping saturation near
0 (saturation quantises the fringe phase — bench finding 2026-09-22). It
also reports the coded-axis check (fringes must run perpendicular to the
baseline), focus ratio and plain-language advice. It never changes the
preset itself: `recommended` is a preset patch the operator (or UI) applies.

measure_gamma(): projects a gray staircase, fits the projector's response
(HDMI video mode applies degamma — DLPU010G §2.4) against the value
actually sent over HDMI, and returns that gamma (= the projection.gamma
precomp to use) plus the residual nonlinearity of the preset's own chain.

Both run the rig once, through the normal projection chain, and put a JSON
report in the output bin (output/tuning/). Polarity is always MEASURED
(the bench HDMI path lands patterns inverted with projection.invert).

Exposure vs the video frame: in HDMI video mode the DLPC350 builds every
60 Hz frame from sequential R/G/B LED segments and PWM bit-planes, and the
camera free-runs on its own clock. An exposure that is not a whole number
of video frames (VIDEO_FRAME_US) catches a different slice of that
sequence each frame, so gray levels (every phase-shift fringe) and
single-colour floods flicker frame to frame; averaging only partly hides
it. tune therefore sweeps whole-frame exposures by default, prefers them,
and says so when it has to fall back to a sub-frame one.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Optional

import numpy as np

from .. import _seed
from ..bases.hybrid import SAT_LEVEL
# the video frame and the lock test live in config (the preset's exposure
# warning uses them too); tune(frame_us=) overrides the frame period
from ..config import FRAME_LOCK_TOL, VIDEO_FRAME_US, frame_locked
from . import signal

DEFAULT_FRAME_MULTIPLES = (1, 2, 3)          # whole-frame exposures tune sweeps by default
SUBFRAME_FALLBACK_US = (4000, 8000)          # for scenes where a whole frame clips


def frame_exposures(multiples=DEFAULT_FRAME_MULTIPLES, frame_us: float = VIDEO_FRAME_US) -> tuple:
    """Whole-video-frame exposures in us, e.g. (16667, 33333, 50000) at 60 Hz."""
    return tuple(int(round(n * frame_us)) for n in multiples)


def snap_exposure_us(exposure_us, frame_us: float = VIDEO_FRAME_US) -> int:
    """Nearest whole-frame exposure (at least one frame)."""
    return int(round(max(1, round(float(exposure_us) / frame_us)) * frame_us))


# us; the preset's own is always added. Whole frames first-class, two sub-frame fallbacks.
DEFAULT_EXPOSURES = tuple(sorted(set(SUBFRAME_FALLBACK_US) | set(frame_exposures())))
MAX_EXPOSURE_US = 1_000_000      # the bench camera grab times out at 2 s
# SAT_LEVEL (counts treated as clipped in an averaged frame) is the scan
# decoder's own (bases/hybrid.py), so tune and a scan agree on "clipped"
SAT_MAX_PCT = 0.1                # % of lit px allowed to clip at the recommended step
                                 # (= hybrid.SAT_WARN_PCT, where a scan starts warning)
VALID_KEEP = 0.9                 # recommended step keeps >= 90 % of the best valid %
LOW_VALID_PCT = 20.0             # bench reference scan decoded ~38 %
HEADROOM_P99 = 180.0             # brightest lit px below this = exposure could go longer
GAMMA_FIT_MIN = 0.1              # fit levels above ~10 % (below that, log(noise) dominates)
GAMMA_LINEAR_TOL = 0.05          # |gamma - 1| below this: the path is linear
GAMMA_NARROW_SPAN = 0.7          # fitted levels all within 70-100 % of the top: weak fit
USABLE = ("usable", "good", "excellent")

TUNE_UNITS = {"exposure_us": "us", "gc_conf": "camera counts (8-bit DN, fringe modulation)",
              "valid_pct": "% of camera px", "saturated_pct": "% of lit camera px",
              "focus_ratio": "ratio (modulation / half contrast; 1 = sharp)",
              "duty": "fraction of full DMD drive", "bright_p99": "camera counts (8-bit DN)",
              "contrast_mean": "camera counts (8-bit DN)",
              "frame_locked": "bool (exposure is a whole number of HDMI video frames)",
              "video_frame_us": "us"}
GAMMA_UNITS = {"gamma": "dimensionless (power-law exponent vs the value sent over HDMI)",
               "residual_gamma": "dimensionless (power-law exponent vs the preset chain's "
                                 "value before the precomp; 1 = linear)",
               "levels": "value sent over HDMI 0-255 (precomp included), brightness-ordered",
               "sent_levels": "value sent over HDMI 0-255 (precomp included)",
               "chain_levels": "preset chain value 0-255 before the gamma precomp",
               "drive_levels": "pattern value sent through the preset chain, 0-255",
               "response": "fraction of full-scale camera response above mirrors-off",
               "response_counts": "camera counts (8-bit DN) above the mirrors-off frame",
               "r2": "dimensionless (log-log fit)"}


@dataclass
class TuneStep:
    exposure_us: int
    gc_conf: float
    valid_pct: float
    saturated_pct: float
    focus_ratio: float
    coded_axis: str
    duty: Optional[float] = None       # None = the preset's duty (no duty sweep)
    quality: str = ""                  # signal.quality_band(gc_conf)
    axis_ok: Optional[bool] = None     # coded axis matches rig.projector_side?
    polarity_inverted: Optional[bool] = None   # measured: black pattern lands brighter
    bright_p99: float = 0.0            # 99th pct of the brighter flood over lit/clipped px
    contrast_mean: float = 0.0
    frame_locked: Optional[bool] = None  # exposure = whole number of HDMI video frames?


@dataclass
class TuneReport:
    steps: list                 # [TuneStep]
    recommended: dict           # preset patch, e.g. {"camera": {"exposure_us": 8000}}
    advice: list                # plain-language findings / next actions
    ok: bool
    path: Optional[Path] = None
    units: dict = field(default_factory=lambda: dict(TUNE_UNITS))
    geometry_id: str = ""
    video_frame_us: float = VIDEO_FRAME_US


@dataclass
class GammaReport:
    gamma: float                # projector's own response vs the value sent (= precomp to use)
    levels: list                # brightness-ordered sent levels (see measure_gamma)
    response: list              # normalised camera response per level, above mirrors-off
    r2: float
    recommended: dict           # {"projection": {"gamma": ...}}, {} if no precomp can help
    path: Optional[Path] = None
    warnings: list = field(default_factory=list)
    drive_levels: list = field(default_factory=list)   # pattern values put into the chain
    polarity_inverted: Optional[bool] = None           # the preset chain (measured)
    units: dict = field(default_factory=lambda: dict(GAMMA_UNITS))
    residual_gamma: Optional[float] = None   # the preset chain as driven (1 = linear)
    sent_levels: list = field(default_factory=list)    # value sent over HDMI per drive level
    hdmi_inverted: Optional[bool] = None     # raw black lands brighter than raw white


# ------------------------------------------------------------------ tune

class _Tap:
    """Stands in for the rig's sync so signal.probe runs unchanged — tune
    reports exactly what check_signal would at each setting — while keeping
    the flood frames the saturation check needs."""

    def __init__(self, sync):
        self._sync = sync
        self.floods = {}                   # drive level -> frame

    def shoot(self, pattern, raw: bool = False):
        frame = self._sync.shoot(pattern, raw=raw)
        p = np.asarray(pattern)
        if not raw and p.ndim == 2 and p.size and p.min() == p.max():
            self.floods[int(p.flat[0])] = frame
        return frame


def _polarity(f_on: np.ndarray, f_off: np.ndarray):
    """(bright, dark, inverted) — which flood is physically bright is
    measured, never assumed: with projection.invert the bench shows flood(0)
    as full light."""
    inverted = float(f_off.mean()) > float(f_on.mean())
    return (f_off, f_on, True) if inverted else (f_on, f_off, False)


def _probe_step(sync, preset, exposure_us: int, duty: Optional[float],
                frame_us: float = VIDEO_FRAME_US) -> TuneStep:
    tap = _Tap(sync)
    rep = signal.probe(tap, preset)
    if 0 not in tap.floods or 255 not in tap.floods:
        raise RuntimeError("signal.probe no longer shoots white/black floods; "
                           "assist.tune needs them for the saturation check")
    bright, dark, inverted = _polarity(tap.floods[255], tap.floods[0])
    if abs(float(bright.mean()) - float(dark.mean())) < 0.5:
        inverted = None                    # both floods clipped (or dark): can't tell
    lit = (bright - dark) > preset.decode.contrast_min
    sat = bright >= SAT_LEVEL
    # A pixel clipped in BOTH floods has zero measured contrast and drops out
    # of `lit` — exactly when saturation is worst — so clipped pixels always
    # count, whether or not they still look lit.
    region = lit | sat
    sat_pct = 100.0 * float(sat.sum()) / float(region.sum()) if region.any() else 0.0
    p99 = float(np.percentile(bright[region], 99)) if region.any() else 0.0
    return TuneStep(exposure_us=int(exposure_us), gc_conf=rep.gc_conf, valid_pct=rep.valid_pct,
                    saturated_pct=round(sat_pct, 3), focus_ratio=rep.focus_ratio,
                    coded_axis=rep.coded_axis, duty=duty, quality=rep.quality,
                    axis_ok=rep.axis_ok, polarity_inverted=inverted, bright_p99=round(p99, 1),
                    contrast_mean=rep.contrast_mean,
                    frame_locked=frame_locked(exposure_us, frame_us))


def _frame_us(frame_us) -> float:
    if frame_us is None:
        return VIDEO_FRAME_US
    if isinstance(frame_us, bool) or not isinstance(frame_us, (int, float)) \
            or not 1000.0 <= frame_us <= 100_000.0:
        raise ValueError(f"frame_us must be the video frame period in us (1000..100000), "
                         f"got {frame_us!r}")
    return float(frame_us)


def _exposure_list(preset, exposures, frame_us: float = VIDEO_FRAME_US) -> list:
    if exposures is None:
        exposures = (DEFAULT_EXPOSURES if frame_us == VIDEO_FRAME_US else
                     tuple(SUBFRAME_FALLBACK_US) + frame_exposures(frame_us=frame_us))
    elif isinstance(exposures, (int, float)):
        exposures = [exposures]
    out = set()
    for e in list(exposures) + [preset.camera.exposure_us]:
        if isinstance(e, bool) or not isinstance(e, (int, float)) or not 0 < e <= MAX_EXPOSURE_US:
            raise ValueError(f"exposures must be numbers in (0, {MAX_EXPOSURE_US}] us, got {e!r}")
        out.add(int(round(e)))
    return sorted(out)


def _duty_list(preset, duties) -> list:
    """[None] without a sweep; else the preset's duty first, then the variants."""
    if duties is None:
        return [None]
    if isinstance(duties, (int, float)):
        duties = [duties]
    out = [float(preset.projection.duty)]
    for d in duties:
        if isinstance(d, bool) or not isinstance(d, (int, float)) or not 0 < d <= 1:
            raise ValueError(f"duties must be in (0, 1] (projection.duty), got {d!r}")
        if round(float(d), 4) not in [round(x, 4) for x in out]:
            out.append(float(d))
    return out


def _pool(steps: list) -> list:
    """Unsaturated steps that keep >= 90 % of the best unsaturated valid %.
    The valid-% floor stops a long exposure that only brightens a smaller
    patch from winning; the pool is the unsaturated steps (a clipped step
    can't be recommended, so it can't set the bar)."""
    clean = [s for s in steps if s.saturated_pct < SAT_MAX_PCT and s.valid_pct > 0]
    if not clean:
        return []
    best_valid = max(s.valid_pct for s in clean)
    return [s for s in clean if s.valid_pct >= VALID_KEEP * best_valid]


def _best(steps: list) -> TuneStep:
    return max(steps, key=lambda s: (s.gc_conf, -s.exposure_us))


def _choose(steps: list) -> Optional[TuneStep]:
    """Highest gc_conf in the pool (see _pool), taken from the frame-locked
    steps whenever one of them is at least 'usable': a sub-frame exposure
    samples a random slice of the video frame's LED/bit-plane sequence, so its
    gray levels flicker frame to frame (module docstring) - more modulation
    doesn't pay for that. A locked step too weak to scan never wins over a
    usable sub-frame one (the advice then says how to get a locked one)."""
    pool = _pool(steps)
    if not pool:
        return None
    locked = [s for s in pool if s.frame_locked and s.quality in USABLE]
    return _best(locked or pool)


def _bands_text() -> str:
    b = sorted(signal.QUALITY_BANDS)            # ascending floors
    parts = [f"<{b[1][0]:g} {b[0][1]}"]
    for (lo, name), (hi, _) in zip(b[1:], b[2:]):
        parts.append(f"{lo:g}-{hi:g} {name}")
    parts.append(f">={b[-1][0]:g} {b[-1][1]}")
    return ", ".join(parts)


def _frame_advice(preset, steps: list, rec: TuneStep, frame_us: float) -> list:
    """Why a sub-frame recommendation is second best, and how to get a
    whole-frame one; or, when a whole-frame step won over a brighter
    sub-frame one, why."""
    adv = []
    frame_txt = f"{frame_us:.0f} us, 60 Hz" if frame_us == VIDEO_FRAME_US else f"{frame_us:.0f} us"
    why = ("the DLPC350 builds each HDMI video frame from sequential R/G/B LED segments and "
           "PWM bit-planes, and the free-running camera catches a different slice of that "
           "sequence every frame, so gray levels (every phase-shift fringe) and colour floods "
           "flicker frame to frame - phase noise in scans, unstable band ratios in spectral ID")
    if rec.frame_locked is False:
        n = max(1, int(rec.exposure_us // frame_us))
        sug = int(round(n * frame_us))
        line = (f"{rec.exposure_us} us is not a whole number of video frames ({frame_txt}): "
                f"{why}. camera.average {preset.camera.average} only partly evens it out.")
        at = [s for s in steps if s.exposure_us == sug and s.duty == rec.duty]
        if at and at[0].saturated_pct >= SAT_MAX_PCT:
            line += (f" {sug} us (whole frames) clips {at[0].saturated_pct:.1f}% here: cut the "
                     "light reaching the camera (ND filter, stop the lens down, or lower the LED "
                     "current - a human step in the TI GUI) and re-run tune so it fits.")
        elif at and at[0].valid_pct > 0 and at[0].quality not in USABLE:
            up = int(round((n + 1) * frame_us))
            line += (f" {sug} us (whole frames) is too weak here (gc_conf {at[0].gc_conf:g}, "
                     f"'{at[0].quality}'): try {up} us (one frame more) if it does not clip, or "
                     "bring more light onto the target.")
        elif at:
            line += (f" {sug} us (whole frames) decodes less of the frame here "
                     f"({at[0].valid_pct:g}% valid): check aim/focus at that exposure.")
        else:
            line += f" Try {sug} us (whole frames): re-run tune with exposures=[{sug}]."
        line += " If you must stay off a whole frame, keep camera.average >= 8."
        adv.append(line)
    elif rec.frame_locked:
        brighter = [s for s in _pool(steps) if not s.frame_locked and s.gc_conf > rec.gc_conf]
        if brighter:
            b = _best(brighter)
            adv.append(f"Chose {rec.exposure_us} us (whole video frames) over {b.exposure_us} us "
                       f"(gc_conf {b.gc_conf:g} vs {rec.gc_conf:g}): at an exposure that is "
                       f"not a whole number of frames {why}.")
    return adv


def _advise(preset, steps: list, rec: Optional[TuneStep], swept_duty: bool,
            frame_us: float = VIDEO_FRAME_US) -> list:
    adv = []
    cam_exp = int(preset.camera.exposure_us)
    lit_any = any(s.valid_pct > 0 or s.saturated_pct > 0
                  or s.contrast_mean > preset.decode.contrast_min for s in steps)
    if not lit_any:
        return ["No lit pixels at any setting - is the projector on, showing the HDMI "
                "window, and aimed at the scene? Nothing else can be judged until it is."]

    # -- recommendation
    if rec is None:
        if any(s.saturated_pct < SAT_MAX_PCT for s in steps):
            adv.append("No unsaturated step decodes a single pixel (fringe modulation below "
                       "decode.conf_min): refocus, aim the projector at the target, or cut "
                       "ambient light, then re-run tune.")
        else:
            # with projection.invert, drive 0 always sends full light whatever
            # the duty, so a lower duty can't fix saturation there
            duty_hint = not swept_duty and not preset.projection.invert
            one_frame = int(round(frame_us))
            adv.append(f"Every step saturates (>= {SAT_MAX_PCT:g}% of lit px clipped). Best fix: "
                       "cut the light reaching the camera (ND filter, stop the lens down, or lower "
                       "the LED current - a human step in the TI GUI"
                       + (", or sweep a lower projection.duty" if duty_hint else "")
                       + f") so a whole-video-frame exposure ({one_frame} us) stays unclipped; "
                       f"shorter exposures (e.g. {max(min(s.exposure_us for s in steps) // 2, 1)}"
                       " us) also stop the clipping, but a sub-frame exposure samples the video "
                       "frame's LED/bit-plane sequence unevenly (gray levels flicker frame to "
                       "frame): keep camera.average >= 8 if you go that way.")
    else:
        what = f"exposure {rec.exposure_us} us" + (f" at duty {rec.duty:g}" if swept_duty else "")
        adv.append(f"Recommended {what}: gc_conf {rec.gc_conf:g} ({rec.quality}), valid "
                   f"{rec.valid_pct:g}%, saturation {rec.saturated_pct:g}% of lit px. Apply it "
                   "as a preset patch - tune never edits the preset.")
        if rec.exposure_us == cam_exp and (not swept_duty or rec.duty == preset.projection.duty):
            adv.append(f"The preset's own setting ({cam_exp} us) is already the best of the sweep.")
        elif rec.exposure_us != cam_exp:
            adv.append(f"Change camera.exposure_us {cam_exp} -> {rec.exposure_us}.")
        if swept_duty and rec.duty != preset.projection.duty:
            adv.append(f"Change projection.duty {preset.projection.duty:g} -> {rec.duty:g}.")
        adv.extend(_frame_advice(preset, steps, rec, frame_us))

    # -- saturation
    clipped = [s for s in steps if s.saturated_pct >= SAT_MAX_PCT]
    if clipped:
        lst = ", ".join(f"{s.exposure_us} us" + (f"/duty {s.duty:g}" if swept_duty else "")
                        + f" ({s.saturated_pct:.1f}%)" for s in clipped)
        adv.append(f"Saturation (>= {SAT_MAX_PCT:g}% of lit px at >= {SAT_LEVEL:g} counts) at "
                   f"{lst}: clipped fringe peaks flatten the sinusoid and quantise the phase "
                   "(bench finding 2026-09-22) - never scan at these settings.")
        own = [s for s in clipped if s.exposure_us == cam_exp
               and (s.duty is None or s.duty == preset.projection.duty)]
        if own:
            adv.append(f"The preset's own exposure ({cam_exp} us) saturates "
                       f"{own[0].saturated_pct:.1f}% of lit px - apply the recommended patch "
                       "before scanning.")
    if rec is not None:
        more = [s for s in clipped if s.valid_pct * VALID_KEEP > rec.valid_pct]
        if more:
            m = max(more, key=lambda s: s.valid_pct)
            adv.append(f"{m.exposure_us} us decodes more of the frame ({m.valid_pct:g}% vs "
                       f"{rec.valid_pct:g}%) but clips {m.saturated_pct:.1f}%: the scene mixes "
                       "bright and dark surfaces. Scan the dark part separately, or dull the "
                       "shiny one (matte spray / angle it off the specular direction).")
        longest = max(s.exposure_us for s in steps)
        if rec.exposure_us == longest and rec.bright_p99 < HEADROOM_P99:
            adv.append(f"Even the longest exposure ({longest} us) leaves headroom (brightest lit "
                       f"px ~{rec.bright_p99:g} of 255): extend the sweep to longer exposures "
                       "for more fringe modulation.")

    # -- coded axis (the rotate90 mistake)
    ref = rec or max(steps, key=lambda s: s.valid_pct)
    bad_axis = next((s for s in ([ref] + steps) if s.axis_ok is False), None)
    if bad_axis is not None:
        need = preset.rig.coded_axis
        adv.append(f"Coded-axis mismatch: fringes vary along camera {bad_axis.coded_axis.upper()} "
                   f"but the projector is {preset.rig.projector_side} of the camera, so depth "
                   f"shifts run along {need.upper()} and sensitivity is near zero. Toggle "
                   "projection.rotate90 (or check rig.projector_side).")

    # -- focus
    if ref.valid_pct > 0 and ref.focus_ratio < 0.7:
        adv.append(f"Focus ratio {ref.focus_ratio:g}: {signal.focus_hint(ref.focus_ratio)} "
                   "(camera and projector both focus on the target plane; human step).")

    # -- coverage
    if ref.valid_pct < LOW_VALID_PCT:
        adv.append(f"Only {ref.valid_pct:g}% of the camera frame decodes (bench reference "
                   "~38%): aim the projector at the target and concentrate the lit patch on it "
                   "(move closer / zoom), and check nothing shadows the scene.")

    # -- polarity (informational)
    inv = [s.polarity_inverted for s in steps if s.polarity_inverted is not None]
    if inv and sum(inv) * 2 >= len(inv):
        adv.append("Polarity is inverted (the black pattern lands brighter than white"
                   + (", as projection.invert sends it" if preset.projection.invert else "")
                   + "): normal on this bench's HDMI path; decode measures polarity, "
                   "no action needed.")

    # -- gc_conf band
    q = ref.quality or signal.quality_band(ref.gc_conf)
    line = (f"gc_conf {ref.gc_conf:g} at {ref.exposure_us} us is '{q}' (bands: "
            f"{_bands_text()}; bench reference ~36 at 38% valid).")
    if q not in USABLE:
        line += (" Too weak to scan: raise exposure (if unsaturated), cut ambient light, "
                 "or bring the projector closer.")
    adv.append(line)
    return adv


def tune(session, preset, exposures=None, duties=None, frame_us=None) -> TuneReport:
    """Sweep exposure (x duty, only if `duties` given) with the check_signal
    probe; recommend the best unsaturated setting, preferring whole-video-
    frame exposures (frame_us, default VIDEO_FRAME_US). The default sweep is
    1-3 whole frames plus two sub-frame fallbacks. The rig opens once. The
    preset object and its file are never modified: exposure goes to the
    camera directly and duty to a private copy of the projector's cfg."""
    fus = _frame_us(frame_us)
    exps = _exposure_list(preset, exposures, fus)
    duty_list = _duty_list(preset, duties)
    swept = duties is not None
    steps = []
    with session.rig(preset) as sync:
        cam, proj = sync.camera, sync.projector
        if not hasattr(cam, "set_exposure_us"):
            raise RuntimeError(f"{type(cam).__name__} cannot change exposure; tune needs "
                               "set_exposure_us")
        cfg0 = proj.cfg
        try:
            for d in duty_list:
                # replace, never mutate: proj.cfg IS preset.projection
                proj.cfg = cfg0 if d is None else replace(cfg0, duty=d)
                for e in exps:
                    cam.set_exposure_us(e)
                    steps.append(_probe_step(sync, preset, e, d, fus))
        finally:
            proj.cfg = cfg0

    rec = _choose(steps)
    recommended = {}
    if rec is not None:
        recommended["camera"] = {"exposure_us": rec.exposure_us}
        if swept:
            recommended["projection"] = {"duty": rec.duty}
    ok = bool(rec is not None and rec.quality in USABLE and rec.valid_pct > 1.0
              and rec.axis_ok is not False)
    advice = _advise(preset, steps, rec, swept, fus)

    store = session.store
    label = f"tune_{preset.name}"
    out = store.output_path("tuning", label, ".json")
    report = TuneReport(steps=steps, recommended=recommended, advice=advice, ok=ok, path=out,
                        geometry_id=preset.geometry_id, video_frame_us=round(fus, 3))
    store.write_json(out, {
        "kind": "tune", "preset": preset.name, "geometry_id": preset.geometry_id,
        "mock": preset.mock, "ok": ok, "recommended": recommended,
        "recommended_step": asdict(rec) if rec else None, "advice": advice,
        "steps": [asdict(s) for s in steps],
        "preset_settings": {"camera": {"exposure_us": preset.camera.exposure_us,
                                       "average": preset.camera.average,
                                       "gain_db": preset.camera.gain_db},
                            "projection": {"duty": preset.projection.duty,
                                           "invert": preset.projection.invert,
                                           "rotate90": preset.projection.rotate90,
                                           "gamma": preset.projection.gamma}},
        "criteria": {"saturation_level_counts": SAT_LEVEL, "max_saturated_pct": SAT_MAX_PCT,
                     "min_valid_fraction_of_best": VALID_KEEP,
                     "rank": "highest gc_conf among unsaturated steps, whole-video-frame "
                             "exposures first when any qualifies",
                     "video_frame_us": round(fus, 3), "frame_lock_tolerance": FRAME_LOCK_TOL},
        "units": report.units,
        "note": "tune never edits the preset; apply `recommended` as a patch"})
    store.record_output("tuning", out, label=label, geometry_id=preset.geometry_id)
    return report


# ---------------------------------------------------------- measure_gamma

def _fit_power(x: np.ndarray, y: np.ndarray):
    """log y = gamma * log x + c, least squares -> (gamma, c, r2 in log space)."""
    lx, ly = np.log(x), np.log(y)
    A = np.stack([lx, np.ones_like(lx)], axis=1)
    (g, c), *_ = np.linalg.lstsq(A, ly, rcond=None)
    res = ly - A @ np.array([g, c])
    ss_tot = float(((ly - ly.mean()) ** 2).sum())
    r2 = 1.0 - float((res ** 2).sum()) / ss_tot if ss_tot > 0 else 0.0
    return float(g), float(c), r2


def _fit_mask(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Levels a power-law fit can use: above GAMMA_FIT_MIN with a positive
    response, needing >= 3 points over >= 2 distinct levels."""
    m = (x >= GAMMA_FIT_MIN) & (y > 0)
    if int(m.sum()) < 3 or np.unique(x[m]).size < 2:
        return None
    return m


def measure_gamma(session, preset, n_levels: int = 12) -> GammaReport:
    """Gray staircase through the preset's own projection chain
    (duty -> invert -> gamma precomp -> HDMI video mode -> LEDs -> camera),
    plus a raw black/white pair that finds the mirrors-off frame.

    Projector.prepare applies the precomp LAST, to the value actually sent,
    and the projector's response acts on that sent value, so the precomp
    that linearises the chain is the projector's own exponent whatever duty
    and invert are. Two fits share the same frames:
      * `gamma`: response vs the value sent over HDMI (sent/255, any precomp
        included) - the projector's own curve, and the recommended
        projection.gamma directly (it replaces the current precomp, it is
        not multiplied by it);
      * `residual_gamma`: response vs the chain value before the precomp -
        the WHOLE path as this preset drives it, what the phase decoders
        see (1 = linear; equals `gamma` while projection.gamma is 1).
    The dark reference is the physically dark (mirrors-off) raw frame, never
    the chain's darkest level: with invert and duty < 1 that level still
    sends 255 x (1 - duty) (bench: 115, mirrors ~45 % on), and subtracting
    it would fit a compressed, non-power-law curve. Polarity is measured on
    both pairs, never assumed. The camera must be linear (Blackfly gamma
    off, the default) or its curve is folded in.
    """
    if isinstance(n_levels, bool) or not isinstance(n_levels, int) or not 4 <= n_levels <= 256:
        raise ValueError(f"n_levels must be an int in 4..256, got {n_levels!r}")
    P = _seed.load("patterns")
    drive = sorted({int(round(v)) for v in np.linspace(0, 255, n_levels)})
    order = [0, 255] + [L for L in drive if L not in (0, 255)]   # polarity pair first
    frames, sent = {}, {}
    with session.rig(preset) as sync:
        proj = sync.projector
        # raw pair, sent exactly as given (no duty/invert/precomp): whichever
        # lands darker is the mirrors-off frame - measured, never assumed
        raw_white = sync.shoot(P.flood(255), raw=True)
        raw_black = sync.shoot(P.flood(0), raw=True)
        for L in order:
            pat = P.flood(L)
            frames[L] = sync.shoot(pat)
            sent[L] = int(proj.prepare(pat).flat[0])
        cfg0 = proj.cfg
        try:
            # replace, never mutate: proj.cfg IS preset.projection
            proj.cfg = replace(cfg0, gamma=1.0)
            chain = {L: int(proj.prepare(P.flood(L)).flat[0]) for L in drive}
        finally:
            proj.cfg = cfg0

    bright, _, inverted = _polarity(frames[255], frames[0])
    _, off, hdmi_inverted = _polarity(raw_white, raw_black)
    lit = (bright - off) > preset.decode.contrast_min
    sat = bright >= SAT_LEVEL
    use = lit & ~sat                      # clipped px would flatten the top of the curve
    if not use.any():
        raise RuntimeError("no lit, unclipped pixels: is the projector on and aimed at the "
                           "scene? If the lit patch is all clipped, lower camera.exposure_us "
                           "(run tune) first.")
    warnings = []
    sat_pct = 100.0 * float(sat.sum()) / float((lit | sat).sum())
    if sat_pct >= SAT_MAX_PCT:
        warnings.append(f"the brightest level clips {sat_pct:.1f}% of lit px (excluded from the "
                        "fit); lower camera.exposure_us (see tune) if that is much of the scene")

    # brightness-ordered physical level: what the mirrors are driven with
    def phys(v: int) -> int:
        return 255 - v if hdmi_inverted else v

    counts = {L: float(np.median((frames[L] - off)[use])) for L in drive}
    full = max(counts.values())
    if not full > 0:
        raise RuntimeError("no response above the mirrors-off frame at any level; check the "
                           "projector")
    by_lv = sorted(drive, key=lambda L: (phys(sent[L]), counts[L]))
    levels = [phys(sent[L]) for L in by_lv]
    response = [counts[L] / full for L in by_lv]

    y = np.array(response, np.float64)
    xs = np.array(levels, np.float64) / 255.0
    xc = np.array([phys(chain[L]) for L in by_lv], np.float64) / 255.0
    fit = _fit_mask(xs, y)
    if fit is None:
        n = int(((xs >= GAMMA_FIT_MIN) & (y > 0)).sum())
        raise ValueError(f"only {n} levels above {GAMMA_FIT_MIN:.0%} with a positive response "
                         f"(sent levels {min(levels)}-{max(levels)}); raise n_levels, check "
                         "exposure (run tune) or measure with a higher projection.duty")
    gamma, c, r2 = _fit_power(xs[fit], y[fit])
    fit_c = _fit_mask(xc, y)
    residual = c_res = r2_res = None
    if fit_c is not None:
        residual, c_res, r2_res = _fit_power(xc[fit_c], y[fit_c])
    else:
        warnings.append("the preset chain drives too few distinct levels above "
                        f"{GAMMA_FIT_MIN:.0%} to fit its residual gamma")

    drops = [i for i in range(1, len(response)) if response[i] < response[i - 1] - 0.02]
    if drops:
        warnings.append("response is not monotonic (camera frames may lag the pattern): check "
                        "projection.push_twice / min_hold_s and re-run")
    if r2 < 0.98:
        warnings.append(f"power-law fit is poor (r2 {r2:.3f}); inspect the response curve before "
                        "using the gamma")
    lo, hi = float(xs[fit].min()), float(xs[fit].max())
    pc = preset.projection
    if lo > GAMMA_NARROW_SPAN * hi:
        warnings.append(f"the preset chain only sends {round(lo * 255)}-{round(hi * 255)} of 255 "
                        f"(duty {pc.duty:g}, invert {pc.invert}): a gamma fitted over that narrow "
                        "span is noise-sensitive; for the projector's whole curve run "
                        "measure_gamma on a copy of the preset with projection.duty 1 (the precomp "
                        "acts on the value sent, so the result carries over)")

    g0 = float(pc.gamma)
    res_nonlinear = residual is not None and abs(residual - 1.0) > GAMMA_LINEAR_TOL
    if hdmi_inverted:
        recommended = {}
        warnings.append("the HDMI path itself inverts (a raw black frame lands brighter than raw "
                        "white): projection.gamma acts on the value sent, on the other side of "
                        "that inversion, so no precomp can linearise this path - none "
                        "recommended; check the projector's input/display settings")
    else:
        recommended = {"projection": {"gamma": round(gamma, 2)}}
        if g0 != 1.0 and residual is not None:
            warnings.append(f"the preset already precompensates gamma {g0:g}: the path as driven "
                            f"has residual gamma {residual:.3f} ("
                            + ("linear" if not res_nonlinear else "still nonlinear")
                            + f"). The recommended precomp is the projector's own gamma "
                            f"{gamma:.3f}, fitted against the value actually sent - it replaces "
                            f"{g0:g}, it does not multiply it")
        if pc.invert and res_nonlinear:
            warnings.append("projection.invert is on: Projector.prepare applies the gamma precomp "
                            "LAST, to the value actually sent, so the recommended precomp "
                            "linearises the inverted chain too - no need to measure or scan with "
                            "invert off")
    cam_exp = preset.camera.exposure_us
    cam_locked = frame_locked(cam_exp)
    if not cam_locked:
        warnings.append(f"camera.exposure_us {cam_exp:g} is not a whole number of video frames "
                        f"({VIDEO_FRAME_US:.0f} us): each frame catches a different slice of the "
                        "PWM bit-plane / LED sequence, which adds frame-to-frame noise to every "
                        f"gray level; measure at a whole-frame exposure (e.g. "
                        f"{snap_exposure_us(cam_exp)} us) if it does not clip - run tune")

    store = session.store
    label = f"gamma_{preset.name}"
    out = store.output_path("tuning", label, ".json")
    report = GammaReport(gamma=round(gamma, 4), levels=levels,
                         response=[round(r, 5) for r in response], r2=round(r2, 5),
                         recommended=recommended, path=out, warnings=warnings,
                         drive_levels=by_lv, polarity_inverted=inverted,
                         residual_gamma=None if residual is None else round(residual, 4),
                         sent_levels=[sent[L] for L in by_lv], hdmi_inverted=hdmi_inverted)
    store.write_json(out, {
        "kind": "gamma", "preset": preset.name, "geometry_id": preset.geometry_id,
        "mock": preset.mock, "gamma": report.gamma, "r2": report.r2,
        "residual_gamma": report.residual_gamma,
        "fit": {"model": "log(response) = gamma*log(sent/255) + c",
                "dark": "mirrors-off raw frame (raw black/white pair, polarity measured)",
                "c": round(c, 5), "min_level_fraction": GAMMA_FIT_MIN,
                "n_points": int(fit.sum())},
        "residual_fit": None if residual is None else {
            "model": "log(response) = residual_gamma*log(chain/255) + c",
            "c": round(c_res, 5), "r2": round(r2_res, 5), "n_points": int(fit_c.sum())},
        "levels": levels, "drive_levels": by_lv, "sent_levels": report.sent_levels,
        "chain_levels": [chain[L] for L in by_lv],
        "response": report.response,
        "response_counts": [round(counts[L], 3) for L in by_lv],
        "polarity_inverted": inverted, "hdmi_inverted": hdmi_inverted,
        "lit_pct": round(100.0 * float(lit.mean()), 3),
        "saturated_pct": round(sat_pct, 3),
        "chain": {"gamma": g0, "duty": pc.duty, "invert": pc.invert},
        "camera": {"exposure_us": cam_exp, "average": preset.camera.average,
                   "frame_locked": cam_locked, "video_frame_us": round(VIDEO_FRAME_US, 3)},
        "recommended": report.recommended, "warnings": warnings, "units": report.units,
        "note": "gamma = the projector's own response vs the value sent over HDMI (duty -> "
                "invert -> precomp, then HDMI video-mode degamma -> LEDs -> camera); it is the "
                "projection.gamma to use, directly. residual_gamma = the WHOLE path as driven by "
                "this preset, vs the chain value before the precomp (1 = linear). Both are "
                "measured above the mirrors-off frame; levels are the sent values, "
                "brightness-ordered"})
    store.record_output("tuning", out, label=label, geometry_id=preset.geometry_id)
    return report

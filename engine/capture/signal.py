"""
check_signal: a short live probe (white, black, N phase fringes) that
tells a human or agent whether the setup is good enough to scan.

The contrast mask and polarity handling are first_scan_v10's; the fringe
modulation comes from swiss_army_sensor's decode_phase_shift and is the
quantity the bench logs call gc_conf (stored in the `modulation` field).
gc_conf = median modulation over VALID pixels — the statistic behind the
briefing's numbers (v10 reference.npz: 36.5 at 38.5% valid).
Bands are from the briefing: <8 unusable, ~18 valid px appear, ~30
usable, 120-170 excellent.

The fringes are the preset's own: along its basis axis ('x' = vertical
stripes coding projector columns, 'y' = horizontal stripes coding rows)
with the steps / period the scan decodes (bases.hybrid.from_preset_bases),
so the probe judges exactly what a scan will see. Depth needs the coded
axis, as the camera sees it, to be the rig's baseline axis
(rig.coded_axis); the report carries the measured axis and axis_ok, and a
warning names the preset's basis axis and the rig's axis when they differ.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from .. import _seed, config
from ..bases import hybrid
from ..results import SignalReport

QUALITY_BANDS = ((120.0, "excellent"), (30.0, "good"), (18.0, "usable"),
                 (8.0, "marginal"), (0.0, "unusable"))
# clipped counts / warn level: the scan decoder's (bases/hybrid.py), so the
# probe, tune and a scan agree on what "saturated" means
SAT_LEVEL = hybrid.SAT_LEVEL
SAT_WARN_PCT = hybrid.SAT_WARN_PCT
_ORIENTATION = {"x": "vertical", "y": "horizontal"}
POLARITY_MARGIN = 2.0          # camera counts (frame mean) white vs black to judge polarity


def quality_band(gc_conf: float) -> str:
    for floor, name in QUALITY_BANDS:
        if gc_conf >= floor:
            return name
    return "unusable"


def focus_hint(ratio: float) -> str:
    if ratio >= 0.7:
        return "sharp"
    if ratio >= 0.4:
        return "soft - refocus on the target plane"
    return "blurred - fringes not resolved; refocus or use coarser period"


def coded_axis(phase: np.ndarray, valid: np.ndarray) -> str:
    """Camera axis along which the fringe phase changes fastest."""
    dx = np.abs(np.angle(np.exp(1j * (phase[:, 1:] - phase[:, :-1]))))[valid[:, 1:] & valid[:, :-1]]
    dy = np.abs(np.angle(np.exp(1j * (phase[1:, :] - phase[:-1, :]))))[valid[1:, :] & valid[:-1, :]]
    if dx.size == 0 or dy.size == 0:
        return ""
    return "x" if np.median(dx) > np.median(dy) else "y"


def fringe_spec(preset) -> tuple:
    """(basis axis 'x' | 'y', steps, period_px) of the fringes the preset's
    scan projects. Falls back to the phaseshift spec for a basis list the
    hybrid composite does not cover."""
    try:
        b = hybrid.from_preset_bases(preset.bases)
        return b.axis, int(b.steps), int(b.period_px)
    except (ValueError, NotImplementedError, KeyError, TypeError):
        return (config.basis_axis(preset.bases),
                int(preset.basis_param("phaseshift", "steps", 4)),
                int(preset.basis_param("phaseshift", "period_px", 24)))


def saturated_pct(bright: np.ndarray, dark: np.ndarray, contrast_min: float) -> float:
    """% of lit camera px at/above SAT_LEVEL in the physically bright flood.
    A px clipped in both floods has no measured contrast, so clipped px count
    whether or not they still look lit (same rule as assist.tune)."""
    sat = bright >= SAT_LEVEL
    region = ((bright - dark) > contrast_min) | sat
    return 100.0 * float(sat.sum()) / float(region.sum()) if region.any() else 0.0


def _axis_warning(measured: str, need: str, basis_axis: str, preset) -> str:
    other = "y" if basis_axis == "x" else "x"
    rot = " (rotated by projection.rotate90)" if preset.projection.rotate90 else ""
    return (f"coded axis: fringes vary along camera {measured.upper()}, but the projector is "
            f"{preset.rig.projector_side} of the camera, so depth shifts run along camera "
            f"{need.upper()} (rig axis {need}): sensitivity is near zero. The preset codes basis "
            f"axis '{basis_axis}' ({_ORIENTATION[basis_axis]} DMD stripes{rot}). Toggle "
            f"projection.rotate90, or code basis axis '{other}', or fix rig.projector_side - "
            "each is a geometry-bound setting: save it under a new geometry_id and capture a "
            "new base.")


def probe(sync, preset) -> SignalReport:
    P = _seed.load("patterns")
    D = _seed.load("decoders")
    if tuple(preset.projection.size) != (P.DMD_W, P.DMD_H):
        raise ValueError(f"projection.size {preset.projection.size} != DMD "
                         f"{(P.DMD_W, P.DMD_H)}")
    basis_axis, steps, period = fringe_spec(preset)

    white = sync.shoot(P.flood(255))
    black = sync.shoot(P.black())
    fringes = [sync.shoot(p) for p in P.phase_shift_fringes(
        n_steps=steps, period_px=period, orientation=_ORIENTATION[basis_axis])]

    warnings = list(preset.warnings)
    contrast = white - black
    flipped = float(contrast.mean()) < 0
    invert = bool(preset.projection.invert)
    # below this the white/black difference is noise (projector off / aimed
    # away): say nothing about polarity, the "no lit pixels" warning covers it
    measurable = abs(float(contrast.mean())) >= POLARITY_MARGIN
    if flipped:                            # polarity flip (v10)
        contrast = -contrast
    if flipped and measurable:
        if invert:
            warnings.append("polarity inverted: the dark pattern lands brighter than white "
                            "(expected: projection.invert is on); decode measures polarity, "
                            "no action needed")
        else:
            warnings.append("polarity flipped: dark frame brighter than white with "
                            "projection.invert off - the HDMI path inverts; decode measures "
                            "polarity, but check projection.invert / the projector's input "
                            "settings")
    elif invert and measurable and not flipped:
        warnings.append("polarity: projection.invert is on but the white pattern lands "
                        "brighter - something else in the HDMI path inverts too (the two "
                        "cancel); decode measures polarity, no action needed")
    bright, dark = (black, white) if flipped else (white, black)
    sat_pct = saturated_pct(bright, dark, preset.decode.contrast_min)

    lit = contrast > preset.decode.contrast_min
    ps = D.decode_phase_shift(fringes)
    mod = ps.modulation
    valid = lit & (mod > preset.decode.conf_min)
    axis = coded_axis(ps.phase, valid)
    need = preset.rig.coded_axis
    axis_ok = (axis == need) if axis else None
    if axis_ok is False:
        warnings.append(_axis_warning(axis, need, basis_axis, preset))

    gc_conf = float(np.median(mod[valid])) if valid.any() else 0.0
    ratio = float(np.median(mod[valid] / (contrast[valid] / 2.0))) if valid.any() else 0.0
    valid_pct = 100.0 * float(valid.mean())
    if not lit.any():
        warnings.append("no lit pixels — is the projector on and aimed at the scene?")
    if sat_pct > SAT_WARN_PCT:
        warnings.append(f"saturated_pct {sat_pct:.2f}: {sat_pct:.2f}% of lit camera px clip "
                        f"(>= {SAT_LEVEL:g} counts) under full light - clipped fringe peaks bias "
                        "the phase; lower camera.exposure_us or the light (run tune)")
    warnings += _rig_setup_warnings(sync)
    q = quality_band(gc_conf)
    extra = _optional_fields(saturated_pct=round(sat_pct, 3), rig_axis=need,
                             basis_axis=basis_axis)
    return SignalReport(gc_conf=round(gc_conf, 2), valid_pct=round(valid_pct, 2),
                        contrast_mean=round(float(contrast.mean()), 2),
                        focus_ratio=round(ratio, 3), focus_hint=focus_hint(ratio),
                        quality=q, ok=(q in ("usable", "good", "excellent") and valid_pct > 1.0
                                       and axis_ok is not False),
                        geometry_id=preset.geometry_id, mock=preset.mock,
                        warnings=warnings, coded_axis=axis, axis_ok=axis_ok, **extra)


def _rig_setup_warnings(sync) -> list:
    """What the real rig noticed while opening: the DMD window not at the
    expected display rect (HDMIWindow.rect_ok False) and camera nodes the
    post-open reset could not restore (RealCamera.reset_state). Mock rigs
    and wrapped syncs (assist.tune's tap) have neither."""
    out = []
    window = getattr(getattr(sync, "projector", None), "window", None)
    if getattr(window, "rect_ok", None) is False and getattr(window, "rect_warning", None):
        out.append(window.rect_warning)
    failed = (getattr(getattr(sync, "camera", None), "reset_state", None) or {}).get("failed")
    if failed:
        out.append("camera reset after open could not restore "
                   + ", ".join(sorted(failed)) + " - if the frame size, mirroring or gamma "
                   "differ from the bench default, fix them in SpinView (camera defaults)")
    return out


def _optional_fields(**values) -> dict:
    """Only the fields SignalReport (engine/results.py) actually has, so the
    probe fills new report fields as soon as they exist without breaking
    before."""
    have = {f.name for f in dataclasses.fields(SignalReport)}
    return {k: v for k, v in values.items() if k in have}

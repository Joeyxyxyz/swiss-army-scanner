"""
The end-face radii: every one a fraction of the face radius.

A detector that inspects out to a fixed 9.75 mm is right for one face only: on a smaller flat
it inspects the chamfer (false REJECTs on good parts), on a larger one it leaves a ring of
the face unchecked. So every radius the detectors, the profile and the synthetic-dent set use
is derived from the face radius R (= face_diameter_mm / 2) unless it is given explicitly.
The fractions are the values the detectors were developed with on a 21.05 mm face
(R = 10.525 mm), which stay exactly what they were there:

    inspect_r_mm    0.926 R   (9.75)   the inspected zone r <= this (MF-CFAR, with or without
                                       the FIR): the chamfer / rounded rim beyond is never
                                       inspected. The band prototype compares the whole face
                                       (r <= R, its rim zone at the looser rim band): of
                                       these radii it uses the rim radius only
    fit_r_mm        = inspect_r_mm     data the matched filter's local fits use
    rim_r_mm        0.855 R   (9.0)    ONE rim radius for every detector: a trigger beyond it
                                       is labelled 'rim' and held to the rim tolerance (band:
                                       x rim_factor, min rim_min_mm; MF: rim_z_thresh when set)
    plane_r_mm      = rim_r_mm         robust plane region (the flat inside the rim)
    r_flat_mm       = rim_r_mm         face_profile: flatness inside this radius
    battery_r_mm    = rim_r_mm         synthetic dents: random centres r <= this
    fir_r_max_mm    0.974 inspect_r    (9.5) the reflectance FIR is fitted inside this radius
    noise_bands_mm  0, 0.285 R, 0.570 R, 0.760 R, rim_r, inspect_r, outer (0.974 R)
                                       (0, 3, 6, 8, 9, 9.75, 10.25) the CFAR's radial noise bands

face_radii(R) resolves them; given values win (and what follows from them follows: a given
inspect_r_mm moves fit_r_mm, the FIR radius and the noise bands with it, a given rim_r_mm the
plane / flatness / battery radii). A given radius must lie on the face and the rim radius
inside the inspected zone: anything else is refused, never clamped silently. The resolved
set is stored with every golden and in every check result ('radii').
"""

from __future__ import annotations

from typing import Optional

REF_RADIUS_MM = 10.525              # the 21.05 mm face the fractions come from
INSPECT_FRAC = 9.75 / REF_RADIUS_MM
RIM_FRAC = 9.0 / REF_RADIUS_MM
FIR_OF_INSPECT = 9.5 / 9.75
OUTER_FRAC = 10.25 / REF_RADIUS_MM
INNER_BAND_FRACS = (3.0 / REF_RADIUS_MM, 6.0 / REF_RADIUS_MM, 8.0 / REF_RADIUS_MM)
MIN_BAND_MM = 0.05                  # noise band edges closer than this are merged
KEYS = ("inspect_r_mm", "fit_r_mm", "rim_r_mm", "plane_r_mm", "r_flat_mm", "battery_r_mm",
        "fir_r_max_mm", "noise_bands_mm")
EPS = 1e-6


def _num(v, name: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not v == v or v in (
            float("inf"), float("-inf")):
        raise ValueError(f"{name} must be a finite number (mm), got {v!r}")
    return float(v)


def _bands(R: float, rim: float, inspect: float) -> tuple:
    """The CFAR noise bands: the scaled inner edges below the rim radius, the rim and the
    inspect radius, and one band beyond (the detector's sigma there is never used to trigger,
    it only keeps the outermost inspected band from borrowing chamfer noise)."""
    edges = [0.0] + [f * R for f in INNER_BAND_FRACS if f * R < rim - MIN_BAND_MM]
    edges += [rim, inspect, max(OUTER_FRAC * R, inspect + (OUTER_FRAC - INSPECT_FRAC) * R)]
    out = []
    for e in sorted(edges):
        if not out or e - out[-1] > MIN_BAND_MM:
            out.append(float(e))
    return tuple(out)


def face_radii(radius_mm: float, over: Optional[dict] = None) -> dict:
    """The resolved radii of a face of radius radius_mm (mm). over: explicit values by KEYS
    (None = not given). -> {every KEYS entry, face_radius_mm, given: [the names given]}.
    ValueError on an unknown name, a radius that is not on the face, a rim radius outside
    the inspected zone, or noise bands that do not cover it."""
    R = _num(radius_mm, "the face radius")
    if not R > 0:
        raise ValueError(f"the face radius must be > 0 mm, got {radius_mm!r}")
    over = {k: v for k, v in (over or {}).items() if v is not None}
    bad = sorted(set(over) - set(KEYS))
    if bad:
        raise ValueError(f"unknown end-face radii {bad} (known: {list(KEYS)})")
    g = {k: _num(over[k], k) for k in over if k != "noise_bands_mm"}
    for k, v in g.items():
        if not 0 < v <= R + EPS:
            raise ValueError(f"{k} {v:g} mm is not on the {2 * R:g} mm face (0 < r <= {R:g} "
                             "mm): every radius is given in mm from the face centre")
    inspect = g.get("inspect_r_mm", INSPECT_FRAC * R)
    fit = g.get("fit_r_mm", inspect)
    rim = g.get("rim_r_mm", min(RIM_FRAC * R, inspect * RIM_FRAC / INSPECT_FRAC))
    if rim > inspect + EPS:
        raise ValueError(f"rim_r_mm {rim:g} mm lies outside the inspected zone r <= "
                         f"{inspect:g} mm of this {2 * R:g} mm face: the rim zone is the outer "
                         "part of what is inspected (rim_r_mm <= inspect_r_mm)")
    if fit < inspect - EPS:
        raise ValueError(f"fit_r_mm {fit:g} mm is inside the inspected zone r <= {inspect:g} "
                         "mm: the matched filter needs the data it inspects")
    out = {"face_radius_mm": R, "inspect_r_mm": inspect, "fit_r_mm": fit, "rim_r_mm": rim,
           "plane_r_mm": g.get("plane_r_mm", rim), "r_flat_mm": g.get("r_flat_mm", rim),
           "battery_r_mm": g.get("battery_r_mm", rim),
           "fir_r_max_mm": g.get("fir_r_max_mm", FIR_OF_INSPECT * inspect)}
    if "noise_bands_mm" in over:
        try:
            b = tuple(float(x) for x in over["noise_bands_mm"])
        except (TypeError, ValueError):
            raise ValueError(f"noise_bands_mm must be band edges (mm), got "
                             f"{over['noise_bands_mm']!r}") from None
        if len(b) < 2 or any(y <= x for x, y in zip(b[:-1], b[1:])) or b[0] > EPS \
                or b[-1] < inspect - EPS:
            raise ValueError(f"noise_bands_mm {list(b)} must rise from 0 and cover the "
                             f"inspected zone r <= {inspect:g} mm")
        out["noise_bands_mm"] = b
    else:
        out["noise_bands_mm"] = _bands(R, rim, inspect)
    out["given"] = sorted(over)
    return out


def scale_note(r: dict) -> str:
    """One line: the radii and where they come from."""
    R = r["face_radius_mm"]
    src = (f"given: {', '.join(r['given'])}; the rest" if r.get("given") else "all")
    return (f"face {2 * R:g} mm: inspect r <= {r['inspect_r_mm']:.3f} mm, rim r > "
            f"{r['rim_r_mm']:.3f} mm, plane r <= {r['plane_r_mm']:.3f} mm ({src} derived from "
            "the face diameter)")


def public(r: dict) -> dict:
    """The radii as stored (rounded, lists)."""
    out = {k: (round(float(v), 4) if not isinstance(v, (tuple, list)) else
               [round(float(x), 4) for x in v]) for k, v in r.items() if k != "given"}
    out["given"] = list(r.get("given") or [])
    out["note"] = ("mm from the face centre; not given = a fixed fraction of the face radius "
                   "(the band detector compares the whole face and uses rim_r_mm only)")
    return out

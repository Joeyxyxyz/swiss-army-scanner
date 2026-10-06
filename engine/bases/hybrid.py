"""
Hybrid basis: gray code (coarse, absolute) + N-step phase shift (fine) —
the proven 3D path. Generation and decode are first_scan_v10's
`capture_stack` / `decode_correspondence`, unchanged in substance:

  1. white + black        -> contrast mask, polarity, threshold datum (white+black)/2
  2. gray code, n bits    -> absolute projector column / row (coarse)
  3. phase shift, N steps -> wrapped fine phase (swiss_army_sensor decoder)
  4. unwrap fine against coarse -> projector column / row per camera pixel

Options (all off by default = v10, bit-identical):

  axis="y"            horizontal stripes -> projector ROW per camera pixel
                      (corr_y). Extent is DMD_H (1140), so the stripe width
                      comes from 1140 exactly like the seed generator does.
  complementary=True  TI DLP SDK "include_inverted": every gray plane is
                      followed by its inverse and a bit is normal-vs-inverse,
                      so albedo, ambient and blur do not bias it the way a
                      global threshold is biased. A pixel is invalid when any
                      used plane differs from its inverse by less than
                      `min_bit_contrast` grey levels (TI default 5): that bit
                      is a coin flip there.
  coarse_bits=N       project and decode only the N coarsest planes. On the
                      bench one DMD column is ~1.5 camera px and 1-mirror
                      stripes keep ~1% contrast, so the fine planes are noise;
                      dropping them makes the gray code a coarse stripe index
                      and the phase does the fine work. Needs the coarse
                      stripe narrower than period/2 (checked at construction).
  fix_period_jumps=True
                      after unwrapping, snap each pixel's fringe order to a
                      local median (decode.unwrap.fix_period_jumps). Opt-in:
                      it removes isolated ±period spikes left by misread fine
                      gray planes, but can pull pixels at a depth edge taller
                      than period/2 to the wrong side.

Clipped fringe samples (on by default, sat_level=250 Mono8 counts; None =
off, the v10 behaviour): a sinusoid clipped at the sensor's full scale gives
the N-step decoder a systematic phase error with a period of 1/N of the
fringe (0.7 projector px at N=4 on a bright part = ~1.4 mm of false height
on the bench). A fringe sample >= sat_level is dropped and the phase and
modulation are re-solved from the unclipped samples: least squares with >= 3
(exact for a sinusoid); with 2, only when the clip level and the dim flood
bound the phase to within DECLIP_TOL_PX; otherwise the pixel is invalid —
never kept with a rippled phase. Pixels with no clipped sample decode
exactly as before. meta records saturated_pct / saturated_resolved_pct /
saturated_masked_pct (% of lit camera px) and a warning when the scene
clips.

The white/black frames decide polarity in every mode: the bench HDMI path
lands patterns inverted, and that is measured per capture, never assumed.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .. import _seed
from ..decode.deformation import DeformationField
from .base import PatternBasis
from .registry import register

# TI DLP Structured Light SDK default: a gray bit whose pattern and inverse
# differ by fewer grey levels than this is not trusted.
TI_MIN_BIT_CONTRAST = 5.0
# Mono8 counts at/above which a (camera-averaged) frame is treated as
# clipped — the same level assist.tune uses (capture/assist.py SAT_LEVEL).
SAT_LEVEL = 250.0
# warn when more than this % of the lit camera px clip in a used frame
SAT_WARN_PCT = 0.1
# unknowns of a fringe sample A + B cos(phi + delta): offset, B cos, B sin
_MIN_PHASE_SAMPLES = 3
# a pixel with only 2 unclipped fringe samples is kept when every phase that
# fits the data lies within this many projector px (worst error: half of it,
# ~0.1 mm at the bench's ~1.9 mm per projector px)
DECLIP_TOL_PX = 0.1
# counts of slack on the bounds (clipped >= sat_level, trough >= dim flood):
# about 2 sigma of the Blackfly's single-grab noise
DECLIP_MARGIN = 3.0

_AXIS = {"x": ("vertical", "DMD_W", "column"), "y": ("horizontal", "DMD_H", "row")}


@register
class HybridBasis(PatternBasis):
    name = "hybrid"
    axes = {"x", "y"}            # what the basis can yield; an instance yields {axis}

    def __init__(self, bits: int = 10, steps: int = 4, period_px: int = 24,
                 complementary: bool = False, axis: str = "x",
                 coarse_bits: Optional[int] = None,
                 min_bit_contrast: float = TI_MIN_BIT_CONTRAST,
                 sat_level: Optional[float] = SAT_LEVEL,
                 fix_period_jumps: bool = False):
        if axis not in _AXIS:
            raise ValueError(
                f"hybrid: axis must be 'x' (vertical stripes -> projector column) or "
                f"'y' (horizontal stripes -> projector row), got {axis!r}")
        bits, steps, period_px = int(bits), int(steps), int(period_px)
        if bits < 1:
            raise ValueError(f"hybrid: bits must be >= 1, got {bits}")
        if steps < 3:
            raise ValueError(f"hybrid: phase shift needs >= 3 steps, got {steps}")
        if period_px < 2:
            raise ValueError(f"hybrid: period_px must be >= 2, got {period_px}")
        if coarse_bits is not None:
            if isinstance(coarse_bits, bool) or int(coarse_bits) != coarse_bits:
                raise ValueError(f"hybrid: coarse_bits must be an integer, got {coarse_bits!r}")
            coarse_bits = int(coarse_bits)
            if not 1 <= coarse_bits <= bits:
                raise ValueError(
                    f"hybrid: coarse_bits must be in 1..bits ({bits}), got {coarse_bits}")
        min_bit_contrast = float(min_bit_contrast)
        if not min_bit_contrast >= 0:
            raise ValueError(f"hybrid: min_bit_contrast must be >= 0 grey levels, "
                             f"got {min_bit_contrast}")
        if sat_level is not None:
            if isinstance(sat_level, bool):
                raise ValueError(f"hybrid: sat_level must be a number of counts or None, "
                                 f"got {sat_level!r}")
            sat_level = float(sat_level)
            if not 0 < sat_level <= 255:
                raise ValueError(f"hybrid: sat_level must be in (0, 255] Mono8 counts "
                                 f"(None = off), got {sat_level:g}")
        if not isinstance(fix_period_jumps, bool):
            raise ValueError(f"hybrid: fix_period_jumps must be true/false, "
                             f"got {fix_period_jumps!r}")
        super().__init__(bits=bits, steps=steps, period_px=period_px,
                         complementary=bool(complementary), axis=axis,
                         coarse_bits=coarse_bits, min_bit_contrast=min_bit_contrast,
                         sat_level=sat_level, fix_period_jumps=fix_period_jumps)
        self.bits, self.steps, self.period_px = bits, steps, period_px
        self.axis = axis
        self.axes = {axis}
        self.coarse_bits = coarse_bits
        self.min_bit_contrast = min_bit_contrast
        self.sat_level = sat_level
        self.fix_period_jumps = fix_period_jumps
        self._comp = bool(complementary)

        P = _seed.load("patterns")
        self.orientation = _AXIS[axis][0]
        self.extent = int(getattr(P, _AXIS[axis][1]))     # DMD px along the coded axis
        # finest stripe of the full code — the seed generator's own formula
        self.stripe_px = max(1, -(-self.extent // (2 ** bits)))
        self.used_bits = coarse_bits if coarse_bits is not None else bits
        self.dropped_bits = bits - self.used_bits
        # the top N planes of an n-bit gray code are the N-bit gray code of
        # idx >> (n - N), so the decoded stripe is 2**dropped fine stripes wide
        self.coarse_stripe_px = self.stripe_px << self.dropped_bits
        self._check_unwrap()

    def _check_unwrap(self) -> None:
        """k = round((coarse - fine) / period) is only unambiguous while the
        coarse position is well inside half a period of the truth; a stripe
        narrower than period/2 keeps the stripe-centre datum within period/4,
        leaving the other period/4 as margin for blur and edge misreads."""
        if 2 * self.coarse_stripe_px < self.period_px:
            return
        ok = [n for n in range(1, self.bits + 1)
              if 2 * (self.stripe_px << (self.bits - n)) < self.period_px]
        fix = (f"use coarse_bits >= {ok[0]}" if ok else
               f"use more bits (the finest {self.bits}-bit stripe is already "
               f"{self.stripe_px} px)")
        raise ValueError(
            f"hybrid axis {self.axis}: gray-code stripes would be {self.coarse_stripe_px} "
            f"DMD px wide ({self.used_bits} of {self.bits} bits over {self.extent} px), but "
            f"unwrapping the {self.period_px} px fringe needs stripes < period/2 = "
            f"{self.period_px / 2:g} px — {fix} or a longer period_px")

    def n_frames(self) -> int:
        return 2 + self.used_bits * (2 if self._comp else 1) + self.steps

    def generate(self, dmd_shape=None) -> list[np.ndarray]:
        P = _seed.load("patterns")
        planes = P.gray_code_stripes(n_bits=self.bits,
                                     orientation=self.orientation)[self.dropped_bits:]
        if self._comp:
            planes = [q for g in planes for q in (g, 255 - g)]
        return ([P.flood(255), P.black()] + planes
                + P.phase_shift_fringes(n_steps=self.steps, period_px=self.period_px,
                                        orientation=self.orientation))

    def decode(self, frames, geometry_id: str, base_frames=None,
               contrast_min: float = 8.0, conf_min: float = 4.0) -> DeformationField:
        D = _seed.load("decoders")
        if len(frames) != self.n_frames():
            raise ValueError(
                f"hybrid expects {self.n_frames()} frames (white, black, {self.used_bits} "
                f"gray planes{' + inverses' if self._comp else ''}, {self.steps} fringes), "
                f"got {len(frames)}")
        n_gray = self.used_bits * (2 if self._comp else 1)
        white, black = frames[0].astype(np.float64), frames[1].astype(np.float64)
        gray_frames = frames[2:2 + n_gray]
        ps_frames = frames[2 + n_gray:]

        contrast = white - black
        flipped = float(contrast.mean()) < 0
        if flipped:
            white, black = black, white
            contrast = -contrast
        mask = contrast > contrast_min
        lit = mask.copy()

        r = D.decode_phase_shift(ps_frames)
        phase, modulation = r.phase, r.modulation
        sat = None
        if self.sat_level is not None:
            phase, modulation, clipped, starved = _declip_phase(
                ps_frames, phase, modulation, self.sat_level, black, self.period_px)
            sat = {"clipped": clipped, "starved": starved,
                   "flood": np.maximum(white, black) >= self.sat_level}
        if flipped:
            # Inverted fringes are cos(...)+pi. Without this the fine phase sits
            # half a period from the gray-code column — exactly on the rounding
            # boundary — and k is a coin flip (dense ±period errors). v10 had
            # this bug; found on the bench 2026-09-22.
            phase = np.angle(np.exp(1j * (phase + np.pi)))
        mask &= modulation > conf_min
        if sat is not None:
            # clipped beyond recovery (see _declip_phase): never keep a guess
            sat["dropped"] = mask & sat["starved"]
            mask &= ~sat["starved"]

        # per-plane contrast (grey levels) on a subsample of the lit pixels:
        # the readout that says which planes the camera resolves (-> coarse_bits)
        sample = mask[::4, ::4]
        plane_contrast = []

        code = np.zeros(white.shape, dtype=np.uint64)
        weak = None
        if self._comp:
            weak = np.zeros(white.shape, dtype=bool)
            for b in range(self.used_bits):
                d = (gray_frames[2 * b].astype(np.float64)
                     - gray_frames[2 * b + 1].astype(np.float64))
                bit = (d < 0) if flipped else (d > 0)
                weak |= np.abs(d) < self.min_bit_contrast
                code |= bit.astype(np.uint64) << np.uint64(b)
                plane_contrast.append(_median(np.abs(d[::4, ::4]), sample))
            weak_pct = 100.0 * float(weak[mask].mean()) if mask.any() else 0.0
            mask &= ~weak
        else:
            thresh = (white + black) / 2.0
            for b, g in enumerate(gray_frames):
                bit = (g < thresh) if flipped else (g > thresh)
                code |= bit.astype(np.uint64) << np.uint64(b)
                # 2|g - datum| is on the same scale as |normal - inverse|
                plane_contrast.append(_median(
                    2.0 * np.abs(g[::4, ::4] - thresh[::4, ::4]), sample))
        binary = code.copy()
        shift = 1
        while shift < self.used_bits:
            binary ^= binary >> np.uint64(shift)
            shift <<= 1
        coarse = binary.astype(np.float64) * self.coarse_stripe_px
        if self.coarse_bits is not None:
            # wide stripes: the stripe centre halves the worst-case coarse
            # error. The full code keeps v10's left edge (stripes of 1-2 px,
            # and bit-identical output to the proven path).
            coarse += (self.coarse_stripe_px - 1) / 2.0

        phi_coarse = 2 * np.pi * coarse / self.period_px
        k = np.round((phi_coarse - phase) / (2 * np.pi))
        pos = (phase + 2 * np.pi * k) * self.period_px / (2 * np.pi)
        jumps_pct = None
        if self.fix_period_jumps:
            # before the clip, so an edge pixel one period out is still
            # recognisable as a jump rather than a clamped value
            from ..decode.unwrap import fix_period_jumps
            fixed = fix_period_jumps(pos, mask, self.period_px)
            moved = mask & (np.abs(fixed - pos) > 0.5 * self.period_px)
            jumps_pct = round(100.0 * float(moved.sum()) / max(int(mask.sum()), 1), 3)
            pos = fixed
        pos = np.clip(pos, 0, self.extent - 1)
        corr = np.where(mask, pos, np.nan).astype(np.float32)

        meta = {"polarity_flipped": flipped, "bits": self.bits, "steps": self.steps,
                "period_px": self.period_px,
                "gc_conf": float(np.median(modulation[mask])) if mask.any() else 0.0,
                "contrast_mean": float(contrast.mean()),
                "axis": self.axis, "coded": _AXIS[self.axis][2],
                "units": "projector px",
                "complementary": self._comp, "coarse_bits": self.coarse_bits,
                "gray_bits_used": self.used_bits,
                "coarse_stripe_px": self.coarse_stripe_px,
                "bit_contrast_coarse_to_fine": plane_contrast[::-1],
                "bit_contrast_units": "grey levels"}
        if self._comp:
            meta.update({"min_bit_contrast": self.min_bit_contrast,
                         "weak_bit_pct": round(weak_pct, 3)})
        meta.update({"fix_period_jumps": self.fix_period_jumps,
                     "period_jumps_fixed_pct": jumps_pct})
        meta.update(self._saturation_meta(sat, lit))
        return DeformationField(
            corr_x=corr if self.axis == "x" else None,
            corr_y=corr if self.axis == "y" else None,
            corr_diag=None, conf=modulation.astype(np.float32), mask=mask,
            geometry_id=geometry_id, basis_names=["graycode", "phaseshift"], meta=meta)

    def _saturation_meta(self, sat: Optional[dict], lit: np.ndarray) -> dict:
        """saturated_* as % of the lit camera px. A pixel clipped in both
        floods has ~0 measured contrast and drops out of `lit` exactly when
        clipping is worst, so clipped pixels always count (as assist.tune)."""
        out = {"sat_level": self.sat_level, "saturated_pct": None,
               "saturated_resolved_pct": None, "saturated_masked_pct": None,
               "saturated_units": "% of lit camera px", "warnings": []}
        if sat is None:
            return out
        any_clip = sat["clipped"] | sat["flood"]
        region = lit | any_clip
        n = int(region.sum())

        def pct(m):
            return round(100.0 * float((m & region).sum()) / n, 3) if n else 0.0

        resolved = sat["clipped"] & ~sat["starved"]
        out.update({"saturated_pct": pct(any_clip),
                    "saturated_resolved_pct": pct(resolved),
                    "saturated_masked_pct": pct(sat["dropped"])})
        if out["saturated_pct"] > SAT_WARN_PCT:
            out["warnings"].append(
                f"{out['saturated_pct']:.2f}% of lit camera px clip (>= {self.sat_level:g} "
                f"counts) in the white or fringe frames: {out['saturated_resolved_pct']:.2f}% "
                f"had their phase re-solved from the unclipped fringe frames (noisier), "
                f"{out['saturated_masked_pct']:.2f}% clipped too far to recover and were "
                f"dropped. Lower the exposure or the duty (assist.tune) for full accuracy "
                f"on bright/shiny surfaces.")
        return out


def _declip_phase(ps_frames, phase: np.ndarray, modulation: np.ndarray, sat_level: float,
                  floor: np.ndarray, period_px: float):
    """Re-solve the wrapped phase where fringe samples clipped.

    Model (the seed decode_phase_shift's convention): sample k of N is
    I_k = A + B cos(phi + 2 pi k / N), phase = phi, modulation = B. Samples
    >= sat_level are clipped; per pixel:

      >= 3 unclipped  (A, B cos phi, B sin phi) by least squares from them —
                      exact for a sinusoid.
      2 unclipped     under-determined: a line of offsets A fits them. The
                      clipped samples must reach sat_level and the fringe
                      trough A - B can't sit below `floor` (the dim flood,
                      sent at the trough's drive, so blur only raises the
                      trough), which bounds A. When every phase in that
                      bounded family lies within DECLIP_TOL_PX projector px,
                      the pixel keeps the family's middle phase (worst error
                      half that); otherwise it is dropped. Mild
                      over-exposure is recovered, a glint is not.
      < 2 unclipped   dropped.

    Pixels are grouped by which samples clipped, so each distinct pattern
    costs one small solve.

    -> (phase, modulation, clipped (H, W) bool = any sample clipped,
        starved (H, W) bool = not recoverable, to be masked by the caller).
    Unclipped pixels keep the seed values bit-for-bit.
    """
    n = len(ps_frames)
    shape = np.shape(phase)
    flat = [np.asarray(f).reshape(-1) for f in ps_frames]
    clip = [f >= sat_level for f in flat]
    n_clip = np.zeros(flat[0].shape, np.int32)
    for c in clip:
        n_clip += c
    starved = (n - n_clip) < 2
    redo = (n_clip > 0) & ~starved
    clipped = (n_clip > 0).reshape(shape)
    if not redo.any():
        return phase, modulation, clipped, starved.reshape(shape)
    idx = np.flatnonzero(redo)
    sub = [c[idx] for c in clip]
    if n <= 62:                  # one int key per clip pattern: 1-D grouping is fast
        key = np.zeros(idx.size, np.int64)
        for j, c in enumerate(sub):
            key |= c.astype(np.int64) << j
        codes = np.flatnonzero(np.bincount(key)) if n <= 20 else np.unique(key)
        groups = [(((int(code) >> np.arange(n)) & 1).astype(bool), key == code)
                  for code in codes]
    else:
        pats, which = np.unique(np.stack(sub, axis=1), axis=0, return_inverse=True)
        which = np.asarray(which).reshape(-1)
        groups = [(pat, which == j) for j, pat in enumerate(pats)]
    delta = 2 * np.pi * np.arange(n) / n
    design = np.stack([np.ones(n), np.cos(delta), -np.sin(delta)], axis=1)   # (N, 3)
    tol_rad = 2 * np.pi * DECLIP_TOL_PX / float(period_px)
    floor_flat = np.asarray(floor, dtype=np.float64).reshape(-1)
    phase, modulation = phase.copy(), modulation.copy()
    ph_flat, mod_flat = phase.reshape(-1), modulation.reshape(-1)             # views
    for pat, sel in groups:
        pix = idx[sel]
        good = np.flatnonzero(~pat)
        vals = np.stack([flat[k][pix].astype(np.float64) for k in good])     # (K, M)
        if good.size >= _MIN_PHASE_SAMPLES:
            _, a, b = np.linalg.pinv(design[good]) @ vals
            ok = np.ones(pix.size, bool)
        else:
            a, b, ok = _bounded_pair(vals, design[good, 1:], design[pat, 1:],
                                     sat_level, floor_flat[pix], tol_rad)
        ph_flat[pix[ok]] = np.arctan2(b[ok], a[ok])
        mod_flat[pix[ok]] = np.hypot(a[ok], b[ok])
        starved[pix[~ok]] = True
    return phase, modulation, clipped, starved.reshape(shape)


def _bounded_pair(vals: np.ndarray, rows_good: np.ndarray, rows_clip: np.ndarray,
                  sat_level: float, floor: np.ndarray, tol_rad: float):
    """Two unclipped samples -> (a, b) = (B cos phi, B sin phi) at the middle
    of the feasible family, and whether that family's phase spread is
    within tol_rad (else the pixel is not recoverable).

    For offset A the two samples fit exactly with (a, b) = u - A v, where
    u = R^-1 I, v = R^-1 1 and R holds their [cos d, -sin d] rows. Feasible
    A: each clipped sample A + c.(u - A v) >= sat_level - margin (linear in
    A), and the trough A - |u - A v| >= floor - margin (quadratic in A,
    bounded when |v| > 1). (a, b) moves along a straight line, so the
    phase spread is the angle between the two end points.
    """
    m = vals.shape[1]
    none = (np.zeros(m), np.zeros(m), np.zeros(m, bool))
    if abs(np.linalg.det(rows_good)) < 1e-9:               # samples 180 deg apart
        return none
    ri = np.linalg.inv(rows_good)
    u = ri @ vals                                          # (2, M)
    v = ri @ np.ones(2)                                    # (2,)
    qa = float(v @ v) - 1.0
    if qa <= 1e-9:                                         # the trough does not bound A
        return none
    lo, hi = np.full(m, -np.inf), np.full(m, np.inf)
    ok = np.ones(m, bool)
    for c in rows_clip:
        s = 1.0 - float(c @ v)
        r = (sat_level - DECLIP_MARGIN) - c @ u
        if s > 1e-12:
            lo = np.maximum(lo, r / s)
        elif s < -1e-12:
            hi = np.minimum(hi, r / s)
        else:
            ok &= r <= 0
    d = floor - DECLIP_MARGIN
    qb = -2.0 * (v @ u - d)
    qc = (u * u).sum(axis=0) - d * d
    disc = qb * qb - 4.0 * qa * qc
    ok &= disc >= 0
    sq = np.sqrt(np.maximum(disc, 0.0))
    lo = np.maximum.reduce([lo, (-qb - sq) / (2 * qa), d])
    hi = np.minimum(hi, (-qb + sq) / (2 * qa))
    ok &= lo <= hi
    lo, hi = np.where(ok, lo, 0.0), np.where(ok, hi, 0.0)
    p1 = u - lo * v[:, None]
    p2 = u - hi * v[:, None]
    n1, n2 = np.hypot(*p1), np.hypot(*p2)
    ok &= (n1 > 1e-9) & (n2 > 1e-9)
    n1, n2 = np.where(ok, n1, 1.0), np.where(ok, n2, 1.0)
    spread = np.abs(np.angle(np.exp(1j * (np.arctan2(p2[1], p2[0])
                                          - np.arctan2(p1[1], p1[0])))))
    ok &= spread <= tol_rad
    mid = p1 / n1 + p2 / n2                                # bisector: the middle phase
    b_mid = np.hypot(*(u - 0.5 * (lo + hi) * v[:, None]))  # modulation at the middle A
    scale = b_mid / np.maximum(np.hypot(*mid), 1e-12)
    return mid[0] * scale, mid[1] * scale, ok


def _median(values: np.ndarray, sample: np.ndarray) -> float:
    v = values[sample]
    return round(float(np.median(v)), 2) if v.size else 0.0


def from_preset_bases(specs: list[dict], axis: Optional[str] = None) -> HybridBasis:
    """graycode + phaseshift listed in a preset -> the hybrid composite.

    `axis` overrides the preset's axis, so one preset drives both the x
    (columns) and y (rows) runs of a projector map. The gray-code spec's
    coarse_bits applies to either axis; if it is too coarse for the other
    axis's extent the ValueError names the smallest coarse_bits that works.
    fix_period_jumps rides on the gray-code spec, sat_level (clipped-sample
    handling; null = off) on the phase-shift spec.
    """
    by = {s["name"]: s for s in specs}
    if "hybrid" in by:
        s = {k: v for k, v in by["hybrid"].items() if k != "name"}
        if axis is not None:
            s["axis"] = axis
        return HybridBasis(**s)
    if {"graycode", "phaseshift"} <= set(by):
        g, p = by["graycode"], by["phaseshift"]
        if "axis" in g and "axis" in p and g["axis"] != p["axis"]:
            raise ValueError(
                f"bases: graycode axis {g['axis']!r} != phaseshift axis {p['axis']!r}; "
                "the hybrid decode unwraps one against the other, so they must match")
        return HybridBasis(bits=g.get("bits", 10), complementary=g.get("complementary", False),
                           axis=axis or g.get("axis", p.get("axis", "x")),
                           coarse_bits=g.get("coarse_bits"),
                           min_bit_contrast=g.get("min_bit_contrast", TI_MIN_BIT_CONTRAST),
                           fix_period_jumps=g.get("fix_period_jumps", False),
                           steps=p.get("steps", 4), period_px=p.get("period_px", 24),
                           sat_level=p.get("sat_level", SAT_LEVEL))
    raise NotImplementedError(
        f"bases {sorted(by)}: only graycode+phaseshift (hybrid) is implemented so far")

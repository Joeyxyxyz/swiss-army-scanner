"""
Flash pattern loop + free-running camera, sorted in software ("soft sync").

The DLPC350 plays a sequence of 1-bit patterns out of ONE 24-bit flash image
on its internal timer, on repeat. A human sets that up: the images go into
flash through the firmware link in the EKB 4500 MKII GUI, and the GUI's
Pattern Sequence tab plays them (Pattern Source Flash, Trigger Mode
Internal/External, Play Once off: the loop repeats until Stop).
There is no trigger cable: the camera free-runs faster than the pattern
rate, and every frame is labelled afterwards from its timestamp and its
content (docs/FLASH_SOFTSYNC.md).

Why 1-bit: an 8-bit pattern is a PWM sequence of bit-planes inside its
exposure, so an unsynchronised camera integrates a random slice of it. A
1-bit pattern is constant for its whole exposure: any frame whose exposure
lies inside one pattern exposure is a clean sample of it. Sinusoids come
from dithered stripes, low-passed by a slight projector defocus.

    FlashLayout       the loop in play order: fastscan_layout() is the 16-slot
                      1-bit pack of engine/bases/onebit.py (tools/
                      make_flash_pack.py --fastscan); layout_from_manifest()
                      reads a pack's manifest.json
    plan_timing()     camera ROI / fps / exposure and the pattern exposure +
                      period that give >= 1 clean frame per pattern per loop
    FlashFreeRunSync  camera only: ROI, fixed exposure, fixed frame rate,
                      OldestFirst buffers, N frames with device + host
                      timestamps and frame ids -> Recording
    sort_frames()     pure (offline): loop period, pattern boundaries, clean /
                      straddling frames, sync-marker anchoring, content checks,
                      per-pattern averages -> SortResult, or SequenceError
    decode_sorted()   the sorted images -> the unchanged MultiFreqBasis decode
                      (mid / coarse prefiltered as the pack says)

What this module never does: talk to the projector. It imports no USB / HID
code (rule 1: the scan path never sends a DLPC350 command; dmd_control is
not imported) and FlashFreeRunSync runs rules.check_scan_path_usb_free().
The PySpin code at the bottom is bench-only and never runs in tests.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .. import rules

# ------------------------------------------------------------ camera model
# FLIR Blackfly S BFS-U3-16S2M: Sony IMX273 global shutter, 1440 x 1080 Mono8.
SENSOR_SHAPE = (1080, 1440)             # (rows, cols)
SENSOR_FULL_FPS = 276.0                 # IMX273 all-pixel readout: the sensor limit
SENSOR_OVERHEAD_ROWS = 36               # vertical blanking in row times (estimate)
USB_BYTES_PER_S = 1440 * 1080 * 226.0   # FLIR's 226 fps full-frame Mono8 = the USB3 link
EXPOSURE_OVERHEAD_US = 20.0             # exposure end -> next exposure start (estimate)
FPS_HEADROOM = 0.97                     # run a little under the computed maximum
# ROI alignment. Conservative multiples; on the bench the camera's own node
# increments (Width/Height/OffsetX/OffsetY .GetInc()) are read and used.
ROI_STEP = {"x": 16, "y": 4, "w": 16, "h": 4}
ROI_MIN = (32, 64)                      # (h, w)

# ------------------------------------------------------------ DLPC350 timing
MIN_1BIT_EXPOSURE_US = 235              # DLPU010G: minimum 1-bit pattern exposure
PERIOD_DIFFERENCE_MIN_US = 230          # exposure == period, or period - exposure >= 230 us
DEFAULT_GUARD_US = 60.0                 # timing margin kept from each pattern boundary
DEFAULT_LOOPS = 3
DEFAULT_CAMERA_EXPOSURE_US = 1000.0
MAX_CAMERA_EXPOSURE_US = 3000.0
EXTRA_LOOP_FRACTION = 0.05              # record this much more than `loops` loops
TARGET_CLEAN_FRAMES = 4.0               # loops=None: record until ~4 clean frames per pattern
DEFAULT_STEPS = 4
DEFAULT_PERIODS = (8, 80, 1280)
# the fast-scan pack's defaults (tools/make_flash_pack.py FASTSCAN_*; a test keeps them equal).
# The pack's manifest.json overrides them.
FASTSCAN_MODE = "flash_fastscan"
FASTSCAN_PERIOD_US = 10000.0            # 1-bit, exposure == period, internal timer, repeat
FASTSCAN_CAMERA_EXPOSURE_US = 3000.0
FASTSCAN_GUARD_US = 250.0

# ------------------------------------------------------------ sorting
STATIC_RATIO = 1.5                      # median frame distance / noise floor below this: static
STILL_RATIO = 1.5                       # adjacent frames < this x the noise pair RMS: same image
LIT_MIN_DN = 8.0                        # a pixel must change this much over the recording
PDM_MAX_RATIO = 0.5                     # folded cost at the loop period / median over the search
MAX_INCONSISTENT_FRACTION = 0.10
MAX_DROP_FRACTION = 0.05
ANCHOR_MAX_RMS = 0.15                   # layout relations misfit (white = 1 units) to accept
ANCHOR_MIN_SEPARATION = 2.0             # next-best rotation misfit / best
MARKER_LEVEL_TOL = 0.15
FRINGE_MEAN_TOL = 0.15
ORIENT_MIN_RATIO = 2.0                  # a 'y' sync marker: y structure / the next slot's
ORIENT_BLUR_PX = 1.5                    # camera px, before the orientation gradients
PHASE_SNR_MIN = 5.0                     # phase slope measured well enough to compare groups
VARIANT_ZONE_RATIO = 0.8                # decode-white lit area / sync-white lit area below: zone
FULL_WHITE_FRACTION = 0.8               # relations only where white >= this x SYNC A white
# zone variant decode: keep only px whose decode white (zone only) reaches this share of
# the full-field SYNC A white. Below it the px sits on the zone's defocused edge, where the
# fringes are clipped by the mask: mock e2e, ratio < 0.5 -> errors up to 17 columns (a
# wrong fringe order), 0.8-0.9 -> p95 0.18 col, >= 0.95 -> p95 0.04 col.
ZONE_EDGE_MIN = 0.9
ZONE_EDGE_BLUR_PX = 1.0                 # camera px, on both images before the ratio (noise)


class SequenceError(ValueError):
    """The frames do not show the expected flash loop playing (not playing,
    incomplete, out of order, too short). Nothing was decoded. `report`: what
    the sorter measured before it refused (json-safe)."""
    report: dict = {}


# ================================================================ layout

@dataclass
class FlashLayout:
    """The pattern loop in PLAY order (what the Pattern Sequence tab plays).

    entries: dicts {name, role, basis_index, period_px, step, level, ...}
      role 'white' / 'black' (the decode floods; basis_index 0 / 1),
      'marker' (a sync marker, not decoded: `level` = its expected mean in
      white = 1 units, None = not checked; `orient` 'y' = the only pattern
      structured along camera y; `sync` its label A / B), or
      'fringe' (basis_index in MultiFreqBasis order, its period_px and step;
      `group_level` = what the group's steps average to, 0.5 for a cosine,
      the profile mean for the fast-scan fine level).
    prefilter_px: camera-px Gaussian applied to the non-finest fringe groups
    before the decode (the pack's decode.prefilter; None = none).
    fine_prefilter_px: optional camera-px Gaussian on the finest group only
    (onebit.prefilter_frames fine_px; the pack's
    decode.prefilter.fine_sigma_camera_px_recommended; None = off, the default).
    """
    entries: list
    steps: int = DEFAULT_STEPS
    periods: list = field(default_factory=lambda: list(DEFAULT_PERIODS))
    axis: str = "x"
    diamond_parity: Optional[int] = None
    pattern_exposure_us: Optional[float] = None
    pattern_period_us: Optional[float] = None
    source: str = "default"
    mode: str = "synthetic"
    variant: Optional[str] = None
    prefilter_px: Optional[float] = None
    guard_us: Optional[float] = None
    camera_exposure_us: Optional[float] = None
    fine_prefilter_px: Optional[float] = None

    @property
    def n(self) -> int:
        return len(self.entries)

    def index(self, role: str) -> list:
        return [i for i, e in enumerate(self.entries) if e.get("role") == role]

    def names(self) -> list:
        return [e["name"] for e in self.entries]

    def n_basis(self) -> int:
        return 2 + len(self.periods) * self.steps

    def basis_slots(self) -> list:
        """Layout index of every MultiFreqBasis frame (white, black, fringes)."""
        by = {e.get("basis_index"): i for i, e in enumerate(self.entries)
              if e.get("basis_index") is not None}
        missing = [b for b in range(self.n_basis()) if b not in by]
        if missing:
            raise ValueError(f"flash layout has no pattern for basis frame(s) {missing}")
        return [by[b] for b in range(self.n_basis())]

    def fringe_groups(self) -> list:
        """[(period_px, [layout index per step 0..N-1])], fine -> coarse."""
        out = []
        for T in self.periods:
            idx = {e.get("step"): i for i, e in enumerate(self.entries)
                   if e.get("role") == "fringe" and float(e.get("period_px", -1)) == float(T)}
            if sorted(idx) == list(range(self.steps)):
                out.append((float(T), [idx[k] for k in range(self.steps)]))
        return out

    def group_level(self, idx: list) -> float:
        return float(np.mean([float(self.entries[i].get("group_level", 0.5)) for i in idx]))

    def validate(self) -> "FlashLayout":
        if self.n < 3:
            raise ValueError(f"a flash loop needs >= 3 patterns, got {self.n}")
        if len(self.index("white")) != 1 or len(self.index("black")) != 1:
            raise ValueError("a flash layout needs exactly one 'white' and one 'black' pattern "
                             "(the sync anchor and the decode's polarity pair)")
        if self.axis != "x":
            raise ValueError("soft sync decodes the column axis (x) only")
        for e in self.entries:
            if int(e.get("bitdepth", 1)) != 1:
                raise ValueError(
                    f"pattern {e.get('name')!r} is {e.get('bitdepth')}-bit. Soft sync needs 1-bit "
                    "patterns: an 8-bit pattern is a PWM bit-plane sequence inside its exposure, "
                    "and a free-running camera catches a random slice of it. Use a dithered "
                    "1-bit pack")
            if e.get("role") not in ("white", "black", "marker", "fringe"):
                raise ValueError(f"pattern {e.get('name')!r}: unknown role {e.get('role')!r}")
        self.basis_slots()
        if len(self.fringe_groups()) != len(self.periods):
            raise ValueError(f"flash layout: every period {self.periods} needs steps "
                             f"0..{self.steps - 1}")
        if len(set(self.names())) != self.n:
            raise ValueError("flash layout: pattern names must be unique")
        return self

    def to_json(self) -> dict:
        return {"entries": [dict(e) for e in self.entries], "steps": self.steps,
                "periods": [float(p) for p in self.periods], "axis": self.axis,
                "diamond_parity": self.diamond_parity,
                "pattern_exposure_us": self.pattern_exposure_us,
                "pattern_period_us": self.pattern_period_us, "source": self.source,
                "mode": self.mode, "variant": self.variant, "prefilter_px": self.prefilter_px,
                "guard_us": self.guard_us, "camera_exposure_us": self.camera_exposure_us,
                "fine_prefilter_px": self.fine_prefilter_px}

    @classmethod
    def from_json(cls, d: dict) -> "FlashLayout":
        return cls(entries=[dict(e) for e in d["entries"]], steps=int(d["steps"]),
                   periods=[float(p) for p in d["periods"]], axis=d.get("axis", "x"),
                   diamond_parity=d.get("diamond_parity"),
                   pattern_exposure_us=d.get("pattern_exposure_us"),
                   pattern_period_us=d.get("pattern_period_us"),
                   source=d.get("source", "json"), mode=d.get("mode", "synthetic"),
                   variant=d.get("variant"), prefilter_px=d.get("prefilter_px"),
                   guard_us=d.get("guard_us"),
                   camera_exposure_us=d.get("camera_exposure_us"),
                   fine_prefilter_px=d.get("fine_prefilter_px")).validate()


def fastscan_layout(variant: Optional[str] = None, *,
                    pattern_period_us: Optional[float] = FASTSCAN_PERIOD_US,
                    pattern_exposure_us: Optional[float] = None,
                    camera_exposure_us: Optional[float] = FASTSCAN_CAMERA_EXPOSURE_US,
                    guard_us: Optional[float] = FASTSCAN_GUARD_US,
                    diamond_parity: Optional[int] = None, source: str = "onebit") -> FlashLayout:
    """The 16-slot 1-bit fast-scan loop of engine/bases/onebit.py, in play order:
    sync_white (SYNC A, level 1), sync_code (SYNC B, horizontal bars: the only slot
    structured along camera y), white, black, fine T=8 x4, mid T=80 x4, coarse
    T=1280 x4. Levels are what the pack's patterns average to (the fine profile
    mean for the fine group, 0.5 for the Floyd-Steinberg cosines). The sync_code
    level is not checked (a small zone can sit on one bar); its orientation is."""
    from ..bases import onebit as ob
    fine_level = float(ob.fine_profile().mean())
    ents = []
    for s, name in enumerate(ob.SLOTS):
        m = ob.slot_meaning(s)
        role = m["role"]
        if role == "sync_white":
            e = {"name": name, "role": "marker", "basis_index": None, "level": 1.0, "sync": "A"}
        elif role == "sync_code":
            e = {"name": name, "role": "marker", "basis_index": None, "level": None,
                 "orient": "y", "sync": "B"}
        elif role in ("white", "black"):
            e = {"name": name, "role": role, "basis_index": int(m["decode_index"])}
        else:
            e = {"name": name, "role": "fringe", "basis_index": int(m["decode_index"]),
                 "period_px": float(m["period_px"]), "step": int(m["step"]),
                 "group_level": fine_level if name.startswith("fine") else 0.5}
        e["slot"] = s
        ents.append(e)
    X = pattern_exposure_us if pattern_exposure_us else pattern_period_us
    return FlashLayout(ents, ob.STEPS, [float(p) for p in ob.PERIODS], "x", diamond_parity,
                       float(X) if X else None,
                       float(pattern_period_us) if pattern_period_us else None, source,
                       FASTSCAN_MODE, variant, float(ob.PREFILTER_PX),
                       float(guard_us) if guard_us else None,
                       float(camera_exposure_us) if camera_exposure_us else None).validate()


def default_layout(variant: Optional[str] = None, **kw) -> FlashLayout:
    """What the engine assumes without a pack: the fast-scan pack (fastscan_layout)."""
    return fastscan_layout(variant, **kw)


def synthetic_layout(steps: int = DEFAULT_STEPS, periods=DEFAULT_PERIODS, *,
                     diamond_parity: Optional[int] = None, marker: bool = True,
                     pattern_exposure_us: Optional[float] = None,
                     pattern_period_us: Optional[float] = None) -> FlashLayout:
    """A generic loop for tests / other period sets: white, black, marker (a 50 %
    Bayer checker: uniform half level after defocus), then the column-axis multifreq
    fringes fine -> coarse, steps in order, Bayer-dithered (synthetic_patterns)."""
    ents = [{"name": "white", "role": "white", "basis_index": 0},
            {"name": "black", "role": "black", "basis_index": 1}]
    if marker:
        ents.append({"name": "marker_half", "role": "marker", "basis_index": None, "level": 0.5})
    for g, T in enumerate(periods):
        for k in range(int(steps)):
            ents.append({"name": f"x_T{float(T):g}_s{k}", "role": "fringe",
                         "basis_index": 2 + g * int(steps) + k, "period_px": float(T), "step": k})
    return FlashLayout(ents, int(steps), [float(p) for p in periods], "x", diamond_parity,
                       pattern_exposure_us, pattern_period_us, "synthetic").validate()


def _pack_variants(manifest: dict) -> list:
    v = manifest.get("variants")
    if isinstance(v, dict) and v:
        return list(v)
    found = []
    for e in manifest.get("sequence") or []:
        if e.get("variant") and e["variant"] not in found:
            found.append(e["variant"])
    return found


def _plane_label(bit: int) -> str:
    """Bit position -> DLPC350 plane name (tools/make_flash_pack.plane_label: bits 0..7 =
    G0..G7, 8..15 = R0..R7, 16..23 = B0..B7)."""
    return "GRB"[int(bit) // 8] + str(int(bit) % 8)


def fastscan_manifest_problems(manifest: dict, variant: Optional[str] = None) -> list:
    """Every way a fast-scan pack's manifest differs from the 16-slot loop this engine
    sorts and decodes (engine/bases/onebit.py): pattern version and dither design (fine
    levels + row order, Floyd-Steinberg mid / coarse periods, sync bars), decode order and
    basis parameters, and per slot of the variant: slot k on bit k = plane G0..G7, R0..R7,
    1-bit, one image index, exposure == period as in `timing`, internal trigger, repeat.
    [] = the capture expects exactly the packed set."""
    from ..bases import onebit as ob
    out = []
    des = manifest.get("design") or {}
    if des.get("pattern_version") != ob.PATTERN_VERSION:
        out.append(f"pattern_version {des.get('pattern_version')!r}, this engine makes "
                   f"{ob.PATTERN_VERSION}")
    fine = des.get("fine") or {}
    if list(fine.get("levels_q4") or []) != list(ob.FINE_LEVELS_Q4):
        out.append(f"fine dither levels {fine.get('levels_q4')} != {list(ob.FINE_LEVELS_Q4)}")
    if list(fine.get("row_order_q4") or []) != list(ob.ROW_ORDER_Q4):
        out.append(f"fine dither row order {fine.get('row_order_q4')} != {list(ob.ROW_ORDER_Q4)}")
    mc = des.get("mid_coarse") or {}
    if [float(p) for p in (mc.get("periods_px") or [])] != [float(p) for p in ob.PERIODS[1:]]:
        out.append(f"mid / coarse periods {mc.get('periods_px')} != {list(ob.PERIODS[1:])}")
    if (des.get("sync_code") or {}).get("bar_rows") != ob.SYNC_BAR_ROWS:
        out.append(f"sync bars {(des.get('sync_code') or {}).get('bar_rows')!r} rows != "
                   f"{ob.SYNC_BAR_ROWS}")
    dec = manifest.get("decode") or {}
    if list(dec.get("order") or []) != [ob.SLOTS[s] for s in ob.DECODE_SLOTS]:
        out.append("decode order differs from onebit.DECODE_SLOTS")
    par = dec.get("params") or {}
    if (par.get("steps") != ob.STEPS or [float(p) for p in (par.get("periods_px") or [])]
            != [float(p) for p in ob.PERIODS] or par.get("axis") != "x"):
        out.append(f"decode params {par} != steps {ob.STEPS}, periods {list(ob.PERIODS)}, axis x")
    t = manifest.get("timing") or {}
    X, P = t.get("pattern_exposure_us"), t.get("pattern_period_us")
    if X is None or P is None:
        out.append("timing has no pattern_exposure_us / pattern_period_us")
    else:
        out += [f"timing: {e}" for e in _dlp_1bit_errors(float(X), float(P))]
    if t.get("trigger", "internal") != "internal" or t.get("repeat", True) is not True:
        out.append(f"timing: trigger {t.get('trigger')!r}, repeat {t.get('repeat')!r} (the loop "
                   "plays on the internal timer, on repeat)")
    variants = _pack_variants(manifest)
    v = variant or (variants[0] if variants else None)
    seq = sorted([e for e in manifest.get("sequence") or [] if v is None or e.get("variant") == v],
                 key=lambda e: e.get("seq", 0))
    if len(seq) != len(ob.SLOTS):
        out.append(f"variant {v!r}: {len(seq)} sequence entries, the loop has {len(ob.SLOTS)}")
        return out
    vinfo = (manifest.get("variants") or {}).get(v) or {}
    idx = {e.get("image_index") for e in seq}
    if len(idx) != 1 or (vinfo and vinfo.get("image_index") not in idx):
        out.append(f"variant {v!r}: image_index {sorted(idx, key=str)} (one image per variant, "
                   f"{vinfo.get('image_index')!r} in `variants`)")
    if not any(im.get("index") in idx and im.get("sha256") for im in manifest.get("images") or []):
        out.append(f"variant {v!r}: its flash image is not listed (with sha256) in `images`")
    for k, e in enumerate(seq):
        want = _plane_label(k)
        bad = []
        if e.get("slot") != k:
            bad.append(f"slot {e.get('slot')!r}")
        if e.get("bit_position") != k:
            bad.append(f"bit_position {e.get('bit_position')!r}")
        if list(e.get("planes") or []) != [want]:
            bad.append(f"planes {e.get('planes')!r}")
        if int(e.get("bitdepth", 0)) != 1:
            bad.append(f"bitdepth {e.get('bitdepth')!r}")
        if X is not None and (e.get("exposure_us", X) != X or e.get("period_us", P) != P):
            bad.append(f"exposure/period {e.get('exposure_us')}/{e.get('period_us')} us")
        if bad:
            out.append(f"sequence entry {k} ({e.get('name')!r}, expected slot {k} on {want}): "
                       + ", ".join(bad))
    return out


def layout_from_manifest(manifest: dict, variant: Optional[str] = None) -> FlashLayout:
    """A flash pack's manifest.json -> FlashLayout (play order = `seq`).

    Fast-scan packs (mode 'flash_fastscan', tools/make_flash_pack.py --fastscan):
    every variant (full / zone) plays the same 16 slots on its own flash image, so
    the layout is onebit's. The slot names AND the pack contract are checked against
    it (fastscan_manifest_problems: pattern version, dither design, slot k on bit k =
    G0..G7 / R0..R7, decode order, legal 1-bit timing): the capture expects exactly
    the packed set, or refuses before the camera opens. Timing from
    manifest['timing'] (pattern_exposure_us / pattern_period_us, camera exposure
    and guard), prefilter from manifest['decode']['prefilter'], diamond_parity from
    the top level. `variant` names the image the operator plays (None: unknown,
    the sorter reports which one the frames show).

    Other packs: sequence[] {seq, name, role, bitdepth, basis_index, period_px?,
    step?, level?} with roles white / black / fringe / marker (sync_white and
    sync_code are read as markers); steps / periods from manifest['softsync'] or
    manifest['decode'][axis]['params']; timing from manifest['softsync'] or
    manifest['timing'] (per_bitdepth['1'] or uniform). 8-bit patterns are refused
    (make_flash_pack's HybridBasis packs are for the hardware-trigger path)."""
    variants = _pack_variants(manifest)
    if variant is not None and variants and variant not in variants:
        raise ValueError(f"the pack has variants {variants}, not {variant!r}")
    seq = sorted(manifest.get("sequence") or [], key=lambda e: e.get("seq", 0))
    if variants:
        seq = [e for e in seq if e.get("variant") == (variant or variants[0])]
    if not seq:
        raise ValueError("pack manifest has no sequence")
    for e in seq:
        if int(e.get("bitdepth", 1)) != 1:
            raise ValueError(
                f"pack pattern {e.get('name')!r} is {e.get('bitdepth')}-bit: soft sync needs a "
                "1-bit (dithered) pack. 8-bit packs (make_flash_pack) are for the hardware-trigger "
                "path (docs/FLASH_PATTERN_MODE.md)")
    if manifest.get("mode") == FASTSCAN_MODE:
        from ..bases import onebit as ob
        prefix = f"{variant or (variants[0] if variants else '')}_"
        names = [str(e.get("name", "")) for e in seq]
        names = [n[len(prefix):] if prefix != "_" and n.startswith(prefix) else n for n in names]
        if names != list(ob.SLOTS):
            raise ValueError(f"pack slots {names} are not the fast-scan slots {list(ob.SLOTS)} of "
                             "engine/bases/onebit.py (rebuild the pack with this engine)")
        probs = []
        for v in ([variant] if variant else (variants or [None])):   # every image it may play
            probs += [p for p in fastscan_manifest_problems(manifest, v) if p not in probs]
        if probs:
            raise ValueError("the pack is not the 16-slot loop this engine sorts and decodes ("
                             + "; ".join(probs[:6]) + (f"; {len(probs) - 6} more" if len(probs) > 6
                                                       else "")
                             + "): rebuild it with tools/make_flash_pack.py --fastscan and load "
                               "the new images")
        t = manifest.get("timing") or {}
        cam = t.get("camera") or {}
        pre = ((manifest.get("decode") or {}).get("prefilter") or {})
        lay = fastscan_layout(variant, pattern_period_us=t.get("pattern_period_us"),
                              pattern_exposure_us=t.get("pattern_exposure_us"),
                              camera_exposure_us=cam.get("exposure_us"),
                              guard_us=cam.get("guard_us"),
                              diamond_parity=manifest.get("diamond_parity"),
                              source=str(manifest.get("name") or "pack"))
        if pre.get("sigma_camera_px") is not None:
            lay.prefilter_px = float(pre["sigma_camera_px"])
        return lay
    axes = manifest.get("axes") or ["x"]
    axis = axes[0] if len(axes) == 1 else None
    if axis != "x":
        raise ValueError(f"soft sync decodes the column axis only; this pack codes axes {axes}")
    ss = manifest.get("softsync") or {}
    params = (((manifest.get("decode") or {}).get(axis) or {}).get("params")
              or (manifest.get("decode") or {}).get("params") or {})
    steps = ss.get("steps") or params.get("steps")
    periods = ss.get("periods_px") or ss.get("periods") or params.get("periods_px")
    ents = []
    for e in seq:
        role = e.get("role", "")
        ent = {"name": e.get("name", f"p{len(ents)}"), "role": role,
               "basis_index": e.get("basis_index"), "bitdepth": 1}
        if role == "fringe":
            T, k = e.get("period_px"), e.get("step")
            if (T is None or k is None) and e.get("basis_index") is not None and steps and periods:
                b = int(e["basis_index"]) - 2
                T, k = periods[b // int(steps)], b % int(steps)
            if T is None or k is None:
                raise ValueError(f"pack fringe {ent['name']!r} has no period_px / step")
            ent.update(period_px=float(T), step=int(k))
            if e.get("group_level") is not None:
                ent["group_level"] = float(e["group_level"])
        elif role in ("marker", "sync_white", "sync_code"):
            ent.update(role="marker", basis_index=None)
            if role == "sync_white":
                ent.update(level=1.0, sync="A")
            elif role == "sync_code":
                ent.update(level=None, orient="y", sync="B")
            elif e.get("level") is not None:
                ent["level"] = float(e["level"])
        elif role not in ("white", "black"):
            raise ValueError(f"pack pattern {ent['name']!r}: unknown role {role!r} (white, black, "
                             "marker, sync_white, sync_code, fringe)")
        ents.append(ent)
    fr = [e for e in ents if e["role"] == "fringe"]
    if not steps:
        steps = 1 + max(e["step"] for e in fr) if fr else DEFAULT_STEPS
    if not periods:
        periods = sorted({e["period_px"] for e in fr})
    t = manifest.get("timing") or {}
    per = ((t.get("per_bitdepth") or {}).get("1") or t.get("uniform") or {})
    exp = ss.get("pattern_exposure_us") or per.get("exposure_us") or t.get("pattern_exposure_us")
    prd = ss.get("pattern_period_us") or per.get("period_us") or t.get("pattern_period_us")
    parity = ss.get("diamond_parity", manifest.get("diamond_parity"))
    return FlashLayout(ents, int(steps), [float(p) for p in periods], "x", parity,
                       float(exp) if exp else None, float(prd) if prd else None,
                       str(manifest.get("name") or "pack"), str(manifest.get("mode") or "pack"),
                       variant).validate()


def load_pack_layout(pack, variant: Optional[str] = None) -> tuple:
    """(FlashLayout, manifest, pack dir or None) from a pack dir / manifest path / dict."""
    if isinstance(pack, dict):
        return layout_from_manifest(pack, variant), pack, None
    p = Path(pack)
    f = p / "manifest.json" if p.is_dir() else p
    if not f.is_file():
        raise ValueError(f"flash pack {pack}: no manifest.json")
    m = json.loads(f.read_text(encoding="utf-8"))
    return layout_from_manifest(m, variant), m, f.parent


def pack_zone_mask(pack_dir, manifest: Optional[dict] = None) -> Optional[np.ndarray]:
    """The zone variant's DMD mask (zone_mask.png of a fast-scan pack), or None. With the
    pack's manifest, the file must match the sha256 recorded when the pack was made."""
    import cv2
    import hashlib
    if pack_dir is None:
        return None
    zb = ((((manifest or {}).get("variants") or {}).get("zone") or {}).get("zone") or {})
    f = Path(pack_dir) / (zb.get("file") or "zone_mask.png")
    if not f.is_file():
        return None
    raw = f.read_bytes()
    if zb.get("sha256") and hashlib.sha256(raw).hexdigest() != zb["sha256"]:
        raise ValueError(f"{f}: sha256 differs from the pack manifest (changed since the pack "
                         "was made)")
    m = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
    return None if m is None else m > 0


# ================================================================ patterns

_BAYER4 = np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], np.float64)


def ordered_dither(duty: np.ndarray) -> np.ndarray:
    """duty 0..1 (H, W) -> 1-bit 0/255 by 4x4 Bayer ordered dithering (DMD index space)."""
    H, W = duty.shape
    thr = (np.tile(_BAYER4, (H // 4 + 1, W // 4 + 1))[:H, :W] + 0.5) / 16.0
    return np.where(duty > thr, 255, 0).astype(np.uint8)


def synthetic_patterns(layout: FlashLayout, dmd_shape=(1140, 912)) -> list:
    """1-bit DMD images for a layout, in play order: floods, the marker as a checker
    of its level, fringes as Bayer-dithered duty 0.5 (1 + cos(2 pi pos / T + 2 pi k / N))
    in MultiFreqBasis coordinates (native columns, diamond parity applied).
    A stand-in for a pattern designer's pack (mock / offline use)."""
    from ..bases.multifreq import MultiFreqBasis
    basis = MultiFreqBasis(steps=layout.steps, periods_px=layout.periods, axis="x",
                           diamond_parity=layout.diamond_parity)
    pos = basis._coord("x", dmd_shape)
    out = []
    for e in layout.entries:
        r = e["role"]
        if r == "white":
            out.append(np.full(dmd_shape, 255, np.uint8))
        elif r == "black":
            out.append(np.zeros(dmd_shape, np.uint8))
        elif r == "marker":
            if e.get("orient") == "y":
                out.append(np.repeat(((np.arange(dmd_shape[0]) // 16) % 2 == 0)[:, None],
                                     dmd_shape[1], 1).astype(np.uint8) * 255)
            else:
                lv = e.get("level")
                out.append(ordered_dither(np.full(dmd_shape, 0.5 if lv is None else float(lv))))
        else:
            ph = 2 * np.pi * pos / float(e["period_px"]) + 2 * np.pi * int(e["step"]) / layout.steps
            out.append(ordered_dither(0.5 * (1 + np.cos(ph))))
    return out


def layout_patterns(layout: FlashLayout, zone_mask: Optional[np.ndarray] = None) -> list:
    """The DMD images the loop plays (layout order): the fast-scan pack's own patterns
    (onebit.fastscan_patterns; zone_mask = the zone variant) or synthetic ones."""
    if layout.mode == FASTSCAN_MODE:
        from ..bases import onebit as ob
        pats = ob.fastscan_patterns(zone_mask)
        if [p["name"] for p in pats] != layout.names():
            raise ValueError("layout names differ from onebit.fastscan_patterns")
        return [p["image"] for p in pats]
    if zone_mask is not None:
        z = np.asarray(zone_mask, bool)
        return [np.where(z, p, 0).astype(np.uint8) if e["role"] != "marker" else p
                for p, e in zip(synthetic_patterns(layout), layout.entries)]
    return synthetic_patterns(layout)


# ================================================================ planning

def estimate_max_fps(roi_w: int, roi_h: int, exposure_us: float = 0.0) -> dict:
    """Blackfly S BFS-U3-16S2M frame-rate estimate for an ROI (free run, exposure
    overlapped with readout): min(sensor readout of roi_h rows, USB3 bytes, exposure).
    An estimate for planning: on the bench the camera's AcquisitionFrameRate maximum,
    read after the ROI and exposure are set, is what runs."""
    row_us = 1e6 / (SENSOR_FULL_FPS * (SENSOR_SHAPE[0] + SENSOR_OVERHEAD_ROWS))
    sensor = 1e6 / (row_us * (int(roi_h) + SENSOR_OVERHEAD_ROWS))
    usb = USB_BYTES_PER_S / float(int(roi_w) * int(roi_h))
    expo = 1e6 / (float(exposure_us) + EXPOSURE_OVERHEAD_US) if exposure_us else math.inf
    fps = min(sensor, usb, expo)
    limit = "sensor readout" if fps == sensor else ("USB3 bandwidth" if fps == usb else "exposure")
    return {"fps": fps, "sensor_fps": sensor, "usb_fps": usb, "exposure_fps": expo, "limit": limit}


def align_roi(bbox, *, sensor=SENSOR_SHAPE, step: Optional[dict] = None, margin: int = 16) -> tuple:
    """Camera bbox (x0, y0, x1, y1), x1 / y1 exclusive -> an ROI (x, y, w, h) the camera
    accepts: grown by `margin`, offsets rounded down and sizes up to the node increments,
    kept on the sensor."""
    st = {**ROI_STEP, **(step or {})}
    H, W = int(sensor[0]), int(sensor[1])
    x0, y0, x1, y1 = (int(v) for v in bbox)
    x0, y0 = max(0, x0 - margin), max(0, y0 - margin)
    x1, y1 = min(W, x1 + margin), min(H, y1 + margin)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"ROI {bbox} is empty or off the {W}x{H} sensor")
    x, y = (x0 // st["x"]) * st["x"], (y0 // st["y"]) * st["y"]
    w = -(-(x1 - x) // st["w"]) * st["w"]
    h = -(-(y1 - y) // st["h"]) * st["h"]
    w = min(max(w, -(-ROI_MIN[1] // st["w"]) * st["w"]), (W // st["w"]) * st["w"])
    h = min(max(h, -(-ROI_MIN[0] // st["h"]) * st["h"]), (H // st["h"]) * st["h"])
    if x + w > W:
        x = ((W - w) // st["x"]) * st["x"]
    if y + h > H:
        y = ((H - h) // st["y"]) * st["y"]
    return (int(x), int(y), int(w), int(h))


def roi_from_mask(mask: np.ndarray, **kw) -> tuple:
    v, u = np.nonzero(np.asarray(mask, bool))
    if not len(u):
        raise ValueError("the camera zone mask is empty")
    return align_roi((u.min(), v.min(), u.max() + 1, v.max() + 1),
                     sensor=np.asarray(mask).shape, **kw)


def _dlp_1bit_errors(exposure_us: float, period_us: float) -> list:
    e, p = int(round(exposure_us)), int(round(period_us))
    errs = []
    if e < MIN_1BIT_EXPOSURE_US:
        errs.append(f"1-bit exposure {e} us < minimum {MIN_1BIT_EXPOSURE_US} us")
    if not (e == p or e <= p - PERIOD_DIFFERENCE_MIN_US):
        errs.append(f"exposure {e} us / period {p} us: must be equal, or period - exposure "
                    f">= {PERIOD_DIFFERENCE_MIN_US} us")
    return errs


def plan_timing(n_patterns: int, roi=None, *, exposure_us: Optional[float] = None,
                fps: Optional[float] = None, pattern_exposure_us: Optional[float] = None,
                pattern_period_us: Optional[float] = None, loops: Optional[int] = DEFAULT_LOOPS,
                guard_us: Optional[float] = DEFAULT_GUARD_US, fps_max: Optional[float] = None,
                target_clean: float = TARGET_CLEAN_FRAMES) -> dict:
    """Camera + DLPC350 timing for soft sync.

    A frame is CLEAN when its exposure [t, t + E] lies inside one pattern exposure X,
    `guard_us` away from both ends. A free-running camera at F fps then gets at least
    one clean frame of every pattern in every loop when X >= E + 1/F + 2 guard.

    Pattern timing given (a pack / what the GUI plays): pick the camera exposure E
    (default min(2 X / 3, X - 1/F - 2 guard, 3 ms)) and F (default the ROI maximum).
    Not given: design it: E default 1 ms, F the ROI maximum, X = period = E + 1/F +
    2 guard (+5 %), rounded up to 10 us (exposure == period: no dark gap).
    loops None: the fewest loops (>= 2) that give ~target_clean clean frames per
    pattern (they are averaged). -> dict (all times in us / ms, json-safe)."""
    M = int(n_patterns)
    if M < 3:
        raise ValueError(f"n_patterns must be >= 3, got {n_patterns}")
    if loops is not None and int(loops) < 2:
        raise ValueError(f"loops must be >= 2 (the loop period is found from repeats), got {loops}")
    x, y, w, h = roi if roi is not None else (0, 0, SENSOR_SHAPE[1], SENSOR_SHAPE[0])
    g = float(DEFAULT_GUARD_US if guard_us is None else guard_us)
    notes = []

    def fmax_for(e):
        est = estimate_max_fps(w, h, e)
        cap = FPS_HEADROOM * est["fps"]
        if fps_max is not None:
            cap = min(cap, float(fps_max))
        return cap, est

    X = float(pattern_exposure_us) if pattern_exposure_us else None
    P = float(pattern_period_us) if pattern_period_us else None
    if X is None and P is not None:
        X = P
    if X is not None and P is None:
        P = X
    if X is not None:
        f0, _ = fmax_for(0.0)
        F = min(float(fps), f0) if fps else f0
        if exposure_us:
            E = float(exposure_us)
        else:
            E = min(2.0 * X / 3.0, X - 1e6 / F - 2 * g, MAX_CAMERA_EXPOSURE_US)
        if E <= 0:
            raise ValueError(f"pattern exposure {X:g} us leaves no camera exposure at {F:.0f} "
                             f"fps (needs X > 1/F + 2 x {g:g} us guard): lengthen the pattern "
                             "exposure or raise the frame rate (smaller ROI)")
        f1, est = fmax_for(E)
        F = min(F, f1)
        if not exposure_us:
            E = min(E, X - 1e6 / F - 2 * g)
        designed = False
    else:
        E = float(exposure_us or DEFAULT_CAMERA_EXPOSURE_US)
        f1, est = fmax_for(E)
        F = min(float(fps), f1) if fps else f1
        X = math.ceil((E + 1e6 / F + 2 * g) * 1.05 / 10.0) * 10.0
        X = max(X, float(MIN_1BIT_EXPOSURE_US))
        P = X
        designed = True
    if E <= 0:
        raise ValueError(f"pattern exposure {X:g} us leaves no camera exposure at {F:.0f} fps "
                         f"(needs X > 1/F + 2 x {g:g} us guard): lengthen the pattern exposure "
                         "or raise the frame rate (smaller ROI)")
    frame_us = 1e6 / F
    window = X - E - 2 * g                          # start times that give a clean frame
    clean_per_loop = window / frame_us              # expected clean frames per pattern per loop
    if window < frame_us:
        notes.append(f"pattern exposure {X:g} us < camera exposure {E:g} + frame period "
                     f"{frame_us:.0f} + 2 guard {g:g} us: some loops will have no clean frame "
                     "of some patterns (expect gaps; record more loops or lengthen X)")
    loop_us = M * P
    auto = loops is None
    if auto:
        loops = max(2, int(math.ceil(float(target_clean) / max(clean_per_loop, 1e-6) - 1e-9)))
        loops = min(loops, 12)
    n_frames = int(math.ceil((int(loops) + EXTRA_LOOP_FRACTION) * loop_us / frame_us)) + 4
    dlp_errors = _dlp_1bit_errors(X, P)
    if E > frame_us:
        notes.append(f"camera exposure {E:.0f} us > frame period {frame_us:.0f} us")
    per_loop = loop_us / frame_us
    frac = per_loop - math.floor(per_loop)
    if min(frac, 1 - frac) * int(loops) < 0.5:
        want = math.floor(per_loop - 1.0 / int(loops)) + 1.0 / int(loops)
        keeps = (X - E - 2 * g) >= loop_us / want or window < frame_us
        if not fps and per_loop > 2 and keeps:
            # the rate is ours to pick: slow down a little so successive loops sample
            # pattern phases 1 / loops of a frame apart (more distinct straddle mixtures)
            F = want * 1e6 / loop_us
            frame_us = 1e6 / F
            window = X - E - 2 * g
            clean_per_loop = window / frame_us
            n_frames = int(math.ceil((int(loops) + EXTRA_LOOP_FRACTION) * loop_us / frame_us)) + 4
            per_loop = want
        else:
            notes.append(f"{per_loop:.3f} camera frames per loop: every loop samples nearly the "
                         "same pattern phases (fewer distinct straddle mixtures for the boundary "
                         "fit); a slightly different fps spreads them")
    return {"roi": [int(x), int(y), int(w), int(h)], "n_patterns": M,
            "camera_exposure_us": round(E, 1), "fps": round(F, 2),
            "frame_period_us": round(frame_us, 2), "fps_limit": est["limit"],
            "fps_estimate": {k: round(v, 1) for k, v in est.items() if isinstance(v, float)
                             and math.isfinite(v)},
            "pattern_exposure_us": round(X, 1), "pattern_period_us": round(P, 1),
            "pattern_timing": "designed" if designed else "given",
            "guard_us": g, "loop_ms": round(loop_us / 1000, 3), "loops": int(loops),
            "loops_auto": auto, "frames_per_loop": round(per_loop, 3),
            "n_frames": n_frames, "capture_ms": round(n_frames * frame_us / 1000, 1),
            "clean_frames_per_pattern_per_loop": round(clean_per_loop, 2),
            "clean_frames_per_pattern": round(clean_per_loop * int(loops), 1),
            "min_clean_per_loop": int(max(0, math.floor(clean_per_loop))),
            "dlp_timing_errors": dlp_errors, "notes": notes}


# ================================================================ recording

@dataclass
class Recording:
    """N free-run frames of the camera ROI (uint8), oldest first."""
    frames: np.ndarray                  # (N, h, w)
    frame_ids: np.ndarray               # int64 camera frame ids (gaps = dropped frames)
    device_ts_ns: Optional[np.ndarray]  # int64 camera clock (None: unknown)
    host_ts_ns: np.ndarray              # int64 perf_counter_ns when the host got the frame
    roi: tuple                          # (x, y, w, h) camera px
    full_shape: tuple = SENSOR_SHAPE    # (H, W) of the full camera frame
    exposure_us: float = 0.0
    fps: float = 0.0
    meta: dict = field(default_factory=dict)

    def times_s(self) -> tuple:
        """(t seconds from the first frame, source): the camera's clock when it is
        increasing, else frame id / fps."""
        if self.device_ts_ns is not None and len(self.device_ts_ns) == len(self.frames):
            t = (np.asarray(self.device_ts_ns, np.int64) - int(self.device_ts_ns[0])) * 1e-9
            if len(t) < 2 or np.all(np.diff(t) > 0):
                return t, "device timestamps"
        if self.fps > 0:
            ids = np.asarray(self.frame_ids, np.int64)
            return (ids - ids[0]) / float(self.fps), "frame ids / fps"
        raise ValueError("recording has neither increasing device timestamps nor an fps")

    def save(self, path: Path) -> Path:
        rules.ensure_new_path(path)
        doc = {"roi": list(self.roi), "full_shape": list(self.full_shape),
               "exposure_us": self.exposure_us, "fps": self.fps, "meta": self.meta}
        arrays = {"frames": np.asarray(self.frames, np.uint8),
                  "frame_ids": np.asarray(self.frame_ids, np.int64),
                  "host_ts_ns": np.asarray(self.host_ts_ns, np.int64),
                  "meta_json": np.array(json.dumps(_plain(doc)))}
        if self.device_ts_ns is not None:
            arrays["device_ts_ns"] = np.asarray(self.device_ts_ns, np.int64)
        with open(path, "xb") as f:
            np.savez_compressed(f, **arrays)
        return Path(path)

    @classmethod
    def load(cls, path) -> "Recording":
        with np.load(Path(path), allow_pickle=False) as z:
            doc = json.loads(str(z["meta_json"]))
            return cls(frames=z["frames"], frame_ids=z["frame_ids"],
                       device_ts_ns=z["device_ts_ns"] if "device_ts_ns" in z.files else None,
                       host_ts_ns=z["host_ts_ns"], roi=tuple(doc["roi"]),
                       full_shape=tuple(doc["full_shape"]), exposure_us=float(doc["exposure_us"]),
                       fps=float(doc["fps"]), meta=doc.get("meta") or {})


class FlashFreeRunSync:
    """Record the running flash loop with a free-running camera. Camera only: the
    projector is never touched (a human plays the sequence in the GUI).

    camera: configure(roi=, exposure_us=, fps=, gain_db=, buffers=) -> applied dict,
    start(), next_frame(timeout_ms) -> (img, frame_id, device_ts_ns, host_ts_ns) or None
    (an incomplete frame, released), stop()."""

    def __init__(self, camera):
        rules.check_scan_path_usb_free()
        for m in ("configure", "start", "next_frame", "stop"):
            if not callable(getattr(camera, m, None)):
                raise TypeError(f"{type(camera).__name__} is not a free-run camera (needs {m})")
        self.camera = camera

    def record(self, n_frames: int, roi, exposure_us: float, fps: float,
               gain_db: Optional[float] = None, full_shape=SENSOR_SHAPE,
               meta: Optional[dict] = None, min_duration_s: Optional[float] = None) -> Recording:
        """n_frames at the configured rate; min_duration_s: record at least this long
        (more frames when the camera runs slower than `fps`)."""
        n = int(n_frames)
        if n < 8:
            raise ValueError(f"n_frames must be >= 8, got {n_frames}")
        x, y, w, h = (int(v) for v in roi)
        buffers = n + 16
        if min_duration_s:
            buffers = max(buffers, int(math.ceil(float(min_duration_s) * float(fps) * 1.2)) + 16)
        applied = self.camera.configure(roi=(x, y, w, h), exposure_us=float(exposure_us),
                                        fps=float(fps), gain_db=gain_db, buffers=buffers)
        run_fps = float(applied.get("fps") or fps)
        if min_duration_s:
            n = max(n, int(math.ceil(float(min_duration_s) * run_fps)) + 4)
        timeout = int(max(1000.0, 20e3 / run_fps))
        frames = np.empty((n, h, w), np.uint8)
        ids = np.empty(n, np.int64)
        dev = np.empty(n, np.int64)
        host = np.empty(n, np.int64)
        have_dev = True
        k = incomplete = 0
        self.camera.start()
        try:
            while k < n:
                got = self.camera.next_frame(timeout)
                if got is None:
                    incomplete += 1
                    if incomplete > max(8, n // 10):
                        raise RuntimeError(f"{incomplete} incomplete frames: USB3 bandwidth / "
                                           "cable (lower the fps or the ROI)")
                    continue
                img, fid, dts, hts = got
                frames[k] = np.asarray(img, np.uint8).reshape(h, w)
                ids[k], host[k] = int(fid), int(hts)
                if dts is None:
                    have_dev = False
                else:
                    dev[k] = int(dts)
                k += 1
        finally:
            self.camera.stop()
        m = {"camera": {k2: v for k2, v in applied.items()}, "incomplete_frames": incomplete,
             **(meta or {})}
        return Recording(frames, ids, dev if have_dev else None, host, (x, y, w, h),
                         tuple(full_shape), float(applied.get("exposure_us") or exposure_us),
                         run_fps, m)


# ================================================================ sorting

@dataclass
class SortResult:
    images: list                # per layout entry (play order): float32 (h, w) mean of clean frames
    counts: list                # clean frames averaged per layout entry
    loops_seen: list            # loops with >= 1 clean frame, per layout entry
    labels: np.ndarray          # per frame: layout index, -1 = not used
    status: np.ndarray          # per frame: 'clean' | 'straddle' | 'gap' | 'inconsistent'
    loop_period_s: float
    pattern_period_s: float
    confidence: float
    report: dict
    roi: tuple = (0, 0, 0, 0)
    full_shape: tuple = SENSOR_SHAPE

    def frames_for_basis(self, layout: FlashLayout, full: bool = True) -> list:
        """MultiFreqBasis order (white, black, fringes): the ROI images embedded in full
        camera frames (0 outside, which the decode's contrast floor masks out)."""
        out = []
        x, y, w, h = self.roi
        for i in layout.basis_slots():
            img = self.images[i]
            if full:
                f = np.zeros(self.full_shape, np.float32)
                f[y:y + h, x:x + w] = img
                img = f
            out.append(img)
        return out


def _time_base(N, times_s, frame_ids, fps) -> tuple:
    if times_s is not None:
        t = np.asarray(times_s, np.float64)
        if t.shape != (N,):
            raise ValueError(f"times_s has shape {t.shape}, expected ({N},)")
        t = t - t[0]
        if np.all(np.diff(t) > 0):
            return t, "timestamps"
        if frame_ids is None or not fps:
            raise SequenceError("frame timestamps are not increasing and there are no frame "
                                "ids + fps to fall back on")
    if frame_ids is not None and fps:
        ids = np.asarray(frame_ids, np.int64)
        return (ids - ids[0]) / float(fps), "frame ids / fps"
    if fps:
        return np.arange(N) / float(fps), "frame index / fps"
    raise ValueError("sort_frames needs times_s, or fps (with frame_ids when frames were dropped)")


def _noise_sigma(flat, idx, rng, n_px: int = 20000, n_pairs: int = 5) -> float:
    """Camera noise (DN RMS per pixel) from the few most similar frame pairs (the same
    pattern in two loops, or twice in one slot): mean |difference| over every pixel
    = 1.128 sqrt(2) sigma for Gaussian noise. Works whether or not the window holds
    unlit pixels."""
    sub = idx if len(idx) <= n_px else rng.choice(idx, n_px, replace=False)
    D = _pairwise_rms(flat[:, sub].astype(np.float64))
    np.fill_diagonal(D, np.inf)
    iu = np.triu_indices(len(D), 1)
    best = np.argsort(D[iu])[:n_pairs]
    vals = []
    for b in best:
        a, c = int(iu[0][b]), int(iu[1][b])
        vals.append(float(np.mean(np.abs(flat[a].astype(np.int16) - flat[c].astype(np.int16)))))
    return float(np.median(vals)) / (1.1284 * math.sqrt(2.0))


def _features(frames, lit_mask, max_features, seed) -> tuple:
    """(features (N, n) float64, lit mask, feature px, noise sigma DN). Lit: px whose
    range over the recording clears the floor, 20 % of the p99 range, and the noise
    range of the camera (measured, so a dim loop next to unlit px still separates)."""
    N = len(frames)
    flat = frames.reshape(N, -1)
    lo = frames.min(axis=0).astype(np.float32)
    hi = frames.max(axis=0).astype(np.float32)
    rng_img = (hi - lo).reshape(-1)
    cand = (np.ones(rng_img.shape, bool) if lit_mask is None
            else np.asarray(lit_mask, bool).reshape(-1))
    if not cand.any():
        raise SequenceError("the zone mask does not overlap the recorded camera window")
    rng = np.random.default_rng(seed)
    thr = max(LIT_MIN_DN, 0.2 * float(np.percentile(rng_img[cand], 99)))
    sigma = _noise_sigma(flat, np.flatnonzero(cand), rng)
    noise_range = 2.0 * sigma * math.sqrt(2.0 * math.log(max(N, 2)))
    thr = max(thr, 1.5 * noise_range)
    lit = cand & (rng_img > thr)
    idx = np.flatnonzero(lit)
    if len(idx) < 64:
        raise SequenceError(
            f"no modulated light in the camera window ({len(idx)} px change by > {thr:.0f} DN over "
            "the recording): the pattern sequence is not playing, the zone is outside the ROI, or "
            "the exposure / LED current is too low")
    if len(idx) > max_features:
        if len(idx) > 200_000:
            idx = np.sort(rng.choice(idx, 200_000, replace=False))
        # zone variant: outside the zone only the two full-field sync slots light the
        # scene. Pixels lit in many frames (the zone) get half the features, so the
        # 14 zone-only slots are not drowned by pixels that see them all as dark.
        mid = ((lo + hi) / 2).reshape(-1)[idx]
        frac = (flat[:, idx] > mid[None, :]).mean(0)
        rich = frac >= 0.3
        nr = int(rich.sum())
        if 64 <= nr < 0.5 * len(idx):
            k = min(nr, max_features // 2)
            pick = np.r_[rng.choice(idx[rich], k, replace=False),
                         rng.choice(idx[~rich], max_features - k, replace=False)]
            idx = np.sort(pick)
        else:
            idx = np.sort(rng.choice(idx, max_features, replace=False))
    return flat[:, idx].astype(np.float64), lit.reshape(frames.shape[1:]), idx, sigma


def _pairwise_rms(X) -> np.ndarray:
    sq = (X * X).sum(1)
    D2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    return np.sqrt(np.maximum(D2, 0.0) / X.shape[1])


def _pca(X, q: int) -> np.ndarray:
    Xc = X - X.mean(0)
    C = Xc @ Xc.T
    w, V = np.linalg.eigh(C)
    q = min(q, len(w) - 1)
    top = np.argsort(w)[::-1][:q]
    return V[:, top] * np.sqrt(np.maximum(w[top], 0.0))


def _fold_cost(t, Y, T) -> float:
    o = np.argsort(np.mod(t, T), kind="stable")
    Z = Y[o]
    d = np.diff(Z, axis=0)
    e = Z[0] - Z[-1]
    return float((d * d).sum() + (e * e).sum())


def _find_loop_period(t, Y, lo, hi, frame_dt, duration) -> tuple:
    """Phase-dispersion minimisation: fold the frames at T, order by phase, sum the
    squared feature steps between phase neighbours; the loop period folds every
    loop onto the others (smallest cost). Geometric grid fine enough that the fold
    drifts < 1/6 frame over the recording, then a local refinement."""
    ratio = 1.0 + frame_dt / (3.0 * duration)
    n = int(math.ceil(math.log(hi / lo) / math.log(ratio))) + 1
    grid = lo * ratio ** np.arange(n)
    cost = np.array([_fold_cost(t, Y, T) for T in grid])
    b = int(np.argmin(cost))
    # the cost is piecewise constant (only the phase order matters): take the
    # centre of the minimal plateau, on a grid 10x finer
    fine = grid[b] * ratio ** np.linspace(-6, 6, 121)
    fc = np.array([_fold_cost(t, Y, T) for T in fine])
    i0 = int(np.argmin(fc))
    ok = fc <= fc[i0] * (1 + 1e-9) + 1e-9
    a_, b_ = i0, i0
    while a_ > 0 and ok[a_ - 1]:
        a_ -= 1
    while b_ < len(fc) - 1 and ok[b_ + 1]:
        b_ += 1
    T = float(np.sqrt(fine[a_] * fine[b_]))
    return T, float(fc.min()) / max(float(np.median(cost)), 1e-12), n


def _label(ph, b0, T, P, M, E, X, g) -> tuple:
    o = np.mod(ph - b0, T)
    s = np.floor(o / P).astype(np.int64)
    wpos = o - s * P
    valid = s < M
    s = np.where(valid, s, M)
    clean = valid & (wpos >= g) & (wpos + E <= X - g)
    return s, wpos, clean


def _within_cost(Y, s, clean, M, penalty) -> float:
    n_c = int(clean.sum())
    if n_c == 0:
        return math.inf
    lab = s[clean]
    Yc = Y[clean]
    cnt = np.bincount(lab, minlength=M)[:M]
    S = np.stack([np.bincount(lab, weights=Yc[:, d], minlength=M)[:M]
                  for d in range(Y.shape[1])], 1)
    nz = cnt > 0
    within = float((Yc * Yc).sum() - ((S[nz] ** 2).sum(1) / cnt[nz]).sum())
    return (max(within, 0.0) + penalty * float((~nz).sum())) / n_c


def _plateau_centre(costs, step) -> float:
    n = len(costs)
    i0 = int(np.argmin(costs))
    lo, med = float(costs[i0]), float(np.median(costs))
    thr = lo + 0.1 * max(med - lo, 0.0) + 1e-12
    ok = costs <= thr
    a = i0
    while ok[(a - 1) % n] and (i0 - a) < n - 1:
        a -= 1
    b = i0
    while ok[(b + 1) % n] and (b - a) < n - 1:
        b += 1
    return ((a + b) / 2.0) * step


def _slot_means(X, s, use, M) -> tuple:
    cnt = np.bincount(s[use], minlength=M + 1)[:M]
    S = np.zeros((M, X.shape[1]))
    np.add.at(S, s[use], X[use])
    with np.errstate(invalid="ignore", divide="ignore"):
        C = S / cnt[:, None]
    return C, cnt


def _levels(C, cnt) -> np.ndarray:
    return np.array([float(np.mean(C[j])) if cnt[j] else math.nan for j in range(len(cnt))])


def _unmix(x, A, B, D, gapped: bool) -> tuple:
    """x ~ a A + b B + (1 - a - b) D (a dark gap between the exposures), or
    x ~ a A + (1 - a) B (exposure == period). -> (a, b)."""
    if not gapped:
        d = A - B
        den = float(d @ d)
        if den <= 0:
            return math.nan, math.nan
        a = float((x - B) @ d) / den
        return a, 1.0 - a
    U = np.stack([A - D, B - D], 1)
    sol, *_ = np.linalg.lstsq(U, x - D, rcond=None)
    return float(sol[0]), float(sol[1])


def _loop_start_hint(X, s, clean, layout, contrast_min, b0, P, T, seed) -> Optional[float]:
    """Where the pack's loop starts (its first pattern, after which a loop-end gap sits),
    from coarse seamless labels: slot means -> the rotation of the pack's order that fits
    them -> start time of the slot playing layout index 0, mod T. None: no rotation fits."""
    M = layout.n
    C, cnt = _slot_means(X, s, clean, M)
    if (cnt == 0).any():
        return None
    rng = np.random.default_rng(seed)
    vecs = [C[j] for j in range(M)]
    cost = [_relation_cost(vecs, layout, r, contrast_min, None, rng) for r in range(M)]
    rot = int(np.argmin(cost))
    if not math.isfinite(cost[rot]) or cost[rot] > ANCHOR_MAX_RMS:
        return None
    return (b0 + ((-rot) % M) * P) % T


def _dark_fraction(X, s, clean, M, gap_mode) -> Optional[float]:
    """Median dark share of the frames that see two neighbouring patterns: each is
    unmixed as a A + b B + (1 - a - b) D (A, B the two slot means, D the darkest slot);
    frames seeing both (a, b > 0.1) span the whole gap, so 1 - a - b = gap / E. ~0 when
    exposure == period. None: too few such frames."""
    C, cnt = _slot_means(X, s, clean, M)
    ok = cnt > 0
    if ok.sum() < 3:
        return None
    lev = np.array([float(C[j].mean()) if ok[j] else np.inf for j in range(M)])
    D = C[int(np.argmin(lev))]
    span = float(np.max(np.linalg.norm(C[ok] - D, axis=1)))
    ds = []
    for k in np.flatnonzero(~clean & (s < M)):
        j = int(s[k])
        jn = (j + 1) % M
        if (j == M - 1 and gap_mode) or not (ok[j] and ok[jn]):
            continue
        A, B = C[j] - D, C[jn] - D
        na, nb = float(np.linalg.norm(A)), float(np.linalg.norm(B))
        if min(na, nb) < 0.25 * span or abs(float(A @ B)) / (na * nb) > 0.9:
            continue
        sol, *_ = np.linalg.lstsq(np.stack([A, B], 1), X[k] - D, rcond=None)
        a, b = float(sol[0]), float(sol[1])
        if a > 0.1 and b > 0.1:
            ds.append(1.0 - a - b)
    if len(ds) < 3:
        return None
    return max(0.0, float(np.median(ds)))


def _refine_boundaries(t, X, s, wpos, clean, C, dark, *, b0, T, P, M, E, Xexp, gapped,
                       gap_mode) -> Optional[dict]:
    """Boundary times from the straddling frames' mixture fractions, then a line fit
    time = b0 + P G (seamless loop, G = global pattern index) or b0 + T n + P j (a gap
    at the loop end). Precise to microseconds when adjacent patterns differ."""
    ests = []                                   # (time, n_loop, j)
    t_rel = t - b0
    fin = np.all(np.isfinite(C), axis=1)
    span = float(np.max(np.linalg.norm(C[fin] - dark, axis=1))) if fin.any() else 0.0
    for k in np.flatnonzero(~clean):
        j = int(s[k])
        if j >= M:
            continue
        nloop = int(math.floor(t_rel[k] / T))
        wk = float(wpos[k])
        jn = j + 1
        if jn >= M and gap_mode:
            continue
        nn, jj = (nloop + 1, 0) if jn >= M else (nloop, jn)
        A, B = C[j], C[jj]
        if not (np.all(np.isfinite(A)) and np.all(np.isfinite(B))):
            continue
        if wk < 0 or wk + E <= Xexp:            # not crossing this pattern's exposure end
            continue
        if gapped:
            # a pattern that looks like the dark between exposures (black) cannot be
            # told from it: unmix against the other pattern alone
            okA = float(np.linalg.norm(A - dark)) > 0.25 * span
            okB = float(np.linalg.norm(B - dark)) > 0.25 * span
            if not (okA or okB):
                continue
            if okA and okB:
                a, b = _unmix(X[k], A, B, dark, True)
            elif okA:
                d_ = A - dark
                a, b = float((X[k] - dark) @ d_) / float(d_ @ d_), math.nan
            else:
                d_ = B - dark
                a, b = math.nan, float((X[k] - dark) @ d_) / float(d_ @ d_)
        else:
            a, b = _unmix(X[k], A, B, dark, False)
        if not (math.isfinite(a) or math.isfinite(b)):
            continue
        if gapped:
            if 0.03 < a < 0.97:                 # covers the end of pattern j: b_j = t + aE - X
                ests.append((t[k] + a * E - Xexp, nloop, j))
            if 0.03 < b < 0.97:                 # covers the start of the next: b_next = t + E - bE
                ests.append((t[k] + E - b * E, nn, jj))
        elif 0.03 < a < 0.97:
            ests.append((t[k] + a * E, nn, jj))
    if len(ests) < 4:
        return None
    e = np.array(ests, np.float64)
    tt, nl, jj = e[:, 0], e[:, 1], e[:, 2]
    if gap_mode:
        A_ = np.stack([np.ones_like(tt), nl, jj], 1)
        if len(np.unique(nl)) < 2 or len(np.unique(jj)) < 2:
            return None
    else:
        A_ = np.stack([np.ones_like(tt), nl * M + jj], 1)
    keep = np.ones(len(tt), bool)
    for _ in range(4):
        sol, *_ = np.linalg.lstsq(A_[keep], tt[keep], rcond=None)
        res = tt - A_ @ sol
        mad = float(np.median(np.abs(res[keep] - np.median(res[keep])))) * 1.4826
        new = np.abs(res) <= max(4.0 * mad, 5e-6)
        if new.sum() < 4 or np.array_equal(new, keep):
            keep = new if new.sum() >= 4 else keep
            break
        keep = new
    sol, *_ = np.linalg.lstsq(A_[keep], tt[keep], rcond=None)
    res = tt[keep] - A_[keep] @ sol
    if gap_mode:
        nb0, nT, nP = float(sol[0]), float(sol[1]), float(sol[2])
    else:
        nb0, nP = float(sol[0]), float(sol[1])
        nT = nP * M
    return {"b0": nb0, "T": nT, "P": nP, "n": int(keep.sum()), "n_estimates": len(tt),
            "rms_s": float(np.sqrt(np.mean(res ** 2))) if len(res) else 0.0}


def _fit_model(t, X, Y, *, T, P, gap_mode, M, E, Xe, g, tot, b0_hint=None) -> dict:
    """One timing model (seamless loop, or a dark gap at the loop end): the loop start
    b0 from the within-slot dispersion (plateau centre; in gap mode near b0_hint, the
    pack's loop start found by content, when given), then T / P / b0 refined from the
    straddling frames. -> {ok, why, b0, T, P, Xe, gap, gap_mode, rms, timing_fit}."""
    ph = np.mod(t, T)
    gapped = Xe < P - 1e-6 or gap_mode
    tf = {"model": "gap" if gap_mode else "seamless", "refined": False}
    out = {"ok": False, "why": None, "b0": 0.0, "T": T, "P": P, "Xe": Xe, "gap_mode": gap_mode,
           "gap": max(T - M * P, 0.0) if gap_mode else 0.0, "rms": math.inf, "timing_fit": tf}

    def cost_at(b):
        s_, _, c_ = _label(ph, b, T, P, M, E, Xe, g)
        return _within_cost(Y, s_, c_, M, 2.0 * tot)

    nb = int(np.clip(P / 5e-6, 50, 400))
    step = P / nb
    if gap_mode:                    # the gap pins the loop start: search the whole loop
        coarse = np.arange(0.0, T, P / 40.0)
        cc = np.array([cost_at(b) for b in coarse])
        if not np.isfinite(cc).any():
            out["why"] = ("no clean frame at any pattern phase: the pattern exposure is too short "
                          "for this camera exposure / frame rate")
            return out
        bc = float(coarse[int(np.argmin(cc))]) if b0_hint is None else float(b0_hint) % T
        tf["gap_hint_ms"] = None if b0_hint is None else round(float(b0_hint) % T * 1e3, 4)
        grid = bc - P / 2 + np.arange(nb) * step
        costs = np.array([cost_at(b) for b in grid])
        b0 = (float(grid[0]) + _plateau_centre(np.r_[costs, np.inf], step)) % T
    else:
        costs = np.array([cost_at(i * step) for i in range(nb)])
        if not np.isfinite(costs).any():
            out["why"] = ("no clean frame at any pattern phase: the pattern exposure is too short "
                          "for this camera exposure / frame rate")
            return out
        b0 = _plateau_centre(costs, step)
    tf["coarse_b0_ms"] = round(b0 * 1e3, 4)
    s, wpos, clean = _label(ph, b0, T, P, M, E, Xe, g)
    C, cnt = _slot_means(X, s, clean, M)
    lev = _levels(C, cnt)
    dark = C[int(np.nanargmin(lev))] if np.isfinite(lev).any() else np.zeros(X.shape[1])
    ref = _refine_boundaries(t, X, s, wpos, clean, C, dark, b0=b0, T=T, P=P, M=M, E=E, Xexp=Xe,
                             gapped=gapped, gap_mode=gap_mode)
    out.update(ok=True, b0=b0)
    if ref is None:
        tf["warning"] = ("too few frames straddle a pattern change to refine the boundaries: "
                         "coarse timing only (lower confidence)")
        return out
    shift = ((ref["b0"] - b0 + T / 2) % T) - T / 2
    tf.update(n_boundaries=ref["n"], n_estimates=ref["n_estimates"],
              boundary_rms_us=round(ref["rms_s"] * 1e6, 2), fit_period_us=round(ref["P"] * 1e6, 3),
              fit_loop_ms=round(ref["T"] * 1e3, 4), shift_us=round(shift * 1e6, 2))
    why = None
    if abs(ref["P"] / P - 1) >= 0.01 or abs(ref["T"] / T - 1) >= 0.01:
        why = (f"the pattern changes are {ref['P'] * 1e6:.1f} us apart, not the {P * 1e6:.1f} us "
               "the loop implies")
    elif ref["rms_s"] > max(g / 2.0, 10e-6):
        why = (f"the pattern changes do not fall on a uniform grid (fit residual "
               f"{ref['rms_s'] * 1e6:.0f} us RMS)")
    elif abs(shift) >= P / 2:
        why = f"the boundary fit moved the loop start by {shift * 1e6:.0f} us"
    if why:
        out.update(ok=False, rms=ref["rms_s"], why=(
            f"{why}: the GUI's pattern period / exposure differs from the pack's, or the loop is "
            "not uniform. Give the pack (or pattern_period_us / pattern_exposure_us as entered "
            "in the GUI)"))
        return out
    T2, P2 = ref["T"], ref["P"]
    out.update(b0=(b0 + shift) % T2, T=T2, P=P2, rms=ref["rms_s"],
               gap=(T2 - M * P2) if gap_mode else 0.0, Xe=Xe if gap_mode else min(Xe, P2))
    if gap_mode and out["gap"] < -0.05 * P2:
        out.update(ok=False, why=f"negative loop gap {out['gap'] * 1e6:.0f} us")
        return out
    tf["refined"] = True
    if 3 * ref["rms_s"] > g:
        tf["warning"] = (f"boundary jitter {ref['rms_s'] * 1e6:.1f} us RMS: the {g * 1e6:.0f} us "
                         "guard is < 3 sigma")
    return out


def _blur(img, sigma) -> np.ndarray:
    import cv2
    a = np.asarray(img, np.float32)
    if not sigma or sigma <= 0:
        return a
    return cv2.GaussianBlur(a, (0, 0), float(sigma), borderType=cv2.BORDER_REFLECT)


def _row_step(shape, max_px: int = 300_000) -> int:
    return max(1, int(math.ceil(shape[0] * shape[1] / float(max_px))))


def _orientations(images, lit, blur: float = ORIENT_BLUR_PX) -> np.ndarray:
    """Per image: mean |d/dy| / mean |d/dx| of (image - the darkest image) over lit px
    (onebit.sync_signature's orientation), on row pairs (r, r + 1) every few rows.
    Subtracting the darkest image and comparing the slots with each other keeps scene
    edges from deciding; ~1 for a flood or noise, << 1 for column fringes, >> 1 for
    horizontal bars."""
    H = lit.shape[0]
    r0 = np.arange(0, H - 1, _row_step(lit.shape))
    L0, L1 = lit[r0], lit[r0 + 1]
    my = L0 & L1
    mx = L0[:, 1:] & L0[:, :-1]
    means = [float(np.asarray(im)[r0][L0].mean()) if L0.any() else float(np.mean(im))
             for im in images]
    k_dark = int(np.argmin(means))
    Dk = np.asarray(images[k_dark], np.float32)
    if blur:
        Dk = _blur(Dk, blur)
    D0, D1 = Dk[r0], Dk[r0 + 1]
    out = []
    for im in images:
        a = np.asarray(im, np.float32)
        if blur:
            a = _blur(a, blur)
        d0, d1 = a[r0] - D0, a[r0 + 1] - D1
        gy = float(np.mean(np.abs(d1 - d0)[my])) if my.any() else 0.0
        gx = float(np.mean(np.abs(np.diff(d0, axis=1))[mx])) if mx.any() else 0.0
        out.append(gy / max(gx, 1e-6))
    return np.asarray(out)


def _good_pixels(W, B, lit, contrast_min, A=None) -> tuple:
    """(white - black, px where it clears the contrast floor, its p99). A: a full-field
    white (SYNC A) when the layout has one: then only px where the decode white
    reaches FULL_WHITE_FRACTION of it count (inside a zone variant's zone, not on its
    defocused edge, where the relations in white = 1 units do not hold)."""
    con = W - B
    vals = con[lit] if lit is not None and lit.any() else con.reshape(-1)
    p99 = float(np.percentile(vals, 99)) if vals.size else 0.0
    good = (con > max(float(contrast_min), 0.1 * p99))
    if lit is not None:
        good &= lit
    if A is not None:
        good &= con >= FULL_WHITE_FRACTION * (A - B)
    return con, good, p99


def _full_white_index(layout) -> Optional[int]:
    a = [i for i in layout.index("marker") if layout.entries[i].get("level") == 1.0]
    return a[0] if a else None


def _relation_cost(vecs, layout, rot, contrast_min, orient, rng, n_px: int = 4000) -> float:
    """How badly slot images (blurred alike, so every linear relation survives) break
    the layout's relations when layout index i plays in slot (i - rot) % M: every
    pattern inside [black, white], markers at their level, each fringe group
    averaging its group level, complementary steps (k, k + N/2) summing to twice
    it; plus 1 for every 'y' sync marker whose slot is not the most y-structured.
    RMS of the terms in white = 1 units (inf: no white / black contrast)."""
    M = layout.n
    iw, ib = layout.index("white")[0], layout.index("black")[0]
    W, B = vecs[(iw - rot) % M], vecs[(ib - rot) % M]
    ia = _full_white_index(layout)
    con, good, _ = _good_pixels(W, B, None, contrast_min,
                                None if ia is None else vecs[(ia - rot) % M])
    idx = np.flatnonzero(good)
    if len(idx) < 20:
        return math.inf
    if len(idx) > n_px:
        idx = np.sort(rng.choice(idx, n_px, replace=False))
    c = con[idx]
    b = B[idx]
    v = [(vecs[(i - rot) % M][idx] - b) / c for i in range(M)]
    terms = [float(np.sqrt(np.mean([np.mean(np.maximum(vi - 1.05, 0) ** 2
                                            + np.maximum(-0.05 - vi, 0) ** 2) for vi in v])))]
    for i in layout.index("marker"):
        e = layout.entries[i]
        if e.get("level") is not None:
            terms.append(abs(float(np.mean(v[i])) - float(e["level"])))
        if e.get("orient") == "y" and orient is not None:
            terms.append(0.0 if (i - rot) % M == int(np.argmax(orient)) else 1.0)
    for _, idxs in layout.fringe_groups():
        gl = layout.group_level(idxs)
        vv = [v[i] for i in idxs]
        terms.append(float(np.sqrt(np.mean((np.mean(vv, 0) - gl) ** 2))))
        N = len(vv)
        if N % 2 == 0:
            for k in range(N // 2):
                terms.append(float(np.sqrt(np.mean((vv[k] + vv[k + N // 2] - 2 * gl) ** 2))))
    return float(np.sqrt(np.mean(np.square(terms))))


def _phase_slope(v, good) -> dict:
    """Mean wrapped phase step along camera x of an N-step group (rad / px) and its SNR."""
    N = len(v)
    d = 2 * np.pi * np.arange(N) / N
    Sx = sum(vi * np.sin(dk) for vi, dk in zip(v, d))
    Cx = sum(vi * np.cos(dk) for vi, dk in zip(v, d))
    z = np.exp(1j * np.arctan2(-Sx, Cx))
    gm = good[:, 1:] & good[:, :-1]
    if gm.sum() <= 50:
        return {}
    dph = np.angle(z[:, 1:] * np.conj(z[:, :-1]))[gm]
    dph = dph[np.abs(dph) < np.pi / 2]
    if len(dph) <= 20:
        return {}
    mu = float(np.mean(dph))
    se = float(np.std(dph)) / math.sqrt(len(dph)) + 1e-12
    return {"phase_step_rad_per_px": round(mu, 5), "phase_snr": round(abs(mu) / se, 1)}


def _check_content(images, layout, lit, contrast_min, orient=None) -> tuple:
    """Content checks in white = 1, black = 0 units on the per-pattern images (layout
    order), on every few rows (statistics, not per-pixel results). Non-finest fringe
    groups are low-passed by the layout's prefilter first (the decode does the same).
    orient: _orientations of the images (computed here when None).
    -> (checks dict, errors list)."""
    iw, ib = layout.index("white")[0], layout.index("black")[0]
    if orient is None:
        orient = _orientations(images, lit if lit is not None and lit.any()
                               else np.ones(np.shape(images[0]), bool))
    rows = np.arange(0, np.shape(images[0])[0], _row_step(np.shape(images[0])))
    full_images, images = images, [np.asarray(im, np.float32)[rows] for im in images]
    lit = None if lit is None else lit[rows]
    W, B = images[iw], images[ib]
    ia = _full_white_index(layout)
    con, good, p99 = _good_pixels(W, B, lit, contrast_min, None if ia is None else images[ia])
    errs = []
    checks = {"white_minus_black_p99_dn": round(p99, 1), "contrast_px": int(good.sum())}
    if good.sum() < 50:
        errs.append(f"white and black barely differ in the lit zone (p99 {p99:.1f} DN, "
                    f"{int(good.sum())} px above the contrast floor): raise the LED current / "
                    "exposure / gain, or the sequence is not the pack's")
        return checks, errs
    c = con[good]
    lev = [float(np.mean((im[good] - B[good]) / c)) for im in images]
    checks["levels"] = {e["name"]: round(v, 3) for e, v in zip(layout.entries, lev)}
    others = [v for i, v in enumerate(lev) if i != iw and layout.entries[i].get("level") != 1.0]
    checks["white_margin"] = round(1.0 - max(others), 3) if others else None
    checks["orientation"] = {e["name"]: round(float(o), 2) for e, o in zip(layout.entries, orient)}
    for i in layout.index("marker"):
        e = layout.entries[i]
        want = e.get("level")
        if want is not None and abs(lev[i] - float(want)) > MARKER_LEVEL_TOL:
            errs.append(f"sync marker {e['name']!r} at level {lev[i]:.2f}, expected "
                        f"{float(want):.2f} (white = 1): the loop order differs from the pack")
        if e.get("orient") == "y":
            rest = [o for j, o in enumerate(orient) if j != i]
            nxt = max(rest) if rest else 0.0
            checks["sync_orientation_ratio"] = round(float(orient[i] / max(nxt, 1e-6)), 2)
            if not orient[i] >= ORIENT_MIN_RATIO * nxt:
                j = int(np.argmax([o if k != i else -1 for k, o in enumerate(orient)]))
                errs.append(f"sync marker {e['name']!r} (horizontal bars) not found in its slot: "
                            f"its y-structure {orient[i]:.2f} vs {nxt:.2f} for "
                            f"{layout.entries[j]['name']!r} (needs {ORIENT_MIN_RATIO:g}x): the "
                            "loop order differs from the pack, or the projector is defocused so "
                            "far that the bars are gone")
    groups = []
    ref = None
    periods = [T for T, _ in layout.fringe_groups()]
    finest = min(periods) if periods else None
    for T, idx in layout.fringe_groups():
        sg = layout.prefilter_px if (layout.prefilter_px and T != finest) else 0.0
        if sg:
            Bf = np.asarray(full_images[ib], np.float32)
            Bs = _blur(Bf, sg)[rows]
            cs = _blur(np.asarray(full_images[iw], np.float32) - Bf, sg)[rows]
            v = [(_blur(full_images[i], sg)[rows] - Bs) / np.where(cs > 0, cs, 1.0) for i in idx]
        else:
            v = [(images[i] - B) / np.where(con > 0, con, 1.0) for i in idx]
        N = len(v)
        gl = layout.group_level(idx)
        g = {"period_px": T, "names": [layout.entries[i]["name"] for i in idx],
             "expected_level": round(gl, 4), "prefilter_px": sg}
        mean_dev = float(np.sqrt(np.mean((np.mean([vi[good] for vi in v], 0) - gl) ** 2)))
        g["mean_rms_dev"] = round(mean_dev, 3)
        if mean_dev > FRINGE_MEAN_TOL:
            errs.append(f"fringe group T={T:g}: its steps do not average to {gl:.2f} "
                        f"(RMS {mean_dev:.2f}): a pattern of another group, a marker or a flood "
                        "sits in its slots (loop order differs from the pack)")
        if N % 2 == 0:
            pairs = []
            for k in range(N // 2):
                s_ = v[k][good] + v[k + N // 2][good] - 2 * gl
                ac = float(np.sqrt(np.mean(((v[k][good] - v[k + N // 2][good]) / 2) ** 2)))
                res = float(np.sqrt(np.mean(s_ ** 2)))
                pairs.append({"steps": [k, k + N // 2], "sum_rms_dev": round(res, 3),
                              "ac_rms": round(ac, 3)})
                if res > 0.8 * ac + 0.06:
                    errs.append(f"fringe group T={T:g}: steps {k} and {k + N // 2} are not "
                                f"complementary (sum deviates {res:.2f}, modulation {ac:.2f}): "
                                "the steps play in another order than the pack's")
            g["pairs"] = pairs
        g.update(_phase_slope(v, good))
        groups.append(g)
    checks["fringe_groups"] = groups
    # the period groups in the pack's order: the phase runs Tb / Ta times faster in group a
    meas = [g for g in groups if g.get("phase_snr", 0) >= PHASE_SNR_MIN]
    for ga, gb in zip(meas, meas[1:]):
        want = gb["period_px"] / ga["period_px"]
        got = abs(ga["phase_step_rad_per_px"]) / max(abs(gb["phase_step_rad_per_px"]), 1e-9)
        ga.setdefault("slope_ratio_to_next", round(got, 2))
        if want > 1 and got < math.sqrt(want):
            errs.append(f"fringe group T={ga['period_px']:g} is not finer in the image than "
                        f"T={gb['period_px']:g} (phase slope ratio {got:.2f}, expected ~{want:g}): "
                        "the period groups play in another order than the pack's")
    sig = [(g["period_px"], np.sign(g["phase_step_rad_per_px"])) for g in meas
           if g.get("phase_step_rad_per_px")]
    if len({s for _, s in sig}) > 1:
        errs.append(f"fringe phase runs one way in some period groups and the other way in "
                    f"others ({[(T, int(s)) for T, s in sig]} as (period, direction)): a group's "
                    "steps play in reverse order")
    return checks, errs


def _detect_variant(images, layout, lit) -> tuple:
    """'zone' when the decode white lights much less than the full-field SYNC A white
    (the zone variant of a fast-scan pack), 'full' otherwise; None without SYNC A."""
    ia = _full_white_index(layout)
    if ia is None:
        return None, None
    iw, ib = layout.index("white")[0], layout.index("black")[0]
    B = images[ib]
    sw = images[ia] - B
    wt = images[iw] - B
    m = lit if lit is not None and lit.any() else np.ones(B.shape, bool)
    p = float(np.percentile(sw[m], 99))
    if p <= 0:
        return None, None
    area_a = int(((sw > 0.5 * p) & m).sum())
    area_w = int(((wt > 0.5 * p) & m).sum())
    ratio = area_w / max(area_a, 1)
    return ("zone" if ratio < VARIANT_ZONE_RATIO else "full"), round(ratio, 4)


def sort_frames(frames, layout: FlashLayout, *, times_s=None, frame_ids=None,
                fps: Optional[float] = None, cam_exposure_s: float,
                pattern_exposure_s: Optional[float] = None,
                pattern_period_s: Optional[float] = None,
                guard_s: float = DEFAULT_GUARD_US * 1e-6, lit_mask=None, min_loops: int = 2,
                max_features: int = 3000, contrast_min: float = 8.0, seed: int = 0,
                roi=None, full_shape=SENSOR_SHAPE) -> SortResult:
    """Sort a free-running recording of the flash loop into per-pattern images.

    frames: (N, h, w) camera ROI frames, oldest first. times_s: exposure-start times
    (the camera clock; any constant offset is fine), else frame_ids / fps.
    cam_exposure_s: the camera exposure E. pattern_period_s / pattern_exposure_s:
    what the GUI plays (a pack's timing; without them the loop is assumed seamless
    with exposure == period and its period is found from the data alone).

    1. lit pixels + features; refuse a static / dark stream
    2. loop period T by phase-dispersion minimisation over the folded frames;
       refuse when no period folds the frames (not repeating) or T disagrees with
       the pack (patterns missing / extra)
    3. pattern boundaries: coarse from within-slot dispersion, then refined from
       the straddling frames' mixture fractions (a line fit over all loops)
    4. every frame -> slot + CLEAN (exposure inside the pattern exposure, guard
       from both ends) or STRADDLE (rejected); clean frames that do not match
       their slot's other frames -> INCONSISTENT (rejected; too many: refuse)
    5. anchor: the brightest slot is 'white', the darkest must be 'black';
       markers, fringe pair sums, period-group order and phase direction checked
    6. per-pattern mean of the clean frames over all loops; counts + confidence.
    Raises SequenceError with the reason when the loop is not found / incomplete /
    out of order; its `.report` holds what was measured up to the refusal."""
    rep: dict = {}
    try:
        return _sort_frames(rep, frames, layout, times_s=times_s, frame_ids=frame_ids, fps=fps,
                            cam_exposure_s=cam_exposure_s, pattern_exposure_s=pattern_exposure_s,
                            pattern_period_s=pattern_period_s, guard_s=guard_s,
                            lit_mask=lit_mask, min_loops=min_loops, max_features=max_features,
                            contrast_min=contrast_min, seed=seed, roi=roi, full_shape=full_shape)
    except SequenceError as e:
        e.report = _plain(rep)
        raise


def _sort_frames(rep, frames, layout, *, times_s, frame_ids, fps, cam_exposure_s,
                 pattern_exposure_s, pattern_period_s, guard_s, lit_mask, min_loops,
                 max_features, contrast_min, seed, roi, full_shape) -> SortResult:
    t_start = time.perf_counter()
    layout.validate()
    Fr = np.asarray(frames)
    if Fr.ndim != 3:
        raise ValueError(f"frames must be (N, h, w), got shape {Fr.shape}")
    N, h, w = Fr.shape
    M = layout.n
    E = float(cam_exposure_s)
    g = float(guard_s)
    if not E > 0:
        raise ValueError(f"cam_exposure_s must be > 0, got {cam_exposure_s!r}")
    if N < 2 * M:
        raise SequenceError(f"only {N} frames for a {M}-pattern loop: record >= {min_loops} loops")
    ids = None if frame_ids is None else np.asarray(frame_ids, np.int64)
    t, t_src = _time_base(N, times_s, ids, fps)
    n_drop = 0
    if ids is not None:
        gaps = np.diff(ids)
        if np.any(gaps < 1):
            raise SequenceError("camera frame ids are not increasing")
        n_drop = int((gaps - 1).sum())
        if n_drop > max(2, MAX_DROP_FRACTION * (N + n_drop)):
            raise SequenceError(
                f"the camera dropped {n_drop} of {N + n_drop} frames (frame id gaps): USB3 "
                "bandwidth - lower the fps or the ROI, use a port on its own controller")
    dts = np.diff(t)
    frame_dt = float(np.median(dts))
    duration = float(t[-1] - t[0]) + frame_dt
    rep.update({"n_frames": N, "time_source": t_src, "frames_dropped": n_drop,
                "fps_measured": round(1.0 / frame_dt, 2), "camera_exposure_ms": round(E * 1e3, 4),
                "duration_ms": round(duration * 1e3, 2), "n_patterns": M})

    # 1. lit pixels, features, static / dark refusal
    lm = None
    if lit_mask is not None:
        lm = np.asarray(lit_mask, bool)
        if lm.shape != (h, w):
            raise ValueError(f"lit_mask shape {lm.shape} != frame shape {(h, w)}")
    X, lit, _, noise_sigma = _features(Fr, lm, int(max_features), seed)
    D = _pairwise_rms(X)
    np.fill_diagonal(D, np.inf)
    d_nn = D.min(1)
    d0 = float(np.percentile(d_nn, 20))
    iu = np.triu_indices(N, 1)
    med = float(np.median(D[iu]))
    rep.update(lit_px=int(lit.sum()), noise_pair_rms_dn=round(d0, 3),
               camera_noise_dn=round(noise_sigma, 3),
               median_frame_distance_dn=round(med, 3))
    if med < STATIC_RATIO * max(d0, 0.3):
        raise SequenceError(
            f"the image does not change between frames (median frame distance {med:.2f} DN vs "
            f"noise {d0:.2f} DN): the pattern sequence is not playing. In the 4500 MKII GUI: "
            "Operating Mode Pattern Sequence, Pattern Source Flash, Trigger Mode Internal/"
            "External, Play Once off, then Pattern Sequence Start/Stop/Pause -> Play")
    Pn = float(pattern_period_s) if pattern_period_s else None
    d_adj = D[np.arange(N - 1), np.arange(1, N)]
    still = d_adj < STILL_RATIO * d0 + 0.3        # same pattern: ~d0 (a stable RMS)
    run, longest, end_at = 0, 0, 0
    for k, st in enumerate(still):
        run = run + 1 if st else 0
        if run > longest:
            longest, end_at = run, k + 1
    allowed = max(2.0 * Pn / frame_dt + 3, 6) if Pn else max(12, N // 4)
    rep["longest_still_run_frames"] = int(longest)
    if longest > allowed:
        where = ("at the end of" if end_at >= N - 2 else
                 "at the start of" if end_at - longest <= 1 else "in the middle of")
        raise SequenceError(
            f"the image stays the same for {longest + 1} frames "
            f"({(longest + 1) * frame_dt * 1e3:.1f} "
            f"ms) {where} the recording: the sequence stopped or paused (Play Once on in the "
            "GUI?), or it was started after the camera")

    # 2. loop period
    Y = _pca(X, 16)
    if Pn:
        Tn = M * Pn
        lo, hi = 0.7 * Tn, min(1.45 * Tn, duration / min_loops)
        if duration < min_loops * Tn * 0.97:
            raise SequenceError(f"the recording ({duration * 1e3:.1f} ms) covers "
                                f"{duration / Tn:.2f} loops of {Tn * 1e3:.2f} ms; need {min_loops}")
    else:
        lo, hi = max(M * frame_dt / 4.0, 4 * frame_dt), duration / min_loops
    if not hi > lo:
        raise SequenceError(f"the recording ({duration * 1e3:.1f} ms) is too short to find the "
                            "loop "
                            f"period of {M} patterns ({min_loops} loops needed)")
    T, pdm_ratio, n_grid = _find_loop_period(t, Y, lo, hi, frame_dt, duration)
    rep.update(loop_search_ms=[round(lo * 1e3, 3), round(hi * 1e3, 3)], loop_search_points=n_grid,
               pdm_ratio=round(pdm_ratio, 4))
    if pdm_ratio > PDM_MAX_RATIO:
        raise SequenceError(
            f"no repeating loop of {lo * 1e3:.0f}..{hi * 1e3:.0f} ms found in the frames (best "
            f"fold cost {pdm_ratio:.2f} of the median): the sequence is not repeating (Play Once "
            "off in the GUI?), stopped during the recording, or its timing is not the pack's")
    models = []

    def slot_scan() -> dict:
        """How many equal pattern slots does the loop T hold? within-slot cost per m."""
        tot0 = float(((Y - Y.mean(0)) ** 2).sum()) / N
        scan = {}
        phm = np.mod(t, T)
        for m in range(max(3, M - 3), M + 4):
            Pm = T / m
            Xm = Pm if not pattern_exposure_s else min(float(pattern_exposure_s), Pm)
            if Xm - E - 2 * g <= 0:
                continue
            scan[m] = min(_within_cost(Y, *_label(phm, b, T, Pm, m, E, Xm, g)[::2], m, 2.0 * tot0)
                          for b in np.linspace(0, Pm, 60, endpoint=False))
        rep["loop_slots_scan"] = {str(k): round(v, 2) for k, v in scan.items() if math.isfinite(v)}
        return scan

    if Pn:
        k_meas = T / Pn
        rep["loop_patterns_measured"] = round(k_meas, 3)
        if T < Tn - 0.5 * Pn or T > Tn + 0.5 * Pn:
            scan = slot_scan()
            m_best = min(scan, key=scan.get) if scan else None
            if m_best == M:
                raise SequenceError(
                    f"the loop is {T * 1e3:.3f} ms: {M} patterns of {T / M * 1e3:.3f} ms, not the "
                    f"pack's {Pn * 1e3:.3f} ms. Another pattern period / exposure was entered in "
                    f"the GUI: enter the pack's ({Pn * 1e6:.0f} us)")
            if m_best is not None and m_best < M:
                raise SequenceError(
                    f"the loop is {T * 1e3:.3f} ms = {k_meas:.2f} pattern periods of "
                    f"{Pn * 1e3:.3f} ms, expected {M}: {M - m_best} pattern(s) are missing from "
                    "the sequence in the GUI")
            if T > Tn + 1.5 * Pn or m_best is None or m_best > M:
                raise SequenceError(
                    f"the loop is {T * 1e3:.3f} ms = {k_meas:.2f} pattern periods of "
                    f"{Pn * 1e3:.3f} ms, expected {M}: extra patterns in the sequence (or another "
                    "period was entered in the GUI)")
        models = [(T / M, False), (Pn, True)]      # seamless loop / dark gap at the loop end
    else:
        P0 = T / M
        # no pack timing: how many equal pattern slots does the loop hold?
        scan = slot_scan()
        if scan:
            m_best = min(scan, key=scan.get)
            if m_best != M and scan.get(M, math.inf) > 2.0 * scan[m_best]:
                raise SequenceError(
                    f"the loop ({T * 1e3:.3f} ms) holds {m_best} equal pattern slots of "
                    f"{T / m_best * 1e3:.3f} ms, the pack {M}: pattern(s) "
                    f"{'missing from' if m_best < M else 'added to'} the sequence in the GUI")
        models = [(P0, False)]

    # 3. pattern boundaries per timing model: coarse (within-slot dispersion), then
    #    refined from the straddling frames; keep the model that fits best
    tot = float(((Y - Y.mean(0)) ** 2).sum()) / N

    def run_models(T, dark_s: float = 0.0, gap_hint=None) -> list:
        out = []
        for P_m, gm in models:
            if not gm:
                P_m = T / M
            Xe_m = min(float(pattern_exposure_s) if pattern_exposure_s else P_m, P_m)
            if Pn and pattern_exposure_s and not gm:
                Xe_m = min(float(pattern_exposure_s) * (P_m / Pn), P_m)
            if dark_s:
                Xe_m = min(Xe_m, P_m - dark_s)
            if Xe_m - E - 2 * g <= 0:
                out.append({"ok": False, "why": (
                    f"no frame can be clean: pattern exposure {Xe_m * 1e6:.0f} us <= camera "
                    f"exposure {E * 1e6:.0f} us + 2 x {g * 1e6:.0f} us guard. Shorten the camera "
                    "exposure or lengthen the pattern exposure"),
                    "timing_fit": {"model": "gap" if gm else "seamless"}})
                continue
            out.append(_fit_model(t, X, Y, T=T, P=P_m, gap_mode=gm, M=M, E=E, Xe=Xe_m, g=g,
                                  tot=tot, b0_hint=gap_hint if gm else None))
        return out

    fits = run_models(T)
    # the fold gives the loop period to ~ a frame / loops; the straddling frames' line fit
    # to microseconds, even when that model is refused (e.g. seamless across a loop gap).
    # With few loops the fold's error can put the slots off the gap: refit at the better T.
    cand = [f_["timing_fit"] for f_ in fits if f_["timing_fit"].get("n_boundaries", 0) >= 8
            and f_["timing_fit"].get("fit_loop_ms")]
    if cand:
        T2 = min(cand, key=lambda c: c["boundary_rms_us"])["fit_loop_ms"] * 1e-3
        if 1e-5 * T < abs(T2 - T) < 0.02 * T:
            rep["loop_fold_ms"] = round(T * 1e3, 4)
            T = T2
            fits = run_models(T)
    # a dark gap at the loop end is too short to show in the within-slot dispersion:
    # locate it from the frames that saw it (seamless labels) and refit the gap model
    seam = next((f_ for f_ in fits if f_.get("gap_mode") is False and "b0" in f_), None)
    if seam is not None and any(gm for _, gm in models) and not any(f_["ok"] for f_ in fits):
        s_s, _, c_s = _label(np.mod(t, seam["T"]), seam["b0"], seam["T"], seam["P"], M, E,
                             seam["Xe"], g)
        hint = _loop_start_hint(X, s_s, c_s, layout, contrast_min, seam["b0"], seam["P"], T,
                                seed)
        if hint is not None:
            fits = run_models(T, gap_hint=hint)
    # exposure shorter than the period in the GUI (dark between the patterns)? The frames
    # that see two neighbouring patterns tell (a + b < 1 of them), with the slots from the
    # coarse timing of the first model. Refit with the measured exposure.
    probe = next((f for f in fits if "gap_mode" in f and f.get("b0") is not None), None)
    if probe is not None:
        s_p, _, c_p = _label(np.mod(t, probe["T"]), probe["b0"], probe["T"], probe["P"], M, E,
                             probe["Xe"], g)
        dark_frac = _dark_fraction(X, s_p, c_p, M, probe["gap_mode"])
        if dark_frac is not None:
            gap_est = dark_frac * E
            rep["dark_between_patterns_us"] = round(gap_est * 1e6, 1)
            if gap_est > max(2 * g, 0.03 * probe["P"]) and probe["Xe"] >= probe["P"] - 1e-6:
                fits = run_models(T, 1.1 * gap_est)
                for f_ in fits:
                    f_["timing_fit"]["exposure_measured"] = True
                rep.setdefault("warnings", []).append(
                    f"the patterns are dark for ~{gap_est * 1e6:.0f} us between exposures (the "
                    "GUI's exposure is shorter than its period; the pack plays exposure = "
                    "period): only frames inside each pattern's lit part are used")
    rep["timing_models"] = [f["timing_fit"] for f in fits]
    good = [f for f in fits if f["ok"]]
    if not good:
        raise SequenceError(fits[0]["why"] if len(fits) == 1 else " / ".join(
            f"{f['timing_fit'].get('model')}: {f['why']}" for f in fits))
    best_fit = min(good, key=lambda f: f["rms"])
    b0, T, P, Xe = best_fit["b0"], best_fit["T"], best_fit["P"], best_fit["Xe"]
    gap_mode, gap = best_fit["gap_mode"], best_fit["gap"]
    timing_fit = best_fit["timing_fit"]
    rep["timing_fit"] = timing_fit
    ph = np.mod(t, T)
    s, wpos, clean = _label(ph, b0, T, P, M, E, Xe, g)

    # 4. consistency of the clean frames with their slot
    C, cnt = _slot_means(X, s, clean, M)
    resid = np.full(N, np.nan)
    for j in range(M):
        kk = np.flatnonzero(clean & (s == j))
        if len(kk) == 0:
            continue
        ref_img = np.median(X[kk], 0) if len(kk) >= 3 else C[j]
        resid[kk] = np.sqrt(np.mean((X[kk] - ref_img) ** 2, 1))
    lev = _levels(C, cnt)
    # noise model sigma^2 = a + b level from the slots with >= 3 clean frames
    rows = [(float(lev[j]), float(np.median(resid[clean & (s == j)]) ** 2), int(cnt[j]))
            for j in range(M) if cnt[j] >= 3]
    if len(rows) >= 2:
        A_ = np.array([[1.0, r[0]] for r in rows])
        yv = np.array([r[1] for r in rows])
        wv = np.sqrt([r[2] for r in rows])
        sol, *_ = np.linalg.lstsq(A_ * wv[:, None], yv * wv, rcond=None)
        a_n, b_n = max(float(sol[0]), 0.0), max(float(sol[1]), 0.0)
        if a_n + b_n * max(float(np.nanmin(lev)), 0.0) <= 0:
            a_n = float(np.median(yv))
    else:
        a_n, b_n = (d0 / math.sqrt(2)) ** 2, 0.0
    sig = np.sqrt(np.maximum(a_n + b_n * np.clip(np.nan_to_num(lev), 0, None), 1e-6))
    incons = np.zeros(N, bool)
    for j in range(M):
        kk = np.flatnonzero(clean & (s == j))
        if len(kk) >= 2:
            f = math.sqrt((len(kk) - 1) / len(kk)) if len(kk) < 3 else 1.0
            incons[kk] = resid[kk] > 4.0 * sig[j] * f + 1.0
    n_clean0 = int(clean.sum())
    frac_incons = float(incons.sum()) / max(n_clean0, 1)
    rep["noise_model_dn"] = {"read_var": round(a_n, 3), "shot_var_per_dn": round(b_n, 4)}
    if frac_incons > MAX_INCONSISTENT_FRACTION:
        raise SequenceError(
            f"{int(incons.sum())} of {n_clean0} frames the loop timing calls clean do not match "
            "their pattern's other frames: the sequence changed or stopped during the recording, "
            "the pattern exposure is shorter than the period (give the pack / "
            "pattern_exposure_us), or the loop timing is not uniform")
    use = clean & ~incons
    C, cnt = _slot_means(X, s, use, M)
    empty = [j for j in range(M) if cnt[j] == 0]
    lev = np.array([float(np.mean(C[j])) if cnt[j] else math.nan for j in range(M)])

    # 5. per-slot images (mean of the clean frames), then the anchor: the rotation of
    #    the pack's order that fits them (sync markers, levels, fringe relations)
    iw, ib = layout.index("white")[0], layout.index("black")[0]
    if empty:
        raise SequenceError(
            f"{len(empty)} of {M} patterns have no clean frame in {duration * 1e3:.0f} ms: the "
            f"pattern exposure ({Xe * 1e6:.0f} us) must be >= camera exposure ({E * 1e6:.0f} us) + "
            f"frame period ({frame_dt * 1e6:.0f} us) + 2 guard ({g * 1e6:.0f} us); record more "
            "loops, shorten the exposure or raise the frame rate")
    slot_img = []
    for j in range(M):
        kk = np.flatnonzero(use & (s == j))
        acc = np.zeros((h, w), np.float32)
        for k in kk:
            acc += Fr[k]
        slot_img.append(acc / np.float32(len(kk)))
    lv, lu = np.nonzero(lit)
    y0, y1 = max(int(lv.min()) - 2, 0), min(int(lv.max()) + 3, h)
    x0, x1 = max(int(lu.min()) - 2, 0), min(int(lu.max()) + 3, w)
    crop = [im[y0:y1, x0:x1] for im in slot_img]
    litc = lit[y0:y1, x0:x1]
    orient = _orientations(crop, litc)
    rng = np.random.default_rng(seed)
    pix = np.flatnonzero(litc)
    if len(pix) > 60_000:
        pix = np.sort(rng.choice(pix, 60_000, replace=False))
    vecs = [_blur(im, layout.prefilter_px or 1.0).reshape(-1)[pix] for im in crop]
    rcost = np.array([_relation_cost(vecs, layout, r, contrast_min, orient, rng)
                      for r in range(M)])
    order_ = np.argsort(rcost)
    rot = int(order_[0])                    # layout index = (slot + rot) % M
    best, second = float(rcost[order_[0]]), float(rcost[order_[1]])
    separation = second / max(best, 0.01)
    wslot = (iw - rot) % M
    rep["anchor"] = {"white_slot": int(wslot), "rotation": rot, "fit_rms": round(best, 4),
                     "next_best_rms": round(second, 4) if math.isfinite(second) else None,
                     "separation": round(separation, 2) if math.isfinite(separation) else None,
                     "sync_slots": {layout.entries[i]["name"]: int((i - rot) % M)
                                    for i in layout.index("marker")}}
    if not math.isfinite(best):
        raise SequenceError("no pair of recorded patterns looks like white and black: the zone "
                            "is not lit by the sequence, or the loop is not the pack's")
    lay_of_slot = (np.arange(M) + rot) % M
    labels = np.where(use & (s < M), lay_of_slot[np.minimum(s, M - 1)], -1)
    status = np.full(N, "straddle", dtype="<U12")
    status[s >= M] = "gap"
    status[incons] = "inconsistent"
    status[use] = "clean"

    # 6. per-pattern images (full ROI, layout order) + content checks
    images = [None] * M
    counts = [0] * M
    loops_seen = [0] * M
    loop_idx = np.floor((t - b0) / T).astype(np.int64)
    for j in range(M):
        li = int(lay_of_slot[j])
        images[li] = slot_img[j]
        counts[li] = int(cnt[j])
        loops_seen[li] = int(len(np.unique(loop_idx[use & (s == j)])))
    crop_l = [im[y0:y1, x0:x1] for im in images]
    checks, errs = _check_content(crop_l, layout, litc, contrast_min,
                                  orient=np.asarray([orient[(i - rot) % M] for i in range(M)]))
    variant, vratio = _detect_variant(crop_l, layout, litc)
    checks["variant_detected"] = variant
    checks["white_over_sync_white_area"] = vratio
    rep["checks"] = checks
    if best > ANCHOR_MAX_RMS:
        errs = [f"no rotation of the pack's order fits the recorded patterns (best fit "
                f"{best:.2f} RMS in white = 1 units, with white in slot {wslot}): the loop "
                "order differs from the pack"] + errs[:2] + (
                    [f"({len(errs) - 2} more checks fail)"] if len(errs) > 2 else [])
    elif separation < ANCHOR_MIN_SEPARATION:
        errs.insert(0, f"the loop start is ambiguous (two rotations fit: {best:.3f} vs "
                       f"{second:.3f} RMS): the sync markers are missing or not distinct")
    if errs:
        raise SequenceError("; ".join(errs))
    if layout.variant and variant and variant != layout.variant:
        rep.setdefault("warnings", []).append(
            f"the frames show the {variant!r} image, the scan was asked for {layout.variant!r} "
            "(check the flash image index in the GUI)")

    # confidence
    loops_cov = duration / T
    exp_loops = max(1.0, math.floor(loops_cov))
    c_timing = float(np.clip(1.0 - pdm_ratio / PDM_MAX_RATIO, 0, 1))
    c_clean = float(np.clip(min(loops_seen) / exp_loops, 0, 1))
    c_cons = float(np.clip(1.0 - frac_incons / MAX_INCONSISTENT_FRACTION, 0, 1))
    c_anchor = float(np.clip(min((separation - ANCHOR_MIN_SEPARATION) / 3.0,
                                 1.0 - best / ANCHOR_MAX_RMS), 0, 1))
    c_fit = 1.0 if timing_fit.get("refined") else 0.5
    conf = round(min(c_timing, c_clean, c_cons, c_anchor, c_fit), 3)
    n_str = int((status == "straddle").sum())
    rep.update({
        "loop_period_ms": round(T * 1e3, 4), "pattern_period_ms": round(P * 1e3, 4),
        "pattern_exposure_ms": round(Xe * 1e3, 4), "loop_gap_ms": round(gap * 1e3, 4),
        "guard_us": round(g * 1e6, 1), "loops_covered": round(loops_cov, 2),
        "first_loop_start_ms": round(((b0 + ((-rot) % M) * P) % T) * 1e3, 4),
        "timing_fit": timing_fit,
        "n_clean": int(use.sum()), "n_straddle": n_str, "n_gap": int((status == "gap").sum()),
        "n_inconsistent": int(incons.sum()),
        "counts": {e["name"]: c for e, c in zip(layout.entries, counts)},
        "loops_seen": {e["name"]: c for e, c in zip(layout.entries, loops_seen)},
        "min_count": int(min(counts)), "min_loops_seen": int(min(loops_seen)),
        "variant_detected": checks.get("variant_detected"),
        "status_counts": {k: int((status == k).sum()) for k in
                          ("clean", "straddle", "gap", "inconsistent")},
        "confidence": conf,
        "confidence_terms": {"loop_fold": round(c_timing, 3), "clean_every_loop": round(c_clean, 3),
                             "consistency": round(c_cons, 3), "anchor": round(c_anchor, 3),
                             "boundary_fit": c_fit},
        "sort_s": round(time.perf_counter() - t_start, 3)})
    if timing_fit.get("warning"):
        rep.setdefault("warnings", []).append(timing_fit["warning"])
    if min(loops_seen) < exp_loops:
        rep.setdefault("warnings", []).append(
            f"some patterns were clean in only {min(loops_seen)} of {int(exp_loops)} loops")
    return SortResult(images, counts, loops_seen, labels, status, float(T), float(P), conf, rep,
                      tuple(roi) if roi is not None else (0, 0, w, h),
                      tuple(full_shape) if roi is not None else (h, w))


def sort_recording(rec: Recording, layout: FlashLayout, *, lit_mask=None, **kw) -> SortResult:
    """sort_frames on a Recording (its timestamps, ids, exposure and ROI); lit_mask may be
    full-frame (cropped to the ROI here) or ROI-sized."""
    t, src = rec.times_s()
    x, y, w, h = rec.roi
    lm = None
    if lit_mask is not None:
        lm = np.asarray(lit_mask, bool)
        if lm.shape == tuple(rec.full_shape):
            lm = lm[y:y + h, x:x + w]
    kw.setdefault("pattern_exposure_s", layout.pattern_exposure_us * 1e-6
                  if layout.pattern_exposure_us else None)
    kw.setdefault("pattern_period_s", layout.pattern_period_us * 1e-6
                  if layout.pattern_period_us else None)
    if layout.guard_us:
        kw.setdefault("guard_s", float(layout.guard_us) * 1e-6)
    res = sort_frames(rec.frames, layout, times_s=t, frame_ids=rec.frame_ids, fps=rec.fps,
                      cam_exposure_s=rec.exposure_us * 1e-6, lit_mask=lm, roi=rec.roi,
                      full_shape=rec.full_shape, **kw)
    res.report["time_source"] = src
    return res


def recording_layout(rec: Recording, pack=None, variant: Optional[str] = None) -> FlashLayout:
    """The layout to sort a saved recording with: the given pack (dir / manifest), else
    the one the scan recorded in its meta, else the fast-scan defaults."""
    if pack is not None:
        return load_pack_layout(pack, variant)[0]
    d = (rec.meta or {}).get("layout")
    if d:
        lay = FlashLayout.from_json(d)
        if variant is not None:
            lay.variant = variant
        return lay
    return default_layout(variant)


def sort_file(path, pack=None, variant: Optional[str] = None, **kw) -> tuple:
    """Offline: a saved recording (.npz) -> (SortResult, FlashLayout, Recording). Pure:
    no rig, no library."""
    rec = Recording.load(path)
    lay = recording_layout(rec, pack, variant)
    return sort_recording(rec, lay, **kw), lay, rec


def zone_edge_keep(res: SortResult, layout: FlashLayout) -> Optional[np.ndarray]:
    """ROI-sized bool: px whose decode white reaches ZONE_EDGE_MIN of the full-field SYNC A
    white (both minus black, lightly blurred), when the frames show the zone variant.
    Outside it a px sits on the zone mask's defocused edge, where the fringes are clipped
    and the phase (even the fringe order) is wrong. None: not a zone loop, or no SYNC A."""
    ia = _full_white_index(layout)
    seen = (res.report or {}).get("variant_detected") or layout.variant
    if ia is None or seen != "zone":
        return None
    iw, ib = layout.index("white")[0], layout.index("black")[0]
    B = _blur(res.images[ib], ZONE_EDGE_BLUR_PX)
    A = _blur(res.images[ia], ZONE_EDGE_BLUR_PX) - B
    W = _blur(res.images[iw], ZONE_EDGE_BLUR_PX) - B
    return W >= ZONE_EDGE_MIN * np.maximum(A, 1e-3)


def decode_sorted(res: SortResult, layout: FlashLayout, geometry_id: str, *,
                  contrast_min: float = 8.0, conf_min: float = 4.0,
                  fine_px: Optional[float] = None):
    """Sorted per-pattern images -> DeformationField through the unchanged
    MultiFreqBasis (white, black, fringes fine -> coarse; the layout's diamond
    parity). The non-finest fringe groups are low-passed by the layout's prefilter
    first (for the fast-scan pack: onebit.prefilter_frames, Gaussian 3 camera px on
    mid + coarse; white and black untouched; the fine group only with fine_px, default
    layout.fine_prefilter_px = off), on the ROI images, which are then embedded in full
    camera frames (0 outside the ROI: under the contrast floor, left undecoded).
    Left undecoded as well: px where the prefilter mixes two surfaces at a depth edge
    (onebit.phasor_coherence < PREFILTER_COHERENCE_MIN: the mixed phase can pick a wrong
    fringe order), and in the zone variant px on the zone's edge (zone_edge_keep)."""
    from ..bases.multifreq import MultiFreqBasis
    basis = MultiFreqBasis(steps=layout.steps, periods_px=layout.periods, axis="x",
                           diamond_parity=layout.diamond_parity)
    roi_frames = res.frames_for_basis(layout, full=False)
    fpx = layout.fine_prefilter_px if fine_px is None else fine_px
    fpx = float(fpx) if fpx else 0.0
    coh_keep = None
    if layout.prefilter_px:
        # px where the prefilter would mix two surfaces (a depth edge): the mixed phase of a
        # prefiltered group can pick the wrong order of the next finer one (onebit.decode)
        from ..bases.onebit import PREFILTER_COHERENCE_MIN, phasor_coherence
        finest = min(layout.periods)
        coh_keep = np.ones(np.shape(roi_frames[0]), bool)
        i = 2
        for T in layout.periods:
            if T != finest:
                coh_keep &= (phasor_coherence(roi_frames[i:i + layout.steps], layout.prefilter_px)
                             >= PREFILTER_COHERENCE_MIN)
            i += layout.steps
    if layout.prefilter_px or fpx:
        finest = min(layout.periods)
        out, i = list(roi_frames[:2]), 2
        for T in layout.periods:
            for _ in range(layout.steps):
                sg = fpx if T == finest else float(layout.prefilter_px or 0.0)
                out.append(_blur(roi_frames[i], sg))
                i += 1
        roi_frames = out
    edge_keep = zone_edge_keep(res, layout)
    # decode only the rows / columns where white and black differ (the zone), then
    # place the result in the full camera frame
    from ..decode.deformation import DeformationField
    x, y, w, h = res.roi
    con = np.abs(np.asarray(roi_frames[0], np.float32) - np.asarray(roi_frames[1], np.float32))
    vv, uu = np.nonzero(con > contrast_min)
    if len(vv):
        a0, a1 = int(vv.min()), int(vv.max()) + 1
        c0, c1 = int(uu.min()), int(uu.max()) + 1
    else:
        a0, a1, c0, c1 = 0, min(h, 2), 0, min(w, 2)
    sub = [np.asarray(im, np.float32)[a0:a1, c0:c1] for im in roi_frames]
    part = basis.decode(sub, geometry_id, contrast_min=contrast_min, conf_min=conf_min)
    H, W = tuple(res.full_shape)
    Y0, X0 = y + a0, x + c0
    sl = (slice(Y0, Y0 + (a1 - a0)), slice(X0, X0 + (c1 - c0)))
    cx = np.full((H, W), np.nan, np.float32)
    cx[sl] = part.corr_x
    cf = np.zeros((H, W), np.float32)
    cf[sl] = part.conf
    mk = np.zeros((H, W), bool)
    mk[sl] = part.mask
    n_edge = n_coh = 0
    if edge_keep is not None:
        drop = part.mask & ~edge_keep[a0:a1, c0:c1]
        n_edge = int(drop.sum())
        mk[sl] &= ~drop
    if coh_keep is not None:
        drop = mk[sl] & ~coh_keep[a0:a1, c0:c1]
        n_coh = int(drop.sum())
        mk[sl] &= ~drop
    cx[~mk] = np.nan
    fld = DeformationField(corr_x=cx, corr_y=None, corr_diag=None, conf=cf, mask=mk,
                           geometry_id=geometry_id, basis_names=list(part.basis_names),
                           meta=dict(part.meta))
    fld.meta["decoded_box_xywh"] = [int(X0), int(Y0), int(c1 - c0), int(a1 - a0)]
    fld.meta.update({"softsync": {"loop_period_ms": res.report.get("loop_period_ms"),
                                  "pattern_period_ms": res.report.get("pattern_period_ms"),
                                  "confidence": res.confidence, "counts": res.report.get("counts"),
                                  "roi": list(res.roi)},
                     "prefilter_px": layout.prefilter_px, "fine_prefilter_px": fpx or None,
                     "zone_edge": None if edge_keep is None else {
                         "min_white_over_sync_white": ZONE_EDGE_MIN, "masked_px": n_edge},
                     "prefilter_edge": None if coh_keep is None else {
                         "min_coherence": _coherence_min(), "masked_px": n_coh},
                     "layout_source": layout.source})
    return fld


def _coherence_min() -> float:
    from ..bases.onebit import PREFILTER_COHERENCE_MIN
    return float(PREFILTER_COHERENCE_MIN)


def _plain(o):
    if isinstance(o, dict):
        return {str(k): _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if isinstance(o, np.ndarray):
        return _plain(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, Path):
        return str(o)
    return o


# ============================================================ bench only
# Nothing below runs in tests. PySpin is passed in (or imported lazily).

class PySpinFreeRunCamera:  # pragma: no cover - bench only
    """Free-running Blackfly S for soft sync over an already-open PySpin camera `cam`
    (the engine's RealCamera opened it): continuous acquisition, ROI, fixed exposure
    and gain, AcquisitionFrameRate fixed, chunk FrameID + Timestamp, OldestFirst with
    enough buffers for every frame. restore() puts back full frame, no frame-rate cap,
    NewestOnly (the HDMI scan path's setting); the next rig open resets the rest."""

    def __init__(self, cam, PySpin=None):
        if PySpin is None:
            import PySpin  # noqa: N813
        self.cam, self.PySpin = cam, PySpin
        self.applied = {}
        self._chunks = False

    def increments(self) -> dict:
        c = self.cam
        out = {}
        for k, node in (("x", "OffsetX"), ("y", "OffsetY"), ("w", "Width"), ("h", "Height")):
            try:
                out[k] = int(getattr(c, node).GetInc())
            except Exception:  # noqa: BLE001 - keep the conservative default
                out[k] = ROI_STEP[k]
        return out

    def max_fps(self) -> Optional[float]:
        try:
            return float(self.cam.AcquisitionFrameRate.GetMax())
        except Exception:  # noqa: BLE001
            return None

    def configure(self, *, roi, exposure_us, fps, gain_db=None, buffers=64) -> dict:
        P, c = self.PySpin, self.cam
        try:
            c.EndAcquisition()
        except Exception:  # noqa: BLE001 - was not streaming
            pass
        c.TriggerMode.SetValue(P.TriggerMode_Off)
        c.ExposureAuto.SetValue(P.ExposureAuto_Off)
        c.ExposureMode.SetValue(P.ExposureMode_Timed)
        if gain_db is not None:
            c.GainAuto.SetValue(P.GainAuto_Off)
            c.Gain.SetValue(float(gain_db))
        x, y, w, h = (int(v) for v in roi)
        c.OffsetX.SetValue(0)
        c.OffsetY.SetValue(0)
        c.Width.SetValue(w)
        c.Height.SetValue(h)
        c.OffsetX.SetValue(x)
        c.OffsetY.SetValue(y)
        c.ExposureTime.SetValue(float(exposure_us))
        c.AcquisitionMode.SetValue(P.AcquisitionMode_Continuous)
        c.AcquisitionFrameRateEnable.SetValue(True)
        fmax = float(c.AcquisitionFrameRate.GetMax())
        c.AcquisitionFrameRate.SetValue(min(float(fps), fmax))
        try:
            c.ChunkModeActive.SetValue(True)
            for sel in ("FrameID", "Timestamp"):
                c.ChunkSelector.SetValue(getattr(P, f"ChunkSelector_{sel}"))
                c.ChunkEnable.SetValue(True)
            self._chunks = True
        except Exception:  # noqa: BLE001 - image leader ids / timestamps instead
            self._chunks = False
        s = c.TLStream
        s.StreamBufferCountMode.SetValue(P.StreamBufferCountMode_Manual)
        try:
            nmax = int(s.StreamBufferCountManual.GetMax())
        except Exception:  # noqa: BLE001
            nmax = int(buffers)
        s.StreamBufferCountManual.SetValue(int(min(buffers, nmax)))
        s.StreamBufferHandlingMode.SetValue(P.StreamBufferHandlingMode_OldestFirst)
        try:
            resulting = float(c.AcquisitionResultingFrameRate.GetValue())
        except Exception:  # noqa: BLE001
            resulting = None
        self.applied = {"roi": [x, y, w, h], "exposure_us": float(c.ExposureTime.GetValue()),
                        "fps": float(c.AcquisitionFrameRate.GetValue()), "fps_max": fmax,
                        "resulting_fps": resulting, "gain_db": float(c.Gain.GetValue()),
                        "chunks": self._chunks, "buffers": int(min(buffers, nmax)),
                        "increments": self.increments(), "buffer_mode": "OldestFirst"}
        return self.applied

    def start(self) -> None:
        self.cam.BeginAcquisition()

    def next_frame(self, timeout_ms: int):
        img = self.cam.GetNextImage(int(timeout_ms))
        host = time.perf_counter_ns()
        try:
            if img.IsIncomplete():
                return None
            fid, ts = img.GetFrameID(), img.GetTimeStamp()
            if self._chunks:
                try:
                    ch = img.GetChunkData()
                    fid, ts = ch.GetFrameID(), ch.GetTimestamp()
                except Exception:  # noqa: BLE001 - leader values are the same clock
                    pass
            return img.GetNDArray().copy(), int(fid), int(ts), host
        finally:
            img.Release()

    def stop(self) -> None:
        try:
            self.cam.EndAcquisition()
        except Exception:  # noqa: BLE001 - not streaming
            pass

    def restore(self) -> None:
        P, c = self.PySpin, self.cam
        self.stop()
        for step in (lambda: c.ChunkModeActive.SetValue(False),
                     lambda: c.AcquisitionFrameRateEnable.SetValue(False),
                     lambda: c.OffsetX.SetValue(0), lambda: c.OffsetY.SetValue(0),
                     lambda: c.Width.SetValue(c.Width.GetMax()),
                     lambda: c.Height.SetValue(c.Height.GetMax()),
                     lambda: c.TLStream.StreamBufferCountMode.SetValue(
                         P.StreamBufferCountMode_Auto),
                     lambda: c.TLStream.StreamBufferHandlingMode.SetValue(
                         P.StreamBufferHandlingMode_NewestOnly)):
            try:
                step()
            except Exception:  # noqa: BLE001 - best effort; the next open resets the camera
                pass

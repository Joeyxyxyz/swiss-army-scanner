"""
Dent sets: the pattern frames the dent detector reads, and their patterns.

Production capture (capture_dent_set, one pose), patterns in DMD coords
X = column, Y = row / 2 (the diamond array's rows are half a column pitch):

  long  pass (the preset's dark variant, ~66 667 us):
        flood 255, flood 0, sin0 T12 x8, sin0 T36 x8, sin90 T12 x8, sin90 T36 x8
  short pass (~16 667 us):
        flood 0, flood 128, flood 255, the same sinusoid sets,
        optional lines_v x10 (single-mirror vertical lines, pitch 10)
  + ONE metric scan (scan_metric, same zone): cylinder pose, carrier, depth.

sin90 T36 is in the production set because the ablation on the bench sets
showed that without it (the "minimal" set: sin0 T12, sin0 T36, sin90 T12)
the small dent of the re-placed cell (pose 2) loses score: its pattern-warp
evidence is in the coarse horizontal fringes. That choice was made by
scoring pose 2, which was meant as held-out: pose 2 is tuned-on data for
it (see fusion.py, TUNING PROVENANCE).

Flood levels are the levels SENT to the DMD. Which one lands bright depends
on projection.invert, so it is measured from the floods, never assumed.

Two layouts load the same way:
  dentsets/<geometry_id>/<ts>_<label>/   engine entries: meta.json (written
      last) + frames_long.npy / frames_short.npy (float32 camera DN, cropped
      to the zone), cam_zone_mask.npy, dmd_zone_mask.npy, preview.png
  <folder>/dataset.json + frames_<preset>.npy   the legacy offline sets
      (one pattern list shared by both passes)

Pattern subsets (`subset`): "all" = everything the finders use (zigzag,
checker and dot patterns are never used), "production" = exactly what
capture_dent_set records (long / short lists above), "production_nolines"
= the same without lines_v, "minimal" = production without sin90 T36
(capture_dent_set(pattern_set="minimal")). A subset of a legacy set is how
a capture is evaluated on the richer development captures.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ._util import SAT

DMD_SHAPE = (1140, 912)                      # rows, cols
LONG_US_DEFAULT = 66667
SHORT_US_DEFAULT = 16667
PRODUCTION_SINES = ((0, 12), (0, 36), (90, 12), (90, 36))   # (theta_deg, period), 8 steps each
MINIMAL_SINES = ((0, 12), (0, 36), (90, 12))
SIN_STEPS = 8
LINES_PITCH = 10
UNUSED_KINDS = ("zigzag", "checker", "dots")        # never decoded by the dent finders
SUBSETS = ("all", "production", "production_nolines", "minimal")
PATTERN_SETS = ("production", "minimal")                        # what capture_dent_set records
RADIOMETRIC_KEYS = ("gain_db", "duty", "gamma")    # per pass; must match (HDR = exposure ratio)
FORMAT = 1


# ------------------------------------------------------------ pattern images

def _grids():
    H, W = DMD_SHAPE
    col = np.arange(W, dtype=np.float64)[None, :].repeat(H, 0)
    row = np.arange(H, dtype=np.float64)[:, None].repeat(W, 1)
    return col, row


def pattern_image(spec: dict) -> np.ndarray:
    """DMD image (uint8, 1140 x 912) for one pattern spec."""
    col, row = _grids()
    k = spec["kind"]
    if k == "flood":
        return np.full(DMD_SHAPE, int(spec["level"]), np.uint8)
    if k == "sinus":
        t = np.radians(float(spec["theta_deg"]))
        X, Y = col, row / 2.0                       # X = col, Y = row / 2
        ph = 2 * np.pi * int(spec["step"]) / int(spec["steps"])
        arg = 2 * np.pi * (X * np.cos(t) + Y * np.sin(t)) / float(spec["period"]) - ph
        v = 127.5 + 127.5 * np.cos(arg)
        return np.clip(v, 0, 255).astype(np.uint8)
    if k == "lines_v":
        pitch = int(spec.get("pitch", LINES_PITCH))
        return ((((col.astype(int) - int(spec["shift"])) % pitch) == 0) * 255).astype(np.uint8)
    if k == "lines_h":
        pitch = int(spec.get("pitch_rows", 20))
        on = ((row.astype(int) - int(spec["shift_rows"])) % pitch) < 2
        return (on * 255).astype(np.uint8)
    raise ValueError(f"no generator for pattern kind {k!r}")


def _sines(sets=PRODUCTION_SINES, steps: int = SIN_STEPS) -> list:
    return [{"kind": "sinus", "theta_deg": th, "period": T, "step": k, "steps": steps}
            for th, T in sets for k in range(steps)]


def production_specs(which: str, lines_v: bool = True, pattern_set: str = "production") -> list:
    """Pattern specs of one production pass ('long' | 'short'), in capture order."""
    if pattern_set not in PATTERN_SETS:
        raise ValueError(f"pattern_set must be one of {PATTERN_SETS}, got {pattern_set!r}")
    sines = _sines(PRODUCTION_SINES if pattern_set == "production" else MINIMAL_SINES)
    if which == "long":
        return [{"kind": "flood", "level": 255}, {"kind": "flood", "level": 0}] + sines
    if which == "short":
        out = [{"kind": "flood", "level": 0}, {"kind": "flood", "level": 128},
               {"kind": "flood", "level": 255}] + sines
        if lines_v:
            out += [{"kind": "lines_v", "pitch": LINES_PITCH, "shift": k}
                    for k in range(LINES_PITCH)]
        return out
    raise ValueError("which must be 'long' or 'short'")


def _key(spec: dict) -> tuple:
    return tuple(sorted((k, v) for k, v in spec.items() if k != "index"))


def select(specs: list, which: str, subset: str = "all") -> list:
    """Indices of `specs` that a subset keeps for pass `which` ('long' | 'short')."""
    if subset not in SUBSETS:
        raise ValueError(f"subset must be one of {SUBSETS}, got {subset!r}")
    if subset == "all":
        return [i for i, p in enumerate(specs) if p.get("kind") not in UNUSED_KINDS]
    pset = "minimal" if subset == "minimal" else "production"
    want = {_key(p) for p in production_specs(which, lines_v=(subset != "production_nolines"),
                                              pattern_set=pset)}
    return [i for i, p in enumerate(specs) if _key(p) in want]


# ------------------------------------------------------------ in memory

@dataclass
class Pass:
    """One exposure: frames (N, H, W) camera DN and their pattern specs."""
    name: str                       # 'long' | 'short'
    frames: np.ndarray
    specs: list
    exposure_us: float
    preset: str = ""
    average: int = 1
    on_level: int = 0               # flood level that lands BRIGHT (measured)
    off_level: int = 255            # flood level that lands dark (ambient only)

    def idx(self, **kw) -> list:
        return [i for i, p in enumerate(self.specs) if all(p.get(k) == v for k, v in kw.items())]

    def flood(self, level: int) -> Optional[int]:
        i = self.idx(kind="flood", level=level)
        return i[0] if i else None

    @property
    def on(self) -> np.ndarray:
        return self.frames[self.flood(self.on_level)]

    @property
    def off(self) -> np.ndarray:
        return self.frames[self.flood(self.off_level)]

    @property
    def mid(self) -> Optional[np.ndarray]:
        i = self.flood(128)
        return None if i is None else self.frames[i]

    def set_polarity(self) -> None:
        a, b = self.flood(0), self.flood(255)
        if a is None or b is None:
            raise ValueError(f"{self.name} pass: floods 0 and 255 are both needed (polarity)")
        d = np.median(self.frames[a] - self.frames[b])
        self.on_level, self.off_level = (0, 255) if d > 0 else (255, 0)

    @property
    def inverted(self) -> bool:
        return self.on_level == 0


@dataclass
class DentData:
    """A loaded dent set (both passes, same crop)."""
    passes: dict                    # 'long' / 'short' -> Pass
    x0: int
    y0: int
    shape: tuple                    # crop (H, W)
    path: Path
    meta: dict = field(default_factory=dict)
    legacy: bool = False
    subset: str = "all"
    cam_zone: Optional[np.ndarray] = None

    @property
    def long(self) -> Pass:
        return self.passes["long"]

    @property
    def short(self) -> Pass:
        return self.passes["short"]

    @property
    def ratio(self) -> float:
        return float(self.long.exposure_us) / float(self.short.exposure_us)

    def hdr(self, level_on: Optional[bool] = None, level: Optional[int] = None) -> np.ndarray:
        """long where unsaturated, else short x exposure ratio, of one flood.
        level_on True / False picks the bright / ambient flood (measured)."""
        L, S = self.long, self.short
        if level is None:
            lv_l = L.on_level if level_on else L.off_level
            lv_s = S.on_level if level_on else S.off_level
        else:
            lv_l = lv_s = level
        il, is_ = L.flood(lv_l), S.flood(lv_s)
        if il is None or is_ is None:
            raise ValueError(f"flood {lv_l} is needed in both passes")
        a = np.asarray(L.frames[il], float)
        b = np.asarray(S.frames[is_], float)
        return np.where(a < SAT, a, b * self.ratio)

    def pattern_signature(self) -> dict:
        """What the finders saw (a null calibration is only valid for the same)."""
        out = {}
        for nm, P in self.passes.items():
            fams = sorted({(p["kind"], p.get("theta_deg"), p.get("period")) for p in P.specs
                           if p["kind"] in ("sinus", "lines_v", "lines_h")},
                          key=lambda t: (t[0], t[1] if t[1] is not None else -1, t[2] or 0))
            names = [k + ("" if th is None else f"{th}") + ("" if T is None else f"_T{T:g}")
                     for k, th, T in fams]
            out[nm] = {"floods": sorted(int(p["level"]) for p in P.specs if p["kind"] == "flood"),
                       "families": names}
        return out


# ------------------------------------------------------------ loading

def is_dentset_dir(path) -> bool:
    p = Path(path)
    return (p / "dataset.json").is_file() or (p / "meta.json").is_file() and \
        _read(p / "meta.json").get("kind") == "dentset"


def _read(f: Path) -> dict:
    try:
        d = json.loads(Path(f).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return d if isinstance(d, dict) else {}


def read_meta(path) -> dict:
    """meta.json (engine entry) or dataset.json (legacy) of a dent set folder."""
    p = Path(path)
    m = _read(p / "meta.json")
    if m.get("kind") == "dentset":
        return m
    m = _read(p / "dataset.json")
    if m:
        m = dict(m)
        m["legacy"] = True
        return m
    raise ValueError(f"{p} is not a dent set (no meta.json of kind 'dentset' and no dataset.json)")


def load(path, subset: str = "all") -> DentData:
    """Read a dent set folder (engine entry or legacy) -> DentData, frames
    restricted to `subset` per pass (see module doc)."""
    p = Path(path)
    meta = read_meta(p)
    legacy = bool(meta.get("legacy"))
    raw = []
    for q in meta["passes"]:
        f = p / Path(q["file"]).name
        if not f.is_file():
            raise ValueError(f"dent set {p.name}: frames file {f.name} is missing")
        specs = q.get("patterns") or meta.get("patterns")
        if not specs:
            raise ValueError(f"dent set {p.name}: no pattern list for pass {q.get('preset')}")
        raw.append((q, f, specs))
    if len(raw) != 2:
        raise ValueError(f"dent set {p.name}: needs exactly two passes (long + short exposure), "
                         f"found {len(raw)}")
    raw.sort(key=lambda t: -float(t[0]["exposure_us"]))
    if float(raw[0][0]["exposure_us"]) <= float(raw[1][0]["exposure_us"]):
        raise ValueError(f"dent set {p.name}: the two passes have the same exposure")
    for k in RADIOMETRIC_KEYS:          # recorded since capture_dent_set checks them (legacy: not)
        a, b = raw[0][0].get(k), raw[1][0].get(k)
        if a is not None and b is not None and a != b:
            raise ValueError(f"dent set {p.name}: the passes differ in {k} ({a} vs {b}); the HDR "
                             "composite scales the short pass by the exposure ratio only")
    crops = {tuple(int(v) for v in q["crop_x0y0x1y1"]) for q, _, _ in raw}
    if len(crops) != 1:
        raise ValueError(f"dent set {p.name}: the passes were cropped differently {sorted(crops)}")
    x0, y0, x1, y1 = crops.pop()
    passes = {}
    for which, (q, f, specs) in zip(("long", "short"), raw):
        F = np.load(f, mmap_mode="r")
        if F.shape[0] != len(specs):
            raise ValueError(f"dent set {p.name}: {f.name} holds {F.shape[0]} frames for "
                             f"{len(specs)} patterns")
        keep = select(specs, which, subset)
        P = Pass(which, np.asarray(F[keep], np.float32), [dict(specs[i]) for i in keep],
                 float(q["exposure_us"]), str(q.get("preset", "")), int(q.get("average", 1)))
        P.set_polarity()
        passes[which] = P
    shape = passes["long"].frames.shape[1:]
    if passes["short"].frames.shape[1:] != shape:
        raise ValueError(f"dent set {p.name}: the passes have different frame sizes")
    cz = None
    if (p / "cam_zone_mask.npy").is_file():
        cz = np.load(p / "cam_zone_mask.npy").astype(bool)
        if cz.shape != shape:
            cz = None
    return DentData(passes=passes, x0=int(x0), y0=int(y0), shape=tuple(shape), path=p, meta=meta,
                    legacy=legacy, subset=subset, cam_zone=cz)

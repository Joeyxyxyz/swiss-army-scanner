"""
Presets (§6): one JSON per rig geometry, holding ALL tunables — nothing is
hard-coded in the scan path and nothing needs sed-editing.

`load_preset` validates every field (unknown keys are errors, so typos
can't silently fall back to defaults), runs the §9 config rules, and binds
the latest calibration stored for the preset's geometry_id.

Mock mode runs the same preset against the simulated bench, but under
geometry id "mock-<geometry_id>" so mock bases/calibrations can never be
mixed with real ones (rules 5/6).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional

from . import rules
from .bases import registry
from .results import Calibration

PRESETS_DIR = Path(__file__).resolve().parents[1] / "presets"
MOCK_PREFIX = "mock-"
_ID_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")
GAIN_DB_MAX = 48.0          # Blackfly S (IMX273) analog gain range is 0 .. ~48 dB

# HDMI video mode: the DLPC350 builds every 60 Hz frame from sequential R/G/B
# LED segments and PWM bit-planes, and the camera free-runs on its own clock.
# An exposure that is not a whole number of video frames catches a different
# slice of that sequence each frame, so gray levels flicker frame to frame.
VIDEO_FRAME_US = 1e6 / 60.0      # EVM HDMI video frame (60 Hz)
FRAME_LOCK_TOL = 0.005           # |exposure - n x frame| <= 0.5 % of a frame counts as locked


def frame_locked(exposure_us, frame_us: float = VIDEO_FRAME_US) -> bool:
    """True when the exposure is a whole number (>= 1) of video frames, so
    every frame integrates the full R/G/B + bit-plane sequence."""
    n = round(float(exposure_us) / frame_us)
    return n >= 1 and abs(float(exposure_us) - n * frame_us) <= FRAME_LOCK_TOL * frame_us


def valid_id(s) -> bool:
    """A preset name / geometry_id: one path-safe segment, never '.' or '..'."""
    return isinstance(s, str) and bool(_ID_RE.match(s)) and s not in (".", "..")


class ConfigError(ValueError):
    pass


@dataclass
class CameraCfg:
    exposure_us: int = 40000
    gain_db: float = 0.0
    average: int = 1          # frames averaged per pattern (noise ~ 1/sqrt(N))
    mono: bool = True


@dataclass
class ProjectionCfg:
    transport: str = "hdmi"
    monitor_offset: tuple = (1920, 0)   # EVM display top-left in the desktop
    size: tuple = (912, 1140)           # EVM display mode (W, H)
    settle_s: float = 0.0
    min_hold_s: float = 0.35            # pumped hold floor (bench fix, 0.5.1)
    warmup_s: float = 0.5               # window start-up hold before the first pattern
    scheme: str = "bw"
    gamma: float = 1.0                  # projector gamma precompensation; 1 = off
    duty: float = 1.0                   # DMD duty as software ND (v10: 0.55)
    invert: bool = False                # bench HDMI path is polarity-inverted
    rotate90: bool = False              # DMD portrait rotation (bench fix, 0.5.1)
    push_twice: bool = True             # DMD shows HDMI one frame behind
    led: Optional[dict] = None          # GUI-set currents, validated vs thermal limits


PROJECTOR_SIDES = ("left", "right", "above", "below")


@dataclass
class RigCfg:
    """Physical rig geometry — what the operator measures with a tape.

    projector_side: where the projector sits relative to the camera, seen
    from behind the camera looking at the scene. It decides which camera
    axis carries depth: left/right baseline -> depth shifts are horizontal
    in the image, so the fringes must be VERTICAL in the camera image
    (check_signal verifies this and says when to toggle rotate90).
    """
    projector_side: str = "right"
    baseline_mm: Optional[float] = None       # camera-to-projector lens distance
    standoff_mm: Optional[float] = None       # projection distance: rig to base/wall
    throw_ratio: float = 1.4                  # DLP LightCrafter 4500, DLPU011E Table 1-1

    @property
    def coded_axis(self) -> str:
        """Camera image axis the decoded coordinate must vary along."""
        return "x" if self.projector_side in ("left", "right") else "y"

    @property
    def nominal_mm_per_relief_px(self) -> Optional[float]:
        """Pre-calibration ESTIMATE of height per projector-column of relief:
        standoff^2 / (throw_ratio * 912 * baseline). Small-angle geometry,
        uncalibrated — for orientation only, never reported as a measurement."""
        if not (self.baseline_mm and self.standoff_mm and self.throw_ratio):
            return None
        return self.standoff_mm ** 2 / (self.throw_ratio * 912 * self.baseline_mm)

    @property
    def triangulation_angle_deg(self) -> Optional[float]:
        if not (self.baseline_mm and self.standoff_mm):
            return None
        import math
        return math.degrees(math.atan2(self.baseline_mm, self.standoff_mm))


@dataclass
class DecodeCfg:
    conf_min: float = 8.0               # modulation floor for a valid pixel
    contrast_min: float = 8.0           # white-black floor (shadow mask)


@dataclass
class ToleranceCfg:
    ch_fail_mm: float = 0.7
    dent_fail_um: float = 25.0
    min_defect_px: int = 25


@dataclass
class Preset:
    name: str
    geometry_id: str            # effective id (mock-prefixed in mock mode)
    rig_geometry_id: str        # as written in the preset
    camera: CameraCfg
    projection: ProjectionCfg
    bases: list
    decode: DecodeCfg
    tolerance: ToleranceCfg
    rig: RigCfg = field(default_factory=RigCfg)
    calibration: Optional[Calibration] = None
    mock: bool = False
    warnings: list = field(default_factory=list)
    source: Optional[Path] = None

    def basis_param(self, name: str, key: str, default=None):
        for b in self.bases:
            if b["name"] == name and key in b:
                return b[key]
        return default


def _section(cls, raw, section: str):
    if raw is None:
        return cls()
    if not isinstance(raw, dict):
        raise ConfigError(f"{section} must be an object")
    names = {f.name for f in fields(cls)}
    extra = set(raw) - names
    if extra:
        raise ConfigError(f"{section}: unknown keys {sorted(extra)}")
    return cls(**raw)


def _positive(v, what, allow_zero=False):
    if not isinstance(v, (int, float)) or v < 0 or (v == 0 and not allow_zero):
        raise ConfigError(f"{what} must be {'>= 0' if allow_zero else '> 0'}, got {v!r}")


def preset_from_dict(d: dict, *, mock: bool = False,
                     source: Optional[Path] = None) -> Preset:
    top = {"name", "geometry_id", "camera", "projection", "bases", "decode",
           "calibration", "tolerance", "rig"}
    extra = set(d) - top
    if extra:
        raise ConfigError(f"preset: unknown keys {sorted(extra)}")
    for k in ("name", "geometry_id"):
        if not valid_id(d.get(k)):
            raise ConfigError(f"preset.{k} must be a non-empty id [A-Za-z0-9_.-] "
                              "(not '.' or '..')")

    proj_raw = d.get("projection") or {}
    rules.validate_projection(proj_raw)           # rule 1, before anything else
    cam = _section(CameraCfg, d.get("camera"), "camera")
    proj = _section(ProjectionCfg, proj_raw, "projection")
    dec = _section(DecodeCfg, d.get("decode"), "decode")
    tol = _section(ToleranceCfg, d.get("tolerance"), "tolerance")
    rig = _section(RigCfg, d.get("rig"), "rig")
    if rig.projector_side not in PROJECTOR_SIDES:
        raise ConfigError(f"rig.projector_side must be one of {PROJECTOR_SIDES}")
    for k in ("baseline_mm", "standoff_mm"):
        if getattr(rig, k) is not None:
            _positive(getattr(rig, k), f"rig.{k}")
    _positive(rig.throw_ratio, "rig.throw_ratio")

    _positive(cam.exposure_us, "camera.exposure_us")
    if not isinstance(cam.gain_db, (int, float)) or isinstance(cam.gain_db, bool) or \
            not (0 <= cam.gain_db <= GAIN_DB_MAX):
        raise ConfigError(f"camera.gain_db must be 0..{GAIN_DB_MAX:g} dB, got {cam.gain_db!r}")
    if not isinstance(cam.average, int) or cam.average < 1:
        raise ConfigError("camera.average must be an int >= 1")
    _positive(proj.gamma, "projection.gamma")
    if not (0 < proj.duty <= 1):
        raise ConfigError("projection.duty must be in (0, 1]")
    _positive(proj.settle_s, "projection.settle_s", allow_zero=True)
    _positive(proj.min_hold_s, "projection.min_hold_s", allow_zero=True)
    _positive(proj.warmup_s, "projection.warmup_s", allow_zero=True)
    for k in ("monitor_offset", "size"):
        v = getattr(proj, k)
        if len(v) != 2 or not all(isinstance(x, int) for x in v):
            raise ConfigError(f"projection.{k} must be two ints")
        setattr(proj, k, tuple(v))
    rules.validate_led(proj.led)                  # rule 3
    warnings = rules.scheme_warnings(cam.mono, proj.scheme)   # rule 4
    if not mock:
        warnings += _real_rig_warnings(proj)
        warnings += _exposure_warnings(cam)
    _positive(dec.conf_min, "decode.conf_min", allow_zero=True)
    _positive(dec.contrast_min, "decode.contrast_min", allow_zero=True)
    _positive(tol.ch_fail_mm, "tolerance.ch_fail_mm")
    _positive(tol.dent_fail_um, "tolerance.dent_fail_um")

    bases = d.get("bases") or []
    if not bases:
        raise ConfigError("preset.bases must list at least one pattern basis")
    for b in bases:
        try:
            registry.validate_spec(b)
        except ValueError as e:
            raise ConfigError(str(e)) from None
    names = {b.get("name") for b in bases}
    if "hybrid" in names or {"graycode", "phaseshift"} <= names:
        # construct the decoder now so a bad combination (e.g. coarse_bits too
        # coarse for the fringe period) fails at load, not mid-capture
        from .bases import hybrid
        try:
            hybrid.from_preset_bases(bases)
        except (ValueError, NotImplementedError) as e:
            raise ConfigError(f"bases: {e}") from None
    if "multifreq" in names:
        from .bases import multifreq
        try:
            multifreq.from_spec(next(b for b in bases if b.get("name") == "multifreq"))
        except ValueError as e:
            raise ConfigError(f"bases: {e}") from None

    cal_raw = d.get("calibration") or {}
    extra = set(cal_raw) - {"id", "mm_per_px", "intercept_mm", "lateral_px_per_mm"}
    if extra:
        raise ConfigError(f"calibration: unknown keys {sorted(extra)}")

    rig_geo = d["geometry_id"]
    geo = MOCK_PREFIX + rig_geo if mock else rig_geo
    calibration = None
    if cal_raw.get("mm_per_px") is not None or cal_raw.get("lateral_px_per_mm") is not None:
        if mock:
            warnings.append("preset calibration is for the real rig; ignored in mock mode")
        else:
            calibration = Calibration(
                id=cal_raw.get("id") or f"preset:{d['name']}", geometry_id=geo,
                mm_per_px=cal_raw.get("mm_per_px"),
                intercept_mm=cal_raw.get("intercept_mm") or 0.0,
                lateral_px_per_mm=cal_raw.get("lateral_px_per_mm"), source="preset")

    preset = Preset(name=d["name"], geometry_id=geo, rig_geometry_id=rig_geo,
                    camera=cam, projection=proj, bases=[dict(b) for b in bases],
                    decode=dec, tolerance=tol, rig=rig, calibration=calibration, mock=mock,
                    warnings=warnings, source=source)
    if calibration is not None:
        calibration.fingerprint = geometry_fingerprint(preset)
    return preset


def _real_rig_warnings(proj: ProjectionCfg) -> list[str]:
    out = []
    if not proj.led:
        out.append("projection.led is not recorded, so rule 3 (LED thermal limits) can't be "
                   "checked: record the LightCrafter GUI currents here (mode 'continuous' "
                   "for scans and live view)")
    if proj.invert and proj.duty < 1:
        floor = 255 - int(round(255 * proj.duty))
        out.append(f"projection.invert with duty {proj.duty:g} sends every pattern as "
                   f"{floor}..255: the bench DMD shows what is sent, so the dark floor is "
                   f"{1 - proj.duty:.0%} of full light while the peak stays at full. Fringe "
                   f"modulation is {proj.duty:g}x of duty 1.0 at the same exposure. Use "
                   "duty 1.0 (e.g. preset cell_wall_2p7ft_vstripe_fullrange) and set the "
                   "exposure with tune")
    return out


def _exposure_warnings(cam: CameraCfg) -> list[str]:
    """Warn (never refuse) when the exposure is not a whole number of HDMI
    video frames: the free-running camera then integrates a different slice
    of the DMD's R/G/B + PWM bit-plane sequence every frame."""
    if frame_locked(cam.exposure_us):
        return []
    one, two = round(VIDEO_FRAME_US), round(2 * VIDEO_FRAME_US)
    return [f"camera.exposure_us {cam.exposure_us:g} is not a whole number of {one} us HDMI "
            "video frames (60 Hz): the DLPC350 draws each frame as sequential R/G/B LED "
            "segments and PWM bit-planes and the camera free-runs, so every frame catches a "
            "different slice of that sequence and gray levels (every phase-shift fringe) and "
            f"colour floods flicker frame to frame. Use {one} or {two} us, with less light if "
            "that clips (close the iris / stop the lens down, or a lower projection.duty - "
            "duty only dims the peak with projection.invert off); run tune to check. "
            "camera.average only partly evens the flicker out"]


def basis_axis(bases: list) -> str:
    """Camera-side coded axis of the preset's structured-light basis ('x' | 'y');
    same precedence as bases.hybrid.from_preset_bases."""
    by = {b.get("name"): b for b in bases}
    for name in ("hybrid", "graycode", "phaseshift", "multifreq"):
        if name in by and "axis" in by[name]:
            return by[name]["axis"]
    return "x"


def geometry_fingerprint(p: Preset) -> dict:
    """The settings that define what a decoded correspondence (and so a base,
    relief or calibration) means at this geometry_id. Stored with bases,
    objects and calibrations and compared by rules.fingerprint_diffs, so a
    preset edited in place (same geometry_id) can't silently reuse data
    captured before the edit. Camera, duty, invert, gamma and decode floors
    are not geometry: decode measures polarity and relief is in DMD px."""
    return {"rotate90": bool(p.projection.rotate90),
            "size": [int(v) for v in p.projection.size],
            "basis_axis": basis_axis(p.bases),
            "projector_side": p.rig.projector_side,
            "baseline_mm": p.rig.baseline_mm,
            "standoff_mm": p.rig.standoff_mm}


def preset_to_dict(p: Preset) -> dict:
    """The JSON form of a preset (what preset_from_dict reads)."""
    from dataclasses import asdict
    cal = p.calibration if (p.calibration is not None and p.calibration.source == "preset") else None
    return {
        "name": p.name, "geometry_id": p.rig_geometry_id,
        "camera": asdict(p.camera),
        "projection": {**asdict(p.projection), "monitor_offset": list(p.projection.monitor_offset),
                       "size": list(p.projection.size)},
        "bases": [dict(b) for b in p.bases], "decode": asdict(p.decode),
        "rig": asdict(p.rig),
        "calibration": {"id": cal.id if cal else None, "mm_per_px": cal.mm_per_px if cal else None,
                        "intercept_mm": cal.intercept_mm if cal else 0.0,
                        "lateral_px_per_mm": cal.lateral_px_per_mm if cal else None},
        "tolerance": asdict(p.tolerance),
    }


def save_preset(d: dict, *, presets_dir: Optional[Path] = None, overwrite: bool = False) -> Path:
    """Validate and write a preset JSON. Refuses to replace an existing preset
    unless overwrite=True (presets are config, not library data — but a silent
    replace would change the meaning of every scan bound to it)."""
    new = preset_from_dict(d)                            # full validation, rules included
    folder = Path(presets_dir or PRESETS_DIR)
    path = folder / f"{d['name']}.json"
    if path.exists() and not overwrite:
        raise ConfigError(f"preset {path.name} exists; save under a new name or pass overwrite")
    _refuse_geometry_reuse(new, folder)
    with open(path, "w" if overwrite else "x", encoding="utf-8") as f:
        json.dump(d, f, indent=2)
    return path


def _refuse_geometry_reuse(new: Preset, folder: Path) -> None:
    """Rules 5/6 at save time: a geometry_id names ONE geometry. Refuse a
    preset (an in-place overwrite, or a new name reusing the id) whose
    geometry-bound settings differ from a saved preset with the same id —
    bases and calibrations stored under that id would silently be reused."""
    want = geometry_fingerprint(new)
    for f in sorted(folder.glob("*.json")):
        try:
            other = preset_from_dict(json.loads(f.read_text(encoding="utf-8")), source=f)
        except Exception:                       # noqa: BLE001 - a broken file is not ours to judge
            continue
        if other.rig_geometry_id != new.rig_geometry_id:
            continue
        diffs = rules.fingerprint_diffs(geometry_fingerprint(other), want)
        if diffs:
            what = "the saved version" if other.name == new.name else f"preset {other.name}"
            raise ConfigError(
                f"geometry_id {new.rig_geometry_id} is already used by {what} with other "
                f"geometry-bound settings ({'; '.join(diffs)}). Bases and calibrations are "
                "only valid at the geometry they were captured at: save with a new geometry_id.")


def list_presets(presets_dir: Optional[Path] = None) -> list[str]:
    return sorted(p.stem for p in Path(presets_dir or PRESETS_DIR).glob("*.json"))


def resolve_preset_path(name: str, presets_dir: Optional[Path] = None) -> Path:
    p = Path(name)
    if p.suffix == ".json" or p.exists():
        return p
    return Path(presets_dir or PRESETS_DIR) / f"{name}.json"


def load_preset(name: str, *, presets_dir: Optional[Path] = None, store=None,
                mock: bool = False) -> Preset:
    path = resolve_preset_path(name, presets_dir)
    if not path.exists():
        raise ConfigError(f"preset not found: {path}")
    with open(path, encoding="utf-8") as f:
        preset = preset_from_dict(json.load(f), mock=mock, source=path)
    if store is not None:
        cal = store.latest_calibration(preset.geometry_id,
                                       fingerprint=geometry_fingerprint(preset))
        if cal is not None:                        # library beats preset values
            preset.calibration = cal
    return preset

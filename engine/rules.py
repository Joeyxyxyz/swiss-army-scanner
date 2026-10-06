"""
Safety invariants, enforced at runtime (ENGINE_ARCHITECTURE §9).

These are checks the engine runs on itself so that NO caller — UI, CLI or
AI agent — can violate them. They are not advisory: every violation raises.

  1. Scan path is HDMI-only. The DLPC350 USB driver is never imported or
     driven in the scan path. A USB display-mode command (set_mode("video")
     etc.) forces source re-acquisition, the 912x1140 HDMI timing fails to
     re-lock and the DMD drops to its internal blue-lines pattern.
  2. The projector window stays painted during holds (pumped waitKey, no
     bare sleeps) — enforced structurally in capture/projector.py and by
     tests/test_rules.py.
  3. LED thermal limits (config validation).
  4. Mono camera => bw scheme; colour schemes are flagged.
  5. Base and object share geometry_id.
  6. Calibration is geometry-bound; mm only with a matching calibration.
  7. Library data is never overwritten.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Optional


class RuleViolation(RuntimeError):
    """A §9 invariant was about to be broken. The operation did not run."""


class SafetyViolation(RuleViolation):
    """Rule that protects the hardware / the scan (rules 1-3)."""


class GeometryMismatch(RuleViolation):
    """Rule 5 / 6: data from different rig geometries mixed."""


class OverwriteRefused(RuleViolation):
    """Rule 7: an existing library path would have been overwritten."""


class RigBusy(RuleViolation):
    """Another process is driving the camera + DMD (capture/riglock.py)."""


class Cancelled(RuntimeError):
    """The operator aborted the job (ScanSession.cancel was set). Checked between
    captures: the rig is closed (projector black / idle, rig lock released) on
    the way out, and nothing half-measured is reported."""


class EntryInUse(RuleViolation):
    """Rule 7: a library entry could not be moved to trash/ in one atomic
    rename (a file in it is open). Nothing was moved; retry once it is free."""


# ------------------------------------------------------------ rule 1: HDMI

# Modules that must never be loaded in a scan process. Importing the
# swiss_army_sensor package (or its drivers subpackage) executes
# drivers/dlp_driver.py, which is why the engine loads seed modules
# file-by-file (engine/_seed.py).
FORBIDDEN_SCAN_MODULES = (
    "swiss_army_sensor.drivers.dlp_driver",
    "hid",                      # hidapi — the DLPC350 USB-HID transport
)

# DLPC350 USB commands that knock the DMD off HDMI lock. The Projector
# refuses these by name, so a caller "trying" gets a clear error instead
# of a dead scan.
USB_DISPLAY_COMMANDS = frozenset({
    "set_mode", "set_display_mode", "set_pattern_source",
    "set_pattern_trigger_mode", "set_trigger_out1", "load_pattern_sequence",
    "start_sequence", "stop_sequence", "set_led_currents", "usb", "hid",
})

ALLOWED_TRANSPORTS = ("hdmi",)
_FORBIDDEN_PROJECTION_KEYS = {"usb", "dlp_usb", "set_mode", "display_mode",
                              "dlp_mode", "pattern_mode", "hid"}


def check_scan_path_usb_free(modules: Optional[dict] = None) -> None:
    """Raise if the DLPC350 USB driver (or hidapi) is loaded in this process."""
    mods = sys.modules if modules is None else modules
    loaded = [m for m in FORBIDDEN_SCAN_MODULES if m in mods]
    if loaded:
        raise SafetyViolation(
            f"DLPC350 USB driver loaded in the scan process ({', '.join(loaded)}). "
            "The scan path is HDMI-only: a USB display-mode command drops the DMD "
            "to its blue-lines test pattern. Run scans in a process that does not "
            "import swiss_army_sensor.drivers / hid (the engine never does).")


def check_transport(transport) -> None:
    """Only the engine's own HDMI / mock windows may carry patterns."""
    cls_names = [c.__name__ for c in type(transport).__mro__]
    if any("DLP" in n for n in cls_names):
        raise SafetyViolation(
            f"{type(transport).__name__} is a DLPC350 driver — patterns reach the "
            "DMD over HDMI only, never over USB.")
    for attr in USB_DISPLAY_COMMANDS:
        if attr in {"usb", "hid"}:
            continue
        if callable(getattr(type(transport), attr, None)):
            raise SafetyViolation(
                f"transport {type(transport).__name__} exposes USB command "
                f"'{attr}'; not allowed in the scan path.")
    if not callable(getattr(transport, "show", None)) or \
            not callable(getattr(transport, "hold", None)):
        raise SafetyViolation("transport must provide show() and a pumped hold()")


def refuse_usb_command(name: str):
    raise SafetyViolation(
        f"'{name}' is a DLPC350 USB command. The scan path is HDMI-only; "
        "USB display-mode commands knock the DMD off 912x1140 HDMI lock "
        "(blue-lines pattern, dead scan). Hardware-trigger pattern mode will be "
        "a separate, explicitly-entered mode (build step 10), never the default.")


def validate_projection(proj: dict) -> None:
    bad = _FORBIDDEN_PROJECTION_KEYS & set(proj)
    if bad:
        raise SafetyViolation(
            f"projection config has USB/DLP-mode keys {sorted(bad)}; the scan "
            "path is HDMI-only.")
    t = proj.get("transport", "hdmi")
    if t not in ALLOWED_TRANSPORTS:
        raise SafetyViolation(f"projection.transport={t!r}; only 'hdmi' is allowed")


# ------------------------------------------------------ rule 3: LED limits

LED_BURST_MAX = {"r": 180, "g": 255, "b": 230}
LED_AVG_MAX = {"r": 120, "g": 180, "b": 160}


def validate_led(led: Optional[dict]) -> None:
    """LED currents as set in the LightCrafter GUI, recorded in the preset.

    mode 'burst'      -> short-burst ceiling R=180 G=255 B=230
    mode 'continuous' -> running-average limit R=120 G=180 B=160
    """
    if not led:
        return
    mode = led.get("mode", "continuous")
    if mode not in ("burst", "continuous"):
        raise SafetyViolation(f"led.mode must be 'burst' or 'continuous', not {mode!r}")
    limits = LED_BURST_MAX if mode == "burst" else LED_AVG_MAX
    for ch, lim in limits.items():
        v = led.get(ch)
        if v is None:
            continue
        if not (0 <= v <= lim):
            raise SafetyViolation(
                f"LED {ch.upper()}={v} exceeds the {mode} thermal limit {lim} "
                f"(burst R180/G255/B230, average R120/G180/B160)")
    extra = set(led) - {"mode", "r", "g", "b"}
    if extra:
        raise SafetyViolation(f"unknown led keys {sorted(extra)}")


# The longest continuous projection the 'burst' ceiling is taken to cover.
# The bench notes give no number ("not meant for sustained duration"), so
# this is deliberately short: a check_signal-sized burst, not a scan.
LED_BURST_WINDOW_S = 10.0


def check_led_duration(led: Optional[dict], seconds: Optional[float], what: str) -> None:
    """Rule 3 over time: 'burst' currents above the running-average limit are
    only allowed for projections no longer than LED_BURST_WINDOW_S.

    `seconds` None means sustained / unknown length (a full scan, a live
    view, a long hold), which always gets the average limits."""
    if not led or led.get("mode", "continuous") != "burst":
        return
    if seconds is not None and seconds <= LED_BURST_WINDOW_S:
        return
    over = [f"{ch.upper()}={led[ch]}" for ch, lim in LED_AVG_MAX.items()
            if led.get(ch) is not None and led[ch] > lim]
    if over:
        span = "sustained" if seconds is None else f"~{seconds:.0f} s"
        raise SafetyViolation(
            f"{what} projects for {span}, but the preset's LED currents ({', '.join(over)}) "
            f"are only within the short-burst ceiling. Past {LED_BURST_WINDOW_S:.0f} s the "
            "running-average limit R120/G180/B160 applies: lower the currents in the "
            "LightCrafter GUI and record them in projection.led with mode 'continuous'.")


# ---------------------------------------------------- rule 4: scheme / mono

def scheme_warnings(mono: bool, scheme: str) -> list[str]:
    if scheme not in ("bw", "red", "green", "blue", "rgb"):
        raise RuleViolation(f"unknown projection.scheme {scheme!r}")
    if mono and scheme != "bw":
        return [f"scheme '{scheme}' on a MONO camera reduces signal; 'bw' is "
                "maximum contrast (bench-confirmed)"]
    return []


# ---------------------------------------------------- rules 5/6: geometry

def require_same_geometry(a: str, b: str, what: str) -> None:
    if a != b:
        raise GeometryMismatch(
            f"{what}: geometry '{a}' != '{b}'. A base / calibration is valid only "
            "at the geometry it was captured at — re-capture the base (and "
            "recalibrate) after moving the rig.")


def calibration_matches(calibration, geometry_id: str) -> bool:
    return calibration is not None and calibration.geometry_id == geometry_id


def fingerprint_diffs(recorded: Optional[dict], current: Optional[dict]) -> list[str]:
    """Geometry-bound settings (config.geometry_fingerprint) that differ.

    geometry_id is the operator's statement of the geometry; the fingerprint
    catches the same id reused after the settings that define it changed
    (rotate90, projection size, coded axis, rig side / baseline / standoff).
    A key missing on either side, or a None (unknown) value, is not a
    difference: data written before fingerprints existed stays usable."""
    if not recorded or not current:
        return []
    out = []
    for k in sorted(set(recorded) & set(current)):
        a, b = recorded[k], current[k]
        if a is None or b is None:
            continue
        if isinstance(a, (int, float)) and isinstance(b, (int, float)) \
                and not isinstance(a, bool) and not isinstance(b, bool):
            same = abs(float(a) - float(b)) <= 1e-6 * max(1.0, abs(float(a)), abs(float(b)))
        else:
            same = (list(a) if isinstance(a, (list, tuple)) else a) == \
                   (list(b) if isinstance(b, (list, tuple)) else b)
        if not same:
            out.append(f"{k} {a!r} -> {b!r}")
    return out


def require_same_fingerprint(recorded: Optional[dict], current: Optional[dict],
                             what: str) -> None:
    diffs = fingerprint_diffs(recorded, current)
    if diffs:
        raise GeometryMismatch(
            f"{what}: same geometry_id but geometry-bound settings changed "
            f"({'; '.join(diffs)}). Data captured before the change is not valid after "
            "it: give the changed rig a new geometry_id, then capture a new base "
            "(and recalibrate).")


# ------------------------------------------------------ rule 7: no overwrite

def ensure_new_path(path: Path) -> None:
    if Path(path).exists():
        raise OverwriteRefused(f"{path} exists; library data is never overwritten")


def ensure_all_new(paths: Iterable[Path]) -> None:
    for p in paths:
        ensure_new_path(p)

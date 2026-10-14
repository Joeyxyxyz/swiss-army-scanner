"""
Command-line front-end (spec §1 ui/cli.py, build step 8).

    python -m ui.cli [--mock] [--library PATH] [--preset NAME] <command> [...]
    scan.bat --mock wizard

Thin on purpose: every command calls engine.api verbs and prints the
dataclasses they return as tables, with units. Nothing here touches the
hardware or writes library files itself, so every §9 rule the engine
enforces also covers this CLI. Physical steps stay human-only: the
wizard PROMPTS for them and waits. With --mock, the simulated bench
stands in for the human.

No command ever waits for input unless --interactive is given: a command
that needs an answer (delete, teach-presence) takes it as an argument
(--yes, --empty-confirmed) and is refused (exit 2) without it; the wizard,
which is nothing but prompts, needs --interactive. With --interactive
every prompt gives up after --prompt-timeout seconds and the command
stops with nothing done (exit 1): a script, an agent or a detached
terminal can never leave a command hanging on a question nobody sees.
(`zone-paint` is not a prompt but a drawing session on the projector: the
operator draws zones on the part with the mouse and ends it there with
Space / Esc; it gives the rig back after --idle-timeout seconds without
input, nothing saved. With --mock it replays a scripted session.)

The zone workflow is the reason the grid exists. `grid` projects a lettered
grid and photographs it, a person or an AI agent reads the photo, and
`scan part --zone C4:D5` scans only the cells the part sits in.

Exit codes let a script or an agent branch without parsing the text:
  0  ok (PASS, signal ok, codes found, spot found, PRESENT, ALIGNED, ...)
  1  not ok: signal not ok, FAIL/FLAG/INDETERMINATE verdict, ABSENT,
     NOT_ALIGNED, no code / spot found, base auto-rejected, cancelled, or
     an unexpected error
  2  refused: rule / geometry / safety / config violation, rig busy (another
     program holds the camera + DMD) or bad input
  3  the requested mode is not available yet (engine stub)
  4  dent-detect only: the detector itself failed (an unexpected error), so a
     crash is never read as FLAG (1) or CLEAR (0)
  5  run / recipe-run / quick-check / station-check / face-check: PULL (quality
     inspection) - the job ran but its result is in doubt (INVALID, INCOMPLETE, a
     guard band, an unreadable code, a size mismatch): never a pass, never a reject.
     face-locate: the face reads the wrong size (calibration or lens suspect);
     face-scan: no camera exposure fits this seating (the face is not scanned)
     measure-run: NOT MEASURABLE (no calibration, a mobile rig not verified, too
     few points ...): no number is given - never a px as mm
`run <station config>` is the flow-ready entry point: one job, a new run
folder with result.json + run.log.jsonl (ui/run_job.py). `recipe-run <recipe>` runs a
recipe (the check of a work state: zones, checks, projected verdict) the same way. Their exit code
is the disposition (engine/disposition.py): 0 CONTINUE, 1 REJECT, 5 PULL; refused 2,
crashed 4 and ABORTED 1 keep their codes (disposition PULL in result.json).
Operating modes (engine/modes/): `measure-run` = MEASURE (an instrument: numbers with
units and uncertainty, no verdict; exit 0 MEASURED, 5 NOT MEASURABLE), `quick-check` =
QUICK CHECK (a triangulation preset + one check -> CONTINUE / REJECT / PULL: 0 / 1 / 5),
`run` on a station configuration = STATION.
--json prints one JSON document per command instead of tables.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
import traceback
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Optional

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:          # `python ui/cli.py` from anywhere
    sys.path.insert(0, str(_ROOT))

from engine import api, config, rules  # noqa: E402
from engine import disposition as D  # noqa: E402
from engine.library import presets as preset_lib  # noqa: E402
from engine.capture.hwtrigger import TRIGGER_LINES  # noqa: E402

DEFAULT_PRESET = "cell_wall_2p7ft_vstripe"   # fringes vertical in camera, projector right
TRIGGER_LINE_HELP = ("camera input the trigger is wired to: Line0 = opto-isolated "
                     "input (GPIO pin 2), Line3 = non-isolated input (pin 1); never "
                     "pin 3 (Line2, the 3.3 V auxiliary output)")
EXIT_OK, EXIT_NOT_OK, EXIT_REFUSED, EXIT_NOT_AVAILABLE = 0, 1, 2, 3
EXIT_ERROR = 4                  # an unexpected error, for commands whose exit 1 is a verdict
VALIDATION_DIMS = "106.95,56.5,21.9"        # §11 object, H,W,L mm

# Engine messages use typographic punctuation; a redirected Windows console
# (cp1252) cannot encode all of it, so human output is kept ASCII.
_ASCII = str.maketrans({"—": "-", "–": "-", "≈": "~", "±": "+/-",
                        "µ": "u", "²": "^2", "→": "->", "×": "x",
                        "−": "-", "≥": ">=", "≤": "<=", "°": " deg"})


class CLIError(Exception):
    """Bad input from the operator (unknown id, malformed value): exit 2."""


class Abort(Exception):
    """The operator stopped (q, EOF, answered no): exit 1."""


# ------------------------------------------------------------ formatting

def _one_line(text) -> str:
    return " ".join(str(text).split()).translate(_ASCII)


def _f(v, nd: int = 3) -> str:
    """A number without units; '-' for missing."""
    if v is None:
        return "-"
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.{nd}f}" if math.isfinite(v) else "nan"
    return str(v)


def _u(v, unit: str, nd: int = 3) -> str:
    """A number WITH its units (every numeric result carries units)."""
    return "-" if v is None else f"{_f(v, nd)} {unit}"


def _jsonable(o):
    """Dataclasses/arrays/paths -> JSON types. Big arrays become a shape
    summary: an agent reads numbers, not 1.5 Mpx maps."""
    if o is None or isinstance(o, (bool, int, str)):
        return o
    if isinstance(o, float):
        return o if math.isfinite(o) else None
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, np.generic):
        return _jsonable(o.item())
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist()) if o.size <= 16 else \
            {"array_shape": list(o.shape), "dtype": str(o.dtype)}
    if is_dataclass(o) and not isinstance(o, type):
        return {f.name: _jsonable(getattr(o, f.name)) for f in fields(o)
                if not f.name.startswith("_")}
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_jsonable(v) for v in o]
    return str(o)


def _get(item, key, default=None):
    """Field of a dataclass handle or a dict row (listings return both)."""
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _flatten(d: dict, prefix: str = "") -> list:
    out = []
    for k, v in (d or {}).items():
        if isinstance(v, dict):
            out += _flatten(v, f"{prefix}{k}.")
        else:
            out.append((f"{prefix}{k}", v))
    return out


def _bases_text(bases: list) -> str:
    return ", ".join(b["name"] + "(" + ",".join(f"{k}={v}" for k, v in b.items()
                                                  if k != "name") + ")" for b in bases)


def _dims_text(known: Optional[dict]) -> str:
    if not known:
        return "-"
    return " ".join(f"{k}{known[k]:g}" for k in ("H", "W", "L") if k in known) + " mm"


# ------------------------------------------------------------ parsing input

def _number(text: str, what: str) -> float:
    t = text.strip().lower()
    if t.endswith("mm"):
        t = t[:-2].strip()
    try:
        return float(t)
    except ValueError:
        raise CLIError(f"{what}: {text!r} is not a number") from None


_DIM_NAMES = {"h": "H", "height": "H", "w": "W", "width": "W", "l": "L", "length": "L"}


def parse_known(text: Optional[str]) -> Optional[dict]:
    """'106.95,56.5,21.9' | '106.95x56.5x21.9' | 'H=106.95,W=56.5' | '106.95,,'
    -> {'H': .., 'W': .., 'L': ..} in mm. Checked here so a typo is caught
    BEFORE a capture (the wizard asks again); the engine re-validates."""
    if text is None or not text.strip():
        return None
    t = text.strip().replace(";", ",")
    if "," not in t and "=" not in t and "x" in t.lower():
        t = re.sub(r"\s*[xX]\s*", ",", t)
    parts = [s.strip() for s in t.split(",")]
    what = "known dims (mm, H,W,L)"
    out = {}
    if any("=" in s for s in parts):
        for s in parts:
            if not s:
                continue
            if "=" not in s:
                raise CLIError(f"{what}: mix of 'key=value' and plain values in {text!r}")
            k, v = s.split("=", 1)
            key = _DIM_NAMES.get(k.strip().lower())
            if key is None:
                raise CLIError(f"{what}: unknown dimension {k.strip()!r}; use H, W or L")
            out[key] = _number(v, what)
    else:
        if len(parts) > 3:
            raise CLIError(f"{what}: at most 3 values (H,W,L), got {text!r}")
        for key, s in zip(("H", "W", "L"), parts):
            if s:
                out[key] = _number(s, what)
    if not out:
        raise CLIError(f"{what}: no value in {text!r}")
    bad = [k for k, v in out.items() if not (v > 0 and math.isfinite(v))]
    if bad:
        raise CLIError(f"{what}: {', '.join(bad)} must be a positive number of mm")
    return out


def _numbers(text: str, n: int, what: str, cast=float) -> list:
    parts = [s for s in re.split(r"[,\s]+", text.strip()) if s]
    if len(parts) != n:
        raise CLIError(f"{what} needs {n} comma-separated numbers, got {text!r}")
    try:
        return [cast(float(s)) if cast is int else cast(s) for s in parts]
    except ValueError:
        raise CLIError(f"{what}: {text!r} is not {n} numbers") from None


def _list_of(cast, what: str):
    def conv(text: str):
        try:
            return [cast(s) for s in re.split(r"[,\s]+", text.strip()) if s]
        except ValueError:
            raise argparse.ArgumentTypeError(f"{what}: {text!r} is not a comma list") from None
    return conv


def _finite_nonneg(text: str) -> float:
    try:
        v = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from None
    if not math.isfinite(v) or v < 0:
        raise argparse.ArgumentTypeError("must be a finite number >= 0")
    return v


def _positive_int(text: str) -> int:
    try:
        v = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not an integer") from None
    if v < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return v


CYL_DEFAULT = (24.05, 90.0, 1.5)        # radius, height, dent depth (mm): a standing cylinder


def _mock_part(text: str) -> str:
    if text.strip().lower().startswith("scene:"):
        from engine.capture import mock_scenes
        try:
            name, variant = mock_scenes.parse(text)
        except ValueError as e:
            raise argparse.ArgumentTypeError(str(e)) from None
        return f"scene:{name}:{variant}"
    if text.strip().lower().startswith("qr:") and text.strip()[3:].strip():
        return "qr:" + text.strip()[3:].strip()          # a QR label (codes), text kept as typed
    t = text.strip().lower()
    if t in ("validation", "none"):
        return t
    if t.startswith("cylinder") or t.startswith("cyl"):
        body = t.split(":", 1)[1] if ":" in t else ""
        try:
            vals = [float(v) for v in re.split(r"[,xX]", body) if v.strip()]
        except ValueError:
            vals = None
        if vals is None or len(vals) > 3 or any(v < 0 for v in vals) or \
                (vals and min(vals[:2]) <= 0):
            raise argparse.ArgumentTypeError(
                "use cylinder or cylinder:R,H[,dent_mm] (mm; dent 0 = no dent)")
        vals = vals + list(CYL_DEFAULT[len(vals):])
        return "cylinder:" + ",".join(f"{v:g}" for v in vals)
    try:
        vals = [float(s) for s in re.split(r"[,xX]", t) if s.strip()]
    except ValueError:
        vals = []
    if len(vals) not in (3, 4) or min(vals[:3]) <= 0 or \
            (len(vals) == 4 and not 0.0 <= vals[3] <= 1.0):
        raise argparse.ArgumentTypeError(
            "use 'validation', 'none' or H,W,L[,albedo] (mm; albedo 0..1, the bench is 0.6)")
    return ",".join(f"{v:g}" for v in vals)


_AT_POINT = re.compile(r"^(dmd|cam|camera)\s*[:=]?\s*\(?\s*(-?\d+(?:\.\d+)?)\s*[,\s]\s*"
                       r"(-?\d+(?:\.\d+)?)\s*\)?$", re.I)


def parse_at(text: str):
    """Where a spot / sensor goes. 'H8', 'H8@96' (cell of a 96 px grid) and
    zones 'G7:H9' go to the engine as text; 'dmd:456,570' -> {'dmd': [col, row]}
    (DMD px); 'cam:700,500' -> {'camera': [x, y]} (camera px, needs a map)."""
    t = (text or "").strip()
    if not t:
        raise CLIError("where? a grid cell (H8), a zone (G7:H9), dmd:COL,ROW or cam:X,Y")
    m = _AT_POINT.match(t)
    if m:
        pt = [float(m.group(2)), float(m.group(3))]
        return {"dmd": pt} if m.group(1).lower() == "dmd" else {"camera": pt}
    if re.match(r"^(dmd|cam|camera)\b", t, re.I):
        raise CLIError(f"{text!r}: write dmd:COL,ROW (DMD px) or cam:X,Y (camera px)")
    return t


def _kw(**kw) -> dict:
    """Only the options the operator gave: the engine keeps its own defaults."""
    return {k: v for k, v in kw.items() if v is not None}


# ------------------------------------------------------------ run context

class Ctx:
    """One CLI invocation: the engine session, the lazily loaded preset and
    the output-bin index position when the command started."""

    def __init__(self, args):
        self.args = args
        self.json = bool(args.json)
        self.mock = bool(args.mock)
        self.mode = args.mode
        self.session = None
        self._preset = None
        self._outputs_before = 0
        self.selection = None           # preset_lib.selection() when no --preset was given
        # prompts are OFF unless --interactive: no command waits for input by default
        self.interactive = bool(getattr(args, "interactive", False))
        self.prompt_timeout = getattr(args, "prompt_timeout", None)

    # -- output
    def say(self, *lines) -> None:
        if not self.json:
            for line in lines:
                print(str(line).translate(_ASCII))

    def warn(self, warnings, indent: str = "  ") -> None:
        for w in warnings or []:
            self.say(f"{indent}! {_one_line(w)}")

    def table(self, headers: list, rows: list, indent: str = "  ") -> None:
        if not rows:
            return
        rows = [[str(c) for c in r] for r in rows]
        widths = [max(len(r[i]) for r in [headers] + rows) for i in range(len(headers))]
        fmt = lambda r: (indent + "  ".join(c.ljust(w) for c, w in zip(r, widths))).rstrip()
        self.say(fmt(headers), indent + "  ".join("-" * w for w in widths), *map(fmt, rows))

    def kv(self, rows: list, indent: str = "  ") -> None:
        """Aligned 'key  value' lines (a record, not a table: no header)."""
        rows = [(str(k), str(v)) for k, v in rows]
        width = max((len(k) for k, _ in rows), default=0)
        for k, v in rows:
            self.say(f"{indent}{k.ljust(width)}  {v}".rstrip())

    # -- engine
    def start(self) -> None:
        lib = Path(self.args.library).resolve() if self.args.library else None
        self.session = api.configure(lib, mock=self.mock)
        self._outputs_before = len(self.session.store.outputs())
        if not getattr(self.args, "preset_given", True):
            self.use_selected_preset()

    def use_selected_preset(self) -> None:
        """No --preset: this session library's selected preset (preset-select), else
        DEFAULT_PRESET when none was ever selected. A selection that no longer resolves
        (deleted, renamed, a failed replace) is kept as the name, so loading it stops with a
        clear error instead of silently scanning with the default."""
        sel = preset_lib.selection(self.session.store,
                                   Path(self.session.presets_dir or config.PRESETS_DIR))
        self.selection = sel
        self.args.preset = sel["name"] or DEFAULT_PRESET
        if sel["skipped"]:
            self.say(f"  ! {sel['skipped']} unreadable line(s) skipped in "
                     f"{preset_lib.user_dir(self.session.store) / preset_lib.ACTIVE_FILE} "
                     "(not UTF-8 JSON); `scan preset-select NAME` appends a good one")

    def preset(self):
        if self._preset is None:
            sel = self.selection
            if sel and sel["name"] and not sel["exists"] and self.args.preset == sel["name"]:
                raise CLIError(_selection_gone(sel, self.session.store.root))
            p = api.load_preset(self.args.preset)
            if self.args.average is not None:       # this run only; the file is untouched
                p.camera.average = self.args.average
            self._preset = p
        return self._preset

    def header(self) -> None:
        p = self.preset()
        rig = "MOCK" if self.mock else "REAL RIG"
        self.say(f"[{rig}] preset {p.name} | geometry {p.geometry_id} | "
                 f"exposure {p.camera.exposure_us} us, average {p.camera.average} frames")
        self.warn([f"preset: {w}" for w in p.warnings])

    def new_outputs(self) -> list:
        root = self.session.store.root
        return [{"kind": r["kind"], "label": r["label"], "path": str(root / r["file"])}
                for r in self.session.store.outputs()[self._outputs_before:]]

    def report_outputs(self, title: str = "Output bin") -> list:
        """Print output-bin files written since the last report (once each)."""
        new = self.new_outputs()
        if new:
            self.say("", f"{title} ({len(new)} new):")
            self.table(["kind", "file"], [[o["kind"], o["path"]] for o in new])
            self._outputs_before += len(new)
        return new

    def meta(self) -> dict:
        d = {"mock": self.mock, "library": str(self.session.store.root) if self.session else None}
        if self._preset is not None:
            d.update(preset=self._preset.name, geometry_id=self._preset.geometry_id)
        return d


# ------------------------------------------------------------ prompts (only with --interactive)

PROMPT_TIMEOUT_S = 60.0         # a y/N question (delete, teach-presence)
WIZARD_TIMEOUT_S = 600.0        # a physical step of the wizard (clear the scene, place the part)
PROMPT_TIMEOUT_MAX_S = 3600.0


def _prompt_timeout(text: str) -> float:
    """--prompt-timeout: seconds, finite, > 0, at most an hour (never 'wait for ever')."""
    try:
        v = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected seconds, got {text!r}") from None
    if not (math.isfinite(v) and 0 < v <= PROMPT_TIMEOUT_MAX_S):
        raise argparse.ArgumentTypeError(
            f"must be > 0 and <= {PROMPT_TIMEOUT_MAX_S:g} s (a prompt always gives up), got {text}")
    return v


def _timed_input(prompt: str, timeout_s: float) -> str:
    """input() that gives up: the line typed, EOFError when stdin is closed, or Abort after
    timeout_s with no answer (the reading thread is a daemon: it dies with the process)."""
    import threading
    box: dict = {}

    def read() -> None:
        try:
            box["answer"] = input(prompt)
        except EOFError:
            box["eof"] = True
        except BaseException as e:              # noqa: BLE001 - handed to the caller's thread
            box["error"] = e

    th = threading.Thread(target=read, name="cli-prompt", daemon=True)
    th.start()
    th.join(timeout_s)
    if th.is_alive():
        raise Abort(f"no answer within {timeout_s:g} s (--prompt-timeout): nothing was done")
    if "error" in box:
        raise box["error"]
    if box.get("eof"):
        raise EOFError
    return box["answer"]


def _prompt(ctx, prompt: str, default_timeout: float = PROMPT_TIMEOUT_S) -> str:
    """One question to the operator. Only ever reached with --interactive (the callers refuse
    before, with the argument that answers the question); never waits for ever."""
    if not ctx.interactive:
        raise CLIError("this command would ask a question, and no command waits for input "
                       "unless --interactive is given")
    return _timed_input(prompt, ctx.prompt_timeout or default_timeout)


def _selection_gone(sel: dict, root, extra: str = "",
                    other: str = "name one with --preset") -> str:
    return (f"the selected preset {sel['name']!r} no longer exists in this library ({root})"
            f"{extra}: select one with `scan preset-select NAME` (see `scan presets`) or in the "
            f"web UI's Setup, or {other}")


# ------------------------------------------------------------ library lookup

def _pick(items: list, key: str, what: str, list_cmd: str, *, id_of=None, label_of=None,
          allow_latest: bool = True):
    """Find a library entry by exact id, label (newest wins), 'latest', or a
    unique piece of its id: ids are long timestamps nobody wants to type."""
    id_of = id_of or (lambda x: str(_get(x, "id")))
    label_of = label_of or (lambda x: str(_get(x, "label", "")))
    key = key.strip()
    if not items:
        raise CLIError(f"no {what} in the library yet (see `{list_cmd}`)")
    if allow_latest and key.lower() in ("latest", "last"):
        return items[-1]
    for test in (lambda x: id_of(x) == key, lambda x: label_of(x) == key):
        hits = [x for x in items if test(x)]
        if hits:
            return hits[-1]
    hits = [x for x in items if key and key in id_of(x)]
    if len(hits) == 1:
        return hits[0]
    if hits:
        shown = ", ".join(id_of(x) for x in hits[:4]) + (" ..." if len(hits) > 4 else "")
        raise CLIError(f"{what} {key!r} is ambiguous ({len(hits)} match: {shown}); "
                       "give more of the id")
    raise CLIError(f"no {what} {key!r} in the library (see `{list_cmd}`)")


def _find_base(key: str):
    return _pick(api.list_library("bases"), key, "base", "bases")


def _find_scan(key: str):
    return _pick(api.list_library("objects"), key, "scan", "scans")


def _find_spectral(key: str):
    return _pick(api.list_library("spectral"), key, "spectral reading", "spectral")


def _deletable(ctx) -> list:
    """(kind, id, label, path) for every library entry `delete` can trash."""
    P = api.paths()
    out = [("base", b.id, b.label, Path(b.path)) for b in api.list_library("bases")]
    out += [("scan", o.id, o.label, Path(o.path)) for o in api.list_library("objects")]
    out += [("dent set", d.id, d.label, Path(d.path)) for d in api.list_library("dentsets")]
    out += [(name, x.id, x.label, Path(x.path)) for kind, name in
            (("faces", "face target"), ("facescans", "face scan"), ("face_golden", "golden face"))
            for x in api.list_library(kind)]
    out += [("calibration", c["id"], c["id"], P["calibrations"] / c["id"])
            for c in api.list_library("calibrations")]
    # a calibration board pose: its id only (a set name names many poses, never one)
    out += [("calibration pose", m["id"], m["id"], Path(m["path"]))
            for m in api.list_library("calib_poses")]
    # the other entries of calibrations/: rig references and row-parity measurements
    store = ctx.session.store
    for geo in store.geometry_ids("calibrations"):
        for d in store.entries("calibrations", geo):
            for marker, name in (("rigref.json", "rig reference"),
                                 ("parity.json", "parity measurement")):
                if (d / marker).is_file():
                    out.append((name, f"{geo}/{d.name}", d.name, d))
    out += [("reading", r["id"], r["id"], Path(r["path"])) for r in api.list_library("readings")]
    # a sensor is named per geometry: its id is geometry/name, its label the name
    out += [("sensor", f"{s['geometry_id']}/{s['name']}", s["name"], Path(s["path"]))
            for s in api.list_library("sensors")]
    for kind, name in (("spectral", "spectral reading"), ("maps", "projector map"),
                       ("banks", "bank"), ("materials", "material")):
        try:
            items = api.list_library(kind)
        except NotImplementedError:
            continue                          # that store is not built yet: nothing to delete
        for it in items:
            ident = _get(it, "id") or _get(it, "name")
            path = _get(it, "path")
            if ident and path:
                out.append((name, str(ident), str(_get(it, "label") or _get(it, "name") or ident),
                            Path(path)))
    return out


# ------------------------------------------------------------ mock bench

def _mock_clear(ctx) -> None:
    if ctx.mock:
        api.mock_bench().clear()
        ctx.say("[mock] simulated bench cleared (stands in for the human)")


def _mock_place(ctx, spec: Optional[str] = None) -> None:
    """Mock mode only: put the part on the simulated bench, as a human would."""
    if not ctx.mock:
        return
    from engine.capture.mock_bench import VALIDATION_OBJECT, Box
    spec = spec or ctx.args.mock_part
    bench = api.mock_bench()
    bench.clear()
    if spec == "none":
        ctx.say("[mock] simulated bench left empty")
        return
    if spec.startswith("scene:"):
        raise CLIError(f"--mock-part {spec}: scenes are for the metric (stereo) bench: "
                       "metric / recipe jobs, recipe-run, recipe-golden")
    if spec.startswith("qr:"):
        from engine.capture.mock_bench import Label
        obj, desc = Label(spec[3:], center_mm=(0, 0), size_mm=45), f"a QR label {spec[3:]!r}"
    elif spec == "validation":
        obj, desc = VALIDATION_OBJECT, "the validation object (H 106.95, W 56.50, L 21.90 mm)"
    elif spec.startswith("cylinder:"):
        from engine.capture.mock_bench import Cylinder
        r, hh, dent = (float(v) for v in spec.split(":", 1)[1].split(","))
        dents = [{"depth_mm": dent, "size_mm": 12.0, "angle_deg": 10.0, "z_mm": hh / 2}] \
            if dent > 0 else []
        obj = Cylinder(radius_mm=r, height_mm=hh, dents=dents)
        desc = f"a standing cylinder (R {r:g}, H {hh:g} mm" + \
            (f", one {dent:g} mm dent at 10 deg, mid-height)" if dents else ", no dent)")
    else:
        vals = [float(v) for v in spec.split(",")]
        h, w, l = vals[:3]
        kw = {"albedo": vals[3]} if len(vals) == 4 else {}
        obj = Box(h, w, l, **kw)
        desc = f"a box (H {h:g}, W {w:g}, L {l:g} mm" + \
            (f", albedo {vals[3]:g}" if kw else "") + ")"
    bench.place(obj)
    ctx.say(f"[mock] simulated bench: placed {desc} (stands in for the human)")


# ------------------------------------------------------------ printers

def _print_signal(ctx, sig) -> None:
    verdict = f"{sig.quality.upper()} - ok to scan" if sig.ok else \
        f"{sig.quality.upper()} - NOT ok, fix the setup before scanning"
    if sig.axis_ok is None:
        axis = f"{sig.coded_axis or '?'} (not checked: no valid pixels)"
    elif sig.axis_ok:
        axis = f"{sig.coded_axis} (ok for this rig)"
    else:
        axis = f"{sig.coded_axis} (WRONG for this rig: depth sensitivity ~0)"
    ctx.say(f"Signal: {verdict}")
    ctx.table(["", "value", "meaning"], [
        ["gc_conf", _u(sig.gc_conf, "counts", 2),
         "fringe modulation: <8 unusable, ~18 marginal, ~30 usable, 120+ excellent"],
        ["valid", _u(sig.valid_pct, "%", 2), "camera px lit AND modulated"],
        ["contrast", _u(sig.contrast_mean, "counts", 2), "mean white - black"],
        ["focus", _u(sig.focus_ratio, "ratio", 3), sig.focus_hint],
        ["coded axis", axis, "fringes must vary across the baseline"],
    ])
    ctx.warn(sig.warnings)


def _print_scan(ctx, scan, base=None) -> None:
    m = scan.meta or {}
    ctx.say(f"Scan {scan.id}  (label {scan.label})")
    ctx.kv([
        ["base", f"{scan.base_id}" + (f" ({base.label})" if base is not None else "")],
        ["geometry", scan.geometry_id],
        ["zone", m.get("zone") or "whole field"],
        ["valid", _u(m.get("valid_pct"), "% of camera px", 2)],
        ["overlap", _u(m.get("overlap_px"), "px valid in base AND object")],
        ["datum", _u(m.get("datum_px"), "px (base plane offset removed)", 3)],
        ["relief", f"{m.get('tier', 'relief')} tier, {m.get('units', 'px')}"],
        ["known dims", _dims_text(scan.known_dims)],
        ["calibration", scan.calibration_id or "none for this geometry -> px (rule 6)"],
        ["stored", str(scan.path)],
    ])
    ctx.warn(m.get("warnings"))


def _print_dims(ctx, d, scan) -> None:
    known = scan.known_dims or {}
    rows = []
    for k in ("H", "W", "L"):
        unit = (d.unit_by_dim or {}).get(k, d.units)
        err, pct = (d.error or {}).get(k), (d.error_pct or {}).get(k)
        if k in known and err is None:
            err_txt = "n/a (not in mm)"
        else:
            err_txt = _u(err, unit) if err is not None else "-"
        rows.append([k, _f(getattr(d, k)), unit, _u(known.get(k), "mm", 2) if k in known else "-",
                     err_txt, f"{pct:+.2f} %" if pct is not None else "-"])
    ctx.say(f"Dimensions of {scan.id}  (H height, W major footprint, L minor footprint)")
    ctx.table(["dim", "value", "unit", "known", "error", "error %"], rows)
    ctx.say(f"  tier {d.tier} | units {d.units} | part blob {d.n_px} camera px | "
            f"raw relief {_u(d.relief_px, 'px')}")
    ctx.warn(d.warnings)


def _print_inspection(ctx, insp, scan) -> None:
    u = insp.units
    ctx.say(f"Inspection of {scan.id}: {insp.verdict}")
    ctx.kv([
        ["units", f"{u} ({insp.tier} tier)"],
        ["noise sigma", _u(insp.noise_sigma, u, 4)],
        ["threshold", _u(insp.threshold, u, 4)],
        ["FAIL trigger", _u(insp.trigger, "mm", 3) if insp.trigger is not None else "-"],
        ["region", _u(insp.region_px, "camera px")],
        ["defects", str(len(insp.defects))],
    ])
    rows = []
    for i, d in enumerate(insp.defects[:20], 1):
        rows.append([i, _u(d.height, d.units, 3), _u(d.peak, d.units, 3), _f(d.snr, 1),
                     d.area_px, _f(d.area_mm2, 2) if d.area_mm2 is not None else "-",
                     ",".join(_f(v, 0) for v in d.bbox), _f(d.exceeds_trigger)])
    ctx.table(["#", "height", "peak", "SNR", "area px", "area mm^2", "bbox x0,y0,x1,y1 (px)",
               "over trigger"], rows)
    if len(insp.defects) > 20:
        ctx.say(f"  ... {len(insp.defects) - 20} more in the report")
    ctx.warn(insp.warnings)


def _print_inspection_files(ctx, insp) -> None:
    """The annotated PNG sits next to the report but is not in the output
    index, so name it explicitly."""
    if insp.path:
        png = Path(insp.path).with_suffix(".png")
        ctx.say(f"  report: {insp.path}", *([f"  image:  {png}"] if png.exists() else []))


def _stat(key: str, v, unit: str) -> str:
    """A stats-dict value with the right units: counts and sigma ratios are
    not heights, so they must not be labelled px/mm."""
    if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating)):
        return str(v)
    k = key.lower()
    if k.startswith("n_") or "_n_" in k or k.endswith("_n") or k in ("n", "count"):
        return _f(v, 0) + " samples"
    if "sigma" in k or k.endswith("_z") or "snr" in k:
        return _u(v, "sigma", 2)
    if "pct" in k or "percent" in k:
        return _u(v, "%", 2)
    return _u(v, unit, 4)


def _pointcloud_units(path: Path) -> str:
    """The exporter writes the units into the header; report exactly those."""
    try:
        with open(path, encoding="ascii", errors="replace") as f:
            head = [next(f, "") for _ in range(8)]
    except OSError:
        return "see the file header"
    for line in head:
        m = re.search(r"units\s+(xy=\S+\s+z=\S+)", line)
        if m:
            return m.group(1)
    return "see the file header"


# ------------------------------------------------------------ commands

def cmd_presets(ctx):
    from engine import rigstate
    rows, data = [], []
    tiers = {i["name"]: i for i in api.preset_list()}
    selected = api.preset_active()
    for name in api.list_presets():
        info = tiers.get(name, {})
        mark = "*" if name == ctx.args.preset else ""
        tier = info.get("tier", "?")
        stereo = "stereo" if info.get("calibrated") else "-"
        try:
            p = config.load_preset(name, presets_dir=ctx.session.presets_dir,
                                   store=ctx.session.store, mock=ctx.mock)
        except (config.ConfigError, rules.RuleViolation) as e:
            rows.append([mark, name, tier, f"INVALID: {_one_line(e)}", "", "", "", "", "", ""])
            data.append({"name": name, "tier": tier, "valid": False, "error": str(e)})
            continue
        cal = p.calibration
        cal_txt = (f"{cal.id} ({'Z' if cal.has_z else ''}{'+XY' if cal.has_xy else ''})"
                   if cal else "none -> px")
        w, h = p.projection.size
        # a geometry_id names one rig: a preset at a mobile preset's geometry is gated as mobile
        gated, by = rigstate.mount_of(ctx.session.store, p, ctx.session.presets_dir)
        by = by if (by and by != p.name) else None
        rows.append([mark, p.name, tier,
                     p.geometry_id + (" [mobile]" if p.rig.mount == "mobile" else
                                      f" [mobile: as {by}]" if by else ""),
                     p.rig.projector_side, f"{w}x{h} px",
                     f"{p.camera.exposure_us} us, avg {p.camera.average}", _bases_text(p.bases),
                     cal_txt, stereo])
        data.append({"name": p.name, "valid": True, "tier": tier, "geometry_id": p.geometry_id,
                     "mount": p.rig.mount, "gated_as": gated, "mount_declared_by": by,
                     "projector_side": p.rig.projector_side, "camera": p.camera,
                     "bases": p.bases, "calibration": cal, "warnings": p.warnings,
                     "calibrated": bool(info.get("calibrated")), "selected": name == selected,
                     "description": info.get("description", "")})
    ctx.say(f"Presets: built-in {ctx.session.presets_dir or config.PRESETS_DIR} (read-only), "
            f"user {config.user_presets_dir(ctx.session.store.root)}")
    sel = ctx.selection or {}
    stale = sel.get("name") if sel.get("name") and not sel.get("exists") else None
    shown = selected or (f"{stale} (GONE)" if stale else "- none selected, the default")
    ctx.say(f"  (* = in use for this run: --preset NAME, else the selected preset {shown})")
    ctx.table(["", "name", "tier", "geometry", "projector", "DMD", "camera", "bases",
               "calibration", "stereo cal"], rows)
    if stale:
        ctx.say(f"  ! the selected preset {stale!r} no longer exists: commands that scan stop "
                "until you `scan preset-select NAME` (or pass --preset)")
    return EXIT_OK, {"presets": data, "active": ctx.args.preset, "selected": selected,
                     "selected_missing": stale}


def _mount_words(mount: str) -> str:
    """rig.mount in words: how far the stored calibration is trusted."""
    return {"fixed": "fixed (the calibration is trusted until a rig check says MOVED / UNKNOWN)",
            "mobile": "mobile (assumed moved since its last use: verified again every session "
                      "- rig-check / re-home / calibrate - before any mm)"}.get(mount, str(mount))


def _parse_sets(items) -> dict:
    out = {}
    for it in items or []:
        key, sep, val = it.partition("=")
        if not sep or not key.strip():
            raise Abort(f"--set wants key=value (e.g. camera.exposure_us=33333), got {it!r}")
        out[key.strip()] = val
    return out


def cmd_preset_add(ctx):
    a = ctx.args
    r = api.preset_add(a.name, a.base, _parse_sets(a.set), a.geometry, a.desc or "", a.replace)
    ctx.say(f"User preset {r['name']} added (copied from {a.base}): {r['path']}")
    if r.get("replaced"):
        ctx.say(f"  the previous {r['name']} moved to {r['replaced']}")
    if r.get("note"):
        ctx.say(f"  note: {r['note']}")
    ctx.warn([f"preset: {w}" for w in r.get("warnings") or []])
    ctx.say(f"  use it: --preset {r['name']}, or scan preset-select {r['name']}")
    return EXIT_OK, {"preset": r}


def cmd_preset_select(ctx):
    r = api.preset_select(ctx.args.name)
    ctx.say(f"Selected preset {r['name']} ({r['tier']}): used when no --preset is given "
            "(a station config's own preset still wins)")
    return EXIT_OK, {"selected": r}


def cmd_preset_delete(ctx):
    a = ctx.args
    r = api.preset_delete(a.name, a.reason, force=a.force,
                          stations_dir=Path(a.stations).resolve() if a.stations else None)
    ctx.say(f"User preset {r['name']} moved to {r['trashed']} (recoverable: move it back)")
    if r.get("forced"):
        used = [f"station config(s) {', '.join(r['stations'])}"] if r.get("stations") else []
        used += [f"recipe(s) {', '.join(r['recipes'])}"] if r.get("recipes") else []
        used += ([f"recipe station config(s) {', '.join(r['recipe_stations'])}"]
                 if r.get("recipe_stations") else [])
        ctx.say(f"  ! still referenced by {'; '.join(used)}")
    return EXIT_OK, {"deleted": r}


def cmd_preset_show(ctx):
    r = api.preset_show(ctx.args.name)
    ctx.kv([["name", r["name"]], ["tier", r["tier"] + (" (read-only)" if r["tier"] == "builtin" else "")],
            ["selected", "yes" if r.get("active") else "no"], ["geometry_id", r.get("geometry_id")],
            ["mount (rig.mount)", _mount_words(r.get("mount") or "fixed")],
            ["stereo calibration", "yes" if r.get("calibrated") else "none (uncalibrated)"],
            ["description", r.get("description") or "-"], ["copied from", r.get("base") or "-"],
            ["created", r.get("created") or "-"], ["author", r.get("author") or "-"]])
    if not r.get("valid", True):
        ctx.say(f"  ! INVALID: {_one_line(r.get('error'))}")
    ctx.say(json.dumps(r["preset"], indent=2))
    return EXIT_OK, {"preset": r}


# ------------------------------------------------------------ recipes (what verifies a work state)

def cmd_recipes(ctx):
    items = api.recipe_list()
    sel = api.recipe_active()
    paths = api.paths()
    ctx.say(f"Recipes: built-in {paths['recipes']} (read-only), user {paths['user_recipes']}")
    ctx.say(f"  (* = selected: {sel or 'none'})")
    ctx.table(["", "name", "tier", "preset", "scan", "zones", "checks", "golden", "valid"],
              [["*" if i["name"] == sel else "", i["name"], i["tier"], i.get("preset", "-"),
                i.get("scan", "-"), ",".join(i.get("zones") or []) or "-",
                ",".join(i.get("checks") or []) or "-", i.get("golden") or "-",
                "yes" if i.get("valid") else f"NO: {_one_line(i.get('error'))[:60]}"]
               for i in items])
    return EXIT_OK, {"recipes": items, "selected": sel}


def cmd_recipe_show(ctx):
    r = api.recipe_show(ctx.args.name)
    ctx.kv([["name", r["name"]], ["tier", r["tier"] + (" (read-only)" if r["tier"] == "builtin"
                                                      else "")],
            ["selected", "yes" if r.get("active") else "no"], ["preset", r.get("preset") or "-"],
            ["file", r.get("path")], ["description", r.get("description") or "-"],
            ["golden", r.get("golden") or "-"], ["mock scene", r.get("mock_scene") or "-"]])
    if not r.get("valid", True):
        ctx.say(f"  ! INVALID: {_one_line(r.get('error'))}")
    ctx.say(json.dumps(r["recipe"], indent=2))
    return EXIT_OK, {"recipe": r}


def cmd_recipe_checks(ctx):
    av = api.recipe_checks()
    for cid, d in av.items():
        ctx.say(f"{cid}: {d['title']}" + (f"  (needs {', '.join(d['needs'])})" if d["needs"] else ""))
        ctx.table(["param", "kind", "unit", "default", "doc"],
                  [[k, v["kind"], v.get("unit") or "-",
                    "required" if v["required"] else json.dumps(v.get("default")), v["doc"]]
                   for k, v in d["params"].items()], indent="    ")
    return EXIT_OK, {"checks": av}


def cmd_recipe_validate(ctx):
    v = api.recipe_validate(ctx.args.recipe)
    if v["valid"]:
        r = v["recipe"]
        ctx.say(f"Recipe {r['name']}: valid (schema {r['schema_version']}, preset {r['preset']}, "
                f"{len(r['zones'])} zone(s))")
    else:
        ctx.say(f"Recipe {ctx.args.recipe}: {len(v['problems'])} problem(s)")
        for pr in v["problems"]:
            ctx.say(f"  - {_one_line(pr)}")
    return (EXIT_OK if v["valid"] else EXIT_REFUSED), {"validation": v}


def cmd_recipe_add(ctx):
    a = ctx.args
    r = api.recipe_add(a.name, a.base, _parse_sets(a.set), a.desc or "", a.replace)
    ctx.say(f"User recipe {r['name']} added (copied from {a.base}): {r['path']}")
    if r.get("replaced"):
        ctx.say(f"  the previous {r['name']} moved to {r['replaced']}")
    ctx.say(f"  run it: scan recipe-run {r['name']}, or scan recipe-select {r['name']}")
    return EXIT_OK, {"recipe": r}


def cmd_recipe_select(ctx):
    r = api.recipe_select(ctx.args.name)
    ctx.say(f"Selected recipe {r['name']} ({r['tier']}): the default of recipe-run")
    return EXIT_OK, {"selected": r}


def cmd_recipe_delete(ctx):
    a = ctx.args
    r = api.recipe_delete(a.name, a.reason, force=a.force,
                          stations_dir=Path(a.stations).resolve() if a.stations else None)
    ctx.say(f"User recipe {r['name']} moved to {r['trashed']} (recoverable: move it back)")
    if r.get("forced"):
        ctx.say(f"  ! still referenced by station config(s): {', '.join(r['stations'])}")
    return EXIT_OK, {"deleted": r}


def _recipe_arg(ctx):
    name = ctx.args.recipe or api.recipe_active()
    if not name:
        raise CLIError("which recipe? give a name or a file (scan recipe-list), or select one "
                       "with scan recipe-select NAME")
    r = api.recipe_load(name)
    if ctx.args.average is not None:
        r.scan["average"] = ctx.args.average
    return r


def _recipe_mock_scene(ctx, r) -> Optional[str]:
    """MOCK: the scene to place (--mock-part when given, else the recipe's mock_scene)."""
    if not ctx.mock:
        return None
    if getattr(ctx.args, "mock_part_given", False):
        spec = ctx.args.mock_part
        if not spec.startswith("scene:"):
            _mock_place_metric(ctx)            # a block / cylinder / the empty wall
            return None
        return spec
    return r.part.get("mock_scene")


def cmd_recipe_run(ctx):
    a = ctx.args
    r = _recipe_arg(ctx)
    preset = ctx.preset() if a.preset_given else None
    want_fb = r.feedback["enabled"] and not a.no_feedback
    offline = bool(a.scan) and not want_fb          # a stored scan, nothing projected: no rig
    if not ctx.mock and not a.operator_confirmed and not offline:
        raise CLIError("the real rig needs a human to confirm the physical setup first (part "
                       "placed, rig untouched): pass --operator-confirmed (a stored scan "
                       "re-checked offline: --scan ID --no-feedback, no confirmation needed)")
    if a.pattern_mode_set_by_human and r.scan["kind"] != "trigger":
        raise CLIError("--pattern-mode-set-by-human is only for scan.kind trigger recipes")
    scene = None if a.scan else _recipe_mock_scene(ctx, r)
    if want_fb and not ctx.json:
        ctx.say(f"[{'MOCK' if ctx.mock else 'REAL RIG'}] recipe {r.name} (preset "
                f"{preset.name if preset else r.preset}); the verdict is projected on the part "
                f"for {a.hold if a.hold is not None else r.feedback['hold_s']:g} s")
    out = api.recipe_run(r, preset=preset, operator_confirmed=bool(a.operator_confirmed),
                         project_feedback=want_fb, scan=a.scan, runs_dir=a.runs_dir,
                         hold_s=a.hold, label=a.label, mock_scene=scene,
                         pattern_mode_ack=bool(a.pattern_mode_set_by_human))
    res = out["result"]
    dev = res["deviation_mm"]["max"]
    ctx.say(f"Run {out['run_id']}: {res['result']}"
            + (f"  ({res['invalid_reason']})" if res["invalid_reason"] else ""))
    ctx.say(f"  disposition: {_one_line(D.summary(res))}")
    ctx.kv([["recipe", f"{r.name} (preset {res['capture']['preset'] or r.preset})"],
            ["scan", res["part"]["scan_id"] or "-"],
            ["headline", f"{res['deviation_mm']['method']}: {_u(dev, 'mm', 4)}"
             if dev is not None else "-"],
            ["result code", str((res.get("recipe") or {}).get("result_code"))],
            ["result.json", str(Path(out["run_dir"]) / "result.json")],
            ["log", str(Path(out["run_dir"]) / "run.log.jsonl")]])
    _recipe_summary(ctx, res)
    ctx.warn(res["warnings"])
    return out["exit_code"], {"run_id": out["run_id"], "run_dir": str(out["run_dir"]),
                              "result": res}


def cmd_recipe_golden(ctx):
    a = ctx.args
    r = _recipe_arg(ctx)
    preset = ctx.preset() if a.preset_given else None
    scene = _recipe_mock_scene(ctx, r)
    ms = api.recipe_golden(r, preset=preset, label=a.label, mock_scene=scene)
    ctx.say(f"Golden scan for recipe {r.name}: {ms.id} (label {ms.label}, "
            f"{(ms.meta or {}).get('n_points')} points, calibration {ms.calibration_id})")
    if (r.golden or {}).get("scan") not in (ms.label, "latest", "last"):
        ctx.say(f"  ! the recipe's golden.scan is {(r.golden or {}).get('scan')!r}: point it at "
                f"{ms.label!r} (scan recipe-add ... --set golden.scan={ms.label})")
    return EXIT_OK, {"golden": ms}


def cmd_paths(ctx):
    P = api.paths()
    ctx.say("Where everything lives:")
    ctx.table(["", "path", "exists"], [[k, str(v), _f(Path(v).exists())] for k, v in P.items()])
    return EXIT_OK, {"paths": P}


def cmd_signal(ctx):
    ctx.header()
    sig = api.check_signal(ctx.preset())
    _print_signal(ctx, sig)
    return (EXIT_OK if sig.ok else EXIT_NOT_OK), {"signal": sig}


def cmd_tune(ctx):
    ctx.header()
    a = ctx.args
    rep = api.tune(ctx.preset(), a.exposures, a.duties)
    ctx.say(f"Scan assistant: {'settings look good' if rep.ok else 'settings need work'}")
    locked = {True: "yes", False: "NO", None: "-"}
    ctx.table(["exposure", "duty", "gc_conf", "valid", "saturated", "focus", "axis",
               "frame-locked"],
              [[_u(s.exposure_us, "us", 0), _f(s.duty, 2), _u(s.gc_conf, "counts", 1),
                _u(s.valid_pct, "%", 1), _u(s.saturated_pct, "%", 2),
                _u(s.focus_ratio, "ratio", 2), s.coded_axis or "?",
                locked[getattr(s, "frame_locked", None)]] for s in rep.steps])
    if any(getattr(s, "frame_locked", None) is not None for s in rep.steps):
        ctx.say(f"  frame-locked = the exposure is a whole number of HDMI video frames "
                f"({_f(getattr(rep, 'video_frame_us', None), 1)} us each); others can beat "
                "against the frame rate")
    if rep.recommended:
        ctx.say("Recommended preset changes (NOT applied; edit the preset to use them):")
        ctx.table(["setting", "value"], [[k, v] for k, v in _flatten(rep.recommended)])
    for line in rep.advice or []:
        ctx.say(f"  - {_one_line(line)}")
    if rep.path:
        ctx.say(f"  report: {rep.path}")
    return (EXIT_OK if rep.ok else EXIT_NOT_OK), {"tune": rep}


def cmd_gamma(ctx):
    ctx.header()
    rep = api.measure_gamma(ctx.preset(), ctx.args.levels)
    residual = getattr(rep, "residual_gamma", None)
    ctx.say("Projector gray response:")
    ctx.kv([["gamma", f"{_f(rep.gamma, 3)} (projector vs the value sent over HDMI: the precomp "
                      "to use)"],
            ["residual gamma", f"{_f(residual, 3)} (the preset chain as driven now; 1 = linear)"
             if residual is not None else "- (not measured)"],
            ["R^2", _f(rep.r2, 4)]])
    ctx.table(["DMD level", "response"],
              [[_u(lv, "/255", 0), _u(r, "(normalised)", 3)]
               for lv, r in zip(rep.levels, rep.response)])
    if rep.recommended:
        ctx.say("Recommended (NOT applied):")
        ctx.table(["setting", "value"], [[k, v] for k, v in _flatten(rep.recommended)])
    if rep.path:
        ctx.say(f"  report: {rep.path}")
    ctx.warn(rep.warnings)
    return EXIT_OK, {"gamma": rep}


def cmd_base(ctx):
    ctx.header()
    p = ctx.preset()
    _mock_clear(ctx)
    base = api.capture_base(p, ctx.args.label)
    rejected = base.meta.get("rejected")
    ctx.say(f"Base {base.id}  (label {base.label})"
            + ("  AUTO-REJECTED: scans will not use it" if rejected else ""))
    ctx.kv([
        ["geometry", base.geometry_id],
        ["valid", _u(base.meta.get("valid_pct"), "% of camera px", 2)],
        ["bases", ", ".join(base.meta.get("basis_names") or [])],
        ["stored", str(base.path)],
    ])
    ctx.warn(base.warnings)                 # auto-reject, valid % far under the last base
    if rejected:
        ctx.say("  Fix the setup (see `signal`), then capture the base again.")
    else:
        ctx.say("  Do not move camera, projector or focus until the part scans are done.",
                "  Next: place the part, then  scan <label> [--known H,W,L]")
    return (EXIT_NOT_OK if rejected else EXIT_OK), {"base": base, "rejected": rejected}


def cmd_bases(ctx):
    bases = api.list_library("bases", ctx.args.geometry)
    default = None
    try:
        default = api.latest_base(ctx.preset())
    except (config.ConfigError, rules.RuleViolation):
        pass                                    # listing still works without a preset
    rows, data = [], []
    for b in bases:
        rej = b.meta.get("rejected")
        is_default = default is not None and b.id == default.id
        status = f"REJECTED: {rej}" if rej else ("<- default for scan" if is_default else "")
        rows.append([b.id, b.label, b.geometry_id, _u(b.meta.get("valid_pct"), "%", 1), status])
        data.append({"id": b.id, "label": b.label, "geometry_id": b.geometry_id,
                     "created": b.created, "path": b.path, "valid_pct": b.meta.get("valid_pct"),
                     "rejected": rej, "default_for_scan": is_default})
    ctx.say(f"Bases ({len(bases)})" + (f" for geometry {ctx.args.geometry}"
                                      if ctx.args.geometry else ""))
    ctx.table(["id", "label", "geometry", "valid", "status"], rows)
    if not bases:
        ctx.say("  none yet: clear the scene and run  base")
    return EXIT_OK, {"bases": data}


def cmd_reject_base(ctx):
    base = _find_base(ctx.args.id)
    reason = " ".join(ctx.args.reason)
    api.reject_base(base, reason)
    usable = [b for b in api.list_library("bases", base.geometry_id) if not b.meta.get("rejected")]
    ctx.say(f"Base {base.id} marked REJECTED ({reason}); it stays in the library, "
            "scans no longer pick it.",
            f"  default base for {base.geometry_id} is now: "
            f"{usable[-1].id if usable else 'none - capture a new base'}")
    return EXIT_OK, {"rejected": base.id, "reason": reason,
                     "default_base": usable[-1].id if usable else None}


def cmd_delete(ctx):
    a = ctx.args
    key = a.id.strip()
    if key.lower() in ("latest", "last"):
        raise CLIError("delete needs an explicit id or label, not 'latest'")
    entries = _deletable(ctx)
    same = [e for e in entries if e[1] == key] or [e for e in entries if e[2] == key]
    kinds = sorted({e[0] for e in same})
    if len(kinds) > 1:                   # never guess which KIND of thing to trash
        raise CLIError(f"{key!r} names a {' and a '.join(kinds)}; give the full id")
    if kinds == ["sensor"] and len(same) > 1:     # same name taught at several geometries
        raise CLIError(f"sensor {key!r} exists at several geometries; give geometry/name: "
                       + ", ".join(e[1] for e in same))
    hit = _pick(entries, key, "library entry", "bases / scans", id_of=lambda e: e[1],
                label_of=lambda e: e[2], allow_latest=False)
    kind, ident, label, path = hit
    if not a.yes:
        if ctx.json:
            raise CLIError("delete with --json needs --yes (no interactive prompt)")
        if not ctx.interactive:             # never wait for an answer nobody may give
            raise CLIError(f"delete moves {kind} {ident} to trash/ only with --yes (this "
                           "command does not wait for input; --interactive asks [y/N] and "
                           "gives up after --prompt-timeout s). Nothing moved")
        try:
            ans = _prompt(ctx, f"Move {kind} {ident} to trash/? It stays recoverable. [y/N] ")
        except EOFError:
            ans = ""
        if ans.strip().lower() not in ("y", "yes"):
            ctx.say("Cancelled; nothing moved.")
            return EXIT_NOT_OK, {"deleted": None, "cancelled": True}
    dest = api.delete(path, a.reason)
    ctx.say(f"Moved {kind} {ident} to trash:", f"  {dest}",
            f"  (recoverable: move it back to {path} to restore)")
    return EXIT_OK, {"deleted": {"kind": kind, "id": ident, "from": path, "trash": dest,
                                 "reason": a.reason}}


def cmd_scan(ctx):
    a = ctx.args
    known = parse_known(a.known)
    ctx.header()
    p = ctx.preset()
    if a.base:
        base = _find_base(a.base)
        if base.meta.get("rejected"):
            raise CLIError(f"base {base.id} is REJECTED ({base.meta['rejected']}); "
                           "pick another or capture a new one with `base`")
    else:
        base = api.latest_base(p)
        if base is None:
            raise CLIError(f"no usable base for geometry {p.geometry_id}: clear the scene "
                           "and run `base` first")
    if a.zone:
        ctx.mode = "zone scanning"
    _mock_place(ctx)
    scan = api.capture_object(p, base, a.label, known_dims=known, zone=a.zone)
    _print_scan(ctx, scan, base)
    ctx.say(f"  Next: measure {scan.label} | inspect {scan.label} | reconstruct {scan.label}")
    return EXIT_OK, {"scan": scan, "base_id": base.id}


def cmd_scans(ctx):
    scans = api.list_library("objects", ctx.args.geometry)
    rows = [[s.id, s.label, s.geometry_id, s.base_id, _dims_text(s.known_dims),
             _u(s.meta.get("valid_pct"), "%", 1), s.meta.get("zone") or "-"] for s in scans]
    ctx.say(f"Scans ({len(scans)})")
    ctx.table(["id", "label", "geometry", "base", "known", "valid", "zone"], rows)
    if not scans:
        ctx.say("  none yet: place a part and run  scan <label>")
    return EXIT_OK, {"scans": [{"id": s.id, "label": s.label, "geometry_id": s.geometry_id,
                                "base_id": s.base_id, "known_dims": s.known_dims,
                                "calibration_id": s.calibration_id, "created": s.created,
                                "path": s.path, "zone": s.meta.get("zone"),
                                "valid_pct": s.meta.get("valid_pct")} for s in scans]}


def cmd_reconstruct(ctx):
    ctx.preset()                  # binds a preset-level calibration (real rig)
    scan = _find_scan(ctx.args.id)
    out = api.reconstruct(scan, fmt=ctx.args.fmt)
    units = _pointcloud_units(out)
    ctx.say(f"Point cloud of {scan.id}: units {units}", f"  {out}",
            "  open it in MeshLab / CloudCompare / Blender")
    if "px" in units:
        ctx.warn(["uncalibrated: coordinates are px, not mm (rule 6)"])
    return EXIT_OK, {"scan_id": scan.id, "file": out, "units": units}


def cmd_measure(ctx):
    ctx.preset()
    scan = _find_scan(ctx.args.id)
    dims = api.measure(scan)
    _print_dims(ctx, dims, scan)
    return EXIT_OK, {"scan_id": scan.id, "dimensions": dims}


def cmd_calibrate(ctx):
    p = ctx.preset()
    scans = [_find_scan(k) for k in ctx.args.ids]
    cal = api.calibrate(p, scans)
    ctx.say(f"Calibration {cal.id}  (geometry {cal.geometry_id}, {cal.n_samples} sample(s))")
    z = (f"height_mm = {_f(cal.mm_per_px, 5)} mm/px * relief_px + {_f(cal.intercept_mm, 3)} mm"
         + (f", R^2 {_f(cal.r2, 5)}" if cal.r2 is not None else ", R^2 n/a (1 sample)")
         if cal.has_z else "not fitted (no sample with known H)")
    xy = (f"{_f(cal.lateral_px_per_mm, 4)} camera px per mm on the base plane"
          if cal.has_xy else "not fitted (no sample with known W/L)")
    ctx.kv([["Z", z], ["XY", xy], ["samples", ", ".join(s.id for s in scans)],
                         ["stored", str(cal.path)]])
    ctx.say(f"  measure / inspect / reconstruct now report mm for geometry {cal.geometry_id}. "
            "Moving the rig invalidates it.")
    return EXIT_OK, {"calibration": cal}


def cmd_inspect(ctx):
    ctx.preset()                  # the FAIL trigger comes from the preset tolerance
    a = ctx.args
    scan = _find_scan(a.id)
    insp = api.inspect(scan, n_sigma=a.sigma, min_area_px=a.min_area, whole_field=a.whole_field)
    _print_inspection(ctx, insp, scan)
    _print_inspection_files(ctx, insp)
    return (EXIT_OK if insp.verdict == "PASS" else EXIT_NOT_OK), {"scan_id": scan.id,
                                                                  "inspection": insp}


def cmd_compare(ctx):
    p = ctx.preset()
    a = ctx.args
    scan, golden = _find_scan(a.id), _find_scan(a.golden)
    tol = None
    if a.fail_mm is not None or a.fail_px is not None:
        from engine.results import ToleranceBand
        tol = ToleranceBand(fail_mm=a.fail_mm if a.fail_mm is not None else p.tolerance.ch_fail_mm,
                            fail_px=a.fail_px, min_defect_px=p.tolerance.min_defect_px)
    v = api.compare(scan, golden, tol)
    u = v.units
    ctx.say(f"Compare {scan.id} vs golden {golden.id}: {v.result}")
    ctx.kv([
        ["units", f"{u} ({v.tier} tier)"],
        ["max |dev|", _u(v.max_abs_dev, u, 4)],
        ["p99 |dev|", _u(v.p99_abs_dev, u, 4)],
        ["tolerance", _u(v.tolerance, u, 4) if v.tolerance is not None else f"none in {u}"],
        ["failing", _u(v.n_fail_px, "px")],
        ["compared", _u(v.n_compared_px, "px")],
    ])
    ctx.warn(v.warnings)
    if v.result == "INDETERMINATE" and u == "px":
        ctx.say("  (uncalibrated: give a px band with --fail-px, or calibrate this geometry)")
    return (EXIT_OK if v.result == "PASS" else EXIT_NOT_OK), {"scan_id": scan.id,
                                                              "golden_id": golden.id,
                                                              "verdict": v}


def cmd_profile(ctx):
    ctx.preset()
    a = ctx.args
    if (a.p0 is None) != (a.p1 is None):
        raise CLIError("give both --p0 x,y and --p1 x,y, or neither (then --axis picks the line)")
    p0 = _numbers(a.p0, 2, "--p0") if a.p0 else None
    p1 = _numbers(a.p1, 2, "--p1") if a.p1 else None
    scan = _find_scan(a.id)
    pr = api.profile(scan, p0=p0, p1=p1, width_px=a.width, axis=a.axis)
    uh, ud = pr.units_height, pr.units_distance
    ctx.say(f"Profile of {scan.id}: ({_f(pr.p0[0], 0)},{_f(pr.p0[1], 0)}) -> "
            f"({_f(pr.p1[0], 0)},{_f(pr.p1[1], 0)}) camera px, averaged over {pr.width_px} px")
    n = len(pr.height)
    length = pr.distance[-1] - pr.distance[0] if n > 1 else 0.0
    ctx.kv([["samples", str(n)], ["length", _u(length, ud, 2)],
            ["units", f"height {uh}, distance {ud} ({pr.tier} tier)"]] +
           [[k, _stat(k, v, uh)] for k, v in (pr.stats or {}).items() if k != "units"])
    for label, path in (("csv", pr.csv_path), ("plot", pr.png_path)):
        if path:
            ctx.say(f"  {label}: {path}")
    ctx.warn(pr.warnings)
    return EXIT_OK, {"scan_id": scan.id, "profile": pr}


def cmd_bank_add(ctx):
    ctx.preset()
    scan = _find_scan(ctx.args.id)
    info = api.bank_add(scan, ctx.args.bank)
    ctx.say(f"Bank '{info.name}' ({info.geometry_id}): {info.n_scans} known-good scan(s)"
            + (f", zone {info.zone}" if info.zone else ""), f"  stored: {info.path}")
    ctx.warn(info.warnings)
    return EXIT_OK, {"bank": info}


def cmd_bank_check(ctx):
    ctx.preset()
    a = ctx.args
    scan = _find_scan(a.id)
    v = api.bank_check(scan, a.bank, a.k, a.min_area)
    ctx.say(f"Bank check of {scan.id} vs '{v.bank}' ({v.n_ref} good scans): {v.result}")
    ctx.kv([["threshold", _u(v.k_sigma, "sigma", 1)],
                         ["worst", _u(v.max_z, "sigma", 2)],
                         ["deviation units", v.units], ["anomalies", str(len(v.anomalies))]])
    ctx.table(["#", "area px", "max", "mean dev", "bbox x0,y0,x1,y1 (px)"],
              [[i, _get(an, "area_px"), _u(_get(an, "max_z"), "sigma", 2),
                _u(_get(an, "mean_dev"), _get(an, "units") or v.units, 4),
                ",".join(_f(x, 0) for x in (_get(an, "bbox") or []))]
               for i, an in enumerate(v.anomalies[:20], 1)])
    for label, path in (("report", v.json_path), ("image", v.png_path)):
        if path:
            ctx.say(f"  {label}: {path}")
    ctx.warn(v.warnings)
    return (EXIT_OK if v.result == "PASS" else EXIT_NOT_OK), {"scan_id": scan.id, "verdict": v}


def cmd_codes(ctx):
    ctx.header()
    a = ctx.args
    strip_cols = None
    if a.strip_cols:
        if not a.anti_glare:
            raise CLIError("--strip-cols goes with --anti-glare")
        strip_cols = tuple(_numbers(a.strip_cols, 2, "--strip-cols C0,C1 (DMD columns)", cast=int))
    if ctx.mock and str(a.mock_part).startswith("qr:"):
        _mock_place(ctx)
    r = api.read_codes(ctx.preset(), label=a.label,
                       **({"anti_glare": True, "strip_cols": strip_cols} if a.anti_glare else {}))
    ctx.say(f"Codes: {len(r.codes)} found"
            + (" (DMD polarity inverted, handled)" if r.polarity_inverted else ""))
    ctx.table(["kind", "data", "illumination", "decoder"],
              [[c.kind, c.data + (f" (read as {c.raw})" if c.raw else ""), c.illumination,
                c.decoder] for c in r.codes])
    ctx.say(f"  illuminations tried: {', '.join(map(str, r.illuminations))}",
            f"  frames: {r.path}")
    ctx.warn(r.warnings)
    return (EXIT_OK if r.found else EXIT_NOT_OK), {"reading": r}


def cmd_spectral(ctx):
    ctx.header()
    r = api.spectral_capture(ctx.preset(), ctx.args.label)
    ctx.say(f"Spectral reading {r.id}  (label {r.label}, bands {', '.join(r.bands)})",
            f"  stored: {r.path}",
            f"  Next: learn {r.label} <material> x0,y0,x1,y1   then   classify {r.label}")
    ctx.warn(r.warnings)
    return EXIT_OK, {"reading": r}


def cmd_learn(ctx):
    a = ctx.args
    roi = tuple(_numbers(a.roi, 4, "ROI x0,y0,x1,y1 (camera px)", cast=int))
    if roi[2] <= roi[0] or roi[3] <= roi[1]:
        raise CLIError(f"ROI {a.roi}: need x1 > x0 and y1 > y0 (camera px)")
    reading = _find_spectral(a.reading_id)
    m = api.learn_material(reading, a.name, roi)
    ctx.say(f"Material '{m.name}' learned from {m.n_px} camera px of {_get(reading, 'id')}")
    ctx.table(["band", "signature", "spread"],
              [[b, _u(s, "(fraction)", 4), _f(sp, 4)] for b, s, sp in
               zip(m.bands, m.signature, m.spread)])
    ctx.say(f"  brightness {_u(m.brightness, '(relative reflectance)', 4)}")
    if m.path:
        ctx.say(f"  stored: {m.path}")
    return EXIT_OK, {"material": m}


def cmd_classify(ctx):
    a = ctx.args
    reading = _find_spectral(a.reading_id)
    c = api.classify(reading, a.materials, a.max_distance)
    ctx.say(f"Classification of {c.reading_id} against {', '.join(map(str, c.materials))}")
    ctx.table(["material", "coverage"],
              [[k, _u(v, "% of valid px", 2)] for k, v in (c.coverage_pct or {}).items()])
    ctx.table(["material", "area px", "bbox x0,y0,x1,y1 (px)"],
              [[_get(b, "name"), _get(b, "area_px"),
                ",".join(_f(x, 0) for x in (_get(b, "bbox") or []))] for b in c.blobs[:20]])
    for label, path in (("map", c.map_png), ("report", c.json_path)):
        if path:
            ctx.say(f"  {label}: {path}")
    ctx.warn(c.warnings)
    return EXIT_OK, {"classification": c}


def cmd_light(ctx):
    ctx.header()
    a = ctx.args
    r = api.separate_light(ctx.preset(), a.label, a.square, a.shifts)
    ctx.say(f"Direct vs global light ({r.shifts} shifts of {r.square_px} px checkerboards)")
    ctx.kv([
        ["global fraction", _u(100.0 * r.global_fraction_median, "% (median over lit px)", 1)],
        ["direct", str(r.direct_png or "-")], ["global", str(r.global_png or "-")],
        ["stored", str(r.path or "-")]])
    ctx.warn(r.warnings)
    return EXIT_OK, {"light": r}


def cmd_map(ctx):
    ctx.header()
    m = api.capture_map(ctx.preset(), ctx.args.label)
    n = int(np.count_nonzero(m.mask)) if m.mask is not None else 0
    total = int(m.mask.size) if m.mask is not None else 0
    ctx.say(f"Projector map {m.id}  (geometry {m.geometry_id})")
    ctx.kv([["mapped", f"{n} camera px ({100.0 * n / max(total, 1):.1f} %) "
                                    "know their DMD column AND row"],
                         ["DMD", f"{m.dmd_shape[1]} x {m.dmd_shape[0]} mirrors"],
                         ["stored", str(m.path)]])
    ctx.warn(m.warnings)
    ctx.say("  Next: grid   (then scan <label> --zone C4)")
    return EXIT_OK, {"map": m}


def _print_zoneshot(ctx, shot) -> None:
    g = shot.grid
    if g is not None and g.col_labels and g.row_labels:
        ctx.say(f"Grid: {g.n_cols} columns ({g.col_labels[0]}-{g.col_labels[-1]}) x "
                f"{g.n_rows} rows ({g.row_labels[0]}-{g.row_labels[-1]}), "
                f"{g.spacing_px} DMD px per cell")
    ctx.kv([
        ["camera photo", f"{shot.camera_png or '-'}   <- read the cell labels here"],
        ["overlay", str(shot.overlay_png or "- (needs a projector map: run `map`)")],
        ["sent to DMD", str(shot.dmd_png or "-")],
        ["cells seen", (", ".join(map(str, shot.cells_visible[:40])) +
                        (" ..." if len(shot.cells_visible) > 40 else ""))
         if shot.cells_visible else "-"]])
    ctx.warn(shot.warnings)


def cmd_grid(ctx):
    ctx.header()
    a = ctx.args
    shot = api.project_grid(ctx.preset(), a.spacing, a.label, a.hold)
    _print_zoneshot(ctx, shot)
    howto = ["scan <label> --zone C4        one cell",
             "scan <label> --zone C4:D5     inclusive block",
             "scan <label> --zone B2,E5     union of cells"]
    ctx.say("To scan only where the part is, name its cells:", *(f"  {h}" for h in howto))
    return EXIT_OK, {"grid": shot, "next": {
        "read": "camera_png (grid photo): find the cells the part covers",
        "then": [" ".join(h.split()[:4]) for h in howto]}}


# ------------------------------------------------------------ painted zone sets

def _zone_rows(zones: list) -> list:
    return [[z["id"], z.get("name") or "-", z["shape"],
             " x ".join(f"{v:g}" for v in z["size_px"]) if z.get("size_px") else "-",
             _f(z.get("rotation_deg"), 1),
             ",".join(str(v) for v in z["bbox_dmd_px"]) if z.get("bbox_dmd_px") else "-",
             _f(z.get("area_px"), 0)] for z in zones]


_ZONE_HEAD = ["zone", "name", "shape", "size DMD px", "deg", "box col0,row0,col1,row1", "area px"]


def _zones_text(spec: str) -> str:
    """--zones SET[:Z1,Z3] -> the engine's zone text 'zoneset:SET[:Z1,Z3]' (checked here so
    a typo is refused before anything is loaded)."""
    from engine.zones import use as zone_use
    name, ids = zone_use.parse_spec(spec, bare=True)
    return zone_use.spec_text(name, ids)


def cmd_zone_paint(ctx):
    """The zone painter: draw zones on the part with the mouse (MOCK: a scripted session)."""
    from engine.zones import painter
    a = ctx.args
    events = None
    if a.events:
        if not ctx.mock:
            raise CLIError("--events is MOCK ONLY: on the real rig the operator draws on the part")
        events = list(painter.DEMO_EVENTS) if a.events == "demo" else painter.load_events(a.events)
    elif ctx.mock:
        raise CLIError("MOCK: the painter runs headless from a scripted session: --events "
                       "FILE.json (a list of painter events), or --events demo")
    ctx.header()
    if not ctx.mock:
        ctx.say("Zone painter: the projector window now takes the mouse. Move the cursor onto",
                "the part and draw there (the rig is held until Space / Esc, or "
                f"{a.idle_timeout:g} s without input).")
        ctx.table(["key / mouse", "does"], [list(k) for k in painter.KEYS])
    labels = [s.strip() for s in a.labels.split(",")] if a.labels else None
    out = api.zone_paint(a.name, ctx.preset(), source=a.source, events=events,
                         replace=a.replace, labels=labels, polarity=a.polarity,
                         note=a.note or "", idle_timeout_s=a.idle_timeout)
    ctx.warn(out.get("warnings"))
    if not out["saved"]:
        ctx.say(f"Zone set {a.name}: NOT saved - {_one_line(out['reason'])}")
        return EXIT_NOT_OK, {"zone_paint": out}
    ctx.say(f"Zone set {out['name']} saved: {out['n_zones']} zone(s), geometry "
            f"{out['geometry_id']} (projector px, valid at this rig placement)")
    ctx.table(_ZONE_HEAD, _zone_rows(out["zones"]))
    ctx.kv([["zones", str(out["zones_json"])], ["masks", str(out["masks"])],
            ["preview", str(out["preview"])], ["photo", str(out["camera"] or "-")]])
    ctx.say(f"  Next: scan zone-show {out['name']}   (where the zones land in the camera image)",
            f"        scan scan-fast LABEL --zones {out['name']}[:Z1,Z3]",
            f"        scan zones {out['name']} --recipe   (the zones as a recipe's zones)")
    return EXIT_OK, {"zone_paint": out}


def cmd_zones(ctx):
    """List the painted zone sets, or show one (its zones, a recipe fragment, a scan mask)."""
    a = ctx.args
    if not a.set:
        try:
            here = ctx.preset().geometry_id
        except (config.ConfigError, rules.RuleViolation, CLIError):
            here = None
        items = [i for i in api.zone_sets(all_geometries=True)
                 if not a.geometry or i["geometry_id"] == a.geometry]
        ctx.say(f"Zone sets ({len(items)}; * = painted at this preset's geometry"
                f"{' ' + here if here else ''}):")
        ctx.table(["", "name", "geometry", "zones", "created", ""],
                  [["*" if i["geometry_id"] == here else "", i["name"], i["geometry_id"],
                    ", ".join(z["id"] + (f" ({z['name']})" if z.get("name") else "")
                              for z in i["zones"]) or "-",
                    i.get("created") or "-",
                    "" if i["valid"] else f"INVALID: {_one_line(i['error'])}"] for i in items])
        if not items:
            ctx.say("  none yet - paint one on the part: scan zone-paint --name NAME")
        return EXIT_OK, {"zone_sets": items}
    zs = api.zone_set(a.set, ctx.preset())
    text = _zones_text(a.set)
    ids = text.split(":")[2].split(",") if text.count(":") > 1 else None
    summary = zs.summary()
    chosen = [z for z in summary["zones"] if ids is None or z["id"] in ids]
    ctx.say(f"Zone set {zs.name}  (geometry {zs.geometry_id}, preset {zs.preset or '?'}, "
            f"painted {zs.created or '?'})")
    ctx.table(_ZONE_HEAD, _zone_rows(chosen))
    ctx.kv([["stored", str(zs.path)], ["use as", f"--zones {zs.name}" + (":" + ",".join(ids)
                                                                         if ids else "")]])
    changes = next((i.get("rig_changes") for i in api.zone_sets(ctx.preset())
                    if i["name"] == zs.name), None) or []
    if changes:
        ctx.warn([f"since it was painted the rig-state record has: {'; '.join(changes[:4])} - "
                  f"check the zones (scan zone-show {zs.name}) or paint the set again"])
    data = {"zone_set": {**summary, "zones": chosen}, "spec": text, "rig_changes": changes}
    if a.recipe or a.by_ref:
        data["recipe_zones"] = api.zone_recipe(a.set, ctx.preset(), by_ref=a.by_ref)
        ctx.say("", "recipe zones (paste under \"zones\" in a recipe; edit the checks; a "
                "projector_px zone needs scan.axis xy):",
                json.dumps(data["recipe_zones"], indent=2))
    if a.mask_out:
        try:
            data["mask"] = str(api.zone_mask(a.set, ctx.preset(), out=a.mask_out))
        except FileExistsError:
            raise CLIError(f"--mask-out {a.mask_out}: the file exists (never overwritten)") from None
        ctx.say(f"  mask written: {data['mask']}  (scan scan-fast LABEL --mask {data['mask']})")
    return EXIT_OK, data


def cmd_zone_show(ctx):
    """Project a saved zone set, photograph it, report where each zone lands."""
    ctx.header()
    a = ctx.args
    r = api.zone_show(a.set, ctx.preset(), a.hold, a.label)
    ctx.say(f"Zone set {r['zoneset']}: {sum(z['seen'] for z in r['zones'])} of "
            f"{len(r['zones'])} zone(s) land in the camera image")
    ctx.table(["zone", "name", "seen", "camera px", "centre x,y", "box x0,y0,x1,y1", "DMD px"],
              [[z["id"], z.get("name") or "-", _f(z["seen"]), z["camera_px"],
                ",".join(_f(v, 0) for v in z["centroid_px"]) if z["centroid_px"] else "-",
                ",".join(str(v) for v in z["bbox_px"]) if z["bbox_px"] else "-", z["dmd_px"]]
               for z in r["zones"]])
    ctx.kv([["camera photo", str(r["camera_png"] or "-")], ["overlay", str(r["overlay_png"] or "-")],
            ["sent to DMD", str(r["dmd_png"] or "-")], ["record", str(r["json_path"])]])
    ctx.warn(r["warnings"])
    return (EXIT_OK if r["all_seen"] else EXIT_NOT_OK), {"zone_show": r}


def cmd_zone_delete(ctx):
    a = ctx.args
    r = api.zone_delete(a.name, a.reason, ctx.preset(), force=a.force)
    ctx.say(f"Zone set {r['name']} moved to the library trash: {r['trashed']}")
    if r["recipes"]:
        ctx.say(f"  ! recipe(s) {', '.join(r['recipes'])} named it (forced)")
    return EXIT_OK, {"deleted": r}


def _load_patterns(src: str) -> list:
    """DMD images from an .npz (key 'patterns', else its first array) or a folder of PNGs
    (sorted by name; grey stays one channel)."""
    p = Path(src)
    if p.is_dir():
        import cv2
        files = sorted(p.glob("*.png"))
        if not files:
            raise ValueError(f"no .png files in {p}")
        out = []
        for f in files:
            img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"could not read {f}")
            out.append(img[..., :3] if img.ndim == 3 else img)
        return out
    if p.suffix.lower() == ".npz" and p.is_file():
        with np.load(p) as z:
            if not z.files:
                raise ValueError(f"{p} holds no arrays")
            return z["patterns" if "patterns" in z.files else z.files[0]]   # api splits a stack
    raise ValueError(f"{src}: expected an .npz of patterns or a folder of .png files")


def cmd_project_capture(ctx):
    ctx.header()
    a = ctx.args
    # the global --average already sets camera.average on ctx.preset() for this run
    r = api.project_capture(_load_patterns(a.patterns), ctx.preset(), a.label, a.raw,
                            hold_s=a.hold)
    ctx.say(f"Captured {r['n']} frames in {r['capture_s']} s")
    ctx.kv([["frames", str(r["frames_npz"])], ["meta", str(r["meta_json"])]])
    return EXIT_OK, {"project_capture": r}


# ------------------------------------------------------------ virtual sensors

def _pt(p, nd: int = 1) -> str:
    """[x, y] -> 'x, y' ('-' when missing)."""
    return "-" if p is None else ", ".join(_f(float(v), nd) for v in p)


def _at_text(at) -> str:
    """An `at` as typed or as the engine stores it -> short text, in the form
    the commands accept (dmd:456,570), so it can be pasted back."""
    if not isinstance(at, dict):
        return str(at)
    sp = f"@{at['spacing_px']}" if at.get("spacing_px") else ""
    g = (lambda p: ",".join(f"{float(v):g}" for v in p))
    if at.get("cell"):
        return f"{at['cell']}{sp}"
    if at.get("zone"):
        return f"{at['zone']}{sp} (zone)"
    if at.get("camera") is not None:
        return f"cam:{g(at['camera'])}" + (f" (dmd:{g(at['dmd'])})" if at.get("dmd") else "")
    if at.get("dmd") is not None:
        return f"dmd:{g(at['dmd'])}"
    return str(at)


def _dmd_size(s) -> str:
    """A spot on the DMD: columns x rows (diamond array: rows are half the column pitch)."""
    return f"{_f(s.diameter_dmd_px, 2)} x {_f(s.diameter_dmd_rows, 2)} DMD px (cols x rows)"


def cmd_spot_size(ctx):
    a = ctx.args
    p = ctx.preset()
    nominal = measured = None
    nominal_error = None
    try:
        nominal = api.spot_size(a.diameter, None, p)
    except ValueError as e:              # no rig.standoff_mm: fine when a measured scale is given
        if a.measured is None:
            raise
        nominal_error = _one_line(e)
    if a.measured is not None:
        measured = api.spot_size(a.diameter, a.measured, p)
    used = measured or nominal
    ctx.say(f"Spot of {a.diameter:g} mm at the part -> {_dmd_size(used)}, {used.source} scale")
    ctx.table(["scale", "mm per DMD column", "spot on the DMD", "from"],
              [[name, _u(s.mm_per_dmd_px, "mm", 4), _dmd_size(s),
                "your measurement" if s.source == "measured" else
                f"standoff {_f(s.standoff_mm, 0)} mm / throw ratio {_f(s.throw_ratio, 2)} / 912"]
               for name, s in (("nominal", nominal), ("measured", measured)) if s is not None])
    if nominal is not None and measured is not None:
        ctx.say(f"  your beam: {100 * (measured.mm_per_dmd_px / nominal.mm_per_dmd_px - 1):+.1f} % "
                "vs nominal; spot / teach-presence use the scale you pass with --measured")
    if nominal_error:
        ctx.say(f"  nominal: not available ({nominal_error})")
    ctx.say("  DLP4500 diamond array: rows sit at half the column pitch, so a round spot is "
            "drawn with twice as many rows as columns.")
    ctx.warn(used.warnings)
    if measured is None:
        ctx.say("  Measure your beam: spot <cell> --diameter D, measure it on the part, then "
                "--measured <measured mm / drawn columns>")
    return EXIT_OK, {"diameter_mm": a.diameter, "nominal": nominal, "measured": measured,
                     "nominal_error": nominal_error, "used": used.source}


def _print_spot(ctx, r) -> None:
    s = r.spec
    ctx.say(f"Spot {'FOUND' if r.found else 'NOT FOUND'} at {_at_text(r.at)}  "
            f"({s.source} scale {_f(s.mm_per_dmd_px, 4)} mm per DMD column)")
    diam = _u(r.diameter_px, "camera px", 2) + (
        f" = {_u(r.diameter_mm, 'mm', 3)}" if r.diameter_mm is not None
        else " (no lateral calibration: px only, rule 6)")
    ctx.kv([
        ["DMD target", f"{_pt(r.target_dmd)} DMD px [col, row]"],
        ["drawn", f"{_dmd_size(s)} = {s.diameter_mm:g} mm"],
        ["centroid", f"{_pt(r.centroid_px)} camera px"],
        ["expected", f"{_pt(r.expected_px)} camera px" + (" (projector map)" if r.map_id
                                                          else " (needs a projector map)")],
        ["offset", f"{_pt(r.offset_px)} camera px (centroid - expected)"],
        ["diameter", diam],
        ["extent", f"{r.extent_px[0]} x {r.extent_px[1]} camera px" if r.extent_px else "-"],
        ["peak", f"{_u(r.peak_counts, 'counts above dark', 1)}, SNR {_f(r.snr, 1)}"],
        ["photo", str(r.photo_png or "-")], ["overlay", str(r.overlay_png or "-")],
    ])
    ctx.warn(r.warnings)


def cmd_spot(ctx):
    a = ctx.args
    at = parse_at(a.at)
    ctx.header()
    r = api.project_spot(at, a.diameter, ctx.preset(),
                         **_kw(mm_per_dmd_px=a.measured, shape=a.shape, hold_s=a.hold,
                               label=a.label, level=a.level))
    _print_spot(ctx, r)
    return (EXIT_OK if r.found else EXIT_NOT_OK), {"spot": r}


def _print_teach(ctx, t) -> None:
    ctx.say(f"Sensor '{t.name}' taught at {t.geometry_id}: {t.method}, {t.mode}")
    trig = (f"|height| > {t.min_height_mm:g} mm above the taught background -> PRESENT"
            if t.method == "bgs" else
            f"|change| > {100 * t.threshold:g} % of the empty reference -> PRESENT")
    rows = [["at", _at_text(t.at)],
            ["reference", _u(t.reference_signal, "counts per lit px (EMPTY field)", 2)],
            ["lit area", _u(t.n_px, "camera px")],
            ["window", f"{_pt(t.window, 0)} camera px [x0, y0, x1, y1)"],
            ["trigger", trig]]
    if t.spot is not None:
        rows.append(["spot", f"{t.spot.diameter_mm:g} mm = {_dmd_size(t.spot)}, "
                             f"{t.spot.source} scale"])
    if t.jacobian is not None:
        rows.append(["jacobian", f"{t.jacobian} camera px per DMD px"])
    rows += [["photo", str(t.photo_png or "-")], ["stored", str(t.version_path)]]
    ctx.kv(rows)
    ctx.warn(t.warnings)
    ctx.say(f"  Next: place a part, then  check-presence {t.name}")


def cmd_teach_presence(ctx):
    a = ctx.args
    at = parse_at(a.at)
    # clearing the field is physical and human-only: the operator says it is done - as an
    # argument (--empty-confirmed / --field-empty yes). Without it the command is refused at
    # once: it never waits for an answer (seen on a rig: it sat on its y/N question for two
    # days in a session nobody was typing into). --interactive asks, with a timeout.
    if a.field_empty == "no":
        if a.empty_confirmed:
            raise CLIError("--empty-confirmed and --field-empty no contradict each other")
        ctx.say("Not taught: the sensor's field is not empty (--field-empty no). Clear it, then "
                "run again with --empty-confirmed.")
        return EXIT_NOT_OK, {"sensor": None, "cancelled": True}
    confirmed = bool(a.empty_confirmed or a.field_empty == "yes")
    if not confirmed:
        if ctx.json:
            raise CLIError("teach-presence with --json needs --empty-confirmed (no interactive "
                           "prompt): the operator confirms the sensor's field is EMPTY")
        if not ctx.interactive:
            raise CLIError(
                "teach-presence records the EMPTY reference: have a person clear the sensor's "
                f"field (nothing in the light at {_at_text(at)}), then run it with "
                "--empty-confirmed (or --field-empty yes). This command does not wait for "
                "input; --interactive asks [y/N] and gives up after --prompt-timeout s. "
                "Nothing was projected or taught")
    ctx.header()
    p = ctx.preset()
    _mock_clear(ctx)
    if not confirmed:
        try:
            ans = _prompt(ctx, f">> Is the sensor's field EMPTY (nothing in the light at "
                               f"{_at_text(at)})? [y/N] ")
        except EOFError:
            ans = ""
        if ans.strip().lower() not in ("y", "yes"):
            ctx.say("Not taught: clear the sensor's field, then run again "
                    "(or pass --empty-confirmed).")
            return EXIT_NOT_OK, {"sensor": None, "cancelled": True}
    t = api.teach_presence(a.name, at, a.diameter, a.method, p, empty_confirmed=True,
                           **_kw(mm_per_dmd_px=a.measured, threshold=a.threshold,
                                 min_height_mm=a.min_height, level=a.level))
    _print_teach(ctx, t)
    return EXIT_OK, {"sensor": t}


def _print_presence(ctx, r) -> None:
    ctx.say(f"Sensor '{r.name}' ({r.method}): {r.verdict}")
    if r.method == "bgs":
        src = {"calibrated": "depth calibration", "nominal": "NOMINAL rig estimate"}.get(
            r.height_source, "no mm scale")
        rows = [["height", f"{_u(r.height_mm, 'mm', 2)} ({src})"],
                ["trigger", f"|height| > {_u(r.min_height_mm, 'mm', 1)}"],
                ["relief", _u(r.relief_px, "DMD px", 3)],
                ["spot shift", _u(r.shift_px, "camera px along the baseline", 2)],
                ["off-axis", _u(r.off_axis_px, "camera px across the baseline", 2)],
                ["centroid", f"{_pt(r.centroid_px)} camera px (taught "
                             f"{_pt(r.reference_centroid_px)})"]]
    else:
        pct = (lambda v: None if v is None else 100.0 * v)
        rows = [["signal", _u(r.signal, "counts per lit px", 2)],
                ["reference", _u(r.reference_signal, "counts per lit px (EMPTY field)", 2)],
                ["change", _u(pct(r.relative_change), "%", 1)],
                ["trigger", f"|change| > {_u(pct(r.threshold), '%', 0)}"],
                ["saturated", _u(pct(r.saturated_fraction), "% of lit px", 1)]]
    rows += [["photo", str(r.photo_png or "-")], ["report", str(r.json_path or "-")]]
    ctx.kv(rows)
    ctx.warn(r.warnings)


def cmd_check_presence(ctx):
    a = ctx.args
    ctx.header()
    p = ctx.preset()
    _mock_place(ctx)
    r = api.check_presence(a.name, p, **_kw(threshold=a.threshold, min_height_mm=a.min_height,
                                            min_relief_px=a.min_relief_px, label=a.label))
    _print_presence(ctx, r)
    return (EXIT_OK if r.verdict == "PRESENT" else EXIT_NOT_OK), {"presence": r}


def cmd_sensors(ctx):
    rows_ = api.list_library("sensors", ctx.args.geometry)
    rows = []
    for s in rows_:
        if s.get("kind") == "presence":
            what = f"{s.get('method')}, {s.get('mode')}"
            trig = (f"|height| > {_f(s.get('min_height_mm'), 1)} mm" if s.get("method") == "bgs"
                    else f"|change| > {_f(100 * (s.get('threshold') or 0), 0)} %")
            where = _at_text(s.get("at"))
        else:
            what, trig = "alignment", f"pose {_f(s.get('angle_deg'), 1)} deg"
            where = f"cam:{_pt(s.get('centroid_px'), 0)}"
        rows.append([s["name"], what, where, trig, s.get("geometry_id"), s.get("versions"),
                     s.get("created")])
    ctx.say(f"Sensors ({len(rows_)})" + (f" for geometry {ctx.args.geometry}"
                                          if ctx.args.geometry else ""))
    ctx.table(["name", "type", "at", "trigger", "geometry", "versions", "taught"], rows)
    if not rows_:
        ctx.say("  none yet: teach-presence <name> <cell> --diameter D   or   "
                "teach-align <name> <scan>")
    return EXIT_OK, {"sensors": rows_}


def cmd_teach_align(ctx):
    p = ctx.preset()
    scan = _find_scan(ctx.args.scan)
    ref = api.teach_alignment(ctx.args.name, scan, p)
    ctx.say(f"Alignment '{ref.name}' taught from {scan.id} at {ref.geometry_id}")
    ctx.kv([["centroid", f"{_pt(ref.centroid_px)} camera px"],
            ["angle", f"{_f(ref.angle_deg, 2)} deg (major axis; x right, y down)"],
            ["extents", f"{_pt(ref.extents_px)} camera px (major, minor)"],
            ["area", _u(ref.area_px, "camera px")],
            ["axis ratio", f"{_f(ref.axis_ratio, 2)} (the angle needs >= 1.15)"],
            ["stored", str(ref.version_path)]])
    ctx.warn(ref.warnings)
    ctx.say(f"  Next: scan the next part, then  check-align {ref.name} <scan>")
    return EXIT_OK, {"alignment": ref}


def cmd_check_align(ctx):
    a = ctx.args
    p = ctx.preset()
    scan = _find_scan(a.scan)
    r = api.check_alignment(a.name, scan, a.tol_mm, a.tol_deg, a.tol_px, p)
    judged = {True: "ok", False: "OUT", None: "not judged"}
    ctx.say(f"Alignment of {scan.id} vs '{r.name}': {r.verdict}  (position in {r.position_units})")
    rows = []
    if r.position_units == "mm":
        rows += [["dx, dy", f"{_f(r.dx_mm, 3)}, {_f(r.dy_mm, 3)} mm"],
                 ["offset", f"{_u(r.offset_mm, 'mm', 3)} (tolerance {_u(r.tol_mm, 'mm', 2)}): "
                            f"{judged[r.position_ok]}"]]
    rows.append(["dx, dy (px)", f"{_f(r.dx_px, 2)}, {_f(r.dy_px, 2)} camera px"])
    if r.position_units != "mm":
        tol = _u(r.tol_px, "px", 1) if r.tol_px is not None else "none: pass --tol-px"
        rows.append(["offset", f"{_u(r.offset_px, 'camera px', 2)} (tolerance {tol}): "
                               f"{judged[r.position_ok]}"])
    rows += [["dtheta", f"{_u(r.dtheta_deg, 'deg', 2)} (tolerance {_u(r.tol_deg, 'deg', 1)}): "
                        f"{judged[r.angle_ok]}"],
             ["area vs taught", f"{_f(r.area_ratio, 3)} x"],
             ["overlay", str(r.overlay_png or "-")], ["report", str(r.json_path or "-")]]
    ctx.kv(rows)
    ctx.warn(r.warnings)
    return (EXIT_OK if r.verdict == "ALIGNED" else EXIT_NOT_OK), {"scan_id": scan.id,
                                                                   "alignment": r}


# ------------------------------------------------------------ library listings

def cmd_calibrations(ctx):
    cals = api.list_library("calibrations", ctx.args.geometry)
    in_use = None
    try:
        p = ctx.preset()
        cur = api.calibration_for(p.geometry_id, config.geometry_fingerprint(p))
        in_use = cur.id if cur is not None else None
    except (config.ConfigError, rules.RuleViolation):
        pass                                    # listing still works without a preset

    def note(c) -> str:
        if c.get("id") == in_use:
            return "<- in use for the preset"
        if c.get("kind") == "stereo":       # a re-homed copy says where it came from
            return "stereo" + (f": {c['rehomed'].get('note')}" if c.get("rehomed") else "")
        return ""
    rows = [[c.get("id"), c.get("geometry_id"),
             _u(c.get("mm_per_px"), "mm/px", 5) if c.get("mm_per_px") is not None else "-",
             _u(c.get("lateral_px_per_mm"), "px/mm", 4)
             if c.get("lateral_px_per_mm") is not None else "-",
             _f(c.get("r2"), 5), c.get("n_samples"), note(c)] for c in cals]
    ctx.say(f"Calibrations ({len(cals)})" + (f" for geometry {ctx.args.geometry}"
                                              if ctx.args.geometry else ""))
    ctx.table(["id", "geometry", "Z", "XY", "R^2", "samples", ""], rows)
    if not cals:
        ctx.say("  none yet: scan a part with --known H,W,L, then  calibrate <label>")
    return EXIT_OK, {"calibrations": cals, "in_use": in_use}


def cmd_calib_poses(ctx):
    a = ctx.args
    poses = [m for m in api.list_library("calib_poses", a.geometry)
             if a.set is None or m.get("set") == a.set]
    ctx.say(f"Calibration poses ({len(poses)})" + (f" of set {a.set}" if a.set else "")
            + (f" for geometry {a.geometry}" if a.geometry else ""))
    ctx.table(["id", "set", "geometry", "corners", "projector corners", "periods (px)",
               "accepted", "captured"],
              [[m.get("id"), m.get("set"), m.get("geometry_id"), m.get("corners_found"),
                (f"{m['projector_corners']} ({m.get('projector_corner_pct', 0):.0f} %)"
                 if m.get("projector_corners") is not None else "-"),
                _periods_words(m.get("periods_px")),
                {True: "yes", False: "NO"}.get(m.get("accepted"), "-"), m.get("created")]
               for m in poses])
    if not poses:
        ctx.say("  none yet: calib-pose [--set NAME]  (12-20 board poses, then calibrate-stereo)")
    else:
        ctx.say("  delete <id> --reason WHY --yes  moves one pose to the library trash")
    return EXIT_OK, {"calib_poses": [{k: v for k, v in m.items() if k not in ("decode", "fixture",
                                                                              "boards")}
                                     for m in poses]}


def cmd_readings(ctx):
    codes = api.list_library("readings")
    try:
        spectral = api.list_library("spectral")
    except NotImplementedError:
        spectral = []
    rows = []
    for r in codes:
        found = ", ".join(f"{c.get('kind', '')} {c.get('data', '')}".strip()
                          for c in r.get("codes") or []) or "no code"
        rows.append(["codes", r["id"], found, r.get("preset") or "-", str(r["path"])])
    for s in spectral:
        rows.append(["spectral", _get(s, "id"), f"bands {','.join(_get(s, 'bands') or [])}, "
                     f"valid {_f(_get(s, 'valid_pct'), 1)} %", _get(s, "geometry_id"),
                     str(_get(s, "path"))])
    ctx.say(f"Readings: {len(codes)} code, {len(spectral)} spectral")
    ctx.table(["kind", "id", "content", "preset / geometry", "stored"], rows)
    return EXIT_OK, {"code_readings": codes, "spectral_readings": spectral}


def cmd_materials(ctx):
    mats = api.list_library("materials")
    rows = [[m.name, " / ".join(f"{b} {_f(v, 3)}" for b, v in zip(m.bands, m.signature)),
             _f(m.brightness, 3), _u(m.n_px, "px"), m.geometry_id or "-", m.created or "-"]
            for m in mats]
    ctx.say(f"Materials ({len(mats)})")
    ctx.table(["name", "signature (fraction per band)", "brightness", "learned from",
               "geometry", "created"], rows)
    if not mats:
        ctx.say("  none yet: spectral, then  learn <reading> <name> x0,y0,x1,y1")
    return EXIT_OK, {"materials": mats}


def cmd_banks(ctx):
    banks = api.list_library("banks", ctx.args.geometry)
    rows = [[b.name, b.geometry_id, b.n_scans, b.zone or "-",
             ", ".join(b.scan_ids[-3:]) + (" ..." if len(b.scan_ids) > 3 else "")]
            for b in banks]
    ctx.say(f"Good memory banks ({len(banks)})")
    ctx.table(["name", "geometry", "good scans", "zone", "newest members"], rows)
    if not banks:
        ctx.say("  none yet: bank-add <scan> --bank NAME")
    return EXIT_OK, {"banks": banks}


def cmd_maps(ctx):
    if getattr(ctx.args, "id", None):           # maps <scan>: bake the surface map stack
        ctx.mode = "surface maps"
        return cmd_maps_bake(ctx)
    maps = api.list_library("maps", ctx.args.geometry)
    rows = [[m["id"], m.get("label") or "-", m.get("geometry_id"),
             _u(m.get("valid_pct"), "%", 1), m.get("created") or "-"] for m in maps]
    ctx.say(f"Projector maps ({len(maps)})")
    ctx.table(["id", "label", "geometry", "mapped", "created"], rows)
    if not maps:
        ctx.say("  none yet: map   (the scene in place, rig fixed)")
    return EXIT_OK, {"maps": maps}


def cmd_bank_remove(ctx):
    ctx.preset()
    scan = _find_scan(ctx.args.id)
    info = api.bank_remove(scan, ctx.args.bank, ctx.args.reason)
    ctx.say(f"Removed {scan.id} from bank '{info.name}' ({info.geometry_id}): "
            f"{info.n_scans} known-good scan(s) left",
            "  (the scan itself is kept; the bank records the removal, append-only)")
    ctx.warn(info.warnings)
    return EXIT_OK, {"bank": info, "removed": scan.id}


def cmd_outputs(ctx):
    a = ctx.args
    rows = api.list_library("outputs", a.kind)
    root = api.paths()["library"]
    shown = rows[-a.last:] if a.last > 0 else rows
    ctx.say(f"Output bin {api.paths()['output']}: {len(rows)} file(s)"
            + (f" of kind {a.kind}" if a.kind else "")
            + (f", showing the last {len(shown)} (--last 0 for all)" if len(shown) < len(rows)
               else ""))
    ctx.table(["time", "kind", "label", "file"],
              [[r["time"], r["kind"], r["label"], str(root / r["file"])] for r in shown])
    return EXIT_OK, {"outputs": [{**r, "path": str(root / r["file"])} for r in shown],
                     "total": len(rows)}


# ------------------------------------------------------------ wizard

def cmd_wizard(ctx):
    """base -> part -> measure -> inspect -> 3D, prompting for every
    physical step. Always human-readable (the prompts need a person). Needs --interactive
    (it is nothing but prompts); each prompt gives up after --prompt-timeout s (default
    WIZARD_TIMEOUT_S) and the wizard stops."""
    ctx.json = False
    a = ctx.args
    if not ctx.interactive:
        raise CLIError("the wizard asks a person at every physical step, and no command waits "
                       "for input unless --interactive is given: run  scan wizard "
                       "--interactive  from a terminal (each prompt gives up after "
                       f"--prompt-timeout s, default {WIZARD_TIMEOUT_S:g}). For scripts and "
                       "agents use the single commands: signal, base, scan, measure, inspect, "
                       "reconstruct")

    def _ask(prompt: str) -> str:
        """One human step; q / EOF / no answer in time stop the wizard cleanly."""
        try:
            ans = _prompt(ctx, prompt, WIZARD_TIMEOUT_S)
        except EOFError:
            raise Abort("no input (stdin closed); wizard stopped") from None
        if ans.strip().lower() in ("q", "quit", "exit"):
            raise Abort("stopped by the operator")
        return ans

    p = ctx.preset()
    ctx.say("Scan wizard: empty base -> part -> measure -> inspect -> 3D point cloud")
    ctx.header()
    ctx.say("Physical steps are yours: the wizard waits for you at each one. Type q to quit.", "")

    ctx.say("Step 1/5  Empty base")
    _ask(">> Clear the scene (empty base), press Enter ")
    _mock_clear(ctx)
    while True:
        sig = api.check_signal(p)
        _print_signal(ctx, sig)
        if sig.ok:
            break
        ans = _ask(">> Signal is NOT ok. Fix the setup (aim, focus, exposure) and press Enter "
                   "to re-check, or type 'force' to continue anyway: ")
        if ans.strip().lower() == "force":
            ctx.say("  continuing on a signal that is not ok (forced by the operator)")
            break
    base = api.capture_base(p, a.base_label)
    ctx.say(f"  base {base.id} stored ({_u(base.meta.get('valid_pct'), '% valid', 1)})")
    ctx.warn(base.warnings)
    if base.meta.get("rejected"):
        raise Abort("the base was auto-REJECTED (almost nothing decoded): fix the setup, then "
                    "run the wizard again")
    ctx.say("")

    ctx.say("Step 2/5  Part")
    _ask(">> Place the part, do not touch camera/projector, press Enter ")
    _mock_place(ctx)
    label = ""
    while not re.search(r"[A-Za-z0-9_.\-]", label):
        label = _ask(f">> Part label [{a.label}]: ").strip() or a.label
    hint = f" (mock part: {VALIDATION_DIMS})" if ctx.mock and a.mock_part == "validation" else ""
    while True:
        text = _ask(f">> Known dimensions in mm as H,W,L{hint} (optional, Enter to skip): ")
        try:
            known = parse_known(text)
            break
        except CLIError as e:
            ctx.say(f"  {e}")
    zone = None
    if _ask(">> Scan only one zone of a projected grid? [y/N] ").strip().lower() in ("y", "yes"):
        try:
            shot = api.project_grid(p)
            _print_zoneshot(ctx, shot)
            zone = _ask(">> Zone from the grid photo (e.g. C4 or C4:D5; Enter = whole field): "
                        ).strip() or None
        except NotImplementedError:
            ctx.say("  the projected grid is not available yet; scanning the whole field")
    try:
        scan = api.capture_object(p, base, label, known_dims=known, zone=zone)
    except NotImplementedError:
        if zone is None:
            raise
        ctx.say("  zone scanning is not available yet; scanning the whole field")
        scan = api.capture_object(p, base, label, known_dims=known)
    _print_scan(ctx, scan, base)
    ctx.say("")

    ctx.say("Step 3/5  Measure")
    dims = None
    try:
        dims = api.measure(scan)
        _print_dims(ctx, dims, scan)
    except ValueError as e:                 # e.g. no part found in the relief
        ctx.say(f"  measure: {_one_line(e)}")
    ctx.say("", "Step 4/5  Inspect")
    insp = api.inspect(scan)
    _print_inspection(ctx, insp, scan)
    _print_inspection_files(ctx, insp)
    ctx.say("", "Step 5/5  3D point cloud")
    ply = api.reconstruct(scan)
    ctx.say(f"  {ply}  (units {_pointcloud_units(ply)})")

    ctx.say("", "Library entries:", f"  base  {base.path}", f"  scan  {scan.path}")
    ctx.report_outputs("Output bin, written by this wizard")
    if known and dims is not None and dims.units != "mm":
        ctx.say("", f"This part has known dims: calibrate the rig with  calibrate {scan.label}"
                    "  then  measure again for mm.")
    return EXIT_OK, {"base": base, "scan": scan, "dimensions": dims, "inspection": insp,
                     "pointcloud": ply}


# ------------------------------------------------------------ parser

# ------------------------------------------------ 3D meshes, cell, maps, surfer

def _new_files(ctx, n0: int, exts=(".png",)) -> dict:
    """Output-bin files written since index row n0 -> {name without timestamp: path}."""
    store = ctx.session.store
    out = {}
    for r in store.outputs()[n0:]:
        f = store.root / r["file"]
        if f.suffix.lower() in exts:
            out[re.sub(r"^\d{8}T\d{6}_\d+_", "", f.name)] = f
    return out


def cmd_mesh(ctx):
    a = ctx.args
    ctx.preset()
    scan = _find_scan(a.id)
    p = Path(api.export_mesh(scan, a.fmt, a.stride, a.kind, a.radius))
    mp = p.with_name(p.name + ".json")
    try:
        meta = json.loads(mp.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        meta, mp = {}, None
    units = (meta.get("units") or {}).get("xyz", "?")
    ctx.say(f"{a.kind} mesh of {scan.id} ({a.fmt.upper()}, stride {a.stride}): units {units}",
            f"  {p}")
    ctx.kv([["vertices", meta.get("n_vertices", "?")], ["triangles", meta.get("n_triangles", "?")],
            ["finite vertices", meta.get("finite_vertices", "?")],
            ["to scale", meta.get("to_scale", "?")], ["meta", str(mp or "-")]])
    if a.kind == "cylinder":
        ctx.kv([["arc", _u(meta.get("arc_deg"), "deg", 1)],
                ["end-face height", _u(meta.get("cell_height"), units, 2)]])
    if meta.get("note"):
        ctx.say(f"  note: {meta['note']}")
    if units != "mm":
        ctx.warn(["uncalibrated: coordinates are not mm (rule 6)"])
    return EXIT_OK, {"scan_id": scan.id, "file": p, "meta_file": mp, "meta": meta}


def cmd_unwrap(ctx):
    a = ctx.args
    ctx.preset()
    scan = _find_scan(a.id)
    n0 = len(ctx.session.store.outputs())
    u = api.cell_unwrap(scan, a.radius, a.step)
    files = _new_files(ctx, n0, (".png", ".npz"))
    ctx.say(f"Cylinder unwrap of {scan.id}  (units {u.units}, {u.calibration_id})")
    ctx.kv([["seen arc", _u(u.arc_deg, "deg (open arc only)", 1)],
            ["end-face height", _u(u.cell_height, u.units, 2)], ["radius", _u(u.radius, u.units, 3)],
            ["grid step", _u(u.step, u.units, 3)], ["fit rms", _u(u.fit.rms, u.units, 4)],
            *[[k, str(v)] for k, v in files.items()]])
    ctx.warn(u.warnings)
    summary = {"arc_deg": u.arc_deg, "cell_height": u.cell_height, "radius": u.radius,
               "step": u.step, "units": u.units, "calibration_id": u.calibration_id,
               "fit": u.fit, "stats": u.stats, "warnings": u.warnings}
    return EXIT_OK, {"scan_id": scan.id, "unwrap": summary,
                     "files": {k: str(v) for k, v in files.items()}}


def cmd_dents(ctx):
    a = ctx.args
    ctx.preset()
    scan = _find_scan(a.id)
    rep = api.cell_dents(scan, a.radius, a.sigma, a.min_area, a.trigger)
    ctx.say(f"Cylinder dents of {scan.id}: {rep.verdict}  (units {rep.units}, seen arc "
            f"{_f(rep.arc_deg, 1)} deg)")
    ctx.kv([["noise sigma", _u(rep.noise_sigma, rep.units, 4)],
            ["threshold", _u(rep.threshold, rep.units, 4)],
            ["trigger", _u(rep.trigger, rep.units, 3) if rep.trigger is not None else "-"]])
    ctx.table(["theta deg", "z", "depth", "area bins", "snr", "over trigger"],
              [[_f(d.theta_deg, 1), _f(d.z, 2), _u(d.depth, d.units, 3), d.area_bins,
                _f(d.snr, 1), {True: "yes", False: "no"}.get(d.exceeds_trigger, "-")]
               for d in rep.dents])
    if not rep.dents:
        ctx.say("  no dents found")
    ctx.warn(rep.warnings)
    return (EXIT_OK if rep.verdict == "PASS" else EXIT_NOT_OK), {"scan_id": scan.id,
                                                                  "dents": rep}


# ------------------------------------------ metric calibration + measurement

def _find_metric(key: str):
    return _pick(api.list_library("metric"), key, "metric scan", "metric")


def _periods(text):
    return None if not text else [float(x) for x in str(text).replace(";", ",").split(",") if x]


def cmd_charuco(ctx):
    a = ctx.args
    r = api.make_charuco(a.squares_x, a.squares_y, a.square, a.marker, a.dictionary, a.dpi)
    ctx.say(f"ChArUco board {a.squares_x}x{a.squares_y}, square {a.square:g} mm, marker "
            f"{a.marker:g} mm ({r['board_mm'][0]:g} x {r['board_mm'][1]:g} mm)")
    ctx.kv([["pdf (print at 100 %)", r["pdf"]], ["png", r["png"]], ["spec", r["spec_json"]]])
    ctx.say("  then measure a printed square with calipers: calibrate-stereo --square MM")
    return EXIT_OK, r


def cmd_parity(ctx):
    p = ctx.preset()
    if ctx.mock:            # the diamond DMD model every metric command runs on, an empty wall
        bench = api.mock_bench()
        bench.clear()
        bench.use_stereo()
        ctx.say("[mock] stereo bench: the empty wall (the diamond DMD model the metric "
                "commands use)")
    r = api.row_parity(p)
    if r.get("ok"):         # a result either way: "none" = measured, no half-column offset
        ctx.say(f"Diamond row parity: {_parity_words(r.get('parity'), 'row_parity')}; stored "
                "with the geometry: calib-pose and calibrate-stereo use it")
    else:
        ctx.say(f"Diamond row parity: NOT determined ({r.get('note')}); nothing stored")
    ctx.kv([["shift", _u(r.get("shift_cols"), "columns", 3)],
            ["camera px / column", _f(r.get("camera_px_per_column"), 3)],
            ["lines matched", r.get("n_lines", "-")]])
    return (EXIT_OK if r.get("ok") else EXIT_NOT_OK), r


def _fixture_arg(path):
    """--fixture SPEC: read and validated here, so a bad spec is refused before the rig."""
    if not path:
        return None
    from engine.calibrate import fixture as fixture_mod
    return fixture_mod.load(Path(path))


def _mock_place_fixture(ctx, fxs: dict, set_label: str) -> None:
    """Mock only: the simulated human sets the fixture down at the next placement of the
    plan (left, right, centre, ... each turned differently)."""
    from engine import config as cfg
    from engine.capture.mock_stereo import fixture_placements, fixture_shapes
    p = ctx.preset()
    fp = cfg.geometry_fingerprint(p)
    done = [x for x in api.list_library("calib_poses", p.geometry_id)
            if x.get("set") == set_label and x.get("fixture")
            and not rules.fingerprint_diffs(x.get("geometry_fp"), fp)]
    R, t = fixture_placements(len(done) + 1)[len(done)]
    bench = api.mock_bench()
    bench.clear()
    bench.use_stereo()
    for shape in fixture_shapes(fxs, R, t):
        bench.place(shape)
    ctx.say(f"[mock] stereo bench: fixture placement {len(done) + 1} of the plan (stands in "
            "for the human)")


def cmd_calib_pose(ctx):
    a = ctx.args
    fxs = _fixture_arg(a.fixture)
    if ctx.mock and fxs is not None:
        _mock_place_fixture(ctx, fxs, a.set)
    r = api.capture_calib_pose(ctx.preset(), a.set, steps=a.steps, periods=_periods(a.periods),
                               fixture=str(Path(a.fixture).resolve()) if fxs else None)
    parity = _parity_words(r["diamond_parity"], r["parity_source"])
    if fxs is None:
        ctx.say(f"Calibration pose {r['id']} (set {r['set']}): {r['corners_found']} ChArUco "
                f"corners, {r['valid_pct']:.1f}% decoded, parity {parity}")
    else:
        ff = r["fixture"]["faces_found"]
        ctx.say(f"Fixture placement {r['id']} (set {r['set']}, fixture {r['fixture']['name']}): "
                f"{len(r['faces_seen'])} of {len(ff)} faces, {r['corners_found']} ChArUco "
                f"corners, {r['valid_pct']:.1f}% decoded, parity {parity}")
        ctx.table(["face", "marker ids", "corners", "projector corners", "usable"],
                  [[f["face"], f"{f['id_range'][0]}-{f['id_range'][1]}", f["corners"],
                    f.get("projector_corners", "-"), "yes" if f["corners"] >= 6 else "no"]
                   for f in ff])
    # how many of the corners found got a projector position: what the projector fit gets
    ctx.say(f"  projector corners {r['projector_corners']}/{r['corners_found']} "
            f"({r['projector_corner_pct']:.0f} %) at fringe periods "
            f"{_periods_words(r['periods_px'])} px ({_PERIOD_SOURCES.get(r['periods_source'], '?')})"
            + ("" if r["accepted"] else "  - NOT accepted"))
    if len(r.get("attempts") or []) > 1:
        ctx.table(["tried periods (px)", "corners", "projector corners", "%", ""],
                  [[_periods_words(t["periods_px"]), t["corners_found"], t["projector_corners"],
                    f"{t['projector_corner_pct']:.0f}", "stored" if t["stored"] else ""]
                   for t in r["attempts"]], indent="    ")
    if fxs is not None:
        ctx.say("  next: move the fixture (left / right / centre), turn it 20-45 deg about its "
                "own axis, tip it a little, change the distance; 3 placements, then "
                "calibrate-stereo")
    ctx.warn(r.get("warnings"))
    return (EXIT_OK if r["accepted"] else EXIT_NOT_OK), r


_PERIOD_SOURCES = {"argument": "--periods", "set": "this pose set's", "preset": "the default",
                   "auto": "coarser than the default: retried automatically"}


def _periods_words(periods) -> str:
    return ",".join(f"{float(x):g}" for x in periods or []) or "-"


def _parity_words(parity, source) -> str:
    """The diamond row parity with how it is known: measured (also 'no offset'), set, or
    not measured."""
    from engine.calibrate.stereo import parity_text
    return parity_text(parity, source)


def _print_calibration_poses(ctx, rows: list) -> None:
    """Which poses gave the projector fit its points (>= 6 corners with a projector
    position): the per-pose table of calibrate-stereo."""
    if not rows:
        return
    n = sum(1 for x in rows if x.get("contributed"))
    ctx.say(f"  projector points from {n} of {len(rows)} poses:")
    ctx.table(["pose", "corners", "projector corners", "%", "contributed"],
              [[x.get("id"), x.get("corners"), x.get("projector_corners"),
                _f(x.get("projector_corner_pct"), 0), "yes" if x.get("contributed") else "NO"]
               for x in rows], indent="    ")


def cmd_calibrate_stereo(ctx):
    a = ctx.args
    if a.mock_poses and not ctx.mock:
        raise ValueError("--mock-poses needs --mock (on the rig: calib-pose per board pose)")
    if a.mock_fixture and not ctx.mock:
        raise ValueError("--mock-fixture needs --mock (on the rig: calib-pose --fixture SPEC per "
                         "placement)")
    fxs = _fixture_arg(a.fixture)
    from engine.calibrate.stereo import CalibrationRejected
    try:
        r = api.calibrate_stereo(ctx.preset(), a.set, a.square, mock_poses=a.mock_poses or 0,
                                 steps=a.steps, periods=_periods(a.periods),
                                 projector_distortion=a.projector_distortion,
                                 fixture=str(Path(a.fixture).resolve()) if fxs else None,
                                 mock_fixture=a.mock_fixture or 0, rigid=not a.no_rigid)
    except CalibrationRejected as e:
        # fail loudly: not a calibration, nothing saved, never exit 0
        ctx.say("Stereo calibration REJECTED - NOTHING SAVED (the calibration in use, if any, "
                "is unchanged):")
        for why in e.reasons:
            ctx.say(f"  ! {_one_line(why)}")
        _print_calibration_poses(ctx, e.per_pose)
        if e.numbers:
            ctx.kv([[k, _f(v, 4)] for k, v in sorted(e.numbers.items())], indent="    ")
        if e.report_path is not None:
            ctx.say(f"  report: {e.report_path}")
        if not ctx.json:
            print(f"error: stereo calibration rejected: {_one_line('; '.join(e.reasons))}",
                  file=sys.stderr)
        return EXIT_NOT_OK, {"rejected": True, "saved": False, "calibration_id": None,
                             "reasons": e.reasons, "per_pose": e.per_pose,
                             "numbers": e.numbers, "report": e.report_path,
                             "error": {"type": "CalibrationRejected", "message": str(e)}}
    rms = r["rms"]
    if r.get("fixture"):
        ctx.say(f"Stereo calibration {r['calibration_id']}  ({r['n_views']} face views from "
                f"{r['n_placements']} fixture placements, set {r['set']})")
    else:
        ctx.say(f"Stereo calibration {r['calibration_id']}  ({r['n_poses_used']} of "
                f"{r['n_poses_captured']} poses, set {r['set']})")
    ctx.kv([["camera RMS", _u(rms.get("camera_px"), "px", 4) + "  (target < 0.3)"],
            ["projector RMS u / v", f"{_f(rms.get('projector_u_px'), 4)} / "
                                    f"{_f(rms.get('projector_v_px'), 4)} px  (target < 0.5)"],
            ["stereo RMS", _u(rms.get("stereo_px"), "px", 4)],
            ["baseline", _u(r["baseline_mm"], "mm", 2)], ["angle", _u(r["angle_deg"], "deg", 2)],
            ["f_v / f_u (diamond ~2)", _f(r["fv_over_fu"], 4)],
            ["diamond parity", r.get("diamond_parity_text") or r["diamond_parity"]],
            ["fringe periods", f"{_periods_words(r.get('periods_px'))} px"]])
    _print_calibration_poses(ctx, r.get("per_pose") or [])
    if r.get("fixture"):
        rg, fc = r["rigid"], r["face_check"]
        ctx.kv([["rigid-fixture refinement",
                 "used (face-to-face poses estimated, shared by all placements)" if rg["used"]
                 else "not used: the independent per-face views"],
                ["face-angle check", f"max |measured - spec| {_f(fc['max_abs_residual_deg'], 3)} "
                                     f"deg ({fc['source']}; warn > {fc['warn_deg']:g})"]])
    if r.get("vs_truth"):
        t = r["vs_truth"]
        ctx.say(f"  [mock] vs truth: R {t['R_deg']:.4f} deg, t {t['t_mm']:.3f} mm, cam f "
                f"{_f(t['cam_f_px'][0], 2)} px, proj f {_f(t['proj_f_px'][0], 2)} / "
                f"{_f(t['proj_f_px'][1], 2)} px")
    ctx.warn(r.get("warnings"))
    ok = rms.get("camera_ok") and rms.get("projector_ok")
    return (EXIT_OK if ok else EXIT_NOT_OK), r


def cmd_scan_metric(ctx):
    a = ctx.args
    ms = api.scan_metric(ctx.preset(), a.label, a.method, a.steps, _periods(a.periods),
                         a.max_consistency, a.zone)
    c = ms.meta.get("consistency") or {}
    ctx.say(f"Metric scan {ms.id}: {ms.meta['n_points']} points in mm ({ms.meta['method']}), "
            f"calibration {ms.calibration_id}")
    ctx.kv([["valid", f"{ms.meta['valid_pct']:.1f} %"],
            ["consistency median / p95", f"{_f(c.get('median_mm'), 4)} / "
                                         f"{_f(c.get('p95_mm'), 4)} mm"]])
    ctx.warn(ms.meta.get("warnings"))       # a fixed rig in doubt, a calibration that may not fit
    return EXIT_OK, {"metric_scan": ms, **_scan_says(ms.meta)}


def _scan_says(meta: dict) -> dict:
    """--json of a scan verb: what its entry says it rests on, at the top level too: its
    warnings (a fixed rig in doubt first, in the rig check's words; a calibration that may
    not fit) and its rig stamp (meta "rig")."""
    return {"warnings": list((meta or {}).get("warnings") or []), "rig": (meta or {}).get("rig")}


def cmd_triangulate(ctx):
    a = ctx.args
    ctx.preset()
    try:
        scan = _find_metric(a.id)
    except CLIError:
        scan = _find_scan(a.id)
    r = api.triangulate(scan, a.method)
    ctx.say(f"Triangulated {scan.id} -> metric scan {r['scan'].id}: {r['n_points']} points "
            f"(mm, {r['method']})")
    ctx.warn(r.get("warnings"))
    return EXIT_OK, r


def cmd_cyldev(ctx):
    a = ctx.args
    ctx.preset()
    ms = _find_metric(a.id)
    dev = api.cylinder_deviation(ms, a.radius, a.sigma, a.min_depth)
    ctx.say(f"Cylinder deviation of {ms.id}: diameter {dev.fit.diameter:.4f} mm, seen arc "
            f"{_f(dev.arc_deg, 1)} deg")
    ctx.kv([["noise sigma", _u(dev.noise_sigma_mm, "mm", 4)],
            ["threshold", _u(dev.threshold_mm, "mm", 4)],
            *[[k, str(v)] for k, v in dev.files.items() if v is not None]])
    ctx.table(["depth mm", "area mm2", "theta deg", "z mm", "snr"],
              [[_f(d.depth_mm, 4), _f(d.area_mm2, 2), _f(d.theta_deg, 1), _f(d.z_mm, 2),
                _f(d.snr, 1)] for d in dev.dents])
    if not dev.dents:
        ctx.say("  no dents above the threshold")
    ctx.warn(dev.warnings)
    return EXIT_OK, {"deviation": dev}


def cmd_accuracy(ctx):
    a = ctx.args
    known = {}
    if a.height is not None:
        known["height_mm"] = a.height
    if a.diameter is not None:
        known["diameter_mm"] = a.diameter
    scans = [_find_metric(k) for k in a.scans] if a.scans else None
    r = api.accuracy_check(a.kind, known, scans, a.repeats, ctx.preset(), a.tol, a.repeat_tol)
    ctx.say(f"Accuracy check ({a.kind}, {r['n_repeats']} scans): {r['verdict']}")
    ctx.kv([["mean error", _u(r["mean_error_mm"], "mm", 4)],
            ["max |error|", _u(r["max_abs_error_mm"], "mm", 4)],
            ["repeatability sigma", _u(r["repeatability_sigma_mm"], "mm", 4)],
            ["mean measured", _u(r["mean_measured_mm"], "mm", 4)]])
    ctx.warn(r.get("warnings"))             # a fixed rig in doubt, rays that miss
    return (EXIT_OK if r["verdict"] == "GO" else EXIT_NOT_OK), r


def _mock_place_block(ctx, spec: dict) -> None:
    """Mock only: the simulated human sets the check block down facing the camera (the
    simulated print is exactly its spec)."""
    from engine.capture.mock_stereo import check_block_placement, check_block_shapes
    bench = api.mock_bench()
    bench.clear()
    bench.use_stereo()
    for shape in check_block_shapes(spec, *check_block_placement()):
        bench.place(shape)
    ctx.say(f"[mock] stereo bench: placed the check block {spec['name']} (printed exactly to "
            "its spec)")


def cmd_accuracy_block(ctx):
    a = ctx.args
    from engine.measure import checkblock
    spec = checkblock.load(Path(a.block))                  # refused before the rig
    meas = checkblock.load_measured(Path(a.measured), spec) if a.measured else None
    scans = [_find_metric(k) for k in a.scans] if a.scans else None
    p = ctx.preset()
    if ctx.mock and scans is None:
        _mock_place_block(ctx, spec)
    r = api.accuracy_block(str(Path(a.block)), {"values": meas} if meas else None, scans,
                           a.repeats, p, a.tol_height, a.tol_diameter, a.flatness_max, a.label)
    ctx.say(f"Check block {r['block']} ({r['n_scans']} scan(s), calibration "
            f"{r.get('calibration_id')}): {r['result']}")
    ctx.say(f"  {checkblock.verdict_text(r)}")
    ctx.table(["quantity", "scanned mm", "reference mm", "from", "error mm", "tol mm", "verdict"],
              [[x["name"], _f(x.get("scanned_mm"), 4),
                _f(x.get("reference_mm"), 4) if x.get("reference_mm") is not None else "-",
                x.get("reference_source", "-"), _f(x.get("error_mm"), 4),
                _f(x.get("tol_mm", x.get("limit_mm")), 3), x["verdict"]] for x in r["rows"]])
    ctx.warn([f"{q['reason']}: {q['message']}" for q in r.get("pull_reasons") or []])
    ctx.warn(r.get("warnings"))             # a fixed rig in doubt, rays that miss
    return D.exit_code(r), r


def cmd_precision(ctx):
    a = ctx.args
    scan = _find_metric(a.scan) if a.scan else None
    r = api.predict_precision(ctx.preset(), a.z, a.A, a.B, a.period, a.steps, scan=scan)
    ctx.say(f"Predicted sigma_z: {r['sigma_z_mm']:.4f} mm  (sigma_phi {r['sigma_phi_rad']:.5f} "
            f"rad, {r['mm_per_projector_px']:.3f} mm per projector px)")
    ctx.warn(r.get("notes"))
    return EXIT_OK, r


def cmd_metric_mesh(ctx):
    a = ctx.args
    ctx.preset()
    ms = _find_metric(a.id)
    p = api.export_metric_mesh(ms, a.fmt, a.stride, a.max_edge, a.foreground)
    ctx.say(f"Metric mesh of {ms.id} ({a.fmt.upper()}, mm, open surface): {p}")
    return EXIT_OK, {"file": p}


def cmd_metric_list(ctx):
    ctx.preset()
    rows = api.list_library("metric")
    ctx.table(["id", "label", "points", "calibration"],
              [[m.id, m.label, m.meta.get("n_points"), m.calibration_id] for m in rows])
    if not rows:
        ctx.say("  no metric scans yet (scan-metric)")
    return EXIT_OK, {"metric": rows}


# ------------------------------------------------ dent detection (shiny cylinders)

def _find_dentset(key: str):
    """A dent set by id / label / 'latest', or a folder (engine entry or legacy dataset.json)."""
    p = Path(key)
    if p.is_dir():
        return str(p.resolve())
    return _pick(api.list_library("dentsets"), key, "dent set", "dentsets")


def _metric_arg(keys):
    """Metric scans by id / label / 'latest' or folder paths (None = the dent set's own)."""
    if not keys:
        return None
    return [str(Path(k).resolve()) if Path(k).is_dir() else _find_metric(k) for k in keys]


def _other_preset(ctx, name: Optional[str]):
    """A second preset for this run only (read + validated; the active preset is unchanged)."""
    if not name:
        return None
    p = config.load_preset(name, store=ctx.session.store, mock=ctx.mock)
    if ctx.args.average is not None:
        p.camera.average = ctx.args.average
    return p


def cmd_dent_scan(ctx):
    a = ctx.args
    lp = ctx.preset()
    sp = _other_preset(ctx, a.short_preset)
    mp = _other_preset(ctx, a.metric_preset)
    ds = api.capture_dent_set(a.zone, a.label, lp, sp, a.short_exposure, not a.no_lines,
                              not a.no_metric, mp, a.steps, _periods(a.periods), a.pattern_set)
    passes = ds.meta.get("passes", [])
    ctx.say(f"Dent set {ds.id} (label {ds.label}, geometry {ds.geometry_id}, zone "
            f"{ds.zone or 'whole field'})")
    no_metric = "none (--no-metric)" if a.no_metric else "NONE: the metric scan failed (see below)"
    ctx.kv([*[[f"{q['name']} pass", f"{q['preset']} @ {q['exposure_us']} us, {len(q['patterns'])} "
                                     f"patterns, {q['capture_s']} s"] for q in passes],
            ["crop x0,y0,x1,y1", ",".join(str(v) for v in ds.meta.get("crop_x0y0x1y1", []))],
            ["metric scan", ", ".join(ds.metric_scan_ids) or no_metric],
            ["stored", str(ds.path)]])
    ctx.warn(ds.warnings)                   # the metric scan's first: a fixed rig in doubt
    return EXIT_OK, {"dentset": ds, **_scan_says(ds.meta)}


def cmd_dent_null(ctx):
    a = ctx.args
    sets = [_find_dentset(k) for k in a.dentsets]
    r = api.calibrate_dent_null(sets, a.diameter, a.label, _metric_arg(a.metric), a.fixture,
                                a.subset)
    ctx.say(f"Dent null calibration {r['id']}  (diameter {r['cell_diameter_mm']:g} mm, fixture "
            f"{r.get('fixture') or '-'}, patterns {r['subset']})")
    ctx.table(["finder", "good-part max (T)"],
              [[k, _f(v, 3) if v is not None else "EXCLUDED (no response)"]
               for k, v in r["T"].items()] + [["fused (T_F)", _f(r["T_F"], 3)]])
    ctx.warn(r.get("warnings"))
    return EXIT_OK, r


def cmd_dent_detect(ctx):
    a = ctx.args
    ds = _find_dentset(a.dentset)
    null = None if a.null in (None, "latest", "auto") else a.null
    rep = api.detect_dents(ds, _metric_arg(a.metric), a.diameter, null, a.label, a.subset,
                           a.fixture)
    thr = rep.thresholds.get("thr")
    ctx.say(f"Dents on {rep.dentset_id}: {rep.verdict} ({rep.n_detections} detection(s) above "
            f"{_f(thr, 2)}; max score {_f(rep.max_score, 2)} x the good-part maximum; null "
            f"{rep.null_calibration_id}, diameter {rep.cell_diameter_mm:g} mm)")
    rows = []
    for i, c in enumerate(rep.candidates[:25], 1):
        dep = "-" if c["depth_mm"] is None else \
            _u(c["depth_mm"], "mm", 3) + ("" if c["depth_reliable"] else " (unreliable)") \
            + ("" if c.get("depth_significant", True) else " (in noise)")
        rows.append([i, c["level"], c["kind"], f"{c['x']},{c['y']}", _f(c["theta_deg"], 1),
                     _f(c.get("s_from_top_mm"), 1), _f(c["score"], 2), dep,
                     _u(c["size_mm"], "mm", 2),
                     " ".join(f"{k[:5]}={v:.2f}" for k, v in c["evidence"].items())])
    ctx.table(["#", "level", "kind", "x,y px", "theta deg", "s from top mm", "score", "depth",
               "size", "evidence (x good max)"], rows)
    if not rep.candidates:
        ctx.say(f"  nothing above the watch level ({_f(rep.thresholds.get('watch'), 2)} x the "
                "good-part maximum)")
    ctx.kv([[k, str(v)] for k, v in rep.files.items()])
    ctx.warn(rep.warnings)
    return (EXIT_OK if rep.verdict == "CLEAR" else EXIT_NOT_OK), {"report": rep}


def cmd_dentsets(ctx):
    rows = api.list_library("dentsets", ctx.args.geometry)
    ctx.table(["id", "label", "zone", "metric scans", "created"],
              [[d.id, d.label, d.zone or "-", ", ".join(d.metric_scan_ids) or "-", d.created]
               for d in rows])
    if not rows:
        ctx.say("  no dent sets yet (dent-scan)")
    return EXIT_OK, {"dentsets": rows}


# ------------------------------------------------ end faces (a cylinder's end face toward the camera)

def _find_face(key: Optional[str]):
    """A face target by id / label / 'latest' or its folder; None = the engine's default."""
    if not key:
        return None
    if Path(key).is_dir():
        return str(Path(key).resolve())
    return _pick(api.list_library("faces"), key, "face target", "faces")


def _find_facescan(key: str):
    if Path(key).is_dir():
        return str(Path(key).resolve())
    return _pick(api.list_library("facescans"), key, "face scan", "faces")


def _find_golden(key: Optional[str]):
    if not key or key.lower() in ("auto", "latest-matching"):
        return None
    if Path(key).is_dir():
        return str(Path(key).resolve())
    return _pick(api.list_library("face_golden"), key, "golden face", "faces")


def _placement(key: str):
    """A placement for face-golden / face-check: a face-profile .npz or a face scan."""
    p = Path(key)
    if p.suffix.lower() == ".npz" and p.is_file():
        return str(p.resolve())
    return _find_facescan(key)


def _roi(text: Optional[str]):
    return None if not text else _numbers(text, 4, "--roi x0,y0,x1,y1 (camera px)", int)


def _size_line(sc: Optional[dict]) -> str:
    """The known-size check of a result, one line."""
    if not sc:
        return "-"
    return {"ok": "ok", "mismatch": "SIZE MISMATCH", "not_checked": "not checked"}.get(
        sc.get("status"), str(sc.get("status"))) + f": {sc.get('message')}"


def _pull(reasons: list) -> dict:
    """{"disposition": "PULL", "pull_reasons"} of reasons the engine gave ({reason, message})."""
    return {"disposition": "PULL",
            "pull_reasons": [{"reason": str(r.get("reason")),
                              "message": " ".join(str(r.get("message") or "").split())}
                             for r in reasons]}


def _print_face(ctx, t) -> None:
    a, e = t.affine, t.camera_ellipse
    ls = t.meta.get("locate_scan") or {}
    ctx.say(f"Face target {t.id}  (face {t.face_diameter_mm:g} mm, from metric scan {t.source_scan})")
    ctx.kv([["centre", "(" + ", ".join(f"{v:.2f}" for v in t.centre_mm) + ") mm, camera frame"],
            ["tilt", _u(t.tilt_deg, "deg", 1)],
            ["rim fit", f"{_u(t.rim_rms_mm, 'mm', 3)} rms over {t.n_rim_points} rim points "
                        f"({t.meta.get('rim_source') or 'depth'}"
                        + (f"; depth silhouette {_u(t.meta['silhouette_diameter_mm'], 'mm', 2)} "
                           f"across, {t.meta.get('silhouette_source')}"
                           if t.meta.get("silhouette_diameter_mm") else "") + ")"],
            ["size check", _size_line(t.meta.get("size_check"))],
            ["centre check", (t.meta.get("centre_check") or {}).get("message") or "-"],
            *([["locate scan exposure", f"{_u(ls['exposure_us'] / 1000.0, 'ms', 1)} "
                                        f"({ls.get('exposure_source')})"]]
              if ls.get("exposure_us") else []),
            ["camera ellipse", f"({e['cx']:.1f}, {e['cy']:.1f}) px, {e['a']:.1f} x {e['b']:.1f} px"],
            ["camera -> DMD affine", f"residual col {_f(a['rms_col_px'], 3)} / row "
                                     f"{_f(a['rms_row_px'], 3)} DMD px over {a['n_px']} face px"],
            ["DMD centre", f"({t.dmd_centre[0]:.1f}, {t.dmd_centre[1]:.1f}) (col, row)"],
            ["stored", str(t.path)]])
    for n in ls.get("notes") or []:
        ctx.say(f"  {n}")
    ctx.warn(t.warnings)


def _exposure_range(text: Optional[str]):
    """--exposure-range LO,HI in ms -> [lo, hi] us (None = the engine's 16.7-100 ms)."""
    if not text:
        return None
    lo, hi = _numbers(text, 2, "--exposure-range lo,hi (ms)")
    return [lo * 1000.0, hi * 1000.0]


def cmd_face_locate(ctx):
    a = ctx.args
    params = _kw(near_band_mm=a.near_band, far_gap_mm=a.far_gap, near_gap_mm=a.near_gap)
    if a.metric:
        if a.auto_exposure is not None or a.exposure_us is not None:
            raise CLIError("--auto-exposure / --exposure-us belong to the locate scan "
                           "(--zone): a stored metric scan was made at its own exposure")
        ms = _find_metric(a.metric)
        if ms.field.corr_x is None or ms.field.corr_y is None:
            raise CLIError(f"metric scan {ms.id} has only one decoded axis (x-only); face-locate "
                           "needs x and y - rescan with `scan-fast --axis xy` (or scan-metric)")
        t = api.locate_face(ms, a.diameter, a.label, _roi(a.roi), params or None,
                            _flood_arg(a.flood), size_tol_pct=a.size_tol,
                            silhouette_diameter_mm=a.outer_diameter)
    elif a.zone:
        # the locate scan keeps its flood pair (the size check needs it) and is made at the
        # exposure the face needs: scanned again when the located face is dark or clipped
        t = api.locate_face_live(a.zone, ctx.preset(), a.diameter, a.label, a.steps, None,
                                 _roi(a.roi), params or None, size_tol_pct=a.size_tol,
                                 auto_exposure=a.auto_exposure, exposure_us=a.exposure_us,
                                 exposure_range_us=_exposure_range(a.exposure_range),
                                 silhouette_diameter_mm=a.outer_diameter)
        ctx.say(f"Locate scan {t.source_scan} (zone {a.zone}, ray-plane, flood pair kept)")
    else:
        raise CLIError("face-locate needs a metric scan (id / label / latest) or --zone to scan one")
    _print_face(ctx, t)
    if t.meta.get("disposition") == "PULL":          # the face reads the wrong size
        disp = _pull(t.meta.get("pull_reasons") or [])
        ctx.say(f"  disposition: {_one_line(D.summary(disp))}")
        return D.exit_code(disp), {"face": t, **disp}
    return EXIT_OK, {"face": t}


def cmd_face_aim(ctx):
    a = ctx.args
    off = _numbers(a.offset, 2, "--offset cols,rows (DMD px)") if a.offset else None
    face = _find_face(a.face)
    if a.check:
        r = api.check_ring(face, ctx.preset(), a.hold or 0.0, a.label or "ring_check", a.tol, off)
    else:
        r = api.aim_ring(face, ctx.preset(), a.hold or 0.0, a.label or "aim", off)
    ctx.say(f"{'Ring check' if a.check else 'Aim ring'} for face {r.face_id}"
            + (f": {r.verdict}" if a.check else "") + f"  (DMD centre {r.dmd_centre[0]:.1f}, "
            f"{r.dmd_centre[1]:.1f}; held {r.held_s:g} s)")
    if a.check:
        ctx.kv([["face lit by its flood", _u(r.lit_pct, "%", 1) + " (>= 90)"],
                ["flood outside the face", _u(r.spill_pct, "%", 1) + " (<= 10)"],
                ["ring centre offset", ("-" if r.centre_offset_px is None else
                                        ", ".join(f"{v:+.2f}" for v in r.centre_offset_px) + " px")
                 + ("" if r.centre_offset_mm is None else f" ({r.centre_offset_mm:.2f} mm)")]])
    ctx.kv([["photo", r.photo], ["overlay", r.overlay]])
    ctx.warn(r.warnings)
    ok = (r.verdict == "ON_FACE") if a.check else not r.hold_cancelled
    return (EXIT_OK if ok else EXIT_NOT_OK), {"ring": r}


def _need_upright_projection(ctx, what: str) -> None:
    """auto intensity scales the lit level, which an inverted preset does not have:
    one clear line naming a preset that works, before anything reaches the rig."""
    p = ctx.preset()
    if not p.projection.invert:
        return
    names = list(api.list_presets())
    alt = [n for n in (f"{p.name}_bright",) if n in names] or         [n for n in names if n.endswith("_bright")]
    raise CLIError(f"{what} needs a preset with projection.invert false; preset {p.name} is "
                   "inverted (its duty scales the dark level)"
                   + (f" - use --preset {alt[0]}" if alt else
                      " - use a *_bright preset or one with projection.invert false"))


def cmd_face_intensity(ctx):
    a = ctx.args
    _need_upright_projection(ctx, "face-intensity")
    r = api.auto_intensity(_find_face(a.face), ctx.preset(), a.target, a.tol)
    ctx.say(f"Auto intensity: projection.duty {r['duty']:g} (preset {r['preset_duty']:g}) -> face "
            f"p99 {r['p99_dn']:.0f} DN, target {r['target_dn']:g} DN"
            + ("" if r["verified"] else "  (NOT verified by a shot)"))
    ctx.table(["duty", "p99 DN", "p50 DN", "saturated %"],
              [[_f(x["duty"], 4), _f(x["p99_dn"], 1), _f(x["p50_dn"], 1), _f(x["saturated_pct"], 2)]
               for x in r["shots"]])
    ctx.say("  use it: face-scan --duty " + f"{r['duty']:g}" + " (the preset file is not edited; "
            "LED currents are never touched)")
    ctx.warn(r.get("warnings"))
    return (EXIT_OK if r["verified"] else EXIT_NOT_OK), r


def cmd_face_scan(ctx):
    a = ctx.args
    p = ctx.preset()
    face = _find_face(a.face)
    duty = a.duty
    if a.auto_intensity:
        _need_upright_projection(ctx, "face-scan --auto-intensity")
        r = api.auto_intensity(face, p)
        duty = r["duty"]
        ctx.say(f"auto intensity: duty {duty:g} (face p99 {r['p99_dn']:.0f} DN)")
        ctx.warn(r.get("warnings"))
    from engine.measure.endface.exposure import ExposureNotFound
    try:
        fs = api.targeted_face_scan(face, p, a.label, a.repeats, a.margin, not a.no_polar,
                                    duty, a.steps, _periods(a.periods), a.polar_steps,
                                    mode=a.mode, code_bits=a.code_bits,
                                    auto_exposure=a.auto_exposure, exposure_us=a.exposure_us,
                                    exposure_target_dn=a.exposure_target,
                                    exposure_range_us=_exposure_range(a.exposure_range))
    except ExposureNotFound as e:
        # no exposure fits this seating: the face was not scanned - PULL, with every shot
        rep_ = e.report or {}
        disp = _pull([{"reason": "exposure", "message": rep_.get("message") or str(e)}])
        ctx.say(f"Face scan: NOT scanned - {_one_line(rep_.get('message') or e)}",
                f"  disposition: {_one_line(D.summary(disp))}")
        ctx.table(["exposure ms", "p95 DN", "clipped %", "ok"],
                  [[_f(x["exposure_us"] / 1000.0, 1), _f(x["p95_dn"], 0), _f(x["clipped_pct"], 2),
                    _f(bool(x["ok"]))] for x in rep_.get("shots") or []])
        return D.exit_code(disp), {"facescan": None, "auto_exposure": rep_,
                                   "error": {"type": type(e).__name__, "message": str(e)}, **disp}
    ae = fs.meta.get("auto_exposure") or {}
    ctx.say(f"Face scan {fs.id} (face {fs.face_id}, {len(fs.metric_scan_ids)} repeat(s), polar "
            f"{'yes' if fs.polar else 'no'}, duty {fs.meta.get('duty'):g}, exposure "
            f"{fs.meta.get('exposure_us', 0) / 1000.0:.1f} ms ({fs.meta.get('exposure_source')}"
            + (f": face p95 {ae.get('p95_dn'):.0f} DN, {ae.get('clipped_pct'):.2f} % clipped"
               if ae.get("ok") else "")
            + f"), depth set {a.mode}: {fs.meta.get('depth_frames')} frames)")
    ctx.table(["repeat", "metric scan", "face decoded", "capture"],
              [[i + 1, r["metric_scan"], _u(r["face_decoded_pct"], "%", 1), _u(r["capture_s"], "s", 1)]
               for i, r in enumerate(fs.repeats)])
    if ae.get("shots"):
        ctx.table(["exposure ms", "p95 DN", "clipped %", "ok"],
                  [[_f(x["exposure_us"] / 1000.0, 1), _f(x["p95_dn"], 0), _f(x["clipped_pct"], 2),
                    _f(bool(x["ok"]))] for x in ae["shots"]])
    ctx.kv([["stored", str(fs.path)]])
    ctx.warn(fs.warnings)
    return EXIT_OK, {"facescan": fs, **_scan_says(fs.meta)}


def cmd_face_profile(ctx):
    a = ctx.args
    face = _find_face(a.face)
    if len(a.scans) == 1:
        try:
            scans = _find_facescan(a.scans[0])
        except CLIError:
            scans = [_find_metric(a.scans[0])]
    else:
        scans = [_find_metric(k) for k in a.scans]
    pr = api.face_profile(scans, face, a.label, a.grid, a.r_flat, not a.no_stl,
                          size_tol_pct=a.size_tol)
    st = pr.stats
    ro = st.get("rim_outline") or {}
    ctx.say(f"Face profile {pr.label}: {st['n_repeats']} repeat(s), face {pr.face_id} "
            f"({pr.face_diameter_mm:g} mm), grid {pr.grid_mm:g} mm")
    ctx.kv([["flatness (r < %g mm)" % st["r_flat_mm"], f"{_u(st['flatness_rms_mm'], 'mm', 4)} rms, "
                                                        f"{_u(st['flatness_p2_p98_mm'], 'mm', 3)} p2-p98"],
            ["rim outline", (f"{_u(ro.get('diameter_mm'), 'mm', 3)} across, roundness "
                             f"{_u(ro.get('rms_mm'), 'mm', 3)} rms / {_u(ro.get('max_abs_mm'), 'mm', 3)} "
                             f"max ({_f(ro.get('valid_pct'), 0)} % of the rim)")
             if ro.get("measured") else f"not measured ({ro.get('why') or 'no flood pair'})"],
            ["size check", _size_line(st.get("size_check"))],
            ["centre", (st.get("centre") or {}).get("message") or "the located circle's"],
            ["repeatability", f"{_u(st['repeatability_median_mm'], 'mm', 4)} median, "
                              f"{_u(st['repeatability_p90_mm'], 'mm', 4)} p90 (per point, across repeats)"],
            ["tilt", _u(st["tilt_deg"], "deg", 1)],
            ["coverage", f"{_u(st['grid_coverage_pct'], '%', 1)} of the face grid, to r "
                         f"{_u(st['covered_radius_mm'], 'mm', 2)}"],
            *[[k, str(v)] for k, v in pr.files.items()]])
    ctx.table(["r mm", "median h mm", "std mm", "points"],
              [[_f(r["r_mm"], 2), _f(r["h_median_mm"], 4), _f(r["h_std_mm"], 4), r["n"]]
               for r in pr.radial])
    ctx.warn(pr.warnings)
    return EXIT_OK, {"profile": pr, "warnings": list(pr.warnings or []), "rig": st.get("rig")}


def cmd_face_golden(ctx):
    a = ctx.args
    places = [_placement(k) for k in a.placements]
    synthetic = False if a.no_synthetic else ({"positions": a.positions} if a.positions else True)
    # every radius derives from the face diameter unless given here; --rim-r is ONE rim
    # radius for every detector
    g = api.face_golden(places, a.label,
                        **_kw(k_sigma=a.k, floor_mm=a.floor, smooth_cells=a.smooth_cells,
                              rim_factor=a.rim_factor, rim_min_mm=a.rim_min,
                              height_smooth_mm=a.height_smooth, min_area_mm2=a.min_area,
                              z_thresh=a.z, min_depth_mm=a.min_depth),
                        synthetic=synthetic, detector=a.detector or "mf_fir", fir=not a.no_fir,
                        radii=_kw(inspect_r_mm=a.inspect_r, rim_r_mm=a.rim_r,
                                  plane_r_mm=a.plane_r) or None,
                        rim=_kw(tol_mm=a.rim_tol, min_arc_mm=a.rim_arc) or None,
                        size_tol_pct=a.size_tol)
    sp = g.meta.get("placement_spread") or {}
    prm = g.params
    det = prm.get("detector", "band")
    band, mf, fir = prm.get("band", prm), prm.get("mf") or {}, prm.get("fir") or {}
    rr, ro = prm.get("radii") or {}, g.meta.get("rim_outline") or {}
    ctx.say(f"Golden face {g.id}: {g.n_placements} placements, face {g.face_diameter_mm:g} mm, "
            f"calibration {g.calibration_id}; default detector {det}")
    fit = fir.get("fit") or {}
    ctx.kv([["placement spread", f"{_u(sp.get('sigma_rms_mm'), 'mm', 4)} sigma rms, pairs "
                                 f"{_u(sp.get('pair_rms_median_mm'), 'mm', 4)} median rms"],
            ["radii", f"inspect r <= {_u(rr.get('inspect_r_mm'), 'mm', 3)}, rim r > "
                      f"{_u(rr.get('rim_r_mm'), 'mm', 3)}, plane r <= "
                      f"{_u(rr.get('plane_r_mm'), 'mm', 3)} ("
                      + (f"given: {', '.join(rr['given'])}; the rest " if rr.get("given") else "")
                      + "from the face diameter)"],
            ["rim outline", (f"learned from {ro.get('n_placements')} placements: "
                             f"{_u(ro.get('diameter_mm'), 'mm', 3)} across, good parts max "
                             f"{_f(max(ro.get('max_abs_mm') or [0]), 3)} mm; a notch / bulge > "
                             f"{_u(ro.get('tol_mm'), 'mm', 2)} over {_u(ro.get('min_arc_mm'), 'mm', 1)} "
                             "of rim triggers") if ro else "not in this golden (no flood outline)"],
            ["MF-CFAR", f"trigger z >= {_f(mf.get('z_thresh'), 2)}, depth >= "
                        f"{_u(mf.get('min_depth_mm'), 'mm', 3)}, r <= {_u(mf.get('inspect_r_mm'), 'mm', 2)}, "
                        f"dent widths {mf.get('scales_mm')} mm"],
            ["reflectance FIR", ("fitted on " + _f(fit.get("n_px")) + " face px, explains "
                                 + _u(fit.get("explained_pct"), "%", 1) + " of the high-passed height")
             if fir.get("used") else "not used (no flood pair / --no-fir)"],
            ["band", f"max({band['k_sigma']:g} sigma, {band['floor_mm']:g} mm); rim r > "
                     f"{band['rim_r_mm']:g} mm x{band['rim_factor']:g} (min {band['rim_min_mm']:g} mm)"],
            ["stored", str(g.path)]])
    if g.leave_one_out:
        dets = [d for d in ("mf_fir", "mf", "band") if d in g.leave_one_out[0]]
        ctx.say("  leave-one-out (each GOOD placement vs the golden of the others): false triggers")
        ctx.table(["held out", "dev rms mm", "max z"] + [f"{d} face/rim" for d in dets],
                  [[r["placement"], _f(r["dev_rms_mm"], 4), _f(r.get("max_z"), 2)]
                   + [f"{r[d]['n_face']}/{r[d]['n_rim']}" for d in dets] for r in g.leave_one_out])
    sens = g.sensitivity or {}
    if sens.get("total"):
        ctx.say(f"  synthetic dents found ({sens.get('detector', det)}, leave-one-out, "
                f"{sens.get('positions_per_size_per_fold', '?')} random centres per size per "
                f"placement): {sens['detected']} of {sens['total']}")
        mfb = bool(sens.get("z_thresh"))
        ctx.table(["depth mm", "width mm", "found"] + (["at z 5.0", "at z 4.5"] if mfb else []),
                  [[_f(b["depth_mm"], 3), _f(b["width_mm"], 1), f"{b['detected']}/{b['of']} "
                    f"({b.get('pct', 0):.0f}%)"]
                   + ([_u((b.get("pct_at_z") or {}).get("5"), "%", 0),
                       _u((b.get("pct_at_z") or {}).get("4.5"), "%", 0)] if mfb else [])
                   for b in sens["by_size"]])
    ctx.warn(g.warnings)
    return EXIT_OK, {"golden": g}


def cmd_face_check(ctx):
    a = ctx.args
    r = api.check_face(_placement(a.placement), _find_golden(a.golden), a.label, a.min_area,
                       a.detector, a.z, a.min_depth,
                       **_kw(rim_tol_mm=a.rim_tol, rim_min_arc_mm=a.rim_arc,
                             size_tol_pct=a.size_tol))
    ctx.say(f"Face check of {r.placement['label']}: {r.verdict} ({r.n_triggers} trigger(s); "
            f"golden {r.golden_id}, detector {r.detector})")
    disp = _face_disposition(r)
    ctx.say(f"  disposition: {_one_line(D.summary(disp))}")
    st = r.stats
    ro, rr = st.get("rim_outline") or {}, st.get("radii") or {}
    if r.detector in ("mf", "mf_fir"):
        # depth = depth_centre_mm (the 3 x 3 centre average: THE depth, thresholds apply to
        # it); deepest pt = the single deepest grid point (over-reads); rim_outline rows: how
        # far the outline moved (+ in: a notch) and over how much rim
        ctx.table(["#", "zone", "r mm", "theta deg", "depth mm", "deepest pt mm",
                   "width@half mm", "z", "area mm^2"],
                  [[i, t["zone"], _f(t["r_mm"], 2), _f(t["theta_deg"], 1), _f(t["depth_mm"], 3),
                    _f(t.get("depth_min_mm"), 3),
                    _f(t.get("width_half_depth_mm", t.get("arc_mm")), 2), _f(t.get("z"), 2),
                    _f(t.get("area_mm2"), 2)] for i, t in enumerate(r.triggers, 1)])
        nf = st.get("noise_floor_mm") or {}
        ctx.kv([["dent z", f"max {_f(st.get('max_z'), 2)} (trigger >= {_f(st.get('z_thresh'), 2)})"],
                ["noise floor (1 sigma depth)", ", ".join(f"{_u(v, 'mm', 3)} at {k} mm"
                                                          for k, v in nf.items()) or "-"],
                ["reflectance FIR", "applied" if st.get("fir_applied") else "not applied"]])
    else:
        ctx.table(["#", "zone", "r mm", "theta deg", "dev mm", "area mm^2", "band mm"],
                  [[i, t["zone"], _f(t["r_mm"], 2), _f(t["theta_deg"], 1), _f(t.get("dev_mm"), 3),
                    _f(t.get("area_mm2"), 2), _f(t.get("band_mm"), 3)]
                   for i, t in enumerate(r.triggers, 1)])
    ctx.kv([["deviation", f"{_u(st['dev_rms_mm'], 'mm', 4)} rms, max |dev| "
                          f"{_u(st['max_abs_dev_mm'], 'mm', 3)} (r < {_f(st.get('dev_r_mm'), 2)} mm)"],
            ["compared", _u(st["compared_pct"], "% of the inspected face", 1)],
            ["radii", f"inspect r <= {_u(rr.get('inspect_r_mm'), 'mm', 3)}, rim r > "
                      f"{_u(rr.get('rim_r_mm'), 'mm', 3)} (face "
                      f"{_u(2 * (rr.get('face_radius_mm') or 0), 'mm', 2)})"],
            ["rim outline", (f"{ro.get('status')}: roundness {_u(ro.get('rms_mm'), 'mm', 3)} rms / "
                             f"{_u(ro.get('max_abs_mm'), 'mm', 3)} max, tolerance "
                             f"{_u(ro.get('tol_mm'), 'mm', 2)} over {_u(ro.get('min_arc_mm'), 'mm', 1)} "
                             "of rim" + ("" if ro.get("golden") else " (no golden outline)")
                             + (f"; NOT judged over {len(ro['unjudged'])} stretch(es) of rim"
                                if ro.get("unjudged") else ""))
             if ro.get("measured") else
             f"{ro.get('status') or 'not measured'} ({ro.get('why') or '-'})"],
            ["size check", _size_line(st.get("size_check"))],
            ["centre check", (st.get("centre_check") or {}).get("message") or "-"],
            *[[k, str(v)] for k, v in r.files.items() if v is not None]])
    ctx.warn(r.warnings)
    return D.exit_code(disp), {"check": r, **disp}


def _face_disposition(r) -> dict:
    """A face check's disposition: the engine's own (stats: CONTINUE / REJECT / PULL with its
    pull reasons - size_mismatch, guard_band, incomplete); a result without one is settled
    from its verdict. Never CONTINUE unless the verdict is PASS."""
    st = r.stats or {}
    d = st.get("disposition")
    if d == "PULL" or (d in D.DISPOSITIONS and r.verdict != "PASS" and d == "CONTINUE"):
        return _pull(st.get("pull_reasons") or [{"reason": "incomplete",
                                                 "message": f"verdict {r.verdict}"}])
    if d == "REJECT" and r.verdict == "FAIL":
        return {"disposition": "REJECT", "pull_reasons": []}
    if d == "CONTINUE" and r.verdict == "PASS":
        return {"disposition": "CONTINUE", "pull_reasons": []}
    return _check_disposition(r.verdict, "less than the needed share of the face was measured "
                              f"({_f(st.get('compared_pct'), 1)} % compared)")


def _check_disposition(verdict: str, incomplete_why: str) -> dict:
    """face-check / station-check: PASS -> CONTINUE, FAIL -> REJECT, INCOMPLETE (or anything
    else) -> PULL with its reason. -> {"disposition", "pull_reasons"}"""
    res = {"result": verdict if verdict in ("PASS", "FAIL") else "INVALID",
           "invalid_reason": None, "pull_reasons": []}
    if res["result"] == "INVALID":
        D.add(res, "incomplete" if verdict == "INCOMPLETE" else "invalid",
              f"{verdict}: {incomplete_why}")
    D.settle(res)
    return {"disposition": res["disposition"], "pull_reasons": res["pull_reasons"]}


def cmd_faces(ctx):
    g = ctx.args.geometry
    faces, scans, golds = (api.list_library(k, g) for k in ("faces", "facescans", "face_golden"))
    ctx.say("Face targets:")
    ctx.table(["id", "face mm", "tilt deg", "rim rms mm", "affine col/row px", "from scan"],
              [[t.id, _f(t.face_diameter_mm, 2), _f(t.tilt_deg, 1), _f(t.rim_rms_mm, 3),
                f"{_f(t.affine['rms_col_px'], 3)}/{_f(t.affine['rms_row_px'], 3)}", t.source_scan]
               for t in faces])
    ctx.say("Face scans:")
    ctx.table(["id", "face", "repeats", "polar"],
              [[f.id, f.face_id, len(f.metric_scan_ids), _f(f.polar)] for f in scans])
    ctx.say("Golden faces:")
    ctx.table(["id", "face mm", "placements", "calibration"],
              [[x.id, _f(x.face_diameter_mm, 2), x.n_placements, x.calibration_id] for x in golds])
    if not (faces or scans or golds):
        ctx.say("  nothing yet (face-locate, face-scan, face-golden)")
    return EXIT_OK, {"faces": faces, "facescans": scans, "face_golden": golds}


# ------------------------------------------------ HDMI fast zone scan + station (parts + tube)

def cmd_scan_fast(ctx):
    a = ctx.args
    zone = a.zone
    if a.mask:
        if zone:
            raise CLIError("give --zone or --mask, not both")
        if not Path(a.mask).is_file():
            raise CLIError(f"--mask {a.mask!r}: no such .npy file")
        zone = str(Path(a.mask).resolve())
    faces = [_find_face(f) for f in (a.face or [])] or None
    avg = a.average if a.average is not None else 2
    zones = None
    if a.zones:
        if zone:
            raise CLIError("give --zones (a painted zone set) or --zone / --mask, not both")
        zones = _zones_text(a.zones)
    # a zone set: both DMD axes unless the operator asks for columns only, so every point is
    # assigned to its zone exactly
    axis = a.axis or ("xy" if zones else "x")
    ms = api.scan_fast(zone, ctx.preset(), a.label, a.steps, _periods(a.periods), avg, a.hold,
                       axis, faces, a.margin, **({"zones": zones} if zones else {}))
    m = ms.meta
    ctx.say(f"Fast scan {ms.id}: {m['n_points']} points in mm ({m['method']}), zone {m['zone']}")
    zr = m.get("zones") if zones else None
    if zr:
        ctx.table(["zone", "name", "points", "DMD px"],
                  [[zid, zr["names"].get(zid) or "-", "?" if n is None else n, zr["dmd_px"][zid]]
                   for zid, n in zr["counts"].items()]
                  + [["(outside)", "-", zr["outside"], "-"]])
    ctx.kv([["patterns", f"{m['n_patterns']} ({m['steps']} steps x periods "
                         f"{', '.join(f'{p:g}' for p in m['periods_px'])} DMD px + floods)"],
            ["capture", f"{_u(m['capture_s'], 's', 2)} ({_u(m['per_pattern_s'], 's', 3)} per pattern; "
                        f"average {m['average']}, hold {m['min_hold_s']:g} s)"],
            ["valid", f"{m['valid_pct']:.1f} %"], ["flood pair", "kept (flood.npz)"]])
    ctx.warn(m.get("warnings"))
    return EXIT_OK, {"metric_scan": ms, **({"zones": zr} if zr else {}), **_scan_says(m)}


def _flood_arg(key: Optional[str]):
    """--flood: a .npy / .npz file, or a metric scan (id / label) whose flood pair is used."""
    if not key:
        return None
    if Path(key).is_file():
        return str(Path(key).resolve())
    return _find_metric(key)


def _station_params(a, band: str = "band", flag: str = "--band") -> dict:
    """Station parameters from the options; `band` names the tube edge height band option
    (`station --band`, `station-golden --edge-band`: there --band is the per-item limit)."""
    bv = getattr(a, band, None)
    prm = _kw(ct_band_mm=(_numbers(bv, 2, f"{flag} lo,hi (mm)") if bv else None),
              ct_roi=_roi(a.roi),
              zone_skip=(_numbers(a.skip_zones, 2, "--skip-zones left,right", int)
                         if a.skip_zones else None),
              centre_method=a.centre)
    if a.no_ct:
        prm["ct"] = False
    if getattr(a, "no_strict", False):
        prm["strict"] = False
    if getattr(a, "size_tol", None) is not None:
        prm["size_tol_pct"] = a.size_tol
    return prm


def _print_station(ctx, m) -> None:
    ctx.table(["face", "camera px", "height mm", "flatness mm", "coverage", "outline mm"],
              [[c["name"], f"({c['camera_px'][0]:.1f}, {c['camera_px'][1]:.1f})",
                _f(c["height_mm"], 4), _f(c["flat_rms_mm"], 4), _f(c["coverage"], 2),
                _f(c["outline_diameter_mm"], 2)] for c in m.cells])
    ctx.table(["pair", "height diff mm", "spacing mm", "face angle deg"],
              [[p["pair"], _f(p["height_diff_mm"], 4), _f(p["spacing_inplane_mm"], 4),
                _f(p["face_angle_deg"], 3)] for p in m.pairs])
    ct = m.ct or {}
    if ct.get("trap_fit"):
        t = ct["trap_fit"]
        ctx.kv([["tube edge", f"{ct['n_px']} px, band {ct['band_mm']} mm ({ct['band_source']}), "
                            f"median {_u((ct.get('edge_height_all') or {}).get('median_mm'), 'mm', 3)}"],
                ["tube wave", f"pitch {_u(t['pitch_mm'], 'mm', 3)}, amplitude {_u(t['amp_mm'], 'mm', 3)}, "
                            f"phase {_u(t['phase_mm'], 'mm', 2)}, fit rms {_u(t['fit_rms_mm'], 'mm', 3)}"]])
    zs = [z for z in m.zones if z.get("included")]
    ctx.table(["zone", "k", "edge height mm", "p5", "p95", "px", "coverage"],
              [[z["label"], z["k"], _f(z.get("median_mm"), 3), _f(z.get("p5_mm"), 3),
                _f(z.get("p95_mm"), 3), z["n_px"], _f(z["coverage"], 2)] for z in zs])


def cmd_station(ctx):
    a = ctx.args
    ms = _find_metric(a.metric)
    m = api.station_measure(ms, _flood_arg(a.flood), a.label, a.cells, a.diameter,
                            _station_params(a) or None)
    ctx.say(f"Station {m.label}: scan {m.scan_id}, {len(m.cells)} end faces, "
            f"{sum(1 for z in m.zones if z.get('included'))} tube zones (flood: {m.flood_source})")
    _print_station(ctx, m)
    for x in (m.meta or {}).get("set_aside") or []:
        ctx.say(f"  set aside at camera px ({x['camera_px'][0]:.0f}, {x['camera_px'][1]:.0f}): "
                f"{x['why']}")
    ctx.warn(m.warnings)
    return EXIT_OK, {"station": m, "warnings": list(m.warnings or []),
                     "rig": (m.meta or {}).get("rig")}


def _station_item(key: str):
    p = Path(key)
    if p.suffix.lower() == ".json" and p.is_file():
        return str(p.resolve())
    return _find_metric(key)


def cmd_station_golden(ctx):
    a = ctx.args
    bands = {}
    for b in a.band or []:
        if "=" not in b:
            raise CLIError(f"--band {b!r}: use ITEM=LO,HI (e.g. C2-C1=-0.4,0.1 or T3=-5,-4)")
        k, v = b.split("=", 1)
        bands[k.strip()] = _numbers(v, 2, f"--band {k}")
    prm = _station_params(a, "edge_band", "--edge-band")
    prm.update(_kw(n_cells=a.cells, face_diameter_mm=a.diameter))
    g = api.station_golden([_station_item(k) for k in a.items], a.label,
                           params=prm or None,
                           **_kw(cell_height_tol_mm=a.cell_tol, ct_zone_tol_mm=a.ct_tol,
                                 spacing_tol_mm=a.spacing_tol,
                                 face_position_tol_mm=a.position_tol,
                                 cell_height_band_mm=(_numbers(a.cell_band, 2, "--cell-band")
                                                      if a.cell_band else None),
                                 ct_zone_band_mm=(_numbers(a.ct_band, 2, "--ct-band")
                                                  if a.ct_band else None),
                                 bands=bands or None, flood=_flood_arg(a.flood)))
    ctx.say(f"Station golden {g.id}: {g.n_measurements} measurement(s), "
            f"{len(g.nominal['zones'])} tube zones, calibration {g.calibration_id}")
    ctx.table(["pair", "height diff mm", "spread", "spacing mm"],
              [[p["pair"], _f(p["height_diff_mm"], 4), _f(p["height_diff_spread_mm"], 4),
                _f(p["spacing_inplane_mm"], 4)] for p in g.nominal["pairs"]])
    ctx.table(["zone", "k", "edge height mm", "spread", "n"],
              [[z["label"], z["k"], _f(z["median_mm"], 3), _f(z["spread_mm"], 3), z["n"]]
               for z in g.nominal["zones"]])
    lim = g.limits
    ctx.kv([["limits", f"end-face height +-{_f(lim['cell_height_tol_mm'], 3)} mm"
                       + (f" and within {lim['cell_height_band_mm']}" if lim["cell_height_band_mm"] else "")
                       + f"; tube zone +-{_f(lim['ct_zone_tol_mm'], 3)} mm"
                       + (f" and within {lim['ct_zone_band_mm']}" if lim["ct_zone_band_mm"] else "")
                       + (f"; spacing +-{lim['spacing_tol_mm']:g} mm" if lim["spacing_tol_mm"] else "")
                       + "; end-face position within "
                       + (f"{lim['face_position_tol_mm']:g} mm" if lim.get("face_position_tol_mm")
                          else "the face diameter")
                       + (f"; item bands {lim['bands']}" if lim["bands"] else "")],
            ["stored", str(g.path)]])
    ctx.warn(g.warnings)
    return EXIT_OK, {"station_golden": g}


def cmd_station_check(ctx):
    a = ctx.args
    gold = None
    if a.golden:
        gold = (str(Path(a.golden).resolve()) if Path(a.golden).is_dir() else
                _pick(api.list_library("station_golden"), a.golden, "station golden", "stations"))
    try:
        r = api.station_check(_station_item(a.scan), gold, _flood_arg(a.flood), a.label)
    except ValueError as e:                 # StationError is a ValueError
        if "size mismatch" in str(e):
            # an end face reads the wrong size (calibration or lens suspect): the strict
            # gates refuse the measurement, nothing was measured - PULL, said as what it is
            res = _pull([{"reason": "size_mismatch", "message": str(e)}])
            ctx.say(f"Station check: not measured ({_one_line(e)})",
                    f"  disposition: {_one_line(D.summary(res))}")
            return D.exit_code(res), {"check": None, "error": {"type": type(e).__name__,
                                                                "message": str(e)}, **res}
        if D.classify(str(e)) != "ambiguous_faces":
            raise
        # which discs are the end faces is in doubt: never measured, never a verdict - PULL
        res = {"result": "INVALID", "invalid_reason": str(e), "pull_reasons": []}
        D.add(res, "ambiguous_faces", str(e))
        D.settle(res)
        ctx.say(f"Station check: not measured ({_one_line(e)})",
                f"  disposition: {_one_line(D.summary(res))}")
        return D.exit_code(res), {"check": None, "error": {"type": type(e).__name__,
                                                            "message": str(e)},
                                  "disposition": res["disposition"],
                                  "pull_reasons": res["pull_reasons"]}
    ctx.say(f"Station check of {r.measurement['scan_id']}: {r.verdict} ({r.n_triggers} "
            f"trigger(s); golden {r.golden_id})")
    missing = [it["item"] for it in r.items if it.get("status") == "missing"]
    wrong = [it for it in r.items if it.get("status") == "size_mismatch"]
    if wrong:                               # the scan reads the wrong size: PULL, never a FAIL
        disp = _pull([{"reason": "size_mismatch", "message": f"{it['item']}: {it.get('message')}"}
                      for it in wrong])
    else:
        disp = _check_disposition(r.verdict, "not every golden item was measured"
                                  + (f" ({', '.join(map(str, missing[:5]))})" if missing else ""))
    ctx.say(f"  disposition: {_one_line(D.summary(disp))}")
    ctx.table(["item", "value mm", "nominal mm", "band mm", "status"],
              [[it["item"], _f(it["value_mm"], 3), _f(it["nominal_mm"], 3),
                "-" if not it.get("band_mm") else
                f"{_f(it['band_mm'][0], 3)} .. {_f(it['band_mm'][1], 3)}", it["status"]]
               for it in r.items])
    ctx.kv([[k, str(v)] for k, v in r.files.items() if v is not None])
    ctx.warn(r.warnings)
    return D.exit_code(disp), {"check": r, **disp}


def cmd_stations(ctx):
    gs = api.list_library("station_golden", ctx.args.geometry)
    ctx.say("Station goldens:")
    ctx.table(["id", "measurements", "faces", "zones", "calibration"],
              [[g.id, g.n_measurements, g.meta.get("n_cells"), len(g.nominal["zones"]),
                g.calibration_id] for g in gs])
    if not gs:
        ctx.say("  none yet (station-golden)")
    return EXIT_OK, {"station_golden": gs}


def _find_light(ctx, key: str):
    """A light separation (separate_light / `light`) by id, label or 'latest'."""
    rows = [{"id": d.name, "label": d.name.split("_", 2)[-1], "path": d}
            for d in ctx.session.store.entries("spectral", "light")
            if (d / "separation.npz").is_file()]
    return _pick(rows, key, "light separation", "light")["path"]


def cmd_maps_bake(ctx):
    a = ctx.args
    ctx.preset()
    scan = _find_scan(a.id)
    sp = _find_spectral(a.spectral) if a.spectral else None
    gold = _find_spectral(a.golden) if a.golden else None
    light = _find_light(ctx, a.light) if a.light else None
    ms = api.bake_maps(scan, spectral=sp, light=light, golden=gold, sigma_px=a.sigma)
    ctx.say(f"Surface maps of {scan.id}  (height {'mm' if ms.calibrated else 'relief px'})")
    ctx.table(["map", "png", "units"], [[n, str(ms.png_paths.get(n, "-")), ms.units.get(n, "")]
                                        for n in ms.maps()])
    ctx.warn(ms.warnings)
    return EXIT_OK, {"scan_id": scan.id, "maps": ms}


def cmd_checker(ctx):
    ctx.header()
    a = ctx.args
    c = api.checker_scan(ctx.preset(), a.square, a.shifts, a.label)
    jac = c.jacobian

    def med(name):
        v = np.asarray(getattr(jac, name, []), float)
        v = v[np.isfinite(v)]
        return float(np.median(v)) if v.size else None

    medians = {k: med(k) for k in jac.VALUES}
    ctx.say(f"Checker scan: {len(c.corners)} corners, {len(jac)} cells "
            f"({c.corners.square_px} DMD px squares = {_f(c.corners.cam_square_px, 1)} camera px)")
    ctx.kv([[f"median {k}", _f(v, 3)] for k, v in medians.items()] +
           [[f"{k} png", str(v)] for k, v in (c.png_paths or {}).items()])
    ctx.warn(list(c.warnings) + list(c.corners.warnings))
    return EXIT_OK, {"checker": {"label": c.label, "geometry_id": c.geometry_id,
                                 "n_corners": len(c.corners), "n_cells": len(jac),
                                 "square_px": c.corners.square_px,
                                 "cam_square_px": c.corners.cam_square_px, "median": medians,
                                 "png_paths": c.png_paths, "json_path": c.json_path,
                                 "npz_path": c.npz_path, "units": c.units,
                                 "warnings": list(c.warnings) + list(c.corners.warnings)}}


def cmd_atlas_import(ctx):
    a = ctx.args
    src = Path(a.path)
    if not src.is_file():
        raise CLIError(f"no file {a.path}")
    out = api.import_atlas(str(src), a.name, ctx.preset())
    ctx.say(f"Atlas {a.name!r} imported for {ctx.preset().geometry_id} (source only read)",
            f"  {out}", f"  zones: atlas:{a.name}/<region>  (see `atlases`)")
    return EXIT_OK, {"atlas": a.name, "file": out, "source": str(src)}


def cmd_atlases(ctx):
    from engine.capture.atlas import load_atlas
    rows = api.list_atlases(ctx.args.geometry)
    out = []
    ctx.say(f"Atlases ({len(rows)})")
    for r in rows:
        at = load_atlas(r["latest"], r["name"])
        regs = [{"name": g.name, "rect": list(g.rect), "label": g.label, "zone": g.zone_spec,
                 "spec": f"atlas:{r['name']}/{g.name}"} for g in at.regions.values()]
        ctx.say(f"  {r['name']}  ({r['geometry_id']}, {r['versions']} version(s), "
                f"step {at.step_px} px)")
        ctx.table(["zone spec", "DMD rect", "grid"], [[g["spec"], g["rect"], g["zone"] or "-"]
                                                      for g in regs], indent="    ")
        out.append({**r, "step_px": at.step_px, "regions": regs})
    if not rows:
        ctx.say("  none yet: atlas-import <path.json> <name>")
    return EXIT_OK, {"atlases": out}


def _carriers(text: str) -> list:
    out = []
    for part in [p for p in re.split(r"[;|]", text) if p.strip()]:
        try:
            v = [float(x) for x in re.split(r"[,\s]+", part.strip()) if x]
        except ValueError:
            v = []
        if len(v) not in (2, 3):
            raise argparse.ArgumentTypeError("--carriers: fx,fy[,z] separated by ';'")
        out.append(v)
    if not out:
        raise argparse.ArgumentTypeError("--carriers: fx,fy[,z] separated by ';'")
    return out


def _region_arg(text: str):
    t = text.strip()
    if t.lower().startswith("atlas:"):
        return "atlas:" + t[6:].strip()
    nums = re.split(r"[,\s]+", t)
    if len(nums) == 4 and all(re.fullmatch(r"-?\d+", n) for n in nums):
        return tuple(int(n) for n in nums)
    return t.upper()


def cmd_flood(ctx):
    ctx.header()
    a = ctx.args
    r = api.flood_verify(_region_arg(a.region), ctx.preset(), a.level, a.label)
    ctx.say(f"Flood verify {r.region}: {r.verdict}")
    ctx.kv([["lands", "yes" if r.lands else "no"], ["lit bbox", r.lit_bbox or "-"],
            ["lit area", _u(r.lit_area_px, "camera px", 0)],
            ["lit mean", _u(r.lit_mean_counts, "counts", 1)],
            ["outside mean", _u(r.outside_mean_counts, "counts", 1)],
            ["photo", str(r.camera_png or "-")], ["diff", str(r.diff_png or "-")]])
    ctx.warn(r.warnings)
    return (EXIT_OK if r.lands else EXIT_NOT_OK), {"flood": r}


def cmd_antiglare(ctx):
    ctx.header()
    a = ctx.args
    r = api.anti_glare_sweep(_region_arg(a.region), a.carriers, ctx.preset(), a.steps,
                             a.cycles, a.level, a.label)
    ctx.say(f"Anti-glare sweep {r.region}: lock {r.lock if r.lock else 'none'}",
            f"  {r.wording}")
    ctx.kv([["baseline clipping", _u(100.0 * r.baseline_clip_frac, "% (flat flood)", 2)],
            ["stored", str(r.json_path or "-")]])
    ctx.warn(r.warnings)
    return (EXIT_OK if r.lock else EXIT_NOT_OK), {"anti_glare": r}


def cmd_multiorient(ctx):
    ctx.header()
    a = ctx.args
    r = api.multi_orientation_scan(_region_arg(a.region), ctx.preset(), tuple(a.orientations),
                                   a.steps, a.period, a.level, a.label)
    ctx.say(f"Multi-orientation scan {r.region} ({', '.join(r.orientations)})")
    ctx.kv([[k, _f(v, 3) if isinstance(v, float) else str(v)] for k, v in (r.stats or {}).items()]
           + [["npz", str(r.npz_path or "-")], ["stored", str(r.json_path or "-")]])
    ctx.say(f"  note: {r.note}")
    ctx.warn(r.warnings)
    return EXIT_OK, {"multi_orientation": r}


def cmd_flash_pack(ctx):
    a = ctx.args
    p = Path(api.make_flash_pack(ctx.preset(), (a.axes,), a.video_port, a.label, a.period_us,
                                 a.pattern_exposure_us, a.trigger_mode or "trigout"))
    ctx.say(f"Flash pack ({'video port' if a.video_port else 'flash'}, axis {a.axes}): {p}",
            "  files only; the rig was not touched. A HUMAN enters pattern mode in the TI GUI.",
            "  procedure: procedure.txt in the pack;  docs/FLASH_PATTERN_MODE.md")
    return EXIT_OK, {"folder": p, "doc": "docs/FLASH_PATTERN_MODE.md"}


def cmd_fastscan_pack(ctx):
    a = ctx.args
    ctx.preset()                                  # geometry_id of the active preset
    p = Path(api.make_fastscan_pack(a.zone_ring, a.zone_mask, a.band_cam, a.band_field,
                                    a.margin, a.calibration, a.study, a.label, a.period_us,
                                    a.trigger_mode or "freerun", a.pattern_exposure_us,
                                    a.camera_exposure))
    m = json.loads((p / "manifest.json").read_text(encoding="utf-8"))
    t = m["timing"]
    hw = t.get("hardware_trigger")
    ctx.say(f"Fast-scan 1-bit pack ({', '.join(m['variants'])}): {p}",
            f"  16 one-bit slots per image on G0..G7, R0..R7; {t['trigger']} timer, "
            f"{t['pattern_period_us']} us per slot, loop {t['loop_ms']:g} ms, repeat",
            (f"  hardware trigger ({hw['mode']}): TRIG_OUT_1 -> camera, one frame per pattern, "
             f"camera exposure {hw['camera']['exposure_us']:g} us; {hw['set_ms']:g} ms per set"
             if hw else "  soft sync (free-running camera)"),
            "  files only; the rig was not touched. A HUMAN loads the images and plays the "
            "sequence in the 4500 MKII GUI.",
            "  procedure: procedure.txt in the pack;  docs/FLASH_FASTSCAN_4500.md")
    return EXIT_OK, {"folder": p, "variants": list(m["variants"]),
                     "doc": "docs/FLASH_FASTSCAN_4500.md"}


def cmd_scan_flash(ctx):
    a = ctx.args
    roi = _numbers(a.roi, 4, "--roi x0,y0,x1,y1 (camera px)", int) if a.roi else None
    r = api.fast_scan_flash(a.pack, a.variant, a.label, ctx.preset(), a.zone, roi, a.loops,
                            a.exposure, a.fps, a.gain, "plane", not a.no_recording, a.dry_run,
                            a.fine_prefilter)
    pl = r["plan"]
    cam = (f"camera {pl['roi'][2]}x{pl['roi'][3]} at ({pl['roi'][0]},{pl['roi'][1]}), "
           f"{pl['camera_exposure_us']:g} us, {pl['fps']:g} fps ({pl['fps_limit']})")
    loop = (f"loop {pl['loop_ms']:g} ms ({pl['n_patterns']} x {pl['pattern_period_us']:g} us), "
            f"{pl['loops']} loops -> {pl['n_frames']} frames, {pl['capture_ms']:g} ms")
    if r.get("dry_run"):
        e = r["estimate_s"]
        ctx.say("Flash soft-sync scan plan (dry run: nothing was opened)", f"  {cam}", f"  {loop}",
                f"  {pl['clean_frames_per_pattern']:g} clean frames per pattern expected; "
                f"about {e['total_s']:g} s in all (capture {e['capture_s']:g} s)",
                f"  calibration: {r['calibration'] or 'NONE (the scan will be refused, rule 6)'}")
        ctx.warn(r.get("warnings"))
        ctx.warn(pl.get("notes"))
        return EXIT_OK, r
    fs = r["frame_stats"]
    sc = fs["status_counts"]
    ctx.say(f"Flash soft-sync scan {r['id']}: {r['n_points']} points in mm (plane), "
            f"calibration {r['calibration_id']}", f"  {cam}", f"  {loop}")
    ze = fs.get("zone_edge") or {}
    pe = fs.get("prefilter_edge") or {}
    ctx.kv([["frames", f"{fs['n_frames']} ({sc['clean']} clean, {sc['straddle']} straddling a "
                       f"change, {sc['inconsistent']} rejected, {fs['frames_dropped']} dropped)"],
            ["per pattern", f"min {fs['min_count']} clean frames averaged"],
            ["signal", f"white - black p99 {_f(fs.get('signal_p99_dn'), 1)} DN"],
            ["loop measured", f"{r['loop_ms']:g} ms"],
            ["variant seen", str(r.get("variant_detected"))],
            ["confidence", _f(r["confidence"], 3)],
            ["valid", f"{_f(r['valid_pct'], 1)} %"]]
           + ([["zone edge", f"{ze['masked_px']} px left undecoded (decode white < "
                             f"{ze['min_white_over_sync_white']:g} x SYNC A)"]] if ze else [])
           + ([["depth edges", f"{pe['masked_px']} px left undecoded (mid / coarse phase "
                               f"coherence < {pe['min_coherence']:g})"]] if pe else [])
           + [["times", ", ".join(f"{k[:-2]} {v:g} s" for k, v in r["times_s"].items())],
              ["recording", str(r.get("recording") or "-")]])
    ctx.warn(r.get("warnings"))
    return EXIT_OK, r


def cmd_trigger_test(ctx):
    a = ctx.args
    r = api.trigger_test(a.line, a.seconds, a.expect_period_ms, a.exposure, ctx.preset())
    pm = r.get("period_ms") or {}
    ctx.say(f"Trigger test on {r['line']} ({r['seconds']:g} s, camera exposure "
            f"{r['exposure_us']:g} us): {r['verdict']}")
    ctx.kv([["pulses", str(r["count"])],
            ["rate", f"{_f(r.get('rate_hz'), 2)} Hz"],
            ["period", (f"mean {pm['mean']:g} ms, std {pm['std']:g}, min {pm['min']:g}, max "
                        f"{pm['max']:g}") if pm else "-"],
            ["expected", f"{r['expect_period_ms']:g} ms" if r.get("expect_period_ms") else "-"],
            ["missed (est.)", "-" if r.get("missed_pulses") is None else str(r["missed_pulses"])],
            ["frame-id gaps", str(r["frame_id_gaps"])], ["timeouts", str(r["timeouts"])],
            ["incomplete", str(r.get("incomplete_frames", 0))],
            ["expected count", "-" if r.get("expected_count") is None
             else str(r["expected_count"])],
            ["TriggerMode", f"{r['trigger_mode_restored']} (restored)"]])
    ctx.warn(r.get("notes"))
    return (EXIT_OK if r["verdict"] == "OK" else EXIT_NOT_OK), r


def cmd_scan_trigger(ctx):
    a = ctx.args
    from engine.capture.patternmode import ACK_PHRASE
    if not a.pattern_mode_set_by_human:
        raise CLIError("scan-trigger needs --pattern-mode-set-by-human: a HUMAN has put the "
                       "projector in Pattern Sequence mode (flash, TRIG_OUT_1 wired to the "
                       "camera) in the vendor program. This command never sends the projector "
                       "a command, and it gives mm only with a stereo calibration made in "
                       "pattern-sequence display (docs/TRIGGER_FULLSPEED_TEST.md)")
    r = api.scan_trigger(a.pack, a.variant, a.label, ctx.preset(), a.line, a.repeats,
                         a.exposure, ACK_PHRASE, a.first_timeout)
    t = r["timing"]
    pf = t.get("per_frame_ms") or {}
    ctx.say(f"Hardware-triggered scan {r['id']} (EXPERIMENTAL): {r['n_points']} points in mm "
            f"(plane), calibration {r['calibration_id']}")
    ctx.kv([["frames", f"{t['frames']} ({t['loops_captured']} x {t['frames_per_loop']}), "
                       f"{t['drops']} dropped"],
            ["capture", f"{t['total_ms']:g} ms; per frame {pf.get('mean', '-')} ms "
                        f"(min {pf.get('min', '-')}, max {pf.get('max', '-')}); expected "
                        f"{t['expected_period_ms']:g} ms, {t['set_ms']:g} ms per set"],
            ["valid", f"{_f(r['valid_pct'], 1)} %"],
            ["times", ", ".join(f"{k[:-2]} {v:g} s" for k, v in r["times_s"].items())]])
    ctx.warn(r.get("warnings"))
    return EXIT_OK, r


def cmd_sort_flash(ctx):
    a = ctx.args
    r = api.sort_flash_recording(a.recording, a.pack, a.variant, a.decode, a.label,
                                 a.fine_prefilter)
    sc = r["status_counts"]
    ctx.say(f"Sorted {r['recording']}: loop {r['loop_ms']:g} ms, confidence "
            f"{_f(r['confidence'], 3)}, variant seen {r.get('variant_detected')}")
    ctx.kv([["frames", f"{sc['clean']} clean, {sc['straddle']} straddling, "
                       f"{sc['inconsistent']} rejected"],
            ["per pattern", ", ".join(f"{k} {v}" for k, v in r["counts"].items())],
            ["sorted images", str(r["sorted"])]]
           + ([["decoded px", str(r["decoded_px"])]] if "decoded_px" in r else []))
    ctx.warn(r.get("warnings"))
    return EXIT_OK, r


# ------------------------------------------------------------ rig check (HOME)

def cmd_rig_check(ctx):
    a = ctx.args
    shift = None
    if a.mock_shift:
        if not ctx.mock:
            raise CLIError("--mock-shift needs --mock (it fakes a nudged camera)")
        n = len([s for s in re.split(r"[,\s]+", a.mock_shift.strip()) if s])
        shift = _numbers(a.mock_shift, min(max(n, 2), 4),
                         "--mock-shift dx,dy[,roll_deg[,scale]] (camera px, deg, x)")
    if a.rehome and a.set_reference:
        raise CLIError("--rehome and --set-reference exclude each other: a re-home is fitted "
                       "against the stored reference and stores the new one itself")
    r = api.rig_check(ctx.preset(), a.set_reference, a.tol, shift, rehome=bool(a.rehome))
    ctx.say(f"Rig check: {r['status']}  {r['message']}")
    rows = [["reference", r.get("reference_id") or "-"], ["base", r.get("base_id") or "-"],
            ["stereo calibration", r.get("calibration_id") or "-"]]
    for k in ("flood", "pattern"):
        v = r.get(k)
        if v:
            rows.append([f"{k} shift", f"dx {v['dx']:+.2f}, dy {v['dy']:+.2f} px "
                                       f"(|{v['shift_px']:.2f}| px, match {v['peak']:.2f})"])
            if v.get("motion_px") is not None:      # what the verdict judges (shift + roll + zoom)
                rz = (f"roll {v['rotation_deg']:+.3f} deg, scale {v['scale_pct']:+.3f} %, "
                      if v.get("similarity") else "roll / zoom not determined, ")
                rows.append([f"{k} motion", f"{rz}largest {v['motion_px']:.2f} px "
                                            f"({v.get('tiles_used', 0)}/{v.get('tiles', 0)} "
                                            "tiles)"])
    if r.get("shift_mm") is not None:
        rows.append(["~mm", f"{r['shift_mm']:.3f} mm at {r.get('working_distance_mm')} mm"])
    if r.get("camera_motion_px") is not None:       # each device's own (rigcheck.who_moved)
        pm = r.get("projector_motion_px")
        n = r.get("tiles_set_aside") or 0
        rows.append(["camera / projector",
                     f"camera's view {r['camera_motion_px']:.2f} px, projector's picture "
                     + (f"{pm:.2f} px against the scene" if pm is not None else "not compared")
                     + (f" ({n} part(s) of the frame moved with neither and were set aside: "
                        "the scene changed there since the reference?)" if n else "")])
    rh = r.get("rehome") or {}
    if rh.get("rotation_deg") is not None:          # the camera rotation fitted to the flood
        rows.append(["camera rotation",
                     f"{rh['rotation_deg']:.4f} deg (pan {rh['pan_deg']:+.4f}, tilt "
                     f"{rh['tilt_deg']:+.4f}, roll {rh['roll_deg']:+.4f} deg; shift at the "
                     f"principal point {rh['shift_px'][0]:+.2f}, {rh['shift_px'][1]:+.2f} px, "
                     f"scale {rh['scale_pct']:+.3f} %)"])
        rows.append(["rotation fit", f"residual {rh['residual_px']:.2f} px rms, "
                                     f"{rh.get('tiles_agree', 0)} of {rh.get('tiles_used', 0)} "
                                     f"flood tiles agree, match {_f(rh.get('match'), 2)}"
                                     + (f", pattern vs flood {rh['pattern_rms_px']:.2f} px"
                                        if rh.get("pattern_rms_px") is not None else "")])
    if rh.get("done"):
        rows.append(["re-homed calibration", f"{rh['calibration_id']}  ({rh['note']})"])
    elif a.rehome:
        rows.append(["re-home", f"NOT done: {rh.get('why') or 'not applicable'}"])
    elif rh.get("applicable"):
        rows.append(["re-home", "possible: scan rig-check --rehome  (rotation only; a full "
                                "calibration is the reference)"])
    elif rh.get("why"):
        rows.append(["re-home", f"not possible: {rh['why']}"])
    gate = api.rig_state(ctx.preset())
    if gate["mount"] == "mobile" or not gate["ok"]:     # what this check means for mm results
        rows.append(["rig", _rig_words(gate)])
    ctx.kv(rows)
    ctx.warn(r.get("warnings"))
    return (EXIT_OK if r["status"] in ("OK", "REFERENCE_SET", "REHOMED") else EXIT_NOT_OK), \
        {**r, "rig_state": gate}


def _rig_words(gate: dict) -> str:
    """One line: may this rig give mm now (engine/rigstate.gate)?"""
    st = gate.get("state") or {}
    if gate["ok"]:
        return (f"{gate['mount']}: verified by {st.get('verified_by')}"
                if gate["mount"] == "mobile" else "fixed: the calibration is trusted")
    return f"{gate['mount']}: NO mm - {_one_line(gate['message'])}"


def cmd_rig_session(ctx):
    """The rig's state for mm results; --new starts a new session (the rig was moved)."""
    a = ctx.args
    p = ctx.preset()
    ctx.header()
    marker = None
    if a.new:
        out = api.rig_session(p, a.reason or "")
        marker, gate = out["marker"], out["gate"]
        ctx.say(f"New rig session at geometry {p.geometry_id}: the rig counts as moved from "
                f"{marker['time']}" + (f" ({marker['reason']})" if marker.get("reason") else ""))
        if p.rig.mount != "mobile":
            ctx.say("  ! this preset's rig is fixed (rig.mount): the marker is recorded and "
                    "changes nothing")
    else:
        gate = api.rig_state(p)
    st = gate.get("state") or {}
    last = st.get("last_check") or {}
    ctx.kv([["mount", _mount_words(gate["mount"])
             + (f" - declared by preset {st['declared_by']} at this geometry (a geometry_id "
                "names one rig)" if st.get("declared_by") else "")],
            ["stereo calibration", st.get("calibration_id") or "none (uncalibrated)"],
            ["session started", st.get("session_start") or "-  (a fixed rig has no session)"],
            ["last rig check", f"{last.get('status')} at {last.get('time')}" if last else
             "none since the calibration"],
            ["mm results", "allowed" if gate["ok"] else
             f"NOT allowed: {_one_line(gate['message'])}"],
            *([["how to fix", _one_line(gate["how_to_fix"])]] if not gate["ok"] else [])])
    return (EXIT_OK if gate["ok"] else EXIT_NOT_OK), {"marker": marker, **gate}


# ------------------------------------------------------------ flow-ready job (run)

EXIT_INVALID = 5                # run only: the job ran, but its numbers cannot be trusted


def _run_inspect(ctx, cfg, p, base, log, res) -> None:
    scan = _run_capture(ctx, cfg, p, base, log, res)
    insp = api.inspect(scan, n_sigma=cfg.n_sigma, min_area_px=cfg.min_area_px)
    peaks = [abs(d.height) for d in insp.defects]     # blob heights, as the engine trigger
    res["engine_verdict"] = insp.verdict
    res["deviation_units"] = insp.units
    res["warnings"] += list(insp.warnings)
    if insp.path:
        res["outputs"]["report"] = str(insp.path)
        png = Path(insp.path).with_suffix(".png")
        res["outputs"]["deviation_map"] = str(png) if png.exists() else None
    log.event("inspect", verdict=insp.verdict, units=insp.units, defects=len(insp.defects),
              trigger_mm=insp.trigger, report=str(insp.path) if insp.path else None)
    if insp.units == "mm":
        res["deviation_mm"].update(max=max(peaks) if peaks else 0.0, method="relief_inspect")
        trigger = cfg.threshold_mm if cfg.threshold_mm is not None else insp.trigger
        res["threshold_mm"] = trigger
        if trigger is None:
            res["invalid_reason"] = "no FAIL threshold (threshold_mm or the preset tolerance)"
        elif cfg.threshold_mm is not None:
            res["result"] = "FAIL" if peaks and max(peaks) >= trigger else "PASS"
        elif insp.verdict in ("PASS", "FAIL"):
            res["result"] = insp.verdict
        else:
            res["invalid_reason"] = f"engine verdict {insp.verdict}"
    else:
        res["invalid_reason"] = (f"deviation in {insp.units}, not mm: no calibration at "
                                 f"geometry {scan.geometry_id} (rule 6)")
        if not cfg.require_mm and insp.verdict in ("PASS", "FAIL"):
            res["result"] = insp.verdict
    if res["result"] != "INVALID":
        res["invalid_reason"] = None


def _run_compare(ctx, cfg, p, base, log, res) -> None:
    golden = _find_station_golden(cfg, "objects")
    scan = _run_capture(ctx, cfg, p, base, log, res)
    tol = None
    if cfg.threshold_mm is not None:
        from engine.results import ToleranceBand
        tol = ToleranceBand(fail_mm=cfg.threshold_mm, min_defect_px=p.tolerance.min_defect_px)
    v = api.compare(scan, golden, tol)
    res["engine_verdict"] = v.result
    res["deviation_units"] = v.units
    res["warnings"] += list(v.warnings)
    log.event("compare", golden=golden.id, verdict=v.result, units=v.units,
              max_abs_dev=v.max_abs_dev, p99_abs_dev=v.p99_abs_dev)
    if v.units == "mm":
        res["deviation_mm"].update(max=v.max_abs_dev, method="compare_to_golden")
        res["threshold_mm"] = v.tolerance
    if v.units != "mm" and cfg.require_mm:
        res["invalid_reason"] = (f"deviation in {v.units}, not mm: no calibration at "
                                 f"geometry {scan.geometry_id} (rule 6)")
    elif v.result in ("PASS", "FAIL"):
        res["result"], res["invalid_reason"] = v.result, None
    else:
        res["invalid_reason"] = f"engine verdict {v.result}"


CALIBRATE_STEREO_HOWTO = ("calibrate it first, with the preset this run uses: `scan "
                          "--preset {preset} charuco` (print at 100 %, caliper a square), `scan "
                          "--preset {preset} calib-pose` per board pose (12-20, tilted +-30 "
                          "deg), then `scan --preset {preset} calibrate-stereo --square MM` "
                          "(mock: `scan --mock --preset {preset} calibrate-stereo --mock-poses "
                          "12`)")
METRIC_MATCH_MM = 5.0           # golden: registration pairing radius (mm)
# golden: the background (dominant) plane is the datum; a test scan whose background plane is
# turned more than this from the golden's (or METRIC_MATCH_MM away) is INVALID: the rig or
# the fixture moved, or it is another background
METRIC_DATUM_MAX_DEG = 2.0
# out-of-tolerance px this close (camera px, 8-neighbourhood distance) are one cluster: a
# stray cluster is dropped only when it is isolated, never a patch of pits or spikes
METRIC_ISOLATION_PX = 3
# ... and dropped stray px are never dense: a METRIC_DENSE_WIN_PX square holding min_defect_px
# or more of them, METRIC_DENSE_SHARE or more of its area (pits a few px apart, a noisy patch),
# makes the run INVALID - never a PASS
METRIC_DENSE_WIN_PX = 21
METRIC_DENSE_SHARE = 0.02
# golden: only px decoded with enough decoded neighbours (METRIC_MIN_SUPPORT of a
# METRIC_SUPPORT_PX square) and seen within METRIC_MAX_INCIDENCE_DEG of face-on are judged
METRIC_SUPPORT_PX = 5
METRIC_MIN_SUPPORT = 0.5
METRIC_MAX_INCIDENCE_DEG = 75.0
# ... and a foreground region (decoded px joined across small gaps) of METRIC_REGION_MIN_PX or
# more decoded px with under METRIC_MIN_JUDGED_SHARE of them judged (decoded too sparsely, a
# saturating face) can't be compared: INVALID, never a component left out unseen
METRIC_REGION_MIN_PX = 50
METRIC_MIN_JUDGED_SHARE = 0.2
_CONN8 = np.ones((3, 3), bool)  # out-of-tolerance clusters: 8-connected camera px
_CONN4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)


def _mock_place_metric(ctx) -> None:
    """Mock only: the stereo (mm) bench. validation / none = the empty wall (flat),
    H,W,L = a block standing out of the wall, cylinder[:R,H,dent] = a standing cylinder."""
    if not ctx.mock:
        return
    from engine.capture.mock_stereo import Block, MetricCylinder
    bench = api.mock_bench()
    bench.clear()
    bench.use_stereo()
    spec = ctx.args.mock_part
    if spec.startswith("scene:"):
        from engine.capture import mock_scenes
        ctx.say(f"[mock] stereo bench: placed {mock_scenes.place(bench, spec)}")
        return
    if spec in ("none", "validation") or spec.startswith("qr:"):
        ctx.say("[mock] stereo bench: the empty wall (a flat reference)")
        return
    if spec.startswith("cylinder:"):
        r, hh, dent = (float(v) for v in spec.split(":", 1)[1].split(","))
        dents = ([{"depth_mm": dent, "size_mm": 8.0, "angle_deg": 0.0, "z_mm": 0.0}]
                 if dent > 0 else [])
        bench.place(MetricCylinder(center=(0, 0, 823.0 - r - 1.0), radius=r, height=hh,
                                   dents=dents))
        ctx.say(f"[mock] stereo bench: placed a standing cylinder (R {r:g}, H {hh:g} mm)")
        return
    h, w, l = [float(v) for v in spec.split(",")][:3]
    bench.place(Block(center=(0.0, 0.0, 823.0 - l / 2), half=(w / 2, h / 2, l / 2)))
    ctx.say(f"[mock] stereo bench: placed a block (H {h:g}, W {w:g}, L {l:g} mm) on the wall")


def _roi_mask(shape, roi) -> np.ndarray:
    m = np.zeros(shape, bool)
    if roi is None:
        m[:] = True
    else:
        x0, y0, x1, y1 = roi
        m[y0:y1, x0:x1] = True
    return m


def _tolerance_clusters(dev_abs: np.ndarray, thr: float, min_px: int,
                        area: Optional[np.ndarray] = None, side: str = "test") -> dict:
    """|deviation| map (H, W) mm, NaN = not measured -> its out-of-tolerance clusters:
    camera px with |dev| > thr, joined when within METRIC_ISOLATION_PX px of each other (a
    patch of small pits or spikes is ONE cluster). A cluster of fewer than min_px px is an
    isolated outlier (a stray triangulated point) and is dropped; every other cluster is a
    defect, and the max over everything not dropped is the robust max. No statistic
    (percentile, mean) can hide a defect of min_px px or more. What was dropped is returned
    (outlier_px, outlier_clusters, of n_measured px; `dense`: the densest METRIC_DENSE_WIN_PX
    window where dropped px are min_px or more AND METRIC_DENSE_SHARE or more of its area -
    pits a few px apart, a noisy patch - else None): the caller refuses a verdict when that is
    more than a few stray points (job.max_outlier_pct) or dense anywhere."""
    from scipy import ndimage
    ok = np.isfinite(dev_abs)
    a = np.where(ok, dev_abs, 0.0)
    over = ok & (a > thr)
    k = max(int(METRIC_ISOLATION_PX) - 1, 0) // 2      # joins px up to 2k+1 px apart
    near = ndimage.binary_dilation(over, structure=np.ones((2 * k + 1,) * 2, bool)) if k else over
    grp, n = ndimage.label(near, structure=_CONN8)
    lab = np.where(over, grp, 0)
    drop = np.zeros(n + 1, bool)
    defects = []
    if n:
        sizes = np.bincount(lab.ravel(), minlength=n + 1)
        for i, sl in enumerate(ndimage.find_objects(lab), 1):
            if sl is None:
                continue
            if sizes[i] < min_px:
                drop[i] = True
                continue
            b = lab[sl] == i
            ar = float(np.nansum(area[sl][b])) if area is not None else None
            defects.append({"side": side, "n_px": int(sizes[i]),
                            "area_mm2": round(ar, 3) if ar is not None else None,
                            "max_mm": round(float(a[sl][b].max()), 5),
                            "bbox_px": [sl[1].start, sl[0].start, sl[1].stop, sl[0].stop]})
    outlier = drop[lab]
    kept = ok & ~outlier
    dense = None
    if int(outlier.sum()) >= min_px:
        w = int(METRIC_DENSE_WIN_PX)
        cnt = ndimage.uniform_filter(outlier.astype(np.float64), size=w, mode="constant") * w * w
        # of the window's AREA, not of its measured px: a few strays along an outline (a
        # window mostly off the part) are not a dense patch
        hot = cnt >= max(min_px, METRIC_DENSE_SHARE * w * w) - 1e-6
        if hot.any():
            i = int(np.argmax(np.where(hot, cnt, -1.0)))
            y, x = divmod(i, cnt.shape[1])
            dense = {"side": side, "n_px": int(round(cnt.flat[i])),
                     "share_pct": round(100.0 * float(cnt.flat[i]) / (w * w), 2),
                     "window_px": [x - w // 2, y - w // 2, x + w // 2 + 1, y + w // 2 + 1]}
    return {"robust_max": float(a[kept].max()) if kept.any() else None, "defects": defects,
            "outlier_px": int(outlier.sum()), "outlier_clusters": int(drop.sum()),
            "n_measured": int(ok.sum()), "dense": dense}


def _defect_order(d: dict) -> tuple:
    return (d["area_mm2"] if d["area_mm2"] is not None else -1.0, d["n_px"])


def _foreground_plane(cloud, min_height_mm: float = 2.0) -> tuple:
    """(mask of every decoded px more than min_height_mm in front of the dominant (background)
    plane: ALL parts and components, not only the largest blob (metric.foreground keeps one);
    that plane, or None when too little is decoded)."""
    from engine.measure import metric
    m = cloud.mask
    if int(m.sum()) < 100:
        return np.zeros_like(m), None
    bg, _ = metric.ransac_plane(cloud.xyz[m], 0.5)
    h = np.full(m.shape, -np.inf)
    h[m] = bg.distance(cloud.xyz[m])
    return m & (h > min_height_mm), bg


def _foreground_all(cloud, min_height_mm: float = 2.0) -> np.ndarray:
    """Every decoded px more than min_height_mm in front of the dominant (background) plane:
    ALL parts and components, not only the largest blob (metric.foreground keeps one)."""
    return _foreground_plane(cloud, min_height_mm)[0]


def _boundary_px(fg: np.ndarray, background: np.ndarray, region: np.ndarray) -> np.ndarray:
    """The part's outline: fg px with a 4-neighbour in the well-decoded background (the
    part's silhouette against the wall). Not the edges of undecoded holes and shadows (they
    differ from scan to scan) and not the image or roi border (it does not move with the
    part)."""
    from scipy import ndimage
    bg_next = ndimage.binary_dilation(background, structure=_CONN4)
    keep = ndimage.binary_erosion(region, structure=_CONN8, border_value=0)
    return fg & bg_next & keep


def _well_decoded(mask: np.ndarray) -> np.ndarray:
    """Decoded px with at least METRIC_MIN_SUPPORT of their METRIC_SUPPORT_PX square
    neighbourhood decoded too: a px decoded alone in a shadow or a glint is a flying point,
    different in every scan, never a surface to judge (golden compare)."""
    from scipy import ndimage
    share = ndimage.uniform_filter(mask.astype(np.float32), size=METRIC_SUPPORT_PX,
                                   mode="constant")
    return mask & (share >= METRIC_MIN_SUPPORT - 1e-6)


def _judged(xyz: np.ndarray, fg: np.ndarray, k: int = 12) -> np.ndarray:
    """fg px whose surface faces the camera within METRIC_MAX_INCIDENCE_DEG (golden compare):
    a surface seen grazing (a side wall, a cylinder's silhouette) samples other points in
    every placement, so it is not judged; it still serves as the reference surface. The
    surface is that of the k points AROUND each px, not the px itself: a pit or a spike of a
    px or a few is judged where it sits, never tilted out of the compare by its own offset."""
    from scipy.spatial import cKDTree
    out = np.zeros_like(fg)
    if int(fg.sum()) < 12:
        return out
    P = xyz[fg]
    _, nn = cKDTree(P).query(P, k=min(k + 1, len(P)))
    Q = P[nn[:, 1:]]                            # the neighbours, not the px itself
    Q = Q - Q.mean(1, keepdims=True)
    nP = np.linalg.eigh(np.einsum("nki,nkj->nij", Q, Q))[1][:, :, 0]
    cos_in = np.abs(np.einsum("ij,ij->i", nP, P)) / np.linalg.norm(P, axis=1)
    out[fg] = cos_in >= math.cos(math.radians(METRIC_MAX_INCIDENCE_DEG))
    return out


def _outline_normals(xyz: np.ndarray, fg: np.ndarray, bnd: np.ndarray, n: np.ndarray,
                     spacing: float, win: int = 7) -> np.ndarray:
    """Outward unit normals ALONG the datum plane (normal n) of the outline px `bnd` of `fg`,
    in raster order: away from the mean of the fg px around each one on its surface (within
    2 mm). A zero row where the outline is too thin to tell its outside (fg on both sides)."""
    ys, xs = np.nonzero(bnd)
    P = xyz[ys, xs]
    H, W = fg.shape
    acc, cnt = np.zeros_like(P), np.zeros(len(P))
    r = win // 2
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if not (dx or dy):
                continue
            yy, xx = ys + dy, xs + dx
            inside = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
            yy, xx = np.clip(yy, 0, H - 1), np.clip(xx, 0, W - 1)
            d = xyz[yy, xx] - P
            ok = inside & fg[yy, xx] & (np.einsum("ij,ij->i", d, d) < 4.0)    # NaN: False
            acc[ok] += d[ok]
            cnt[ok] += 1
    v = -acc / np.maximum(cnt, 1.0)[:, None]
    v -= np.outer(v @ n, n)
    nv = np.linalg.norm(v, axis=1)
    out = np.zeros_like(v)
    good = (cnt >= 3) & (nv > 0.5 * spacing)
    out[good] = v[good] / nv[good, None]
    return out


def _unjudged_regions(dec: np.ndarray, judged: np.ndarray, roi: np.ndarray, side: str) -> list:
    """Foreground regions in the roi (decoded fg px joined across gaps of a few px) with
    METRIC_REGION_MIN_PX or more decoded px of which under METRIC_MIN_JUDGED_SHARE are judged:
    a component decoded too sparsely (a saturating face, a dark one) that the compare can't
    see -> [{side, n_px, n_judged, bbox_px}], largest first."""
    from scipy import ndimage
    m = dec & roi
    if not m.any():
        return []
    lab, nl = ndimage.label(ndimage.binary_closing(m, structure=np.ones((5, 5), bool)) | m,
                            structure=_CONN8)
    n_dec = np.bincount(lab[m], minlength=nl + 1)
    n_jud = np.bincount(lab[m & judged], minlength=nl + 1)
    out = []
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        if sl is not None and n_dec[i] >= METRIC_REGION_MIN_PX and \
                n_jud[i] < METRIC_MIN_JUDGED_SHARE * n_dec[i]:
            out.append({"side": side, "n_px": int(n_dec[i]), "n_judged": int(n_jud[i]),
                        "bbox_px": [sl[1].start, sl[0].start, sl[1].stop, sl[0].stop]})
    return sorted(out, key=lambda r: -r["n_px"])


def _rot_onto(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The smallest rotation taking unit vector a onto unit vector b."""
    from engine.measure import metric
    v = np.cross(a, b)
    sn = float(np.linalg.norm(v))
    if sn < 1e-12:
        return np.eye(3)
    return metric._rodrigues(v / sn * math.atan2(sn, float(a @ b)))


def _tukey_sqrt_w(r: np.ndarray, c: float) -> np.ndarray:
    u = np.clip(np.abs(r) / c, 0.0, 1.0)
    return 1.0 - u * u                          # sqrt of Tukey's biweight (1 - u^2)^2


def _register_in_plane(A: np.ndarray, Ab: np.ndarray, G: np.ndarray, nG: np.ndarray,
                       Gb: np.ndarray, n: np.ndarray, R: np.ndarray, t: np.ndarray, thr: float,
                       max_iter: int = 100) -> dict:
    """Place the test part (A, its edge px Ab) on the golden (G, normals nG, edge px Gb) by a
    turn about the datum normal n and a shift along the datum plane ONLY: height and tilt are
    the part's own, measured against the background (a cell 0.8 mm proud, a thinner part or a
    tilted lid are never fitted away). Point-to-plane on the surface plus the edges (the
    in-plane position a flat face can't give), Tukey-weighted with a scale annealed from
    METRIC_MATCH_MM down to the tolerance (the edges: to no less than 3 sample spacings, their
    own sampling noise), so a defect far out never drags the placement. A placement that
    splits an in-plane size error between two ends is a legitimate best fit: the verdict
    measures each end's offset past the other part's outline in full (_golden_compare)."""
    from scipy.spatial import cKDTree
    from engine.measure import metric
    rng = np.random.default_rng(0)
    if len(A) > 30000:
        A = A[rng.choice(len(A), 30000, replace=False)]
    if len(Ab) > 20000:
        Ab = Ab[rng.choice(len(Ab), 20000, replace=False)]
    e1 = np.cross(n, (1.0, 0.0, 0.0) if abs(n[0]) < 0.9 else (0.0, 1.0, 0.0))
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    Pn = np.eye(3) - np.outer(n, n)             # onto the datum plane
    tG = cKDTree(G)
    tB = cKDTree(Gb) if len(Gb) >= 10 and len(Ab) >= 10 else None
    probe = G[rng.choice(len(G), min(len(G), 3000), replace=False)]
    sp = float(np.median(tG.query(probe, k=2)[0][:, 1]))      # golden sample spacing
    c_end = max(thr, 3.0 * sp)
    c, converged, it = METRIC_MATCH_MM, False, 0
    r = rb = np.zeros(0)
    for it in range(1, max_iter + 1):
        p = A @ R.T + t
        d, j = tG.query(p, distance_upper_bound=METRIC_MATCH_MM)
        ok = np.isfinite(d)
        p, q, m = p[ok], G[j[ok]], nG[j[ok]]
        r = np.einsum("ij,ij->i", p - q, m)
        sig = 1.4826 * float(np.median(np.abs(r))) if len(r) else 0.0
        w = _tukey_sqrt_w(r, max(c, thr, 4.685 * sig))
        rows = [np.c_[np.einsum("ij,ij->i", np.cross(n, p), m), m @ e1, m @ e2] * w[:, None]]
        rhs = [-r * w]
        rb = np.zeros(0)
        if tB is not None:
            pb = Ab @ R.T + t
            db, jb = tB.query(pb, distance_upper_bound=METRIC_MATCH_MM)
            okb = np.isfinite(db)
            pb = pb[okb]
            rv = (pb - Gb[jb[okb]]) @ Pn.T      # edge offset along the datum plane
            rb = np.linalg.norm(rv, axis=1)
            if len(pb):
                wb = _tukey_sqrt_w(rb, max(c, c_end)) * math.sqrt(max(len(p), 1) / len(pb))
                Jb = np.stack([np.cross(n, pb) @ Pn.T, np.broadcast_to(Pn @ e1, pb.shape),
                               np.broadcast_to(Pn @ e2, pb.shape)], axis=2)
                rows.append((Jb * wb[:, None, None]).reshape(-1, 3))
                rhs.append((-rv * wb[:, None]).reshape(-1))
        J, b = np.concatenate(rows), np.concatenate(rhs)
        if len(b) < 6:
            break
        x = np.linalg.lstsq(J, b, rcond=None)[0]
        dR = metric._rodrigues(n * x[0])        # a turn about n keeps the datum plane
        R, t = dR @ R, dR @ t + x[1] * e1 + x[2] * e2
        small = abs(x[0]) < 1e-5 and math.hypot(x[1], x[2]) < 1e-3     # um: far under noise
        if small and c <= c_end:
            converged = True
            break
        if small or it % 3 == 0:
            c = max(c_end, c * 0.5)
    score = (float(np.mean(np.abs(r) <= thr)) if len(r) else 0.0) * len(r) / max(len(A), 1)
    score += (float(np.mean(rb <= c_end)) * len(rb) / max(len(Ab), 1)) if len(rb) else 0.0
    return {"R": R, "t": t, "converged": converged, "iterations": it, "score": score,
            "rms_mm": float(np.sqrt(np.mean(r * r))) if len(r) else float("nan"),
            "fitness": float(len(r) / max(len(A), 1))}


def _golden_compare(s, ms, golden, roi, thr: float, min_px: int) -> dict:
    """Golden metric compare in BOTH directions, on every foreground component in the roi.

    The background (dominant) plane is the datum: the test scan's is brought onto the
    golden's (beyond METRIC_DATUM_MAX_DEG / METRIC_MATCH_MM: refused, INVALID), and the test
    part is then placed on the golden only by a turn and a shift ALONG that plane
    (_register_in_plane): a re-placed good part is found again, while a height or tilt
    change of the part or of any one component stays in the deviation.
    Judged px (both scans, same rule): foreground, well decoded (_well_decoded) and not seen
    grazing (_judged); every foreground px still serves as the surface the other scan is
    measured against. A foreground region with too few judged px (decoded too sparsely) is
    named in `unjudged`: the caller refuses a verdict, it is never left out unseen.
    test -> golden: each test point's distance to the golden surface - the largest of the
    point-to-plane distance at the nearest golden point, the straight distance to it less the
    golden's own sample spacing (material far off the golden surface), and, when that nearest
    point is on the golden's outline against the wall, how far the test point lies beyond
    that outline along its outward normal in the datum plane (extra material or a component
    moved sideways, measured in full: no sample spacing is taken off it). golden -> test: the
    same for each golden point against the placed test part (missing material). Both maps go
    through the same out-of-tolerance clustering (_tolerance_clusters); the stray share and
    any dense strays are judged per side (outlier_*: the worse side)."""
    from scipy.spatial import cKDTree
    from engine.library.store import json_safe
    from engine.measure import metric
    from engine.reconstruct import metric_scan
    metric_scan.check_cal(s, ms, "compare_metric")
    metric_scan.check_cal(s, golden, "compare_metric (golden)")
    rules.require_same_geometry(ms.geometry_id, golden.geometry_id, "compare_metric")
    a, g = ms.cloud, golden.cloud
    (fa, bga), (fg, bgg) = _foreground_plane(a), _foreground_plane(g)
    wa, wg = _well_decoded(a.mask), _well_decoded(g.mask)
    roim = _roi_mask(fa.shape, roi)
    # judged px: well decoded and not seen grazing, by the same rule on both scans; every
    # foreground px stays part of the surface the other scan is measured against
    ra = _judged(a.xyz, fa & wa) & roim
    rg = _judged(g.xyz, fg & wg) & roim
    out = {"n_test": int(ra.sum()), "n_golden": int(rg.sum()), "golden": golden.id}
    if out["n_test"] < 100 or out["n_golden"] < 100:
        return out
    n = bgg.normal
    R0 = _rot_onto(bga.normal, n)
    off = float(n @ (bgg.point - R0 @ bga.point))
    tilt = math.degrees(math.acos(float(np.clip(bga.normal @ n, -1.0, 1.0))))
    out["datum"] = {"tilt_deg": round(tilt, 4), "offset_mm": round(off, 3)}
    if tilt > METRIC_DATUM_MAX_DEG or abs(off) > METRIC_MATCH_MM:
        out["refused"] = (f"the background plane (the datum) is {tilt:.2f} deg / {off:.2f} mm "
                          f"off the golden {golden.id}'s (limits {METRIC_DATUM_MAX_DEG:g} deg / "
                          f"{METRIC_MATCH_MM:g} mm): the rig or the fixture moved, or another "
                          "background - check the rig (or scan a new golden)")
        return out
    G = g.xyz[fg]
    nG = metric.estimate_normals(G)
    idx = np.full(fg.shape, -1)
    idx[fg] = np.arange(len(G))
    gbm = _boundary_px(fg, wg & ~fg, np.ones_like(fg))
    Gb = G[idx[gbm]]
    A, Ab = a.xyz[ra], a.xyz[_boundary_px(fa, wa & ~fa, roim)]
    t0 = off * n
    Pn = np.eye(3) - np.outer(n, n)
    shift = Pn @ (g.xyz[rg].mean(0) - (A @ R0.T + t0).mean(0))
    fits = [_register_in_plane(A, Ab, G, nG, Gb, n, R0, t0, thr)]
    if float(np.linalg.norm(shift)) > 0.5:      # also from the centroids: the better wins
        fits.append(_register_in_plane(A, Ab, G, nG, Gb, n, R0, t0 + shift, thr))
    reg = max(fits, key=lambda f: f["score"])
    R, t = reg["R"], reg["t"]
    T = a.xyz[fa] @ R.T + t                              # the whole test part, placed
    tG, tT = cKDTree(G), cKDTree(T)
    # each point's own sample spacing: to its 4th nearest neighbour, so the spacing ACROSS a
    # surface seen at a slant (a cylinder's flank, sampled coarsely across it) counts too; on
    # a face-on surface (4 neighbours one px away) it is the px spacing
    sG = tG.query(G, k=5)[0][:, 4]
    sT = tT.query(T, k=5)[0][:, 4]
    # outward normals of each part's outline against the wall, along the datum plane (zero
    # elsewhere): how far a point lies beyond the other part's outline
    oG = np.zeros_like(G)
    oG[idx[gbm]] = _outline_normals(g.xyz, fg, gbm, n, float(np.median(sG)))
    ia = np.full(fa.shape, -1)
    ia[fa] = np.arange(len(T))
    abm = _boundary_px(fa, wa & ~fa, np.ones_like(fa))
    oT = np.zeros_like(T)
    oT[ia[abm]] = _outline_normals(a.xyz, fa, abm, bga.normal, float(np.median(sT))) @ R.T
    Ta = A @ R.T + t                                     # test -> golden (judged px)
    e, j = tG.query(Ta)
    pp = np.einsum("ij,ij->i", Ta - G[j], nG[j])
    beyond = np.einsum("ij,ij->i", Ta - G[j], oG[j])     # past the golden's outline
    fwd = np.maximum.reduce([np.abs(pp), e - sG[j], beyond])
    Gr = g.xyz[rg]                                       # golden -> test (judged px)
    gi = idx[rg]
    e2, k = tT.query(Gr)
    pp2 = np.einsum("ij,ij->i", T[k] - Gr, nG[gi])
    beyond2 = np.einsum("ij,ij->i", Gr - T[k], oT[k])    # past the test part's outline
    rev = np.maximum.reduce([np.abs(pp2), e2 - sT[k], beyond2])
    signed = np.full(ra.shape, np.nan)
    signed[ra] = np.where((pp >= 0) | (beyond > np.abs(pp)), 1.0, -1.0) * fwd
    fwd_map = np.full(ra.shape, np.nan)
    fwd_map[ra] = fwd
    rev_map = np.full(rg.shape, np.nan)
    rev_map[rg] = rev
    area_a = metric.pixel_area_mm2(np.where(a.mask[..., None], a.xyz, np.nan))
    area_g = metric.pixel_area_mm2(np.where(g.mask[..., None], g.xyz, np.nan))
    ct = _tolerance_clusters(fwd_map, thr, min_px, area_a, "test")
    cg = _tolerance_clusters(rev_map, thr, min_px, area_g, "golden")
    maxes = [v for v in (ct["robust_max"], cg["robust_max"]) if v is not None]
    turn = math.degrees(math.acos(float(np.clip((np.trace(R @ R0.T) - 1.0) / 2.0, -1.0, 1.0))))
    # the stray share is judged per side: strays on the test part show in its map only, and
    # the other side's clean px must not water them down
    worse = max((ct, cg), key=lambda c: c["outlier_px"] / max(c["n_measured"], 1))
    dense = [c["dense"] for c in (ct, cg) if c["dense"]]
    unjudged = (_unjudged_regions(fa, ra, roim, "test")
                + _unjudged_regions(fg, rg, roim, "golden"))
    out.update(icp={"rms_mm": reg["rms_mm"], "fitness": reg["fitness"],
                    "iterations": reg["iterations"], "converged": reg["converged"]},
               placement={"shift_mm": round(float(np.linalg.norm(Pn @ (t - t0))), 3),
                          "turn_deg": round(turn, 3)},
               signed=signed, fwd=fwd, robust_max=max(maxes) if maxes else None,
               defects=sorted(ct["defects"] + cg["defects"], key=_defect_order, reverse=True),
               outlier_px=worse["outlier_px"], outlier_clusters=worse["outlier_clusters"],
               n_measured=worse["n_measured"], outlier_side=("test" if worse is ct else "golden"),
               outlier_all_pct=100.0 * (ct["outlier_px"] + cg["outlier_px"])
               / max(ct["n_measured"] + cg["n_measured"], 1),
               dense_strays=max(dense, key=lambda d: d["share_pct"]) if dense else None,
               unjudged=unjudged,
               test_matched_pct=round(100.0 * float(np.mean(fwd <= thr)), 2),
               golden_found_pct=round(100.0 * float(np.mean(rev <= thr)), 2))
    rep = {"scan": ms.id, "golden": golden.id, "units": "mm", "threshold_mm": thr, "roi": roi,
           "min_defect_px": min_px, "n_test": out["n_test"], "n_golden": out["n_golden"],
           "datum": out["datum"], "placement": out["placement"],
           "icp": {**out["icp"], "R": R.tolist(), "t": t.tolist()},
           "test_matched_pct": out["test_matched_pct"],
           "golden_found_pct": out["golden_found_pct"], "robust_max_mm": out["robust_max"],
           "defects": out["defects"], "outlier_px": out["outlier_px"],
           "outlier_clusters": out["outlier_clusters"], "n_measured": out["n_measured"],
           "outlier_side": out["outlier_side"], "dense_strays": out["dense_strays"],
           "unjudged": unjudged}
    f = s.store.output_path("comparisons", f"{ms.label}_vs_{golden.label}_metric", ".json")
    s.store.write_json(f, json_safe(rep))
    s.store.record_output("comparisons", f, label=ms.label, geometry_id=ms.geometry_id,
                          source=ms.path)
    out["report"] = str(f)
    return out


def _find_station_golden(cfg, kind: str = "metric"):
    """job.golden -> the golden scan ('metric' scans for metric jobs, 'objects' for compare
    jobs): an explicit id or label, never 'latest' (every run saves its own test scan, so that
    is the previous part) and never this station's part.label (its test scans carry it)."""
    key = str(cfg.golden or "").strip()
    what, cmd_ = ("metric scan", "metric") if kind == "metric" else ("scan", "scans")
    if key.lower() in ("latest", "last") or key == cfg.part_label:
        raise CLIError(f"job.golden {key!r}: name the golden {what} by its id or label (a GOOD "
                       "part scanned once), never 'latest' or this station's part.label (that "
                       "would compare the part with an earlier test scan)")
    golden = _pick(api.list_library(kind), key, what, cmd_, allow_latest=False)
    if golden.label == cfg.part_label:
        raise CLIError(f"job.golden {key!r} resolves to {golden.id}, labelled with this "
                       f"station's part.label {cfg.part_label!r}: an earlier test scan, not a golden")
    return golden


def _deviation_png(run_dir: Path, dev: np.ndarray, scale_mm: float) -> Optional[str]:
    """Signed deviation map -> run_dir/deviation.png (red above, blue below, +-scale_mm)."""
    try:
        import cv2
    except ImportError:
        return None
    ok = np.isfinite(dev)
    d = np.clip(np.nan_to_num(dev, nan=0.0) / max(scale_mm, 1e-6), -1, 1)
    pos, neg = np.clip(d, 0, 1), np.clip(-d, 0, 1)
    img = np.full(dev.shape + (3,), 40, np.uint8)               # BGR, grey = no data
    img[..., 2] = np.where(ok, 128 + 127 * pos - 100 * neg, 40).astype(np.uint8)
    img[..., 1] = np.where(ok, 128 - 100 * np.abs(d), 40).astype(np.uint8)
    img[..., 0] = np.where(ok, 128 + 127 * neg - 100 * pos, 40).astype(np.uint8)
    h, w = img.shape[:2]
    if w > 960:
        img = cv2.resize(img, (960, int(round(h * 960 / w))), interpolation=cv2.INTER_AREA)
    out = Path(run_dir) / "deviation.png"
    ok_write, buf = cv2.imencode(".png", img)
    if not ok_write:
        return None
    with open(out, "xb") as f:                  # never overwrite (a run folder is new)
        f.write(buf.tobytes())
    return str(out)


def _run_metric(ctx, cfg, p, log, res, run_dir) -> None:
    """metric job: scan_fast / scan_metric (stereo calibration, rule 6) -> mm, then a plane
    fit in the roi or a golden metric compare (both directions); PASS / FAIL against
    threshold_mm. The verdict comes from out-of-tolerance clusters (_tolerance_clusters):
    only isolated clusters smaller than job.min_defect_px are dropped, as stray points, and
    more of them than job.max_outlier_pct of the measured px (golden: of both scans together,
    or of either scan's own once they are two defect clusters' worth), or dropped px dense
    anywhere, or (golden) a foreground region too sparsely decoded to judge, is INVALID
    (never a PASS). A golden FAIL whose placement did not converge, or whose surfaces agree
    below job.min_coverage_pct with no defect cluster to account for it, stays FAIL but
    carries an `incomplete` doubt: PULL, never a REJECT on numbers it marks untrusted."""
    from engine import config as cfgmod
    from engine.calibrate import stereo
    from engine.measure import metric
    s = api._engine()
    if cfg.threshold_mm is None:
        res["invalid_reason"] = ("metric jobs need threshold_mm (mm) in the station config: "
                                 "there is no preset default for a metric measurement")
        D.add(res, "invalid", res["invalid_reason"])
        log.event("refused", level="warning", reason=res["invalid_reason"])
        return
    cal = stereo.latest(s.store, p.geometry_id, cfgmod.geometry_fingerprint(p))
    if cal is None:
        res["calibration"]["same_geometry"] = False
        res["invalid_reason"] = (f"no stereo calibration for geometry {p.geometry_id} (these rig "
                                 f"settings), so no mm (rule 6): "
                                 + CALIBRATE_STEREO_HOWTO.format(preset=p.name))
        D.add(res, "uncalibrated", res["invalid_reason"])
        log.event("refused", level="warning", reason=res["invalid_reason"])
        return
    res["calibration"].update(calibration_id=cal.id, same_geometry=True)
    golden = _find_station_golden(cfg, "metric") if cfg.measure == "golden" else None
    s.check_cancel()
    _mock_place_metric(ctx)
    if cfg.scan == "metric":
        ms = api.scan_metric(p, cfg.part_label, steps=cfg.steps, periods=cfg.periods,
                             zone=cfg.zone)
    else:                                       # --average wins, else job.average (cmd_run)
        avg = ctx.args.average if ctx.args.average is not None else (cfg.average or 2)
        ms = api.scan_fast(cfg.zone, p, cfg.part_label, cfg.steps or 4, cfg.periods, avg,
                           axis=cfg.axis)
    m = ms.meta or {}
    res["part"]["scan_id"] = ms.id
    res["calibration"]["calibration_id"] = ms.calibration_id
    pats = f"scan_fast axis {cfg.axis}" if cfg.scan == "fast" else "scan_metric"
    res["capture"].update(patterns=f"multifreq ({pats})",
                          average=m.get("average", res["capture"]["average"]))
    res["warnings"] += [str(w) for w in m.get("warnings") or []]
    log.event("scan", scan_id=ms.id, kind=cfg.scan, calibration_id=ms.calibration_id,
              n_points=m.get("n_points"), valid_pct=m.get("valid_pct"))
    s.check_cancel()
    cloud = ms.cloud
    thr, min_px = cfg.threshold_mm, cfg.min_defect_px
    roi = _roi_mask(cloud.mask.shape, cfg.roi)
    incomplete = None
    untrusted = None            # why every deviation is in doubt (a FAIL on them is a PULL)
    if cfg.measure == "plane":
        method = "plane_fit"
        region = cloud.mask & roi
        P = cloud.xyz[region]
        cov = 100.0 * len(P) / max(int(roi.sum()), 1)
        extra = {"roi_valid_pct": round(cov, 2), "min_valid_pct": cfg.min_valid_pct}
        if len(P) < 100 or cov < cfg.min_valid_pct:
            res["invalid_reason"] = (f"only {len(P)} points ({cov:.1f} % of the roi) decoded, "
                                     f"below job.min_valid_pct {cfg.min_valid_pct:g} %: too little "
                                     "to measure (check the part, roi, zone and exposure)")
            D.add(res, "too_few_points", res["invalid_reason"])
            log.event("measure", level="warning", method=method, n_points=int(len(P)), **extra)
            return
        fit, _ = metric.ransac_plane(P, 0.5)
        dist = abs(float(fit.normal @ fit.point))       # plane to camera centre, mm
        extra.update(plane_rms_inliers_mm=round(fit.rms, 5), plane_distance_mm=round(dist, 3))
        if cfg.plane_range_mm is not None and not (
                cfg.plane_range_mm[0] <= dist <= cfg.plane_range_mm[1]):
            res["invalid_reason"] = (f"the fitted plane is {dist:.1f} mm from the camera, outside "
                                     f"job.plane_range_mm {list(cfg.plane_range_mm)}: no part "
                                     "surface where it should be (part absent or misplaced?)")
            D.add(res, "invalid", res["invalid_reason"])
            log.event("measure", level="warning", method=method, n_points=int(len(P)), **extra)
            return
        dev = np.full(cloud.mask.shape, np.nan)
        dev[region] = fit.distance(P)
        d = dev[region]
        area = metric.pixel_area_mm2(np.where(cloud.mask[..., None], cloud.xyz, np.nan))
        cl = _tolerance_clusters(np.abs(dev), thr, min_px, area, "test")
        mx, outlier_px = cl["robust_max"], cl["outlier_px"]
        outlier_n, n_measured, dense = cl["outlier_clusters"], cl["n_measured"], cl["dense"]
        defects = sorted(cl["defects"], key=_defect_order, reverse=True)
    else:
        method = "compare_metric_golden"
        gc = _golden_compare(s, ms, golden, cfg.roi, thr, min_px)
        extra = {"golden": golden.id, "n_golden_points": gc["n_golden"],
                 "datum": gc.get("datum")}
        if gc.get("refused"):
            res["invalid_reason"] = gc["refused"]
            D.add(res, D.classify(gc["refused"], "golden"), res["invalid_reason"])
            log.event("measure", level="warning", method=method, n_points=gc["n_test"], **extra)
            return
        if "robust_max" not in gc:
            res["invalid_reason"] = (f"only {gc['n_test']} test / {gc['n_golden']} golden "
                                     f"foreground points in the roi (golden {golden.id}): wrong "
                                     "part, placement or roi?")
            D.add(res, "too_few_points", res["invalid_reason"])
            log.event("measure", level="warning", method=method, n_points=gc["n_test"], **extra)
            return
        extra.update(icp_rms_mm=gc["icp"]["rms_mm"], icp_fitness=gc["icp"]["fitness"],
                     icp_converged=gc["icp"]["converged"], placement=gc["placement"],
                     test_matched_pct=gc["test_matched_pct"],
                     golden_found_pct=gc["golden_found_pct"],
                     min_coverage_pct=cfg.min_coverage_pct, report=gc["report"])
        res["outputs"]["report"] = gc["report"]
        dev = gc["signed"]
        d = dev[np.isfinite(dev)]
        mx, defects, outlier_px = gc["robust_max"], gc["defects"], gc["outlier_px"]
        outlier_n, n_measured, dense = gc["outlier_clusters"], gc["n_measured"], gc["dense_strays"]
        extra.update(outlier_side=gc["outlier_side"], unjudged=gc["unjudged"][:5])
        if not gc["icp"]["converged"]:
            incomplete = untrusted = (
                "placing the part on the golden (ICP along the background plane) did not "
                "converge: every deviation depends on it, can't be trusted (check the "
                "placement)")
        elif min(gc["golden_found_pct"], gc["test_matched_pct"]) < cfg.min_coverage_pct:
            incomplete = (f"only {gc['golden_found_pct']:g} % of the golden surface and "
                          f"{gc['test_matched_pct']:g} % of the test surface agree within "
                          f"{thr:g} mm (job.min_coverage_pct {cfg.min_coverage_pct:g} %) "
                          "without one defect cluster: scattered or partial data, can't pass")
            # a defect cluster accounts for the surface that does not agree (missing material
            # is a defect, measured); without one, a FAIL rests on scattered data
            untrusted = None if defects else incomplete
        elif gc["unjudged"]:
            u = gc["unjudged"][0]
            incomplete = (f"{len(gc['unjudged'])} foreground region(s) can't be compared: "
                          f"e.g. {u['n_px']} decoded px of the {u['side']} scan at camera px "
                          f"{u['bbox_px']} and only {u['n_judged']} of them well enough decoded "
                          "to judge (decoded too sparsely: a saturating or dark surface?) - "
                          "a component left out unseen can't pass")
    outlier_pct = 100.0 * outlier_px / max(n_measured, 1)
    over = outlier_pct > cfg.max_outlier_pct
    if cfg.measure == "golden":
        # either scan's own share (strays on the test part alone are never watered down by the
        # golden's clean px) once they are two defect clusters' worth - a handful of stray px on
        # a small part is not noisy data - or both scans together, as one scan's share
        over = ((over and outlier_px >= 2 * min_px)
                or gc["outlier_all_pct"] > cfg.max_outlier_pct)
    if incomplete is None and over:
        incomplete = (f"{outlier_px} px ({outlier_pct:.3g} % of {n_measured} measured) beyond "
                      f"{thr:g} mm sit in {outlier_n} isolated stray cluster(s) under "
                      f"job.min_defect_px {min_px} px, more than job.max_outlier_pct "
                      f"{cfg.max_outlier_pct:g} %: noisy data or a pitted surface, can't pass")
    if incomplete is None and dense:
        incomplete = (f"{dense['n_px']} stray px ({dense['share_pct']:g} % of the measured px) "
                      f"beyond {thr:g} mm within {METRIC_DENSE_WIN_PX} camera px of each other "
                      f"at camera px {dense['window_px']} ({dense['side']} side), each under "
                      f"job.min_defect_px {min_px} px: pits or a noisy patch, never stray "
                      "points - can't pass")
    if mx is None:                              # every measured px was a dropped stray cluster
        res["invalid_reason"] = ("nothing left to judge once the stray points (clusters under "
                                 f"job.min_defect_px {min_px} px) are dropped")
        D.add(res, "too_few_points", res["invalid_reason"])
        log.event("measure", level="warning", method=method, n_points=int(d.size), **extra)
        return
    big = defects[0] if defects else None
    res["deviation_units"] = "mm"
    res["deviation_mm"].update(max=round(mx, 5), mean=round(float(np.mean(d)), 5),
                               rms=round(float(np.sqrt(np.mean(d * d))), 5), method=method,
                               n_points=int(d.size), n_defects=len(defects), largest_defect=big,
                               outlier_px=outlier_px, outlier_clusters=outlier_n,
                               outlier_pct=round(outlier_pct, 4), dense_strays=dense)
    res["outputs"]["deviation_map"] = _deviation_png(run_dir, dev, thr)
    if defects or mx > thr:                     # any defect cluster fails, whatever else
        verdict = "FAIL"
    elif incomplete:
        verdict = None                          # INVALID: never a PASS on untrusted numbers
    else:
        verdict = "PASS"
    if verdict is None:
        res["invalid_reason"] = incomplete
        D.add(res, "incomplete", incomplete)
    else:
        res["result"] = res["engine_verdict"] = verdict
        res["invalid_reason"] = None
        if verdict == "FAIL" and untrusted:     # the verdict stands, the reject does not
            D.add(res, "incomplete", f"FAIL on numbers that can't be trusted: {untrusted}")
    log.event("measure", level="info" if verdict else "warning", method=method,
              max_deviation_mm=res["deviation_mm"]["max"], rms_mm=res["deviation_mm"]["rms"],
              n_points=int(d.size), threshold_mm=thr, min_defect_px=min_px,
              n_defects=len(defects), largest_defect=big, outliers_dropped_px=outlier_px,
              outlier_clusters=outlier_n, outlier_pct=round(outlier_pct, 4),
              max_outlier_pct=cfg.max_outlier_pct, dense_strays=dense, **extra)


def _run_capture(ctx, cfg, p, base, log, res):
    _mock_place(ctx)                        # mock: the simulated human places the part
    scan = api.capture_object(p, base, cfg.part_label, zone=cfg.zone)
    cal = scan.calibration_id
    if cal is None:
        c = api._scan_calibration(scan)
        cal = c.id if c is not None else None
    res["part"]["scan_id"] = scan.id
    res["calibration"].update(calibration_id=cal, same_geometry=cal is not None)
    res["capture"]["patterns"] = "+".join((scan.meta or {}).get("basis_names") or []) or None
    log.event("scan", scan_id=scan.id, base_id=base.id, calibration_id=cal,
              valid_pct=(scan.meta or {}).get("valid_pct"))
    return scan


def _flow_inputs(items) -> dict:
    """--input SIGNAL=true|false (repeatable) -> the sensor claims {signal: bool}."""
    out = {}
    for it in items or []:
        key, sep, val = str(it).partition("=")
        val = val.strip().lower()
        if not sep or not key.strip() or val not in ("true", "false"):
            raise CLIError(f"--input {it!r}: write SIGNAL=true or SIGNAL=false (what the "
                           "line's sensor says)")
        out[key.strip()] = val == "true"
    return out


def _flow_table(ctx, res: dict) -> None:
    """The work states of a cycle: what the sensor said, what the DLP said, the outcome."""
    def says(d: dict) -> str:
        return ", ".join(f"{k}={'?' if v is None else str(v).lower()}"
                         for k, v in (d or {}).items()) or "-"
    rows = []
    for s in res.get("steps") or []:
        dlp = s.get("dlp_result") or {}
        seen = says(s.get("observed")) if s.get("expects") else (dlp.get("result") or "-")
        # GRANTED only when it still stands at the end of the cycle (what `permissions` says)
        note = str(s.get("permission_note") or "")
        perm = "-" if not s.get("permit") else f"{s['permit']} " + (
            "GRANTED" if s.get("permission_granted") else
            "dropped" if note.startswith("dropped") else "withheld")
        rows.append([str(s.get("index")), s.get("id"), says(s.get("sensor_inputs")), seen,
                     s.get("outcome"), perm, _one_line(s.get("message") or "")[:70]])
    ctx.table(["#", "work state", "sensor says", "DLP says", "outcome", "permission", ""], rows)
    for s in res.get("steps") or []:
        for act in s.get("actions") or []:
            bits = [f"{k}={v}" for k, v in act.items() if k != "action" and v not in (None, "")]
            ctx.say(f"    {s.get('id')}: {act.get('action')}"
                    + (f" ({_one_line(', '.join(bits))[:110]})" if bits else ""))


def cmd_flow_show(ctx):
    """A station configuration as the engine reads it: parameters, profiles, work states."""
    a = ctx.args
    chk = api.flow_check(a.config)
    if not chk["valid"]:
        if ctx.json:
            return EXIT_REFUSED, {"valid": False, "problems": chk["problems"],
                                  "config": chk["config"]}
        raise CLIError(f"{a.config}: {len(chk['problems'])} problem(s):\n  - "
                       + "\n  - ".join(chk["problems"]))
    c = chk["config"]
    ctx.say(f"Station configuration {c['station_id']}"
            + (" (a single-job config: one implicit profile, a one-step flow)"
               if c["legacy"] else "") + (f": {c['description']}" if c["description"] else ""))
    ctx.kv([["preset", c["preset"] or "(each recipe's own / the selected one)"],
            ["sensor inputs", ", ".join(c["signals"]) or "none"],
            ["permissions", ", ".join("PERMIT_" + p for p in c["permissions"]) or "none"],
            ["flow", c["flow_source"] or "inline"]])
    ctx.say("", "Profiles (the profile library set):")
    ctx.table(["profile", "title", "preset", "recipes"],
              [[p["name"] + (" *" if p["name"] == c["default_profile"] else ""), p["title"],
                p["preset"] or "-", ", ".join(f"{k}={v}" for k, v in p["recipes"].items()) or "-"]
               for p in c["profiles"]])
    ctx.say("", "Work states (the flow):")
    ctx.table(["#", "id", "title", "expects", "observed by", "verify", "permit", "trigger"],
              [[str(s["index"]), s["id"], s["title"],
                ", ".join(f"{k}={str(v).lower()}" for k, v in s["expects"].items()) or "-",
                # the check whose answer IS that sensor signal (verify.observe)
                ", ".join(f"{k} <- {v}" for k, v in (s.get("observe") or {}).items()) or "-",
                s["verify"] + ("" if s.get("required", True) else " (not required)"),
                s["permit"] or "-", s["trigger"]] for s in c["flow"]])
    for s in c["flow"]:
        for key in ("on_mismatch", "on_not_met", "on_defect", "on_doubt"):
            if s[key]:
                ctx.say(f"    {s['id']}.{key}: " + " -> ".join(_one_line(x) for x in s[key]))
    return EXIT_OK, {"valid": True, "problems": [], "config": c}


def _run_flow(ctx):
    """`scan run` on a station configuration with profiles + a flow: one whole cycle, in
    line - every work state is verified in turn against the sensor claims (--input), its
    actions run, a new cycle folder is written (result.json: station config, profile, steps,
    permissions, disposition). The operator page and a PLC walk the same cycle step by step
    through the command bus."""
    from engine.flow import schema as FS
    from engine.flow import verify as FV
    a = ctx.args
    try:
        cfg = FS.load(a.config)
    except FS.StationConfigError as e:
        raise CLIError(str(e)) from e
    params = cfg.parameters
    lib = FS.rel_path(cfg.source.parent, params["mock_library"] if (
        ctx.mock and params["mock_library"] is not None) else params["library"])
    if lib is not None and not a.library:           # --library on the command line wins
        ctx.session = api.configure(lib.resolve(), mock=ctx.mock)
        ctx._outputs_before = len(ctx.session.store.outputs())
    probs = FV.check_config(cfg)
    if probs:
        raise CLIError(f"station configuration {cfg.station_id}: {len(probs)} problem(s):\n  - "
                       + "\n  - ".join(probs))
    profile = a.profile or cfg.default_profile
    if profile not in cfg.profiles:
        raise CLIError(f"--profile {profile!r}: station configuration {cfg.station_id} has "
                       f"profiles {', '.join(cfg.profiles)}")
    inputs = _flow_inputs(a.input)
    scene = None
    if ctx.mock:
        try:
            case = cfg.mock_case(a.mock_case, profile)
        except FS.StationConfigError as e:
            raise CLIError(f"--mock-case: {e}") from e
        scene = case["scene"]
        inputs = {**case["inputs"], **inputs}       # --input wins over the case's sensors
        if getattr(a, "mock_part_given", False):
            scene = a.mock_part
            if scene != "none" and not scene.startswith("scene:"):
                raise CLIError(f"--mock-part {scene}: a flow's work states scan the metric "
                               "bench: use scene:<name>[:<variant>] or none")
    elif a.mock_case:
        raise CLIError("--mock-case is MOCK ONLY: on the real rig a human places the part and "
                       "the sensors are the line's own (--input)")
    unknown = sorted(set(inputs) - set(cfg.signals))
    if unknown:
        raise CLIError(f"--input {', '.join(unknown)}: not a sensor signal of this configuration "
                       f"({', '.join(cfg.signals) or 'it expects none'})")
    confirmed = bool(a.operator_confirmed or params["operator_confirmed"])
    if not ctx.mock and not confirmed:
        raise CLIError("the real rig needs a human to confirm the physical setup first (part "
                       "placed, rig untouched since the calibration): pass --operator-confirmed, "
                       "or set parameters.operator_confirmed in the station configuration")

    def on_prompt(p: dict) -> bool:
        if a.acknowledge:
            ctx.say("    acknowledged (--acknowledge)")
            return True
        if not ctx.interactive:
            return False
        return _prompt(ctx, f"    {p.get('title')}: {p.get('message')} - done? [y/N] ",
                       WIZARD_TIMEOUT_S).strip().lower() in ("y", "yes")

    ctx.say(f"[{'MOCK' if ctx.mock else 'REAL RIG'}] station configuration {cfg.station_id}, "
            f"profile {profile}: {len(cfg.flow)} work state(s)"
            + (f"; mock: {scene or 'the empty wall'}" if ctx.mock else ""))
    out = FV.run_cycle(cfg, profile, inputs=inputs, operator_confirmed=confirmed,
                       mock_scene=scene, on_prompt=on_prompt, project=not a.no_feedback,
                       say=ctx.say)
    res = out["result"]
    ctx.say(f"Cycle {out['cycle_id']}: {res['result']}"
            + (f"  ({_one_line(res['invalid_reason'])})" if res["invalid_reason"] else ""))
    ctx.say(f"  disposition: {_one_line(D.summary(res))}")
    _flow_table(ctx, res)
    on = [k for k, v in (res.get("permissions") or {}).items() if v]
    ctx.kv([["station", cfg.station_id], ["profile", profile],
            ["permissions", ", ".join(on) or "none granted"],
            ["result.json", str(Path(out["cycle_dir"]) / "result.json")],
            ["log", str(Path(out["cycle_dir"]) / "run.log.jsonl")]])
    return out["exit_code"], {"run_id": out["cycle_id"], "run_dir": str(out["cycle_dir"]),
                              "result": res}


def cmd_run(ctx):
    """One job from a station config -> a new run folder with result.json + run.log.jsonl.
    A station configuration with profiles + a flow runs one whole cycle (_run_flow)."""
    from engine.flow import schema as FS
    from ui import run_job as RJ
    a = ctx.args
    try:
        is_flow = not FS.is_legacy(FS.read_doc(a.config))
    except FS.StationConfigError:
        is_flow = False                 # missing / unreadable: load_station_config says why
    if is_flow:
        return _run_flow(ctx)
    for opt, given in (("--profile", a.profile), ("--input", a.input),
                       ("--mock-case", a.mock_case), ("--acknowledge", a.acknowledge)):
        if given:
            raise CLIError(f"{opt} is for a station configuration with profiles and a flow; "
                           f"{Path(a.config).name} is a single-job station config")
    try:
        cfg = RJ.load_station_config(a.config)
    except RJ.StationConfigError as e:
        raise CLIError(str(e)) from e
    return _run_job(ctx, cfg)


def _run_job(ctx, cfg, quick: Optional[dict] = None):
    """One single job (a StationConfig) -> its run folder. quick: the generated config of a
    QUICK CHECK (engine/modes/quick.py; a config file marked operating_mode quick_check is
    one too): the same job, with result.json marked, the config kept in the run folder and
    the rig asked strictly (any rig whose newest rig check said MOVED / UNKNOWN is a PULL)."""
    from engine import modes, rigstate
    from engine.modes import quick as quick_mode
    from ui import run_job as RJ
    a = ctx.args
    if quick is None and cfg.raw.get(modes.RESULT_KEY) == quick_mode.MODE:
        quick = cfg.raw
    lib = cfg.mock_library if (ctx.mock and cfg.mock_library is not None) else cfg.library
    if lib is not None and not a.library:           # --library on the command line wins
        ctx.session = api.configure(lib.resolve(), mock=ctx.mock)
        ctx._outputs_before = len(ctx.session.store.outputs())
        if not a.preset_given and ctx.selection is not None:
            ctx.use_selected_preset()               # THIS library's selection, not the default's
    recipe = None
    if cfg.job_type == "recipe":                    # the recipe pins its preset
        try:
            recipe = api.recipe_load(cfg.recipe)
        except ValueError as e:
            raise CLIError(f"job.recipe {cfg.recipe!r}: {e}") from e
        if not a.preset_given and not cfg.preset:
            a.preset = recipe.preset
    if cfg.preset and not a.preset_given:           # --preset on the command line wins
        a.preset = cfg.preset
    if cfg.average is not None and a.average is None:
        a.average = cfg.average
    if ctx.mock and cfg.mock_part and not getattr(a, "mock_part_given", False):
        try:                                        # the config's simulated human (MOCK only)
            a.mock_part = _mock_part(cfg.mock_part)
        except argparse.ArgumentTypeError as e:
            raise CLIError(f"mock_part {cfg.mock_part!r}: {e}") from e
        a.mock_part_given = True
    confirmed = bool(a.operator_confirmed or cfg.operator_confirmed)
    run_id, run_dir = RJ.new_run(cfg)
    try:
        folder = str(run_dir.relative_to(Path.cwd()))
    except ValueError:
        folder = str(run_dir)
    res = RJ.empty_result(cfg, run_id, folder, ctx.mock, confirmed)
    if quick is not None:                           # QUICK CHECK: one check, no sensors
        res[modes.RESULT_KEY] = quick_mode.MODE
        if cfg.source is None:                      # generated now: kept with its run
            (run_dir / quick_mode.CONFIG_FILE).write_text(json.dumps(quick, indent=2) + "\n",
                                                          encoding="utf-8")
            res["job"]["config"] = quick_mode.CONFIG_FILE
    log = RJ.RunLog(run_dir / "run.log.jsonl", run_id)
    code = EXIT_INVALID
    try:
        log.event("run_start", station=cfg.station_id, mode=res["mode"], job=cfg.job_type,
                  config=str(cfg.source) if cfg.source else res["job"]["config"],
                  operator_confirmed=confirmed,
                  **({"operating_mode": quick_mode.MODE} if quick is not None else {}))
        if not ctx.mock and not confirmed:
            raise CLIError("the real rig needs a human to confirm the physical setup first "
                           "(part placed, rig untouched since the base): pass "
                           "--operator-confirmed, or set operator_confirmed in the station config")
        if not a.preset_given and not cfg.preset and recipe is None:
            # the config names no preset, so it runs the library's selection: one that no
            # longer resolves stops the run, whatever preset is loaded in its place (the web
            # server's station runs pass the loaded one, possibly its default)
            sel = ctx.selection if ctx.selection is not None else api.preset_selection()
            if sel.get("name") and not sel.get("exists"):
                instead = (f", it would run {a.preset!r} instead"
                           if a.preset and a.preset != sel["name"] else "")
                raise CLIError(_selection_gone(
                    sel, ctx.session.store.root,
                    f" and station config {cfg.source.name} names no preset{instead}",
                    "name one in the station config (or with --preset)"))
        ctx.header()
        p = ctx.preset()
        res["capture"].update(preset=p.name, geometry_id=p.geometry_id,
                              exposure_us=p.camera.exposure_us, average=p.camera.average)
        log.event("preset", preset=p.name, geometry_id=p.geometry_id,
                  exposure_us=p.camera.exposure_us, average=p.camera.average)
        # a MOBILE rig (preset rig.mount) gives no mm until it was verified in this session;
        # a quick check also refuses any rig whose newest rig check said MOVED / UNKNOWN.
        # Station mode on a fixed rig is not gated (engine/rigstate.py): unchanged
        gate = rigstate.gate(ctx.session.store, p, strict=quick is not None,
                             presets_dir=ctx.session.presets_dir)
        if not gate["ok"]:
            res["invalid_reason"] = gate["message"]
            D.add(res, gate["code"], gate["message"])
            log.event("refused", level="warning", reason=gate["message"], mount=gate["mount"],
                      how_to_fix=gate["how_to_fix"])
            _settle_run(res, recipe)                # a doubt: PULL
            return _run_done(ctx, cfg, run_id, run_dir, res, log, D.exit_code(res))
        if cfg.job_type == "metric":            # stereo calibration: no base involved
            _run_metric(ctx, cfg, p, log, res, run_dir)
            _settle_metric(res, cfg)
            api._engine().check_cancel()        # an ABORT up to here: ABORTED, not judged
            return _run_done(ctx, cfg, run_id, run_dir, res, log, D.exit_code(res))
        if cfg.job_type == "recipe":            # stereo calibration: zones, checks, feedback
            return _run_recipe(ctx, cfg, recipe, p, log, res, run_dir, run_id)
        base = None if cfg.base == "capture" else api.latest_base(p)
        if base is None and cfg.base == "latest" and not ctx.mock:
            raise CLIError(f"no usable base for geometry {p.geometry_id}: clear the scene "
                           "and run `base` first (or set job.base: capture)")
        captured = base is None
        if captured:                        # mock: the simulated human clears the scene
            if cfg.base == "latest":
                res["warnings"].append("mock: no usable base yet, captured one first")
            _mock_clear(ctx)
            base = api.capture_base(p, f"{cfg.station_id}_base")
            if base.meta.get("rejected"):
                raise CLIError(f"base {base.id} was auto-rejected: {base.meta['rejected']}")
        log.event("base", base_id=base.id, captured=captured)
        (_run_compare if cfg.job_type == "compare" else _run_inspect)(ctx, cfg, p, base, log, res)
        D.settle(res)
        code = D.exit_code(res)
        api._engine().check_cancel()            # an ABORT up to here: ABORTED, not judged
    except rules.Cancelled as e:              # the operator pressed ABORT: rig already closed
        res["result"], res["invalid_reason"] = "ABORTED", None
        res["warnings"].append(f"aborted: {e}")
        D.add(res, "aborted", f"aborted: {e}")
        _settle_run(res, recipe)              # ABORTED: PULL (exit stays 1, as documented)
        log.event("aborted", level="warning", message=str(e))
        RJ.write_result(run_dir, res)
        log.event("run_end", result="ABORTED", disposition=res["disposition"],
                  pull_reasons=res["pull_reasons"], exit_code=EXIT_NOT_OK,
                  result_json="result.json")
        log.close()
        ctx.say(f"Run {run_id}: ABORTED ({e})", f"  disposition: {_one_line(D.summary(res))}",
                f"  run folder: {run_dir}")
        return EXIT_NOT_OK, {"run_id": run_id, "run_dir": str(run_dir), "result": res}
    except (Exception, KeyboardInterrupt) as e:
        ecode, line = _classify_error(e, ctx.mode)
        ecode = _exit_code_for(e, ecode, a)     # what main() exits with: a crash is 4, not 1
        res["result"] = "INVALID"
        res["invalid_reason"] = line
        # what happened, never keyword-classified (a crash text is not engine wording): a
        # run that never finished is a PULL, never a reject
        D.add(res, "refused" if ecode == EXIT_REFUSED else "error", line)
        _settle_run(res, recipe)                # refused / crashed: PULL (exit 2 / 4 kept)
        log.event("error", level="error", type=type(e).__name__, message=line, exit_code=ecode)
        RJ.write_result(run_dir, res)
        try:                                # callers in-process (the web UI) read the run back
            e.run_result, e.run_dir, e.run_exit_code = res, str(run_dir), ecode
        except AttributeError:
            pass
        log.event("run_end", result="INVALID", disposition=res["disposition"],
                  pull_reasons=res["pull_reasons"], exit_code=ecode, result_json="result.json")
        log.close()
        ctx.say(f"Run {run_id}: INVALID ({line})", f"  disposition: {_one_line(D.summary(res))}",
                f"  run folder: {run_dir}")
        raise
    return _run_done(ctx, cfg, run_id, run_dir, res, log, code)


def _settle_run(res: dict, recipe=None) -> dict:
    """A run's disposition on the ABORTED / refused / crashed paths: a recipe job's through
    the recipe (its result code follows: a pulled cycle never keeps the PASS code an earlier
    settle wrote), any other job's from its verdict."""
    if recipe is not None:
        from engine.recipes import runner
        return runner.settle(res, recipe)
    return D.settle(res)


def _settle_metric(res: dict, cfg) -> dict:
    """A metric job's disposition: its PASS within job.guard_band_pct of threshold_mm (the
    robust max |deviation| is held to it, one-sided) is a PULL."""
    tol = None
    if res.get("result") == "PASS" and res.get("threshold_mm") is not None:
        tol = [D.tolerance("deviation_mm.max", res["deviation_mm"]["max"], hi=res["threshold_mm"])]
    return D.settle(res, guard_band_pct=cfg.guard_band_pct, tolerances=tol)


def _recipe_mock_place(ctx, recipe, station_part: Optional[str] = None):
    """MOCK only: what the simulated human places for a recipe run: --mock-part when given,
    else the station config's mock_part, else the recipe's part.mock_scene
    (scene:<name>[:<variant>]), else the empty wall."""
    if not ctx.mock:
        return None
    spec = ctx.args.mock_part if getattr(ctx.args, "mock_part_given", False) else \
        (station_part or recipe.part.get("mock_scene") or "none")
    if not str(spec).startswith("scene:"):
        spec = _mock_part(str(spec))

    def place():
        old = ctx.args.mock_part
        ctx.args.mock_part = spec
        try:
            _mock_place_metric(ctx)
        finally:
            ctx.args.mock_part = old
    return place


def _recipe_summary(ctx, res: dict) -> None:
    for z in res.get("zones") or []:
        ctx.say(f"  zone {z['id']}: {z['verdict']}"
                + (f"  ({z['n_points']} points)" if z.get("n_points") is not None else ""))
        rows = []
        for ch in z["checks"]:
            k = ch.get("key")
            v = (ch.get("values") or {}).get(k) if k else None
            rows.append([ch["id"], ch["type"], ch["verdict"],
                         f"{k} {_f(v, 4)}" + (f" (limit {_f(ch.get('limit'), 4)})"
                                               if ch.get("limit") is not None else "")
                         if k else "-", _one_line(ch.get("message", ""))[:90]])
        ctx.table(["check", "type", "verdict", "value", "message"], rows, indent="    ")
    fb = res.get("feedback") or {}
    if fb.get("rendered"):
        drawn = [z["id"] for z in fb.get("zones", []) if z.get("drawn")]
        ctx.say(f"  feedback: {len(drawn)} zone(s) drawn, {len(fb.get('crosses', []))} "
                f"cross(es){', text ' + repr(fb['text']) if fb.get('text') else ''}: "
                f"{fb.get('image')}")
    elif fb.get("enabled"):
        ctx.say(f"  feedback: not projected ({_one_line(fb.get('refused') or '-')})")


def _run_recipe(ctx, cfg, recipe, p, log, res, run_dir, run_id) -> tuple:
    """job.type recipe: engine/recipes/runner (scan, zones, checks, verdict, overlay + DMD
    feedback image), result.json, then the projected verdict (unless --no-feedback)."""
    from engine.recipes import runner
    s = api._engine()
    if ctx.args.average is not None:                # --average, else the config's job.average
        recipe.scan["average"] = ctx.args.average
    label = cfg.part_label if (cfg.raw.get("part") or {}).get("label") else None
    if label is None:
        res["part"]["label"] = recipe.part.get("label") or recipe.name
    res["job"]["recipe"] = recipe.name
    state = runner.execute(s, recipe, p, res=res, log=log, run_dir=run_dir,
                           mock_place=_recipe_mock_place(ctx, recipe, cfg.mock_part),
                           label=label)
    runner.settle(res, recipe)                      # disposition + its result code
    code = D.exit_code(res)
    res["outputs"]["mesh"] = [o["path"] for o in ctx.new_outputs()
                              if o["path"].lower().endswith((".stl", ".obj", ".ply"))]
    s.check_cancel()                    # an ABORT up to here: ABORTED (cmd_run), not judged
    want = recipe.feedback["enabled"] if cfg.feedback is None else bool(cfg.feedback)
    if want and state.get("img") is not None and not ctx.args.no_feedback:
        ctx.say(f"  projecting the verdict on the part for {recipe.feedback['hold_s']:g} s "
                "(Ctrl+C / ABORT clears it)")
    out = runner.finish(s, p, recipe, run_dir, res, log, state,
                        project_feedback=bool(want and not ctx.args.no_feedback),
                        cancel=s.cancel, exit_code=code)
    dev = res["deviation_mm"]["max"]
    ctx.say(f"Run {run_id}: {res['result']}"
            + (f"  ({res['invalid_reason']})" if res["invalid_reason"] else ""))
    ctx.say(f"  disposition: {_one_line(D.summary(res))}")
    ctx.kv([["station", cfg.station_id], ["recipe", f"{recipe.name} (preset {p.name})"],
            ["headline", f"{res['deviation_mm']['method']}: {_u(dev, 'mm', 4)}"
             if dev is not None else "-"],
            ["result.json", str(out)], ["log", str(run_dir / "run.log.jsonl")]])
    _recipe_summary(ctx, res)
    ctx.warn(res["warnings"])
    return code, {"run_id": run_id, "run_dir": str(run_dir), "result": res}


def _run_done(ctx, cfg, run_id, run_dir, res, log, code) -> tuple:
    from ui import run_job as RJ
    res["outputs"]["mesh"] = [o["path"] for o in ctx.new_outputs()
                              if o["path"].lower().endswith((".stl", ".obj", ".ply"))]
    out = RJ.write_result(run_dir, res)
    log.event("run_end", result=res["result"], disposition=res["disposition"],
              pull_reasons=res["pull_reasons"], exit_code=code, result_json=out.name,
              invalid_reason=res["invalid_reason"])
    log.close()
    dev = res["deviation_mm"]["max"]
    ctx.say(f"Run {run_id}: {res['result']}"
            + (f"  ({res['invalid_reason']})" if res["invalid_reason"] else ""))
    ctx.say(f"  disposition: {_one_line(D.summary(res))}")
    ctx.kv([["station", cfg.station_id], ["part", f"{cfg.part_label} ({cfg.part_role})"],
            ["max deviation", _u(dev, "mm", 4) if dev is not None else "-"],
            ["threshold", _u(res["threshold_mm"], "mm", 3)
             if res["threshold_mm"] is not None else "-"],
            ["result.json", str(out)], ["log", str(run_dir / "run.log.jsonl")]])
    ctx.warn(res["warnings"])
    return code, {"run_id": run_id, "run_dir": str(run_dir), "result": res}


# ------------------------------------------------------------ operating modes (engine/modes/)

def _use_preset_arg(ctx) -> None:
    """`measure-run PRESET` / `quick-check PRESET`: the triangulation preset named as the
    command's own argument wins over --preset and the selected one."""
    name = getattr(ctx.args, "preset_name", None)
    if name:
        ctx.args.preset, ctx.args.preset_given = name, True
        ctx._preset = None


def _mode_runs_dir(ctx) -> Path:
    """Where a measure / quick-check run folder goes: --runs-dir, $SCANNER_RUNS_DIR, else
    <library>/runs."""
    a = ctx.args
    if getattr(a, "runs_dir", None):
        return Path(a.runs_dir).resolve()
    env = os.environ.get("SCANNER_RUNS_DIR")
    return Path(env).resolve() if env else Path(api.paths()["runs"]).resolve()


def _px_point(text: Optional[str], what: str):
    return None if not text else _numbers(text, 2, f"{what} x,y (camera px)", float)


def _fmt_value(v: dict) -> list:
    u = v.get("uncertainty") or {}
    nd = 4 if v["units"] == "mm" else 3
    return [v["label"], f"{_f(v['value'], nd)} {v['units']}",
            f"+- {_f(u.get('value'), nd)} {u.get('units', '')}".strip() if u else "-",
            (u.get("method") or "-") + (f" x{u['n_repeats']}" if (u.get("n_repeats") or 1) > 1
                                        else ""),
            str(v["n_points"]), "NEAR LIMIT" if v.get("near_resolution_limit") else ""]


def cmd_measure_run(ctx):
    """MEASURE mode: scan and report numbers (units, n points, uncertainty), no verdict."""
    a = ctx.args
    _use_preset_arg(ctx)
    if not a.what and not a.spec:
        raise CLIError("what to measure? --what flatness | step | distance | cylinder | endface "
                       "| dims | deviation (or --spec FILE)")
    zone = a.zone
    if a.zones:
        if zone:
            raise CLIError("give --zones (a painted zone set) or --zone (grid cells), not both")
        zone = _zones_text(a.zones)
    try:
        spec = api.measure_spec(
            a.what, a.spec, roi=_roi(a.roi), roi2=_roi(a.roi2), zone=zone, zone2=a.zone2,
            p1=_px_point(a.p1, "--p1"), p2=_px_point(a.p2, "--p2"), golden=a.golden,
            radius_mm=a.radius, axis=a.axis, min_height_mm=a.min_height,
            face_diameter_mm=a.diameter, n_faces=a.faces, repeats=a.repeats,
            scan_kind=a.scan_kind, steps=a.steps, periods=_periods(a.periods),
            average=a.average, name=a.name)
    except ValueError as e:
        raise CLIError(str(e)) from e
    if spec.preset and not a.preset_given:          # the spec's own preset, unless one is named
        a.preset = spec.preset
        ctx._preset = None
    ctx.header()
    p = ctx.preset()
    if spec.preset and spec.preset != p.name:
        ctx.say(f"  ! the spec was made for preset {spec.preset}; measuring with {p.name}")
    spec.preset = p.name                            # the preset it runs with (--save-spec)
    scans = [_find_metric(k) for k in a.scan] if a.scan else None
    if ctx.mock and scans is None:
        _mock_place_metric(ctx)                     # the simulated human places the part
    out = api.measure_run(preset=p, spec=spec, scans=scans, runs_dir=_mode_runs_dir(ctx),
                          label=a.label)
    doc, run_dir = out["result"], Path(out["run_dir"])
    if doc["status"] == "MEASURED":
        ctx.say(f"Measurement {out['run_id']}: MEASURED  ({spec.what}, {doc['repeats']} scan(s))")
        ctx.table(["value", "", "uncertainty", "from", "n points", ""],
                  [_fmt_value(v) for v in doc["values"]])
        for v in doc["values"]:
            for n in v.get("notes") or []:
                ctx.say(f"    {v['label']}: {_one_line(n)}")
    else:
        ctx.say(f"Measurement {out['run_id']}: NOT MEASURABLE ({doc['reason']}) - no number "
                "is given")
        ctx.say(f"  why: {_one_line(doc['message'])}")
        if doc.get("how_to_fix"):
            ctx.say(f"  fix: {_one_line(doc['how_to_fix'])}")
    ctx.say(f"  validity: {_one_line(doc['validity'])}")
    files = [["result.json", str(run_dir / "result.json")],
             ["values.csv", str(run_dir / doc["outputs"]["values_csv"])]]
    files += [[k, str(run_dir / doc["outputs"][k])] for k in ("overlay", "ply")
              if doc["outputs"].get(k)]
    files.append(["spec.json", str(run_dir / doc["outputs"]["spec"])])
    ctx.kv(files)
    if a.save_spec:
        dest = Path(a.save_spec)
        with open(dest, "x", encoding="utf-8") as f:       # never overwritten
            json.dump(spec.to_dict(), f, indent=2)
            f.write("\n")
        ctx.say(f"  spec saved: {dest}  (run it again: scan measure-run --spec {dest}; make it "
                "a check: scan measure-to-check)")
    ctx.warn(doc["warnings"])
    return out["exit_code"], {"run_id": out["run_id"], "run_dir": str(run_dir),
                              "status": doc["status"], "reason": doc["reason"],
                              "values": doc["values"], "result": doc}


def _parse_tols(items) -> dict:
    """--tol NAME=LO,HI (a band; either side may be empty) or NAME=HI (an upper limit)."""
    out = {}
    for it in items or []:
        key, sep, val = str(it).partition("=")
        parts = [s.strip() for s in val.split(",")]
        if not sep or not key.strip() or not 1 <= len(parts) <= 2 or not any(parts):
            raise CLIError(f"--tol {it!r}: write NAME=LO,HI (a band, e.g. step_height=19.5,20.5; "
                           "an empty side is open) or NAME=HI (an upper limit, e.g. "
                           "flatness_p2_p98=0.3)")
        nums = [None if s == "" else _number(s, f"--tol {key}") for s in parts]
        out[key.strip()] = nums[0] if len(nums) == 1 else nums
    return out


def cmd_measure_to_check(ctx):
    """A measurement spec + tolerances -> a check (a recipe / a single-job station config)."""
    a = ctx.args
    try:
        out = api.measure_to_check(a.spec, _parse_tols(a.tol), a.name, a.guard_band or 0.0)
    except ValueError as e:
        raise CLIError(str(e)) from e
    text = json.dumps(out["document"], indent=2)
    what = "recipe" if out["kind"] == "recipe" else "single-job station config (a quick check)"
    if a.out:
        dest = Path(a.out)
        with open(dest, "x", encoding="utf-8") as f:       # never overwritten
            f.write(text + "\n")
        ctx.say(f"Check written: {dest}  ({what})")
        if out["kind"] == "recipe":
            ctx.say(f"  run it: scan quick-check {out['document']['preset']} --check recipe "
                    f"--recipe {dest}   (or scan recipe-add NAME --from {dest}, then name it in "
                    "a profile of a station configuration)")
        else:
            ctx.say(f"  run it: scan run {dest}")
    else:
        ctx.say(f"The {what} (--out FILE writes it):", text)
    return EXIT_OK, {"kind": out["kind"], "document": out["document"],
                     "file": str(Path(a.out)) if a.out else None, "run_with": out["run_with"]}


def cmd_quick_check(ctx):
    """QUICK CHECK mode: a triangulation preset + ONE check -> CONTINUE / REJECT / PULL. The
    minimal single-job station config is generated (engine/modes/quick.py) and run by the
    station job runner, exactly as `scan run` would."""
    from ui import run_job as RJ
    a = ctx.args
    _use_preset_arg(ctx)
    preset = a.preset if a.preset_given else None
    if preset is None and a.check != "recipe":      # the selected preset (refused when gone)
        preset = ctx.preset().name
    try:
        doc = api.quick_check_config(
            a.check, preset, threshold_mm=a.threshold, roi=_roi(a.roi), golden=a.golden,
            recipe=a.recipe, guard_band_pct=a.guard_band or 0.0, runs_dir=_mode_runs_dir(ctx),
            zone=a.zone, scan=a.scan_kind or "fast")
        if a.recipe and a.recipe.lower().endswith((".json", ".yaml", ".yml")):
            doc["job"]["recipe"] = str(Path(a.recipe).resolve())
        cfg = RJ.config_from_doc(doc, Path.cwd())
    except (ValueError, RJ.StationConfigError) as e:
        raise CLIError(str(e)) from e
    return _run_job(ctx, cfg, quick=doc)


def _add_globals(p: argparse.ArgumentParser, sub: bool) -> None:
    """Globals work before OR after the command. Sub-parsers use SUPPRESS so
    an option given before the command is not reset by the sub-parser."""
    d = (lambda v: argparse.SUPPRESS) if sub else (lambda v: v)
    p.add_argument("--mock", action="store_true", default=d(False),
                   help="simulated rig (no hardware); the mock bench stands in for the human")
    p.add_argument("--library", metavar="PATH", default=d(None),
                   help="library root (default: the engine's, or $SWISS_SCANNER_LIBRARY)")
    p.add_argument("--preset", metavar="NAME", default=d(DEFAULT_PRESET),
                   help=f"rig preset (default {DEFAULT_PRESET})")
    p.add_argument("--average", metavar="N", type=_positive_int, default=d(None),
                   help="frames averaged per pattern for THIS run (the preset file is unchanged)")
    p.add_argument("--mock-part", metavar="SPEC", type=_mock_part, default=d("validation"),
                   help="mock only: what the simulated human places for scan / wizard / "
                        "check-presence: validation | none | H,W,L[,albedo] (mm; albedo 0..1) | "
                        "cylinder[:R,H,dent_mm] (a standing cylinder, default 24.05,90,1.5) | "
                        "qr:TEXT (a QR label, for codes) | scene:NAME[:VARIANT] (a recipe "
                        "scene: tube, endfaces, box_lid, plate)")
    p.add_argument("--json", action="store_true", default=d(False),
                   help="print one JSON document (for scripts and AI agents)")
    p.add_argument("--interactive", action="store_true", default=d(False),
                   help="allow questions on the terminal (wizard; delete without --yes; "
                        "teach-presence without --empty-confirmed). Without it no command "
                        "ever waits for input: it is refused (exit 2) and says which "
                        "argument answers the question")
    p.add_argument("--prompt-timeout", metavar="S", type=_prompt_timeout, default=d(None),
                   help=f"with --interactive: a question gives up after S seconds and the "
                        f"command stops, nothing done (default {PROMPT_TIMEOUT_S:g} s for a "
                        f"y/N, {WIZARD_TIMEOUT_S:g} s per wizard step; at most "
                        f"{PROMPT_TIMEOUT_MAX_S:g})")
    p.add_argument("--debug", action="store_true", default=d(False),
                   help="show full tracebacks on errors")


EPILOG = """\
typical flow:
  scan signal                      is the setup good? (gc_conf, valid %, focus)
  scan base                        empty scene -> base (the datum)
  scan scan part --known 106.95,56.5,21.9
  scan measure part                H/W/L (px until calibrated, then mm)
  scan calibrate part              fit Z/XY from known dims at this geometry
  scan wizard --interactive        guided: base -> part -> measure -> inspect -> 3D (asks
                                   at every physical step; needs --interactive)

zones ("scan the zone the part is in"):
  scan map                         camera<->projector map (once per rig position)
  scan grid                        project + photograph a lettered grid (A1, B2, ...)
  scan scan part --zone C4:D5      light and measure only those cells

painted zones (draw them ON THE PART: the projector is a second monitor, the mouse cursor
is projected; zones are projector px, valid at this rig placement):
  scan zone-paint --name lid       drag = rectangle, E ellipse, P polygon (clicks; Enter or
                                   double-click closes); click selects, drag inside moves,
                                   handles resize, wheel / [ ] rotate, N names, Del removes,
                                   Z undo, G grid; SPACE saves the set, ESC cancels
  scan zones [lid]                 the saved sets / one set's zones (--recipe, --mask-out)
  scan zone-show lid               project it, photograph, where each zone lands
  scan scan-fast part --zones lid[:Z1,Z3]   light + decode only those zones; points per zone
  scan measure-run P --what flatness --zones lid:Z1   the zone as the measured region
  recipes: a zone {"id": "lid", "ref": "zoneset:lid/Z1", "checks": [...]}, scan.zone
      "zoneset:lid"; mock: scan --mock zone-paint --name lid --events demo

virtual sensors (the DMD draws the spot, the camera is the receiver):
  scan spot-size 6.5 [--measured K]      6.5 mm spot -> DMD px (cols x rows), nominal vs
                                         measured (K = measured mm / drawn columns)
  scan spot H8 --diameter 6.5            project + measure a spot (or dmd:COL,ROW, cam:X,Y)
  scan teach-presence pe H8 --diameter 6.5 [--method bgs] --empty-confirmed
                                         teach on the EMPTY field: a person clears it first;
                                         refused (exit 2) without --empty-confirmed - it
                                         never waits for an answer (--interactive asks)
  scan check-presence pe                 exit 0 PRESENT, 1 ABSENT or INDETERMINATE
  scan teach-align fixture <scan>        reference pose of the part in a scan
  scan check-align fixture <scan>        exit 0 ALIGNED, 1 NOT_ALIGNED or INDETERMINATE
  scan sensors                           taught sensors;  delete <name> trashes one

3D meshes, cylinders, surface maps (work on a stored scan; mm only when calibrated):
  scan mesh <scan> [--kind relief|base|cylinder] [--fmt stl|obj] [--stride 2] [--radius 24.05]
  scan unwrap <scan> [--radius MM]       cylinder fit: seen arc, end-face height, dent heatmap
  scan dents <scan> [--trigger MM]       exit 0 PASS, 1 FAIL or INDETERMINATE (open arc only)
  scan maps <scan> [--spectral ID --light ID --golden ID]   height/cavity/normal/... PNGs
  scan checker [--square 32 --shifts 1]  checker corners + local Jacobian (stretch, contrast)

metric (mm from a stereo calibration of the same geometry; refused otherwise):
  scan charuco [--square 25]             printable board (print at 100 %, caliper a square)
  scan parity                            diamond DMD row parity (bench step B3)
  scan calib-pose [--set NAME]           one board pose (repeat 12-20x, tilt +-30 deg); prints
                                         projector corners k/n (< 60 %: retried at coarser
                                         fringes 16,160,1280 / 24,192,1280 unless --periods;
                                         one period set per pose set); exit 1 not accepted
  scan calibrate-stereo --square MM      camera + projector + stereo; RMS per pose / axis and
                                         which poses gave projector points. REJECTED, nothing
                                         saved, exit 1: degenerate projector fit, stereo RMS
                                         > 2 px, or too few poses with projector points
  scan calib-pose --fixture SPEC --set fx   one fixture placement = one view per face
                                         (3 placements; docs/CALIBRATION_FIXTURE.md)
  scan calibrate-stereo --set fx --fixture SPEC   + rigid-fixture refinement, face-angle check
  scan accuracy-block --block SPEC --measured JSON   printed check block: exit 0 CONTINUE,
                                         1 REJECT (recalibrate), 5 PULL
  scan scan-metric [label]               both-axis scan -> point cloud in mm (+ .ply)
  scan cyldev <metric>                   cylinder fit -> deviation map + dents (mm)
  scan accuracy cylinder --diameter 29.84 [--repeats 5]   exit 0 GO, 1 NO-GO
  scan precision --scan <metric>         predicted sigma_z before scanning
  scan metric-mesh <metric>              STL / OBJ in mm, open surface only
  mock: scan --mock calibrate-stereo --mock-poses 12   (or --mock-fixture 3)

dents on shiny cylinders (needs a stereo calibration; the null calibration is the good part):
  scan --preset DARK dent-scan good1 --zone H9:I12 --short-preset SHORT   a KNOWN-GOOD part
  scan dent-null good1 --diameter 21.05     finders' maxima on the good part (per fixture)
  scan --preset DARK dent-scan cell7 --zone H9:I12 --short-preset SHORT
  scan dent-detect cell7                    exit 0 CLEAR, 1 FLAG, 4 detector error (+ PNG, json)
  scan dentsets                             captured dent sets (legacy folders work too)

end faces (a cylinder lying with its circular end face toward the camera; stereo calibration first):
  scan --preset P face-locate --zone H10:I12 --diameter 19.5
                                            locate scan (ray-plane, flood pair kept; scanned
                                            again at a longer / shorter exposure when the face
                                            is dark / clipped) -> face plane, rim circle,
                                            camera -> DMD affine (or face-locate <metric scan>:
                                            it needs x AND y decoded - scan-fast --axis xy or
                                            scan-metric; an x-only scan-fast is refused).
                                            --diameter = the bright flat's own diameter (a 21
                                            mm class cylinder with a rounded rim: its 19.5 mm
                                            flat); --outer-diameter 21.05 = the part's outer
                                            circle when that rim returns depth points (else
                                            found from the rim). The flood outline is held to
                                            --diameter: > 4 % off (--size-tol) = size
                                            mismatch, exit 5 PULL (calibration or lens
                                            suspect; nothing is rescaled)
  scan face-aim --hold 30                   ring + cross on the face: the placement guide
  scan face-aim --check                     exit 0 ON_FACE, 1 OFF (the part moved: re-locate)
  scan face-intensity                       projection.duty for a ~200 DN face (never the LEDs)
  scan face-scan c1                         only the face lit: 3 repeats + polar set, at a
                                            frame-locked camera exposure chosen for THIS
                                            seating (auto exposure: face p95 ~170 DN, < 0.5 %
                                            clipped, 16.7-100 ms; the default when the
                                            preset's exposure is whole video frames; exit 5
                                            when none fits). --exposure-us N forces one,
                                            --no-auto-exposure keeps the preset's;
                                            --auto-intensity tunes the duty first
  scan face-profile c1                      height map, radial, sections, flatness (+ PNG/CSV/STL),
                                            centred on the flood outline of the flat
  scan face-golden good1 good2 good3 good4  golden from GOOD placements: MF-CFAR + reflectance
                                            FIR (default) and band, leave-one-out, synthetic
                                            dents, the rim outline's per-angle spread. Every
                                            radius derives from the face diameter (inspect
                                            0.926 R, rim 0.855 R); --inspect-r / --rim-r set them
  scan face-check c7                        exit 0 PASS, 1 FAIL, 5 PULL (quality inspection:
                                            INCOMPLETE, a size mismatch, a rim outline in its
                                            guard band or not judged all the way round, a
                                            profile off its outline's centre). Dents: depth
                                            (the 3 x 3 centre average),
                                            deepest point, width at half depth, z; edge dents:
                                            zone rim_outline (notch / bulge of the flood
                                            outline, 0.4 mm over 1.5 mm of rim); overlay PNG
  scan faces                                face targets, face scans, golden faces

HDMI fast zone scan + station (cylinders with end faces toward the camera + a ribbed tube edge):
  scan --preset P scan-fast st1 --zone F10:J12   columns only, 4 steps x 8/80/1280 + floods,
                                            average 2 (--average N; < 2 is raised: HDMI
                                            flicker), --hold 0.12 s; keeps its flood pair;
                                            --axis xy for a scan face-locate can use
  scan station st1                          cylinders C1..Cn: heights, Cj-Ci, spacing; tube edge
                                            band, trapezoid wave, zones T1..Tn (+ PNG/JSON/CSV)
  scan station-golden st1 st2 --cell-tol 0.15 --ct-tol 0.3 [--band T3=-5,-4]
  scan station-check st7                    exit 0 PASS, 1 FAIL (triggers named), 5 PULL
                                            (INCOMPLETE, ambiguous end faces), 2 refused (e.g.
                                            an end face not found), 4 crashed
  scan stations                             station goldens

Surfer atlas (regions become zones atlas:<name>/<region>, e.g. scan part --zone atlas:part/part):
  scan atlas-import <path.json> <name>   copy an atlas into the library (source only read)
  scan atlases                           imported atlases + their regions
  scan flood <region>                    light only the region, photograph (exit 0 lands)
  scan antiglare <region> [--carriers fx,fy;fx,fy]   carrier lock (exit 0 locked)
  scan multiorient <region>              v/h/d45/d135 fringes fused by modulation
  scan flash-pack [--axes x|y] [--video-port]   DLPC350 pattern pack (files only; a human
                                         enters pattern mode; docs/FLASH_PATTERN_MODE.md)
  scan fastscan-pack [--zone-ring NPY]... [--band-cam NPY --band-field NPZ] [--study]
                                         1-bit soft-sync pack, full + zone images (files only;
                                         docs/FLASH_FASTSCAN_4500.md)
  scan scan-flash [label] --pack DIR [--variant zone] [--roi X0,Y0,X1,Y1 | --zone Z]
                                         fast metric scan while a HUMAN plays the pack in the
                                         4500 MKII GUI: camera only, frames sorted in software
                                         (exit 2: the loop is not playing / not in order);
                                         --dry-run prints the timing plan; --fine-prefilter 1
                                         low-passes the fine fringes (sharp projector)
  scan trigger-test [--line Line0] [--seconds 5] [--expect-period-ms P]
                                         EXPERIMENTAL: count TRIG_OUT_1 pulses at the camera
                                         (camera only; exit 0 OK, 1 NO PULSES / RATE MISMATCH,
                                         4 an error, e.g. the camera could not be set up)
  scan scan-trigger <label> --pack DIR [--variant full] [--line Line0] [--repeats N]
          --pattern-mode-set-by-human    EXPERIMENTAL hardware-triggered scan, one frame per
                                         pattern (docs/TRIGGER_FULLSPEED_TEST.md)
  scan sort-flash <recording.npz> [--decode]   offline: re-sort a saved recording
                                         (docs/FLASH_SOFTSYNC.md)

rig check (has the camera / projector been nudged? also the operator UI's HOME):
  scan rig-check [--set-reference] [--tol 0.5]   exit 0 OK / reference set, 1 MOVED or UNKNOWN
  scan rig-check --rehome                 a small CAMERA move only: a NEW stereo calibration
                                         with the camera's rotation taken out (R1 = R0 @ Rd,
                                         t unchanged; flagged re-homed, the original kept);
                                         exit 0 re-homed / nothing to do, 1 refused (projector
                                         moved, > ~5 px, poor match). Rotation only: absolute
                                         depth may keep a small offset - recalibrate when you can

flow-ready job (one station config in, one run folder out; see stations/):
  inspect / compare: base-relief depth calibration;  metric: stereo calibration (calibrate-stereo)
  scan --mock run stations/example_cell_station.json
  scan run stations/<station>.json --operator-confirmed
                                         result.json + run.log.jsonl in <runs_dir>/<run_id>/;
                                         exit 0 CONTINUE (PASS), 1 REJECT (FAIL), 5 PULL
                                         (quality inspection: INVALID, guard band, ...),
                                         2 refused, 4 crashed (both PULL in result.json)

recipes (what the DLP checks to verify a work state: preset -> zones -> checks -> verdict
painted on the part):
  scan recipe-list                       built-in (recipes/, read-only) + user recipes
  scan recipe-checks                     check types: presence, position, height, flatness,
                                         defect, edge, code (+ plugins)
  scan recipe-golden <recipe>            a GOOD part as the recipe's golden scan
  scan --mock recipe-run box_lid --mock-part scene:box_lid:ajar
  scan recipe-run <recipe> --operator-confirmed    exit 0 CONTINUE, 1 REJECT, 5 PULL
                                         (quality inspection), 2 refused, 4 crashed; red on
                                         the part at the fault, an amber frame for PULL
  scan recipe-add|recipe-select|recipe-delete|recipe-show|recipe-validate
  station configs: job.type recipe + job.recipe NAME (a single job verified by the recipe),
      or profiles + a flow (each work state verifies with a recipe of the active profile)

operating modes (the triangulation preset - rig.mount fixed | mobile - is its own layer):
  scan measure-run PRESET --what flatness [--roi X0,Y0,X1,Y1] [--repeats N]
                                         MEASURE: an instrument - numbers with units, n points
                                         and uncertainty; no tolerance, no verdict. --what
                                         flatness | step (--roi --roi2) | distance (--p1 --p2,
                                         or --p1 + --roi: point to plane) | cylinder | endface
                                         (--diameter) | dims | deviation (--golden); a new run
                                         folder: result.json, values.csv, overlay.png, scan.ply,
                                         spec.json. Exit 0 MEASURED, 5 NOT MEASURABLE (no
                                         calibration, a mobile rig not verified ...: no number,
                                         never a px as mm). Not `measure` (H/W/L of a relief scan)
  scan measure-to-check SPEC --tol NAME=LO,HI [--out FILE]
                                         a measurement spec + tolerances -> a check (a recipe)
  scan quick-check PRESET --check flatness --threshold MM [--roi ...] [--guard-band PCT]
  scan quick-check PRESET --check golden --golden ID --threshold MM
  scan quick-check [PRESET] --check recipe --recipe NAME
                                         QUICK CHECK: one check, no sensors, no permissions;
                                         exit 0 CONTINUE, 1 REJECT, 5 PULL (2 refused, 4 crashed)
  scan run stations/<config>.json        STATION: configuration -> profile -> work states
  scan rig-session [--new]               a MOBILE rig (preset-add --set rig.mount=mobile): is it
                                         verified for mm? --new: it was moved / powered up now

station configuration -> profiles -> work states (stations/README.md):
  scan flow-show stations/example_tube_line.json     parameters, profiles (the profile library
                                         set), work states (the flow: with the check that
                                         observes each sensor signal), permissions; validates
  scan run stations/example_tube_line.json [--profile NAME] [--input SIGNAL=true|false ...]
                                         one whole cycle in line: every work state is verified
                                         (the DLP confirms or corrects the sensor claims), exit
                                         0 CONTINUE (every required work state CONFIRMED), 1
                                         REJECT (a kickout), 5 PULL; MOCK: --mock-case NAME

library: bases | scans | calibrations | calib-poses | readings | materials | banks | maps |
         sensors | outputs;  bank-remove <scan> --bank NAME drops one bank member.
         delete <id> --yes [--reason WHY] moves one entry to the library trash (bases, scans,
         calibrations, calibration poses, rig references, parity measurements, ...).
no command waits for input: answers are arguments (--yes, --empty-confirmed); only with
--interactive does a command ask, and a question gives up after --prompt-timeout seconds.
mock: --mock-part H,W,L[,albedo] is what the simulated human places for scan,
      wizard and check-presence (the bench is albedo 0.6; 'none' = empty);
      --mock-part cylinder[:R,H,dent_mm] places a standing cylinder; qr:TEXT a QR
      label for codes; scene:<name>[:<variant>] a recipe scene (tube[:misaligned],
      endfaces[:dent|proud], box_lid[:ajar|missing], plate[:bump|burr], tube_long /
      tube_short[:absent|misaligned|dent]). A station configuration with a flow names its
      simulated line as cases: run CONFIG --mock-case NAME (the scene + the sensor claims).

ids: a full id, a label (newest wins), 'latest', or any unique piece of the id.
exit codes: 0 ok | 1 not ok (verdict/signal/cancel/unexpected) | 2 refused (rule,
geometry, safety, config, rig busy, bad input) | 3 mode not available yet |
4 dent-detect's detector failed (its unexpected errors are not 1 = FLAG) |
5 PULL (quality inspection: run / recipe-run / quick-check INVALID or in doubt,
station-check / face-check INCOMPLETE); measure-run: NOT MEASURABLE (no number given).
--json prints one JSON document ({"ok", "exit_code", ...}) for scripts and agents.
"""


def build_parser() -> argparse.ArgumentParser:
    # allow_abbrev=False everywhere: an abbreviated option (--pres for --preset) is refused
    # instead of half-working (main() only sees the spelled-out --preset / --mock-part)
    ap = argparse.ArgumentParser(
        prog="scan", formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Swiss Army Scanner: DMD structured-light scanner, command line.",
        epilog=EPILOG, allow_abbrev=False)
    _add_globals(ap, sub=False)
    common = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    _add_globals(common, sub=True)
    sp = ap.add_subparsers(dest="command", metavar="<command>")

    def cmd(name, func, help_, mode=None):
        p = sp.add_parser(name, parents=[common], help=help_,
                          description=help_.replace("%%", "%"), allow_abbrev=False)
        p.set_defaults(func=func, mode=mode or name)
        return p

    cmd("presets", cmd_presets, "list rig presets: built-in + user (tier, geometry, calibration)")
    p = cmd("preset-add", cmd_preset_add, "add a user preset (part recipe): a copy of another "
            "preset with fields overridden", "preset add")
    p.add_argument("name", help="new user preset name [A-Za-z0-9_.-]")
    p.add_argument("--from", dest="base", required=True, metavar="BASE",
                   help="preset to copy (built-in or user)")
    p.add_argument("--set", action="append", metavar="KEY=VALUE",
                   help="override a field, e.g. camera.exposure_us=33333, projection.duty=0.8, "
                        "projection.invert=false (repeatable; values parse as JSON when they can)")
    p.add_argument("--geometry", metavar="ID", help="new geometry_id (a new rig geometry: starts "
                   "UNCALIBRATED)")
    p.add_argument("--desc", metavar="TEXT", help="description (what part / line it is for)")
    p.add_argument("--replace", action="store_true",
                   help="replace an existing USER preset of this name (the old one goes to trash/)")
    p = cmd("preset-select", cmd_preset_select, "select the library's active preset (used when "
            "no --preset is given)", "preset select")
    p.add_argument("name")
    p = cmd("preset-delete", cmd_preset_delete, "move a user preset to the library trash "
            "(built-ins are read-only)", "preset delete")
    p.add_argument("name")
    p.add_argument("--reason", required=True, help="why (recorded with the trashed preset)")
    p.add_argument("--force", action="store_true",
                   help="delete even when a station config, a recipe (built-in or user) or a "
                   "recipe station config uses it (never the selected recipe's preset)")
    p.add_argument("--stations", metavar="DIR", help="station configs folder checked for "
                   "references (default: the repo's stations/)")
    p = cmd("preset-show", cmd_preset_show, "show one preset (tier, calibration, JSON)",
            "preset show")
    p.add_argument("name")
    cmd("recipe-list", cmd_recipes, "list recipes (the checks of a work state): built-in + user (tier, "
                                    "preset, zones, checks, golden)")
    p = cmd("recipe-show", cmd_recipe_show, "show one recipe (tier, preset, JSON)", "recipe show")
    p.add_argument("name")
    cmd("recipe-checks", cmd_recipe_checks, "the check types a recipe can use (params, units)")
    p = cmd("recipe-validate", cmd_recipe_validate, "validate a recipe (name or file) without "
            "running it: exit 0 valid, 2 with every problem listed", "recipe validation")
    p.add_argument("recipe", help="recipe name or a .json / .yaml file")
    p = cmd("recipe-add", cmd_recipe_add, "add a user recipe: a copy of another recipe (or a "
            "recipe file) with fields overridden", "recipe add")
    p.add_argument("name", help="new user recipe name [A-Za-z0-9_.-]")
    p.add_argument("--from", dest="base", required=True, metavar="BASE",
                   help="recipe to copy (built-in / user name, or a .json / .yaml file)")
    p.add_argument("--set", action="append", metavar="KEY=VALUE",
                   help="override a field, e.g. feedback.hold_s=10, "
                        "zones.0.checks.0.params.tol_mm=0.4, golden.scan=my_good_part "
                        "(repeatable; values parse as JSON when they can)")
    p.add_argument("--desc", metavar="TEXT", help="description (what part / line it is for)")
    p.add_argument("--replace", action="store_true",
                   help="replace an existing USER recipe of this name (the old one goes to trash/)")
    p = cmd("recipe-select", cmd_recipe_select, "select the library's recipe (the default of "
            "recipe-run)", "recipe select")
    p.add_argument("name")
    p = cmd("recipe-delete", cmd_recipe_delete, "move a user recipe to the library trash "
            "(built-ins are read-only)", "recipe delete")
    p.add_argument("name")
    p.add_argument("--reason", required=True, help="why (recorded with the trashed recipe)")
    p.add_argument("--force", action="store_true",
                   help="delete even when a station config's job.recipe names it")
    p.add_argument("--stations", metavar="DIR", help="station configs folder checked for "
                   "references (default: the repo's stations/)")
    p = cmd("recipe-run", cmd_recipe_run, "run a recipe: scan, zones, checks, verdict, the "
            "verdict projected on the part; a new run folder (exit 0 PASS, 1 FAIL, 2 refused, "
            "4 crashed, 5 INVALID)", "the recipe run")
    p.add_argument("recipe", nargs="?", help="recipe name or file (default: the selected recipe)")
    p.add_argument("--scan", metavar="ID", help="check a stored metric scan (id / label / latest) "
                   "instead of capturing")
    p.add_argument("--no-feedback", action="store_true", help="do not project the verdict")
    p.add_argument("--hold", type=float, metavar="S", help="feedback hold, s (default the "
                   "recipe's feedback.hold_s)")
    p.add_argument("--runs-dir", metavar="DIR", help="where the run folder goes (default "
                   "$SCANNER_RUNS_DIR, the recipe's outputs.runs_dir, else <library>/runs)")
    p.add_argument("--label", help="scan label (default the recipe's part.label)")
    p.add_argument("--operator-confirmed", action="store_true",
                   help="a human confirmed the physical setup (required on the real rig)")
    p.add_argument("--pattern-mode-set-by-human", action="store_true",
                   help="scan.kind trigger only: a HUMAN put the projector in pattern mode")
    p.set_defaults(unexpected_exit=EXIT_ERROR)          # a crash (4) is not a FAIL verdict (1)
    p = cmd("recipe-golden", cmd_recipe_golden, "scan a GOOD part with a recipe's scan settings, "
            "recorded as the recipe's golden (labelled the recipe's golden.scan; golden.scan "
            "latest = the newest golden of this recipe)", "the recipe golden")
    p.add_argument("recipe", nargs="?", help="recipe name or file (default: the selected recipe)")
    p.add_argument("--label", help="scan label (default the recipe's golden.scan)")
    cmd("paths", cmd_paths, "where the library, output bin and presets live")
    cmd("signal", cmd_signal, "live signal check: gc_conf, valid %%, focus, fringe axis",
        "the signal check")
    p = cmd("tune", cmd_tune, "scan assistant: sweep exposure/duty, recommend settings",
            "the scan assistant (tune)")
    p.add_argument("--exposures", type=_list_of(int, "--exposures"), metavar="US,US,..",
                   help="exposures to try, microseconds")
    p.add_argument("--duties", type=_list_of(float, "--duties"), metavar="D,D,..",
                   help="DMD duties to try, 0..1")
    p = cmd("gamma", cmd_gamma, "measure the projector's gray response (gamma)",
            "gamma measurement")
    p.add_argument("--levels", type=_positive_int, default=12, help="gray levels (default 12)")
    p = cmd("base", cmd_base, "capture the empty-scene base (the datum)", "base capture")
    p.add_argument("--label", default="empty")
    p = cmd("bases", cmd_bases, "list bases (default one for scan is marked)")
    p.add_argument("--geometry", help="only this geometry id")
    p = cmd("reject-base", cmd_reject_base, "mark a base unusable (kept, never picked again)")
    p.add_argument("id")
    p.add_argument("reason", nargs="+")
    p = cmd("delete", cmd_delete, "move a base / scan / calibration / calibration pose / rig "
                                  "reference / parity measurement / reading / map / bank / "
                                  "material / sensor to trash/ (recoverable, never destroyed)")
    p.add_argument("id", help="entry id or label (a calibration pose: its id, see calib-poses)")
    p.add_argument("--reason", default="", help="why (recorded with the trashed entry)")
    p.add_argument("--yes", "-y", action="store_true",
                   help="confirm: required (the command never waits for input) unless "
                        "--interactive, which asks")
    p = cmd("scan", cmd_scan, "scan a part against a base (relief, stored in objects/)",
            "object scanning")
    p.add_argument("label")
    p.add_argument("--known", metavar="H,W,L", help="true dimensions in mm (any subset)")
    p.add_argument("--zone", metavar="ZONE", help="grid zone: C4 | C4:D5 | B2,E5 (see `grid`) or atlas:<name>/<region>")
    p.add_argument("--base", metavar="ID", help="base to use (default: newest usable)")
    p = cmd("scans", cmd_scans, "list scans")
    p.add_argument("--geometry", help="only this geometry id")
    p = cmd("reconstruct", cmd_reconstruct, "export a scan as a 3D point cloud",
            "point-cloud export")
    p.add_argument("id")
    p.add_argument("--fmt", choices=("ply", "xyz"), default="ply")
    p = cmd("measure", cmd_measure, "H/W/L of a stored relief scan (mm when calibrated, else "
                                    "px). Measure MODE - a scan + numbers with uncertainty - is "
                                    "`measure-run`")
    p.add_argument("id")
    p = cmd("calibrate", cmd_calibrate, "fit Z/XY calibration from scans with known dims",
            "calibration")
    p.add_argument("ids", nargs="+", metavar="id")
    p = cmd("inspect", cmd_inspect, "defect blobs + PASS/FAIL vs the preset trigger",
            "inspection")
    p.add_argument("id")
    p.add_argument("--sigma", type=float, default=6.0, help="detection threshold, x noise sigma")
    p.add_argument("--min-area", type=_positive_int, default=50, help="min blob area, camera px")
    p.add_argument("--whole-field", action="store_true", help="inspect everything, not the part")
    p = cmd("compare", cmd_compare, "deviation vs a golden scan + PASS/FAIL", "golden compare")
    p.add_argument("id")
    p.add_argument("golden")
    p.add_argument("--fail-mm", type=float, help="tolerance in mm (default: preset ch_fail_mm)")
    p.add_argument("--fail-px", type=float, help="tolerance in relief px (uncalibrated)")
    p = cmd("profile", cmd_profile, "height profile along a line", "the 2D profile")
    p.add_argument("id")
    p.add_argument("--p0", metavar="X,Y", help="line start, camera px")
    p.add_argument("--p1", metavar="X,Y", help="line end, camera px")
    p.add_argument("--axis", choices=("auto", "x", "y"), default="auto")
    p.add_argument("--width", type=_positive_int, default=5, help="averaging width, px")
    p = cmd("bank-add", cmd_bank_add, "add a known-good scan to a good memory bank",
            "the good memory bank")
    p.add_argument("id")
    p.add_argument("--bank", default="default")
    p = cmd("bank-check", cmd_bank_check, "flag anything not seen in the bank's good parts",
            "the good memory bank")
    p.add_argument("id")
    p.add_argument("--bank", default="default")
    p.add_argument("--k", type=float, default=4.0, help="threshold in bank sigmas")
    p.add_argument("--min-area", type=_positive_int, default=25, help="min anomaly, camera px")
    p = cmd("bank-remove", cmd_bank_remove, "drop one scan from a good memory bank (the scan "
                                            "is kept; append-only)", "the good memory bank")
    p.add_argument("id")
    p.add_argument("--bank", default="default")
    p.add_argument("--reason", default="removed from the command line")
    p = cmd("banks", cmd_banks, "list good memory banks and their members")
    p.add_argument("--geometry", help="only this geometry id")
    p = cmd("codes", cmd_codes, "read barcodes / QR codes under DMD light", "code reading")
    p.add_argument("--label", default="codes")
    p.add_argument("--anti-glare", dest="anti_glare", action="store_true",
                   help="EXPERIMENTAL: if nothing reads, shoot a many-level flood bracket, fuse "
                        "it to HDR and remove the specular veil of a glossy label, then decode")
    p.add_argument("--strip-cols", dest="strip_cols", metavar="C0,C1",
                   help="with --anti-glare: also sweep narrow DMD column strips over these DMD "
                        "columns (the code region) to drop bloom")
    p = cmd("spectral", cmd_spectral, "R/G/B LED reflectance capture", "spectral mode")
    p.add_argument("--label", default="spectral")
    p = cmd("learn", cmd_learn, "teach a material from a camera ROI of a spectral reading",
            "material learning")
    p.add_argument("reading_id")
    p.add_argument("name")
    p.add_argument("roi", metavar="x0,y0,x1,y1")
    p = cmd("classify", cmd_classify, "per-pixel material map of a spectral reading",
            "material classification")
    p.add_argument("reading_id")
    p.add_argument("--materials", type=_list_of(str, "--materials"), metavar="A,B,..",
                   help="only these materials (default: all learned)")
    p.add_argument("--max-distance", type=float, default=3.0)
    p = cmd("light", cmd_light, "direct vs global (inter-reflection) light",
            "direct/global light separation")
    p.add_argument("--label", default="light")
    p.add_argument("--square", type=_positive_int, default=16, help="checker square, DMD px")
    p.add_argument("--shifts", type=_positive_int, default=8)
    p = cmd("map", cmd_map, "capture the camera<->projector map (needed for zone cuts)",
            "the projector map")
    p.add_argument("--label", default="map")
    p = cmd("grid", cmd_grid, "project + photograph the lettered zone grid",
            "the projected grid")
    p.add_argument("--spacing", type=_positive_int, default=96, help="cell size, DMD px")
    p.add_argument("--label", default="grid")
    p.add_argument("--hold", type=float, default=0.0, help="keep the grid up N s to look at it")
    p = cmd("zone-paint", cmd_zone_paint,
            "paint zones ON THE PART: the projector window takes the mouse (its cursor is "
            "projected), drag rectangles / ellipses (E) / polygons (P) over the areas, move, "
            "resize, rotate, name (N); Space confirms and saves the zone set, Esc cancels. "
            "Exit 0 saved, 1 not saved, 2 refused (rig busy, the name exists ...)",
            "the zone painter")
    p.add_argument("--name", required=True, metavar="SET",
                   help="the zone set to save [A-Za-z0-9_.-] (at this preset's geometry)")
    p.add_argument("--from", dest="source", metavar="SET",
                   help="start from the zones of a saved set (edit it; with the same --name "
                        "add --replace)")
    p.add_argument("--replace", action="store_true",
                   help="replace an existing set of this name (the old one goes to trash/)")
    p.add_argument("--events", metavar="FILE|demo",
                   help="MOCK ONLY: the scripted session, a .json list of painter events "
                        "({\"type\": \"drag\", \"from\": [c, r], \"to\": [c, r]}, {\"type\": "
                        "\"key\", \"key\": \"space\"} ...), or `demo` (a built-in one)")
    p.add_argument("--labels", metavar="A,B,..",
                   help="the names N cycles through (default part, datum, end face, tube, box, "
                        "edge, fixture)")
    p.add_argument("--polarity", choices=("measure", "normal", "inverted"), default="measure",
                   help="how bright is sent: measure (default: raw black vs raw white, one lit "
                        "field for one hold before the drawing) | normal | inverted")
    p.add_argument("--note", metavar="TEXT", help="a note stored with the set")
    p.add_argument("--idle-timeout", type=float, default=600.0, metavar="S",
                   help="give the rig back after S seconds without input, nothing saved "
                        "(default 600)")
    p = cmd("zones", cmd_zones, "list the painted zone sets; zones SET[:Z1,Z3] shows one (its "
                                "zones in projector px), --recipe prints it as a recipe's "
                                "zones, --mask-out writes its scan mask", "the zone sets")
    p.add_argument("set", nargs="?", metavar="SET[:IDS]", help="a zone set (at this preset's "
                                                               "geometry), optionally some zones")
    p.add_argument("--geometry", metavar="ID", help="listing: only this geometry id")
    p.add_argument("--recipe", action="store_true",
                   help="print the zones as a recipe's `zones` (projector_px coordinates)")
    p.add_argument("--by-ref", action="store_true",
                   help="... as references (\"ref\": \"zoneset:SET/Z1\": a re-painted set "
                        "updates the recipe)")
    p.add_argument("--mask-out", metavar="NPY",
                   help="write the union DMD mask (1140 x 912 .npy) for scan-fast --mask")
    p = cmd("zone-show", cmd_zone_show, "project a saved zone set on the part, photograph it "
                                        "and report where each zone lands in the camera image "
                                        "(exit 0 every zone seen, 1 one is not)",
            "the zone set projection")
    p.add_argument("set", metavar="SET[:IDS]", help="zone set, optionally some zones (lid:Z1,Z3)")
    p.add_argument("--hold", type=float, default=5.0, metavar="S",
                   help="keep the outlines up S seconds to look at them (default 5)")
    p.add_argument("--label", help="output label (default zones_<set>)")
    p = cmd("zone-delete", cmd_zone_delete, "move a zone set to the library trash "
                                            "(recoverable)", "zone set delete")
    p.add_argument("name", metavar="SET")
    p.add_argument("--reason", required=True, help="why (recorded with the trashed set)")
    p.add_argument("--force", action="store_true", help="delete even when a recipe names it")
    p = cmd("project-capture", cmd_project_capture, "project your own DMD images (1140 x 912 "
                                                    "uint8, <= 256) and photograph each one",
            "the custom pattern capture")
    p.add_argument("patterns", help="an .npz (key 'patterns' or its first array) or a folder of "
                                    "PNGs (name order)")
    p.add_argument("--label", default="custom")
    p.add_argument("--hold", type=float, metavar="S", help="HDMI hold per pattern (default: the "
                                                           "preset's min_hold_s)")
    p.add_argument("--raw", action="store_true", help="send the images as given (no duty / "
                                                      "polarity / gamma chain)")
    p = cmd("maps", cmd_maps, "list projector maps; maps <scan> bakes the surface map stack")
    p.add_argument("id", nargs="?", help="scan id / label / latest: bake its surface maps")
    p.add_argument("--geometry", help="only this geometry id (listing)")
    p.add_argument("--spectral", metavar="ID", help="spectral reading -> albedo")
    p.add_argument("--light", metavar="ID", help="light separation (`light`) -> subsurface")
    p.add_argument("--golden", metavar="ID", help="spectral reading of the golden part -> dust")
    p.add_argument("--sigma", type=float, default=2.0, help="smoothing, camera px (default 2)")

    # 3D meshes, cells, checker
    p = cmd("mesh", cmd_mesh, "export a scan as an STL / OBJ mesh", "mesh export")
    p.add_argument("id")
    p.add_argument("--kind", choices=("relief", "base", "cylinder"), default="relief")
    p.add_argument("--fmt", choices=("stl", "obj"), default="stl")
    p.add_argument("--stride", type=_positive_int, default=2, help="camera px per vertex")
    p.add_argument("--radius", type=float, metavar="MM", help="cylinder: known radius, mm")
    p = cmd("unwrap", cmd_unwrap, "cylinder fit + (theta, z) unwrap of a standing cylinder",
            "the cylinder unwrap")
    p.add_argument("id")
    p.add_argument("--radius", type=float, metavar="MM", help="known radius, mm")
    p.add_argument("--step", type=float, help="grid step, units")
    p = cmd("dents", cmd_dents, "dents on the unwrapped cylinder: exit 0 PASS, 1 FAIL / "
                                "INDETERMINATE", "cylinder dent inspection")
    p.add_argument("id")
    p.add_argument("--radius", type=float, metavar="MM", help="known radius, mm")
    p.add_argument("--sigma", type=float, default=5.0, help="detection threshold, x noise")
    p.add_argument("--min-area", type=_positive_int, default=6, help="min dent, grid bins")
    p.add_argument("--trigger", type=float, metavar="MM",
                   help="FAIL depth, mm (default: preset ch_fail_mm)")
    p = cmd("checker", cmd_checker, "checkerboard corners + local Jacobian (stretch, contrast)",
            "the checker scan")
    p.add_argument("--square", type=_positive_int, default=32, help="square, DMD px")
    p.add_argument("--shifts", type=_positive_int, default=1)
    p.add_argument("--label", default="checker")

    # metric calibration + measurement (mm only with a stereo calibration)
    p = cmd("charuco", cmd_charuco, "printable ChArUco board (PDF at true size + PNG)")
    p.add_argument("--squares-x", dest="squares_x", type=_positive_int, default=10)
    p.add_argument("--squares-y", dest="squares_y", type=_positive_int, default=7)
    p.add_argument("--square", type=float, default=25.0, help="square side, mm")
    p.add_argument("--marker", type=float, default=18.0, help="marker side, mm")
    p.add_argument("--dictionary", default="DICT_5X5_100")
    p.add_argument("--dpi", type=float, default=600.0)
    cmd("parity", cmd_parity, "diamond DMD row-parity check (bench step B3)", "the parity check")
    p = cmd("calib-pose", cmd_calib_pose, "capture one ChArUco board pose for calibrate-stereo",
            "the calibration pose")
    p.add_argument("--set", default="stereo", help="pose set name")
    p.add_argument("--steps", type=_positive_int)
    p.add_argument("--periods", help="T,T,T DMD px, fine -> coarse. Default: this pose set's "
                   "(fixed by its first accepted pose), else 8,80,1280 - captured again at "
                   "16,160,1280, then 24,192,1280 when under 60 %% of the corners get a "
                   "projector position. Given explicitly: used as they are, no retry")
    p.add_argument("--fixture", metavar="SPEC", help="calibration fixture spec JSON "
                   "(tools/make_calib_fixture.py): one PLACEMENT of the fixture, every face "
                   "detected on its own (3 placements in all)")
    p = cmd("calibrate-stereo", cmd_calibrate_stereo,
            "camera + projector + stereo calibration from a pose set", "stereo calibration")
    p.add_argument("--set", help="pose set (default: newest)")
    p.add_argument("--square", type=float, metavar="MM", help="caliper-measured square, mm "
                   "(single board; a fixture takes its squares from its spec)")
    p.add_argument("--mock-poses", dest="mock_poses", type=int, default=0,
                   help="MOCK: place N board poses on the simulated bench")
    p.add_argument("--fixture", metavar="SPEC", help="fixture set: the spec with each face's "
                   "caliper-measured square (default: the spec stored with the placements)")
    p.add_argument("--mock-fixture", dest="mock_fixture", type=int, default=0, metavar="N",
                   help="MOCK: place the calibration fixture N times (3 recommended)")
    p.add_argument("--no-rigid", dest="no_rigid", action="store_true",
                   help="fixture set: skip the rigid-fixture refinement (independent face "
                        "views only)")
    p.add_argument("--steps", type=_positive_int)
    p.add_argument("--periods", help="T,T,T DMD px")
    p.add_argument("--projector-distortion", dest="projector_distortion", action="store_true")
    p = cmd("scan-metric", cmd_scan_metric, "both-axis scan -> point cloud in mm", "metric scan")
    p.add_argument("label", nargs="?", default="metric")
    p.add_argument("--method", choices=("auto", "plane", "rays"), default="auto")
    p.add_argument("--steps", type=_positive_int)
    p.add_argument("--periods", help="T,T,T DMD px")
    p.add_argument("--max-consistency", dest="max_consistency", type=float, metavar="MM")
    p.add_argument("--zone", help='light only these grid cells, e.g. "F11:F12" (black surround)')
    p = cmd("triangulate", cmd_triangulate, "re-triangulate a stored scan in mm")
    p.add_argument("id")
    p.add_argument("--method", choices=("auto", "plane", "rays"), default="auto")
    p = cmd("cyldev", cmd_cyldev, "cylinder fit -> radial deviation map + dents (mm)",
            "cylinder deviation")
    p.add_argument("id", help="metric scan id / label / latest")
    p.add_argument("--radius", type=float, metavar="MM")
    p.add_argument("--sigma", type=float, default=4.0)
    p.add_argument("--min-depth", dest="min_depth", type=float, default=0.05)
    p = cmd("accuracy", cmd_accuracy, "known reference object: mean/max error + repeatability "
                                      "(exit 0 GO)", "the accuracy check")
    p.add_argument("kind", choices=("plane", "step", "cylinder", "sphere"))
    p.add_argument("--height", type=float, metavar="MM", help="step height")
    p.add_argument("--diameter", type=float, metavar="MM", help="cylinder / sphere")
    p.add_argument("--repeats", type=_positive_int, default=3)
    p.add_argument("--scans", nargs="*", help="existing metric scans instead of new ones")
    p.add_argument("--tol", type=float, default=0.1)
    p.add_argument("--repeat-tol", dest="repeat_tol", type=float, default=0.03)
    p = cmd("accuracy-block", cmd_accuracy_block, "scan the printed check block: step heights, "
            "flatness, bar and dome vs your caliper values (exit 0 CONTINUE, 1 REJECT, 5 PULL)",
            "the check block")
    p.add_argument("--block", required=True, metavar="SPEC", help="check block spec JSON "
                   "(tools/make_calib_fixture.py)")
    p.add_argument("--measured", metavar="JSON", help="the filled-in measured template "
                   "(without it every verdict is a PULL: a print is never exactly its model)")
    p.add_argument("--repeats", type=_positive_int, default=1)
    p.add_argument("--scans", nargs="*", help="existing metric scans of the block instead of "
                                              "new ones")
    p.add_argument("--tol-height", dest="tol_height", type=float, metavar="MM",
                   help="step / top heights (default: the spec's, 0.1)")
    p.add_argument("--tol-diameter", dest="tol_diameter", type=float, metavar="MM",
                   help="bar / dome diameters (default: the spec's, 0.15)")
    p.add_argument("--flatness-max", dest="flatness_max", type=float, metavar="MM",
                   help="reference face plane-fit RMS limit (default: the spec's, 0.1)")
    p.add_argument("--label", default="block")
    p.set_defaults(unexpected_exit=EXIT_ERROR)          # a crash (4) is not a REJECT verdict (1)
    p = cmd("precision", cmd_precision, "predicted depth noise sigma_z (mm)")
    p.add_argument("--z", type=float, metavar="MM")
    p.add_argument("--A", type=float, help="fringe offset, grey levels")
    p.add_argument("--B", type=float, help="fringe modulation, grey levels")
    p.add_argument("--period", type=float, help="finest period, DMD px")
    p.add_argument("--steps", type=_positive_int)
    p.add_argument("--scan", help="read A / B / T / N from a metric scan")
    p = cmd("metric-mesh", cmd_metric_mesh, "metric scan -> STL / OBJ in mm (open surface)",
            "metric mesh export")
    p.add_argument("id")
    p.add_argument("--fmt", choices=("stl", "obj"), default="stl")
    p.add_argument("--stride", type=_positive_int, default=1)
    p.add_argument("--max-edge", dest="max_edge", type=float, metavar="MM")
    p.add_argument("--foreground", action="store_true", help="drop the background plane")
    cmd("metric", cmd_metric_list, "list metric scans")

    # dent detection on shiny cylindrical parts (mm only with a stereo calibration)
    p = cmd("dent-scan", cmd_dent_scan, "one pose of a cylinder for the dent detector: long + short "
                                        "exposure patterns + one metric scan", "the dent capture")
    p.add_argument("label", nargs="?", default="dentset")
    p.add_argument("--zone", help='light and keep only these grid cells, e.g. "H9:I12"')
    p.add_argument("--short-preset", dest="short_preset", metavar="NAME",
                   help="preset of the short pass (default: --preset at --short-exposure)")
    p.add_argument("--short-exposure", dest="short_exposure", type=_positive_int, metavar="US",
                   help="short-pass exposure, us (default 16667)")
    p.add_argument("--metric-preset", dest="metric_preset", metavar="NAME",
                   help="preset of the metric scan (default: --preset)")
    p.add_argument("--no-lines", dest="no_lines", action="store_true",
                   help="skip the 10 lines_v frames of the short pass")
    p.add_argument("--no-metric", dest="no_metric", action="store_true",
                   help="no metric scan (then detection needs --metric)")
    p.add_argument("--steps", type=_positive_int, help="metric scan phase steps")
    p.add_argument("--periods", help="metric scan T,T,T DMD px")
    p.add_argument("--pattern-set", dest="pattern_set", choices=("production", "minimal"),
                   default="production", help="minimal drops sin90 T36 (weaker on small dents)")
    p = cmd("dent-null", cmd_dent_null, "calibrate the dent finders on KNOWN-GOOD part(s)",
            "the dent null calibration")
    p.add_argument("dentsets", nargs="+", help="dent set id / label / latest / folder")
    p.add_argument("--diameter", type=float, required=True, metavar="MM", help="cylinder diameter, mm")
    p.add_argument("--label", default="null")
    p.add_argument("--metric", nargs="*", help="metric scans of the (one) good dent set")
    p.add_argument("--fixture", help="fixture name (default: the dent set's zone)")
    p.add_argument("--subset", choices=("all", "production", "production_nolines", "minimal"),
                   default="all")
    p = cmd("dent-detect", cmd_dent_detect, "dents / scratches on a cylinder: exit 0 CLEAR, 1 FLAG, "
                                            "4 the detector failed", "dent detection")
    p.set_defaults(unexpected_exit=EXIT_ERROR)
    p.add_argument("dentset", help="dent set id / label / latest / folder")
    p.add_argument("--diameter", type=float, metavar="MM",
                   help="cylinder diameter, mm (default: that of the newest matching null "
                        "calibration; needed when nulls of several diameters match)")
    p.add_argument("--metric", nargs="*",
                   help="metric scans of the same pose (default: the dent set's own)")
    p.add_argument("--null", help="dent null calibration id / folder (default: newest matching)")
    p.add_argument("--fixture", help="fixture the null was calibrated for (default: the dent "
                                     "set's zone; give the name used with dent-null --fixture)")
    p.add_argument("--label")
    p.add_argument("--subset", choices=("all", "production", "production_nolines", "minimal"),
                   default="all")
    p = cmd("dentsets", cmd_dentsets, "list dent sets")
    p.add_argument("--geometry", help="only this geometry id")

    # end faces: a cylinder lying with its circular END FACE toward the camera
    face_help = "face target id / label / latest (default: the newest at the preset's geometry)"
    p = cmd("face-locate", cmd_face_locate, "locate a circular end face in a metric scan: plane, "
                                            "centre, rim circle, camera -> DMD map", "face locate")
    p.add_argument("metric", nargs="?", help="metric scan id / label / latest (or --zone)")
    p.add_argument("--zone", help="no scan given: scan this zone first (ray-plane), e.g. H10:I12")
    p.add_argument("--steps", type=_positive_int, help="phase steps of that locate scan")
    p.add_argument("--diameter", type=float, default=21.05, metavar="MM",
                   help="face diameter, mm (default 21.05): the diameter of the bright flat "
                        "the flood shows (a rounded rim is not part of it)")
    p.add_argument("--outer-diameter", dest="outer_diameter", type=float, metavar="MM",
                   help="the part's OUTER diameter when the rim around the flat returns depth "
                        "points (e.g. 21.05 for a 19.5 mm flat inside a rounded rim): the rim "
                        "circle is fitted to the depth silhouette at this size, the face keeps "
                        "--diameter (default: found from the rim points when they cover "
                        "enough of the circle)")
    p.add_argument("--size-tol", dest="size_tol", type=float, metavar="PCT",
                   help="known-size check: the flood outline vs --diameter; further off is a "
                        "size mismatch, exit 5 PULL (default 4; 0 = off, with a warning)")
    p.add_argument("--auto-exposure", dest="auto_exposure", action="store_true", default=None,
                   help="--zone: scan again at the frame-locked exposure the located face "
                        "needs when it is dark / clipped (default when the preset's exposure "
                        "is whole video frames)")
    p.add_argument("--no-auto-exposure", dest="auto_exposure", action="store_false",
                   help="--zone: scan at the preset's exposure only")
    p.add_argument("--exposure-us", dest="exposure_us", type=float, metavar="US",
                   help="--zone: force this camera exposure for the locate scan")
    p.add_argument("--exposure-range", dest="exposure_range", metavar="LO,HI",
                   help="auto exposure range, ms (default 16.7,100)")
    p.add_argument("--label", default="face")
    p.add_argument("--roi", metavar="X0,Y0,X1,Y1", help="search only these camera px")
    p.add_argument("--near-band", dest="near_band", type=float, metavar="MM",
                   help="depth face: within this of the nearest surface (default 12)")
    p.add_argument("--far-gap", dest="far_gap", type=float, metavar="MM",
                   help="background: this far behind the face (default 30)")
    p.add_argument("--near-gap", dest="near_gap", type=float, metavar="MM",
                   help="a neighbour (ribbed tube) this far behind the face plane marks the rim "
                        "(default 2)")
    p.add_argument("--flood", help="flood .npy/.npz or a metric scan: the face outline when the "
                                   "depth rim fails (default: the scan's own, else its decode "
                                   "modulation)")
    p = cmd("face-aim", cmd_face_aim, "project the face's ring + cross and photograph it (the "
                                      "placement guide); --check: exit 0 ON_FACE, 1 OFF",
            "the face ring")
    p.add_argument("--face", help=face_help)
    p.add_argument("--hold", type=float, metavar="S", help="keep the ring up N s (pumped hold)")
    p.add_argument("--offset", metavar="COLS,ROWS", help="shift the ring, DMD px")
    p.add_argument("--check", action="store_true",
                   help="also check the face is still where it was located (flood + ring)")
    p.add_argument("--tol", type=float, default=1.5, metavar="PX",
                   help="--check: ring centre tolerance, camera px (default 1.5)")
    p.add_argument("--label")
    p = cmd("face-intensity", cmd_face_intensity, "projection.duty for a ~200 DN face p99 (never "
                                                  "the LEDs): exit 0 verified", "auto intensity")
    p.add_argument("--face", help=face_help)
    p.add_argument("--target", type=float, default=200.0, metavar="DN")
    p.add_argument("--tol", type=float, default=10.0, metavar="DN")
    p = cmd("face-scan", cmd_face_scan, "targeted end-face scan: only the face lit, N repeats + "
                                        "polar set", "the targeted face scan")
    p.add_argument("label", nargs="?", default="face")
    p.add_argument("--face", help=face_help)
    p.add_argument("--repeats", type=_positive_int, default=3)
    p.add_argument("--margin", type=float, default=6.0, metavar="DMD_PX",
                   help="lit margin around the face ellipse (default 6)")
    p.add_argument("--no-polar", dest="no_polar", action="store_true",
                   help="no rings / spokes (the flood pair is still captured: the FIR needs it)")
    p.add_argument("--polar-steps", dest="polar_steps", type=_positive_int, default=8)
    p.add_argument("--duty", type=float, help="projection.duty for this scan only")
    p.add_argument("--auto-intensity", dest="auto_intensity", action="store_true",
                   help="run face-intensity first and scan at its duty")
    p.add_argument("--auto-exposure", dest="auto_exposure", action="store_true", default=None,
                   help="choose a frame-locked camera exposure for this seating: face flood p95 "
                        "near the target with < 0.5 %% clipped (default when the preset's "
                        "exposure is whole video frames; works with inverted presets; exit 5 "
                        "and no scan when none fits)")
    p.add_argument("--no-auto-exposure", dest="auto_exposure", action="store_false",
                   help="scan at the preset's exposure")
    p.add_argument("--exposure-us", dest="exposure_us", type=float, metavar="US",
                   help="force this camera exposure for this scan (the preset is not edited)")
    p.add_argument("--exposure-target", dest="exposure_target", type=float, metavar="DN",
                   help="auto exposure: target face p95 (default 170 of 255)")
    p.add_argument("--exposure-range", dest="exposure_range", metavar="LO,HI",
                   help="auto exposure range, ms (default 16.7,100)")
    p.add_argument("--steps", type=_positive_int, help="depth-set phase steps (default 12)")
    p.add_argument("--periods", help="depth-set T,T,T DMD px (tilephase: one T, default 8)")
    p.add_argument("--mode", choices=("multifreq", "tilephase"), default="multifreq",
                   help="depth set: multifreq (default; 3 phase levels, 74 frames) or tilephase "
                        "(EXPERIMENTAL: 1 phase level + Gray tile code, 36 frames, ~2x faster; the "
                        "order is unique only in a 2^(bits-1) x 8 px window on the face, more "
                        "sensitive to blur / tilt / glints / motion, weaker consistency check)")
    p.add_argument("--code-bits", dest="code_bits", type=_positive_int, default=5,
                   help="tilephase: Gray code planes per axis (5 = a 128 DMD px window)")
    p = cmd("face-profile", cmd_face_profile, "face height map, radial profile, sections, "
                                              "flatness, repeatability (+ PNG, CSV, STL)",
            "the face profile")
    p.add_argument("scans", nargs="+", help="a face scan (id / label / latest) or its metric scans")
    p.add_argument("--face", help="face target (default: the face scan's own)")
    p.add_argument("--label")
    p.add_argument("--grid", type=float, default=0.25, metavar="MM")
    p.add_argument("--r-flat", dest="r_flat", type=float, metavar="MM",
                   help="flatness inside this radius (default 0.855 x the face radius: 9.0 on "
                        "a 21.05 mm face; a value given is used as given)")
    p.add_argument("--size-tol", dest="size_tol", type=float, metavar="PCT",
                   help="known-size check: the flood outline vs the face diameter (default 4)")
    p.add_argument("--no-stl", dest="no_stl", action="store_true")
    p = cmd("face-golden", cmd_face_golden, "golden end face from GOOD placements: MF-CFAR "
                                            "(+ reflectance FIR) and band models, leave-one-out, "
                                            "synthetic dents", "the golden face")
    p.add_argument("placements", nargs="+", help="face scans (id / label) or face-profile .npz")
    p.add_argument("--label", default="golden")
    p.add_argument("--detector", choices=("mf_fir", "mf", "band"),
                   help="default detector of the golden (default mf_fir)")
    p.add_argument("--z", type=float, help="MF: trigger at dent z >= this (default 5.5; 5.0 is "
                                           "~10x more false triggers, +~10 points detection)")
    p.add_argument("--min-depth", dest="min_depth", type=float, metavar="MM",
                   help="MF: ignore dents shallower than this (default 0)")
    p.add_argument("--no-fir", dest="no_fir", action="store_true",
                   help="no reflectance FIR (from the flood pair)")
    p.add_argument("--no-synthetic", dest="no_synthetic", action="store_true",
                   help="skip the synthetic-dent set")
    p.add_argument("--positions", type=_positive_int,
                   help="synthetic set: random dent centres per size per placement (default 6)")
    p.add_argument("--k", type=float, help="band = k x sigma (default 4)")
    p.add_argument("--floor", type=float, metavar="MM", help="band floor (default 0.10)")
    p.add_argument("--smooth-cells", dest="smooth_cells", type=_positive_int,
                   help="sigma median window, cells (default 5)")
    p.add_argument("--rim-r", dest="rim_r", type=float, metavar="MM",
                   help="rim zone r > this, for EVERY detector (default 0.855 x the face "
                        "radius: 9.0 on a 21.05 mm face)")
    p.add_argument("--inspect-r", dest="inspect_r", type=float, metavar="MM",
                   help="inspected zone r <= this (default 0.926 x the face radius: 9.75 on a "
                        "21.05 mm face); the chamfer / rounded rim beyond is never inspected")
    p.add_argument("--plane-r", dest="plane_r", type=float, metavar="MM",
                   help="robust plane region (default: the rim radius)")
    p.add_argument("--rim-tol", dest="rim_tol", type=float, metavar="MM",
                   help="rim outline: a notch / bulge beyond this ... (default 0.4)")
    p.add_argument("--rim-arc", dest="rim_arc", type=float, metavar="MM",
                   help="... over at least this much rim triggers (default 1.5)")
    p.add_argument("--size-tol", dest="size_tol", type=float, metavar="PCT",
                   help="known-size check: a placement whose flood outline is further off the "
                        "face diameter is refused (default 4)")
    p.add_argument("--rim-factor", dest="rim_factor", type=float, help="rim band x (default 2)")
    p.add_argument("--rim-min", dest="rim_min", type=float, metavar="MM", help="rim band min (0.3)")
    p.add_argument("--height-smooth", dest="height_smooth", type=float, metavar="MM",
                   help="Gaussian sigma on the height maps (default 0.25)")
    p.add_argument("--min-area", dest="min_area", type=float, metavar="MM2",
                   help="band: smallest trigger (default 0.25)")
    p = cmd("face-check", cmd_face_check, "a placement vs the golden face: exit 0 PASS, 1 FAIL, "
                                          "5 INCOMPLETE = PULL (+ overlay PNG, json)",
            "the face check")
    p.add_argument("placement", help="face scan (id / label / latest) or face-profile .npz")
    p.add_argument("--golden", help="golden face id / label (default: newest matching)")
    p.add_argument("--label")
    p.add_argument("--min-area", dest="min_area", type=float, metavar="MM2")
    p.add_argument("--detector", choices=("mf_fir", "mf", "band"),
                   help="default: the golden's (mf_fir for new goldens)")
    p.add_argument("--z", type=float, help="MF: trigger z for this check (default: the golden's)")
    p.add_argument("--min-depth", dest="min_depth", type=float, metavar="MM",
                   help="MF: ignore dents shallower than this (depth_centre_mm) for this check")
    p.add_argument("--rim-tol", dest="rim_tol", type=float, metavar="MM",
                   help="rim outline tolerance for this check (default: the golden's, 0.4)")
    p.add_argument("--rim-arc", dest="rim_arc", type=float, metavar="MM",
                   help="rim outline minimum arc for this check (default: the golden's, 1.5)")
    p.add_argument("--size-tol", dest="size_tol", type=float, metavar="PCT",
                   help="known-size tolerance for this check (default: the golden's, 4)")
    p = cmd("faces", cmd_faces, "list face targets, face scans and golden faces")
    p.add_argument("--geometry", help="only this geometry id")

    # HDMI fast zone scan + station (cylinder end faces + tube edge)
    p = cmd("scan-fast", cmd_scan_fast, "HDMI fast zone scan: columns only (4 steps x 8/80/1280 + "
                                        "floods), average 2, hold 0.12 s -> metric scan + flood "
                                        "pair", "fast scan")
    p.add_argument("label", nargs="?", default="fast")
    p.add_argument("--zone", help='light only these grid cells ("F10:J12") or atlas:<name>/<region>')
    p.add_argument("--mask", metavar="NPY", help="or a DMD mask (1140 x 912 .npy)")
    p.add_argument("--zones", metavar="SET[:IDS]",
                   help="or a painted zone set (scan zone-paint): light and decode only its "
                        "zones (lid, or lid:Z1,Z3; their union is lit); the result gives the "
                        "points per zone")
    p.add_argument("--face", action="append", metavar="FACE",
                   help="also light this face target's ellipse (+ --margin); repeatable")
    p.add_argument("--margin", type=float, default=6.0, metavar="DMD_PX")
    p.add_argument("--steps", type=_positive_int, default=4)
    p.add_argument("--periods", help="T,T,T DMD px (default 8,80,1280)")
    p.add_argument("--hold", type=float, default=0.12, metavar="S",
                   help="HDMI hold per pattern for this run (default 0.12 s)")
    p.add_argument("--axis", choices=("x", "xy"),
                   help="x (default, enough for station heights) or xy (needed by face-locate; "
                        "the default with --zones, so every point is assigned to its zone)")
    station_help = ("station parameters: --band LO,HI (tube edge height band, mm; default auto), "
                    "--roi, --skip-zones L,R, --centre")
    p = cmd("station", cmd_station, "cylinder end faces + tube edge zones of a metric scan: "
                                    "heights, spacing, T1..Tn (+ PNG, JSON, CSV)", "station")
    p.add_argument("metric", help="metric scan id / label / latest")
    p.add_argument("--flood", help="flood .npy/.npz or a metric scan with a flood pair (default: "
                                   "the scan's own, else its decode modulation)")
    p.add_argument("--label")
    p.add_argument("--cells", type=_positive_int, default=2)
    p.add_argument("--diameter", type=float, default=21.05, metavar="MM")
    p.add_argument("--band", metavar="LO,HI", help=station_help)
    p.add_argument("--roi", metavar="X0,Y0,X1,Y1", help="tube search window, camera px")
    p.add_argument("--skip-zones", dest="skip_zones", metavar="L,R",
                   help="drop this many tube zones at the left / right end (e.g. a lead-in)")
    p.add_argument("--centre", choices=("rim_circle", "ellipse_ray"),
                   help="face centre method (default rim_circle)")
    p.add_argument("--no-ct", dest="no_ct", action="store_true", help="end faces only")
    p.add_argument("--no-strict", dest="no_strict", action="store_true",
                   help="drop the strict end-face gates (rim outline rms <= 1.0 mm, >= 70 %% of the "
                        "disc measured - 50-70 %% only with a clean rim circle - a plausible "
                        "height / spacing to the other end faces, and the known size)")
    p.add_argument("--size-tol", dest="size_tol", type=float, metavar="PCT",
                   help="known-size check: an end face whose flood outline is further off "
                        "--diameter is a size mismatch (default 4; 0 = off)")
    p = cmd("station-golden", cmd_station_golden, "station golden from GOOD measurements (metric "
                                                  "scans or station .json) + limits", "station golden")
    p.add_argument("items", nargs="+", help="metric scans (id / label) or station .json files")
    p.add_argument("--label", default="golden")
    p.add_argument("--cell-tol", dest="cell_tol", type=float, metavar="MM",
                   help="end-face height difference: nominal +- this (default 0.15)")
    p.add_argument("--ct-tol", dest="ct_tol", type=float, metavar="MM",
                   help="tube zone edge height: nominal +- this (default 0.30)")
    p.add_argument("--spacing-tol", dest="spacing_tol", type=float, metavar="MM")
    p.add_argument("--position-tol", dest="position_tol", type=float, metavar="MM",
                   help="each end face within this of its golden position, 3D (default: the "
                        "face diameter): a face found elsewhere is never the part")
    p.add_argument("--cell-band", dest="cell_band", metavar="LO,HI",
                   help="absolute band for every end-face height difference, mm")
    p.add_argument("--ct-band", dest="ct_band", metavar="LO,HI",
                   help="absolute band for every tube zone height, mm")
    p.add_argument("--band", action="append", metavar="ITEM=LO,HI",
                   help="per-item band, e.g. C2-C1=-0.4,0.1 or T3=-5,-4 (repeatable)")
    p.add_argument("--flood", help="flood for a single metric scan (file or scan)")
    # station parameters for metric-scan items (as `station` takes them; station .json items
    # keep the parameters they were measured with)
    p.add_argument("--cells", type=_positive_int, help="cylinder end faces per scan (default 2)")
    p.add_argument("--diameter", type=float, metavar="MM", help="face diameter (default 21.05)")
    p.add_argument("--edge-band", dest="edge_band", metavar="LO,HI",
                   help="tube edge height band for the measurement, mm (default auto)")
    p.add_argument("--roi", metavar="X0,Y0,X1,Y1", help="tube search window, camera px")
    p.add_argument("--skip-zones", dest="skip_zones", metavar="L,R",
                   help="drop this many tube zones at the left / right end (the golden's numbering)")
    p.add_argument("--centre", choices=("rim_circle", "ellipse_ray"),
                   help="face centre method (default rim_circle)")
    p.add_argument("--no-ct", dest="no_ct", action="store_true", help="end faces only")
    p.add_argument("--size-tol", dest="size_tol", type=float, metavar="PCT",
                   help="known-size check for metric-scan items (default 4; 0 = off)")
    p = cmd("station-check", cmd_station_check, "a metric scan vs the station golden: exit 0 PASS, "
                                                "1 FAIL, 5 INCOMPLETE / ambiguous / size mismatch = PULL, "
                                                "2 refused, 4 crashed (+ PNG, JSON, CSV)",
            "the station check")
    p.set_defaults(unexpected_exit=EXIT_ERROR)          # a crash (4) is not a FAIL verdict (1)
    p.add_argument("scan", help="metric scan id / label / latest, or a station .json")
    p.add_argument("--golden", help="station golden id / label (default: newest matching)")
    p.add_argument("--flood", help="flood .npy/.npz or a metric scan (default: the scan's own)")
    p.add_argument("--label")
    p = cmd("stations", cmd_stations, "list station goldens")
    p.add_argument("--geometry", help="only this geometry id")

    # Surfer atlas + methods
    region_help = "atlas:<name>/<region> | grid zone P17:P22@48 | DMD rect x0,y0,x1,y1"
    p = cmd("atlas-import", cmd_atlas_import, "copy a Surfer atlas JSON into the library",
            "atlas import")
    p.add_argument("path")
    p.add_argument("name")
    p = cmd("atlases", cmd_atlases, "list imported atlases and their regions")
    p.add_argument("--geometry", help="only this geometry id")
    p = cmd("flood", cmd_flood, "light only a region and photograph it: exit 0 it lands",
            "flood verify")
    p.add_argument("region", help=region_help)
    p.add_argument("--level", type=int, default=255)
    p.add_argument("--label", default="flood_verify")
    p = cmd("antiglare", cmd_antiglare, "carrier sweep that least clips glare in a region: "
                                        "exit 0 locked", "the anti-glare sweep")
    p.add_argument("region", help=region_help)
    p.add_argument("--carriers", type=_carriers, metavar="FX,FY[,Z];..",
                   help="carriers to try (default set)")
    p.add_argument("--steps", type=_positive_int, default=4)
    p.add_argument("--cycles", type=float, default=6.0)
    p.add_argument("--level", type=int, default=255)
    p.add_argument("--label", default="anti_glare")
    p = cmd("multiorient", cmd_multiorient, "v / h / diagonal fringes fused by modulation",
            "the multi-orientation scan")
    p.add_argument("region", help=region_help)
    p.add_argument("--orientations", type=_list_of(str, "--orientations"),
                   default=["v", "h", "d45", "d135"], metavar="v,h,d45,d135")
    p.add_argument("--steps", type=_positive_int, default=4)
    p.add_argument("--period", type=float, default=24.0, help="fringe period, DMD px")
    p.add_argument("--level", type=int, default=255)
    p.add_argument("--label", default="multi_orient")
    p = cmd("flash-pack", cmd_flash_pack, "build a DLPC350 flash / video-port pattern pack "
                                          "(files only)", "the flash pack")
    p.add_argument("--axes", choices=("x", "y"), default="x", help="one axis per pack")
    p.add_argument("--video-port", action="store_true")
    p.add_argument("--label")
    p.add_argument("--period-us", type=int, help="uniform pattern period, us (8-bit >= 8563)")
    p.add_argument("--pattern-exposure-us", type=int, help="pattern exposure, us")
    p.add_argument("--trigger-mode", choices=("trigout", "external"),
                   help="DLPC350 timer + TRIG_OUT_1 (default) or external TRIG_IN_1")
    p = cmd("fastscan-pack", cmd_fastscan_pack, "build the 1-bit soft-sync fast-scan pack "
                                                "(full + zone flash images; files only)",
            "the fast-scan pack")
    p.add_argument("--zone-ring", action="append", default=[], metavar="NPY",
                   help="face_dmd_ring.npy (DMD col,row rim points) -> ellipse + margin")
    p.add_argument("--zone-mask", action="append", default=[], metavar="NPY",
                   help="a DMD-shaped mask .npy added to the zone")
    p.add_argument("--band-cam", metavar="NPY", help="camera-space band mask (tube band)")
    p.add_argument("--band-field", metavar="NPZ",
                   help="both-axes field.npz mapping that band to the DMD")
    p.add_argument("--margin", type=int, default=6, help="ellipse margin, DMD px")
    p.add_argument("--calibration", metavar="DIR",
                   help="stereo calibration folder (camera px / column, mm / column)")
    p.add_argument("--study", action="store_true",
                   help="simulate the per-level decode error into the manifest")
    p.add_argument("--label")
    p.add_argument("--period-us", type=int, help="pattern period, us (default 10000; triggered "
                                                  "full frame >= 4425)")
    p.add_argument("--pattern-exposure-us", type=int, help="pattern exposure (default = period)")
    p.add_argument("--camera-exposure", type=int, metavar="US",
                   help="camera exposure, us (default: soft sync 3000; triggered min(3000, "
                        "pattern exposure - 1000))")
    p.add_argument("--trigger-mode", choices=("freerun", "trigout", "external"),
                   help="freerun = soft sync (default); trigout / external = TRIG_OUT_1 -> "
                        "camera (docs/TRIGGER_FULLSPEED_TEST.md)")
    p = cmd("scan-flash", cmd_scan_flash, "fast metric scan: the flash loop a human plays in the "
                                          "4500 MKII GUI + free-running camera (soft sync)",
            "the flash soft-sync scan")
    p.add_argument("label", nargs="?", default="flash")
    p.add_argument("--pack", metavar="DIR", help="the fast-scan pack the GUI plays (its folder)")
    p.add_argument("--variant", choices=("full", "zone"),
                   help="the pack image the GUI plays (default: reported from the frames)")
    p.add_argument("--zone", help='camera window from grid cells / an atlas region ("F11:F12")')
    p.add_argument("--roi", metavar="X0,Y0,X1,Y1", help="camera window, camera px")
    p.add_argument("--loops", type=_positive_int, help="pattern loops to record (default: auto)")
    p.add_argument("--exposure", type=float, metavar="US", help="camera exposure (default: the "
                                                               "pack's)")
    p.add_argument("--fps", type=float, help="camera frame rate (default: the window's maximum)")
    p.add_argument("--gain", type=float, metavar="DB", help="camera gain (default: the preset's)")
    p.add_argument("--no-recording", action="store_true",
                   help="do not keep the raw frames in output/flashscans")
    p.add_argument("--dry-run", action="store_true", help="timing plan only; opens nothing")
    p.add_argument("--fine-prefilter", type=float, metavar="PX",
                   help="camera-px low-pass of the fine fringes (1 = the pack's recommendation "
                        "at a sharp projector; default off)")
    p = cmd("trigger-test", cmd_trigger_test, "EXPERIMENTAL: count hardware trigger pulses at "
                                              "the camera (camera only)", "the trigger test")
    p.add_argument("--line", default="Line0", choices=TRIGGER_LINES,
                   help=TRIGGER_LINE_HELP)
    p.add_argument("--seconds", type=float, default=5.0)
    p.add_argument("--expect-period-ms", type=float, help="the GUI's Pattern Period, ms")
    p.add_argument("--exposure", type=float, default=1000.0, metavar="US",
                   help="camera exposure (short; default 1000 us)")
    p.set_defaults(unexpected_exit=EXIT_ERROR)          # an error (4) is not NO PULSES (1)
    p = cmd("scan-trigger", cmd_scan_trigger, "EXPERIMENTAL hardware-triggered metric scan of a "
                                              "flash pack (TRIG_OUT_1 -> camera)",
            "the hardware-triggered scan")
    p.add_argument("label", nargs="?", default="trigger")
    p.add_argument("--pack", metavar="DIR", required=True, help="the pack the GUI plays")
    p.add_argument("--variant", choices=("full", "zone"), help="fast-scan pack image played")
    p.add_argument("--line", default="Line0", choices=TRIGGER_LINES,
                   help=TRIGGER_LINE_HELP)
    p.add_argument("--repeats", type=_positive_int, default=1, help="loops to average")
    p.add_argument("--exposure", type=float, metavar="US", help="camera exposure (default: "
                                                               "the pack's)")
    p.add_argument("--first-timeout", type=float, default=30.0, metavar="S",
                   help="seconds to wait for the first trigger")
    p.add_argument("--pattern-mode-set-by-human", action="store_true",
                   help="REQUIRED: a human put the projector in pattern mode in the GUI")
    p = cmd("sort-flash", cmd_sort_flash, "offline: sort a saved flash recording into "
                                          "per-pattern images", "the offline sort")
    p.add_argument("recording", help="output/flashscans/<...>.npz from scan-flash")
    p.add_argument("--pack", metavar="DIR", help="default: the layout stored in the recording")
    p.add_argument("--variant", choices=("full", "zone"))
    p.add_argument("--decode", action="store_true", help="also decode projector columns")
    p.add_argument("--fine-prefilter", type=float, metavar="PX",
                   help="with --decode: camera-px low-pass of the fine fringes (default off)")
    p.add_argument("--label")
    p = cmd("rig-check", cmd_rig_check, "rig check: fresh flood + pattern vs the stored "
                                        "reference -> camera / projector motion (shift, roll, "
                                        "zoom; px, ~mm)", "the rig check")
    p.add_argument("--set-reference", action="store_true",
                   help="store a new reference (right after base / calibration). On a fixed rig "
                        "it also answers an earlier MOVED (\"the rig is known good\"); on a "
                        "mobile rig that is not verified it verifies nothing, now or later")
    p.add_argument("--tol", type=_finite_nonneg, default=0.5,
                   help="MOVED above this image motion, px (default 0.5; finite, >= 0). An OK at "
                        "a looser tolerance than the default is reported but does not verify the "
                        "rig for mm results")
    p.add_argument("--mock-shift", metavar="DX,DY[,ROLL[,SCALE]]",
                   help="mock only: move the fresh frames (px, deg, x; simulates a nudged "
                        "camera)")
    p.add_argument("--rehome", action="store_true",
                   help="after a small CAMERA move (MOVED: flood and pattern moved together): "
                        "fit the camera rotation from the reference flood vs the fresh one and "
                        "write a NEW stereo calibration R1 = R0 @ Rd (t unchanged), flagged "
                        "re-homed (rotation only); the original is kept. Refused when the "
                        "projector moved, the motion is > ~5 px or badly registered. Absolute "
                        "depth may keep a small offset: a full calibration is the reference")
    p = cmd("run", cmd_run, "flow-ready job: run a station config, write result.json + "
                            "run.log.jsonl in a new run folder (exit 0 CONTINUE, 1 REJECT, "
                            "2 refused, 4 crashed, 5 PULL). A single-job config runs its job; a "
                            "station configuration with profiles + a flow runs one whole cycle "
                            "of its work states", "the station job")
    p.add_argument("config", help="station config .json (or .yaml with PyYAML), "
                                  "e.g. stations/example_cell_station.json, "
                                  "stations/example_tube_line.json")
    p.add_argument("--operator-confirmed", action="store_true",
                   help="a human confirmed the physical setup (required on the real rig)")
    p.add_argument("--no-feedback", action="store_true",
                   help="do not project the verdict / the faults on the part")
    p.add_argument("--profile", metavar="NAME",
                   help="flow: the profile of the configuration's profile library set to check "
                        "against (default: its default_profile)")
    p.add_argument("--input", action="append", metavar="SIGNAL=true|false",
                   help="flow: what a line sensor says for THIS cycle (repeatable), e.g. "
                        "part_present=true; a work state that expects a signal nobody gave is "
                        "a DOUBT (PULL)")
    p.add_argument("--mock-case", metavar="NAME",
                   help="flow, MOCK only: a case of the configuration's simulated line (the "
                        "scene placed + what the sensors say); default: its default_case")
    p.add_argument("--acknowledge", action="store_true",
                   help="flow: acknowledge every prompt as an operator would (default: a prompt "
                        "nobody acknowledges ends the cycle PULL; --interactive asks). A work "
                        "state's prompts are answered at most retries + 1 times, then the "
                        "cycle ends PULL")
    p.set_defaults(unexpected_exit=EXIT_ERROR)          # a crash (4) is not a FAIL verdict (1)
    p = cmd("flow-show", cmd_flow_show, "show + validate a station configuration: parameters, "
            "profiles (the profile library set), work states (the flow), permissions; exit 2 "
            "lists every problem", "the station configuration")
    p.add_argument("config", help="station config .json (or .yaml with PyYAML)")

    # operating modes: Measure / Quick check (Station = `run` on a station configuration)
    roi_help = "x0,y0,x1,y1 camera px (x1, y1 exclusive)"
    p = cmd("measure-run", cmd_measure_run,
            "MEASURE mode: scan and report numbers with units, n points and uncertainty - no "
            "tolerance, no pass / fail. A new run folder (result.json, values.csv, overlay.png, "
            "scan.ply, spec.json). Exit 0 MEASURED, 5 NOT MEASURABLE (uncalibrated, a mobile rig "
            "not verified, rig check MOVED, too few points: no number is given), 2 refused, "
            "4 crashed", "the measurement")
    p.add_argument("preset_name", nargs="?", metavar="preset",
                   help="the triangulation preset (default: --preset / the selected one, or "
                        "the spec's)")
    p.add_argument("--what", choices=("flatness", "step", "distance", "cylinder", "endface",
                                      "dims", "deviation"),
                   help="flatness: RMS, p2-p98, tilt of the region | step: region B (--roi2) "
                        "above region A's plane (--roi) | distance: --p1 to --p2, or --p1 to "
                        "the plane of --roi | cylinder: radius, diameter, axis | endface: "
                        "end-face diameter (--diameter = the expected one) | dims: L x W x H of "
                        "the part above the reference plane | deviation: vs --golden (values "
                        "only)")
    p.add_argument("--roi", metavar="X0,Y0,X1,Y1", help=f"region A: {roi_help}")
    p.add_argument("--roi2", metavar="X0,Y0,X1,Y1", help=f"region B (step): {roi_help}")
    p.add_argument("--zone", metavar="ZONE", help="region A as grid cells (H8:I11; also lights "
                                                  "only them)")
    p.add_argument("--zone2", metavar="ZONE", help="region B as grid cells (needs a projector "
                                                   "map: scan map)")
    p.add_argument("--zones", metavar="SET[:IDS]",
                   help="region A as a painted zone set (scan zone-paint): only its zones are "
                        "lit, and the camera px that decode into them are the region")
    p.add_argument("--p1", metavar="X,Y", help="a picked point, camera px")
    p.add_argument("--p2", metavar="X,Y", help="a second picked point, camera px")
    p.add_argument("--golden", metavar="ID", help="deviation: the golden metric scan (id / label)")
    p.add_argument("--radius", type=float, metavar="MM",
                   help="cylinder: fix the radius (it is then not measured)")
    p.add_argument("--axis", choices=("auto", "x", "y", "z"),
                   help="cylinder: axis hint in the camera frame (default auto)")
    p.add_argument("--diameter", type=float, metavar="MM",
                   help="endface: the expected face diameter (the face is found near it; its "
                        "diameter is then measured)")
    p.add_argument("--faces", type=_positive_int, metavar="N",
                   help="endface: end faces in view (default 1)")
    p.add_argument("--min-height", type=float, metavar="MM",
                   help="dims: the part stands at least this far above the reference plane "
                        "(default 2)")
    p.add_argument("--repeats", type=_positive_int, metavar="N",
                   help="scan N times: each value's mean and its repeatability (default 1: the "
                        "predicted precision at that distance)")
    p.add_argument("--scan", nargs="+", metavar="ID",
                   help="measure stored metric scan(s) (id / label / latest) instead of "
                        "capturing; several = the repeats")
    p.add_argument("--scan-kind", choices=("metric", "fast"),
                   help="metric (default: both axes, the preset's set) | fast (HDMI fast scan)")
    p.add_argument("--steps", type=int, help="phase steps of the scan")
    p.add_argument("--periods", metavar="T,T,..", help="fringe periods, DMD px (fine -> coarse)")
    p.add_argument("--spec", metavar="FILE", help="a measurement spec (.json, e.g. the spec.json "
                                                  "of an earlier run); the options override it")
    p.add_argument("--save-spec", metavar="FILE", help="also write the spec that ran to FILE")
    p.add_argument("--name", help="a name for the spec")
    p.add_argument("--label", help="scan label (default measure_<what>)")
    p.add_argument("--runs-dir", metavar="DIR", help="where the run folder goes (default "
                   "$SCANNER_RUNS_DIR, else <library>/runs)")
    p.set_defaults(unexpected_exit=EXIT_ERROR)          # a crash (4) is not a status
    p = cmd("measure-to-check", cmd_measure_to_check,
            "promote a measurement spec to a check by adding tolerances: flatness / step -> a "
            "recipe, deviation -> a single-job station config", "the promotion")
    p.add_argument("spec", help="measurement spec .json (the spec.json of a measure run)")
    p.add_argument("--tol", action="append", metavar="NAME=LO,HI", required=True,
                   help="a tolerance on a value of the measurement: NAME=LO,HI (a band) or "
                        "NAME=HI (an upper limit), e.g. flatness_p2_p98=0.3, "
                        "step_height=19.5,20.5, deviation_max=0.5 (repeatable)")
    p.add_argument("--name", help="the check's name (default <what>_check)")
    p.add_argument("--guard-band", type=_finite_nonneg, metavar="PCT",
                   help="guard band in %% of the tolerance: a PASS that close to a limit is "
                        "PULLed (default 0)")
    p.add_argument("--out", metavar="FILE", help="write the check to FILE (never overwritten)")
    p = cmd("quick-check", cmd_quick_check,
            "QUICK CHECK mode: a triangulation preset + ONE check (a plane flatness limit, a "
            "golden compare or one recipe) -> CONTINUE / REJECT / PULL; no sensors, no "
            "permissions. Runs as a one-job station config (kept in the run folder as "
            "quick_check.json). Exit 0 CONTINUE, 1 REJECT, 5 PULL, 2 refused, 4 crashed",
            "the quick check")
    p.add_argument("preset_name", nargs="?", metavar="preset",
                   help="the triangulation preset (default: --preset / the selected one; a "
                        "recipe check: the recipe's own)")
    p.add_argument("--check", choices=("flatness", "golden", "recipe"), required=True,
                   help="flatness: max |deviation| from the plane in --roi vs --threshold | "
                        "golden: vs the golden metric scan --golden, within --threshold | "
                        "recipe: the recipe --recipe (its own tolerances)")
    p.add_argument("--threshold", "--limit", dest="threshold", type=float, metavar="MM",
                   help="flatness / golden: the limit, mm")
    p.add_argument("--roi", metavar="X0,Y0,X1,Y1", help=f"flatness / golden: {roi_help}")
    p.add_argument("--zone", metavar="ZONE", help="flatness / golden: light only these grid cells")
    p.add_argument("--golden", metavar="ID", help="golden: the golden metric scan (id / label)")
    p.add_argument("--recipe", metavar="NAME", help="recipe: a recipe name or file")
    p.add_argument("--guard-band", type=_finite_nonneg, metavar="PCT",
                   help="flatness / golden: a PASS within this %% of the threshold is PULLed "
                        "(default 0 = off)")
    p.add_argument("--scan-kind", choices=("fast", "metric"),
                   help="flatness / golden: fast (default) | metric")
    p.add_argument("--operator-confirmed", action="store_true",
                   help="a human confirmed the physical setup (required on the real rig)")
    p.add_argument("--no-feedback", action="store_true",
                   help="recipe: do not project the verdict on the part")
    p.add_argument("--runs-dir", metavar="DIR", help="where the run folder goes (default "
                   "$SCANNER_RUNS_DIR, else <library>/runs)")
    p.set_defaults(unexpected_exit=EXIT_ERROR)          # a crash (4) is not a verdict (1)
    p = cmd("rig-session", cmd_rig_session,
            "the rig's state for mm results (mount, calibration, last rig check, verified?); "
            "--new starts a new session: a MOBILE rig (rig.mount) counts as moved from now "
            "and gives no mm until rig-check / re-home / calibrate. Exit 0 mm allowed, 1 not",
            "the rig session")
    p.add_argument("--new", action="store_true",
                   help="the rig was set down / powered up / moved now")
    p.add_argument("--reason", help="why (recorded with the marker)")

    # virtual sensors
    at_help = ("grid cell H8 (H8@96 pins the 96 px grid), zone G7:H9 (presence only), "
               "dmd:COL,ROW (DMD px) or cam:X,Y (camera px, needs a map)")
    measured_help = "MEASURED mm per DMD column (measured mm / drawn columns); default nominal"
    p = cmd("spot-size", cmd_spot_size, "spot of D mm at the part -> DMD px (cols x rows), "
                                        "nominal vs measured", "spot sizing")
    p.add_argument("diameter", type=float, metavar="MM", help="spot diameter at the part, mm")
    p.add_argument("--measured", type=float, metavar="K", help=measured_help)
    p = cmd("spot", cmd_spot, "project one spot, photograph and measure it", "the projected spot")
    p.add_argument("at", help=at_help)
    p.add_argument("--diameter", type=float, required=True, metavar="MM",
                   help="spot diameter at the part, mm")
    p.add_argument("--measured", type=float, metavar="K", help=measured_help)
    p.add_argument("--shape", choices=("disk", "square"))
    p.add_argument("--level", type=int, help="DMD level 1-255 (default 255)")
    p.add_argument("--label", help="output label (default spot)")
    p.add_argument("--hold", type=float, help="keep the spot up N s to aim (holds the rig)")
    p = cmd("teach-presence", cmd_teach_presence, "teach a part-present sensor on the EMPTY "
                                                  "field", "the part-present sensor")
    p.add_argument("name")
    p.add_argument("at", help=at_help)
    p.add_argument("--diameter", type=float, metavar="MM", help="spot diameter, mm (spot sensors)")
    p.add_argument("--method", choices=("photoelectric", "bgs"), default="photoelectric",
                   help="photoelectric: returned light; bgs: background suppression "
                        "(triangulation, blind to colour)")
    p.add_argument("--measured", type=float, metavar="K", help=measured_help)
    p.add_argument("--threshold", type=float, help="photoelectric: relative change that means "
                                                   "PRESENT, fraction (default 0.25)")
    p.add_argument("--min-height", type=float, metavar="MM",
                   help="bgs: height above the taught background, mm (default 3)")
    p.add_argument("--level", type=int, help="DMD level 1-255 (default 255)")
    p.add_argument("--empty-confirmed", action="store_true",
                   help="the operator has cleared the sensor's field. Required: without it "
                        "the command is refused at once (it never waits for input), unless "
                        "--interactive, which asks y/N with a timeout")
    p.add_argument("--field-empty", choices=("yes", "no"),
                   help="the operator's answer as an argument: yes = --empty-confirmed, no = "
                        "not taught (exit 1)")
    p = cmd("check-presence", cmd_check_presence, "fire a taught sensor: exit 0 PRESENT, "
                                                  "1 ABSENT / INDETERMINATE",
            "the part-present sensor")
    p.add_argument("name")
    p.add_argument("--threshold", type=float, help="override the taught fraction")
    p.add_argument("--min-height", type=float, metavar="MM", help="override the taught bgs height")
    p.add_argument("--min-relief-px", type=float, metavar="PX",
                   help="bgs without any mm scale: decide on DMD px of relief")
    p.add_argument("--label", help="output label (default <name>_check)")
    p = cmd("sensors", cmd_sensors, "list taught virtual sensors")
    p.add_argument("--geometry", help="only this geometry id")
    p = cmd("teach-align", cmd_teach_align, "store the reference pose of the part in a scan",
            "the alignment sensor")
    p.add_argument("name")
    p.add_argument("scan", help="scan id / label / latest")
    p = cmd("check-align", cmd_check_align, "dx / dy / dtheta vs the taught pose: exit 0 "
                                            "ALIGNED, 1 otherwise", "the alignment sensor")
    p.add_argument("name")
    p.add_argument("scan", help="scan id / label / latest")
    p.add_argument("--tol-mm", type=float, default=1.0, help="position tolerance, mm (default 1)")
    p.add_argument("--tol-deg", type=float, default=2.0, help="angle tolerance, deg (default 2)")
    p.add_argument("--tol-px", type=float, help="position tolerance, camera px (uncalibrated)")

    # library listings
    p = cmd("calibrations", cmd_calibrations, "list calibrations (the one in use is marked)")
    p.add_argument("--geometry", help="only this geometry id")
    p = cmd("calib-poses", cmd_calib_poses, "list calibration board poses (set, corners, "
                                            "projector corners, fringe periods)")
    p.add_argument("--geometry", help="only this geometry id")
    p.add_argument("--set", help="only this pose set")
    cmd("readings", cmd_readings, "list code readings and spectral readings")
    cmd("materials", cmd_materials, "list learned materials")
    p = cmd("outputs", cmd_outputs, "list the output bin (every derived file)")
    p.add_argument("--kind", help="pointclouds | measurements | defects | previews | ...")
    p.add_argument("--last", type=int, default=50, help="newest N (0 = all)")
    p = cmd("wizard", cmd_wizard, "guided base -> part -> measure -> inspect -> 3D", "the wizard")
    p.add_argument("--label", default="part", help="default part label")
    p.add_argument("--base-label", default="empty")
    return ap


# ------------------------------------------------------------ errors + main

def _classify_error(e: BaseException, mode: str) -> tuple:
    msg = _one_line(e)
    if isinstance(e, NotImplementedError):
        return EXIT_NOT_AVAILABLE, f"{mode} is not available yet" + (f" ({msg})" if msg else "")
    if isinstance(e, rules.RigBusy):               # nothing was sent to the rig
        return EXIT_REFUSED, (f"RigBusy: {msg} | hint: another program holds the rig - wait or "
                              "use the web UI job queue")
    if isinstance(e, rules.RuleViolation):         # Safety / Geometry / Overwrite / Rule
        return EXIT_REFUSED, f"{type(e).__name__}: {msg}"
    if isinstance(e, config.ConfigError):
        return EXIT_REFUSED, f"ConfigError: {msg}"
    if isinstance(e, CLIError):
        return EXIT_REFUSED, f"error: {msg}"
    if isinstance(e, Abort):
        return EXIT_NOT_OK, f"stopped: {msg}"
    if isinstance(e, ValueError):                  # engine precondition / bad value
        return EXIT_REFUSED, f"{type(e).__name__}: {msg}"
    if isinstance(e, KeyboardInterrupt):
        return 130, "interrupted"
    if isinstance(e, ModuleNotFoundError) and e.name == "PySpin":
        from engine.capture.camera import PYSPIN_MISSING
        return EXIT_NOT_AVAILABLE, (msg if msg.startswith("real rig needs PySpin") else
                                    PYSPIN_MISSING.format(exe=sys.executable))
    return EXIT_NOT_OK, f"unexpected {type(e).__name__}: {msg} (run with --debug for details)"


def _exit_code_for(e: BaseException, code: int, args) -> int:
    """The process exit code for an error _classify_error rated `code`: for commands whose
    exit 1 is a verdict (run, dent-detect, station-check ...) an unexpected error exits with
    their unexpected_exit (4), never 1."""
    if code == EXIT_NOT_OK and not isinstance(e, Abort):
        return getattr(args, "unexpected_exit", code)
    return code


def _safe_streams() -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass


# options whose value is a comma list that may start with a minus ("--band -8,-1"): argparse
# takes "-8,-1" for an option and stops with "expected one argument", so it is joined first
_NEG_LIST_OPTS = frozenset({"--band", "--cell-band", "--ct-band", "--edge-band", "--offset"})
_NEG_LIST = re.compile(r"^-(\d+\.?\d*|\.\d+)([,\s]+-?(\d+\.?\d*|\.\d+))*$")


def _join_negative_lists(argv: list) -> list:
    out, i = [], 0
    while i < len(argv):
        a = argv[i]
        if a in _NEG_LIST_OPTS and i + 1 < len(argv) and _NEG_LIST.match(str(argv[i + 1])):
            out.append(f"{a}={argv[i + 1]}")
            i += 2
            continue
        out.append(a)
        i += 1
    return out


def main(argv: Optional[list] = None) -> int:
    _safe_streams()
    parser = build_parser()
    try:
        raw = list(sys.argv[1:] if argv is None else argv)
        args = parser.parse_args(_join_negative_lists(raw))
    except SystemExit as e:                     # --help (0) or a usage error (2)
        return e.code if isinstance(e.code, int) else (0 if e.code is None else EXIT_REFUSED)
    if getattr(args, "func", None) is None:
        parser.print_help()
        return EXIT_REFUSED
    args.preset_given = any(t == "--preset" or t.startswith("--preset=") for t in raw)
    args.mock_part_given = any(t == "--mock-part" or t.startswith("--mock-part=") for t in raw)
    ctx = Ctx(args)
    try:
        ctx.start()
        code, data = args.func(ctx)
        if ctx.json:
            doc = {"ok": code == EXIT_OK, "exit_code": code, "command": args.command,
                   **ctx.meta(), **data, "outputs_written": ctx.new_outputs()}
            run = data.get("result") if isinstance(data, dict) else None
            if (args.command in DISPOSITION_COMMANDS and "disposition" not in doc
                    and isinstance(run, dict)):     # run / recipe-run: top level, as the others
                doc.update(disposition=run.get("disposition") or "PULL",
                           pull_reasons=run.get("pull_reasons") or [])
            print(json.dumps(_jsonable(doc), indent=2))
        else:
            ctx.report_outputs()
        return code
    except (Exception, KeyboardInterrupt) as e:
        code, line = _classify_error(e, ctx.mode)
        code = _exit_code_for(e, code, args)                    # a crash is not a verdict
        if args.debug:
            traceback.print_exc()
        print(line.translate(_ASCII), file=sys.stderr)
        if ctx.json:
            print(json.dumps(_jsonable({"ok": False, "exit_code": code, "command": args.command,
                                        **ctx.meta(),
                                        "error": {"type": type(e).__name__, "message": line},
                                        **_error_disposition(e, args.command, code, line)}),
                             indent=2))
        return code


DISPOSITION_COMMANDS = ("run", "recipe-run", "quick-check", "station-check", "face-check")


def _error_disposition(e: BaseException, command: str, code: int, line: str) -> dict:
    """--json of a refused / crashed run, recipe-run, station-check or face-check: its
    disposition, never missing - the run's own (result.json) when it wrote one, else PULL
    with `refused` / `error` (it never finished). Other commands: nothing."""
    if command not in DISPOSITION_COMMANDS:
        return {}
    res = getattr(e, "run_result", None)
    if isinstance(res, dict) and res.get("disposition") in D.DISPOSITIONS:
        return {"disposition": res["disposition"], "pull_reasons": res.get("pull_reasons") or [],
                "run_id": res.get("run_id"), "run_dir": getattr(e, "run_dir", None)}
    r = {"result": "INVALID", "invalid_reason": line, "pull_reasons": []}
    D.add(r, "refused" if code == EXIT_REFUSED else "error", line)
    D.settle(r)
    return {"disposition": r["disposition"], "pull_reasons": r["pull_reasons"]}


def _exit(code: int) -> None:
    """Leave with `code` and nothing else: flush, then skip interpreter
    teardown. Native extensions (the camera SDK, decoder libraries) can crash
    in their destructors at shutdown and turn a finished run into a bogus
    exit status (seen on the rig: `codes` found a code, printed it, and the
    process still exited 255)."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:                           # noqa: BLE001 - closed / broken pipe
            pass
    logging.shutdown()
    os._exit(code if isinstance(code, int) and 0 <= code <= 255 else EXIT_NOT_OK)


if __name__ == "__main__":
    _exit(main())

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
  5  run only: the job ran but its result is INVALID (numbers not trusted)
`run <station config>` is the flow-ready entry point: one job, a new run
folder with result.json + run.log.jsonl (ui/run_job.py).
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

DEFAULT_PRESET = "cell_wall_2p7ft_vstripe"   # fringes vertical in camera, projector right
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


def _positive_int(text: str) -> int:
    try:
        v = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not an integer") from None
    if v < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return v


CYL_DEFAULT = (24.05, 90.0, 1.5)        # radius, height, dent depth (mm): a standing cell


def _mock_part(text: str) -> str:
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

    def preset(self):
        if self._preset is None:
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
        desc = f"a standing cell (R {r:g}, H {hh:g} mm" + \
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
    rows, data = [], []
    for name in config.list_presets():
        mark = "*" if name == ctx.args.preset else ""
        try:
            p = config.load_preset(name, store=ctx.session.store, mock=ctx.mock)
        except (config.ConfigError, rules.RuleViolation) as e:
            rows.append([mark, name, f"INVALID: {_one_line(e)}", "", "", "", "", ""])
            data.append({"name": name, "valid": False, "error": str(e)})
            continue
        cal = p.calibration
        cal_txt = (f"{cal.id} ({'Z' if cal.has_z else ''}{'+XY' if cal.has_xy else ''})"
                   if cal else "none -> px")
        w, h = p.projection.size
        rows.append([mark, p.name, p.geometry_id, p.rig.projector_side, f"{w}x{h} px",
                     f"{p.camera.exposure_us} us, avg {p.camera.average}", _bases_text(p.bases),
                     cal_txt])
        data.append({"name": p.name, "valid": True, "geometry_id": p.geometry_id,
                     "projector_side": p.rig.projector_side, "camera": p.camera,
                     "bases": p.bases, "calibration": cal, "warnings": p.warnings})
    ctx.say(f"Presets in {config.PRESETS_DIR}  (* = active; pick with --preset NAME)")
    ctx.table(["", "name", "geometry", "projector", "DMD", "camera", "bases", "calibration"],
              rows)
    return EXIT_OK, {"presets": data, "active": ctx.args.preset}


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
        try:
            ans = input(f"Move {kind} {ident} to trash/? It stays recoverable. [y/N] ")
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
    ctx.header()
    p = ctx.preset()
    _mock_clear(ctx)
    if not a.empty_confirmed:
        # clearing the field is physical and human-only: the operator says it is done
        if ctx.json:
            raise CLIError("teach-presence with --json needs --empty-confirmed (no interactive "
                           "prompt): the operator confirms the sensor's field is EMPTY")
        try:
            ans = input(f">> Is the sensor's field EMPTY (nothing in the light at "
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
    rows = [[c.get("id"), c.get("geometry_id"),
             _u(c.get("mm_per_px"), "mm/px", 5) if c.get("mm_per_px") is not None else "-",
             _u(c.get("lateral_px_per_mm"), "px/mm", 4)
             if c.get("lateral_px_per_mm") is not None else "-",
             _f(c.get("r2"), 5), c.get("n_samples"),
             "<- in use for the preset" if c.get("id") == in_use else ""] for c in cals]
    ctx.say(f"Calibrations ({len(cals)})" + (f" for geometry {ctx.args.geometry}"
                                              if ctx.args.geometry else ""))
    ctx.table(["id", "geometry", "Z", "XY", "R^2", "samples", ""], rows)
    if not cals:
        ctx.say("  none yet: scan a part with --known H,W,L, then  calibrate <label>")
    return EXIT_OK, {"calibrations": cals, "in_use": in_use}


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

def _ask(prompt: str) -> str:
    """input() for a human step; q / EOF stop the wizard cleanly."""
    try:
        ans = input(prompt)
    except EOFError:
        raise Abort("no input (stdin closed); wizard stopped") from None
    if ans.strip().lower() in ("q", "quit", "exit"):
        raise Abort("stopped by the operator")
    return ans


def cmd_wizard(ctx):
    """base -> part -> measure -> inspect -> 3D, prompting for every
    physical step. Always human-readable (the prompts need a person)."""
    ctx.json = False
    a = ctx.args
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
                ["cell height", _u(meta.get("cell_height"), units, 2)]])
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
    ctx.say(f"Cell unwrap of {scan.id}  (units {u.units}, {u.calibration_id})")
    ctx.kv([["seen arc", _u(u.arc_deg, "deg (open arc only)", 1)],
            ["cell height", _u(u.cell_height, u.units, 2)], ["radius", _u(u.radius, u.units, 3)],
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
    ctx.say(f"Cell dents of {scan.id}: {rep.verdict}  (units {rep.units}, seen arc "
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
    r = api.row_parity(ctx.preset())
    ctx.say(f"Diamond row parity: {r.get('parity')}  ({r.get('note')})")
    ctx.kv([["shift", _u(r.get("shift_cols"), "columns", 3)],
            ["camera px / column", _f(r.get("camera_px_per_column"), 3)],
            ["lines matched", r.get("n_lines", "-")]])
    return (EXIT_OK if r.get("ok") else EXIT_NOT_OK), r


def cmd_calib_pose(ctx):
    a = ctx.args
    r = api.capture_calib_pose(ctx.preset(), a.set, steps=a.steps, periods=_periods(a.periods))
    ctx.say(f"Calibration pose {r['id']} (set {r['set']}): {r['corners_found']} ChArUco corners, "
            f"{r['valid_pct']:.1f}% decoded, parity {r['diamond_parity']} ({r['parity_source']})")
    ctx.warn(r.get("warnings"))
    return (EXIT_OK if r["corners_found"] >= 6 else EXIT_NOT_OK), r


def cmd_calibrate_stereo(ctx):
    a = ctx.args
    if a.mock_poses and not ctx.mock:
        raise ValueError("--mock-poses needs --mock (on the rig: calib-pose per board pose)")
    r = api.calibrate_stereo(ctx.preset(), a.set, a.square, mock_poses=a.mock_poses or 0,
                             steps=a.steps, periods=_periods(a.periods),
                             projector_distortion=a.projector_distortion)
    rms = r["rms"]
    ctx.say(f"Stereo calibration {r['calibration_id']}  ({r['n_poses_used']} of "
            f"{r['n_poses_captured']} poses, set {r['set']})")
    ctx.kv([["camera RMS", _u(rms.get("camera_px"), "px", 4) + "  (target < 0.3)"],
            ["projector RMS u / v", f"{_f(rms.get('projector_u_px'), 4)} / "
                                    f"{_f(rms.get('projector_v_px'), 4)} px  (target < 0.5)"],
            ["stereo RMS", _u(rms.get("stereo_px"), "px", 4)],
            ["baseline", _u(r["baseline_mm"], "mm", 2)], ["angle", _u(r["angle_deg"], "deg", 2)],
            ["f_v / f_u (diamond ~2)", _f(r["fv_over_fu"], 4)],
            ["diamond parity", r["diamond_parity"]]])
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
    return EXIT_OK, {"metric_scan": ms}


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
    return (EXIT_OK if r["verdict"] == "GO" else EXIT_NOT_OK), r


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


# ------------------------------------------------ dent detection (shiny cells)

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
    ctx.warn(ds.warnings)
    return EXIT_OK, {"dentset": ds}


def cmd_dent_null(ctx):
    a = ctx.args
    sets = [_find_dentset(k) for k in a.dentsets]
    r = api.calibrate_dent_null(sets, a.diameter, a.label, _metric_arg(a.metric), a.fixture,
                                a.subset)
    ctx.say(f"Dent null calibration {r['id']}  (cell {r['cell_diameter_mm']:g} mm, fixture "
            f"{r.get('fixture') or '-'}, patterns {r['subset']})")
    ctx.table(["finder", "good-cell max (T)"],
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
            f"{_f(thr, 2)}; max score {_f(rep.max_score, 2)} x the good-cell maximum; null "
            f"{rep.null_calibration_id}, cell {rep.cell_diameter_mm:g} mm)")
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
                "good-cell maximum)")
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


# ------------------------------------------------ end faces (a cell's end face toward the camera)

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


def _print_face(ctx, t) -> None:
    a, e = t.affine, t.camera_ellipse
    ctx.say(f"Face target {t.id}  (face {t.face_diameter_mm:g} mm, from metric scan {t.source_scan})")
    ctx.kv([["centre", "(" + ", ".join(f"{v:.2f}" for v in t.centre_mm) + ") mm, camera frame"],
            ["tilt", _u(t.tilt_deg, "deg", 1)],
            ["rim fit", f"{_u(t.rim_rms_mm, 'mm', 3)} rms over {t.n_rim_points} rim points "
                        f"({t.meta.get('rim_source') or 'depth'})"],
            ["camera ellipse", f"({e['cx']:.1f}, {e['cy']:.1f}) px, {e['a']:.1f} x {e['b']:.1f} px"],
            ["camera -> DMD affine", f"residual col {_f(a['rms_col_px'], 3)} / row "
                                     f"{_f(a['rms_row_px'], 3)} DMD px over {a['n_px']} face px"],
            ["DMD centre", f"({t.dmd_centre[0]:.1f}, {t.dmd_centre[1]:.1f}) (col, row)"],
            ["stored", str(t.path)]])
    ctx.warn(t.warnings)


def cmd_face_locate(ctx):
    a = ctx.args
    if a.metric:
        ms = _find_metric(a.metric)
    elif a.zone:
        ms = api.scan_metric(ctx.preset(), f"{a.label}_locate", "plane", a.steps, None, None, a.zone)
        ctx.say(f"Locate scan {ms.id} (zone {a.zone}, ray-plane): {ms.meta['n_points']} points")
    else:
        raise CLIError("face-locate needs a metric scan (id / label / latest) or --zone to scan one")
    params = _kw(near_band_mm=a.near_band, far_gap_mm=a.far_gap, near_gap_mm=a.near_gap)
    if ms.field.corr_x is None or ms.field.corr_y is None:
        raise CLIError(f"metric scan {ms.id} has only one decoded axis (x-only); face-locate needs "
                       "x and y - rescan with `scan-fast --axis xy` (or scan-metric)")
    t = api.locate_face(ms, a.diameter, a.label, _roi(a.roi), params or None, _flood_arg(a.flood))
    _print_face(ctx, t)
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
    fs = api.targeted_face_scan(face, p, a.label, a.repeats, a.margin, not a.no_polar, duty,
                                a.steps, _periods(a.periods), a.polar_steps, mode=a.mode,
                                code_bits=a.code_bits)
    ctx.say(f"Face scan {fs.id} (face {fs.face_id}, {len(fs.metric_scan_ids)} repeat(s), polar "
            f"{'yes' if fs.polar else 'no'}, duty {fs.meta.get('duty'):g}, depth set "
            f"{a.mode}: {fs.meta.get('depth_frames')} frames)")
    ctx.table(["repeat", "metric scan", "face decoded", "capture"],
              [[i + 1, r["metric_scan"], _u(r["face_decoded_pct"], "%", 1), _u(r["capture_s"], "s", 1)]
               for i, r in enumerate(fs.repeats)])
    ctx.kv([["stored", str(fs.path)]])
    ctx.warn(fs.warnings)
    return EXIT_OK, {"facescan": fs}


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
    pr = api.face_profile(scans, face, a.label, a.grid, a.r_flat, not a.no_stl)
    st = pr.stats
    ctx.say(f"Face profile {pr.label}: {st['n_repeats']} repeat(s), face {pr.face_id} "
            f"({pr.face_diameter_mm:g} mm), grid {pr.grid_mm:g} mm")
    ctx.kv([["flatness (r < %g mm)" % st["r_flat_mm"], f"{_u(st['flatness_rms_mm'], 'mm', 4)} rms, "
                                                        f"{_u(st['flatness_p2_p98_mm'], 'mm', 3)} p2-p98"],
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
    return EXIT_OK, {"profile": pr}


def cmd_face_golden(ctx):
    a = ctx.args
    places = [_placement(k) for k in a.placements]
    synthetic = False if a.no_synthetic else ({"positions": a.positions} if a.positions else True)
    g = api.face_golden(places, a.label, **_kw(k_sigma=a.k, floor_mm=a.floor,
                                               smooth_cells=a.smooth_cells, rim_r_mm=a.rim_r,
                                               rim_factor=a.rim_factor, rim_min_mm=a.rim_min,
                                               height_smooth_mm=a.height_smooth,
                                               min_area_mm2=a.min_area, detector=a.detector,
                                               z_thresh=a.z, min_depth_mm=a.min_depth),
                        synthetic=synthetic, fir=not a.no_fir)
    sp = g.meta.get("placement_spread") or {}
    prm = g.params
    det = prm.get("detector", "band")
    band, mf, fir = prm.get("band", prm), prm.get("mf") or {}, prm.get("fir") or {}
    ctx.say(f"Golden face {g.id}: {g.n_placements} placements, face {g.face_diameter_mm:g} mm, "
            f"calibration {g.calibration_id}; default detector {det}")
    fit = fir.get("fit") or {}
    ctx.kv([["placement spread", f"{_u(sp.get('sigma_rms_mm'), 'mm', 4)} sigma rms, pairs "
                                 f"{_u(sp.get('pair_rms_median_mm'), 'mm', 4)} median rms"],
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
                       a.detector, a.z, a.min_depth)
    ctx.say(f"Face check of {r.placement['label']}: {r.verdict} ({r.n_triggers} trigger(s); "
            f"golden {r.golden_id}, detector {r.detector})")
    st = r.stats
    if r.detector in ("mf", "mf_fir"):
        ctx.table(["#", "zone", "r mm", "theta deg", "depth mm", "width mm", "z", "dev mm",
                   "area mm^2"],
                  [[i, t["zone"], _f(t["r_mm"], 2), _f(t["theta_deg"], 1), _f(t["depth_mm"], 3),
                    _f(t["width_mm"], 1), _f(t["z"], 2), _f(t.get("dev_mm"), 3), _f(t["area_mm2"], 2)]
                   for i, t in enumerate(r.triggers, 1)])
        nf = st.get("noise_floor_mm") or {}
        ctx.kv([["dent z", f"max {_f(st.get('max_z'), 2)} (trigger >= {_f(st.get('z_thresh'), 2)})"],
                ["noise floor (1 sigma depth)", ", ".join(f"{_u(v, 'mm', 3)} at {k} mm"
                                                          for k, v in nf.items()) or "-"],
                ["reflectance FIR", "applied" if st.get("fir_applied") else "not applied"]])
    else:
        ctx.table(["#", "zone", "r mm", "theta deg", "dev mm", "area mm^2", "band mm"],
                  [[i, t["zone"], _f(t["r_mm"], 2), _f(t["theta_deg"], 1), _f(t["dev_mm"], 3),
                    _f(t["area_mm2"], 2), _f(t.get("band_mm"), 3)] for i, t in enumerate(r.triggers, 1)])
    ctx.kv([["deviation", f"{_u(st['dev_rms_mm'], 'mm', 4)} rms, max |dev| "
                          f"{_u(st['max_abs_dev_mm'], 'mm', 3)} (r < {_f(st.get('dev_r_mm'), 2)} mm)"],
            ["compared", _u(st["compared_pct"], "% of the inspected face", 1)],
            *[[k, str(v)] for k, v in r.files.items() if v is not None]])
    ctx.warn(r.warnings)
    return (EXIT_OK if r.verdict == "PASS" else EXIT_NOT_OK), {"check": r}


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


# ------------------------------------------------ HDMI fast zone scan + station (cells + CT)

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
    ms = api.scan_fast(zone, ctx.preset(), a.label, a.steps, _periods(a.periods), avg, a.hold,
                       a.axis, faces, a.margin)
    m = ms.meta
    ctx.say(f"Fast scan {ms.id}: {m['n_points']} points in mm ({m['method']}), zone {m['zone']}")
    ctx.kv([["patterns", f"{m['n_patterns']} ({m['steps']} steps x periods "
                         f"{', '.join(f'{p:g}' for p in m['periods_px'])} DMD px + floods)"],
            ["capture", f"{_u(m['capture_s'], 's', 2)} ({_u(m['per_pattern_s'], 's', 3)} per pattern; "
                        f"average {m['average']}, hold {m['min_hold_s']:g} s)"],
            ["valid", f"{m['valid_pct']:.1f} %"], ["flood pair", "kept (flood.npz)"]])
    ctx.warn(m.get("warnings"))
    return EXIT_OK, {"metric_scan": ms}


def _flood_arg(key: Optional[str]):
    """--flood: a .npy / .npz file, or a metric scan (id / label) whose flood pair is used."""
    if not key:
        return None
    if Path(key).is_file():
        return str(Path(key).resolve())
    return _find_metric(key)


def _station_params(a, band: str = "band", flag: str = "--band") -> dict:
    """Station parameters from the options; `band` names the CT edge height band option
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
    return prm


def _print_station(ctx, m) -> None:
    ctx.table(["cell", "camera px", "height mm", "flatness mm", "coverage", "outline mm"],
              [[c["name"], f"({c['camera_px'][0]:.1f}, {c['camera_px'][1]:.1f})",
                _f(c["height_mm"], 4), _f(c["flat_rms_mm"], 4), _f(c["coverage"], 2),
                _f(c["outline_diameter_mm"], 2)] for c in m.cells])
    ctx.table(["pair", "height diff mm", "spacing mm", "face angle deg"],
              [[p["pair"], _f(p["height_diff_mm"], 4), _f(p["spacing_inplane_mm"], 4),
                _f(p["face_angle_deg"], 3)] for p in m.pairs])
    ct = m.ct or {}
    if ct.get("trap_fit"):
        t = ct["trap_fit"]
        ctx.kv([["CT edge", f"{ct['n_px']} px, band {ct['band_mm']} mm ({ct['band_source']}), "
                            f"median {_u((ct.get('edge_height_all') or {}).get('median_mm'), 'mm', 3)}"],
                ["CT wave", f"pitch {_u(t['pitch_mm'], 'mm', 3)}, amplitude {_u(t['amp_mm'], 'mm', 3)}, "
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
    ctx.say(f"Station {m.label}: scan {m.scan_id}, {len(m.cells)} cells, "
            f"{sum(1 for z in m.zones if z.get('included'))} CT zones (flood: {m.flood_source})")
    _print_station(ctx, m)
    for x in (m.meta or {}).get("set_aside") or []:
        ctx.say(f"  set aside at camera px ({x['camera_px'][0]:.0f}, {x['camera_px'][1]:.0f}): "
                f"{x['why']}")
    ctx.warn(m.warnings)
    return EXIT_OK, {"station": m}


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
                                 cell_height_band_mm=(_numbers(a.cell_band, 2, "--cell-band")
                                                      if a.cell_band else None),
                                 ct_zone_band_mm=(_numbers(a.ct_band, 2, "--ct-band")
                                                  if a.ct_band else None),
                                 bands=bands or None, flood=_flood_arg(a.flood)))
    ctx.say(f"Station golden {g.id}: {g.n_measurements} measurement(s), "
            f"{len(g.nominal['zones'])} CT zones, calibration {g.calibration_id}")
    ctx.table(["pair", "height diff mm", "spread", "spacing mm"],
              [[p["pair"], _f(p["height_diff_mm"], 4), _f(p["height_diff_spread_mm"], 4),
                _f(p["spacing_inplane_mm"], 4)] for p in g.nominal["pairs"]])
    ctx.table(["zone", "k", "edge height mm", "spread", "n"],
              [[z["label"], z["k"], _f(z["median_mm"], 3), _f(z["spread_mm"], 3), z["n"]]
               for z in g.nominal["zones"]])
    lim = g.limits
    ctx.kv([["limits", f"cell height +-{_f(lim['cell_height_tol_mm'], 3)} mm"
                       + (f" and within {lim['cell_height_band_mm']}" if lim["cell_height_band_mm"] else "")
                       + f"; CT zone +-{_f(lim['ct_zone_tol_mm'], 3)} mm"
                       + (f" and within {lim['ct_zone_band_mm']}" if lim["ct_zone_band_mm"] else "")
                       + (f"; spacing +-{lim['spacing_tol_mm']:g} mm" if lim["spacing_tol_mm"] else "")
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
    r = api.station_check(_station_item(a.scan), gold, _flood_arg(a.flood), a.label)
    ctx.say(f"Station check of {r.measurement['scan_id']}: {r.verdict} ({r.n_triggers} "
            f"trigger(s); golden {r.golden_id})")
    ctx.table(["item", "value mm", "nominal mm", "band mm", "status"],
              [[it["item"], _f(it["value_mm"], 3), _f(it["nominal_mm"], 3),
                "-" if not it.get("band_mm") else
                f"{_f(it['band_mm'][0], 3)} .. {_f(it['band_mm'][1], 3)}", it["status"]]
               for it in r.items])
    ctx.kv([[k, str(v)] for k, v in r.files.items() if v is not None])
    ctx.warn(r.warnings)
    return (EXIT_OK if r.verdict == "PASS" else EXIT_NOT_OK), {"check": r}


def cmd_stations(ctx):
    gs = api.list_library("station_golden", ctx.args.geometry)
    ctx.say("Station goldens:")
    ctx.table(["id", "measurements", "cells", "zones", "calibration"],
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
            ["TriggerMode", f"{r['trigger_mode_restored']} (restored)"]])
    ctx.warn(r.get("notes"))
    return (EXIT_OK if r["verdict"] == "OK" else EXIT_NOT_OK), r


def cmd_scan_trigger(ctx):
    a = ctx.args
    from engine.capture.patternmode import ACK_PHRASE
    if not a.pattern_mode_set_by_human:
        raise CLIError("scan-trigger needs --pattern-mode-set-by-human: a HUMAN has put the "
                       "projector in Pattern Sequence mode (flash, TRIG_OUT_1 wired to the "
                       "camera) in the 4500 MKII GUI. The engine never switches the projector "
                       "mode (docs/TRIGGER_FULLSPEED_TEST.md)")
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
        shift = _numbers(a.mock_shift, 2, "--mock-shift dx,dy (camera px)")
    r = api.rig_check(ctx.preset(), a.set_reference, a.tol, shift)
    ctx.say(f"Rig check: {r['status']}  {r['message']}")
    rows = [["reference", r.get("reference_id") or "-"], ["base", r.get("base_id") or "-"],
            ["stereo calibration", r.get("calibration_id") or "-"]]
    for k in ("flood", "pattern"):
        v = r.get(k)
        if v:
            rows.append([f"{k} shift", f"dx {v['dx']:+.2f}, dy {v['dy']:+.2f} px "
                                       f"(|{v['shift_px']:.2f}| px, match {v['peak']:.2f})"])
    if r.get("shift_mm") is not None:
        rows.append(["~mm", f"{r['shift_mm']:.3f} mm at {r.get('working_distance_mm')} mm"])
    ctx.kv(rows)
    ctx.warn(r.get("warnings"))
    return (EXIT_OK if r["status"] in ("OK", "REFERENCE_SET") else EXIT_NOT_OK), r


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
    golden = _find_scan(cfg.golden)
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


CALIBRATE_STEREO_HOWTO = ("calibrate it first: `scan charuco` (print at 100 %, caliper a "
                          "square), `scan calib-pose` per board pose (12-20, tilted +-30 deg), "
                          "then `scan calibrate-stereo --square MM` (mock: `scan --mock "
                          "calibrate-stereo --mock-poses 12`)")
METRIC_MAX_PCT = 99.9           # 'max' deviation: robust to isolated triangulation outliers


def _mock_place_metric(ctx) -> None:
    """Mock only: the stereo (mm) bench. validation / none = the empty wall (flat),
    H,W,L = a block standing out of the wall, cylinder[:R,H,dent] = a standing cell."""
    if not ctx.mock:
        return
    from engine.capture.mock_stereo import Block, MetricCylinder
    bench = api.mock_bench()
    bench.clear()
    bench.use_stereo()
    spec = ctx.args.mock_part
    if spec in ("none", "validation") or spec.startswith("qr:"):
        ctx.say("[mock] stereo bench: the empty wall (a flat reference)")
        return
    if spec.startswith("cylinder:"):
        r, hh, dent = (float(v) for v in spec.split(":", 1)[1].split(","))
        dents = ([{"depth_mm": dent, "size_mm": 8.0, "angle_deg": 0.0, "z_mm": 0.0}]
                 if dent > 0 else [])
        bench.place(MetricCylinder(center=(0, 0, 823.0 - r - 1.0), radius=r, height=hh,
                                   dents=dents))
        ctx.say(f"[mock] stereo bench: placed a standing cell (R {r:g}, H {hh:g} mm)")
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
    fit in the roi or a golden metric compare; PASS / FAIL against threshold_mm only."""
    from engine import config as cfgmod
    from engine.calibrate import stereo
    from engine.measure import metric
    from engine.reconstruct import metric_scan
    s = api._engine()
    if cfg.threshold_mm is None:
        res["invalid_reason"] = ("metric jobs need threshold_mm (mm) in the station config: "
                                 "there is no preset default for a metric measurement")
        log.event("refused", level="warning", reason=res["invalid_reason"])
        return
    cal = stereo.latest(s.store, p.geometry_id, cfgmod.geometry_fingerprint(p))
    if cal is None:
        res["calibration"]["same_geometry"] = False
        res["invalid_reason"] = (f"no stereo calibration for geometry {p.geometry_id} (these rig "
                                 f"settings), so no mm (rule 6): {CALIBRATE_STEREO_HOWTO}")
        log.event("refused", level="warning", reason=res["invalid_reason"])
        return
    res["calibration"].update(calibration_id=cal.id, same_geometry=True)
    golden = _find_metric(cfg.golden) if cfg.measure == "golden" else None
    s.check_cancel()
    _mock_place_metric(ctx)
    if cfg.scan == "metric":
        ms = api.scan_metric(p, cfg.part_label, steps=cfg.steps, periods=cfg.periods,
                             zone=cfg.zone)
    else:
        avg = cfg.average if cfg.average is not None else (ctx.args.average or 2)
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
    region = cloud.mask & _roi_mask(cloud.mask.shape, cfg.roi)
    if cfg.measure == "plane":
        P = cloud.xyz[region]
        method = "plane_fit"
        if len(P) < 100:
            res["invalid_reason"] = (f"only {len(P)} points in the roi: nothing to fit a "
                                     "plane to (check the roi, zone and exposure)")
            log.event("measure", level="warning", method=method, n_points=int(len(P)))
            return
        fit, _ = metric.ransac_plane(P, 0.5)
        d = fit.distance(P)
        dev = np.full(cloud.mask.shape, np.nan)
        dev[region] = d
        extra = {"plane_rms_inliers_mm": round(fit.rms, 5)}
    else:
        method = "compare_metric_golden"
        rep_, dev = metric_scan.compare_metric(s, ms, golden)
        dev = np.where(_roi_mask(dev.shape, cfg.roi), dev, np.nan)
        d = dev[np.isfinite(dev)]
        extra = {"golden": golden.id, "icp_rms_mm": rep_["icp"]["rms_mm"],
                 "icp_converged": rep_["icp"]["converged"], "blobs": len(rep_["blobs"])}
        if not rep_["icp"]["converged"]:
            res["warnings"].append("ICP onto the golden did not converge: check the placement")
        if d.size < 100:
            res["invalid_reason"] = (f"only {d.size} points overlap the golden {golden.id} "
                                     "(in the roi): wrong part or placement?")
            log.event("measure", level="warning", method=method, n_points=int(d.size), **extra)
            return
    mx = float(np.percentile(np.abs(d), METRIC_MAX_PCT))
    res["deviation_units"] = "mm"
    res["deviation_mm"].update(max=round(mx, 5), mean=round(float(np.mean(d)), 5),
                               rms=round(float(np.sqrt(np.mean(d * d))), 5), method=method,
                               n_points=int(d.size))
    res["outputs"]["deviation_map"] = _deviation_png(run_dir, dev, cfg.threshold_mm)
    res["result"] = res["engine_verdict"] = "FAIL" if mx > cfg.threshold_mm else "PASS"
    res["invalid_reason"] = None
    log.event("measure", method=method, max_deviation_mm=res["deviation_mm"]["max"],
              rms_mm=res["deviation_mm"]["rms"], n_points=int(d.size),
              threshold_mm=cfg.threshold_mm, max_percentile=METRIC_MAX_PCT, **extra)


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


def cmd_run(ctx):
    """One job from a station config -> a new run folder with result.json + run.log.jsonl."""
    from ui import run_job as RJ
    a = ctx.args
    try:
        cfg = RJ.load_station_config(a.config)
    except RJ.StationConfigError as e:
        raise CLIError(str(e)) from e
    lib = cfg.mock_library if (ctx.mock and cfg.mock_library is not None) else cfg.library
    if lib is not None and not a.library:           # --library on the command line wins
        ctx.session = api.configure(lib.resolve(), mock=ctx.mock)
        ctx._outputs_before = len(ctx.session.store.outputs())
    if cfg.preset and not a.preset_given:           # --preset on the command line wins
        a.preset = cfg.preset
    if cfg.average is not None and a.average is None:
        a.average = cfg.average
    confirmed = bool(a.operator_confirmed or cfg.operator_confirmed)
    run_id, run_dir = RJ.new_run(cfg)
    try:
        folder = str(run_dir.relative_to(Path.cwd()))
    except ValueError:
        folder = str(run_dir)
    res = RJ.empty_result(cfg, run_id, folder, ctx.mock, confirmed)
    log = RJ.RunLog(run_dir / "run.log.jsonl", run_id)
    code = EXIT_INVALID
    try:
        log.event("run_start", station=cfg.station_id, mode=res["mode"], job=cfg.job_type,
                  config=str(cfg.source), operator_confirmed=confirmed)
        if not ctx.mock and not confirmed:
            raise CLIError("the real rig needs a human to confirm the physical setup first "
                           "(part placed, rig untouched since the base): pass "
                           "--operator-confirmed, or set operator_confirmed in the station config")
        ctx.header()
        p = ctx.preset()
        res["capture"].update(preset=p.name, geometry_id=p.geometry_id,
                              exposure_us=p.camera.exposure_us, average=p.camera.average)
        log.event("preset", preset=p.name, geometry_id=p.geometry_id,
                  exposure_us=p.camera.exposure_us, average=p.camera.average)
        if cfg.job_type == "metric":            # stereo calibration: no base involved
            _run_metric(ctx, cfg, p, log, res, run_dir)
            code = {"PASS": EXIT_OK, "FAIL": EXIT_NOT_OK}.get(res["result"], EXIT_INVALID)
            return _run_done(ctx, cfg, run_id, run_dir, res, log, code)
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
        code = {"PASS": EXIT_OK, "FAIL": EXIT_NOT_OK}.get(res["result"], EXIT_INVALID)
    except rules.Cancelled as e:              # the operator pressed ABORT: rig already closed
        res["result"], res["invalid_reason"] = "ABORTED", None
        res["warnings"].append(f"aborted: {e}")
        log.event("aborted", level="warning", message=str(e))
        RJ.write_result(run_dir, res)
        log.event("run_end", result="ABORTED", exit_code=EXIT_NOT_OK, result_json="result.json")
        log.close()
        ctx.say(f"Run {run_id}: ABORTED ({e})", f"  run folder: {run_dir}")
        return EXIT_NOT_OK, {"run_id": run_id, "run_dir": str(run_dir), "result": res}
    except (Exception, KeyboardInterrupt) as e:
        ecode, line = _classify_error(e, ctx.mode)
        res["result"] = "INVALID"
        res["invalid_reason"] = line
        log.event("error", level="error", type=type(e).__name__, message=line,
                  exit_code=EXIT_REFUSED if ecode == EXIT_REFUSED else EXIT_ERROR)
        RJ.write_result(run_dir, res)
        try:                                # callers in-process (the web UI) read the run back
            e.run_result, e.run_dir = res, str(run_dir)
        except AttributeError:
            pass
        log.event("run_end", result="INVALID", exit_code=ecode, result_json="result.json")
        log.close()
        ctx.say(f"Run {run_id}: INVALID ({line})", f"  run folder: {run_dir}")
        raise
    return _run_done(ctx, cfg, run_id, run_dir, res, log, code)


def _run_done(ctx, cfg, run_id, run_dir, res, log, code) -> tuple:
    from ui import run_job as RJ
    res["outputs"]["mesh"] = [o["path"] for o in ctx.new_outputs()
                              if o["path"].lower().endswith((".stl", ".obj", ".ply"))]
    out = RJ.write_result(run_dir, res)
    log.event("run_end", result=res["result"], exit_code=code, result_json=out.name,
              invalid_reason=res["invalid_reason"])
    log.close()
    dev = res["deviation_mm"]["max"]
    ctx.say(f"Run {run_id}: {res['result']}"
            + (f"  ({res['invalid_reason']})" if res["invalid_reason"] else ""))
    ctx.kv([["station", cfg.station_id], ["part", f"{cfg.part_label} ({cfg.part_role})"],
            ["max deviation", _u(dev, "mm", 4) if dev is not None else "-"],
            ["threshold", _u(res["threshold_mm"], "mm", 3)
             if res["threshold_mm"] is not None else "-"],
            ["result.json", str(out)], ["log", str(run_dir / "run.log.jsonl")]])
    ctx.warn(res["warnings"])
    return code, {"run_id": run_id, "run_dir": str(run_dir), "result": res}


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
                        "cylinder[:R,H,dent_mm] (a standing cell, default 24.05,90,1.5) | "
                        "qr:TEXT (a QR label, for codes)")
    p.add_argument("--json", action="store_true", default=d(False),
                   help="print one JSON document (for scripts and AI agents)")
    p.add_argument("--debug", action="store_true", default=d(False),
                   help="show full tracebacks on errors")


EPILOG = """\
typical flow:
  scan signal                      is the setup good? (gc_conf, valid %, focus)
  scan base                        empty scene -> base (the datum)
  scan scan part --known 106.95,56.5,21.9
  scan measure part                H/W/L (px until calibrated, then mm)
  scan calibrate part              fit Z/XY from known dims at this geometry
  scan wizard                      guided: base -> part -> measure -> inspect -> 3D

zones ("scan the zone the part is in"):
  scan map                         camera<->projector map (once per rig position)
  scan grid                        project + photograph a lettered grid (A1, B2, ...)
  scan scan part --zone C4:D5      light and measure only those cells

virtual sensors (the DMD draws the spot, the camera is the receiver):
  scan spot-size 6.5 [--measured K]      6.5 mm spot -> DMD px (cols x rows), nominal vs
                                         measured (K = measured mm / drawn columns)
  scan spot H8 --diameter 6.5            project + measure a spot (or dmd:COL,ROW, cam:X,Y)
  scan teach-presence pe H8 --diameter 6.5 [--method bgs] [--empty-confirmed]
                                         teach on the EMPTY field (asks unless confirmed)
  scan check-presence pe                 exit 0 PRESENT, 1 ABSENT or INDETERMINATE
  scan teach-align fixture <scan>        reference pose of the part in a scan
  scan check-align fixture <scan>        exit 0 ALIGNED, 1 NOT_ALIGNED or INDETERMINATE
  scan sensors                           taught sensors;  delete <name> trashes one

3D meshes, cells, surface maps (work on a stored scan; mm only when calibrated):
  scan mesh <scan> [--kind relief|base|cylinder] [--fmt stl|obj] [--stride 2] [--radius 24.05]
  scan unwrap <scan> [--radius MM]       cylinder fit: seen arc, cell height, dent heatmap
  scan dents <scan> [--trigger MM]       exit 0 PASS, 1 FAIL or INDETERMINATE (open arc only)
  scan maps <scan> [--spectral ID --light ID --golden ID]   height/cavity/normal/... PNGs
  scan checker [--square 32 --shifts 1]  checker corners + local Jacobian (stretch, contrast)

metric (mm from a stereo calibration of the same geometry; refused otherwise):
  scan charuco [--square 25]             printable board (print at 100 %, caliper a square)
  scan parity                            diamond DMD row parity (bench step B3)
  scan calib-pose [--set NAME]           one board pose (repeat 12-20x, tilt +-30 deg)
  scan calibrate-stereo --square MM      camera + projector + stereo; RMS per pose / axis
  scan scan-metric [label]               both-axis scan -> point cloud in mm (+ .ply)
  scan cyldev <metric>                   cylinder fit -> deviation map + dents (mm)
  scan accuracy cylinder --diameter 29.84 [--repeats 5]   exit 0 GO, 1 NO-GO
  scan precision --scan <metric>         predicted sigma_z before scanning
  scan metric-mesh <metric>              STL / OBJ in mm, open surface only
  mock: scan --mock calibrate-stereo --mock-poses 12

dents on shiny cells (needs a stereo calibration; the null calibration is the good cell):
  scan --preset DARK dent-scan good1 --zone H9:I12 --short-preset SHORT   a KNOWN-GOOD cell
  scan dent-null good1 --diameter 21.05     finders' maxima on the good cell (per fixture)
  scan --preset DARK dent-scan cell7 --zone H9:I12 --short-preset SHORT
  scan dent-detect cell7                    exit 0 CLEAR, 1 FLAG, 4 detector error (+ PNG, json)
  scan dentsets                             captured dent sets (legacy folders work too)

end faces (a cell lying with its circular end face toward the camera; stereo calibration first):
  scan --preset P face-locate --zone H10:I12   locate scan (ray-plane) -> face plane, rim circle,
                                            camera -> DMD affine (or face-locate <metric scan>:
                                            it needs x AND y decoded - scan-fast --axis xy or
                                            scan-metric; an x-only scan-fast is refused)
  scan face-aim --hold 30                   ring + cross on the face: the placement guide
  scan face-aim --check                     exit 0 ON_FACE, 1 OFF (the cell moved: re-locate)
  scan face-intensity                       projection.duty for a ~200 DN face (never the LEDs)
  scan face-scan c1 --auto-intensity        only the face lit: 3 repeats + polar set
  scan face-profile c1                      height map, radial, sections, flatness (+ PNG/CSV/STL)
  scan face-golden good1 good2 good3 good4  golden from GOOD placements: MF-CFAR + reflectance
                                            FIR (default) and band, leave-one-out, synthetic dents
  scan face-check c7                        exit 0 PASS, 1 FAIL / INCOMPLETE (dents: depth, width,
                                            z; overlay PNG)
  scan faces                                face targets, face scans, golden faces

HDMI fast zone scan + station (cells with end faces toward the camera + the cooling tube, CT):
  scan --preset P scan-fast st1 --zone F10:J12   columns only, 4 steps x 8/80/1280 + floods,
                                            average 2 (--average N; < 2 is raised: HDMI
                                            flicker), --hold 0.12 s; keeps its flood pair;
                                            --axis xy for a scan face-locate can use
  scan station st1                          cells C1..Cn: heights, Cj-Ci, spacing; CT edge
                                            band, trapezoid wave, zones T1..Tn (+ PNG/JSON/CSV)
  scan station-golden st1 st2 --cell-tol 0.15 --ct-tol 0.3 [--band T3=-5,-4]
  scan station-check st7                    exit 0 PASS, 1 FAIL / INCOMPLETE (triggers named),
                                            2 refused (e.g. a cell not found), 4 crashed
  scan stations                             station goldens

Surfer atlas (regions become zones atlas:<name>/<region>, e.g. scan part --zone atlas:cell/cell):
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
                                         (camera only; exit 0 OK, 1 NO PULSES / RATE MISMATCH)
  scan scan-trigger <label> --pack DIR [--variant full] [--line Line0] [--repeats N]
          --pattern-mode-set-by-human    EXPERIMENTAL hardware-triggered scan, one frame per
                                         pattern (docs/TRIGGER_FULLSPEED_TEST.md)
  scan sort-flash <recording.npz> [--decode]   offline: re-sort a saved recording
                                         (docs/FLASH_SOFTSYNC.md)

rig check (has the camera / projector been nudged? also the operator UI's HOME):
  scan rig-check [--set-reference] [--tol 0.5]   exit 0 OK / reference set, 1 MOVED or UNKNOWN

flow-ready job (one station config in, one run folder out; see stations/):
  inspect / compare: base-relief depth calibration;  metric: stereo calibration (calibrate-stereo)
  scan --mock run stations/example_cell_station.json
  scan run stations/<station>.json --operator-confirmed
                                         result.json + run.log.jsonl in <runs_dir>/<run_id>/;
                                         exit 0 PASS, 1 FAIL, 2 refused, 4 crashed, 5 INVALID

library: bases | scans | calibrations | readings | materials | banks | maps |
         sensors | outputs;  bank-remove <scan> --bank NAME drops one bank member.
mock: --mock-part H,W,L[,albedo] is what the simulated human places for scan,
      wizard and check-presence (the bench is albedo 0.6; 'none' = empty);
      --mock-part cylinder[:R,H,dent_mm] places a standing cell; qr:TEXT a QR
      label for codes.

ids: a full id, a label (newest wins), 'latest', or any unique piece of the id.
exit codes: 0 ok | 1 not ok (verdict/signal/cancel/unexpected) | 2 refused (rule,
geometry, safety, config, rig busy, bad input) | 3 mode not available yet |
4 dent-detect's detector failed (its unexpected errors are not 1 = FLAG) |
5 run: INVALID result.
--json prints one JSON document ({"ok", "exit_code", ...}) for scripts and agents.
"""


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="scan", formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Swiss Army Scanner: DMD structured-light scanner, command line.",
        epilog=EPILOG)
    _add_globals(ap, sub=False)
    common = argparse.ArgumentParser(add_help=False)
    _add_globals(common, sub=True)
    sp = ap.add_subparsers(dest="command", metavar="<command>")

    def cmd(name, func, help_, mode=None):
        p = sp.add_parser(name, parents=[common], help=help_,
                          description=help_.replace("%%", "%"))
        p.set_defaults(func=func, mode=mode or name)
        return p

    cmd("presets", cmd_presets, "list rig presets (geometry, camera, calibration)")
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
    p = cmd("delete", cmd_delete, "move a base / scan / calibration / reading / map / bank / "
                                  "material / sensor to trash/")
    p.add_argument("id")
    p.add_argument("--reason", default="")
    p.add_argument("--yes", "-y", action="store_true", help="do not ask for confirmation")
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
    p = cmd("measure", cmd_measure, "H/W/L of a scan (mm when calibrated, else px)")
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
    p = cmd("unwrap", cmd_unwrap, "cylinder fit + (theta, z) unwrap of a standing cell",
            "the cell unwrap")
    p.add_argument("id")
    p.add_argument("--radius", type=float, metavar="MM", help="known radius, mm")
    p.add_argument("--step", type=float, help="grid step, units")
    p = cmd("dents", cmd_dents, "dents on the unwrapped cell: exit 0 PASS, 1 FAIL / "
                                "INDETERMINATE", "cell dent inspection")
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
    p.add_argument("--periods", help="T,T,T DMD px, fine -> coarse")
    p = cmd("calibrate-stereo", cmd_calibrate_stereo,
            "camera + projector + stereo calibration from a pose set", "stereo calibration")
    p.add_argument("--set", help="pose set (default: newest)")
    p.add_argument("--square", type=float, metavar="MM", help="caliper-measured square, mm")
    p.add_argument("--mock-poses", dest="mock_poses", type=int, default=0,
                   help="MOCK: place N board poses on the simulated bench")
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

    # dent detection on shiny cylindrical cells (mm only with a stereo calibration)
    p = cmd("dent-scan", cmd_dent_scan, "one pose of a cell for the dent detector: long + short "
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
    p = cmd("dent-null", cmd_dent_null, "calibrate the dent finders on KNOWN-GOOD cell(s)",
            "the dent null calibration")
    p.add_argument("dentsets", nargs="+", help="dent set id / label / latest / folder")
    p.add_argument("--diameter", type=float, required=True, metavar="MM", help="cell diameter, mm")
    p.add_argument("--label", default="null")
    p.add_argument("--metric", nargs="*", help="metric scans of the (one) good dent set")
    p.add_argument("--fixture", help="fixture name (default: the dent set's zone)")
    p.add_argument("--subset", choices=("all", "production", "production_nolines", "minimal"),
                   default="all")
    p = cmd("dent-detect", cmd_dent_detect, "dents / scratches on a cell: exit 0 CLEAR, 1 FLAG, "
                                            "4 the detector failed", "dent detection")
    p.set_defaults(unexpected_exit=EXIT_ERROR)
    p.add_argument("dentset", help="dent set id / label / latest / folder")
    p.add_argument("--diameter", type=float, metavar="MM",
                   help="cell diameter, mm (default: that of the newest matching null "
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

    # end faces: a cell lying with its circular END FACE toward the camera
    face_help = "face target id / label / latest (default: the newest at the preset's geometry)"
    p = cmd("face-locate", cmd_face_locate, "locate a circular end face in a metric scan: plane, "
                                            "centre, rim circle, camera -> DMD map", "face locate")
    p.add_argument("metric", nargs="?", help="metric scan id / label / latest (or --zone)")
    p.add_argument("--zone", help="no scan given: scan this zone first (ray-plane), e.g. H10:I12")
    p.add_argument("--steps", type=_positive_int, help="phase steps of that locate scan")
    p.add_argument("--diameter", type=float, default=21.05, metavar="MM",
                   help="face diameter, mm (default 21.05)")
    p.add_argument("--label", default="face")
    p.add_argument("--roi", metavar="X0,Y0,X1,Y1", help="search only these camera px")
    p.add_argument("--near-band", dest="near_band", type=float, metavar="MM",
                   help="depth face: within this of the nearest surface (default 12)")
    p.add_argument("--far-gap", dest="far_gap", type=float, metavar="MM",
                   help="background: this far behind the face (default 30)")
    p.add_argument("--near-gap", dest="near_gap", type=float, metavar="MM",
                   help="a neighbour (cooling tube) this far behind the face plane marks the rim "
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
    p.add_argument("--r-flat", dest="r_flat", type=float, default=9.0, metavar="MM",
                   help="flatness inside this radius (default 9)")
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
                   help="skip the synthetic-dent battery")
    p.add_argument("--positions", type=_positive_int,
                   help="synthetic battery: random dent centres per size per placement (default 6)")
    p.add_argument("--k", type=float, help="band = k x sigma (default 4)")
    p.add_argument("--floor", type=float, metavar="MM", help="band floor (default 0.10)")
    p.add_argument("--smooth-cells", dest="smooth_cells", type=_positive_int,
                   help="sigma median window, cells (default 5)")
    p.add_argument("--rim-r", dest="rim_r", type=float, metavar="MM", help="rim zone r > (9.75)")
    p.add_argument("--rim-factor", dest="rim_factor", type=float, help="rim band x (default 2)")
    p.add_argument("--rim-min", dest="rim_min", type=float, metavar="MM", help="rim band min (0.3)")
    p.add_argument("--height-smooth", dest="height_smooth", type=float, metavar="MM",
                   help="Gaussian sigma on the height maps (default 0.25)")
    p.add_argument("--min-area", dest="min_area", type=float, metavar="MM2",
                   help="band: smallest trigger (default 0.25)")
    p = cmd("face-check", cmd_face_check, "a placement vs the golden face: exit 0 PASS, 1 FAIL / "
                                          "INCOMPLETE (+ overlay PNG, json)", "the face check")
    p.add_argument("placement", help="face scan (id / label / latest) or face-profile .npz")
    p.add_argument("--golden", help="golden face id / label (default: newest matching)")
    p.add_argument("--label")
    p.add_argument("--min-area", dest="min_area", type=float, metavar="MM2")
    p.add_argument("--detector", choices=("mf_fir", "mf", "band"),
                   help="default: the golden's (mf_fir for new goldens)")
    p.add_argument("--z", type=float, help="MF: trigger z for this check (default: the golden's)")
    p.add_argument("--min-depth", dest="min_depth", type=float, metavar="MM",
                   help="MF: ignore dents shallower than this for this check")
    p = cmd("faces", cmd_faces, "list face targets, face scans and golden faces")
    p.add_argument("--geometry", help="only this geometry id")

    # HDMI fast zone scan + station (cells with end faces + cooling tube)
    p = cmd("scan-fast", cmd_scan_fast, "HDMI fast zone scan: columns only (4 steps x 8/80/1280 + "
                                        "floods), average 2, hold 0.12 s -> metric scan + flood "
                                        "pair", "fast scan")
    p.add_argument("label", nargs="?", default="fast")
    p.add_argument("--zone", help='light only these grid cells ("F10:J12") or atlas:<name>/<region>')
    p.add_argument("--mask", metavar="NPY", help="or a DMD mask (1140 x 912 .npy)")
    p.add_argument("--face", action="append", metavar="FACE",
                   help="also light this face target's ellipse (+ --margin); repeatable")
    p.add_argument("--margin", type=float, default=6.0, metavar="DMD_PX")
    p.add_argument("--steps", type=_positive_int, default=4)
    p.add_argument("--periods", help="T,T,T DMD px (default 8,80,1280)")
    p.add_argument("--hold", type=float, default=0.12, metavar="S",
                   help="HDMI hold per pattern for this run (default 0.12 s)")
    p.add_argument("--axis", choices=("x", "xy"), default="x",
                   help="x (default, enough for station heights) or xy (needed by face-locate)")
    station_help = ("station parameters: --band LO,HI (CT edge height band, mm; default auto), "
                    "--roi, --skip-zones L,R, --centre")
    p = cmd("station", cmd_station, "cells (end faces) + cooling-tube edge zones of a metric scan: "
                                    "heights, spacing, T1..Tn (+ PNG, JSON, CSV)", "station")
    p.add_argument("metric", help="metric scan id / label / latest")
    p.add_argument("--flood", help="flood .npy/.npz or a metric scan with a flood pair (default: "
                                   "the scan's own, else its decode modulation)")
    p.add_argument("--label")
    p.add_argument("--cells", type=_positive_int, default=2)
    p.add_argument("--diameter", type=float, default=21.05, metavar="MM")
    p.add_argument("--band", metavar="LO,HI", help=station_help)
    p.add_argument("--roi", metavar="X0,Y0,X1,Y1", help="CT search window, camera px")
    p.add_argument("--skip-zones", dest="skip_zones", metavar="L,R",
                   help="drop this many CT zones at the left / right end (e.g. a lead-in)")
    p.add_argument("--centre", choices=("rim_circle", "ellipse_ray"),
                   help="face centre method (default rim_circle)")
    p.add_argument("--no-ct", dest="no_ct", action="store_true", help="cells only")
    p.add_argument("--no-strict", dest="no_strict", action="store_true",
                   help="drop the strict cell gates (rim outline rms <= 1.0 mm, >= 70 %% of the "
                        "disc measured, a plausible height / spacing to the other cells)")
    p = cmd("station-golden", cmd_station_golden, "station golden from GOOD measurements (metric "
                                                  "scans or station .json) + limits", "station golden")
    p.add_argument("items", nargs="+", help="metric scans (id / label) or station .json files")
    p.add_argument("--label", default="golden")
    p.add_argument("--cell-tol", dest="cell_tol", type=float, metavar="MM",
                   help="cell height difference: nominal +- this (default 0.15)")
    p.add_argument("--ct-tol", dest="ct_tol", type=float, metavar="MM",
                   help="CT zone edge height: nominal +- this (default 0.30)")
    p.add_argument("--spacing-tol", dest="spacing_tol", type=float, metavar="MM")
    p.add_argument("--cell-band", dest="cell_band", metavar="LO,HI",
                   help="absolute band for every cell height difference, mm")
    p.add_argument("--ct-band", dest="ct_band", metavar="LO,HI",
                   help="absolute band for every CT zone height, mm")
    p.add_argument("--band", action="append", metavar="ITEM=LO,HI",
                   help="per-item band, e.g. C2-C1=-0.4,0.1 or T3=-5,-4 (repeatable)")
    p.add_argument("--flood", help="flood for a single metric scan (file or scan)")
    # station parameters for metric-scan items (as `station` takes them; station .json items
    # keep the parameters they were measured with)
    p.add_argument("--cells", type=_positive_int, help="cells per scan (default 2)")
    p.add_argument("--diameter", type=float, metavar="MM", help="face diameter (default 21.05)")
    p.add_argument("--edge-band", dest="edge_band", metavar="LO,HI",
                   help="CT edge height band for the measurement, mm (default auto)")
    p.add_argument("--roi", metavar="X0,Y0,X1,Y1", help="CT search window, camera px")
    p.add_argument("--skip-zones", dest="skip_zones", metavar="L,R",
                   help="drop this many CT zones at the left / right end (the golden's numbering)")
    p.add_argument("--centre", choices=("rim_circle", "ellipse_ray"),
                   help="face centre method (default rim_circle)")
    p.add_argument("--no-ct", dest="no_ct", action="store_true", help="cells only")
    p = cmd("station-check", cmd_station_check, "a metric scan vs the station golden: exit 0 PASS, "
                                                "1 FAIL / INCOMPLETE, 2 refused, 4 crashed (+ PNG, JSON, CSV)",
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
    p.add_argument("--period-us", type=int, help="uniform pattern period, us (8-bit >= ~8563)")
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
    p.add_argument("--camera-exposure", type=int, metavar="US", help="camera exposure, us")
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
    p.add_argument("--line", default="Line0", choices=("Line0", "Line2", "Line3"))
    p.add_argument("--seconds", type=float, default=5.0)
    p.add_argument("--expect-period-ms", type=float, help="the GUI's Pattern Period, ms")
    p.add_argument("--exposure", type=float, default=1000.0, metavar="US",
                   help="camera exposure (short; default 1000 us)")
    p = cmd("scan-trigger", cmd_scan_trigger, "EXPERIMENTAL hardware-triggered metric scan of a "
                                              "flash pack (TRIG_OUT_1 -> camera)",
            "the hardware-triggered scan")
    p.add_argument("label", nargs="?", default="trigger")
    p.add_argument("--pack", metavar="DIR", required=True, help="the pack the GUI plays")
    p.add_argument("--variant", choices=("full", "zone"), help="fast-scan pack image played")
    p.add_argument("--line", default="Line0", choices=("Line0", "Line2", "Line3"))
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
                                        "reference -> camera / projector shift (px, ~mm)",
            "the rig check")
    p.add_argument("--set-reference", action="store_true",
                   help="store a new reference (right after base / calibration)")
    p.add_argument("--tol", type=float, default=0.5, help="MOVED above this shift, px "
                                                         "(default 0.5)")
    p.add_argument("--mock-shift", metavar="DX,DY",
                   help="mock only: shift the fresh frames (simulates a nudged camera)")
    p = cmd("run", cmd_run, "flow-ready job: run a station config, write result.json + "
                            "run.log.jsonl in a new run folder (exit 0 PASS, 1 FAIL, "
                            "2 refused, 4 crashed, 5 INVALID)", "the station job")
    p.add_argument("config", help="station config .json (or .yaml with PyYAML), "
                                  "e.g. stations/example_cell_station.json")
    p.add_argument("--operator-confirmed", action="store_true",
                   help="a human confirmed the physical setup (required on the real rig)")
    p.set_defaults(unexpected_exit=EXIT_ERROR)          # a crash (4) is not a FAIL verdict (1)

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
                   help="the operator has cleared the sensor's field (else asks y/N)")
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
    ctx = Ctx(args)
    try:
        ctx.start()
        code, data = args.func(ctx)
        if ctx.json:
            doc = {"ok": code == EXIT_OK, "exit_code": code, "command": args.command,
                   **ctx.meta(), **data, "outputs_written": ctx.new_outputs()}
            print(json.dumps(_jsonable(doc), indent=2))
        else:
            ctx.report_outputs()
        return code
    except (Exception, KeyboardInterrupt) as e:
        code, line = _classify_error(e, ctx.mode)
        if code == EXIT_NOT_OK and not isinstance(e, Abort):
            code = getattr(args, "unexpected_exit", code)       # a crash is not a verdict
        if args.debug:
            traceback.print_exc()
        print(line.translate(_ASCII), file=sys.stderr)
        if ctx.json:
            print(json.dumps(_jsonable({"ok": False, "exit_code": code, "command": args.command,
                                        **ctx.meta(),
                                        "error": {"type": type(e).__name__, "message": line}}),
                             indent=2))
        return code


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

"""
Flow-ready job runner: one station config file in, one run folder out.

    scan.bat [--mock] run stations/example_cell_station.json [--operator-confirmed]

A station config (JSON, or YAML when PyYAML is installed) holds every
station parameter: station id, preset, job type, thresholds, zone, where
run folders go. Nothing station-specific lives in code. Each run gets a NEW
folder (never an existing one) with

    result.json      the machine-readable result (RESULT_SCHEMA_VERSION)
    run.log.jsonl    one JSON event per line, every line carries run_id

so a script, a CI job or an automation flow can act on the run without
parsing console text. The verdict is only ever PASS / FAIL / INVALID (or
ABORTED when an operator stopped the run): INVALID means the numbers cannot
be trusted (no mm calibration at this geometry when mm are required, an
INDETERMINATE engine verdict, a refused or crashed run) and must never be
reported as a pass or a fail.

Job types and the calibration each needs:
    inspect, compare   base-relief depth calibration (`calibrate <scan>`)
    metric             stereo calibration (`calibrate-stereo`): mm from
                       triangulation, measured as a plane fit or against a
                       golden metric scan
    recipe             stereo calibration: a recipe (engine/recipes/, recipes/*.json or a
                       library recipe) - its preset, scan, zones, checks, projected feedback
                       and outputs; the work state then IS that recipe

This module only builds the config; the run folder, the log and the result document come
from engine/runs.py (shared with api.recipe_run). The measurement itself goes through
engine.api verbs (ui/cli.py cmd_run), so every engine rule still applies.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from engine import runs as _runs
from engine.runs import RESULT_SCHEMA_VERSION, RESULTS, RunLog, software_info  # noqa: F401
from engine.runs import write_result  # noqa: F401  (re-exported: callers use run_job.*)

JOB_TYPES = ("inspect", "compare", "metric", "recipe")
METRIC_SCANS = ("fast", "metric")           # api.scan_fast | api.scan_metric
METRIC_MEASURES = ("plane", "golden")
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_ROOT = Path(__file__).resolve().parents[1]


class StationConfigError(ValueError):
    """The station config is missing, unreadable or invalid (exit 2: refused)."""


@dataclass
class StationConfig:
    station_id: str
    preset: Optional[str] = None
    library: Optional[Path] = None          # None: the CLI's --library / the engine default
    mock_library: Optional[Path] = None     # used instead of `library` with --mock
    runs_dir: Path = Path("runs")
    job_type: str = "inspect"
    base: str = "latest"                    # latest | capture (capture: the scene is empty)
    zone: Optional[str] = None
    average: Optional[int] = None
    golden: Optional[str] = None            # compare: golden scan id / label
    part_role: str = "test"                 # golden | test
    part_label: str = "part"
    threshold_mm: Optional[float] = None    # None: the preset's FAIL trigger
    require_mm: bool = True                 # px-only results are INVALID
    n_sigma: float = 6.0
    min_area_px: int = 50
    operator_confirmed: bool = False        # a human confirmed the physical setup
    # metric jobs (stereo calibration): scan + measure settings
    scan: str = "fast"                      # fast (api.scan_fast) | metric (api.scan_metric)
    axis: str = "xy"                        # scan_fast axis
    steps: Optional[int] = None
    periods: Optional[list] = None
    measure: str = "plane"                  # plane | golden
    roi: Optional[list] = None              # [x0, y0, x1, y1] camera px; None = whole field
    # recipe jobs: a recipe name (built-in recipes/ or the library) or a recipe file
    recipe: Optional[str] = None
    feedback: Optional[bool] = None         # None: the recipe's feedback.enabled
    mock_part: Optional[str] = None         # MOCK only: what the simulated human places
    source: Optional[Path] = None
    raw: dict = field(default_factory=dict, repr=False)


def _read_doc(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml  # optional: PyYAML
        except ImportError as e:
            raise StationConfigError(f"{path.name}: YAML station configs need PyYAML "
                                     "(pip install pyyaml), or use the .json form") from e
        doc = yaml.safe_load(text)
    else:
        try:
            doc = json.loads(text)
        except json.JSONDecodeError as e:
            raise StationConfigError(f"{path.name}: not valid JSON ({e})") from e
    if not isinstance(doc, dict):
        raise StationConfigError(f"{path.name}: the top level must be a mapping")
    return doc


def _rel(base_dir: Path, value) -> Optional[Path]:
    """Config paths are relative to the config file, after env expansion."""
    if value in (None, ""):
        return None
    p = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return p if p.is_absolute() else (base_dir / p)


def load_station_config(path) -> StationConfig:
    path = Path(path)
    if not path.is_file():
        raise StationConfigError(f"station config not found: {path}")
    doc = _read_doc(path)
    base_dir = path.resolve().parent
    sid = str(doc.get("station_id") or "").strip()
    if not _ID.match(sid):
        raise StationConfigError("station_id is required: letters, digits, _ . - "
                                 "(max 64 chars), e.g. example_cell_station")
    job = doc.get("job") or {}
    part = doc.get("part") or {}
    if not isinstance(job, dict) or not isinstance(part, dict):
        raise StationConfigError("'job' and 'part' must be mappings")
    jt = str(job.get("type", "inspect"))
    if jt not in JOB_TYPES:
        raise StationConfigError(f"job.type {jt!r} is not one of {', '.join(JOB_TYPES)}")
    base = str(job.get("base", "latest"))
    if base not in ("latest", "capture"):
        raise StationConfigError(f"job.base {base!r} must be 'latest' or 'capture'")
    if jt == "compare" and not job.get("golden"):
        raise StationConfigError("job.type compare needs job.golden (a golden scan id / label)")
    role = str(part.get("role", "test"))
    if role not in ("golden", "test"):
        raise StationConfigError(f"part.role {role!r} must be 'golden' or 'test'")
    label = str(part.get("label", "part"))
    if not _ID.match(label):
        raise StationConfigError(f"part.label {label!r}: letters, digits, _ . - only")
    metric = _metric_keys(job, jt)
    rec = _recipe_keys(job, jt, base_dir)
    thr = doc.get("threshold_mm")
    if thr is not None and (not isinstance(thr, (int, float)) or thr <= 0):
        raise StationConfigError("threshold_mm must be a positive number (mm) or null")
    avg = job.get("average")
    if avg is not None and (not isinstance(avg, int) or avg < 1):
        raise StationConfigError("job.average must be a positive integer or null")
    runs_env = os.environ.get("SCANNER_RUNS_DIR")
    return StationConfig(
        station_id=sid,
        preset=doc.get("preset") or None,
        library=_rel(base_dir, doc.get("library")),
        mock_library=_rel(base_dir, doc.get("mock_library")),
        runs_dir=Path(runs_env) if runs_env else (_rel(base_dir, doc.get("runs_dir"))
                                                  or base_dir / "runs"),
        job_type=jt, base=base, zone=job.get("zone") or None, average=avg,
        golden=job.get("golden") or None,
        part_role=role, part_label=label,
        threshold_mm=float(thr) if thr is not None else None,
        require_mm=bool(doc.get("require_mm", True)),
        n_sigma=float(job.get("n_sigma", 6.0)),
        min_area_px=int(job.get("min_area_px", 50)),
        operator_confirmed=bool(doc.get("operator_confirmed", False)),
        mock_part=_mock_part(doc.get("mock_part")),
        source=path.resolve(), raw=doc, **metric, **rec)


def _mock_part(v) -> Optional[str]:
    if v in (None, ""):
        return None
    if not isinstance(v, str):
        raise StationConfigError("mock_part must be text (as --mock-part), e.g. "
                                 "scene:tube:misaligned, or null")
    return v


def _recipe_keys(job: dict, jt: str, base_dir: Path) -> dict:
    """job.recipe (+ job.feedback) for recipe jobs: a recipe name or a .json / .yaml file
    (relative to this config)."""
    if jt != "recipe":
        return {}
    r = job.get("recipe")
    if not isinstance(r, str) or not r.strip():
        raise StationConfigError("job.type recipe needs job.recipe: a recipe name (recipes/ or "
                                 "the library, see `scan recipe-list`) or a recipe file")
    r = r.strip()
    if r.lower().endswith((".json", ".yaml", ".yml")):
        r = str(_rel(base_dir, r))
    fb = job.get("feedback")
    if fb is not None and not isinstance(fb, bool):
        raise StationConfigError("job.feedback must be true, false or null (null: the recipe's "
                                 "feedback.enabled)")
    return {"recipe": r, "feedback": fb}


def _metric_keys(job: dict, jt: str) -> dict:
    """job.scan / axis / steps / periods / measure / roi / golden for metric jobs."""
    if jt != "metric":
        return {}
    scan = str(job.get("scan", "fast"))
    if scan not in METRIC_SCANS:
        raise StationConfigError(f"job.scan {scan!r} must be one of {', '.join(METRIC_SCANS)}")
    axis = str(job.get("axis", "xy"))
    if axis not in ("x", "xy"):
        raise StationConfigError(f"job.axis {axis!r} must be 'x' or 'xy'")
    measure = str(job.get("measure", "plane"))
    if measure not in METRIC_MEASURES:
        raise StationConfigError(f"job.measure {measure!r} must be one of "
                                 f"{', '.join(METRIC_MEASURES)}")
    if measure == "golden" and not job.get("golden"):
        raise StationConfigError("job.measure golden needs job.golden (a golden metric scan "
                                 "id / label, or 'latest')")
    steps = job.get("steps")
    if steps is not None and (not isinstance(steps, int) or steps < 3):
        raise StationConfigError("job.steps must be an integer >= 3 or null")
    periods = job.get("periods")
    if periods is not None and (not isinstance(periods, list) or not periods or not all(
            isinstance(v, (int, float)) and v > 0 for v in periods)):
        raise StationConfigError("job.periods must be a list of positive numbers (DMD px) or null")
    roi = job.get("roi")
    if roi is not None:
        if (not isinstance(roi, list) or len(roi) != 4
                or not all(isinstance(v, int) and v >= 0 for v in roi)
                or roi[2] <= roi[0] or roi[3] <= roi[1]):
            raise StationConfigError("job.roi must be [x0, y0, x1, y1] camera px (x1 > x0, "
                                     "y1 > y0) or null")
    return {"scan": scan, "axis": axis, "steps": steps,
            "periods": [float(v) for v in periods] if periods else None,
            "measure": measure, "roi": list(roi) if roi else None}


# ------------------------------------------------------------ run folder + log

def new_run(cfg: StationConfig, now: Optional[datetime] = None) -> tuple:
    """(run_id, run_dir): a fresh folder, never an existing one."""
    return _runs.new_run(cfg.runs_dir, cfg.station_id, now)


# ------------------------------------------------------------ result document

def empty_result(cfg: StationConfig, run_id: str, run_folder: str, mock: bool,
                 operator_confirmed: bool) -> dict:
    """The result skeleton: the same keys every run, filled in as the job goes."""
    job = {"type": cfg.job_type, "base": cfg.base, "zone": cfg.zone,
           "config": str(cfg.source.name) if cfg.source else None}
    if cfg.job_type == "recipe":
        job["recipe"] = cfg.recipe
    return _runs.empty_result(run_id=run_id, run_folder=run_folder, mock=mock,
                              operator_confirmed=operator_confirmed, station=cfg.station_id,
                              job=job, part={"role": cfg.part_role, "label": cfg.part_label},
                              threshold_mm=cfg.threshold_mm, zone=cfg.zone)

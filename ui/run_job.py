"""
Flow-ready job runner: one station config file in, one run folder out.

    scan.bat [--mock] run stations/example_cell_station.json [--operator-confirmed]

A station config (JSON, or YAML when PyYAML is installed) holds every
station parameter: station id, preset, job type, thresholds, zone, where
run folders go, where a part judged REJECT or PULL is copied (bad_part_output,
engine/badparts.py). Nothing station-specific lives in code. This module reads the
SINGLE-JOB form (station_id, preset, job, part): one job, which the engine
treats as a station configuration with one implicit profile ("default") and a
one-step flow (its one work state is that job; engine/flow/schema.py
lift_legacy). A station configuration with a profile library set and a flow of
work states (profiles + flow keys) is read by engine/flow/schema.py and run by
engine/flow/verify.py; `scan run` takes both. Each run gets a NEW
folder (never an existing one) with

    result.json      the machine-readable result (RESULT_SCHEMA_VERSION)
    run.log.jsonl    one JSON event per line, every line carries run_id

so a script, a CI job or an automation flow can act on the run without
parsing console text. The verdict is only ever PASS / FAIL / INVALID (or
ABORTED when an operator stopped the run): INVALID means the numbers cannot
be trusted (no mm calibration at this geometry when mm are required, an
INDETERMINATE engine verdict, a refused or crashed run) and must never be
reported as a pass or a fail. On top of it the disposition (engine/
disposition.py): CONTINUE (PASS), REJECT (a definite FAIL), PULL for quality
inspection (any doubt; metric jobs also a PASS within job.guard_band_pct of
threshold_mm). Exit codes follow it: 0 / 1 / 5.

Job types and the calibration each needs:
    inspect, compare   base-relief depth calibration (`calibrate <scan>`)
    metric             stereo calibration (`calibrate-stereo`): mm from
                       triangulation, measured as a plane fit or against a
                       golden metric scan
    recipe             stereo calibration: a recipe (engine/recipes/, recipes/*.json or a
                       library recipe) - its preset, scan, zones, checks, projected feedback
                       and outputs; the config's one work state is verified by that recipe

This module only builds the config; the run folder, the log and the result document come
from engine/runs.py (shared with api.recipe_run). The measurement itself goes through
engine.api verbs (ui/cli.py cmd_run), so every engine rule still applies.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from engine import badparts as _badparts
from engine import disposition as _disp
from engine import runs as _runs
from engine.flow import schema as _flow
from engine.flow.schema import StationConfigError  # noqa: F401  (one class for both forms)
from engine.runs import RESULT_SCHEMA_VERSION, RESULTS, RunLog, software_info  # noqa: F401
from engine.runs import write_result  # noqa: F401  (re-exported: callers use run_job.*)

JOB_TYPES = ("inspect", "compare", "metric", "recipe")
METRIC_SCANS = ("fast", "metric")           # api.scan_fast | api.scan_metric
METRIC_MEASURES = ("plane", "golden")
# metric verdicts: an isolated out-of-tolerance cluster of fewer camera px than this is an
# outlier (a stray triangulated point) and is dropped; every larger cluster FAILs. Capped: a
# larger setting would quietly stop seeing real defects
METRIC_MIN_DEFECT_PX = 5
METRIC_MAX_MIN_DEFECT_PX = 50
# dropped stray px beyond this share of the measured px -> INVALID (noisy data or a pitted
# surface, never a PASS); capped, so the guard can't be switched off by a config
METRIC_MAX_OUTLIER_PCT = 0.1
METRIC_MAX_OUTLIER_PCT_CAP = 5.0
METRIC_MIN_VALID_PCT = 30.0     # plane: decoded share of the roi below this -> INVALID
METRIC_MIN_COVERAGE_PCT = 95.0  # golden: surface agreeing within threshold, each direction
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_ROOT = Path(__file__).resolve().parents[1]


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
    min_defect_px: int = METRIC_MIN_DEFECT_PX
    max_outlier_pct: float = METRIC_MAX_OUTLIER_PCT
    min_valid_pct: float = METRIC_MIN_VALID_PCT
    min_coverage_pct: float = METRIC_MIN_COVERAGE_PCT
    plane_range_mm: Optional[list] = None   # plane: [near, far] plane-to-camera distance, mm
    guard_band_pct: float = 0.0             # a PASS within this % of threshold_mm: PULL
    # recipe jobs: a recipe name (built-in recipes/ or the library) or a recipe file
    recipe: Optional[str] = None
    feedback: Optional[bool] = None         # None: the recipe's feedback.enabled
    mock_part: Optional[str] = None         # MOCK only: what the simulated human places
    # where a part judged REJECT / PULL is copied (engine/badparts.py): a full folder path,
    # "off", or None (it says nothing: the recipe's, else the computer's)
    bad_part_output: Optional[str] = None
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


def _number(v) -> bool:
    """A real, finite number: not a bool (True would read as 1), not NaN / Infinity."""
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _count(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _flag(doc: dict, key: str, default: bool) -> bool:
    """A true / false switch: only a real boolean (null = the default). A quoted "false"
    or 'no' is refused, never read as true."""
    v = doc.get(key)
    if v is None:
        return default
    if not isinstance(v, bool):
        raise StationConfigError(f"{key} must be true or false (unquoted), got {v!r}")
    return v


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
    if not _flow.is_legacy(doc):
        raise StationConfigError(
            f"{path.name} is a station configuration with profiles and a flow, not a single "
            "job: `scan run` runs its work states (engine/flow/schema.load reads it)")
    return config_from_doc(doc, path.resolve().parent, source=path.resolve())


def config_from_doc(doc: dict, base_dir: Path, source: Optional[Path] = None) -> StationConfig:
    """A single-job station config document (already read: a file's, or one generated for a
    quick check, engine/modes/quick.py) -> StationConfig. Paths in it are relative to
    base_dir."""
    if not isinstance(doc, dict):
        raise StationConfigError("a station config is a mapping (JSON object)")
    base_dir = Path(base_dir)
    sid = str(doc.get("station_id") or "").strip()
    if not _ID.fullmatch(sid):
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
    if not _ID.fullmatch(label):
        raise StationConfigError(f"part.label {label!r}: letters, digits, _ . - only")
    if jt == "compare":
        _golden_ref(job.get("golden"), label)
    metric = _metric_keys(job, jt, label)
    rec = _recipe_keys(job, jt, base_dir)
    thr = doc.get("threshold_mm")
    if thr is not None and (not _number(thr) or thr <= 0):
        raise StationConfigError("threshold_mm must be a positive, finite number (mm) or null")
    avg = job.get("average")
    if avg is not None and (not _count(avg) or avg < 1):
        raise StationConfigError("job.average must be a positive integer or null")
    n_sigma = job.get("n_sigma", 6.0)
    if not _number(n_sigma) or n_sigma <= 0:
        raise StationConfigError("job.n_sigma must be a positive, finite number")
    min_area = job.get("min_area_px", 50)
    if not _count(min_area) or min_area < 1:
        raise StationConfigError("job.min_area_px must be a positive integer")
    try:
        bad_out = _badparts.configured(doc.get(_badparts.KEY), base_dir)
    except ValueError as e:
        raise StationConfigError(str(e)) from None
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
        require_mm=_flag(doc, "require_mm", True),
        n_sigma=float(n_sigma),
        min_area_px=int(min_area),
        operator_confirmed=_flag(doc, "operator_confirmed", False),
        mock_part=_mock_part(doc.get("mock_part")), bad_part_output=bad_out,
        source=Path(source).resolve() if source else None, raw=doc, **metric, **rec)


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


def _golden_ref(golden, part_label: str, what: str = "scan") -> str:
    """job.golden: the id or label of a golden scan. Every run saves its own test scan under
    part.label, so 'latest' / 'last' or that label would compare each part with the previous
    part, not with a golden: refused."""
    if not isinstance(golden, str) or not golden.strip():
        raise StationConfigError(f"job.golden must be the id or label of a golden {what} (a GOOD "
                                 "part scanned once)")
    g = golden.strip()
    if g.lower() in ("latest", "last") or g == part_label:
        raise StationConfigError(f"job.golden {golden!r}: name the golden {what} by its id or "
                                 "label; 'latest' and this station's part.label are the previous "
                                 "run's own test scan, not a golden")
    return g


def _metric_keys(job: dict, jt: str, part_label: str) -> dict:
    """job.scan / axis / steps / periods / measure / roi / golden / min_defect_px /
    max_outlier_pct / min_valid_pct / min_coverage_pct / plane_range_mm / guard_band_pct
    for metric jobs."""
    gb = job.get("guard_band_pct")
    if jt != "metric":
        if gb is not None:              # a guard band that does nothing is refused, not ignored
            raise StationConfigError(
                "job.guard_band_pct is for metric jobs (a recipe job sets pull.guard_band_pct "
                f"in its recipe); job.type {jt} has none")
        return {}
    try:
        gb = _disp.check_pct(0.0 if gb is None else gb, "job.guard_band_pct")
    except ValueError as e:
        raise StationConfigError(str(e)) from None
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
    if measure == "golden":
        _golden_ref(job.get("golden"), part_label, "metric scan")
    steps = job.get("steps")
    if steps is not None and (not _count(steps) or steps < 3):
        raise StationConfigError("job.steps must be an integer >= 3 or null")
    periods = job.get("periods")
    if periods is not None and (not isinstance(periods, list) or not periods or not all(
            _number(v) and v > 0 for v in periods)):
        raise StationConfigError("job.periods must be a list of positive numbers (DMD px) or null")
    roi = job.get("roi")
    if roi is not None:
        if (not isinstance(roi, list) or len(roi) != 4
                or not all(_count(v) and v >= 0 for v in roi)
                or roi[2] <= roi[0] or roi[3] <= roi[1]):
            raise StationConfigError("job.roi must be [x0, y0, x1, y1] camera px (x1 > x0, "
                                     "y1 > y0) or null")
    min_px = job.get("min_defect_px", METRIC_MIN_DEFECT_PX)
    if not _count(min_px) or not 1 <= min_px <= METRIC_MAX_MIN_DEFECT_PX:
        raise StationConfigError(f"job.min_defect_px must be an integer 1.."
                                 f"{METRIC_MAX_MIN_DEFECT_PX} (camera px): larger would drop "
                                 "real defects as stray points")
    max_out = job.get("max_outlier_pct", METRIC_MAX_OUTLIER_PCT)
    if not _number(max_out) or not 0 <= max_out <= METRIC_MAX_OUTLIER_PCT_CAP:
        raise StationConfigError(f"job.max_outlier_pct must be a number 0.."
                                 f"{METRIC_MAX_OUTLIER_PCT_CAP:g} (% of the measured px that may "
                                 "be dropped as stray points)")
    pct = {}
    for key, dflt in (("min_valid_pct", METRIC_MIN_VALID_PCT),
                      ("min_coverage_pct", METRIC_MIN_COVERAGE_PCT)):
        v = job.get(key, dflt)
        if not _number(v) or not 0 <= v <= 100:
            raise StationConfigError(f"job.{key} must be a number 0..100 (%)")
        pct[key] = float(v)
    rng = job.get("plane_range_mm")
    if rng is not None and (not isinstance(rng, list) or len(rng) != 2
                            or not all(_number(v) and v > 0 for v in rng) or rng[1] <= rng[0]):
        raise StationConfigError("job.plane_range_mm must be [near, far] mm (0 < near < far) or "
                                 "null")
    return {"scan": scan, "axis": axis, "steps": steps,
            "periods": [float(v) for v in periods] if periods else None,
            "measure": measure, "roi": list(roi) if roi else None,
            "min_defect_px": int(min_px), "max_outlier_pct": float(max_out), **pct,
            "plane_range_mm": [float(v) for v in rng] if rng is not None else None,
            "guard_band_pct": gb}


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
    res = _runs.empty_result(run_id=run_id, run_folder=run_folder, mock=mock,
                             operator_confirmed=operator_confirmed, station=cfg.station_id,
                             job=job, part={"role": cfg.part_role, "label": cfg.part_label},
                             threshold_mm=cfg.threshold_mm, zone=cfg.zone,
                             # a single job: the configuration's one implicit profile
                             station_config=str(cfg.source.name) if cfg.source else None,
                             profile=_flow.LEGACY_PROFILE)
    # the station's bad-part output (a recipe job: it wins over the recipe's own)
    res["outputs"][_badparts.KEY] = cfg.bad_part_output
    return res

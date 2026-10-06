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
parsing console text. The verdict is only ever PASS / FAIL / INVALID:
INVALID means the numbers cannot be trusted (no mm calibration at this
geometry when mm are required, an INDETERMINATE engine verdict, a refused
or crashed run) and must never be reported as a pass or a fail.

This module only builds the config, the run folder, the log and the result
document. The measurement itself goes through engine.api verbs (ui/cli.py
cmd_run), so every engine rule still applies.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

RESULT_SCHEMA_VERSION = "0.1"
JOB_TYPES = ("inspect", "compare")
RESULTS = ("PASS", "FAIL", "INVALID")
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
        source=path.resolve(), raw=doc)


# ------------------------------------------------------------ run folder + log

def new_run(cfg: StationConfig, now: Optional[datetime] = None) -> tuple:
    """(run_id, run_dir): a fresh folder, never an existing one."""
    now = now or datetime.now(timezone.utc)
    root = Path(cfg.runs_dir)
    root.mkdir(parents=True, exist_ok=True)
    for _ in range(5):
        run_id = f"{now.strftime('%Y%m%dT%H%M%SZ')}_{cfg.station_id}_{uuid.uuid4().hex[:8]}"
        run_dir = root / run_id
        try:
            run_dir.mkdir()
            return run_id, run_dir
        except FileExistsError:
            continue
    raise FileExistsError(f"could not create a new run folder under {root}")


class RunLog:
    """JSON-lines event log: one event per line, run_id on every line."""

    def __init__(self, path: Path, run_id: str):
        self.path = Path(path)
        self.run_id = run_id
        self._f = open(self.path, "x", encoding="utf-8")

    def event(self, event: str, level: str = "info", **fields) -> None:
        rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
               "run_id": self.run_id, "level": level, "event": event, **fields}
        self._f.write(json.dumps(rec, default=str) + "\n")
        self._f.flush()

    def close(self) -> None:
        if not self._f.closed:
            self._f.close()


# ------------------------------------------------------------ result document

def software_info() -> dict:
    try:
        from importlib.metadata import version
        pkg = version("swiss-army-scanner")
    except Exception:                       # noqa: BLE001 - running from a source tree
        pkg = None
    commit = None
    if (_ROOT / ".git").exists():
        try:
            out = subprocess.run(["git", "-C", str(_ROOT), "rev-parse", "--short", "HEAD"],
                                 capture_output=True, text=True, timeout=10)
            commit = out.stdout.strip() or None if out.returncode == 0 else None
        except Exception:                   # noqa: BLE001 - git not installed
            commit = None
    return {"package_version": pkg, "git_commit": commit}


def empty_result(cfg: StationConfig, run_id: str, run_folder: str, mock: bool,
                 operator_confirmed: bool) -> dict:
    """The result skeleton: the same keys every run, filled in as the job goes."""
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "run_id": run_id,
        "run_folder": run_folder,
        "mode": "MOCK" if mock else "REAL",
        "station": cfg.station_id,
        "operator_confirmed": bool(operator_confirmed),
        "job": {"type": cfg.job_type, "base": cfg.base, "zone": cfg.zone,
                "config": str(cfg.source.name) if cfg.source else None},
        "calibration": {"calibration_id": None, "same_geometry": None,
                        "reprojection_error_px": {"camera": None, "projector": None}},
        "capture": {"preset": None, "geometry_id": None, "patterns": None,
                    "exposure_us": None, "average": None, "zone": cfg.zone},
        "part": {"role": cfg.part_role, "label": cfg.part_label, "scan_id": None},
        "deviation_mm": {"max": None, "mean": None, "rms": None, "method": None},
        "deviation_units": None,
        "threshold_mm": cfg.threshold_mm,
        "result": "INVALID",
        "invalid_reason": "not run",
        "engine_verdict": None,
        "warnings": [],
        "outputs": {"mesh": [], "deviation_map": None, "report": None, "log": "run.log.jsonl"},
        "software": software_info(),
        "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "finished": None,
    }


def write_result(run_dir: Path, result: dict) -> Path:
    """result.json, written once per run (exclusive create: never overwritten)."""
    if result.get("result") not in RESULTS:
        raise ValueError(f"result must be one of {RESULTS}")
    result["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out = Path(run_dir) / "result.json"
    with open(out, "x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
        f.write("\n")
    return out

"""
Run folders: one job in, one NEW folder out (never an existing one), shared by the station
jobs (`scan run`, ui/run_job.py) and the recipe runner (api.recipe_run, engine/recipes/).

    <runs_dir>/<UTC timestamp>_<station or recipe>_<8 hex>/
        result.json      the machine-readable result (RESULT_SCHEMA_VERSION; written once)
        run.log.jsonl    one JSON event per line, every line carries run_id

The result keys are the same for every run (a script or an automation flow reads them
without knowing the job type). The verdict is only ever PASS / FAIL / INVALID / ABORTED:
INVALID means the numbers cannot be trusted and must never be reported as a pass or a fail.

Schema 0.2 (extends 0.1, every 0.1 key kept): `recipe` (the recipe a recipe job ran, else
null), `zones` (per zone: id, verdict, checks; [] for other jobs) and `feedback` (what the
projector showed after the verdict, else null).
"""

from __future__ import annotations

import json
import re
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

RESULT_SCHEMA_VERSION = "0.2"
RESULTS = ("PASS", "FAIL", "INVALID", "ABORTED")
RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_ROOT = Path(__file__).resolve().parents[1]


def new_run(runs_dir, name: str, now: Optional[datetime] = None) -> tuple:
    """(run_id, run_dir): a fresh folder under runs_dir, never an existing one."""
    if not RUN_ID.match(str(name)):
        raise ValueError(f"run name {name!r}: letters, digits, _ . - only (max 64)")
    now = now or datetime.now(timezone.utc)
    root = Path(runs_dir)
    root.mkdir(parents=True, exist_ok=True)
    for _ in range(5):
        run_id = f"{now.strftime('%Y%m%dT%H%M%SZ')}_{name}_{uuid.uuid4().hex[:8]}"
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

    @property
    def closed(self) -> bool:
        return self._f.closed

    def close(self) -> None:
        if not self._f.closed:
            self._f.close()


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


def empty_result(*, run_id: str, run_folder: str, mock: bool, operator_confirmed: bool,
                 station: str, job: dict, part: dict, threshold_mm: Optional[float] = None,
                 zone=None) -> dict:
    """The result skeleton: the same keys every run, filled in as the job goes."""
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "run_id": run_id,
        "run_folder": run_folder,
        "mode": "MOCK" if mock else "REAL",
        "station": station,
        "operator_confirmed": bool(operator_confirmed),
        "job": job,
        "calibration": {"calibration_id": None, "same_geometry": None,
                        "reprojection_error_px": {"camera": None, "projector": None}},
        "capture": {"preset": None, "geometry_id": None, "patterns": None,
                    "exposure_us": None, "average": None, "zone": zone},
        "part": {"role": part.get("role", "test"), "label": part.get("label", "part"),
                 "scan_id": None},
        "deviation_mm": {"max": None, "mean": None, "rms": None, "method": None,
                         "n_points": None},
        "deviation_units": None,
        "threshold_mm": threshold_mm,
        "result": "INVALID",
        "invalid_reason": "not run",
        "engine_verdict": None,
        "warnings": [],
        "outputs": {"mesh": [], "deviation_map": None, "report": None, "log": "run.log.jsonl"},
        "recipe": None,
        "zones": [],
        "feedback": None,
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

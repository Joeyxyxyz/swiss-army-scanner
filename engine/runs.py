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
projector showed after the verdict, else null). `deviation_mm` also carries `n_defects`,
`largest_defect`, `outlier_px` / `outlier_clusters` / `outlier_pct` and `dense_strays`
(metric jobs: the out-of-tolerance clusters, the isolated stray px dropped - golden: the
worse side - and where dropped px were dense, else null; null for the other jobs).

Schema 0.3 (extends 0.2, every key kept, `result` unchanged): `disposition` (CONTINUE /
REJECT / PULL: what happens to the part, engine/disposition.py) and `pull_reasons` ([{zone?,
check?, reason, message}]: every doubt found). write_result never writes a run without a
disposition, nor a CONTINUE that is not a doubt-free PASS (fail closed).

Schema 0.4 (extends 0.3, every key kept): the work states. `station_config` (the station
configuration file the run belongs to, else null), `profile` (the active profile of that
configuration, else null), `steps` (the work states of the cycle, in flow order: id, title,
expects, sensor_inputs, dlp_result, outcome CONFIRMED / CORRECTED / NOT_MET / DEFECT / DOUBT
(or SKIPPED / ABORTED / PENDING), sensor_mismatch, permit, permission_granted (it still
stood when the cycle ended: the same as `permissions`), permission_note (why a CONFIRMED work
state's permission was withheld or dropped), actions, disposition), `permissions` (name ->
still granted when the cycle ended) and `cycle` (how a flow's cycle ended: ended_by, kickout
{step, disposition, reason, detail, by} for every cycle that did not CONTINUE,
sensor_mismatch (a sensor and the DLP disagreed at any verify, also one put right since),
sensor_mismatch_open, sensor_mismatches [{step, attempt, signal, sensor, dlp, expected}],
unconfirmed [{step, outcome, required}]; null for a single job). A single
job (a station config with one job, a recipe run) is ONE implicit work state "run", whose
outcome is the job's own disposition (CONTINUE -> CONFIRMED, REJECT -> DEFECT, anything
else DOUBT). A cycle of a flow (engine/flow/) is CONTINUE only when every required work
state was CONFIRMED; its `pull_reasons` may carry `step` (the work state they belong to).

`outputs.bad_part_output` (added in 0.4, every other key kept): the bad-part output folder
in force for the run, null when off (engine/badparts.py). After result.json is written,
write_result hands a REJECT / PULL to engine/badparts.record: its scans are marked as a bad
part's (never a reference) and it is copied to that folder; a copy that failed is said in
the run's warnings (the result the caller prints and returns), never in a changed judgment.
"""

from __future__ import annotations

import json
import re
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from . import badparts, eventlog
from . import disposition as D

RESULT_SCHEMA_VERSION = "0.4"
RESULTS = ("PASS", "FAIL", "INVALID", "ABORTED")
RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_ROOT = Path(__file__).resolve().parents[1]


def new_run(runs_dir, name: str, now: Optional[datetime] = None) -> tuple:
    """(run_id, run_dir): a fresh folder under runs_dir, never an existing one."""
    if not RUN_ID.fullmatch(str(name)):
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


def append_event(log_path: Path, run_id: str, event: str, level: str = "info",
                 **fields) -> None:
    """One more event on a run's CLOSED log (what happened to the run after run_end, e.g.
    the line counting the cycle ABORTED after its verdict was written). result.json is
    never rewritten; the log records it. The inspection event log gets the run's row again
    when the event changes what the row says (ABORTED after the verdict: PULL)."""
    rec = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
           "run_id": run_id, "level": level, "event": event, **fields}
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    if event == eventlog.ABORTED_EVENT:
        eventlog.amend(Path(log_path).parent)   # its row again, now PULL (never raises)
        badparts.record(Path(log_path).parent)  # a PULL now: marked, copied (never raises)


def software_info() -> dict:
    """{package_version, git_commit}: the installed package's version, else the version the
    source tree states (engine.__version__, as pyproject.toml); the commit when it runs from
    a git checkout (a copied tree has none)."""
    try:
        from importlib.metadata import version
        pkg = version("swiss-army-scanner")
    except Exception:                       # noqa: BLE001 - running from a source tree
        from . import __version__
        pkg = __version__
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
                 zone=None, station_config: Optional[str] = None,
                 profile: Optional[str] = None) -> dict:
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
        # n_defects / largest_defect / outlier_* / dense_strays: metric jobs (out-of-tolerance
        # clusters, the isolated stray px dropped, where they were dense), else null
        "deviation_mm": {"max": None, "mean": None, "rms": None, "method": None,
                         "n_points": None, "n_defects": None, "largest_defect": None,
                         "outlier_px": None, "outlier_clusters": None, "outlier_pct": None,
                         "dense_strays": None},
        "deviation_units": None,
        "threshold_mm": threshold_mm,
        "result": "INVALID",
        "invalid_reason": "not run",
        "engine_verdict": None,
        "disposition": None,                # CONTINUE | REJECT | PULL (engine/disposition.py)
        "pull_reasons": [],
        "warnings": [],
        # bad_part_output: the folder a part judged REJECT / PULL is copied to (null: off).
        # Whoever builds the run puts its configuration's there; write_result settles it
        # (engine/badparts.py: the command line, the configuration, else the environment)
        "outputs": {"mesh": [], "deviation_map": None, "report": None, "log": "run.log.jsonl",
                    "bad_part_output": None},
        "recipe": None,
        "zones": [],
        "feedback": None,
        # the work states (schema 0.4): a flow's cycle fills them in; a single job gets its
        # one implicit work state when the result is written (write_result)
        "station_config": station_config,
        "profile": profile,
        "steps": [],
        "permissions": {},
        "cycle": None,
        "software": software_info(),
        "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "finished": None,
    }


def implicit_step(result: dict) -> dict:
    """The one work state of a single job (no flow): its outcome is the job's own
    disposition - CONTINUE (a doubt-free PASS) is CONFIRMED, REJECT (a definite FAIL) is
    DEFECT, anything else DOUBT (ABORTED as such). It expects no sensor, grants no
    permission."""
    from .flow.cycle import job_outcome
    job = result.get("job") or {}
    res, disp = result.get("result"), result.get("disposition")
    return {"id": "run", "title": f"{job.get('type') or 'single'} job", "index": 1,
            "required": True, "expects": {}, "sensor_inputs": {},
            "dlp_result": {"result": res, "disposition": disp, "run_id": result.get("run_id")},
            "outcome": job_outcome(res, disp), "sensor_mismatch": [], "permit": None,
            "permission_granted": False, "permission_note": None, "actions": [],
            "disposition": disp}


def write_result(run_dir: Path, result: dict) -> Path:
    """result.json, written once per run (exclusive create: never overwritten). A missing or
    inconsistent disposition is settled first (engine/disposition.ensure: never a CONTINUE
    that is not a doubt-free PASS)."""
    if result.get("result") not in RESULTS:
        raise ValueError(f"result must be one of {RESULTS}")
    D.ensure(result)
    for key, empty in (("station_config", None), ("profile", None), ("permissions", {}),
                       ("cycle", None)):
        result.setdefault(key, empty)
    if not result.get("steps"):             # a single job: one implicit work state
        result["steps"] = [implicit_step(result)]
    badparts.settle(result)                 # outputs.bad_part_output: where a bad part goes
    result["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out = Path(run_dir) / "result.json"
    with open(out, "x", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
        f.write("\n")
    eventlog.record(run_dir)                # its row in the inspection event log (never raises)
    # a REJECT / PULL: marked on its scans and copied to the bad-part output (never raises);
    # what failed is said in the run's output (its warnings), result.json stays as written
    got = badparts.record(run_dir)
    if got and got.get("warnings") and isinstance(result.get("warnings"), list):
        result["warnings"].extend(got["warnings"])
    return out

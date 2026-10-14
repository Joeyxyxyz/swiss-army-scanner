"""
Inspection event log: one row per inspection, in the library, for a spreadsheet.

    <library>/events/inspections_v1_<YYYY-MM>.csv          one row per inspection
    <library>/events/inspection_values_v1_<YYYY-MM>.csv    one row per measured value
    <library>/events/img/<inspection id>.png               a copy of the run's picture
    <library>/events/pending/<the same two file names>     rows that could not be appended
    <library>/events/last_v1.csv                           the newest inspection row, alone

One file pair per month (the month of the inspection's time, UTC). `v1` in the name is the
format: the two headers below. A later format gets a new name; a v1 file is never rewritten
with other columns. A month's file grows to tens of MB: whoever wants only the row of the
inspection that just ended (its part id) reads last_v1.csv, the header and that one row.

An inspection is a run that opened a run folder and wrote its result.json: a recipe run, a
quick check, a station job, a cycle of a station configuration, a measurement. A cycle is
ONE inspection: the checks of its work states (their run folders under <cycle>/steps/) give
its value rows, each naming its step. The check of one work state run outside a cycle is an
inspection of its own. A START that is refused before a run folder exists writes no row.

An inspection id can have more than one row: the row is written again (amend) when
something happened to the run after its result.json was written - the line counting it
ABORTED after its verdict (the run log's `aborted_after_verdict` event), which makes its
judgment PULL. engine/runs.append_event, which writes that event, calls amend(run folder)
right after; a rebuild reads the event itself. THE LAST ROW OF AN INSPECTION ID COUNTS, for
every reader. Its value rows are written once.

The inspection row (INSPECTION_FIELDS, in this order):
    time            when the result was written (the document's `finished`, UTC)
    inspection_id   the run id (the name of the run folder)
    part_id         the code read from the part: the run log's `codes` event (codes read
                    before the scan), else the codes the run's `code` checks read from the
                    scan's own picture and matched (details.codes in result.json); a cycle:
                    of its work states; several joined by " | "; else empty
    part            the part label: a type name, not a serial (a measurement: its label)
    program         what was checked: a recipe run or a recipe job: the recipe; a cycle: station
                    configuration / profile; one work state's check: ... / work state; any
                    other single job: its station configuration (or station id); a
                    measurement: measure:<what>
    mode            MOCK or REAL, as the document says: a MOCK row is a simulated inspection
    judgment        PASS (disposition CONTINUE), FAIL (REJECT), PULL (PULL, and any
                    disposition that is none of the three: never a pass by default; a run
                    the line counted ABORTED after its verdict; a cycle that never ended); a
                    measurement decides nothing: "-"
    reason          FAIL: the first failing check (a cycle: where it was kicked out; a job
                    without checks: the value outside its limit); PULL: the first pull reason
                    (code: where: message) and how many more; a measurement that is NOT
                    MEASURABLE: why. Cut after 300 characters: result.json has all of it
    n_values, n_out_of_limit, values
                    the value rows of the inspection: how many, how many outside their
                    limits, and the first 12 as `name=value unit [lower..upper] OK`
    time_s          seconds from the first event of the run log to the writing of result.json
                    (the file's own time; a folder copied without its file times: the
                    document's `finished`, which is cut to the whole second)
    image_file      events/img/<inspection id>.png: the run's overlay.png or deviation.png,
                    copied, so the link survives a cleaned run folder
    profile_file    profile_x.csv of the run folder (below), when a check wrote one
    data_3d_file    the scan's point cloud (.ply), when the run has one
    result_file     the run's result.json: the source of truth for the row (empty for a
                    cycle that never ended: it wrote none)
    preset, calibration_id   as the document says (a cycle: of its work states' checks)
    rig             the rig geometry the scan was made at (geometry_id)
    software        package version and commit, as the document says

File cells are relative to the library with forward slashes (what the web server's /files/
route serves) when the file is inside the library, else the full path.

The value row (VALUE_FIELDS): inspection_id, step (the work state of a cycle, else empty),
zone, check, name, value, unit, lower_limit, upper_limit, result. One row per value a check
held to a limit: its `tolerances`, or its `key` / `limit` when it gives none (|value| <=
limit, as engine/disposition.tolerances_of); a single job without zones: deviation_mm.max
against threshold_mm; a measurement: every value, no limits. result: OK (inside its limits),
OUT (outside), INVALID (the check could not measure, or there is no finite number), "-" (no
limit to hold it to). A value in the guard band is OK here: the inspection row says PULL.

Height profiles: where a check grids the part's heights (the end-face defect check), the
recipe runner writes the two lines through the part's centre into the run folder as
profile_x.csv (along camera right) and profile_y.csv (along camera down): one header line,
then position mm, height mm. The first checked zone that has one; nothing in result.json.

A cycle that never ended (the program that ran it was closed or stopped while it waited for
the next work state: its folder has the cycle's log and the checks of its work states under
steps/, but no result.json) is written by no hook. `--rebuild` lists it: judgment PULL,
reason "error: the cycle never ended", with the value rows of the work states it checked;
its time is that of the last thing it wrote. A cycle that may still be running is left out:
one whose log is still open for writing (Windows: the program that runs it is alive), or,
where that cannot be told, whose files were written in the last hour.

Rules of the log:
  * It never changes a result: the hook (record) is called after result.json is written,
    never raises, and reads the document back from the file, so a live row and a rebuilt
    row are the same row.
  * Append only, whole rows: every write holds an OS file lock (events/eventlog.lock; as
    engine/capture/riglock.py) and is one write of whole lines. Value rows first, the
    inspection row last.
  * A row is never dropped while there is a place to write it. Rows that cannot be added
    now - their file is open in a spreadsheet program (which denies writers), or the lock
    could not be had in LOCK_WAIT_S (another program is busy with the log) - go to
    events/pending/<same name> and are added on the next append (or by `--merge`). The
    waiting rows have a lock of their own (events/pending.lock), held only for one write
    or one rename, so a busy log holds an inspection up for LOCK_WAIT_S and no longer.
    The page counts the waiting rows ("N rows waiting").
  * A text cell that starts with = + - or @ and is not a number gets a leading apostrophe
    (a code read from a part is foreign text; a spreadsheet must not run it). The lone "-"
    the log itself writes stays. Line breaks and tabs in a cell become one space.
  * The log is on unless the environment says SAS_EVENTLOG=off (read at every call): then
    nothing is written, anywhere - no row, no waiting row, no picture, no profile file.
    The tests run with it off.

    python -m engine.eventlog --rebuild [--library DIR] [--runs DIR] [--stations DIR]
                                        [--recipes DIR]     (each may be given again)
    python -m engine.eventlog --merge   [--library DIR]

--rebuild writes both files again from the run folders: the library's runs/, the runs
folder of every station file (--stations, default: the repo's stations/),
$SCANNER_RUNS_DIR, every recipe's outputs.runs_dir (--recipes, default: the repo's
recipes/, and the library's own), and each --runs. Rows of the present files whose run
folder is gone are kept as they are. The files it replaces are moved to
events/replaced_<time>/, never deleted. It may run while inspections are made: the run
folders and the present files are read and the new files written WITHOUT the lock; the lock
is held only to take over the rows that were appended meanwhile and to swap the files.
--merge adds the waiting rows now.

No hardware, and nothing of the engine is loaded with this module but engine/disposition.py
(the unit of a value name, the name of a code's kind and the default library are looked up
when they are needed).
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import math
import os
import re
import shutil
import sys
import threading
import time
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from . import disposition as D

log = logging.getLogger(__name__)

FORMAT = "v1"
ENV = "SAS_EVENTLOG"                    # "off": write nothing, anywhere (read at every call)
EVENTS_DIR, IMG_DIR, PENDING_DIR = "events", "img", "pending"
INSPECTIONS = f"inspections_{FORMAT}_"
VALUES = f"inspection_values_{FORMAT}_"
LAST = f"last_{FORMAT}.csv"             # the newest inspection row alone: a small file
LOCK_FILE = "eventlog.lock"             # one writer of the month files at a time
PENDING_LOCK_FILE = "pending.lock"      # one writer of the waiting rows (events/pending/)
LOCK_WAIT_S = 0.5                       # an append waits this long for another writer ...
LOCK_POLL_S = 0.001                     # ... trying again this often; then its rows wait
HOOK_SETTLE_S = 1.5                     # a run's own hook has written its row this long after
                                        # its result.json (the lock wait and a margin)
ABORTED_EVENT = "aborted_after_verdict"     # run log: the line counted the run ABORTED later
ABORTED_TEXT = "the line counted the run ABORTED after its verdict was written"
AMEND_WAIT_S = 0.05                     # a row written again waits this long for a lock (its
                                        # caller may hold a lock of its own), then waits in
                                        # events/pending/
UNENDED_TEXT = ("the cycle never ended: no result.json was written (the program that ran it "
                "was closed or stopped before the cycle was over)")
UNENDED_QUIET_S = 3600.0                # where an open log cannot be told: a cycle whose files
                                        # were written since may still be running
INSPECTION_FIELDS = ("time", "inspection_id", "part_id", "part", "program", "mode", "judgment",
                     "reason", "n_values", "n_out_of_limit", "values", "time_s", "image_file",
                     "profile_file", "data_3d_file", "result_file", "preset", "calibration_id",
                     "rig", "software")
VALUE_FIELDS = ("inspection_id", "step", "zone", "check", "name", "value", "unit",
                "lower_limit", "upper_limit", "result")
JUDGMENTS = {"CONTINUE": "PASS", "REJECT": "FAIL", "PULL": "PULL"}
NO_JUDGMENT = "-"                       # a measurement; a value with no limit
PROFILE_X, PROFILE_Y = "profile_x.csv", "profile_y.csv"
PROFILE_HEADER = "position_mm,height_mm"
RESULT_FILE, LOG_FILE = "result.json", "run.log.jsonl"
STEPS_DIR = "steps"                     # a cycle folder's step run folders (engine/flow/verify.py)
PICTURES = ("overlay.png", "deviation.png")
MAX_VALUES_SHOWN = 12                   # in the `values` cell; the value rows hold all
MAX_REASON = 300
_BOM = b"\xef\xbb\xbf"                  # so a spreadsheet program reads the file as UTF-8
_NUMBER = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,199}$")
_MONTH = re.compile(r"^\d{4}-\d{2}")
_ROOT = Path(__file__).resolve().parents[1]


def enabled() -> bool:
    """False when SAS_EVENTLOG says off (off / 0 / false / no): nothing is written then.
    Read at every call, never once at import."""
    return os.environ.get(ENV, "on").strip().lower() not in ("off", "0", "false", "no")


# ------------------------------------------------------------------ cells

def _num(v) -> Optional[float]:
    """A finite number, also one that result.json holds as text (a numpy scalar the
    document was written with), else None."""
    if isinstance(v, str) and _NUMBER.match(v):
        v = float(v)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    v = float(v)
    return v if math.isfinite(v) else None


def _g(v) -> str:
    return "" if v is None else f"{v:.6g}"


def cell(v) -> str:
    """One cell as text: a number as it is, None empty, any other value on one line (line
    breaks and tabs become a space). Text that a spreadsheet would take for a formula (it
    starts with = + - or @ and is not a number) gets a leading apostrophe; the lone "-" of
    this log stays."""
    if v is None:
        return ""
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return _g(v) if math.isfinite(v) else ""
    s = " ".join(str(v).split())
    if s[:1] in ("=", "+", "-", "@") and s != NO_JUDGMENT and not _NUMBER.match(s):
        s = "'" + s
    return s


def _short(text, n: int = MAX_REASON) -> str:
    s = " ".join(str(text or "").split())
    return s if len(s) <= n else s[:n - 3].rstrip() + "..."


def _encode(fields: tuple, rows: list, header: bool = False) -> bytes:
    """Rows -> csv lines (bytes, UTF-8), every line whole and ended. What UTF-8 cannot write
    (half a character in a code that was read from a part) becomes a ?: never a lost row."""
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\r\n")
    if header:
        w.writerow(fields)
    for r in rows:
        w.writerow([cell(r.get(k)) for k in fields])
    return buf.getvalue().encode("utf-8", errors="replace")


# ------------------------------------------------------------------ reading a run folder

def _read_json(path: Path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _log_facts(run_dir: Path) -> dict:
    """What a run's log says that its result.json does not: `first` / `last` (the time of
    its first and last event), `start` (its first event), `codes` (the codes read before
    the scan, each once) and `aborted` (the message of an `aborted_after_verdict` event: the
    line counted the run ABORTED after its verdict was written; else None). Nothing when the
    log cannot be read. It may still be open for writing: every event was flushed."""
    out = {"first": None, "last": None, "start": {}, "codes": [], "aborted": None}
    try:
        with open(run_dir / LOG_FILE, encoding="utf-8") as f:
            for line in f:
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(ev, dict):
                    continue
                if not out["start"]:
                    out["start"] = ev
                if out["first"] is None:
                    out["first"] = ev.get("ts")
                out["last"] = ev.get("ts") or out["last"]
                if ev.get("event") == "codes" and isinstance(ev.get("data"), list):
                    out["codes"] += [str(c) for c in ev["data"] if c not in (None, "")]
                elif ev.get("event") == ABORTED_EVENT:
                    out["aborted"] = str(ev.get("message") or ABORTED_TEXT)
    except OSError:
        pass
    out["codes"] = list(dict.fromkeys(out["codes"]))
    return out


def _kind(kind) -> str:
    try:
        from .sense.codes import normalise_kind     # one name per symbology
        return normalise_kind(kind)
    except Exception:                       # noqa: BLE001 - a name never stops a row
        return "".join(ch for ch in str(kind or "").upper() if ch.isalnum())


def _doc_codes(doc: dict) -> list:
    """The codes a run's `code` checks read from the scan's own picture (their details.codes
    in result.json: the run log has no `codes` event then), each once: the ones that match
    what the check asked for (its params kind / expect / pattern, as the check matched them:
    engine/recipes/checks/builtin.Code). A code that is on the part but is not the one the
    check looks for is no part id."""
    out = []
    for z in doc.get("zones") or []:
        for ch in (z.get("checks") or []) if isinstance(z, dict) else []:
            if not isinstance(ch, dict) or ch.get("type") != "code":
                continue
            det = ch.get("details") if isinstance(ch.get("details"), dict) else {}
            p = ch.get("params") if isinstance(ch.get("params"), dict) else {}
            for c in det.get("codes") if isinstance(det.get("codes"), list) else []:
                if not isinstance(c, dict) or c.get("data") in (None, ""):
                    continue
                data = str(c["data"])
                if p.get("kind") is not None and _kind(c.get("kind")) != _kind(p["kind"]):
                    continue
                if p.get("expect") is not None and data != p["expect"]:
                    continue
                try:
                    if p.get("pattern") is not None and not re.search(p["pattern"], data):
                        continue
                except (re.error, TypeError):
                    continue
                out.append(data)
    return list(dict.fromkeys(out))


def _when(text) -> Optional[datetime]:
    try:
        t = datetime.fromisoformat(str(text))
    except ValueError:
        return None
    return t if t.tzinfo is not None else t.replace(tzinfo=timezone.utc)


def _time_s(run_dir: Path, doc: dict, first_ts) -> Optional[float]:
    """Seconds from the first event of the run log (else the document's `started`) to the
    writing of result.json: the file's own time while it agrees with the document's
    `finished` (which is cut to the whole second), else `finished` (a folder copied without
    its file times). The same number live and rebuilt."""
    start, end = _when(first_ts) or _when(doc.get("started")), _when(doc.get("finished"))
    if start is None or end is None:
        return None
    try:
        written = datetime.fromtimestamp((run_dir / RESULT_FILE).stat().st_mtime, timezone.utc)
        if 0.0 <= (written - end).total_seconds() < 2.0:
            end = written
    except (OSError, OverflowError, ValueError):
        pass
    return max(round((end - start).total_seconds(), 1), 0.0)


def _link(path, library: Path) -> str:
    """A file as the log names it: relative to the library with forward slashes (what
    /files/ serves) when it is inside the library, else its full path."""
    if path is None:
        return ""
    try:
        p = Path(path).resolve()
    except OSError:
        return str(path)
    try:
        return p.relative_to(Path(library).resolve()).as_posix()
    except (ValueError, OSError):
        return str(p)


def _first(items):
    return next((x for x in items if x), None)


def _picture_of(run_dir: Path, doc: dict) -> Optional[Path]:
    for name in PICTURES:
        if (run_dir / name).is_file():
            return run_dir / name
    out = doc.get("outputs") if isinstance(doc.get("outputs"), dict) else {}
    for v in (out.get("overlay"), out.get("deviation_map")):
        if isinstance(v, str) and v.lower().endswith(".png"):
            for p in (run_dir / v, Path(v)):
                if p.is_file():
                    return p
    return None


def _scan_ply(library: Path, doc: dict) -> Optional[Path]:
    """The point cloud the library exported for the run's scan (the scan entry's
    exports.jsonl names it)."""
    part = doc.get("part") if isinstance(doc.get("part"), dict) else {}
    scans = doc.get("scans") if isinstance(doc.get("scans"), list) else []
    scan = part.get("scan_id") or (scans[0] if scans else None)
    cap = doc.get("capture") if isinstance(doc.get("capture"), dict) else {}
    pre = doc.get("preset") if isinstance(doc.get("preset"), dict) else {}
    geo = cap.get("geometry_id") or pre.get("geometry_id")
    if not isinstance(scan, str) or not _SAFE_ID.match(scan):
        return None
    entries = [library / "objects" / scan]
    if isinstance(geo, str) and _SAFE_ID.match(geo):
        entries.insert(0, library / "metric" / geo / scan)
    for entry in entries:
        try:
            lines = (entry / "exports.jsonl").read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            f = rec.get("file") if isinstance(rec, dict) else None
            if isinstance(f, str) and f.lower().endswith(".ply") and (library / f).is_file():
                return library / f
    return None


def _ply_of(run_dir: Path, doc: dict, library: Path) -> Optional[Path]:
    out = doc.get("outputs") if isinstance(doc.get("outputs"), dict) else {}
    mesh = out.get("mesh") if isinstance(out.get("mesh"), list) else []
    for v in [out.get("ply"), *mesh]:
        if isinstance(v, str) and v.lower().endswith(".ply"):
            for p in (run_dir / v, Path(v)):
                if p.is_file():
                    return p
    return _scan_ply(library, doc)


# ------------------------------------------------------------------ values

def _unit(name: str, units) -> str:
    u = units.get(name) if isinstance(units, dict) else None
    if isinstance(u, str):
        return u
    try:
        from .recipes.checks.base import unit_of    # the unit a value name carries
        return unit_of(name)
    except Exception:                       # noqa: BLE001 - a unit never stops a row
        return ""


def _value(step, zone, check, name, value, unit, lo, hi, verdict) -> dict:
    """One value row (without its inspection id). verdict: the check's PASS / FAIL / INVALID,
    or NO_JUDGMENT for a value nothing judges."""
    v, lo, hi = _num(value), _num(lo), _num(hi)
    if verdict == NO_JUDGMENT:
        result = NO_JUDGMENT
    elif verdict not in ("PASS", "FAIL") or v is None:
        result = "INVALID"
    elif lo is None and hi is None:
        result = NO_JUDGMENT
    elif (lo is not None and v < lo) or (hi is not None and v > hi):
        result = "OUT"
    else:
        result = "OK"
    return {"step": step or "", "zone": zone or "", "check": check or "", "name": str(name),
            "value": v, "unit": unit or "", "lower_limit": lo, "upper_limit": hi,
            "result": result}


def _run_values(doc: dict, step: str = "") -> list:
    """The values a run held to a limit: every tolerance of every check (its `tolerances`,
    else its key / limit); a job without checked zones: deviation_mm.max against
    threshold_mm, when there is a number."""
    out = []
    zones = [z for z in doc.get("zones") or [] if isinstance(z, dict) and z.get("checks")]
    for z in zones:
        for ch in z["checks"]:
            if not isinstance(ch, dict):
                continue
            try:
                tols = D.tolerances_of(ch)
            except (ValueError, TypeError, AttributeError):  # malformed: no value to show
                continue
            for t in tols:
                out.append(_value(step, z.get("id"), ch.get("id"), t["name"], t.get("value"),
                                  _unit(t["name"], ch.get("units")), t.get("lo"), t.get("hi"),
                                  ch.get("verdict")))
    dev = doc.get("deviation_mm") if isinstance(doc.get("deviation_mm"), dict) else {}
    if not zones and _num(dev.get("max")) is not None:
        job = doc.get("job") if isinstance(doc.get("job"), dict) else {}
        out.append(_value(step, "", job.get("type"), "deviation_mm.max", dev["max"],
                          doc.get("deviation_units") or "mm", None, doc.get("threshold_mm"),
                          doc.get("result")))
    return out


def _summary(values: list) -> str:
    """The `values` cell: name=value unit [lower..upper] OK; ... (the first few)."""
    names = [v["name"] for v in values]
    bits = []
    for v in values[:MAX_VALUES_SHOWN]:
        label = v["name"]
        where = "/".join(x for x in (v["step"], v["zone"], v["check"]) if x)
        if names.count(label) > 1 and where:        # the same name twice: say whose
            label = f"{where}: {label}"
        text = f"{label}={_g(v['value'])}" + (f" {v['unit']}" if v["unit"] else "")
        if v["lower_limit"] is not None or v["upper_limit"] is not None:
            text += f" [{_g(v['lower_limit'])}..{_g(v['upper_limit'])}]"
        if v["result"] != NO_JUDGMENT:
            text += f" {v['result']}"
        bits.append(text)
    more = len(values) - len(bits)
    return "; ".join(bits) + (f"; (+{more} more)" if more > 0 else "")


# ------------------------------------------------------------------ judgment + reason

def _pull_text(pulls: list) -> str:
    r = pulls[0]
    where = "/".join(str(x) for x in (r.get("step"), r.get("zone"), r.get("check")) if x)
    text = f"{r.get('reason')}: " + (f"{where}: " if where else "") + str(r.get("message") or "")
    return _short(text) + (f" (+{len(pulls) - 1} more)" if len(pulls) > 1 else "")


def _run_reason(doc: dict, values: list) -> str:
    """Why a run did not PASS, in one line (the module docstring says which)."""
    disp = doc.get("disposition")
    if disp == "CONTINUE":
        return ""
    pulls = [r for r in doc.get("pull_reasons") or [] if isinstance(r, dict)]
    if disp == "REJECT":
        for z in doc.get("zones") or []:
            for ch in (z.get("checks") or []) if isinstance(z, dict) else []:
                if isinstance(ch, dict) and ch.get("verdict") == "FAIL" and not ch.get("pull"):
                    return _short(f"{z.get('id')}/{ch.get('id')}: {ch.get('message') or 'FAIL'}")
        cyc = doc.get("cycle") if isinstance(doc.get("cycle"), dict) else {}
        kick = cyc.get("kickout") if isinstance(cyc.get("kickout"), dict) else None
        if kick:
            why = " - ".join(str(x) for x in (kick.get("reason"), kick.get("detail")) if x)
            return _short(f"{kick.get('step')}: {why}")
        out = next((v for v in values if v["result"] == "OUT"), None)
        if out is not None:
            return _short(f"{out['name']} {_g(out['value'])} {out['unit']} is outside its "
                          f"limits [{_g(out['lower_limit'])}..{_g(out['upper_limit'])}]")
    if pulls:
        return _pull_text(pulls)
    return _short(doc.get("invalid_reason")
                  or ("" if disp in JUDGMENTS else f"disposition {disp!r}"))


# ------------------------------------------------------------------ one inspection

def library_of(run_dir) -> Path:
    """The library whose event log a run belongs to: the configured session's, else the
    folder its run folder is in."""
    api = sys.modules.get(f"{__package__}.api") if __package__ else None
    store = getattr(getattr(api, "_session", None), "store", None)
    root = getattr(store, "root", None)
    return Path(root) if root else Path(run_dir).parent


def in_cycle(run_dir, doc: dict) -> bool:
    """The check of one work state that ran INSIDE a cycle (<cycle folder>/steps/<run>): its
    values are rows of the cycle's inspection, it is no inspection of its own."""
    run_dir = Path(run_dir)
    job = doc.get("job") if isinstance(doc.get("job"), dict) else {}
    return bool(job.get("work_state")) and run_dir.parent.name == STEPS_DIR \
        and (run_dir.parent.parent / LOG_FILE).is_file()


def _steps(run_dir: Path, doc: dict) -> list:
    """The checked work states of a cycle, in flow order: [(step id, the run folder of the
    check that stands for it (its last), that run's result document)]."""
    out = []
    for st in doc.get("steps") or []:
        dlp = st.get("dlp_result") if isinstance(st, dict) else None
        rid = dlp.get("run_id") if isinstance(dlp, dict) else None
        if not isinstance(rid, str) or not _SAFE_ID.match(rid):
            continue
        folders = [run_dir / STEPS_DIR / rid]
        if isinstance(dlp.get("run_folder"), str) and dlp["run_folder"]:
            folders.append(Path(dlp["run_folder"]))
        for d in folders:
            sub = _read_json(d / RESULT_FILE)
            if isinstance(sub, dict) and sub.get("run_id") == rid:
                out.append((str(st.get("id") or ""), d, sub))
                break
    return out


def _speaker_first(steps: list, doc: dict) -> list:
    """The steps with the one that speaks for the cycle first: where it was kicked out,
    else its last checked work state. Its picture, profile and 3D data are the row's."""
    cyc = doc.get("cycle") if isinstance(doc.get("cycle"), dict) else {}
    kick = (cyc.get("kickout") or {}).get("step") if isinstance(cyc.get("kickout"), dict) else None
    first = next((s for s in steps if s[0] == kick), steps[-1] if steps else None)
    return [first] + [s for s in steps if s is not first] if first else []


def _being_written(path: Path) -> Optional[bool]:
    """Does some process have the file open for writing now? True / False on Windows (an
    open that lets nobody else write is refused while one has); None where that cannot be
    told. A run's log is open for writing for as long as the run goes on."""
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateFileW.restype = wintypes.HANDLE
        k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                    wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD,
                                    wintypes.HANDLE]
        k32.CloseHandle.argtypes = [wintypes.HANDLE]
        # read access, others may only read, the file must exist
        h = k32.CreateFileW(str(path), 0x80000000, 0x1, None, 3, 0x80, None)
        if h in (None, wintypes.HANDLE(-1).value):
            return True if ctypes.get_last_error() == 32 else None      # sharing violation
        k32.CloseHandle(h)
        return False
    except Exception:                       # noqa: BLE001 - not told: the caller decides
        return None


def unended_cycle(run_dir) -> Optional[dict]:
    """A cycle that never ended: its folder has the cycle's log and the checks of its work
    states under steps/, but no result.json (the program that ran it was closed or stopped
    while the cycle waited). -> the document its row is built from - what
    engine/flow/verify.cycle_document writes for a cycle recorded before it ended: INVALID /
    PULL, reason `error` - or None: no such folder, or the cycle may still be running (its
    log is open for writing; where that cannot be told: one of its files was written in the
    last UNENDED_QUIET_S)."""
    run_dir = Path(run_dir)
    log_file = run_dir / LOG_FILE
    if (run_dir / RESULT_FILE).exists() or not log_file.is_file():
        return None
    subs, written = {}, []
    try:
        folders = sorted(p for p in (run_dir / STEPS_DIR).iterdir() if p.is_dir())
        written.append(log_file.stat().st_mtime)
    except OSError:
        return None
    for d in folders:
        sub = _read_json(d / RESULT_FILE)
        job = sub.get("job") if isinstance(sub, dict) else None
        state = job.get("work_state") if isinstance(job, dict) else None
        rid = sub.get("run_id") if state else None
        if not isinstance(rid, str) or not _SAFE_ID.match(rid):
            continue
        subs.pop(str(state), None)          # checked again: its last check stands for it
        subs[str(state)] = (d, sub)
        try:
            written.append((d / RESULT_FILE).stat().st_mtime)
        except OSError:
            pass
    if not subs:
        return None
    live = _being_written(log_file)
    if live is None:
        live = time.time() - max(written) < UNENDED_QUIET_S
    if live:
        return None
    facts = _log_facts(run_dir)
    start = facts["start"]                  # run_start, as engine/flow/verify.open_cycle wrote it
    last = subs[next(reversed(subs))][1]    # the last check it made
    flow = start.get("work_states") if isinstance(start.get("work_states"), list) else []
    flow = [str(w) for w in flow]
    ids = [w for w in flow if w in subs] + [w for w in subs if w not in flow]
    times = [t for t in map(_when, [facts["last"], *(s.get("finished") for _, s in subs.values())])
             if t is not None]
    config = start.get("config") or last.get("station_config")
    profile = start.get("profile") or last.get("profile")
    return {"run_id": start.get("run_id") or run_dir.name, "run_folder": str(run_dir),
            "mode": start.get("mode") or last.get("mode"),
            "station": start.get("station") or last.get("station"),
            "station_config": config, "profile": profile,
            "job": {"type": "flow", "config": config, "work_states": flow or ids},
            "part": {"role": "test", "label": profile or "part"},
            "started": facts["first"],
            "finished": max(times).astimezone(timezone.utc).isoformat(timespec="seconds")
            if times else None,
            "steps": [{"id": w, "dlp_result": {"run_id": subs[w][1]["run_id"],
                                               "run_folder": str(subs[w][0])}} for w in ids],
            "result": "INVALID", "engine_verdict": "INVALID", "invalid_reason": UNENDED_TEXT,
            "disposition": "PULL", "pull_reasons": [{"reason": "error", "message": UNENDED_TEXT}],
            "cycle": {"ended_by": None, "kickout": None}, "software": last.get("software")}


def _image(library: Path, iid: str, picture: Optional[Path], copy: bool) -> str:
    """events/img/<inspection id>.png: the run's picture, copied once (never replaced)."""
    dest = library / EVENTS_DIR / IMG_DIR / f"{iid}.png"
    if copy and picture is not None and not dest.exists():
        try:
            with open(picture, "rb") as src, _open(dest, "xb") as dst:
                shutil.copyfileobj(src, dst)
        except OSError as e:
            log.warning("inspection event log: no picture copy for %s: %s", iid, e)
    return _link(dest, library) if dest.is_file() else ""


def rows_of(run_dir, library=None, copy_picture: bool = True) -> Optional[tuple]:
    """(the inspection row, its value rows) of one run folder, read from its result.json
    and what lies beside it (a cycle that never ended has none: unended_cycle). None when
    the folder is no inspection of its own: no readable result.json with a run id, or the
    check of a work state inside a cycle (in_cycle).
    copy_picture: also copy the run's picture to events/img/ (not with the log off)."""
    run_dir = Path(run_dir)
    doc = _read_json(run_dir / RESULT_FILE)
    if doc is None:
        doc = unended_cycle(run_dir)
    if not isinstance(doc, dict) or not isinstance(doc.get("run_id"), str) \
            or not _SAFE_ID.match(doc["run_id"]) or in_cycle(run_dir, doc):
        return None
    library = Path(library) if library is not None else library_of(run_dir)
    iid = doc["run_id"]
    measure = doc.get("operating_mode") == "measure"
    job = doc.get("job") if isinstance(doc.get("job"), dict) else {}
    steps = _steps(run_dir, doc) if job.get("type") == "flow" else []
    facts = _log_facts(run_dir)
    first_ts, codes = facts["first"], facts["codes"]
    for _, d, _ in steps:                   # a cycle's code is read by its work states
        codes = codes or _log_facts(d)["codes"]
    for x in [doc] + [sub for _, _, sub in steps]:      # ... or by a `code` check, from the
        codes = codes or _doc_codes(x)                  # scan's own picture: no log event
    folders = [(run_dir, doc)] + [(d, sub) for _, d, sub in _speaker_first(steps, doc)]
    pre = doc.get("preset") if isinstance(doc.get("preset"), dict) else {}

    def said(section: str, key: str):
        """A fact of the document, else of the cycle's work states (speaker first)."""
        return _first((x.get(section) or {}).get(key) for _, x in folders
                      if isinstance(x.get(section), (dict, type(None))))

    if measure:
        vals = [_value("", "", doc.get("what"), v.get("name"), v.get("value"), v.get("units"),
                       None, None, NO_JUDGMENT)
                for v in doc.get("values") or [] if isinstance(v, dict)]
        judgment = NO_JUDGMENT
        reason = "" if doc.get("status") == "MEASURED" else _short(
            f"{doc.get('reason') or 'not measurable'}: {doc.get('message') or ''}")
        part, program = doc.get("label"), f"measure:{doc.get('what')}"
        preset, rig = pre.get("name"), pre.get("geometry_id")
    else:
        vals = ([v for sid, _, sub in steps for v in _run_values(sub, sid)]
                if job.get("type") == "flow" else _run_values(doc))
        judgment = JUDGMENTS.get(doc.get("disposition"), "PULL")
        reason = _run_reason(doc, vals)
        if facts["aborted"] is not None:    # ABORT after the verdict: the part is PULLed,
            pulls = [r for r in doc.get("pull_reasons") or []       # whatever result.json says
                     if isinstance(r, dict)] if judgment == "PULL" else []
            judgment = "PULL"
            reason = _pull_text([*pulls, {"reason": "aborted", "message": facts["aborted"]}])
        part = (doc.get("part") or {}).get("label") if isinstance(doc.get("part"), dict) else None
        program = _program(doc, job)
        preset, rig = said("capture", "preset"), said("capture", "geometry_id")
    for v in vals:
        v["inspection_id"] = iid
    end = doc.get("finished") or doc.get("started")
    sw = doc.get("software") if isinstance(doc.get("software"), dict) else {}
    picture = _first(_picture_of(d, x) for d, x in folders)
    profile = _first((d / PROFILE_X) if (d / PROFILE_X).is_file() else None for d, _ in folders)
    row = {
        "time": end, "inspection_id": iid, "part_id": " | ".join(codes), "part": part,
        "program": program, "mode": doc.get("mode"), "judgment": judgment, "reason": reason,
        "n_values": len(vals), "n_out_of_limit": sum(v["result"] == "OUT" for v in vals),
        "values": _summary(vals), "time_s": _time_s(run_dir, doc, first_ts),
        "image_file": _image(library, iid, picture, copy_picture and enabled()),
        "profile_file": _link(profile, library),
        "data_3d_file": _link(_first(_ply_of(d, x, library) for d, x in folders), library),
        "result_file": _link(run_dir / RESULT_FILE, library)
        if (run_dir / RESULT_FILE).is_file() else "",
        "preset": preset, "calibration_id": said("calibration", "calibration_id"), "rig": rig,
        "software": " ".join(str(x) for x in (sw.get("package_version"), sw.get("git_commit"))
                             if x)}
    return row, vals


def _program(doc: dict, job: dict) -> str:
    """What was checked (the module docstring's `program`)."""
    rec = doc.get("recipe") if isinstance(doc.get("recipe"), dict) else {}
    recipe = str(rec.get("name") or job.get("recipe") or "")
    if recipe.lower().endswith((".json", ".yaml", ".yml")):     # refused before it was loaded
        recipe = Path(recipe).stem
    if job.get("type") == "recipe" and recipe and not job.get("work_state"):
        return recipe
    bits = [doc.get("station_config") or doc.get("station")]
    if doc.get("profile") and (job.get("type") == "flow" or job.get("work_state")):
        bits.append(doc["profile"])
    bits.append(job.get("work_state"))
    return " / ".join(str(b) for b in bits if b)


# ------------------------------------------------------------------ the files

def month_of(time_text) -> str:
    """YYYY-MM of an inspection's time (UTC); this month when it has none."""
    m = _MONTH.match(str(time_text or ""))
    return m.group(0) if m else datetime.now(timezone.utc).strftime("%Y-%m")


def files_for(library, month: str) -> tuple:
    """(the inspections file, the values file) of a month."""
    events = Path(library) / EVENTS_DIR
    return events / f"{INSPECTIONS}{month}.csv", events / f"{VALUES}{month}.csv"


def _fields_of(name: str) -> Optional[tuple]:
    if name.startswith(INSPECTIONS):
        return INSPECTION_FIELDS
    return VALUE_FIELDS if name.startswith(VALUES) else None


def _open(path: Path, mode: str):
    """Every file the log writes is opened here: with the log off (SAS_EVENTLOG) it
    refuses, so nothing is written, anywhere."""
    if not enabled():
        raise PermissionError(f"the inspection event log is off ({ENV}=off)")
    path.parent.mkdir(parents=True, exist_ok=True)
    return open(path, mode)


def _move(src: Path, dst: Path, tries: int = 1) -> None:
    """One move (the old name or the new one, never half a file). tries: a file that another
    program reads this very instant cannot be moved on Windows; it can a moment later."""
    if not enabled():
        raise PermissionError(f"the inspection event log is off ({ENV}=off)")
    dst.parent.mkdir(parents=True, exist_ok=True)
    for left in range(tries - 1, -1, -1):
        try:
            os.replace(src, dst)
            return
        except OSError:
            if not left:
                raise
            time.sleep(5 * LOCK_POLL_S)


def _lock(fh) -> None:
    if os.name == "nt":
        import msvcrt
        fh.seek(0)
        msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
    else:  # pragma: no cover - the bench is Windows
        import fcntl
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock(fh) -> None:
    if os.name == "nt":
        import msvcrt
        fh.seek(0)
        msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
    else:  # pragma: no cover
        import fcntl
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


@contextmanager
def _locked(events: Path, wait_s: Optional[float] = None, name: str = LOCK_FILE):
    """An OS file lock (one writer at a time, across processes; the OS drops it when the
    holder dies): events/eventlog.lock for the month files, events/pending.lock for the
    waiting rows in events/pending/ (taken alone, or inside the first: never the other way
    round). Yields False when it could not be had in wait_s: do not write what it guards."""
    wait_s = LOCK_WAIT_S if wait_s is None else wait_s
    fh = None
    try:
        fh = _open(events / name, "a+b")
        end = time.monotonic() + wait_s
        while True:
            try:
                _lock(fh)
                break
            except OSError:
                if time.monotonic() >= end:
                    fh.close()
                    fh = None
                    break
                time.sleep(LOCK_POLL_S)
    except OSError:
        fh = None
    try:
        yield fh is not None
    finally:
        if fh is not None:
            try:
                _unlock(fh)
            except OSError:
                pass
            fh.close()
            # the lock is not a queue: a writer that came straight back would shut the
            # others out, so it stays away for longer than they wait between two tries
            time.sleep(2 * LOCK_POLL_S)


def _append(path: Path, fields: tuple, body: bytes) -> None:
    """Whole lines onto the end of a file in ONE write (OSError when it cannot be opened for
    writing). A new file gets its header first; a file whose last line was cut short gets a
    line end first, so the new rows stay whole."""
    head = b""
    try:
        size = path.stat().st_size
    except OSError:
        size = 0
    if size == 0:
        head = _BOM + _encode(fields, [], header=True)
    elif _ENDED.get(str(path)) != size:     # not as this process left it: look at its end
        try:
            with open(path, "rb") as f:
                f.seek(-1, os.SEEK_END)
                if f.read(1) != b"\n":
                    head = b"\r\n"
        except OSError:
            pass
    with _open(path, "ab") as f:
        f.write(head + body)
    _ENDED[str(path)] = size + len(head) + len(body)


_ENDED: dict = {}                       # file -> its size when this process last ended it whole


def _data_lines(raw: bytes) -> bytes:
    """A csv file's bytes without its header line (and mark)."""
    if raw.startswith(_BOM):
        raw = raw[len(_BOM):]
    head, sep, rest = raw.partition(b"\n")
    return rest if sep else b""


@contextmanager
def _locked_all(events: Path, wait_s: float):
    """The lock of the month files and the lock of the waiting rows, in that order: while
    both are held nobody writes a row anywhere, and every append is whole. Yields False
    when one of them could not be had."""
    with _locked(events, wait_s) as held:
        if not held:
            yield False
            return
        with _locked(events, wait_s, PENDING_LOCK_FILE) as both:
            yield both


def _to_pending(events: Path, parts: list, wait_s: Optional[float] = None) -> tuple:
    """Rows that cannot be added to their files now ([(file, fields, lines, how many rows)])
    -> events/pending/<same names>: whole lines, all of them under one hold of the waiting
    rows' lock (which is only ever held for a moment - this, one rename, a rebuild's cut or
    swap - so it can be had while the log's lock cannot).
    -> (rows now waiting, None or why the others are not)."""
    n = 0
    wait_s = LOCK_WAIT_S if wait_s is None else wait_s
    with _locked(events, wait_s, PENDING_LOCK_FILE) as held:
        if not held:
            return n, f"no lock on {events / PENDING_LOCK_FILE} in {wait_s:g} s"
        try:
            for path, fields, body, rows in parts:
                _append(events / PENDING_DIR / path.name, fields, body)
                n += rows
        except OSError as e:
            return n, str(e)
    return n, None


def merge_pending(library) -> int:
    """Add the waiting rows (events/pending/) to their files, where those can be written
    again -> how many files were merged. The caller holds the log's lock. A waiting file is
    taken (renamed .merging, under the waiting rows' lock: nobody is writing into it then)
    before its rows are added, so no row is ever added twice; a writer that dies in between
    leaves that file behind, and `--rebuild` brings its rows back."""
    events = Path(library) / EVENTS_DIR
    done = 0
    for p in sorted((events / PENDING_DIR).glob("*.csv")):
        fields = _fields_of(p.name)
        if fields is None:
            continue
        main = events / p.name
        taken = p.with_suffix(".merging")
        try:
            with _open(main, "ab"):         # still open elsewhere? then they keep waiting
                pass
            with _locked(events, name=PENDING_LOCK_FILE) as held:
                if not held:
                    continue
                _move(p, taken)             # ours now (fails while someone has it open)
        except OSError:
            continue
        try:
            body = _data_lines(taken.read_bytes())
            if body:
                _append(main, fields, body if body.endswith(b"\n") else body + b"\r\n")
        except OSError:
            try:
                _move(taken, p)             # back: they still wait
            except OSError:
                pass
            continue
        try:
            taken.unlink()
        except OSError:
            pass
        done += 1
    return done


def _write_last(events: Path, row: dict) -> None:
    """events/last_v1.csv: the header and the newest inspection row, replaced in one move (a
    reader sees the old file or the new one, never half of one). Never raises: the row
    itself is in the month's file."""
    last = events / LAST
    tmp = events / f"{LAST}.{os.getpid()}_{threading.get_ident()}.tmp"
    try:
        with _open(tmp, "wb") as f:
            f.write(_BOM + _encode(INSPECTION_FIELDS, [row], header=True))
        _move(tmp, last, tries=3)
    except OSError as e:
        log.warning("inspection event log: %s not written: %s", LAST, e)
        try:
            tmp.unlink()
        except OSError:
            pass


def append(library, row: dict, values: list, wait_s: Optional[float] = None) -> dict:
    """One inspection onto the month's two files, under the log's lock: the waiting rows
    first (merge_pending), then its value rows, then its inspection row. Rows that cannot
    be added now go to events/pending/<same name> instead (_to_pending): a file that cannot
    be written (open in a spreadsheet program), or no lock in wait_s (default LOCK_WAIT_S:
    another program is busy with the log) - then all of them. The inspection row is also
    written to events/last_v1.csv.
    -> {"written": rows in the files, "pending": rows now waiting, "lost": rows written
    nowhere (neither place could be written: `--rebuild` brings them back)}"""
    out = {"written": 0, "pending": 0, "lost": 0}
    if not enabled():
        return out
    wait_s = LOCK_WAIT_S if wait_s is None else wait_s
    events = Path(library) / EVENTS_DIR
    insp, vals = files_for(library, month_of(row.get("time")))
    parts = [(path, fields, _encode(fields, rows), len(rows)) for path, fields, rows in
             ((vals, VALUE_FIELDS, values), (insp, INSPECTION_FIELDS, [row])) if rows]
    with _locked(events, wait_s) as held:
        wait, why = [], "no lock"
        if held:
            merge_pending(library)
            for part in parts:
                try:
                    _append(*part[:3])
                    out["written"] += part[3]
                except OSError as e:
                    wait.append(part)
                    why = f"{part[0].name}: {e}"
        else:
            wait = parts
            log.warning("inspection event log: no lock on %s in %g s: inspection %s waits in "
                        "%s/%s/ until the next append", events / LOCK_FILE, wait_s,
                        row.get("inspection_id"), EVENTS_DIR, PENDING_DIR)
        n, why_not = _to_pending(events, wait, wait_s) if wait else (0, None)
        out["pending"] += n
        if why_not is not None:
            out["lost"] += sum(part[3] for part in wait) - n
            log.warning("inspection event log: %s; waiting rows: %s", why, why_not)
        if why_not is None or parts[-1] not in wait:    # the inspection row has a place
            _write_last(events, row)
    return out


def record(run_dir) -> Optional[dict]:
    """THE HOOK (the end of engine/runs.write_result and of Measure's _write_doc): one
    inspection into the event log of its library. It never raises and never changes a
    result or an exit code: whatever goes wrong here is logged, and `--rebuild` writes the
    row again from result.json. -> what append() says, or None (the log is off, the folder
    is the check of a work state inside a cycle, or it failed)."""
    try:
        if not enabled():
            return None
        library = library_of(run_dir)
        got = rows_of(run_dir, library)
        if got is None:
            return None
        return append(library, *got)
    except Exception as e:                  # noqa: BLE001 - the log never changes a result
        log.warning("inspection event log: %s: %s", type(e).__name__, e)
        return None


def amend(run_dir) -> Optional[dict]:
    """The row of an inspection AGAIN, as its run folder reads now: for what happened to a
    run after its result.json was written. Today that is the line counting the run ABORTED
    after its verdict: engine/runs.append_event calls this when it has appended the run
    log's `aborted_after_verdict` event, and the judgment becomes PULL with the reason. Only
    the inspection row is written again (its value rows are in the log, once); every reader
    takes the last row of an inspection id. Whoever counted the run may hold a lock of its
    own: the row waits for the log's lock for AMEND_WAIT_S only, then waits in
    events/pending/. Never raises. -> as record()."""
    try:
        if not enabled():
            return None
        library = library_of(run_dir)
        got = rows_of(run_dir, library)
        if got is None:
            return None
        return append(library, got[0], [], wait_s=AMEND_WAIT_S)
    except Exception as e:                  # noqa: BLE001 - the log never changes a result
        log.warning("inspection event log: %s: %s", type(e).__name__, e)
        return None


def read_rows(path) -> list:
    """The rows of one of the log's csv files, as text cells ([] when there is none)."""
    try:
        with open(path, encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    except OSError:
        return []


def waiting(library, month: Optional[str] = None) -> int:
    """How many inspection rows wait in events/pending/ (of one month, or of all)."""
    pend = Path(library) / EVENTS_DIR / PENDING_DIR
    return sum(len(read_rows(p)) for p in pend.glob(f"{INSPECTIONS}{month or '*'}.csv"))


# ------------------------------------------------------------------ height profiles

def write_profiles(run_dir, sections: Optional[dict]) -> list:
    """profile_x.csv / profile_y.csv in a NEW run folder: the two height lines through the
    centre of the first zone a check gridded (engine/recipes/context.keep_sections: pos_mm,
    x_mm along camera right, y_mm along camera down). One header line, then position mm,
    height mm; places with no height are left out. Nothing with the log off, nothing when no
    check gridded a zone; never raises (a file never changes a verdict). -> the names
    written."""
    out = []
    try:
        if not sections or not enabled():
            return out
        sec = next(iter(sections.values()))
        for name, key in ((PROFILE_X, "x_mm"), (PROFILE_Y, "y_mm")):
            lines = [f"{float(p):.3f},{float(h):.5f}" for p, h in zip(sec["pos_mm"], sec[key])
                     if math.isfinite(float(h))]
            if not lines:
                continue
            with open(Path(run_dir) / name, "x", encoding="utf-8", newline="") as f:
                f.write("\n".join([PROFILE_HEADER] + lines) + "\n")
            out.append(name)
    except Exception as e:                  # noqa: BLE001
        log.warning("height profile files: %s: %s", type(e).__name__, e)
    return out


# ------------------------------------------------------------------ rebuild

def _rel_path(base: Path, value) -> Optional[Path]:
    """A path of a config file: relative to that file, after env expansion."""
    if not isinstance(value, str) or not value:
        return None
    p = Path(os.path.expandvars(os.path.expanduser(value)))
    return p if p.is_absolute() else base / p


def _read_config(path: Path):
    if path.suffix.lower() == ".json":
        return _read_json(path)
    try:
        import yaml                         # optional: PyYAML
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:                       # noqa: BLE001 - not readable here: skipped
        return None


def runs_dirs_of(library, stations=None, recipes=None, extra=()) -> list:
    """Every runs folder an inspection of this library may be in: <library>/runs,
    $SCANNER_RUNS_DIR, runs/ beside the station files and the runs_dir each of them names
    (stations: default the repo's stations/), every recipe's outputs.runs_dir (recipes:
    default the repo's recipes/, and always the library's own), and `extra`. Existing
    folders only, each once."""
    library = Path(library)
    out = [library / "runs"]
    if os.environ.get("SCANNER_RUNS_DIR"):
        out.append(Path(os.environ["SCANNER_RUNS_DIR"]))
    if isinstance(stations, (str, Path)):
        stations = [stations]
    for st in [Path(d) for d in (stations or [_ROOT / "stations"])]:
        out.append(st / "runs")             # where a station file that names none puts them
        for f in sorted(st.glob("*.*")) if st.is_dir() else []:
            if f.suffix.lower() not in (".json", ".yaml", ".yml"):
                continue
            doc = _read_config(f)
            if isinstance(doc, dict):
                par = doc.get("parameters") if isinstance(doc.get("parameters"), dict) else {}
                rd = _rel_path(f.parent, par.get("runs_dir") or doc.get("runs_dir"))
                if rd is not None:
                    out.append(rd)
    files = []
    for d in [Path(r) for r in (recipes or [_ROOT / "recipes"])]:
        files += sorted(d.glob("*.json")) if d.is_dir() else []
    files += sorted((library / "recipes").glob("*/recipe.json"))
    for f in files:
        doc = _read_json(f)
        outs = doc.get("outputs") if isinstance(doc, dict) else None
        rd = _rel_path(f.resolve().parent, outs.get("runs_dir")) if isinstance(outs, dict) else None
        if rd is not None:
            out.append(rd)
    out += [Path(p) for p in extra]
    seen, dirs = set(), []
    for d in out:
        try:
            key = os.path.normcase(str(d.resolve()))
        except OSError:
            continue
        if key not in seen and d.is_dir():
            seen.add(key)
            dirs.append(d)
    return dirs


def run_folders(runs_dirs) -> list:
    """The run folders directly in the given runs folders, by name: a folder with a
    result.json, or with a run log and a steps/ folder but no result.json (a cycle that
    never ended, or one still running: unended_cycle tells them apart). The step folders of
    a cycle are not listed: they belong to their cycle."""
    out = []
    for d in runs_dirs:
        try:
            out += sorted(p for p in Path(d).iterdir() if (p / RESULT_FILE).is_file()
                          or ((p / LOG_FILE).is_file() and (p / STEPS_DIR).is_dir()))
        except OSError:
            continue
    return out


def _log_files(events: Path) -> list:
    """The log's csv files as they are now: the month files, then the waiting ones."""
    return [p for p in sorted(events.glob("*.csv")) + sorted((events / PENDING_DIR).glob("*.csv"))
            if _fields_of(p.name) is not None]


def _data_rows(body: bytes, fields: tuple) -> list:
    """Whole csv lines without a header line -> rows as text cells."""
    text = body.decode("utf-8", errors="replace")
    return list(csv.DictReader(io.StringIO(text, newline=""), fieldnames=fields))


def _which(path: Path) -> tuple:
    """(what tells a file from another one of the same name, its size), asked without
    opening it: a virus scanner reads a big file that has grown each time it is opened."""
    st = os.stat(path)
    return (st.st_dev, st.st_ino), st.st_size


def _cut(events: Path) -> dict:
    """{file: (which file it is, its size, its bytes when it is a waiting file)} of the
    log's files as they are now. Taken under both locks, so every append is whole below the
    cut or whole above it. A month's file only grows and is never deleted (a rebuild moves
    it aside): its bytes below the cut are read later, without a lock. A waiting file is
    deleted when its rows are merged, and the next one of its name can look the same from
    outside: it is small, and is read here."""
    cut = {}
    for p in _log_files(events):
        try:
            which, size = _which(p)
            raw = p.read_bytes() if p.parent.name == PENDING_DIR else None
        except OSError:
            continue
        cut[p] = (which, size if raw is None else len(raw), raw)
    return cut


def _present(cut: dict) -> tuple:
    """What the log held at the cut, files and waiting rows alike, read WITHOUT a lock ->
    ({inspection id: its row - the last one written}, {inspection id: [value rows]}, {file:
    (which file, bytes read, the bytes of a waiting file)} of the files that were read). A
    month's file that is another file by now (a second rebuild swapped it meanwhile) is left
    out: the caller sees that not every file was read, and cuts and reads again - half an
    inspection must not be taken."""
    rows, values, read = {}, {}, {}
    for p, (which, size, raw) in cut.items():
        fields = _fields_of(p.name)
        head = raw
        if raw is None:
            try:
                with open(p, "rb") as f:
                    if _which(p)[0] != which:
                        continue
                    raw = f.read(size)
            except OSError:
                continue
        read[p] = (which, len(raw), head)
        for r in _data_rows(_data_lines(raw), fields):
            iid = r.get("inspection_id")
            if not iid:
                continue
            if fields is INSPECTION_FIELDS:
                rows[iid] = r               # written again (amend): the last one counts
            else:
                values.setdefault(iid, []).append(r)
    return rows, values, read


def _since(events: Path, read: dict) -> tuple:
    """The rows the log's files got since the cut (the caller holds both locks) ->
    ([(file name, inspection row)], [(file name, value row)], [the files as they are now]).
    A file that was read and is still that file gives what follows the bytes read (a
    month's file is not even opened when it has not grown; a waiting file is told by what
    it holds); any other (new since, or not read) all its rows."""
    rows, values, files = [], [], _log_files(events)
    for p in files:
        fields = _fields_of(p.name)
        which, n, head = read.get(p, (None, 0, None))
        try:
            now, size = _which(p)
            same = bool(n) and size >= n and (head is not None or which == now)
            if same and head is None and size == n:
                continue
            with open(p, "rb") as f:
                if same and head is not None:
                    same = f.read(n) == head
                f.seek(n if same else 0)
                body = f.read() if same else _data_lines(f.read())
        except OSError:
            continue
        for r in _data_rows(body, fields):
            if r.get("inspection_id"):
                (rows if fields is INSPECTION_FIELDS else values).append((p.name, r))
    return rows, values, files


def rebuild(library, runs_dirs=None, stations=None, recipes=None) -> dict:
    """Write the log's files again from the run folders (result.json is the source of
    truth). runs_dirs: the runs folders to read (default: runs_dirs_of). An inspection whose
    run folder is found is written from it (a cycle that never ended: as PULL,
    unended_cycle); a row of the present files that no run folder gave (the folder is gone,
    or its run ended while the folders were read) is kept as it is (`kept`).
    The files it replaces (and the waiting rows) are moved to events/replaced_<time>/.
    Nothing is changed when one of them cannot be moved (open in a spreadsheet program):
    close it and run again.
    It may run while inspections are made: the locks are held twice, each time for a
    moment - to note how long the present files are (the cut), and at the end to add the
    rows they got since (`late`) and swap the files. Reading the run folders and the files
    up to the cut, and writing the new files, needs no lock. A run that ended a moment ago
    is waited for (HOOK_SETTLE_S): its own hook may still be writing its row.
    -> {ok, error, inspections, values, kept, late, unended, months, runs_dirs, replaced}"""
    library = Path(library)
    events = library / EVENTS_DIR
    dirs = [Path(d) for d in runs_dirs] if runs_dirs is not None else \
        runs_dirs_of(library, stations, recipes)
    out = {"ok": False, "error": None, "inspections": 0, "values": 0, "kept": 0, "late": 0,
           "unended": 0, "months": [], "runs_dirs": [str(d) for d in dirs], "replaced": None}
    if not enabled():
        return {**out, "error": f"the inspection event log is off ({ENV}=off): nothing written"}
    busy = f"no lock on {events / LOCK_FILE}: another program is writing the log"
    found, newest = {}, 0.0
    for d in run_folders(dirs):
        try:
            got = rows_of(d, library)
        except Exception as e:              # noqa: BLE001 - one bad folder never stops the rest
            log.warning("inspection event log: %s skipped: %s: %s", d, type(e).__name__, e)
            continue
        if got is None or got[0]["inspection_id"] in found:
            continue
        found[got[0]["inspection_id"]] = got
        try:
            newest = max(newest, (d / RESULT_FILE).stat().st_mtime)
        except OSError:                     # no result.json: a cycle that never ended
            out["unended"] += 1
    # a run that ended a moment ago: its row is below the cut once its own hook is done
    time.sleep(min(max(newest + HOOK_SETTLE_S - time.time(), 0.0), HOOK_SETTLE_S))
    for unlocked in (True, True, True, False):
        with _locked_all(events, 10.0) as held:
            if not held:
                return {**out, "error": busy}
            cut = _cut(events)
            if not unlocked:                # the files moved each time: read them held
                old_rows, old_values, read = _present(cut)
        if unlocked:
            old_rows, old_values, read = _present(cut)
        if len(read) == len(cut):           # every file was read as it was measured
            break
    kept = {i: (r, old_values.get(i, [])) for i, r in old_rows.items() if i not in found}
    have = {**found, **kept}
    months: dict = {}
    for iid, (row, vals) in have.items():
        months.setdefault(month_of(row.get("time")), []).append((row, vals))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    n = 0
    while any((events / f"{kind}_{stamp}").exists() for kind in ("rebuild", "replaced")):
        n += 1                              # twice in one second: never onto the first
        stamp = stamp.split("-")[0] + f"-{n}"
    stage, aside = events / f"rebuild_{stamp}", events / f"replaced_{stamp}"
    staged = set()                          # the names of the new files, written beside
    try:
        for month, items in sorted(months.items()):
            items.sort(key=lambda it: (str(it[0].get("time") or ""), it[0]["inspection_id"]))
            insp, vals = files_for(library, month)
            for path, fields, rows in (
                    (vals, VALUE_FIELDS, [v for _, vs in items for v in vs]),
                    (insp, INSPECTION_FIELDS, [r for r, _ in items])):
                with _open(stage / path.name, "xb") as f:
                    f.write(_BOM + _encode(fields, rows, header=True))
                staged.add(path.name)
                out["values" if fields is VALUE_FIELDS else "inspections"] += len(rows)
    except OSError as e:
        shutil.rmtree(stage, ignore_errors=True)
        return {**out, "error": str(e)}
    with _locked_all(events, 10.0) as held:
        if not held:
            shutil.rmtree(stage, ignore_errors=True)
            return {**out, "error": busy}
        try:
            # what was appended since the cut: an inspection the new files do not have, or
            # have with another row (written again: the last row counts), and its values
            late_rows, late_values, files = _since(events, read)
            add, last = {}, {}              # file name -> rows; inspection id -> its last row
            for name, r in late_rows:
                iid = r["inspection_id"]
                if iid not in last and iid in have:
                    last[iid] = [cell(have[iid][0].get(k)) for k in INSPECTION_FIELDS]
                cells = [cell(r.get(k)) for k in INSPECTION_FIELDS]
                if last.get(iid) != cells:
                    last[iid] = cells
                    add.setdefault(name, []).append(r)
                    out["late"] += 1
            for name, v in late_values:
                if v["inspection_id"] not in have:
                    add.setdefault(name, []).append(v)
            for name, rows in sorted(add.items()):
                fields = _fields_of(name)
                _append(stage / name, fields, _encode(fields, rows))
                staged.add(name)
                months.setdefault(name[-11:-4], [])
                out["values" if fields is VALUE_FIELDS else "inspections"] += len(rows)
            moved = []
            for src, dst in [(p, aside / p.relative_to(events)) for p in files] + [
                    (stage / name, events / name) for name in sorted(staged)]:
                try:
                    _move(src, dst, tries=3)
                except OSError as e:
                    for was, now in reversed(moved):    # put everything back: nothing changed
                        os.replace(now, was)
                    raise OSError(f"{src.name} could not be moved ({e}): close the program "
                                  "that has it open and run again; nothing was changed") from None
                moved.append((src, dst))
        except OSError as e:
            out["error"] = str(e)
        shutil.rmtree(stage, ignore_errors=True)
        for d in (aside / PENDING_DIR, aside):      # only when nothing was moved there
            try:
                d.rmdir()
            except OSError:
                pass
        if out["error"]:
            return out
    out.update(ok=True, kept=len(kept), months=sorted(months),
               replaced=_link(aside, library) if aside.is_dir() else None)
    return out


# ------------------------------------------------------------------ command line

def _default_library() -> Path:
    from . import api                       # the engine's default ($SWISS_SCANNER_LIBRARY)
    return Path(api.DEFAULT_ROOT)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m engine.eventlog",
        description="The inspection event log of a library (events/inspections_v1_<month>.csv "
                    "and events/inspection_values_v1_<month>.csv). Touches no rig.")
    do = ap.add_mutually_exclusive_group(required=True)
    do.add_argument("--rebuild", action="store_true",
                    help="write both files again from the run folders (result.json is the "
                         "source of truth); a cycle that never ended is listed as PULL; the "
                         "files replaced are moved aside, never deleted. May run while "
                         "inspections are made")
    do.add_argument("--merge", action="store_true",
                    help="add the rows waiting in events/pending/ to their files now")
    ap.add_argument("--library", metavar="DIR",
                    help="the library (default: the engine's, or $SWISS_SCANNER_LIBRARY)")
    ap.add_argument("--runs", metavar="DIR", action="append", default=[],
                    help="one more runs folder to read (--rebuild; may be given again)")
    ap.add_argument("--stations", metavar="DIR", action="append",
                    help="a folder of station files (--rebuild; default: the repo's "
                         "stations/; may be given again): the runs folder of each is read")
    ap.add_argument("--recipes", metavar="DIR", action="append",
                    help="a folder of recipe files (--rebuild; default: the repo's recipes/): "
                         "each recipe's outputs.runs_dir is read")
    a = ap.parse_args(argv)
    library = Path(a.library) if a.library else _default_library()
    if not library.is_dir():
        print(f"no library folder {library}", file=sys.stderr)
        return 2
    if not enabled():
        print(f"the inspection event log is off ({ENV}=off): nothing written", file=sys.stderr)
        return 2
    if a.merge:
        with _locked(library / EVENTS_DIR, wait_s=10.0) as held:
            if not held:
                print("another program is writing the log: nothing merged", file=sys.stderr)
                return 1
            n = merge_pending(library)
        left = waiting(library)
        print(f"{n} file(s) of waiting rows merged; {left} row(s) still waiting"
              + (" (their file is open in another program)" if left else ""))
        return 1 if left else 0
    r = rebuild(library, runs_dirs_of(library, a.stations, a.recipes, a.runs))
    for d in r["runs_dirs"]:
        print(f"read {d}")
    if not r["ok"]:
        print(f"not rebuilt: {r['error']}", file=sys.stderr)
        return 1
    print(f"{r['inspections']} inspection row(s), {r['values']} value row(s) in "
          f"{len(r['months'])} month(s)"
          + (f"; {r['kept']} row(s) kept as they were (no run folder read gave them)"
             if r["kept"] else "")
          + (f"; {r['unended']} cycle(s) that never ended listed as PULL" if r["unended"] else "")
          + (f"; {r['late']} row(s) written while this ran taken over" if r["late"] else "")
          + (f"; the files replaced are in {r['replaced']}" if r["replaced"] else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())

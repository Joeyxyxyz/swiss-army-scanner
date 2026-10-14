"""
The bad-part output, and references made from good parts only.

BAD-PART OUTPUT (off unless a folder is set). Every inspection whose disposition is REJECT or
PULL gets a folder there, so a factory can track its bad parts; the good parts are what the
scanner checks against.

    <folder>/BAD_PART_OUTPUT.txt          what the folder is: written first, it marks the
                                          folder (nothing in it is ever a reference)
    <folder>/bad_parts_v1.csv             the index: one row per bad part (INDEX_FIELDS)
    <folder>/<part id>/                   one folder per bad part, named by the part ID (the
                                          code read from the part, made safe as a folder
                                          name), else by the inspection id; the same part ID
                                          again gets <part id>-2, -3 ... (never an existing
                                          folder, never an overwrite)
        result.json                       the run's result (a cycle: the cycle's)
        overlay.png | deviation.png       the run's picture (a cycle: of the work state that
                                          speaks for it, as in the event log)
        profile_x.csv, profile_y.csv      the height profiles, when the run wrote them
        scan.ply                          the 3D file, when the run has one
    <folder>/bad_parts.lock               one writer of the index at a time (an OS lock)
    <folder>/pending/bad_parts_v1.csv     index rows that wait (the index was open in a
                                          spreadsheet program, or busy) until the next bad part
    <folder>/replaced_<time>/             an index changed in another program, as it was (its
                                          rows go on in a new index: eventlog._settle_header)

Where it is set - the first that says, wins:
    1. the run command's own option --bad-part-output DIR (or off): `scan run`, `scan
       recipe-run`, `scan quick-check`
    2. the station configuration the run belongs to: parameters.bad_part_output (one with
       profiles + a flow), bad_part_output (a single-job station config)
    3. the recipe: outputs.bad_part_output of the recipe run on its own or as the single job
       of a station config; for a cycle of a flow, of the recipes of the work states it
       checked (when they name different folders: the one of the work state that speaks for
       the cycle, and a warning in the cycle's output)
    4. the environment variable SAS_BAD_PART_OUTPUT: this computer's default
A folder is relative to the file that names it (as runs_dir); "off" switches it off at that
level. The run's result.json says which folder applied: outputs.bad_part_output (null: off).
The setting is text only until a part is copied: nothing touches the folder for a good part.

A bad-part output is a folder of its own: one that is, holds or lies inside the library or a
runs folder is refused (nothing is copied; the run's output says why) - else everything the
scanner checks against would lie in a bad-part output, or the bad parts would be cleaned
with the runs.

The index row (INDEX_FIELDS): time (local, with its offset to UTC, as the event log's),
part_id, inspection_id, program, operating_mode, hardware (REAL / MOCK), disposition (REJECT
or PULL, never merged: a run the line counted ABORTED after its verdict is a PULL), reason,
out_of_limit (the values outside their limits: where: name=value unit [lower..upper]),
folder (the part's folder, relative to the bad-part output). Cells as in the event log: a
text a spreadsheet would run gets a leading apostrophe. An index whose header was changed in
another program (a column added and saved) is moved to replaced_<time>/ and its rows go on
in a new index; the run's output says so (work on a copy of the index).

Rules of the output:
  * It never changes a result or an exit code. The hook (record) is called after result.json
    is written (engine/runs.write_result; again when the line counts a run ABORTED after its
    verdict, engine/runs.append_event; and Measure's, which writes nothing: a measurement
    judges nothing), never raises, and reads the document back from the file.
  * A good part (CONTINUE) writes nothing; a measurement writes nothing; the check of a work
    state inside a cycle writes nothing there (its cycle is the inspection).
  * It works with the inspection event log off (SAS_EVENTLOG): its own files, its own lock.
  * Once per inspection: the run folder notes where its copy went (bad_part_output.json), so
    a second call (ABORT after the verdict) copies nothing again.
  * A file is copied under a temporary name and then renamed: a part's file is whole or
    not there. A copy cut short is made again from the start.
  * A write that fails is reported in the run's output (a warning in the result the run
    returns and prints; result.json stays as it was written) and waits: a note in
    <library>/bad_part_pending/<inspection id>.json. The folder is tried ONCE per bad part:
    when it cannot be reached, the part waits at once (a network folder that is down holds
    a bad part up by one attempt, a good part by none). When it can, the copies that wait
    for it go first - at most RETRY_MAX of them, for at most RETRY_BUDGET_S - then this
    part's. `python -m engine.badparts --retry` copies all that wait. A part folder half
    filled is filled up, not made twice. A path longer than Windows allows (260 characters)
    is refused with that reason.

REFERENCES ARE GOOD PARTS ONLY. A part judged REJECT or PULL is kept for tracking, never
checked against. Every verb that makes a reference a later part is checked against (a
recipe's golden, the golden end face, the station golden, the good-part memory bank, a
taught alignment reference, the dent null calibration) refuses a source judged bad, and
every verb that compares a part with a reference refuses a reference whose source was
(require_good). There is no override. Judged bad (findings):
  * by an inspection: the hook marks every scan a REJECT / PULL run judged (part.scan_id, a
    cycle's work states' scans, and the scans each was made from: meta.json source_scan) in
    its library entry: bad_part.jsonl, one line per inspection, with the inspection's reason;
  * by a check verb whose result is recorded against the entry (its exports.jsonl names
    it): a disposition REJECT / PULL (face check, station check: INCOMPLETE is a PULL), else
    a definite bad verdict of a verb that gives no disposition (FAIL, FLAG: inspect,
    compare, bank check, dent detection; their INDETERMINATE judged no part);
  * a file or folder inside a run folder whose result was REJECT / PULL, or inside a
    bad-part output (BAD_PART_OUTPUT.txt in a folder above it);
  * a scan MADE FROM a scan judged bad (meta.json source_scan, followed up to MAX_LINEAGE
    steps: a scan triangulated again after a recalibration is the same part);
  * the run folders (sweep: runs made before the marks existed, or by an older build): every
    run folder the event log's rebuild reads whose result.json judged the scan REJECT / PULL.
    When a reference is MADE, all of them; when one is USED (a golden named by id, a job's
    golden, a compare), the runs folders not searched for that scan before - the search is
    noted in its entry (bad_part_swept.json) and what it found is marked, so a scan is
    searched once, not at every part.
A source with no recorded judgment (a plain scan of a good part) is used as before.

    python -m engine.badparts --retry [--library DIR]        the copies that wait, now
    python -m engine.badparts --copy RUN_FOLDER [--to DIR]   one inspection's copy, by hand

No hardware; nothing of the engine is loaded with it but eventlog / disposition / rules.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from . import eventlog as E
from . import rules
from .library import longpath

log = logging.getLogger(__name__)

ENV = "SAS_BAD_PART_OUTPUT"             # this computer's default folder ("off": none)
KEY = "bad_part_output"                 # recipe outputs / station configuration / result outputs
OFF = "off"
FORMAT = E.FORMAT
INDEX = f"bad_parts_{FORMAT}.csv"
INDEX_FIELDS = ("time", "part_id", "inspection_id", "program", "operating_mode", "hardware",
                "disposition", "reason", "out_of_limit", "folder")
MARKER = "BAD_PART_OUTPUT.txt"          # at the top of the output: what the folder is
LOCK_FILE = "bad_parts.lock"            # one writer of the index at a time
PENDING_LOCK_FILE = "bad_parts_pending.lock"    # one writer of the waiting index rows
PENDING_DIR = E.PENDING_DIR             # <output>/pending/: index rows that wait
DONE_FILE = "bad_part_output.json"      # in the run folder: where its copy went
QUEUE_DIR = "bad_part_pending"          # in the library: inspections whose copy waits
STAMP = "bad_part.jsonl"                # in a scan's library entry: the inspections that
                                        # judged its part REJECT / PULL
SWEPT = "bad_part_swept.json"           # in a scan's library entry: the runs folders searched
                                        # for inspections that judged it (findings, sweep)
BAD = ("REJECT", "PULL")
WARNING = "bad-part output"             # how its warnings in a run's output begin
MAX_NAME = 64                           # a part's folder name, characters
MAX_PATH = 259                          # Windows: 260 with the closing null
MAX_OUT = 12                            # values in the out_of_limit cell
MAX_LINEAGE = 8                         # source_scan links followed (a scan made from a scan)
RETRY_MAX = 5                           # copies that wait, tried by one bad part's hook ...
RETRY_BUDGET_S = 2.0                    # ... started within this long (`--retry`: all)
PLY_NAME = "scan.ply"                   # the 3D file in a part's folder
PICTURE_NAME = "overlay.png"            # a picture by another name in the run folder
PART_SUFFIX = ".part"                   # a file being copied (renamed when it is whole)
JUDGING_KINDS = ("defects", "comparisons", "anomalies")     # output bin kinds that judge
REJECTS = ("FAIL", "REJECT", "FLAG")    # a check verb's definite bad verdicts
RULE = ("references are made from GOOD parts only: a part judged REJECT or PULL is kept for "
        "tracking, never checked against")
HOW_MAKE = ("make it from parts that passed (CONTINUE), or from scans of good parts that no "
            "check has judged")
HOW_USE = "make the reference again from good parts"
HOW_GOLDEN = "name the scan of a good part instead (one that passed, or that no check has judged)"
MARKER_TEXT = (
    "Bad-part output of the scanner (engine/badparts.py).\n\n"
    "One folder per part judged REJECT or PULL: its result.json, its picture, its height\n"
    "profiles and its 3D file. bad_parts_v1.csv lists them, one row per bad part.\n"
    "Nothing in this folder is ever used as a reference: references are made from good\n"
    "parts only.\n")
_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]+")
_RESERVED = re.compile(r"^(CON|PRN|AUX|NUL|COM[0-9]|LPT[0-9])(\..*)?$", re.IGNORECASE)
# the output's own names: never a part's folder (it would block the index or its lock)
_OWN_NAMES = {n.lower() for n in (MARKER, INDEX, LOCK_FILE, PENDING_LOCK_FILE, PENDING_DIR)}
_COMMAND_LINE: Optional[str] = None     # the run command's --bad-part-output (set_command_line)


# ------------------------------------------------------------------ where it goes

def set_command_line(value) -> None:
    """The --bad-part-output of the run command this process runs: a folder, "off", or None
    (not given). It wins over the configuration and the environment."""
    global _COMMAND_LINE
    _COMMAND_LINE = None if value in (None, "") else str(value)


def configured(value, base_dir=None) -> Optional[str]:
    """A bad-part output as a configuration names it -> "off", a full folder path, or None
    (not set). A relative folder is relative to base_dir (the folder of the file that names
    it; else the current folder), after ~ and environment variables. Text only: the folder
    is not touched (a network folder that is down holds nothing up here). ValueError for
    anything but text or null; never another error."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{KEY} must be a folder path, \"{OFF}\" or null, got {value!r}")
    v = value.strip()
    if not v:
        return None
    if v.lower() == OFF:
        return OFF
    p = os.path.expandvars(os.path.expanduser(v))
    try:
        if not os.path.isabs(p) and base_dir is not None:
            p = os.path.join(str(base_dir), p)
        return os.path.abspath(p)
    except (OSError, ValueError):           # no current folder: the text as it is
        return p


def _destination(result: dict) -> tuple:
    """(the folder a run's bad part goes to | None, a warning | None): the command line, else
    what the run's configuration put in outputs.bad_part_output, else SAS_BAD_PART_OUTPUT
    (read at every call). The first level that says anything decides: "off" is off, and a
    value that is no folder path copies nothing and says so (never the next level)."""
    outs = result.get("outputs") if isinstance(result.get("outputs"), dict) else {}
    for where, v in (("--bad-part-output", _COMMAND_LINE), (f"outputs.{KEY}", outs.get(KEY)),
                     (ENV, os.environ.get(ENV))):
        try:
            c = configured(v)
        except ValueError as e:
            return None, f"{WARNING}: {where}: {e}: no part is copied"
        if c is not None:
            return (None if c == OFF else c), None
    return None, None


def destination(result: dict) -> Optional[str]:
    """The folder a run's bad part goes to, or None (off; _destination)."""
    return _destination(result)[0]


def settle(result: dict) -> None:
    """outputs.bad_part_output of a run document about to be written: the folder its part
    goes to when it is judged REJECT or PULL (destination), else null; a setting that is no
    folder path is said in its warnings. Never raises, never touches the folder."""
    try:
        if not isinstance(result.get("outputs"), dict):
            result["outputs"] = {}
        result["outputs"][KEY], why = _destination(result)
        if why and isinstance(result.get("warnings"), list):
            result["warnings"].append(why)
    except Exception as e:                  # noqa: BLE001 - never stops a result
        log.warning("%s: %s: %s", WARNING, type(e).__name__, e)


def _key(p) -> str:
    """A path as two paths are compared here: absolute, case as the OS takes it (text only)."""
    return os.path.normcase(os.path.abspath(str(p)))


def _within(inner, outer) -> bool:
    """Is `inner` the folder `outer` or inside it (by their text)?"""
    i, o = _key(inner), _key(outer)
    return i == o or i.startswith(o.rstrip("\\/") + os.sep)


def _forms(p) -> set:
    """A local folder's paths as compared here: as given, and with links resolved."""
    out = {_key(p)}
    try:
        out.add(os.path.normcase(os.path.realpath(str(p))))
    except (OSError, ValueError):
        pass
    return out


def _clash(it: dict, root) -> Optional[str]:
    """Why the folder `root` cannot be the bad-part output of the inspection `it`, or None:
    it is, holds or lies inside the library or a runs folder (the run's own; every one the
    event log's rebuild reads). Text only for root (a network folder is not touched)."""
    places = [("the library", Path(it["library"])),
              ("the runs folder", Path(it["run_dir"]).parent)]
    try:
        places += [("the runs folder", d) for d in E.runs_dirs_of(it["library"])]
    except Exception:                       # noqa: BLE001 - the two above still count
        pass
    r = _key(root)
    for what, d in places:
        for f in _forms(d):
            if _within(r, f) or _within(f, r):
                how = ("is" if _key(f) == r else "holds" if _within(f, r) else "lies inside")
                return (f"the folder {how} {what} {d}: a bad-part output must be a folder of "
                        "its own, outside the library and the runs folders (else what the "
                        "scanner checks against would lie in it, or the bad parts would be "
                        "cleaned with the runs) - set another folder")
    return None


# ------------------------------------------------------------------ the hook

def record(run_dir) -> Optional[dict]:
    """THE HOOK (engine/runs.write_result after result.json, engine/runs.append_event after an
    ABORT after the verdict, Measure's _write_doc): a run judged REJECT or PULL is marked on
    the scans it judged (bad_part.jsonl in their library entries) and copied to its bad-part
    output (outputs.bad_part_output): the folder is tried once (_ready); when it answers, the
    copies that wait for it go first (retry: RETRY_MAX, RETRY_BUDGET_S), then this one.
    Never raises, never changes a result. -> None (nothing to do: a good part, a
    measurement, no run) or {disposition, stamped: [entry folders], folder: the part's
    folder | None, index: "written" | "waiting" | None, warnings: [what the run's output
    says]}."""
    try:
        run_dir = Path(run_dir)
        doc = E._read_json(run_dir / E.RESULT_FILE)
        if isinstance(doc, dict) and E.in_cycle(run_dir, doc):
            # the check of a work state inside a cycle: its own judgment marks its scan; the
            # cycle is the inspection (copied when the cycle is written)
            disp = _bad(doc.get("disposition"))
            if disp is None:
                return None
            stamped, warn = stamp(E.library_of(run_dir), doc, run_dir, disp)
            return {"disposition": disp, "stamped": stamped, "folder": None, "index": None,
                    "warnings": warn}
        it = E.inspection(run_dir)
        if it is None or it["disposition"] not in BAD:
            return None
        iid, disp = it["inspection_id"], it["disposition"]
        out = {"disposition": disp, "stamped": [], "folder": None, "index": None,
               "warnings": []}
        out["stamped"], out["warnings"] = stamp(it["library"], it["doc"], run_dir, disp,
                                                it["reason"])
        outs = it["doc"].get("outputs") if isinstance(it["doc"].get("outputs"), dict) else {}
        root = outs.get(KEY)
        if not root or root == OFF:
            return out
        root = Path(root)
        clash = _clash(it, root)
        if clash is not None:
            out["warnings"].append(
                f"{WARNING}: inspection {iid} ({disp}) NOT copied to {root}: {clash}; copy it "
                f"there by hand with `python -m engine.badparts --copy {run_dir} --to DIR`; "
                "the judgment is unchanged")
        else:
            try:
                _ready(root)                # ONE attempt at the folder
            except OSError as e:
                out["warnings"].append(_not_copied(it, root, _why(e), None))
            else:
                retry(it["library"], root=root, limit=RETRY_MAX, budget_s=RETRY_BUDGET_S)
                try:
                    got = _copy(it, root)
                    out.update(folder=got["folder"], index=got["index"])
                    out["warnings"] += got["notes"]
                    if got["index"] == "waiting":
                        out["warnings"].append(
                            f"{WARNING}: {got['folder']} written; its row waits in "
                            f"{root / PENDING_DIR / INDEX} ({got['why']}): it is added to "
                            f"{INDEX} with the next bad part once no other program has that "
                            "file open")
                except OSError as e:
                    out["warnings"].append(_not_copied(it, root, _why(e),
                                                       getattr(e, "bad_part_folder", None)))
        for w in out["warnings"]:
            log.warning("%s", w)
        return out
    except Exception as e:                  # noqa: BLE001 - never changes a result
        msg = f"{WARNING}: {type(e).__name__}: {e} (the judgment is unchanged)"
        log.warning("%s", msg)
        return {"disposition": None, "stamped": [], "folder": None, "index": None,
                "warnings": [msg]}


def _not_copied(it: dict, root: Path, why: str, folder) -> str:
    """A copy that failed: queued (_queue) -> the warning for the run's output."""
    queued = _queue(it, root, why, folder)
    return (f"{WARNING}: inspection {it['inspection_id']} ({it['disposition']}) NOT copied "
            f"to {root}: {why} - "
            + (f"it waits ({queued}) and is copied with the next bad part, or now with "
               "`python -m engine.badparts --retry`" if queued else
               "and it could not be queued either: copy it with `python -m "
               f"engine.badparts --copy {it['run_dir']}`")
            + "; the judgment is unchanged")


def _why(e: OSError) -> str:
    if longpath.too_long(e):                # a path the OS can't take: its length, the limit
        return longpath.explain(e, fix="give the bad-part output a folder with a shorter path, "
                                       "nearer the drive root, or have Windows long paths "
                                       "turned on, then copy it again").rstrip(".")
    return " ".join(str(e).split()) or type(e).__name__


def _bad(disposition) -> Optional[str]:
    """A run document's disposition -> REJECT / PULL (any value that is none of the three is
    a PULL: never a pass by default), or None for CONTINUE and a measurement's NONE."""
    if disposition in ("CONTINUE", "NONE"):
        return None
    return "REJECT" if disposition == "REJECT" else "PULL"


# ------------------------------------------------------------------ marks on the scans

def judged_scans(doc: dict, run_dir=None) -> list:
    """The scan ids a run judged: its part.scan_id, and a cycle's work states' (their
    dlp_result and their run folders), each once."""
    ids = []
    part = doc.get("part") if isinstance(doc.get("part"), dict) else {}
    ids.append(part.get("scan_id"))
    for st in doc.get("steps") or []:
        dlp = st.get("dlp_result") if isinstance(st, dict) else None
        if isinstance(dlp, dict):
            ids.append(dlp.get("scan_id"))
    job = doc.get("job") if isinstance(doc.get("job"), dict) else {}
    if run_dir is not None and job.get("type") == "flow":
        for _, _, sub in E._steps(Path(run_dir), doc):
            ids.append((sub.get("part") or {}).get("scan_id")
                       if isinstance(sub.get("part"), dict) else None)
    return list(dict.fromkeys(i for i in ids if isinstance(i, str) and E._SAFE_ID.match(i)))


def scan_entries(library, scan_id: str, geometry_id=None) -> list:
    """The library entry folders of a scan id: metric/<geometry>/<id>, objects/<id>."""
    library = Path(library)
    if not isinstance(scan_id, str) or not E._SAFE_ID.match(scan_id):
        return []
    found = []
    if isinstance(geometry_id, str) and E._SAFE_ID.match(geometry_id):
        found.append(library / "metric" / geometry_id / scan_id)
    found += sorted((library / "metric").glob(f"*/{scan_id}"))
    found.append(library / "objects" / scan_id)
    out = []
    for d in found:
        if d.is_dir() and not any(os.path.normcase(str(d)) == os.path.normcase(str(x))
                                  for x in out):
            out.append(d)
    return out


def _lineage(library, places) -> list:
    """The scans the entries `places` were made from (meta.json source_scan: a scan
    triangulated again, a face target), and theirs, up to MAX_LINEAGE steps:
    [(scan id, its entry folders in this library)] - an id whose entries are gone too."""
    out, seen = [], {Path(d).name for d in places}
    todo = [Path(d) for d in places]
    for _ in range(MAX_LINEAGE):
        nxt = []
        for d in todo:
            meta = E._read_json(d / "meta.json") if d.is_dir() else None
            src = meta.get("source_scan") if isinstance(meta, dict) else None
            if not isinstance(src, str) or not E._SAFE_ID.match(src) or src in seen:
                continue
            seen.add(src)
            dirs = scan_entries(library, src, meta.get("geometry_id"))
            out.append((src, dirs))
            nxt += dirs
        if not nxt:
            break
        todo = nxt
    return out


def stamp(library, doc: dict, run_dir, disposition: str, reason=None) -> tuple:
    """Mark every scan a REJECT / PULL run judged (judged_scans), and the scans each was made
    from (_lineage: the same part), in its library entry: one line in bad_part.jsonl per
    inspection (written once), with the inspection's reason (default: what result.json
    says). -> (the entries marked, warnings for the run's output: an entry that could not be
    written). A scan with no entry in this library (another library, trashed) is noted in
    the log only: a reference made from it later still finds the run (the sweep)."""
    marked, warn = [], []
    iid = doc.get("run_id")
    geo = (doc.get("capture") or {}).get("geometry_id") if isinstance(doc.get("capture"),
                                                                       dict) else None
    why = E._short(reason if reason is not None else E._run_reason(doc, []), 200)
    for sid in judged_scans(doc, run_dir):
        entries = scan_entries(library, sid, geo)
        if not entries:
            log.info("%s: scan %s of inspection %s has no entry in %s: not marked", WARNING,
                     sid, iid, library)
        targets = [(d, sid) for d in entries] + [(d, sid) for _, dirs in
                                                 _lineage(library, entries) for d in dirs]
        for d, judged in targets:
            f = d / STAMP
            try:
                if any(r.get("inspection_id") == iid for r in _jsonl(f)):
                    continue                # marked before (an ABORT after the verdict)
                line = {"time": E.local_time(datetime.now().astimezone().isoformat()),
                        "inspection_id": iid, "disposition": disposition,
                        "result_file": str(Path(run_dir) / E.RESULT_FILE), "reason": why,
                        "scan_id": judged}
                with open(f, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(line) + "\n")
                marked.append(d)
            except OSError as e:
                warn.append(f"{WARNING}: scan {d.name} could not be marked as a bad part "
                            f"({f}: {_why(e)}): never make a reference from it")
    return marked, warn


def _jsonl(path: Path) -> list:
    """The JSON objects of a .jsonl file ([] when there is none; a bad line is skipped)."""
    out = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if isinstance(rec, dict):
                    out.append(rec)
    except OSError:
        pass
    return out


# ------------------------------------------------------------------ the copy

def folder_name(part_id: str, inspection_id: str) -> str:
    """The name of a bad part's folder: its part ID made safe as a folder name (letters,
    digits, _ . -; anything else becomes _; at most MAX_NAME characters; never a name
    Windows reserves, nor one of the output's own (pending, the index, the marker, the
    locks): those get a leading _), else the inspection id."""
    s = _UNSAFE.sub("_", str(part_id or "")).strip("._- ")[:MAX_NAME].rstrip("._- ")
    if not s:
        return str(inspection_id)
    return "_" + s if _RESERVED.match(s) or s.lower() in _OWN_NAMES else s


def _open(path: Path, mode: str):
    """Every file of the bad-part output is opened here (whatever the event log's switch
    says); a Windows path longer than it allows is refused with that reason."""
    _fits(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return open(path, mode)


def _fits(path: Path) -> None:
    if os.name == "nt" and len(str(path)) > MAX_PATH:
        raise OSError(f"the path {path} is {len(str(path))} characters long: Windows allows "
                      f"{MAX_PATH + 1} (without long paths switched on) - choose a shorter "
                      "bad-part output folder")


def _ready(root: Path) -> None:
    """The bad-part output folder and its marker (written first: nothing in the folder is
    ever a reference). OSError when the folder cannot be reached or written."""
    root.mkdir(parents=True, exist_ok=True)
    try:
        with _open(root / MARKER, "x") as f:
            f.write(MARKER_TEXT)
    except FileExistsError:
        pass


def _out_of_limit(values: list) -> str:
    """The `out_of_limit` cell: where: name=value unit [lower..upper]; ... (the first few)."""
    bits = []
    outs = [v for v in values if v.get("result") == "OUT"]
    for v in outs[:MAX_OUT]:
        where = "/".join(x for x in (v["step"], v["zone"], v["check"]) if x)
        text = (f"{where}: " if where else "") + f"{v['name']}={E._g(v['value'])}" + (
            f" {v['unit']}" if v["unit"] else "")
        bits.append(text + f" [{E._g(v['lower_limit'])}..{E._g(v['upper_limit'])}]")
    more = len(outs) - len(bits)
    return "; ".join(bits) + (f"; (+{more} more)" if more > 0 else "")


def _real(p) -> Path:
    """A path with its links resolved; as given (made absolute) when that cannot be done."""
    try:
        return Path(p).resolve()
    except (OSError, RuntimeError, ValueError):
        return Path(os.path.abspath(str(p)))


def _copy_file(src, dst: Path) -> None:
    """One file into a part's folder, whole or not at all: copied under a temporary name,
    then renamed. A file there already whose size is its source's is a whole copy (a retry
    leaves it); any other is made again."""
    try:
        if dst.stat().st_size == os.stat(src).st_size:
            return
    except OSError:
        pass
    tmp = dst.with_name(dst.name + PART_SUFFIX)
    try:
        with open(src, "rb") as s, _open(tmp, "wb") as d:
            shutil.copyfileobj(s, d)
        os.replace(tmp, dst)
    except OSError:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def _copy(it: dict, root: Path, folder: Optional[Path] = None) -> dict:
    """One inspection (eventlog.inspection) into the bad-part output `root`: the marker, the
    part's folder (a new one; `folder`: the one a failed attempt made, filled up), its files
    (each whole: _copy_file), its index row, and bad_part_output.json in the run folder.
    -> {folder, index: "written" | "waiting", why, notes: [warnings for the run's output]}.
    ValueError when root may not be a bad-part output (_clash); OSError when the part's
    files could not be written (its `bad_part_folder`: the folder made, None: none - the
    caller queues it with that folder: nothing is half reported)."""
    run_dir, iid = Path(it["run_dir"]), it["inspection_id"]
    clash = _clash(it, root)
    if clash is not None:
        raise ValueError(f"{root}: {clash}")
    done = _jsonl_one(run_dir / DONE_FILE)
    if done and _key(done.get(KEY)) == _key(root) and Path(str(done.get("folder"))).is_dir():
        return {"folder": Path(done["folder"]), "index": done.get("index") or "written",
                "why": "copied before", "notes": []}
    _ready(root)
    if folder is None or not Path(folder).is_dir():
        folder = None
        name = folder_name(" | ".join(it["codes"]), iid)
        for n in range(1, 1000):
            cand = root / (name if n == 1 else f"{name}-{n}")
            _fits(cand / ("profile_x.csv" + PART_SUFFIX))
            try:
                cand.mkdir()                # never an existing folder: a repeat gets -2, -3
                folder = cand
                break
            except FileExistsError:
                continue
        if folder is None:
            raise OSError(f"999 folders named {name} in {root} already")
    folder = Path(folder)
    pic = it["picture"]
    files = [(it["result"], E.RESULT_FILE),     # short names of their own: the folder may be
             (pic, pic.name if pic is not None and pic.name in E.PICTURES else PICTURE_NAME),
             *[(p, p.name) for p in it["profiles"]], (it["ply"], PLY_NAME)]    # deep
    try:
        for src, name in files:
            if src is not None and Path(src).is_file():
                _copy_file(src, folder / name)
    except OSError as e:
        e.bad_part_folder = folder          # the retry fills this folder up
        raise
    row = {"time": it["time"], "part_id": " | ".join(it["codes"]), "inspection_id": iid,
           "program": it["program"], "operating_mode": it["operating_mode"],
           "hardware": it["hardware"], "disposition": it["disposition"], "reason": it["reason"],
           "out_of_limit": _out_of_limit(it["values"]),
           "folder": folder.relative_to(root).as_posix()}
    asides = []
    index, why = _index(root, row, asides)
    try:
        with open(run_dir / DONE_FILE, "x", encoding="utf-8") as f:
            json.dump({KEY: str(root), "folder": str(folder), "inspection_id": iid,
                       "disposition": it["disposition"], "index": index,
                       "time": E.local_time(datetime.now().astimezone().isoformat())}, f,
                      indent=2)
            f.write("\n")
    except OSError as e:                    # copied all the same: only a second copy is
        log.warning("%s: %s not written: %s", WARNING, run_dir / DONE_FILE, e)   # possible
    return {"folder": folder, "index": index, "why": why, "notes": _aside_notes(asides)}


def _aside_notes(asides: list) -> list:
    """An index changed in another program and moved aside -> warnings for the run's output."""
    return [f"{WARNING}: {path} {what}: it was moved to {aside} as it was, and its rows go on "
            f"in a new {path.name} (open a copy of the index in another program, never the "
            "index itself)" for path, aside, what in asides]


def _jsonl_one(path: Path) -> Optional[dict]:
    doc = E._read_json(path)
    return doc if isinstance(doc, dict) else None


def _index(root: Path, row: dict, asides: Optional[list] = None) -> tuple:
    """The part's row onto the index, under its lock, after the rows that wait; when the
    index cannot take it now (open in a spreadsheet program, or no lock in time) the row
    waits in <root>/pending/. asides: gets an index that was moved aside (its header was
    changed in another program: eventlog._settle_header). -> ("written" | "waiting", why).
    OSError when it cannot even wait."""
    body = E._encode(INDEX_FIELDS, [row])
    why = f"no lock on {root / LOCK_FILE} in {E.LOCK_WAIT_S:g} s"
    with E._locked(root, E.LOCK_WAIT_S, LOCK_FILE, opener=_open) as held:
        if held:
            _merge_waiting(root, asides)
            try:
                E._append(root / INDEX, INDEX_FIELDS, body, opener=_open, asides=asides)
                return "written", None
            except OSError as e:
                why = f"{INDEX}: {_why(e)}"
    with E._locked(root, E.LOCK_WAIT_S, PENDING_LOCK_FILE, opener=_open) as held:
        if not held:
            raise OSError(f"the index row could not be written ({why}) and no lock on "
                          f"{root / PENDING_LOCK_FILE}")
        E._append(root / PENDING_DIR / INDEX, INDEX_FIELDS, body, opener=_open, asides=asides)
    return "waiting", why


def _merge_waiting(root: Path, asides: Optional[list] = None) -> bool:
    """The index rows that wait (<root>/pending/) onto the index; the caller holds the
    index's lock. The waiting file is taken first (renamed, under its own lock), so no row is
    ever added twice; put back when the index cannot take them. -> merged."""
    waiting = root / PENDING_DIR / INDEX
    if not waiting.is_file():
        return False
    taken = waiting.with_suffix(".merging")
    try:
        with _open(root / INDEX, "ab"):     # still open elsewhere? then they keep waiting
            pass
        with E._locked(root, E.LOCK_WAIT_S, PENDING_LOCK_FILE, opener=_open) as held:
            if not held:
                return False
            os.replace(waiting, taken)
    except OSError:
        return False
    try:
        rows = E._rows_by_header(taken.read_bytes(), INDEX_FIELDS)
        if rows:
            E._append(root / INDEX, INDEX_FIELDS, E._encode(INDEX_FIELDS, rows), opener=_open,
                      asides=asides)
    except OSError:
        try:
            os.replace(taken, waiting)      # back: they still wait
        except OSError:
            pass
        return False
    try:
        taken.unlink()
    except OSError:
        pass
    return True


# ------------------------------------------------------------------ copies that wait

def _queue(it: dict, root: Path, why: str, folder=None) -> Optional[str]:
    """A copy that failed waits: <library>/bad_part_pending/<inspection id>.json, naming the
    part's folder when the attempt made one (it is filled up, never made twice). -> its
    path, or None when it cannot be written either. The bad-part output is not touched."""
    note = Path(it["library"]) / QUEUE_DIR / f"{it['inspection_id']}.json"
    try:
        note.parent.mkdir(parents=True, exist_ok=True)
        with open(note, "x", encoding="utf-8") as f:
            json.dump({"run_dir": str(it["run_dir"]), KEY: str(root),
                       "folder": str(folder) if folder is not None else None,
                       "inspection_id": it["inspection_id"], "disposition": it["disposition"],
                       "why": why,
                       "time": E.local_time(datetime.now().astimezone().isoformat()),
                       "queued_ns": time.time_ns()}, f,     # the order they are copied in
                      indent=2)
            f.write("\n")
    except FileExistsError:
        pass                                # it waits already
    except OSError as e:
        log.warning("%s: %s could not be queued: %s", WARNING, it["inspection_id"], e)
        return None
    return str(note)


def _notes(library, taken_too: bool = False) -> list:
    """The notes of the copies that wait, oldest first: [(note file, its content | None)]."""
    q = Path(library) / QUEUE_DIR
    try:
        files = list(q.glob("*.json")) + (list(q.glob("*.retrying")) if taken_too else [])
    except OSError:
        return []
    notes = [(p, E._read_json(p)) for p in files]

    def order(n):
        d = n[1] if isinstance(n[1], dict) else {}
        return (d.get("queued_ns") if isinstance(d.get("queued_ns"), int) else 0, n[0].name)
    return sorted(notes, key=order)


def waiting(library) -> list:
    """The inspections whose copy waits, oldest first: [the notes' contents]."""
    return [d for _, d in _notes(library) if isinstance(d, dict)]


def retry(library, taken_too: bool = False, root=None, limit: Optional[int] = None,
          budget_s: Optional[float] = None) -> dict:
    """The copies that wait (<library>/bad_part_pending/), oldest first - each taken first
    (renamed: two programs never copy one twice), copied, its note removed; put back when it
    fails again. Each folder is tried once (_ready): when it cannot be reached, its copies
    keep waiting untried. root: only the copies to that folder (the hook). limit /
    budget_s: no more copies than that, started no later than that (the hook; `--retry`:
    all). taken_too: also notes another program took and never gave back (`--retry` by
    hand). A copy that may not go where its note says (_clash) is dropped, said in the log.
    Never raises. -> {"done": n, "waiting": m}"""
    out = {"done": 0, "waiting": 0}
    end = time.monotonic() + budget_s if budget_s is not None else None
    ready, tried = {}, 0
    for note, data in _notes(library, taken_too):
        dest = data.get(KEY) if isinstance(data, dict) else None
        if root is not None and (not dest or _key(dest) != _key(root)):
            continue                        # another folder's: not this hook's
        if (limit is not None and tried >= limit) or (end is not None and time.monotonic() > end):
            out["waiting"] += 1
            continue
        if dest and _key(dest) not in ready:
            try:
                _ready(Path(dest))
                ready[_key(dest)] = True
            except OSError as e:
                ready[_key(dest)] = False
                log.info("%s: %s cannot be reached: %s", WARNING, dest, _why(e))
        if dest and not ready[_key(dest)]:
            out["waiting"] += 1             # the folder does not answer: not tried again now
            continue
        tried += 1
        taken = note.with_suffix(".retrying")
        try:
            if note != taken:
                os.replace(note, taken)     # ours now
            it = E.inspection(data["run_dir"])
            if it is None or it["disposition"] not in BAD:
                taken.unlink()              # nothing to copy (any more)
                continue
            try:
                _copy(it, Path(dest), Path(data["folder"]) if data.get("folder") else None)
            except ValueError as e:         # it may not go there: dropped, said
                log.warning("%s: %s not copied: %s", WARNING, note.stem, e)
                taken.unlink()
                continue
            taken.unlink()
            out["done"] += 1
        except Exception as e:              # noqa: BLE001 - it keeps waiting
            log.info("%s: %s still waits: %s", WARNING, note.stem, e)
            made = getattr(e, "bad_part_folder", None)
            try:
                if made is not None and not data.get("folder"):     # filled up next time
                    with open(taken, "w", encoding="utf-8") as f:
                        json.dump({**data, "folder": str(made)}, f, indent=2)
                        f.write("\n")
                os.replace(taken, note.with_suffix(".json"))
            except OSError:
                pass
            out["waiting"] += 1
    return out


def copy_run(run_dir, root=None) -> dict:
    """One inspection's copy, by hand (`--copy`): to `root`, else the folder its result.json
    names. -> as _copy; ValueError when there is nothing to copy, or the folder may not be a
    bad-part output."""
    it = E.inspection(run_dir)
    if it is None:
        raise ValueError(f"{run_dir}: no inspection (no result.json with a run id, or the "
                         "check of a work state inside a cycle: copy the cycle)")
    if it["disposition"] not in BAD:
        raise ValueError(f"inspection {it['inspection_id']} is {it['judgment']}: only a part "
                         "judged REJECT or PULL goes to the bad-part output")
    outs = it["doc"].get("outputs") if isinstance(it["doc"].get("outputs"), dict) else {}
    dest = configured(str(root)) if root is not None else outs.get(KEY)
    if not dest or dest == OFF:
        raise ValueError("no bad-part output: give --to DIR (the run's result.json names none)")
    return _copy(it, Path(dest))


# ------------------------------------------------------------------ references: good parts only

class BadPartReference(rules.RuleViolation):
    """A reference was to be made from a part judged REJECT or PULL, or a part to be checked
    against a reference made from one. Refused: there is no override."""


def output_of(path) -> Optional[Path]:
    """The bad-part output a file or folder lives in (a folder above it, or itself, holds
    BAD_PART_OUTPUT.txt), else None."""
    p = _real(path)
    for d in [p, *p.parents]:
        try:
            if (d / MARKER).is_file():
                return d
        except OSError:
            continue
    return None


def _judgment_of(doc) -> Optional[str]:
    """A recorded result of a check -> CONTINUE / REJECT / PULL, or None (it judged no part):
    its disposition (face check, station check: any word but CONTINUE is REJECT / PULL; a
    measurement's NONE judges nothing), else - a check verb that gives no disposition
    (inspect, compare, bank check, dent detection) - a definite bad verdict (FAIL, FLAG,
    REJECT) is a REJECT; a pass, or no verdict it could give (INDETERMINATE: uncalibrated, no
    band, an empty bank), judged nothing."""
    if not isinstance(doc, dict):
        return None
    d = doc.get("disposition")
    if isinstance(d, str) and d:
        return None if d.upper() == "NONE" else ("CONTINUE" if d.upper() == "CONTINUE"
                                                 else _bad(d.upper()))
    for k in ("verdict", "result"):
        v = doc.get(k)
        if isinstance(v, str) and v:
            return "REJECT" if v.upper() in REJECTS else None
    return None


def _runs_above(path: Path) -> list:
    """The run folders a file or folder lies in (a work state's check and its cycle):
    [(run folder, the judgment it reads: REJECT / PULL / CONTINUE / None)]."""
    out = []
    p = _real(path)
    for d in [p, *p.parents][:6]:
        if (d / E.RESULT_FILE).is_file() and (d / E.LOG_FILE).is_file():
            doc = E._read_json(d / E.RESULT_FILE)
            if not isinstance(doc, dict):
                out.append((d, "PULL"))     # a run whose result cannot be read: a doubt
                continue
            it = E.inspection(d)            # a run or a cycle, as the event log reads it
            # (a measurement: None); else the check of a work state inside a cycle: its own
            out.append((d, it["disposition"] if it is not None else _bad(doc.get("disposition"))))
    return out


def _entry_findings(library: Path, d: Path, label: str) -> list:
    """What says the library entry `d` (or any folder / file) is a bad part."""
    out = []
    root = output_of(d)
    if root is not None:
        whole = _within(library, root) and not _within(root, library)
        out.append({"item": label, "disposition": None,
                    "why": f"lives in a bad-part output ({root})" + (
                        f", and so does the whole library {library} (every entry of it is "
                        f"refused): set the bad-part output to a folder outside the library "
                        f"and its runs folders, and move {MARKER} and the part folders of "
                        f"{root} there" if whole else "")})
    for run, disp in _runs_above(d):
        if disp in BAD:
            out.append({"item": label, "disposition": disp,
                        "why": f"lies in the run folder of inspection {run.name}, judged {disp}"})
    if not Path(d).is_dir():
        return out
    for rec in _jsonl(Path(d) / STAMP):
        disp = _bad(rec.get("disposition"))
        if disp is not None:
            via = rec.get("scan_id")
            out.append({"item": label, "disposition": disp,
                        "why": f"was judged {disp} by inspection {rec.get('inspection_id')}"
                        + (f" (as scan {via}, made from it)" if isinstance(via, str)
                           and via != Path(d).name else "")
                        + (f": {rec['reason']}" if rec.get("reason") else "")})
    for rec in _jsonl(Path(d) / "exports.jsonl"):
        f = rec.get("file")
        if rec.get("kind") not in JUDGING_KINDS or not isinstance(f, str) \
                or not f.lower().endswith(".json"):
            continue
        p = Path(f) if Path(f).is_absolute() else library / f
        if not p.is_file():
            continue                        # the record is gone: nothing says so any more
        doc = E._read_json(p)
        disp = _judgment_of(doc) if doc is not None else "PULL"     # unreadable: a doubt
        if disp in BAD:
            out.append({"item": label, "disposition": disp,
                        "why": f"was judged {disp} by {rec.get('kind')} {p.name}"})
    return out


def _resolve(library: Path, item) -> tuple:
    """A source as a verb holds it -> (its label, [folders / files to look at], [scan ids]):
    a handle with a path (a scan, a face scan, a dent set: its entry), a path that exists
    (a file of the output bin, a folder), else a scan id (its entries in the library)."""
    path = getattr(item, "path", None)
    if path is not None and not isinstance(item, (str, Path)):
        sid = getattr(item, "id", None)
        return str(sid or path), [Path(path)], [sid] if isinstance(sid, str) else []
    text = str(item)
    p = Path(text)
    try:
        if p.is_dir():                      # an entry of the library is named by its id
            return p.name if _inside(library, p) else text, [p], [p.name]
        if p.exists():
            return text, [p], []
    except OSError:
        pass
    return text, scan_entries(library, text), [text]


def _inside(root: Path, p: Path) -> bool:
    try:
        _real(p).relative_to(_real(root))
        return True
    except ValueError:
        return False


def findings(store, items, sweep=False) -> list:
    """What says any of `items` is a bad part (the module docstring's list) -> [{item,
    disposition (REJECT / PULL, None: a bad-part output), why}], each once. An item made from
    another scan (source_scan) has that scan's findings too. sweep: also the run folders
    (runs made before the marks existed) - True: all of them (a reference is MADE); "once":
    those not searched for that scan before (a reference is USED); False: none."""
    library = Path(getattr(store, "root", store))
    out, ids = [], {}
    for item in items:
        if item is None:
            continue
        label, places, scan_ids = _resolve(library, item)
        for d in places:
            out += _entry_findings(library, d, label)
        for sid in scan_ids:
            ids.setdefault(sid, (label, ""))
        for src, dirs in _lineage(library, places):
            pre = f"is made from scan {src}, which "
            for d in dirs:
                out += [{**f, "why": pre + f["why"]} for f in _entry_findings(library, d, label)]
            ids.setdefault(src, (label, pre))
    if sweep and ids:
        out += _sweep(library, ids, again=sweep is True)
    seen, uniq = set(), []
    for f in out:
        if (f["item"], f["why"]) not in seen:
            seen.add((f["item"], f["why"]))
            uniq.append(f)
    return uniq


def _swept(entry: Path) -> set:
    """The runs folders searched for inspections of a scan before (its entry's SWEPT)."""
    doc = E._read_json(Path(entry) / SWEPT)
    dirs = doc.get("runs_dirs") if isinstance(doc, dict) else None
    return {str(d) for d in dirs} if isinstance(dirs, list) else set()


def _sweep(library: Path, ids: dict, again: bool) -> list:
    """The run folders searched for inspections that judged one of `ids` ({scan id: (label,
    how the finding begins)}) REJECT / PULL: every runs folder the event log's rebuild reads
    (again), else only those not searched for that scan before (its entries' SWEPT). What is
    found is marked on the scans (_run_findings); a scan nothing was found for gets the
    search noted in its entries, so it is not searched again (later inspections mark it
    themselves). A scan with no entry here is searched every time."""
    dirs = {_key(d): d for d in E.runs_dirs_of(library)}
    plan = {}
    for sid, said in ids.items():
        entries = scan_entries(library, sid)
        done = set() if again or not entries else set.intersection(*map(_swept, entries))
        left = [k for k in dirs if k not in done]
        if left:
            plan[sid] = (said, entries, left)
    by_dir = {}
    for sid, (_, _, left) in plan.items():
        for k in left:
            by_dir.setdefault(k, {})[sid] = plan[sid][0]
    found = []
    for k, want in by_dir.items():
        found += _run_findings(library, want, [dirs[k]])
    bad = {f.pop("scan") for f in found}
    for sid, (_, entries, _) in plan.items():
        if sid in bad:
            continue                        # marked (or searched again next time)
        for e in entries:
            try:
                with open(Path(e) / SWEPT, "w", encoding="utf-8") as f:
                    json.dump({"time": E.local_time(datetime.now().astimezone().isoformat()),
                               "runs_dirs": sorted(set(dirs) | _swept(e)),
                               "note": "the runs folders searched for inspections that judged "
                                       "this scan REJECT / PULL (engine/badparts.py)"}, f,
                              indent=2)
                    f.write("\n")
            except OSError:
                pass                        # searched again next time
    return found


def _run_findings(library: Path, ids: dict, runs_dirs: list) -> list:
    """Every run folder in `runs_dirs` whose result judged one of the scan ids ({scan id:
    (label, how the finding begins)}) REJECT / PULL (a cycle: or one of its work states'
    checks). The scans it judged are marked (stamp), so the next use finds the mark.
    -> findings, each with its `scan`."""
    out = []
    keys = [i.encode("utf-8") for i in ids]
    for d in E.run_folders(runs_dirs):
        try:
            raw = (d / E.RESULT_FILE).read_bytes()
        except OSError:
            continue                        # a cycle that never ended: its checks say
        if not any(k in raw for k in keys):
            continue
        try:
            doc = json.loads(raw.decode("utf-8"))
        except ValueError:
            continue
        if not isinstance(doc, dict):
            continue
        it = E.inspection(d)
        disp = it["disposition"] if it is not None else None
        hit = [sid for sid in judged_scans(doc, d) if sid in ids]
        if hit and disp in BAD:
            stamp(library, doc, d, disp, it["reason"])
            for sid in hit:
                label, pre = ids[sid]
                out.append({"item": label, "disposition": disp, "scan": sid,
                            "why": f"{pre}was judged {disp} by inspection {d.name}"})
        job = doc.get("job") if isinstance(doc.get("job"), dict) else {}
        for sid_step, sd, sub in E._steps(d, doc) if job.get("type") == "flow" else []:
            sdisp = _bad(sub.get("disposition"))
            sid = (sub.get("part") or {}).get("scan_id") if isinstance(sub.get("part"),
                                                                       dict) else None
            if sid in ids and sdisp is not None:
                stamp(library, sub, sd, sdisp)
                label, pre = ids[sid]
                out.append({"item": label, "disposition": sdisp, "scan": sid,
                            "why": f"{pre}was judged {sdisp} by work state {sid_step} of cycle "
                                   f"{d.name}"})
    return out


def require_good(store, what: str, items, sweep="once", how: str = HOW_MAKE) -> None:
    """Refuse (BadPartReference) when any of `items` is a bad part (findings): a reference
    is made from good parts only, and nothing is compared with a bad part. what: the verb and
    the role of the items ("face_golden placement", "recipe golden"). sweep: True when a
    reference is MADE; "once" (default) when one is USED. how: the way out the message
    gives."""
    found = findings(store, [i for i in items if i is not None], sweep)
    if not found:
        return
    f = found[0]
    more = f" (and {len(found) - 1} more)" if len(found) > 1 else ""
    raise BadPartReference(f"{what}: {f['item']} {f['why']}{more}. The rule: {RULE}. "
                           f"{how[:1].upper()}{how[1:]}.")


# ------------------------------------------------------------------ command line

def _default_library() -> Path:
    from . import api                       # the engine's default ($SWISS_SCANNER_LIBRARY)
    return Path(api.DEFAULT_ROOT)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m engine.badparts",
        description="The bad-part output: a folder per part judged REJECT or PULL and "
                    f"{INDEX}. Touches no rig.")
    do = ap.add_mutually_exclusive_group(required=True)
    do.add_argument("--retry", action="store_true",
                    help="copy the inspections whose copy waits (<library>/bad_part_pending/)")
    do.add_argument("--copy", metavar="RUN_FOLDER",
                    help="copy one inspection (judged REJECT or PULL) by hand")
    ap.add_argument("--library", metavar="DIR",
                    help="the library (--retry; default: the engine's, or $SWISS_SCANNER_LIBRARY)")
    ap.add_argument("--to", metavar="DIR",
                    help="--copy: the bad-part output (default: the one its result.json names)")
    a = ap.parse_args(argv)
    if a.copy:
        try:
            got = copy_run(a.copy, a.to)
        except (ValueError, OSError) as e:
            print(f"not copied: {e}", file=sys.stderr)
            return 1 if isinstance(e, OSError) else 2
        for n in got["notes"]:
            print(n, file=sys.stderr)
        print(f"copied to {got['folder']}"
              + (f"; its index row waits ({got['why']})" if got["index"] == "waiting" else ""))
        return 0
    library = Path(a.library) if a.library else _default_library()
    if not library.is_dir():
        print(f"no library folder {library}", file=sys.stderr)
        return 2
    r = retry(library, taken_too=True)
    print(f"{r['done']} copied; {r['waiting']} still waiting")
    return 1 if r["waiting"] else 0


if __name__ == "__main__":
    sys.exit(main())

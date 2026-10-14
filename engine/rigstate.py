"""
Rig state: is the rig where its calibration was made? What a mm result may rest on.

The TRIANGULATION PRESET (rig geometry, exposure, fringes, its stereo calibration) is its own
layer, and it says how the rig is mounted (preset `rig.mount`, engine/config.MOUNTS):

    fixed    bolted to its station. The stored calibration is TRUSTED until a rig check says
             MOVED / UNKNOWN. Such an objection stands until it is ANSWERED (below).
    mobile   carried / re-positioned between uses. It is ASSUMED MOVED since the last use: no
             mm until it was VERIFIED in this session.

A geometry_id names ONE rig: when any preset at a geometry says mobile, every preset at that
geometry is gated as mobile (mount_of(): a fixed copy of a mobile preset is not a way round).

What ANSWERS (fixed: clears an objection; mobile: verifies the rig for this session), always
for the stereo calibration in use:
    - a calibration solved from poses that were ALL captured after the objection (mobile: in
      this session). Solving stored poses again writes a new entry and answers nothing.
    - a rig check OK at the default tolerance or tighter (a looser one proves nothing),
      against a reference that belongs to that calibration and was TRUSTED when it was
      stored: mobile - the rig was verified at that moment; fixed - no objection stood, or
      the operator asked for it. A reference stored while the rig was in doubt verifies
      nothing, in that session or any later one.
    - a re-home fitted against such a reference (the re-homed calibration it wrote).
    - fixed only: a reference stored on the operator's request (`scan rig-check
      --set-reference`: "the rig is known good").

A mobile session starts at the newest SESSION MARKER of the geometry (new_session(): the
operator set the rig down / powered it up / moved it - `scan rig-session --new`, the web UI
server's start-up) and is never older than MOBILE_SESSION_MAX_AGE_S, than the start of this
computer (boot_time(), where the operating system reports it) or than the start of the
long-running process that owns the rig (open_process_session(): the web UI server). A
verification holds for one session. The rig alone being switched off or carried can't be
seen: the operator says so with `scan rig-session --new`.

The record (append-only, one JSON object per line, never rewritten):

    <library>/calibrations/<geometry_id>/rig_state.jsonl
        {"time", "event": "session", "reason", "by", "preset", "mount"}
        {"time", "event": "calib_pose", "pose_id", "set", "preset"}
        {"time", "event": "calibration", "calibration_id", "poses": [pose ids], "set"}
        {"time", "event": "rig_check", "status": OK | MOVED | UNKNOWN | REFERENCE_SET |
         REHOMED, "tol_px", "moved", "shift_px", "camera_motion_px", "projector_motion_px",
         "calibration_id", "reference_id", "reference_calibration_id", "reference_trust":
         {"mobile", "fixed"}, "requested", "rehome_from", "previous_reference_id", "message",
         "preset", "mount"}

The ORDER of the lines is the order of events: what answers what is decided by the record's
order, never by comparing clock times (a clock set back by daylight saving or a time sync
can't put an old OK after a newer MOVED or session marker). Only the session's maximum age
and a stored scan's own time are clock times. A line that cannot be read counts as a rig
check that said UNKNOWN (fail closed: a torn MOVED line never reads as "no objection").
Only the newest lines are read: back to the session marker (mobile) or to the newest
objection / answer (fixed), and never more than MAX_TAIL of them - a record that grew by a
line per cycle for a year costs what a short one does. Fail closed there too: when those
lines neither verify a mobile rig nor confirm a fixed one, the answer is no (run a rig
check: its line is then the newest).

gate() is the one answer the operating modes ask (engine/modes/): Measure -> NOT_MEASURABLE,
Quick check / Station -> PULL when it says no. Station mode on a FIXED rig is not gated here
(its behaviour is unchanged: the command bus goes to FAULT on a MOVED / UNKNOWN rig check);
Measure and Quick check (strict=True) also refuse a fixed rig with an unanswered objection.
require() is the same gate for the engine's own capture verbs (scan_metric, scan_fast, ...):
a mobile rig that is not verified gives no mm cloud at all. A FIXED rig is not refused there
(the design: trusted until a check objects, the mm still given), but never silently: while
an objection stands - what Measure and Quick check would refuse, doubt() - the mm go out
with a warning in the rig check's own words and the way out (IN_DOUBT; printed, in the
entry's meta "warnings" and its "rig" stamp, stamp(): trusted false).

Pure library I/O: no hardware, no engine session.
"""

from __future__ import annotations

import json
import re
import sys
import time as _time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterator, Optional

from . import config, rules

LOG_FILE = "rig_state.jsonl"
MOBILE_SESSION_MAX_AGE_S = 12 * 3600.0      # a verification holds for one session: at most this
EVENTS = ("session", "calib_pose", "calibration", "rig_check")
GOOD = ("OK", "REFERENCE_SET", "REHOMED")   # every other status of a rig check objects
BAD = ("MOVED", "UNKNOWN")                  # ... as these two do (the ones a check reports)
CLOCK_SLACK_S = 2.0                         # a stored scan stamped later than now + this: refused
MAX_TAIL = 5000                             # at most this many of the newest lines are read
MESSAGE_CHARS = 500                         # a rig check's message is kept to this many chars
                                            # (its headline: which device moved, what to do)
UNVERIFIED = "mobile rig not verified: run rig-check / re-home / calibrate"
HOW_MOBILE = ("verify the rig for this session: `scan rig-check` (OK against the reference of "
              "the calibration in use, one stored while the rig was verified), `scan rig-check "
              "--rehome` after a small camera move, or calibrate again from poses captured now "
              "(`scan calib-pose` per pose, then `scan calibrate-stereo`; 3 placements with the "
              "calibration fixture)")
HOW_MOVED = ("put the rig right, then `scan rig-check` (OK against the reference stored before "
             "it moved), `scan rig-check --rehome` after a small camera move, or calibrate "
             "again from poses captured now (`scan calib-pose`, `scan calibrate-stereo`); with "
             "the rig known good `scan rig-check --set-reference` stores a new reference")
IN_DOUBT = "fixed rig not confirmed: these mm may be off (Measure and Quick check give none)"
_TS = re.compile(r"^(\d{8}T\d{6})_(\d{6})")  # library/store.timestamp(): an entry's own time
_BLOCK = 1 << 16                            # the record is read backwards in blocks this big


def log_path(store, geometry_id: str) -> Path:
    if not config.valid_id(geometry_id):
        raise ValueError(f"geometry_id must be one id [A-Za-z0-9_.-], got {geometry_id!r}")
    return Path(store.root) / "calibrations" / geometry_id / LOG_FILE


def _now() -> datetime:
    return datetime.now()


def _iso(t: datetime) -> str:
    return t.isoformat(timespec="microseconds")


def _parse(text) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(str(text))
    except (TypeError, ValueError):
        return None


def entry_time(path_or_name) -> Optional[datetime]:
    """When a library entry was made, from its folder name (<timestamp>_<label>, to the
    microsecond); None when the name is not a library timestamp."""
    if path_or_name is None:
        return None
    m = _TS.match(Path(str(path_or_name)).name)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1) + m.group(2), "%Y%m%dT%H%M%S%f")
    except ValueError:
        return None


# ------------------------------------------------------------------ the record

def append(store, geometry_id: str, record: dict, now: Optional[datetime] = None) -> dict:
    """One more line on the geometry's rig-state record (append-only). -> the record."""
    f = log_path(store, geometry_id)
    f.parent.mkdir(parents=True, exist_ok=True)
    rec = {"time": _iso(now or _now()), **record}
    lead = ""
    try:                                    # a torn last line stays a line of its own
        with open(f, "rb") as fh:
            fh.seek(0, 2)
            if fh.tell():
                fh.seek(-1, 2)
                lead = "" if fh.read(1) == b"\n" else "\n"
    except FileNotFoundError:
        pass
    with open(f, "a", encoding="utf-8") as fh:
        fh.write(lead + json.dumps(rec, default=str) + "\n")
    return rec


def _lines_back(path: Path) -> Iterator[bytes]:
    """The record's lines, newest first, read in blocks from the end of the file (a long
    record costs what is read of it, not its length)."""
    try:
        fh = open(path, "rb")
    except (FileNotFoundError, NotADirectoryError):
        return
    with fh:
        fh.seek(0, 2)
        pos, tail = fh.tell(), b""
        while pos > 0:
            n = min(_BLOCK, pos)
            pos -= n
            fh.seek(pos)
            parts = (fh.read(n) + tail).split(b"\n")
            tail = parts[0]                 # may go on in the block before
            for raw in reversed(parts[1:]):
                if raw.strip():
                    yield raw
        if tail.strip():
            yield tail


def _record(raw: bytes) -> Optional[dict]:
    try:
        d = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    t = _parse(d.get("time")) if isinstance(d, dict) else None
    if t is None or d.get("event") not in EVENTS:
        return None
    return {**d, "time": t}


def read_back(store, geometry_id: str, now: Optional[datetime] = None) -> Iterator[dict]:
    """The record, NEWEST first: {"time" (datetime), "event", ...}. A line that is not a
    readable record becomes {"event": "rig_check", "status": "UNKNOWN", "unreadable": True}
    timed like the next readable line (or now, when it is the last): it objects until a
    newer record answers it. A generator: stop reading when enough is known."""
    nxt = now or _now()
    for raw in _lines_back(log_path(store, geometry_id)):
        rec = _record(raw)
        if rec is None:
            rec = {"time": nxt, "event": "rig_check", "status": "UNKNOWN", "unreadable": True,
                   "message": "an unreadable line in the rig-state record"}
        else:
            nxt = rec["time"]
        yield rec


def read(store, geometry_id: str, now: Optional[datetime] = None) -> list:
    """The whole record, oldest first (read_back() for the newest lines only)."""
    out = list(read_back(store, geometry_id, now))
    out.reverse()
    return out


def new_session(store, preset, reason: str = "", by: str = "",
                now: Optional[datetime] = None) -> dict:
    """A session marker: the rig was set down / powered up / moved NOW, so whatever verified
    it before no longer does (a mobile rig; on a fixed rig the marker is recorded and changes
    nothing). -> the record."""
    return append(store, preset.geometry_id,
                  {"event": "session", "reason": " ".join(str(reason or "").split())[:200],
                   "by": str(by or "")[:80], "preset": preset.name,
                   "mount": preset.rig.mount}, now)


def record_pose(store, preset, pose_id: str, set_label: str = "",
                now: Optional[datetime] = None) -> dict:
    """A calibration pose was captured NOW (engine/calibrate/flow.capture_pose calls it): a
    calibration answers an objection / verifies a mobile rig only when every pose it was
    solved from is on the record after that objection / in the session."""
    return append(store, preset.geometry_id,
                  {"event": "calib_pose", "pose_id": str(pose_id), "set": str(set_label or ""),
                   "preset": preset.name}, now)


def record_calibration(store, preset, calibration_id: str, pose_ids, set_label: str = "",
                       now: Optional[datetime] = None) -> dict:
    """A stereo calibration was solved NOW from these poses (flow.calibrate_stereo)."""
    return append(store, preset.geometry_id,
                  {"event": "calibration", "calibration_id": calibration_id,
                   "poses": [str(p) for p in pose_ids or []], "set": str(set_label or ""),
                   "preset": preset.name}, now)


def record_check(store, preset, result: dict, now: Optional[datetime] = None) -> dict:
    """A rig check's result on the record (engine/calibrate/rigref.rig_check calls it)."""
    return append(store, preset.geometry_id, {
        "event": "rig_check", "status": result.get("status"), "moved": result.get("moved"),
        "shift_px": result.get("shift_px"), "camera_motion_px": result.get("camera_motion_px"),
        "projector_motion_px": result.get("projector_motion_px"), "tol_px": result.get("tol_px"),
        "calibration_id": result.get("calibration_id"),
        "reference_id": result.get("reference_id"),
        "reference_calibration_id": result.get("reference_calibration_id"),
        "reference_trust": result.get("reference_trust"),
        "reference_reason": result.get("reference_reason"),
        "requested": bool(result.get("requested")),
        "rehome_from": result.get("rehome_from"),
        "previous_reference_id": result.get("previous_reference_id"),
        "message": " ".join(str(result.get("message") or "").split())[:MESSAGE_CHARS],
        "preset": preset.name, "mount": preset.rig.mount, "mock": bool(result.get("mock"))}, now)


# ------------------------------------------------------------------ the mount of a geometry

_MOBILE_CACHE: dict = {}        # (builtin dir, user dir) -> (files' signature, {geometry: names})


def _builtin_dir(store, preset, presets_dir) -> Path:
    if presets_dir is not None:
        return Path(presets_dir)
    src = getattr(preset, "source", None)
    user = config.user_presets_dir(store.root).resolve()
    if src is not None:
        folder = Path(src).resolve().parent
        if user != folder and user not in folder.parents:
            return folder                   # a built-in preset: its own folder
    return Path(config.PRESETS_DIR)


def _mobile_presets(store, builtin: Path) -> dict:
    """{geometry_id as written in the presets: [names of presets with rig.mount mobile]} of
    the built-in folder and this library's user presets (re-read only when a file changed)."""
    user = config.user_presets_dir(store.root)
    files = []
    try:
        files += sorted(f for f in Path(builtin).glob("*.json") if f.is_file())
    except OSError:
        pass
    files += [user / n / config.USER_PRESET_FILE for n in config.user_preset_names(user)]
    sig = []
    for f in files:
        try:
            st = f.stat()
            sig.append((str(f), st.st_mtime_ns, st.st_size))
        except OSError:
            continue
    key, sig = (str(builtin), str(user)), tuple(sig)
    hit = _MOBILE_CACHE.get(key)
    if hit is not None and hit[0] == sig:
        return hit[1]
    out: dict = {}
    for name, _, _ in sig:
        try:
            d = json.loads(Path(name).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        rig = d.get("rig") if isinstance(d, dict) else None
        if isinstance(rig, dict) and rig.get("mount") == "mobile" and d.get("geometry_id"):
            out.setdefault(str(d["geometry_id"]), []).append(str(d.get("name") or Path(name).stem))
    _MOBILE_CACHE[key] = (sig, out)
    return out


def mount_of(store, preset, presets_dir=None) -> tuple:
    """(mount, declared_by): "mobile" when this preset says so, or when another preset at the
    same geometry does (declared_by: its name) - a geometry_id names one rig, and a rig that is
    carried is carried whichever preset drives it. Else the preset's own mount."""
    if preset.rig.mount == "mobile":
        return "mobile", preset.name
    geo = getattr(preset, "rig_geometry_id", None) or preset.geometry_id
    names = _mobile_presets(store, _builtin_dir(store, preset, presets_dir)).get(str(geo))
    if names:
        return "mobile", names[0]
    return preset.rig.mount, None


# ------------------------------------------------------------------ the session's floor

_PROCESS_SESSIONS: dict = {}    # library root -> when the long-running owner of the rig started
_BOOT: list = []                # [datetime | None] once asked


def _root_key(store) -> str:
    return str(Path(store.root).resolve())


def open_process_session(store, now: Optional[datetime] = None) -> datetime:
    """This long-running process owns the rig from NOW (the web UI server's start-up): for
    every mobile geometry of this library, also one it first sees later, the session is never
    older than this - whatever verified the rig before the process started no longer does."""
    t = now or _now()
    _PROCESS_SESSIONS[_root_key(store)] = t
    return t


def close_process_session(store) -> None:
    _PROCESS_SESSIONS.pop(_root_key(store), None)


def boot_time() -> Optional[datetime]:
    """When this computer (the rig's controller) started, by its own clock; None when the
    operating system does not say. A verification never outlives a restart of the computer.
    (A hybrid shutdown that keeps the up-time running is not seen as one.)"""
    if not _BOOT:
        up = None
        try:
            if sys.platform == "win32":
                import ctypes
                fn = ctypes.windll.kernel32.GetTickCount64
                fn.restype = ctypes.c_ulonglong
                up = float(fn()) / 1000.0
            else:
                with open("/proc/uptime", encoding="ascii") as f:
                    up = float(f.read().split()[0])
        except Exception:                   # noqa: BLE001 - unknown: no such floor
            up = None
        _BOOT.append(None if up is None or up < 0 else
                     datetime.fromtimestamp(_time.time() - up))
    return _BOOT[0]


def _floor(store, now: datetime) -> tuple:
    """(when the session may have started at the earliest, in words)."""
    best = now - timedelta(seconds=MOBILE_SESSION_MAX_AGE_S)
    why = f"{MOBILE_SESSION_MAX_AGE_S / 3600.0:g} h ago (the longest a session lasts)"
    boot = boot_time()
    if boot is not None and best < boot <= now:
        best, why = boot, f"this computer started ({_say(boot)})"
    proc = _PROCESS_SESSIONS.get(_root_key(store))
    if proc is not None and best < proc <= now:
        best, why = proc, f"this program started ({_say(proc)}: a new rig session)"
    return best, why


# ------------------------------------------------------------------ what the record says

def _calibration(store, preset):
    from .calibrate import stereo
    return stereo.latest(store, preset.geometry_id, config.geometry_fingerprint(preset))


def _cal_time(cal) -> Optional[datetime]:
    if cal is None:
        return None
    return entry_time(getattr(cal, "path", None)) or _parse(getattr(cal, "created", None))


def _say(t: Optional[datetime]) -> str:
    return t.isoformat(sep=" ", timespec="seconds") if t is not None else "never"


_TOL: list = []


def default_tol_px() -> float:
    """The rig check's default MOVED tolerance (px): only an OK at this or tighter answers."""
    if not _TOL:
        from .capture import rigcheck
        _TOL.append(float(rigcheck.DEFAULT_TOL_PX))
    return _TOL[0]


def _objects(r: dict) -> bool:
    return r["event"] == "rig_check" and r.get("status") not in GOOD


def _answer(r: dict, mobile: bool) -> tuple:
    """(does this good rig check answer?, why not - "" when there is nothing to say)."""
    st, when = r.get("status"), _say(r["time"])
    trust = r.get("reference_trust") if isinstance(r.get("reference_trust"), dict) else {}
    if st == "REFERENCE_SET":
        if mobile:
            return False, (f"the reference stored at {when} was taken while the rig was not "
                           "verified: a check against it verifies nothing"
                           if trust.get("mobile") is not True else "")
        return bool(r.get("requested")), ""
    ref = r.get("previous_reference_id") if st == "REHOMED" else r.get("reference_id")
    if st == "OK":
        tol, lim = r.get("tol_px"), default_tol_px()
        if not isinstance(tol, (int, float)) or isinstance(tol, bool) or not tol <= lim + 1e-9:
            return False, (f"the rig check of {when} said OK at a tolerance of {tol!r} px: only "
                           f"{lim:g} px or tighter verifies")
        if r.get("reference_calibration_id") != r.get("calibration_id"):
            return False, (f"the rig check of {when} compared with the reference {ref}, taken "
                           f"with calibration {r.get('reference_calibration_id') or 'none'}, not "
                           f"with the one in use ({r.get('calibration_id') or 'none'})")
    elif r.get("reference_calibration_id") != r.get("rehome_from") or not r.get("rehome_from"):
        return False, (f"the re-home of {when} was not fitted against a reference of the "
                       "calibration it re-homed")
    ok = trust.get("mobile") is True if mobile else trust.get("fixed") is not False
    if not ok:
        what = "re-home" if st == "REHOMED" else "rig check"
        return False, (f"the {what} of {when} used the reference {ref}, stored while the rig "
                       + ("was not verified" if mobile else "check's objection stood")
                       + ": a reference taken after the rig may have moved verifies nothing")
    return True, ""


def _tail(back, mobile: bool, cal_id, floor: Optional[datetime],
          split: Optional[datetime]) -> tuple:
    """(the newest records that decide, oldest first; capped). mobile: back to the newest
    session marker (kept) and no older than the floor. fixed: back to the newest objection
    (kept), to a rig check that answers for the calibration in use, or to the last pose of
    that calibration's own record (a calibration from poses captured since answers
    everything older); with `split` (a stored scan's time) those ends only count below it.
    capped: MAX_TAIL lines were read without reaching such an end (the caller fails
    closed)."""
    out, pending, seen_cal, n = [], None, False, 0
    for r in back:
        if mobile:
            if r["time"] < floor:
                break
            out.append(r)
            if r["event"] == "session":
                break
        else:
            out.append(r)
            if split is not None and r["time"] >= split:
                continue                    # after the stored scan: all of it is looked at
            ev = r["event"]
            if ev == "rig_check":
                if _objects(r):
                    break
                if r.get("calibration_id") == cal_id and _answer(r, False)[0]:
                    break
            elif ev == "calibration" and not seen_cal and r.get("calibration_id") == cal_id:
                seen_cal = True
                pending = set(r.get("poses") or []) or None
            elif ev == "calib_pose" and pending is not None:
                pending.discard(r.get("pose_id"))
                if not pending:
                    break
        n += 1
        if n >= MAX_TAIL:
            out.reverse()
            return out, True
    out.reverse()
    return out, False


def _run(tail: list, mobile: bool, cal_id, split: Optional[datetime] = None) -> dict:
    """Walk the records in their order -> {"objection" (the newest rig check that objects),
    "answer" ({"by", "rec"}: what answered since, FOR THE CALIBRATION IN USE - an answer for
    another calibration is only a note), "last" (the newest rig check), "notes" (why a good
    record did not answer), "seen" (the latest clock time on the record up to the answer),
    and with `split`: "at_split" (objection, answer just below it) and "after" (the records
    at / after it that object or answer)}."""
    poses, objection, answer, last, notes = set(), None, None, None, []
    at_split, after, clock, seen = None, [], None, None

    def note(text: str) -> None:
        notes[:] = [text]                   # only the newest one is ever said

    def other(what: str, r: dict) -> None:
        note(f"{what} {'verified' if mobile else 'answers for'} calibration "
             f"{r.get('calibration_id') or 'none'}, not {'' if mobile else 'for '}the one in "
             f"use ({cal_id or 'none'})")
    for r in tail:
        if split is not None and at_split is None and r["time"] >= split:
            at_split = (objection, answer)
        clock = r["time"] if clock is None or r["time"] > clock else clock
        was = answer
        ev = r["event"]
        if ev == "session":
            if mobile:
                poses, answer = set(), None
        elif ev == "calib_pose":
            poses.add(r.get("pose_id"))
        elif ev == "calibration":
            ps = r.get("poses") or []
            by = (f"the calibration {r.get('calibration_id')} made "
                  + ("in this session" if mobile else "since"))
            if ps and all(p in poses for p in ps):
                if r.get("calibration_id") == cal_id:
                    answer = {"by": by, "rec": r}
                else:
                    other(by, r)
            else:
                note(f"the calibration {r.get('calibration_id')} of {_say(r['time'])} was "
                     "solved from poses captured "
                     + ("before this session" if mobile and objection is None else
                        "before the rig was last in doubt")
                     + " (or not on record): solving stored poses again verifies nothing - "
                       "capture the poses now")
        else:
            last = r
            if _objects(r):
                objection, answer, poses = r, None, set()
            else:
                yes, why = _answer(r, mobile)
                if yes:
                    kind = {"OK": "the rig check OK", "REHOMED": "the re-home",
                            "REFERENCE_SET": "the reference stored on request"}[r["status"]]
                    by = f"{kind} of {_say(r['time'])}"
                    if r.get("calibration_id") == cal_id:
                        answer = {"by": by, "rec": r}
                    else:
                        other(by, r)
                elif why:
                    note(why)
        if answer is not None and answer is not was:
            seen = clock
        if at_split is not None and (answer is not was and answer is not None or _objects(r)
                                     or r.get("status") == "REHOMED"):
            after.append(r)                 # (a re-home: its check had found the camera moved)
    if split is not None and at_split is None:
        at_split = (objection, answer)
    return {"objection": objection, "answer": answer, "last": last, "notes": notes,
            "seen": seen, "at_split": at_split, "after": after}


def state(store, preset, now: Optional[datetime] = None, presets_dir=None,
          mount: Optional[str] = None, scan_time: Optional[datetime] = None) -> dict:
    """What is known about the rig at this preset's geometry: {"mount", "declared_by" (the
    preset that says mobile, when it is not this one), "calibration_id", "calibration_time",
    "session_start" (mobile; None for a fixed rig), "session_marker", "last_check" (the
    newest rig check that still matters: {time, status, ...} or None), "objection" (the
    newest rig check that objects and was not answered, else None), "verified" (mobile: True
    / False; fixed: None), "verified_at", "verified_by", "why"}. Times are datetimes.
    mount: judge the record as "fixed" / "mobile" whatever the preset says (what a reference
    stored now could be trusted for). scan_time (fixed): also "scan_objection", why a stored
    scan made then is in doubt (None: it is not)."""
    now = now or _now()
    said, by = mount_of(store, preset, presets_dir)
    mnt = mount or said
    cal = _calibration(store, preset)
    cal_id = cal.id if cal is not None else None
    out = {"mount": mnt, "declared_by": by if (by and by != preset.name and mount is None) else None,
           "calibration_id": cal_id, "calibration_time": _cal_time(cal),
           "session_start": None, "session_marker": None, "last_check": None,
           "objection": None, "verified": None, "verified_at": None, "verified_by": None,
           "why": None, "max_age_s": MOBILE_SESSION_MAX_AGE_S}
    back = read_back(store, preset.geometry_id, now)
    if mnt != "mobile":
        tail, capped = _tail(back, False, cal_id, None, scan_time)
        got = _run(tail, False, cal_id, scan_time)
        bad = got["objection"] if got["answer"] is None else None
        if bad is None and got["answer"] is None and capped:
            # MAX_TAIL lines and none of them confirms the rig for this calibration (nor
            # objects): what is older is not read, so the rig is not taken for confirmed
            bad = {"time": tail[0]["time"], "event": "rig_check", "status": "UNKNOWN",
                   "message": (f"none of the newest {MAX_TAIL} lines of the rig-state record "
                               "confirms the rig for the calibration in use (rig checks "
                               "against another calibration's reference, or at a loose "
                               "tolerance): what is older is not read")}
        if bad is not None and got["notes"]:
            out["why"] = got["notes"][-1]
        out.update(last_check=got["last"], objection=bad)
        if scan_time is not None:
            out["scan_objection"] = _scan_doubt(got, scan_time)
        return out
    floor, floor_why = _floor(store, now)
    tail, capped = _tail(back, True, cal_id, floor, None)
    marker = tail[0]["time"] if tail and tail[0]["event"] == "session" else None
    if capped:
        floor_why = (f"the newest {MAX_TAIL} lines of the rig-state record began (what is "
                     "older is not read: run a rig check)")
    got = _run(tail, True, cal_id)
    out.update(session_start=marker if marker is not None else floor, session_marker=marker,
               last_check=got["last"], objection=got["objection"])
    ans = got["answer"]
    if cal_id is not None and ans is not None:
        return {**out, "objection": None, "verified": True, "verified_at": ans["rec"]["time"],
                "verified_by": ans["by"], "verified_floor": got["seen"]}
    since = f"the session marker of {_say(marker)}" if marker is not None else floor_why
    why = f"nothing has verified it since {since}"
    if cal_id is None:
        why = "no stereo calibration at this geometry and these rig settings"
    elif got["objection"] is not None:
        bad = got["objection"]
        why = f"the rig check of {_say(bad['time'])} said {bad.get('status')}"
        if got["notes"]:
            why += "; " + got["notes"][-1]
    elif got["notes"]:
        why += ": " + got["notes"][-1]
    return {**out, "verified": False, "why": why}


def _scan_doubt(got: dict, t: datetime) -> Optional[str]:
    """Why a stored scan made at t by a FIXED rig is in doubt: an objection stood when it was
    made, or the first rig check after it objected (the rig moved at some time before that
    check - the scan may have been made after it did)."""
    bad, ans = got["at_split"]
    if bad is not None and ans is None:
        return (f"it was made at {_say(t)}, while the rig check of {_say(bad['time'])} "
                f"({bad.get('status')}) stood")
    for r in got["after"]:
        moved = r.get("status") if _objects(r) else (
            "MOVED (and re-homed it)" if r.get("status") == "REHOMED" else None)
        if moved:
            return (f"it was made at {_say(t)} and the next rig check ({_say(r['time'])}) said "
                    f"{moved}: the rig moved at some time before that check")
        return None                         # the rig was confirmed in place after the scan
    return None


def trust_now(store, preset, requested: bool = False, now: Optional[datetime] = None,
              presets_dir=None) -> dict:
    """What a rig reference stored NOW can be trusted for (kept with it and with every check
    made against it): {"mobile": the rig is verified at this moment, by the mobile rule;
    "fixed": no rig check objection stands, or the operator asked for the reference}."""
    mob = state(store, preset, now, presets_dir, mount="mobile")
    fix = state(store, preset, now, presets_dir, mount="fixed")
    return {"mobile": mob["verified"] is True,
            "fixed": bool(requested) or fix["objection"] is None}


# ------------------------------------------------------------------ the gate

def _scan_time(scan) -> Optional[datetime]:
    return entry_time(getattr(scan, "path", None)) or _parse(getattr(scan, "created", None))


def gate(store, preset, strict: bool = False, scan=None, now: Optional[datetime] = None,
         presets_dir=None) -> dict:
    """May this rig give a mm result now? -> {"ok", "code" (None, or the doubt:
    engine/disposition.REASONS `rig_unverified` | `rig_check`), "message", "how_to_fix",
    "mount", "state" (state(), times as text)}.

    mobile (the preset, or another preset at its geometry, says so): ok only when verified
    in this session (module docstring); a stored `scan` (a library entry: its folder is its
    time) must also have been made after that verification. The doubt is always
    `rig_unverified` - also after a MOVED: it says nothing about the part, so no recipe can
    turn it into a reject.
    fixed: ok - unless strict (Measure and Quick check) and a rig check's MOVED / UNKNOWN
    stands unanswered (`rig_check`); a stored `scan` must then have been made while no
    objection stood, and not just before one. Station mode calls it with strict False, so a
    fixed rig is never gated here (unchanged behaviour)."""
    now = now or _now()
    mount, by = mount_of(store, preset, presets_dir)
    if mount != "mobile" and not strict:
        return {"ok": True, "code": None, "message": "", "how_to_fix": "",
                "mount": mount, "state": None}                  # Station mode, fixed: unchanged
    t = _scan_time(scan) if scan is not None else None
    st = state(store, preset, now, presets_dir, scan_time=t if mount != "mobile" else None)
    out = {"ok": True, "code": None, "message": "", "how_to_fix": "", "mount": mount,
           "state": public(st)}
    name = getattr(scan, "id", scan)
    late = scan is not None and (t is None or t > now + timedelta(seconds=CLOCK_SLACK_S))
    if mount == "mobile":
        twin = (f" - preset {by} declares the rig at this geometry mobile"
                if by and by != preset.name else "")
        if not st["verified"]:
            return {**out, "ok": False, "code": "rig_unverified",
                    "message": f"{UNVERIFIED} ({st['why']}){twin}", "how_to_fix": HOW_MOBILE}
        if scan is not None:
            since = max(st["verified_at"], st.get("verified_floor") or st["verified_at"])
            if late or t < since:
                when = ("has no time of its own, or one later than now (the clock was set "
                        "back)" if late else f"was made at {_say(t)}, before the rig was "
                        f"verified at {_say(since)}")
                return {**out, "ok": False, "code": "rig_unverified",
                        "message": f"{UNVERIFIED} (the stored scan {name} {when}: scan "
                                   f"again){twin}",
                        "how_to_fix": "scan again: the rig is verified now"}
        return out
    bad = st["objection"]
    if bad is not None:
        return {**out, "ok": False, "code": "rig_check",
                "message": (f"rig check {bad.get('status')} at {_say(bad['time'])}: "
                            f"{bad.get('message') or 'the rig position is not confirmed'}"
                            + (f" ({st['why']})" if st.get("why") else "")),
                "how_to_fix": HOW_MOVED}
    if scan is not None:
        doubt = ("it has no time of its own, or one later than now (the clock was set back)"
                 if late else st.get("scan_objection"))
        if doubt:
            return {**out, "ok": False, "code": "rig_check",
                    "message": f"the stored scan {name} is in doubt: {doubt} - scan again",
                    "how_to_fix": "scan again: no rig check objects now"}
    return out


def require(s, preset, what: str, scan=None) -> dict:
    """The gate for the engine's own capture verbs (a metric scan, a fast scan, ...): a
    MOBILE rig that is not verified in this session gives no mm at all - rules.RuleViolation
    before the rig is touched. A fixed rig is never refused here (unchanged); the gate then
    also carries "doubt" (doubt(): what Measure and Quick check would say against its mm now,
    None when nothing objects), which stamp() records. -> the gate."""
    pdir = getattr(s, "presets_dir", None)
    g = gate(s.store, preset, strict=False, scan=scan, presets_dir=pdir)
    if not g["ok"]:
        raise rules.RuleViolation(f"{what}: {g['message']} | fix: {g['how_to_fix']}")
    if g["mount"] != "mobile":
        g = {**g, "doubt": doubt(s.store, preset, scan=scan, presets_dir=pdir)}
    return g


def doubt(store, preset, scan=None, now: Optional[datetime] = None,
          presets_dir=None) -> Optional[dict]:
    """A FIXED rig's mm that are given (require() never refuses a fixed rig) but that Measure
    and Quick check would refuse now - a rig check's MOVED / UNKNOWN stands unanswered, or a
    stored `scan` was made while it stood or just before it (gate(strict=True)) ->
    {"code", "message" (the gate's: the rig check's own words), "how_to_fix" (rig check OK
    again, a new reference, a recalibration), "warning" (one line: IN_DOUBT, the message and
    the way out)}; None when nothing objects, and for a mobile rig (refused instead). Never
    raises: a record that can't be read is a doubt too (fail closed) - this only adds a
    warning to mm that are given anyway, it never stops them."""
    try:
        if mount_of(store, preset, presets_dir)[0] == "mobile":
            return None
        g = gate(store, preset, strict=True, scan=scan, now=now, presets_dir=presets_dir)
    except Exception as e:                  # noqa: BLE001 - a warning never stops the mm
        g = {"ok": False, "code": "rig_check", "how_to_fix": HOW_MOVED,
             "message": (f"the rig-state record could not be read ({type(e).__name__}: {e}): "
                         "the rig position is not confirmed")}
    if g["ok"]:
        return None
    return {"code": g["code"], "message": g["message"], "how_to_fix": g["how_to_fix"],
            "warning": f"{IN_DOUBT} - {g['message']} | fix: {g['how_to_fix']}"}


def stamp(g: dict) -> Optional[dict]:
    """What a metric scan records of the gate it was captured under (meta "rig"): a mobile
    rig's verification; a fixed rig's state from require() - {"mount": "fixed", "trusted"}
    and, when not trusted, doubt()'s code / message / how_to_fix / warning. None for a fixed
    rig's gate that require() did not make."""
    st = g.get("state") or {}
    if g.get("mount") != "mobile":
        if "doubt" not in g:
            return None
        d = g["doubt"]
        return {"mount": "fixed", "trusted": d is None, **(d or {})}
    return {"mount": "mobile", "verified_by": st.get("verified_by"),
            "verified_at": st.get("verified_at")}


def public(st: dict) -> dict:
    """state() as plain JSON data (times as text, the checks in short)."""
    def t(v):
        return v.isoformat(timespec="seconds") if isinstance(v, datetime) else v

    def check(r):
        if r is None:
            return None
        return {"time": t(r["time"]), "status": r.get("status"), "moved": r.get("moved"),
                "shift_px": r.get("shift_px"), "camera_motion_px": r.get("camera_motion_px"),
                "projector_motion_px": r.get("projector_motion_px"), "tol_px": r.get("tol_px"),
                "calibration_id": r.get("calibration_id"),
                "reference_id": r.get("reference_id"),
                "unreadable": bool(r.get("unreadable")),
                "message": r.get("message")}
    skip = ("last_check", "objection", "verified_floor")
    return {**{k: t(v) for k, v in st.items() if k not in skip},
            "last_check": check(st["last_check"]), "objection": check(st["objection"])}

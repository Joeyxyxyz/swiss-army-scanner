"""
The zone selection: which painted zone sets (and which zones inside them) are switched on at
one rig placement. It reads like an inspection profile: paint, name, save, switch on, run.

    <library>/zones/<geometry_id>/_selection.jsonl

Append-only. The NEWEST line is the selection; the lines before it are its history (who
selected what, when). One line:

    {"schema": "zone-selection-1", "geometry_id": "...", "time": "2026-01-02T09:14:02",
     "by": "operator page", "sets": [{"set": "cells", "on": true, "ids": null},
                                     {"set": "tube", "on": false, "ids": ["Z1", "Z2"]}],
     "text": "zoneset:cells"}

ids null = every zone of the set (also zones painted into it later); a list = exactly those,
never empty for a set that is on. An OFF entry only remembers the per-zone switches of a set
that is switched off. `text` is the zone text of the ON entries ("" = the whole field).

Nothing here is repaired silently:
  - a selection that names something gone (a set deleted or renamed, a zone id a repaint
    removed, a set painted at other rig settings, an unreadable set) is a PROBLEM: status()
    says which, and every run that asks for the selection is refused until the operator
    switches that set off or paints it again;
  - a newest line that cannot be read makes the selection UNKNOWN: an older line is never
    used in its place (that would bring back a selection somebody changed);
  - a write is refused when an entry that is on does not resolve now.

The word `selection` stands for the stored selection wherever a zone text is taken (a scan's
zone, a measurement's zone, a quick check's zone, a metric station job's zone). expand() turns
it into the zone text "zoneset:<a>+<b>:Z1" ONCE, at the start of a verb; that text is what is
lit and recorded. `selection@<stamp>` is "the selection exactly as I was shown it": the stamp
(status()["stamp"]) changes with the sets, their zones and the painted time of every set that
is on, so a selection that changed under a page, or a set that was painted again, is refused
instead of run. With a stamp an empty selection is the whole field (the sender proved it knew
that); the bare word refuses an empty selection (a script or a station file that asks for
"the selected zones" never falls back to the whole field).

Files only: never the rig.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from .. import config, rules
from . import store as zstore
from . import use
from .model import ZoneError, valid_set_name, valid_zone_id

FILE = "_selection.jsonl"
SCHEMA = "zone-selection-1"
WORD = "selection"
STATES = ("ok", "problem", "unknown")
_WORD = re.compile(r"^selection(?:@([A-Za-z0-9]{1,40}))?$", re.I)


class SelectionChanged(ZoneError):
    """The stored selection is not the one the sender was shown (its stamp differs): what
    was asked for is refused, and the sender must look again."""


# ------------------------------------------------------------------ the word

def is_word(text) -> bool:
    """Is this zone text the reserved word (`selection`, `selection@<stamp>`)?"""
    return isinstance(text, str) and bool(_WORD.fullmatch(text.strip()))


def word_stamp(text) -> Optional[str]:
    """The stamp of `selection@<stamp>` (None: the bare word, or not the word)."""
    m = _WORD.fullmatch(text.strip()) if isinstance(text, str) else None
    return m.group(1) if m else None


# ------------------------------------------------------------------ the file

def file_of(store, geometry_id: str) -> Path:
    return Path(store.root) / zstore.KIND / zstore._geo(geometry_id) / FILE


def _record(raw: bytes, geometry_id: str) -> tuple:
    """One line -> (record, None), or (None, why it cannot be read)."""
    try:
        rec = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as e:
        return None, f"not UTF-8 JSON ({e})"
    if not isinstance(rec, dict) or rec.get("schema") != SCHEMA:
        return None, f"not a {SCHEMA} record"
    if rec.get("geometry_id") != geometry_id:
        return None, f"it is for geometry {rec.get('geometry_id')!r}"
    sets = rec.get("sets")
    if not isinstance(sets, list):
        return None, "sets is not a list"
    seen = set()
    for e in sets:
        ids = e.get("ids") if isinstance(e, dict) else 0
        if (not isinstance(e, dict) or not valid_set_name(e.get("set"))
                or not isinstance(e.get("on"), bool) or e["set"] in seen
                or not (ids is None or (isinstance(ids, list) and len(set(ids)) == len(ids)
                                        and all(valid_zone_id(i) for i in ids)))):
            return None, f"a malformed entry ({e!r})"
        seen.add(e["set"])
    return rec, None


def read(store, geometry_id: str) -> dict:
    """The geometry's stored selection -> {"state": "none" (never set) | "ok" | "unreadable",
    "record": the newest line (ok) or None, "lines": how many, "why": why unreadable}."""
    f = file_of(store, geometry_id)
    try:
        data = f.read_bytes()
    except FileNotFoundError:
        return {"state": "none", "record": None, "lines": 0, "why": None}
    except OSError as e:
        return {"state": "unreadable", "record": None, "lines": 0,
                "why": f"{f.name}: {e.strerror or e}"}
    lines = [raw for raw in data.splitlines() if raw.strip()]
    if not lines:
        return {"state": "none", "record": None, "lines": 0, "why": None}
    rec, why = _record(lines[-1], geometry_id)
    if rec is None:
        return {"state": "unreadable", "record": None, "lines": len(lines),
                "why": f"the newest line of {f.name} cannot be read: {why}"}
    return {"state": "ok", "record": rec, "lines": len(lines), "why": None}


def _append(store, geometry_id: str, entries: list, text: str, by: str) -> dict:
    from ..library.presets import append_selection
    rec = {"schema": SCHEMA, "geometry_id": geometry_id,
           "time": datetime.now().isoformat(timespec="seconds"), "by": str(by or ""),
           "sets": [{"set": e["set"], "on": bool(e["on"]),
                     "ids": list(e["ids"]) if e.get("ids") else None} for e in entries],
           "text": text}
    f = file_of(store, geometry_id)
    f.parent.mkdir(parents=True, exist_ok=True)
    append_selection(f, rec)
    return rec


# ------------------------------------------------------------------ status

def _stamp(geometry_id: str, state: str, text: str, painted: list) -> str:
    raw = json.dumps([geometry_id, state == "unknown", text, painted], sort_keys=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _text(rows: list) -> str:
    """The zone text of the rows that are on ('' = nothing on: the whole field)."""
    terms = [(r["set"], r["ids"]) for r in rows if r["on"]]
    return use.text_multi(terms) if terms else ""


def status(store, preset) -> dict:
    """The selection of the preset's rig placement, checked against the library NOW:

        state     ok | problem (an entry that is on does not resolve) | unknown (unreadable)
        ok        state == "ok"; empty: nothing is switched on (the whole field)
        text      the zone text of what is on ("" when empty); label: the same in words
                  ("cells (2), tube (2 of 3)" / "whole field" / "NOT KNOWN")
        stamp     changes whenever what a run would light changes (expand, the page's check)
        problems  [{set, zone, reason: missing | other_geometry | other_settings |
                   zone_missing | invalid | unreadable, message}]
        sets      every set painted at this geometry + every set the selection has on:
                  [{set, on, ids (null = all), n_on, n_zones, found, usable, error, created,
                    preset, zones [{id, name, on}], rig_changes [...]}]
        time, by  when and by whom it was last changed (None: never)

    Files only. A set's zone switches are remembered while it is off."""
    geo = preset.geometry_id
    got = read(store, geo)
    rec = got["record"] or {}
    entries = {e["set"]: e for e in rec.get("sets") or []}
    fp = config.geometry_fingerprint(preset)
    rows, problems = [], []

    def problem(name, reason, message, zone=None):
        problems.append({"set": name, "zone": zone, "reason": reason,
                         "message": " ".join(str(message).split())})

    names = zstore.names(store, geo)
    for name in sorted(set(names) | {n for n, e in entries.items() if e["on"]}):
        e = entries.get(name) or {"on": False, "ids": None}
        row = {"set": name, "on": bool(e["on"]), "ids": list(e["ids"]) if e["ids"] is not None
               else None, "found": name in names, "usable": False, "error": None,
               "n_zones": 0, "n_on": 0, "created": None, "preset": None, "zones": [],
               "rig_changes": []}
        zs = None
        if not row["found"]:
            others = zstore.geometries_of(store, name)
            row["error"] = (f"painted at geometry {', '.join(others)}, not at this rig "
                            f"placement ({geo})" if others else
                            "not in the library any more (deleted or renamed)")
            problem(name, "other_geometry" if others else "missing",
                    f"zone set {name!r} is switched on but is {row['error']}")
        else:
            try:
                zs = zstore.load(store, geo, name)
            except ZoneError as err:
                row["error"] = str(err).splitlines()[0]
                if row["on"]:
                    problem(name, "invalid", f"zone set {name!r} cannot be read: {row['error']}")
        if zs is not None:
            ids = row["ids"]
            have = zs.ids
            row.update(n_zones=len(have), created=zs.created, preset=zs.preset,
                       zones=[{"id": z.id, "name": z.name, "on": ids is None or z.id in ids}
                              for z in zs.zones],
                       rig_changes=zstore.rig_changes(store, zs))
            row["n_on"] = sum(1 for z in row["zones"] if z["on"])
            diffs = rules.fingerprint_diffs(zs.fingerprint, fp)
            if diffs:
                row["error"] = f"painted at other rig settings ({'; '.join(diffs)})"
                if row["on"]:
                    problem(name, "other_settings",
                            f"zone set {name!r} was {row['error']}: paint it again with "
                            f"preset {preset.name}")
            else:
                row["usable"] = True
            if row["on"] and ids is not None:
                gone = [i for i in ids if i not in have]
                for i in gone:
                    problem(name, "zone_missing", f"zone set {name!r} has no zone {i!r} (it "
                            f"has {', '.join(have) or 'none'}): it was painted again", zone=i)
                if not ids:
                    problem(name, "invalid", f"zone set {name!r} is switched on with no zone "
                            "switched on in it")
        rows.append(row)

    state = "unknown" if got["state"] == "unreadable" else ("problem" if problems else "ok")
    if state == "unknown":
        problems = [{"set": None, "zone": None, "reason": "unreadable", "message": got["why"]}]
        for r in rows:                              # nothing is known to be on
            r["on"] = False
    on = [r for r in rows if r["on"]]
    text = "" if state == "unknown" else _text(on)
    bad = {p["set"]: p["reason"] for p in reversed(problems)}
    words = {"missing": "NOT FOUND", "other_geometry": "NOT FOUND", "zone_missing": "ZONE MISSING",
             "other_settings": "NOT USABLE", "invalid": "NOT USABLE"}

    def piece(r) -> str:
        if r["set"] in bad:
            return f"{r['set']} ({words.get(bad[r['set']], 'NOT USABLE')})"
        n = (str(r["n_on"]) if r["n_on"] == r["n_zones"] else f"{r['n_on']} of {r['n_zones']}")
        return f"{r['set']} ({n}{', CHECK' if r['rig_changes'] else ''})"

    label = "NOT KNOWN" if state == "unknown" else (", ".join(piece(r) for r in on)
                                                    or "whole field")
    return {"geometry_id": geo, "preset": preset.name, "state": state, "ok": state == "ok",
            "empty": state != "unknown" and not on, "text": text, "label": label,
            "stamp": _stamp(geo, state, text, [[r["set"], r["created"]] for r in on]),
            "problems": problems, "sets": rows,
            "n_sets_on": len(on), "n_zones_on": sum(r["n_on"] for r in on),
            "check": [f"{r['set']}: {'; '.join(r['rig_changes'][:3])}" for r in on
                      if r["rig_changes"]],
            "time": rec.get("time"), "by": rec.get("by"), "history_lines": got["lines"]}


def describe(st: dict) -> str:
    """The selection in one line of words (the CLI, a log)."""
    if st["state"] == "unknown":
        return f"NOT KNOWN ({st['problems'][0]['message']})"
    return st["label"] + ("" if st["ok"] else "  [cannot be used: "
                          + "; ".join(p["message"] for p in st["problems"]) + "]")


# ------------------------------------------------------------------ changing it

def _clean(store, preset, sets) -> list:
    """Entries as given ([{set, on, ids}]) -> the entries that are stored: every ON entry
    resolves now (the set at this geometry and these rig settings, every id in the set, at
    least one zone on), a full id list becomes null, an OFF entry of a set that is gone is
    dropped. ZoneError / GeometryMismatch name what does not resolve; nothing is written."""
    if not isinstance(sets, (list, tuple)):
        raise ZoneError("the selection is a list of {set, on, ids}")
    out, seen = [], set()
    for e in sets:
        name = e.get("set") if isinstance(e, dict) else None
        if not valid_set_name(name):
            raise ZoneError(f"selection entry {e!r}: `set` must be a zone set name")
        if name in seen:
            raise ZoneError(f"zone set {name!r} is named twice in the selection")
        seen.add(name)
        on, ids = e.get("on"), e.get("ids")
        if not isinstance(on, bool):
            raise ZoneError(f"zone set {name!r}: `on` must be true or false")
        if ids is not None and (not isinstance(ids, (list, tuple)) or len(set(ids)) != len(ids)
                                or not all(valid_zone_id(i) for i in ids)):
            raise ZoneError(f"zone set {name!r}: `ids` must be null (every zone) or a list of "
                            "its zone ids, each once")
        ids = None if ids is None else list(ids)
        if on:
            try:                                # geometry + rig settings, or refused
                zs = use.for_preset(store, preset, name)
            except (ZoneError, rules.GeometryMismatch) as err:
                raise type(err)(f"{err} - a zone set that is switched on must be usable at "
                                f"this rig placement: switch {name} off, or paint it "
                                "again") from None
            if ids is not None and not ids:
                raise ZoneError(f"zone set {name!r}: no zone is switched on in it - switch the "
                                "set off instead (an empty list never means every zone)")
            zs.select(ids)                                  # an unknown id is refused
            if ids is not None:
                ids = [i for i in zs.ids if i in ids]       # the set's own order
                if ids == zs.ids:
                    ids = None
        else:
            if not zstore.exists(store, preset.geometry_id, name):
                continue
            ids = ids or None
        out.append({"set": name, "on": on, "ids": ids})
    return sorted(out, key=lambda x: x["set"])


def write(store, preset, sets, by: str = "", expect: Optional[str] = None) -> dict:
    """Replace the selection of the preset's geometry with `sets` ([{set, on, ids}]) -> the
    new status(). expect: the stamp the sender was shown; refused (ZoneError, nothing
    written) when the stored selection is another one by now."""
    if expect is not None:
        now = status(store, preset)
        if str(expect) != now["stamp"]:
            raise SelectionChanged("the zone selection changed meanwhile (it is now: "
                                   f"{now['label']}): look again, then change it")
    entries = _clean(store, preset, sets)
    _append(store, preset.geometry_id, entries, _text(entries), by)
    return status(store, preset)


def change(store, preset, *, zones=None, on=None, off=None, clear: bool = False, sets=None,
           by: str = "", expect: Optional[str] = None) -> dict:
    """Change the selection of the preset's geometry -> the new status().

        sets    the whole selection as entries [{set, on, ids}] (what a page sends)
        zones   a zone text or a list of terms ("cells+tube:Z1"): exactly these are on (a
                set without ids: every zone of it)
        clear   everything off (the whole field)
        on      terms to switch on: "tube:Z1,Z2" = exactly those zones; "tube" = the set
                with the zone switches it had when it was switched off (every zone when
                it had none)
        off     set names to switch off (their zone switches are remembered)

    zones / clear start from "everything off"; on / off change what is stored. An entry that
    is on and does not resolve refuses the whole change (nothing is written)."""
    if sets is not None:
        if zones is not None or on or off or clear:
            raise ZoneError("give `sets` (the whole selection) or zones / on / off / clear, "
                            "not both")
        return write(store, preset, sets, by=by, expect=expect)
    got = read(store, preset.geometry_id)
    cur = {e["set"]: dict(e) for e in (got["record"] or {}).get("sets") or []}
    if clear or zones is not None:
        for e in cur.values():
            e["on"] = False
    named = []
    if zones is not None:
        for name, ids in use.parse_multi(use.join_terms(zones)):
            cur[name] = {"set": name, "on": True, "ids": ids}
            named.append(name)
    for t in ([on] if isinstance(on, str) else list(on or [])):
        for name, ids in use.parse_multi(use.join_terms(t)):
            if name in named:
                raise ZoneError(f"zone set {name!r} is named twice in one change")
            kept = (cur.get(name) or {}).get("ids")
            cur[name] = {"set": name, "on": True, "ids": ids if ids is not None else kept}
            named.append(name)
    for name in ([off] if isinstance(off, str) else list(off or [])):
        if not valid_set_name(name):
            raise ZoneError(f"--off takes a zone set name, got {name!r}")
        if name in named:
            raise ZoneError(f"zone set {name!r} is switched on and off in one change")
        if name in cur:
            cur[name]["on"] = False
        elif not zstore.exists(store, preset.geometry_id, name):
            raise ZoneError(f"no zone set {name!r} at geometry {preset.geometry_id} to switch off")
    return write(store, preset, list(cur.values()), by=by, expect=expect)


def after_paint(store, preset, zs, *, replaced: bool, select: bool, by: str = "") -> dict:
    """The selection after the set `zs` was saved by the painter -> {"on": is it switched on
    now, "reset": its per-zone switches were reset, "note": the same in words}.

    A NEW set: switched on when `select`, else left off - and never switched on by a stale
    entry of an earlier set of that name. A set that was painted over keeps its on / off.
    Either way its per-zone switches go back to "every zone": the painter re-issues zone ids,
    so an id that was switched off may be another drawing now. The other entries are not
    touched. An unreadable selection is left alone (never repaired silently)."""
    geo, name, n = preset.geometry_id, zs.name, len(zs.zones)
    got = read(store, geo)
    if got["state"] == "unreadable":
        return {"on": None, "reset": False,
                "note": "The stored zone selection cannot be read, so it was not changed: "
                        "set the selection again."}
    entries = [dict(e) for e in (got["record"] or {}).get("sets") or []]
    old = next((e for e in entries if e["set"] == name), None)
    was_on = bool(old and old["on"]) if replaced else False
    now_on = was_on if replaced else bool(select)
    reset = bool(old and old["ids"])
    keep = [e for e in entries if e["set"] != name]
    new = keep + ([{"set": name, "on": True, "ids": None}] if now_on else [])
    if old is not None or now_on:
        new.sort(key=lambda x: x["set"])
        _append(store, geo, new, _text(new), by)
    if not replaced:
        note = (f"It is switched ON (all {n} zones)." if now_on else
                "It is NOT switched on.")
    elif now_on:
        note = (f"It stays switched ON, now with all {n} zones on (the per-zone switches were "
                "reset: the zones were painted again)." if reset else
                f"It stays switched ON (all {n} zones).")
    else:
        note = "It stays switched OFF, as before."
    return {"on": now_on, "reset": reset, "note": note}


def after_delete(store, preset, name: str, by: str = "") -> bool:
    """A set was moved to the trash: its OFF entry leaves the selection (a later set of that
    name starts clean). An entry that is ON stays - only a forced delete gets here - and is
    a problem (NOT FOUND) until the operator switches it off. -> True when a line was written."""
    geo = preset.geometry_id
    got = read(store, geo)
    entries = [dict(e) for e in (got["record"] or {}).get("sets") or []]
    old = next((e for e in entries if e["set"] == name), None)
    if old is None or old["on"]:
        return False
    new = [e for e in entries if e["set"] != name]
    _append(store, geo, new, _text(new), by)
    return True


# ------------------------------------------------------------------ the word -> a zone text

def expand(store, preset, text, expect: Optional[str] = None) -> tuple:
    """A zone text as a verb was given it -> (the zone text to light, info).

    Anything but the reserved word comes back unchanged, with info None. The word is
    expanded into the zone text of the stored selection of the preset's geometry
    ("zoneset:cells+tube:Z1"), with info {source: "selection", selected: {time, by}, stamp,
    label}. Refused (ZoneError; nothing falls back to the whole field) when the selection
    is unknown, has a problem, is not the one the stamp names (`selection@<stamp>`, or
    `expect`), or - for the bare word - is empty. With a stamp IN THE WORD an empty
    selection is the whole field: (None, info). `expect` (the stamp a station START was
    accepted with) only compares: a file that says "the selected zones" with nothing
    switched on stays refused, whatever stamp travels with the run."""
    if not is_word(text):
        return text, None
    stamped = word_stamp(text)
    want = stamped or (str(expect) if expect else None)
    st = status(store, preset)
    geo = preset.geometry_id
    if st["state"] == "unknown":
        raise ZoneError(f"the zone selection of geometry {geo} is not known: "
                        f"{st['problems'][0]['message']}. Set it again (operator page: Scan "
                        "zones; scan zone-select)")
    if want and want != st["stamp"]:
        raise SelectionChanged("the zone selection changed since it was shown (it is now: "
                               f"{st['label']}): look at the selection again, then start again")
    if not st["ok"]:
        p = st["problems"][0]
        raise ZoneError(f"the zone selection cannot be used: {p['message']}. Switch that set "
                        f"off (scan zone-select --off {p['set']}) or paint it again")
    info = {"source": "selection", "selected": {"time": st["time"], "by": st["by"]},
            "stamp": st["stamp"], "label": st["label"]}
    if st["empty"]:
        if stamped:
            return None, info
        raise ZoneError(f"the zone selection of geometry {geo} is empty (the whole field): "
                        "switch a zone set on first (operator page: Scan zones; scan "
                        "zone-select --on SET)")
    return st["text"], info

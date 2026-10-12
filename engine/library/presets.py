"""
Preset library: built-in presets (the repo's presets/ folder, READ-ONLY here)
and user presets — part recipes an operator or integrator adds, selects and
deletes per library:

    <library>/presets/<name>/preset.json    the preset (validated like any other)
    <library>/presets/<name>/meta.json      created, author, description, copied from
    <library>/presets/active.jsonl          the selected preset; append-only, last line wins

A user preset is a copy of an existing one with fields overridden. Nothing is
replaced in place: `replace` moves the old entry to trash/ first, and delete
moves it to trash/presets/ (recoverable). A new geometry_id starts
UNCALIBRATED: nothing here creates or copies a calibration.
"""

from __future__ import annotations

import copy
import getpass
import json
import logging
import math
import os
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from .. import config
from ..config import ConfigError
from .store import LibraryStore

ACTIVE_FILE = "active.jsonl"
META_FILE = "meta.json"
FILE_SUFFIXES = (".json", ".yaml", ".yml")      # a name ending so reads as a file, not a name
# a YAML file that can't be parsed here (PyYAML is optional) is judged by what its text
# mentions, never skipped and never guessed at by a pattern for its keys: the recipe-file
# paths in it (quoted, plain or after a key, block or flow style) ...
_FILE_REFS = (re.compile(r"""(['"])([^'"\r\n]+?\.(?:json|ya?ml))\1""", re.I),       # quoted
              re.compile(r"""(?<![^\s'"{}\[\],:])([^\s'"{}\[\],#]+\.(?:json|ya?ml))"""
                         r"""(?![A-Za-z0-9_])""", re.I),                          # a plain word
              re.compile(r"""(?:^|:)[ \t]+([^'"#\r\n{}\[\],][^#\r\n{}\[\],]*?"""
                         r"""\.(?:json|ya?ml))[ \t]*(?:#.*)?$""", re.I | re.M))  # after a key
_log = logging.getLogger(__name__)


def user_dir(store: LibraryStore) -> Path:
    return config.user_presets_dir(store.root)


def builtin_names(builtin: Path) -> list:
    return config.list_presets(builtin)


def user_names(store: LibraryStore) -> list:
    d = user_dir(store)
    if not d.is_dir():
        return []
    return sorted(e.name for e in d.iterdir()
                  if e.is_dir() and config.valid_id(e.name)
                  and (e / config.USER_PRESET_FILE).is_file() and (e / META_FILE).is_file())


def tier_of(store: LibraryStore, builtin: Path, name: str) -> Optional[str]:
    """'builtin' | 'user' | None, by EXACT name (case-sensitive on every platform)."""
    if not isinstance(name, str):
        return None
    if name in builtin_names(builtin):
        return "builtin"
    if name in user_names(store):
        return "user"
    return None


def check_new_name(name, taken_builtin: list, taken_user: list, what: str = "preset",
                   replace: bool = False) -> None:
    """A new library entry name: an id that is the same folder name on every platform
    (config.valid_id), not a file name, and no clash with an existing name that differs
    only by case (a case-insensitive file system would serve the other one)."""
    if not config.valid_id(name):
        raise ConfigError(f"{what} name must be an id [A-Za-z0-9_.-] that is a portable folder "
                          "name: not '.' / '..', no trailing dot, not a Windows device name "
                          f"(CON, NUL, COM1 ...), got {name!r}")
    if name.lower().endswith(FILE_SUFFIXES):
        raise ConfigError(f"{what} name {name!r} reads as a file name: drop the "
                          f"{Path(name).suffix} ending")
    if name == Path(ACTIVE_FILE).stem:
        raise ConfigError(f"{name!r} is reserved")
    fold = name.casefold()
    for b in taken_builtin:
        if b == name:
            raise ConfigError(f"{name!r} is a built-in {what} (read-only): pick another name")
        if b.casefold() == fold:
            raise ConfigError(f"{name!r} differs from the built-in {what} {b!r} only by case: "
                              "pick another name")
    for u in taken_user:
        if u != name and u.casefold() == fold:
            raise ConfigError(f"{name!r} differs from the user {what} {u!r} only by case: pick "
                              f"another name (or replace {u!r})")


def _raw(store: LibraryStore, builtin: Path, name: str) -> tuple:
    """(dict, tier) — the preset JSON as written."""
    tier = tier_of(store, builtin, name)
    if tier is None:
        raise ConfigError(f"no preset named {name!r} (built-in in {builtin} or user in "
                          f"{user_dir(store)})")
    if tier == "builtin" and name in user_names(store):
        config.resolve_preset_path(name, builtin, user_dir(store))      # raises: both tiers
    path = (Path(builtin) / f"{name}.json" if tier == "builtin"
            else user_dir(store) / name / config.USER_PRESET_FILE)
    return json.loads(path.read_text(encoding="utf-8")), tier


def _no_constant(token):
    raise ValueError(f"{token} is not a number here")


def require_finite(v, key: str):
    """Refuse NaN / Infinity anywhere in an override value (NaN fails every comparison, so a
    NaN tolerance would never FAIL anything; Infinity never trips either)."""
    if isinstance(v, float) and not math.isfinite(v):
        raise ConfigError(f"override {key}: {v!r} is not a finite number")
    if isinstance(v, dict):
        for x in v.values():
            require_finite(x, key)
    elif isinstance(v, (list, tuple)):
        for x in v:
            require_finite(x, key)
    return v


def _parse_value(v):
    """--set key=value: JSON when it parses (numbers, true/false, lists, objects), else text.
    The non-JSON tokens NaN / Infinity / -Infinity are refused, not read as numbers."""
    if not isinstance(v, str):
        return v
    try:
        return json.loads(v, parse_constant=_no_constant)
    except ValueError:
        if v.strip().lstrip("+-").lower() in ("nan", "inf", "infinity"):
            raise ConfigError(f"{v!r} is not a finite number") from None
        return v


def apply_overrides(d: dict, overrides: Optional[dict]) -> dict:
    """Override preset fields: {"camera.exposure_us": 33333, "projection": {"duty": 0.8}}.
    Dotted keys walk into sections; nested dicts merge. Unknown keys are not
    dropped here: the preset loader refuses them."""
    out = copy.deepcopy(d)
    for key, val in (overrides or {}).items():
        if not isinstance(key, str) or not key:
            raise ConfigError(f"override key must be a non-empty string, got {key!r}")
        parts = key.split(".")
        if parts[0] in ("name", "geometry_id"):
            raise ConfigError(f"{parts[0]} is set by the command (name / --geometry), "
                              "not by an override")
        try:
            val = require_finite(_parse_value(val), key)
        except ConfigError as e:
            raise ConfigError(f"override {key}: {e}" if not str(e).startswith("override")
                              else str(e)) from None
        node = out
        for p in parts[:-1]:
            nxt = node.get(p)
            if nxt is None:
                nxt = node[p] = {}
            if not isinstance(nxt, dict):
                raise ConfigError(f"override {key}: {p} is not a section")
            node = nxt
        last = parts[-1]
        if isinstance(val, dict) and isinstance(node.get(last), dict):
            node[last] = {**node[last], **val}
        else:
            node[last] = val
    return out


def _all_preset_files(store: LibraryStore, builtin: Path, skip_user: Optional[str] = None) -> list:
    files = sorted(Path(builtin).glob("*.json"))
    files += [user_dir(store) / n / config.USER_PRESET_FILE for n in user_names(store)
              if n != skip_user]
    return files


def add(store: LibraryStore, builtin: Path, name: str, base: str,
        overrides: Optional[dict] = None, geometry_id: Optional[str] = None,
        description: str = "", replace: bool = False, author: Optional[str] = None) -> dict:
    users = user_names(store)
    check_new_name(name, builtin_names(builtin), users, "preset", replace)
    exists = name in users
    if exists and not replace:
        raise ConfigError(f"user preset {name!r} exists: pick another name, or replace it "
                          "(the old one goes to trash/)")
    src, src_tier = _raw(store, builtin, base)
    d = apply_overrides(src, overrides)
    d["name"] = name
    if geometry_id is not None:
        if not config.valid_id(geometry_id):
            raise ConfigError("geometry_id must be a non-empty, portable id [A-Za-z0-9_.-] (no "
                              "trailing dot, not a Windows device name)")
        d["geometry_id"] = geometry_id
        if geometry_id != src.get("geometry_id"):
            d["calibration"] = {"id": None, "mm_per_px": None, "intercept_mm": 0.0,
                                "lateral_px_per_mm": None}  # never carried to a new rig
    new = config.preset_from_dict(d)                         # same validation as load_preset
    # the version being replaced is checked too (as config.save_preset(overwrite=True)): a
    # replace keeping the geometry_id must keep its geometry-bound settings, else the bases
    # and calibrations stored under that id (and the preset's own calibration) would be reused
    config._refuse_geometry_reuse(new, Path(builtin), _all_preset_files(store, builtin))
    folder = user_dir(store)
    folder.mkdir(parents=True, exist_ok=True)
    entry = folder / name
    meta = {"name": name, "created": datetime.now().isoformat(timespec="seconds"),
            "author": author if author is not None else _user(),
            "description": str(description or ""), "from": base, "from_tier": src_tier,
            "geometry_id": d["geometry_id"], "overrides": overrides or {}, "replaced": None}
    if not exists and entry.exists():
        raise ConfigError(f"{entry} exists but is not a complete user preset: move it away first")
    # the new preset is written under a temporary name (never a valid id, so never listed)
    # BEFORE the old one is touched, and the old one comes back if finishing the new one fails
    tmp = folder / f"~{name}.{uuid.uuid4().hex[:8]}"
    tmp.mkdir()
    with store.filling(tmp):
        with open(tmp / config.USER_PRESET_FILE, "x", encoding="utf-8") as f:
            json.dump(d, f, indent=2)
    trashed = store.trash(entry, reason=f"replaced by preset_add {name}") if exists else None
    meta["replaced"] = str(trashed) if trashed else None
    try:
        with store.filling(tmp):
            LibraryStore.write_json(tmp / META_FILE, meta)   # meta last: a complete entry
            tmp.rename(entry)                                # never onto a live folder
    except BaseException:
        if trashed is not None and not entry.exists():
            try:
                Path(trashed).rename(entry)                  # the old version back in place
            except OSError:
                _log.error("preset %s: the replaced version stays in %s", name, trashed)
        raise
    cal_note = None
    if geometry_id is not None and geometry_id != src.get("geometry_id"):
        cal_note = (f"new geometry_id {geometry_id} starts UNCALIBRATED: no bases and no "
                    "calibration exist for it - run calibrate-stereo (metric) or calibrate "
                    "(relief) with this preset before measuring in mm")
    return {"name": name, "tier": "user", "path": entry / config.USER_PRESET_FILE,
            "meta": meta, "warnings": list(new.warnings), "note": cal_note,
            "replaced": trashed}


def _user() -> str:
    try:
        return getpass.getuser()
    except Exception:                       # noqa: BLE001 - no user name: not an error
        return ""


def last_selected(path: Path) -> tuple:
    """An append-only selection history (active.jsonl, last line wins) -> (the last recorded
    name or None, lines skipped). A line that is not UTF-8 JSON with a text name (another
    program's encoding, a torn write) is skipped with a warning; it never stops the reader,
    so the CLI can still list and select."""
    try:
        data = Path(path).read_bytes()
    except FileNotFoundError:
        return None, 0
    name, skipped = None, 0
    for raw in data.splitlines():
        if not raw.strip():
            continue
        try:
            n = json.loads(raw.decode("utf-8")).get("name")
        except (UnicodeDecodeError, ValueError, AttributeError):
            skipped += 1
            continue
        if isinstance(n, str) and n:
            name = n
    if skipped:
        _log.warning("%s: skipped %d unreadable line(s) (not UTF-8 JSON)", path, skipped)
    return name, skipped


def append_selection(path: Path, rec: dict) -> None:
    """One line onto the selection history; a torn last line stays a line of its own."""
    path = Path(path)
    lead = ""
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            if f.tell():
                f.seek(-1, 2)
                lead = "" if f.read(1) == b"\n" else "\n"
    except FileNotFoundError:
        pass
    with open(path, "a", encoding="utf-8") as f:           # append-only history
        f.write(lead + json.dumps(rec) + "\n")


def select(store: LibraryStore, builtin: Path, name: str, by: str = "") -> dict:
    tier = tier_of(store, builtin, name)
    if tier is None:
        raise ConfigError(f"no preset named {name!r}")
    raw, _ = _raw(store, builtin, name)
    config.preset_from_dict(raw)            # an invalid preset can't become the default
    folder = user_dir(store)
    folder.mkdir(parents=True, exist_ok=True)
    rec = {"name": name, "tier": tier, "time": datetime.now().isoformat(timespec="seconds"),
           "by": by or _user()}
    append_selection(folder / ACTIVE_FILE, rec)
    return rec


def selection(store: LibraryStore, builtin: Path) -> dict:
    """{"name": the last selected name or None, "exists": it still resolves, "skipped":
    unreadable history lines}. Tells 'never selected' from 'selected but gone' (deleted,
    renamed, a failed replace): the latter must stop a scan, not fall back to a default."""
    name, skipped = last_selected(user_dir(store) / ACTIVE_FILE)
    return {"name": name, "exists": bool(name) and tier_of(store, builtin, name) is not None,
            "skipped": skipped}


def active(store: LibraryStore, builtin: Path) -> Optional[str]:
    """The selected preset, or None (never selected, or it no longer exists: selection()
    tells the two apart)."""
    sel = selection(store, builtin)
    return sel["name"] if sel["exists"] else None


def read_or_text(path: Path) -> tuple:
    """A station config / recipe file -> (its mapping, None), read exactly; or (None, its text)
    for a YAML file that can't be parsed here (PyYAML is optional, or not valid YAML): the
    station may run where PyYAML is, so it is judged by what its text mentions (mentions(),
    file_refs()) - an over-match only asks for --force, a miss would delete what it runs.
    Raises for an unreadable or missing file, or a JSON file that is not a mapping."""
    from ..recipes import schema
    try:
        return schema.read_doc(path), None
    except schema.RecipeError:
        if Path(path).suffix.lower() in (".yaml", ".yml") and Path(path).is_file():
            return None, Path(path).read_text(encoding="utf-8", errors="replace")
        raise


def mentions(text: str, word: str) -> bool:
    """`word` anywhere in text as a name of its own (not inside a longer identifier), any
    case: a fail-closed reading of a YAML file that can't be parsed."""
    return bool(word) and re.search(rf"(?<![A-Za-z0-9_]){re.escape(word)}(?![A-Za-z0-9_])",
                                    text, re.I) is not None


def file_refs(text: str, base_dir: Path) -> list:
    """Every recipe-file path a YAML text mentions, resolved as run_job resolves job.recipe
    ($VARS, ~, relative to the config's folder)."""
    out = []
    for rx in _FILE_REFS:
        for m in rx.finditer(text):
            v = m.group(m.lastindex).strip()
            p = Path(os.path.expandvars(os.path.expanduser(v)))
            p = p if p.is_absolute() else Path(base_dir) / p
            if p not in out:
                out.append(p)
    return out


def _pins(path: Path, name: str) -> bool:
    """The recipe file at `path` pins preset `name` (a YAML one that can't be parsed here:
    when its text mentions the name)."""
    doc, text = read_or_text(path)
    return mentions(text, name) if doc is None else doc.get("preset") == name


def station_refs(stations_dir: Optional[Path], name: str) -> list:
    """Station configs (stations/*.json|yaml) that pin preset `name`: an old single-job
    config's `preset`, a configuration's `parameters.preset` or one of its profiles' `preset`
    (a YAML one that can't be parsed here: whose text mentions it)."""
    from ..flow.schema import references
    d = Path(stations_dir) if stations_dir else None
    if d is None or not d.is_dir():
        return []
    out = []
    for f in sorted(d.iterdir()):
        if f.suffix.lower() not in FILE_SUFFIXES:
            continue
        try:
            doc, text = read_or_text(f)
        except Exception:                       # noqa: BLE001 - not a station config we can read
            continue
        if (mentions(text, name) if doc is None else name in references(doc)["presets"]):
            out.append(f.name)
    return out


def recipe_refs(store: LibraryStore, name: str, recipes_dir: Optional[Path] = None,
                stations_dir: Optional[Path] = None) -> dict:
    """What pins preset `name` through a recipe: {"recipes": built-in + user recipes whose
    preset it is, "selected": the selected recipe if it is one of them, "stations": station
    configs whose recipe job (job.recipe: a name or a recipe file) or whose profiles'
    recipes run one of them with no preset of their own to override it}.
    job.recipe is read as run_job reads it (stripped; a file: $VARS, ~, relative to the
    config). A YAML file that can't be parsed here is never skipped: a recipe pins `name`
    when its text mentions it, a station runs one when its text mentions a pinning recipe's
    name or a recipe file that pins it (its own preset is not trusted to a pattern)."""
    from ..flow.schema import references
    from ..recipes import library as recipe_lib
    builtin = Path(recipes_dir) if recipes_dir else recipe_lib.BUILTIN_DIR
    pinned = []
    for rname in recipe_lib.builtin_names(builtin) + recipe_lib.user_names(store):
        try:
            if _pins(recipe_lib.path_of(store, builtin, rname), name):
                pinned.append(rname)
        except Exception:                       # noqa: BLE001 - unreadable: listed as invalid
            continue
    sel = recipe_lib.active(store, builtin)
    stations = []
    d = Path(stations_dir) if stations_dir else None
    if d is not None and d.is_dir():
        for f in sorted(d.iterdir()):
            if f.suffix.lower() not in (".json", ".yaml", ".yml"):
                continue
            try:
                doc, text = read_or_text(f)
            except Exception:                   # noqa: BLE001 - not a station config we can read
                continue
            if doc is None:                     # YAML that can't be parsed here: fail closed
                if any(mentions(text, r) for r in pinned) or any(
                        _pins_quietly(rp, name) for rp in file_refs(text, f.resolve().parent)):
                    stations.append(f.name)
                continue
            for r in references(doc)["recipes"]:    # a single job's recipe, or a profile's
                ref = r["ref"]
                if r["preset"]:
                    continue                    # the station's own preset wins over the recipe's
                if isinstance(ref, dict):       # a profile's inline recipe mapping
                    hit = ref.get("preset") == name
                elif ref in pinned:
                    hit = True
                elif ref.lower().endswith((".json", ".yaml", ".yml")):
                    rp = Path(os.path.expandvars(os.path.expanduser(ref)))   # as run_job does
                    hit = _pins_quietly(rp if rp.is_absolute() else f.resolve().parent / rp,
                                        name)
                else:
                    hit = False
                if hit:
                    stations.append(f.name)
                    break
    return {"recipes": pinned, "selected": sel if sel in pinned else None, "stations": stations}


def _pins_quietly(path: Path, name: str) -> bool:
    try:
        return _pins(path, name)
    except Exception:                           # noqa: BLE001 - a missing recipe file pins nothing
        return False


def delete(store: LibraryStore, builtin: Path, name: str, reason: str,
           stations_dir: Optional[Path] = None, force: bool = False,
           recipes_dir: Optional[Path] = None) -> dict:
    if not reason or not str(reason).strip():
        raise ConfigError("preset_delete needs a reason (recorded with the trashed preset)")
    tier = tier_of(store, builtin, name)
    if tier == "builtin" and name not in user_names(store):
        raise ConfigError(f"{name!r} is a built-in preset: read-only, it can't be deleted")
    if tier is None:
        raise ConfigError(f"no user preset named {name!r}")
    # a user preset that a later built-in of the same name shadows: the user one is the
    # only one a delete can mean (built-ins are never deleted)
    if active(store, builtin) == name:
        raise ConfigError(f"{name!r} is the active preset: select another preset first")
    rec = recipe_refs(store, name, recipes_dir, stations_dir)
    if rec["selected"]:
        raise ConfigError(f"{name!r} is the preset of the selected recipe {rec['selected']!r}: "
                          "select another recipe first")
    refs = station_refs(stations_dir, name)
    if (refs or rec["recipes"] or rec["stations"]) and not force:
        used = [f"station config(s) {', '.join(refs)}"] if refs else []
        used += [f"recipe(s) {', '.join(rec['recipes'])}"] if rec["recipes"] else []
        used += ([f"recipe station config(s) {', '.join(rec['stations'])}"]
                 if rec["stations"] else [])
        raise ConfigError(f"{name!r} is used by {'; '.join(used)}: change them first, or force "
                          "the delete")
    dest = store.trash(user_dir(store) / name, reason=str(reason))
    return {"name": name, "trashed": dest, "stations": refs, "recipes": rec["recipes"],
            "recipe_stations": rec["stations"],
            "forced": bool((refs or rec["recipes"] or rec["stations"]) and force)}


def stereo_calibrated(store: LibraryStore, geometry_id: str) -> bool:
    return any((d / "stereo_calibration.json").is_file()
               for d in store.entries("calibrations", geometry_id))


def listing(store: LibraryStore, builtin: Path, mock: bool = False) -> list:
    act = active(store, builtin)
    out = []
    for tier, names in (("builtin", builtin_names(builtin)), ("user", user_names(store))):
        for n in names:
            item = {"name": n, "tier": tier, "active": n == act}
            try:
                raw, _ = _raw(store, builtin, n)
                geo = raw.get("geometry_id")
                item["geometry_id"] = geo
                eff = (config.MOCK_PREFIX + geo) if (mock and geo) else geo
                item["calibrated"] = bool(geo) and stereo_calibrated(store, eff)
                config.preset_from_dict(raw, mock=mock)
                item["valid"] = True
            except Exception as e:          # noqa: BLE001 - listed as unusable
                item.update(valid=False, error=f"{type(e).__name__}: {e}")
            meta = {}
            if tier == "user":
                try:
                    meta = json.loads((user_dir(store) / n / META_FILE).read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    meta = {}
            item.update(description=meta.get("description", ""), created=meta.get("created"),
                        author=meta.get("author"), base=meta.get("from"))
            out.append(item)
    return out


def show(store: LibraryStore, builtin: Path, name: str, mock: bool = False) -> dict:
    raw, tier = _raw(store, builtin, name)
    item = next((i for i in listing(store, builtin, mock) if i["name"] == name), {})
    return {**item, "tier": tier, "preset": raw}

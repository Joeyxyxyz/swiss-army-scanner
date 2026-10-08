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
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from .. import config
from ..config import ConfigError
from .store import LibraryStore

ACTIVE_FILE = "active.jsonl"
META_FILE = "meta.json"
_YAML_PRESET = re.compile(r"^preset:\s*['\"]?([A-Za-z0-9_.\-]+)", re.M)


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
    if name in builtin_names(builtin):
        return "builtin"
    if name in user_names(store):
        return "user"
    return None


def _raw(store: LibraryStore, builtin: Path, name: str) -> tuple:
    """(dict, tier) — the preset JSON as written."""
    tier = tier_of(store, builtin, name)
    if tier is None:
        raise ConfigError(f"no preset named {name!r} (built-in in {builtin} or user in "
                          f"{user_dir(store)})")
    path = (Path(builtin) / f"{name}.json" if tier == "builtin"
            else user_dir(store) / name / config.USER_PRESET_FILE)
    return json.loads(path.read_text(encoding="utf-8")), tier


def _parse_value(v):
    """--set key=value: JSON when it parses (numbers, true/false, lists, objects), else text."""
    if not isinstance(v, str):
        return v
    try:
        return json.loads(v)
    except ValueError:
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
        val = _parse_value(val)
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
    if not config.valid_id(name):
        raise ConfigError("preset name must be a non-empty id [A-Za-z0-9_.-] (not '.' or '..')")
    if name == Path(ACTIVE_FILE).stem:
        raise ConfigError(f"{name!r} is reserved")
    if name in builtin_names(builtin):
        raise ConfigError(f"{name!r} is a built-in preset (read-only): pick another name")
    exists = name in user_names(store)
    if exists and not replace:
        raise ConfigError(f"user preset {name!r} exists: pick another name, or replace it "
                          "(the old one goes to trash/)")
    src, src_tier = _raw(store, builtin, base)
    d = apply_overrides(src, overrides)
    d["name"] = name
    if geometry_id is not None:
        if not config.valid_id(geometry_id):
            raise ConfigError("geometry_id must be a non-empty id [A-Za-z0-9_.-]")
        d["geometry_id"] = geometry_id
        if geometry_id != src.get("geometry_id"):
            d["calibration"] = {"id": None, "mm_per_px": None, "intercept_mm": 0.0,
                                "lateral_px_per_mm": None}  # never carried to a new rig
    new = config.preset_from_dict(d)                         # same validation as load_preset
    config._refuse_geometry_reuse(new, Path(builtin),
                                  _all_preset_files(store, builtin, skip_user=name))
    folder = user_dir(store)
    folder.mkdir(parents=True, exist_ok=True)
    entry = folder / name
    trashed = None
    if exists:
        trashed = store.trash(entry, reason=f"replaced by preset_add {name}")
    entry.mkdir()                                            # never reuse a live folder
    meta = {"name": name, "created": datetime.now().isoformat(timespec="seconds"),
            "author": author if author is not None else _user(),
            "description": str(description or ""), "from": base, "from_tier": src_tier,
            "geometry_id": d["geometry_id"], "overrides": overrides or {},
            "replaced": str(trashed) if trashed else None}
    with store.filling(entry):
        with open(entry / config.USER_PRESET_FILE, "x", encoding="utf-8") as f:
            json.dump(d, f, indent=2)
        LibraryStore.write_json(entry / META_FILE, meta)     # meta last: a complete entry
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
    with open(folder / ACTIVE_FILE, "a", encoding="utf-8") as f:     # append-only history
        f.write(json.dumps(rec) + "\n")
    return rec


def active(store: LibraryStore, builtin: Path) -> Optional[str]:
    """The selected preset, or None (never selected, or it no longer exists)."""
    f = user_dir(store) / ACTIVE_FILE
    if not f.is_file():
        return None
    name = None
    for line in f.read_text(encoding="utf-8").splitlines():
        try:
            name = json.loads(line).get("name") or name
        except (ValueError, AttributeError):
            continue
    return name if name and tier_of(store, builtin, name) else None


def station_refs(stations_dir: Optional[Path], name: str) -> list:
    """Station configs (stations/*.json|yaml) whose preset is `name`."""
    d = Path(stations_dir) if stations_dir else None
    if d is None or not d.is_dir():
        return []
    out = []
    for f in sorted(d.iterdir()):
        if f.suffix.lower() == ".json":
            try:
                p = json.loads(f.read_text(encoding="utf-8")).get("preset")
            except (ValueError, AttributeError, OSError):
                continue
        elif f.suffix.lower() in (".yaml", ".yml"):
            m = _YAML_PRESET.search(f.read_text(encoding="utf-8", errors="replace"))
            p = m.group(1) if m else None
        else:
            continue
        if p == name:
            out.append(f.name)
    return out


def delete(store: LibraryStore, builtin: Path, name: str, reason: str,
           stations_dir: Optional[Path] = None, force: bool = False) -> dict:
    if not reason or not str(reason).strip():
        raise ConfigError("preset_delete needs a reason (recorded with the trashed preset)")
    tier = tier_of(store, builtin, name)
    if tier == "builtin":
        raise ConfigError(f"{name!r} is a built-in preset: read-only, it can't be deleted")
    if tier is None:
        raise ConfigError(f"no user preset named {name!r}")
    if active(store, builtin) == name:
        raise ConfigError(f"{name!r} is the active preset: select another preset first")
    refs = station_refs(stations_dir, name)
    if refs and not force:
        raise ConfigError(f"{name!r} is used by station config(s) {', '.join(refs)}: change "
                          "them first, or force the delete")
    dest = store.trash(user_dir(store) / name, reason=str(reason))
    return {"name": name, "trashed": dest, "stations": refs, "forced": bool(refs and force)}


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

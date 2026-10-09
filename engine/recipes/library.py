"""
Recipe library: built-in recipes (the repo's recipes/ folder, READ-ONLY here) and user
recipes an operator or integrator adds, selects and deletes per library - the same two
tiers as the presets (engine/library/presets.py):

    <library>/recipes/<name>/recipe.json    the recipe (validated like any other)
    <library>/recipes/<name>/meta.json      created, author, description, copied from
    <library>/recipes/active.jsonl          the selected recipe; append-only, last line wins

A user recipe is a copy of a built-in / user recipe (or a recipe file) with fields
overridden. Nothing is replaced in place: `replace` moves the old one to trash/ first, and
delete moves it to trash/recipes/ (recoverable).
"""

from __future__ import annotations

import copy
import getpass
import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..config import ConfigError
from ..library import presets as _presets
from ..library.store import LibraryStore
from . import schema
from .schema import RecipeError

BUILTIN_DIR = Path(__file__).resolve().parents[2] / "recipes"
USER_DIR = "recipes"
RECIPE_FILE = "recipe.json"
META_FILE = "meta.json"
ACTIVE_FILE = "active.jsonl"
EXTS = (".json", ".yaml", ".yml")
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def user_dir(store: LibraryStore) -> Path:
    return Path(store.root) / USER_DIR


def _builtin_files(builtin: Path) -> dict:
    d = Path(builtin)
    if not d.is_dir():
        return {}
    out = {}
    for f in sorted(d.iterdir()):
        if f.suffix.lower() in EXTS and f.is_file() and _ID.match(f.stem):
            out.setdefault(f.stem, f)                 # .json wins over .yaml of the same name
    return out


def builtin_names(builtin: Path) -> list:
    return sorted(_builtin_files(builtin))


def user_names(store: LibraryStore) -> list:
    d = user_dir(store)
    if not d.is_dir():
        return []
    return sorted(e.name for e in d.iterdir()
                  if e.is_dir() and _ID.match(e.name) and (e / RECIPE_FILE).is_file()
                  and (e / META_FILE).is_file())


def tier_of(store: LibraryStore, builtin: Path, name: str) -> Optional[str]:
    """'builtin' | 'user' | None, by EXACT name (a file path is never a library recipe)."""
    if not isinstance(name, str):
        return None
    if name in builtin_names(builtin):
        return "builtin"
    if name in user_names(store):
        return "user"
    return None


def path_of(store: LibraryStore, builtin: Path, name: str) -> Path:
    tier = tier_of(store, builtin, name)
    if tier is None:
        raise RecipeError(f"no recipe named {name!r} (built-in in {builtin} or user in "
                          f"{user_dir(store)}; see `scan recipe-list`)")
    return _builtin_files(builtin)[name] if tier == "builtin" else \
        user_dir(store) / name / RECIPE_FILE


def raw(store: LibraryStore, builtin: Path, name: str) -> tuple:
    """(raw dict, tier, path) of a named recipe as written."""
    p = path_of(store, builtin, name)
    return schema.read_doc(p), tier_of(store, builtin, name), p


def resolve(store: LibraryStore, builtin: Path, ref) -> schema.Recipe:
    """A recipe from a name (built-in / user), a .json / .yaml file, a raw dict or a Recipe."""
    if isinstance(ref, schema.Recipe):
        return ref
    if isinstance(ref, dict):
        return schema.validate(ref)
    s = str(ref)
    p = Path(s)
    if p.suffix.lower() in EXTS and (p.is_file() or not _ID.match(s)):
        return schema.load(p)
    doc, _tier, path = raw(store, builtin, s)
    r = schema.validate(doc, source=path)
    if r.name != s:
        raise RecipeError(f"recipe file {path.name} names itself {r.name!r}: the name inside must "
                          f"be the file / folder name {s!r}")
    return r


def _parse_value(v):
    """--set key=value: JSON when it parses, else text; NaN / Infinity are refused."""
    try:
        return _presets.require_finite(_presets._parse_value(v), "value")
    except ConfigError as e:
        raise RecipeError(str(e)) from None


def apply_overrides(d: dict, overrides: Optional[dict]) -> dict:
    """Override recipe fields: {"feedback.hold_s": 10, "zones.0.checks.1.params.tol_mm": 0.4}.
    Dotted keys walk into sections, numbers index lists; nested dicts merge."""
    out = copy.deepcopy(d)
    for key, val in (overrides or {}).items():
        if not isinstance(key, str) or not key:
            raise RecipeError(f"override key must be a non-empty string, got {key!r}")
        parts = key.split(".")
        if parts[0] == "name":
            raise RecipeError("name is set by the command, not by an override")
        val = _parse_value(val)
        node = out
        for k in parts[:-1]:
            node = _child(node, k, key)
        last = parts[-1]
        if isinstance(node, list):
            i = _index(node, last, key)
            node[i] = val
        elif isinstance(val, dict) and isinstance(node.get(last), dict):
            node[last] = {**node[last], **val}
        else:
            node[last] = val
    return out


def _index(node: list, k: str, key: str) -> int:
    if not k.isdigit() or int(k) >= len(node):
        raise RecipeError(f"override {key}: {k!r} is not an index of a {len(node)}-item list")
    return int(k)


def _child(node, k: str, key: str):
    if isinstance(node, list):
        return node[_index(node, k, key)]
    if not isinstance(node, dict):
        raise RecipeError(f"override {key}: {k} is not a section")
    nxt = node.get(k)
    if nxt is None:
        nxt = node[k] = {}
    if not isinstance(nxt, (dict, list)):
        raise RecipeError(f"override {key}: {k} is not a section")
    return nxt


def _user() -> str:
    try:
        return getpass.getuser()
    except Exception:                       # noqa: BLE001 - no user name: not an error
        return ""


def add(store: LibraryStore, builtin: Path, name: str, base=None, overrides: Optional[dict] = None,
        description: str = "", replace: bool = False, author: Optional[str] = None,
        recipe: Optional[dict] = None, preset_names: Optional[list] = None) -> dict:
    """A new USER recipe: a copy of `base` (a recipe name or file) or the mapping `recipe`,
    with fields overridden, validated like any recipe (+ its preset must exist)."""
    if not isinstance(name, str) or not _ID.fullmatch(name):
        raise RecipeError("recipe name must be an id [A-Za-z0-9_.-] (max 64)")
    users = user_names(store)
    try:
        _presets.check_new_name(name, builtin_names(builtin), users, "recipe", replace)
    except ConfigError as e:
        raise RecipeError(str(e)) from None
    exists = name in users
    if exists and not replace:
        raise RecipeError(f"user recipe {name!r} exists: pick another name, or replace it (the "
                          "old one goes to trash/)")
    if (base is None) == (recipe is None):
        raise RecipeError("give a base recipe to copy (name or file) or a recipe mapping")
    if recipe is not None:
        src, src_tier, src_name = copy.deepcopy(recipe), "mapping", None
    else:
        p = Path(str(base))
        if p.suffix.lower() in EXTS and p.is_file():
            src, src_tier, src_name = schema.read_doc(p), "file", str(p)
        else:
            src, src_tier, _ = raw(store, builtin, str(base))
            src_name = str(base)
    d = apply_overrides(src, overrides)
    d["name"] = name
    r = schema.validate(d)
    if preset_names is not None and r.preset not in preset_names:
        raise RecipeError(f"recipe {name}: preset {r.preset!r} does not exist (built-in or user "
                          "presets: `scan presets`)")
    folder = user_dir(store)
    folder.mkdir(parents=True, exist_ok=True)
    entry = folder / name
    if not exists and entry.exists():
        raise RecipeError(f"{entry} exists but is not a complete user recipe: move it away first")
    meta = {"name": name, "created": datetime.now().isoformat(timespec="seconds"),
            "author": author if author is not None else _user(),
            "description": str(description or r.description or ""), "from": src_name,
            "from_tier": src_tier, "preset": r.preset, "overrides": overrides or {},
            "replaced": None}
    # written under a temporary name (never an id, so never listed) BEFORE the old one is
    # touched; the old one comes back if finishing the new one fails
    tmp = folder / f"~{name}.{uuid.uuid4().hex[:8]}"
    tmp.mkdir()
    with store.filling(tmp):
        with open(tmp / RECIPE_FILE, "x", encoding="utf-8") as f:
            json.dump(d, f, indent=2)
    trashed = store.trash(entry, reason=f"replaced by recipe_add {name}") if exists else None
    meta["replaced"] = str(trashed) if trashed else None
    try:
        with store.filling(tmp):
            LibraryStore.write_json(tmp / META_FILE, meta)   # meta last: a complete entry
            tmp.rename(entry)
    except BaseException:
        if trashed is not None and not entry.exists():
            try:
                Path(trashed).rename(entry)
            except OSError:
                pass
        raise
    return {"name": name, "tier": "user", "path": entry / RECIPE_FILE, "meta": meta,
            "replaced": trashed, "recipe": r.summary()}


def select(store: LibraryStore, builtin: Path, name: str, by: str = "") -> dict:
    tier = tier_of(store, builtin, name)
    if tier is None:
        raise RecipeError(f"no recipe named {name!r}: only a built-in or library recipe can be "
                          "selected (see `scan recipe-list`); for a recipe file, add it first "
                          "(scan recipe-add NAME --from <file>)")
    resolve(store, builtin, name)            # an invalid recipe can't become the selected one
    folder = user_dir(store)
    folder.mkdir(parents=True, exist_ok=True)
    rec = {"name": name, "tier": tier,
           "time": datetime.now().isoformat(timespec="seconds"), "by": by or _user()}
    _presets.append_selection(folder / ACTIVE_FILE, rec)
    return rec


def active(store: LibraryStore, builtin: Path) -> Optional[str]:
    """The selected recipe, or None (never selected, or it no longer exists). Unreadable
    history lines are skipped (with a warning), never fatal."""
    name, _skipped = _presets.last_selected(user_dir(store) / ACTIVE_FILE)
    return name if name and tier_of(store, builtin, name) else None


def station_refs(stations_dir: Optional[Path], name: str) -> list:
    """Station configs (stations/*.json|yaml) whose job.recipe is `name`, read as run_job
    reads it (stripped); a YAML one that can't be parsed here (PyYAML is optional): whose
    text mentions the name - never skipped, never guessed at by a pattern for its keys."""
    d = Path(stations_dir) if stations_dir else None
    if d is None or not d.is_dir():
        return []
    out = []
    for f in sorted(d.iterdir()):
        if f.suffix.lower() not in EXTS:
            continue
        try:
            doc, text = _presets.read_or_text(f)
        except Exception:                       # noqa: BLE001 - not a station config we can read
            continue
        if doc is None:
            hit = _presets.mentions(text, name)
        else:
            job = doc.get("job")
            r = job.get("recipe") if isinstance(job, dict) else None
            hit = isinstance(r, str) and r.strip() == name      # as run_job reads job.recipe
        if hit:
            out.append(f.name)
    return out


def delete(store: LibraryStore, builtin: Path, name: str, reason: str,
           stations_dir: Optional[Path] = None, force: bool = False) -> dict:
    if not reason or not str(reason).strip():
        raise RecipeError("recipe_delete needs a reason (recorded with the trashed recipe)")
    tier = tier_of(store, builtin, name)
    if tier == "builtin":
        raise RecipeError(f"{name!r} is a built-in recipe: read-only, it can't be deleted")
    if tier is None:
        raise RecipeError(f"no user recipe named {name!r}")
    if active(store, builtin) == name:
        raise RecipeError(f"{name!r} is the selected recipe: select another recipe first")
    refs = station_refs(stations_dir, name)
    if refs and not force:
        raise RecipeError(f"{name!r} is used by station config(s) {', '.join(refs)}: change them "
                          "first, or force the delete")
    dest = store.trash(user_dir(store) / name, reason=str(reason))
    return {"name": name, "trashed": dest, "stations": refs, "forced": bool(refs and force)}


def listing(store: LibraryStore, builtin: Path, preset_names: Optional[list] = None) -> list:
    act = active(store, builtin)
    out = []
    for tier, names in (("builtin", builtin_names(builtin)), ("user", user_names(store))):
        for n in names:
            item = {"name": n, "tier": tier, "active": n == act}
            try:
                r = resolve(store, builtin, n)
                item.update(preset=r.preset, description=r.description,
                            zones=[z.id for z in r.zones],
                            checks=sorted({c.type for z in r.zones for c in z.checks}),
                            scan=r.scan["kind"], golden=(r.golden or {}).get("scan"),
                            mock_scene=r.part.get("mock_scene"), valid=True)
                if preset_names is not None and r.preset not in preset_names:
                    item.update(valid=False, error=f"preset {r.preset!r} does not exist")
            except Exception as e:          # noqa: BLE001 - listed as unusable
                item.update(valid=False, error=f"{type(e).__name__}: {e}")
            if tier == "user":
                try:
                    meta = json.loads((user_dir(store) / n / META_FILE).read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    meta = {}
                item.update(created=meta.get("created"), author=meta.get("author"),
                            base=meta.get("from"))
                if meta.get("description"):
                    item["description"] = meta["description"]
            out.append(item)
    return out


def show(store: LibraryStore, builtin: Path, name: str, preset_names: Optional[list] = None) -> dict:
    doc, tier, path = raw(store, builtin, name)
    item = next((i for i in listing(store, builtin, preset_names) if i["name"] == name), {})
    out = {**item, "tier": tier, "path": str(path), "recipe": doc}
    try:
        out["normalised"] = resolve(store, builtin, name).to_dict()
    except Exception:                       # noqa: BLE001 - the listing carries the error
        pass
    return out

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
from datetime import datetime
from pathlib import Path
from typing import Optional

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
_YAML_RECIPE = re.compile(r"^\s*recipe:\s*['\"]?([A-Za-z0-9_.\-]+)", re.M)


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
    if not isinstance(v, str):
        return v
    try:
        return json.loads(v)
    except ValueError:
        return v


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
    if not isinstance(name, str) or not _ID.match(name):
        raise RecipeError("recipe name must be an id [A-Za-z0-9_.-] (max 64)")
    if name == Path(ACTIVE_FILE).stem:
        raise RecipeError(f"{name!r} is reserved")
    if name in builtin_names(builtin):
        raise RecipeError(f"{name!r} is a built-in recipe (read-only): pick another name")
    exists = name in user_names(store)
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
    trashed = None
    if exists:
        trashed = store.trash(entry, reason=f"replaced by recipe_add {name}")
    entry.mkdir()
    meta = {"name": name, "created": datetime.now().isoformat(timespec="seconds"),
            "author": author if author is not None else _user(),
            "description": str(description or r.description or ""), "from": src_name,
            "from_tier": src_tier, "preset": r.preset, "overrides": overrides or {},
            "replaced": str(trashed) if trashed else None}
    with store.filling(entry):
        with open(entry / RECIPE_FILE, "x", encoding="utf-8") as f:
            json.dump(d, f, indent=2)
        LibraryStore.write_json(entry / META_FILE, meta)     # meta last: a complete entry
    return {"name": name, "tier": "user", "path": entry / RECIPE_FILE, "meta": meta,
            "replaced": trashed, "recipe": r.summary()}


def select(store: LibraryStore, builtin: Path, name: str, by: str = "") -> dict:
    resolve(store, builtin, name)            # an invalid recipe can't become the selected one
    folder = user_dir(store)
    folder.mkdir(parents=True, exist_ok=True)
    rec = {"name": name, "tier": tier_of(store, builtin, name),
           "time": datetime.now().isoformat(timespec="seconds"), "by": by or _user()}
    with open(folder / ACTIVE_FILE, "a", encoding="utf-8") as f:     # append-only history
        f.write(json.dumps(rec) + "\n")
    return rec


def active(store: LibraryStore, builtin: Path) -> Optional[str]:
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
    """Station configs (stations/*.json|yaml) whose job.recipe is `name`."""
    d = Path(stations_dir) if stations_dir else None
    if d is None or not d.is_dir():
        return []
    out = []
    for f in sorted(d.iterdir()):
        if f.suffix.lower() == ".json":
            try:
                job = json.loads(f.read_text(encoding="utf-8")).get("job") or {}
                r = job.get("recipe") if isinstance(job, dict) else None
            except (ValueError, AttributeError, OSError):
                continue
        elif f.suffix.lower() in (".yaml", ".yml"):
            m = _YAML_RECIPE.search(f.read_text(encoding="utf-8", errors="replace"))
            r = m.group(1) if m else None
        else:
            continue
        if r == name:
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

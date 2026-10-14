"""
Zone sets in the library: save, load, list, delete (to trash, never destroyed).

    <library>/zones/<geometry_id>/<name>/
        zones.json    the set (model.ZoneSet.to_dict: zones in projector px, the geometry it
                      was painted at + its fingerprint, created); written LAST: a folder
                      without a readable one is a half-written save and is never listed
        masks.npz     union (bool, 1140 x 912), ids, masks (one bool mask per zone)
        preview.png   the zones as a DMD-space picture (dim fill, outline, id)
        camera.png    a photo of the zones projected on the part (when one was taken)
    <library>/zones/<geometry_id>/_shown.jsonl      one line per zone show that ran to its
                      end with every shown zone seen (record_shown): the operator had the
                      outlines on the part to look at. It answers the rig changes before it
    <library>/zones/<geometry_id>/_selection.jsonl  which sets are switched on
                      (engine/zones/selection.py). A leading '_' is never a set name.

A set is bound to its geometry_id: the same name may exist at another geometry (another rig
placement) and is a different set. Nothing is replaced in place: saving an existing name is
refused unless replace=True, which moves the old one to trash/zones/... first (and brings it
back if finishing the new one fails). Delete moves the entry to trash/ (recoverable).
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from ..library.store import LibraryStore, json_safe
from .model import ZoneError, ZoneSet, valid_set_name

KIND = "zones"
SHOWN_FILE = "_shown.jsonl"     # per geometry: the zone shows that answered rig changes
ZONES_FILE, MASKS_FILE, PREVIEW_FILE, PHOTO_FILE = "zones.json", "masks.npz", "preview.png", \
    "camera.png"
FILL_LEVEL = 55                 # preview: a zone's dim fill
_GEO_OK = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")


def _geo(geometry_id) -> str:
    if not isinstance(geometry_id, str) or not _GEO_OK.fullmatch(geometry_id) \
            or geometry_id in (".", ".."):
        raise ZoneError(f"geometry_id must be one id [A-Za-z0-9_.-], got {geometry_id!r}")
    return geometry_id


def _name(name) -> str:
    if not valid_set_name(name):
        raise ZoneError(f"zone set name must be letters, digits, _ . - (max 64), got {name!r}")
    return name


def entry(store: LibraryStore, geometry_id: str, name: str) -> Path:
    return Path(store.root) / KIND / _geo(geometry_id) / _name(name)


def exists(store: LibraryStore, geometry_id: str, name: str) -> bool:
    return (entry(store, geometry_id, name) / ZONES_FILE).is_file()


def geometries(store: LibraryStore) -> list:
    d = Path(store.root) / KIND
    if not d.is_dir():
        return []
    return sorted(p.name for p in d.iterdir() if p.is_dir() and _GEO_OK.fullmatch(p.name))


def names(store: LibraryStore, geometry_id: str) -> list:
    d = Path(store.root) / KIND / _geo(geometry_id)
    if not d.is_dir():
        return []
    return sorted(p.name for p in d.iterdir()
                  if p.is_dir() and valid_set_name(p.name) and (p / ZONES_FILE).is_file())


def geometries_of(store: LibraryStore, name: str) -> list:
    """The geometries a set of this name is painted at."""
    _name(name)
    return [g for g in geometries(store) if exists(store, g, name)]


def load(store: LibraryStore, geometry_id: str, name: str) -> ZoneSet:
    """The saved set, validated again (a hand-edited file is checked like a painted one)."""
    d = entry(store, geometry_id, name)
    f = d / ZONES_FILE
    if not f.is_file():
        have = names(store, geometry_id)
        raise ZoneError(f"no zone set {name!r} at geometry {geometry_id} (it has: "
                        f"{', '.join(have) or 'none'}; `scan zones` lists them, `scan "
                        f"zone-paint --name {name}` paints one)")
    try:
        doc = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise ZoneError(f"zone set {name!r}: {f.name} is unreadable ({e})") from None
    zs = ZoneSet.from_dict(doc, path=d)
    if zs.name != name or zs.geometry_id != geometry_id:
        raise ZoneError(f"zone set folder {geometry_id}/{name} holds a set that names itself "
                        f"{zs.geometry_id}/{zs.name}: the names inside must be the folder's")
    return zs.check()


def listing(store: LibraryStore, geometry_id: Optional[str] = None) -> list:
    """Every saved set (of one geometry, or all): its summary + valid / error, files."""
    out = []
    for g in ([_geo(geometry_id)] if geometry_id else geometries(store)):
        for n in names(store, g):
            d = entry(store, g, n)
            item = {"name": n, "geometry_id": g, "path": d, "valid": True, "error": None,
                    "preview": d / PREVIEW_FILE if (d / PREVIEW_FILE).is_file() else None,
                    "camera": d / PHOTO_FILE if (d / PHOTO_FILE).is_file() else None}
            try:
                zs = load(store, g, n)
                item.update(zs.summary(), rig_changes=rig_changes(store, zs))
            except ZoneError as e:              # listed as unusable, never hidden
                item.update(valid=False, error=str(e), zones=[], n_zones=0, rig_changes=[])
            out.append(item)
    return out


RIG_EVENTS_MAX = 500            # how far back the rig-state record is read for rig_changes


def record_shown(store: LibraryStore, zs: ZoneSet, ids=None, by: str = "") -> dict:
    """One line onto the geometry's _shown.jsonl: this painting of the set (name + created)
    was projected on the part and every shown zone landed in the camera image. The rig
    changes before it count as looked at (rig_changes)."""
    from ..library.presets import append_selection
    rec = {"time": datetime.now().isoformat(timespec="seconds"), "set": zs.name,
           "created": zs.created, "ids": [z.id for z in zs.select(ids)], "by": str(by or "")}
    f = Path(store.root) / KIND / _geo(zs.geometry_id) / SHOWN_FILE
    f.parent.mkdir(parents=True, exist_ok=True)
    append_selection(f, rec)
    return rec


def last_shown(store: LibraryStore, zs: ZoneSet) -> Optional[datetime]:
    """When THIS painting of the set was last shown on the part with every shown zone seen
    (None: never). An unreadable line is skipped: a missing answer keeps the warning."""
    from .. import rigstate
    f = Path(store.root) / KIND / _geo(zs.geometry_id) / SHOWN_FILE
    try:
        data = f.read_bytes()
    except OSError:
        return None
    last = None
    for raw in data.splitlines():
        try:
            rec = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            continue
        if isinstance(rec, dict) and rec.get("set") == zs.name and rec.get("created") == zs.created:
            t = rigstate._parse(rec.get("time"))
            if t is not None and (last is None or t > last):
                last = t
    return last


def rig_changes(store: LibraryStore, zs: ZoneSet) -> list:
    """What the geometry's rig-state record (engine/rigstate.py) says happened to the rig
    AFTER the set was painted - and after it was last shown on the part (last_shown): a rig
    check that found the rig MOVED (or could not tell), a re-home, a new stereo calibration,
    a new rig session (a mobile rig set down again). Any of them may mean the zones no longer
    sit where they were drawn - the engine cannot see where the projector points now, so this
    is a WARNING to check (`scan zone-show`) or paint again, never a gate here (clock times
    are compared: the set is not on that record). The judged runs lit through zone sets
    (Quick check, Station: ui/cli.py) PULL the part while a set they use has one."""
    from .. import rigstate
    t0 = rigstate._parse(zs.created)
    if t0 is None or not zs.geometry_id:
        return []
    shown = last_shown(store, zs)
    if shown is not None and shown > t0:
        t0 = shown
    out = []
    try:
        for i, rec in enumerate(rigstate.read_back(store, zs.geometry_id)):
            if i >= RIG_EVENTS_MAX or rec["time"] <= t0:
                break
            when = rec["time"].isoformat(timespec="seconds")
            ev = rec.get("event")
            if ev == "rig_check" and rec.get("status") in ("MOVED", "UNKNOWN", "REHOMED"):
                out.append(f"rig check {rec['status']} ({when})")
            elif ev == "calibration":
                out.append(f"a new stereo calibration ({when})")
            elif ev == "session":
                out.append(f"a new rig session ({when})")
    except (OSError, ValueError):
        return []
    out.reverse()
    return out


def stale_warning(store: LibraryStore, zs: ZoneSet) -> Optional[str]:
    """rig_changes() as one warning line, or None."""
    ch = rig_changes(store, zs)
    if not ch:
        return None
    shown = "; ".join(ch[:4]) + (f"; ... {len(ch) - 4} more" if len(ch) > 4 else "")
    return (f"zone set {zs.name} was painted {zs.created}; since then: {shown}. Zones are "
            "projector px at the rig placement they were painted at: check them (`scan "
            f"zone-show {zs.name}`) or paint the set again")


def preview_image(zs: ZoneSet, ids=None) -> np.ndarray:
    """The zones as a DMD-space picture, (rows, cols) uint8: dim fill, bright outline, id."""
    import cv2
    from . import painter
    img = np.zeros(tuple(zs.dmd_shape), np.uint8)
    chosen = zs.select(ids)
    for z in chosen:
        img[z.mask(tuple(zs.dmd_shape))] = FILL_LEVEL
    for z in chosen:
        cv2.polylines(img, [painter._pts(z.outline())], True, 255, 2, cv2.LINE_AA)
        painter._label(img, z, 255)
    return img


def outline_image(zs: ZoneSet, ids=None, keys: bool = False) -> np.ndarray:
    """Outlines + ids only on black (what `scan zone-show` projects: the part stays visible).
    keys: label each zone "<set>/<id>" (several sets shown together: their ids repeat)."""
    import cv2
    from . import painter
    img = np.zeros(tuple(zs.dmd_shape), np.uint8)
    for z in zs.select(ids):
        cv2.polylines(img, [painter._pts(z.outline())], True, 255, 3, cv2.LINE_AA)
        painter._label(img, z, 255, f"{zs.name}/{z.id}" if keys else None)
    return img


def _write_png(path: Path, img: np.ndarray) -> None:
    from ..decode.projmap import write_png
    if write_png(path, img) is None:
        raise OSError(f"could not encode {path.name}")


def save(store: LibraryStore, zs: ZoneSet, replace: bool = False,
         photo: Optional[np.ndarray] = None) -> ZoneSet:
    """Write the set as a NEW library entry -> the saved ZoneSet (path set). An existing name
    at that geometry is refused unless replace (the old one goes to trash/)."""
    zs.check()
    d = entry(store, zs.geometry_id, zs.name)
    had = (d / ZONES_FILE).is_file()
    if had and not replace:
        raise ZoneError(f"zone set {zs.name!r} exists at geometry {zs.geometry_id}: pick "
                        "another name, or replace it (the old one goes to trash/)")
    if d.exists() and not had:
        raise ZoneError(f"{d} exists but is not a complete zone set: move it away first")
    if not zs.created:
        zs.created = datetime.now().isoformat(timespec="seconds")
    d.parent.mkdir(parents=True, exist_ok=True)
    masks = zs.masks()
    # written under a temporary name (never a valid set name, so never listed) BEFORE the old
    # one is touched; the old one comes back if finishing the new one fails
    tmp = d.parent / f"~{zs.name}.{uuid.uuid4().hex[:8]}"
    tmp.mkdir()
    with store.filling(tmp):
        store.write_npz(tmp / MASKS_FILE, union=zs.mask(), ids=np.array(list(masks), dtype=str),
                        masks=np.stack([masks[i] for i in masks]))
        _write_png(tmp / PREVIEW_FILE, preview_image(zs))
        if photo is not None:
            _write_png(tmp / PHOTO_FILE, np.asarray(photo))
    trashed = store.trash(d, reason=f"replaced by zone-paint {zs.name}") if had else None
    try:
        with store.filling(tmp):
            doc = {**zs.to_dict(), "replaced": _library_path(store, trashed),
                   "areas_px": {i: int(m.sum()) for i, m in masks.items()},
                   "bboxes_dmd_px": zs.bboxes()}
            store.write_json(tmp / ZONES_FILE, json_safe(doc))      # last: a complete entry
            tmp.rename(d)
    except BaseException:
        if trashed is not None and not d.exists():
            try:
                Path(trashed).rename(d)
            except OSError:
                pass
        raise
    zs.path = d
    zs.replaced = _library_path(store, trashed)     # where the set it replaced went (or None)
    return zs


def _library_path(store: LibraryStore, p) -> Optional[str]:
    """A path inside the library as stored in a record: relative to the library root, posix
    (never an absolute path of this machine)."""
    if p is None:
        return None
    try:
        return Path(p).resolve().relative_to(store.root.resolve()).as_posix()
    except (ValueError, OSError):
        return Path(p).name


def read_masks(zs: ZoneSet) -> Optional[dict]:
    """The stored masks.npz of a saved set -> {"union", "ids", "masks"}, or None."""
    f = (zs.path / MASKS_FILE) if zs.path is not None else None
    if f is None or not f.is_file():
        return None
    with np.load(f) as z:
        return {"union": np.asarray(z["union"], bool), "ids": [str(i) for i in z["ids"]],
                "masks": np.asarray(z["masks"], bool)}


def delete(store: LibraryStore, geometry_id: str, name: str, reason: str) -> Path:
    """Move the set to trash/zones/<geometry_id>/<name> (recoverable) -> its path there."""
    if not reason or not str(reason).strip():
        raise ZoneError("deleting a zone set needs a reason (recorded with the trashed set)")
    d = entry(store, geometry_id, name)
    if not (d / ZONES_FILE).is_file():
        raise ZoneError(f"no zone set {name!r} at geometry {geometry_id}")
    return store.trash(d, reason=str(reason))

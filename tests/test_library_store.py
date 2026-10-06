"""Library store hardening (review fixes): trash moves exactly one entry, by
one atomic rename; half-written entries never break listings; listing
paths stay inside the library; output-bin JSON is strict JSON."""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from engine import rules
from engine.library import store as store_mod
from engine.library.bases import list_bases
from engine.library.objects import list_objects
from engine.library.store import LibraryStore, json_safe


def _entry(root, *parts, files=("field.npz", "meta.json", "preview.png")):
    d = root.joinpath(*parts)
    d.mkdir(parents=True)
    for f in files:
        (d / f).write_text(json.dumps({"label": "x", "geometry_id": parts[1] if parts[0] == "bases"
                                       else "g", "created": "2026-01-01T00:00:00",
                                       "base_ref": "b"}) if f == "meta.json" else "data")
    return d


@pytest.fixture
def lib(tmp_path):
    s = LibraryStore(tmp_path / "library")
    s.ensure_readme()
    return s


# ------------------------------------------------ trash: one entry only

def test_trash_refuses_geometry_folders_files_and_namespaces(lib):
    root = lib.root
    b1 = _entry(root, "bases", "geoA", "20260101T000000_000001_b1")
    _entry(root, "bases", "geoA", "20260101T000000_000002_b2")
    obj = _entry(root, "objects", "20260101T000000_000003_o1",
                 files=("field.npz", "relief.npz", "meta.json"))
    _entry(root, "spectral", "materials", "steel", files=("20260101T000000_000004.json",))
    for bad in (root / "bases" / "geoA", root / "bases", root / "calibrations" / "geoA",
                obj / "relief.npz", obj / "meta.json", root / "spectral" / "materials",
                root / "sessions" / "x", root.parent):
        with pytest.raises(ValueError):
            lib.trash(bad)
    # nothing moved, nothing half-deleted
    assert len(list_bases(lib, "geoA")) == 2
    assert (obj / "relief.npz").exists() and (obj / "meta.json").exists()
    assert not (root / "trash").exists() or not any((root / "trash").rglob("*.json"))
    # the real entries still go, one at a time
    dest = lib.trash(b1, "test")
    assert (dest / "TRASHED.json").exists() and not b1.exists()
    assert json.loads((dest / "TRASHED.json").read_text())["from"] == \
        "bases/geoA/20260101T000000_000001_b1"
    assert lib.trash(root / "spectral" / "materials" / "steel").exists()
    assert lib.trash(obj).exists()


def test_trash_refuses_a_folder_that_is_not_an_entry(lib):
    odd = _entry(lib.root, "maps", "geoA", "my_notes", files=("notes.txt",))
    with pytest.raises(ValueError, match="does not look like"):
        lib.trash(odd)
    empty = lib.root / "maps" / "geoA" / "empty_one"
    empty.mkdir()
    assert lib.trash(empty).exists()          # an empty folder can always be cleared


def test_trash_never_copies_when_a_file_is_locked(lib, monkeypatch):
    """Finding: shutil.move fell back to copytree+rmtree on a sharing
    violation and left a half-deleted entry that broke list_bases."""
    b = _entry(lib.root, "bases", "geoA", "20260101T000000_000001_b1")
    calls = []

    def locked(src, dst):
        calls.append((src, dst))
        raise PermissionError(32, "The process cannot access the file")
    monkeypatch.setattr(store_mod.os, "rename", locked)
    monkeypatch.setattr(store_mod, "TRASH_RETRY_DELAYS_S", (0.0, 0.0))
    with pytest.raises(rules.EntryInUse, match="retry"):
        lib.trash(b)
    assert len(calls) == 3                                     # retried, then gave up
    assert sorted(p.name for p in b.iterdir()) == ["field.npz", "meta.json", "preview.png"]
    assert not (lib.root / "trash" / "bases" / "geoA" / b.name).exists()
    assert [x.id for x in list_bases(lib, "geoA")] == [b.name]
    monkeypatch.undo()
    assert lib.trash(b).exists()                               # the retry just works


@pytest.mark.skipif(sys.platform != "win32", reason="Windows sharing violations")
def test_trash_with_a_really_open_file_leaves_the_entry_whole(lib, monkeypatch):
    monkeypatch.setattr(store_mod, "TRASH_RETRY_DELAYS_S", (0.0,))
    b = _entry(lib.root, "bases", "geoA", "20260101T000000_000001_b1")
    with open(b / "preview.png", "rb"):                        # like /files streaming it
        with pytest.raises(rules.EntryInUse):
            lib.trash(b)
        assert sorted(p.name for p in b.iterdir()) == ["field.npz", "meta.json", "preview.png"]
    assert [x.id for x in list_bases(lib, "geoA")] == [b.name]
    assert lib.trash(b).exists() and not b.exists()


# --------------------------------------- half-written entries (listings)

def test_listings_skip_entries_without_meta(lib):
    good = _entry(lib.root, "bases", "geoA", "20260101T000000_000001_good")
    (lib.root / "bases" / "geoA" / "20260101T000000_000002_partial").mkdir()
    half = _entry(lib.root, "objects", "20260101T000000_000003_half", files=("field.npz",))
    torn = _entry(lib.root, "objects", "20260101T000000_000004_torn", files=("field.npz",))
    (torn / "meta.json").write_text('{"label": "torn", "geometry_id"')      # cut mid-write
    assert [b.id for b in list_bases(lib)] == [good.name]
    assert list_objects(lib) == []
    inc = {(i["kind"], i["id"]) for i in lib.incomplete_entries()}
    assert inc == {("bases", "20260101T000000_000002_partial"), ("objects", half.name),
                   ("objects", torn.name)}
    for i in lib.incomplete_entries():                         # and they can be cleared
        lib.trash(i["path"], "incomplete")
    assert lib.incomplete_entries() == []


def test_failed_save_moves_the_partial_entry_to_trash(lib, monkeypatch):
    from engine.decode.deformation import DeformationField
    from engine.library import bases as base_lib
    fld = DeformationField(corr_x=np.zeros((4, 4), np.float32), corr_y=None, corr_diag=None,
                           conf=np.ones((4, 4), np.float32), mask=np.ones((4, 4), bool),
                           geometry_id="geoA", basis_names=["t"])

    def boom(*a, **k):
        raise OSError(28, "No space left on device")
    monkeypatch.setattr(LibraryStore, "write_preview", staticmethod(boom))
    with pytest.raises(OSError, match="No space"):
        base_lib.save_base(lib, fld, "b", {})
    assert list_bases(lib) == [] and lib.incomplete_entries() == []
    moved = list((lib.root / "trash" / "bases" / "geoA").iterdir())
    assert len(moved) == 1 and (moved[0] / "field.npz").exists()
    assert "incomplete save" in json.loads((moved[0] / "TRASHED.json").read_text())["reason"]


# ------------------------------------------------ listing path safety

def test_entries_stay_inside_the_library(lib):
    (lib.root.parent / "outside" / "secret").mkdir(parents=True)
    for bad in (("banks", "../../outside"), ("banks", ".."), ("..",), ("bases", "a/../..")):
        with pytest.raises(ValueError):
            lib.entries(*bad)
    assert lib.entries("banks", "nothing_here") == []


# ----------------------------------------------------- strict JSON out

def test_json_safe_makes_strict_json():
    doc = {"a": float("nan"), "b": [np.float32("inf"), 1.5, np.int64(3)],
           "c": np.array([1.0, np.nan]), "d": (np.bool_(True), None), "e": {"f": -np.inf}}
    safe = json_safe(doc)

    def strict(s):
        return json.loads(s, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
    out = strict(json.dumps(safe, allow_nan=False))
    assert out == {"a": None, "b": [None, 1.5, 3], "c": [1.0, None], "d": [True, None],
                   "e": {"f": None}}


def test_rename_is_atomic_os_rename(lib, monkeypatch):
    """trash must use os.rename only (no shutil.move copy fallback)."""
    assert not hasattr(store_mod, "shutil")
    b = _entry(lib.root, "readings", "20260101T000000_000001_r", files=("codes.json",))
    seen = []
    real = os.rename
    monkeypatch.setattr(store_mod.os, "rename", lambda s, d: (seen.append(1), real(s, d)))
    lib.trash(b)
    assert seen == [1]


# ------------------------------------- trash: same name trashed again

def test_trash_same_name_twice_gets_a_suffix_never_replaces(lib):
    """A material re-learned (or a sensor re-taught) after its folder was
    trashed has the same entry name: the second trash goes to <name>-1,
    the third to <name>-2 - one atomic rename each, nothing replaced."""
    mats = lib.root / "spectral" / "materials"
    dests = []
    for i in range(3):
        m = _entry(lib.root, "spectral", "materials", "steel",
                   files=(f"20260101T00000{i}_000001.json",))
        dests.append(lib.trash(m, f"take {i}"))
        assert not m.exists()
    trash = lib.root / "trash" / "spectral" / "materials"
    assert [d.name for d in dests] == ["steel", "steel-1", "steel-2"]
    assert all(d.parent == trash and d.is_dir() for d in dests)
    for i, d in enumerate(dests):                  # each keeps its own contents + record
        assert (d / f"20260101T00000{i}_000001.json").exists()
        rec = json.loads((d / "TRASHED.json").read_text())
        assert rec["from"] == "spectral/materials/steel" and rec["reason"] == f"take {i}"
        assert rec.get("trashed_as") == (None if i == 0 else d.name)
    assert not mats.joinpath("steel").exists()
    # still refuses non-entries, still an atomic rename
    with pytest.raises(ValueError):
        lib.trash(lib.root / "spectral" / "materials")


def test_trash_suffix_survives_a_race_for_the_same_name(lib, monkeypatch):
    """The target appearing between the exists() check and the rename (another
    process trashing the same name) takes the next suffix instead of failing."""
    e = _entry(lib.root, "maps", "geoA", "20260101T000000_000001_m", files=("map.npz",))
    real = os.rename
    first = lib.root / "trash" / "maps" / "geoA" / e.name

    def racy(src, dst):
        if Path(dst) == first:
            first.mkdir(parents=True)              # someone else got there first
            raise FileExistsError(183, "Cannot create a file when that file already exists")
        return real(src, dst)
    monkeypatch.setattr(store_mod.os, "rename", racy)
    dest = lib.trash(e)
    assert dest.name == f"{e.name}-1" and (dest / "map.npz").exists() and first.is_dir()


def test_restored_then_retrashed_entry_keeps_both_records(lib):
    r = _entry(lib.root, "readings", "20260101T000000_000001_r", files=("codes.json",))
    t1 = lib.trash(r, "first")
    os.rename(t1, r)                               # the operator restores it by hand
    t2 = lib.trash(r, "second")
    assert t2.name == f"{r.name}" and t2 == t1     # the old slot is free again
    assert json.loads((t2 / "TRASHED.json").read_text())["reason"] == "first"
    assert json.loads((t2 / "TRASHED-1.json").read_text())["reason"] == "second"


def test_sensor_entries_can_be_trashed(lib):
    """sensors/<geometry_id>/<name>/ is a library entry (was a KeyError)."""
    s = _entry(lib.root, "sensors", "geoA", "tray", files=("20260101T000000_000001.json",))
    dest = lib.trash(s, "re-teach")
    assert dest == lib.root / "trash" / "sensors" / "geoA" / "tray" and not s.exists()
    for bad in (lib.root / "sensors" / "geoA", lib.root / "sensors"):
        with pytest.raises(ValueError):
            lib.trash(bad)
    odd = _entry(lib.root, "sensors", "geoA", "notes", files=("readme.txt",))
    with pytest.raises(ValueError, match="does not look like"):
        lib.trash(odd)
    s2 = _entry(lib.root, "sensors", "geoA", "tray", files=("20260102T000000_000001.json",))
    assert lib.trash(s2).name == "tray-1"

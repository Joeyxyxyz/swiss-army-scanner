"""Zone sets in the library (engine/zones/store.py + the binding rules of use.py): save / load
round trip, masks.npz, list / show / delete to trash (never destroyed), bound to the rig
geometry, and a recipe zone that refers to a set. No rig: zone sets are written directly and
the recipe runs on a synthetic stored scan."""

import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from engine import api, config, rules
from engine.recipes import RecipeError, schema
from engine.recipes.geometry import ZoneError as RecipeZoneError
from engine.recipes.geometry import ScanView, dmd_zone_mask, part_frame, resolve_zone
from engine.zones import DMD_SHAPE, Zone, ZoneError, ZoneSet
from engine.zones import store as zstore
from engine.zones import use

import recipe_synth as RS

PRESET = "bench_9in_metric"


@pytest.fixture
def lib(tmp_path):
    s = api.configure(tmp_path / "library", mock=True)
    p = api.load_preset(PRESET)
    yield s, p
    api._session = None


def make(p, name="lid", zones=None, **kw) -> ZoneSet:
    return ZoneSet(name=name, geometry_id=p.geometry_id, preset=p.name,
                   fingerprint=config.geometry_fingerprint(p), mock=True, zones=zones or [
                       Zone.rect("Z1", 100, 200, 400, 500, name="part"),
                       Zone.ellipse("Z2", 500, 300, 800, 460, rotation_deg=20),
                       Zone.polygon("Z3", [(150, 650), (420, 680), (380, 950), (140, 900)])], **kw)


# ------------------------------------------------------------------ save / load

def test_save_writes_zones_masks_preview_and_loads_back(lib):
    s, p = lib
    zs = zstore.save(s.store, make(p), photo=np.full((40, 60), 99, np.uint8))
    d = s.store.root / "zones" / p.geometry_id / "lid"
    assert zs.path == d and zs.created
    assert sorted(f.name for f in d.iterdir()) == ["camera.png", "masks.npz", "preview.png",
                                                   "zones.json"]
    doc = json.loads((d / "zones.json").read_text(encoding="utf-8"))
    assert doc["kind"] == "zoneset" and doc["name"] == "lid" and doc["geometry_id"] == p.geometry_id
    assert doc["geometry_fp"] == config.geometry_fingerprint(p) and doc["replaced"] is None
    assert doc["areas_px"]["Z1"] == 300 * 300 and doc["bboxes_dmd_px"]["Z1"] == [100, 200, 400, 500]
    assert "projector px" in doc["units"]
    back = zstore.load(s.store, p.geometry_id, "lid")
    assert back.ids == ["Z1", "Z2", "Z3"] and back.path == d and back.zone("Z1").name == "part"
    for a, b in zip(zs.zones, back.zones):
        assert np.array_equal(a.mask(), b.mask()) and a.shape == b.shape
    stored = zstore.read_masks(back)
    assert stored["ids"] == ["Z1", "Z2", "Z3"] and stored["masks"].shape == (3, *DMD_SHAPE)
    assert stored["union"].dtype == bool and np.array_equal(stored["union"], back.mask())
    for i, zid in enumerate(stored["ids"]):
        assert np.array_equal(stored["masks"][i], back.zone(zid).mask())
    # the preview: a DMD-space picture, the zones dim, their outlines bright, the rest black
    img = cv2.imread(str(d / "preview.png"), cv2.IMREAD_GRAYSCALE)
    assert img.shape == DMD_SHAPE and img[350, 250] == zstore.FILL_LEVEL and img[200, 250] > 200
    assert img[100, 850] == 0 and (img > 0).mean() < 0.3
    assert cv2.imread(str(d / "camera.png"), cv2.IMREAD_GRAYSCALE).shape == (40, 60)


def test_an_invalid_set_is_never_written(lib):
    s, p = lib
    bad = make(p, zones=[Zone.rect("Z1", 800, 100, 1000, 300)])
    with pytest.raises(ZoneError, match="outside the DMD"):
        zstore.save(s.store, bad)
    for name in ("../up", "a/b", "", "x" * 65, ".."):
        with pytest.raises(ZoneError):
            zstore.save(s.store, make(p, name=name))
    with pytest.raises(ZoneError, match="geometry_id"):
        zstore.entry(s.store, "../elsewhere", "lid")
    assert not (s.store.root / "zones").exists() or not any((s.store.root / "zones").rglob("*.json"))


def test_an_existing_name_is_refused_and_replace_moves_the_old_one_to_trash(lib):
    s, p = lib
    first = zstore.save(s.store, make(p))
    with pytest.raises(ZoneError, match="exists at geometry"):
        zstore.save(s.store, make(p, zones=[Zone.rect("Z1", 10, 10, 100, 100)]))
    assert zstore.load(s.store, p.geometry_id, "lid").ids == ["Z1", "Z2", "Z3"]     # untouched
    second = zstore.save(s.store, make(p, zones=[Zone.rect("Z1", 10, 10, 100, 100)]), replace=True)
    assert zstore.load(s.store, p.geometry_id, "lid").ids == ["Z1"] and second.path == first.path
    t = s.store.root / "trash" / "zones" / p.geometry_id / "lid"
    old = json.loads((t / "zones.json").read_text(encoding="utf-8"))
    assert [z["id"] for z in old["zones"]] == ["Z1", "Z2", "Z3"] and (t / "masks.npz").is_file()
    assert "replaced by zone-paint" in json.loads((t / "TRASHED.json").read_text())["reason"]
    doc = json.loads((second.path / "zones.json").read_text(encoding="utf-8"))
    # where the old set went, relative to the library (never a path of this machine)
    assert doc["replaced"] == f"trash/zones/{p.geometry_id}/lid"
    assert (s.store.root / doc["replaced"] / "zones.json").is_file()
    zstore.save(s.store, make(p), replace=True)             # again: the trash keeps both
    assert (t.parent / "lid-1" / "zones.json").is_file() and (t / "zones.json").is_file()
    # no temporary folder is left behind, and none is ever listed
    assert sorted(x.name for x in second.path.parent.iterdir()) == ["lid"]


def test_list_show_and_delete_to_trash(lib):
    s, p = lib
    zstore.save(s.store, make(p, "lid"))
    zstore.save(s.store, make(p, "base", zones=[Zone.rect("Z1", 10, 10, 200, 100)]))
    other = replace(p, geometry_id="other_geo")
    zstore.save(s.store, make(other, "lid", zones=[Zone.rect("Z1", 10, 10, 200, 100)]))
    rows = zstore.listing(s.store)
    assert [(r["geometry_id"], r["name"], r["n_zones"], r["valid"]) for r in rows] == [
        (p.geometry_id, "base", 1, True), (p.geometry_id, "lid", 3, True), ("other_geo", "lid", 1, True)]
    assert [r["name"] for r in zstore.listing(s.store, p.geometry_id)] == ["base", "lid"]
    lid = rows[1]
    assert lid["zones"][0] == {"id": "Z1", "name": "part", "shape": "rect",
                               "size_px": [300.0, 300.0],
                               "bbox_dmd_px": [100, 200, 400, 500], "area_px": 90000.0,
                               "rotation_deg": 0.0, "note": ""}
    assert lid["preview"] == lid["path"] / "preview.png" and lid["camera"] is None
    assert zstore.geometries_of(s.store, "lid") == sorted([p.geometry_id, "other_geo"])
    # a hand-edited set that is no longer valid is listed as unusable, never hidden or used
    f = rows[0]["path"] / "zones.json"
    doc = json.loads(f.read_text(encoding="utf-8"))
    doc["zones"][0]["center"] = [900, 50]
    f.write_text(json.dumps(doc), encoding="utf-8")
    bad = zstore.listing(s.store, p.geometry_id)[0]
    assert bad["name"] == "base" and bad["valid"] is False and "outside the DMD" in bad["error"]
    with pytest.raises(ZoneError, match="outside the DMD"):
        zstore.load(s.store, p.geometry_id, "base")
    # a folder without zones.json (a half-written save) is not a set
    (s.store.root / "zones" / p.geometry_id / "half").mkdir()
    assert [r["name"] for r in zstore.listing(s.store, p.geometry_id)] == ["base", "lid"]
    # delete: a reason, then to trash - whole and readable there
    with pytest.raises(ZoneError, match="needs a reason"):
        zstore.delete(s.store, p.geometry_id, "lid", "  ")
    dest = zstore.delete(s.store, p.geometry_id, "lid", "painted again")
    assert dest == s.store.root / "trash" / "zones" / p.geometry_id / "lid"
    assert sorted(x.name for x in dest.iterdir()) == ["TRASHED.json", "masks.npz", "preview.png",
                                                      "zones.json"]
    assert json.loads((dest / "TRASHED.json").read_text())["reason"] == "painted again"
    assert not zstore.exists(s.store, p.geometry_id, "lid")
    assert zstore.exists(s.store, "other_geo", "lid")       # the other geometry's set is another set
    with pytest.raises(ZoneError, match="no zone set"):
        zstore.delete(s.store, p.geometry_id, "lid", "again")
    with pytest.raises(ZoneError, match="no zone set 'nope'"):
        zstore.load(s.store, p.geometry_id, "nope")


def test_a_set_whose_file_names_another_set_is_refused(lib):
    s, p = lib
    zs = zstore.save(s.store, make(p))
    moved = zs.path.parent / "renamed"
    zs.path.rename(moved)
    with pytest.raises(ZoneError, match="names itself"):
        zstore.load(s.store, p.geometry_id, "renamed")


# ------------------------------------------------------------------ bound to the rig geometry

def test_a_set_is_bound_to_its_geometry_and_fingerprint(lib):
    s, p = lib
    zstore.save(s.store, make(p))
    assert use.for_preset(s.store, p, "lid").ids == ["Z1", "Z2", "Z3"]
    zs, ids, text = use.select(s.store, p, "lid:Z3,Z1", bare=True)
    assert ids == ["Z3", "Z1"] and text == "zoneset:lid:Z3,Z1"
    with pytest.raises(ZoneError, match="no zone 'Z9'"):
        use.select(s.store, p, "zoneset:lid:Z9")
    # another rig geometry: refused, and it says where the set WAS painted
    moved = replace(p, geometry_id="mock-bench_moved")
    with pytest.raises(rules.GeometryMismatch, match="painted at geometry mock-bench_9in_v1"):
        use.for_preset(s.store, moved, "lid")
    # the same geometry_id after a geometry-bound setting changed: refused too
    changed = replace(p, rig=replace(p.rig, baseline_mm=(p.rig.baseline_mm or 200.0) + 25.0))
    with pytest.raises(rules.GeometryMismatch, match="baseline_mm"):
        use.for_preset(s.store, changed, "lid")
    with pytest.raises(ZoneError, match="no zone set 'nope'"):
        use.for_preset(s.store, p, "nope")
    # as a lit zone anywhere in the engine: the union mask, never a stored camera mask
    from engine.capture import zones as gz
    pm, cam, cells = gz.resolve(s, p, "zoneset:lid:Z1")
    assert cam is None and pm.dtype == bool and np.array_equal(pm, zs.zone("Z1").mask())
    assert cells == [(c, r) for c in range(1, 5) for r in range(2, 6)]
    with pytest.raises(rules.GeometryMismatch):
        gz.resolve(s, moved, "zoneset:lid")
    assert gz.resolve(s, p, "C4")[2] == [(2, 3)]            # grid cells still resolve as before


def test_api_listing_recipe_fragment_mask_and_delete(lib, tmp_path):
    s, p = lib
    zstore.save(s.store, make(p))
    assert [z["name"] for z in api.zone_sets(p)] == ["lid"]
    assert [z["name"] for z in api.zone_sets(all_geometries=True)] == ["lid"]
    assert [z["name"] for z in api.list_library("zones")] == ["lid"]
    assert api.zone_set("lid:Z1", p).name == "lid" and api.paths()["zones"] == s.store.root / "zones"
    frag = api.zone_recipe("lid", p)
    assert [z["shape"] for z in frag] == ["rect", "polygon", "polygon"]
    assert api.zone_recipe("lid:Z1", p, by_ref=True)[0]["ref"] == "zoneset:lid/Z1"
    m = api.zone_mask("lid:Z1,Z3", p)
    assert m.shape == DMD_SHAPE and m.dtype == bool and int(m.sum()) > 90000
    out = api.zone_mask("lid", p, out=tmp_path / "lid.npy")
    assert np.array_equal(np.load(out), api.zone_mask("lid", p))
    with pytest.raises(FileExistsError):
        api.zone_mask("lid", p, out=tmp_path / "lid.npy")           # never overwritten
    with pytest.raises(ZoneError, match=".npy"):
        api.zone_mask("lid", p, out=tmp_path / "lid.png")
    # the fast scan takes that file as its --mask
    from engine.reconstruct.fast_scan import _dmd_mask
    assert np.array_equal(_dmd_mask(str(out)), m | api.zone_mask("lid:Z2", p))
    # a recipe names the set: deleting it is refused unless forced
    api.recipe_add("uses_lid", recipe=_recipe([{"id": "a", "ref": "zoneset:lid/Z1",
                                               "checks": [{"type": "presence"}]}]))
    with pytest.raises(ZoneError, match="used by recipe.*uses_lid"):
        api.zone_delete("lid", "tidy", p)
    r = api.zone_delete("lid", "tidy", p, force=True)
    assert r["forced"] and r["recipes"] == ["uses_lid"] and Path(r["trashed"]).is_dir()
    assert api.zone_sets(p) == []
    assert api.recipe_validate("uses_lid")["valid"] is False        # ... and the recipe says so


def test_rig_events_after_painting_are_a_warning_to_check_the_zones(lib):
    """The rig-state record (rig checks, calibrations, sessions) is read for what happened
    AFTER a set was painted: the set may no longer sit where it was drawn. A warning, never a
    silent pass - and never a refusal either (the engine cannot see where the projector points)."""
    from datetime import datetime, timedelta
    from engine import rigstate
    s, p = lib
    zs = zstore.save(s.store, make(p, created="2026-03-01T10:00:00"))
    t = datetime(2026, 3, 1, 10, 0, 0)
    assert zstore.rig_changes(s.store, zs) == [] and zstore.stale_warning(s.store, zs) is None
    geo = p.geometry_id
    rigstate.append(s.store, geo, {"event": "rig_check", "status": "MOVED"}, now=t - timedelta(hours=1))
    rigstate.append(s.store, geo, {"event": "rig_check", "status": "OK"}, now=t + timedelta(minutes=5))
    assert zstore.rig_changes(s.store, zs) == []            # before painting / a good check: nothing
    rigstate.append(s.store, geo, {"event": "rig_check", "status": "MOVED"}, now=t + timedelta(hours=1))
    rigstate.append(s.store, geo, {"event": "calibration", "calibration_id": "c"},
                    now=t + timedelta(hours=2))
    rigstate.append(s.store, geo, {"event": "session", "reason": "moved"}, now=t + timedelta(hours=3))
    ch = zstore.rig_changes(s.store, zs)
    assert ch == ["rig check MOVED (2026-03-01T11:00:00)", "a new stereo calibration (2026-03-01T12:00:00)",
                  "a new rig session (2026-03-01T13:00:00)"]
    w = zstore.stale_warning(s.store, zs)
    assert "painted 2026-03-01T10:00:00" in w and "rig check MOVED" in w and "zone-show lid" in w
    assert zstore.listing(s.store, geo)[0]["rig_changes"] == ch
    # a recipe that refers to the set carries the warning to its run (bound when it is loaded)
    r = api.recipe_load(_recipe([{"id": "a", "ref": "zoneset:lid/Z1", "checks": [{"type": "presence"}]}]))
    z = r.zone("a")
    assert use.recipe_warnings(z, {"geometry_id": geo}) == [f"zone a: {w}"]
    assert use.recipe_warnings(z, {"geometry_id": "elsewhere"}) == []
    # painted again: the new set has nothing after it
    fresh = make(p, created=(t + timedelta(hours=4)).isoformat(timespec="seconds"))
    assert zstore.rig_changes(s.store, zstore.save(s.store, fresh, replace=True)) == []


# ------------------------------------------------------------------ a recipe zone that refers to a set

def _recipe(zones, **top) -> dict:
    d = {"schema_version": 1, "name": "t", "preset": PRESET, "scan": {"kind": "fast", "axis": "xy"},
         "zones": zones, "feedback": {"enabled": False}}
    d.update(top)
    return d


def test_recipe_ref_is_checked_for_its_form_and_against_the_library(lib):
    s, p = lib
    zstore.save(s.store, make(p))
    ok = _recipe([{"id": "lid", "label": "the lid", "ref": "zoneset:lid/Z1",
                   "checks": [{"type": "presence"}]}])
    # without a library: the form only, and the zone stays unbound
    r = schema.validate(ok)
    z = r.zone("lid")
    assert (z.shape, z.units, z.coords) == ("zoneset", "projector_px", "lid/Z1")
    assert z.ref == {"spec": "zoneset:lid/Z1", "set": "lid", "zone": "Z1", "bound": None}
    with pytest.raises(RecipeZoneError, match="not resolved against a library"):
        dmd_zone_mask(z, {"geometry_id": p.geometry_id})
    # through the library: bound to every geometry the set is painted at
    r = api.recipe_load(ok)
    b = r.zone("lid").ref["bound"]
    assert list(b) == [p.geometry_id] and b[p.geometry_id]["zone"]["id"] == "Z1"
    assert b[p.geometry_id]["geometry_fp"] == config.geometry_fingerprint(p)
    assert r.summary()["zones"] == [{"id": "lid", "shape": "zoneset", "units": "projector_px",
                                     "checks": ["presence"], "ref": "zoneset:lid/Z1"}]
    # the normalised recipe stays a reference (and validates again)
    norm = r.to_dict()
    assert norm["zones"][0] == {"id": "lid", "label": "the lid", "ref": "zoneset:lid/Z1",
                                "datum": False, "feedback": True,
                                "checks": [{"type": "presence", "id": "presence",
                                            "params": norm["zones"][0]["checks"][0]["params"]}]}
    assert api.recipe_load(norm).zone("lid").ref["bound"]
    for zone, msg in (
            ({"id": "a", "ref": "zoneset:lid/Z9", "checks": [{"type": "presence"}]}, "no zone 'Z9'"),
            ({"id": "a", "ref": "zoneset:nope/Z1", "checks": [{"type": "presence"}]},
             "no zone set 'nope'"),
            ({"id": "a", "ref": "lid/Z1", "checks": [{"type": "presence"}]}, "zoneset:<set>/<zone id>"),
            ({"id": "a", "ref": "zoneset:lid/Z1", "shape": "rect", "coords": [1, 2, 3, 4],
              "checks": [{"type": "presence"}]}, "remove coords, shape"),
            ({"id": "a", "ref": "zoneset:lid/Z1", "units": "part_mm",
              "checks": [{"type": "presence"}]}, "remove units"),
            ({"id": "a", "shape": "zoneset", "units": "projector_px", "coords": "lid/Z1",
              "checks": [{"type": "presence"}]}, "must be one of")):
        with pytest.raises(RecipeError, match=msg):
            api.recipe_load(_recipe([zone]))
    # a projector-px zone needs both axes decoded: said at validation, as for any DMD zone
    with pytest.raises(RecipeError, match="axis xy"):
        api.recipe_load(_recipe(ok["zones"], scan={"kind": "fast", "axis": "x"}))
    # the lit zone of the recipe's scan may be a zone set too
    assert api.recipe_load(_recipe(ok["zones"], scan={"kind": "fast", "axis": "xy",
                                                      "zone": "zoneset:lid"})).scan["zone"] == "zoneset:lid"
    assert use.recipe_refs(s.store, api.paths()["recipes"], "lid") == []
    api.recipe_add("uses_lid", recipe=ok)
    assert use.recipe_refs(s.store, api.paths()["recipes"], "lid") == ["uses_lid"]
    assert use.recipe_refs(s.store, api.paths()["recipes"], "li") == []


@pytest.fixture(scope="module")
def scanned(tmp_path_factory):
    """A mock library with the bench's true stereo calibration and one synthetic metric scan of
    the box-lid scene (both DMD axes decoded) - and where the lid lands on the DMD."""
    root = tmp_path_factory.mktemp("zrecipe") / "library"
    s = api.configure(root, mock=True)
    try:
        p = api.load_preset(PRESET)
        ms = RS.oracle_scan(s, p, "scene:box_lid", "box_lid", seed=3)
        z = ms.cloud.xyz[..., 2]
        ok = ms.cloud.mask & np.isfinite(z)
        lid = ok & (z < np.nanmedian(z[ok]) - 20.0)         # 40 mm proud of the wall
        fld = ms.field
        cx, cy = float(np.median(fld.corr_x[lid])), float(np.median(fld.corr_y[lid]))
        wall = ok & ~lid & (np.abs(fld.corr_x - cx) > 220)
        wx, wy = float(np.median(fld.corr_x[wall])), float(np.median(fld.corr_y[wall]))
    finally:
        api._session = None
    return root, ms.id, (cx, cy), (wx, wy)


@pytest.fixture
def scan_lib(scanned):
    s = api.configure(scanned[0], mock=True)
    p = api.load_preset(PRESET)
    yield s, p, scanned[1], scanned[2], scanned[3]
    for geo in zstore.geometries(s.store):                  # each test paints its own sets
        for n in zstore.names(s.store, geo):
            zstore.delete(s.store, geo, n, "test cleanup")
    api._session = None


def _lid_recipe():
    return _recipe([
        {"id": "wall", "ref": "zoneset:box/Z2", "datum": True},
        {"id": "lid", "ref": "zoneset:box/Z1", "checks": [
            {"type": "presence", "params": {"min_height_mm": 20.0, "min_area_pct": 50.0}},
            {"type": "height", "params": {"nominal_mm": 40.0, "tol_mm": 2.0, "min_height_mm": 20.0}}]}],
        part={"frame": {"reference": {"zone": "wall"}}})


def _paint_box(s, p, lid_xy, wall_xy, half=24):
    (cx, cy), (wx, wy) = lid_xy, wall_xy
    return zstore.save(s.store, make(p, "box", zones=[
        Zone.rect("Z1", cx - half, cy - half, cx + half, cy + half, name="part"),
        Zone.rect("Z2", wx - 30, wy - 60, wx + 30, wy + 60, name="datum")]), replace=True)


def test_recipe_runs_on_the_zones_painted_at_the_scans_geometry(scan_lib, tmp_path):
    s, p, scan_id, lid_xy, wall_xy = scan_lib
    _paint_box(s, p, lid_xy, wall_xy)
    out = api.recipe_run(_lid_recipe(), scan=scan_id, runs_dir=tmp_path / "runs",
                         project_feedback=False)
    res = out["result"]
    assert res["result"] == "PASS" and out["exit_code"] == 0, (res["invalid_reason"], res["zones"])
    z = {z["id"]: z for z in res["zones"]}
    assert z["lid"]["shape"] == "zoneset" and z["lid"]["units"] == "projector_px"
    assert z["lid"]["n_points"] > 500 and z["lid"]["verdict"] == "PASS"
    h = next(c for c in z["lid"]["checks"] if c["type"] == "height")
    assert h["values"]["height_mm"] == pytest.approx(40.0, abs=0.5)
    assert res["recipe"]["frame"]["reference"] == "datum zone"
    assert res["recipe"]["zones"][1]["ref"] == "zoneset:box/Z1"
    # the set painted again somewhere else: the SAME recipe now looks there (a reference)
    _paint_box(s, p, wall_xy, lid_xy, half=20)              # "lid" zone on the wall: no lid in it
    again = api.recipe_run(_lid_recipe(), scan=scan_id, runs_dir=tmp_path / "runs",
                           project_feedback=False)["result"]
    assert again["result"] != "PASS", again["zones"]


def test_recipe_zones_painted_elsewhere_are_never_used(scan_lib, tmp_path):
    """Fail closed: a set painted at another geometry, or at other rig settings, cannot be
    resolved on this scan -> its checks are INVALID and the part is PULLed (never passed)."""
    s, p, scan_id, lid_xy, wall_xy = scan_lib
    other = replace(p, geometry_id="mock-bench_other")
    (cx, cy), (wx, wy) = lid_xy, wall_xy
    zstore.save(s.store, make(other, "box", zones=[
        Zone.rect("Z1", cx - 24, cy - 24, cx + 24, cy + 24), Zone.rect("Z2", wx - 30, wy - 60, wx + 30, wy + 60)]))
    recipe = _recipe([{"id": "lid", "ref": "zoneset:box/Z1", "checks": [
        {"type": "presence", "params": {"min_height_mm": 20.0, "min_area_pct": 50.0}}]}])
    r = api.recipe_load(recipe)                             # it exists in the library: loads
    assert list(r.zone("lid").ref["bound"]) == ["mock-bench_other"]
    out = api.recipe_run(recipe, scan=scan_id, runs_dir=tmp_path / "runs", project_feedback=False)
    res = out["result"]
    assert res["result"] == "INVALID" and res["disposition"] == "PULL" and out["exit_code"] == 5
    assert any("not painted at this scan's geometry" in w for w in res["warnings"]), res["warnings"]
    assert res["zones"][0]["checks"][0]["pull"] == "zone_unresolved"
    # the same geometry_id, other rig settings (a fingerprint that no longer matches)
    stale = make(p, "box", zones=[Zone.rect("Z1", cx - 24, cy - 24, cx + 24, cy + 24)])
    stale.fingerprint = {**stale.fingerprint, "standoff_mm": 1.0}
    zstore.save(s.store, stale)
    out = api.recipe_run(recipe, scan=scan_id, runs_dir=tmp_path / "runs", project_feedback=False)
    assert out["result"]["result"] == "INVALID" and out["result"]["disposition"] == "PULL"
    assert any("other rig settings" in w for w in out["result"]["warnings"])
    # resolved directly: the same refusal, as a ZoneError naming the geometry
    ms = next(m for m in api.list_library("metric") if m.id == scan_id)
    from engine.calibrate import stereo
    cal = stereo.latest(s.store, p.geometry_id, config.geometry_fingerprint(p))
    sv = ScanView.from_scan(ms, cal)
    with pytest.raises(RecipeZoneError, match="other rig settings"):
        resolve_zone(api.recipe_load(recipe).zone("lid"), sv, part_frame(sv))

"""Painted zones, the model (engine/zones/model.py + the text / recipe forms of use.py):
pure geometry, no display, no library, no rig."""

import json
import math

import numpy as np
import pytest

from engine.zones import DMD_SHAPE, MIN_SIZE_PX, Zone, ZoneError, ZoneSet, next_id, zone_problems
from engine.zones import model as M
from engine.zones import use

H, W = DMD_SHAPE


def test_dmd_shape_is_rows_by_cols():
    assert DMD_SHAPE == (1140, 912)


# ------------------------------------------------------------------ rasterising vs analytic area

def test_rect_mask_is_exact_and_lands_where_it_is_drawn():
    z = Zone.rect("Z1", 100, 200, 400, 500)
    m = z.mask()
    assert m.shape == DMD_SHAPE and m.dtype == bool
    assert int(m.sum()) == 300 * 300 == round(z.area())
    rows, cols = np.nonzero(m)
    assert (cols.min(), cols.max(), rows.min(), rows.max()) == (100, 399, 200, 499)
    assert z.bbox() == (100.0, 200.0, 400.0, 500.0)
    # the two corners may be given in any order
    assert np.array_equal(Zone.rect("Z1", 400, 500, 100, 200).mask(), m)


@pytest.mark.parametrize("deg", [0, 15, 30, 45, 90, -120, 179.5])
def test_rotated_rect_area_matches_within_its_outline(deg):
    z = Zone.rect("Z1", 300, 400, 620, 600, rotation_deg=deg)
    assert z.area() == pytest.approx(320 * 200)
    px = int(z.mask().sum())
    # far less than its outline (1040 px): at most the mirrors whose centre is ON an edge
    assert abs(px - z.area()) <= 0.15 * 2 * (320 + 200), (deg, px)
    cx, cy = z.center
    rows, cols = np.nonzero(z.mask())
    assert abs(cols.mean() + 0.5 - cx) < 0.6 and abs(rows.mean() + 0.5 - cy) < 0.6


@pytest.mark.parametrize("w,h,deg", [(300, 300, 0), (400, 160, 0), (400, 160, 37), (90, 220, -60)])
def test_ellipse_area_matches_pi_a_b(w, h, deg):
    z = Zone.ellipse("Z1", 400 - w / 2, 500 - h / 2, 400 + w / 2, 500 + h / 2, rotation_deg=deg)
    assert z.area() == pytest.approx(math.pi * w * h / 4)
    px = int(z.mask().sum())
    assert abs(px - z.area()) / z.area() < 0.004, (px, z.area())
    x0, y0, x1, y1 = z.bbox()
    rows, cols = np.nonzero(z.mask())
    assert x0 - 1 <= cols.min() and cols.max() + 1 <= x1 + 1 and y0 - 1 <= rows.min() \
        and rows.max() + 1 <= y1 + 1


def test_polygon_area_is_the_shoelace_area():
    tri = Zone.polygon("Z1", [(100, 100), (500, 100), (100, 400)])
    assert tri.area() == pytest.approx(0.5 * 400 * 300)
    assert abs(int(tri.mask().sum()) - tri.area()) <= 0.5 * (400 + 300 + 500) * 0.2
    quad = Zone.polygon("Z2", [(100, 100), (300, 120), (280, 400), (90, 300)])
    assert abs(int(quad.mask().sum()) - quad.area()) / quad.area() < 0.003
    # a concave (L-shaped) polygon: the notch stays dark
    ell = Zone.polygon("Z3", [(100, 100), (300, 100), (300, 200), (200, 200), (200, 300), (100, 300)])
    m = ell.mask()
    assert int(m.sum()) == 200 * 100 + 100 * 100 and not m[250, 250] and m[250, 150] and m[150, 250]
    assert ell.contains(150, 250) and not ell.contains(250, 250)
    # winding order does not matter
    assert np.array_equal(Zone.polygon("Z3", list(reversed(ell.points))).mask(), m)


def test_mask_is_clipped_to_the_dmd_never_wraps():
    z = Zone.rect("Z1", -50, -50, 60, 40)                   # invalid (outside), but rasterisable
    m = z.mask()
    assert int(m.sum()) == 60 * 40 and m[0, 0] and not m[:, 60:].any()


def test_contains_matches_the_mask():
    rng = np.random.default_rng(3)
    for z in (Zone.rect("a", 200, 300, 500, 420, 25), Zone.ellipse("b", 200, 300, 500, 420, 25),
              Zone.polygon("c", [(200, 300), (520, 330), (480, 600), (260, 520)])):
        m = z.mask()
        pts = rng.uniform((150, 250), (560, 650), (400, 2))
        for x, y in pts:
            c, r = int(x), int(y)
            if abs((c + 0.5) - x) > 0.4 or abs((r + 0.5) - y) > 0.4:
                continue                                    # compare at mirror centres only
            assert z.contains(c + 0.5, r + 0.5) == bool(m[r, c])


# ------------------------------------------------------------------ validation

def test_validation_inside_the_dmd_min_size_no_nan():
    assert zone_problems(Zone.rect("Z1", 0, 0, W, H)) == []                 # the whole DMD is fine
    assert any("outside the DMD" in p for p in zone_problems(Zone.rect("Z1", 800, 100, 913, 300)))
    assert any("outside the DMD" in p for p in zone_problems(Zone.rect("Z1", 10, -1, 100, 300)))
    # a rotated shape whose corners leave the field
    assert any("outside the DMD" in p
               for p in zone_problems(Zone.rect("Z1", 0, 0, 200, 200, rotation_deg=45)))
    small = zone_problems(Zone.rect("Z1", 100, 100, 100 + MIN_SIZE_PX - 1, 300))
    assert any("smaller than the minimum" in p for p in small)
    assert zone_problems(Zone.rect("Z1", 100, 100, 100 + MIN_SIZE_PX, 300)) == []
    nan = Zone.rect("Z1", 100, 100, 300, 300)
    nan.center = (float("nan"), 200.0)
    assert any("no NaN" in p for p in zone_problems(nan))
    inf = Zone.polygon("Z1", [(100, 100), (300, float("inf")), (200, 300)])
    assert any("no NaN" in p for p in zone_problems(inf))
    assert any("3.." in p for p in zone_problems(Zone.polygon("Z1", [(1, 1), (50, 50)])))
    bow = Zone.polygon("Z1", [(100, 100), (300, 300), (300, 100), (100, 300)])
    assert any("crosses itself" in p for p in zone_problems(bow))
    assert any("id must be" in p for p in zone_problems(Zone.rect("Z 1", 10, 10, 100, 100)))
    assert any("name must be" in p for p in zone_problems(Zone.rect("Z1", 10, 10, 100, 100,
                                                                    name="a/b")))
    z = Zone.rect("Z1", 10, 10, 100, 100)
    z.shape = "star"
    assert any("shape" in p for p in zone_problems(z))


def test_zoneset_check_lists_every_problem():
    zs = ZoneSet(name="bad name", geometry_id="", zones=[
        Zone.rect("Z1", 10, 10, 100, 100), Zone.rect("Z1", 800, 10, 1000, 100)])
    with pytest.raises(ZoneError) as e:
        zs.check()
    text = "\n".join(e.value.problems)
    for want in ("name must be", "geometry_id is required", "duplicate zone id", "outside the DMD"):
        assert want in text, text
    with pytest.raises(ZoneError, match="at least one zone"):
        ZoneSet(name="empty", geometry_id="g").check()
    many = [Zone.rect(f"Z{i + 1}", 10, 10, 100, 100) for i in range(M.MAX_ZONES + 1)]
    assert any("at most" in p for p in ZoneSet(name="many", geometry_id="g", zones=many).problems())


# ------------------------------------------------------------------ the set: masks, boxes, JSON

def _set() -> ZoneSet:
    return ZoneSet(name="lid_zones", geometry_id="bench_a", fingerprint={"rotate90": False},
                   created="2026-01-02T03:04:05", preset="p", zones=[
                       Zone.rect("Z1", 100, 100, 300, 250, name="part"),
                       Zone.ellipse("Z2", 400, 500, 700, 620, rotation_deg=20, note="rim"),
                       Zone.polygon("Z3", [(120, 700), (360, 720), (300, 1000)])])


def test_union_and_individual_masks_and_boxes():
    zs = _set().check()
    masks = zs.masks()
    assert list(masks) == ["Z1", "Z2", "Z3"] and all(m.shape == DMD_SHAPE for m in masks.values())
    union = zs.mask()
    assert np.array_equal(union, masks["Z1"] | masks["Z2"] | masks["Z3"])
    assert int(union.sum()) == sum(int(m.sum()) for m in masks.values())       # disjoint here
    assert np.array_equal(zs.mask(["Z3", "Z1"]), masks["Z1"] | masks["Z3"])
    boxes = zs.bboxes()
    assert boxes["Z1"] == [100, 100, 300, 250]
    x0, y0, x1, y1 = boxes["Z3"]
    assert (x0, y0) == (120, 700) and x1 <= 361 and y1 <= 1001
    assert [z.id for z in zs.select(["Z3", "Z1"])] == ["Z1", "Z3"]            # the set's order
    with pytest.raises(ZoneError, match="no zone 'Z9'"):
        zs.select(["Z9"])
    with pytest.raises(ZoneError, match="repeated"):
        zs.select(["Z1", "Z1"])


def test_json_round_trip_keeps_shapes_and_masks():
    zs = _set()
    doc = json.loads(json.dumps(zs.to_dict()))
    assert doc["kind"] == "zoneset" and doc["schema"] == M.SCHEMA and "projector px" in doc["units"]
    assert doc["geometry_id"] == "bench_a" and doc["geometry_fp"] == {"rotate90": False}
    assert doc["zones"][0] == {"id": "Z1", "name": "part", "shape": "rect",
                               "center": [200.0, 175.0], "size": [200.0, 150.0],
                               "rotation_deg": 0.0, "note": ""}
    back = ZoneSet.from_dict(doc).check()
    assert back.name == zs.name and back.created == zs.created and back.ids == zs.ids
    for a, b in zip(zs.zones, back.zones):
        assert np.array_equal(a.mask(), b.mask()) and (a.name, a.note) == (b.name, b.note)
    for bad, msg in (({**doc, "kind": "x"}, "not a zone set"), ({**doc, "schema": "v0"}, "schema"),
                     ({**doc, "dmd_shape": ["a", 912]}, "dmd_shape"), ({**doc, "zones": {}}, "list"),
                     ({**doc, "geometry_fp": "x"}, "geometry_fp"),
                     ({**doc, "zones": [{"id": "Z1", "shape": "rect", "center": [1, "a"],
                                         "size": [9, 9]}]}, "center must be"),
                     ({**doc, "zones": [{"id": "Z1", "shape": "rect", "center": [50, 50],
                                         "size": [9, 9], "colour": "red"}]}, "unknown keys"),
                     ({**doc, "zones": [{"id": "Z1", "shape": "polygon",
                                         "points": [[1, 2], [3, None], [5, 6]]}]}, "points must be")):
        with pytest.raises(ZoneError, match=msg):
            ZoneSet.from_dict(bad)


def test_next_id_never_reuses_a_live_number():
    assert next_id([]) == "Z1"
    zs = [Zone.rect("Z1", 0, 0, 9, 9), Zone.rect("Z4", 0, 0, 9, 9), Zone.rect("datum", 0, 0, 9, 9)]
    assert next_id(zs) == "Z5"


# ------------------------------------------------------------------ the text form and recipe zones

def test_zone_set_text_forms():
    assert use.parse_spec("zoneset:lid") == ("lid", None)
    assert use.parse_spec("zoneset:lid:Z1,Z3") == ("lid", ["Z1", "Z3"])
    assert use.parse_spec("zoneset:lid/Z2") == ("lid", ["Z2"])
    assert use.parse_spec(" lid:Z1 , Z3 ", bare=True) == ("lid", ["Z1", "Z3"])
    assert use.spec_text("lid", ["Z1", "Z3"]) == "zoneset:lid:Z1,Z3" and use.spec_text("lid") == "zoneset:lid"
    assert use.is_spec("ZoneSet:lid") and not use.is_spec("F10:J12") and not use.is_spec(None)
    for bad in ("lid", "zoneset:", "zoneset:li d", "zoneset:lid:", "zoneset:lid:Z1,,Z2",
                "zoneset:lid:Z1,Z1", "zoneset:../x"):
        with pytest.raises(ZoneError):
            use.parse_spec(bad)
    assert use.parse_ref("zoneset:lid_zones/Z1") == ("lid_zones", "Z1")
    for bad in ("lid/Z1", "zoneset:lid", "zoneset:lid/Z1,Z2", 3, "zoneset:lid/"):
        with pytest.raises(ValueError, match="zoneset:<set>/<zone id>"):
            use.parse_ref(bad)


def test_recipe_zones_are_projector_px_shapes_the_schema_accepts():
    from engine.recipes import schema
    zs = _set()
    zs.zones.append(Zone.ellipse("Z4", 500, 800, 700, 1000))                  # round: a circle
    zs.zones.append(Zone.rect("Z5", 500, 100, 700, 200, rotation_deg=30))     # turned: a polygon
    out = {z["id"]: z for z in use.recipe_zones(zs)}
    assert out["Z1"]["shape"] == "rect" and out["Z1"]["coords"] == [100.0, 100.0, 300.0, 250.0]
    assert out["Z1"]["units"] == "projector_px" and out["Z1"]["label"] == "part"
    assert out["Z4"]["shape"] == "circle" and out["Z4"]["coords"] == {"center": [600.0, 900.0],
                                                                    "radius": 100.0}
    assert out["Z2"]["shape"] == "polygon" and len(out["Z2"]["coords"]) == M.ELLIPSE_PTS
    assert out["Z5"]["shape"] == "polygon" and len(out["Z5"]["coords"]) == 4
    assert out["Z3"]["coords"] == [[120.0, 700.0], [360.0, 720.0], [300.0, 1000.0]]
    doc = {"schema_version": 1, "name": "painted", "preset": "bench_9in_metric",
           "scan": {"kind": "fast", "axis": "xy"}, "zones": list(out.values())}
    r = schema.validate(doc)
    assert [z.id for z in r.zones] == ["Z1", "Z2", "Z3", "Z4", "Z5"]
    # the recipe's own raster of each zone is the zone's mask to within its outline
    from engine.recipes.geometry import dmd_zone_mask
    for z, spec in zip(zs.zones, r.zones):
        a, b = z.mask(), dmd_zone_mask(spec)
        assert (a ^ b).sum() <= 0.03 * a.sum() + 700, (z.id, int((a ^ b).sum()))
    refs = use.recipe_zones(zs, ["Z1"], by_ref=True, checks=[{"type": "flatness",
                                                             "params": {"tol_mm": 0.3}}])
    assert refs == [{"id": "Z1", "label": "part", "ref": "zoneset:lid_zones/Z1",
                     "checks": [{"type": "flatness", "params": {"tol_mm": 0.3}}]}]

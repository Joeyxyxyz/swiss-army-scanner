"""The zone painter without a display: PainterModel (every mouse / key interaction as a state
machine), render(state) (the DMD image), the window shell's loop on a scripted MockWindow,
and the projector window's mouse / key translation on a fake cv2 (no window is ever opened)."""

import json
import math
import threading

import numpy as np
import pytest

from engine import rules
from engine.capture import projector as PJ
from engine.capture.projector import HDMIWindow, MockWindow
from engine.zones import DMD_SHAPE, MIN_SIZE_PX, PainterModel, Zone, ZoneError, render
from engine.zones import painter as P
from engine.zones import session as S

H, W = DMD_SHAPE


def drag(m, p0, p1, **kw):
    return m.handle({"type": "drag", "from": list(p0), "to": list(p1), **kw})


def click(m, p, **kw):
    return m.handle({"type": "click", "at": list(p), **kw})


def bbox(z):
    return tuple(round(v, 6) for v in z.bbox())


def three() -> PainterModel:
    """A rectangle Z1, an ellipse Z2 and a polygon Z3, nothing selected."""
    m = PainterModel(name="t")
    drag(m, (100, 200), (400, 500))
    m.on_key("e")
    drag(m, (500, 200), (800, 400))
    m.on_key("p")
    for p in ((150, 650), (420, 680), (380, 950)):
        click(m, p)
    m.handle({"type": "dblclick", "at": [140, 900]})
    m.on_key("r")
    click(m, (860, 1000))                                   # empty space: deselect
    assert [z.id for z in m.state.zones] == ["Z1", "Z2", "Z3"] and m.state.selected is None
    return m


# ------------------------------------------------------------------ create

def test_drag_creates_a_rectangle_by_default():
    m = PainterModel()
    assert m.state.tool == "rect" and m.state.zones == []
    m.on_mouse_down(100, 200)
    m.on_mouse_move(250, 300)
    assert m.state.preview is not None and bbox(m.state.preview) == (100, 200, 250, 300)
    assert m.state.zones == []                              # nothing exists before the release
    m.on_mouse_up(400, 500)
    (z,) = m.state.zones
    assert (z.id, z.shape, z.rotation_deg) == ("Z1", "rect", 0.0) and bbox(z) == (100, 200, 400, 500)
    assert m.state.selected == "Z1" and m.state.preview is None and m.state.done is None
    drag(m, (800, 900), (600, 700))                         # dragged up and to the left
    assert bbox(m.state.zones[1]) == (600, 700, 800, 900) and m.state.zones[1].id == "Z2"


def test_a_click_or_a_tiny_drag_creates_nothing():
    m = PainterModel()
    click(m, (300, 300))
    drag(m, (300, 300), (300 + MIN_SIZE_PX - 1, 500))
    assert m.state.zones == [] and "too small" in m.state.message
    drag(m, (300, 300), (300 + MIN_SIZE_PX, 300 + MIN_SIZE_PX))
    assert len(m.state.zones) == 1


def test_e_creates_ellipses_and_r_goes_back_to_rectangles():
    m = PainterModel()
    m.on_key("E")
    drag(m, (200, 300), (500, 420))
    z = m.state.zones[0]
    assert z.shape == "ellipse" and bbox(z) == (200, 300, 500, 420) and tuple(z.size) == (300, 120)
    drag(m, (600, 300), (700, 400))
    assert m.state.zones[1].shape == "ellipse"              # the tool stays until changed
    m.on_key("r")
    drag(m, (600, 600), (700, 700))
    assert [z.shape for z in m.state.zones] == ["ellipse", "ellipse", "rect"]


@pytest.mark.parametrize("close", ["dblclick", "enter", "first_corner"])
def test_p_clicks_a_polygon_closed_by_double_click_enter_or_the_first_corner(close):
    m = PainterModel()
    m.on_key("p")
    pts = [(100, 100), (400, 120), (380, 400), (90, 300)]
    for p in pts[:3]:
        click(m, p)
    assert m.state.poly == [tuple(map(float, p)) for p in pts[:3]] and m.state.zones == []
    if close == "dblclick":
        # the window sends down / up / dblclick / up for a double click
        click(m, pts[3])
        m.handle({"type": "dblclick", "at": list(pts[3])})
        m.on_mouse_up(*pts[3])
    elif close == "enter":
        click(m, pts[3])
        m.on_key("enter")
    else:
        click(m, pts[3])
        click(m, (103, 102))                                # on the first corner
    (z,) = m.state.zones
    assert z.shape == "polygon" and z.points == [tuple(map(float, p)) for p in pts]
    assert m.state.poly == [] and m.state.selected == "Z1" and m.state.tool == "polygon"


def test_a_polygon_that_is_not_a_zone_is_not_added():
    m = PainterModel()
    m.on_key("p")
    click(m, (100, 100))
    click(m, (300, 300))
    m.on_key("enter")                                       # two corners
    assert m.state.zones == [] and "not added" in m.state.message
    for p in ((100, 100), (300, 300), (300, 100), (100, 300)):      # a bow-tie
        click(m, p)
    m.on_key("enter")
    assert m.state.zones == [] and "crosses itself" in m.state.message
    # backspace / undo take the last corner back while drawing
    for p in ((100, 100), (300, 100), (300, 300)):
        click(m, p)
    m.on_key("backspace")
    m.on_key("z")
    assert m.state.poly == [(100.0, 100.0)]


def test_shift_drag_draws_inside_an_existing_zone():
    m = PainterModel()
    drag(m, (100, 100), (600, 600))
    drag(m, (200, 200), (300, 300))                         # starts inside Z1: that moves Z1
    assert len(m.state.zones) == 1 and bbox(m.state.zones[0]) == (200, 200, 700, 700)
    drag(m, (300, 300), (400, 400), shift=True)
    assert len(m.state.zones) == 2 and bbox(m.state.zones[1]) == (300, 300, 400, 400)


# ------------------------------------------------------------------ select, hit-testing

def test_hit_testing():
    m = three()
    assert m.hit_test(250, 350) == ("Z1", ("inside",))
    assert m.hit_test(650, 300) == ("Z2", ("inside",))
    assert m.hit_test(505, 205) is None                     # the ellipse's box corner is outside it
    assert m.hit_test(300, 800) == ("Z3", ("inside",))
    assert m.hit_test(860, 100) is None
    click(m, (250, 350))
    assert m.state.selected == "Z1"
    # the selected rectangle: its 8 handles, then its edges
    assert m.hit_test(100, 200) == ("Z1", ("handle", -1, -1))
    assert m.hit_test(403, 497) == ("Z1", ("handle", 1, 1))
    assert m.hit_test(250, 203) == ("Z1", ("handle", 0, -1))
    assert m.hit_test(100, 350) == ("Z1", ("handle", -1, 0))
    assert m.hit_test(180, 202) == ("Z1", ("handle", 0, -1))        # the top edge, off the handles
    assert m.hit_test(398, 430) == ("Z1", ("handle", 1, 0))         # the right edge
    click(m, (650, 300))
    assert m.state.selected == "Z2"
    assert m.hit_test(500, 200) == ("Z2", ("handle", -1, -1))       # its box corner is a handle now
    click(m, (300, 800))
    assert m.hit_test(150, 650) == ("Z3", ("vertex", 0)) and m.hit_test(382, 948) == ("Z3", ("vertex", 2))


def test_the_smallest_zone_under_the_cursor_is_selected():
    m = PainterModel()
    drag(m, (100, 100), (600, 600))
    drag(m, (300, 300), (400, 400), shift=True)
    click(m, (850, 1000))
    click(m, (350, 350))
    assert m.state.selected == "Z2"                         # the small one inside the big one
    click(m, (150, 150))
    assert m.state.selected == "Z1"
    m.on_key("tab")
    assert m.state.selected == "Z2"
    m.on_key("tab")
    assert m.state.selected == "Z1"
    m.handle({"type": "down", "at": [850, 1000], "button": "right"})
    assert m.state.selected is None


# ------------------------------------------------------------------ move

@pytest.mark.parametrize("zid,inside", [("Z1", (250, 350)), ("Z2", (650, 300)), ("Z3", (300, 800))])
def test_drag_inside_moves_each_shape_and_undo_puts_it_back(zid, inside):
    m = three()
    z0 = m.state.zone(zid).copy()
    drag(m, inside, (inside[0] + 37, inside[1] - 21))
    z1 = m.state.zone(zid)
    assert m.state.selected == zid
    assert bbox(z1) == tuple(round(v + d, 6) for v, d in zip(z0.bbox(), (37, -21, 37, -21)))
    assert z1.area() == pytest.approx(z0.area()) and z1.shape == z0.shape
    assert [z.id for z in m.state.zones] == ["Z1", "Z2", "Z3"]      # ids and order kept
    m.on_key("z")
    assert bbox(m.state.zone(zid)) == bbox(z0)


def test_a_click_that_only_selects_is_not_an_undo_step():
    m = three()
    click(m, (250, 350))
    click(m, (650, 300))
    m.on_key("ctrl+z")                                      # undoes the last real edit: Z3
    assert [z.id for z in m.state.zones] == ["Z1", "Z2"]


# ------------------------------------------------------------------ resize

def test_resize_a_rectangle_by_corner_edge_handle_and_edge():
    m = three()
    click(m, (250, 350))
    drag(m, (400, 500), (450, 520))                         # the bottom-right corner
    assert bbox(m.state.zone("Z1")) == (100, 200, 450, 520)
    drag(m, (100, 360), (150, 300))                         # the left edge's handle: width only
    assert bbox(m.state.zone("Z1")) == (150, 200, 450, 520)
    drag(m, (180, 201), (180, 260))                         # the top edge, off its handles
    assert bbox(m.state.zone("Z1")) == (150, 260, 450, 520)
    # dragged past the opposite edge: it stops at the minimum size, it never flips
    drag(m, (450, 390), (20, 390))
    assert bbox(m.state.zone("Z1")) == (150, 260, 150 + MIN_SIZE_PX, 520)
    for _ in range(4):
        m.on_key("z")
    assert bbox(m.state.zone("Z1")) == (100, 200, 400, 500)


def test_resize_an_ellipse_and_a_rotated_rectangle_keeps_the_opposite_corner():
    m = three()
    click(m, (650, 300))
    drag(m, (800, 400), (840, 460))
    z = m.state.zone("Z2")
    assert z.shape == "ellipse" and bbox(z) == (500, 200, 840, 460)
    # a rectangle turned 30 deg: pull its (+u, +v) corner 50 px along its own u axis
    m2 = PainterModel([Zone.rect("Z1", 300, 350, 500, 450, rotation_deg=30)])
    click(m2, (400, 400))
    z0 = m2.state.zone("Z1").copy()
    fixed = z0.world(-100, -50)
    grab = z0.world(100, 50)
    c, s = math.cos(math.radians(30)), math.sin(math.radians(30))
    drag(m2, grab, (grab[0] + 50 * c, grab[1] + 50 * s), steps=8)
    z1 = m2.state.zone("Z1")
    assert z1.size == pytest.approx((250, 100), abs=1e-6) and z1.rotation_deg == 30
    assert z1.world(-125, -50) == pytest.approx(fixed, abs=1e-6)


def test_drag_a_polygon_corner():
    m = three()
    click(m, (300, 800))
    before = list(m.state.zone("Z3").points)
    drag(m, (420, 680), (520, 640))
    after = m.state.zone("Z3").points
    assert after[1] == (520.0, 640.0) and after[0] == before[0] and after[2:] == before[2:]
    # a position that would make it cross itself is not taken: the last good shape stays
    drag(m, (520, 640), (60, 1050))
    assert not __import__("engine.zones.model", fromlist=["x"]).self_crossing(m.state.zone("Z3").points)


# ------------------------------------------------------------------ rotate

def test_rotate_with_the_wheel_and_the_bracket_keys():
    m = three()
    click(m, (250, 350))
    m.on_wheel(250, 350, 1)
    assert m.state.zone("Z1").rotation_deg == P.WHEEL_DEG
    m.on_wheel(250, 350, -3)
    assert m.state.zone("Z1").rotation_deg == -2 * P.WHEEL_DEG
    m.handle({"type": "wheel", "at": [250, 350], "delta": 1, "shift": True})
    assert m.state.zone("Z1").rotation_deg == -2 * P.WHEEL_DEG + P.FINE_DEG
    for k, want in (("]", 2), ("]", 3), ("[", 2), ("}", 17), ("{", 2)):
        m.on_key(k)
        assert m.state.zone("Z1").rotation_deg == pytest.approx(-2 * P.WHEEL_DEG + want)
    assert tuple(m.state.zone("Z1").size) == (300, 300)             # turning changes no size
    click(m, (650, 300))
    m.on_key("]")
    assert m.state.zone("Z2").shape == "ellipse" and m.state.zone("Z2").rotation_deg == 1
    # a polygon turns about its centroid: same area, same centroid, corners moved
    click(m, (300, 800))
    z0 = m.state.zone("Z3").copy()
    m.on_wheel(300, 800, 2)
    z1 = m.state.zone("Z3")
    assert z1.area() == pytest.approx(z0.area()) and z1.centroid() == pytest.approx(z0.centroid())
    assert z1.points != z0.points
    m.on_key("z")
    assert m.state.zone("Z3").points == z0.points
    # with nothing selected the wheel changes nothing and says why
    click(m, (860, 1000))
    m.on_wheel(860, 1000, 1)
    assert "select a zone" in m.state.message


# ------------------------------------------------------------------ delete, undo, names

@pytest.mark.parametrize("key", ["delete", "backspace"])
@pytest.mark.parametrize("zid,inside", [("Z1", (250, 350)), ("Z2", (650, 300)), ("Z3", (300, 800))])
def test_delete_removes_the_selected_shape_and_undo_restores_it(key, zid, inside):
    m = three()
    click(m, inside)
    z0 = m.state.zone(zid).copy()
    m.on_key(key)
    assert zid not in [z.id for z in m.state.zones] and m.state.selected is None
    assert len(m.state.zones) == 2
    m.on_key("z")
    assert [z.id for z in m.state.zones] == ["Z1", "Z2", "Z3"] and m.state.selected == zid
    assert bbox(m.state.zone(zid)) == bbox(z0) and m.state.zone(zid).shape == z0.shape


def test_undo_walks_all_the_way_back_and_ids_are_not_reused_while_alive():
    m = PainterModel()
    drag(m, (100, 100), (200, 200))
    drag(m, (300, 100), (400, 200))
    drag(m, (500, 100), (600, 200))
    click(m, (350, 150))
    m.on_key("delete")                                      # Z2 gone
    drag(m, (300, 400), (400, 500))
    assert [z.id for z in m.state.zones] == ["Z1", "Z3", "Z4"]      # never renumbered
    drag(m, (150, 150), (160, 300))                         # move Z1
    m.on_key("]")
    m.on_key("n")
    assert m.state.zone("Z1").name == P.LABELS[1]
    for want in (["Z1", "Z3", "Z4"],) * 3 + (["Z1", "Z3"], ["Z1", "Z2", "Z3"], ["Z1", "Z2"],
                                             ["Z1"], []):
        m.on_key("z")
        assert [z.id for z in m.state.zones] == want
    m.on_key("z")
    assert m.state.zones == [] and "nothing to undo" in m.state.message and m.edits == 0


def test_n_cycles_the_label_and_shift_n_goes_back():
    m = three()
    click(m, (250, 350))
    seen = []
    for _ in range(len(P.LABELS)):
        m.on_key("n")
        seen.append(m.state.zone("Z1").name)
    assert seen == list(P.LABELS[1:]) + [""]
    m.on_key("N")
    assert m.state.zone("Z1").name == P.LABELS[-1]
    own = PainterModel([Zone.rect("Z1", 10, 10, 100, 100)], labels=("", "lid", "base"))
    click(own, (50, 50))
    own.on_key("n")
    own.on_key("n")
    assert own.state.zone("Z1").name == "base"


def test_arrow_keys_nudge_one_px():
    m = three()
    click(m, (250, 350))
    for k in ("left", "left", "up", "right", "down", "down"):
        m.on_key(k)
    assert bbox(m.state.zone("Z1")) == (99, 201, 399, 501)


# ------------------------------------------------------------------ clamping at the DMD borders

def test_shapes_are_clamped_to_the_dmd():
    m = PainterModel()
    drag(m, (800, 1000), (5000, 9000))                      # dragged off the field
    assert bbox(m.state.zones[0]) == (800, 1000, W, H)
    drag(m, (100, 100), (-300, -50), shift=True)
    assert bbox(m.state.zones[1]) == (0, 0, 100, 100)
    # moved against the border: it stops there, whole
    drag(m, (50, 50), (-400, 700))
    assert bbox(m.state.zone("Z2")) == (0, 650, 100, 750)
    drag(m, (50, 700), (50, 5000))
    assert bbox(m.state.zone("Z2")) == (0, H - 100, 100, H)
    # resized into the border: the edge stops at it
    drag(m, (100, H - 50), (3000, H - 50))
    assert bbox(m.state.zone("Z2")) == (0, H - 100, W, H)
    # a nudge at the border does nothing (and is no undo step)
    n = m.edits
    m.on_key("down")
    assert bbox(m.state.zone("Z2")) == (0, H - 100, W, H) and m.edits == n
    for z in m.state.zones:
        assert __import__("engine.zones", fromlist=["x"]).zone_problems(z) == []


def test_rotation_at_the_border_shifts_the_shape_back_in_or_is_refused():
    m = PainterModel([Zone.rect("Z1", 0, 0, 300, 100)])
    click(m, (150, 50))
    m.on_key("}")
    m.on_key("}")
    m.on_key("}")                                           # 45 deg in the top-left corner
    z = m.state.zone("Z1")
    x0, y0, x1, y1 = z.bbox()
    assert z.rotation_deg == 45 and x0 >= -1e-6 and y0 >= -1e-6 and tuple(z.size) == (300, 100)
    wide = PainterModel([Zone.rect("Z1", 0, 400, W, 700)])  # as wide as the DMD: no room to turn
    click(wide, (450, 550))
    wide.on_key("]")
    assert wide.state.zone("Z1").rotation_deg == 0 and "can't turn" in wide.state.message
    assert wide.edits == 0


def test_a_polygon_corner_cannot_leave_the_dmd():
    m = PainterModel()
    m.on_key("p")
    for p in ((700, 100), (5000, -40), (800, 400)):
        click(m, p)
    m.on_key("enter")
    assert m.state.zones[0].points[1] == (float(W), 0.0)


# ------------------------------------------------------------------ confirm / cancel

def test_space_confirms_only_a_finished_non_empty_session():
    m = PainterModel()
    m.on_key("space")
    assert m.state.done is None and "no zones yet" in m.state.message
    m.on_key("p")
    click(m, (100, 100))
    click(m, (300, 100))
    m.on_key(" ")
    assert m.state.done is None and "finish the shape first" in m.state.message
    click(m, (300, 300))
    m.on_key("enter")
    m.on_key("space")
    assert m.state.done == "save" and len(m.zones()) == 1
    # a finished session takes no more input
    m.on_key("delete")
    drag(m, (500, 500), (700, 700))
    assert len(m.state.zones) == 1 and m.state.done == "save"


def test_esc_drops_the_shape_in_progress_first_then_cancels():
    m = PainterModel()
    drag(m, (100, 100), (300, 300))
    m.on_key("p")
    click(m, (500, 500))
    click(m, (700, 500))
    m.on_key("escape")
    assert m.state.done is None and m.state.poly == [] and len(m.state.zones) == 1
    m.on_key("escape")
    assert m.state.done == "cancel" and "nothing saved" in m.state.message
    # Esc in the middle of a move puts the zone back first
    m = PainterModel()
    drag(m, (100, 100), (300, 300))
    m.on_mouse_down(200, 200)
    m.on_mouse_move(500, 600)
    assert bbox(m.state.zones[0]) == (400, 500, 600, 700)
    m.on_key("esc")
    assert bbox(m.state.zones[0]) == (100, 100, 300, 300) and m.state.done is None
    # a right click drops a polygon being drawn
    m.on_key("p")
    click(m, (500, 500))
    m.handle({"type": "down", "at": [600, 600], "button": "right"})
    assert m.state.poly == []


def test_a_release_outside_the_window_still_ends_the_drag():
    """The projector is one of two monitors: a button released over the other one never
    arrives as an "up". The next move without the button held ends the drag where it is."""
    m = PainterModel()
    m.handle({"type": "down", "at": [100, 100]})
    m.handle({"type": "move", "at": [300, 250], "left": True})
    assert m.state.zones == [] and m.state.preview is not None
    m.handle({"type": "move", "at": [320, 260], "left": False})        # back, button up
    assert bbox(m.state.zones[0]) == (100, 100, 320, 260) and m.state.preview is None
    m.handle({"type": "move", "at": [500, 500], "left": False})
    assert bbox(m.state.zones[0]) == (100, 100, 320, 260)              # no longer following
    # the same for a move; and a new press always starts clean
    m.handle({"type": "down", "at": [200, 200]})
    m.handle({"type": "move", "at": [260, 240], "left": True})
    m.handle({"type": "move", "at": [270, 250], "left": False})
    assert bbox(m.state.zones[0]) == (170, 150, 390, 310)
    m.on_mouse_down(600, 600)
    m.on_mouse_move(700, 700)
    m.on_mouse_down(250, 250)                                           # the release was lost
    assert m.state.preview is None and m.state.selected == "Z1" and len(m.state.zones) == 1


def test_grid_and_legend_toggles_and_unbound_keys():
    m = PainterModel()
    assert not m.state.show_grid and m.state.show_legend
    m.on_key("g")
    m.on_key("h")
    assert m.state.show_grid and not m.state.show_legend
    m.dirty = False
    m.on_key("q")
    assert m.dirty is False and m.state.done is None        # an unbound key changes nothing


def test_the_painter_starts_from_saved_zones_and_refuses_invalid_ones():
    m = PainterModel([Zone.rect("Z1", 10, 10, 100, 100), Zone.rect("Z7", 200, 10, 300, 100)])
    drag(m, (400, 400), (500, 500))
    assert [z.id for z in m.state.zones] == ["Z1", "Z7", "Z8"]
    with pytest.raises(ZoneError, match="outside the DMD"):
        PainterModel([Zone.rect("Z1", 800, 10, 1000, 100)])


def test_malformed_scripted_events_are_refused():
    m = PainterModel()
    for ev in ({"type": "jump"}, {"type": "drag", "from": [1, 2]}, {"type": "click", "at": [1]},
               {"type": "click", "at": [1, float("nan")]}, {"type": "key"}, {"at": [1, 2]},
               {"type": "wheel", "at": [1, 2], "delta": "up"}, "space"):
        with pytest.raises(ValueError):
            m.handle(ev)
    with pytest.raises(ValueError, match="list of events"):
        P.check_events({"type": "key"})


def test_scripted_demo_session_and_events_file(tmp_path):
    m = PainterModel(name="demo")
    for ev in P.DEMO_EVENTS:
        m.handle(ev)
    assert m.state.done == "save"
    assert [(z.id, z.shape) for z in m.state.zones] == [("Z1", "rect"), ("Z2", "rect"),
                                                        ("Z3", "ellipse"), ("Z4", "polygon")]
    assert m.state.zones[0].name == P.LABELS[1] and m.state.zones[1].rotation_deg == 15
    f = tmp_path / "session.json"
    f.write_text(json.dumps({"events": list(P.DEMO_EVENTS)}))
    assert P.load_events(f) == [dict(e) for e in P.DEMO_EVENTS]
    f.write_text(json.dumps(list(P.DEMO_EVENTS)))
    assert len(P.load_events(f)) == len(P.DEMO_EVENTS)
    f.write_text("{ not json")
    with pytest.raises(ValueError, match="not valid JSON"):
        P.load_events(f)
    with pytest.raises(ValueError, match="events file"):
        P.load_events(tmp_path / "missing.json")
    snap = m.state.to_dict()
    assert json.loads(json.dumps(snap))["zones"][3]["shape"] == "polygon" and snap["done"] == "save"


# ------------------------------------------------------------------ the picture

def test_render_draws_outlines_where_the_shapes_are_on_black():
    m = three()
    img = render(m.state)
    assert img.shape == DMD_SHAPE and img.dtype == np.uint8
    # the rectangle's four edges are lit, its inside and the field around it are dark
    for x, y in ((250, 200), (250, 500), (100, 350), (400, 350)):
        assert img[y - 1:y + 2, x - 1:x + 2].max() >= P.OUTLINE - 40, (x, y)
    assert img[230:470, 130:370].max() == 0 and img[520:600, 100:400].max() == 0
    # the ellipse: lit on its outline (ends of its axes), dark at its centre and box corners
    for x, y in ((500, 300), (800, 300), (650, 200), (650, 400)):
        assert img[y - 2:y + 3, x - 2:x + 3].max() >= P.OUTLINE - 40, (x, y)
    assert img[260:340, 600:700].max() == 0 and img[205:215, 505:515].max() == 0
    # the polygon's corners
    for x, y in m.state.zone("Z3").points:
        assert img[int(y) - 2:int(y) + 3, int(x) - 2:int(x) + 3].max() >= P.OUTLINE - 40
    # every zone's id is written next to it (something lit just above its box)
    for z in m.state.zones:
        x0, y0, _x1, _y1 = (int(v) for v in z.bbox())
        assert img[y0 - 26:y0 - 3, x0:x0 + 40].max() >= P.OUTLINE - 40, z.id


def test_render_marks_the_selection_with_handles_and_the_cursor_with_a_crosshair():
    m = three()
    plain = render(m.state)
    click(m, (250, 350))
    m.on_mouse_move(600, 700)
    img = render(m.state)
    for hx, hy in ((100, 200), (250, 200), (400, 200), (400, 350), (400, 500), (250, 500),
                   (100, 500), (100, 350)):
        assert (img[hy - 4:hy + 5, hx - 4:hx + 5] == P.BRIGHT).all(), (hx, hy)
    assert (plain[346:355, 96:105] < P.BRIGHT).any()        # no handle while it is not selected
    assert img[200, 180] == P.BRIGHT and plain[200, 180] <= P.OUTLINE       # brighter + thicker
    assert img[203, 180] > 0 or img[197, 180] > 0
    # the crosshair: bright arms with a gap at the very centre, dim full-length hairlines
    assert img[700, 610] == P.BRIGHT and img[690, 600] == P.BRIGHT and img[700, 600] <= P.HAIR_LEVEL
    assert img[700, 50] == P.HAIR_LEVEL and img[50, 600] == P.HAIR_LEVEL
    assert plain[700, 50] == 0


def test_render_shows_the_shape_being_drawn():
    m = PainterModel()
    m.on_mouse_down(100, 200)
    m.on_mouse_move(300, 400)
    img = render(m.state)
    assert img[200, 200] == P.BRIGHT and img[300, 100] == P.BRIGHT and img[300, 200] <= P.HAIR_LEVEL
    m.on_mouse_up(300, 400)
    m.on_key("p")
    click(m, (500, 500))
    click(m, (700, 520))
    m.on_mouse_move(650, 800)
    img = render(m.state)
    assert img[510, 600] >= P.OUTLINE - 60                  # the edge already clicked
    assert img[497:504, 497:504].max() == P.BRIGHT          # the first corner's marker
    assert img[660, 675] > 0 or img[658:663, 672:679].max() > 0     # the rubber band to the cursor


def test_legend_strip_is_at_the_bottom_edge_and_can_be_hidden():
    m = three()
    lh = P.legend_height(m.state)
    assert 30 <= lh <= 80
    img = render(m.state)
    strip = img[H - lh:, :]
    assert (strip[0] == P.GRID_LEVEL).all()                 # its separator line
    assert strip[1:].max() >= P.LEGEND_LEVEL - 60 and (strip[1:] > 0).mean() < 0.15
    assert "tool RECT" in P.status_line(m.state) and "3 zone(s)" in P.status_line(m.state)
    for word in ("SPACE save", "ESC cancel", "G grid", "Z undo", "N name"):
        assert word in " ".join(P.LEGEND)
    m.on_key("h")
    hidden = render(m.state)[H - lh:, :]                    # only the cursor's hairline is left
    assert P.legend_height(m.state) == 0 and hidden.max() <= P.HAIR_LEVEL
    assert (hidden > 0).sum() <= lh
    # every key in the table does something (the table is what the docs print)
    assert {k for k, _ in P.KEYS} >= {"Space", "Esc", "G", "N", "Z, Ctrl+Z", "Delete, Backspace"}


def test_render_never_floods_the_field():
    m = PainterModel(name="many")
    for j in range(6):
        for i in range(5):
            tool = ("r", "e")[(i + j) % 2]
            m.on_key(tool)
            drag(m, (20 + i * 175, 20 + j * 175), (170 + i * 175, 170 + j * 175))
    m.on_key("g")
    m.on_mouse_move(456, 570)
    assert len(m.state.zones) == 30
    img = render(m.state)
    # 30 shapes edge to edge, the grid, the legend and the crosshair: still lines on black
    assert (img >= 128).mean() < 0.08 and img.mean() < 0.08 * 255 and (img > 0).mean() < 0.15
    demo = PainterModel(name="demo")
    for ev in P.DEMO_EVENTS:
        demo.handle(ev)
    assert (render(demo.state) >= 128).mean() < 0.02        # a usual session: under 2 % lit
    # the grid underlay alone is dim and thin
    g = PainterModel()
    g.on_key("g")
    g.on_key("h")
    grid = render(g.state)
    assert 0 < grid.max() <= P.GRID_TEXT and (grid > 0).mean() < 0.04
    assert (grid[:, 96] == P.GRID_LEVEL).mean() > 0.9 and (grid[96, :] == P.GRID_LEVEL).mean() > 0.9
    # an empty painter: black but for the legend
    empty = render(PainterModel().state)
    assert empty[:H - P.legend_height(PainterModel().state)].max() == 0


# ------------------------------------------------------------------ the window shell (headless)

def test_shell_replays_a_scripted_session_on_the_mock_window():
    m = PainterModel(name="demo")
    w = MockWindow()
    w.enable_mouse()
    w.feed(P.DEMO_EVENTS)
    done = S.run_painter(m, w, mock=True)
    assert done == "save" and len(m.zones()) == 4 and w.mouse_enabled
    assert w.polled == len(P.DEMO_EVENTS)                   # it stopped at Space, not at the end
    assert w.count >= len(P.DEMO_EVENTS) and w.held_s == 0.0
    assert np.array_equal(w.last, render(m.state))          # what is on the DMD is the final state
    assert w.last.dtype == np.uint8 and (w.last >= 128).mean() < 0.05


def test_shell_sends_the_picture_inverted_on_an_inverting_path_and_twice_when_asked():
    m = PainterModel()
    w = MockWindow()
    w.feed([{"type": "drag", "from": [100, 100], "to": [300, 300]}, {"type": "key", "key": "space"}])
    S.run_painter(m, w, mock=True, inverted=True, twice=True)
    assert np.array_equal(w.last, 255 - render(m.state))
    assert (w.last == 255).mean() > 0.9                     # sent bright = physically dark there
    assert w.count % 2 == 0


def test_shell_cancels_when_the_script_ends_without_space():
    m = PainterModel()
    w = MockWindow()
    w.feed([{"type": "drag", "from": [100, 100], "to": [300, 300]}])
    assert S.run_painter(m, w, mock=True) == "cancel"
    assert "without Space" in m.state.message and len(m.state.zones) == 1


def test_shell_refuses_the_real_dmd_without_the_rig_lock():
    m = PainterModel()
    w = MockWindow()
    w.feed(P.DEMO_EVENTS)
    with pytest.raises(rules.RuleViolation, match="rig lock"):
        S.run_painter(m, w, mock=False)
    assert w.count == 0 and w.polled == 0 and m.state.zones == []    # nothing shown, nothing read
    assert S.run_painter(m, w, mock=False, rig_locked=True) == "save"

    class NoPoll:
        def show(self, img):
            raise AssertionError("shown")
    with pytest.raises(rules.RuleViolation, match="poll"):
        S.run_painter(PainterModel(), NoPoll(), mock=True)


def test_shell_ends_on_abort_and_on_idle_timeout_saving_nothing():
    class Quiet(MockWindow):
        def poll(self, wait_ms=15):
            self.polled += 1
            return []                                       # a real window: nothing happened

    ev = threading.Event()
    w = Quiet()
    orig = w.poll

    def poll(ms=15):
        if w.polled == 3:
            ev.set()
        return orig(ms)
    w.poll = poll
    m = PainterModel()
    assert S.run_painter(m, w, mock=True, cancel=ev) == "cancel" and "aborted" in m.state.message
    t = {"now": 0.0}

    def clock():
        t["now"] += 1.0
        return t["now"]
    m, w = PainterModel(), Quiet()
    assert S.run_painter(m, w, mock=True, idle_timeout_s=20.0, clock=clock) == "cancel"
    assert "no input for 20 s" in m.state.message and 15 <= w.polled <= 25


# ------------------------------------------------------------------ the window's mouse + keys

def test_key_names():
    names = {27: "escape", 32: "space", 13: "enter", 8: "backspace", 9: "tab", 26: "ctrl+z",
             0x2E0000: "delete", 0x250000: "left", 0x260000: "up", 0x270000: "right",
             0x280000: "down", 0xFFFF: "delete", 0xFF51: "left", ord("e"): "e", ord("N"): "N",
             ord("["): "[", ord("}"): "}", 0x100000 | ord("g"): "g"}
    for code, want in names.items():
        assert PJ.key_name(code) == want, hex(code)
    for code in (-1, None, 0x700000, 0):
        assert PJ.key_name(code) is None


@pytest.mark.parametrize("rot", [False, True])
def test_window_px_map_back_to_pattern_px(rot):
    """Where the mock window physically shows a pattern px is where a click maps back from."""
    w = MockWindow(rotate90=rot)
    for col, row in ((300, 700), (10, 20), (900, 1100)):
        pat = np.zeros(DMD_SHAPE, np.uint8)
        pat[row, col] = 255
        w.show(pat)
        ys, xs = np.nonzero(w.physical)
        c, r = PJ.window_to_pattern(xs.mean(), ys.mean(), w.size, DMD_SHAPE, rot)
        assert abs(c - col) <= 1.0 and abs(r - row) <= 1.0, (rot, col, row, c, r)


def test_the_event_codes_are_opencvs():
    """The window reads cv2's mouse events by number: they must be the installed OpenCV's."""
    import cv2
    want = {cv2.EVENT_MOUSEMOVE: ("move", None), cv2.EVENT_LBUTTONDOWN: ("down", "left"),
            cv2.EVENT_RBUTTONDOWN: ("down", "right"), cv2.EVENT_MBUTTONDOWN: ("down", "middle"),
            cv2.EVENT_LBUTTONUP: ("up", "left"), cv2.EVENT_RBUTTONUP: ("up", "right"),
            cv2.EVENT_MBUTTONUP: ("up", "middle"), cv2.EVENT_LBUTTONDBLCLK: ("dblclick", "left"),
            cv2.EVENT_MOUSEWHEEL: ("wheel", None)}
    assert PJ._MOUSE_EVENTS == want
    assert (PJ._FLAG_LEFT, PJ._FLAG_CTRL, PJ._FLAG_SHIFT) == (
        cv2.EVENT_FLAG_LBUTTON, cv2.EVENT_FLAG_CTRLKEY, cv2.EVENT_FLAG_SHIFTKEY)
    for fn in ("setMouseCallback", "waitKeyEx"):
        assert callable(getattr(cv2, fn)), fn
    # the wheel's notches are the signed high word of the event's flags (no helper needed)
    assert PJ.wheel_delta(120 << 16) == 120 and PJ.wheel_delta((240 << 16) | 16) == 240
    assert PJ.wheel_delta(-120 << 16) == -120 and PJ.wheel_delta((-360 << 16) | 8) == -360
    assert PJ.wheel_delta(0xFF880000) == -120 and PJ.wheel_delta(16) == 0


class FakeCv2:
    def __init__(self, keys=()):
        self.keys = list(keys)
        self.callback = None
        self.waits = []

    def setMouseCallback(self, name, fn):
        self.callback = (name, fn)

    def waitKeyEx(self, ms):
        self.waits.append(ms)
        return self.keys.pop(0) if self.keys else -1



def _window(rotate90=False, keys=()):
    w = HDMIWindow.__new__(HDMIWindow)                      # no window is created
    w._cv2, w.size, w.rotate90 = FakeCv2(keys), (912, 1140), rotate90
    w._events, w._pattern_shape, w.mouse_enabled = [], DMD_SHAPE, False
    return w


def test_hdmi_window_translates_mouse_and_keys_into_painter_events():
    w = _window(keys=[-1, ord("e"), 0x2E0000])
    w.enable_mouse()
    assert w.mouse_enabled and w._cv2.callback[0] == HDMIWindow.WINDOW
    cb = w._cv2.callback[1]
    cb(1, 100, 200, 0, None)                                # left down
    for x in (110, 120, 130):
        cb(0, x, 210, 1, None)                              # moves: one is kept
    cb(4, 130, 210, 16, None)                               # left up, shift held
    cb(2, 5, 6, 0, None)                                    # right down
    cb(7, 50, 60, 0, None)                                  # double click
    cb(10, 50, 60, 240 << 16, None)                         # wheel forward, 2 notches
    cb(10, 50, 60, (-120 << 16) | 16, None)                 # wheel back, shift held
    cb(10, 50, 60, 16, None)                                # no movement: no event
    cb(99, 1, 1, 0, None)                                   # an event with no meaning here
    ev = w.poll(15)
    assert w._cv2.waits == [15]
    assert [e["type"] for e in ev] == ["down", "move", "up", "down", "dblclick", "wheel", "wheel"]
    assert ev[0] == {"type": "down", "at": [100.0, 200.0], "shift": False, "ctrl": False,
                     "button": "left"}
    assert ev[1]["at"] == [130.0, 210.0] and ev[1]["left"] is True
    assert ev[2]["shift"] is True and ev[3]["button"] == "right"
    assert ev[5]["delta"] == 2 and ev[6]["delta"] == -1 and ev[6]["shift"] is True
    assert w.poll(15) == [{"type": "key", "key": "e"}]
    assert w.poll(15) == [{"type": "key", "key": "delete"}]
    assert w.poll(15) == []
    # the events drive the model as they are
    m = PainterModel()
    for e in ev:
        m.handle(e)
    assert m.state.cursor == (50.0, 60.0)
    # a turned display: positions come back in PATTERN px
    r = _window(rotate90=True)
    r.enable_mouse()
    r._cv2.callback[1](1, 560, 764, 0, None)
    (e,) = r.poll(1)
    assert abs(e["at"][0] - 300) <= 1 and abs(e["at"][1] - 700) <= 1


# ------------------------------------------------------------------ keys and the wheel during a drag

@pytest.mark.parametrize("key", ["delete", "backspace"])
@pytest.mark.parametrize("then", ["up", "move"])
def test_delete_while_the_button_is_down_on_a_zone_ends_the_drag(key, then):
    """Delete under a held button removed the zone but left the drag pointing at it: the
    release (or the next move) then crashed and the unsaved session was lost."""
    m = PainterModel()
    drag(m, (100, 100), (300, 300))                         # Z1
    drag(m, (400, 400), (600, 600))                         # Z2: the work a crash would lose
    m.handle({"type": "down", "at": [200, 200]})            # press on Z1 ...
    m.handle({"type": "move", "at": [210, 215], "left": True})
    m.on_key(key)                                           # ... and delete it before releasing
    assert [z.id for z in m.state.zones] == ["Z2"] and m._drag is None
    m.handle({"type": then, "at": [230, 240], "left": True} if then == "move"
             else {"type": "up", "at": [230, 240]})
    assert [z.id for z in m.state.zones] == ["Z2"] and m.state.preview is None
    m.on_key("z")                                           # the delete is one undo step ...
    assert [z.id for z in m.state.zones] == ["Z1", "Z2"]
    m.on_key("z")                                           # ... the move before it another
    assert bbox(m.state.zone("Z1")) == (100, 100, 300, 300)
    m.on_key("space")
    assert m.state.done == "save"


def test_delete_during_a_polygon_corner_drag_and_through_the_shell():
    m = PainterModel()
    m.on_key("p")
    for pt in ((500, 500), (800, 520), (700, 800)):
        click(m, pt)
    m.on_key("enter")
    m.handle({"type": "down", "at": [500, 500]})            # its first corner
    m.on_key("delete")
    m.handle({"type": "move", "at": [520, 520], "left": True})
    m.handle({"type": "up", "at": [520, 520]})
    assert m.state.zones == []
    # the same through the window shell: the session goes on and saves
    w = MockWindow()
    w.feed([{"type": "drag", "from": [100, 100], "to": [300, 300]},
            {"type": "drag", "from": [400, 400], "to": [600, 600]},
            {"type": "down", "at": [200, 200]}, {"type": "key", "key": "delete"},
            {"type": "up", "at": [200, 200]}, {"type": "key", "key": "space"}])
    m = PainterModel()
    assert S.run_painter(m, w, mock=True) == "save" and [z.id for z in m.zones()] == ["Z2"]


def test_a_stale_drag_never_crashes_a_mouse_move():
    """Whatever removed the zone under a drag (here: the list itself), the move ends the
    drag instead of looking the zone up."""
    m = PainterModel()
    drag(m, (100, 100), (300, 300))
    m.handle({"type": "down", "at": [200, 200]})
    m.state.zones = []
    m.handle({"type": "move", "at": [220, 220], "left": True})
    assert m._drag is None and m.state.zones == []


@pytest.mark.parametrize("key", ["n", "]", "}", "left", "tab"])
def test_a_key_during_a_drag_ends_it_and_is_not_undone_by_the_next_move(key):
    m = PainterModel()
    drag(m, (100, 100), (300, 300))
    drag(m, (500, 500), (700, 700))
    m.handle({"type": "down", "at": [200, 200]})
    m.handle({"type": "move", "at": [220, 230], "left": True})          # Z1 moved by (20, 30)
    depth = len(m._undo)
    m.on_key(key)
    after = m.state.zone("Z1").to_dict()
    for at in ((260, 260), (300, 310)):                     # the button is still held
        m.handle({"type": "move", "at": list(at), "left": True})
    m.handle({"type": "up", "at": [300, 310]})
    z = m.state.zone("Z1")
    assert z.to_dict() == after                             # the key's edit stands
    assert m._drag is None
    if key == "n":
        assert z.name == "part" and bbox(z) == (120, 130, 320, 330)
    elif key in ("]", "}"):
        assert z.rotation_deg in (1.0, 15.0) and z.center == (220.0, 230.0)
    elif key == "left":
        assert bbox(z) == (119, 130, 319, 330)
    else:
        assert m.state.selected == "Z2" and bbox(z) == (120, 130, 320, 330)
    # the drag is one undo step, the key's edit one more (Tab edits nothing)
    assert len(m._undo) == depth + (0 if key == "tab" else 1)


def test_the_wheel_during_a_drag_turns_the_zone_where_it_is():
    m = PainterModel()
    drag(m, (100, 100), (300, 300))
    m.handle({"type": "down", "at": [200, 200]})
    m.handle({"type": "move", "at": [220, 220], "left": True})
    m.handle({"type": "wheel", "at": [220, 220], "delta": 2})
    m.handle({"type": "move", "at": [240, 240], "left": True})
    m.handle({"type": "up", "at": [240, 240]})
    z = m.state.zones[0]
    assert z.rotation_deg == 10.0 and z.center == (220.0, 220.0)
    assert len(m._undo) == 3                                # created, moved, turned


def test_switching_to_the_polygon_tool_drops_a_shape_being_dragged_out():
    m = PainterModel()
    m.handle({"type": "down", "at": [100, 100]})
    m.handle({"type": "move", "at": [300, 300], "left": True})
    assert m.state.preview is not None
    m.on_key("p")                                           # a polygon is clicked, not dragged
    assert m.state.preview is None and m._drag is None and m.state.tool == "polygon"
    m.handle({"type": "up", "at": [320, 320]})
    assert m.state.zones == [] and m.state.poly == []
    # R / E during the drag change what is being dragged out
    m = PainterModel()
    m.handle({"type": "down", "at": [100, 100]})
    m.handle({"type": "move", "at": [300, 260], "left": True})
    m.on_key("e")
    m.handle({"type": "up", "at": [300, 260]})
    assert [z.shape for z in m.state.zones] == ["ellipse"]


# ------------------------------------------------------------------ small zones, quick presses

@pytest.mark.parametrize("tool", ["r", "e"])
@pytest.mark.parametrize("side", [8, 12, 16, 18, 30])
def test_a_small_selected_zone_moves_when_dragged_from_its_middle(tool, side):
    """With a fixed 9 px pick radius every point of a selected zone under ~18 px was a
    handle: dragging it from its middle resized it."""
    m = PainterModel()
    m.on_key(tool)
    drag(m, (400, 400), (400 + side, 400 + side))
    z0 = m.state.zones[0]
    assert m.state.selected == z0.id and m.hit_test(*z0.center) == (z0.id, ("inside",))
    drag(m, z0.center, (z0.center[0] + 50, z0.center[1] + 30))
    z = m.state.zones[0]
    assert tuple(z.size) == tuple(z0.size) == (side, side)
    assert z.center == (z0.center[0] + 50, z0.center[1] + 30)
    # its corner still resizes it
    c = z.corners()[2]
    drag(m, c, (c[0] + 20, c[1] + 10))
    assert tuple(m.state.zones[0].size) == (side + 20, side + 10)


def test_a_small_polygon_keeps_a_middle_and_large_shapes_keep_their_pick_radius():
    m = PainterModel()
    m.on_key("p")
    for pt in ((400, 400), (414, 400), (414, 414), (400, 414)):
        click(m, pt)
    m.on_key("enter")
    assert m.hit_test(407, 407) == ("Z1", ("inside",))
    assert m.hit_test(400.5, 400.5) == ("Z1", ("vertex", 0))
    big = PainterModel()
    drag(big, (100, 100), (400, 300))
    assert big.hit_test(100 + P.HIT_PX, 100 + P.HIT_PX) == ("Z1", ("handle", -1, -1))
    assert big.hit_test(250, 300 - P.EDGE_PX) == ("Z1", ("handle", 0, 1))


def test_a_quick_second_press_is_a_press_not_a_lost_double_click():
    """Windows reports the second press of a quick pair as a double click: click a zone to
    select it, press again at once and drag - the drag must start."""
    m = PainterModel()
    drag(m, (100, 100), (300, 300))
    click(m, (600, 600))                                    # deselect
    for ev in ({"type": "down", "at": [200, 200]}, {"type": "up", "at": [200, 200]},
               {"type": "dblclick", "at": [200, 200]},      # the 2nd press, as cv2 reports it
               {"type": "move", "at": [260, 250], "left": True},
               {"type": "move", "at": [320, 310], "left": True},
               {"type": "up", "at": [320, 310]}):
        m.handle(ev)
    assert bbox(m.state.zones[0]) == (220, 210, 420, 410) and m._drag is None
    # on empty space it drags out a new shape; with a polygon open it still closes it
    for ev in ({"type": "dblclick", "at": [500, 700]}, {"type": "move", "at": [560, 760]},
               {"type": "up", "at": [560, 760]}):
        m.handle(ev)
    assert len(m.state.zones) == 2 and bbox(m.state.zones[1]) == (500, 700, 560, 760)
    m.on_key("p")
    for pt in ((600, 100), (800, 100), (800, 250)):
        click(m, pt)
    m.handle({"type": "dblclick", "at": [600, 250]})
    m.handle({"type": "up", "at": [600, 250]})
    assert len(m.state.zones) == 3 and m.state.zones[2].shape == "polygon"
    assert len(m.state.zones[2].points) == 4 and m._drag is None


# ------------------------------------------------------------------ labels, the legend strip

def test_custom_labels_cycle_back_to_no_name():
    m = PainterModel(labels=("seal", "lid", "seal"))        # a repeated label counts once
    assert m.labels == ("", "seal", "lid")
    drag(m, (100, 100), (300, 300))
    seen = []
    for _ in range(4):
        m.on_key("n")
        seen.append(m.state.zones[0].name)
    assert seen == ["seal", "lid", "", "seal"]
    m.on_key("N")
    assert m.state.zones[0].name == ""
    assert PainterModel().labels == P.LABELS and P.LABELS[0] == ""


def test_a_zone_in_the_legend_rows_stays_visible():
    """The legend strip used to be painted over the drawing: a zone drawn in its rows, and
    the crosshair there, vanished until H."""
    m = PainterModel()
    lh = P.legend_height(m.state)
    drag(m, (300, H - lh + 6), (600, H - 3))
    z = m.state.zones[0]
    m.state.selected = None
    m.state.cursor = None
    img = render(m.state)
    x0, y0, x1, y1 = (int(round(v)) for v in z.bbox())
    for row in (y0, y1 - 1):                                # its top and bottom edges
        assert (img[row - 1:row + 2, x0 + 20:x1 - 20].max(0) >= P.OUTLINE - 5).mean() > 0.95
    for col in (x0, x1 - 1):                                # its sides
        assert (img[y0 + 8:y1 - 8, col - 1:col + 2].max(1) >= P.OUTLINE - 5).mean() > 0.95
    m.state.cursor = (450.0, float(H - lh // 2))            # the crosshair inside the strip
    img = render(m.state)
    cy = H - lh // 2
    assert img[cy, 100] >= P.HAIR_LEVEL and img[H - 2, 450] >= P.HAIR_LEVEL
    assert img[cy, 450 + 10] == P.BRIGHT and img[cy - 10, 450] == P.BRIGHT
    # the legend text is still there, and the strip has no grid underlay
    m.on_key("g")
    strip = render(m.state)[H - lh:, :]
    assert (strip[0] == P.GRID_LEVEL).all() and (strip[1:, :280] >= P.LEGEND_LEVEL - 60).any()
    assert not (strip[3:, 96] == P.GRID_LEVEL).all()


def test_the_worst_case_session_is_still_lines():
    """64 zones (the most a set holds) that each span the field, the grid on: the measured
    upper bound of the lit share the module doc gives - outlines, never a filled field."""
    zones = [Zone.rect(f"Z{k + 1}", 4 + 7 * k, 4 + 8 * k, W - 4 - 7 * k, H - 60 - 8 * k)
             for k in range(P.MAX_ZONES)]
    m = PainterModel(zones)
    m.on_key("g")
    m.on_key("tab")
    img = render(m.state)
    lit = float((img >= 128).mean())
    assert 0.05 < lit < 0.45, lit
    # no filled area: every lit run along a row through the middle is a line's width
    row = img[H // 2 - 3] >= 128
    runs = np.diff(np.flatnonzero(np.diff(np.r_[0, row.astype(np.int8), 0])))[::2]
    assert runs.max() <= 6


# ------------------------------------------------------------------ keys while the picture is redrawn

class PumpCv2:
    """OpenCV's key queue: a key press is handed to the NEXT waitKey / waitKeyEx call,
    whichever it is - also the one inside show()."""
    COLOR_GRAY2BGR, INTER_NEAREST = 8, 0

    def __init__(self):
        self.pending = []                                   # key codes not yet handed out
        self.callback = None
        self.plain_waits = self.ex_waits = self.shown = 0

    def setMouseCallback(self, name, fn):
        self.callback = fn

    def cvtColor(self, img, code):
        return img

    def resize(self, img, size, interpolation=0):
        return img

    def getWindowImageRect(self, name):
        return (1920, 0, 912, 1140)

    def imshow(self, name, img):
        self.shown += 1

    def waitKey(self, ms):
        self.plain_waits += 1
        return (self.pending.pop(0) & 0xFF) if self.pending else -1

    def waitKeyEx(self, ms):
        self.ex_waits += 1
        return self.pending.pop(0) if self.pending else -1


def _pump_window():
    w = HDMIWindow.__new__(HDMIWindow)                      # no window is created
    w._cv2, w.size, w.rotate90 = PumpCv2(), (912, 1140), False
    w._events, w._pattern_shape, w.mouse_enabled = [], DMD_SHAPE, False
    w._rect_settled = True
    w.rect = w.rect_ok = w.rect_warning = None
    return w


def test_a_key_pressed_while_the_picture_is_redrawn_is_not_lost():
    """show() pumps the window, and OpenCV hands a key to whichever pump runs next: the
    painter's show() used to throw that key away (keys were ignored while the mouse moved)."""
    w = _pump_window()
    w.enable_mouse()
    img = np.zeros(DMD_SHAPE, np.uint8)
    cb = w._cv2.callback
    cb(0, 100, 100, 0, None)                                # the mouse moves ...
    w._cv2.pending.append(ord("e"))                         # ... a key is pressed ...
    w.show(img)                                             # ... and the redraw's pump gets it
    cb(0, 120, 110, 0, None)
    w._cv2.pending.append(0x2E0000)                         # Delete, received by poll's pump
    ev = w.poll(15)
    assert [e["type"] for e in ev] == ["move", "key", "move", "key"]
    assert ev[1]["key"] == "e" and ev[3]["key"] == "delete" and ev[2]["at"] == [120.0, 110.0]
    assert w._cv2.plain_waits == 0                          # no pump that drops keys
    assert w.poll(15) == []
    # twice in one redraw (push_twice): both keys arrive, in order
    w._cv2.pending += [ord("g"), 27]
    w.show(img)
    w.show(img)
    assert [e["key"] for e in w.poll(15)] == ["g", "escape"]


def test_the_painter_loop_gets_every_key_while_the_mouse_moves():
    """The real show() / poll() under the painter's loop, the mouse moving all the time (so
    every pass redraws, twice with push_twice): each of 200 key presses reaches the model."""
    for twice in (False, True):
        w = _pump_window()
        w.enable_mouse()
        fake = w._cv2
        keys = [ord("g")] * 200 + [27, 27]                  # G is harmless and countable
        state = {"n": 0}
        ex = fake.waitKeyEx

        def pump(ms, _ex=ex, _fake=fake, _state=state, _keys=keys):
            # between two pumps: the mouse moved, and now and then a key was pressed
            _state["n"] += 1
            _fake.callback(0, 100 + _state["n"] % 600, 300, 0, None)
            if _keys and _state["n"] % 3 == 0:
                _fake.pending.append(_keys.pop(0))
            return _ex(ms)
        fake.waitKeyEx = pump
        m = PainterModel()
        got = {"g": 0}
        on_key = m.on_key

        def count(k, _on_key=on_key, _got=got):
            if k == "g":
                _got["g"] += 1
            return _on_key(k)
        m.on_key = count
        assert S.run_painter(m, w, mock=True, twice=twice) == "cancel"
        assert got["g"] == 200, (twice, got)
        assert fake.plain_waits == 0 and fake.shown > 200


def test_a_window_without_the_mouse_shows_as_before():
    w = _pump_window()
    w._cv2.pending.append(ord("e"))
    w.show(np.zeros(DMD_SHAPE, np.uint8))
    assert w._cv2.plain_waits == 1 and w._cv2.ex_waits == 0 and w._events == []

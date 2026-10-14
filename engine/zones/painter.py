"""
Zone painter: the interaction logic and the picture, with NO display.

The projector is a second monitor, so the mouse cursor is projected onto the real part: the
operator drags shapes out directly on it. Everything that decides what a click, a drag or a
key does lives here, in PainterModel (a state machine over PainterState); render(state)
turns a state into the DMD image (thin bright outlines, handles, zone ids, a crosshair at the
cursor and a small legend strip - on black, never a lit field). The window
(engine/zones/session.py) is a thin shell: it feeds events in and shows the image.

    model = PainterModel()
    model.on_mouse_down(100, 200); model.on_mouse_move(400, 500); model.on_mouse_up(400, 500)
    model.on_key("space")                    # -> model.state.done == "save"
    img = render(model.state)                # (1140, 912) uint8, physical intent (bright = lit)

Mouse (projector px, col right / row down):
    drag on empty space      a new shape with the current tool (rectangle; E ellipse)
    P, then clicks           a polygon: one click per corner; double-click, Enter or a click
                             on the first corner closes it
    click a shape            select it (the smallest shape under the cursor)
    drag inside it           move
    drag a handle            resize (a rectangle: its edges too; a polygon: drag a corner)
    wheel                    rotate the selected shape (5 deg a notch; Shift: 1 deg)
    Shift + drag             a new shape even when the drag starts inside one
    right click              drop the polygon being drawn / deselect
Keys: see KEYS (the table the legend, the CLI help and the README are written from).

A scripted session (tests, docs, `scan --mock zone-paint --events FILE`) is a list of events
for PainterModel.handle(): {"type": "drag", "from": [c, r], "to": [c, r]}, {"type": "click",
"at": [c, r]}, {"type": "dblclick", "at": [c, r]}, {"type": "key", "key": "e"}, {"type":
"wheel", "at": [c, r], "delta": 1}, and the raw ones the window sends: down / move / up
(a move may say "left": false - the button is not held: a drag whose release happened outside
the window ends there). A "dblclick" is the SECOND PRESS of a quick pair, as a window reports
it (its release follows as an "up"): it closes the polygon being drawn, and is a press like
any other when there is none (click a shape, then at once drag it).

A key or the wheel that edits the selected zone (N, [ ] { }, the arrows, Delete, Tab) while
it is being dragged ENDS the drag where it is: the edit is never undone by the next mouse
move, and a zone removed under the cursor is never looked for again.

The picture is lines: about 2 % of the mirrors are lit in an ordinary session. It grows
with the drawing (64 zones that each span the whole field, with the grid: about 40 % of the
mirrors at half level or more, measured in tests/test_zones_painter.py) - still outlines,
never a filled field.
"""

from __future__ import annotations

import functools
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .model import (DMD_SHAPE, MAX_POINTS, MAX_ZONES, MIN_SIZE_PX, Zone, ZoneError, next_id,
                    zone_problems)

TOOLS = ("rect", "ellipse", "polygon")
HANDLE_PX = 5                   # half side of a handle square, DMD px
HIT_PX = 9                      # pick radius around a handle / a corner
EDGE_PX = 5                     # pick distance from a rectangle's edge
CLOSE_PX = 10                   # a click this near the first corner closes the polygon
WHEEL_DEG = 5.0                 # per wheel notch (Shift: FINE_DEG)
FINE_DEG = 1.0                  # [ and ]
COARSE_DEG = 15.0               # { and }
NUDGE_PX = 1.0                  # arrow keys
UNDO_MAX = 200
LABELS = ("", "part", "datum", "end face", "tube", "box", "edge", "fixture")
GRID_SPACING = 96               # the lettered grid of `scan grid` (same cells, same names)

OUTLINE = 200                   # an unselected outline, grey level
BRIGHT = 255                    # the selected outline, handles, labels
OUTLINE_PX, SELECTED_PX = 2, 3
GRID_LEVEL, GRID_TEXT = 70, 110
HAIR_LEVEL = 60                 # the cursor's full-length hairlines (dim)
LEGEND_LEVEL = 170
LEGEND_LINE_PX = 15
LEGEND_PAD = 5

# key -> what it does (one table: the legend strip, `scan zone-paint --help`, the README)
KEYS = (
    ("drag", "new shape (rectangle; E ellipse)"),
    ("P + clicks", "polygon: click the corners; double-click or Enter closes"),
    ("click", "select a shape; drag inside moves, handles resize (a rectangle: its edges too)"),
    ("wheel, [ ]", "rotate the selected shape (wheel 5 deg, [ ] 1 deg, { } 15 deg)"),
    ("arrows", "nudge the selected shape 1 px"),
    ("R / E / P", "tool: rectangle / ellipse / polygon"),
    ("N", "name: cycle the selected zone's label (Shift+N backwards)"),
    ("Tab", "select the next zone"),
    ("Delete, Backspace", "remove the selected zone (a polygon being drawn: its last corner)"),
    ("Z, Ctrl+Z", "undo"),
    ("G", "lettered grid underlay on / off"),
    ("H", "legend strip on / off"),
    ("Space", "confirm and save"),
    ("Esc", "cancel without saving (first drops a polygon being drawn)"),
)
LEGEND = ("drag: new   click: select   drag inside: move   handles: resize   wheel / [ ]: rotate"
          "   arrows: nudge",
          "R rect  E ellipse  P polygon (Enter closes)  N name  Tab next  Del remove  Z undo"
          "  G grid  H legend  SPACE save  ESC cancel")


@dataclass
class PainterState:
    zones: list = field(default_factory=list)       # [Zone] in id order
    tool: str = "rect"
    selected: Optional[str] = None                  # zone id
    cursor: Optional[tuple] = None                  # (col, row) of the mouse
    preview: Optional[Zone] = None                  # the shape being dragged out
    poly: list = field(default_factory=list)        # the polygon being clicked: [(col, row)]
    show_grid: bool = False
    show_legend: bool = True
    message: str = ""
    done: Optional[str] = None                      # None | "save" | "cancel"
    name: str = ""                                  # the set's name (legend)
    shape: tuple = DMD_SHAPE
    grid_spacing: int = GRID_SPACING

    def zone(self, zid) -> Optional[Zone]:
        return next((z for z in self.zones if z.id == zid), None)

    def to_dict(self) -> dict:
        return {"tool": self.tool, "selected": self.selected, "done": self.done,
                "message": self.message, "cursor": list(self.cursor) if self.cursor else None,
                "drawing": [list(p) for p in self.poly], "show_grid": self.show_grid,
                "show_legend": self.show_legend, "zones": [z.to_dict() for z in self.zones]}


def handles(z: Zone) -> list:
    """[(part, col, row)] of a zone's handles: a rectangle / an ellipse has the 4 corners and
    4 edge midpoints of its (rotated) box, a polygon its corners."""
    if z.shape == "polygon":
        return [(("vertex", i), x, y) for i, (x, y) in enumerate(z.points)]
    w, h = z.size
    out = []
    for su, sv in ((-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)):
        x, y = z.world(su * w / 2.0, sv * h / 2.0)
        out.append((("handle", su, sv), x, y))
    return out


class PainterModel:
    """The painter as a state machine: on_mouse_down / on_mouse_move / on_mouse_up /
    on_double_click / on_wheel / on_key -> PainterState. `dirty` says the picture changed."""

    def __init__(self, zones=None, *, name: str = "", labels=LABELS, shape: tuple = DMD_SHAPE,
                 grid_spacing: int = GRID_SPACING):
        zs = [z.copy() for z in (zones or [])]
        for z in zs:
            probs = zone_problems(z, shape)
            if probs:
                raise ZoneError(f"the painter can't start from an invalid zone: {probs[0]}", probs)
        self.state = PainterState(zones=zs, name=name, shape=tuple(shape),
                                  grid_spacing=int(grid_spacing))
        labs = tuple(dict.fromkeys(labels or ())) or LABELS
        self.labels = labs if "" in labs else ("",) + labs    # N always cycles back to no name
        self.dirty = True
        self.edits = 0                              # committed changes (0: nothing to save)
        self._undo: list = []
        self._drag: Optional[dict] = None

    # ------------------------------------------------------------ helpers
    def _clamp(self, x, y) -> tuple:
        H, W = self.state.shape
        return (min(max(float(x), 0.0), float(W)), min(max(float(y), 0.0), float(H)))

    def _say(self, text: str) -> None:
        self.state.message = text
        self.dirty = True

    def _push(self) -> None:
        self._undo.append(([z.copy() for z in self.state.zones], self.state.selected))
        del self._undo[:-UNDO_MAX]
        self.edits += 1

    def _replace(self, z: Zone) -> None:
        zs = self.state.zones
        zs[next(i for i, o in enumerate(zs) if o.id == z.id)] = z
        self.dirty = True

    def _end_drag(self) -> None:
        """The drag in progress ends where it is (what it moved so far stays, one undo step);
        a shape being dragged out is dropped. The mouse moves that follow, button still held,
        only move the cursor."""
        self._drag, self.state.preview = None, None

    def _fits(self, z: Zone) -> bool:
        H, W = self.state.shape
        x0, y0, x1, y1 = z.bbox()
        return x0 >= -1e-6 and y0 >= -1e-6 and x1 <= W + 1e-6 and y1 <= H + 1e-6

    def _into_field(self, z: Zone) -> Optional[Zone]:
        """z shifted back inside the DMD (None: it is bigger than the DMD)."""
        H, W = self.state.shape
        x0, y0, x1, y1 = z.bbox()
        if x1 - x0 > W + 1e-6 or y1 - y0 > H + 1e-6:
            return None
        dx = -x0 if x0 < 0 else (W - x1 if x1 > W else 0.0)
        dy = -y0 if y0 < 0 else (H - y1 if y1 > H else 0.0)
        return z.moved(dx, dy) if (dx or dy) else z

    def _make(self, tool: str, p0, p1, zid: str) -> Zone:
        (x0, y0), (x1, y1) = p0, p1
        return (Zone.ellipse if tool == "ellipse" else Zone.rect)(zid, x0, y0, x1, y1)

    def hit_test(self, x: float, y: float) -> Optional[tuple]:
        """What is under (col, row): (zone id, part) with part ("handle", su, sv) | ("vertex",
        i) | ("inside",), or None. The selected zone's handles and edges win; then the
        SMALLEST zone that contains the point (a zone drawn inside another stays reachable)."""
        st = self.state
        sel = st.zone(st.selected)
        if sel is not None:
            # a small shape: the pick radius shrinks with it (a quarter of its smaller side),
            # so its middle is always "inside" and a drag there moves it
            x0, y0, x1, y1 = sel.bbox()
            small = min(sel.size) if sel.shape != "polygon" else min(x1 - x0, y1 - y0)
            hit, edge = min(HIT_PX, small / 4.0), min(EDGE_PX, small / 4.0)
            best = None
            for part, hx, hy in handles(sel):
                d = max(abs(x - hx), abs(y - hy))
                if d <= hit and (best is None or d < best[0]):
                    best = (d, part)
            if best is not None:
                return (sel.id, best[1])
            if sel.shape == "rect":
                u, v = sel.local(x, y)
                a, b = sel.size[0] / 2.0, sel.size[1] / 2.0
                if abs(abs(u) - a) <= edge and abs(v) <= b:
                    return (sel.id, ("handle", 1 if u > 0 else -1, 0))
                if abs(abs(v) - b) <= edge and abs(u) <= a:
                    return (sel.id, ("handle", 0, 1 if v > 0 else -1))
        inside = [z for z in st.zones if z.contains(x, y)]
        if not inside:
            return None
        z = min(inside, key=lambda q: (q.area(), q.id != st.selected))
        return (z.id, ("inside",))

    # ------------------------------------------------------------ mouse
    def on_mouse_down(self, x, y, button: str = "left", shift: bool = False) -> PainterState:
        st = self.state
        if st.done:
            return st
        x, y = self._clamp(x, y)
        st.cursor, self.dirty = (x, y), True
        if self._drag is not None:              # a drag whose release never arrived: it ended
            self._drag, st.preview = None, None
        if button != "left":
            if st.poly:
                st.poly = []
                self._say("polygon dropped")
            elif button == "right":
                st.selected = None
            return st
        if st.poly:
            self._poly_click(x, y)
            return st
        hit = None if shift else self.hit_test(x, y)
        if hit is None:
            st.selected = None
            if st.tool == "polygon":
                self._poly_click(x, y)
            else:
                self._drag = {"kind": "create", "p0": (x, y)}
            return st
        zid, part = hit
        st.selected = zid
        z0 = st.zone(zid).copy()
        self._drag = {"kind": {"inside": "move", "handle": "resize", "vertex": "vertex"}[part[0]],
                      "id": zid, "p0": (x, y), "z0": z0, "part": part, "pushed": False}
        return st

    def on_mouse_move(self, x, y) -> PainterState:
        st = self.state
        if st.done:
            return st
        x, y = self._clamp(x, y)
        st.cursor, self.dirty = (x, y), True
        d = self._drag
        if d is None:
            return st
        if d["kind"] == "create":
            st.preview = self._make(st.tool, d["p0"], (x, y), "")
            return st
        z0 = d["z0"]
        if st.zone(z0.id) is None:              # the zone under the drag is gone: no drag
            self._end_drag()
            return st
        if d["kind"] == "move":
            z = self._into_field(z0.moved(x - d["p0"][0], y - d["p0"][1]))
        elif d["kind"] == "resize":
            z = self._resized(z0, d["part"][1], d["part"][2], x, y)
        else:
            z = z0.copy()
            z.points[d["part"][1]] = (x, y)
        if z is None or not self._fits(z) or zone_problems(z, st.shape):
            return st                           # this position is not a valid shape: keep the last
        if z.to_dict() == st.zone(z.id).to_dict():
            return st                           # nothing moved (a click that only selects)
        if not d["pushed"]:
            self._push()                        # one undo step per drag, only when it changed
            d["pushed"] = True
        self._replace(z)
        return st

    def on_mouse_up(self, x, y, button: str = "left") -> PainterState:
        st = self.state
        if st.done or button != "left":
            return st
        self.on_mouse_move(x, y)
        d, self._drag = self._drag, None
        if d is None or d["kind"] != "create":
            return st
        st.preview = None
        x, y = st.cursor
        (x0, y0) = d["p0"]
        if min(abs(x - x0), abs(y - y0)) < MIN_SIZE_PX:
            if max(abs(x - x0), abs(y - y0)) > 2:
                self._say(f"too small: a zone is at least {MIN_SIZE_PX:g} px each way")
            return st                           # a click on empty space: deselected, no shape
        if len(st.zones) >= MAX_ZONES:
            self._say(f"at most {MAX_ZONES} zones")
            return st
        z = self._make(st.tool, (x0, y0), (x, y), next_id(st.zones))
        self._push()
        st.zones.append(z)
        st.selected = z.id
        self._say(f"{z.id} added ({z.shape})")
        return st

    def on_double_click(self, x, y, shift: bool = False) -> PainterState:
        """The second press of a quick pair (a window reports it as a double click instead
        of a press; its release follows): it closes the polygon being drawn. With no polygon
        open it is a press like any other - a click that selects, then at once a drag."""
        st = self.state
        if st.done:
            return st
        if not st.poly:
            return self.on_mouse_down(x, y, shift=shift)
        x, y = self._clamp(x, y)
        st.cursor, self.dirty = (x, y), True
        lx, ly = st.poly[-1]
        if math.hypot(x - lx, y - ly) > 3.0:
            st.poly.append((x, y))
        self._poly_close()
        return st

    def on_wheel(self, x, y, delta, shift: bool = False) -> PainterState:
        """delta: wheel notches (+ = away from the operator = clockwise in the DMD image)."""
        if self.state.done or not delta:
            return self.state
        notches = (1 if delta > 0 else -1) * max(1, abs(int(delta)))
        step = (FINE_DEG if shift else WHEEL_DEG) * notches
        self._rotate(step)
        return self.state

    # ------------------------------------------------------------ polygon
    def _poly_click(self, x, y) -> None:
        st = self.state
        if len(st.poly) >= 3 and math.hypot(x - st.poly[0][0], y - st.poly[0][1]) <= CLOSE_PX:
            self._poly_close()
            return
        if st.poly and math.hypot(x - st.poly[-1][0], y - st.poly[-1][1]) < 1.0:
            return                              # the same spot twice: not a new corner
        if len(st.poly) >= MAX_POINTS:
            self._say(f"at most {MAX_POINTS} corners: close the polygon (Enter)")
            return
        st.poly.append((x, y))
        self._say(f"polygon: {len(st.poly)} corner(s) - double-click or Enter closes")

    def _poly_close(self) -> None:
        st = self.state
        pts, st.poly = st.poly, []
        self.dirty = True
        if len(st.zones) >= MAX_ZONES:
            self._say(f"at most {MAX_ZONES} zones")
            return
        z = Zone.polygon(next_id(st.zones), pts)
        probs = zone_problems(z, st.shape)
        if probs:
            self._say("polygon not added: " + probs[0].split(": ", 1)[-1])
            return
        self._push()
        st.zones.append(z)
        st.selected = z.id
        self._say(f"{z.id} added (polygon, {len(pts)} corners)")

    # ------------------------------------------------------------ edits
    def _resized(self, z0: Zone, su: int, sv: int, x: float, y: float) -> Zone:
        """z0 with the handle (su, sv) dragged to (x, y): the opposite edge / corner stays."""
        u, v = z0.local(x, y)
        w, h = z0.size
        cu = cv = 0.0
        if su:
            anchor = -su * z0.size[0] / 2.0
            w = max(MIN_SIZE_PX, su * (u - anchor))
            cu = anchor + su * w / 2.0
        if sv:
            anchor = -sv * z0.size[1] / 2.0
            h = max(MIN_SIZE_PX, sv * (v - anchor))
            cv = anchor + sv * h / 2.0
        z = z0.copy()
        z.center, z.size = z0.world(cu, cv), (w, h)
        return z

    def _selected(self) -> Optional[Zone]:
        z = self.state.zone(self.state.selected)
        if z is None:
            self._say("select a zone first (click it, or Tab)")
        return z

    def _rotate(self, ddeg: float) -> None:
        z = self._selected()
        if z is None:
            return
        self._end_drag()
        new = self._into_field(z.rotated(ddeg))
        if new is None or zone_problems(new, self.state.shape):
            self._say(f"{z.id} can't turn further here: it would leave the DMD")
            return
        self._push()
        self._replace(new)
        rot = f"{new.rotation_deg:g} deg" if new.shape != "polygon" else f"{ddeg:+g} deg"
        self._say(f"{z.id} rotated: {rot}")

    def _nudge(self, dx: float, dy: float) -> None:
        z = self._selected()
        if z is None:
            return
        self._end_drag()
        new = self._into_field(z.moved(dx, dy))
        if new is None or new.bbox() == z.bbox():
            return
        self._push()
        self._replace(new)

    def undo(self) -> None:
        st = self.state
        if st.poly:
            st.poly.pop()
            self._say(f"polygon: {len(st.poly)} corner(s)")
            return
        if not self._undo:
            self._say("nothing to undo")
            return
        st.zones, st.selected = self._undo.pop()
        self.edits = max(0, self.edits - 1)
        self._drag, st.preview = None, None
        self._say("undone")

    def cancel(self, why: str = "cancelled") -> PainterState:
        self.state.done, self.state.preview, self.state.poly = "cancel", None, []
        self._drag = None
        self._say(why)
        return self.state

    # ------------------------------------------------------------ keys
    def on_key(self, key: str) -> PainterState:
        """key: one character ("e", "N", "[", " ") or a name: space, enter, escape, delete,
        backspace, tab, left / right / up / down, ctrl+z."""
        st = self.state
        if st.done or not isinstance(key, str) or not key:
            return st
        k = key if len(key) == 1 else key.strip().lower()
        if k == " ":
            k = "space"
        low = k.lower()
        self.dirty = True
        if low in ("r", "e", "p"):
            tool = {"r": "rect", "e": "ellipse", "p": "polygon"}[low]
            if st.poly and tool != "polygon":
                st.poly = []
            if tool == "polygon" and self._drag is not None and self._drag["kind"] == "create":
                self._end_drag()                # a polygon is clicked, not dragged out
            st.tool = tool
            self._say(f"tool: {tool}" + (" - click the corners, double-click or Enter closes"
                                         if tool == "polygon" else " - drag it out"))
        elif low == "n":
            z = self._selected()
            if z is not None:
                self._end_drag()
                i = self.labels.index(z.name) if z.name in self.labels else -1
                new = z.copy()
                new.name = self.labels[(i + (-1 if k == "N" else 1)) % len(self.labels)]
                self._push()
                self._replace(new)
                self._say(f"{z.id} named {new.name!r}" if new.name else f"{z.id}: no name")
        elif low in ("delete", "backspace", "del"):
            if st.poly:
                st.poly.pop()
                self._say(f"polygon: {len(st.poly)} corner(s)")
            else:
                z = self._selected()
                if z is not None:
                    self._end_drag()            # the zone under the cursor goes: so does its drag
                    self._push()
                    st.zones = [o for o in st.zones if o.id != z.id]
                    st.selected = None
                    self._say(f"{z.id} removed (Z undoes)")
        elif low in ("z", "ctrl+z"):
            self.undo()
        elif low == "g":
            st.show_grid = not st.show_grid
        elif low == "h":
            st.show_legend = not st.show_legend
        elif low == "tab":
            if st.zones:
                self._end_drag()
                ids = [z.id for z in st.zones]
                st.selected = ids[(ids.index(st.selected) + 1) % len(ids)] \
                    if st.selected in ids else ids[0]
        elif k in ("[", "]"):
            self._rotate(FINE_DEG if k == "]" else -FINE_DEG)
        elif k in ("{", "}"):
            self._rotate(COARSE_DEG if k == "}" else -COARSE_DEG)
        elif low in ("left", "right", "up", "down"):
            dx, dy = {"left": (-1, 0), "right": (1, 0), "up": (0, -1), "down": (0, 1)}[low]
            self._nudge(dx * NUDGE_PX, dy * NUDGE_PX)
        elif low in ("enter", "return"):
            if st.poly:
                self._poly_close()
        elif low == "space":
            if st.poly or self._drag is not None:
                self._say("finish the shape first (Enter closes the polygon), then Space")
            elif not st.zones:
                self._say("no zones yet: drag one out first (Esc cancels)")
            else:
                probs = [p for z in st.zones for p in zone_problems(z, st.shape)]
                if probs:
                    self._say("not saved: " + probs[0])
                else:
                    st.done, st.preview = "save", None
                    self._say(f"confirmed: {len(st.zones)} zone(s)")
        elif low in ("escape", "esc"):
            d = self._drag
            if st.poly or d is not None:
                if d is not None and d.get("pushed"):
                    self.undo()                 # a move / resize in progress: back where it was
                st.poly, st.preview, self._drag = [], None, None
                self._say("dropped (Esc again cancels without saving)")
            else:
                self.cancel("cancelled: nothing saved")
        else:
            self.dirty = False                  # an unbound key changes nothing
        return st

    # ------------------------------------------------------------ scripted events
    def handle(self, ev) -> PainterState:
        """One event (a mapping, see the module doc) -> state. A malformed event is a
        ValueError: a typo in a scripted session never silently does nothing."""
        if not isinstance(ev, dict) or "type" not in ev:
            raise ValueError(f"a painter event is a mapping with a type, got {ev!r}")
        t = ev["type"]

        def at(key="at"):
            p = ev.get(key)
            if p is None and key == "at" and "x" in ev and "y" in ev:
                p = (ev["x"], ev["y"])
            if not (isinstance(p, (list, tuple)) and len(p) == 2
                    and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                            and math.isfinite(v) for v in p)):
                raise ValueError(f"event {ev!r}: {key} must be [col, row] projector px")
            return float(p[0]), float(p[1])

        shift = bool(ev.get("shift", False))
        if t == "key":
            if not isinstance(ev.get("key"), str) or not ev["key"]:
                raise ValueError(f"event {ev!r}: key must be text (e.g. \"e\", \"space\")")
            return self.on_key(ev["key"])
        if t == "down":
            return self.on_mouse_down(*at(), button=ev.get("button", "left"), shift=shift)
        if t == "move":
            if ev.get("left") is False and self._drag is not None:
                return self.on_mouse_up(*at())  # the button was released outside the window
            return self.on_mouse_move(*at())
        if t == "up":
            return self.on_mouse_up(*at(), button=ev.get("button", "left"))
        if t == "dblclick":
            return self.on_double_click(*at(), shift=shift)
        if t == "wheel":
            delta = ev.get("delta", 1)
            if isinstance(delta, bool) or not isinstance(delta, (int, float)) \
                    or not math.isfinite(delta):
                raise ValueError(f"event {ev!r}: delta must be wheel notches (a number)")
            return self.on_wheel(*at(), delta, shift=shift)
        if t == "click":
            x, y = at()
            self.on_mouse_down(x, y, button=ev.get("button", "left"), shift=shift)
            return self.on_mouse_up(x, y, button=ev.get("button", "left"))
        if t == "drag":
            (x0, y0), (x1, y1) = at("from"), at("to")
            n = max(1, int(ev.get("steps", 4)))
            self.on_mouse_down(x0, y0, shift=shift)
            for i in range(1, n + 1):
                self.on_mouse_move(x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n)
            return self.on_mouse_up(x1, y1)
        raise ValueError(f"event {ev!r}: type must be key, down, move, up, click, dblclick, "
                         "wheel or drag")

    def zones(self) -> list:
        return [z.copy() for z in self.state.zones]


def load_events(path) -> list:
    """A scripted session file (.json): a list of events, or {"events": [...]}."""
    p = Path(path)
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except OSError as e:
        raise ValueError(f"events file {p}: {e.strerror or e}") from None
    except ValueError as e:
        raise ValueError(f"events file {p.name}: not valid JSON ({e})") from None
    return check_events(doc.get("events") if isinstance(doc, dict) else doc)


def check_events(events) -> list:
    if not isinstance(events, list) or not all(isinstance(e, dict) and "type" in e
                                               for e in events):
        raise ValueError("a scripted painter session is a list of events [{\"type\": ...}, ...] "
                         "(or {\"events\": [...]})")
    return [dict(e) for e in events]


# a short session for the mock UI and the docs: two rectangles (one turned), an ellipse, a
# polygon, names, confirmed
DEMO_EVENTS = (
    {"type": "drag", "from": [180, 260], "to": [420, 470]},
    {"type": "key", "key": "n"},
    {"type": "drag", "from": [520, 300], "to": [760, 430]},
    {"type": "wheel", "at": [640, 365], "delta": 3},
    {"type": "key", "key": "e"},
    {"type": "drag", "from": [240, 620], "to": [460, 800]},
    {"type": "key", "key": "p"},
    {"type": "click", "at": [560, 620]},
    {"type": "click", "at": [760, 660]},
    {"type": "click", "at": [720, 840]},
    {"type": "dblclick", "at": [540, 800]},
    {"type": "key", "key": "space"},
)


# ------------------------------------------------------------------ the picture

@functools.lru_cache(maxsize=4)
def _grid(spacing: int, shape: tuple) -> np.ndarray:
    """The lettered grid of `scan grid` as a dim underlay: thin lines and small cell names,
    no fill (the field stays dark)."""
    import cv2
    from ..capture import zones as gz
    H, W = shape
    spec = gz.grid_spec(spacing)
    img = np.zeros((H, W), np.uint8)
    for k in range(1, spec.n_cols):
        img[:, min(k * spacing, W - 1)] = GRID_LEVEL
    for k in range(1, spec.n_rows):
        img[min(k * spacing, H - 1), :] = GRID_LEVEL
    font = cv2.FONT_HERSHEY_SIMPLEX
    for cell in gz.all_cells(spec):
        x0, y0, x1, y1 = gz.cell_rect(cell, spec)
        if x1 - x0 < 40 or y1 - y0 < 24:
            continue                            # a sliver at the DMD edge
        cv2.putText(img, gz.cell_name(cell, spec), (x0 + 5, y0 + 16), font, 0.42, GRID_TEXT, 1,
                    cv2.LINE_AA)
    img.setflags(write=False)
    return img


def _pts(a) -> np.ndarray:
    return np.rint(np.asarray(a, np.float64)).astype(np.int32).reshape(-1, 1, 2)


def _square(img, x, y, half: int, level: int, filled: bool = True) -> None:
    import cv2
    p0 = (int(round(x)) - half, int(round(y)) - half)
    p1 = (int(round(x)) + half, int(round(y)) + half)
    cv2.rectangle(img, p0, p1, int(level), -1 if filled else 1)


def _label(img, z: Zone, level: int, name: Optional[str] = None) -> None:
    """The zone's id (or `name` in its place: a key "<set>/<id>") + its label, above it."""
    import cv2
    H, W = img.shape[:2]
    text = (name or z.id) + (f" {z.name}" if z.name else "")
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    (tw, th), base = cv2.getTextSize(text, font, scale, thick)
    x0, y0, _x1, _y1 = z.bbox()
    x = int(min(max(x0, 2), W - tw - 2))
    y = int(y0 - 7) if y0 - 7 - th >= 2 else int(min(y0 + th + 9, H - base - 2))
    cv2.putText(img, text, (x, y), font, scale, int(level), thick, cv2.LINE_AA)


def legend_height(state: PainterState) -> int:
    return (len(LEGEND) + 1) * LEGEND_LINE_PX + 2 * LEGEND_PAD if state.show_legend else 0


def status_line(state: PainterState) -> str:
    z = state.zone(state.selected)
    sel = ""
    if z is not None:
        x0, y0, x1, y1 = z.bbox()
        size = (f"{z.size[0]:.0f} x {z.size[1]:.0f} px, {z.rotation_deg:g} deg"
                if z.shape != "polygon"
                else f"{len(z.points)} corners, {x1 - x0:.0f} x {y1 - y0:.0f} px")
        sel = f"   {z.id}{' ' + z.name if z.name else ''}: {z.shape} {size}"
    head = f"{state.name or 'zones'}   tool {state.tool.upper()}   {len(state.zones)} zone(s){sel}"
    return head + (f"   | {state.message}" if state.message else "")


def render(state: PainterState) -> np.ndarray:
    """A painter state -> the DMD image, (rows, cols) uint8, as it should LOOK on the part
    (255 = lit): outlines, handles, zone ids, the cursor's crosshair and the legend strip on
    black. Polarity (an inverting HDMI path) is the window shell's business, not this one's."""
    import cv2
    H, W = state.shape
    img = (_grid(state.grid_spacing, (H, W)).copy() if state.show_grid
           else np.zeros((H, W), np.uint8))
    lh = legend_height(state)
    if lh:
        # the legend strip is drawn FIRST: a zone, the shape being drawn and the cursor's
        # crosshair stay visible when they are in its rows (only the grid underlay is not)
        img[H - lh:, :] = 0
        img[H - lh, :] = GRID_LEVEL
        font = cv2.FONT_HERSHEY_SIMPLEX
        lines = (status_line(state),) + LEGEND
        scale = 0.42
        widest = max(cv2.getTextSize(t, font, scale, 1)[0][0] for t in lines)
        if widest > W - 12:
            scale *= (W - 12) / widest
        for i, text in enumerate(lines):
            y = H - lh + LEGEND_PAD + (i + 1) * LEGEND_LINE_PX - 3
            cv2.putText(img, text, (6, y), font, scale, BRIGHT if i == 0 else LEGEND_LEVEL, 1,
                        cv2.LINE_AA)
    sel = state.zone(state.selected)
    for z in state.zones:
        if z is not sel:
            cv2.polylines(img, [_pts(z.outline())], True, OUTLINE, OUTLINE_PX, cv2.LINE_AA)
            _label(img, z, OUTLINE)
    if sel is not None:
        cv2.polylines(img, [_pts(sel.outline())], True, BRIGHT, SELECTED_PX, cv2.LINE_AA)
        for _part, hx, hy in handles(sel):
            _square(img, hx, hy, HANDLE_PX, BRIGHT)
        _label(img, sel, BRIGHT)
    if state.preview is not None:
        cv2.polylines(img, [_pts(state.preview.outline())], True, BRIGHT, 1, cv2.LINE_8)
    if state.poly:
        if len(state.poly) > 1:
            cv2.polylines(img, [_pts(state.poly)], False, BRIGHT, OUTLINE_PX, cv2.LINE_AA)
        if state.cursor is not None:
            a, b = state.poly[-1], state.cursor
            cv2.line(img, (int(round(a[0])), int(round(a[1]))),
                     (int(round(b[0])), int(round(b[1]))), OUTLINE, 1, cv2.LINE_AA)
        for i, (px, py) in enumerate(state.poly):
            _square(img, px, py, HANDLE_PX if i == 0 else 3, BRIGHT, filled=i != 0)
    if state.cursor is not None:
        cx, cy = (int(round(v)) for v in state.cursor)
        cx, cy = min(max(cx, 0), W - 1), min(max(cy, 0), H - 1)
        img[cy, :] = np.maximum(img[cy, :], HAIR_LEVEL)
        img[:, cx] = np.maximum(img[:, cx], HAIR_LEVEL)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):       # a gap at the very centre
            cv2.line(img, (cx + 4 * dx, cy + 4 * dy), (cx + 16 * dx, cy + 16 * dy), BRIGHT, 1)
    return img

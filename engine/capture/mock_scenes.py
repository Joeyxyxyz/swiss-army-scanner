"""
Named mock scenes for the simulated stereo bench (engine/capture/mock_stereo.py shapes in
front of the wall, 823 mm away): what the simulated human places for the example recipes,
in a good and a bad variant, so every example recipe runs end to end in --mock.

    scene:<name>[:<variant>]       e.g. scene:tube, scene:tube:misaligned

    tube       a ribbed tube lying left-right        ok | misaligned (turned 1.2 deg about its
                                                     left end: the right end ~2.8 mm off)
    endfaces   two cylinders, end faces toward       ok | dent (0.4 x 3 mm dent on the right
               the camera                            face) | proud (right face 0.8 mm proud)
    box_lid    a box with its lid on                 ok | ajar (lid tilted 3 deg about its left
                                                     edge) | missing (no box)
    plate      a flat plate                          ok | bump (0.8 mm bump on the top face) |
                                                     burr (a 3 mm tab on the top edge)

Frame: camera mm (x right, y down, z forward); the wall is z = 823, so a point's height
above the wall is 823 - z. Part-frame (u, v) of the recipes ~ (x, y) at the wall.
"""

from __future__ import annotations

import math

import numpy as np

from .mock_stereo import Block, EndFace, MetricCylinder, Sphere

WALL_Z = 823.0


def _rot_z(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0],
                     [0.0, 0.0, 1.0]])


def tube(variant: str = "ok") -> list:
    r, length, y0 = 10.0, 160.0, -20.0
    z = WALL_Z - r - 2.0
    turn = {"ok": 0.0, "misaligned": 1.2}[variant]
    R = _rot_z(turn)
    pivot = np.array([-length / 2, y0, z])

    def place(c):
        return tuple(pivot + R @ (np.asarray(c, float) - pivot))

    ax = tuple(R @ np.array([1.0, 0.0, 0.0]))
    out = [MetricCylinder(center=place((0.0, y0, z)), radius=r, height=length, axis=ax,
                          albedo=0.8)]
    for x in np.arange(-64.0, 65.0, 16.0):                      # the ribs
        out.append(MetricCylinder(center=place((float(x), y0, z)), radius=r + 1.2, height=2.5,
                                  axis=ax, albedo=0.8))
    return out


def endfaces(variant: str = "ok") -> list:
    radius, length = 10.5, 38.0
    n = np.array([0.0, 0.0, -1.0])
    out = []
    for i, x in enumerate((-25.0, 25.0)):
        proud = 0.8 if (variant == "proud" and i == 1) else 0.0
        face_z = WALL_Z - 2.0 - length - proud
        c = np.array([x, 20.0, face_z])
        dents = ([{"depth_mm": 0.4, "size_mm": 3.0, "u_mm": 3.0, "v_mm": -2.0}]
                 if (variant == "dent" and i == 1) else [])
        out.append(EndFace(center=tuple(c), normal=tuple(n), radius=radius, dents=dents,
                           chamfer_mm=0.5, chamfer_depth_mm=0.1, albedo=0.85))
        body = length + proud
        out.append(MetricCylinder(center=tuple(c - n * body / 2), radius=radius, height=body,
                                  axis=tuple(n), albedo=0.5))
    return out


def box_lid(variant: str = "ok") -> list:
    if variant == "missing":
        return []
    out = [Block(center=(0.0, 0.0, WALL_Z - 19.0), half=(45.0, 30.0, 19.0), albedo=0.7)]
    lid = Block(center=(0.0, 0.0, WALL_Z - 39.0), half=(46.0, 31.0, 1.0), albedo=0.75)
    if variant == "ajar":
        a = math.radians(3.0)                     # about the left edge (x = -46, top of the box)
        hinge = np.array([-46.0, 0.0, WALL_Z - 38.0])
        Ry = np.array([[math.cos(a), 0.0, math.sin(a)], [0.0, 1.0, 0.0],
                       [-math.sin(a), 0.0, math.cos(a)]])
        c = hinge + Ry @ (np.array(lid.center) - hinge)
        lid = Block(center=tuple(c), half=lid.half, rot=Ry.T, albedo=0.75)
    out.append(lid)
    return out


def plate(variant: str = "ok") -> list:
    out = [Block(center=(0.0, 0.0, WALL_Z - 3.0), half=(60.0, 40.0, 3.0), albedo=0.9)]
    top = WALL_Z - 6.0
    if variant == "bump":
        R = 15.0
        out.append(Sphere(center=(10.0, 5.0, top + R - 0.8), radius=R, albedo=0.9))
    elif variant == "burr":
        out.append(Block(center=(20.0, -41.5, WALL_Z - 3.0), half=(4.0, 1.5, 3.0), albedo=0.9))
    return out


SCENES = {"tube": (tube, ("ok", "misaligned")),
          "endfaces": (endfaces, ("ok", "dent", "proud")),
          "box_lid": (box_lid, ("ok", "ajar", "missing")),
          "plate": (plate, ("ok", "bump", "burr"))}


def parse(spec: str) -> tuple:
    """'scene:tube:misaligned' | 'tube:misaligned' | 'tube' -> ('tube', 'misaligned')."""
    s = str(spec).strip()
    if s.lower().startswith("scene:"):
        s = s[6:]
    name, _, variant = s.partition(":")
    name, variant = name.strip().lower(), (variant.strip().lower() or "ok")
    if name not in SCENES:
        raise ValueError(f"no mock scene {name!r} (scenes: {', '.join(SCENES)})")
    if variant not in SCENES[name][1]:
        raise ValueError(f"mock scene {name}: variant {variant!r} is not one of "
                         f"{', '.join(SCENES[name][1])}")
    return name, variant


def shapes(spec: str) -> list:
    name, variant = parse(spec)
    return SCENES[name][0](variant)


def place(bench, spec: str) -> str:
    """Clear the simulated stereo bench and place the scene (stands in for the human)."""
    name, variant = parse(spec)
    bench.clear()
    bench.use_stereo()
    for s in SCENES[name][0](variant):
        bench.place(s)
    return f"scene:{name}:{variant}"

"""
Calibration fixture: N flat faces at known, distinct angles on one printed bracket, each
face carrying its own ChArUco board with a DISJOINT marker id range. One capture of the
fixture shows N boards; each is detected on its own (BoardSpec.first_id) and used as an
independent planar view for the camera and the projector (OpenCV gives every planar view
its own board pose), so 3 placements of the fixture give the 12 views that took 12
hand-held board poses before.

MOCK-TESTED ONLY: not yet used on a real rig.

Geometry (all mm). Fixture frame F: origin at the centre of the base plate's top surface,
x along its width, y along its depth, z up out of the plate (toward the camera when the
fixture faces it; with x to the right, y is UP in the image). The base plate (thickness
`base_mm`) has one rectangular pocket per face (`pocket_mm` deep); a solid wedge drops into
each pocket. A wedge's top is the face, tilted by `tilt_deg` and descending toward its
`azimuth_deg` (0 = +x, 90 = +y, 180 = -x, 270 = -y): the face looks outward, so no face
hides another from the front.

Wedge frame W (the part's own, for printing): u along the slope from the LOW edge (u = 0)
to the high edge (u = Lu), v across (+-Lv / 2), z up from the wedge bottom; the face is
z = low_mm + u tan(tilt). In F: X = c + (u - Lu / 2) e_u + v e_v + (z - pocket_mm) z_F with
e_u = -(cos az, sin az, 0) (low -> high, away from where the face looks), e_v = z_F x e_u.

Board on a face: x_board along the slope (low -> high), y_board = -v, z_board INTO the face
(cv2.aruco's convention), its centre on the face centre, lifted `sticker_mm` along the
outward normal (paper + laminate + adhesive). Where on the face the sticker sits, and how it
is turned in the face plane, does not matter for the calibration (every face is its own
view); the consistency check below uses face NORMALS only.

Spec (JSON, written by tools/make_calib_fixture.py, read by calib-pose --fixture):
    {"kind": "calib_fixture", "version": 1, "name", "units": "mm", "frame": ...,
     "faces": [{"face", "board": BoardSpec dict, "id_range": [a, b], "tilt_deg",
                "azimuth_deg", "R", "t" (board -> fixture: X_F = R X_board + t),
                "normal" (outward, F), "square_mm_measured": null | caliper value}],
     "parts": [...], "design": {...}}
Fill each face's `square_mm_measured` with the caliper value of a printed square: the
printer's scale error goes straight into every mm the scanner reports.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Optional, Union

import numpy as np

from .charuco import BoardSpec, dictionary_size

KIND = "calib_fixture"
VERSION = 1
FRAME = ("fixture F: origin at the centre of the base plate's top surface, x along its "
         "width, y along its depth, z up out of the plate (toward the camera when the "
         "fixture faces it; x right -> y up in the image)")
ANGLE_WARN_DEG = 1.0                    # face-to-face angle residual worth a warning
MIN_FACES, MAX_FACES = 2, 8


# ------------------------------------------------------------------ design

def _azimuths(n: int, cols: int) -> list:
    """Cells clockwise around a 2 x cols grid (top row left -> right, bottom row right ->
    left) -> [(col, row, azimuth)]; row 1 = top (+y). The first top cell looks -x, the
    other top cells +y; the first bottom cell (rightmost) looks +x, the others -y."""
    cells = [(c, 1) for c in range(cols)] + [(c, 0) for c in reversed(range(cols))]
    out = []
    for k, (c, r) in enumerate(cells[:n]):
        if r == 1:
            az = 180.0 if c == 0 else 90.0
        else:
            az = 0.0 if c == cols - 1 else 270.0
        out.append((c, r, az))
    return out


def _wedge_frame(az_deg: float) -> tuple:
    a = math.radians(az_deg)
    e_u = -np.array([math.cos(a), math.sin(a), 0.0])
    e_u[np.abs(e_u) < 1e-12] = 0.0
    e_v = np.cross([0.0, 0.0, 1.0], e_u)
    return e_u, e_v


def design(n_faces: int = 4, square_mm: float = 16.0, squares=(6, 5),
           marker_ratio: float = 0.7, dictionary: str = "DICT_5X5_250", first_id: int = 100,
           tilt_deg=30.0, sticker_margin_mm: float = 3.0, face_border_mm: float = 2.0,
           sticker_mm: float = 0.3, face_style: str = "ridges", bed_mm=(220.0, 220.0),
           base_mm: float = 8.0, pocket_mm: float = 3.0, low_mm: float = 6.0,
           wall_mm: float = 5.0, clearance_mm: float = 0.2, name: str = "fixture") -> dict:
    """The parametric fixture -> spec dict (faces with boards + poses, printable parts).
    squares = (along the slope, across) per face; tilt_deg: one value or one per face.
    Raises ValueError when a part does not fit the bed (fewer / smaller squares)."""
    n = int(n_faces)
    if not MIN_FACES <= n <= MAX_FACES:
        raise ValueError(f"n_faces must be {MIN_FACES}..{MAX_FACES}, got {n_faces}")
    if face_style not in ("ridges", "recess", "flat"):
        raise ValueError(f"face_style must be ridges, recess or flat, got {face_style!r}")
    tilts = [float(tilt_deg)] * n if np.isscalar(tilt_deg) else [float(t) for t in tilt_deg]
    if len(tilts) != n or not all(10.0 <= t <= 45.0 for t in tilts):
        raise ValueError(f"tilt_deg: one value or {n} values, each 10..45 deg, got {tilt_deg!r}")
    su, sv = (int(squares[0]), int(squares[1]))
    sq = float(square_mm)
    probe = BoardSpec(su, sv, sq, sq * float(marker_ratio), dictionary, int(first_id))
    size = dictionary_size(dictionary)
    if size is not None and int(first_id) + n * probe.n_markers > size:
        raise ValueError(f"{n} faces x {probe.n_markers} markers from id {first_id} do not fit "
                         f"{dictionary} ({size} markers)")
    sticker = (su * sq + 2 * sticker_margin_mm, sv * sq + 2 * sticker_margin_mm)
    face = (sticker[0] + 2 * face_border_mm, sticker[1] + 2 * face_border_mm)
    cols = math.ceil(n / 2)
    layout = _azimuths(n, cols)
    feet = []
    for t in tilts:
        Lu, Lv = face[0] * math.cos(math.radians(t)), face[1]
        feet.append((Lu, Lv, low_mm + Lu * math.tan(math.radians(t))))
    cell = max(max(f[0], f[1]) for f in feet) + 2 * clearance_mm
    bed = (float(bed_mm[0]), float(bed_mm[1]))
    fit = (bed[0] - 10.0, bed[1] - 10.0)                     # 5 mm clear of the bed edge
    W = cols * cell + (cols + 1) * wall_mm
    D = 2 * cell + 3 * wall_mm
    per_tile, seams = cols, 0
    if W > fit[0]:
        # split the base along x between pocket columns: every tile keeps a full wall at
        # both ends (the seam wall is doubled) and a splice plate is glued under each seam
        per_tile = int((fit[0] - wall_mm) // (cell + wall_mm))
        if per_tile < 1:
            raise ValueError(f"one face cell ({cell:.1f} mm) does not fit the {bed[0]:g} mm "
                             "bed: use fewer or smaller squares")
        seams = math.ceil(cols / per_tile) - 1
        W += seams * wall_mm
    if D > fit[1]:
        raise ValueError(f"the base plate is {D:.1f} mm deep, the bed {bed[1]:g} mm: use "
                         "fewer or smaller squares, or a smaller tilt")
    xs_c, x = [], -W / 2 + wall_mm
    for c in range(cols):
        if c and c % per_tile == 0:
            x += wall_mm                                      # the doubled seam wall
        xs_c.append(x + cell / 2)
        x += cell + wall_mm
    ys_c = {0: -D / 2 + wall_mm + cell / 2, 1: D / 2 - wall_mm - cell / 2}
    faces, wedges, pockets = [], [], []
    for k, ((c, r, az), t, (Lu, Lv, hi)) in enumerate(zip(layout, tilts, feet)):
        b = BoardSpec(su, sv, sq, round(sq * float(marker_ratio), 4), dictionary,
                      int(first_id) + k * probe.n_markers)
        centre = np.array([xs_c[c], ys_c[r], 0.0])
        e_u, e_v = _wedge_frame(az)
        ext = np.abs(e_u[:2]) * Lu + np.abs(e_v[:2]) * Lv      # footprint along x, y
        pockets.append({"face": k, "centre": centre[:2].tolist(),
                        "size": (ext + 2 * clearance_mm).tolist()})
        wedge = {"name": f"wedge_{k}", "kind": "wedge", "face": k, "count": 1,
                 "Lu": Lu, "Lv": Lv, "low_mm": low_mm, "tilt_deg": t, "azimuth_deg": az,
                 "centre": centre.tolist(), "pocket_mm": pocket_mm, "face_style": face_style,
                 "sticker_mm_uv": [sticker[0], sticker[1]], "face_mm": [face[0], face[1]],
                 "print": "face-down" if face_style != "recess" else "bottom-down"}
        R, tt, nrm = _board_pose(wedge, sticker_mm, b)
        faces.append({"face": k, "board": b.to_dict(), "id_range": list(b.marker_ids),
                      "tilt_deg": t, "azimuth_deg": az, "R": R.tolist(), "t": tt.tolist(),
                      "normal": nrm.tolist(), "square_mm_measured": None,
                      "sticker_mm": [sticker[0], sticker[1]], "face_mm": [face[0], face[1]]})
        wedges.append(wedge)
    base = {"name": "base", "kind": "base", "count": 1, "W": W, "D": D,
            "base_mm": base_mm, "pocket_mm": pocket_mm, "pockets": pockets,
            "seams_x": _seam_positions(xs_c, cell, wall_mm, per_tile, seams)}
    parts = [base] + wedges
    if seams:
        parts.append({"name": "splice", "kind": "splice", "count": 2 * seams,
                      "size": [40.0, 20.0, 3.0]})
    spec = {"kind": KIND, "version": VERSION, "name": str(name), "units": "mm", "frame": FRAME,
            "faces": faces, "parts": parts,
            "design": {"n_faces": n, "square_mm": sq, "squares": [su, sv],
                       "marker_ratio": float(marker_ratio), "dictionary": dictionary,
                       "first_id": int(first_id), "tilt_deg": tilts,
                       "sticker_margin_mm": sticker_margin_mm, "face_border_mm": face_border_mm,
                       "sticker_mm": sticker_mm, "face_style": face_style,
                       "bed_mm": list(bed), "base_mm": base_mm, "pocket_mm": pocket_mm,
                       "low_mm": low_mm, "wall_mm": wall_mm, "clearance_mm": clearance_mm,
                       "base_size_mm": [W, D], "max_height_mm": max(f[2] for f in feet)
                       - pocket_mm, "base_tiles": seams + 1},
            "status": "mock-tested only, not yet used on a real rig"}
    check_bed(spec)
    return load(spec)


def _seam_positions(xs_c, cell, wall, per_tile, seams) -> list:
    """x of each base seam: the middle of the doubled wall between two tiles."""
    return [(xs_c[c - 1] + xs_c[c]) / 2 for c in range(1, len(xs_c))
            if seams and c % per_tile == 0]


def wedge_to_fixture(w: dict) -> tuple:
    """(R, t): X_F = R X_W + t for a wedge part."""
    e_u, e_v = _wedge_frame(float(w["azimuth_deg"]))
    R = np.stack([e_u, e_v, [0.0, 0.0, 1.0]], axis=1)
    t = np.asarray(w["centre"], float) - (float(w["Lu"]) / 2) * e_u \
        - np.array([0.0, 0.0, float(w["pocket_mm"])])
    return R, t


def face_plane_w(w: dict) -> tuple:
    """(point, outward unit normal) of the wedge's face plane (ridge tops) in W."""
    a = math.radians(float(w["tilt_deg"]))
    u_c = float(w["Lu"]) / 2
    return (np.array([u_c, 0.0, float(w["low_mm"]) + u_c * math.tan(a)]),
            np.array([-math.sin(a), 0.0, math.cos(a)]))


def _board_pose(w: dict, sticker_mm: float, b: BoardSpec) -> tuple:
    """Board -> fixture (R, t) and the face's outward normal in F."""
    a = math.radians(float(w["tilt_deg"]))
    p0, n_w = face_plane_w(w)
    x_b = np.array([math.cos(a), 0.0, math.sin(a)])         # up the slope
    z_b = -n_w                                                # into the face
    y_b = np.cross(z_b, x_b)
    bw, bh = b.board_size_mm
    o_w = p0 + float(sticker_mm) * n_w - (bw / 2) * x_b - (bh / 2) * y_b
    Rw, tw = wedge_to_fixture(w)
    R = Rw @ np.stack([x_b, y_b, z_b], axis=1)
    t = Rw @ o_w + tw
    R[np.abs(R) < 1e-15] = 0.0
    return R, t, Rw @ n_w


def check_bed(spec: dict) -> None:
    """Every part fits the print bed (5 mm clear of its edge), in its print orientation."""
    d = spec["design"]
    fit = (d["bed_mm"][0] - 10.0, d["bed_mm"][1] - 10.0)
    for p in spec["parts"]:
        for w, h in [_footprint(p, spec)]:
            if not (min(w, h) <= min(fit) and max(w, h) <= max(fit)):
                raise ValueError(f"part {p['name']} is {w:.1f} x {h:.1f} mm on the bed; the "
                                 f"bed is {d['bed_mm'][0]:g} x {d['bed_mm'][1]:g} mm: use fewer "
                                 "or smaller squares")


def _footprint(p: dict, spec: dict) -> tuple:
    if p["kind"] == "base":
        xs = [-p["W"] / 2] + list(p.get("seams_x") or []) + [p["W"] / 2]
        return max(b - a for a, b in zip(xs, xs[1:])), p["D"]
    if p["kind"] == "wedge":
        if p["print"] == "face-down":
            return tuple(p["face_mm"])
        return p["Lu"], p["Lv"]
    return tuple(p["size"][:2])


# ------------------------------------------------------------------ spec I/O

def load(spec: Union[str, Path, dict]) -> dict:
    """A fixture spec (path or dict) -> the validated dict. Faces need distinct, disjoint
    marker id ranges of one dictionary."""
    if not isinstance(spec, dict):
        p = Path(spec)
        try:
            spec = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise ValueError(f"fixture spec {p}: {e}") from None
    if spec.get("kind") != KIND:
        raise ValueError(f"not a calibration fixture spec (kind {spec.get('kind')!r}, want "
                         f"{KIND!r})")
    if int(spec.get("version", 0)) != VERSION:
        raise ValueError(f"fixture spec version {spec.get('version')!r}: this engine reads "
                         f"version {VERSION}")
    faces = spec.get("faces") or []
    if len(faces) < MIN_FACES:
        raise ValueError(f"a fixture spec needs >= {MIN_FACES} faces, got {len(faces)}")
    spans = []
    for f in faces:
        b = BoardSpec.from_dict(f["board"])
        R = np.asarray(f["R"], float)
        if R.shape != (3, 3) or np.abs(R.T @ R - np.eye(3)).max() > 1e-6 \
                or np.asarray(f["t"], float).shape != (3,):
            raise ValueError(f"face {f.get('face')}: R must be a rotation and t a 3-vector")
        m = f.get("square_mm_measured")
        if m is not None and not (isinstance(m, (int, float)) and not isinstance(m, bool)
                                  and 0.5 * b.square_mm < float(m) < 1.5 * b.square_mm):
            raise ValueError(f"face {f.get('face')}: square_mm_measured {m!r} is not a caliper "
                             f"value near the nominal {b.square_mm:g} mm")
        spans.append((b.dictionary, *b.marker_ids, f.get("face")))
    if len({s[0] for s in spans}) != 1:
        raise ValueError("fixture faces must share one marker dictionary")
    spans.sort(key=lambda s: s[1])
    for a, b in zip(spans, spans[1:]):
        if b[1] <= a[2]:
            raise ValueError(f"faces {a[3]} and {b[3]} share marker ids ({a[1]}..{a[2]} vs "
                             f"{b[1]}..{b[2]}): each face needs its own id range")
    return spec


def save(spec: dict, path: Path) -> Path:
    with open(path, "x", encoding="utf-8") as f:
        json.dump(spec, f, indent=2)
        f.write("\n")
    return Path(path)


def fingerprint(spec: dict) -> str:
    """Short hash of the faces (boards + poses): which fixture a capture was made with."""
    core = [{k: f[k] for k in ("board", "R", "t")} for f in spec["faces"]]
    return hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()[:12]


def boards(spec: dict) -> list:
    """One BoardSpec per face, scaled to the caliper-measured square where it is given."""
    out = []
    for f in spec["faces"]:
        b = BoardSpec.from_dict(f["board"])
        m = f.get("square_mm_measured")
        out.append(b.scaled(float(m)) if m else b)
    return out


def face_pose(face: dict) -> tuple:
    """(R, t) board -> fixture."""
    return np.asarray(face["R"], float), np.asarray(face["t"], float)


def summary(spec: dict) -> dict:
    return {"name": spec.get("name"), "fingerprint": fingerprint(spec),
            "n_faces": len(spec["faces"]),
            "faces": [{"face": f["face"], "id_range": f["id_range"],
                       "tilt_deg": f["tilt_deg"], "azimuth_deg": f["azimuth_deg"],
                       "square_mm_measured": f.get("square_mm_measured")}
                      for f in spec["faces"]]}


# ------------------------------------------------------------------ consistency check

def face_angles(spec: dict) -> dict:
    """{(i, j): angle between the face normals, deg} from the spec."""
    n = [np.asarray(f["R"], float)[:, 2] for f in spec["faces"]]
    out = {}
    for i in range(len(n)):
        for j in range(i + 1, len(n)):
            out[(i, j)] = math.degrees(math.acos(float(np.clip(n[i] @ n[j], -1, 1))))
    return out


def consistency(spec: dict, views: list) -> dict:
    """The print / assembly sanity check, NOT a constraint of the calibration: for every
    placement, the angle between each pair of detected face normals (from the camera's own
    board poses) against the spec. views: [{"placement", "face", "R" (board -> camera)}].
    -> {"pairs": [...], "max_abs_residual_deg", "ok", "warnings"}"""
    ref = face_angles(spec)
    by_pl = {}
    for v in views:
        by_pl.setdefault(v["placement"], {})[int(v["face"])] = np.asarray(v["R"], float)[:, 2]
    acc = {}
    for pl, fs in by_pl.items():
        ks = sorted(fs)
        for a in range(len(ks)):
            for b in range(a + 1, len(ks)):
                i, j = ks[a], ks[b]
                ang = math.degrees(math.acos(float(np.clip(fs[i] @ fs[j], -1, 1))))
                acc.setdefault((i, j), []).append(ang - ref[(i, j)])
    pairs = [{"faces": [i, j], "spec_deg": round(ref[(i, j)], 4),
              "residual_mean_deg": round(float(np.mean(r)), 4),
              "residual_max_abs_deg": round(float(np.max(np.abs(r))), 4), "n": len(r)}
             for (i, j), r in sorted(acc.items())]
    worst = max((p["residual_max_abs_deg"] for p in pairs), default=None)
    warnings = []
    if worst is None:
        warnings.append("no placement showed two faces: the face-angle check was not made")
    elif worst > ANGLE_WARN_DEG:
        warnings.append(f"face-to-face angles differ from the fixture spec by up to {worst:.2f} "
                        f"deg (> {ANGLE_WARN_DEG:g}): check the print, the wedges in their "
                        "pockets and that each sticker lies flat. Each face is still its own "
                        "view, so the calibration itself does not rest on these angles")
    return {"pairs": pairs, "max_abs_residual_deg": worst,
            "ok": worst is not None and worst <= ANGLE_WARN_DEG, "warn_deg": ANGLE_WARN_DEG,
            "units": "deg (measured - spec)", "warnings": warnings}

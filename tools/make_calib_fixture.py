"""
make_calib_fixture - print files for the calibration fixture and the check block.

    python tools/make_calib_fixture.py --out DIR                     # 4 faces + check block
    python tools/make_calib_fixture.py --out DIR --faces 3 --square 18 --squares 5,4
    python tools/make_calib_fixture.py --out DIR --face-style recess --bed 250,210
    python tools/make_calib_fixture.py --out DIR --no-block | --block-only

MOCK-TESTED ONLY: the fixture and the block have not yet been used on a real rig.

The FIXTURE (engine/calibrate/fixture.py) is a base plate with one pocket per face and one
solid wedge per face; each wedge's top is a flat face tilted by --tilt and carries its own
ChArUco sticker with a disjoint marker id range, so ONE capture of the fixture is one planar
calibration view per face (`scan calib-pose --fixture <spec.json>`, 3 placements instead of
12-20 hand-held board poses). The CHECK BLOCK (engine/measure/checkblock.py) is a step
gauge, a flat reference face, a half-round bar and a dome: `scan accuracy-block` scans it
and compares with what YOUR print measures.

Written into a fresh folder <out>/<timestamp>_<name>/ (never over an existing file):
    <name>_spec.json                    the fixture spec (faces, boards, poses, parts)
    <name>_face<k>_ids<a>-<b>.png/.pdf  one sticker per face at exact print scale
    <name>_base[_tile<i>].stl, <name>_wedge_<k>.stl, [<name>_splice.stl]
    <block>_spec.json, <block>.stl, <block>_measured_TEMPLATE.json
    README.txt                          print, assembly and measuring steps
STLs are closed (every edge shared by exactly two triangles), mm, already in print
orientation. Pure numpy (+ cv2 for the sticker images): no CAD package.

This tool never talks to the camera or the projector.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.calibrate import charuco, fixture as fx  # noqa: E402
from engine.measure import checkblock as cb  # noqa: E402

EPS_MM = 0.05                   # a vertical wall in a heightfield: two grid lines this apart
RIDGE_PITCH_MM = 4.0            # face ridges: 2 mm ridge, 2 mm groove
RIDGE_GROOVE_MM = 2.0
RIDGE_DEPTH_MM = 0.4            # groove depth, perpendicular to the face
RECESS_DEPTH_MM = 0.4           # sticker recess (face_style "recess"), perpendicular
SPLICE_MM = (40.0, 20.0, 3.0)


# ------------------------------------------------------------------ meshes

def closed_heightfield(xs, ys, H) -> tuple:
    """A closed solid over the rectangle xs x ys: top z = H (ny, nx) (> 0 everywhere),
    bottom z = 0, vertical skirt around the outline. -> (verts (n, 3), faces (m, 3)),
    triangles counter-clockwise seen from outside (outward normals)."""
    xs, ys, H = np.asarray(xs, float), np.asarray(ys, float), np.asarray(H, float)
    ny, nx = H.shape
    if (nx, ny) != (len(xs), len(ys)) or nx < 2 or ny < 2:
        raise ValueError("H must be (len(ys), len(xs)), at least 2 x 2")
    if not (np.all(np.diff(xs) > 0) and np.all(np.diff(ys) > 0)):
        raise ValueError("xs and ys must be strictly increasing")
    if not np.all(H > 0):
        raise ValueError("a closed heightfield needs a height > 0 everywhere")
    X, Y = np.meshgrid(xs, ys)
    top = np.stack([X, Y, H], -1).reshape(-1, 3)
    idx = np.arange(nx * ny).reshape(ny, nx)
    a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]
    faces = [np.stack([a, b, c], -1).reshape(-1, 3), np.stack([a, c, d], -1).reshape(-1, 3)]
    ring = np.r_[idx[0, :], idx[1:, -1], idx[-1, -2::-1], idx[-2:0:-1, 0]]    # ccw from above
    nt = len(top)
    bottom = top[ring].copy()
    bottom[:, 2] = 0.0
    centre = np.array([[(xs[0] + xs[-1]) / 2, (ys[0] + ys[-1]) / 2, 0.0]])
    verts = np.vstack([top, bottom, centre])
    nb = len(ring)
    k = np.arange(nb)
    k1 = (k + 1) % nb
    tb, bb, cc = ring, nt + k, nt + nb
    faces.append(np.stack([tb[k], bb, nt + k1], -1))              # skirt, outward
    faces.append(np.stack([tb[k], nt + k1, tb[k1]], -1))
    faces.append(np.stack([np.full(nb, cc), nt + k1, bb], -1))     # bottom fan, facing down
    return verts, np.vstack(faces).astype(np.int64)


def transform(verts: np.ndarray, R: np.ndarray, t=np.zeros(3)) -> np.ndarray:
    return np.asarray(verts, float) @ np.asarray(R, float).T + np.asarray(t, float)


def to_bed(verts: np.ndarray) -> np.ndarray:
    """Translate so the part sits on z = 0, centred on x = y = 0."""
    lo, hi = verts.min(0), verts.max(0)
    return verts - np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])


def mesh_report(verts: np.ndarray, faces: np.ndarray) -> dict:
    """Closedness check: every undirected edge in exactly two triangles, once in each
    direction (a consistently oriented closed surface), plus volume and size."""
    f = np.asarray(faces, np.int64)
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    und = np.sort(e, axis=1)
    _, cnt = np.unique(und, axis=0, return_counts=True)
    _, dcnt = np.unique(e, axis=0, return_counts=True)
    v = np.asarray(verts, float)
    vol = float(np.einsum("ij,ij->i", v[f[:, 0]], np.cross(v[f[:, 1]], v[f[:, 2]])).sum() / 6)
    area = np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1) / 2
    return {"n_vertices": int(len(v)), "n_triangles": int(len(f)),
            "edges_not_in_2_triangles": int((cnt != 2).sum()),
            "directed_edges_repeated": int((dcnt != 1).sum()),
            "degenerate_triangles": int((area < 1e-9).sum()),
            "closed": bool((cnt == 2).all() and (dcnt == 1).all()),
            "volume_mm3": vol, "size_mm": (v.max(0) - v.min(0)).round(3).tolist()}


def write_stl(path: Path, verts: np.ndarray, faces: np.ndarray, header: str = "",
              ascii_: bool = False) -> Path:
    """Binary (default) or ASCII STL, exclusive create."""
    if not ascii_:
        from engine.reconstruct.mesh import write_stl as wb
        return wb(Path(path), verts, faces, header)
    v = np.asarray(verts, float)
    f = np.asarray(faces, np.int64)
    n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
    lines = [f"solid {header or 'part'}"]
    for k in range(len(f)):
        lines.append(f" facet normal {n[k, 0]:.6e} {n[k, 1]:.6e} {n[k, 2]:.6e}\n  outer loop")
        lines += [f"   vertex {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}" for p in v[f[k]]]
        lines.append("  endloop\n endfacet")
    lines.append(f"endsolid {header or 'part'}\n")
    with open(path, "x", encoding="ascii", newline="\n") as fh:
        fh.write("\n".join(lines))
    return Path(path)


def read_stl(path: Path) -> tuple:
    """STL (binary or ASCII) -> (verts, faces), identical vertices welded."""
    data = Path(path).read_bytes()
    if data[:5] == b"solid" and b"facet" in data[:400]:
        nums = [float(t) for line in data.decode("ascii").splitlines()
                if line.strip().startswith("vertex") for t in line.split()[1:4]]
        tri = np.asarray(nums, np.float32).reshape(-1, 3, 3)
    else:
        n = struct.unpack("<I", data[80:84])[0]
        rec = np.frombuffer(data[84:84 + 50 * n], dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)),
                                                         ("attr", "<u2")])
        tri = rec["v"]
    verts, inv = np.unique(tri.reshape(-1, 3), axis=0, return_inverse=True)
    return verts.astype(np.float64), inv.reshape(-1, 3)


# ------------------------------------------------------------------ fixture parts

def _pairs(vals) -> list:
    out = []
    for v in vals:
        out += [v - EPS_MM, v + EPS_MM]
    return out


def wedge_mesh(w: dict) -> tuple:
    """A wedge part -> (verts, faces) in its print orientation (face-down for ridges /
    flat: the face is printed on the bed, as flat as the bed; bottom-down for recess)."""
    a = math.radians(float(w["tilt_deg"]))
    Lu, Lv = float(w["Lu"]), float(w["Lv"])
    Su, Sv = (float(v) for v in w["sticker_mm_uv"])
    Fu = float(w["face_mm"][0])
    us, vs = [0.0, Lu], [-Lv / 2, Lv / 2]
    style = w["face_style"]
    grooves = []
    if style == "ridges":
        rw = RIDGE_PITCH_MM - RIDGE_GROOVE_MM
        n = int((Sv - rw) // RIDGE_PITCH_MM) + 1
        v0 = -((n - 1) * RIDGE_PITCH_MM + rw) / 2
        grooves = [(v0 + k * RIDGE_PITCH_MM + rw, v0 + (k + 1) * RIDGE_PITCH_MM)
                   for k in range(n - 1)]
        vs += _pairs([g for gr in grooves for g in gr])
    u_s = ((Fu - Su) / 2 * math.cos(a), (Fu + Su) / 2 * math.cos(a))
    if style == "recess":
        us += _pairs(u_s)
        vs += _pairs([-Sv / 2, Sv / 2])
    us, vs = np.unique(np.round(us, 6)), np.unique(np.round(vs, 6))
    U, V = np.meshgrid(us, vs)
    H = float(w["low_mm"]) + U * math.tan(a)
    if grooves:
        g = np.zeros_like(V, bool)
        for lo, hi in grooves:
            g |= (V > lo + EPS_MM / 2) & (V < hi - EPS_MM / 2)
        H = H - np.where(g, RIDGE_DEPTH_MM / math.cos(a), 0.0)
    if style == "recess":
        r = ((U > u_s[0] + EPS_MM / 2) & (U < u_s[1] - EPS_MM / 2)
             & (np.abs(V) < Sv / 2 - EPS_MM / 2))
        H = H - np.where(r, RECESS_DEPTH_MM / math.cos(a), 0.0)
    verts, faces = closed_heightfield(us, vs, H)
    if w["print"] == "face-down":
        b = a + math.pi                                  # face normal -> -z: face on the bed
        Ry = np.array([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]])
        verts = transform(verts, Ry)
    return to_bed(verts), faces


def base_meshes(base: dict) -> list:
    """The base plate (or its tiles) -> [(name suffix, verts, faces)], bottom-down."""
    W, Dd = float(base["W"]), float(base["D"])
    T, dp = float(base["base_mm"]), float(base["pocket_mm"])
    xs, ys = [-W / 2, W / 2], [-Dd / 2, Dd / 2]
    for p in base["pockets"]:
        (cx, cy), (sx, sy) = p["centre"], p["size"]
        xs += _pairs([cx - sx / 2, cx + sx / 2])
        ys += _pairs([cy - sy / 2, cy + sy / 2])
    cuts = list(np.round([-W / 2] + list(base.get("seams_x") or []) + [W / 2], 6))
    xs = np.unique(np.round(xs + cuts, 6))
    ys = np.unique(np.round(ys, 6))
    out = []
    for i, (x0, x1) in enumerate(zip(cuts, cuts[1:])):
        xt = xs[(xs >= x0 - 1e-7) & (xs <= x1 + 1e-7)]
        X, Y = np.meshgrid(xt, ys)
        H = np.full(X.shape, T)
        for p in base["pockets"]:
            (cx, cy), (sx, sy) = p["centre"], p["size"]
            inside = ((np.abs(X - cx) < sx / 2 - EPS_MM / 2)
                      & (np.abs(Y - cy) < sy / 2 - EPS_MM / 2))
            H = np.where(inside, T - dp, H)
        v, f = closed_heightfield(xt, ys, H)
        out.append(("" if len(cuts) == 2 else f"_tile{i + 1}", to_bed(v), f))
    return out


def box_mesh(size) -> tuple:
    sx, sy, sz = (float(v) for v in size)
    v, f = closed_heightfield([-sx / 2, sx / 2], [-sy / 2, sy / 2], np.full((2, 2), sz))
    return v, f


def block_mesh(spec: dict) -> tuple:
    """The check block, bottom-down: z = slab + the nominal top surface."""
    xs, ys = cb.grid(spec)
    X, Y = np.meshgrid(xs, ys)
    H = float(spec["slab"]["T"]) + cb.height(spec, X, Y)
    v, f = closed_heightfield(xs, ys, H)
    return to_bed(v), f


# ------------------------------------------------------------------ files

def write_fixture(out: Path, spec: dict, dpi: float = 600.0, ascii_: bool = False) -> dict:
    name = spec["name"]
    files = {"spec": fx.save(spec, out / f"{name}_spec.json"), "stickers": [], "stl": [],
             "mesh": {}}
    margin = float(spec["design"]["sticker_margin_mm"])
    for f, b in zip(spec["faces"], fx.boards(spec)):
        a, z = f["id_range"]
        stem = out / f"{name}_face{f['face']}_ids{a}-{z}"
        # board + white margin only: text next to the board could be taken for a marker
        files["stickers"].append(charuco.write_png(stem.with_suffix(".png"), b, dpi, margin))
        files["stickers"].append(charuco.write_pdf(stem.with_suffix(".pdf"), b, dpi, margin))
    parts = []
    for p in spec["parts"]:
        if p["kind"] == "base":
            parts += [(f"base{sfx}", v, f) for sfx, v, f in base_meshes(p)]
        elif p["kind"] == "wedge":
            parts.append((p["name"], *wedge_mesh(p)))
        elif p["kind"] == "splice":
            parts.append((f"splice_x{p['count']}", *box_mesh(p["size"])))
    for pn, v, f in parts:
        path = write_stl(out / f"{name}_{pn}.stl", v, f, f"{name} {pn} mm", ascii_)
        files["stl"].append(path)
        files["mesh"][pn] = mesh_report(v, f)
    return files


def write_block(out: Path, spec: dict, ascii_: bool = False) -> dict:
    name = spec["name"]
    files = {"spec": out / f"{name}_spec.json", "template": out / f"{name}_measured_TEMPLATE.json"}
    for k in ("spec", "template"):
        with open(files[k], "x", encoding="utf-8") as fh:
            json.dump(spec if k == "spec" else cb.measured_template(spec), fh, indent=2)
            fh.write("\n")
    v, f = block_mesh(spec)
    files["stl"] = write_stl(out / f"{name}.stl", v, f, f"{name} mm", ascii_)
    files["mesh"] = mesh_report(v, f)
    return files


def readme(fix: Optional[dict], block: Optional[dict], fix_files: Optional[dict],
           block_files: Optional[dict]) -> str:
    L = ["CALIBRATION FIXTURE + CHECK BLOCK - print, assemble, measure",
         "MOCK-TESTED ONLY: not yet used on a real rig. Prove it on your rig before you rely "
         "on it.", ""]
    if fix:
        d = fix["design"]
        L += ["== The fixture (" + fix["name"] + ") ==",
              f"{d['n_faces']} faces, tilt {', '.join(f'{t:g}' for t in d['tilt_deg'])} deg, "
              f"ChArUco {d['squares'][0]}x{d['squares'][1]} squares of {d['square_mm']:g} mm, "
              f"{d['dictionary']}, marker ids from {d['first_id']} (each face its own range).",
              f"Base plate {d['base_size_mm'][0]:.1f} x {d['base_size_mm'][1]:.1f} mm"
              + (f" in {d['base_tiles']} tiles" if d["base_tiles"] > 1 else "")
              + f", wedges up to {d['max_height_mm']:.1f} mm above it.", "",
              "PRINT",
              "  * Matte WHITE PLA or PETG (no glossy / silk filament: the fringes must not "
              "glint).",
              "  * Layer 0.12-0.2 mm, 3 perimeters, 15-20 % infill (gyroid / grid), no supports.",
              "  * Wedges are exported face-DOWN (ridges / flat): the face is printed on the bed "
              "and is as flat as the bed. Use a flat glass / PEI bed, first layer not squished. "
              "Remove any elephant-foot lip at the face edge."
              if d["face_style"] != "recess" else
              "  * Wedges are exported bottom-down (recess style): the face is a printed slope, "
              "flat only to the layer steps - use the finest layer height you can.",
              "  * The base plate prints bottom-down; check its pockets take a wedge with a "
              "slight push fit (clearance 0.2 mm per side).", "",
              "STICKERS (one per face)",
              "  * Print each <name>_face<k>_ids<a>-<b>.pdf at 100 % / ACTUAL SIZE (never "
              "'fit to page'), on matte paper, laser preferred.",
              "  * Laminate MATTE (or no laminate: glossy film glints) and glue it flat with "
              "spray adhesive; no bubbles, no wrinkles.",
              "  * Stick face k's sticker on wedge_k, roughly centred. Its position and its "
              "turn in the face plane do not matter to the calibration.",
              "  * Ridges (default): the sticker rests on the ridge tops; glue lumps go in the "
              "grooves.", "",
              "ASSEMBLE",
              "  * Drop wedge_k into the base pocket the spec gives face k (its LOW edge toward "
              "the outside: every face looks outward). A drop of glue per wedge is enough.",
              "  * Tiles (if any): butt them, glue a splice plate under each seam.", "",
              "CHECK BEFORE USE",
              "  * Flatness: lay a steel straightedge across each face, length and diagonals; "
              "no gap a 0.1 mm feeler gauge enters.",
              "  * Caliper ONE printed square per face (over 5 squares, divide by 5), write it "
              "into faces[k].square_mm_measured of <name>_spec.json. Printer scale error goes "
              "straight into every mm otherwise.", "",
              "CALIBRATE (3 placements, about 1-2 minutes)",
              "  scan parity                                            (once per rig)",
              "  scan calib-pose --fixture <name>_spec.json --set fx    x3: set the fixture "
              "down facing the camera, capture, move it (left / right / centre, turned "
              "differently, at different distances), capture again",
              "  scan calibrate-stereo --set fx --fixture <name>_spec.json",
              "  The report gives camera / projector RMS and the face-angle check (each "
              "pair of faces vs the spec: a print / assembly sanity check, not a constraint).",
              ""]
    if block:
        L += ["== The check block (" + block["name"] + ") ==",
              "  * Print bottom-down, matte white, layer 0.12-0.2 mm, 4 perimeters, 30 % infill.",
              "  * MEASURE THE PRINT, fill " + block["name"] + "_measured_TEMPLATE.json "
              "(save it under another name): each step height above the reference face (caliper "
              "depth rod or height gauge, 3 places, average), the bar's diameter across its flats "
              "and its top height, the dome's apex height.",
              "  * Place it reference face to the camera, long side horizontal, the TALLEST step "
              "on the side AWAY from the projector.",
              "  scan accuracy-block --block " + block["name"] + "_spec.json --measured "
              "<yours>.json",
              "  exit 0 CONTINUE (within tolerance), 1 REJECT (out of tolerance: recalibrate), "
              "5 PULL (doubt: unmeasured block, a feature not seen).", ""]
    L += ["LIMITS: a printed face is only as flat as the print and the sticker on it; a printed "
          "block is only as true as you measured it. Re-measure after a knock or a hot day."]
    if fix_files:
        L += ["", "Meshes (closed: every edge in exactly 2 triangles):"]
        L += [f"  {k}: {m['n_triangles']} triangles, {m['size_mm']} mm, closed={m['closed']}"
              for k, m in fix_files["mesh"].items()]
    if block_files:
        m = block_files["mesh"]
        L += [f"  {block['name']}: {m['n_triangles']} triangles, {m['size_mm']} mm, "
              f"closed={m['closed']}"]
    text = "\n".join(L) + "\n"
    return text.replace("<name>", fix["name"]) if fix else text


def make(out_root: Path, fixture_kw: Optional[dict] = None, block_kw: Optional[dict] = None,
         dpi: float = 600.0, ascii_: bool = False, label: str = "fixture") -> dict:
    """Write the package into a fresh <out_root>/<timestamp>_<label>/ -> what was written."""
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for n in range(1000):
        out = out_root / (f"{stamp}_{label}" if n == 0 else f"{stamp}_{label}-{n}")
        try:
            out.mkdir()
            break
        except FileExistsError:
            continue
    else:
        raise FileExistsError(f"could not allocate a new folder under {out_root}")
    res = {"dir": out}
    fix = fx.design(**fixture_kw) if fixture_kw is not None else None
    block = cb.design(**block_kw) if block_kw is not None else None
    if fix:
        res["fixture"] = write_fixture(out, fix, dpi, ascii_)
    if block:
        res["block"] = write_block(out, block, ascii_)
    p = out / "README.txt"
    with open(p, "x", encoding="utf-8", newline="\n") as fh:
        fh.write(readme(fix, block, res.get("fixture"), res.get("block")))
    res["readme"] = p
    res["fixture_spec"], res["block_spec"] = fix, block
    return res


def _pair(text: str, what: str, cast=float) -> tuple:
    try:
        a, b = (cast(v) for v in str(text).replace("x", ",").split(","))
    except ValueError:
        raise argparse.ArgumentTypeError(f"{what}: two numbers A,B, got {text!r}") from None
    return a, b


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="output root; a fresh <ts>_<name> folder "
                                                 "is made inside")
    ap.add_argument("--name", default="fixture")
    ap.add_argument("--faces", type=int, default=4, help="number of faces (2..8)")
    ap.add_argument("--square", type=float, default=16.0,
                    help="ChArUco square, mm (>= ~15 mm at 650-800 mm)")
    ap.add_argument("--squares", default="6,5", help="squares per face: along the slope, across")
    ap.add_argument("--marker-ratio", type=float, default=0.7, help="marker / square")
    ap.add_argument("--dictionary", default="DICT_5X5_250")
    ap.add_argument("--first-id", type=int, default=100,
                    help="first marker id (face k: first + k * markers per face)")
    ap.add_argument("--tilt", default="30", help="face tilt, deg: one value or one per face")
    ap.add_argument("--face-style", choices=("ridges", "recess", "flat"), default="ridges")
    ap.add_argument("--sticker-mm", type=float, default=0.3, help="sticker thickness")
    ap.add_argument("--bed", default="220,220", help="print bed, mm")
    ap.add_argument("--dpi", type=float, default=600.0, help="sticker image resolution")
    ap.add_argument("--block-name", default="checkblock")
    ap.add_argument("--steps", default="25,20,15,10,5", help="check block step heights, mm")
    ap.add_argument("--no-block", action="store_true", help="the fixture only")
    ap.add_argument("--block-only", action="store_true", help="the check block only")
    ap.add_argument("--ascii", action="store_true", help="ASCII STL (default binary)")
    a = ap.parse_args(argv)
    try:
        tilts = [float(t) for t in a.tilt.split(",")]
        fixture_kw = None if a.block_only else {
            "n_faces": a.faces, "square_mm": a.square,
            "squares": _pair(a.squares, "--squares", int), "marker_ratio": a.marker_ratio,
            "dictionary": a.dictionary, "first_id": a.first_id,
            "tilt_deg": tilts[0] if len(tilts) == 1 else tilts, "face_style": a.face_style,
            "sticker_mm": a.sticker_mm, "bed_mm": _pair(a.bed, "--bed"), "name": a.name}
        block_kw = None if a.no_block else {
            "step_heights": [float(v) for v in a.steps.split(",")],
            "bed_mm": _pair(a.bed, "--bed"), "name": a.block_name}
        res = make(Path(a.out), fixture_kw, block_kw, a.dpi, a.ascii, a.name)
    except (ValueError, argparse.ArgumentTypeError) as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    print(res["dir"])
    meshes = dict((res.get("fixture") or {}).get("mesh", {}))
    if res.get("block"):
        meshes[a.block_name] = res["block"]["mesh"]
    for k, m in meshes.items():
        print(f"  {k:<14} {m['n_triangles']:>7} triangles  {m['size_mm']} mm  "
              f"closed={m['closed']}")
    print(f"  README: {res['readme']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

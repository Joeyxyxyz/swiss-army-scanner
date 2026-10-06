"""
Meshes for 3D viewers (Blender, slicers): heightmap STL/OBJ and the open
cylinder-arc mesh of an unwrapped cell. Ported from the ideas in
swiss_army_sensor.mesh (relief_to_stl, relief_to_cylinder_mesh); nothing
is imported from that package.

Rules kept here (session report 2026-09-23, section 8.5):
  * binary STL (ASCII is too slow at 1080x1440), optional OBJ
  * hole close + largest connected component before faces (no islands)
  * NO non-finite vertex is ever written (asserted, and reported in the meta)
  * heightmap = a frontal "curtain", NOT a closed solid; the cylinder mesh
    is the OPEN arc the camera saw, never an invented 360 degree tube
  * a meta JSON beside every mesh: geometry id, units, calibration id or
    'uncalibrated', zone/atlas when the scan has them, finite check,
    vertex/triangle counts
  * mm only with a calibration for the scan's geometry (rule 6); otherwise
    camera px (x, y) and relief px (z), and the meta says the axes are
    NOT to scale

These functions only write the files they are given (new paths, never an
overwrite); the api verb allocates the paths and records the outputs.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import ndimage

from .. import rules
from ..decode.unwrap import _fill_invalid

FORMATS = ("stl", "obj")


# ------------------------------------------------------------------ writers

def _normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    a, b, c = (verts[faces[:, i]] for i in range(3))
    n = np.cross(b - a, c - a)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    return np.where(ln > 0, n / np.where(ln > 0, ln, 1), 0.0)


def write_stl(path: Path, verts: np.ndarray, faces: np.ndarray, header: str = "") -> Path:
    """Binary STL (80-byte header, uint32 count, 50 bytes per triangle)."""
    rules.ensure_new_path(path)
    v = np.asarray(verts, np.float64)
    f = np.asarray(faces, np.int64)
    rec = np.zeros(len(f), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    rec["n"] = _normals(v, f)
    rec["v"] = v[f]
    with open(path, "xb") as fh:
        fh.write(header.encode("ascii", "replace")[:80].ljust(80, b" "))
        fh.write(struct.pack("<I", len(f)))
        fh.write(rec.tobytes())
    return path


def write_obj(path: Path, verts: np.ndarray, faces: np.ndarray, header: str = "") -> Path:
    rules.ensure_new_path(path)
    with open(path, "x", encoding="ascii", newline="\n") as fh:
        for line in header.splitlines():
            fh.write(f"# {line}\n")
        np.savetxt(fh, verts, fmt="v %.5f %.5f %.5f")
        np.savetxt(fh, np.asarray(faces, np.int64) + 1, fmt="f %d %d %d")
    return path


def read_stl_header(path: Path) -> dict:
    """{'header', 'n_triangles', 'size_ok', 'finite'} of a binary STL (for tests / QC)."""
    data = Path(path).read_bytes()
    n = struct.unpack("<I", data[80:84])[0]
    ok = len(data) == 84 + 50 * n
    finite = None
    if ok and n:
        rec = np.frombuffer(data[84:], dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)),
                                              ("attr", "<u2")])
        finite = bool(np.isfinite(rec["v"]).all())
    return {"header": data[:80].decode("ascii", "replace").strip(), "n_triangles": n,
            "size_ok": ok, "finite": finite}


def _zu(meta) -> str:
    u = meta.get("units") or {}
    return str(u.get("z", u.get("xyz", "?")))


def _write(path: Path, fmt: str, verts, faces, meta: dict) -> tuple:
    """Assert finiteness, write the mesh and its meta JSON beside it.
    -> (mesh path, meta path, meta)."""
    if fmt not in FORMATS:
        raise ValueError(f"fmt must be one of {FORMATS}")
    verts = np.asarray(verts, np.float64)
    faces = np.asarray(faces, np.int64).reshape(-1, 3)
    if len(faces) == 0:
        raise ValueError("mesh has no triangles (no valid surface to mesh)")
    finite = bool(np.isfinite(verts).all())
    assert finite, "non-finite vertex in mesh - refusing to write (report 8.5)"
    path = Path(path)
    meta_path = path.with_name(path.name + ".json")   # cyl.stl.json: stl and obj never collide
    rules.ensure_all_new([path, meta_path])
    header = (f"swiss_army_scanner {meta.get('kind', '')} geometry={meta.get('geometry_id', '')} "
              f"units={_zu(meta)}")
    (write_stl if fmt == "stl" else write_obj)(path, verts, faces, header)
    meta = {**meta, "format": fmt, "file": path.name, "n_vertices": int(len(verts)),
            "n_triangles": int(len(faces)), "finite_vertices": finite,
            "bbox_min": verts.min(0).round(4).tolist(), "bbox_max": verts.max(0).round(4).tolist()}
    with open(meta_path, "x", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)
    return path, meta_path, meta


# ------------------------------------------------------------ grid -> mesh

def clean_mask(mask: np.ndarray, close_iter: int = 2) -> np.ndarray:
    """Hole close + largest connected component."""
    m = ndimage.binary_closing(mask, structure=np.ones((3, 3)), iterations=close_iter) | mask
    m = ndimage.binary_fill_holes(m)
    lab, n = ndimage.label(m)
    if n > 1:
        sizes = ndimage.sum(m, lab, range(1, n + 1))
        m = lab == (int(np.argmax(sizes)) + 1)
    return m


def grid_mesh(X, Y, Z, keep) -> tuple:
    """Regular grid (all 2-D, same shape) -> (verts, faces) over quads whose 4
    corners are kept and finite. Only used vertices are returned."""
    keep = keep & np.isfinite(X) & np.isfinite(Y) & np.isfinite(Z)
    h, w = keep.shape
    idx = np.arange(h * w).reshape(h, w)
    q = keep[:-1, :-1] & keep[:-1, 1:] & keep[1:, :-1] & keep[1:, 1:]
    a, b = idx[:-1, :-1][q], idx[:-1, 1:][q]
    c, d = idx[1:, :-1][q], idx[1:, 1:][q]
    faces = np.concatenate([np.column_stack([a, c, b]), np.column_stack([b, c, d])])
    used = np.unique(faces)
    remap = np.full(h * w, -1, np.int64)
    remap[used] = np.arange(len(used))
    verts = np.column_stack([X.ravel()[used], Y.ravel()[used], Z.ravel()[used]])
    return verts, remap[faces]


def _units_and_scale(calibration, geometry_id):
    if calibration is not None and not rules.calibration_matches(calibration, geometry_id):
        calibration = None
    z = calibration if (calibration is not None and calibration.has_z) else None
    xy = calibration.lateral_px_per_mm if (calibration is not None and calibration.has_xy) else None
    return calibration, z, xy


def _scan_meta(scan, calibration, kind: str) -> dict:
    m = getattr(scan, "meta", None) or {}
    return {"kind": kind, "scan": getattr(scan, "id", None), "label": getattr(scan, "label", None),
            "geometry_id": scan.geometry_id,
            "calibration_id": calibration.id if calibration is not None else "uncalibrated",
            "zone": m.get("zone"), "atlas": m.get("atlas"),
            "wavefront_lock": m.get("wavefront_lock"), "mock": m.get("mock")}


def _roi(shape, roi):
    if roi is None:
        return slice(0, shape[0]), slice(0, shape[1])
    x0, y0, x1, y1 = (int(v) for v in roi)
    return slice(max(y0, 0), min(y1, shape[0])), slice(max(x0, 0), min(x1, shape[1]))


def _heightfield_mesh(z_img, valid, stride, roi, z_scale, z_off, xy_ppmm):
    ys, xs = _roi(valid.shape, roi)
    h, w = valid.shape
    z = np.asarray(z_img, np.float64)[ys, xs][::stride, ::stride]
    m = (np.asarray(valid, bool) & np.isfinite(np.asarray(z_img, np.float64)))[ys, xs][::stride, ::stride]
    if not m.any():
        raise ValueError("no valid pixels in the requested region")
    keep = clean_mask(m)
    zf = _fill_invalid(np.where(m, z, 0.0), m)          # closed holes get nearest real value
    v, u = np.mgrid[ys, xs][:, ::stride, ::stride].astype(np.float64)
    X, Y = u - w / 2.0, -(v - h / 2.0)                  # image up = +Y
    if xy_ppmm:
        X, Y = X / xy_ppmm, Y / xy_ppmm
    Z = z_scale * zf + z_off
    return grid_mesh(X, Y, Z, keep), int(m.sum()), int(keep.sum())


def heightmap_to_stl(scan, calibration, path: Path, stride: int = 2, roi=None,
                     fmt: str = "stl") -> tuple:
    """Scan relief -> heightmap mesh (a frontal curtain, not a solid).
    x, y: camera px (or mm with a lateral calibration), z: relief px (or mm with
    a depth calibration). roi = (x0, y0, x1, y1) camera px.
    -> (mesh path, meta path, meta)."""
    stride = max(1, int(stride))
    cal, zc, xy = _units_and_scale(calibration, scan.geometry_id)
    (verts, faces), n_valid, n_kept = _heightfield_mesh(
        scan.relief, scan.relief_mask, stride, roi,
        zc.mm_per_px if zc else 1.0, zc.intercept_mm if zc else 0.0, xy)
    units = {"xy": "mm" if xy else "camera_px", "z": "mm" if zc else "relief_px"}
    meta = {**_scan_meta(scan, cal, "relief_heightmap"), "units": units,
            "to_scale": bool(xy and zc), "stride": stride, "roi": roi,
            "valid_samples": n_valid, "meshed_samples": n_kept,
            "note": ("frontal heightmap = open curtain, not a closed solid"
                     + ("" if (xy and zc) else "; axes NOT to scale (no matching "
                        "calibration for this geometry, rule 6)"))}
    return _write(path, fmt, verts, faces, meta)


def base_to_stl(base, path: Path, stride: int = 2, roi=None, fmt: str = "stl",
                axis: Optional[str] = None) -> tuple:
    """A base (BaseRef or DeformationField) correspondence surface -> mesh.
    z = the decoded projector coordinate along `axis` (default the first coded
    axis) minus its best-fit plane, in projector px; x, y camera px. It is a
    QC view of the reference surface, never a metric shape."""
    fld = getattr(base, "field", base)
    axis = axis or fld.axes[0]
    c = np.asarray(getattr(fld, axis), np.float64)
    m = np.asarray(fld.mask, bool) & np.isfinite(c)
    if m.sum() < 3:
        raise ValueError("base has no valid correspondence")
    v, u = np.nonzero(m)
    A = np.column_stack([u, v, np.ones_like(u)]).astype(np.float64)
    coef, *_ = np.linalg.lstsq(A, c[m], rcond=None)
    vv, uu = np.mgrid[0:c.shape[0], 0:c.shape[1]]
    detr = c - (coef[0] * uu + coef[1] * vv + coef[2])
    (verts, faces), n_valid, n_kept = _heightfield_mesh(detr, m, max(1, int(stride)), roi,
                                                        1.0, 0.0, None)
    gid = getattr(base, "geometry_id", None) or fld.geometry_id
    bm = getattr(base, "meta", None) or {}
    meta = {"kind": "base_correspondence", "base": getattr(base, "id", None),
            "label": getattr(base, "label", None), "geometry_id": gid,
            "calibration_id": "uncalibrated", "zone": bm.get("zone"), "atlas": bm.get("atlas"),
            "wavefront_lock": bm.get("wavefront_lock"), "mock": (fld.meta or {}).get("mock"),
            "units": {"xy": "camera_px", "z": f"projector_px ({axis}, plane-detrended)"},
            "to_scale": False, "stride": int(stride), "roi": roi,
            "plane_fit": [float(x) for x in coef],
            "valid_samples": n_valid, "meshed_samples": n_kept,
            "note": "reference correspondence surface for QC; not a metric shape"}
    return _write(path, fmt, verts, faces, meta)


def cylinder_arc_mesh(unwrap, path: Path, fmt: str = "stl", fill_gaps: int = 2) -> tuple:
    """CellUnwrap -> open cylinder-arc mesh: x = (R+d) sin(theta), y = z along
    the axis, z = (R+d) cos(theta). Only the arc that was SEEN (unwrap.valid,
    small gaps <= fill_gaps bins closed within the seen band) is meshed; no
    unseen angle is invented."""
    valid = np.asarray(unwrap.valid, bool)
    if not valid.any():
        raise ValueError("unwrap has no valid bins")
    keep = valid
    if fill_gaps:
        closed = ndimage.binary_closing(valid, structure=np.ones((3, 3)), iterations=fill_gaps)
        # never extend past the seen band: only bins between seen bins on their row/col
        rows = np.maximum.accumulate(valid, 1) & np.maximum.accumulate(valid[:, ::-1], 1)[:, ::-1]
        cols = np.maximum.accumulate(valid, 0) & np.maximum.accumulate(valid[::-1], 0)[::-1]
        keep = valid | (closed & rows & cols)
    lab, n = ndimage.label(keep)
    if n > 1:
        sizes = ndimage.sum(keep, lab, range(1, n + 1))
        keep = lab == (int(np.argmax(sizes)) + 1)
    dev = _fill_invalid(np.where(valid, unwrap.deviation, 0.0), valid)
    TH, ZZ = np.meshgrid(np.radians(unwrap.theta_deg), unwrap.z)
    RAD = float(unwrap.radius) + dev
    X, Y, Z = RAD * np.sin(TH), ZZ, RAD * np.cos(TH)
    verts, faces = grid_mesh(X, Y, Z, keep)
    meta = {"kind": "cylinder_arc", "scan": unwrap.scan_id, "label": unwrap.label,
            "geometry_id": unwrap.geometry_id, "calibration_id": unwrap.calibration_id,
            "zone": unwrap.zone, "atlas": unwrap.atlas,
            "units": {"xyz": unwrap.units}, "to_scale": unwrap.units == "mm",
            "radius": unwrap.radius, "arc_deg": unwrap.arc_deg,
            "cell_height": unwrap.cell_height, "open_arc_only": True,
            "note": "measured open arc only; no unseen angle is invented (no closed tube)"}
    return _write(path, fmt, verts, faces, meta)


# ------------------------------------------------------ metric cloud mesh

def cloud_mesh(cloud, path: Path, fmt: str = "stl", stride: int = 1,
               max_edge_mm: Optional[float] = None, region: Optional[np.ndarray] = None,
               meta: Optional[dict] = None) -> tuple:
    """A triangulated MetricCloud (mm, camera frame) -> STL / OBJ.

    Open surface only: quads of the camera grid whose four corners are
    measured, minus any triangle with an edge longer than max_edge_mm
    (default 4 x the median grid spacing) so depth jumps and shadow edges
    are left open, never bridged. No hole filling, no closure.
    -> (mesh path, meta path, meta)."""
    if stride < 1:
        raise ValueError("stride must be >= 1")
    m = cloud.mask if region is None else (cloud.mask & region)
    xyz = cloud.xyz[::stride, ::stride]
    keep = m[::stride, ::stride]
    if not keep.any():
        raise ValueError("no measured points to mesh")
    verts, faces = grid_mesh(xyz[..., 0], xyz[..., 1], xyz[..., 2], keep)
    if len(faces) == 0:
        raise ValueError("no complete grid cells to mesh")
    e = np.stack([np.linalg.norm(verts[faces[:, i]] - verts[faces[:, (i + 1) % 3]], axis=1)
                  for i in range(3)], 1)
    if max_edge_mm is None:
        max_edge_mm = 4.0 * float(np.median(e))
    faces = faces[(e <= max_edge_mm).all(1)]
    used = np.unique(faces)
    remap = np.full(len(verts), -1, np.int64)
    remap[used] = np.arange(len(used))
    verts, faces = verts[used], remap[faces]
    meta = {"kind": "metric", "units": {"xyz": "mm"}, "to_scale": True,
            "frame": "camera: x right, y down, z forward (mm)",
            "geometry_id": cloud.geometry_id, "calibration_id": cloud.calibration_id,
            "stride": stride, "max_edge_mm": round(float(max_edge_mm), 4),
            "note": "open surface as measured; nothing closed or invented", **(meta or {})}
    return _write(path, fmt, verts, faces, meta)

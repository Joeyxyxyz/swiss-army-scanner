"""Relief / point-cloud -> triangle mesh -> STL/OBJ for Blender/CAD.

Structured-light pipeline stops at relief or correspondence today.
This module turns a Z height map (or Nx3 cloud) into a watertight-ish
mesh you can open in Blender, Fusion, etc.

Dependencies: numpy always; for STL write prefer `numpy-stl` or a tiny
ASCII STL writer (bundled below) so we do not require Open3D at import.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np


def relief_to_cloud(relief: np.ndarray, mask: Optional[np.ndarray] = None,
                    pixel_size_mm: float = 1.0,
                    z_scale: float = 1.0) -> np.ndarray:
    """H x W relief -> Nx3 XYZ (mm if pixel_size_mm and z_scale set)."""
    z = np.asarray(relief, dtype=np.float64)
    if mask is None:
        mask = np.isfinite(z)
    else:
        mask = np.asarray(mask, dtype=bool) & np.isfinite(z)
    ys, xs = np.where(mask)
    if ys.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    cloud = np.column_stack([
        xs.astype(np.float64) * pixel_size_mm,
        ys.astype(np.float64) * pixel_size_mm,
        z[ys, xs] * z_scale,
    ])
    return cloud


def heightmap_to_mesh(relief: np.ndarray, mask: Optional[np.ndarray] = None,
                      pixel_size_mm: float = 1.0,
                      z_scale: float = 1.0):
    """Grid relief -> (vertices Nx3, faces Mx3) via adjacent quads.

    Skips quads where any corner is masked/invalid. Good first STL for
    Blender; for denser CAD use Poisson on the cloud later.
    """
    z = np.asarray(relief, dtype=np.float64)
    H, W = z.shape
    if mask is None:
        valid = np.isfinite(z)
    else:
        valid = np.asarray(mask, dtype=bool) & np.isfinite(z)

    # vertex index per pixel (-1 invalid)
    vid = -np.ones((H, W), dtype=np.int32)
    ys, xs = np.where(valid)
    verts = np.column_stack([
        xs.astype(np.float64) * pixel_size_mm,
        ys.astype(np.float64) * pixel_size_mm,
        z[ys, xs] * z_scale,
    ])
    vid[ys, xs] = np.arange(len(verts), dtype=np.int32)

    faces = []
    for y in range(H - 1):
        for x in range(W - 1):
            a, b = vid[y, x], vid[y, x + 1]
            c, d = vid[y + 1, x], vid[y + 1, x + 1]
            if a < 0 or b < 0 or c < 0 or d < 0:
                continue
            faces.append((a, b, d))
            faces.append((a, d, c))
    faces_arr = np.asarray(faces, dtype=np.int32) if faces else np.zeros((0, 3), dtype=np.int32)
    return verts, faces_arr


def write_stl_ascii(path: Union[str, Path], vertices: np.ndarray,
                    faces: np.ndarray, name: str = "swiss_army_scan") -> Path:
    """Write ASCII STL (Blender / most CAD open this)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int32)
    lines = [f"solid {name}"]
    for i0, i1, i2 in f:
        p0, p1, p2 = v[i0], v[i1], v[i2]
        n = np.cross(p1 - p0, p2 - p0)
        nn = np.linalg.norm(n)
        if nn < 1e-18:
            n = np.array([0.0, 0.0, 1.0])
        else:
            n = n / nn
        lines.append(
            f"  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}\n"
            f"    outer loop\n"
            f"      vertex {p0[0]:.6e} {p0[1]:.6e} {p0[2]:.6e}\n"
            f"      vertex {p1[0]:.6e} {p1[1]:.6e} {p1[2]:.6e}\n"
            f"      vertex {p2[0]:.6e} {p2[1]:.6e} {p2[2]:.6e}\n"
            f"    endloop\n"
            f"  endfacet"
        )
    lines.append(f"endsolid {name}\n")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def relief_to_stl(relief: np.ndarray, path: Union[str, Path],
                  mask: Optional[np.ndarray] = None,
                  pixel_size_mm: float = 1.0,
                  z_scale: float = 1.0,
                  name: str = "swiss_army_scan") -> Path:
    verts, faces = heightmap_to_mesh(relief, mask=mask,
                                     pixel_size_mm=pixel_size_mm,
                                     z_scale=z_scale)
    return write_stl_ascii(path, verts, faces, name=name)


def relief_to_cylinder_mesh(
    relief,
    mask,
    radius_mm: float = 24.05,
    height_mm: float = 48.0,
    theta_span_deg: float = 100.0,
    n_theta: int = 220,
    n_h: int = 200,
    bump_mm: float = 2.0,
    smooth_sigma: float = 0.7,
):
    """Map frontal relief onto an open cylinder arc. No fake 360 tube."""
    from scipy import ndimage
    z = np.asarray(relief, dtype=np.float64)
    m = np.asarray(mask, dtype=bool) & np.isfinite(z)
    if not m.any():
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int32)
    m = ndimage.binary_fill_holes(ndimage.binary_closing(m, iterations=2))
    lab, nlab = ndimage.label(m)
    if nlab:
        sizes = ndimage.sum(m, lab, index=np.arange(1, nlab + 1))
        m = lab == (int(np.argmax(sizes)) + 1)
    m = m & np.isfinite(z)
    ys, xs = np.where(m)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    zz = z[y0:y1, x0:x1]
    mm = m[y0:y1, x0:x1]
    zz = zz - float(np.median(zz[mm]))
    p5, p95 = np.percentile(zz[mm], [5, 95])
    span = float(max(abs(p5), abs(p95), 1e-3))
    bump = np.zeros_like(zz)
    bump[mm] = np.clip(zz[mm] / span * bump_mm, -bump_mm, bump_mm)
    Hh, Ww = bump.shape
    grid_sum = np.zeros((n_h, n_theta), np.float64)
    wgt = np.zeros((n_h, n_theta), np.float64)
    yy, xx = np.where(mm)
    ti = np.clip(((xx + 0.5) / Ww * (n_theta - 1)).astype(np.int32), 0, n_theta - 1)
    hi = np.clip(((yy + 0.5) / Hh * (n_h - 1)).astype(np.int32), 0, n_h - 1)
    np.add.at(wgt, (hi, ti), 1.0)
    np.add.at(grid_sum, (hi, ti), bump[yy, xx])
    has = wgt > 0
    grid = np.zeros_like(grid_sum)
    grid[has] = grid_sum[has] / wgt[has]
    inds = ndimage.distance_transform_edt(~has, return_distances=False, return_indices=True)
    grid = ndimage.gaussian_filter(grid[tuple(inds)], sigma=smooth_sigma)
    grid = np.nan_to_num(grid, nan=0.0)
    ths = np.linspace(-0.5 * np.deg2rad(theta_span_deg), 0.5 * np.deg2rad(theta_span_deg), n_theta)
    hs = np.linspace(0.0, float(height_mm), n_h)
    TH, HH = np.meshgrid(ths, hs)
    RAD = float(radius_mm) + grid
    verts = np.stack([RAD * np.sin(TH), HH, RAD * np.cos(TH)], axis=-1).reshape(-1, 3)
    vid = np.arange(n_h * n_theta, dtype=np.int32).reshape(n_h, n_theta)
    faces = []
    for ih in range(n_h - 1):
        for it in range(n_theta - 1):
            a = int(vid[ih, it]); b = int(vid[ih, it + 1])
            c = int(vid[ih + 1, it]); d = int(vid[ih + 1, it + 1])
            faces.append((a, b, d)); faces.append((a, d, c))
    return verts, np.asarray(faces, dtype=np.int32)

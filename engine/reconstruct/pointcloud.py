"""Relief (+ calibration) -> .ply / .xyz point cloud. Units in the header."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import rules


def points(scan, calibration=None):
    """-> (N x 3 float array, xy_units, z_units)."""
    relief, mask = scan.relief, scan.relief_mask
    v, u = np.nonzero(mask)
    z = relief[v, u].astype(np.float64)
    h, w = mask.shape
    x, y = u - w / 2.0, v - h / 2.0
    xy_units = z_units = "px"
    if calibration is not None and calibration.has_z:
        z = calibration.mm_per_px * z + calibration.intercept_mm
        z_units = "mm"
    if calibration is not None and calibration.has_xy:
        x, y = x / calibration.lateral_px_per_mm, y / calibration.lateral_px_per_mm
        xy_units = "mm"
    return np.column_stack([x, y, z]), xy_units, z_units


def export(scan, path: Path, fmt: str = "ply", calibration=None) -> Path:
    if fmt not in ("ply", "xyz"):
        raise ValueError("fmt must be 'ply' or 'xyz'")
    rules.ensure_new_path(path)
    pts, xy_u, z_u = points(scan, calibration)
    with open(path, "x", encoding="ascii", newline="\n") as f:
        if fmt == "ply":
            f.write("ply\nformat ascii 1.0\n"
                    f"comment swiss_army_scanner scan={scan.id} geometry={scan.geometry_id}\n"
                    f"comment units xy={xy_u} z={z_u}\n"
                    f"element vertex {len(pts)}\n"
                    "property float x\nproperty float y\nproperty float z\nend_header\n")
        else:
            f.write(f"# swiss_army_scanner scan={scan.id} units xy={xy_u} z={z_u}\n")
        np.savetxt(f, pts, fmt="%.4f")
    return path

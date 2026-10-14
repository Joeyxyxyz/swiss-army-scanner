"""
CheckContext: everything a check may read about ONE zone of ONE scan (never the rig).

    ctx.points()      (N, 3) measured points of the zone, mm, camera frame
    ctx.uvh()         (N, 3) the same in the part frame: u, v (mm in the reference plane),
                      h (mm above it, + toward the camera)
    ctx.heights()     (N,) h only
    ctx.index()       (ys, xs) camera px of those points (same order)
    ctx.fault_px(i)   {"px": [x, y]} of point i, for CheckResult.fault_points
    ctx.zone          the ZoneView (mask, outline, id); ctx.zones every zone by id
    ctx.scan          the ScanView (cloud, decoded DMD correspondences, flood pair, intrinsics)
    ctx.frame         the PartFrame (reference plane, axes)
    ctx.golden        GoldenView or None: the good part's scan, frame and the same zones
    ctx.codes         codes read before the scan (scan.codes: true), or None
    ctx.keep_sections(grid_mm, height_mm)   hand over a gridded height map's two centre lines:
                      the runner writes them beside the result as profile_x.csv / profile_y.csv
                      (engine/eventlog.py); they are never part of a result
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class GoldenView:
    scan: object                         # ScanView
    frame: object                        # PartFrame
    zones: dict                          # zone id -> ZoneView
    scan_id: str = ""


@dataclass
class CheckContext:
    scan: object                         # ScanView
    frame: object                        # PartFrame
    zone: object                         # ZoneView
    zones: dict = field(default_factory=dict)
    golden: Optional[GoldenView] = None
    golden_error: Optional[str] = None   # why there is no golden view (for INVALID messages)
    codes: Optional[list] = None
    recipe_name: str = ""
    min_points: int = 30
    sections: Optional[dict] = None      # out: zone id -> height sections (keep_sections)

    # ---------------------------------------------------------- the zone's points
    def index(self, zone=None) -> tuple:
        z = zone or self.zone
        return np.nonzero(z.valid)

    def points(self, zone=None) -> np.ndarray:
        ys, xs = self.index(zone)
        return self.scan.xyz[ys, xs]

    def uvh(self, zone=None) -> np.ndarray:
        ys, xs = self.index(zone)
        return self.frame.uvh[ys, xs]

    def heights(self, zone=None) -> np.ndarray:
        return self.uvh(zone)[:, 2]

    def fault_px(self, i: int, label: str = "", zone=None) -> dict:
        ys, xs = self.index(zone)
        return {"px": [float(xs[i]), float(ys[i])], **({"label": label} if label else {})}

    def px_of(self, xyz) -> list:
        p = self.scan.project(np.asarray(xyz, np.float64).reshape(1, 3))[0]
        return [float(p[0]), float(p[1])]

    def mm_per_px(self, zone=None) -> Optional[float]:
        """Median 3-D distance between horizontally adjacent measured px of the zone."""
        z = zone or self.zone
        v = z.valid
        both = v[:, :-1] & v[:, 1:]
        ys, xs = np.nonzero(both)
        if len(xs) < 10:
            return None
        X = self.scan.xyz
        d = np.linalg.norm(X[ys, xs + 1] - X[ys, xs], axis=1)
        d = d[np.isfinite(d) & (d < 5.0)]
        return float(np.median(d)) if d.size else None

    def keep_sections(self, grid_mm, height_mm) -> None:
        """A check that gridded this zone's heights (height_mm (G, G) mm on the square grid
        grid_mm (G,) centred on the part; rows = camera down, columns = camera right) hands
        over the two lines through the centre. Files only (the runner writes them into the
        run folder): nothing of it reaches the check's result. The first check of a zone
        that calls it counts; without a runner (sections None) it does nothing. It never
        raises: a file is never a reason for a check to fail."""
        try:
            if self.sections is None or self.zone.id in self.sections:
                return
            g, h = np.array(grid_mm, np.float64), np.array(height_mm, np.float64)
            ic = len(g) // 2
            self.sections[self.zone.id] = {"pos_mm": g, "x_mm": h[ic, :].copy(),
                                           "y_mm": h[:, ic].copy()}
        except Exception:                   # noqa: BLE001
            pass

    def other(self, zone_id: str):
        try:
            return self.zones[zone_id]
        except KeyError:
            raise KeyError(f"no zone {zone_id!r} in this recipe") from None

    def need_golden(self):
        """-> GoldenView, or raises LookupError with the reason (the check answers INVALID)."""
        if self.golden is None:
            raise LookupError(self.golden_error or "no golden scan (recipe golden.scan)")
        return self.golden

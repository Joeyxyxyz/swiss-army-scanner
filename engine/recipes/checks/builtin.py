"""
Built-in recipe checks: thin wrappers over the engine's measurements.

  presence   part present / absent: enough of the zone stands above the reference plane
  position   alignment vs the golden part: in-plane offset (mm), rotation (deg), height;
             on a circle zone also the part's outline size vs the golden's / a nominal
  height     height (or step) of the zone vs the reference plane or another zone
  flatness   plane fit of the zone's points: peak-to-valley and RMS (mm)
  defect     local dents / bumps: MF-CFAR end-face detector on a circular face zone
             (engine/measure/endface/mfcfar.py, self-referenced; its radii derive from the
             zone's radius), a local-residual blob finder on any other zone. On a face
             zone also the RIM OUTLINE in the flood (edge dents: a notch / bulge of the
             outline, engine/measure/endface/outline.py) and the KNOWN SIZE (the outline's
             diameter vs the zone's own / a nominal: more than size_tol_pct off = INVALID
             with pull size_mismatch - calibration or lens suspect, never a pass)
  edge       straightness of the part's edge (sub-pixel edge in the flood image, robust
             line fit in the part frame: mm) and its position / angle vs the golden part
  code       a barcode / QR code read in the zone (engine/sense/codes.py decoders)

Every value carries its unit in its name (_mm, _deg, _mm2, _pct); every check answers
INVALID (never PASS) when it cannot measure. Every value a check holds to a limit is listed
in its `tolerances` (the recipe's guard band, engine/disposition.py, reads them).
"""

from __future__ import annotations

import math
import re
import warnings
from typing import Optional

import numpy as np

from ...disposition import tolerance
from . import register
from .base import Check, CheckResult, Param, ParamError

STATS = ("median", "mean", "p95", "p5", "max", "min")


def _stat(h: np.ndarray, stat: str) -> float:
    return float({"median": np.median, "mean": np.mean, "max": np.max, "min": np.min,
                  "p95": lambda a: np.percentile(a, 95), "p5": lambda a: np.percentile(a, 5)}
                 [stat](h))


def _part(ctx, min_height_mm, zone=None, golden=False):
    """Indices of the zone's points at least min_height_mm above the reference plane (all
    points when None) -> (uvh (N, 3), keep mask over the zone's points)."""
    if golden:
        g = ctx.golden
        z = g.zones[(zone or ctx.zone).id]
        ys, xs = np.nonzero(z.valid)
        uvh = g.frame.uvh[ys, xs]
    else:
        uvh = ctx.uvh(zone)
    keep = np.ones(len(uvh), bool) if min_height_mm is None else uvh[:, 2] >= min_height_mm
    return uvh, keep


# ================================================================== presence

@register
class Presence(Check):
    """Part present (or absent): a point is 'part' when it stands at least min_height_mm
    above the reference plane; PRESENT when part points cover min_area_pct of the zone's
    whole extent (in the zone's own units: camera px, DMD px or part-frame mm; see
    geometry.Coverage). NOT present is a verdict only when even the unseen extent (off the
    image, occluded, shadowed or undecoded), all counted as part, could not make it present
    (and at least min_valid_pct of the zone was measured): else INVALID, an unseen part is
    not an absent one."""
    id = "presence"
    title = "Part present / absent in the zone"
    params = {
        "expect": Param("choice", "present", choices=("present", "absent"),
                        doc="what a good part looks like"),
        "min_height_mm": Param("number", 2.0, "mm", min=0, open_min=True,
                               doc="a point counts as part this far above the reference plane"),
        "min_area_pct": Param("number", 30.0, "pct", min=0, max=100, open_min=True,
                              doc="part as % of the zone's whole extent for PRESENT"),
        "min_valid_pct": Param("number", 10.0, "pct", min=0, max=100,
                               doc="an ABSENT reading needs at least this % of the zone measured "
                                   "(else INVALID: unseen is not absent)"),
        "min_points": Param("int", 30, "count", min=1,
                            doc="fewer measured points in the zone: INVALID"),
    }

    def run(self, ctx) -> CheckResult:
        p, z = self.p, ctx.zone
        cov = z.cover
        h = ctx.heights()
        ys, xs = ctx.index()
        is_part = h >= p["min_height_mm"]
        n_part = int(is_part.sum())
        if cov is not None:                  # over the zone's whole extent, in its own units
            n_cells = int(cov.zmask.sum())
            if n_cells == 0:
                return CheckResult.invalid(f"zone {z.id} has no extent on this scan")
            sel = np.zeros(z.valid.shape, bool)
            sel[ys[is_part], xs[is_part]] = True
            part_pct = 100.0 * int(cov.hits(sel).sum()) / n_cells
            cover = 100.0 * int(cov.seen.sum()) / n_cells
            part_max_pct = min(part_pct + 100.0 - cover, 100.0)   # every unseen cell = part
        else:
            if z.n_px == 0:
                return CheckResult.invalid(f"zone {z.id} covers no camera px in this scan")
            part_pct = 100.0 * n_part / max(z.n_px, 1)
            cover = part_max_pct = None
        part_pct = min(part_pct, 100.0)
        present = part_pct >= p["min_area_pct"]
        values = {"part_pct": round(part_pct, 2),
                  "valid_pct": round(cover, 2) if cover is not None else None,
                  "unseen_pct": round(100.0 - cover, 2) if cover is not None else None,
                  "height_p95_mm": round(float(np.percentile(h, 95)), 4) if h.size else None,
                  "n_points": int(h.size)}
        if not present:
            if cover is None:
                return CheckResult.invalid(
                    f"zone {z.id}: the camera's sampling of it is unknown, so is its coverage: "
                    "an unseen part is not an absent one", values=values)
            if cover < p["min_valid_pct"]:
                return CheckResult.invalid(
                    f"only {cover:.1f} % of the zone measured (< {p['min_valid_pct']:g} %): an "
                    "unseen part is not an absent one", values=values)
            if part_max_pct >= p["min_area_pct"]:
                return CheckResult.invalid(
                    f"part on {part_pct:.1f} % of the zone, but {100.0 - cover:.1f} % of it was "
                    f"not measured: a part there could reach {part_max_pct:.1f} % (>= "
                    f"{p['min_area_pct']:g} %); an unseen part is not an absent one",
                    values=values)
        if h.size < p["min_points"]:
            return CheckResult.invalid(f"only {h.size} measured points in zone {z.id} (need "
                                       f"{p['min_points']})", values=values)
        ok = present == (p["expect"] == "present")
        state = "present" if present else "absent"
        # the values judged (not their rounded copies); a share can't pass 100 %, so the
        # guard band of "present" is a % of the room above the limit (span), not of the limit
        if p["expect"] == "present":
            tols = [tolerance("part_pct", float(part_pct), lo=p["min_area_pct"],
                              span=100.0 - p["min_area_pct"])]
        else:                                   # absent: the worst case (every unseen cell part)
            worst = part_pct if part_max_pct is None else part_max_pct
            tols = [tolerance("part_max_pct" if part_max_pct is not None else "part_pct",
                              float(worst), hi=p["min_area_pct"])]
        faults = []
        if not ok:
            if present:
                faults = [ctx.fault_px(int(np.argmax(h)), "unexpected part")]
            else:
                faults = [{"px": z.centroid_px(), "label": "missing part"}]
        return CheckResult(
            verdict="PASS" if ok else "FAIL", values=values, key="part_pct",
            limit=p["min_area_pct"], fault_points=faults, tolerances=tols,
            # what the check SAW, whatever it expected: a work state compares it with the
            # line's own part-present sensor (engine/flow/verify.py)
            details={"present": bool(present)},
            message=(f"part {state}: {part_pct:.1f} % of the zone >= {p['min_height_mm']:g} mm "
                     f"above the reference (expected {p['expect']})"))


# ================================================================== position

def _pca_angle(uv: np.ndarray) -> tuple:
    """(principal-axis angle deg in the part plane, elongation sqrt(l1 / l2))."""
    c = np.cov((uv - uv.mean(0)).T)
    w, V = np.linalg.eigh(c)
    ax = V[:, 1]
    elong = math.sqrt(w[1] / max(w[0], 1e-12))
    return math.degrees(math.atan2(ax[1], ax[0])), elong


def _wrap90(a: float) -> float:
    return (a + 90.0) % 180.0 - 90.0


BEHIND_MM = 2.0                     # measured this far behind a face plane: not the face


def _disc_outline(sv, px_xy: np.ndarray, uv: np.ndarray, C0, n, radius_mm: float):
    """The outline (endface.outline.measure) of the bright disc of about radius_mm around
    the origin of the in-plane frame (u, v) in the scan's flood image: the plane -> image
    map is the homography of the disc's own measured points (px_xy (N, 2) camera px, uv
    (N, 2) mm), so the outline is in the cloud's mm whatever the camera model; points the
    scan measured more than BEHIND_MM behind the plane (C0, n) are dark to it (a lit
    background is not the part). -> (outline | None, why not)."""
    from ...measure.endface import outline as outline_mod
    flood = sv.flood
    if flood is None:
        return None, "the scan kept no flood pair (scan.kind fast keeps one)"
    to_px = outline_mod.homography_map(px_xy, uv)
    if to_px is None:
        return None, "too few measured points to map the face plane into the image"
    with np.errstate(invalid="ignore"):
        behind = sv.valid & (((sv.xyz - np.asarray(C0, np.float64)) @ np.asarray(n, np.float64))
                             < -BEHIND_MM)
    sat = None if sv.flood_white is None else np.asarray(sv.flood_white, np.float32) >= 254.0
    o = outline_mod.measure(np.asarray(flood, np.float32), to_px, float(radius_mm), sat=sat,
                            exclude=behind)
    return o, o["why"]


def _size_invalid(sc: dict, values: dict, zone_id: str, warn: list) -> "CheckResult":
    """The INVALID of a size mismatch: never a pass, never a definite fail - pull reason
    size_mismatch, which no recipe can turn into a REJECT (engine/disposition.NOT_REJECTABLE:
    it says the calibration or the lens is suspect, nothing about the part)."""
    return CheckResult.invalid(f"size mismatch on zone {zone_id}: {sc['message']}",
                               values=values, pull="size_mismatch",
                               warnings=list(warn) + [sc["message"]],
                               details={"size_check": sc})


@register
class Position(Check):
    """Alignment vs the golden part: the part's points in this zone (>= min_height_mm above
    the reference plane) against the golden scan's in the same zone, in the part frame:
    in-plane offset (centroid, mm), rotation about the plane normal (principal axis, deg;
    not defined for a round footprint), height offset (median, mm)."""
    id = "position"
    title = "Position / alignment vs the golden part"
    needs = ("golden",)
    params = {
        "tol_mm": Param("number", unit="mm", min=0, open_min=True,
                        doc="FAIL when the in-plane offset exceeds this"),
        "tol_deg": Param("number", 2.0, "deg", min=0, open_min=True,
                         doc="FAIL when the rotation exceeds this"),
        "tol_height_mm": Param("number", None, "mm", min=0, open_min=True,
                               doc="optional: FAIL when the height offset exceeds this"),
        "min_height_mm": Param("number", 2.0, "mm", min=0,
                               doc="part points: this far above the reference plane"),
        "min_points": Param("int", 200, "count", min=10, doc="fewer part points: INVALID"),
        "nominal_diameter_mm": Param("number", None, "mm", min=0, open_min=True,
                                     doc="circle zone: the round part's known diameter (its "
                                         "flood outline is held to it)"),
        "size_tol_pct": Param("number", 4.0, "pct", min=0, max=50,
                              doc="circle zone: the part's flood outline vs the golden part's "
                                  "(and nominal_diameter_mm): further off = size mismatch "
                                  "(INVALID, pull size_mismatch); 0 = off"),
    }

    def _outline(self, ctx, golden: bool):
        """The round part's flood outline in a circle zone (the scan's, or the golden's) ->
        (outline | None, why)."""
        from ...measure.endface.profile import robust_plane
        from ..geometry import face_axes
        if golden:
            g = ctx.golden
            sv, fr, z = g.scan, g.frame, g.zones[ctx.zone.id]
        else:
            sv, fr, z = ctx.scan, ctx.frame, ctx.zone
        m = z.valid & (np.nan_to_num(fr.uvh[..., 2], nan=-1e9) >= self.p["min_height_mm"])
        ys, xs = np.nonzero(m)
        if len(xs) < self.p["min_points"]:
            return None, "too few part points"
        P = sv.xyz[ys, xs]
        c, n = robust_plane(P)
        e1, e2 = face_axes(n)
        W = P - c
        u, v = W @ e1, W @ e2
        on = np.abs(W @ n) <= max(2.0, 10 * 1.4826 * float(np.median(np.abs(W @ n))))
        if on.sum() < self.p["min_points"]:
            return None, "too few points on the part's top plane"
        u, v = u - u[on].mean(), v - v[on].mean()            # about the part's own centre
        r = self.p["nominal_diameter_mm"] / 2 if self.p["nominal_diameter_mm"] else \
            2.0 * float(np.sqrt(0.5 * (u[on].var() + v[on].var())))   # a uniform disc: R = 2 sigma
        C0 = c + float(np.mean((P[on] - c) @ e1)) * e1 + float(np.mean((P[on] - c) @ e2)) * e2
        return _disc_outline(sv, np.c_[xs[on], ys[on]], np.c_[u[on], v[on]], C0, n, r)

    def _size(self, ctx) -> tuple:
        """Circle zone: the known-size checks -> ([size_check, ...], values, warnings)."""
        from ...measure.endface import outline as outline_mod
        p = self.p
        if ctx.zone.shape != "circle" or not p["size_tol_pct"]:
            return [], {}, []
        o, why = self._outline(ctx, False)
        if o is None or not o["ok"]:
            return [], {}, [f"size not checked: the part's flood outline could not be measured "
                            f"({why})"]
        vals = {"outline_diameter_mm": round(float(o["diameter_mm"]), 4)}
        out = []
        if p["nominal_diameter_mm"]:
            out.append(outline_mod.outline_size(o, p["nominal_diameter_mm"], p["size_tol_pct"],
                                                "part's flood outline", source="scan"))
        og, gwhy = self._outline(ctx, True)
        if og is not None and og["ok"]:
            vals["golden_outline_diameter_mm"] = round(float(og["diameter_mm"]), 4)
            out.append(outline_mod.size_check(o["diameter_mm"], og["diameter_mm"],
                                              p["size_tol_pct"], "part's flood outline",
                                              source="vs the golden part's outline"))
        if not out:
            return [], vals, [f"size not checked: no nominal_diameter_mm and the golden part's "
                              f"outline could not be measured ({gwhy})"]
        vals["size_deviation_pct"] = max((s["deviation_pct"] for s in out), key=abs)
        return out, vals, []

    def run(self, ctx) -> CheckResult:
        p = self.p
        try:
            ctx.need_golden()
        except LookupError as e:
            return CheckResult.invalid(str(e))
        A, ka = _part(ctx, p["min_height_mm"])
        B, kb = _part(ctx, p["min_height_mm"], golden=True)
        A, B = A[ka], B[kb]
        if len(A) < p["min_points"] or len(B) < p["min_points"]:
            return CheckResult.invalid(
                f"too few part points (scan {len(A)}, golden {len(B)}; need "
                f"{p['min_points']}): is the part in the zone?",
                values={"n_points": int(len(A)), "n_golden_points": int(len(B))})
        d = A[:, :2].mean(0) - B[:, :2].mean(0)
        dh = float(np.median(A[:, 2]) - np.median(B[:, 2]))
        aa, ea = _pca_angle(A[:, :2])
        ab, eb = _pca_angle(B[:, :2])
        rot_ok = min(ea, eb) >= 1.3
        rot = _wrap90(aa - ab) if rot_ok else None
        off = float(np.hypot(*d))
        values = {"dx_mm": round(float(d[0]), 4), "dy_mm": round(float(d[1]), 4),
                  "dz_mm": round(dh, 4), "offset_mm": round(off, 4),
                  "rotation_deg": round(rot, 3) if rot is not None else None,
                  "n_points": int(len(A))}
        sizes, svals, swarn = self._size(ctx)
        values.update(svals)
        bad_size = next((s for s in sizes if s["status"] == "mismatch"), None)
        if bad_size is not None:                # the mm of this scan cannot be trusted
            return _size_invalid(bad_size, values, ctx.zone.id, swarn)
        why = []
        if off > p["tol_mm"]:
            why.append(f"offset {off:.2f} mm > {p['tol_mm']:g} mm")
        if rot is not None and abs(rot) > p["tol_deg"]:
            why.append(f"rotation {rot:+.2f} deg > {p['tol_deg']:g} deg")
        if p["tol_height_mm"] is not None and abs(dh) > p["tol_height_mm"]:
            why.append(f"height {dh:+.2f} mm > {p['tol_height_mm']:g} mm")
        warn = [] if rot_ok else ["rotation not defined: the part's footprint in this zone is "
                                  "round (no principal axis)"]
        warn += swarn
        faults = []
        if why:
            from scipy.spatial import cKDTree
            sub = B if len(B) <= 40000 else B[np.random.default_rng(0).choice(len(B), 40000,
                                                                             replace=False)]
            dist, _ = cKDTree(sub).query(A)
            ys, xs = ctx.index()
            sel = np.nonzero(ka)[0]
            i = sel[int(np.argmax(dist))]
            faults = [{"px": [float(xs[i]), float(ys[i])], "label": "misaligned"}]
        key, limit = "offset_mm", p["tol_mm"]                 # the mm value nearest its limit
        if p["tol_height_mm"] is not None and abs(dh) / p["tol_height_mm"] > off / p["tol_mm"]:
            key, limit = "dz_mm", p["tol_height_mm"]
        tols = [tolerance("offset_mm", off, hi=p["tol_mm"])]          # the values judged
        if rot is not None:
            tols.append(tolerance("rotation_deg", float(rot), -p["tol_deg"], p["tol_deg"]))
        if p["tol_height_mm"] is not None:
            tols.append(tolerance("dz_mm", dh, -p["tol_height_mm"], p["tol_height_mm"]))
        if sizes:
            tols.append(tolerance("size_deviation_pct", float(values["size_deviation_pct"]),
                                  -p["size_tol_pct"], p["size_tol_pct"]))
        return CheckResult(verdict="FAIL" if why else "PASS", values=values, key=key,
                           limit=limit, fault_points=faults, warnings=warn, tolerances=tols,
                           details={"size_check": sizes} if sizes else {},
                           message="; ".join(why) if why else
                           f"aligned: offset {off:.2f} mm" + (f", rotation {rot:+.2f} deg"
                                                              if rot is not None else ""))


# ================================================================== height

@register
class Height(Check):
    """Height of the zone (a statistic of its points' height above the reference plane, or
    above another zone: ref 'zone:<id>', e.g. a step or the height difference of two end
    faces) vs nominal_mm +- tol_mm, or an absolute band_mm [lo, hi]."""
    id = "height"
    title = "Height / step vs the reference plane or another zone"
    params = {
        "ref": Param("str", "plane", doc="'plane' (the reference plane) or 'zone:<id>' (the "
                                         "median height of another zone)"),
        "stat": Param("choice", "median", choices=STATS, doc="statistic of the zone's heights"),
        "nominal_mm": Param("number", None, "mm", doc="nominal height (with tol_mm)"),
        "tol_mm": Param("number", None, "mm", min=0, open_min=True, doc="+- around nominal_mm"),
        "band_mm": Param("pair", None, "mm", doc="absolute [lo, hi] (instead of nominal / tol)"),
        "min_height_mm": Param("number", None, "mm",
                               doc="optional: only points this far above the reference plane "
                                   "(the part's top), in this zone and the ref zone"),
        "min_points": Param("int", 30, "count", min=3, doc="fewer points: INVALID"),
    }

    @classmethod
    def check_params(cls, p: dict) -> None:
        band = p["band_mm"] is not None
        nom = p["nominal_mm"] is not None and p["tol_mm"] is not None
        if band == nom:
            raise ParamError("give either band_mm [lo, hi] or nominal_mm + tol_mm (mm)")
        r = p["ref"]
        if r != "plane" and not (r.startswith("zone:") and len(r) > 5):
            raise ParamError(f"ref must be 'plane' or 'zone:<id>', got {r!r}")

    def zone_refs(self) -> list:
        r = self.p["ref"]
        return [r[5:]] if r.startswith("zone:") else []

    def run(self, ctx) -> CheckResult:
        p = self.p
        uvh, keep = _part(ctx, p["min_height_mm"])
        h = uvh[keep, 2]
        if h.size < p["min_points"]:
            return CheckResult.invalid(f"only {h.size} points in the zone (need "
                                       f"{p['min_points']})", values={"n_points": int(h.size)})
        ref = 0.0
        if p["ref"].startswith("zone:"):
            oz = ctx.other(p["ref"][5:])
            ou, ok = _part(ctx, p["min_height_mm"], zone=oz)
            oh = ou[ok, 2]
            if oh.size < p["min_points"]:
                return CheckResult.invalid(f"reference zone {oz.id}: only {oh.size} points")
            ref = float(np.median(oh))
        val = _stat(h, p["stat"]) - ref
        lo, hi = (p["band_mm"] if p["band_mm"] is not None
                  else (p["nominal_mm"] - p["tol_mm"], p["nominal_mm"] + p["tol_mm"]))
        # deviation from the band's centre (nominal_mm) vs its half width (tol_mm): the
        # headline's deviation semantics (|deviation| > half width <=> outside the band)
        centre, half = (lo + hi) / 2.0, (hi - lo) / 2.0
        values = {"height_mm": round(val, 4), "ref_mm": round(ref, 4),
                  "deviation_mm": round(val - centre, 4), "n_points": int(h.size)}
        bad = not lo <= val <= hi
        faults = []
        if bad:
            sel = np.nonzero(keep)[0]
            i = sel[int(np.argmax(h) if val > hi else np.argmin(h))]
            faults = [ctx.fault_px(int(i), "too high" if val > hi else "too low")]
        what = "above " + (p["ref"][5:] if p["ref"] != "plane" else "the reference plane")
        return CheckResult(verdict="FAIL" if bad else "PASS", values=values, key="deviation_mm",
                           limit=half, fault_points=faults,
                           tolerances=[tolerance("height_mm", float(val), lo, hi)],
                           details={"band_mm": [lo, hi], "centre_mm": centre},
                           message=f"{p['stat']} height {val:+.3f} mm {what} "
                                   f"({'outside' if bad else 'inside'} [{lo:g}, {hi:g}] mm)")


# ================================================================== flatness

def _robust_plane_residual(P: np.ndarray, iters: int = 3, k: float = 3.0) -> np.ndarray:
    from ...measure.endface.profile import robust_plane
    c, n = robust_plane(P, iters, k)
    return (P - c) @ n


@register
class Flatness(Check):
    """Flatness of the zone: robust plane fit, flatness = peak-to-valley of the residuals
    between the trim_pct and 100 - trim_pct percentiles (mm), plus RMS. A flat reading needs
    min_valid_pct of the zone's whole extent (less what was measured below min_height_mm)
    measured: a few strips of a warped face can read flat."""
    id = "flatness"
    title = "Flatness (plane fit)"
    params = {
        "tol_mm": Param("number", unit="mm", min=0, open_min=True,
                        doc="FAIL when the peak-to-valley flatness exceeds this"),
        "rms_tol_mm": Param("number", None, "mm", min=0, open_min=True,
                            doc="optional: FAIL when the RMS exceeds this"),
        "min_height_mm": Param("number", None, "mm",
                               doc="optional: only points this far above the reference plane "
                                   "(the part's top face)"),
        "trim_pct": Param("number", 0.5, "pct", min=0, max=10,
                          doc="percent dropped at each tail (stray triangulated points)"),
        "min_valid_pct": Param("number", 70.0, "pct", min=0, max=100,
                               doc="a flat reading needs at least this % of the zone measured "
                                   "(else INVALID)"),
        "min_points": Param("int", 100, "count", min=10, doc="fewer points: INVALID"),
    }

    def run(self, ctx) -> CheckResult:
        p = self.p
        uvh, keep = _part(ctx, p["min_height_mm"])
        P = ctx.points()[keep]
        if len(P) < p["min_points"]:
            return CheckResult.invalid(f"only {len(P)} points to fit (need {p['min_points']})",
                                       values={"n_points": int(len(P))})
        r = _robust_plane_residual(P)
        lo, hi = np.percentile(r, [p["trim_pct"], 100 - p["trim_pct"]])
        flat = float(hi - lo)
        rms = float(np.sqrt(np.mean((r - np.median(r)) ** 2)))
        ys, xs = ctx.index()
        sel = np.zeros(ctx.zone.valid.shape, bool)
        sel[ys[keep], xs[keep]] = True
        cover = ctx.zone.coverage_pct(sel)
        values = {"flatness_mm": round(flat, 4), "rms_mm": round(rms, 4),
                  "max_dev_mm": round(float(max(abs(lo), abs(hi))), 4),
                  "valid_pct": round(cover, 2) if cover is not None else None,
                  "n_points": int(len(P))}
        why = []
        if flat > p["tol_mm"]:
            why.append(f"flatness {flat:.3f} mm > {p['tol_mm']:g} mm")
        if p["rms_tol_mm"] is not None and rms > p["rms_tol_mm"]:
            why.append(f"RMS {rms:.3f} mm > {p['rms_tol_mm']:g} mm")
        if not why and (cover is None or cover < p["min_valid_pct"]):   # a found warp stands
            return CheckResult.invalid(
                f"only {cover:.0f} % of the zone measured (< {p['min_valid_pct']:g} %): a "
                "partly measured face is never passed flat" if cover is not None else
                f"zone {ctx.zone.id}: the camera's sampling of it is unknown, so is its "
                "coverage: a partly measured face is never passed flat",
                values=values)
        faults = []
        if why:
            inside = (r >= lo) & (r <= hi)
            j = int(np.argmax(np.where(inside, np.abs(r - np.median(r)), -1)))
            faults = [ctx.fault_px(int(np.nonzero(keep)[0][j]), "out of flat")]
        key, limit = "flatness_mm", p["tol_mm"]               # the mm value nearest its limit
        if p["rms_tol_mm"] is not None and rms / p["rms_tol_mm"] > flat / p["tol_mm"]:
            key, limit = "rms_mm", p["rms_tol_mm"]
        tols = [tolerance("flatness_mm", flat, hi=p["tol_mm"])]       # the values judged
        if p["rms_tol_mm"] is not None:
            tols.append(tolerance("rms_mm", rms, hi=p["rms_tol_mm"]))
        return CheckResult(verdict="FAIL" if why else "PASS", values=values, key=key,
                           limit=limit, fault_points=faults, tolerances=tols,
                           message="; ".join(why) if why else f"flat: {flat:.3f} mm "
                                                               f"peak-to-valley, RMS {rms:.3f} mm")


# ================================================================== defect

def _nan_blur(a: np.ndarray, sigma: float) -> np.ndarray:
    import cv2
    m = np.isfinite(a).astype(np.float32)
    num = cv2.GaussianBlur(np.nan_to_num(a).astype(np.float32), (0, 0), sigma)
    den = cv2.GaussianBlur(m, (0, 0), sigma)
    out = num / np.maximum(den, 1e-6)
    out[m == 0] = np.nan
    return out


@register
class Defect(Check):
    """Local dents / bumps. mode 'face' (a circular zone: an end face): the MF-CFAR detector
    of engine/measure/endface (matched dent filters, z against the face's own noise,
    self-referenced: the face's radial profile), triggers deeper than depth_mm FAIL; a
    dent's depth is its depth_centre_mm (the 3 x 3 grid-cell average at its deepest place
    against the local background; the deepest single point over-reads and is reported as
    depth_min_mm). The detector's radii derive from the zone's radius (radii.py). On a face
    zone the flood outline is checked too: its roundness (rim_check: a notch / bulge beyond
    rim_tol_mm over rim_min_arc_mm of rim FAILs - an edge dent, which the height map cannot
    show) and its size (the outline's diameter vs nominal_diameter_mm, default a part_mm
    zone's own diameter: more than size_tol_pct off is a size mismatch - INVALID with pull
    size_mismatch, whatever else was found: calibration or lens suspect).
    mode 'local' (any zone): heights above the reference plane minus their own smooth
    surface (Gaussian, smooth_mm), connected blobs with |residual| >= depth_mm over at least
    min_area_mm2 FAIL; no blob is a PASS only when at least min_valid_pct of the zone's
    inspected extent was measured (its whole extent less what was measured below
    min_height_mm) and no unmeasured gap anywhere in the zone (enclosed or not: a dropout
    reaching the zone border, the part edge or a glint streak counts) could hold a round
    defect of max_gap_mm2 (default min_area_mm2): else INVALID. 'auto' = face on circle
    zones, local otherwise."""
    id = "defect"
    title = "Dents / bumps (local depth and area)"
    params = {
        "depth_mm": Param("number", unit="mm", min=0, open_min=True,
                          doc="a defect at least this deep (or high) FAILs"),
        "min_area_mm2": Param("number", 1.0, "mm2", min=0, doc="smaller blobs are ignored"),
        "mode": Param("choice", "auto", choices=("auto", "face", "local"),
                      doc="face: MF-CFAR on a circular end face; local: residual blobs"),
        "sign": Param("choice", "both", choices=("dent", "both"),
                      doc="dent: only depressions; both: dents and bumps"),
        "smooth_mm": Param("number", 6.0, "mm", min=0, open_min=True,
                           doc="local: the background surface scale (wider than the defects)"),
        "edge_mm": Param("number", 1.5, "mm", min=0,
                         doc="local: ignore this band along the zone / part border"),
        "min_height_mm": Param("number", None, "mm",
                               doc="optional: only points this far above the reference plane"),
        "z_thresh": Param("number", 5.5, "sigma", min=0, open_min=True,
                          doc="face: MF-CFAR z threshold (5.5 ~ 1e-3 false triggers per face)"),
        "rim_margin_mm": Param("number", 0.8, "mm", min=0,
                               doc="face: the outer rim band not inspected (chamfer)"),
        "rim_check": Param("choice", "auto", choices=("auto", "on", "off"),
                           doc="face: the flood outline's roundness (edge dents). auto: "
                               "judged when the flood shows the face's outline (skipped, "
                               "with a warning, only when the scan kept no flood pair or "
                               "shows no bright disc on a dark surround); an outline that "
                               "shows but cannot be judged all the way round is INVALID. "
                               "on: no outline at all is INVALID too; off"),
        "rim_tol_mm": Param("number", 0.4, "mm", min=0, open_min=True,
                            doc="face: a notch / bulge of the outline beyond this ..."),
        "rim_min_arc_mm": Param("number", 1.5, "mm", min=0, open_min=True,
                                doc="face: ... over at least this much rim FAILs"),
        "nominal_diameter_mm": Param("number", None, "mm", min=0, open_min=True,
                                     doc="face: the face's known diameter (default: a part_mm "
                                         "circle zone's own diameter)"),
        "size_tol_pct": Param("number", 4.0, "pct", min=0, max=50,
                              doc="face: the flood outline's diameter vs the nominal one: "
                                  "further off = size mismatch (INVALID, pull size_mismatch: "
                                  "calibration or lens suspect); 0 = off"),
        "min_valid_pct": Param("number", 70.0, "pct", min=0, max=100,
                               doc="a clean reading needs at least this % measured (local: the "
                                   "zone's inspected area; face: the face), else INVALID"),
        "max_gap_mm2": Param("number", None, "mm2", min=0, open_min=True,
                             doc="local: an unmeasured gap in the zone that can hold a disc this "
                                 "large could hide a defect: INVALID unless a defect was found "
                                 "(default min_area_mm2)"),
        "min_points": Param("int", 200, "count", min=20, doc="fewer points: INVALID"),
    }

    def run(self, ctx) -> CheckResult:
        mode = self.p["mode"]
        if mode == "auto":
            mode = "face" if ctx.zone.shape == "circle" else "local"
        if mode == "face" and ctx.zone.shape != "circle":
            return CheckResult.invalid("mode face needs a circle zone (the end face)")
        return self._face(ctx) if mode == "face" else self._local(ctx)

    # ---- generic: residual blobs
    def _local(self, ctx) -> CheckResult:
        import cv2
        from ...measure.metric import pixel_area_mm2
        p, z = self.p, ctx.zone
        sv, fr = ctx.scan, ctx.frame
        m = z.valid.copy()
        if p["min_height_mm"] is not None:
            m &= np.nan_to_num(fr.uvh[..., 2], nan=-1e9) >= p["min_height_mm"]
        n_m = int(m.sum())
        if n_m < p["min_points"]:
            return CheckResult.invalid(f"only {n_m} points in the zone")
        ys, xs = np.nonzero(m)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        H = np.where(m, fr.uvh[..., 2], np.nan)[y0:y1, x0:x1]
        mpp = ctx.mm_per_px() or 0.3
        bg = _nan_blur(H, max(p["smooth_mm"] / mpp / 2.0, 1.0))
        res = H - bg
        k = int(round(p["edge_mm"] / mpp))
        mm = m[y0:y1, x0:x1].astype(np.uint8)
        if k > 0:
            mm = cv2.erode(mm, np.ones((2 * k + 1, 2 * k + 1), np.uint8))
        res[mm == 0] = np.nan
        # the residual of the inspected sign, + = a defect (dent: depth below the surface)
        sig = np.abs(res) if p["sign"] == "both" else -res
        hit = np.nan_to_num(sig, nan=0) >= p["depth_mm"]
        area_px = pixel_area_mm2(np.where(m[..., None], sv.xyz, np.nan))[y0:y1, x0:x1]
        n, lab = cv2.connectedComponents(hit.astype(np.uint8), connectivity=8)
        found = []
        for i in range(1, n):
            b = lab == i
            area = float(np.nansum(area_px[b]))
            if area < p["min_area_mm2"]:
                continue
            j = int(np.nanargmax(np.where(b, np.abs(res), -np.inf)))
            yy, xx = np.unravel_index(j, b.shape)
            found.append({"depth_mm": round(float(-res[yy, xx]), 4), "area_mm2": round(area, 3),
                          "px": [float(xx + x0), float(yy + y0)]})
        found.sort(key=lambda f: -abs(f["depth_mm"]))
        fin = np.isfinite(sig)
        worst = max(float(np.nanmax(sig)), 0.0) if fin.any() else 0.0
        # the headline: the deepest defect counted, else the deepest residual below depth_mm
        # (blobs under min_area_mm2 are not counted either way)
        below = fin & ~hit
        depth = (abs(found[0]["depth_mm"]) if found else
                 max(float(np.max(sig[below])), 0.0) if below.any() else 0.0)
        cover = z.coverage_pct(m)
        max_gap = p["max_gap_mm2"] if p["max_gap_mm2"] is not None else p["min_area_mm2"]
        gap, holds = self._gaps(z, mpp, max_gap)
        values = {"max_depth_mm": round(depth, 4), "max_residual_mm": round(worst, 4),
                  "n_defects": len(found),
                  "worst_depth_mm": found[0]["depth_mm"] if found else None,
                  "worst_area_mm2": found[0]["area_mm2"] if found else None,
                  "valid_pct": round(cover, 2) if cover is not None else None,
                  "largest_gap_mm2": round(gap, 3) if gap is not None else None,
                  "n_points": n_m}
        if not found:                       # a found defect stands; "clean" needs coverage
            if cover is None or gap is None:
                return CheckResult.invalid(f"zone {z.id}: the camera's sampling of it is "
                                           "unknown, so is its coverage: an unmeasured part is "
                                           "never passed", values=values)
            if cover < p["min_valid_pct"]:
                return CheckResult.invalid(f"only {cover:.0f} % of the zone's inspected area "
                                           f"measured (< {p['min_valid_pct']:g} %): an "
                                           "unmeasured area is never passed", values=values)
            if holds:
                return CheckResult.invalid(f"an unmeasured gap of {gap:.1f} mm2 in the zone "
                                           f"can hold a {max_gap:g} mm2 defect: it could hide "
                                           "one", values=values)
        faults = [{"px": f["px"], "label": f"{'dent' if f['depth_mm'] > 0 else 'bump'} "
                                         f"{abs(f['depth_mm']):.2f} mm"} for f in found[:10]]
        return CheckResult(
            verdict="FAIL" if found else "PASS", values=values, key="max_depth_mm",
            limit=p["depth_mm"], fault_points=faults, details={"defects": found[:20]},
            tolerances=[tolerance("max_depth_mm", float(depth), hi=p["depth_mm"])],
            message=(f"{len(found)} defect(s), worst {abs(found[0]['depth_mm']):.3f} mm over "
                     f"{found[0]['area_mm2']:.1f} mm2" if found else
                     f"no defect >= {p['depth_mm']:g} mm (largest residual {worst:.3f} mm, "
                     f"{cover:.0f} % measured)"))

    @staticmethod
    def _gaps(z, mpp: float, max_gap_mm2: float) -> tuple:
        """The zone's unmeasured gaps (its unseen extent, enclosed or not) -> (largest gap
        mm2, True when one can hold a disc of max_gap_mm2: a defect could hide in it);
        (None, None) when the coverage is unknown."""
        import cv2
        c = z.cover
        if c is None:
            return None, None
        u = c.unseen.astype(np.uint8)
        if not u.any():
            return 0.0, False
        cell_mm = c.cell if z.units == "part_mm" else c.px_per_cell * mpp
        n, _lab, st, _ = cv2.connectedComponentsWithStats(u, connectivity=8)
        largest = float(st[1:, cv2.CC_STAT_AREA].max()) * cell_mm * cell_mm if n > 1 else 0.0
        r = math.sqrt(max_gap_mm2 / math.pi) / cell_mm            # the disc's radius, cells
        if r < 1.0:                                             # finer than a cell: by area
            return largest, largest >= max_gap_mm2
        k = int(math.ceil(r))
        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        core = cv2.erode(u, disc, borderType=cv2.BORDER_CONSTANT, borderValue=0)
        return largest, bool(core.any())

    # ---- circular end face: MF-CFAR + the rim outline + the known size
    def _face(self, ctx) -> CheckResult:
        from ...measure.endface import mfcfar
        from ...measure.endface import outline as outline_mod
        from ...measure.endface import radii as radii_mod
        from ...measure.endface.profile import grid_axis, robust_plane
        from ..geometry import face_axes
        p, z, sv, fr = self.p, ctx.zone, ctx.scan, ctx.frame
        ys, xs = np.nonzero(z.valid)
        if len(xs) < p["min_points"]:
            return CheckResult.invalid(f"only {len(xs)} measured points on the face")
        P = sv.xyz[ys, xs]
        c, n = robust_plane(P)
        # the face centre: the zone circle's centre on the face plane
        nominal = p["nominal_diameter_mm"]
        if z.units == "part_mm":
            (cu, cv), radius = z.coords["center"], z.coords["radius"]
            a = fr.to_xyz(cu, cv, 0.0)
            den = float(fr.n @ n)
            C0 = a + fr.n * (float((c - a) @ n) / den if abs(den) > 1e-6 else 0.0)
            if nominal is None:
                nominal = 2.0 * float(radius)           # the zone IS the face, in mm
        else:
            if z.units == "camera_px":
                cpx, rpx = z.coords["center"], z.coords["radius"]
            else:                                        # a DMD circle: its camera centroid
                cpx, rpx = z.centroid_px(), math.sqrt(z.n_px / math.pi)
            d = sv.ray(cpx)
            C0 = d * float(c @ n) / float(d @ n)
            radius = rpx * (ctx.mm_per_px() or 0.3)
        e1, e2 = face_axes(n)
        W = P - C0
        u, v, h = W @ e1, W @ e2, W @ n
        med = float(np.median(h))
        mad = float(np.median(np.abs(h - med)))
        good = (np.abs(h - med) <= max(2.0, 10 * 1.4826 * mad)) & (np.hypot(u, v) <= radius + 1.0)
        if good.sum() < p["min_points"]:
            return CheckResult.invalid(f"only {int(good.sum())} points on the face plane")
        # every radius from the zone's: the inspected flat ends rim_margin_mm inside it
        inspect = min(max(radius - p["rim_margin_mm"], 1.0), radius)
        try:
            rr = radii_mod.face_radii(radius, {"inspect_r_mm": inspect})
        except ValueError as e:
            return CheckResult.invalid(f"zone {z.id}: {e}")
        prm = mfcfar.mf_params({
            "z_thresh": p["z_thresh"], "min_depth_mm": p["depth_mm"], "min_area_mm2": 0.0,
            "sign": "dent" if p["sign"] == "dent" else "both"}, radii=rr)
        px = mfcfar.Pixels(iy=ys[good], ix=xs[good], U=u[good][None], V=v[good][None],
                           H=h[good][None], cam_mask=np.ones((1, 1), bool))
        g = grid_axis(radius, 0.25)
        mpp = ctx.mm_per_px()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            gr = mfcfar.Gridder(px, g, radius, prm["max_edge_mm"])
            maps = gr.repeat_maps()
            ctx.keep_sections(g, maps[0])       # the face's two centre lines: files, no result
            det = mfcfar.detect(maps, None, g, prm, px_mm=mpp)
        trig = [t for t in det["triggers"] if abs(t["depth_mm"]) >= p["depth_mm"]
                and (p["sign"] == "both" or t["kind"] == "dent")]
        faults = []
        for t in trig[:10]:
            X = C0 + t["u_mm"] * e1 + t["v_mm"] * e2
            faults.append({"px": ctx.px_of(X), "label": f"{t['kind']} {abs(t['depth_mm']):.2f} mm"})
        # the headline: the deepest dent counted, else the deepest significant one (z over the
        # threshold, the counted sign) still under depth_mm - so a PASS reports how close it
        # came to the limit (the recipe's guard band reads it), never a flat 0
        worst = (max(abs(t["depth_mm"]) for t in trig) if trig else
                 self._face_near(det, p["depth_mm"], p["sign"]))
        values = {"max_depth_mm": round(worst, 4), "n_defects": len(trig),
                  "max_z": det.get("max_z"), "coverage_pct": det.get("coverage_pct"),
                  "face_radius_mm": round(radius, 3), "n_points": int(good.sum())}
        warn = []
        tols = [tolerance("max_depth_mm", float(worst), hi=p["depth_mm"])]
        details = {"triggers": trig[:20], "noise_floor_mm": det.get("noise_floor_mm"),
                   "radii": radii_mod.public(rr)}
        # ---- the flood outline: the known size, then edge dents
        o, o_why = None, "rim_check off and size_tol_pct 0"
        if p["rim_check"] != "off" or p["size_tol_pct"]:
            o, o_why = _disc_outline(sv, np.c_[xs[good], ys[good]], np.c_[u[good], v[good]],
                                     C0 + med * n, n, radius)
        sc = outline_mod.outline_size(o, nominal, p["size_tol_pct"], "face's flood outline",
                                      source="scan flood", why=o_why)
        details["size_check"] = sc
        if o is not None and o["ok"]:
            values["outline_diameter_mm"] = round(float(o["diameter_mm"]), 4)
            values["size_deviation_pct"] = sc.get("deviation_pct")
        if sc["status"] == "mismatch":          # nothing measured here can be trusted
            return _size_invalid(sc, values, z.id, warn)
        if sc["status"] == "not_checked" and p["size_tol_pct"]:
            warn.append(sc["message"])
        elif sc["status"] == "ok":
            tols.append(tolerance("size_deviation_pct", float(sc["deviation_pct"]),
                                  -p["size_tol_pct"], p["size_tol_pct"]))
        rim_fail = []
        if p["rim_check"] != "off":
            j = outline_mod.judge(o, p["rim_tol_mm"], p["rim_min_arc_mm"], 0.0)
            details["rim_outline"] = outline_mod.public(o, j) if o is not None else \
                {"measured": False, "status": j["status"], "why": o_why}
            if j["status"] == "incomplete" and not trig:
                # the outline shows, but not all of it could be judged (a chip deep enough
                # leaves no edge, a glint hides one): never passed, whatever rim_check says
                why = j.get("why") or (o or {}).get("why") or o_why
                return CheckResult.invalid(f"the rim outline could not be judged all the way "
                                           f"round ({why}): edge dents unchecked there, the "
                                           "face can't pass", values=values, pull="incomplete",
                                           warnings=warn, details=details)
            if j["status"] in ("not_measured", "incomplete"):
                why = j.get("why") or (o or {}).get("why") or o_why
                if p["rim_check"] == "on" and not trig:
                    return CheckResult.invalid(f"the rim outline could not be judged ({why}): "
                                               "edge dents unchecked (rim_check on)",
                                               values=values, warnings=warn, details=details)
                warn.append(f"rim outline not checked ({why}): edge dents are not looked for")
            else:
                rim_fail = j["findings"]
                peak = max([float(j["arc_peak_mm"] or 0.0)]
                           + [abs(f["depth_mm"]) for f in rim_fail])
                values.update(rim_rms_mm=round(float(j["rms_mm"]), 4),
                              rim_max_mm=round(float(j["max_abs_mm"]), 4),
                              rim_arc_peak_mm=round(peak, 4), n_rim_defects=len(rim_fail))
                tols.append(tolerance("rim_arc_peak_mm", peak, hi=p["rim_tol_mm"]))
                r0 = float(o["radius_mm"])
                off = np.asarray(o["centre_offset_mm"], np.float64)
                for f in rim_fail[:10]:         # the mark sits just inside the rim, on the face
                    t = math.radians(f["theta_deg"])
                    X = C0 + med * n + (off[0] + 0.9 * r0 * math.cos(t)) * e1 \
                        + (off[1] + 0.9 * r0 * math.sin(t)) * e2
                    faults.append({"px": ctx.px_of(X),
                                   "label": f"rim {f['kind']} {abs(f['depth_mm']):.2f} mm"})
        cov = det.get("coverage_pct") or 0.0
        if not trig and not rim_fail and cov < p["min_valid_pct"]:
            return CheckResult.invalid(f"only {cov:.0f} % of the face measured (< "
                                       f"{p['min_valid_pct']:g} %): an unmeasured face is never "
                                       "passed", values=values, warnings=warn, details=details)
        bad = bool(trig or rim_fail)
        msg = []
        if trig:
            msg.append(f"{len(trig)} defect(s) on the face, deepest "
                       f"{abs(trig[0]['depth_mm']):.3f} mm (z {trig[0]['z']:.1f})")
        if rim_fail:
            f0 = rim_fail[0]
            msg.append(f"{len(rim_fail)} rim {f0['kind']}(es): {abs(f0['depth_mm']):.2f} mm over "
                       f"{f0['arc_mm']:g} mm of rim at {f0['theta_deg']:g} deg")
        return CheckResult(
            verdict="FAIL" if bad else "PASS", values=values,
            key="max_depth_mm" if trig or not rim_fail else "rim_arc_peak_mm",
            limit=p["depth_mm"] if trig or not rim_fail else p["rim_tol_mm"],
            fault_points=faults, tolerances=tols, warnings=warn, details=details,
            message=("; ".join(msg) if bad else
                     f"face clear (no dent >= {p['depth_mm']:g} mm at z >= {p['z_thresh']:g}"
                     + (f"; deepest significant {worst:.3f} mm" if worst > 0 else "")
                     + (f"; rim outline within {p['rim_tol_mm']:g} mm"
                        if "rim_arc_peak_mm" in values else "") + ")"))

    @staticmethod
    def _face_near(det: dict, depth_mm: float, sign: str) -> float:
        """The deepest significant dent (z >= the detector's threshold, in the inspected
        zone, the counted sign) shallower than depth_mm - the detector's candidates that
        did not reach the depth limit (depth_centre_mm, as every depth); 0.0 when none."""
        # (a depth that rounds up to the limit counts as ON it: inside any guard band)
        d = [min(abs(c["depth_mm"]), depth_mm) for c in det.get("candidates") or []
             if not c.get("trigger") and (sign == "both" or c["kind"] == "dent")]
        return float(max(d)) if d else 0.0


# ================================================================== edge

def _blob(ctx, min_h, golden=False):
    """The part's region in the zone (points >= min_h above the reference, largest
    connected component) -> (blob mask, scan view, frame) of the scan or the golden."""
    import cv2
    if golden:
        g = ctx.golden
        sv, fr, z = g.scan, g.frame, g.zones[ctx.zone.id]
    else:
        sv, fr, z = ctx.scan, ctx.frame, ctx.zone
    m = z.valid & (np.nan_to_num(fr.uvh[..., 2], nan=-1e9) >= min_h)
    if m.sum() < 50:
        return None, sv, fr
    m8 = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m8, connectivity=8)
    if n < 2:
        return None, sv, fr
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    return lab == k, sv, fr


def edge_points(img: np.ndarray, blob: np.ndarray, step_px: float = 2.0, search_px: float = 4.0,
                trim: float = 0.05, ref_axis=None):
    """Sub-pixel edges of the blob's two long sides in img: intensity profiles across the
    part's axis every step_px, the gradient peak within search_px of the blob border, a
    parabolic sub-pixel fit of that peak -> (side a (N, 2), side b (M, 2) camera px, axis,
    normal). The axis points along ref_axis (camera px) when given, so the sides are named
    as on that part (the golden's); else along its dominant component (+x for a left-right
    part, +y for a top-bottom one)."""
    import cv2
    ys, xs = np.nonzero(blob)
    (cx, cy), (w, h), ang = cv2.minAreaRect(np.c_[xs, ys].astype(np.float32))
    if w < h:
        w, h, ang = h, w, ang + 90.0
    t = math.radians(ang)
    ax = np.array([math.cos(t), math.sin(t)])
    if ref_axis is not None:
        if float(ax @ np.asarray(ref_axis, float)) < 0:
            ax = -ax
    elif ax[int(np.argmax(np.abs(ax)))] < 0:
        ax = -ax
    nr = np.array([-ax[1], ax[0]])                       # side a = -nr (top of a left-right part)
    g = cv2.GaussianBlur(np.nan_to_num(img).astype(np.float32), (0, 0), 1.0)
    bf = blob.astype(np.float32)
    span = h / 2 + 12
    offs = np.arange(-span, span, 0.25)
    S = int(round(search_px / 0.25))
    A, B = [], []
    for s in np.arange(-w / 2 + trim * w, w / 2 - trim * w, step_px):
        c = np.array([cx, cy]) + ax * s
        pts = (c[None] + nr[None] * offs[:, None]).astype(np.float32)
        prof = cv2.remap(g, pts[:, 0:1], pts[:, 1:2], cv2.INTER_LINEAR).ravel()
        inb = cv2.remap(bf, pts[:, 0:1], pts[:, 1:2], cv2.INTER_NEAREST).ravel() > 0.5
        if inb.sum() < 8:
            continue
        gr = np.abs(np.gradient(prof))
        idx = np.nonzero(inb)[0]
        for side, k0, out in (("a", idx[0], A), ("b", idx[-1], B)):
            lo, hi = max(k0 - S, 1), min(k0 + S, len(gr) - 1)
            gg = gr[lo:hi]
            if len(gg) < 3:
                continue
            k = int(np.argmax(gg))
            dlt = 0.0
            if 0 < k < len(gg) - 1:
                den = gg[k - 1] - 2 * gg[k] + gg[k + 1]
                dlt = (gg[k - 1] - gg[k + 1]) / (2 * den) if abs(den) > 1e-9 else 0.0
            out.append(c + nr * (offs[lo + k] + dlt * 0.25))
    return np.array(A).reshape(-1, 2), np.array(B).reshape(-1, 2), ax, nr


def _edge_uv(pts_px: np.ndarray, blob: np.ndarray, sv, fr, nr, inward: float,
             win: int = 6) -> np.ndarray:
    """Edge points (camera px, sub-pixel) -> part-frame (u, v): a local affine camera px ->
    (u, v) fitted on the part's measured px just inside the edge, evaluated at the edge point
    (keeps the sub-pixel edge; NaN where the window holds too few points)."""
    H, W = blob.shape
    uvf = fr.uvh[..., :2]
    good = blob & np.isfinite(uvf).all(-1)
    out = np.full((len(pts_px), 2), np.nan)
    for i, p in enumerate(np.asarray(pts_px, float)):
        c = p + nr * inward * 3.0
        x0, y0 = max(int(c[0]) - win, 0), max(int(c[1]) - win, 0)
        x1, y1 = min(int(c[0]) + win + 1, W), min(int(c[1]) + win + 1, H)
        if x1 <= x0 or y1 <= y0:
            continue
        ys, xs = np.nonzero(good[y0:y1, x0:x1])
        if len(xs) < 12:
            continue
        xs, ys = xs + x0, ys + y0
        A = np.c_[xs, ys, np.ones(len(xs))]
        co = np.linalg.lstsq(A, uvf[ys, xs], rcond=None)[0]
        out[i] = np.array([p[0], p[1], 1.0]) @ co
    return out


def _line(uv: np.ndarray):
    """Robust line through (u, v) points -> (point, direction, residuals (N,) mm, keep). The
    direction's sign is the line's dominant component's (+u for a left-right edge, +v for a
    top-bottom one); a line has no direction, so compare two with _line_angle."""
    from scipy.ndimage import median_filter
    c = uv.mean(0)
    _, _, vt = np.linalg.svd(uv - c, full_matrices=False)
    d = vt[0] if vt[0][int(np.argmax(np.abs(vt[0])))] >= 0 else -vt[0]
    nrm = np.array([-d[1], d[0]])
    s = (uv - c) @ d
    o = np.argsort(s)
    r = median_filter(((uv - c) @ nrm)[o], size=5, mode="nearest")   # single glints out
    S = s[o]
    keep = np.ones(len(S), bool)
    co = np.polyfit(S, r, 1)
    for _ in range(2):
        res = r - np.polyval(co, S)
        keep = np.abs(res) < 3 * np.std(res[keep]) + 0.05
        co = np.polyfit(S[keep], r[keep], 1)
    res = r - np.polyval(co, S)
    inv = np.empty_like(o)
    inv[o] = np.arange(len(o))
    # the fitted line in the plane: through c + nrm * co[1], direction d + nrm * co[0]
    dirn = d + nrm * co[0]
    dirn /= np.linalg.norm(dirn)
    return c + nrm * co[1], dirn, res[inv], keep[inv]


def _line_angle(ref: np.ndarray, d: np.ndarray) -> float:
    """Angle (deg, (-90, 90]) that turns the LINE along ref onto the line along d: lines are
    undirected, so a direction and its opposite are the same line (never ~180 deg)."""
    a = math.degrees(math.atan2(ref[0] * d[1] - ref[1] * d[0], float(ref @ d)))
    a = (a + 90.0) % 180.0 - 90.0
    return 90.0 if a == -90.0 else a


@register
class Edge(Check):
    """Straightness of the part's long edge(s) in the zone, and (with the golden) the edge
    position / angle. Edges are found sub-pixel in the flood image (white - black; the height
    map when the scan has no flood pair, or source 'height'), mapped to the part frame (mm)
    and fitted with a robust line. side 'top' = the -normal side of the part's axis (the top
    edge of a part running left-right in the image), 'bottom' the other, 'both' = both."""
    id = "edge"
    title = "Edge straightness / position"
    params = {
        "side": Param("choice", "both", choices=("both", "top", "bottom"),
                      doc="which long edge (top = smaller image y for a left-right part)"),
        "straightness_tol_mm": Param("number", None, "mm", min=0, open_min=True,
                                     doc="FAIL when the edge deviates from its line by more"),
        "position_tol_mm": Param("number", None, "mm", min=0, open_min=True,
                                 doc="FAIL when the edge moved more than this vs the golden "
                                     "(perpendicular, at the zone centre)"),
        "angle_tol_deg": Param("number", None, "deg", min=0, open_min=True,
                               doc="FAIL when the edge turned more than this vs the golden"),
        "source": Param("choice", "auto", choices=("auto", "flood", "height"),
                        doc="edge image: auto = flood when the scan kept one"),
        "min_height_mm": Param("number", 2.0, "mm", min=0,
                               doc="the part: points this far above the reference plane"),
        "trim_pct": Param("number", 5.0, "pct", min=0, max=40,
                          doc="percent of the part length skipped at each end"),
        "min_points": Param("int", 20, "count", min=5, doc="fewer edge samples: INVALID"),
    }

    @classmethod
    def check_params(cls, p: dict) -> None:
        if p["straightness_tol_mm"] is None and p["position_tol_mm"] is None \
                and p["angle_tol_deg"] is None:
            raise ParamError("set at least one of straightness_tol_mm, position_tol_mm (golden), "
                             "angle_tol_deg (golden)")

    def needs_for(self) -> set:
        return {"golden"} if (self.p["position_tol_mm"] is not None
                              or self.p["angle_tol_deg"] is not None) else set()

    def _edges(self, ctx, golden: bool, ref_axis=None):
        """-> ({side: edge}, None) or (None, why); out["_axis"] = the part's axis (camera
        px), to name the scan's sides as the golden's (ref_axis)."""
        p = self.p
        blob, sv, fr = _blob(ctx, p["min_height_mm"], golden)
        if blob is None:
            return None, "no part region in the zone (nothing above min_height_mm)"
        src = p["source"]
        img = sv.flood if src in ("auto", "flood") else None
        if img is None:
            if src == "flood":
                return None, "source flood but the scan kept no flood pair (scan.kind fast)"
            img = np.nan_to_num(fr.uvh[..., 2], nan=0.0)
        a, b, ax, nr = edge_points(img, blob, trim=p["trim_pct"] / 100.0, ref_axis=ref_axis)
        out = {}
        for side, pts, inward in (("top", a, 1.0), ("bottom", b, -1.0)):
            if p["side"] not in ("both", side):
                continue
            # every side asked for must be measured: a missing side is never skipped silently
            if len(pts) < p["min_points"]:
                return None, (f"only {len(pts)} edge samples on the {side} edge (need "
                              f"{p['min_points']})")
            uv = _edge_uv(pts, blob, sv, fr, nr, inward)
            ok = np.isfinite(uv).all(1)
            if ok.sum() < p["min_points"]:
                return None, (f"only {int(ok.sum())} {side}-edge samples map to the part frame "
                              f"(need {p['min_points']})")
            c, d, res, keep = _line(uv[ok])
            out[side] = {"px": pts[ok], "uv": uv[ok], "point": c, "dir": d, "res": res,
                         "keep": keep}
        if not out:
            return None, f"fewer than {p['min_points']} edge samples on the {p['side']} edge(s)"
        out["_axis"] = ax
        return out, None

    def run(self, ctx) -> CheckResult:
        p = self.p
        gold, ref = None, None
        if self.needs_for():
            try:
                ctx.need_golden()
            except LookupError as e:
                return CheckResult.invalid(str(e))
            gold, gwhy = self._edges(ctx, True)
            if gold is None:
                return CheckResult.invalid(f"golden: {gwhy}")
            ref = gold.pop("_axis")             # the scan's sides named as the golden's
        cur, why = self._edges(ctx, False, ref_axis=ref)
        if cur is None:
            return CheckResult.invalid(why)
        cur.pop("_axis")
        values, fails, faults, ratios, tols = {}, [], [], [], []
        compared = 0                                   # tolerances actually compared
        for side, e in cur.items():
            st = float(np.max(np.abs(e["res"])))      # every sample (the trim only fits the line)
            values[f"{side}_straightness_mm"] = round(st, 4)
            values[f"n_{side}_points"] = int(len(e["res"]))
            if p["straightness_tol_mm"] is not None and st > p["straightness_tol_mm"]:
                fails.append(f"{side} edge straightness {st:.3f} mm > {p['straightness_tol_mm']:g}")
                i = int(np.argmax(np.abs(e["res"])))
                faults.append({"px": [float(e["px"][i, 0]), float(e["px"][i, 1])],
                               "label": f"{side} edge {st:.2f} mm"})
            if p["straightness_tol_mm"] is not None:
                compared += 1
                ratios.append((st / p["straightness_tol_mm"], f"{side}_straightness_mm",
                               p["straightness_tol_mm"]))
                tols.append(tolerance(f"{side}_straightness_mm", st,
                                      hi=p["straightness_tol_mm"]))
            if gold is not None:
                if side not in gold:
                    return CheckResult.invalid(f"golden: no {side} edge to compare with",
                                               values=values)
                g = gold[side]
                mid = e["uv"].mean(0)
                gn = np.array([-g["dir"][1], g["dir"][0]])
                off = float((e["point"] + e["dir"] * float((mid - e["point"]) @ e["dir"])
                             - g["point"]) @ gn)
                ang = _line_angle(g["dir"], e["dir"])      # lines: never ~180 deg
                compared += (p["position_tol_mm"] is not None) + (p["angle_tol_deg"] is not None)
                values[f"{side}_offset_mm"] = round(off, 4)
                values[f"{side}_angle_deg"] = round(ang, 3)
                bad = []
                if p["position_tol_mm"] is not None and abs(off) > p["position_tol_mm"]:
                    bad.append(f"{side} edge moved {off:+.2f} mm (> {p['position_tol_mm']:g})")
                if p["angle_tol_deg"] is not None and abs(ang) > p["angle_tol_deg"]:
                    bad.append(f"{side} edge turned {ang:+.2f} deg (> {p['angle_tol_deg']:g})")
                if bad:
                    fails += bad
                    # the edge sample farthest from the golden line marks the fault
                    dg = np.abs((e["uv"] - g["point"]) @ gn)
                    i = int(np.argmax(dg))
                    faults.append({"px": [float(e["px"][i, 0]), float(e["px"][i, 1])],
                                   "label": f"{side} edge off {float(dg[i]):.2f} mm"})
                if p["position_tol_mm"] is not None:
                    ratios.append((abs(off) / p["position_tol_mm"], f"{side}_offset_mm",
                                   p["position_tol_mm"]))
                    tols.append(tolerance(f"{side}_offset_mm", off, -p["position_tol_mm"],
                                          p["position_tol_mm"]))
                if p["angle_tol_deg"] is not None:
                    tols.append(tolerance(f"{side}_angle_deg", float(ang),
                                          -p["angle_tol_deg"], p["angle_tol_deg"]))
        if not compared:                                # nothing was held to a tolerance
            return CheckResult.invalid("no edge tolerance could be compared", values=values)
        _, key, limit = max(ratios) if ratios else (0, None, None)
        return CheckResult(verdict="FAIL" if fails else "PASS", values=values, key=key,
                           limit=limit, fault_points=faults, tolerances=tols,
                           message="; ".join(fails) if fails else
                           "edge(s) " + ", ".join(f"{k} {v:g}" for k, v in values.items()
                                                  if k.endswith("_mm")))


# ================================================================== code

@register
class Code(Check):
    """A barcode / QR code in the zone: the codes read under DMD light before the scan
    (scan.codes: true, engine/sense/codes.py: bracketed floods + red light) whose corners lie
    in the zone, else the zone of the scan's flood image decoded. FAIL when fewer than
    min_count codes match (expect: exact text; pattern: a regular expression; kind: QR,
    EAN_13, CODE_128, DATA_MATRIX, ...). A FAIL is definite only when a code of the expected
    kind was read with the wrong text; when the shortfall is codes that could not be read at
    all, the FAIL carries pull "code_unreadable" (unreadable is not wrong: the part is PULLed
    for quality inspection, engine/disposition.py)."""
    id = "code"
    title = "Code read (QR / Data Matrix / 1D)"
    params = {
        "expect": Param("str", None, doc="exact text the code must carry (optional)"),
        "pattern": Param("str", None, doc="regular expression the text must match (optional)"),
        "kind": Param("str", None, doc="symbology, e.g. QR, DATA_MATRIX, EAN_13 (optional)"),
        "min_count": Param("int", 1, "count", min=1, doc="codes that must match"),
        "margin_px": Param("int", 20, "px", min=0, doc="camera px added around the zone"),
    }

    @classmethod
    def check_params(cls, p: dict) -> None:
        if p["pattern"] is not None:
            try:
                re.compile(p["pattern"])
            except re.error as e:
                raise ParamError(f"pattern is not a valid regular expression: {e}") from None

    def _kind_ok(self, c) -> bool:
        from ...sense.codes import normalise_kind
        return self.p["kind"] is None or normalise_kind(c.kind) == normalise_kind(self.p["kind"])

    def _match(self, c) -> bool:
        p = self.p
        if not self._kind_ok(c):
            return False
        if p["expect"] is not None and c.data != p["expect"]:
            return False
        if p["pattern"] is not None and not re.search(p["pattern"], c.data or ""):
            return False
        return True

    def run(self, ctx) -> CheckResult:
        from ...sense import codes as codes_mod
        p, z = self.p, ctx.zone
        m = z.mask
        if not m.any():
            return CheckResult.invalid(f"zone {z.id} covers no camera px")
        ys, xs = np.nonzero(m)
        H, W = m.shape
        k = p["margin_px"]
        x0, y0 = max(int(xs.min()) - k, 0), max(int(ys.min()) - k, 0)
        x1, y1 = min(int(xs.max()) + 1 + k, W), min(int(ys.max()) + 1 + k, H)
        found, source = [], None
        if ctx.codes:
            source = "read_codes"
            for c in ctx.codes:
                if c.corners:
                    cc = np.asarray(c.corners, float).mean(0)
                    if x0 <= cc[0] < x1 and y0 <= cc[1] < y1:
                        found.append(c)
        if not found:
            img = ctx.scan.flood_white
            if img is None and not ctx.codes:
                return CheckResult.invalid("no image to read a code from: the scan kept no flood "
                                           "pair and scan.codes is off")
            if img is not None:
                source = source or "flood"
                crop = np.asarray(img, np.float32)[y0:y1, x0:x1]
                for c in codes_mod.decode_image(crop, illumination="flood"):
                    if c.corners:
                        c.corners = [[float(a) + x0, float(b) + y0] for a, b in c.corners]
                    found.append(c)
        hits = [c for c in found if self._match(c)]
        values = {"n_codes": len(found), "n_matching": len(hits)}
        ok = len(hits) >= p["min_count"]
        wrong = [c for c in found if self._kind_ok(c) and not self._match(c)]
        unreadable = not ok and not wrong       # nothing of that kind read wrong: unread
        faults = [] if ok else [{"px": z.centroid_px(),
                                 "label": "code unreadable" if unreadable else "code wrong"}]
        return CheckResult(
            verdict="PASS" if ok else "FAIL", values=values, key=None, limit=None,
            fault_points=faults, tolerances=[],
            pull="code_unreadable" if unreadable else None,
            details={"codes": [{"kind": c.kind, "data": c.data, "corners": c.corners}
                               for c in found], "source": source},
            message=(f"{len(hits)} matching code(s): " + ", ".join(repr(c.data) for c in hits)
                     if ok else f"{len(hits)} of {p['min_count']} matching code(s) "
                                f"({len(found)} read: " + ", ".join(repr(c.data) for c in found)
                                + ")" + (": the expected code could not be read (unreadable, "
                                         "not wrong)" if unreadable else "")))

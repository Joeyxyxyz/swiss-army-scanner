"""
Known-geometry check block: is the stereo calibration giving true millimetres?

MOCK-TESTED ONLY: not yet used on a real rig.

One printed block (tools/make_calib_fixture.py --block writes the STL, this spec as JSON
and a template for the caliper values) with, on a flat REFERENCE face:
    step1..stepN   a step gauge, tops parallel to the reference face (default 25, 20, 15,
                   10, 5 mm high, tallest at -x: put the tall end AWAY from the projector
                   so no step shadows the next one)
    cylinder       a lying half-round bar (D profile: a half cylinder on a pedestal of the
                   same width, so calipers measure its diameter across the flats)
    dome           a hemisphere on the reference face (its apex height is its radius)

Block frame B: origin at the centre of the reference face, x along the long side, y across,
z up out of the face (toward the camera when it faces the camera); slab bottom z = -T.

The scan is registered to the nominal model (reference plane, then point-to-plane ICP, four
in-plane turns tried), each feature is cut out by its nominal footprint with a margin, and:
    reference flatness   plane fit RMS (+ peak-to-valley) over the bare reference face
    step heights         median distance of each step top above the reference plane
    cylinder             diameter (cylinder fit) and the height of its top line
    dome                 diameter (sphere fit) and its apex height
are compared with what the PRINTED block measures (calipers / height gauge, the --measured
JSON). Against nominal values only, every verdict is a doubt (PULL): a print is never exactly
its model.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional, Union

import numpy as np

from .. import disposition as D
from . import metric

KIND = "check_block"
MEASURED_KIND = "check_block_measured"
VERSION = 1
EPS_MM = 0.05                   # width of a vertical wall in the heightfield mesh
TOL_HEIGHT_MM = 0.10
TOL_DIAMETER_MM = 0.15
FLATNESS_MAX_MM = 0.10          # reference face plane-fit RMS
MIN_POINTS = 150
OUTLIER_MM = 0.5                # off the reference plane by more: an outlier, not flatness
MAX_OUTLIER_PCT = 2.0           # more outliers than this: the flatness is not measured


# ------------------------------------------------------------------ design

def design(step_heights=(25.0, 20.0, 15.0, 10.0, 5.0), step_width_mm: float = 12.0,
           depth_mm: float = 40.0, cylinder_radius_mm: float = 10.0,
           pedestal_mm: float = 3.0, dome_radius_mm: float = 12.0, gap_mm: float = 10.0,
           margin_mm: float = 12.0, slab_mm: float = 8.0, bed_mm=(220.0, 220.0),
           name: str = "checkblock") -> dict:
    """The parametric check block -> spec dict (nominal dimensions in the block frame)."""
    hs = [float(h) for h in step_heights]
    if len(hs) < 2 or min(hs) <= 0 or len(set(hs)) != len(hs):
        raise ValueError(f"step_heights: >= 2 distinct positive heights, got {step_heights!r}")
    r, p, rd = float(cylinder_radius_mm), float(pedestal_mm), float(dome_radius_mm)
    if min(r, rd, step_width_mm, depth_mm) <= 0 or p < 0:
        raise ValueError("cylinder / dome radius, step width and depth must be > 0")
    if 2 * rd > depth_mm + 2 * margin_mm - 4:
        raise ValueError("the dome does not fit the block depth")
    L = 2 * margin_mm + len(hs) * step_width_mm + gap_mm + 2 * r + gap_mm + 2 * rd
    Dd = 2 * margin_mm + max(depth_mm, 2 * rd)
    fit = (float(bed_mm[0]) - 10.0, float(bed_mm[1]) - 10.0)
    if not (min(L, Dd) <= min(fit) and max(L, Dd) <= max(fit)):
        raise ValueError(f"the block is {L:.1f} x {Dd:.1f} mm, the bed {bed_mm[0]:g} x "
                         f"{bed_mm[1]:g} mm: fewer / narrower steps or smaller features")
    x = -L / 2 + margin_mm
    y = [-depth_mm / 2, depth_mm / 2]
    feats = []
    for i, h in enumerate(hs):
        feats.append({"name": f"step{i + 1}", "kind": "step", "x": [x, x + step_width_mm],
                      "y": y, "height_mm": h})
        x += step_width_mm
    x += gap_mm
    feats.append({"name": "cylinder", "kind": "cylinder", "axis": "y", "x_c": x + r, "y": y,
                  "radius_mm": r, "pedestal_mm": p, "diameter_mm": 2 * r,
                  "height_mm": p + r})
    x += 2 * r + gap_mm
    feats.append({"name": "dome", "kind": "dome", "centre": [x + rd, 0.0], "radius_mm": rd,
                  "diameter_mm": 2 * rd, "height_mm": rd})
    spec = {"kind": KIND, "version": VERSION, "name": str(name), "units": "mm",
            "frame": "block B: origin at the centre of the reference face, x along the long "
                     "side, y across, z up out of the face; slab bottom at z = -slab_mm",
            "slab": {"L": L, "D": Dd, "T": float(slab_mm)}, "features": feats,
            "tolerances": {"height_mm": TOL_HEIGHT_MM, "diameter_mm": TOL_DIAMETER_MM,
                           "flatness_rms_max_mm": FLATNESS_MAX_MM},
            "place": "reference face toward the camera, long side horizontal, the TALLEST step "
                     "on the side AWAY from the projector (no step shadows the next)",
            "status": "mock-tested only, not yet used on a real rig"}
    return load(spec)


def load(spec: Union[str, Path, dict]) -> dict:
    if not isinstance(spec, dict):
        p = Path(spec)
        try:
            spec = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise ValueError(f"check block spec {p}: {e}") from None
    if spec.get("kind") != KIND:
        raise ValueError(f"not a check block spec (kind {spec.get('kind')!r}, want {KIND!r})")
    if int(spec.get("version", 0)) != VERSION:
        raise ValueError(f"check block spec version {spec.get('version')!r}: this engine reads "
                         f"version {VERSION}")
    kinds = [f.get("kind") for f in spec.get("features") or []]
    if "step" not in kinds or not spec.get("slab"):
        raise ValueError("a check block spec needs a slab and step features")
    return spec


def references(spec: dict) -> dict:
    """{row name: nominal mm} of every quantity the block checks."""
    out = {}
    for f in spec["features"]:
        if f["kind"] == "step":
            out[f"{f['name']}_height_mm"] = f["height_mm"]
        else:
            out[f"{f['name']}_diameter_mm"] = f["diameter_mm"]
            out[f"{f['name']}_height_mm"] = f["height_mm"]
    return out


def measured_template(spec: dict) -> dict:
    """The JSON the operator fills in with what the PRINTED block measures."""
    return {"kind": MEASURED_KIND, "block": spec["name"], "units": "mm",
            "measured_by": "", "date": "", "instrument": "",
            "values": {k: None for k in references(spec)},
            "nominal": references(spec),
            "how": {"stepN_height_mm": "caliper depth rod or height gauge: step top above the "
                                       "reference face next to it (3 places, average)",
                    "cylinder_diameter_mm": "calipers across the flats of the D profile",
                    "cylinder_height_mm": "its top line above the reference face",
                    "dome_height_mm": "its apex above the reference face",
                    "dome_diameter_mm": "optional; left null = 2 x dome_height_mm "
                                        "(a hemisphere)"},
            "note": "fill in `values` (null = not measured: that row is compared with the "
                    "nominal value and the result is a PULL, never a pass)"}


def load_measured(measured, spec: dict) -> dict:
    """The operator's caliper values -> {row: mm} (rows left null are absent)."""
    if measured is None:
        return {}
    if not isinstance(measured, dict):
        p = Path(measured)
        try:
            measured = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise ValueError(f"measured values {p}: {e}") from None
    if measured.get("kind", MEASURED_KIND) != MEASURED_KIND:
        raise ValueError(f"not a check block measurement (kind {measured.get('kind')!r})")
    vals = measured.get("values", measured)
    refs = references(spec)
    out = {}
    for k, v in vals.items():
        if k in ("kind", "block", "units") or v is None:
            continue
        if k not in refs:
            raise ValueError(f"measured value {k!r} is not a quantity of block "
                             f"{spec['name']} ({', '.join(refs)})")
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) \
                or not 0.5 * refs[k] < float(v) < 1.5 * refs[k]:
            raise ValueError(f"measured {k} = {v!r} is not near its nominal {refs[k]:g} mm")
        out[k] = float(v)
    return out


# ------------------------------------------------------------------ the model

def height(spec: dict, x, y) -> np.ndarray:
    """Top surface height above the reference face (mm) at block-frame (x, y)."""
    x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
    h = np.zeros(x.shape)
    for f in spec["features"]:
        if f["kind"] == "step":
            m = (x >= f["x"][0]) & (x <= f["x"][1]) & (y >= f["y"][0]) & (y <= f["y"][1])
            h = np.where(m, np.maximum(h, f["height_mm"]), h)
        elif f["kind"] == "cylinder":
            r, dx = f["radius_mm"], x - f["x_c"]
            m = (np.abs(dx) <= r) & (y >= f["y"][0]) & (y <= f["y"][1])
            top = f["pedestal_mm"] + np.sqrt(np.clip(r * r - dx * dx, 0, None))
            h = np.where(m, np.maximum(h, top), h)
        elif f["kind"] == "dome":
            rd = f["radius_mm"]
            q = np.hypot(x - f["centre"][0], y - f["centre"][1])
            h = np.maximum(h, np.sqrt(np.clip(rd * rd - q * q, 0, None)))
    return h


def grid(spec: dict, n_arc: int = 64) -> tuple:
    """Rectilinear (xs, ys) for a closed heightfield mesh: every vertical wall is an
    EPS_MM-wide pair of grid lines, curved surfaces are sampled n_arc times."""
    L, Dd = spec["slab"]["L"], spec["slab"]["D"]
    xs, ys = [-L / 2, L / 2], [-Dd / 2, Dd / 2]

    def wall(lst, v):
        lst += [v - EPS_MM, v + EPS_MM]

    th = np.linspace(-math.pi / 2, math.pi / 2, n_arc + 1)
    for f in spec["features"]:
        if f["kind"] == "step":
            for v in f["x"]:
                wall(xs, v)
            for v in f["y"]:
                wall(ys, v)
        elif f["kind"] == "cylinder":
            r = f["radius_mm"]
            xs += list(f["x_c"] + r * np.sin(th))
            xs += [f["x_c"] - r - EPS_MM, f["x_c"] + r + EPS_MM]
            for v in f["y"]:
                wall(ys, v)
        elif f["kind"] == "dome":
            rd = f["radius_mm"]
            xs += list(f["centre"][0] + rd * np.sin(th))
            ys += list(f["centre"][1] + rd * np.sin(th))
    xs = np.unique(np.clip(np.round(xs, 6), -L / 2, L / 2))
    ys = np.unique(np.clip(np.round(ys, 6), -Dd / 2, Dd / 2))
    return xs, ys


def model_points(spec: dict, step_mm: float = 0.5, max_slope: float = 2.5) -> tuple:
    """Nominal top surface sampled on a grid -> (points (N, 3), unit normals (N, 3)) in the
    block frame, near-vertical walls left out (ICP target)."""
    L, Dd = spec["slab"]["L"], spec["slab"]["D"]
    xs = np.arange(-L / 2, L / 2 + 1e-9, step_mm)
    ys = np.arange(-Dd / 2, Dd / 2 + 1e-9, step_mm)
    X, Y = np.meshgrid(xs, ys)
    Z = height(spec, X, Y)
    gy, gx = np.gradient(Z, step_mm)
    keep = np.hypot(gx, gy) < max_slope
    N = np.stack([-gx, -gy, np.ones_like(Z)], -1)
    N /= np.linalg.norm(N, axis=-1, keepdims=True)
    return np.stack([X, Y, Z], -1)[keep], N[keep]


# ------------------------------------------------------------------ registration

def register(P: np.ndarray, spec: dict, up=None) -> dict:
    """Scan points of the block (camera mm) -> {R, t (camera -> block frame), rms, fitness,
    ok}. The reference face fixes z; four in-plane turns start the ICP. up: camera
    direction the block's +x is closest to (default camera +x)."""
    P = np.asarray(P, np.float64)
    ref, inl = metric.ransac_plane(P, 0.4)
    n = ref.normal                                   # toward the camera = block +z
    xh = np.asarray(up if up is not None else (1.0, 0.0, 0.0), float)
    x0 = xh - (xh @ n) * n
    x0 /= np.linalg.norm(x0)
    c = ref.point + ((P.mean(0) - ref.point) - ((P.mean(0) - ref.point) @ n) * n)
    dst, dn = model_points(spec)
    best = None
    for k in range(4):
        a = k * math.pi / 2
        xk = math.cos(a) * x0 + math.sin(a) * np.cross(n, x0)
        R0 = np.stack([xk, np.cross(n, xk), n])      # rows: block axes in camera
        t0 = -R0 @ c
        icp = metric.icp_point_to_plane(P, dst, dn, max_dist_mm=6.0, R0=R0, t0=t0,
                                        max_iter=60)
        icp = metric.icp_point_to_plane(P, dst, dn, max_dist_mm=1.5, R0=icp.R, t0=icp.t,
                                        max_iter=40)
        score = (icp.fitness, -icp.rms_mm)
        if best is None or score > best[0]:
            best = (score, icp, k)
    _, icp, k = best
    ok = icp.fitness >= 0.7 and icp.rms_mm <= 0.5
    return {"R": icp.R, "t": icp.t, "rms_mm": icp.rms_mm, "fitness": icp.fitness,
            "turn_deg": 90 * k, "converged": icp.converged, "ok": bool(ok)}


# ------------------------------------------------------------------ measurement

def _plane_dist(fit: metric.PlaneFit, P: np.ndarray, up: np.ndarray) -> np.ndarray:
    n = fit.normal if fit.normal @ up >= 0 else -fit.normal
    return (P - fit.point) @ n


def measure(Q: np.ndarray, spec: dict) -> dict:
    """Block-frame points of one scan -> {row: (value mm, n points, extra)}; a row with too
    few points has value None."""
    L, Dd = spec["slab"]["L"], spec["slab"]["D"]
    x, y, z = Q[:, 0], Q[:, 1], Q[:, 2]
    inside = (np.abs(x) < L / 2 - 2.0) & (np.abs(y) < Dd / 2 - 2.0)
    bare = inside & (np.abs(z) < 1.5)
    for f in spec["features"]:
        if f["kind"] in ("step", "cylinder"):
            x0, x1 = (f["x"] if f["kind"] == "step" else
                      (f["x_c"] - f["radius_mm"], f["x_c"] + f["radius_mm"]))
            bare &= ~((x > x0 - 2.5) & (x < x1 + 2.5) & (y > f["y"][0] - 2.5)
                      & (y < f["y"][1] + 2.5))
        else:
            bare &= np.hypot(x - f["centre"][0], y - f["centre"][1]) > f["radius_mm"] + 2.5
    out = {}
    if bare.sum() < MIN_POINTS:
        out["reference_flatness_rms_mm"] = (None, int(bare.sum()), {})
        return out
    # gross outliers (an unwrapping error is a whole fringe off) are counted, not fitted
    ref, inl = metric.ransac_plane(Q[bare], OUTLIER_MM)
    up = np.array([0.0, 0.0, 1.0])
    d0 = _plane_dist(ref, Q[bare][inl], up)
    out_pct = 100.0 * float(1 - inl.mean())
    flat = ref.rms if out_pct <= MAX_OUTLIER_PCT else None
    out["reference_flatness_rms_mm"] = (flat, int(inl.sum()), {
        "peak_to_valley_mm": float(np.percentile(d0, 99.5) - np.percentile(d0, 0.5)),
        "outlier_pct": round(out_pct, 3)})
    for f in spec["features"]:
        nm = f["name"]
        if f["kind"] == "step":
            m = ((x > f["x"][0] + 1.5) & (x < f["x"][1] - 1.5) & (y > f["y"][0] + 1.5)
                 & (y < f["y"][1] - 1.5) & (np.abs(z - f["height_mm"]) < 2.0))
            d = _plane_dist(ref, Q[m], up)
            out[f"{nm}_height_mm"] = ((float(np.median(d)) if m.sum() >= MIN_POINTS else None),
                                      int(m.sum()), {"rms_mm": float(np.std(d)) if m.any()
                                                     else None})
        elif f["kind"] == "cylinder":
            r, p = f["radius_mm"], f["pedestal_mm"]
            m = ((np.abs(x - f["x_c"]) < 0.8 * r) & (y > f["y"][0] + 2.0)
                 & (y < f["y"][1] - 2.0) & (z > p + 0.3 * r) & (z < p + r + 2.0))
            if m.sum() < MIN_POINTS:
                out[f"{nm}_diameter_mm"] = (None, int(m.sum()), {})
                out[f"{nm}_height_mm"] = (None, int(m.sum()), {})
                continue
            cf = metric.fit_cylinder(Q[m], axis_hint=(0.0, 1.0, 0.0))
            top = float(_plane_dist(ref, cf.point[None], up)[0]) + cf.radius
            ext = {"fit_rms_mm": cf.rms, "axis": cf.axis.round(5).tolist()}
            out[f"{nm}_diameter_mm"] = (cf.diameter, int(m.sum()), ext)
            out[f"{nm}_height_mm"] = (top, int(m.sum()), ext)
        elif f["kind"] == "dome":
            rd = f["radius_mm"]
            m = (np.hypot(x - f["centre"][0], y - f["centre"][1]) < 0.85 * rd) & (z > 0.4 * rd)
            if m.sum() < MIN_POINTS:
                out[f"{nm}_diameter_mm"] = (None, int(m.sum()), {})
                out[f"{nm}_height_mm"] = (None, int(m.sum()), {})
                continue
            sf = metric.fit_sphere(Q[m])
            e = np.linalg.norm(Q[m] - sf.center, axis=1) - sf.radius
            mad = 1.4826 * float(np.median(np.abs(e - np.median(e))))
            keep = np.abs(e - np.median(e)) <= max(4 * mad, 0.02)
            if keep.sum() >= MIN_POINTS and not keep.all():
                sf = metric.fit_sphere(Q[m][keep])              # a trimmed refit
            apex = float(_plane_dist(ref, sf.center[None], up)[0]) + sf.radius
            ext = {"fit_rms_mm": sf.rms}
            out[f"{nm}_diameter_mm"] = (sf.diameter, int(m.sum()), ext)
            out[f"{nm}_height_mm"] = (apex, int(m.sum()), ext)
    return out


def block_points(cloud, region: Optional[np.ndarray] = None) -> np.ndarray:
    """The block's points: the largest blob >= 3 mm in front of the background plane."""
    fg, _ = metric.foreground(cloud, min_height_mm=3.0, region=region)
    return cloud.xyz[fg]


def analyze(clouds: list, spec: dict, measured=None, tol_height_mm: Optional[float] = None,
            tol_diameter_mm: Optional[float] = None,
            flatness_max_mm: Optional[float] = None, up=None) -> dict:
    """Metric clouds of the check block (one per repeat) -> the report: rows (scanned vs
    reference), verdict PASS / FAIL / INVALID and the disposition (CONTINUE = the rig
    measures the block within tolerance; REJECT = it does not: recalibrate; PULL = doubt)."""
    spec = load(spec)
    meas = load_measured(measured, spec)
    tols = spec.get("tolerances") or {}
    th = float(tol_height_mm if tol_height_mm is not None else tols.get("height_mm", TOL_HEIGHT_MM))
    td = float(tol_diameter_mm if tol_diameter_mm is not None
               else tols.get("diameter_mm", TOL_DIAMETER_MM))
    fm = float(flatness_max_mm if flatness_max_mm is not None
               else tols.get("flatness_rms_max_mm", FLATNESS_MAX_MM))
    nominal = references(spec)
    res = {"kind": "check_block_report", "block": spec["name"], "units": "mm",
           "n_scans": len(clouds), "registration": [], "rows": [], "pull_reasons": [],
           "tolerances": {"height_mm": th, "diameter_mm": td, "flatness_rms_max_mm": fm},
           "status": "mock-tested only, not yet used on a real rig"}
    per = []
    for i, cl in enumerate(clouds):
        P = block_points(cl) if hasattr(cl, "xyz") else np.asarray(cl, float)
        if len(P) < 5 * MIN_POINTS:
            res["registration"].append({"scan": i, "ok": False, "n_points": int(len(P))})
            continue
        reg = register(P, spec, up)
        res["registration"].append({"scan": i, "ok": reg["ok"], "rms_mm": reg["rms_mm"],
                                    "fitness": reg["fitness"], "turn_deg": reg["turn_deg"],
                                    "n_points": int(len(P))})
        if reg["ok"]:
            per.append(measure(P @ reg["R"].T + reg["t"], spec))
    if not per:
        res.update(result="INVALID", invalid_reason="the check block could not be registered "
                   "in any scan (not in view, too few points, or not the block of this spec)")
        D.add(res, "incomplete", res["invalid_reason"])
        D.settle(res)
        return res
    if len(per) < len(clouds):                       # a repeat is never silently dropped
        D.add(res, "incomplete", f"the check block could not be registered in "
              f"{len(clouds) - len(per)} of {len(clouds)} scans")
    tolerances = []
    for name in ["reference_flatness_rms_mm"] + list(nominal):
        vals = [p[name][0] for p in per if name in p and p[name][0] is not None]
        npts = [p[name][1] for p in per if name in p]
        row = {"name": name, "n_points": npts}
        if name == "reference_flatness_rms_mm":
            row.update(limit_mm=fm, reference_mm=None, reference_source="limit")
            pv = [p[name][2].get("peak_to_valley_mm") for p in per if name in p and p[name][2]]
            oc = [p[name][2].get("outlier_pct") for p in per if name in p and p[name][2]]
            row["peak_to_valley_mm"] = max(pv) if pv else None
            row["outlier_pct"] = max(oc) if oc else None
        else:
            ref, src = _reference(name, meas, nominal)
            tol = td if name.endswith("_diameter_mm") else th
            row.update(nominal_mm=nominal[name], reference_mm=ref, reference_source=src,
                       tol_mm=tol)
        if len(vals) < len(per):
            row.update(scanned_mm=float(np.mean(vals)) if vals else None, verdict="INVALID")
            outl = [p[name][2].get("outlier_pct") for p in per if name in p and p[name][2]]
            if outl and max(outl) > MAX_OUTLIER_PCT:
                D.add(res, "invalid", f"{name}: {max(outl):.1f} % of the reference face is "
                      f"more than {OUTLIER_MM:g} mm off its plane (> {MAX_OUTLIER_PCT:g} %): "
                      "decode errors or a warped print - not measured", check=name)
            else:
                D.add(res, "too_few_points", f"{name}: fewer than {MIN_POINTS} points in "
                      f"{len(per) - len(vals)} of {len(per)} scans", check=name)
        else:
            row["scanned_mm"] = float(np.mean(vals))
            row["scanned_sigma_mm"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else None
            if name == "reference_flatness_rms_mm":
                worst = max(vals)
                row["verdict"] = "PASS" if worst <= fm else "FAIL"
                tolerances.append(D.tolerance(name, worst, None, fm))
            else:
                errs = [v - row["reference_mm"] for v in vals]
                worst = max(errs, key=abs)
                row["error_mm"] = float(worst)
                row["verdict"] = "PASS" if abs(worst) <= row["tol_mm"] else "FAIL"
                tolerances.append(D.tolerance(name.replace("_mm", "_error_mm"), worst,
                                              -row["tol_mm"], row["tol_mm"]))
        res["rows"].append(row)
    verdicts = {r["verdict"] for r in res["rows"]}
    res["result"] = "FAIL" if "FAIL" in verdicts else ("INVALID" if "INVALID" in verdicts
                                                      else "PASS")
    if res["result"] == "INVALID":
        res["invalid_reason"] = "a feature of the block was not measured in every scan"
    unmeasured = [r["name"] for r in res["rows"] if r.get("reference_source") == "nominal"]
    if unmeasured:
        D.add(res, "invalid", f"compared with NOMINAL values ({', '.join(unmeasured)}): measure "
              "the printed block (calipers / height gauge) and pass --measured; a print is never "
              "exactly its model")
    D.settle(res, tolerances=tolerances)
    return res


def _reference(name: str, meas: dict, nominal: dict) -> tuple:
    if name in meas:
        return meas[name], "measured"
    if name.endswith("_diameter_mm"):
        h = name.replace("_diameter_mm", "_height_mm")
        if name.startswith("dome") and h in meas:
            return 2 * meas[h], "2 x measured height (hemisphere)"
    return nominal[name], "nominal"


def verdict_text(res: dict) -> str:
    d = res.get("disposition")
    return {"CONTINUE": "CONTINUE: the rig measures the block within tolerance",
            "REJECT": "REJECT: the rig measures the block OUT of tolerance - recalibrate"}.get(
        d, D.summary(res))

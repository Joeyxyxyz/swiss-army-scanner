"""
Station verbs behind engine/api.py: station_measure, station_golden, station_check.

Library (append-only, rule 7; meta.json written LAST = the completeness marker):
  station_golden/<geometry_id>/<ts>_<label>/  golden.json (nominal values, zone model +
                                              numbering, limits, station parameters, the
                                              measurements it came from), preview.png, meta.json
Output bin: measurements/ (station_measure: <label>_station .json / .csv / .png),
defects/ (station_check: <label>_station_check .json / .csv / .png).

Rules: mm only from the stereo calibration the metric scan was triangulated with, still the
library's current one for its geometry (rule 6); a golden is bound to the geometry +
fingerprint and that calibration and is only used with scans that match (rule 5). Nothing
here touches the rig.
"""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from ... import rules
from ...library.store import json_safe, read_meta, safe_label
from ...results import MetricScan, StationCheck, StationGolden, StationMeasurement
from . import figure
from . import geometry as geo_mod

UNITS = {"centre_mm": "mm, camera frame (x right, y down, z forward)",
         "height_mm": "mm along the common face normal; + = toward the camera (proud), "
                      "- = behind the mean cell-face plane",
         "spacing_mm": "mm, face centres in the face plane", "camera_px": "camera px (full frame)",
         "angles": "deg", "s_mm": "mm along the cooling tube's run (e1, ~ image right) from the "
                                  "mean face centre"}
DEFAULT_LIMITS = {"cell_height_tol_mm": 0.15, "cell_height_band_mm": None,
                  "ct_zone_tol_mm": 0.30, "ct_zone_band_mm": None,
                  "spacing_tol_mm": None, "bands": {}}
KIND = "station_golden"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _write_bytes(path: Path, data: Optional[bytes]) -> Optional[Path]:
    if not data:
        return None
    rules.ensure_new_path(path)
    with open(path, "xb") as f:
        f.write(data)
    return path


def _write_text(path: Path, text: str) -> Path:
    rules.ensure_new_path(path)
    with open(path, "x", encoding="utf-8", newline="\n") as f:
        f.write(text)
    return path


def _r(v, k: int = 4):
    if v is None:
        return None
    if isinstance(v, (list, tuple, np.ndarray)):
        return [_r(x, k) for x in (v.tolist() if isinstance(v, np.ndarray) else v)]
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return round(float(v), k) if np.isfinite(v) else None
    return v


# ------------------------------------------------------------ inputs

def _scan_and_cal(s, scan) -> tuple:
    from ..endface.verbs import _scans_and_cal
    [ms], cal, geo, fp = _scans_and_cal(s, [scan], "station_measure")      # rules 5 + 6
    return ms, cal, geo, fp


def _load_flood_file(p: Path) -> np.ndarray:
    if p.suffix.lower() == ".npz":
        with np.load(p) as z:
            if "white" in z.files and "black" in z.files:
                return np.stack([z["white"], z["black"]])
            if "image" in z.files:
                return z["image"]
            raise geo_mod.StationError(f"{p} holds no flood (white + black, or image)")
    return np.load(p)


def _scan_flood(ms: MetricScan) -> Optional[np.ndarray]:
    f = Path(ms.path) / "flood.npz"
    return _load_flood_file(f) if f.is_file() else None


def resolve_flood(s, ms: MetricScan, flood, geo: str, fp) -> tuple:
    """-> (flood difference image 0..255, source spec, warnings). flood: None (the scan's own
    flood pair, else its decode modulation), a (2, H, W) / (H, W) array, a .npy / .npz file,
    or another metric scan (id / handle) whose flood pair is used."""
    from ..endface.verbs import as_metric_scan
    warnings = []
    if flood is None:
        a = _scan_flood(ms)
        if a is not None:
            return geo_mod.flood_diff(a), "scan", warnings
        warnings.append("no flood pair stored with this scan: the cells were found on the "
                        "decode's fringe modulation (pass flood=, or scan with scan_fast, which "
                        "keeps its flood pair)")
        return geo_mod.conf_flood(ms.field.conf), "decode", warnings
    if isinstance(flood, np.ndarray):
        return geo_mod.flood_diff(flood), "array", warnings
    if isinstance(flood, (str, Path)) and Path(str(flood)).is_file():
        p = Path(str(flood)).resolve()
        return geo_mod.flood_diff(_load_flood_file(p)), f"file:{p}", warnings
    other = as_metric_scan(s, flood)
    rules.require_same_geometry(other.geometry_id, geo, "station flood (metric scan)")
    rules.require_same_fingerprint((other.meta or {}).get("geometry_fp"), fp,
                                   "station flood (metric scan)")
    a = _scan_flood(other)
    if a is None:
        raise ValueError(f"metric scan {other.id} has no flood pair (flood.npz): only scan_fast "
                         "stores one")
    if other.id != ms.id:
        warnings.append(f"flood pair from another scan ({other.id}): the cells must not have moved")
    return geo_mod.flood_diff(a), f"scan:{other.id}", warnings


def _flood_arg_from_source(src: str):
    """The flood argument that re-creates a measurement's flood (None = the scan's own)."""
    if src in ("scan", "decode"):
        return None
    if src.startswith("file:"):
        return src[5:]
    if src.startswith("scan:"):
        return src[5:]
    raise ValueError("this measurement's flood was given as an array: pass flood= again")


# ------------------------------------------------------------ station_measure

def _report(r: dict, ms: MetricScan, cal, geo, fp, prm, fsrc, label, warnings) -> dict:
    cells = []
    for c, f in zip(r["cells"], r["faces"]):
        cells.append({"name": c["name"], "camera_px": _r(c["px"], 2), "r_px": _r(c["r_px"], 2),
                      "hough_px": _r(c["hough_px"], 1), "centre_mm": _r(f["centre"], 4),
                      "normal": _r(f["normal"], 5), "height_mm": _r(f["height_mm"], 4),
                      "flat_rms_mm": _r(f["flat_rms_mm"], 4), "flat_p2_p98_mm": _r(f["flat_p2_p98_mm"], 4),
                      "coverage": _r(f["coverage"], 3), "n_pts": f["n_pts"],
                      "tilt_vs_common_deg": _r(f["tilt_vs_common_deg"], 3),
                      "centre_method": f["centre_method"],
                      "outline_diameter_mm": _r((f["outline"] or {}).get("diameter_mm"), 3),
                      "outline_rms_mm": _r((f["outline"] or {}).get("rms_mm"), 3)})
    pairs = [{k: _r(v, 4) for k, v in p.items()} for p in r["pairs"]]
    fr = r["frame"]
    zones = [{k: _r(v, 4) for k, v in z.items()} for z in r["zones"]]
    ct = json.loads(json.dumps(json_safe(r["ct"]))) if r["ct"] else None
    if ct:
        for k in ("edge_height_all", "edge_height_zones", "trap_fit", "zone_model"):
            if isinstance(ct.get(k), dict):
                ct[k] = {kk: _r(vv, 4) for kk, vv in ct[k].items()}
        if ct.get("band_mm"):
            ct["band_mm"] = _r(ct["band_mm"], 3)
    return {"kind": "station_measurement", "label": label, "scan_id": ms.id,
            "scan_label": ms.label, "geometry_id": geo, "geometry_fp": fp,
            "calibration_id": cal.id, "created": _now(), "flood_source": fsrc,
            "params": prm.to_dict(), "cells": cells, "pairs": pairs,
            "frame": {"origin_mm": _r(fr["O"], 4), "n": _r(fr["n"], 6), "e1": _r(fr["e1"], 6),
                      "e2": _r(fr["e2"], 6)},
            "ct": ct, "zones": zones, "units": UNITS, "mock": False, "warnings": warnings}


def _csv_measure(rep: dict) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["item", "quantity", "value", "units", "p5", "p95", "n_px", "camera_u", "camera_v"])
    for c in rep["cells"]:
        w.writerow([c["name"], "face height vs the mean face plane", c["height_mm"], "mm", "", "",
                    c["n_pts"], *c["camera_px"]])
        w.writerow([c["name"], "face flatness rms", c["flat_rms_mm"], "mm", "", "", "", "", ""])
    for p in rep["pairs"]:
        w.writerow([p["pair"], "face height difference along the common normal",
                    p["height_diff_mm"], "mm", "", "", "", "", ""])
        w.writerow([p["pair"], "centre spacing in the face plane", p["spacing_inplane_mm"], "mm",
                    "", "", "", "", ""])
    ct = rep.get("ct") or {}
    if ct.get("edge_height_all"):
        e = ct["edge_height_all"]
        w.writerow(["CT", "edge height vs the mean face plane (all edge px)", e["median_mm"], "mm",
                    e["p5_mm"], e["p95_mm"], e["n_px"], "", ""])
    if ct.get("trap_fit"):
        t = ct["trap_fit"]
        for k in ("pitch_mm", "amp_mm", "phase_mm", "plateau_frac", "fit_rms_mm"):
            w.writerow(["CT wave", k, t[k], "" if k == "plateau_frac" else "mm", "", "", "", "", ""])
    for z in rep["zones"]:
        if not z.get("included"):
            continue
        uv = z.get("cam_uv") or ["", ""]
        w.writerow([z["label"], f"CT edge height, zone k={z['k']} (median)", z.get("median_mm"),
                    "mm", z.get("p5_mm"), z.get("p95_mm"), z["n_px"], *uv])
    return buf.getvalue()


def _zone_image(X, mask, fr, model) -> Optional[np.ndarray]:
    if model is None or mask is None:
        return None
    k = np.full(mask.shape, -10 ** 6, np.int64)
    v, u = np.nonzero(mask)
    P = X[v, u]
    ok = np.all(np.isfinite(P), 1)
    k[v[ok], u[ok]] = geo_mod.zone_index((P[ok] - fr["O"]) @ fr["e1"], model["pitch_mm"],
                                         model["phase_mm"])
    return k


def station_measure(s, scan, flood=None, label: Optional[str] = None,
                    params: Optional[dict] = None, zone_model: Optional[dict] = None,
                    zone_labels: Optional[dict] = None, write: bool = True) -> tuple:
    """-> (StationMeasurement, internals {X, d, r}). zone_model / zone_labels: a golden's."""
    prm = geo_mod.StationParams.from_dict(params)
    ms, cal, geo, fp = _scan_and_cal(s, scan)
    label = safe_label(label or f"{ms.label}_station")
    d, fsrc, warnings = resolve_flood(s, ms, flood, geo, fp)
    cloud = ms.cloud
    X = np.where(cloud.mask[..., None], cloud.xyz, np.nan).astype(np.float64)
    r = geo_mod.measure(X, d, cal.K_cam, cal.dist_cam, prm, zone_model,
                        {int(k): v for k, v in zone_labels.items()} if zone_labels else None)
    warnings += r["warnings"]
    rep = _report(r, ms, cal, geo, fp, prm, fsrc, label, warnings)
    rep["mock"] = bool(s.mock)
    files = {}
    if write:
        files = _write_measure(s, rep, r, X, d, ms)
        rep["files"] = {k: str(v) for k, v in files.items() if v}
    m = StationMeasurement(label=label, scan_id=ms.id, geometry_id=geo, calibration_id=cal.id,
                           cells=rep["cells"], pairs=rep["pairs"], ct=rep["ct"], zones=rep["zones"],
                           flood_source=fsrc, params=rep["params"], frame=rep["frame"],
                           units=UNITS, files=files, meta=rep, warnings=warnings)
    return m, {"X": X, "d": d, "r": r, "ms": ms}


def _write_measure(s, rep, r, X, d, ms) -> dict:
    out = s.store.output_path("measurements", f"{rep['label']}_station", ".json")
    s.store.write_json(out, json_safe(rep))
    s.store.record_output("measurements", out, label=rep["label"], geometry_id=rep["geometry_id"],
                          source=ms.path)
    csv_p = _write_text(out.with_suffix(".csv"), _csv_measure(rep))
    s.store.record_output("measurements", csv_p, label=rep["label"], geometry_id=rep["geometry_id"],
                          source=ms.path)
    zmodel = (rep.get("ct") or {}).get("zone_model")
    png = figure.station_png(rep, d, r["arrays"].get("ct_mask"),
                             _zone_image(X, r["arrays"].get("ct_mask"), r["frame"], zmodel),
                             title=f"station {rep['label']} - scan {rep['scan_id']}")
    png_p = _write_bytes(out.with_suffix(".png"), png)
    if png_p:
        s.store.record_output("measurements", png_p, label=rep["label"],
                              geometry_id=rep["geometry_id"], source=ms.path)
    return {"json": out, "csv": csv_p, "png": png_p}


def load_measurement(s, item) -> StationMeasurement:
    """A StationMeasurement, or its JSON file in the output bin."""
    if isinstance(item, StationMeasurement):
        return item
    p = Path(str(item))
    if p.suffix.lower() == ".json" and p.is_file():
        rep = json.loads(p.read_text(encoding="utf-8"))
        if rep.get("kind") != "station_measurement":
            raise ValueError(f"{p} is not a station measurement")
        return StationMeasurement(label=rep["label"], scan_id=rep["scan_id"],
                                  geometry_id=rep["geometry_id"], calibration_id=rep["calibration_id"],
                                  cells=rep["cells"], pairs=rep["pairs"], ct=rep.get("ct"),
                                  zones=rep.get("zones") or [], flood_source=rep["flood_source"],
                                  params=rep.get("params") or {}, frame=rep.get("frame") or {},
                                  units=rep.get("units") or UNITS, files={"json": p}, meta=rep,
                                  warnings=list(rep.get("warnings") or []))
    raise ValueError(f"{item!r} is not a station measurement (a StationMeasurement or its .json)")


# ------------------------------------------------------------ station_golden

def _golden_handle(d: Path) -> Optional[StationGolden]:
    m = read_meta(d)
    if m is None or m.get("kind") != KIND:
        return None
    try:
        g = json.loads((Path(d) / "golden.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return StationGolden(id=m["id"], label=m.get("label", d.name), geometry_id=m["geometry_id"],
                         path=Path(d), created=m.get("created", ""),
                         calibration_id=m.get("calibration_id", ""),
                         n_measurements=int(m.get("n_measurements", 0)), limits=g["limits"],
                         nominal=g["nominal"], zone_model=g.get("zone_model"),
                         params=g.get("params") or {}, units=g.get("units") or UNITS,
                         files={"json": Path(d) / "golden.json", "png": Path(d) / "preview.png"},
                         meta=m, warnings=list(m.get("warnings") or []))


def list_goldens(store, geometry_id: Optional[str] = None) -> list:
    geos = [geometry_id] if geometry_id else store.geometry_ids(KIND)
    out = []
    for g in geos:
        for d in store.entries(KIND, g):
            h = _golden_handle(d)
            if h is not None:
                out.append(h)
    return out


def _band(v, what: str):
    if v is None:
        return None
    if not isinstance(v, (list, tuple)) or len(v) != 2:
        raise ValueError(f"{what} must be [lo, hi] mm")
    lo, hi = float(v[0]), float(v[1])
    if not (np.isfinite(lo) and np.isfinite(hi)) or lo >= hi:
        raise ValueError(f"{what} must be [lo, hi] mm with lo < hi, got {v!r}")
    return [lo, hi]


def _tol(v, what: str):
    if v is None:
        return None
    v = float(v)
    if not np.isfinite(v) or v <= 0:
        raise ValueError(f"{what} must be > 0 mm, got {v!r}")
    return v


def check_limits(limits: Optional[dict]) -> dict:
    lim = dict(DEFAULT_LIMITS)
    bad = set(limits or {}) - set(DEFAULT_LIMITS)
    if bad:
        raise ValueError(f"unknown station limits {sorted(bad)}; known: {sorted(DEFAULT_LIMITS)}")
    lim.update({k: v for k, v in (limits or {}).items()})
    out = {"cell_height_tol_mm": _tol(lim["cell_height_tol_mm"], "cell_height_tol_mm"),
           "cell_height_band_mm": _band(lim["cell_height_band_mm"], "cell_height_band_mm"),
           "ct_zone_tol_mm": _tol(lim["ct_zone_tol_mm"], "ct_zone_tol_mm"),
           "ct_zone_band_mm": _band(lim["ct_zone_band_mm"], "ct_zone_band_mm"),
           "spacing_tol_mm": _tol(lim["spacing_tol_mm"], "spacing_tol_mm"),
           "bands": {str(k): _band(v, f"bands[{k!r}]") for k, v in (lim["bands"] or {}).items()}}
    if out["cell_height_tol_mm"] is None and out["cell_height_band_mm"] is None:
        raise ValueError("set cell_height_tol_mm or cell_height_band_mm (or both)")
    if out["ct_zone_tol_mm"] is None and out["ct_zone_band_mm"] is None:
        raise ValueError("set ct_zone_tol_mm or ct_zone_band_mm (or both)")
    return out


def item_band(kind: str, item: str, nominal, lim: dict) -> Optional[list]:
    """The band [lo, hi] mm an item is checked against: its own absolute band (bands[item])
    decides alone; else nominal +- the kind's tolerance, cut to the kind's absolute band.
    None = not checked (spacing without spacing_tol_mm / a band)."""
    b = (lim.get("bands") or {}).get(item)
    if b is not None:
        return [float(b[0]), float(b[1])]
    if kind == "spacing":
        t = lim.get("spacing_tol_mm")
        return None if t is None or nominal is None else [nominal - t, nominal + t]
    tol_k, band_k = {"cell_height": ("cell_height_tol_mm", "cell_height_band_mm"),
                     "ct_zone": ("ct_zone_tol_mm", "ct_zone_band_mm")}[kind]
    lo, hi = -np.inf, np.inf
    if lim.get(tol_k) is not None and nominal is not None:
        lo, hi = nominal - lim[tol_k], nominal + lim[tol_k]
    if lim.get(band_k) is not None:
        lo, hi = max(lo, lim[band_k][0]), min(hi, lim[band_k][1])
    return [lo, hi]


def _golden_items(pairs: list, zones: list) -> list:
    """(kind, item, nominal) of every item a golden checks."""
    out = []
    for p in pairs:
        out.append(("cell_height", p["pair"], p["height_diff_mm"]))
        out.append(("spacing", f"{p['pair']} spacing", p["spacing_inplane_mm"]))
    out += [("ct_zone", z["label"], z["median_mm"]) for z in zones]
    return out


def _check_golden_limits(lim: dict, items: list, warnings: list) -> None:
    """A per-item band must name an item of this golden (a typo would silently fall back to
    the tolerance), and every item's band must be non-empty (a tolerance and an absolute band
    that do not overlap would FAIL every part); a nominal outside its own band is warned."""
    names = [it for _, it, _ in items]
    bad = sorted(set(lim.get("bands") or {}) - set(names))
    if bad:
        raise ValueError(f"station_golden: bands for unknown item(s) {bad}; this golden's items: "
                         f"{', '.join(names)}")
    for kind, it, nom in items:
        b = item_band(kind, it, nom, lim)
        if b is None:
            continue
        if b[0] > b[1]:
            raise ValueError(f"station_golden: {it}: the tolerance around its nominal "
                             f"({nom:+.4f} mm) and the absolute band do not overlap - every part "
                             "would FAIL; widen or drop one of them")
        if nom is not None and not b[0] <= nom <= b[1]:
            warnings.append(f"{it}: the nominal {nom:+.4f} mm (the GOOD measurements' mean) is "
                            f"outside its band [{b[0]:.4f}, {b[1]:.4f}] mm")


def _plain(d) -> dict:
    """Parameters as JSON would store them (tuples -> lists, numpy -> Python): a measurement
    read back from its .json compares equal to one made in memory with the same settings."""
    return json.loads(json.dumps(json_safe(d or {})))


def _match_zones(ref: list, other: list, pitch: float) -> dict:
    """ref zone k -> the other measurement's zone nearest in s (within a quarter pitch): a
    phase that wrapped by a pitch between the two fits renumbers k, not the part."""
    out = {}
    for z in ref:
        best = min(other, key=lambda o: abs(o["s_centre_mm"] - z["s_centre_mm"]), default=None)
        if best is not None and abs(best["s_centre_mm"] - z["s_centre_mm"]) < pitch / 4:
            out[z["k"]] = best
    return out


def station_golden(s, measurements: list, label: str = "golden", limits: Optional[dict] = None,
                   params: Optional[dict] = None, flood=None) -> StationGolden:
    """Golden from >= 1 GOOD station measurements (StationMeasurements, their .json, or metric
    scans - measured here with `params` and their own flood, or `flood` for a single scan):
    nominal = the mean; the first measurement's zone model (pitch + phase vs the cell-pair
    centre) and numbering become the golden's."""
    safe_label(label)
    lim = check_limits(limits)
    if not measurements:
        raise ValueError("station_golden needs at least one GOOD station measurement")
    is_meas = [isinstance(it, StationMeasurement) or (isinstance(it, (str, Path))
                                                      and str(it).lower().endswith(".json"))
               for it in measurements]
    if flood is not None and (len(measurements) != 1 or is_meas[0]):
        raise ValueError("station_golden: flood= applies to a single metric scan only; measure "
                         "each scan with station_measure(scan, flood) and pass the measurements")
    if params is not None and all(is_meas):
        raise ValueError("station_golden: params= applies to metric scans measured here; station "
                         "measurements keep the parameters they were measured with")
    ms_list = []
    for i, it in enumerate(measurements):
        if is_meas[i]:
            ms_list.append(load_measurement(s, it))
        else:
            if isinstance(it, (str, Path, MetricScan)):
                m, _ = station_measure(s, it, flood, params=params)
                ms_list.append(m)
            else:
                raise ValueError(f"measurement {i + 1}: {it!r} is not a station measurement or a "
                                 "metric scan")
    ref = ms_list[0]
    for m in ms_list[1:]:
        rules.require_same_geometry(m.geometry_id, ref.geometry_id, "station_golden")
        rules.require_same_fingerprint(m.meta.get("geometry_fp"), ref.meta.get("geometry_fp"),
                                       "station_golden")
        if m.calibration_id != ref.calibration_id:
            raise rules.GeometryMismatch("station_golden: measurements under different stereo "
                                         f"calibrations ({ref.calibration_id} vs {m.calibration_id})")
        if len(m.cells) != len(ref.cells):
            raise ValueError("station_golden: the measurements have different cell counts")
        if _plain(m.params) != _plain(ref.params):
            raise ValueError("station_golden: the measurements were made with different station "
                             "parameters; measure them alike")
    from ..endface.verbs import _require_current_cal
    fp = ref.meta.get("geometry_fp")
    _require_current_cal(s, ref.geometry_id, fp, ref.calibration_id, "station_golden")  # rule 6
    # the measurements' own warnings (e.g. cells found on the decode modulation, a borrowed
    # flood) stay visible on the golden
    warnings = [f"measurement {m.label}: {w}" for m in ms_list for w in (m.warnings or [])]
    # cell pairs
    pairs = []
    for p in ref.pairs:
        vals = [next((q for q in m.pairs if q["pair"] == p["pair"]), None) for m in ms_list]
        hv = [q["height_diff_mm"] for q in vals if q]
        sv = [q["spacing_inplane_mm"] for q in vals if q]
        pairs.append({"pair": p["pair"], "height_diff_mm": _r(np.mean(hv), 4),
                      "height_diff_spread_mm": _r(np.ptp(hv), 4),
                      "spacing_inplane_mm": _r(np.mean(sv), 4), "spacing_spread_mm": _r(np.ptp(sv), 4),
                      "n": len(hv)})
    # zones (the reference's included zones; matched by position along the run)
    zmodel = (ref.ct or {}).get("zone_model")
    zones = []
    if zmodel:
        rz = [z for z in ref.zones if z.get("included") and z.get("median_mm") is not None]
        matches = [_match_zones(rz, [z for z in m.zones if z.get("median_mm") is not None],
                                zmodel["pitch_mm"]) for m in ms_list]
        for z in rz:
            v = [mt[z["k"]]["median_mm"] for mt in matches if z["k"] in mt]
            zones.append({"label": z["label"], "k": z["k"], "s_centre_mm": z["s_centre_mm"],
                          "s_lo_mm": z["s_lo_mm"], "s_hi_mm": z["s_hi_mm"],
                          "cam_uv": z.get("cam_uv"), "median_mm": _r(np.mean(v), 4),
                          "spread_mm": _r(np.ptp(v), 4), "n": len(v)})
            if len(v) < len(ms_list):
                warnings.append(f"zone {z['label']} measured in {len(v)} of {len(ms_list)} "
                                "measurements")
    if not zones:
        warnings.append("no cooling-tube zones in the reference measurement: the golden checks "
                        "the cells only")
    _check_golden_limits(lim, _golden_items(pairs, zones), warnings)
    nominal = {"cells": [{"name": c["name"], "height_mm": c["height_mm"]} for c in ref.cells],
               "pairs": pairs, "zones": zones,
               "ct_edge_height_mm": ((ref.ct or {}).get("edge_height_zones") or {}).get("median_mm")}
    g = {"kind": KIND, "limits": lim, "nominal": nominal, "zone_model": zmodel,
         "zone_labels": {str(z["k"]): z["label"] for z in zones}, "params": ref.params,
         "units": UNITS, "measurements": [{"label": m.label, "scan_id": m.scan_id,
                                           "flood_source": m.flood_source,
                                           "file": str(m.files.get("json") or "")} for m in ms_list]}
    d = s.store.new_dir(KIND, ref.geometry_id, label=label)
    with s.store.filling(d):
        s.store.write_json(d / "golden.json", json_safe(g))
        png = ref.files.get("png")
        if png and Path(png).is_file():
            _write_bytes(d / "preview.png", Path(png).read_bytes())
        meta = {"kind": KIND, "id": d.name, "label": label, "geometry_id": ref.geometry_id,
                "geometry_fp": fp, "calibration_id": ref.calibration_id, "created": _now(),
                "n_measurements": len(ms_list), "n_cells": len(ref.cells),
                "face_diameter_mm": (ref.params or {}).get("face_diameter_mm"),
                "n_zones": len(zones), "mock": bool(s.mock), "warnings": warnings}
        s.store.write_json(d / "meta.json", json_safe(meta))
    return _golden_handle(d)


def as_golden(s, golden, geometry_id: str, fp, calibration_id: str) -> StationGolden:
    """A StationGolden, an id / label / folder, or None = the newest golden at this geometry +
    fingerprint made under this stereo calibration."""
    if isinstance(golden, StationGolden):
        return golden
    if golden is None:
        cands = [x for x in list_goldens(s.store, geometry_id)
                 if not rules.fingerprint_diffs(x.meta.get("geometry_fp"), fp)
                 and x.calibration_id == calibration_id]
        if not cands:
            raise ValueError(f"no station golden at geometry {geometry_id} for stereo calibration "
                             f"{calibration_id}: make one with station_golden from good "
                             "measurements")
        return cands[-1]
    p = Path(str(golden))
    g = _golden_handle(p) if p.is_dir() else None
    if g is None:
        hits = [x for x in list_goldens(s.store) if x.id == str(golden) or x.label == str(golden)]
        if not hits:
            raise ValueError(f"no station golden {golden!r} in the library "
                             "(see list_library('station_golden'))")
        g = hits[-1]
    return g


# ------------------------------------------------------------ station_check

def _item(item, kind, value, nominal, band, extra=None) -> dict:
    it = {"item": item, "kind": kind, "value_mm": _r(value, 4), "nominal_mm": _r(nominal, 4),
          "band_mm": _r(band, 4) if band else None, **(extra or {})}
    if value is None:
        it["status"] = "missing"
    elif band is None:
        it["status"] = "info"
    else:
        it["status"] = "ok" if band[0] <= value <= band[1] else "trigger"
        it["dev_mm"] = _r(value - nominal, 4) if nominal is not None else None
    return it


def compare(m: StationMeasurement, g: StationGolden) -> list:
    lim = g.limits
    items = []
    for p in g.nominal["pairs"]:
        q = next((x for x in m.pairs if x["pair"] == p["pair"]), None)
        v = q["height_diff_mm"] if q else None
        nom = p["height_diff_mm"]
        items.append(_item(p["pair"], "cell_height", v, nom,
                           item_band("cell_height", p["pair"], nom, lim)))
        sv = q["spacing_inplane_mm"] if q else None
        sname = f"{p['pair']} spacing"
        items.append(_item(sname, "spacing", sv, p["spacing_inplane_mm"],
                           item_band("spacing", sname, p["spacing_inplane_mm"], lim)))
    by_k = {z["k"]: z for z in m.zones}
    mprm = geo_mod.StationParams.from_dict(g.params)
    for z in g.nominal["zones"]:
        mz = by_k.get(z["k"])
        v = None
        extra = {"k": z["k"]}
        if mz is not None:
            extra.update({"coverage": mz.get("coverage"), "n_px": mz.get("n_px")})
            if mz.get("median_mm") is not None and (mz.get("coverage") or 0) >= mprm.zone_min_coverage:
                v = mz["median_mm"]
        nom = z["median_mm"]
        items.append(_item(z["label"], "ct_zone", v, nom, item_band("ct_zone", z["label"], nom, lim),
                           extra))
    return items


def _csv_check(items: list) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["item", "kind", "value_mm", "nominal_mm", "band_lo_mm", "band_hi_mm", "dev_mm",
                "status"])
    for it in items:
        b = it.get("band_mm") or [None, None]
        w.writerow([it["item"], it["kind"], it["value_mm"], it["nominal_mm"], b[0], b[1],
                    it.get("dev_mm"), it["status"]])
    return buf.getvalue()


def station_check(s, scan, golden=None, flood=None, label: Optional[str] = None) -> StationCheck:
    """A scan (or a StationMeasurement: its scan and flood) vs a station golden: re-measured
    with the golden's parameters and zone model -> PASS / FAIL / INCOMPLETE."""
    if isinstance(scan, StationMeasurement) or (isinstance(scan, (str, Path))
                                                and str(scan).lower().endswith(".json")):
        prev = load_measurement(s, scan)
        if flood is None:
            flood = _flood_arg_from_source(prev.flood_source)
        scan = prev.scan_id
    from ..endface.verbs import as_metric_scan
    ms0 = as_metric_scan(s, scan)
    g = as_golden(s, golden, ms0.geometry_id, (ms0.meta or {}).get("geometry_fp"),
                  ms0.calibration_id)
    rules.require_same_geometry(ms0.geometry_id, g.geometry_id, f"station_check: golden {g.id}")
    rules.require_same_fingerprint((ms0.meta or {}).get("geometry_fp"), g.meta.get("geometry_fp"),
                                   f"station_check: golden {g.id}")
    from ..endface.verbs import _require_current_cal
    _require_current_cal(s, g.geometry_id, g.meta.get("geometry_fp"), g.calibration_id,
                         f"station_check: golden {g.id}")                          # rule 6
    if ms0.calibration_id != g.calibration_id:
        raise rules.GeometryMismatch(
            f"station_check: scan {ms0.id} was triangulated with {ms0.calibration_id}, the golden "
            f"{g.id} with {g.calibration_id}: re-triangulate the scan (triangulate)")
    label = safe_label(label or f"{ms0.label}_check")
    labels = {int(z["k"]): z["label"] for z in g.nominal.get("zones") or []}
    m, inner = station_measure(s, ms0, flood, label, params=g.params, zone_model=g.zone_model,
                               zone_labels=labels or None, write=True)
    items = compare(m, g)
    trig = [it for it in items if it["status"] == "trigger"]
    missing = [it for it in items if it["status"] == "missing"]
    verdict = "FAIL" if trig else ("INCOMPLETE" if missing else "PASS")
    warnings = list(m.warnings)
    if missing:
        warnings.append("not measured: " + ", ".join(it["item"] for it in missing)
                        + " (never passed: re-scan with those parts lit and decoded)")
    rep = {"kind": "station_check", "label": label, "verdict": verdict, "golden_id": g.id,
           "golden_label": g.label, "measurement": {"label": m.label, "scan_id": m.scan_id,
                                                   "file": str(m.files.get("json") or "")},
           "limits": g.limits, "items": items, "triggers": trig, "n_triggers": len(trig),
           "units": UNITS, "created": _now(), "mock": bool(s.mock), "warnings": warnings}
    out = s.store.output_path("defects", f"{label}_station_check", ".json")
    s.store.write_json(out, json_safe(rep))
    src = Path(inner["ms"].path)
    s.store.record_output("defects", out, label=label, geometry_id=g.geometry_id, source=src)
    csv_p = _write_text(out.with_suffix(".csv"), _csv_check(items))
    s.store.record_output("defects", csv_p, label=label, geometry_id=g.geometry_id, source=src)
    r, X, d = inner["r"], inner["X"], inner["d"]
    png = figure.station_png(m.meta, d, r["arrays"].get("ct_mask"),
                             _zone_image(X, r["arrays"].get("ct_mask"), r["frame"], g.zone_model),
                             check={"items": items, "verdict": verdict},
                             title=f"station check {label} vs golden {g.label} - scan {m.scan_id}")
    png_p = _write_bytes(out.with_suffix(".png"), png)
    if png_p:
        s.store.record_output("defects", png_p, label=label, geometry_id=g.geometry_id, source=src)
    return StationCheck(verdict=verdict, triggers=trig, n_triggers=len(trig), items=items,
                        golden_id=g.id, measurement=rep["measurement"], units=UNITS,
                        files={"json": out, "csv": csv_p, "png": png_p,
                               "measurement_json": m.files.get("json")},
                        warnings=warnings)

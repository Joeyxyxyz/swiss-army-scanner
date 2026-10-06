"""
Stereo calibration workflow behind the api verbs: printable boards, the
diamond row-parity bench step, board-pose capture and calibrate_stereo.

Library layout (append-only, rule 7):
    calib_poses/<geometry_id>/<ts>_<set>/   flood.png, field.npz (both axes), meta.json (last)
    calibrations/<geometry_id>/<ts>_parity/ parity.json
    calibrations/<geometry_id>/<ts>_stereo/ report.json, stereo_calibration.json (last)

Poses and parity carry the preset's geometry fingerprint; calibrate_stereo
only uses poses captured at the same geometry_id AND fingerprint (rule 6),
so a moved rig can never be calibrated from old poses.

In mock mode, `mock_poses=N` makes calibrate_stereo place N board poses on
the simulated bench itself (standing in for the operator's hands).
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from .. import config, rules
from ..library.store import json_safe, read_meta, safe_label
from . import parity as parity_mod
from . import stereo
from .charuco import BoardSpec, detect, write_pdf, write_png

PARITY_FILE = "parity.json"
AUTO = "auto"


def board_spec(board: Optional[dict] = None, square_mm: Optional[float] = None) -> BoardSpec:
    spec = BoardSpec.from_dict(board)
    return spec.scaled(square_mm) if square_mm else spec


# ------------------------------------------------------------ board files

def make_charuco(s, spec: BoardSpec, dpi: float = 600.0, label: str = "charuco",
                 margin_mm: float = 10.0) -> dict:
    """PNG + PDF (true size) + spec JSON in output/calibration/."""
    lab = (f"{label}_{spec.squares_x}x{spec.squares_y}_{spec.square_mm:g}mm"
           .replace(".", "p"))
    png = s.store.output_path("calibration", lab, ".png")
    pdf = png.with_suffix(".pdf")
    js = png.with_suffix(".json")
    rules.ensure_all_new([png, pdf, js])
    write_png(png, spec, dpi, margin_mm)
    write_pdf(pdf, spec, dpi, margin_mm)
    w, h = spec.board_size_mm
    s.store.write_json(js, {**spec.to_dict(), "board_mm": [w, h], "dpi": dpi,
                            "margin_mm": margin_mm, "n_corners": spec.n_corners,
                            "print": "print at 100 % / actual size; measure a square with "
                                     "calipers and pass it as square_mm to calibrate_stereo"})
    for f in (png, pdf, js):
        s.store.record_output("calibration", f, label=label)
    return {"png": png, "pdf": pdf, "spec_json": js, "board": spec.to_dict(),
            "board_mm": [w, h], "dpi": dpi}


# ------------------------------------------------------------- the rig

def capture_patterns(s, preset, patterns: list, what: str, zone_mask=None) -> list:
    """Project + capture a list of DMD patterns in one rig session.
    zone_mask: DMD mask from zones.resolve - only that zone is lit (black surround)."""
    rules.check_led_duration(preset.projection.led, None, what)
    with s.rig(preset, zone_mask=zone_mask) as sync:
        return [sync.shoot(p) for p in patterns]


def _flood_image(frames) -> np.ndarray:
    """The brighter of the white / black frames (the HDMI path may invert)."""
    a, b = frames[0], frames[1]
    img = a if float(np.mean(a)) >= float(np.mean(b)) else b
    return np.clip(img, 0, 255).astype(np.uint8)


# ------------------------------------------------------------- row parity

def row_parity(s, preset, label: str = "parity") -> dict:
    """Bench step B3: project the three row-parity patterns (+ a dark frame),
    measure which rows sit half a column right, store it with the geometry."""
    frames = capture_patterns(s, preset, parity_mod.parity_patterns()
                              + [np.zeros((1140, 912), np.uint8)], "the row-parity check")
    rep = parity_mod.measure_parity(frames[:3], dark=frames[3])
    fp = config.geometry_fingerprint(preset)
    rep = {**rep, "geometry_id": preset.geometry_id, "geometry_fp": fp, "preset": preset.name,
           "created": datetime.now().isoformat(timespec="seconds"), "mock": s.mock}
    if rep.get("ok"):
        d = s.store.new_dir("calibrations", preset.geometry_id, label="parity")
        s.store.write_json(d / PARITY_FILE, json_safe(rep))
        rep["path"] = d
    out = s.store.output_path("calibration", label, ".json")
    s.store.write_json(out, json_safe(rep))
    s.store.record_output("calibration", out, label=label, geometry_id=preset.geometry_id)
    return rep


def latest_parity(store, geometry_id: str, fp: Optional[dict]) -> Optional[dict]:
    import json
    for d in reversed(store.entries("calibrations", geometry_id)):
        f = d / PARITY_FILE
        if not f.is_file():
            continue
        try:
            rep = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not rules.fingerprint_diffs(rep.get("geometry_fp"), fp):
            return rep
    return None


def resolve_parity(s, preset, parity=AUTO) -> tuple:
    """-> (parity 0 | 1 | None, source, warnings). 'auto' = the preset's
    multifreq diamond_parity if it sets one, else the latest row_parity
    measured at this geometry, else None (square grid) with a warning."""
    if parity != AUTO:
        if parity not in (None, 0, 1):
            raise ValueError(f"diamond_parity must be 'auto', null, 0 or 1, got {parity!r}")
        return parity, "argument", []
    for b in preset.bases:
        if b.get("name") == "multifreq" and "diamond_parity" in b:
            return b["diamond_parity"], "preset", []
    rep = latest_parity(s.store, preset.geometry_id, config.geometry_fingerprint(preset))
    if rep is not None:
        return rep.get("parity"), "row_parity", []
    return None, "none", ["diamond row parity unknown (run row_parity first): columns are "
                          "decoded as a square grid, which leaves a +-0.25 column zigzag on a "
                          "diamond DMD"]


def multifreq_basis(preset, parity, steps=None, periods=None, axis="xy"):
    from ..bases import multifreq
    spec = next((b for b in preset.bases if b.get("name") == "multifreq"), {})
    return multifreq.from_spec({k: v for k, v in spec.items() if k != "axis"}, axis=axis,
                               steps=steps, periods_px=periods,
                               **({"diamond_parity": parity} if parity is not None else {}))


# ---------------------------------------------------------- board poses

def capture_pose(s, preset, set_label: str = "stereo", spec: Optional[BoardSpec] = None,
                 steps=None, periods=None, parity=AUTO) -> dict:
    """One board pose: flood + both-axis multifreq, ChArUco detection, saved
    under calib_poses/<geometry_id>/. The operator holds the board still."""
    safe_label(set_label)
    spec = spec or BoardSpec()
    par, src, warn = resolve_parity(s, preset, parity)
    basis = multifreq_basis(preset, par, steps, periods)
    t0 = time.perf_counter()
    frames = capture_patterns(s, preset, basis.generate(), "a calibration pose")
    fld = basis.decode(frames, preset.geometry_id, contrast_min=preset.decode.contrast_min,
                       conf_min=preset.decode.conf_min)
    flood = _flood_image(frames)
    det = detect(flood, spec)
    n = 0 if det is None else len(det[1])
    fp = config.geometry_fingerprint(preset)
    d = s.store.new_dir("calib_poses", preset.geometry_id, label=set_label)
    with s.store.filling(d):
        s.store.write_field(d / "field.npz", fld)
        s.store.write_preview(d / "flood.png", flood.astype(np.float64))
        s.store.write_npz(d / "flood.npz", image=flood)
        meta = {"id": d.name, "set": set_label, "geometry_id": preset.geometry_id,
                "geometry_fp": fp, "preset": preset.name, "board": spec.to_dict(),
                "corners_found": n, "diamond_parity": par, "parity_source": src,
                "steps": basis.steps, "periods_px": basis.periods, "mock": s.mock,
                "valid_pct": round(fld.valid_pct, 2), "decode": fld.meta,
                "capture_s": round(time.perf_counter() - t0, 2),
                "created": datetime.now().isoformat(timespec="seconds"), "warnings": warn}
        s.store.write_json(d / "meta.json", json_safe(meta))
    if n < 6:
        meta["warnings"] = warn + [f"only {n} ChArUco corners found: this pose will not be used "
                                   "(board out of view, too steep, or too dark)"]
    return {**meta, "path": d}


def list_poses(store, geometry_id: str, fp: Optional[dict] = None,
               set_label: Optional[str] = None) -> list:
    out = []
    for d in store.entries("calib_poses", geometry_id):
        m = read_meta(d)
        if m is None or rules.fingerprint_diffs(m.get("geometry_fp"), fp):
            continue
        if set_label is not None and m.get("set") != set_label:
            continue
        out.append({**m, "path": d})
    return out


def _load_pose(d: Path) -> dict:
    from ..decode.deformation import DeformationField
    fld = DeformationField.load(d / "field.npz")
    with np.load(d / "flood.npz") as z:
        img = z["image"]
    return {"image": img, "corr_x": fld.corr_x, "corr_y": fld.corr_y}


def calibrate_stereo(s, preset, set_label: Optional[str] = None, square_mm: Optional[float] = None,
                     board: Optional[dict] = None, projector_distortion: bool = False,
                     mock_poses: int = 0, steps=None, periods=None, parity=AUTO,
                     seed: int = 0) -> tuple:
    """-> (StereoCalibration, summary dict). Uses every pose of `set_label`
    (default: the newest set) at this geometry + fingerprint."""
    fp = config.geometry_fingerprint(preset)
    if mock_poses:
        if not s.mock:
            raise rules.RuleViolation("mock_poses is mock-only: on the rig the operator moves "
                                      "the board and runs capture_calib_pose per pose")
        from ..capture.mock_stereo import CharucoTarget, board_poses
        spec0 = board_spec(board)
        set_label = set_label or f"mock{datetime.now().strftime('%H%M%S')}"
        bench = s.bench
        for R, t in board_poses(spec0, int(mock_poses), seed=seed):
            bench.clear()
            bench.use_stereo()
            bench.place(CharucoTarget(spec=spec0, R=R, t=t))
            capture_pose(s, preset, set_label, spec0, steps, periods, parity)
        bench.clear()
    poses = list_poses(s.store, preset.geometry_id, fp)
    if not poses:
        raise ValueError(f"no calibration poses at geometry {preset.geometry_id} (these "
                         "settings): capture 12-20 with capture_calib_pose first")
    if set_label is None:
        set_label = poses[-1].get("set")
    poses = [p for p in poses if p.get("set") == set_label]
    specs = {tuple(sorted(p["board"].items())) for p in poses}
    pars = {p.get("diamond_parity") for p in poses}
    if len(specs) != 1 or len(pars) != 1:
        raise ValueError(f"pose set {set_label!r} mixes boards or diamond parities; "
                         "capture one consistent set")
    spec = board_spec(poses[0]["board"], square_mm)
    par = pars.pop()
    data = [_load_pose(p["path"]) for p in poses]
    fit = stereo.calibrate(data, spec, cam_size=data[0]["image"].shape[::-1],
                           projector_distortion=projector_distortion)
    truth = s.bench.stereo_truth() if s.mock and getattr(s.bench, "stereo_rig", None) is not None \
        else None
    cal = stereo.save(s.store, preset.geometry_id, fit, fingerprint=fp, spec=spec, parity=par,
                      created=datetime.now().isoformat(timespec="seconds"),
                      extra_report={"set": set_label, "poses": [p["id"] for p in poses],
                                    "square_mm_used": spec.square_mm},
                      truth=truth)
    summary = {"calibration_id": cal.id, "geometry_id": cal.geometry_id, "set": set_label,
               "n_poses_captured": len(poses), "n_poses_used": fit["n_poses"],
               "rms": cal.rms, "baseline_mm": round(cal.baseline_mm, 3),
               "angle_deg": round(cal.angle_deg, 3), "diamond_parity": par,
               "fv_over_fu": round(fit["proj"]["fv_over_fu"], 4),
               "camera_per_pose_rms": [round(x, 4) for x in fit["cam"]["per_pose_rms"]],
               "path": cal.path, "units": "px (reprojection), mm (baseline)", "warnings": []}
    if truth is not None:
        summary["vs_truth"] = stereo.compare_truth(cal, truth)
    if not fit["cam"]["ok"]:
        summary["warnings"].append(f"camera RMS {fit['cam']['rms']:.3f} px >= 0.3 px target")
    if not fit["proj"]["ok"]:
        summary["warnings"].append(f"projector RMS u {fit['proj']['rms_u']:.3f} / v "
                                   f"{fit['proj']['rms_v']:.3f} px: target < 0.5 px per axis")
    if par is None:
        summary["warnings"].append("calibrated without a diamond row parity (square-grid "
                                   "columns); run row_parity and recalibrate for best accuracy")
    out = s.store.output_path("calibration", f"stereo_{set_label}", ".json")
    s.store.write_json(out, json_safe(summary))
    s.store.record_output("calibration", out, label=set_label, geometry_id=preset.geometry_id,
                          source=cal.path)
    return cal, summary

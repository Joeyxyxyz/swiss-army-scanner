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

Each pose records how many of its corners got a projector position; the default
fringe periods are retried coarser when they map too few (capture_pose), a pose
set uses one period set, and calibrate_stereo fails loudly - nothing saved,
stereo.CalibrationRejected - when the projector fit is degenerate, the stereo
RMS is beyond a sane bound or too few poses gave projector points.

The diamond row parity is either MEASURED (row_parity; "no half-column offset:
square grid" is a result), set by the preset / the caller, or not measured: only
the last one is a warning.

In mock mode, `mock_poses=N` makes calibrate_stereo place N board poses on
the simulated bench itself (standing in for the operator's hands).

Calibration fixture (calibrate/fixture.py, MOCK-TESTED ONLY): capture_pose(fixture=spec)
detects every face's board in the one capture (disjoint marker ids) and stores the
placement once (one field, one flood) with all its boards; calibrate_stereo expands each
stored placement into one planar view per detected face, runs the unchanged per-view
calibration, then the rigid-fixture refinement (calibrate/rigid.py: the face-to-face poses
are unknowns shared by every placement) and the face-angle check against the spec.
`mock_fixture=N` places the fixture N times on the simulated bench.
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
from . import fixture as fixture_mod
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
    measured at this geometry, else None (square grid) with a warning.
    source says how the value is known: "argument" / "preset" (set), "row_parity"
    (MEASURED - parity None then means the measurement found no half-column offset: a
    square grid, not a missing measurement) or "none" (not measured)."""
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

MIN_POSE_CORNERS = 6                # ChArUco corners for a board view to be usable
MIN_PROJECTOR_CORNER_PCT = 60.0     # a pose with fewer corners mapped to the projector warns
# coarser fringe sets tried, in this order, when the default periods map too few corners
# (a printed board at ~0.8 m with the lens opened: 8 px fringes mapped 0-11 of ~50 corners,
# 16 px mapped them all)
COARSER_PERIODS = ((16.0, 160.0, 1280.0), (24.0, 192.0, 1280.0))


def periods_text(periods) -> str:
    return ",".join(f"{float(x):g}" for x in periods or [])


def _same_periods(a, b) -> bool:
    return (a is not None and b is not None and len(a) == len(b)
            and all(abs(float(x) - float(y)) < 1e-9 for x, y in zip(a, b)))


def pose_accepted(m: dict) -> bool:
    """A stored pose that fixes its set's fringe periods: the board is usable in it and
    enough of its corners got a projector position. Poses stored before the projector-corner
    count existed count when their board is usable."""
    if "accepted" in m:
        return bool(m["accepted"])
    if m.get("fixture"):
        return len(m.get("faces_seen") or []) >= 2
    return int(m.get("corners_found") or 0) >= MIN_POSE_CORNERS


def set_periods(poses: list) -> Optional[list]:
    """The fringe periods of a pose set: those of its first accepted pose (None: no accepted
    pose yet). Every pose of a set is captured, and calibrated, with ONE period set."""
    for m in poses:
        if pose_accepted(m) and m.get("periods_px"):
            return [float(x) for x in m["periods_px"]]
    return None


def _shoot_pose(s, preset, boards: list, fxs, par, steps, periods) -> dict:
    """One capture of the board(s) at these fringe periods: decode, detection, and how many
    corners got a projector position."""
    from .projector import projector_corner_count
    basis = multifreq_basis(preset, par, steps, periods)
    t0 = time.perf_counter()
    frames = capture_patterns(s, preset, basis.generate(), "a calibration pose")
    fld = basis.decode(frames, preset.geometry_id, contrast_min=preset.decode.contrast_min,
                       conf_min=preset.decode.conf_min)
    flood = _flood_image(frames)
    found, mapped = [], []
    for b in boards:
        det = detect(flood, b)
        found.append(0 if det is None else len(det[1]))
        mapped.append(0 if det is None else
                      projector_corner_count(det[0], fld.corr_x, fld.corr_y))
    n, k = int(sum(found)), int(sum(mapped))
    usable = (sum(c >= MIN_POSE_CORNERS for c in found) >= 2) if fxs else n >= MIN_POSE_CORNERS
    pct = round(100.0 * k / n, 1) if n else 0.0
    return {"basis": basis, "fld": fld, "flood": flood, "found": found, "mapped": mapped,
            "n": n, "k": k, "pct": pct, "usable": bool(usable),
            "accepted": bool(usable and pct >= MIN_PROJECTOR_CORNER_PCT),
            "capture_s": round(time.perf_counter() - t0, 2)}


def capture_pose(s, preset, set_label: str = "stereo", spec: Optional[BoardSpec] = None,
                 steps=None, periods=None, parity=AUTO, fixture=None,
                 auto_periods: bool = True) -> dict:
    """One board pose: flood + both-axis multifreq, ChArUco detection, saved
    under calib_poses/<geometry_id>/. The operator holds the board still.
    fixture: a calibration-fixture spec (path or dict): one placement of the fixture, every
    face's board detected on its own (meta "boards" / "fixture"), stored once.

    Projector corners: the pose records how many of the corners found got a projector
    position (projector_corners / corners_found, projector_corner_pct) and warns below
    MIN_PROJECTOR_CORNER_PCT - fringes too fine for the print, the focus or the distance
    decode nothing on the board, and a calibration from such poses is garbage.
    Fringe periods, one set per pose set:
      - periods given: used as they are (refused when the set already uses others)
      - the set has an accepted pose: its periods (the first accepted pose fixes them)
      - else the preset's (default 8, 80, 1280 DMD px); when they map too few corners and
        auto_periods is on, the pose is captured again at COARSER_PERIODS (16,160,1280, then
        24,192,1280) and the first set that reaches the limit is stored; when none does, the
        best attempt is stored, not accepted (it does not fix the set's periods)."""
    safe_label(set_label)
    fxs = fixture_mod.load(fixture) if fixture is not None else None
    boards = fixture_mod.boards(fxs) if fxs else [spec or BoardSpec()]
    par, src, warn = resolve_parity(s, preset, parity)
    fp = config.geometry_fingerprint(preset)
    fixed = set_periods(list_poses(s.store, preset.geometry_id, fp, set_label))
    default = [float(x) for x in multifreq_basis(preset, par, steps, None).periods]
    if periods is not None:
        want = [float(x) for x in periods]
        if fixed is not None and not _same_periods(want, fixed):
            raise ValueError(
                f"pose set {set_label!r} is captured with fringe periods {periods_text(fixed)} "
                f"DMD px (fixed by its first accepted pose); periods {periods_text(want)} would "
                "mix two period sets in one calibration. Leave the periods out to use the "
                "set's, or start a new pose set (another set name)")
        ladder, source = [want], "argument"
    elif fixed is not None:
        ladder, source = [fixed], "set"
    else:
        ladder, source = [default], "preset"
        if auto_periods:
            ladder += [list(c) for c in COARSER_PERIODS if c[0] > default[0] + 1e-9]
    shots = []
    for per in ladder:
        shots.append(_shoot_pose(s, preset, boards, fxs, par, steps, per))
        if shots[-1]["accepted"] or not shots[-1]["usable"]:
            break                       # good, or no board to map: coarser fringes can't help
    shot = shots[-1] if shots[-1]["accepted"] else max(shots, key=lambda a: a["pct"])
    basis, fld, found = shot["basis"], shot["fld"], shot["found"]
    n = shot["n"]
    if shot is not shots[0]:
        source = "auto"
    attempts = [{"periods_px": a["basis"].periods, "corners_found": a["n"],
                 "projector_corners": a["k"], "projector_corner_pct": a["pct"],
                 "valid_pct": round(a["fld"].valid_pct, 2), "accepted": a["accepted"],
                 "stored": a is shot} for a in shots]
    # what the operator is told (and what the entry records)
    warnings = list(warn)
    seen = [f["face"] for f, c in zip(fxs["faces"], found) if c >= MIN_POSE_CORNERS] \
        if fxs else []
    if fxs:
        miss = [f"face {f['face']} ({c} corners)" for f, c in zip(fxs["faces"], found)
                if c < MIN_POSE_CORNERS]
        if miss:
            warnings.append(f"not usable in this placement: {', '.join(miss)} (out of view, "
                            "too steep, in shadow or too dark)")
        if len(seen) < 2:
            warnings.append(f"only {len(seen)} face(s) seen: turn / move the fixture so at "
                            "least 2 (ideally all) faces show")
    elif n < MIN_POSE_CORNERS:
        warnings.append(f"only {n} ChArUco corners found: this pose will not be used (board "
                        "out of view, too steep, or too dark)")
    warnings += _projector_corner_warnings(shot, shots, source, set_label, fixed)
    d = s.store.new_dir("calib_poses", preset.geometry_id, label=set_label)
    with s.store.filling(d):
        s.store.write_field(d / "field.npz", fld)
        s.store.write_preview(d / "flood.png", shot["flood"].astype(np.float64))
        s.store.write_npz(d / "flood.npz", image=shot["flood"])
        meta = {"id": d.name, "set": set_label, "geometry_id": preset.geometry_id,
                "geometry_fp": fp, "preset": preset.name,
                "board": None if fxs else boards[0].to_dict(),
                "corners_found": n, "projector_corners": shot["k"],
                "projector_corner_pct": shot["pct"], "accepted": shot["accepted"],
                "diamond_parity": par, "parity_source": src,
                "steps": basis.steps, "periods_px": basis.periods, "periods_source": source,
                "attempts": attempts, "mock": s.mock,
                "valid_pct": round(fld.valid_pct, 2), "decode": fld.meta,
                "capture_s": round(sum(a["capture_s"] for a in shots), 2),
                "created": datetime.now().isoformat(timespec="seconds"), "warnings": warnings}
        if fxs:
            meta["boards"] = [b.to_dict() for b in boards]
            meta["fixture"] = {**fixture_mod.summary(fxs), "spec": fxs,
                               "faces_found": [{"face": f["face"], "corners": c,
                                                "projector_corners": k,
                                                "id_range": f["id_range"]}
                                               for f, c, k in zip(fxs["faces"], found,
                                                                  shot["mapped"])]}
            meta["faces_seen"] = seen
        s.store.write_json(d / "meta.json", json_safe(meta))
    return {**meta, "path": d}


def _projector_corner_warnings(shot: dict, shots: list, source: str, set_label: str,
                               fixed) -> list:
    """What to tell the operator about the projector-corner count of a stored pose."""
    per = periods_text(shot["basis"].periods)
    out = []
    if len(shots) > 1 and shot["accepted"]:
        first = shots[0]
        out.append(f"fringe periods {periods_text(first['basis'].periods)} px gave a projector "
                   f"position to only {first['k']} of {first['n']} corners "
                   f"({first['pct']:.0f} %): captured again at {per} px ({shot['k']} of "
                   f"{shot['n']}, {shot['pct']:.0f} %). Pose set {set_label!r} now uses "
                   f"{per} px for every pose")
    if not shot["usable"] or shot["accepted"]:
        return out
    low = (f"only {shot['k']} of {shot['n']} board corners got a projector position "
           f"({shot['pct']:.0f} % < {MIN_PROJECTOR_CORNER_PCT:g} %) at fringe periods {per} "
           "px: the fringes are not resolved on the board (too fine for this print, focus "
           "or distance, or the board is partly outside the projected light)")
    if source == "argument":
        out.append(low + "; try coarser periods (16,160,1280 or 24,192,1280) in a new pose "
                         "set")
    elif source == "set":
        out.append(low + f"; pose set {set_label!r} keeps the periods of its first accepted "
                         "pose: tilt the board less, check the focus, or start a new pose set "
                         "with coarser periods")
    else:
        tried = "; ".join(f"{periods_text(a['basis'].periods)} px: {a['k']}/{a['n']}"
                          for a in shots)
        out.append(low + f" - tried {tried}. Not accepted: it does not fix the set's periods. "
                         "Check the projector focus and aperture, the board's print, and "
                         "that it sits inside the projected light, then capture it again")
    return out


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
                     seed: int = 0, fixture=None, mock_fixture: int = 0,
                     rigid: bool = True) -> tuple:
    """-> (StereoCalibration, summary dict). Uses every pose of `set_label`
    (default: the newest set) at this geometry + fingerprint.
    fixture: the calibration-fixture spec (path / dict) of a fixture pose set, read for its
    caliper-measured squares (default: the spec stored with the captures); mock_fixture=N
    (mock only) places the fixture N times; rigid: the rigid-fixture refinement."""
    fp = config.geometry_fingerprint(preset)
    if mock_fixture:
        if not s.mock:
            raise rules.RuleViolation("mock_fixture is mock-only: on the rig the operator places "
                                      "the fixture and runs capture_calib_pose --fixture per "
                                      "placement")
        from ..capture.mock_stereo import fixture_placements, fixture_shapes
        fxs = fixture_mod.load(fixture) if fixture is not None else fixture_mod.design()
        fixture = fxs
        set_label = set_label or f"mockfx{datetime.now().strftime('%H%M%S')}"
        bench = s.bench
        for R, t in fixture_placements(int(mock_fixture), seed=seed):
            bench.clear()
            bench.use_stereo()
            for shape in fixture_shapes(fxs, R, t):
                bench.place(shape)
            capture_pose(s, preset, set_label, None, steps, periods, parity, fixture=fxs)
        bench.clear()
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
    n_captured = len(poses)
    poses, skipped, periods_used = _one_period_set(poses, set_label)
    pars = {p.get("diamond_parity") for p in poses}
    is_fx = {bool(p.get("fixture")) for p in poses}
    if len(is_fx) != 1:
        raise ValueError(f"pose set {set_label!r} mixes single-board poses and fixture "
                         "placements; capture one consistent set")
    info = {"n_captured": n_captured, "skipped": skipped, "periods_px": periods_used}
    if is_fx.pop():
        return _calibrate_fixture(s, preset, poses, set_label, square_mm, projector_distortion,
                                  fixture, rigid, fp, pars, info)
    specs = {tuple(sorted(p["board"].items())) for p in poses}
    if len(specs) != 1 or len(pars) != 1:
        raise ValueError(f"pose set {set_label!r} mixes boards or diamond parities; "
                         "capture one consistent set")
    spec = board_spec(poses[0]["board"], square_mm)
    par = pars.pop()
    psrc, pinfo, pwarn = _parity_of(s, preset, poses, par)
    data = [_load_pose(p["path"]) for p in poses]
    ids = [p["id"] for p in poses]
    try:
        fit = stereo.calibrate(data, spec, cam_size=data[0]["image"].shape[::-1],
                               projector_distortion=projector_distortion)
        _label_poses(fit["per_pose"], ids)
        truth = _mock_truth(s)
        cal = stereo.save(s.store, preset.geometry_id, fit, fingerprint=fp, spec=spec,
                          parity=par, created=datetime.now().isoformat(timespec="seconds"),
                          extra_report={"set": set_label, "poses": ids,
                                        "square_mm_used": spec.square_mm,
                                        "periods_px": periods_used, "poses_skipped": skipped},
                          truth=truth, parity_source=psrc, parity_info=pinfo)
    except stereo.CalibrationRejected as e:
        raise _rejected(s, preset, e, ids, set_label, info) from None
    summary = _summary(cal, fit, set_label, n_captured, par, info)
    summary["warnings"] += pwarn
    if truth is not None:
        summary["vs_truth"] = stereo.compare_truth(cal, truth)
    return _finish(s, preset, cal, fit, summary, set_label, par)


def _one_period_set(poses: list, set_label: str) -> tuple:
    """-> (the poses calibrated together, [skipped {id, periods_px, why}], the periods).
    A calibration uses ONE fringe period set: that of the set's first accepted pose. Poses
    captured at other periods that were NOT accepted (a failed attempt left in the set) are
    left out and named; accepted poses at different periods are refused."""
    fixed = set_periods(poses)
    if fixed is None:
        kinds = {tuple(float(x) for x in (p.get("periods_px") or [])) for p in poses}
        if len(kinds) > 1:
            raise ValueError(
                f"pose set {set_label!r} mixes fringe periods "
                f"({'; '.join(periods_text(k) for k in sorted(kinds))} px) and none of its "
                "poses was accepted: capture one consistent set")
        return poses, [], (list(next(iter(kinds))) or None) if kinds else None
    keep, skipped = [], []
    for p in poses:
        per = p.get("periods_px")
        if not per or _same_periods(per, fixed):
            keep.append(p)
        elif pose_accepted(p):
            raise ValueError(
                f"pose set {set_label!r} mixes fringe periods: pose {p['id']} was captured at "
                f"{periods_text(per)} px, the set uses {periods_text(fixed)} px (its first "
                "accepted pose). One calibration takes one period set: move that pose to the "
                f"trash (`scan delete {p['id']}`) or capture a new set")
        else:
            skipped.append({"id": p["id"], "periods_px": [float(x) for x in per],
                            "why": f"captured at {periods_text(per)} px and not accepted; the "
                                   f"set uses {periods_text(fixed)} px"})
    return keep, skipped, fixed


_PARITY_RANK = ("row_parity", "preset", "argument", "none")


def _parity_of(s, preset, poses: list, par) -> tuple:
    """How the diamond parity the poses were decoded with is known -> (source, info,
    warnings): MEASURED (row_parity; also when it found no offset), set, or not measured. A
    measurement made after the poses counts when it agrees with how they were decoded."""
    srcs = {p.get("parity_source") for p in poses} - {None}
    src = next((k for k in _PARITY_RANK if k in srcs), None)
    rep = latest_parity(s.store, preset.geometry_id, config.geometry_fingerprint(preset))
    warn = []
    if src in (None, "none") and rep is not None:
        if rep.get("parity") == par:
            src = "row_parity"              # measured since: the same decode
        else:
            warn.append(f"the poses were decoded with diamond parity {par}, but the row parity "
                        f"has since been measured as {rep.get('parity')}: capture the poses "
                        "again, then recalibrate")
    info = {}
    if src == "row_parity" and rep is not None and rep.get("parity") == par:
        info = {k: rep.get(k) for k in ("shift_cols", "shift_spread_cols", "n_lines", "note",
                                        "created") if rep.get(k) is not None}
    return src or "none", info, warn


def _label_poses(per_pose: list, ids: list, views: Optional[list] = None) -> None:
    """Give every per-pose row of a fit its library id (and face view)."""
    for r in per_pose:
        i = int(r["pose"])
        r["id"] = views[i]["view"] if views is not None else ids[i]


def _rejected(s, preset, e, ids: list, set_label: str, info: dict, views=None):
    """A CalibrationRejected with its poses named, and the rejection written to the output
    bin (a report, never a calibration)."""
    rows = [dict(r) for r in e.per_pose]
    _label_poses(rows, ids, views)
    err = stereo.CalibrationRejected(e.reasons, rows, e.numbers)
    out = s.store.output_path("calibration", f"stereo_{set_label}_REJECTED", ".json")
    s.store.write_json(out, json_safe({
        "rejected": True, "saved": False, "set": set_label, "geometry_id": preset.geometry_id,
        "reasons": err.reasons, "per_pose": rows, "numbers": err.numbers,
        "periods_px": info.get("periods_px"), "poses_skipped": info.get("skipped"),
        "created": datetime.now().isoformat(timespec="seconds")}))
    s.store.record_output("calibration", out, label=set_label, geometry_id=preset.geometry_id)
    err.report_path = out
    return err


def _mock_truth(s) -> Optional[dict]:
    return s.bench.stereo_truth() if s.mock and getattr(s.bench, "stereo_rig", None) is not None \
        else None


def _summary(cal, fit: dict, set_label: str, n_captured: int, par,
             info: Optional[dict] = None) -> dict:
    info = info or {}
    rows = [{k: r.get(k) for k in ("id", "corners", "projector_corners",
                                   "projector_corner_pct", "contributed", "camera_rms_px")}
            for r in fit["per_pose"]]
    return {"calibration_id": cal.id, "geometry_id": cal.geometry_id, "set": set_label,
            "n_poses_captured": n_captured, "n_poses_used": fit["n_poses"],
            "rms": cal.rms, "baseline_mm": round(cal.baseline_mm, 3),
            "angle_deg": round(cal.angle_deg, 3), "diamond_parity": par,
            "diamond_parity_source": cal.parity_source,
            "diamond_parity_measured": cal.parity_measured,
            "diamond_parity_text": stereo.parity_text(par, cal.parity_source),
            "fv_over_fu": round(fit["proj"]["fv_over_fu"], 4),
            "camera_per_pose_rms": [round(x, 4) for x in fit["cam"]["per_pose_rms"]],
            "per_pose": rows,
            "projector_poses": [r["id"] for r in rows if r.get("contributed")],
            "periods_px": info.get("periods_px"), "poses_skipped": info.get("skipped") or [],
            "path": cal.path, "units": "px (reprojection), mm (baseline)", "warnings": []}


def _finish(s, preset, cal, fit: dict, summary: dict, set_label: str, par) -> tuple:
    if not fit["cam"]["ok"]:
        summary["warnings"].append(f"camera RMS {fit['cam']['rms']:.3f} px >= 0.3 px target")
    if not fit["proj"]["ok"]:
        summary["warnings"].append(f"projector RMS u {fit['proj']['rms_u']:.3f} / v "
                                   f"{fit['proj']['rms_v']:.3f} px: target < 0.5 px per axis")
    if cal.parity_source == "none":         # NOT measured (a measured "no offset" is fine)
        summary["warnings"].append("the diamond row parity was not measured: the columns were "
                                   "decoded as a square grid, which leaves a +-0.25 column "
                                   "zigzag on a diamond DMD. Run row_parity (`scan parity`), "
                                   "capture the poses again and recalibrate")
    for sk in summary.get("poses_skipped") or []:
        summary["warnings"].append(f"pose {sk['id']} left out: {sk['why']}")
    idle = [r["id"] for r in summary.get("per_pose") or [] if not r.get("contributed")]
    if idle:
        summary["warnings"].append(f"{len(idle)} pose(s) gave no projector points (fewer than "
                                   f"{stereo.MIN_PROJECTOR_CORNERS} corners with a projector "
                                   f"position): {', '.join(str(i) for i in idle)}")
    out = s.store.output_path("calibration", f"stereo_{set_label}", ".json")
    s.store.write_json(out, json_safe(summary))
    s.store.record_output("calibration", out, label=set_label, geometry_id=preset.geometry_id,
                          source=cal.path)
    return cal, summary


# ------------------------------------------------------- calibration fixture

RIGID_RMS_FACTOR, RIGID_RMS_SLACK_PX = 1.5, 0.05


def _calibrate_fixture(s, preset, poses: list, set_label: str, square_mm, projector_distortion,
                       fixture, rigid: bool, fp, pars, info: Optional[dict] = None) -> tuple:
    """A fixture pose set: one planar view per detected face of every placement, the
    unchanged per-view calibration, then the rigid-fixture refinement + face-angle check."""
    import cv2
    from . import rigid as rigid_mod
    if square_mm is not None:
        raise ValueError("square_mm is for the single board: a fixture set takes each face's "
                         "caliper value from faces[k].square_mm_measured in the fixture spec; "
                         "fill it in and pass the spec (calibrate-stereo --fixture SPEC)")
    if len(pars) != 1:
        raise ValueError(f"pose set {set_label!r} mixes diamond parities; capture one "
                         "consistent set")
    par = pars.pop()
    info = info or {"n_captured": len(poses)}
    psrc, pinfo, pwarn = _parity_of(s, preset, poses, par)
    fps = {p["fixture"].get("fingerprint") for p in poses}
    if len(fps) != 1:
        raise ValueError(f"pose set {set_label!r} mixes fixtures ({sorted(map(str, fps))}); "
                         "capture one consistent set")
    if fixture is not None:
        fxs = fixture_mod.load(fixture)
        if fixture_mod.fingerprint(fxs) != next(iter(fps)):
            raise ValueError(f"fixture spec {fxs.get('name')!r} is not the fixture these "
                             "placements were captured with (other boards or face poses)")
    else:
        fxs = fixture_mod.load(poses[0]["fixture"]["spec"])
    boards = fixture_mod.boards(fxs)
    views, where = [], []
    for k, p in enumerate(poses):
        dat = _load_pose(p["path"])
        for f, b in zip(fxs["faces"], boards):
            views.append({**dat, "spec": b, "view": f"{p['id']}/face{f['face']}"})
            where.append((k, int(f["face"])))
    try:
        fit = stereo.calibrate(views, boards[0], cam_size=views[0]["image"].shape[::-1],
                               projector_distortion=projector_distortion)
    except stereo.CalibrationRejected as e:
        raise _rejected(s, preset, e, [], set_label, info, views) from None
    _label_poses(fit["per_pose"], [], views)
    warnings = list(pwarn)
    indep = {"camera_rms_px": fit["cam"]["rms"], "projector_u_px": fit["proj"]["rms_u"],
             "projector_v_px": fit["proj"]["rms_v"], "K_cam": fit["cam"]["K"].tolist(),
             "dist_cam": np.ravel(fit["cam"]["dist"]).tolist(),
             "K_proj": fit["proj"]["K"].tolist(),
             "R": np.asarray(fit["R"]).tolist(), "t": np.ravel(fit["t"]).tolist()}
    n_pl = len({where[o["view"]][0] for o in fit["observations"]})
    rig = {"used": False, "independent": indep}
    check_views = [{"placement": where[i][0], "face": where[i][1],
                    "R": cv2.Rodrigues(np.asarray(rv, np.float64))[0]}
                   for i, rv in zip(fit["used"], fit["cam"]["rvecs"])]
    if rigid and n_pl >= 2:
        obs = [{**o, "placement": where[o["view"]][0], "face": where[o["view"]][1]}
               for o in fit["observations"]]
        try:
            r = rigid_mod.refine(obs, fit["cam"], fit["proj"], fit["R"], fit["t"],
                                 {int(f["face"]): fixture_mod.face_pose(f) for f in fxs["faces"]},
                                 projector_distortion)
        except (ValueError, np.linalg.LinAlgError) as e:
            r = None
            warnings.append(f"rigid-fixture refinement failed ({e}): the independent per-face "
                            "views are used")
        if r is not None:
            lim_c = max(RIGID_RMS_FACTOR * fit["cam"]["rms"],
                        fit["cam"]["rms"] + RIGID_RMS_SLACK_PX)
            lim_p = max(RIGID_RMS_FACTOR * fit["proj"]["rms"],
                        fit["proj"]["rms"] + RIGID_RMS_SLACK_PX)
            rig.update(rms_cam_px=r["rms_cam"], rms_proj_px=r["rms_proj"],
                       converged=r["success"], nfev=r["nfev"], ref_face=r["ref_face"])
            if r["success"] and r["rms_cam"] <= lim_c and r["rms_proj"] <= lim_p:
                fit = _apply_rigid(fit, r)
                rig["used"] = True
                rig["face_poses"] = {str(f): {"R": np.asarray(R).tolist(),
                                              "t": np.ravel(t).tolist()}
                                     for f, (R, t) in sorted(r["faces"].items())}
                check_views = [{"placement": 0, "face": f, "R": R}
                               for f, (R, _) in sorted(r["faces"].items())]
            else:
                warnings.append(
                    f"rigid-fixture refinement rejected: camera RMS {r['rms_cam']:.3f} px / "
                    f"projector {r['rms_proj']:.3f} px vs {fit['cam']['rms']:.3f} / "
                    f"{fit['proj']['rms']:.3f} px for the independent views - did the fixture "
                    "move or flex between placements (a loose wedge, a lifting sticker)? The "
                    "independent per-face views are used")
    elif rigid:
        warnings.append("one placement only: the rigid-fixture refinement needs >= 2 (3 "
                        "recommended: left, right, centre, turned differently)")
    check = fixture_mod.consistency(fxs, check_views)
    check["source"] = "rigid estimate" if rig["used"] else "per placement, independent views"
    warnings += check["warnings"]
    truth = _mock_truth(s)
    board_rec = {"fixture": fxs.get("name"), "fingerprint": fixture_mod.fingerprint(fxs),
                 "faces": [b.to_dict() for b in boards]}
    per_view = [{"view": views[i]["view"], "placement": where[i][0], "face": where[i][1]}
                for i in fit["used"]]
    try:
        cal = stereo.save(s.store, preset.geometry_id, fit, fingerprint=fp, spec=boards[0],
                          parity=par, created=datetime.now().isoformat(timespec="seconds"),
                          extra_report={"set": set_label, "poses": [p["id"] for p in poses],
                                        "fixture": fixture_mod.summary(fxs), "views": per_view,
                                        "squares_mm_used": [b.square_mm for b in boards],
                                        "rigid": rig, "face_check": check,
                                        "periods_px": info.get("periods_px"),
                                        "poses_skipped": info.get("skipped") or []},
                          truth=truth, board=board_rec, parity_source=psrc, parity_info=pinfo)
    except stereo.CalibrationRejected as e:
        raise _rejected(s, preset, e, [], set_label, info, views) from None
    summary = _summary(cal, fit, set_label, info.get("n_captured", len(poses)), par, info)
    summary.update(n_placements=len(poses), n_views=fit["n_poses"],
                   fixture=fixture_mod.summary(fxs), rigid=rig, face_check=check)
    summary["warnings"] += warnings
    if truth is not None:
        summary["vs_truth"] = stereo.compare_truth(cal, truth)
        ind = stereo.StereoCalibration(id="independent", geometry_id=cal.geometry_id,
                                       K_cam=np.asarray(indep["K_cam"]),
                                       dist_cam=np.asarray(indep["dist_cam"]),
                                       K_proj=np.asarray(indep["K_proj"]), dist_proj=cal.dist_proj,
                                       R=np.asarray(indep["R"]), t=np.asarray(indep["t"]))
        summary["vs_truth_independent"] = stereo.compare_truth(ind, truth)
    return _finish(s, preset, cal, fit, summary, set_label, par)


def _apply_rigid(fit: dict, r: dict) -> dict:
    """The fit dict stereo.save expects, with the refined values (RMS recomputed by
    refine over the same corners)."""
    from .camera import CAMERA_RMS_TARGET_PX
    from .projector import PROJECTOR_RMS_TARGET_PX
    cam = {**fit["cam"], "K": r["cam"]["K"], "dist": r["cam"]["dist"], "rms": r["rms_cam"],
           "per_pose_rms": r["per_view_cam_rms"], "ok": r["rms_cam"] < CAMERA_RMS_TARGET_PX}
    pv = [x for x in r["per_view_proj_rms"] if x is not None]
    proj = {**fit["proj"], "K": r["proj"]["K"], "dist": r["proj"]["dist"],
            "rms": r["rms_proj"], "rms_u": r["rms_proj_u"], "rms_v": r["rms_proj_v"],
            "per_pose_rms": pv, "fv_over_fu": float(r["proj"]["K"][1, 1] / r["proj"]["K"][0, 0]),
            "ok": r["rms_proj_u"] < PROJECTOR_RMS_TARGET_PX
            and r["rms_proj_v"] < PROJECTOR_RMS_TARGET_PX}
    rms = float(np.sqrt((r["rms_cam"] ** 2 + r["rms_proj"] ** 2) / 2))
    per_pose = [{**pp, "camera_rms_px": c}                       # same view order (used)
                for pp, c in zip(fit["per_pose"], r["per_view_cam_rms"])]
    return {**fit, "cam": cam, "proj": proj, "R": r["R"], "t": r["t"], "rms_stereo": rms,
            "per_pose": per_pose, "rigid": True}

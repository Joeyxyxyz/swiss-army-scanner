"""
The public verbs (§2) — the ONLY thing a UI, CLI or AI agent imports.

    from engine import api
    api.configure(mock=True)                 # real hardware: api.configure(mock=False)
    p    = api.load_preset("cell_wall_2p7ft_vstripe")   # fringes across the baseline
    sig  = api.check_signal()
    base = api.capture_base(p, "empty")
    obj  = api.capture_object(p, base, "part", known_dims={"H": 106.95, "W": 56.5, "L": 21.9})
    cal  = api.calibrate(p, [obj])
    dims = api.measure(obj)
    ply  = api.reconstruct(obj)
    v    = api.compare(obj2, obj, tol)
    api.list_library("objects")
    codes = api.read_codes()                 # barcode / QR under DMD illumination
    insp = api.inspect(obj)                  # defect blobs, mm verdict vs 0.7 mm trigger
    ds   = api.capture_dent_set("H9:I12", "cell1", long_p, short_p)   # dents on a shiny cell
    null = api.calibrate_dent_null([good_ds], 21.05)                  # known-good cell first
    rep  = api.detect_dents(ds)                                         # the matching null
    face = api.locate_face(locate_scan, 21.05)            # a cell's end face toward the camera
    fs   = api.targeted_face_scan(face, repeats=3)        # face-only lighting, N repeats
    chk  = api.check_face(api.face_profile(fs), golden)   # golden = face_golden([good ...])
    fast = api.fast_scan_flash(pack_dir, "zone")          # flash loop + free-run camera, ~0.3-0.5 s
    scan = api.scan_fast("F10:J12")                       # HDMI: columns only, avg 2, ~5 s
    st   = api.station_measure(scan)                      # cells + cooling tube (CT) zones
    gold = api.station_golden([st], cell_height_tol_mm=0.15, ct_zone_tol_mm=0.3)
    chk  = api.station_check(api.scan_fast("F10:J12"))    # PASS / FAIL / INCOMPLETE

Every verb validates preconditions (rules.py), logs one record to the
session log, never overwrites prior data, and works in mock mode.
Nothing runs before configure(): real hardware is never the silent default.
("cell_wall_2p7ft" is the old rotate90 preset, kept for reference only: its
fringes run along the baseline, so it has almost no depth sensitivity.)
Physical steps (placing objects, focus, iris) are human-only; in mock
mode `mock_bench()` stands in for the human.
"""

from __future__ import annotations

import functools
import inspect as _pyinspect

import numpy as np
import json
import os
from pathlib import Path
from typing import Optional, Union

from . import config, rules
from .calibrate import depth as depth_cal
from .calibrate import lateral as lateral_cal
from .capture import signal
from .capture.mock_bench import MockBench
from .config import Preset
from .library import bases as base_lib
from .library import objects as object_lib
from .library.store import json_safe, timestamp
from .measure import dimensions
from .measure import inspect as inspect_mod
from .sense import antiglare as antiglare_mod
from .sense import codes as codes_mod
from .measure import golden as golden_mod
from .reconstruct import pointcloud, relief
from .session import ScanSession
from .results import (BaseRef, CADRef, Calibration, CodeReading, Dimensions,
                      Inspection, ObjectScan, SignalReport, ToleranceBand, Verdict)

DEFAULT_ROOT = Path(os.environ.get(
    "SWISS_SCANNER_LIBRARY", Path(__file__).resolve().parents[1] / "library"))

_session: Optional[ScanSession] = None

# capture_base quality gate: a base with fewer decoded camera px than this is
# saved (append-only) but auto-REJECTED, so latest_base never hands it out
BASE_MIN_VALID_PCT = 5.0
# ...and one under this fraction of the previous good base's valid % is flagged
BASE_DROP_WARN_FRACTION = 0.5
# zone scans have no datum (raw object - base): warn when the base is older
ZONE_BASE_MAX_AGE_S = 30 * 60


# ------------------------------------------------------------ engine setup

def configure(root: Union[str, Path, None] = None, *, mock: bool,
              presets_dir: Union[str, Path, None] = None,
              bench: Optional[MockBench] = None, holder: str = "") -> ScanSession:
    """Select library root and mock/real. Starts a new session log.

    `mock` has no default: the real rig (camera + fullscreen DMD window) is
    only ever driven after an explicit configure(mock=False)."""
    global _session
    _session = ScanSession(Path(root) if root else DEFAULT_ROOT, mock=mock,
                           presets_dir=Path(presets_dir) if presets_dir else None,
                           bench=bench, holder=holder)
    _session.store.ensure_readme()
    return _session


def paths() -> dict:
    """Where everything lives (for UIs / the operator)."""
    r = _engine().store.root
    return {"library": r, "bases": r / "bases", "objects": r / "objects",
            "calibrations": r / "calibrations", "readings": r / "readings",
            "spectral": r / "spectral", "banks": r / "banks", "maps": r / "maps",
            "sensors": r / "sensors", "atlases": r / "atlases", "dentsets": r / "dentsets",
            "faces": r / "faces", "facescans": r / "facescans", "face_golden": r / "face_golden",
            "output": r / "output", "output_index": r / "output" / "index.csv",
            "sessions": r / "sessions", "trash": r / "trash",
            "presets": config.PRESETS_DIR}


def calibration_for(geometry_id: str, fingerprint: Optional[dict] = None) -> Optional[Calibration]:
    """The calibration measurements at this geometry use (library first,
    then a preset-supplied one). Public so UIs don't re-implement rule 6.
    `fingerprint`: the scan's meta['geometry_fp'], to skip calibrations
    fitted at other geometry-bound settings under the same geometry_id."""
    return _calibration_for(geometry_id, fingerprint)


def list_presets() -> list:
    """Preset names available to load_preset."""
    return config.list_presets(_engine().presets_dir)


def latest_base(preset: Optional[Preset] = None) -> Optional[BaseRef]:
    """Newest usable base for the preset's geometry: not rejected, above the
    valid-pixel floor, and captured at the preset's geometry-bound settings."""
    p = _preset(preset)
    fp = config.geometry_fingerprint(p)
    good = [b for b in base_lib.list_bases(_engine().store, p.geometry_id)
            if not b.meta.get("rejected") and not _below_floor(b.meta.get("valid_pct"))
            and not rules.fingerprint_diffs(b.meta.get("geometry_fp"), fp)]
    return good[-1] if good else None


def _below_floor(valid_pct) -> bool:
    return isinstance(valid_pct, (int, float)) and valid_pct < BASE_MIN_VALID_PCT


def _engine() -> ScanSession:
    if _session is None:
        raise RuntimeError(
            "the engine is not configured: call api.configure(mock=True) for the simulated "
            "bench, or api.configure(mock=False) to drive the real rig, before any verb")
    return _session


def mock_bench() -> MockBench:
    """The simulated bench (mock mode only) — place()/clear() objects on it."""
    s = _engine()
    if not s.mock:
        raise RuntimeError("mock_bench() is only available in mock mode; "
                           "physical steps on the real rig are human-only")
    return s.bench


def _loggable(v):
    """An argument as the session log records it: a cancel Event (a lock inside, which the
    log's deep copy can't take) is recorded as its state."""
    if callable(getattr(v, "is_set", None)) and callable(getattr(v, "wait", None)):
        return f"<cancel event, set={v.is_set()}>"
    return v


def _verb(fn):
    sig = _pyinspect.signature(fn)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        return _engine().run(fn.__name__, {k: _loggable(v) for k, v in bound.arguments.items()},
                             lambda: fn(*args, **kwargs))
    return wrapper


def _preset(preset: Optional[Preset]) -> Preset:
    s = _engine()
    p = preset or s.preset
    if p is None:
        raise RuntimeError("no preset loaded — call load_preset(name) first")
    if p.mock != s.mock:
        raise rules.RuleViolation(
            f"preset was loaded in {'mock' if p.mock else 'real'} mode but the engine "
            f"is in {'mock' if s.mock else 'real'} mode; reload it")
    return p


def _calibration_for(geometry_id: str, fingerprint: Optional[dict] = None) -> Optional[Calibration]:
    """Rule 6. Library calibrations are always read from the library (so one
    trashed by another process, or a CLI next to the UI, stops giving mm at
    once); the in-memory preset calibration is only a fallback when it is
    the preset's OWN values (source 'preset')."""
    s = _engine()
    p = s.preset
    if p is not None and p.calibration is not None and p.calibration.source != "preset" \
            and (p.calibration.path is None or not Path(p.calibration.path).exists()):
        _rebind_calibration(s, p)       # the bound library fit was trashed / moved
    cal = s.store.latest_calibration(geometry_id, fingerprint=fingerprint)
    if cal is None and p is not None and p.calibration is not None \
            and p.calibration.source == "preset" \
            and rules.calibration_matches(p.calibration, geometry_id) \
            and not rules.fingerprint_diffs(p.calibration.fingerprint, fingerprint):
        cal = p.calibration
    return cal


def _rebind_calibration(s: ScanSession, p: Preset) -> None:
    """Point the active preset at its current calibration: the newest library
    fit at its geometry, else the preset file's own values (if any)."""
    cal = s.store.latest_calibration(p.geometry_id, fingerprint=config.geometry_fingerprint(p))
    if cal is None and p.source is not None:
        try:
            cal = config.load_preset(str(p.source), mock=p.mock).calibration
        except Exception:               # noqa: BLE001 - no fallback is a valid answer
            cal = None
    p.calibration = cal


def _age_s(created) -> Optional[float]:
    """Seconds since an entry's 'created' stamp (local isoformat), or None."""
    from datetime import datetime
    try:
        return max(0.0, (datetime.now() - datetime.fromisoformat(str(created))).total_seconds())
    except (TypeError, ValueError):
        return None


def _read_json(f: Path) -> Optional[dict]:
    """A library JSON file, or None when missing / half-written / not an object."""
    try:
        d = json.loads(Path(f).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return d if isinstance(d, dict) else None


def _scan_calibration(scan: ObjectScan) -> Optional[Calibration]:
    return _calibration_for(scan.geometry_id, (scan.meta or {}).get("geometry_fp"))


def _tolerance_for(scan: ObjectScan) -> tuple:
    """(ToleranceCfg, warnings): the FAIL trigger of the preset the scan was
    taken with — not whatever preset happens to be active."""
    s = _engine()
    p = s.preset
    name = (scan.meta or {}).get("preset")
    if p is not None and p.geometry_id == scan.geometry_id and (not name or name == p.name):
        return p.tolerance, []
    if name and config.valid_id(name):
        try:
            own = config.load_preset(name, presets_dir=s.presets_dir, mock=s.mock)
            if own.geometry_id == scan.geometry_id:
                return own.tolerance, []
        except Exception:               # noqa: BLE001 - fall back below, with a warning
            pass
    if p is not None and p.geometry_id == scan.geometry_id:
        return p.tolerance, [f"tolerance taken from the active preset {p.name}: the scan's "
                             f"preset {name!r} could not be loaded"]
    d = config.ToleranceCfg()
    why = (f"the active preset {p.name} is for another geometry" if p is not None
           else "no preset is loaded")
    return d, [f"FAIL trigger {d.ch_fail_mm:g} mm is the engine default: the scan's preset "
               f"{name!r} could not be loaded and {why}"]


# ------------------------------------------------------------------ verbs

@_verb
def load_preset(name: str) -> Preset:
    """Load geometry + settings, validate (§9 rules), bind latest calibration."""
    s = _engine()
    p = config.load_preset(name, presets_dir=s.presets_dir, store=s.store, mock=s.mock)
    s.preset = p
    return p


@_verb
def check_signal(preset: Optional[Preset] = None) -> SignalReport:
    """Project a short test sequence; report gc_conf, valid %, focus hint."""
    p = _preset(preset)
    s = _engine()
    with s.rig(p) as sync:
        return signal.probe(sync, p)


@_verb
def capture_base(preset: Preset, label: str, zone: Optional[str] = None) -> BaseRef:
    """Empty scene -> deformation field -> base library (geometry-tagged).

    zone: light only that grid zone while capturing the base, exactly as a
    zone scan does. Lighting only a zone changes the stray light in it (bench
    2026-09-22: zone scans vs a fully-lit base read sigma 0.50 px, vs 0.20 px
    full field), so precision zone scans should use a base captured with the
    SAME zone; capture_object warns when they differ."""
    p = _preset(preset)
    s = _engine()
    prev = latest_base(p)
    proj_mask = None
    cells = []
    if zone:
        from .capture import zones as zones_mod
        proj_mask, _cam, cells = zones_mod.resolve(s, p, zone)
    fld = s.acquire_field(p, label, zone_mask=proj_mask)
    rules.require_same_geometry(fld.geometry_id, p.geometry_id, "capture_base")
    base = base_lib.save_base(s.store, fld, label,
                              {"preset": p.name, "mock": p.mock, "tier": "relief",
                               "geometry_fp": config.geometry_fingerprint(p),
                               "zone": zone,
                               "zone_cells": [list(map(int, c)) for c in cells] if zone else None,
                               "decode_warnings": list(fld.meta.get("warnings", [])),
                               "saturated_pct": fld.meta.get("saturated_pct")})
    base.warnings.extend(fld.meta.get("warnings", []))
    valid = float(fld.valid_pct)
    if valid < BASE_MIN_VALID_PCT:
        # quality gate: kept (append-only) but never the default base
        reason = (f"only {valid:.1f}% of camera px decoded (floor {BASE_MIN_VALID_PCT:g}%): the "
                  "camera saw almost no pattern (DMD on its blue-lines test pattern, lens cap, "
                  "window on the wrong monitor, exposure far off?). Fix the setup with "
                  "check_signal and capture the base again")
        s.store.write_json(Path(base.path) / "REJECTED.json",
                           {"reason": reason, "do_not_use": True, "auto": "capture_base"})
        base.meta["rejected"] = reason
        base.warnings.append(f"base auto-REJECTED: {reason}. latest_base skips it.")
    elif prev is not None and isinstance(prev.meta.get("valid_pct"), (int, float)) \
            and valid < BASE_DROP_WARN_FRACTION * prev.meta["valid_pct"]:
        base.warnings.append(
            f"valid {valid:.1f}% is under half of the previous base {prev.id} "
            f"({prev.meta['valid_pct']:.1f}%): check the scene, focus and projector before "
            "scanning against it (reject_base if it is bad)")
    return base


@_verb
def capture_object(preset: Preset, base: BaseRef, label: str,
                   known_dims: Optional[dict] = None, zone: Optional[str] = None) -> ObjectScan:
    """Object scene -> field -> relief vs base -> object library.

    zone: a grid zone ("C4", "C4:D6", "B2,E5") from project_grid(). Only that
    zone is lit (less inter-reflection) and the relief is cut to it (needs a
    projector map for the camera-side cut; see capture_map)."""
    p = _preset(preset)
    rules.require_same_geometry(base.geometry_id, p.geometry_id, "capture_object")  # rule 5
    fp = config.geometry_fingerprint(p)
    rules.require_same_fingerprint(base.meta.get("geometry_fp"), fp,
                                   f"capture_object: base {base.id}")
    # relief is taken along the axis the preset's basis codes; a base decoded
    # along another axis (older base, pre-fingerprint) is refused BEFORE capturing
    axis = f"corr_{config.basis_axis(p.bases)}"
    if axis not in base.field.axes:
        raise rules.GeometryMismatch(
            f"capture_object: base {base.id} was decoded along {base.field.axes} but this preset "
            f"codes {axis}: capture a new base with this preset")
    known = object_lib.normalize_known_dims(known_dims)
    s = _engine()
    warnings = []
    if base.meta.get("rejected"):
        warnings.append(f"base {base.id} is REJECTED ({base.meta['rejected']}); this scan's "
                        "relief is against a base marked do-not-use")
    base_age_s = _age_s(base.created)
    proj_mask = cam_mask = None
    cells = []
    if zone:
        from .capture import zones as zones_mod
        proj_mask, cam_mask, cells = zones_mod.resolve(s, p, zone)
        if cam_mask is None:
            warnings.append(f"zone {zone}: no projector map for {p.geometry_id} - zone is lit "
                            "but relief is not cut (run capture_map)")
        base_cells = base.meta.get("zone_cells")
        if not base_cells or sorted(map(tuple, base_cells)) != sorted(map(tuple, cells)):
            warnings.append(
                f"zone scan {zone} against a base captured "
                f"{'with zone ' + str(base.meta.get('zone')) if base_cells else 'fully lit'}: "
                "the stray light differs, which adds noise/offset to zone relief (bench: "
                "sigma 0.50 px vs 0.20 px). For precision, capture_base with the same zone")
        if base_age_s is not None and base_age_s > ZONE_BASE_MAX_AGE_S:
            warnings.append(
                f"zone scan against a base {base_age_s / 60:.0f} min old: zone relief is the raw "
                "object - base difference (no datum), so projector/camera warm-up drift since "
                "the base shows up as height over the whole zone. Capture a fresh base first "
                "for sub-mm zone results")
    fld = s.acquire_field(p, label, zone_mask=proj_mask)
    # zone scans keep the raw object-base difference: only the zone is lit,
    # so a median datum would be the part top when the part fills the zone
    rel, mask, datum = relief.compute(fld, base.field, axis=axis, datum=not zone)
    if cam_mask is not None:
        mask = mask & cam_mask
        rel = np.where(mask, rel, np.nan).astype(np.float32)
    if not mask.any():
        warnings.append("0 pixels valid in both base and object — re-shoot both "
                        "WITHOUT moving camera/projector/focus between them")
    warnings.extend(w for w in fld.meta.get("warnings", []) if w not in warnings)
    cal = _calibration_for(p.geometry_id, fp)
    scan = object_lib.save_object(
        s.store, fld, rel, mask, label, base.id, known, cal.id if cal else None,
        {"preset": p.name, "mock": p.mock, "tier": "relief", "units": "px",
         "datum_px": datum, "overlap_px": int(mask.sum()), "warnings": warnings,
         "relief_axis": axis, "geometry_fp": fp,
         "saturated_pct": fld.meta.get("saturated_pct"),
         "base_age_s": round(base_age_s, 1) if base_age_s is not None else None,
         "zone": zone, "zone_cells": [list(map(int, c)) for c in cells] if zone else None})
    out = s.store.output_path("previews", f"{label}_relief", ".png")
    if s.store.write_preview(out, rel, mask):
        s.store.record_output("previews", out, label=label, geometry_id=p.geometry_id,
                              source=scan.path)
    return scan


@_verb
def reconstruct(scan: ObjectScan, fmt: str = "ply") -> Path:
    """Relief (+ calibration) -> .ply/.xyz in output/pointclouds/."""
    s = _engine()
    out = s.store.output_path("pointclouds", scan.label, f".{fmt}")
    pointcloud.export(scan, out, fmt, _scan_calibration(scan))
    s.store.record_output("pointclouds", out, label=scan.label,
                          geometry_id=scan.geometry_id, source=scan.path)
    return out


@_verb
def measure(scan: ObjectScan) -> Dimensions:
    """H/W/L — mm with a matching calibration, else px + warning (rule 6)."""
    s = _engine()
    dims = dimensions.measure(scan, _scan_calibration(scan))
    out = s.store.output_path("measurements", scan.label, ".json")
    s.store.write_json(out, json_safe({"scan": scan.id, "known_dims": scan.known_dims,
                                       **{k: v for k, v in vars(dims).items()}}))
    s.store.record_output("measurements", out, label=scan.label,
                          geometry_id=scan.geometry_id, source=scan.path)
    return dims


@_verb
def compare(scan: ObjectScan, golden: Union[ObjectScan, CADRef],
            tol: Optional[ToleranceBand] = None) -> Verdict:
    """Deviation map + PASS/FAIL against a golden scan (CAD: build step 7)."""
    if isinstance(golden, CADRef):
        raise NotImplementedError("scan-to-CAD lands in build step 7 (golden.py port)")
    tol_warn = []
    if tol is None:
        t, tol_warn = _tolerance_for(scan)
        tol = ToleranceBand(fail_mm=t.ch_fail_mm, min_defect_px=t.min_defect_px)
    v = golden_mod.compare_relief(scan, golden, tol, _scan_calibration(scan))
    if tol_warn and v.units == "mm":
        v.warnings = list(v.warnings) + tol_warn
    s = _engine()
    out = s.store.output_path("comparisons", f"{scan.label}_vs_{golden.label}", ".json")
    s.store.write_json(out, json_safe({"scan": scan.id, "golden": golden.id,
                                       **{k: val for k, val in vars(v).items()
                                          if k != "deviation"}}))
    s.store.record_output("comparisons", out, label=scan.label, geometry_id=scan.geometry_id,
                          source=scan.path)
    if v.deviation is not None:
        png = out.with_suffix(".png")
        if s.store.write_preview(png, v.deviation, np.isfinite(v.deviation)):
            s.store.record_output("comparisons", png, label=scan.label,
                                  geometry_id=scan.geometry_id, source=scan.path)
    return v


@_verb
def calibrate(preset: Preset, samples: list) -> Calibration:
    """Fit Z (from known H) and XY (from known W/L); bind to the geometry."""
    p = _preset(preset)
    s = _engine()
    if not samples:
        raise ValueError("calibrate needs at least one scan with known_dims")
    rows = []
    fp = config.geometry_fingerprint(p)
    for sc in samples:
        rules.require_same_geometry(sc.geometry_id, p.geometry_id, "calibrate")  # rule 6
        rules.require_same_fingerprint((sc.meta or {}).get("geometry_fp"), fp,
                                       f"calibrate: scan {sc.id}")
        if not sc.known_dims:
            raise ValueError(f"scan {sc.id} has no known_dims; it can't calibrate")
        rows.append({**dimensions.raw_sample(sc), "known_dims": sc.known_dims})

    z_rows = [r for r in rows if "H" in r["known_dims"]]
    z = depth_cal.fit([r["relief_px"] for r in z_rows],
                      [r["known_dims"]["H"] for r in z_rows]) if z_rows else None
    px, mm = [], []
    for r in rows:
        if not r.get("wl_ok", True):
            continue            # footprint clipped by a zone / no base in view: W/L untrusted
        for dim, key in (("W", "W_px"), ("L", "L_px")):
            if dim in r["known_dims"]:
                px.append(r[key])
                mm.append(r["known_dims"][dim])
    xy = lateral_cal.fit(px, mm) if px else None
    if z is None and xy is None:
        raise ValueError("samples carry no usable known dims (H for Z, W/L for XY)")

    d = s.store.new_dir("calibrations", p.geometry_id)
    cal = Calibration(id=f"{p.geometry_id}/{d.name}", geometry_id=p.geometry_id,
                      mm_per_px=z["mm_per_px"] if z else None,
                      intercept_mm=z["intercept_mm"] if z else 0.0,
                      lateral_px_per_mm=xy["lateral_px_per_mm"] if xy else None,
                      r2=z["r2"] if z else None, n_samples=len(rows), path=d, fingerprint=fp)
    # supporting files first, depth_calibration.json LAST: latest_calibration
    # only ever binds a calibration whose main file is complete
    if xy:
        s.store.write_json(d / "lateral.json", json_safe(xy))
    (d / "samples").mkdir()
    s.store.write_json(d / "samples" / "samples.json", json_safe(rows))
    s.store.write_json(d / "depth_calibration.json", json_safe({
        "id": cal.id, "geometry_id": cal.geometry_id, "mm_per_px": cal.mm_per_px,
        "intercept_mm": cal.intercept_mm, "lateral_px_per_mm": cal.lateral_px_per_mm,
        "r2": cal.r2, "n_samples": cal.n_samples, "depth_fit": z, "geometry_fp": fp,
        "note": "height_mm = mm_per_px*relief_px + intercept_mm; valid only at this geometry"}))
    p.calibration = cal
    return cal


@_verb
def read_codes(preset: Optional[Preset] = None, label: str = "codes",
               levels: tuple = codes_mod.DEFAULT_LEVELS, anti_glare: bool = False,
               strip_cols: Optional[tuple] = None) -> CodeReading:
    """Read barcodes / QR codes in view, bracketing DMD illumination levels.

    Needs no base and no calibration. Stores the full-on frame + codes.json
    under library/readings/.

    anti_glare (EXPERIMENTAL): when the normal bracket reads nothing, shoot a
    many-level flood bracket (+ a DMD column-strip sweep over `strip_cols`,
    (c0, c1) DMD columns, if given), fuse it to HDR, remove the specular veil
    and decode again (engine/sense/antiglare.py).
    """
    p = _preset(preset)
    s = _engine()
    if anti_glare:
        with s.rig(p) as sync:
            frames = codes_mod.capture(sync, levels)
            ag_frames = antiglare_mod.capture(sync, strip_cols=strip_cols)
        found, full_on, inverted = codes_mod.decode(frames, levels)
        if not found:
            found = antiglare_mod.decode(ag_frames)[0]
        frames.update({f"ag_{k}": v for k, v in ag_frames.items()})
    elif hasattr(codes_mod, "capture") and hasattr(codes_mod, "decode"):
        # capture inside the rig, decode after releasing it: decoding ~8
        # frames x 3 variants takes seconds and must not leave the HDMI
        # window unpumped (or the rig locked) meanwhile
        with s.rig(p) as sync:
            frames = codes_mod.capture(sync, levels)
        found, full_on, inverted = codes_mod.decode(frames, levels)
    else:
        with s.rig(p) as sync:
            found, frames, full_on, inverted = codes_mod.read(sync, levels)
    d = s.store.new_dir("readings", label=label)
    s.store.write_preview(d / "full_on.png", frames[full_on])
    out = s.store.output_path("codes", label, ".json")
    reading = CodeReading(codes=found, illuminations=list(frames), polarity_inverted=inverted,
                          path=d, warnings=[] if found else
                          ["no code found — check focus, glare, and that the code is in the lit patch"])
    doc = {"codes": [vars(c) for c in found], "illuminations": reading.illuminations,
           "polarity_inverted": inverted, "preset": p.name, "mock": p.mock}
    s.store.write_json(d / "codes.json", doc)
    s.store.write_json(out, {**doc, "reading": d.name})
    s.store.record_output("codes", out, label=label, geometry_id=p.geometry_id, source=d)
    return reading


PROJECT_CAPTURE_MAX = 256


def _custom_patterns(patterns) -> list:
    """Validate user DMD images: (1140, 912) or (1140, 912, 3) uint8, 1..PROJECT_CAPTURE_MAX."""
    if isinstance(patterns, np.ndarray):
        # one image (H, W) / (H, W, 3), or a stack (N, H, W) / (N, H, W, 3)
        one = patterns.ndim == 2 or (patterns.ndim == 3 and patterns.shape[-1] == 3)
        pats = [patterns] if one else list(patterns)
    else:
        pats = list(patterns)
    if not 1 <= len(pats) <= PROJECT_CAPTURE_MAX:
        raise ValueError(f"project_capture takes 1..{PROJECT_CAPTURE_MAX} patterns, got {len(pats)}")
    out = []
    for i, p in enumerate(pats):
        a = np.asarray(p)
        if a.dtype != np.uint8:
            raise ValueError(f"pattern {i}: dtype must be uint8, got {a.dtype}")
        if a.shape not in ((1140, 912), (1140, 912, 3)):
            raise ValueError(f"pattern {i}: shape must be (1140, 912) or (1140, 912, 3) "
                             f"(DMD rows x cols), got {a.shape}")
        out.append(a)
    return out


@_verb
def project_capture(patterns, preset: Optional[Preset] = None, label: str = "custom",
                    raw: bool = False, average: Optional[int] = None,
                    hold_s: Optional[float] = None) -> dict:
    """Project your own DMD images and photograph each one: one camera frame per pattern, in
    order, through the normal rig (HDMI window only, never USB; the rig lock; pumped holds).
    patterns: a list / stack of (1140, 912) or (1140, 912, 3) uint8 images, at most 256.
    raw=False sends them through the preset chain (duty, polarity, gamma) like a scan; raw=True
    exactly as given. average / hold_s override camera.average / projection.min_hold_s for
    this run only. Stores frames.npz (frames: (N, H, W) float32 camera DN, + the patterns) and
    meta.json under library/readings/ -> {path, frames_npz, meta_json, n, capture_s}. No
    decoding: what the frames mean is up to the caller."""
    import time
    from dataclasses import replace as _replace
    from .library.store import safe_label
    pats = _custom_patterns(patterns)
    safe_label(label)
    p = _preset(preset)
    if average is not None:
        if isinstance(average, bool) or int(average) != average or not 1 <= int(average) <= 64:
            raise ValueError(f"average must be a whole number 1..64, got {average!r}")
        p = _replace(p, camera=_replace(p.camera, average=int(average)))
    if hold_s is not None:
        if not 0 <= float(hold_s) <= 10:
            raise ValueError(f"hold_s must be 0..10 s, got {hold_s!r}")
        p = _replace(p, projection=_replace(p.projection, min_hold_s=float(hold_s)))
    s = _engine()
    rules.check_led_duration(p.projection.led, None, "a custom pattern capture")
    t0 = time.perf_counter()
    with s.rig(p) as sync:
        frames = [np.asarray(sync.shoot(x, raw=bool(raw)), np.float32) for x in pats]
    cap = round(time.perf_counter() - t0, 2)
    d = s.store.new_dir("readings", label=label)
    with s.store.filling(d):
        same = len({x.shape for x in pats}) == 1
        npz = s.store.write_npz(d / "frames.npz", frames=np.stack(frames),
                                **({"patterns": np.stack(pats)} if same else {}))
        meta = {"kind": "custom_capture", "label": label, "preset": p.name,
                "geometry_id": p.geometry_id, "n": len(pats), "raw": bool(raw),
                "average": p.camera.average, "min_hold_s": p.projection.min_hold_s,
                "invert": p.projection.invert, "capture_s": cap, "mock": s.mock,
                "units": {"frames": "camera DN (averaged), pattern order"}}
        mj = s.store.write_json(d / "meta.json", meta)
    return {"path": d, "frames_npz": npz, "meta_json": mj, "n": len(pats), "capture_s": cap}


@_verb
def inspect(scan: ObjectScan, n_sigma: float = 6.0, min_area_px: int = 50,
            bg_window: int = 121, whole_field: bool = False) -> Inspection:
    """Defect blobs on a scan's relief; mm + PASS/FAIL vs the trigger
    (tolerance.ch_fail_mm) of the preset the SCAN was taken with, with a
    matching calibration, else px (rule 6)."""
    s = _engine()
    tol, tol_warn = _tolerance_for(scan)
    insp = inspect_mod.inspect(scan, _scan_calibration(scan), trigger_mm=tol.ch_fail_mm,
                               n_sigma=n_sigma, min_area=min_area_px, bg_window=bg_window,
                               whole_field=whole_field)
    if tol_warn and insp.units == "mm":
        insp.warnings = list(insp.warnings) + tol_warn
    out = s.store.output_path("defects", scan.label, ".json")
    s.store.write_json(out, json_safe({"scan": scan.id,
                                       **{k: ([vars(d) for d in v] if k == "defects" else v)
                                          for k, v in vars(insp).items() if k != "path"}}))
    png = inspect_mod.annotate_png(scan, insp)
    if png:
        with open(out.with_suffix(".png"), "xb") as f:
            f.write(png)
        s.store.record_output("defects", out.with_suffix(".png"), label=scan.label,
                              geometry_id=scan.geometry_id, source=scan.path)
    s.store.record_output("defects", out, label=scan.label, geometry_id=scan.geometry_id,
                          source=scan.path)
    insp.path = out
    return insp


@_verb
def delete(entry, reason: str = "") -> Path:
    """'Delete' a base / scan / calibration / reading: MOVE it to trash/.

    Recoverable — nothing is destroyed (rule 7); emptying trash/ is the
    operator's call. `entry` is a BaseRef / ObjectScan handle or a library path."""
    s = _engine()
    path = Path(getattr(entry, "path", entry))
    dest = s.store.trash(path, reason)
    cal = s.preset.calibration if s.preset is not None else None
    if cal is not None and cal.source != "preset" and \
            (cal.path is None or not Path(cal.path).exists()):
        # the active calibration was trashed: fall back to the next newest
        _rebind_calibration(s, s.preset)
    return dest


@_verb
def reject_base(base: BaseRef, reason: str) -> BaseRef:
    """Mark a base unusable (REJECTED.json) without moving it; latest_base skips it."""
    s = _engine()
    s.store.write_json(Path(base.path) / "REJECTED.json", {"reason": reason, "do_not_use": True})
    base.meta["rejected"] = reason
    return base


# ------------------------------------------------------------- 2D profiles

@_verb
def profile(scan: ObjectScan, p0=None, p1=None, width_px: int = 5, axis: str = "auto",
            label: Optional[str] = None):
    """Height along a line (camera px p0 -> p1, or axis='auto'/'x'/'y' through
    the part). mm with calibration, else px. -> measure.profile.Profile"""
    from .measure import profile as m
    return m.profile(_engine(), scan, _scan_calibration(scan), p0, p1,
                     width_px, axis, label)


# ------------------------------------------------------- good memory bank

@_verb
def bank_add(scan: ObjectScan, bank: str = "default"):
    """Add a known-good scan to a good memory bank. -> measure.bank.BankInfo"""
    from .measure import bank as m
    return m.add(_engine(), scan, bank)


@_verb
def bank_check(scan: ObjectScan, bank: str = "default", k_sigma: float = 4.0,
               min_area_px: int = 25):
    """Flag anything in `scan` not seen in the bank's good parts. -> BankVerdict"""
    from .measure import bank as m
    return m.check(_engine(), scan, bank, k_sigma, _scan_calibration(scan),
                   min_area_px)


@_verb
def bank_remove(scan: ObjectScan, bank: str = "default", reason: str = ""):
    """Remove a scan from a bank (appends a remove record; nothing deleted)."""
    from .measure import bank as m
    return m.remove(_engine(), scan, bank, reason)


# ---------------------------------------------------------- scan assistant

@_verb
def tune(preset: Optional[Preset] = None, exposures: Optional[list] = None,
         duties: Optional[list] = None, frame_us: Optional[float] = None):
    """Sweep settings, recommend exposure etc. Never edits the preset. -> TuneReport
    frame_us: video frame period for frame-locked exposures (default 1e6/60)."""
    from .capture import assist as m
    return m.tune(_engine(), _preset(preset), exposures, duties, frame_us)


@_verb
def measure_gamma(preset: Optional[Preset] = None, n_levels: int = 12):
    """Fit the projector's gray response (video-mode degamma). -> GammaReport"""
    from .capture import assist as m
    return m.measure_gamma(_engine(), _preset(preset), n_levels)


# ------------------------------------------------ projector map, grid, zones

@_verb
def capture_map(preset: Optional[Preset] = None, label: str = "map"):
    """2-axis camera<->projector map of the current scene (for grid zones)."""
    from .decode import projmap as m
    return m.capture(_engine(), _preset(preset), label)


@_verb
def project_grid(preset: Optional[Preset] = None, spacing_px: int = 96,
                 label: str = "grid", hold_s: float = 0.0, cancel=None):
    """Project a labelled grid (A1, B2, ...) and photograph it -> ZoneShot.
    cancel: optional threading.Event that ends a long hold early."""
    from .capture import zones as m
    return m.project_grid(_engine(), _preset(preset), spacing_px, label, hold_s, cancel=cancel)


@_verb
def project_zones(zones: list, preset: Optional[Preset] = None, label: str = "zones",
                  hold_s: float = 0.0, cancel=None):
    """Light custom zones with their patterns (machine-vision marking) -> ZoneShot."""
    from .capture import zones as m
    return m.project_zones(_engine(), _preset(preset), zones, None, label, hold_s,
                           cancel=cancel)


# ------------------------------------------------------------ spectral mode

@_verb
def spectral_capture(preset: Optional[Preset] = None, label: str = "spectral"):
    """R/G/B LED reflectance capture -> sense.spectral.SpectralReading"""
    from .sense import spectral as m
    return m.capture(_engine(), _preset(preset), label)


@_verb
def learn_material(reading, name: str, roi):
    """Teach a material from a camera ROI (x0, y0, x1, y1) -> Material"""
    from .sense import spectral as m
    return m.learn(_engine(), reading, name, roi)


@_verb
def classify(reading, materials: Optional[list] = None, max_distance: float = 3.0,
             brightness_weight: float = 1.0):
    """Per-pixel material map vs the learned materials -> Classification.
    brightness_weight=0 compares hue only."""
    from .sense import spectral as m
    return m.classify(_engine(), reading, materials, max_distance,
                      brightness_weight=brightness_weight)


@_verb
def separate_light(preset: Optional[Preset] = None, label: str = "light",
                   square_px: int = 16, shifts: int = 8):
    """Direct vs global (inter-reflection) light via shifted checkerboards."""
    from .sense import spectral as m
    return m.separate_light(_engine(), _preset(preset), label, square_px, shifts)


# --------------------------------------------------------- virtual sensors

@_verb
def spot_size(diameter_mm: float, mm_per_dmd_px: Optional[float] = None,
              preset: Optional[Preset] = None):
    """Laser-spot-sensor-style spot sizing: D mm at the part -> DMD px (nominal from rig
    standoff / throw ratio unless a measured mm_per_dmd_px is given) -> SpotSpec"""
    from .sense import virtual as m
    return m.spot_px_for_mm(_preset(preset), diameter_mm, mm_per_dmd_px)


@_verb
def project_spot(at, diameter_mm: float, preset: Optional[Preset] = None,
                 mm_per_dmd_px: Optional[float] = None, shape: str = "disk",
                 hold_s: float = 0.0, label: str = "spot", level: int = 255):
    """Bright spot at a grid cell ('H8'), {'dmd': [col, row]} or {'camera': [x, y]};
    photographed and measured -> SpotReading"""
    from .sense import virtual as m
    return m.project_spot(_engine(), _preset(preset), at, diameter_mm, mm_per_dmd_px,
                          shape, hold_s, label, level)


@_verb
def teach_presence(name: str, at, diameter_mm: Optional[float] = None,
                   method: str = "photoelectric", preset: Optional[Preset] = None,
                   empty_confirmed: bool = False, mm_per_dmd_px: Optional[float] = None,
                   threshold: float = 0.25, min_height_mm: float = 3.0, level: int = 255):
    """Teach a part-present sensor on the EMPTY scene. Clearing the field is a
    human step: pass empty_confirmed=True once the operator has. -> PresenceTeach"""
    from .sense import virtual as m
    return m.teach_presence(_engine(), _preset(preset), name, at, diameter_mm, method,
                            mm_per_dmd_px=mm_per_dmd_px, threshold=threshold,
                            min_height_mm=min_height_mm, level=level,
                            empty_confirmed=empty_confirmed)


@_verb
def check_presence(name: str, preset: Optional[Preset] = None,
                   threshold: Optional[float] = None, min_height_mm: Optional[float] = None,
                   min_relief_px: Optional[float] = None, label: Optional[str] = None):
    """Fire a taught sensor -> PresenceResult (PRESENT / ABSENT / INDETERMINATE)"""
    from .sense import virtual as m
    return m.check_presence(_engine(), _preset(preset), name, threshold=threshold,
                            min_height_mm=min_height_mm, min_relief_px=min_relief_px,
                            label=label)


@_verb
def teach_alignment(name: str, scan: ObjectScan, preset: Optional[Preset] = None):
    """Reference pose (centroid, angle, extents) of the part in a scan -> AlignmentRef"""
    from .sense import virtual as m
    return m.teach_alignment(_engine(), _preset(preset), name, scan)


@_verb
def check_alignment(name: str, scan: ObjectScan, tol_mm: float = 1.0, tol_deg: float = 2.0,
                    tol_px: Optional[float] = None, preset: Optional[Preset] = None):
    """dx / dy / dtheta vs the taught pose -> AlignmentResult (mm with a lateral cal)"""
    from .sense import virtual as m
    return m.check_alignment(_engine(), _preset(preset), name, scan, tol_mm, tol_deg, tol_px)


# ------------------------------------------------ 3D meshes + cell unwrap

@_verb
def export_mesh(scan, fmt: str = "stl", stride: int = 2, kind: str = "relief",
                radius_mm: Optional[float] = None, roi=None) -> Path:
    """Relief heightmap (kind='relief'), the base surface (kind='base') or the
    open cylinder arc (kind='cylinder') -> STL/OBJ + meta JSON in
    output/pointclouds/. mm only with a matching calibration (rule 6)."""
    from .reconstruct import mesh as mesh_mod
    from .measure import cell as cell_mod
    s = _engine()
    if kind == "base":
        if isinstance(scan, BaseRef):
            base = scan
        else:
            base = next((b for b in base_lib.list_bases(s.store, scan.geometry_id)
                         if b.id == scan.base_id), None)
            if base is None:
                raise ValueError(f"base {scan.base_id} of scan {scan.id} is not in the library")
        src, gid, label = base, base.geometry_id, f"{base.label}_base"
    else:
        src, gid, label = scan, scan.geometry_id, f"{scan.label}_{kind}"
    out = s.store.output_path("pointclouds", label, f".{fmt}")
    if kind == "relief":
        p, mp, _ = mesh_mod.heightmap_to_stl(scan, _scan_calibration(scan), out, stride, roi, fmt)
    elif kind == "base":
        p, mp, _ = mesh_mod.base_to_stl(src, out, stride, roi, fmt)
    elif kind == "cylinder":
        u = cell_mod.unwrap_cell(scan, _scan_calibration(scan), radius_mm)
        p, mp, _ = mesh_mod.cylinder_arc_mesh(u, out, fmt)
    else:
        raise ValueError("kind must be 'relief', 'base' or 'cylinder'")
    for f in (p, mp):
        s.store.record_output("pointclouds", f, label=label, geometry_id=gid,
                              source=getattr(src, "path", None))
    return p


@_verb
def cell_unwrap(scan, radius_mm: Optional[float] = None, step: Optional[float] = None):
    """Cylinder fit + (theta, z) unwrap of a standing cell / tube -> CellUnwrap
    (open arc only, dent map, cell height); saves npz + heatmap preview."""
    from .measure import cell as cell_mod
    s = _engine()
    u = cell_mod.unwrap_cell(scan, _scan_calibration(scan), radius_mm, step)
    npz = s.store.output_path("previews", f"{scan.label}_unwrap", ".npz")
    cell_mod.save_unwrap(npz, u)
    s.store.record_output("previews", npz, label=scan.label, geometry_id=scan.geometry_id,
                          source=scan.path)
    png = cell_mod.dent_png(u)
    if png:
        pp = npz.with_suffix(".png")
        with open(pp, "xb") as f:
            f.write(png)
        s.store.record_output("previews", pp, label=scan.label, geometry_id=scan.geometry_id,
                              source=scan.path)
    return u


@_verb
def cell_dents(scan, radius_mm: Optional[float] = None, n_sigma: float = 5.0,
               min_area_bins: int = 6, depth_trigger_mm: Optional[float] = None):
    """Dents on the unwrapped cell; mm verdict vs the trigger (rule 6)."""
    from .measure import cell as cell_mod
    s = _engine()
    tol, tol_warn = _tolerance_for(scan)
    trig = depth_trigger_mm if depth_trigger_mm is not None else tol.ch_fail_mm
    u = cell_mod.unwrap_cell(scan, _scan_calibration(scan), radius_mm)
    rep = cell_mod.find_dents(u, trig, n_sigma=n_sigma, min_area=min_area_bins)
    if tol_warn and rep.units == "mm" and depth_trigger_mm is None:
        rep.warnings = list(rep.warnings) + list(tol_warn)
    out = s.store.output_path("defects", f"{scan.label}_cell", ".json")
    s.store.write_json(out, json_safe(cell_mod.report_dict(u, rep)))
    png = cell_mod.dent_png(u, rep)
    if png:
        with open(out.with_suffix(".png"), "xb") as f:
            f.write(png)
        s.store.record_output("defects", out.with_suffix(".png"), label=scan.label,
                              geometry_id=scan.geometry_id, source=scan.path)
    s.store.record_output("defects", out, label=scan.label, geometry_id=scan.geometry_id,
                          source=scan.path)
    return rep


# --------------------------------------------- surface maps + checkerboard

@_verb
def bake_maps(scan: ObjectScan, spectral=None, light=None, golden=None,
              sigma_px: float = 2.0, label: Optional[str] = None):
    """Aligned per-pixel map stack (height, cavity, normal, validity, density,
    + albedo / subsurface / dust when given) -> output bin."""
    from .measure import maps
    return maps.bake_maps(_engine(), scan, spectral=spectral, light=light, golden=golden,
                          calibration=_scan_calibration(scan), sigma_px=sigma_px, label=label)


@_verb
def checker_scan(preset: Optional[Preset] = None, square_px: int = 32, shifts: int = 1,
                 label: str = "checker", reference=None):
    """Project a checker, find its corners (camera px <-> DMD mirrors), local Jacobian."""
    from .bases.checkerboard import capture_checker
    return capture_checker(_engine(), _preset(preset), square_px=square_px, shifts=shifts,
                           label=label, reference=reference)


# ------------------------------------------------- Surfer atlas + methods

@_verb
def import_atlas(path: str, name: str, preset: Optional[Preset] = None):
    """Copy a Surfer atlas JSON into library/atlases/<geometry>/<name>/ (append-only).
    Its regions then work as zones: zone='atlas:<name>/<region>'."""
    from .capture import atlas as m
    return m.import_atlas(_engine(), path, name, _preset(preset).geometry_id)


@_verb
def list_atlases(geometry_id: Optional[str] = None):
    from .capture import atlas as m
    return m.list_atlases(_engine(), geometry_id)


@_verb
def flood_verify(region, preset: Optional[Preset] = None, level: int = 255,
                 label: str = "flood_verify"):
    """Light only `region` (optically black elsewhere), photograph, report where it lands."""
    from .capture import surfer as m
    return m.flood_verify(_engine(), _preset(preset), region, level, label)


@_verb
def anti_glare_sweep(region, carriers=None, preset: Optional[Preset] = None, steps: int = 4,
                     cycles: float = 6.0, level: int = 255, label: str = "anti_glare"):
    """Pick the tilted-carrier pattern that least clips glare in `region`
    (pattern choice, not optical wavefront shaping - the LEDs are incoherent)."""
    from .capture import surfer as m
    return m.anti_glare_sweep(_engine(), _preset(preset), region, carriers, steps, cycles,
                              level, label)


@_verb
def multi_orientation_scan(region, preset: Optional[Preset] = None,
                           orientations=("v", "h", "d45", "d135"), steps: int = 4,
                           period_px: float = 24, level: int = 255,
                           label: str = "multi_orient"):
    """H/V/diagonal fringes in `region`, fused by modulation (quality maps)."""
    from .capture import surfer as m
    return m.multi_orientation_scan(_engine(), _preset(preset), region, orientations, steps,
                                    period_px, level, label)


# ------------------------------------------- flash / triggered pattern mode

@_verb
def make_flash_pack(preset: Optional[Preset] = None, axes: tuple = ("x",),
                    video_port: bool = False, label: Optional[str] = None,
                    period_us: Optional[int] = None, exposure_us: Optional[int] = None,
                    trigger_mode: str = "trigout") -> Path:
    """Build a DLPC350 flash / video-port pattern pack (files only; never touches the rig).
    One axis per pack (the controller holds 2 images ready). period_us / exposure_us: a
    uniform pattern timing, trigger_mode 'trigout' | 'external'; infeasible DLPC350 /
    camera timing is refused (TimingError)."""
    import sys as _sys
    root = str(Path(__file__).resolve().parents[1])
    if root not in _sys.path:
        _sys.path.insert(0, root)
    from tools.make_flash_pack import make_pack, DEFAULT_OUT
    p = _preset(preset)
    return make_pack(p.name, axes=tuple(axes), video_port=video_port,
                     out_root=DEFAULT_OUT, label=label, period_us=period_us,
                     exposure_us=exposure_us, trigger_mode=trigger_mode)


@_verb
def make_fastscan_pack(zone_rings: tuple = (), zone_masks: tuple = (),
                       band_cam: Optional[str] = None, band_field: Optional[str] = None,
                       margin_px: int = 6, calibration: Optional[str] = None,
                       study: bool = False, label: Optional[str] = None,
                       period_us: Optional[int] = None, trigger_mode: str = "freerun",
                       pattern_exposure_us: Optional[int] = None,
                       camera_exposure_us: Optional[int] = None) -> Path:
    """Build the 1-bit soft-sync fast-scan pack (files only; never touches the rig):
    one 24-bit flash image full field + one zone-masked when zone sources are given
    (face_dmd_ring.npy rims -> ellipse + margin_px, DMD mask .npy files, a camera band
    mapped by a both-axes field.npz) -> output/flashpacks/<ts>_<label>/.
    A human loads it and plays it in the 4500 MKII GUI (docs/FLASH_FASTSCAN_4500.md).
    period_us (default 10000) / pattern_exposure_us / trigger_mode 'freerun' (soft sync) |
    'trigout' | 'external' (hardware trigger, docs/TRIGGER_FULLSPEED_TEST.md); infeasible
    timing is refused (TimingError)."""
    import sys as _sys
    root = str(Path(__file__).resolve().parents[1])
    if root not in _sys.path:
        _sys.path.insert(0, root)
    from tools.make_flash_pack import make_fastscan_pack as _make, FASTSCAN_PERIOD_US, \
        FASTSCAN_CAMERA_EXPOSURE_US
    s = _engine()
    return _make(period_us=int(period_us or FASTSCAN_PERIOD_US), trigger_mode=trigger_mode,
                 pattern_exposure_us=pattern_exposure_us,
                 camera_exposure_us=int(camera_exposure_us or FASTSCAN_CAMERA_EXPOSURE_US),zone_rings=tuple(zone_rings or ()), zone_masks=tuple(zone_masks or ()),
                 band_cam=band_cam, band_field=band_field, margin_px=int(margin_px),
                 calibration=calibration, study=bool(study), label=label,
                 library=s.store.root,
                 geometry_id=s.preset.geometry_id if s.preset is not None else None)


@_verb
def fast_scan_flash(pack: Optional[str] = None, variant: Optional[str] = None,
                    label: str = "flash", preset: Optional[Preset] = None,
                    zone: Optional[str] = None, roi: Optional[list] = None,
                    loops: Optional[int] = None, exposure_us: Optional[float] = None,
                    fps: Optional[float] = None, gain_db: Optional[float] = None,
                    method: str = "plane", keep_recording: bool = True,
                    dry_run: bool = False, fine_prefilter_px: Optional[float] = None) -> dict:
    """Fast metric scan from the DLPC350 flash loop: the fast-scan pack's 1-bit patterns
    play on the controller's internal timer (loaded with the 4500 MKII GUI's firmware link
    and played in its Pattern Sequence tab BY A HUMAN), the camera free-runs faster than
    the pattern rate (roi / zone window, fixed exposure, every frame kept) and the frames
    are sorted in software (soft sync) -> column multifreq decode -> 'plane' triangulation
    -> a metric scan (meta mode 'flash_softsync', frame stats). Camera only: the projector
    is never touched. Refused (SequenceError) when the frames do not show the pack's loop
    playing in order; rule 6 before the camera. dry_run: the timing plan only.
    fine_prefilter_px: optional camera-px low-pass of the fine level (1 = the pack's
    recommendation at a sharp projector; default off) (docs/FLASH_SOFTSYNC.md)."""
    from .reconstruct import flash_scan
    return flash_scan.fast_scan_flash(_engine(), _preset(preset), pack, variant, label, zone,
                                      roi, loops, exposure_us, fps, gain_db, method,
                                      bool(keep_recording), bool(dry_run),
                                      fine_prefilter_px=fine_prefilter_px)


@_verb
def trigger_test(line: str = "Line0", seconds: float = 5.0,
                 expect_period_ms: Optional[float] = None, exposure_us: float = 1000.0,
                 preset: Optional[Preset] = None) -> dict:
    """EXPERIMENTAL hardware-trigger check, camera only (the projector is untouched: a
    HUMAN plays the pattern sequence with TRIG_OUT_1 in the 4500 MKII GUI). The camera
    is put in hardware trigger on `line` (FrameStart, RisingEdge, short exposure),
    triggered frames are counted for `seconds` (timeouts count zero) -> count, rate,
    period mean / std / min / max from camera timestamps, missed pulses vs
    expect_period_ms and the verdict NO PULSES | RATE MISMATCH | OK. TriggerMode is
    set back to Off at the end, always (docs/TRIGGER_FULLSPEED_TEST.md)."""
    from .reconstruct import trigger_scan
    return trigger_scan.trigger_test(_engine(), _preset(preset), line, float(seconds),
                                     expect_period_ms, float(exposure_us))


@_verb
def scan_trigger(pack: str, variant: Optional[str] = None, label: str = "trigger",
                 preset: Optional[Preset] = None, line: str = "Line0", repeats: int = 1,
                 exposure_us: Optional[float] = None, acknowledge: Optional[str] = None,
                 first_timeout_s: float = 30.0) -> dict:
    """EXPERIMENTAL hardware-triggered metric scan of a flash pack (the 8-bit hybrid pack
    or the 1-bit --fastscan pack): TRIG_OUT_1 -> camera `line`, exactly one full frame
    per pattern, `repeats` loops (missing / extra frames, spacing +-50 % and the first
    trigger checked), aligned (1-bit: the sync markers; 8-bit: trigger order), decoded
    by the existing decoders -> 'plane' triangulation -> a metric scan (meta mode
    'flash_trigger', capture timing). Refused (PatternModeRefused) unless acknowledge
    == patternmode.ACK_PHRASE: a HUMAN put the projector in pattern mode. Rule 6 before
    the camera; the projector is never touched (docs/TRIGGER_FULLSPEED_TEST.md)."""
    from .reconstruct import trigger_scan
    return trigger_scan.scan_trigger(_engine(), _preset(preset), pack, variant, label, line,
                                     int(repeats), exposure_us, acknowledge,
                                     first_timeout_s=float(first_timeout_s))


@_verb
def sort_flash_recording(recording: str, pack: Optional[str] = None,
                         variant: Optional[str] = None, decode: bool = False,
                         label: Optional[str] = None,
                         fine_prefilter_px: Optional[float] = None) -> dict:
    """Offline soft sync: a recording saved by fast_scan_flash (output/flashscans/*.npz) ->
    per-pattern images + sort report -> output/flashscans/<ts>_<label>.npz; decode=True
    also the projector-column map (no mm). Refused (SequenceError) like the scan. Never
    touches the rig."""
    from .reconstruct import flash_scan
    s = _engine()
    return flash_scan.sort_flash_recording(
        s, recording, pack, variant, bool(decode), label,
        s.preset.geometry_id if s.preset is not None else None,
        fine_prefilter_px=fine_prefilter_px)


# -------------------------------- metric calibration + measurement (mm)
#
# Stereo calibration (camera + projector as an inverse camera) turns decoded
# projector coordinates into a point cloud in mm. Rule 6: every mm below
# comes from a stereo calibration at the scan's geometry_id AND geometry
# fingerprint, read from the library at use time; without one the verbs
# refuse (GeometryMismatch) before the rig is touched.

@_verb
def make_charuco(squares_x: int = 10, squares_y: int = 7, square_mm: float = 25.0,
                 marker_mm: float = 18.0, dictionary: str = "DICT_5X5_100", dpi: float = 600.0,
                 label: str = "charuco") -> dict:
    """Printable ChArUco board (PNG + true-size PDF + spec JSON) -> output/calibration/.
    Print at 100 %, then measure a square with calipers (square_mm for calibrate_stereo)."""
    from .calibrate import flow
    from .calibrate.charuco import BoardSpec
    spec = BoardSpec(int(squares_x), int(squares_y), float(square_mm), float(marker_mm),
                     str(dictionary))
    return flow.make_charuco(_engine(), spec, float(dpi), label)


@_verb
def row_parity(preset: Optional[Preset] = None, label: str = "parity") -> dict:
    """Bench step B3: which DMD rows sit half a column right (diamond DLP4500).
    Stored with the geometry; scans and calibration poses then draw straight fringes."""
    from .calibrate import flow
    return flow.row_parity(_engine(), _preset(preset), label)


@_verb
def capture_calib_pose(preset: Optional[Preset] = None, set_label: str = "stereo",
                       board: Optional[dict] = None, steps: Optional[int] = None,
                       periods: Optional[list] = None, diamond_parity="auto") -> dict:
    """One ChArUco board pose for calibrate_stereo: flood + both-axis multifreq decode.
    The operator places / holds the board (12-20 poses, tilts +-30 deg, fill the volume)."""
    from .calibrate import flow
    return flow.capture_pose(_engine(), _preset(preset), set_label, flow.board_spec(board),
                             steps, periods, diamond_parity)


@_verb
def calibrate_stereo(preset: Optional[Preset] = None, set_label: Optional[str] = None,
                     square_mm: Optional[float] = None, board: Optional[dict] = None,
                     projector_distortion: bool = False, mock_poses: int = 0,
                     steps: Optional[int] = None, periods: Optional[list] = None,
                     diamond_parity="auto") -> dict:
    """Camera + projector (inverse camera) + stereoCalibrate from the pose set; saved as a
    geometry-bound calibration. mock_poses=N (mock only) places N board poses itself."""
    from .calibrate import flow
    cal, summary = flow.calibrate_stereo(_engine(), _preset(preset), set_label, square_mm,
                                         board, projector_distortion, int(mock_poses or 0),
                                         steps, periods, diamond_parity)
    return {**summary, "calibration": cal}


@_verb
def scan_metric(preset: Optional[Preset] = None, label: str = "metric", method: str = "auto",
                steps: Optional[int] = None, periods: Optional[list] = None,
                max_consistency_mm: Optional[float] = None, zone: Optional[str] = None):
    """Both-axis multifreq scan -> triangulated point cloud in mm (+ .ply, consistency map).
    Refused without a stereo calibration at this geometry + fingerprint (rule 6).
    zone: light and keep only these grid cells ("F11:F12") or an atlas region."""
    from .reconstruct import metric_scan
    return metric_scan.scan_metric(_engine(), _preset(preset), label, method, steps, periods,
                                   max_consistency_mm, zone)


@_verb
def triangulate(scan, method: str = "auto", label: Optional[str] = None) -> dict:
    """Re-triangulate a stored scan (capture_object's ObjectScan or a MetricScan) with the
    current stereo calibration of its geometry -> new metric scan + .ply (mm)."""
    from .reconstruct import metric_scan
    return metric_scan.triangulate_scan(_engine(), scan, method, label)


@_verb
def cylinder_deviation(scan, radius_mm: Optional[float] = None, n_sigma: float = 4.0,
                       min_depth_mm: float = 0.05, roi=None):
    """Cylinder fit on a metric scan -> radial deviation map (mm, dents negative) + dent
    blobs with depth and area -> output/defects (json, png, npz)."""
    from .reconstruct import metric_scan
    dev, files = metric_scan.cylinder_deviation(_engine(), scan, radius_mm, n_sigma,
                                                min_depth_mm, roi=roi)
    dev.files = files
    return dev


@_verb
def compare_metric(scan, golden, max_dist_mm: float = 5.0, n_sigma: float = 4.0) -> dict:
    """Point-to-plane ICP of a metric scan onto a golden metric scan, then the signed
    deviation (mm) and its below-surface blobs -> output/comparisons."""
    from .reconstruct import metric_scan
    rep, _ = metric_scan.compare_metric(_engine(), scan, golden, max_dist_mm, n_sigma)
    return rep


@_verb
def accuracy_check(kind: str, known: Optional[dict] = None, scans: Optional[list] = None,
                   repeats: int = 3, preset: Optional[Preset] = None, tol_mm: float = 0.1,
                   repeat_tol_mm: float = 0.03, label: str = "accuracy",
                   steps: Optional[int] = None, periods: Optional[list] = None) -> dict:
    """Known reference (plane | step {height_mm} | cylinder / sphere {diameter_mm}) over
    `repeats` new metric scans (or the given ones) -> mean / max error, repeatability, GO."""
    from .reconstruct import metric_scan
    p = _preset(preset) if scans is None else (preset or _engine().preset)
    return metric_scan.accuracy_check(_engine(), p, kind, known or {}, scans, repeats, tol_mm,
                                      repeat_tol_mm, label, steps, periods)


@_verb
def predict_precision(preset: Optional[Preset] = None, z_mm: Optional[float] = None,
                      A_dn: Optional[float] = None, B_dn: Optional[float] = None,
                      period_px: Optional[float] = None, steps: Optional[int] = None,
                      frames_averaged: Optional[int] = None, gain_db: Optional[float] = None,
                      scan=None, extra_noise_dn: float = 0.0) -> dict:
    """Predicted depth noise sigma_z (mm) before scanning (Lv & Kemao). Baseline / f_p
    from the stereo calibration (else the preset's nominal rig); A / B from a metric scan."""
    from .calibrate import stereo as stereo_mod
    from .measure import precision
    p = _preset(preset)
    s = _engine()
    cal = stereo_mod.latest(s.store, p.geometry_id, config.geometry_fingerprint(p))
    notes = []
    if cal is not None:
        b, fp = cal.baseline_mm, float(cal.K_proj[0, 0])
    else:
        if not p.rig.baseline_mm:
            raise ValueError("no stereo calibration and no rig.baseline_mm in the preset")
        b, fp = float(p.rig.baseline_mm), float(p.rig.throw_ratio) * p.projection.size[0]
        notes.append("nominal geometry (no stereo calibration): f_p = throw_ratio x 912")
    spec = next((x for x in p.bases if x.get("name") == "multifreq"), {})
    if scan is not None:
        dec = (scan.meta or {}).get("decode") or {}
        ax = (dec.get("per_axis") or {}).get("x", {})
        A_dn = A_dn if A_dn is not None else ax.get("offset_median")
        B_dn = B_dn if B_dn is not None else ax.get("modulation_median")
        period_px = period_px or (dec.get("periods_px") or [None])[0]
        steps = steps or dec.get("steps")
    if A_dn is None or B_dn is None:
        raise ValueError("A_dn and B_dn (fringe offset / modulation, grey levels) are needed, "
                         "or pass a metric scan to read them from")
    period_px = period_px or (spec.get("periods_px") or [8])[0]
    steps = steps or spec.get("steps", 12)
    z = z_mm or p.rig.standoff_mm or 823.0
    gdb = gain_db if gain_db is not None else p.camera.gain_db
    out = precision.predict_sigma_z(float(z), b, fp, float(period_px), float(A_dn), float(B_dn),
                                    int(steps), int(frames_averaged or p.camera.average),
                                    float(gdb), extra_noise_dn=float(extra_noise_dn))
    out.update({"calibration_id": cal.id if cal is not None else None, "notes": notes})
    return out


@_verb
def export_metric_mesh(scan, fmt: str = "stl", stride: int = 1,
                       max_edge_mm: Optional[float] = None, foreground_only: bool = False) -> Path:
    """Metric scan -> STL / OBJ in mm (camera frame), open surface only (depth jumps are
    left open, nothing closed or invented) + meta JSON -> output/pointclouds/."""
    from .reconstruct import metric_scan
    p, _, _ = metric_scan.export_metric_mesh(_engine(), scan, fmt, int(stride), max_edge_mm,
                                             bool(foreground_only))
    return p


# ------------------------------------ dent detection (shiny cylindrical cells)
#
# A dent set is ONE pose of a cell under two exposures (+ one metric scan of
# the same zone). Four finders (pattern warp, photometric, HDR phase depth,
# reflected-line kink) are calibrated on a KNOWN-GOOD cell of the same type in
# the same fixture (calibrate_dent_null: the negative control) and fused by
# detect_dents. Depth, size and the cell coordinates are mm only with the
# stereo calibration the metric scans were made with (rule 6); detection
# refuses without it. Legacy offline dent sets (a folder with dataset.json)
# are accepted wherever a dent set is.

@_verb
def capture_dent_set(zone: Optional[str] = None, label: str = "dentset",
                     long_preset: Optional[Preset] = None, short_preset: Optional[Preset] = None,
                     short_exposure_us: Optional[int] = None, lines_v: bool = True,
                     metric: bool = True, metric_preset: Optional[Preset] = None,
                     steps: Optional[int] = None, periods: Optional[list] = None,
                     pattern_set: str = "production"):
    """One pose of a cell for the dent detector: long pass (the dark preset, 66 667 us) with
    floods 255/0 + sin0 T12, sin0 T36, sin90 T12, sin90 T36 (8 steps each), short pass
    (16 667 us) with floods 0/128/255 + the same sinusoids (+ lines_v x10), one rig session
    each, then ONE metric scan of the same zone -> library/dentsets/<geometry>/.
    pattern_set "minimal" drops sin90 T36 (fewer frames, weaker on small dents).
    long_preset defaults to the active preset; without short_preset the short pass is the
    long preset at short_exposure_us (default 16 667 us); a short_preset must differ from
    the long one ONLY in exposure (gain, duty, gamma, LED... refused). Refused before the
    rig without a stereo calibration when metric=True (rule 6). If the metric scan fails
    the pattern frames are still saved (metric_scans [], a warning): scan_metric the same
    zone and pass it to detect_dents as metric_scans."""
    from .measure.dents import dentset as ds_mod
    from .measure.dents import verbs as dent_verbs
    lp = _preset(long_preset)
    if short_preset is not None:
        sp = _preset(short_preset)
        if short_exposure_us is not None:
            sp = dent_verbs.with_exposure(sp, short_exposure_us)
    else:
        sp = dent_verbs.with_exposure(lp, short_exposure_us or ds_mod.SHORT_US_DEFAULT)
    mp = _preset(metric_preset) if metric_preset is not None else lp
    return dent_verbs.capture_dent_set(_engine(), lp, sp, label, zone, bool(lines_v), bool(metric),
                                       mp, steps, periods, pattern_set)


@_verb
def calibrate_dent_null(dentsets: list, cell_diameter_mm: float, label: str = "null",
                        metric_scans: Optional[list] = None, fixture: Optional[str] = None,
                        subset: str = "all") -> dict:
    """Negative control: run the dent finders on dent set(s) of KNOWN-GOOD cells of one type
    in one fixture; each finder's maximum on the good face and the fused maximum are stored
    as a geometry-bound calibration (calibrations/<geometry>/<ts>_dentnull_<label>/), bound
    to the cell diameter, the fixture (default: the dent sets' zone; a named fixture must
    then be passed to detect_dents too), the pattern set and the stereo calibration. A
    finder with no response on the good cell (e.g. no mirrored room line) is excluded, with
    a warning. A dent set is a DentSet, a dentset id or a folder (legacy dataset.json
    accepted, then metric_scans names its metric scans). subset: all | production |
    production_nolines | minimal (the frames the finders use; detect_dents must use the
    same)."""
    from .measure.dents import verbs as dent_verbs
    return dent_verbs.calibrate_dent_null(_engine(), list(dentsets), float(cell_diameter_mm), label,
                                          metric_scans, fixture, subset)


@_verb
def detect_dents(dentset, metric_scans: Optional[list] = None,
                 cell_diameter_mm: Optional[float] = None, null_calibration=None,
                 label: Optional[str] = None, subset: str = "all",
                 fixture: Optional[str] = None):
    """Dents / scratches on a cell -> DentReport: candidates with camera px, (theta, s) on the
    cell, fused score (1.0 = the good cell's maximum), level (detection > 1.25, watch
    0.8-1.25, edge), depth in mm (+ = pushed in, depth_reliable false in specular / dark-band
    zones, depth_significant above 1.5 x the scan's depth noise p90), size, per-finder
    evidence and kind (dent / scratch / speck); overlay PNG, json and maps ->
    output/defects. null_calibration defaults to the newest one matching the geometry, the
    cell diameter (None: the matching nulls' one diameter), the fixture (default: the dent
    set's zone), the pattern set and the current stereo calibration. Refused without the
    metric scans' stereo calibration, or with a null made under another one (rule 6)."""
    from .measure.dents import verbs as dent_verbs
    return dent_verbs.detect_dents(_engine(), dentset, metric_scans, cell_diameter_mm,
                                   null_calibration, label, subset, fixture)


# ------------------------------------------ end faces (a cylinder's circular end face)
#
# A cell lying with its END FACE toward the camera + projector (bench: 21.05 mm face ~795 mm
# away, tilted 7-23 deg). locate_face finds the face in a metric scan (depth, the true rim
# against far background, fixed-radius circle, camera -> DMD affine on the decoded face),
# aim_ring / check_ring project its ring + cross (placement guide / check), auto_intensity
# picks projection.duty, targeted_face_scan lights only the face (N repeats + polar set),
# face_profile maps it in mm, face_golden / check_face compare a placement with good ones.
# mm only with the stereo calibration the scans were made with, still current (rule 6);
# targets, scans and goldens are bound to the geometry + fingerprint (rule 5).

@_verb
def locate_face(scan, face_diameter_mm: float = 21.05, label: str = "face", roi=None,
                params: Optional[dict] = None, flood=None):
    """Locate a circular end face in a metric scan (both axes decoded; scan_metric with
    method='plane' and a zone around the face) -> FaceTarget in faces/<geometry>/: face plane,
    centre, fixed-radius rim circle (face_diameter_mm), rim residual, tilt, the camera face
    mask + ellipse, and the camera -> DMD affine fitted on the decoded face pixels (residual in
    DMD px; warns above 0.5) with the aim ring in native DMD px. The face is the nearest
    face-shaped depth-continuous surface (nearer non-face objects are set aside); roi
    (x0, y0, x1, y1 camera px) narrows the search; params: LocateParams fields of
    engine.measure.endface.locate (near_band_mm 12, far_gap_mm 30, ...). A neighbour a few mm
    behind the face (a cooling tube) merges with it in depth: then the face is split off by its
    robust plane (face_band_mm 1.0) and its rim is where surface lies near_gap_mm (2.0) behind
    that plane; if the depth rim test still fails, the face outline in `flood` (None = the
    scan's own flood pair, else its decode modulation; an array / file / another metric scan)
    is used (meta rim_source; a warning says so). Refused without the scan's stereo
    calibration (rule 6). Never touches the rig."""
    from .measure.endface import verbs as face_verbs
    return face_verbs.locate_face(_engine(), scan, float(face_diameter_mm), label, roi, params,
                                  flood)


@_verb
def aim_ring(face=None, preset: Optional[Preset] = None, hold_s: float = 0.0, label: str = "aim",
             offset_dmd=None, cancel=None):
    """Project the ring + cross of a face target (default: the newest at the preset's
    geometry) and photograph it -> RingShot (photo + overlay in output/previews). hold_s keeps
    it up afterwards as the operator's placement guide (pumped window hold, never a sleep;
    `cancel` = threading.Event ends it early). offset_dmd = [cols, rows] shifts the ring."""
    from .measure.endface import verbs as face_verbs
    return face_verbs.project_ring(_engine(), _preset(preset), face, "aim", hold_s, label,
                                   offset_dmd, cancel=cancel)


@_verb
def check_ring(face=None, preset: Optional[Preset] = None, hold_s: float = 0.0,
               label: str = "ring_check", tol_px: float = 1.5, offset_dmd=None, cancel=None):
    """Is the face still where locate_face found it? A flood through the DMD face ellipse must
    light >= 90 % of the located camera face mask with <= 10 % spilling outside it (a face
    that slid or tilted), and the ring + cross, photographed, must land on the located rim
    within tol_px camera px (a face that moved toward / away from the camera: parallax) ->
    RingShot verdict ON_FACE | OFF | NOT_FOUND, photo + overlay (located face orange, ring as
    photographed green) in output/previews; optional pumped hold as for aim_ring."""
    from .measure.endface import verbs as face_verbs
    return face_verbs.project_ring(_engine(), _preset(preset), face, "check", hold_s, label,
                                   offset_dmd, float(tol_px), cancel)


@_verb
def auto_intensity(face=None, preset: Optional[Preset] = None, target_dn: float = 200.0,
                   tol_dn: float = 10.0, max_shots: int = 4, label: str = "auto_intensity") -> dict:
    """Pick projection.duty so the face's p99 is ~target_dn camera DN at the preset's
    (whole-frame) exposure, from a dark + a few flood shots lit only through the face mask ->
    {duty, verified, p99_dn, shots, warnings} (+ output/tuning JSON). Never edits the preset
    and never touches the LED currents (no USB): pass the duty to targeted_face_scan. Needs
    projection.invert false."""
    from .measure.endface import verbs as face_verbs
    return face_verbs.auto_intensity(_engine(), _preset(preset), face, float(target_dn),
                                     float(tol_dn), int(max_shots), label=label)


@_verb
def targeted_face_scan(face=None, preset: Optional[Preset] = None, label: str = "face",
                       repeats: int = 3, margin_dmd_px: float = 6.0, polar: bool = True,
                       duty: Optional[float] = None, steps: Optional[int] = None,
                       periods: Optional[list] = None, polar_steps: int = 8):
    """Targeted end-face scan: the DMD lights only the face ellipse (+ margin_dmd_px); per
    repeat the multifreq depth set (x + y) triangulated ray-plane (column axis; the calibration's
    row axis is not trusted) -> a metric scan, and (polar) concentric rings (0.08 / 0.25 rim
    radii) + spokes (24 / 6 per turn), polar_steps each, + the flood pair (always: the MF
    detector's reflectance FIR needs it), saved as camera frames -> FaceScan in
    facescans/<geometry>/ naming its repeat metric scans. duty overrides
    projection.duty for this scan only (auto_intensity). Refused before the rig without a
    stereo calibration (rule 6) or with a face target of another geometry (rule 5)."""
    from .measure.endface import verbs as face_verbs
    return face_verbs.targeted_face_scan(_engine(), _preset(preset), face, label, int(repeats),
                                         float(margin_dmd_px), bool(polar), duty, steps, periods,
                                         int(polar_steps))


@_verb
def face_profile(scans, face=None, label: Optional[str] = None, grid_mm: float = 0.25,
                 r_flat_mm: float = 9.0, stl: bool = True):
    """Height map of one placement in the face frame (u ~ camera right, v ~ camera down, h +
    toward the camera, mm on a grid_mm grid) from a FaceScan (or its repeat metric scans + a
    face target) -> FaceProfile: flatness (RMS, p2-p98 inside r_flat_mm), per-point
    repeatability across the repeats, tilt, radial profile, horizontal / vertical sections;
    PNG + CSV + JSON + NPZ in output/profiles and an open-surface STL of the face (mm, face
    frame) in output/pointclouds. mm only with the scans' stereo calibration, still current
    (rule 6)."""
    from .measure.endface import verbs as face_verbs
    return face_verbs.face_profile(_engine(), scans, face, label, float(grid_mm),
                                   float(r_flat_mm), bool(stl))


@_verb
def face_golden(placements: list, label: str = "golden", k_sigma: float = 4.0,
                floor_mm: float = 0.10, smooth_cells: int = 5, rim_r_mm: float = 9.75,
                rim_factor: float = 2.0, rim_min_mm: float = 0.3, height_smooth_mm: float = 0.25,
                min_area_mm2: float = 0.25, synthetic=True, detector: str = "mf_fir",
                z_thresh: float = 5.5, min_depth_mm: float = 0.0, fir: bool = True,
                mf_params: Optional[dict] = None):
    """Golden end face from >= 2 (3+ for leave-one-out) GOOD placements (FaceProfiles,
    face-profile .npz files or FaceScans) -> FaceGolden in face_golden/<geometry>/
    (append-only), bound to geometry + fingerprint, face diameter, grid and stereo
    calibration. It holds every detector's model; `detector` is the default one:
    'mf_fir' (default): MF-CFAR - a dent-shaped matched filter bank (1 / 1.5 / 2 / 3 mm) on
    the deviation from the placements' median RADIAL profile, z against the face's own robust
    noise, trigger at z >= z_thresh (5.5 ~ 1e-3 false per good face, 5.0 ~ 1e-2) and depth >=
    min_depth_mm, r <= 9.75 mm; with the reflectance FIR (fir: 7x7 camera-px kernel from the
    targeted scans' flood pair, fitted on these placements, its coefficients stored here);
    'mf': the same without the FIR; 'band': median map + max(k_sigma x sigma (median over
    smooth_cells), floor_mm), rim zone r > rim_r_mm x rim_factor (min rim_min_mm), maps
    Gaussian-smoothed height_smooth_mm, triggers >= min_area_mm2. mf_params: other MF-CFAR
    settings (engine.measure.endface.mfcfar.MF_DEFAULTS). Records leave-one-out false triggers
    of every detector and the synthetic-dent battery of the default one (synthetic: True, False
    or {depths_mm, widths_mm, positions, seed}; dents injected per camera pixel)."""
    from .measure.endface import verbs as face_verbs
    mf = dict(mf_params or {})
    mf.update(z_thresh=float(z_thresh), min_depth_mm=float(min_depth_mm))
    return face_verbs.face_golden(_engine(), list(placements), label,
                                  {"k_sigma": k_sigma, "floor_mm": floor_mm,
                                   "smooth_cells": int(smooth_cells), "rim_r_mm": rim_r_mm,
                                   "rim_factor": rim_factor, "rim_min_mm": rim_min_mm,
                                   "height_smooth_mm": height_smooth_mm,
                                   "min_area_mm2": min_area_mm2},
                                  synthetic if isinstance(synthetic, dict) else bool(synthetic),
                                  detector, bool(fir), mf)


@_verb
def check_face(placement, golden=None, label: Optional[str] = None,
               min_area_mm2: Optional[float] = None, detector=None,
               z_thresh: Optional[float] = None, min_depth_mm: Optional[float] = None):
    """A placement (FaceProfile, face-profile .npz or FaceScan) vs a golden face (default: the
    newest matching geometry, face diameter, grid and stereo calibration) -> FaceCheck
    PASS / FAIL / INCOMPLETE (< 90 % of the inspected face measured: never passed) with the
    triggers (zone face / rim, r, theta, u, v (face frame, mm), dev_mm (- = below: a dent),
    area_mm2; MF: depth_mm (+ = a dent), width_mm, z), overlay PNG + JSON + maps NPZ in
    output/defects. detector: None = the golden's default, 'mf_fir' | 'mf' | 'band', or a
    callable(dev_map, band, params) replacing the band's trigger function
    (engine.measure.endface.golden.detect_face_triggers). z_thresh / min_depth_mm /
    min_area_mm2 override the golden's trigger settings for this check."""
    from .measure.endface import verbs as face_verbs
    over = {k: float(v) for k, v in (("min_area_mm2", min_area_mm2), ("z_thresh", z_thresh),
                                     ("min_depth_mm", min_depth_mm)) if v is not None}
    return face_verbs.check_face(_engine(), placement, golden, label, over or None, detector)


# ------------------------------------------------------------------ rig check (HOME)
@_verb
def rig_check(preset: Optional[Preset] = None, set_reference: bool = False,
              tol_px: float = 0.5, mock_shift_px=None, label: str = "rigref") -> dict:
    """Has the camera or projector been nudged? A fresh flood + a seeded random block
    pattern (3 frames, the DMD ends black) vs the reference stored for this geometry
    (calibrations/<geometry>/<ts>_rigref) -> image shift in camera px by phase correlation
    (sub-pixel), and ~mm at the working distance when a stereo calibration and a metric
    scan exist. status OK | MOVED (names camera / projector) | UNKNOWN (no texture) |
    REFERENCE_SET (first check at this geometry, or set_reference=True). mock_shift_px
    (dx, dy; mock only) shifts the fresh frames to simulate a nudged camera."""
    from .calibrate import rigref
    return rigref.rig_check(_engine(), _preset(preset), bool(set_reference), float(tol_px),
                            mock_shift_px, label)


# ------------------------------------------------------------------ HDMI fast zone scan
@_verb
def scan_fast(zone=None, preset: Optional[Preset] = None, label: str = "fast", steps: int = 4,
              periods: Optional[list] = None, average: int = 2, min_hold_s: float = 0.12,
              axis: str = "x", faces: Optional[list] = None, margin_dmd_px: float = 6.0):
    """HDMI fast zone scan: the column-axis multifreq set only (steps x periods, default 4 x
    8 / 80 / 1280 DMD px, + the white / black floods) lit through `zone` (grid cells "F10:J12",
    'atlas:<name>/<region>', a DMD mask array or .npy (1140 x 912), or None = full field) and /
    or located end faces (`faces`: face targets, their DMD ellipses + margin_dmd_px), camera
    average and HDMI hold overridden for this run only (the preset file is untouched; average
    < 2 is raised to 2 with a warning: HDMI flicker), ray-plane triangulated -> a metric scan
    (mm) that also keeps its flood pair (flood.npz, for station_measure). Bench: 14 patterns,
    average 2, hold 0.12 s = 5.3 s for the whole station scene. Refused before the rig without a
    stereo calibration (rule 6)."""
    from .reconstruct import fast_scan
    return fast_scan.scan_fast(_engine(), _preset(preset), zone, int(steps),
                               periods or list(fast_scan.FAST_PERIODS), average,
                               float(min_hold_s), axis, label, faces, float(margin_dmd_px))


# ------------------------------------------------------------------ station (cells + cooling tube)
@_verb
def station_measure(scan, flood=None, label: Optional[str] = None, n_cells: int = 2,
                    face_diameter_mm: float = 21.05, params: Optional[dict] = None):
    """Cells with their end faces toward the camera + the cooling tube (CT) between them, from
    one metric scan (mm; its stereo calibration must be current, rule 6) and a flood pair
    (flood: None = the scan's own (scan_fast keeps one; else the decode's fringe modulation,
    with a warning), a (2, H, W) [white, black] array, a .npy / .npz file, or another metric
    scan's flood) -> StationMeasurement: cells C1..Cn (image order) with face centre, height
    vs the mean face plane (mm, + toward the camera), flatness; pairs Cj-Ci: height difference
    along the common face normal, spacing in the face plane; CT: edge height band (auto),
    trapezoid wave (pitch, amplitude, phase vs the cell-pair centre), zones T1..Tn with the
    edge height vs the mean face plane (median / p5 / p95) -> PNG + JSON + CSV in
    output/measurements. params: engine.measure.station.geometry.StationParams fields
    (ct_band_mm, ct_roi, zone_skip, centre_method, ...). Never touches the rig."""
    from .measure.station import verbs as st
    prm = {"n_cells": int(n_cells), "face_diameter_mm": float(face_diameter_mm), **(params or {})}
    return st.station_measure(_engine(), scan, flood, label, prm)[0]


@_verb
def station_golden(measurements: list, label: str = "golden", cell_height_tol_mm: float = 0.15,
                   ct_zone_tol_mm: float = 0.30, spacing_tol_mm: Optional[float] = None,
                   cell_height_band_mm: Optional[list] = None,
                   ct_zone_band_mm: Optional[list] = None, bands: Optional[dict] = None,
                   params: Optional[dict] = None, flood=None):
    """Station golden from >= 1 GOOD measurements (StationMeasurements, their .json, or metric
    scans measured here with `params`) -> station_golden/<geometry>/ (append-only, bound to
    geometry + fingerprint + stereo calibration): nominal cell height differences, spacings and
    CT zone heights (mean), the first measurement's zone model (pitch + phase) and numbering,
    and the user-set limits: a cell pair triggers outside nominal +- cell_height_tol_mm (and /
    or the absolute cell_height_band_mm [lo, hi]), a CT zone outside nominal +-
    ct_zone_tol_mm (and / or ct_zone_band_mm), spacing only with spacing_tol_mm; bands =
    {"C2-C1": [lo, hi], "T3": [lo, hi], "C2-C1 spacing": [lo, hi]} per-item absolute bands."""
    from .measure.station import verbs as st
    lim = {"cell_height_tol_mm": cell_height_tol_mm, "ct_zone_tol_mm": ct_zone_tol_mm,
           "spacing_tol_mm": spacing_tol_mm, "cell_height_band_mm": cell_height_band_mm,
           "ct_zone_band_mm": ct_zone_band_mm, "bands": dict(bands or {})}
    return st.station_golden(_engine(), list(measurements), label, lim, params, flood)


@_verb
def station_check(scan, golden=None, flood=None, label: Optional[str] = None):
    """A metric scan (or a StationMeasurement: its scan + flood) vs a station golden (default:
    the newest at the scan's geometry + fingerprint + stereo calibration), re-measured with the
    golden's parameters and zones -> StationCheck PASS / FAIL (any item outside its band) /
    INCOMPLETE (a golden item not measured: never passed), the items (value, nominal, band,
    status) and triggers, PNG + JSON + CSV in output/defects. Never touches the rig."""
    from .measure.station import verbs as st
    return st.station_check(_engine(), scan, golden, flood, label)


@_verb
def list_library(kind: str = "bases", geometry_id: Optional[str] = None) -> list:
    """Browse the library: 'bases' | 'objects' | 'calibrations' | 'readings' |
    'metric' (metric scans) | 'calib_poses' (stereo calibration board poses) |
    'dentsets' (dent-detector captures) | 'faces' (located end faces) |
    'facescans' (targeted end-face scans) | 'face_golden' (golden end faces) |
    'station_golden' (station goldens: cells + cooling tube) |
    'outputs' (the output bin index; the 2nd argument filters by output kind) |
    'incomplete' (half-written base/object entries the listings skip; move
    them to trash/ with delete(path))."""
    s = _engine()
    if kind == "outputs":
        return s.store.outputs(geometry_id)
    if geometry_id and not config.valid_id(geometry_id):
        # it becomes a folder name under the library root: no separators, no '..'
        raise ValueError(f"geometry_id must be one id [A-Za-z0-9_.-], got {geometry_id!r}")
    if kind == "incomplete":
        return s.store.incomplete_entries()
    if kind == "materials":
        from .sense import spectral as m
        return m.list_materials(s)
    if kind == "spectral":
        from .sense import spectral as m
        return m.list_readings(s)
    if kind == "banks":
        from .measure import bank as m
        return m.list_banks(s, geometry_id)
    if kind == "sensors":
        from .sense import virtual as m
        return m.list_sensors(s, geometry_id)
    if kind == "maps":
        from .decode import projmap as m
        return m.list_maps(s, geometry_id)
    if kind == "readings":
        out = []
        for d in s.store.entries("readings"):
            out.append({"id": d.name, "path": d, **(_read_json(d / "codes.json") or {}),
                        **(_read_json(d / "meta.json") or {})})
        return out
    if kind == "bases":
        return base_lib.list_bases(s.store, geometry_id)
    if kind == "objects":
        return object_lib.list_objects(s.store, geometry_id)
    if kind == "calibrations":
        geos = [geometry_id] if geometry_id else s.store.geometry_ids("calibrations")
        out = []
        for g in geos:
            for d in s.store.entries("calibrations", g):
                c = _read_json(d / "depth_calibration.json")
                if c is not None:
                    out.append(c)
                c = _read_json(d / "stereo_calibration.json")
                if c is not None:
                    out.append({**c, "path": d})
                c = _read_json(d / "dent_null.json")
                if c is not None and c.get("kind") == "dent_null":
                    out.append({**c, "path": d})
        return out
    if kind == "dentsets":
        from .measure.dents import verbs as dent_verbs
        return dent_verbs.list_dentsets(s.store, geometry_id)
    if kind in ("faces", "facescans", "face_golden"):
        from .measure.endface import verbs as face_verbs
        return {"faces": face_verbs.list_faces, "facescans": face_verbs.list_facescans,
                "face_golden": face_verbs.list_goldens}[kind](s.store, geometry_id)
    if kind == "metric":
        from .reconstruct import metric_scan
        return metric_scan.list_metric(s.store, geometry_id)
    if kind == "station_golden":
        from .measure.station import verbs as st
        return st.list_goldens(s.store, geometry_id)
    if kind == "calib_poses":
        geos = [geometry_id] if geometry_id else s.store.geometry_ids("calib_poses")
        out = []
        for g in geos:
            for d in s.store.entries("calib_poses", g):
                m = _read_json(d / "meta.json")
                if m is not None:
                    out.append({**m, "path": d})
        return out
    raise ValueError("kind must be one of: bases, objects, calibrations, readings, outputs, "
                     "materials, spectral, banks, maps, sensors, incomplete, metric, "
                     "calib_poses, dentsets, faces, facescans, face_golden, station_golden")

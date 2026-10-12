"""
Hardware-triggered flash scan and trigger test behind the api verbs
(docs/TRIGGER_FULLSPEED_TEST.md). EXPERIMENTAL: not yet run on the rig.

The DLPC350 plays a flash pack from its own flash (a HUMAN sets Pattern
Sequence mode, the sequence, TRIG_OUT_1 and presses Play in the 4500 MKII GUI);
TRIG_OUT_1 is wired to the camera's trigger line, so the camera takes exactly
one full frame per pattern.

    trigger_test()   camera only: count the pulses for N seconds -> rate, period
                     stats, missed pulses, verdict NO PULSES | RATE MISMATCH | OK
    scan_trigger()   N triggered frames per loop (TriggeredSequenceCapture: missing /
                     extra frames, spacing +-50 %, first-trigger timeout) -> aligned
                     (1-bit: sync markers; 8-bit: white / black, refused unless exactly
                     one start fits) -> the existing decoders -> 'plane' triangulation
                     -> a metric scan

Rule 1: nothing here imports or sends a DLPC350 USB / HID command, and the HDMI
projector window is never opened. Rule 6: the stereo calibration is checked
before the camera. The camera's TriggerMode is set back to Off on the way out,
always.
"""

from __future__ import annotations

import json
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import numpy as np

from .. import config, rules
from ..capture import hwtrigger, patternmode, softsync
from ..library.store import json_safe, safe_label
from . import metric_scan
from .triangulate import triangulate

MODE = "flash_trigger"
DEFAULT_TEST_EXPOSURE_US = 1000.0
MAX_REPEATS = 20


def _mfp():
    import sys
    root = str(Path(__file__).resolve().parents[2])
    if root not in sys.path:
        sys.path.insert(0, root)
    from tools import make_flash_pack
    return make_flash_pack


def _drop_frames(e: Optional[BaseException]) -> None:
    """Clear the locals of the finished frames an exception (and its causes) holds:
    a frame of the PySpin calls keeps the camera alive, and PySpin refuses to release
    its system while a reference to the camera exists."""
    seen = set()
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        traceback.clear_frames(e.__traceback__)
        e = e.__cause__ or e.__context__


def _deinit_camera(real) -> None:
    """EndAcquisition and DeInit, each step on its own (as camera._release_driver
    does): the triggered capture has ended acquisition already, and the seed driver's
    close() runs both in ONE try, so its EndAcquisition would raise and skip DeInit.
    The driver's close() then drops its camera and releases the system."""
    drv = getattr(real, "_drv", None)
    for step in ("EndAcquisition", "DeInit"):
        try:
            if getattr(drv, "_cam", None) is not None:
                getattr(drv._cam, step)()
        except Exception:                   # noqa: BLE001 - not acquiring / de-inited already
            pass


@contextmanager
def triggered_camera(s, preset, *, line: str, exposure_us: float, mock_factory=None):
    """The hardware-triggered camera, and nothing else of the rig. mock: mock_factory()
    (or bench.trigger_sim, a dict of FakeTriggeredCamera fields: {"pulses": False},
    {"period_us": 5000}, ...). real: the rig lock, the Blackfly S opened by the engine's
    RealCamera, PySpinTriggeredCamera on `line`. On the way out, always and in this
    order: EndAcquisition + TriggerMode Off (disarm), every reference to the camera
    dropped, DeInit, the camera's system released (RealCamera.close), the rig lock
    released. A TriggerMode Off that fails is raised (noted on the error that is
    already propagating). The projector is never opened."""
    rules.check_scan_path_usb_free()
    hwtrigger.check_line(line)
    if s.mock:
        cam = mock_factory() if mock_factory is not None else hwtrigger.FakeTriggeredCamera(
            **(getattr(s.bench, "trigger_sim", None) or {}))
        try:
            cam.configure_trigger(line, exposure_us)
            yield cam
        finally:
            cam.disarm()                     # TriggerMode Off, always
        return
    from ..capture.camera import open_camera
    from ..capture.riglock import RigLock
    from ..session import _release_rig, default_holder
    lock = RigLock(holder=getattr(s, "holder", "") or default_holder()).acquire()
    real = tc = None
    primary = None
    try:
        real = open_camera(preset.camera, mock=False)
        tc = patternmode.PySpinTriggeredCamera(real._drv._cam, exposure_us=float(exposure_us),
                                               source=line)
        yield tc
    except BaseException as e:
        primary = e
        raise
    finally:
        off_error = None
        if tc is not None:
            try:
                tc.disarm()                  # EndAcquisition (guarded) + TriggerMode Off
            except Exception as e:           # noqa: BLE001 - reported after the release
                off_error = e
            tc.release()                     # drop the camera reference
        tc = None
        _drop_frames(primary)
        _drop_frames(off_error)
        try:
            _deinit_camera(real)
            _release_rig(real, None, lock, primary if primary is not None else off_error)
        finally:
            real = None
        if off_error is not None:
            note = (f"camera TriggerMode Off failed: {type(off_error).__name__}: {off_error} - "
                    "the next normal scan resets it, or set it in SpinView")
            if primary is not None:
                if hasattr(primary, "add_note"):
                    primary.add_note(note)
            else:
                raise RuntimeError(note) from off_error


# ---------------------------------------------------------------- trigger test

def trigger_test(s, preset, line: str = "Line0", seconds: float = 5.0,
                 expect_period_ms: Optional[float] = None,
                 exposure_us: float = DEFAULT_TEST_EXPOSURE_US) -> dict:
    hwtrigger.check_line(line)
    seconds = float(seconds)
    if not 0.2 <= seconds <= 120:
        raise ValueError(f"seconds must be 0.2..120, got {seconds:g}")
    if expect_period_ms is not None:
        P = float(expect_period_ms)
        if P <= 0:
            raise ValueError("expect_period_ms must be > 0")
        if exposure_us > P * 1000 - _mfp().CAMERA_TRIGGER_GUARD_US:
            raise ValueError(f"camera exposure {exposure_us:g} us does not fit the expected "
                             f"{P:g} ms period")
    timeout_ms = int(max(1000, 3 * (expect_period_ms or 0)))
    factory = None
    if s.mock and getattr(s.bench, "trigger_sim", None) is None and expect_period_ms:
        def factory():                       # the simulated human set the period entered
            return hwtrigger.FakeTriggeredCamera(float(expect_period_ms) * 1000)
    t0 = time.perf_counter()
    with triggered_camera(s, preset, line=line, exposure_us=exposure_us,
                          mock_factory=factory) as cam:
        cam.arm(64)
        try:
            ts, ids, timeouts, incomplete = hwtrigger.count_pulses(cam, seconds,
                                                                   timeout_ms=timeout_ms)
        finally:
            cam.disarm()                     # TriggerMode Off, always
    r = hwtrigger.trigger_stats(ts, ids, seconds=seconds, expect_period_ms=expect_period_ms,
                                timeouts=timeouts, incomplete=incomplete)
    r.update({"line": line, "exposure_us": float(exposure_us), "mock": s.mock,
              "timeout_ms": timeout_ms, "trigger_mode_restored": "Off",
              "wall_s": round(time.perf_counter() - t0, 3)})
    return r


# ---------------------------------------------------------------- triggered scan

def _load_pack(pack) -> tuple:
    p = Path(pack)
    f = p / "manifest.json" if p.is_dir() else p
    if not f.is_file():
        raise ValueError(f"flash pack {pack}: no manifest.json")
    return json.loads(f.read_text(encoding="utf-8")), f.parent


def _pack_timing(m: dict, exposure_us: Optional[float]) -> dict:
    """period, pattern exposure, bit depth and the camera exposure to use, checked
    against the DLPC350 and camera rules (ValueError)."""
    mfp = _mfp()
    t = m.get("timing") or {}
    hw = t.get("hardware_trigger") or {}
    if m.get("mode") == softsync.FASTSCAN_MODE:
        bd, P, X = 1, t.get("pattern_period_us"), t.get("pattern_exposure_us")
        E = exposure_us or (hw.get("camera") or {}).get("exposure_us") or \
            (t.get("camera") or {}).get("exposure_us") or mfp.TRIGGERED_CAMERA_EXPOSURE_US
    elif m.get("mode") == "flash":
        u = t.get("uniform") or {}
        bd = max(int(e["bitdepth"]) for e in m["sequence"])
        P, X = u.get("period_us"), u.get("exposure_us")
        E = exposure_us or u.get("camera_exposure_us")
    else:
        raise ValueError(f"pack mode {m.get('mode')!r}: the triggered scan takes a flash pack "
                         "(make_flash_pack, 8-bit hybrid) or a --fastscan pack, not a video-port "
                         "pack")
    if not (P and X and E):
        raise ValueError("the pack's manifest has no pattern period / exposure")
    errs = mfp.timing_errors(bd, int(X), int(P)) + mfp.trigger_timing_errors(bd, P, X, E)
    if errs:
        raise ValueError("triggered capture not feasible for this pack: " + "; ".join(errs))
    return {"bit_depth": bd, "period_us": float(P), "pattern_exposure_us": float(X),
            "camera_exposure_us": float(E), "hardware_trigger": hw or None}


def _capture_manifest(names: list, n_frames: int, period_us: float, axis: str = "x") -> dict:
    seq = [{"seq": k, "name": f"{names[k % len(names)]}#{k // len(names)}", "axis": axis,
            "basis_index": k} for k in range(n_frames)]
    return {"sequence": seq, "timing": {"mode": "flash", "uniform": {"period_us": period_us}}}


def scan_trigger(s, preset, pack, variant: Optional[str] = None, label: str = "trigger",
                 line: str = "Line0", repeats: int = 1, exposure_us: Optional[float] = None,
                 acknowledge: Optional[str] = None, method: str = "plane",
                 first_timeout_s: float = 30.0) -> dict:
    safe_label(label)
    if acknowledge != patternmode.ACK_PHRASE:            # before anything else
        raise patternmode.PatternModeRefused(
            "the hardware-triggered scan needs the projector in Pattern Sequence mode, set by "
            "a HUMAN in the 4500 MKII GUI (flash source, TRIG_OUT_1 wired to the camera, Play): "
            "pass the acknowledgement (CLI --pattern-mode-set-by-human). The engine never "
            "switches the projector mode itself (docs/TRIGGER_FULLSPEED_TEST.md)")
    rules.check_scan_path_usb_free()
    hwtrigger.check_line(line)
    if method not in ("plane", "auto"):
        raise ValueError("method must be 'plane' (one coded axis: the column plane)")
    repeats = int(repeats)
    if not 1 <= repeats <= MAX_REPEATS:
        raise ValueError(f"repeats must be 1..{MAX_REPEATS}, got {repeats}")
    m, pack_dir = _load_pack(pack)
    tm = _pack_timing(m, exposure_us)
    onebit = m.get("mode") == softsync.FASTSCAN_MODE
    warnings = []
    if onebit:
        layout, _, _ = softsync.load_pack_layout(pack_dir, variant)    # the pack contract
        names = layout.names()
    else:
        axes = m.get("axes") or ["x"]
        if axes != ["x"]:
            raise ValueError(f"the pack codes axes {axes}: the plane triangulation needs the "
                             "column axis alone (make_flash_pack --axes x)")
        seq = sorted(m["sequence"], key=lambda e: e["seq"])
        names = [e["name"] for e in seq]
        if variant is not None:
            raise ValueError("--variant is for fast-scan packs")
        if sorted(int(e["basis_index"]) for e in seq) != list(range(len(seq))):
            raise ValueError("the pack's basis indices are not 0..N-1 for one axis")
        # HybridBasis order: basis index 0 = white, 1 = black
        roles = [e.get("role") or {0: "white", 1: "black"}.get(int(e["basis_index"]))
                 for e in seq]
        if roles.count("white") != 1 or roles.count("black") != 1:
            raise ValueError("the pack needs one white and one black pattern: the triggered "
                             "frames are aligned on them")
    pack_geo = m.get("geometry_id")
    if pack_geo and pack_geo != preset.geometry_id:
        warnings.append(f"the pack was made for geometry {pack_geo!r}, the preset is "
                        f"{preset.geometry_id!r}")
    fp = config.geometry_fingerprint(preset)
    cal = metric_scan._require_cal(s, preset.geometry_id, fp, "scan_trigger")      # rule 6
    rig = metric_scan._require_rig(s, preset, "scan_trigger")           # a mobile rig: verified
    n = len(names)
    n_frames = (repeats + 1) * n if onebit else repeats * n
    P = tm["period_us"]
    cman = _capture_manifest(names, n_frames, P)

    factory = None
    if s.mock and getattr(s.bench, "trigger_sim", None) is None:
        from ..capture import mock_flashloop
        if onebit:
            pats = mock_flashloop.pack_patterns(pack_dir, variant or "full")
        else:
            mfp = _mfp()
            ms_, imgs = mfp.load_pack(pack_dir)
            pats = mfp.unpack(imgs, sorted(ms_["sequence"], key=lambda e: e["seq"]))
        seed = int(s.bench._rng.integers(0, 2**31 - 1))

        def factory():                       # the simulated human pressed Play
            # a looping sequence (Repeat on) is caught anywhere in the loop; one pass
            # (Repeat off) starts at its first pattern
            return hwtrigger.mock_scene_camera(s.bench, pats, P,
                                               exposure_us=tm["camera_exposure_us"],
                                               repeat=onebit or repeats > 1, start=None,
                                               seed=seed)
    t_all = time.perf_counter()
    with triggered_camera(s, preset, line=line, exposure_us=tm["camera_exposure_us"],
                          mock_factory=factory) as cam:
        cap = patternmode.TriggeredSequenceCapture(
            cam, cman, acknowledge=acknowledge, first_timeout_s=first_timeout_s,
            allow_extra=onebit or repeats > 1).capture()
    capture_s = round(time.perf_counter() - t_all, 3)
    ts = np.asarray(cap.timestamps_ns, np.float64)
    dts = np.diff(ts) / 1e6
    timing = {"frames": len(cap.frames), "frames_per_loop": n, "loops_captured": n_frames // n,
              "total_ms": round(float(ts[-1] - ts[0]) / 1e6, 3) if len(ts) > 1 else 0.0,
              "per_frame_ms": {"mean": round(float(dts.mean()), 4),
                               "min": round(float(dts.min()), 4),
                               "max": round(float(dts.max()), 4)} if len(dts) else None,
              "expected_period_ms": P / 1000.0, "set_ms": round(n * P / 1000.0, 3),
              "drops": 0, "frame_ids": cap.report.get("frame_ids"),
              "camera_exposure_us": tm["camera_exposure_us"], "line": line,
              "capture_wall_s": capture_s}
    t0 = time.perf_counter()
    H, W = np.shape(cap.frames[0])
    if onebit:
        images, arep = hwtrigger.align_onebit(cap.frames, layout, repeats)
        ia = softsync._full_white_index(layout)          # SYNC A: full field in every image
        lit = (images[ia] - images[layout.index("black")[0]]) > softsync.LIT_MIN_DN
        seen, ratio = softsync._detect_variant(images, layout, lit)
        if variant is not None and seen is not None and seen != variant:
            warnings.append(f"--variant {variant} but the frames look like the {seen} image")
        rep = {"variant_detected": seen, "variant_ratio": ratio, "loop_period_ms": n * P / 1e3,
               "pattern_period_ms": P / 1e3, "counts": {nm: repeats for nm in names},
               "align": arep}
        res = softsync.SortResult(images, [repeats] * n, [repeats] * n,
                                  np.full(len(cap.frames), -1), np.array(["clean"] * len(
                                      cap.frames)), n * P / 1e6, P / 1e6, 1.0, rep,
                                  (0, 0, int(W), int(H)), (int(H), int(W)))
        fld = softsync.decode_sorted(res, layout, preset.geometry_id,
                                     contrast_min=preset.decode.contrast_min,
                                     conf_min=preset.decode.conf_min)
        dec = {"basis": "multifreq (1-bit fast-scan loop)", "align": arep,
               "variant": variant, "variant_detected": seen}
        steps, periods, parity = layout.steps, list(layout.periods), layout.diamond_parity
    else:
        from ..bases.hybrid import from_preset_bases
        pr = preset
        if m.get("preset") and m["preset"] != preset.name:
            try:
                pr = config.load_preset(m["preset"])
            except Exception:                # noqa: BLE001 - fall back to the active preset
                warnings.append(f"pack preset {m['preset']!r} not found; decoding with "
                                f"{preset.name!r}'s bases")
        basis = from_preset_bases(pr.bases, axis="x")
        # the camera may arm anywhere in a looping sequence: the start is proven from
        # the content (white brightest, black darkest; loops identical) or refused
        images, arep = hwtrigger.align_sequence(cap.frames, roles, repeats)
        frames = [None] * n
        for e, im in zip(seq, images):
            frames[int(e["basis_index"])] = im
        fld = basis.decode(frames, preset.geometry_id, contrast_min=preset.decode.contrast_min,
                           conf_min=preset.decode.conf_min)
        dec = {"basis": "hybrid (8-bit pack)", "order": "aligned on white / black",
               "align": arep}
        steps, periods, parity = None, None, None
    decode_s = round(time.perf_counter() - t0, 3)
    t0 = time.perf_counter()
    cloud = triangulate(fld, cal, "plane")
    fld.meta.update({"preset": preset.name, "label": label, "mock": s.mock})
    trig = {"pack": str(pack_dir), "pack_name": m.get("name"), "pack_mode": m.get("mode"),
            "bit_depth": tm["bit_depth"], "pattern_period_us": P,
            "pattern_exposure_us": tm["pattern_exposure_us"],
            "hardware_trigger": tm["hardware_trigger"], "timing": timing, "decode": dec,
            "repeats": repeats}
    ms = metric_scan._save(s, fld, cloud, label, cal, fp, preset.name, {
        "mode": MODE, "capture_s": capture_s, "steps": steps, "periods_px": periods,
        "diamond_parity": parity, "fast": True, "experimental": True,
        "trigger": json_safe(trig), "rig": rig})
    times = {"capture_s": capture_s, "decode_s": decode_s,
             "triangulate_save_s": round(time.perf_counter() - t0, 3),
             "total_s": round(time.perf_counter() - t_all, 3)}
    return {"scan": ms, "id": ms.id, "mode": MODE, "n_points": int(cloud.mask.sum()),
            "valid_pct": ms.meta.get("valid_pct"), "calibration_id": cal.id,
            "timing": timing, "decode": dec, "times_s": times, "warnings": warnings}

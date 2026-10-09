"""
Hardware trigger at full speed: TRIG_OUT_1 of the DLPC350 -> the camera's
trigger line, one camera frame per pattern (docs/TRIGGER_FULLSPEED_TEST.md).

    trigger_stats()      camera timestamps of a pulse count -> rate, period stats, verdict
    count_pulses()       grab triggered frames for N seconds (timeouts and incomplete
                         frames are counted, not errors)
    align_onebit()       triggered frames of the repeating 1-bit fast-scan loop -> one
                         image per slot, the loop start found by the sync markers
    align_sequence()     triggered frames of a pattern sequence (8-bit hybrid pack) ->
                         one image per pattern, the start found from white / black
    FakeTriggeredCamera  mock: pulses at a period, frames rendered from DMD images

Camera only. Nothing here imports or sends a DLPC350 command; the projector's
mode, sequence and TRIG_OUT_1 are set by a HUMAN in the 4500 MKII GUI.
"""

from __future__ import annotations

import math
import time
from typing import Callable, Optional

import numpy as np

from . import softsync
from .patternmode import TRIGGER_INPUT_LINES, IncompleteFrame

# camera inputs only (patternmode.TRIGGER_INPUT_LINES): Line0 opto-isolated, Line3
# non-isolated. Line1 is the opto OUTPUT, Line2 the 3.3 V auxiliary output pin.
TRIGGER_LINES = TRIGGER_INPUT_LINES
RATE_TOL = 0.05                               # mean period vs expected, fraction
COUNT_SLACK = 3                               # pulses short of the window's expected count
NO_PULSE_HINT = ("no trigger reached the camera: is the sequence playing (Play, Repeat on)? "
                 "TRIG_OUT_1 on J14 pin 2 and GND on pin 3 of the EVM? Line0 = GPIO pin 2 with "
                 "the opto ground pin 5; the opto input needs > 2.6 V high - set the EVM's "
                 "TRIG_OUT_1 level jumper (J13) to 3.3 V, or try Line3 (GPIO pin 1, GND pin 6) "
                 "or a level shifter (docs/TRIGGER_FULLSPEED_TEST.md)")


def check_line(line: str) -> str:
    if line not in TRIGGER_LINES:
        raise ValueError(f"trigger line must be one of {TRIGGER_LINES} (Line1 is the opto "
                         "output; Line2 is GPIO pin 3, the 3.3 V auxiliary output - never wire "
                         f"the trigger to it), got {line!r}")
    return line


# ---------------------------------------------------------------- statistics

def trigger_stats(timestamps_ns: list, frame_ids: list, *, seconds: float,
                  expect_period_ms: Optional[float] = None, timeouts: int = 0,
                  incomplete: int = 0) -> dict:
    """Pulse count, rate, period mean / std / min / max (camera timestamps), missed
    pulses vs the expected period (a gap of k periods = k - 1 missed), camera frame-id
    gaps, incomplete frames, and the verdict: NO PULSES | RATE MISMATCH | OK.
    The pulses must cover the whole window (the sequence plays before the test starts):
    a timeout while pulses were seen (the train stopped, started late or is
    intermittent), or fewer pulses than the window holds at the expected period, is not
    OK; nor is an incomplete frame (image data lost after the trigger)."""
    n = len(timestamps_ns)
    ts = np.asarray(timestamps_ns, np.float64)
    dt_ms = np.diff(ts) / 1e6 if n > 1 else np.zeros(0)
    ids = np.asarray(frame_ids, np.int64)
    id_gaps = int(np.sum(np.diff(ids) - 1)) if n > 1 else 0
    out = {"count": n, "seconds": float(seconds), "timeouts": int(timeouts),
           "incomplete_frames": int(incomplete), "expect_period_ms": expect_period_ms,
           "expected_count": None, "frame_id_gaps": id_gaps, "rate_hz": None,
           "period_ms": None, "missed_pulses": None, "notes": []}
    if expect_period_ms:
        out["expected_count"] = int(math.floor(float(seconds) * 1000.0 / float(expect_period_ms)))
    if n > 1:
        span_s = (ts[-1] - ts[0]) / 1e9
        out["rate_hz"] = round((n - 1) / span_s, 3) if span_s > 0 else None
        out["period_ms"] = {"mean": round(float(dt_ms.mean()), 4),
                            "std": round(float(dt_ms.std()), 4),
                            "min": round(float(dt_ms.min()), 4),
                            "max": round(float(dt_ms.max()), 4)}
    lost = (f"{int(incomplete)} incomplete frame(s): the trigger arrived but the image data "
            "was lost on USB3 - use a USB3 port on its own controller, a shorter cable, close "
            "other programs streaming from the camera, or a longer period")
    if n == 0:
        if incomplete:
            out["verdict"] = "RATE MISMATCH"
            out["notes"].append(lost)
            return out
        out["verdict"] = "NO PULSES"
        out["notes"].append(NO_PULSE_HINT)
        return out
    if n == 1:
        out["verdict"] = "RATE MISMATCH"
        out["notes"].append("one pulse only: the sequence plays once (Repeat off) or stopped")
        if incomplete:
            out["notes"].append(lost)
        return out
    bad = []
    if timeouts:
        bad.append(f"{int(timeouts)} timeout(s) while pulses were arriving: the pulse train "
                   "stopped, started late or is intermittent - the sequence must be playing "
                   "(Repeat on) before the test starts and all through it; check the "
                   "connector and the ground")
    if out["expected_count"] is not None:
        want = out["expected_count"]
        if n < want * (1 - RATE_TOL) - COUNT_SLACK:
            bad.append(f"only {n} pulse(s) in {float(seconds):g} s, ~{want} expected at "
                       f"{float(expect_period_ms):g} ms: the train stopped or started late, "
                       "or the PC reads frames slower than they arrive")
    if incomplete:
        bad.append(lost)
    if expect_period_ms:
        P = float(expect_period_ms)
        k = np.maximum(np.rint(dt_ms / P), 1)
        missed = int(np.sum(k - 1))
        out["missed_pulses"] = missed
        single = dt_ms[k == 1]
        mean1 = float(single.mean()) if len(single) else float(dt_ms.mean())
        if abs(mean1 - P) > RATE_TOL * P:
            bad.append(f"pulse period {mean1:.3f} ms vs expected {P:g} ms: the GUI's Pattern "
                       "Period differs from the pack, or the pulses are not TRIG_OUT_1")
        if missed:
            bad.append(f"~{missed} pulse(s) missed (gaps of 2+ periods): the camera dropped "
                       "triggers - level near the threshold, or period shorter than the camera "
                       "allows")
        if np.any(dt_ms < 0.5 * P):
            bad.append(f"{int(np.sum(dt_ms < 0.5 * P))} pulse(s) closer than half a period: "
                       "double edges (ringing / a noisy ground) - shorten the wires, common GND")
    if id_gaps:
        bad.append(f"camera frame ids skip {id_gaps}: frames lost after the trigger (USB "
                   "bandwidth / buffers)")
    out["notes"] += bad
    out["verdict"] = "RATE MISMATCH" if bad else "OK"
    return out


def count_pulses(cam, seconds: float, *, timeout_ms: int = 1000,
                 clock: Optional[Callable] = None) -> tuple:
    """Grab triggered frames until `seconds` have passed -> (timestamps_ns, frame_ids,
    timeouts, incomplete). Neither a timeout nor an incomplete frame (patternmode.
    IncompleteFrame: the pulse arrived, its image was lost on USB) stops the count:
    both are counted, and an incomplete frame's pulse is kept when the camera gave its
    frame id and timestamp. Any other camera error raises. `clock` (or cam.clock):
    seconds, the wall clock by default (a fake camera brings its simulated one)."""
    clock = clock or getattr(cam, "clock", None) or time.perf_counter
    ts, ids, timeouts, incomplete = [], [], 0, 0
    t0 = clock()
    while clock() - t0 < float(seconds):
        try:
            _img, fid, t = cam.grab_triggered(int(timeout_ms))
        except TimeoutError:
            timeouts += 1
            continue
        except IncompleteFrame as e:
            incomplete += 1
            if e.frame_id is not None and e.timestamp_ns is not None:
                ts.append(int(e.timestamp_ns))
                ids.append(int(e.frame_id))
            continue
        ts.append(int(t))
        ids.append(int(fid))
    return ts, ids, timeouts, incomplete


# ---------------------------------------------------------------- 1-bit loop alignment

def align_onebit(frames: list, layout: softsync.FlashLayout, repeats: int) -> tuple:
    """Triggered frames of the repeating 16-slot loop (any start slot; at least
    (repeats + 1) x 16 frames) -> (per-slot mean images in layout order, report).
    The loop start: sync_code (slot 1, the only pattern structured along camera y)
    has the highest orientation summed over the loops; sync_white sits just before
    it. Refused (SequenceError) when no slot stands out, or black is not the darkest."""
    n = layout.n
    need = (int(repeats) + 1) * n
    if len(frames) < need:
        raise softsync.SequenceError(f"{len(frames)} frames < {need} for {repeats} loop(s) + "
                                     "one to find the start")
    from ..bases import onebit as ob
    sig = [ob.sync_signature(f) for f in frames[:need]]
    orient = np.array([s["orientation"] for s in sig])
    means = np.array([s["mean"] for s in sig])
    score = np.array([orient[j::n].mean() for j in range(n)])
    order = np.argsort(score)[::-1]
    j_code = int(order[0])
    ratio = float(score[order[0]] / max(score[order[1]], 1e-6))
    i_code = [i for i, e in enumerate(layout.entries) if e.get("orient") == "y"]
    if not i_code:
        raise ValueError("the layout has no y-oriented sync marker")
    rep = {"sync_code_phase": j_code, "orientation_ratio": round(ratio, 3)}
    if ratio < softsync.ORIENT_MIN_RATIO:
        e = softsync.SequenceError(
            f"no sync_code frame stands out (orientation ratio {ratio:.2f} < "
            f"{softsync.ORIENT_MIN_RATIO:g}): the frames are not the fast-scan loop, the "
            "projector shows something else, or the trigger is not one frame per pattern")
        e.report = rep
        raise e
    start = (j_code - i_code[0]) % n
    use = frames[start:start + int(repeats) * n]
    images = [np.mean(np.stack(use[k::n]), axis=0).astype(np.float32) for k in range(n)]
    ib = layout.index("black")[0]
    iw = layout.index("white")[0]
    # compare where the decode white lights (the zone image lights only the zone)
    lit = (images[iw] - np.min(np.stack(images), axis=0)) > softsync.LIT_MIN_DN
    if not lit.any():
        lit = np.ones(images[iw].shape, bool)
    slot_means = [float(np.mean(im[lit])) for im in images]
    rep.update({"start_frame": int(start), "loops": int(repeats),
                "slot_means": [round(v, 2) for v in slot_means]})
    if int(np.argmin(slot_means)) != ib or slot_means[iw] <= slot_means[ib] + 2.0:
        e = softsync.SequenceError(
            f"after aligning on sync_code the black slot is not the darkest / white not "
            f"brighter (slot means {rep['slot_means']}): wrong plane order in the GUI, or "
            "the trigger is not one frame per pattern")
        e.report = rep
        raise e
    return images, rep


# ---------------------------------------------------------------- sequence alignment
# A triggered capture of a pattern SEQUENCE (the 8-bit hybrid pack) starts wherever
# the sequence is when the camera arms: a sequence already looping (Repeat on) shows
# pattern (start + k) % n in frame k. The start is found from the content: white
# lights everything the projector reaches, black nothing, so at every lit pixel the
# white frame is the brightest and the black frame the darkest (light only adds).
ALIGN_TOL_FRACTION = 0.25     # per pixel: within this share of its range of the max / min
ALIGN_TOL_DN = 3.0            # ... plus this (noise)
ALIGN_MIN_AGREE = 0.9         # share of lit px where white is the max and black the min
ALIGN_MIN_LIT_PX = 500        # sampled px that the patterns change by > LIT_MIN_DN
LOOP_MAX_RMS = 0.1            # repeats > 1: loop-to-loop rms / median white-black range
ALIGN_SAMPLE_PX = 270         # the checks run on a grid of about this many rows


def align_sequence(frames: list, roles: list, repeats: int) -> tuple:
    """Triggered frames of `repeats` loops of a pattern sequence (roles in sequence
    order, one 'white' and one 'black'; any start) -> (per-pattern mean images in
    sequence order, report). The start must be proven: exactly one start puts the
    white pattern at the per-pixel maximum and black at the minimum on >= ALIGN_MIN_AGREE
    of the lit pixels, and with repeats > 1 every loop must show the same images (the
    sequence is len(roles) patterns long). Refused (SequenceError) otherwise: a
    mis-assigned set is never returned."""
    n, R = len(roles), int(repeats)
    if "white" not in roles or "black" not in roles:
        raise ValueError("the sequence has no white / black pattern to align on")
    if R < 1 or len(frames) != R * n:
        raise softsync.SequenceError(f"{len(frames)} frames for {R} loop(s) of {n} patterns")
    iw, ib = roles.index("white"), roles.index("black")
    H, W = np.shape(frames[0])
    st = max(1, min(H, W) // ALIGN_SAMPLE_PX)
    acc = [np.zeros((H, W), np.float64) for _ in range(n)]
    for k, f in enumerate(frames):
        acc[k % n] += f
    means = [(a / R).astype(np.float32) for a in acc]      # by phase (frame index % n)
    sub = np.stack([m[::st, ::st] for m in means]).astype(np.float64)
    hi, lo = sub.max(axis=0), sub.min(axis=0)
    rng = hi - lo
    lit = rng > softsync.LIT_MIN_DN
    rep = {"lit_px": int(lit.sum()), "sample_step": int(st)}

    def refuse(msg):
        e = softsync.SequenceError(msg)
        e.report = rep
        raise e

    if rep["lit_px"] < ALIGN_MIN_LIT_PX:
        refuse(f"only {rep['lit_px']} sampled px change between the patterns (< "
               f"{ALIGN_MIN_LIT_PX}): no projector light in the frames, or they are not the "
               "pattern sequence")
    tol = ALIGN_TOL_FRACTION * rng[lit] + ALIGN_TOL_DN
    top = np.array([float(np.mean(s[lit] >= hi[lit] - tol)) for s in sub])
    bot = np.array([float(np.mean(s[lit] <= lo[lit] + tol)) for s in sub])
    # candidate j = the phase showing white; black then sits (ib - iw) phases later
    agree = np.array([min(top[j], bot[(j + ib - iw) % n]) for j in range(n)])
    fits = [j for j in range(n) if agree[j] >= ALIGN_MIN_AGREE]
    order = np.argsort(agree)[::-1]
    rep.update({"agree": round(float(agree[order[0]]), 3),
                "runner_up": round(float(agree[order[1]]), 3) if n > 1 else None,
                "min_agree": ALIGN_MIN_AGREE})
    if len(fits) != 1:
        if not fits:
            refuse(f"no start of the {n}-pattern sequence puts white brightest and black "
                   f"darkest (best agreement {rep['agree']:.2f} < {ALIGN_MIN_AGREE:g}): the GUI "
                   "plays another sequence or order, the patterns are inverted (Invert "
                   "Pattern), or the trigger is not one frame per pattern")
        refuse(f"{len(fits)} starts of the sequence fit white / black equally: the camera sees "
               "too little of the pattern field to find where the sequence starts")
    jw = fits[0]
    # every other pattern must look like neither white nor black: a loop that shows a white /
    # black frame where one of the pack's patterns belongs (the GUI plays another sequence)
    # would decode that frame in the pattern's place
    jb = (jw + ib - iw) % n
    odd = [j for j in range(n) if j not in (jw, jb)
           and max(top[j], bot[j]) >= ALIGN_MIN_AGREE]
    if odd:
        refuse(f"{len(odd)} more frame(s) of the {n}-pattern loop look like the white or the "
               "black pattern: the GUI plays another sequence than the pack's (a duplicated "
               "white / black entry?)")
    start = (iw - jw) % n                                   # frame k shows (start + k) % n
    rep.update({"start_seq": int(start), "white_frame": int(jw), "loops": R})
    if R > 1:
        scale = float(np.median(rng[lit]))
        ss = 0.0
        for k, f in enumerate(frames):
            d = np.asarray(f)[::st, ::st].astype(np.float64)[lit] - sub[k % n][lit]
            ss += float(np.sum(d * d))
        rms = math.sqrt(ss / (len(frames) * rep["lit_px"])) / max(scale, 1e-6)
        rep["loop_rms"] = round(rms, 4)
        if rms > LOOP_MAX_RMS:
            refuse(f"the {R} loops do not show the same images (rms {rms:.3f} of the white-"
                   f"black range > {LOOP_MAX_RMS:g}): the GUI sequence is not the pack's {n} "
                   "patterns, or the scene moved")
    return [means[(q - start) % n] for q in range(n)], rep


# ---------------------------------------------------------------- mock camera

class FakeTriggeredCamera:
    """Mock hardware-triggered camera: a pulse every period_us (TRIG_OUT_1 of a
    playing sequence), one frame per pulse, frame ids and device timestamps like
    PySpin's. `frame_fn(i)` renders the i-th image of `play` (DMD images, play order;
    the loop repeats when `repeat`); without play, a small dark frame.
    pulses=False: nothing arrives (wiring / level fault). skip: pulse indices the
    camera never sees (missed triggers: no frame, no id). drop: pulse indices whose
    frame is lost after the trigger (the frame id still advances). incomplete: pulse
    indices whose frame arrives incomplete (IncompleteFrame with its id and timestamp:
    USB3 packet loss). start: the play index of the first pulse after arming (a loop
    already playing). lead_us: arming -> the first pulse (default one period: the
    sequence is already playing). Like the real camera, the trigger is configured only
    while not acquiring, on an input line, and disarm() may be called again."""

    def __init__(self, period_us: float = 10000.0, *, frame_fn: Optional[Callable] = None,
                 n_play: int = 0, repeat: bool = True, start: int = 0, pulses: bool = True,
                 skip=(), drop=(), incomplete=(), jitter_us: float = 1.0, seed: int = 0,
                 lead_us: Optional[float] = None):
        self.period_us, self.frame_fn, self.n_play = float(period_us), frame_fn, int(n_play)
        self.repeat, self.start, self.pulses = bool(repeat), int(start), bool(pulses)
        self.skip, self.drop, self.jitter_us = set(skip), set(drop), float(jitter_us)
        self.incomplete = set(incomplete)
        self.lead_us = float(self.period_us if lead_us is None else lead_us)
        self._rng = np.random.default_rng(seed)
        self._t_s = 0.0
        self._k = 0              # pulses since arming
        self._fid = 0
        self.armed = False
        self.trigger = {}
        self.restored = 0

    def clock(self) -> float:
        return self._t_s

    def configure_trigger(self, line: str, exposure_us: float) -> None:
        if self.armed:
            raise RuntimeError("trigger configured while acquiring (TriggerMode Off and "
                               "EndAcquisition first)")
        check_line(line)
        self.trigger = {"TriggerMode": "On", "TriggerSource": line,
                        "TriggerSelector": "FrameStart", "TriggerActivation": "RisingEdge",
                        "ExposureTime": float(exposure_us)}

    def arm(self, n: int = 0) -> None:
        if not self.trigger.get("TriggerSource"):
            raise RuntimeError("armed without a configured trigger line")
        self.trigger["TriggerMode"] = "On"
        self.armed = True

    def disarm(self) -> None:
        self.armed = False
        self.trigger["TriggerMode"] = "Off"
        self.restored += 1

    def pending(self) -> int:
        return 1 if self.pulses and self.repeat else 0

    def _exhausted(self) -> bool:
        return not self.repeat and self.n_play and self.start + self._k >= self.n_play

    def grab_triggered(self, timeout_ms: int):
        if not self.armed:
            raise RuntimeError("camera not armed")
        while True:
            if not self.pulses or self._exhausted():
                self._t_s += timeout_ms / 1000.0
                raise TimeoutError("no trigger")
            k = self._k
            self._k += 1
            t_us = self.lead_us + k * self.period_us + float(self._rng.normal(0, self.jitter_us))
            gap_s = t_us / 1e6 - self._t_s
            if gap_s > timeout_ms / 1000.0:
                self._t_s += timeout_ms / 1000.0
                self._k -= 1
                raise TimeoutError("no trigger")
            self._t_s = max(self._t_s, t_us / 1e6)
            if k in self.skip:
                continue
            fid = self._fid
            self._fid += 1
            if k in self.drop:
                continue
            if k in self.incomplete:
                raise IncompleteFrame("incomplete frame (status 3)", fid, int(round(t_us * 1e3)))
            idx = (self.start + k) % self.n_play if self.n_play else None
            img = (self.frame_fn(idx) if self.frame_fn is not None and idx is not None
                   else np.zeros((8, 8), np.uint8))
            return img, fid, int(round(t_us * 1e3))


def mock_scene_camera(bench, patterns: list, period_us: float, *, exposure_us: float,
                      repeat: bool = True, start: Optional[int] = None, seed: int = 0,
                      **kw) -> FakeTriggeredCamera:
    """FakeTriggeredCamera over the mock bench: each pulse shows the next of `patterns`
    (DMD images, play order) through the same optics, ambient and noise as the soft-sync
    mock (mock_flashloop: projector defocus, LED level), held for the whole camera
    exposure - what a triggered capture of the pack sees."""
    from . import mock_flashloop as mfl
    rng = np.random.default_rng(seed)
    loop = mfl.MockFlashLoop(list(patterns), float(period_us), float(period_us),
                             seed=int(rng.integers(0, 2**31 - 1)))
    cam = mfl.MockFreeRunCamera(bench.render, loop, full_shape=tuple(bench.cam_shape),
                                seed=int(rng.integers(0, 2**31 - 1)))
    H, W = tuple(bench.cam_shape)
    cam.configure(roi=(0, 0, W, H), exposure_us=float(exposure_us), fps=1e6 / float(period_us))
    if start is None:
        start = int(rng.integers(0, len(patterns))) if repeat else 0
    return FakeTriggeredCamera(period_us, frame_fn=lambda i: cam.direct(i, 1),
                               n_play=len(patterns), repeat=repeat, start=start,
                               seed=int(rng.integers(0, 2**31 - 1)), **kw)


def expected_set_ms(periods_us=(10000, 5000, 4500), n_patterns: int = 16) -> list:
    """Pattern-set time at each period (the TRIGGER_FULLSPEED_TEST.md table)."""
    return [{"period_us": int(P), "set_ms": round(n_patterns * P / 1000.0, 3),
             "rate_hz": round(1e6 / P, 3), "fps_needed": round(1e6 / P, 1),
             "camera_ok_full_frame": P >= math.ceil(1e6 / 226.0)} for P in periods_us]

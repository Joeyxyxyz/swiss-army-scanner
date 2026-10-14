"""
Hardware-trigger pattern mode capture (ENGINE_ARCHITECTURE §9, build step 10).

A SEPARATE, EXPLICITLY ENTERED, GUARDED mode — never the default scan path.

What it does: the DLPC350 plays a pattern sequence out of flash (or from
the video port) with TRIG_OUT_1 wired to the camera's Line0; the camera is
in hardware-trigger mode and takes exactly one frame per pattern. This
module grabs exactly N frames, checks none was dropped (frame-id gaps and
timestamp gaps) and returns them in the order HybridBasis.decode expects
(white, black, gray planes, fringes) per axis.

What it does NOT do: it never sends a DLPC350 command. Putting the
projector into Pattern Sequence mode, loading the LUT and pressing Play is
a HUMAN step in the TI LightCrafter 4500 GUI for now (docs/
FLASH_PATTERN_MODE.md). It does not import engine.capture.dmd_control,
swiss_army_sensor.drivers or hid; `rules.check_scan_path_usb_free` is run
on entry.

Guard: TriggeredSequenceCapture refuses to exist unless the caller passes
acknowledge=ACK_PHRASE, i.e. states that a human has put the projector in
pattern mode and wired the trigger. The camera must be a triggered camera
(`grab_triggered(timeout_ms) -> (frame, frame_id, timestamp_ns)`); a free-
running camera is refused.

The PySpin trigger setup (`configure_pyspin_trigger`, `restore_free_run`,
`PySpinTriggeredCamera`) is bench code; tests run it only against a fake
PySpin module (tests/test_trigger_pyspin_mock.py), never a camera.

Stream buffers: the camera driver is asked for STREAM_BUFFERS at most, never
more than its own maximum (which moves with the PC's free memory), whatever
the length of the capture. Every frame is read as it arrives and converted
only after the last one, so the count covers a short stall of the reader,
not the capture. A count the driver does not give is a CameraSetupError
with the numbers - never a 'no pulses' result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .. import rules

ACK_PHRASE = "projector is in pattern mode, set by a human in the TI GUI"
# The camera lines a trigger may be wired to (docs/TRIGGER_FULLSPEED_TEST.md §5):
# Line0 = opto-isolated input (GPIO pin 2), Line3 = non-isolated input (pin 1).
# Line1 is the opto-isolated OUTPUT and Line2 the bidirectional line / 3.3 V
# auxiliary output (pin 3): never a trigger input here.
TRIGGER_INPUT_LINES = ("Line0", "Line3")
# Stream buffers asked of the camera driver for a triggered capture. A scan that asked
# for frames + 8 was refused on the bench (172 asked, 164 allowed with memory busy).
STREAM_BUFFERS = 64
STREAM_BUFFERS_MIN = 8        # the driver gives fewer: the camera is not armed


class PatternModeRefused(rules.SafetyViolation):
    """The guard was not satisfied; nothing was armed or grabbed."""


class CameraSetupError(RuntimeError):
    """The camera could not be set up for a triggered capture: the driver did not
    give the stream buffers. Raised before the camera is armed, so no trigger was
    looked at - not a TimeoutError, never counted as 'no pulses'."""


class MissingFrames(RuntimeError):
    """The camera did not deliver exactly one frame per pattern."""


class IncompleteFrame(RuntimeError):
    """A trigger arrived but its image did not (USB3 packet loss). frame_id /
    timestamp_ns: the camera's, when it still reported them (else None)."""

    def __init__(self, msg: str, frame_id: Optional[int] = None,
                 timestamp_ns: Optional[int] = None):
        super().__init__(msg)
        self.frame_id, self.timestamp_ns = frame_id, timestamp_ns


@dataclass
class TriggeredCapture:
    frames: list                  # sequence (= trigger) order, float32
    frame_ids: list
    timestamps_ns: list
    sequence: list                # manifest entries, same order
    report: dict = field(default_factory=dict)

    def axes(self) -> list:
        return sorted({e["axis"] for e in self.sequence})

    def frames_for(self, axis: str = "x") -> list:
        """HybridBasis order for one axis: white, black, gray..., fringes."""
        pairs = [(e["basis_index"], f) for e, f in zip(self.sequence, self.frames)
                 if e["axis"] == axis]
        if not pairs:
            raise ValueError(f"no patterns for axis {axis!r} in this sequence")
        pairs.sort(key=lambda p: p[0])
        idx = [p[0] for p in pairs]
        if idx != list(range(len(idx))):
            raise ValueError(f"axis {axis}: basis indices {idx} are not 0..{len(idx) - 1}")
        return [p[1] for p in pairs]


class TriggeredSequenceCapture:
    """Grab exactly len(sequence) hardware-triggered frames, in order."""

    def __init__(self, camera, manifest: dict, *, acknowledge: Optional[str] = None,
                 first_timeout_s: float = 30.0, frame_timeout_ms: Optional[int] = None,
                 gap_tolerance: float = 0.5, allow_extra: bool = False):
        if acknowledge != ACK_PHRASE:
            raise PatternModeRefused(
                "hardware-trigger pattern mode is a separate, explicitly entered mode. "
                "A human must put the DLPC350 in Pattern Sequence mode in the TI GUI "
                "(docs/FLASH_PATTERN_MODE.md), then pass acknowledge=patternmode.ACK_PHRASE. "
                "The engine never switches the projector mode itself.")
        rules.check_scan_path_usb_free()
        if not callable(getattr(camera, "grab_triggered", None)):
            raise PatternModeRefused(
                f"{type(camera).__name__} is not a hardware-triggered camera "
                "(needs grab_triggered); the free-run scan camera cannot be used here")
        seq = list(manifest.get("sequence") or [])
        if not seq:
            raise ValueError("manifest has no sequence")
        if [e["seq"] for e in seq] != list(range(len(seq))):
            raise ValueError("manifest sequence numbers are not 0..N-1 in order")
        self.camera = camera
        self.manifest = manifest
        self.sequence = seq
        self.period_us = self._expected_periods_us(manifest, seq)
        self.first_timeout_ms = int(first_timeout_s * 1000)
        longest = max(self.period_us) if self.period_us else 20000
        self.frame_timeout_ms = int(frame_timeout_ms if frame_timeout_ms is not None
                                    else max(1000, 10 * longest / 1000))
        self.gap_tolerance = float(gap_tolerance)
        self.allow_extra = bool(allow_extra)     # a repeating loop: frames keep coming
        # after the N frames, wait this long for one more (a sequence left on repeat)
        self.extra_probe_ms = int(max(250, 3 * longest / 1000))

    @staticmethod
    def _expected_periods_us(manifest: dict, seq: list) -> list:
        """Trigger-to-trigger time before each pattern (0 for the first)."""
        t = manifest.get("timing") or {}
        if t.get("mode") == "flash":
            per = t.get("per_bitdepth") or {}
            uni = (t.get("uniform") or {}).get("period_us")
            use_uniform = t.get("use") != "variable"
            out = []
            for e in seq:
                p = uni if use_uniform and uni else (per.get(str(e["bitdepth"])) or {}).get(
                    "period_us")
                out.append(int(p) if p else 0)
            return out
        if t.get("mode") == "video_port":
            frames = {f["frame_index"]: f["period_us"] for f in t.get("per_frame", [])}
            return [int(frames.get(e["image_index"], 0)) for e in seq]
        return [0] * len(seq)

    def capture(self) -> TriggeredCapture:
        n = len(self.sequence)
        frames, ids, ts = [], [], []
        extra = 0
        arm = getattr(self.camera, "arm", None)
        if callable(arm):
            arm(n)
        try:
            for k in range(n):
                timeout = self.first_timeout_ms if k == 0 else self.frame_timeout_ms
                try:
                    img, fid, t = self.camera.grab_triggered(timeout)
                except TimeoutError as e:
                    raise MissingFrames(
                        f"got {k} of {n} frames: no trigger within {timeout} ms "
                        f"(pattern {self.sequence[k]['name']}). Sequence not playing, "
                        "TRIG_OUT_1 not wired to the trigger line, or the sequence stopped "
                        "early") from e
                except IncompleteFrame as e:
                    raise MissingFrames(
                        f"frame {k + 1} of {n} (pattern {self.sequence[k]['name']}) arrived "
                        f"incomplete ({e}): image data lost on USB3 - use a USB3 port on its "
                        "own controller, a shorter cable, or a longer pattern period") from e
                # kept as grabbed: nothing slow between two grabs, so the reader keeps
                # up with the camera and STREAM_BUFFERS are enough for any length
                frames.append(_keep(img))
                ids.append(int(fid))
                ts.append(int(t))
            if not self.allow_extra:
                extra = self._extra_frames()     # while the camera still takes triggers
        finally:
            disarm = getattr(self.camera, "disarm", None)
            if callable(disarm):
                disarm()
        report = self._check(ids, ts)
        if extra:
            raise MissingFrames(
                f"{extra} extra frame(s) after the {n} expected: the sequence is set to "
                "repeat, or the camera was triggered by something else — uncheck Repeat")
        for k in range(n):                   # float32 one by one, the camera is disarmed
            frames[k] = np.asarray(frames[k], dtype=np.float32)
        return TriggeredCapture(frames, ids, ts, self.sequence, report)

    def _extra_frames(self) -> int:
        """Frames that keep coming after the N expected: the camera's own count
        (pending(), when it has one), else one more grab within extra_probe_ms -
        a frame (even an incomplete one) means the sequence did not stop."""
        pending = getattr(self.camera, "pending", None)
        if callable(pending):
            k = int(pending())
            if k > 0:
                return k
        try:
            self.camera.grab_triggered(self.extra_probe_ms)
        except TimeoutError:
            return 0
        except IncompleteFrame:
            return 1
        return 1

    def _check(self, ids: list, ts: list) -> dict:
        n = len(ids)
        gaps = [ids[i] - ids[i - 1] for i in range(1, n)]
        dropped = [i for i, g in enumerate(gaps, 1) if g != 1]
        if dropped:
            raise MissingFrames(
                f"camera frame ids jump before pattern(s) "
                f"{[self.sequence[i]['name'] for i in dropped]} (gaps {[gaps[i - 1] for i in dropped]}): "
                "the camera dropped or missed triggers — period shorter than the camera's "
                "readout, or USB bandwidth. Lengthen the pattern period")
        late = []
        dts = []
        for i in range(1, n):
            dt_us = (ts[i] - ts[i - 1]) / 1000.0
            dts.append(dt_us)
            exp = self.period_us[i]
            if exp and not (1 - self.gap_tolerance) * exp <= dt_us <= (1 + self.gap_tolerance) * exp:
                late.append((self.sequence[i]["name"], round(dt_us), exp))
        if late:
            raise MissingFrames(
                f"trigger spacing off the manifest period for {late} (name, measured us, "
                "expected us): a pattern was skipped by the camera or the GUI timing differs "
                "from the manifest")
        return {"n_frames": n, "frame_ids": [ids[0], ids[-1]],
                "trigger_dt_us": {"min": round(min(dts), 1) if dts else None,
                                  "max": round(max(dts), 1) if dts else None},
                "sequence_ms": round((ts[-1] - ts[0]) / 1e6, 3) if n > 1 else 0.0}


def _keep(img):
    """A grabbed frame, safe to hold until the capture ends: a view of a buffer the
    camera may fill again is copied, anything else is kept as it came."""
    if isinstance(img, np.ndarray) and not img.flags.owndata:
        return img.copy()
    return img


# ============================================================ bench
# PySpin is passed in (or imported lazily) so importing this module never needs
# the camera SDK. Tests run this code only against a fake PySpin module.

def require_input_line(cam, PySpin, line: str) -> None:
    """The trigger line must be a camera INPUT: one of TRIGGER_INPUT_LINES,
    selected, LineMode set to Input where the camera lets it be written (a
    non-isolated line may have been left an output by another program), and
    read back as Input - else PatternModeRefused before the trigger is wired
    to it in software (a push-pull trigger output against a camera output
    driver is contention)."""
    if line not in TRIGGER_INPUT_LINES:
        raise PatternModeRefused(
            f"trigger line {line!r} is not a camera input: use one of {TRIGGER_INPUT_LINES} "
            "(Line1 is the opto output, Line2 the 3.3 V auxiliary output / bidirectional pin "
            "3; docs/TRIGGER_FULLSPEED_TEST.md)")
    cam.LineSelector.SetValue(getattr(PySpin, f"LineSelector_{line}"))
    if cam.LineMode.GetAccessMode() == PySpin.RW:
        cam.LineMode.SetValue(PySpin.LineMode_Input)
    if int(cam.LineMode.GetValue()) != int(PySpin.LineMode_Input):
        raise PatternModeRefused(f"camera {line} does not read back as an input (LineMode "
                                 f"{cam.LineMode.GetValue()!r}): not used as the trigger")


def _stream_buffer_max(stream) -> Optional[int]:
    """The most stream buffers the driver allows now (it follows the PC's free
    memory), or None when the driver does not say."""
    for read in (lambda: stream.StreamBufferCountManual.GetMax(),
                 lambda: stream.StreamBufferCountMax.GetValue()):
        try:
            return int(read())
        except Exception:  # noqa: BLE001 - not reported this way by this driver
            continue
    return None


def set_stream_buffers(cam, PySpin, want: int = STREAM_BUFFERS) -> dict:
    """Manual stream buffer count: `want`, at most STREAM_BUFFERS and never more than
    the driver's maximum -> {"asked", "max", "set"} (max None: not reported).
    CameraSetupError, with the numbers, when the maximum is under STREAM_BUFFERS_MIN,
    when the driver refuses the count, or when fewer than that read back."""
    s = cam.TLStream
    s.StreamBufferCountMode.SetValue(PySpin.StreamBufferCountMode_Manual)
    asked = max(STREAM_BUFFERS_MIN, min(int(want), STREAM_BUFFERS))
    nmax = _stream_buffer_max(s)
    said = "not reported" if nmax is None else str(nmax)
    hint = ("free memory on the PC (close other programs) and run again. The camera was "
            "not armed: this is a camera set-up error, the trigger was not looked at")
    if nmax is not None and nmax < STREAM_BUFFERS_MIN:
        raise CameraSetupError(
            f"stream buffers: the camera driver allows only {nmax} now ({asked} asked, at "
            f"least {STREAM_BUFFERS_MIN} needed); {hint}")
    n = asked if nmax is None else min(asked, nmax)
    try:
        s.StreamBufferCountManual.SetValue(int(n))
    except Exception as e:  # noqa: BLE001 - whatever the driver raises, it is this request
        raise CameraSetupError(
            f"stream buffers: the camera driver refused {n} ({asked} asked, driver maximum "
            f"{said}): {e}; {hint}") from e
    try:
        got = int(s.StreamBufferCountManual.GetValue())
    except Exception:  # noqa: BLE001 - no read-back: what was written counts
        got = n
    if got < STREAM_BUFFERS_MIN:
        raise CameraSetupError(
            f"stream buffers: {n} were set but the camera driver holds {got} ({asked} "
            f"asked, driver maximum {said}, at least {STREAM_BUFFERS_MIN} needed); {hint}")
    return {"asked": asked, "max": nmax, "set": got}


def configure_pyspin_trigger(cam, PySpin=None, *, exposure_us: float,
                             source: str = "Line0", activation: str = "RisingEdge",
                             buffers: int = STREAM_BUFFERS) -> dict:
    """Hardware-trigger a Blackfly S (acquisition must be stopped) -> what was
    applied: {"buffers": {"asked", "max", "set"}} (set_stream_buffers).
    Order matters: TriggerMode Off before changing selector/source; the
    source line is checked to be an input first (require_input_line). The
    stream buffers come before TriggerMode On: a count the driver does not give
    (CameraSetupError) leaves the trigger off."""
    if PySpin is None:
        import PySpin  # noqa: N813
    cam.TriggerMode.SetValue(PySpin.TriggerMode_Off)
    require_input_line(cam, PySpin, source)
    cam.TriggerSelector.SetValue(PySpin.TriggerSelector_FrameStart)
    cam.TriggerSource.SetValue(getattr(PySpin, f"TriggerSource_{source}"))
    cam.TriggerActivation.SetValue(getattr(PySpin, f"TriggerActivation_{activation}"))
    try:
        cam.TriggerOverlap.SetValue(PySpin.TriggerOverlap_ReadOut)
    except Exception:  # noqa: BLE001 - older firmware lacks the node
        pass
    cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
    cam.ExposureMode.SetValue(PySpin.ExposureMode_Timed)
    cam.ExposureTime.SetValue(float(exposure_us))
    cam.AcquisitionMode.SetValue(PySpin.AcquisitionMode_Continuous)
    # every frame matters: keep them all, oldest first
    cam.TLStream.StreamBufferHandlingMode.SetValue(PySpin.StreamBufferHandlingMode_OldestFirst)
    applied = {"buffers": set_stream_buffers(cam, PySpin, buffers)}
    cam.TriggerMode.SetValue(PySpin.TriggerMode_On)
    return applied


def restore_free_run(cam, PySpin=None) -> None:
    if PySpin is None:
        import PySpin  # noqa: N813
    cam.TriggerMode.SetValue(PySpin.TriggerMode_Off)
    cam.TLStream.StreamBufferHandlingMode.SetValue(PySpin.StreamBufferHandlingMode_NewestOnly)


def _is_timeout(e: BaseException, PySpin) -> bool:
    """A SpinnakerException that is GetNextImage's grab timeout (and nothing
    else: a lost camera or a stopped stream is an error, not 'no pulse')."""
    code = getattr(PySpin, "SPINNAKER_ERR_TIMEOUT", -1011)
    got = getattr(e, "errorcode", None)
    if got is not None:
        return int(got) == int(code)
    return f"[{int(code)}]" in str(e)      # Spinnaker messages end with "[<code>]"


class PySpinTriggeredCamera:
    """grab_triggered() over an already-open PySpin camera `cam`. The caller
    closes the camera: disarm() (EndAcquisition, TriggerMode Off), then
    release() (drops this object's camera reference: PySpin refuses to release
    its system while one is alive), then DeInit and the system release."""

    def __init__(self, cam, PySpin=None, *, exposure_us: float, source: str = "Line0"):
        if PySpin is None:
            import PySpin  # noqa: N813
        self.cam, self.PySpin = cam, PySpin
        self.exposure_us, self.source = exposure_us, source
        self.applied = {}                    # what the last arm() set (stream buffers)

    def arm(self, n: int) -> dict:
        """Trigger on, acquisition started -> what was applied (kept in .applied).
        The stream buffers do not follow n (the frames expected): STREAM_BUFFERS at
        most, never more than the driver allows - the caller reads every frame as it
        arrives. CameraSetupError when the driver does not give them."""
        try:
            self.cam.EndAcquisition()
        except Exception:  # noqa: BLE001 - was not streaming
            pass
        self.applied = configure_pyspin_trigger(self.cam, self.PySpin,
                                                exposure_us=self.exposure_us,
                                                source=self.source, buffers=STREAM_BUFFERS)
        self.cam.BeginAcquisition()
        return self.applied

    def grab_triggered(self, timeout_ms: int):
        """-> (image, frame id, timestamp ns). TimeoutError only for the grab
        timeout; IncompleteFrame (with the frame id / timestamp when the camera
        still gives them) for an image lost on USB; other camera errors raise."""
        try:
            img = self.cam.GetNextImage(int(timeout_ms))
        except self.PySpin.SpinnakerException as e:
            if _is_timeout(e, self.PySpin):
                raise TimeoutError(str(e)) from e
            raise
        try:
            if img.IsIncomplete():
                fid = t = status = None
                try:
                    fid, t = int(img.GetFrameID()), int(img.GetTimeStamp())
                except Exception:  # noqa: BLE001 - not reported for this frame
                    fid = t = None
                try:
                    status = img.GetImageStatus()
                except Exception:  # noqa: BLE001 - status is only for the message
                    pass
                raise IncompleteFrame(f"incomplete frame (status {status})", fid, t)
            return img.GetNDArray().copy(), img.GetFrameID(), img.GetTimeStamp()
        finally:
            img.Release()

    def disarm(self) -> None:
        """EndAcquisition (guarded: it may have ended already), then TriggerMode
        Off - every time; safe to call again. Raises only when TriggerMode Off
        (or the free-run buffering) cannot be set."""
        if self.cam is None:
            return
        try:
            self.cam.EndAcquisition()
        except Exception:  # noqa: BLE001 - not acquiring (disarmed already)
            pass
        restore_free_run(self.cam, self.PySpin)

    def release(self) -> None:
        """Drop the camera reference (after disarm, before the system release)."""
        self.cam = None

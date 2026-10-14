"""Stream buffers of the hardware-triggered camera (engine/capture/patternmode.py), against
tests/fake_pyspin.py: no SDK, no camera, no USB.

The camera driver is asked for patternmode.STREAM_BUFFERS at most and never for more than
its own maximum, whatever the length of the capture; the capture reads every frame as it
arrives, so a long capture still keeps them all. A count the driver does not give is a
CameraSetupError with the numbers, never a 'no pulses' result."""

import sys
import time
import types

import numpy as np
import pytest

from engine.capture import patternmode as pm
from engine.capture import riglock
from engine.reconstruct import trigger_scan

from fake_pyspin import FakePySpin, SpinnakerException


def _open(fake, line="Line0"):
    cam = fake.System.GetInstance().GetCameras()[0]
    cam.Init()
    return cam, pm.PySpinTriggeredCamera(cam, fake, exposure_us=1000.0, source=line)


def _manifest(n, period_us):
    return trigger_scan._capture_manifest([f"p{k}" for k in range(n)], n, period_us)


def _capture(tc, n, period_us, **kw):
    return pm.TriggeredSequenceCapture(tc, _manifest(n, period_us), acknowledge=pm.ACK_PHRASE,
                                       first_timeout_s=2.0, **kw).capture()


@pytest.fixture
def rig(tmp_path, monkeypatch):
    """-> install(**fake kwargs): a FakePySpin as `PySpin`, a temp rig lock, and the
    stand-ins for a non-mock session and a preset (the real-camera path)."""
    monkeypatch.setattr(riglock, "lock_candidates", lambda: [tmp_path / "rig.lock"])

    def install(**kw):
        fake = FakePySpin(**kw)
        monkeypatch.setitem(sys.modules, "PySpin", fake)
        import PySpin
        assert PySpin is fake                    # never the vendor SDK
        s = types.SimpleNamespace(mock=False, bench=None, holder="pytest")
        preset = types.SimpleNamespace(camera=types.SimpleNamespace(exposure_us=1000.0,
                                                                    gain_db=0.0))
        return fake, s, preset
    return install


# ------------------------------------------------------------- the fake driver

def test_fake_driver_refuses_a_count_above_its_maximum():
    fake = FakePySpin(buffer_max=164)
    cam = fake.System.GetInstance().GetCameras()[0]
    cam.Init()
    node = cam.TLStream.StreamBufferCountManual
    assert node.GetMax() == 164 and cam.TLStream.StreamBufferCountMax.GetValue() == 164
    with pytest.raises(SpinnakerException, match="Value = 172 must be equal or smaller than "
                                                 "Max = 164"):
        node.SetValue(172)
    node.SetValue(164)
    fake.buffer_max = 421                        # it moves with the free memory
    node.SetValue(172)
    assert node.GetValue() == 172


# ------------------------------------------------------------- the count asked

def test_long_capture_keeps_every_frame_with_a_low_buffer_maximum():
    """500 frames expected, the driver allows 164 buffers: 64 are asked (not 508), the
    capture runs and every frame is kept, in order."""
    fake = FakePySpin(period_s=0.003, n_pulses=500, buffer_max=164)
    cam, tc = _open(fake)
    cap = _capture(tc, 500, 3000.0)
    assert tc.applied == {"buffers": {"asked": 64, "max": 164, "set": 64}}
    assert fake.last_value("StreamBufferCountManual") == 64
    assert len(cap.frames) == 500 and cap.frame_ids == list(range(500))
    assert fake.device.lost == 0
    assert all(f.dtype == np.float32 and f.shape == (6, 8) for f in cap.frames)
    assert cap.report["n_frames"] == 500
    assert fake.last_value("TriggerMode") == "TriggerMode_Off"


@pytest.mark.parametrize("buffer_max,expected,want", [
    (1024, 64, {"asked": 64, "max": 1024, "set": 64}),     # never frames + 8
    (40, 300, {"asked": 64, "max": 40, "set": 40}),        # never above the driver's maximum
    (8, 16, {"asked": 64, "max": 8, "set": 8}),
])
def test_arm_asks_for_64_at_most_and_never_above_the_maximum(buffer_max, expected, want):
    fake = FakePySpin(buffer_max=buffer_max)
    cam, tc = _open(fake)
    assert tc.arm(expected) == {"buffers": want} and tc.applied == {"buffers": want}
    assert fake.device.buffers == want["set"]
    assert [e[2] for e in fake.events("set") if e[1] == "StreamBufferCountManual"] == [
        want["set"]]
    tc.disarm()


def test_capture_runs_on_fewer_buffers_than_asked():
    fake = FakePySpin(period_s=0.004, n_pulses=100, buffer_max=40)
    cam, tc = _open(fake)
    cap = _capture(tc, 100, 4000.0)
    assert tc.applied["buffers"] == {"asked": 64, "max": 40, "set": 40}
    assert cap.frame_ids == list(range(100)) and fake.device.lost == 0


def test_the_maximum_is_read_again_for_every_capture():
    fake = FakePySpin(period_s=0.002, n_pulses=8, buffer_max=421)
    cam, tc = _open(fake)
    _capture(tc, 8, 2000.0)
    assert tc.applied["buffers"]["set"] == 64
    fake.buffer_max = 30                         # memory got busy in between
    cap = _capture(tc, 8, 2000.0)
    assert tc.applied["buffers"] == {"asked": 64, "max": 30, "set": 30} and len(cap.frames) == 8


# ------------------------------------------------------------- a count the driver does not give

@pytest.mark.parametrize("kw,words", [
    ({"buffer_max": 4}, "allows only 4 now (64 asked, at least 8 needed)"),
    ({"fail_buffers": True}, "refused 64 (64 asked, driver maximum 1024)"),
])
def test_refused_buffers_are_a_camera_setup_error_with_the_numbers(kw, words):
    fake = FakePySpin(**kw)
    cam, tc = _open(fake)
    with pytest.raises(pm.CameraSetupError) as ei:
        tc.arm(500)
    msg = str(ei.value)
    assert msg.startswith("stream buffers: the camera driver") and words in msg
    assert "the trigger was not looked at" in msg and "no pulses" not in msg.lower()
    assert not isinstance(ei.value, (TimeoutError, pm.MissingFrames))
    assert not fake.device.acquiring                     # never armed
    assert fake.last_value("TriggerMode") == "TriggerMode_Off"
    if "fail_buffers" in kw:
        assert isinstance(ei.value.__cause__, SpinnakerException)    # the driver's own words
        assert "Could not allocate the stream buffers" in msg
    tc.disarm()


@pytest.mark.parametrize("kw", [{"buffer_max": 4}, {"fail_buffers": True}])
def test_trigger_test_reports_refused_buffers_not_no_pulses(rig, tmp_path, kw):
    """The trigger test on the real-camera path: the pulses are there (the fake keeps
    pulsing), the driver does not give the buffers. The command raises the buffer error;
    it does not return a NO PULSES verdict. The camera is closed and the lock is free."""
    fake, s, preset = rig(period_s=0.005, **kw)
    with pytest.raises(pm.CameraSetupError, match="stream buffers") as ei:
        trigger_scan.trigger_test(s, preset, "Line0", 0.3, 5.0)
    assert "64 asked" in str(ei.value) and "no pulses" not in str(ei.value).lower()
    assert fake.released and not fake.device.init, fake.log
    assert fake.last_value("TriggerMode") == "TriggerMode_Off"
    riglock.RigLock(path=tmp_path / "rig.lock").acquire().release()


def test_scan_capture_reports_refused_buffers_not_missing_frames(rig, tmp_path):
    fake, s, preset = rig(period_s=0.005, n_pulses=16, fail_buffers=True)
    with pytest.raises(pm.CameraSetupError, match="refused 64"):
        with trigger_scan.triggered_camera(s, preset, line="Line3", exposure_us=1000.0) as cam:
            _capture(cam, 16, 5000.0)
    assert fake.released and not fake.device.init, fake.log
    riglock.RigLock(path=tmp_path / "rig.lock").acquire().release()


class _Node:
    """A stream node of a driver that is not the fake: value, optional maximum."""

    def __init__(self, value=0, maximum=None, keeps=None):
        self.value, self.maximum, self.keeps = value, maximum, keeps

    def GetValue(self):
        if self.value is None:
            raise RuntimeError("not readable")
        return self.value

    def GetMax(self):
        if self.maximum is None:
            raise RuntimeError("no maximum")
        return self.maximum

    def SetValue(self, v):
        self.value = v if self.keeps is None else self.keeps


def _driver(manual, count_max=None):
    stream = types.SimpleNamespace(StreamBufferCountMode=_Node(), StreamBufferCountManual=manual,
                                   StreamBufferCountMax=_Node(count_max))
    return types.SimpleNamespace(TLStream=stream), types.SimpleNamespace(
        StreamBufferCountMode_Manual="Manual")


def test_set_stream_buffers_on_other_drivers():
    # no maximum reported at all: the count asked is written
    cam, P = _driver(_Node(10))
    assert pm.set_stream_buffers(cam, P) == {"asked": 64, "max": None, "set": 64}
    assert cam.TLStream.StreamBufferCountMode.value == "Manual"
    # the maximum only as its own node
    cam, P = _driver(_Node(10), count_max=20)
    assert pm.set_stream_buffers(cam, P) == {"asked": 64, "max": 20, "set": 20}
    # a caller's own count: 64 at most, the minimum at least
    cam, P = _driver(_Node(10, maximum=500))
    assert pm.set_stream_buffers(cam, P, 300) == {"asked": 64, "max": 500, "set": 64}
    assert pm.set_stream_buffers(cam, P, 16) == {"asked": 16, "max": 500, "set": 16}
    assert pm.set_stream_buffers(cam, P, 2)["set"] == pm.STREAM_BUFFERS_MIN
    # the driver takes the write but holds fewer than can work
    cam, P = _driver(_Node(10, maximum=500, keeps=3))
    with pytest.raises(pm.CameraSetupError, match="64 were set but the camera driver holds 3"):
        pm.set_stream_buffers(cam, P)


# ------------------------------------------------------------- every frame is read as it arrives

class _SlowReader:
    """A triggered camera whose reader is late by delay_s before every grab."""

    def __init__(self, tc, delay_s):
        self.tc, self.delay_s = tc, delay_s
        self.arm, self.disarm = tc.arm, tc.disarm

    def grab_triggered(self, timeout_ms):
        time.sleep(self.delay_s)
        return self.tc.grab_triggered(timeout_ms)


def test_reader_slower_than_the_camera_is_refused_never_a_short_set():
    """Pulses every 1 ms, the reader takes one frame every 5 ms, 8 buffers: frames are
    lost in the driver. The capture says so (frame ids jump); it does not return."""
    fake = FakePySpin(period_s=0.001, buffer_max=8)
    cam, tc = _open(fake)
    with pytest.raises(pm.MissingFrames, match="frame ids jump"):
        _capture(_SlowReader(tc, 0.005), 40, 1000.0, allow_extra=True)
    assert fake.device.lost > 0
    assert fake.last_value("TriggerMode") == "TriggerMode_Off"


class _Lazy:
    """A frame that notes when it is turned into an array."""

    def __init__(self, log, k):
        self.log, self.k = log, k

    def __array__(self, dtype=None, copy=None):
        self.log.append(("convert", self.k))
        return np.full((4, 6), self.k, dtype or np.uint8)


class _NotingCamera:
    def __init__(self, frame):
        self.log, self.k, self.frame = [], 0, frame

    def arm(self, n):
        self.log.append(("arm", n))

    def disarm(self):
        self.log.append(("disarm",))

    def grab_triggered(self, timeout_ms):
        k, self.k = self.k, self.k + 1
        self.log.append(("grab", k))
        return self.frame(self.log, k), k, k * 5_000_000


def test_frames_are_converted_only_after_the_last_grab():
    """Nothing slow between two grabs: the float32 conversion of every frame comes
    after the camera is disarmed, so the reader keeps up with the camera."""
    cam = _NotingCamera(_Lazy)
    cap = _capture(cam, 12, 5000.0, allow_extra=True)
    kinds = [e[0] for e in cam.log]
    assert kinds == ["arm"] + ["grab"] * 12 + ["disarm"] + ["convert"] * 12, cam.log
    assert [int(f[0, 0]) for f in cap.frames] == list(range(12))
    assert all(f.dtype == np.float32 for f in cap.frames)


def test_a_view_of_a_reused_camera_buffer_is_copied_at_once():
    buf = np.zeros((4, 6), np.uint8)

    def frame(log, k):
        buf[:] = k + 1                           # the camera fills its one buffer again
        return buf[:]

    cap = _capture(_NotingCamera(frame), 5, 5000.0, allow_extra=True)
    assert [int(f[0, 0]) for f in cap.frames] == [1, 2, 3, 4, 5]

"""The real-camera code path of the hardware trigger (engine/reconstruct/trigger_scan.py
triggered_camera, engine/capture/patternmode.py PySpinTriggeredCamera, the engine's
RealCamera and the seed camera driver) run against tests/fake_pyspin.py: a fake PySpin
module that enforces Spinnaker's ordering rules. No SDK, no camera, no USB: the fake is
installed in sys.modules before anything imports PySpin, and the rig lock goes to a
temporary file."""

import sys
import types

import pytest

from engine.capture import hwtrigger as hw
from engine.capture import patternmode as pm
from engine.capture import riglock
from engine.reconstruct import trigger_scan

from fake_pyspin import SPINNAKER_ERR_TIMEOUT, FakePySpin, SpinnakerException


@pytest.fixture
def rig(tmp_path, monkeypatch):
    """-> install(**fake kwargs): a FakePySpin as `PySpin`, a temp rig lock, and the
    stand-ins for a non-mock session and a preset."""
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


def _lock_free(tmp_path):
    lk = riglock.RigLock(path=tmp_path / "rig.lock").acquire()
    lk.release()


def _assert_closed_in_order(fake):
    """TriggerMode Off after the last EndAcquisition, then DeInit, every camera object
    gone (de-initialised), then the system released - and nothing refused."""
    assert fake.released, fake.log
    assert ("ReleaseInstance", "refused") not in [e[:2] for e in fake.log]
    assert fake.last_value("TriggerMode") == "TriggerMode_Off"
    i_end = fake.index(("EndAcquisition",))
    i_off = fake.index(("set", "TriggerMode", "TriggerMode_Off"))
    i_deinit = fake.index(("DeInit",))
    i_gone = fake.index(("destroyed",))
    i_rel = fake.index(("ReleaseInstance", "ok"))
    assert None not in (i_end, i_off, i_deinit, i_gone, i_rel), fake.log
    assert i_end < i_off < i_deinit < i_gone < i_rel, fake.log
    assert fake.events("destroyed") == [("destroyed", "deinit")]
    assert not fake.device.init and not fake.device.acquiring


def test_real_trigger_test_closes_camera_in_order(rig, tmp_path):
    """A successful trigger test on the real path returns its verdict: the camera is
    disarmed, the camera reference dropped, de-initialised and its system released
    (PySpin refuses ReleaseInstance while a camera reference lives - the old cleanup
    raised here on every real run)."""
    fake, s, preset = rig(period_s=0.005)
    r = trigger_scan.trigger_test(s, preset, "Line0", 0.3, 5.0)
    assert r["verdict"] == "OK" and r["count"] >= 40 and not r["mock"], r
    assert r["trigger_mode_restored"] == "Off"
    _assert_closed_in_order(fake)
    _lock_free(tmp_path)


def test_real_scan_capture_closes_camera_in_order(rig, tmp_path):
    """The scan path's capture (TriggeredSequenceCapture inside triggered_camera)."""
    fake, s, preset = rig(period_s=0.005, n_pulses=16)
    man = trigger_scan._capture_manifest([f"p{k}" for k in range(16)], 16, 5000.0)
    with trigger_scan.triggered_camera(s, preset, line="Line3", exposure_us=1000.0) as cam:
        cap = pm.TriggeredSequenceCapture(cam, man, acknowledge=pm.ACK_PHRASE,
                                          first_timeout_s=2.0).capture()
    assert len(cap.frames) == 16
    _assert_closed_in_order(fake)
    _lock_free(tmp_path)


@pytest.mark.parametrize("kw,err", [
    ({"n_pulses": 0}, pm.MissingFrames),            # no trigger: refused mid-capture
    ({"fail_source": True}, SpinnakerException),    # arm fails inside the trigger setup
])
def test_real_path_error_keeps_cause_and_still_releases(rig, tmp_path, kw, err):
    """An error while armed propagates as itself, and the camera is still closed in the
    same order (its traceback frames no longer hold the camera)."""
    fake, s, preset = rig(**kw)
    man = trigger_scan._capture_manifest([f"p{k}" for k in range(4)], 4, 5000.0)
    with pytest.raises(err) as ei:
        with trigger_scan.triggered_camera(s, preset, line="Line0", exposure_us=1000.0) as cam:
            pm.TriggeredSequenceCapture(cam, man, acknowledge=pm.ACK_PHRASE,
                                        first_timeout_s=0.05).capture()
    assert not any("rig cleanup" in n for n in getattr(ei.value, "__notes__", []))
    assert fake.released and not fake.device.init, fake.log
    assert fake.last_value("TriggerMode") == "TriggerMode_Off"
    _lock_free(tmp_path)


def test_real_path_repeating_sequence_is_refused(rig):
    """PySpinTriggeredCamera has no pending(): the extra-frame check grabs once more, so a
    sequence left on Repeat is refused on the real camera too."""
    fake, s, preset = rig(period_s=0.005)                     # keeps pulsing
    man = trigger_scan._capture_manifest([f"p{k}" for k in range(8)], 8, 5000.0)
    with pytest.raises(pm.MissingFrames, match="extra frame"):
        with trigger_scan.triggered_camera(s, preset, line="Line0", exposure_us=1000.0) as cam:
            pm.TriggeredSequenceCapture(cam, man, acknowledge=pm.ACK_PHRASE,
                                        first_timeout_s=2.0).capture()
    assert fake.released


def test_real_path_refuses_line2_before_opening(rig):
    fake, s, preset = rig()
    with pytest.raises(ValueError, match="pin 3"):
        with trigger_scan.triggered_camera(s, preset, line="Line2", exposure_us=1000.0):
            pass
    assert fake.log == []                                     # the camera was never opened


# ------------------------------------------------------------- PySpinTriggeredCamera

def _armed(fake, line="Line0"):
    cam = fake.System.GetInstance().GetCameras()[0]
    cam.Init()
    tc = pm.PySpinTriggeredCamera(cam, fake, exposure_us=1000.0, source=line)
    tc.arm(16)
    return cam, tc


def test_grab_maps_only_the_timeout_to_timeouterror():
    fake = FakePySpin(n_pulses=2, period_s=0.002)
    cam, tc = _armed(fake)
    tc.grab_triggered(100)
    tc.grab_triggered(100)
    with pytest.raises(TimeoutError):
        tc.grab_triggered(20)                                 # the train stopped
    fake.error_at = 0
    with pytest.raises(SpinnakerException) as ei:             # a lost stream is no 'no pulse'
        tc.grab_triggered(20)
    assert ei.value.errorcode != SPINNAKER_ERR_TIMEOUT
    assert not isinstance(ei.value, TimeoutError)
    tc.disarm()
    tc.disarm()                                               # safe again
    assert fake.last_value("TriggerMode") == "TriggerMode_Off"


def test_incomplete_frame_is_counted_not_fatal():
    fake = FakePySpin(period_s=0.002, incomplete={5})
    cam, tc = _armed(fake)
    try:
        ts, ids, timeouts, inc = hw.count_pulses(tc, 0.1, timeout_ms=100)
    finally:
        tc.disarm()
    assert inc == 1 and timeouts == 0 and len(ts) >= 30
    assert ids == list(range(len(ids)))                       # its pulse is kept
    r = hw.trigger_stats(ts, ids, seconds=0.1, expect_period_ms=2.0, timeouts=timeouts,
                         incomplete=inc)
    assert r["verdict"] == "RATE MISMATCH" and r["incomplete_frames"] == 1
    assert any("incomplete" in n for n in r["notes"])


def test_trigger_line_must_read_back_as_input():
    # Line3 left as an output by another program: set to Input before the trigger
    fake = FakePySpin(line_modes={"Line3": "Output"})
    cam, tc = _armed(fake, "Line3")
    assert fake.device.line_mode["Line3"] == fake.enum("LineMode_Input")
    i_mode = fake.index(("set", "LineMode", "LineMode_Input"))
    i_src = fake.index(("set", "TriggerSource", "TriggerSource_Line3"))
    assert i_mode < i_src
    tc.disarm()
    # Line0 (opto input) is read-only Input: nothing to write, accepted
    fake = FakePySpin()
    cam, tc = _armed(fake, "Line0")
    assert fake.index(("set", "LineMode")) is None
    tc.disarm()
    # Line2 (3.3 V aux out / bidirectional) is refused before any trigger node is touched
    fake = FakePySpin()
    cam = fake.System.GetInstance().GetCameras()[0]
    cam.Init()
    with pytest.raises(pm.PatternModeRefused, match="Line2"):
        pm.configure_pyspin_trigger(cam, fake, exposure_us=1000.0, source="Line2")
    assert fake.index(("set", "TriggerSource")) is None
    assert fake.last_value("TriggerMode") == "TriggerMode_Off"


def test_trigger_line_that_stays_an_output_is_refused():
    fake = FakePySpin(line_modes={"Line3": "Output"}, stuck_lines={"Line3"})
    cam = fake.System.GetInstance().GetCameras()[0]
    cam.Init()
    with pytest.raises(pm.PatternModeRefused, match="input"):
        pm.configure_pyspin_trigger(cam, fake, exposure_us=1000.0, source="Line3")
    assert fake.index(("set", "TriggerSource")) is None
    assert fake.last_value("TriggerMode") == "TriggerMode_Off"

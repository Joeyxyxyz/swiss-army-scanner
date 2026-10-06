"""Real-rig code paths (RealCamera = PySpin, HDMIWindow = cv2 fullscreen)
exercised against in-memory fakes. Nothing here opens a camera, a window
or a USB device: PySpin and cv2 are replaced in sys.modules for the test,
and the seed CameraDriver is swapped for a fake."""

import logging
import sys
import types

import numpy as np
import pytest

from engine.capture import camera as camera_mod
from engine.capture import projector as projector_mod
from engine.config import CameraCfg


# ------------------------------------------------------------ fake PySpin

class Node:
    def __init__(self, value=0, max_value=None, fail=None):
        self.value, self.max_value, self.fail = value, max_value, fail
        self.sets = []

    def SetValue(self, v):
        if self.fail:
            raise RuntimeError(self.fail)
        self.sets.append(v)
        self.value = v

    def GetValue(self):
        return self.value

    def GetMax(self):
        return self.max_value


class Image:
    def __init__(self, incomplete=False, value=7):
        self.incomplete, self.value, self.released = incomplete, value, False

    def IsIncomplete(self):
        return self.incomplete

    def GetImageStatus(self):
        return 3

    def GetNDArray(self):
        return np.full((4, 6), self.value, np.uint8)

    def Release(self):
        self.released = True


class Cam:
    def __init__(self, missing=(), locked=()):
        self.calls = []
        self.images = []
        self.BinningHorizontal, self.BinningVertical = Node(2), Node(2)
        self.OffsetX, self.OffsetY = Node(64), Node(32)
        self.Width, self.Height = Node(720, 1440), Node(540, 1080)
        self.ReverseX, self.ReverseY = Node(True), Node(True)
        self.GammaEnable = Node(True)
        self.AcquisitionFrameRateEnable = Node(True)
        self.TriggerMode = Node("On")
        self.ExposureTime = Node(8000.0)
        self.TLStream = types.SimpleNamespace(StreamBufferHandlingMode=Node("OldestFirst"))
        for name in missing:
            delattr(self, name)
        for name in locked:
            getattr(self, name).fail = f"{name} is not writable"

    def EndAcquisition(self):
        self.calls.append("EndAcquisition")

    def BeginAcquisition(self):
        self.calls.append("BeginAcquisition")

    def DeInit(self):
        self.calls.append("DeInit")

    def GetNextImage(self, timeout_ms):
        self.calls.append(("GetNextImage", timeout_ms))
        return self.images.pop(0) if self.images else Image()


class Driver:
    """The seed CameraDriver's shape: cfg, open(), close(), _cam."""
    instances = []

    def __init__(self, cam):
        self.cfg = types.SimpleNamespace(trigger_mode="hardware", exposure_us=0, gain_db=0.0)
        self._cam_to_open, self._cam = cam, None
        self.closed = 0
        Driver.instances.append(self)

    def open(self):
        self._cam = self._cam_to_open
        return self

    def close(self):
        self.closed += 1


@pytest.fixture
def fake_spin(monkeypatch):
    """-> function(cam) that makes RealCamera open `cam` through a fake driver."""
    spin = types.ModuleType("PySpin")
    spin.TriggerMode_Off = "Off"
    spin.StreamBufferHandlingMode_NewestOnly = "NewestOnly"
    monkeypatch.setitem(sys.modules, "PySpin", spin)
    Driver.instances = []

    def use(cam):
        seed = types.SimpleNamespace(CameraDriver=lambda: Driver(cam))
        monkeypatch.setattr(camera_mod, "_seed", types.SimpleNamespace(load=lambda name: seed))
        return cam
    return use


def test_real_camera_resets_state_after_open(fake_spin):
    cam = fake_spin(Cam())
    rc = camera_mod.RealCamera(CameraCfg(exposure_us=16667, gain_db=3.0))
    drv = Driver.instances[0]
    assert drv.cfg.trigger_mode == "free" and drv.cfg.exposure_us == 16667
    assert rc.reset_state["failed"] == {}
    assert rc.reset_state["reset"] == [
        "BinningHorizontal", "BinningVertical", "OffsetX", "OffsetY", "Width", "Height",
        "ReverseX", "ReverseY", "GammaEnable", "AcquisitionFrameRateEnable", "TriggerMode"]
    assert (cam.BinningHorizontal.value, cam.BinningVertical.value) == (1, 1)
    assert (cam.OffsetX.value, cam.OffsetY.value) == (0, 0)
    assert (cam.Width.value, cam.Height.value) == (1440, 1080)       # the node maxima
    assert not cam.ReverseX.value and not cam.ReverseY.value
    assert cam.GammaEnable.value is False and cam.AcquisitionFrameRateEnable.value is False
    assert cam.TriggerMode.value == "Off"
    assert cam.TLStream.StreamBufferHandlingMode.value == "NewestOnly"
    assert cam.calls == ["EndAcquisition", "BeginAcquisition"]      # reset while stopped
    rc.close()
    assert drv.closed == 1


def test_real_camera_reset_is_best_effort_and_recorded(fake_spin, caplog):
    fake_spin(Cam(missing=("GammaEnable",), locked=("Width", "ReverseY")))
    with caplog.at_level(logging.WARNING, logger=camera_mod.__name__):
        rc = camera_mod.RealCamera(CameraCfg(exposure_us=16667))
    assert set(rc.reset_state["failed"]) == {"GammaEnable", "Width", "ReverseY"}
    assert "not writable" in rc.reset_state["failed"]["Width"]
    assert "AttributeError" in rc.reset_state["failed"]["GammaEnable"]
    assert "Height" in rc.reset_state["reset"] and "TriggerMode" in rc.reset_state["reset"]
    assert any("could not reset" in r.message and "GammaEnable" in r.message
               for r in caplog.records)


@pytest.mark.parametrize("stage", ["EndAcquisition", "stream", "BeginAcquisition", "interrupt"])
def test_real_camera_failure_after_open_closes_the_driver(fake_spin, stage):
    cam = Cam()
    if stage == "EndAcquisition":
        cam.EndAcquisition = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    elif stage == "stream":
        cam.TLStream.StreamBufferHandlingMode.fail = "boom"
    elif stage == "BeginAcquisition":
        cam.BeginAcquisition = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    else:
        cam.TLStream.StreamBufferHandlingMode.SetValue = \
            lambda v: (_ for _ in ()).throw(KeyboardInterrupt())
    fake_spin(cam)
    with pytest.raises(KeyboardInterrupt if stage == "interrupt" else RuntimeError):
        camera_mod.RealCamera(CameraCfg())
    drv = Driver.instances[0]
    assert drv.closed == 1                              # released, not left held
    assert "DeInit" in cam.calls                        # de-inited even though not acquiring


def test_real_camera_retries_incomplete_frames(fake_spin):
    cam = fake_spin(Cam())
    rc = camera_mod.RealCamera(CameraCfg(exposure_us=16667))
    bad = [Image(incomplete=True) for _ in range(3)]
    cam.images = bad + [Image(value=42)]
    frame = rc.grab()                                  # 3 retries, then a complete frame
    assert frame.dtype == np.uint8 and int(frame[0, 0]) == 42
    assert all(i.released for i in bad) and rc.incomplete_frames == 3
    cam.images = [Image(incomplete=True) for _ in range(2)]
    rc.flush()                                         # flush retries the same way
    assert rc.incomplete_frames == 5
    cam.images = [Image(incomplete=True) for _ in range(4)] + [Image()]
    with pytest.raises(RuntimeError, match="4 incomplete frames in a row.*status 3"):
        rc.grab()                                      # 1 + 3 retries, then it gives up
    assert len(cam.images) == 1                        # the 5th was never grabbed
    grabs = [c for c in cam.calls if isinstance(c, tuple)]
    assert all(c == ("GetNextImage", camera_mod.GRAB_TIMEOUT_MS) for c in grabs)


def test_set_exposure_warns_when_not_frame_locked(fake_spin, caplog):
    cam = fake_spin(Cam())
    rc = camera_mod.RealCamera(CameraCfg(exposure_us=16667))
    with caplog.at_level(logging.WARNING, logger=camera_mod.__name__):
        rc.set_exposure_us(33333)
        assert not caplog.records
        rc.set_exposure_us(8000)
    assert cam.ExposureTime.value == 8000.0             # allowed, only warned
    assert len(caplog.records) == 1 and "not a whole number" in caplog.records[0].message
    assert "16667" in caplog.records[0].message
    with pytest.raises(ValueError):
        rc.set_exposure_us(0)


# ------------------------------------------------------------ fake cv2 window

class FakeCV2(types.ModuleType):
    WINDOW_NORMAL, WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN = 0, 1, 1
    INTER_NEAREST, COLOR_GRAY2BGR = 0, 8

    def __init__(self, rects):
        super().__init__("cv2")
        self.rects = list(rects)                       # successive getWindowImageRect answers
        self.calls = []

    def namedWindow(self, *a):
        self.calls.append("namedWindow")

    def moveWindow(self, name, x, y):
        self.calls.append(("moveWindow", x, y))

    def setWindowProperty(self, *a):
        self.calls.append("fullscreen")

    def waitKey(self, ms):
        self.calls.append(("waitKey", ms))
        return -1

    def getWindowImageRect(self, name):
        self.calls.append("rect")
        r = self.rects.pop(0) if len(self.rects) > 1 else self.rects[0]
        if isinstance(r, Exception):
            raise r
        return r

    def imshow(self, name, img):
        self.calls.append(("imshow", img.shape))

    def cvtColor(self, img, code):
        return np.repeat(img[..., None], 3, axis=2)

    def resize(self, img, size, interpolation=None):
        return np.zeros((size[1], size[0]) + img.shape[2:], img.dtype)

    def destroyWindow(self, name):
        self.calls.append("destroyWindow")


def _window(monkeypatch, rects):
    cv2 = FakeCV2(rects)
    monkeypatch.setitem(sys.modules, "cv2", cv2)
    return cv2, projector_mod.HDMIWindow((1920, 0), (912, 1140))


def test_hdmi_window_rect_ok(monkeypatch, caplog):
    with caplog.at_level(logging.WARNING, logger=projector_mod.__name__):
        cv2, w = _window(monkeypatch, [(1920, 0, 912, 1140)])
        w.show(np.zeros((1140, 912), np.uint8))
    assert w.rect == (1920, 0, 912, 1140) and w.rect_ok is True and w.rect_warning is None
    assert not caplog.records
    i = cv2.calls.index("fullscreen")
    assert cv2.calls[i + 1] == ("waitKey", 1) and cv2.calls[i + 2] == "rect"


def test_hdmi_window_wrong_rect_is_logged_never_raised(monkeypatch, caplog):
    """Settling (first read wrong, fine after the first paint) is not warned;
    a window still wrong after it is (wrong monitor, display scaling)."""
    with caplog.at_level(logging.WARNING, logger=projector_mod.__name__):
        _, w = _window(monkeypatch, [(0, 0, 640, 480), (1920, 0, 912, 1140)])
        assert w.rect_ok is False                       # read right after fullscreen
        w.show(np.zeros((1140, 912), np.uint8))
        assert w.rect_ok is True and not caplog.records  # settled after the first paint

        _, w = _window(monkeypatch, [(0, 0, 1920, 1080)])
        w.show(np.zeros((1140, 912), np.uint8))
        w.show(np.zeros((1140, 912), np.uint8))          # checked once, not per frame
    assert w.rect_ok is False and w.rect == (0, 0, 1920, 1080)
    assert "1920x1080 at (0, 0), expected 912x1140 at (1920, 0)" in w.rect_warning
    assert len(caplog.records) == 1 and "expected 912x1140" in caplog.records[0].message


def test_hdmi_window_rect_unreadable_is_not_fatal(monkeypatch):
    _, w = _window(monkeypatch, [AttributeError("no getWindowImageRect")])
    w.show(np.zeros((1140, 912), np.uint8))
    assert w.rect is None and w.rect_ok is None and "could not read" in w.rect_warning


def test_projector_file_still_has_no_sleep():
    import ast
    src = projector_mod.__file__
    tree = ast.parse(open(src, encoding="utf-8").read())
    assert not [n for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr == "sleep"]

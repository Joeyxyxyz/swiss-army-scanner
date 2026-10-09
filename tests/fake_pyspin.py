"""A fake PySpin module for tests: no SDK, no camera, no USB. It enforces the
Spinnaker rules the engine's real-camera code paths depend on, so a wrong
order fails here the way it would on the bench:

- every node needs an initialised camera (Init, not yet DeInit);
- TriggerSelector / Source / Activation / Overlap only while TriggerMode is Off;
- the stream buffer nodes only while not acquiring;
- EndAcquisition raises when the camera is not acquiring ("Camera is not
  started"); DeInit raises while acquiring;
- System.ReleaseInstance raises SPINNAKER_ERR_RESOURCE_IN_USE while any Python
  reference to a camera is alive (the camera objects are tracked by weakref;
  nothing in this module keeps one alive).

Install with monkeypatch.setitem(sys.modules, "PySpin", FakePySpin(...)) so every
`import PySpin` in the code under test gets it.

Triggered grabs: after BeginAcquisition with TriggerMode On, pulse k arrives
lead_s + k * period_s later (wall clock); `n_pulses` stops the train,
`incomplete` makes those pulses' images incomplete, `error_at` raises a
non-timeout SpinnakerException at that pulse. Line modes: Line0 is a read-only
input (opto), the others writable; `line_modes` sets their starting mode and
`stuck_lines` accept a LineMode write but keep their mode.
"""

import itertools
import time
import types
import weakref

import numpy as np

SPINNAKER_ERR_TIMEOUT = -1011
SPINNAKER_ERR_RESOURCE_IN_USE = -1004
SPINNAKER_ERR_ACCESS_DENIED = -1005
SPINNAKER_ERR_NOT_AVAILABLE = -1008
RW, RO = 4, 3

TRIGGER_CONFIG = ("TriggerSelector", "TriggerSource", "TriggerActivation", "TriggerOverlap")
STREAM_NODES = ("StreamBufferHandlingMode", "StreamBufferCountMode", "StreamBufferCountManual")


class SpinnakerException(Exception):
    def __init__(self, msg, errorcode=-1001):
        super().__init__(f"Spinnaker: {msg} [{errorcode}]")
        self.errorcode = errorcode


class _Device:
    """The camera's state and the event log (never holds a camera object)."""

    def __init__(self, fake):
        self.fake = fake
        self.init = False
        self.acquiring = False
        self.values = {"TriggerMode": fake.enum("TriggerMode_Off"), "LineSelector": None}
        self.line_mode = {ln: fake.enum(f"LineMode_{m}") for ln, m in fake.line_modes.items()}
        self.k = 0
        self.fid = 0
        self.t_begin = None
        self.log = fake.log

    def need_init(self, what):
        if not self.init:
            raise SpinnakerException(f"{what}: camera is not initialized",
                                     SPINNAKER_ERR_NOT_AVAILABLE)


class _Node:
    def __init__(self, dev, name):
        self._dev, self._name = dev, name

    def GetAccessMode(self):
        d = self._dev
        if self._name == "LineMode" and d.values.get("LineSelector") == d.fake.enum(
                "LineSelector_Line0"):
            return RO
        return RW

    def GetValue(self):
        d = self._dev
        d.need_init(f"{self._name}.GetValue")
        if self._name == "LineMode":
            line = d.fake.name_of(d.values.get("LineSelector")).split("_")[-1]
            return d.line_mode[line]
        if self._name == "PixelFormat":
            return d.values.get("PixelFormat", d.fake.enum("PixelFormat_Mono8"))
        return d.values.get(self._name, 0)

    def GetMax(self):
        self._dev.need_init(f"{self._name}.GetMax")
        return 1440 if self._name == "Width" else 1080

    def SetValue(self, v):
        d, f = self._dev, self._dev.fake
        d.need_init(f"{self._name}.SetValue")
        if self._name in TRIGGER_CONFIG and d.values["TriggerMode"] == f.enum("TriggerMode_On"):
            raise SpinnakerException(f"{self._name} is not writable while TriggerMode is On",
                                     SPINNAKER_ERR_ACCESS_DENIED)
        if self._name in STREAM_NODES and d.acquiring:
            raise SpinnakerException(f"{self._name} is not writable while acquiring",
                                     SPINNAKER_ERR_ACCESS_DENIED)
        if self._name == "TriggerSource" and f.fail_source:
            raise SpinnakerException("TriggerSource write failed", SPINNAKER_ERR_ACCESS_DENIED)
        if self._name == "LineMode":
            if self.GetAccessMode() != RW:
                raise SpinnakerException("LineMode is read-only for this line",
                                         SPINNAKER_ERR_ACCESS_DENIED)
            line = f.name_of(d.values.get("LineSelector")).split("_")[-1]
            if line not in f.stuck_lines:                # a stuck line ignores the write
                d.line_mode[line] = v
        d.values[self._name] = v
        d.log.append(("set", self._name, f.name_of(v)))


class _Stream:
    def __init__(self, dev):
        self._dev = dev

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return _Node(self._dev, name)


class _Image:
    def __init__(self, fid, ts_ns, incomplete):
        self._fid, self._ts, self._incomplete = fid, ts_ns, incomplete

    def IsIncomplete(self):
        return self._incomplete

    def GetImageStatus(self):
        return 3 if self._incomplete else 0

    def GetFrameID(self):
        return self._fid

    def GetTimeStamp(self):
        return self._ts

    def GetNDArray(self):
        return np.full((6, 8), 100, np.uint8)

    def Release(self):
        pass


class _Camera:
    """A CameraPtr: methods, and any node by attribute (QuickSpin style)."""

    def __init__(self, dev):
        self._dev = dev
        weakref.finalize(self, _Camera._gone, dev)

    @staticmethod
    def _gone(dev):
        dev.log.append(("destroyed", "initialized" if dev.init else "deinit"))

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        if name == "TLStream":
            return _Stream(self._dev)
        return _Node(self._dev, name)

    def Init(self):
        self._dev.init = True
        self._dev.log.append(("Init",))

    def DeInit(self):
        d = self._dev
        d.need_init("DeInit")
        if d.acquiring:
            raise SpinnakerException("DeInit while acquiring", SPINNAKER_ERR_RESOURCE_IN_USE)
        d.init = False
        d.log.append(("DeInit",))

    def BeginAcquisition(self):
        d = self._dev
        d.need_init("BeginAcquisition")
        if d.acquiring:
            raise SpinnakerException("Camera is already streaming", SPINNAKER_ERR_RESOURCE_IN_USE)
        d.acquiring, d.t_begin, d.k = True, time.perf_counter(), 0
        d.log.append(("BeginAcquisition",))

    def EndAcquisition(self):
        d = self._dev
        if not d.init or not d.acquiring:
            raise SpinnakerException("Camera is not started", SPINNAKER_ERR_NOT_AVAILABLE)
        d.acquiring = False
        d.log.append(("EndAcquisition",))

    def GetNextImage(self, timeout_ms):
        d, f = self._dev, self._dev.fake
        d.need_init("GetNextImage")
        if not d.acquiring:
            raise SpinnakerException("Camera is not started", SPINNAKER_ERR_NOT_AVAILABLE)
        triggered = d.values["TriggerMode"] == f.enum("TriggerMode_On")
        if not triggered:
            return _Image(d.fid, 0, False)
        k = d.k
        if f.error_at is not None and k >= f.error_at:
            raise SpinnakerException("Stream has been aborted", -1012)
        due = d.t_begin + f.lead_s + k * f.period_s
        now = time.perf_counter()
        if (f.n_pulses is not None and k >= f.n_pulses) or due > now + timeout_ms / 1000.0:
            time.sleep(timeout_ms / 1000.0)
            raise SpinnakerException("Failed waiting for EventData on NEW_BUFFER_DATA event",
                                     SPINNAKER_ERR_TIMEOUT)
        if due > now:
            time.sleep(due - now)
        d.k += 1
        fid = d.fid
        d.fid += 1
        return _Image(fid, int(round((f.lead_s + k * f.period_s) * 1e9)), k in f.incomplete)


class _CameraList:
    def __init__(self, cams):
        self._cams = list(cams)

    def GetSize(self):
        return len(self._cams)

    def __getitem__(self, i):
        return self._cams[i]

    def __iter__(self):
        return iter(self._cams)

    def Clear(self):
        self._cams = []


class _System:
    def __init__(self, fake):
        self._fake = fake

    def GetCameras(self):
        f = self._fake
        cam = _Camera(f.device)
        f.handed_out.append(weakref.ref(cam))
        return _CameraList([cam])

    def ReleaseInstance(self):
        f = self._fake
        alive = sum(1 for r in f.handed_out if r() is not None)
        if alive:
            f.log.append(("ReleaseInstance", "refused", alive))
            raise SpinnakerException(f"Can't clear a camera because something still holds a "
                                     f"reference to the camera ({alive})",
                                     SPINNAKER_ERR_RESOURCE_IN_USE)
        f.released = True
        f.log.append(("ReleaseInstance", "ok"))


class FakePySpin(types.ModuleType):
    """The module object. Enum constants (TriggerMode_On, LineMode_Input, ...)
    are distinct ints created on first use; name_of() maps one back."""

    def __init__(self, *, period_s=0.005, lead_s=0.005, n_pulses=None, incomplete=(),
                 error_at=None, line_modes=None, fail_source=False, stuck_lines=()):
        super().__init__("PySpin")
        self._enums, self._names = {}, {}
        self._ids = itertools.count(100)
        self.period_s, self.lead_s, self.n_pulses = float(period_s), float(lead_s), n_pulses
        self.incomplete, self.error_at, self.fail_source = set(incomplete), error_at, fail_source
        self.stuck_lines = set(stuck_lines)
        self.line_modes = {"Line0": "Input", "Line1": "Output", "Line2": "Input",
                           "Line3": "Input", **(line_modes or {})}
        self.log = []
        self.handed_out = []
        self.released = False
        self.RW, self.RO = RW, RO
        self.SPINNAKER_ERR_TIMEOUT = SPINNAKER_ERR_TIMEOUT
        self.SPINNAKER_ERR_RESOURCE_IN_USE = SPINNAKER_ERR_RESOURCE_IN_USE
        self.SpinnakerException = SpinnakerException
        self.device = _Device(self)
        system = _System(self)
        self.System = types.SimpleNamespace(GetInstance=lambda: system)

    def enum(self, name):
        if name not in self._enums:
            v = next(self._ids)
            self._enums[name], self._names[v] = v, name
        return self._enums[name]

    def name_of(self, v):
        return self._names.get(v, v)

    def __getattr__(self, name):
        if name[:1].isupper() and "_" in name:
            return self.enum(name)
        raise AttributeError(name)

    # ---- what the tests read
    def events(self, kind=None):
        return [e for e in self.log if kind is None or e[0] == kind]

    def last_value(self, node):
        vals = [e[2] for e in self.log if e[0] == "set" and e[1] == node]
        return vals[-1] if vals else None

    def index(self, event):
        """Position of the last occurrence of `event` (a tuple prefix) in the log."""
        hits = [i for i, e in enumerate(self.log) if e[:len(event)] == event]
        return hits[-1] if hits else None

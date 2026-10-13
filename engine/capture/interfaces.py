"""
Hardware adapter interfaces: what the engine needs from a projector, a camera and an I/O
device, so the scanner ports to another line (docs/PORTING.md). Documentation in code:
typing Protocols the existing drivers already satisfy; nothing here opens hardware.

    DisplayWindow   the projector as a display: show a DMD image, keep it painted (pumped
                    hold, never a bare sleep), close. Today: HDMIWindow (an extended display:
                    any DLP / LCoS / LCD projector or DMD board driven as a monitor) and
                    MockWindow. Pattern / trigger modes (an on-board pattern sequence fired
                    by a trigger line) are optional and live outside this path
                    (engine/capture/patternmode.py, flash packs; a human enters them).
                    Optional, for the zone painter only (engine/zones/: the display is a
                    second monitor, so the mouse cursor is projected onto the part):
                    enable_mouse() and poll(wait_ms) -> the mouse / key events since the
                    last poll in pattern px (a pumped wait like hold(); a key the window's
                    show() pump received is kept for the next poll, never dropped). A
                    window without them still scans; the painter refuses it.
    CameraDriver    set the exposure, drop stale frames, grab one frame (H, W) mono, close.
                    Today: RealCamera (the FLIR Spinnaker SDK) and MockCamera. A GenICam
                    camera (GigE Vision, USB3 Vision: exposure = ExposureTime, NewestOnly
                    buffering, AcquisitionMode Continuous) fits the same four methods; a
                    hardware trigger adds TriggerMode / TriggerSource / LineSelector nodes
                    (engine/reconstruct/trigger_scan.py).
    IO backends     engine/control/commands.py: InputBackend.start(bus) delivers commands
                    (START / ABORT / HOME / TRIGGER / STEP / ACK / ARM, the station
                    configuration and profile) and the line's sensor claims
                    (bus.set_inputs), OutputBackend.publish(signals) drives READY / BUSY /
                    PASS / FAIL / ALARM / PULL / KICKOUT / PROMPT / SENSOR_MISMATCH and one
                    PERMIT_<NAME> per permission, .publish_registers(registers) the
                    result_code / state_code / station_config / profile / work_state /
                    work_state_code numbers. Placeholders: camera GPIO, serial, 24 V digital
                    I/O, Modbus TCP, OPC UA, EtherNet/IP.

The rig (engine/session.py ScanSession.rig) owns them: camera first, then the projector,
the machine-wide rig lock around both, every part released on the way out.
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class DisplayWindow(Protocol):
    """The projector seen as a display (HDMI / DisplayPort extended desktop)."""
    size: tuple                          # (W, H) of the display mode, px
    rect_ok: Optional[bool]              # the window covers exactly that display (None: unknown)

    def show(self, pattern: np.ndarray) -> None:
        """Paint one image (H, W) grey or (H, W, 3) BGR uint8, native DMD orientation."""

    def hold(self, seconds: float) -> None:
        """Keep it painted for `seconds`, pumping the window's event loop (never a sleep)."""

    def close(self) -> None:
        """Remove the window (the display shows the desktop / black again)."""


@runtime_checkable
class CameraDriver(Protocol):
    """A mono machine-vision camera in free run."""

    def set_exposure_us(self, us) -> None:
        """Exposure in microseconds (whole video frames over HDMI: 16 667 us at 60 Hz)."""

    def flush(self) -> None:
        """Drop buffered frames: the next grab shows what is projected now."""

    def grab(self) -> np.ndarray:
        """One frame (H, W), camera DN."""

    def close(self) -> None:
        """Release the device (the next program can open it)."""


@runtime_checkable
class InputDriver(Protocol):
    """A command source (engine/control/commands.py InputBackend)."""
    name: str
    mapping: dict

    def start(self, bus) -> None:
        """Begin delivering commands: bus.command(name, source=...)."""

    def stop(self) -> None:
        """Release the hardware."""


@runtime_checkable
class OutputDriver(Protocol):
    """Signals + registers out (engine/control/commands.py OutputBackend)."""
    name: str
    mapping: dict

    def publish(self, signals: dict) -> None:
        """READY / BUSY / PASS / FAIL / ALARM booleans, on every bus change."""

    def publish_registers(self, registers: dict) -> None:
        """result_code / state_code / station_config / profile / work_state /
        work_state_code integers, on every bus change."""

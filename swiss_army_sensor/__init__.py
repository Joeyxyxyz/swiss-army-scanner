"""
Swiss Army Sensor
=================

One DMD + one camera = many virtual sensors.

The DMD is not a display. It is an optical computational probe that
physically instantiates the measurement basis projected onto the scene.
By choosing what to project and how to decode what comes back, a single
DLP4500 / DLP3010 / DLP472TP module replaces a rack of dedicated sensors.

v0.4.0 modules:
  core             unified SensorRig.read() dispatcher
  patterns         pattern generators (flood, point, phase-shift, Hadamard, ...)
  decoders         capture-frame to sensor-reading inverters
  spectral         laser-spot calibration, material library, NIR spectroscopy
  hardware         DMD profiles, LightSource coherence model, RigConfig
  holography       Lee binary CGH, Fresnel propagation, 4f reconstruction
  barcode          flood + raster + Hadamard + line-scan barcode decoding
  nullification    MNF operators (DifferentialPair, PhaseShiftNu, HadamardNu)
  drivers          DLPC350 USB-HID + FLIR Spinnaker adapters + mocks
"""

from .core import SensorRig, SensorMode, BarcodeMethod
from .drivers.dlp_driver import DLPDriver, MockDLPDriver
from .drivers.camera_driver import CameraDriver, MockCameraDriver
from . import spectral, hardware, holography, barcode, nullification, golden, aim, patterns, mesh

__version__ = "0.5.2"
__all__ = [
    "SensorRig", "SensorMode", "BarcodeMethod",
    "DLPDriver", "MockDLPDriver", "CameraDriver", "MockCameraDriver",
    "spectral", "hardware", "holography", "barcode", "nullification", "golden", "aim", "patterns", "mesh",
]


# Swiss Army Sensor

**One DMD + one camera = many virtual sensors.**

A Python utility that turns a Texas Instruments DLP / DMD module (DLP4500 LightCrafter,
DLP3010, DLP472TP, etc.) plus an industrial camera into a software-defined sensor head
that can replace a barcode reader, a photoelectric switch, a laser
displacement sensor, a 3D profilometer, an edge detector, a color sensor, a
vibration probe, a surface-defect inspector, an OCR/mark verifier, a
compressed-sensing single-pixel imager, a laser-spot sensor, a
thin-film coating gauge, a material-ID spectroscopy probe, a speckle roughness
meter, an NIR spectrometer, and a CGH projector — without changing any hardware.

> The DMD is not a display. It is an optical computational probe that physically
> instantiates the measurement basis projected onto the scene. The choice of
> patterns *is* the choice of sensor.

---

This package lives inside the `swiss-army-scanner` repo and is installed with
it (`pip install -e .` from the repo root). The scanner engine loads a few of its
numpy-only modules file-by-file (`engine/_seed.py`); see the repo README.

## Quick start

```bash
pip install -e .

swiss-army-sensor --mode profilometer_3d \
                  --params '{"n_steps":4,"period_px":24}' \
                  --scene sphere
```

```python
from swiss_army_sensor import SensorRig, SensorMode, hardware

# Default rig: DLP4500 + stock LEDs + FLIR Blackfly S
with SensorRig.create(mock=True, scene="qr") as rig:
    rig.set_mode(SensorMode.PROFILOMETRY_3D, n_steps=8, period_px=18)
    phase = rig.read().phase

    rig.set_mode(SensorMode.SINGLE_PIXEL, order=6)
    print(rig.read().image.shape)              # (64, 64)

# Holography rig: pinhole-conditioned LEDs + HeNe
with SensorRig.create(mock=True, config=hardware.HOLOGRAPHY_RIG) as rig:
    rig.set_mode(SensorMode.HOLOGRAPHY, propagation_distance_mm=200)
    r = rig.read()
    print(r.coherence_quality)                 # 1.0 with conditioned sources
```

Switch to real hardware:
```python
rig = SensorRig.create(mock=False)
```

---

## Modes (v0.4.0)

| Mode                        | Basis                  | Emulates the function of                     |
|----------------------------|------------------------|----------------------------------------------|
| `barcode_reader`           | flood/raster/Hadamard/line | Industrial barcode reader (4 methods inc. DMD-native)|
| `photoelectric_switch`     | single bright spot     | Photoelectric proximity switch               |
| `distance_sensor`          | point + triangulation  | Laser displacement sensor                    |
| `profilometer_3d`          | N-step phase-shift     | Structured-light 3D scanner                  |
| `edge_detector`            | Sobel pair             | Edge-detection vision sensors                |
| `color_sensor`             | sequential R/G/B       | Industrial color sensor                      |
| `single_pixel_imager`      | Hadamard / Walsh       | Compressed-sensing camera                    |
| `vibration_sensor`         | flicker carrier        | Non-contact accelerometer                    |
| `defect_inspector`         | high-freq stripes      | Specular / diffuse defect cam (Nayar)       |
| `mark_verifier`            | flood + OCR            | DPM / mark verifier                          |
| `spot_emulation`           | spot at calibrated mm  | Laser-spot sensor (0.75 / 6.5 mm spot)       |
| `spectral_3band`           | RGB ratio              | 3-band visible colorimetry                   |
| `thin_film_thickness`      | Newton's rings (3-band)| Transparent coating gauge 50nm–3µm           |
| `material_identification`  | per-LED signature      | Material library cosine-similarity match     |
| `speckle_roughness`        | coherent illumination  | Surface Ra meter (Goodman model)             |
| `nir_spectroscopy`         | NIR band scan          | TI NIRscan EVM analog                        |
| `holography`               | Lee binary CGH         | Holographic projector                        |
| `interferometry`           | phase-shift fringes    | Phase-shifting interferometer                |

---

## Hardware

Target stack (math verified against ground truth; live-silicon validation pending — see Verification status):

* **Projector**: DLP4500 LightCrafter (DLPC350) — VID `0x0451`, PID `0x6401`.
  USB-HID wire format cross-checked byte-for-byte against **pycrafter4500**
  (the established, hardware-proven LightCrafter library) in
  `tests/test_driver_packets.py`. Command codes traced to TI DLPU010G.
* **Pattern delivery**: the DLPC350 has **no USB frame push** (DLPU010G
  §2.4.3) — patterns stream through the EVM's HDMI/RGB port. The
  `HDMITransport` renders patterns fullscreen on the display output wired
  to the EVM; **Pattern Trigger Mode 0** lets VSYNC clock the sequence
  while **TRIG_OUT_1 frames every exposure** for the camera. No flash
  programming, no firmware builds — any host with a spare display output.
* **Camera**: FLIR Blackfly S BFS-U3-16S2M-CS (Sony IMX273, mono) via
  PySpin, hardware-triggered on Line0 from TRIG_OUT_1.
* **Host**: Windows 10 desktop (Cypress driver). Windows 11 has known
  signing issues.

Wiring:
```
DLP4500 J14 TRIG_OUT_1 ── 1 kΩ ── FLIR Hirose pin 4 (OPTO_IN, Line0)
                     └─────────── FLIR Hirose pin 1 (GND)
```
**J13 jumper on 3.3 V** (not 1.8 V). Start LED current ~25%.

**Bench rule (verified on silicon):** the HDMI stream and the USB
command channel are mutually exclusive while streaming. USB display-mode
commands force source re-acquisition; the non-standard 912x1140 timing
fails to re-lock and the DMD reverts to its internal test pattern.
`SensorRig.create(mock=False)` is therefore **HDMI-only by default** —
it never opens USB. Pass `usb=True` only for the trigger-synced path,
and configure over USB *before* establishing the HDMI lock, never after.

## Verification status (v0.5)

| Layer | Status |
|---|---|
| Sensing math (phase shift, gray code, Hadamard) | ✅ verified against analytic ground truth (`test_reference_physics.py`) |
| All 18 modes + 4 barcode methods | ✅ end-to-end against simulator (`test_modes.py`) |
| DLPC350 wire format | ✅ byte-identical to pycrafter4500 (`test_driver_packets.py`) |
| Packaging | ✅ wheel + sdist build, console script, numpy 1.x/2.x CI |
| Live DLPC350 / FLIR optics | ⏳ pending first light |

v0.5 fixed five bugs that simulator-only testing could not see: CMD2/CMD3
wire-order swap, reply parsed from the wrong byte, wrong LED-current
command code, mutually inconsistent gray-code bit-planes, and a DC-term
blowout in the single-pixel decoder. Three were caught by cross-checking
against established references, two by ground-truth physics tests.

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                      SensorRig (core.py)                       │
│  - set_mode() / read() / stream()                              │
│  - RigConfig: DMD + sources + camera, with can_run() checks    │
└──────────────────┬───────────────────────┬─────────────────────┘
                   │                       │
        ┌──────────▼──────────┐  ┌─────────▼───────────┐
        │  patterns.py        │  │  decoders.py         │
        │  flood / point /    │  │  barcode / phase /   │
        │  phase / Hadamard / │  │  Hadamard / FFT /    │
        │  Gray / Sobel / ... │  │  Nayar / ...         │
        └─────────────────────┘  └──────────────────────┘
        ┌──────────────────────────────────────────────┐
        │  spectral.py | holography.py | barcode.py    │
        │  hardware.py | nullification.py              │
        └──────────────────────────────────────────────┘
                   │                       │
        ┌──────────▼──────────┐  ┌─────────▼───────────┐
        │  DLPDriver          │  │  CameraDriver       │
        │  (USB HID,          │  │  (PySpin / FLIR)    │
        │   DLPC350 cmds)     │  │                     │
        ├─────────────────────┤  ├─────────────────────┤
        │  MockDLPDriver      │  │  MockCameraDriver   │
        │  (numpy sim)        │  │  (forward-model)    │
        └─────────────────────┘  └─────────────────────┘
```

Patterns are physics, decoders are math, drivers are I/O. Every component is
independently testable. Real and mock drivers are drop-in interchangeable.

---

## MNF Nullification (theoretical module)

`nullification.py` is the operator algebra for the Messina Nullification Framework
applied to the sensor rig. Three operators implemented:

- `DifferentialPair` — common-mode rejection with ν detuning
- `PhaseShiftNu` — Zhang N-step arctan2 with detuned basis angles
- `HadamardNu` — ±1 pair differential with ν weighting

All three verify against their ν=1 standard-algorithm limits to machine precision.
The `sweep_nu` driver finds the optimal ν* on synthetic test cases. The strongest
empirical signal in v0.4 is the Hadamard ν* ≈ 0.94 result, statistically stable
across 30 seeds × 11 SNR levels, +0.3–0.8% MSE improvement over ν=1. Bench
verification on real DMD bucket measurements is the next step.

The module deliberately stays at the operator level — it does not wire into any
sensor mode yet. Hooks are documented for later integration after bench data.

---

## License

Apache License 2.0. See `LICENSE` and `NOTICE` at the repo root.

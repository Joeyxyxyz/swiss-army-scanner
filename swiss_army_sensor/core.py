"""
SensorRig
=========

The orchestrator. One uniform `read()` interface for every virtual sensor.

Example:
    >>> from swiss_army_sensor import SensorRig, SensorMode
    >>> rig = SensorRig.create(mock=True, scene="qr")
    >>> rig.set_mode(SensorMode.PROFILOMETRY_3D, n_steps=4, period_px=24)
    >>> r = rig.read()                         # ProfilometryResult(...)
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from . import patterns as P
from . import decoders as D
from .drivers.dlp_driver import DLPDriver, MockDLPDriver, WriteRefused
from .drivers.camera_driver import CameraDriver, MockCameraDriver
from ._utils import synthetic_scene

USB_REFUSED = (
    "SensorRig.create(mock=False, usb=True) is refused: it used to switch the "
    "projector to pattern display over USB without being asked. The projector "
    "display is switched by a person, never by this package. Nothing was "
    "opened and nothing was sent. Use usb=False (the HDMI-only path).")


class SensorMode(str, Enum):
    BARCODE        = "barcode_reader"
    PHOTOELECTRIC  = "photoelectric_switch"
    DISTANCE       = "distance_sensor"
    PROFILOMETRY_3D = "profilometer_3d"
    EDGE           = "edge_detector"
    COLOR          = "color_sensor"
    SINGLE_PIXEL   = "single_pixel_imager"
    VIBRATION      = "vibration_sensor"
    DEFECT         = "defect_inspector"
    MARK           = "mark_verifier"
    SPOT_EMULATION    = "spot_emulation"
    SPECTRAL_3BAND    = "spectral_3band"
    THIN_FILM         = "thin_film_thickness"
    MATERIAL_ID       = "material_identification"
    SPECKLE_ROUGHNESS = "speckle_roughness"
    NIR_SPECTROSCOPY  = "nir_spectroscopy"
    HOLOGRAPHY        = "holography"
    INTERFEROMETRY    = "interferometry"


class BarcodeMethod(str, Enum):
    FLOOD_PASSIVE         = "flood_passive"
    RASTER_SINGLE_PIXEL   = "raster_single_pixel"
    HADAMARD_SINGLE_PIXEL = "hadamard_single_pixel"
    LINE_SCAN             = "line_scan"


class _HDMIOnlyDLP:
    """
    DLP stand-in for the HDMI-only real path: satisfies the driver
    interface SensorRig uses, delivers every pattern through the
    HDMITransport, and cannot touch USB by construction. LED color
    selection is unavailable on this path (no USB), so color modes
    degrade to whatever the EVM's current LED state shows — a person
    selects the LEDs in the vendor program if a mode needs another color.
    """

    def __init__(self, transport):
        self.transport = transport
        self.current_pattern = None

    def project_static(self, image):
        self.current_pattern = image
        self.transport.show(image)

    def set_leds(self, *a, **k):
        import logging
        logging.getLogger(__name__).warning(
            "HDMI-only path: this package does not select LEDs on a real "
            "projector (a person does, in the vendor program). Ignoring "
            "set_leds%s.", a)

    def start_sequence(self):
        pass

    def stop_sequence(self):
        pass

    def close(self):
        self.transport.close()


@dataclass
class SensorRig:
    dlp: DLPDriver
    camera: CameraDriver
    mode: SensorMode = SensorMode.BARCODE
    params: dict = None
    config: "RigConfig" = None

    def __post_init__(self):
        if self.params is None:
            self.params = {}
        if self.config is None:
            from .hardware import DEFAULT_RIG
            self.config = DEFAULT_RIG

    @classmethod
    def create(cls, mock: bool = True, scene: str = "plane", config=None,
               monitor_offset=(1920, 0), usb: bool = False) -> "SensorRig":
        """
        mock=True  -> numpy simulator end to end.
        mock=False -> real hardware, the HDMI-ONLY path: patterns render
        fullscreen to the EVM's HDMI input and the DLPC350 is NEVER
        opened over USB.

        BENCH RULE (verified on real silicon, 2026-09): the HDMI stream
        and the USB command channel are mutually exclusive while
        streaming. USB display-mode/source commands force the DLPC350 to
        re-run source acquisition; the non-standard 912x1140 timing then
        fails to re-lock and the DMD falls back to its internal test
        pattern (blue lines = "no valid source"). Whoever is streaming
        owns the device.

        usb=True on real hardware is REFUSED, always (WriteRefused,
        before the window, the camera or the USB device is opened): it
        used to send Display Mode Select unasked. The projector display
        is switched by a person, never by this package.
        """
        if mock:
            dlp = MockDLPDriver().open()
            scene_img = synthetic_scene((1080, 1440), kind=scene)
            cam = MockCameraDriver(scene=scene_img, dlp=dlp).open()
        else:
            if usb:
                raise WriteRefused(USB_REFUSED)
            from .drivers.pattern_transport import HDMITransport
            transport = HDMITransport(monitor_offset=monitor_offset)
            dlp = _HDMIOnlyDLP(transport)
            cam = CameraDriver().open()
        return cls(dlp=dlp, camera=cam, config=config)

    def close(self):
        try: self.camera.close()
        finally: self.dlp.close()

    def set_mode(self, mode: SensorMode, **params):
        self.mode = SensorMode(mode)
        self.params = params

    def read(self) -> Any:
        m = self.mode

        # ============ Barcode ============
        if m == SensorMode.BARCODE:
            from . import barcode as B
            method = self.params.get("method", "hadamard_single_pixel")
            if method == "flood_passive":
                self.dlp.project_static(P.barcode_flood())
                return B.decode_flood_passive(self.camera.capture())

            def bucket(pattern):
                self.dlp.project_static(pattern)
                return float(self.camera.capture().mean())

            if method == "raster_single_pixel":
                return B.decode_raster_single_pixel(
                    bucket, pattern_size=(P.DMD_H, P.DMD_W),
                    bucket_size=self.params.get("bucket_size", (64, 64)),
                    spot_radius_px=self.params.get("spot_radius_px", 4))
            if method == "hadamard_single_pixel":
                return B.decode_hadamard_single_pixel(
                    bucket, pattern_size=(P.DMD_H, P.DMD_W),
                    order=self.params.get("order", 5),
                    fov_px=self.params.get("fov_px", 256))
            if method == "line_scan":
                return B.decode_line_scan(
                    bucket, pattern_size=(P.DMD_H, P.DMD_W),
                    line_width_px=self.params.get("line_width_px", 4))
            raise ValueError(f"Unknown barcode method: {method}")

        # ============ Geometric ============
        if m == SensorMode.PHOTOELECTRIC:
            cx = self.params.get("cx", P.DMD_W // 2)
            cy = self.params.get("cy", P.DMD_H // 2)
            r  = self.params.get("radius", 30)
            self.dlp.project_static(P.single_point(cx, cy, r))
            frame = self.camera.capture()
            return D.decode_photoelectric(
                frame, threshold=self.params.get("threshold", 80),
                roi=self.params.get("roi"))

        if m == SensorMode.DISTANCE:
            self.dlp.project_static(P.single_point(
                self.params.get("cx", P.DMD_W // 2),
                self.params.get("cy", P.DMD_H // 2),
                self.params.get("radius", 6)))
            return D.decode_distance(self.camera.capture(),
                baseline_mm=self.params.get("baseline_mm", 50.0),
                focal_px=self.params.get("focal_px", 1800.0),
                reference_x=self.params.get("reference_x", 720.0))

        if m == SensorMode.PROFILOMETRY_3D:
            n = self.params.get("n_steps", 4)
            T = self.params.get("period_px", 24)
            frames = []
            for pat in P.phase_shift_fringes(n_steps=n, period_px=T):
                self.dlp.project_static(pat)
                frames.append(self.camera.capture())
            return D.decode_phase_shift(frames)

        if m == SensorMode.EDGE:
            pats = P.sobel_pair()
            frames = []
            for pat in pats:
                self.dlp.project_static(pat); frames.append(self.camera.capture())
            return D.decode_edges(*frames)

        if m == SensorMode.COLOR:
            roi = self.params.get("roi")
            self.dlp.set_leds(True, False, False); self.dlp.project_static(P.flood())
            r = self.camera.capture()
            self.dlp.set_leds(False, True, False); self.dlp.project_static(P.flood())
            g = self.camera.capture()
            self.dlp.set_leds(False, False, True); self.dlp.project_static(P.flood())
            b = self.camera.capture()
            return D.decode_color(r, g, b, roi=roi)

        if m == SensorMode.SINGLE_PIXEL:
            order = self.params.get("order", 5)
            fov   = self.params.get("fov_px", 256)
            pats = P.hadamard_patterns(order=order, fov_px=fov)
            intensities = []
            for pat in pats:
                self.dlp.project_static(pat)
                frame = self.camera.capture()
                intensities.append(float(frame.mean()))
            return D.decode_single_pixel(intensities, n_per_side=2 ** order)

        if m == SensorMode.VIBRATION:
            freq = self.params.get("modulation_hz", 50.0)
            frames = self.params.get("n_frames", 256)
            fps  = self.params.get("fps", 1000.0)
            pats = P.flicker_carrier(freq, frames, fps)
            intensities = []
            for pat in pats:
                self.dlp.project_static(pat)
                intensities.append(float(self.camera.capture().mean()))
            return D.decode_vibration(intensities, fps)

        if m == SensorMode.DEFECT:
            T = self.params.get("period_px", 6)
            pats = P.high_freq_stripes(T)
            frames = []
            for pat in pats:
                self.dlp.project_static(pat); frames.append(self.camera.capture())
            return D.decode_defect(*frames)

        if m == SensorMode.MARK:
            self.dlp.project_static(P.flood())
            return D.decode_mark(self.camera.capture(),
                                 expected=self.params.get("expected"))

        # ============ Spectral / material ============
        if m == SensorMode.SPOT_EMULATION:
            from . import spectral as S
            spot_mm = self.params.get("spot_diameter_mm", 0.85)
            z_mm = self.params.get("working_distance_mm", 300.0)
            r_px = S.spot_radius_to_pixels(spot_mm / 2, z_mm)
            self.dlp.project_static(P.single_point(P.DMD_W // 2, P.DMD_H // 2, r_px))
            frame = self.camera.capture()
            yy, xx = np.ogrid[:frame.shape[0], :frame.shape[1]]
            cy, cx = frame.shape[0] // 2, frame.shape[1] // 2
            cam_r = max(2, r_px * frame.shape[1] // P.DMD_W)
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= cam_r ** 2
            intensity = float(frame[mask].mean()) / 255.0 if mask.any() else 0.0
            return S.emulate_spot(spot_mm, z_mm, intensity)

        if m == SensorMode.SPECTRAL_3BAND:
            from . import spectral as S
            roi = self.params.get("roi")
            self.dlp.set_leds(True, False, False); self.dlp.project_static(P.flood())
            fr = self.camera.capture()
            self.dlp.set_leds(False, True, False); self.dlp.project_static(P.flood())
            fg = self.camera.capture()
            self.dlp.set_leds(False, False, True); self.dlp.project_static(P.flood())
            fb = self.camera.capture()
            def m_(f):
                if roi: y0,x0,y1,x1 = roi; f = f[y0:y1, x0:x1]
                return float(f.mean()) / 255.0
            return S.three_band_spectral(m_(fr), m_(fg), m_(fb))

        if m == SensorMode.THIN_FILM:
            from . import spectral as S
            n_film = self.params.get("n_film", 1.5)
            self.dlp.set_leds(True, False, False); self.dlp.project_static(P.flood())
            r = float(self.camera.capture().mean()) / 255.0
            self.dlp.set_leds(False, True, False); self.dlp.project_static(P.flood())
            g = float(self.camera.capture().mean()) / 255.0
            self.dlp.set_leds(False, False, True); self.dlp.project_static(P.flood())
            b = float(self.camera.capture().mean()) / 255.0
            return S.estimate_thin_film_thickness(r, g, b, n_film=n_film)

        if m == SensorMode.MATERIAL_ID:
            from . import spectral as S
            ints = {}
            for name in self.params.get("leds", ["red", "green", "blue"]):
                self.dlp.set_leds(name == "red", name == "green", name == "blue")
                self.dlp.project_static(P.flood())
                ints[name] = float(self.camera.capture().mean()) / 255.0
            return S.identify_material(ints, use_nir=self.params.get("use_nir", False))

        if m == SensorMode.SPECKLE_ROUGHNESS:
            from . import spectral as S
            lam = self.params.get("wavelength_nm", 638.0)
            self.dlp.project_static(P.flood())
            frame = self.camera.capture()
            return S.estimate_speckle_roughness(frame, wavelength_nm=lam)

        if m == SensorMode.NIR_SPECTROSCOPY:
            from . import spectral as S
            ok, why = self.config.can_run("nir_spectroscopy") if self.config else (True, "")
            bands = self.params.get("bands", {})
            if not bands:
                self.dlp.project_static(P.flood())
                base = float(self.camera.capture().mean()) / 255.0
                bands = {"nir_940": base, "swir_1310": base * 0.8,
                         "swir_1550": base * 0.6}
            result = S.nir_spectroscopy(bands)
            if not ok:
                result.detected_components.append(f"⚠ {why}")
            return result

        # ============ Coherent / holographic ============
        if m == SensorMode.HOLOGRAPHY:
            from . import holography as Holo
            ok, why = self.config.can_run("holography") if self.config else (True, "")
            target = self.params.get("target_image")
            if target is None:
                target = self._default_holo_target()
            z_mm = self.params.get("propagation_distance_mm", 200.0)
            wavelength_nm = self.params.get("wavelength_nm",
                self._best_visible_source_wavelength())
            carrier = self.params.get("carrier_period_px", 8)
            amp, phase = Holo.fresnel_hologram(target, z_mm, wavelength_nm,
                self.config.dmd.pitch_um if self.config else 7.6)
            binary = Holo.lee_binary_hologram(amp, phase, carrier_period_px=carrier)
            self.dlp.project_static(self._upscale_to_dmd(binary))
            reconstruction = Holo.simulate_4f_reconstruction(binary,
                carrier_period_px=carrier)
            quality = 1.0
            if self.config:
                best = max(self.config.sources.values(),
                           key=lambda s: s.spatial_coherence_um * s.temporal_coherence_um,
                           default=None)
                if best is not None:
                    quality = Holo.coherence_penalty(amp, phase,
                        best.spatial_coherence_um, best.temporal_coherence_um,
                        self.config.dmd.pitch_um, wavelength_nm)
            note = "OK — physically meaningful." if ok else f"⚠ {why}"
            return Holo.HolographyResult(binary_pattern=binary,
                reconstruction=reconstruction, coherence_quality=quality,
                method=f"Lee binary CGH at {carrier}-px carrier, Fresnel-propagated {z_mm} mm",
                notes=note)

        if m == SensorMode.INTERFEROMETRY:
            patterns = P.phase_shift_fringes(
                n_steps=self.params.get("n_steps", 4),
                period_px=self.params.get("period_px", 24))
            frames = []
            for pat in patterns:
                self.dlp.project_static(pat)
                frames.append(self.camera.capture())
            return D.decode_phase_shift(frames)

        raise ValueError(f"Unknown mode: {m}")

    def _default_holo_target(self):
        """Default target: a centered 'M'."""
        img = np.zeros((128, 128), dtype=np.uint8)
        img[30:100,  30:38] = 255
        img[30:100,  90:98] = 255
        for k in range(40):
            img[30 + k, 38 + k] = 255
            img[30 + k, 90 - k] = 255
        return img

    def _upscale_to_dmd(self, arr):
        from ._utils import resample_nearest
        return resample_nearest(arr, (P.DMD_H, P.DMD_W))

    def _best_visible_source_wavelength(self):
        if not self.config:
            return 520.0
        visible = [s for s in self.config.sources.values()
                   if 400 <= s.center_nm <= 700]
        if not visible:
            return 520.0
        return max(visible, key=lambda s: s.temporal_coherence_um).center_nm

    def __enter__(self): return self
    def __exit__(self, *exc): self.close()

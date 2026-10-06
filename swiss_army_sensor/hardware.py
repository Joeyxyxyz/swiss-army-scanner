"""
swiss_army_sensor.hardware
==========================

DMD chip profiles, light-source coherence model, camera profiles, and
RigConfig that bundles them together with a `can_run(mode)` feasibility check.

Coherence is NOT binary. Three numbers fully characterize a source:
  spatial_coherence_um   — diameter of the spatially-coherent area at the DMD
  temporal_coherence_um  — L_c = λ²/Δλ
  transverse_modes       — 1 = single-mode, >1 = multi-mode

These determine which sensor modes are physically meaningful, not just
whether to call something a "laser".
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import math


@dataclass(frozen=True)
class DMDProfile:
    name: str
    cols: int
    rows: int
    pitch_um: float
    tilt_deg: float
    controller: str
    window_material: str = "Corning Eagle XG"
    window_T_min_nm: float = 420.0
    window_T_max_nm: float = 680.0
    fill_factor: float = 0.92
    throw_ratio: float = 1.4
    f_number: float = 2.1

    def littrow_wavelength_nm(self, order: int = 1) -> float:
        return 2 * self.pitch_um * 1000 * math.sin(math.radians(self.tilt_deg)) / order

    def projected_pixel_mm(self, working_distance_mm: float) -> float:
        return (working_distance_mm / self.throw_ratio) / self.cols


DMD_PROFILES = {
    "DLP4500":     DMDProfile("DLP4500",     912, 1140, 7.6,  12.0, "DLPC350"),
    "DLP4500NIR":  DMDProfile("DLP4500NIR",  912, 1140, 7.6,  12.0, "DLPC350",
                              window_material="NIR-AR coated",
                              window_T_min_nm=850.0, window_T_max_nm=2500.0),
    "DLP3010":     DMDProfile("DLP3010",    1280,  720, 5.4,  17.0, "DLPC3433"),
    "DLP3010ULC":  DMDProfile("DLP3010ULC", 1280,  720, 5.4,  17.0, "DLPC3438"),
    "DLP471TP":    DMDProfile("DLP471TP",  3840, 2160, 5.4,  17.0, "DLPC6540"),
    "DLP472TP":    DMDProfile("DLP472TP",  3840, 2160, 5.4,  17.0, "DLPC8445"),
    "DLP670S":     DMDProfile("DLP670S",   1920, 1080, 5.4,  17.0, "DLPC3479"),
}


@dataclass(frozen=True)
class LightSource:
    name: str
    center_nm: float
    fwhm_nm: float
    spatial_coherence_um: float = 0.0
    temporal_coherence_um: float = 0.0
    transverse_modes: int = 1000
    flux_mw: float = 100.0
    notes: str = ""

    @property
    def is_coherent_temporal(self) -> bool:
        return self.temporal_coherence_um >= 10.0

    @property
    def is_coherent_spatial(self) -> bool:
        return self.spatial_coherence_um >= 50.0

    @property
    def supports_holography(self) -> bool:
        return self.is_coherent_temporal and self.is_coherent_spatial

    @property
    def supports_speckle(self) -> bool:
        return (self.temporal_coherence_um >= 100.0
                and self.spatial_coherence_um >= 200.0)

    def coherence_length_from_fwhm(self) -> float:
        if self.fwhm_nm <= 0:
            return float("inf")
        return (self.center_nm * 1e-3) ** 2 * 1000 / (self.fwhm_nm * 1e-3)


RAW_LEDS = {
    "red_raw":   LightSource("red_raw",   625, 20,  spatial_coherence_um=0.5,
                             temporal_coherence_um=20, transverse_modes=10000,
                             notes="Stock OSRAM-class red LED."),
    "green_raw": LightSource("green_raw", 520, 35,  spatial_coherence_um=0.4,
                             temporal_coherence_um=8,  transverse_modes=10000),
    "blue_raw":  LightSource("blue_raw",  460, 25,  spatial_coherence_um=0.6,
                             temporal_coherence_um=9,  transverse_modes=10000),
}

CONDITIONED_LEDS = {
    "red_pinhole_75um":   LightSource("red_pinhole_75um", 625, 1.0,
        spatial_coherence_um=270, temporal_coherence_um=390, transverse_modes=8,
        flux_mw=2.0, notes="Stock red LED + 75 µm pinhole + 1 nm filter (Chu et al.)"),
    "green_pinhole_75um": LightSource("green_pinhole_75um", 520, 1.0,
        spatial_coherence_um=270, temporal_coherence_um=270, transverse_modes=8,
        flux_mw=2.5),
    "blue_pinhole_75um":  LightSource("blue_pinhole_75um",  460, 1.0,
        spatial_coherence_um=270, temporal_coherence_um=210, transverse_modes=8,
        flux_mw=1.5),
    "green_smf_coupled":  LightSource("green_smf_coupled",  520, 5.0,
        spatial_coherence_um=10000, temporal_coherence_um=54, transverse_modes=1,
        flux_mw=0.5),
}

LASERS = {
    "hene_633":   LightSource("hene_633",   633, 0.002,
                              spatial_coherence_um=1e6, temporal_coherence_um=2e8,
                              transverse_modes=1, flux_mw=5,
                              notes="HeNe — matches DLP4500 order-5 Littrow."),
    "diode_638":  LightSource("diode_638",  638, 0.5,
                              spatial_coherence_um=1e5, temporal_coherence_um=800,
                              transverse_modes=1, flux_mw=50),
    "ld_532":     LightSource("ld_532",     532, 0.1,
                              spatial_coherence_um=1e5, temporal_coherence_um=2800,
                              transverse_modes=1, flux_mw=50,
                              notes="DPSS — matches DLP4500 order-6 Littrow."),
    "ld_488":     LightSource("ld_488",     488, 0.1,
                              spatial_coherence_um=1e5, temporal_coherence_um=2400,
                              transverse_modes=1, flux_mw=20),
    "ld_405":     LightSource("ld_405",     405, 0.5,
                              spatial_coherence_um=1e5, temporal_coherence_um=330,
                              transverse_modes=1, flux_mw=80),
    "swir_1550":  LightSource("swir_1550", 1550, 1.0,
                              spatial_coherence_um=1e5, temporal_coherence_um=2400,
                              transverse_modes=1, flux_mw=20,
                              notes="Order-2 Littrow blazed on DLP4500."),
}

NIR_LEDS = {
    "nir_780":       LightSource("nir_780",       780,  30),
    "nir_850":       LightSource("nir_850",       850,  40),
    "nir_940":       LightSource("nir_940",       940,  45),
    "swir_1310_led": LightSource("swir_1310_led", 1310, 60),
    "swir_1550_led": LightSource("swir_1550_led", 1550, 70),
}


@dataclass(frozen=True)
class CameraProfile:
    name: str
    width: int
    height: int
    pixel_um: float
    max_fps: float
    qe_peak_pct: float
    qe_curve_band: tuple
    has_hardware_trigger: bool = True


CAMERA_PROFILES = {
    "FLIR_BFS_U3_16S2M":  CameraProfile("FLIR Blackfly S BFS-U3-16S2M-CS",
                                        1440, 1080, 3.45, 226.0, 76, (400, 950)),
    "FLIR_BFS_U3_50S5M":  CameraProfile("FLIR Blackfly S BFS-U3-50S5M (IMX264)",
                                        2448, 2048, 3.45, 75.0,  72, (400, 950)),
    "Daheng_MER2_160":    CameraProfile("Daheng MER2-160-227U3M",
                                        1440, 1080, 3.45, 227.0, 76, (400, 950)),
    "IDS_U3_3040CP":      CameraProfile("IDS U3-3040CP-M-GL",
                                        1456, 1088, 3.45, 251.0, 76, (400, 950)),
    "InGaAs_640_SWIR":    CameraProfile("Lynred/Allied SWIR InGaAs 640×512",
                                        640, 512, 15.0, 120.0,  80, (900, 1700)),
}


@dataclass
class RigConfig:
    dmd: DMDProfile
    sources: dict
    camera: CameraProfile

    def can_run(self, mode: str) -> tuple:
        any_visible = any(400 <= s.center_nm <= 700 for s in self.sources.values())
        any_nir     = any(700 <  s.center_nm <= 2500 for s in self.sources.values())
        any_coherent_full = any(s.supports_holography for s in self.sources.values())
        any_supports_speckle = any(s.supports_speckle for s in self.sources.values())

        if mode == "holography":
            if not any_coherent_full:
                return (False, "Holography needs spatial AND temporal coherence. "
                               "Add a pinhole-conditioned LED or a laser.")
            return (True, "OK — partial-coherence holography possible.")
        if mode == "speckle_roughness":
            if not any_supports_speckle:
                return (False, "Speckle needs L_c ≳ 100 µm and large spatial coherence.")
            return (True, "OK")
        if mode == "interferometry":
            if not any_coherent_full:
                return (False, "Need coherent source.")
            return (True, "OK")
        if mode == "nir_spectroscopy":
            if not any_nir:
                return (False, "Add a NIR/SWIR LED via the external LED driver header.")
            if self.dmd.window_T_max_nm < 1000:
                return (False, f"{self.dmd.name} window cuts off at "
                               f"{self.dmd.window_T_max_nm} nm — use DLP4500NIR.")
            if self.camera.qe_curve_band[1] < 1000:
                return (False, f"Camera {self.camera.name} is silicon — "
                               "NIR/SWIR above 950 nm needs InGaAs.")
            return (True, "OK")
        return (True, "Mode is broad-band tolerant.")


DEFAULT_RIG = RigConfig(
    dmd=DMD_PROFILES["DLP4500"],
    sources={**RAW_LEDS},
    camera=CAMERA_PROFILES["FLIR_BFS_U3_16S2M"],
)

HOLOGRAPHY_RIG = RigConfig(
    dmd=DMD_PROFILES["DLP4500"],
    sources={**CONDITIONED_LEDS, "hene_633": LASERS["hene_633"]},
    camera=CAMERA_PROFILES["FLIR_BFS_U3_16S2M"],
)

NIR_SPECTROSCOPY_RIG = RigConfig(
    dmd=DMD_PROFILES["DLP4500NIR"],
    sources={**NIR_LEDS},
    camera=CAMERA_PROFILES["InGaAs_640_SWIR"],
)

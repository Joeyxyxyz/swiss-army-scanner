"""
swiss_army_sensor.spectral
==========================

Spectral / wavelength-aware sensor extensions:

  - spot_emulation              0.75–0.9 mm and 6.5 mm laser-spot sensor calibration
  - spectral_3band              ratio-based color/coating tint
  - thin_film_thickness         Newton's rings interferometry, 50 nm – 5 µm
  - material_identification     RGB or NIR-band reflectance signature match
  - speckle_roughness           coherent-light Ra estimation
  - nir_spectroscopy            DLP4500NIR / external NIR LED bank

Hardware references (DLPC350 + Corning Eagle XG window):
  Mirror pitch          7.6 µm (DLP4500); 5.4 µm (DLP3010)
  Window transmittance  ≥ 97% over 420–680 nm (AOI 0°–30°)
  Mirror reflectance    > 85% across 400 nm – 2.5 µm (aluminium)

Coherent-light caveats (TI DLPA037):
  Orders at  (m·λ/d, n·λ/d), fixed by λ and incidence — switching mirrors
  does NOT move them; toggles which order carries energy.
  Littrow blaze: θ_in = θ_out = ±θ_tilt.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import math
import numpy as np


class DLP4500:
    PITCH_UM = 7.6
    COLS = 912
    ROWS = 1140
    THROW_RATIO = 1.4
    F_NUMBER = 2.1
    TILT_DEG = 12.0


@dataclass(frozen=True)
class LED:
    name: str
    center_nm: float
    fwhm_nm: float
    typical_lm: float
    coherent: bool = False

STOCK_LEDS = {
    "red":   LED("red",   625.0, 20.0, 50.0),
    "green": LED("green", 520.0, 35.0, 60.0),
    "blue":  LED("blue",  460.0, 25.0, 40.0),
}

EXTENDED_SOURCES = {
    "uv_405":     LED("uv_405",     405.0,  3.0,  20.0, coherent=True),
    "blue_laser": LED("blue_laser", 488.0,  0.1,  50.0, coherent=True),
    "green_laser":LED("green_laser",532.0,  0.1,  50.0, coherent=True),
    "red_laser":  LED("red_laser",  638.0,  0.1,  60.0, coherent=True),
    "nir_780":    LED("nir_780",    780.0, 30.0,  10.0),
    "nir_850":    LED("nir_850",    850.0, 40.0,  10.0),
    "nir_940":    LED("nir_940",    940.0, 45.0,  10.0),
    "swir_1310":  LED("swir_1310", 1310.0, 60.0,  5.0),
    "swir_1550":  LED("swir_1550", 1550.0, 70.0,  5.0),
}


def pixel_size_at_distance(z_mm: float, dlp=DLP4500) -> float:
    return (z_mm / dlp.THROW_RATIO) / dlp.COLS


def spot_radius_to_pixels(spot_mm: float, z_mm: float, dlp=DLP4500) -> int:
    return max(1, int(round(spot_mm / pixel_size_at_distance(z_mm, dlp))))


def diffraction_order_angle(m: int, n: int, wavelength_nm: float,
                            pitch_um: float = DLP4500.PITCH_UM):
    """Order positions per TI DLPA037: sin(θ) = m·λ/d."""
    lam_um = wavelength_nm * 1e-3
    sx = m * lam_um / pitch_um
    sy = n * lam_um / pitch_um
    if abs(sx) > 1 or abs(sy) > 1:
        return (float("nan"), float("nan"))
    return (math.degrees(math.asin(sx)), math.degrees(math.asin(sy)))


def littrow_blaze_wavelength(order: int = 1, pitch_um: float = DLP4500.PITCH_UM,
                             tilt_deg: float = DLP4500.TILT_DEG) -> float:
    """λ_blaze(m) = 2·d·sin(θ_tilt) / m. For DLP4500: m=2 → 1580nm, m=5 → 632nm."""
    return 2 * pitch_um * 1000 * math.sin(math.radians(tilt_deg)) / order


WAVELENGTHS_NM = np.array([400, 460, 520, 580, 625, 700, 780, 850, 940, 1100, 1310, 1550], dtype=float)

MATERIAL_LIBRARY = {
    "aluminum_polished":  np.array([0.92,0.91,0.91,0.90,0.89,0.88,0.87,0.87,0.86,0.85,0.84,0.82]),
    "aluminum_anodized":  np.array([0.55,0.62,0.66,0.68,0.70,0.71,0.72,0.73,0.74,0.74,0.73,0.71]),
    "steel_brushed":      np.array([0.55,0.58,0.60,0.62,0.63,0.64,0.65,0.66,0.66,0.67,0.67,0.66]),
    "steel_oxidized":     np.array([0.18,0.20,0.22,0.24,0.25,0.27,0.30,0.33,0.36,0.41,0.45,0.46]),
    "copper":             np.array([0.32,0.35,0.40,0.55,0.78,0.93,0.96,0.97,0.97,0.97,0.96,0.95]),
    "battery_cathode":    np.array([0.04,0.05,0.06,0.07,0.08,0.09,0.10,0.11,0.12,0.13,0.14,0.14]),
    "battery_anode_gr":   np.array([0.06,0.07,0.08,0.08,0.08,0.09,0.10,0.10,0.11,0.11,0.12,0.12]),
    "cu_foil":            np.array([0.30,0.33,0.40,0.55,0.78,0.92,0.95,0.96,0.96,0.96,0.95,0.94]),
    "al_foil":            np.array([0.92,0.91,0.91,0.90,0.89,0.88,0.87,0.87,0.86,0.85,0.84,0.82]),
    "pet_plastic":        np.array([0.06,0.07,0.08,0.08,0.08,0.08,0.10,0.20,0.35,0.55,0.30,0.10]),
    "pp_plastic":         np.array([0.07,0.08,0.08,0.08,0.09,0.10,0.12,0.25,0.45,0.65,0.20,0.55]),
    "pe_plastic":         np.array([0.07,0.08,0.08,0.08,0.09,0.10,0.13,0.30,0.50,0.60,0.15,0.45]),
    "kapton":             np.array([0.02,0.05,0.20,0.45,0.55,0.65,0.72,0.76,0.78,0.80,0.79,0.76]),
    "white_paper":        np.array([0.78,0.84,0.86,0.85,0.84,0.82,0.78,0.74,0.70,0.55,0.30,0.15]),
    "silicon_wafer":      np.array([0.45,0.40,0.37,0.35,0.34,0.33,0.32,0.32,0.32,0.32,0.30,0.28]),
    "FR4_pcb_green":      np.array([0.05,0.10,0.45,0.30,0.20,0.18,0.20,0.22,0.25,0.30,0.30,0.25]),
}


def led_response(material: str, led: LED) -> float:
    R = MATERIAL_LIBRARY[material]
    sigma = led.fwhm_nm / 2.355
    weights = np.exp(-0.5 * ((WAVELENGTHS_NM - led.center_nm) / sigma) ** 2)
    if weights.sum() < 1e-9:
        return 0.0
    return float(np.sum(R * weights) / weights.sum())


def material_signature(material: str, leds) -> np.ndarray:
    return np.array([led_response(material, led) for led in leds])


@dataclass
class SpotEmulationResult:
    spot_diameter_mm: float
    spot_pixels_radius: int
    working_distance_mm: float
    pixel_size_mm: float
    on_target: bool
    return_intensity: float

@dataclass
class SpectralBandResult:
    bands: dict
    chromaticity: tuple
    notes: str

@dataclass
class ThinFilmResult:
    thickness_nm: float
    confidence: float
    n_film_assumed: float
    method: str
    fringe_orders_seen: list

@dataclass
class MaterialIDResult:
    measured_signature: np.ndarray
    leds_used: list
    top_matches: list
    confidence: float

@dataclass
class SpeckleRoughnessResult:
    Ra_um_estimate: float
    contrast: float
    coherence_assumed_um: float
    warning: Optional[str] = None

@dataclass
class NIRSpectroscopyResult:
    band_responses: dict
    detected_components: list
    confidence: float


def emulate_spot(spot_diameter_mm: float, working_distance_mm: float = 300,
                 returned_intensity: float = 0.5, dlp=DLP4500) -> SpotEmulationResult:
    """
    Reference laser-spot sensor measurements (2026):
        small spot: 0.75–0.9 mm → ~3-4 DMD px radius @ 300 mm WD
        large spot: 6.5 mm     → ~14 DMD px radius   @ 300 mm WD
    """
    pix_mm = pixel_size_at_distance(working_distance_mm, dlp)
    r_px = spot_radius_to_pixels(spot_diameter_mm / 2, working_distance_mm, dlp)
    return SpotEmulationResult(
        spot_diameter_mm=spot_diameter_mm,
        spot_pixels_radius=r_px,
        working_distance_mm=working_distance_mm,
        pixel_size_mm=pix_mm,
        on_target=returned_intensity > 0.1,
        return_intensity=returned_intensity,
    )


def three_band_spectral(r: float, g: float, b: float) -> SpectralBandResult:
    s = r + g + b + 1e-9
    return SpectralBandResult(
        bands={"red_625nm": r, "green_520nm": g, "blue_460nm": b},
        chromaticity=(r / s, g / s, b / s),
        notes="Stock LightCrafter LEDs. For polymer ID add NIR.",
    )


def estimate_thin_film_thickness(r: float, g: float, b: float,
                                 n_film: float = 1.5) -> ThinFilmResult:
    """
    Three-band Newton's rings inversion. Works for transparent films 50 nm – 3 µm.
    """
    lams = np.array([625, 520, 460])
    bands = np.array([r, g, b])
    bands = (bands - bands.min()) / (np.ptp(bands) + 1e-9)

    best_t, best_err = 0.0, float("inf")
    fringe_orders = []
    for t in np.linspace(50, 3000, 591):
        delta = 2 * n_film * t / lams
        modeled = 0.5 + 0.5 * np.cos(2 * math.pi * delta)
        err = float(np.mean((modeled - bands) ** 2))
        if err < best_err:
            best_err = err
            best_t = t
            fringe_orders = [int(d) for d in delta]
    return ThinFilmResult(
        thickness_nm=best_t,
        confidence=float(np.exp(-10 * best_err)),
        n_film_assumed=n_film,
        method="3-band visible interference",
        fringe_orders_seen=fringe_orders,
    )


def identify_material(intensities_per_led: dict, use_nir: bool = False) -> MaterialIDResult:
    led_objs = []
    measured = []
    for name, val in intensities_per_led.items():
        if name in STOCK_LEDS:
            led_objs.append(STOCK_LEDS[name])
        elif name in EXTENDED_SOURCES:
            led_objs.append(EXTENDED_SOURCES[name])
        else:
            continue
        measured.append(val)
    measured = np.asarray(measured, dtype=float)
    if measured.size == 0:
        return MaterialIDResult(measured, [], [], 0.0)

    scores = []
    for mat in MATERIAL_LIBRARY:
        sig = material_signature(mat, led_objs)
        num = float(np.dot(measured, sig))
        den = float(np.linalg.norm(measured) * np.linalg.norm(sig) + 1e-9)
        scores.append((mat, num / den))
    scores.sort(key=lambda kv: -kv[1])
    return MaterialIDResult(
        measured_signature=measured,
        leds_used=[l.name for l in led_objs],
        top_matches=scores[:5],
        confidence=float(scores[0][1] - scores[1][1]),
    )


def estimate_speckle_roughness(speckle_image: np.ndarray,
                               wavelength_nm: float = 638.0) -> SpeckleRoughnessResult:
    """Goodman 1976: Ra ≈ (λ / 4π) · √(-ln(1 - K²)), valid Ra ≲ λ/4."""
    I = speckle_image.astype(np.float32)
    mean = float(I.mean())
    std  = float(I.std())
    K = std / (mean + 1e-9)
    K = min(K, 0.999)
    Ra_um = (wavelength_nm * 1e-3 / (4 * math.pi)) * math.sqrt(-math.log(1 - K * K))
    warning = None
    if mean < 5:
        warning = "Very low signal — speckle contrast may be noise-dominated."
    if K < 0.05:
        warning = "Low speckle contrast suggests incoherent source. Coherent laser required."
    return SpeckleRoughnessResult(Ra_um_estimate=Ra_um, contrast=K,
                                  coherence_assumed_um=wavelength_nm * 1e-3,
                                  warning=warning)


def nir_spectroscopy(band_intensities: dict) -> NIRSpectroscopyResult:
    """
    Diagnostic NIR bands:
      1210 nm  -CH₂ overtone  → fats, oils, polymers
      1450 nm  O-H overtone   → water content
      1660 nm  C-H aromatic   → PET vs PE/PP
    """
    detected = []
    if band_intensities.get("nir_940", 0) > 0.4 and band_intensities.get("swir_1310", 0) < 0.2:
        detected.append("water (high O-H absorption)")
    if (band_intensities.get("swir_1310", 0) > 0.5
        and band_intensities.get("swir_1550", 0) > 0.4):
        detected.append("polyolefin (PE/PP) — CH₂ overtone")
    if (band_intensities.get("swir_1310", 0) < 0.3
        and band_intensities.get("swir_1550", 0) < 0.2
        and band_intensities.get("nir_940", 0) > 0.4):
        detected.append("PET / ester polymer")
    return NIRSpectroscopyResult(
        band_responses=band_intensities,
        detected_components=detected,
        confidence=min(0.95, 0.5 + 0.1 * len(detected)),
    )

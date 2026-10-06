"""
swiss_army_sensor.holography
============================

Computer-generated holography (CGH) on a DMD.

Lee binary holographic encoding (Lee, 1974) — high-frequency carrier
grating modulated by target amplitude and phase. After a 4f relay with
a Fourier-plane aperture selecting the +1 diffraction order:
  E(x,y) ∝ A(x,y) · exp(i·φ(x,y))

References:
  WO2005099386                              DMD holographic projector
  Lee (1970)                                Sampled FT hologram by computer
  Shin et al. (2015) arXiv:1506.06237       Active illumination QPI
  Peng et al., Sci. Adv. 7 (2021)          Speckle-free CGH with partial coherence
  Pourreza Ghoushchi et al., SPIE 10126     LED spatial coherence in CGH displays
  Chu et al.                                Optimal 270 µm spatial filter

LED-with-pinhole CGH is real. supports_holography is True when
spatial_coherence ≥ 50 µm AND L_c ≥ 10 µm.
"""

from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np


def lee_binary_hologram(amplitude: np.ndarray, phase: np.ndarray,
                        carrier_period_px: int = 8,
                        carrier_direction: str = "x") -> np.ndarray:
    """
    Lee binary encoding:
        h(x,y) = 1 if cos(2π·x/T - φ(x,y)) > 1 - 2·A(x,y) else 0
    After 4f Fourier filtering of the +1 order, reconstructed field matches U.
    """
    H, W = amplitude.shape
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    coord = xx if carrier_direction == "x" else yy
    carrier = 2 * math.pi * coord / carrier_period_px
    pattern_continuous = np.cos(carrier - phase)
    threshold = 1.0 - 2.0 * amplitude
    return (pattern_continuous > threshold).astype(np.uint8) * 255


def fresnel_hologram(target_image: np.ndarray, propagation_distance_mm: float,
                     wavelength_nm: float, pixel_pitch_um: float):
    """Fresnel hologram via angular-spectrum (Goodman §4.2). Returns (amp, phase)."""
    H, W = target_image.shape
    k = 2 * math.pi / (wavelength_nm * 1e-6)
    fx = np.fft.fftfreq(W, d=pixel_pitch_um * 1e-3)
    fy = np.fft.fftfreq(H, d=pixel_pitch_um * 1e-3)
    FX, FY = np.meshgrid(fx, fy)
    arg = 1 - (wavelength_nm * 1e-6 * FX) ** 2 - (wavelength_nm * 1e-6 * FY) ** 2
    arg = np.clip(arg, 0, None)
    transfer = np.exp(1j * k * propagation_distance_mm * np.sqrt(arg))
    target_field = np.sqrt(np.maximum(target_image.astype(np.float32) / 255.0, 0))
    spectrum = np.fft.fft2(target_field)
    dmd_field = np.fft.ifft2(spectrum * np.conj(transfer))
    amp = np.abs(dmd_field)
    amp = amp / (amp.max() + 1e-9)
    phase = np.angle(dmd_field)
    return amp, phase


def simulate_4f_reconstruction(binary_hologram: np.ndarray,
                               carrier_period_px: int = 8,
                               carrier_direction: str = "x",
                               aperture_radius_frac: float = 0.10) -> np.ndarray:
    """Simulate 4f optical reconstruction: FFT → select +1 order → IFFT → |·|."""
    H, W = binary_hologram.shape
    F = np.fft.fftshift(np.fft.fft2(binary_hologram.astype(np.float32) / 255.0))
    fy, fx = np.indices((H, W))
    cy, cx = H // 2, W // 2
    if carrier_direction == "x":
        order_x = cx + W // carrier_period_px
        order_y = cy
    else:
        order_x = cx
        order_y = cy + H // carrier_period_px
    r = min(H, W) * aperture_radius_frac
    mask = ((fy - order_y) ** 2 + (fx - order_x) ** 2) <= r ** 2
    F_filtered = F * mask
    if carrier_direction == "x":
        F_filtered = np.roll(F_filtered, -W // carrier_period_px, axis=1)
    else:
        F_filtered = np.roll(F_filtered, -H // carrier_period_px, axis=0)
    reconstructed = np.fft.ifft2(np.fft.ifftshift(F_filtered))
    return np.abs(reconstructed)


def coherence_penalty(amplitude: np.ndarray, phase: np.ndarray,
                      spatial_coherence_um: float, temporal_coherence_um: float,
                      pixel_pitch_um: float, wavelength_nm: float) -> float:
    """
    η = η_spatial · η_temporal, in [0, 1].
    Stock LED → 0.001, pinhole-conditioned LED → 0.5, laser → 1.0.
    """
    Npx = max(amplitude.shape)
    hologram_extent_um = Npx * pixel_pitch_um
    eta_s = min(1.0, spatial_coherence_um / hologram_extent_um)
    depth_extent_um = (wavelength_nm * 1e-3)
    eta_t = min(1.0, temporal_coherence_um / max(depth_extent_um, 1e-6))
    eta_t = min(1.0, eta_t)
    return eta_s * eta_t


@dataclass
class HolographyResult:
    binary_pattern: np.ndarray
    reconstruction: np.ndarray
    coherence_quality: float
    method: str
    notes: str

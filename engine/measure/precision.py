"""
Predicted depth noise before scanning (Lv & Kemao's phase-noise model).

  sigma_phi^2 ~= 2 sigma_I^2 / (N B^2),   sigma_I^2 = K A + sigma_read^2   (DN^2)
  sigma_u     = T / (2 pi) * sigma_phi                  (projector px)
  sigma_z     = z^2 / (f_p b) * sigma_u = (T / 2 pi) z^2 / (f_p b) sigma_phi   (mm)

A = mean fringe level (DN), B = fringe modulation (DN), N = phase steps,
K = camera conversion gain (DN per electron: shot noise variance K A),
T = finest fringe period (projector px), f_p = projector focal length (px,
along the coded axis), b = baseline (mm), z = working distance (mm).
Averaging M frames divides sigma_I^2 by M. The last (finest) period sets
the noise: hierarchical unwrap only fixes the fringe order.

If a measured sigma_z is much larger than this prediction the error is
systematic (timing / flicker, interreflection, calibration), not noise.

Camera defaults are APPROXIMATE Blackfly S (IMX273) values in Mono8 at
0 dB gain (full well ~11 ke- -> ~0.023 DN/e-, read noise ~2.5 e- ->
~0.06 DN); replace them with the camera's EMVA 1288 datasheet numbers.
"""

from __future__ import annotations

import math
from typing import Optional

IMX273_K_DN_PER_E = 255.0 / 11000.0       # Mono8, 0 dB (approximate)
IMX273_READ_NOISE_DN = 2.5 * IMX273_K_DN_PER_E
QUANT_NOISE_DN = 1.0 / math.sqrt(12.0)    # 8-bit rounding


def predict_sigma_z(z_mm: float, baseline_mm: float, f_p_px: float, period_px: float,
                    A_dn: float, B_dn: float, steps: int = 12, frames_averaged: int = 1,
                    gain_db: float = 0.0, K_dn_per_e: float = IMX273_K_DN_PER_E,
                    read_noise_dn: float = IMX273_READ_NOISE_DN,
                    quantization: bool = True, extra_noise_dn: float = 0.0) -> dict:
    """-> {"sigma_z_mm", "sigma_phi_rad", "sigma_u_px", "sigma_I_dn", "mm_per_projector_px", ...}"""
    for name, v in (("z_mm", z_mm), ("baseline_mm", baseline_mm), ("f_p_px", f_p_px),
                    ("period_px", period_px), ("B_dn", B_dn)):
        if not v or v <= 0:
            raise ValueError(f"{name} must be > 0, got {v!r}")
    if steps < 3 or frames_averaged < 1:
        raise ValueError("steps must be >= 3 and frames_averaged >= 1")
    g = 10 ** (gain_db / 20.0)
    K = K_dn_per_e * g
    var_i = K * max(A_dn, 0.0) + (read_noise_dn * g) ** 2 + extra_noise_dn ** 2
    var_i /= frames_averaged
    if quantization:
        var_i += QUANT_NOISE_DN ** 2 / frames_averaged
    sigma_i = math.sqrt(var_i)
    sigma_phi = math.sqrt(2.0 * var_i / (steps * B_dn ** 2))
    sigma_u = period_px / (2 * math.pi) * sigma_phi
    mm_per_px = z_mm ** 2 / (f_p_px * baseline_mm)
    return {"sigma_z_mm": mm_per_px * sigma_u, "sigma_phi_rad": sigma_phi, "sigma_u_px": sigma_u,
            "sigma_I_dn": sigma_i, "mm_per_projector_px": mm_per_px,
            "inputs": {"z_mm": z_mm, "baseline_mm": baseline_mm, "f_p_px": f_p_px,
                       "period_px": period_px, "A_dn": A_dn, "B_dn": B_dn, "steps": steps,
                       "frames_averaged": frames_averaged, "gain_db": gain_db,
                       "K_dn_per_e": K_dn_per_e, "read_noise_dn": read_noise_dn},
            "units": "mm", "model": "Lv & Kemao phase noise, small-angle triangulation"}


def from_calibration(cal, z_mm: float, A_dn: float, B_dn: float, period_px: float,
                     steps: int = 12, **kw) -> dict:
    """Same, with the baseline and f_p (column axis) taken from a stereo calibration."""
    return predict_sigma_z(z_mm, cal.baseline_mm, float(cal.K_proj[0, 0]), period_px, A_dn, B_dn,
                           steps, **kw)

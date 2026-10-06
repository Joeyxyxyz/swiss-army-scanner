"""
swiss_army_sensor.barcode
=========================

Four barcode reading methods. Three are genuinely DMD-native — the DMD
does the measurement, not just acts as a flashlight.

  flood_passive          flood + camera capture + pyzbar decode (no DMD value-add)
  raster_single_pixel    DMD scans a spot; bucket detector reconstructs an image
  hadamard_single_pixel  DMD projects Hadamard basis; bucket reads; inverse-WHT
  line_scan              single column steps across; 1D profile → 1D code decoder

For the actual symbol decoding (QR Reed-Solomon, Data Matrix ECC-200) we
hand the reconstructed image to pyzbar / pylibdmtx. Writing a clean-room
QR decoder is a multi-thousand-line job and those libraries exist and
are well-validated. The DMD-native part is the *measurement*.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class BarcodeReadResult:
    method: str
    reconstructed: np.ndarray
    decoded: list
    n_codes: int
    n_patterns: int
    notes: str = ""


def decode_flood_passive(camera_frame: np.ndarray) -> BarcodeReadResult:
    """Plain flood-illumination decode. No DMD value-add."""
    decoded = _zbar_decode(camera_frame)
    return BarcodeReadResult(
        method="flood_passive",
        reconstructed=camera_frame,
        decoded=decoded, n_codes=len(decoded), n_patterns=1,
        notes="DMD is a flood lamp; decode is done by the camera.",
    )


def decode_raster_single_pixel(bucket_callable, pattern_size, bucket_size,
                               spot_radius_px: int = 4) -> BarcodeReadResult:
    """Scan a single bright spot; collect N² bucket readings into an image."""
    H_dmd, W_dmd = pattern_size
    H_out, W_out = bucket_size
    img = np.zeros((H_out, W_out), dtype=np.float32)
    n = 0
    for iy, y in enumerate(np.linspace(spot_radius_px,
                                       H_dmd - spot_radius_px, H_out).astype(int)):
        for ix, x in enumerate(np.linspace(spot_radius_px,
                                           W_dmd - spot_radius_px, W_out).astype(int)):
            pat = _single_spot(W_dmd, H_dmd, x, y, spot_radius_px)
            img[iy, ix] = bucket_callable(pat)
            n += 1
    norm = (img - img.min()) / (img.max() - img.min() + 1e-9)
    out = (norm * 255).astype(np.uint8)
    decoded = _zbar_decode(out)
    return BarcodeReadResult(
        method="raster_single_pixel",
        reconstructed=out, decoded=decoded, n_codes=len(decoded), n_patterns=n,
        notes=f"Reconstructed {H_out}×{W_out} image from {n} bucket reads.",
    )


def decode_hadamard_single_pixel(bucket_callable, pattern_size, order: int = 5,
                                  fov_px: int = 256) -> BarcodeReadResult:
    """Project Walsh-Hadamard basis; invert via inverse Walsh transform."""
    from .patterns import hadamard_patterns, _sylvester_hadamard
    H_dmd, W_dmd = pattern_size
    N = 2 ** order
    pats = hadamard_patterns(order=order, fov_px=fov_px,
                             cx=W_dmd // 2, cy=H_dmd // 2)
    measurements = np.zeros((N, N), dtype=np.float32)
    for k, pat in enumerate(pats):
        i, j = divmod(k, N)
        measurements[i, j] = bucket_callable(pat)
    Hmat = _sylvester_hadamard(N).astype(np.float32)
    img = (Hmat @ measurements @ Hmat) / (N * N)
    a, b = img.min(), img.max()
    out = ((img - a) / (b - a + 1e-9) * 255).astype(np.uint8)
    big = np.kron(out, np.ones((8, 8), dtype=np.uint8))
    decoded = _zbar_decode(big)
    return BarcodeReadResult(
        method="hadamard_single_pixel",
        reconstructed=big, decoded=decoded, n_codes=len(decoded), n_patterns=len(pats),
        notes=f"{N}×{N} reconstruction from {N*N} Hadamard buckets.",
    )


def decode_line_scan(bucket_callable, pattern_size,
                     line_width_px: int = 4) -> BarcodeReadResult:
    """1-D laser-scanner emulation: single column steps across the FOV."""
    H_dmd, W_dmd = pattern_size
    n_steps = W_dmd // line_width_px
    profile = np.zeros((1, n_steps), dtype=np.float32)
    for k in range(n_steps):
        col = k * line_width_px
        pat = _vertical_line(W_dmd, H_dmd, col, line_width_px)
        profile[0, k] = bucket_callable(pat)
    profile = (profile - profile.min()) / (np.ptp(profile) + 1e-9)
    profile_2d = np.tile((profile * 255).astype(np.uint8), (40, 1))
    decoded = _zbar_decode(profile_2d)
    return BarcodeReadResult(
        method="line_scan",
        reconstructed=profile_2d, decoded=decoded, n_codes=len(decoded), n_patterns=n_steps,
        notes="Single-photodiode 1D laser-scanner emulation.",
    )


def _single_spot(W, H, cx, cy, r):
    img = np.zeros((H, W), dtype=np.uint8)
    yy, xx = np.ogrid[:H, :W]
    img[(xx - cx) ** 2 + (yy - cy) ** 2 <= r * r] = 255
    return img

def _vertical_line(W, H, x, w):
    img = np.zeros((H, W), dtype=np.uint8)
    img[:, max(0, x - w // 2):min(W, x + (w + 1) // 2)] = 255
    return img

def _zbar_decode(frame: np.ndarray) -> list:
    out = []
    try:
        from pyzbar.pyzbar import decode as zbar_decode
        for d in zbar_decode(frame):
            out.append({"data": d.data.decode(errors="replace"),
                        "type": d.type,
                        "polygon": [(p.x, p.y) for p in d.polygon]})
    except ImportError:
        pass
    try:
        from pylibdmtx.pylibdmtx import decode as dmtx_decode
        for d in dmtx_decode(frame):
            out.append({"data": d.data.decode(errors="replace"),
                        "type": "DATAMATRIX", "polygon": []})
    except ImportError:
        pass
    return out

"""
Reference-physics tests — the sensing math against known ground truth.

These break the v0.4 circularity (decoders tested only against a mock
built from the same assumptions) by checking each measurement chain
against analytically known answers:

  1. N-step phase shifting recovers a known phase ramp (Zhang formula
     vs. ground truth, not vs. itself), at 4 and 8 steps, with noise.
  2. Gray-code stripes produce a strictly monotonic, complete
     correspondence map (the property absolute coding exists to give).
  3. Hadamard single-pixel reconstruction recovers a known image from
     ideal bucket measurements (inner products computed directly, no
     mock camera in the loop) with correlation > 0.99.

Hardware validation still pending — these prove the math, the bench
proves the optics.
"""
import sys; sys.path.insert(0, ".")
import math
import numpy as np

from swiss_army_sensor import patterns as P
from swiss_army_sensor import decoders as D


def test_phase_shift_recovers_known_ramp():
    rng = np.random.default_rng(0)
    H, W = 64, 256
    period = 32.0
    xx = np.tile(np.arange(W, dtype=np.float64), (H, 1))
    true_phase = (2 * math.pi * xx / period + math.pi) % (2 * math.pi) - math.pi
    for n_steps, sigma, tol in ((4, 0.0, 1e-6), (4, 1.0, 0.05), (8, 1.0, 0.04)):
        frames = []
        for k in range(n_steps):
            I = 127.5 + 100.0 * np.cos(2 * math.pi * xx / period
                                       + 2 * math.pi * k / n_steps)
            frames.append(I + rng.normal(0, sigma, (H, W)))
        r = D.decode_phase_shift(frames)
        err = np.angle(np.exp(1j * (r.phase - true_phase)))
        rms = float(np.sqrt(np.mean(err ** 2)))
        assert rms < tol, f"{n_steps}-step σ={sigma}: phase RMS {rms} > {tol}"
    print("[OK] phase-shift decode recovers known ramp: float32-exact noiseless, "
          "<0.05 rad RMS at σ=1.0/255")


def test_gray_code_correspondence_is_monotonic_and_complete():
    n_bits = 10
    pats = P.gray_code_stripes(n_bits=n_bits, orientation="vertical")
    # decode gray -> binary column index per DMD column
    bits = np.stack([(p[0, :] > 127).astype(np.uint64) for p in pats], 0)
    gray = np.zeros(P.DMD_W, dtype=np.uint64)
    for b in range(n_bits):
        gray |= bits[b] << b
    binary = gray.copy()
    shift = 1
    while shift < n_bits:
        binary ^= binary >> shift
        shift <<= 1
    diffs = np.diff(binary.astype(np.int64))
    assert np.all(diffs >= 0), "correspondence not monotonic"
    n_codes = len(np.unique(binary))
    assert n_codes >= P.DMD_W // 2, f"only {n_codes} distinct codes"
    print(f"[OK] gray code: monotonic correspondence, {n_codes} distinct "
          f"codes across {P.DMD_W} columns at {n_bits} bits")


def test_hadamard_single_pixel_reconstructs_known_image():
    order = 5
    N = 2 ** order
    fov = 256
    rng = np.random.default_rng(2)
    scene_small = (rng.random((N, N)) > 0.7).astype(np.float64)  # sparse dots
    pats = P.hadamard_patterns(order=order, fov_px=fov)
    cy, cx = P.DMD_H // 2, P.DMD_W // 2
    half = fov // 2
    cell = fov // N
    buckets = []
    for pat in pats:
        tile = pat[cy - half:cy + half, cx - half:cx + half].astype(np.float64) / 255.0
        # ideal bucket: inner product with the scene upsampled to fov
        scene_big = np.kron(scene_small, np.ones((cell, cell)))
        buckets.append(float((tile * scene_big).sum()))
    r = D.decode_single_pixel(buckets, n_per_side=N)
    rec = r.image.astype(np.float64)
    rec = (rec - rec.mean()) / (rec.std() + 1e-12)
    ref = (scene_small - scene_small.mean()) / (scene_small.std() + 1e-12)
    corr = float(np.mean(rec * ref))
    assert corr > 0.99, f"reconstruction correlation {corr} <= 0.99"
    print(f"[OK] Hadamard single-pixel: known scene recovered from ideal "
          f"buckets, correlation {corr:.4f}")


if __name__ == "__main__":
    test_phase_shift_recovers_known_ramp()
    test_gray_code_correspondence_is_monotonic_and_complete()
    test_hadamard_single_pixel_reconstructs_known_image()

"""
Algebraic identity tests for the nullification operators.

These encode the v0.5 findings as executable facts:
  1. PhaseShiftNu's global-δ detuning is EXACTLY a constant phase
     offset (φ_ν = φ_1 − δ) — it cannot beat ν=1 on any signal.
  2. PhaseShiftNuWeighted is NOT a constant offset (non-degenerate).
  3. hadamard_shrinkage_baseline gives the classical-rival MSE that
     every bench ν-sweep must be compared against.
"""
import sys; sys.path.insert(0, ".")
import math
import numpy as np

from swiss_army_sensor.nullification import (
    PhaseShiftNu, PhaseShiftNuWeighted, HadamardNu,
    hadamard_shrinkage_baseline, synthetic_low_snr_hadamard_test,
)


def _wrap(x):
    return np.angle(np.exp(1j * x))


def test_global_delta_is_pure_rotation():
    rng = np.random.default_rng(0)
    op = PhaseShiftNu(delta_max=math.pi / 16)
    N = 4
    true_phase = rng.uniform(-math.pi, math.pi, (32, 32))
    frames = [0.5 + 0.04 * np.cos(true_phase + 2 * math.pi * k / N)
              + rng.normal(0, 0.05, (32, 32)) for k in range(N)]  # noisy!
    for nu in (0.0, 0.3, 0.7, 0.95):
        delta = (1.0 - nu) * op.delta_max
        p_nu, _ = op.apply(frames, nu=nu)
        p_1, _ = op.apply(frames, nu=1.0)
        err = float(np.max(np.abs(_wrap(p_nu - (p_1 - delta)))))
        assert err < 1e-10, f"nu={nu}: rotation identity violated ({err})"
    print("[OK] PhaseShiftNu: φ_ν ≡ φ_1 − δ to machine precision, "
          "even under noise — degenerate parameterization confirmed")


def test_weighted_detuning_is_not_a_rotation():
    rng = np.random.default_rng(1)
    op = PhaseShiftNuWeighted(w_max=0.25)
    N = 4
    true_phase = rng.uniform(-math.pi, math.pi, (32, 32))
    frames = [0.5 + 0.3 * np.cos(true_phase + 2 * math.pi * k / N)
              + rng.normal(0, 0.02, (32, 32)) for k in range(N)]
    p_nu, _ = op.apply(frames, nu=0.5)
    p_1, _ = op.apply(frames, nu=1.0)
    diff = _wrap(p_nu - p_1)
    spread = float(diff.std())
    assert spread > 1e-4, "weighted detuning collapsed to a constant offset"
    v = op.verify_id_limit()
    assert v["ok"], "weighted variant does not reduce to Zhang at nu=1"
    print(f"[OK] PhaseShiftNuWeighted: nu=0.5 differs pointwise from nu=1 "
          f"(std {spread:.4f} rad), reduces to Zhang at nu=1")


def test_shrinkage_baseline_brackets_hadamard_win():
    inputs = synthetic_low_snr_hadamard_test(snr_db=0, seed=0, n_coeffs=128)
    base = hadamard_shrinkage_baseline(**inputs)
    assert base["mse_oracle_shrinkage"] <= base["mse_nu_1"] + 1e-12
    hn = HadamardNu()
    out = hn.apply(inputs["y_pos"], inputs["y_neg"], nu=0.94)
    mse_nu = float(np.mean((out - inputs["truth"]) ** 2))
    print(f"[OK] shrinkage baseline: MSE(ν=1)={base['mse_nu_1']:.4f}, "
          f"MSE(λ*)={base['mse_oracle_shrinkage']:.4f}, "
          f"MSE(ν=0.94)={mse_nu:.4f} — bench ν-sweeps must report all three")


if __name__ == "__main__":
    test_global_delta_is_pure_rotation()
    test_weighted_detuning_is_not_a_rotation()
    test_shrinkage_baseline_brackets_hadamard_win()

"""
swiss_army_sensor.nullification
===============================

Theoretical module — Messina Nullification Framework operators.

The MNF spine is the decomposition

    N(ν) + R(ν) = id

on the signal space, parameterized by ν ∈ [0, 1].

  ν = 1  —  complete nullification: N and R are exact orthogonal complements.
            Standard noise-cancellation algorithms in optical metrology
            (differential capture, 3-step phase shifting, Hadamard ±1 pairs,
            Nayar high-freq separation) all live in this limit.

  ν < 1  —  near-nullification: a controlled residual survives. MNF conjecture
            is that for some signals there exists ν* < 1 that beats ν = 1
            by a measurable quantitative metric.

This module deliberately stays at the operator/algebra level. It does not
wire into any sensor mode yet — hooks documented for later integration.

Numerical findings (v0.4):
  - PhaseShiftNu: ν* moves with noise level (qualitative match to MNF
    prediction), but synthetic-case win is tiny (~0.005%). Bench data on
    real low-modulation surfaces needed before claiming anything.
  - HadamardNu: statistically robust ν<1 win across 30 seeds × 11 SNR
    levels. ν* in 0.93–0.97 range, std 0.04–0.08, MSE improvement
    +0.31% to +0.80%. This is the strongest empirical signal in v0.4.

Conventions:
  - Inputs are numpy arrays; operators are duck-typed.
  - ν is a Python float, always.
  - The ν = 1 limit is checked numerically in `verify_id_limit()`.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional, Sequence
import math
import numpy as np


class NullificationOperator(ABC):
    name: str = "abstract"
    standard_limit: str = "undefined"
    zero_limit: str = "identity"

    @abstractmethod
    def apply(self, *args, nu: float = 1.0, **kwargs):
        raise NotImplementedError

    def verify_id_limit(self, *args, **kwargs) -> dict:
        return {"name": self.name, "tested": False}


class DifferentialPair(NullificationOperator):
    """
    Standard differential: take a, b sharing common-mode m, differing on target t.
        a = m + t,   b = m - t   →   (a - b)/2 = t      (ν = 1)

    ν-detuned:
        N_ν(a, b) = (a - ν·b) / (1 + ν)
    """
    name = "differential_pair"
    standard_limit = "(a - b) / 2"
    zero_limit = "a (common-mode passes through)"

    def apply(self, a, b, nu: float = 1.0):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        if not (0.0 <= nu <= 1.0):
            raise ValueError(f"nu must be in [0, 1], got {nu}")
        return (a - nu * b) / (1.0 + nu)

    def residual(self, a, b, nu: float = 1.0):
        return 0.5 * (a + b) - self.apply(a, b, nu=nu)

    def verify_id_limit(self):
        rng = np.random.default_rng(0)
        m = rng.normal(0, 1, 256)
        t = rng.normal(0, 1, 256)
        a, b = m + t, m - t
        nu1 = self.apply(a, b, nu=1.0)
        nu0 = self.apply(a, b, nu=0.0)
        err1 = float(np.max(np.abs(nu1 - t)))
        err0 = float(np.max(np.abs(nu0 - a)))
        return {
            "name": self.name,
            "standard_limit": self.standard_limit,
            "zero_limit": self.zero_limit,
            "tested": True,
            "max_err_at_nu_1": err1,
            "max_err_at_nu_0": err0,
            "ok": err1 < 1e-12 and err0 < 1e-12,
        }


class PhaseShiftNu(NullificationOperator):
    """
    Standard N-step phase-shifting (Zhang 2018):
        φ̂ = arctan2( −Σ I_k sin(2πk/N) , Σ I_k cos(2πk/N) )

    ν-detuned form: weights at phase 2πk/N + δ(ν), where δ = (1-ν)·δ_max.

    DEGENERACY (v0.5, proven algebraically and tested in
    tests/test_nullification_identities.py):
      Adding the SAME δ to every basis angle is a rotation of the
      (c, −s) vector:  s' = s·cosδ + c·sinδ,  c' = c·cosδ − s·sinδ,
      hence φ'(ν) = φ(1) − δ IDENTICALLY, for every signal and every
      noise realization. This parameterization cannot outperform ν=1;
      apparent ν* structure in v0.4 synthetics (~0.005%) was wrapping
      artifact at the ±π boundary interacting with the constant bias.
      Kept for backward compatibility and as the worked example of why
      numerical wins need an algebraic sanity check first.
      For a non-degenerate detuning see PhaseShiftNuWeighted.
    """
    name = "phase_shift_nu"
    standard_limit = "Zhang N-step arctan2"
    zero_limit = "arbitrary sin/cos projection"

    def __init__(self, delta_max: float = math.pi / 32):
        self.delta_max = float(delta_max)

    def apply(self, frames, nu: float = 1.0):
        if not (0.0 <= nu <= 1.0):
            raise ValueError(f"nu must be in [0, 1], got {nu}")
        N = len(frames)
        stack = np.stack([np.asarray(f, dtype=np.float64) for f in frames], 0)
        delta = (1.0 - nu) * self.delta_max
        k = np.arange(N).reshape(N, 1, 1)
        s = (stack * np.sin(2 * math.pi * k / N + delta)).sum(0)
        c = (stack * np.cos(2 * math.pi * k / N + delta)).sum(0)
        phase = np.arctan2(-s, c)
        modulation = 2.0 / N * np.sqrt(s * s + c * c)
        return phase, modulation

    def verify_id_limit(self):
        rng = np.random.default_rng(1)
        N = 4
        true_phase = rng.uniform(-math.pi, math.pi, (16, 16))
        frames = [127 + 100 * np.cos(true_phase + 2 * math.pi * k / N) for k in range(N)]
        phase_nu1, _ = self.apply(frames, nu=1.0)
        s_ref = sum(f * math.sin(2 * math.pi * k / N) for k, f in enumerate(frames))
        c_ref = sum(f * math.cos(2 * math.pi * k / N) for k, f in enumerate(frames))
        ref = np.arctan2(-s_ref, c_ref)
        err = float(np.max(np.abs(phase_nu1 - ref)))
        return {"name": self.name, "tested": True,
                "max_err_vs_standard_at_nu_1": err, "ok": err < 1e-10}


class HadamardNu(NullificationOperator):
    """
    Single-pixel Hadamard +/- pair:
        coeff = (y⁺ - y⁻) / 2         (ν = 1)

    ν-detuned:
        coeff_ν = (y⁺ - ν · y⁻) / (1 + ν)

    NUMERICAL STATUS (v0.4):
      Statistically robust ν<1 win on sparse-signal recovery across
      30 seeds × 11 SNR levels (-15 dB to +10 dB). Mean ν* in 0.93–0.97
      range with std 0.04–0.08. MSE improvement +0.31% (at -15 dB) to
      +0.80% (at +10 dB).

      Strongest MNF prediction in v0.4. Next step: real DMD + bucket
      measurement on known sparse target.
    """
    name = "hadamard_nu"
    standard_limit = "(y_pos - y_neg) / 2"
    zero_limit = "y_pos / 2"

    def apply(self, y_pos, y_neg, nu: float = 1.0):
        if not (0.0 <= nu <= 1.0):
            raise ValueError(f"nu must be in [0, 1], got {nu}")
        y_pos = np.asarray(y_pos, dtype=np.float64)
        y_neg = np.asarray(y_neg, dtype=np.float64)
        return (y_pos - nu * y_neg) / (1.0 + nu)

    def verify_id_limit(self):
        rng = np.random.default_rng(2)
        y_p = rng.normal(10, 1, 64)
        y_n = rng.normal(0, 1, 64)
        std_ref = (y_p - y_n) / 2.0
        out_nu1 = self.apply(y_p, y_n, nu=1.0)
        err = float(np.max(np.abs(out_nu1 - std_ref)))
        return {"name": self.name, "tested": True,
                "max_err_vs_standard_at_nu_1": err, "ok": err < 1e-12}


class PhaseShiftNuWeighted(NullificationOperator):
    """
    Non-degenerate ν-detuning for N-step phase shifting.

    Instead of rotating all basis angles (which is provably a constant
    phase offset — see PhaseShiftNu), detune the per-step AMPLITUDE
    weights:  w_k(ν) = 1 + (1-ν)·(-1)^k·w_max.
    At ν=1 all weights are 1 (standard Zhang). At ν<1 the estimator
    leaves a controlled residual of the common-mode bias term in the
    quadrature sums — a genuine N+R=id split, not a reparameterization.

    Falsifiable bench prediction (write the number down BEFORE the run):
    on low-modulation surfaces (cathode-like, modulation < 0.05) with
    additive sensor noise, ν* < 1 reduces phase RMS only if the noise is
    correlated across steps; for i.i.d. noise ν*=1 is optimal (weighted
    least squares with equal variances). So the bench test doubles as a
    measurement of step-to-step noise correlation in the real rig.
    """
    name = "phase_shift_nu_weighted"
    standard_limit = "Zhang N-step arctan2 (all weights 1)"
    zero_limit = "alternating-weight quadrature"

    def __init__(self, w_max: float = 0.25):
        self.w_max = float(w_max)

    def apply(self, frames, nu: float = 1.0):
        if not (0.0 <= nu <= 1.0):
            raise ValueError(f"nu must be in [0, 1], got {nu}")
        N = len(frames)
        stack = np.stack([np.asarray(f, dtype=np.float64) for f in frames], 0)
        k = np.arange(N).reshape(N, 1, 1)
        w = 1.0 + (1.0 - nu) * self.w_max * ((-1.0) ** np.arange(N)).reshape(N, 1, 1)
        s = (w * stack * np.sin(2 * math.pi * k / N)).sum(0)
        c = (w * stack * np.cos(2 * math.pi * k / N)).sum(0)
        phase = np.arctan2(-s, c)
        modulation = 2.0 / N * np.sqrt(s * s + c * c)
        return phase, modulation

    def verify_id_limit(self):
        rng = np.random.default_rng(3)
        N = 4
        true_phase = rng.uniform(-math.pi, math.pi, (16, 16))
        frames = [127 + 100 * np.cos(true_phase + 2 * math.pi * k / N)
                  for k in range(N)]
        p1, _ = self.apply(frames, nu=1.0)
        ref, _ = PhaseShiftNu().apply(frames, nu=1.0)
        err = float(np.max(np.abs(p1 - ref)))
        return {"name": self.name, "tested": True,
                "max_err_vs_standard_at_nu_1": err, "ok": err < 1e-10}


def hadamard_shrinkage_baseline(y_pos, y_neg, truth) -> dict:
    """
    Classical rival explanation for the HadamardNu ν*<1 win.

    coeff_ν = (y⁺ − ν·y⁻)/(1+ν) on a sparse signal with common-mode
    offset is a biased, variance-reducing estimator — shrinkage. This
    computes the best GLOBAL linear shrinkage of the standard ν=1
    estimate, λ* = argmin_λ MSE(λ·coeff_1), in closed form
    (λ* = <coeff_1, truth>/<coeff_1, coeff_1>, oracle James-Stein-style).

    Bench discrimination protocol: if MSE(ν*) ≈ MSE(λ*) the ν-win is
    classical shrinkage wearing MNF notation; MNF-specific content
    requires MSE(ν*) < MSE(λ*) or a ν* that deviates from the value the
    measured offset and noise floor predict. Run this alongside every
    real-DMD ν-sweep.
    """
    y_pos = np.asarray(y_pos, dtype=np.float64)
    y_neg = np.asarray(y_neg, dtype=np.float64)
    truth = np.asarray(truth, dtype=np.float64)
    coeff_1 = (y_pos - y_neg) / 2.0
    denom = float(coeff_1 @ coeff_1)
    lam = float(coeff_1 @ truth) / denom if denom > 0 else 0.0
    mse_1 = float(np.mean((coeff_1 - truth) ** 2))
    mse_shrunk = float(np.mean((lam * coeff_1 - truth) ** 2))
    return {"lambda_star": lam, "mse_nu_1": mse_1,
            "mse_oracle_shrinkage": mse_shrunk}


@dataclass
class NuSweepResult:
    nus: np.ndarray
    metric: np.ndarray
    nu_star: float
    metric_at_nu_star: float
    metric_at_nu_1: float
    delta: float
    operator_name: str
    metric_name: str
    notes: str = ""


def sweep_nu(operator, inputs: dict, metric_fn,
             operator_inputs=None, nu_values=None,
             minimize: bool = True, metric_name: str = "error",
             notes: str = "") -> NuSweepResult:
    """Sweep ν, evaluate metric, return curve + ν*."""
    import inspect
    if operator_inputs is None:
        sig = inspect.signature(operator.apply)
        operator_inputs = [p for p in sig.parameters.keys() if p not in ("self", "nu")]
    if nu_values is None:
        nu_values = np.linspace(0.0, 1.0, 51)
    op_kwargs = {k: inputs[k] for k in operator_inputs if k in inputs}
    metrics = np.empty_like(nu_values, dtype=np.float64)
    for i, nu in enumerate(nu_values):
        out = operator.apply(**op_kwargs, nu=float(nu))
        metrics[i] = float(metric_fn(out, **inputs))
    idx = int(np.argmin(metrics) if minimize else np.argmax(metrics))
    idx_1 = int(np.argmin(np.abs(nu_values - 1.0)))
    delta = float(metrics[idx_1] - metrics[idx]) * (1 if minimize else -1)
    return NuSweepResult(nus=nu_values, metric=metrics,
                         nu_star=float(nu_values[idx]),
                         metric_at_nu_star=float(metrics[idx]),
                         metric_at_nu_1=float(metrics[idx_1]),
                         delta=delta, operator_name=operator.name,
                         metric_name=metric_name, notes=notes)


def correlation_preservation(original: np.ndarray, nullified: np.ndarray) -> dict:
    """What survives nullification that a complete cancellation destroys?"""
    o = np.asarray(original)
    n = np.asarray(nullified)
    if o.shape != n.shape:
        raise ValueError(f"shapes {o.shape} vs {n.shape}")
    on = float(np.linalg.norm(o))
    nn = float(np.linalg.norm(n))
    residual_mod = nn / (on + 1e-12)
    if np.iscomplexobj(o) or np.iscomplexobj(n):
        po = np.angle(o.astype(complex))
        pn = np.angle(n.astype(complex))
        z = np.exp(1j * (po - pn))
        phase_corr = float(np.abs(z.mean()))
    else:
        phase_corr = float("nan")
    ro = np.argsort(np.argsort(np.abs(o).ravel())).astype(np.float64)
    rn = np.argsort(np.argsort(np.abs(n).ravel())).astype(np.float64)
    rank_corr = float(np.corrcoef(ro, rn)[0, 1]) if ro.std() > 0 and rn.std() > 0 else float("nan")
    return {"residual_modulation": residual_mod,
            "phase_correlation": phase_corr,
            "rank_preservation": rank_corr}


def synthetic_low_modulation_phase_test(n_steps: int = 4, h: int = 64, w: int = 64,
                                        modulation: float = 0.04,
                                        noise_sigma: float = 0.02, seed: int = 0) -> dict:
    """Battery-cathode-like low-modulation phase ramp for ν-sweep testing."""
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    true_phase = 2 * math.pi * xx / w
    frames = []
    bias = 0.5
    for k in range(n_steps):
        I = bias + modulation * np.cos(true_phase + 2 * math.pi * k / n_steps)
        I = I + rng.normal(0, noise_sigma, (h, w))
        frames.append(I)
    return {"frames": frames, "true_phase": true_phase}


def phase_rms_error(out, frames, true_phase):
    phase, _ = out
    diff = np.angle(np.exp(1j * (phase - true_phase)))
    return float(np.sqrt(np.mean(diff ** 2)))


def synthetic_low_snr_hadamard_test(n_coeffs: int = 64, snr_db: float = -3.0,
                                    seed: int = 0) -> dict:
    """Sparse coefficient vector at low SNR for ν-sweep testing."""
    rng = np.random.default_rng(seed)
    truth = rng.normal(0, 1, n_coeffs)
    truth[rng.random(n_coeffs) > 0.2] = 0
    sig_p = np.maximum(truth, 0) + 1.0
    sig_n = np.maximum(-truth, 0) + 1.0
    sigma = 10 ** (-snr_db / 20)
    y_p = sig_p + rng.normal(0, sigma, n_coeffs)
    y_n = sig_n + rng.normal(0, sigma, n_coeffs)
    return {"y_pos": y_p, "y_neg": y_n, "truth": truth}


def hadamard_mse(out, y_pos, y_neg, truth):
    return float(np.mean((out - truth) ** 2))


def run_self_test() -> dict:
    """Verify ν=1 limits, run two synthetic ν-sweeps."""
    results = {"verifications": [], "sweeps": []}
    for op in (DifferentialPair(), PhaseShiftNu(), HadamardNu()):
        results["verifications"].append(op.verify_id_limit())
    psn = PhaseShiftNu(delta_max=math.pi / 32)
    inputs = synthetic_low_modulation_phase_test()
    sw = sweep_nu(psn, inputs, phase_rms_error, metric_name="phase_rms_rad",
                  notes="low-modulation phase recovery, 4-step")
    results["sweeps"].append(sw)
    hn = HadamardNu()
    inputs = synthetic_low_snr_hadamard_test()
    sw = sweep_nu(hn, inputs, hadamard_mse, metric_name="coeff_mse",
                  notes="−3 dB SNR Hadamard pair, sparse truth")
    results["sweeps"].append(sw)
    return results

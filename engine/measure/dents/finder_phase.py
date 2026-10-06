"""
Finder (c): HDR phase-residual depth z (fringe reflection / triangulation).

Each sin0 / sin90 phase-shift set is reduced to one HDR phasor per camera
pixel: the long pass where none of the set's frames saturate, else the
short pass mapped to long-pass DN by an affine fit of the two projector-on
floods. Frames of a pass that fail a per-frame QC (the long frame must be
an affine copy of the short one) are replaced by the other pass's copy.

On a smooth cylinder the fringe phase is a smooth function of the pixel,
so a local quadratic carrier (from wrapped phase gradients, no unwrapping)
predicts each pixel's phasor from a RING of neighbours (2.6..9 px); the
residual psi is a band-pass of the projector-coordinate displacement. A
dent pushed in by h along the normal moves the seen point along the CAMERA
RAY by h / cos(view): psi_s = a_s h with a_s from the stereo calibration's
projector shift per mm along the ray (pattern units X = col, Y = row / 2).
Per pixel, weighted least squares over the sets gives h (mm, + = in) and a
nuisance shift q perpendicular to the epipolar direction (no geometry makes
it; glint / blur phase errors do). Streaks along the axis are removed by a
running median along the image columns, a sigma-1 matched filter follows,
and h is divided by a robust local noise measured in a window elongated
along the axis (so each azimuth column has its own noise):
  score = z_h - 0.5 max(0, |z_q| - 2).

Deviations from the development phase finder:
  - sets: sin0 and sin90 only (thetas), also for subset "all"; the
    development finder used all nine phase-shift sets of the development
    capture (4 sinusoid orientations x 2 periods + the lines_v set).
  - frame QC region: the cylinder silhouette (hit) dilated by 12 px, not
    the zone's camera mask (engine dent sets need not have one); the
    typical residual is the median over the pairs of the USED sets only,
    not over every frame of the capture.
  - short -> long gain fitted on the projector-ON floods (the development
    finder fitted pattern 0 = flood 255, which on the inverted bench is
    the ambient flood: little signal to fit).
On the bench good cell these change the frames marked bad and T_phase
(4.64 here vs 3.55 with the development finder's bad-frame sets), which
moves pose 2's small dent between 0.99 and 1.04 x the null: that dent's
margin is within the QC choice.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter, label, median_filter

from ._util import Quiet, nanmedian_axis

PARAMS = dict(
    sat=245.0, r_in=2.6, r_out=9.0, carrier_sigma=4.0, min_mod_snr=4.0, max_clip_frac=0.5,
    axis_median=31, match_sigma=1.0, noise_sig_x=4.0, noise_sig_y=24.0, edge_margin=4,
    q_penalty=0.5, zone_grow=12, thetas=(0, 90),
)


# ------------------------------------------------------------ sets / HDR

def phase_sets(data, thetas=(0, 90)) -> list:
    """Sinusoid sets present in BOTH passes: (name, theta, T, long idx, short idx, delta)."""
    out = []
    L, S = data.long, data.short

    def rows(P, th, T):
        return sorted((p["step"], i, p["steps"]) for i, p in enumerate(P.specs)
                      if p["kind"] == "sinus" and p["theta_deg"] == th and p["period"] == T)

    sets = sorted({(p["theta_deg"], p["period"]) for p in L.specs if p["kind"] == "sinus"})
    for th, T in sets:
        if th not in thetas:
            continue
        rl, rs = rows(L, th, T), rows(S, th, T)
        if len(rl) < 3 or [r[0] for r in rl] != [r[0] for r in rs]:
            continue
        out.append(dict(name=f"sin{th:g}_T{T:g}", theta=float(th), T=float(T),
                        idx_l=[i for _, i, _ in rl], idx_s=[i for _, i, _ in rs],
                        delta=np.array([2 * np.pi * k / n for k, _, n in rl])))
    return out


def exposure_map(data, region, sat):
    """Affine short -> long DN from the projector-on floods (unsaturated pixels)."""
    L, S = np.asarray(data.long.on, float), np.asarray(data.short.on, float)
    ok = region & (L < sat) & (S > 5)
    if ok.sum() < 100:
        return data.ratio, 0.0
    a, b = np.polyfit(S[ok], L[ok], 1)
    return float(a), float(b)


def frame_qc(data, sets, region, sat):
    """Per-frame glitch QC: long ~ a short + b per frame; frames whose affine residual
    is > 2.5x typical (or whose exposure ratio is off by > 1.5 %) are bad in the pass
    whose frame mean deviates more from the rest of its set."""
    bad = {"L": set(), "S": set()}
    pairs = [(il, is_) for st in sets for il, is_ in zip(st["idx_l"], st["idx_s"])]
    Lf, Sf = data.long.frames, data.short.frames
    rms, rat = {}, {}
    for il, is_ in pairs:
        Li, Si = np.asarray(Lf[il], float), np.asarray(Sf[is_], float)
        ok = region & (Li < sat) & (Si > 3)
        if ok.sum() < 500:
            continue
        A = np.vstack([Si[ok], np.ones(ok.sum())]).T
        c, *_ = np.linalg.lstsq(A, Li[ok], rcond=None)
        r = Li[ok] - A @ c
        rms[(il, is_)] = float(np.sqrt(np.mean(r * r)))
        ok2 = ok & (Si > 12)
        if ok2.sum() > 200:
            rat[(il, is_)] = float(np.median(Li[ok2] / Si[ok2]))
    if not rms:
        return bad
    med = np.median(list(rms.values()))
    medr = np.median(list(rat.values())) if rat else None
    for st in sets:
        ok = region & (np.asarray(Lf[st["idx_l"]]).max(0) < sat) \
            & (np.asarray(Sf[st["idx_s"]]).min(0) > 3)
        if ok.sum() < 500:
            continue
        mL = np.array([np.mean(np.asarray(Lf[i])[ok]) for i in st["idx_l"]])
        mS = np.array([np.mean(np.asarray(Sf[i])[ok]) for i in st["idx_s"]])
        dL, dS = 100 * (mL / np.median(mL) - 1), 100 * (mS / np.median(mS) - 1)
        for j, (il, is_) in enumerate(zip(st["idx_l"], st["idx_s"])):
            g = rms.get((il, is_), 0) > 2.5 * med
            if medr is not None and (il, is_) in rat:
                g |= abs(rat[(il, is_)] / medr - 1) > 0.015
            if g:
                if abs(dL[j]) >= abs(dS[j]):
                    bad["L"].add(int(il))
                else:
                    bad["S"].add(int(is_))
    return bad


def set_phasor(data, st, bad, sat, gain, offset):
    """HDR phasor of one set in long-pass DN -> (Z, sigma_phi, n clipped short frames)."""
    n = len(st["idx_l"])
    e = np.exp(1j * st["delta"])
    l = np.asarray(data.long.frames[st["idx_l"]], float)
    s = np.asarray(data.short.frames[st["idx_s"]], float)
    sL = s * gain + offset
    bl = np.array([i in bad["L"] for i in st["idx_l"]])
    bs = np.array([i in bad["S"] for i in st["idx_s"]])
    if bl.all():
        useL = np.zeros(l.shape[1:], bool)
    else:
        useL = l[~bl].max(0) < sat
        if bl.any():
            useL &= sL[bl].max(0) < sat
    stack = np.where(useL[None], l, sL)
    if bl.any():
        stack[bl] = np.where(useL[None], sL[bl], stack[bl])
    if bs.any():
        stack[bs] = np.where(l[bs] < sat, l[bs], stack[bs])
    Z = np.tensordot(e, stack, axes=(0, 0)) * 2 / n
    nclip = np.where(useL, 0, (s >= sat).sum(0))
    fromL = np.where(useL[None], ~bl[:, None, None], bs[:, None, None] & (l < sat))
    var = np.where(fromL, 1.0 + 0.05 * l, gain ** 2 * (0.3 + 0.05 * s)).mean(0)
    sig_phi = np.sqrt(var * 2.0 / n) / np.maximum(np.abs(Z), 1e-6)
    sig_phi = sig_phi * (1.0 + 3.0 * nclip / n)
    return Z, sig_phi, nclip


# ------------------------------------------------------------ ring residual

def _mwgauss(a, w, s):
    return gaussian_filter(a * w, s) / np.maximum(gaussian_filter(w, s), 1e-12)


def ring_residual(Z, w, valid, r_in=2.6, r_out=9.0, carrier_sigma=4.0):
    """psi = angle(u conj(pred)), pred = ring average of neighbours rotated by a local
    quadratic carrier (wrapped-gradient estimate). Also the ring coverage."""
    u = Z / np.maximum(np.abs(Z), 1e-12)
    gx = np.zeros(u.shape); gy = np.zeros(u.shape)
    gx[:, 1:-1] = np.angle(u[:, 2:] * np.conj(u[:, :-2])) / 2
    gy[1:-1, :] = np.angle(u[2:, :] * np.conj(u[:-2, :])) / 2
    vv = valid.copy(); vv[:, [0, -1]] = False; vv[[0, -1], :] = False
    vx = vv.copy(); vx[:, 1:-1] &= valid[:, 2:] & valid[:, :-2]
    vy = vv.copy(); vy[1:-1, :] &= valid[2:, :] & valid[:-2, :]
    kx = _mwgauss(median_filter(gx, 5), vx * w, carrier_sigma)
    ky = _mwgauss(median_filter(gy, 5), vy * w, carrier_sigma)
    kxx = np.gradient(kx, axis=1); kyy = np.gradient(ky, axis=0)
    kxy = 0.5 * (np.gradient(kx, axis=0) + np.gradient(ky, axis=1))
    H, W = u.shape
    Rr = int(np.ceil(r_out))
    ww = w * valid
    up = np.pad(u * ww, Rr); wp = np.pad(ww, Rr)
    acc = np.zeros(u.shape, complex); wacc = np.zeros(u.shape); wful = 0.0
    sg = r_out / 2.0
    for dy in range(-Rr, Rr + 1):
        for dx in range(-Rr, Rr + 1):
            r = np.hypot(dx, dy)
            if r < r_in or r > r_out:
                continue
            g = np.exp(-r * r / (2 * sg * sg))
            sl = (slice(Rr + dy, Rr + dy + H), slice(Rr + dx, Rr + dx + W))
            th = kx * dx + ky * dy + 0.5 * (kxx * dx * dx + 2 * kxy * dx * dy + kyy * dy * dy)
            acc += g * up[sl] * np.exp(-1j * th)
            wacc += g * (wp[sl] > 0)
            wful += g
    psi = np.angle(u * np.conj(acc))
    return psi, wacc / wful


def axis_running_median(a, valid, win):
    """Running median along image y (NaN-aware)."""
    if win <= 1:
        return np.zeros_like(a)
    h = win // 2
    b = np.where(valid, a, np.nan)
    bp = np.pad(b, ((h, h), (0, 0)), constant_values=np.nan)
    sw = np.lib.stride_tricks.sliding_window_view(bp, win, axis=0)
    with Quiet():
        med = nanmedian_axis(sw, -1)
    return np.where(np.isfinite(med), med, 0.0)


def robust_local_sigma(a, valid, sx, sy, clip=3.5):
    """Clipped RMS in an anisotropic Gaussian window (two passes)."""
    v = a[valid]
    sig_g = 1.4826 * np.median(np.abs(v)) + 1e-12 if v.size else 1.0
    s = np.full(a.shape, sig_g)
    for _ in range(2):
        c = np.minimum(a * a, (clip * s) ** 2) * valid
        num = gaussian_filter(c, (sy, sx)); den = gaussian_filter(valid.astype(float), (sy, sx))
        s = np.sqrt(num / np.maximum(den, 1e-9)) * 1.12
        s = np.maximum(s, 0.5 * sig_g)
    return s, sig_g


# ------------------------------------------------------------ finder

def score_map(data, geo: dict, params: Optional[dict] = None) -> tuple:
    """-> (score map in crop coords, info). h (mm) is kept for the depth fallback."""
    p = dict(PARAMS)
    p.update(params or {})
    H, W = data.shape
    face0 = geo["face"]
    sets = phase_sets(data, p["thetas"])
    if not sets:
        raise ValueError("phase finder: no sin0 / sin90 set in both passes")
    region = binary_dilation(geo["hit"], iterations=int(p["zone_grow"]))
    gain, offset = exposure_map(data, region, p["sat"])
    bad = frame_qc(data, sets, region, p["sat"])
    cosv = np.nan_to_num(geo["cosv"], nan=1.0)
    sens = geo["sens_ray"] / np.maximum(cosv, 0.2)[..., None]      # per mm of inward normal push
    vX = np.nan_to_num(sens[..., 0])
    vY = np.nan_to_num(sens[..., 1]) / 2.0                         # pattern units: Y = row / 2
    face = face0.copy()
    lab, nl = label(face)
    if nl > 1:
        sizes = np.bincount(lab.ravel()); sizes[0] = 0
        face = lab == np.argmax(sizes)
    ys_, xs_ = np.nonzero(face)
    mg = int(np.ceil(p["r_out"])) + 12
    by0, by1 = max(ys_.min() - mg, 0), min(ys_.max() + mg + 1, H)
    bx0, bx1 = max(xs_.min() - mg, 0), min(xs_.max() + mg + 1, W)
    bb = (slice(by0, by1), slice(bx0, bx1))
    Hb, Wb = by1 - by0, bx1 - bx0
    zone = region[bb]
    fb = face[bb]
    vXb, vYb = vX[bb], vY[bb]
    vn = np.sqrt(vXb ** 2 + vYb ** 2) + 1e-12
    pX, pY = -vYb / vn, vXb / vn
    N11 = np.zeros((Hb, Wb)); N12 = np.zeros((Hb, Wb)); N22 = np.zeros((Hb, Wb))
    b1 = np.zeros((Hb, Wb)); b2 = np.zeros((Hb, Wb))
    used = []
    for st in sets:
        Z, sphi, nclip = set_phasor(data, st, bad, p["sat"], gain, offset)
        Z, sphi, nclip = Z[bb], sphi[bb], nclip[bb]
        valid = zone & (sphi < 1.0 / p["min_mod_snr"])
        valid &= ~(nclip > p["max_clip_frac"] * len(st["idx_l"]))
        w = 1.0 / (sphi ** 2 + 0.01 ** 2)
        psi, cover = ring_residual(Z, w, valid, p["r_in"], p["r_out"], p["carrier_sigma"])
        valid &= cover > 0.55
        th = np.radians(st["theta"])
        k = 2 * np.pi / st["T"]
        a_s = k * (np.cos(th) * vXb + np.sin(th) * vYb)
        b_s = k * (np.cos(th) * pX + np.sin(th) * pY) * vn
        fv = valid & fb
        if fv.sum() < 200:
            continue
        psi = np.where(fv, psi - axis_running_median(psi, fv, int(p["axis_median"])), 0.0)
        sl, _ = robust_local_sigma(psi, fv, p["noise_sig_x"], p["noise_sig_y"])
        wz = fv / (sl ** 2)
        N11 += wz * a_s * a_s; N12 += wz * a_s * b_s; N22 += wz * b_s * b_s
        b1 += wz * a_s * psi; b2 += wz * b_s * psi
        used.append(st["name"])
    if not used:
        raise ValueError("phase finder: no set has enough valid pixels on the face")
    det = N11 * N22 - N12 ** 2
    ok = (det > 1e-9 * np.maximum(N11 * N22, 1e-30)) & (N11 > 0)
    with Quiet():
        h = np.where(ok, (N22 * b1 - N12 * b2) / np.where(ok, det, 1), 0.0)
        q = np.where(ok, (N11 * b2 - N12 * b1) / np.where(ok, det, 1), 0.0)
        var_h = np.where(ok, N22 / np.where(ok, det, 1), np.inf)
        var_q = np.where(ok, N11 / np.where(ok, det, 1), np.inf)
    face_ok = fb & ok & np.isfinite(var_h)
    face_ok = binary_erosion(face_ok, iterations=int(p["edge_margin"])) & face_ok
    if face_ok.sum() < 50:
        raise ValueError("phase finder: the eroded face is empty")
    h_ax = h - axis_running_median(h, face_ok, int(p["axis_median"]))
    q_ax = q - axis_running_median(q, face_ok, int(p["axis_median"]))
    with Quiet():
        wv = face_ok / np.where(face_ok, var_h, 1)
        hs = _mwgauss(h_ax, wv, p["match_sigma"]); hs[~face_ok] = 0
        qs = _mwgauss(q_ax, face_ok / np.where(face_ok, var_q, 1), p["match_sigma"])
        qs[~face_ok] = 0
    sh, sgh = robust_local_sigma(hs, face_ok, p["noise_sig_x"], p["noise_sig_y"])
    sq, _ = robust_local_sigma(qs, face_ok, p["noise_sig_x"], p["noise_sig_y"])
    zh = np.where(face_ok, hs / sh, 0.0)
    zq = np.where(face_ok, qs / sq, 0.0)
    score = zh - p["q_penalty"] * np.maximum(0.0, np.abs(zq) - 2.0)
    score[~face_ok] = 0.0
    h_depth = _mwgauss(h_ax, wv, 0.6)
    h_depth[~face_ok] = np.nan
    out = np.full((H, W), np.nan, np.float32)
    out[bb] = score
    out[~face0] = np.nan
    hd = np.full((H, W), np.nan, np.float32)
    hd[bb] = h_depth
    info = {"sets": used, "exposure_gain": round(gain, 4),
            "bad_frames": {k: sorted(v) for k, v in bad.items()},
            "sigma_h_mm": round(float(sgh), 4), "h_mm": hd}
    return out, info

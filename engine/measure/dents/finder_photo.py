"""
Finder (b): photometric - reflectance / glint anomaly.

A dent tilts the local surface normal, so shading, the specular lobe, the
fringe modulation and the direct / global split all change locally. On a
clean cylinder every channel is smooth along each rect column (axis
vertical), so a dent is a small residual of an along-axis model.

Channels (log, crop coords, then rectified):
  projector-on flood of the SHORT pass, the mid-grey (L128) flood of the
  short pass, the ambient (projector-dark) flood of the LONG pass; per pass
  the sinusoid offset (mean of all its sinusoid frames) and each sinusoid
  set's modulation ratio B / (A - ambient); per pass with line sets the
  direct (max - min) and global (min - ambient) light.
Saturated flood / offset samples are CENSORED: kept as lower bounds, they
only count as positive evidence (a glint on a dent). Ratios are invalid
where an input saturates.

Per channel: along-axis model (running median, window clamped at the ends,
+ Gaussian), SHIFT-TOLERANT residual (what a <= 0.75 px lateral move of the
column profile explains is not an anomaly), local robust scale (31 x 3)
inflated by a glint-jitter prior. Combined by a robust Mahalanobis
distance (trimmed covariance shrunk 50 % toward identity), median polish
(rows flagged as structure - rim, crimp, bench reflection - are dropped to
the cell end), a dent-scale blob filter (sigma 1.5 and 2.5 px, FWHM ~1.2 and
~2 mm at 0.35 mm / px: the smallest dent of interest is ~1 mm, while glitter
specks of the surface texture are 1-2 px; limited by the 3rd-largest value
of each 3 x 3 so single glitter pixels cannot peak), standardised against
the face -> z.

Deviations from the development photometric finder:
  - blob sigma 1.5 / 2.5 px instead of 1 / 2 px. The coarser pair lifts the
    dents relative to the good cell's largest speck (which sets this
    finder's null); it was chosen while scoring the bench's pose 2 small
    dent, so pose 2 is tuned-on data for it (fusion.DEV_PARAMS: 1 / 2).
  - flood channels: the development finder used all three floods (0, 255,
    128) of BOTH passes; here the projector-on flood of the short pass, the mid-grey
    flood of the short pass and the ambient flood of the long pass only
    (the three dropped - long-pass on and mid-grey, short-pass ambient -
    repeat these at the other exposure; not re-validated on the bench sets).
  - no dot-pattern channel (dots are not captured by capture_dent_set).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import ndimage
from scipy.stats import chi2

from ._util import SAT, Quiet, Rect, nanmedian_axis

PARAMS = dict(win=21, smooth=2.0, tol=0.75, scale_rows=31, scale_cols=3, floor_frac=0.25,
              shrink=0.5, zclip=40.0, jitter=0.5, blob_sigma=(1.5, 2.5), rank_k=3,
              struct_z=6.0)


# ------------------------------------------------------------ channels

def build_channels(data) -> dict:
    """name -> (image float32, valid bool, censored bool), crop coords."""
    ch = {}
    L, S = data.long, data.short

    def add_log(name, img, sat):
        ch[name] = (np.log(np.asarray(img, float) + 2.0).astype(np.float32),
                    np.ones(img.shape, bool), np.asarray(sat, bool))

    add_log("S_on", S.on, S.on >= SAT)
    if S.mid is not None:
        add_log("S_mid", S.mid, S.mid >= SAT)
    add_log("L_amb", L.off, L.off >= SAT)
    for tag, P in (("L", L), ("S", S)):
        A = P.frames
        amb = np.asarray(P.off, float)
        sin_all = P.idx(kind="sinus")
        if sin_all:
            st = np.asarray(A[sin_all], float)
            ch[f"{tag}_sinA"] = (np.log(st.mean(0) + 2.0).astype(np.float32),
                                 np.ones(amb.shape, bool), (st >= SAT).any(0))
            sets = sorted({(p["theta_deg"], p["period"]) for p in P.specs if p["kind"] == "sinus"})
            for th, T in sets:
                rows = sorted((p["step"], i, p["steps"]) for i, p in enumerate(P.specs)
                              if p["kind"] == "sinus" and p["theta_deg"] == th and p["period"] == T)
                st = np.asarray(A[[i for _, i, _ in rows]], float)
                ph = np.array([2 * np.pi * k / n for k, _, n in rows])
                n = len(rows)
                c = np.tensordot(np.cos(ph), st, 1) * 2 / n
                s = np.tensordot(np.sin(ph), st, 1) * 2 / n
                off = st.mean(0)
                ok = ~(st >= SAT).any(0)
                ratio = np.hypot(c, s) / np.maximum(off - amb, 3.0)
                ch[f"{tag}_m{th:g}_{T:g}"] = (ratio.astype(np.float32), ok,
                                              np.zeros(ok.shape, bool))
        for kind, lab in (("lines_v", "V"), ("lines_h", "H")):
            ii = P.idx(kind=kind)
            if len(ii) < 3:
                continue
            st = np.asarray(A[ii], float)
            mx, mn = st.max(0), st.min(0)
            ok = ~(st >= SAT).any(0)
            z = np.zeros(ok.shape, bool)
            direct = np.log(np.maximum(mx - mn, 0) + 2.0)
            glob = np.log(np.maximum(mn - amb, 0) + 2.0)
            ch[f"{tag}_dir{lab}"] = (direct.astype(np.float32), ok, z)
            ch[f"{tag}_glo{lab}"] = (glob.astype(np.float32), ok, z)
    return ch


# ------------------------------------------------------------ along-axis model

def _nanmedfilt_cols(Rm, valid, win):
    """Running median along axis 0 ignoring invalid samples; the window is clamped
    inside the array at the ends (always `win` samples)."""
    n, m = Rm.shape
    win = min(win, n)
    h = win // 2
    X = np.where(valid, Rm, np.nan)
    Wv = sliding_window_view(X, win, axis=0)
    with Quiet():
        med_c = nanmedian_axis(Wv, 2)
    start = np.clip(np.arange(n) - h, 0, n - win)
    return med_c[start]


def along_axis_model(Rm, valid, win=25, smooth=2.0):
    med = _nanmedfilt_cols(Rm, valid, win)
    ok = np.isfinite(med)
    medf = np.where(ok, med, 0.0)
    w = ndimage.gaussian_filter1d(ok.astype(float), smooth, axis=0)
    M = ndimage.gaussian_filter1d(medf, smooth, axis=0) / np.maximum(w, 1e-6)
    M[w < 0.2] = np.nan
    return M


def shift_tolerant_residual(Rm, M, tol=0.75):
    """Distance of Rm from the interval spanned by M at lateral shifts in [-tol, tol]."""
    Ml = np.roll(M, 1, axis=1); Ml[:, 0] = M[:, 0]
    Mr = np.roll(M, -1, axis=1); Mr[:, -1] = M[:, -1]
    lo_l = M + tol * (Ml - M); lo_r = M + tol * (Mr - M)
    with Quiet():
        lo = np.minimum(np.minimum(M, lo_l), lo_r)
        hi = np.maximum(np.maximum(M, lo_l), lo_r)
        return np.where(Rm > hi, Rm - hi, np.where(Rm < lo, Rm - lo, 0.0))


def local_scale(r, valid, win_rows=41, win_cols=3, floor=1e-3):
    a = np.where(valid & np.isfinite(r), np.abs(r), np.nan)
    hr, hc = win_rows // 2, win_cols // 2
    pad = np.pad(a, ((hr, hr), (hc, hc)), mode="constant", constant_values=np.nan)
    Wv = sliding_window_view(pad, (win_rows, win_cols))
    with Quiet():
        s = 1.4826 * nanmedian_axis(Wv.reshape(*a.shape, -1), 2)
        s = np.where(np.isfinite(s), s, np.nanmedian(s))
    s = np.nan_to_num(s, nan=floor)
    s = ndimage.uniform_filter(s, (9, 1))
    return np.maximum(s, floor)


def channel_z(Rm, valid, cens, p):
    M = along_axis_model(Rm, valid, p["win"], p["smooth"])
    r = shift_tolerant_residual(Rm, M, p["tol"])
    r_plain = Rm - M
    okr = valid & np.isfinite(r)
    s = local_scale(r_plain, okr & ~cens, p["scale_rows"], p["scale_cols"])
    with Quiet():
        gfloor = p["floor_frac"] * np.nanmedian(s[okr & ~cens]) if (okr & ~cens).any() else 1e-3
    s = np.maximum(s, gfloor)
    if p["jitter"] > 0:
        # a lateral wobble of `jitter` px of the column profile is expected noise where
        # the profile is steep (glint-line edges)
        Mf = np.where(np.isfinite(M), M, 0.0)
        gl = np.abs(np.diff(Mf, axis=1, prepend=Mf[:, :1]))
        gr = np.abs(np.diff(Mf, axis=1, append=Mf[:, -1:]))
        g = ndimage.maximum_filter(np.maximum(gl, gr), size=(1, 3))
        s = np.sqrt(s * s + (p["jitter"] * g) ** 2)
    r = np.where(cens, np.maximum(r, 0.0), r)
    with Quiet():
        z = np.where(okr, r / s, 0.0)
    return np.nan_to_num(z).astype(np.float32)


# ------------------------------------------------------------ combining

def robust_cov(Z, iters=3, shrink=0.5):
    keep = np.ones(len(Z), bool)
    k = Z.shape[1]
    mu = np.median(Z, 0)
    Ci = np.eye(k)
    for _ in range(iters):
        mu = np.median(Z[keep], 0)
        C = np.atleast_2d(np.cov((Z[keep] - mu).T))
        Cs = (1 - shrink) * C + shrink * np.eye(k) * max(float(np.mean(np.diag(C))), 1e-9)
        Ci = np.linalg.inv(Cs)
        d2 = np.einsum("ij,jk,ik->i", Z - mu, Ci, Z - mu)
        keep = d2 < chi2.ppf(0.975, k)
        if keep.sum() < k + 5:
            break
    return mu, Ci


def blob_filter(raw, sigma=1.0, rank_k=3):
    """Gaussian mean limited by the rank_k-th largest value of each 3 x 3."""
    sm = ndimage.gaussian_filter(raw, sigma)
    if rank_k and rank_k > 1:
        rk = ndimage.rank_filter(raw, rank=9 - rank_k, size=3)
        rk = ndimage.gaussian_filter(rk, 0.5)
        sm = np.minimum(sm, rk)
    return sm


def median_polish(raw, face, struct_z=6.0, iters=2, guard=4):
    """Remove row / column medians over the face; rows that stand out as a whole are
    structure (rim, crimp, bench reflection) and are dropped to the cell end."""
    Rr = np.where(face, raw, np.nan)
    rowm = np.zeros(Rr.shape[0]); colm = np.zeros(Rr.shape[1])
    with Quiet():
        for _ in range(iters):
            rr = np.nanmedian(Rr - colm[None, :], 1); rowm = np.where(np.isfinite(rr), rr, 0.0)
            cc = np.nanmedian(Rr - rowm[:, None], 0); colm = np.where(np.isfinite(cc), cc, 0.0)
        rp = np.nanpercentile(Rr - colm[None, :], 75, axis=1)
    ok = np.isfinite(rp)
    struct = np.zeros(Rr.shape[0], bool)
    if ok.sum() > 5:
        med = np.median(rp[ok])
        mad = 1.4826 * np.median(np.abs(rp[ok] - med)) + 1e-6
        struct[ok] = (rp[ok] - med) / mad > struct_z
    fr = np.nonzero(face.any(1))[0]
    if len(fr) and struct.any():
        n = len(fr)
        lo_end, hi_end = fr[0] + n // 4, fr[-1] - n // 4
        s_idx = np.nonzero(struct)[0]
        bot = s_idx[s_idx >= hi_end]
        top = s_idx[s_idx <= lo_end]
        if len(bot):
            struct[bot.min():] = True
        if len(top):
            struct[:top.max() + 1] = True
    struct = ndimage.binary_dilation(struct, iterations=guard)
    return raw - rowm[:, None] - colm[None, :], struct


# ------------------------------------------------------------ finder

def score_map(data, geo: dict, R: Rect, params: Optional[dict] = None) -> tuple:
    """-> (z map in crop coords, info)."""
    p = dict(PARAMS)
    p.update(params or {})
    chans = build_channels(data)
    names = sorted(chans)
    face_r = R.fwd_bool(geo["face"])
    cell_r = R.fwd_bool(geo["hit"])
    Zs = []
    for n in names:
        img, ok, cen = chans[n]
        Rm = R.fwd(img)
        V = (np.nan_to_num(R.fwd(ok.astype(np.float32)), nan=0.0) > 0.999) & cell_r \
            & np.isfinite(Rm)
        Cn = np.nan_to_num(R.fwd(cen.astype(np.float32)), nan=0.0) > 0.001
        Zs.append(channel_z(np.nan_to_num(Rm), V, Cn, p))
    Z = np.stack(Zs, -1)
    k = Z.shape[-1]
    Zc = np.clip(Z, -p["zclip"], p["zclip"])
    samp = Zc[face_r]
    mu, Ci = robust_cov(samp, shrink=p["shrink"])
    d2 = np.einsum("...j,jk,...k->...", Zc - mu, Ci, Zc - mu)
    raw = np.sqrt(np.maximum(d2, 0) / k)
    raw = np.where(np.isfinite(raw), raw, 0.0)
    raw, struct_rows = median_polish(raw, face_r, p["struct_z"])
    face_use = face_r & ~struct_rows[:, None]
    zr = None
    for sgm in p["blob_sigma"]:
        sm = blob_filter(raw, sgm, p["rank_k"])
        bg = sm[face_use]
        if bg.size < 20:
            continue
        med = np.median(bg)
        mad = 1.4826 * np.median(np.abs(bg - med))
        mad = max(mad, (np.percentile(bg, 90) - med) / 1.2816, 1e-6)
        z_s = (sm - med) / mad
        zr = z_s if zr is None else np.maximum(zr, z_s)
    if zr is None:
        raise ValueError("photometric finder: the face is empty after removing structure rows")
    zr = np.where(face_use, zr, np.nan)
    out = R.back(zr, data.shape)
    out[~geo["face"]] = np.nan
    info = {"channels": names, "structure_rows": int(struct_rows.sum())}
    return out.astype(np.float32), info

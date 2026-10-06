"""
Finder (a): pattern warp - local distortion of the projected phase patterns.

Every phase-stepped family (sinusoids, single-mirror line sets) gives per
camera pixel a first-harmonic phasor whose phase says which DMD point
lights the surface there. It is demodulated with a smooth CARRIER (the
metric scan's decoded correspondence, robust quadratic along the axis per
rect column) and compared with an ALONG-AXIS robust background: in the
rect (axis vertical) every pixel of a column sees the same surface normal
on a perfect cylinder, so the background is an IRLS local-linear fit of the
same column over an annulus 6..20 rows above / below. What is local along
the axis survives:
  phase residual (warp of each family), the 2-D warp (du, dv) by weighted
  least squares over all families and both exposures -> its epipolar part
  is a pattern depth (per-pixel projector shift per mm along the camera
  ray, from the stereo calibration), its perpendicular part a non-
  geometric warp; fringe-contrast residual; family inconsistency; ambient
  and projector-flood log residuals.
Each channel -> local along-axis robust z, then a row-wise robust
normalisation (horizontal structures: rim, horizon, zone edge), weighted
Stouffer sum, robust z over the face. Matched filter sigma 1 px.
Two-exposure AND rule: the long and short passes are fused on their own
and a pixel keeps min(pooled, min(long, short)) (0.75 x when only one pass
is valid there, e.g. a glint core saturated in the long pass).

Deviation from the development pattern-warp finder: no zigzag or
checker families (capture_dent_set records neither; subset "all" of a
development capture drops them too).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ._util import SAT, Quiet, Rect, nan_gauss, nanmedian_axis, robust_global_norm, shift_stack

PARAMS = dict(
    ann_in=6, ann_out=20,       # along-axis annulus of the background (rows)
    smooth=1.0,                 # matched filter sigma (px)
    zclip=6.0, zcap=12.0,
    min_mod=2.0,                # phasor modulation floor (DN)
    min_fam_depth=6,
    w_depth=1.5, w_phase=1.0, w_contrast=0.5, w_perp=0.25, w_incons=0.25,
    w_amb=0.75, w_flood=0.5,
)


# ------------------------------------------------------------ families

def families(specs: list) -> dict:
    """Phase-stepped families of one pass: name -> idx, step phases d, ku, kv (cycles per
    DMD col / row), off, sinus."""
    fam = {}
    sets = sorted({(p["theta_deg"], p["period"]) for p in specs if p["kind"] == "sinus"})
    for th, T in sets:
        rows = sorted((p["step"], i, p["steps"]) for i, p in enumerate(specs)
                      if p["kind"] == "sinus" and p["theta_deg"] == th and p["period"] == T)
        if len(rows) < 3:
            continue
        t = np.radians(th)
        fam[f"sin{th:g}_{T:g}"] = dict(idx=[i for _, i, _ in rows],
                                       d=np.array([2 * np.pi * k / n for k, _, n in rows]),
                                       ku=np.cos(t) / T, kv=np.sin(t) / (2 * T), off=0.0,
                                       sinus=True)
    lv = sorted((p["shift"], i, p.get("pitch", 10)) for i, p in enumerate(specs)
                if p["kind"] == "lines_v")
    if len(lv) >= 3:
        pitch = lv[0][2]
        fam["lines_v"] = dict(idx=[i for _, i, _ in lv],
                              d=np.array([2 * np.pi * k / pitch for k, _, _ in lv]),
                              ku=1.0 / pitch, kv=0.0, off=0.0, sinus=False)
    lh = sorted((p["shift_rows"], i, p.get("pitch_rows", 20)) for i, p in enumerate(specs)
                if p["kind"] == "lines_h")
    if len(lh) >= 3:
        pitch = lh[0][2]
        fam["lines_h"] = dict(idx=[i for _, i, _ in lh],
                              d=np.array([2 * np.pi * k / pitch for k, _, _ in lh]),
                              ku=0.0, kv=1.0 / pitch, off=-0.5 / pitch, sinus=False)
    return fam


def carrier_phase(f: dict, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    return 2 * np.pi * (u * f["ku"] + v * f["kv"] + f["off"])


def phasor(F: np.ndarray, f: dict, inverted: bool, sat: float = SAT, min_frames: int = 6,
           fit_sat: float = 240.0):
    """First-harmonic phasor of a phase-stepped set -> (Z complex, valid). Sinusoid sets:
    near-saturated frames are dropped and the sinusoid least-squares fitted on the rest."""
    S = np.asarray(F[f["idx"]], np.float64)
    d = f["d"]
    N = len(d)
    satm = S >= sat
    Z = np.tensordot(np.exp(1j * d), S, axes=(0, 0)) * 2 / N
    valid = satm.sum(0) == 0
    satm2 = S >= fit_sat
    nsat2 = satm2.sum(0)
    if f["sinus"] and N >= 5:
        codes = (satm2 * (1 << np.arange(N))[:, None, None]).sum(0)
        for code in np.unique(codes[(nsat2 > 0) & (nsat2 <= N - min_frames)]):
            keep = np.array([(int(code) >> k) & 1 == 0 for k in range(N)])
            M = np.stack([np.ones(keep.sum()), np.cos(d[keep]), np.sin(d[keep])], 1)
            Pm = np.linalg.pinv(M)
            pix = codes == code
            coef = Pm @ S[keep][:, pix]
            Z[pix] = coef[1] + 1j * coef[2]
            valid[pix] = True
        valid &= nsat2 <= N - min_frames
    return (-Z if inverted else Z), valid


# ------------------------------------------------------------ along-axis statistics

def along_axis_bg(Wc: np.ndarray, face: np.ndarray, ann=(6, 20), iters: int = 3) -> np.ndarray:
    """Robust along-axis background (complex or real): IRLS local-linear fit in the row
    offset over the annulus ann_in <= |j| <= ann_out of the same rect column, face only."""
    ai, ao = ann
    SH = np.array([s for s in range(-ao, ao + 1) if abs(s) >= ai], float)
    A = np.array(Wc, copy=True)
    A[~face] = np.nan
    St = shift_stack(A, SH.astype(int))
    valid = np.isfinite(St)
    St = np.where(valid, St, 0)
    j = SH[:, None, None]
    w = valid.astype(float)
    a = b = None
    with Quiet():
        for it in range(iters + 1):
            S0 = w.sum(0)
            S1 = (w * j).sum(0)
            S2 = (w * j * j).sum(0)
            T0 = (w * St).sum(0)
            T1 = (w * j * St).sum(0)
            den = S0 * S2 - S1 ** 2
            lin = den > 1e-5 * np.maximum(S0, 1) ** 2
            a = np.where(lin, (S2 * T0 - S1 * T1) / np.where(lin, den, 1),
                         T0 / np.maximum(S0, 1e-9))
            b = np.where(lin, (S0 * T1 - S1 * T0) / np.where(lin, den, 1), 0)
            if it == iters:
                break
            e = np.where(valid, np.abs(St - a - b * j), np.nan)
            s = 1.4826 * nanmedian_axis(e, 0) + 1e-9
            w = np.nan_to_num(valid / (1 + (e / (2.5 * s)) ** 2))
    return np.where(valid.sum(0) >= 8, a, np.nan)


def col_sigma(r, face, floor):
    with Quiet():
        rr = np.where(face, r, np.nan)
        med = nanmedian_axis(rr, 0)
        s = 1.4826 * nanmedian_axis(np.abs(rr - med), 0)
        n = np.isfinite(rr).sum(0)
    s = np.where(n < 25, np.nan, s)
    return med, np.maximum(s, floor)


def col_z(r, face, floor, center=True):
    med, s = col_sigma(r, face, floor)
    return ((r - med) if center else r) / s


def _local_z(E, face, ann=(6, 30), dcol=1, floor_frac=0.5):
    """Robust z of E against a local along-axis annulus (same column +-dcol)."""
    A = np.where(face, E, np.nan)
    SH = [s for s in range(-ann[1], ann[1] + 1) if abs(s) >= ann[0]]
    stacks = []
    for dc in range(-dcol, dcol + 1):
        Ac = np.full_like(A, np.nan)
        if dc >= 0:
            Ac[:, :A.shape[1] - dc] = A[:, dc:]
        else:
            Ac[:, -dc:] = A[:, :A.shape[1] + dc]
        stacks.append(shift_stack(Ac, SH))
    St = np.concatenate(stacks, 0)
    with Quiet():
        med = nanmedian_axis(St, 0)
        mad = 1.4826 * nanmedian_axis(np.abs(St - med[None]), 0)
        n = np.isfinite(St).sum(0)
        gmad = np.nanmedian(mad[face])
    mad = np.maximum(mad, floor_frac * gmad)
    z = (E - med) / mad
    z[n < 20] = np.nan
    return z


def _two_way(E, face):
    z = _local_z(E, face)
    with Quiet():
        zf = np.where(face, z, np.nan)
        rmed = nanmedian_axis(zf, 1)
        rmad = 1.4826 * nanmedian_axis(np.abs(zf - rmed[:, None]), 1)
        rn = np.isfinite(zf).sum(1)
        rmed = np.where(rn >= 15, rmed, 0.0)
        rmad = np.where(rn >= 15, rmad, 1.0)
        z = (z - rmed[:, None]) / np.maximum(np.nan_to_num(rmad, nan=1.0), 1.0)[:, None]
    return z


def _colfit(a, face, deg=2, min_n=40):
    """Robust per-column polynomial along the axis (the carrier only has to be smooth)."""
    H, W = a.shape
    yy = np.arange(H, dtype=float)
    out = np.full(a.shape, np.nan)
    good = np.zeros(W, bool)
    for j in range(W):
        rows = np.nonzero(face[:, j])[0]
        v = a[rows, j]
        ok = np.isfinite(v)
        if ok.sum() < min_n:
            continue
        t = yy[rows][ok]
        vv = v[ok].astype(float)
        w = np.ones_like(vv)
        pp = None
        for _ in range(4):
            pp = np.polyfit(t, vv, deg, w=w)
            r = vv - np.polyval(pp, t)
            s = 1.4826 * np.median(np.abs(r)) + 1e-3
            w = 1 / np.maximum(1, np.abs(r) / (2.5 * s))
        out[:, j] = np.polyval(pp, yy)
        good[j] = True
    gj = np.nonzero(good)[0]
    if len(gj) == 0:
        raise ValueError("pattern warp: no metric correspondence on the face (carrier)")
    for i in range(H):
        out[i] = np.interp(np.arange(W), gj, out[i, gj])
    return out


def _lut(r0, Phi, ok, nb=32):
    """Circular-mean residual vs carrier phase -> correction (rad): removes the harmonic
    aliasing of binary patterns and residual carrier non-linearity."""
    ph = np.mod(np.nan_to_num(Phi), 2 * np.pi)
    b = np.clip((ph / (2 * np.pi) * nb).astype(int), 0, nb - 1)
    good = ok & np.isfinite(r0)
    corr = np.zeros(nb)
    e = np.exp(1j * r0[good])
    bb = b[good]
    sr = np.bincount(bb, e.real, nb)
    si = np.bincount(bb, e.imag, nb)
    cnt = np.bincount(bb, None, nb)
    has = cnt >= 15
    corr[has] = np.angle(sr[has] + 1j * si[has])
    return np.where(np.isfinite(Phi), corr[b], 0.0)


# ------------------------------------------------------------ stage 1: fields

def compute_fields(data, R: Rect, face, ucar, vcar, p) -> dict:
    fields = {}
    ann = (p["ann_in"], p["ann_out"])
    for pn, P in data.passes.items():
        flood = R.fwd(np.maximum(P.on - P.off, 1.0))
        for fn, f in families(P.specs).items():
            Z, valid = phasor(P.frames, f, P.inverted)
            Zr = R.fwd(Z.real) + 1j * R.fwd(Z.imag)
            vr = R.fwd_bool(valid)
            with Quiet():
                mod = np.abs(Zr)
                Phi = carrier_phase(f, ucar, vcar)
                Wd = Zr * np.exp(-1j * Phi)
                ok = vr & np.isfinite(Wd) & (mod > p["min_mod"])
            Wd = np.where(ok, Wd, np.nan)
            bg = along_axis_bg(Wd, face, ann)
            with Quiet():
                r0 = np.angle(Wd * np.conj(bg))
            Wd = Wd * np.exp(-1j * _lut(r0, Phi, face & ok))
            bg = along_axis_bg(Wd, face, ann)
            with Quiet():
                r = np.angle(Wd * np.conj(bg))
                lc = np.log(np.maximum(np.abs(Wd), 1e-3) / flood)
            m = lc - along_axis_bg(lc, face, ann)
            with Quiet():
                med = np.nanmedian(np.where(face, r, np.nan), 0)
            r = r - np.nan_to_num(med)[None, :]
            fields[(pn, fn)] = dict(r=r.astype(np.float32), m=m.astype(np.float32), ok=ok,
                                    au=2 * np.pi * f["ku"], av=2 * np.pi * f["kv"])
    return fields


def photometric_fields(data, R: Rect, face) -> dict:
    """Along-axis log residuals of the ambient flood (projector dark: the mirrored room)
    and of the projector flood (on - off), per pass, as column robust z."""
    out = {}
    for pn, P in data.passes.items():
        sat = (P.on >= SAT) | (P.off >= SAT)
        for nm, im in (("amb", P.off), ("flood", P.on - P.off)):
            with Quiet():
                L = np.log(np.maximum(R.fwd(np.where(sat, np.nan, im)), 1.0))
            res = L - along_axis_bg(L, face, (4, 16))
            out[(pn, nm)] = col_z(nan_gauss(res, 0.8), face, 0.01)
    return out


# ------------------------------------------------------------ stage 2: channels

def build_channels(fields, face, p, sens, photo=None, pass_filter=None) -> dict:
    H, W = face.shape
    chans = {}
    Ep = np.zeros((H, W)); Np = np.zeros((H, W)); Em = np.zeros((H, W)); Nm = np.zeros((H, W))
    A11 = np.zeros((H, W)); A12 = np.zeros((H, W)); A22 = np.zeros((H, W))
    B1 = np.zeros((H, W)); B2 = np.zeros((H, W))
    D1 = np.zeros((H, W)); D2 = np.zeros((H, W))
    a_mm, b_mm = sens                                   # DMD col / row per mm along the ray
    rs = []
    for key, d in fields.items():
        if pass_filter and not pass_filter(key[0]):
            continue
        _, sig = col_sigma(d["r"], face, 0.01)
        with Quiet():
            zr = np.clip(d["r"] / sig, -p["zclip"], p["zclip"])
            zm = np.clip(col_z(d["m"], face, 0.01), -p["zclip"], p["zclip"])
        e = nan_gauss(zr ** 2, p["smooth"])
        em = nan_gauss(zm ** 2, p["smooth"])
        ok = np.isfinite(e); Ep[ok] += e[ok]; Np[ok] += 1
        ok = np.isfinite(em); Em[ok] += em[ok]; Nm[ok] += 1
        sg = np.where(np.isfinite(sig), sig, np.inf)[None, :]
        with Quiet():
            rr = np.clip(d["r"], -p["zclip"] * sg, p["zclip"] * sg)
            w = np.where(np.isfinite(rr), 1.0 / sg ** 2, 0.0)
        rr = np.nan_to_num(rr)
        au, av = d["au"], d["av"]
        A11 += w * au * au; A12 += w * au * av; A22 += w * av * av
        B1 += w * au * rr; B2 += w * av * rr
        sdep = au * a_mm + av * b_mm                    # rad per mm of depth
        D1 += np.nan_to_num(w * sdep * rr); D2 += np.nan_to_num(w * sdep * sdep)
        rs.append((w, au, av, rr))
    with Quiet():
        chans["phase"] = dict(E=np.where(Np >= 3, Ep / np.maximum(Np, 1), np.nan), w=p["w_phase"])
        chans["contrast"] = dict(E=np.where(Nm >= 3, Em / np.maximum(Nm, 1), np.nan),
                                 w=p["w_contrast"])
        det = A11 * A22 - A12 ** 2
        okd = (det > 1e-6 * np.maximum(A11 * A22, 1e-30)) & (A11 > 0)
        du = np.where(okd, (A22 * B1 - A12 * B2) / np.where(okd, det, 1), np.nan)
        dv = np.where(okd, (A11 * B2 - A12 * B1) / np.where(okd, det, 1), np.nan)
        chi = np.zeros((H, W)); nn = np.zeros((H, W))
        for w, au, av, rr in rs:
            e = rr - au * np.nan_to_num(du) - av * np.nan_to_num(dv)
            chi += w * e * e; nn += w > 0
        chi = np.where(nn >= 4, chi / np.maximum(nn - 2, 1), np.nan)
        depth = np.where(D2 > 0, D1 / np.maximum(D2, 1e-30), np.nan)     # mm, + = deeper
        nfam = sum((w > 0).astype(int) for w, _, _, _ in rs) if rs else np.zeros((H, W), int)
        need = min(p["min_fam_depth"], max(len(rs) - 1, 1))
        depth = np.where(nfam >= need, depth, np.nan)
        perp = (-du * b_mm + dv * a_mm) / np.hypot(a_mm, b_mm)
    dep_s = nan_gauss(depth, p["smooth"])
    perp_s = nan_gauss(perp, p["smooth"])
    chans["depth"] = dict(E=dep_s, w=p["w_depth"], depth_raw=depth)
    zp = col_z(perp_s, face, 1e-4)
    chans["perp"] = dict(E=np.clip(zp, -2 * p["zclip"], 2 * p["zclip"]) ** 2, w=p["w_perp"])
    with Quiet():
        chans["incons"] = dict(E=nan_gauss(np.log(np.maximum(chi, 1e-6)), p["smooth"]),
                               w=p["w_incons"])
    if photo is not None:
        for nm in ("amb", "flood"):
            zs = [v for (pn, k), v in photo.items()
                  if k == nm and (not pass_filter or pass_filter(pn))]
            if not zs:
                continue
            if len(zs) == 2:
                za, zb = zs
                both = np.isfinite(za) & np.isfinite(zb)
                agree = np.sign(za) == np.sign(zb)
                with Quiet():
                    E = np.where(both, np.where(agree, np.fmin(np.abs(za), np.abs(zb)), 0.0),
                                 0.5 * np.abs(np.where(np.isfinite(za), za, zb)))
            else:
                E = np.abs(zs[0])
            chans[nm] = dict(E=nan_gauss(E, 0.7), w=p[f"w_{nm}"])
    return chans


def fuse(chans: dict, face: np.ndarray, p: dict) -> np.ndarray:
    H, W = face.shape
    num = np.zeros((H, W)); have = np.zeros((H, W)); den = 0.0
    for nm, c in chans.items():
        if c["w"] <= 0:
            continue
        zc = _two_way(c["E"], face)
        c["z"] = zc
        ok = np.isfinite(zc)
        num[ok] += c["w"] * np.clip(zc[ok], -2, p["zcap"])
        have[ok] += c["w"]
        den += c["w"] ** 2                  # a missing channel is no evidence, not a free pass
    S = np.where((have > 0) & face, num / np.sqrt(max(den, 1e-12)), np.nan)
    return robust_global_norm(S, face)[0]


# ------------------------------------------------------------ finder

def score_map(data, geo: dict, R: Rect, params: Optional[dict] = None) -> tuple:
    """-> (score map in crop coords, info). Needs the metric correspondence (carrier)."""
    p = dict(PARAMS)
    p.update(params or {})
    face = R.fwd_bool(geo["face"])
    cx = geo.get("corr_x")
    cy = geo.get("corr_y")
    if cx is None or cy is None or not np.isfinite(cx[geo["face"]]).any():
        raise ValueError("pattern warp needs the metric scan's decoded correspondence (field.npz)")
    ucar = _colfit(R.fwd(cx), face)
    vcar = _colfit(R.fwd(cy), face)
    sens = (R.fwd(geo["sens_ray"][..., 0]), R.fwd(geo["sens_ray"][..., 1]))
    fields = compute_fields(data, R, face, ucar, vcar, p)
    photo = photometric_fields(data, R, face)
    chans = build_channels(fields, face, p, sens, photo=photo)
    pooled = fuse(chans, face, p)
    per = {}
    for tag in data.passes:
        ch = build_channels(fields, face, p, sens, photo=photo,
                            pass_filter=lambda pn, t=tag: pn == t)
        per[tag] = fuse(ch, face, p)
    Sz = pooled
    if len(per) == 2:
        a, b = per["short"], per["long"]
        both = np.isfinite(a) & np.isfinite(b)
        single = np.where(np.isfinite(a), a, b) * 0.75
        with Quiet():
            cons = np.where(both, np.fmin(a, b), single)
            Sz = np.where(face, np.fmin(pooled, cons), np.nan)
        Sz = robust_global_norm(Sz, face)[0]
    out = R.back(Sz, data.shape)
    out[~geo["face"]] = np.nan
    info = {"families": sorted({k[1] for k in fields}), "n_fields": len(fields),
            "depth_rect": chans["depth"]["depth_raw"]}
    return out.astype(np.float32), info


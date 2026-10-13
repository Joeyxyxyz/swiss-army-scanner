"""
MF-CFAR end-face dent detector with the reflectance FIR pre-step (the offline research
method 'mf_fir' of 2026-09-28, productised; research/endface_detect.py + efreflect.py).

What limits detection on the bench is a pixel-scale height texture (correlation ~0.4 mm,
0.03-0.046 mm RMS after high-pass) that is ~90 % REPEATABLE within a placement (averaging
repeats barely helps) and does NOT carry over between placements (so a 2-D golden or a
rotation / shift registration cannot remove it). 8-20 % of it is predictable from the flood
image (the phase decode is pulled toward the brighter side where the reflectance changes
inside the blur spot): the FIR removes that part.

  pixels   per face camera pixel (crop px iy, ix) and per repeat: u, v, h in the face frame
           (mm, face_profile) + the flood pair (255 / 0) of the targeted scan (camera DN)
  gridder  per-pixel repeat mean of u, v; ONE linear (Delaunay, barycentric) operator onto the
           face grid, triangles with an edge > max_edge_mm dropped (undecoded holes); each
           repeat gridded through it (its missing pixels filled with the pixel mean)
  FIR      7x7 camera-px kernel on the high-passed (4 px Gaussian) log flood (on - off),
           ridge least squares on GOOD placements (r < fir r_max, unsaturated px), predicting
           the high-passed per-repeat-plane-removed height: the predicted reflectance-induced
           height error (mm) is subtracted from every repeat's pixel heights before gridding.
           The face mask's border (edge_px = 2 camera px) is left out of the log flood: there
           the flood shows the face's outline (the flat ends, the rim is dark), not its
           reflectance, and on a small face that edge lies within the kernel of the inspected
           zone (a 16 mm face: the FIR 'explained' 16 % of pure noise and made rim triggers).
           SATURATED flood pixels are left out of the fit, and a placement whose flood
           saturates more than max_sat_pct (2 %) of its face is not corrected at all: the
           clipped reflectance under-predicts the error there, and those patches then stand out
           against the CFAR noise of the corrected rest (synthetic, a strong FIR: max good z
           <= 4.0 up to 0.6 % saturated, 5.4 at 3 %, 6-7 (false triggers) at 6-13 %; the
           research: "the flood must not be saturated")
  map      per repeat a robust plane (piston + tilt, r <= plane_r_mm) removed, then the mean
           (repeats drift 0.05-0.1 mm in piston between scans)
  golden   RADIAL profile (0.25 mm rings, median) of each good placement, median over
           placements (the cell's rotation about its axis is arbitrary)
  dev      map - golden radial profile, plane re-fitted
  MF       per dent diameter w (scales_mm): local least squares at every grid point,
           h(q) = a + b u + c v + depth * bowl_w(q) over |q| <= support_factor * w / 2,
           bowl_w = -(1 - (|q| / (w / 2))^2)+ (unit-depth paraboloid dent) -> depth (mm, +
           = a dent) and its geometric standard-error factor
  CFAR     z = (depth / factor) / sigma, sigma = 1.4826 MAD of THIS face's own map per radial
           band (dents are sparse; the MAD ignores them)
  trigger  a connected region with z >= z_thresh inside r <= inspect_r_mm (the grazing,
           pose-dependent chamfer beyond is not inspected) whose depth_centre_mm (depth.py:
           the 3 x 3 grid-cell average at its deepest place against the local background) is
           at least min_depth_mm; reported with r, theta, u, v, depth_centre_mm (= depth_mm,
           THE depth), depth_min_mm (single point, over-reads), width_half_depth_mm,
           depth_fit_mm / width_mm (the matched filter's paraboloid: its amplitude and the
           best diameter of the bank), z, area

Radii: inspect_r_mm, fit_r_mm, rim_r_mm, plane_r_mm and noise_bands_mm are fractions of the
face radius unless given (radii.py: 0.926 R, 0.926 R, 0.855 R, 0.855 R, bands scaled - on a
21.05 mm face they stay 9.75 / 9.75 / 9.0 / 9.0 mm); mf_params(over, radius_mm) resolves them.

Research evaluation (4 good placements of 2 cells, leave-one-placement-out, synthetic dents
injected per camera pixel): 0 false triggers at every z_thresh >= 4.5 (band method: 19);
at 5.5 about 1e-3 false triggers per good face extrapolated; ~50 % at 0.15 x 1.5 mm, ~90 %
at 0.20 x 2 mm / 0.15 x 3 mm; 1 mm wide dents are below the camera's resolution
(0.38 mm/px). Plain numpy / OpenCV / scipy; mm on the face grid of face_profile.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import depth as depth_mod
from . import radii as radii_mod

MF_DEFAULTS = {
    "scales_mm": (1.0, 1.5, 2.0, 3.0),       # dent diameters of the matched-filter bank
    "support_factor": 2.0,                   # fit window radius = support_factor x w / 2
    "noise_bands_mm": None,                  # None = scaled to the face (radii.py; 21.05 mm:
                                             # 0, 3, 6, 8, 9, 9.75, 10.25)
    "z_thresh": 5.5,                         # ~1e-3 false / good face; 5.0 ~1e-2 (+10 pts)
    "rim_z_thresh": None,                    # separate threshold for r > rim_r_mm (None = z)
    "min_depth_mm": 0.0,                     # ignore significant dents whose depth_centre_mm
                                             # is shallower than this
    "inspect_r_mm": None,                    # inspected zone r <= this (chamfer beyond);
                                             # None = 0.926 x the face radius (21.05 mm: 9.75)
    "fit_r_mm": None,                        # data the local fits use; None = inspect_r_mm
    "rim_r_mm": None,                        # 'rim' beyond this (still inspected); None =
                                             # 0.855 x the face radius (21.05 mm: 9.0)
    "sign": "dent",                          # 'dent' (depressions) | 'both' (+ bumps)
    "min_area_mm2": 0.0,
    "plane_r_mm": None,                      # robust plane region; None = rim_r_mm
    "max_edge_mm": 1.2,                      # gridding: drop triangles bridging holes
    "cover_frac": 0.6,                       # a depth needs this much of the dent footprint
}
MF_RADII = ("inspect_r_mm", "fit_r_mm", "rim_r_mm", "plane_r_mm", "noise_bands_mm")
FIR_DEFAULTS = {
    "half_px": 3,                            # 7 x 7 camera px kernel
    "hp_sigma_px": 4.0,                      # high-pass of the log flood / the height
    "ridge": 0.1,
    "r_max_mm": None,                        # fit on face points inside this radius; None =
                                             # 0.974 x inspect_r_mm (21.05 mm: 9.5)
    "edge_px": 2,                            # the face mask's border pixels are left out of
                                             # the log flood: there the flood shows the
                                             # face's OUTLINE (the flat ends, the rim is
                                             # dark), not its reflectance - on a small face
                                             # that edge sits within the kernel of the
                                             # inspected zone and the FIR fitted it
    "sat_dn": 254.5,                         # flood pixels at / above: saturated (no FIR input)
    "max_sat_pct": 2.0,                      # a placement saturated over more of its face is
                                             # checked without the FIR (see fir_usable)
}
BATTERY_DEPTHS_MM = (0.10, 0.15, 0.20, 0.30)
BATTERY_WIDTHS_MM = (1.0, 1.5, 2.0, 3.0)
BATTERY_TOL_MM = 0.75                         # detected: within w / 2 + this of the centre
BATTERY_R_MM = 9.0                            # random dent centres r <= this on a 21.05 mm
                                              # face (elsewhere: the face's rim_r_mm)
Z_REPORT = (4.5, 5.0, 5.5, 6.0)


def check_mf_keys(over: Optional[dict]) -> None:
    bad = set(over or {}) - set(MF_DEFAULTS)
    if bad:
        raise ValueError(f"unknown MF-CFAR params {sorted(bad)} (known: {sorted(MF_DEFAULTS)})")


def mf_params(over: Optional[dict] = None, radius_mm: Optional[float] = None,
              radii: Optional[dict] = None) -> dict:
    """The MF-CFAR settings with every radius resolved. radius_mm: the face radius the
    radii not given in `over` derive from (radii.face_radii); radii: an already resolved set
    (face_radii's result) to take them from instead. Without either, every radius must be in
    `over`: a radius is never assumed."""
    check_mf_keys(over)
    p = {**MF_DEFAULTS, **dict(over or {})}
    p["scales_mm"] = tuple(float(w) for w in p["scales_mm"])
    if not p["scales_mm"] or min(p["scales_mm"]) <= 0:
        raise ValueError("scales_mm must be dent diameters > 0 mm")
    if not float(p["z_thresh"]) > 0:
        raise ValueError(f"z_thresh must be > 0, got {p['z_thresh']!r}")
    if p["sign"] not in ("dent", "both"):
        raise ValueError("sign must be 'dent' or 'both'")
    if radii is None and radius_mm is not None:
        radii = radii_mod.face_radii(radius_mm, {k: p[k] for k in MF_RADII})
    for k in MF_RADII:
        if p[k] is None:
            if radii is None:
                raise ValueError(f"MF-CFAR: {k} is not given and there is no face radius to "
                                 "derive it from (mf_params(over, radius_mm)): a radius is "
                                 "never assumed")
            p[k] = radii[k]
    p["noise_bands_mm"] = tuple(float(b) for b in p["noise_bands_mm"])
    for k in MF_RADII[:-1]:
        p[k] = float(p[k])
    return p


def fir_params(over: Optional[dict] = None, radii: Optional[dict] = None) -> dict:
    """The reflectance FIR settings; r_max_mm from the resolved radii when not given."""
    bad = set(over or {}) - set(FIR_DEFAULTS)
    if bad:
        raise ValueError(f"unknown FIR params {sorted(bad)}")
    p = {**FIR_DEFAULTS, **(over or {})}
    if p["r_max_mm"] is None:
        if radii is None:
            raise ValueError("the reflectance FIR's r_max_mm is not given and there are no "
                             "face radii to derive it from")
        p["r_max_mm"] = float(radii["fir_r_max_mm"])
    return p


def grid_mesh(g: np.ndarray) -> tuple:
    GU, GV = np.meshgrid(g, g)
    return GU, GV, np.hypot(GU, GV)


# ------------------------------------------------------------------ pixel data

@dataclass
class Pixels:
    """One placement in camera space (face_profile's crop): face pixels iy, ix (N,), per
    repeat face-frame u, v, h (K, N) mm (NaN = not decoded / off the face), the camera face
    mask (h, w) and the flood pair (h, w) camera DN (None without a flood pair). origin:
    the crop's (x0, y0) in the camera frame (None: a profile stored without it); behind:
    the crop's pixels the scans measured BEHIND the face plane (background, the part's side
    wall: not the face, whatever the flood shows there); outline: the rim outline measured
    in the flood pair (outline.measure; None = not measured yet)."""
    iy: np.ndarray
    ix: np.ndarray
    U: np.ndarray
    V: np.ndarray
    H: np.ndarray
    cam_mask: np.ndarray
    flood_on: Optional[np.ndarray] = None
    flood_off: Optional[np.ndarray] = None
    flood_sat: Optional[np.ndarray] = None
    origin: Optional[tuple] = None
    outline: Optional[dict] = None
    behind: Optional[np.ndarray] = None

    @property
    def K(self) -> int:
        return int(self.H.shape[0])

    @property
    def has_flood(self) -> bool:
        return self.flood_on is not None and self.flood_off is not None

    def arrays(self) -> dict:
        out = {"px_iy": self.iy.astype(np.int32), "px_ix": self.ix.astype(np.int32),
               "px_u": self.U.astype(np.float32), "px_v": self.V.astype(np.float32),
               "px_h": self.H.astype(np.float32), "px_cam_mask": self.cam_mask.astype(np.uint8)}
        if self.behind is not None:
            out["px_behind"] = np.asarray(self.behind, np.uint8)
        if self.has_flood:
            out.update(flood_on=self.flood_on.astype(np.float32),
                       flood_off=self.flood_off.astype(np.float32),
                       flood_sat=np.asarray(self.flood_sat, bool) if self.flood_sat is not None
                       else np.zeros(self.flood_on.shape, bool))
        return out

    @classmethod
    def from_arrays(cls, z) -> Optional["Pixels"]:
        keys = set(z.files if hasattr(z, "files") else z)
        if not {"px_iy", "px_ix", "px_u", "px_v", "px_h", "px_cam_mask"} <= keys:
            return None
        f = (lambda k: np.asarray(z[k], np.float64) if k in keys else None)   # noqa: E731
        origin = tuple(int(v) for v in np.asarray(z["crop"]).ravel()[:2]) if "crop" in keys \
            else None
        return cls(iy=np.asarray(z["px_iy"], np.int64), ix=np.asarray(z["px_ix"], np.int64),
                   U=f("px_u"), V=f("px_v"), H=f("px_h"),
                   cam_mask=np.asarray(z["px_cam_mask"], bool), flood_on=f("flood_on"),
                   flood_off=f("flood_off"),
                   flood_sat=np.asarray(z["flood_sat"], bool) if "flood_sat" in keys else None,
                   origin=origin,
                   behind=np.asarray(z["px_behind"], bool) if "px_behind" in keys else None)

    def with_heights(self, H: np.ndarray) -> "Pixels":
        return Pixels(self.iy, self.ix, self.U, self.V, np.asarray(H, np.float64), self.cam_mask,
                      self.flood_on, self.flood_off, self.flood_sat, self.origin, self.outline,
                      self.behind)

    def flood_diff(self) -> Optional[np.ndarray]:
        """The flood pair's lit - dark image of the crop (NaN where either is unknown)."""
        if not self.has_flood:
            return None
        return np.asarray(self.flood_on, np.float64) - np.asarray(self.flood_off, np.float64)


class Gridder:
    """Per-pixel repeat mean of (u, v) -> one linear operator onto the face grid; pixels kept
    where at least half of the repeats measured them."""

    def __init__(self, px: Pixels, g: np.ndarray, radius_mm: float, max_edge_mm: float = 1.2):
        from scipy import sparse
        from scipy.spatial import Delaunay, QhullError
        self.g = np.asarray(g, np.float64)
        K = px.K
        nv = np.isfinite(px.H).sum(0)
        self.keep = nv >= max(1, (K + 1) // 2)
        G = len(self.g)
        GU, GV, R = grid_mesh(self.g)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.u = np.nanmean(px.U[:, self.keep], 0)
            self.v = np.nanmean(px.V[:, self.keep], 0)
            self.h = np.nanmean(px.H[:, self.keep], 0)
        good = np.isfinite(self.u) & np.isfinite(self.v) & np.isfinite(self.h)
        if not good.all():                       # u / v from other repeats than h: drop
            kk = np.nonzero(self.keep)[0]
            self.keep[kk[~good]] = False
            self.u, self.v, self.h = self.u[good], self.v[good], self.h[good]
        self.iy, self.ix = px.iy[self.keep], px.ix[self.keep]
        self.Hrep = px.H[:, self.keep]
        q = np.c_[GU.ravel(), GV.ravel()]
        n = int(self.keep.sum())
        self.ok = np.zeros(G * G, bool)
        self.W = sparse.csr_matrix((G * G, n))
        if n < 3:
            return
        pts = np.c_[self.u, self.v]
        try:
            tri = Delaunay(pts)
        except (QhullError, ValueError):
            return
        s = tri.find_simplex(q)
        ok = (s >= 0) & (R.ravel() <= radius_mm)
        P = pts[tri.simplices]
        e = np.max(np.linalg.norm(P - np.roll(P, 1, 1), axis=2), 1)
        ok &= np.where(s >= 0, (e <= max_edge_mm)[np.maximum(s, 0)], False)
        gi = np.nonzero(ok)[0]
        si = s[gi]
        T = tri.transform[si]
        b = np.einsum("mij,mj->mi", T[:, :2], q[gi] - T[:, 2])
        w = np.c_[b, 1 - b.sum(1)]
        self.W = sparse.csr_matrix((w.ravel(), (np.repeat(gi, 3), tri.simplices[si].ravel())),
                                   shape=(G * G, n))
        self.ok = ok

    @property
    def n(self) -> int:
        return int(self.keep.sum())

    def grid(self, vals: np.ndarray) -> np.ndarray:
        """Values at the kept pixels (n,) -> (G, G) map (NaN off the face / in holes)."""
        G = len(self.g)
        out = np.full(G * G, np.nan)
        if self.n:
            out[self.ok] = (self.W @ np.asarray(vals, np.float64))[self.ok]
        return out.reshape(G, G)

    def repeat_maps(self, correction: Optional[np.ndarray] = None) -> np.ndarray:
        """(K, G, G): every repeat gridded (missing pixels = the pixel mean), minus the
        correction (n,) mm when given."""
        c = 0.0 if correction is None else np.asarray(correction, np.float64)
        out = []
        for hk in self.Hrep:
            out.append(self.grid(np.where(np.isfinite(hk), hk, self.h) - c))
        return np.stack(out) if out else np.full((0, len(self.g), len(self.g)), np.nan)

    def jacobian(self) -> np.ndarray:
        """d(u, v) / d(camera x, y): (2, 2) mm per px (the pixel footprint on the face)."""
        A = np.c_[self.ix, self.iy, np.ones(self.n)]
        return np.linalg.lstsq(A, np.c_[self.u, self.v], rcond=None)[0][:2].T

    def dent_pixels(self, cx: float, cy: float, depth_mm: float, width_mm: float,
                    ss: int = 4, J: Optional[np.ndarray] = None) -> np.ndarray:
        """A paraboloid dent (depth, diameter, centre in the face frame, mm) integrated over
        each kept pixel's footprint (ss x ss) -> (n,) mm (negative = below)."""
        J = self.jacobian() if J is None else J
        o = (np.arange(ss) + 0.5) / ss - 0.5
        ox, oy = np.meshgrid(o, o)
        ox, oy = ox.ravel(), oy.ravel()
        du = J[0, 0] * ox + J[0, 1] * oy
        dv = J[1, 0] * ox + J[1, 1] * oy
        rr = np.hypot(self.u[:, None] + du[None] - cx, self.v[:, None] + dv[None] - cy)
        return -depth_mm * np.clip(1 - (rr / (width_mm / 2)) ** 2, 0, None).mean(1)


# ------------------------------------------------------------------ reflectance FIR

def _gs_nan(a: np.ndarray, sigma: float) -> np.ndarray:
    import cv2
    m = np.isfinite(a).astype(np.float64)
    num = cv2.GaussianBlur(np.nan_to_num(a), (0, 0), sigma)
    den = cv2.GaussianBlur(m, (0, 0), sigma)
    out = num / np.maximum(den, 1e-9)
    out[m == 0] = np.nan
    return out


def _sat(px: Pixels) -> np.ndarray:
    if px.flood_sat is None:
        return np.zeros(px.cam_mask.shape, bool)
    return np.asarray(px.flood_sat, bool)


def log_flood_hp(px: Pixels, prm: dict) -> np.ndarray:
    """High-passed log flood (on - off) on the camera face mask less its edge_px border
    (NaN elsewhere: the border shows the face's outline, not its reflectance). Saturated
    pixels keep their clipped value: an under-estimate of the reflectance step, still a better
    prediction than none (zeroing the correction near them leaves uncorrected patches that
    stand out against the corrected rest)."""
    import cv2
    on = np.asarray(px.flood_on, np.float64)
    off = np.asarray(px.flood_off, np.float64)
    with np.errstate(invalid="ignore"):
        L = np.log(np.maximum(np.nan_to_num(on - off, nan=1.0), 1.0))
    m = np.asarray(px.cam_mask, bool)
    k = int(prm.get("edge_px") or 0)
    if k > 0:
        m = cv2.erode(m.astype(np.uint8), np.ones((2 * k + 1, 2 * k + 1), np.uint8),
                      borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    L = np.where(m & np.isfinite(on) & np.isfinite(off), L, np.nan)
    return L - _gs_nan(L, float(prm["hp_sigma_px"]))


def fir_features(Lhp: np.ndarray, half: int) -> np.ndarray:
    Lz = np.nan_to_num(Lhp)
    F = [np.roll(np.roll(Lz, dy, 0), dx, 1) for dy in range(-half, half + 1)
         for dx in range(-half, half + 1)]
    return np.stack(F, -1)


def _height_hp_image(px: Pixels, prm: dict) -> np.ndarray:
    """Camera-space high-passed height: per repeat a plane (u, v) removed, repeat mean."""
    img = np.full((px.K,) + px.cam_mask.shape, np.nan)
    for i in range(px.K):
        h, u, v = px.H[i], px.U[i], px.V[i]
        m = np.isfinite(h) & np.isfinite(u) & np.isfinite(v) & (np.hypot(u, v) < prm["r_max_mm"])
        if m.sum() < 10:
            continue
        c = np.linalg.lstsq(np.c_[np.ones(m.sum()), u[m], v[m]], h[m], rcond=None)[0]
        img[i, px.iy, px.ix] = h - (c[0] + c[1] * u + c[2] * v)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        Hm = np.nanmean(img, 0)
    return Hm - _gs_nan(Hm, float(prm["hp_sigma_px"]))


def fit_fir(pixels: list, prm: Optional[dict] = None) -> tuple:
    """Ridge LS of the high-passed height on the 7x7 log-flood features over GOOD
    placements -> (coef (49,), info {n_px, explained_pct})."""
    prm = fir_params(prm)
    half = int(prm["half_px"])
    Xs, ys = [], []
    for px in pixels:
        if not px.has_flood:
            raise ValueError("the reflectance FIR needs the flood pair of every placement")
        Lhp = log_flood_hp(px, prm)
        hp = _height_hp_image(px, prm)
        rr = np.full(px.cam_mask.shape, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            rr[px.iy, px.ix] = np.hypot(np.nanmean(px.U, 0), np.nanmean(px.V, 0))
        m = (rr < prm["r_max_mm"]) & np.isfinite(hp) & np.isfinite(Lhp) & ~_sat(px)
        Xs.append(fir_features(Lhp, half)[m])
        ys.append(hp[m])
    X, y = np.vstack(Xs), np.concatenate(ys)
    if len(y) < 4 * X.shape[1]:
        raise ValueError(f"only {len(y)} usable face pixels to fit the reflectance FIR")
    coef = np.linalg.solve(X.T @ X + float(prm["ridge"]) * np.eye(X.shape[1]), X.T @ y)
    res = y - X @ coef
    expl = 100.0 * (1.0 - float(np.var(res)) / max(float(np.var(y)), 1e-18))
    return coef, {"n_px": int(len(y)), "explained_pct": round(expl, 2),
                  "hp_rms_mm": round(float(np.std(y)), 5), "residual_rms_mm": round(float(np.std(res)), 5)}


def fir_pixels(px: Pixels, coef: np.ndarray, prm: Optional[dict] = None) -> np.ndarray:
    """Predicted reflectance-induced height error (mm) at every pixel of px (N,)."""
    prm = fir_params(prm)
    Lhp = log_flood_hp(px, prm)
    pred = fir_features(Lhp, int(prm["half_px"])) @ np.asarray(coef, np.float64)
    pred[~np.isfinite(Lhp)] = 0.0
    return pred[px.iy, px.ix]


def fir_usable(px: Optional[Pixels], prm: Optional[dict] = None) -> tuple:
    """(usable, why): a flood pair, and at most max_sat_pct of the face saturated in it."""
    prm = fir_params(prm)
    if px is None or not px.has_flood:
        return False, "no flood pair"
    sp = saturated_pct(px) or 0.0
    if sp > float(prm["max_sat_pct"]):
        return False, (f"the flood saturates {sp:.0f}% of the face (> {prm['max_sat_pct']:g}%)")
    return True, ""


def saturated_pct(px: Pixels) -> Optional[float]:
    if not px.has_flood or px.flood_sat is None:
        return None
    return round(100.0 * float(px.flood_sat[px.iy, px.ix].mean()), 2) if len(px.iy) else None


# ------------------------------------------------------------------ maps

def plane_fit(H: np.ndarray, GU, GV, m, robust: bool = True) -> np.ndarray:
    m = m & np.isfinite(H)
    if m.sum() < 10:
        return H - np.nanmedian(H[m]) if m.any() else H
    A = np.c_[np.ones(m.sum()), GU[m], GV[m]]
    y = H[m]
    c = np.linalg.lstsq(A, y, rcond=None)[0]
    if robust:
        for _ in range(3):
            r = y - A @ c
            s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
            k = np.abs(r) < 3 * s
            if k.sum() < 3:
                break
            c = np.linalg.lstsq(A[k], y[k], rcond=None)[0]
    return H - (c[0] + c[1] * GU + c[2] * GV)


def align_and_average(stack: np.ndarray, g, prm: dict) -> np.ndarray:
    """(K, G, G) repeat maps (or one (G, G) map) -> per-repeat robust plane removed -> mean."""
    stack = np.asarray(stack, np.float64)
    if stack.ndim == 2:
        stack = stack[None]
    GU, GV, R = grid_mesh(np.asarray(g))
    al = np.stack([plane_fit(h, GU, GV, R <= prm["plane_r_mm"]) for h in stack])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(al, 0)


def placement_maps(px: Optional[Pixels], g, radius_mm: float, prm: dict,
                   coef: Optional[np.ndarray] = None, fir_prm: Optional[dict] = None,
                   extra_px: Optional[np.ndarray] = None,
                   gridder: Optional[Gridder] = None) -> tuple:
    """-> ((K, G, G) repeat maps, gridder): the FIR correction (coef) and extra per-pixel
    heights (extra_px (n,), e.g. a synthetic dent) applied before gridding."""
    gr = gridder or Gridder(px, g, radius_mm, prm["max_edge_mm"])
    corr = np.zeros(gr.n)
    if coef is not None:
        corr = corr + fir_pixels(px, coef, fir_prm)[gr.keep]
    if extra_px is not None:
        corr = corr - extra_px
    return gr.repeat_maps(corr), gr


def radial_profile(H: np.ndarray, R: np.ndarray, step: float,
                   rmax: Optional[float] = None) -> tuple:
    """Median height per ring of width step, out to rmax (default: the grid's own reach - a
    fixed limit would lump everything beyond it into the last ring on a larger face)."""
    if rmax is None:
        rmax = float(R[R.shape[0] // 2].max()) if R.size else 0.0
    rb = np.arange(0, rmax + step, step)
    idx = np.clip(np.round(R / step).astype(int), 0, len(rb) - 1)
    prof = np.full(len(rb), np.nan)
    fin = np.isfinite(H)
    for i in range(len(rb)):
        v = H[(idx == i) & fin]
        if v.size >= 3:
            prof[i] = np.median(v)
    ok = np.isfinite(prof)
    if ok.sum() >= 2:
        prof = np.interp(rb, rb[ok], prof[ok])
    return rb, prof


def build_model(maps: list, g, prm: dict) -> dict:
    """Good placements' maps (each (G, G), plane-removed repeat means) -> the golden model:
    radial profile (median of the placements' radial profiles) + 2-D median (display)."""
    if not maps:
        raise ValueError("no placement maps")
    g = np.asarray(g, np.float64)
    step = float(g[1] - g[0])
    GU, GV, R = grid_mesh(g)
    maps = [plane_fit(np.asarray(m, np.float64), GU, GV, R <= prm["plane_r_mm"]) for m in maps]
    rb = None
    profs = []
    for m in maps:
        rb, pr = radial_profile(m, R, step)
        profs.append(pr)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return {"radial_r_mm": rb, "radial_h_mm": np.nanmedian(np.stack(profs), 0),
                "map_mm": np.nanmedian(np.stack(maps), 0), "n_placements": len(maps)}


def deviation(H: np.ndarray, model: Optional[dict], g, prm: dict) -> np.ndarray:
    GU, GV, R = grid_mesh(np.asarray(g))
    pm = R <= prm["plane_r_mm"]
    H = plane_fit(np.asarray(H, np.float64), GU, GV, pm)
    if model is not None and np.isfinite(model["radial_h_mm"]).any():
        H = H - np.interp(R, model["radial_r_mm"], model["radial_h_mm"])
    else:                                                          # self-referenced
        rb, prof = radial_profile(H, R, float(g[1] - g[0]))
        if np.isfinite(prof).any():
            H = H - np.interp(R, rb, prof)
    return plane_fit(H, GU, GV, pm)


# ------------------------------------------------------------------ matched filter

_MF_CACHE: dict = {}


def _kernels(w_mm: float, step: float, support_factor: float) -> list:
    a = w_mm / 2.0
    rs = max(support_factor * a, a + 2 * step)
    n = int(np.ceil(rs / step))
    q = np.arange(-n, n + 1) * step
    QU, QV = np.meshgrid(q, q)
    rho = np.hypot(QU, QV)
    S = (rho <= rs + 1e-9).astype(np.float64)
    bowl = -np.clip(1 - (rho / a) ** 2, 0, None) * S               # unit-depth dent
    return [S, QU * S, QV * S, bowl]


def matched_filter(dev: np.ndarray, g, prm: dict) -> tuple:
    """dev (G, G) mm -> depth (S, G, G) mm (+ = a dent), standard-error factor (S, G, G)."""
    import cv2
    g = np.asarray(g, np.float64)
    step = float(g[1] - g[0])
    _, _, R = grid_mesh(g)
    valid = np.isfinite(dev) & (R <= prm["fit_r_mm"])
    W = valid.astype(np.float64)
    x = np.where(valid, dev, 0.0)
    depths, facs = [], []
    bc = cv2.BORDER_CONSTANT
    for w in prm["scales_mm"]:
        ph = _kernels(w, step, prm["support_factor"])
        k = len(ph)
        key = (hash(W.tobytes()), W.shape, float(w), step, float(prm["support_factor"]),
               float(prm["cover_frac"]))
        c = _MF_CACHE.get(key)
        if c is None:
            G = np.empty(dev.shape + (k, k))
            for i in range(k):
                for j in range(i, k):
                    G[..., i, j] = G[..., j, i] = cv2.filter2D(W, cv2.CV_64F, ph[i] * ph[j],
                                                               borderType=bc)
            G = G + np.eye(k) * 1e-9
            ok = (G[..., 3, 3] > 1e-6) & (G[..., 0, 0] > 3)
            with np.errstate(all="ignore"):
                Gi = np.linalg.inv(G[ok]) if ok.any() else np.zeros((0, k, k))
            inv_dd = np.full(dev.shape, np.nan)
            inv_dd[ok] = Gi[:, 3, 3]
            fp = (ph[3] < 0).astype(np.float64)
            cover = cv2.filter2D(valid.astype(np.float64), cv2.CV_64F, fp, borderType=bc)
            full = cv2.filter2D(np.ones_like(W), cv2.CV_64F, fp, borderType=bc)
            bad = (cover < prm["cover_frac"] * full) | ~valid
            c = (ok, Gi[:, 3, :].copy(), inv_dd, bad)
            if len(_MF_CACHE) > 64:              # ~0.3 MB each
                _MF_CACHE.clear()
            _MF_CACHE[key] = c
        ok, g3, inv_dd, bad = c
        T = np.stack([cv2.filter2D(W * x, cv2.CV_64F, ph[i], borderType=bc) for i in range(k)], -1)
        d = np.full(dev.shape, np.nan)
        if ok.any():
            d[ok] = np.einsum("nj,nj->n", g3, T[ok])
        d[bad] = np.nan
        depths.append(d)
        with np.errstate(invalid="ignore"):
            facs.append(np.sqrt(inv_dd))
    return np.stack(depths), np.stack(facs)


def _band_sigma(t: np.ndarray, R: np.ndarray, bands, zone) -> tuple:
    """Robust sigma of t per radial band (1.4826 MAD, >= 30 points; bands with fewer borrow
    their neighbours) -> (per-point sigma, per-band values)."""
    sig = np.full(t.shape, np.nan)
    vals = []
    fin = np.isfinite(t)
    for a, b in zip(bands[:-1], bands[1:]):
        m = zone & (R >= a) & (R < b) & fin
        vals.append(1.4826 * np.median(np.abs(t[m] - np.median(t[m]))) if m.sum() >= 30
                    else np.nan)
    vals = np.array(vals, float)
    good = np.isfinite(vals) & (vals > 0)
    if not good.any():
        return sig, vals
    mids = (np.array(bands[:-1]) + np.array(bands[1:])) / 2
    vf = np.interp(mids, mids[good], vals[good])
    for i, (a, b) in enumerate(zip(bands[:-1], bands[1:])):
        sig[(R >= a) & (R < b)] = vf[i]
    return sig, vf


def detect(stack, model: Optional[dict], g, prm: dict, px_mm: Optional[float] = None) -> dict:
    """One placement: (K, G, G) repeat maps (or a (G, G) map) + the golden model -> z map,
    depth map, triggers (+ deviation, the 1-sigma depth noise floor per dent width).
    A significant region (z >= the threshold) is a trigger when its depth_centre_mm
    (depth.dent_metrics) reaches min_depth_mm and its area min_area_mm2; every significant
    region, trigger or not, is in 'candidates' (a shallower one tells how close a PASS came).
    px_mm: the camera's pixel footprint on the face (for the width note)."""
    import cv2
    g = np.asarray(g, np.float64)
    step = float(g[1] - g[0])
    GU, GV, R = grid_mesh(g)
    H = align_and_average(stack, g, prm)
    dev = deviation(H, model, g, prm)
    depth, fac = matched_filter(dev, g, prm)
    zone = R <= prm["inspect_r_mm"]
    with np.errstate(invalid="ignore", divide="ignore"):
        t = depth / fac
    S = len(prm["scales_mm"])
    Z = np.full(depth.shape, np.nan)
    sig_t = np.full(depth.shape, np.nan)
    for s in range(S):
        sig_t[s], _ = _band_sigma(t[s], R, prm["noise_bands_mm"], zone)
        with np.errstate(invalid="ignore", divide="ignore"):
            Z[s] = t[s] / sig_t[s]
    zs = Z if prm["sign"] == "dent" else np.abs(Z)
    zs = np.where(np.isfinite(zs), zs, -np.inf)
    best = np.argmax(zs, 0)
    Zb = np.take_along_axis(zs, best[None], 0)[0]
    Zb[~np.isfinite(Zb)] = np.nan
    Db = np.take_along_axis(depth, best[None], 0)[0]
    thr = np.full(R.shape, float(prm["z_thresh"]))
    if prm.get("rim_z_thresh") is not None:
        thr[R > prm["rim_r_mm"]] = float(prm["rim_z_thresh"])
    hit = zone & np.isfinite(Zb) & (Zb >= thr)
    n, lab, _, _ = cv2.connectedComponentsWithStats(hit.astype(np.uint8), connectivity=8)
    trig, cands = [], []
    fitted = R <= prm["fit_r_mm"]
    dev3 = depth_mod.box3(np.where(fitted, dev, np.nan)) if n > 1 else None
    for i in range(1, n):
        m = lab == i
        area = float(m.sum()) * step * step
        k = int(np.nanargmax(np.where(m, Zb, -np.inf)))
        y, x = np.unravel_index(k, m.shape)
        d_fit = float(Db[y, x])
        w = float(prm["scales_mm"][best[y, x]])
        r = float(R[y, x])
        met = depth_mod.dent_metrics(np.where(fitted, dev, np.nan), g, y, x, w,
                                     1 if d_fit > 0 else -1, px_mm=px_mm, dev3=dev3)
        # THE depth: the 3 x 3 centre average against the local background; the matched
        # filter's amplitude only where those cells are not measured
        d = met["depth_centre_mm"] if met["depth_centre_mm"] is not None else d_fit
        c = {"zone": "rim" if r > prm["rim_r_mm"] else "face", "r_mm": round(r, 2),
             "theta_deg": round(float(np.degrees(np.arctan2(GV[y, x], GU[y, x]))), 1),
             "u_mm": round(float(GU[y, x]), 2), "v_mm": round(float(GV[y, x]), 2),
             "depth_mm": d if met["depth_centre_mm"] is not None else round(d, 3),
             "depth_centre_mm": met["depth_centre_mm"],
             "depth_min_mm": met["depth_min_mm"], "depth_min_note": met["depth_min_note"],
             "width_half_depth_mm": met["width_half_depth_mm"],
             "width_note": met["width_note"], "depth_fit_mm": round(d_fit, 3),
             "depth_source": "centre_3x3" if met["depth_centre_mm"] is not None else "fit",
             "background_mm": met["background_mm"], "width_mm": w,
             "z": round(float(Zb[y, x]), 2),
             "dev_mm": round(float(dev[y, x]), 3) if np.isfinite(dev[y, x]) else None,
             "area_mm2": round(area, 2), "n_cells": int(m.sum()),
             "kind": "dent" if d_fit > 0 else "bump"}
        c["trigger"] = bool(area >= prm["min_area_mm2"] and abs(d) >= prm["min_depth_mm"])
        cands.append(c)
        if c["trigger"]:
            trig.append({k_: v for k_, v in c.items() if k_ != "trigger"})
    trig.sort(key=lambda t_: -t_["z"])
    cands.sort(key=lambda t_: -t_["z"])
    inner = R <= min(prm["rim_r_mm"], prm["inspect_r_mm"])
    floor = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for s, w in enumerate(prm["scales_mm"]):
            v = float(np.nanmedian((sig_t[s] * fac[s])[inner]))
            floor[f"{w:g}"] = round(v, 4) if np.isfinite(v) else None
        zmax = float(np.nanmax(np.where(zone, Zb, np.nan))) if np.isfinite(Zb[zone]).any() else None
    cov = float(np.isfinite(dev[zone]).mean()) if zone.any() else 0.0
    return {"triggers": trig, "candidates": cands, "z_map": Zb, "depth_map": Db,
            "scale_map": np.asarray(prm["scales_mm"])[best], "deviation": dev, "height": H,
            "noise_floor_mm": floor, "max_z": zmax, "coverage_pct": round(100.0 * cov, 2)}


def z_near(z_map: np.ndarray, g, cx: float, cy: float, width_mm: float, prm: dict,
           tol_mm: float = BATTERY_TOL_MM) -> float:
    """Largest dent z within width / 2 + tol_mm of (cx, cy) in the inspected zone."""
    GU, GV, R = grid_mesh(np.asarray(g))
    near = (R <= prm["inspect_r_mm"]) & (np.hypot(GU - cx, GV - cy) <= width_mm / 2 + tol_mm) \
        & np.isfinite(z_map)
    return float(np.max(z_map[near])) if near.any() else float("nan")


def battery_centres(n: int, seed, r_mm: float = BATTERY_R_MM) -> np.ndarray:
    """n uniformly random dent centres (u, v) in the disc r <= r_mm (seeded)."""
    rng = np.random.default_rng(seed)
    rr = r_mm * np.sqrt(rng.random(n))
    th = rng.random(n) * 2 * np.pi
    return np.c_[rr * np.cos(th), rr * np.sin(th)]

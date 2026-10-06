"""Synthetic dent sets for the dent-detector tests (no rig, no capture path).

A standing shiny cell (fixed-radius cylinder with optional cosine-bowl dents,
engine.capture.mock_stereo.MetricCylinder) seen by the mock bench's pinhole
camera + projector (StereoRig.bench, square DMD grid). Per camera pixel:

    I = exposure x [ proj x P(u, v) x (kd cos_i + ks (n.h)^40) + env(mirror ray) ] + noise

P = the production pattern at the projector point (analytic: floods,
sinusoids in X = col, Y = row / 2, Gaussian-blurred single-mirror lines),
env = a dim room with one bright vertical line (a lamp) that the cell
mirrors (the reflected-line finder's source). Normals come from the dented
surface, so a dent bends the glint, the room line and the fringes; the
metric repeats get the true points + noise and the true projector
coordinates + noise. Written exactly as the library writes them:
metric/<geo>/<entry>, calibrations/<geo>/<entry>/stereo_calibration.json and
dentsets/<geo>/<entry> (engine layout), or a legacy dataset.json folder.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np

from engine.calibrate.stereo import StereoCalibration
from engine.capture.mock_stereo import MetricCylinder, StereoRig, undistort_normalized
from engine.measure.dents import dentset as ds_mod

GEO = "synthetic_dent_rig"
FP = {"rotate90": False, "size": [912, 1140], "basis_axis": "x", "projector_side": "right",
      "baseline_mm": 279.4, "standoff_mm": 823.0}
RADIUS = 10.525
CENTER = (0.0, 0.0, 780.0)
HEIGHT = 60.0
LONG_US, SHORT_US = 66667, 16667


def rig():
    return StereoRig.bench(parity=None)


def cylinder(dents=()):
    return MetricCylinder(center=CENTER, radius=RADIUS, height=HEIGHT, dents=list(dents))


def crop_box(rg) -> tuple:
    """Camera bbox of the cell (+ margin), full-camera px."""
    c = np.array(CENTER)
    pts = []
    for dx in (-RADIUS - 2, RADIUS + 2):
        for dy in (-HEIGHT / 2 - 3, HEIGHT / 2 + 3):
            pts.append(rg.camera.project(c + np.array([dx, dy, -RADIUS])))
    pts = np.array(pts)
    x0, y0 = np.floor(pts.min(0)).astype(int) - 12
    x1, y1 = np.ceil(pts.max(0)).astype(int) + 12
    return int(x0), int(y0), int(x1), int(y1)


def pixel_of(rg, angle_deg: float, z_mm: float) -> tuple:
    """Camera px of the surface point at MetricCylinder (angle, z)."""
    cyl = cylinder()
    c, a, e0, e1 = cyl._frame()
    th = math.radians(angle_deg)
    X = c + RADIUS * (math.cos(th) * e0 + math.sin(th) * e1) + z_mm * a
    u, v = rg.camera.project(X)
    return float(u), float(v)


def _surface(rg, cyl, box):
    x0, y0, x1, y1 = box
    v, u = np.mgrid[y0:y1, x0:x1].astype(np.float64)
    xy = undistort_normalized(np.stack([u, v], -1), rg.camera.K, rg.camera.dist)
    d = np.concatenate([xy, np.ones(xy.shape[:-1] + (1,))], -1)
    H, W = d.shape[:2]
    t, _ = cyl.intersect(np.zeros(3), d.reshape(-1, 3))
    hit = np.isfinite(t).reshape(H, W)
    X = (d.reshape(-1, 3) * np.where(np.isfinite(t), t, np.nan)[:, None]).reshape(H, W, 3)
    # normals of the (dented) surface from its neighbours
    gx = np.gradient(X, axis=1)
    gy = np.gradient(X, axis=0)
    n = np.cross(gx, gy)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    n *= -np.sign(np.einsum("...i,...i", n, X))[..., None]           # toward the camera
    ok = hit & np.isfinite(n).all(-1)
    return X, n, ok


def _pattern(spec: dict, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    k = spec["kind"]
    if k == "flood":
        return np.full(u.shape, spec["level"] / 255.0)
    if k == "sinus":
        t = math.radians(spec["theta_deg"])
        ph = 2 * math.pi * spec["step"] / spec["steps"]
        arg = 2 * math.pi * (u * math.cos(t) + v / 2 * math.sin(t)) / spec["period"] - ph
        return 0.5 + 0.5 * np.cos(arg)
    if k == "lines_v":
        pitch = spec.get("pitch", 10)
        dd = np.mod(u - spec["shift"] + pitch / 2, pitch) - pitch / 2
        return np.exp(-dd ** 2 / (2 * 0.6 ** 2))
    raise ValueError(k)


def render(rg, cyl, box, specs: list, exposure_scale: float, seed: int, invert: bool = False,
           noise_dn: float = 0.6):
    """-> (frames (N, H, W) float32 DN, X, n, ok, uv)."""
    rng = np.random.default_rng(seed)
    X, n, ok = _surface(rg, cyl, box)
    Xp = np.where(ok[..., None], X, 1.0) @ rg.R.T + rg.t
    uv = rg.projector.project(Xp)
    Cp = rg.projector_center
    L = Cp - X
    L /= np.linalg.norm(L, axis=-1, keepdims=True)
    V = -X / np.linalg.norm(X, axis=-1, keepdims=True)
    Hh = (L + V) / np.linalg.norm(L + V, axis=-1, keepdims=True)
    cos_i = np.clip(np.einsum("...i,...i", n, L), 0, 1)
    spec_ = np.clip(np.einsum("...i,...i", n, Hh), 0, 1) ** 40
    refl = -V - 2 * np.einsum("...i,...i", -V, n)[..., None] * n        # mirrored camera ray
    az = np.degrees(np.arctan2(refl[..., 0], -refl[..., 2]))
    env = 3.0 + 60.0 * np.exp(-((az + 25.0) ** 2) / (2 * 1.2 ** 2))    # a lamp 25 deg left
    proj_gain = 20.0 * cos_i + 160.0 * spec_
    frames = []
    for sp in specs:
        P = _pattern(sp, uv[..., 0], uv[..., 1])
        if invert:
            P = 1.0 - P
        img = np.where(ok, exposure_scale * (proj_gain * P + env), 2.0)
        img = img + rng.normal(0, noise_dn, img.shape)
        frames.append(np.clip(img, 0, 255).astype(np.float32))
    return np.stack(frames), X, n, ok, uv


def write_calibration(lib: Path, rg) -> StereoCalibration:
    name = "20260101T000000_000000_stereo"
    d = lib / "calibrations" / GEO / name
    d.mkdir(parents=True)
    tr = rg.truth()
    cal = StereoCalibration(id=f"{GEO}/{name}", geometry_id=GEO, K_cam=tr["K_cam"],
                            dist_cam=tr["dist_cam"], K_proj=tr["K_proj"], dist_proj=np.zeros(5),
                            R=tr["R"], t=np.ravel(tr["t"]),
                            diamond_parity=None, fingerprint=dict(FP), path=d,
                            created=datetime.now().isoformat(timespec="seconds"))
    (d / "stereo_calibration.json").write_text(json.dumps(cal.to_json()))
    return cal


def write_metric(lib: Path, rg, cyl, box, name: str, seed: int, cal) -> Path:
    """One metric-scan repeat (library layout) of the cell."""
    rng = np.random.default_rng(seed)
    x0, y0, x1, y1 = box
    X, n, ok, = _surface(rg, cyl, box)
    V = -X / np.linalg.norm(X, axis=-1, keepdims=True)
    cosv = np.einsum("...i,...i", n, V)
    ok = ok & (cosv > 0.25)
    ray = -V
    Xn = X + ray * rng.normal(0, 0.02, ok.shape)[..., None]
    Xp = np.where(ok[..., None], X, 1.0) @ rg.R.T + rg.t
    uv = rg.projector.project(Xp)
    Hc, Wc = 1080, 1440
    xyz = np.full((Hc, Wc, 3), np.nan, np.float32)
    mask = np.zeros((Hc, Wc), bool)
    cx = np.full((Hc, Wc), np.nan, np.float32)
    cy = np.full((Hc, Wc), np.nan, np.float32)
    xyz[y0:y1, x0:x1] = np.where(ok[..., None], Xn, np.nan)
    mask[y0:y1, x0:x1] = ok
    cx[y0:y1, x0:x1] = np.where(ok, uv[..., 0] + rng.normal(0, 0.03, ok.shape), np.nan)
    cy[y0:y1, x0:x1] = np.where(ok, uv[..., 1] + rng.normal(0, 0.06, ok.shape), np.nan)
    d = lib / "metric" / GEO / name
    d.mkdir(parents=True)
    np.savez_compressed(d / "cloud.npz", xyz=xyz, mask=mask,
                        consistency=np.zeros((Hc, Wc), np.float32))
    hdr = json.dumps({"geometry_id": GEO, "basis_names": ["multifreq"], "meta": {}})
    np.savez_compressed(d / "field.npz", corr_x=cx, corr_y=cy, mask=mask,
                        conf=mask.astype(np.float32), _header=np.array(hdr))
    meta = {"id": name, "label": name, "geometry_id": GEO, "geometry_fp": dict(FP),
            "created": datetime.now().isoformat(timespec="seconds"), "calibration_id": cal.id,
            "method": "rays", "units": "mm", "n_points": int(mask.sum()), "zone": "Z1"}
    (d / "meta.json").write_text(json.dumps(meta))
    return d


def write_dentset(lib: Path, rg, cyl, box, name: str, seed: int, metric_ids: list,
                  lines_v: bool = True, invert: bool = False,
                  legacy_dir: Path | None = None) -> Path:
    """Both production passes of one pose; engine layout under lib/dentsets/<geo>/ (or a
    legacy dataset.json folder when legacy_dir is given: one pattern list, both passes)."""
    x0, y0, x1, y1 = box
    specs = {"long": ds_mod.production_specs("long"),
             "short": ds_mod.production_specs("short", lines_v)}
    if legacy_dir is not None:
        union = ds_mod.production_specs("short", True)
        specs = {"long": union, "short": union}
    scale = {"long": LONG_US / SHORT_US, "short": 1.0}
    d = legacy_dir if legacy_dir is not None else lib / "dentsets" / GEO / name
    d.mkdir(parents=True)
    passes = []
    for i, which in enumerate(("long", "short")):
        F, *_ = render(rg, cyl, box, specs[which], scale[which], seed + 17 * i, invert=invert)
        fn = f"frames_{which}.npy" if legacy_dir is None else f"frames_synth_{which}.npy"
        np.save(d / fn, F)
        passes.append({"name": which, "preset": f"synth_{which}",
                       "exposure_us": LONG_US if which == "long" else SHORT_US,
                       "average": 8, "file": fn, "crop_x0y0x1y1": [x0, y0, x1, y1],
                       **({} if legacy_dir is not None else {"patterns": specs[which]})})
    if legacy_dir is not None:
        (d / "dataset.json").write_text(json.dumps({"zone": "Z1", "dmd_shape": [1140, 912],
                                                    "patterns": specs["long"], "passes": passes}))
    else:
        meta = {"kind": "dentset", "format": 1, "id": name, "label": name, "geometry_id": GEO,
                "geometry_fp": dict(FP), "created": datetime.now().isoformat(timespec="seconds"),
                "zone": "Z1", "crop_x0y0x1y1": [x0, y0, x1, y1], "passes": passes,
                "metric_scans": list(metric_ids), "mock": True}
        (d / "meta.json").write_text(json.dumps(meta))
    return d


def pose(lib: Path, rg, cal, name: str, dents=(), seed: int = 0, repeats: int = 2, **kw) -> Path:
    """Metric repeats + a dent set of one pose -> the dent set folder."""
    cyl = cylinder(dents)
    box = crop_box(rg)
    ids = [write_metric(lib, rg, cyl, box, f"20260101T0000{seed % 100:02d}_{k:06d}_{name}_r{k + 1}",
                        seed * 10 + k, cal).name for k in range(repeats)]
    return write_dentset(lib, rg, cyl, box, f"20260101T0100{seed % 100:02d}_000000_{name}", seed,
                         ids, **kw)

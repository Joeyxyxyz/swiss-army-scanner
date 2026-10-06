"""Synthetic station scenes for the station tests (no rig, no capture path).

Two cells lying with their circular END FACES toward the camera (tests/face_synth: EndFace +
its cylindrical body) and a cooling tube (CT) between them: a thin metal ribbon (thickness
CT_T) bent into a trapezoid wave in the face plane, seen edge-on, its front edge a few mm
behind the faces (a chain of oriented Blocks, one per straight segment of the wave), in front
of the wall; ray-cast by the mock bench's pinhole camera + projector (face_synth.render) and
written as the library writes a metric scan (face_synth.write_metric), plus a flood pair
(flood.npz: white, black) as scan_fast keeps one: faces bright, the CT edge and the wall mid
grey, cell bodies dark, projector shadows black.

Scene frame (the station module's common face frame): n = the face normal toward the camera,
e1 = y_cam x n (~ image right), e2 = n x e1 (~ image down), origin O = the mean face centre.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import face_synth as S
from engine.capture.mock_stereo import Block, EndFace, MetricCylinder, Plane

R0 = S.RADIUS
CT_T = 3.33                     # ribbon thickness, mm
CT_DEPTH = 22.0                 # ribbon width along the cells' axes (behind its front edge), mm
PITCH, AMP, PLATEAU, PHASE = 23.5, 1.2, 0.32, 6.0


def frame(tilt=(20.0, 8.0)):
    n = S.normal_from_tilt(*tilt)
    e1 = np.cross([0.0, 1.0, 0.0], n)
    e1 /= np.linalg.norm(e1)
    e1 *= np.sign(e1[0])
    e2 = np.cross(n, e1)
    e2 *= np.sign(e2[1])
    return n, e1, e2


def wave_vertices(s_lo: float, s_hi: float, pitch=PITCH, amp=AMP, f=PLATEAU, ph=PHASE) -> np.ndarray:
    """Vertices (s, w) of the trapezoid wave (the station module's trap(): x = 0 bottom plateau
    centre at w = -amp ... x = 0.5 top plateau at +amp)."""
    pts = []
    k0 = int(np.floor((s_lo - ph) / pitch)) - 1
    k1 = int(np.ceil((s_hi - ph) / pitch)) + 1
    for k in range(k0, k1):
        for x, w in ((f / 4, -amp), (0.5 - f / 4, amp), (0.5 + f / 4, amp), (1 - f / 4, -amp)):
            pts.append((ph + pitch * (k + x), w))
    pts = np.array(pts)
    return pts[(pts[:, 0] >= s_lo - pitch) & (pts[:, 0] <= s_hi + pitch)]


def ct_blocks(O, n, e1, e2, h_front: float, s_lo: float, s_hi: float, bump=None, **wave) -> list:
    """The ribbon as oriented Blocks between consecutive wave vertices, front edge at h_front
    (along n from O; negative = behind the faces), clipped to [s_lo, s_hi]. bump = (s0, s1, dh):
    the front edge raised by dh between s0 and s1 (a local CT height change)."""
    V = wave_vertices(s_lo, s_hi, **wave)
    out = []
    for (sa, wa), (sb, wb) in zip(V[:-1], V[1:]):
        sa_, sb_ = max(sa, s_lo), min(sb, s_hi)
        if sb_ <= sa_:
            continue
        wa_ = wa + (wb - wa) * (sa_ - sa) / (sb - sa)
        wb_ = wa + (wb - wa) * (sb_ - sa) / (sb - sa)
        segs = [(sa_, wa_, sb_, wb_, 0.0)]
        if bump is not None:
            b0, b1, dh = bump
            segs = []
            cuts = sorted({sa_, sb_, *[c for c in (b0, b1) if sa_ < c < sb_]})
            for c0, c1 in zip(cuts[:-1], cuts[1:]):
                w0 = wa_ + (wb_ - wa_) * (c0 - sa_) / (sb_ - sa_)
                w1 = wa_ + (wb_ - wa_) * (c1 - sa_) / (sb_ - sa_)
                segs.append((c0, w0, c1, w1, dh if b0 <= 0.5 * (c0 + c1) <= b1 else 0.0))
        for s0, w0, s1, w1, dh in segs:
            a = O + s0 * e1 + w0 * e2
            b = O + s1 * e1 + w1 * e2
            u = b - a
            L = float(np.linalg.norm(u))
            u /= L
            p = np.cross(n, u)
            hf = h_front + dh
            c = 0.5 * (a + b) + (hf - CT_DEPTH / 2) * n
            out.append(Block(center=tuple(c), half=(L / 2 + 0.05, CT_T / 2, CT_DEPTH / 2),
                             rot=np.stack([u, p, n]), albedo=0.6))
    return out


def scene(c2_minus_c1: float = -0.2, h_ct: float = -4.5, tilt=(20.0, 8.0),
          O=(-50.0, 0.0, 780.0), cell_s=(-6.0, 6.0), bump=None, s_range=(-75.0, 95.0)) -> dict:
    n, e1, e2 = frame(tilt)
    O = np.asarray(O, float)
    v_off = R0 + CT_T / 2 + AMP + 0.25          # cells clear of the ribbon on either side
    cells = []
    for i, (s_, v_, h_) in enumerate(((cell_s[0], -v_off, -c2_minus_c1 / 2),
                                      (cell_s[1], v_off, c2_minus_c1 / 2))):
        c = O + s_ * e1 + v_ * e2 + h_ * n
        cells.append(c)
    objs = []
    for c in cells:
        objs += [EndFace(center=tuple(c), normal=tuple(n), radius=R0, chamfer_mm=0.6,
                         chamfer_depth_mm=0.15, albedo=0.9),
                 MetricCylinder(center=tuple(c - n * 20.0), radius=R0, height=40.0, axis=tuple(n))]
    objs += ct_blocks(O, n, e1, e2, h_ct, s_range[0], s_range[1], bump)
    objs.append(Plane())
    return {"objs": objs, "cells": cells, "n": n, "e1": e1, "e2": e2, "O": O,
            "truth": {"C2-C1": c2_minus_c1, "spacing": float(np.hypot(cell_s[1] - cell_s[0],
                                                                     2 * v_off)),
                      "h_ct": h_ct, "pitch": PITCH, "amp": AMP}}


def box_for(rg, sc, s_range=(-75.0, 95.0), margin_mm: float = 30.0) -> tuple:
    pts = []
    for s_ in s_range:
        for v_ in (-margin_mm, margin_mm):
            pts.append(rg.camera.project(sc["O"] + s_ * sc["e1"] + v_ * sc["e2"]))
    pts = np.array(pts)
    x0, y0 = np.floor(pts.min(0)).astype(int)
    x1, y1 = np.ceil(pts.max(0)).astype(int)
    return max(int(x0), 0), max(int(y0), 0), min(int(x1), S.CAM[1]), min(int(y1), S.CAM[0])


def write_station(lib: Path, rg, cal, name: str, seed: int = 0, noise_mm: float = 0.02,
                  flood: bool = True, corr_noise_px: float = 0.03, **kw) -> tuple:
    """A station metric scan (library layout, as face_synth.write_metric writes one) + its
    flood pair (flood.npz: faces 225, CT edge 140, wall 110, bodies 55, projector shadow 8),
    from ONE ray-cast -> (scan id, scene dict)."""
    import cv2
    sc = scene(**kw)
    box = box_for(rg, sc, kw.get("s_range", (-75.0, 95.0)))
    objs = sc["objs"]
    rng = np.random.default_rng(seed)
    x0, y0, x1, y1 = box
    X, uv, ok, idx = S.render(rg, objs, box)
    ray = X / np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), 1e-9)
    Xn = X + ray * rng.normal(0, noise_mm, ok.shape)[..., None]
    xyz = np.full(S.CAM + (3,), np.nan, np.float32)
    mask = np.zeros(S.CAM, bool)
    cx = np.full(S.CAM, np.nan, np.float32)
    cy = np.full(S.CAM, np.nan, np.float32)
    xyz[y0:y1, x0:x1] = np.where(ok[..., None], Xn, np.nan)
    mask[y0:y1, x0:x1] = ok
    cx[y0:y1, x0:x1] = np.where(ok, uv[..., 0] + rng.normal(0, corr_noise_px, ok.shape), np.nan)
    cy[y0:y1, x0:x1] = np.where(ok, uv[..., 1] + rng.normal(0, corr_noise_px, ok.shape), np.nan)
    d = Path(lib) / "metric" / S.GEO / name
    d.mkdir(parents=True)
    np.savez_compressed(d / "cloud.npz", xyz=xyz, mask=mask)
    hdr = json.dumps({"geometry_id": S.GEO, "basis_names": ["multifreq"], "meta": {}})
    np.savez_compressed(d / "field.npz", corr_x=cx, corr_y=cy, mask=mask,
                        conf=mask.astype(np.float32), _header=np.array(hdr))
    if flood:
        lvl = np.array([225.0 if isinstance(o, EndFace) else 55.0 if isinstance(o, MetricCylinder)
                        else 140.0 if isinstance(o, Block) else 110.0 for o in objs])
        img = np.where(ok, lvl[np.clip(idx, 0, None)], 8.0).astype(np.float32)
        img = cv2.GaussianBlur(img, (0, 0), 0.8)
        white = np.zeros(S.CAM, np.float32)
        black = np.zeros(S.CAM, np.float32)
        white[y0:y1, x0:x1] = np.clip(img + rng.normal(0, 1.0, img.shape), 0, 255)
        black[y0:y1, x0:x1] = np.clip(4.0 + rng.normal(0, 0.5, img.shape), 0, 255)
        np.savez_compressed(d / "flood.npz", white=white, black=black)
    meta = {"id": name, "label": name, "geometry_id": S.GEO, "geometry_fp": dict(S.FP),
            "created": "2026-01-01T00:00:00", "calibration_id": cal.id, "method": "plane",
            "units": "mm", "n_points": int(mask.sum()), "zone": "synthetic station"}
    (d / "meta.json").write_text(json.dumps(meta))
    sc["box"] = box
    return name, sc

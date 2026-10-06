"""
Diamond row-parity check (DLP4500): which rows sit half a column to the right.

On the diamond DMD every other row is offset by half a mirror pitch. A
pattern that varies only with the column index is then a zigzag in space;
the multifreq basis draws straight fringes once it knows the parity, and
the projector calibration's (col, row) convention depends on it — so it is
recorded once per rig (bench step B3) and stored with the calibration.

Three patterns, one-mirror vertical lines every `spacing` columns:
  A  even rows only, at columns k * spacing + offset
  B  odd rows only,  same columns
  C  even rows only, `ref_shift` columns further right (the scale + sign reference)
The camera sees each as a row of vertical lines. With x_A, x_B, x_C the
line centroids along the camera x axis (a central band of rows averaged),
    shift_cols = ref_shift * median(x_B - x_A) / median(x_C - x_A)
is +0.5 when odd rows sit half a column right of even rows (parity 1),
-0.5 when even rows sit right of odd rows (parity 0), ~0 for a square grid.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

DMD_SHAPE = (1140, 912)


def parity_patterns(dmd_shape=DMD_SHAPE, spacing: int = 16, offset: int = 8,
                    ref_shift: int = 2) -> list:
    """[A, B, C] uint8 DMD images (see module doc)."""
    H, W = dmd_shape
    rows = np.arange(H)[:, None]
    cols = np.arange(W)[None, :]
    line = (cols % spacing) == offset
    line_ref = (cols % spacing) == (offset + ref_shift) % spacing
    even, odd = (rows & 1) == 0, (rows & 1) == 1
    return [np.where(even & line, 255, 0).astype(np.uint8),
            np.where(odd & line, 255, 0).astype(np.uint8),
            np.where(even & line_ref, 255, 0).astype(np.uint8)]


def _centroids(profile: np.ndarray, min_sep: int) -> np.ndarray:
    p = profile - np.median(profile)
    if abs(np.percentile(p, 1)) > abs(np.percentile(p, 99)):
        p = -p                      # an inverting HDMI path: dark lines on a lit field
    p = p - np.percentile(p, 20)
    thr = 0.3 * np.percentile(p, 99)
    peaks = [i for i in range(2, len(p) - 2)
             if p[i] > thr and p[i] >= p[i - 1] and p[i] > p[i + 1]]
    out, last = [], -10 ** 9
    for i in peaks:
        if i - last < min_sep:
            continue
        w = np.clip(p[i - 2:i + 3], 0, None)
        out.append(i - 2 + float((w * np.arange(5)).sum() / max(w.sum(), 1e-9)))
        last = i
    return np.array(out)


def _match(a: np.ndarray, b: np.ndarray, tol: float) -> np.ndarray:
    if not len(a) or not len(b):
        return np.array([])
    j = np.abs(b[None, :] - a[:, None]).argmin(1)
    d = b[j] - a
    return d[np.abs(d) < tol]


STRIP_ROWS, STRIP_STEP = 6, 8


def measure_parity(frames: list, band: Optional[tuple] = None, spacing: int = 16,
                   ref_shift: int = 2, dark: Optional[np.ndarray] = None) -> dict:
    """Camera frames of [A, B, C] (+ optional dark frame) -> parity report.

    band: (y0, y1) camera rows to average. Default: every narrow strip of
    STRIP_ROWS rows down the frame, measured on its own, then the median
    over the strips that found lines. Averaging a tall band smears lines
    that are tilted in the camera (bench 2026-09-28: a ~5 deg tilt over a
    216-row band drifted each line by almost a line spacing, spread 0.9
    columns; 6-row strips gave spread 0.02)."""
    if band is None:
        H = np.asarray(frames[0]).shape[0]
        per = [measure_parity(frames, (y, y + STRIP_ROWS), spacing, ref_shift, dark)
               for y in range(0, H - STRIP_ROWS, STRIP_STEP)]
        per = [r for r in per if r.get("shift_cols") is not None and r.get("n_lines", 0) >= 5]
        if not per:
            return {"parity": None, "shift_cols": None, "ok": False,
                    "note": "no strip found the line pattern: check it is in view and in focus"}
        shifts = np.array([r["shift_cols"] for r in per])
        shift = float(np.median(shifts))
        spread = float(np.median([r["shift_spread_cols"] for r in per]))
        out = _verdict(shift, spread, float(np.median([r["camera_px_per_column"] for r in per])),
                       int(sum(r["n_lines"] for r in per)))
        out["n_strips"] = len(per)
        out["strip_shift_iqr_cols"] = [round(float(x), 4) for x in np.percentile(shifts, [25, 75])]
        return out
    fa, fb, fc = (np.asarray(f, np.float64) for f in frames[:3])
    if dark is not None:            # the inverse polarity case is handled per profile
        fa, fb, fc = (f - np.asarray(dark, np.float64) for f in (fa, fb, fc))
    H = fa.shape[0]
    y0, y1 = band if band is not None else (int(0.4 * H), int(0.6 * H))
    prof = [f[y0:y1].mean(0) for f in (fa, fb, fc)]
    xa = _centroids(prof[0], 3)
    if len(xa) < 3:
        return {"parity": None, "shift_cols": None, "ok": False,
                "note": "fewer than 3 lines found: check the pattern is in view and in focus"}
    px_per_line = float(np.median(np.diff(xa)))
    tol = 0.4 * px_per_line
    xb, xc = _centroids(prof[1], 3), _centroids(prof[2], 3)
    dab, dac = _match(xa, xb, tol), _match(xa, xc, tol)
    if len(dab) < 3 or len(dac) < 3 or abs(np.median(dac)) < 1e-6:
        return {"parity": None, "shift_cols": None, "ok": False,
                "note": "could not match the line sets"}
    px_per_col = float(np.median(dac)) / ref_shift       # signed: camera x per DMD column
    shift = float(np.median(dab)) / px_per_col
    spread = float(np.std(dab / px_per_col))
    return _verdict(shift, spread, px_per_col, int(len(dab)))


def _verdict(shift: float, spread: float, px_per_col: float, n_lines: int) -> dict:
    if abs(abs(shift) - 0.5) < 0.15:
        parity, ok = (1 if shift > 0 else 0), True
    elif abs(shift) < 0.15:
        parity, ok = None, True
    else:
        parity, ok = None, False
    return {"parity": parity, "shift_cols": round(shift, 4), "shift_spread_cols": round(spread, 4),
            "camera_px_per_column": round(px_per_col, 4), "n_lines": n_lines, "ok": ok,
            "note": ("odd rows sit half a column right of even rows" if parity == 1 else
                     "even rows sit half a column right of odd rows" if parity == 0 else
                     "no half-column offset seen (square grid)" if ok else
                     f"offset {shift:+.3f} columns is neither 0 nor +-0.5: check focus / band")}

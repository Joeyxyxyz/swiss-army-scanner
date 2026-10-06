"""
Pattern generators.

Each function produces a 912x1140 uint8 array (or a list of them)
suitable for streaming to the DMD or feeding the simulator.

The physics:
  The DMD performs an inner product in the optical domain. A scene s(x,y)
  illuminated by a pattern p_k(x,y) returns total intensity
      y_k = ∫∫ p_k(x,y) · s(x,y) dx dy
  The choice of {p_k} is the choice of measurement basis. Sensors differ
  only in their basis.
"""

from __future__ import annotations
import math
import numpy as np

DMD_W = 912
DMD_H = 1140


def flood(value: int = 255) -> np.ndarray:
    return np.full((DMD_H, DMD_W), value, dtype=np.uint8)


def black() -> np.ndarray:
    return np.zeros((DMD_H, DMD_W), dtype=np.uint8)


def single_point(cx: int = DMD_W // 2, cy: int = DMD_H // 2, radius: int = 4) -> np.ndarray:
    img = np.zeros((DMD_H, DMD_W), dtype=np.uint8)
    yy, xx = np.ogrid[:DMD_H, :DMD_W]
    img[(xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2] = 255
    return img


def vertical_line(x: int, width: int = 2) -> np.ndarray:
    img = np.zeros((DMD_H, DMD_W), dtype=np.uint8)
    img[:, max(0, x - width // 2):min(DMD_W, x + (width + 1) // 2)] = 255
    return img


def horizontal_line(y: int, width: int = 2) -> np.ndarray:
    img = np.zeros((DMD_H, DMD_W), dtype=np.uint8)
    img[max(0, y - width // 2):min(DMD_H, y + (width + 1) // 2), :] = 255
    return img


def crosshair(cx: int = DMD_W // 2, cy: int = DMD_H // 2) -> np.ndarray:
    return np.maximum(vertical_line(cx), horizontal_line(cy))


def phase_shift_fringes(n_steps: int = 4, period_px: int = 24,
                        orientation: str = "vertical") -> list:
    """N-step phase-shifting profilometry patterns.
        I_k(x,y) = 127 + 127 * cos(2π x / T  +  2π k / N)
    """
    out = []
    yy, xx = np.meshgrid(np.arange(DMD_H), np.arange(DMD_W), indexing="ij")
    coord = xx if orientation == "vertical" else yy
    for k in range(n_steps):
        phase = 2 * math.pi * k / n_steps
        I = 127.5 + 127.5 * np.cos(2 * math.pi * coord / period_px + phase)
        out.append(I.astype(np.uint8))
    return out


def gray_code_stripes(n_bits: int = 10, orientation: str = "vertical") -> list:
    """Gray-coded binary stripes for absolute correspondence.

    All bit-planes are bits of ONE gray code of ONE column index, so the
    captured stack decodes to a consistent absolute position:
        idx = x // width;  g = idx ^ (idx >> 1);  pattern_b = bit b of g
    (bit 0 = finest stripes). v0.4 rescaled idx per bit, producing
    mutually inconsistent planes — caught by
    tests/test_reference_physics.py.
    """
    out = []
    extent = DMD_W if orientation == "vertical" else DMD_H
    width = max(1, -(-extent // (2 ** n_bits)))   # ceil div
    idx = np.arange(extent) // width
    gray = idx ^ (idx >> 1)
    for b in range(n_bits):
        stripe = ((gray >> b) & 1).astype(np.uint8) * 255
        if orientation == "vertical":
            img = np.tile(stripe[np.newaxis, :], (DMD_H, 1))
        else:
            img = np.tile(stripe[:, np.newaxis], (1, DMD_W))
        out.append(img)
    return out


def hadamard_patterns(order: int = 6, fov_px: int = 256, cx=None, cy=None) -> list:
    """Walsh-Hadamard patterns at order 2^order, tiled into fov_px×fov_px."""
    N = 2 ** order
    H = _sylvester_hadamard(N)
    cx = DMD_W // 2 if cx is None else cx
    cy = DMD_H // 2 if cy is None else cy
    half = fov_px // 2
    out = []
    for i in range(N):
        for j in range(N):
            base = np.outer(H[i], H[j])
            tile = ((base + 1) // 2 * 255).astype(np.uint8)
            up = _kron_resize(tile, fov_px)
            img = np.zeros((DMD_H, DMD_W), dtype=np.uint8)
            img[cy - half:cy + half, cx - half:cx + half] = up
            out.append(img)
    return out


def _sylvester_hadamard(N: int) -> np.ndarray:
    """Recursive Sylvester construction of order-N Hadamard."""
    assert (N & (N - 1)) == 0, "N must be a power of 2"
    H = np.array([[1]])
    while H.shape[0] < N:
        H = np.block([[H, H], [H, -H]])
    return H


def _kron_resize(arr: np.ndarray, target: int) -> np.ndarray:
    scale = target // arr.shape[0]
    if scale < 1:
        raise ValueError(
            f"fov_px={target} is smaller than the {arr.shape[0]}x"
            f"{arr.shape[0]} Hadamard tile — lower `order` or raise "
            f"`fov_px` (need fov_px >= 2**order).")
    return np.kron(arr, np.ones((scale, scale), dtype=arr.dtype))


def sobel_pair() -> list:
    """Two complementary gradient masks; subtract camera frames to get edges."""
    g_x = np.tile(np.linspace(0, 255, DMD_W).astype(np.uint8), (DMD_H, 1))
    g_y = np.tile(np.linspace(0, 255, DMD_H).astype(np.uint8)[:, None], (1, DMD_W))
    return [g_x, 255 - g_x, g_y, 255 - g_y]


def random_bernoulli(n_patterns: int = 256, p: float = 0.5, seed: int = 0,
                     block: int = 8) -> list:
    """Block-Bernoulli patterns for compressed sensing."""
    rng = np.random.default_rng(seed)
    out = []
    bh, bw = DMD_H // block, DMD_W // block
    for _ in range(n_patterns):
        small = (rng.random((bh, bw)) < p).astype(np.uint8) * 255
        out.append(np.kron(small, np.ones((block, block), dtype=np.uint8)))
    return [p_[:DMD_H, :DMD_W] for p_ in out]


def flicker_carrier(freq_hz: float, n_frames: int = 64, fps: float = 1000.0,
                    base=None) -> list:
    """Modulate intensity at freq_hz for lock-in style vibration sensing."""
    out = []
    b = flood() if base is None else base
    for k in range(n_frames):
        t = k / fps
        amp = 0.5 + 0.5 * math.cos(2 * math.pi * freq_hz * t)
        out.append((b.astype(np.float32) * amp).astype(np.uint8))
    return out


def high_freq_stripes(period_px: int = 6) -> list:
    """Nayar high-frequency complementary stripes for specular/diffuse separation."""
    yy, xx = np.meshgrid(np.arange(DMD_H), np.arange(DMD_W), indexing="ij")
    p1 = (((xx // (period_px // 2)) & 1) * 255).astype(np.uint8)
    p2 = 255 - p1
    return [p1, p2]


def barcode_flood() -> np.ndarray:
    return flood(255)


# ---------------------------------------------------------------------------
# Numbered aim grid (Joey / Surfer contract)
# origin: DMD (0,0) = top-left of 912x1140
# col A = x [0, step), B = [step, 2*step), ...
# row 1 = y [0, step), 2 = [step, 2*step), ...
# ---------------------------------------------------------------------------

_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class GridIndex:
    """Parse A1 / P17-Q22 style labels into DMD pixel rects."""

    def __init__(self, step_px: int = 48, dmd_size: tuple = (DMD_W, DMD_H)):
        self.step = int(step_px)
        self.W, self.H = int(dmd_size[0]), int(dmd_size[1])
        self.cols = self.W // self.step
        self.rows = self.H // self.step

    def _col(self, letter: str) -> int:
        letter = letter.strip().upper()
        if len(letter) != 1 or letter not in _LETTERS:
            raise ValueError(f"bad column letter: {letter!r}")
        i = _LETTERS.index(letter)
        if i >= self.cols:
            raise ValueError(f"column {letter} out of range (max {_LETTERS[self.cols-1]})")
        return i

    def _row(self, n: int) -> int:
        n = int(n)
        if n < 1 or n > self.rows:
            raise ValueError(f"row {n} out of range 1..{self.rows}")
        return n - 1

    def cell_rect(self, label: str):
        """'P17' -> (x1,y1,x2,y2) half-open pixel rect."""
        label = label.strip().upper()
        i = 0
        while i < len(label) and label[i].isalpha():
            i += 1
        col = self._col(label[:i])
        row = self._row(label[i:])
        x1 = col * self.step
        y1 = row * self.step
        return (x1, y1, x1 + self.step, y1 + self.step)

    def parse(self, spec: str):
        """'P17' or 'P17-P22' or 'P17-Q22' -> inclusive (x1,y1,x2,y2)."""
        spec = spec.strip().upper().replace(" ", "")
        if "-" in spec:
            a, b = spec.split("-", 1)
            ax1, ay1, ax2, ay2 = self.cell_rect(a)
            bx1, by1, bx2, by2 = self.cell_rect(b)
            return (min(ax1, bx1), min(ay1, by1), max(ax2, bx2), max(ay2, by2))
        return self.cell_rect(spec)


def numbered_grid(step_px: int = 48, major_every: int = 4,
                  dmd_size: tuple = (DMD_W, DMD_H),
                  held=None, label: bool = True,
                  room_level: int = 80, held_level: int = 230) -> np.ndarray:
    """Room lattice + optional brighter held overlay; A1 labels in-cell.

    held: None | (x1,y1,x2,y2) | str like 'P17-P22'
    Returns uint8 HxW grayscale image.
    """
    W, H = int(dmd_size[0]), int(dmd_size[1])
    step = int(step_px)
    img = np.full((H, W), 18, dtype=np.uint8)
    for x in range(0, W, step):
        img[:, x:x + 1] = room_level
    for y in range(0, H, step):
        img[y:y + 1, :] = room_level
    maj = max(int(major_every), 1) * step
    for x in range(0, W, maj):
        img[:, x:x + 2] = min(200, room_level + 100)
    for y in range(0, H, maj):
        img[y:y + 2, :] = min(200, room_level + 100)

    gi = GridIndex(step_px=step, dmd_size=(W, H))
    if isinstance(held, str):
        held = gi.parse(held)
    if held is not None:
        x1, y1, x2, y2 = [int(v) for v in held]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(W, x2); y2 = min(H, y2)
        for x in range(x1, x2, max(step // 3, 8)):
            img[y1:y2, x] = held_level
        for y in range(y1, y2, max(step // 3, 8)):
            img[y, x1:x2] = held_level
        img[y1:y1 + 3, x1:x2] = 255
        img[y2 - 3:y2, x1:x2] = 255
        img[y1:y2, x1:x1 + 3] = 255
        img[y1:y2, x2 - 3:x2] = 255

    if label:
        try:
            import cv2
        except ImportError:
            return img
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cols = W // step
        rows = H // step
        for c in range(cols):
            for r in range(rows):
                lab = f"{_LETTERS[c]}{r + 1}"
                cv2.putText(bgr, lab, (c * step + 3, r * step + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1,
                            cv2.LINE_AA)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return img

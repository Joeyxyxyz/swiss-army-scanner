"""Internal utilities."""
import numpy as np


def resample_nearest(arr: np.ndarray, shape) -> np.ndarray:
    """Nearest-neighbor resample to (H, W)."""
    h_in, w_in = arr.shape[:2]
    h_out, w_out = shape
    ys = np.linspace(0, h_in - 1, h_out).astype(int)
    xs = np.linspace(0, w_in - 1, w_out).astype(int)
    return arr[np.ix_(ys, xs)]


def synthetic_scene(shape, kind: str = "plane") -> np.ndarray:
    """Generate a simulated reflectance scene as a float [0,1] image."""
    h, w = shape
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    if kind == "plane":
        return np.full((h, w), 0.7, dtype=np.float32)
    if kind == "sphere":
        cx, cy, r = w // 2, h // 2, min(h, w) // 3
        d2 = (xx - cx) ** 2 + (yy - cy) ** 2
        z = np.maximum(0, 1 - d2 / r ** 2)
        return (0.3 + 0.7 * np.sqrt(z)).astype(np.float32)
    if kind == "step":
        s = np.full((h, w), 0.6, dtype=np.float32)
        s[:, w // 2:] = 0.9
        return s
    if kind == "qr":
        s = np.full((h, w), 0.95, dtype=np.float32)
        block = max(8, min(h, w) // 32)
        rng = np.random.default_rng(7)
        nh, nw = h // block + 1, w // block + 1
        bits = rng.integers(0, 2, (nh, nw))
        bits[:7, :7] = 0; bits[:7, -7:] = 0; bits[-7:, :7] = 0
        bits[1:6, 1:6] = 1; bits[1:6, -6:-1] = 1; bits[-6:-1, 1:6] = 1
        big = np.kron(1.0 - bits.astype(np.float32) * 0.9, np.ones((block, block)))
        return big[:h, :w].astype(np.float32)
    return np.full((h, w), 0.5, dtype=np.float32)

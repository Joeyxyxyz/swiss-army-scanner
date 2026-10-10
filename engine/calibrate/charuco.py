"""
ChArUco board: definition, printable files (PNG / PDF at true size) and
detection (OpenCV aruco).

Print the PDF at 100 % ("actual size", no fit-to-page), glue it flat on
glass or foam board, then MEASURE a printed square with calipers and pass
that number as square_mm to calibrate_stereo: the printer's scale error
goes straight into every mm the scanner reports.

Board coordinates: mm, origin at the top-left of the square grid, x right,
y down, z = 0 (cv2.aruco's convention).

first_id: the board's markers are ids first_id .. first_id + n_markers - 1 of
the dictionary (0 = the standard board). Boards with DISJOINT id ranges can
share one camera frame and are detected separately (the calibration fixture,
calibrate/fixture.py: one capture = one planar view per face).
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

MM_PER_INCH = 25.4
PT_PER_INCH = 72.0


@dataclass(frozen=True)
class BoardSpec:
    squares_x: int = 10
    squares_y: int = 7
    square_mm: float = 25.0
    marker_mm: float = 18.0
    dictionary: str = "DICT_5X5_100"
    first_id: int = 0                   # markers first_id .. first_id + n_markers - 1

    def __post_init__(self):
        if self.squares_x < 3 or self.squares_y < 3:
            raise ValueError("a ChArUco board needs at least 3 x 3 squares")
        if not 0 < self.marker_mm < self.square_mm:
            raise ValueError(f"marker_mm must be in (0, square_mm={self.square_mm}), "
                             f"got {self.marker_mm}")
        if (isinstance(self.first_id, bool) or int(self.first_id) != self.first_id
                or self.first_id < 0):
            raise ValueError(f"first_id must be an integer >= 0, got {self.first_id!r}")
        n = dictionary_size(self.dictionary)
        if n is not None and self.first_id + self.n_markers > n:
            raise ValueError(f"{self.dictionary} has {n} markers: ids {self.first_id}.."
                             f"{self.first_id + self.n_markers - 1} do not fit")

    @property
    def board_size_mm(self) -> tuple:
        return (self.squares_x * self.square_mm, self.squares_y * self.square_mm)

    @property
    def n_corners(self) -> int:
        return (self.squares_x - 1) * (self.squares_y - 1)

    @property
    def n_markers(self) -> int:
        return (self.squares_x * self.squares_y) // 2

    @property
    def marker_ids(self) -> tuple:
        """(first, last) marker id of this board, inclusive."""
        return (int(self.first_id), int(self.first_id) + self.n_markers - 1)

    def to_dict(self) -> dict:
        d = asdict(self)
        if not d["first_id"]:
            del d["first_id"]               # the standard board keeps its stored form
        return d

    @classmethod
    def from_dict(cls, d: Optional[dict]) -> "BoardSpec":
        return cls(**(d or {}))

    def scaled(self, square_mm: float) -> "BoardSpec":
        """The same board with the caliper-measured square size."""
        k = float(square_mm) / self.square_mm
        return BoardSpec(self.squares_x, self.squares_y, float(square_mm),
                         self.marker_mm * k, self.dictionary, self.first_id)

    # -------------------------------------------------------------- cv2
    def board(self):
        import cv2
        d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.dictionary))
        if not self.first_id:
            return cv2.aruco.CharucoBoard((self.squares_x, self.squares_y),
                                          float(self.square_mm), float(self.marker_mm), d)
        ids = np.arange(self.first_id, self.first_id + self.n_markers, dtype=np.int32)
        return cv2.aruco.CharucoBoard((self.squares_x, self.squares_y), float(self.square_mm),
                                      float(self.marker_mm), d, ids)

    def object_points(self) -> np.ndarray:
        """(n_corners, 3) chessboard corners in board mm, id order."""
        return np.asarray(self.board().getChessboardCorners(), np.float64).reshape(-1, 3)

    def image(self, px_per_mm: float, margin_mm: float = 0.0) -> np.ndarray:
        """uint8 board image at px_per_mm (white margin of margin_mm)."""
        w, h = self.board_size_mm
        m = int(round(margin_mm * px_per_mm))
        size = (int(round(w * px_per_mm)) + 2 * m, int(round(h * px_per_mm)) + 2 * m)
        return self.board().generateImage(size, marginSize=m, borderBits=1)


def dictionary_size(name: str) -> Optional[int]:
    """Markers in a predefined cv2.aruco dictionary (DICT_5X5_250 -> 250), None if the
    name does not say (the AprilTag / original sets)."""
    tail = str(name).rsplit("_", 1)[-1]
    return int(tail) if tail.isdigit() else None


def detect(image: np.ndarray, spec: BoardSpec, min_corners: int = 6):
    """-> (corners (N, 2) camera px, ids (N,), object points (N, 3) mm), or
    None when fewer than min_corners corners are found."""
    import cv2
    img = np.asarray(image)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    det = cv2.aruco.CharucoDetector(spec.board())
    corners, ids, _, _ = det.detectBoard(img)
    if corners is None or ids is None or len(ids) < min_corners:
        return None
    corners = np.asarray(corners, np.float64).reshape(-1, 2)
    ids = np.asarray(ids).reshape(-1).astype(int)
    return corners, ids, spec.object_points()[ids]


# ------------------------------------------------------------ print files

def write_png(path: Path, spec: BoardSpec, dpi: float = 600.0, margin_mm: float = 10.0) -> Path:
    """PNG with the physical DPI recorded (pHYs), so a viewer prints it at size."""
    import cv2
    img = spec.image(dpi / MM_PER_INCH, margin_mm)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("PNG encode failed")
    data = _png_with_dpi(buf.tobytes(), dpi)
    with open(path, "xb") as f:
        f.write(data)
    return Path(path)


def _png_with_dpi(png: bytes, dpi: float) -> bytes:
    ppm = int(round(dpi / 0.0254))
    body = b"pHYs" + struct.pack(">IIB", ppm, ppm, 1)
    chunk = struct.pack(">I", 9) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
    i = png.find(b"IDAT") - 4                      # insert before the first IDAT
    return png[:i] + chunk + png[i:]


def write_pdf(path: Path, spec: BoardSpec, dpi: float = 600.0, margin_mm: float = 10.0) -> Path:
    """One-page PDF, page = board + margin, image placed at exact physical size."""
    img = spec.image(dpi / MM_PER_INCH, margin_mm)
    h, w = img.shape
    w_mm, h_mm = spec.board_size_mm
    page_w = (w_mm + 2 * margin_mm) / MM_PER_INCH * PT_PER_INCH
    page_h = (h_mm + 2 * margin_mm) / MM_PER_INCH * PT_PER_INCH
    stream = zlib.compress(img.tobytes(), 9)
    content = f"q {page_w:.4f} 0 0 {page_h:.4f} 0 0 cm /Im0 Do Q".encode()
    note = (f"ChArUco {spec.squares_x}x{spec.squares_y} square {spec.square_mm:g} mm "
            f"marker {spec.marker_mm:g} mm {spec.dictionary} - print at 100%")
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {page_w:.4f} {page_h:.4f}] "
         f"/Resources << /XObject << /Im0 5 0 R >> >> /Contents 4 0 R >>").encode(),
        b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream",
        (f"<< /Type /XObject /Subtype /Image /Width {w} /Height {h} /ColorSpace /DeviceGray "
         f"/BitsPerComponent 8 /Filter /FlateDecode /Length {len(stream)} >>\nstream\n").encode()
        + stream + b"\nendstream",
        f"<< /Title ({note}) /Producer (swiss_army_scanner) >>".encode(),
    ]
    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for i, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R /Info {len(objs)} 0 R >>\n"
            f"startxref\n{xref}\n%%EOF\n").encode()
    with open(path, "xb") as f:
        f.write(bytes(out))
    return Path(path)

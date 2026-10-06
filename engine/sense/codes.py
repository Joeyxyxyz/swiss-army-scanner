"""
Barcode / QR reading — the DMD as a programmable illuminator.

The DMD lights the scene; the camera images it; symbol decoding is done by
well-validated libraries (writing a clean-room QR/Reed-Solomon decoder is
not the point — swiss_army_sensor.barcode says the same):

  * OpenCV QRCodeDetector (multi)       QR
  * OpenCV barcode.BarcodeDetector      EAN-8/13, UPC-A/E, Code 39/93/128, ITF...
  * zxing-cpp, if installed             QR, Data Matrix, Aztec, PDF417, 1D —
                                        strongest on low-resolution real codes
  * pyzbar / pylibdmtx, if installed    extra 1D + Data Matrix (optional)

Resolution rule of thumb: decoders need ~2.5-3 camera px per module. At
the 2.7 ft bench standoff the camera resolves ~1.2 px/mm, so a dense QR
(33 modules) must be >~70 mm wide to read; smaller codes need the lens
zoomed in or the camera closer (bench 2026-09-22: a 50 px QR was DETECTED
by every decoder but decoded by none).

What the DMD adds over a flashlight: illumination BRACKETING without
touching camera exposure, and COLOUR. The LightCrafter has separate R/G/B
LEDs; in HDMI video mode a red image lights only the red LED — no USB. Red
is the barcode-scanner standard: on a mono camera red/orange print
vanishes and black / blue / green print goes dark, so codes stand out on
coloured packaging. Colour floods are sent raw (no duty/polarity chain). Glossy labels saturate under full light and
dark print vanishes under dim light, so we read under several DMD levels
and keep every code found. Polarity is measured, not assumed (the bench
HDMI path lands patterns inverted), so "full on" is whichever frame is
brighter.
"""

from __future__ import annotations

import numpy as np

from .. import _seed
from ..results import Code

DEFAULT_LEVELS = (255, 170, 110)       # DMD drive levels for bracketing
COLOURS = {"red": (0, 0, 255), "green": (0, 255, 0), "blue": (255, 0, 0),
           "white": (255, 255, 255)}   # BGR, as drawn in the HDMI window


def _to_u8(frame: np.ndarray) -> np.ndarray:
    f = np.asarray(frame, dtype=np.float32)
    hi = float(np.percentile(f, 99.8)) or 1.0
    return np.clip(f * (255.0 / hi), 0, 255).astype(np.uint8)


def _variants(img: np.ndarray):
    """Preprocessing variants: raw, local-contrast, 2x upscale (small codes)."""
    import cv2
    yield "raw", img, 1.0
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img)
    yield "clahe", clahe, 1.0
    yield "clahe_x2", cv2.resize(clahe, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC), 2.0


# decoder spellings -> one kind name: OpenCV says EAN_13, zxing EAN13, zbar QRCODE
_KINDS = {"QRCODE": "QR", "QR": "QR", "MICROQRCODE": "MICRO_QR", "EAN13": "EAN_13",
          "EAN8": "EAN_8", "UPCA": "UPC_A", "UPCE": "UPC_E", "CODE128": "CODE_128",
          "CODE39": "CODE_39", "CODE93": "CODE_93", "DATAMATRIX": "DATA_MATRIX",
          "PDF417": "PDF417", "AZTEC": "AZTEC", "ITF": "ITF", "I25": "ITF",
          "CODABAR": "CODABAR"}


def normalise_kind(kind) -> str:
    """One name per symbology whatever the decoder: 'EAN_13', 'EAN13', 'ean-13'
    -> 'EAN_13'; 'QRCode' / 'QRCODE' -> 'QR'. Unknown names: upper case."""
    k = "".join(ch for ch in str(kind or "?").upper() if ch.isalnum())
    return _KINDS.get(k, str(kind or "?").strip().upper().replace(" ", "_").replace("-", "_"))


def normalise_code(kind, data: str) -> tuple:
    """(kind, data, raw): an EAN-13 whose first digit is 0 is a UPC-A, and the
    label prints the 12 digits; raw keeps the decoder's EAN-13 ('' if unchanged)."""
    kind, data = normalise_kind(kind), str(data)
    if kind == "EAN_13" and len(data) == 13 and data.isdigit() and data[0] == "0":
        return "UPC_A", data[1:], data
    return kind, data, ""


def decode_image(img: np.ndarray, illumination: str = "") -> list[Code]:
    """All codes found in one camera image (uint8 or float). Every decoder
    runs on every preprocessing variant (raw, CLAHE, CLAHE x2): a dim, noisy
    Data Matrix is often only readable after the contrast boost. One entry
    per (kind, data), kinds normalised across decoders."""
    import cv2
    img = img if img.dtype == np.uint8 else _to_u8(img)
    found: dict = {}

    def add(kind, data, pts, decoder, variant, scale):
        if not data:
            return
        kind, data, raw = normalise_code(kind, data)
        key = (kind, data)
        if key in found:
            return
        corners = None if pts is None else (np.asarray(pts, float).reshape(-1, 2) / scale).round(1).tolist()
        found[key] = Code(kind=kind, data=data, corners=corners,
                          decoder=f"{decoder}:{variant}", illumination=illumination, raw=raw)

    qr = cv2.QRCodeDetector()
    bc = cv2.barcode.BarcodeDetector() if hasattr(cv2, "barcode") else None
    for variant, im, scale in _variants(img):
        ok, texts, pts, _ = qr.detectAndDecodeMulti(im)
        if ok:
            for t, p in zip(texts, pts if pts is not None else [None] * len(texts)):
                add("QR", t, p, "opencv-qr", variant, scale)
        if bc is not None:
            try:
                ok, texts, types, pts = bc.detectAndDecodeWithType(im)
            except Exception:
                ok = False
            if ok:
                for t, ty, p in zip(texts, types, pts):
                    add(ty or "BARCODE", t, p, "opencv-barcode", variant, scale)
    try:
        import zxingcpp
        for variant, im, scale in _variants(img):
            for r in zxingcpp.read_barcodes(im):
                pos = r.position
                pts = [[pos.top_left.x, pos.top_left.y], [pos.top_right.x, pos.top_right.y],
                       [pos.bottom_right.x, pos.bottom_right.y], [pos.bottom_left.x, pos.bottom_left.y]]
                add(getattr(r.format, "name", str(r.format)), r.text, pts, "zxing", variant, scale)
    except ImportError:
        pass
    # optional extra decoders (pyzbar / pylibdmtx) via the seed module
    for d in _seed.load("decoders").decode_barcode(img).decoded:
        add(d.get("type", "?"), d.get("data", ""), d.get("polygon") or None, "zbar/dmtx", "raw", 1.0)
    return list(found.values())


def bracket_patterns(levels=DEFAULT_LEVELS):
    P = _seed.load("patterns")
    return [(f"flood_{lv}", P.flood(int(lv))) for lv in levels]


def colour_flood(bgr, shape=(1140, 912)) -> np.ndarray:
    img = np.zeros(shape + (3,), np.uint8)
    img[...] = bgr
    return img


DEFAULT_COLOURS = ("red", "white", "green", "blue")


def capture(sync, levels=DEFAULT_LEVELS, colours=DEFAULT_COLOURS) -> dict:
    """Shoot the illumination bracket: DMD black, one flood per level, then
    the raw colour floods. Only the shoot calls — nothing slow — so the rig
    (pumped HDMI window, camera, rig lock) can be released before decode().

    -> frames {name: frame}, in shooting order ("dmd_black", "flood_<lv>"...,
    then the colour names)."""
    if not levels:
        raise ValueError("levels must list at least one DMD flood level")
    unknown = [c for c in colours if c not in COLOURS]
    if unknown:
        raise ValueError(f"unknown colour(s) {unknown}; choose from {sorted(COLOURS)}")
    P = _seed.load("patterns")
    frames = {"dmd_black": sync.shoot(P.black())}
    for name, pat in bracket_patterns(levels):
        frames[name] = sync.shoot(pat)
    for c in colours:
        frames[c] = sync.shoot(colour_flood(COLOURS[c]), raw=True)
    return frames


def decode(frames: dict, levels=DEFAULT_LEVELS):
    """Decode every frame of a capture() bracket (no rig needed).

    -> (codes, full_on_name, polarity_inverted)
    """
    flood = [f"flood_{lv}" for lv in levels]
    missing = [n for n in ["dmd_black"] + flood if n not in frames]
    if missing:
        raise ValueError(f"frames lack {missing}: decode() takes the frames capture() returned "
                         "with the same levels")
    # bench polarity: if 'black' is the brightest, the path inverts — the
    # physical full-on is the black pattern, and the levels run the other way
    inverted = bool(frames["dmd_black"].mean() > frames[flood[0]].mean())
    order = (["dmd_black"] + flood[::-1] if inverted else flood + ["dmd_black"])
    order += [n for n in frames if n not in order]          # the colour floods
    codes: dict = {}
    for name in order:
        for c in decode_image(frames[name], illumination=name):
            codes.setdefault((normalise_kind(c.kind), c.data), c)
    return list(codes.values()), order[0], inverted


def read(sync, levels=DEFAULT_LEVELS, colours=DEFAULT_COLOURS):
    """capture() + decode() in one call (decoding while the rig is still
    open; api.read_codes decodes after leaving it).

    -> (codes, frames{name: frame}, full_on_name, polarity_inverted)
    """
    frames = capture(sync, levels, colours)
    codes, full_on, inverted = decode(frames, levels)
    return codes, frames, full_on, inverted

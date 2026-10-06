"""
make_flash_pack — pack a preset's structured-light patterns into DLPC350
pattern-mode images (24-bit 912x1140 BMPs) + a manifest + a GUI procedure.

    python tools/make_flash_pack.py                       # default preset, x axis, flash
    python tools/make_flash_pack.py --axes x,y            # + horizontal-stripe variants
    python tools/make_flash_pack.py --video-port          # frames for HDMI Pattern Sequence
    python tools/make_flash_pack.py --unpack-check DIR    # re-verify a written pack (any kind)

    # 1-bit soft-sync fast scan (internal timer, free-running camera):
    python tools/make_flash_pack.py --fastscan                        # full-field image only
    python tools/make_flash_pack.py --fastscan --zone-ring C1.npy --zone-ring C2.npy \
        [--zone-band-cam CT_CAM.npy --zone-band-field FIELD.npz] [--zone-mask DMD.npy] \
        [--calibration CAL_DIR] [--study]                             # + zone-masked image

Output: <library>/output/flashpacks/<ts>_<name>/ (a fresh directory,
never an existing one — LibraryStore.new_dir + write_json). --fastscan packs
go to the library output bin, <library>/output/flashpacks/<ts>_<name>/
(recorded in output/index.csv), or under --out.

This tool NEVER talks to the projector or the camera. Flashing / entering
pattern mode is a human step in the LightCrafter 4500 MKII GUI (see
docs/FLASH_PATTERN_MODE.md, docs/FLASH_FASTSCAN_4500.md and the
procedure.txt written into each pack).

Packing (verified against TI DLP SDK lcr4500.cpp, CreateFirmwareImages /
StartPatternImageStorage / SavePatternIntImageAsRGBfile):

  * a pattern of bit depth d is stored at a "bit position" p in 0..23 of a
    24-bit image, p aligned as the SDK does (1-bit: any p; 8-bit: p in
    {0, 8, 16}); the SDK walks patterns in order and moves to the next
    aligned slot, opening a new image when p > 23.
  * bit position p -> colour plane: 0..7 = G0..G7, 8..15 = R0..R7,
    16..23 = B0..B7 (DLPU010G Fig. 2-5, Table 2-69 "pattern number mapping";
    SDK: r = bits 8-15, g = bits 0-7, b = bits 16-23 of the packed int).
  * a 1-bit pattern is its bit (pixel > 0 -> 1); an 8-bit pattern is its
    grey value (G, R or B byte whole).

Timing rules (DLPU010G §2.4.3.4.3; SDK lcr4500.cpp lines 39-62, 835-836):
exposure >= the bit-depth minimum (1-bit 235 us ... 8-bit 8333 us), and
exposure == period, or period - exposure >= 230 us. In VSYNC trigger
modes exposure must equal period and a frame's patterns must fit in the
VSYNC period.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine import config  # noqa: E402
from engine.bases.hybrid import from_preset_bases  # noqa: E402
from engine.library.store import LibraryStore  # noqa: E402

DMD_W, DMD_H = 912, 1140
DEFAULT_PRESET = "cell_wall_2p7ft_vstripe_fullrange"
# 8-bit packs default to the library output bin (the same library as
# engine.api: SWISS_SCANNER_LIBRARY, else <repo>/library); --out overrides.
DEFAULT_OUT = Path(os.environ.get("SWISS_SCANNER_LIBRARY", ROOT / "library")) / "output" / "flashpacks"

# ---------------------------------------------------------------- timing
# minimum pattern exposure (us) per mono bit depth (SDK Exposure::MININUM)
MIN_EXPOSURE_US = {1: 235, 2: 700, 3: 1570, 4: 1700, 5: 2000, 6: 2500, 7: 4500, 8: 8333}
PERIOD_DIFFERENCE_MIN_US = 230
FLASH_MAX_IMAGES = 64            # DLPU011E §3.3: up to 64 24-bit images in flash
LUT_MAX_ENTRIES = 128            # DLPU010G pattern LUT: 1..128 entries
VSYNC_60HZ_US = 16666
# Blackfly S BFS-U3-16S2M (IMX273, 1440x1080 Mono8): ~226 fps max -> ~4.43 ms
# per frame. Trigger faster than that and frames are dropped.
CAMERA_MIN_PERIOD_US = 4500
# DLPU011E §3.3.3 / §4: loading one 24-bit image from flash takes up to
# ~200 ms; the DLPC350 has two 24-bit buffers (48 bit-planes preloaded).
# A sequence over > 2 images must show each image for longer than a load.
FLASH_IMAGE_LOAD_MS = 200.0
PRELOADED_IMAGES = 2

# SDK StartPatternImageStorage tables, index = bpp-1
_STORAGE = (1, 2, 3, 4, 6, 6, 8, 8)
_OFFSET = (0, 0, 0, 0, 1, 0, 1, 0)


class TimingError(ValueError):
    """An exposure/period the DLPC350 would reject (Validate Data fails)."""


def timing_errors(bitdepth: int, exposure_us: int, period_us: int, *,
                  vsync: bool = False) -> list:
    """Every DLPU010G rule the (bitdepth, exposure, period) pair breaks."""
    errs = []
    if bitdepth not in MIN_EXPOSURE_US:
        return [f"bit depth must be 1..8, got {bitdepth}"]
    e, p = int(exposure_us), int(period_us)
    if e < MIN_EXPOSURE_US[bitdepth]:
        errs.append(f"{bitdepth}-bit exposure {e} us < minimum {MIN_EXPOSURE_US[bitdepth]} us")
    if not (e == p or e <= p - PERIOD_DIFFERENCE_MIN_US):
        errs.append(f"exposure {e} us / period {p} us: must be equal, or period - exposure "
                    f">= {PERIOD_DIFFERENCE_MIN_US} us")
    if vsync and e != p:
        errs.append(f"VSYNC trigger modes need exposure == period (got {e} / {p} us)")
    return errs


def validate_timing(bitdepth: int, exposure_us: int, period_us: int, *,
                    vsync: bool = False) -> None:
    errs = timing_errors(bitdepth, exposure_us, period_us, vsync=vsync)
    if errs:
        raise TimingError("; ".join(errs))


# ------------------------------------------------------------- patterns
def build_patterns(preset, axes=("x",)) -> list:
    """The scan's pattern set in HybridBasis order per axis:
    white, black, gray planes (coarse bits only if coarse_bits), fringes.
    Each item: {name, axis, role, basis_index, bitdepth, image (H, W) uint8}.
    No HDMI invert/rotate is applied: pattern mode shows mirrors directly,
    and HybridBasis.decode measures polarity from white/black anyway."""
    out = []
    for axis in axes:
        basis = from_preset_bases(preset.bases, axis=axis)
        imgs = basis.generate()
        n_gray = basis.used_bits * (2 if basis._comp else 1)
        for i, img in enumerate(imgs):
            if i == 0:
                role, name = "white", "white"
            elif i == 1:
                role, name = "black", "black"
            elif i < 2 + n_gray:
                role = "gray"
                j = i - 2
                if basis._comp:
                    name = f"gray{j // 2 + basis.dropped_bits}{'_inv' if j % 2 else ''}"
                else:
                    name = f"gray{j + basis.dropped_bits}"
            else:
                role, name = "fringe", f"fringe{i - 2 - n_gray}"
            img = np.ascontiguousarray(img, dtype=np.uint8)
            if img.shape != (DMD_H, DMD_W):
                raise ValueError(f"pattern {name} is {img.shape}, DMD is {(DMD_H, DMD_W)}")
            binary = bool(np.isin(img, (0, 255)).all())
            if role == "fringe" and binary:
                binary = False              # fringes are grey-level by definition
            out.append({"name": f"{axis}_{name}", "axis": axis, "role": role,
                        "basis_index": i, "bitdepth": 1 if binary else 8, "image": img})
    return out


def plane_label(bit_position: int) -> str:
    return "GRB"[bit_position // 8] + str(bit_position % 8)


def _aligned(pos: int, bpp: int) -> bool:
    return (pos - _OFFSET[bpp - 1]) % _STORAGE[bpp - 1] == 0 and pos - _OFFSET[bpp - 1] >= 0


# ------------------------------------------------------------------ pack
def pack(patterns: list, *, policy: str = "sdk", max_per_image: Optional[dict] = None):
    """-> (images [(H, W, 3) uint8, R, G, B channel order], sequence [dict]).

    policy "sdk": TI SDK order — walk patterns, next aligned slot, new image
      past bit 23 (mixed depths share an image). Used for flash.
    policy "grouped": a new image whenever the bit depth changes, and at most
      max_per_image[bpp] patterns per image. Used for the video port, where
      each image is one VSYNC frame and all its patterns must fit in it."""
    if policy not in ("sdk", "grouped"):
        raise ValueError(f"policy must be 'sdk' or 'grouped', got {policy!r}")
    max_per_image = max_per_image or {}
    acc = [np.zeros((DMD_H, DMD_W), np.uint32)]
    counts = [0]
    seq = []
    pos, img_i, last_bpp = 0, 0, None

    def new_image():
        nonlocal pos, img_i
        acc.append(np.zeros((DMD_H, DMD_W), np.uint32))
        counts.append(0)
        img_i += 1
        pos = 0

    for k, pat in enumerate(patterns):
        bpp = int(pat["bitdepth"])
        if bpp not in (1, 8):
            raise ValueError(f"{pat['name']}: only 1- and 8-bit patterns are packed")
        if policy == "grouped" and counts[img_i] and (
                bpp != last_bpp or counts[img_i] >= max_per_image.get(bpp, 99)):
            new_image()
        while not _aligned(pos, bpp) or pos + bpp > 24:
            pos += 1
            if pos > 23:
                new_image()
        img = np.asarray(pat["image"])
        if bpp == 1:
            val = (img > 0).astype(np.uint32)
        else:
            val = img.astype(np.uint32)
        acc[img_i] |= val << np.uint32(pos)
        counts[img_i] += 1
        seq.append({"seq": k, "name": pat["name"], "axis": pat["axis"], "role": pat["role"],
                    "basis_index": pat["basis_index"], "bitdepth": bpp,
                    "image_index": img_i, "bit_position": pos,
                    "pattern_number": (pos - _OFFSET[bpp - 1]) // _STORAGE[bpp - 1],
                    "planes": [plane_label(pos + b) for b in range(bpp)]})
        last_bpp = bpp
        pos += bpp
        if pos > 23 and k + 1 < len(patterns):
            new_image()
    if counts[-1] == 0:
        acc.pop()
        counts.pop()
    images = []
    for a in acc:
        rgb = np.empty((DMD_H, DMD_W, 3), np.uint8)
        rgb[..., 0] = (a >> 8) & 255        # R = bits 8..15
        rgb[..., 1] = a & 255               # G = bits 0..7
        rgb[..., 2] = (a >> 16) & 255       # B = bits 16..23
        images.append(rgb)
    return images, seq


def unpack(images: list, sequence: list) -> list:
    """Inverse of pack: the DMD image each sequence entry shows, in sequence
    order (1-bit -> 0/255, 8-bit -> the grey byte)."""
    out = []
    for e in sequence:
        rgb = images[e["image_index"]]
        packed = (rgb[..., 1].astype(np.uint32) | (rgb[..., 0].astype(np.uint32) << 8)
                  | (rgb[..., 2].astype(np.uint32) << 16))
        v = (packed >> np.uint32(e["bit_position"])) & np.uint32((1 << e["bitdepth"]) - 1)
        out.append((v * 255).astype(np.uint8) if e["bitdepth"] == 1 else v.astype(np.uint8))
    return out


# ------------------------------------------------------------------- BMP
def write_bmp(path: Path, rgb: np.ndarray) -> Path:
    """Uncompressed 24-bit BMP (bottom-up, BGR bytes), exclusive create."""
    h, w, _ = rgb.shape
    row = w * 3
    pad = (-row) % 4
    data = np.ascontiguousarray(rgb[::-1, :, ::-1])
    if pad:
        data = np.concatenate([data.reshape(h, row), np.zeros((h, pad), np.uint8)], axis=1)
    raw = data.tobytes()
    header = struct.pack("<2sIHHI", b"BM", 54 + len(raw), 0, 0, 54)
    info = struct.pack("<IiiHHIIiiII", 40, w, h, 1, 24, 0, len(raw), 2835, 2835, 0, 0)
    with open(path, "xb") as f:
        f.write(header + info + raw)
    return path


def read_bmp(path: Path) -> np.ndarray:
    b = Path(path).read_bytes()
    if b[:2] != b"BM":
        raise ValueError(f"{path}: not a BMP")
    off, = struct.unpack_from("<I", b, 10)
    _, w, h, _, bpp, comp = struct.unpack_from("<IiiHHI", b, 14)
    if bpp != 24 or comp != 0:
        raise ValueError(f"{path}: need uncompressed 24-bit BMP, got {bpp} bpp comp {comp}")
    top_down = h < 0
    h = abs(h)
    row = w * 3 + (-(w * 3)) % 4
    a = np.frombuffer(b, np.uint8, count=row * h, offset=off).reshape(h, row)[:, :w * 3]
    a = a.reshape(h, w, 3)[..., ::-1]
    return np.ascontiguousarray(a if top_down else a[::-1])


# ---------------------------------------------------------------- timing
def suggest_timing(mode: str, sequence: list, *, vsync_us: int = VSYNC_60HZ_US,
                   camera_min_period_us: int = CAMERA_MIN_PERIOD_US) -> dict:
    """Legal exposure/period per bit depth + the camera exposure to use."""
    depths = sorted({e["bitdepth"] for e in sequence})
    if mode == "flash":
        per = {}
        for d in depths:
            exp = max(MIN_EXPOSURE_US[d], camera_min_period_us - PERIOD_DIFFERENCE_MIN_US - 270)
            period = max(exp + PERIOD_DIFFERENCE_MIN_US + 270, camera_min_period_us)
            validate_timing(d, exp, period)
            per[str(d)] = {"exposure_us": exp, "period_us": period,
                           "camera_exposure_us": exp + 100}
        # one global exposure/period (plain Pattern Sequence tab) must satisfy
        # the deepest pattern
        g = per[str(max(depths))]
        total_var = sum(per[str(e["bitdepth"])]["period_us"] for e in sequence)
        total_uni = g["period_us"] * len(sequence)
        return {"mode": "flash", "trigger_mode": "Internal (Pattern Trigger Mode 1) — or "
                "Mode 3 variable exposure to use per-depth times",
                "per_bitdepth": per, "uniform": dict(g),
                "sequence_ms_uniform": round(total_uni / 1000, 1),
                "sequence_ms_variable": round(total_var / 1000, 1),
                "notes": ["camera exposure = pattern exposure + 100 us, inside the >=230 us "
                          "dark gap (period - exposure), so it integrates the whole PWM "
                          "bit-plane sequence of 8-bit patterns",
                          f"period >= {camera_min_period_us} us: Blackfly S full-frame "
                          "Mono8 tops out near 226 fps"]}
    # video port: VSYNC (trigger mode 0/4), exposure == period, a frame's
    # patterns fill at most one VSYNC period
    frames = {}
    for e in sequence:
        frames.setdefault(e["image_index"], []).append(e)
    per_frame = []
    for i, es in sorted(frames.items()):
        d = es[0]["bitdepth"]
        exp = vsync_us // len(es)
        errs = timing_errors(d, exp, exp, vsync=True)
        per_frame.append({"frame_index": i, "bitdepth": d, "patterns": len(es),
                          "exposure_us": exp, "period_us": exp,
                          "camera_exposure_us": max(exp - 150, 50), "errors": errs})
    bad = [f for f in per_frame if f["errors"]]
    if bad:
        raise TimingError(f"video-port frames that cannot fit one VSYNC: {bad}")
    return {"mode": "video_port", "trigger_mode": "VSYNC (Pattern Trigger Mode 4, "
            "variable exposure; exposure == period)", "vsync_us": vsync_us,
            "per_frame": per_frame,
            "sequence_ms": round(len(per_frame) * vsync_us / 1000, 1),
            "notes": ["exposure == period in VSYNC modes: there is no dark gap, so the "
                      "camera exposure is set 150 us SHORT of the pattern exposure",
                      "fringes above 1 per frame need the camera to finish readout in "
                      "exposure_us: check the per-frame period against ~4.5 ms"]}


def flash_load_warnings(sequence: list, timing: dict) -> list:
    """Buffer-load rule: beyond the 2 preloaded images, each image must be
    displayed for longer than a flash load (~200 ms) or the sequence stalls."""
    n_img = 1 + max(e["image_index"] for e in sequence)
    if n_img <= PRELOADED_IMAGES:
        return []
    per = timing["uniform"]["period_us"]
    show = {}
    for e in sequence:
        show[e["image_index"]] = show.get(e["image_index"], 0) + per / 1000.0
    short = {i: round(ms, 1) for i, ms in show.items() if ms < FLASH_IMAGE_LOAD_MS}
    if not short:
        return []
    return [f"{n_img} flash images > {PRELOADED_IMAGES} preloaded buffers and image(s) "
            f"{short} (index: ms shown) display for less than the ~{FLASH_IMAGE_LOAD_MS:g} ms "
            "flash load time: the sequence will stall or glitch. Make one pack per axis "
            "(--axes x, then --axes y) so each sequence fits the two preloaded buffers."]


# ------------------------------------------------------------- procedure
def procedure_text(manifest: dict) -> str:
    m, t = manifest, manifest["timing"]
    L = [f"FLASH PATTERN PACK  {m['name']}   preset {m['preset']}   mode {m['mode']}",
         "=" * 72, "",
         "Nothing here was sent to the projector. Every step below is done by a",
         "human in the TI DLP LightCrafter 4500 GUI and SpinView.", ""]
    for w in t.get("warnings", []):
        L += ["!! WARNING: " + w, ""]
    if m["mode"] == "flash":
        L += ["A. BACK UP FIRST (once): open the firmware link in the 4500 MKII GUI, note the",
              "   current firmware version and save the unit's ORIGINAL (vendor-supplied)",
              "   firmware .bin somewhere safe. A bad flash is recovered only by re-uploading",
              "   a known-good .bin (docs/FLASH_FASTSCAN_4500.md, rollback).", "",
              "B. BUILD FIRMWARE: Image/Firmware -> Firmware Build:",
              "   1. Firmware File: browse to your known-good base .bin",
              "   2. Firmware tag: e.g. " + m["name"][:32],
              "   3. Add these 24-bit images, in this flash order:"]
        for im in m["images"]:
            L.append(f"        flash index {im['index']:2d}  <- {im['file']}")
        L += ["      (the base firmware may already hold images: the flash index a pattern",
              "       uses is its position in the Firmware Build list — if you KEEP",
              "       existing images, add FLASH_OFFSET to every index below)",
              "   4. Illumination: Monochrome, one LED channel (e.g. Green) — or RGB",
              "   5. Save Updates -> <name>.bin ; then Firmware Upload -> Upload. Wait.",
              ""]
        L += ["C. PATTERN SEQUENCE TAB (after reboot):",
              "   Pattern Source: Flash    Trigger Mode: Internal/External",
              f"   Pattern Exposure {t['uniform']['exposure_us']} us   "
              f"Pattern Period {t['uniform']['period_us']} us   (uniform; "
              f"sequence ~{t['sequence_ms_uniform']} ms)",
              "   Uncheck Repeat (play once) so the camera sees exactly one pass.",
              "   Individual patterns, add in THIS order (LED = one colour, trigger = Internal",
              "   for the first, No Trigger for the rest):"]
    else:
        L += ["A. HDMI: the laptop must output 912x1140 @ 60 Hz, 24-bit, NO scaling,",
              "   gamma / colour management off. Pattern Sequence with Video port is a",
              "   DISPLAY MODE CHANGE — it is exactly the switch that has dropped the DMD",
              "   to blue lines (docs/HDMI_TIMING_FIX.md). Fix the HDMI timing first.", "",
              "B. Show frames_video/frame_NN.png full-screen in order, one per VSYNC",
              "   (a looped image player; the scan engine does NOT do this yet).", "",
              "C. PATTERN SEQUENCE TAB: Pattern Source: Video port   Trigger Mode: Vsync",
              "   Variable exposure (Mode 4); per-pattern exposure == period:"]
    L.append("")
    L.append("   seq  name                 image  bit-depth  planes (click)")
    for e in m["sequence"]:
        L.append(f"   {e['seq']:3d}  {e['name']:<20} {e['image_index']:5d}  "
                 f"{e['bitdepth']:9d}  {','.join(e['planes']) if e['bitdepth'] == 1 else e['planes'][0] + '..' + e['planes'][-1]}")
    L += ["",
          "D. TRIGGER OUT (Trigger Controls subtab): Trigger Out 1 = normal polarity",
          "   (active high), rising/falling delay 0. TRIG_OUT_1 frames each pattern",
          "   exposure. Wire the EVM trigger connector's TRIG_OUT_1 -> camera Line0 (opto input, via the",
          "   GPIO cable) and ground -> the camera's opto ground. CHECK the voltage",
          "   levels first (EVM trigger out is 1.8/3.3 V; the opto input may need a",
          "   level shifter — else use non-isolated Line3 and set trigger_source).", "",
          "E. CAMERA (SpinView, or engine.capture.patternmode.configure_pyspin_trigger):",
          "   TriggerMode Off -> TriggerSelector FrameStart -> TriggerSource Line0 ->",
          "   TriggerActivation RisingEdge -> ExposureMode Timed ->",
          f"   ExposureTime {t.get('uniform', {}).get('camera_exposure_us', 'per frame')} us -> "
          "TriggerOverlap ReadOut -> TriggerMode On.",
          "   StreamBufferHandlingMode OldestFirst, StreamBufferCountManual >= "
          f"{len(m['sequence']) + 4}.", "",
          "F. Arm the camera (BeginAcquisition), THEN press Play in the GUI.",
          f"   Expect exactly {len(m['sequence'])} frames.", "",
          "G. BACK TO NORMAL: Stop the sequence, switch to Video mode (Operating Mode),",
          "   camera TriggerMode Off. If the DMD shows blue lines: HDMI_TIMING_FIX.md.", ""]
    return "\n".join(L)


# --------------------------------------------------------------- driver
def make_pack(preset_name: str = DEFAULT_PRESET, *, axes=("x",), video_port: bool = False,
              out_root: Path = DEFAULT_OUT, label: Optional[str] = None,
              vsync_us: int = VSYNC_60HZ_US) -> Path:
    preset = config.load_preset(preset_name)
    patterns = build_patterns(preset, axes)
    mode = "video_port" if video_port else "flash"
    if video_port:
        max8 = max(1, vsync_us // MIN_EXPOSURE_US[8])
        # the camera, not the DMD, limits patterns per frame (~4.5 ms readout)
        max1 = min(24, vsync_us // max(MIN_EXPOSURE_US[1], CAMERA_MIN_PERIOD_US))
        images, seq = pack(patterns, policy="grouped", max_per_image={8: max8, 1: max1})
    else:
        images, seq = pack(patterns, policy="sdk")
        if len(images) > FLASH_MAX_IMAGES:
            raise ValueError(f"{len(images)} images > {FLASH_MAX_IMAGES} flash slots")
    if len(seq) > LUT_MAX_ENTRIES:
        raise ValueError(f"{len(seq)} patterns > {LUT_MAX_ENTRIES} LUT entries")
    # the round trip is the contract: refuse to write a pack that does not invert
    for pat, back in zip(patterns, unpack(images, seq)):
        want = (pat["image"] > 0).astype(np.uint8) * 255 if pat["bitdepth"] == 1 else pat["image"]
        if not np.array_equal(want, back):
            raise RuntimeError(f"pack/unpack round trip failed for {pat['name']}")
    timing = suggest_timing(mode, seq, vsync_us=vsync_us)
    if not video_port:
        timing["warnings"] = flash_load_warnings(seq, timing)

    name = label or f"{preset.name}_{''.join(axes)}_{mode}"
    store = LibraryStore(Path(out_root))
    d = store.new_dir(label=name)                        # never an existing dir
    sub = d / ("frames_video" if video_port else "images")
    sub.mkdir()
    files = []
    for i, rgb in enumerate(images):
        f = sub / f"{'frame' if video_port else 'flash'}_{i:02d}.bmp"
        write_bmp(f, rgb)
        entry = {"index": i, "file": str(f.relative_to(d)).replace("\\", "/"),
                 "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
        if video_port:
            import cv2
            png = sub / f"frame_{i:02d}.png"
            if png.exists():
                raise FileExistsError(png)
            cv2.imwrite(str(png), rgb[..., ::-1])        # lossless, for a full-screen player
            entry["png"] = str(png.relative_to(d)).replace("\\", "/")
        files.append(entry)
    decode = {}
    for axis in axes:
        b = from_preset_bases(preset.bases, axis=axis)
        decode[axis] = {"basis": "hybrid", "params": b.params if hasattr(b, "params") else None,
                        "n_frames": b.n_frames(),
                        "order": [e["name"] for e in sorted(
                            (e for e in seq if e["axis"] == axis),
                            key=lambda e: e["basis_index"])]}
    manifest = {"format": "swiss_army_flash_pack/1", "name": name, "preset": preset.name,
                "geometry_id": preset.geometry_id, "mode": mode, "axes": list(axes),
                "created": datetime.now().isoformat(timespec="seconds"),
                "dmd": [DMD_W, DMD_H], "hdmi_invert_applied": False,
                "bit_plane_order": "bit 0..7 = G0..G7, 8..15 = R0..R7, 16..23 = B0..B7",
                "bmp_channel_note": "BMP stores B,G,R bytes; the green byte holds G0..G7",
                "images": files, "sequence": seq, "timing": timing, "decode": decode}
    LibraryStore.write_json(d / "manifest.json", manifest)
    with open(d / "procedure.txt", "x", encoding="utf-8") as f:
        f.write(procedure_text(manifest))
    return d


def load_pack(d: Path):
    """-> (manifest, images) from a written pack; BMP hashes checked."""
    d = Path(d)
    m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    images = []
    for im in m["images"]:
        f = d / im["file"]
        if hashlib.sha256(f.read_bytes()).hexdigest() != im["sha256"]:
            raise ValueError(f"{f}: sha256 mismatch (file changed since the pack was made)")
        images.append(read_bmp(f))
    return m, images


def verify_pack(d: Path, preset_name: Optional[str] = None) -> int:
    m, images = load_pack(d)
    if m.get("mode") == FASTSCAN_MODE:
        return verify_fastscan_pack(d)
    preset = config.load_preset(preset_name or m["preset"])
    patterns = build_patterns(preset, tuple(m["axes"]))
    back = unpack(images, m["sequence"])
    bad = 0
    for pat, b in zip(patterns, back):
        want = (pat["image"] > 0).astype(np.uint8) * 255 if pat["bitdepth"] == 1 else pat["image"]
        bad += not np.array_equal(want, b)
    return bad


# =============================================================== fastscan
# 1-bit soft-sync fast scan (docs/FLASH_FASTSCAN_4500.md). One 24-bit flash
# image per variant — full field, and zone-masked when zones are given — each
# holding the 16 one-bit slots of engine/bases/onebit.py on G0..G7, R0..R7
# (B0..B7 unused, zero). Both images use the SAME plane layout, so switching
# variant in the GUI is only a change of flash image index. The controller
# plays one image on its internal timer, on repeat; the camera free-runs
# faster than the pattern rate and frames are sorted in software.

FASTSCAN_MODE = "flash_fastscan"
FASTSCAN_FORMAT = "swiss_army_flash_pack/2"
FASTSCAN_OUT_KIND = "flashpacks"              # <library>/output/flashpacks/
FASTSCAN_PERIOD_US = 10000                    # 1-bit, exposure == period, internal timer
FASTSCAN_PERIOD_RANGE_US = (8000, 12000)      # the recommended window
FASTSCAN_CAMERA_EXPOSURE_US = 3000
FASTSCAN_CAMERA_FPS = 226.0                   # Blackfly S full frame Mono8
FASTSCAN_GUARD_US = 250                       # per slot edge: clock skew / timestamp jitter
FASTSCAN_LOOPS = 3                            # 3 x 160 ms = 0.48 s
# LED start = the HDMI bench preset. In pattern mode the selected LEDs are on for
# the whole exposure (exposure == period: ~100 % duty; 'White' = all three at
# once), in video mode each LED gets ~1/3 of the frame: the same setting draws
# ~3x the average LED power. start x FASTSCAN_LED_DUTY_FACTOR stays inside the
# continuous limits; a start above limit / factor is allowed (still <= the
# continuous limits) but warned about.
FASTSCAN_LED_START = {"r": 35, "g": 50, "b": 45}
FASTSCAN_LED_DUTY_FACTOR = 3.0
FASTSCAN_STUDY_SIGMAS = (0.5, 0.75, 1.0, 1.5, 2.0)
FASTSCAN_ZONE_MARGIN_PX = 6
FASTSCAN_DEPTH_MM = 780.0
NOMINAL_SCALE = {"cam_px_per_col": 1.47, "mm_per_col": 1.87,
                 "source": "nominal: measured on the bench faces (1.47 camera px / column) "
                           "and the stereo calibration at 780 mm (1.87 mm / column)"}


def library_root() -> Path:
    from engine.api import DEFAULT_ROOT
    return Path(DEFAULT_ROOT)


def fastscan_timing(period_us: int = FASTSCAN_PERIOD_US,
                    camera_exposure_us: int = FASTSCAN_CAMERA_EXPOSURE_US,
                    camera_fps: float = FASTSCAN_CAMERA_FPS, loops: int = FASTSCAN_LOOPS,
                    n_slots: int = 16, guard_us: int = FASTSCAN_GUARD_US) -> dict:
    """Internal-timer, repeat, 1-bit, exposure == period; free-running camera.
    A camera frame is CLEAN for a slot when its whole exposure lies inside
    the slot with guard_us to spare at both ends; the camera frame period
    must be <= period - exposure - 2 guard, else a slot can be missed."""
    period_us, camera_exposure_us = int(period_us), int(camera_exposure_us)
    validate_timing(1, period_us, period_us)
    if loops < 1:
        raise ValueError(f"loops must be >= 1, got {loops}")
    frame_us = 1e6 / float(camera_fps)
    window = period_us - camera_exposure_us - 2 * guard_us
    if window < frame_us:
        raise TimingError(
            f"pattern period {period_us} us leaves a {window} us clean-start window for a "
            f"{camera_exposure_us} us camera exposure; the camera frame period "
            f"{frame_us:.0f} us ({camera_fps:g} fps) is longer, so a slot could get no clean "
            "frame. Lengthen the pattern period, shorten the camera exposure or raise the "
            "camera frame rate (smaller ROI)")
    loop_ms = n_slots * period_us / 1000.0
    warn = []
    lo, hi = FASTSCAN_PERIOD_RANGE_US
    if not lo <= period_us <= hi:
        warn.append(f"pattern period {period_us} us is outside the recommended {lo}..{hi} us")
    return {"mode": FASTSCAN_MODE, "source": "flash", "trigger": "internal",
            "repeat": True, "bit_depth": 1,
            "pattern_exposure_us": period_us, "pattern_period_us": period_us,
            "n_slots": n_slots, "loop_ms": loop_ms, "capture_loops": int(loops),
            "capture_ms": round(loops * loop_ms, 1),
            "camera": {"mode": "free-run (no trigger cable)",
                       "exposure_us": camera_exposure_us, "fps_assumed": float(camera_fps),
                       "frame_period_us": round(frame_us, 1),
                       "max_frame_period_us": window, "guard_us": guard_us,
                       "clean_frames_per_slot_per_loop": round(window / frame_us, 2),
                       "buffering": "every frame with its timestamp (OldestFirst), not "
                                    "NewestOnly"},
            "rules": ["1-bit: exposure == period is legal (DLPU010G); no dark gap, so every "
                      "clean frame sees the same light",
                      "camera frame period <= period - camera exposure - 2 x guard: at least "
                      "one clean frame per slot per loop, whatever the phase",
                      "capture any window of capture_loops x loop_ms: every slot is covered "
                      "capture_loops times (fold by time; no need to start at the sync)"],
            "warnings": warn}


def fastscan_led(led: Optional[dict] = None) -> dict:
    """The LED block of a fastscan manifest. Refuses currents above the
    CONTINUOUS limits (rules.validate_led); warns when a channel is above
    limit / FASTSCAN_LED_DUTY_FACTOR, where pattern mode's full duty draws
    more average LED power than video mode does at the continuous limit."""
    from engine import rules
    led = {k: int(v) for k, v in {**FASTSCAN_LED_START, **(led or {})}.items()}
    rules.validate_led({"mode": "continuous", **led})
    f = FASTSCAN_LED_DUTY_FACTOR
    duty_safe = {ch: int(lim // f) for ch, lim in rules.LED_AVG_MAX.items()}
    warn = [f"LED {ch.upper()}={led[ch]} > {duty_safe[ch]} (= continuous limit "
            f"{rules.LED_AVG_MAX[ch]} / {f:g}): pattern mode keeps the LED on for the whole "
            "exposure, ~3x the per-LED duty of video mode at the same setting. Allowed (within "
            "the continuous limit), but run the fan at max and Stop the sequence between scans"
            for ch in ("r", "g", "b") if led.get(ch) is not None and led[ch] > duty_safe[ch]]
    return {"select": "White (red + green + blue together) on every slot - the colour mix of "
                      "the HDMI 'bw' scans the stereo calibration was made with",
            "start": led, "mode": "continuous",
            "limits_continuous": dict(rules.LED_AVG_MAX),
            "limits_short_burst_gui_only": dict(rules.LED_BURST_MAX),
            "duty_factor_vs_video": f, "duty_adjusted_max": duty_safe,
            "video_equivalent_of_start": {ch: int(round(v * f)) for ch, v in led.items()},
            "warnings": warn,
            "note": "the tool refuses anything above the continuous limits; the short-burst "
                    "limits are for the separate short-burst GUI only and never for a repeating "
                    "sequence. Start at the HDMI bench values; raise only if the camera signal "
                    "is too low. Fan at max; Stop the sequence between scans"}


def plane_swap_play_order(n_slots: int = 16) -> list:
    """The slot order the controller would play if the GUI's G and R planes
    read the other two bytes of the BMP than this pack assumes (G0..G7 =
    green byte). Slots 0..7 sit on G, 8..15 on R: a swap plays 8..15 then
    0..7 - a cyclic rotation by 8 of the repeating loop, which the soft-sync
    sorter absorbs (it anchors on the sync markers, not on the start)."""
    if n_slots != 16:
        raise ValueError("the fastscan layout has 16 slots (G0..G7, R0..R7)")
    return list(range(8, 16)) + list(range(8))


# ------------------------------------------------------------------ zones
def _load_npy(path) -> np.ndarray:
    return np.load(Path(path), allow_pickle=False)


def zone_from_rings(ring_paths, margin_px: int = FASTSCAN_ZONE_MARGIN_PX,
                    shape=(DMD_H, DMD_W)):
    """face_dmd_ring.npy files ((N, 2) DMD col, row points on a face rim) ->
    filled ellipses (fitted, + margin_px on each side) -> (mask, info)."""
    import cv2
    mask = np.zeros(shape, np.uint8)
    info = []
    for p in ring_paths:
        pts = np.asarray(_load_npy(p), np.float32)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 5:
            raise ValueError(f"{p}: need an (N >= 5, 2) array of DMD (col, row) points, "
                             f"got {pts.shape}")
        (cx, cy), (a, b), ang = cv2.fitEllipse(pts)
        cv2.ellipse(mask, ((cx, cy), (a + 2 * margin_px, b + 2 * margin_px), ang), 1, -1)
        info.append({"kind": "ring", "file": str(p), "centre_col_row": [round(cx, 2), round(cy, 2)],
                     "axes_cols_rows": [round(a, 2), round(b, 2)], "angle_deg": round(ang, 2),
                     "margin_px": margin_px, "sha256": hashlib.sha256(Path(p).read_bytes()).hexdigest()})
    return mask.astype(bool), info


def zone_band_from_camera(cam_mask_path, field_path, cam_dilate: int = 9, dmd_dilate: int = 5,
                          close_px: int = 7, min_component_px: int = 40,
                          shape=(DMD_H, DMD_W)):
    """A camera-space band (bool mask, camera shape) -> DMD mask through the
    decoded correspondence of a BOTH-axes scan (field.npz with corr_x and
    corr_y), as the HDMI fast scan builds its tube band; then closed
    (close_px) and stripped of specks under min_component_px (stray
    camera-mask pixels). -> (mask, info)."""
    import cv2
    cam = _load_npy(cam_mask_path).astype(np.uint8)
    with np.load(Path(field_path), allow_pickle=False) as F:
        if "corr_x" not in F.files or "corr_y" not in F.files:
            raise ValueError(f"{field_path}: needs corr_x AND corr_y (an axis 'xy' scan) to map "
                             "camera pixels to DMD (col, row)")
        cx, cy = F["corr_x"], F["corr_y"]
    if cam.shape != cx.shape:
        raise ValueError(f"camera mask {cam.shape} and field {cx.shape} differ in shape")
    band = cv2.dilate(cam, np.ones((cam_dilate, cam_dilate), np.uint8)).astype(bool)
    v, u = np.nonzero(band & np.isfinite(cx) & np.isfinite(cy))
    du, dv = np.round(cx[v, u]).astype(int), np.round(cy[v, u]).astype(int)
    ok = (du >= 0) & (du < shape[1]) & (dv >= 0) & (dv < shape[0])
    m = np.zeros(shape, np.uint8)
    m[dv[ok], du[ok]] = 1
    m = cv2.dilate(m, np.ones((dmd_dilate, dmd_dilate), np.uint8))
    if close_px > 1:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((close_px, close_px), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= min_component_px
    dropped = int((~keep[1:]).sum()) if n > 1 else 0
    m = keep[lab]
    info = {"kind": "band", "camera_mask": str(cam_mask_path), "field": str(field_path),
            "camera_mask_sha256": hashlib.sha256(Path(cam_mask_path).read_bytes()).hexdigest(),
            "field_sha256": hashlib.sha256(Path(field_path).read_bytes()).hexdigest(),
            "camera_px": int(band.sum()), "mapped_px": int(ok.sum()),
            "cam_dilate": cam_dilate, "dmd_dilate": dmd_dilate, "close_px": close_px,
            "min_component_px": min_component_px, "specks_dropped": dropped,
            "dmd_px": int(m.sum())}
    return m.astype(bool), info


def zone_from_masks(mask_paths, shape=(DMD_H, DMD_W)):
    """Any DMD-shaped mask .npy (bool / 0..255) files -> union."""
    m = np.zeros(shape, bool)
    info = []
    for p in mask_paths:
        a = _load_npy(p)
        if a.shape != tuple(shape):
            raise ValueError(f"{p}: mask is {a.shape}, the DMD is {tuple(shape)}")
        m |= a > 0
        info.append({"kind": "mask", "file": str(p), "px": int((a > 0).sum()),
                     "sha256": hashlib.sha256(Path(p).read_bytes()).hexdigest()})
    return m, info


def build_zone(rings=(), masks=(), band_cam=None, band_field=None,
               margin_px: int = FASTSCAN_ZONE_MARGIN_PX):
    """-> (mask or None, info). None when no zone source is given."""
    if bool(band_cam) != bool(band_field):
        raise ValueError("a tube band needs both --zone-band-cam and --zone-band-field")
    if not (rings or masks or band_cam):
        return None, None
    m = np.zeros((DMD_H, DMD_W), bool)
    parts = []
    if rings:
        r, i = zone_from_rings(rings, margin_px)
        m |= r
        parts += i
    if masks:
        r, i = zone_from_masks(masks)
        m |= r
        parts += i
    if band_cam:
        r, i = zone_band_from_camera(band_cam, band_field)
        m |= r
        parts.append(i)
    if not m.any():
        raise ValueError("the zone mask is empty")
    rr, cc = np.nonzero(m)
    return m, {"sources": parts, "lit_px": int(m.sum()),
               "lit_pct": round(100.0 * m.mean(), 3),
               "bbox_col_row": [int(cc.min()), int(rr.min()), int(cc.max()), int(rr.max())]}


# ----------------------------------------------------------- calibration
def calibration_scale(cal_path, depth_mm: float = FASTSCAN_DEPTH_MM) -> dict:
    """Camera px per DMD column and range (mm) per DMD column at depth_mm on
    the camera's principal ray, fronto-parallel, from a stereo calibration
    (folder or stereo_calibration.json). The 1-bit fine fringes are drawn on
    the plain column grid, so a calibration with a diamond row parity is
    refused."""
    from engine.calibrate.stereo import STEREO_FILE, StereoCalibration
    from engine.reconstruct.triangulate import camera_rays, ray_plane
    f = Path(cal_path)
    if f.is_dir():
        f = f / STEREO_FILE
    cal = StereoCalibration.from_json(json.loads(f.read_text(encoding="utf-8")), f.parent)
    if cal.diamond_parity is not None:
        raise ValueError(f"{cal.id}: diamond_parity={cal.diamond_parity}; the fastscan 1-bit "
                         "fringes assume the square column grid (parity null)")
    u0, v0 = float(cal.K_cam[0, 2]), float(cal.K_cam[1, 2])
    us = u0 + np.arange(-100.0, 100.1, 20.0)
    d = camera_rays(cal, None, np.stack([us, np.full_like(us, v0)], -1))
    X = d * (depth_mm / d[:, 2:3])
    Xp = X @ cal.R.T + np.ravel(cal.t)
    Kp = cal.K_proj
    yv = Xp[:, 1] / Xp[:, 2]
    cols = Kp[0, 0] * Xp[:, 0] / Xp[:, 2] + Kp[0, 1] * yv + Kp[0, 2]
    slope = float(np.polyfit(us, cols, 1)[0])          # DMD columns per camera px
    c0 = float(cols[len(cols) // 2])
    dc = camera_rays(cal, None, np.array([[u0, v0]]))
    X0 = ray_plane(dc, np.array([c0]), cal)
    X1 = ray_plane(dc, np.array([c0 + 0.01]), cal)
    return {"cam_px_per_col": round(1.0 / abs(slope), 4),
            "mm_per_col": round(float(np.linalg.norm(X1 - X0)) / 0.01, 4),
            "depth_mm": depth_mm, "centre_col": round(c0, 1),
            "camera_mm_per_px": round(depth_mm / float(cal.K_cam[0, 0]), 4),
            "calibration_id": cal.id, "geometry_id": cal.geometry_id,
            "diamond_parity": cal.diamond_parity,
            "source": f"stereo calibration {cal.id}: camera principal ray at {depth_mm:g} mm, "
                      "fronto-parallel plane"}


# ------------------------------------------------------------------ pack
def _contact_sheet(pats: list, cols: int = 4, scale: float = 0.25) -> np.ndarray:
    import cv2
    h, w = int(DMD_H * scale), int(DMD_W * scale)
    rows = -(-len(pats) // cols)
    sheet = np.full((rows * (h + 4) + 4, cols * (w + 4) + 4), 64, np.uint8)
    for i, p in enumerate(pats):
        r, c = divmod(i, cols)
        small = cv2.resize(p["image"], (w, h), interpolation=cv2.INTER_AREA)
        y, x = 4 + r * (h + 4), 4 + c * (w + 4)
        sheet[y:y + h, x:x + w] = small
        cv2.putText(sheet, f"{p['slot']} {p['name']}", (x + 3, y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, 128, 1, cv2.LINE_AA)
    return sheet


def _write_png(path: Path, img: np.ndarray) -> Path:
    import cv2
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError(f"PNG encode failed for {path}")
    with open(path, "xb") as f:
        f.write(buf.tobytes())
    return path


def _read_png(path: Path) -> np.ndarray:
    import cv2
    a = cv2.imdecode(np.frombuffer(Path(path).read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED)
    if a is None:
        raise ValueError(f"{path}: not a readable PNG")
    return a


def fastscan_layout(zone_mask: Optional[np.ndarray] = None):
    """-> (variants [(name, zone_mask|None)], patterns per variant [list],
    images [(H, W, 3) uint8], sequence [dict]). One image per variant,
    the SDK packing order (slot k -> bit k: G0..G7, R0..R7)."""
    from engine.bases import onebit as ob
    variants = [("full", None)] + ([("zone", zone_mask)] if zone_mask is not None else [])
    per, images, seq = [], [], []
    for vi, (vname, zm) in enumerate(variants):
        pats = ob.fastscan_patterns(zm)
        items = [{"name": f"{vname}_{p['name']}", "axis": "x", "role": p["role"],
                  "basis_index": p["decode_index"], "bitdepth": 1, "image": p["image"]}
                 for p in pats]
        ims, sq = pack(items, policy="sdk")
        if len(ims) != 1:
            raise RuntimeError(f"{vname}: {len(ims)} images; the 16 slots must fit one image")
        for p, e in zip(pats, sq):
            e.update({"seq": len(seq), "variant": vname, "slot": p["slot"], "image_index": vi,
                      "decode_index": p["decode_index"], "period_px": p["period_px"],
                      "step": p["step"], "method": p["method"], "meaning": p["meaning"],
                      "zone": p["zone"], "fill": round(p["fill"], 4)})
            seq.append(e)
        per.append(pats)
        images.append(ims[0])
    return variants, per, images, seq


def _check_round_trip(images: list, seq: list, per: list) -> dict:
    """Every plane unpacks to its pattern exactly; unused planes are zero."""
    back = unpack(images, seq)
    by_var = {}
    for e, b in zip(seq, back):
        by_var.setdefault(e["image_index"], []).append((e, b))
    for vi, pats in enumerate(per):
        pairs = by_var[vi]
        for (e, b), p in zip(pairs, pats):
            if e["slot"] != p["slot"] or not np.array_equal(b, p["image"]):
                raise RuntimeError(f"round trip failed: image {vi} slot {p['slot']} {p['name']}")
        rgb = images[vi]
        packed = (rgb[..., 1].astype(np.uint32) | (rgb[..., 0].astype(np.uint32) << 8)
                  | (rgb[..., 2].astype(np.uint32) << 16))
        used = 0
        for e, _ in pairs:
            used |= 1 << e["bit_position"]
        if np.any(packed & np.uint32(~used & 0xFFFFFF)):
            raise RuntimeError(f"image {vi}: bits outside the listed planes are set")
    return {"patterns_checked": len(seq), "exact": True,
            "unused_planes_zero": True}


def make_fastscan_pack(*, zone_rings=(), zone_masks=(), band_cam=None, band_field=None,
                       margin_px: int = FASTSCAN_ZONE_MARGIN_PX, calibration=None,
                       depth_mm: float = FASTSCAN_DEPTH_MM, study: bool = False,
                       out_root: Optional[Path] = None, library: Optional[Path] = None,
                       label: Optional[str] = None, geometry_id: Optional[str] = None,
                       led: Optional[dict] = None, period_us: int = FASTSCAN_PERIOD_US,
                       camera_exposure_us: int = FASTSCAN_CAMERA_EXPOSURE_US,
                       camera_fps: float = FASTSCAN_CAMERA_FPS, loops: int = FASTSCAN_LOOPS,
                       singles: bool = False) -> Path:
    """Build the 1-bit fast-scan pack. Files only; never touches the rig.
    out_root: a plain folder for the pack; else the library output bin
    (library or the engine's default library)/output/flashpacks/, recorded
    in output/index.csv."""
    from engine import rules
    from engine.bases import onebit as ob
    zone, zinfo = build_zone(zone_rings, zone_masks, band_cam, band_field, margin_px)
    scale = calibration_scale(calibration, depth_mm) if calibration else dict(NOMINAL_SCALE)
    if calibration:
        if geometry_id is None:
            geometry_id = scale.get("geometry_id")
        rules.require_same_geometry(geometry_id, scale.get("geometry_id"),
                                    "fastscan pack vs its stereo calibration")
    led_block = fastscan_led(led)                          # continuous limits, always
    timing = fastscan_timing(period_us, camera_exposure_us, camera_fps, loops)
    variants, per, images, seq = fastscan_layout(zone)
    rt = _check_round_trip(images, seq, per)
    for e in seq:
        e.update({"exposure_us": timing["pattern_exposure_us"],
                  "period_us": timing["pattern_period_us"], "trigger": "internal",
                  "led": "red+green+blue", "invert": False})
    std = ob.study(FASTSCAN_STUDY_SIGMAS, cam_px_per_col=scale["cam_px_per_col"],
                   mm_per_col=scale["mm_per_col"], parities=(None, 0, 1),
                   fine_prefilter=(ob.FINE_PREFILTER_PX,)) if study else None

    name = label or ("fastscan_" + "_".join(v for v, _ in variants))
    if out_root is not None:
        store = LibraryStore(Path(out_root))
        d = store.new_dir(label=name)
        lib_store = None
    else:
        lib_store = LibraryStore(Path(library) if library else library_root())
        d = lib_store.new_dir("output", FASTSCAN_OUT_KIND, label=name)
    sub = d / "images"
    sub.mkdir()
    files = []
    for vi, ((vname, _), rgb) in enumerate(zip(variants, images)):
        f = sub / f"flash_{vi:02d}_{vname}.bmp"
        write_bmp(f, rgb)
        files.append({"index": vi, "variant": vname,
                      "file": str(f.relative_to(d)).replace("\\", "/"),
                      "sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
                      "planes_used": "G0-G7, R0-R7 (slot k = bit k)",
                      "planes_free": "B0-B7 (zero)"})
    # the contract, from disk: re-read the BMPs and unpack every plane
    rt_disk = _check_round_trip([read_bmp(d / im["file"]) for im in files], seq, per)
    zone_block = None
    if zone is not None:
        zf = _write_png(d / "zone_mask.png", zone.astype(np.uint8) * 255)
        zone_block = dict(zinfo, file=zf.name,
                          sha256=hashlib.sha256(zf.read_bytes()).hexdigest())
    for (vname, _), pats in zip(variants, per):
        _write_png(d / f"preview_{vname}.png", _contact_sheet(pats))
        if singles:
            sd = d / "singles" / vname
            sd.mkdir(parents=True)
            for p in pats:
                one = np.repeat(p["image"][..., None], 3, 2)
                write_bmp(sd / f"slot_{p['slot']:02d}_{p['name']}.bmp", one)

    b = ob.basis()
    manifest = {
        "format": FASTSCAN_FORMAT, "mode": FASTSCAN_MODE, "name": name,
        "created": datetime.now().isoformat(timespec="seconds"),
        "geometry_id": geometry_id, "dmd": [DMD_W, DMD_H], "diamond_parity": None,
        "hdmi_invert_applied": False,
        "bit_plane_order": "bit 0..7 = G0..G7, 8..15 = R0..R7, 16..23 = B0..B7",
        "bmp_channel_note": "BMP stores B,G,R bytes; the green byte holds G0..G7",
        "images": files,
        "variants": {vname: {"image_index": vi, "slots": 16,
                             "zone": zone_block if vname == "zone" else None,
                             "note": ("decode frames lit only inside the zone; sync frames "
                                      "full field") if vname == "zone" else "full field"}
                     for vi, (vname, _) in enumerate(variants)},
        "sequence": seq,
        "timing": timing,
        "led": led_block,
        "planes": {"assumed": "BMP green byte = G0..G7, red byte = R0..R7, blue byte = "
                              "B0..B7 (bit 0 = LSB); slot k = bit k",
                   "if_gui_swaps_g_and_r": {"play_order_slots": plane_swap_play_order(),
                                            "effect": "the repeating loop is rotated by 8 "
                                                      "slots; the sync markers still find "
                                                      "the start and order"},
                   "operator_check": "one-pattern sequence of G1 (exposure = period = the "
                                     "pack's, Play Once off): horizontal bars = as assumed; "
                                     "vertical stripes ~80 mirrors apart = G/R swapped "
                                     "(harmless, note it); a fine 8-mirror texture = bits "
                                     "reversed (enter planes G7..G0, R7..R0)"},
        "sync": {"slots": {"sync_white": 0, "sync_code": 1, "black": 3},
                 "recipe": [
                     "per frame: timestamp (exposure start) + sync_signature(frame) = mean, "
                     "grad_y, grad_x, orientation = grad_y / grad_x (onebit.sync_signature)",
                     f"pattern period P = {timing['pattern_period_us']} us, loop = 16 P; refine "
                     "the loop period from the repeats of the sync_code frames",
                     "sync_code (slot 1) = the frames with orientation >> 1 (horizontal bars); "
                     "sync_white (slot 0) = the full-field white immediately before it (as "
                     "bright as slot 2 in the full image; the only full-field white in the zone "
                     "image); black (slot 3) = the darkest",
                     "engine: engine/capture/softsync.py sort_frames (layout_from_manifest "
                     "checks this manifest against the engine's loop first)",
                     "t0 = start of slot 0; slot(t) = floor((t - t0) / P) mod 16",
                     "keep a frame only if its exposure [t, t + E] lies inside one slot with "
                     "guard_us to spare at both ends; average the kept frames per slot",
                     "check: slots 4..7 have near-equal means, slot 3 darkest, slot 2 ~ slot 0 "
                     "inside the zone; else reject the capture"]},
        "decode": {"basis": "multifreq", "params": dict(b.params),
                   "n_frames": b.n_frames(), "decode_slots": list(ob.DECODE_SLOTS),
                   "order": [ob.SLOTS[s] for s in ob.DECODE_SLOTS],
                   "prefilter": {"sigma_camera_px": ob.PREFILTER_PX,
                                 "frames": "mid + coarse fringes only (decode indices 6..13); "
                                           "white, black, fine untouched",
                                 "function": "engine.bases.onebit.prefilter_frames / decode",
                                 "fine_sigma_camera_px_recommended": ob.FINE_PREFILTER_PX,
                                 "fine_note": "optional (fine_px=): at a sharply focused "
                                              "projector a 1 camera px Gaussian on the fine "
                                              "frames removes most of the dither texture "
                                              "(simulated 0.018 -> 0.004 column rms) for ~0.36 "
                                              "mm of lateral blur; off by default"}},
        "design": {"pattern_version": ob.PATTERN_VERSION,
                   "fine": {"period_px": 8, "method": "column-exact ordered dither",
                            "levels_q4": list(ob.FINE_LEVELS_Q4),
                            "row_order_q4": list(ob.ROW_ORDER_Q4), **ob.fine_harmonics()},
                   "mid_coarse": {"periods_px": [80, 1280],
                                  "method": "Floyd-Steinberg (serpentine) strip, step k = "
                                            "window shifted by k T/4 columns (exact translate)"},
                   "sync_code": {"bar_rows": ob.SYNC_BAR_ROWS},
                   "diamond": "calibration parity null: on the diamond DMD the fine level "
                              "reads a constant ~0.043 column (~0.08 mm) off the HDMI 8-bit "
                              "fringes, sign = the row parity; relative heights unaffected "
                              "(docs/FLASH_FASTSCAN_4500.md)"},
        "scale": scale,
        "round_trip": {"in_memory": rt, "from_disk": rt_disk},
    }
    if std is not None:
        manifest["study"] = std
    with open(d / "procedure.txt", "x", encoding="utf-8") as fh:
        fh.write(fastscan_procedure_text(manifest))
    LibraryStore.write_json(d / "manifest.json", manifest)      # last: the completeness marker
    if lib_store is not None:
        lib_store.record_output(FASTSCAN_OUT_KIND, d / "manifest.json", label=name,
                                geometry_id=geometry_id or "")
    return d


def verify_fastscan_pack(d: Path) -> int:
    """Re-read a fastscan pack from disk, regenerate every slot (zone mask
    from the pack) and count planes that differ (0 = the round trip holds;
    set bits outside the listed planes count as one failure per image)."""
    from engine.bases import onebit as ob
    d = Path(d)
    m, images = load_pack(d)
    ver = (m.get("design") or {}).get("pattern_version")
    if ver != ob.PATTERN_VERSION:
        raise ValueError(f"{d.name}: pattern_version {ver} but engine/bases/onebit.py makes "
                         f"version {ob.PATTERN_VERSION}; the planes cannot be regenerated to "
                         "compare (the BMP hashes were still checked)")
    zone = None
    zb = (m["variants"].get("zone") or {}).get("zone")
    if zb:
        zf = d / zb["file"]
        if hashlib.sha256(zf.read_bytes()).hexdigest() != zb["sha256"]:
            raise ValueError(f"{zf}: sha256 mismatch")
        zone = _read_png(zf) > 0
    bad = 0
    for vname, v in m["variants"].items():
        pats = ob.fastscan_patterns(zone if vname == "zone" else None)
        ents = [e for e in m["sequence"] if e["variant"] == vname]
        back = unpack(images, ents)
        for e, b in zip(ents, back):
            bad += not np.array_equal(b, pats[e["slot"]]["image"])
        rgb = images[v["image_index"]]
        packed = (rgb[..., 1].astype(np.uint32) | (rgb[..., 0].astype(np.uint32) << 8)
                  | (rgb[..., 2].astype(np.uint32) << 16))
        used = 0
        for e in ents:
            used |= 1 << e["bit_position"]
        bad += bool(np.any(packed & np.uint32(~used & 0xFFFFFF)))
    return bad


def fastscan_procedure_text(m: dict) -> str:
    """The click list for one pack, in the 4500 MKII GUI's own words."""
    t = m["timing"]
    led = m["led"]
    s = led["start"]
    lim = led["limits_continuous"]
    L = [f"FAST-SCAN 1-BIT PACK  {m['name']}   ({m['created']})", "=" * 76, "",
         "Nothing here was sent to the projector. A human does every step below in the",
         "4500 MKII GUI. Full guide, with the reasons: docs/FLASH_FASTSCAN_4500.md", ""]
    for w in t.get("warnings", []) + led.get("warnings", []):
        L += ["!! WARNING: " + w, ""]
    L += ["0. HDMI scans idle. Modes are changed only here, in the GUI, never while the engine",
          "   is capturing (the engine never sends the projector a command).", "",
          "1. BACK UP FIRST (once per unit; nothing is changed yet)",
          "   - System Control: write down Firmware Version and Firmware Tag.",
          "   - LED Driver Control: write down the LED Current R / G / B.",
          "   - Save Solution: keep the .ini (today's video-mode settings).",
          "   - Image / Firmware tab: open the firmware link (\"Download the latest Firmware",
          "     release from ...\": the DLPC350 firmware page) and save the firmware .bin next to",
          "     the .ini. Its version MUST equal the Firmware Version written down above; if it",
          "     does not, stop: the unit runs its supplier's firmware - get that .bin from the",
          "     supplier. If the unit came with its own firmware .bin, that file is the preferred",
          "     base and rollback image (the unit's notes: supplier firmware only). The GUI",
          "     cannot read the flash back: the saved .bin IS the backup.",
          "",
          "2. ADD THE IMAGES (Image / Firmware -> Firmware Build)",
          "   - Firmware File: Browse -> the backup .bin (the supplier's .bin when you have it).",
          "     Note \"# Original Images\" = N.",
          "   - Add these 24-bit 912x1140 BMPs AFTER the original images, in this order:"]
    for im in m["images"]:
        L.append(f"       {im['file']:<34} -> flash index N+{im['index']}   ({im['variant']})")
    L += ["     (\"Total Images\" must read N + " + str(len(m["images"])) + "; the images are 3 MB "
          "uncompressed, the GUI compresses them.)",
          "   - Add Firmware Tag: e.g. fastscan1bit (32 characters at most).",
          "   - Illumination Control: RGB Color.  Select .ini File: leave empty.",
          "   - Save Updates -> a new .bin (do NOT overwrite the backup).",
          "   - Firmware Upload subtab: Browse -> the new .bin; tick Skip Bootloader Update;",
          "     Upload. Wait for the end; do not unplug, power off or close the GUI.",
          "   - After it reconnects: System Control shows the new Firmware Tag.", "",
          "3. PATTERN SEQUENCE (Operating Mode: Pattern Sequence; Sequence Settings)",
          "   Pattern Source: Flash        Trigger Mode: Internal/External",
          f"   Pattern Exposure (us): {t['pattern_exposure_us']}    Pattern Period (us): "
          f"{t['pattern_period_us']}   (equal: legal for 1-bit)",
          "   Clear the pattern list, then for EACH row below, in this order:",
          "     Choose Color: White      trigger: Internal Trigger",
          "     Image Index: N (full)  or  N+1 (zone)      Bit Depth: 1",
          "     Bit Plane: the row's plane   Invert Pattern Data: off   Clear DMD after "
          "exposure: off",
          "     -> Add Pattern to Sequence", "",
          "     slot  plane  meaning"]
    for e in m["sequence"]:
        if e["image_index"] == 0:
            L.append(f"     {e['slot']:4d}  {e['planes'][0]:<5}  {e['meaning']}")
    L += ["",
          "   Play Once: OFF (the loop repeats until Stop).  Send -> Validate Sequence: every",
          "   indicator clear, \"Sequence Validated\".",
          f"   Loop = 16 x {t['pattern_period_us']} us = {t['loop_ms']:g} ms. The zone image uses "
          "the SAME planes on its own Image Index.",
          "   Save Solution (optional): fastscan_full.ini / fastscan_zone.ini to switch later.", "",
          "   PLANE CHECK (once, before the first scan): a ONE-pattern sequence, Image Index N,",
          "   Bit Depth 1, Bit Plane G1, the same Exposure = Period (10000 us), Play Once off,",
          "   Send, Validate, Play (one pattern on repeat = a still image):",
          "     HORIZONTAL BARS (16 mirror rows on / 16 off) = as this pack assumes.",
          "     Then R0: vertical stripes ~80 mirrors apart (mid step 0). And G2: all white",
          "     (full) / only the zones (zone image, index N+1).",
          "     G1 shows vertical stripes ~80 mirrors apart = the GUI reads G and R the other",
          "       way round: harmless (the loop starts 8 slots later; the sync markers fix the",
          "       order) - note it.",
          "     G1 shows a fine grey texture (8-mirror stripes) = bits reversed in the byte:",
          "       enter the 16 rows with planes G7..G0, R7..R0 instead (slot k on G(7-k),",
          "       R(15-k)).",
          "     R0 black = the GUI's R is not the BMP's red byte: use the --singles route",
          "       (docs/FLASH_FASTSCAN_4500.md section 4). Anything else (a splash image, all",
          "       black): wrong Image Index - step it until G1 shows the bars.",
          "   Stop, then build the 16-pattern sequence above.", "",
          f"4. LED Driver Control -> LED Current: R{s['r']} G{s['g']} B{s['b']} (start) -> Set.",
          f"   Never above the continuous limits R{lim['r']} G{lim['g']} B{lim['b']}. Pattern mode "
          "keeps the LEDs on the",
          "   whole exposure (~3x video-mode duty): above R{r} G{g} B{b} the LEDs run hotter than "
          "HDMI at".format(**led["duty_adjusted_max"]),
          "   the limit. Fan at max. The short-burst values are never used for a repeating loop.",
          "",
          "5. SCAN: Pattern Sequence Start/Stop/Pause -> Play; run the fast-scan capture",
          f"   (free-running camera, exposure {t['camera']['exposure_us']} us, >= "
          f"{1e6 / t['camera']['max_frame_period_us']:.0f} fps, every frame kept;",
          f"   ~{t['capture_ms']:g} ms full frame, less in a camera window); then Stop.",
          "     scan.bat --preset <the HDMI scan's preset> scan-flash <label>",
          "         --pack <this folder> --variant "
          + ("zone" if "zone" in (m.get("variants") or {}) else "full")
          + " [--roi X0,Y0,X1,Y1]     (--dry-run first: the plan, opens nothing)",
          "   The engine only opens the camera (docs/FLASH_SOFTSYNC.md).", "",
          "6. BACK TO HDMI: Stop -> Operating Mode: Video Mode -> Input Configuration, Source",
          "   Select: Parallel RGB -> the desktop is on the DMD -> LED Current back to the values",
          "   from step 1 -> Set. Blue lines / test pattern: docs/HDMI_TIMING_FIX.md.", "",
          "7. ROLLBACK: Image / Firmware -> Firmware Upload -> Browse -> the backup .bin -> Upload",
          "   -> System Control shows the original Firmware Version / Tag -> Video Mode -> Apply",
          "   Solution (the step-1 .ini) or re-enter the LED currents.", ""]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--preset", default=DEFAULT_PRESET)
    ap.add_argument("--axes", default="x", help="x, y or x,y (y = horizontal stripes)")
    ap.add_argument("--video-port", action="store_true",
                    help="pack as 912x1140 frames for Pattern Sequence + Video port + VSYNC")
    ap.add_argument("--out", default=None,
                    help=f"pack folder root (default {DEFAULT_OUT}; --fastscan: the library "
                         "output bin output/flashpacks/)")
    ap.add_argument("--label", default=None)
    ap.add_argument("--unpack-check", default=None, metavar="PACK_DIR")
    fs = ap.add_argument_group("fastscan (1-bit, internal timer, free-running camera)")
    fs.add_argument("--fastscan", action="store_true", help="build the 1-bit fast-scan pack")
    fs.add_argument("--zone-ring", action="append", default=[], metavar="NPY",
                    help="face_dmd_ring.npy (DMD col,row rim points); ellipse + margin")
    fs.add_argument("--zone-mask", action="append", default=[], metavar="NPY",
                    help="any DMD-shaped mask .npy to add to the zone")
    fs.add_argument("--zone-band-cam", default=None, metavar="NPY",
                    help="camera-space band mask (e.g. the cooling-tube band)")
    fs.add_argument("--zone-band-field", default=None, metavar="NPZ",
                    help="field.npz with corr_x + corr_y mapping that band to the DMD")
    fs.add_argument("--zone-margin", type=int, default=FASTSCAN_ZONE_MARGIN_PX)
    fs.add_argument("--calibration", default=None, metavar="DIR",
                    help="stereo calibration folder: camera px / column and mm / column")
    fs.add_argument("--depth-mm", type=float, default=FASTSCAN_DEPTH_MM)
    fs.add_argument("--study", action="store_true",
                    help="simulate the decode error per level (a few s) into the manifest")
    fs.add_argument("--led", default=None, metavar="R,G,B",
                    help="starting LED currents (continuous limits R120 G180 B160)")
    fs.add_argument("--period-us", type=int, default=FASTSCAN_PERIOD_US)
    fs.add_argument("--camera-exposure-us", type=int, default=FASTSCAN_CAMERA_EXPOSURE_US)
    fs.add_argument("--camera-fps", type=float, default=FASTSCAN_CAMERA_FPS)
    fs.add_argument("--loops", type=int, default=FASTSCAN_LOOPS)
    fs.add_argument("--geometry-id", default=None)
    fs.add_argument("--singles", action="store_true",
                    help="also write every slot as its own 912x1140 BMP (for the GUI's own "
                         "image composer; ~3 MB each)")
    a = ap.parse_args(argv)
    if a.unpack_check:
        bad = verify_pack(Path(a.unpack_check))
        print("round trip OK" if not bad else f"round trip FAILED for {bad} pattern(s)")
        return 1 if bad else 0
    if a.fastscan:
        led = None
        if a.led:
            r, g, b = (int(v) for v in a.led.split(","))
            led = {"r": r, "g": g, "b": b}
        d = make_fastscan_pack(zone_rings=a.zone_ring, zone_masks=a.zone_mask,
                               band_cam=a.zone_band_cam, band_field=a.zone_band_field,
                               margin_px=a.zone_margin, calibration=a.calibration,
                               depth_mm=a.depth_mm, study=a.study,
                               out_root=Path(a.out) if a.out else None, label=a.label,
                               geometry_id=a.geometry_id, led=led, period_us=a.period_us,
                               camera_exposure_us=a.camera_exposure_us,
                               camera_fps=a.camera_fps, loops=a.loops, singles=a.singles)
        print(d)
        print((d / "procedure.txt").read_text(encoding="utf-8"))
        if a.study:
            m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
            print(study_table(m["study"]))
        return 0
    axes = tuple(s.strip() for s in a.axes.split(",") if s.strip())
    d = make_pack(a.preset, axes=axes, video_port=a.video_port,
                  out_root=Path(a.out) if a.out else DEFAULT_OUT, label=a.label)
    print(d)
    print((d / "procedure.txt").read_text(encoding="utf-8"))
    return 0


def study_table(s: dict) -> str:
    nm = s.get("noise_model") or {}
    L = [f"simulated decode error, {s['cam_px_per_col']} camera px / column, "
         f"{s['mm_per_col']} mm / column, prefilter {s['prefilter_px']} px on mid + coarse",
         "systematic = noise-free, vs the ideal phase; noise = random per camera px"
         + (f" ({nm.get('noise_dn')} DN / frame at {nm.get('white_dn')} DN white, "
            f"{nm.get('frames')} frames averaged)" if nm else ""),
         f"{'sigma':>5}  {'level':<16} {'rms col':>8} {'p99 col':>8} {'rms mm':>7} "
         f"{'p99 mm':>7} {'mod':>6} {'noise mm':>8}"]
    for sg, row in s["sigmas"].items():
        for lv in ("fine", "fine_square_wave", "mid", "coarse"):
            r = row[lv]
            nz = r.get("noise_rms_mm")
            L.append(f"{sg:>5}  {lv:<16} {r['rms_cols']:8.4f} {r['p99_cols']:8.4f} "
                     f"{r['rms_mm']:7.4f} {r['p99_mm']:7.4f} {r['modulation']:6.3f} "
                     f"{'' if nz is None else format(nz, '8.4f')}")
        cf = row["fine_square_wave"].get("closed_form")
        if cf:
            L.append(f"{sg:>5}  {'square, closed':<16} {cf['rms_cols']:8.4f} peak "
                     f"{cf['peak_cols']:.4f} col, |a3/a1| after optics {cf['a3_over_a1']:.4f}")
        for par, v in (row.get("fine_vs_hdmi") or {}).items():
            L.append(f"{sg:>5}  {'fine vs HDMI ' + par:<16} {v['rms_cols']:8.4f} "
                     f"{v['p99_cols']:8.4f} {v['rms_mm']:7.4f}  bias {v['bias_cols']:+.4f} col "
                     f"({v['bias_mm']:+.4f} mm)")
        for px, v in (row.get("fine_prefiltered") or {}).items():
            r = v.get("vs_ideal")
            if r:
                L.append(f"{sg:>5}  {'fine, pf ' + px + ' px':<16} {r['rms_cols']:8.4f} "
                         f"{r['p99_cols']:8.4f} {r['rms_mm']:7.4f} {r['p99_mm']:7.4f} "
                         f"{r['modulation']:6.3f} {r['noise_rms_mm']:8.4f}")
            for par, w in v["vs_hdmi"].items():
                L.append(f"{sg:>5}  {'  vs HDMI ' + par:<16} {w['rms_cols']:8.4f} "
                         f"{w['p99_cols']:8.4f} {w['rms_mm']:7.4f}  bias {w['bias_cols']:+.4f} col")
        e = row["end_to_end"]
        L.append(f"{sg:>5}  {'end-to-end':<16} rms {e['rms_err_cols']} col, max "
                 f"{e['max_abs_err_cols']} col, unwrap rejected {e['unwrap_rejected_pct']} %")
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())

"""Flash / video-port pattern packs + guarded hardware-trigger capture, mock only.

A fake triggered camera renders each UNPACKED flash pattern through the
MockBench in LUT order; decoding those frames with HybridBasis must equal
decoding the video-path (basis.generate()) frames. No hardware, no USB."""

import sys

import numpy as np
import pytest

from engine import config, rules
from engine.bases.hybrid import from_preset_bases
from engine.capture import patternmode as pm
from tools import make_flash_pack as mfp

from conftest import MockBench, VALIDATION_OBJECT

PRESET = mfp.DEFAULT_PRESET
HALF = {"cam_shape": (540, 720), "cam_px_per_mm": 1.5}


@pytest.fixture(scope="module")
def preset():
    return config.load_preset(PRESET)


@pytest.fixture(scope="module")
def patterns(preset):
    return mfp.build_patterns(preset, ("x", "y"))


def _want(p):
    return (p["image"] > 0).astype(np.uint8) * 255 if p["bitdepth"] == 1 else p["image"]


# ------------------------------------------------------------- packing
def test_pattern_set_matches_basis(preset, patterns):
    b = from_preset_bases(preset.bases, axis="x")
    xs = [p for p in patterns if p["axis"] == "x"]
    assert len(xs) == b.n_frames() == 16
    assert [p["bitdepth"] for p in xs] == [1] * 12 + [8] * 4
    for p, g in zip(xs, b.generate()):
        assert np.array_equal(p["image"], g)


@pytest.mark.parametrize("policy,kw", [("sdk", {}),
                                       ("grouped", {"max_per_image": {1: 3, 8: 2}})])
def test_pack_unpack_round_trip_exact(patterns, policy, kw):
    images, seq = mfp.pack(patterns, policy=policy, **kw)
    assert all(im.shape == (1140, 912, 3) and im.dtype == np.uint8 for im in images)
    for p, back in zip(patterns, mfp.unpack(images, seq)):
        assert np.array_equal(_want(p), back), p["name"]
    # no two patterns share a bit
    used = {}
    for e in seq:
        for b in range(e["bitdepth"]):
            key = (e["image_index"], e["bit_position"] + b)
            assert key not in used
            used[key] = e["name"]


def test_sdk_layout_matches_ti_order(patterns):
    xs = [p for p in patterns if p["axis"] == "x"]
    images, seq = mfp.pack(xs, policy="sdk")
    # 12 one-bit at G0..G7,R0..R3, first fringe skips to B0, then a new image G,R,B
    assert [e["planes"][0] for e in seq[:12]] == [f"G{i}" for i in range(8)] + \
        [f"R{i}" for i in range(4)]
    assert [(e["image_index"], e["planes"][0], e["planes"][-1]) for e in seq[12:]] == [
        (0, "B0", "B7"), (1, "G0", "G7"), (1, "R0", "R7"), (1, "B0", "B7")]
    assert len(images) == 2
    # G byte of image 0 carries white in bit 0, black (0) in bit 1
    assert (images[0][..., 1] & 1).all() and not (images[0][..., 1] & 2).any()


def test_bmp_file_round_trip(tmp_path, patterns):
    images, seq = mfp.pack(patterns[:16], policy="sdk")
    for i, im in enumerate(images):
        f = mfp.write_bmp(tmp_path / f"i{i}.bmp", im)
        assert f.stat().st_size == 54 + 912 * 1140 * 3
        assert np.array_equal(mfp.read_bmp(f), im)
    with pytest.raises(FileExistsError):
        mfp.write_bmp(tmp_path / "i0.bmp", images[0])


def test_make_pack_writes_new_dir_and_verifies(tmp_path):
    d1 = mfp.make_pack(PRESET, axes=("x",), out_root=tmp_path)
    d2 = mfp.make_pack(PRESET, axes=("x",), out_root=tmp_path)
    assert d1 != d2                                     # never overwrite
    m, images = mfp.load_pack(d1)
    assert m["mode"] == "flash" and len(m["sequence"]) == 16 and len(images) == 2
    assert mfp.verify_pack(d1) == 0
    assert "Firmware Build" in (d1 / "procedure.txt").read_text(encoding="utf-8")
    t = m["timing"]
    for d, v in t["per_bitdepth"].items():
        mfp.validate_timing(int(d), v["exposure_us"], v["period_us"])
    dv = mfp.make_pack(PRESET, axes=("x",), video_port=True, out_root=tmp_path)
    mv, iv = mfp.load_pack(dv)
    # 12 one-bit at <= 3 per frame (camera-limited) + 4 fringes at 2 per frame
    assert len(iv) == 6 and mfp.verify_pack(dv) == 0
    assert all(not f["errors"] for f in mv["timing"]["per_frame"])


# -------------------------------------------------------------- timing
@pytest.mark.parametrize("bpp,exp,per,vsync", [
    (1, 234, 234, False),        # below 1-bit minimum
    (8, 8000, 9000, False),      # below 8-bit minimum
    (1, 1000, 1100, False),      # period - exposure < 230
    (1, 1000, 1229, False),
    (1, 1000, 1500, True),       # VSYNC needs exposure == period
    (9, 9000, 9000, False),
])
def test_timing_rejects_illegal(bpp, exp, per, vsync):
    with pytest.raises(mfp.TimingError):
        mfp.validate_timing(bpp, exp, per, vsync=vsync)


@pytest.mark.parametrize("bpp,exp,per,vsync", [
    (1, 235, 235, False), (1, 1000, 1230, False), (8, 8333, 8333, True),
    (8, 8333, 8600, False)])
def test_timing_accepts_legal(bpp, exp, per, vsync):
    mfp.validate_timing(bpp, exp, per, vsync=vsync)


# -------------------------------------------------------- triggered capture
class FakeTriggeredCamera:
    """Plays `dmd_images` through a MockBench, one per trigger, with frame
    ids and timestamps like PySpin's. `drop` skips frame ids (camera missed
    a trigger); `stop_after` stops the sequence early."""

    def __init__(self, bench, dmd_images, period_us, drop=(), stop_after=None, noise=1.5,
                 extra=0):
        self.bench, self.imgs, self.period_us = bench, dmd_images, period_us
        self.drop, self.stop_after, self.noise, self.extra = set(drop), stop_after, noise, extra
        self.rng = np.random.default_rng(7)
        self.k = self.fid = 0
        self.armed = False

    def arm(self, n):
        self.armed = True

    def disarm(self):
        self.armed = False

    def pending(self):
        return self.extra

    def grab_triggered(self, timeout_ms):
        assert self.armed
        if self.k >= len(self.imgs) or (self.stop_after is not None and self.k >= self.stop_after):
            raise TimeoutError("no trigger")
        if self.k in self.drop:
            self.fid += 1
        img = self.bench.render(self.imgs[self.k])
        img = np.clip(img + self.rng.normal(0, self.noise, img.shape), 0, 255).astype(np.uint8)
        out = (img, self.fid, 10_000_000 + self.k * int(self.period_us) * 1000)
        self.k += 1
        self.fid += 1
        return out


def _render_frames(bench, imgs, seed=7):
    rng = np.random.default_rng(seed)
    out = []
    for im in imgs:
        f = bench.render(im)
        out.append(np.clip(f + rng.normal(0, 1.5, f.shape), 0, 255).astype(np.uint8)
                   .astype(np.float32))
    return out


@pytest.fixture(scope="module")
def pack_x(preset):
    pats = mfp.build_patterns(preset, ("x",))
    images, seq = mfp.pack(pats, policy="sdk")
    manifest = {"sequence": seq, "timing": mfp.suggest_timing("flash", seq)}
    return pats, images, manifest


def _bench():
    b = MockBench(seed=1, **HALF)
    b.place(VALIDATION_OBJECT)
    return b


def test_triggered_decode_matches_video_path(preset, pack_x):
    pats, images, manifest = pack_x
    basis = from_preset_bases(preset.bases, axis="x")
    bench = _bench()
    period = manifest["timing"]["uniform"]["period_us"]
    cam = FakeTriggeredCamera(bench, mfp.unpack(images, manifest["sequence"]), period)
    cap = pm.TriggeredSequenceCapture(cam, manifest, acknowledge=pm.ACK_PHRASE).capture()
    assert cap.report["n_frames"] == 16 and not cam.armed
    trig = basis.decode(cap.frames_for("x"), "g")
    video = basis.decode(_render_frames(_bench(), basis.generate()), "g")
    assert np.array_equal(trig.mask, video.mask)
    assert np.array_equal(trig.corr_x, video.corr_x, equal_nan=True)
    # and it is a real decode: object relief vs the bench truth
    _, _, valid, cx, _ = bench._scene()
    m = trig.mask & valid
    assert m.mean() > 0.3
    assert np.nanmedian(np.abs(trig.corr_x[m] - cx[m])) < 0.5


def test_frames_for_orders_by_basis_index(pack_x):
    _, _, manifest = pack_x
    seq = list(reversed(manifest["sequence"]))
    cap = pm.TriggeredCapture([np.full((2, 2), e["basis_index"], np.float32) for e in seq],
                              list(range(len(seq))), [0] * len(seq), seq)
    assert [int(f[0, 0]) for f in cap.frames_for("x")] == list(range(16))


def test_guard_requires_acknowledge(pack_x):
    _, images, manifest = pack_x
    cam = FakeTriggeredCamera(_bench(), [], 5000)
    for ack in (None, True, "yes"):
        with pytest.raises(pm.PatternModeRefused):
            pm.TriggeredSequenceCapture(cam, manifest, acknowledge=ack)

    class FreeRun:
        def grab(self):
            return None
    with pytest.raises(pm.PatternModeRefused):
        pm.TriggeredSequenceCapture(FreeRun(), manifest, acknowledge=pm.ACK_PHRASE)


def test_guard_refuses_when_usb_driver_loaded(pack_x, monkeypatch):
    _, _, manifest = pack_x
    monkeypatch.setitem(sys.modules, "hid", object())
    with pytest.raises(rules.SafetyViolation):
        pm.TriggeredSequenceCapture(FakeTriggeredCamera(_bench(), [], 5000), manifest,
                                    acknowledge=pm.ACK_PHRASE)


def test_missing_frames_detected(pack_x):
    _, images, manifest = pack_x
    imgs = mfp.unpack(images, manifest["sequence"])
    period = manifest["timing"]["uniform"]["period_us"]
    bench = MockBench(seed=1, cam_shape=(60, 80))
    for kw, msg in [({"drop": {5}}, "frame ids jump"), ({"stop_after": 9}, "got 9 of 16"),
                    ({"extra": 2}, "extra frame")]:
        cam = FakeTriggeredCamera(bench, imgs, period, **kw)
        with pytest.raises(pm.MissingFrames, match=msg):
            pm.TriggeredSequenceCapture(cam, manifest, acknowledge=pm.ACK_PHRASE,
                                        frame_timeout_ms=10).capture()
        assert not cam.armed
    # timestamps at the wrong spacing (GUI period != manifest)
    cam = FakeTriggeredCamera(bench, imgs, period * 3)
    with pytest.raises(pm.MissingFrames, match="spacing"):
        pm.TriggeredSequenceCapture(cam, manifest, acknowledge=pm.ACK_PHRASE).capture()


def test_patternmode_imports_no_usb():
    import ast
    import inspect
    src = inspect.getsource(pm) + inspect.getsource(mfp)
    names = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
    assert not any(n and ("dmd_control" in n or "dlp_driver" in n or n == "hid"
                          or n.startswith("swiss_army_sensor")) for n in names)


def test_flash_buffer_load_rule(patterns):
    xs = [p for p in patterns if p["axis"] == "x"]
    for pats, warn in ((xs, False), (patterns, True)):
        _, seq = mfp.pack(pats, policy="sdk")
        t = mfp.suggest_timing("flash", seq)
        assert bool(mfp.flash_load_warnings(seq, t)) is warn

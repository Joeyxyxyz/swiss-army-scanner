"""Hybrid gray code + phase shift: axis y (projector rows), complementary
(TI include_inverted) and coarse_bits — both polarities, against exact
identity-camera frames and the MockBench ground truth — plus a bit-exact
regression of the v10 default path."""

import numpy as np
import pytest
from scipy import ndimage

import dataclasses

from engine import _seed, config
from engine.bases import registry
from engine.bases.hybrid import (SAT_LEVEL, TI_MIN_BIT_CONTRAST, HybridBasis,
                                 from_preset_bases)
from engine.config import ConfigError, preset_from_dict
from engine.session import ScanSession

from conftest import PRESET, VALIDATION_OBJECT, MockBench

TOL_PX = 1.5
# half-size camera for the shared rig: same optics per mm and per DMD px
# (render is point-sampled), ~4x cheaper per frame than the 1440x1080 default
HALF = {"cam_shape": (540, 720), "cam_px_per_mm": 1.5}
COARSE = {"x": 7, "y": 8}     # 8 DMD px stripes on either axis (x: 1 px x 2**3, y: 2 px x 2**2)
DEFAULT_SPECS = [{"name": "graycode", "bits": 10, "complementary": False},
                 {"name": "phaseshift", "steps": 4, "period_px": 24}]


# ------------------------------------------------------ v10 reference copy
def _v10_decode(frames, bits=10, steps=4, period_px=24, contrast_min=8.0, conf_min=4.0):
    """Verbatim copy of HybridBasis.decode before axis/complementary/coarse
    landed (2026-09-22). The default path must stay bit-identical to it."""
    P = _seed.load("patterns")
    D = _seed.load("decoders")
    white, black = frames[0].astype(np.float64), frames[1].astype(np.float64)
    gray_frames = frames[2:2 + bits]
    ps_frames = frames[2 + bits:]

    contrast = white - black
    flipped = float(contrast.mean()) < 0
    if flipped:
        white, black = black, white
        contrast = -contrast
    mask = contrast > contrast_min
    thresh = (white + black) / 2.0

    code = np.zeros(white.shape, dtype=np.uint64)
    for b, g in enumerate(gray_frames):
        bit = (g < thresh) if flipped else (g > thresh)
        code |= bit.astype(np.uint64) << np.uint64(b)
    binary = code.copy()
    shift = 1
    while shift < bits:
        binary ^= binary >> np.uint64(shift)
        shift <<= 1
    stripe_width = max(1, -(-P.DMD_W // (2 ** bits)))
    coarse_col = binary.astype(np.float64) * stripe_width

    r = D.decode_phase_shift(ps_frames)
    phase = r.phase
    if flipped:
        phase = np.angle(np.exp(1j * (phase + np.pi)))
    mask &= r.modulation > conf_min
    phi_coarse = 2 * np.pi * coarse_col / period_px
    k = np.round((phi_coarse - phase) / (2 * np.pi))
    column = (phase + 2 * np.pi * k) * period_px / (2 * np.pi)
    column = np.clip(column, 0, P.DMD_W - 1)
    corr_x = np.where(mask, column, np.nan).astype(np.float32)
    meta = {"polarity_flipped": flipped, "bits": bits, "steps": steps,
            "period_px": period_px,
            "gc_conf": float(np.median(r.modulation[mask])) if mask.any() else 0.0,
            "contrast_mean": float(contrast.mean())}
    return corr_x, r.modulation.astype(np.float32), mask, meta


def assert_same_as_v10(frames, **decode_kw):
    # v10 had no clipped-sample handling (sat_level=None turns it off); the
    # default decode equals v10 wherever no fringe sample clips, see
    # test_default_decode_is_v10_wherever_no_fringe_sample_clips
    fld = HybridBasis(sat_level=None).decode(frames, "g", **decode_kw)
    corr_x, conf, mask, meta = _v10_decode(frames, **decode_kw)
    assert fld.corr_y is None and fld.corr_x.dtype == corr_x.dtype
    np.testing.assert_array_equal(fld.corr_x, corr_x)       # NaNs in the same places
    np.testing.assert_array_equal(fld.conf, conf)
    np.testing.assert_array_equal(fld.mask, mask)
    for k, v in meta.items():
        assert fld.meta[k] == v, k
    assert fld.basis_names == ["graycode", "phaseshift"]
    return fld


# ----------------------------------------------------------------- helpers
def identity_frames(basis, invert=False, gain=200.0, ambient=10.0):
    """Camera == DMD (one camera px per mirror, no blur), subsampled across
    the stripes for speed: decode must return each pixel's own column/row."""
    out = []
    for q in basis.generate():
        q = q.astype(np.float32)
        if invert:
            q = 255.0 - q
        f = ambient + gain * q / 255.0
        out.append(f[::57, :] if basis.axis == "x" else f[:, ::48])
    return out


def identity_truth(basis):
    rows, cols = np.mgrid[0:1140, 0:912].astype(np.float64)
    t = cols if basis.axis == "x" else rows
    return t[::57, :] if basis.axis == "x" else t[:, ::48]


def corr_of(fld, axis):
    other = fld.corr_y if axis == "x" else fld.corr_x
    assert other is None, "a one-axis run must not invent the other axis"
    return fld.corr_x if axis == "x" else fld.corr_y


def truth_score(fld, bench, axis):
    """(fraction of decoded pixels within TOL_PX of the bench truth,
    fraction of truly-valid pixels that decoded)."""
    _, _, valid, tx, ty = bench._scene()
    truth = tx if axis == "x" else ty
    corr = corr_of(fld, axis)
    m = fld.mask
    assert m.any()
    err = np.abs(corr[m] - truth[m])
    return float(np.mean(err <= TOL_PX)), float((m & valid).sum() / valid.sum())


@pytest.fixture(scope="module")
def rig(tmp_path_factory):
    """One mock session for the module: a capture costs a few seconds, so each
    (basis, polarity) stack is shot once and shared between tests."""
    s = ScanSession(tmp_path_factory.mktemp("hyb") / "library", mock=True,
                    bench=MockBench(seed=1, **HALF))
    s.bench.place(VALIDATION_OBJECT)          # relief, a shadow and an albedo edge
    p = config.load_preset(PRESET, store=s.store, mock=True)
    p.camera.average = 1
    cache = {}

    def shoot(invert=True, **kw):
        key = (invert, tuple(sorted(kw.items())))
        if key not in cache:
            basis = HybridBasis(**kw)
            p.projection.invert = invert      # True = the bench's inverted HDMI path
            with s.rig(p) as sync:
                cache[key] = (basis, [sync.shoot(q) for q in basis.generate()])
        return cache[key]

    shoot.bench = s.bench
    return shoot


POLARITY = pytest.mark.parametrize("invert", [True, False], ids=["inverted", "upright"])


# -------------------------------------------------------- regression (v10)
@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@POLARITY
def test_default_path_bit_identical_to_v10_on_synthetic_stack(dtype, invert):
    rng = np.random.default_rng(7)
    shape = (48, 64)
    white = rng.uniform(120, 250, shape)
    black = rng.uniform(0, 60, shape)
    if invert:
        white, black = black, white
    rest = [rng.uniform(0, 255, shape) for _ in range(14)]
    frames = [np.asarray(f).astype(dtype) for f in [white, black] + rest]
    fld = assert_same_as_v10(frames)
    assert fld.meta["polarity_flipped"] is invert


@POLARITY
def test_default_path_bit_identical_to_v10_on_bench_capture(rig, invert):
    basis, frames = rig(invert)
    assert len(frames) == 16
    fld = assert_same_as_v10(frames, contrast_min=8.0, conf_min=4.0)
    assert fld.meta["polarity_flipped"] is invert
    frac, cover = truth_score(fld, rig.bench, "x")
    assert frac >= 0.99 and cover >= 0.95, (frac, cover)


def test_default_generate_unchanged():
    P = _seed.load("patterns")
    pats = HybridBasis().generate()
    ref = ([P.flood(255), P.black()] + P.gray_code_stripes(n_bits=10)
           + P.phase_shift_fringes(n_steps=4, period_px=24))
    assert len(pats) == len(ref) == HybridBasis().n_frames() == 16
    for a, b in zip(pats, ref):
        np.testing.assert_array_equal(a, b)


# --------------------------------------------- exact decode, every mode
@pytest.mark.parametrize("axis", ["x", "y"])
@pytest.mark.parametrize("complementary", [False, True], ids=["global", "complementary"])
@pytest.mark.parametrize("coarse", [False, True], ids=["allbits", "coarse"])
@POLARITY
def test_identity_camera_decodes_own_position(axis, complementary, coarse, invert):
    basis = HybridBasis(axis=axis, complementary=complementary,
                        coarse_bits=COARSE[axis] if coarse else None)
    frames = identity_frames(basis, invert=invert)
    assert len(frames) == basis.n_frames()
    fld = basis.decode(frames, "g")
    corr = corr_of(fld, axis)
    assert fld.meta["polarity_flipped"] is invert
    assert fld.mask.all()
    np.testing.assert_allclose(corr, identity_truth(basis), atol=0.1)
    assert np.nanmin(corr) >= 0 and np.nanmax(corr) <= basis.extent - 1
    assert fld.meta["axis"] == axis and fld.meta["units"] == "projector px"


# ------------------------------------------------- bench ground truth
@POLARITY
def test_axis_y_matches_bench_rows(rig, invert):
    basis, frames = rig(invert, axis="y")
    fld = basis.decode(frames, "g")
    assert fld.meta["polarity_flipped"] is invert and fld.meta["coded"] == "row"
    frac, cover = truth_score(fld, rig.bench, "y")
    assert frac >= 0.95 and cover >= 0.95, (frac, cover)
    assert np.nanmax(fld.corr_y) <= 1139 and np.nanmin(fld.corr_y) >= 0


@POLARITY
def test_complementary_matches_bench(rig, invert):
    basis, frames = rig(invert, complementary=True)
    assert len(frames) == basis.n_frames() == 2 + 2 * 10 + 4
    fld = basis.decode(frames, "g")
    assert fld.meta["polarity_flipped"] is invert and fld.meta["complementary"]
    frac, cover = truth_score(fld, rig.bench, "x")
    # only pixels sitting on a fine stripe edge fall under the TI floor
    # (~8% here: the mock point-samples, so 1-mirror stripes stay crisp)
    assert frac >= 0.95 and cover >= 0.85, (frac, cover)
    assert 0 < fld.meta["weak_bit_pct"] < 15
    assert fld.meta["min_bit_contrast"] == TI_MIN_BIT_CONTRAST


@POLARITY
def test_coarse_bits_7_matches_bench(rig, invert):
    basis, frames = rig(invert, coarse_bits=7)
    assert len(frames) == basis.n_frames() == 2 + 7 + 4
    assert basis.coarse_stripe_px == 8
    fld = basis.decode(frames, "g")
    assert fld.meta["polarity_flipped"] is invert and fld.meta["gray_bits_used"] == 7
    frac, cover = truth_score(fld, rig.bench, "x")
    assert frac >= 0.95 and cover >= 0.95, (frac, cover)


@POLARITY
def test_axis_y_complementary_coarse_matches_bench(rig, invert):
    basis, frames = rig(invert, axis="y", complementary=True, coarse_bits=8)
    assert len(frames) == 2 + 2 * 8 + 4
    fld = basis.decode(frames, "g")
    assert fld.meta["polarity_flipped"] is invert
    frac, cover = truth_score(fld, rig.bench, "y")
    assert frac >= 0.95 and cover >= 0.95, (frac, cover)


def test_coarse_bits_rescue_complementary_when_fine_planes_wash_out(rig):
    """Bench physics: 1-mirror stripes keep ~1% contrast. Blur the capture
    along the coded axis: the full complementary code rejects most pixels
    (the finest plane is under the TI floor everywhere) and says so in the
    per-plane readout; dropping the fine planes decodes the scene."""
    sigma = 1.5          # camera px on the half-size rig = ~2.4 DMD columns
    blur = lambda frames: [ndimage.gaussian_filter1d(f, sigma, axis=1) for f in frames]  # noqa: E731
    full, frames = rig(True, complementary=True)
    f_full = full.decode(blur(frames), "g")
    bc = f_full.meta["bit_contrast_coarse_to_fine"]
    assert len(bc) == 10 and bc[0] > 100 and bc[-1] < 0.1 * bc[0]
    assert f_full.meta["weak_bit_pct"] > 50
    _, cover_full = truth_score(f_full, rig.bench, "x")

    coarse, frames7 = rig(True, complementary=True, coarse_bits=7)
    f7 = coarse.decode(blur(frames7), "g")
    frac, cover = truth_score(f7, rig.bench, "x")
    assert cover_full < 0.5 and cover >= 0.9, (cover_full, cover)
    assert frac >= 0.95, frac


# ---------------------------------------------------- complementary rule
def test_min_bit_contrast_is_the_per_pixel_floor():
    basis = HybridBasis(complementary=True)
    frames = identity_frames(basis, invert=True)
    # make the 4th plane's inverse differ from the normal by only 3 grey
    # levels (same sign) over columns 100..199
    n, inv = frames[2 + 2 * 3], frames[2 + 2 * 3 + 1].copy()
    cols = slice(100, 200)
    inv[:, cols] = n[:, cols] + 3.0 * np.sign(inv[:, cols] - n[:, cols])
    frames[2 + 2 * 3 + 1] = inv

    fld = basis.decode(frames, "g")
    assert not fld.mask[:, cols].any()
    assert fld.mask[:, :100].all() and fld.mask[:, 200:].all()
    assert fld.meta["weak_bit_pct"] == pytest.approx(100 * 100 / 912, abs=0.01)

    loose = HybridBasis(complementary=True, min_bit_contrast=2.0).decode(frames, "g")
    assert loose.mask.all()          # 3 > 2: trusted, and the bit is still right
    np.testing.assert_allclose(loose.corr_x, identity_truth(basis), atol=0.1)


# ------------------------------------------------- clipped fringe samples
def clipped(frames):
    """What a Mono8 camera returns when the scene is driven past full scale."""
    return [np.minimum(f, 255.0) for f in frames]


@pytest.mark.parametrize("gain", [300.0, 800.0], ids=["mild", "glint"])
@pytest.mark.parametrize("steps", [3, 4, 8])
@POLARITY
def test_clipped_fringe_samples_are_resolved_or_dropped_not_rippled(steps, invert, gain):
    """Review 2026-09-22: a sinusoid clipped at full scale gives the N-step
    decoder a systematic ripple (~0.7 projector px at N=4, ~1.4 mm of false
    height on the bench) and nothing masked it. Now a clipped sample is
    dropped and the phase re-solved from the rest: exact with >= 3 left,
    bounded (clip level + dim flood) with 2, else the pixel is invalid —
    never kept rippled. Pixels with no clipped sample are untouched."""
    basis = HybridBasis(steps=steps)
    truth = identity_truth(basis)
    frames = clipped(identity_frames(basis, invert=invert, gain=gain))   # peaks 310 / 810
    ps = np.stack(frames[-steps:])
    n_clip = (ps >= SAT_LEVEL).sum(axis=0)
    left = steps - n_clip
    assert (n_clip >= 1).any()
    if gain == 300.0:
        assert (n_clip == 0).any() == (steps == 3)   # ~106 deg of every fringe clips

    old = HybridBasis(steps=steps, sat_level=None).decode(frames, "g")
    old_err = np.abs(old.corr_x - truth)[old.mask]
    # the bug: every pixel kept, rippled (measured 0.25-0.34 px at N=3/4, 0.04 at N=8)
    assert old.mask.all() and old_err.max() > (0.03 if steps == 8 else 0.2)

    fld = basis.decode(frames, "g")
    assert fld.meta["polarity_flipped"] is invert
    err = np.abs(fld.corr_x - truth)[fld.mask]
    # kept: exact solves, or a 2-sample family within DECLIP_TOL_PX (0.1 px)
    assert err.max() < 0.5 * old_err.max() and err.max() < 0.06, err.max()
    assert fld.mask[left >= 3].all()                  # >= 3 samples left: always recovered
    assert not fld.mask[left < 2].any()               # nothing to solve from: dropped
    clean = n_clip == 0
    np.testing.assert_array_equal(fld.corr_x[clean], old.corr_x[clean])   # bit-identical
    np.testing.assert_array_equal(fld.conf[clean], old.conf[clean])

    m = fld.meta
    assert m["sat_level"] == SAT_LEVEL
    assert m["saturated_pct"] == 100.0                      # every white flood clips
    assert m["saturated_masked_pct"] == pytest.approx(100 * (~fld.mask).mean(), abs=1e-2)
    assert m["saturated_resolved_pct"] + m["saturated_masked_pct"] == pytest.approx(
        100 * (n_clip > 0).mean(), abs=1e-2)
    assert any("clip" in w and "exposure" in w for w in m["warnings"])
    if steps == 8:
        assert fld.mask.all()                               # always >= 3 samples left
    if gain == 800.0 and steps == 4:
        assert (~fld.mask).mean() > 0.5                    # a glint is dropped, not guessed


def test_two_sample_recovery_keeps_mild_clipping_and_drops_a_glint():
    """N=4, the mock box on the vstripe preset (inverted, duty 0.55: fringes
    span the dim flood .. full drive). At albedo 0.7 every pixel with two
    clipped samples is pinned to < 0.1 px by the clip level and the dim
    flood; at 0.95 most are not and must be dropped."""
    from engine.bases.hybrid import DECLIP_TOL_PX
    phi = np.linspace(-np.pi, np.pi, 1441)[:-1]
    frac = {}
    for albedo in (0.7, 0.95):
        dim, full = 12 + 380 * albedo * 115 / 255, 12 + 380 * albedo
        a, b = (dim + full) / 2, (full - dim) / 2
        ps = [np.minimum(a + b * np.cos(phi + k * np.pi / 2), 255.0)[None, :] for k in range(4)]
        frames = ([np.full((1, phi.size), dim), np.full((1, phi.size), 255.0)]
                  + [np.full((1, phi.size), dim)] * 10 + ps)      # inverted: dim flood first
        fld = HybridBasis(sat_level=SAT_LEVEL).decode(frames, "g")
        two = ((np.stack(ps) >= SAT_LEVEL).sum(axis=0) == 2)[0]
        assert two.any()
        # the phase alone (the gray code here is junk): compare wrapped
        got = np.angle(np.exp(1j * (2 * np.pi * fld.corr_x[0] / 24)))
        want = np.angle(np.exp(1j * (phi + np.pi)))                 # inverted: + pi
        e = np.abs(np.angle(np.exp(1j * (got - want))))[fld.mask[0]] * 24 / (2 * np.pi)
        assert e.max() <= DECLIP_TOL_PX / 2 + 0.01, e.max()
        frac[albedo] = fld.mask[0][two].mean()
    assert frac[0.7] == 1.0 and frac[0.95] < 0.2, frac


def test_unclipped_scene_decodes_exactly_as_without_clip_handling():
    basis = HybridBasis()
    frames = identity_frames(basis, invert=True)            # peaks 210 counts
    a, b = basis.decode(frames, "g"), HybridBasis(sat_level=None).decode(frames, "g")
    np.testing.assert_array_equal(a.corr_x, b.corr_x)
    np.testing.assert_array_equal(a.mask, b.mask)
    assert a.meta["saturated_pct"] == 0 and not a.meta["warnings"]
    assert b.meta["sat_level"] is None and b.meta["saturated_pct"] is None


def test_default_decode_is_v10_wherever_no_fringe_sample_clips(rig):
    """The shared rig's validation box (albedo 0.7) clips at the fringe
    peaks under the bench presets' inverted full drive: only those pixels
    change, everything else stays the proven v10 decode."""
    basis, frames = rig(True)
    old = HybridBasis(sat_level=None).decode(frames, "g")
    new = basis.decode(frames, "g")
    clean = (np.stack(frames[-4:]) < SAT_LEVEL).all(axis=0)
    assert clean.any() and (~clean).any()
    for a in ("corr_x", "conf", "mask"):
        np.testing.assert_array_equal(getattr(new, a)[clean], getattr(old, a)[clean])


def _box_top_errors(fld, bench):
    height, _, valid, tx, _ = bench._scene()
    top = ndimage.binary_erosion(height > 0, iterations=4) & valid
    m = top & fld.mask
    return top, m, np.abs(fld.corr_x[m] - tx[m])


@pytest.fixture(scope="module")
def glint_rig(tmp_path_factory):
    """The review's scenario: the validation box at albedo 0.95 on the vstripe
    preset (invert, duty 0.55) — clipped in the bright flood and at every
    fringe peak."""
    s = ScanSession(tmp_path_factory.mktemp("glint") / "library", mock=True,
                    bench=MockBench(seed=1, **HALF))
    s.bench.place(dataclasses.replace(VALIDATION_OBJECT, albedo=0.95))
    p = config.load_preset(PRESET, store=s.store, mock=True)
    p.camera.average = 1
    basis = from_preset_bases(p.bases)
    with s.rig(p) as sync:
        frames = [sync.shoot(q) for q in basis.generate()]
    return basis, frames, s.bench


def test_bright_box_top_has_no_clipping_ripple(rig, glint_rig):
    """Bench-truth check on the box top. Measured before the fix: relief
    ripple p99 0.34 px at albedo 0.7 and 0.79 px at 0.95, every pixel valid,
    no warning."""
    basis, frames = rig(True)
    for (b, f, bench), keep_min in (((basis, frames, rig.bench), 0.95),
                                    (glint_rig, 0.0)):
        top, _, old_err = _box_top_errors(HybridBasis(sat_level=None).decode(f, "g"), bench)
        assert old_err.size == top.sum() and np.percentile(old_err, 99) > 0.3
        fld = b.decode(f, "g")
        _, m, err = _box_top_errors(fld, bench)
        assert m.sum() >= keep_min * top.sum(), (m.sum(), top.sum())
        if err.size:
            assert np.percentile(err, 99) < 0.25 and err.max() < 0.35, err.max()
        assert fld.meta["saturated_pct"] > 0.1
        assert any("clip" in w for w in fld.meta["warnings"])
    # the 0.95 top clips in >= 2 of 4 samples almost everywhere: dropped
    top, m, _ = _box_top_errors(glint_rig[0].decode(glint_rig[1], "g"), glint_rig[2])
    assert m.sum() < 0.2 * top.sum()


def test_projector_map_reports_clipping_per_axis(tmp_path):
    from engine.decode import projmap
    s = ScanSession(tmp_path / "library", mock=True, bench=MockBench(seed=2, **HALF))
    s.bench.place(dataclasses.replace(VALIDATION_OBJECT, albedo=0.95))
    p = config.load_preset(PRESET, store=s.store, mock=True)
    p.camera.average = 1
    m = projmap.capture(s, p, "glint")
    for ax in ("x", "y"):
        d = m.meta["decode"][ax]
        assert d["sat_level"] == SAT_LEVEL and d["saturated_pct"] > 0.1
        assert d["saturated_masked_pct"] > 0 and d["fix_period_jumps"] is False
    assert any(w.startswith("column") and "clip" in w for w in m.warnings)
    assert any(w.startswith("row") and "clip" in w for w in m.warnings)


# ------------------------------------------------------- period jumps
def test_fix_period_jumps_is_opt_in_and_removes_misread_fine_bits():
    """A misread fine gray plane moves k by one period (±24 px spikes);
    fix_period_jumps=True snaps them back to the local fringe order."""
    basis = HybridBasis()
    frames = identity_frames(basis)
    # a 3 x 8 px patch whose gray code reads one period (24 columns) right
    rows, cols, shifted = slice(8, 11), slice(396, 404), slice(420, 428)
    for i in range(2, 2 + basis.used_bits):
        g = frames[i].copy()
        g[rows, cols] = g[rows, shifted]
        frames[i] = g
    truth = identity_truth(basis)

    raw = basis.decode(frames, "g")
    jumps = np.abs(raw.corr_x - truth) > 12
    assert jumps.sum() == 24 and np.allclose((raw.corr_x - truth)[jumps], 24, atol=0.1)
    assert raw.meta["fix_period_jumps"] is False and raw.meta["period_jumps_fixed_pct"] is None

    fixed = HybridBasis(fix_period_jumps=True).decode(frames, "g")
    np.testing.assert_allclose(fixed.corr_x, truth, atol=0.1)
    assert fixed.meta["period_jumps_fixed_pct"] == pytest.approx(
        100 * jumps.sum() / fixed.mask.sum(), abs=1e-3)
    clean = HybridBasis(fix_period_jumps=True).decode(identity_frames(basis), "g")
    np.testing.assert_allclose(clean.corr_x, truth, atol=0.1)
    assert clean.meta["period_jumps_fixed_pct"] == 0


# ---------------------------------------------------------- generation
def test_generate_layouts():
    P = _seed.load("patterns")
    full_v = P.gray_code_stripes(n_bits=10)
    full_h = P.gray_code_stripes(n_bits=10, orientation="horizontal")

    y = HybridBasis(axis="y")
    pats = y.generate()
    assert len(pats) == y.n_frames() == 16 and y.extent == 1140 and y.stripe_px == 2
    for q in pats[2:]:
        assert q.shape == (1140, 912)
        assert (q == q[:, :1]).all()                 # horizontal stripes: rows constant
    for a, b in zip(pats[2:12], full_h):
        np.testing.assert_array_equal(a, b)

    c = HybridBasis(complementary=True)
    pats = c.generate()
    assert len(pats) == c.n_frames() == 26 and c.complementary()
    for i, g in enumerate(full_v):
        np.testing.assert_array_equal(pats[2 + 2 * i], g)
        np.testing.assert_array_equal(pats[3 + 2 * i], 255 - g)

    k = HybridBasis(coarse_bits=7)
    pats = k.generate()
    assert len(pats) == k.n_frames() == 13 and not k.complementary()
    for a, b in zip(pats[2:9], full_v[3:]):           # the 7 coarsest planes
        np.testing.assert_array_equal(a, b)

    assert HybridBasis(axis="y", complementary=True, coarse_bits=8).n_frames() == 2 + 16 + 4
    assert HybridBasis(axis="y").axes == {"y"} and HybridBasis().axes == {"x"}


# ------------------------------------------------------- configuration
@pytest.mark.parametrize("kw,match", [
    ({"coarse_bits": 6}, r"16 DMD px.*period/2 = 12.*coarse_bits >= 7"),
    ({"axis": "y", "coarse_bits": 7}, r"axis y.*16 DMD px.*coarse_bits >= 8"),
    ({"coarse_bits": 7, "period_px": 16}, r"period/2 = 8.*coarse_bits >= 8"),
    ({"bits": 4}, r"57 DMD px.*use more bits"),
    ({"coarse_bits": 0}, r"coarse_bits must be in 1\.\.bits"),
    ({"coarse_bits": 11}, r"coarse_bits must be in 1\.\.bits"),
    ({"coarse_bits": 7.5}, r"coarse_bits must be an integer"),
    ({"axis": "z"}, r"axis must be 'x'"),
    ({"steps": 2}, r">= 3 steps"),
    ({"min_bit_contrast": -1}, r"min_bit_contrast must be >= 0"),
    ({"sat_level": 0}, r"sat_level must be in \(0, 255\]"),
    ({"sat_level": 300}, r"sat_level must be in \(0, 255\]"),
    ({"sat_level": True}, r"sat_level must be a number"),
    ({"fix_period_jumps": "yes"}, r"fix_period_jumps must be true/false"),
])
def test_invalid_configuration_raises(kw, match):
    with pytest.raises(ValueError, match=match):
        HybridBasis(**kw)


def test_wrong_frame_count_raises():
    basis = HybridBasis(complementary=True, coarse_bits=7)
    with pytest.raises(ValueError, match=r"expects 20 frames .*7 gray planes \+ inverses"):
        basis.decode([np.zeros((4, 4))] * 16, "g")


def test_from_preset_bases_and_registry():
    b = from_preset_bases(DEFAULT_SPECS)
    assert (b.axis, b.complementary(), b.coarse_bits, b.bits) == ("x", False, None, 10)
    assert from_preset_bases(DEFAULT_SPECS, axis="y").axis == "y"

    specs = [{"name": "graycode", "bits": 10, "complementary": True, "axis": "y",
              "coarse_bits": 8, "min_bit_contrast": 3},
             {"name": "phaseshift", "steps": 4, "period_px": 24, "axis": "y"}]
    b = from_preset_bases(specs)
    assert (b.axis, b.complementary(), b.coarse_bits, b.min_bit_contrast) == ("y", True, 8, 3.0)
    assert from_preset_bases(specs, axis="x").axis == "x"      # map runs override
    with pytest.raises(ValueError, match="coarse_bits >= 8"):
        from_preset_bases([dict(specs[0], coarse_bits=7), specs[1]])
    with pytest.raises(ValueError, match="must match"):
        from_preset_bases([specs[0], dict(specs[1], axis="x")])

    h = registry.get({"name": "hybrid", "axis": "y", "coarse_bits": 8,
                      "complementary": True, "min_bit_contrast": 4.0})
    assert isinstance(h, HybridBasis) and h.axes == {"y"} and h.n_frames() == 22
    assert from_preset_bases([{"name": "hybrid", "coarse_bits": 7}], axis="x").coarse_bits == 7


def test_preset_accepts_new_gray_params_and_rejects_bad_values():
    import json
    raw = json.loads(config.resolve_preset_path(PRESET).read_text())
    assert config.load_preset(PRESET, mock=True).bases == raw["bases"]    # shipped preset
    raw["bases"][0].update({"axis": "y", "coarse_bits": 8, "min_bit_contrast": 5,
                            "complementary": True})
    p = preset_from_dict(raw, mock=True)
    assert from_preset_bases(p.bases).n_frames() == 2 + 16 + 4
    for bad in ({"axis": "z"}, {"coarse_bits": 0}, {"coarse_bits": 11},
                {"coarse_bits": True}, {"min_bit_contrast": -1},
                {"complementary": "yes"}, {"coarse_bitz": 7}):
        d = json.loads(json.dumps(raw))
        d["bases"][0].update(bad)
        with pytest.raises(ConfigError):
            preset_from_dict(d, mock=True)


def test_preset_sat_level_and_fix_period_jumps_reach_the_decoder():
    import json
    raw = json.loads(config.resolve_preset_path(PRESET).read_text())
    b = from_preset_bases(preset_from_dict(raw, mock=True).bases)
    assert b.sat_level == SAT_LEVEL and b.fix_period_jumps is False      # defaults
    raw["bases"][0]["fix_period_jumps"] = True
    raw["bases"][1]["sat_level"] = 245
    b = from_preset_bases(preset_from_dict(raw, mock=True).bases, axis="y")
    assert (b.axis, b.sat_level, b.fix_period_jumps) == ("y", 245.0, True)
    raw["bases"][1]["sat_level"] = None                                 # off = v10
    assert from_preset_bases(preset_from_dict(raw, mock=True).bases).sat_level is None
    h = registry.get({"name": "hybrid", "sat_level": 252, "fix_period_jumps": True})
    assert h.sat_level == 252.0 and h.fix_period_jumps
    for i, bad in ((1, {"sat_level": 0}), (1, {"sat_level": 256}), (1, {"sat_level": "hi"}),
                   (1, {"sat_level": True}), (0, {"fix_period_jumps": 1}),
                   (0, {"sat_level": 250})):              # sat_level is a phase option
        d = json.loads(json.dumps(raw))
        d["bases"][i].update(bad)
        with pytest.raises(ConfigError):
            preset_from_dict(d, mock=True)


def test_session_acquire_field_runs_new_modes(eng, preset):
    """The preset spec reaches the capture: complementary + coarse through
    ScanSession.acquire_field (the path capture_base/object use)."""
    preset.camera.average = 1
    preset.bases = [{"name": "graycode", "bits": 10, "complementary": True, "coarse_bits": 7},
                    {"name": "phaseshift", "steps": 4, "period_px": 24}]
    fld = eng.acquire_field(preset, "comp-coarse")
    assert fld.meta["complementary"] and fld.meta["coarse_bits"] == 7
    assert fld.meta["polarity_flipped"]                       # preset invert: like the bench
    frac, cover = truth_score(fld, eng.bench, "x")
    assert frac >= 0.95 and cover >= 0.95, (frac, cover)

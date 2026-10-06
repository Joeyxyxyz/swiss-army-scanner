"""1-bit soft-sync fast-scan pack (engine/bases/onebit.py + tools/make_flash_pack.py
--fastscan): pattern design, dithering stats, the simulated decode error, the
24-bit packing (pack -> BMP -> unpack round trip), zone masks, timing / LED rules
and the verb / CLI wiring. Files only: nothing here touches the rig."""

import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pytest

from engine import api, rules
from engine.bases import onebit as ob
from tools import make_flash_pack as mfp

from conftest import ROOT

H, W = ob.DMD_SHAPE
# a real stereo calibration folder (<library>/calibrations/<geometry>/<ts>_stereo); skipped if unset
REAL_CAL = Path(os.environ.get("SCANNER_TEST_REAL_CAL", "")) if os.environ.get("SCANNER_TEST_REAL_CAL") else ROOT / "_no_real_cal"


@pytest.fixture(scope="module")
def pats():
    return ob.fastscan_patterns()


@pytest.fixture
def ring_files(tmp_path):
    """Two synthetic face rims (DMD col, row) like face_dmd_ring.npy."""
    t = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    out = []
    for i, (cx, cy) in enumerate(((687.0, 977.0), (667.0, 1058.0))):
        pts = np.stack([cx + 18 * np.cos(t), cy + 34 * np.sin(t)], -1)
        f = tmp_path / f"ring{i}.npy"
        np.save(f, pts)
        out.append(f)
    return out


# ------------------------------------------------------------- fine level
def test_fine_profile_is_the_searched_optimum():
    f = ob.fine_profile()
    assert np.array_equal(f, ob.best_fine_profile(4))
    assert all(f[c] == f[(-c) % 8] for c in range(8))          # symmetric: no phase bias
    h = ob.fine_harmonics()
    assert h["f3_over_f1"] < 0.03 and h["f1"] >= 0.85 * h["f1_cosine"]
    # the plain quarter rounding of the cosine (a triangle) would be 6x worse
    tri = np.array([4, 3, 2, 1, 0, 1, 2, 3]) / 4
    F = np.fft.fft(tri)
    assert abs(F[3]) / abs(F[1]) > 5 * h["f3_over_f1"]


def test_fine_fringe_is_column_exact_and_translates():
    s0 = ob.fine_fringe(0)
    assert s0.shape == (H, W) and s0.dtype == bool
    assert H % 4 == 0
    lv = ob.fine_profile()[np.arange(W) % 8]
    # every column lights exactly level x 4 of each 4 rows
    blocks = s0.reshape(H // 4, 4, W).sum(1)
    assert np.array_equal(blocks, np.repeat((lv * 4).astype(int)[None, :], H // 4, 0))
    for k in range(1, 4):
        assert np.array_equal(ob.fine_fringe(k), np.roll(s0, -2 * k, axis=1)), k
    # texture <= 4 rows in y: a column repeats every 4 rows
    assert np.array_equal(s0[4:], s0[:-4])


# -------------------------------------------------- mid / coarse levels
@pytest.mark.parametrize("T", [80, 1280])
def test_fs_steps_are_exact_translates_and_track_the_cosine(T):
    steps = [ob.fs_fringe(T, k) for k in range(4)]
    s = T // 4
    for k in range(1, 4):
        n = W - k * s
        if n > 0:
            assert np.array_equal(steps[k][:, :n], steps[0][:, k * s:k * s + n]), k
    # dithering stats: 16 x 16 block means follow the target within the FS noise
    c = np.arange(W)
    target = 0.5 * (1 + np.cos(2 * np.pi * c / T))
    b = steps[0][:H // 16 * 16, :W // 16 * 16].reshape(H // 16, 16, W // 16, 16).mean((1, 3))
    tb = np.repeat(target[:W // 16 * 16].reshape(-1, 16).mean(1)[None, :], H // 16, 0)
    err = b - tb
    assert abs(float(err.mean())) < 0.01 and float(np.abs(err).max()) < 0.08
    assert 0.3 < steps[0].mean() < 0.7


def test_fs_dither_is_deterministic_and_mean_preserving():
    img = np.full((40, 50), 0.3)
    a, b = ob.fs_dither(img), ob.fs_dither(img)
    assert np.array_equal(a, b) and abs(a.mean() - 0.3) < 0.01


# ------------------------------------------------------------------ slots
def test_slot_set_order_and_meaning(pats):
    assert [p["name"] for p in pats] == list(ob.SLOTS)
    assert [p["slot"] for p in pats] == list(range(16))
    assert [p["decode_index"] for p in pats] == [None, None] + list(range(14))
    assert pats[0]["image"].min() == 255 and pats[3]["image"].max() == 0
    assert all(p["image"].dtype == np.uint8 and set(np.unique(p["image"])) <= {0, 255}
               for p in pats)
    b = ob.basis()
    assert b.steps == 4 and b.periods == [8.0, 80.0, 1280.0] and b.n_frames() == 14
    assert [pats[s]["period_px"] for s in ob.DECODE_SLOTS[2:]] == [8] * 4 + [80] * 4 + [1280] * 4


@pytest.mark.parametrize("sigma", [0.5, 2.0])
def test_sync_code_is_the_only_frame_structured_along_y(pats, sigma):
    """On the simulated camera view (the sorter sees camera frames, not mirrors)."""
    sig = [ob.sync_signature(ob.observe(p["image"], sigma)[0] * 200 + 5) for p in pats]
    others = [g["orientation"] for i, g in enumerate(sig) if i != 1]
    assert sig[1]["orientation"] > 10 * max(others)
    for s in range(4, 16):
        assert sig[s]["orientation"] < 1.0, pats[s]["name"]
    assert sig[0]["mean"] == max(g["mean"] for g in sig)
    assert sig[3]["mean"] == min(g["mean"] for g in sig)


def test_zone_variant_masks_decode_slots_only(ring_files):
    zone, info = mfp.build_zone(ring_files)
    assert info["lit_px"] == int(zone.sum()) > 0
    zp = ob.fastscan_patterns(zone)
    full_sync = {s: ob.slot_image(s).astype(np.uint8) * 255 for s in ob.SYNC_SLOTS}
    for p in zp:
        if p["slot"] in ob.SYNC_SLOTS:                # sync markers stay full field
            assert not p["zone"] and np.array_equal(p["image"], full_sync[p["slot"]])
        else:
            assert p["zone"] and not p["image"][~zone].any(), p["name"]
    full = ob.fastscan_patterns()
    for a, b in zip(zp, full):
        if a["zone"]:                             # same texture inside the zone
            assert np.array_equal(a["image"][zone], b["image"][zone])
    with pytest.raises(ValueError):
        ob.fastscan_patterns(np.zeros((H, W), bool))


def test_ring_ellipse_gets_the_margin(ring_files):
    z0, i0 = mfp.zone_from_rings(ring_files[:1], margin_px=0)
    z6, i6 = mfp.zone_from_rings(ring_files[:1], margin_px=6)
    assert z6.sum() > z0.sum() and (z0 & ~z6).sum() == 0
    cols = np.nonzero(z6.any(0))[0]
    assert abs((cols.max() - cols.min() + 1) - (36 + 12)) <= 2       # 2 x 18 + 2 x 6


def test_band_from_camera_needs_both_axes(tmp_path):
    cam = np.zeros((60, 80), bool)
    cam[30, 10:70] = True
    np.save(tmp_path / "band.npy", cam)
    v, u = np.mgrid[0:60, 0:80].astype(np.float32)
    np.savez(tmp_path / "xy.npz", corr_x=u * 2 + 100, corr_y=v * 3 + 500)
    np.savez(tmp_path / "x.npz", corr_x=u * 2 + 100)
    m, info = mfp.zone_band_from_camera(tmp_path / "band.npy", tmp_path / "xy.npz")
    rr, cc = np.nonzero(m)
    assert 500 + 3 * 25 <= rr.min() and rr.max() <= 500 + 3 * 35 and cc.min() < 130
    assert info["mapped_px"] > 0
    assert len(info["camera_mask_sha256"]) == 64 and len(info["field_sha256"]) == 64
    with pytest.raises(ValueError, match="corr_y"):
        mfp.zone_band_from_camera(tmp_path / "band.npy", tmp_path / "x.npz")
    with pytest.raises(ValueError):
        mfp.build_zone(band_cam=tmp_path / "band.npy")             # field missing


# -------------------------------------------------------------- packing
def test_layout_is_one_image_per_variant_same_planes(ring_files):
    zone, _ = mfp.build_zone(ring_files)
    variants, per, images, seq = mfp.fastscan_layout(zone)
    assert [v for v, _ in variants] == ["full", "zone"] and len(images) == 2
    planes = [f"G{i}" for i in range(8)] + [f"R{i}" for i in range(8)]
    for vi in range(2):
        ents = [e for e in seq if e["image_index"] == vi]
        assert [e["slot"] for e in ents] == list(range(16))
        assert [e["planes"][0] for e in ents] == planes             # slot k -> bit k
        assert [e["bit_position"] for e in ents] == list(range(16))
    assert images[0][..., 2].max() == 0 and images[1][..., 2].max() == 0     # B0..B7 free
    rt = mfp._check_round_trip(images, seq, per)
    assert rt["exact"] and rt["patterns_checked"] == 32
    # G0 = sync_white: bit 0 of green set everywhere; G3 = black: never
    assert (images[0][..., 1] & 1).all() and not (images[0][..., 1] & 8).any()


def test_round_trip_detects_a_flipped_bit(pats):
    variants, per, images, seq = mfp.fastscan_layout(None)
    bad = images[0].copy()
    bad[500, 400, 0] ^= 1 << 3                     # R3 = slot 11 (mid step 3)
    with pytest.raises(RuntimeError, match="slot 11"):
        mfp._check_round_trip([bad], seq, per)
    stray = images[0].copy()
    stray[10, 10, 2] |= 1                          # a bit on a free plane (B0)
    with pytest.raises(RuntimeError, match="outside"):
        mfp._check_round_trip([stray], seq, per)


def test_a_g_r_plane_swap_only_rotates_the_loop(pats):
    """If the GUI read G0..G7 from the BMP's red byte (and R from green), the
    controller would play slots 8..15 then 0..7: a rotation of the repeating
    loop. Anchoring on the sync pair recovers the play order exactly."""
    variants, per, images, seq = mfp.fastscan_layout(None)
    swapped = images[0][..., [1, 0, 2]]                    # red <-> green bytes
    played = mfp.unpack([swapped], seq)                    # the GUI's G0..G7, R0..R7
    order = mfp.plane_swap_play_order()
    assert order == list(range(8, 16)) + list(range(8))
    for j, img in enumerate(played):
        assert np.array_equal(img, pats[order[j]]["image"]), j
    code = [j for j, img in enumerate(played) if np.array_equal(img, pats[1]["image"])]
    assert len(code) == 1
    start = (code[0] - 1) % 16                             # sync_white precedes sync_code
    realigned = [played[(start + i) % 16] for i in range(16)]
    assert all(np.array_equal(a, p["image"]) for a, p in zip(realigned, pats))


def test_led_start_is_duty_safe_and_higher_values_warn():
    b = mfp.fastscan_led()
    assert b["start"] == mfp.FASTSCAN_LED_START and not b["warnings"]
    for ch, v in b["video_equivalent_of_start"].items():
        assert v <= rules.LED_AVG_MAX[ch]                  # 3x duty stays inside continuous
    hot = mfp.fastscan_led({"r": 100, "g": 150, "b": 150})
    assert len(hot["warnings"]) == 3 and hot["start"]["g"] == 150
    assert mfp.fastscan_led({"g": 55})["start"] == {"r": 35, "g": 55, "b": 45}
    with pytest.raises(rules.SafetyViolation):
        mfp.fastscan_led({"r": 121, "g": 50, "b": 45})     # above the continuous limit


def test_make_fastscan_pack_writes_and_verifies(tmp_path, ring_files, capsys):
    d = mfp.make_fastscan_pack(zone_rings=ring_files, out_root=tmp_path, label="t")
    d2 = mfp.make_fastscan_pack(out_root=tmp_path, label="t")
    assert d != d2                                                     # never overwrite
    names = {p.name for p in d.iterdir()}
    assert {"images", "manifest.json", "procedure.txt", "zone_mask.png",
            "preview_full.png", "preview_zone.png"} <= names
    m, images = mfp.load_pack(d)
    assert m["mode"] == mfp.FASTSCAN_MODE and len(images) == 2 and len(m["sequence"]) == 32
    assert all(im["file"].endswith(".bmp") for im in m["images"])
    assert (d / m["images"][0]["file"]).stat().st_size == 54 + 912 * 1140 * 3
    assert m["round_trip"]["from_disk"]["exact"]
    assert mfp.verify_pack(d) == 0
    assert mfp.main(["--unpack-check", str(d)]) == 0
    assert "round trip OK" in capsys.readouterr().out
    t = m["timing"]
    mfp.validate_timing(1, t["pattern_exposure_us"], t["pattern_period_us"])
    assert t["loop_ms"] == 160 and t["repeat"] and t["trigger"] == "internal"
    rules.validate_led({"mode": "continuous", **m["led"]["start"]})
    assert m["decode"]["order"][0] == "white" and len(m["decode"]["order"]) == 14
    proc = (d / "procedure.txt").read_text(encoding="utf-8")
    assert "firmware link" in proc and "BACK UP FIRST" in proc and "R7" in proc
    for gui_words in ("Image / Firmware", "Firmware Build", "Firmware Upload", "Pattern Source: Flash",
                      "Internal/External", "Play Once: OFF", "Validate Sequence", "PLANE CHECK",
                      "Video Mode", "ROLLBACK"):
        assert gui_words in proc, gui_words
    assert m["planes"]["if_gui_swaps_g_and_r"]["play_order_slots"][0] == 8
    assert not m["led"]["warnings"]
    # a changed BMP is caught by its hash
    f = d / m["images"][1]["file"]
    raw = bytearray(f.read_bytes())
    raw[1000] ^= 0xFF
    f.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="sha256"):
        mfp.verify_pack(d)
    m2, im2 = mfp.load_pack(d2)
    assert list(m2["variants"]) == ["full"] and len(im2) == 1


def test_library_output_bin_and_index(tmp_path):
    lib = tmp_path / "library"
    d = mfp.make_fastscan_pack(library=lib, label="bin")
    assert d.parent == lib / "output" / "flashpacks"
    rows = (lib / "output" / "index.csv").read_text(encoding="utf-8")
    assert "flashpacks" in rows and "manifest.json" in rows


def test_timing_and_led_rules():
    t = mfp.fastscan_timing()
    assert t["camera"]["max_frame_period_us"] >= t["camera"]["frame_period_us"]
    assert t["capture_ms"] <= 500 and not t["warnings"]
    with pytest.raises(mfp.TimingError):
        mfp.fastscan_timing(period_us=6000, camera_exposure_us=3000)   # no clean frame
    with pytest.raises(mfp.TimingError):
        mfp.fastscan_timing(period_us=200)                              # < 235 us 1-bit
    assert mfp.fastscan_timing(period_us=20000)["warnings"]
    with pytest.raises(rules.SafetyViolation):
        mfp.make_fastscan_pack(led={"r": 70, "g": 200, "b": 90}, out_root=Path("unused"))


# ------------------------------------------------ simulated decode error
@pytest.fixture(scope="module")
def sim():
    return ob.study(sigmas=(0.5, 1.0), parities=(None, 0, 1))


def test_study_fine_level_beats_the_square_wave(sim):
    s05, s10 = sim["sigmas"]["0.5"], sim["sigmas"]["1.0"]
    assert s05["fine"]["rms_cols"] < 0.03 and s10["fine"]["rms_cols"] < 0.005
    assert abs(s05["fine"]["bias_cols"]) < 1e-3
    assert s05["fine_square_wave"]["rms_cols"] > 5 * s05["fine"]["rms_cols"]
    assert s10["fine_square_wave"]["rms_cols"] > 5 * s10["fine"]["rms_cols"]
    assert s05["fine"]["modulation"] > 0.4


def test_study_order_levels_stay_inside_their_unwrap_tolerance(sim):
    for row in sim["sigmas"].values():
        assert row["mid"]["max_cols"] < row["mid"]["tolerance_cols"] / 4
        assert row["coarse"]["max_cols"] < row["coarse"]["tolerance_cols"] / 2
        e = row["end_to_end"]
        assert e["unwrap_rejected_pct"] < 1.0 and abs(e["median_err_cols"]) < 0.01
        assert e["max_abs_err_cols"] < 0.1


def test_square_wave_closed_form_matches_the_simulation(sim):
    """The 4-step harmonic error of a binary square wave: closed form (harmonics
    3, 5, 7, ... through the optics MTF) vs the rendered simulation."""
    ideal = ob.square_wave_error(0.0, cam_px_per_col=1e9)      # mirror box only
    assert abs(ideal["a3_over_a1"] - np.sinc(3 / 8) / np.sinc(1 / 8) / 3) < 1e-4
    sharp = ob.square_wave_error(0.0)                          # + the camera pixel
    assert 0.3 < sharp["peak_cols"] < ideal["peak_cols"] < 0.55   # ~0.4 col = ~0.7 mm
    prev = None
    for sg in (0.5, 0.75, 1.0, 1.5, 2.0):
        cf = ob.square_wave_error(sg)
        assert prev is None or cf["rms_cols"] < prev
        prev = cf["rms_cols"]
    r = sim["sigmas"]["1.0"]["fine_square_wave"]
    assert 0.7 < r["closed_form"]["rms_cols"] / r["rms_cols"] < 1.3


def test_diamond_rows_give_a_constant_bias_not_a_ripple(sim):
    """On the diamond DMD the 1-bit fine tile reads a constant offset against the
    HDMI 8-bit fringes (sign = row parity); relative heights do not see it."""
    for sg in ("0.5", "1.0"):
        v = sim["sigmas"][sg]["fine_vs_hdmi"]
        assert abs(v["none"]["bias_cols"]) < 0.005
        assert 0.03 < -v["0"]["bias_cols"] < 0.06 and 0.03 < v["1"]["bias_cols"] < 0.06
        assert v["0"]["rms_cols"] < 0.05 and v["1"]["rms_cols"] < 0.05
    assert sim["sigmas"]["1.0"]["fine_vs_hdmi"]["0"]["rms_cols"] < 0.01


def test_optional_fine_prefilter_cuts_the_texture_at_sharp_focus():
    st = ob.study(sigmas=(0.5,), parities=(None,), fine_prefilter=(ob.FINE_PREFILTER_PX,))
    row = st["sigmas"]["0.5"]
    pf = row["fine_prefiltered"][str(ob.FINE_PREFILTER_PX)]["vs_ideal"]
    assert pf["rms_cols"] < 0.4 * row["fine"]["rms_cols"]
    assert pf["noise_rms_cols"] < row["fine"]["noise_rms_cols"]
    rng = np.random.default_rng(1)
    fr = [rng.random((30, 40)).astype(np.float32) for _ in range(14)]
    out = ob.prefilter_frames(fr, fine_px=1.0)
    assert all(np.array_equal(out[i], fr[i]) for i in (0, 1))
    assert all(out[i].std() < fr[i].std() for i in range(2, 14))
    assert all(np.array_equal(a, b) for a, b in zip(ob.prefilter_frames(fr)[:6], fr[:6]))


def test_noise_model():
    assert abs(ob.noise_rms_cols(0.5, 8, white_dn=200, noise_dn=2.0, frames=4)
               - 8 / (2 * np.pi) * np.sqrt(0.5) * 1.0 / 100.0) < 1e-12
    assert ob.noise_rms_cols(0.0, 8) == float("inf")


def test_prefilter_touches_only_mid_and_coarse():
    rng = np.random.default_rng(0)
    fr = [rng.random((30, 40)).astype(np.float32) for _ in range(14)]
    out = ob.prefilter_frames(fr)
    for i in range(6):
        assert np.array_equal(out[i], fr[i])
    for i in range(6, 14):
        assert out[i].std() < fr[i].std()
    with pytest.raises(ValueError):
        ob.prefilter_frames(fr[:13])
    assert len(ob.decode_order(list(range(16)))) == 14


@pytest.mark.skipif(not (REAL_CAL / "stereo_calibration.json").is_file(),
                    reason="bench stereo calibration not on this machine")
def test_calibration_scale_from_the_bench_calibration():
    s = mfp.calibration_scale(REAL_CAL, 780.0)
    assert 1.3 < s["cam_px_per_col"] < 1.8 and 1.6 < s["mm_per_col"] < 2.1
    assert 0.3 < s["camera_mm_per_px"] < 0.42          # ~0.36 mm per camera px, not px per column
    assert s["geometry_id"] == "bench_9in_v1" and s["diamond_parity"] is None


# ----------------------------------------------------- rule 1 / wiring
def test_operator_guide_covers_the_gui_steps():
    doc = (ROOT / "docs" / "FLASH_FASTSCAN_4500.md").read_text(encoding="utf-8")
    for must in ("firmware link", "Back up", "Rollback", "Firmware Build", "Firmware Upload",
                 "Pattern Source", "Flash", "Internal/External", "Play Once", "Validate Sequence",
                 "Video Mode", "Parallel RGB", "R120 G180 B160", "G0", "R7", "never mid-scan",
                 "make_flash_pack.py --fastscan", "--unpack-check", "Skip Bootloader Update"):
        assert must in doc, must


def test_no_projector_usb_code_in_the_pack_path():
    for f in (ROOT / "engine" / "bases" / "onebit.py", ROOT / "tools" / "make_flash_pack.py"):
        src = f.read_text(encoding="utf-8")
        imports = "\n".join(l for l in src.splitlines() if re.match(r"\s*(from|import)\s", l))
        for bad in ("dmd_control", "hid", "dlp_driver", "swiss_army_sensor", "PySpin"):
            assert bad not in imports, (f.name, bad)
    rules.check_scan_path_usb_free()


def test_api_verb_writes_to_the_engine_library(eng, ring_files):
    p = api.make_fastscan_pack(zone_rings=tuple(str(f) for f in ring_files), label="verb")
    assert Path(p).parent == eng.store.root / "output" / "flashpacks"
    m = json.loads((Path(p) / "manifest.json").read_text(encoding="utf-8"))
    assert list(m["variants"]) == ["full", "zone"]


def test_cli_fastscan_pack(tmp_path, capsys):
    from ui import cli
    lib = tmp_path / "library"
    try:
        code = cli.main(["--mock", "--library", str(lib), "--json", "fastscan-pack",
                         "--label", "cli"])
        out = capsys.readouterr().out
    finally:
        api._session = None
    assert code == 0, out
    doc = json.loads(out)
    assert Path(doc["folder"]).parent == lib / "output" / "flashpacks"
    assert doc["variants"] == ["full"]

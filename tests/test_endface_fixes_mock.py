"""End-face inspection fixes from the bench of 2026-10-12 (a 21 mm class cylinder whose flat
end face is ~19.5 mm across inside a rounded rim), MOCK / synthetic only:

  1. radii     every detector radius derives from the face diameter (radii.py): faces of 16,
               19.5, 21.05 and 30 mm - no rim false triggers on good faces, dents still found;
               the former fixed radii inspect the chamfer of a smaller face (false triggers)
               and leave a ring of a larger one unchecked; explicit values are kept, one rim
               radius for every detector, the resolved set in the golden and in each check
  2. exposure  a frame-locked camera exposure per face (exposure.py): the search (dim ->
               longer, clipped -> shorter, glint / mirror / too dark -> none, with the reason)
               and, on the mock bench, a dim face scanned at 100 ms instead of 16.7 ms, a face
               that glints refused (PULL, nothing stored), inverted presets, a forced exposure
  3. size      the flood outline against the known diameter (outline.size_check): a face
               that reads 16 % small is a size mismatch - PULL (never PASS, never a definite
               FAIL), said by locate / profile / check / golden; a mock scene seen through a
               lens zoomed after the calibration
  4. rim       the outline's roundness (zone rim_outline): notches / bulges of 0.2 / 0.4 /
               0.8 mm, the golden's per-angle model, a placement without the outline
  5. depth     depth_centre_mm (3 x 3 centre average, local background) is THE depth, the
               deepest single point over-reads; thresholds apply to the centre depth

The real rig is never opened."""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

import face_synth as S
from conftest import FIXTURE_PRESETS
from engine import api, badparts, config
from engine.capture.mock_bench import MockBench
from engine.capture.mock_stereo import EndFace, MetricCylinder, Plane
from engine.measure.endface import depth as depth_mod
from engine.measure.endface import exposure as X
from engine.measure.endface import golden as golden_mod
from engine.measure.endface import locate as locate_mod
from engine.measure.endface import mfcfar
from engine.measure.endface import outline as outline_mod
from engine.measure.endface import profile as profile_mod
from engine.measure.endface import radii as radii_mod
from engine.measure.endface import verbs as V

SIZES = (16.0, 19.5, 21.05, 30.0)
POSES = {"g1": ((-6.0, 4.0, 780.0), (10, 6)), "g2": ((-4.5, 5.0, 781.0), (14, 5)),
         "g3": ((-6.5, 3.0, 779.0), (8, 9)), "g4": ((-5.0, 4.5, 780.5), (12, 8)),
         "g5": ((-5.5, 3.5, 780.0), (11, 7))}


@pytest.fixture(autouse=True)
def _no_session_left():
    yield
    api._session = None


# ============================================================ 1. radii (pure)

def test_radii_are_fractions_of_the_face_radius():
    r = radii_mod.face_radii(10.525)                        # the face they were developed on
    assert r["inspect_r_mm"] == pytest.approx(9.75) and r["fit_r_mm"] == pytest.approx(9.75)
    assert r["rim_r_mm"] == pytest.approx(9.0) and r["plane_r_mm"] == pytest.approx(9.0)
    assert r["r_flat_mm"] == pytest.approx(9.0) and r["battery_r_mm"] == pytest.approx(9.0)
    assert r["fir_r_max_mm"] == pytest.approx(9.5)
    assert r["noise_bands_mm"] == pytest.approx((0.0, 3.0, 6.0, 8.0, 9.0, 9.75, 10.25))
    assert r["given"] == []
    for D in SIZES:
        R = D / 2
        r = radii_mod.face_radii(R)
        assert r["inspect_r_mm"] / R == pytest.approx(9.75 / 10.525)        # 0.926 R
        assert r["rim_r_mm"] / R == pytest.approx(9.0 / 10.525)             # 0.855 R
        assert r["plane_r_mm"] == r["rim_r_mm"] and r["fit_r_mm"] == r["inspect_r_mm"]
        b = r["noise_bands_mm"]
        assert b[0] == 0 and all(y > x for x, y in zip(b[:-1], b[1:])) and b[-1] > r["inspect_r_mm"]
        assert r["rim_r_mm"] in b and r["inspect_r_mm"] in b
        prm = mfcfar.mf_params(radius_mm=R)
        assert prm["inspect_r_mm"] == r["inspect_r_mm"] and prm["noise_bands_mm"] == b
    # explicit values are kept, and what follows from them follows
    r = radii_mod.face_radii(9.75, {"inspect_r_mm": 8.6, "rim_r_mm": 8.0})
    assert (r["inspect_r_mm"], r["fit_r_mm"], r["rim_r_mm"], r["plane_r_mm"]) == (8.6, 8.6, 8.0, 8.0)
    assert r["given"] == ["inspect_r_mm", "rim_r_mm"] and 8.6 in r["noise_bands_mm"]
    assert radii_mod.face_radii(9.75, {"inspect_r_mm": 8.0})["rim_r_mm"] < 8.0   # stays inside
    prm = mfcfar.mf_params({"rim_r_mm": 8.0}, radius_mm=9.75)
    assert prm["rim_r_mm"] == 8.0 and prm["inspect_r_mm"] == pytest.approx(0.926 * 9.75, abs=0.01)
    # never clamped silently, never assumed
    with pytest.raises(ValueError, match="outside the inspected zone"):
        radii_mod.face_radii(9.75, {"rim_r_mm": 9.5})
    with pytest.raises(ValueError, match="not on the 19.5 mm face"):
        radii_mod.face_radii(9.75, {"inspect_r_mm": 10.5})
    with pytest.raises(ValueError, match="unknown end-face radii"):
        radii_mod.face_radii(9.75, {"inspekt": 9.0})
    with pytest.raises(ValueError, match="never assumed"):
        mfcfar.mf_params()
    assert profile_mod.flat_radius(9.75) == pytest.approx(0.855 * 9.75, abs=0.01)
    assert profile_mod.flat_radius(9.75, 7.0) == 7.0


def _face_map(g, R: float, seed: int, dent=None) -> np.ndarray:
    """A synthetic end-face height map (mm): a flat with pixel-scale texture (0.012 mm rms)
    out to 0.94 R, then the chamfer / rounded rim: steep, different in every placement (how
    deep, and toward which side) and rough (grazing, glinting: 0.1 mm of mm-scale relief) -
    what the bench faces show beyond their flat."""
    import cv2
    rng = np.random.default_rng(seed)
    GU, GV, RR = mfcfar.grid_mesh(g)
    H = cv2.GaussianBlur(rng.normal(0, 1, RR.shape), (0, 0), 1.2)
    H = 0.012 * H / H.std()
    a, ph = rng.uniform(0.15, 0.45), rng.uniform(0, 2 * np.pi)
    ramp = np.clip((RR - 0.94 * R) / (0.06 * R), 0, 1)
    rough = cv2.GaussianBlur(rng.normal(0, 1, RR.shape), (0, 0), 2.0)
    H = H - ramp ** 1.5 * a * (1 + 0.6 * np.cos(np.arctan2(GV, GU) - ph)) \
        + np.clip(4 * ramp, 0, 1) * 0.10 * rough / rough.std()
    if dent is not None:
        H = H + golden_mod.bowl(g, *dent)
    H[RR > R] = np.nan
    return H


@pytest.mark.parametrize("D", SIZES)
def test_scaled_radii_no_rim_false_triggers_and_dents_found(D):
    R = D / 2
    g = profile_mod.grid_axis(R, 0.25)
    prm = mfcfar.mf_params(radius_mm=R)
    model = mfcfar.build_model([_face_map(g, R, s) for s in range(1, 6)], g, prm)
    for s in range(11, 17):                                 # good placements: nothing
        r = mfcfar.detect(_face_map(g, R, s), model, g, prm)
        assert r["triggers"] == [] and r["max_z"] < prm["z_thresh"], (D, s, r["triggers"])
        assert r["coverage_pct"] > 99
    # dents are still found: mid-face, and in the rim zone just inside the inspected radius
    for u, v, zone in ((0.4 * R, -0.3 * R, "face"), (0.0, 0.885 * R, "rim")):
        r = mfcfar.detect(_face_map(g, R, 21, (0.15, 2.0, u, v)), model, g, prm)
        assert len(r["triggers"]) == 1, (D, zone, r["triggers"])
        t = r["triggers"][0]
        assert np.hypot(t["u_mm"] - u, t["v_mm"] - v) < 0.8 and t["zone"] == zone
        assert t["kind"] == "dent" and t["z"] > prm["z_thresh"]
    # the former fixed radii (right for a 21.05 mm face only)
    old = mfcfar.mf_params({"inspect_r_mm": 9.75, "fit_r_mm": 9.75, "rim_r_mm": 9.0,
                            "plane_r_mm": 9.0,
                            "noise_bands_mm": (0.0, 3.0, 6.0, 8.0, 9.0, 9.75, 10.25)})
    old_model = mfcfar.build_model([_face_map(g, R, s) for s in range(1, 6)], g, old)
    false = sum(len(mfcfar.detect(_face_map(g, R, s), old_model, g, old)["triggers"])
                for s in range(11, 17))
    if D < 21:                      # they inspect the chamfer: good parts trigger
        assert false > 0, D
    if D > 25:                      # they stop at 9.75 mm: a dent further out is never seen
        far = (0.15, 2.0, 0.80 * R, 0.0)
        assert not mfcfar.detect(_face_map(g, R, 21, far), old_model, g, old)["triggers"]
        assert len(mfcfar.detect(_face_map(g, R, 21, far), model, g, prm)["triggers"]) == 1


# ============================================================ 5. dent depth (pure)

def test_dent_depth_is_the_centre_average_not_the_deepest_point():
    """The bench dent (0.46 mm deep, 1.33 mm wide by a depth gauge): the deepest single grid
    point reads deeper than the 3 x 3 centre average in every noisy map; the centre average
    is the depth of the dent as the grid resolves it."""
    import cv2
    g = np.arange(-40, 41) * 0.25
    GU, GV = np.meshgrid(g, g)
    cu, cv_ = 2.1, -1.2
    truth = -0.46 * np.clip(1 - (np.hypot(GU - cu, GV - cv_) / (1.33 / 2)) ** 2, 0, None)
    seen = cv2.GaussianBlur(truth, (0, 0), 0.6)             # the camera's pixel footprint
    y, x = np.unravel_index(int(np.argmin(seen)), seen.shape)
    box = float(-depth_mod.box3(seen)[y, x])                # what a 3 x 3 average can read
    centre, deepest, width = [], [], []
    for seed in range(30):
        rng = np.random.default_rng(seed)
        noise = cv2.GaussianBlur(rng.normal(0, 1, seen.shape), (0, 0), 1.2)
        m = depth_mod.dent_metrics(seen + 0.03 * noise / noise.std(), g, y + 1, x - 1, 1.5, 1,
                                   px_mm=0.36)
        centre.append(m["depth_centre_mm"])
        deepest.append(m["depth_min_mm"])
        width.append(m["width_half_depth_mm"])
        assert m["depth_min_mm"] >= m["depth_centre_mm"] > 0
        assert abs(m["centre_u_mm"] - cu) < 0.6 and abs(m["centre_v_mm"] - cv_) < 0.6
    assert abs(np.mean(centre) - box) < 0.03 and np.mean(centre) < 0.46     # no over-read
    assert np.mean(deepest) > np.mean(centre) + 0.04                         # the single point
    assert 0.8 < np.mean(width) < 1.7                                        # ~1.33 mm + blur
    assert m["depth_min_note"] == "single point, over-reads"
    assert "one camera px" in m["width_note"] and "0.36" in m["width_note"]
    # a bump is the same with the sign turned; an unmeasured dent has no centre depth
    b = depth_mod.dent_metrics(-seen, g, y, x, 1.5, -1)
    assert b["depth_centre_mm"] == pytest.approx(-box, abs=0.01) and b["depth_min_mm"] < 0
    hole = np.where(np.hypot(GU - cu, GV - cv_) < 1.0, np.nan, seen)
    assert depth_mod.dent_metrics(hole, g, y, x, 1.5, 1)["depth_centre_mm"] is None


def test_trigger_depth_keys_and_thresholds_apply_to_the_centre_depth():
    R = 10.525
    g = profile_mod.grid_axis(R, 0.25)
    model = mfcfar.build_model([_face_map(g, R, s) for s in range(1, 6)], g,
                               mfcfar.mf_params(radius_mm=R))
    H = _face_map(g, R, 31, (0.30, 2.0, -3.0, 2.0))
    r = mfcfar.detect(H, model, g, mfcfar.mf_params(radius_mm=R), px_mm=0.38)
    (t,) = r["triggers"]
    assert t["depth_mm"] == pytest.approx(t["depth_centre_mm"], abs=1e-3)    # the alias
    assert t["depth_min_mm"] > t["depth_centre_mm"] > 0.2 and t["depth_source"] == "centre_3x3"
    assert t["depth_min_note"] == "single point, over-reads" and "camera px" in t["width_note"]
    assert 1.0 < t["width_half_depth_mm"] < 2.2 and t["depth_fit_mm"] > 0 and t["width_mm"] >= 1.5
    assert r["candidates"][0]["trigger"] is True
    # min_depth_mm between the centre depth and the deepest point: NOT a trigger (the
    # threshold is on depth_centre_mm), but the region is still reported as a candidate
    between = 0.5 * (t["depth_centre_mm"] + t["depth_min_mm"])
    r2 = mfcfar.detect(H, model, g, mfcfar.mf_params({"min_depth_mm": between}, radius_mm=R))
    assert r2["triggers"] == [] and r2["candidates"][0]["trigger"] is False
    assert r2["candidates"][0]["depth_centre_mm"] == t["depth_centre_mm"]
    r3 = mfcfar.detect(H, model, g, mfcfar.mf_params({"min_depth_mm": t["depth_centre_mm"] - 0.01},
                                                     radius_mm=R))
    assert len(r3["triggers"]) == 1
    # the band detector reports the same depth keys
    dev = np.where(np.isfinite(H), H, np.nan) - model["map_mm"]
    bt = golden_mod.detect_face_triggers(dev, np.full(dev.shape, 0.1),
                                         {"grid_mm": g, "radius_mm": R, "rim_r_mm": 9.0,
                                          "min_area_mm2": 0.25})
    assert bt and bt[0]["depth_mm"] == bt[0]["depth_centre_mm"] > 0.2
    assert bt[0]["depth_min_mm"] >= bt[0]["depth_centre_mm"] and bt[0]["kind"] == "dent"
    assert bt[0]["dev_mm"] < 0 and bt[0]["width_half_depth_mm"] > 0.8


# ============================================================ 2. exposure search (pure)

def _linear(p95_16ms: float, clip=lambda e: 0.0):
    return lambda e: {"p95_dn": min(255.0, p95_16ms * e / 16667), "p50_dn": 0.8 * p95_16ms * e / 16667,
                      "clipped_pct": float(clip(e)), "n_face_px": 3000}


def test_exposure_candidates_are_whole_video_frames():
    c = X.candidates()
    assert c == [16667, 33333, 50000, 66667, 83333, 100000]
    assert all(config.frame_locked(e) for e in c)
    assert X.candidates((20000, 60000)) == [33333, 50000]
    assert X.candidates((16667, 16667)) == [16667]
    for bad in ((1000, 5000), (0, 100000), (50000, 20000), "x"):
        with pytest.raises(ValueError):
            X.candidates(bad)
    lo, hi = X.band(170.0)
    assert 90 < lo < 100 and 235 < hi <= 240


def test_exposure_search_dim_bright_clipped_and_refused():
    c = X.candidates()
    r = X.choose(_linear(30.0), c, 16667)                   # dim: 30 DN at 16.7 ms
    assert r["ok"] and r["exposure_us"] == 100000 and r["p95_dn"] == pytest.approx(180, abs=1)
    assert len(r["shots"]) <= 3 and r["shots"][0]["exposure_us"] == 16667
    assert not r["shots"][0]["ok"] and "too dark" in r["shots"][0]["why"]
    r = X.choose(_linear(55.0), c, 16667)                   # needs 50 ms
    assert r["ok"] and r["exposure_us"] == 50000 and len(r["shots"]) == 2
    r = X.choose(_linear(150.0), c, 16667)                  # fine as it is: one shot
    assert r["ok"] and r["exposure_us"] == 16667 and len(r["shots"]) == 1
    r = X.choose(_linear(80.0), c, 50000)                   # the preset is too long: shorter
    assert r["ok"] and r["exposure_us"] == 33333
    # clipping decides: 200 DN at 33 ms would be nearer the target, but a glint clips there
    r = X.choose(_linear(100.0, lambda e: 0.0 if e < 30000 else 2.4), c, 16667)
    assert r["ok"] and r["exposure_us"] == 16667
    assert [s["exposure_us"] for s in r["shots"]] == [16667, 33333]
    assert "clipped" in r["shots"][1]["why"]
    # no exposure fits: why, in the operator's words, and every shot
    r = X.choose(_linear(10.0), c, 16667)
    assert not r["ok"] and r["exposure_us"] is None and "too dark at the longest" in r["why"]
    r = X.choose(_linear(900.0, lambda e: 97.0), c, 50000)  # a mirror: clipped at any exposure
    assert not r["ok"] and "mirrors the projector" in r["why"]
    assert [s["exposure_us"] for s in r["shots"]] == [16667, 33333, 50000]
    r = X.choose(_linear(60.0, lambda e: 1.5 * e / 16667), c, 16667)    # dim AND glinting
    assert not r["ok"] and "glints before it is bright enough" in r["why"]
    r = X.choose(_linear(70.0, lambda e: 0.0 if e < 30000 else 3.0), c, 16667)
    assert not r["ok"] and "too dark without clipping" in r["why"]
    assert all(len(X.choose(_linear(p), c, s)["shots"]) <= len(c)
               for p in (5, 40, 120, 400) for s in (16667, 60000, 100000))


def test_face_stats_and_the_preset_gate():
    f = np.full((60, 60), 100.0)
    yy, xx = np.mgrid[0:60, 0:60]
    rr = np.hypot(xx - 30, yy - 30)
    f[rr < 4] = 255.0
    st = X.face_stats(f, rr < 16, rr < 20)
    assert st["p50_dn"] == 100 and 3.5 < st["clipped_pct"] < 4.5       # (4 / 20) ** 2
    assert "clipped" in X.judge(st, 170.0, 0.5) and X.judge({**st, "clipped_pct": 0.0,
                                                           "p95_dn": 170.0}, 170.0, 0.5) == ""
    with pytest.raises(ValueError):
        X.face_stats(f, np.zeros_like(f, bool), rr < 20)
    p = config.load_preset("cell_wall_2p7ft_vstripe", presets_dir=FIXTURE_PRESETS, mock=True)
    ok, why = X.allowed(p)                                  # 2000 us: not whole video frames
    assert not ok and "not a whole number" in why
    p2 = dataclasses.replace(p, camera=dataclasses.replace(p.camera, exposure_us=33333))
    assert X.allowed(p2) == (True, "")
    with pytest.raises(ValueError):
        X.check_forced(0)


# ============================================================ 3 + 4. the outline (pure)

def _outline(rg, c, n, R, box, nominal=None, shift=(0.4, -0.3), **kw):
    e1, e2 = locate_mod.face_axes(np.asarray(n, float))
    C = np.asarray(c, float) + shift[0] * e1 + shift[1] * e2     # a locate is never exact
    img = S.flood_image(rg, c, n, box, radius=R, **kw)
    to_px = outline_mod.ray_plane_map(rg.camera.K, rg.camera.dist, C, e1, e2, box[:2])
    return outline_mod.measure(img.astype(np.float32), to_px, nominal or R), img, to_px


@pytest.mark.parametrize("D", SIZES)
def test_outline_reads_the_diameter_and_is_round(D):
    rg = S.rig()
    R = D / 2
    for i, (c, (tx, ty)) in enumerate(list(POSES.values())[:2]):
        box = S.crop_box(rg, c, R + 12)
        o, _, _ = _outline(rg, c, S.normal_from_tilt(tx, ty), R, box, seed=i)
        assert o["ok"] and o["round_ok"] and o["valid_pct"] == 100
        assert o["diameter_mm"] == pytest.approx(D, abs=0.06)
        assert o["centre_offset_mm"] == pytest.approx([-0.4, 0.3], abs=0.05)
        assert o["rms_mm"] < 0.03 and o["max_abs_mm"] < 0.1          # sub-pixel (0.32 mm / px)
        assert o["ellipse"]["axis_ratio"] > 0.995
        j = outline_mod.judge(o)
        assert j["status"] == "ok" and not j["findings"] and j["arc_peak_mm"] < 0.1
        sc = outline_mod.outline_size(o, D)
        assert sc["status"] == "ok" and abs(sc["deviation_pct"]) < 0.5


@pytest.mark.parametrize("depth, expect", [(0.2, "ok"), (-0.2, "ok"), (0.4, "near"), (-0.4, "near"),
                                           (0.8, "fail"), (-0.8, "fail")])
def test_rim_notches_and_bulges(depth, expect):
    """A notch (the outline moved in) / bulge (out) over 4 mm of the rim of a 19.5 mm face:
    0.2 mm is inside the tolerance (0.4 mm over >= 1.5 mm of arc), 0.8 mm is a finding at the
    right place with the right sign, 0.4 mm is AT the tolerance: never passed."""
    rg = S.rig()
    c, n, R = POSES["g1"][0], S.normal_from_tilt(*POSES["g1"][1]), 9.75
    box = S.crop_box(rg, c, R + 12)
    for th in (30.0, 200.0):
        o, _, _ = _outline(rg, c, n, R, box, seed=int(th),
                           rim_notches=[{"theta_deg": th, "depth_mm": depth, "arc_mm": 4.0}])
        assert o["round_ok"] and o["diameter_mm"] == pytest.approx(19.5, abs=0.08)
        assert o["max_abs_mm"] == pytest.approx(abs(depth), abs=0.06)
        j = outline_mod.judge(o)
        if expect == "ok":
            assert j["status"] == "ok" and j["arc_peak_mm"] < 0.3
            assert outline_mod.judge(o, tol_mm=0.15)["status"] == "fail"    # a tighter tolerance
            continue
        if expect == "near":
            assert j["status"] in ("marginal", "fail")                      # never 'ok'
            assert outline_mod.judge(o, tol_mm=0.3)["status"] == "fail"
            assert outline_mod.judge(o, guard_band_pct=0)["status"] in ("ok", "fail")
            f = (j["findings"] or j["marginal"])[0]
        else:
            assert j["status"] == "fail" and len(j["findings"]) == 1
            f = j["findings"][0]
            assert f["arc_mm"] >= 2.4 and abs(f["depth_mm"]) == pytest.approx(0.8, abs=0.06)
            assert np.hypot(f["u_mm"] - R * np.cos(np.radians(th)),
                            f["v_mm"] - R * np.sin(np.radians(th))) < 1.2
        assert f["kind"] == ("notch" if depth > 0 else "bulge")
        assert np.sign(f["depth_mm"]) == -np.sign(depth)                    # - = inside
        assert abs((f["theta_deg"] - th + 180) % 360 - 180) < 8
    # a notch shorter than the minimum arc is reported in the max, not as a finding
    o, _, _ = _outline(rg, c, n, R, box, seed=3,
                       rim_notches=[{"theta_deg": 90.0, "depth_mm": 0.8, "arc_mm": 1.0}])
    j = outline_mod.judge(o)
    assert j["status"] == "ok" and j["max_abs_mm"] > 0.3 and j["arc_peak_mm"] < 0.4


def test_outline_size_mismatch_glints_and_what_is_not_an_outline():
    rg = S.rig()
    c, n, R = POSES["g2"][0], S.normal_from_tilt(*POSES["g2"][1]), 9.75
    box = S.crop_box(rg, c, R + 12)
    # the bench incident: every lateral size 0.84 x - a 19.5 mm face reads 16.4 mm
    o, _, _ = _outline(rg, c, n, 0.84 * R, box, nominal=R, shift=(1.2, 0.0), seed=1)
    assert o["ok"] and o["diameter_mm"] == pytest.approx(0.84 * 19.5, abs=0.08)
    sc = outline_mod.outline_size(o, 19.5)
    assert sc["status"] == "mismatch" and sc["reason"] == "size_mismatch"
    assert sc["deviation_pct"] == pytest.approx(-16.0, abs=0.6)
    assert sc["message"].startswith("part reads 16.") and "% small: calibration or lens suspect" \
        in sc["message"] and "run rig-check / recalibrate" in sc["message"]
    assert "nothing is rescaled" in sc["message"]
    assert outline_mod.pull_reason(sc)["reason"] == "size_mismatch"
    big = outline_mod.size_check(20.5, 19.5)
    assert big["status"] == "mismatch" and "5.1 % large" in big["message"]
    assert outline_mod.size_check(19.2, 19.5)["status"] == "ok"             # -1.5 %
    assert outline_mod.size_check(18.6, 19.5)["status"] == "mismatch"       # -4.6 %
    assert outline_mod.size_check(18.6, 19.5, tol_pct=6.0)["status"] == "ok"
    # an ellipse (a lens zoom also tilts the triangulated plane): the axis furthest off counts
    ell = outline_mod.size_check(19.4, 19.5, axes=(21.4, 17.4))
    assert ell["status"] == "mismatch" and ell["deviation_pct"] == pytest.approx(-10.8, abs=0.1)
    # nothing to compare is 'not checked' (said), never 'ok'
    assert outline_mod.size_check(None, 19.5, why="no flood")["status"] == "not_checked"
    assert outline_mod.size_check(19.5, None)["status"] == "not_checked"
    assert outline_mod.size_check(16.0, 19.5, tol_pct=0)["status"] == "not_checked"
    # a saturated glint AT the rim of a dim face: no notch, no bulge (the clipped rays are
    # not measured; a glint does not raise the face level the edge is taken against) - and
    # the rim under it is not judged: never a finding, never passed either
    o, img, to_px = _outline(rg, c, n, R, box, seed=2, level_dn=60.0, glint=(7.5, 2.0, 1.2))
    assert (img >= 254).sum() > 20 and o["round_ok"] and o["saturated_pct"] > 0
    j = outline_mod.judge(o)
    assert j["status"] == "incomplete" and not j["findings"] and not j["marginal"]
    assert len(j["unjudged"]) == 1 and "saturated flood" in j["unjudged"][0]["why"]
    assert j["unjudged"][0]["arc_mm"] >= 1.5 and abs(j["unjudged"][0]["theta_deg"] - 15) < 12
    assert "no edge" in j["why"]
    assert o["diameter_mm"] == pytest.approx(19.5, abs=0.08)
    assert outline_mod.judge(o, min_arc_mm=6.0)["status"] == "ok"   # under the arc that counts
    # the same glint further in (not within 3 px of the edge) costs no ray at all
    o, img, _ = _outline(rg, c, n, R, box, seed=2, level_dn=60.0, glint=(5.0, 1.5, 1.2))
    assert (img >= 254).sum() > 20 and o["saturated_pct"] == 0 and o["valid_pct"] == 100
    assert outline_mod.judge(o)["status"] == "ok"
    # a lit surface behind the rim is not a bulge: pixels measured behind the plane are dark
    lit = img.copy()
    yy, xx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    p0 = to_px(np.zeros(1), np.zeros(1))[0]
    crescent = (np.hypot(xx - p0[0], yy - p0[1]) > 30.5) & (np.hypot(xx - p0[0], yy - p0[1]) < 40) \
        & (xx > p0[0] + 10)
    lit[crescent] = 150.0
    bad = outline_mod.measure(lit.astype(np.float32), to_px, R)
    assert outline_mod.judge(bad)["status"] != "ok" or not bad["round_ok"]
    ok = outline_mod.measure(lit.astype(np.float32), to_px, R, exclude=crescent)
    assert ok["round_ok"] and outline_mod.judge(ok)["status"] == "ok"
    # no bright disc on a dark surround: not measured (never a size, never a finding)
    flat = np.full(img.shape, 90.0, np.float32) + np.random.default_rng(0).normal(0, 1, img.shape)
    o = outline_mod.measure(flat.astype(np.float32), to_px, R)
    assert not o["ok"] and "no bright disc" in o["why"]
    assert outline_mod.judge(o)["status"] == "not_measured"
    assert outline_mod.outline_size(o, 19.5)["status"] == "not_checked"
    assert not o["seen"]
    # half the face in the flood (cut by the lit zone): there IS an outline, but it cannot be
    # judged all the way round - 'incomplete' (never passed), not 'no outline'
    cut = img.copy()
    cut[:, :int(p0[0]) - 12] = 8.0
    o = outline_mod.measure(cut.astype(np.float32), to_px, R)
    assert o["seen"] and not o["round_ok"] and outline_mod.judge(o)["status"] == "incomplete"
    # ... also when only a little more than half of it is there (the fit that most of the
    # rim lies on breaks down under 60 %; an outline is still seen)
    for keep_px in (6, 2):
        cut = img.copy()
        cut[:, :int(p0[0]) - keep_px] = 8.0
        o = outline_mod.measure(cut.astype(np.float32), to_px, R)
        assert o["seen"] and outline_mod.judge(o)["status"] == "incomplete", (keep_px, o["why"])
    # a textured flood that fills the image: edges everywhere, no outline seen
    tex = S.refl_texture(7)[:img.shape[0], :img.shape[1]] * 90.0
    o = outline_mod.measure(tex.astype(np.float32), to_px, R)
    assert not o["seen"] and not o["ok"] and outline_mod.judge(o)["status"] == "not_measured"
    with pytest.raises(ValueError, match="at least 180 rays"):
        outline_mod.measure(img.astype(np.float32), to_px, R, n_angles=90)


@pytest.mark.parametrize("depth", [0.6, -0.6, 1.0])
@pytest.mark.parametrize("arc", [10.0, 16.0, 20.0])
def test_a_wide_edge_dent_is_not_absorbed_by_the_fit(depth, arc):
    """A notch / bulge over 10-20 mm of a 19.5 mm face's rim (up to a third of it): a
    centre + ellipse fit trimmed at 3 MAD bends into it and reads half its depth ('ok' at
    0.6 mm, a size mismatch at 0.8 mm). The curve is the one most of the rim lies on."""
    rg = S.rig()
    c, n, R = POSES["g1"][0], S.normal_from_tilt(*POSES["g1"][1]), 9.75
    box = S.crop_box(rg, c, R + 12)
    o, _, _ = _outline(rg, c, n, R, box, seed=5,
                       rim_notches=[{"theta_deg": 100.0, "depth_mm": depth, "arc_mm": arc}])
    assert o["ok"] and o["round_ok"]
    assert o["diameter_mm"] == pytest.approx(19.5, abs=0.08)             # not shrunk / grown
    assert o["centre_offset_mm"] == pytest.approx([-0.4, 0.3], abs=0.06)  # not pulled
    assert o["ellipse"]["axis_ratio"] > 0.995
    assert outline_mod.outline_size(o, 19.5)["status"] == "ok"
    j = outline_mod.judge(o)
    assert j["status"] == "fail" and len(j["findings"]) == 1
    f = j["findings"][0]
    assert f["kind"] == ("notch" if depth > 0 else "bulge")
    assert abs(f["depth_mm"]) == pytest.approx(abs(depth), abs=0.08)
    assert f["arc_mm"] == pytest.approx(arc, rel=0.35) and abs(f["theta_deg"] - 100) < 0.3 * arc / R * 57.3
    assert j["arc_peak_mm"] == pytest.approx(abs(depth), abs=0.08)


def test_a_flat_on_the_rim_is_a_notch_not_a_size_mismatch():
    """The rim crushed flat (a chord, sagitta 1.5 / 2 mm): it read as a 5-7 % small part with
    two bulges beside a notch."""
    rg = S.rig()
    c, n, R = POSES["g1"][0], S.normal_from_tilt(*POSES["g1"][1]), 9.75
    box = S.crop_box(rg, c, R + 12)
    for sag in (1.5, 2.0):
        half = np.degrees(np.arccos((R - sag) / R))
        # a chord as a run of narrow notches: depth R - (R - sag) / cos(d)
        ds = np.arange(-half + 0.5, half, 1.0)
        notches = [{"theta_deg": 100.0 + d, "depth_mm": R - (R - sag) / np.cos(np.radians(d)),
                    "arc_mm": 1.25 * R * np.radians(1.0) / 0.6} for d in ds]
        o, _, _ = _outline(rg, c, n, R, box, seed=6, rim_notches=notches)
        j = outline_mod.judge(o)
        assert o["ok"] and outline_mod.outline_size(o, 19.5)["status"] == "ok"
        assert j["status"] == "fail" and [f["kind"] for f in j["findings"]] == ["notch"]
        assert abs(j["findings"][0]["depth_mm"]) > 0.8 * sag


@pytest.mark.parametrize("depth, arc, at_least", [(2.5, 4.0, False), (3.0, 4.0, True),
                                                  (4.0, 6.0, True), (6.0, 8.0, True),
                                                  (2.5, 12.0, False)])
def test_a_deep_chip_out_of_the_rim_is_a_finding(depth, arc, at_least):
    """A chip deeper than the edge search reaches left its middle rays without an edge:
    they counted as nothing wrong and split the notch into two flanks, each under the
    minimum arc - status 'ok'. A ray that is dark where the search starts has its edge
    there (a notch at least that deep), and rays without an edge never split an arc."""
    rg = S.rig()
    c, n, R = POSES["g1"][0], S.normal_from_tilt(*POSES["g1"][1]), 9.75
    box = S.crop_box(rg, c, R + 12)
    o, _, _ = _outline(rg, c, n, R, box, seed=1,
                       rim_notches=[{"theta_deg": 70.0, "depth_mm": depth, "arc_mm": arc}])
    assert o["seen"]
    j = outline_mod.judge(o)
    if not o["ok"]:                                 # too much of the rim off the curve to
        assert j["status"] == "incomplete"          # call it an outline: never passed
        return
    assert o["diameter_mm"] == pytest.approx(19.5, abs=0.1)
    assert j["status"] == "fail" and j["findings"][0]["kind"] == "notch"
    f = j["findings"][0]
    assert abs(f["theta_deg"] - 70) < 6 and f["arc_mm"] >= 0.6 * arc     # the middle of its floor
    assert abs(f["depth_mm"]) >= min(depth, 0.27 * R) - 0.35
    assert f["at_least"] is at_least and (o["inner_limit_pct"] > 0) is at_least


def test_a_narrow_deep_slot_is_never_passed():
    """3 mm deep, 2 mm of rim: inside the slot a ray's own level is the light that leaks in,
    and the edge was put back on the original rim (residual 0, 'ok'). Its level is never
    taken under half its neighbours'."""
    rg = S.rig()
    c, n, R = POSES["g1"][0], S.normal_from_tilt(*POSES["g1"][1]), 9.75
    box = S.crop_box(rg, c, R + 12)
    for depth, arc in ((3.0, 2.0), (2.5, 2.0), (4.0, 2.0), (6.0, 2.0)):
        o, _, _ = _outline(rg, c, n, R, box, seed=1,
                           rim_notches=[{"theta_deg": 70.0, "depth_mm": depth, "arc_mm": arc}])
        j = outline_mod.judge(o)
        assert o["ok"] and o["max_abs_mm"] > 1.5            # the slot IS in the outline
        assert j["status"] in ("fail", "marginal")          # 2 mm is at the minimum arc
        assert (j["findings"] or j["marginal"])[0]["kind"] == "notch"
    # 3 mm of rim: a definite finding
    o, _, _ = _outline(rg, c, n, R, box, seed=1,
                       rim_notches=[{"theta_deg": 70.0, "depth_mm": 3.0, "arc_mm": 3.0}])
    assert outline_mod.judge(o)["status"] == "fail"


def test_rays_without_an_edge_never_split_a_notch_and_are_never_passed():
    th = np.arange(360.0)

    def o(res, no_edge=None):
        res = np.asarray(res, float)
        ne = np.where(np.isfinite(res), outline_mod.NO_EDGE_OK, outline_mod.NO_EDGE_DARK) \
            if no_edge is None else np.asarray(no_edge)
        return {"seen": True, "ok": True, "round_ok": True, "residual_mm": res,
                "theta_deg": th, "radius_mm": 9.75, "diameter_mm": 19.5, "why": "",
                "no_edge": ne, "inner_limit": np.zeros(360, bool)}
    base = np.zeros(360)
    # the skeptic's chip: two flanks of 4 and 3 rays out of tolerance, 3 rays with no edge
    chip = base.copy()
    chip[60:70] = [-0.31, -0.62, -0.88, -1.14, np.nan, np.nan, np.nan, -1.32, -0.9, -0.5]
    j = outline_mod.judge(o(chip))
    assert j["status"] == "fail" and len(j["findings"]) == 1
    f = j["findings"][0]
    assert f["kind"] == "notch" and f["depth_mm"] == -1.32 and f["arc_mm"] == pytest.approx(
        9 * 2 * np.pi * 9.75 / 360, abs=0.01) and not j["unjudged"]
    # the same gap SATURATED: a glint is never part of a finding - and its rim is not judged
    sat = np.where(np.isfinite(chip), outline_mod.NO_EDGE_OK, outline_mod.NO_EDGE_SAT)
    j = outline_mod.judge(o(chip, sat))
    assert j["status"] == "incomplete" and not j["findings"] and len(j["unjudged"]) == 1
    assert "both sides of saturated flood" in j["unjudged"][0]["why"]
    # a stretch with no edge and nothing beside it out of tolerance: unjudged, never passed
    hole = base.copy()
    hole[200:212] = np.nan
    j = outline_mod.judge(o(hole))
    assert j["status"] == "incomplete" and not j["findings"]
    assert j["unjudged"] == [{"theta_deg": 206.0, "arc_mm": pytest.approx(2.04, abs=0.01),
                              "why": "no edge (dark, or no contrast)"}]
    hole[203:212] = 0.0                              # 3 rays: under the minimum arc
    assert outline_mod.judge(o(hole))["status"] == "ok"
    # an outline that is seen but not measured round is 'incomplete'; none at all: not measured
    assert outline_mod.judge({"seen": True, "ok": True, "round_ok": False,
                              "why": "an edge on 80 % of the rays"})["status"] == "incomplete"
    assert outline_mod.judge({"seen": False, "ok": False, "round_ok": False,
                              "why": "no bright disc"})["status"] == "not_measured"
    assert outline_mod.judge(None)["status"] == "not_measured"
    # a run right round the rim (no ray inside the tolerance) is one finding
    assert outline_mod.judge(o(base + 0.6))["findings"][0]["arc_mm"] == pytest.approx(61.26, abs=0.01)


def test_the_robust_fit_is_the_curve_most_of_the_rim_lies_on():
    th = np.radians(np.arange(360.0))
    rng = np.random.default_rng(3)
    true = 9.75 + 0.4 * np.cos(th) - 0.3 * np.sin(th) + 0.05 * np.cos(2 * th)
    for frac in (0.0, 0.15, 0.33, 0.38):
        rho = true + rng.normal(0, 0.02, 360)
        k = int(frac * 360)
        rho[40:40 + k] -= 0.7
        co, res, keep, sig = outline_mod._robust_harmonics(th, rho, 2)
        assert co == pytest.approx([9.75, 0.4, -0.3, 0.05, 0.0], abs=0.012), frac
        assert sig < 0.06 and np.abs(res[40:40 + k] + 0.7).max(initial=0) < 0.1
        if k:
            assert not keep[40:40 + k].any()
            plain = outline_mod._harmonics(th, rho, 2)[0]
            assert abs(plain[0] - 9.75) > 5 * abs(co[0] - 9.75)        # what it replaces
    # NaN rays are ignored; too few rays: no fit, no crash
    rho = true.copy()
    rho[100:130] = np.nan
    co, res, keep, _ = outline_mod._robust_harmonics(th, rho, 2)
    assert co[0] == pytest.approx(9.75, abs=1e-6) and np.isnan(res[100:130]).all()
    co, res, keep, sig = outline_mod._robust_harmonics(th, np.full(360, np.nan), 2)
    assert not keep.any() and np.isnan(sig)


def test_golden_outline_model_follows_what_repeats():
    """What the good placements' outlines share (lighting, camera angle) is not a defect:
    a check takes its residual against their per-angle median; the tolerance widens only
    where they differ, and never beyond 2 x."""
    th = np.radians(np.arange(360.0))
    base = 0.25 * np.exp(-0.5 * ((np.degrees(th) - 140) / 6.0) ** 2)     # a shared 'bulge'

    def o(seed, extra=0.0):
        r = np.random.default_rng(seed).normal(0, 0.02, 360) + base + extra
        return {"ok": True, "round_ok": True, "residual_mm": r, "theta_deg": np.degrees(th),
                "radius_mm": 9.75, "diameter_mm": 19.5, "rms_mm": float(r.std()),
                "max_abs_mm": float(np.abs(r).max()), "why": ""}
    m = outline_mod.golden_model([o(s) for s in range(5)])
    assert m["n_placements"] == 5 and m["median_mm"][140] == pytest.approx(0.25, abs=0.03)
    assert outline_mod.golden_model([o(1)]) is None and outline_mod.golden_model([]) is None
    notch = -0.6 * np.exp(-0.5 * ((np.degrees(th) - 300) / 8.0) ** 2)
    j = outline_mod.judge(o(9, notch), model=m)
    assert j["golden"] and j["status"] == "fail" and len(j["findings"]) == 1
    assert abs(j["findings"][0]["theta_deg"] - 300) < 6 and j["findings"][0]["kind"] == "notch"
    assert outline_mod.judge(o(9), model=m)["status"] == "ok"
    assert outline_mod.judge(o(9), model=m)["max_abs_mm"] < 0.1          # the shared part is gone
    # where the good placements differ a lot, the tolerance is wider - but at most 2 x
    wild = [dict(o(s), residual_mm=o(s)["residual_mm"] + (s - 2) * 0.3 * (np.abs(np.degrees(th) - 40) < 10))
            for s in range(5)]
    mw = outline_mod.golden_model(wild)
    jw = outline_mod.judge(o(9, 0.55 * (np.abs(np.degrees(th) - 40) < 10)), model=mw)
    assert jw["tol_max_mm"] == pytest.approx(0.8) and jw["status"] == "ok"
    assert outline_mod.judge(o(9, 0.55 * (np.abs(np.degrees(th) - 40) < 10)))["status"] == "fail"
    assert outline_mod.judge(o(9, 0.7 * (np.abs(np.degrees(th) - 40) < 10)),
                             model=mw)["status"] == "marginal"            # its guard band
    assert outline_mod.judge(o(9, 1.0 * (np.abs(np.degrees(th) - 40) < 10)), model=mw)["status"] == "fail"


# ============================================================ a 19.5 mm face through the verbs

D195 = 19.5
R195 = D195 / 2
CHAMFER = lambda R: (0.6 * R / 10.525, 0.15 * R / 10.525)       # noqa: E731  (scales with the face)
NOTCH = lambda d, th: [{"theta_deg": th, "depth_mm": d, "arc_mm": 4.0}]     # noqa: E731
DENT195 = {"depth_mm": 0.3, "size_mm": 3.0, "u_mm": 0.3 * R195, "v_mm": -0.4 * R195}
EXTRA = {"dent": dict(dents=[DENT195]), "edge8": dict(rim_notches=NOTCH(0.8, 30.0)),
         "edge4": dict(rim_notches=NOTCH(0.4, 200.0)), "edge2": dict(rim_notches=NOTCH(0.2, 30.0)),
         "bulge8": dict(rim_notches=NOTCH(-0.8, 120.0)), "small": {}}


def _use(ctx):
    api._session = ctx["s"]
    return ctx


@pytest.fixture(scope="module")
def f195(tmp_path_factory):
    """Placements of a 19.5 mm face with the targeted scan's flood pair (the bright flat on a
    dark surround): five good ones, a dented one, edge dents (the outline 0.8 / 0.4 / 0.2 mm
    in, 0.8 mm out) and one whose flood shows the face 0.84 x its size (a zoomed lens)."""
    lib = tmp_path_factory.mktemp("faces195") / "library"
    lib.mkdir()
    rg = S.rig()
    cal = S.write_calibration(lib, rg)
    s = api.configure(lib, mock=True)
    cases = dict(POSES)
    cases.update({k: ((-5.5, 4.0, 780.2), (12, 6)) for k in EXTRA})
    prof, fs, faces, plain = {}, {}, {}, {}
    for i, (name, (c, (tx, ty))) in enumerate(cases.items()):
        n = S.normal_from_tilt(tx, ty)
        kw = EXTRA.get(name, {})
        loc, reps = S.placement(lib, rg, cal, name, c, n, seed=i + 1, radius=R195,
                                margin_mm=R195 + 18, chamfer=CHAMFER(R195), **kw)
        box = S.crop_box(rg, c, R195 + 18)
        faces[name] = api.locate_face(loc, D195, f"{name}_face")
        flood = S.flood_image(rg, c, n, box, radius=R195 * (0.84 if name == "small" else 1.0),
                              rim_notches=kw.get("rim_notches", ()), seed=50 + i)
        fs[name] = S.write_facescan(lib, f"20260101T0300{i:02d}_000000_fs_{name}", faces[name].id,
                                    reps, box=box, flood=flood)
        prof[name] = V.face_profile(s, fs[name], label=name, stl=False)
        if name in ("g1", "g2", "edge8"):               # the same placements without a flood
            plain[name] = V.face_profile(s, reps, face=faces[name], label=f"{name}_plain",
                                         stl=False)
    golden = api.face_golden([prof[k] for k in ("g1", "g2", "g3", "g4")], "good4",
                             synthetic={"positions": 1, "depths_mm": [0.2], "widths_mm": [2.0]})
    yield {"lib": lib, "s": s, "cal": cal, "prof": prof, "fs": fs, "faces": faces,
           "plain": plain, "golden": golden}
    api._session = None


def test_locate_and_profile_record_the_size_and_the_outline(f195):
    _use(f195)
    t = f195["faces"]["g1"]                     # located on a scan without a flood pair
    sc = t.meta["size_check"]
    assert sc["status"] == "not_checked" and "no flood pair" in sc["message"]
    assert "disposition" not in t.meta and not t.warnings
    pr = f195["prof"]["g1"]
    st = pr.stats
    assert st["r_flat_mm"] == pytest.approx(0.855 * R195, abs=0.01)          # not a fixed 9.0
    assert st["size_check"]["status"] == "ok" and abs(st["size_check"]["deviation_pct"]) < 1.5
    ro = st["rim_outline"]
    assert ro["measured"] and ro["roundness_measured"] and ro["valid_pct"] == 100
    assert ro["diameter_mm"] == pytest.approx(D195, abs=0.25) and ro["rms_mm"] < 0.05
    assert ro["max_abs_mm"] < 0.15 and ro["n_angles"] >= 180
    with np.load(pr.files["npz"]) as z:
        assert {"rim_theta_deg", "rim_rho_mm", "rim_residual_mm", "px_behind", "crop"} <= set(z.files)
        assert z["rim_residual_mm"].shape == (360,)
    again = V.load_profile(f195["s"], pr.files["npz"])                       # outline from the .npz
    assert again._pixels.origin is not None and again._pixels.outline is None
    o = V._profile_outline(again, f195["cal"])
    assert o["round_ok"] and o["diameter_mm"] == pytest.approx(ro["diameter_mm"], abs=0.02)
    # the mis-scaled placement says so in its profile already
    small = f195["prof"]["small"]
    assert small.stats["size_check"]["status"] == "mismatch"
    assert any("part reads 16." in w and "small" in w for w in small.warnings)
    doc = json.loads(Path(small.files["json"]).read_text())
    assert doc["stats"]["size_check"]["reason"] == "size_mismatch"
    # an explicit flatness radius is kept; one beyond the face is refused
    assert V.face_profile(f195["s"], f195["fs"]["g2"], label="g2_r7", r_flat_mm=7.0,
                          stl=False).stats["r_flat_mm"] == 7.0
    with pytest.raises(ValueError, match="not on the 19.5 mm face"):
        V.face_profile(f195["s"], f195["fs"]["g2"], label="g2_bad", r_flat_mm=12.0, stl=False)


def test_golden_stores_the_scaled_radii_and_the_rim_model(f195):
    _use(f195)
    g = f195["golden"]
    rr = g.params["radii"]
    assert rr["face_radius_mm"] == R195 and rr["given"] == []
    assert rr["inspect_r_mm"] == pytest.approx(0.926 * R195, abs=0.01)       # 9.03, not 9.75
    assert rr["rim_r_mm"] == pytest.approx(0.855 * R195, abs=0.01)           # 8.34, not 9.0
    mf, band, fir = g.params["mf"], g.params["band"], g.params["fir"]
    assert mf["inspect_r_mm"] == pytest.approx(rr["inspect_r_mm"], abs=1e-3)
    assert mf["fit_r_mm"] == mf["inspect_r_mm"] and mf["plane_r_mm"] == mf["rim_r_mm"]
    assert band["rim_r_mm"] == mf["rim_r_mm"]                # ONE rim radius for every detector
    assert max(mf["noise_bands_mm"]) < R195 and fir["r_max_mm"] < mf["inspect_r_mm"]
    assert fir["edge_px"] == 2 and fir["used"]
    assert fir["fit"]["explained_pct"] < 5        # the flood's outline is not reflectance
    assert rr["notes"] == []                    # nothing given: nothing to say (the API's
                                                # rim_r_mm default is None, not 9.75)
    # good placements of this face: no false trigger, by any detector
    assert all(r["n_face"] + r["n_rim"] == 0 for r in g.leave_one_out), g.leave_one_out
    assert all(r[d]["n_face"] + r[d]["n_rim"] == 0 for r in g.leave_one_out
               for d in ("mf_fir", "mf", "band"))
    assert all(r["max_z"] < 5.5 for r in g.leave_one_out)
    sens = g.sensitivity
    assert sens["r_mm"] == pytest.approx(rr["rim_r_mm"], abs=1e-3) and sens["detected"] == sens["total"]
    ro = g.meta["rim_outline"]
    assert ro["n_placements"] == 4 and ro["tol_mm"] == 0.4 and ro["min_arc_mm"] == 1.5
    assert ro["diameter_mm"] == pytest.approx(D195, abs=0.25) and max(ro["max_abs_mm"]) < 0.15
    assert [r["status"] for r in ro["leave_one_out"]] == ["ok"] * 4
    assert [z["status"] for z in g.meta["size_check"]] == ["ok"] * 4
    with np.load(g.path / "golden.npz") as z:
        assert z["rim_median_mm"].shape == z["rim_spread_mm"].shape == (360,)
    # explicit radii are kept (one rim radius for all), the rest still derive
    g2 = V.face_golden(f195["s"], [f195["prof"][k] for k in ("g1", "g2", "g3")], "given",
                       synthetic=False, radii={"inspect_r_mm": 8.8, "rim_r_mm": 8.0})
    r2 = g2.params["radii"]
    assert r2["given"] == ["inspect_r_mm", "rim_r_mm"] and r2["inspect_r_mm"] == 8.8
    assert g2.params["mf"]["rim_r_mm"] == g2.params["band"]["rim_r_mm"] == 8.0
    assert g2.params["mf"]["plane_r_mm"] == 8.0 and g2.params["mf"]["fit_r_mm"] == 8.8
    # the band's own rim_r_mm (the former --rim-r) now sets the same single radius
    g3 = V.face_golden(f195["s"], [f195["prof"][k] for k in ("g1", "g2", "g3")], "legacy",
                       {"rim_r_mm": 7.5}, synthetic=False)
    assert g3.params["mf"]["rim_r_mm"] == g3.params["band"]["rim_r_mm"] == 7.5
    for bad, msg in (({"rim_r_mm": 9.5}, "outside the inspected zone"),
                     ({"inspect_r_mm": 9.9}, "not on the 19.5 mm face"), ({"nope": 1}, "unknown radii")):
        with pytest.raises(ValueError, match=msg):
            V.face_golden(f195["s"], [f195["prof"][k] for k in ("g1", "g2")], "bad", radii=bad,
                          synthetic=False)
    with pytest.raises(ValueError, match="given twice"):
        V.face_golden(f195["s"], [f195["prof"][k] for k in ("g1", "g2")], "bad", synthetic=False,
                      radii={"rim_r_mm": 8.0}, mf_params={"rim_r_mm": 8.2})
    with pytest.raises(ValueError, match="unknown rim outline settings"):
        V.face_golden(f195["s"], [f195["prof"][k] for k in ("g1", "g2")], "bad", rim={"tol": 1})


def test_check_passes_the_good_face_and_reports_the_dent_depth(f195):
    _use(f195)
    g, prof = f195["golden"], f195["prof"]
    ok = V.check_face(f195["s"], prof["g5"], g, "ok195")
    assert ok.verdict == "PASS" and ok.n_triggers == 0 and ok.stats["max_z"] < 5.5
    assert ok.stats["disposition"] == "CONTINUE" and ok.stats["pull_reasons"] == []
    assert ok.stats["radii"]["inspect_r_mm"] == pytest.approx(0.926 * R195, abs=0.01)
    assert ok.params["radii"] == ok.stats["radii"] and ok.stats["inspect_r_mm"] < R195
    assert ok.stats["size_check"]["status"] == "ok"
    ro = ok.stats["rim_outline"]
    assert ro["status"] == "ok" and ro["golden"] and ro["max_abs_mm"] < 0.15 and not ro["findings"]
    bad = V.check_face(f195["s"], prof["dent"], g, "dent195")
    assert bad.verdict == "FAIL" and bad.stats["disposition"] == "REJECT"
    (t,) = [x for x in bad.triggers if x["zone"] == "face"]
    assert bad.n_triggers == 1, bad.triggers
    assert np.hypot(t["u_mm"] - DENT195["u_mm"], t["v_mm"] - DENT195["v_mm"]) < 0.6
    # THE depth is the centre average (a 0.3 mm cosine bowl 3 mm wide, as the 0.25 mm grid
    # and the camera px resolve it); the single deepest point reads deeper
    assert t["depth_mm"] == pytest.approx(t["depth_centre_mm"], abs=1e-3)
    assert 0.22 < t["depth_centre_mm"] < 0.30 and t["depth_min_mm"] > t["depth_centre_mm"]
    assert t["depth_min_note"] == "single point, over-reads"
    assert 1.2 < t["width_half_depth_mm"] < 2.0                  # a cosine bowl: 1.5 mm
    assert "camera px" in t["width_note"] and t["depth_fit_mm"] > 0.1 and t["z"] > 10
    rep = json.loads(Path(bad.files["json"]).read_text())
    assert rep["disposition"] == "REJECT" and rep["pull_reasons"] == []
    assert rep["units"]["depth_centre_mm"].startswith("mm") and "over-reads" in rep["units"]["depth_min_mm"]
    assert rep["stats"]["radii"]["rim_r_mm"] == pytest.approx(0.855 * R195, abs=0.01)
    # min_depth_mm applies to the centre depth: between it and the deepest point = no trigger
    between = 0.5 * (t["depth_centre_mm"] + t["depth_min_mm"])
    assert V.check_face(f195["s"], prof["dent"], g, "dent_min",
                        {"min_depth_mm": between}).verdict == "PASS"
    assert V.check_face(f195["s"], prof["dent"], g, "dent_min2",
                        {"min_depth_mm": t["depth_centre_mm"] - 0.02}).verdict == "FAIL"
    with pytest.raises(ValueError, match="unknown params"):
        V.check_face(f195["s"], prof["dent"], g, "x", {"inspect_r_mm": 9.0})


def test_edge_dents_are_rim_outline_triggers(f195):
    _use(f195)
    g, prof, s = f195["golden"], f195["prof"], f195["s"]
    r = V.check_face(s, prof["edge8"], g, "edge8")
    assert r.verdict == "FAIL" and r.stats["disposition"] == "REJECT"
    (t,) = [x for x in r.triggers if x["zone"] == "rim_outline"]
    assert t["kind"] == "notch" and t["depth_mm"] == pytest.approx(0.8, abs=0.08)   # + = in
    assert t["dev_mm"] < 0 and t["arc_mm"] >= 2.4 and abs(t["theta_deg"] - 30) < 8
    assert t["r_mm"] == pytest.approx(R195, abs=0.2) and r.stats["n_rim_outline"] == 1
    assert r.stats["rim_outline"]["status"] == "fail" and r.stats["rim_outline"]["golden"]
    for k in ("png", "json", "npz"):
        assert Path(r.files[k]).is_file()
    with np.load(r.files["npz"]) as z:
        assert z["rim_residual_mm"].min() < -0.6
    b = V.check_face(s, prof["bulge8"], g, "bulge8")
    (t,) = [x for x in b.triggers if x["zone"] == "rim_outline"]
    assert b.verdict == "FAIL" and t["kind"] == "bulge" and t["depth_mm"] < -0.7
    assert abs(t["theta_deg"] - 120) < 8
    # 0.2 mm: inside the tolerance, reported
    small = V.check_face(s, prof["edge2"], g, "edge2")
    assert small.verdict == "PASS" and small.stats["rim_outline"]["status"] == "ok"
    assert 0.12 < small.stats["rim_outline"]["max_abs_mm"] < 0.3
    assert V.check_face(s, prof["edge2"], g, "edge2_tight", {"rim_tol_mm": 0.12}).verdict == "FAIL"
    # 0.4 mm: AT the tolerance - never passed (a trigger, or its guard band: PULL)
    m = V.check_face(s, prof["edge4"], g, "edge4")
    assert m.verdict != "PASS" and m.stats["disposition"] in ("REJECT", "PULL")
    if m.verdict == "INCOMPLETE":
        assert m.stats["pull_reasons"][0]["reason"] == "guard_band"
        assert any("within 20 % of the tolerance" in w for w in m.warnings)
    assert V.check_face(s, prof["edge4"], g, "edge4_03", {"rim_tol_mm": 0.3}).verdict == "FAIL"
    assert V.check_face(s, prof["edge4"], g, "edge4_08", {"rim_tol_mm": 0.8}).verdict == "PASS"
    # the golden holds a rim outline: a placement that does not show one is never passed
    inc = V.check_face(s, f195["plain"]["g1"], g, "plain_vs_rim")
    assert inc.verdict == "INCOMPLETE" and inc.stats["disposition"] == "PULL"
    assert inc.stats["pull_reasons"][0]["reason"] == "incomplete"
    assert any("the golden checks it" in w for w in inc.warnings)
    assert inc.stats["rim_outline"]["status"] == "not_measured"
    # a golden without one (placements without a flood pair): warned; a placement that
    # shows its outline is still judged, against its own fit
    gp = V.face_golden(s, [f195["plain"]["g1"], f195["plain"]["g2"]], "plain", synthetic=False)
    assert gp.meta["rim_outline"] is None and any("no rim outline in the golden" in w for w in gp.warnings)
    assert any("size not checked" in w for w in gp.warnings)
    own = V.check_face(s, prof["edge8"], gp, "edge8_own")
    assert own.verdict == "FAIL" and not own.stats["rim_outline"]["golden"]
    assert [x["zone"] for x in own.triggers if x["zone"] == "rim_outline"] == ["rim_outline"]
    blind = V.check_face(s, f195["plain"]["edge8"], gp, "edge8_blind")
    assert any("rim outline not checked" in w for w in blind.warnings)
    assert any("size not checked" in w for w in blind.warnings)


def test_size_mismatch_is_a_pull_never_a_pass_never_a_reject(f195, capsys):
    _use(f195)
    g, prof, s = f195["golden"], f195["prof"], f195["s"]
    # a golden is never built on it
    with pytest.raises(ValueError, match="placement small: part reads 16"):
        api.face_golden([prof["g1"], prof["g2"], prof["small"]], "bad", synthetic=False)
    assert len(api.list_library("face_golden")) == len(
        [d for d in (f195["lib"] / "face_golden" / S.GEO).iterdir()])
    r = V.check_face(s, prof["small"], g, "small")
    assert r.verdict == "INCOMPLETE" and r.stats["disposition"] == "PULL"
    (why,) = [p for p in r.stats["pull_reasons"] if p["reason"] == "size_mismatch"]
    assert "part reads 16." in why["message"] and "% small: calibration or lens suspect - run " \
        "rig-check / recalibrate" in why["message"]
    assert r.stats["size_check"]["deviation_pct"] == pytest.approx(-16.0, abs=0.7)
    assert r.stats["size_check"]["nominal_mm"] == D195
    assert any(w.startswith("part reads 16.") for w in r.warnings)
    doc = json.loads(Path(r.files["json"]).read_text())
    assert doc["disposition"] == "PULL" and doc["pull_reasons"][0]["reason"] == "size_mismatch"
    # whatever it found: triggers on a mis-scaled measurement are not a definite FAIL
    if r.n_triggers:
        assert any("not a definite FAIL" in w for w in r.warnings)
    # nothing is rescaled: the radii are still the golden's
    assert r.stats["radii"] == V.check_face(s, prof["g5"], g, "g5_again").stats["radii"]
    # a wider tolerance is the operator's call (per check), recorded
    wide = V.check_face(s, prof["small"], g, "small_wide", {"size_tol_pct": 20.0})
    assert wide.stats["size_check"]["status"] == "ok" and wide.params["size_tol_pct"] == 20.0
    assert not [p for p in wide.stats["pull_reasons"] if p["reason"] == "size_mismatch"]
    # ... and now that a check has pulled it, also as a part judged PULL (engine/badparts.py)
    with pytest.raises(badparts.BadPartReference, match="judged PULL"):
        api.face_golden([prof["g1"], prof["g2"], prof["small"]], "bad", synthetic=False)
    # the command line: exit 5 = PULL, the reason in the JSON; a good face 0, an edge dent 1
    from ui import cli
    base = ["--mock", "--library", str(f195["lib"]), "--json"]
    code = cli.main(base + ["face-check", str(prof["small"].files["npz"]), "--golden", g.id,
                            "--label", "cli_small"])
    doc = json.loads(capsys.readouterr().out)
    assert code == 5 and doc["disposition"] == "PULL"
    assert doc["pull_reasons"][0]["reason"] == "size_mismatch"
    assert doc["check"]["verdict"] == "INCOMPLETE"
    code = cli.main(base + ["face-check", str(prof["g5"].files["npz"]), "--golden", g.id,
                            "--label", "cli_ok"])
    doc = json.loads(capsys.readouterr().out)
    assert code == 0 and doc["disposition"] == "CONTINUE" and doc["check"]["verdict"] == "PASS"
    code = cli.main(base[:-1] + ["face-check", str(prof["edge8"].files["npz"]), "--golden", g.id,
                                 "--label", "cli_edge"])
    out = capsys.readouterr().out
    assert code == 1 and "rim_outline" in out and "REJECT" in out and "size check" in out
    code = cli.main(base + ["face-check", str(prof["small"].files["npz"]), "--golden", g.id,
                            "--label", "cli_small_wide", "--size-tol", "20", "--rim-tol", "0.8"])
    doc = json.loads(capsys.readouterr().out)
    assert doc["check"]["stats"]["size_check"]["status"] == "ok"
    assert doc["check"]["params"]["rim"]["tol_mm"] == 0.8
    api._session = f195["s"]


def test_cli_face_golden_and_profile_take_the_radii(f195, capsys):
    """face-golden --inspect-r / --rim-r (one rim radius for every detector), --rim-tol;
    face-profile without --r-flat: 0.855 x the face radius."""
    from ui import cli
    base = ["--mock", "--library", str(f195["lib"]), "--json"]
    try:
        code = cli.main(base + ["face-golden", *[f195["fs"][k] for k in ("g1", "g2", "g3")],
                                "--label", "cli_gold", "--no-synthetic", "--inspect-r", "8.8",
                                "--rim-r", "8.0", "--rim-tol", "0.5", "--rim-arc", "2"])
        doc = json.loads(capsys.readouterr().out)
        assert code == 0, doc
        prm = doc["golden"]["params"]
        assert prm["radii"]["given"] == ["inspect_r_mm", "rim_r_mm"]
        assert prm["mf"]["inspect_r_mm"] == 8.8 and prm["mf"]["rim_r_mm"] == prm["band"]["rim_r_mm"] == 8.0
        assert prm["rim"]["tol_mm"] == 0.5 and prm["rim"]["min_arc_mm"] == 2.0
        code = cli.main(base + ["face-golden", *[f195["fs"][k] for k in ("g1", "g2", "small")],
                                "--label", "cli_bad", "--no-synthetic"])
        out = capsys.readouterr().out                       # refused: for its size, or - once
        assert code == 2 and ("part reads 16" in out        # a face check pulled it (the test
                              or "was judged PULL" in out)  # above) - as a part judged PULL
        code = cli.main(base + ["face-profile", f195["fs"]["g4"], "--label", "cli_prof", "--no-stl"])
        doc = json.loads(capsys.readouterr().out)
        assert code == 0 and doc["profile"]["stats"]["r_flat_mm"] == pytest.approx(0.855 * R195, abs=0.01)
        assert doc["profile"]["stats"]["size_check"]["status"] == "ok"
        code = cli.main(base[:-1] + ["face-profile", f195["fs"]["small"], "--label", "cli_small_prof",
                                     "--no-stl"])
        out = capsys.readouterr().out
        assert code == 0 and "SIZE MISMATCH" in out and "rim outline" in out
    finally:
        api._session = f195["s"]


@pytest.mark.parametrize("D", [16.0, 30.0])
def test_other_face_sizes_through_the_verbs(D, tmp_path):
    """A 16 mm and a 30 mm face, heights + flood pair, default settings: the golden's radii
    are theirs, good placements trigger nothing (the reflectance FIR included: the flood's
    own outline at the mask border is not reflectance), a dent is found where it is."""
    R = D / 2
    lib = tmp_path / "library"
    lib.mkdir()
    rg = S.rig()
    cal = S.write_calibration(lib, rg)
    s = api.configure(lib, mock=True)
    dent = {"depth_mm": 0.3, "size_mm": 3.0, "u_mm": 0.3 * R, "v_mm": -0.4 * R}
    cases = dict(POSES)
    cases["dent"] = ((-5.5, 4.0, 780.2), (12, 6))
    prof = {}
    for i, (name, (c, (tx, ty))) in enumerate(cases.items()):
        n = S.normal_from_tilt(tx, ty)
        loc, reps = S.placement(lib, rg, cal, name, c, n, seed=i + 1, radius=R,
                                margin_mm=R + 18, chamfer=CHAMFER(R),
                                dents=[dent] if name == "dent" else ())
        box = S.crop_box(rg, c, R + 18)
        face = api.locate_face(loc, D, f"{name}_face")
        assert np.linalg.norm(np.asarray(face.centre_mm) - c) < 0.15 and not face.warnings
        fs = S.write_facescan(lib, f"20260101T0300{i:02d}_000000_fs_{name}", face.id, reps, box=box,
                              flood=S.flood_image(rg, c, n, box, radius=R, seed=70 + i))
        prof[name] = V.face_profile(s, fs, label=name, stl=False)
        assert prof[name].stats["size_check"]["status"] == "ok"
        assert prof[name].stats["rim_outline"]["diameter_mm"] == pytest.approx(D, abs=0.3)
    g = api.face_golden([prof[k] for k in ("g1", "g2", "g3", "g4")], "good4", synthetic=False)
    rr = g.params["radii"]
    assert rr["inspect_r_mm"] == pytest.approx(0.926 * R, abs=0.01)
    assert rr["rim_r_mm"] == pytest.approx(0.855 * R, abs=0.01) and g.params["fir"]["used"]
    assert all(r[d]["n_face"] + r[d]["n_rim"] == 0 for r in g.leave_one_out
               for d in ("mf_fir", "mf")), g.leave_one_out
    assert all(r["max_z"] < 5.5 for r in g.leave_one_out)
    # the band prototype shares the ONE rim radius, but it still inspects the whole face,
    # out to its last cell (its own rim zone: x 2, at least 0.3 mm): nothing on the flat
    assert all(r["band"]["n_face"] == 0 for r in g.leave_one_out), g.leave_one_out
    assert g.params["band"]["rim_r_mm"] == pytest.approx(rr["rim_r_mm"], abs=1e-3)
    ok = V.check_face(s, prof["g5"], g, "ok")
    assert ok.verdict == "PASS" and ok.stats["rim_outline"]["status"] == "ok", ok.triggers
    bad = V.check_face(s, prof["dent"], g, "dent")
    assert bad.verdict == "FAIL" and bad.stats["disposition"] == "REJECT", bad.triggers
    top = max(bad.triggers, key=lambda t: t["z"] or 0)          # the dent itself, by far
    assert top["zone"] == "face" and top["z"] > 20
    assert np.hypot(top["u_mm"] - dent["u_mm"], top["v_mm"] - dent["v_mm"]) < 0.6
    assert 0.2 < top["depth_centre_mm"] < 0.31 and top["depth_min_mm"] > top["depth_centre_mm"]


# ============================================================ the mock bench: exposure, lens

FACE_C = np.array([-8.0, 6.0, 780.0])
FACE_N = S.normal_from_tilt(10.4, 7.0)


def _place(bench, normal=FACE_N, body_albedo=None, **kw):
    """The cylinder with its end face toward the camera; kw: EndFace's (albedo, specular,
    shininess, rim_notches). A dim bare-metal part is dim all over (body_albedo)."""
    bench.clear()
    bench.place(EndFace(center=tuple(FACE_C), normal=tuple(normal), radius=S.RADIUS,
                        chamfer_mm=0.6, chamfer_depth_mm=0.15, **kw))
    bench.place(MetricCylinder(center=tuple(FACE_C - normal * 15), radius=S.RADIUS, height=30,
                               axis=tuple(normal),
                               albedo=kw.get("albedo", 0.7) if body_albedo is None else body_albedo))


def _mirror_normal(bench) -> np.ndarray:
    """The face normal that mirrors the projector into the camera (the bisector)."""
    to_p = bench.rig_model().projector_center - FACE_C
    h = to_p / np.linalg.norm(to_p) - FACE_C / np.linalg.norm(FACE_C)
    return h / np.linalg.norm(h)


@pytest.fixture(scope="module")
def rig(tmp_path_factory):
    """The mock bench with its true stereo calibration, three presets at one geometry
    (16.667 ms upright / inverted, and a 2 ms one: not whole video frames) and a 21.05 mm
    face located by the live locate (its scan keeps the flood pair)."""
    tmp = tmp_path_factory.mktemp("facefix")
    pd = tmp / "presets"
    pd.mkdir()
    d = json.loads((FIXTURE_PRESETS / "cell_wall_2p7ft_vstripe_fullrange.json").read_text())
    d.update(name="face_fast")
    d["camera"].update(average=1, exposure_us=16667)
    d["projection"].update(invert=False, duty=0.6)
    (pd / "face_fast.json").write_text(json.dumps(d))
    (pd / "face_invert.json").write_text(json.dumps(
        dict(d, name="face_invert", projection=dict(d["projection"], invert=True))))
    (pd / "face_2ms.json").write_text(json.dumps(
        dict(d, name="face_2ms", camera=dict(d["camera"], exposure_us=2000))))
    s = api.configure(tmp / "library", mock=True, bench=MockBench(seed=5), presets_dir=pd)
    p = api.load_preset("face_fast")
    bench = api.mock_bench()
    bench.use_stereo()
    from engine.calibrate.stereo import StereoCalibration
    tr = bench.stereo_truth()
    dd = s.store.new_dir("calibrations", p.geometry_id, label="stereo")
    cal = StereoCalibration(id=f"{p.geometry_id}/{dd.name}", geometry_id=p.geometry_id,
                            K_cam=tr["K_cam"], dist_cam=tr["dist_cam"], K_proj=tr["K_proj"],
                            dist_proj=np.zeros(5), R=tr["R"], t=np.ravel(tr["t"]),
                            diamond_parity=tr["parity"], fingerprint=config.geometry_fingerprint(p),
                            path=dd, created="2026-01-01T00:00:00")
    s.store.write_json(dd / "stereo_calibration.json", cal.to_json())
    _place(bench)
    face = V.locate_face_live(s, p, None, S.DIAMETER, "live", steps=4)
    yield {"s": s, "p": p, "pd": pd, "face": face, "cal": cal, "bench": bench, "tmp": tmp}
    api._session = None


def _m(ctx):
    api._session = ctx["s"]
    ctx["s"].preset = ctx["p"]
    return ctx["p"]


def test_live_locate_keeps_the_flood_and_checks_the_size(rig):
    _m(rig)
    t = rig["face"]
    assert np.linalg.norm(np.asarray(t.centre_mm) - FACE_C) < 0.3
    ls = t.meta["locate_scan"]
    assert ls["exposure_us"] == 16667 and ls["exposure_source"] == "preset" and ls["auto_exposure_on"]
    ms = next(m for m in api.list_library("metric") if m.id == t.source_scan)
    assert (ms.path / "flood.npz").is_file() and ms.meta["locate_scan"] and ms.meta["method"] == "plane"
    sc = t.meta["size_check"]
    assert sc["status"] == "ok" and abs(sc["deviation_pct"]) < 1.5 and sc["source"] == "scan"
    ro = t.meta["rim_outline"]
    assert ro["measured"] and ro["diameter_mm"] == pytest.approx(S.DIAMETER, abs=0.3)
    assert ro["rms_mm"] < 0.15 and ro["max_abs_mm"] < 0.4       # the mock camera is not anti-aliased
    assert "disposition" not in t.meta and not [w for w in t.warnings if "part reads" in w]


def test_the_api_passes_the_live_face_options_on(rig, capsys):
    """engine.api: locate_scan / locate_face_live / face_exposure and the exposure, size and
    silhouette options of the face verbs (the CLI and the web UI call these, not the
    engine's verbs behind them)."""
    from ui import cli
    p = _m(rig)
    _place(rig["bench"])
    ms = api.locate_scan(None, p, "api_locate", steps=4)
    assert ms.meta["locate_scan"] and (ms.path / "flood.npz").is_file()
    assert ms.meta["exposure_us"] == 16667
    t = api.locate_face_live(None, p, S.DIAMETER, "api_live", steps=4)
    assert np.linalg.norm(np.asarray(t.centre_mm) - FACE_C) < 0.3
    assert t.meta["size_check"]["status"] == "ok" and t.meta["centre_check"]["status"] == "ok"
    assert t.meta["centre_check"]["offset_mm"] < t.meta["centre_check"]["tol_mm"]
    assert t.meta["locate_scan"]["auto_exposure_on"] and t.meta["silhouette_diameter_mm"] is None
    _place(rig["bench"], albedo=0.35)           # a seating half as bright: 33 ms shows it well
    try:
        forced = api.locate_face_live(None, p, S.DIAMETER, "api_live_33", steps=4,
                                      exposure_us=33333)
    finally:
        _place(rig["bench"])
    assert forced.meta["locate_scan"]["exposure_source"] == "forced"
    assert forced.meta["locate_scan"]["exposure_us"] == 33333
    wide = api.locate_face(ms, S.DIAMETER, "api_sized", size_tol_pct=0)
    assert wide.meta["size_check"]["status"] == "not_checked"
    assert any("known-size check is OFF" in w for w in wide.warnings)
    rep = api.face_exposure(t, p, label="api_exposure")
    assert rep["ok"] and config.frame_locked(rep["exposure_us"]) and Path(rep["file"]).is_file()
    rep = api.face_exposure(t, p, target_dn=120.0, exposure_range_us=[16667, 33334],
                            label="api_exposure_120")
    assert rep["target_dn"] == 120.0 and rep["range_us"] == [16667, 33333]
    fs = api.targeted_face_scan(t, p, "api_scan", repeats=1, steps=4, polar=False,
                                exposure_us=33333)
    assert fs.meta["exposure_us"] == 33333 and fs.meta["exposure_source"] == "forced"
    fs = api.targeted_face_scan(t, p, "api_scan_auto", repeats=1, steps=4, polar=False,
                                exposure_range_us=[16667, 50000])
    assert fs.meta["exposure_source"] == "auto"
    assert fs.meta["auto_exposure"]["range_us"] == [16667, 50000]
    pr = api.face_profile(fs, label="api_prof", stl=False, size_tol_pct=6.0)
    assert pr.stats["size_check"]["tol_pct"] == 6.0 and pr.stats["size_check"]["status"] == "ok"
    assert pr.stats["centre"]["source"] in ("flood outline", "located circle")
    assert pr.stats["centre"]["status"] in ("ok", "recentred")
    # the command line goes the same way: --outer-diameter reaches the locate
    try:
        code = cli.main(["--mock", "--library", str(rig["s"].store.root), "--json", "face-locate",
                         ms.id, "--diameter", str(S.DIAMETER), "--label", "cli_outer",
                         "--outer-diameter", str(S.DIAMETER)])
        doc = json.loads(capsys.readouterr().out)
        assert code == 0 and doc["face"]["meta"]["silhouette_source"] == "given"
        assert doc["face"]["meta"]["params"]["silhouette_diameter_mm"] == S.DIAMETER
        code = cli.main(["--mock", "--library", str(rig["s"].store.root), "--json", "face-locate",
                         ms.id, "--diameter", str(S.DIAMETER), "--outer-diameter", "12"])
        assert code == 2 and "must be the part's outer" in capsys.readouterr().out
    finally:
        api._session = rig["s"]
        _m(rig)


def test_auto_exposure_scans_a_dim_face_at_the_exposure_it_needs(rig):
    p = _m(rig)
    s, bench = rig["s"], rig["bench"]
    _place(bench, albedo=0.08)                               # a dim seating of the same part
    try:
        r = V.face_exposure(s, p, rig["face"], label="dim")
        assert r["ok"] and r["exposure_us"] == 100000 and config.frame_locked(r["exposure_us"])
        assert 150 < r["p95_dn"] < 200 and r["clipped_pct"] < 0.5 and r["lit_pattern"] == 255
        assert r["shots"][0]["exposure_us"] == 16667 and r["shots"][0]["p95_dn"] < 45
        assert not r["shots"][0]["ok"] and len(r["shots"]) <= 3 and Path(r["file"]).is_file()
        assert r["preset_exposure_us"] == 16667 and p.camera.exposure_us == 16667   # untouched
        # the targeted scan does it by default (the preset's exposure is whole video frames)
        fs = V.targeted_face_scan(s, p, rig["face"], "dim_auto", repeats=1, steps=4, polar=False)
        assert fs.meta["exposure_us"] == 100000 and fs.meta["exposure_source"] == "auto"
        assert fs.meta["preset_exposure_us"] == 16667 and fs.meta["auto_exposure"]["ok"]
        assert fs.repeats[0]["face_decoded_pct"] > 90 and not fs.warnings
        ms = next(m for m in api.list_library("metric") if m.id == fs.metric_scan_ids[0])
        assert ms.meta["exposure_us"] == 100000 and ms.meta["exposure_source"] == "auto"
        # ... which a fixed exposure cannot: the same face at the preset's 16.7 ms
        fx = V.targeted_face_scan(s, p, rig["face"], "dim_fixed", repeats=1, steps=4, polar=False,
                                  auto_exposure=False)
        assert fx.meta["exposure_us"] == 16667 and fx.meta["exposure_source"] == "preset"
        assert fx.meta["auto_exposure"] is None and fx.repeats[0]["face_decoded_pct"] < 50
        assert any("only" in w and "decoded" in w for w in fx.warnings)
        # a forced exposure is used as given (and recorded as forced)
        fo = V.targeted_face_scan(s, p, rig["face"], "dim_forced", repeats=1, steps=4, polar=False,
                                  exposure_us=83333)
        assert fo.meta["exposure_us"] == 83333 and fo.meta["exposure_source"] == "forced"
        assert fo.repeats[0]["face_decoded_pct"] > 90
        with pytest.raises(ValueError, match="not both"):
            V.targeted_face_scan(s, p, rig["face"], "x", repeats=1, auto_exposure=True,
                                 exposure_us=50000)
        # an inverted preset (the duty-based auto_intensity refuses it): the lit flood is
        # pattern 0, measured, and the exposure is found all the same
        inv = api.load_preset("face_invert")
        with pytest.raises(ValueError, match="invert"):
            api.auto_intensity(rig["face"], inv)
        ri = V.face_exposure(s, inv, rig["face"], label="dim_inv")
        assert ri["ok"] and ri["lit_pattern"] == 0 and ri["inverted"]
        assert ri["exposure_us"] > 33333 and 94 < ri["p95_dn"] < 238
        # a preset whose exposure is not whole video frames: auto exposure is not the
        # default there (said), unless it is asked for
        sub = api.load_preset("face_2ms")
        fz = V.targeted_face_scan(s, sub, rig["face"], "sub", repeats=1, steps=4, polar=False)
        assert fz.meta["exposure_source"] == "preset" and fz.meta["exposure_us"] == 2000
        assert any("auto exposure is off" in w for w in fz.warnings)
        # asked for, the search is in whole video frames all the same; the mock's brightness
        # is relative to the preset's exposure, so 16.7 ms is 8 x this preset's 2 ms: too
        # bright at the shortest whole frame - said, and the face is not scanned
        ra = V.face_exposure(s, sub, rig["face"], label="sub_exp")
        assert not ra["ok"] and ra["disposition"] == "PULL" and "too bright" in ra["why"]
        assert ra["candidates_us"][0] == 16667 and ra["preset_exposure_us"] == 2000
        assert all(config.frame_locked(x["exposure_us"]) for x in ra["shots"])
        with pytest.raises(X.ExposureNotFound, match="the face is NOT scanned"):
            V.targeted_face_scan(s, sub, rig["face"], "sub_auto", repeats=1, steps=4, polar=False,
                                 auto_exposure=True)
        # face-locate --zone on this dim seating: the locate scan is repeated at the exposure
        # the face needs (no face / a dark face at 16.7 ms), and the target says how
        api.load_preset("face_fast")
        t = V.locate_face_live(s, p, None, S.DIAMETER, "live_dim", steps=4)
        ls = t.meta["locate_scan"]
        assert ls["exposure_source"] == "auto" and ls["exposure_us"] > 50000 and ls["notes"]
        assert config.frame_locked(ls["exposure_us"]) and ls["preset_exposure_us"] == 16667
        # at the exposure the dim face needs the bright surround is saturated: the face is
        # located on all the scans of this seating together (per pixel the longest exposure
        # that did not saturate it) - its depth rim and its size are there again
        assert len(ls["scans"]) >= 2 and ls["scan"] == t.source_scan not in ls["scans"]
        mm = next(m for m in api.list_library("metric") if m.id == t.source_scan)
        assert mm.meta["merged"]["scans"] and len(mm.meta["merged"]["exposures_us"]) >= 2
        assert all(n > 0 for n in mm.meta["merged"]["points"]) and (mm.path / "hdr.npz").is_file()
        assert np.linalg.norm(np.asarray(t.centre_mm) - FACE_C) < 0.3
        assert t.meta["rim_source"] == "depth" and t.meta["size_check"]["status"] == "ok"
        assert not [w for w in t.warnings if "still" in w or "check_ring" in w]
        with pytest.raises(ValueError, match="not both"):
            V.locate_face_live(s, p, None, S.DIAMETER, "x", auto_exposure=True, exposure_us=33333)
    finally:
        api.load_preset("face_fast")
        _place(bench)


def test_a_face_that_glints_is_not_scanned(rig):
    p = _m(rig)
    s, bench = rig["s"], rig["bench"]
    n_scans = len(api.list_library("facescans"))
    try:
        # a small specular patch on a dim face: it clips before the face is bright enough
        _place(bench, normal=_mirror_normal(bench), albedo=0.25, specular=3.0, shininess=1e6)
        r = V.face_exposure(s, p, rig["face"], label="glint")
        assert not r["ok"] and r["exposure_us"] is None and r["disposition"] == "PULL"
        assert r["shots"][0]["clipped_pct"] >= 0.5 and "mirrors the projector" in r["why"]
        with pytest.raises(X.ExposureNotFound, match="the face is NOT scanned") as ei:
            V.targeted_face_scan(s, p, rig["face"], "glint", repeats=1, steps=4, polar=False)
        assert ei.value.report["shots"] and not ei.value.report["ok"]
        assert len(api.list_library("facescans")) == n_scans           # nothing stored
        # a clean but brighter patch: the exposure nearest the target clips, the shorter one
        # does not - clipping decides
        _place(bench, normal=_mirror_normal(bench), albedo=0.394, specular=0.6, shininess=9e5)
        r = V.face_exposure(s, p, rig["face"], label="glint_ok")
        assert r["ok"] and r["exposure_us"] == 16667, r["shots"]
        by = {x["exposure_us"]: x for x in r["shots"]}
        assert by[33333]["clipped_pct"] >= 0.5 and not by[33333]["ok"] and by[16667]["clipped_pct"] < 0.5
        # a face that mirrors the projector outright: saturated at the shortest exposure
        _place(bench, normal=_mirror_normal(bench), specular=3.0, shininess=200.0)
        r = V.face_exposure(s, p, rig["face"], label="mirror")
        assert not r["ok"] and "saturated at the shortest exposure" in r["why"]
        assert r["shots"][0]["clipped_pct"] > 90
    finally:
        _place(bench)


def test_cli_face_scan_exposure_options(rig, capsys, monkeypatch):
    """face-scan: auto exposure is the default, --exposure-us forces one, a face no exposure
    fits is exit 5 (PULL) with the search in the JSON."""
    _m(rig)
    from ui import cli
    s = rig["s"]
    base = ["--mock", "--library", str(s.store.root), "--preset", "face_fast", "--json"]

    def same_session(*a, **k):                  # the CLI on THIS bench (its own would be new)
        api._session = s
        return s
    monkeypatch.setattr(api, "configure", same_session)
    try:
        code = cli.main(base + ["face-scan", "cli_scan", "--face", rig["face"].id, "--repeats", "1",
                                "--steps", "4", "--no-polar", "--exposure-us", "33333"])
        doc = json.loads(capsys.readouterr().out)
        assert code == 0 and doc["facescan"]["meta"]["exposure_source"] == "forced"
        assert doc["facescan"]["meta"]["exposure_us"] == 33333
        _place(rig["bench"], normal=_mirror_normal(rig["bench"]), albedo=0.25, specular=3.0,
               shininess=1e6)
        code = cli.main(base + ["face-scan", "cli_glint", "--face", rig["face"].id, "--repeats",
                                "1", "--steps", "4", "--no-polar"])
        doc = json.loads(capsys.readouterr().out)
        assert code == 5 and doc["facescan"] is None and doc["disposition"] == "PULL"
        assert doc["auto_exposure"]["shots"] and "clipped" in doc["pull_reasons"][0]["message"]
    finally:
        _place(rig["bench"])
        api._session = s


def test_an_edge_dent_shows_in_the_targeted_scans_flood(rig):
    """The mock bench end to end: a notch in the rim (0.8 mm over 4 mm) -> the targeted scan's
    flood pair -> the profile's rim outline: found at its place, the size still right."""
    p = _m(rig)
    s, bench = rig["s"], rig["bench"]
    try:
        _place(bench, rim_notches=[{"theta_deg": 200.0, "depth_mm": 0.9, "arc_mm": 5.0}])
        fs = V.targeted_face_scan(s, p, rig["face"], "notched", repeats=1, steps=4, polar=False)
        pr = V.face_profile(s, fs, stl=False)
        assert pr.stats["size_check"]["status"] == "ok"
        o = pr._pixels.outline
        assert o["round_ok"]
        j = outline_mod.judge(o)
        assert j["status"] == "fail", (j["status"], j["max_abs_mm"])
        f = j["findings"][0]
        assert f["kind"] == "notch" and abs((f["theta_deg"] - 200 + 180) % 360 - 180) < 10
        assert f["depth_mm"] < -0.5 and f["arc_mm"] >= 1.5
    finally:
        _place(bench)


def test_a_lens_zoomed_after_the_calibration_is_a_size_mismatch(rig, capsys):
    """The bench incident (2026-10-05): the zoom ring turned after the calibration - every
    lateral size 0.84 x and nothing warned. Now: locate says so (PULL), the station's strict
    gates refuse the face, the command line exits 5."""
    p = _m(rig)
    s, bench = rig["s"], rig["bench"]
    from engine.measure.station import geometry as geo_mod
    from ui import cli
    try:
        _place(bench)
        bench.zoom_lens(0.84)
        ms = api.scan_fast(None, p, "zoomed", axis="xy")              # keeps its flood pair
        t = api.locate_face(ms, S.DIAMETER, "zoomed")
        sc = t.meta["size_check"]
        assert sc["status"] == "mismatch" and sc["reason"] == "size_mismatch"
        assert -22 < sc["deviation_pct"] < -10 and sc["nominal_mm"] == S.DIAMETER
        assert t.meta["disposition"] == "PULL"
        assert t.meta["pull_reasons"][0]["reason"] == "size_mismatch"
        assert any(w.startswith("part reads 1") and "small: calibration or lens suspect - run "
                   "rig-check / recalibrate" in w for w in t.warnings)
        assert t.face_diameter_mm == S.DIAMETER                       # nothing is rescaled
        # every later verb on this face says so too
        fs = V.targeted_face_scan(s, p, t, "zoomed_scan", repeats=1, steps=4, polar=False)
        assert any("part reads" in w for w in fs.warnings)
        assert fs.meta["face_size_check"]["status"] == "mismatch"
        pr = V.face_profile(s, fs, stl=False)
        assert pr.stats["size_check"]["status"] == "mismatch"
        assert any("part reads" in w for w in pr.warnings)
        # the station: its strict gates set the mis-sized disc aside (nothing is measured)
        with pytest.raises(geo_mod.StationError, match="size mismatch: part reads 1"):
            api.station_measure(ms, None, "zoomed_station", 1, S.DIAMETER, {"ct": False})
        m = api.station_measure(ms, None, "zoomed_station_loose", 1, S.DIAMETER,
                                {"ct": False, "strict": False})
        assert m.cells[0]["size_check"]["status"] == "mismatch"
        assert any("size mismatch: part reads" in w for w in m.warnings)
        # the command line on the stored scan: exit 5, the reason in the JSON
        code = cli.main(["--mock", "--library", str(s.store.root), "--json", "face-locate", ms.id,
                         "--diameter", str(S.DIAMETER), "--label", "cli_zoomed"])
        doc = json.loads(capsys.readouterr().out)
        assert code == 5 and doc["disposition"] == "PULL"
        assert doc["pull_reasons"][0]["reason"] == "size_mismatch"
        assert doc["face"]["meta"]["size_check"]["status"] == "mismatch"
        code = cli.main(["--mock", "--library", str(s.store.root), "face-locate", ms.id,
                         "--diameter", str(S.DIAMETER), "--label", "cli_zoomed_wide",
                         "--size-tol", "30"])
        out = capsys.readouterr().out
        assert code == 0 and "size check" in out and "SIZE MISMATCH" not in out
    finally:
        bench.zoom_lens(1 / 0.84)
        _place(bench)
        api._session = s


# ============================================================ the station: the known size

def _zoomed(rg, factor: float):
    """The same rig seen through a lens zoomed by `factor` (the calibration keeps rg's)."""
    from engine.capture.mock_stereo import PinholeCamera, StereoRig
    K = rg.camera.K.copy()
    K[0, 0] *= factor
    K[1, 1] *= factor
    return StereoRig(PinholeCamera(K, rg.camera.dist.copy(), tuple(rg.camera.size)),
                     rg.projector, rg.R, rg.t)


@pytest.fixture(scope="module")
def stn(tmp_path_factory):
    """Two synthetic station scans (two end faces + the ribbed tube): a good one, and the
    same scene seen through a lens zoomed 0.84 x after the calibration."""
    import station_synth as SS
    lib = tmp_path_factory.mktemp("station_size") / "library"
    lib.mkdir()
    rg = S.rig()
    cal = S.write_calibration(lib, rg)
    good, _ = SS.write_station(lib, rg, cal, "20260101T050001_000000_good", seed=1,
                               s_range=(-75.0, 95.0))
    zoom, _ = SS.write_station(lib, _zoomed(rg, 0.84), cal, "20260101T050002_000000_zoom", seed=2,
                               s_range=(-75.0, 95.0))
    s = api.configure(lib, mock=True)
    yield {"lib": lib, "s": s, "good": good, "zoom": zoom}
    api._session = None


def test_station_holds_every_end_face_to_its_known_size(stn, capsys, monkeypatch):
    from engine.measure.station import geometry as geo_mod
    from engine.measure.station import verbs as st_verbs
    from ui import cli
    _use(stn)
    assert "size_tol_pct" in st_verbs.STRICT_KEYS and geo_mod.StationParams().size_tol_pct == 4.0
    m = api.station_measure(stn["good"], label="good")
    assert m.params["size_tol_pct"] == 4.0
    for c in m.cells:
        assert c["size_check"]["status"] == "ok" and abs(c["size_check"]["deviation_pct"]) < 1.5
        assert c["size_check"]["nominal_mm"] == 21.05
    assert not any("size mismatch" in w for w in m.warnings)
    # the zoomed lens: the strict gates refuse the measurement - nothing is measured
    with pytest.raises(geo_mod.StationError,
                       match=r"size mismatch: part reads 1[56]\.\d % small") as ei:
        api.station_measure(stn["zoom"], label="zoom")
    assert "calibration or lens suspect - run rig-check / recalibrate" in str(ei.value)
    # without the gates the faces are measured, and flagged (never silently)
    loose = api.station_measure(stn["zoom"], label="zoom_loose", params={"strict": False})
    for c in loose.cells:
        sc = c["size_check"]
        assert sc["status"] == "mismatch" and sc["deviation_pct"] == pytest.approx(-16.0, abs=1.0)
    assert sum("size mismatch: part reads 1" in w for w in loose.warnings) == 2
    doc = json.loads(Path(loose.files["json"]).read_text())
    assert doc["cells"][0]["size_check"]["reason"] == "size_mismatch"
    # the tolerance is a station parameter: off, or wide enough for this lens (the operator's call)
    off = api.station_measure(stn["zoom"], label="zoom_off", params={"size_tol_pct": 0})
    assert off.cells[0]["size_check"]["status"] == "not_checked"
    # ... and a check that is off says so, as does a face whose size could not be read
    assert any("known-size check is OFF (size_tol_pct 0)" in w for w in off.warnings)
    assert not any("known-size check is OFF" in w or "size not checked" in w for w in m.warnings)
    with monkeypatch.context() as mp:
        mp.setattr(geo_mod, "face_size", lambda f, prm: outline_mod.size_check(
            None, prm.face_diameter_mm, prm.size_tol_pct, "end face's flood outline",
            why="its outline is ragged (rms 0.41 mm about its ellipse)"))
        blind = api.station_measure(stn["good"], label="good_unsized")
    assert sum("size not checked" in w and "ragged" in w for w in blind.warnings) == 2
    assert all(c["size_check"]["status"] == "not_checked" for c in blind.cells)
    assert api.station_measure(stn["zoom"], label="zoom_wide",
                               params={"size_tol_pct": 30}).cells[0]["size_check"]["status"] == "ok"
    for bad in (-1, 50, "4", True):
        with pytest.raises(ValueError, match="size_tol_pct"):
            geo_mod.StationParams.from_dict({"size_tol_pct": bad})
    # a golden is never built on a mis-sized measurement
    with pytest.raises(ValueError, match="not a GOOD measurement"):
        api.station_golden([loose], "bad")
    # the check: a golden made without the gates measures the zoomed scan - INCOMPLETE,
    # disposition PULL, reason size_mismatch (never PASS, and not a FAIL: the heights are
    # not trusted either)
    g_loose = api.station_golden([stn["good"]], "loose", params={"strict": False})
    r = api.station_check(stn["zoom"], g_loose, label="zoom_check")
    assert r.verdict == "INCOMPLETE"
    sizes = [it for it in r.items if it["kind"] == "face_size"]
    assert len(sizes) == 2 and all(it["status"] == "size_mismatch" for it in sizes)
    assert sizes[0]["nominal_mm"] == 21.05 and sizes[0]["value_mm"] == pytest.approx(17.7, abs=0.3)
    rep = json.loads(Path(r.files["json"]).read_text())
    assert rep["disposition"] == "PULL" and rep["pull_reasons"][0]["reason"] == "size_mismatch"
    ok = api.station_check(stn["good"], g_loose, label="good_check")
    assert ok.verdict == "PASS"
    assert [it["status"] for it in ok.items if it["kind"] == "face_size"] == ["info", "info"]
    assert json.loads(Path(ok.files["json"]).read_text())["disposition"] == "CONTINUE"
    # the command line: exit 5 with the reason, measured (no gates) or refused (gates)
    g_strict = api.station_golden([stn["good"]], "strict")
    base = ["--mock", "--library", str(stn["lib"]), "--json", "station-check", stn["zoom"]]
    try:
        for gold, measured in ((g_loose, True), (g_strict, False)):
            code = cli.main(base + ["--golden", gold.id, "--label", f"cli_{gold.label}"])
            doc = json.loads(capsys.readouterr().out)
            assert code == 5 and doc["disposition"] == "PULL", doc
            assert doc["pull_reasons"][0]["reason"] == "size_mismatch"
            assert "part reads 1" in doc["pull_reasons"][0]["message"]
            assert (doc["check"] is not None) == measured
        code = cli.main(["--mock", "--library", str(stn["lib"]), "--json", "station", stn["zoom"],
                         "--label", "cli_wide", "--size-tol", "30"])
        doc = json.loads(capsys.readouterr().out)
        assert code == 0 and doc["station"]["cells"][0]["size_check"]["status"] == "ok"
    finally:
        api._session = stn["s"]


# ============================================================ a recipe through a zoomed lens

def test_recipe_run_through_a_zoomed_lens_is_a_pull(tmp_path):
    """The whole mock capture path: the recipe's faces scanned through a lens zoomed 0.84 x
    after the calibration. The end-face defect check finds the outline the wrong size:
    INVALID with the size-mismatch reason, the run is a PULL (exit 5) - never CONTINUE."""
    import recipe_synth as RS
    s = api.configure(tmp_path / "library", mock=True)
    p = api.load_preset("bench_9in_metric_bright")
    RS.truth_calibration(s, p)                              # the calibration: the lens as it was
    s.bench.zoom_lens(0.84)                                 # ... then the zoom ring is turned
    out = api.recipe_run("cylinder_end_faces", project_feedback=False,
                         runs_dir=tmp_path / "runs", mock_scene="scene:endfaces")
    res = out["result"]
    assert res["result"] != "PASS" and res["disposition"] == "PULL" and out["exit_code"] == 5
    checks = [c for z in res["zones"] for c in z["checks"] if c["type"] == "defect"]
    sized = [c for c in checks if "size mismatch" in c["message"]]
    assert sized and all(c["verdict"] == "INVALID" for c in checks)
    assert all("part reads 1" in c["message"] and "calibration or lens suspect" in c["message"]
               and c["pull"] == "size_mismatch" for c in sized)
    assert sized[0]["details"]["size_check"]["status"] == "mismatch"
    assert any("part reads 1" in r["message"] and r["reason"] == "size_mismatch"
               for r in res["pull_reasons"])
    # a line that REJECTS every doubt it may reject still PULLs this part: a size mismatch
    # says the calibration or the lens is suspect, nothing about the part (with 'invalid' in
    # the list the same run used to be a REJECT, exit 1)
    from engine import disposition as D
    doc = json.loads((Path(api.__file__).resolve().parents[1] / "recipes"
                      / "cylinder_end_faces.json").read_text(encoding="utf-8"))
    doc["pull"] = {**(doc.get("pull") or {}), "invalid_reasons_to_reject": list(D.REJECTABLE)}
    out2 = api.recipe_run(doc, project_feedback=False, runs_dir=tmp_path / "runs_reject",
                          mock_scene="scene:endfaces")
    assert out2["result"]["disposition"] == "PULL" and out2["exit_code"] == 5
    assert {r["reason"] for r in out2["result"]["pull_reasons"]} >= {"size_mismatch"}
    # measure mode on the same scan (what = endface): the station's strict gates hold the
    # faces to the diameter given - not measurable (exit 5) with the size mismatch said,
    # never a number
    mo = api.measure_run("endface", p, scans=[res["part"]["scan_id"]],
                         runs_dir=tmp_path / "mruns", face_diameter_mm=21.0, n_faces=2)
    assert mo["exit_code"] == 5 and mo["result"]["status"] == "NOT_MEASURABLE"
    assert mo["result"]["values"] == [] and "size mismatch" in mo["result"]["message"]
    assert "calibration or lens suspect" in mo["result"]["message"]
    # its own reason (not 'incomplete': the placement is not the problem), and what to do
    assert mo["result"]["reason"] == "size_mismatch"
    assert "recalibrate" in mo["result"]["how_to_fix"] and "rig-check" in mo["result"]["how_to_fix"]


# ============================================================ the part as it is: a flat inside a rounded rim

RB = 10.525                 # the cylinder's outer radius: a 21.05 mm part, its flat is 19.5 mm
RIM_DN = 40.0               # the rounded rim in the flood: dim (the flat is 170, the dark 8)


def _rim_shapes(c, n, dents=()):
    """A 19.5 mm flat, a rim that recedes 0.7 mm from it to the 21.05 mm outer circle (and
    returns depth points), the body, the background."""
    c, n = np.asarray(c, float), np.asarray(n, float)
    flat = EndFace(center=tuple(c), normal=tuple(n), radius=R195, dents=list(dents))
    rim = EndFace(center=tuple(c - 0.05 * n), normal=tuple(n), radius=RB,
                  chamfer_mm=RB - R195, chamfer_depth_mm=0.7, dents=list(dents))
    body = MetricCylinder(center=tuple(c - n * 15.0), radius=RB, height=30.0, axis=tuple(n))
    return [flat, rim, body, Plane()]


def _rim_flood(rg, c, n, box, seed):
    a = S.flood_image(rg, c, n, box, radius=R195, seed=seed)
    b = S.flood_image(rg, c, n, box, radius=RB, level_dn=RIM_DN, seed=seed + 1)
    return np.maximum(a, b)


@pytest.fixture(scope="module")
def rimmed(tmp_path_factory):
    """Placements of the part the bench describes - a 21.05 mm cylinder whose flat end face
    is 19.5 mm across inside a rounded rim that DECODES (its depth silhouette is the
    cylinder's): four good ones for a golden, four more good ones, one with a dent. Located
    with the flat's diameter (19.5), three ways: the silhouette found from the rim points
    (default), given (21.05), and not looked for (the former behaviour)."""
    lib = tmp_path_factory.mktemp("rimmed") / "library"
    lib.mkdir()
    rg = S.rig()
    cal = S.write_calibration(lib, rg)
    s = api.configure(lib, mock=True)
    rng = np.random.default_rng(7)
    poses = dict(list(POSES.items())[:4])
    for k in range(4):
        poses[f"t{k}"] = ((-5.5 + rng.uniform(-1, 1), 4.0 + rng.uniform(-1, 1),
                           780.0 + rng.uniform(-1, 1)),
                          (float(rng.uniform(6, 16)), float(rng.uniform(3, 10))))
    poses["dent"] = ((-5.5, 4.0, 780.2), (12, 6))
    out = {"lib": lib, "s": s, "cal": cal, "poses": poses, "faces": {}, "prof": {},
           "plain": {}, "given": {}, "fs": {}}
    for i, (name, (c, (tx, ty))) in enumerate(poses.items()):
        n = S.normal_from_tilt(tx, ty)
        objs = _rim_shapes(c, n, [DENT195] if name == "dent" else ())
        box = S.crop_box(rg, c, 28.0)
        loc = S.write_metric(lib, rg, objs, box, f"20260101T0100{i:02d}_000000_{name}_locate",
                             i * 10, cal)
        reps = [S.write_metric(lib, rg, objs, box,
                               f"20260101T0200{i:02d}_{k:06d}_{name}_r{k + 1}", i * 10 + k + 1, cal)
                for k in range(2)]
        out["faces"][name] = api.locate_face(loc, D195, f"{name}_face")
        if name in ("g1", "g2", "t0"):
            out["plain"][name] = api.locate_face(loc, D195, f"{name}_plain",
                                                 params={"silhouette_auto": False})
            out["given"][name] = api.locate_face(loc, D195, f"{name}_given",
                                                 silhouette_diameter_mm=2 * RB)
        flood = _rim_flood(rg, c, n, box, 50 + i)
        out["fs"][name] = S.write_facescan(lib, f"20260101T0300{i:02d}_000000_fs_{name}",
                                           out["faces"][name].id, reps, box=box, flood=flood)
        out["prof"][name] = V.face_profile(s, out["fs"][name], label=name, stl=False)
        if name in out["plain"]:                    # the same scans about the slid centre
            fsp = S.write_facescan(lib, f"20260101T0400{i:02d}_000000_fsp_{name}",
                                   out["plain"][name].id, reps, box=box, flood=flood)
            out["prof"][name + "_plain"] = V.face_profile(s, fsp, label=name + "_plain",
                                                          stl=False)
    out["golden"] = api.face_golden([out["prof"][k] for k in ("g1", "g2", "g3", "g4")],
                                    "rimmed4", synthetic=False)
    yield out
    api._session = None


def _centre_err(t_or_p, c) -> float:
    return float(np.linalg.norm(np.asarray(t_or_p.centre_mm) - np.asarray(c)))


def test_locate_centres_a_flat_inside_a_rounded_rim_on_its_silhouette(rimmed):
    """Located with --diameter 19.5 as documented, a 19.5 mm circle was fitted to the 21.05
    mm depth silhouette: it slid until it touched one side, 0.75-0.83 mm off, with no
    warning. The rim circle is fitted at the silhouette's own size; the face stays 19.5."""
    _use(rimmed)
    for name, t in rimmed["faces"].items():
        c = rimmed["poses"][name][0]
        assert _centre_err(t, c) < 0.25, (name, _centre_err(t, c))
        assert t.face_diameter_mm == D195 and t.meta["radius_mm"] == R195
        assert t.meta["silhouette_source"] == "rim points"
        assert t.meta["silhouette_diameter_mm"] == pytest.approx(2 * RB, abs=0.35)
        assert t.rim_rms_mm < 0.3
        assert any("depth silhouette is 2" in w and "larger than the 19.5 mm face" in w
                   for w in t.warnings)
    # the camera mask and the aim ring are the FACE's (19.5 mm), not the silhouette's
    t = rimmed["faces"]["g1"]
    e = t.camera_ellipse
    assert 0.5 * (e["a"] + e["b"]) * t.mm_per_camera_px == pytest.approx(D195, rel=0.01)
    # given: the same centre, said to be given
    for name, t in rimmed["given"].items():
        assert _centre_err(t, rimmed["poses"][name][0]) < 0.25
        assert t.meta["silhouette_source"] == "given" and t.meta["silhouette_diameter_mm"] == 2 * RB
        assert not any("depth silhouette" in w for w in t.warnings)
    # not looked for (the former behaviour): the slid centre
    for name, t in rimmed["plain"].items():
        assert 0.5 < _centre_err(t, rimmed["poses"][name][0]) < 1.1
        assert t.meta["silhouette_diameter_mm"] is None and t.rim_rms_mm > 0.4
    with pytest.raises(ValueError, match="silhouette_diameter_mm 15.0 must be the part's outer"):
        api.locate_face(rimmed["faces"]["g1"].source_scan, D195, "bad", silhouette_diameter_mm=15.0)
    with pytest.raises(ValueError, match="must be the part's outer"):
        api.locate_face(rimmed["faces"]["g1"].source_scan, D195, "bad", silhouette_diameter_mm=40.0)


def test_a_plain_face_is_located_as_before():
    """A face whose depth silhouette IS the face (no rim beyond it): nothing is found, the
    fixed-radius fit stands; a silhouette smaller than the face is never taken."""
    rg = S.rig()
    c, n = POSES["g1"][0], S.normal_from_tilt(*POSES["g1"][1])
    box = S.crop_box(rg, c, 28.0)
    X, uv, ok, _ = S.render(rg, S.shapes(c, n, radius=R195, chamfer=CHAMFER(R195)), box)
    xyz = np.full((S.CAM[0], S.CAM[1], 3), np.nan)
    mask = np.zeros(S.CAM, bool)
    cx = np.full(S.CAM, np.nan)
    cy = np.full(S.CAM, np.nan)
    x0, y0, x1, y1 = box
    xyz[y0:y1, x0:x1], mask[y0:y1, x0:x1] = np.where(ok[..., None], X, np.nan), ok
    cx[y0:y1, x0:x1] = np.where(ok, uv[..., 0], np.nan)
    cy[y0:y1, x0:x1] = np.where(ok, uv[..., 1], np.nan)
    for D in (D195, 18.2, 21.0):
        try:
            r = locate_mod.locate(xyz, mask, cx, cy, mask, rg.camera.K, rg.camera.dist, D)
        except locate_mod.FaceNotFound:
            assert D != D195                        # the wrong diameter may simply misfit
            continue
        assert r["silhouette_source"] is None or D < D195
        if D == D195:
            assert r["silhouette_diameter_mm"] is None
            assert float(np.linalg.norm(r["centre_mm"] - np.asarray(c))) < 0.2


def test_the_profile_is_centred_on_the_flat_whatever_the_locate_said(rimmed):
    """face_profile centres the face frame on the flood outline of the bright flat: the
    centre the detectors' radii are measured from is the flat's even when the located
    circle slid (0.8 mm: the inspected zone then reached the rim, and good parts triggered)."""
    _use(rimmed)
    for name in ("g1", "g2", "t0"):
        c = rimmed["poses"][name][0]
        pr, slid = rimmed["prof"][name], rimmed["prof"][name + "_plain"]
        for q in (pr, slid):
            ce = q.stats["centre"]
            assert _centre_err(q, c) < 0.12, (name, q.label, _centre_err(q, c))
            assert ce["source"] == "flood outline" and ce["residual_offset_mm"] < 0.06
            assert q.stats["rim_outline"]["centre_offset_mm"] == pytest.approx([0, 0], abs=0.06)
            assert q.stats["size_check"]["status"] == "ok"
        assert pr.stats["centre"]["status"] in ("ok", "recentred")
        assert pr.stats["centre"]["located_offset_mm"] < 0.3
        ce = slid.stats["centre"]
        assert ce["status"] == "recentred" and 0.5 < ce["located_offset_mm"] < 1.1
        assert ce["tol_mm"] == pytest.approx(0.035 * R195, abs=1e-3)
        assert any("from the centre of the face's flood outline" in w and "centred on the outline"
                   in w and "--outer-diameter" in w for w in slid.warnings)
        assert not any("flood outline" in w and "tolerance" in w for w in pr.warnings)
        # the face target of the slid locate carried the same finding? (its scan has no flood
        # pair: not checked there, said)
        assert rimmed["plain"][name].meta["centre_check"]["status"] == "not_checked"
    # a profile made about the slid centre and NOT re-centred (as one stored before this
    # change) is never passed: its outline says where the flat is
    slid = rimmed["prof"]["t0_plain"]
    o = slid._pixels.outline
    fake = dict(o, centre_offset_mm=[0.7, -0.3])
    cc = outline_mod.centre_check(fake, R195, "profile centre")
    assert cc["status"] == "off" and cc["offset_mm"] == pytest.approx(0.762, abs=1e-3)
    assert "profile centre is 0.76 mm from the centre" in cc["message"]
    assert outline_mod.centre_check(None, R195)["status"] == "not_checked"
    assert outline_mod.centre_check(dict(o, round_ok=False), R195)["status"] == "not_checked"


def test_good_parts_with_a_rounded_rim_pass_and_a_dent_is_found(rimmed):
    _use(rimmed)
    g = rimmed["golden"]
    rr = g.params["radii"]
    assert rr["inspect_r_mm"] == pytest.approx(0.926 * R195, abs=0.01)
    assert all(r["n_face"] == 0 and r["n_rim"] == 0 for r in g.leave_one_out), g.leave_one_out
    assert max(r["max_z"] for r in g.leave_one_out) < 5.0
    assert g.meta["rim_outline"] is not None
    assert all(r["status"] == "ok" for r in g.meta["rim_outline"]["leave_one_out"])
    for k in ("t0", "t1", "t2", "t3"):
        r = api.check_face(rimmed["prof"][k], g, f"rim_{k}")
        assert (r.verdict, r.disposition, r.pull_reasons) == ("PASS", "CONTINUE", []), (k, r.warnings)
        assert r.stats["max_z"] < 5.5 and r.stats["centre_check"]["status"] == "ok"
        assert r.stats["rim_outline"]["status"] == "ok" and r.stats["size_check"]["status"] == "ok"
    # the same good part located on the slid centre: re-centred by its profile, it passes too
    r = api.check_face(rimmed["prof"]["t0_plain"], g, "rim_t0_plain")
    assert (r.verdict, r.disposition) == ("PASS", "CONTINUE"), r.warnings
    r = api.check_face(rimmed["prof"]["dent"], g, "rim_dent")
    assert r.verdict == "FAIL" and r.disposition == "REJECT" and r.pull_reasons == []
    (t,) = r.triggers
    assert t["zone"] == "face" and 0.2 < t["depth_mm"] < 0.32 and t["z"] > 10
    assert np.hypot(t["u_mm"] - DENT195["u_mm"], t["v_mm"] - DENT195["v_mm"]) < 0.6


def test_a_profile_off_its_outlines_centre_is_a_pull_never_a_reject(rimmed, monkeypatch):
    """A profile whose centre is not its flood outline's (made before profiles were centred
    on the outline, or a face that leaves the frame): the inspected zone reaches the rim on
    one side, so its triggers are not definite either - INCOMPLETE / PULL, and a golden
    refuses it."""
    _use(rimmed)
    real = V._profile_outline

    def off_centre(prof, cal):
        o = real(prof, cal)
        return None if o is None else dict(o, centre_offset_mm=[0.6, 0.4])
    monkeypatch.setattr(V, "_profile_outline", off_centre)
    r = api.check_face(rimmed["prof"]["t1"], rimmed["golden"], "rim_off")
    assert r.verdict == "INCOMPLETE" and r.disposition == "PULL"
    assert [p["reason"] for p in r.pull_reasons] == ["invalid"]
    assert "0.72 mm from the centre of the face's flood outline" in r.pull_reasons[0]["message"]
    assert r.stats["centre_check"]["status"] == "off"
    r = api.check_face(rimmed["prof"]["dent"], rimmed["golden"], "rim_off_dent")
    assert r.n_triggers >= 1 and r.verdict == "INCOMPLETE" and r.disposition == "PULL"
    assert any("not a definite FAIL" in w for w in r.warnings)
    with pytest.raises(ValueError, match="face_golden: placement g1: the profile centre is 0.72 mm"):
        api.face_golden([rimmed["prof"][k] for k in ("g1", "g2", "g3")], "off", synthetic=False)


def test_explicit_radii_are_used_as_given_or_refused(rimmed):
    """`--r-flat 9.0` was silently replaced by 8.337 on a 19.5 mm face, and a band
    rim_r_mm of 9.75 dropped with a note: the former fixed defaults counted as 'not given'."""
    _use(rimmed)
    s = rimmed["s"]
    pr = api.face_profile(rimmed["fs"]["g1"], label="g1_r9", r_flat_mm=9.0, stl=False)
    assert pr.stats["r_flat_mm"] == 9.0
    assert api.face_profile(rimmed["fs"]["g1"], label="g1_rd", stl=False).stats["r_flat_mm"] \
        == pytest.approx(radii_mod.RIM_FRAC * R195, abs=1e-6)        # 0.855 R = 8.337 mm
    good = [rimmed["prof"][k] for k in ("g1", "g2", "g3")]
    # rim 9.0 with the inspected zone given wide enough: used, for every detector
    g = api.face_golden(good, "r9", rim_r_mm=9.0, radii={"inspect_r_mm": 9.5}, synthetic=False)
    rr = g.params["radii"]
    assert rr["rim_r_mm"] == 9.0 and rr["inspect_r_mm"] == 9.5
    assert sorted(rr["given"]) == ["inspect_r_mm", "rim_r_mm"]
    assert g.params["band"]["rim_r_mm"] == 9.0 and g.params["mf"]["rim_r_mm"] == 9.0
    # 9.75 on this face is outside the inspected zone (9.03): refused, never dropped
    with pytest.raises(ValueError, match="rim_r_mm 9.75 mm lies outside the inspected zone"):
        api.face_golden(good, "r975", rim_r_mm=9.75, synthetic=False)
    with pytest.raises(ValueError, match="rim_r_mm given twice"):
        V.face_golden(s, good, "twice", {"rim_r_mm": 8.0}, False, radii={"rim_r_mm": 8.5})
    assert not hasattr(V, "LEGACY_R_FLAT_MM") and not hasattr(V, "LEGACY_BAND_RIM_R_MM")


def test_the_size_check_switched_off_says_so(rimmed):
    _use(rimmed)
    pr = api.face_profile(rimmed["fs"]["g2"], label="g2_nosize", stl=False, size_tol_pct=0)
    assert pr.stats["size_check"]["status"] == "not_checked"
    assert any("known-size check is OFF" in w for w in pr.warnings)
    r = api.check_face(rimmed["prof"]["t2"], rimmed["golden"], "nosize", size_tol_pct=0)
    assert r.verdict == "PASS" and any("known-size check is OFF (size tolerance 0)" in w
                                       for w in r.warnings)
    g0 = api.face_golden([rimmed["prof"][k] for k in ("g1", "g2", "g3")], "nosize",
                         synthetic=False, size_tol_pct=0)
    assert any("known-size check is OFF" in w and "stored in this golden" in w for w in g0.warnings)
    r = api.check_face(rimmed["prof"]["t2"], g0, "nosize_g")
    assert any("size tolerance 0, stored in the golden" in w for w in r.warnings)
    assert not any("known-size check is OFF" in w for w in
                   api.check_face(rimmed["prof"]["t2"], rimmed["golden"], "size_on").warnings)

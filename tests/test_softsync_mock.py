"""Flash soft sync (engine/capture/softsync.py, engine/reconstruct/flash_scan.py):
the DLPC350 plays the fast-scan pack's 16 one-bit patterns on its internal timer
(simulated: MockFlashLoop with defocus, clock error, loop gaps, GUI mistakes), a
free-running camera records it (MockFreeRunCamera: exposure, frame rate, random
phase, jitter, noise, dropped frames), the frames are sorted in software and
decoded. Checks: every clean frame is labelled right and every mixture rejected,
the sorted images and the decode match a direct (synchronous) capture, the
refusals are clean, the verbs / CLI / web wiring, and that nothing here can
talk to the projector. Mock only: no hardware anywhere."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from engine import api, config, rules
from engine.bases import onebit as ob
from engine.capture import mock_flashloop as mfl
from engine.capture import softsync as ss
from engine.capture.mock_bench import MockBench
from engine.reconstruct import flash_scan

from conftest import FIXTURE_PRESETS, ROOT

ROI_BOX = (600, 470, 840, 640)          # camera px on the flat mock bench, DMD ~ cols 360-550
ZONE_CENTRE = (450, 570)                # DMD col, row of the synthetic zone ellipse


def _zone_mask():
    import cv2
    zm = np.zeros(ob.DMD_SHAPE, np.uint8)
    cv2.ellipse(zm, (ZONE_CENTRE, (90, 150), 0), 1, -1)
    cv2.rectangle(zm, (380, 690), (540, 702), 1, -1)             # a thin band, like the tube
    return zm > 0


@pytest.fixture(scope="module")
def bench():
    return MockBench(seed=1)


@pytest.fixture(scope="module")
def lay():
    return ss.fastscan_layout()


def _record(bench, layout, *, gui=None, camera=None, zone=None, box=ROI_BOX, loops=None,
            seed=7, patterns=None):
    roi = ss.align_roi(box) if box is not None else None
    plan = ss.plan_timing(layout.n, roi, exposure_us=layout.camera_exposure_us,
                          pattern_exposure_us=layout.pattern_exposure_us,
                          pattern_period_us=layout.pattern_period_us, loops=loops,
                          guard_us=layout.guard_us)
    cam = mfl.mock_camera(bench, layout, zone_mask=zone, gui=gui, camera=camera, seed=seed,
                          patterns=patterns)
    rec = ss.FlashFreeRunSync(cam).record(plan["n_frames"], plan["roi"],
                                          plan["camera_exposure_us"], plan["fps"],
                                          meta={"layout": layout.to_json()})
    return rec, cam, plan


def _truth(cam, rec):
    """Per frame: {pattern: share of the exposure} from the simulated DLP loop."""
    t_us = rec.times_s()[0] * 1e6
    return [cam.loop.weights(t, t + rec.exposure_us) for t in t_us]


def _check_labels(res, cam, rec):
    truth = _truth(cam, rec)
    pure = np.array([max(w.values()) > 0.999 if w else False for w in truth])
    top = np.array([max(w, key=w.get) if w else -1 for w in truth])
    used = res.labels >= 0
    assert used.sum() >= 2 * len(res.images)
    assert np.all(pure[used]), "a clean frame saw two patterns (or dark)"
    assert np.array_equal(res.labels[used], top[used]), "a clean frame got the wrong pattern"
    assert not np.any(used & ~pure)                  # every mixture was rejected
    return pure, used


# ------------------------------------------------------------------ layout / pack

def test_fastscan_layout_is_the_onebit_pack(lay):
    import tools.make_flash_pack as mfp
    assert lay.names() == list(ob.SLOTS) and lay.n == 16 and lay.mode == ss.FASTSCAN_MODE
    assert [lay.names()[i] for i in lay.basis_slots()] == [ob.SLOTS[s] for s in ob.DECODE_SLOTS]
    assert lay.periods == [8.0, 80.0, 1280.0] and lay.steps == 4
    assert lay.prefilter_px == ob.PREFILTER_PX
    sync = {e["name"]: e for e in lay.entries if e["role"] == "marker"}
    assert sync["sync_white"]["level"] == 1.0 and sync["sync_code"]["orient"] == "y"
    fine = [e for e in lay.entries if e["name"].startswith("fine")]
    assert all(e["group_level"] == pytest.approx(ob.fine_profile().mean()) for e in fine)
    # the engine's defaults are the pack tool's
    assert ss.FASTSCAN_PERIOD_US == mfp.FASTSCAN_PERIOD_US
    assert ss.FASTSCAN_CAMERA_EXPOSURE_US == mfp.FASTSCAN_CAMERA_EXPOSURE_US
    assert ss.FASTSCAN_GUARD_US == mfp.FASTSCAN_GUARD_US
    assert ss.FASTSCAN_MODE == mfp.FASTSCAN_MODE
    back = ss.FlashLayout.from_json(json.loads(json.dumps(lay.to_json())))
    assert back.to_json() == lay.to_json()


@pytest.fixture(scope="module")
def pack(tmp_path_factory):
    """A real fast-scan pack (full + zone images) from tools/make_flash_pack.py."""
    import tools.make_flash_pack as mfp
    tmp = tmp_path_factory.mktemp("pack")
    t = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    ring = np.stack([ZONE_CENTRE[0] + 30 * np.cos(t), ZONE_CENTRE[1] + 60 * np.sin(t)], -1)
    np.save(tmp / "ring.npy", ring)
    return mfp.make_fastscan_pack(zone_rings=[tmp / "ring.npy"], out_root=tmp, label="ss")


def test_layout_from_the_pack_manifest(pack):
    lay_z, m, d = ss.load_pack_layout(pack, "zone")
    assert d == Path(pack) and lay_z.variant == "zone" and lay_z.names() == list(ob.SLOTS)
    assert lay_z.pattern_period_us == m["timing"]["pattern_period_us"]
    assert lay_z.pattern_exposure_us == m["timing"]["pattern_exposure_us"]
    assert lay_z.camera_exposure_us == m["timing"]["camera"]["exposure_us"]
    assert lay_z.guard_us == m["timing"]["camera"]["guard_us"]
    assert lay_z.prefilter_px == m["decode"]["prefilter"]["sigma_camera_px"]
    zm = ss.pack_zone_mask(d)
    assert zm.shape == ob.DMD_SHAPE and zm[ZONE_CENTRE[1], ZONE_CENTRE[0]] and zm.mean() < 0.05
    assert ss.load_pack_layout(pack, None)[0].variant is None
    with pytest.raises(ValueError, match="variants"):
        ss.load_pack_layout(pack, "diagonal")
    bad = json.loads(json.dumps(m))
    for e in bad["sequence"]:
        if e["slot"] == 5:
            e["name"] = e["name"].replace("fine_k1", "fine_kX")
    with pytest.raises(ValueError, match="fast-scan slots"):
        ss.layout_from_manifest(bad, "full")
    eight = json.loads(json.dumps(m))
    eight["sequence"][7]["bitdepth"] = 8
    with pytest.raises(ValueError, match="1-bit"):
        ss.layout_from_manifest(eight, "full")


def test_the_mock_plays_the_packs_own_bit_planes(pack, lay):
    """The BMP read back and unpacked is the pattern designer's image, slot for slot."""
    full = mfl.pack_patterns(pack, "full")
    assert len(full) == 16
    for p, q in zip(full, ob.fastscan_patterns()):
        assert np.array_equal(p, q["image"]), q["name"]
    zone = mfl.pack_patterns(pack, "zone")
    assert zone[0].min() == 255 and zone[2].mean() < 0.05 * 255     # sync full, white zone only


def test_generic_manifest_roles():
    seq = [{"seq": 0, "name": "a", "role": "sync_white", "bitdepth": 1},
           {"seq": 1, "name": "b", "role": "sync_code", "bitdepth": 1},
           {"seq": 2, "name": "w", "role": "white", "basis_index": 0, "bitdepth": 1},
           {"seq": 3, "name": "k", "role": "black", "basis_index": 1, "bitdepth": 1}]
    seq += [{"seq": 4 + k, "name": f"f{k}", "role": "fringe", "basis_index": 2 + k,
             "period_px": 16, "step": k, "bitdepth": 1} for k in range(4)]
    lay = ss.layout_from_manifest({"sequence": seq, "timing": {"pattern_period_us": 5000}})
    assert lay.pattern_period_us == 5000 and lay.periods == [16.0]
    assert lay.entries[0]["level"] == 1.0 and lay.entries[1]["orient"] == "y"
    with pytest.raises(ValueError, match="role"):
        ss.layout_from_manifest({"sequence": seq[:-1] + [dict(seq[-1], role="gray")]})


# ------------------------------------------------------------------ planning

def test_plan_for_the_pack_timing(lay):
    full = ss.plan_timing(16, None, exposure_us=3000, pattern_exposure_us=10000,
                          pattern_period_us=10000, loops=None, guard_us=250)
    assert full["loops"] == 3 and full["capture_ms"] <= 550 and full["fps"] <= 226
    assert full["clean_frames_per_pattern_per_loop"] >= 1.0
    assert full["fps_limit"] == "USB3 bandwidth"
    frac = full["frames_per_loop"] % 1
    assert min(frac, 1 - frac) * full["loops"] >= 0.5 - 1e-6       # loops sample spread phases
    roi = ss.align_roi(ROI_BOX)
    small = ss.plan_timing(16, roi, exposure_us=3000, pattern_exposure_us=10000,
                           pattern_period_us=10000, loops=None, guard_us=250)
    assert small["loops"] == 2 and small["capture_ms"] <= 360 and small["fps"] > 300
    assert small["clean_frames_per_pattern"] >= 4.0 and not small["dlp_timing_errors"]
    assert ss._dlp_1bit_errors(200, 200) and ss._dlp_1bit_errors(1000, 1100)   # DLPU010G rules
    assert not ss._dlp_1bit_errors(10000, 10000) and not ss._dlp_1bit_errors(1000, 1230)
    with pytest.raises(ValueError, match="no camera exposure"):
        ss.plan_timing(16, roi, pattern_period_us=200)
    with pytest.raises(ValueError, match="loops"):
        ss.plan_timing(16, roi, loops=1)


def test_roi_alignment_and_fps_estimate():
    x, y, w, h = ss.align_roi((101, 203, 333, 377))
    st = ss.ROI_STEP
    assert x % st["x"] == 0 and y % st["y"] == 0 and w % st["w"] == 0 and h % st["h"] == 0
    assert x <= 101 - 16 + st["x"] and x + w >= 333 + 16 and y + h >= 377 + 16
    corner = ss.align_roi((1400, 1050, 1440, 1080))
    assert corner[0] + corner[2] <= 1440 and corner[1] + corner[3] <= 1080
    f_full = ss.estimate_max_fps(1440, 1080)
    f_band = ss.estimate_max_fps(1440, 200)
    assert f_full["fps"] == pytest.approx(226, abs=1) and f_band["fps"] > 3 * f_full["fps"]
    assert ss.estimate_max_fps(400, 300, 3000)["limit"] == "exposure"


def test_expected_total_scan_times():
    rows = {(r["P_us"], tuple(r["roi"])): r for r in flash_scan.expected_times()}
    pack_full = rows[(10000, (1440, 1080))]
    pack_zone = rows[(10000, (480, 352))]
    assert pack_full["capture_ms"] <= 550 and pack_zone["capture_ms"] <= 360
    assert rows[(5000, (480, 352))]["capture_ms"] <= 300
    assert rows[(2500, (480, 352))]["capture_ms"] <= 200
    assert all(r["clean_per_pattern"] >= 4.0 for r in rows.values())
    assert pack_zone["estimate_s"]["total_s"] < pack_full["estimate_s"]["total_s"]


# ------------------------------------------------------------------ sorting

@pytest.mark.parametrize("seed, gui, camera", [
    (7, {}, {}),
    (11, {"clock_ppm": 300.0}, {"jitter_us": 40.0}),       # DLP clock off, timestamp jitter
    (23, {"defocus_px": 0.5}, {"read_noise_dn": 4.0}),     # sharp focus, noisy camera
    (31, {"gap_us": 600.0}, {}),                           # a dark gap at the loop end
    # the GUI reads G0..G7 from the red byte: slots 8..15 then 0..7, a rotated loop
    # (tools/make_flash_pack.plane_swap_play_order); the sync markers anchor it
    (37, {"order": list(range(8, 16)) + list(range(8))}, {}),
])
def test_sort_labels_every_frame_right(bench, lay, seed, gui, camera):
    rec, cam, _ = _record(bench, lay, gui=gui, camera=camera, seed=seed)
    res = ss.sort_recording(rec, lay)
    pure, used = _check_labels(res, cam, rec)
    r = res.report
    k = 1 + gui.get("clock_ppm", 0) * 1e-6
    assert r["loop_period_ms"] == pytest.approx(cam.loop.loop_us() / 1e3 / k, rel=2e-4)
    assert min(res.counts) >= 3 and res.confidence >= 0.7
    assert sum(r["status_counts"].values()) == len(rec.frames)
    sync = r["anchor"]["sync_slots"]
    assert sync["sync_code"] == (sync["sync_white"] + 1) % 16
    assert r["variant_detected"] == "full" and json.dumps(ss._plain(r))


def test_sorted_images_match_a_direct_capture(bench, lay):
    """Each sorted image is the noise-free image of its pattern held for the whole
    exposure (what a synchronous capture converges to), within the noise of the
    averaged clean frames: no bias from a mixed or mislabelled frame."""
    rec, cam, _ = _record(bench, lay, seed=5)
    res = ss.sort_recording(rec, lay)
    for i in range(16):
        clean = cam._ambient() + cam.level * cam._signal(i)
        sd = np.sqrt((cam.read_noise_dn ** 2 + cam.shot_dn * np.clip(clean, 0, None)
                      + 1 / 12) / res.counts[i])
        d = res.images[i] - clean
        assert abs(float(d.mean())) < 0.3, lay.names()[i]
        rms, want = float(np.sqrt(np.mean(d ** 2))), float(np.sqrt(np.mean(sd ** 2)))
        assert rms < 1.25 * want, lay.names()[i]
        dd = res.images[i] - cam.direct(i, 16)
        assert abs(float(dd.mean())) < 0.5, lay.names()[i]


def test_decode_matches_a_direct_capture_and_the_truth(bench, lay):
    """Same patterns, same number of frames averaged per pattern: the soft-sync decode
    is as good as a synchronous (hardware-triggered) capture of the 1-bit set."""
    rec, cam, _ = _record(bench, lay, seed=9)
    res = ss.sort_recording(rec, lay)
    fld = ss.decode_sorted(res, lay, "mock")
    direct = ss.SortResult([cam.direct(i, res.counts[i]) for i in range(16)], res.counts,
                           [1] * 16, res.labels, res.status, res.loop_period_s,
                           res.pattern_period_s, 1.0, {}, res.roi, res.full_shape)
    ref = ss.decode_sorted(direct, lay, "mock")
    both = fld.mask & ref.mask
    assert both.sum() >= 0.95 * ref.mask.sum() > 20000
    d = (fld.corr_x - ref.corr_x)[both]
    assert abs(float(np.median(d))) < 0.005 and float(np.percentile(np.abs(d), 95)) < 0.1
    truth = bench._scene()[3]                       # corr_x the mock bench renders with
    e = (fld.corr_x - truth)[both]
    e_ref = (ref.corr_x - truth)[both]
    assert abs(float(np.median(e))) < 0.02
    p95, p95_ref = (float(np.percentile(np.abs(x), 95)) for x in (e, e_ref))
    assert p95 < 1.2 * p95_ref + 0.005
    assert fld.meta["prefilter_px"] == ob.PREFILTER_PX and fld.shape == tuple(rec.full_shape)


def test_zone_variant_is_sorted_and_recognised(bench, lay):
    zone_lay = ss.fastscan_layout("full")           # the scan was asked for the full image
    rec, cam, _ = _record(bench, lay, zone=_zone_mask(), seed=13)
    res = ss.sort_recording(rec, zone_lay)
    _check_labels(res, cam, rec)
    assert res.report["variant_detected"] == "zone"
    assert any("zone" in w for w in res.report.get("warnings", []))
    fld = ss.decode_sorted(res, lay, "mock")
    assert 2000 < fld.mask.sum() < 0.5 * np.prod(rec.frames.shape[1:])


def test_exposure_shorter_than_the_period_is_measured(bench, lay):
    """The GUI got exposure 9 ms / period 10 ms (the pack says 10 / 10): 1 ms dark."""
    rec, cam, _ = _record(bench, lay, gui={"exposure_us": 9000.0}, seed=17)
    res = ss.sort_recording(rec, lay)
    _check_labels(res, cam, rec)
    assert 700 < res.report["dark_between_patterns_us"] < 1300
    assert any("dark" in w for w in res.report["warnings"])


def _swap(order, a, b):
    o = list(order)
    o[a], o[b] = o[b], o[a]
    return o


REFUSALS = {
    "not_running": ({"running": False}, {}, None, "not playing"),
    "stopped": ({"stop_after_us": 150e3}, {}, None, "stopped"),
    "missing": ({"order": [i for i in range(16) if i != 9]}, {}, None, "missing"),
    "extra": ({"order": list(range(16)) + [9]}, {}, 3, "extra"),
    "step_order": ({"order": _swap(range(16), 4, 8)}, {}, None, "order"),
    "group_order": ({"order": list(range(4)) + list(range(8, 12)) + list(range(4, 8))
                     + list(range(12, 16))}, {}, None, "another order"),
    "reversed": ({"order": list(range(8)) + [11, 10, 9, 8] + list(range(12, 16))}, {}, None,
                 "reverse"),
    "sync_swapped": ({"order": _swap(range(16), 0, 1)}, {}, None, "sync marker"),
    "other_period": ({"period_us": 9000.0}, {}, None, "period"),
    "dropped": ({}, {"drop_frames": tuple(range(20, 32))}, None, "dropped"),
}


@pytest.mark.parametrize("case", sorted(REFUSALS))
def test_refusals(bench, lay, case):
    gui, camera, loops, why = REFUSALS[case]
    rec, _, _ = _record(bench, lay, gui=gui, camera=camera, loops=loops, seed=3)
    with pytest.raises(ss.SequenceError, match=why) as ei:
        ss.sort_recording(rec, lay)
    assert json.dumps(ei.value.report)                      # what was measured, json-safe


def test_too_short_and_bad_times(bench, lay):
    rec, _, _ = _record(bench, lay, seed=4)
    n = len(rec.frames) // 3                                  # < 1 loop
    with pytest.raises(ss.SequenceError, match="loops"):
        ss.sort_frames(rec.frames[:n], lay, times_s=rec.times_s()[0][:n], cam_exposure_s=3e-3,
                       pattern_period_s=0.01, pattern_exposure_s=0.01)
    with pytest.raises(ValueError, match="fps"):
        ss.sort_frames(rec.frames, lay, times_s=np.zeros(len(rec.frames)), cam_exposure_s=3e-3)
    t = rec.times_s()[0]
    res = ss.sort_frames(rec.frames, lay, times_s=t + 123.0, cam_exposure_s=3e-3,
                         pattern_period_s=0.01, pattern_exposure_s=0.01, guard_s=250e-6)
    assert res.report["time_source"] == "timestamps"              # any clock offset is fine


def test_recording_round_trip_and_offline_sort(bench, lay, tmp_path):
    rec, _, _ = _record(bench, lay, seed=21)
    f = rec.save(tmp_path / "rec.npz")
    with pytest.raises(rules.OverwriteRefused):
        rec.save(f)
    back = ss.Recording.load(f)
    assert np.array_equal(back.frames, rec.frames) and back.roi == rec.roi
    assert np.array_equal(back.device_ts_ns, rec.device_ts_ns) and back.fps == rec.fps
    a = ss.sort_recording(rec, lay)
    b, lay2, _ = ss.sort_file(f)                          # the layout stored in the recording
    assert lay2.to_json() == lay.to_json()
    assert np.array_equal(a.labels, b.labels) and a.counts == b.counts


# ------------------------------------------------------------------ verbs

def _truth_calibration(s, p):
    from engine.calibrate.stereo import StereoCalibration
    tr = api.mock_bench().stereo_truth()
    d = s.store.new_dir("calibrations", p.geometry_id, label="stereo")
    cal = StereoCalibration(id=f"{p.geometry_id}/{d.name}", geometry_id=p.geometry_id,
                            K_cam=tr["K_cam"], dist_cam=tr["dist_cam"], K_proj=tr["K_proj"],
                            dist_proj=np.zeros(5), R=tr["R"], t=np.ravel(tr["t"]),
                            diamond_parity=tr["parity"], fingerprint=config.geometry_fingerprint(p),
                            path=d, created="2026-01-01T00:00:00")
    s.store.write_json(d / "stereo_calibration.json", cal.to_json())
    return cal


@pytest.fixture(scope="module")
def metric(tmp_path_factory, pack):
    """Mock stereo bench (square DMD grid, like the bench calibration) with a standing
    cell, the rig's truth as the stereo calibration, and one flash scan of it."""
    from engine.capture.mock_stereo import MetricCylinder, StereoRig
    tmp = tmp_path_factory.mktemp("flash")
    pd = tmp / "presets"
    pd.mkdir()
    d = json.loads((FIXTURE_PRESETS / "cell_wall_2p7ft_vstripe_fullrange.json").read_text())
    d["name"] = "flash_mock"
    (pd / "flash_mock.json").write_text(json.dumps(d))
    b = MockBench(seed=3)
    b.stereo_rig = StereoRig.bench(parity=None)
    s = api.configure(tmp / "library", mock=True, bench=b, presets_dir=pd)
    p = api.load_preset("flash_mock")
    api.mock_bench().use_stereo()
    api.mock_bench().place(MetricCylinder(center=(0, 0, 780), radius=24.05, height=60))
    cal = _truth_calibration(s, p)
    r = api.fast_scan_flash(str(pack), "full", "cell", p, roi=[500, 300, 940, 780])
    yield {"s": s, "p": p, "cal": cal, "r": r, "tmp": tmp}
    api._session = None


def _use(metric):
    api._session = metric["s"]
    try:
        api.mock_bench().flash_gui = None
    except RuntimeError:
        pass
    return metric["p"]


def test_fast_scan_flash_makes_a_metric_scan(metric):
    _use(metric)
    r = metric["r"]
    ms = r["scan"]
    assert r["mode"] == "flash_softsync" and r["calibration_id"] == metric["cal"].id
    assert ms.meta["mode"] == "flash_softsync" and ms.meta["units"] == "mm" and ms.meta["fast"]
    fl = ms.meta["flash"]
    assert fl["frame_stats"]["min_count"] >= 3 and fl["variant_detected"] == "full"
    assert fl["pack"] and fl["camera"]["roi"] == r["plan"]["roi"] and fl["recording"]
    assert r["plan"]["loops"] == 2 and r["plan"]["capture_ms"] <= 360
    assert r["confidence"] >= 0.7 and r["n_points"] > 50000
    assert set(r["times_s"]) >= {"record_s", "sort_s", "decode_s", "total_s"}
    assert Path(r["recording"]).is_file() and Path(r["recording"]).parent.name == "flashscans"
    ox, _, om = api.mock_bench().stereo_scene().oracle()
    fld = ms.field
    m = fld.mask & om
    e = (fld.corr_x - ox)[m]
    assert abs(float(np.median(e))) < 0.02 and float(np.percentile(np.abs(e), 95)) < 0.1
    assert [m_.id for m_ in api.list_library("metric")][-1] == ms.id


def test_dry_run_opens_nothing(metric, monkeypatch):
    p = _use(metric)
    monkeypatch.setattr(mfl, "mock_camera", lambda *a, **k: pytest.fail("camera opened"))
    before = sorted(x.name for x in (metric["s"].store.root / "output").rglob("*"))
    r = api.fast_scan_flash(None, None, "plan", p, roi=[500, 300, 940, 780], dry_run=True)
    assert r["dry_run"] and r["plan"]["loops"] == 2 and r["calibration"] == metric["cal"].id
    assert r["estimate_s"]["total_s"] < 3 and any("defaults" in w for w in r["warnings"])
    assert sorted(x.name for x in (metric["s"].store.root / "output").rglob("*")) == before
    with pytest.raises(ValueError, match="clean frames"):
        api.fast_scan_flash(None, None, "plan", p, exposure_us=9000, dry_run=True)


def test_scan_refused_when_the_loop_is_not_playing(metric):
    p = _use(metric)
    n_metric = len(api.list_library("metric"))
    api.mock_bench().flash_gui = {"running": False}
    try:
        with pytest.raises(ss.SequenceError, match="not playing") as ei:
            api.fast_scan_flash(None, None, "idle", p, roi=[500, 300, 940, 780])
    finally:
        api.mock_bench().flash_gui = None
    assert any("recording kept" in n for n in getattr(ei.value, "__notes__", []))
    assert len(api.list_library("metric")) == n_metric          # nothing decoded / stored


def test_scan_without_calibration_is_refused_before_the_camera(tmp_path, monkeypatch):
    s = api.configure(tmp_path / "lib", mock=True, bench=MockBench(seed=2),
                      presets_dir=FIXTURE_PRESETS)
    try:
        p = api.load_preset("cell_wall_2p7ft_vstripe")
        monkeypatch.setattr(mfl, "mock_camera", lambda *a, **k: pytest.fail("camera opened"))
        with pytest.raises(rules.GeometryMismatch, match="rule 6"):
            api.fast_scan_flash(label="nocal", preset=p)
        assert not (s.store.root / "output" / "flashscans").exists()
    finally:
        api._session = None


def test_sort_flash_recording_verb(metric):
    _use(metric)
    r = api.sort_flash_recording(str(metric["r"]["recording"]), decode=True, label="resort")
    assert r["sorted"].is_file() and r["sorted"].parent.name == "flashscans"
    with np.load(r["sorted"]) as z:
        assert z["images"].shape[0] == 16 and list(z["names"]) == list(ob.SLOTS)
        assert z["mask"].sum() == r["decoded_px"] > 50000
    assert r["confidence"] >= 0.7 and r["layout_source"] and r["counts"]["sync_code"] >= 3
    with pytest.raises(ValueError, match="no recording"):
        api.sort_flash_recording(str(metric["tmp"] / "nothing.npz"))


def test_the_flash_scan_never_opens_the_projector(metric, monkeypatch):
    """Camera only: no HDMI window, no DLPC350 USB. Rule 1 still guards the process."""
    p = _use(metric)
    from engine.capture import projector as proj_mod
    monkeypatch.setattr(proj_mod.Projector, "__init__",
                        lambda *a, **k: pytest.fail("projector opened"))
    r = api.fast_scan_flash(None, None, "noproj", p, roi=[560, 360, 880, 720], loops=2,
                            keep_recording=False)
    assert r["recording"] is None and r["n_points"] > 10000
    monkeypatch.setitem(sys.modules, "hid", object())
    with pytest.raises(rules.SafetyViolation):
        api.fast_scan_flash(None, None, "usb", p, roi=[560, 360, 880, 720])
    with pytest.raises(rules.SafetyViolation):
        ss.FlashFreeRunSync(mfl.MockFreeRunCamera(lambda x: x, None))


def test_no_usb_or_projector_code_in_the_softsync_path():
    import ast
    for f in (ROOT / "engine" / "capture" / "softsync.py",
              ROOT / "engine" / "capture" / "mock_flashloop.py",
              ROOT / "engine" / "reconstruct" / "flash_scan.py"):
        names = []
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names += [f"{node.module or ''}.{a.name}" for a in node.names]
        assert names, f.name
        for bad in ("dmd_control", "hid", "dlp_driver", "swiss_army_sensor", "projector",
                    "patternmode"):
            assert not [n for n in names if bad in n.lower().split(".")
                        or n.lower().endswith(bad)], (f.name, bad, names)
    rules.check_scan_path_usb_free()


# ------------------------------------------------------------------ CLI / web

def test_cli_scan_flash(tmp_path, capsys, metric):
    from ui import cli
    lib = tmp_path / "library"
    try:
        code = cli.main(["--mock", "--library", str(lib), "--json", "scan-flash", "--dry-run",
                         "--roi", "500,300,940,780"])
        doc = json.loads(capsys.readouterr().out)
        assert code == 0 and doc["dry_run"] and doc["plan"]["loops"] == 2, doc
        code = cli.main(["--mock", "--library", str(lib), "scan-flash", "t"])
        out = capsys.readouterr()
        assert code == 2 and "rule 6" in (out.out + out.err)            # no calibration here
        code = cli.main(["--mock", "--library", str(lib), "--json", "sort-flash",
                         str(metric["r"]["recording"])])
        doc = json.loads(capsys.readouterr().out)
        assert code == 0 and doc["confidence"] >= 0.7 and Path(doc["sorted"]).is_file()
    finally:
        api._session = None
        api._session = metric["s"]


def test_web_verbs(tmp_path, metric):
    import threading
    from ui.web import server as ui_server
    from test_ui_server import Client
    assert {"fast_scan_flash", "sort_flash_recording"} <= set(ui_server.VERBS)
    assert "sort_flash_recording" in ui_server.RIG_FREE_VERBS
    assert "fast_scan_flash" not in ui_server.RIG_FREE_VERBS           # it takes the camera
    presets = tmp_path / "presets"
    presets.mkdir()
    (presets / "cell_wall_2p7ft_vstripe.json").write_text(
        (FIXTURE_PRESETS / "cell_wall_2p7ft_vstripe.json").read_text())
    httpd, app = ui_server.make_server(mock=True, library=tmp_path / "library",
                                       presets_dir=presets, port=0)
    t = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
    t.start()
    try:
        c = Client(app.port)
        status, j = c.post("/api/presets/cell_wall_2p7ft_vstripe/load", {})
        assert status == 200 and j["status"] == "done", j
        j = c.done("fast_scan_flash", {"dry_run": True, "roi": [500, 300, 940, 780]})
        assert j["result"]["plan"]["loops"] == 2 and j["result"]["dry_run"]
        j = c.job("fast_scan_flash", {"label": "x"})                      # no calibration
        assert j["status"] == "error" and "rule 6" in j["error"]
        j = c.job("sort_flash_recording", {"recording": str(tmp_path / "nope.npz")})
        assert j["status"] == "error" and j["error_type"] == "ArgError"
        j = c.done("sort_flash_recording", {"recording": str(metric["r"]["recording"])})
        assert j["result"]["confidence"] >= 0.7
    finally:
        httpd.shutdown()
        httpd.server_close()
        api._session = metric["s"]

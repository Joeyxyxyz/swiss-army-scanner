"""Flash fast scan end to end, mock only: the pack side (tools/make_flash_pack.py
--fastscan, engine/bases/onebit.py) and the capture side (engine/capture/softsync.py,
engine/reconstruct/flash_scan.py) fit together.

1. Contract: for every variant the capture expects exactly the packed set - the pack's
   BMP read back and unpacked plane by plane equals the loop the sorter / decoder assume
   (same order, sync markers, dithering), and a manifest that differs (another bit order,
   pattern version, dither, timing, a changed zone mask) is refused before any camera.
2. Pack -> simulated DLP loop playing the pack's own bit-planes -> free-running camera ->
   sort -> decode -> metric scan (fast_scan_flash) of a bench-like scene: two cell end
   faces tilted 20 deg, a 0.6 mm dent in C1, C2 0.48 mm proud of C1. Compared with a
   direct (ideal, synchronous) capture of the same 1-bit planes through the same camera,
   with the HDMI 8-bit multifreq fringes through the same optics, and with the truth.

Nothing here touches hardware."""

import copy
import json
import shutil

import numpy as np
import pytest

import face_synth as S
from conftest import FIXTURE_PRESETS
from engine import api, config, rules
from engine.bases import onebit as ob
from engine.bases.multifreq import MultiFreqBasis
from engine.calibrate.stereo import StereoCalibration
from engine.capture import mock_flashloop as mfl
from engine.capture import softsync as ss
from engine.capture.mock_bench import MockBench
from engine.capture.mock_stereo import EndFace, MetricCylinder, StereoRig
from engine.reconstruct.triangulate import triangulate
from tools import make_flash_pack as mfp

N = S.normal_from_tilt(20.0, 0.0)                 # face normal, toward the camera
E1 = np.cross(N, [0.0, 1.0, 0.0])
E1 /= np.linalg.norm(E1)
E2 = np.cross(E1, N)
C1 = np.array([-13.3, 0.0, 780.0])
C2 = C1 + 26.6 * E1 + 0.48 * N                    # 0.48 mm proud of C1 (bench: +0.47..0.48)
DENT = {"depth_mm": 0.6, "size_mm": 3.0, "u_mm": 1.75, "v_mm": -0.75}   # FWHM 1.5 mm


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    """Mock stereo bench (square DMD grid, as the bench calibration), the two faces, the
    rig's truth as the stereo calibration, and a fast-scan pack whose zone is the two face
    rims on the DMD (face_dmd_ring.npy, as the bench pack was made)."""
    tmp = tmp_path_factory.mktemp("flash_e2e")
    pd = tmp / "presets"
    pd.mkdir()
    d = json.loads((FIXTURE_PRESETS / "cell_wall_2p7ft_vstripe_fullrange.json").read_text())
    d["name"] = "flash_e2e"
    (pd / "flash_e2e.json").write_text(json.dumps(d))
    b = MockBench(seed=3)
    b.stereo_rig = StereoRig.bench(parity=None)
    s = api.configure(tmp / "library", mock=True, bench=b, presets_dir=pd)
    p = api.load_preset("flash_e2e")
    bench = api.mock_bench()
    bench.use_stereo()
    for c, dents in ((C1, [DENT]), (C2, [])):
        bench.place(EndFace(center=tuple(c), normal=tuple(N), radius=S.RADIUS, dents=list(dents),
                            chamfer_mm=0.6, chamfer_depth_mm=0.15))
        bench.place(MetricCylinder(center=tuple(c - N * 15), radius=S.RADIUS, height=30,
                                   axis=tuple(N)))
    rig = bench.rig_model()
    tr = rig.truth()
    dc = s.store.new_dir("calibrations", p.geometry_id, label="stereo")
    cal = StereoCalibration(id=f"{p.geometry_id}/{dc.name}", geometry_id=p.geometry_id,
                            K_cam=tr["K_cam"], dist_cam=tr["dist_cam"], K_proj=tr["K_proj"],
                            dist_proj=np.zeros(5), R=tr["R"], t=np.ravel(tr["t"]),
                            diamond_parity=tr["parity"],
                            fingerprint=config.geometry_fingerprint(p), path=dc,
                            created="2026-01-01T00:00:00")
    s.store.write_json(dc / "stereo_calibration.json", cal.to_json())
    th = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    rings, cam = [], []
    for i, c in enumerate((C1, C2)):
        X = c + S.RADIUS * (np.cos(th)[:, None] * E1 + np.sin(th)[:, None] * E2)
        f = tmp / f"face_c{i + 1}_dmd_ring.npy"
        np.save(f, rig.projector.project(X @ rig.R.T + rig.t))
        rings.append(f)
        cam.append(rig.camera.project(X))
    cp = np.concatenate(cam)
    roi = [int(cp[:, 0].min()) - 30, int(cp[:, 1].min()) - 30,
           int(cp[:, 0].max()) + 30, int(cp[:, 1].max()) + 30]
    pack = mfp.make_fastscan_pack(zone_rings=rings, out_root=tmp / "packs", label="e2e",
                                  geometry_id=p.geometry_id)
    g = bench.stereo_scene().geometry()
    shapes = bench.stereo_scene().all_shapes()
    yield {"s": s, "p": p, "cal": cal, "bench": bench, "pack": pack, "roi": roi, "tmp": tmp,
           "X": g["X"], "shape": g["shape"],
           "faces": [shapes.index(x) for x in shapes if isinstance(x, EndFace)]}
    api._session = None


def _use(sc):
    api._session = sc["s"]
    sc["bench"].flash_gui = None
    return sc["p"]


def face_metrics(sc, xyz, mask) -> dict:
    """What the bench evaluation reads off a scan of the scene: per face a plane fit
    (C1 without its dent), its residual, its offset at the face centre along the true
    normal, its tilt error; the C1 dent depth (least-squares depth of the known bowl
    shape, and the deepest 1 px smoothed height, which the noise biases); C2 - C1 at the
    face centres along the mean fitted normal (as the bench does: a tilt error of the
    fit tips this over the 26.6 mm between the centres, 0.01 mm per 0.02 deg)."""
    import cv2
    xyz = np.where(mask[..., None], xyz, np.nan)
    out, q = {}, []
    for k, (fi, c) in enumerate(zip(sc["faces"], (C1, C2))):
        w = sc["X"] - c
        u, v = w @ E1, w @ E2
        on = (sc["shape"] == fi) & (np.hypot(u, v) <= 9.0) & np.isfinite(xyz[..., 2])
        rd = np.hypot(u - DENT["u_mm"], v - DENT["v_mm"])
        away = on & (rd > 2.5) if k == 0 else on
        P = xyz[away].astype(np.float64)
        cen = P.mean(0)
        n = np.linalg.svd(P - cen, full_matrices=False)[2][2]
        n = n if n @ N > 0 else -n
        qc = c - ((c - cen) @ n) * n
        h = np.where(on, (xyz - qc) @ n, np.nan)
        r = {"n_px": int(on.sum()), "plane_rms_mm": float(np.nanstd(h[away])),
             "offset_mm": float((qc - c) @ N),
             "tilt_err_deg": float(np.degrees(np.arccos(min(1.0, float(n @ N)))))}
        if k == 0:
            hh = np.nan_to_num(h).astype(np.float32)
            ww = np.isfinite(h).astype(np.float32)
            sm = (cv2.GaussianBlur(hh, (0, 0), 1.0)
                  / np.maximum(cv2.GaussianBlur(ww, (0, 0), 1.0), 1e-6))
            half = DENT["size_mm"] / 2                               # least-squares depth of
            sel = on & (rd < half)                                   # the known bowl shape
            bowl = 0.5 * (1 + np.cos(np.pi * rd[sel] / half))
            r["dent_mm"] = float(h[sel] @ bowl / (bowl @ bowl))
            r["dent_min_mm"] = float(sm[on & (rd < 1.0)].min())       # bench-like: deepest
        out[f"c{k + 1}"] = r
        q.append((qc, n))
    nm = q[0][1] + q[1][1]
    out["c2_c1_mm"] = float((q[1][0] - q[0][0]) @ (nm / np.linalg.norm(nm)))
    return out


def _direct_1bit(sc, variant, r):
    """The same pack planes held on the DMD for the whole exposure (an ideal, synchronous
    capture) through the same camera, averaged as often as the soft sync averaged each."""
    p = sc["p"]
    lay, _, _ = ss.load_pack_layout(sc["pack"], variant)
    cam = mfl.mock_camera(sc["bench"], lay, patterns=mfl.pack_patterns(sc["pack"], variant),
                          seed=99)
    pl = r["plan"]
    cam.configure(roi=tuple(pl["roi"]), exposure_us=pl["camera_exposure_us"], fps=pl["fps"])
    counts = [r["frame_stats"]["per_pattern"][nm] for nm in lay.names()]
    res = ss.SortResult([cam.direct(i, counts[i]) for i in range(lay.n)], counts, [1] * lay.n,
                        None, None, 0.16, 0.01, 1.0, {}, tuple(pl["roi"]),
                        tuple(sc["bench"].cam_shape))
    fld = ss.decode_sorted(res, lay, p.geometry_id, contrast_min=p.decode.contrast_min,
                           conf_min=p.decode.conf_min)
    return fld, triangulate(fld, sc["cal"], "plane")


def _direct_hdmi(sc, variant, r, n_avg):
    """The HDMI fast scan's 8-bit multifreq fringes (4 x 8 / 80 / 1280 + white / black,
    zone-masked like the HDMI zone scan) through the same optics and camera."""
    p = sc["p"]
    pats = MultiFreqBasis(steps=4, periods_px=(8, 80, 1280), axis="x").generate()
    if variant == "zone":
        _, m, d = ss.load_pack_layout(sc["pack"], variant)
        z = ss.pack_zone_mask(d, m)
        pats = [np.where(z, x, 0).astype(np.uint8) for x in pats]
    loop = mfl.MockFlashLoop(pats, 10000, 10000, seed=1)
    cam = mfl.MockFreeRunCamera(sc["bench"].render, loop, full_shape=tuple(sc["bench"].cam_shape),
                                seed=5)
    pl = r["plan"]
    cam.configure(roi=tuple(pl["roi"]), exposure_us=pl["camera_exposure_us"], fps=pl["fps"])
    lay = ss.synthetic_layout(4, (8, 80, 1280), marker=False)
    res = ss.SortResult([cam.direct(i, n_avg) for i in range(14)], [n_avg] * 14, [1] * 14, None,
                        None, 0.16, 0.01, 1.0, {}, tuple(pl["roi"]), tuple(sc["bench"].cam_shape))
    fld = ss.decode_sorted(res, lay, p.geometry_id, contrast_min=p.decode.contrast_min,
                           conf_min=p.decode.conf_min)
    return fld, triangulate(fld, sc["cal"], "plane")


# ------------------------------------------------------------------ 1. contract

@pytest.mark.parametrize("variant", ["full", "zone"])
def test_the_capture_expects_exactly_the_packed_set(scene, variant):
    lay, m, d = ss.load_pack_layout(scene["pack"], variant)
    assert ss.fastscan_manifest_problems(m, variant) == []
    assert lay.variant == variant and lay.names() == list(ob.SLOTS)
    # the pack's own flash image, read back from disk (hash-checked) and unpacked plane by
    # plane, is what the sorter / decoder assume the DLP plays - same order, sync
    # markers and dithering
    planes = mfl.pack_patterns(scene["pack"], variant)
    zm = ss.pack_zone_mask(d, m) if variant == "zone" else None
    assert (zm is not None) == (variant == "zone")
    want = ss.layout_patterns(lay, zm)
    assert len(planes) == len(want) == 16
    for k, (a, b) in enumerate(zip(planes, want)):
        assert np.array_equal(a, b), (variant, k, lay.names()[k])
    ents = sorted((e for e in m["sequence"] if e["variant"] == variant), key=lambda e: e["seq"])
    assert [e["planes"][0] for e in ents] == ([f"G{i}" for i in range(8)]
                                              + [f"R{i}" for i in range(8)])
    assert [lay.names()[i] for i in lay.basis_slots()] == m["decode"]["order"]
    assert planes[0].min() == 255
    assert np.array_equal(planes[1], ob.sync_code().astype(np.uint8) * 255)
    if variant == "zone":                      # decode slots zone-only, sync markers full field
        assert all(not p[~zm].any() for p in planes[2:])
    t = m["timing"]
    assert lay.pattern_period_us == t["pattern_period_us"] == lay.pattern_exposure_us
    assert lay.camera_exposure_us == t["camera"]["exposure_us"]
    assert lay.guard_us == t["camera"]["guard_us"]
    assert lay.prefilter_px == m["decode"]["prefilter"]["sigma_camera_px"]


def _tampered(m, what):
    m = copy.deepcopy(m)
    ents = [e for e in m["sequence"] if e["variant"] == "zone"]
    if what == "bit_order":                    # slot k on bit k + 1: planes shifted by one
        for e in ents:
            e["bit_position"] = (e["slot"] + 1) % 16
            e["planes"] = [mfp.plane_label(e["bit_position"])]
    elif what == "pattern_version":
        m["design"]["pattern_version"] = ob.PATTERN_VERSION + 1
    elif what == "dither":
        m["design"]["fine"]["levels_q4"] = [4, 3, 2, 1, 0, 1, 2, 3]
    elif what == "sync_bars":
        m["design"]["sync_code"]["bar_rows"] = 8
    elif what == "timing":
        m["timing"]["pattern_exposure_us"] = 9900         # 100 us short of the period
    elif what == "trigger":
        m["timing"]["trigger"] = "external"
    elif what == "decode_order":
        m["decode"]["order"] = list(reversed(m["decode"]["order"]))
    elif what == "image_index":
        ents[5]["image_index"] = 0
    return m


@pytest.mark.parametrize("what", ["bit_order", "pattern_version", "dither", "sync_bars",
                                  "timing", "trigger", "decode_order", "image_index"])
def test_a_pack_that_differs_is_refused_before_the_camera(scene, what, monkeypatch):
    _, m, _ = ss.load_pack_layout(scene["pack"], "zone")
    bad = _tampered(m, what)
    assert ss.fastscan_manifest_problems(bad, "zone")
    with pytest.raises(ValueError, match="rebuild it"):
        ss.layout_from_manifest(bad, "zone")
    d = scene["tmp"] / f"bad_{what}"
    shutil.copytree(scene["pack"], d)
    (d / "manifest.json").write_text(json.dumps(bad))
    p = _use(scene)
    monkeypatch.setattr(mfl, "mock_camera", lambda *a, **k: pytest.fail("camera opened"))
    with pytest.raises(ValueError, match="rebuild it"):
        api.fast_scan_flash(str(d), "zone", "bad", p, roi=scene["roi"])


def test_a_changed_zone_mask_is_refused(scene):
    d = scene["tmp"] / "bad_zone_png"
    shutil.copytree(scene["pack"], d)
    raw = bytearray((d / "zone_mask.png").read_bytes())
    raw[-20] ^= 0xFF
    (d / "zone_mask.png").write_bytes(bytes(raw))
    _, m, _ = ss.load_pack_layout(d, "zone")
    with pytest.raises(ValueError, match="sha256"):
        ss.pack_zone_mask(d, m)


def test_prefilter_coherence_flags_depth_edges_only():
    """onebit.prefilter_keep: a smooth fringe ramp keeps its phase coherent under the 3 px
    mid / coarse low-pass; a depth edge (the projector column jumps by 30) does not."""
    H, W = 40, 200
    cols = np.arange(W, dtype=np.float64) / 1.5 + np.where(np.arange(W) < 100, 0.0, 30.0)
    cols = np.repeat(cols[None, :], H, 0)
    frames = [np.full((H, W), 200.0, np.float32), np.full((H, W), 10.0, np.float32)]
    for T in ob.PERIODS:
        for k in range(ob.STEPS):
            frames.append((105 + 95 * np.cos(2 * np.pi * cols / T + 2 * np.pi * k / ob.STEPS))
                          .astype(np.float32))
    keep = ob.prefilter_keep(frames)
    assert keep[:, :92].all() and keep[:, 108:].all()
    assert not keep[:, 99:101].any()
    assert ob.phasor_coherence(frames[6:10], 3.0)[:, 20:80].min() > 0.98


# ------------------------------------------------------------------ 2. end to end

@pytest.fixture(scope="module")
def runs(scene):
    """One flash scan per variant (the pack played by the simulated DLP, free-running
    camera, sort, decode, metric scan) + the direct 1-bit and HDMI 8-bit references."""
    p = _use(scene)
    out = {}
    for variant in ("zone", "full"):
        r = api.fast_scan_flash(str(scene["pack"]), variant, f"e2e_{variant}", p,
                                roi=scene["roi"])
        fd, cd = _direct_1bit(scene, variant, r)
        n_avg = int(np.median(list(r["frame_stats"]["per_pattern"].values())))
        fh, ch = _direct_hdmi(scene, variant, r, n_avg)
        out[variant] = {"r": r, "direct": (fd, cd), "hdmi": (fh, ch)}
    return out


def _truth_metrics(scene):
    X = scene["X"]
    return face_metrics(scene, X, np.isfinite(X[..., 2]))


@pytest.mark.parametrize("variant", ["zone", "full"])
def test_flash_scan_matches_the_direct_capture(scene, runs, variant):
    """Soft sync loses nothing against an ideal synchronous capture of the same planes."""
    r = runs[variant]["r"]
    ms = r["scan"]
    fd, _ = runs[variant]["direct"]
    assert r["mode"] == "flash_softsync" and r["variant_detected"] == variant
    assert r["confidence"] >= 0.7 and r["frame_stats"]["min_count"] >= 3
    assert r["plan"]["loops"] == 2 and r["plan"]["capture_ms"] <= 360
    fs = r["frame_stats"]
    assert fs["status_counts"]["inconsistent"] == 0 and fs["frames_dropped"] == 0
    assert fs["signal_p99_dn"] > 50
    x0, y0, x1, y1 = scene["roi"]              # the scan keeps the camera window asked for
    win = np.zeros(ms.field.mask.shape, bool)
    win[y0:y1, x0:x1] = True
    dm = fd.mask & win
    both = ms.field.mask & dm
    # the two decodes cover the same px (the edge cuts move by a px or two with noise)
    assert both.sum() >= 0.95 * max(dm.sum(), ms.field.mask.sum()) > 5000
    d = (ms.field.corr_x - fd.corr_x)[both]
    assert abs(float(np.median(d))) < 0.005 and float(np.percentile(np.abs(d), 95)) < 0.1
    meta = ms.meta
    assert meta["mode"] == "flash_softsync" and meta["units"] == "mm"
    assert meta["flash"]["pack"] == str(scene["pack"]) and meta["flash"]["variant"] == variant


@pytest.mark.parametrize("variant", ["zone", "full"])
def test_no_wrong_fringe_orders_in_the_cloud(scene, runs, variant):
    """Every decoded px is within a fraction of a column of the truth: in the zone image the
    px on the zone's defocused edge (clipped fringes, errors up to whole fringe orders) are
    left undecoded, found with the full-field SYNC A white."""
    ms = runs[variant]["r"]["scan"]
    ox, _, om = scene["bench"].stereo_scene().oracle()
    m = ms.field.mask & om
    e = np.abs(ms.field.corr_x - ox)[m]
    assert float(np.percentile(e, 99)) < 0.25 and float(e.max()) < 1.0
    # at the face rims the 3 px mid / coarse prefilter mixes face and background: those
    # px (8-column errors without the check) are left undecoded; the HDMI 8-bit decode
    # has no prefilter and no such errors at depth edges - but its zone scan has no
    # full-field white to find the zone's edge with, and keeps a few wrong orders there
    pe = ms.meta["decode"]["prefilter_edge"]
    assert pe["masked_px"] > 10 and pe["min_coherence"] == ob.PREFILTER_COHERENCE_MIN
    fh = runs[variant]["hdmi"][0]
    eh = np.abs(fh.corr_x - ox)[fh.mask & om]
    if variant == "full":
        assert float(eh.max()) < 1.0
    else:
        assert float(eh.max()) > 4.0                 # the flash zone scan is the cleaner one
        assert float(np.percentile(eh, 99)) > float(np.percentile(e, 99))
    ze = ms.meta["decode"].get("zone_edge")
    if variant == "zone":
        assert ze["masked_px"] > 200 and ze["min_white_over_sync_white"] == ss.ZONE_EDGE_MIN
        assert runs[variant]["r"]["frame_stats"]["zone_edge"] == ze
    else:
        assert ze is None


@pytest.mark.parametrize("variant", ["zone", "full"])
def test_metric_results_match_direct_hdmi_and_truth(scene, runs, variant):
    """The bench's numbers - C2 - C1 height, C1 dent depth, face flatness - from the flash
    scan agree with the ideal 1-bit capture, the HDMI 8-bit fringes and the truth."""
    truth = _truth_metrics(scene)
    assert abs(truth["c2_c1_mm"] - 0.48) < 1e-6 and abs(truth["c1"]["dent_mm"] + 0.6) < 1e-3
    ms = runs[variant]["r"]["scan"]
    fl = face_metrics(scene, ms.cloud.xyz, ms.cloud.mask)
    dr = face_metrics(scene, runs[variant]["direct"][1].xyz, runs[variant]["direct"][1].mask)
    hd = face_metrics(scene, runs[variant]["hdmi"][1].xyz, runs[variant]["hdmi"][1].mask)
    for f in ("c1", "c2"):
        assert fl[f]["n_px"] >= 0.98 * truth[f]["n_px"], f
        assert fl[f]["plane_rms_mm"] < 1.2 * dr[f]["plane_rms_mm"] + 0.005, f
        assert fl[f]["plane_rms_mm"] < 0.06, f
        assert abs(fl[f]["offset_mm"]) < 0.006, f          # face height at its centre
    # C2 - C1 along the fitted normal: +-0.01 from the plane tilt noise, like the bench
    assert abs(fl["c2_c1_mm"] - truth["c2_c1_mm"]) < 0.015
    assert abs(fl["c2_c1_mm"] - dr["c2_c1_mm"]) < 0.02
    assert abs(fl["c2_c1_mm"] - hd["c2_c1_mm"]) < 0.02
    assert abs(fl["c1"]["dent_mm"] - truth["c1"]["dent_mm"]) < 0.02
    assert abs(fl["c1"]["dent_mm"] - dr["c1"]["dent_mm"]) < 0.02
    assert abs(fl["c1"]["dent_mm"] - hd["c1"]["dent_mm"]) < 0.02


def test_the_recording_resorts_to_the_same_images(scene, runs):
    """The raw frames kept for the bench re-sort offline to the scan's result."""
    _use(scene)
    r = runs["zone"]["r"]
    doc = api.sort_flash_recording(str(r["recording"]), decode=True, label="e2e_resort")
    assert doc["counts"] == r["frame_stats"]["per_pattern"]
    assert doc["variant_detected"] == "zone" and doc["zone_edge"]["masked_px"] > 200
    with np.load(doc["sorted"]) as z:
        assert int(z["mask"].sum()) == doc["decoded_px"] > 5000


def test_fine_prefilter_option(scene, runs):
    """The pack's optional fine low-pass (manifest decode.prefilter.fine_sigma_camera_px_
    recommended) reaches the decode and is recorded. It cuts the per-px scatter ~3x, but
    it is a lateral blur: face heights and C2 - C1 keep, a 1.5 mm FWHM dent reads
    shallower (~11-14 % here), so it stays off by default and is compared on the bench."""
    p = _use(scene)
    m = json.loads((scene["pack"] / "manifest.json").read_text(encoding="utf-8"))
    px = m["decode"]["prefilter"]["fine_sigma_camera_px_recommended"]
    r = api.fast_scan_flash(str(scene["pack"]), "zone", "e2e_fine", p, roi=scene["roi"],
                            fine_prefilter_px=px, keep_recording=False)
    ms = r["scan"]
    assert ms.meta["decode"]["fine_prefilter_px"] == px
    assert ms.meta["flash"]["fine_prefilter_px"] == px
    truth = _truth_metrics(scene)
    fl = face_metrics(scene, ms.cloud.xyz, ms.cloud.mask)
    off = face_metrics(scene, runs["zone"]["r"]["scan"].cloud.xyz,
                       runs["zone"]["r"]["scan"].cloud.mask)
    assert abs(fl["c2_c1_mm"] - truth["c2_c1_mm"]) < 0.015
    assert all(abs(fl[f]["offset_mm"]) < 0.006 for f in ("c1", "c2"))
    assert fl["c2"]["plane_rms_mm"] < 0.5 * off["c2"]["plane_rms_mm"]  # less texture + noise
    assert 0.75 < fl["c1"]["dent_mm"] / off["c1"]["dent_mm"] < 0.95    # but a shallower dent
    with pytest.raises(ValueError, match="fine_prefilter_px"):
        api.fast_scan_flash(str(scene["pack"]), "zone", "e2e_bad", p, roi=scene["roi"],
                            fine_prefilter_px=9.0, dry_run=True)


def test_cli_and_web_take_the_fine_prefilter(scene, tmp_path, capsys):
    import threading
    from ui import cli
    from ui.web import server as ui_server
    from test_ui_server import Client
    lib = tmp_path / "library"
    try:
        code = cli.main(["--mock", "--library", str(lib), "--json", "scan-flash", "--dry-run",
                         "--pack", str(scene["pack"]), "--variant", "zone", "--roi",
                         ",".join(str(v) for v in scene["roi"]), "--fine-prefilter", "1"])
        doc = json.loads(capsys.readouterr().out)
        assert code == 0 and doc["layout"]["fine_prefilter_px"] == 1.0, doc
    finally:
        api._session = None
    presets = tmp_path / "presets"
    presets.mkdir()
    (presets / "cell_wall_2p7ft_vstripe.json").write_text(
        (FIXTURE_PRESETS / "cell_wall_2p7ft_vstripe.json").read_text())
    httpd, app = ui_server.make_server(mock=True, library=tmp_path / "wlib",
                                       presets_dir=presets, port=0)
    t = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
    t.start()
    try:
        c = Client(app.port)
        status, j = c.post("/api/presets/cell_wall_2p7ft_vstripe/load", {})
        assert status == 200 and j["status"] == "done", j
        j = c.done("fast_scan_flash", {"dry_run": True, "pack": str(scene["pack"]),
                                       "variant": "zone", "roi": scene["roi"],
                                       "fine_prefilter_px": 1})
        assert j["result"]["layout"]["fine_prefilter_px"] == 1.0
        j = c.job("fast_scan_flash", {"dry_run": True, "fine_prefilter_px": 7})
        assert j["status"] == "error" and j["error_type"] == "ArgError"
    finally:
        httpd.shutdown()
        httpd.server_close()
        api._session = scene["s"]


def test_nothing_here_can_reach_the_projector():
    """Rule 1 for the whole flash path, pack side and capture side: no DLPC350 USB / HID,
    no dmd_control, no projector window in the modules this test drove."""
    import ast
    from conftest import ROOT
    for f in ("engine/bases/onebit.py", "tools/make_flash_pack.py", "engine/capture/softsync.py",
              "engine/capture/mock_flashloop.py", "engine/reconstruct/flash_scan.py"):
        names = []
        for node in ast.walk(ast.parse((ROOT / f).read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                names += [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names += [f"{node.module or ''}.{a.name}" for a in node.names]
        for bad in ("dmd_control", "hid", "dlp_driver", "swiss_army_sensor", "projector",
                    "patternmode"):
            assert not [n for n in names if bad in n.lower().split(".")], (f, bad)
    rules.check_scan_path_usb_free()

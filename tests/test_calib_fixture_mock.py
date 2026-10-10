"""Calibration fixture + check block, mock only (MOCK-TESTED ONLY: neither has been used on a
real rig): the print generator (closed STL meshes on the bed, specs that round-trip, stickers
at exact scale), one fixture capture = one view per face with disjoint marker ids, the
3-placement calibration (independent face views + the rigid-fixture refinement) against the
mock rig's true K / R / t, and the check-block measurement (registration, step heights,
caliper values, CONTINUE / REJECT / PULL)."""

import json
import math
import struct

import numpy as np
import pytest

from engine import api, rules
from engine.bases.multifreq import MultiFreqBasis
from engine.calibrate import fixture as fx, rigid, stereo
from engine.calibrate.charuco import BoardSpec, detect
from engine.capture.mock_bench import MockBench
from engine.capture.mock_stereo import (ConvexSolid, StereoRig, StereoScene, check_block_placement,
                                        check_block_shapes, fixture_placements, fixture_shapes)
from engine.measure import checkblock as cb
from engine.reconstruct.triangulate import triangulate
from tools import make_calib_fixture as mcf
from conftest import FIXTURE_PRESETS

BED = (220.0, 220.0)


def _cal(d, gid="mock-g"):
    """A StereoCalibration from a fit-like dict (cam / proj / R / t) or the rig truth."""
    if "cam" in d:
        return stereo.StereoCalibration(id="fit", geometry_id=gid, K_cam=d["cam"]["K"],
                                        dist_cam=d["cam"]["dist"], K_proj=d["proj"]["K"],
                                        dist_proj=d["proj"]["dist"], R=d["R"], t=d["t"],
                                        diamond_parity=1)
    return stereo.StereoCalibration(id="truth", geometry_id=gid, K_cam=d["K_cam"],
                                    dist_cam=d["dist_cam"], K_proj=d["K_proj"],
                                    dist_proj=d["dist_proj"], R=d["R"], t=d["t"],
                                    diamond_parity=d["parity"])


# ------------------------------------------------------------------ the print files

@pytest.fixture(scope="module")
def package(tmp_path_factory):
    out = tmp_path_factory.mktemp("pkg")
    return mcf.make(out, fixture_kw={}, block_kw={}, dpi=200.0)


def test_generator_writes_closed_meshes_that_fit_the_bed(package):
    stls = list(package["fixture"]["stl"]) + [package["block"]["stl"]]
    names = {p.name for p in stls}
    assert {"fixture_base.stl", "fixture_wedge_0.stl", "fixture_wedge_3.stl",
            "checkblock.stl"} <= names
    for p in stls:
        data = p.read_bytes()
        n = struct.unpack("<I", data[80:84])[0]
        assert len(data) == 84 + 50 * n, p.name                    # a well-formed binary STL
        v, f = mcf.read_stl(p)
        rep = mcf.mesh_report(v, f)
        # every edge in exactly two triangles, once each way: closed + consistently oriented
        assert rep["closed"] and rep["edges_not_in_2_triangles"] == 0, (p.name, rep)
        assert rep["degenerate_triangles"] == 0 and rep["volume_mm3"] > 0, (p.name, rep)
        w, h, z = rep["size_mm"]
        assert max(w, h) <= max(BED) - 10 and min(w, h) <= min(BED) - 10, (p.name, rep)
        assert v[:, 2].min() == pytest.approx(0.0, abs=1e-4), p.name     # on the bed
    blk = mcf.mesh_report(*mcf.read_stl(package["block"]["stl"]))
    spec = package["block_spec"]
    assert blk["size_mm"][:2] == pytest.approx([spec["slab"]["L"], spec["slab"]["D"]], abs=1e-3)
    tallest = max(f["height_mm"] for f in spec["features"] if f["kind"] == "step")
    assert blk["size_mm"][2] == pytest.approx(spec["slab"]["T"] + tallest, abs=1e-3)
    readme = package["readme"].read_text(encoding="utf-8")
    assert "MOCK-TESTED ONLY" in readme and "square_mm_measured" in readme


@pytest.mark.parametrize("kw", [{"n_faces": 3}, {"n_faces": 6}, {"face_style": "recess"},
                                {"face_style": "flat", "tilt_deg": [25, 35, 25, 35]}])
def test_other_designs_are_closed_and_fit(kw):
    spec = fx.design(**kw)
    parts = []
    for p in spec["parts"]:
        if p["kind"] == "base":
            parts += [m[1:] for m in mcf.base_meshes(p)]
        elif p["kind"] == "wedge":
            parts.append(mcf.wedge_mesh(p))
        else:
            parts.append(mcf.box_mesh(p["size"]))
    assert len([p for p in spec["parts"] if p["kind"] == "wedge"]) == len(spec["faces"])
    for v, f in parts:
        rep = mcf.mesh_report(v, f)
        assert rep["closed"] and rep["volume_mm3"] > 0, rep
        assert max(rep["size_mm"][:2]) <= max(BED) - 10, rep
    if kw.get("n_faces") == 6:                       # too wide for one bed: tiles + splices
        assert spec["design"]["base_tiles"] == 2
        assert any(p["kind"] == "splice" for p in spec["parts"])


def test_too_big_for_the_bed_is_refused():
    with pytest.raises(ValueError, match="bed"):
        fx.design(square_mm=25.0)
    with pytest.raises(ValueError, match="bed"):
        cb.design(step_heights=[5, 10, 15, 20, 25, 30, 35, 40, 45, 50], step_width_mm=20)


def test_specs_round_trip(package):
    d = package["dir"]
    spec = fx.load(d / "fixture_spec.json")
    assert spec == json.loads(json.dumps(package["fixture_spec"]))
    assert fx.fingerprint(spec) == fx.fingerprint(package["fixture_spec"])
    ranges = [tuple(f["id_range"]) for f in spec["faces"]]
    assert len(ranges) == 4 and ranges[0] == (100, 114)
    for (a0, a1), (b0, b1) in zip(ranges, ranges[1:]):
        assert b0 == a1 + 1                                        # disjoint, contiguous
    for f in spec["faces"]:
        R, t = fx.face_pose(f)
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-9) and np.linalg.det(R) > 0
        assert f["square_mm_measured"] is None
    blk = cb.load(d / "checkblock_spec.json")
    assert blk == json.loads(json.dumps(package["block_spec"]))
    tmpl = json.loads((d / "checkblock_measured_TEMPLATE.json").read_text(encoding="utf-8"))
    assert set(tmpl["values"]) == set(cb.references(blk)) and not any(tmpl["values"].values())
    filled = {**tmpl, "values": {**tmpl["values"], "step3_height_mm": 15.08}}
    assert cb.load_measured(filled, blk) == {"step3_height_mm": 15.08}


def test_bad_specs_are_refused(package):
    spec = json.loads(json.dumps(package["fixture_spec"]))
    clash = json.loads(json.dumps(spec))
    clash["faces"][1]["board"]["first_id"] = 110                   # overlaps face 0
    with pytest.raises(ValueError, match="share marker ids"):
        fx.load(clash)
    off = json.loads(json.dumps(spec))
    off["faces"][0]["square_mm_measured"] = 30.0
    with pytest.raises(ValueError, match="caliper"):
        fx.load(off)
    with pytest.raises(ValueError, match="not a calibration fixture"):
        fx.load(package["block_spec"])
    blk = package["block_spec"]
    with pytest.raises(ValueError, match="not a quantity"):
        cb.load_measured({"values": {"step9_height_mm": 1.0}}, blk)
    with pytest.raises(ValueError, match="not near"):
        cb.load_measured({"values": {"step1_height_mm": 2.5}}, blk)
    with pytest.raises(ValueError, match="do not fit"):
        BoardSpec(6, 5, 16.0, 11.2, "DICT_5X5_100", 90)            # ids 90..104 > 100


def test_stickers_print_at_exact_scale_and_detect(package):
    import cv2
    pngs = sorted(p for p in package["fixture"]["stickers"] if p.suffix == ".png")
    assert len(pngs) == 4 and pngs[0].name == "fixture_face0_ids100-114.png"
    spec = package["fixture_spec"]
    margin = spec["design"]["sticker_margin_mm"]
    for p, f in zip(pngs, spec["faces"]):
        b = BoardSpec.from_dict(f["board"])
        data = p.read_bytes()
        i = data.find(b"pHYs")
        ppm = struct.unpack(">I", data[i + 4:i + 8])[0]
        assert ppm == round(200.0 / 0.0254)                          # 200 dpi recorded
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_GRAYSCALE)
        w_mm, h_mm = (s + 2 * margin for s in b.board_size_mm)
        assert img.shape[1] == pytest.approx(w_mm / 25.4 * 200, abs=1.5)
        assert img.shape[0] == pytest.approx(h_mm / 25.4 * 200, abs=1.5)
        det = detect(img, b)
        assert det is not None and len(det[1]) == b.n_corners
        others = [BoardSpec.from_dict(g["board"]) for g in spec["faces"] if g is not f]
        assert all(detect(img, o) is None for o in others)          # its ids only


def test_face_poses_sit_on_the_wedge_tops(package):
    """The board of each face lies on its wedge's top plane (lifted by the sticker), inside
    the face, and the wedge mesh's top is that plane: one geometry for the spec and the STL."""
    spec = package["fixture_spec"]
    s_mm = spec["design"]["sticker_mm"]
    for f, w in zip(spec["faces"], [p for p in spec["parts"] if p["kind"] == "wedge"]):
        R, t = fx.face_pose(f)
        b = BoardSpec.from_dict(f["board"])
        bw, bh = b.board_size_mm
        corners = np.array([[0, 0, 0], [bw, 0, 0], [0, bh, 0], [bw, bh, 0.0]]) @ R.T + t
        Rw, tw = fx.wedge_to_fixture(w)
        p0, n_w = fx.face_plane_w(w)
        cw = (corners - tw) @ Rw                                      # into the wedge frame
        assert np.allclose((cw - p0) @ n_w, s_mm, atol=1e-9)          # sticker on the face
        u = cw[:, 0]
        assert u.min() > 0 and u.max() < w["Lu"] and np.abs(cw[:, 1]).max() < w["Lv"] / 2
        assert np.allclose(R[:, 2], -np.asarray(f["normal"]), atol=1e-12)
        # the wedge mesh (W frame, before the print turn) has its top on that plane
        a = math.radians(w["tilt_deg"])
        top_z = w["low_mm"] + u * math.tan(a)
        assert np.allclose(cw[:, 2] - s_mm / math.cos(a), top_z, atol=1e-9)
    ang = fx.face_angles(spec)
    assert ang[(0, 2)] == pytest.approx(60.0) and ang[(0, 1)] == pytest.approx(41.4096, abs=1e-3)


# ------------------------------------------------------------------ the mock rig

def test_convex_solid_ray_hits():
    box = ConvexSolid.box([-10, -10, 790], [10, 10, 800])
    d = np.array([[0.0, 0.0, 1.0], [0.05, 0.0, 1.0], [0.0, 0.0, -1.0]])
    t, n = box.intersect(np.zeros(3), d)
    assert t[0] == pytest.approx(790.0) and np.allclose(n[0], [0, 0, -1])
    assert not np.isfinite(t[1]) and not np.isfinite(t[2])          # misses / behind


@pytest.fixture(scope="module")
def rig():
    return StereoRig.bench()


@pytest.fixture(scope="module")
def placements(rig):
    """Three fixture placements rendered noise-free (decoded both axes, 4 steps)."""
    spec = fx.design()
    basis = MultiFreqBasis(steps=4, axis="xy", diamond_parity=1)
    pats = basis.generate()
    out = []
    for R, t in fixture_placements(3):
        sc = StereoScene(rig, fixture_shapes(spec, R, t), gain=250)
        frames = [sc.render(p) for p in pats]
        fld = basis.decode(frames, "mock-g")
        out.append({"image": np.clip(frames[0], 0, 255).astype(np.uint8),
                    "corr_x": fld.corr_x, "corr_y": fld.corr_y})
    return spec, out


def test_one_capture_gives_one_view_per_face_with_disjoint_ids(placements):
    import cv2
    spec, pls = placements
    img = pls[0]["image"]
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
    _, ids, _ = cv2.aruco.ArucoDetector(d).detectMarkers(img)
    ids = set(np.ravel(ids).tolist())
    per_face = []
    for f, b in zip(spec["faces"], fx.boards(spec)):
        lo, hi = f["id_range"]
        mine = {i for i in ids if lo <= i <= hi}
        det = detect(img, b)
        assert det is not None and len(det[1]) >= 12, f["face"]
        assert len(mine) >= 10
        per_face.append(mine)
    assert set().union(*per_face) == ids                            # nothing outside the faces
    assert sum(len(m) for m in per_face) == len(ids)                # disjoint


@pytest.fixture(scope="module")
def fixture_fit(placements):
    spec, pls = placements
    views, where = [], []
    for k, p in enumerate(pls):
        for f, b in zip(spec["faces"], fx.boards(spec)):
            views.append({**p, "spec": b})
            where.append((k, f["face"]))
    fit = stereo.calibrate(views, fx.boards(spec)[0])
    obs = [{**o, "placement": where[o["view"]][0], "face": where[o["view"]][1]}
           for o in fit["observations"]]
    r = rigid.refine(obs, fit["cam"], fit["proj"], fit["R"], fit["t"],
                     {f["face"]: fx.face_pose(f) for f in spec["faces"]})
    return spec, fit, r


def test_three_placements_recover_the_rig(fixture_fit, rig):
    """Same tolerances as the 8-pose single-board test (test_stereo_calibration_mock)."""
    spec, fit, r = fixture_fit
    assert fit["n_poses"] == 12 and r["success"]
    assert r["rms_cam"] < 0.3 and r["rms_proj_u"] < 0.5 and r["rms_proj_v"] < 0.5
    K, Kp = rig.camera.K, rig.projector.K
    Kc, Kf = r["cam"]["K"], r["proj"]["K"]
    assert abs(Kc[0, 0] - K[0, 0]) < 2.0 and abs(Kc[1, 1] - K[1, 1]) < 2.0
    assert abs(Kc[0, 2] - K[0, 2]) < 1.5 and abs(Kc[1, 2] - K[1, 2]) < 1.5
    assert np.allclose(r["cam"]["dist"][:2], rig.camera.dist[:2], atol=0.02)
    assert abs(Kf[0, 0] - Kp[0, 0]) < 3 and abs(Kf[1, 1] - Kp[1, 1]) < 6
    assert abs(Kf[0, 2] - Kp[0, 2]) < 2 and abs(Kf[1, 2] - Kp[1, 2]) < 3
    assert Kf[1, 1] / Kf[0, 0] == pytest.approx(2.0, abs=0.01)
    d = stereo.compare_truth(_cal(r), rig.truth())
    assert d["R_deg"] < 0.05 and d["t_mm"] < 1.0
    assert _cal(r).baseline_mm == pytest.approx(279.4, abs=1.0)
    assert _cal(r).angle_deg == pytest.approx(18.75, abs=0.1)
    # the face-to-face angles the refinement estimated match the (mock = exact) print
    est, ref = rigid.face_normal_angles(r["faces"]), fx.face_angles(spec)
    assert max(abs(est[k] - ref[k]) for k in ref) < 0.1


def test_rigid_refinement_beats_independent_small_faces(fixture_fit, rig):
    """Why the refinement exists: ~100 mm faces at ~700 mm are weak views on their own."""
    _, fit, r = fixture_fit
    ind = stereo.compare_truth(_cal(fit), rig.truth())
    ref = stereo.compare_truth(_cal(r), rig.truth())
    assert ref["R_deg"] < ind["R_deg"] and ref["t_mm"] < ind["t_mm"]
    assert abs(ref["cam_f_px"][0]) < abs(ind["cam_f_px"][0])
    # the per-placement face-angle check from the independent views is still sane
    import cv2
    spec = fixture_fit[0]
    views = [{"placement": i // 4, "face": i % 4, "R": cv2.Rodrigues(rv)[0]}
             for i, rv in zip(fit["used"], fit["cam"]["rvecs"])]
    chk = fx.consistency(spec, views)
    assert chk["ok"] and len(chk["pairs"]) == 6 and all(p["n"] == 3 for p in chk["pairs"])


def test_face_angle_check_flags_a_bad_assembly(fixture_fit):
    spec, _, r = fixture_fit
    bent = {k: (v[0], v[1]) for k, v in r["faces"].items()}
    a = math.radians(3.0)                                   # face 2 sits 3 deg off its spec
    Rx = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]])
    bent[2] = (bent[2][0] @ Rx, bent[2][1])
    chk = fx.consistency(spec, [{"placement": 0, "face": k, "R": v[0]} for k, v in bent.items()])
    assert not chk["ok"] and chk["max_abs_residual_deg"] > 1.0 and chk["warnings"]


# ------------------------------------------------------------------ the check block

@pytest.fixture(scope="module")
def printed_block(rig):
    """The check block as PRINTED: step3 came out 0.12 mm tall (15.12), the rest exact;
    scanned noise-free and triangulated with the true calibration."""
    spec = cb.design()
    printed = json.loads(json.dumps(spec))
    next(f for f in printed["features"] if f["name"] == "step3")["height_mm"] = 15.12
    sc = StereoScene(rig, check_block_shapes(printed, *check_block_placement()), gain=250)
    basis = MultiFreqBasis(steps=8, axis="xy", diamond_parity=1)
    fld = basis.decode([sc.render(p) for p in basis.generate()], "mock-g")
    return spec, triangulate(fld, _cal(rig.truth()))


def test_block_with_caliper_values_continues(printed_block):
    spec, cloud = printed_block
    meas = {**cb.references(spec), "step3_height_mm": 15.12}
    rep = cb.analyze([cloud], spec, {"values": meas})
    assert rep["result"] == "PASS" and rep["disposition"] == "CONTINUE", rep["pull_reasons"]
    rows = {r["name"]: r for r in rep["rows"]}
    for k in ("step1", "step2", "step3", "step4", "step5"):
        assert abs(rows[f"{k}_height_mm"]["error_mm"]) < 0.01, rows[f"{k}_height_mm"]
    assert rows["step3_height_mm"]["scanned_mm"] == pytest.approx(15.12, abs=0.01)
    assert abs(rows["cylinder_diameter_mm"]["error_mm"]) < 0.03
    assert abs(rows["dome_diameter_mm"]["error_mm"]) < 0.03
    assert rows["reference_flatness_rms_mm"]["scanned_mm"] < 0.02
    assert rep["registration"][0]["ok"] and rep["registration"][0]["rms_mm"] < 0.2


def test_block_out_of_tolerance_is_rejected_and_nominal_is_pulled(printed_block):
    spec, cloud = printed_block
    # the operator's caliper says 15.00 where the scan finds 15.12: out of tolerance -> REJECT
    rep = cb.analyze([cloud], spec, {"values": cb.references(spec)})
    assert rep["result"] == "FAIL" and rep["disposition"] == "REJECT"
    bad = [r["name"] for r in rep["rows"] if r["verdict"] == "FAIL"]
    assert bad == ["step3_height_mm"]
    # with a looser tolerance it passes (the tolerance is the user's)
    assert cb.analyze([cloud], spec, {"values": cb.references(spec)},
                      tol_height_mm=0.15)["disposition"] == "CONTINUE"
    # nominal values only: the same FAIL is a doubt (the print, not the rig?) -> PULL
    rep = cb.analyze([cloud], spec)
    assert rep["result"] == "FAIL" and rep["disposition"] == "PULL"
    assert any("NOMINAL" in r["message"] for r in rep["pull_reasons"])


def test_a_repeat_without_the_block_is_a_doubt(printed_block):
    """Of two repeats one never saw the block: the other one's PASS is not enough."""
    spec, cloud = printed_block
    meas = {**cb.references(spec), "step3_height_mm": 15.12}
    rep = cb.analyze([cloud, np.zeros((10, 3))], spec, {"values": meas})
    assert rep["result"] == "PASS" and rep["disposition"] == "PULL"
    assert [r["reason"] for r in rep["pull_reasons"]] == ["incomplete"]
    assert [r["ok"] for r in rep["registration"]] == [True, False]


def test_block_not_in_view_is_invalid(rig):
    spec = cb.design()
    sc = StereoScene(rig, [], gain=250)                               # the empty wall
    cx, cy, m = sc.oracle()
    from engine.decode.deformation import DeformationField
    fld = DeformationField(corr_x=cx, corr_y=cy, corr_diag=None, conf=m.astype(np.float32),
                           mask=m, geometry_id="mock-g", basis_names=["oracle"])
    cloud = triangulate(fld, _cal(rig.truth()))
    for scan in (cloud, np.zeros((10, 3))):                  # nothing in front / too few points
        rep = cb.analyze([scan], spec)
        assert rep["result"] == "INVALID" and rep["disposition"] == "PULL"
        assert rep["pull_reasons"][0]["reason"] == "incomplete" and not rep["rows"]


# ------------------------------------------------------------------ through the engine

@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("fxeng")
    pd = tmp / "presets"
    pd.mkdir()
    d = json.loads((FIXTURE_PRESETS / "cell_wall_2p7ft_vstripe_fullrange.json").read_text())
    d["name"] = "metric_fast"
    d["camera"]["average"] = 1
    (pd / "metric_fast.json").write_text(json.dumps(d))
    s = api.configure(tmp / "library", mock=True, bench=MockBench(seed=5), presets_dir=pd)
    p = api.load_preset("metric_fast")
    api.mock_bench().use_stereo()
    yield {"s": s, "p": p, "tmp": tmp}
    api._session = None


def _use(e):
    api._session = e["s"]
    return e["p"]


def test_engine_fixture_calibration_then_check_block(engine):
    """calib-pose --fixture x3 (the simulator places it) -> calibrate-stereo -> accuracy-block:
    camera noise and the normal capture path, end to end."""
    p = _use(engine)
    api.row_parity(p)
    r = api.calibrate_stereo(p, mock_fixture=3, steps=4)
    assert r["n_placements"] == 3 and r["n_views"] == 12 and r["rigid"]["used"]
    assert r["rms"]["camera_ok"] and r["rms"]["projector_ok"] and r["diamond_parity"] == 1
    t = r["vs_truth"]
    assert t["R_deg"] < 0.1 and t["t_mm"] < 1.5
    assert abs(t["cam_f_px"][0]) < 3 and abs(t["proj_f_px"][0]) < 4
    assert r["face_check"]["ok"] and r["face_check"]["source"] == "rigid estimate"
    poses = [x for x in api.list_library("calib_poses") if x.get("fixture")]
    assert len(poses) == 3 and all(len(x["boards"]) == 4 for x in poses)
    assert all(x["board"] is None and len(x["faces_seen"]) >= 3 for x in poses)
    stored = json.loads((r["path"] / "stereo_calibration.json").read_text())
    assert stored["board"]["fixture"] == "fixture" and len(stored["board"]["faces"]) == 4
    # a fixture set never takes a single-board --square
    with pytest.raises(ValueError, match="square_mm_measured"):
        api.calibrate_stereo(p, set_label=r["set"], square_mm=16.0)
    # the check block, printed exactly to spec, measured with calipers = nominal
    spec = cb.design()
    bench = api.mock_bench()
    bench.clear()
    bench.use_stereo()
    for shape in check_block_shapes(spec, *check_block_placement()):
        bench.place(shape)
    rep = api.accuracy_block(spec, {"values": cb.references(spec)}, steps=6)
    rows = {x["name"]: x for x in rep["rows"]}
    assert rep["result"] == "PASS" and rep["disposition"] == "CONTINUE", rep["rows"]
    assert all(abs(rows[f"step{i}_height_mm"]["error_mm"]) < 0.05 for i in range(1, 6))
    assert rep["file"].is_file() and rep["calibration_id"] == r["calibration_id"]


def test_mock_fixture_is_refused_on_the_real_rig(tmp_path):
    from engine import config
    from engine.calibrate import flow
    api.configure(tmp_path / "lib", mock=False, presets_dir=FIXTURE_PRESETS)
    try:
        with pytest.raises(rules.RuleViolation, match="mock-only"):
            flow.calibrate_stereo(api._session, config.preset_from_dict(
                json.loads((FIXTURE_PRESETS / "cell_wall_2p7ft_vstripe.json").read_text())),
                mock_fixture=3)
    finally:
        api._session = None


# ------------------------------------------------------------------ the command line

def test_cli_refuses_bad_fixture_and_block_input(package, tmp_path, capsys):
    """Bad specs and mock-only flags are refused (exit 2) before anything is captured."""
    from ui import cli
    lib = tmp_path / "lib"
    d = package["dir"]
    try:
        for argv in (["--mock", "accuracy-block", "--block", str(d / "fixture_spec.json")],
                     ["--mock", "calib-pose", "--fixture", str(d / "checkblock_spec.json")],
                     ["calibrate-stereo", "--mock-fixture", "3"],
                     ["--mock", "accuracy-block", "--block", str(d / "checkblock_spec.json"),
                      "--measured", str(d / "fixture_spec.json")]):
            code = cli.main(["--library", str(lib), *argv])
            out, err = capsys.readouterr()
            assert code == 2, (argv, out, err)
            assert "Traceback" not in out + err
    finally:
        api._session = None
    code = cli.main(["--help"])
    out, _ = capsys.readouterr()
    assert code == 0 and "accuracy-block" in out and "calib-pose --fixture" in out

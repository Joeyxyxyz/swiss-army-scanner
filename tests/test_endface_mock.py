"""End faces (engine/measure/endface) through the verbs.

Synthetic clouds (tests/face_synth.py: a cell's circular end face toward the camera, with a
rim chamfer, its body behind it and the wall, ray-cast by the mock bench's pinhole camera +
projector; metric scans written as the library writes them, next to the rig's own stereo
calibration) cover locate_face, face_profile, face_golden and check_face: a clean placement
PASSES, a 0.3 mm bowl dent FAILS at its place (MF-CFAR, the default detector, and the band);
with a reflectance texture (flood pair + the height error it causes) the golden's FIR learns
and removes it; edge inputs (half a face, no data, 2 placements) never PASS by accident; the
golden is append-only and bound to its stereo calibration; rule 6 refuses mm without / after
the calibration. The mock bench
(stereo mode + EndFace) drives the capture path: the locate scan, aim_ring / check_ring
(pumped, cancellable holds; a moved cell is OFF), auto_intensity and targeted_face_scan ->
face_profile, and the refusals before the rig (rules 5 / 6). The real rig is never opened."""

import ast
import dataclasses
import http.client
import json
import shutil
import threading
import time
from pathlib import Path

import numpy as np
import pytest

import face_synth as S
from conftest import FIXTURE_PRESETS, ROOT
from engine import api, config, rules
from engine.capture.mock_bench import MockBench
from engine.capture.mock_stereo import EndFace, MetricCylinder
from engine.measure.endface import golden as golden_mod
from engine.measure.endface import locate as locate_mod
from engine.measure.endface import mfcfar
from engine.reconstruct import mesh as mesh_mod

DENT = {"depth_mm": 0.3, "size_mm": 3.0, "u_mm": 3.0, "v_mm": -4.0}
POSES = {"g1": ((-6.0, 4.0, 780.0), (10, 6)), "g2": ((-4.5, 5.0, 781.0), (14, 5)),
         "g3": ((-6.5, 3.0, 779.0), (8, 9)), "g4": ((-5.0, 4.5, 780.5), (12, 8)),
         "g5": ((-5.5, 3.5, 780.0), (11, 7))}


# ------------------------------------------------------------ synthetic clouds

@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    lib = tmp_path_factory.mktemp("faces") / "library"
    lib.mkdir()
    rg = S.rig()
    cal = S.write_calibration(lib, rg)
    pl = {}
    for i, (name, (c, (tx, ty))) in enumerate(POSES.items()):
        pl[name] = S.placement(lib, rg, cal, name, c, S.normal_from_tilt(tx, ty), seed=i + 1)
    pl["bad"] = S.placement(lib, rg, cal, "bad", (-5.5, 4.0, 780.2), S.normal_from_tilt(12, 6),
                            dents=[DENT], seed=9)
    ctx = {"lib": lib, "rg": rg, "cal": cal, "pl": pl, "s": api.configure(lib, mock=True)}
    ctx["faces"] = {k: api.locate_face(loc, S.DIAMETER, f"{k}_face") for k, (loc, _) in pl.items()}
    ctx["profiles"] = {k: api.face_profile(reps, face=ctx["faces"][k], label=k)
                       for k, (_, reps) in pl.items()}
    ctx["golden"] = api.face_golden([ctx["profiles"][k] for k in ("g1", "g2", "g3", "g4")], "good4")
    yield ctx
    api._session = None


def _use(ctx):
    api._session = ctx["s"]


def test_locate_face_finds_the_true_face(synth):
    _use(synth)
    for k, (c, (tx, ty)) in POSES.items():
        t = synth["faces"][k]
        n = S.normal_from_tilt(tx, ty)
        assert np.linalg.norm(np.asarray(t.centre_mm) - c) < 0.12, (k, t.centre_mm, c)
        assert np.degrees(np.arccos(min(1.0, abs(np.asarray(t.normal) @ n)))) < 0.3, k
        assert t.rim_rms_mm < 0.2 and t.n_rim_points >= 30 and t.face_diameter_mm == S.DIAMETER
        assert max(t.affine["rms_col_px"], t.affine["rms_row_px"]) < 0.1 and not t.warnings
        assert t.calibration_id == synth["cal"].id and t.geometry_id == S.GEO
        assert (t.path / "face.npz").is_file() and (t.path / "meta.json").is_file()
        assert t.path.parent.parent.name == "faces"
    ids = [t.id for t in api.list_library("faces")]
    assert synth["faces"]["g1"].id in ids and len(ids) == len(synth["faces"])
    # the DMD ring maps the located face: the camera ellipse centre goes to the DMD centre
    t = synth["faces"]["g1"]
    u, v = synth["rg"].projector.project(np.asarray(t.centre_mm) @ synth["rg"].R.T + synth["rg"].t)
    assert abs(u - t.dmd_centre[0]) < 0.6 and abs(v - t.dmd_centre[1]) < 1.2


def test_locate_skips_a_nearer_non_face_object(tmp_path):
    """The prototype's crop is gone: a fixture edge nearer than the face (a long thin bar, as on
    the bench) is set aside and the face behind it is found."""
    from engine.capture.mock_stereo import Block
    rg = S.rig()
    c, n = np.array((-6.0, 4.0, 780.0)), S.normal_from_tilt(10, 6)
    objs = S.shapes(c, n) + [Block(center=(-6.0, 24.0, 720.0), half=(40.0, 2.0, 5.0))]
    box = S.crop_box(rg, c, margin_mm=34.0)
    X, uv, ok, _ = S.render(rg, objs, box)
    x0, y0, x1, y1 = box
    xyz = np.full(S.CAM + (3,), np.nan)
    m = np.zeros(S.CAM, bool)
    xyz[y0:y1, x0:x1], m[y0:y1, x0:x1] = np.where(ok[..., None], X, np.nan), ok
    cx = np.where(m, 0.0, np.nan)
    cx[y0:y1, x0:x1] = np.where(ok, uv[..., 0], np.nan)
    cy = np.where(m, 0.0, np.nan)
    cy[y0:y1, x0:x1] = np.where(ok, uv[..., 1], np.nan)
    r = locate_mod.locate(xyz, m, cx, cy, m, rg.camera.K, rg.camera.dist, S.DIAMETER)
    assert np.linalg.norm(r["centre_mm"] - c) < 0.15
    assert r["rejected_components"] and r["rejected_components"][0]["z_near_mm"] < 730
    with pytest.raises(locate_mod.FaceNotFound):                   # no face in this roi
        locate_mod.locate(xyz, m, cx, cy, m, rg.camera.K, rg.camera.dist, S.DIAMETER,
                          roi=(x0, y0, x0 + 40, y0 + 40))


def test_face_profile_maps_the_face_in_mm(synth):
    _use(synth)
    pr = synth["profiles"]["g1"]
    st = pr.stats
    assert pr.units["h_mm"].startswith("mm") and pr.calibration_id == synth["cal"].id
    assert st["n_repeats"] == 2 and st["flatness_rms_mm"] < 0.03
    assert 0.005 < st["repeatability_median_mm"] < 0.03              # 0.025 mm noise per scan
    assert st["grid_coverage_pct"] > 95 and abs(st["tilt_deg"] - 11.4) < 1.5
    rad = {round(r["r_mm"], 2): r["h_median_mm"] for r in pr.radial}
    assert abs(rad[5.25]) < 0.02 and rad[10.25] < -0.04              # the 0.6 x 0.15 mm chamfer
    for k in ("png", "csv_radial", "csv_sections", "json", "npz", "stl", "stl_meta"):
        assert Path(pr.files[k]).is_file(), k
    assert Path(pr.files["csv_radial"]).read_text().startswith("r_mm,h_median_mm")
    doc = json.loads(Path(pr.files["json"]).read_text())
    assert doc["kind"] == "face_profile" and doc["stats"]["flatness_rms_mm"] == pytest.approx(
        st["flatness_rms_mm"])
    hdr = mesh_mod.read_stl_header(pr.files["stl"])
    assert hdr["size_ok"] and hdr["finite"] and hdr["n_triangles"] > 2000
    meta = json.loads(Path(pr.files["stl_meta"]).read_text())
    assert meta["units"]["xyz"] == "mm" and "face" in meta["frame"]
    assert meta["bbox_max"][0] - meta["bbox_min"][0] == pytest.approx(S.DIAMETER, abs=1.0)
    with np.load(pr.files["npz"]) as z:
        assert z["height_mm"].shape == (len(z["grid_mm"]),) * 2
        assert json.loads(str(z["meta"]))["face_diameter_mm"] == S.DIAMETER


def test_golden_passes_a_clean_face_and_flags_a_bowl_dent(synth):
    _use(synth)
    g = synth["golden"]
    assert g.n_placements == 4 and g.face_diameter_mm == S.DIAMETER
    assert (g.path / "golden.npz").is_file() and g.path.parent.parent.name == "face_golden"
    assert g.params["detector"] == "mf_fir" and g.meta["detector"] == "mf_fir"
    assert g.params["mf"]["z_thresh"] == 5.5 and g.params["band"]["k_sigma"] == 4.0
    assert not g.params["fir"]["used"] and any("flood pair" in w for w in g.warnings)
    with np.load(g.path / "golden.npz") as z:
        assert {"golden_mm", "band_mm", "mf_radial_h_mm", "mf_map_mm"} <= set(z.files)
        assert "fir_coef" not in z.files
    assert [r["n_face"] + r["n_rim"] for r in g.leave_one_out] == [0, 0, 0, 0]
    assert all(r["mf_fir"]["n_face"] == 0 and r["band"]["n_face"] == 0 for r in g.leave_one_out)
    assert all(r["max_z"] < 5.5 for r in g.leave_one_out)
    assert g.meta["leave_one_out_false_triggers"]["mf_fir"] == {"face": 0, "rim": 0}
    sens = g.sensitivity
    assert sens["detector"] == "mf_fir" and sens["total"] == 4 * 4 * 6 * 4
    big = [b for b in sens["by_size"] if b["depth_mm"] >= 0.2 and b["width_mm"] >= 2]
    assert all(b["detected"] == b["of"] for b in big), big
    assert all(set(b["pct_at_z"]) == {"4.5", "5", "5.5", "6"} for b in sens["by_size"])
    assert g.meta["placement_spread"]["pair_rms_median_mm"] < 0.03
    ok = api.check_face(synth["profiles"]["g5"])                      # newest matching golden
    assert ok.verdict == "PASS" and ok.n_triggers == 0 and ok.golden_id == g.id
    assert ok.detector == "mf_fir" and ok.stats["max_z"] < 5.5 and not ok.stats["fir_applied"]
    assert ok.stats["compared_pct"] > 95 and ok.stats["noise_floor_mm"]["1.5"] < 0.02
    bad = api.check_face(synth["profiles"]["bad"], g)
    assert bad.verdict == "FAIL"
    face = [t for t in bad.triggers if t["zone"] == "face"]
    assert len(face) == 1 and bad.n_triggers == 1, bad.triggers
    t = face[0]
    assert np.hypot(t["u_mm"] - DENT["u_mm"], t["v_mm"] - DENT["v_mm"]) < 0.6
    assert 0.18 < t["depth_mm"] < 0.4 and t["z"] > 10 and t["kind"] == "dent"   # 0.3 mm bowl
    assert t["width_mm"] >= 2 and t["dev_mm"] < -0.15 and t["area_mm2"] > 0.5
    assert t["theta_deg"] == pytest.approx(np.degrees(np.arctan2(-4, 3)), abs=8)
    for k in ("png", "json", "npz"):
        assert Path(bad.files[k]).is_file(), k
    rep = json.loads(Path(bad.files["json"]).read_text())
    assert rep["verdict"] == "FAIL" and rep["triggers"][0]["depth_mm"] > 0
    assert rep["units"]["depth_mm"].startswith("mm") and rep["units"]["dev_mm"].startswith("mm")
    assert rep["detector"] == "mf_fir" and rep["params"]["mf"]["z_thresh"] == 5.5
    with np.load(bad.files["npz"]) as z:
        assert {"dev_mm", "z_map", "depth_map_mm"} <= set(z.files)
    # the placement's .npz (output bin) is a placement too (per-pixel data included)
    again = api.check_face(synth["profiles"]["bad"].files["npz"], g, "bad_from_npz")
    assert again.verdict == "FAIL" and again.triggers[0]["z"] == pytest.approx(t["z"])
    # the band detector from the same golden: the same dent, as a band excess
    band = api.check_face(synth["profiles"]["bad"], g, "bad_band", detector="band")
    assert band.detector == "band" and band.verdict == "FAIL"
    bt = [x for x in band.triggers if x["zone"] == "face"]
    assert len(bt) == 1 and -0.33 < bt[0]["dev_mm"] < -0.18 and bt[0]["area_mm2"] > 1.5
    assert np.hypot(bt[0]["u_mm"] - DENT["u_mm"], bt[0]["v_mm"] - DENT["v_mm"]) < 0.6
    assert api.check_face(synth["profiles"]["g5"], g, "g5_band", detector="band").verdict == "PASS"
    # per-check trigger override: an absurd z threshold passes the dent (and is recorded)
    hi = api.check_face(synth["profiles"]["bad"], g, "bad_hi_z", z_thresh=1000.0)
    assert hi.verdict == "PASS" and hi.params["mf"]["z_thresh"] == 1000.0
    deep = api.check_face(synth["profiles"]["bad"], g, "bad_deep", min_depth_mm=0.6)
    assert deep.verdict == "PASS"


def test_the_detector_is_one_swappable_function(synth):
    _use(synth)
    seen = {}

    def mine(dev_map, band, params):
        seen.update(params)
        return []
    r = api.check_face(synth["profiles"]["bad"], synth["golden"], "swapped", detector=mine)
    assert r.verdict == "PASS" and r.params["detector"] == "mine" and r.detector == "mine"
    assert {"grid_mm", "radius_mm", "rim_r_mm", "min_area_mm2"} <= set(seen)
    mf = api.check_face(synth["profiles"]["bad"], synth["golden"], "mf_named", detector="mf")
    assert mf.verdict == "FAIL" and mf.detector == "mf" and not mf.stats["fir_applied"]
    with pytest.raises(ValueError, match="detector must be one of"):
        api.check_face(synth["profiles"]["bad"], synth["golden"], detector="nope")
    with pytest.raises(ValueError, match="unknown MF-CFAR params"):
        api.face_golden([synth["profiles"][k] for k in ("g1", "g2")], mf_params={"zz": 1})
    dev = np.zeros((9, 9))
    dev[3:6, 3:6] = -0.5
    g = np.arange(-4, 5) * 0.25
    trig = golden_mod.detect_face_triggers(dev, np.full(dev.shape, 0.1),
                                           {"grid_mm": g, "radius_mm": 1.0, "rim_r_mm": 0.9,
                                            "min_area_mm2": 0.25})
    assert len(trig) == 1 and trig[0]["zone"] == "face" and trig[0]["area_mm2"] == 0.56


def test_golden_is_append_only_and_bound(synth, tmp_path):
    _use(synth)
    first = synth["golden"]
    before = (first.path / "meta.json").read_bytes()
    second = api.face_golden([synth["profiles"][k] for k in ("g1", "g2", "g3")], "good4",
                             synthetic=False)
    assert second.path != first.path and second.label == first.label
    assert (first.path / "meta.json").read_bytes() == before
    with pytest.raises(rules.OverwriteRefused):
        synth["s"].store.write_json(first.path / "meta.json", {"kind": "face_golden"})
    ids = [x.id for x in api.list_library("face_golden")]
    assert first.id in ids and second.id in ids
    # one golden per face type: another diameter is refused
    other = dataclasses.replace(synth["profiles"]["g5"], face_diameter_mm=18.0)
    with pytest.raises(ValueError, match="golden is for a 21.05 mm face"):
        api.check_face(other, first)
    with pytest.raises(ValueError, match="at least 2"):
        api.face_golden([synth["profiles"]["g1"]])
    # a band-default golden: check_face uses its detector unless told otherwise
    gb = api.face_golden([synth["profiles"][k] for k in ("g1", "g2", "g3")], "bandgold",
                         synthetic=False, detector="band")
    assert gb.params["detector"] == "band" and not gb.sensitivity
    r = api.check_face(synth["profiles"]["bad"], gb)
    assert r.detector == "band" and r.verdict == "FAIL"


def test_edge_inputs_never_pass_by_accident(synth):
    """Half the face unmeasured (face partly off the projection), no data at all (an all-NaN
    band / deviation), and a golden of only 2 placements: no crash, and nothing PASSES that was
    not measured."""
    _use(synth)
    g = synth["golden"]
    pr = synth["profiles"]["g5"]
    px = pr._pixels
    GU, _ = np.meshgrid(pr._grid, pr._grid)
    half_h = np.where(np.nanmean(px.U, 0)[None] < 0, np.nan, px.H)          # u < 0 not lit
    half = dataclasses.replace(pr, label="half", _pixels=px.with_heights(half_h),
                               _height=np.where(GU < 0, np.nan, pr._height))
    for det in ("mf_fir", "band"):
        r = api.check_face(half, g, f"half_{det}", detector=det)
        assert r.verdict == "INCOMPLETE" and r.stats["compared_pct"] < 60, (det, r.stats)
        assert any("NOT passed" in w for w in r.warnings)
    empty = dataclasses.replace(pr, label="empty", _pixels=px.with_heights(np.full(px.H.shape, np.nan)),
                                _height=np.full(pr._height.shape, np.nan))
    for det in ("mf_fir", "mf", "band"):
        r = api.check_face(empty, g, f"empty_{det}", detector=det)
        assert r.verdict == "INCOMPLETE" and r.n_triggers == 0, (det, r.verdict)
    # no per-pixel data (a profile .npz from before the MF): runs on the gridded map, warned
    old = dataclasses.replace(synth["profiles"]["bad"], label="old", _pixels=None)
    r = api.check_face(old, g, "old_npz")
    assert r.verdict == "FAIL" and any("per-pixel" in w for w in r.warnings)
    # a golden of 2 placements: no leave-one-out, but a working model
    g2 = api.face_golden([synth["profiles"]["g1"], synth["profiles"]["g2"]], "two")
    assert g2.n_placements == 2 and not g2.leave_one_out and not g2.sensitivity
    assert any("fewer than 3" in w for w in g2.warnings)
    assert api.check_face(synth["profiles"]["g5"], g2, "vs_two").verdict == "PASS"
    assert api.check_face(synth["profiles"]["bad"], g2, "bad_vs_two").verdict == "FAIL"
    with pytest.raises(ValueError, match="synthetic"):
        api.face_golden([synth["profiles"][k] for k in ("g1", "g2", "g3")],
                        synthetic={"positions": 0})


def test_rule6_no_calibration_no_mm(synth, tmp_path):
    """A copy of the library: its calibration trashed -> every face verb refuses; a newer
    calibration -> scans / goldens made under the old one are refused."""
    lib = Path(shutil.copytree(synth["lib"], tmp_path / "library"))
    loc, reps = synth["pl"]["g5"]
    face_dir = lib / "faces" / S.GEO / synth["faces"]["g5"].id
    try:
        api.configure(lib, mock=True)
        api.delete(lib / "calibrations" / S.GEO / Path(synth["cal"].path).name, reason="test")
        with pytest.raises(rules.GeometryMismatch, match="rule 6"):
            api.locate_face(loc, S.DIAMETER)
        with pytest.raises(rules.GeometryMismatch, match="rule 6"):
            api.face_profile(reps, face=str(face_dir))
        prof = synth["profiles"]["g5"]
        with pytest.raises(rules.GeometryMismatch, match="rule 6"):
            api.check_face(prof, str(lib / "face_golden" / S.GEO / synth["golden"].id))
        S.write_calibration(lib, synth["rg"], name="20260202T000000_000000_stereo")   # recalibrated
        with pytest.raises(rules.GeometryMismatch, match="re-triangulate"):
            api.face_profile(reps, face=str(face_dir))
        with pytest.raises(rules.GeometryMismatch, match="rebuild"):
            api.check_face(prof, str(lib / "face_golden" / S.GEO / synth["golden"].id))
        n_gold = len(list((lib / "face_golden" / S.GEO).iterdir()))
        with pytest.raises(rules.GeometryMismatch):
            api.face_golden([synth["profiles"][k] for k in ("g1", "g2", "g3")])
        assert len(list((lib / "face_golden" / S.GEO).iterdir())) == n_gold   # nothing written
    finally:
        _use(synth)


def test_cli_face_check_exit_codes_and_json(synth, capsys):
    from ui import cli
    fs_less = synth["profiles"]["bad"].files["npz"]
    try:
        code = cli.main(["--mock", "--library", str(synth["lib"]), "--json", "face-check",
                         str(fs_less)])
        doc = json.loads(capsys.readouterr().out)
        assert code == 1 and doc["check"]["verdict"] == "FAIL"
        assert doc["check"]["triggers"][0]["zone"] == "face"
        code = cli.main(["--mock", "--library", str(synth["lib"]), "face-check",
                         str(synth["profiles"]["g5"].files["npz"])])
        out = capsys.readouterr().out
        assert code == 0 and "PASS" in out and "mf_fir" in out
        code = cli.main(["--mock", "--library", str(synth["lib"]), "face-check", "--detector",
                         "band", str(fs_less)])
        out = capsys.readouterr().out
        assert code == 1 and "FAIL" in out and "band mm" in out
        code = cli.main(["--mock", "--library", str(synth["lib"]), "--json", "face-check", "--z",
                         "1000", str(fs_less)])
        doc = json.loads(capsys.readouterr().out)
        assert code == 0 and doc["check"]["verdict"] == "PASS"
        loc, _ = synth["pl"]["g3"]
        code = cli.main(["--mock", "--library", str(synth["lib"]), "--json", "face-locate", loc,
                         "--diameter", str(S.DIAMETER), "--label", "cli_face"])
        doc = json.loads(capsys.readouterr().out)
        assert code == 0 and doc["face"]["label"] == "cli_face"
        assert doc["face"]["affine"]["rms_col_px"] < 0.1
        code = cli.main(["--mock", "--library", str(synth["lib"]), "faces"])
        out = capsys.readouterr().out
        assert code == 0 and "cli_face" in out and "good4" in out
    finally:
        _use(synth)


def test_web_face_jobs_take_library_ids_only(synth, tmp_path):
    """The web job handlers: locate_face on a metric scan id, face_golden from face-scan ids,
    check_face on a face scan (FAIL on the dented one); server-side paths are refused."""
    from ui.web import server as ui_server
    lib = Path(shutil.copytree(synth["lib"], tmp_path / "library"))
    fs = {k: S.write_facescan(lib, f"20260101T0300{i:02d}_000000_fs_{k}", synth["faces"][k].id,
                              synth["pl"][k][1])
          for i, k in enumerate(("g1", "g2", "g3", "g4", "bad"))}
    presets = tmp_path / "presets"
    shutil.copytree(FIXTURE_PRESETS, presets)
    httpd, app = ui_server.make_server(mock=True, library=lib, presets_dir=presets, port=0)
    t = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
    t.start()

    def job(verb, args):
        conn = http.client.HTTPConnection("127.0.0.1", app.port, timeout=300)
        conn.request("POST", "/api/jobs", body=json.dumps({"verb": verb, "args": args}),
                     headers={"Content-Type": "application/json"})
        j = json.loads(conn.getresponse().read())
        conn.close()
        t0 = time.monotonic()
        while j.get("status") in ("queued", "running"):
            assert time.monotonic() - t0 < 240, j
            time.sleep(0.2)
            conn = http.client.HTTPConnection("127.0.0.1", app.port, timeout=60)
            conn.request("GET", f"/api/jobs/{j['id']}")
            j = json.loads(conn.getresponse().read())
            conn.close()
        return j

    try:
        j = job("locate_face", {"metric_id": synth["pl"]["g2"][0], "face_diameter_mm": S.DIAMETER,
                                "label": "webface"})
        assert j["status"] == "done", j.get("error")
        assert j["result"]["label"] == "webface" and j["result"]["affine"]["rms_col_px"] < 0.1
        j = job("face_golden", {"facescan_ids": [fs[k] for k in ("g1", "g2", "g3", "g4")],
                                "label": "webgold", "synthetic": False})
        assert j["status"] == "done", j.get("error")
        assert j["result"]["n_placements"] == 4
        j = job("check_face", {"facescan_id": fs["bad"]})
        assert j["status"] == "done", j.get("error")
        assert j["result"]["verdict"] == "FAIL" and j["result"]["triggers"][0]["zone"] == "face"
        assert j["result"]["detector"] == "mf_fir" and j["result"]["triggers"][0]["depth_mm"] > 0.15
        j = job("check_face", {"facescan_id": fs["bad"], "detector": "band"})
        assert j["status"] == "done" and j["result"]["detector"] == "band", j.get("error")
        j = job("check_face", {"facescan_id": fs["bad"], "detector": "nope"})
        assert j["status"] == "error" and "detector" in j["error"]
        j = job("face_profile", {"metric_ids": list(synth["pl"]["g5"][1]),
                                 "face_id": synth["faces"]["g5"].id, "label": "web_g5"})
        assert j["status"] == "done", j.get("error")
        assert j["result"]["stats"]["flatness_rms_mm"] < 0.03
        j = job("check_face", {"facescan_id": str(lib / "facescans" / S.GEO / fs["bad"])})
        assert j["status"] == "error" and j["error_type"] == "EntryNotFound"
    finally:
        httpd.shutdown()
        httpd.server_close()
        app.close()
        _use(synth)


# ------------------------------------------------------------ reflectance FIR (synthetic)

REFL_K_MM = 0.15            # height error = k x d log(reflectance) / d x_cam (~0.03 mm rms)


@pytest.fixture(scope="module")
def synth_refl(tmp_path_factory):
    """4 good placements + a clean and a dented test placement, each with its own reflectance
    texture: the flood pair shows it, every repeat carries the height error it causes."""
    lib = tmp_path_factory.mktemp("faces_refl") / "library"
    lib.mkdir()
    rg = S.rig()
    cal = S.write_calibration(lib, rg)
    s = api.configure(lib, mock=True)
    prof = {}
    cases = dict(POSES)
    cases["dent"] = ((-5.5, 4.0, 780.2), (12, 6))
    try:
        for i, (name, (c, (tx, ty))) in enumerate(cases.items()):
            refl = S.refl_texture(100 + i)
            loc, reps = S.placement(lib, rg, cal, f"r{name}", c, S.normal_from_tilt(tx, ty),
                                    dents=[{**DENT, "depth_mm": 0.12, "size_mm": 2.0}]
                                    if name == "dent" else (), seed=20 + i, refl=refl,
                                    refl_k_mm=REFL_K_MM)
            face = api.locate_face(loc, S.DIAMETER, f"r{name}_face")
            fs = S.write_facescan(lib, f"20260101T0400{i:02d}_000000_fs_r{name}", face.id, reps,
                                  refl=refl, box=S.crop_box(rg, c))
            prof[name] = api.face_profile(fs, label=f"r{name}")
        gold = api.face_golden([prof[k] for k in ("g1", "g2", "g3", "g4")], "refl4",
                               synthetic=False)
        yield {"s": s, "prof": prof, "golden": gold}
    finally:
        api._session = None


def test_fir_learns_and_removes_the_reflectance_height_error(synth_refl):
    api._session = synth_refl["s"]
    g, prof = synth_refl["golden"], synth_refl["prof"]
    assert all(p.stats["flood_pair"] and p._pixels.has_flood for p in prof.values())
    fir = g.params["fir"]
    assert fir["used"] and len(fir["coef"]) == 49 and fir["fit"]["explained_pct"] > 50, fir
    with np.load(g.path / "golden.npz") as z:
        assert z["fir_coef"].shape == (49,)
    # the learned kernel is the camera-x derivative: antisymmetric along x, ~0 along y
    k = np.asarray(fir["coef"]).reshape(7, 7)
    assert abs(k[3, 4] - k[3, 2]) > 3 * abs(k[2, 3] - k[4, 3])
    assert all(r["mf_fir"]["n_face"] + r["mf_fir"]["n_rim"] == 0 for r in g.leave_one_out), \
        [(r["placement"], r["mf_fir"], r["mf"]) for r in g.leave_one_out]
    ok_fir = api.check_face(prof["g5"], g, "r5_fir")
    ok_mf = api.check_face(prof["g5"], g, "r5_mf", detector="mf")
    assert ok_fir.verdict == "PASS" and ok_fir.stats["fir_applied"] and ok_mf.verdict == "PASS"
    nf, nm = ok_fir.stats["noise_floor_mm"]["1.5"], ok_mf.stats["noise_floor_mm"]["1.5"]
    assert nf < 0.8 * nm, (nf, nm)                      # the texture is (mostly) gone
    d_fir = api.check_face(prof["dent"], g, "rdent_fir")
    d_mf = api.check_face(prof["dent"], g, "rdent_mf", detector="mf")
    assert d_fir.verdict == "FAIL", d_fir.stats
    t = d_fir.triggers[0]
    assert np.hypot(t["u_mm"] - DENT["u_mm"], t["v_mm"] - DENT["v_mm"]) < 0.8
    zf = max(x["z"] for x in d_fir.triggers)
    zm = max([x["z"] for x in d_mf.triggers], default=d_mf.stats["max_z"])
    assert zf > zm                                       # the FIR makes the dent more significant
    # a profile without the flood pair against a FIR golden: checked as 'mf', warned
    nof = dataclasses.replace(prof["g5"], label="nof", _pixels=mfcfar.Pixels(
        **{k: getattr(prof["g5"]._pixels, k) for k in ("iy", "ix", "U", "V", "H", "cam_mask")}))
    r = api.check_face(nof, g, "nof")
    assert r.verdict == "PASS" and not r.stats["fir_applied"]
    assert any("no flood pair" in w for w in r.warnings)
    # an over-exposed flood (clipped at 255 over > 2 % of the face): no FIR for it, warned
    px = prof["g5"]._pixels
    on = np.clip(px.flood_on * 3.0, 0, 255)
    hot = dataclasses.replace(prof["g5"], label="hot", _pixels=mfcfar.Pixels(
        px.iy, px.ix, px.U, px.V, px.H, px.cam_mask, on, px.flood_off, on >= 254.5))
    assert mfcfar.saturated_pct(hot._pixels) > 2.0
    r = api.check_face(hot, g, "hot")
    assert r.verdict == "PASS" and not r.stats["fir_applied"]
    assert any("saturates" in w and "auto_intensity" in w for w in r.warnings)


def test_face_partly_off_the_dmd_is_measured():
    from engine.measure.endface import patterns
    e = {"cx": 456.0, "cy": 570.0, "a": 36.0, "b": 72.0, "angle_deg": 0.0}
    assert patterns.off_dmd_pct(e) == 0.0
    assert 45 < patterns.off_dmd_pct(dict(e, cx=0.0)) < 55              # half off the left edge
    assert patterns.off_dmd_pct(dict(e, cx=-100.0)) == 100.0
    m = patterns.face_mask(dict(e, cx=0.0), 6.0)                         # clipped, no crash
    assert m.shape == patterns.DMD_SHAPE and 0 < m.sum() < patterns.face_mask(e, 6.0).sum()


def test_mfcfar_unit_edges():
    """The MF-CFAR pieces on bare arrays: an all-NaN map, a map with too few points, and a
    clean map with one paraboloid dent (depth, width and place recovered)."""
    g = np.arange(-44, 45) * 0.25
    prm = mfcfar.mf_params()
    r = mfcfar.detect(np.full((len(g), len(g)), np.nan), None, g, prm)
    assert r["triggers"] == [] and r["coverage_pct"] == 0 and r["max_z"] is None
    sparse_map = np.full((len(g), len(g)), np.nan)
    sparse_map[40:43, 40:43] = 0.0
    r = mfcfar.detect(sparse_map, None, g, prm)
    assert r["triggers"] == []
    rng = np.random.default_rng(3)
    GU, GV, R = mfcfar.grid_mesh(g)
    H = rng.normal(0, 0.01, R.shape) + golden_mod.bowl(g, 0.12, 2.0, -3.0, 2.0)
    H[R > 10.525] = np.nan
    model = mfcfar.build_model([rng.normal(0, 0.01, R.shape)], g, prm)
    r = mfcfar.detect(np.stack([H, H + rng.normal(0, 0.005, R.shape)]), model, g, prm)
    assert len(r["triggers"]) == 1, r["triggers"]
    t = r["triggers"][0]
    assert np.hypot(t["u_mm"] + 3, t["v_mm"] - 2) < 0.5 and t["width_mm"] in (1.5, 2.0, 3.0)
    assert 0.08 < t["depth_mm"] < 0.16 and t["z"] > prm["z_thresh"]


# ------------------------------------------------------------ mock bench: the capture path

FACE_C = np.array([-8.0, 6.0, 780.0])
FACE_N = S.normal_from_tilt(10.4, 7.0)


def _place(bench, centre=FACE_C, normal=FACE_N):
    bench.clear()
    bench.place(EndFace(center=tuple(centre), normal=tuple(normal), radius=S.RADIUS,
                        chamfer_mm=0.6, chamfer_depth_mm=0.15))
    bench.place(MetricCylinder(center=tuple(centre - normal * 15), radius=S.RADIUS, height=30,
                               axis=tuple(normal)))


@pytest.fixture(scope="module")
def mock_face(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("facemock")
    pd = tmp / "presets"
    pd.mkdir()
    d = json.loads((FIXTURE_PRESETS / "cell_wall_2p7ft_vstripe_fullrange.json").read_text())
    d.update(name="face_fast")
    d["camera"].update(average=1, exposure_us=16667)
    d["projection"].update(invert=False, duty=0.6)
    (pd / "face_fast.json").write_text(json.dumps(d))
    (pd / "face_invert.json").write_text(json.dumps(
        dict(d, name="face_invert", projection=dict(d["projection"], invert=True))))
    (pd / "face_moved.json").write_text(json.dumps(
        dict(d, name="face_moved", rig=dict(d["rig"], baseline_mm=300.0))))
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
    ms = api.scan_metric(p, "locate", method="plane", steps=4)
    face = api.locate_face(ms, S.DIAMETER, "mockface")
    yield {"s": s, "p": p, "pd": pd, "face": face, "cal": cal}
    api._session = None


def _m(ctx):
    api._session = ctx["s"]
    return ctx["p"]


def test_mock_locate_scan_to_face(mock_face):
    _m(mock_face)
    t = mock_face["face"]
    assert np.linalg.norm(np.asarray(t.centre_mm) - FACE_C) < 0.3
    assert np.degrees(np.arccos(min(1.0, abs(np.asarray(t.normal) @ FACE_N)))) < 0.5
    assert max(t.affine["rms_col_px"], t.affine["rms_row_px"]) < 0.15 and not t.warnings
    assert (t.path / "preview.png").is_file()


def test_aim_ring_holds_pumped_and_can_be_cancelled(mock_face):
    p = _m(mock_face)
    r = api.aim_ring(None, p, hold_s=0.6)                     # the newest face target
    assert r.face_id == mock_face["face"].id and r.mode == "aim"
    assert r.held_s == pytest.approx(0.6) and not r.hold_cancelled
    assert Path(r.photo).is_file() and Path(r.overlay).is_file()
    ev = threading.Event()
    ev.set()
    r = api.aim_ring(mock_face["face"], p, hold_s=5.0, offset_dmd=[2, -4], cancel=ev)
    assert r.hold_cancelled and r.held_s == 0 and r.offset_dmd == [2.0, -4.0]
    with pytest.raises(ValueError, match="hold_s"):
        api.aim_ring(None, p, hold_s=-1)


def test_check_ring_on_face_and_off_when_the_cell_moved(mock_face):
    p = _m(mock_face)
    bench = api.mock_bench()
    try:
        r = api.check_ring(mock_face["face"], p)
        assert r.verdict == "ON_FACE", r.warnings
        assert r.lit_pct > 95 and r.spill_pct < 10 and np.hypot(*r.centre_offset_px) < 1.5
        _place(bench, FACE_C + [3.0, 0.0, 0.0])                   # slid 3 mm in its plane
        r = api.check_ring(mock_face["face"], p)
        assert r.verdict == "OFF" and r.lit_pct < 90 and any("moved" in w for w in r.warnings)
        _place(bench, FACE_C + [0.0, 0.0, -4.0])                  # 4 mm toward the camera
        r = api.check_ring(mock_face["face"], p)
        assert r.verdict == "OFF" and np.hypot(*r.centre_offset_px) > 1.5
    finally:
        _place(bench)


def test_auto_intensity_sets_duty_not_leds(mock_face):
    p = _m(mock_face)
    led_before, duty_before = p.projection.led, p.projection.duty
    r = api.auto_intensity(None, p)
    assert r["verified"] and abs(r["p99_dn"] - 200) <= 10 and 0.6 < r["duty"] < 0.9
    assert r["shots"][0]["duty"] == pytest.approx(0.6) and Path(r["file"]).is_file()
    assert p.projection.duty == duty_before and p.projection.led == led_before
    inv = api.load_preset("face_invert")
    try:
        with pytest.raises(ValueError, match="invert"):
            api.auto_intensity(None, inv)
    finally:
        api.load_preset("face_fast")
        mock_face["s"].preset = p


def test_targeted_face_scan_then_profile(mock_face):
    p = _m(mock_face)
    fs = api.targeted_face_scan(None, p, "mk", repeats=2, duty=0.7, steps=4, polar_steps=4)
    assert fs.face_id == mock_face["face"].id and len(fs.metric_scan_ids) == 2 and fs.polar
    assert fs.path.parent.parent.name == "facescans" and (fs.path / "masks.npz").is_file()
    specs = fs.meta["polar_patterns"]
    assert len(specs) == 2 * 4 + 2 * 4 + 2 and specs[-1] == {"kind": "flood", "level": 0}
    x0, y0, x1, y1 = fs.meta["crop_x0y0x1y1"]
    F = np.load(fs.path / "polar_frames_r1.npy")
    assert F.shape == (len(specs), y1 - y0, x1 - x0) and F.dtype == np.float32
    assert F[-2].mean() > F[-1].mean() + 20                  # flood 255 lands bright (invert false)
    assert all(r["face_decoded_pct"] > 90 for r in fs.repeats) and fs.meta["duty"] == 0.7
    ms = [m for m in api.list_library("metric") if m.id in fs.metric_scan_ids]
    assert len(ms) == 2 and all(m.meta["targeted"] and m.meta["method"] == "plane"
                                and m.meta["facescan"] == fs.id for m in ms)
    with np.load(fs.path / "masks.npz") as z:
        dm = z["dmd_mask"]
    assert 1000 < dm.sum() < 20000                           # only the face ellipse is lit
    # decoded pixels are the face (+ margin), not the scene
    assert ms[0].meta["n_points"] < 3 * int(mock_face["face"].meta["face_cam_px"])
    pr = api.face_profile(fs)
    assert pr.facescan_id == fs.id and pr.face_id == mock_face["face"].id
    assert pr.stats["flatness_rms_mm"] < 0.06 and pr.stats["repeatability_median_mm"] < 0.05
    assert pr.stats["grid_coverage_pct"] > 90 and Path(pr.files["stl"]).is_file()
    assert pr.stats["flood_pair"] and pr._pixels.has_flood and pr._pixels.K == 2
    fl = pr._pixels
    assert np.nanmedian(fl.flood_on[fl.iy, fl.ix]) > np.nanmedian(fl.flood_off[fl.iy, fl.ix]) + 20
    # no rings / spokes: the flood pair is still captured (the FIR needs it)
    fs1 = api.targeted_face_scan(None, p, "mk_nopolar", repeats=1, duty=0.7, steps=4, polar=False)
    assert not fs1.polar and fs1.meta["polar_patterns"] == [{"kind": "flood", "level": 255},
                                                            {"kind": "flood", "level": 0}]
    assert np.load(fs1.path / "polar_frames_r1.npy").shape[0] == 2
    assert api.face_profile(fs1).stats["flood_pair"]


def test_face_verbs_refuse_before_the_rig(mock_face, monkeypatch):
    p = _m(mock_face)
    from engine.calibrate import flow

    def boom(*a, **k):
        raise AssertionError("the rig must not be driven")
    monkeypatch.setattr(flow, "capture_patterns", boom)
    monkeypatch.setattr(mock_face["s"], "rig", boom)
    moved = config.load_preset("face_moved", presets_dir=mock_face["pd"], mock=True)
    with pytest.raises(rules.GeometryMismatch):              # rule 5: other fingerprint
        api.aim_ring(mock_face["face"], moved)
    with pytest.raises(rules.GeometryMismatch, match="rule 6"):  # no calibration at that fp
        api.targeted_face_scan(mock_face["face"], moved)
    with pytest.raises(rules.GeometryMismatch):
        api.auto_intensity(mock_face["face"], moved)
    with pytest.raises(ValueError, match="repeats"):
        api.targeted_face_scan(None, p, repeats=0)


def test_endface_code_has_no_bare_sleeps_and_no_usb():
    for f in (ROOT / "engine" / "measure" / "endface").glob("*.py"):
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for n in ast.walk(tree):
            if isinstance(n, ast.Call):
                fn = n.func
                assert not ((isinstance(fn, ast.Attribute) and fn.attr == "sleep")
                            or (isinstance(fn, ast.Name) and fn.id == "sleep")), f.name
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in n.names] + [getattr(n, "module", "") or ""]
                assert not any("dmd_control" in x or x == "hid" or "dlp_driver" in x
                               for x in names), f.name

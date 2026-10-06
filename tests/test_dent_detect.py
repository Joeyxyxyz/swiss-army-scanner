"""The dent detector (engine/measure/dents) through the verbs.

Synthetic shiny cells (tests/dent_synth.py: fixed-radius cylinder, cosine-bowl
dent, diffuse + specular + mirrored-room shading, the production patterns of
both exposures, metric repeats + their stereo calibration) cover the null
calibration, detection, measurement in mm and the refusals (rule 6: no
stereo calibration -> no mm, nothing written; a null made under another
stereo calibration; no matching null calibration; another geometry). Clean
cells - a second one and the null's own good cell - must come out CLEAR.
The CLI and the web job handler are driven without a cell diameter (the
null's). The production capture runs on the MOCK bench only - the real rig
is never opened by these tests."""

import http.client
import json
import shutil
import threading
import time

import numpy as np
import pytest

import dent_synth as S
from conftest import FIXTURE_PRESETS
from engine import api, config, rules
from engine.capture.mock_bench import MockBench
from engine.capture.mock_stereo import MetricCylinder
from engine.library.store import LibraryStore
from engine.measure.dents import dentset as ds_mod
from engine.measure.dents import fusion
from engine.measure.dents import verbs as dent_verbs

DENT = {"depth_mm": 0.3, "size_mm": 3.0, "angle_deg": -10.0, "z_mm": 8.0}
D_MM = 2 * S.RADIUS


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    lib = tmp_path_factory.mktemp("dents") / "library"
    lib.mkdir()
    rg = S.rig()
    cal = S.write_calibration(lib, rg)
    ctx = {"lib": lib, "rg": rg, "cal": cal,
           "good": S.pose(lib, rg, cal, "good", seed=1),
           "bad": S.pose(lib, rg, cal, "bad", dents=[DENT], seed=2),
           "good2": S.pose(lib, rg, cal, "good2", seed=3)}
    ctx["s"] = api.configure(lib, mock=True)
    ctx["null"] = api.calibrate_dent_null([ctx["good"]], D_MM, "synthgood", subset="production")
    yield ctx
    api._session = None


def _use(ctx):
    api._session = ctx["s"]


# ------------------------------------------------------------ patterns + loading

def test_production_patterns_are_dmd_col_and_half_row():
    long_, short = ds_mod.production_specs("long"), ds_mod.production_specs("short")
    assert len(long_) == 2 + 32 and len(short) == 3 + 32 + 10          # sin0 / sin90 x T12 / T36
    assert len(ds_mod.production_specs("short", lines_v=False)) == 35
    assert len(ds_mod.production_specs("long", pattern_set="minimal")) == 2 + 24   # no sin90 T36
    assert len(ds_mod.production_specs("short", pattern_set="minimal")) == 3 + 24 + 10
    assert long_[0] == {"kind": "flood", "level": 255} and long_[1] == {"kind": "flood", "level": 0}
    sin = {"kind": "sinus", "period": 12, "step": 0, "steps": 8}
    s0 = ds_mod.pattern_image({**sin, "theta_deg": 0})
    s90 = ds_mod.pattern_image({**sin, "theta_deg": 90})
    assert s0.shape == (1140, 912) and s0.dtype == np.uint8
    assert np.array_equal(s0[:, 0], s0[:, 12]) and not np.array_equal(s0[:, 0], s0[:, 6])
    assert np.array_equal(s90[0], s90[24]) and not np.array_equal(s90[0], s90[12])   # Y = row / 2
    lv = ds_mod.pattern_image({"kind": "lines_v", "pitch": 10, "shift": 3})
    assert set(np.nonzero(lv[0])[0] % 10) == {3}


def test_legacy_layout_subsets_and_polarity(tmp_path):
    rg = S.rig()
    cal = S.write_calibration(tmp_path / "lib", rg)
    legacy = tmp_path / "legacy_set"
    box = S.crop_box(rg)
    S.write_dentset(tmp_path / "lib", rg, S.cylinder(), box, "x", 5, [], invert=True,
                    legacy_dir=legacy)
    assert cal.id.startswith(S.GEO)
    full = ds_mod.load(legacy, "all")
    prod = ds_mod.load(legacy, "production")
    nol = ds_mod.load(legacy, "production_nolines")
    mini = ds_mod.load(legacy, "minimal")
    assert full.legacy and full.long.frames.shape[0] == 45 and full.short.frames.shape[0] == 45
    assert prod.long.frames.shape[0] == 34 and prod.short.frames.shape[0] == 45
    assert nol.short.frames.shape[0] == 35 and not nol.short.idx(kind="lines_v")
    assert mini.long.frames.shape[0] == 26 and mini.short.frames.shape[0] == 37
    assert not mini.long.idx(kind="sinus", theta_deg=90, period=36)
    assert prod.long.flood(128) is None and prod.short.flood(128) is not None
    # projection inverted: level 0 lands bright, measured from the floods
    assert full.long.on_level == 0 and full.short.on_level == 0 and full.long.inverted
    assert full.pattern_signature() != prod.pattern_signature()
    assert full.ratio == pytest.approx(S.LONG_US / S.SHORT_US)
    # passes recorded with different camera gain: the HDR composite would be wrong -> refused
    m = json.loads((legacy / "dataset.json").read_text())
    m["passes"][0]["gain_db"], m["passes"][1]["gain_db"] = 0.0, 6.0
    (legacy / "dataset.json").write_text(json.dumps(m))
    with pytest.raises(ValueError, match="gain_db"):
        ds_mod.load(legacy, "production")


# ------------------------------------------------------------ fusion units

def _fake_analysis(seed, line=True):
    rng = np.random.default_rng(seed)
    face = np.zeros((60, 50), bool)
    face[5:55, 5:45] = True
    maps = {n: np.where(face, np.abs(rng.normal(0, 1.0, face.shape)), np.nan).astype(np.float32)
            for n in fusion.FINDERS}
    if not line:
        maps["line_kink"] = np.where(face, 0.0, np.nan).astype(np.float32)
    return {"maps": maps, "geo": {"face": face},
            "info": {"line_kink": [{"x_line": 20}] if line else []}}


def test_null_excludes_a_finder_that_saw_nothing_on_the_good_cell():
    """No mirrored room line on the good cell: the line finder is excluded (T None, a
    warning), not divided by a near-zero maximum that would flood later cells."""
    null = fusion.calibrate_null([_fake_analysis(0, line=False)])
    assert null["T"]["line_kink"] is None and "line_kink" not in null["finders"]
    assert any("line_kink" in w and "EXCLUDED" in w for w in null["warnings"])
    assert all(null["T"][n] > 1.0 for n in ("pattern_warp", "photometric", "phase_depth"))
    other = _fake_analysis(1)                                  # this cell does see a line
    E, F = fusion.evidence(other["maps"], other["geo"]["face"], null["T"], dict(fusion.DEFAULTS))
    assert "line_kink" not in E and np.nanmax(F) / null["T_F"] < 2.0
    ok = fusion.calibrate_null([_fake_analysis(0)])
    assert ok["T"]["line_kink"] is not None and not ok["warnings"]
    weak = _fake_analysis(0)
    weak["maps"]["phase_depth"] = weak["maps"]["phase_depth"] * 0.1         # max < min_T
    assert fusion.calibrate_null([weak])["T"]["phase_depth"] is None


def test_peak_refinement_never_turns_an_edge_peak_into_a_detection():
    face = np.zeros((40, 40), bool)
    face[2:38, 2:38] = True
    inner = fusion._inner(face, 3)
    F = np.where(face, 0.0, np.nan)
    F[2:8, 17:24] = 1.40                  # a plateau across the inner / edge boundary
    F[3, 20] = 1.41                       # its NMS pick sits in the edge band
    Gu = np.zeros(face.shape)
    Gu[7, 20] = 5.0                       # the uncapped evidence peaks inside ...
    Gu[3, 21] = 1.0                       # ... and, weaker, in the edge band
    p = dict(fusion.DEFAULTS)
    (y, x, f, at_edge), = fusion.pick_peaks(F, Gu, inner, p)
    assert (y, x) == (3, 21) and at_edge and f == pytest.approx(1.41)
    assert not inner[3, 20] and inner[7, 20] and f > p["thr"]


# ------------------------------------------------------------ null + detection

def test_null_calibration_is_geometry_bound_and_append_only(synth):
    _use(synth)
    n = synth["null"]
    assert n["kind"] == "dent_null" and n["geometry_id"] == S.GEO and n["geometry_fp"] == S.FP
    assert n["cell_diameter_mm"] == pytest.approx(D_MM) and n["subset"] == "production"
    assert set(n["T"]) == {"pattern_warp", "photometric", "phase_depth", "line_kink"}
    assert all(v and v > 0 for v in n["T"].values()) and n["T_F"] > 0
    assert n["finders"] == list(fusion.FINDERS) and not n["warnings"]
    assert n["stereo_calibration_id"] == synth["cal"].id and n["fixture"] == "Z1"
    f = n["path"] / "dent_null.json"
    assert f.is_file() and n["path"].parent.name == S.GEO
    assert n["path"].parent.parent.name == "calibrations"
    assert json.loads(f.read_text())["T_F"] == pytest.approx(n["T_F"])
    assert any(c.get("kind") == "dent_null" for c in api.list_library("calibrations"))


def test_detects_the_synthetic_dent_in_px_cell_coords_and_mm(synth):
    _use(synth)
    rep = api.detect_dents(synth["bad"], cell_diameter_mm=D_MM, subset="production")
    assert rep.verdict == "FLAG" and rep.n_detections >= 1
    assert rep.null_calibration_id == synth["null"]["id"]
    assert rep.thresholds["thr"] == pytest.approx(1.25) and rep.finders == list(fusion.FINDERS)
    top = rep.candidates[0]
    ux, uy = S.pixel_of(synth["rg"], DENT["angle_deg"], DENT["z_mm"])
    assert np.hypot(top["x"] - ux, top["y"] - uy) <= 4.0, (top, ux, uy)
    assert top["level"] == "detection" and top["kind"] == "dent" and top["score"] >= 1.8
    # cell coordinates: theta about the axis (+ toward camera +x), s along it (mm)
    assert top["theta_deg"] == pytest.approx(-DENT["angle_deg"], abs=4.0)
    assert top["s_mm"] == pytest.approx(DENT["z_mm"], abs=1.5)
    s_top = rep.geometry["s_range_mm"][0]
    assert top["s_from_top_mm"] == pytest.approx(top["s_mm"] - s_top, abs=0.02)
    assert "not a cell end" in rep.units["s_mm"].lower()
    assert top["depth_reliable"] and 0.1 <= top["depth_mm"] <= 0.45
    assert top["depth_significant"] and \
        top["depth_mm"] > rep.geometry["depth_significant_above_mm"]
    assert rep.geometry["depth_significant_above_mm"] == pytest.approx(
        1.5 * rep.geometry["depth_noise_p90_mm"])
    assert 0.5 <= top["size_mm"] <= 6.0
    assert set(top["evidence"]) == {"pattern_warp", "photometric", "phase_depth", "line_kink"}
    assert rep.units["depth_mm"].startswith("mm") and rep.units["x"].startswith("px")
    assert rep.geometry["radius_mm"] == pytest.approx(S.RADIUS)
    assert rep.geometry["fit_rms_mm"] < 0.05
    for k in ("json", "png", "npz"):
        assert rep.files[k].is_file(), k
    doc = json.loads(rep.files["json"].read_text())
    assert doc["verdict"] == "FLAG" and doc["candidates"][0]["x"] == top["x"]
    assert doc["thresholds"]["thr"] == pytest.approx(1.25) and doc["fixture"] == "Z1"
    assert rep.files["png"].read_bytes()[:4] == b"\x89PNG"
    with np.load(rep.files["npz"]) as z:
        assert {"fused", "face", "finder_pattern_warp", "evidence_line_kink"} <= set(z.files)
    rows = [r for r in api.list_library("outputs") if r["kind"] == "defects"]
    assert any(r["file"].endswith(rep.files["png"].name) for r in rows)


def test_clean_cells_are_clear(synth):
    """A second clean cell AND the null's own good cell: CLEAR. The null is one cell's
    maximum (the good cell scores exactly 1.0 against itself; another clean cell lands near
    it), so a detection needs the margin thr > 1."""
    _use(synth)
    for key in ("good2", "good"):
        rep = api.detect_dents(synth[key], cell_diameter_mm=D_MM, subset="production")
        assert rep.verdict == "CLEAR" and rep.n_detections == 0, (key, rep.candidates[:3])
        assert rep.max_score is not None and rep.max_score <= rep.thresholds["thr"], key
        assert all(c["level"] != "detection" for c in rep.candidates)
        if key == "good":
            assert rep.max_score == pytest.approx(1.0, abs=1e-6)       # its own null


def test_cli_dent_detect_json_exit_code(synth, capsys):
    from ui import cli
    try:
        code = cli.main(["--mock", "--library", str(synth["lib"]), "--json", "dent-detect",
                         str(synth["bad"]), "--diameter", str(D_MM), "--subset", "production"])
        doc = json.loads(capsys.readouterr().out)
    finally:
        _use(synth)
    assert code == 1 and doc["exit_code"] == 1                  # FLAG
    rep = doc["report"]
    assert rep["verdict"] == "FLAG" and rep["candidates"][0]["kind"] == "dent"
    assert rep["units"]["depth_mm"].startswith("mm")


def test_cli_documented_flow_needs_no_diameter(synth, capsys, monkeypatch):
    """dent-null --diameter once, then dent-detect <set> (the epilog / README / agent.md
    flow): the diameter comes from the matching null. A crash is exit 4, never 1 = FLAG."""
    from ui import cli
    try:
        code = cli.main(["--mock", "--library", str(synth["lib"]), "--json", "dent-detect",
                         str(synth["bad"]), "--subset", "production"])
        doc = json.loads(capsys.readouterr().out)
        assert code == 1 and doc["report"]["verdict"] == "FLAG"
        assert doc["report"]["cell_diameter_mm"] == pytest.approx(D_MM)
        assert doc["report"]["null_calibration_id"] == synth["null"]["id"]

        def crash(*a, **k):
            raise KeyError("a finder bug")
        monkeypatch.setattr(api, "detect_dents", crash)
        code = cli.main(["--mock", "--library", str(synth["lib"]), "--json", "dent-detect",
                         str(synth["bad"])])
        doc = json.loads(capsys.readouterr().out)
        assert code == 4 and doc["exit_code"] == 4 and doc["error"]["type"] == "KeyError"
    finally:
        _use(synth)


def test_web_detect_dents_without_diameter_and_library_ids_only(synth, tmp_path):
    """The web job handler: the documented call {dentset_id, subset} works (diameter from
    the null); server-side paths are refused like everywhere else in the web UI."""
    from ui.web import server as ui_server
    presets = tmp_path / "presets"
    shutil.copytree(FIXTURE_PRESETS, presets)
    httpd, app = ui_server.make_server(mock=True, library=synth["lib"], presets_dir=presets,
                                       port=0)
    t = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
    t.start()

    def job(args):
        conn = http.client.HTTPConnection("127.0.0.1", app.port, timeout=300)
        conn.request("POST", "/api/jobs", body=json.dumps({"verb": "detect_dents", "args": args}),
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
        j = job({"dentset_id": synth["bad"].name, "subset": "production"})
        assert j["status"] == "done", j.get("error")
        assert j["result"]["verdict"] == "FLAG"
        assert j["result"]["cell_diameter_mm"] == pytest.approx(D_MM)
        j = job({"dentset_id": str(synth["bad"]), "subset": "production"})       # a path
        assert j["status"] == "error" and j["error_type"] == "EntryNotFound"
        j = job({"dentset_id": synth["bad"].name, "subset": "production",
                 "null_id": str(synth["null"]["path"] / "dent_null.json")})
        assert j["status"] == "error" and j["error_type"] == "EntryNotFound"
        # a library null id resolves (then the pattern set does not match it: refused fast)
        j = job({"dentset_id": synth["bad"].name, "subset": "production_nolines",
                 "null_id": synth["null"]["id"]})
        assert j["status"] == "error" and j["error_type"] == "ValueError"
        assert "pattern set" in j["error"]
    finally:
        httpd.shutdown()
        httpd.server_close()
        app.close()
        _use(synth)


def test_no_matching_null_calibration_is_refused(synth, tmp_path):
    _use(synth)
    with pytest.raises(ValueError, match="calibrate_dent_null"):
        api.detect_dents(synth["bad"], cell_diameter_mm=30.0, subset="production")
    with pytest.raises(ValueError, match="calibrate_dent_null"):          # other pattern set
        api.detect_dents(synth["bad"], cell_diameter_mm=D_MM, subset="production_nolines")
    with pytest.raises(ValueError, match="21.05 mm cell"):
        api.detect_dents(synth["bad"], cell_diameter_mm=30.0, subset="production",
                         null_calibration=synth["null"]["path"])
    # an explicitly named null of another geometry fingerprint / stereo calibration
    rec = json.loads((synth["null"]["path"] / "dent_null.json").read_text())
    moved = dict(rec, geometry_fp=dict(rec["geometry_fp"], baseline_mm=300.0))
    (tmp_path / "moved.json").write_text(json.dumps(moved))
    with pytest.raises(rules.GeometryMismatch):
        api.detect_dents(synth["bad"], subset="production", null_calibration=tmp_path / "moved.json")
    recal = dict(rec, stereo_calibration_id=f"{S.GEO}/20260102T000000_000000_stereo")
    (tmp_path / "recal.json").write_text(json.dumps(recal))
    with pytest.raises(rules.GeometryMismatch, match="rule 6"):
        api.detect_dents(synth["bad"], subset="production", null_calibration=tmp_path / "recal.json")


def test_named_fixture_null_is_picked_with_its_fixture_and_nulls_are_append_only(synth):
    """A null made with a fixture NAME is a new calibration folder (the first one is
    untouched); detection finds it when told the fixture, and otherwise says that a newer
    null was skipped instead of silently using the older one."""
    _use(synth)
    first = synth["null"]["path"] / "dent_null.json"
    before = first.read_bytes()
    named = api.calibrate_dent_null([synth["good"]], D_MM, "rack", fixture="rackA",
                                    subset="production")
    assert named["path"] != synth["null"]["path"] and named["fixture"] == "rackA"
    assert (named["path"] / "dent_null.json").is_file() and first.read_bytes() == before
    rep = api.detect_dents(synth["bad"], subset="production", fixture="rackA")
    assert rep.null_calibration_id == named["id"] and rep.fixture == "rackA"
    rep = api.detect_dents(synth["bad"], subset="production")               # the zone, Z1
    assert rep.null_calibration_id == synth["null"]["id"]
    assert any("newer dent null" in w and named["id"] in w for w in rep.warnings), rep.warnings


def test_latest_null_rules(tmp_path):
    """Auto-pick: newest match on fingerprint, diameter, fixture, pattern set and stereo
    calibration; without a diameter the matches must agree on one."""
    store = LibraryStore(tmp_path / "lib")
    sig = {"long": {"floods": [0, 255], "families": ["sinus0_T12"]}}
    base = {"kind": "dent_null", "geometry_id": S.GEO, "geometry_fp": dict(S.FP),
            "cell_diameter_mm": 21.05, "fixture": "Z1", "pattern_signature": sig,
            "stereo_calibration_id": "cal1"}

    def put(label, **kw):
        d = store.new_dir("calibrations", S.GEO, label=f"dentnull_{label}")
        store.write_json(d / "dent_null.json", {**base, "id": f"{S.GEO}/{d.name}", **kw})
        return f"{S.GEO}/{d.name}"

    a = put("a")
    b = put("b", fixture="rackB")
    c, w = dent_verbs.latest_null(store, S.GEO, S.FP, None, "Z1", sig, "cal1")
    assert c["id"] == a and w and b in w[0]
    c, w = dent_verbs.latest_null(store, S.GEO, S.FP, None, "rackB", sig, "cal1")
    assert c["id"] == b and not w
    assert dent_verbs.latest_null(store, S.GEO, S.FP, None, "Z1", sig, "cal2") == (None, [])
    assert dent_verbs.latest_null(store, S.GEO, dict(S.FP, baseline_mm=1.0), 21.05, "Z1", sig,
                                  "cal1") == (None, [])
    put("wide", cell_diameter_mm=30.0)
    with pytest.raises(ValueError, match="several cell diameters"):
        dent_verbs.latest_null(store, S.GEO, S.FP, None, "Z1", sig, "cal1")
    c, w = dent_verbs.latest_null(store, S.GEO, S.FP, 21.05, "Z1", sig, "cal1")
    assert c["id"] == a and not any("30 mm" in x for x in w)       # other cell types: no noise


def test_rule6_no_stereo_calibration_no_mm(tmp_path):
    lib = tmp_path / "library"
    lib.mkdir()
    rg = S.rig()
    cal = S.write_calibration(lib, rg)
    good = S.pose(lib, rg, cal, "g", seed=7)
    api.configure(lib, mock=True)
    try:
        api.delete(cal.path, reason="test: recalibrating")         # trashed -> mm stop at once
        with pytest.raises(rules.GeometryMismatch, match="rule 6"):
            api.calibrate_dent_null([good], D_MM)
        with pytest.raises(rules.GeometryMismatch, match="rule 6"):
            api.detect_dents(good, cell_diameter_mm=D_MM, null_calibration={"kind": "dent_null"})
        assert not (lib / "output" / "defects").exists()
        assert not [c for c in api.list_library("calibrations") if c.get("kind") == "dent_null"]
    finally:
        api._session = None


# ------------------------------------------------------------ capture (mock only)

@pytest.fixture
def mock_cell(tmp_path):
    pd = tmp_path / "presets"
    pd.mkdir()
    d = json.loads((FIXTURE_PRESETS / "cell_wall_2p7ft_vstripe_fullrange.json").read_text())
    d["name"] = "dark_fast"
    d["camera"]["average"] = 1
    d["camera"]["exposure_us"] = 66667
    (pd / "dark_fast.json").write_text(json.dumps(d))
    d2 = dict(d, name="moved_fast", rig=dict(d["rig"], baseline_mm=300.0))
    (pd / "moved_fast.json").write_text(json.dumps(d2))
    d3 = dict(d, name="other_geo", geometry_id=d["geometry_id"] + "_b")
    (pd / "other_geo.json").write_text(json.dumps(d3))
    d4 = dict(d, name="gain_fast", camera=dict(d["camera"], exposure_us=16667, gain_db=6.0))
    (pd / "gain_fast.json").write_text(json.dumps(d4))
    s = api.configure(tmp_path / "library", mock=True, bench=MockBench(seed=4), presets_dir=pd)
    p = api.load_preset("dark_fast")
    bench = api.mock_bench()
    bench.use_stereo()
    bench.place(MetricCylinder(center=(0, 0, 780), radius=S.RADIUS, height=60))
    yield s, p, pd
    api._session = None


def _truth_calibration(s, p):
    """The mock rig's own K / R / t stored as the geometry's stereo calibration (what
    calibrate_stereo recovers to ~0.04 px; test_metric_api_mock covers that path)."""
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


def test_mock_capture_to_detection_end_to_end(mock_cell):
    """capture_dent_set (two passes + metric scan, zone-lit) -> null on a good cell ->
    detect_dents on a dented one, all on the MOCK bench."""
    s, p, _ = mock_cell
    _truth_calibration(s, p)
    good = api.capture_dent_set("E5:E7", "good", p, steps=4)       # lit zone, no projector map
    m = json.loads((good.path / "meta.json").read_text())
    assert good.path.parent.parent.name == "dentsets" and good.geometry_id == p.geometry_id
    assert m["kind"] == "dentset" and m["pattern_set"] == "production" and m["zone"] == "E5:E7"
    assert len(good.metric_scan_ids) == 1 and (good.path / "dmd_zone_mask.npy").is_file()
    assert any("projector map" in w for w in good.warnings)
    lp, sp = m["passes"]
    assert lp["name"] == "long" and lp["exposure_us"] == 66667 and len(lp["patterns"]) == 34
    assert sp["name"] == "short" and sp["exposure_us"] == ds_mod.SHORT_US_DEFAULT
    assert lp["gain_db"] == sp["gain_db"] and lp["duty"] == sp["duty"]
    assert len(sp["patterns"]) == 45
    x0, y0, x1, y1 = m["crop_x0y0x1y1"]
    F = np.load(good.path / "frames_long.npy")
    assert F.shape == (34, y1 - y0, x1 - x0) and F.dtype == np.float32
    assert (y1 - y0) * (x1 - x0) < 0.2 * 1080 * 1440              # cropped to the lit zone
    data = ds_mod.load(good.path)
    assert data.long.inverted == p.projection.invert      # polarity measured from the floods
    assert [d.id for d in api.list_library("dentsets")] == [good.id]
    bench = api.mock_bench()
    bench.clear()
    dent = {"depth_mm": 0.5, "size_mm": 4.0, "angle_deg": -10.0, "z_mm": 5.0}
    bench.place(MetricCylinder(center=(0, 0, 780), radius=S.RADIUS, height=60, dents=[dent]))
    bad = api.capture_dent_set("E5:E7", "bad", p, steps=4)
    null = api.calibrate_dent_null([good.id], D_MM, "mockgood")
    assert null["fixture"] == "E5:E7" and null["sources"][0]["metric_scans"] == good.metric_scan_ids
    rep = api.detect_dents(bad)                     # the newest matching null, its diameter
    assert rep.null_calibration_id == null["id"] and rep.metric_scan_ids == bad.metric_scan_ids
    assert rep.cell_diameter_mm == pytest.approx(D_MM)
    top = rep.candidates[0]
    assert top["level"] == "detection" and top["kind"] == "dent"
    assert top["theta_deg"] == pytest.approx(-dent["angle_deg"], abs=4.0)
    assert top["s_mm"] == pytest.approx(dent["z_mm"], abs=1.5)
    assert top["depth_reliable"] and 0.25 <= top["depth_mm"] <= 0.8


def test_capture_dent_set_refuses_before_the_rig(mock_cell, monkeypatch):
    s, p, pd = mock_cell
    from engine.calibrate import flow

    def boom(*a, **k):
        raise AssertionError("the rig must not be driven")
    monkeypatch.setattr(flow, "capture_patterns", boom)
    with pytest.raises(rules.GeometryMismatch, match="rule 6"):          # no stereo calibration
        api.capture_dent_set("all", "nocal", p)
    moved = config.load_preset("moved_fast", presets_dir=pd, mock=True)
    with pytest.raises(rules.GeometryMismatch):              # rule 5: same id, other fingerprint
        api.capture_dent_set(None, "mixed", p, moved, metric=False)
    other = config.load_preset("other_geo", presets_dir=pd, mock=True)
    assert other.geometry_id != p.geometry_id
    with pytest.raises(rules.GeometryMismatch):              # rule 5: another geometry_id
        api.capture_dent_set(None, "mixed", p, other, metric=False)
    gain = config.load_preset("gain_fast", presets_dir=pd, mock=True)
    with pytest.raises(ValueError, match="gain_db"):         # HDR = exposure ratio only
        api.capture_dent_set(None, "gain", p, gain, metric=False)
    with pytest.raises(ValueError, match="longer"):
        api.capture_dent_set(None, "x", p, short_exposure_us=100000, metric=False)
    assert api.list_library("dentsets") == []


def test_failed_metric_scan_keeps_the_captured_frames(mock_cell, monkeypatch):
    s, p, _ = mock_cell
    _truth_calibration(s, p)
    from engine.reconstruct import metric_scan

    def fail(*a, **k):
        raise ValueError("no decodable pixels in the zone")
    monkeypatch.setattr(metric_scan, "scan_metric", fail)
    ds = api.capture_dent_set("E5:E7", "nometric", p, steps=4)
    assert ds.metric_scan_ids == [] and (ds.path / "frames_long.npy").is_file()
    assert (ds.path / "frames_short.npy").is_file() and (ds.path / "meta.json").is_file()
    assert any("metric scan failed" in w and "no decodable" in w for w in ds.warnings)
    m = json.loads((ds.path / "meta.json").read_text())
    assert m["metric_scans"] == [] and "no decodable" in m["metric_error"]
    assert [d.id for d in api.list_library("dentsets")] == [ds.id]

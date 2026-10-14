"""The metric verbs end to end through the engine (mock bench, camera noise,
the normal capture path): refusal without calibration, printable board,
row parity, calibrate_stereo, scan_metric, cylinder_deviation, golden
compare, accuracy_check, predict_precision, metric mesh, and rule 6 after
the calibration is trashed or the geometry changes."""

import json

import pytest

from engine import api, config, rules
from engine.capture.mock_bench import MockBench
from engine.capture.mock_stereo import MetricCylinder
from conftest import FIXTURE_PRESETS

DENTS = [{"depth_mm": 1.0, "size_mm": 8.0, "angle_deg": -15, "z_mm": -20},
         {"depth_mm": 0.3, "size_mm": 6.0, "angle_deg": 15, "z_mm": 20}]


@pytest.fixture(scope="module")
def ctx(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("metric")
    pd = tmp / "presets"
    pd.mkdir()
    d = json.loads((FIXTURE_PRESETS / "cell_wall_2p7ft_vstripe_fullrange.json").read_text())
    d["name"] = "metric_fast"
    d["camera"]["average"] = 1                  # one grab per pattern keeps the test fast
    (pd / "metric_fast.json").write_text(json.dumps(d))
    s = api.configure(tmp / "library", mock=True, bench=MockBench(seed=3), presets_dir=pd)
    p = api.load_preset("metric_fast")
    api.mock_bench().use_stereo()
    out = {"s": s, "p": p}
    yield out
    api._session = None


def _use(ctx):
    api._session = ctx["s"]
    return ctx["p"]


def test_mm_refused_without_a_stereo_calibration(ctx):
    p = _use(ctx)
    with pytest.raises(rules.GeometryMismatch, match="rule 6"):
        api.scan_metric(p)


def test_make_charuco_writes_printable_files(ctx):
    _use(ctx)
    r = api.make_charuco(square_mm=25, dpi=200)
    assert r["pdf"].read_bytes().startswith(b"%PDF") and r["png"].is_file()
    assert json.loads(r["spec_json"].read_text())["square_mm"] == 25
    kinds = {o["kind"] for o in api.list_library("outputs")}
    assert "calibration" in kinds


def test_row_parity_is_measured_and_stored(ctx):
    p = _use(ctx)
    r = api.row_parity(p)
    assert r["ok"] and r["parity"] == 1 and r["shift_cols"] == pytest.approx(0.5, abs=0.05)
    assert (r["path"] / "parity.json").is_file()


def test_calibrate_stereo_with_mock_poses(ctx):
    p = _use(ctx)
    r = api.calibrate_stereo(p, mock_poses=6, steps=4)
    ctx["cal"] = r
    assert r["n_poses_used"] == 6 and r["diamond_parity"] == 1
    assert r["rms"]["camera_px"] < 0.3 and r["rms"]["camera_ok"] and r["rms"]["projector_ok"]
    t = r["vs_truth"]
    assert t["R_deg"] < 0.1 and t["t_mm"] < 1.5
    assert abs(t["cam_f_px"][0]) < 3 and abs(t["proj_f_px"][0]) < 4
    assert r["fv_over_fu"] == pytest.approx(2.0, abs=0.01)
    cals = [c for c in api.list_library("calibrations") if c.get("kind") == "stereo"]
    assert len(cals) == 1 and cals[0]["geometry_id"] == p.geometry_id
    assert len(api.list_library("calib_poses")) == 6


def test_scan_metric_cylinder_dents_in_mm(ctx):
    p = _use(ctx)
    if "cal" not in ctx:
        pytest.skip("needs the calibration test")
    bench = api.mock_bench()
    bench.clear()
    bench.place(MetricCylinder(center=(0, 0, 799), radius=24.0, height=90, dents=DENTS))
    ms = api.scan_metric(p, "cell", steps=6)
    ctx["ms"] = ms
    assert ms.units == "mm" and ms.meta["n_points"] > 100000
    assert ms.meta["consistency"]["median_mm"] < 0.05
    dev = api.cylinder_deviation(ms)
    assert dev.fit.diameter == pytest.approx(48.0, abs=0.05)
    assert dev.units == "mm" and dev.noise_sigma_mm < 0.05
    depths = sorted(d.depth_mm for d in dev.dents)
    assert len(depths) == 2, [vars(d) for d in dev.dents]
    assert depths[0] == pytest.approx(0.3, abs=0.05)
    assert depths[1] == pytest.approx(1.0, abs=0.05)
    assert dev.files["png"].is_file() and dev.files["json"].is_file()
    rep = json.loads(dev.files["json"].read_text())
    assert rep["units"] == "mm" and len(rep["dents"]) == 2


def test_accuracy_check_on_existing_scans(ctx):
    _use(ctx)
    if "ms" not in ctx:
        pytest.skip("needs the scan test")
    r = api.accuracy_check("cylinder", {"diameter_mm": 48.0}, scans=[ctx["ms"]])
    assert r["units"] == "mm" and abs(r["mean_error_mm"]) < 0.05 and r["verdict"] == "GO"
    assert r["file"].is_file()


def test_predict_precision_matches_the_mock_noise(ctx):
    p = _use(ctx)
    if "ms" not in ctx:
        pytest.skip("needs the scan test")
    ms = ctx["ms"]
    pred = api.predict_precision(p, scan=ms, extra_noise_dn=1.5)   # the mock camera's noise
    dev = api.cylinder_deviation(ms)
    assert pred["units"] == "mm" and pred["calibration_id"] == ms.calibration_id
    # the prediction adds shot noise the mock does not draw: it may only be pessimistic
    assert 0.7 < pred["sigma_z_mm"] / dev.noise_sigma_mm < 2.5


def test_compare_metric_to_golden_and_mesh(ctx):
    p = _use(ctx)
    if "ms" not in ctx:
        pytest.skip("needs the scan test")
    bench = api.mock_bench()
    bench.clear()
    bench.place(MetricCylinder(center=(0.3, 0, 799.2), radius=24.0, height=90))   # a good part
    golden = api.scan_metric(p, "golden", steps=4)
    rep = api.compare_metric(ctx["ms"], golden)
    assert rep["icp"]["converged"] and rep["units"] == "mm"
    assert rep["blobs"] and rep["blobs"][0]["depth_mm"] == pytest.approx(1.0, abs=0.1)
    path = api.export_metric_mesh(ctx["ms"], "stl", stride=4, foreground_only=True)
    meta = json.loads(path.with_name(path.name + ".json").read_text())
    assert meta["units"]["xyz"] == "mm" and meta["n_triangles"] > 1000
    assert 30 < meta["bbox_max"][0] - meta["bbox_min"][0] < 49      # the seen arc, not 360


def test_consistency_check_against_what_the_calibration_gave():
    """The rule (engine/reconstruct/metric_scan.py): a two-axis scan warns when its camera and
    projector rays miss each other by more than CONSISTENCY_FACTOR x the calibration's own fit
    residual at the scan's distance, never below CONSISTENCY_FLOOR_MM. The real rig's numbers:
    its calibration's 0.26 px is 0.09 mm at 766 mm; 0.044 mm (end faces) and 0.25 mm (a whole
    board scene) while it fitted - no warning; 1.2 mm after the projector was nudged - the
    warning."""
    from types import SimpleNamespace
    from engine.reconstruct import metric_scan as M
    K = [[2144.5, 0.0, 720.0], [0.0, 2128.7, 540.0], [0.0, 0.0, 1.0]]
    cal = SimpleNamespace(id="rig/20261013_stereo", rms={"stereo_px": 0.2614}, K_cam=K)
    good = M.consistency_check({"median_mm": 0.044}, cal, 766.0)
    assert good["fits"] and good["warning"] is None
    assert good["at_calibration_mm"] == pytest.approx(0.2614 * 766.0 / 2136.6, abs=1e-3)
    assert good["limit_mm"] == pytest.approx(M.CONSISTENCY_FACTOR * good["at_calibration_mm"],
                                             abs=1e-3)
    assert 0.4 < good["limit_mm"] < 0.55
    assert M.consistency_check({"median_mm": 0.25}, cal, 766.0)["fits"]
    bad = M.consistency_check({"median_mm": 1.2}, cal, 766.0)
    assert not bad["fits"] and bad["warning"].endswith(M.CALIBRATION_MAY_NOT_FIT)
    assert "1.20 mm (median)" in bad["warning"] and cal.id in bad["warning"]
    assert M.CALIBRATION_MAY_NOT_FIT == ("the calibration may not fit this rig any more: run "
                                         "the rig check")
    # no fit residual on the record: the floor is the limit
    bare = SimpleNamespace(id="c", rms={}, K_cam=K)
    assert M.consistency_check({"median_mm": 0.29}, bare, 766.0)["fits"]
    floor = M.consistency_check({"median_mm": 0.31}, bare, 766.0)
    assert not floor["fits"] and floor["limit_mm"] == M.CONSISTENCY_FLOOR_MM
    assert "records no fit residual" in floor["warning"]
    # a calibration with a very small residual: the floor keeps a scene's noise from warning
    tiny = SimpleNamespace(id="c", rms={"stereo_px": 0.02}, K_cam=K)
    assert M.consistency_check({"median_mm": 0.25}, tiny, 766.0)["fits"]
    # one decoded axis: nothing to check; a median that is not a number does not fit
    assert M.consistency_check(None, cal, 766.0) is None
    assert not M.consistency_check({"median_mm": float("nan")}, cal, 766.0)["fits"]


def test_a_calibration_that_no_longer_fits_is_said_by_the_two_axis_scan(ctx):
    """On the real rig a calibration that no longer fitted went unnoticed without a rig check:
    every two-axis scan's rays missed each other by ~1.2 mm (0.044 mm while it fitted) and
    nothing was said. Here the scan made with the fitting calibration passes its check; the
    same frames triangulated with a calibration whose projector is tilted 0.3 deg (what a
    nudged projector makes of the old one) warn - in meta, in the result, printed - and the
    mm are still given (a warning, not a refusal)."""
    import contextlib
    import io
    import numpy as np
    from engine.calibrate import stereo
    from engine.capture import rigcheck
    from engine.reconstruct import metric_scan as M
    from ui import cli
    p = _use(ctx)
    if "ms" not in ctx:
        pytest.skip("needs the scan test")
    ms = ctx["ms"]
    chk = ms.meta["consistency_check"]
    assert chk["fits"] and chk["warning"] is None and chk["at_calibration_mm"] > 0
    assert not any(M.CALIBRATION_MAY_NOT_FIT in w for w in ms.meta["warnings"])
    store = ctx["s"].store
    cal = stereo.latest(store, p.geometry_id, config.geometry_fingerprint(p))
    Rt = rigcheck.rotation_from_rotvec(np.radians([0.3, 0.0, 0.0]))
    d = store.new_dir("calibrations", p.geometry_id, label="stereo")
    bad = stereo.StereoCalibration.from_json(
        {**cal.to_json(), "id": f"{p.geometry_id}/{d.name}", "R": (Rt @ cal.R).tolist(),
         "t": (Rt @ np.ravel(cal.t)).tolist()}, d)
    store.write_json(d / "stereo_calibration.json", bad.to_json())
    try:
        r = api.triangulate(ms)
        new = r["scan"]
        assert new.calibration_id == bad.id and new.units == "mm" and r["n_points"] > 100000
        c = new.meta["consistency_check"]
        assert not c["fits"] and c["median_mm"] > c["limit_mm"] > 0
        assert r["warnings"] == new.meta["warnings"]
        assert any(w.endswith(M.CALIBRATION_MAY_NOT_FIT) for w in r["warnings"])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = cli.main(["--mock", "--library", str(store.root), "triangulate", ms.id])
        assert code == cli.EXIT_OK and M.CALIBRATION_MAY_NOT_FIT in out.getvalue()
        assert "  ! the camera and projector rays miss each other by" in out.getvalue()
    finally:
        _use(ctx)
        api.delete(d, reason="test: the calibration that does not fit")
    assert stereo.latest(store, p.geometry_id, config.geometry_fingerprint(p)).id == cal.id


def test_rule6_changed_geometry_and_trashed_calibration(ctx):
    p = _use(ctx)
    if "ms" not in ctx:
        pytest.skip("needs the scan test")
    moved = config.load_preset("metric_fast", presets_dir=ctx["s"].presets_dir, mock=True)
    moved.rig.baseline_mm = 300.0                     # same geometry_id, other fingerprint
    with pytest.raises(rules.GeometryMismatch):
        api.scan_metric(moved)
    api.delete(ctx["cal"]["path"], reason="test: recalibrating")
    with pytest.raises(rules.GeometryMismatch):
        api.cylinder_deviation(ctx["ms"])
    with pytest.raises(rules.GeometryMismatch):
        api.scan_metric(p)


def test_mock_poses_refused_on_the_real_rig(tmp_path):
    api.configure(tmp_path / "lib", mock=False, presets_dir=FIXTURE_PRESETS)
    try:
        from engine.calibrate import flow
        with pytest.raises(rules.RuleViolation, match="mock-only"):
            flow.calibrate_stereo(api._session, config.preset_from_dict(
                json.loads((FIXTURE_PRESETS / "cell_wall_2p7ft_vstripe.json").read_text())),
                mock_poses=3)
    finally:
        api._session = None

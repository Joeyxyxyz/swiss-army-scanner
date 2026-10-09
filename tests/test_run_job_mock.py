"""
`scan run <station config>`: the flow-ready entry point (ui/run_job.py +
ui/cli.py cmd_run). Mock only: one station config in, one NEW run folder out
with result.json (same keys every run) + run.log.jsonl (run_id on every
line); the exit code is the verdict (0 PASS, 1 FAIL, 2 refused, 5 INVALID).
"""

import contextlib
import io
import json
import sys
from pathlib import Path

import pytest

from engine import api
from ui import cli
from ui import run_job as RJ

from conftest import ROOT

FAST = ["--mock", "--average", "1"]        # speed: decode quality is not the subject
EXAMPLE = ROOT / "stations" / "example_cell_station.json"
RESULT_KEYS = {"schema_version", "run_id", "run_folder", "mode", "station",
               "operator_confirmed", "job", "calibration", "capture", "part", "deviation_mm",
               "deviation_units", "threshold_mm", "result", "invalid_reason", "engine_verdict",
               "warnings", "outputs", "software", "started", "finished",
               # schema 0.2: recipe jobs fill these; null / [] for the other job types
               "recipe", "zones", "feedback"}


def _station(tmp_path, name="cfg.json", **over) -> Path:
    doc = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    doc.update({"runs_dir": str(tmp_path / "runs"), "mock_library": None})
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(doc.get(k), dict):
            doc[k] = {**doc[k], **v}
        else:
            doc[k] = v
    p = tmp_path / name
    p.write_text(json.dumps(doc), encoding="utf-8")
    return p


def _cli(argv) -> tuple:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = cli.main(argv)
    api._session = None
    return code, buf.getvalue()


def _only_run(runs: Path) -> Path:
    dirs = [d for d in runs.iterdir() if d.is_dir()]
    assert len(dirs) == 1, dirs
    return dirs[0]


def _check_run(run_dir: Path, station: str) -> dict:
    res = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    assert set(res) == RESULT_KEYS
    assert res["run_id"] == run_dir.name and station in res["run_id"]
    assert res["station"] == station and res["mode"] == "MOCK"
    assert res["result"] in ("PASS", "FAIL", "INVALID")
    assert (res["invalid_reason"] is None) == (res["result"] != "INVALID")
    lines = (run_dir / "run.log.jsonl").read_text(encoding="utf-8").splitlines()
    events = [json.loads(line) for line in lines]
    assert events and all(e["run_id"] == res["run_id"] for e in events)
    assert events[0]["event"] == "run_start" and events[-1]["event"] == "run_end"
    assert events[-1]["result"] == res["result"]
    return res


# ------------------------------------------------------------ config + run folder

def test_example_configs_load():
    cfg = RJ.load_station_config(EXAMPLE)
    assert cfg.station_id == "example_cell_station" and cfg.job_type == "inspect"
    assert cfg.require_mm is True and cfg.base == "latest"
    try:
        import yaml  # noqa: F401
    except ImportError:
        return
    y = RJ.load_station_config(EXAMPLE.with_suffix(".yaml"))
    assert (y.station_id, y.preset, y.job_type) == (cfg.station_id, cfg.preset, cfg.job_type)


@pytest.mark.parametrize("over, msg", [
    ({"station_id": ""}, "station_id"),
    ({"station_id": "bad id/with slash"}, "station_id"),
    ({"job": {"type": "teleport"}}, "job.type"),
    ({"job": {"type": "compare", "golden": None}}, "golden"),
    ({"job": {"base": "sometimes"}}, "job.base"),
    ({"threshold_mm": -1}, "threshold_mm"),
    ({"part": {"role": "maybe"}}, "part.role"),
])
def test_bad_config_refused(tmp_path, over, msg):
    with pytest.raises(RJ.StationConfigError, match=msg):
        RJ.load_station_config(_station(tmp_path, **over))


def test_bad_config_exit_2_no_run_folder(tmp_path):
    code, _ = _cli([*FAST, "--library", str(tmp_path / "lib"), "run",
                    str(_station(tmp_path, job={"type": "teleport"}))])
    assert code == cli.EXIT_REFUSED
    assert not (tmp_path / "runs").exists()
    code, _ = _cli([*FAST, "--library", str(tmp_path / "lib"), "run", str(tmp_path / "nope.json")])
    assert code == cli.EXIT_REFUSED


def test_run_folders_are_new_and_files_never_overwritten(tmp_path):
    cfg = RJ.load_station_config(_station(tmp_path))
    ids = {RJ.new_run(cfg)[0] for _ in range(5)}
    assert len(ids) == 5
    run_id, run_dir = RJ.new_run(cfg)
    log = RJ.RunLog(run_dir / "run.log.jsonl", run_id)
    log.event("x", a=1)
    log.close()
    with pytest.raises(FileExistsError):
        RJ.RunLog(run_dir / "run.log.jsonl", run_id)
    res = RJ.empty_result(cfg, run_id, str(run_dir), True, False)
    assert set(res) == RESULT_KEYS and res["result"] == "INVALID"
    assert res["schema_version"] == "0.2" and res["zones"] == [] and res["recipe"] is None
    with pytest.raises(ValueError):
        RJ.write_result(run_dir, {**res, "result": "MAYBE"})
    RJ.write_result(run_dir, res)
    with pytest.raises(FileExistsError):
        RJ.write_result(run_dir, res)


def test_runs_dir_from_env(tmp_path, monkeypatch):
    monkeypatch.setenv("SCANNER_RUNS_DIR", str(tmp_path / "elsewhere"))
    assert RJ.load_station_config(_station(tmp_path)).runs_dir == tmp_path / "elsewhere"


# ------------------------------------------------------------ the job, mock

def test_real_rig_refused_without_operator_confirmation(tmp_path):
    """No --mock and no confirmation: refused (2) before anything touches the rig,
    and the run folder still says why (INVALID)."""
    cfg = _station(tmp_path)
    code, _ = _cli(["--library", str(tmp_path / "lib"), "run", str(cfg)])
    assert code == cli.EXIT_REFUSED
    res = json.loads((_only_run(tmp_path / "runs") / "result.json").read_text(encoding="utf-8"))
    assert res["mode"] == "REAL" and res["result"] == "INVALID"
    assert "confirm" in res["invalid_reason"]
    assert "PySpin" not in sys.modules and "hid" not in sys.modules


def test_uncalibrated_mock_run_is_invalid(tmp_path):
    """A fresh mock library has no calibration: px only -> INVALID (5), never PASS."""
    cfg = _station(tmp_path)
    code, out = _cli([*FAST, "--library", str(tmp_path / "lib"), "run", str(cfg)])
    assert code == cli.EXIT_INVALID, out
    res = _check_run(_only_run(tmp_path / "runs"), "example_cell_station")
    assert res["result"] == "INVALID" and res["deviation_units"] == "px"
    assert "rule 6" in res["invalid_reason"]
    assert res["calibration"]["calibration_id"] is None
    assert res["part"]["scan_id"] and res["outputs"]["log"] == "run.log.jsonl"


@pytest.fixture(scope="module")
def calibrated_lib(tmp_path_factory):
    """A mock library with a base and a Z/XY calibration from the validation object."""
    lib = tmp_path_factory.mktemp("run_job") / "library"
    for argv in (["base"], ["scan", "cal", "--known", cli.VALIDATION_DIMS],
                 ["calibrate", "cal"]):
        code, out = _cli([*FAST, "--library", str(lib), *argv])
        assert code == 0, out
    return lib


def test_calibrated_mock_run_pass(tmp_path, calibrated_lib):
    cfg = _station(tmp_path)
    code, out = _cli([*FAST, "--library", str(calibrated_lib), "--json", "run", str(cfg)])
    doc = json.loads(out)
    res = _check_run(_only_run(tmp_path / "runs"), "example_cell_station")
    assert doc["result"] == res and doc["exit_code"] == code
    assert res["deviation_units"] == "mm" and res["calibration"]["same_geometry"] is True
    assert res["calibration"]["calibration_id"]
    assert res["result"] == "PASS" and code == cli.EXIT_OK, res
    assert res["deviation_mm"]["method"] == "relief_inspect"
    assert res["threshold_mm"] == pytest.approx(0.7)


def test_calibrated_mock_compare_fail(tmp_path, calibrated_lib):
    """compare job: a shorter box against the validation-object golden -> FAIL (1)."""
    cfg = _station(tmp_path, station_id="cell_station_b", threshold_mm=0.5,
                   job={"type": "compare", "golden": "cal"}, part={"label": "short_box"})
    code, out = _cli([*FAST, "--library", str(calibrated_lib), "--mock-part",
                      "80,56.5,21.9", "run", str(cfg)])
    res = _check_run(_only_run(tmp_path / "runs"), "cell_station_b")
    assert res["result"] == "FAIL" and code == cli.EXIT_NOT_OK, (res, out)
    assert res["deviation_units"] == "mm" and res["deviation_mm"]["method"] == "compare_to_golden"
    assert res["deviation_mm"]["max"] > 0.5 and res["threshold_mm"] == pytest.approx(0.5)
    assert res["part"]["label"] == "short_box" and res["job"]["type"] == "compare"


def test_run_listed_in_help():
    code, out = _cli(["--help"])
    assert code == 0 and "run stations/example_cell_station.json" in out


# ------------------------------------------------------------ metric job (stereo calibration)

METRIC = ROOT / "stations" / "example_metric_check.json"


def _metric_station(tmp_path, name="metric.json", **over) -> Path:
    doc = json.loads(METRIC.read_text(encoding="utf-8"))
    doc.update({"runs_dir": str(tmp_path / "runs"), "mock_library": None})
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(doc.get(k), dict):
            doc[k] = {**doc[k], **v}
        else:
            doc[k] = v
    p = tmp_path / name
    p.write_text(json.dumps(doc), encoding="utf-8")
    return p


@pytest.fixture(scope="module")
def stereo_lib(tmp_path_factory):
    """A mock library with the bench's TRUE stereo calibration at the example preset's
    geometry (calibrate-stereo itself is covered in test_metric_api_mock; here it would
    only cost a minute)."""
    import numpy as np
    from engine import config
    from engine.calibrate.stereo import StereoCalibration
    lib = tmp_path_factory.mktemp("run_metric") / "library"
    s = api.configure(lib, mock=True)
    try:
        p = api.load_preset(RJ.load_station_config(METRIC).preset)
        tr = s.bench.stereo_truth()
        d = s.store.new_dir("calibrations", p.geometry_id, label="stereo")
        cal = StereoCalibration(id=f"{p.geometry_id}/{d.name}", geometry_id=p.geometry_id,
                                K_cam=tr["K_cam"], dist_cam=tr["dist_cam"], K_proj=tr["K_proj"],
                                dist_proj=np.zeros(5), R=tr["R"], t=np.ravel(tr["t"]),
                                diamond_parity=tr["parity"],
                                fingerprint=config.geometry_fingerprint(p), path=d,
                                created="2026-01-01T00:00:00")
        s.store.write_json(d / "stereo_calibration.json", cal.to_json())
    finally:
        api._session = None
    return lib


def test_example_metric_configs_load():
    cfg = RJ.load_station_config(METRIC)
    assert cfg.job_type == "metric" and cfg.scan == "fast" and cfg.axis == "xy"
    assert cfg.measure == "plane" and cfg.threshold_mm == pytest.approx(0.5)
    try:
        import yaml  # noqa: F401
    except ImportError:
        return
    y = RJ.load_station_config(METRIC.with_suffix(".yaml"))
    assert (y.station_id, y.job_type, y.measure, y.threshold_mm) == \
        (cfg.station_id, cfg.job_type, cfg.measure, cfg.threshold_mm)


@pytest.mark.parametrize("job, msg", [
    ({"scan": "slow"}, "job.scan"),
    ({"measure": "vibes"}, "job.measure"),
    ({"measure": "golden", "golden": None}, "golden"),
    ({"roi": [10, 10, 5, 50]}, "job.roi"),
    ({"axis": "y"}, "job.axis"),
])
def test_bad_metric_config_refused(tmp_path, job, msg):
    with pytest.raises(RJ.StationConfigError, match=msg):
        RJ.load_station_config(_metric_station(tmp_path, job=job))


def test_metric_uncalibrated_is_invalid_and_says_how(tmp_path):
    """No stereo calibration: INVALID (5) before the rig, and the reason names the
    calibration commands; never a px number reported as mm."""
    code, out = _cli([*FAST, "--library", str(tmp_path / "lib"), "run",
                      str(_metric_station(tmp_path))])
    assert code == cli.EXIT_INVALID, out
    res = _check_run(_only_run(tmp_path / "runs"), "example_metric_check")
    assert res["job"]["type"] == "metric" and res["result"] == "INVALID"
    assert "stereo calibration" in res["invalid_reason"]
    assert "calibrate-stereo" in res["invalid_reason"] and "calib-pose" in res["invalid_reason"]
    assert res["deviation_units"] is None and res["deviation_mm"]["max"] is None
    assert res["part"]["scan_id"] is None                        # nothing was scanned


def test_metric_missing_threshold_is_invalid(tmp_path, stereo_lib):
    code, out = _cli([*FAST, "--library", str(stereo_lib), "run",
                      str(_metric_station(tmp_path, threshold_mm=None))])
    assert code == cli.EXIT_INVALID, out
    res = _check_run(_only_run(tmp_path / "runs"), "example_metric_check")
    assert "threshold_mm" in res["invalid_reason"] and res["part"]["scan_id"] is None


def test_metric_flat_wall_passes(tmp_path, stereo_lib):
    code, out = _cli([*FAST, "--library", str(stereo_lib), "--mock-part", "none", "--json",
                      "run", str(_metric_station(tmp_path))])
    doc = json.loads(out)
    res = _check_run(_only_run(tmp_path / "runs"), "example_metric_check")
    assert doc["result"] == res and code == cli.EXIT_OK, res
    assert res["result"] == "PASS" and res["deviation_units"] == "mm"
    dm = res["deviation_mm"]
    assert dm["method"] == "plane_fit" and 0 < dm["max"] < 0.5 and dm["rms"] < dm["max"]
    assert dm["n_points"] > 10000
    assert res["calibration"]["calibration_id"] and res["calibration"]["same_geometry"] is True
    assert res["part"]["scan_id"] and res["threshold_mm"] == pytest.approx(0.5)
    run_dir = _only_run(tmp_path / "runs")
    events = [json.loads(x) for x in (run_dir / "run.log.jsonl").read_text().splitlines()]
    meas = [e for e in events if e["event"] == "measure"]
    assert meas and meas[0]["max_deviation_mm"] == dm["max"]
    if res["outputs"]["deviation_map"]:
        assert Path(res["outputs"]["deviation_map"]).is_file()


def test_metric_block_on_the_wall_fails(tmp_path, stereo_lib):
    """A 10 mm block inside the roi: the plane fit (RANSAC: the wall) sees ~10 mm -> FAIL."""
    cfg = _metric_station(tmp_path, station_id="metric_b",
                          job={"roi": [200, 150, 1000, 850]})
    code, out = _cli([*FAST, "--library", str(stereo_lib), "--mock-part", "80,40,10",
                      "run", str(cfg)])
    res = _check_run(_only_run(tmp_path / "runs"), "metric_b")
    assert res["result"] == "FAIL" and code == cli.EXIT_NOT_OK, (res, out)
    assert res["deviation_mm"]["max"] == pytest.approx(10.0, abs=0.5)


# ------------------------------------------------------------ review regressions: config

@pytest.mark.parametrize("over, msg", [
    ({"operator_confirmed": "false"}, "operator_confirmed"),
    ({"operator_confirmed": "no"}, "operator_confirmed"),
    ({"operator_confirmed": 1}, "operator_confirmed"),
    ({"require_mm": "false"}, "require_mm"),
    ({"threshold_mm": float("nan")}, "threshold_mm"),
    ({"threshold_mm": float("inf")}, "threshold_mm"),
    ({"threshold_mm": True}, "threshold_mm"),
    ({"job": {"n_sigma": float("nan")}}, "n_sigma"),
    ({"job": {"average": True}}, "job.average"),
    ({"job": {"type": "compare", "golden": "latest"}}, "job.golden"),
    ({"job": {"type": "compare", "golden": "part"}, "part": {"label": "part"}}, "job.golden"),
])
def test_config_values_are_strict(tmp_path, over, msg):
    """A quoted "false" used to read as true (bool("false")) and pass the real-rig guard;
    NaN / Infinity / booleans in numbers made every compare PASS."""
    with pytest.raises(RJ.StationConfigError, match=msg):
        RJ.load_station_config(_station(tmp_path, **over))


def test_operator_confirmed_string_never_confirms_the_real_rig(tmp_path):
    cfg = _station(tmp_path, operator_confirmed="false")
    code, _ = _cli(["--library", str(tmp_path / "lib"), "run", str(cfg)])
    assert code == cli.EXIT_REFUSED and not (tmp_path / "runs").exists()
    assert RJ.load_station_config(_station(tmp_path, operator_confirmed=None)) \
        .operator_confirmed is False
    assert "PySpin" not in sys.modules and "hid" not in sys.modules


@pytest.mark.parametrize("job, msg", [
    ({"measure": "golden", "golden": "latest"}, "job.golden"),
    ({"measure": "golden", "golden": "LAST"}, "job.golden"),
    ({"measure": "golden", "golden": "flat_panel"}, "job.golden"),      # = part.label
    ({"min_defect_px": 0}, "min_defect_px"),
    ({"min_valid_pct": 101}, "min_valid_pct"),
    ({"min_coverage_pct": float("nan")}, "min_coverage_pct"),
    ({"plane_range_mm": [900, 800]}, "plane_range_mm"),
    ({"periods": [8, float("inf")]}, "job.periods"),
])
def test_bad_metric_config_refused_review(tmp_path, job, msg):
    with pytest.raises(RJ.StationConfigError, match=msg):
        RJ.load_station_config(_metric_station(tmp_path, job=job))


def test_golden_latest_refused_at_run_time_too():
    """Even a config built in code: never the previous run's test scan as the golden."""
    cfg = RJ.StationConfig(station_id="s", golden="latest", part_label="p")
    with pytest.raises(cli.CLIError, match="latest"):
        cli._find_station_golden(cfg, "metric")
    cfg = RJ.StationConfig(station_id="s", golden="p", part_label="p")
    with pytest.raises(cli.CLIError, match="part.label"):
        cli._find_station_golden(cfg, "objects")


# ------------------------------------------------------------ review regressions: verdicts

def test_tolerance_clusters_drop_only_isolated_outliers():
    """No statistic hides a defect: a cluster of min_defect_px px or more FAILs whatever
    its share of the field; only smaller clusters are dropped as stray points."""
    import numpy as np
    dev = np.full((400, 400), 0.05)
    dev[10, 10] = 9.0                                  # one stray triangulated point
    dev[100:103, 200:203] = 2.0                        # 9 px, 0.006 % of the field
    dev[300, 300] = np.nan
    r = cli._tolerance_clusters(dev, 0.5, 5)
    assert r["outlier_px"] == 1 and r["outlier_clusters"] == 1
    assert [d["n_px"] for d in r["defects"]] == [9]
    assert r["robust_max"] == pytest.approx(2.0)
    r = cli._tolerance_clusters(dev, 0.5, 10)          # min_defect_px above the cluster
    assert r["defects"] == [] and r["robust_max"] == pytest.approx(0.05)


def test_foreground_keeps_every_component():
    """Golden compare measures all foreground (two cells), not the largest blob only."""
    import numpy as np
    from types import SimpleNamespace
    yy, xx = np.mgrid[0:200, 0:300].astype(float)
    z = np.full(yy.shape, 800.0)
    z[50:150, 30:90] -= 20.0                           # big part
    z[60:90, 200:230] -= 20.0                          # small second part
    xyz = np.dstack([xx * 0.3, yy * 0.3, z])
    cloud = SimpleNamespace(mask=np.ones(yy.shape, bool), xyz=xyz)
    fg = cli._foreground_all(cloud)
    assert fg[100, 60] and fg[75, 215] and not fg[10, 10]
    assert int(fg.sum()) == 100 * 60 + 30 * 30


def _place(spec: str) -> None:
    from engine.capture.mock_stereo import Block
    bench = api.mock_bench()
    bench.clear()
    bench.use_stereo()
    h, w, l = (float(v) for v in spec.split(","))
    bench.place(Block(center=(0.0, 0.0, 823.0 - l / 2), half=(w / 2, h / 2, l / 2)))


@pytest.fixture(scope="module")
def canned(stereo_lib):
    """Metric scans made ONCE (a mock scan takes ~30 s): a full block, half of it, and a
    small 4x4x3 mm block on the wall. Runs reuse them through a patched api.scan_fast."""
    s = api.configure(stereo_lib, mock=True)
    try:
        p = api.load_preset(RJ.load_station_config(METRIC).preset)
        out = {}
        for key, spec in (("full", "80,40,10"), ("half", "40,40,10"), ("small", "4,4,3")):
            _place(spec)
            out[key] = api.scan_fast(None, p, f"canned_{key}", 4, None, 1, axis="xy")
            out[key].cloud                             # loaded once, reused by every run
        assert s.mock
    finally:
        api._session = None
    return out


def _canned_run(tmp_path, stereo_lib, monkeypatch, scan, argv=(), **over):
    seen = {}

    def fake_scan_fast(zone, p, label, steps, periods, average, **kw):
        seen["average"] = average
        return scan
    monkeypatch.setattr(api, "scan_fast", fake_scan_fast)
    monkeypatch.setattr(cli, "_mock_place_metric", lambda ctx: None)
    tmp_path.mkdir(parents=True, exist_ok=True)
    cfg = _metric_station(tmp_path, **over)
    code, out = _cli([*FAST, "--library", str(stereo_lib), *argv, "run", str(cfg)])
    run_dir = _only_run(tmp_path / "runs")
    res = _check_run(run_dir, json.loads(cfg.read_text())["station_id"])
    events = [json.loads(x) for x in (run_dir / "run.log.jsonl").read_text().splitlines()]
    return code, res, events, seen


def test_metric_small_defect_fails_on_the_whole_field(tmp_path, stereo_lib, canned, monkeypatch):
    """A 4x4 mm, 3 mm bump is ~0.01 % of the field: the 99.9th percentile used to PASS it."""
    code, res, events, _ = _canned_run(tmp_path, stereo_lib, monkeypatch, canned["small"])
    assert res["result"] == "FAIL" and code == cli.EXIT_NOT_OK, res
    dm = res["deviation_mm"]
    assert dm["max"] == pytest.approx(3.0, abs=0.3) and dm["n_defects"] == 1
    big = dm["largest_defect"]
    assert big["n_px"] >= 50 and big["max_mm"] > 2.5 and big["area_mm2"] > 5
    meas = [e for e in events if e["event"] == "measure"][0]
    assert meas["n_defects"] == 1 and "max_percentile" not in meas


def test_metric_plane_needs_coverage_and_the_expected_plane(tmp_path, stereo_lib, canned,
                                                            monkeypatch):
    """An (almost) empty roi, or a plane that is not where the part should be (an empty
    fixture: the wall), is INVALID, never PASS."""
    code, res, _, _ = _canned_run(tmp_path / "a", stereo_lib, monkeypatch, canned["small"],
                                  station_id="cov", job={"min_valid_pct": 100})
    assert res["result"] == "INVALID" and code == cli.EXIT_INVALID
    assert "min_valid_pct" in res["invalid_reason"]
    code, res, _, _ = _canned_run(tmp_path / "b", stereo_lib, monkeypatch, canned["small"],
                                  station_id="rng", job={"plane_range_mm": [700, 800]})
    assert res["result"] == "INVALID" and "plane_range_mm" in res["invalid_reason"]


def _golden_run(tmp_path, stereo_lib, monkeypatch, test_scan, golden_label, **job):
    return _canned_run(tmp_path, stereo_lib, monkeypatch, test_scan, station_id="gold",
                       job={"measure": "golden", "golden": golden_label, **job},
                       part={"label": "unit"})


def test_metric_golden_same_part_passes(tmp_path, stereo_lib, canned, monkeypatch):
    code, res, events, _ = _golden_run(tmp_path, stereo_lib, monkeypatch, canned["full"],
                                       "canned_full")
    assert res["result"] == "PASS" and code == cli.EXIT_OK, res
    meas = [e for e in events if e["event"] == "measure"][0]
    assert meas["golden_found_pct"] == 100.0 and meas["test_matched_pct"] == 100.0
    assert res["outputs"]["report"] and Path(res["outputs"]["report"]).is_file()


def test_metric_golden_missing_material_fails(tmp_path, stereo_lib, canned, monkeypatch):
    """Half the part missing: every test point still sits on the golden (the old one-way
    compare gave PASS, max 3e-05 mm); golden -> test finds the missing half."""
    code, res, events, _ = _golden_run(tmp_path, stereo_lib, monkeypatch, canned["half"],
                                       "canned_full")
    assert res["result"] == "FAIL" and code == cli.EXIT_NOT_OK, res
    big = res["deviation_mm"]["largest_defect"]
    assert big["side"] == "golden" and big["area_mm2"] > 500
    meas = [e for e in events if e["event"] == "measure"][0]
    assert meas["golden_found_pct"] < 60


def test_metric_golden_extra_material_fails(tmp_path, stereo_lib, canned, monkeypatch):
    code, res, _, _ = _golden_run(tmp_path, stereo_lib, monkeypatch, canned["full"],
                                  "canned_half")
    assert res["result"] == "FAIL" and code == cli.EXIT_NOT_OK, res
    assert res["deviation_mm"]["largest_defect"]["side"] == "test"


def test_metric_average_flag_wins_over_job_average(tmp_path, stereo_lib, canned, monkeypatch):
    _, res, _, seen = _canned_run(tmp_path / "a", stereo_lib, monkeypatch, canned["small"],
                                  argv=["--average", "7"], job={"average": 3})
    assert seen["average"] == 7
    _, res, _, seen = _canned_run(tmp_path / "b", stereo_lib, monkeypatch, canned["small"],
                                  argv=["--average", "3"])
    assert seen["average"] == 3


def test_crashed_run_logs_the_process_exit_code(tmp_path, stereo_lib, monkeypatch):
    """run_end.exit_code is what the process exits with: 4 for a crash, never 1 (FAIL)."""
    def boom(*a, **k):
        raise RuntimeError("simulated crash")
    monkeypatch.setattr(cli, "_run_metric", boom)
    cfg = _metric_station(tmp_path, station_id="crash")
    code, _ = _cli([*FAST, "--library", str(stereo_lib), "run", str(cfg)])
    assert code == cli.EXIT_ERROR
    run_dir = _only_run(tmp_path / "runs")
    events = [json.loads(x) for x in (run_dir / "run.log.jsonl").read_text().splitlines()]
    codes = {e["event"]: e.get("exit_code") for e in events}
    assert codes["error"] == codes["run_end"] == cli.EXIT_ERROR


def test_run_uses_the_station_library_selection(tmp_path, monkeypatch):
    """No --preset and preset null: the station library's own selection (preset-select),
    not the one of the library the CLI started on."""
    default_lib, station_lib = tmp_path / "default_lib", tmp_path / "station_lib"
    assert _cli(["--mock", "--library", str(default_lib), "preset-select",
                 "cell_wall_2p7ft"])[0] == 0
    assert _cli(["--mock", "--library", str(station_lib), "preset-select",
                 "cell_wall_2p7ft_vstripe_fullrange"])[0] == 0
    monkeypatch.setattr(api, "DEFAULT_ROOT", default_lib)
    cfg = _metric_station(tmp_path, station_id="sel", preset=None,
                          mock_library=str(station_lib))
    code, _ = _cli(["--mock", "run", str(cfg)])
    res = _check_run(_only_run(tmp_path / "runs"), "sel")
    assert res["capture"]["preset"] == "cell_wall_2p7ft_vstripe_fullrange", res
    # a station library with nothing selected: the CLI default, not the other library's pick
    other = tmp_path / "other_lib"
    (tmp_path / "o").mkdir()
    cfg = _metric_station(tmp_path / "o", station_id="sel2", preset=None,
                          mock_library=str(other))
    _cli(["--mock", "run", str(cfg)])
    res = _check_run(_only_run(tmp_path / "o" / "runs"), "sel2")
    assert res["capture"]["preset"] == cli.DEFAULT_PRESET

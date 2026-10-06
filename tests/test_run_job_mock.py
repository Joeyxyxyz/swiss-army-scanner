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
               "warnings", "outputs", "software", "started", "finished"}


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

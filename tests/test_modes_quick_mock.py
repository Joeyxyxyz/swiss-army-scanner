"""QUICK CHECK mode (engine/modes/quick.py, `scan quick-check`), MOCK only: a triangulation
preset + ONE check -> CONTINUE / REJECT / PULL, no sensors, no permissions.

It is the station job runner on a generated one-job config, so these tests check what the
mode adds - the config it builds (and refuses), the three checks (a flatness limit, a golden
compare, one recipe) ending CONTINUE (exit 0), REJECT (1) and PULL (5: the guard band, no
calibration, a missing golden), the run document (the run schema + operating_mode, the
config kept as quick_check.json and runnable again) - not the measurements themselves
(test_run_job_mock.py, test_recipes_mock.py). Scans are synthetic (tests/recipe_synth.py)
and handed to the runs through a patched api.scan_fast; the mobile / fixed rig gate is in
test_modes_rig_mock.py, the full mock capture path in test_modes_ui.py."""

import contextlib
import io
import json

import pytest

from engine import api
from engine.capture.mock_bench import MockBench
from engine.capture.mock_stereo import Block
from engine.flow import schema as FS
from engine.modes import quick as Q
from ui import cli
from ui import run_job as RJ

import recipe_synth as RS
from test_run_job_mock import RESULT_KEYS

PRESET = "bench_9in_metric"


@pytest.fixture(autouse=True)
def _fresh_engine():
    yield
    api._session = None


# ================================================================== the generated config

def test_the_config_of_a_quick_check_is_a_one_job_station_config(tmp_path):
    doc = Q.build_config("flatness", PRESET, threshold_mm=0.5, roi=[10, 20, 300, 400],
                         guard_band_pct=10, runs_dir=tmp_path / "runs")
    assert doc["operating_mode"] == "quick_check" and doc["station_id"] == "quick_check"
    assert doc["preset"] == PRESET and doc["threshold_mm"] == 0.5
    assert doc["job"] == {"type": "metric", "scan": "fast", "axis": "xy", "zone": None,
                          "measure": "plane", "roi": [10, 20, 300, 400], "golden": None,
                          "guard_band_pct": 10.0}
    assert "io" not in doc and "profiles" not in doc and "flow" not in doc
    # the station job runner reads it as any single job: one implicit work state that
    # expects no sensor and grants no permission
    cfg = RJ.config_from_doc(doc, tmp_path)
    assert (cfg.job_type, cfg.measure, cfg.roi, cfg.guard_band_pct) == (
        "metric", "plane", [10, 20, 300, 400], 10.0)
    assert cfg.source is None and cfg.runs_dir == tmp_path / "runs"
    flow = FS.lift_legacy(doc)
    assert flow.legacy and flow.signals == [] and flow.permissions == []
    assert [(s.id, s.expects, s.permit) for s in flow.flow] == [("run", {}, None)]
    g = Q.build_config("golden", PRESET, threshold_mm=0.3, golden="good_part")
    assert g["job"]["measure"] == "golden" and g["job"]["golden"] == "good_part"
    r = Q.build_config("recipe", recipe="flat_plate")
    assert r["job"] == {"type": "recipe", "recipe": "flat_plate"} and r["preset"] is None
    assert RJ.config_from_doc(r, tmp_path).recipe == "flat_plate"
    assert api.quick_check_config("flatness", PRESET, threshold_mm=1.0)["job"]["roi"] is None


@pytest.mark.parametrize("args, kw, needle", [
    (("teleport", PRESET), {}, "not one of"),
    (("flatness", PRESET), {}, "threshold_mm"),
    (("flatness", None), {"threshold_mm": 0.5}, "needs a triangulation preset"),
    (("flatness", PRESET), {"threshold_mm": -1}, "threshold_mm"),
    (("flatness", PRESET), {"threshold_mm": float("nan")}, "threshold_mm"),
    (("flatness", PRESET), {"threshold_mm": 0.5, "roi": [5, 5, 5, 9]}, "roi must be"),
    (("flatness", PRESET), {"threshold_mm": 0.5, "guard_band_pct": 100}, "guard_band_pct"),
    (("flatness", PRESET), {"threshold_mm": 0.5, "golden": "g"}, "golden is for check golden"),
    (("golden", PRESET), {"threshold_mm": 0.5}, "needs golden"),
    (("recipe", PRESET), {}, "needs recipe"),
    (("recipe", PRESET), {"recipe": "r", "threshold_mm": 0.5}, "not for a recipe check"),
    (("recipe", PRESET), {"recipe": "r", "guard_band_pct": 5}, "the recipe sets its own"),
])
def test_a_bad_quick_check_is_refused_with_every_problem(args, kw, needle):
    with pytest.raises(ValueError, match=needle):
        Q.build_config(*args, **kw)


# ================================================================== runs (synthetic scans)

@pytest.fixture(scope="module")
def lab(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("quick_check")
    s = api.configure(tmp / "library", mock=True, bench=MockBench(seed=1))
    try:
        p = api.load_preset(PRESET)
        s.bench.use_stereo()
        RS.truth_calibration(s, p)
        blk = [Block(center=(0.0, 0.0, 813.0), half=(40.0, 25.0, 10.0))]
        tall = [Block(center=(0.0, 0.0, 812.0), half=(40.0, 25.0, 11.0))]      # 2 mm taller
        out = {"lib": str(s.store.root), "tmp": tmp,
               "wall": RS.oracle_scan(s, p, [], "wall"),
               "box": RS.oracle_scan(s, p, blk, "box", seed=1),
               "good_box": RS.oracle_scan(s, p, blk, "good_box", seed=2),
               "tall": RS.oracle_scan(s, p, tall, "tall", seed=3),
               "plate": RS.oracle_scan(s, p, "scene:plate", "plate", seed=4),
               "bump": RS.oracle_scan(s, p, "scene:plate:bump", "bump", seed=5)}
        for k in ("wall", "box", "good_box", "tall", "plate", "bump"):
            out[k].cloud                            # loaded once, reused by every run
    finally:
        api._session = None
    return out


def _cli(argv) -> tuple:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = cli.main(argv)
    api._session = None
    return code, buf.getvalue()


def _quick(lab, monkeypatch, runs, scan, *argv, preset=PRESET, lib=None) -> tuple:
    """`scan quick-check ...` with the scan the rig would have made -> (exit code, the run's
    result.json, its folder, stdout)."""
    seen = []

    def fake_scan_fast(zone, p, label, steps, periods, average, **kw):
        seen.append(p.name)
        return lab[scan]
    monkeypatch.setattr(api, "scan_fast", fake_scan_fast)
    monkeypatch.setattr(cli, "_mock_place_metric", lambda ctx: None)
    before = set(runs.iterdir()) if runs.is_dir() else set()
    code, out = _cli(["--mock", "--library", lib or lab["lib"], "quick-check",
                      *([preset] if preset else []), *argv, "--runs-dir", str(runs)])
    new = sorted(set(runs.iterdir()) - before) if runs.is_dir() else []
    if not new:
        return code, None, None, out
    assert len(new) == 1 and new[0].name.split("_", 1)[1].startswith("quick_check_")
    return code, json.loads((new[0] / "result.json").read_text(encoding="utf-8")), new[0], out


def test_flatness_limit_continue_reject_and_the_guard_band_pull(lab, monkeypatch, tmp_path):
    runs = tmp_path / "runs"
    code, res, run, out = _quick(lab, monkeypatch, runs, "wall", "--check", "flatness",
                                 "--threshold", "0.5")
    assert code == 0 and res["result"] == "PASS" and res["disposition"] == "CONTINUE"
    assert "disposition: CONTINUE" in out
    # the run document: the run schema + the mode; the generated config is kept and named
    assert set(res) == RESULT_KEYS | {"operating_mode"} and res["operating_mode"] == "quick_check"
    assert res["schema_version"] == "0.4" and res["station"] == "quick_check"
    assert res["job"] == {"type": "metric", "base": "latest", "zone": None,
                          "config": "quick_check.json"}
    kept = json.loads((run / Q.CONFIG_FILE).read_text(encoding="utf-8"))
    assert kept["operating_mode"] == "quick_check" and kept["threshold_mm"] == 0.5
    assert kept["preset"] == PRESET and kept["runs_dir"] == str(runs.resolve())
    # no sensors, no permissions: one implicit work state
    assert res["permissions"] == {} and res["cycle"] is None and len(res["steps"]) == 1
    step = res["steps"][0]
    assert (step["expects"], step["sensor_inputs"], step["permit"], step["outcome"]) == (
        {}, {}, None, "CONFIRMED")
    assert 0.03 < res["deviation_mm"]["max"] < 0.2 and res["threshold_mm"] == 0.5
    events = [json.loads(x) for x in (run / "run.log.jsonl").read_text().splitlines()]
    assert events[0]["operating_mode"] == "quick_check" and events[-1]["exit_code"] == 0
    # REJECT: a block on the plane is a defect cluster far beyond the limit
    code, res, _, out = _quick(lab, monkeypatch, runs, "box", "--check", "flatness",
                               "--limit", "0.5")
    assert code == 1 and res["result"] == "FAIL" and res["disposition"] == "REJECT"
    assert res["deviation_mm"]["n_defects"] >= 1 and res["steps"][0]["outcome"] == "DEFECT"
    # ... but not in an roi beside it
    code, res, _, _ = _quick(lab, monkeypatch, runs, "box", "--check", "flatness",
                             "--threshold", "0.5", "--roi", "100,100,400,400")
    assert code == 0 and res["disposition"] == "CONTINUE"
    assert res["deviation_mm"]["n_points"] == 300 * 300
    # PULL: a PASS inside the guard band of its limit goes to quality inspection
    mx = res["deviation_mm"]["max"]
    thr = f"{mx * 1.5:.4f}"
    code, res, _, out = _quick(lab, monkeypatch, runs, "box", "--check", "flatness",
                               "--threshold", thr, "--roi", "100,100,400,400", "--guard-band",
                               "50")
    assert code == 5 and res["result"] == "PASS" and res["disposition"] == "PULL"
    assert res["pull_reasons"][0]["reason"] == "guard_band" and "PULL" in out
    assert res["steps"][0]["outcome"] == "DOUBT"
    code, res, _, _ = _quick(lab, monkeypatch, runs, "box", "--check", "flatness",
                             "--threshold", thr, "--roi", "100,100,400,400")
    assert code == 0 and res["disposition"] == "CONTINUE"          # no guard band: a pass


def test_the_kept_config_runs_the_same_quick_check_again(lab, monkeypatch, tmp_path):
    runs = tmp_path / "runs"
    code, res, run, _ = _quick(lab, monkeypatch, runs, "wall", "--check", "flatness",
                               "--threshold", "0.5")
    assert code == 0
    code, out = _cli(["--mock", "--library", lab["lib"], "--json", "run",
                      str(run / Q.CONFIG_FILE)])
    doc = json.loads(out)
    assert code == 0 and doc["disposition"] == "CONTINUE"
    again = doc["result"]
    assert again["operating_mode"] == "quick_check" and again["run_id"] != res["run_id"]
    assert again["job"]["config"] == "quick_check.json"
    assert len([d for d in runs.iterdir() if d.is_dir()]) == 2     # next to the first run
    assert not (runs / again["run_id"] / Q.CONFIG_FILE).exists()   # run from its file: not copied


def test_golden_compare_continue_reject_and_a_missing_golden_pull(lab, monkeypatch, tmp_path):
    runs = tmp_path / "runs"
    code, res, _, _ = _quick(lab, monkeypatch, runs, "box", "--check", "golden", "--golden",
                             "good_box", "--threshold", "0.5")
    assert code == 0 and res["disposition"] == "CONTINUE", res["invalid_reason"]
    assert res["deviation_mm"]["method"] == "compare_metric_golden"
    assert res["operating_mode"] == "quick_check"
    code, res, _, _ = _quick(lab, monkeypatch, runs, "tall", "--check", "golden", "--golden",
                             "good_box", "--threshold", "0.5")
    assert code == 1 and res["result"] == "FAIL" and res["disposition"] == "REJECT"
    assert res["deviation_mm"]["max"] == pytest.approx(2.0, abs=0.3)
    # a golden that is not there: refused (2), a PULL in the document - never a pass
    code, res, _, _ = _quick(lab, monkeypatch, runs, "box", "--check", "golden", "--golden",
                             "no_such_scan", "--threshold", "0.5")
    assert code == 2 and res["result"] == "INVALID" and res["disposition"] == "PULL"
    assert res["pull_reasons"][0]["reason"] == "refused"


def test_one_recipe_continue_and_reject(lab, monkeypatch, tmp_path):
    runs = tmp_path / "runs"
    code, res, run, _ = _quick(lab, monkeypatch, runs, "plate", "--check", "recipe", "--recipe",
                               "flat_plate", "--no-feedback", preset=None)
    assert code == 0 and res["result"] == "PASS" and res["disposition"] == "CONTINUE"
    assert res["operating_mode"] == "quick_check" and res["job"]["type"] == "recipe"
    assert res["recipe"]["name"] == "flat_plate" and res["capture"]["preset"] == PRESET
    assert json.loads((run / Q.CONFIG_FILE).read_text())["job"]["recipe"] == "flat_plate"
    assert res["permissions"] == {} and res["steps"][0]["permit"] is None
    code, res, _, out = _quick(lab, monkeypatch, runs, "bump", "--check", "recipe", "--recipe",
                               "flat_plate", "--no-feedback", preset=None)
    assert code == 1 and res["result"] == "FAIL" and res["disposition"] == "REJECT"
    assert any(z["verdict"] == "FAIL" for z in res["zones"]) and "REJECT" in out


def test_quick_check_exit_codes_for_doubt_and_bad_input(lab, monkeypatch, tmp_path):
    runs = tmp_path / "runs"
    # no stereo calibration at the preset's geometry: INVALID -> PULL (5), and it says how
    code, res, _, out = _quick(lab, monkeypatch, runs, "wall", "--check", "flatness",
                               "--threshold", "0.5", lib=str(tmp_path / "empty_library"))
    assert code == 5 and res["result"] == "INVALID" and res["disposition"] == "PULL"
    assert res["pull_reasons"][0]["reason"] == "uncalibrated" and "calibrate-stereo" in out
    n = len(list(runs.iterdir()))
    # bad input: refused (2) before anything runs - no run folder
    for argv in (["--check", "flatness"], ["--check", "golden", "--threshold", "0.5"],
                 ["--check", "recipe"], ["--check", "flatness", "--threshold", "0.5",
                                         "--guard-band", "100"],
                 ["--check", "recipe", "--recipe", "flat_plate", "--threshold", "0.5"]):
        code, res, _, _ = _quick(lab, monkeypatch, runs, "wall", *argv)
        assert code == cli.EXIT_REFUSED and res is None, argv
    assert len(list(runs.iterdir())) == n
    assert _cli(["--mock", "--library", lab["lib"], "quick-check", PRESET])[0] == 2   # no --check
    # --json: the disposition at the top level, as for run / recipe-run
    monkeypatch.setattr(api, "scan_fast", lambda *a, **k: lab["wall"])
    monkeypatch.setattr(cli, "_mock_place_metric", lambda ctx: None)
    code, out = _cli(["--mock", "--library", lab["lib"], "--json", "quick-check", PRESET,
                      "--check", "flatness", "--threshold", "0.5", "--runs-dir", str(runs)])
    doc = json.loads(out)
    assert code == 0 and doc["ok"] and doc["disposition"] == "CONTINUE" and doc["pull_reasons"] == []
    assert doc["result"]["operating_mode"] == "quick_check"

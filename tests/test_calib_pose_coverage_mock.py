"""Calibration poses that map too few corners to the projector, in mock.

Seen on a rig: a printed board at ~0.8 m, the lens opened, the default 8 px fringes - the
projector got a position for 0-11 of ~50 corners per pose, the projector fit collapsed
(f_u ~ 0, stereo RMS 164 px) and the tool reported a calibration. Here the fine fringes
are made unresolved in the simulated capture (fringes under a given period come back with
no modulation), and:
  - calib-pose counts and prints the projector corners, warns under 60 %
  - the default periods are captured again coarser (16,160,1280, then 24,192,1280) unless
    periods were given; the first accepted pose fixes the set's periods; no mixing
  - calibrate-stereo fails loudly: nothing saved, non-zero exit, the poses named
  - the diamond parity is 'measured: no offset' or 'not measured', never confused."""

import contextlib
import io
import json

import numpy as np
import pytest

from engine import api, config
from engine.calibrate import flow, stereo
from engine.calibrate.charuco import BoardSpec
from engine.capture.mock_bench import MockBench
from engine.capture.mock_stereo import CharucoTarget, StereoRig, board_poses
from engine.library.store import LibraryStore
from conftest import FIXTURE_PRESETS
from ui import cli

DEFAULT = [8.0, 80.0, 1280.0]
COARSE = [16.0, 160.0, 1280.0]


def _fringe_period(pattern) -> float:
    """The period (DMD px) of a fringe pattern, from its zero crossings along the coded axis."""
    p = np.asarray(pattern, np.float64)
    row, col = p[0], p[:, 0]
    v = (row if np.ptp(row) >= np.ptp(col) else col) - 127.5
    n = int(np.count_nonzero(np.signbit(v[1:]) != np.signbit(v[:-1])))
    return 2.0 * len(v) / max(n, 1)


@pytest.fixture
def unresolved(monkeypatch):
    """unresolved(below_px): fringes finer than below_px come back with no modulation (a
    defocused projector, ink bleed on a printed board): the flood stays sharp, the fine
    phase decodes nothing. -> the list of fringe periods each capture projected."""
    real = flow.capture_patterns
    state = {"below": 0.0, "captures": []}

    def capture(s, preset, patterns, what, zone_mask=None):
        frames = real(s, preset, patterns, what, zone_mask)
        periods = [_fringe_period(p) for p in patterns[2:]]
        if what == "a calibration pose":
            state["captures"].append(round(min(periods)))
        for i, per in enumerate(periods, start=2):
            if per < state["below"]:
                frames[i] = np.full_like(np.asarray(frames[i], np.float32),
                                         float(np.mean(frames[i])))
        return frames

    monkeypatch.setattr(flow, "capture_patterns", capture)

    def set_below(below_px: float) -> list:
        state["below"] = float(below_px)
        return state["captures"]
    return set_below


def _engine(tmp, seed: int, parity=1):
    pd = tmp / "presets"
    pd.mkdir()
    d = json.loads((FIXTURE_PRESETS / "cell_wall_2p7ft_vstripe_fullrange.json").read_text())
    d["name"] = "metric_fast"
    d["camera"]["average"] = 1
    (pd / "metric_fast.json").write_text(json.dumps(d))
    bench = MockBench(seed=seed)
    bench.stereo_rig = StereoRig.bench(parity=parity)
    s = api.configure(tmp / "library", mock=True, bench=bench, presets_dir=pd)
    p = api.load_preset("metric_fast")
    bench.use_stereo()
    return {"s": s, "p": p, "bench": bench, "poses": board_poses(BoardSpec(), 6, seed=0),
            "tmp": tmp, "pd": pd}


@pytest.fixture(scope="module")
def eng(tmp_path_factory):
    e = _engine(tmp_path_factory.mktemp("cover"), seed=11)
    yield e
    api._session = None


def _use(e):
    api._session = e["s"]
    return e["p"]


def _pose(e, k: int, set_label: str, **kw) -> dict:
    """The simulated operator holds the board at pose k, then one calib-pose."""
    R, t = e["poses"][k]
    e["bench"].clear()
    e["bench"].use_stereo()
    e["bench"].place(CharucoTarget(spec=BoardSpec(), R=R, t=t))
    return api.capture_calib_pose(e["p"], set_label, steps=4, **kw)


def _set(e, set_label: str) -> list:
    return [x for x in api.list_library("calib_poses", e["p"].geometry_id)
            if x.get("set") == set_label]


# ------------------------------------------------------------ the pose: count, retry, one set

def test_low_coverage_at_the_default_periods_is_captured_again_coarser(eng, unresolved):
    """8 px fringes decode nothing on the board, 16 px do: the pose reports the projector
    corners of both tries, stores the 16 px one, and the SET now uses 16,160,1280."""
    _use(eng)
    shots = unresolved(below_px=12)
    r = _pose(eng, 0, "auto")
    assert shots == [8, 16]                                   # captured twice, 24 px not needed
    assert r["corners_found"] >= 40 and r["accepted"]
    assert r["periods_px"] == COARSE and r["periods_source"] == "auto"
    assert r["projector_corner_pct"] >= 60 and r["projector_corners"] >= 0.6 * r["corners_found"]
    tries = r["attempts"]
    assert [t["periods_px"] for t in tries] == [DEFAULT, COARSE]
    assert tries[0]["projector_corners"] == 0 and tries[0]["projector_corner_pct"] == 0.0
    assert not tries[0]["accepted"] and not tries[0]["stored"] and tries[1]["stored"]
    assert any("captured again at 16,160,1280 px" in w and "now uses" in w for w in r["warnings"])
    stored = _set(eng, "auto")
    assert len(stored) == 1 and stored[0]["periods_px"] == COARSE      # the 8 px try: not kept
    assert stored[0]["projector_corners"] == r["projector_corners"] and stored[0]["accepted"]
    # the next pose of the set: the set's periods at once (no 8 px try)
    del shots[:]
    r2 = _pose(eng, 1, "auto")
    assert shots == [16] and r2["periods_px"] == COARSE and r2["periods_source"] == "set"
    assert r2["accepted"] and len(r2["attempts"]) == 1
    # other periods in this set would mix two period sets in one calibration: refused
    # before anything is captured
    del shots[:]
    with pytest.raises(ValueError, match="mix two period sets"):
        _pose(eng, 2, "auto", periods=[8, 80, 1280])
    assert shots == [] and len(_set(eng, "auto")) == 2
    assert _pose(eng, 2, "auto", periods=[16, 160, 1280])["periods_source"] == "argument"


def test_explicit_periods_are_never_retried_and_warn(eng, unresolved):
    _use(eng)
    shots = unresolved(below_px=12)
    r = _pose(eng, 0, "explicit", periods=[8, 80, 1280])
    assert shots == [8] and r["periods_px"] == DEFAULT and r["periods_source"] == "argument"
    assert r["corners_found"] >= 40 and r["projector_corners"] == 0 and not r["accepted"]
    w = " | ".join(r["warnings"])
    assert "only 0 of" in w and "got a projector position" in w and "< 60 %" in w
    assert "try coarser periods (16,160,1280 or 24,192,1280)" in w
    assert len(_set(eng, "explicit")) == 1                    # stored (append-only), flagged
    assert not _set(eng, "explicit")[0]["accepted"]


def test_when_no_period_set_resolves_the_pose_does_not_fix_the_set(eng, unresolved):
    """Every rung fails (the fine level is dead at 8, 16 and 24 px): all three are tried,
    the best is stored as NOT accepted, and the set's periods stay open."""
    _use(eng)
    shots = unresolved(below_px=40)
    r = _pose(eng, 0, "dead")
    assert shots == [8, 16, 24] and len(r["attempts"]) == 3 and not r["accepted"]
    assert sum(t["stored"] for t in r["attempts"]) == 1
    w = " | ".join(r["warnings"])
    assert "tried 8,80,1280 px: 0/" in w and "24,192,1280 px: 0/" in w
    assert "does not fix the set's periods" in w
    p = eng["p"]
    poses = flow.list_poses(eng["s"].store, p.geometry_id, config.geometry_fingerprint(p), "dead")
    assert len(poses) == 1 and flow.set_periods(poses) is None
    # a board that is not found at all is not retried: coarser fringes can't find a board
    del shots[:]
    eng["bench"].clear()
    eng["bench"].use_stereo()
    r = api.capture_calib_pose(p, "empty", steps=4)
    assert shots == [8] and r["corners_found"] == 0 and not r["accepted"]
    assert any("ChArUco corners found" in x for x in r["warnings"])
    assert not any("projector position" in x for x in r["warnings"])


def test_calibration_from_the_coarser_set_names_its_poses(eng, unresolved):
    """The set captured at 16,160,1280 calibrates: every pose listed with its projector
    corners, the periods recorded, the calibration close to the rig's truth."""
    p = _use(eng)
    unresolved(below_px=12)
    if len(_set(eng, "auto")) < 3:
        pytest.skip("needs the retry test")
    _pose(eng, 3, "auto")
    r = api.calibrate_stereo(p, set_label="auto")
    ids = [x["id"] for x in _set(eng, "auto")]
    assert r["periods_px"] == COARSE and r["n_poses_used"] == len(ids) == 4
    assert [row["id"] for row in r["per_pose"]] == ids and r["projector_poses"] == ids
    assert all(row["contributed"] and row["projector_corner_pct"] >= 60 for row in r["per_pose"])
    assert r["rms"]["stereo_px"] < stereo.MAX_STEREO_RMS_PX and not r["poses_skipped"]
    assert r["vs_truth"]["R_deg"] < 0.2 and abs(r["fv_over_fu"] - 2.0) < 0.05
    rep = json.loads((r["path"] / "report.json").read_text())
    assert rep["periods_px"] == COARSE and rep["per_pose"][0]["id"] == ids[0]
    eng["cal"] = r


# ------------------------------------------------------------ calibrate-stereo fails loudly

def _cli(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(argv)
    api._session = None
    return code, out.getvalue(), err.getvalue()


def test_calibrate_stereo_fails_loudly_when_the_projector_gets_no_points(tmp_path, unresolved):
    """The rig's failure, end to end on the command line: poses at 8 px fringes that decode
    nothing on the board -> REJECTED, nothing saved, exit 1, the poses named."""
    unresolved(below_px=12)
    lib = tmp_path / "lib"
    base = ["--mock", "--average", "1", "--library", str(lib)]
    code, out, err = _cli([*base, "calibrate-stereo", "--mock-poses", "3", "--steps", "4",
                           "--periods", "8,80,1280", "--set", "collapse"])
    assert code == cli.EXIT_NOT_OK, (out, err)
    assert "REJECTED - NOTHING SAVED" in out and "rejected" in err
    assert "only 0 of the 3 poses" in out and "projector points from 0 of 3 poses" in out
    assert "coarser fringe periods (16,160,1280 or 24,192,1280" in out
    assert "Stereo calibration mock" not in out and "camera RMS" not in out
    s = api.configure(lib, mock=True)
    try:
        p = api.load_preset(cli.DEFAULT_PRESET)
        assert not [c for c in api.list_library("calibrations") if c.get("kind") == "stereo"]
        assert stereo.latest(s.store, p.geometry_id, config.geometry_fingerprint(p)) is None
        poses = [x for x in api.list_library("calib_poses") if x["set"] == "collapse"]
        assert len(poses) == 3 and all(x["projector_corners"] == 0 for x in poses)
        assert not any(x["accepted"] for x in poses)
        # the rejection is on record in the output bin - as a report, not as a calibration
        reps = [o for o in api.list_library("outputs", "calibration") if "REJECTED" in o["file"]]
        assert len(reps) == 1
        doc = json.loads((lib / reps[0]["file"]).read_text())
        assert doc["rejected"] and not doc["saved"] and len(doc["per_pose"]) == 3
        assert [row["id"] for row in doc["per_pose"]] == [x["id"] for x in poses]
        assert all(row["projector_corners"] == 0 and not row["contributed"]
                   for row in doc["per_pose"])
        # the engine verb: an exception with the reasons and the poses, nothing saved
        with pytest.raises(stereo.CalibrationRejected) as ei:
            api.calibrate_stereo(p, set_label="collapse")
        assert ei.value.reasons and len(ei.value.per_pose) == 3
        assert poses[0]["id"] in str(ei.value) and "nothing saved" in str(ei.value)
        assert stereo.latest(s.store, p.geometry_id, config.geometry_fingerprint(p)) is None
    finally:
        api._session = None
    code, out, _ = _cli([*base, "--json", "calibrate-stereo", "--set", "collapse"])
    doc = json.loads(out)
    assert code == cli.EXIT_NOT_OK and doc["ok"] is False and doc["rejected"] is True
    assert doc["calibration_id"] is None and doc["saved"] is False and len(doc["per_pose"]) == 3


def test_cli_calib_pose_prints_the_projector_corners_and_the_retry(tmp_path, unresolved):
    """`scan calib-pose` says how many corners got a projector position, at which fringe
    periods, and shows the coarser try it made; `calib-poses` lists the stored pose."""
    from engine.calibrate import fixture as fx
    unresolved(below_px=12)
    spec = tmp_path / "fixture_spec.json"
    spec.write_text(json.dumps(fx.design()), encoding="utf-8")
    base = ["--mock", "--average", "1", "--library", str(tmp_path / "lib")]
    code, out, err = _cli([*base, "calib-pose", "--fixture", str(spec), "--set", "fx",
                           "--steps", "4"])
    assert code == cli.EXIT_OK, (out, err)
    line = next(x for x in out.splitlines() if "projector corners" in x and "fringe periods" in x)
    assert "at fringe periods 16,160,1280 px" in line and "retried automatically" in line
    assert "NOT accepted" not in line
    k, n = (int(v) for v in line.split("projector corners")[1].split()[0].split("/"))
    assert n >= 60 and k >= 0.6 * n                            # k / n, as printed
    tried = [x for x in out.splitlines() if x.strip().startswith(("8,80,1280", "16,160,1280"))]
    assert len(tried) == 2 and " 0 " in tried[0] + " " and tried[1].rstrip().endswith("stored")
    assert "captured again at 16,160,1280 px" in out
    code, out, _ = _cli([*base, "calib-poses", "--set", "fx"])
    assert code == cli.EXIT_OK and "Calibration poses (1) of set fx" in out
    assert "16,160,1280" in out and "yes" in out
    # explicit periods on another set: used as they are, the low count printed, exit 1
    code, out, _ = _cli([*base, "calib-pose", "--fixture", str(spec), "--set", "fx8",
                         "--steps", "4", "--periods", "8,80,1280"])
    assert code == cli.EXIT_NOT_OK
    assert "at fringe periods 8,80,1280 px (--periods)  - NOT accepted" in out
    assert "projector corners 0/" in out and "< 60 %" in out


def _fit(fu=1200.0, fv=2400.0, rms=0.4, cam_f=2450.0) -> dict:
    """A fit dict as stereo.calibrate returns it (what stereo.save / fit_problems read)."""
    return {"cam": {"K": np.array([[cam_f, 0, 720.0], [0, cam_f, 540.0], [0, 0, 1.0]]),
                    "dist": np.zeros(5), "rms": 0.2, "ok": True, "per_pose_rms": [0.2],
                    "image_size": (1440, 1080)},
            "proj": {"K": np.array([[fu, 0, 456.0], [0, fv, 570.0], [0, 0, 1.0]]),
                     "dist": np.zeros(5), "rms": 0.3, "rms_u": 0.2, "rms_v": 0.3, "ok": True,
                     "per_pose_rms": [0.3], "fv_over_fu": fv / fu if fu else float("inf"),
                     "image_size": (912, 1140)},
            "R": np.eye(3), "t": np.array([-270.0, 0.0, 90.0]), "rms_stereo": rms,
            "per_pose": [{"pose": 0, "id": "p0", "corners": 50, "projector_corners": 50,
                          "contributed": True}], "n_poses": 4}


@pytest.mark.parametrize("kw, why", [
    ({"fu": 0.0}, "degenerate projector intrinsics"),          # the rig's collapse: f_u ~ 0
    ({"fu": -900.0}, "degenerate projector intrinsics"),
    ({"fv": float("nan")}, "degenerate projector intrinsics"),
    ({"fu": 3.0, "fv": 6.0}, "absurd projector focal length"),
    ({"fu": 1200.0, "fv": 1650.0}, "f_v / f_u"),               # neither ~1 nor ~2
    ({"fu": 700.0, "fv": 2400.0}, "f_v / f_u"),
    ({"rms": 164.0}, "stereo reprojection RMS 164.00 px > 2 px"),
    ({"rms": float("nan")}, "stereo reprojection RMS"),
    ({"cam_f": 0.0}, "degenerate camera intrinsics"),
])
def test_a_degenerate_fit_is_never_saved_as_a_calibration(tmp_path, kw, why):
    fit = _fit(**kw)
    bad = stereo.fit_problems(fit)
    assert bad and any(why in b for b in bad), bad
    store = LibraryStore(tmp_path / "lib")
    with pytest.raises(stereo.CalibrationRejected, match="nothing saved") as ei:
        stereo.save(store, "geo", fit, fingerprint=None, spec=BoardSpec(), parity=None,
                    created="2026-01-01T00:00:00")
    assert ei.value.per_pose[0]["id"] == "p0" and "p0 50/50" in str(ei.value)
    assert not (tmp_path / "lib" / "calibrations").exists()    # not even an empty entry


def test_a_sane_fit_passes_the_gates(tmp_path):
    assert stereo.fit_problems(_fit()) == []
    assert stereo.fit_problems(_fit(rms=1.9, fu=1100.0, fv=2500.0)) == []
    assert stereo.fit_problems(_fit(fu=1500.0, fv=1480.0)) == []    # a square-pixel projector
    cal = stereo.save(LibraryStore(tmp_path / "lib"), "geo", _fit(), fingerprint=None,
                      spec=BoardSpec(), parity=None, created="2026-01-01T00:00:00",
                      parity_source="row_parity", parity_info={"shift_cols": 0.01})
    doc = json.loads((cal.path / "stereo_calibration.json").read_text())
    assert doc["diamond_parity"] is None and doc["diamond_parity_source"] == "row_parity"
    assert doc["diamond_parity_measured"] is True and doc["diamond_parity_info"]["shift_cols"] == 0.01
    back = stereo.StereoCalibration.from_json(doc, cal.path)
    assert back.parity_measured is True and back.rehomed is None
    old = stereo.StereoCalibration.from_json({k: v for k, v in doc.items()
                                              if not k.startswith("diamond_parity_")})
    assert old.parity_measured is None and old.parity_source is None    # not recorded then


# ------------------------------------------------------------ one period set per pose set

def _m(pid, periods, accepted=True, **kw):
    return {"id": pid, "periods_px": periods, "accepted": accepted, "corners_found": 50, **kw}


def test_a_calibration_takes_one_period_set():
    a, b, c = _m("a", DEFAULT, accepted=False), _m("b", COARSE), _m("c", COARSE)
    assert flow.set_periods([a, b, c]) == COARSE              # the first ACCEPTED pose
    keep, skipped, per = flow._one_period_set([a, b, c], "s")
    assert [p["id"] for p in keep] == ["b", "c"] and per == COARSE
    assert skipped[0]["id"] == "a" and "8,80,1280 px and not accepted" in skipped[0]["why"]
    # two ACCEPTED poses at different periods: refused, with what to do
    with pytest.raises(ValueError, match="mixes fringe periods.*scan delete d"):
        flow._one_period_set([b, _m("d", DEFAULT)], "s")
    # nothing accepted and mixed: refused; nothing accepted, one period set: used as it is
    with pytest.raises(ValueError, match="none of its poses was accepted"):
        flow._one_period_set([a, _m("e", COARSE, accepted=False)], "s")
    keep, skipped, per = flow._one_period_set([a, _m("f", DEFAULT, accepted=False)], "s")
    assert len(keep) == 2 and not skipped and per == DEFAULT
    # poses stored before the count existed: accepted when their board was usable
    old = [{"id": "o1", "periods_px": DEFAULT, "corners_found": 3},
           {"id": "o2", "periods_px": DEFAULT, "corners_found": 48}]
    assert not flow.pose_accepted(old[0]) and flow.pose_accepted(old[1])
    assert flow.pose_accepted({"fixture": {"name": "fx"}, "faces_seen": [0, 2]})
    assert not flow.pose_accepted({"fixture": {"name": "fx"}, "faces_seen": [1]})
    assert flow.set_periods(old) == DEFAULT and flow.set_periods([]) is None


# ------------------------------------------------------------ parity: measured vs not measured

def test_parity_words():
    assert "measured" in stereo.parity_text(None, "row_parity")
    assert "square grid" in stereo.parity_text(None, "row_parity")
    assert stereo.parity_text(1, "row_parity").startswith("1 (odd rows")
    assert "NOT measured" in stereo.parity_text(None, "none")
    assert "set by the preset" in stereo.parity_text(0, "preset")
    assert "not recorded" in stereo.parity_text(None, None)


def test_measured_no_offset_is_not_reported_as_missing(tmp_path, capsys):
    """A DMD measured as a square grid (parity none, no half-column offset): the
    calibration used to print 'calibrated without a diamond row parity' as if it had never
    been measured. Not measured -> a warning; measured 'none' -> a result, stored with the
    calibration."""
    e = _engine(tmp_path, seed=13, parity=None)             # the rig: a square-grid DMD
    try:
        p = e["p"]
        r = api.calibrate_stereo(p, mock_poses=4, steps=4, set_label="nopar")
        assert r["diamond_parity"] is None and r["diamond_parity_source"] == "none"
        assert r["diamond_parity_measured"] is False
        assert "NOT measured" in r["diamond_parity_text"]
        assert any("row parity was not measured" in w and "scan parity" in w
                   for w in r["warnings"])
        doc = json.loads((r["path"] / "stereo_calibration.json").read_text())
        assert doc["diamond_parity_source"] == "none" and doc["diamond_parity_measured"] is False
        assert all(x["parity_source"] == "none" for x in api.list_library("calib_poses"))
        # now measure it: no half-column offset (parity none, measured)
        par = api.row_parity(p)
        assert par["ok"] and par["parity"] is None and abs(par["shift_cols"]) < 0.15
        r = api.calibrate_stereo(p, set_label="nopar")       # the same poses, same decode
        assert r["diamond_parity"] is None and r["diamond_parity_source"] == "row_parity"
        assert r["diamond_parity_measured"] is True
        assert r["diamond_parity_text"].startswith("none (no half-column offset: square grid)")
        assert "measured" in r["diamond_parity_text"]
        assert not any("parity" in w for w in r["warnings"]), r["warnings"]
        doc = json.loads((r["path"] / "stereo_calibration.json").read_text())
        assert doc["diamond_parity"] is None and doc["diamond_parity_measured"] is True
        assert doc["diamond_parity_info"]["shift_cols"] == par["shift_cols"]
        rep = json.loads((r["path"] / "report.json").read_text())
        assert rep["diamond_parity_source"] == "row_parity" and rep["diamond_parity_measured"]
        cal = stereo.latest(e["s"].store, p.geometry_id, config.geometry_fingerprint(p))
        assert cal.parity_measured is True and cal.diamond_parity is None
        # a pose captured after the measurement says where its parity comes from
        R, t = e["poses"][0]
        e["bench"].clear()
        e["bench"].use_stereo()
        e["bench"].place(CharucoTarget(spec=BoardSpec(), R=R, t=t))
        pose = api.capture_calib_pose(p, "after", steps=4)
        assert pose["diamond_parity"] is None and pose["parity_source"] == "row_parity"
        assert not any("parity" in w for w in pose["warnings"])
    finally:
        api._session = None
    # the command line says the same (the default preset is the same geometry + settings)
    lib = ["--mock", "--average", "1", "--library", str(tmp_path / "library")]
    code, out, err = _cli([*lib, "calibrate-stereo", "--set", "nopar"])
    assert "diamond parity" in out and "square grid) - measured" in out, (out, err)
    assert "without a diamond row parity" not in out and "not measured" not in out
    assert "projector points from 4 of 4 poses" in out and "fringe periods" in out

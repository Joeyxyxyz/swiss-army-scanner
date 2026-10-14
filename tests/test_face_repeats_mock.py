"""A face-scan repeat that keeps far fewer points than its siblings is named, MOCK only.

The bench (2026-10-13): one of three face-scan repeats kept half its points and took longer,
and neither the face scan nor the face profile said so. Now the face scan names such a
repeat with its face points and the other repeats' median (its warnings, its stored meta
`low_repeats`, the per-repeat table and the JSON of `face-scan`), and the profile counts
what each repeat gave it (stats repeat_face_points, repeats_used, all_repeats_pct,
low_repeats; the `repeats` line of `face-profile`), with its own warning - also for a scan
made before, or plain metric scans. Repeats that agree say nothing. The judgment of a face
check is not changed by it.

The mock bench (stereo mode + EndFace) drives the capture path; the disturbed repeat is the
real capture with a third of the face washed out (no fringe contrast there: a glint or a
shadow), so it decodes about two thirds of the face: more than half, which the face scan's
old rule let pass in silence. The synthetic placements
(tests/face_synth.py) cover the profile offline. The real rig is never opened."""

import json
from pathlib import Path

import numpy as np
import pytest

import face_synth as S
from conftest import FIXTURE_PRESETS
from engine import api, config
from engine.capture.mock_bench import MockBench
from engine.capture.mock_stereo import EndFace, MetricCylinder
from engine.measure.endface import profile as profile_mod
from engine.measure.endface import verbs as face_verbs
from ui import cli

FACE_C = np.array([-8.0, 6.0, 780.0])
FACE_N = S.normal_from_tilt(10.4, 7.0)
SCAN = "a targeted end-face scan"           # what _capture is told for a repeat's frames


def _place(bench):
    bench.clear()
    bench.place(EndFace(center=tuple(FACE_C), normal=tuple(FACE_N), radius=S.RADIUS,
                        chamfer_mm=0.6, chamfer_depth_mm=0.15))
    bench.place(MetricCylinder(center=tuple(FACE_C - FACE_N * 15), radius=S.RADIUS, height=30,
                               axis=tuple(FACE_N)))


def _cut(face) -> float:
    """The camera column from which a disturbed repeat sees nothing: 0.2 face radius right of
    the centre, so it loses about 37 % of the face - still more than half decoded, which the
    face scan's old rule (under 50 % decoded) let pass in silence."""
    e = face.camera_ellipse
    return e["cx"] + 0.2 * e["a"] / 2


def _disturb(monkeypatch, cx: float, which: int = 2) -> list:
    """The `which`-th repeat's frames lose the face right of camera column cx (every frame
    flat there: no fringe). -> the repeats captured so far ([n])."""
    real = face_verbs._capture
    seen = [0]

    def capture(s, preset, pats, what, zone_mask=None, exposure_us=None):
        frames = real(s, preset, pats, what, zone_mask, exposure_us)
        if what != SCAN:
            return frames
        seen[0] += 1
        if seen[0] != which:
            return frames
        out = []
        for f in frames:
            g = np.array(f, copy=True)
            g[:, int(cx):] = g[:, int(cx):].mean()
            out.append(g)
        return out
    monkeypatch.setattr(face_verbs, "_capture", capture)
    return seen


@pytest.fixture(scope="module")
def rig(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("facerepeats")
    pd = tmp / "presets"
    pd.mkdir()
    d = json.loads((FIXTURE_PRESETS / "cell_wall_2p7ft_vstripe_fullrange.json").read_text())
    d.update(name="face_fast")
    d["camera"].update(average=1, exposure_us=16667)
    d["projection"].update(invert=False, duty=0.6)
    (pd / "face_fast.json").write_text(json.dumps(d))
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
    face = api.locate_face(api.scan_metric(p, "locate", method="plane", steps=4), S.DIAMETER,
                           "repface")
    mp = pytest.MonkeyPatch()
    try:
        clean = api.targeted_face_scan(face, p, "clean", repeats=3, steps=4, polar=False,
                                       auto_exposure=False)
        _disturb(mp, _cut(face))
        hit = api.targeted_face_scan(face, p, "hit", repeats=3, steps=4, polar=False,
                                     auto_exposure=False)
    finally:
        mp.undo()
    yield {"s": s, "p": p, "face": face, "clean": clean, "hit": hit, "lib": tmp / "library"}
    api._session = None


def _use(rig):
    api._session = rig["s"]
    rig["s"].preset = rig["p"]
    return rig["p"]


def test_which_repeat_kept_far_fewer_points():
    lr = profile_mod.low_repeats
    assert lr([]) == [] and lr([5000]) == [] and lr([0, 0]) == []
    assert lr([9650, 9700, 9600]) == []                             # repeats that agree
    assert lr([9650, 4812, 9701]) == [{"repeat": 2, "points": 4812, "others_median": 9675.5,
                                       "pct_of_others": 49.7}]
    assert [x["repeat"] for x in lr([100, 40])] == [2]              # two: the lower one
    assert [x["repeat"] for x in lr([100, 40, 40])] == [2, 3]
    assert lr([100, 76, 100]) == [] and [x["repeat"] for x in lr([100, 74, 100])] == [2]
    assert profile_mod.LOW_REPEAT_FRACTION == 0.75


def test_a_repeat_that_kept_far_fewer_points_is_named_by_the_face_scan(rig):
    """3 repeats, the 2nd washed out on a third of the face: the face scan names it with both
    counts, in its warnings and in the meta it stores; the other two are not named. More
    than half the face decoded, so the old rule alone said nothing (as on the bench)."""
    _use(rig)
    fs = rig["hit"]
    pts = [r["face_points"] for r in fs.repeats]
    assert pts[1] < 0.75 * min(pts[0], pts[2]), pts
    assert fs.repeats[1]["face_decoded_pct"] > 50
    assert not [x for x in fs.warnings if x.startswith("repeat 2:")]   # the old rule: silent
    low = fs.meta["low_repeats"]
    assert [lo["repeat"] for lo in low] == [2] and low[0]["metric_scan"] == fs.metric_scan_ids[1]
    assert low[0]["points"] == pts[1] and low[0]["others_median"] == pytest.approx(
        float(np.median([pts[0], pts[2]])))
    assert low[0]["capture_s"] == fs.repeats[1]["capture_s"] and low[0]["others_capture_s"]
    w = [x for x in fs.warnings if x.startswith("repeat 2 ")]
    assert len(w) == 1 and not [x for x in fs.warnings if x.startswith(("repeat 1 ", "repeat 3 "))]
    assert f"kept {pts[1]} face points, far fewer than the other repeats " \
           f"({np.median([pts[0], pts[2]]):.0f}, their median" in w[0]
    assert fs.metric_scan_ids[1] in w[0] and "its capture took" in w[0]
    stored = json.loads((fs.path / "meta.json").read_text(encoding="utf-8"))
    assert stored["low_repeats"] == json.loads(json.dumps(low)) and w[0] in stored["warnings"]


def test_the_profile_says_how_many_repeats_it_really_used(rig):
    """The profile of that scan counts what each repeat gave it, names the 2nd with both
    counts (its own warning, its stored JSON) and still measures the face."""
    _use(rig)
    fs = rig["hit"]
    pr = api.face_profile(fs, label="hit_prof", stl=False)
    st = pr.stats
    pts = st["repeat_face_points"]
    assert len(pts) == 3 and pts[1] < 0.75 * min(pts[0], pts[2]) and st["n_repeats"] == 3
    assert st["repeats_used"] == 3 and st["all_repeats_pct"] < 75
    assert [lo["repeat"] for lo in st["low_repeats"]] == [2]
    assert st["low_repeats"][0]["metric_scan"] == fs.metric_scan_ids[1]
    w = [x for x in pr.warnings if x.startswith("repeat 2 ")]
    assert len(w) == 1 and f"gave the profile {pts[1]} face points" in w[0]
    assert "3 of 3 repeats measured the face" in w[0] and "measured by all 3" in w[0]
    assert st["flatness_rms_mm"] < 0.06 and st["grid_coverage_pct"] > 90  # measured as before
    doc = json.loads(Path(pr.files["json"]).read_text(encoding="utf-8"))
    assert doc["stats"]["repeat_face_points"] == pts and w[0] in doc["warnings"]
    assert doc["stats"]["low_repeats"][0]["repeat"] == 2


def test_repeats_that_agree_say_nothing(rig):
    _use(rig)
    fs = rig["clean"]
    assert fs.meta["low_repeats"] == [] and not [w for w in fs.warnings if "far fewer" in w]
    pr = api.face_profile(fs, label="clean_prof", stl=False)
    st = pr.stats
    assert st["repeats_used"] == 3 and st["low_repeats"] == [] and st["all_repeats_pct"] > 90
    assert min(st["repeat_face_points"]) > 0.9 * max(st["repeat_face_points"])
    assert not [w for w in pr.warnings if "far fewer" in w]


def test_cli_face_scan_and_face_profile_print_and_dump_it(rig, capsys, monkeypatch):
    """face-scan: the table's face points (the repeat marked) and the warning; --json: the
    warning and meta low_repeats. face-profile: the repeats line and the warning; --json:
    the stats."""
    _use(rig)
    s = rig["s"]

    def same_session(*a, **k):                  # the CLI on THIS bench (its own would be new)
        api._session = s
        return s
    monkeypatch.setattr(api, "configure", same_session)
    base = ["--mock", "--library", str(rig["lib"]), "--preset", "face_fast"]
    _disturb(monkeypatch, _cut(rig["face"]))
    try:
        code = cli.main(base + ["--json", "face-scan", "cli_hit", "--face", rig["face"].id,
                                "--repeats", "2", "--steps", "4", "--no-polar",
                                "--exposure-us", "16667"])
        doc = json.loads(capsys.readouterr().out)
        assert code == 0
        meta = doc["facescan"]["meta"]
        assert [lo["repeat"] for lo in meta["low_repeats"]] == [2]
        assert [w for w in doc["facescan"]["warnings"] if w.startswith("repeat 2 ")]
        code = cli.main(base + ["face-profile", rig["hit"].id, "--label", "cli_prof", "--no-stl"])
        out = capsys.readouterr().out
        assert code == 0 and "3 of 3 used; face points per repeat" in out
        line = next(x for x in out.splitlines() if x.strip().startswith("repeats "))
        per = line.split("face points per repeat ")[1].split(";")[0].split(" / ")
        assert [p.endswith("(!)") for p in per] == [False, True, False], line
        assert "measured by all:" in line
        assert "! repeat 2 (" in out and "far fewer than the other repeats" in out
        code = cli.main(base + ["--json", "face-profile", rig["hit"].id, "--label",
                                "cli_prof_j", "--no-stl"])
        doc = json.loads(capsys.readouterr().out)
        assert code == 0 and doc["profile"]["stats"]["low_repeats"][0]["repeat"] == 2
        assert [w for w in doc["warnings"] if w.startswith("repeat 2 ")]
    finally:
        api._session = s


def test_cli_face_scan_table_marks_the_repeat(rig, capsys, monkeypatch):
    _use(rig)
    s = rig["s"]
    monkeypatch.setattr(api, "configure", lambda *a, **k: s)
    monkeypatch.setattr(api, "targeted_face_scan", lambda *a, **k: rig["hit"])
    try:
        code = cli.main(["--mock", "--library", str(rig["lib"]), "--preset", "face_fast",
                         "face-scan", "x", "--face", rig["face"].id])
        out = capsys.readouterr().out
    finally:
        api._session = s
    pts = [r["face_points"] for r in rig["hit"].repeats]
    assert code == 0 and "face points" in out
    row = next(x for x in out.splitlines() if x.strip().startswith("2 !"))
    assert str(pts[1]) in row and rig["hit"].metric_scan_ids[1] in row
    assert "  ! repeat 2 (" in out and f"kept {pts[1]} face points" in out


# ------------------------------------------------------------ offline: synthetic placements

@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    lib = tmp_path_factory.mktemp("facerep_synth") / "library"
    lib.mkdir()
    rg = S.rig()
    cal = S.write_calibration(lib, rg)
    c, n = (-6.0, 4.0, 780.0), S.normal_from_tilt(10, 6)
    loc, reps = S.placement(lib, rg, cal, "p", c, n, repeats=3, seed=3)
    s = api.configure(lib, mock=True)
    face = api.locate_face(loc, S.DIAMETER, "p_face")
    yield {"lib": lib, "s": s, "face": face, "reps": reps}
    api._session = None


def _thin(lib, scan_id: str, face, keep: str) -> None:
    """A repeat that measured only part of the face: its cloud keeps the left half of the
    face ("left") or nothing of it ("none") - as written by a scan that went wrong."""
    f = Path(lib) / "metric" / S.GEO / scan_id / "cloud.npz"
    with np.load(f) as z:
        xyz, mask = z["xyz"], z["mask"].copy()
    cx = int(face.camera_ellipse["cx"])
    if keep == "left":
        mask[:, cx:] = False
    else:
        mask[:] = False
    np.savez_compressed(f, xyz=xyz, mask=mask)


def test_the_profile_names_a_thin_repeat_of_plain_metric_scans(synth, tmp_path):
    """No face scan to say it (an older one, or metric scans given by hand): the profile
    still counts each repeat and names the one that measured half the face."""
    import shutil
    api._session = synth["s"]
    lib = Path(shutil.copytree(synth["lib"], tmp_path / "library"))
    api.configure(lib, mock=True)
    _thin(lib, synth["reps"][2], synth["face"], "left")
    pr = api.face_profile(synth["reps"], face=synth["face"].id, label="thin", stl=False)
    st = pr.stats
    pts = st["repeat_face_points"]
    assert pts[2] < 0.7 * min(pts[:2]) and st["repeats_used"] == 3
    assert [lo["repeat"] for lo in st["low_repeats"]] == [3]
    w = [x for x in pr.warnings if x.startswith("repeat 3 ")]
    assert len(w) == 1 and synth["reps"][2] in w[0] and f"{pts[2]} face points" in w[0]
    assert "its capture took" not in w[0]                       # no capture time recorded


def test_a_repeat_with_no_face_point_is_not_used_and_said(synth, tmp_path):
    import shutil
    api._session = synth["s"]
    lib = Path(shutil.copytree(synth["lib"], tmp_path / "library"))
    api.configure(lib, mock=True)
    _thin(lib, synth["reps"][0], synth["face"], "none")
    pr = api.face_profile(synth["reps"], face=synth["face"].id, label="none", stl=False)
    st = pr.stats
    assert st["repeat_face_points"][0] == 0 and st["repeats_used"] == 2 and st["n_repeats"] == 3
    assert st["all_repeats_pct"] == 0.0
    w = [x for x in pr.warnings if x.startswith("repeat 1 ")]
    assert len(w) == 1 and "gave the profile 0 face points" in w[0]
    assert "2 of 3 repeats measured the face" in w[0]
    assert cli._repeats_line(st).startswith("2 of 3 used; face points per repeat 0 (!) / ")
    assert cli._repeats_line({"n_repeats": 2}) == "2 (not counted per repeat)"

"""mm and the projector display (rule 6), mock only.

A capture is made in video display or in pattern-sequence display, and the same
pattern lands a fraction of a projector column apart in the two. So a stereo
calibration counts only for captures of the display it was made in:

- the calibration record carries its display (missing = video), the lookup returns
  only records of the display asked for, and calibrate_stereo stamps it from the poses;
- a pose carries the display its capture step says (capture_pose stores nothing when
  the step does not say), and a pose set holds one display;
- `_save`, the one function that writes mm, stamps the capture's display and refuses a
  calibration of the other display before a file exists;
- a stored scan is looked up, and re-triangulated, with its own display;
- a pattern-display calibration counts only while no video calibration is newer (the
  rig check and the re-home renew the video one only), and a mobile rig verified for
  one calibration gives no mm with another;
- only the simulated bench (a mock session at a mock geometry) lets one calibration
  count for both, when it has none of the display asked for, and a real session never
  gets that answer.

No test here reaches a device: the scans and calibrations are small made-up records in
a temporary library, the board poses come from the mock bench's oracle, and every way
into the rig fails the test instead (`no_rig`).
"""

import json
import sys
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

import recipe_synth as RS
from conftest import FIXTURE_PRESETS, PRESET
from engine import api, config, rigstate, rules
from engine.calibrate import flow, stereo
from engine.calibrate.charuco import BoardSpec, detect
from engine.calibrate.projector import projector_corner_count
from engine.capture.mock_bench import MockBench
from engine.capture.mock_stereo import CharucoTarget, StereoRig, StereoScene, board_poses
from engine.decode.deformation import DeformationField
from engine.library.store import LibraryStore
from engine.reconstruct import metric_scan
from engine.reconstruct.triangulate import triangulate
from engine.session import ScanSession
from tools import make_flash_pack as mfp

GEO = "rig_a"                               # a real rig's geometry id
MOCK_GEO = config.MOCK_PREFIX + GEO         # where the simulated bench keeps its data
H, W = 24, 32


@pytest.fixture(autouse=True)
def no_rig(monkeypatch):
    """Every way into the real rig fails the test: the machine-wide rig lock (the first
    thing a real scan takes) and the real camera."""
    from engine import session as session_mod
    from engine.capture import camera, riglock

    def refuse(*a, **k):
        raise AssertionError("a test reached the real rig")
    opened = camera.open_camera

    def mock_only(cfg, *, mock, **k):
        if not mock:
            refuse()
        return opened(cfg, mock=mock, **k)
    monkeypatch.setattr(riglock.RigLock, "acquire", refuse)
    monkeypatch.setattr(camera, "open_camera", mock_only)
    monkeypatch.setattr(session_mod, "open_camera", mock_only)
    before = {m for m in ("PySpin", "hid") if m in sys.modules}
    yield
    api._session = None
    assert {m for m in ("PySpin", "hid") if m in sys.modules} <= before


@pytest.fixture
def real(tmp_path):
    return ScanSession(tmp_path / "lib", mock=False)


@pytest.fixture
def mock(tmp_path):
    return ScanSession(tmp_path / "lib", mock=True, bench=MockBench(seed=1))


# ------------------------------------------------------------------ made-up records

def make_cal(store, geo=GEO, display="video", fp=None, **doc_changes):
    """A stereo calibration record in the library. display None: a record written
    before the field existed."""
    d = store.new_dir("calibrations", geo, label="stereo")
    cal = stereo.StereoCalibration(
        id=f"{geo}/{d.name}", geometry_id=geo,
        K_cam=np.array([[100.0, 0, 16], [0, 100.0, 12], [0, 0, 1]]), dist_cam=np.zeros(5),
        K_proj=np.array([[100.0, 0, 16], [0, 200.0, 12], [0, 0, 1]]), dist_proj=np.zeros(5),
        R=np.eye(3), t=np.array([-50.0, 0.0, 0.0]), cam_size=(W, H), fingerprint=fp, path=d,
        created="2026-01-01T00:00:00")
    doc = cal.to_json()
    if display is None:
        del doc["display_mode"]
    else:
        doc["display_mode"] = display
    doc.update(doc_changes)
    store.write_json(d / stereo.STEREO_FILE, doc)
    return stereo.StereoCalibration.from_json(doc, d)


def wall(geo=GEO, **meta):
    """The decoded field of a flat wall 500 mm away, as that calibration sees it."""
    v, u = np.mgrid[0:H, 0:W].astype(np.float32)
    ok = np.ones((H, W), bool)
    return DeformationField(corr_x=u - 10.0, corr_y=2.0 * (v - 12.0) + 12.0, corr_diag=None,
                            conf=ok.astype(np.float32), mask=ok, geometry_id=geo,
                            basis_names=["multifreq"], meta={"synthetic": True, **meta})


def pattern_copy(store, cal):
    """The same calibration once more, as a NEW record stamped pattern: the newest entry
    at its geometry."""
    d = store.new_dir("calibrations", cal.geometry_id, label="stereo")
    doc = {**cal.to_json(), "id": f"{cal.geometry_id}/{d.name}", "display_mode": "pattern"}
    store.write_json(d / stereo.STEREO_FILE, doc)
    return stereo.StereoCalibration.from_json(doc, d)


def save(s, cal, label="scan", display="video", extra=None, geo=GEO, fp=None, preset=None,
         **field_meta):
    fld = wall(geo, **field_meta)
    return metric_scan._save(s, fld, triangulate(fld, cal, "plane"), label, cal, fp, preset,
                             dict(extra or {}), display_mode=display)


def stored_before(ms, **changes):
    """The entry as it would lie in a library written before the display was stamped:
    no stamp, and whatever else `changes` says of it."""
    f = ms.path / "meta.json"
    m = json.loads(f.read_text(encoding="utf-8"))
    m.pop("display_mode", None)
    m.update(changes)
    f.write_text(json.dumps(m), encoding="utf-8")
    ms.meta = m
    return ms


def everything(store) -> list:
    """Every folder and file of the library (but the session logs)."""
    return sorted(str(p.relative_to(store.root)) for p in store.root.rglob("*")
                  if p.relative_to(store.root).parts[0] != "sessions")


def one_grab(preset):
    """The preset with one camera grab per pattern (the camera is not part of the
    geometry): a mock capture through the normal path, several times faster."""
    return replace(preset, camera=replace(preset.camera, average=1))


# ------------------------------------------------------------------ the calibration record

def test_a_record_without_the_field_is_a_video_calibration(tmp_path):
    store = LibraryStore(tmp_path)
    old = make_cal(store, display=None)
    assert old.display_mode == "video" and old.to_json()["display_mode"] == "video"
    assert make_cal(store, display=None, display_mode=None).display_mode == "video"
    pat = make_cal(store, display="pattern")
    back = stereo.StereoCalibration.from_json(json.loads(json.dumps(pat.to_json())), pat.path)
    assert back.display_mode == "pattern"


def test_a_pattern_record_is_never_returned_to_a_video_lookup(tmp_path):
    """The pattern-display calibration is the NEWEST at the geometry, and every lookup
    that does not ask for it (all the normal scans) still gets the video one."""
    store = LibraryStore(tmp_path)
    assert stereo.latest(store, GEO) is None
    with pytest.raises(rules.GeometryMismatch, match="Run calibrate_stereo") as e:
        stereo.require(store, GEO, None, "scan")
    assert not isinstance(e.value, rules.DisplayMismatch)       # nothing there at all
    pat0 = make_cal(store, display="pattern")
    assert stereo.latest(store, GEO) is None                    # a pattern record alone
    with pytest.raises(rules.DisplayMismatch, match="made in pattern-sequence display"):
        stereo.require(store, GEO, None, "scan")
    vid = make_cal(store, display=None)
    pat = make_cal(store, display="pattern")                    # the newest entry
    assert store.entries("calibrations", GEO)[-1] == pat.path
    assert stereo.latest(store, GEO).id == vid.id
    assert stereo.latest(store, GEO, None).id == vid.id
    assert stereo.require(store, GEO, None, "scan").id == vid.id
    assert stereo.latest(store, GEO, None, "pattern").id == pat.id != pat0.id
    assert stereo.require(store, GEO, None, "scan", "pattern").id == pat.id
    assert stereo.require(store, GEO, None, "scan", display_mode="video").id == vid.id


def test_a_video_record_is_never_returned_to_a_pattern_lookup(tmp_path):
    store = LibraryStore(tmp_path)
    make_cal(store, display="video")
    make_cal(store, display=None)
    assert stereo.latest(store, GEO, None, "pattern") is None
    with pytest.raises(rules.DisplayMismatch, match="no stereo calibration made in "
                                                    "pattern-sequence display") as e:
        stereo.require(store, GEO, None, "scan_trigger", "pattern")
    assert "made in video display" in str(e.value) and "rule 6" in str(e.value)
    assert isinstance(e.value, rules.GeometryMismatch)          # handled as every rule-6 refusal


@pytest.mark.parametrize("value", ["both", "any", "Video", "pattern ", ["video", "pattern"], 1,
                                   True])
def test_a_record_of_an_unknown_display_counts_for_none(tmp_path, value):
    store = LibraryStore(tmp_path)
    make_cal(store, display=value, mock=True)
    for display in rules.DISPLAYS:
        assert stereo.latest(store, GEO, None, display) is None
    with pytest.raises(ValueError, match="display_mode"):
        stereo.latest(store, GEO, None, "both")
    with pytest.raises(ValueError, match="display_mode"):
        stereo.latest(store, GEO, None, None)


def test_a_rehomed_calibration_keeps_its_display(tmp_path):
    store = LibraryStore(tmp_path)
    vid = make_cal(store, display="video")
    pat = make_cal(store, display="pattern")
    new = stereo.rehome(store, pat, np.eye(3), created="2026-01-02T00:00:00")
    assert new.display_mode == "pattern"
    assert json.loads((new.path / stereo.STEREO_FILE).read_text())["display_mode"] == "pattern"
    assert stereo.latest(store, GEO).id == vid.id
    assert stereo.latest(store, GEO, None, "pattern").id == new.id


# ------------------------------------------------------------------ the one writer of mm

def test_save_stamps_the_display(real):
    vid, pat = make_cal(real.store), make_cal(real.store, display="pattern")
    a = save(real, vid, "a")
    b = save(real, pat, "b", display="pattern")
    c = save(real, vid, "c", extra={"display_mode": "pattern"})    # extra cannot set it
    for ms, display in ((a, "video"), (b, "pattern"), (c, "video")):
        m = json.loads((ms.path / "meta.json").read_text())
        assert m["display_mode"] == ms.meta["display_mode"] == display and m["units"] == "mm"
        assert metric_scan.scan_display(real, ms) == display


@pytest.mark.parametrize("cal_display,capture", [("video", "pattern"), (None, "pattern"),
                                                 ("pattern", "video"), ("both", "video"),
                                                 ("both", "pattern"),
                                                 (["video", "pattern"], "pattern")])
def test_save_across_displays_leaves_no_file(real, cal_display, capture):
    """A capture of one display against a calibration of the other (also: a video scan
    saved against a pattern-display calibration): refused before the entry's folder
    exists. No cloud, no .ply, no output record, nothing in the trash."""
    cal = make_cal(real.store, display=cal_display)
    before = everything(real.store)
    with pytest.raises(rules.DisplayMismatch, match="rule 6"):
        save(real, cal, "x", display=capture)
    assert everything(real.store) == before
    assert real.store.entries("metric", GEO) == []
    with pytest.raises(ValueError, match="display_mode"):
        save(real, cal, "x", display="both")
    assert everything(real.store) == before


# ------------------------------------------------------------------ stored scans

# what makes a stored entry a pattern-display capture
PATTERN_ENTRIES = [
    {"mode": "flash_trigger"},                                  # scan_trigger, before the stamp
    {"mode": "flash_softsync"},                                 # fast_scan_flash, before it
    {"decode": {"softsync": {"loop_period_ms": 160.0}}},        # the one-bit flash decode
    {"display_mode": "pattern"},                                # the stamp
    {"display_mode": "video", "mode": "flash_softsync"},        # a stamp never outranks the mode
    {"display_mode": "video", "decode": {"softsync": {}}},
]


@pytest.mark.parametrize("says", PATTERN_ENTRIES)
def test_a_stored_pattern_scan_gives_no_mm_with_a_video_calibration(real, says):
    """check_cal and re-triangulating are refused; nothing is written."""
    vid = make_cal(real.store)
    ms = stored_before(save(real, vid, "flash"), **says)
    assert metric_scan.scan_display(real, ms) == "pattern"
    before = everything(real.store)
    with pytest.raises(rules.DisplayMismatch, match="pattern-sequence display"):
        metric_scan.check_cal(real, ms, "measure")
    with pytest.raises(rules.DisplayMismatch, match="pattern-sequence display"):
        metric_scan.triangulate_scan(real, ms)
    with pytest.raises(rules.DisplayMismatch):
        metric_scan.export_metric_mesh(real, ms)                # every use goes through check_cal
    assert everything(real.store) == before


def test_a_stored_pattern_scan_gets_mm_from_a_pattern_calibration_only(real):
    vid = make_cal(real.store)
    video_scan = save(real, vid, "normal")
    old = stored_before(save(real, vid, "flash"), mode="flash_trigger")
    pat = make_cal(real.store, display="pattern")               # now the newest calibration
    # its cloud was made with the video calibration: still not usable, re-triangulate
    with pytest.raises(rules.GeometryMismatch, match="re-triangulate"):
        metric_scan.check_cal(real, old, "measure")
    r = metric_scan.triangulate_scan(real, old)
    new = r["scan"]
    assert r["calibration_id"] == pat.id == new.calibration_id
    assert new.meta["display_mode"] == "pattern" and new.meta["source_scan"] == old.id
    assert metric_scan.check_cal(real, new, "measure").id == pat.id
    # and the normal scan keeps the video calibration, which is not the newest any more
    assert metric_scan.check_cal(real, video_scan, "measure").id == vid.id
    again = metric_scan.triangulate_scan(real, video_scan)["scan"]
    assert again.calibration_id == vid.id and again.meta["display_mode"] == "video"


def test_a_scan_of_an_unknown_display_gives_no_mm(real):
    vid = make_cal(real.store)
    ms = stored_before(save(real, vid, "odd"), display_mode="both")
    with pytest.raises(rules.DisplayMismatch, match="unknown projector display"):
        metric_scan.check_cal(real, ms, "measure")
    with pytest.raises(rules.DisplayMismatch):
        metric_scan.triangulate_scan(real, ms)


# entries triangulated from another scan before the stamp existed carry neither their
# mode nor a display: only "source_scan" (and the decode block of the source's field)

def test_an_entry_with_a_source_takes_its_sources_display(real):
    vid = make_cal(real.store)
    src_pat = stored_before(save(real, vid, "flash"), mode="flash_trigger")
    src_vid = stored_before(save(real, vid, "normal"))
    from_pat = stored_before(save(real, vid, "flash_tri"), source_scan=src_pat.id)
    from_vid = stored_before(save(real, vid, "normal_tri"), source_scan=src_vid.id)
    chain = stored_before(save(real, vid, "flash_tri_tri"), source_scan=from_pat.id)
    assert metric_scan.scan_display(real, from_pat) == "pattern"
    assert metric_scan.scan_display(real, chain) == "pattern"
    assert metric_scan.scan_display(real, from_vid) == "video"
    for ms in (from_pat, chain):
        with pytest.raises(rules.DisplayMismatch, match="pattern-sequence display"):
            metric_scan.check_cal(real, ms, "measure")
        with pytest.raises(rules.DisplayMismatch):
            metric_scan.triangulate_scan(real, ms)
    assert metric_scan.check_cal(real, from_vid, "measure").id == vid.id
    # re-triangulated today, the new entry says so itself
    new = metric_scan.triangulate_scan(real, from_vid)["scan"]
    assert new.meta["display_mode"] == "video" and new.meta["source_scan"] == from_vid.id


def test_an_entry_from_an_object_scan_is_a_video_capture(real):
    vid = make_cal(real.store)
    d = real.store.new_dir("objects", label="part")
    real.store.write_json(d / "meta.json", {"id": d.name, "label": "part", "geometry_id": GEO})
    ms = stored_before(save(real, vid, "part_tri"), source_scan=d.name)
    assert metric_scan.scan_display(real, ms) == "video"
    assert metric_scan.check_cal(real, ms, "measure").id == vid.id


def test_an_entry_with_the_softsync_decode_is_a_pattern_capture_whatever_its_source(real):
    vid = make_cal(real.store)
    src = stored_before(save(real, vid, "normal"))
    for source in (src.id, "20250101T000000_000000_gone"):
        ms = stored_before(save(real, vid, "tri", softsync={"confidence": 0.9}),
                           source_scan=source)
        assert "softsync" in ms.meta["decode"]                  # _save copies the field's meta
        assert metric_scan.scan_display(real, ms) == "pattern"
        with pytest.raises(rules.DisplayMismatch, match="pattern-sequence display"):
            metric_scan.check_cal(real, ms, "measure")


def test_an_entry_whose_source_cannot_be_found_is_refused(real):
    vid = make_cal(real.store)
    src = stored_before(save(real, vid, "flash"), mode="flash_trigger")
    ms = stored_before(save(real, vid, "flash_tri"), source_scan=src.id)
    never = stored_before(save(real, vid, "tri2"), source_scan="20250101T000000_000000_gone")
    loop = stored_before(save(real, vid, "tri3"))
    stored_before(loop, source_scan=loop.id)                    # names itself
    real.store.trash(src.path, reason="test: the source is gone")
    before = everything(real.store)
    for scan in (ms, never, loop):
        with pytest.raises(rules.DisplayMismatch, match="not in the library any more"):
            metric_scan.scan_display(real, scan)
        with pytest.raises(rules.DisplayMismatch, match="cannot be shown"):
            metric_scan.check_cal(real, scan, "measure")
        with pytest.raises(rules.DisplayMismatch):
            metric_scan.triangulate_scan(real, scan)
    assert everything(real.store) == before
    # an entry that carries its own stamp needs no source
    stamped = save(real, vid, "new", extra={"source_scan": "20250101T000000_000000_gone"})
    assert metric_scan.check_cal(real, stamped, "measure").id == vid.id


# ------------------------------------------------------------------ mock and real

def test_a_mock_sessions_calibration_counts_for_both_displays(mock):
    """The simulated projector has no offset between the displays."""
    cal = make_cal(mock.store, MOCK_GEO, display=None)
    assert metric_scan._require_cal(mock, MOCK_GEO, None, "scan_trigger", "pattern").id == cal.id
    assert metric_scan._find_cal(mock, MOCK_GEO, None, "pattern").id == cal.id
    ms = save(mock, cal, "flash", display="pattern", extra={"mode": "flash_softsync"},
              geo=MOCK_GEO)
    assert ms.meta["display_mode"] == "pattern"                 # stamped as what it is
    assert metric_scan.check_cal(mock, ms, "measure").id == cal.id
    new = metric_scan.triangulate_scan(mock, ms)["scan"]
    assert new.calibration_id == cal.id and new.meta["display_mode"] == "pattern"
    with pytest.raises(rules.GeometryMismatch, match="Run calibrate_stereo"):
        metric_scan._require_cal(mock, MOCK_GEO + "_other", None, "scan_trigger", "pattern")
    # the other way round too: a bench that has a pattern-display calibration only
    only = MOCK_GEO + "_pattern_only"
    pat = make_cal(mock.store, only, display="pattern")
    assert metric_scan._require_cal(mock, only, None, "scan_metric").id == pat.id
    assert save(mock, pat, "normal", geo=only).meta["display_mode"] == "video"


def test_on_the_mock_bench_each_display_takes_its_own_calibration_first(mock):
    """With a record of each display, a capture takes its own: a NORMAL scan gets the
    record stereo.latest() gives Measure, the recipes and the station jobs, also when a
    pattern-display record is the newest at the geometry. The other display's record
    stands in only when there is none of its own."""
    vid = make_cal(mock.store, MOCK_GEO)
    scan = save(mock, vid, "normal", geo=MOCK_GEO)
    pat = make_cal(mock.store, MOCK_GEO, display="pattern")     # the newest entry
    assert mock.store.entries("calibrations", MOCK_GEO)[-1] == pat.path
    for what in ("scan_metric", "scan_fast"):
        assert metric_scan._require_cal(mock, MOCK_GEO, None, what).id == vid.id
    assert metric_scan._find_cal(mock, MOCK_GEO, None).id == vid.id \
        == stereo.latest(mock.store, MOCK_GEO).id
    assert metric_scan.check_cal(mock, scan, "measure").id == vid.id   # still measurable
    again = metric_scan.triangulate_scan(mock, scan)["scan"]
    assert again.calibration_id == vid.id and again.meta["display_mode"] == "video"
    assert metric_scan._require_cal(mock, MOCK_GEO, None, "scan_trigger", "pattern").id == pat.id
    # a newer video calibration: the pattern-display one is overtaken, and on the bench
    # the video one counts for both displays
    vid2 = make_cal(mock.store, MOCK_GEO)
    assert metric_scan._require_cal(mock, MOCK_GEO, None, "scan_metric").id == vid2.id
    assert metric_scan._require_cal(mock, MOCK_GEO, None, "scan_trigger", "pattern").id == vid2.id


def test_a_normal_mock_scan_is_not_changed_by_a_pattern_calibration(eng, preset):
    """The whole path on the simulated bench: a pattern-display record is the newest at
    the geometry, and a normal scan is still triangulated with the calibration every
    other lookup of the run gets (stereo.latest), so the run is not told to re-scan."""
    fp, geo = config.geometry_fingerprint(preset), preset.geometry_id
    eng.bench.use_stereo()
    vid = RS.truth_calibration(eng, preset)
    pat = pattern_copy(eng.store, vid)
    assert eng.store.entries("calibrations", geo)[-1] == pat.path
    ms = api.scan_metric(one_grab(preset), "normal", steps=4)
    assert ms.calibration_id == vid.id == stereo.latest(eng.store, geo, fp).id
    assert ms.meta["display_mode"] == "video" and ms.meta["units"] == "mm"
    assert metric_scan.check_cal(eng, ms, "recipe_run").id == vid.id


def test_a_real_session_never_gets_that_answer(tmp_path):
    """The same library, with the entries a mock session wrote into it, opened by a real
    session: every one of those answers is a refusal. So is real data in a mock session,
    and no record can talk its way in."""
    mock = ScanSession(tmp_path / "lib", mock=True, bench=MockBench(seed=1))
    cal = make_cal(mock.store, MOCK_GEO, display=None, mock=True)
    ms = save(mock, cal, "flash", display="pattern", extra={"mode": "flash_softsync"},
              geo=MOCK_GEO)
    real = ScanSession(tmp_path / "lib", mock=False)
    for s in (real, SimpleSession(real.store, mock=0), SimpleSession(real.store, mock="yes"),
              SimpleSession(real.store, mock=1), SimpleSession(real.store)):
        assert not metric_scan._both_displays(s, MOCK_GEO)
        assert metric_scan._find_cal(s, MOCK_GEO, None, "pattern") is None
        with pytest.raises(rules.DisplayMismatch, match="pattern-sequence display"):
            metric_scan._require_cal(s, MOCK_GEO, None, "scan_trigger", "pattern")
        with pytest.raises(rules.DisplayMismatch):
            metric_scan.check_cal(s, ms, "measure")
    before = everything(real.store)
    with pytest.raises(rules.DisplayMismatch):
        metric_scan.triangulate_scan(real, ms)
    with pytest.raises(rules.DisplayMismatch):
        save(real, cal, "x", display="pattern", geo=MOCK_GEO)
    assert everything(real.store) == before
    # real data opened in a mock session: the rig's geometry is not the simulated bench
    vid = make_cal(mock.store, GEO)
    old = stored_before(save(mock, vid, "flash"), mode="flash_trigger")
    assert not metric_scan._both_displays(mock, GEO)
    with pytest.raises(rules.DisplayMismatch):
        metric_scan.check_cal(mock, old, "measure")
    with pytest.raises(rules.DisplayMismatch):
        metric_scan.triangulate_scan(mock, old)
    with pytest.raises(rules.DisplayMismatch):
        save(mock, vid, "x", display="pattern")
    # records that claim to be for both displays, or to be mock data
    for claim in ("both", "any", ["video", "pattern"]):
        make_cal(real.store, GEO + "_claims", display=claim, mock=True)
    for display in rules.DISPLAYS:
        assert metric_scan._find_cal(real, GEO + "_claims", None, display) is None


class SimpleSession:
    """Anything that is not a session of the simulated bench (mock is not True)."""

    def __init__(self, store, **attrs):
        self.store = store
        self.__dict__.update(attrs)


def _real_engine(tmp_path):
    s = api.configure(tmp_path / "lib", mock=False, presets_dir=FIXTURE_PRESETS)
    p = api.load_preset(PRESET)
    assert not p.geometry_id.startswith(config.MOCK_PREFIX)
    return s, p, config.geometry_fingerprint(p)


def test_flash_scan_is_refused_on_a_real_rig_before_the_camera(tmp_path):
    s, p, fp = _real_engine(tmp_path)
    vid = make_cal(s.store, p.geometry_id, fp=fp)
    before = everything(s.store)
    roi = [500, 300, 940, 780]
    with pytest.raises(rules.DisplayMismatch, match="pattern-sequence display"):
        api.fast_scan_flash(None, None, "flash", p, roi=roi)
    plan = api.fast_scan_flash(None, None, "flash", p, roi=roi, dry_run=True)
    assert plan["dry_run"] and plan["calibration"] is None      # the plan says so too
    assert everything(s.store) == before
    assert stereo.latest(s.store, p.geometry_id, fp).id == vid.id      # a normal scan's
    pat = make_cal(s.store, p.geometry_id, display="pattern", fp=fp)
    plan = api.fast_scan_flash(None, None, "flash", p, roi=roi, dry_run=True)
    assert plan["calibration"] == pat.id
    # the rig is calibrated again: the plan names no calibration, as the run refuses
    make_cal(s.store, p.geometry_id, fp=fp)
    plan = api.fast_scan_flash(None, None, "flash", p, roi=roi, dry_run=True)
    assert plan["calibration"] is None
    with pytest.raises(rules.DisplayMismatch, match="older than the rig's calibration"):
        api.fast_scan_flash(None, None, "flash", p, roi=roi)


def test_a_live_trigger_recipe_on_a_real_rig_ends_pull(tmp_path):
    """A recipe that scans with the hardware-triggered scan (pattern-sequence display),
    run live on a rig that has a video calibration only: refused before the camera, the
    run is INVALID and the part is pulled. No scan entry, no mm."""
    s, p, fp = _real_engine(tmp_path)
    make_cal(s.store, p.geometry_id, fp=fp)
    pack = mfp.make_fastscan_pack(out_root=tmp_path / "packs", label="t",
                                  trigger_mode="trigout", period_us=10000)
    recipe = {"schema_version": 1, "name": "trig", "preset": PRESET,
              "scan": {"kind": "trigger", "pack": str(pack)},
              "zones": [{"id": "top", "shape": "rect", "units": "part_mm",
                         "coords": [-30, -20, 30, 20],
                         "checks": [{"type": "flatness",
                                     "params": {"tol_mm": 0.3, "min_height_mm": 3.0}}]}],
              "feedback": {"enabled": False}}
    out = api.recipe_run(recipe, preset=p, operator_confirmed=True, project_feedback=False,
                         runs_dir=tmp_path / "runs", pattern_mode_ack=True)
    res = out["result"]
    assert res["mode"] == "REAL" and res["result"] == "INVALID", res
    assert res["disposition"] == "PULL" and out["exit_code"] == 2
    assert "DisplayMismatch" in res["invalid_reason"]
    assert "pattern-sequence display" in res["invalid_reason"]
    assert res["part"]["scan_id"] is None
    assert s.store.entries("metric", p.geometry_id) == []
    assert not list((s.store.root / "output").rglob("*.ply"))
    # the scan itself, asked directly
    from engine.capture.patternmode import ACK_PHRASE
    with pytest.raises(rules.DisplayMismatch, match="pattern-sequence display"):
        api.scan_trigger(str(pack), None, "trig", p, acknowledge=ACK_PHRASE)
    assert s.store.entries("metric", p.geometry_id) == []


# ------------------------------------------------------------------ the rig moved since
# The rig check, the re-home and the rig gate work on the video calibration. Each display
# has its own newest record, so nothing they do reaches a pattern-display calibration.

@pytest.mark.parametrize("how", ["re-home", "new calibration"])
def test_a_pattern_calibration_older_than_the_rigs_newest_gives_no_mm(real, how):
    """A fixed rig. After a rig check MOVED the operator re-homes, or calibrates again:
    a new VIDEO record. The pattern-display calibration was made where the rig stood
    before, and it is not used any more: refused until it is made again."""
    vid = make_cal(real.store)
    pat = make_cal(real.store, display="pattern")
    ms = save(real, pat, "flash", display="pattern", extra={"mode": "flash_trigger"})
    assert metric_scan.check_cal(real, ms, "measure").id == pat.id
    new = stereo.rehome(real.store, vid, np.eye(3), created="2026-01-02T00:00:00") \
        if how == "re-home" else make_cal(real.store)
    assert new.display_mode == "video"
    assert metric_scan._require_cal(real, GEO, None, "scan_metric").id == new.id
    assert stereo.latest(real.store, GEO, None, "pattern").id == pat.id    # still on record
    assert metric_scan._find_cal(real, GEO, None, "pattern") is None
    before = everything(real.store)
    for call in (lambda: metric_scan._require_cal(real, GEO, None, "scan_trigger", "pattern"),
                 lambda: metric_scan.check_cal(real, ms, "measure"),
                 lambda: metric_scan.triangulate_scan(real, ms)):
        with pytest.raises(rules.DisplayMismatch, match="older than the rig's calibration") as e:
            call()
        assert pat.id in str(e.value) and new.id in str(e.value)
        assert "Make the pattern-display calibration again" in str(e.value)
    assert everything(real.store) == before
    # made again: it is the one in use, and the stored scan wants re-triangulating
    pat2 = make_cal(real.store, display="pattern")
    assert metric_scan._require_cal(real, GEO, None, "scan_trigger", "pattern").id == pat2.id
    with pytest.raises(rules.GeometryMismatch, match="re-triangulate"):
        metric_scan.check_cal(real, ms, "measure")
    assert metric_scan.triangulate_scan(real, ms)["calibration_id"] == pat2.id
    # the other way round nothing changes: a newer pattern record never unseats the video one
    assert metric_scan._require_cal(real, GEO, None, "scan_metric").id == new.id


def test_a_mobile_rig_verified_for_one_calibration_gives_no_mm_with_another(tmp_path):
    """A mobile rig is verified in this session by its (video) calibration. A
    pattern-display calibration is another record: the verification says nothing of
    it, so the two pattern-display scans and re-triangulating are refused before the
    camera. A normal scan is gated as before."""
    s, p, fp = _real_engine(tmp_path)
    api.preset_add("mob", PRESET, {"rig.mount": "mobile"})
    mob = api.load_preset("mob")
    geo = mob.geometry_id
    assert geo == p.geometry_id and rigstate.mount_of(s.store, mob)[0] == "mobile"
    rigstate.new_session(s.store, mob, "set down")
    rigstate.record_pose(s.store, mob, "pose1", "today")
    vid = make_cal(s.store, geo, fp=fp)
    rigstate.record_calibration(s.store, mob, vid.id, ["pose1"], "today")
    pat = make_cal(s.store, geo, display="pattern", fp=fp)      # from an earlier position
    g = rigstate.gate(s.store, mob, presets_dir=FIXTURE_PRESETS)
    assert g["ok"] and g["state"]["calibration_id"] == vid.id
    # a normal scan: gated as before, with or without the calibration named
    stamp = metric_scan._require_rig(s, mob, "scan_metric", cal=vid)
    assert stamp["mount"] == "mobile" and vid.id in stamp["verified_by"]
    assert metric_scan._require_rig(s, mob, "scan_metric")["verified_by"] == stamp["verified_by"]
    # a capture triangulated with the pattern-display calibration
    cal = metric_scan._require_cal(s, geo, fp, "scan_trigger", "pattern")
    assert cal.id == pat.id
    with pytest.raises(rules.RuleViolation, match="mobile rig not verified") as e:
        metric_scan._require_rig(s, mob, "scan_trigger", cal=cal)
    assert vid.id in str(e.value) and pat.id in str(e.value)
    assert "pattern-sequence display" in str(e.value)
    before = everything(s.store)
    pack = mfp.make_fastscan_pack(out_root=tmp_path / "packs", label="t",
                                  trigger_mode="trigout", period_us=10000)
    from engine.capture.patternmode import ACK_PHRASE
    with pytest.raises(rules.RuleViolation, match="mobile rig not verified"):
        api.scan_trigger(str(pack), None, "trig", mob, acknowledge=ACK_PHRASE)
    with pytest.raises(rules.RuleViolation, match="mobile rig not verified"):
        api.fast_scan_flash(None, None, "flash", mob, roi=[500, 300, 940, 780])
    assert everything(s.store) == before
    # stored captures of this session: the same answer when they are triangulated again
    flash = save(s, pat, "flash", display="pattern", extra={"mode": "flash_trigger"}, geo=geo,
                 fp=fp, preset="mob")
    normal = save(s, vid, "normal", geo=geo, fp=fp, preset="mob")
    with pytest.raises(rules.RuleViolation, match="mobile rig not verified"):
        metric_scan.triangulate_scan(s, flash)
    again = metric_scan.triangulate_scan(s, normal)["scan"]
    assert again.calibration_id == vid.id and again.meta["rig"]["mount"] == "mobile"


# ------------------------------------------------------------------ calibrate_stereo

@pytest.fixture(scope="module")
def board_views():
    """Four views of the calibration board as a capture would decode them: the exact
    projector position per camera pixel (the mock stereo bench's oracle) and the
    white-flood picture. Made once; no capture path."""
    rig, spec = StereoRig.bench(), BoardSpec()
    white = np.full((1140, 912), 255, np.uint8)
    views = []
    for R, t in board_poses(spec, 4, seed=0):
        sc = StereoScene(rig, [CharucoTarget(spec=spec, R=R, t=t)], gain=250)
        cx, cy, m = sc.oracle()
        views.append((cx.astype(np.float32), cy.astype(np.float32), m,
                      np.clip(sc.render(white), 0, 255).astype(np.uint8)))
    return spec, views


NOT_SAID = object()
PERIODS = [8.0, 80.0, 1280.0]


def capture_step(board_views, display=NOT_SAID):
    """A capture step for flow.capture_pose that hands over decoded arrays instead of
    driving the rig, as a program does that captures the board in its own way: a
    calibration tool for pattern-sequence display puts its function in the place of
    flow._shoot_pose and stores the pose with the engine's own verb.
    display: what the step says its frames were captured in; NOT_SAID: it does not say,
    as such a program did before the key existed."""
    left = list(board_views[1])

    def shoot(s, preset, boards, fxs, par, steps, periods):
        cx, cy, m, flood = left.pop(0)
        fld = DeformationField(corr_x=cx, corr_y=cy, corr_diag=None, conf=m.astype(np.float32),
                               mask=m, geometry_id=preset.geometry_id,
                               basis_names=["multifreq"], meta={})
        det = detect(flood, boards[0])
        n = 0 if det is None else len(det[1])
        k = 0 if det is None else projector_corner_count(det[0], cx, cy)
        pct = round(100.0 * k / n, 1) if n else 0.0
        usable = n >= flow.MIN_POSE_CORNERS
        shot = {"basis": SimpleNamespace(steps=int(steps), periods=[float(x) for x in periods]),
                "fld": fld, "flood": flood, "found": [n], "mapped": [k], "n": n, "k": k,
                "pct": pct, "usable": usable,
                "accepted": usable and pct >= flow.MIN_PROJECTOR_CORNER_PCT, "capture_s": 0.0}
        if display is not NOT_SAID:
            shot["display_mode"] = display
        return shot
    return shoot


def store_poses(monkeypatch, preset, board_views, set_label: str, display=NOT_SAID,
                n: int = 4) -> list:
    """n board poses stored by the engine's own verb (the session api.configure made),
    with its capture step replaced by capture_step."""
    monkeypatch.setattr(flow, "_shoot_pose", capture_step(board_views, display))
    return [api.capture_calib_pose(preset, set_label, board=board_views[0].to_dict(), steps=4,
                                   periods=PERIODS, diamond_parity=None, auto_periods=False)
            for _ in range(n)]


def pose_meta(pose) -> dict:
    return json.loads((pose["path"] / "meta.json").read_text(encoding="utf-8"))


def pose_on_disk(pose, **changes) -> None:
    """Rewrite a stored pose's record: without its display (a pose stored before the key
    existed), or saying what `changes` says."""
    m = pose_meta(pose)
    m.pop("display_mode", None)
    m.update(changes)
    (pose["path"] / "meta.json").write_text(json.dumps(m), encoding="utf-8")


def test_the_engines_own_pose_capture_is_a_video_pose(eng, preset):
    """calib-pose projects its fringes through the HDMI window, and the pose says so."""
    spec = BoardSpec()
    R, t = board_poses(spec, 4, seed=0)[0]
    eng.bench.use_stereo()
    eng.bench.place(CharucoTarget(spec=spec, R=R, t=t))
    pose = api.capture_calib_pose(one_grab(preset), "own", steps=4, auto_periods=False)
    assert pose["corners_found"] >= flow.MIN_POSE_CORNERS
    assert pose["display_mode"] == pose_meta(pose)["display_mode"] == "video"


@pytest.mark.parametrize("said", [NOT_SAID, None, "", "both", "pattern sequence", ["pattern"], 1],
                         ids=["no key", "None", "empty", "both", "other words", "a list", "1"])
def test_a_capture_step_that_does_not_say_its_display_stores_nothing(tmp_path, board_views,
                                                                     monkeypatch, said):
    """A program that stores poses through the engine's verb with its own capture step,
    and does not say which display it captured in (every such program, before the key
    existed): the pose is refused. It never becomes a video pose by default, which would
    make a pattern-display calibration the calibration of every normal scan."""
    s, p, fp = _real_engine(tmp_path)
    before = everything(s.store)
    with pytest.raises(rules.DisplayMismatch, match="did not say which projector display"):
        store_poses(monkeypatch, p, board_views, "native", said, n=1)
    assert everything(s.store) == before
    assert flow.list_poses(s.store, p.geometry_id, fp) == []
    assert rigstate.read(s.store, p.geometry_id) == []          # nor on the rig's record
    with pytest.raises(ValueError, match="no calibration poses"):
        api.calibrate_stereo(p, "native")


def test_calibrate_stereo_stamps_the_record_from_its_poses(eng, preset, board_views, monkeypatch):
    fp = config.geometry_fingerprint(preset)
    normal = store_poses(monkeypatch, preset, board_views, "normal", "video")
    assert all(x["accepted"] and x["display_mode"] == "video" for x in normal)
    for pose in normal[:3]:
        pose_on_disk(pose)                  # stored before the key existed: video poses
    vid, summary = flow.calibrate_stereo(eng, preset, set_label="normal")
    assert vid.display_mode == summary["display_mode"] == "video"
    native = store_poses(monkeypatch, preset, board_views, "native", "pattern")
    for pose in native:                     # the pose says it, in the result and on disk
        assert pose["display_mode"] == pose_meta(pose)["display_mode"] == "pattern"
    pat, summary = flow.calibrate_stereo(eng, preset, set_label="native")
    assert pat.display_mode == summary["display_mode"] == "pattern"
    for cal, display in ((vid, "video"), (pat, "pattern")):
        doc = json.loads((cal.path / stereo.STEREO_FILE).read_text(encoding="utf-8"))
        assert doc["display_mode"] == display
        assert json.loads((cal.path / "report.json").read_text())["display_mode"] == display
    # the pattern-display calibration is the newest, and no normal lookup returns it
    assert eng.store.entries("calibrations", preset.geometry_id)[-1] == pat.path
    assert stereo.latest(eng.store, preset.geometry_id, fp).id == vid.id
    assert stereo.latest(eng.store, preset.geometry_id, fp, "pattern").id == pat.id


def test_capture_pose_refuses_a_pose_of_the_other_display_than_its_set(eng, preset, board_views,
                                                                       monkeypatch):
    geo = preset.geometry_id
    store_poses(monkeypatch, preset, board_views, "normal", "video", n=2)
    old = store_poses(monkeypatch, preset, board_views, "old", "video", n=1)
    pose_on_disk(old[0])                    # a pose from before the key: a video pose
    store_poses(monkeypatch, preset, board_views, "native", "pattern", n=2)
    before, lines = everything(eng.store), len(rigstate.read(eng.store, geo))
    for set_label, display, held in (("normal", "pattern", "video display"),
                                     ("old", "pattern", "video display"),
                                     ("native", "video", "pattern-sequence display")):
        with pytest.raises(ValueError, match=f"holds poses captured in {held}") as e:
            store_poses(monkeypatch, preset, board_views, set_label, display, n=1)
        assert "would mix the two projector displays" in str(e.value)
        assert "another set name" in str(e.value)
    assert everything(eng.store) == before and len(rigstate.read(eng.store, geo)) == lines
    # each set still takes poses of its own display
    more = store_poses(monkeypatch, preset, board_views, "native", "pattern", n=1)
    assert pose_meta(more[0])["display_mode"] == "pattern"
    assert len(flow.list_poses(eng.store, geo, config.geometry_fingerprint(preset), "native")) == 3


@pytest.mark.parametrize("displays", [[None, None, None, "pattern"],
                                      ["pattern", "video", "pattern", "pattern"]])
def test_calibrate_stereo_refuses_a_mixed_pose_set(eng, preset, board_views, monkeypatch,
                                                   displays):
    """A set that is mixed on disk (poses stored before capture_pose kept the two apart,
    or put there by hand): refused when it is calibrated, nothing saved."""
    poses = store_poses(monkeypatch, preset, board_views, "mixed", "pattern")
    for pose, display in zip(poses, displays):
        pose_on_disk(pose, **({} if display is None else {"display_mode": display}))
    with pytest.raises(ValueError, match="mixes poses captured in video display and in "
                                         "pattern-sequence display"):
        flow.calibrate_stereo(eng, preset, set_label="mixed")
    assert eng.store.entries("calibrations", preset.geometry_id) == []     # nothing saved
    odd = store_poses(monkeypatch, preset, board_views, "odd", "pattern")
    for pose in odd:
        pose_on_disk(pose, display_mode="both")
    with pytest.raises(ValueError, match="display_mode"):
        flow.calibrate_stereo(eng, preset, set_label="odd")
    assert eng.store.entries("calibrations", preset.geometry_id) == []


def test_a_pattern_calibration_in_the_rigs_library_changes_no_normal_scan(tmp_path, board_views,
                                                                          monkeypatch):
    """The calibration tool's day, in the REAL rig's library, by the route the tool
    takes: board poses captured in pattern-sequence display, stored with the engine's
    verb (its capture step replaced), then calibrate_stereo. The record is the newest
    of the geometry and a normal scan does not take it: with no video calibration it is
    refused before the rig, with one it keeps that one."""
    s, p, fp = _real_engine(tmp_path)
    geo = p.geometry_id
    poses = store_poses(monkeypatch, p, board_views, "native", "pattern")
    assert [pose_meta(x)["display_mode"] for x in poses] == ["pattern"] * 4
    pat = api.calibrate_stereo(p, "native")["calibration"]
    assert pat.display_mode == "pattern"
    doc = json.loads((pat.path / stereo.STEREO_FILE).read_text(encoding="utf-8"))
    assert doc["display_mode"] == "pattern"
    assert stereo.latest(s.store, geo, fp) is None              # no normal lookup returns it
    with pytest.raises(rules.DisplayMismatch, match="made in pattern-sequence display"):
        api.scan_metric(p, "normal")                             # refused before the rig
    assert metric_scan._require_cal(s, geo, fp, "scan_trigger", "pattern").id == pat.id
    store_poses(monkeypatch, p, board_views, "normal", "video")
    vid = api.calibrate_stereo(p, "normal")["calibration"]
    assert vid.display_mode == "video"
    assert metric_scan._require_cal(s, geo, fp, "scan_metric").id == vid.id
    # the rig was calibrated again after the pattern-display calibration was made
    with pytest.raises(rules.DisplayMismatch, match="older than the rig's calibration"):
        metric_scan._require_cal(s, geo, fp, "scan_trigger", "pattern")
    newest = make_cal(s.store, geo, display="pattern", fp=fp)
    assert metric_scan._require_cal(s, geo, fp, "scan_metric").id == vid.id
    assert metric_scan._require_cal(s, geo, fp, "x", "pattern").id == newest.id

"""Painted zones end to end on the MOCK rig: a scripted painter session saves a zone set,
zone-show reports where the zones land, a fast scan lit through the set lights only its
zones and counts the points per zone, Measure mode takes a zone as its region, and the CLI
commands. The real rig is never opened: its refusals are checked with the rig lock and the
drivers stubbed out."""

import contextlib
import io
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from engine import api, rules
from engine.calibrate import flow
from engine.zones import DMD_SHAPE, ZoneError
from engine.zones import painter as P
from engine.zones import store as zstore
from engine.zones import use
from ui import cli

import recipe_synth as RS

PRESET = "bench_9in_metric"
GEO = "mock-bench_9in_v1"


@pytest.fixture(scope="module")
def painted(tmp_path_factory):
    """One mock library: the bench's true stereo calibration, the zone set `demo` painted by
    the scripted demo session, and ONE fast scan lit through its zones Z1 + Z3 (both axes)."""
    lib = tmp_path_factory.mktemp("zscan") / "library"
    s = api.configure(lib, mock=True)
    seen = {"masks": []}
    real = flow.capture_patterns

    def spy(sess, preset, patterns, what, zone_mask=None):
        seen["masks"].append(None if zone_mask is None else np.array(zone_mask, bool))
        seen["n_patterns"] = len(patterns)
        return real(sess, preset, patterns, what, zone_mask=zone_mask)

    flow.capture_patterns = spy
    try:
        p = api.load_preset(PRESET)
        RS.truth_calibration(s, p)
        out = api.zone_paint("demo", p, events=list(P.DEMO_EVENTS))
        ms = api.scan_fast(None, p, "zfast", zones="demo:Z1,Z3", axis="xy")
    finally:
        flow.capture_patterns = real
        api._session = None
    return {"lib": lib, "paint": out, "scan_id": ms.id, "scan_meta": ms.meta, "seen": seen}


@pytest.fixture
def rig(painted):
    s = api.configure(painted["lib"], mock=True)
    p = api.load_preset(PRESET)
    yield s, p, painted
    api._session = None


def _scan(painted):
    return next(m for m in api.list_library("metric") if m.id == painted["scan_id"])


# ------------------------------------------------------------------ the painter session

def test_scripted_session_end_to_end_saves_a_zone_set(rig):
    s, p, painted = rig
    out = painted["paint"]
    assert out["saved"] is True and out["reason"] is None and out["mock"] is True
    assert out["name"] == "demo" and out["geometry_id"] == GEO == p.geometry_id
    assert out["spec"] == "zoneset:demo" and out["polarity_inverted"] is False
    assert [(z["id"], z["shape"]) for z in out["zones"]] == [
        ("Z1", "rect"), ("Z2", "rect"), ("Z3", "ellipse"), ("Z4", "polygon")]
    assert out["zones"][0]["name"] == "part" and out["zones"][1]["rotation_deg"] == 15.0
    d = painted["lib"] / "zones" / GEO / "demo"
    assert out["path"] == d
    assert sorted(f.name for f in d.iterdir() if f.name != "exports.jsonl") == [
        "camera.png", "masks.npz", "preview.png", "zones.json"]
    zs = zstore.load(s.store, GEO, "demo")
    assert zs.preset == PRESET and zs.mock and zs.source is None and zs.created
    assert zs.fingerprint == __import__("engine.config", fromlist=["x"]).geometry_fingerprint(p)
    assert zs.bboxes()["Z1"] == [180, 260, 420, 470]
    # the photo taken on confirm: the outlines on the scene, thin bright lines on a dark frame
    photo = cv2.imread(str(d / "camera.png"), cv2.IMREAD_GRAYSCALE)
    assert photo is not None and photo.ndim == 2
    assert 0.001 < (photo > 128).mean() < 0.05


def test_everything_refusable_is_refused_before_the_rig_opens(rig, monkeypatch):
    s, p, painted = rig

    def no_rig(*a, **k):
        raise AssertionError("the rig was opened")
    monkeypatch.setattr(type(s), "rig", no_rig)
    ev = list(P.DEMO_EVENTS)
    with pytest.raises(ZoneError, match="exists at geometry.*--replace"):
        api.zone_paint("demo", p, events=ev)
    for name in ("bad name", "../x", ""):
        with pytest.raises(ZoneError, match="name must be"):
            api.zone_paint(name, p, events=ev)
    with pytest.raises(ValueError, match="MOCK: the painter runs headless"):
        api.zone_paint("fresh", p)
    with pytest.raises(ValueError, match="list of events"):
        api.zone_paint("fresh", p, events=[{"no": "type"}])
    with pytest.raises(ValueError, match="polarity"):
        api.zone_paint("fresh", p, events=ev, polarity="guess")
    with pytest.raises(ValueError, match="idle timeout"):
        api.zone_paint("fresh", p, events=ev, idle_timeout_s=0)
    with pytest.raises(ZoneError, match="no zone set 'nope'"):
        api.zone_paint("fresh", p, events=ev, source="nope")
    with pytest.raises(ValueError, match="cancel"):
        api.zone_paint("fresh", p, events=ev, cancel="yes")
    # a label is a zone's name: one that Space would refuse to save is refused here
    for labels in (["seal #1"], ["ok", "a/b"], ["x" * 33], [7]):
        with pytest.raises(ValueError, match="labels must be zone names"):
            api.zone_paint("fresh", p, events=ev, labels=labels)
    assert zstore.names(s.store, GEO) == ["demo"]


def test_a_polarity_that_cannot_be_measured_refuses_the_session(rig, monkeypatch):
    """Raw white and raw black look the same to the camera: assuming "not inverted" would
    hold a lit field on the part for the whole session on an inverting path. The painter
    does not start; saying the polarity starts it."""
    s, p, painted = rig
    real = type(s).rig
    sent = []

    @contextlib.contextmanager
    def blind(self, preset, zone_mask=None):
        with real(self, preset, zone_mask=zone_mask) as sync:
            shoot = sync.shoot

            def same(pattern, raw=False):
                sent.append(np.asarray(pattern).copy())
                return np.full_like(shoot(pattern, raw=raw), 40.0)
            sync.shoot = same
            yield sync
    monkeypatch.setattr(type(s), "rig", blind)
    with pytest.raises(ZoneError, match="polarity of the HDMI path could not be measured"):
        api.zone_paint("blind", p, events=list(P.DEMO_EVENTS))
    assert len(sent) == 2 and sent[0].max() == 0 and sent[1].min() == 255   # the two probes only
    assert "blind" not in zstore.names(s.store, GEO)
    out = api.zone_paint("blind", p, events=list(P.DEMO_EVENTS), polarity="normal")
    assert out["saved"] is True and out["polarity_inverted"] is False
    zstore.delete(s.store, GEO, "blind", "test")


def test_the_grid_underlay_is_the_grid_last_projected_for_the_geometry(rig, monkeypatch):
    """G draws the cells `--zone C4` means: the spacing of the grid last projected at this
    geometry (96 px until one was), not a fixed 96."""
    s, p, painted = rig
    spacings = []
    real = P.PainterModel

    def spy(*a, **kw):
        spacings.append(kw.get("grid_spacing"))
        return real(*a, **kw)
    monkeypatch.setattr(P, "PainterModel", spy)
    ev = [{"type": "key", "key": "g"}, {"type": "drag", "from": [100, 100], "to": [300, 300]},
          {"type": "key", "key": "space"}]
    out = api.zone_paint("g96", p, events=ev, polarity="normal")
    assert out["grid_spacing_px"] == 96 and spacings == [96]
    api.project_grid(p, 64, "grid64")
    out = api.zone_paint("g64", p, events=ev, polarity="normal")
    assert out["grid_spacing_px"] == 64 and spacings == [96, 64]
    m = real(grid_spacing=64)
    m.on_key("g")
    m.on_key("h")
    img = P.render(m.state)
    assert (img[:, 64] == P.GRID_LEVEL).mean() > 0.9 and (img[:, 96] == P.GRID_LEVEL).mean() < 0.2
    for name in ("g96", "g64"):
        zstore.delete(s.store, GEO, name, "test")


def test_edit_a_saved_set_and_a_session_without_space_saves_nothing(rig):
    s, p, painted = rig
    # start from `demo`, remove Z1, add a rectangle, confirm -> a new set that says where from
    out = api.zone_paint("demo_edit", p, source="demo", polarity="normal", events=[
        {"type": "click", "at": [300, 365]}, {"type": "key", "key": "delete"},
        {"type": "drag", "from": [60, 60], "to": [160, 140]}, {"type": "key", "key": "space"}])
    assert out["saved"] and [z["id"] for z in out["zones"]] == ["Z2", "Z3", "Z4", "Z5"]
    zs = zstore.load(s.store, GEO, "demo_edit")
    assert zs.source == "demo" and zs.bboxes()["Z5"] == [60, 60, 160, 140]
    assert zstore.load(s.store, GEO, "demo").ids == ["Z1", "Z2", "Z3", "Z4"]        # untouched
    # the script ends without Space: cancelled, nothing written
    out = api.zone_paint("never", p, polarity="normal",
                         events=[{"type": "drag", "from": [100, 100], "to": [300, 300]}])
    assert out["saved"] is False and "without Space" in out["reason"] and out["n_zones"] == 1
    # Esc: cancelled too
    out = api.zone_paint("never", p, polarity="inverted", events=[
        {"type": "drag", "from": [100, 100], "to": [300, 300]}, {"type": "key", "key": "escape"}])
    assert out["saved"] is False and "nothing saved" in out["reason"]
    assert out["polarity_inverted"] is True
    assert "never" not in zstore.names(s.store, GEO)
    assert not (painted["lib"] / "zones" / GEO / "never").exists()


def test_a_failed_photo_never_loses_the_painted_set(rig, monkeypatch):
    """The operator confirmed: the set is saved while the rig is still held, and a camera
    problem on the closing photo is a warning, not the loss of the drawing."""
    s, p, painted = rig
    from engine.capture import sync as sync_mod

    def boom(self, pattern, raw=False):
        raise RuntimeError("camera timeout")
    monkeypatch.setattr(sync_mod.FreeRunSync, "shoot", boom)
    out = api.zone_paint("nophoto", p, polarity="normal", events=[
        {"type": "drag", "from": [100, 100], "to": [300, 300]}, {"type": "key", "key": "space"}])
    assert out["saved"] is True and out["camera"] is None and out["n_zones"] == 1
    assert any("no photo" in w and "camera timeout" in w for w in out["warnings"])
    d = painted["lib"] / "zones" / GEO / "nophoto"
    assert (d / "zones.json").is_file() and (d / "masks.npz").is_file()
    assert not (d / "camera.png").exists()
    zstore.delete(s.store, GEO, "nophoto", "test cleanup")


def test_real_rig_runs_only_inside_the_rig_lock_and_never_from_a_script(tmp_path, monkeypatch):
    """REAL mode with the rig lock held elsewhere and every driver stubbed: the painter is
    refused (RigBusy) before a camera or a window is opened, and nothing is saved."""
    from engine import session as session_mod
    from engine.capture import riglock
    touched = []

    def busy(self):
        touched.append("lock asked")
        raise rules.RigBusy("the camera + DMD are in use by another process (test)")

    def hardware(*a, **k):
        touched.append("HARDWARE")
        raise AssertionError("the real rig was opened")
    monkeypatch.setattr(riglock.RigLock, "acquire", busy)
    monkeypatch.setattr(session_mod, "open_camera", hardware)
    monkeypatch.setattr(session_mod, "Projector", hardware)
    s = api.configure(tmp_path / "library", mock=False)
    try:
        p = api.load_preset(PRESET)
        with pytest.raises(ValueError, match="MOCK ONLY"):
            api.zone_paint("lid", p, events=list(P.DEMO_EVENTS))
        assert touched == []                                # refused before the lock was asked
        with pytest.raises(rules.RigBusy, match="in use by another process"):
            api.zone_paint("lid", p)
        assert touched == ["lock asked"]
        assert not (tmp_path / "library" / "zones").exists()
    finally:
        api._session = None


# ------------------------------------------------------------------ zone-show

def test_zone_show_reports_where_each_zone_lands(rig):
    s, p, painted = rig
    r = api.zone_show("demo:Z1,Z2,Z3", p, hold_s=0.5)
    assert r["zoneset"] == "demo" and r["all_seen"] is True and r["polarity_inverted"] is False
    assert [z["id"] for z in r["zones"]] == ["Z1", "Z2", "Z3"] and r["held_s"] == 0.5
    cam_h, cam_w = cv2.imread(str(r["camera_png"]), cv2.IMREAD_GRAYSCALE).shape
    boxes = {}
    for z in r["zones"]:
        assert z["seen"] and z["camera_px"] > 1000 and not z["at_image_edge"]
        x0, y0, x1, y1 = boxes[z["id"]] = z["bbox_px"]
        cx, cy = z["centroid_px"]
        assert 0 <= x0 < cx < x1 <= cam_w and 0 <= y0 < cy < y1 <= cam_h
        assert 0.5 < z["camera_px"] / z["dmd_px"] < 2.0      # about one camera px per mirror here
    # Z1 and Z2 sit side by side on the DMD, Z3 below Z1: the same on the part
    assert boxes["Z1"][2] <= boxes["Z2"][0] and boxes["Z1"][3] <= boxes["Z3"][1]
    for k in ("camera_png", "overlay_png", "dmd_png", "json_path"):
        assert Path(r[k]).is_file(), k
    doc = json.loads(Path(r["json_path"]).read_text(encoding="utf-8"))
    assert doc["kind"] == "zoneshow" and doc["zones"][0]["centroid_px"] == r["zones"][0]["centroid_px"]
    dmd = cv2.imread(str(r["dmd_png"]), cv2.IMREAD_GRAYSCALE)
    assert dmd.shape == DMD_SHAPE and (dmd > 128).mean() < 0.02          # outlines, not a flood
    with pytest.raises(ZoneError, match="no zone 'Z9'"):
        api.zone_show("demo:Z9", p)
    with pytest.raises(ValueError, match="hold_s"):
        api.zone_show("demo", p, hold_s=-1)


# ------------------------------------------------------------------ scan-fast --zones

def test_scan_fast_lights_only_the_zones_and_counts_points_per_zone(rig):
    s, p, painted = rig
    zs = zstore.load(s.store, GEO, "demo")
    union = zs.mask(["Z1", "Z3"])
    lit = [m for m in painted["seen"]["masks"] if m is not None]
    assert len(lit) == 1 and np.array_equal(lit[0], union)             # only those mirrors
    assert not lit[0][zs.zone("Z2").mask()].any() and not lit[0][zs.zone("Z4").mask()].any()
    m = painted["scan_meta"]
    assert m["zone"] == "zoneset:demo:Z1,Z3" and m["fast"] and m["axis"] == "xy"
    z = m["zones"]
    assert z["zoneset"] == "demo" and z["ids"] == ["Z1", "Z3"] and z["axis"] == "xy"
    assert z["names"] == {"Z1": "part", "Z3": ""} and z["scan_id"] == painted["scan_id"]
    assert z["dmd_px"] == {"Z1": int(zs.zone("Z1").mask().sum()), "Z3": int(zs.zone("Z3").mask().sum())}
    c = z["counts"]
    assert set(c) == {"Z1", "Z3"} and c["Z1"] > 20000 and c["Z3"] > 12000
    # the points per zone follow the zones' sizes, and nearly every point is in a zone
    assert c["Z1"] / c["Z3"] == pytest.approx(z["dmd_px"]["Z1"] / z["dmd_px"]["Z3"], rel=0.15)
    assert z["in_zones"] == c["Z1"] + c["Z3"] and z["points"] == m["n_points"]
    assert z["outside"] == z["points"] - z["in_zones"] and z["outside"] < 0.02 * z["points"]
    assert z["warnings"] == []
    # the same record is in the scan's library entry
    ms = _scan(painted)
    doc = json.loads((Path(ms.path) / use.SCAN_FILE).read_text(encoding="utf-8"))
    assert doc["counts"] == c and doc["spec"] == "zoneset:demo:Z1,Z3" and doc["kind"] == "scan_zones"
    # what the camera decoded is what was lit: the decoded DMD px lie in the two zones
    fld = ms.field
    ok = ms.cloud.mask & np.isfinite(fld.corr_x) & np.isfinite(fld.corr_y)
    cols, rows = np.rint(fld.corr_x[ok]).astype(int), np.rint(fld.corr_y[ok]).astype(int)
    assert union[np.clip(rows, 0, 1139), np.clip(cols, 0, 911)].mean() > 0.98
    assert not zs.zone("Z2").mask()[np.clip(rows, 0, 1139), np.clip(cols, 0, 911)].any()
    with np.load(Path(ms.path) / "flood.npz") as f:           # and the flood lit only them
        lit_px = int(((f["white"] - f["black"]) > 20).sum())
    assert 0.8 * z["points"] < lit_px < 1.3 * z["points"]


def test_columns_only_counts_never_guess(rig):
    s, p, painted = rig
    zs = zstore.load(s.store, GEO, "demo")
    ms = _scan(painted)
    fld = ms.field
    xy = use.count_points(zs, ["Z1", "Z3"], fld.corr_x, fld.corr_y, ms.cloud.mask)
    assert xy["counts"] == painted["scan_meta"]["zones"]["counts"]
    # Z1 (cols 180-420) and Z3 (cols 240-460) share columns: by columns alone they are unknown
    x = use.count_points(zs, ["Z1", "Z3"], fld.corr_x, None, ms.cloud.mask)
    assert x["axis"] == "x" and x["counts"] == {"Z1": None, "Z3": None}
    assert len(x["warnings"]) == 2 and "--axis xy" in x["warnings"][0]
    assert x["in_zones"] >= xy["in_zones"]
    # Z1 and Z2 (cols 507-773) share none: each is counted by its columns
    x = use.count_points(zs, ["Z1", "Z2"], fld.corr_x, None, ms.cloud.mask)
    assert x["counts"]["Z1"] is not None and x["counts"]["Z2"] == 0
    assert any("zone Z2: no point decoded" in w for w in x["warnings"])
    assert x["counts"]["Z1"] >= xy["counts"]["Z1"]             # Z3's points in Z1's columns too
    # a zone nothing decoded in says so (never a silent 0 that looks measured)
    none = use.count_points(zs, ["Z4"], fld.corr_x, fld.corr_y, ms.cloud.mask)
    assert none["counts"] == {"Z4": 0} and "no point decoded" in none["warnings"][0]


def test_zone_set_text_is_a_zone_everywhere(rig, monkeypatch):
    s, p, painted = rig
    calls = []

    def fake(sess, preset, zones, **kw):
        calls.append((zones, kw))
        return "scan"
    monkeypatch.setattr(use, "scan_fast", fake)
    assert api.scan_fast("zoneset:demo:Z1", p, "a") == "scan"            # given as the zone
    assert api.scan_fast(None, p, "b", zones="demo", axis="xy") == "scan"
    assert [c[0] for c in calls] == ["zoneset:demo:Z1", "demo"]
    assert calls[0][1]["zone"] is None and calls[1][1]["axis"] == "xy" and calls[0][1]["label"] == "a"
    monkeypatch.undo()
    with pytest.raises(ValueError, match="not both"):
        api.scan_fast("F10:J12", p, "c", zones="demo")
    with pytest.raises(ZoneError, match="no zone 'Z9'"):
        api.scan_fast(None, p, "c", zones="demo:Z9")           # before the rig
    with pytest.raises(ZoneError, match="no zone set 'nope'"):
        api.scan_fast(None, p, "c", zones="nope")
    assert len([m for m in api.list_library("metric")]) == 1    # nothing was scanned


# ------------------------------------------------------------------ Measure mode: a zone as the region

def test_measure_takes_a_zone_set_as_its_region(rig, tmp_path):
    s, p, painted = rig
    ms = _scan(painted)
    counts = painted["scan_meta"]["zones"]["counts"]
    out = api.measure_run("flatness", p, zone="zoneset:demo:Z1", scans=[ms],
                          runs_dir=tmp_path / "runs")
    doc = out["result"]
    assert doc["status"] == "MEASURED" and out["exit_code"] == 0, doc
    n = {v["name"]: v["n_points"] for v in doc["values"]}
    assert all(abs(k - counts["Z1"]) <= 0.02 * counts["Z1"] for k in n.values()), (n, counts)
    # two zones: each one is measured on its own (one plane per zone, named set/id) - a
    # plane is never fitted across separate zones
    both = api.measure_run("flatness", p, zone="zoneset:demo:Z1,Z3", scans=[ms],
                           runs_dir=tmp_path / "runs")["result"]
    assert both["status"] == "MEASURED", both
    assert [v["name"] for v in both["values"]] == [
        f"demo/{z}:{k}" for z in ("Z1", "Z3")
        for k in ("flatness_rms", "flatness_p2_p98", "tilt", "plane_distance")]
    v0 = both["values"][0]
    assert v0["zone"] == "demo/Z1" and v0["label"] == "demo/Z1 Flatness (RMS)"
    assert v0["n_points"] == n["flatness_rms"]                  # the same px as Z1 alone
    assert abs(both["values"][4]["n_points"] - counts["Z3"]) <= 0.02 * counts["Z3"]
    zsel = both["zone_selection"]
    assert zsel["text"] == "zoneset:demo:Z1,Z3" and zsel["source"] == "argument"
    assert [z["key"] for z in zsel["by_zone"]] == ["demo/Z1", "demo/Z3"]
    assert all(z["coverage_pct"] > 90 for z in zsel["by_zone"])
    # a zone nothing of this scan decoded into: refused, never the whole field instead
    bad = api.measure_run("flatness", p, zone="zoneset:demo:Z4", scans=[ms],
                          runs_dir=tmp_path / "runs")
    assert bad["result"]["status"] == "NOT_MEASURABLE" and bad["exit_code"] != 0
    assert "no camera px of this scan decoded into its zones" in bad["result"]["message"]
    # the region from the scan's own decode; a columns-only scan lit through other zones: refused
    region = use.camera_region(s, p, "zoneset:demo:Z3", ms, ms.cloud.mask.shape, [], "zoneset:demo:Z1,Z3")
    assert region.shape == ms.cloud.mask.shape and abs(int((region & ms.cloud.mask).sum()) - counts["Z3"]) \
        <= 0.02 * counts["Z3"]

    class XOnly:
        field = type("F", (), {"corr_x": ms.field.corr_x, "corr_y": None})()
    warnings = []
    assert use.camera_region(s, p, "zoneset:demo:Z1", XOnly, ms.cloud.mask.shape, warnings, "?") is None
    assert "columns only" in warnings[0]
    assert use.camera_region(s, p, "zoneset:demo:Z1", XOnly, ms.cloud.mask.shape, [],
                             "zoneset:demo:Z1") is None
    with pytest.raises(ValueError, match="columns only and was lit through zoneset:demo:Z1,Z3"):
        use.camera_region(s, p, "zoneset:demo:Z1", XOnly, ms.cloud.mask.shape, [],
                          "zoneset:demo:Z1,Z3")
    with pytest.raises(ValueError, match="both axes"):
        use.camera_region(s, p, "zoneset:demo:Z1", XOnly, ms.cloud.mask.shape, [], "?", need=True)


# ------------------------------------------------------------------ the CLI

def run_cli(lib, *argv, mock=True):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main([*(["--mock"] if mock else []), "--library", str(lib), "--preset", PRESET,
                         *argv])
    api._session = None
    assert "Traceback" not in out.getvalue() + err.getvalue(), err.getvalue()
    return code, out.getvalue(), err.getvalue()


def test_cli_zone_paint_zones_show_delete(painted, tmp_path):
    lib = painted["lib"]
    script = tmp_path / "session.json"
    script.write_text(json.dumps([
        {"type": "drag", "from": [100, 150], "to": [330, 420]},
        {"type": "key", "key": "e"}, {"type": "drag", "from": [500, 600], "to": [760, 820]},
        {"type": "key", "key": "space"}]))
    code, out, err = run_cli(lib, "zone-paint", "--name", "cli_set", "--events", str(script),
                             "--note", "from the test")
    assert code == 0, err
    assert "[MOCK]" in out and "Zone set cli_set saved: 2 zone(s)" in out and "Z2" in out
    assert "scan-fast LABEL --zones cli_set" in out
    assert json.loads((lib / "zones" / GEO / "cli_set" / "zones.json").read_text())["note"] == \
        "from the test"
    # the same name again: refused (2) before anything is drawn; --replace takes it
    code, out, err = run_cli(lib, "zone-paint", "--name", "cli_set", "--events", "demo")
    assert code == cli.EXIT_REFUSED and "--replace" in err
    code, out, err = run_cli(lib, "--json", "zone-paint", "--name", "cli_set", "--events", "demo",
                             "--replace", "--polarity", "normal")
    doc = json.loads(out)
    assert code == 0 and doc["ok"] and doc["zone_paint"]["n_zones"] == 4, err
    assert (lib / "trash" / "zones" / GEO / "cli_set" / "zones.json").is_file()
    # a session that is not confirmed: exit 1, nothing saved
    script.write_text(json.dumps([{"type": "drag", "from": [100, 150], "to": [330, 420]},
                                  {"type": "key", "key": "escape"}]))
    code, out, err = run_cli(lib, "zone-paint", "--name", "unsaved", "--events", str(script),
                             "--polarity", "normal")
    assert code == cli.EXIT_NOT_OK and "NOT saved" in out
    # refused before the rig: mock without a script, a script on the real rig, a bad file
    code, out, err = run_cli(lib, "zone-paint", "--name", "x")
    assert code == cli.EXIT_REFUSED and "--events" in err
    code, out, err = run_cli(lib, "zone-paint", "--name", "x", "--events", "demo", mock=False)
    assert code == cli.EXIT_REFUSED and "MOCK ONLY" in err
    code, out, err = run_cli(lib, "zone-paint", "--name", "x", "--events", str(tmp_path / "no.json"))
    assert code == cli.EXIT_REFUSED and "events file" in err
    # zones: the list, one set, its recipe form, its scan mask
    code, out, err = run_cli(lib, "zones")
    assert code == 0 and "cli_set" in out and "demo" in out and "Z1 (part)" in out and "*" in out
    assert "unsaved" not in out
    npy = tmp_path / "union.npy"
    code, out, err = run_cli(lib, "--json", "zones", "demo:Z1,Z3", "--by-ref", "--mask-out", str(npy))
    doc = json.loads(out)
    assert code == 0 and [z["id"] for z in doc["zone_set"]["zones"]] == ["Z1", "Z3"], err
    assert doc["spec"] == "zoneset:demo:Z1,Z3"
    assert [z["ref"] for z in doc["recipe_zones"]] == ["zoneset:demo/Z1", "zoneset:demo/Z3"]
    assert np.load(npy).shape == DMD_SHAPE and Path(doc["mask"]) == npy
    code, out, err = run_cli(lib, "zones", "demo", "--mask-out", str(npy))
    assert code == cli.EXIT_REFUSED and "never overwritten" in err
    code, out, err = run_cli(lib, "zones", "demo", "--recipe")
    assert code == 0 and '"shape": "rect"' in out and '"units": "projector_px"' in out
    code, out, err = run_cli(lib, "zones", "nope")
    assert code == cli.EXIT_REFUSED and "no zone set 'nope'" in err
    # zone-show: exit 0 when every zone lands in the camera image
    code, out, err = run_cli(lib, "--json", "zone-show", "demo:Z1", "--hold", "0")
    doc = json.loads(out)
    assert code == 0 and doc["zone_show"]["zones"][0]["seen"] is True, err
    assert any(o["kind"] == "previews" for o in doc["outputs_written"])
    # delete: to the trash, with the reason
    code, out, err = run_cli(lib, "zone-delete", "cli_set")
    assert code == cli.EXIT_REFUSED                         # argparse: --reason is required
    code, out, err = run_cli(lib, "zone-delete", "cli_set", "--reason", "test over")
    assert code == 0 and "moved to the library trash" in out, err
    assert (lib / "trash" / "zones" / GEO / "cli_set-1" / "TRASHED.json").is_file()
    assert not (lib / "zones" / GEO / "cli_set").exists()


def test_cli_scan_fast_and_measure_take_zones(painted, monkeypatch, tmp_path):
    lib = painted["lib"]
    s = api.configure(lib, mock=True)
    api.load_preset(PRESET)
    ms = _scan(painted)
    api._session = None
    ms.meta["zones"] = painted["scan_meta"]["zones"]
    calls = []

    def fake(zone, preset, label, steps, periods, average, hold, axis, faces, margin, **kw):
        calls.append({"zone": zone, "axis": axis, **kw})
        return ms
    monkeypatch.setattr(api, "scan_fast", fake)
    code, out, err = run_cli(lib, "scan-fast", "part", "--zones", "demo:Z1,Z3")
    assert code == 0, err
    assert calls[-1] == {"zone": None, "axis": "xy", "zones": "zoneset:demo:Z1,Z3"}
    assert "zone zoneset:demo:Z1,Z3" in out and "(outside)" in out and "part" in out
    counts = painted["scan_meta"]["zones"]["counts"]
    assert str(counts["Z1"]) in out and str(counts["Z3"]) in out
    code, out, err = run_cli(lib, "--json", "scan-fast", "part", "--zones", "demo", "--axis", "x")
    assert code == 0 and calls[-1]["axis"] == "x" and calls[-1]["zones"] == "zoneset:demo"
    assert json.loads(out)["zones"]["counts"] == counts
    code, out, err = run_cli(lib, "scan-fast", "part")       # no zone set: columns only, as before
    assert code == 0 and calls[-1] == {"zone": None, "axis": "x"} and "(outside)" not in out
    n = len(calls)
    for bad in (["--zones", "demo", "--zone", "F10:J12"], ["--zones", "de mo"],
                ["--zones", "demo:"]):
        code, out, err = run_cli(lib, "scan-fast", "part", *bad)
        assert code == cli.EXIT_REFUSED, bad
    assert len(calls) == n
    monkeypatch.undo()
    # Measure mode: the zone set is the region (here on the stored scan: nothing is captured)
    code, out, err = run_cli(lib, "--json", "measure-run", PRESET, "--what", "flatness",
                             "--zones", "demo:Z1", "--scan", painted["scan_id"],
                             "--runs-dir", str(tmp_path / "runs"))
    doc = json.loads(out)
    assert code == 0 and doc["status"] == "MEASURED", (err, doc.get("result", {}).get("message"))
    spec = json.loads((Path(doc["run_dir"]) / "spec.json").read_text(encoding="utf-8"))
    assert spec["zone"] == "zoneset:demo:Z1"
    assert abs(doc["values"][0]["n_points"] - counts["Z1"]) <= 0.02 * counts["Z1"]
    code, out, err = run_cli(lib, "measure-run", PRESET, "--what", "flatness", "--zones", "demo",
                             "--zone", "F10", "--scan", painted["scan_id"])
    assert code == cli.EXIT_REFUSED and "not both" in err
    # the help names the workflow
    ap = cli.build_parser()
    a = ap.parse_args(["zone-paint", "--name", "lid", "--from", "old", "--replace"])
    assert a.func is cli.cmd_zone_paint and a.source == "old" and a.polarity == "measure"
    assert ap.parse_args(["zone-show", "lid:Z1"]).hold == 5.0
    assert "zone-paint --name lid" in cli.EPILOG and "--zones lid" in cli.EPILOG


def _second_set(painted):
    """A second set at the painted geometry whose ids repeat `demo`'s (Z1)."""
    from engine import config
    from engine.zones import Zone, ZoneSet
    s = api.configure(painted["lib"], mock=True)
    p = api.load_preset(PRESET)
    if not zstore.exists(s.store, GEO, "band"):
        zstore.save(s.store, ZoneSet(name="band", geometry_id=GEO, preset=p.name, mock=True,
                                     fingerprint=config.geometry_fingerprint(p),
                                     zones=[Zone.rect("Z1", 600, 480, 720, 580, name="edge")]))
    api._session = None


def test_cli_several_sets_in_one_scan_measurement_and_quick_check(painted, monkeypatch, tmp_path):
    lib = painted["lib"]
    _second_set(painted)
    s = api.configure(lib, mock=True)
    api.load_preset(PRESET)
    ms = _scan(painted)
    ms.cloud                                                # loaded once
    api._session = None
    ms.meta["zones"] = painted["scan_meta"]["zones"]
    calls = []

    def fake(zone, preset, label, steps, periods, average, *a, **kw):
        calls.append({"zone": zone, **kw})
        return ms
    monkeypatch.setattr(api, "scan_fast", fake)
    monkeypatch.setattr(cli, "_mock_place_metric", lambda ctx: None)
    # --zones may be repeated, or join sets with '+': one zone text, both axes
    for argv in (["--zones", "demo:Z1,Z3", "--zones", "band"], ["--zones", "demo:Z1,Z3+band"]):
        code, out, err = run_cli(lib, "scan-fast", "part", *argv)
        assert code == 0 and calls[-1]["zones"] == "zoneset:demo:Z1,Z3+band", err
        assert calls[-1]["zone"] is None
    n = len(calls)
    for bad, msg in ((["--zones", "demo", "--zones", "demo:Z1"], "named twice"),
                     (["--zones", "demo+"], "empty zone set"),
                     (["--zones", "selection", "--zones", "band"], "stands alone")):
        code, out, err = run_cli(lib, "scan-fast", "part", *bad)
        assert code == cli.EXIT_REFUSED and msg in err, (bad, err)
    assert len(calls) == n
    # Measure: two zones are two planes - each measured on its own, named set/id
    code, out, err = run_cli(lib, "--json", "measure-run", PRESET, "--what", "flatness",
                             "--zones", "demo:Z1", "--zones", "band", "--scan", painted["scan_id"],
                             "--runs-dir", str(tmp_path / "runs"))
    doc = json.loads(out)
    assert code == 5 and doc["status"] == "NOT_MEASURABLE", err      # band was not lit in that scan
    assert doc["reason"] == "too_few_points" and "zone band/Z1 (edge): no point decoded in it" in \
        doc["result"]["message"]
    code, out, err = run_cli(lib, "--json", "measure-run", PRESET, "--what", "flatness",
                             "--zones", "demo:Z1,Z3", "--scan", painted["scan_id"],
                             "--runs-dir", str(tmp_path / "runs"))
    doc = json.loads(out)
    assert code == 0 and [v["name"] for v in doc["values"]][::4] == ["demo/Z1:flatness_rms",
                                                                   "demo/Z3:flatness_rms"], err
    assert json.loads((Path(doc["run_dir"]) / "spec.json").read_text(encoding="utf-8"))["zone"] \
        == "zoneset:demo:Z1,Z3"
    code, out, err = run_cli(lib, "measure-run", PRESET, "--what", "flatness", "--zones", "demo:Z1,Z3",
                             "--scan", painted["scan_id"], "--runs-dir", str(tmp_path / "runs"))
    assert code == 0 and "demo/Z1 Flatness (RMS)" in out and "demo/Z3 Tilt" in out and "seen %" in out
    # Quick check: the limit applies to each zone (judged on its own px, not the whole image)
    code, out, err = run_cli(lib, "--json", "quick-check", PRESET, "--check", "flatness",
                             "--threshold", "0.5", "--zones", "demo:Z1", "--zones", "demo2:Z1",
                             "--runs-dir", str(tmp_path / "q"))
    doc = json.loads(out)
    assert code == 5 and "no zone set 'demo2'" in doc["result"]["invalid_reason"], err
    assert doc["result"]["disposition"] == "PULL" and len(calls) == n       # nothing was scanned
    code, out, err = run_cli(lib, "--json", "quick-check", PRESET, "--check", "flatness",
                             "--threshold", "100", "--zones", "demo:Z1,Z3",
                             "--runs-dir", str(tmp_path / "q"))
    res = json.loads(out)["result"]
    assert code == 0 and res["result"] == "PASS" and res["disposition"] == "CONTINUE", (err, res)
    assert calls[-1]["zone"] == "zoneset:demo:Z1,Z3" and calls[-1]["axis"] == "xy"
    z = res["zone_selection"]
    assert [(b["key"], b["verdict"]) for b in z["by_zone"]] == [("demo/Z1", "PASS"), ("demo/Z3", "PASS")]
    assert all(b["coverage_pct"] > 90 for b in z["by_zone"]) and z["text"] == "zoneset:demo:Z1,Z3"
    assert res["deviation_mm"]["method"] == "plane_fit_per_zone" and z["failed"] == []
    # a limit no surface holds: every zone fails on its own, and each is named
    code, out, err = run_cli(lib, "quick-check", PRESET, "--check", "flatness", "--threshold",
                             "0.0001", "--zones", "demo:Z1,Z3", "--runs-dir", str(tmp_path / "q"))
    assert code == 1 and "FAIL" in out and "demo/Z1 FAIL" in out and "demo/Z3 FAIL" in out, out
    # the help names the several-set forms
    ap = cli.build_parser()
    assert ap.parse_args(["scan-fast", "x", "--zones", "a", "--zones", "b:Z1"]).zones == ["a", "b:Z1"]
    assert ap.parse_args(["quick-check", "--check", "flatness", "--zones", "selection"]).zones == ["selection"]
    assert "--zones selection" in cli.EPILOG and "zone-select" in cli.EPILOG


def test_cli_zone_select_prints_sets_switches_and_clears(painted, tmp_path):
    lib = painted["lib"]
    _second_set(painted)
    sel_file = lib / "zones" / GEO / "_selection.jsonl"
    sel_file.unlink(missing_ok=True)
    try:
        code, out, err = run_cli(lib, "zone-select")
        assert code == 0 and "whole field" in out and "nothing is switched on" in out, err
        code, out, err = run_cli(lib, "--json", "zone-select", "demo:Z1,Z3", "band")
        st = json.loads(out)["selection"]
        assert code == 0 and st["label"] == "band (1), demo (2 of 4)" and st["by"] == "command line"
        assert st["text"] == "zoneset:band+demo:Z1,Z3"
        code, out, err = run_cli(lib, "zone-select")
        assert code == 0 and "band (1), demo (2 of 4)" in out and "Z1, Z3" in out and "ON" in out
        assert "--zones selection" in out
        code, out, err = run_cli(lib, "zones")                  # the listing marks what is on
        assert code == 0 and "ON (2 of 4)" in out and "selected at this rig placement" in out
        code, out, err = run_cli(lib, "--json", "zone-select", "--off", "band", "--on", "demo:Z2")
        assert code == 0 and json.loads(out)["selection"]["label"] == "demo (1 of 4)"
        # --zones selection: what a scan through the selection would light
        code, out, err = run_cli(lib, "--json", "zone-show", "selection", "--hold", "0")
        assert code == 0 and [z["key"] for z in json.loads(out)["zone_show"]["zones"]] == ["demo/Z2"]
        # refused: nothing is written
        for bad, msg in ((["nope"], "no zone set 'nope'"), (["--on", "demo:Z9"], "no zone 'Z9'"),
                         (["--clear", "demo"], "not with sets to switch on")):
            code, out, err = run_cli(lib, "zone-select", *bad)
            assert code == cli.EXIT_REFUSED and msg in err, (bad, err)
        code, out, err = run_cli(lib, "--json", "zone-select")
        assert json.loads(out)["selection"]["label"] == "demo (1 of 4)"
        # a set that is switched on and gone: exit 1, and every run that asks for it is refused
        code, out, err = run_cli(lib, "zone-select", "--on", "band")
        assert code == 0
        code, out, err = run_cli(lib, "zone-delete", "band", "--reason", "test")
        assert code == cli.EXIT_REFUSED and "switched on in the zone selection" in err
        code, out, err = run_cli(lib, "zone-delete", "band", "--reason", "test", "--force")
        assert code == 0 and "the zone selection now names a set that is gone" in out
        code, out, err = run_cli(lib, "zone-select")
        assert code == cli.EXIT_NOT_OK and "NOT FOUND" in out and "refused until this is fixed" in out + err
        code, out, err = run_cli(lib, "scan-fast", "part", "--zones", "selection")
        assert code == cli.EXIT_REFUSED and "cannot be used" in err and "zone-select --off band" in err
        code, out, err = run_cli(lib, "--json", "measure-run", PRESET, "--what", "flatness",
                                 "--zones", "selection", "--runs-dir", str(tmp_path / "runs"))
        doc = json.loads(out)
        assert code == cli.EXIT_REFUSED and doc["status"] == "NOT_MEASURABLE" and doc["reason"] == "refused"
        code, out, err = run_cli(lib, "zone-select", "--clear")
        assert code == 0 and "whole field" in out
    finally:
        sel_file.unlink(missing_ok=True)

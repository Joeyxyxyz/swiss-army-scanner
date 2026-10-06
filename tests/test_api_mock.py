"""End-to-end mock runs of every verb (build step 1 definition of done)."""

import json

import numpy as np
import pytest

from engine import api
from engine.results import CADRef, ToleranceBand

from conftest import PRESET, Box, VALIDATION_OBJECT

V = {"H": 106.95, "W": 56.50, "L": 21.90}     # §11 validation object


def scan(preset, base, label, *objs, known=None):
    bench = api.mock_bench()
    bench.clear()
    for o in objs:
        bench.place(o)
    return api.capture_object(preset, base, label, known_dims=known)


def test_check_signal_reports_usable_signal(preset):
    sig = api.check_signal()
    assert sig.mock and sig.geometry_id == "mock-wall_2p7ft_vstripe_v1"
    assert sig.gc_conf > 30 and sig.quality in ("good", "excellent")
    assert 20 < sig.valid_pct < 90
    assert sig.focus_hint == "sharp" and sig.ok


def test_base_object_relief(preset):
    base = api.capture_base(preset, "empty wall")
    assert base.geometry_id == preset.geometry_id
    assert (base.path / "field.npz").exists() and (base.path / "meta.json").exists()
    obj = scan(preset, base, "validation", VALIDATION_OBJECT, known=V)
    assert obj.base_id == base.id and obj.known_dims == V
    blob = obj.relief[np.isfinite(obj.relief)]
    # §11: the bench scan's relief range is about ±48 px
    assert 45 < np.nanpercentile(obj.relief, 99.9) < 51
    assert abs(np.median(blob)) < 0.5                  # base plane is the datum
    assert json.loads((obj.path / "known_dims.json").read_text())["H"] == 106.95


def test_measure_without_calibration_stays_in_px(preset):
    base = api.capture_base(preset, "empty")
    obj = scan(preset, base, "v", VALIDATION_OBJECT, known=V)
    d = api.measure(obj)
    assert d.units == "px" and d.tier == "relief"
    assert any("rule 6" in w for w in d.warnings)
    assert d.H == pytest.approx(106.95 * 0.45, abs=0.3)   # raw relief px
    assert d.error is None or d.error == {}


def test_calibrate_then_measure_in_mm(preset):
    base = api.capture_base(preset, "empty")
    truth = api.mock_bench().truth
    v = scan(preset, base, "validation", VALIDATION_OBJECT, known=V)
    b2 = scan(preset, base, "block40", Box(40.0, 50.0, 30.0, center_mm=(10, 20), angle_deg=20),
              known={"H": 40.0, "W": 50.0, "L": 30.0})
    cal = api.calibrate(preset, [v, b2])
    assert cal.geometry_id == preset.geometry_id
    assert cal.mm_per_px == pytest.approx(truth["mm_per_relief_px"], rel=0.01)
    assert cal.lateral_px_per_mm == pytest.approx(truth["cam_px_per_mm"], rel=0.03)
    assert cal.r2 > 0.999
    assert (cal.path / "depth_calibration.json").exists()

    # §11: the engine must dimension the validation object
    d = api.measure(v)
    assert d.units == "mm" and d.tier == "lateral_calibrated"
    assert d.H == pytest.approx(V["H"], abs=0.7)
    assert d.W == pytest.approx(V["W"], abs=1.0)
    assert d.L == pytest.approx(V["L"], abs=1.0)
    assert set(d.error) == {"H", "W", "L"}

    # ...and an object it was NOT calibrated on, rotated
    part = scan(preset, base, "part", Box(25.0, 45.0, 35.0, center_mm=(-40, -30), angle_deg=-35),
                known={"H": 25.0, "W": 45.0, "L": 35.0})
    d = api.measure(part)
    assert abs(d.error["H"]) < 0.7
    assert abs(d.error["W"]) < 1.5 and abs(d.error["L"]) < 1.5
    outs = api.list_library("outputs", "measurements")
    assert any(o["label"] == "part" for o in outs)
    api.measure(part)                                     # second measure appends
    assert len([o for o in api.list_library("outputs", "measurements") if o["label"] == "part"]) == 2
    assert (part.path / "exports.jsonl").read_text().count('"kind": "measurements"') == 2


def test_single_sample_calibration_through_origin(preset):
    base = api.capture_base(preset, "empty")
    v = scan(preset, base, "v", VALIDATION_OBJECT, known={"H": 106.95})
    cal = api.calibrate(preset, [v])
    assert cal.r2 is None and cal.intercept_mm == 0.0 and cal.lateral_px_per_mm is None
    d = api.measure(v)
    assert d.units == "mixed" and d.unit_by_dim == {"H": "mm", "W": "px", "L": "px"}


def test_calibration_binds_on_reload(preset):
    base = api.capture_base(preset, "empty")
    v = scan(preset, base, "v", VALIDATION_OBJECT, known=V)
    cal = api.calibrate(preset, [v])
    again = api.load_preset(PRESET)
    assert again.calibration.id == cal.id and again.calibration.source == "library"


def test_reconstruct_writes_pointcloud(preset):
    base = api.capture_base(preset, "empty")
    v = scan(preset, base, "v", VALIDATION_OBJECT, known=V)
    ply = api.reconstruct(v)
    head = ply.read_text().split("end_header")[0]
    assert "units xy=px z=px" in head
    api.calibrate(preset, [v])
    ply2 = api.reconstruct(v, fmt="xyz")
    assert ply2 != ply and "z=mm" in ply2.read_text().splitlines()[0]
    pts = np.loadtxt(ply2, comments="#")
    top = pts[pts[:, 2] > 50, 2]
    assert np.median(top) == pytest.approx(106.95, abs=0.7)


def test_compare_pass_and_fail(preset):
    base = api.capture_base(preset, "empty")
    part = Box(30.0, 40.0, 30.0)
    cal_ref = scan(preset, base, "cal", VALIDATION_OBJECT, known=V)
    api.calibrate(preset, [cal_ref])
    golden = scan(preset, base, "golden", part)
    good = scan(preset, base, "good", part)
    bad = scan(preset, base, "bad", part, Box(5.0, 10.0, 10.0, center_mm=(60, 60)))

    v_ok = api.compare(good, golden)
    assert v_ok.result == "PASS" and v_ok.units == "mm", v_ok
    v_bad = api.compare(bad, golden)
    assert v_bad.result == "FAIL" and v_bad.n_fail_px > 500
    assert v_bad.max_abs_dev == pytest.approx(5.0, abs=0.5)
    with pytest.raises(NotImplementedError):
        api.compare(good, CADRef(path=base.path))


def test_compare_uncalibrated_is_indeterminate_without_px_band(preset):
    base = api.capture_base(preset, "empty")
    g = scan(preset, base, "g", Box(30, 40, 30))
    s = scan(preset, base, "s", Box(30, 40, 30))
    v = api.compare(s, g)
    assert v.result == "INDETERMINATE" and v.units == "px"
    v = api.compare(s, g, ToleranceBand(fail_px=1.0))
    assert v.result == "PASS"


def test_list_library_and_lazy_handles(preset):
    base = api.capture_base(preset, "empty")
    v = scan(preset, base, "v", VALIDATION_OBJECT, known=V)
    api.calibrate(preset, [v])
    bases = api.list_library("bases")
    objs = api.list_library("objects")
    cals = api.list_library("calibrations")
    assert [b.id for b in bases] == [base.id]
    assert [o.id for o in objs] == [v.id] and len(cals) == 1
    # handles from the library are usable directly by other verbs
    again = api.capture_object(preset, bases[0], "from-listing")
    assert again.base_id == base.id
    assert np.allclose(objs[0].relief, v.relief, equal_nan=True)
    assert api.measure(objs[0]).units == "mm"


def test_session_log_records_every_verb(eng, preset):
    api.capture_base(preset, "empty")
    with pytest.raises(ValueError):
        api.list_library("nope")
    lines = [json.loads(l) for l in eng.log_path.read_text().splitlines()]
    verbs = [l["verb"] for l in lines]
    assert verbs == ["load_preset", "capture_base", "list_library"]
    assert lines[-1]["ok"] is False and "ValueError" in lines[-1]["error"]
    assert all(l["args"]["mock"] for l in lines)


def test_mock_capture_decodes_projected_patterns(preset):
    base = api.capture_base(preset, "empty")
    assert base.field.basis_names == ["graycode", "phaseshift"]
    assert base.field.meta["gc_conf"] > 30
    assert base.field.meta["polarity_flipped"]          # like the bench


def test_flipped_polarity_decodes_without_period_jumps(preset):
    """Regression for the v10 bug: inverted fringes need a pi phase shift."""
    base = api.capture_base(preset, "empty")
    f = base.field
    truth = api.mock_bench()._scene()[3]
    err = f.corr_x[f.mask] - truth[f.mask]
    assert np.mean(np.abs(err) > 6) < 0.01, np.percentile(np.abs(err), [50, 99])


def test_mock_bench_is_mock_only(tmp_path):
    api.configure(tmp_path, mock=False)
    try:
        with pytest.raises(RuntimeError, match="human-only"):
            api.mock_bench()
    finally:
        api._session = None


def test_check_signal_flags_fringes_parallel_to_baseline(eng):
    """The bench mistake: rotate90 made the fringes run along the baseline."""
    p = api.load_preset("cell_wall_2p7ft")          # rotate90: true, projector right
    p.camera.average = 1
    sig = api.check_signal()
    assert sig.coded_axis == "y" and sig.axis_ok is False and not sig.ok
    assert any("rotate90" in w for w in sig.warnings)


def test_check_signal_axis_ok_for_vertical_fringes(preset):
    preset.camera.average = 1
    sig = api.check_signal()
    assert sig.coded_axis == "x" and sig.axis_ok is True


def test_output_bin_structure_and_trash(preset):
    preset.camera.average = 1
    base = api.capture_base(preset, "empty")
    v = scan(preset, base, "v", VALIDATION_OBJECT, known=V)
    api.calibrate(preset, [v])
    ply = api.reconstruct(v)
    api.measure(v)
    api.inspect(v)
    P = api.paths()
    assert (P["library"] / "README.md").exists()
    assert ply.parent == P["output"] / "pointclouds"
    kinds = {o["kind"] for o in api.list_library("outputs")}
    assert {"pointclouds", "measurements", "defects", "previews"} <= kinds
    for o in api.list_library("outputs"):
        assert (P["library"] / o["file"]).exists(), o
    # delete = move to trash (recoverable); the listing no longer shows it
    dest = api.delete(v, reason="test")
    assert dest.exists() and (dest / "TRASHED.json").exists() and not v.path.exists()
    assert v.id not in [o.id for o in api.list_library("objects")]
    api.delete(base)
    assert api.list_library("bases") == []


def test_reject_base_is_skipped_by_latest_base(preset):
    preset.camera.average = 1
    a = api.capture_base(preset, "a")
    b = api.capture_base(preset, "b")
    assert api.latest_base().id == b.id
    api.reject_base(b, "bumped")
    assert api.latest_base().id == a.id
    assert [x.meta.get("rejected") for x in api.list_library("bases")] == [None, "bumped"]


# ------------------------------------------------ review fixes (core area)

from engine import config, rules                                   # noqa: E402
from engine.library.store import LibraryStore                      # noqa: E402
from engine.results import Calibration                             # noqa: E402


def _strict_json(text):
    def refuse(c):
        raise ValueError(f"non-standard JSON token {c}")
    return json.loads(text, parse_constant=refuse)


def test_calibration_trashed_elsewhere_stops_giving_mm(eng, preset):
    """Finding: a library calibration trashed by another process (a CLI next
    to the UI) kept producing mm through the in-memory preset fallback."""
    preset.camera.average = 1
    base = api.capture_base(preset, "empty")
    v = scan(preset, base, "v", VALIDATION_OBJECT, known=V)
    cal = api.calibrate(preset, [v])
    assert api.measure(v).units == "mm"
    LibraryStore(eng.store.root).trash(cal.path, "bad fit")       # "another process"
    assert api.calibration_for(preset.geometry_id) is None
    d = api.measure(v)
    assert d.units == "px" and any("rule 6" in w for w in d.warnings)
    assert preset.calibration is None                              # rebound, not stale
    # a calibration that is the preset's OWN values still applies
    preset.calibration = Calibration(id="preset:x", geometry_id=preset.geometry_id,
                                     mm_per_px=2.2, source="preset")
    assert api.calibration_for(preset.geometry_id).id == "preset:x"


def test_capture_object_refuses_base_from_changed_geometry_settings(eng, preset):
    """Finding: rules 5/6 compared only geometry_id strings, so a preset
    edited in place (rotate90 / rig) reused old bases and calibrations."""
    preset.camera.average = 1
    base = api.capture_base(preset, "empty")
    v = scan(preset, base, "v", VALIDATION_OBJECT, known=V)
    api.calibrate(preset, [v])
    assert base.meta["geometry_fp"] == config.geometry_fingerprint(preset)
    preset.rig.standoff_mm = 1200.0                # "moved the rig, edited the preset in place"
    assert api.latest_base() is None
    with pytest.raises(rules.GeometryMismatch, match="standoff_mm"):
        api.capture_object(preset, base, "after-move")
    with pytest.raises(rules.GeometryMismatch):
        api.calibrate(preset, [v])
    assert api.calibration_for(preset.geometry_id, config.geometry_fingerprint(preset)) is None
    assert api.measure(v).units == "mm"            # scans from before still use their fit
    preset.rig.standoff_mm = 823.0
    assert api.latest_base().id == base.id


def test_axis_y_preset_scans_along_corr_y(eng):
    """Finding: axis 'y' presets loaded but capture_object crashed (relief
    was always taken along corr_x)."""
    raw = json.loads(config.resolve_preset_path(PRESET).read_text())
    raw["name"], raw["rig"]["projector_side"] = "above_y", "above"
    for b in raw["bases"]:
        b["axis"] = "y"
    raw["bases"][0]["coarse_bits"] = 8
    p = config.preset_from_dict(raw, mock=True)
    p.camera.average = 1
    eng.preset = p
    base = api.capture_base(p, "empty")
    assert base.field.axes == ["corr_y"]
    obj = api.capture_object(p, base, "part")
    assert obj.meta["relief_axis"] == "corr_y" and obj.meta["overlap_px"] > 0
    # an x-coded base from before fingerprints is refused BEFORE capturing
    x = api.load_preset(PRESET)
    x.camera.average = 1
    xbase = api.capture_base(x, "x-empty")
    xbase.meta.pop("geometry_fp")
    with pytest.raises(rules.GeometryMismatch, match="corr_y"):
        api.capture_object(p, xbase, "wrong-axis")


def test_capture_base_quality_gate(eng, preset):
    """Finding: a ~0 % valid base (lens cap, blue-lines DMD) became the
    default base for every later scan."""
    preset.camera.average = 1
    good = api.capture_base(preset, "good")
    assert not good.warnings and api.latest_base().id == good.id
    bench = api.mock_bench()
    bench.gain = 0.0                                          # lens cap on
    dark = api.capture_base(preset, "lens-cap")
    assert dark.meta["rejected"] and any("auto-REJECTED" in w for w in dark.warnings)
    assert (dark.path / "REJECTED.json").exists()             # kept (append-only), flagged
    assert api.latest_base().id == good.id
    assert [b.meta.get("rejected") is not None for b in api.list_library("bases")] == [False, True]
    bench.gain = 380.0
    o = api.capture_object(preset, dark, "against-dark")      # explicit: warned, not silent
    assert any("REJECTED" in w for w in o.meta["warnings"])


def test_partial_entries_do_not_break_listings(eng, preset):
    """Finding: one base folder without meta.json made every listing and
    latest_base raise (and the web UI refuse to start)."""
    preset.camera.average = 1
    base = api.capture_base(preset, "empty")
    part = eng.store.root / "bases" / preset.geometry_id / "20990101T000000_000000_partial"
    part.mkdir()
    obj_part = eng.store.new_dir("objects", label="partial")
    assert api.latest_base().id == base.id
    assert [b.id for b in api.list_library("bases")] == [base.id]
    assert api.list_library("objects") == []
    inc = api.list_library("incomplete")
    assert {i["id"] for i in inc} == {part.name, obj_part.name}
    for i in inc:
        api.delete(i["path"], reason="interrupted save")
    assert api.list_library("incomplete") == []


def test_list_library_refuses_path_traversal(eng):
    for bad in ("../../..", "..", "a/b", "..\\x"):
        for kind in ("banks", "maps", "bases", "calibrations"):
            with pytest.raises(ValueError, match="geometry_id"):
                api.list_library(kind, bad)
    assert api.list_library("bases", "mock-wall_2p7ft_vstripe_v1") == []


def test_verbs_refuse_to_run_unconfigured():
    """Finding: api verbs silently auto-configured the REAL rig."""
    old, api._session = api._session, None
    try:
        with pytest.raises(RuntimeError, match=r"configure\(mock=True\)"):
            api.load_preset(PRESET)
        with pytest.raises(RuntimeError, match="not configured"):
            api.paths()
        assert api._session is None
        with pytest.raises(TypeError):
            api.configure()                                   # mock must be explicit
    finally:
        api._session = old


def test_inspect_uses_the_scans_own_trigger(eng, preset):
    """Finding: inspect took the trigger from the active preset (None when no
    preset was loaded -> INDETERMINATE)."""
    preset.camera.average = 1
    base = api.capture_base(preset, "empty")
    v = scan(preset, base, "v", VALIDATION_OBJECT, known=V)
    api.calibrate(preset, [v])
    eng.preset = None
    insp = api.inspect(v)
    assert insp.units == "mm" and insp.trigger == 0.7        # was None -> no FAIL possible
    assert not any("engine default" in w for w in insp.warnings)
    v.meta["preset"] = "no_such_preset"
    insp = api.inspect(v)
    assert insp.trigger == 0.7 and any("engine default" in w for w in insp.warnings)


def test_defects_json_is_strict_json(eng, preset, monkeypatch):
    """Finding: a tiny inspection region wrote bare NaN tokens."""
    from engine.measure import inspect as inspect_mod
    from engine.results import Inspection
    preset.camera.average = 1
    base = api.capture_base(preset, "empty")
    v = scan(preset, base, "v", Box(30, 40, 30))
    monkeypatch.setattr(inspect_mod, "inspect", lambda *a, **k: Inspection(
        verdict="INDETERMINATE", defects=[], units="px", tier="relief",
        noise_sigma=float("nan"), threshold=float("nan"), trigger=None, region_px=12))
    monkeypatch.setattr(inspect_mod, "annotate_png", lambda *a, **k: None)
    insp = api.inspect(v)
    text = insp.path.read_text()
    assert _strict_json(text)["noise_sigma"] is None and "NaN" not in text


def test_rig_cleanup_closes_everything_and_keeps_the_real_error(eng, preset, monkeypatch):
    """Finding: camera.close() raising skipped projector.close() (fullscreen
    pattern left up) and replaced the error that ended the run."""
    from engine import session as session_mod
    closed = []

    class Cam:
        def close(self):
            closed.append("camera")
            raise RuntimeError("ReleaseInstance failed")

    class Proj(session_mod.Projector):
        def close(self):
            closed.append("projector")
            super().close()
    monkeypatch.setattr(session_mod, "open_camera", lambda *a, **k: Cam())
    monkeypatch.setattr(session_mod, "Projector", Proj)
    with pytest.raises(ValueError, match="scan failed"):
        with eng.rig(preset):
            raise ValueError("scan failed")
    assert closed == ["camera", "projector"]
    closed.clear()
    with pytest.raises(RuntimeError, match="ReleaseInstance"):     # nothing else to report
        with eng.rig(preset):
            pass
    assert closed == ["camera", "projector"]


def test_zone_scan_warns_about_an_old_base(eng, preset):
    """Finding: zone scans have no datum, so warm-up drift since an old base
    reads as height with no warning."""
    preset.camera.average = 1
    base = api.capture_base(preset, "empty")
    base.created = "2000-01-01T00:00:00"
    o = api.capture_object(preset, base, "zone", zone="C4")
    assert any("min old" in w for w in o.meta["warnings"]) and o.meta["base_age_s"] > 3600
    fresh = api.capture_object(preset, api.latest_base(), "zone2", zone="C4")
    assert not any("min old" in w for w in fresh.meta["warnings"])

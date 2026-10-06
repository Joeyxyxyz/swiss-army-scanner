"""Cross-module integration fixes (projector gamma order, zone datum, rig
lock, compare/inspect outputs, calibration trash, mock noise)."""

import tempfile
import types
from pathlib import Path

import numpy as np
import pytest

from engine import api, rules
from engine.capture.projector import Projector
from engine.capture.riglock import RigLock
from engine.config import ProjectionCfg
from engine.results import ToleranceBand

from conftest import Box, VALIDATION_OBJECT


def test_gamma_precomp_applies_to_the_sent_value():
    ramp = np.tile(np.arange(256, dtype=np.uint8), (4, 1))
    # gamma 1: identical to the v10 chain (duty -> invert)
    p1 = Projector(ProjectionCfg(duty=0.55, invert=True, gamma=1.0), mock=True)
    assert np.array_equal(p1.prepare(ramp), 255 - (ramp.astype(np.float32) * 0.55).astype(np.uint8))
    # gamma 2.2 with invert: the value SENT is precompensated
    p2 = Projector(ProjectionCfg(duty=1.0, invert=True, gamma=2.2), mock=True)
    sent = p2.prepare(ramp).astype(np.float64)
    target = 255.0 - ramp                                   # linear light we want
    assert np.allclose(255 * (sent / 255) ** 2.2, target, atol=6)


def test_rig_lock_blocks_a_second_holder():
    path = Path(tempfile.gettempdir()) / "sas_rig_test.lock"
    a = RigLock(path, "first").acquire()
    try:
        with pytest.raises(rules.RigBusy, match="first"):
            RigLock(path, "second").acquire()
    finally:
        a.release()
    RigLock(path, "third").acquire().release()


def test_mock_captures_have_fresh_noise(preset):
    preset.camera.average = 1
    a = api.capture_base(preset, "a")
    b = api.capture_base(preset, "b")
    both = a.field.mask & b.field.mask
    d = (a.field.corr_x - b.field.corr_x)[both]
    assert np.std(d) > 0                          # not bit-identical any more
    assert np.median(np.abs(d)) < 0.2             # ...but repeatable


def test_zone_scan_keeps_height_when_part_fills_zone(preset):
    preset.camera.average = 1
    base = api.capture_base(preset, "empty")
    api.capture_map()
    shot = api.project_grid()
    bench = api.mock_bench()
    bench.place(Box(20.0, 140.0, 140.0))          # big part: fills a centre cell
    cell = "E6"
    o = api.capture_object(preset, base, "fill", zone=cell)
    r = o.relief[o.relief_mask]
    assert r.size > 200 and np.median(r) == pytest.approx(20.0 * 0.45, abs=0.6), np.median(r)
    assert o.meta["zone_cells"]


def test_compare_and_inspect_png_reach_the_output_bin(preset):
    preset.camera.average = 1
    base = api.capture_base(preset, "empty")
    bench = api.mock_bench()
    bench.place(Box(30.0, 40.0, 30.0))
    g = api.capture_object(preset, base, "golden")
    s = api.capture_object(preset, base, "same")
    api.compare(s, g, ToleranceBand(fail_px=1.0))
    api.inspect(s)
    rows = api.list_library("outputs")
    kinds = [(r["kind"], r["file"].rsplit(".", 1)[-1]) for r in rows]
    assert ("comparisons", "json") in kinds and ("comparisons", "png") in kinds
    assert ("defects", "png") in kinds


def test_trashing_active_calibration_falls_back(preset):
    preset.camera.average = 1
    base = api.capture_base(preset, "empty")
    api.mock_bench().place(VALIDATION_OBJECT)
    v = api.capture_object(preset, base, "v", known_dims={"H": 106.95})
    cal1 = api.calibrate(preset, [v])
    cal2 = api.calibrate(preset, [v])
    assert api.calibration_for(preset.geometry_id).id == cal2.id
    api.delete(cal2.path)
    assert api.calibration_for(preset.geometry_id).id == cal1.id
    api.delete(cal1.path)
    assert api.calibration_for(preset.geometry_id) is None


def test_list_presets_and_paths(eng):
    assert "cell_wall_2p7ft_vstripe" in api.list_presets()
    P = api.paths()
    assert {"banks", "maps", "output", "trash"} <= set(P)


# ------------------------------------------- machine-wide rig lock path

@pytest.fixture
def lock_env(tmp_path, monkeypatch):
    """PROGRAMDATA and the temp-dir fallback redirected into tmp_path, so the
    test never touches the real machine-wide lock."""
    from engine.capture import riglock
    pd = tmp_path / "ProgramData"
    pd.mkdir()
    monkeypatch.setenv("PROGRAMDATA", str(pd))
    fallback = tmp_path / "tempdir" / "swiss_army_scanner_rig.lock"
    monkeypatch.setattr(riglock, "FALLBACK_LOCK_PATH", fallback)
    return riglock, pd, fallback


def test_rig_lock_lives_in_programdata_whatever_tmp_says(lock_env, tmp_path, monkeypatch):
    riglock, pd, fallback = lock_env
    want = pd / "swiss_army_scanner" / "rig.lock"
    a = RigLock(holder="ui").acquire()
    try:
        assert a.path == want and want.exists()
        assert riglock.holder_info()["holder"] == "ui"
        # another process with a different TMP / TEMP still meets the same lock
        for var in ("TMP", "TEMP"):
            monkeypatch.setenv(var, str(tmp_path / f"other_{var}"))
        with pytest.raises(rules.RigBusy, match="'ui'"):
            RigLock(holder="cli").acquire()
    finally:
        a.release()
    assert not fallback.exists()
    RigLock(holder="cli").acquire().release()


def test_rig_lock_falls_back_to_tempdir_without_programdata(lock_env, monkeypatch):
    riglock, pd, fallback = lock_env
    monkeypatch.delenv("PROGRAMDATA")
    with RigLock(holder="x") as a:
        assert a.path == fallback and a.held
    (pd / "blocker").write_text("a file, so no folder can be made under it")
    monkeypatch.setenv("PROGRAMDATA", str(pd / "blocker"))      # unwritable PROGRAMDATA
    with RigLock(holder="y") as b:
        assert b.path == fallback
        with pytest.raises(rules.RigBusy):
            RigLock(holder="z").acquire()


def test_explicit_lock_path_is_used_as_given(lock_env, tmp_path):
    p = tmp_path / "private" / "rig.lock"
    with RigLock(p, "t") as a:
        assert a.path == p and not (lock_env[1] / "swiss_army_scanner").exists()


def test_session_rig_lock_names_the_session_holder(eng, monkeypatch):
    """The real-rig branch of ScanSession.rig takes the lock with the
    session's holder (or a default) BEFORE opening anything; the camera
    open is stubbed to fail, so no hardware is touched."""
    from engine import session as session_mod
    from engine.capture import riglock
    taken = []

    class Recorder:
        def __init__(self, path=None, holder=""):
            taken.append(holder)

        def acquire(self):
            return self

        def release(self):
            taken.append("released")

    def no_camera(*a, **k):
        raise RuntimeError("no camera in tests")
    monkeypatch.setattr(riglock, "RigLock", Recorder)
    monkeypatch.setattr(session_mod, "open_camera", no_camera)
    monkeypatch.setattr(session_mod, "Projector", no_camera)
    real = session_mod.ScanSession(eng.store.root, mock=False)
    stub = types.SimpleNamespace(camera=None, projection=None)
    assert real.holder == ""
    with pytest.raises(RuntimeError, match="no camera"):
        with real.rig(stub):
            pass
    assert taken[0] == session_mod.default_holder() and "scan engine" in taken[0]
    assert taken[1] == "released"
    taken.clear()
    real.holder = "web UI"
    with pytest.raises(RuntimeError, match="no camera"):
        with real.rig(stub):
            pass
    assert taken == ["web UI", "released"]
    assert session_mod.ScanSession(eng.store.root, mock=True, holder="cli").holder == "cli"

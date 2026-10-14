"""The triangulation preset's mount (rig.mount: fixed | mobile) and the rig gate
(engine/rigstate.py), MOCK only.

fixed: the stored calibration is trusted until a rig check says MOVED / UNKNOWN; that
objection stands until it is answered. mobile: the rig is assumed moved since its last use,
so no mm until it was verified in this session. What answers / verifies, for the calibration
in use: a calibration from poses captured since, a rig check OK at the default tolerance
against a reference of that calibration that was trusted when it was stored, a re-home
against such a reference (fixed also: a reference stored on request). What does not: a
reference stored while the rig was in doubt (in that session or a later one), a loose
tolerance, another calibration's reference, stored poses solved again, an in-doubt
reference's re-home. The record's ORDER decides, not its clock; a session is never older than
the computer's or the owning program's start; a fixed copy of a mobile preset is gated as
mobile; a stored scan of a fixed rig made while or just before it objected is in doubt; only
the newest lines of a long record are read.

Then the record real rig checks and a real (mock) calibration leave, and the gate in all
three operating modes and in the engine's own capture verbs: Measure -> NOT_MEASURABLE,
Quick check and Station -> PULL, a metric / fast scan -> refused, with "mobile rig not
verified: run rig-check / re-home / calibrate"; a fixed rig in Station mode is not gated
(unchanged)."""

import contextlib
import io
import json
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from engine import api, config, rigstate, rules
from engine import disposition as D
from engine.capture.mock_bench import MockBench
from ui import cli

import recipe_synth as RS
from conftest import FIXTURE_PRESETS
from conftest import PRESET as FIXTURE_PRESET

PRESET = "bench_9in_metric"
PHRASE = "mobile rig not verified: run rig-check / re-home / calibrate"
MIN = timedelta(minutes=1)
TRUSTED = {"mobile": True, "fixed": True}
IN_DOUBT = {"mobile": False, "fixed": False}


@pytest.fixture(autouse=True)
def _fresh_engine():
    yield
    api._session = None


def _cli(argv) -> tuple:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = cli.main(argv)
    api._session = None
    return code, buf.getvalue()


# ================================================================== the preset setting

def test_mount_is_a_preset_setting_and_not_part_of_the_geometry(eng, tmp_path):
    p = api.load_preset(FIXTURE_PRESET)
    assert p.rig.mount == "fixed" and config.MOUNTS == ("fixed", "mobile")   # the default
    r = api.preset_add("mob", FIXTURE_PRESET, {"rig.mount": "mobile"})
    assert json.loads(Path(r["path"]).read_text())["rig"]["mount"] == "mobile"
    m = api.load_preset("mob")
    assert m.rig.mount == "mobile" and m.geometry_id == p.geometry_id
    # how far the calibration is trusted, not what a correspondence means: the same geometry
    assert config.geometry_fingerprint(m) == config.geometry_fingerprint(p)
    assert "mount" not in config.geometry_fingerprint(m)
    assert config.preset_to_dict(m)["rig"]["mount"] == "mobile"
    assert config.preset_to_dict(p)["rig"]["mount"] == "fixed"
    by = {i["name"]: i for i in api.preset_list()}
    assert by["mob"]["mount"] == "mobile" and by[FIXTURE_PRESET]["mount"] == "fixed"
    assert api.preset_show("mob")["mount"] == "mobile"
    for bad in ("wheels", "", None, 1, True):
        with pytest.raises(config.ConfigError, match="rig.mount"):
            api.preset_add("bad", FIXTURE_PRESET, {"rig.mount": bad})
    assert "bad" not in api.list_presets()
    # the command line: preset-add --set, preset-show, presets
    lib = str(eng.store.root)
    base = ["--mock", "--library", lib]
    api._session = None
    code, _ = _cli([*base, "preset-add", "cart", "--from", PRESET, "--set", "rig.mount=mobile"])
    assert code == 0
    code, out = _cli([*base, "preset-show", "cart"])
    assert code == 0 and "mount (rig.mount)" in out and "mobile (assumed moved" in out
    assert '"mount": "mobile"' in out
    code, out = _cli([*base, "preset-show", PRESET])
    assert "fixed (the calibration is trusted until a rig check says MOVED" in out
    code, out = _cli([*base, "--json", "presets"])
    rows = {r["name"]: r for r in json.loads(out)["presets"]}
    assert rows["cart"]["mount"] == "mobile" and rows[PRESET]["mount"] == "fixed"
    code, out = _cli([*base, "presets"])
    assert "[mobile]" in next(line for line in out.splitlines() if " cart " in line)
    # ... and the preset it was copied from is the same rig: listed as gated like it
    assert "[mobile: as cart]" in next(line for line in out.splitlines() if f" {PRESET} " in line)
    assert rows[PRESET]["gated_as"] == "mobile" and rows[PRESET]["mount_declared_by"] == "cart"
    assert rows["cart"]["gated_as"] == "mobile" and rows["cart"]["mount_declared_by"] is None
    assert _cli([*base, "preset-add", "x", "--from", PRESET, "--set", "rig.mount=loose"])[0] == 2


# ================================================================== the gate's rules (pure)

@pytest.fixture
def rig(tmp_path):
    """A mock library with a FIXED preset and a MOBILE one at its OWN geometry (a geometry_id
    names one rig), each with the true stereo calibration. The mobile rig was calibrated
    "now": its poses and its calibration are on its rig-state record. T0 = that moment."""
    s = api.configure(tmp_path / "library", mock=True, bench=MockBench(seed=1))
    fixed = api.load_preset(PRESET)
    s.bench.use_stereo()
    cal_fixed = RS.truth_calibration(s, fixed)
    api.preset_add("mob", PRESET, {"rig.mount": "mobile"}, geometry_id="cart_rig_v1")
    mobile = config.load_preset("mob", store=s.store, mock=True)
    assert mobile.geometry_id != fixed.geometry_id
    cal = RS.calibrated_now(s, mobile)
    t0 = rigstate.read(s.store, mobile.geometry_id)[-1]["time"]
    return SimpleNamespace(s=s, store=s.store, fixed=fixed, mobile=mobile, cal=cal,
                           cal_fixed=cal_fixed, t0=t0, tmp=tmp_path)


def _entry(t, label="scan") -> str:
    """A library entry's name, made at time t."""
    return t.strftime("%Y%m%dT%H%M%S_%f") + "_" + label


def _scan(t, name="scan"):
    return SimpleNamespace(id=name, path=Path(_entry(t, name)), created=None)


def _check(rig, status, at=None, preset=None, cal_id="same", ref="ref0", ref_cal="same",
           tol=0.5, trust=TRUSTED, **extra):
    """A rig check's line on the record, as engine/calibrate/rigref.py writes it: by default
    one at the default tolerance, against a trusted reference of the calibration in use."""
    p = preset or rig.mobile
    cal = (rig.cal.id if p is rig.mobile else rig.cal_fixed.id) if cal_id == "same" else cal_id
    return rigstate.record_check(rig.store, p, {
        "status": status, "tol_px": tol, "calibration_id": cal, "reference_id": ref,
        "reference_calibration_id": cal if ref_cal == "same" else ref_cal,
        "reference_trust": trust, "message": f"rig check {status}", **extra}, now=at)


def _calibrated(rig, preset, cal_id, at, poses, capture=True):
    """A calibration's lines: its poses captured (unless they are old ones), then solved."""
    for pose in poses if capture else ():
        rigstate.record_pose(rig.store, preset, pose, "set", at)
    rigstate.record_calibration(rig.store, preset, cal_id, poses, "set", at)


def test_a_fixed_rig_is_trusted_until_a_rig_check_objects(rig):
    t0, p, store = rig.t0, rig.fixed, rig.store
    g = rigstate.gate(store, p)                     # Station mode: never gated, nothing read
    assert g["ok"] and g["mount"] == "fixed" and g["state"] is None
    g = rigstate.gate(store, p, strict=True, now=t0 + 400 * 24 * 60 * MIN)
    assert g["ok"] and g["state"]["verified"] is None          # no session, no expiry
    assert g["state"]["calibration_id"] == rig.cal_fixed.id and g["state"]["last_check"] is None
    _check(rig, "MOVED", t0 + 5 * MIN, preset=p, moved="camera", shift_px=2.3)
    g = rigstate.gate(store, p, strict=True, now=t0 + 6 * MIN)
    assert not g["ok"] and g["code"] == "rig_check" and "MOVED" in g["message"]
    assert "rig-check" in g["how_to_fix"] and g["state"]["objection"]["status"] == "MOVED"
    assert rigstate.gate(store, p, now=t0 + 6 * MIN)["ok"]     # Station mode: unchanged
    _check(rig, "OK", t0 + 7 * MIN, preset=p)       # against the reference stored before it
    g = rigstate.gate(store, p, strict=True, now=t0 + 8 * MIN)
    assert g["ok"] and g["state"]["objection"] is None and g["state"]["last_check"]["status"] == "OK"
    _check(rig, "UNKNOWN", t0 + 9 * MIN, preset=p)
    g = rigstate.gate(store, p, strict=True, now=t0 + 10 * MIN)
    assert not g["ok"] and "UNKNOWN" in g["message"]
    # the operator stored a new reference on request: "the rig is known good"
    _check(rig, "REFERENCE_SET", t0 + 11 * MIN, preset=p, requested=True, ref="ref1")
    assert rigstate.gate(store, p, strict=True, now=t0 + 12 * MIN)["ok"]
    # a line that can't be read is a check that said UNKNOWN (a torn MOVED never reads as OK)
    with open(rigstate.log_path(store, p.geometry_id), "ab") as f:
        f.write(b'{"time": "2026-10-11T10:00:00", "event": "rig_check", "status": "MOV')
    g = rigstate.gate(store, p, strict=True, now=t0 + 13 * MIN)
    assert not g["ok"] and g["state"]["objection"]["unreadable"] is True
    _check(rig, "OK", t0 + 14 * MIN, preset=p)                 # a newer good record answers it
    assert rigstate.gate(store, p, strict=True, now=t0 + 15 * MIN)["ok"]
    recs = rigstate.read(store, p.geometry_id)
    assert [r.get("status") for r in recs] == ["MOVED", "OK", "UNKNOWN", "REFERENCE_SET",
                                               "UNKNOWN", "OK"]
    assert recs[4]["unreadable"] and recs[4]["time"] == recs[5]["time"]
    assert [r["status"] for r in rigstate.read_back(store, p.geometry_id)] == [
        "OK", "UNKNOWN", "REFERENCE_SET", "UNKNOWN", "OK", "MOVED"]      # newest first
    # a status no rig check gives is an objection too (fail closed)
    _check(rig, "MAYBE", t0 + 16 * MIN, preset=p)
    assert not rigstate.gate(store, p, strict=True, now=t0 + 17 * MIN)["ok"]


def test_what_answers_an_objection_on_a_fixed_rig_and_what_does_not(rig):
    t0, p, store, cal = rig.t0, rig.fixed, rig.store, rig.cal_fixed.id

    def objects(minute: int) -> str:
        g = rigstate.gate(store, p, strict=True, now=t0 + minute * MIN)
        assert not g["ok"] and g["code"] == "rig_check" and "MOVED" in g["message"]
        return g["message"]

    # poses captured BEFORE the rig moved, and the calibration in use solved from them
    _calibrated(rig, p, cal, t0 + MIN, ["old1", "old2", "old3"])
    assert rigstate.gate(store, p, strict=True, now=t0 + 2 * MIN)["ok"]
    _check(rig, "MOVED", t0 + 3 * MIN, preset=p, moved="camera")
    objects(4)
    # an OK at a looser tolerance than the default proves nothing
    _check(rig, "OK", t0 + 5 * MIN, preset=p, tol=50.0)
    assert "only 0.5 px or tighter verifies" in objects(6)
    # an OK against a reference that belongs to another calibration
    _check(rig, "OK", t0 + 7 * MIN, preset=p, ref_cal="mock-other/stereo")
    assert "taken with calibration mock-other/stereo" in objects(8)
    # the first check after a "new" calibration stores a reference by itself: no answer, and
    # an OK against that reference (stored while the objection stood) is none either
    _check(rig, "REFERENCE_SET", t0 + 9 * MIN, preset=p, ref="ref_moved", trust=IN_DOUBT)
    objects(10)
    _check(rig, "OK", t0 + 11 * MIN, preset=p, ref="ref_moved", trust=IN_DOUBT)
    assert "stored while the rig check's objection stood" in objects(12)
    # the stored poses solved again: a new entry, the same old rig position
    _calibrated(rig, p, cal, t0 + 13 * MIN, ["old1", "old2", "old3"], capture=False)
    assert "solving stored poses again verifies nothing" in objects(14)
    # an answer for another calibration than the one in use
    _check(rig, "OK", t0 + 15 * MIN, preset=p, cal_id="mock-other/stereo")
    assert "not for the one in use" in objects(16)
    # poses captured since, and the calibration in use solved from them: answered
    _calibrated(rig, p, cal, t0 + 17 * MIN, ["new1", "new2", "new3"])
    g = rigstate.gate(store, p, strict=True, now=t0 + 18 * MIN)
    assert g["ok"] and g["state"]["objection"] is None
    # a reference from before this record kept what it can be trusted for nowhere: on a
    # fixed rig it still answers (the calibration was trusted when it was stored)
    _check(rig, "MOVED", t0 + 19 * MIN, preset=p)
    _check(rig, "OK", t0 + 20 * MIN, preset=p, trust=None)
    assert rigstate.gate(store, p, strict=True, now=t0 + 21 * MIN)["ok"]


def test_a_mobile_rig_needs_a_verification_in_this_session(rig):
    t0, p, store = rig.t0, rig.mobile, rig.store
    # calibrated in this session, from poses captured in it: that is the verification
    g = rigstate.gate(store, p, now=t0 + MIN)
    assert g["ok"] and g["mount"] == "mobile" and g["state"]["verified"] is True
    assert rig.cal.id in g["state"]["verified_by"] and g["state"]["session_marker"] is None
    # a session lasts at most MOBILE_SESSION_MAX_AGE_S: the next day it is unverified again
    later = t0 + timedelta(seconds=rigstate.MOBILE_SESSION_MAX_AGE_S) + MIN
    for strict in (False, True):                    # Station asks too: a mobile rig is gated
        g = rigstate.gate(store, p, strict=strict, now=later)
        assert not g["ok"] and g["code"] == "rig_unverified" and g["message"].startswith(PHRASE)
        assert "rig-check" in g["how_to_fix"] and "calibrate" in g["how_to_fix"]
    assert g["state"]["verified"] is False and "nothing has verified it" in g["state"]["why"]
    # the operator moved it: a session marker ends what the calibration verified
    rec = rigstate.new_session(store, p, "moved to the second bench", by="operator",
                               now=t0 + 10 * MIN)
    assert rec["event"] == "session" and rec["mount"] == "mobile"
    g = rigstate.gate(store, p, now=t0 + 11 * MIN)
    assert not g["ok"] and g["message"].startswith(PHRASE) and "session marker" in g["message"]
    assert rigstate.gate(store, rig.fixed, strict=True, now=t0 + 11 * MIN)["ok"]   # another rig
    # a rig check OK against the calibration's reference (stored while it was verified)
    _check(rig, "OK", t0 + 12 * MIN)
    g = rigstate.gate(store, p, now=t0 + 13 * MIN)
    assert g["ok"] and "rig check OK" in g["state"]["verified_by"]
    # ... a stored scan made before that verification is not covered by it, a later one is
    old, new = _scan(t0 + 11 * MIN, "old"), _scan(t0 + 12 * MIN + timedelta(seconds=5), "new")
    g = rigstate.gate(store, p, scan=old, now=t0 + 13 * MIN)
    assert not g["ok"] and g["code"] == "rig_unverified" and "stored scan old" in g["message"]
    assert rigstate.gate(store, p, scan=new, now=t0 + 13 * MIN)["ok"]
    for odd in (SimpleNamespace(id="odd", path=Path("no_time_in_its_name"), created=None),
                _scan(t0 + 30 * MIN, "tomorrow")):  # no time of its own / later than now
        g = rigstate.gate(store, p, scan=odd, now=t0 + 13 * MIN)
        assert not g["ok"] and "the clock was set back" in g["message"]
    # a later MOVED undoes it. The doubt is rig_unverified, never rig_check: it says nothing
    # about the part, so no recipe's invalid_reasons_to_reject can make a reject of it
    _check(rig, "MOVED", t0 + 14 * MIN, moved="camera")
    g = rigstate.gate(store, p, now=t0 + 15 * MIN)
    assert not g["ok"] and g["code"] == "rig_unverified" and g["message"].startswith(PHRASE)
    assert "MOVED" in g["message"] and g["state"]["objection"]["status"] == "MOVED"
    assert "rig_unverified" in D.NOT_REJECTABLE and "rig_unverified" not in D.REJECTABLE
    _check(rig, "OK", t0 + 16 * MIN)                # the next OK verifies again
    g = rigstate.gate(store, p, now=t0 + 17 * MIN)
    assert g["ok"] and g["state"]["objection"] is None
    # another marker: unverified again
    rigstate.new_session(store, p, now=t0 + 18 * MIN)
    assert not rigstate.gate(store, p, now=t0 + 19 * MIN)["ok"]
    # poses captured now and the calibration solved from them: verified
    _calibrated(rig, p, rig.cal.id, t0 + 20 * MIN, ["a1", "a2", "a3"])
    g = rigstate.gate(store, p, now=t0 + 21 * MIN)
    assert g["ok"] and "made in this session" in g["state"]["verified_by"]


def test_what_does_not_verify_a_mobile_rig(rig):
    t0, p, store = rig.t0, rig.mobile, rig.store
    first = next(r for r in rigstate.read(store, p.geometry_id) if r["event"] == "calibration")
    rigstate.new_session(store, p, now=t0 + 10 * MIN)

    def unverified(minute: int) -> str:
        g = rigstate.gate(store, p, now=t0 + minute * MIN)
        assert not g["ok"] and g["code"] == "rig_unverified" and g["message"].startswith(PHRASE)
        return g["message"]

    # a reference stored IN DOUBT (the first check stores one by itself; --set-reference
    # does on request) and then a check against it would wave any rig through
    _check(rig, "REFERENCE_SET", t0 + 11 * MIN, ref="ref_in_doubt", trust=IN_DOUBT,
           requested=True)
    assert "taken while the rig was not verified" in unverified(12)
    _check(rig, "OK", t0 + 13 * MIN, ref="ref_in_doubt", trust=IN_DOUBT)
    assert "stored while the rig was not verified" in unverified(14)
    # ... and it does not become a good reference in the NEXT session either
    rigstate.new_session(store, p, "the next start", now=t0 + 15 * MIN)
    _check(rig, "OK", t0 + 16 * MIN, ref="ref_in_doubt", trust=IN_DOUBT)
    assert "ref_in_doubt" in unverified(17)
    # a reference whose trust was never recorded
    _check(rig, "OK", t0 + 18 * MIN, trust=None)
    unverified(19)
    # an OK at a loose tolerance (a 3 px move is "in place" at 50 px)
    _check(rig, "OK", t0 + 20 * MIN, tol=50.0)
    assert "only 0.5 px or tighter verifies" in unverified(21)
    _check(rig, "OK", t0 + 22 * MIN, tol=None)      # ... or at one nobody recorded
    unverified(23)
    # an OK against the reference of another calibration than the one in use
    _check(rig, "OK", t0 + 24 * MIN, ref_cal="mock-other/stereo")
    assert "taken with calibration mock-other/stereo" in unverified(25)
    # a verification of another calibration
    _check(rig, "OK", t0 + 26 * MIN, cal_id="mock-other/stereo")
    assert "not the one in use" in unverified(27)
    # the stored poses (captured before this session) solved again
    _calibrated(rig, p, rig.cal.id, t0 + 28 * MIN, first["poses"], capture=False)
    assert "solving stored poses again verifies nothing" in unverified(29)
    # a re-home against the in-doubt reference
    _check(rig, "REHOMED", t0 + 30 * MIN, ref="ref_rehomed", trust=IN_DOUBT,
           previous_reference_id="ref_in_doubt", rehome_from=rig.cal.id)
    assert "re-home" in unverified(31)
    # an UNKNOWN is an objection, and poses captured before it do not answer it
    rigstate.record_pose(store, p, "b1", "set", t0 + 32 * MIN)
    _check(rig, "UNKNOWN", t0 + 33 * MIN)
    _calibrated(rig, p, rig.cal.id, t0 + 34 * MIN, ["b1"], capture=False)
    assert "UNKNOWN" in unverified(35)
    with pytest.raises(ValueError, match="geometry_id"):
        rigstate.log_path(store, "../elsewhere")


def test_the_order_of_the_record_decides_not_its_clock(rig):
    """A clock set back between two records (daylight saving, a time sync after power-up)
    must not put an old OK after a newer session marker or a newer MOVED."""
    store, p, day = rig.store, rig.mobile, rig.t0 + 60 * MIN
    rigstate.new_session(store, p, "session 1", now=day)
    _check(rig, "OK", day + 50 * MIN)               # stamped 50 min in
    rigstate.new_session(store, p, "carried elsewhere", now=day + 10 * MIN)   # the clock went back
    g = rigstate.gate(store, p, now=day + 15 * MIN)
    assert not g["ok"] and g["state"]["verified"] is False     # the OK is before the marker
    _check(rig, "OK", day + 55 * MIN)
    assert rigstate.gate(store, p, now=day + 56 * MIN)["ok"]
    _check(rig, "MOVED", day + 12 * MIN)            # written after that OK, stamped before it
    g = rigstate.gate(store, p, strict=True, now=day + 57 * MIN)
    assert not g["ok"] and g["state"]["last_check"]["status"] == "MOVED"
    # the same on a fixed rig
    _check(rig, "OK", day + 50 * MIN, preset=rig.fixed)
    _check(rig, "MOVED", day + 10 * MIN, preset=rig.fixed)
    assert not rigstate.gate(store, rig.fixed, strict=True, now=day + 57 * MIN)["ok"]
    # a stored scan is placed by its own clock time: it must lie after everything on the
    # record up to the verification, not just after the verifying line's own stamp
    _check(rig, "OK", day + 13 * MIN)               # verified again (stamped 13 min in)
    now = day + 58 * MIN
    assert rigstate.gate(store, p, now=now)["ok"]
    g = rigstate.gate(store, p, scan=_scan(day + 20 * MIN, "between"), now=now)
    assert not g["ok"] and "stored scan between" in g["message"]
    assert rigstate.gate(store, p, scan=_scan(day + 57 * MIN, "after"), now=now)["ok"]


def test_a_session_is_never_older_than_the_computer_or_the_program(rig, monkeypatch):
    t0, p, store = rig.t0, rig.mobile, rig.store
    assert rigstate.gate(store, p, now=t0 + MIN)["ok"]
    # the computer was restarted since: whatever verified the rig before no longer does
    monkeypatch.setattr(rigstate, "boot_time", lambda: t0 + 5 * MIN)
    g = rigstate.gate(store, p, now=t0 + 6 * MIN)
    assert not g["ok"] and "this computer started" in g["message"]
    _check(rig, "OK", t0 + 7 * MIN)
    assert rigstate.gate(store, p, now=t0 + 8 * MIN)["ok"]
    monkeypatch.setattr(rigstate, "boot_time", lambda: None)   # an OS that does not say
    assert rigstate.gate(store, p, now=t0 + 8 * MIN)["ok"]
    # a long-running owner of the rig (the web UI server) started: a new session for every
    # mobile geometry of its library, also one it never wrote a marker for
    rigstate.open_process_session(store, now=t0 + 9 * MIN)
    try:
        g = rigstate.gate(store, p, now=t0 + 10 * MIN)
        assert not g["ok"] and "this program started" in g["message"]
        assert rigstate.gate(store, rig.fixed, strict=True, now=t0 + 10 * MIN)["ok"]   # fixed: none
        _check(rig, "OK", t0 + 11 * MIN)
        assert rigstate.gate(store, p, now=t0 + 12 * MIN)["ok"]
    finally:
        rigstate.close_process_session(store)
    monkeypatch.undo()
    boot = rigstate.boot_time()                     # the real one: in the past, or unknown
    assert boot is None or boot <= datetime.now()


def test_a_fixed_copy_of_a_mobile_preset_is_gated_as_mobile(rig):
    """A geometry_id names ONE rig: a rig that is carried is carried whichever preset drives
    it, so a `fixed` preset at a mobile preset's geometry is no way round the gate."""
    api._session = rig.s
    t0, store = rig.t0, rig.store
    api.preset_add("mob_bolted", "mob", {"rig.mount": "fixed"})
    twin = config.load_preset("mob_bolted", store=store, mock=True)
    assert twin.rig.mount == "fixed" and twin.geometry_id == rig.mobile.geometry_id
    assert rigstate.mount_of(store, twin) == ("mobile", "mob")
    assert rigstate.mount_of(store, rig.mobile) == ("mobile", "mob")
    assert rigstate.mount_of(store, rig.fixed) == ("fixed", None)      # its own geometry
    assert rigstate.gate(store, twin, now=t0 + MIN)["ok"]              # verified: the calibration
    rigstate.new_session(store, rig.mobile, "carried", now=t0 + 2 * MIN)
    for strict in (False, True):
        g = rigstate.gate(store, twin, strict=strict, now=t0 + 3 * MIN)
        assert not g["ok"] and g["code"] == "rig_unverified" and g["mount"] == "mobile"
        assert g["message"].startswith(PHRASE) and "preset mob declares" in g["message"]
        assert g["state"]["declared_by"] == "mob"
    assert api.rig_state(twin)["mount"] == "mobile"
    assert rigstate.gate(store, rig.fixed, strict=True, now=t0 + 3 * MIN)["ok"]
    # the mobile preset gone (to the trash): the copy is the fixed rig it says it is
    api.preset_delete("mob", reason="the rig was bolted down", force=True)
    assert rigstate.mount_of(store, twin) == ("fixed", None)
    assert rigstate.gate(store, twin, now=t0 + 3 * MIN)["ok"]


def test_a_stored_scan_of_a_fixed_rig_made_in_doubt_stays_in_doubt(rig):
    t0, p, store = rig.t0, rig.fixed, rig.store
    _check(rig, "OK", t0 + 1 * MIN, preset=p)
    before = _scan(t0 + 2 * MIN, "before")          # the next check says MOVED
    _check(rig, "MOVED", t0 + 3 * MIN, preset=p)
    during = _scan(t0 + 4 * MIN, "during")          # while the objection stood
    _check(rig, "OK", t0 + 5 * MIN, preset=p)
    after = _scan(t0 + 6 * MIN, "after")
    _check(rig, "OK", t0 + 7 * MIN, preset=p)
    now = t0 + 8 * MIN
    assert rigstate.gate(store, p, strict=True, now=now)["ok"]         # put right: no objection
    g = rigstate.gate(store, p, strict=True, scan=during, now=now)
    assert not g["ok"] and g["code"] == "rig_check" and "stored scan during" in g["message"]
    assert "(MOVED) stood" in g["message"] and "scan again" in g["how_to_fix"]
    g = rigstate.gate(store, p, strict=True, scan=before, now=now)
    assert not g["ok"] and "the next rig check" in g["message"] and "MOVED" in g["message"]
    assert rigstate.gate(store, p, strict=True, scan=after, now=now)["ok"]
    assert rigstate.gate(store, p, scan=during, now=now)["ok"]         # Station mode: unchanged
    g = rigstate.gate(store, p, strict=True, scan=_scan(now + 5 * MIN, "tomorrow"), now=now)
    assert not g["ok"] and "the clock was set back" in g["message"]
    # a scan made before a check that found the camera moved and re-homed it
    _check(rig, "REHOMED", t0 + 9 * MIN, preset=p, rehome_from=rig.cal_fixed.id,
           previous_reference_id="ref0", ref="ref_rehomed")
    g = rigstate.gate(store, p, strict=True, scan=_scan(t0 + 8 * MIN, "nudged"), now=t0 + 10 * MIN)
    assert not g["ok"] and "re-homed" in g["message"]


def test_only_the_newest_lines_of_a_long_record_are_read(rig, monkeypatch):
    """GET /api/cmd/state asks the gate twice a second: a record that grew for a year (a
    line per rig check) must cost what is read of it, not its length."""
    t0, store = rig.t0, rig.store
    read = []
    real = rigstate._record
    monkeypatch.setattr(rigstate, "_record", lambda raw: (read.append(1), real(raw))[1])

    def grow(preset, n, **kw):
        _check(rig, "OK", t0 + 20 * MIN, preset=preset, **kw)
        f = rigstate.log_path(store, preset.geometry_id)
        line = f.read_text(encoding="utf-8").splitlines()[-1]
        with open(f, "a", encoding="utf-8") as fh:
            fh.write((line + "\n") * n)

    grow(rig.fixed, 20000)
    read.clear()
    assert rigstate.gate(store, rig.fixed, strict=True, now=t0 + 21 * MIN)["ok"]
    assert len(read) <= 2                           # the newest OK answers: nothing older matters
    # a mobile rig: back to its session marker, no further
    rigstate.new_session(store, rig.mobile, "carried", now=t0 + 19 * MIN)
    grow(rig.mobile, 20000)
    rigstate.new_session(store, rig.mobile, "carried again", now=t0 + 22 * MIN)
    _check(rig, "OK", t0 + 23 * MIN)
    read.clear()
    assert rigstate.gate(store, rig.mobile, now=t0 + 24 * MIN)["ok"]
    assert len(read) == 2                           # the OK and the marker
    # ... but an objection far back is still found when nothing newer answers it
    _check(rig, "MOVED", t0 + 25 * MIN, preset=rig.fixed)
    grow(rig.fixed, 3000, tol=50.0)                 # loose OKs: none of them answers
    read.clear()
    g = rigstate.gate(store, rig.fixed, strict=True, now=t0 + 26 * MIN)
    assert not g["ok"] and g["state"]["objection"]["status"] == "MOVED" and len(read) > 3000
    # never more than MAX_TAIL lines, and fail closed when they do not settle it: a fixed
    # rig whose newest lines neither confirm it nor object is not taken for confirmed
    # (Station mode, which never asks strictly, is unchanged); a mobile one is unverified
    assert rigstate.MAX_TAIL == 5000
    _check(rig, "OK", t0 + 27 * MIN, preset=rig.fixed)
    assert rigstate.gate(store, rig.fixed, strict=True, now=t0 + 28 * MIN)["ok"]
    grow(rig.fixed, 6000, tol=50.0)
    read.clear()
    g = rigstate.gate(store, rig.fixed, strict=True, now=t0 + 28 * MIN)
    assert not g["ok"] and g["code"] == "rig_check" and len(read) == 5000
    assert "none of the newest 5000 lines" in g["message"]
    assert rigstate.gate(store, rig.fixed, now=t0 + 28 * MIN)["ok"]
    _check(rig, "OK", t0 + 28 * MIN, preset=rig.fixed)         # one proper rig check settles it
    assert rigstate.gate(store, rig.fixed, strict=True, now=t0 + 29 * MIN)["ok"]
    assert rigstate.gate(store, rig.mobile, now=t0 + 29 * MIN)["ok"]
    grow(rig.mobile, 6000, tol=50.0)
    read.clear()
    g = rigstate.gate(store, rig.mobile, now=t0 + 29 * MIN)
    assert not g["ok"] and g["code"] == "rig_unverified" and len(read) == 5000
    assert "the newest 5000 lines" in g["message"]
    _check(rig, "OK", t0 + 29 * MIN)
    assert rigstate.gate(store, rig.mobile, now=t0 + 30 * MIN)["ok"]


# ================================================================== real rig checks

def test_rig_checks_are_recorded_and_verify_a_mobile_rig(rig):
    """The record real rig checks leave, and a mobile rig end to end in Measure mode: set
    down again -> NOT_MEASURABLE, rig check OK -> MEASURED, rig check MOVED -> not again."""
    api._session = rig.s
    s, p, runs = rig.s, rig.mobile, rig.tmp / "runs"
    r = api.rig_check(p)                            # right after the calibration
    assert r["status"] == "REFERENCE_SET" and r["warnings"] == []
    assert r["reference_trust"] == {"mobile": True, "fixed": True}     # the rig was verified
    meta = json.loads((Path(r["path"]) / "rigref.json").read_text(encoding="utf-8"))
    assert meta["rig_trust"] == {"mobile": True, "fixed": True} and meta["mount"] == "mobile"
    log = rigstate.log_path(s.store, p.geometry_id)
    assert log.name == "rig_state.jsonl" and log.parent.name == p.geometry_id
    assert api.rig_state(p)["ok"] and "calibration" in api.rig_state(p)["state"]["verified_by"]
    old = RS.oracle_scan(s, p, [], "before")
    assert api.measure_run("flatness", p, scans=[old], runs_dir=runs)["result"]["status"] == "MEASURED"
    # the rig is carried elsewhere and set down again
    out = api.rig_session(p, "carried to the next part")
    assert out["marker"]["event"] == "session" and out["gate"]["ok"] is False
    bad = api.measure_run("flatness", p, runs_dir=runs)
    doc = bad["result"]
    assert bad["exit_code"] == 5 and doc["status"] == "NOT_MEASURABLE"
    assert doc["reason"] == "rig_unverified" and doc["message"].startswith(PHRASE)
    assert "rig-check" in doc["how_to_fix"] and doc["values"] == [] and doc["scans"] == []
    assert doc["preset"]["mount"] == "mobile" and doc["rig"]["verified"] is False
    assert doc["result"] is None and doc["disposition"] == "NONE"
    # the rig check against the calibration's reference (stored while the rig was verified)
    assert api.rig_check(p)["status"] == "OK"
    gate = api.rig_state(p)
    assert gate["ok"] and "rig check OK" in gate["state"]["verified_by"]
    stale = api.measure_run("flatness", p, scans=[old], runs_dir=runs)     # scanned before it
    assert stale["result"]["reason"] == "rig_unverified" and "stored scan" in stale["result"]["message"]
    new = RS.oracle_scan(s, p, [], "after")
    ok = api.measure_run("flatness", p, scans=[new], runs_dir=runs)
    assert ok["exit_code"] == 0 and "mobile rig verified by the rig check OK" in ok["result"]["validity"]
    # a nudge: the next check says MOVED, and the rig gives no mm again
    assert api.rig_check(p, mock_shift_px=[2.5, 0.0])["status"] == "MOVED"
    moved = api.measure_run("flatness", p, scans=[new], runs_dir=runs)
    assert moved["exit_code"] == 5 and moved["result"]["reason"] == "rig_unverified"
    assert moved["result"]["message"].startswith(PHRASE) and "MOVED" in moved["result"]["message"]
    recs = rigstate.read(s.store, p.geometry_id)
    assert [r["event"] for r in recs] == ["calib_pose"] * 3 + ["calibration", "rig_check",
                                                              "session", "rig_check", "rig_check"]
    assert [r.get("status") for r in recs[4:]] == ["REFERENCE_SET", None, "OK", "MOVED"]
    ok_rec = recs[6]
    assert ok_rec["calibration_id"] == rig.cal.id == ok_rec["reference_calibration_id"]
    assert ok_rec["reference_id"].endswith("_rigref") and ok_rec["tol_px"] == 0.5
    assert ok_rec["reference_trust"] == {"mobile": True, "fixed": True}
    assert ok_rec["mount"] == "mobile" and recs[4]["requested"] is False
    assert all(json.loads(line) for line in log.read_text(encoding="utf-8").splitlines())


def test_a_reference_stored_in_doubt_verifies_nothing_in_any_session(rig):
    """The laundering: a mobile rig is carried (unverified), a rig check stores a reference
    of wherever it stands now - by itself, as the first check of the calibration, or on
    request - and a later check against THAT reference says OK. It verifies nothing, in that
    session or after the next marker."""
    api._session = rig.s
    s, p = rig.s, rig.mobile
    api.rig_session(p, "carried to another bench")
    r = api.rig_check(p)                            # the calibration's first check: stores one
    assert r["status"] == "REFERENCE_SET" and r["reference_trust"]["mobile"] is False
    assert "NOT verified" in r["warnings"][0] and not api.rig_state(p)["ok"]
    assert api.rig_check(p)["status"] == "OK"       # the rig is where that reference was taken
    assert not api.rig_state(p)["ok"]
    api.rig_session(p, "the next day")
    assert api.rig_check(p)["status"] == "OK"
    g = api.rig_state(p)
    assert not g["ok"] and g["code"] == "rig_unverified"
    assert "stored while the rig was not verified" in g["message"]
    r = api.rig_check(p, set_reference=True)        # on request: the same
    assert r["status"] == "REFERENCE_SET" and r["reference_trust"] == {"mobile": False,
                                                                       "fixed": True}
    api.rig_session(p, "and the next")
    assert api.rig_check(p)["status"] == "OK" and not api.rig_state(p)["ok"]
    m = api.measure_run("flatness", p, runs_dir=rig.tmp / "runs", scan_kind="fast")
    assert m["exit_code"] == 5 and m["result"]["reason"] == "rig_unverified"
    assert m["result"]["scans"] == []               # nothing was scanned


def test_a_loose_tolerance_or_another_calibrations_reference_verifies_nothing(rig):
    from engine.calibrate import stereo
    api._session = rig.s
    s, p = rig.s, rig.mobile
    assert api.rig_check(p)["status"] == "REFERENCE_SET"       # trusted: right after calibrating
    api.rig_session(p, "moved")
    assert api.rig_check(p, mock_shift_px=[3.0, 0.0])["status"] == "MOVED"
    r = api.rig_check(p, tol_px=50.0, mock_shift_px=[3.0, 0.0])        # 3 px is "in place" at 50
    assert r["status"] == "OK" and any("does not verify" in w for w in r["warnings"])
    g = api.rig_state(p)
    assert not g["ok"] and "MOVED" in g["message"] and "only 0.5 px or tighter" in g["message"]
    assert rigstate.read(s.store, p.geometry_id)[-1]["tol_px"] == 50.0
    assert api.rig_check(p)["status"] == "OK" and api.rig_state(p)["ok"]       # in place, at 0.5 px
    # a second calibration with no reference of its own: the check compares with the first
    # one's reference and says OK - for a calibration it was never taken with
    d = s.store.new_dir("calibrations", p.geometry_id, label="stereo")
    c2 = stereo.StereoCalibration.from_json(
        {**rig.cal.to_json(), "id": f"{p.geometry_id}/{d.name}",
         "t": (np.ravel(rig.cal.t) * 1.05).tolist()}, d)
    s.store.write_json(d / "stereo_calibration.json", c2.to_json())
    api.rig_session(p, "moved again")
    r = api.rig_check(p)
    assert r["status"] == "OK" and r["calibration_id"] == c2.id
    assert r["reference_calibration_id"] == rig.cal.id
    assert any("the reference was taken with calibration" in w for w in r["warnings"])
    g = api.rig_state(p)
    assert not g["ok"] and f"taken with calibration {rig.cal.id}" in g["message"]
    m = api.measure_run("flatness", p, runs_dir=rig.tmp / "runs", scan_kind="fast")
    assert m["exit_code"] == 5 and m["result"]["reason"] == "rig_unverified"


def test_a_mobile_rig_calibrated_at_a_new_place_gets_that_calibrations_reference(rig):
    """Carried elsewhere and calibrated there: the newest reference is the first place's.
    The next rig check (HOME) must not compare with it - a MOVED / UNKNOWN against where
    the rig stood before would undo what the calibration just verified - but store the new
    calibration's own reference, trusted because the rig is verified."""
    from engine.calibrate import stereo
    api._session = rig.s
    s, p = rig.s, rig.mobile
    first = api.rig_check(p)
    assert first["status"] == "REFERENCE_SET"       # the first place
    api.rig_session(p, "carried to another bench")

    def calibration_entry():
        d = s.store.new_dir("calibrations", p.geometry_id, label="stereo")
        c = stereo.StereoCalibration.from_json(
            {**rig.cal.to_json(), "id": f"{p.geometry_id}/{d.name}"}, d)
        s.store.write_json(d / "stereo_calibration.json", c.to_json())
        return c
    c2 = calibration_entry()
    _calibrated(rig, p, c2.id, None, ["n1", "n2", "n3"])       # its poses captured there, now
    assert api.rig_state(p)["ok"] and c2.id in api.rig_state(p)["state"]["verified_by"]
    r = api.rig_check(p, mock_shift_px=[6.0, 0.0])  # the scene is not the first place's
    assert r["status"] == "REFERENCE_SET" and r["reference_reason"] == "new_calibration", r
    assert r["previous_reference_id"] == first["reference_id"] and r["warnings"] == []
    assert r["reference_trust"]["mobile"] is True and r["calibration_id"] == c2.id
    assert api.rig_state(p)["ok"]
    assert api.rig_check(p, mock_shift_px=[6.0, 0.0])["status"] == "OK"        # in place there
    assert "rig check OK" in api.rig_state(p)["state"]["verified_by"]
    # not verified (carried again): the old reference is compared with, as on any rig
    api.rig_session(p, "carried again")
    calibration_entry()                             # no poses on the record: verifies nothing
    r = api.rig_check(p, mock_shift_px=[6.0, 0.0])
    assert r["status"] == "OK" and r["reference_calibration_id"] == c2.id
    assert not api.rig_state(p)["ok"]


def _mobile_wall_preset(tmp: Path) -> Path:
    """A presets folder with one small MOBILE preset (the fixture wall preset, average 1)."""
    folder = tmp / "presets"
    folder.mkdir()
    d = json.loads((FIXTURE_PRESETS / "cell_wall_2p7ft_vstripe_fullrange.json").read_text())
    d["name"] = "mob"
    d["camera"]["average"] = 1
    d.setdefault("rig", {})["mount"] = "mobile"
    (folder / "mob.json").write_text(json.dumps(d), encoding="utf-8")
    return folder


def test_a_calibration_verifies_only_from_poses_captured_in_the_session(tmp_path):
    """A real (mock) calibration: its poses and the calibration go on the rig-state record,
    and that verifies the mobile rig. Solving the same stored poses again after the rig was
    carried writes a new calibration entry - and verifies nothing."""
    s = api.configure(tmp_path / "library", mock=True, bench=MockBench(seed=3),
                      presets_dir=_mobile_wall_preset(tmp_path))
    p = api.load_preset("mob")
    api.mock_bench().use_stereo()
    cal1 = api.calibrate_stereo(p, mock_poses=6, steps=4)["calibration"]
    recs = rigstate.read(s.store, p.geometry_id)
    assert [r["event"] for r in recs] == ["calib_pose"] * 6 + ["calibration"]
    assert recs[-1]["calibration_id"] == cal1.id
    assert recs[-1]["poses"] == [r["pose_id"] for r in recs[:6]]
    assert {i["id"] for i in api.list_library("calib_poses")} >= set(recs[-1]["poses"])
    g = api.rig_state(p)
    assert g["ok"] and cal1.id in g["state"]["verified_by"]
    out = api.rig_session(p, "carried to another bench")
    assert out["gate"]["ok"] is False
    n = len(api.list_library("calib_poses"))
    cal2 = api.calibrate_stereo(p)["calibration"]   # nothing captured: the stored set again
    assert cal2.id != cal1.id and len(api.list_library("calib_poses")) == n
    g = api.rig_state(p)
    assert not g["ok"] and g["state"]["calibration_id"] == cal2.id
    assert "solving stored poses again verifies nothing" in g["message"]
    with pytest.raises(rules.RuleViolation, match=PHRASE):     # and no mm cloud either
        api.scan_metric(p, "after_resolve")


def test_a_rehome_verifies_only_against_a_trusted_reference(tmp_path):
    """A small camera turn, re-homed against the reference stored right after calibrating:
    verified by the re-home. The same against a reference of the carried (unverified) rig:
    a re-homed calibration is written, and the rig stays unverified."""
    import test_rehome_mock as T
    from engine.capture.mock_stereo import StereoRig
    bench = MockBench(seed=7)
    s = api.configure(tmp_path / "library", mock=True, bench=bench,
                      presets_dir=_mobile_wall_preset(tmp_path))
    p = api.load_preset("mob")
    rig0 = StereoRig.bench()
    T._set_camera(bench, rig0)
    cal0 = RS.calibrated_now(s, p)
    assert api.rig_check(p)["status"] == "REFERENCE_SET" and api.rig_state(p)["ok"]
    nudge = T._rot(pan_deg=0.045, tilt_deg=-0.025, roll_deg=0.04)
    T._set_camera(bench, rig0, nudge)
    assert api.rig_check(p)["status"] == "MOVED" and not api.rig_state(p)["ok"]
    r = api.rig_check(p, rehome=True)
    assert r["status"] == "REHOMED" and r["rehome_from"] == cal0.id
    assert r["reference_trust"]["mobile"] is True
    g = api.rig_state(p)
    assert g["ok"] and "re-home" in g["state"]["verified_by"]
    assert g["state"]["calibration_id"] == r["calibration_id"] != cal0.id
    # carried: a reference of wherever it stands now, another nudge, a re-home against it
    api.rig_session(p, "carried to another bench")
    r = api.rig_check(p, set_reference=True)
    assert r["status"] == "REFERENCE_SET" and r["reference_trust"]["mobile"] is False
    T._set_camera(bench, rig0)                      # turned back: a move against that reference
    r = api.rig_check(p, rehome=True)
    assert r["status"] == "REHOMED" and r["reference_trust"]["mobile"] is False
    g = api.rig_state(p)
    assert not g["ok"] and g["code"] == "rig_unverified"
    assert "stored while the rig was not verified" in g["message"]
    assert [x.get("status") for x in rigstate.read(s.store, p.geometry_id)
            if x["event"] == "rig_check"] == ["REFERENCE_SET", "MOVED", "REHOMED",
                                              "REFERENCE_SET", "REHOMED"]


# ================================================================== the three modes

FLAT_CFG = {"station_id": "flat", "library": None, "mock_library": None, "threshold_mm": 0.5,
            "job": {"type": "metric", "scan": "fast", "measure": "plane"},
            "part": {"role": "test", "label": "panel"}}


def _station(tmp: Path, preset: str, name: str) -> Path:
    f = tmp / f"{name}.json"
    f.write_text(json.dumps({**FLAT_CFG, "station_id": name, "preset": preset,
                             "runs_dir": str(tmp / "runs" / name)}), encoding="utf-8")
    return f


def _last(runs: Path) -> dict:
    d = sorted(x for x in runs.iterdir() if x.is_dir())[-1]
    return json.loads((d / "result.json").read_text(encoding="utf-8"))


def test_mobile_and_fixed_gating_in_station_quick_check_and_measure(rig, monkeypatch):
    """Station and Quick check PULL, Measure is NOT_MEASURABLE, on a mobile rig that was not
    verified - before anything is scanned. A fixed rig: Station unchanged, the other two
    refuse it once a rig check said MOVED."""
    s, tmp = rig.s, rig.tmp
    lib = str(s.store.root)
    base = ["--mock", "--library", lib]
    walls = {p.geometry_id: RS.oracle_scan(s, p, [], "wall") for p in (rig.fixed, rig.mobile)}
    scanned = []

    def fake_scan_fast(zone, p, label, steps, periods, average, **kw):
        scanned.append(p.name)
        return walls[p.geometry_id]
    monkeypatch.setattr(api, "scan_fast", fake_scan_fast)
    monkeypatch.setattr(cli, "_mock_place_metric", lambda ctx: None)
    mob, fix = _station(tmp, "mob", "mob_station"), _station(tmp, PRESET, "fix_station")
    qruns, mruns = tmp / "runs" / "quick", tmp / "runs" / "measure"
    quick = ["--check", "flatness", "--threshold", "0.5", "--runs-dir", str(qruns)]
    meas = ["--what", "flatness", "--scan-kind", "fast", "--runs-dir", str(mruns)]

    # ---- mobile, set down again: nothing in any mode
    rigstate.new_session(s.store, rig.mobile, "moved")
    api._session = None
    code, out = _cli([*base, "run", str(mob)])
    res = _last(tmp / "runs" / "mob_station")
    assert code == 5 and res["result"] == "INVALID" and res["disposition"] == "PULL"
    assert res["pull_reasons"][0]["reason"] == "rig_unverified"
    assert res["pull_reasons"][0]["message"].startswith(PHRASE) and PHRASE in out
    assert "operating_mode" not in res              # a station run's document is unchanged
    code, out = _cli([*base, "quick-check", "mob", *quick])
    res = _last(qruns)
    assert code == 5 and res["disposition"] == "PULL" and res["operating_mode"] == "quick_check"
    assert res["pull_reasons"][0]["reason"] == "rig_unverified" and PHRASE in res["invalid_reason"]
    code, out = _cli([*base, "measure-run", "mob", *meas])
    doc = _last(mruns)
    assert code == 5 and doc["status"] == "NOT_MEASURABLE" and doc["reason"] == "rig_unverified"
    assert "NOT MEASURABLE (rig_unverified)" in out and PHRASE in out
    assert scanned == []                            # refused before the rig was touched
    code, out = _cli([*base, "--preset", "mob", "rig-session"])
    assert code == 1 and "NOT allowed" in out and "how to fix" in out

    # ---- verified (a rig check OK against the calibration's reference): every mode runs
    _check(rig, "OK")
    assert _cli([*base, "--preset", "mob", "rig-session"])[0] == 0
    code, _ = _cli([*base, "run", str(mob)])
    res = _last(tmp / "runs" / "mob_station")
    assert code == 0 and res["result"] == "PASS" and res["disposition"] == "CONTINUE"
    code, _ = _cli([*base, "quick-check", "mob", *quick])
    assert code == 0 and _last(qruns)["disposition"] == "CONTINUE"
    code, _ = _cli([*base, "measure-run", "mob", *meas])
    assert code == 0 and _last(mruns)["status"] == "MEASURED"
    assert scanned == ["mob", "mob", "mob"]

    # ---- fixed (another rig): trusted, until a rig check says MOVED
    scanned.clear()
    assert _cli([*base, "quick-check", PRESET, *quick])[0] == 0
    _check(rig, "MOVED", preset=rig.fixed, moved="camera")
    code, _ = _cli([*base, "run", str(fix)])        # Station mode on a fixed rig: unchanged
    res = _last(tmp / "runs" / "fix_station")
    assert code == 0 and res["disposition"] == "CONTINUE" and "operating_mode" not in res
    code, out = _cli([*base, "quick-check", PRESET, *quick])
    res = _last(qruns)
    assert code == 5 and res["disposition"] == "PULL"
    assert res["pull_reasons"][0]["reason"] == "rig_check" and "MOVED" in res["invalid_reason"]
    code, out = _cli([*base, "measure-run", PRESET, *meas])
    assert code == 5 and _last(mruns)["reason"] == "rig_check"
    assert scanned == [PRESET, PRESET]              # the first quick check and the station run
    # the mobile preset: a MOVED un-verifies it in Station mode too - as rig_unverified,
    # which no recipe can turn into a reject
    _check(rig, "MOVED", moved="camera")
    code, _ = _cli([*base, "run", str(mob)])
    res = _last(tmp / "runs" / "mob_station")
    assert code == 5 and res["pull_reasons"][0]["reason"] == "rig_unverified"
    assert res["pull_reasons"][0]["message"].startswith(PHRASE)
    assert "MOVED" in res["pull_reasons"][0]["message"]


def test_no_mm_cloud_from_an_unverified_mobile_rig(rig, monkeypatch):
    """The engine's own capture verbs are gated too (not only the three modes): a metric
    scan, a fast scan, an accuracy check - refused before the rig is touched; and a stored
    capture is only triangulated when it comes from the session the rig is verified in."""
    from engine.calibrate import flow
    from engine.reconstruct import metric_scan
    api._session = rig.s
    s, p = rig.s, rig.mobile
    touched = []
    monkeypatch.setattr(flow, "capture_patterns",
                        lambda *a, **k: touched.append(a) or pytest.fail("the rig was touched"))
    before = RS.oracle_scan(s, p, [], "before")
    assert metric_scan._require_rig_of(s, before, "triangulate")["mount"] == "mobile"
    # fixed: no gate; the scan records that nothing objects
    assert metric_scan._require_rig(s, rig.fixed, "scan_metric") == {"mount": "fixed",
                                                                     "trusted": True}
    rigstate.new_session(s.store, p, "carried")
    for call in (lambda: api.scan_fast(None, p, "direct"),
                 lambda: api.scan_metric(p, "direct"),
                 lambda: api.accuracy_check("plane", {}, repeats=1, label="direct"),
                 lambda: metric_scan._require_rig_of(s, before, "triangulate")):
        api.load_preset("mob")
        with pytest.raises(rules.RuleViolation, match=PHRASE):
            call()
    assert touched == []
    # verified again: a capture from before that is still not covered, a later one is
    _check(rig, "OK")
    with pytest.raises(rules.RuleViolation, match="stored scan"):
        metric_scan._require_rig_of(s, before, "triangulate")
    after = RS.oracle_scan(s, p, [], "after")
    stamp = metric_scan._require_rig_of(s, after, "triangulate")
    assert stamp["mount"] == "mobile" and "rig check OK" in stamp["verified_by"]
    # the command line: refused (2), never a cloud "in mm"
    rigstate.new_session(s.store, p, "carried again")
    lib = str(s.store.root)
    api._session = None
    for argv in (["scan-fast", "direct", "--axis", "xy"], ["scan-metric", "direct"],
                 ["accuracy", "plane", "--tol", "0.5"]):
        code, out = _cli(["--mock", "--library", lib, "--preset", "mob", *argv])
        assert code == 2 and "points in mm" not in out and "GO" not in out, argv
    assert touched == []


def test_a_fixed_rig_after_moved_gives_mm_and_every_scan_says_so(rig, monkeypatch):
    """The first real-rig run: after a rig check said MOVED on a FIXED rig, Measure and Quick
    check refused (kept), but the plain scans (scan-fast, scan-metric, the station
    measurement) said "points in mm" with no word, and their entries recorded rig None. The
    design (README *Operating modes*): a fixed rig is trusted until a check objects, and then
    only Measure and Quick check refuse its mm - the plain verbs are never refused. So they
    keep the mm and say so: a printed warning in the rig check's own words with the way out,
    first in meta "warnings", the "rig" stamp (trusted false), both at the top of --json too;
    a verb that measures a stored scan says it of a scan made while the objection stood. An
    answer (a rig check OK) clears it for what is scanned after it."""
    from engine.reconstruct import metric_scan
    api._session = rig.s
    s, p = rig.s, rig.fixed
    words = ("camera and projector moved: the camera's view ~0.7 px, the projector's picture "
             "~18.6 px against the scene: recalibrate before measuring")
    _check(rig, "MOVED", preset=p, moved="camera and projector", message=words)
    st = metric_scan._require_rig(s, p, "scan_fast")
    assert st["mount"] == "fixed" and st["trusted"] is False and st["code"] == "rig_check"
    w = st["warning"]
    assert w.startswith(rigstate.IN_DOUBT) and "rig check MOVED at" in w and words in w
    assert "| fix: put the rig right, then `scan rig-check`" in w
    assert "--set-reference" in w and "calibrate-stereo" in w   # every way out is named
    # a real (mock) capture: the mm are given, and say so
    ms = api.scan_fast(None, p, "after_moved", axis="xy")
    assert ms.units == "mm" and ms.meta["n_points"] > 1000
    assert ms.meta["rig"] == st and ms.meta["warnings"][0] == w
    stored = json.loads((ms.path / "meta.json").read_text(encoding="utf-8"))
    assert stored["rig"]["trusted"] is False and stored["warnings"][0] == w
    # the command line: printed, and at the top of --json (the scan is the one above)
    monkeypatch.setattr(api, "scan_fast", lambda *a, **k: ms)
    monkeypatch.setattr(api, "scan_metric", lambda *a, **k: ms)
    base = ["--mock", "--library", str(s.store.root), "--preset", PRESET]
    api._session = None
    for argv in (["scan-fast", "x", "--axis", "xy"], ["scan-metric", "x"]):
        code, out = _cli([*base, *argv])
        assert code == 0 and "points in mm" in out, argv
        assert f"! {rigstate.IN_DOUBT}" in out and "rig check MOVED at" in out, out
        code, out = _cli([*base, "--json", *argv])
        doc = json.loads(out)
        assert doc["warnings"][0] == w and doc["rig"]["trusted"] is False, argv
        assert doc["metric_scan"]["meta"]["rig"]["trusted"] is False
    monkeypatch.undo()
    api._session = s
    # a verb that measures a STORED scan made while the objection stood: the same words
    faces = RS.oracle_scan(s, p, "scene:endfaces", "faces_after_moved")
    m = api.station_measure(faces)
    assert m.warnings[0].startswith(rigstate.IN_DOUBT) and "rig check MOVED" in m.warnings[0]
    assert m.meta["rig"]["trusted"] is False and len(m.cells) == 2      # measured as before
    # answered: what is scanned now is trusted again; the scan made while the objection
    # stood stays in doubt (scan again)
    _check(rig, "OK", preset=p)
    assert metric_scan._require_rig(s, p, "scan_fast") == {"mount": "fixed", "trusted": True}
    m = api.station_measure(faces)
    assert m.meta["rig"]["trusted"] is False and "in doubt" in m.warnings[0]
    assert "scan again" in m.warnings[0]
    m = api.station_measure(RS.oracle_scan(s, p, "scene:endfaces", "faces_after_ok"))
    assert m.meta["rig"] == {"mount": "fixed", "trusted": True}
    assert not any(rigstate.IN_DOUBT in x for x in m.warnings)


def test_accuracy_and_the_check_block_say_a_fixed_rig_in_doubt(rig, monkeypatch):
    """`accuracy` and `accuracy-block` measure metric scans. On a FIXED rig after a MOVED rig
    check their verdict is made as before, and they say what it rests on: the rig check's
    words first in their warnings (printed), the rig stamp trusted false at the top of the
    result (--json) and in the stored report. Before, neither said anything."""
    from engine.capture.mock_stereo import check_block_placement, check_block_shapes
    from engine.measure import checkblock as cb
    api._session = rig.s
    s, p = rig.s, rig.fixed
    spec = cb.design()
    refs = {"values": cb.references(spec)}
    wall = RS.oracle_scan(s, p, [], "wall")
    block = RS.oracle_scan(s, p, check_block_shapes(spec, *check_block_placement()), "block")
    acc0 = api.accuracy_check("plane", {}, scans=[wall])
    blk0 = api.accuracy_block(spec, refs, scans=[block])
    for r in (acc0, blk0):
        assert r["rig"] == {"mount": "fixed", "trusted": True}
        assert not any(rigstate.IN_DOUBT in w for w in r["warnings"])
    words = "projector (or its picture) moved ~18.6 px against the scene"
    _check(rig, "MOVED", preset=p, moved="projector", message=words)
    acc = api.accuracy_check("plane", {}, scans=[wall])
    blk = api.accuracy_block(spec, refs, scans=[block])
    w = acc["warnings"][0]
    assert w.startswith(rigstate.IN_DOUBT) and "rig check MOVED at" in w and words in w
    for r, r0 in ((acc, acc0), (blk, blk0)):
        assert r["warnings"][0] == w and r["rig"]["trusted"] is False
        stored = json.loads(Path(r["file"]).read_text(encoding="utf-8"))
        assert stored["rig"]["trusted"] is False and stored["warnings"][0] == w
    # never a different judgment
    assert acc["verdict"] == acc0["verdict"] and acc["mean_error_mm"] == acc0["mean_error_mm"]
    assert (blk["result"], blk["disposition"]) == (blk0["result"], blk0["disposition"])
    # the command line prints it (exit codes as before) and puts it at the top of --json
    (rig.tmp / "block.json").write_text(json.dumps(spec), encoding="utf-8")
    monkeypatch.setattr(api, "accuracy_check", lambda *a, **k: acc)
    monkeypatch.setattr(api, "accuracy_block", lambda *a, **k: blk)
    base = ["--mock", "--library", str(s.store.root), "--preset", PRESET]
    for argv, code0 in ((["accuracy", "plane"], 0 if acc["verdict"] == "GO" else 1),
                        (["accuracy-block", "--block", str(rig.tmp / "block.json")],
                         D.exit_code(blk))):
        code, out = _cli([*base, *argv])
        assert code == code0 and f"! {rigstate.IN_DOUBT}" in out, (argv, out)
        code, out = _cli([*base, "--json", *argv])
        doc = json.loads(out)
        assert code == code0 and doc["warnings"][0] == w and doc["rig"]["trusted"] is False


def test_a_rig_state_record_that_cant_be_read_is_a_doubt_never_an_error(rig):
    """The fixed rig's record is only read to say whether its mm are in doubt: a record that
    can't be read is a doubt (fail closed), and never stops the scan or the measurement."""
    from engine.reconstruct import metric_scan
    api._session = rig.s
    s, p = rig.s, rig.fixed
    f = rigstate.log_path(s.store, p.geometry_id)
    assert not f.exists()
    f.mkdir(parents=True)                           # the record is a folder: not readable
    st = metric_scan._require_rig(s, p, "scan_fast")
    assert st["mount"] == "fixed" and st["trusted"] is False
    assert st["warning"].startswith(rigstate.IN_DOUBT) and "could not be read" in st["warning"]
    rig_, warnings = metric_scan.stored_doubts(s, [RS.oracle_scan(s, p, [], "unread")])
    assert rig_["trusted"] is False and "could not be read" in warnings[0]


def test_a_work_state_of_a_mobile_rig_is_never_confirmed_unverified(rig):
    """The flow (a recipe run per work state): an unverified mobile rig is a DOUBT - PULL,
    no permission; verified, the same work state is CONFIRMED."""
    api._session = rig.s
    s, tmp = rig.s, rig.tmp
    doc = {"schema_version": 2, "station_id": "mob_line",
           "parameters": {"preset": "mob", "runs_dir": str(tmp / "cycles")},
           "profiles": {"plate": {"recipes": {"flat": "flat_plate"}}},
           "flow": [{"id": "flat", "title": "Plate flat?", "verify": {"recipe": "flat"},
                     "permit": "GO"}]}
    cfg = tmp / "mob_line.json"
    cfg.write_text(json.dumps(doc), encoding="utf-8")
    assert api.flow_check(cfg)["valid"], api.flow_check(cfg)["problems"]
    rigstate.new_session(s.store, rig.mobile, "moved")
    before = RS.oracle_scan(s, rig.mobile, "scene:plate", "plate_before")
    out = api.flow_run(cfg, scans={"flat": before}, project_faults=False)
    res = out["result"]
    assert out["exit_code"] == 5 and res["disposition"] == "PULL"
    assert res["permissions"] == {"GO": False} and res["steps"][0]["outcome"] == "DOUBT"
    assert any(r["reason"] == "rig_unverified" and r["message"].startswith(PHRASE)
               for r in res["pull_reasons"])
    # the same through a recipe run on its own
    r = api.recipe_run("flat_plate", rig.mobile, scan=before, project_feedback=False,
                       runs_dir=tmp / "runs")
    assert r["exit_code"] == 5 and r["result"]["result"] == "INVALID"
    assert r["result"]["pull_reasons"][0]["reason"] == "rig_unverified"
    # verified, and scanned after that: CONFIRMED, the permission granted
    _check(rig, "OK")
    after = RS.oracle_scan(s, rig.mobile, "scene:plate", "plate_after")
    out = api.flow_run(cfg, scans={"flat": after}, project_faults=False)
    assert out["exit_code"] == 0 and out["result"]["disposition"] == "CONTINUE", out["result"]
    assert out["result"]["permissions"] == {"GO": True}
    # ... but the scan made before the verification is still not covered by it
    out = api.flow_run(cfg, scans={"flat": before}, project_faults=False)
    assert out["exit_code"] == 5 and "stored scan" in json.dumps(out["result"]["pull_reasons"])

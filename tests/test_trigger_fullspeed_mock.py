"""Hardware trigger at full speed, mock only (docs/TRIGGER_FULLSPEED_TEST.md):
the trigger test's verdicts with a fake triggered camera, the triggered scan of a
fast-scan pack decoding like the soft-sync scan of the same pack, triggered timing
validation in the pack builder, and the CLI guards. Nothing here touches hardware."""

import json

import numpy as np
import pytest

from engine import api
from engine.capture import hwtrigger as hw
from engine.capture import patternmode as pm
from engine.capture import softsync as ss
from tools import make_flash_pack as mfp
from ui import cli

from conftest import PRESET
from engine.capture.mock_bench import MockBench

from test_flash_e2e_mock import scene, _use  # noqa: F401  (module fixture + helper)


# ------------------------------------------------------------------ trigger test

def _stats(cam, seconds=1.0, expect=None):
    cam.arm()
    try:
        ts, ids, to = hw.count_pulses(cam, seconds, timeout_ms=max(1000, 3 * int(expect or 0)))
    finally:
        cam.disarm()
    return hw.trigger_stats(ts, ids, seconds=seconds, expect_period_ms=expect, timeouts=to)


def test_trigger_stats_ok():
    r = _stats(hw.FakeTriggeredCamera(5000.0), 1.0, 5.0)
    assert r["verdict"] == "OK" and 175 <= r["count"] <= 200
    assert abs(r["rate_hz"] - 200) < 1 and abs(r["period_ms"]["mean"] - 5.0) < 0.01
    assert r["missed_pulses"] == 0 and r["frame_id_gaps"] == 0


def test_trigger_stats_no_pulses():
    cam = hw.FakeTriggeredCamera(pulses=False)
    r = _stats(cam, 3.0, 10.0)
    assert r["verdict"] == "NO PULSES" and r["count"] == 0 and r["timeouts"] == 3
    assert "Line3" in r["notes"][0] and "3.3 V" in r["notes"][0]
    assert cam.trigger.get("TriggerMode", "Off") == "Off" and not cam.armed


def test_trigger_stats_rate_mismatch_and_missed():
    r = _stats(hw.FakeTriggeredCamera(10000.0), 1.0, 5.0)     # GUI at 10 ms, expected 5
    assert r["verdict"] == "RATE MISMATCH" and r["missed_pulses"] > 50
    r = _stats(hw.FakeTriggeredCamera(5000.0, skip={10, 20, 21}), 1.0, 5.0)
    assert r["verdict"] == "RATE MISMATCH" and r["missed_pulses"] == 3
    r = _stats(hw.FakeTriggeredCamera(5000.0, drop={7}), 1.0, 5.0)
    assert r["verdict"] == "RATE MISMATCH" and r["frame_id_gaps"] == 1


def test_api_trigger_test_restores_trigger_mode(tmp_path):
    api.configure(tmp_path / "lib", mock=True)
    api.load_preset(PRESET)
    try:
        r = api.trigger_test("Line0", 1.0, 10.0)
        assert r["verdict"] == "OK" and r["trigger_mode_restored"] == "Off" and r["mock"]
        api.mock_bench().trigger_sim = {"pulses": False}
        assert api.trigger_test("Line3", 1.0, 10.0)["verdict"] == "NO PULSES"
        with pytest.raises(ValueError):
            api.trigger_test("Line1", 1.0)                    # Line1 is an output
    finally:
        api._session = None


def test_cli_trigger_test_exit_codes(tmp_path, capsys, monkeypatch):
    lib = str(tmp_path / "lib")
    assert cli.main(["--mock", "--library", lib, "trigger-test", "--seconds", "1",
                     "--expect-period-ms", "5"]) == 0
    assert "OK" in capsys.readouterr().out
    monkeypatch.setattr(MockBench, "trigger_sim", {"period_us": 10000}, raising=False)
    assert cli.main(["--mock", "--library", lib, "trigger-test", "--seconds", "1",
                     "--expect-period-ms", "5"]) == 1
    assert "RATE MISMATCH" in capsys.readouterr().out
    api._session = None


# ------------------------------------------------------------------ timing validation

@pytest.mark.parametrize("P", [10000, 5000, 4500])
def test_triggered_fastscan_timing_accepted(P, tmp_path):
    d = mfp.make_fastscan_pack(out_root=tmp_path, label=f"t{P}", trigger_mode="trigout",
                               period_us=P)
    m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    t = m["timing"]
    assert t["pattern_period_us"] == t["pattern_exposure_us"] == P
    h = t["hardware_trigger"]
    assert h["set_ms"] == 16 * P / 1000 and h["camera"]["trigger_source"] == "Line0"
    assert h["camera"]["exposure_us"] <= P - mfp.CAMERA_TRIGGER_GUARD_US
    assert ss.fastscan_manifest_problems(m, "full") == []      # the contract still holds
    assert mfp.verify_fastscan_pack(d) == 0
    ss.load_pack_layout(d, "full")
    assert "scan-trigger" in (d / "procedure.txt").read_text(encoding="utf-8")


@pytest.mark.parametrize("bd,P,X,E,why", [
    (1, 4000, None, None, "full-frame period"),              # camera cannot keep up
    (1, 5000, None, 4950, "next pattern"),                   # camera exposure too long
    (1, 5000, 4900, None, "period - exposure"),              # DLP: gap < 230 us
    (8, 5000, None, None, "8333"),                           # 8-bit minimum
])
def test_triggered_timing_refused(bd, P, X, E, why):
    with pytest.raises(mfp.TimingError, match=why):
        mfp.triggered_timing(bd, P, pattern_exposure_us=X, camera_exposure_us=E)


def test_external_trigger_pack_and_8bit_period(tmp_path):
    d = mfp.make_fastscan_pack(out_root=tmp_path, label="ext", trigger_mode="external",
                               period_us=10000)
    m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    assert m["timing"]["trigger"] == "external" and ss.fastscan_manifest_problems(m) == []
    m["timing"].pop("hardware_trigger")                     # a bare "external" is refused
    assert ss.fastscan_manifest_problems(m)
    d8 = mfp.make_pack(mfp.DEFAULT_PRESET, out_root=tmp_path, period_us=10000)
    m8 = json.loads((d8 / "manifest.json").read_text(encoding="utf-8"))
    assert m8["timing"]["uniform"]["period_us"] == 10000
    assert m8["timing"]["hardware_trigger"]["camera"]["exposure_us"] >= 9500
    with pytest.raises(mfp.TimingError):
        mfp.make_pack(mfp.DEFAULT_PRESET, out_root=tmp_path, period_us=5000)
    with pytest.raises(mfp.TimingError):
        mfp.make_fastscan_pack(out_root=tmp_path, trigger_mode="trigout", period_us=4000)
    assert [r["set_ms"] for r in hw.expected_set_ms()] == [160.0, 80.0, 72.0]


# ------------------------------------------------------------------ triggered scan

def test_scan_trigger_matches_soft_sync(scene):  # noqa: F811
    """The same pack and scene: the hardware-triggered decode (one frame per pattern,
    2 loops, loop start found by the sync markers) equals the soft-sync decode."""
    p = _use(scene)
    soft = api.fast_scan_flash(str(scene["pack"]), "zone", "soft", p, roi=scene["roi"])
    trig = api.scan_trigger(str(scene["pack"]), "zone", "trig", p, repeats=2,
                            acknowledge=pm.ACK_PHRASE)
    assert trig["mode"] == "flash_trigger" and trig["decode"]["variant_detected"] == "zone"
    t = trig["timing"]
    assert t["frames"] == 48 and t["drops"] == 0 and t["set_ms"] == 160.0
    assert abs(t["per_frame_ms"]["mean"] - 10.0) < 0.01
    a, b = soft["scan"].field, trig["scan"].field
    x0, y0, x1, y1 = scene["roi"]
    win = np.zeros(a.mask.shape, bool)
    win[y0:y1, x0:x1] = True
    both = a.mask & b.mask & win
    assert both.sum() >= 0.9 * max(a.mask.sum(), (b.mask & win).sum()) > 5000
    d = (a.corr_x - b.corr_x)[both]
    assert abs(float(np.median(d))) < 0.01 and float(np.percentile(np.abs(d), 95)) < 0.15
    meta = trig["scan"].meta
    assert meta["mode"] == "flash_trigger" and meta["units"] == "mm"
    assert meta["trigger"]["timing"]["frames"] == 48


def test_scan_trigger_guards(scene, monkeypatch):  # noqa: F811
    p = _use(scene)
    for ack in (None, "yes", True):
        with pytest.raises(pm.PatternModeRefused):
            api.scan_trigger(str(scene["pack"]), "zone", "x", p, acknowledge=ack)
    api.mock_bench().trigger_sim = {"pulses": False}
    with pytest.raises(pm.MissingFrames, match="no trigger"):
        api.scan_trigger(str(scene["pack"]), "zone", "x", p, acknowledge=pm.ACK_PHRASE,
                         first_timeout_s=0.5)
    api.mock_bench().trigger_sim = None


def test_cli_scan_trigger_refuses_without_acknowledgement(scene, capsys):  # noqa: F811
    _use(scene)
    lib = str(scene["s"].store.root)
    code = cli.main(["--mock", "--library", lib, "scan-trigger", "x", "--pack",
                     str(scene["pack"])])
    assert code == 2 and "pattern-mode-set-by-human" in capsys.readouterr().err


def test_scan_trigger_8bit_hybrid_pack(scene):  # noqa: F811
    """The older 8-bit hybrid pack (play once, trigger order) decodes into a metric scan."""
    p = _use(scene)
    d = mfp.make_pack(mfp.DEFAULT_PRESET, out_root=scene["tmp"] / "packs8", period_us=10000)
    r = api.scan_trigger(str(d), None, "trig8", p, acknowledge=pm.ACK_PHRASE)
    assert r["timing"]["frames"] == 16 and r["timing"]["set_ms"] == 160.0
    assert r["decode"]["basis"].startswith("hybrid") and r["n_points"] > 5000

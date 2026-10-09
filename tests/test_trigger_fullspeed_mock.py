"""Hardware trigger at full speed, mock only (docs/TRIGGER_FULLSPEED_TEST.md):
the trigger test's verdicts with a fake triggered camera, the triggered scan of a
fast-scan pack decoding like the soft-sync scan of the same pack, triggered timing
validation in the pack builder, and the CLI guards. Nothing here touches hardware."""

import json
import re
from pathlib import Path

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

ROOT = Path(__file__).resolve().parents[1]
_SCENE_CAMERA = hw.mock_scene_camera


# ------------------------------------------------------------------ trigger test

def _stats(cam, seconds=1.0, expect=None):
    cam.configure_trigger("Line0", 1000.0)
    cam.arm()
    try:
        ts, ids, to, inc = hw.count_pulses(cam, seconds,
                                           timeout_ms=max(1000, 3 * int(expect or 0)))
    finally:
        cam.disarm()
    return hw.trigger_stats(ts, ids, seconds=seconds, expect_period_ms=expect, timeouts=to,
                            incomplete=inc)


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



# ------------------------------------------------------------------ review regressions

def test_trigger_stats_stopped_or_late_train_is_not_ok():
    """The pulses must cover the whole window: a train that stops (Repeat off, a contact
    that drops) or starts late is RATE MISMATCH, not OK."""
    r = _stats(hw.FakeTriggeredCamera(10000.0, n_play=16, repeat=False), 5.0, 10.0)
    assert r["verdict"] == "RATE MISMATCH" and r["count"] == 16 and r["timeouts"] >= 1
    assert any("stopped" in n for n in r["notes"])
    r = _stats(hw.FakeTriggeredCamera(10000.0, n_play=16, repeat=False), 3.0)   # no period
    assert r["verdict"] == "RATE MISMATCH" and r["timeouts"] >= 1
    r = _stats(hw.FakeTriggeredCamera(10000.0, lead_us=2e6), 5.0, 10.0)        # 2 s late
    assert r["verdict"] == "RATE MISMATCH" and r["timeouts"] >= 1
    r = _stats(hw.FakeTriggeredCamera(10000.0, lead_us=5e5), 5.0, 10.0)        # 0.5 s late
    assert r["verdict"] == "RATE MISMATCH" and r["timeouts"] == 0
    assert r["expected_count"] == 500 and r["count"] < 460
    assert any("expected" in n for n in r["notes"])
    r = _stats(hw.FakeTriggeredCamera(10000.0), 5.0, 10.0)                      # healthy
    assert r["verdict"] == "OK" and r["count"] >= 495 and r["timeouts"] == 0, r


def test_count_pulses_counts_incomplete_frames():
    """An incomplete frame (USB3 loss at full speed) is counted and reported; the test
    goes on to its verdict instead of aborting."""
    r = _stats(hw.FakeTriggeredCamera(5000.0, incomplete={50, 51}), 1.0, 5.0)
    assert r["verdict"] == "RATE MISMATCH" and r["incomplete_frames"] == 2
    assert r["count"] >= 190 and r["frame_id_gaps"] == 0 and r["missed_pulses"] == 0
    assert any("incomplete" in n for n in r["notes"])


def test_api_trigger_test_reports_incomplete_frames(tmp_path):
    api.configure(tmp_path / "lib", mock=True)
    api.load_preset(PRESET)
    try:
        api.mock_bench().trigger_sim = {"period_us": 5000, "incomplete": {20}}
        r = api.trigger_test("Line0", 1.0, 5.0)
        assert r["verdict"] == "RATE MISMATCH" and r["incomplete_frames"] == 1
        assert r["trigger_mode_restored"] == "Off"
    finally:
        api._session = None


def test_trigger_lines_are_camera_inputs_only(tmp_path):
    """Only the inputs the bench doc allows: Line0 (opto) and Line3. Line2 (pin 3, the
    3.3 V auxiliary output) and Line1 (the opto output) are refused everywhere."""
    assert hw.TRIGGER_LINES == pm.TRIGGER_INPUT_LINES == ("Line0", "Line3")
    for bad in ("Line1", "Line2"):
        with pytest.raises(ValueError):
            hw.check_line(bad)
        with pytest.raises(ValueError):
            hw.FakeTriggeredCamera().configure_trigger(bad, 1000.0)
    doc = (ROOT / "docs" / "TRIGGER_FULLSPEED_TEST.md").read_text(encoding="utf-8")
    offered = set(re.findall(r"--line (Line\d)", doc))
    assert offered and offered <= set(hw.TRIGGER_LINES)
    assert "Never connect the trigger to pin 3 (Line2" in doc
    api.configure(tmp_path / "lib", mock=True)
    api.load_preset(PRESET)
    try:
        with pytest.raises(ValueError, match="pin 3"):
            api.trigger_test("Line2", 1.0)
    finally:
        api._session = None


# 8-bit hybrid pack: the start of the sequence is proven from the frames

def _force_scene_camera(monkeypatch, start, *, repeat=True, pending=None, invert=False):
    def forced(bench, pats, P, **kw):
        kw.update(start=start, repeat=repeat)
        if invert:                                   # the GUI's Invert Pattern on every row
            pats = [255 - np.asarray(p, np.uint8) for p in pats]
        cam = _SCENE_CAMERA(bench, pats, P, **kw)
        if pending is not None:
            cam.pending = lambda: pending
        return cam
    monkeypatch.setattr(hw, "mock_scene_camera", forced)


@pytest.fixture(scope="module")
def pack8(scene):  # noqa: F811
    return mfp.make_pack(mfp.DEFAULT_PRESET, out_root=scene["tmp"] / "packs8a", period_us=10000)


def _n_scans(scene):  # noqa: F811
    return sum(1 for _ in Path(scene["s"].store.root).rglob("*.npz"))


def test_scan_trigger_8bit_aligns_a_looping_sequence(scene, pack8, monkeypatch):  # noqa: F811
    """Repeat on, the camera arms mid-loop (frame 0 shows pattern 5): the white / black
    alignment finds the start, and the scan equals one whose first frame was pattern 0
    (before the fix this was saved as a metric scan with ~31 % valid)."""
    p = _use(scene)
    _force_scene_camera(monkeypatch, 0)
    ref = api.scan_trigger(str(pack8), None, "t8ref", p, repeats=2, acknowledge=pm.ACK_PHRASE)
    _force_scene_camera(monkeypatch, 5)
    r = api.scan_trigger(str(pack8), None, "t8mid", p, repeats=2, acknowledge=pm.ACK_PHRASE)
    assert ref["decode"]["align"]["start_seq"] == 0 and r["decode"]["align"]["start_seq"] == 5
    assert r["decode"]["align"]["agree"] >= hw.ALIGN_MIN_AGREE
    assert r["decode"]["align"]["runner_up"] < hw.ALIGN_MIN_AGREE
    assert r["valid_pct"] > 90 and abs(r["valid_pct"] - ref["valid_pct"]) < 1.0
    a, b = ref["scan"].field, r["scan"].field
    both = a.mask & b.mask
    assert both.sum() >= 0.98 * a.mask.sum()
    # two independent single-exposure captures: the noise floor between two start-0 runs
    # is |d| median ~0.075, p95 ~0.22 column; a mis-assigned set is far off
    d = (a.corr_x - b.corr_x)[both]
    assert abs(float(np.median(d))) < 0.01 and float(np.median(np.abs(d))) < 0.1
    assert float(np.percentile(np.abs(d), 95)) < 0.3


def test_scan_trigger_8bit_single_pass_refuses_a_looping_sequence(scene, pack8,  # noqa: F811
                                                                    monkeypatch):
    """repeats=1 needs Repeat off: frames that keep coming after the 16 are refused even
    when the camera has no pending() count (the real camera has none)."""
    p = _use(scene)
    n0 = _n_scans(scene)
    _force_scene_camera(monkeypatch, 5, pending=0)
    with pytest.raises(pm.MissingFrames, match="extra frame"):
        api.scan_trigger(str(pack8), None, "t8rep", p, repeats=1, acknowledge=pm.ACK_PHRASE)
    assert _n_scans(scene) == n0                     # nothing saved


def test_scan_trigger_8bit_refuses_an_unprovable_order(scene, pack8, monkeypatch):  # noqa: F811
    """Frames whose white / black are not where the pack puts them (here the GUI shows
    every pattern inverted) are refused, never decoded into a metric scan."""
    p = _use(scene)
    n0 = _n_scans(scene)
    _force_scene_camera(monkeypatch, 3, invert=True)
    with pytest.raises(ss.SequenceError, match="white brightest"):
        api.scan_trigger(str(pack8), None, "t8inv", p, repeats=2, acknowledge=pm.ACK_PHRASE)
    assert _n_scans(scene) == n0


ROLES16 = ["white", "black"] + ["gray"] * 10 + ["fringe"] * 4


def _patterns16(H=60, W=96):
    """White, black, 10 binary code planes, 4 phase-shifted fringes (0..1)."""
    x = np.arange(W)[None, :].repeat(H, 0)
    pats = [np.ones((H, W)), np.zeros((H, W))]
    pats += [((x >> b) & 1).astype(float) for b in (6, 5, 4, 3, 2, 1, 0, 6, 5, 4)]
    pats += [0.5 + 0.5 * np.cos(2 * np.pi * x / 24 + k * np.pi / 2) for k in range(4)]
    return pats


def _shoot(pats, start, n_frames, *, seed=0, shift_after=None):
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n_frames):
        p = pats[(start + k) % len(pats)]
        if shift_after is not None and k >= shift_after:
            p = np.roll(p, 5, axis=1)                # the part moved between the loops
        out.append((12 + 170 * p + rng.normal(0, 1.5, p.shape)).astype(np.float32))
    return out


@pytest.mark.parametrize("repeats", [1, 3])
def test_align_sequence_finds_every_start(repeats):
    pats = _patterns16()
    for start in range(16):
        imgs, rep = hw.align_sequence(_shoot(pats, start, 16 * repeats, seed=start), ROLES16,
                                      repeats)
        assert rep["start_seq"] == start and rep["runner_up"] < hw.ALIGN_MIN_AGREE
        want = [12 + 170 * p for p in pats]
        assert max(float(np.abs(i - w).mean()) for i, w in zip(imgs, want)) < 2.0


@pytest.mark.parametrize("case,match", [
    ("inverted", "white brightest"),             # Invert Pattern in the GUI
    ("black_moved", "white brightest"),          # the GUI plays another order
    ("dark", "no projector light"),              # no light in the frames
    ("ambiguous", "starts of the sequence fit"),  # too little of the field seen
    ("moved", "loops do not show the same"),     # the scene moved between loops
    ("short", "frames for"),                     # not whole loops
    ("dup_white", "look like the white or the black"),   # a white where a fringe belongs
    ("dup_black", "look like the white or the black"),
])
def test_align_sequence_refuses_what_it_cannot_prove(case, match):
    pats, n_frames, kw = _patterns16(), 32, {}
    if case == "inverted":
        pats = [1 - p for p in pats]
    elif case == "black_moved":
        pats[1], pats[3] = pats[3], pats[1]
    elif case == "dark":
        pats = [np.zeros_like(p) for p in pats]
    elif case == "ambiguous":
        pats[2], pats[3] = np.ones_like(pats[2]), np.zeros_like(pats[3])
    elif case == "moved":
        kw["shift_after"] = 16
    elif case == "short":
        n_frames = 31
    elif case == "dup_white":                    # the GUI loop repeats white at its end
        pats[-1] = np.ones_like(pats[-1])
    elif case == "dup_black":
        pats[8] = np.zeros_like(pats[8])
    with pytest.raises(ss.SequenceError, match=match):
        hw.align_sequence(_shoot(pats, 7, n_frames, **kw), ROLES16, 2)


# timing: the documented 8-bit minimum period works with the default exposure

@pytest.mark.parametrize("P", [8563, 8600, 8800, 8833, 10000])
def test_8bit_default_exposure_reaches_the_documented_period(P):
    t = mfp.triggered_timing(8, P)
    X, E = t["pattern_exposure_us"], t["camera"]["exposure_us"]
    assert X >= mfp.MIN_EXPOSURE_US[8] and P - X >= mfp.PERIOD_DIFFERENCE_MIN_US
    assert X <= E <= P - mfp.CAMERA_TRIGGER_GUARD_US
    assert X == max(8333, P - 500)


def test_8bit_too_short_period_names_the_limit(tmp_path):
    for P in (8562, 5000):
        with pytest.raises(mfp.TimingError, match="period >= 8563 us"):
            mfp.triggered_timing(8, P)
    with pytest.raises(mfp.TimingError, match="8563"):
        mfp.triggered_timing(8, 9000, pattern_exposure_us=8000)      # explicit, too short
    d = mfp.make_pack(mfp.DEFAULT_PRESET, out_root=tmp_path, period_us=8600)
    u = json.loads((d / "manifest.json").read_text(encoding="utf-8"))["timing"]["uniform"]
    assert (u["period_us"], u["exposure_us"], u["camera_exposure_us"]) == (8600, 8333, 8433)


# camera exposure: an explicit 3000 us is kept (it is not the "not given" sentinel)

@pytest.mark.parametrize("given,want", [(None, 2500), (3000, 3000), (2999, 2999)])
def test_fastscan_triggered_camera_exposure_as_given(tmp_path, given, want):
    d = mfp.make_fastscan_pack(out_root=tmp_path, label="cx", trigger_mode="trigout",
                               period_us=4500, pattern_exposure_us=3500,
                               camera_exposure_us=given)
    m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    assert m["timing"]["hardware_trigger"]["camera"]["exposure_us"] == want
    assert m["timing"]["camera"]["exposure_us"] == want


def test_tools_cli_camera_exposure_3000_is_kept(tmp_path, capsys):
    assert mfp.main(["--fastscan", "--trigger-mode", "trigout", "--period-us", "4500",
                     "--pattern-exposure-us", "3500", "--camera-exposure-us", "3000",
                     "--out", str(tmp_path), "--label", "c3000"]) == 0
    d = next(tmp_path.glob("*c3000*"))
    m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    assert m["timing"]["hardware_trigger"]["camera"]["exposure_us"] == 3000
    capsys.readouterr()
    # free run keeps its own default when nothing is given
    d = mfp.make_fastscan_pack(out_root=tmp_path, label="fr")
    m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    assert m["timing"]["camera"]["exposure_us"] == mfp.FASTSCAN_CAMERA_EXPOSURE_US


def _ui_fastscan(tmp_path, capsys, extra):
    try:
        code = cli.main(["--mock", "--library", str(tmp_path / "lib"), "--json", "fastscan-pack",
                         "--trigger-mode", "trigout", "--period-us", "4500"] + extra)
        out = capsys.readouterr().out
    finally:
        api._session = None
    return code, out


@pytest.mark.parametrize("extra,want", [
    (["--pattern-exposure-us", "3500"], 2500),             # not given: min(3000, X - 1000)
    (["--pattern-exposure-us", "3000"], 2000),             # ... and buildable at X = 3000
    (["--pattern-exposure-us", "3500", "--camera-exposure", "3000"], 3000)])   # given: kept
def test_ui_cli_fastscan_camera_exposure_default_and_given(tmp_path, capsys, extra, want):
    """scan fastscan-pack (engine/api.py) used to turn a camera exposure NOT given into an
    explicit 3000 us: X = 3500 built at 3000 (not 2500) and X = 3000 was refused."""
    code, out = _ui_fastscan(tmp_path, capsys, extra + ["--label", f"u{want}"])
    assert code == 0, out
    m = json.loads((Path(json.loads(out)["folder"]) / "manifest.json").read_text(
        encoding="utf-8"))
    assert m["timing"]["hardware_trigger"]["camera"]["exposure_us"] == want


@pytest.mark.parametrize("command", [["trigger-test"], ["scan-trigger", "x", "--pack", "p"]])
def test_cli_trigger_line_choices_are_camera_inputs_only(command, capsys):
    """--help used to offer Line2 (GPIO pin 3, the 3.3 V auxiliary output: never wire the
    trigger to it); the parser now offers exactly hwtrigger.TRIGGER_LINES."""
    assert cli.main(["--mock", *command[:1], "--help"]) == 0
    usage = capsys.readouterr().out
    assert "{Line0,Line3}" in usage and "{Line0,Line2" not in usage, usage
    assert cli.main(["--mock", *command, "--line", "Line2"]) == 2
    assert "invalid choice" in capsys.readouterr().err
    assert set(hw.TRIGGER_LINES) == {"Line0", "Line3"}


def test_8bit_pack_camera_exposure_given_is_used(tmp_path):
    d = mfp.make_pack(mfp.DEFAULT_PRESET, out_root=tmp_path, period_us=10000,
                      camera_exposure_us=9700)
    t = json.loads((d / "manifest.json").read_text(encoding="utf-8"))["timing"]
    assert t["uniform"]["camera_exposure_us"] == t["hardware_trigger"]["camera"][
        "exposure_us"] == 9700
    with pytest.raises(ValueError, match="video port"):
        mfp.make_pack(mfp.DEFAULT_PRESET, out_root=tmp_path, video_port=True,
                      camera_exposure_us=9700)


def test_softsync_doc_resort_names_the_recording_it_makes():
    """docs/FLASH_SOFTSYNC.md: the sort-flash example re-sorts the recording that the
    block's own scan-flash line writes (<ts>_<label>.npz)."""
    doc = (ROOT / "docs" / "FLASH_SOFTSYNC.md").read_text(encoding="utf-8")
    labels = set(re.findall(r"scan scan-flash (\w+) --pack", doc))
    resorted = set(re.findall(r"flashscans\\<ts>_(\w+)\.npz", doc))
    assert labels and resorted and resorted <= labels

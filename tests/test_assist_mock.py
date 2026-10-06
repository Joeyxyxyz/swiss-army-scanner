"""Scan assistant (tune + measure_gamma), end to end in mock mode."""

import json

import numpy as np
import pytest

from engine import api, config
from engine.capture import assist
from engine.capture.assist import DEFAULT_EXPOSURES, GammaReport, TuneReport
from engine.capture.projector import MockWindow

from conftest import PRESET


@pytest.fixture
def fast(eng):
    p = api.load_preset(PRESET)
    p.camera.average = 1          # speed; the decisions don't need averaging in mock
    return p


def _step(report, exposure_us, duty=None):
    return next(s for s in report.steps if s.exposure_us == exposure_us and s.duty == duty)


def test_mock_camera_exposure_scales_and_clips(eng, fast):
    """Longer exposure = proportionally more counts, clipped at 255 like the bench."""
    white = np.zeros((1140, 912), np.uint8)        # inverted chain: drive 0 = full light
    with eng.rig(fast) as sync:
        a = sync.shoot(white)
        sync.camera.set_exposure_us(fast.camera.exposure_us / 2)
        b = sync.shoot(white)
        sync.camera.set_exposure_us(fast.camera.exposure_us * 4)
        c = sync.shoot(white)
        with pytest.raises(ValueError):
            sync.camera.set_exposure_us(0)
    centre = (slice(500, 580), slice(680, 760))
    assert np.median(b[centre]) == pytest.approx(np.median(a[centre]) / 2, rel=0.03)
    assert np.median(c[centre]) == 255.0
    assert fast.camera.exposure_us == 8000         # the preset is never touched


def test_tune_recommends_unsaturated_exposure(fast):
    before = config.preset_to_dict(fast)
    file_before = fast.source.read_bytes()
    r = api.tune(fast)
    assert isinstance(r, TuneReport)

    # one step per exposure, the preset's own always included
    want = sorted(set(DEFAULT_EXPOSURES) | {fast.camera.exposure_us})
    assert [s.exposure_us for s in r.steps] == want
    assert all(s.duty is None for s in r.steps)

    # the mock saturates at long exposures; the recommendation must not
    rec = r.recommended["camera"]["exposure_us"]
    assert set(r.recommended) == {"camera"}
    chosen = _step(r, rec)
    assert chosen.saturated_pct < 0.5
    assert all(s.saturated_pct > 50 for s in r.steps if s.exposure_us >= 16000)
    assert rec == 8000                             # longest unclipped = best modulation
    assert chosen.gc_conf == max(s.gc_conf for s in r.steps if s.saturated_pct < 0.5)
    assert chosen.coded_axis == "x" and chosen.axis_ok is True
    assert chosen.quality in ("usable", "good", "excellent") and chosen.polarity_inverted

    assert r.ok and r.advice
    assert any("saturat" in a.lower() for a in r.advice)
    assert any("polarity" in a.lower() for a in r.advice)
    assert r.units["exposure_us"] == "us" and "%" in r.units["saturated_pct"]

    # the default sweep is built on whole 60 Hz video frames; here every one
    # clips, so the sub-frame 8000 us wins - and the advice says why that is
    # second best and how to get a whole-frame exposure
    assert {16667, 33333, 50000} <= set(DEFAULT_EXPOSURES)
    assert [s.exposure_us for s in r.steps if s.frame_locked] == [16667, 33333, 50000]
    assert chosen.frame_locked is False
    frame_adv = [a for a in r.advice if "whole number of video frames" in a]
    assert len(frame_adv) == 1 and "16667 us (whole frames) clips" in frame_adv[0]
    assert "camera.average" in frame_adv[0]

    # never edits the preset: neither the loaded object nor the file
    assert config.preset_to_dict(fast) == before
    assert fast.source.read_bytes() == file_before

    # report in the output bin
    outs = api.list_library("outputs", "tuning")
    assert len(outs) == 1 and outs[0]["geometry_id"] == fast.geometry_id
    assert r.path.exists() and outs[0]["file"].endswith(r.path.name)
    doc = json.loads(r.path.read_text(encoding="utf-8"))
    assert doc["recommended"] == r.recommended and doc["ok"] is True
    assert len(doc["steps"]) == len(r.steps) and doc["advice"] == r.advice
    assert doc["units"]["gc_conf"].startswith("camera counts")
    assert doc["criteria"]["video_frame_us"] == pytest.approx(1e6 / 60, abs=0.01)
    assert [s["frame_locked"] for s in doc["steps"]] == [s.frame_locked for s in r.steps]


def test_tune_flags_rotate90_mistake(eng):
    """cell_wall_2p7ft has rotate90 on: fringes run along the baseline."""
    p = api.load_preset("cell_wall_2p7ft")
    p.camera.average = 1
    r = api.tune(p, exposures=[14000])
    assert [s.exposure_us for s in r.steps] == [14000, 28000]
    assert all(s.coded_axis == "y" and s.axis_ok is False for s in r.steps)
    assert not r.ok
    assert any("rotate90" in a for a in r.advice)
    assert p.projection.rotate90 is True           # advice only, nothing changed


def test_tune_duty_sweep_uses_a_private_copy(fast):
    before = config.preset_to_dict(fast)
    r = api.tune(fast, exposures=[4000], duties=[0.3, 1.0])
    assert sorted((s.exposure_us, s.duty) for s in r.steps) == \
        [(4000, 0.3), (4000, 0.55), (4000, 1.0), (8000, 0.3), (8000, 0.55), (8000, 1.0)]
    # more duty = more fringe contrast at the same exposure; the inverted
    # chain's full-light level (drive 0) is duty-independent, so no new clipping
    assert _step(r, 8000, 0.3).gc_conf < _step(r, 8000, 0.55).gc_conf < _step(r, 8000, 1.0).gc_conf
    assert r.recommended == {"camera": {"exposure_us": 8000}, "projection": {"duty": 1.0}}
    assert any("projection.duty" in a for a in r.advice)
    assert config.preset_to_dict(fast) == before
    assert fast.projection.duty == 0.55


def test_frame_lock_helpers():
    """Exposures must span whole HDMI video frames (60 Hz = 16667 us) or each
    free-run frame integrates a random slice of the R/G/B + bit-plane sequence."""
    assert assist.frame_exposures() == (16667, 33333, 50000)
    assert assist.frame_locked(16667) and assist.frame_locked(33333) and assist.frame_locked(50000)
    assert assist.frame_locked(16683)                  # 59.94 Hz source: within tolerance
    for e in (2000, 8000, 8333, 16000, 28000):         # the old sweep / presets
        assert not assist.frame_locked(e), e
    assert assist.snap_exposure_us(8000) == 16667      # never below one frame
    assert assist.snap_exposure_us(28000) == 33333
    assert assist.frame_locked(20000, frame_us=20000.0)   # 50 Hz source


def test_tune_prefers_whole_frame_exposure(fast):
    """A dim scene where 16667 and 28000 us are both clean: the whole-frame
    exposure wins even though 28000 us has more modulation."""
    api.mock_bench().gain = 90.0
    r = api.tune(fast, exposures=[16667, 28000])
    assert [(s.exposure_us, s.frame_locked) for s in r.steps] == \
        [(8000, False), (16667, True), (28000, False)]
    assert all(s.saturated_pct < 0.5 for s in r.steps)
    assert _step(r, 28000).gc_conf > _step(r, 16667).gc_conf
    assert r.recommended == {"camera": {"exposure_us": 16667}} and r.ok
    assert any(a.startswith("Chose 16667 us (whole video frames) over 28000 us") for a in r.advice)
    assert any("8000 -> 16667" in a for a in r.advice)


def test_tune_frame_period_is_configurable(eng, fast):
    """A 50 Hz source: the default sweep and the lock test follow frame_us."""
    r = assist.tune(eng, fast, frame_us=20000.0)
    locked = [s.exposure_us for s in r.steps if s.frame_locked]
    assert locked == [20000, 40000, 60000] and r.video_frame_us == 20000.0
    assert not _step(r, 8000).frame_locked
    with pytest.raises(ValueError, match="frame_us"):
        assist.tune(eng, fast, frame_us=0)


def test_tune_all_saturated_or_dark_is_not_ok(fast):
    bench = api.mock_bench()
    bench.gain = 3000.0                            # a glaring scene: clips at every step
    r = api.tune(fast, exposures=[2000])
    assert all(s.saturated_pct > 50 for s in r.steps)
    assert not r.ok and r.recommended == {}
    assert any("shorter exposures" in a for a in r.advice)
    # cutting the light so a whole frame fits comes first; shorter is second best
    assert any("whole-video-frame exposure (16667 us)" in a and "ND filter" in a for a in r.advice)

    bench.gain = 0.0                               # projector off / aimed away
    r = api.tune(fast, exposures=[2000])
    assert not r.ok and r.recommended == {}
    assert any("No lit pixels" in a for a in r.advice)
    assert len(api.list_library("outputs", "tuning")) == 2    # failures are reported too


def test_tune_rejects_bad_sweeps(fast):
    with pytest.raises(ValueError, match="exposures"):
        api.tune(fast, exposures=[-5])
    with pytest.raises(ValueError, match="duties"):
        api.tune(fast, exposures=[4000], duties=[1.5])
    assert api.list_library("outputs", "tuning") == []


def test_measure_gamma_linear_mock(fast):
    g = api.measure_gamma(fast)
    assert isinstance(g, GammaReport)
    assert abs(g.gamma - 1.0) < 0.15 and g.r2 > 0.98
    assert g.residual_gamma == g.gamma                 # no precomp: same fit
    assert g.recommended["projection"]["gamma"] == pytest.approx(1.0, abs=0.15)
    assert len(g.levels) == len(g.response) == len(g.drive_levels) == len(g.sent_levels) == 12
    assert g.levels == sorted(g.levels) and g.response == sorted(g.response)
    # levels are the values actually sent: the inverted duty-0.55 chain never
    # sends below 255 - int(0.55*255) = 115, and the dark reference is the
    # mirrors-off frame, so that level reads ~115/255 - not 0 (the old fit
    # subtracted the 115 frame as "dark" and fitted a compressed curve)
    assert g.hdmi_inverted is False and g.sent_levels == g.levels
    assert g.levels[0] == 115 and g.levels[-1] == 255
    assert g.response[-1] == pytest.approx(1.0)
    assert g.response[0] == pytest.approx(115 / 255, abs=0.03)
    # polarity is measured: with projection.invert, drive 0 is the bright end
    assert g.polarity_inverted is True
    assert g.drive_levels[-1] == 0 and g.drive_levels[0] == 255
    assert "dimensionless" in g.units["gamma"]
    assert fast.projection.gamma == 1.0
    assert any("whole number of video frames" in w for w in g.warnings)   # 8000 us preset

    outs = api.list_library("outputs", "tuning")
    assert len(outs) == 1 and g.path.exists()
    doc = json.loads(g.path.read_text(encoding="utf-8"))
    assert doc["gamma"] == g.gamma and doc["chain"] == {"gamma": 1.0, "duty": 0.55, "invert": True}
    assert "WHOLE path" in doc["note"] and "mirrors-off" in doc["fit"]["dark"]
    assert doc["sent_levels"] == g.sent_levels and doc["chain_levels"] == g.sent_levels
    assert doc["camera"]["frame_locked"] is False


def _nonlinear_projector(bench, gamma):
    """Model HDMI video-mode degamma on the mock: light = (drive/255)^gamma."""
    render = bench.render

    def degamma(pat):
        if pat is None:
            return render(None)
        return render(255.0 * (np.asarray(pat, np.float32) / 255.0) ** gamma)
    bench.render = degamma


def test_measure_gamma_recovers_projector_gamma_and_residual(fast):
    _nonlinear_projector(api.mock_bench(), 2.2)
    fast.projection.invert = False                 # intrinsic curve: full-off to full-on
    fast.projection.duty = 1.0
    g = api.measure_gamma(fast)
    assert g.gamma == pytest.approx(2.2, abs=0.1) and g.r2 > 0.98
    assert g.polarity_inverted is False
    assert g.recommended == {"projection": {"gamma": pytest.approx(2.2, abs=0.1)}}

    # with the precomp applied, the path is linear and the recommendation holds.
    # `gamma` stays the projector's own curve (vs the value sent); the chain's
    # leftover nonlinearity is `residual_gamma`
    fast.projection.gamma = g.recommended["projection"]["gamma"]
    g2 = api.measure_gamma(fast)
    assert g2.residual_gamma == pytest.approx(1.0, abs=0.1)
    assert g2.gamma == pytest.approx(2.2, abs=0.1)
    assert g2.recommended["projection"]["gamma"] == pytest.approx(2.2, abs=0.1)
    assert any("residual" in w for w in g2.warnings)
    assert len(api.list_library("outputs", "tuning")) == 2


def test_measure_gamma_on_inverted_bench_chain_converges(fast):
    """Regression: on the bench chain (invert + duty 0.55) the old fit used the
    L=255 frame (still sent at 115) as dark and x = (255-L)/255, returning
    ~1.26 for a true 2.2 and creeping 1.26 -> 1.46 -> 1.61 on re-runs. Fitting
    against the value actually sent, above the mirrors-off frame, gets 2.2 in
    one step, and applying it linearises the inverted chain."""
    _nonlinear_projector(api.mock_bench(), 2.2)
    assert fast.projection.invert and fast.projection.duty == 0.55   # the bench preset
    g = api.measure_gamma(fast)
    assert g.polarity_inverted is True and g.hdmi_inverted is False
    assert g.gamma == pytest.approx(2.2, abs=0.1) and g.r2 > 0.98
    assert g.recommended == {"projection": {"gamma": pytest.approx(2.2, abs=0.1)}}
    assert g.levels[0] == 115 and g.response[0] == pytest.approx((115 / 255) ** 2.2, abs=0.03)
    assert not any("BEFORE inversion" in w for w in g.warnings)
    assert any("LAST" in w for w in g.warnings)       # the precomp works with invert on

    fast.projection.gamma = g.recommended["projection"]["gamma"]
    g2 = api.measure_gamma(fast)
    assert g2.residual_gamma == pytest.approx(1.0, abs=0.05)      # linearised
    assert g2.recommended["projection"]["gamma"] == \
        pytest.approx(g.recommended["projection"]["gamma"], abs=0.05)   # converged, no creep
    doc = json.loads(g2.path.read_text(encoding="utf-8"))
    assert doc["chain_levels"][0] == 115 and doc["sent_levels"][0] > 115   # precomp lifts it
    assert doc["residual_fit"]["model"].startswith("log(response) = residual_gamma")


def test_measure_gamma_refuses_precomp_when_hdmi_inverts(fast, monkeypatch):
    """A path that inverts in hardware: polarity of the raw pair is measured,
    and since the precomp acts on the value sent, none is recommended."""
    init = MockWindow.__init__

    def inverting(self, *a, **k):
        init(self, *a, **k)
        self.hdmi_inverts = True
    monkeypatch.setattr(MockWindow, "__init__", inverting)
    g = api.measure_gamma(fast)
    assert g.hdmi_inverted is True and g.recommended == {}
    assert g.polarity_inverted is False                # invert + hardware invert cancel
    assert any("HDMI path itself inverts" in w for w in g.warnings)
    assert g.levels == sorted(g.levels) and g.levels[-1] == 140    # 255 - sent


def test_measure_gamma_rejects_bad_levels(fast):
    with pytest.raises(ValueError, match="n_levels"):
        api.measure_gamma(fast, n_levels=2)

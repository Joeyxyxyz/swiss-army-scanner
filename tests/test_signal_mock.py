"""check_signal's probe (engine/capture/signal.py) in mock mode: fringes
along the preset's basis axis with the scan's steps / period, the coded
axis judged against the rig's axis, informational polarity, saturation."""

import dataclasses

import numpy as np
import pytest

from engine import api
from engine.capture import signal
from engine.capture.projector import MockWindow
from engine.results import SignalReport

from conftest import PRESET


@pytest.fixture
def fast(eng):
    p = api.load_preset(PRESET)
    p.camera.average = 1
    return p


class Tap:
    """Records every pattern the probe projects (through the real mock rig)."""

    def __init__(self, sync):
        self.sync, self.patterns = sync, []

    def shoot(self, pattern, raw=False):
        self.patterns.append(np.asarray(pattern))
        return self.sync.shoot(pattern, raw=raw)


def _probe(eng, preset):
    with eng.rig(preset) as sync:
        tap = Tap(sync)
        return signal.probe(tap, preset), tap.patterns


def _set_axis(preset, axis):
    for b in preset.bases:
        b["axis"] = axis


def test_probe_uses_the_scan_fringes(eng, fast):
    """steps / period come from the hybrid the scan decodes; stripes run
    along the preset's basis axis ('x' = vertical stripes)."""
    for b in fast.bases:
        if b["name"] == "phaseshift":
            b.update(steps=6, period_px=32)
    rep, pats = _probe(eng, fast)
    assert len(pats) == 2 + 6                               # white, black, 6 fringes
    f = pats[2].astype(float)
    assert np.ptp(f[0, :64]) > 100                          # varies along a row (x)...
    assert np.ptp(f[:32, 100]) == 0                         # ...constant down a column
    from engine import _seed
    want = _seed.load("patterns").phase_shift_fringes(n_steps=6, period_px=32,
                                                      orientation="vertical")
    assert all(np.array_equal(a, b) for a, b in zip(pats[2:], want))
    assert rep.coded_axis == "x" and rep.axis_ok is True and rep.ok


def test_probe_flags_basis_axis_across_the_baseline(eng, fast):
    """Horizontal stripes (basis axis 'y') with the projector RIGHT of the
    camera: the camera sees the code along y but depth moves along x."""
    _set_axis(fast, "y")
    rep, pats = _probe(eng, fast)
    f = pats[2].astype(float)
    assert np.ptp(f[:64, 100]) > 100 and np.ptp(f[100, :64]) == 0   # horizontal stripes
    assert rep.coded_axis == "y" and rep.axis_ok is False and not rep.ok
    w = [x for x in rep.warnings if x.startswith("coded axis")]
    assert len(w) == 1
    assert "camera Y" in w[0] and "rig axis x" in w[0] and "basis axis 'y'" in w[0]
    assert "rotate90" in w[0] and "basis axis 'x'" in w[0] and "new geometry_id" in w[0]

    # the same horizontal code is RIGHT for a projector above the camera
    fast.rig.projector_side = "above"
    rep, _ = _probe(eng, fast)
    assert rep.coded_axis == "y" and rep.axis_ok is True
    assert not any(x.startswith("coded axis") for x in rep.warnings)


def test_rotate90_mistake_names_both_axes(eng):
    p = api.load_preset("cell_wall_2p7ft")                 # rotate90 on, basis axis x
    p.camera.average = 1
    rep, _ = _probe(eng, p)
    assert rep.coded_axis == "y" and rep.axis_ok is False
    w = next(x for x in rep.warnings if x.startswith("coded axis"))
    assert "basis axis 'x'" in w and "rotated by projection.rotate90" in w and "rig axis x" in w


def test_polarity_note_is_informational_when_invert_explains_it(eng, fast):
    assert fast.projection.invert                            # the bench preset
    rep, _ = _probe(eng, fast)
    pol = [w for w in rep.warnings if "polarity" in w]
    assert len(pol) == 1 and "expected: projection.invert is on" in pol[0]
    assert "no action needed" in pol[0] and rep.ok

    fast.projection.invert = False                           # normal path: nothing to say
    rep, _ = _probe(eng, fast)
    assert not any("polarity" in w for w in rep.warnings)


def test_polarity_flip_without_invert_still_warns(eng, fast, monkeypatch):
    init = MockWindow.__init__

    def inverting(self, *a, **k):
        init(self, *a, **k)
        self.hdmi_inverts = True
    monkeypatch.setattr(MockWindow, "__init__", inverting)
    fast.projection.invert = False
    rep, _ = _probe(eng, fast)
    pol = [w for w in rep.warnings if "polarity" in w]
    assert len(pol) == 1 and "projection.invert off" in pol[0]
    fast.projection.invert = True                            # both invert: they cancel
    rep, _ = _probe(eng, fast)
    pol = [w for w in rep.warnings if "polarity" in w]
    assert len(pol) == 1 and "cancel" in pol[0]


def test_saturation_is_reported(eng, fast):
    has_field = "saturated_pct" in {f.name for f in dataclasses.fields(SignalReport)}
    rep, _ = _probe(eng, fast)
    assert not any("saturated_pct" in w for w in rep.warnings)
    if has_field:
        assert rep.saturated_pct <= signal.SAT_WARN_PCT
    api.mock_bench().gain = 3000.0                           # glare: clips everywhere
    rep, _ = _probe(eng, fast)
    w = [x for x in rep.warnings if x.startswith("saturated_pct")]
    assert len(w) == 1 and "run tune" in w[0]
    pct = float(w[0].split()[1].rstrip(":"))
    assert pct > 50
    if has_field:
        assert rep.saturated_pct == pytest.approx(pct, abs=0.01)
    assert signal.SAT_LEVEL == 250.0


def test_check_signal_verb_still_works(fast):
    rep = api.check_signal(fast)
    assert rep.ok and rep.coded_axis == "x" and rep.axis_ok is True


def test_probe_passes_on_rig_setup_problems(eng, fast):
    """A DMD window at the wrong rect (HDMIWindow.rect_ok False) and camera
    nodes the post-open reset could not restore reach check_signal's report."""
    with eng.rig(fast) as sync:
        rep = signal.probe(sync, fast)
        assert not any("DMD window" in w or "camera reset" in w for w in rep.warnings)
        sync.projector.window.rect_ok = False
        sync.projector.window.rect_warning = "DMD window is 1920x1080 at (0, 0), expected ..."
        sync.camera.reset_state = {"reset": [], "failed": {"GammaEnable": "locked"}}
        rep = signal.probe(sync, fast)
    assert any(w.startswith("DMD window is 1920x1080") for w in rep.warnings)
    assert any("camera reset" in w and "GammaEnable" in w for w in rep.warnings)


def test_no_polarity_note_when_nothing_is_lit(eng, fast):
    api.mock_bench().gain = 0.0                              # projector off / aimed away
    rep, _ = _probe(eng, fast)
    assert not any("polarity" in w for w in rep.warnings)
    assert not rep.ok and rep.valid_pct < 1.0

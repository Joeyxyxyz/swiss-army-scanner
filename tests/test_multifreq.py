"""multifreq basis: N steps x 3 periods, hierarchical temporal unwrap,
modulation mask, diamond-parity fringes, registry / preset validation."""

import json

import numpy as np
import pytest

from engine import config
from engine.bases import registry
from engine.bases.multifreq import MultiFreqBasis
from conftest import FIXTURE_PRESETS


def _ideal_frames(basis, u, v, A=100.0, B=80.0, flip=False):
    """Camera frames of a camera that sees DMD point (u, v) per pixel exactly
    (straight fringes in native coordinates: what the parity-aware pattern gives)."""
    out = [np.full(u.shape, A + B), np.full(u.shape, A - B)]
    for ax in basis._coded:
        pos = u if ax == "x" else v
        for T in basis.periods:
            for k in range(basis.steps):
                out.append(A + B * np.cos(2 * np.pi * pos / T + 2 * np.pi * k / basis.steps))
    if flip:
        out = [2 * A - f for f in out]
    return out


def test_frame_count_and_patterns():
    b = MultiFreqBasis(axis="xy")
    assert b.steps == 12 and b.periods == [8.0, 80.0, 1280.0]
    pats = b.generate()
    assert len(pats) == b.n_frames() == 2 + 2 * 3 * 12
    assert all(p.shape == (1140, 912) and p.dtype == np.uint8 for p in pats)
    assert pats[0].min() == 255 and pats[1].max() == 0


def test_parity_shifts_alternate_rows_by_half_a_column():
    b = MultiFreqBasis(steps=4, periods_px=(8, 80, 1280), axis="x", diamond_parity=1)
    fine = b.generate()[2].astype(float)            # first x fringe, T = 8
    even, odd = fine[0], fine[1]
    # odd rows show the value half a column further along: cos(2 pi (c + 0.5) / 8)
    c = np.arange(912)
    assert np.allclose(odd, np.clip(np.rint(127.5 * (1 + np.cos(2 * np.pi * (c + 0.5) / 8))), 0, 255))
    assert not np.allclose(even, odd)
    sq = MultiFreqBasis(steps=4, axis="x").generate()[2]
    assert np.array_equal(sq[0], sq[1])             # no parity: identical rows


@pytest.mark.parametrize("flip", [False, True])
def test_decode_recovers_both_axes_exactly(flip):
    b = MultiFreqBasis(steps=6, axis="xy")
    rng = np.random.default_rng(0)
    u = rng.uniform(0, 911, (40, 50))
    v = rng.uniform(0, 1139, (40, 50))
    fld = b.decode(_ideal_frames(b, u, v, flip=flip), "g", contrast_min=5, conf_min=5)
    assert fld.mask.all()
    assert np.abs(fld.corr_x - u).max() < 2e-3
    assert np.abs(fld.corr_y - v).max() < 2e-3
    assert fld.meta["polarity_flipped"] is flip
    assert fld.meta["per_axis"]["x"]["offset_median"] == pytest.approx(100.0, abs=1e-6)


def test_modulation_mask_and_unwrap_rejection():
    b = MultiFreqBasis(steps=6, axis="x")
    u = np.tile(np.linspace(1, 900, 64), (8, 1))
    frames = _ideal_frames(b, u, u)
    frames = [f.copy() for f in frames]
    low = (slice(0, 2), slice(None))                # rows with almost no fringe modulation
    for f in frames[2:]:
        f[low] = 100.0 + 0.01 * (f[low] - 100.0)
    bad = (slice(4, 5), slice(10, 20))              # a coarse level 290 px off (not k x 80)
    n = b.steps
    for k in range(n):
        f = frames[2 + 2 * n + k]
        f[bad] = 100 + 80 * np.cos(2 * np.pi * (u[bad] + 290) / 1280 + 2 * np.pi * k / n)
    fld = b.decode(frames, "g", contrast_min=5, conf_min=5)
    assert not fld.mask[low].any()
    assert not fld.mask[bad].any()
    assert fld.mask[6:].all()
    assert fld.meta["per_axis"]["x"]["unwrap_rejected_pct"] > 0


@pytest.mark.parametrize("kw,msg", [
    ({"steps": 2}, "steps"),
    ({"periods_px": (8, 300, 1280)}, "ratio"),
    ({"periods_px": (80, 8, 1280)}, "increasing"),
    ({"periods_px": (8, 80, 640)}, "cover"),
    ({"axis": "z"}, "axis"),
    ({"diamond_parity": 2}, "parity"),
])
def test_bad_parameters_refused(kw, msg):
    with pytest.raises(ValueError, match=msg):
        MultiFreqBasis(**kw)


def test_registry_and_preset_validation(tmp_path):
    registry.validate_spec({"name": "multifreq", "axis": "xy", "steps": 12,
                            "periods_px": [8, 80, 1280], "diamond_parity": 1})
    assert registry.is_implemented("multifreq")
    assert isinstance(registry.get({"name": "multifreq", "axis": "x"}), MultiFreqBasis)
    with pytest.raises(ValueError, match="axis"):
        registry.validate_spec({"name": "hybrid", "axis": "xy"})
    with pytest.raises(ValueError, match="periods_px"):
        registry.validate_spec({"name": "multifreq", "periods_px": [8]})
    with pytest.raises(ValueError, match="unknown params"):
        registry.validate_spec({"name": "multifreq", "period": 8})
    d = json.loads((FIXTURE_PRESETS / "cell_wall_2p7ft_vstripe.json").read_text())
    d["bases"] = [{"name": "multifreq", "steps": 12, "periods_px": [8, 80, 1280],
                   "axis": "x", "diamond_parity": 1}]
    p = config.preset_from_dict(d, mock=True)
    assert config.geometry_fingerprint(p)["basis_axis"] == "x"
    d["bases"][0]["periods_px"] = [8, 80, 400]
    with pytest.raises(config.ConfigError, match="cover"):
        config.preset_from_dict(d, mock=True)

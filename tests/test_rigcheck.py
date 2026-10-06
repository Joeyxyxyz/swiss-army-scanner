"""Rig check (HOME): the sub-pixel shift estimator (engine/capture/rigcheck.py, pure)
on known synthetic shifts, the OK / MOVED / UNKNOWN assessment, and `scan rig-check`
end to end in mock (reference -> OK -> a simulated nudge -> MOVED)."""

import contextlib
import io
import json

import numpy as np
import pytest
from scipy import ndimage

from engine import api
from engine.capture import rigcheck as R
from ui import cli


def _texture(shape=(480, 640), sigma=2.0, seed=0):
    rng = np.random.default_rng(seed)
    return ndimage.gaussian_filter(rng.random(shape), sigma) * 255


@pytest.mark.parametrize("dx, dy", [(0.0, 0.0), (0.3, -0.7), (2.25, 1.6), (-5.4, 3.1),
                                    (12.6, -8.2), (-0.1, 0.15)])
def test_shift_is_recovered_to_a_fifth_of_a_pixel(dx, dy):
    ref = _texture()
    rng = np.random.default_rng(1)
    img = ndimage.shift(ref, (dy, dx), order=3, mode="nearest") + rng.normal(0, 1.0, ref.shape)
    r = R.phase_shift(ref, img)
    assert r["dx"] == pytest.approx(dx, abs=0.2) and r["dy"] == pytest.approx(dy, abs=0.2), r
    assert r["peak"] > 0.5


def test_large_frames_are_downsampled_but_stay_in_input_px():
    ref = ndimage.zoom(_texture((500, 600), 2.0, seed=3), 2.4, order=1)    # 1200 x 1440
    img = ndimage.shift(ref, (1.3, -0.4), order=3, mode="nearest")
    r = R.phase_shift(ref, img, max_side=720)
    assert r["dx"] == pytest.approx(-0.4, abs=0.2) and r["dy"] == pytest.approx(1.3, abs=0.2)


def test_shading_alone_is_not_texture():
    """A smooth vignetted flood with independent noise: no usable match (never a false OK)."""
    rng = np.random.default_rng(2)
    flat = np.outer(np.hanning(480), np.hanning(640)) * 200
    r = R.phase_shift(flat + rng.normal(0, 1, flat.shape), flat + rng.normal(0, 1, flat.shape))
    assert r["peak"] < R.MIN_PEAK
    v = R.assess(r, r)
    assert v["status"] == "UNKNOWN" and v["shift_px"] is None


def test_size_mismatch_refused():
    with pytest.raises(ValueError, match="differ"):
        R.phase_shift(np.zeros((10, 10)), np.zeros((10, 12)))


def _s(dx, dy, peak=0.9):
    return {"dx": dx, "dy": dy, "shift_px": float(np.hypot(dx, dy)), "peak": peak}


def test_assess_ok_moved_and_who_moved():
    ok = R.assess(_s(0.1, 0.0), _s(0.2, 0.1), tol_px=0.5)
    assert ok["status"] == "OK" and ok["moved"] is None and ok["shift_mm"] is None
    cam = R.assess(_s(2.3, 0.0), _s(2.3, 0.1), tol_px=0.5, mm_per_px=0.3, distance_mm=640)
    assert cam["status"] == "MOVED" and cam["moved"] == "camera"
    assert cam["shift_mm"] == pytest.approx(0.69, abs=0.01)
    assert "2.3 px" in cam["message"] and "0.69 mm at 640 mm" in cam["message"]
    proj = R.assess(_s(0.05, 0.0), _s(1.5, 0.2), tol_px=0.5)
    assert proj["status"] == "MOVED" and proj["moved"] == "projector"
    blind = R.assess(_s(0, 0, peak=0.01), _s(3.0, 0.0), tol_px=0.5)    # flood featureless
    assert blind["status"] == "MOVED" and blind["moved"] == "camera or projector"


def test_random_block_pattern_is_seeded_and_not_periodic():
    a, b = R.random_block_pattern(), R.random_block_pattern()
    assert a.shape == (1140, 912) and a.dtype == np.uint8 and np.array_equal(a, b)
    assert 0.4 < (a > 0).mean() < 0.6
    r = R.phase_shift(a, np.roll(a, 16 * 3, axis=1))       # three blocks: one answer
    assert r["dx"] == pytest.approx(48, abs=0.2) and abs(r["dy"]) < 0.2


def test_normalised_pattern_removes_albedo():
    white = np.full((50, 60), 200.0)
    white[:, :30] = 50.0
    pat = np.where(np.indices((50, 60))[1] % 2 == 0, white, 0.0)
    n = R.normalised_pattern(pat, white, np.zeros_like(white))
    assert set(np.unique(np.round(n, 3))) <= {0.0, 1.0}


# ------------------------------------------------------------ scan rig-check (mock)

def _cli(argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = cli.main(argv)
    api._session = None
    return code, buf.getvalue()


def test_cli_rig_check_reference_ok_then_moved(tmp_path):
    lib = ["--mock", "--average", "1", "--library", str(tmp_path / "lib"), "--json"]
    code, out = _cli([*lib, "rig-check"])
    r = json.loads(out)
    assert code == cli.EXIT_OK and r["status"] == "REFERENCE_SET", r
    code, out = _cli([*lib, "rig-check"])
    r = json.loads(out)
    assert code == cli.EXIT_OK and r["status"] == "OK", r
    assert r["pattern"]["shift_px"] < 0.2
    code, out = _cli([*lib, "rig-check", "--mock-shift", "2.3,-0.4"])
    r = json.loads(out)
    assert code == cli.EXIT_NOT_OK and r["status"] == "MOVED", r
    # the mock frames have hard edges (cubic-shifted): looser than the synthetic tests
    assert r["shift_px"] == pytest.approx(np.hypot(2.3, 0.4), abs=0.4)
    assert r["flood"]["dx"] == pytest.approx(2.3, abs=0.4)
    assert r["shift_mm"] is None and any("px only" in w for w in r["warnings"])   # no stereo cal


def test_cli_mock_shift_needs_mock(tmp_path):
    code, _ = _cli(["--library", str(tmp_path / "lib"), "rig-check", "--mock-shift", "1,0"])
    assert code == cli.EXIT_REFUSED

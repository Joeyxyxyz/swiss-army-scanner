"""Rig check (HOME): the sub-pixel shift estimator (engine/capture/rigcheck.py, pure)
on known synthetic shifts, the OK / MOVED / UNKNOWN assessment, which device a MOVED names
(camera / projector or its picture / both / can't tell, at the tolerance edge, with a part
placed since the reference), and `scan rig-check` end to end in mock (reference -> OK -> a
simulated nudge -> MOVED)."""

import contextlib
import io
import json
import re

import numpy as np
import pytest
from scipy import ndimage

from engine import api, rigstate
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
    """A motion() result: a pure shift, roll / zoom determined (zero)."""
    sh = float(np.hypot(dx, dy))
    return {"dx": dx, "dy": dy, "shift_px": sh, "peak": peak, "matched": peak >= R.MIN_PEAK,
            "similarity": peak >= R.MIN_PEAK, "rotation_deg": 0.0, "scale_pct": 0.0,
            "motion_px": sh if peak >= R.MIN_PEAK else None, "texture": True,
            "tiles_used": 9, "tiles": 9}


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


# ------------------------------------------------------------ roll / zoom, a lost channel

def _warp(img, roll_deg=0.0, scale=1.0, dx=0.0, dy=0.0):
    """img moved about its centre: content at p goes to c + scale R(roll) (p - c) + (dx, dy)
    (x right, y down), cubic."""
    H, W = img.shape
    c = np.array([(H - 1) / 2.0, (W - 1) / 2.0])
    t = np.radians(roll_deg)
    M = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]) / scale     # (row, col)
    return ndimage.affine_transform(img, M, offset=c - M @ (c + [dy, dx]), order=3, mode="reflect")


def _pattern_cam(shape=(480, 640)):
    """The random block pattern as the camera sees it on a flat scene, normalised (0 / 1)."""
    p = R.random_block_pattern().astype(np.float64) / 255.0
    return ndimage.zoom(p, (shape[0] / p.shape[0], shape[1] / p.shape[1]), order=0)


def _noisy(img, sd, seed):
    return img + np.random.default_rng(seed).normal(0, sd, img.shape)


@pytest.mark.parametrize("roll, scale", [(0.5, 1.0), (1.0, 1.0), (-0.3, 1.0), (0.0, 1.01),
                                         (0.0, 1.02), (0.0, 0.99)])
def test_camera_roll_and_zoom_about_the_centre_are_moved(roll, scale):
    """The review case: a roll or a zoom / focus nudge about the image centre leaves the
    global (centre-weighted) shift near 0 px, yet the image corners move by several px. The
    tile similarity measures it: MOVED, with the roll / scale named."""
    scene = _texture()
    flood = R.motion(_noisy(scene, 1.0, 1), _noisy(_warp(scene, roll, scale), 1.0, 2))
    pat = _pattern_cam()
    pattern = R.motion(_noisy(pat, 0.01, 3), _noisy(_warp(pat, roll, scale), 0.01, 4))
    assert flood["shift_px"] < 0.5                 # what the single global shift saw
    assert flood["similarity"] and pattern["similarity"]
    assert abs(flood["rotation_deg"]) == pytest.approx(abs(roll), abs=0.05)
    assert flood["scale_pct"] == pytest.approx(100 * (scale - 1), abs=0.1)
    H, W = scene.shape
    corner = max(abs(scale - 1), abs(np.radians(roll))) * np.hypot(H, W) / 2
    assert flood["motion_px"] > 0.6 * corner
    v = R.assess(flood, pattern, tol_px=0.5)
    assert v["status"] == "MOVED" and v["moved"] == "camera", v
    assert "roll" in v["message"] and "scale" in v["message"]


def test_small_true_shift_still_ok_and_motion_is_the_shift():
    scene = _texture()
    pat = _pattern_cam()
    flood = R.motion(_noisy(scene, 1.0, 5), _noisy(_warp(scene, dx=0.2, dy=-0.1), 1.0, 6))
    pattern = R.motion(_noisy(pat, 0.01, 7), _noisy(_warp(pat, dx=0.2, dy=-0.1), 0.01, 8))
    assert flood["motion_px"] == pytest.approx(np.hypot(0.2, 0.1), abs=0.15)
    assert pattern["tiles_used"] == pattern["tiles"] == R.TILE_GRID ** 2
    v = R.assess(flood, pattern, tol_px=0.5)
    assert v["status"] == "OK", v
    assert v["shift_px"] == pytest.approx(np.hypot(0.2, 0.1), abs=0.15)


@pytest.mark.parametrize("roll", [1.0, 3.0, 8.0])
def test_projector_roll_with_the_flood_in_place_is_never_ok(roll):
    """The review case: the projector rolled, the camera did not. Small rolls are measured
    (MOVED projector); a roll so large that the pattern no longer matches at all (it is still
    visible) is MOVED too - before, the unusable pattern was dropped and the flood alone read
    'rig in place'."""
    scene = _texture()
    flood = R.motion(_noisy(scene, 1.0, 1), _noisy(scene, 1.0, 2))
    pat = _pattern_cam()
    pattern = R.motion(_noisy(pat, 0.01, 3), _noisy(_warp(pat, roll), 0.01, 4))
    assert flood["motion_px"] < 0.2
    v = R.assess(flood, pattern, tol_px=0.5)
    assert v["status"] == "MOVED" and v["moved"] == "projector", (v, pattern)


def test_pattern_not_seen_is_unknown_not_ok():
    """Projector off / blocked: no pattern texture at all -> UNKNOWN (projector not checked),
    never OK on the flood alone."""
    scene = _texture()
    flood = R.motion(_noisy(scene, 1.0, 1), _noisy(scene, 1.0, 2))
    pattern = R.motion(_pattern_cam(), np.zeros((480, 640)))
    assert not pattern["matched"] and not pattern["texture"]
    v = R.assess(flood, pattern)
    assert v["status"] == "UNKNOWN" and "projector not checked" in v["message"], v


def test_assess_fail_closed_rules():
    ok = _s(0.1, 0.0)
    # the pattern lost while the flood is in place: MOVED (visible) / UNKNOWN (not visible)
    lost = dict(_s(0, 0, peak=0.02), texture=True)
    assert R.assess(ok, lost)["status"] == "MOVED" and R.assess(ok, lost)["moved"] == "projector"
    dark = dict(lost, texture=False)
    assert R.assess(ok, dark)["status"] == "UNKNOWN"
    assert R.assess(ok, None)["status"] == "UNKNOWN"
    # a bare shift (phase_shift dict: roll / zoom not determined) is never OK
    bare = {"dx": 0.1, "dy": 0.0, "shift_px": 0.1, "peak": 0.9}
    v = R.assess(ok, bare)
    assert v["status"] == "UNKNOWN" and "roll / zoom not checked" in v["message"]
    assert R.assess(bare, bare)["status"] == "UNKNOWN"
    # the pattern matched in too few parts of the frame: roll / zoom not checked
    few = dict(ok, similarity=False, rotation_deg=None, scale_pct=None, tiles_used=1)
    assert R.assess(ok, few)["status"] == "UNKNOWN"
    # the flood shows texture but no longer matches (scene changed?): not OK
    assert R.assess(dict(lost), ok)["status"] == "UNKNOWN"
    # a featureless flood is fine: the pattern sees a camera move too
    v = R.assess(dict(lost, texture=False), ok)
    assert v["status"] == "OK" and "through the pattern" in v["message"]
    # a measured motion beyond tol wins over a lost channel
    assert R.assess(_s(2.0, 0.0), lost)["status"] == "MOVED"


@pytest.mark.parametrize("tol", [float("nan"), float("inf"), -0.1, "abc", None])
def test_assess_refuses_a_tolerance_that_is_not_a_finite_px_number(tol):
    """A NaN tolerance read every motion as 'rig in place' (`worst > nan` is False)."""
    big = _s(40.0, 0.0)
    with pytest.raises(ValueError, match="tolerance"):
        R.assess(big, big, tol)


def test_assess_a_nan_motion_is_moved_never_ok():
    nan = dict(_s(0.1, 0.0), motion_px=float("nan"))
    for flood, pattern in ((nan, _s(0.1, 0.0)), (_s(0.1, 0.0), nan)):
        assert R.assess(flood, pattern)["status"] == "MOVED"


# ------------------------------------------------------------ which device moved

def _moved(cam=(0.0, 0.0), proj=(0.0, 0.0), seed=0, part=None, part_shift=(0.0, 0.0)):
    """flood / pattern motions of a camera move `cam` and a projector move `proj` (camera px):
    the scene shifts by cam in the camera's view, the projected pattern by cam + proj.
    part: a mask where a part stands now that was not there for the reference - its own
    texture in the flood, the pattern on it displaced by part_shift more (another depth)."""
    scene, pat = _texture(), _pattern_cam()
    if part is None:
        sc = _warp(scene, dx=cam[0], dy=cam[1])
        pt = _warp(pat, dx=cam[0] + proj[0], dy=cam[1] + proj[1])
    else:
        on = _warp(pat, dx=proj[0] + part_shift[0], dy=proj[1] + part_shift[1])
        sc = _warp(np.where(part, _texture(seed=99), scene), dx=cam[0], dy=cam[1])
        pt = _warp(np.where(part, on, _warp(pat, dx=proj[0], dy=proj[1])), dx=cam[0], dy=cam[1])
    flood = R.motion(_noisy(scene, 1.0, seed), _noisy(sc, 1.0, seed + 1))
    pattern = R.motion(_noisy(pat, 0.01, seed + 2), _noisy(pt, 0.01, seed + 3))
    return flood, pattern


def _box(y0, y1, x0, x1, shape=(480, 640)):
    m = np.zeros(shape, bool)
    m[y0:y1, x0:x1] = True
    return m


@pytest.mark.parametrize("cam, proj, tol, status, moved", [
    ((1.0, 0.0), (0.0, 0.0), 1.3, "OK", None),
    ((1.0, 0.0), (0.0, 0.0), 0.7, "MOVED", "camera"),
    ((0.0, 0.0), (1.0, 0.0), 1.3, "OK", None),
    ((0.0, 0.0), (1.0, 0.0), 0.7, "MOVED", "projector"),
    ((1.0, 0.0), (0.0, 1.0), 0.7, "MOVED", "camera and projector"),
    ((0.8, 0.0), (0.8, 0.0), 1.2, "MOVED", "camera and projector"),   # each within, together not
    ((0.0, 0.4), (3.0, 0.0), 0.7, "MOVED", "projector"),     # the camera within the tolerance
    ((3.0, 0.0), (0.0, 0.4), 0.7, "MOVED", "camera"),        # the projector within it
])
def test_moved_names_the_device_at_the_tolerance_edge(cam, proj, tol, status, moved):
    """A camera move shifts the scene and the pattern on it alike; a projector move shifts the
    pattern against the scene. Each device is judged by its own motion: within the tolerance
    it is in place, beyond it it moved; two moves each within it but together beyond it are
    both named."""
    flood, pattern = _moved(cam, proj)
    v = R.assess(flood, pattern, tol_px=tol)
    assert v["status"] == status and v["moved"] == moved, v
    assert v["camera_motion_px"] == pytest.approx(np.hypot(*cam), abs=0.15)
    assert v["projector_motion_px"] == pytest.approx(np.hypot(*proj), abs=0.15)
    if moved == "projector":
        assert v["message"].startswith("projector (or its picture) moved ~")
        assert "can't be re-homed" in v["message"] and "camera moved" not in v["message"]
    elif moved == "camera":
        assert v["message"].startswith("camera moved ~") and "can be re-homed" in v["message"]
    elif moved:
        assert v["message"].startswith("camera and projector moved: the camera's view ~")
        assert "recalibrate" in v["message"] and "can't be re-homed" in v["message"]
    # the re-home is never suggested by the check itself (rigref adds it when it fits)
    assert "rig-check --rehome" not in v["message"]


def test_the_real_rig_case_names_the_projector_not_a_24_px_camera_move():
    """The first real-rig run: the camera's view moved 0.68 px (its mount creeps), the
    projected picture 18.5 px (the projector was nudged); the headline said 'camera moved
    ~24.0 px'. Now both are named with their own numbers, the projector's the larger, and the
    advice is a recalibration (a projector move can't be re-homed). With the camera's creep
    inside the tolerance only the projector is named."""
    flood, pattern = _moved((-0.24, 0.64), (18.52, 2.0))
    v = R.assess(flood, pattern, tol_px=0.5, mm_per_px=0.357, distance_mm=766.0)
    assert v["status"] == "MOVED" and v["moved"] == "camera and projector", v
    assert v["camera_motion_px"] == pytest.approx(0.68, abs=0.15)
    assert v["projector_motion_px"] == pytest.approx(18.63, abs=0.3)
    assert re.match(r"camera and projector moved: the camera's view ~0\.\d px \(~0\.\d\d mm\), "
                    r"the projector's picture ~18\.\d px \(~6\.\d\d mm\) against the scene "
                    r"\(largest image motion over the frame ~1\d\.\d px \(~\d\.\d\d mm at 766 mm\)",
                    v["message"]), v["message"]
    assert v["message"].endswith(": recalibrate before measuring (a projector move can't be "
                                 "re-homed)")
    flood, pattern = _moved((-0.1, 0.25), (18.52, 2.0))
    v = R.assess(flood, pattern, tol_px=0.5)
    assert v["moved"] == "projector", v
    assert re.match(r"projector \(or its picture\) moved ~18\.\d px against the scene, the "
                    r"camera's view 0\.\d px", v["message"]), v["message"]


def _grid(shifts, H=480, W=640, peak=0.9):
    """A motion() result from 3 x 3 tile shifts (row by row, camera px), the global shift their
    median, the similarity determined."""
    boxes = R._tile_boxes((H, W), 3)
    ts = [{"centre_px": [(x0 + x1 - 1) / 2.0, (y0 + y1 - 1) / 2.0], "box_px": [x0, y0, x1, y1],
           "dx": float(d[0]), "dy": float(d[1]), "peak": peak, "used": True}
          for (y0, y1, x0, x1), d in zip(boxes, shifts)]
    med = np.median(np.asarray(shifts, float), axis=0)
    mx = float(np.max(np.hypot(*np.asarray(shifts, float).T)))
    return {"dx": float(med[0]), "dy": float(med[1]), "shift_px": float(np.hypot(*med)),
            "peak": peak, "matched": True, "similarity": True, "rotation_deg": 0.0,
            "scale_pct": 0.0, "motion_px": mx, "texture": True, "tiles_used": 9, "tiles": 9,
            "shape": [H, W], "tile_shifts": ts}


def test_a_part_placed_since_the_reference_is_set_aside_not_called_a_move():
    """The reference was taken with the scene empty; a part now sits in one tile: there the
    flood and the pattern show a changed scene (a shift of neither device). That tile is set
    aside before either device is judged: the projector's 18 px move is the projector's, and
    a changed scene alone is never called a camera or projector move."""
    flood = _grid([(0.0, 0.0)] * 4 + [(0.9, 0.6)] + [(0.0, 0.0)] * 4)
    pattern = _grid([(18.0, 3.0)] * 4 + [(5.0, -1.0)] + [(18.0, 3.0)] * 4)
    who = R.who_moved(flood, pattern)
    assert who["camera_motion_px"] == pytest.approx(0.0, abs=1e-3)
    assert who["projector_motion_px"] == pytest.approx(np.hypot(18, 3), abs=1e-3)
    assert who["tiles_set_aside"] == 1
    v = R.assess(flood, pattern)
    assert v["status"] == "MOVED" and v["moved"] == "projector", v
    # nothing moved, the part's tile alone beyond the tolerance: MOVED, but which device can't
    # be told - and the message asks whether the scene changed
    v = R.assess(flood, _grid([(0.0, 0.0)] * 4 + [(-0.3, 0.2)] + [(0.0, 0.0)] * 4))
    assert v["status"] == "MOVED" and v["moved"] == "camera or projector", v
    assert "has the scene changed since the reference?" in v["message"]
    assert v["camera_motion_px"] < 0.05 and v["projector_motion_px"] < 0.05


BAR = _box(170, 320, 100, 520)                  # across the middle row of tiles
L_SHAPE = BAR | _box(170, 470, 100, 220)


@pytest.mark.parametrize("part, shift", [
    (BAR, (0.55, 0.0)), (BAR, (1.0, 0.0)), (BAR, (3.0, 0.0)), (BAR, (6.0, 0.5)),
    (L_SHAPE, (0.55, 0.0)), (L_SHAPE, (0.7, 0.0)),
    (_box(120, 400, 150, 500), (1.0, 0.0)),     # a large block
    (_box(60, 420, 330, 640), (0.55, 0.0)),     # a block over the right side
    (_box(60, 420, 330, 640), (0.7, 0.0)),
])
def test_a_wide_part_with_nothing_moved_names_no_device(part, shift):
    """Nothing moved; a part was placed since an empty-scene reference (the real case), at
    another depth (the pattern on it displaced). The tiles around it match in part and leak
    0.2 - 0.35 px into each device's motion: within the tolerance, beyond half of it - which
    named the camera ('can be re-homed') or both. Two small moves are named only on the
    reference's scene and when they explain the frame's motion: here no device is named."""
    flood, pattern = _moved(part=part, part_shift=shift)
    v = R.assess(flood, pattern, tol_px=0.5)
    assert v["moved"] in (None, "camera or projector"), v
    assert v["camera_motion_px"] <= 0.5 and v["projector_motion_px"] <= 0.5
    if v["status"] == "MOVED":
        assert "has the scene changed since the reference?" in v["message"]
        assert "can be re-homed" not in v["message"]


@pytest.mark.parametrize("cam, proj, moved", [
    ((2.0, 0.0), (0.0, 0.0), "camera"),
    ((0.0, 0.0), (6.0, 0.0), "projector"),
    ((-0.24, 0.64), (18.5, 2.9), "camera and projector"),     # the real rig's numbers
])
def test_a_device_move_is_still_named_with_a_part_placed(cam, proj, moved):
    """The same part (one tile, the pattern on it 3 px off): a device whose own motion is beyond
    the tolerance is named as before."""
    flood, pattern = _moved(cam, proj, part=_box(170, 320, 220, 420), part_shift=(3.0, 0.0))
    v = R.assess(flood, pattern, tol_px=0.5)
    assert v["status"] == "MOVED" and v["moved"] == moved, v


def test_two_small_moves_are_named_only_on_the_references_scene():
    """Camera 0.35 px + projector 0.35 px, tolerance 0.5: each within it, the picture 0.7 px.
    On the reference's scene both are named; with a part placed since the reference (one tile,
    set aside) the same moves can't be told from a changed scene."""
    flood, pattern = _moved((0.35, 0.0), (0.35, 0.0))
    v = R.assess(flood, pattern, tol_px=0.5)
    assert v["status"] == "MOVED" and v["moved"] == "camera and projector", v
    assert R._unchanged(flood, pattern, v)
    flood, pattern = _moved((0.35, 0.0), (0.35, 0.0), part=_box(170, 320, 220, 420),
                            part_shift=(3.0, 0.0))
    v = R.assess(flood, pattern, tol_px=0.5)
    assert v["status"] == "MOVED" and v["moved"] == "camera or projector", v
    assert not R._unchanged(flood, pattern, v)      # the part's tile no longer matches
    assert "two small moves together, or has the scene changed since the reference?" in v["message"]
    # pure: the same numbers, explained or not
    who = {"camera_motion_px": 0.3, "projector_motion_px": 0.3, "tiles_set_aside": 0}
    assert R._mover(0.3, who, 0.5, worst=0.6, unchanged=True) == "camera and projector"
    assert R._mover(0.3, who, 0.5, worst=0.6, unchanged=False) == "camera or projector"
    assert R._mover(0.3, who, 0.5, worst=1.0, unchanged=True) == "camera or projector"
    assert R._mover(0.3, who, 0.5, worst=float("nan"), unchanged=True) == "camera or projector"
    assert R._mover(0.3, who, 0.5, unchanged=True) == "camera or projector"     # worst unknown


def test_one_straight_edge_is_not_a_tile_match():
    """A tile that holds one straight edge only gives no shift along the edge (aperture):
    it is not used, so it cannot report a false 'no motion' along it."""
    img = np.zeros((480, 640))
    img[:, 320:] = 200.0                           # one vertical edge through the middle
    m = R.motion(_noisy(img, 0.5, 1), _noisy(np.roll(img, 3, axis=0), 0.5, 2))
    assert m["tiles_used"] == 0 and not m["similarity"]


def test_fit_similarity_recovers_roll_scale_shift():
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 1000, (9, 2))
    c = (500.0, 400.0)
    t, s = np.radians(0.4), 1.003
    M = s * np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    d = (pts - c) @ M.T + c - pts + [1.5, -0.7]
    sim = R.fit_similarity(pts, d, c)
    assert sim["rotation_deg"] == pytest.approx(0.4, abs=1e-6)
    assert sim["scale"] == pytest.approx(1.003, abs=1e-9)
    assert sim["tx"] == pytest.approx(1.5, abs=1e-6) and sim["rms_px"] < 1e-9
    assert np.allclose(R.similarity_at(sim, pts, c), d)
    assert R.fit_similarity(pts[:1], d[:1], c) is None


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
    # which device: the whole picture moved with the scene - the camera
    assert r["moved"] == "camera" and r["projector_motion_px"] < 0.3, r
    assert r["camera_motion_px"] == pytest.approx(np.hypot(2.3, 0.4), abs=0.4)
    code, out = _cli([*lib[:-1], "rig-check", "--mock-shift", "2.3,-0.4"])
    assert code == cli.EXIT_NOT_OK and "Rig check: MOVED  camera moved ~" in out
    assert re.search(r"camera / projector +camera's view 2\.\d\d px, projector's picture "
                     r"0\.\d\d px against the scene", out), out


def test_cli_mock_shift_needs_mock(tmp_path):
    code, _ = _cli(["--library", str(tmp_path / "lib"), "rig-check", "--mock-shift", "1,0"])
    assert code == cli.EXIT_REFUSED


@pytest.mark.parametrize("tol", ["nan", "inf", "-1"])
def test_cli_rig_check_refuses_a_non_finite_tolerance(tmp_path, tol, capsys):
    """`--tol nan` used to turn a 40 px move into exit 0 'rig in place'."""
    lib = ["--mock", "--average", "1", "--library", str(tmp_path / "lib"), "--json"]
    code, _ = _cli([*lib, "rig-check"])
    assert code == cli.EXIT_OK
    code, _ = _cli([*lib, "rig-check", "--mock-shift", "40,0", "--tol", tol])
    assert code == cli.EXIT_REFUSED and "finite" in capsys.readouterr().err
    api.configure(tmp_path / "lib", mock=True)
    try:
        p = api.load_preset(cli.DEFAULT_PRESET)
        with pytest.raises(ValueError, match="finite"):     # refused before any capture
            api.rig_check(p, tol_px=float(tol))
    finally:
        api._session = None


def test_api_rig_check_mock_roll_and_projector_move(preset, monkeypatch):
    """End to end in mock: a camera roll / zoom about the image centre (global shift ~0) is
    MOVED; a projector-only roll (the flood unchanged, only the pattern frame rolled) is never
    OK - before, the unmatched pattern was dropped and the flood alone read 'rig in place'."""
    from engine.calibrate import flow, rigref
    preset.camera.average = 1
    assert api.rig_check(preset)["status"] == "REFERENCE_SET"
    r = api.rig_check(preset)
    assert r["status"] == "OK" and r["pattern"]["similarity"], r
    r = api.rig_check(preset, mock_shift_px=(0.0, 0.0, 0.4))
    assert r["status"] == "MOVED" and r["shift_px"] > 2.0, r
    assert abs(r["pattern"]["rotation_deg"]) == pytest.approx(0.4, abs=0.1), r["pattern"]
    r = api.rig_check(preset, mock_shift_px=(0.0, 0.0, 0.0, 1.01))
    assert r["status"] == "MOVED" and r["pattern"]["scale_pct"] == pytest.approx(1.0, abs=0.2), r
    with pytest.raises(ValueError, match="mock_shift_px"):
        api.rig_check(preset, mock_shift_px=(0.0, 0.0, 0.0, 1.0, 5.0))
    real = flow.capture_patterns

    def projector_rolled(s, p, pats, what, zone_mask=None):
        fr = real(s, p, pats, what, zone_mask)
        fr[1] = rigref._mock_shift([fr[1]], (0.0, 0.0, 3.0))[0]       # the pattern frame only
        return fr
    monkeypatch.setattr(flow, "capture_patterns", projector_rolled)
    r = api.rig_check(preset)
    assert r["status"] == "MOVED" and r["moved"] == "projector", r


def test_api_rig_check_names_the_device_with_parts_placed_after_an_empty_reference(
        tmp_path, capsys):
    """The real case on the simulated rig: the reference was taken with the scene empty (a
    textured wall), the checks run with two cylinders standing in front of it. The rig itself
    is moved as on the bench - the camera turned about its centre, the projector turned about
    its own - and the check names what moved: never 'camera moved ~26 px' for a projector."""
    import test_rehome_mock as T
    from engine.capture.mock_bench import MockBench
    from engine.capture.mock_stereo import EndFace, MetricCylinder, StereoRig
    from conftest import FIXTURE_PRESETS
    pd = tmp_path / "presets"
    pd.mkdir()
    d = json.loads((FIXTURE_PRESETS / "cell_wall_2p7ft_vstripe_fullrange.json").read_text())
    d["name"], d["camera"]["average"] = "metric_fast", 1
    (pd / "metric_fast.json").write_text(json.dumps(d))
    bench = MockBench(seed=7)
    api.configure(tmp_path / "library", mock=True, bench=bench, presets_dir=pd)
    try:
        p = api.load_preset("metric_fast")
        rig0 = StereoRig.bench()

        def scene(Rn=np.eye(3), Rp=np.eye(3), parts=True):
            """The camera turned by Rn, the projector by Rp; the world (the wall, the parts)
            stays where it is."""
            bench.clear()
            bench.stereo_wall = False
            bench.stereo_rig = StereoRig(rig0.camera, rig0.projector, Rp @ rig0.R @ Rn.T,
                                         Rp @ rig0.t)
            bench.use_stereo()
            bench.place(T._world(Rn)[0])                        # the textured wall only
            n = np.array([0.0, 0.0, -1.0])
            for x in ((-25.0, 25.0) if parts else ()):
                c = np.array([x, 20.0, 783.0])
                bench.place(EndFace(center=tuple(Rn @ c), normal=tuple(Rn @ n), radius=10.5,
                                    albedo=0.85))
                bench.place(MetricCylinder(center=tuple(Rn @ (c - 19.0 * n)), radius=10.5,
                                           height=38.0, axis=tuple(Rn @ n), albedo=0.5))

        scene(parts=False)
        assert api.rig_check(p)["status"] == "REFERENCE_SET"
        scene()                                                 # the parts, nothing moved
        r = api.rig_check(p)
        assert r["status"] == "OK" and r["tiles_set_aside"] >= 1, r
        Rp = R.rotation_from_rotvec(np.radians([0.0, 0.47, 0.05]))     # the picture ~26 px
        scene(Rp=Rp)
        r = api.rig_check(p)
        assert r["status"] == "MOVED" and r["moved"] == "projector", r
        assert r["projector_motion_px"] > 20 and r["camera_motion_px"] < 0.2
        assert r["message"].startswith("projector (or its picture) moved ~")
        assert "camera moved" not in r["message"] and "can't be re-homed" in r["message"]
        assert r["rehome"] is None                              # never offered for a projector
        scene(Rn=T._rot(0.05, 0.02, 0.0))                       # the camera turned, ~2.4 px
        r = api.rig_check(p)
        assert r["status"] == "MOVED" and r["moved"] == "camera", r
        assert 1.8 < r["camera_motion_px"] < 3.0 and r["projector_motion_px"] < 0.3
        assert "the projected pattern moved with the scene" in r["message"]
        # the projector nudged and the camera crept 0.6 px: both, the projector the larger
        scene(Rn=T._rot(0.015, 0.0, 0.0), Rp=Rp)
        r = api.rig_check(p)
        assert r["status"] == "MOVED" and r["moved"] == "camera and projector", r
        assert 0.5 < r["camera_motion_px"] < 1.0 and r["projector_motion_px"] > 20
        assert r["message"].startswith("camera and projector moved: the camera's view ~")
        assert "recalibrate" in r["message"]
        rec = rigstate.read(api._session.store, p.geometry_id)[-1]
        assert rec["moved"] == "camera and projector"
        assert rec["projector_motion_px"] == r["projector_motion_px"]
        assert rec["message"] == r["message"]                   # the whole headline is kept
    finally:
        api._session = None

"""Tile + phase decode (engine/bases/tilephase.py, experimental) offline: a synthetic camera =
the DMD itself (identity map) with blur and noise, a small lit zone as targeted_face_scan
lights it. Regressions: the fringe polarity must come from the lit zone, not the whole-frame
mean (a sub-DN ambient drift in the unlit rest of the frame shifted half the face by T / 2),
and a code plane flipped over a patch (a glint) must drop those pixels, not decode them a
tile or more off in the half of the tile the k1 / k2 parity check does not cover. No rig."""

import cv2
import numpy as np
import pytest

from engine.bases.tilephase import TilePhaseBasis, linked_regions

Y0, Y1, X0, X1 = 540, 600, 430, 480          # a small lit face: ~0.3 % of the frame
CROP = (slice(Y0 - 40, Y1 + 40), slice(X0 - 40, X1 + 40))     # camera window for the fast cases


def _frames(pats, lit, inverted=True, seed=0):
    """Camera frames: lit zone 20 + 0.7 x the (blurred) pattern, ambient 3 DN elsewhere; the
    inverted HDMI path shows DMD 255 dark."""
    rng = np.random.default_rng(seed)
    out = []
    for pt in pats:
        p = (255.0 - pt) if inverted else pt.astype(np.float64)
        img = cv2.GaussianBlur(p.astype(np.float32), (0, 0), 0.8)
        img = np.where(lit, 20 + 0.7 * img, 3.0) + rng.normal(0, 1.0, img.shape)
        out.append(img.astype(np.float32))
    return out


@pytest.fixture(scope="module")
def tp():
    b = TilePhaseBasis((456.0, 570.0))
    pats = b.generate()
    lit = np.zeros(pats[0].shape, bool)
    lit[Y0:Y1, X0:X1] = True
    cache = {}

    def frames(inverted=True):
        if inverted not in cache:
            cache[inverted] = _frames(pats, lit, inverted)
        return [f.copy() for f in cache[inverted]]
    return {"b": b, "lit": lit, "frames": frames}


def _truth(b, shape, offset=(0, 0)):
    """True DMD (x, y) of each camera px of a window starting at offset (row, col)."""
    H, W = shape
    full = (H + offset[0], W + offset[1])
    return (b._coord("x", full)[offset[0]:, offset[1]:], b._coord("y", full)[offset[0]:, offset[1]:])


def _inner(shape, offset=(0, 0)):
    m = np.zeros(shape, bool)
    m[Y0 + 4 - offset[0]:Y1 - 4 - offset[0], X0 + 4 - offset[1]:X1 - 4 - offset[1]] = True
    return m


@pytest.mark.parametrize("inverted, drift", [(True, -0.6), (True, -1.2), (False, 0.6),
                                             (False, 1.2)])
def test_ambient_drift_outside_the_lit_zone_does_not_flip_the_phase(tp, inverted, drift):
    """The review case: the unlit 99.7 % of the frame a fraction of a DN darker (inverted path)
    or brighter (direct path) in the black frame than in the white one flipped the sign of the
    whole-frame mean of white - black, and every phase got + pi: half the face decoded exactly
    T / 2 = 4 DMD px off, silently."""
    b, lit = tp["b"], tp["lit"]
    fr = tp["frames"](inverted)
    fr[1] = np.where(lit, fr[1], fr[1] + drift).astype(np.float32)
    f = b.decode(fr, "g")
    assert f.meta["polarity_flipped"] is inverted
    tx, ty = _truth(b, lit.shape)
    inner = _inner(lit.shape)
    ex, ey = (f.corr_x - tx)[inner], (f.corr_y - ty)[inner]
    assert np.isfinite(ex).mean() > 0.99 and np.isfinite(ey).mean() > 0.99
    assert np.nanmax(np.abs(ex)) < 1.0 and np.nanmax(np.abs(ey)) < 1.0
    assert not np.isfinite(f.corr_x[~lit]).any()


@pytest.mark.parametrize("axis", ["x", "y"])
@pytest.mark.parametrize("bit", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("level", [255.0, 0.0])
def test_a_code_plane_flipped_over_a_patch_is_dropped_not_wrong(tp, axis, bit, level):
    """The review case: a glint saturates (or a shadow darkens) one Gray code plane over a
    patch of the face. Half the patch fails the k1 / k2 parity check; the other half used to
    decode 8..72 DMD px off with nothing flagging it. Now no decoded pixel is a tile off and
    the face around the patch is still decoded."""
    b, lit = tp["b"], tp["lit"]
    off = (CROP[0].start, CROP[1].start)
    fr = [f[CROP] for f in tp["frames"](True)]
    shape = fr[0].shape
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    rho = np.hypot(xx + off[1] - 455, yy + off[0] - 570)
    blob = (rho < 12) & lit[CROP]
    fi = 2 + b.steps + bit + (0 if axis == "x" else b.steps + b.bits)
    fr[fi] = np.where(blob, level, fr[fi]).astype(np.float32)
    f = b.decode(fr, "g")
    tx, ty = _truth(b, shape, off)
    for e in (f.corr_x - tx, f.corr_y - ty):
        bad = np.abs(e) > 1.0
        assert not bad.any(), (axis, bit, level, np.unique(np.round(e[bad])))
    around = _inner(shape, off) & (rho >= 16)
    assert np.isfinite(f.corr_x[around]).mean() > 0.97


def test_lit_pixels_with_the_wrong_polarity_are_dropped(tp):
    """A glint in the black frame only (it reads brighter than the white frame there): the
    pixel's own polarity disagrees with the lit zone's - not decoded."""
    b = tp["b"]
    off = (CROP[0].start, CROP[1].start)
    lit = np.zeros(tp["lit"].shape, bool)
    lit[Y0:Y1, X0:X1] = True
    fr = [f[CROP] for f in _frames(b.generate(), lit, inverted=False)]
    spot = np.zeros(fr[0].shape, bool)
    spot[560 - off[0]:566 - off[0], 450 - off[1]:456 - off[1]] = True
    fr[1] = np.where(spot, 255.0, fr[1]).astype(np.float32)
    f = b.decode(fr, "g")
    assert not f.meta["polarity_flipped"] and f.meta["polarity_odd_px"] == int(spot.sum())
    assert not np.isfinite(f.corr_x[spot]).any()
    tx, _ = _truth(b, fr[0].shape, off)
    ex = (f.corr_x - tx)[_inner(fr[0].shape, off) & ~spot]
    assert np.isfinite(ex).mean() > 0.99 and np.nanmax(np.abs(ex)) < 1.0


def test_linked_regions_keeps_checked_regions_drops_a_tile_off_patch():
    T = 8.0
    pos = np.tile(np.arange(40, dtype=np.float64) * 0.7, (30, 1))       # a smooth ramp
    cand = np.ones(pos.shape, bool)
    checked = (np.arange(40) % 4 == 0)[None, :] & cand                    # stripes of checked px
    patch = np.zeros(pos.shape, bool)
    patch[10:16, 12:18] = True
    pos2 = np.where(patch, pos + T, pos)                                 # a tile off
    keep = linked_regions(pos2, cand, checked & ~patch, T / 2)
    assert not keep[patch].any() and keep[~patch].all()
    # the same patch, but with parity-checked pixels inside it (two bits wrong together):
    # the documented limit - it is kept
    keep = linked_regions(pos2, cand, checked, T / 2)
    assert keep.all()
    # an isolated fragment with no checked pixel is dropped
    cand3 = cand.copy()
    cand3[:, 25] = False
    keep = linked_regions(pos, cand3, checked & (np.arange(40) < 25)[None, :], T / 2)
    assert keep[:, :25].all() and not keep[:, 26:].any()
    assert not linked_regions(pos, np.zeros(pos.shape, bool), checked, T / 2).any()

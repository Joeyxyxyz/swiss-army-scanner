"""Station strict cell gates (engine/measure/station/geometry.strict_gate): on a deep scene a
bright disc on the background can pass the flat / proud checks and be taken for a cell. The
synthetic station scan (tests/station_synth.py) gets such a disc painted into its flood + cloud
(a flat patch ~55 mm behind the faces, partly measured, a ragged rim): it is set aside with its
reasons and C1 / C2 are still the real cells; strict=False shows the gates are what catches it;
too few cells left -> StationError naming the reasons. No rig."""

import json
import shutil

import numpy as np
import pytest

import face_synth as S
import station_synth as SS
from engine import api
from engine.measure.station import geometry as geo_mod

DZ = -55.0          # the background patch, mm along the face normal (behind the faces)


@pytest.fixture(scope="module")
def gate(tmp_path_factory):
    lib = tmp_path_factory.mktemp("stgate") / "library"
    lib.mkdir()
    rg = S.rig()
    cal = S.write_calibration(lib, rg)
    good, sc = SS.write_station(lib, rg, cal, "20260101T050001_000000_good", seed=1,
                                s_range=(-60.0, 70.0))
    s = api.configure(lib, mock=True)
    ctx = {"lib": lib, "rg": rg, "sc": sc, "good": good, "s": s}
    yield ctx
    api._session = None


def _fake(ctx, name, level=240.0, coverage=0.55, ripple=0.04, at=(72.0, 8.0), seed=0):
    """Copy of the good scan with a background-like disc: flood `level` inside a rim of
    relative ripple `ripple`, cloud a flat patch DZ mm behind the faces, `coverage` of it
    measured, its surround undecoded."""
    rg, sc = ctx["rg"], ctx["sc"]
    src = ctx["lib"] / "metric" / S.GEO / ctx["good"]
    d = src.parent / name
    d.mkdir()
    Q = sc["O"] + at[0] * sc["e1"] + at[1] * sc["e2"] + DZ * sc["n"]
    cx, cy = rg.camera.project(Q)
    r = float(np.linalg.norm(np.subtract(rg.camera.project(Q + SS.R0 * sc["e1"]), (cx, cy))))
    with np.load(src / "cloud.npz") as z:
        xyz, msk = z["xyz"].copy(), z["mask"].copy()
    with np.load(src / "flood.npz") as z:
        w, b = z["white"].copy(), z["black"].copy()
    yy, xx = np.mgrid[0:S.CAM[0], 0:S.CAM[1]]
    rho = np.hypot(xx - cx, yy - cy)
    th = np.arctan2(yy - cy, xx - cx)
    rim = r * (1 + ripple * np.sin(7 * th))
    disc = rho < rim
    ring = (rho >= rim) & (rho < 1.7 * r)
    rng = np.random.default_rng(seed)
    # the patch: rays through the disc px hit the plane through Q with normal n
    ray = xyz / np.linalg.norm(xyz, axis=-1, keepdims=True)
    t = (Q @ sc["n"]) / (ray @ sc["n"])
    P = ray * t[..., None] + ray * rng.normal(0, 0.02, t.shape)[..., None]
    meas = disc & (rng.random(disc.shape) < coverage)
    xyz[disc] = np.nan
    xyz[meas] = P[meas]
    xyz[ring] = np.nan
    msk = np.all(np.isfinite(xyz), -1)
    w[disc] = level
    np.savez_compressed(d / "cloud.npz", xyz=xyz, mask=msk)
    np.savez_compressed(d / "flood.npz", white=w, black=b)
    shutil.copy2(src / "field.npz", d / "field.npz")
    meta = json.loads((src / "meta.json").read_text())
    meta.update(id=name, label=name)
    (d / "meta.json").write_text(json.dumps(meta))
    return name, (cx, cy)


def test_background_disc_set_aside_and_real_cells_kept(gate):
    api._session = gate["s"]
    ref = api.station_measure(gate["good"], label="ref", params={"ct": False})
    assert [c["name"] for c in ref.cells] == ["C1", "C2"] and not ref.meta["set_aside"]
    name, (fx, fy) = _fake(gate, "20260101T050002_000000_bgdisc")
    # 55 % measured and its rim circle not clean (rms ~0.39 mm > 0.3): not shown to be an end
    # face, set aside on its own (the real faces' rims are ~0.06 mm)
    for params, reason in (({"ct": False}, "% of the disc measured"),
                           ({"ct": False, "strict_min_coverage": 0.7}, "% of the disc measured")):
        m = api.station_measure(name, label="bg", params=params)
        assert len(m.cells) == 2
        for c, c0 in zip(m.cells, ref.cells):              # the same two real cells
            assert np.hypot(*np.subtract(c["camera_px"], c0["camera_px"])) < 1.0
        assert m.pairs[0]["height_diff_mm"] == pytest.approx(ref.pairs[0]["height_diff_mm"], abs=0.01)
        aside = [x for x in m.meta["set_aside"]
                 if np.hypot(x["camera_px"][0] - fx, x["camera_px"][1] - fy) < 10]
        assert aside, m.meta["set_aside"]
        assert reason in aside[0]["why"], aside
        assert any("set aside" in w for w in m.warnings)


def test_gates_are_what_catches_it_and_too_few_cells_raise(gate):
    api._session = gate["s"]
    name, (fx, fy) = _fake(gate, "20260101T050003_000000_bgdisc2", seed=1)
    loose = api.station_measure(name, label="loose", params={"ct": False, "strict": False})
    assert any(np.hypot(c["camera_px"][0] - fx, c["camera_px"][1] - fy) < 10 for c in loose.cells)
    with pytest.raises(geo_mod.StationError, match="set aside:.*measured"):
        api.station_measure(name, label="three", n_cells=3, params={"ct": False})


def test_implausible_height_alone_is_rejected(gate):
    """A fully measured, round, dimmer disc: only the pair gate (|dz| > 2 x diameter) fires."""
    api._session = gate["s"]
    name, (fx, fy) = _fake(gate, "20260101T050004_000000_bgdisc3", level=200.0, coverage=1.0,
                           ripple=0.0, seed=2)
    with pytest.raises(geo_mod.StationError, match="in height from an accepted face"):
        api.station_measure(name, label="h3", n_cells=3, params={"ct": False})
    m = api.station_measure(name, label="h2", params={"ct": False})
    assert len(m.cells) == 2


def test_strict_gate_unit_the_bench_case():
    """The numbers the false cell had on a real deep scene: rim rms 1.38 mm, coverage 0.52,
    -55 mm behind the accepted face, 68 mm away: every reason is named."""
    prm = geo_mod.StationParams.from_dict({})
    good = {"centre": np.zeros(3), "normal": np.array([0.0, 0.0, -1.0]), "coverage": 1.0,
            "outline": {"rms_mm": 0.2}}
    bad = {"centre": np.array([42.0, 0.0, 55.0]), "normal": good["normal"], "coverage": 0.52,
           "outline": {"rms_mm": 1.381}}
    why = geo_mod.strict_gate(bad, [good], prm)
    assert "ragged outline" in why and "52 %" in why and "-55.0 mm in height" in why
    assert geo_mod.strict_gate(good, [], prm) == ""
    near = dict(good, centre=np.array([30.0, 0.0, 0.2]))
    assert geo_mod.strict_gate(near, [good], prm) == ""
    far = dict(good, centre=np.array([200.0, 0.0, 0.0]))
    assert "plausible" in geo_mod.strict_gate(far, [good], prm)
    overlap = dict(good, centre=np.array([10.0, 0.0, 0.0]))
    assert "plausible" in geo_mod.strict_gate(overlap, [good], prm)


def test_real_faces_cut_by_the_zone_edge_pass_the_default_coverage():
    """Bench numbers (10-05 zoned scans): a real end face measured 0.50-0.61 of its disc,
    flat (rms 0.05 mm), round (rim rms 0.13 mm) and level with its neighbour; a flat 0.7
    coverage floor refused the whole measurement. Below 0.7 a disc now needs a clean rim
    circle instead: the background patches seen on the bench at 0.57-0.68 had none (or a
    0.8 mm one) and stay out - lowering the floor to 0.5 let them in (one measured as the
    end face of a scene with no part)."""
    prm = geo_mod.StationParams.from_dict({})
    assert prm.strict_min_coverage == 0.7 and prm.strict_partial_outline_rms_mm == 0.3
    c1 = {"centre": np.zeros(3), "normal": np.array([0.0, 0.0, -1.0]), "coverage": 0.83,
          "outline": {"rms_mm": 0.15}}
    for cov, rim in ((0.503, 0.13), (0.606, 0.13), (0.607, 0.11), (0.688, 0.09)):   # real faces
        c2 = dict(c1, centre=np.array([20.0, 0.5, 0.0]), coverage=cov, outline={"rms_mm": rim})
        assert geo_mod.strict_gate(c2, [c1], prm) == "", (cov, rim)
    for cov, outline in ((0.600, {"rms_mm": 0.84}),        # a rib patch of a tube, no part
                         (0.567, None), (0.675, None)):     # wall patches 85-117 mm behind
        why = geo_mod.strict_gate(dict(c1, coverage=cov, outline=outline), [], prm)
        assert "% of the disc measured" in why and "not shown to be an end face" in why, why
    assert geo_mod.strict_gate(dict(c1, coverage=0.75, outline=None), [], prm) == ""


def test_a_row_of_faces_longer_than_the_spacing_bound_is_accepted():
    """Each face needs a neighbour within max_pair_spacing_diam x D, not every other face:
    8 faces in a row at 1.05 D span 7.35 D (they used to be refused whatever the order)."""
    prm = geo_mod.StationParams.from_dict({"n_cells": 8})
    D = prm.face_diameter_mm
    n = np.array([0.0, 0.0, -1.0])
    row = [{"centre": np.array([1.05 * D * k, 0.0, 0.0]), "normal": n, "coverage": 1.0,
            "outline": {"rms_mm": 0.2}} for k in range(8)]
    for k in range(1, 8):
        assert geo_mod.strict_gate(row[k], row[:k], prm) == "", k
    lone = dict(row[0], centre=np.array([1.05 * D * 7 + 7 * D, 0.0, 0.0]))
    assert "nearest accepted face" in geo_mod.strict_gate(lone, row, prm)


@pytest.mark.parametrize("level", [230.0, 250.0, 255.0])
def test_a_brighter_clean_background_disc_does_not_anchor_the_pair_gates(gate, level):
    """The review case: a round, fully measured background disc BRIGHTER than the faces was
    validated first and became the reference the pair gates judged the real faces by - both
    set aside as 'background?'. The set is now chosen over every validated disc, and a disc
    as bright as the faces at another depth is never guessed away either (it may be the part
    with a face missing): the run is refused as ambiguous, naming both; cell_px_hint on the
    real faces measures them."""
    api._session = gate["s"]
    ref = api.station_measure(gate["good"], label="ref2", params={"ct": False})
    name, (fx, fy) = _fake(gate, f"20260101T0500{int(level) % 100:02d}_000001_brightbg",
                           level=level, coverage=1.0, ripple=0.0, seed=3)
    for n in (2, 1):
        with pytest.raises(geo_mod.StationError, match="ambiguous") as ei:
            api.station_measure(name, label=f"bright{n}", n_cells=n, params={"ct": False})
        assert "cell_px_hint" in str(ei.value)          # names both, says how to choose
    hint = [[round(c["camera_px"][0]), round(c["camera_px"][1])] for c in ref.cells]
    m = api.station_measure(name, label="brighth", params={"ct": False, "cell_px_hint": hint})
    assert len(m.cells) == 2
    for c, c0 in zip(m.cells, ref.cells):
        assert np.hypot(*np.subtract(c["camera_px"], c0["camera_px"])) < 1.0
    assert m.pairs[0]["height_diff_mm"] == pytest.approx(ref.pairs[0]["height_diff_mm"], abs=0.01)
    # strict off is the old greedy pick (brightest first): it takes the disc - the gates are
    # what keeps it out
    loose = api.station_measure(name, label="bright_loose", params={"ct": False, "strict": False})
    assert any(np.hypot(c["camera_px"][0] - fx, c["camera_px"][1] - fy) < 10 for c in loose.cells)


# ------------------------------------------------------------ the selection on its own

def _discs(spec, shape=(480, 900)):
    """A flood difference image with filled discs [(x, y, r, level)]."""
    import cv2
    d = np.zeros(shape, np.uint8)
    for x, y, r, lv in spec:
        cv2.circle(d, (int(x), int(y)), int(r), int(lv), -1)
    return d.astype(np.float64)


def _select(spec, depth, n_cells, bad=()):
    """find_cells with the strict selection on synthetic discs: depth[(x, y)] = z mm of each
    disc's face (centre = camera px x 0.8 mm, z); `bad` discs fail the per-disc checks."""
    prm = geo_mod.StationParams.from_dict({"n_cells": n_cells, "face_diameter_mm": 21.0})
    fits = {}

    def key(c):
        return min(depth, key=lambda p: np.hypot(p[0] - c["hough_px"][0], p[1] - c["hough_px"][1]))

    def validate(c):
        k = key(c)
        if k in bad:
            return False, "not a flat face"
        fits[id(c)] = {"centre": np.array([0.8 * k[0], 0.8 * k[1], depth[k]]),
                       "normal": np.array([0.0, 0.0, -1.0])}
        return True, "ok"

    def pair(a, b):
        return geo_mod.strict_pair(fits[id(a)], fits[id(b)], prm)
    cells, rejected = geo_mod.find_cells(_discs(spec), prm, 20.0, validate=validate, pair=pair)
    return [key(c) for c in cells], rejected


def test_selection_is_order_independent():
    faces = [(200, 240, 20, 180), (260, 240, 20, 180)]                # 48 mm apart (2.3 D)
    bg = (700, 240, 20, 250)                                          # brightest, 55 mm behind
    depth = {(200, 240): 640.0, (260, 240): 640.2, (700, 240): 695.0}
    # a dimmer lone disc at another depth: the pair is kept, the disc set aside
    got, rejected = _select(faces + [(700, 240, 20, 120)], depth, 2)
    assert sorted(got) == [(200, 240), (260, 240)]
    assert any("in height" in r["why"] for r in rejected)
    # as bright as the faces (or brighter): which is the part? refused, never guessed
    for n in (2, 1):
        with pytest.raises(geo_mod.StationError, match="ambiguous"):
            _select(faces + [bg], depth, n)
    # the background disc also failing its own checks: no ambiguity left
    got, _ = _select(faces + [bg], depth, 1, bad={(700, 240)})
    assert len(got) == 1 and got[0] in [(200, 240), (260, 240)]
    # two consistent background discs vs the real pair: refused, not the brightest pair
    bg2 = (700, 330, 20, 250)                                         # 72 mm from bg
    depth2 = {**depth, (700, 330): 695.5}
    with pytest.raises(geo_mod.StationError, match="ambiguous"):
        _select(faces + [bg, bg2], depth2, 2)
    # no consistent pair at all: refused with the reasons
    with pytest.raises(geo_mod.StationError, match="in height from an accepted face"):
        _select([faces[0], bg], depth, 2)
    # a background disc over twice as bright as the faces (a reflector): it is not the anchor
    # that makes the faces 'dimmer than the brightest face', and it is not guessed away
    dim = [(x, y, r, 100) for x, y, r, _ in faces]
    for n in (2, 1):
        with pytest.raises(geo_mod.StationError, match="ambiguous"):
            _select(dim + [bg], depth, n)


def test_selection_never_takes_a_background_set_over_a_valid_face():
    """Review cases (the fix's first version): a real face fails its own checks and the
    background holds a full, consistent set at another depth. The most complete set must not
    win silently: a valid face left out, brighter than the chosen set or in a group of its
    own, refuses the run (as the anchored selection before it did)."""
    face, bad = (200, 240, 20, 225), (260, 240, 20, 225)              # its partner fails
    bgs = [(700, 240, 20, 200), (700, 330, 20, 200)]                  # dimmer, consistent
    depth = {(200, 240): 640.0, (260, 240): 640.2, (700, 240): 695.0, (700, 330): 695.4}
    with pytest.raises(geo_mod.StationError, match="ambiguous"):
        _select([face, bad] + bgs, depth, 2, bad={(260, 240)})
    # n = 3: two valid real faces + a failed one vs three dim background discs
    faces3 = [(200, 240, 20, 225), (260, 240, 20, 225), (320, 240, 20, 225)]
    bg3 = [(700, 240, 20, 150), (650, 330, 20, 150), (650, 150, 20, 150)]
    d3 = {(200, 240): 640.0, (260, 240): 640.1, (320, 240): 640.2,
          (700, 240): 695.0, (650, 330): 695.1, (650, 150): 695.2}
    with pytest.raises(geo_mod.StationError, match="ambiguous"):
        _select(faces3 + bg3, d3, 3, bad={(320, 240)})
    # the brightness rule inside the set: a valid bright face is not set aside as 'over 2 x
    # as bright as the faces' in favour of a dim pair at its depth
    dimpair = [(500, 240, 20, 100), (560, 240, 20, 100)]
    d2 = {(200, 240): 640.0, (260, 240): 640.2, (500, 240): 640.1, (560, 240): 640.3}
    with pytest.raises(geo_mod.StationError, match="ambiguous"):
        _select([face, bad] + dimpair, d2, 2, bad={(260, 240)})


def test_selection_a_long_row_needs_neighbours_only():
    spec = [(60 + 100 * k, 240, 20, 200) for k in range(8)]           # 80 mm pitch = 3.8 D
    depth = {(60 + 100 * k, 240): 640.0 + 0.1 * k for k in range(8)}  # spans 560 mm = 27 D
    got, _ = _select(spec, depth, 8)
    assert sorted(got) == sorted(depth)
    # a gap wider than 6 D (144 mm) splits the row: no set of 8 gives every face a neighbour
    spec2 = [(x if x < 400 else x + 80, y, r, lv) for x, y, r, lv in spec]
    depth2 = {(x, y): 640.0 for x, y, _, _ in spec2}
    with pytest.raises(geo_mod.StationError, match="split into groups"):
        _select(spec2, depth2, 8)
    got, _ = _select(spec2, depth2, 4)                               # one group is a set
    assert len(got) == 4


# ------------------------------------------------------------ goldens across the gate change

def _old_style(m, path, cells=None):
    """m's .json as a measurement made before the strict gates existed (no strict keys)."""
    rep = json.loads(m.files["json"].read_text(encoding="utf-8"))
    for k in ("strict", "max_outline_rms_mm", "strict_min_coverage", "max_pair_dz_diam",
              "max_pair_spacing_diam", "min_pair_spacing_diam"):
        rep["params"].pop(k)
    if cells:
        for c in rep["cells"]:
            c.update(cells.get(c["name"], {}))
    path.write_text(json.dumps(rep), encoding="utf-8")
    return path


def test_golden_combines_a_measurement_older_than_the_gates_when_it_passes_them(gate, tmp_path):
    """The review case: a measurement .json from before the strict gates (its params lack the
    new keys) and a new one were refused as 'different station parameters'. Now its stored end
    faces are re-checked against the gates: pass -> combined (warned, the golden checks with
    the gates); fail -> refused; a new strict=False measurement goes with an old one as is;
    different gate settings or other parameters are still refused."""
    api._session = gate["s"]
    new = api.station_measure(gate["good"], label="g_new", params={"ct": False})
    old = _old_style(api.station_measure(gate["good"], label="g_old", params={"ct": False}),
                     tmp_path / "old_station.json")
    g = api.station_golden([str(old), new], "mixed")
    assert g.params["strict"] is True and g.params["strict_min_coverage"] == 0.7
    assert any("re-checked" in w and "pass" in w for w in g.warnings), g.warnings
    g = api.station_golden([new, str(old)], "mixed2")                  # either order
    assert g.params["strict"] is True
    bad = _old_style(api.station_measure(gate["good"], label="g_bad", params={"ct": False}),
                     tmp_path / "old_bad_station.json", {"C2": {"outline_rms_mm": 1.6}})
    with pytest.raises(ValueError, match="does not pass them.*ragged outline"):
        api.station_golden([new, str(bad)], "nope")
    with pytest.raises(ValueError, match="does not pass them"):
        api.station_golden([str(old), str(bad), new], "nope")
    loose = api.station_measure(gate["good"], label="g_loose", params={"ct": False, "strict": False})
    g = api.station_golden([str(bad), loose], "loose_pair")            # no gates on either side
    assert not any("re-checked" in w for w in g.warnings)
    other = api.station_measure(gate["good"], label="g_cov",
                                params={"ct": False, "strict_min_coverage": 0.6})
    with pytest.raises(ValueError, match="different strict end-face gate settings"):
        api.station_golden([new, other], "nope")
    shrink = api.station_measure(gate["good"], label="g_shrink",
                                 params={"ct": False, "face_shrink_px": 4.0})
    with pytest.raises(ValueError, match="different station parameters"):
        api.station_golden([str(old), shrink], "nope")


def _part_absent(ctx, name, ref, coverage=0.6, level=240.0, at=(72.0, 8.0), seed=0):
    """Copy of the good scan with the part gone (its faces' flood blanked, their cloud
    removed) and a clean background disc DZ mm behind, `coverage` of it measured (a
    half-disc cut by a straight edge, like a zone cut)."""
    rg, sc = ctx["rg"], ctx["sc"]
    src = ctx["lib"] / "metric" / S.GEO / ctx["good"]
    d = src.parent / name
    d.mkdir()
    with np.load(src / "cloud.npz") as z:
        xyz = z["xyz"].copy()
    with np.load(src / "flood.npz") as z:
        w, b = z["white"].copy(), z["black"].copy()
    yy, xx = np.mgrid[0:S.CAM[0], 0:S.CAM[1]]
    ray = xyz / np.linalg.norm(xyz, axis=-1, keepdims=True)
    for c in ref.cells:
        cx, cy = c["camera_px"]
        m = np.hypot(xx - cx, yy - cy) < 1.6 * c["r_px"]
        w[m] = b[m]
        xyz[m] = np.nan
    Q = sc["O"] + at[0] * sc["e1"] + at[1] * sc["e2"] + DZ * sc["n"]
    cx, cy = rg.camera.project(Q)
    r = float(np.linalg.norm(np.subtract(rg.camera.project(Q + SS.R0 * sc["e1"]), (cx, cy))))
    rho = np.hypot(xx - cx, yy - cy)
    disc, ring = rho < r, (rho >= r) & (rho < 1.7 * r)
    t = (Q @ sc["n"]) / (ray @ sc["n"])
    P = ray * t[..., None] + ray * np.random.default_rng(seed).normal(0, 0.02, t.shape)[..., None]
    meas = disc & (xx < cx + r * (2 * coverage - 1))
    xyz[disc | ring] = np.nan
    xyz[meas] = P[meas]
    w[disc] = level
    np.savez_compressed(d / "cloud.npz", xyz=xyz, mask=np.all(np.isfinite(xyz), -1))
    np.savez_compressed(d / "flood.npz", white=w, black=b)
    shutil.copy2(src / "field.npz", d / "field.npz")
    meta = json.loads((src / "meta.json").read_text())
    meta.update(id=name, label=name)
    (d / "meta.json").write_text(json.dumps(meta))
    return name, (cx, cy)


def test_part_missing_a_background_disc_never_passes_the_station_check(gate):
    """Review case: the part absent and a clean background disc ~60 % measured left in view.
    A 0.5 coverage floor took it for the end face (this synthetic disc has a clean rim, so it
    still IS measured as a face), and a 1-face golden without tube zones compared nothing:
    the station check said PASS. Every end face is now checked where it is against the
    golden (within the face diameter by default): FAIL. A golden that compares nothing (made
    before the face positions) is INCOMPLETE, never PASS."""
    import dataclasses
    api._session = gate["s"]
    ref = api.station_measure(gate["good"], label="pm_ref", params={"ct": False})
    g = api.station_golden([api.station_measure(gate["good"], label=f"pm_g{i}", n_cells=1,
                                                params={"ct": False}) for i in range(2)],
                           "pm_gold")
    assert g.nominal["cells"][0]["centre_mm"] is not None
    ok = api.station_check(gate["good"], golden=g)
    assert ok.verdict == "PASS", ok.items
    pos = [it for it in ok.items if it["kind"] == "face_position"]
    assert pos and pos[0]["status"] == "ok" and pos[0]["value_mm"] < 0.5
    name, _ = _part_absent(gate, "20260101T090001_000000_partmissing", ref)
    chk = api.station_check(name, golden=g)
    assert chk.verdict == "FAIL", chk.items
    assert any(it["item"] == "C1 position" for it in chk.triggers)
    # a golden from before the face positions compares nothing here: never a PASS
    old = dataclasses.replace(g, nominal={**g.nominal, "cells": [
        {k: v for k, v in c.items() if k != "centre_mm"} for c in g.nominal["cells"]]})
    chk = api.station_check(name, golden=old)
    assert chk.verdict == "INCOMPLETE", chk.items
    assert any("checks nothing" in w for w in chk.warnings)
    assert any("predates the face positions" in w for w in chk.warnings)


def test_strict_params_validated_and_cli_flag():
    with pytest.raises(ValueError, match="strict_min_coverage"):
        geo_mod.StationParams.from_dict({"strict_min_coverage": 0})
    with pytest.raises(ValueError, match="max_outline_rms_mm"):
        geo_mod.StationParams.from_dict({"max_outline_rms_mm": -1})
    from ui import cli
    a = cli.build_parser().parse_args(["station", "x", "--no-strict"])
    assert cli._station_params(a)["strict"] is False

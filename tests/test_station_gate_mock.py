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
    m = api.station_measure(name, label="bg", params={"ct": False})
    assert len(m.cells) == 2
    for c, c0 in zip(m.cells, ref.cells):              # the same two real cells
        assert np.hypot(*np.subtract(c["camera_px"], c0["camera_px"])) < 1.0
    assert m.pairs[0]["height_diff_mm"] == pytest.approx(ref.pairs[0]["height_diff_mm"], abs=0.01)
    aside = [x for x in m.meta["set_aside"] if np.hypot(x["camera_px"][0] - fx, x["camera_px"][1] - fy) < 10]
    assert aside, m.meta["set_aside"]
    why = aside[0]["why"]
    assert "% of the disc measured" in why
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


def test_strict_params_validated_and_cli_flag():
    with pytest.raises(ValueError, match="strict_min_coverage"):
        geo_mod.StationParams.from_dict({"strict_min_coverage": 0})
    with pytest.raises(ValueError, match="max_outline_rms_mm"):
        geo_mod.StationParams.from_dict({"max_outline_rms_mm": -1})
    from ui import cli
    a = cli.build_parser().parse_args(["station", "x", "--no-strict"])
    assert cli._station_params(a)["strict"] is False

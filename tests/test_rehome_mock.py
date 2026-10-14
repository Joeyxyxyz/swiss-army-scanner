"""Re-home after a camera nudge (`scan rig-check --rehome`, api.rig_check(rehome=True)).

The convention is locked here by TURNING THE SIMULATED CAMERA: the mock stereo bench is
rebuilt with the camera rotated by a known small rotation about its own centre (the world -
a textured wall, a sphere, a block - and the projector stay where they are), the rig check
registers the reference flood against the fresh one, and the re-homed calibration
R1 = R0 @ Rd (t unchanged) must triangulate the scene back onto its true points and scale.
The opposite composition must make it worse than doing nothing. Also: the reference
messages (new calibration -> new reference stored), the refusals (projector moved, too
large, not a rotation, no reference of this calibration) and the command line."""

import contextlib
import io
import json
from dataclasses import dataclass, field

import numpy as np
import pytest
from scipy import ndimage

from engine import api, config
from engine.calibrate import stereo
from engine.capture import rigcheck as RC
from engine.capture.mock_bench import MockBench
from engine.capture.mock_stereo import (Block, Plane, Sphere, StereoRig, distort_project)
from engine.decode.deformation import DeformationField
from engine.reconstruct.triangulate import triangulate
from conftest import FIXTURE_PRESETS
from recipe_synth import truth_calibration
from ui import cli


def _rot(pan_deg=0.0, tilt_deg=0.0, roll_deg=0.0) -> np.ndarray:
    """R_new_from_old of a camera turned by these angles: a point at X_old in the camera
    frame before the move is at X_new = R @ X_old after it. pan > 0 moves the image +x,
    tilt > 0 moves it +y, roll > 0 turns it x towards y (the rig check's own signs)."""
    return RC.rotation_from_rotvec(np.radians([-tilt_deg, pan_deg, roll_deg]))


# ------------------------------------------------------------ the pure fit

@pytest.mark.parametrize("pan, tilt, roll", [(0.05, 0.0, 0.0), (0.0, -0.04, 0.0),
                                             (0.0, 0.0, 0.08), (0.04, 0.03, -0.05)])
def test_camera_rotation_is_recovered_with_its_signs(pan, tilt, roll):
    """Points anywhere in depth, a distorted lens: the image shifts of a pure camera
    rotation give that rotation back, and pan / tilt / roll carry the image-motion signs."""
    rig = StereoRig.bench()
    K, dist = rig.camera.K, rig.camera.dist
    rng = np.random.default_rng(0)
    px = np.c_[rng.uniform(80, 1360, 40), rng.uniform(60, 1020, 40)]
    rays = np.c_[RC._normalised(px, K, dist), np.ones(len(px))]
    X_old = rays * rng.uniform(400, 1200, (len(px), 1))          # depth does not matter
    Rn = _rot(pan, tilt, roll)
    moved = distort_project(X_old @ Rn.T, K, dist)
    assert np.abs(distort_project(X_old, K, dist) - px).max() < 1e-6
    fit = RC.camera_rotation(px, moved - px, K, dist)
    assert np.allclose(fit["R_new_from_old"], Rn, atol=1e-9)
    assert np.allclose(fit["R"], Rn.T, atol=1e-9) and fit["residual_px"] < 1e-6
    assert fit["pan_deg"] == pytest.approx(pan, abs=1e-6)
    assert fit["tilt_deg"] == pytest.approx(tilt, abs=1e-6)
    assert fit["roll_deg"] == pytest.approx(roll, abs=1e-6)
    # what a person reads: the shift at the principal point is pan * fx, tilt * fy
    assert fit["shift_px"][0] == pytest.approx(np.radians(pan) * K[0, 0], abs=0.08)
    assert fit["shift_px"][1] == pytest.approx(np.radians(tilt) * K[1, 1], abs=0.08)
    assert abs(fit["scale_pct"]) < 0.02
    assert RC.camera_rotation(px[:2], (moved - px)[:2], K, dist) is None


def _tiles(pts, shifts, peak=0.9):
    return {"peak": peak, "matched": True, "similarity": True, "tiles": len(pts),
            "tiles_used": len(pts), "shift_px": float(np.hypot(*np.mean(shifts, 0))),
            "motion_px": float(np.max(np.hypot(shifts[:, 0], shifts[:, 1]))), "texture": True,
            "tile_shifts": [{"centre_px": [float(p[0]), float(p[1])], "dx": float(d[0]),
                             "dy": float(d[1]), "peak": peak, "used": True}
                            for p, d in zip(pts, shifts)]}


def test_rehome_fit_refuses_everything_that_is_not_a_small_camera_rotation():
    rig = StereoRig.bench()
    K, dist = rig.camera.K, rig.camera.dist
    gx, gy = np.meshgrid(np.linspace(150, 1290, 5), np.linspace(110, 970, 5))
    px = np.c_[gx.ravel(), gy.ravel()]
    X = np.c_[RC._normalised(px, K, dist), np.ones(len(px))] * 800.0

    def shifts(Rn):
        return distort_project(X @ Rn.T, K, dist) - px

    cam = shifts(_rot(0.04, -0.02, 0.03))
    ok = RC.rehome_fit(_tiles(px, cam), _tiles(px, cam), K, dist)
    assert ok["applicable"] and ok["why"] is None and ok["pattern_rms_px"] < 0.01
    assert ok["rotation_deg"] == pytest.approx(np.degrees(np.linalg.norm(
        np.radians([0.02, 0.04, 0.03]))), abs=1e-4)
    # the projector moved too: the pattern does not follow the flood
    proj = cam + [1.2, 0.0]
    r = RC.rehome_fit(_tiles(px, cam), _tiles(px, proj), K, dist)
    assert not r["applicable"] and "not camera-like" in r["why"] and "projector" in r["why"]
    # the projector moved, the camera did not (no flood motion to fit, pattern apart)
    still = np.zeros_like(cam)
    r = RC.rehome_fit(_tiles(px, still), _tiles(px, still + [1.5, 0.3]), K, dist)
    assert not r["applicable"] and "not camera-like" in r["why"]
    # too large
    big = shifts(_rot(0.2, 0.0, 0.0))
    r = RC.rehome_fit(_tiles(px, big), _tiles(px, big), K, dist)
    assert not r["applicable"] and "too large" in r["why"] and r["motion_px"] > RC.REHOME_MAX_PX
    # a zoom / refocus (scale about the centre) is not a rotation
    c = np.array([K[0, 2], K[1, 2]])
    zoom = (px - c) * 0.003
    r = RC.rehome_fit(_tiles(px, zoom), _tiles(px, zoom), K, dist)
    assert not r["applicable"] and "not a camera rotation" in r["why"]
    assert r["tiles_agree"] < RC.REHOME_MIN_AGREE * len(px)
    # a few tiles matched at the wrong place (soft texture) do not drag the rotation ...
    few = cam.copy()
    few[[3, 11, 17]] += [[4.0, -6.0], [-3.0, 1.0], [0.8, 0.9]]
    r = RC.rehome_fit(_tiles(px, few), _tiles(px, cam), K, dist)
    assert r["applicable"] and r["tiles_agree"] == len(px) - 3
    assert r["pan_deg"] == pytest.approx(0.04, abs=1e-4) and r["motion_px"] < 3.0
    # ... but too many of them are not one rotation
    many = cam.copy()
    many[::2] += np.random.default_rng(2).normal(0, 2.0, many[::2].shape)
    r = RC.rehome_fit(_tiles(px, many), _tiles(px, cam), K, dist)
    assert not r["applicable"] and "agree on one rotation" in r["why"]
    # poor registration: a weak match, too few tiles, an unmatched flood or pattern
    r = RC.rehome_fit(_tiles(px, cam, peak=0.4), _tiles(px, cam), K, dist)
    assert not r["applicable"] and "poor" in r["why"]
    r = RC.rehome_fit(_tiles(px[:4], cam[:4]), _tiles(px, cam), K, dist)
    assert not r["applicable"] and "poor" in r["why"]
    noisy = cam + np.random.default_rng(1).normal(0, 0.6, cam.shape)
    r = RC.rehome_fit(_tiles(px, noisy), _tiles(px, cam), K, dist)
    assert not r["applicable"] and "agree on one rotation" in r["why"]
    for flood, pattern in ((None, _tiles(px, cam)), (_tiles(px, cam), None),
                           (dict(_tiles(px, cam), similarity=False), _tiles(px, cam))):
        assert not RC.rehome_fit(flood, pattern, K, dist)["applicable"]


def test_rehome_writes_a_new_flagged_entry_and_refuses_what_is_not_a_nudge(tmp_path):
    """stereo.rehome: R1 = R0 @ Rd in a NEW entry, t and the intrinsics unchanged, flagged
    with where it came from; a second re-home chains to the same root; a rotation that is
    not small, or not a rotation at all, is refused before anything is written."""
    from engine.library.store import LibraryStore
    store = LibraryStore(tmp_path / "lib")
    tr = StereoRig.bench().truth()
    d = store.new_dir("calibrations", "geo", label="stereo")
    cal0 = stereo.StereoCalibration(id=f"geo/{d.name}", geometry_id="geo", K_cam=tr["K_cam"],
                                    dist_cam=tr["dist_cam"], K_proj=tr["K_proj"],
                                    dist_proj=np.zeros(5), R=tr["R"], t=tr["t"],
                                    diamond_parity=None, parity_source="row_parity", path=d)
    store.write_json(d / "stereo_calibration.json", cal0.to_json())
    before = (d / "stereo_calibration.json").read_bytes()
    Rd = _rot(0.04, 0.01, -0.02).T
    cal1 = stereo.rehome(store, cal0, Rd, created="2026-03-03T00:00:00",
                         details={"reference_id": "ref"})
    assert cal1.id != cal0.id and cal1.path != cal0.path and cal1.path.name.endswith("rehomed")
    assert np.allclose(cal1.R, cal0.R @ Rd) and np.allclose(cal1.R @ cal1.R.T, np.eye(3))
    assert np.array_equal(cal1.t, cal0.t) and np.array_equal(cal1.K_proj, cal0.K_proj)
    assert cal1.parity_source == "row_parity" and cal1.parity_measured is True
    assert cal1.rehomed["from"] == cal0.id and cal1.rehomed["generation"] == 1
    assert cal1.rehomed["note"] == f"re-homed (rotation only) from {cal0.id}"
    assert "full calibration is the reference" in cal1.rehomed["caveat"]
    assert (d / "stereo_calibration.json").read_bytes() == before         # never overwritten
    assert stereo.latest(store, "geo").id == cal1.id                      # the one in use now
    back = stereo.latest(store, "geo")
    assert back.rehomed["reference_id"] == "ref" and "re-homed" in back.to_json()["note"]
    cal2 = stereo.rehome(store, back, _rot(0.0, 0.02, 0.0).T, created="2026-03-04T00:00:00")
    assert cal2.rehomed["generation"] == 2 and cal2.rehomed["root"] == cal0.id
    assert cal2.rehomed["from"] == cal1.id
    n = len(store.entries("calibrations", "geo"))
    with pytest.raises(ValueError, match="not a nudge"):
        stereo.rehome(store, cal0, _rot(1.5, 0.0, 0.0).T, created="x")
    with pytest.raises(ValueError, match="not a rotation"):
        stereo.rehome(store, cal0, np.eye(3) * 1.01, created="x")
    with pytest.raises(ValueError, match="not a rotation"):
        stereo.rehome(store, cal0, np.diag([1.0, 1.0, -1.0]), created="x")
    assert len(store.entries("calibrations", "geo")) == n                 # nothing written


# ------------------------------------------------------------ the simulated camera, turned

_TEX: dict = {}


def _texture(seed: int) -> np.ndarray:
    """A random albedo over the wall with fine and coarse detail (a painted, slightly
    stained surface), 0.5 mm per sample, x -700..700, y -520..520."""
    if seed not in _TEX:
        rng = np.random.default_rng(seed)
        n = rng.random((2081, 2801))
        t = ndimage.gaussian_filter(n, 1.5) / 0.19 + ndimage.gaussian_filter(n, 8.0) / 0.035
        _TEX[seed] = ((t - t.min()) / (t.max() - t.min())).astype(np.float32)
    return _TEX[seed]


@dataclass
class TexturedWall(Plane):
    """The wall with a texture fixed to the WORLD (= the camera frame before any move):
    Rn turns world coordinates into the present camera frame."""
    Rn: np.ndarray = field(default_factory=lambda: np.eye(3))
    seed: int = 0

    def albedo_at(self, X):
        Xw = np.asarray(X, float) @ self.Rn                      # back to the world
        return (0.25 + 0.55 * ndimage.map_coordinates(
            _texture(self.seed), [2.0 * (Xw[:, 1] + 520.0), 2.0 * (Xw[:, 0] + 700.0)], order=1,
            mode="nearest")).astype(np.float32)


def _world(Rn: np.ndarray, seed: int = 0) -> list:
    """The fixed scene, expressed in the camera frame of a camera turned by Rn."""
    wall = TexturedWall(point=tuple(Rn @ [0.0, 0.0, 823.0]), normal=tuple(Rn @ [0.0, 0.0, -1.0]),
                        Rn=Rn, seed=seed)
    return [wall, Sphere(center=tuple(Rn @ [-110.0, 40.0, 780.0]), radius=22.0),
            Block(center=tuple(Rn @ [120.0, -60.0, 813.0]), half=(35.0, 25.0, 10.0), rot=Rn.T)]


def _set_camera(bench, rig0, Rn=np.eye(3), Rp=np.eye(3), seed: int = 0) -> None:
    """Rebuild the bench with the camera turned by Rn (X_new = Rn @ X_old) and the projector
    turned by Rp about its own centre; the world does not move."""
    bench.clear()
    bench.stereo_wall = False
    bench.stereo_rig = StereoRig(rig0.camera, rig0.projector, Rp @ rig0.R @ Rn.T, Rp @ rig0.t)
    bench.use_stereo()
    for shape in _world(Rn, seed):
        bench.place(shape)


@pytest.fixture(scope="module")
def rig(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("rehome")
    pd = tmp / "presets"
    pd.mkdir()
    d = json.loads((FIXTURE_PRESETS / "cell_wall_2p7ft_vstripe_fullrange.json").read_text())
    d["name"] = "metric_fast"
    d["camera"]["average"] = 1
    (pd / "metric_fast.json").write_text(json.dumps(d))
    bench = MockBench(seed=7)
    s = api.configure(tmp / "library", mock=True, bench=bench, presets_dir=pd)
    p = api.load_preset("metric_fast")
    rig0 = StereoRig.bench()
    _set_camera(bench, rig0)
    cal0 = truth_calibration(s, p)                    # the calibration of the unmoved rig
    yield {"s": s, "p": p, "bench": bench, "rig0": rig0, "cal0": cal0, "tmp": tmp, "pd": pd}
    api._session = None


def _use(rig):
    api._session = rig["s"]
    return rig["p"]


def _stereo_cals(p) -> list:
    return [c for c in api.list_library("calibrations", p.geometry_id)
            if c.get("kind") == "stereo"]


def _errors(bench, cal) -> dict:
    """Triangulate the present scene's exact correspondences with `cal`: point error vs the
    true surface (mm), and the scale error of distances between far-apart points (%)."""
    sc = bench.stereo_scene()
    cx, cy, m = sc.oracle()
    fld = DeformationField(corr_x=cx, corr_y=cy, corr_diag=None, conf=m.astype(np.float32),
                           mask=m, geometry_id=cal.geometry_id, basis_names=["oracle"])
    mc = triangulate(fld, cal)
    truth = sc.geometry()["X"]
    ok = mc.mask & m
    got, want = mc.xyz[ok][::97], truth[ok][::97]
    err = np.linalg.norm(got - want, axis=1)
    rng = np.random.default_rng(0)
    i, j = rng.integers(0, len(got), (2, 4000))
    far = np.linalg.norm(want[i] - want[j], axis=1) > 150.0
    ratio = np.linalg.norm(got[i] - got[j], axis=1)[far] / np.linalg.norm(want[i] - want[j],
                                                                           axis=1)[far]
    return {"median_mm": float(np.median(err)), "p95_mm": float(np.percentile(err, 95)),
            "scale_pct": float(100.0 * (np.median(ratio) - 1.0)),
            "scale_spread_pct": float(100.0 * np.std(ratio))}


def test_new_calibration_says_new_reference_stored(rig):
    """The first check after a calibration stores the reference and says why: not 'no rig
    reference at this geometry yet' (true but misleading: it read as a silent replace)."""
    p = _use(rig)
    r = api.rig_check(p)
    assert r["status"] == "REFERENCE_SET" and r["reference_reason"] == "new_calibration", r
    assert f"new calibration {rig['cal0'].id}" in r["message"]
    assert "new reference stored" in r["message"] and "no rig reference" not in r["message"]
    assert r["previous_reference_id"] is None
    rig["ref0"] = r["reference_id"]
    r = api.rig_check(p)
    assert r["status"] == "OK" and r["rehome"] is None, r
    assert r["flood"]["similarity"] and r["pattern"]["similarity"]
    # the same calibration, a reference on request: says so, and names the one it replaces
    r2 = api.rig_check(p, set_reference=True)
    assert r2["reference_reason"] == "requested" and "on request" in r2["message"]
    assert r2["previous_reference_id"] == rig["ref0"] and rig["ref0"] in r2["message"]


def test_turned_camera_is_rehomed_and_the_convention_is_r0_rd(rig):
    """The simulated camera turned by a known rotation: MOVED camera, a re-home offered;
    --rehome writes a NEW calibration R1 = R0 @ Rd (t unchanged, flagged, original kept),
    the triangulated scene is back on its true points and scale, and the opposite sign is
    worse than the stale calibration."""
    p = _use(rig)
    if "ref0" not in rig:
        pytest.skip("needs the reference test")
    bench, rig0, cal0 = rig["bench"], rig["rig0"], rig["cal0"]
    Rn = _rot(pan_deg=0.045, tilt_deg=-0.025, roll_deg=0.04)       # ~2 px, ~1 px, corners 0.6 px
    _set_camera(bench, rig0, Rn)
    truth = bench.stereo_truth()
    assert np.allclose(truth["R"], rig0.R @ Rn.T) and np.allclose(truth["t"], rig0.t)

    r = api.rig_check(p)                                           # a plain check: it offers
    assert r["status"] == "MOVED" and r["moved"] == "camera", r
    assert 1.5 < r["shift_px"] < RC.REHOME_MAX_PX
    rh = r["rehome"]
    assert rh["applicable"] and not rh["done"] and "rig-check --rehome" in r["message"]
    assert "full calibration" in r["message"] and len(_stereo_cals(p)) == 1    # nothing written
    assert rh["pan_deg"] == pytest.approx(0.045, abs=0.003)
    assert rh["tilt_deg"] == pytest.approx(-0.025, abs=0.003)
    assert rh["roll_deg"] == pytest.approx(0.04, abs=0.004)
    assert rh["residual_px"] < 0.15 and rh["pattern_rms_px"] < 0.2 and abs(rh["scale_pct"]) < 0.03

    stale = _errors(bench, cal0)
    before = json.loads((cal0.path / "stereo_calibration.json").read_text())
    r = api.rig_check(p, rehome=True)
    assert r["status"] == "REHOMED" and r["rehome"]["done"], r
    new_id = r["rehome"]["calibration_id"]
    assert new_id != cal0.id and r["calibration_id"] == new_id and r["rehome"]["from"] == cal0.id
    assert f"re-homed (rotation only) from {cal0.id}" in r["message"]
    assert "new calibration -> new reference stored" in r["message"]
    assert any("absolute depth" in w and "full calibration" in w for w in r["warnings"])
    assert any("bound to that calibration" in w for w in r["warnings"])
    # the original entry is untouched; the new one is the calibration in use, flagged
    assert json.loads((cal0.path / "stereo_calibration.json").read_text()) == before
    cal1 = stereo.latest(rig["s"].store, p.geometry_id, config.geometry_fingerprint(p))
    assert cal1.id == new_id and cal1.rehomed["from"] == cal0.id
    assert cal1.rehomed["note"] == f"re-homed (rotation only) from {cal0.id}"
    assert cal1.rehomed["generation"] == 1 and cal1.rehomed["root"] == cal0.id
    doc = json.loads((cal1.path / "stereo_calibration.json").read_text())
    assert doc["rehomed"]["t_unchanged"] and "re-homed (rotation only)" in doc["note"]
    assert np.allclose(cal1.t, cal0.t) and np.allclose(cal1.K_cam, cal0.K_cam)
    assert np.allclose(cal1.K_proj, cal0.K_proj) and cal1.diamond_parity == cal0.diamond_parity
    # R1 = R0 @ Rd, and it is the turned rig's true rotation
    Rd = np.asarray(cal1.rehomed["Rd"])
    assert np.allclose(cal1.R, cal0.R @ Rd, atol=1e-9)
    assert stereo.compare_truth(cal1, truth)["R_deg"] < 0.003
    assert stereo.compare_truth(cal0, truth)["R_deg"] > 0.05
    # triangulated positions and scale recover
    good = _errors(bench, cal1)
    assert stale["median_mm"] > 1.0 and abs(stale["scale_pct"]) > 0.05, stale
    assert good["median_mm"] < 0.1 and good["p95_mm"] < 0.2, good
    assert abs(good["scale_pct"]) < 0.01 and good["median_mm"] < stale["median_mm"] / 20
    # the opposite sign (R0 @ Rd^T) is worse than not re-homing at all: the convention
    wrong = stereo.StereoCalibration(**{**{f: getattr(cal1, f) for f in (
        "geometry_id", "K_cam", "dist_cam", "K_proj", "dist_proj", "t", "cam_size", "proj_size",
        "diamond_parity")}, "id": "wrong", "R": cal0.R @ Rd.T})
    bad = _errors(bench, wrong)
    assert bad["median_mm"] > 1.7 * stale["median_mm"], (bad, stale)
    assert abs(bad["scale_pct"]) > 1.5 * abs(stale["scale_pct"])
    # the fresh frames are the new calibration's reference: the next check is OK
    r = api.rig_check(p)
    assert r["status"] == "OK" and r["calibration_id"] == new_id and not r["warnings"][1:], r
    assert r["reference_id"] != rig["ref0"]
    rig.update(cal1=cal1, Rn=Rn)


def test_rehome_with_nothing_moved_does_nothing(rig):
    p = _use(rig)
    if "cal1" not in rig:
        pytest.skip("needs the re-home test")
    n = len(_stereo_cals(p))
    r = api.rig_check(p, rehome=True)
    assert r["status"] == "OK" and not r["rehome"]["done"], r
    assert "nothing to re-home" in r["rehome"]["why"] and len(_stereo_cals(p)) == n


def test_rehome_refused_when_the_projector_moved_or_the_move_is_too_large(rig):
    """Never written: the projector turned (alone, or with the camera), a move beyond ~5 px,
    a zoom. The status stays MOVED and says why."""
    p = _use(rig)
    if "cal1" not in rig:
        pytest.skip("needs the re-home test")
    bench, rig0, Rn = rig["bench"], rig["rig0"], rig["Rn"]
    n = len(_stereo_cals(p))
    Rp = RC.rotation_from_rotvec(np.radians([0.0, 0.12, 0.0]))     # the projector, ~2.5 DMD px
    cases = {"projector": (Rn, Rp), "both": (_rot(0.03, 0.02, 0.0) @ Rn, Rp),
             "large": (_rot(0.16, 0.0, 0.0) @ Rn, np.eye(3))}
    for name, (Rc, Rpj) in cases.items():
        _set_camera(bench, rig0, Rc, Rpj)
        plain = api.rig_check(p)
        r = api.rig_check(p, rehome=True)
        assert plain["status"] == r["status"] == "MOVED", (name, r)
        assert not r["rehome"]["done"] and "re-home REFUSED" in r["message"], (name, r)
        assert not (plain["rehome"] or {}).get("applicable"), (name, plain["rehome"])
        assert "rig-check --rehome" not in plain["message"]
        assert len(_stereo_cals(p)) == n, name
        if name == "projector":
            assert r["moved"] == "projector" and "CAMERA move" in r["rehome"]["why"]
        elif name == "both":                        # the check itself names both
            assert r["moved"] == "camera and projector" and "can't be re-homed" in r["message"]
            assert "(camera and projector moved)" in r["rehome"]["why"]
            assert "CAMERA move" in r["rehome"]["why"]
        else:
            assert "too large" in r["rehome"]["why"] and "recalibrate" in r["rehome"]["why"]
    _set_camera(bench, rig0, Rn)                                   # back in place
    r = api.rig_check(p, mock_shift_px=(0.0, 0.0, 0.0, 1.004), rehome=True)     # a zoom
    assert r["status"] == "MOVED" and not r["rehome"]["done"]
    assert "not a camera rotation" in r["rehome"]["why"] and len(_stereo_cals(p)) == n
    assert api.rig_check(p)["status"] == "OK"


def test_rehome_needs_the_reference_of_the_calibration_in_use(rig):
    """Refused before any frame is captured: both flags at once, no calibration, no
    reference, or a reference that belongs to another calibration."""
    p = _use(rig)
    if "cal1" not in rig:
        pytest.skip("needs the re-home test")
    from engine.calibrate import flow
    shots = []
    real = flow.capture_patterns

    def spy(*a, **k):
        shots.append(1)
        return real(*a, **k)
    flow.capture_patterns, keep = spy, real
    try:
        with pytest.raises(ValueError, match="exclude each other"):
            api.rig_check(p, set_reference=True, rehome=True)
        other = config.load_preset("metric_fast", presets_dir=rig["pd"], mock=True)
        other.rig.baseline_mm = 300.0                 # same geometry id, other settings
        with pytest.raises(ValueError, match="no stereo calibration"):
            api.rig_check(other, rehome=True)
        # a newer calibration without a reference of its own
        cal1 = rig["cal1"]
        d = rig["s"].store.new_dir("calibrations", p.geometry_id, label="stereo")
        newer = stereo.StereoCalibration(**{**{f: getattr(cal1, f) for f in (
            "geometry_id", "K_cam", "dist_cam", "K_proj", "dist_proj", "R", "t", "cam_size",
            "proj_size", "diamond_parity", "fingerprint")}, "id": f"{p.geometry_id}/{d.name}",
            "path": d, "created": "2026-02-02T00:00:00"})
        rig["s"].store.write_json(d / "stereo_calibration.json", newer.to_json())
        with pytest.raises(ValueError, match="belongs to"):
            api.rig_check(p, rehome=True)
        assert shots == []                            # nothing was projected or captured
    finally:
        flow.capture_patterns = keep
    # a plain check warns; a reference on request names the new calibration
    r = api.rig_check(p)
    assert r["status"] == "OK" and any("set a new reference" in w for w in r["warnings"])
    r = api.rig_check(p, set_reference=True)
    assert r["reference_reason"] == "new_calibration"
    assert f"new calibration {newer.id} -> new reference stored" in r["message"]
    assert f"belonged to {cal1.id}" in r["message"]


# ------------------------------------------------------------ the command line

def _cli(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main(argv)
    api._session = None
    return code, out.getvalue(), err.getvalue()


def test_cli_rig_check_rehome(tmp_path):
    """reference -> a simulated 2 px camera nudge: the plain check suggests --rehome,
    --rehome writes the flagged calibration (exit 0), `calibrations` lists both, `delete`
    trashes the re-homed one and the original is in use again."""
    lib = tmp_path / "lib"
    s = api.configure(lib, mock=True)
    try:
        p = api.load_preset(cli.DEFAULT_PRESET)
        cal0 = truth_calibration(s, p)
    finally:
        api._session = None
    base = ["--mock", "--average", "1", "--library", str(lib)]
    code, out, _ = _cli([*base, "--json", "rig-check"])
    r = json.loads(out)
    assert code == cli.EXIT_OK and r["reference_reason"] == "new_calibration", r
    code, out, err = _cli([*base, "rig-check", "--rehome", "--set-reference"])
    assert code == cli.EXIT_REFUSED and "exclude each other" in err
    code, out, _ = _cli([*base, "rig-check", "--rehome"])           # nothing moved
    assert code == cli.EXIT_OK and "nothing to re-home" in out
    code, out, _ = _cli([*base, "rig-check", "--mock-shift", "2.0,-0.8"])
    assert code == cli.EXIT_NOT_OK and "MOVED" in out
    assert "scan rig-check --rehome" in out and "camera rotation" in out
    code, out, _ = _cli([*base, "rig-check", "--mock-shift", "9,0", "--rehome"])
    assert code == cli.EXIT_NOT_OK and "re-home REFUSED" in out and "too large" in out
    code, out, _ = _cli([*base, "--json", "rig-check", "--mock-shift", "2.0,-0.8", "--rehome"])
    r = json.loads(out)
    assert code == cli.EXIT_OK and r["status"] == "REHOMED" and r["rehome"]["done"], r
    new_id = r["rehome"]["calibration_id"]
    assert r["rehome"]["shift_px"][0] == pytest.approx(2.0, abs=0.3)
    assert r["rehome"]["shift_px"][1] == pytest.approx(-0.8, abs=0.3)
    code, out, _ = _cli([*base, "--json", "calibrations"])
    cals = {c["id"]: c for c in json.loads(out)["calibrations"]}
    assert set(cals) == {cal0.id, new_id}
    assert cals[new_id]["rehomed"]["from"] == cal0.id and not cals[cal0.id].get("rehomed")
    code, out, _ = _cli([*base, "calibrations"])
    assert "re-homed (rotation only) from" in out
    code, out, _ = _cli([*base, "delete", new_id, "--yes", "--reason", "back to the original"])
    assert code == cli.EXIT_OK and "Moved calibration" in out
    s = api.configure(lib, mock=True)
    try:
        cur = stereo.latest(s.store, p.geometry_id, config.geometry_fingerprint(p))
        assert cur.id == cal0.id and cur.rehomed is None
        assert (lib / "trash" / "calibrations" / new_id).is_dir()
    finally:
        api._session = None

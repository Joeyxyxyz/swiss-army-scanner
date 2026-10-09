"""Recipes (work-state checks), MOCK only: the schema and its validator, every built-in check
PASS and FAIL on synthetic clouds, the check plugin registry, the runner's verdict and
INVALID rules, result.json zones, the projected feedback landing on the right surfaces of
the mock scene, the recipe library (built-in read-only, user add / select / delete to
trash), the command bus registers and the industrial I/O placeholders.

The runner tests check synthetic metric scans offline (tests/recipe_synth.py: the mock
stereo bench ray-cast, saved as the library saves a metric scan); test_recipe_run_mock.py
runs the full mock capture path (`scan run`, the operator page)."""

import copy
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

from engine import api, recipes
from engine.recipes import schema
from engine.recipes.checks.base import Check, CheckResult, Param
from engine.recipes.context import CheckContext, GoldenView
from engine.recipes import runner
from engine.recipes.geometry import ScanView, part_frame, resolve_zone
from engine.recipes.runner import aggregate

import recipe_synth as RS
from conftest import ROOT

PRESET = "cell_wall_2p7ft_vstripe"
RECIPES = ROOT / "recipes"


@pytest.fixture(autouse=True)
def _fresh_engine():
    yield
    api._session = None


# ================================================================== synthetic clouds

F, Z0, H, W = 1600.0, 500.0, 300, 400        # 0.31 mm / px on a wall 500 mm away


def view(h_fn, bright=None, noise=0.01, seed=0, corr=True) -> ScanView:
    """A wall at Z0 with heights h_fn(u, v) (mm toward the camera; NaN = not measured)
    seen by a pinhole camera; part-frame (u, v) = wall mm, decoded DMD = an affine of the
    camera px + parallax with height; flood: bright where bright(u, v, h)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    u, v = (xx - W / 2) / F * Z0, (yy - H / 2) / F * Z0
    h = np.asarray(h_fn(u, v), float) + np.zeros_like(u)
    valid = np.isfinite(h)
    hz = np.nan_to_num(h)
    xyz = np.stack([u, v, Z0 - hz + rng.normal(0, noise, u.shape)], -1)
    xyz[~valid] = np.nan
    K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1.0]])
    cx = np.where(valid, 0.5 * xx + 100 + 0.25 * hz, np.nan) if corr else None
    cy = np.where(valid, 1.0 * yy + 200, np.nan) if corr else None
    lit = (hz > 1.0) if bright is None else bright(u, v, hz)
    white = 40 + 170 * lit.astype(float) + rng.normal(0, 1.0, u.shape)
    return ScanView(xyz=xyz, valid=valid, K=K, dist=np.zeros(5), corr_x=cx, corr_y=cy,
                    flood_white=white.astype(np.float32),
                    flood_black=np.full(u.shape, 10.0, np.float32), scan_id="synthetic",
                    calibration_id="synthetic")


def box(u0, v0, u1, v1, h):
    return lambda u, v: np.where((u >= u0) & (u <= u1) & (v >= v0) & (v <= v1), h, 0.0)


def doc(zones, **top) -> dict:
    d = {"schema_version": 1, "name": "t", "preset": PRESET, "zones": zones,
         "feedback": {"enabled": True}}
    d.update(top)
    return d


def zone(zid, checks, coords=(-30, -20, 30, 20), shape="rect", units="part_mm", **kw):
    return {"id": zid, "shape": shape, "units": units, "coords": list(coords) if
            isinstance(coords, tuple) else coords, "checks": checks, **kw}


def run_zone(d: dict, sv: ScanView, golden: ScanView = None) -> dict:
    """Validate the recipe, resolve its zones on sv, run every check -> {zone: {check: res}}."""
    r = schema.validate(d)
    fr = part_frame(sv)
    views = {z.id: resolve_zone(z, sv, fr) for z in r.zones}
    g = None
    if golden is not None:
        gf = part_frame(golden)
        g = GoldenView(scan=golden, frame=gf, zones={z.id: resolve_zone(z, golden, gf)
                                                    for z in r.zones})
    out = {}
    for z in r.zones:
        ctx = CheckContext(scan=sv, frame=fr, zone=views[z.id], zones=views, golden=g)
        out[z.id] = {c.id: c.check.run(ctx) for c in z.checks}
    return out


def one(d, sv, golden=None) -> CheckResult:
    res = run_zone(d, sv, golden)
    (z,) = res.values()
    (c,) = z.values()
    return c


# ================================================================== schema

@pytest.mark.parametrize("path", sorted(RECIPES.glob("*.json")), ids=lambda p: p.stem)
def test_builtin_recipes_validate(path):
    r = schema.load(path)
    assert r.name == path.stem and r.schema_version == schema.SCHEMA_VERSION
    assert r.zones and all(z.checks for z in r.zones)
    assert r.part.get("mock_scene", "").startswith("scene:")
    assert r.preset in [p.stem for p in (ROOT / "presets").glob("*.json")]
    again = schema.validate(r.to_dict())            # the normalised form round-trips
    assert again.to_dict() == r.to_dict()


def test_yaml_recipe(tmp_path):
    yaml = pytest.importorskip("yaml")
    d = json.loads((RECIPES / "box_lid.json").read_text())
    f = tmp_path / "box_lid.yaml"
    f.write_text(yaml.safe_dump(d), encoding="utf-8")
    assert schema.load(f).to_dict() == schema.load(RECIPES / "box_lid.json").to_dict()


def test_bad_recipe_lists_every_problem():
    bad = doc([zone("a", [{"type": "flatness", "params": {"tol_mm": -1}}]),
               zone("a", [{"type": "warp_drive"}]),
               zone("c", [{"type": "height", "params": {"ref": "zone:nope", "band_mm": [0, 1]}}]),
               {"id": "d", "shape": "hexagon", "units": "part_mm", "coords": [], "checks": []},
               zone("e", [{"type": "presence", "params": {"min_height_mm": 1, "colour": "x"}}],
                    coords=(10, 10, 0, 0)),
               zone("f", [{"type": "presence"}], coords="F10:J12", shape="grid-cells",
                    units="part_mm"),
               zone("g", [{"type": "position", "params": {"tol_mm": 1}}]),
               zone("h", [], units="inches")],
              schema_version=7, colour="red")
    with pytest.raises(schema.RecipeError) as ei:
        schema.validate(bad)
    msg, probs = str(ei.value), ei.value.problems
    for needle in ("schema_version", "unknown top-level keys ['colour']", "tol_mm must be > 0",
                   "duplicate zone id", "no check type 'warp_drive'", "'nope'",
                   "shape 'hexagon'", "x1 > x0", "unknown params ['colour']",
                   "grid-cells are projector", "golden", "units 'inches'"):
        assert needle in msg, (needle, msg)
    assert len(probs) >= 11


@pytest.mark.parametrize("over, msg", [
    ({"scan": {"axis": "x"}}, "axis xy"),                       # feedback needs both axes
    ({"scan": {"kind": "teleport"}}, "scan.kind"),
    ({"scan": {"kind": "trigger"}}, "scan.pack"),
    ({"feedback": {"hold_s": 9999}}, "feedback.hold_s"),
    ({"feedback": {"fail_colour": "chartreuse"}}, "colour"),
    ({"outputs": {"result_codes": {"PASS": 1}}}, "result_codes"),
    ({"part": {"frame": {"reference": {"zone": "a"}}}}, "DEFINES the part frame"),
    ({"name": "bad name/x"}, "name"),
    ({"preset": None}, "preset"),
])
def test_bad_recipe_sections(over, msg):
    d = doc([zone("a", [{"type": "flatness", "params": {"tol_mm": 0.2}}])])
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            d[k] = {**d[k], **v}
        else:
            d[k] = v
    with pytest.raises(schema.RecipeError, match=re.escape(msg)):
        schema.validate(d)


def test_units_are_explicit_in_every_param():
    for cid, desc in recipes.available().items():
        for name, p in desc["params"].items():
            if p["kind"] in ("number", "pair") and not name.endswith(("_pct", "_z")) \
                    and name != "z_thresh":
                assert p["unit"], f"{cid}.{name} has no unit"
                assert name.endswith(f"_{p['unit']}") or p["unit"] in ("mm", "deg", "mm2"), name


# ================================================================== checks on synthetic clouds

def test_presence_pass_fail_absent():
    d = doc([zone("z", [{"type": "presence", "params": {"min_height_mm": 5}}])])
    assert one(d, view(box(-25, -15, 25, 15, 20))).verdict == "PASS"
    r = one(d, view(lambda u, v: 0.0))
    assert r.verdict == "FAIL" and r.values["part_pct"] == 0 and r.fault_points
    d2 = doc([zone("z", [{"type": "presence", "params": {"expect": "absent"}}])])
    assert one(d2, view(lambda u, v: 0.0)).verdict == "PASS"
    unseen = view(lambda u, v: np.where(np.hypot(u, v) < 60, np.nan, 0.0))
    r = one(doc([zone("z", [{"type": "presence"}], units="camera_px",
                      coords=(150, 100, 250, 200))]), unseen)
    assert r.verdict == "INVALID" and "unseen" in r.message


def test_position_vs_golden_pass_fail():
    d = doc([zone("z", [{"type": "position", "params": {"tol_mm": 1.0, "min_points": 50}}],
                  coords=(-40, -30, 40, 30))], golden={"scan": "g"})
    gold = view(box(-30, -8, 30, 8, 15))
    ok = one(d, view(box(-30, -8, 30, 8, 15), seed=1), gold)
    assert ok.verdict == "PASS" and ok.values["offset_mm"] < 0.2
    moved = one(d, view(box(-30, -5, 30, 11, 15), seed=2), gold)
    assert moved.verdict == "FAIL" and moved.values["dy_mm"] == pytest.approx(3.0, abs=0.3)
    assert moved.fault_points and moved.key == "offset_mm"
    no_gold = one(d, view(box(-30, -8, 30, 8, 15)))
    assert no_gold.verdict == "INVALID" and "golden" in no_gold.message


def test_height_and_step_between_zones():
    two = view(lambda u, v: box(-35, -10, -5, 10, 20)(u, v) + box(5, -10, 35, 10, 21)(u, v))
    d = doc([zone("a", [{"type": "height", "params": {"nominal_mm": 20, "tol_mm": 0.5,
                                                       "min_height_mm": 5}}],
                  coords=(-30, -8, -10, 8)),
             zone("b", [{"type": "height", "params": {"ref": "zone:a", "nominal_mm": 0,
                                                       "tol_mm": 0.3, "min_height_mm": 5}}],
                  coords=(10, -8, 30, 8))])
    res = run_zone(d, two)
    assert res["a"]["height"].verdict == "PASS"
    b = res["b"]["height"]
    assert b.verdict == "FAIL" and b.values["height_mm"] == pytest.approx(1.0, abs=0.05)
    with pytest.raises(schema.RecipeError, match="either band_mm"):
        schema.validate(doc([zone("a", [{"type": "height", "params": {"nominal_mm": 1}}])]))


def test_flatness_pass_fail():
    d = doc([zone("z", [{"type": "flatness", "params": {"tol_mm": 0.2, "min_height_mm": 5}}],
                  coords=(-25, -15, 25, 15))])
    flat = one(d, view(box(-30, -20, 30, 20, 10)))
    assert flat.verdict == "PASS" and flat.values["flatness_mm"] < 0.1
    bent = one(d, view(lambda u, v: box(-30, -20, 30, 20, 10)(u, v) + np.where(
        (abs(u) <= 30) & (abs(v) <= 20), 0.02 * u, 0)))
    assert bent.verdict == "PASS"                     # a tilted plane is still flat
    warped = one(d, view(lambda u, v: box(-30, -20, 30, 20, 10)(u, v) + np.where(
        (abs(u) <= 30) & (abs(v) <= 20), 0.0004 * u * u, 0)))
    assert warped.verdict == "FAIL" and warped.values["flatness_mm"] > 0.2 and warped.fault_points


def _bump(u0, v0, r, depth):
    return lambda u, v: np.where(np.hypot(u - u0, v - v0) < r,
                                 depth * 0.5 * (1 + np.cos(np.pi * np.hypot(u - u0, v - v0) / r)),
                                 0.0)


def test_defect_local_pass_fail():
    d = doc([zone("z", [{"type": "defect", "params": {"depth_mm": 0.3, "min_area_mm2": 2,
                                                       "mode": "local", "min_height_mm": 5}}],
                  coords=(-25, -15, 25, 15))])
    plate = box(-30, -20, 30, 20, 8)
    assert one(d, view(plate)).verdict == "PASS"
    r = one(d, view(lambda u, v: plate(u, v) - _bump(5, 3, 4, 0.6)(u, v)))
    assert r.verdict == "FAIL" and r.values["n_defects"] == 1
    assert r.details["defects"][0]["depth_mm"] == pytest.approx(0.6, abs=0.15)   # + = a dent
    x, y = r.fault_points[0]["px"]
    assert abs((x - W / 2) / F * Z0 - 5) < 1.0 and abs((y - H / 2) / F * Z0 - 3) < 1.0


def test_defect_face_mode_mfcfar():
    face = lambda u, v: np.where(np.hypot(u - 2, v) <= 10.5, 25.0, 0.0)  # noqa: E731
    d = doc([zone("f", [{"type": "defect", "params": {"depth_mm": 0.15, "mode": "face",
                                                       "sign": "dent"}}],
                  shape="circle", coords={"center": [2, 0], "radius": 10.5})])
    clean = one(d, view(face, noise=0.015))
    assert clean.verdict == "PASS", clean.message
    dented = one(d, view(lambda u, v: face(u, v) - _bump(5, -2, 1.5, 0.35)(u, v), noise=0.015))
    assert dented.verdict == "FAIL" and dented.values["max_depth_mm"] > 0.15, dented.message
    x, y = dented.fault_points[0]["px"]
    assert abs((x - W / 2) / F * Z0 - 5) < 1.0 and abs((y - H / 2) / F * Z0 + 2) < 1.0


def test_edge_straightness_and_position():
    plate = box(-40, -15, 40, 30, 6)
    d = doc([zone("e", [{"type": "edge", "params": {"side": "top", "straightness_tol_mm": 0.4,
                                                     "min_height_mm": 3}}],
                  coords=(-35, -25, 35, -5))])
    good = one(d, view(plate))
    assert good.verdict == "PASS" and good.values["top_straightness_mm"] < 0.3, good.values
    tab = lambda u, v: np.maximum(plate(u, v), box(0, -17.5, 8, -15, 6)(u, v))  # noqa: E731
    bad = one(d, view(tab))
    assert bad.verdict == "FAIL" and bad.values["top_straightness_mm"] > 1.5
    x, _ = bad.fault_points[0]["px"]
    assert 0 - 1 <= (x - W / 2) / F * Z0 <= 8 + 1
    dp = doc([zone("e", [{"type": "edge", "params": {"side": "top", "position_tol_mm": 1.0,
                                                      "min_height_mm": 3}}],
                   coords=(-35, -25, 35, -5))], golden={"scan": "g"})
    same = one(dp, view(plate, seed=3), view(plate))
    assert same.verdict == "PASS" and abs(same.values["top_offset_mm"]) < 0.2
    moved = one(dp, view(box(-40, -13, 40, 30, 6), seed=4), view(plate))
    assert moved.verdict == "FAIL" and moved.values["top_offset_mm"] == pytest.approx(2.0, abs=0.3)


def test_code_read_pass_fail_invalid():
    import cv2
    enc = cv2.QRCodeEncoder.create()
    qr = enc.encode("PART-42")
    qr = cv2.resize(qr, None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST).astype(np.float32)
    sv = view(lambda u, v: 0.0)
    q = np.pad(qr, 24, constant_values=255)
    sv.flood_white[40:40 + q.shape[0], 120:120 + q.shape[1]] = q
    z = (110, 30, 130 + q.shape[1], 50 + q.shape[0])
    ok = one(doc([zone("c", [{"type": "code", "params": {"expect": "PART-42", "kind": "QR"}}],
                       units="camera_px", coords=z)]), sv)
    assert ok.verdict == "PASS" and ok.details["codes"][0]["data"] == "PART-42"
    wrong = one(doc([zone("c", [{"type": "code", "params": {"pattern": "^LOT-"}}],
                          units="camera_px", coords=z)]), sv)
    assert wrong.verdict == "FAIL" and wrong.fault_points
    sv.flood_white = None
    assert one(doc([zone("c", [{"type": "code"}], units="camera_px", coords=z)]),
               sv).verdict == "INVALID"


def test_zone_units_camera_projector_grid():
    sv = view(box(-20, -10, 20, 10, 10))
    d = doc([zone("cam", [{"type": "presence"}], units="camera_px", coords=(160, 120, 240, 180)),
             zone("dmd", [{"type": "presence"}], units="projector_px",
                  coords=(180, 320, 220, 380)),
             zone("cells", [{"type": "presence", "params": {"min_area_pct": 1}}],
                  shape="grid-cells", units="projector_px", coords="C3@96")])
    r = schema.validate(d)
    fr = part_frame(sv)
    v = {z.id: resolve_zone(z, sv, fr) for z in r.zones}
    ys, xs = np.nonzero(v["dmd"].mask)                       # corr_x = 0.5 x + 100 + ..
    assert 150 <= xs.min() and xs.max() <= 245 and 118 <= ys.min() and ys.max() <= 182
    assert v["cam"].n_px == 81 * 61 and v["cells"].n_px > 0
    nox = view(box(-20, -10, 20, 10, 10), corr=False)
    with pytest.raises(ValueError, match="both DMD axes"):
        resolve_zone(r.zone("dmd"), nox, part_frame(nox))


def test_aggregate_rules():
    assert aggregate(["PASS", "PASS"]) == "PASS"
    assert aggregate(["PASS", "INVALID"]) == "INVALID"
    assert aggregate(["FAIL", "INVALID", "PASS"]) == "FAIL"
    assert aggregate([]) == "INVALID"


# ================================================================== plugins

DOCS_EXAMPLE_RE = re.compile(r"```python\n(.*?)```", re.S)


def test_docs_plugin_example_registers_and_runs():
    """docs/CHECKS.md's example check, executed as written, runs inside a recipe."""
    text = (ROOT / "docs" / "CHECKS.md").read_text(encoding="utf-8")
    code = DOCS_EXAMPLE_RE.search(text).group(1)
    assert len(code.strip().splitlines()) <= 30
    ns: dict = {}
    try:
        exec(compile(code, "CHECKS.md", "exec"), ns)       # noqa: S102 - our own docs
        d = doc([zone("z", [{"type": "max_height", "params": {"limit_mm": 12.0}}])])
        assert one(d, view(box(-10, -10, 10, 10, 10))).verdict == "PASS"
        bad = one(d, view(box(-10, -10, 10, 10, 15)))
        assert bad.verdict == "FAIL" and bad.fault_points
    finally:
        recipes.unregister("max_height")


def test_register_rules_and_entry_points(monkeypatch):
    class NotACheck:
        id = "x"
    with pytest.raises(TypeError):
        recipes.register(NotACheck)

    class Dup(Check):
        id = "flatness"
    with pytest.raises(ValueError, match="already registered"):
        recipes.register(Dup)

    class FromPlugin(Check):
        """A check shipped by another package."""
        id = "plugin_probe"
        params = {"x_mm": Param("number", 1.0, "mm")}

        def run(self, ctx):
            return CheckResult("PASS")

    class EP:
        name = "plugin_probe"

        def load(self):
            return FromPlugin

    from engine.recipes import checks as reg
    import importlib.metadata as md
    monkeypatch.setattr(md, "entry_points", lambda group=None: [EP()] if group ==
                        reg.ENTRY_POINT_GROUP else [])
    monkeypatch.setitem(reg._LOADED, "entry_points", False)
    try:
        assert "plugin_probe" in recipes.available()
        assert recipes.get("plugin_probe") is FromPlugin
    finally:
        recipes.unregister("plugin_probe")


def test_a_crashing_check_is_invalid_never_a_verdict(eng, tmp_path):
    class Boom(Check):
        id = "boom"

        def run(self, ctx):
            raise RuntimeError("kaboom")
    recipes.register(Boom)
    try:
        p = api.load_preset(PRESET)
        ms = RS.oracle_scan(eng, p, "scene:box_lid", "boom_scan")
        d = doc([zone("z", [{"type": "boom"}])], name="boom_recipe")
        out = api.recipe_run(d, scan=ms.id, project_feedback=False, runs_dir=tmp_path)
        res = out["result"]
        assert res["result"] == "INVALID" and out["exit_code"] == 5
        ch = res["zones"][0]["checks"][0]
        assert ch["verdict"] == "INVALID" and "kaboom" in ch["message"]
        assert "kaboom" in res["invalid_reason"]
    finally:
        recipes.unregister("boom")


# ================================================================== runner on synthetic scans

RESULT_KEYS = {"schema_version", "run_id", "run_folder", "mode", "station",
               "operator_confirmed", "job", "calibration", "capture", "part", "deviation_mm",
               "deviation_units", "threshold_mm", "result", "invalid_reason", "engine_verdict",
               "warnings", "outputs", "recipe", "zones", "feedback", "software", "started",
               "finished"}


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    """A mock library with the bench's true stereo calibration at bench_9in_metric and
    synthetic scans of the box lid (good, ajar) and the tube (golden, misaligned)."""
    root = tmp_path_factory.mktemp("recipes") / "library"
    s = api.configure(root, mock=True)
    try:
        p = api.load_preset("bench_9in_metric")
        scans = {k: RS.oracle_scan(s, p, f"scene:{k}", k.replace(":", "_"), seed=i).id
                 for i, k in enumerate(("box_lid", "box_lid:ajar", "tube:misaligned"))}
        g = RS.oracle_scan(s, p, "scene:tube", "ribbed_tube_alignment_golden", seed=9)
        runner.record_golden(s, api.recipe_load("ribbed_tube_alignment"), p, g)   # recipe-golden
        scans["box_lid:x"] = RS.oracle_scan(s, p, "scene:box_lid", "box_x_only", axis="x").id
    finally:
        api._session = None
    return root, scans


@pytest.fixture
def eng_lib(lib):
    s = api.configure(lib[0], mock=True)
    yield s, lib[1]
    api._session = None


def _run(name, scan, tmp_path, **kw):
    out = api.recipe_run(name, scan=scan, runs_dir=tmp_path / "runs", **kw)
    run_dir = Path(out["run_dir"])
    res = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    assert set(res) == RESULT_KEYS and res == json.loads(json.dumps(out["result"], default=str))
    events = [json.loads(x) for x in (run_dir / "run.log.jsonl").read_text().splitlines()]
    assert events[0]["event"] == "run_start" and events[-1]["event"] == "run_end"
    assert all(e["run_id"] == res["run_id"] for e in events)
    return out, res, run_dir, events


def test_box_lid_pass_and_ajar_fail(eng_lib, tmp_path):
    s, scans = eng_lib
    out, res, run_dir, ev = _run("box_lid", scans["box_lid"], tmp_path, project_feedback=False)
    assert res["result"] == "PASS" and out["exit_code"] == 0, res["invalid_reason"]
    assert res["schema_version"] == "0.2" and res["job"]["type"] == "recipe"
    assert res["recipe"]["name"] == "box_lid" and res["recipe"]["tier"] == "builtin"
    assert res["recipe"]["preset"] == "bench_9in_metric" and res["recipe"]["result_code"] == 1
    assert [z["id"] for z in res["zones"]] == ["lid_left", "lid_right"]
    for z in res["zones"]:
        assert z["verdict"] == "PASS" and {c["type"] for c in z["checks"]} == {
            "presence", "height", "flatness"}
        h = next(c for c in z["checks"] if c["type"] == "height")
        assert h["values"]["height_mm"] == pytest.approx(40.0, abs=0.2)
        assert h["units"]["height_mm"] == "mm"
    assert res["deviation_units"] == "mm" and res["calibration"]["same_geometry"] is True
    assert res["recipe"]["frame"]["reference"] == "dominant_plane"
    assert (run_dir / "overlay.png").is_file() and res["outputs"]["deviation_map"]
    assert res["feedback"]["rendered"] and (run_dir / "feedback_dmd.png").is_file()
    assert [e["event"] for e in ev].count("zone") == 2 and "feedback_skipped" in \
        [e["event"] for e in ev]

    out, res, run_dir, _ = _run("box_lid", scans["box_lid:ajar"], tmp_path, project_feedback=False)
    assert res["result"] == "FAIL" and out["exit_code"] == 1
    z = {z["id"]: z for z in res["zones"]}
    assert z["lid_left"]["verdict"] == "PASS" and z["lid_right"]["verdict"] == "FAIL"
    h = next(c for c in z["lid_right"]["checks"] if c["type"] == "height")
    assert h["verdict"] == "FAIL" and h["fault_points"]
    assert res["deviation_mm"]["method"].startswith("recipe lid_right/height")
    assert res["recipe"]["result_code"] == 2 and res["invalid_reason"] is None


def _no_text(name: str) -> dict:
    """A built-in recipe as a mapping with the feedback text off (red / green = zones only)."""
    d = json.loads((RECIPES / f"{name}.json").read_text(encoding="utf-8"))
    d["feedback"]["text"] = False
    return d


def _seen(scan_spec, img):
    """What each camera px of the mock scene sees of the DMD image (the TRUE projector
    correspondence of the ray-cast, not the scan's decode) -> (H, W, 3)."""
    from engine.capture import mock_scenes
    s = api._engine()
    g = RS.geometry(s.bench.rig_model(), mock_scenes.shapes(scan_spec))
    ok = g["lit"]
    r = np.clip(np.round(np.nan_to_num(g["v"])), 0, 1139).astype(int)
    c = np.clip(np.round(np.nan_to_num(g["u"])), 0, 911).astype(int)
    out = img[r, c]
    out[~ok] = 0
    return out


def _red(px):
    return (px[..., 2] > 120) & (px[..., 1] < 60)


def _green(px):
    return (px[..., 1] > 120) & (px[..., 2] < 60)


def test_feedback_lands_red_on_the_failing_zone(eng_lib, tmp_path):
    from engine.recipes import feedback as FB
    from engine.recipes.geometry import ScanView as SV, frame_for, resolve_zone as rz
    import cv2
    s, scans = eng_lib
    out, res, run_dir, _ = _run(_no_text("box_lid"), scans["box_lid:ajar"], tmp_path,
                                project_feedback=False)
    img = FB.read_png(run_dir / "feedback_dmd.png")
    seen = _seen("scene:box_lid:ajar", img)
    # the zones on the scan (camera px)
    r = api.recipe_load("box_lid")
    from engine.reconstruct import metric_scan
    ms = next(m for m in metric_scan.list_metric(s.store) if m.id == scans["box_lid:ajar"])
    from engine.calibrate import stereo
    cal = stereo.latest(s.store, ms.geometry_id, ms.meta.get("geometry_fp"))
    sv = SV.from_scan(ms, cal)
    fr = frame_for(r, sv)
    zr = rz(r.zone("lid_right"), sv, fr).mask
    zl = rz(r.zone("lid_left"), sv, fr).mask
    near_r = cv2.dilate(zr.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    near_l = cv2.dilate(zl.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    red, green = _red(seen), _green(seen)
    assert red.sum() > 1000
    fb = res["feedback"]
    assert fb["text"] is None
    (cr,) = fb["crosses"]                     # the cross arms may reach past the zone: not fill
    yy, xx = np.mgrid[0:red.shape[0], 0:red.shape[1]]
    arms = np.hypot(xx - cr["px"][0], yy - cr["px"][1]) < 70
    assert (red & near_r).sum() / max((red & ~arms).sum(), 1) > 0.98
    assert (red & zr).sum() / zr.sum() > 0.6                 # the failing zone is filled red
    assert not (red & zl).any() or (red & zl).sum() / zl.sum() < 0.02
    assert (green & near_l).sum() > 200 and not (green & zr).sum() > 0.05 * zr.sum()
    # the cross: its DMD centre is the TRUE correspondence of the fault px (+- 2 DMD px)
    from engine.capture import mock_scenes
    g = RS.geometry(s.bench.rig_model(), mock_scenes.shapes("scene:box_lid:ajar"))
    x, y = (int(round(v)) for v in cr["px"])
    assert abs(g["u"][y, x] - cr["dmd"][0]) < 2.0 and abs(g["v"][y, x] - cr["dmd"][1]) < 2.0
    assert all(z["drawn"] and z["local_p95_px"] < 2.0 for z in fb["zones"])


def test_feedback_pass_is_green_only_and_projects_in_mock(eng_lib, tmp_path):
    from engine.recipes import feedback as FB
    s, scans = eng_lib
    out, res, run_dir, ev = _run("box_lid", scans["box_lid"], tmp_path, hold_s=0.3)
    assert res["result"] == "PASS" and res["feedback"]["text"] == "PASS"
    seen = _seen("scene:box_lid", FB.read_png(run_dir / "feedback_dmd.png"))
    assert _green(seen).sum() > 500 and _red(seen).sum() == 0
    shown = [e for e in ev if e["event"] == "feedback_shown"]
    assert shown and shown[0]["held_s"] == pytest.approx(0.3) and not shown[0]["polarity_inverted"]
    assert (run_dir / "feedback_photo.png").is_file()
    again = api.recipe_feedback(out, hold_s=0.2)            # shown again, from the run folder
    assert again["verdict"] == "PASS" and again["held_s"] == pytest.approx(0.2)


def test_tube_misaligned_red_on_the_right_section(eng_lib, tmp_path):
    from engine.recipes import feedback as FB
    s, scans = eng_lib
    out, res, run_dir, _ = _run(_no_text("ribbed_tube_alignment"), scans["tube:misaligned"],
                                tmp_path, project_feedback=False)
    z = {z["id"]: z for z in res["zones"]}
    assert res["result"] == "FAIL"
    assert z["left"]["verdict"] == "PASS" and z["right"]["verdict"] == "FAIL", z
    pos = next(c for c in z["right"]["checks"] if c["type"] == "position")
    assert pos["values"]["dy_mm"] == pytest.approx(2.8, abs=0.4)
    seen = _seen("scene:tube:misaligned", FB.read_png(run_dir / "feedback_dmd.png"))
    red = _red(seen)
    cols = np.nonzero(red.any(0))[0]
    # red only over the right section (part-frame u 30..75 mm ~ camera x right of centre)
    assert red.sum() > 500 and np.percentile(cols, 2) > 720 + 25 * 2450 / 823


def test_offline_x_only_scan_refuses_feedback_not_the_verdict(eng_lib, tmp_path):
    s, scans = eng_lib
    out, res, run_dir, _ = _run("box_lid", scans["box_lid:x"], tmp_path)
    assert res["result"] == "PASS"
    fb = res["feedback"]
    assert not fb["rendered"] and "columns only" in fb["refused"]
    assert not (run_dir / "feedback_dmd.png").exists()


def test_golden_missing_and_uncalibrated_are_invalid(tmp_path):
    s = api.configure(tmp_path / "lib", mock=True)
    out = api.recipe_run("box_lid", runs_dir=tmp_path / "runs", project_feedback=False)
    res = out["result"]
    assert res["result"] == "INVALID" and out["exit_code"] == 5
    assert "stereo calibration" in res["invalid_reason"] and "calibrate-stereo" in res["invalid_reason"]
    assert res["part"]["scan_id"] is None and res["zones"] == []
    p = api.load_preset("bench_9in_metric")
    ms = RS.oracle_scan(s, p, "scene:tube", "t")
    out = api.recipe_run("ribbed_tube_alignment", scan=ms.id, runs_dir=tmp_path / "runs",
                         project_feedback=False)
    res = out["result"]
    assert res["result"] == "INVALID"
    assert "golden" in res["invalid_reason"] and "recipe-golden" in res["invalid_reason"]


def test_real_rig_without_confirmation_is_refused(tmp_path):
    api.configure(tmp_path / "lib", mock=False)
    out = api.recipe_run("box_lid", runs_dir=tmp_path / "runs")
    res = out["result"]
    assert res["mode"] == "REAL" and res["result"] == "INVALID" and out["exit_code"] == 2
    assert "confirm" in res["invalid_reason"]
    assert "PySpin" not in sys.modules and "hid" not in sys.modules


def test_real_rig_offline_recheck_of_a_stored_scan_needs_no_confirmation(tmp_path):
    """docs/PORTING.md: `recipe-run NAME --scan ID --no-feedback` re-checks a stored scan
    offline. On the real rig it was refused without --operator-confirmed (and with the flag
    and feedback on, the stored scan's verdict was painted on whatever part is there now).
    A stored scan with no feedback captures and projects nothing: no confirmation needed;
    feedback still needs it. Nothing opens the camera or the projector."""
    from engine import config as cfg
    from ui import cli
    s = api.configure(tmp_path / "lib", mock=True)              # build the scan offline
    p = cfg.load_preset("bench_9in_metric", presets_dir=s.presets_dir, store=s.store,
                        mock=False)                                # the REAL geometry id
    RS.truth_calibration(s, p)
    ms = RS.oracle_scan(s, p, "scene:box_lid", "stored_lid")
    api._session = None
    api.configure(tmp_path / "lib", mock=False)
    out = api.recipe_run("box_lid", scan=ms.id, project_feedback=False,
                         runs_dir=tmp_path / "runs")
    res = out["result"]
    assert res["mode"] == "REAL" and res["result"] == "PASS", res.get("invalid_reason")
    out = api.recipe_run("box_lid", scan=ms.id, project_feedback=True, runs_dir=tmp_path / "runs")
    assert out["exit_code"] == 2 and "confirm" in out["result"]["invalid_reason"]
    api._session = None
    argv = ["--library", str(tmp_path / "lib"), "--json", "recipe-run", "box_lid", "--scan",
            ms.id, "--runs-dir", str(tmp_path / "runs")]
    assert cli.main(argv + ["--no-feedback"]) == 0
    api._session = None
    assert cli.main(argv) == cli.EXIT_REFUSED                    # feedback on: confirm first
    api._session = None
    assert "PySpin" not in sys.modules and "hid" not in sys.modules


def test_mock_scene_specs():
    from engine.capture import mock_scenes
    assert mock_scenes.parse("scene:tube:misaligned") == ("tube", "misaligned")
    assert mock_scenes.parse("plate") == ("plate", "ok")
    for bad in ("scene:nope", "tube:bent"):
        with pytest.raises(ValueError):
            mock_scenes.parse(bad)
    assert mock_scenes.shapes("box_lid:missing") == []


# ================================================================== recipe library

def _builtin_bytes():
    return {p.name: p.read_bytes() for p in RECIPES.glob("*.json")}


def test_library_add_select_delete_trash(tmp_path):
    eng = api.configure(tmp_path / "lib", mock=True)       # the repo's presets (recipes pin them)
    before = _builtin_bytes()
    names = {i["name"]: i for i in api.recipe_list()}
    assert {"box_lid", "flat_plate", "cylinder_end_faces", "ribbed_tube_alignment"} <= set(names)
    assert all(names[n]["tier"] == "builtin" and names[n]["valid"] for n in names)
    r = api.recipe_add("my_lid", "box_lid", {"zones.0.checks.1.params.tol_mm": "2.5",
                                             "feedback.hold_s": 8},
                       description="lid, line 2")
    path = eng.store.root / "recipes" / "my_lid" / "recipe.json"
    assert r["path"] == path and path.is_file()
    meta = json.loads((path.parent / "meta.json").read_text())
    assert meta["from"] == "box_lid" and meta["description"] == "lid, line 2"
    got = api.recipe_load("my_lid")
    assert got.zones[0].checks[1].check.p["tol_mm"] == 2.5 and got.feedback["hold_s"] == 8
    show = api.recipe_show("my_lid")
    assert show["tier"] == "user" and show["recipe"]["name"] == "my_lid"
    with pytest.raises(schema.RecipeError, match="built-in"):
        api.recipe_add("box_lid", "box_lid")
    with pytest.raises(schema.RecipeError, match="exists"):
        api.recipe_add("my_lid", "box_lid")
    with pytest.raises(schema.RecipeError, match="tol_mm"):
        api.recipe_add("bad", "box_lid", {"zones.0.checks.1.params.tol_mm": -3})
    with pytest.raises(schema.RecipeError, match="does not exist"):
        api.recipe_add("bad", "box_lid", {"preset": "no_such_preset"})
    assert "bad" not in [i["name"] for i in api.recipe_list()]
    rep = api.recipe_add("my_lid", "box_lid", replace=True)
    assert rep["replaced"] and Path(rep["replaced"]).is_dir()
    api.recipe_select("my_lid")
    assert api.recipe_active() == "my_lid"
    with pytest.raises(schema.RecipeError, match="selected"):
        api.recipe_delete("my_lid", "test")
    api.recipe_select("box_lid")
    st = tmp_path / "stations"
    st.mkdir()
    (st / "s.json").write_text(json.dumps({"station_id": "s", "job": {"type": "recipe",
                                                                      "recipe": "my_lid"}}))
    with pytest.raises(schema.RecipeError, match="s.json"):
        api.recipe_delete("my_lid", "test", stations_dir=st)
    with pytest.raises(schema.RecipeError, match="built-in"):
        api.recipe_delete("box_lid", "test")
    out = api.recipe_delete("my_lid", "retired", force=True, stations_dir=st)
    assert out["forced"] and (eng.store.root / "trash" / "recipes" / "my_lid").is_dir()
    assert "my_lid" not in [i["name"] for i in api.recipe_list()]
    assert _builtin_bytes() == before                       # built-ins never written


def test_validate_reports_problems_without_running(tmp_path):
    api.configure(tmp_path / "lib", mock=True)
    v = api.recipe_validate(json.loads((RECIPES / "flat_plate.json").read_text()))
    assert v["valid"] and v["recipe"]["name"] == "flat_plate"
    d = json.loads((RECIPES / "flat_plate.json").read_text())
    d["zones"][0]["checks"][0]["params"]["tol_mm"] = "thin"
    v = api.recipe_validate(d)
    assert not v["valid"] and any("tol_mm must be a number" in pr for pr in v["problems"])


# ================================================================== command bus + I/O

def test_bus_registers_follow_the_recipe_codes():
    from engine.control import commands as C
    bus = C.CommandBus(C.Handlers(), ["a.json", "b.json"])
    out = C.SimulatedOutput()
    bus.outputs.append(out)
    bus.workstate_info["a.json"] = {"recipe": "box_lid",
                                    "result_codes": {"PASS": 10, "FAIL": 20, "INVALID": 30,
                                                     "ABORTED": 40}}
    bus.command("START")
    assert out.registers == {"result_code": 0, "state_code": C.STATES.index("RUNNING"),
                             "workstate": 1}
    bus.job_finished("run", "FAIL")
    assert out.current["FAIL"] and out.registers["result_code"] == 20
    assert bus.snapshot()["workstate_recipe"] == "box_lid"
    bus.command("SELECT_WORKSTATE", arg="b.json")
    bus.command("START")
    bus.job_finished("run", "PASS")
    assert out.registers == {"result_code": 1, "state_code": 0, "workstate": 2}


@pytest.mark.parametrize("cls_name, proto", [("ModbusTcp", "Modbus TCP"), ("OpcUa", "OPC UA"),
                                             ("EtherNetIp", "EtherNet/IP")])
def test_industrial_io_placeholders_refuse_clearly(cls_name, proto):
    from engine.control import commands as C
    inp = getattr(C, f"{cls_name}Input")({"x": "START"})
    with pytest.raises(NotImplementedError) as e:
        inp.start(C.CommandBus())
    assert "placeholder" in str(e.value) and proto in str(e.value) and "START" in str(e.value)
    o = getattr(C, f"{cls_name}Output")({"PASS": "y"})
    for call in (lambda: o.publish({"PASS": True}), lambda: o.publish_registers({})):
        with pytest.raises(NotImplementedError) as e:
            call()
        assert proto in str(e.value) and "Intended mapping" in str(e.value)
    assert C.INPUTS[inp.name] is type(inp) and C.OUTPUTS[o.name] is type(o)
    bus = C.CommandBus(C.Handlers(), ["a.json"])
    bus.outputs.append(o)
    bus.command("START")                                   # an output error never stops the bus
    assert bus.state == "RUNNING" and any(h[2] == "OUTPUT" for h in bus.history)


def test_io_example_lists_the_fieldbus_placeholders_disabled():
    from engine.control import commands as C
    doc_ = json.loads((ROOT / "stations" / "io_example.json").read_text())
    kinds = {e["kind"] for e in doc_["inputs"] + doc_["outputs"]}
    assert {"modbus_tcp", "opc_ua", "ethernet_ip"} <= kinds
    ins, outs = C.load_io_doc(doc_)
    assert [type(i).__name__ for i in ins] == ["SimulatedInput"]
    assert [type(o).__name__ for o in outs] == ["SimulatedOutput"]
    on = copy.deepcopy(doc_)
    for e in on["outputs"]:
        e["enabled"] = e["kind"] == "modbus_tcp"
    _, outs = C.load_io_doc(on)
    assert isinstance(outs[0], C.ModbusTcpOutput) and outs[0].options["result_register"] == 101


def test_hardware_interfaces_are_satisfied():
    from engine.capture import interfaces as I
    from engine.capture.camera import MockCamera, RealCamera
    from engine.capture.projector import HDMIWindow, MockWindow
    from engine.control import commands as C
    for cls in (MockWindow, HDMIWindow):
        assert all(callable(getattr(cls, m, None)) for m in ("show", "hold", "close"))
    assert isinstance(MockWindow(), I.DisplayWindow)
    for cls in (MockCamera, RealCamera):
        assert all(callable(getattr(cls, m, None)) for m in ("set_exposure_us", "flush", "grab",
                                                              "close"))
    assert isinstance(C.SimulatedInput({}), I.InputDriver)
    assert isinstance(C.SimulatedOutput({}), I.OutputDriver)
    assert isinstance(C.ModbusTcpOutput({}), I.OutputDriver)

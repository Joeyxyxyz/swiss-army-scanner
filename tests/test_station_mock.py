"""Station measurement (engine/measure/station), the HDMI fast zone scan (scan_fast) and the end-face
locate with a near neighbour, through the verbs.

Synthetic station scans (tests/station_synth.py: two cells with their end faces toward the camera
and a ribbed tube (CT: a thin ribbon in a trapezoid wave, its front edge a few mm behind the
faces) ray-cast by the mock bench's camera + projector, written as the library writes a metric
scan, with a flood pair) cover station_measure (cell heights, Cj-Ci, spacing, CT band, wave,
zones T1..Tn vs the truth), station_golden + station_check (PASS on a repeat, FAIL naming the
moved cell pair and the raised CT zones, INCOMPLETE when zones were not measured, never a PASS
without a flood-visible face), the rules (limits, calibration binding), CLI exit codes and the
web jobs. locate_face finds a face whose neighbour (a thin ribbon) is only 3 mm behind it (the
neighbour split) and, when the depth rim fails, from the flood outline. scan_fast runs on the
mock bench: columns only + floods, per-run average / hold overrides (average >= 2), its flood
pair kept, refused without a stereo calibration. The real rig is never opened."""

import http.client
import json
import shutil
import threading
import time
from pathlib import Path

import numpy as np
import pytest

import face_synth as S
import station_synth as SS
from conftest import FIXTURE_PRESETS
from engine import api, config, rules
from engine.capture.mock_bench import MockBench
from engine.capture.mock_stereo import Block, EndFace, MetricCylinder, Plane
from engine.measure.endface import locate as locate_mod
from engine.measure.station import geometry as geo_mod
from engine.reconstruct import fast_scan as fast_mod

S_RANGE = (-60.0, 70.0)
BUMP = (10.0, 40.0, 0.6)        # the CT front edge raised 0.6 mm over this stretch of the run


# ------------------------------------------------------------ synthetic station scans

@pytest.fixture(scope="module")
def station(tmp_path_factory):
    lib = tmp_path_factory.mktemp("station") / "library"
    lib.mkdir()
    rg = S.rig()
    cal = S.write_calibration(lib, rg)
    scans = {}
    scans["good1"], sc = SS.write_station(lib, rg, cal, "20260101T040001_000000_good1", seed=1,
                                          s_range=S_RANGE)
    scans["good2"], _ = SS.write_station(lib, rg, cal, "20260101T040002_000000_good2", seed=2,
                                         s_range=S_RANGE)
    scans["bad"], _ = SS.write_station(lib, rg, cal, "20260101T040003_000000_bad", seed=3,
                                       s_range=S_RANGE, c2_minus_c1=0.15, bump=BUMP)
    scans["cut"], _ = SS.write_station(lib, rg, cal, "20260101T040004_000000_cut", seed=4,
                                       s_range=(-12.0, 70.0))
    scans["noflood"], _ = SS.write_station(lib, rg, cal, "20260101T040005_000000_noflood", seed=5,
                                           s_range=S_RANGE, flood=False)
    s = api.configure(lib, mock=True)
    ctx = {"lib": lib, "rg": rg, "cal": cal, "scans": scans, "scene": sc, "s": s}
    ctx["m1"] = api.station_measure(scans["good1"], label="good1")
    ctx["m2"] = api.station_measure(scans["good2"], label="good2")
    ctx["golden"] = api.station_golden([ctx["m1"], ctx["m2"]], "good12",
                                       cell_height_tol_mm=0.15, ct_zone_tol_mm=0.3)
    yield ctx
    api._session = None


def _use(ctx):
    api._session = ctx["s"]


def test_station_measure_recovers_the_synthetic_truth(station):
    _use(station)
    m, tr = station["m1"], station["scene"]["truth"]
    assert [c["name"] for c in m.cells] == ["C1", "C2"]
    assert m.cells[0]["camera_px"][1] < m.cells[1]["camera_px"][1]         # C1 = the upper cell
    p = m.pairs[0]
    assert p["pair"] == "C2-C1"
    assert p["height_diff_mm"] == pytest.approx(tr["C2-C1"], abs=0.02)
    assert p["spacing_inplane_mm"] == pytest.approx(tr["spacing"], abs=0.1)
    assert all(c["flat_rms_mm"] < 0.05 and c["coverage"] > 0.9 for c in m.cells)
    assert m.cells[0]["height_mm"] == pytest.approx(-m.cells[1]["height_mm"], abs=1e-6)
    ct = m.ct
    assert ct["band_source"] == "auto" and ct["band_mm"][0] < tr["h_ct"] < ct["band_mm"][1]
    assert ct["trap_fit"]["pitch_mm"] == pytest.approx(tr["pitch"], abs=0.3)
    assert ct["trap_fit"]["amp_mm"] == pytest.approx(tr["amp"], abs=0.3)
    inc = [z for z in m.zones if z.get("included")]
    assert len(inc) >= 8
    assert [z["label"] for z in inc] == [f"T{i + 1}" for i in range(len(inc))]
    assert [z["k"] for z in inc] == sorted(z["k"] for z in inc)             # left -> right
    for z in inc:
        assert z["median_mm"] == pytest.approx(tr["h_ct"], abs=0.1), z
        assert z["coverage"] >= 0.8
    assert m.flood_source == "scan" and not m.warnings
    # outputs: JSON + CSV + PNG in the output bin, listed in the index
    for k in ("json", "csv", "png"):
        assert Path(m.files[k]).is_file() and Path(m.files[k]).parent.name == "measurements"
    rep = json.loads(Path(m.files["json"]).read_text())
    assert rep["kind"] == "station_measurement" and rep["calibration_id"] == station["cal"].id
    assert "toward the camera" in rep["units"]["height_mm"]
    assert "T1" in Path(m.files["csv"]).read_text()
    outs = {Path(o["file"]).name for o in api.list_library("outputs", "measurements")}
    assert Path(m.files["png"]).name in outs


def test_station_golden_then_check_pass_and_fail(station):
    _use(station)
    g = station["golden"]
    assert g.n_measurements == 2 and g.geometry_id == S.GEO and g.calibration_id == station["cal"].id
    assert (g.path / "golden.json").is_file() and (g.path / "meta.json").is_file()
    assert g.path.parent.parent.name == "station_golden"
    assert [x.id for x in api.list_library("station_golden")] == [g.id]
    assert g.zone_model and g.nominal["pairs"][0]["pair"] == "C2-C1"
    ok = api.station_check(station["scans"]["good2"], label="rep")
    assert ok.verdict == "PASS" and ok.n_triggers == 0 and ok.golden_id == g.id, ok.triggers
    assert {it["kind"] for it in ok.items} >= {"cell_height", "ct_zone"}
    bad = api.station_check(station["scans"]["bad"], g, label="bad")
    assert bad.verdict == "FAIL"
    trig = {t["item"] for t in bad.triggers}
    assert "C2-C1" in trig
    ct_trig = [t for t in bad.triggers if t["kind"] == "ct_zone"]
    assert ct_trig and all(t["value_mm"] - t["nominal_mm"] > 0.3 for t in ct_trig)
    # exactly the zones over the raised stretch of the run
    zs = {z["label"]: z for z in g.nominal["zones"]}
    for t in ct_trig:
        z = zs[t["item"]]
        assert z["s_hi_mm"] > BUMP[0] - 2 and z["s_lo_mm"] < BUMP[1] + 2, t
    far = [lab for lab, z in zs.items() if z["s_hi_mm"] < BUMP[0] - 12 or z["s_lo_mm"] > BUMP[1] + 12]
    assert far and not (set(far) & trig)
    for k in ("json", "csv", "png"):
        assert Path(bad.files[k]).is_file() and Path(bad.files[k]).parent.name == "defects"
    rep = json.loads(Path(bad.files["json"]).read_text())
    assert rep["verdict"] == "FAIL" and rep["n_triggers"] == len(bad.triggers)


def test_station_check_incomplete_and_never_a_blind_pass(station):
    _use(station)
    r = api.station_check(station["scans"]["cut"], station["golden"], label="cut")
    assert r.verdict == "INCOMPLETE" and r.n_triggers == 0
    missing = [it["item"] for it in r.items if it["status"] == "missing"]
    assert missing and any("not measured" in w for w in r.warnings)
    # no flood pair and a flat decode modulation: no face disc -> refused, never measured
    with pytest.raises(geo_mod.StationError, match="face disc"):
        api.station_measure(station["scans"]["noflood"])
    with pytest.raises(geo_mod.StationError):
        api.station_check(station["scans"]["noflood"], station["golden"])
    # ... but with another scan's flood pair (the cells did not move) it is measured, warned
    m = api.station_measure(station["scans"]["noflood"], station["scans"]["good1"], "borrowed")
    assert m.flood_source == f"scan:{station['scans']['good1']}"
    assert any("another scan" in w for w in m.warnings)


def test_station_limits_bands_and_rules(station, tmp_path):
    _use(station)
    m1 = station["m1"]
    with pytest.raises(ValueError, match="lo < hi"):
        api.station_golden([m1], "x", cell_height_band_mm=[0.2, -0.2])
    with pytest.raises(ValueError, match="> 0"):
        api.station_golden([m1], "x", ct_zone_tol_mm=0)
    with pytest.raises(ValueError, match="station_golden needs"):
        api.station_golden([], "x")
    # a band for an item the golden does not have (a typo) would silently fall back to the
    # tolerance; a tolerance and an absolute band that do not overlap would FAIL every part
    with pytest.raises(ValueError, match="unknown item"):
        api.station_golden([m1], "x", bands={"C2C1": [-1.0, 1.0]})
    with pytest.raises(ValueError, match="do not overlap"):
        api.station_golden([m1], "x", cell_height_band_mm=[5.0, 6.0])
    # flood= / params= that would be silently ignored are refused
    with pytest.raises(ValueError, match="single metric scan"):
        api.station_golden([station["scans"]["good1"], station["scans"]["good2"]], "x",
                           flood=station["scans"]["good1"])
    with pytest.raises(ValueError, match="params= applies"):
        api.station_golden([m1], "x", params={"zone_skip": [1, 0]})
    # an absolute per-item band decides alone for that item. In a copy of the library: the
    # FAIL it gives good2 makes good2 a part judged REJECT there - never a golden again
    # (engine/badparts.py) - and the module's good2 is a golden's measurement elsewhere
    zl = [z["label"] for z in m1.zones if z.get("included")]
    api.configure(Path(shutil.copytree(station["lib"], tmp_path / "banded")), mock=True)
    try:
        g = api.station_golden([m1], "banded", bands={zl[0]: [-4.4, -4.0],
                                                      "C2-C1": [-0.3, -0.1]})
        r = api.station_check(station["scans"]["good2"], g)
        assert r.verdict == "FAIL" and [t["item"] for t in r.triggers] == [zl[0]]
        assert next(it for it in r.items if it["item"] == "C2-C1")["band_mm"] == [-0.3, -0.1]
        # goldens are append-only: a second one is a new entry, the first is unchanged
        assert len(api.list_library("station_golden")) == 2
    finally:
        _use(station)
    # rule 6: a newer stereo calibration makes the scans (and the golden) stale
    lib = Path(shutil.copytree(station["lib"], tmp_path / "library"))
    s2 = api.configure(lib, mock=True)
    try:
        S.write_calibration(lib, station["rg"], name="20260201T000000_000000_stereo")
        with pytest.raises(rules.GeometryMismatch):
            api.station_measure(station["scans"]["good1"])
        with pytest.raises(rules.GeometryMismatch):
            api.station_check(station["scans"]["good1"])
    finally:
        _use(station)
    assert s2 is not station["s"]


def test_station_one_cell_and_a_face_at_the_zone_edge(station):
    """n_cells=1 uses the same frame as n >= 2 (e1 ~ image right: T1 is the leftmost zone); a
    face cut by the zone edge (its flood dark past the cut, as scan_fast lights only the zone)
    keeps its height but is warned about (its centre / spacing moves)."""
    _use(station)
    m = api.station_measure(station["scans"]["good1"], label="one_cell", n_cells=1)
    assert len(m.cells) == 1 and not m.pairs and m.frame["e1"][0] > 0
    inc = [z for z in m.zones if z.get("included")]
    assert len(inc) >= 8 and inc[0]["label"] == "T1"
    us = [z["cam_uv"][0] for z in inc]
    assert us == sorted(us)                                               # left -> right
    # C2 cut 40 % by the zone edge: cloud + flood dark below the cut
    m1 = station["m1"]
    (cx, cy), r = m1.cells[1]["camera_px"], m1.cells[1]["r_px"]
    src = station["lib"] / "metric" / S.GEO / station["scans"]["good1"]
    name = "20260101T040009_000000_edgecut"
    d = src.parent / name
    d.mkdir()
    ycut = int(cy + 0.2 * r)
    with np.load(src / "cloud.npz") as z:
        xyz, msk = z["xyz"].copy(), z["mask"].copy()
    msk[ycut:] = False
    xyz[~msk] = np.nan
    with np.load(src / "flood.npz") as z:
        w, b = z["white"].copy(), z["black"].copy()
    w[ycut:] = b[ycut:]
    shutil.copy2(src / "field.npz", d / "field.npz")
    np.savez_compressed(d / "cloud.npz", xyz=xyz, mask=msk)
    np.savez_compressed(d / "flood.npz", white=w, black=b)
    meta = json.loads((src / "meta.json").read_text())
    meta.update(id=name, label=name)
    (d / "meta.json").write_text(json.dumps(meta))
    mc = api.station_measure(name, label="edgecut")
    assert mc.pairs[0]["height_diff_mm"] == pytest.approx(m1.pairs[0]["height_diff_mm"], abs=0.02)
    assert any(w_.startswith("C2:") and "spacing is less certain" in w_ for w_ in mc.warnings)


def test_station_geometry_units():
    """zone numbering is part-fixed; the thin-run finder keeps only thin vertical runs."""
    assert list(geo_mod.zone_index(np.array([0.0, 5.0, 11.0, 17.0]), 23.5, 0.0)) == [-1, -1, 0, 0]
    m = np.zeros((20, 3), bool)
    m[2:5, 0] = True                     # 3 px: thin
    m[2:19, 1] = True                    # 17 px: too long
    m[[3, 7, 8], 2] = True               # 1 px (too short) + 2 px
    t = geo_mod._thin_runs(m, 0, 3, (2, 16))
    assert t[2:5, 0].all() and not t[:, 1].any() and t[7:9, 2].all() and not t[3, 2]
    with pytest.raises(ValueError, match="unknown station parameter"):
        geo_mod.StationParams.from_dict({"nope": 1})
    # list parameters: refused by name, never a crash deep inside or a silent 'no CT'
    for bad, what in (({"trap_pitch_range_mm": [40, 15]}, "trap_pitch_range_mm"),
                      ({"ct_band_mm": [-1.0, -8.0]}, "ct_band_mm"),
                      ({"zone_skip": [1]}, "zone_skip"), ({"zone_skip": [-1, 0]}, "zone_skip"),
                      ({"ct_roi": [10, 10, 5, 50]}, "ct_roi"), ({"n_cells": 0}, "n_cells"),
                      ({"r_px_range": [30, 20]}, "r_px_range")):
        with pytest.raises(ValueError, match=what):
            geo_mod.StationParams.from_dict(bad)
    d = geo_mod.flood_diff(np.stack([np.zeros((4, 4)), np.full((4, 4), 50.0)]))
    assert d.max() == 50.0                                             # inverted pair: brighter first


def test_station_cli_exit_codes(station, capsys):
    from ui import cli
    _use(station)
    lib = str(station["lib"])
    rc = cli.main(["--mock", "--library", lib, "--json", "station", station["scans"]["good2"],
                   "--label", "cli_st"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0 and out["station"]["pairs"][0]["pair"] == "C2-C1"
    rc = cli.main(["--mock", "--library", lib, "--json", "station-check", station["scans"]["good2"],
                   "--golden", "good12"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0 and out["check"]["verdict"] == "PASS"
    rc = cli.main(["--mock", "--library", lib, "--json", "station-check", station["scans"]["bad"],
                   "--golden", "good12"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 1 and out["check"]["verdict"] == "FAIL" and out["check"]["n_triggers"] >= 2
    rc = cli.main(["--mock", "--library", lib, "--json", "station-golden", station["scans"]["good1"],
                   "--label", "cli_gold", "--band", "C2-C1=-0.5,0.1", "--ct-tol", "0.25"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0 and out["station_golden"]["limits"]["bands"]["C2-C1"] == [-0.5, 0.1]
    rc = cli.main(["--mock", "--library", lib, "--json", "station-golden", station["scans"]["good1"],
                   "--band", "C2-C1"])
    assert rc == 2
    capsys.readouterr()
    # station parameters for metric-scan items (the golden's numbering: a lead-in zone skipped)
    rc = cli.main(["--mock", "--library", lib, "--json", "station-golden", station["scans"]["good1"],
                   "--label", "cli_skip", "--skip-zones", "1,0", "--edge-band", "-8,-1"])
    out = json.loads(capsys.readouterr().out)
    p = out["station_golden"]["params"]
    assert rc == 0 and p["zone_skip"] == [1, 0] and p["ct_band_mm"] == [-8.0, -1.0]
    assert cli.build_parser().parse_args(["station-check", "x"]).unexpected_exit == cli.EXIT_ERROR
    # CT heights are negative: "--band -8,-1" / "--ct-band -5,-4" must not read as an option
    ap = cli.build_parser()
    a = ap.parse_args(cli._join_negative_lists(["station", "x", "--band", "-8,-1"]))
    assert a.band == "-8,-1"
    a = ap.parse_args(cli._join_negative_lists(["station-golden", "x", "--ct-band", "-5.5,-4",
                                                "--cell-band", "-.4,0.1"]))
    assert a.ct_band == "-5.5,-4" and a.cell_band == "-.4,0.1"
    rc = cli.main(["--mock", "--library", lib, "stations"])
    assert rc == 0 and "cli_gold" in capsys.readouterr().out
    api._session = None
    _use(station)


def test_station_web_jobs(station, tmp_path):
    from ui.web import server as ui_server
    lib = Path(shutil.copytree(station["lib"], tmp_path / "library"))
    presets = tmp_path / "presets"
    shutil.copytree(FIXTURE_PRESETS, presets)
    httpd, app = ui_server.make_server(mock=True, library=lib, presets_dir=presets, port=0)
    t = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
    t.start()

    def job(verb, args):
        conn = http.client.HTTPConnection("127.0.0.1", app.port, timeout=300)
        conn.request("POST", "/api/jobs", body=json.dumps({"verb": verb, "args": args}),
                     headers={"Content-Type": "application/json"})
        j = json.loads(conn.getresponse().read())
        conn.close()
        t0 = time.monotonic()
        while j.get("status") in ("queued", "running"):
            assert time.monotonic() - t0 < 240, j
            time.sleep(0.2)
            conn = http.client.HTTPConnection("127.0.0.1", app.port, timeout=60)
            conn.request("GET", f"/api/jobs/{j['id']}")
            j = json.loads(conn.getresponse().read())
            conn.close()
        return j

    sc = station["scans"]
    try:
        j = job("station_measure", {"metric_id": sc["good1"], "label": "web_st"})
        assert j["status"] == "done", j.get("error")
        assert j["result"]["pairs"][0]["pair"] == "C2-C1"
        j = job("station_golden", {"metric_ids": [sc["good1"], sc["good2"]], "label": "webgold",
                                   "ct_zone_tol_mm": 0.3})
        assert j["status"] == "done", j.get("error")
        gid = j["result"]["id"]
        j = job("station_check", {"metric_id": sc["bad"], "golden_id": gid})
        assert j["status"] == "done", j.get("error")
        assert j["result"]["verdict"] == "FAIL"
        j = job("station_check", {"metric_id": sc["good2"], "golden_id": gid})
        assert j["status"] == "done" and j["result"]["verdict"] == "PASS", j.get("error")
        j = job("station_measure", {"metric_id": sc["noflood"], "flood_metric_id": sc["good1"]})
        assert j["status"] == "done", j.get("error")
        j = job("station_measure", {"metric_id": sc["good1"], "ct_band_mm": [1]})
        assert j["status"] == "error" and "ct_band_mm" in j["error"]
    finally:
        httpd.shutdown()
        httpd.server_close()
        app.close()
        _use(station)


# ------------------------------------------------------------ locate_face with a near neighbour

def _crop_cloud(rg, objs, c, margin_mm=34.0):
    box = S.crop_box(rg, c, margin_mm=margin_mm)
    X, uv, ok, idx = S.render(rg, objs, box)
    x0, y0, x1, y1 = box
    xyz = np.full(S.CAM + (3,), np.nan)
    m = np.zeros(S.CAM, bool)
    xyz[y0:y1, x0:x1], m[y0:y1, x0:x1] = np.where(ok[..., None], X, np.nan), ok
    cx = np.full(S.CAM, np.nan)
    cy = np.full(S.CAM, np.nan)
    cx[y0:y1, x0:x1] = np.where(ok, uv[..., 0], np.nan)
    cy[y0:y1, x0:x1] = np.where(ok, uv[..., 1], np.nan)
    return xyz, m, cx, cy, box, ok, idx


def test_locate_face_with_a_thin_ribbon_3mm_behind():
    """The station case: a ribbed tube only 3 mm behind the face, touching its rim in the
    image, merges with the face in depth (near band 12 mm). The old locate (no split) misfits
    that merged surface and then takes a sliver of it for the face (centre mm off) or finds
    none; the plane split (sequential RANSAC planes of the merged surface, rim = surface >=
    near_gap_mm behind the face plane) finds the face."""
    rg = S.rig()
    c, n = np.array((-6.0, 4.0, 780.0)), S.normal_from_tilt(20, 8)
    e1 = np.cross(n, [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(e1, n)
    ribbon_c = c + (S.RADIUS + SS.CT_T / 2 + 0.1) * e2 - (3.0 + 10.0) * n   # front edge 3 mm back
    rib = Block(center=tuple(ribbon_c), half=(60.0, SS.CT_T / 2, 10.0), rot=np.stack([e1, e2, n]))
    objs = S.shapes(c, n) + [rib]
    xyz, m, cx, cy, box, ok, _ = _crop_cloud(rg, objs, c)
    K, dist = rg.camera.K, rg.camera.dist
    try:                                                          # the old behaviour
        old = locate_mod.locate(xyz, m, cx, cy, m, K, dist, S.DIAMETER,
                                prm=locate_mod.LocateParams(neighbour_split=False,
                                                            flood_fallback=False))
        assert np.linalg.norm(old["centre_mm"] - c) > 2.0
    except locate_mod.FaceNotFound:
        pass
    r = locate_mod.locate(xyz, m, cx, cy, m, K, dist, S.DIAMETER)
    assert r["rim_source"] == "depth (plane split)"
    assert np.linalg.norm(r["centre_mm"] - c) < 0.15, r["centre_mm"]
    assert np.degrees(np.arccos(min(1.0, abs(r["normal"] @ n)))) < 0.5
    assert r["rim_arc_deg"] > 150 and r["rim_rms_mm"] < 0.3
    # the ribbon side counts as rim only when it is near_gap_mm behind the face plane
    r5 = locate_mod.locate(xyz, m, cx, cy, m, K, dist, S.DIAMETER,
                           prm=locate_mod.LocateParams(near_gap_mm=5.0))
    assert r5["n_rim"] < r["n_rim"]


def test_locate_face_flood_outline_fallback(tmp_path):
    """No depth rim at all (a plate 1.5 mm behind the face all around it): the face outline in
    the flood (the scan's own flood pair) locates it; without a flood it is not found."""
    lib = tmp_path / "library"
    lib.mkdir()
    rg = S.rig()
    cal = S.write_calibration(lib, rg)
    c, n = np.array((-6.0, 4.0, 780.0)), S.normal_from_tilt(12, 6)
    e1 = np.cross(n, [0.0, 1.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(e1, n)
    plate = Block(center=tuple(c - (1.5 + 1.0) * n), half=(40.0, 40.0, 1.0), rot=np.stack([e1, e2, n]))
    objs = [EndFace(center=tuple(c), normal=tuple(n), radius=S.RADIUS), plate, Plane()]
    xyz, m, cx, cy, box, ok, idx = _crop_cloud(rg, objs, c)
    K, dist = rg.camera.K, rg.camera.dist
    with pytest.raises(locate_mod.FaceNotFound):
        locate_mod.locate(xyz, m, cx, cy, m, K, dist, S.DIAMETER)
    with pytest.raises(ValueError, match="differ in size"):          # never a misaligned outline
        locate_mod.locate(xyz, m, cx, cy, m, K, dist, S.DIAMETER, flood=np.zeros((10, 10)))
    x0, y0, x1, y1 = box
    white = np.zeros(S.CAM, np.float32)
    white[y0:y1, x0:x1] = np.where(ok & (idx == 0), 220.0, np.where(ok, 90.0, 8.0))
    import cv2
    white = cv2.GaussianBlur(white, (0, 0), 0.8)
    r = locate_mod.locate(xyz, m, cx, cy, m, K, dist, S.DIAMETER, flood=white)
    assert r["rim_source"] == "flood outline" and any("flood" in w for w in r["warnings"])
    assert np.linalg.norm(r["centre_mm"] - c) < 0.3, r["centre_mm"]
    assert abs(r["flood_outline_diameter_mm"] - S.DIAMETER) < 0.5
    # through the verb: the scan's own flood pair is used by default
    name = "20260101T050000_000000_plate"
    d = lib / "metric" / S.GEO / name
    d.mkdir(parents=True)
    np.savez_compressed(d / "cloud.npz", xyz=xyz.astype(np.float32), mask=m)
    hdr = json.dumps({"geometry_id": S.GEO, "basis_names": ["multifreq"], "meta": {}})
    np.savez_compressed(d / "field.npz", corr_x=cx.astype(np.float32), corr_y=cy.astype(np.float32),
                        mask=m, conf=m.astype(np.float32), _header=np.array(hdr))
    np.savez_compressed(d / "flood.npz", white=white, black=np.zeros_like(white))
    (d / "meta.json").write_text(json.dumps({"id": name, "label": name, "geometry_id": S.GEO,
                                             "geometry_fp": dict(S.FP), "calibration_id": cal.id,
                                             "method": "plane", "units": "mm",
                                             "created": "2026-01-01T00:00:00"}))
    api.configure(lib, mock=True)
    try:
        t = api.locate_face(name, S.DIAMETER, "plate_face")
        assert t.meta["rim_source"] == "flood outline" and t.meta["flood_source"] == "scan"
        assert np.linalg.norm(np.asarray(t.centre_mm) - c) < 0.3
        with pytest.raises(ValueError, match="unknown params"):
            api.locate_face(name, S.DIAMETER, params={"far_gapp_mm": 3})
    finally:
        api._session = None


# ------------------------------------------------------------ scan_fast on the mock bench

@pytest.fixture
def mock_rig(tmp_path):
    pd = tmp_path / "presets"
    pd.mkdir()
    d = json.loads((FIXTURE_PRESETS / "cell_wall_2p7ft_vstripe_fullrange.json").read_text())
    d.update(name="fast_rig")
    d["camera"].update(average=1, exposure_us=16667)
    d["projection"].update(invert=False, duty=0.6)
    (pd / "fast_rig.json").write_text(json.dumps(d))
    s = api.configure(tmp_path / "library", mock=True, bench=MockBench(seed=7), presets_dir=pd)
    p = api.load_preset("fast_rig")
    bench = api.mock_bench()
    bench.use_stereo()
    yield {"s": s, "p": p, "bench": bench, "pd": pd}
    api._session = None


def _truth_calibration(s, p, bench):
    from engine.calibrate.stereo import StereoCalibration
    tr = bench.stereo_truth()
    dd = s.store.new_dir("calibrations", p.geometry_id, label="stereo")
    cal = StereoCalibration(id=f"{p.geometry_id}/{dd.name}", geometry_id=p.geometry_id,
                            K_cam=tr["K_cam"], dist_cam=tr["dist_cam"], K_proj=tr["K_proj"],
                            dist_proj=np.zeros(5), R=tr["R"], t=np.ravel(tr["t"]),
                            diamond_parity=tr["parity"], fingerprint=config.geometry_fingerprint(p),
                            path=dd, created="2026-01-01T00:00:00")
    s.store.write_json(dd / "stereo_calibration.json", cal.to_json())
    return cal


def test_scan_fast_columns_only_with_run_overrides(mock_rig, monkeypatch):
    s, p, bench = mock_rig["s"], mock_rig["p"], mock_rig["bench"]
    c, n = np.array((-8.0, 6.0, 780.0)), S.normal_from_tilt(10.4, 7.0)
    bench.place(EndFace(center=tuple(c), normal=tuple(n), radius=S.RADIUS))
    bench.place(MetricCylinder(center=tuple(c - n * 15), radius=S.RADIUS, height=30, axis=tuple(n)))
    with pytest.raises(rules.GeometryMismatch):                    # rule 6, before the rig
        api.scan_fast(None, p)
    _truth_calibration(s, p, bench)
    seen = {}
    from engine.calibrate import flow
    real = flow.capture_patterns

    def spy(sess, preset, patterns, what, zone_mask=None):
        seen.update(average=preset.camera.average, hold=preset.projection.min_hold_s,
                    n=len(patterns), zone=None if zone_mask is None else int(zone_mask.sum()))
        return real(sess, preset, patterns, what, zone_mask=zone_mask)

    monkeypatch.setattr(flow, "capture_patterns", spy)
    before = (p.camera.average, p.projection.min_hold_s,
              json.loads((mock_rig["pd"] / "fast_rig.json").read_text()))
    t0 = time.perf_counter()
    ms = api.scan_fast(None, p, "fast1", average=1, min_hold_s=0.05)
    assert time.perf_counter() - t0 < 120
    assert seen == {"average": 2, "hold": 0.05, "n": 14, "zone": None}
    assert (p.camera.average, p.projection.min_hold_s,                    # the preset untouched
            json.loads((mock_rig["pd"] / "fast_rig.json").read_text())) == before
    m = ms.meta
    assert m["fast"] and m["average"] == 2 and m["n_patterns"] == 14 and m["method"] == "plane"
    assert any("flicker" in w for w in m["warnings"])
    assert ms.field.corr_y is None and ms.field.corr_x is not None
    with np.load(ms.path / "flood.npz") as z:
        assert z["white"].mean() > z["black"].mean()
    # the face depth: the fitted plane at the true centre
    X = ms.cloud.xyz[ms.cloud.mask]
    near = X[np.linalg.norm(X - c, axis=1) < S.RADIUS - 1.5]
    assert len(near) > 200
    h = (near - c) @ n
    assert abs(np.median(h)) < 0.3
    # a DMD mask zone lights only that part of the DMD
    mask = np.zeros(fast_mod.DMD_SHAPE, bool)
    mask[300:800, 200:700] = True
    npy = mock_rig["pd"].parent / "zone.npy"
    np.save(npy, mask)
    ms2 = api.scan_fast(str(npy), p, "fast2", average=2)
    assert seen["zone"] == int(mask.sum()) and ms2.meta["zone"].startswith("dmd mask")
    assert ms2.meta["n_points"] < m["n_points"]
    # grid cells (zone text, through zones.resolve: no projector map here, so lit but not cut)
    from engine.capture import zones as zones_mod
    pm, cam, _ = zones_mod.resolve(s, p, "F10:J12")
    ms3 = api.scan_fast("F10:J12", p, "fast3")
    assert seen["zone"] == int(np.asarray(pm, bool).sum()) and ms3.meta["zone"] == "F10:J12"
    assert cam is None and 0 < ms3.meta["n_points"] < m["n_points"]
    with pytest.raises(ValueError, match="1140 rows"):
        api.scan_fast(np.zeros((10, 10), bool), p)
    with pytest.raises(ValueError, match="axis"):
        api.scan_fast(None, p, axis="y")
    with pytest.raises(ValueError, match="average"):
        api.scan_fast(None, p, average=0)
    # the fast scan feeds station_measure's flood (its own flood pair)
    assert (ms.path / "flood.npz").is_file()


def test_scan_fast_cli_and_web_are_wired():
    from ui import cli
    from ui.web import server as ui_server
    ap = cli.build_parser()
    a = ap.parse_args(["scan-fast", "st", "--zone", "F10:J12", "--hold", "0.12"])
    assert a.func is cli.cmd_scan_fast and a.steps == 4 and a.hold == 0.12
    for v in ("scan_fast", "station_measure", "station_golden", "station_check"):
        assert v in ui_server.VERBS and callable(getattr(ui_server.App, f"_v_{v}", None))
    assert "scan_fast" not in ui_server.RIG_FREE_VERBS
    assert {"station_measure", "station_golden", "station_check"} <= ui_server.RIG_FREE_VERBS
    assert "station_golden" in ui_server.LIST_KINDS and "station_golden" in ui_server.DELETABLE

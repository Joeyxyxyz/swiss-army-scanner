"""CLI (ui/cli.py) end to end in mock mode: commands through main().

Verbs other agents are still building (profile, bank-*, tune, gamma, map,
grid, spectral, learn, classify, light) are only exercised through
monkeypatched stand-ins, so these tests do not depend on their progress.
"""

import ast
import contextlib
import io
import os
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from engine import api
from engine.results import SignalReport
from ui import cli

from conftest import PRESET, ROOT

FAST = ["--mock", "--average", "1"]        # speed: decode quality is not the subject
V_DIMS = "106.95,56.5,21.9"                 # §11 validation object, H,W,L mm


@pytest.fixture(autouse=True)
def _fresh_engine():
    yield
    api._session = None


def cli_run(capsys, lib, *argv):
    """main() on `lib` -> (exit code, stdout, stderr). Never a raw traceback."""
    code = cli.main([*FAST, "--library", str(lib), *argv])
    out, err = capsys.readouterr()
    if "--debug" not in argv:
        assert "Traceback" not in out + err, err
    return code, out, err


@pytest.fixture(scope="module")
def shared(tmp_path_factory):
    """One library with a base and a validation-object scan, made through the CLI."""
    lib = tmp_path_factory.mktemp("cli") / "library"
    printed = {}
    for name, argv in (("base", ["base"]), ("scan", ["scan", "v", "--known", V_DIMS])):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = cli.main([*FAST, "--library", str(lib), *argv])
        assert code == 0, buf.getvalue()
        printed[name] = buf.getvalue()
    api._session = None
    return lib, printed


def copy_of(shared, tmp_path):
    """A private copy for tests that change the library (calibrate, delete)."""
    return Path(shutil.copytree(shared[0], tmp_path / "library"))


def paths_in(text: str, suffix: str) -> list:
    return [Path(m) for m in re.findall(r"(\S+\\output\\\S+" + re.escape(suffix) + r")", text)]


def answer(monkeypatch, answers: dict, log: list):
    """input() stand-in: reply by the first keyword found in the prompt
    (a callable reply is called, for sequences)."""
    def fake(prompt=""):
        log.append(prompt)
        for key, reply in answers.items():
            if key in prompt:
                return reply() if callable(reply) else reply
        return ""
    monkeypatch.setattr("builtins.input", fake)


BAD_SIGNAL = SignalReport(gc_conf=3.0, valid_pct=0.4, contrast_mean=2.0, focus_ratio=0.1,
                          focus_hint="blurred", quality="unusable", ok=False,
                          geometry_id="mock-wall_2p7ft_vstripe_v1", mock=True,
                          warnings=["no lit pixels - is the projector on?"], coded_axis="",
                          axis_ok=None)


# ------------------------------------------------------------ listing

def test_presets_lists_the_default(tmp_path, capsys):
    code, out, _ = cli_run(capsys, tmp_path / "lib", "presets")
    assert code == 0
    active = [l for l in out.splitlines() if l.strip().startswith("*")]
    assert len(active) == 1 and PRESET in active[0]
    assert "mock-wall_2p7ft_vstripe_v1" in active[0] and f"{json.load(open(ROOT / 'presets' / 'cell_wall_2p7ft_vstripe.json'))['camera']['exposure_us']} us" in active[0]


def test_paths_shows_library_and_output_bin(tmp_path, capsys):
    code, out, _ = cli_run(capsys, tmp_path / "lib", "paths")
    assert code == 0
    assert str((tmp_path / "lib").resolve()) in out
    assert "output_index" in out and "index.csv" in out and "trash" in out


def test_help_and_unknown_command(tmp_path, capsys):
    assert cli.main(["--help"]) == 0
    assert "zone" in capsys.readouterr().out
    assert cli.main(["--mock", "frobnicate"]) == 2


# ------------------------------------------------------------ signal

def test_signal_ok(tmp_path, capsys):
    code, out, _ = cli_run(capsys, tmp_path / "lib", "signal")
    assert code == 0
    assert "[MOCK]" in out and "ok to scan" in out
    assert re.search(r"gc_conf\s+[\d.]+ counts", out)
    assert re.search(r"valid\s+[\d.]+ %", out)
    assert "x (ok for this rig)" in out


def test_signal_not_ok_exits_1(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(api, "check_signal", lambda preset=None: BAD_SIGNAL)
    code, out, _ = cli_run(capsys, tmp_path / "lib", "signal")
    assert code == 1 and "NOT ok" in out and "no lit pixels" in out


# ------------------------------------------------------------ base / scan

def test_base_and_scan_report_ids_units_and_output_bin(shared):
    _, printed = shared
    assert "[mock] simulated bench cleared" in printed["base"]
    assert re.search(r"Base \S+_empty", printed["base"])
    s = printed["scan"]
    assert "placed the validation object" in s
    assert re.search(r"Scan \S+_v\s", s) and "H106.95 W56.5 L21.9 mm" in s
    assert "px (rule 6)" in s                             # uncalibrated: said, not hidden
    assert "Output bin (1 new)" in s
    pngs = paths_in(s, ".png")
    assert pngs and all(p.exists() for p in pngs)


def test_bases_marks_the_default(shared, capsys):
    code, out, _ = cli_run(capsys, shared[0], "bases")
    assert code == 0
    row = [l for l in out.splitlines() if "_empty" in l]
    assert len(row) == 1 and "default for scan" in row[0]


def test_scans_lists_the_scan(shared, capsys):
    code, out, _ = cli_run(capsys, shared[0], "scans")
    assert code == 0 and "Scans (1)" in out
    assert "H106.95 W56.5 L21.9 mm" in out


def test_measure_uncalibrated_is_px_with_rule6_warning(shared, capsys):
    code, out, _ = cli_run(capsys, shared[0], "measure", "v")
    assert code == 0
    h = re.search(r"^\s*H\s+([\d.]+)\s+px\s+106.95 mm\s+n/a", out, re.M)
    assert h and float(h.group(1)) == pytest.approx(106.95 * 0.45, abs=0.5)   # raw relief px
    assert "rule 6" in out and "units px" in out
    jsons = paths_in(out, ".json")
    assert len(jsons) == 1 and jsons[0].exists() and "measurements" in str(jsons[0])


def test_reconstruct_writes_pointcloud_with_units(shared, capsys):
    code, out, _ = cli_run(capsys, shared[0], "reconstruct", "v", "--fmt", "xyz")
    assert code == 0 and "units xy=px z=px" in out
    files = paths_in(out, ".xyz")
    assert files and files[0].exists() and "pointclouds" in str(files[0])


def test_outputs_lists_every_file(shared, capsys):
    code, out, _ = cli_run(capsys, shared[0], "outputs")
    assert code == 0
    listed = paths_in(out, ".png") + paths_in(out, ".json") + paths_in(out, ".xyz") + \
        paths_in(out, ".ply")
    assert listed and all(p.exists() for p in listed)
    code, out, _ = cli_run(capsys, shared[0], "outputs", "--kind", "previews")
    kinds = {l.split()[1] for l in out.splitlines() if l.startswith("  20")}
    assert code == 0 and kinds == {"previews"}


def test_json_output_for_agents(shared, capsys):
    code, out, _ = cli_run(capsys, shared[0], "--json", "scans")
    doc = json.loads(out)
    assert code == 0 and doc["ok"] and doc["mock"] is True
    assert [s["label"] for s in doc["scans"]] == ["v"]
    assert doc["scans"][0]["known_dims"] == {"H": 106.95, "W": 56.5, "L": 21.9}
    code, out, _ = cli_run(capsys, shared[0], "measure", "v", "--json")   # flag after the command
    doc = json.loads(out)
    assert doc["dimensions"]["units"] == "px" and doc["dimensions"]["tier"] == "relief"
    assert [o["kind"] for o in doc["outputs_written"]] == ["measurements"]
    code, out, err = cli_run(capsys, shared[0], "--json", "measure", "nope")
    assert code == 2 and json.loads(out)["error"]["type"] == "CLIError"


def test_compare_uncalibrated_needs_a_px_band(shared, capsys):
    code, out, _ = cli_run(capsys, shared[0], "compare", "v", "v")
    assert code == 1 and "INDETERMINATE" in out and "--fail-px" in out
    code, out, _ = cli_run(capsys, shared[0], "compare", "v", "latest", "--fail-px", "1")
    assert code == 0 and "PASS" in out and "1.0000 px" in out


def test_calibrate_then_measure_and_inspect_in_mm(shared, tmp_path, capsys):
    lib = copy_of(shared, tmp_path)
    code, out, _ = cli_run(capsys, lib, "calibrate", "v")
    assert code == 0 and "mm/px" in out and "camera px per mm" in out
    code, out, _ = cli_run(capsys, lib, "measure", "v")
    h = re.search(r"^\s*H\s+([\d.]+)\s+mm\s", out, re.M)
    assert code == 0 and h and float(h.group(1)) == pytest.approx(106.95, abs=0.7)
    assert "units mm" in out and "rule 6" not in out
    code, out, _ = cli_run(capsys, lib, "inspect", "v")
    assert code in (0, 1) and "mm (depth_calibrated tier)" in out


# ------------------------------------------------------------ library edits

def test_delete_asks_then_moves_to_trash(shared, tmp_path, capsys, monkeypatch):
    lib = copy_of(shared, tmp_path)
    prompts = []
    answer(monkeypatch, {"trash": "n"}, prompts)
    # no --yes, no --interactive: refused at once, never a question (nothing waits for input)
    code, out, err = cli_run(capsys, lib, "delete", "v")
    assert code == 2 and "--yes" in err and "does not wait for input" in err and prompts == []
    assert list((lib / "objects").iterdir())                          # still there
    code, out, _ = cli_run(capsys, lib, "--interactive", "delete", "v")
    assert code == 1 and "Cancelled" in out and len(prompts) == 1
    assert list((lib / "objects").iterdir())                          # still there
    code, out, _ = cli_run(capsys, lib, "delete", "v", "--yes", "--reason", "test")
    assert code == 0 and "Moved scan" in out and len(prompts) == 1    # --yes: no prompt
    assert not list((lib / "objects").iterdir())
    trashed = list((lib / "trash" / "objects").iterdir())
    assert len(trashed) == 1 and (trashed[0] / "TRASHED.json").exists()
    code, out, _ = cli_run(capsys, lib, "scans")
    assert "Scans (0)" in out


def test_reject_base_then_scan_refuses_it(shared, tmp_path, capsys):
    lib = copy_of(shared, tmp_path)
    code, out, _ = cli_run(capsys, lib, "reject-base", "empty", "bumped", "the", "tripod")
    assert code == 0 and "REJECTED (bumped the tripod)" in out
    code, out, _ = cli_run(capsys, lib, "bases")
    assert "REJECTED: bumped the tripod" in out
    code, _, err = cli_run(capsys, lib, "scan", "x", "--base", "empty")
    assert code == 2 and "REJECTED" in err
    code, _, err = cli_run(capsys, lib, "scan", "x")          # default base: none usable
    assert code == 2 and "no usable base" in err


# ------------------------------------------------------------ errors

def test_unknown_base_is_one_clear_line_exit_2(shared, capsys):
    code, out, err = cli_run(capsys, shared[0], "scan", "x", "--base", "nope")
    assert code == 2
    assert err.strip().count("\n") == 0 and "no base 'nope'" in err
    code, _, err = cli_run(capsys, shared[0], "--debug", "scan", "x", "--base", "nope")
    assert code == 2 and "Traceback" in err                  # only with --debug


def test_geometry_mismatch_is_refused(shared, capsys):
    """Rule 5: the other preset is another geometry; the base can't be used."""
    code, _, err = cli_run(capsys, shared[0], "--preset", "cell_wall_2p7ft",
                           "scan", "x", "--base", "empty")
    assert code == 2 and err.startswith("GeometryMismatch:")


def test_bad_input_exit_2(shared, capsys):
    for argv, msg in ((["scan", "x", "--known", "1,2,3,4"], "at most 3 values"),
                      (["scan", "x", "--known", "H=-1"], "positive"),
                      (["measure", "nope"], "no scan 'nope'"),
                      (["--preset", "nosuch", "signal"], "ConfigError"),
                      (["delete", "latest", "--yes"], "explicit id"),
                      (["profile", "v", "--p0", "1,2"], "--p1")):
        code, _, err = cli_run(capsys, shared[0], *argv)
        assert code == 2 and msg in err, (argv, err)


def test_not_implemented_modes_exit_3(shared, capsys, monkeypatch):
    def pending(*a, **k):
        raise NotImplementedError("pending implementation")
    monkeypatch.setattr(api, "profile", pending)
    code, _, err = cli_run(capsys, shared[0], "profile", "v")
    assert code == 3 and "the 2D profile is not available yet" in err
    monkeypatch.setattr(api, "project_grid", pending)
    code, _, err = cli_run(capsys, shared[0], "grid")
    assert code == 3 and "the projected grid is not available yet" in err

    seen = {}

    def capture_object(preset, base, label, known_dims=None, zone=None):
        seen.update(label=label, zone=zone, known=known_dims)
        raise NotImplementedError("zones.resolve: pending implementation")
    monkeypatch.setattr(api, "capture_object", capture_object)
    code, _, err = cli_run(capsys, shared[0], "scan", "z", "--zone", "C4:D5", "--known", "10")
    assert code == 3 and err.startswith("zone scanning is not available yet")
    assert seen == {"label": "z", "zone": "C4:D5", "known": {"H": 10.0}}


def test_grid_tells_the_operator_how_to_scan_a_zone(tmp_path, capsys, monkeypatch):
    from engine.capture.zones import GridSpec, ZoneShot
    cols, rows = [chr(65 + i) for i in range(9)], [str(i + 1) for i in range(11)]
    photo = tmp_path / "grid.png"

    def project_grid(preset=None, spacing_px=96, label="grid", hold_s=0.0):
        assert (spacing_px, hold_s) == (64, 2.0)
        return ZoneShot(kind="grid", grid=GridSpec(spacing_px=64, n_cols=9, n_rows=11,
                                                   col_labels=cols, row_labels=rows),
                        zones=[], camera_png=photo, overlay_png=None, dmd_png=None,
                        cells_visible=["C4", "C5", "D4", "D5"])
    monkeypatch.setattr(api, "project_grid", project_grid)
    code, out, _ = cli_run(capsys, tmp_path / "lib", "grid", "--spacing", "64", "--hold", "2")
    assert code == 0
    assert "9 columns (A-I) x 11 rows (1-11), 64 DMD px per cell" in out
    assert str(photo) in out and "C4, C5, D4, D5" in out
    assert "--zone C4:D5" in out and "run `map`" in out
    # the same, machine-readable: what an AI agent reads to pick the zone
    code, out, _ = cli_run(capsys, tmp_path / "lib", "--json", "grid", "--spacing", "64",
                           "--hold", "2")
    doc = json.loads(out)
    assert code == 0 and doc["grid"]["camera_png"] == str(photo)
    assert doc["grid"]["grid"]["col_labels"][0] == "A" and doc["grid"]["cells_visible"][0] == "C4"
    assert "scan <label> --zone C4:D5" in doc["next"]["then"]


# ------------------------------------------------------------ virtual sensors

SPOT = "dmd:456,570"          # mock bench: lands at camera (720, 540), base-plane (0, 0)


def test_where_and_mock_part_parsing():
    assert cli.parse_at("H8") == "H8" and cli.parse_at(" g7:h9@64 ") == "g7:h9@64"
    assert cli.parse_at("dmd:456,570") == {"dmd": [456.0, 570.0]}
    assert cli.parse_at("cam 700 500") == {"camera": [700.0, 500.0]}
    assert cli.parse_at("camera=(12.5, 3)") == {"camera": [12.5, 3.0]}
    for bad in ("dmd:1,2,3", "cam:x,y", ""):
        with pytest.raises(cli.CLIError):
            cli.parse_at(bad)
    assert cli._mock_part("10,40,40,0.2") == "10,40,40,0.2"
    assert cli._mock_part("10x40x40") == "10,40,40"
    for bad in ("10,40,40,1.5", "10,40", "0,40,40"):
        with pytest.raises(Exception, match="albedo"):
            cli._mock_part(bad)


def test_spot_size_nominal_vs_measured(tmp_path, capsys):
    lib = tmp_path / "lib"
    code, out, _ = cli_run(capsys, lib, "spot-size", "6.5")
    assert code == 0 and "nominal scale" in out
    assert "10.08 x 20.17 DMD px (cols x rows)" in out and "twice as many rows" in out
    code, out, _ = cli_run(capsys, lib, "--json", "spot-size", "6.5", "--measured", "0.25")
    doc = json.loads(out)
    assert code == 0 and doc["used"] == "measured"
    assert (doc["measured"]["diameter_dmd_px"], doc["measured"]["diameter_dmd_rows"]) == (26.0, 52.0)
    assert doc["nominal"]["source"] == "nominal" and doc["nominal"]["mm_per_dmd_px"] > 0.25
    code, _, err = cli_run(capsys, lib, "spot-size", "0")
    assert code == 2 and "diameter_mm" in err


def test_spot_projects_and_measures(tmp_path, capsys):
    lib = tmp_path / "lib"
    code, out, _ = cli_run(capsys, lib, "spot", SPOT, "--diameter", "6.5", "--label", "s1")
    assert code == 0 and "Spot FOUND at dmd:456,570" in out
    assert re.search(r"centroid\s+720\.0, 540\.0 camera px", out)
    assert "px only, rule 6" in out and "Output bin (3 new)" in out
    code, out, _ = cli_run(capsys, lib, "--json", "spot", "H8@96", "--diameter", "4")
    doc = json.loads(out)
    assert code == 0 and doc["spot"]["found"] and doc["spot"]["target_dmd"] == [719.5, 719.5]
    assert {o["kind"] for o in doc["outputs_written"]} == {"sensors"}
    code, _, err = cli_run(capsys, lib, "spot", "G7:H9", "--diameter", "6.5")
    assert code == 2 and "zone" in err                     # a spot is one cell
    code, _, err = cli_run(capsys, lib, "spot", "dmd:1,2,3", "--diameter", "6.5")
    assert code == 2 and "dmd:COL,ROW" in err


def test_presence_sensor_needs_the_empty_field_confirmed(tmp_path, capsys, monkeypatch):
    """Clearing the field is human-only: the operator's answer is an ARGUMENT
    (--empty-confirmed / --field-empty). Without it the command is refused at once - it
    never waits for an answer (it once sat on its y/N question for two days) - and only
    --interactive asks; a 'no' (or --json, which can't ask) teaches nothing."""
    lib = tmp_path / "lib"
    prompts = []
    answer(monkeypatch, {"EMPTY": "n"}, prompts)
    teach = ["teach-presence", "pe", SPOT, "--diameter", "6.5"]
    code, out, err = cli_run(capsys, lib, *teach)
    assert code == 2 and prompts == [] and "--empty-confirmed" in err
    assert "does not wait for input" in err and "Nothing was projected or taught" in err
    assert "[mock] simulated bench cleared" not in out and not (lib / "sensors").exists()
    code, out, _ = cli_run(capsys, lib, "--interactive", *teach)
    assert code == 1 and len(prompts) == 1 and "Not taught" in out
    code, out, _ = cli_run(capsys, lib, "--json", *teach)
    assert code == 2 and "--empty-confirmed" in json.loads(out)["error"]["message"]
    code, out, _ = cli_run(capsys, lib, *teach, "--field-empty", "no")
    assert code == 1 and "Not taught" in out
    assert len(prompts) == 1 and not (lib / "sensors").exists()
    answer(monkeypatch, {"EMPTY": "y"}, prompts)
    code, out, _ = cli_run(capsys, lib, "--interactive", *teach)
    assert code == 0 and len(prompts) == 2 and "Sensor 'pe' taught" in out
    assert "|change| > 25 % of the empty reference" in out
    code, out, _ = cli_run(capsys, lib, "teach-presence", "pe2", SPOT, "--diameter", "6.5",
                           "--field-empty", "yes")
    assert code == 0 and len(prompts) == 2 and "Sensor 'pe2' taught" in out
    code, out, err = cli_run(capsys, lib, *teach, "--empty-confirmed", "--field-empty", "no")
    assert code == 2 and "contradict" in err


def test_presence_sensor_exit_codes_and_trash(tmp_path, capsys):
    """Teach on the empty mock bench, a darker box -> PRESENT (exit 0); the empty
    bench -> ABSENT (exit 1). Listed by `sensors`, trashed by `delete`."""
    lib = tmp_path / "lib"
    code, out, _ = cli_run(capsys, lib, "teach-presence", "pe", SPOT, "--diameter", "6.5",
                           "--empty-confirmed")
    assert code == 0 and "[mock] simulated bench cleared" in out
    code, out, _ = cli_run(capsys, lib, "--mock-part", "10,40,40,0.2", "check-presence", "pe")
    assert code == 0 and "Sensor 'pe' (photoelectric): PRESENT" in out
    assert "albedo 0.2" in out and re.search(r"change\s+-6\d\.\d %", out)
    code, out, _ = cli_run(capsys, lib, "--mock-part", "none", "--json", "check-presence", "pe")
    doc = json.loads(out)
    assert code == 1 and not doc["ok"] and doc["presence"]["verdict"] == "ABSENT"
    assert doc["presence"]["units"]["signal"].startswith("camera counts")
    code, out, _ = cli_run(capsys, lib, "check-presence", "nope")
    assert code == 2
    code, out, _ = cli_run(capsys, lib, "--json", "sensors")
    doc = json.loads(out)
    assert code == 0 and [s["name"] for s in doc["sensors"]] == ["pe"]
    assert doc["sensors"][0]["kind"] == "presence"
    code, out, _ = cli_run(capsys, lib, "delete", "pe", "--yes", "--reason", "test")
    assert code == 0 and "Moved sensor mock-wall_2p7ft_vstripe_v1/pe to trash" in out
    assert (lib / "trash" / "sensors" / "mock-wall_2p7ft_vstripe_v1" / "pe").is_dir()
    code, out, _ = cli_run(capsys, lib, "sensors")
    assert code == 0 and "Sensors (0)" in out


def test_alignment_exit_codes(shared, tmp_path, capsys):
    lib = copy_of(shared, tmp_path)
    code, out, _ = cli_run(capsys, lib, "teach-align", "fixture", "v")
    assert code == 0 and "Alignment 'fixture' taught" in out and "axis ratio" in out
    code, out, _ = cli_run(capsys, lib, "check-align", "fixture", "v")
    assert code == 1 and "INDETERMINATE" in out and "pass --tol-px" in out    # px, no px band
    code, out, _ = cli_run(capsys, lib, "check-align", "fixture", "v", "--tol-px", "3")
    assert code == 0 and "ALIGNED" in out
    # the same part at the bench centre: 20 mm from the validation object's place
    code, _, _ = cli_run(capsys, lib, "--mock-part", "106.95,56.5,21.9", "scan", "moved")
    assert code == 0
    code, out, _ = cli_run(capsys, lib, "--json", "check-align", "fixture", "moved",
                           "--tol-px", "3")
    doc = json.loads(out)
    assert code == 1 and doc["alignment"]["verdict"] == "NOT_ALIGNED"
    assert doc["alignment"]["position_ok"] is False and doc["alignment"]["dx_px"] > 30
    code, _, err = cli_run(capsys, lib, "check-presence", "fixture")
    assert code == 2 and "presence" in err


def test_library_listings_and_bank_remove(shared, tmp_path, capsys):
    lib = copy_of(shared, tmp_path)
    assert cli_run(capsys, lib, "calibrate", "v")[0] == 0
    code, out, _ = cli_run(capsys, lib, "calibrations")
    assert code == 0 and "Calibrations (1)" in out and "<- in use for the preset" in out
    code, out, _ = cli_run(capsys, lib, "bank-add", "v", "--bank", "b1")
    assert code == 0
    code, out, _ = cli_run(capsys, lib, "banks")
    assert code == 0 and re.search(r"b1\s+mock-wall_2p7ft_vstripe_v1\s+1\s", out)
    code, out, _ = cli_run(capsys, lib, "bank-remove", "v", "--bank", "b1", "--reason", "wrong")
    assert code == 0 and "0 known-good scan(s) left" in out
    code, out, _ = cli_run(capsys, lib, "--json", "banks")
    assert json.loads(out)["banks"][0]["n_scans"] == 0
    code, _, err = cli_run(capsys, lib, "bank-remove", "v", "--bank", "b1")
    assert code == 2 and "not a member" in err
    for cmd, head in (("readings", "Readings: 0 code, 0 spectral"), ("materials", "Materials (0)"),
                      ("maps", "Projector maps (0)"), ("sensors", "Sensors (0)")):
        code, out, _ = cli_run(capsys, lib, cmd)
        assert code == 0 and head in out, (cmd, out)
        code, out, _ = cli_run(capsys, lib, "--json", cmd)
        assert code == 0 and json.loads(out)["ok"], cmd


# ------------------------------------------------------------ review leftovers

def test_rig_busy_error_names_the_way_out(tmp_path, capsys, monkeypatch):
    from engine import rules

    def busy(preset=None):
        raise rules.RigBusy("the camera + DMD are in use by another process "
                            "(pid 1, web UI (port 8765))")
    monkeypatch.setattr(api, "check_signal", busy)
    code, _, err = cli_run(capsys, tmp_path / "lib", "signal")
    assert code == 2 and err.strip().count("\n") == 0
    assert "web UI (port 8765)" in err
    assert "another program holds the rig - wait or use the web UI job queue" in err


def test_auto_rejected_base_is_reported_and_exits_1(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(api, "BASE_MIN_VALID_PCT", 101.0)       # every base is under the floor
    code, out, _ = cli_run(capsys, tmp_path / "lib", "base")
    assert code == 1 and "AUTO-REJECTED" in out and "! base auto-REJECTED" in out
    assert "capture the base again" in out and "Next: place the part" not in out


def test_gamma_shows_residual_and_tune_marks_frame_locked(tmp_path, capsys, monkeypatch):
    from engine.capture.assist import GammaReport, TuneReport, TuneStep
    rep = GammaReport(gamma=2.2, levels=[64, 255], response=[0.05, 1.0], r2=0.999,
                      recommended={}, residual_gamma=1.02)
    monkeypatch.setattr(api, "measure_gamma", lambda preset=None, n_levels=12: rep)
    code, out, _ = cli_run(capsys, tmp_path / "lib", "gamma")
    assert code == 0 and re.search(r"residual gamma\s+1\.020 \(the preset chain", out)
    steps = [TuneStep(exposure_us=e, gc_conf=60.0, valid_pct=40.0, saturated_pct=0.0,
                      focus_ratio=0.9, coded_axis="x", quality="good", frame_locked=lk)
             for e, lk in ((16667, True), (8000, False))]
    monkeypatch.setattr(api, "tune", lambda preset=None, exposures=None, duties=None,
                        frame_us=None: TuneReport(steps=steps, recommended={}, advice=[],
                                                  ok=True))
    code, out, _ = cli_run(capsys, tmp_path / "lib", "tune")
    assert code == 0 and "frame-locked" in out
    assert re.search(r"16667 us.*\byes\b", out) and re.search(r"8000 us.*\bNO\b", out)


def test_help_lists_the_sensor_commands(capsys):
    assert cli.main(["--help"]) == 0
    out = capsys.readouterr().out
    for c in ("spot-size", "teach-presence", "check-presence", "teach-align", "check-align",
              "sensors", "bank-remove", "calibrations"):
        assert c in out, c
    assert "exit 0 PRESENT, 1 ABSENT or INDETERMINATE" in out


# ------------------------------------------------------------ wizard

def _record(monkeypatch, events: list, *verbs):
    for name in verbs:
        real = getattr(api, name)

        def wrapped(*a, _real=real, _name=name, **k):
            events.append(_name)
            return _real(*a, **k)
        monkeypatch.setattr(api, name, wrapped)


def test_wizard_guides_base_part_measure_inspect_reconstruct(tmp_path, capsys, monkeypatch):
    events = []
    _record(monkeypatch, events, "check_signal", "capture_base", "capture_object",
            "measure", "inspect", "reconstruct")
    answer(monkeypatch, {"Clear the scene": lambda: events.append("CLEAR") or "",
                         "Place the part": lambda: events.append("PLACE") or "",
                         "Part label": "widget", "Known dimensions": V_DIMS}, [])
    code, out, _ = cli_run(capsys, tmp_path / "lib", "wizard", "--interactive")
    assert code == 0, out
    # physical steps are prompted BEFORE the verbs that depend on them
    assert events == ["CLEAR", "check_signal", "capture_base", "PLACE", "capture_object",
                      "measure", "inspect", "reconstruct"]
    assert "Dimensions of" in out and "Inspection of" in out and "rule 6" in out
    written = paths_in(out, ".json") + paths_in(out, ".png") + paths_in(out, ".ply")
    kinds = {k for k in ("measurements", "defects", "pointclouds", "previews")
             if any(k in str(p) for p in written)}
    assert kinds == {"measurements", "defects", "pointclouds", "previews"}
    assert all(p.exists() for p in written)
    assert "calibrate widget" in out                    # known dims, still px: says what next


def test_wizard_refuses_bad_signal_unless_forced(tmp_path, capsys, monkeypatch):
    calls = []
    monkeypatch.setattr(api, "check_signal", lambda preset=None: calls.append(1) or BAD_SIGNAL)
    replies = iter(["", "q"])                      # Enter = re-check, then quit
    prompts = []
    answer(monkeypatch, {"NOT ok": lambda: next(replies)}, prompts)
    code, out, err = cli_run(capsys, tmp_path / "lib", "wizard", "--interactive")
    assert code == 1 and "stopped" in err
    assert len(calls) == 2 and api.list_library("bases") == []     # never captured
    assert any("Clear the scene (empty base), press Enter" in p for p in prompts)

    prompts.clear()
    answer(monkeypatch, {"NOT ok": "force", "Place the part": "q"}, prompts)
    code, out, err = cli_run(capsys, tmp_path / "lib2", "wizard", "--interactive")
    assert code == 1 and "forced by the operator" in out
    assert len(api.list_library("bases")) == 1                     # base captured, then quit
    assert any("Place the part, do not touch camera/projector, press Enter" in p
               for p in prompts)


# ------------------------------------------------------------ no command waits for input

def _never_answers(monkeypatch, prompts: list):
    """input() of a terminal nobody types into: it blocks (until the test releases it)."""
    import threading
    gate = threading.Event()

    def fake(prompt=""):
        prompts.append(prompt)
        gate.wait(30)
        return ""
    monkeypatch.setattr("builtins.input", fake)
    return gate


def test_no_command_waits_for_input_without_interactive(shared, tmp_path, capsys, monkeypatch):
    """teach-presence, delete and the wizard used to call input() on their own: run where
    nobody answers (an agent, a detached terminal) they waited for ever. Without
    --interactive none of them asks: refused (exit 2) with the argument to pass."""
    lib = copy_of(shared, tmp_path)
    prompts = []
    gate = _never_answers(monkeypatch, prompts)
    try:
        for argv, want in ((["teach-presence", "pe", SPOT, "--diameter", "6.5"],
                            "--empty-confirmed"),
                           (["delete", "v"], "--yes"),
                           (["wizard"], "--interactive")):
            code, out, err = cli_run(capsys, lib, *argv)
            assert code == 2 and want in err, (argv, err)
        assert prompts == []                               # input() was never reached
        assert list((lib / "objects").iterdir()) and not (lib / "sensors").exists()
        assert api.list_library("bases") and len(api.list_library("bases")) == 1
    finally:
        gate.set()


def test_an_interactive_prompt_gives_up_and_fails_closed(shared, tmp_path, capsys, monkeypatch):
    """--interactive asks, but never for ever: no answer within --prompt-timeout -> the
    command stops with nothing done (exit 1)."""
    import time
    lib = copy_of(shared, tmp_path)
    prompts = []
    gate = _never_answers(monkeypatch, prompts)
    fast = ["--interactive", "--prompt-timeout", "0.3"]
    try:
        for argv in (["teach-presence", "pe", SPOT, "--diameter", "6.5"], ["delete", "v"],
                     ["wizard"]):
            t0 = time.monotonic()
            code, out, err = cli_run(capsys, lib, *fast, *argv)
            assert code == 1 and "no answer within 0.3 s" in err, (argv, err)
            assert "nothing was done" in err and time.monotonic() - t0 < 20
        assert len(prompts) == 3
        assert list((lib / "objects").iterdir()) and not (lib / "sensors").exists()
        assert len(api.list_library("bases")) == 1         # the wizard captured nothing
    finally:
        gate.set()
    for bad in ("0", "-1", "nan", "inf", "7200", "soon"):
        code, out, err = cli_run(capsys, lib, "--prompt-timeout", bad, "bases")
        assert code == 2, bad


def test_commands_return_with_stdin_held_open(tmp_path):
    """The real thing, in a separate process whose stdin is a pipe that is never written to
    and never closed (what a hung terminal or an agent's shell looks like): each command
    must come back on its own. Before, teach-presence waited on its question for ever."""
    lib = tmp_path / "lib"
    env = {**os.environ, "PYTHONPATH": str(ROOT), "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHONIOENCODING": "utf-8"}
    base = [sys.executable, "-m", "ui.cli", "--mock", "--average", "1", "--library", str(lib)]
    cases = [(["teach-presence", "pe", SPOT, "--diameter", "6.5"], 2, "--empty-confirmed"),
             (["delete", "nothing-here"], 2, ""),
             (["wizard"], 2, "--interactive"),
             (["--interactive", "--prompt-timeout", "1", "teach-presence", "pe", SPOT,
               "--diameter", "6.5"], 1, "no answer within 1 s"),
             (["teach-presence", "pe", SPOT, "--diameter", "6.5", "--empty-confirmed"], 0, ""),
             (["--mock-part", "none", "check-presence", "pe"], 1, ""),
             (["spot", SPOT, "--diameter", "6.5"], 0, "")]
    for argv, want, text in cases:
        proc = subprocess.Popen([*base, *argv], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, env=env, cwd=str(tmp_path))
        try:
            out, err = _wait_open(proc, 180)
        finally:
            if proc.poll() is None:
                proc.kill()
        assert proc.returncode == want, (argv, proc.returncode, err[-400:])
        assert text in err, (argv, err[-400:])


def _wait_open(proc, timeout: float) -> tuple:
    """Wait for the process WITHOUT closing its stdin (communicate() would close it and hand
    the command an EOF, which is exactly what a hung terminal never does)."""
    import threading
    chunks = {"out": b"", "err": b""}

    def drain(name, pipe):
        chunks[name] = pipe.read()
    readers = [threading.Thread(target=drain, args=(n, p), daemon=True)
               for n, p in (("out", proc.stdout), ("err", proc.stderr))]
    for th in readers:
        th.start()
    try:
        proc.wait(timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        raise AssertionError("the command did not return: it is waiting for input") from None
    for th in readers:
        th.join(10)
    proc.stdin.close()
    return chunks["out"].decode("utf-8", "replace"), chunks["err"].decode("utf-8", "replace")


# ------------------------------------------------------------ delete: calibration data

def test_delete_trashes_calibration_poses_and_calibration_entries(tmp_path, capsys):
    """A bad calibration pose had to be moved to the library trash by hand: `delete` takes
    calibration poses (by id), stereo calibrations, rig references and parity measurements.
    Moved with the reason, never destroyed; a pose SET name is not an id."""
    import numpy as np
    from engine import config
    from recipe_synth import truth_calibration
    lib = tmp_path / "lib"
    s = api.configure(lib, mock=True)
    try:
        p = api.load_preset(PRESET)
        geo = p.geometry_id
        fp = config.geometry_fingerprint(p)
        poses = []
        for k in range(3):
            d = s.store.new_dir("calib_poses", geo, label="setA")
            np.savez(d / "flood.npz", image=np.zeros((4, 4), np.uint8))
            s.store.write_json(d / "meta.json", {
                "id": d.name, "set": "setA", "geometry_id": geo, "geometry_fp": fp,
                "corners_found": 50, "projector_corners": 4 if k == 1 else 48,
                "projector_corner_pct": 8.0 if k == 1 else 96.0, "accepted": k != 1,
                "periods_px": [8.0, 80.0, 1280.0], "created": "2026-01-01T00:00:00"})
            poses.append(d)
        cal = truth_calibration(s, p)
        par = s.store.new_dir("calibrations", geo, label="parity")
        s.store.write_json(par / "parity.json", {"parity": None, "ok": True, "geometry_fp": fp})
        ref = s.store.new_dir("calibrations", geo, label="rigref")
        s.store.write_json(ref / "rigref.json", {"kind": "rigref", "geometry_fp": fp})
    finally:
        api._session = None
    bad = poses[1].name
    code, out, _ = cli_run(capsys, lib, "calib-poses", "--set", "setA")
    assert code == 0 and "Calibration poses (3) of set setA" in out and bad in out
    assert "4 (8 %)" in out and "NO" in out and "8,80,1280" in out
    code, out, err = cli_run(capsys, lib, "delete", "setA", "--yes")      # a set is not an id
    assert code == 2 and "ambiguous" in err and all(d.is_dir() for d in poses)
    code, out, err = cli_run(capsys, lib, "delete", bad)                  # no --yes: no wait
    assert code == 2 and "--yes" in err and poses[1].is_dir()
    code, out, _ = cli_run(capsys, lib, "delete", bad, "--yes", "--reason", "8 px fringes: "
                           "4 of 50 projector corners")
    assert code == 0 and f"Moved calibration pose {bad} to trash" in out
    trashed = lib / "trash" / "calib_poses" / geo / bad
    assert trashed.is_dir() and not poses[1].exists()                     # moved, not destroyed
    assert (trashed / "meta.json").is_file() and (trashed / "flood.npz").is_file()
    rec = json.loads((trashed / "TRASHED.json").read_text())
    assert rec["reason"].startswith("8 px fringes") and rec["from"].endswith(bad)
    code, out, _ = cli_run(capsys, lib, "--json", "calib-poses")
    assert [m["id"] for m in json.loads(out)["calib_poses"]] == [poses[0].name, poses[2].name]
    # the calibration entries: the stereo calibration, the parity measurement, the reference
    for what, ident, folder in (("calibration", cal.id, cal.path),
                                ("parity measurement", f"{geo}/{par.name}", par),
                                ("rig reference", f"{geo}/{ref.name}", ref)):
        code, out, _ = cli_run(capsys, lib, "--json", "delete", ident, "--yes", "--reason",
                               "recalibrating")
        doc = json.loads(out)
        assert code == 0 and doc["deleted"]["kind"] == what, (what, doc)
        assert not folder.exists() and (lib / "trash" / "calibrations" / geo / folder.name).is_dir()
        assert json.loads((lib / "trash" / "calibrations" / geo / folder.name / "TRASHED.json")
                          .read_text())["reason"] == "recalibrating"
    code, out, _ = cli_run(capsys, lib, "--json", "calibrations")
    assert json.loads(out)["calibrations"] == []


# ------------------------------------------------------------ rule 1 + entry points

def test_cli_never_imports_usb_modules():
    tree = ast.parse((ROOT / "ui" / "cli.py").read_text(encoding="utf-8"))
    mods = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mods += [node.module or ""] + [a.name for a in node.names]
    bad = [m for m in mods if m == "hid" or m.startswith("swiss_army_sensor")
           or m.split(".")[-1] == "dmd_control"]
    assert not bad, bad


def test_cli_run_never_loads_usb_driver(tmp_path):
    code = f"""
import sys; sys.path.insert(0, {str(ROOT)!r})
from ui import cli
rc = cli.main(["--mock", "--average", "1", "--library", {str(tmp_path)!r}, "signal"])
bad = [m for m in ("swiss_army_sensor.drivers.dlp_driver", "swiss_army_sensor.drivers",
                   "swiss_army_sensor", "hid") if m in sys.modules]
print("RC", rc, "BAD", bad)
"""
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         timeout=300)
    assert out.returncode == 0, out.stderr
    assert "RC 0 BAD []" in out.stdout, out.stdout


def test_python_dash_m_entry_point(tmp_path):
    out = subprocess.run([sys.executable, "-m", "ui.cli", "--mock", "--library",
                          str(tmp_path / "lib"), "paths"], cwd=ROOT, capture_output=True,
                         text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert "output_index" in out.stdout


@pytest.mark.skipif(sys.platform != "win32", reason="scan.bat is the Windows launcher")
def test_scan_bat_launcher(tmp_path):
    bat = ROOT / "scan.bat"
    text = bat.read_text(encoding="ascii")
    assert '"%SCAN_PYTHON%" -m ui.cli %*' in text
    out = subprocess.run(["cmd", "/c", str(bat), "--mock", "--library", str(tmp_path / "lib"),
                          "paths"], cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert "output_index" in out.stdout


# ------------------------------------------------ cell & 3D, maps, checker, surfer

ATLAS_SRC = Path(os.environ.get("SCANNER_TEST_ATLAS", ""))   # a real held-cell atlas.json; skipped if unset


@pytest.fixture(scope="module")
def cell_lib(tmp_path_factory):
    """A base and a scan of the mock standing cell (--mock-part cylinder)."""
    lib = tmp_path_factory.mktemp("cli_cell") / "library"
    for argv in (["base"], ["--mock-part", "cylinder:24.05,90,1.5", "scan", "cell"]):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = cli.main([*FAST, "--library", str(lib), *argv])
        assert code == 0, buf.getvalue()
    api._session = None
    return lib


def test_mock_part_cylinder_parsing():
    assert cli._mock_part("cylinder") == "cylinder:24.05,90,1.5"
    assert cli._mock_part("cyl:20,60") == "cylinder:20,60,1.5"
    assert cli._mock_part("cylinder:20,60,0") == "cylinder:20,60,0"
    for bad in ("cylinder:0,60", "cylinder:a,b", "cylinder:1,2,3,4"):
        with pytest.raises(Exception, match="cylinder"):
            cli._mock_part(bad)


def test_mesh_unwrap_dents_maps_json_and_exit_codes(cell_lib, capsys):
    code, out, _ = cli_run(capsys, cell_lib, "--json", "mesh", "cell", "--kind", "cylinder",
                           "--radius", "24.05", "--stride", "4")
    doc = json.loads(out)
    assert code == 0 and doc["file"].endswith(".stl") and Path(doc["file"]).is_file()
    assert doc["meta"]["finite_vertices"] is True and doc["meta"]["open_arc_only"] is True
    code, out, _ = cli_run(capsys, cell_lib, "mesh", "cell", "--fmt", "obj")
    assert code == 0 and ".obj" in out and "rule 6" in out
    code, out, _ = cli_run(capsys, cell_lib, "--json", "unwrap", "cell", "--radius", "24.05")
    doc = json.loads(out)
    assert code == 0 and 60 < doc["unwrap"]["arc_deg"] < 180
    assert any(k.endswith(".png") for k in doc["files"])
    code, out, _ = cli_run(capsys, cell_lib, "--json", "dents", "cell")
    doc = json.loads(out)
    verdict = doc["dents"]["verdict"]
    assert code == (0 if verdict == "PASS" else 1)
    assert doc["dents"]["dents"] and {"theta_deg", "z", "depth", "units"} <= set(
        doc["dents"]["dents"][0])
    if doc["dents"]["units"] == "px":
        assert verdict == "INDETERMINATE"
    code, out, _ = cli_run(capsys, cell_lib, "dents", "cell")
    assert code == 1 and "theta deg" in out and "rule 6" in out
    code, out, _ = cli_run(capsys, cell_lib, "--json", "maps", "cell")
    doc = json.loads(out)
    assert code == 0 and {"height", "normal", "dust"} & set(doc["maps"]["png_paths"]) >= \
        {"height", "normal"}
    assert doc["maps"]["height"]["array_shape"]            # arrays summarised, not dumped
    code, out, _ = cli_run(capsys, cell_lib, "maps")        # no scan: the projector-map list
    assert code == 0 and "Projector maps" in out
    code, out, _ = cli_run(capsys, cell_lib, "maps", "cell", "--light", "nope")
    assert code == 2


def test_checker_and_flash_pack(cell_lib, capsys, monkeypatch, tmp_path):
    code, out, _ = cli_run(capsys, cell_lib, "--json", "checker", "--square", "32")
    doc = json.loads(out)
    assert code == 0 and doc["checker"]["n_corners"] > 0
    assert set(doc["checker"]["png_paths"]) >= {"stretch", "contrast"}
    import tools.make_flash_pack as fp
    monkeypatch.setattr(fp, "DEFAULT_OUT", tmp_path / "packs")
    code, out, _ = cli_run(capsys, cell_lib, "--json", "flash-pack", "--axes", "y")
    doc = json.loads(out)
    assert code == 0 and Path(doc["folder"]).parent == tmp_path / "packs"
    assert (Path(doc["folder"]) / "manifest.json").is_file()


def test_atlas_import_and_surfer_commands(cell_lib, capsys, tmp_path):
    if not ATLAS_SRC.is_file():
        pytest.skip(f"{ATLAS_SRC} not on this machine")
    before = ATLAS_SRC.read_bytes()
    src = Path(shutil.copy(ATLAS_SRC, tmp_path / "atlas.json"))
    code, out, _ = cli_run(capsys, cell_lib, "atlas-import", str(src), "cell")
    assert code == 0 and "atlas:cell/<region>" in out
    code, out, _ = cli_run(capsys, cell_lib, "--json", "atlases")
    doc = json.loads(out)
    specs = {g["spec"] for a in doc["atlases"] for g in a["regions"]}
    assert {"atlas:cell/cell", "atlas:cell/floor"} <= specs
    code, out, _ = cli_run(capsys, cell_lib, "--json", "flood", "atlas:cell/cell")
    doc = json.loads(out)
    assert code == (0 if doc["flood"]["lands"] else 1) and doc["flood"]["verdict"]
    code, out, _ = cli_run(capsys, cell_lib, "--json", "antiglare", "atlas:cell/cell",
                           "--carriers", "0,0;0.3,-0.5")
    doc = json.loads(out)
    assert code == (0 if doc["anti_glare"]["lock"] else 1) and doc["anti_glare"]["wording"]
    code, out, _ = cli_run(capsys, cell_lib, "multiorient", "atlas:cell/cell",
                           "--orientations", "v,h")
    assert code == 0 and "median_quality" in out
    code, _, _ = cli_run(capsys, cell_lib, "atlas-import", str(tmp_path / "missing.json"), "x")
    assert code == 2
    assert ATLAS_SRC.read_bytes() == before


def test_help_lists_the_cell_and_surfer_commands(capsys):
    assert cli.main(["--help"]) == 0
    out = capsys.readouterr().out
    for c in ("mesh", "unwrap", "dents", "checker", "atlas-import", "atlases", "flood",
              "antiglare", "multiorient", "flash-pack", "atlas:<name>/<region>"):
        assert c in out, c

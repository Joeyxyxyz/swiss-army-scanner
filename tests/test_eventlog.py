"""Inspection event log (engine/eventlog.py) and its page (ui/web/static/events.html), MOCK only.

One row per inspection that opened a run folder - a recipe run, a quick check, a station job,
a cycle (ONE row for its three work states, plus their value rows), the check of a work state
run outside a cycle, a measurement - with the judgment an engineer reads (PASS / FAIL / PULL,
"-" for a measurement), the reason of a PULL (guard band, refused, crashed), the values with
their limits (a check's tolerances, its key / limit, a single job's deviation against its
threshold), the code read as the part id, the picture copied to events/img/, the links to the
height profile, the 3D data and result.json. The rules of the file: one pair per month with
v1 in the name, whole rows from two processes at once, a row that cannot be appended (its file
open elsewhere, or the log busy) waits in events/pending/ and is merged at the next append, a
cell a spreadsheet would run gets an apostrophe, rebuild from the run folders gives the same
rows and may run while inspections are made, the newest row is also a small file of its own -
and none of it ever changes a result.json or an exit code. What happens to a run after its
result is in the log too: ABORT after the verdict reads PULL (the last row of an inspection
counts), a cycle that never ended is a PULL row after a rebuild.

The hand-built runs go through engine/runs.write_result (the hook) with no engine session:
their log is beside their run folders. The real runs check synthetic metric scans offline
(tests/recipe_synth.py: nothing is captured). The log is off for every test (tests/conftest.py,
SAS_EVENTLOG=off); a test that wants rows takes the `eventlog_on` fixture and keeps its
library under the temporary folder. The page logic runs in node on the real events.js with a
tiny fake DOM."""

import contextlib
import csv
import io
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from engine import api, eventlog
from engine import disposition as D
from engine import runs as R
from engine.flow import verify as FV
from engine.modes import measure as MM
from engine.recipes import runner
from ui import cli

import recipe_synth as RS
from conftest import EVENTLOG_ENV, ROOT
from test_run_job_mock import RESULT_KEYS

STATIC = ROOT / "ui" / "web" / "static"
EXAMPLE = ROOT / "stations" / "example_tube_line.json"
PRESET = "bench_9in_metric_bright"
# the format `v1`: the columns of the two files, in this order (a change needs a new name)
INSPECTION_COLUMNS = ["time", "inspection_id", "part_id", "part", "program", "operating_mode",
                      "hardware", "judgment", "reason", "n_values", "n_out_of_limit", "values",
                      "time_s", "image_file", "profile_file", "data_3d_file", "result_file",
                      "preset", "calibration_id", "rig", "software"]
# the header written before v1 was settled (2026-10-13): still read, brought to v1 on append
EARLY_V1_COLUMNS = ["time", "inspection_id", "part_id", "part", "program", "mode", "judgment",
                    "reason", "n_values", "n_out_of_limit", "values", "time_s", "image_file",
                    "profile_file", "data_3d_file", "result_file", "preset", "calibration_id",
                    "rig", "software"]
LOCAL_TIME = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d[+-]\d\d:\d\d")
VALUE_COLUMNS = ["inspection_id", "step", "zone", "check", "name", "value", "unit",
                 "lower_limit", "upper_limit", "result"]
PNG = b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 8        # stands in for a run's picture


@pytest.fixture(autouse=True)
def _fresh_engine():
    yield
    api._session = None


@pytest.fixture(autouse=True)
def _nothing_to_wait_for(monkeypatch):
    """rebuild waits for a run that ended a moment ago: its own hook may still be writing its
    row. In these tests every hook has returned before rebuild is called, so the wait is
    off; test_rebuild_waits_for_a_run_that_ended_a_moment_ago turns it on again."""
    monkeypatch.setattr(eventlog, "HOOK_SETTLE_S", 0.0)


# ================================================================== helpers

def _job(runs, name="plate_check", *, result="PASS", max_dev=None, threshold=None, zones=None,
         codes=None, part="plate", picture=None, settle=None, job=None, **keys) -> tuple:
    """One run folder written as the station job runner writes it: run.log.jsonl (run_start,
    the codes read), the picture, then result.json through runs.write_result - which calls
    the event-log hook. settle(res): how the run's disposition is decided (default: from its
    verdict). -> (run id, run folder, the result document as written)."""
    run_id, run_dir = R.new_run(runs, name)
    res = R.empty_result(run_id=run_id, run_folder=str(run_dir), mock=True,
                         operator_confirmed=False, station=name,
                         job=job or {"type": "metric", "base": None, "zone": None,
                                     "config": f"{name}.json"},
                         part={"role": "test", "label": part}, threshold_mm=threshold,
                         station_config=f"{name}.json", profile="default")
    log = R.RunLog(run_dir / "run.log.jsonl", run_id)
    log.event("run_start", station=name, mode="MOCK")
    if codes:
        log.event("codes", n=len(codes), data=list(codes))
    res.update(result=result, engine_verdict=result,
               invalid_reason=None if result in ("PASS", "FAIL") else f"result {result}")
    res["capture"].update(preset="p1", geometry_id="rig_a")
    res["calibration"]["calibration_id"] = "rig_a/cal_1"
    if max_dev is not None:
        res["deviation_mm"].update(max=max_dev, method="plane_fit")
        res["deviation_units"] = "mm"
    if zones is not None:
        res["zones"] = zones
    res.update(keys)
    if picture:
        (run_dir / picture).write_bytes(PNG)
    (settle or D.settle)(res)
    R.write_result(run_dir, res)
    log.event("run_end", result=res["result"], disposition=res["disposition"])
    log.close()
    return run_id, run_dir, json.loads((run_dir / "result.json").read_text(encoding="utf-8"))


def _month(doc) -> str:
    """The month file a run's rows are in: the month of its local time."""
    return eventlog.month_of(eventlog.local_time(doc["finished"]))


def _log(library, month=None) -> tuple:
    """(the inspection rows, the value rows) of a library's event log, as text cells."""
    events = Path(library) / "events"
    months = [f.stem[-7:] for f in sorted(events.glob(f"inspections_v1_{month or '*'}.csv"))]
    rows = [r for m in months for r in eventlog.read_rows(events / f"inspections_v1_{m}.csv")]
    vals = [r for m in months
            for r in eventlog.read_rows(events / f"inspection_values_v1_{m}.csv")]
    return rows, vals


def _row(library, run_id) -> dict:
    (row,) = [r for r in _log(library)[0] if r["inspection_id"] == run_id]
    return row


def _values(library, run_id) -> list:
    return [v for v in _log(library)[1] if v["inspection_id"] == run_id]


def _check(cid, verdict, *, tolerances=None, key=None, limit=None, values=None, message="",
           pull=None) -> dict:
    """A check result as the recipe runner records it."""
    values = dict(values or {})
    return {"id": cid, "type": cid, "verdict": verdict, "values": values,
            "units": {k: ("mm" if k.endswith("_mm") else "%" if k.endswith("_pct") else "")
                      for k in values},
            "key": key, "limit": limit, "message": message, "fault_points": [], "warnings": [],
            "details": {}, "tolerances": tolerances, "pull": pull}


@contextlib.contextmanager
def _open_elsewhere(monkeypatch, *names):
    """While inside: the named files cannot be written, as when a spreadsheet program has
    them open (it denies writers). Everything else the log writes is untouched."""
    real = eventlog._open

    def refusing(path, mode):
        if Path(path).name in names and Path(path).parent.name == "events":
            raise PermissionError(13, "the file is open in another program", str(path))
        return real(path, mode)
    with monkeypatch.context() as m:
        m.setattr(eventlog, "_open", refusing)
        yield


# ================================================================== off for tests

def test_with_the_log_off_nothing_is_written_anywhere(tmp_path, monkeypatch):
    """SAS_EVENTLOG=off (every test, unless it asks): no row, no waiting row, no picture, no
    profile file, no lock file, not even the folder."""
    assert os.environ[EVENTLOG_ENV] == "off" and not eventlog.enabled()
    run_id, run_dir, _ = _job(tmp_path / "runs", max_dev=0.1, threshold=0.2, picture="overlay.png")
    assert eventlog.record(run_dir) is None and eventlog.amend(run_dir) is None
    R.append_event(run_dir / "run.log.jsonl", run_id, "aborted_after_verdict")  # (amends)
    got = eventlog.rows_of(run_dir)                         # reading is never refused
    assert got[0]["inspection_id"] == run_id and got[0]["image_file"] == ""
    assert eventlog.append(tmp_path / "runs", *got) == {"written": 0, "pending": 0, "lost": 0}
    assert eventlog.write_profiles(run_dir, {"z": {"pos_mm": [0.0], "x_mm": [0.1],
                                                   "y_mm": [0.2]}}) == []
    out = eventlog.rebuild(tmp_path / "runs", [tmp_path / "runs"])
    assert out["ok"] is False and "off" in out["error"]
    assert eventlog.main(["--rebuild", "--library", str(tmp_path / "runs")]) == 2
    assert eventlog.main(["--merge", "--library", str(tmp_path / "runs")]) == 2
    assert sorted(p.name for p in run_dir.iterdir()) == ["overlay.png", "result.json",
                                                         "run.log.jsonl"]
    assert [p.name for p in (tmp_path / "runs").iterdir()] == [run_id]
    # the switch is read at every call, never once at import
    for value, on in (("on", True), ("OFF", False), ("0", False), ("false", False), ("", True)):
        monkeypatch.setenv(EVENTLOG_ENV, value)
        assert eventlog.enabled() is on
    monkeypatch.delenv(EVENTLOG_ENV)
    assert eventlog.enabled() is True                       # a real session: on


# ================================================================== the row of a single job

def test_a_station_job_is_one_row_with_its_value_and_limit(tmp_path, eventlog_on):
    runs = tmp_path / "runs"
    run_id, run_dir, doc = _job(runs, max_dev=0.1234567, threshold=0.2, picture="deviation.png",
                                codes=["SN-0042"])
    month = _month(doc)
    events = runs / "events"                # no engine session: the log is beside the run folders
    assert sorted(p.name for p in events.iterdir()) == [
        "eventlog.lock", "img", f"inspection_values_v1_{month}.csv", f"inspections_v1_{month}.csv",
        "last_v1.csv"]
    raw = (events / f"inspections_v1_{month}.csv").read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf" + ",".join(INSPECTION_COLUMNS).encode() + b"\r\n")
    assert raw.endswith(b"\r\n") and raw.count(b"\r\n") == 2
    assert list(eventlog.INSPECTION_FIELDS) == INSPECTION_COLUMNS
    assert list(eventlog.VALUE_FIELDS) == VALUE_COLUMNS
    (row,), (val,) = _log(runs)
    assert list(row) == INSPECTION_COLUMNS and list(val) == VALUE_COLUMNS
    # the shop's own clock, with its offset to UTC: the same moment as the document's UTC time
    assert LOCAL_TIME.fullmatch(row["time"]) and row["time"] == eventlog.local_time(doc["finished"])
    assert datetime.fromisoformat(row["time"]) == datetime.fromisoformat(doc["finished"])
    assert row["inspection_id"] == run_id == run_dir.name
    assert (row["part_id"], row["part"], row["program"]) == ("SN-0042", "plate", "plate_check.json")
    # REAL / MOCK is the hardware; how the scanner was used is the operating mode
    assert (row["operating_mode"], row["hardware"]) == ("station", "MOCK")
    assert (row["judgment"], row["reason"]) == ("PASS", "")
    assert (row["n_values"], row["n_out_of_limit"]) == ("1", "0")
    assert row["values"] == "deviation_mm.max=0.123457 mm [..0.2] OK"
    assert float(row["time_s"]) >= 0.0
    assert (row["preset"], row["calibration_id"], row["rig"]) == ("p1", "rig_a/cal_1", "rig_a")
    # the software that judged it: the version (installed, or the source tree's) and the commit
    assert row["software"] == " ".join(str(v) for v in doc["software"].values() if v)
    assert row["software"].split()[0] == doc["software"]["package_version"]
    assert doc["software"]["package_version"] not in (None, "")
    assert row["result_file"] == f"{run_id}/result.json" and row["profile_file"] == ""
    assert row["data_3d_file"] == ""
    # the picture is copied: the link survives a cleaned run folder
    assert row["image_file"] == f"events/img/{run_id}.png"
    assert (events / "img" / f"{run_id}.png").read_bytes() == PNG
    assert val == {"inspection_id": run_id, "step": "", "zone": "", "check": "metric",
                   "name": "deviation_mm.max", "value": "0.123457", "unit": "mm",
                   "lower_limit": "", "upper_limit": "0.2", "result": "OK"}
    # result.json is what it was: the log adds nothing to it
    assert set(doc) == RESULT_KEYS


def _guard_band(res):
    D.settle(res, guard_band_pct=10, tolerances=[
        D.tolerance("deviation_mm.max", res["deviation_mm"]["max"], hi=res["threshold_mm"])])


def _refused(res):
    D.add(res, "refused", "RuleViolation: the real rig needs a human to confirm the setup")
    D.settle(res)


def _crashed(res):
    D.add(res, "error", "RuntimeError: partial write:\n only 3 points written")
    D.settle(res)


@pytest.mark.parametrize("kw, judgment, reason, value", [
    (dict(max_dev=0.1, threshold=0.2), "PASS", "", "OK"),
    (dict(result="FAIL", max_dev=0.31, threshold=0.2), "FAIL",
     "deviation_mm.max 0.31 mm is outside its limits [..0.2]", "OUT"),
    (dict(max_dev=0.195, threshold=0.2, settle=_guard_band), "PULL", "guard_band: deviation", "OK"),
    (dict(result="INVALID", settle=_refused), "PULL", "refused: RuleViolation: the real rig", None),
    (dict(result="INVALID", settle=_crashed), "PULL",
     "error: RuntimeError: partial write: only 3 points written", None),
    (dict(result="ABORTED"), "PULL", "aborted: the cycle was aborted", None),
    (dict(result="INVALID", max_dev=0.1, threshold=0.2), "PULL", "invalid: result INVALID",
     "INVALID"),
], ids=["pass", "fail", "guard_band", "refused", "crashed", "aborted", "invalid"])
def test_judgment_is_pass_fail_or_pull_with_the_reason(tmp_path, eventlog_on, kw, judgment,
                                                       reason, value):
    """The lamp's word: CONTINUE reads PASS, REJECT reads FAIL, every doubt reads PULL and
    says why - a pass in the guard band, a run refused, a run that crashed."""
    run_id, _, doc = _job(tmp_path, **kw)
    row = _row(tmp_path, run_id)
    assert row["judgment"] == judgment == eventlog.JUDGMENTS[doc["disposition"]]
    assert row["reason"].startswith(reason) and "\n" not in row["reason"]
    vals = _values(tmp_path, run_id)
    assert [v["result"] for v in vals] == ([value] if value else [])
    assert row["n_out_of_limit"] == str(sum(v["result"] == "OUT" for v in vals))


def test_a_number_the_document_holds_as_text_is_still_a_number(tmp_path, eventlog_on):
    """result.json is written with default=str: a numpy scalar that is no Python float
    arrives as text. It is still the value, and still held to its limit."""
    run_id, _, doc = _job(tmp_path, result="FAIL", max_dev=np.float32(0.25), threshold=0.2)
    assert doc["deviation_mm"]["max"] == "0.25"
    (val,) = _values(tmp_path, run_id)
    assert (val["value"], val["upper_limit"], val["result"]) == ("0.25", "0.2", "OUT")
    assert _row(tmp_path, run_id)["reason"].startswith("deviation_mm.max 0.25 mm is outside")


def test_a_disposition_that_is_none_of_the_three_never_reads_pass(tmp_path):
    run_id, run_dir, doc = _job(tmp_path, max_dev=0.1, threshold=0.2)
    doc["disposition"] = "MAYBE"                            # a document from somewhere else
    (run_dir / "result.json").write_text(json.dumps(doc), encoding="utf-8")
    row, _ = eventlog.rows_of(run_dir, tmp_path)
    assert row["judgment"] == "PULL" and row["reason"] == "disposition 'MAYBE'"
    (run_dir / "result.json").write_text("{ not json", encoding="utf-8")
    assert eventlog.rows_of(run_dir, tmp_path) is None      # unreadable: no row, no error
    assert eventlog.rows_of(tmp_path / "nowhere", tmp_path) is None


def test_value_rows_for_tolerances_and_for_key_limit_checks(tmp_path, eventlog_on):
    """A check's tolerances give one row each; a check that reports only key / limit gives
    its headline value held to +-limit; a check that could not measure gives none."""
    zones = [
        {"id": "lid", "verdict": "FAIL", "checks": [
            _check("flatness", "FAIL", values={"flatness_mm": 0.31, "rms_mm": 0.04},
                   key="flatness_mm", limit=0.2, message="flatness 0.31 mm > 0.2 mm",
                   tolerances=[D.tolerance("flatness_mm", 0.31, hi=0.2),
                               D.tolerance("rms_mm", 0.04, hi=0.1)]),
            _check("height", "PASS", values={"height_mm": 12.02},
                   tolerances=[D.tolerance("height_mm", 12.02, 11.5, 12.5)])]},
        {"id": "edge", "verdict": "PASS", "checks": [
            _check("my_plugin", "PASS", values={"offset_mm": -0.4, "n_px": 812}, key="offset_mm",
                   limit=1.5),                                      # key / limit, no tolerances
            _check("coverage", "PASS", values={"part_pct": 97.0},
                   tolerances=[D.tolerance("part_pct", 97.0, lo=92, span=8)])]},
        {"id": "label", "verdict": "INVALID", "checks": [
            _check("code", "INVALID", message="no image to read", pull="incomplete")]}]
    run_id, _, doc = _job(tmp_path, "lid_recipe", result="FAIL", zones=zones,
                          job={"type": "recipe", "recipe": "lid_recipe", "config": None})
    assert doc["disposition"] == "REJECT"
    row, vals = _row(tmp_path, run_id), _values(tmp_path, run_id)
    assert [(v["zone"], v["check"], v["name"], v["value"], v["unit"], v["lower_limit"],
             v["upper_limit"], v["result"]) for v in vals] == [
        ("lid", "flatness", "flatness_mm", "0.31", "mm", "", "0.2", "OUT"),
        ("lid", "flatness", "rms_mm", "0.04", "mm", "", "0.1", "OK"),
        ("lid", "height", "height_mm", "12.02", "mm", "11.5", "12.5", "OK"),
        ("edge", "my_plugin", "offset_mm", "-0.4", "mm", "-1.5", "1.5", "OK"),
        ("edge", "coverage", "part_pct", "97", "%", "92", "", "OK")]
    assert all(v["step"] == "" for v in vals)
    assert (row["judgment"], row["n_values"], row["n_out_of_limit"]) == ("FAIL", "5", "1")
    assert row["reason"] == "lid/flatness: flatness 0.31 mm > 0.2 mm"
    assert row["program"] == "lid_recipe"                  # a recipe job: the recipe
    assert row["values"].startswith("flatness_mm=0.31 mm [..0.2] OUT; rms_mm=0.04 mm [..0.1] OK; "
                                    "height_mm=12.02 mm [11.5..12.5] OK; offset_mm=-0.4 mm "
                                    "[-1.5..1.5] OK")


def test_the_values_cell_names_whose_value_it_is_and_stops_after_twelve(tmp_path, eventlog_on):
    zones = [{"id": f"z{i}", "verdict": "PASS", "checks": [
        _check("height", "PASS", values={"height_mm": 1.0 + i},
               tolerances=[D.tolerance("height_mm", 1.0 + i, 0, 99)])]} for i in range(14)]
    run_id, _, _ = _job(tmp_path, zones=zones)
    row = _row(tmp_path, run_id)
    bits = row["values"].split("; ")
    assert bits[0] == "z0/height: height_mm=1 mm [0..99] OK" and len(bits) == 13
    assert bits[-1] == "(+2 more)" and row["n_values"] == "14"
    assert len(_values(tmp_path, run_id)) == 14             # the value rows hold all


# ================================================================== foreign text

@pytest.mark.parametrize("text, cell", [
    ("=HYPERLINK(\"http://x\",\"y\")", "'=HYPERLINK(\"http://x\",\"y\")"),
    ("+1 800 CALL", "'+1 800 CALL"), ("-x", "'-x"), ("@SUM(A1)", "'@SUM(A1)"), ("--5", "'--5"),
    ("=1+1", "'=1+1"), ("-", "-"), ("-0.5", "-0.5"), ("+3", "+3"), ("-1e-05", "-1e-05"),
    ("-.5", "-.5"), ("SN-0042", "SN-0042"), ("a = b", "a = b"), ("", ""),
    ("  two\r\nlines\tand a tab ", "two lines and a tab"), ("\t=1+1", "'=1+1"),
    (-0.25, "-0.25"), (3, "3"), (None, ""), (float("nan"), ""), (True, "yes"),
])
def test_a_cell_a_spreadsheet_would_run_gets_a_leading_apostrophe(text, cell):
    assert eventlog.cell(text) == cell
    assert eventlog.cell(cell) == cell                      # writing a cell again changes nothing


def test_a_code_read_from_a_part_is_foreign_text(tmp_path, eventlog_on):
    codes = ['=cmd|\' /C calc\'!A0', "@two, with a comma"]
    run_id, _, _ = _job(tmp_path, max_dev=0.1, threshold=0.2, codes=codes + [codes[0]],
                        part="-plate")
    row = _row(tmp_path, run_id)
    assert row["part_id"] == "'=cmd|' /C calc'!A0 | @two, with a comma"    # each code once
    assert row["part"] == "'-plate"
    raw = next((tmp_path / "events").glob("inspections_v1_*.csv")).read_text(encoding="utf-8-sig")
    assert len(raw.splitlines()) == 2                       # one line per row, whatever the text
    # half a character (a code decoded from bytes that are no text; UTF-8 cannot write it):
    # a ? in its place - the row is written all the same, and a rebuild does not stop at it
    run_id, _, _ = _job(tmp_path, max_dev=0.1, threshold=0.2, codes=["SN-\udcff-1"])
    assert _row(tmp_path, run_id)["part_id"] == "SN-?-1"
    out = eventlog.rebuild(tmp_path, [tmp_path])
    assert out["ok"] and _row(tmp_path, run_id)["part_id"] == "SN-?-1"
    assert not list((tmp_path / "events").glob("rebuild_*"))


def _code_check(monkeypatch, read, **params) -> dict:
    """The entry the recipe runner records for the real `code` check, with the scan's own
    picture `read` as these (kind, text) codes (the default: scan.codes is off, so the run
    log has no `codes` event)."""
    from engine.recipes.checks import builtin
    from engine.results import Code
    from engine.sense import codes as codes_mod
    monkeypatch.setattr(codes_mod, "decode_image", lambda img, illumination="": [
        Code(kind=k, data=d, corners=None, decoder="test") for k, d in read])
    zone = SimpleNamespace(id="label", mask=np.ones((40, 60), bool), centroid_px=lambda: [30, 20])
    ctx = SimpleNamespace(zone=zone, codes=None,
                          scan=SimpleNamespace(flood_white=np.zeros((40, 60), np.float32)))
    entry = runner._run_check(SimpleNamespace(check=builtin.Code(params), id="code", type="code"),
                              ctx)
    assert entry["details"]["source"] == "flood" and len(entry["details"]["codes"]) == len(read)
    return entry


def _label_zone(check) -> dict:
    return {"id": "label", "verdict": check["verdict"], "checks": [check]}


def test_part_id_is_the_code_a_check_read_from_the_scans_own_picture(tmp_path, eventlog_on,
                                                                     monkeypatch):
    """Without scan.codes (the default) nothing reads codes before the scan and the run log
    has no `codes` event: the `code` check decodes the scan's own picture and keeps what it
    read in result.json. That is the part id - the codes that match what the check asked for."""
    read = [("QRCode", "SN-0042"), ("EAN13", "4006381333931"), ("QR", "LOT-7")]
    job = {"type": "recipe", "recipe": "label_recipe", "config": None}
    check = _code_check(monkeypatch, read, kind="qr", pattern="^SN-")
    assert check["verdict"] == "PASS" and check["params"]["pattern"] == "^SN-"
    run_id, _, doc = _job(tmp_path, "label_recipe", zones=[_label_zone(check)], job=job)
    assert [c["data"] for c in doc["zones"][0]["checks"][0]["details"]["codes"]] == [
        "SN-0042", "4006381333931", "LOT-7"]
    assert _row(tmp_path, run_id)["part_id"] == "SN-0042"   # not the other codes on the part
    # a check that takes any code: every code it read, each once
    anything = _code_check(monkeypatch, read + [read[0]])
    run_id, _, _ = _job(tmp_path, zones=[_label_zone(anything)], job=job)
    assert _row(tmp_path, run_id)["part_id"] == "SN-0042 | 4006381333931 | LOT-7"
    # the exact text it expects, of the kind it expects
    exact = _code_check(monkeypatch, [("EAN_13", "LOT-7"), ("QR", "LOT-7"), ("QR", "LOT-70")],
                        expect="LOT-7", kind="QR")
    run_id, _, _ = _job(tmp_path, zones=[_label_zone(exact)], job=job)
    assert _row(tmp_path, run_id)["part_id"] == "LOT-7"
    # the wrong code, or none: the check says so (FAIL / PULL); it is no part id
    wrong = _code_check(monkeypatch, [("QR", "XX-1")], pattern="^SN-")
    assert wrong["verdict"] == "FAIL" and wrong["pull"] is None
    run_id, _, _ = _job(tmp_path, result="FAIL", zones=[_label_zone(wrong)], job=job)
    row = _row(tmp_path, run_id)
    assert (row["part_id"], row["judgment"]) == ("", "FAIL") and "'XX-1'" in row["reason"]
    # codes read before the scan (scan.codes: the run log's event) come first, as before
    run_id, _, _ = _job(tmp_path, zones=[_label_zone(check)], job=job, codes=["PRE-1"])
    assert _row(tmp_path, run_id)["part_id"] == "PRE-1"
    # a document with something else where the codes should be: no part id, no error
    odd = dict(check, details={"codes": "SN-0042"}, params=None)
    run_id, _, _ = _job(tmp_path, zones=[_label_zone(odd)], job=job)
    assert _row(tmp_path, run_id)["part_id"] == ""
    # a cycle: the code its work state's check read
    cid, _, _ = _cycle(tmp_path / "line", codes_in=None,
                       more_zones={"aligned": [_label_zone(check)]})
    assert _row(tmp_path / "line", cid)["part_id"] == "SN-0042"
    # ... and the rebuilt rows say the same
    for runs in (tmp_path, tmp_path / "line"):
        live = {r["inspection_id"]: r for r in _log(runs)[0]}
        assert eventlog.rebuild(runs, [runs])["ok"]
        assert {r["inspection_id"]: r for r in _log(runs)[0]} == live and len(live) >= 1


# ================================================================== a cycle (hand-built)

WORK_STATES = ["loaded", "aligned", "end_face"]


def _cycle_steps(runs, *, kick=None, codes_in="aligned", pictures=True, upto="end_face",
                 more_zones=None) -> tuple:
    """A cycle folder as engine/flow/verify.py writes it, without scans, BEFORE the cycle
    ends: the cycle's log (open, as while the cycle runs) and one run folder per work state
    under steps/ (written through the hook), up to and with `upto`. more_zones: {work state:
    zones its check also has}. -> (cycle id, cycle folder, the open log, the steps of the
    cycle's document, [(work state, run id)])."""
    cid, cdir = R.new_run(runs, "line1")
    clog = R.RunLog(cdir / "run.log.jsonl", cid)
    clog.event("run_start", station="line1", mode="MOCK", job="flow", config="line1.json",
               profile="long", work_states=WORK_STATES)
    steps, rows = [], []
    for sid, value, lo, hi in (("loaded", 62.0, 60, 64), ("aligned", 0.4, None, 1.5),
                               ("end_face", 0.35 if kick else 0.02, None, 0.15)):
        bad = kick == sid
        zones = [{"id": "seat", "verdict": "FAIL" if bad else "PASS", "checks": [
            _check("gauge", "FAIL" if bad else "PASS", values={"v_mm": value},
                   message=f"{sid}: v {value} mm",
                   tolerances=[D.tolerance("v_mm", value, lo, hi)])]}]
        zones += (more_zones or {}).get(sid, [])
        rid, rdir, sub = _job(cdir / FV.STEPS_DIR, f"line1.{sid}", result="FAIL" if bad else "PASS",
                              zones=zones, codes=["LOT-7/0042"] if sid == codes_in else None,
                              picture="overlay.png" if pictures else None,
                              job={"type": "recipe", "recipe": f"r_{sid}", "config": "line1.json",
                                   "work_state": sid}, station_config="line1.json", profile="long")
        if sid == "end_face":
            (rdir / "profile_x.csv").write_text("position_mm,height_mm\n0.000,0.01000\n")
        steps.append({"id": sid, "title": sid, "outcome": "DEFECT" if bad else "CONFIRMED",
                      "dlp_result": {"run_id": rid, "run_folder": str(rdir),
                                     "result": sub["result"], "disposition": sub["disposition"]},
                      "disposition": sub["disposition"]})
        rows.append((sid, rid))
        if bad or sid == upto:
            break
    return cid, cdir, clog, steps, rows


def _cycle(runs, *, kick=None, **kw) -> tuple:
    """A cycle that ended: _cycle_steps, then the cycle's own result.json (through the hook).
    -> (cycle id, cycle folder, [(work state, run id)])."""
    cid, cdir, clog, steps, rows = _cycle_steps(runs, kick=kick, **kw)
    doc = R.empty_result(run_id=cid, run_folder=str(cdir), mock=True, operator_confirmed=False,
                         station="line1", job={"type": "flow", "config": "line1.json",
                                               "work_states": ["loaded", "aligned", "end_face"]},
                         part={"role": "test", "label": "long"}, station_config="line1.json",
                         profile="long")
    doc.update(steps=steps, result="FAIL" if kick else "PASS", invalid_reason=None,
               disposition="REJECT" if kick else "CONTINUE", pull_reasons=[],
               cycle={"ended_by": "kickout" if kick else "flow", "kickout": None if not kick else {
                   "step": kick, "disposition": "REJECT", "reason": "end face not OK",
                   "detail": "seat/gauge: v 0.35 mm", "by": "kickout"}})
    R.write_result(cdir, doc)
    clog.event("run_end", result=doc["result"])
    clog.close()
    return cid, cdir, rows


def test_a_cycle_with_three_work_states_is_one_row_plus_its_value_rows(tmp_path, eventlog_on):
    runs = tmp_path / "runs"
    cid, cdir, steps = _cycle(runs)
    rows, vals = _log(runs)
    assert [r["inspection_id"] for r in rows] == [cid]      # the three checks are not rows
    row = rows[0]
    assert (row["judgment"], row["program"], row["part"]) == ("PASS", "line1.json / long", "long")
    assert row["part_id"] == "LOT-7/0042"                   # read by a work state, not the cycle
    assert [(v["inspection_id"], v["step"], v["zone"], v["check"], v["name"], v["value"],
             v["lower_limit"], v["upper_limit"], v["result"]) for v in vals] == [
        (cid, "loaded", "seat", "gauge", "v_mm", "62", "60", "64", "OK"),
        (cid, "aligned", "seat", "gauge", "v_mm", "0.4", "", "1.5", "OK"),
        (cid, "end_face", "seat", "gauge", "v_mm", "0.02", "", "0.15", "OK")]
    assert (row["n_values"], row["n_out_of_limit"]) == ("3", "0")
    assert row["values"].startswith("loaded/seat/gauge: v_mm=62 mm [60..64] OK; aligned/")
    # the row's picture, profile and facts are those of the last work state checked
    last = cdir / "steps" / steps[-1][1]
    assert row["image_file"] == f"events/img/{cid}.png"
    assert row["profile_file"] == f"{cid}/steps/{last.name}/profile_x.csv"
    assert row["result_file"] == f"{cid}/result.json"
    assert (row["preset"], row["calibration_id"], row["rig"]) == ("p1", "rig_a/cal_1", "rig_a")
    assert sorted(p.name for p in (runs / "events" / "img").iterdir()) == [f"{cid}.png"]
    # a check of a work state inside a cycle is told by where it lies, not by what it says
    for _, rid in steps:
        sub = json.loads((cdir / "steps" / rid / "result.json").read_text(encoding="utf-8"))
        assert eventlog.in_cycle(cdir / "steps" / rid, sub)
        assert eventlog.rows_of(cdir / "steps" / rid, runs) is None


def test_a_cycle_that_was_kicked_out_says_where_and_why(tmp_path, eventlog_on):
    runs = tmp_path / "runs"
    cid, cdir, steps = _cycle(runs, kick="end_face", codes_in=None)
    row = _row(runs, cid)
    assert row["judgment"] == "FAIL" and row["part_id"] == ""
    assert row["reason"] == "end_face: end face not OK - seat/gauge: v 0.35 mm"
    assert row["n_out_of_limit"] == "1" and row["values"].endswith("v_mm=0.35 mm [..0.15] OUT")
    assert row["profile_file"].endswith(f"{steps[-1][1]}/profile_x.csv")


def test_a_work_state_check_outside_a_cycle_is_its_own_row(tmp_path, eventlog_on):
    """The same check, run on its own (no cycle folder above it): an inspection."""
    zones = [{"id": "seat", "verdict": "PASS", "checks": [
        _check("gauge", "PASS", values={"v_mm": 62.0},
               tolerances=[D.tolerance("v_mm", 62.0, 60, 64)])]}]
    job = {"type": "recipe", "recipe": "r_loaded", "config": "line1.json", "work_state": "loaded"}
    for runs in (tmp_path / "runs", tmp_path / "steps"):    # also in a folder that is only named so
        run_id, run_dir, doc = _job(runs, "line1.loaded", zones=zones, job=job, profile="long",
                                    station_config="line1.json")
        assert not eventlog.in_cycle(run_dir, doc)
        row = _row(runs, run_id)
        assert row["program"] == "line1.json / long / loaded" and row["judgment"] == "PASS"
        assert [v["step"] for v in _values(runs, run_id)] == [""]


# ================================================================== after the result

ABORTED = ("ABORT was pressed after this run wrote its verdict (PASS / CONTINUE in result.json): "
           "the cycle is ABORTED on the line and the part is PULLed")


def _abort_after_verdict(run_dir, run_id) -> None:
    """What the web server does when ABORT comes after a run wrote its verdict: result.json
    is never rewritten, the run's closed log gets one more event (engine/runs.append_event,
    which also has the event log write the run's row again)."""
    R.append_event(run_dir / "run.log.jsonl", run_id, "aborted_after_verdict", level="warning",
                   result="ABORTED", disposition="PULL", message=ABORTED)


def test_a_run_aborted_after_its_verdict_reads_pull(tmp_path, eventlog_on, monkeypatch):
    """The line counts a part PULLed when ABORT is pressed after its run wrote PASS: the job's
    result, the page and the outputs say so, and the run's log records it. The event log must
    not be the one record that still says PASS: the row is written again when the event is
    recorded (the last row of an inspection counts), and a rebuild gives that row."""
    run_id, run_dir, doc = _job(tmp_path, max_dev=0.1, threshold=0.2, codes=["SN-7"],
                                picture="overlay.png")
    before = (run_dir / "result.json").read_bytes()
    assert _row(tmp_path, run_id)["judgment"] == "PASS"
    R.append_event(run_dir / "run.log.jsonl", run_id, "note", message="any other event")
    assert _row(tmp_path, run_id)["judgment"] == "PASS"     # ... writes no row
    _abort_after_verdict(run_dir, run_id)
    assert (run_dir / "result.json").read_bytes() == before         # never rewritten
    first, last = [r for r in _log(tmp_path)[0] if r["inspection_id"] == run_id]
    assert (first["judgment"], first["reason"]) == ("PASS", "")
    assert (last["judgment"], last["reason"]) == ("PULL", f"aborted: {ABORTED}")
    same = [k for k in INSPECTION_COLUMNS if k not in ("judgment", "reason")]
    assert [last[k] for k in same] == [first[k] for k in same]      # the same inspection
    assert len(_values(tmp_path, run_id)) == 1              # its value row is there once
    (newest,) = eventlog.read_rows(tmp_path / "events" / "last_v1.csv")
    assert newest == last
    # rebuilt from the run folder: one row, the PULL
    out = eventlog.rebuild(tmp_path, [tmp_path])
    assert out["ok"] and _row(tmp_path, run_id) == last and len(_values(tmp_path, run_id)) == 1
    # the run folder cleaned away while the log held both rows: the last one is kept
    other, other_dir, _ = _job(tmp_path, result="FAIL", max_dev=0.3, threshold=0.2)
    _abort_after_verdict(other_dir, other)
    assert [r["judgment"] for r in _log(tmp_path)[0] if r["inspection_id"] == other] == [
        "FAIL", "PULL"]                                     # a FAIL too: nobody took the part
    shutil.rmtree(other_dir)
    out = eventlog.rebuild(tmp_path, [tmp_path])
    assert out["ok"] and out["kept"] == 1 and _row(tmp_path, other)["judgment"] == "PULL"
    # a run that was a PULL already keeps its own reason and counts the abort
    third, third_dir, _ = _job(tmp_path, max_dev=0.195, threshold=0.2, settle=_guard_band)
    _abort_after_verdict(third_dir, third)
    row = [r for r in _log(tmp_path)[0] if r["inspection_id"] == third][-1]
    assert row["judgment"] == "PULL" and row["reason"].startswith("guard_band: deviation")
    assert row["reason"].endswith("(+1 more)")
    # the one who counts the run ABORTED may hold a lock of its own: with the log busy the
    # row does not wait for long - it waits in events/pending/, and is still the last row
    fourth, fourth_dir, _ = _job(tmp_path, max_dev=0.1, threshold=0.2)
    with eventlog._locked(tmp_path / "events") as held:
        t0 = time.perf_counter()
        _abort_after_verdict(fourth_dir, fourth)
        assert held and time.perf_counter() - t0 < eventlog.LOCK_WAIT_S
        assert eventlog.waiting(tmp_path) == 1
    out = eventlog.rebuild(tmp_path, [tmp_path])            # (the waiting row is taken in)
    assert out["ok"] and _row(tmp_path, fourth)["judgment"] == "PULL"
    # an event that brings no text still has a reason
    R.append_event(run_dir / "run.log.jsonl", run_id, "aborted_after_verdict")
    assert [r["reason"] for r in _log(tmp_path)[0] if r["inspection_id"] == run_id][-1] == (
        "aborted: the line counted the run ABORTED after its verdict was written")
    # whatever goes wrong in the log, the one who records the event sees nothing of it
    monkeypatch.setattr(eventlog, "rows_of", lambda *a, **k: 1 / 0)
    _abort_after_verdict(third_dir, third)


def test_abort_after_the_verdict_over_the_command_bus_reads_pull_in_the_log(
        tmp_path, eventlog_on, monkeypatch):
    """The whole way, on a MOCK server (tests/test_disposition_mock.py has the same case for
    the bus, the job and the run log): START, the run writes PASS, ABORT comes before its job
    ends. The line PULLs the part - and so does the event log, without a rebuild."""
    from test_disposition_mock import _wait, _web_app
    app = _web_app(tmp_path)
    lib = tmp_path / "lib"
    go, made = threading.Event(), {}

    def fake_cmd_run(ctx):                  # the run is past its last cancel check
        assert go.wait(30)
        run_id, run_dir, doc = _job(lib / "runs", max_dev=0.1, threshold=0.2, codes=["SN-9"])
        made.update(run_id=run_id, doc=doc)
        return 0, {"run_dir": str(run_dir), "result": doc}
    monkeypatch.setattr(cli, "cmd_run", fake_cmd_run)
    try:
        assert app.bus.command("START").accepted
        (jid,) = app.bus_jobs()
        _wait(lambda: app._jobs[jid].status == "running", "the run never started")
        assert app.bus.command("ABORT").accepted and app.bus.state == "ABORTING"
        go.set()
        _wait(lambda: app.bus.state == "READY", "the bus never settled")
        job = app._jobs[jid]
        _wait(lambda: (job.result or {}).get("disposition") == "PULL", "job result not PULL")
        assert (made["doc"]["result"], made["doc"]["disposition"]) == ("PASS", "CONTINUE")
        first, last = [r for r in _log(lib)[0] if r["inspection_id"] == made["run_id"]]
        assert (first["judgment"], last["judgment"], last["part_id"]) == ("PASS", "PULL", "SN-9")
        assert last["reason"].startswith("aborted: ABORT was pressed after this run wrote its "
                                         "verdict (PASS / CONTINUE in result.json)")
        assert eventlog.read_rows(lib / "events" / "last_v1.csv") == [last]
    finally:
        go.set()
        app.close()
        api._session = None


def test_a_cycle_that_never_ended_is_a_pull_row_after_a_rebuild(tmp_path, eventlog_on,
                                                                monkeypatch, capsys):
    """The program is closed (or dies) while a cycle waits for its next work state: the
    cycle's result.json is never written, so no hook writes its row - although its work
    states were checked. rebuild lists it: PULL, why, and the values of the work states it
    did check. A cycle that may still be running is left alone."""
    runs = tmp_path / "runs"
    done, _, _ = _cycle(runs)                               # one that ended, as before
    cid, cdir, clog, _, steps = _cycle_steps(runs, upto="aligned")
    assert not (cdir / "result.json").exists() and len(steps) == 2
    assert [r["inspection_id"] for r in _log(runs)[0]] == [done]    # no hook wrote a row
    # still running: its log is open for writing (where that cannot be told: written just now)
    assert eventlog.unended_cycle(cdir) is None and eventlog.rows_of(cdir, runs) is None
    out = eventlog.rebuild(runs, [runs])
    assert out["ok"] and (out["inspections"], out["unended"]) == (1, 0)
    clog.close()                                            # the program is gone
    monkeypatch.setattr(eventlog, "UNENDED_QUIET_S", 0.0)   # (not Windows: no hour to wait)
    assert eventlog.main(["--rebuild", "--library", str(runs), "--runs", str(runs),
                          "--stations", str(tmp_path / "none"), "--recipes",
                          str(tmp_path / "none")]) == 0
    assert "1 cycle(s) that never ended listed as PULL" in capsys.readouterr().out
    rows, vals = _log(runs)
    assert sorted(r["inspection_id"] for r in rows) == sorted([done, cid])
    row = _row(runs, cid)
    assert row["judgment"] == "PULL" and row["reason"].startswith(
        "error: the cycle never ended: no result.json was written")
    assert (row["program"], row["part"], row["hardware"]) == ("line1.json / long", "long", "MOCK")
    assert row["operating_mode"] == "station"
    assert row["part_id"] == "LOT-7/0042"                   # read by the work state `aligned`
    assert [(v["step"], v["name"], v["value"], v["result"]) for v in vals
            if v["inspection_id"] == cid] == [("loaded", "v_mm", "62", "OK"),
                                              ("aligned", "v_mm", "0.4", "OK")]
    assert (row["n_values"], row["n_out_of_limit"]) == ("2", "0")
    last = cdir / "steps" / steps[-1][1]
    last_doc = json.loads((last / "result.json").read_text(encoding="utf-8"))
    assert LOCAL_TIME.fullmatch(row["time"])
    assert datetime.fromisoformat(row["time"]) >= datetime.fromisoformat(last_doc["finished"])
    assert eventlog.month_of(row["time"]) == _month(last_doc)
    assert float(row["time_s"]) >= 0.0
    assert row["result_file"] == "" and row["image_file"] == f"events/img/{cid}.png"
    assert (row["preset"], row["calibration_id"], row["rig"]) == ("p1", "rig_a/cal_1", "rig_a")
    assert row["software"] == " ".join(str(v) for v in last_doc["software"].values() if v)
    # the same row at every rebuild; the work-state checks stay rows of nobody else
    out = eventlog.rebuild(runs, [runs])
    assert out["ok"] and (out["inspections"], out["unended"], out["kept"]) == (2, 1, 0)
    assert _row(runs, cid) == row
    # a work state that FAILed does not make it a FAIL: nobody saw the cycle end
    bad, bad_dir, bad_log, _, _ = _cycle_steps(runs, kick="end_face", codes_in=None)
    bad_log.close()
    row, vals = eventlog.rows_of(bad_dir, runs)
    assert (row["judgment"], row["n_values"], row["n_out_of_limit"]) == ("PULL", 3, 1)
    assert [v["step"] for v in vals] == WORK_STATES and vals[-1]["result"] == "OUT"
    # no cycle: a run folder without a result (a run that crashed) has no steps to list
    _, lone = R.new_run(runs, "lone")
    R.RunLog(lone / "run.log.jsonl", lone.name).close()
    (lone / "steps").mkdir()
    assert eventlog.unended_cycle(lone) is None and eventlog.rows_of(lone, runs) is None


@pytest.mark.skipif(sys.platform == "win32", reason="Windows tells a log that is still open")
def test_where_an_open_log_cannot_be_told_a_quiet_hour_decides(tmp_path, eventlog_on,
                                                               monkeypatch):
    cid, cdir, clog, _, _ = _cycle_steps(tmp_path, upto="loaded")
    clog.close()
    assert eventlog.unended_cycle(cdir) is None             # written a moment ago
    monkeypatch.setattr(eventlog, "UNENDED_QUIET_S", 0.0)
    assert eventlog.unended_cycle(cdir)["run_id"] == cid


@pytest.mark.skipif(sys.platform != "win32", reason="the share mode of an open log")
def test_a_log_that_is_open_for_writing_is_told(tmp_path):
    path = tmp_path / "run.log.jsonl"
    assert eventlog._being_written(path) is None            # no such file: not told
    with open(path, "x", encoding="utf-8") as f:
        f.write("{}\n")
        f.flush()
        assert eventlog._being_written(path) is True
        with open(path, encoding="utf-8") as reader:        # a reader is no writer
            assert reader.read() == "{}\n" and eventlog._being_written(path) is True
    assert eventlog._being_written(path) is False
    with open(path, encoding="utf-8"):
        assert eventlog._being_written(path) is False


# ================================================================== the files

def test_one_file_pair_per_month_with_v1_in_the_name(tmp_path, eventlog_on):
    lib = tmp_path / "library"
    for i, when in enumerate(("2026-01-31T23:59:59+00:00", "2026-02-01T00:00:00+00:00",
                              "2026-02-14T08:00:00+00:00")):
        out = eventlog.append(lib, {"time": when, "inspection_id": f"run{i}", "judgment": "PASS"},
                              [{"inspection_id": f"run{i}", "name": "v", "value": float(i)}])
        assert out == {"written": 2, "pending": 0, "lost": 0}
    assert sorted(p.name for p in (lib / "events").glob("*.csv")) == [
        "inspection_values_v1_2026-01.csv", "inspection_values_v1_2026-02.csv",
        "inspections_v1_2026-01.csv", "inspections_v1_2026-02.csv", "last_v1.csv"]
    assert [r["inspection_id"] for r in _log(lib, "2026-01")[0]] == ["run0"]
    assert [r["inspection_id"] for r in _log(lib, "2026-02")[0]] == ["run1", "run2"]
    assert [v["value"] for v in _log(lib, "2026-02")[1]] == ["1", "2"]
    assert eventlog.FORMAT == "v1" and eventlog.month_of("2026-02-14T08:00:00+00:00") == "2026-02"
    assert re.fullmatch(r"\d{4}-\d{2}", eventlog.month_of(None))    # no time: this month
    # a file whose last line was cut short (a crash mid-write): the next rows stay whole
    f = lib / "events" / "inspections_v1_2026-02.csv"
    with open(f, "ab") as fh:
        fh.write(b"2026-02-15T00:00:00+00:00,cut")
    eventlog.append(lib, {"time": "2026-02-16T00:00:00+00:00", "inspection_id": "run3"}, [])
    assert [r["inspection_id"] for r in _log(lib, "2026-02")[0]] == ["run1", "run2", "cut", "run3"]


def test_a_file_written_before_the_v1_header_settled_is_brought_to_it(tmp_path, eventlog_on,
                                                                      caplog):
    """The files of the first real-rig day have `mode` (REAL / MOCK) and no `operating_mode`.
    A rebuild reads them by their own header; the first row appended to one brings it to the
    v1 header - its rows kept cell for cell, the file as it was moved aside, never a row under
    the wrong column - and so does a waiting file when it is merged. Any other header (a
    file changed in a spreadsheet program) is never appended to either: the file is moved
    aside as it is and its rows go on, by their column names, in a new file."""
    lib = tmp_path / "library"
    events = lib / "events"
    (events / "pending").mkdir(parents=True)
    old = {"time": "2026-10-13T11:28:00+00:00", "inspection_id": "old1", "program": "p",
           "mode": "REAL", "judgment": "PULL", "reason": "rig_check: MOVED"}
    raw = eventlog._BOM + eventlog._encode(tuple(EARLY_V1_COLUMNS), [old], header=True)
    month_file = events / "inspections_v1_2026-10.csv"
    month_file.write_bytes(raw)
    # a rebuild with no run folder keeps the row, read by its own header, under v1
    out = eventlog.rebuild(lib, [tmp_path / "no_runs"])
    assert out["ok"] and out["kept"] == 1
    (row,) = eventlog.read_rows(month_file)
    assert list(row) == INSPECTION_COLUMNS
    assert (row["hardware"], row["judgment"]) == ("REAL", "PULL")
    # the first append to an early file brings it to v1 first; a waiting early file too
    month_file.write_bytes(raw)
    eventlog._ENDED.clear()
    waiting = events / "pending" / month_file.name
    waiting.write_bytes(eventlog._BOM + eventlog._encode(tuple(EARLY_V1_COLUMNS), [
        {**old, "inspection_id": "old2", "mode": "MOCK", "judgment": "FAIL"}], header=True))
    new = {"time": "2026-10-13T16:00:00+02:00", "inspection_id": "new1",
           "operating_mode": "station", "hardware": "MOCK", "judgment": "PASS"}
    assert eventlog.append(lib, new, []) == {"written": 1, "pending": 0, "lost": 0}
    rows = eventlog.read_rows(month_file)
    assert list(rows[0]) == INSPECTION_COLUMNS
    assert [(r["inspection_id"], r["hardware"], r["operating_mode"], r["judgment"])
            for r in rows] == [("old1", "REAL", "", "PULL"), ("old2", "MOCK", "", "FAIL"),
                               ("new1", "MOCK", "station", "PASS")]
    assert not waiting.exists() and eventlog.waiting(lib) == 0
    # the file as it was, twice: moved aside by the rebuild, then by the append (never deleted)
    assert [p.read_bytes() for p in events.glob("replaced_*/inspections_v1_2026-10.csv")] \
        == [raw, raw]
    assert "before the v1 header was settled" in caplog.text
    # a file with a header of its own is never added to: it is moved aside as it is, and the
    # month's rows go on in a new file
    foreign = events / "inspections_v1_2026-09.csv"
    foreign.write_bytes(b"a,b\r\n1,2\r\n")
    out = eventlog.append(lib, {"time": "2026-09-01T00:00:00+00:00", "inspection_id": "x"}, [])
    assert out == {"written": 1, "pending": 0, "lost": 0}
    assert [r["inspection_id"] for r in eventlog.read_rows(foreign)] == ["x"]
    assert [p.read_bytes() for p in events.glob("replaced_*/inspections_v1_2026-09.csv")] \
        == [b"a,b\r\n1,2\r\n"]
    assert "had a header that is not the v1 header (a, b)" in caplog.text
    # a file changed in a spreadsheet program (a column added, saved): its rows go on under
    # the v1 header, the file as it was is kept - the log never stops growing
    rows = eventlog.read_rows(month_file)
    edited = eventlog._BOM + eventlog._encode((*INSPECTION_COLUMNS, "qa_note"),
                                              [{**r, "qa_note": "ok"} for r in rows], header=True)
    month_file.write_bytes(edited)
    new2 = {**new, "inspection_id": "new2"}
    assert eventlog.append(lib, new2, []) == {"written": 1, "pending": 0, "lost": 0}
    assert [r["inspection_id"] for r in eventlog.read_rows(month_file)] == [
        "old1", "old2", "new1", "new2"]
    assert list(eventlog.read_rows(month_file)[0]) == INSPECTION_COLUMNS
    assert edited in [p.read_bytes() for p in events.glob("replaced_*/inspections_v1_2026-10.csv")]
    assert "was changed in another program (column(s) added: qa_note)" in caplog.text


def test_the_newest_row_is_also_a_small_file_of_its_own(tmp_path, eventlog_on, monkeypatch,
                                                        caplog):
    """events/last_v1.csv: the header and the row of the inspection that just ended, for a
    reader that wants that one row (the operator page's Part ID) and not a month of them. It
    is replaced in one move: never half a file, nothing left behind."""
    runs = tmp_path / "runs"
    events, last = runs / "events", runs / "events" / "last_v1.csv"
    assert eventlog.LAST == last.name
    first, _, doc = _job(runs, max_dev=0.1, threshold=0.2, codes=["SN-1"])
    month = _month(doc)
    raw = last.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf" + ",".join(INSPECTION_COLUMNS).encode() + b"\r\n")
    assert raw.endswith(b"\r\n") and raw.count(b"\r\n") == 2 and len(raw) < 2000
    assert eventlog.read_rows(last) == [_row(runs, first)]
    second, _, _ = _job(runs, max_dev=0.3, threshold=0.2, result="FAIL", codes=["=SN-2"])
    (row,) = eventlog.read_rows(last)                       # replaced, not appended to
    assert (row["inspection_id"], row["part_id"], row["judgment"]) == (second, "'=SN-2", "FAIL")
    # a row that has to wait is the newest row all the same
    with _open_elsewhere(monkeypatch, f"inspections_v1_{month}.csv"):
        third, _, _ = _job(runs, max_dev=0.1, threshold=0.2)
    assert [r["inspection_id"] for r in eventlog.read_rows(last)] == [third]
    assert eventlog.waiting(runs) == 1
    # someone reads it this very instant (Windows refuses the move then): the inspection and
    # its row are what they would have been, and no half-written file is left behind
    with open(last, "rb") as reader:
        fourth, _, doc4 = _job(runs, max_dev=0.1, threshold=0.2)
        assert reader.read().count(b"\r\n") == 2
    assert doc4["disposition"] == "CONTINUE"
    assert [r["inspection_id"] for r in _log(runs)[0]] == [first, second, third, fourth]
    assert eventlog.read_rows(last)[0]["inspection_id"] in (third, fourth)
    if sys.platform == "win32":
        assert "last_v1.csv not written" in caplog.text
    assert sorted(p.name for p in events.iterdir() if p.is_file()) == [
        "eventlog.lock", f"inspection_values_v1_{month}.csv", f"inspections_v1_{month}.csv",
        "last_v1.csv", "pending.lock"]
    # it is no month's file: a rebuild neither reads it as one nor moves it away
    out = eventlog.rebuild(runs, [runs])
    assert out["ok"] and out["inspections"] == 4 and last.is_file()
    (aside,) = list(events.glob("replaced_*"))
    assert sorted(p.name for p in aside.iterdir()) == [
        f"inspection_values_v1_{month}.csv", f"inspections_v1_{month}.csv"]


def test_names_the_log_shares_with_the_run_folders():
    assert eventlog.STEPS_DIR == FV.STEPS_DIR
    assert eventlog.RESULT_FILE == MM.RESULT_FILE and eventlog.LOG_FILE == MM.LOG_FILE
    assert eventlog.PICTURES[0] == runner.OVERLAY_IMAGE == MM.OVERLAY_FILE
    assert eventlog.ENV == EVENTLOG_ENV


def test_the_log_is_the_sessions_library_else_beside_the_run_folders(eng, tmp_path):
    """One log per library: wherever a run folder lies (beside a station file, where
    $SCANNER_RUNS_DIR points), its row goes to the library of the configured session."""
    assert eventlog.library_of(tmp_path / "line" / "runs" / "run1") == eng.store.root
    api._session = None                     # no session (a script that only writes results)
    assert eventlog.library_of(tmp_path / "line" / "runs" / "run1") == tmp_path / "line" / "runs"


# ================================================================== a failed append

def test_a_failed_append_changes_nothing_waits_and_is_merged_next_time(tmp_path, eventlog_on,
                                                                       monkeypatch):
    runs = tmp_path / "runs"
    first, _, doc1 = _job(runs, max_dev=0.1, threshold=0.2)
    month = _month(doc1)
    name = f"inspections_v1_{month}.csv"
    main, pending = runs / "events" / name, runs / "events" / "pending" / name
    before = main.read_bytes()
    with _open_elsewhere(monkeypatch, name):
        second, run_dir, doc2 = _job(runs, max_dev=0.3, threshold=0.2, result="FAIL")
        # the run is what it would have been: verdict, disposition, result.json, exit code
        assert (doc2["result"], doc2["disposition"], D.exit_code(doc2)) == ("FAIL", "REJECT", 1)
        assert set(doc2) == RESULT_KEYS and not list(run_dir.glob("*.tmp"))
        assert main.read_bytes() == before                  # the file was not touched
        (waiting,) = eventlog.read_rows(pending)
        assert waiting["inspection_id"] == second and waiting["judgment"] == "FAIL"
        assert eventlog.waiting(runs) == eventlog.waiting(runs, month) == 1
        assert eventlog.waiting(runs, "1999-01") == 0
        # its values went where they could: the values file was not open
        assert [v["inspection_id"] for v in _log(runs)[1]] == [first, second]
        third, _, _ = _job(runs, max_dev=0.1, threshold=0.2)        # still open: it waits too
        assert [r["inspection_id"] for r in eventlog.read_rows(pending)] == [second, third]
        assert eventlog.main(["--merge", "--library", str(runs)]) == 1      # still waiting
    fourth, _, _ = _job(runs, max_dev=0.1, threshold=0.2)   # closed again: merged, then appended
    assert [r["inspection_id"] for r in _log(runs)[0]] == [first, second, third, fourth]
    assert not pending.exists() and eventlog.waiting(runs) == 0
    assert list((runs / "events" / "pending").iterdir()) == []
    assert main.read_bytes().count(b"time,inspection_id") == 1      # one header, no second


def test_waiting_rows_are_merged_on_request_too(tmp_path, eventlog_on, monkeypatch, capsys):
    lib = tmp_path / "library"
    lib.mkdir()
    row = {"time": "2026-03-01T10:00:00+00:00", "inspection_id": "r1", "judgment": "PULL"}
    with _open_elsewhere(monkeypatch, "inspections_v1_2026-03.csv",
                         "inspection_values_v1_2026-03.csv"):
        out = eventlog.append(lib, row, [{"inspection_id": "r1", "name": "v", "value": 1.0}])
    assert out == {"written": 0, "pending": 2, "lost": 0} and _log(lib) == ([], [])
    assert eventlog.main(["--merge", "--library", str(lib)]) == 0
    assert "2 file(s) of waiting rows merged; 0 row(s) still waiting" in capsys.readouterr().out
    rows, vals = _log(lib)
    assert [r["inspection_id"] for r in rows] == ["r1"] and [v["name"] for v in vals] == ["v"]


@pytest.mark.skipif(sys.platform != "win32", reason="the share mode of a spreadsheet program")
def test_a_file_held_as_a_spreadsheet_program_holds_it(tmp_path, eventlog_on):
    """No stand-in: the month's file is really open with a share mode that lets others read
    and nobody write. The run is unchanged, its row waits, and is merged once it is closed."""
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateFileW.restype = wintypes.HANDLE
    k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                                wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    runs = tmp_path / "runs"
    first, _, doc = _job(runs, max_dev=0.1, threshold=0.2)
    name = f"inspections_v1_{_month(doc)}.csv"
    main = runs / "events" / name
    GENERIC_READ, SHARE_READ, OPEN_EXISTING = 0x80000000, 0x1, 3
    h = k32.CreateFileW(str(main), GENERIC_READ, SHARE_READ, None, OPEN_EXISTING, 0x80, None)
    assert h not in (None, wintypes.HANDLE(-1).value), ctypes.get_last_error()
    try:
        with pytest.raises(PermissionError):
            open(main, "ab")
        second, _, doc2 = _job(runs, max_dev=0.1, threshold=0.2)
        assert doc2["disposition"] == "CONTINUE"
        assert [r["inspection_id"] for r in _log(runs)[0]] == [first]       # readable, unchanged
        assert eventlog.waiting(runs) == 1
        # rebuild refuses to move a file that is open, and changes nothing
        out = eventlog.rebuild(runs, [runs])
        assert out["ok"] is False and "close the program" in out["error"]
        assert eventlog.waiting(runs) == 1 and not list((runs / "events").glob("re*_*"))
    finally:
        k32.CloseHandle(h)
    third, _, _ = _job(runs, max_dev=0.1, threshold=0.2)
    assert [r["inspection_id"] for r in _log(runs)[0]] == [first, second, third]
    assert eventlog.waiting(runs) == 0


def test_whatever_fails_in_the_log_the_run_is_unchanged(tmp_path, eventlog_on, monkeypatch,
                                                        caplog):
    """The hook never raises: a bug in the log, a full disk, no lock - the result is written
    as before and the caller sees nothing."""
    def boom(*a, **k):
        raise RuntimeError("the log has a bug")
    with monkeypatch.context() as m:
        m.setattr(eventlog, "rows_of", boom)
        run_id, run_dir, doc = _job(tmp_path, max_dev=0.1, threshold=0.2)
        assert doc["disposition"] == "CONTINUE" and eventlog.record(run_dir) is None
    assert "the log has a bug" in caplog.text and not (tmp_path / "events").exists()
    # neither lock can be had (nothing may be written anywhere): the run is unchanged, the
    # hook says the rows are written nowhere, and rebuild brings them back
    monkeypatch.setattr(eventlog, "LOCK_WAIT_S", 0.05)
    with eventlog._locked_all(tmp_path / "events", 1.0) as held:
        assert held
        run2, run_dir2, doc2 = _job(tmp_path, max_dev=0.1, threshold=0.2)
        assert doc2["disposition"] == "CONTINUE"
        assert eventlog.record(run_dir2) == {"written": 0, "pending": 0, "lost": 2}
    assert _log(tmp_path) == ([], []) and eventlog.waiting(tmp_path) == 0
    assert not (tmp_path / "events" / "last_v1.csv").exists()
    assert "no lock" in caplog.text and "waiting rows: no lock" in caplog.text
    out = eventlog.rebuild(tmp_path, [tmp_path])
    assert out["ok"] and sorted(r["inspection_id"] for r in _log(tmp_path)[0]) == sorted(
        [run_id, run2])


def test_a_row_is_not_dropped_while_the_log_is_busy(tmp_path, eventlog_on, monkeypatch, caplog):
    """Another program holds the log's lock for longer than an append waits (a rebuild, a
    slow disk): the inspection is held up for LOCK_WAIT_S and no longer, its rows wait in
    events/pending/ - whole, counted by the page - and are added at the next append."""
    first, _, doc = _job(tmp_path, max_dev=0.1, threshold=0.2)
    month = _month(doc)
    pending = tmp_path / "events" / "pending"
    monkeypatch.setattr(eventlog, "LOCK_WAIT_S", 0.2)
    with eventlog._locked(tmp_path / "events") as held:
        assert held
        t0 = time.perf_counter()
        second, run_dir, doc2 = _job(tmp_path, max_dev=0.3, threshold=0.2, result="FAIL",
                                     codes=["SN-2"])
        took = time.perf_counter() - t0
        # the run is what it would have been, and was not held up for long
        assert (doc2["result"], doc2["disposition"], D.exit_code(doc2)) == ("FAIL", "REJECT", 1)
        assert 0.2 <= took < 0.2 + 1.0, took
        assert "no lock" in caplog.text and "waits in events/pending/" in caplog.text
        assert [r["inspection_id"] for r in _log(tmp_path)[0]] == [first]
        (waits,) = eventlog.read_rows(pending / f"inspections_v1_{month}.csv")
        assert (waits["inspection_id"], waits["judgment"], waits["part_id"]) == (second, "FAIL",
                                                                                "SN-2")
        (val,) = eventlog.read_rows(pending / f"inspection_values_v1_{month}.csv")
        assert (val["inspection_id"], val["result"]) == (second, "OUT")
        assert eventlog.waiting(tmp_path) == eventlog.waiting(tmp_path, month) == 1
        for name in ("inspections", "inspection_values"):  # whole files: a header, whole lines
            raw = (pending / f"{name}_v1_{month}.csv").read_bytes()
            assert raw.startswith(b"\xef\xbb\xbf") and raw.endswith(b"\r\n")
            assert raw.count(b"\r\n") == 2
        # the newest row is the waiting one all the same (events/last_v1.csv)
        (last,) = eventlog.read_rows(tmp_path / "events" / "last_v1.csv")
        assert last == waits
        third, _, _ = _job(tmp_path, max_dev=0.1, threshold=0.2)       # still busy: it waits too
        assert eventlog.waiting(tmp_path) == 2
    fourth, _, _ = _job(tmp_path, max_dev=0.1, threshold=0.2)   # free again: merged, then appended
    rows, vals = _log(tmp_path)
    assert [r["inspection_id"] for r in rows] == [first, second, third, fourth]
    assert [v["inspection_id"] for v in vals] == [first, second, third, fourth]
    assert eventlog.waiting(tmp_path) == 0 and list(pending.iterdir()) == []
    raw = next((tmp_path / "events").glob("inspections_v1_*.csv")).read_bytes()
    assert raw.count(b"time,inspection_id") == 1 and raw.count(b"\r\n") == 5


# ================================================================== two processes

_WRITER = r"""
import sys, time
from pathlib import Path
from engine import eventlog
lib, tag, n = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
while not (lib / "go").exists():
    time.sleep(0.002)
for i in range(n):
    iid = f"{tag}{i:03d}"
    row = {"time": "2026-03-01T00:00:00+00:00", "inspection_id": iid, "judgment": "PASS",
           "part": 'a, "b" c', "reason": tag * 400, "n_values": 4}
    vals = [{"inspection_id": iid, "name": f"v{k}", "value": k / 7, "unit": "mm",
             "result": "OK"} for k in range(4)]
    out = eventlog.append(lib, row, vals)
    # no row is ever written nowhere; with the log free every row goes into its file (one
    # that had to wait longer than LOCK_WAIT_S for the other writer waits in pending/)
    assert out["lost"] == 0 and out["written"] + out["pending"] == 5, out
    assert out["pending"] in ((5,) if sys.argv[4] == "busy" else (0, 5)), out
"""


def _two_writers(lib, n, how) -> None:
    procs = [subprocess.Popen([sys.executable, "-c", _WRITER, str(lib), tag, str(n), how],
                              cwd=ROOT, stderr=subprocess.PIPE, text=True) for tag in ("A", "B")]
    (lib / "go").write_text("go", encoding="utf-8")
    for p in procs:
        _, err = p.communicate(timeout=300)
        assert p.returncode == 0, err


def test_two_processes_appending_at_once_leave_whole_rows(tmp_path, eventlog_on):
    lib = tmp_path / "library"
    lib.mkdir()
    n = 60
    _two_writers(lib, n, "free")
    assert eventlog.main(["--merge", "--library", str(lib)]) == 0   # (rows that had to wait)
    _whole_rows(lib, n)


def test_two_processes_writing_while_the_log_is_busy_leave_whole_waiting_rows(tmp_path,
                                                                              eventlog_on):
    """The log's lock is held all the while (a third program is busy with it): both writers
    put every row into events/pending/ under the waiting rows' own lock - whole rows, none
    dropped, none twice - and one merge adds them all."""
    lib = tmp_path / "library"
    lib.mkdir()
    n = 6
    with eventlog._locked(lib / "events") as held:
        assert held
        _two_writers(lib, n, "busy")
        assert _log(lib) == ([], []) and eventlog.waiting(lib) == 2 * n
    assert eventlog.main(["--merge", "--library", str(lib)]) == 0
    assert eventlog.waiting(lib) == 0
    _whole_rows(lib, n)


def _whole_rows(lib, n) -> None:
    """Every row of both writers is in the month's files: once, whole, its values together."""
    rows, vals = _log(lib, "2026-03")
    assert sorted(r["inspection_id"] for r in rows) == sorted(
        f"{t}{i:03d}" for t in "AB" for i in range(n))
    for r in rows:                                          # every row whole, in its own columns
        assert list(r) == INSPECTION_COLUMNS and None not in r.values()
        assert r["reason"] == r["inspection_id"][0] * 400
        assert (r["part"], r["judgment"], r["n_values"]) == ('a, "b" c', "PASS", "4")
    assert len(vals) == 2 * n * 4 and all(list(v) == VALUE_COLUMNS for v in vals)
    for iid in {r["inspection_id"] for r in rows}:          # value rows of one inspection together
        idx = [i for i, v in enumerate(vals) if v["inspection_id"] == iid]
        assert idx == list(range(idx[0], idx[0] + 4))
        assert [vals[i]["name"] for i in idx] == ["v0", "v1", "v2", "v3"]
    raw = (lib / "events" / "inspections_v1_2026-03.csv").read_bytes()
    assert raw.count(b"\r\n") == 2 * n + 1 and raw.count(b"time,inspection_id") == 1


# ================================================================== rebuild

def test_rebuild_reads_every_runs_folder_and_keeps_rows_whose_folder_is_gone(
        eng, tmp_path, eventlog_on, monkeypatch, capsys):
    """One library, runs in four places: its own runs/, beside a station file, where
    $SCANNER_RUNS_DIR points and where a recipe's outputs.runs_dir points."""
    lib = eng.store.root
    stations, recipes = tmp_path / "stations", tmp_path / "recipes"
    places = {"library": lib / "runs", "station": stations / "runs",
              "station_named": tmp_path / "line2_runs", "env": tmp_path / "env_runs",
              "recipe": recipes / "out", "extra": tmp_path / "extra"}
    stations.mkdir()
    recipes.mkdir()
    (stations / "single.json").write_text(json.dumps({"station_id": "single"}))
    (stations / "line2.json").write_text(json.dumps(
        {"station_id": "line2", "parameters": {"runs_dir": str(places["station_named"])}}))
    (stations / "broken.json").write_text("{ not json")
    (recipes / "lid.json").write_text(json.dumps({"name": "lid", "outputs": {"runs_dir": "out"}}))
    monkeypatch.setenv("SCANNER_RUNS_DIR", str(places["env"]))
    ids = {}
    for i, (k, runs) in enumerate(places.items()):
        ids[k] = _job(runs, f"job_{k}", max_dev=0.05 * (i + 1) + (0.1 if i >= 3 else 0.0),
                      threshold=0.2, result="PASS" if i < 3 else "FAIL", picture="overlay.png",
                      codes=[f"SN-{i}"])[0]
    cid, _, _ = _cycle(places["station"])
    found = eventlog.runs_dirs_of(lib, stations, [recipes], [places["extra"], tmp_path / "nowhere"])
    assert {str(d.resolve()) for d in found} == {str(p.resolve()) for p in places.values()}
    live = _log(lib)                        # one log for all of them: the session's library
    assert sorted(r["inspection_id"] for r in live[0]) == sorted([*ids.values(), cid])
    assert not any((runs / "events").exists() for runs in places.values())
    row = {r["inspection_id"]: r for r in live[0]}
    assert row[ids["library"]]["result_file"] == f"runs/{ids['library']}/result.json"
    assert row[ids["env"]]["result_file"] == str(
        (places["env"] / ids["env"] / "result.json").resolve())             # outside: full path
    assert row[ids["env"]]["image_file"] == f"events/img/{ids['env']}.png"     # in the library
    code = eventlog.main(["--rebuild", "--library", str(lib), "--stations", str(stations),
                          "--recipes", str(recipes), "--runs", str(places["extra"])])
    said = capsys.readouterr().out
    assert code == 0 and f"{len(ids) + 1} inspection row(s)" in said, said
    again = _log(lib)
    by_id = lambda rows: sorted(rows, key=lambda r: r["inspection_id"])     # noqa: E731
    assert by_id(again[0]) == by_id(live[0])                # the same rows, cell for cell
    assert sorted(again[1], key=lambda v: (v["inspection_id"], v["step"], v["name"])) == \
        sorted(live[1], key=lambda v: (v["inspection_id"], v["step"], v["name"]))
    assert [r["time"] for r in again[0]] == sorted(r["time"] for r in again[0])     # by time
    # the files it replaced are kept, not deleted
    (aside,) = list((lib / "events").glob("replaced_*"))
    assert len(list(aside.glob("*.csv"))) == 2 and not list((lib / "events").glob("rebuild_*"))
    # a run folder cleaned away: its row (and its values, its picture) stay in the log
    shutil.rmtree(places["env"])
    out = eventlog.rebuild(lib, eventlog.runs_dirs_of(lib, stations, [recipes], [places["extra"]]))
    assert out["ok"] and out["kept"] == 1 and out["inspections"] == len(ids) + 1
    assert by_id(_log(lib)[0]) == by_id(live[0])
    assert (lib / "events" / "img" / f"{ids['env']}.png").read_bytes() == PNG
    # a row that was lost (the log's files removed by hand) comes back from its run folder
    shutil.rmtree(lib / "events")
    out = eventlog.rebuild(lib, [places["library"], places["station"]])
    assert out["ok"] and out["kept"] == 0 and out["replaced"] is None
    assert sorted(r["inspection_id"] for r in _log(lib)[0]) == sorted(
        [ids["library"], ids["station"], cid])


def test_rebuild_may_run_while_inspections_are_made(tmp_path, eventlog_on, monkeypatch):
    """rebuild reads the run folders and the present files, and writes the new files, WITHOUT
    the lock. An inspection that ends meanwhile is appended at once (it does not wait, its row
    is not dropped); rebuild takes over what the files got since it looked - a new row, a row
    that waits, a row written again - and nothing twice."""
    runs = tmp_path / "runs"
    first, first_dir, doc = _job(runs, max_dev=0.1, threshold=0.2)
    month = _month(doc)
    events = runs / "events"
    real_present, late, free = eventlog._present, [], []

    def present(cut):
        got = real_present(cut)
        # the present files were measured; the new ones are not written yet. Now ...
        with eventlog._locked(events, wait_s=0) as held:
            free.append(held)                               # ... nobody holds the lock,
        late.append(_job(runs, max_dev=0.3, threshold=0.2, result="FAIL")[0])   # a run ends,
        # the hook of a run that rebuild has read from its folder writes its row only now,
        eventlog.append(runs, *eventlog.rows_of(first_dir, runs))
        # that run is counted ABORTED on the line after its verdict (its row again),
        R.append_event(first_dir / "run.log.jsonl", first, "aborted_after_verdict",
                       disposition="PULL", message="ABORT after the verdict")
        with _open_elsewhere(monkeypatch, f"inspections_v1_{month}.csv"):
            late.append(_job(runs, max_dev=0.1, threshold=0.2)[0])      # and one row waits
        return got
    monkeypatch.setattr(eventlog, "_present", present)
    out = eventlog.rebuild(runs, [runs])
    assert out["ok"] and free == [True], out
    assert (out["inspections"], out["late"], out["kept"]) == (4, 3, 0)
    rows, vals = _log(runs)
    # the run folders rebuild read, then what came meanwhile, in the order it came
    assert [(r["inspection_id"], r["judgment"]) for r in rows] == [
        (first, "PASS"), (late[0], "FAIL"), (first, "PULL"), (late[1], "PASS")]
    assert rows[2]["reason"] == "aborted: ABORT after the verdict"     # the last row counts
    assert sorted(v["inspection_id"] for v in vals) == sorted([first, *late])   # each once
    assert eventlog.waiting(runs) == 0 and not list((events / "pending").glob("*.csv"))
    (aside,) = list(events.glob("replaced_*"))              # what it replaced is kept
    assert [r["inspection_id"] for r in eventlog.read_rows(
        aside / "pending" / f"inspections_v1_{month}.csv")] == [late[1]]
    # the next rebuild reads all three run folders: one row each, the same cells
    monkeypatch.setattr(eventlog, "_present", real_present)
    out = eventlog.rebuild(runs, [runs])
    assert out["ok"] and (out["inspections"], out["late"]) == (3, 0)
    again = {r["inspection_id"]: r for r in _log(runs)[0]}
    assert again == {first: rows[2], late[0]: rows[1], late[1]: rows[3]}
    assert sorted(v["inspection_id"] for v in _log(runs)[1]) == sorted([first, *late])


def _inspector(lib, monkeypatch) -> tuple:
    """(inspect(waits=False): one hand-made inspection with one value into the log of `lib` -
    with waits its row cannot be written and waits in events/pending/, its value is in the
    month's file; the ids made so far)."""
    made = []

    def inspect(waits=False):
        iid = f"run{len(made)}"
        made.append(iid)
        row = {"time": "2026-03-01T10:00:00+00:00", "inspection_id": iid, "judgment": "PASS"}
        vals = [{"inspection_id": iid, "name": "v", "value": 1.0, "result": "OK"}]
        if not waits:
            assert eventlog.append(lib, row, vals) == {"written": 2, "pending": 0, "lost": 0}
            return
        with _open_elsewhere(monkeypatch, "inspections_v1_2026-03.csv"):
            assert eventlog.append(lib, row, vals) == {"written": 1, "pending": 1, "lost": 0}
    return inspect, made


def _every_inspection_once(lib, made) -> None:
    rows, vals = _log(lib)
    assert sorted(r["inspection_id"] for r in rows) == sorted(made)         # its row once,
    assert sorted(v["inspection_id"] for v in vals) == sorted(made)         # and its value


def test_rebuild_takes_no_half_inspection_when_waiting_rows_are_merged_meanwhile(
        tmp_path, eventlog_on, monkeypatch):
    """An inspection whose row waits has its value in the month's file and its row in
    events/pending/. Just after rebuild measured the files, the next inspection merges the
    waiting row (the waiting file is gone) and the one after it waits (a new file of the same
    name). Every inspection keeps its row AND its value, once."""
    lib = tmp_path / "library"
    inspect, made = _inspector(lib, monkeypatch)
    inspect()
    inspect(waits=True)
    real_present = eventlog._present

    def present(cut):
        assert sorted(p.parent.name for p in cut) == ["events", "events", "pending"]
        inspect()                                           # merges the row that waits ...
        inspect(waits=True)                                 # ... and the next one waits
        return real_present(cut)
    monkeypatch.setattr(eventlog, "_present", present)
    out = eventlog.rebuild(lib, [])
    assert out["ok"] and (out["inspections"], out["values"], out["late"]) == (4, 4, 2), out
    _every_inspection_once(lib, made)
    assert eventlog.waiting(lib) == 0 and len(made) == 4


@pytest.mark.parametrize("others, reads", [(1, 2), (9, 4)], ids=["once", "every_time"])
def test_rebuild_reads_again_when_another_rebuild_swapped_the_files(
        tmp_path, eventlog_on, monkeypatch, others, reads):
    """Two rebuilds at once: the files this one measured are other files by the time it reads
    them. It measures and reads again - when that happens every time, the last time with the
    lock held - and no inspection is lost or doubled."""
    lib = tmp_path / "library"
    inspect, made = _inspector(lib, monkeypatch)
    inspect()
    inspect()
    real_present, calls = eventlog._present, []

    def present(cut):
        calls.append(len(cut))
        if len(calls) <= min(others, 3):                    # (the fourth time the lock is held)
            monkeypatch.setattr(eventlog, "_present", real_present)
            assert eventlog.rebuild(lib, [])["ok"]          # the other rebuild, start to end
            monkeypatch.setattr(eventlog, "_present", present)
            inspect()                                       # and an inspection ends
        return real_present(cut)
    monkeypatch.setattr(eventlog, "_present", present)
    out = eventlog.rebuild(lib, [])
    assert out["ok"] and calls == [2] * reads, (out, calls)
    _every_inspection_once(lib, made)
    assert len(list((lib / "events").glob("replaced_*"))) == min(others, 3) + 1


def test_rebuild_reads_again_when_one_file_could_not_be_read(tmp_path, eventlog_on, monkeypatch):
    """One of the files rebuild measured cannot be read at that moment (another program has
    it for an instant). With the rows of the one file and without the values of the other it
    would write the inspections without their values: it measures and reads again."""
    lib = tmp_path / "library"
    inspect, made = _inspector(lib, monkeypatch)
    inspect()
    inspect()
    real_which, real_present, fail, calls = eventlog._which, eventlog._present, [], []

    def which(path):
        if fail and Path(path).name.startswith("inspection_values"):
            fail.pop()
            raise PermissionError(13, "the file is in use", str(path))
        return real_which(path)

    def present(cut):
        calls.append(len(cut))
        if len(calls) == 1:
            fail.append("the values file, once")
        return real_present(cut)
    monkeypatch.setattr(eventlog, "_which", which)
    monkeypatch.setattr(eventlog, "_present", present)
    out = eventlog.rebuild(lib, [])
    assert out["ok"] and calls == [2, 2] and not fail, (out, calls)
    assert (out["inspections"], out["values"], out["kept"], out["late"]) == (2, 2, 2, 0)
    _every_inspection_once(lib, made)


def test_rebuild_holds_the_lock_only_to_measure_and_to_swap(tmp_path, eventlog_on, monkeypatch):
    """Whatever takes long in a rebuild - reading every run folder, reading the present files,
    writing the new ones - happens with the lock free."""
    runs = tmp_path / "runs"
    ids = [_job(runs, max_dev=0.1, threshold=0.2)[0] for _ in range(3)]
    free = {}

    def watched(name):
        real = getattr(eventlog, name)

        def f(*a, **k):
            with eventlog._locked_all(runs / "events", 0) as held:
                free.setdefault(name, []).append(held)
            return real(*a, **k)
        monkeypatch.setattr(eventlog, name, f)
    for name in ("rows_of", "_present", "_encode"):
        watched(name)
    out = eventlog.rebuild(runs, [runs])
    assert out["ok"] and sorted(r["inspection_id"] for r in _log(runs)[0]) == sorted(ids)
    assert len(free["rows_of"]) == 3 and len(free["_present"]) == 1 and len(free["_encode"]) >= 2
    assert all(held for helds in free.values() for held in helds), free


def test_rebuild_waits_for_a_run_that_ended_a_moment_ago(tmp_path, eventlog_on, monkeypatch):
    """A run whose result.json was written a moment ago: its own hook may still be writing its
    row. rebuild waits (at most HOOK_SETTLE_S) before it looks at the present files, so that
    row is not added a second time behind its back. Older runs are not waited for."""
    real_sleep, slept = time.sleep, []
    monkeypatch.setattr(eventlog.time, "sleep", lambda s: (slept.append(s), real_sleep(s))[1])
    monkeypatch.setattr(eventlog, "HOOK_SETTLE_S", 1.0)
    run_id, run_dir, _ = _job(tmp_path, max_dev=0.1, threshold=0.2)
    del slept[:]
    assert eventlog.rebuild(tmp_path, [tmp_path])["ok"]
    assert 0.5 < max(slept) <= 1.0, slept
    old = time.time() - 60.0
    os.utime(run_dir / "result.json", (old, old))
    del slept[:]
    assert eventlog.rebuild(tmp_path, [tmp_path])["ok"]
    assert max(slept) < 0.05, slept
    assert [r["inspection_id"] for r in _log(tmp_path)[0]] == [run_id]


def test_the_command_line_needs_a_library_and_one_thing_to_do(tmp_path, eventlog_on, capsys):
    assert eventlog.main(["--rebuild", "--library", str(tmp_path / "nowhere")]) == 2
    assert "no library folder" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        eventlog.main(["--library", str(tmp_path)])
    with pytest.raises(SystemExit):
        eventlog.main(["--rebuild", "--merge", "--library", str(tmp_path)])
    r = subprocess.run([sys.executable, "-m", "engine.eventlog", "--help"], cwd=ROOT,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and "--rebuild" in r.stdout and "Touches no rig" in r.stdout


# ================================================================== height profiles

def test_profile_files_are_two_plain_columns_and_never_overwritten(tmp_path, eventlog_on):
    nan = float("nan")
    sec = {"face": {"pos_mm": [-0.25, 0.0, 0.25], "x_mm": [0.011, nan, -0.02],
                    "y_mm": [nan, 0.5, 0.25]},
           "other": {"pos_mm": [0.0], "x_mm": [9.0], "y_mm": [9.0]}}
    assert eventlog.write_profiles(tmp_path, sec) == ["profile_x.csv", "profile_y.csv"]
    assert (tmp_path / "profile_x.csv").read_text() == \
        "position_mm,height_mm\n-0.250,0.01100\n0.250,-0.02000\n"
    assert (tmp_path / "profile_y.csv").read_text() == \
        "position_mm,height_mm\n0.000,0.50000\n0.250,0.25000\n"
    assert eventlog.write_profiles(tmp_path, sec) == []     # a run folder is new: never twice
    assert eventlog.write_profiles(tmp_path / "b", {}) == [] and not (tmp_path / "b").exists()
    (tmp_path / "c").mkdir()                                # nothing measured on a line: no file
    assert eventlog.write_profiles(tmp_path / "c", {"z": {"pos_mm": [0.0], "x_mm": [nan],
                                                          "y_mm": [1.0]}}) == ["profile_y.csv"]


def test_a_check_hands_its_two_centre_lines_to_the_runner_only():
    """CheckContext.keep_sections: the lines through the centre of a gridded height map,
    copied, the first check of a zone counts; without a runner nothing is kept."""
    from engine.recipes.context import CheckContext
    g, h = np.array([-0.5, 0.0, 0.5]), np.arange(9.0).reshape(3, 3)
    alone = CheckContext(scan=None, frame=None, zone=SimpleNamespace(id="face"))
    alone.keep_sections(g, h)
    assert alone.sections is None
    kept = {}
    ctx = CheckContext(scan=None, frame=None, zone=SimpleNamespace(id="face"), sections=kept)
    ctx.keep_sections(g, h)
    ctx.keep_sections(g, h + 100)
    assert list(kept) == ["face"] and kept["face"]["pos_mm"].tolist() == [-0.5, 0.0, 0.5]
    assert kept["face"]["x_mm"].tolist() == [3.0, 4.0, 5.0]        # the row: along camera right
    assert kept["face"]["y_mm"].tolist() == [1.0, 4.0, 7.0]        # the column: along camera down
    h[1, 1] = -1.0
    assert kept["face"]["x_mm"][1] == 4.0                   # a copy: the map stays the check's
    ctx.zone = None                                         # whatever goes wrong: no error
    ctx.keep_sections(g, h)


# ================================================================== real runs (synthetic scans)

@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    """A mock library with the bench's TRUE stereo calibration, the golden of the long tube
    and synthetic scans of a good and a dented one (recipe_synth: nothing is captured)."""
    root = tmp_path_factory.mktemp("eventlog") / "library"
    s = api.configure(root, mock=True)
    try:
        p = api.load_preset(PRESET)
        RS.truth_calibration(s, p)
        g = RS.oracle_scan(s, p, "scene:tube_long", "tube_long_alignment_golden", seed=9)
        runner.record_golden(s, api.recipe_load("tube_long_alignment"), p, g)
        scans = {v: RS.oracle_scan(s, p, f"scene:tube_long:{v}", f"t_long_{v}", seed=3 + j)
                 for j, v in enumerate(("ok", "dent"))}
        for ms in scans.values():
            ms.cloud                                        # loaded once, reused by every run
    finally:
        api._session = None
    return root, scans


@pytest.fixture
def real(lib, eventlog_on):
    """The engine on that library with the log on and empty."""
    root, scans = lib
    shutil.rmtree(root / "events", ignore_errors=True)
    s = api.configure(root, mock=True)
    yield s, scans
    api._session = None


def _same_after_rebuild(root, runs_dirs) -> None:
    live = _log(root)
    out = eventlog.rebuild(root, runs_dirs)
    assert out["ok"], out
    assert _log(root) == live                               # the same rows, in the same order


def test_a_recipe_run_is_one_row_with_values_picture_profile_and_3d_data(real, monkeypatch):
    s, scans = real
    root = s.store.root
    runs = root / "runs_recipe"
    out = api.recipe_run("tube_long_end_face", scan=scans["ok"].id, project_feedback=False,
                         runs_dir=runs)
    res, run_dir = out["result"], Path(out["run_dir"])
    assert out["exit_code"] == 0 and res["disposition"] == "CONTINUE"
    (row,), vals = _log(root)
    rid = res["run_id"]
    assert (row["inspection_id"], row["judgment"], row["reason"]) == (rid, "PASS", "")
    assert (row["program"], row["part"], row["hardware"]) == (
        "tube_long_end_face", "tube_long", "MOCK")
    assert row["operating_mode"] == "recipe_run"            # a recipe run on its own
    assert row["preset"] == PRESET and row["calibration_id"] == res["calibration"]["calibration_id"]
    assert row["rig"] == res["capture"]["geometry_id"] and row["part_id"] == ""
    # one value row per tolerance of every check, with the limits the check held it to
    want = [(z["id"], ch["id"], t["name"]) for z in res["zones"] for ch in z["checks"]
            for t in ch["tolerances"]]
    assert [(v["zone"], v["check"], v["name"]) for v in vals] == want and len(want) >= 3
    (depth,) = [v for v in vals if v["name"] == "max_depth_mm"]
    assert (depth["zone"], depth["check"], depth["unit"], depth["lower_limit"],
            depth["upper_limit"], depth["result"]) == ("end_face", "defect", "mm", "", "0.15", "OK")
    assert all(v["inspection_id"] == rid and v["step"] == "" for v in vals)
    assert row["n_values"] == str(len(want)) and row["n_out_of_limit"] == "0"
    # the links: all inside the library, so the page can open them
    assert row["result_file"] == f"runs_recipe/{rid}/result.json"
    assert row["image_file"] == f"events/img/{rid}.png"
    assert (root / row["image_file"]).read_bytes() == (run_dir / "overlay.png").read_bytes()
    assert row["data_3d_file"].startswith("output/pointclouds/") and row["data_3d_file"].endswith(
        "_t_long_ok.ply")
    assert (root / row["data_3d_file"]).read_bytes().startswith(b"ply")
    # the end-face check gridded the face: its two centre lines are files of the run folder
    assert row["profile_file"] == f"runs_recipe/{rid}/profile_x.csv"
    for name in ("profile_x.csv", "profile_y.csv"):
        with open(run_dir / name, newline="") as f:
            head, *lines = list(csv.reader(f))
        assert head == ["position_mm", "height_mm"] and len(lines) > 60
        pos, h = [float(a) for a, _ in lines], [float(b) for _, b in lines]
        assert pos == sorted(pos) and -10.5 <= pos[0] < -9 and 9 < pos[-1] <= 10.5   # r 10.5 mm
        assert max(abs(v) for v in h) < 0.2                 # a good face: flat within the noise
    # ... and nothing of it is in the result: the document and the check are what they were
    on_disk = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    assert set(on_disk) == RESULT_KEYS and "profile" not in json.dumps(on_disk["outputs"])
    face = on_disk["zones"][1]["checks"][1]
    assert face["type"] == "defect" and "sections" not in json.dumps(face)
    assert sorted(p.name for p in run_dir.iterdir()) == [
        "feedback_dmd.png", "overlay.png", "profile_x.csv", "profile_y.csv", "result.json",
        "run.log.jsonl"]
    _same_after_rebuild(root, [runs])
    # the same run with the log off: the same verdict, no row, and the run folder of before
    monkeypatch.setenv(EVENTLOG_ENV, "off")
    off = api.recipe_run("tube_long_end_face", scan=scans["ok"].id, project_feedback=False,
                         runs_dir=runs)
    assert (off["exit_code"], off["result"]["disposition"]) == (0, "CONTINUE")
    assert sorted(p.name for p in Path(off["run_dir"]).iterdir()) == [
        "feedback_dmd.png", "overlay.png", "result.json", "run.log.jsonl"]
    assert [r["inspection_id"] for r in _log(root)[0]] == [rid]
    assert not (root / "events" / "img" / f"{off['run_id']}.png").exists()


def test_a_real_cycle_and_a_work_state_on_its_own(real, tmp_path):
    """The example line: three work states, a dented tube. One row for the cycle (FAIL, kicked
    out at the end face) with the values of its three checks; the same check of one work state
    run outside a cycle is a row of its own. The cycle's folders are beside the station, not in
    the library: its picture is copied in, its files are named by their full path."""
    s, scans = real
    root = s.store.root
    runs = tmp_path / "station" / "runs"
    out = api.flow_run(EXAMPLE, "long_tube", inputs={"part_present": True}, runs_dir=runs,
                       scans={k: scans["dent"].id for k in ("loaded", "aligned", "end_face")},
                       acknowledge=True, project_faults=False)
    res, cdir = out["result"], Path(out["cycle_dir"])
    assert out["exit_code"] == 1 and res["disposition"] == "REJECT"
    assert len(list((cdir / "steps").iterdir())) == 3
    (row,), vals = _log(root)
    assert row["inspection_id"] == res["run_id"] and row["judgment"] == "FAIL"
    assert row["program"] == "example_tube_line.json / long_tube" and row["part"] == "long_tube"
    assert row["reason"].startswith("end_face: end face not OK - end_face/defect: 1 defect(s)")
    assert {v["step"] for v in vals} == {"loaded", "aligned", "end_face"}
    assert [v["step"] for v in vals] == sorted((v["step"] for v in vals),
                                               key=["loaded", "aligned", "end_face"].index)
    assert all(v["inspection_id"] == res["run_id"] for v in vals)
    (out_of_limit,) = [v for v in vals if v["result"] == "OUT"]
    assert (out_of_limit["step"], out_of_limit["zone"], out_of_limit["check"],
            out_of_limit["name"], out_of_limit["upper_limit"]) == (
        "end_face", "end_face", "defect", "max_depth_mm", "0.15")
    assert row["n_values"] == str(len(vals)) and row["n_out_of_limit"] == "1"
    assert (row["preset"], row["rig"]) == (PRESET, scans["dent"].geometry_id)
    assert row["calibration_id"] == scans["dent"].calibration_id
    last = Path(res["steps"][-1]["dlp_result"]["run_folder"])
    assert row["image_file"] == f"events/img/{res['run_id']}.png"
    assert (root / row["image_file"]).read_bytes() == (last / "overlay.png").read_bytes()
    assert row["result_file"] == str((cdir / "result.json").resolve())
    assert row["profile_file"] == str((last / "profile_x.csv").resolve())
    assert row["data_3d_file"].endswith("_t_long_dent.ply")        # the scan's, in the library
    assert not (runs / "events").exists() and not (cdir / "events").exists()
    # the check of one work state, outside a cycle: its own row
    one = api.flow_verify(EXAMPLE, "long_tube", "loaded", runs_dir=runs, scan=scans["ok"].id)
    rows, vals2 = _log(root)
    assert [r["inspection_id"] for r in rows] == [res["run_id"], one["run_id"]]
    assert rows[1]["program"] == "example_tube_line.json / long_tube / loaded"
    assert rows[1]["judgment"] == "PASS"
    assert [r["operating_mode"] for r in rows] == ["station", "station"]
    mine = [v for v in vals2 if v["inspection_id"] == one["run_id"]]
    assert mine and all(v["step"] == "" for v in mine)
    _same_after_rebuild(root, [runs])


def test_a_real_cycle_that_never_ended(real, tmp_path, monkeypatch):
    """The example line, as the engine writes it: the cycle is opened, its first work state is
    checked - and the program is closed before the next. No result.json of the cycle, no row.
    After a rebuild the cycle is a PULL row with the values of the work state it checked."""
    s, scans = real
    root = s.store.root
    runs = tmp_path / "station" / "runs"
    cfg = api.flow_load(EXAMPLE)
    opened = FV.open_cycle(cfg, "long_tube", runs_dir=runs, mock=True, source="test")
    cid, cdir = opened["cycle_id"], Path(opened["cycle_dir"])
    one = FV.verify_step(cfg, "long_tube", "loaded", runs_dir=cdir / FV.STEPS_DIR,
                         scan=scans["ok"].id)
    assert one["result"]["disposition"] == "CONTINUE" and _log(root) == ([], [])
    out = eventlog.rebuild(root, [runs])                    # the cycle is still running
    assert out["ok"] and out["unended"] == 0 and _log(root) == ([], [])
    opened["log"].close()                                   # the program is closed
    monkeypatch.setattr(eventlog, "UNENDED_QUIET_S", 0.0)   # (not Windows: no hour to wait)
    out = eventlog.rebuild(root, [runs])
    assert out["ok"] and (out["inspections"], out["unended"]) == (1, 1)
    (row,), vals = _log(root)
    assert (row["inspection_id"], row["judgment"]) == (cid, "PULL")
    assert row["reason"].startswith("error: the cycle never ended")
    assert (row["program"], row["part"], row["hardware"], row["operating_mode"]) == (
        "example_tube_line.json / long_tube", "long_tube", "MOCK", "station")
    want = [(z["id"], ch["id"], t["name"]) for z in one["result"]["zones"] for ch in z["checks"]
            for t in ch["tolerances"] or []]
    assert [(v["zone"], v["check"], v["name"]) for v in vals] == want and want
    assert all((v["inspection_id"], v["step"], v["result"]) == (cid, "loaded", "OK") for v in vals)
    assert (row["preset"], row["rig"]) == (PRESET, scans["ok"].geometry_id)
    assert row["calibration_id"] == scans["ok"].calibration_id
    assert row["image_file"] == f"events/img/{cid}.png" and row["result_file"] == ""
    assert (root / row["image_file"]).read_bytes() == (
        Path(one["run_dir"]) / "overlay.png").read_bytes()
    assert row["data_3d_file"].endswith("_t_long_ok.ply")
    assert row["time"] == eventlog.local_time(one["result"]["finished"])
    _same_after_rebuild(root, [runs])


def test_a_measurement_is_a_row_without_a_judgment(real, tmp_path):
    s, scans = real
    root = s.store.root
    runs = root / "runs_measure"
    p = api.load_preset(PRESET)
    out = api.measure_run("flatness", p, scans=[scans["ok"].id], runs_dir=runs,
                          roi=[100, 100, 400, 400])
    doc = out["result"]
    assert out["exit_code"] == 0 and doc["status"] == "MEASURED"
    (row,), vals = _log(root)
    assert (row["inspection_id"], row["judgment"], row["reason"]) == (doc["run_id"], "-", "")
    assert (row["program"], row["part"], row["hardware"], row["operating_mode"]) == (
        "measure:flatness", "measure_flatness", "MOCK", "measure")
    assert [(v["name"], v["unit"]) for v in vals] == [(v["name"], v["units"])
                                                      for v in doc["values"]]
    assert all((v["lower_limit"], v["upper_limit"], v["result"], v["check"]) == (
        "", "", "-", "flatness") for v in vals)
    assert float(vals[0]["value"]) == pytest.approx(doc["values"][0]["value"], rel=1e-5)
    assert row["n_values"] == str(len(vals)) and row["n_out_of_limit"] == "0"
    assert row["values"].startswith(f"flatness_rms={vals[0]['value']} mm; ")
    assert " OK" not in row["values"] and "[" not in row["values"]         # nothing judges it
    rid = doc["run_id"]
    assert row["data_3d_file"] == f"runs_measure/{rid}/scan.ply"
    assert row["image_file"] == f"events/img/{rid}.png" and row["profile_file"] == ""
    assert (row["preset"], row["rig"]) == (PRESET, p.geometry_id)
    assert row["calibration_id"] == doc["calibration"]["calibration_id"]
    # the document still decides nothing
    on_disk = json.loads((Path(out["run_dir"]) / "result.json").read_text(encoding="utf-8"))
    assert on_disk["result"] is None and on_disk["disposition"] == "NONE"
    # NOT MEASURABLE: still a row, still no judgment, and the reason why there is no number
    again = api.measure_run("flatness", p, scans=[scans["ok"].id, scans["ok"].id], runs_dir=runs)
    assert again["exit_code"] == 2 and again["result"]["status"] == "NOT_MEASURABLE"
    rows, vals = _log(root)
    assert [r["judgment"] for r in rows] == ["-", "-"] and rows[1]["n_values"] == "0"
    assert rows[1]["reason"].startswith("refused: ValueError: scan ")
    assert rows[1]["image_file"] == "" and rows[1]["data_3d_file"] == ""
    _same_after_rebuild(root, [runs])


def _cli(argv) -> tuple:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = cli.main(argv)
    api._session = None
    return code, buf.getvalue()


def test_a_quick_check_is_one_row_and_a_pass_in_the_guard_band_reads_pull(real, monkeypatch,
                                                                          tmp_path):
    """`scan quick-check` (the station job runner on a generated one-job config), with the scan
    the rig would have made: the flatness limit held, then a limit so close that the pass is in
    its guard band - exit 5, and the row says PULL and why."""
    s, scans = real
    root = s.store.root
    runs = tmp_path / "runs"
    monkeypatch.setattr(api, "scan_fast", lambda zone, p, label, steps, periods, average, **kw:
                        scans["ok"])
    monkeypatch.setattr(cli, "_mock_place_metric", lambda ctx: None)
    base = ["--mock", "--library", str(root), "quick-check", PRESET, "--check", "flatness",
            "--roi", "100,100,400,400", "--runs-dir", str(runs)]
    code, _ = _cli([*base, "--threshold", "0.5"])
    assert code == 0
    (row,), (val,) = _log(root)
    assert (row["judgment"], row["program"], row["part"]) == ("PASS", "quick_check", "quick_check")
    assert (row["operating_mode"], row["hardware"]) == ("quick_check", "MOCK")
    assert (val["name"], val["check"], val["upper_limit"], val["result"]) == (
        "deviation_mm.max", "metric", "0.5", "OK")
    limit = f"{float(val['value']) * 1.5:.4f}"
    code, _ = _cli([*base, "--threshold", limit, "--guard-band", "50"])
    assert code == 5                                        # the exit code is the run's own
    rows, vals = _log(root)
    assert [r["judgment"] for r in rows] == ["PASS", "PULL"]
    assert rows[1]["reason"].startswith("guard_band: deviation_mm.max ")
    assert vals[1]["result"] == "OK" and rows[1]["n_out_of_limit"] == "0"   # inside its limit
    assert float(vals[1]["upper_limit"]) == pytest.approx(float(limit))
    assert rows[1]["image_file"] == f"events/img/{rows[1]['inspection_id']}.png"
    assert rows[1]["data_3d_file"].endswith("_t_long_ok.ply")
    _same_after_rebuild(root, [runs])


def test_a_failed_append_changes_neither_the_result_nor_the_exit_code_of_a_real_run(
        real, monkeypatch):
    s, scans = real
    root = s.store.root
    runs = root / "runs_blocked"

    def run():
        out = api.recipe_run("tube_long_end_face", scan=scans["dent"].id, project_feedback=False,
                             runs_dir=runs)
        doc = json.loads((Path(out["run_dir"]) / "result.json").read_text(encoding="utf-8"))
        return out["exit_code"], doc
    month = eventlog.month_of(None)
    with _open_elsewhere(monkeypatch, f"inspections_v1_{month}.csv",
                         f"inspection_values_v1_{month}.csv"):
        code, doc = run()
    assert _log(root) == ([], []) and eventlog.waiting(root) == 1
    code2, doc2 = run()                                     # the files can be written again
    assert code == code2 == 1 and doc["disposition"] == doc2["disposition"] == "REJECT"
    drop = ("run_id", "run_folder", "started", "finished", "steps", "outputs", "feedback")
    strip = lambda d: {k: v for k, v in d.items() if k not in drop}    # noqa: E731
    for z in doc["zones"] + doc2["zones"]:
        for ch in z["checks"]:
            ch.pop("duration_s")
    assert strip(doc) == strip(doc2)
    rows, vals = _log(root)
    assert [r["inspection_id"] for r in rows] == [doc["run_id"], doc2["run_id"]]
    assert [r["judgment"] for r in rows] == ["FAIL", "FAIL"] and eventlog.waiting(root) == 0
    assert len(vals) == 2 * int(rows[0]["n_values"])


# ================================================================== the page

HEADINGS = ["Time", "Inspection no.", "Part ID", "Part", "Program", "Judgment", "Reason", "Values",
            "Files"]


def test_the_page_is_plain_and_self_contained():
    html = (STATIC / "events.html").read_text(encoding="utf-8")
    js = (STATIC / "events.js").read_text(encoding="utf-8")
    assert re.findall(r"<th>([^<]+)</th>", html) == HEADINGS
    assert '<script src="/static/events.js" defer></script>' in html
    assert not re.search(r"\son[a-z]+\s*=", html), "inline handler (the CSP forbids them)"
    assert len(re.findall(r"<script", html)) == 1 and "javascript:" not in html.lower()
    for text in (html, js):                                 # no external scripts, fonts, images
        assert "cdn" not in text.lower() and "https://" not in text and "//fonts" not in text
        assert not re.search(r"""(src|href)\s*=\s*["']\s*(https?:)?//""", text)
    assert "http://" not in js and html.count("http://") == 1      # the icon's SVG namespace
    assert "@import" not in html and "url(" not in html
    used = set(re.findall(r"\$\('#([\w-]+)", js))
    defined = set(re.findall(r'\bid="([\w-]+)"', html))
    assert used and not (used - defined), sorted(used - defined)
    assert 'type="month"' in html and 'id="evWaiting"' in html
    said = " ".join(html.split())
    assert "refused before a run folder exists is not listed" in said
    assert "reads the whole month" in said and "can take a few seconds" in said
    assert "this page shows the last one" in said and "last_v1.csv" in js
    # it reads the files through the route the server already has, by their known names
    assert "'/files/'" in js and "inspections_v1_${month}.csv" in js and "pending/" in js
    assert "innerHTML" not in js and "eval(" not in js      # text only: a cell is never markup
    node = shutil.which("node")
    if node:
        r = subprocess.run([node, "--check", str(STATIC / "events.js")], capture_output=True,
                           text=True)
        assert r.returncode == 0, r.stderr


_PAGE_HARNESS = r"""
const fs = require('fs'), vm = require('vm');
class El {
  constructor(tag) { this.tagName = String(tag).toUpperCase(); this.children = []; this._text = '';
    this.className = ''; this.hidden = false; this.title = ''; this.href = ''; this.value = ''; }
  set textContent(t) { this._text = String(t); this.children = []; }
  get textContent() { return this._text + this.children.map((c) => c.textContent).join(''); }
  appendChild(c) { this.children.push(c); return c; }
  addEventListener() {}
}
const byId = {};
const el = (id) => byId[id] || (byId[id] = new El('div'));
let files = {};
const asked = [];
const sandbox = {
  console, URLSearchParams, window: { location: { search: '' } },
  document: { addEventListener() {}, querySelector: (s) => el(s.slice(1)),
              createElement: (t) => new El(t),
              createTextNode: (t) => ({ textContent: String(t), children: [] }) },
  fetch: async (path) => {
    asked.push(path);
    const t = files[path];
    const status = t === undefined ? 404 : t === null ? 500 : 200;
    return { ok: status === 200, status, text: async () => t };
  },
};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync(process.argv[2], 'utf8'), sandbox, { filename: 'events.js' });
const run = (code) => vm.runInContext(code, sandbox);
const walk = (e, tag, out = []) => {
  for (const c of e.children || []) { if (c.tagName === tag) out.push(c); walk(c, tag, out); }
  return out;
};
const table = () => el('evBody').children.map((tr) => ({
  cells: tr.children.map((td) => td.textContent),
  time: tr.children[0].title,
  judgment: (walk(tr.children[5], 'SPAN')[0] || {}).className,
  links: walk(tr.children[8], 'A').map((a) => [a.textContent, a.href]),
  named: walk(tr.children[8], 'SPAN').map((s) => [s.textContent, s.title]),
}));
const shown = () => ({ rows: table(), count: el('evCount').textContent,
  waiting: el('evWaiting').hidden ? null : el('evWaiting').textContent,
  empty: el('evEmpty').hidden ? null : el('evEmpty').textContent,
  error: el('evErr').hidden ? null : el('evErr').textContent, month: el('evMonth').value,
  more: el('evMore').hidden ? null : el('evMore').textContent,
  csv: el('evCsv').hidden ? null : el('evCsv').href });
const out = {};
(async () => {
  const data = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
  files = data.files;
  await run("load('2026-10')");
  out.october = shown();
  el('evFilter').value = 'sn-7';
  run('render()');
  out.filtered = shown();
  el('evFilter').value = '';
  run('state.bad = true; render()');                    // "REJECT and PULL only"
  out.bad = shown();
  run('state.bad = false; render()');
  await run("load(shiftMonth('2026-10', -1))");
  out.september = shown();
  out.asked = asked.slice();
  files = { '/files/events/inspections_v1_2026-12.csv': null };
  await run("load(shiftMonth('2026-10', 2))");
  out.december = shown();
  files = { '/files/events/inspections_v1_2027-01.csv': data.many };
  await run("load(shiftMonth('2026-12', 1))");
  out.many = { n: el('evBody').children.length, more: el('evMore').textContent,
               count: el('evCount').textContent };
  await run("load('nonsense')");
  out.stays = el('evMonth').value;
  files = { '/files/events/inspections_v1_2026-11.csv': data.early };
  await run("load('2026-11')");
  out.early = table().map((x) => [x.cells[0], x.cells[1], x.time, x.cells[4]]);
  out.parsed = run(`parseCsv('a,"b ""x"", y",c\\r\\n1,,"two\\nlines"\\n')`);
  out.odd = ['', 'a,b\\n\\n\\nc', '"never closed,\\nrow', 'ab"cd,ef"gh,i\\r',
             '\\x22\\x22\\x22\\x22,\\x22\\x22\\r\\n,', 'a\\rb\\rc']
    .map((t) => run(`parseCsv('${t}')`));
  const fileUrl = run('fileUrl');
  out.urls = ['events/img/a b.png', data.outside, '/abs/result.json', '\\\\host\\x.json', '']
    .map((p) => fileUrl(p));
  process.stdout.write(JSON.stringify(out));
})().catch((e) => { console.error(e); process.exit(1); });
"""


def _csv_text(fields, rows) -> str:
    return (eventlog._BOM + eventlog._encode(tuple(fields), rows, header=True)).decode("utf-8")


def test_the_page_shows_the_newest_rows_first_with_links_and_waiting_rows(tmp_path):
    """The real events.js (node, fake DOM) on files as the engine writes them."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node not installed; events.js logic not checked")
    outside = str(tmp_path / "line" / "runs" / "run2" / "result.json")     # not in the library

    def row(i, when, **kw):
        return {"time": when, "inspection_id": f"run{i}", "part_id": f"SN-{i}", "part": "tube",
                "program": "line1.json / long", "operating_mode": "station", "hardware": "REAL",
                "judgment": "PASS", "reason": "",
                "n_values": 2, "n_out_of_limit": 0, "values": "v_mm=1 mm [0..2] OK",
                "image_file": f"events/img/run{i}.png", "result_file": f"runs/run{i}/result.json",
                **kw}
    main = [row(1, "2026-10-13T07:00:00+00:00"),
            row(3, "2026-10-13T09:00:00+00:00", judgment="FAIL", n_out_of_limit=1,
                hardware="MOCK", reason='lid/flatness: 0.31 mm > 0.2 mm, "too high"',
                part_id='=cmd|\' /C calc\'!A0', profile_file="runs/run3/profile_x.csv",
                data_3d_file="output/pointclouds/a b.ply"),
            row(2, "2026-10-13T08:00:00+00:00", judgment="-", program="measure:flatness",
                operating_mode="measure", result_file=outside, image_file=""),
            # run1 again: ABORT came after its verdict, so its row was written a second time
            row(1, "2026-10-13T07:00:00+00:00", judgment="PULL",
                reason="aborted: ABORT was pressed after this run wrote its verdict")]
    waiting = [row(7, "2026-10-13T10:00:00+00:00", judgment="PULL",
                   reason="guard_band: v_mm 1.99 is within 10 % of its upper limit"),
               row(6, "2026-10-13T06:00:00+00:00")]
    many = [row(i, f"2027-01-01T00:{i // 60:02d}:{i % 60:02d}+00:00") for i in range(620)]
    # a file written before the v1 header was settled (`mode`, no operating_mode), with times
    # at three offsets: by the moment the newest is 01:10 UTC, by the text it is the oldest
    early = [row(21, "2026-11-01T01:30:00+01:00", mode="MOCK"),
             row(22, "2026-11-01T01:10:00+00:00", mode="REAL"),
             row(23, "2026-11-01T02:20:00+02:00", mode="REAL")]
    data = {"files": {
        "/files/events/inspections_v1_2026-10.csv": _csv_text(INSPECTION_COLUMNS, main),
        "/files/events/pending/inspections_v1_2026-10.csv": _csv_text(INSPECTION_COLUMNS, waiting)},
        "many": _csv_text(INSPECTION_COLUMNS, many), "outside": outside,
        "early": _csv_text(EARLY_V1_COLUMNS, early)}
    (tmp_path / "data.json").write_text(json.dumps(data), encoding="utf-8")
    (tmp_path / "harness.js").write_text(_PAGE_HARNESS, encoding="utf-8")
    r = subprocess.run([node, str(tmp_path / "harness.js"), str(STATIC / "events.js"),
                        str(tmp_path / "data.json")], capture_output=True, text=True, timeout=60,
                       env={**os.environ, "TZ": "UTC"})
    assert r.returncode == 0, r.stderr
    out = json.loads(r.stdout)
    oct_ = out["october"]
    rows = oct_["rows"]
    assert [x["time"] for x in rows] == sorted((x["time"] for x in rows), reverse=True)
    assert [x["cells"][1].split()[0] for x in rows] == ["run7", "run3", "run2", "run1", "run6"]
    assert oct_["month"] == "2026-10" and oct_["count"] == "5 inspection(s)"
    assert oct_["waiting"].startswith("2 rows waiting") and oct_["empty"] is None
    assert oct_["error"] is None and oct_["more"] is None
    assert oct_["csv"] == "/files/events/inspections_v1_2026-10.csv?download=1"
    seven, three, two, one = rows[0], rows[1], rows[2], rows[3]
    # an inspection whose row was written again is shown once, as its last row says
    assert (one["cells"][1], one["cells"][5], one["judgment"]) == ("run1", "PULL", "tag warn")
    assert one["cells"][6].startswith("aborted: ABORT was pressed after")
    assert re.fullmatch(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d waiting", seven["cells"][0])
    assert seven["judgment"] == "tag warn" and seven["cells"][5] == "PULL"
    assert seven["cells"][6].startswith("guard_band: v_mm 1.99")
    # a MOCK row says so; foreign text is shown as text, without the spreadsheet apostrophe
    assert three["cells"][1] == "run3 MOCK" and three["cells"][2] == "=cmd|' /C calc'!A0"
    assert three["judgment"] == "tag bad" and three["cells"][7].endswith("1 out of limit")
    assert three["cells"][6] == 'lid/flatness: 0.31 mm > 0.2 mm, "too high"'
    assert three["links"] == [
        ["image", "/files/events/img/run3.png"], ["profile", "/files/runs/run3/profile_x.csv"],
        ["3D data", "/files/output/pointclouds/a%20b.ply?download=1"],
        ["result", "/files/runs/run3/result.json"]]
    # a file outside the library is named, not linked (the server does not serve it)
    assert two["links"] == [] and two["named"][0][0] == "result"
    assert two["named"][0][1].startswith(f"{outside} (outside the library")
    assert two["cells"][5] == "-" and two["judgment"] == "muted"
    f = out["filtered"]
    assert [x["cells"][1] for x in f["rows"]] == ["run7"] and f["count"] == "1 of 5 inspection(s)"
    # how the scanner was used, under the program
    assert two["cells"][4] == "measure:flatnessMeasure" and three["cells"][4].endswith("Station")
    # "REJECT and PULL only": the parts that did not continue (FAIL is REJECT), newest first
    b = out["bad"]
    assert [x["cells"][1] for x in b["rows"]] == ["run7", "run3 MOCK", "run1"]
    assert b["count"] == "3 of 5 inspection(s)" and b["empty"] is None
    # an early v1 file: read by its own header (MOCK still marked), sorted by the moment, each
    # time shown as the file holds it (the scanner's clock; the offset in the tooltip)
    assert out["early"] == [
        ["2026-11-01 01:10:00", "run22", "2026-11-01T01:10:00+00:00", "line1.json / long"],
        ["2026-11-01 01:30:00", "run21 MOCK", "2026-11-01T01:30:00+01:00", "line1.json / long"],
        ["2026-11-01 02:20:00", "run23", "2026-11-01T02:20:00+02:00", "line1.json / long"]]
    # another month: its own two files are asked for; none there is not an error
    sep = out["september"]
    assert out["asked"][-2:] == ["/files/events/inspections_v1_2026-09.csv",
                                 "/files/events/pending/inspections_v1_2026-09.csv"]
    assert sep["rows"] == [] and sep["empty"] == "No inspections recorded in 2026-09."
    assert sep["waiting"] is None and sep["error"] is None and sep["csv"] is None
    dec = out["december"]
    assert dec["month"] == "2026-12" and "could not be read" in dec["error"] and dec["rows"] == []
    assert out["many"] == {"n": 500, "more": "Show 120 more", "count": "620 inspection(s)"}
    assert out["stays"] == "2027-01"                        # not a month: nothing is asked
    assert out["parsed"] == [["a", 'b "x", y', "c"], ["1", "", "two\nlines"]]
    # nothing, empty lines, a quote that is never closed, quotes inside a cell, a quote as
    # a cell and a last line without its end, bare carriage returns
    assert out["odd"] == [[], [["a", "b"], ["c"]], [["never closed,\nrow"]],
                          [["abcd,efgh", "i"]], [['"', ""], ["", ""]], [["a"], ["b"], ["c"]]]
    assert out["urls"] == ["/files/events/img/a%20b.png", None, None, None, None]


def test_the_server_serves_the_page_and_the_log_with_the_routes_it_has(eng, eventlog_on, tmp_path):
    """No new route: the page is a static file, the csv files and the picture are library
    files (/files/...). A month with no file is a 404, which the page reads as no rows."""
    from test_ui_server import Client
    from ui.web import server as ui_server
    lib = eng.store.root
    run_id, _, doc = _job(lib / "runs", max_dev=0.1, threshold=0.2, picture="overlay.png")
    month = _month(doc)
    api._session = None
    (tmp_path / "stations").mkdir()
    httpd, app = ui_server.make_server(mock=True, library=lib, port=0,
                                       stations_dir=tmp_path / "stations")
    t = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
    t.start()
    try:
        c = Client(app.port)
        status, raw, hdrs = c.request("GET", "/static/events.html")
        assert status == 200 and hdrs["Content-Type"].startswith("text/html")
        assert b"<th>Inspection no.</th>" in raw and "script-src 'self'" in hdrs[
            "Content-Security-Policy"]
        assert c.request("GET", "/static/events.js")[0] == 200
        status, raw, hdrs = c.request("GET", f"/files/events/inspections_v1_{month}.csv")
        assert status == 200 and hdrs["Content-Type"].startswith("text/csv")
        (row,) = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
        assert row["inspection_id"] == run_id and row["judgment"] == "PASS"
        assert c.request("GET", f"/files/events/inspection_values_v1_{month}.csv")[0] == 200
        # the newest row alone, for a reader that wants no month of them (the Part ID)
        status, raw, hdrs = c.request("GET", "/files/events/last_v1.csv")
        assert status == 200 and hdrs["Content-Type"].startswith("text/csv")
        assert list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))) == [row]
        status, raw, hdrs = c.request("GET", "/files/" + row["image_file"])
        assert status == 200 and raw == PNG and hdrs["Content-Type"] == "image/png"
        assert c.request("GET", "/files/" + row["result_file"])[0] == 200
        assert c.request("GET", "/files/events/inspections_v1_1999-01.csv")[0] == 404
        assert c.request("GET", f"/files/events/pending/inspections_v1_{month}.csv")[0] == 404
    finally:
        httpd.shutdown()
        httpd.server_close()
        app.close()
        api._session = None

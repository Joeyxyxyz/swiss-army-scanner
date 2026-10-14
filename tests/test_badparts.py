"""Bad-part output and references made from good parts only (engine/badparts.py), MOCK only.

THE BAD-PART OUTPUT: off unless a folder is set (the run command's --bad-part-output, the
station configuration's, the recipe's outputs, SAS_BAD_PART_OUTPUT - in that order; "off" at
any of them). Every inspection whose disposition is REJECT or PULL gets a folder named by its
part ID (made safe; the inspection id without one; a repeat gets -2, never an overwrite) with
its result.json, picture, height profiles and 3D file, and one index row - REJECT and PULL
told apart. A good part, a measurement and the check of a work state inside a cycle write
nothing there. It works with the event log off; a failed write never changes a judgment or an
exit code, is said in the run's output and is copied later (never twice).

REFERENCES ARE GOOD PARTS ONLY: every verb that makes a reference refuses a source judged
REJECT / PULL (by an inspection - the hook marks its scans -, by a check whose result is
recorded against it, or one that lives in a bad-part output), and every compare refuses a
reference whose source was judged so since. No override.

The hand-built runs go through engine/runs.write_result (the hook) with no engine session;
the real runs check synthetic metric scans offline (tests/recipe_synth.py: nothing is
captured). SAS_BAD_PART_OUTPUT is off for every test (tests/conftest.py); a test's bad-part
output is under its temporary folder."""

import contextlib
import inspect
import io
import json
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import conftest
from engine import api, badparts, eventlog, rules
from engine import disposition as D
from engine import runs as R
from engine.flow import schema as FS
from engine.flow import verify as FV
from engine.measure import bank as bank_mod
from engine.measure.dents import verbs as dent_verbs
from engine.measure.endface import verbs as face_verbs
from engine.measure.station import verbs as station_verbs
from engine.modes import measure as MM
from engine.recipes import runner
from engine.recipes import schema as recipe_schema
from engine.results import FaceGolden, FaceProfile, ObjectScan, StationGolden, StationMeasurement
from engine.sense import virtual
from ui import cli
from ui import run_job as RJ

import recipe_synth as RS
from conftest import BAD_PART_ENV

PRESET = "bench_9in_metric_bright"
PNG = b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 4      # stands in for a run's picture
GEO = "rig_a"
RULE = "references are made from GOOD parts only"


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.setattr(badparts, "_COMMAND_LINE", None)
    yield
    api._session = None


# ================================================================== helpers

def _job(runs, name="plate_check", *, result="PASS", max_dev=0.1, threshold=0.2, codes=None,
         root=None, picture="overlay.png", profiles=False, scan_id=None, settle=None,
         job=None) -> tuple:
    """One run folder as the station job runner writes it (run log with the codes read, the
    picture, the height profiles), result.json through runs.write_result - the hook. root:
    the bad-part output its station configuration names (as run_job puts it in the
    document). -> (run id, run folder, the result as the run returns it)."""
    run_id, run_dir = R.new_run(runs, name)
    res = R.empty_result(run_id=run_id, run_folder=str(run_dir), mock=True,
                         operator_confirmed=False, station=name,
                         job=job or {"type": "metric", "base": None, "zone": None,
                                     "config": f"{name}.json"},
                         part={"role": "test", "label": "plate"}, threshold_mm=threshold,
                         station_config=f"{name}.json", profile="default")
    if root is not None:
        res["outputs"]["bad_part_output"] = str(root)
    log = R.RunLog(run_dir / "run.log.jsonl", run_id)
    log.event("run_start", station=name, mode="MOCK")
    if codes:
        log.event("codes", n=len(codes), data=list(codes))
    res.update(result=result, engine_verdict=result,
               invalid_reason=None if result in ("PASS", "FAIL") else f"result {result}")
    res["capture"].update(preset="p1", geometry_id=GEO)
    res["deviation_mm"].update(max=max_dev, method="plane_fit")
    res["deviation_units"] = "mm"
    res["part"]["scan_id"] = scan_id
    if picture:
        (run_dir / picture).write_bytes(PNG)
    if profiles:
        for n in ("profile_x.csv", "profile_y.csv"):
            (run_dir / n).write_text("position_mm,height_mm\n0.000,0.01000\n", encoding="utf-8")
    (settle or D.settle)(res)
    R.write_result(run_dir, res)
    log.event("run_end", result=res["result"], disposition=res["disposition"])
    log.close()
    return run_id, run_dir, res


def _index(root) -> list:
    return eventlog.read_rows(Path(root) / badparts.INDEX)


def _folders(root) -> list:
    """The parts' folders of a bad-part output (not pending/, where index rows wait)."""
    return sorted(p.name for p in Path(root).iterdir() if p.is_dir()
                  and p.name != badparts.PENDING_DIR) if Path(root).is_dir() else []


def _entry(lib, kind="metric", sid="s1", geo=GEO) -> Path:
    """A library entry folder (a scan, a face scan, a dent set ...): nothing in it yet."""
    d = Path(lib) / kind / geo / sid if kind != "objects" else Path(lib) / kind / sid
    d.mkdir(parents=True, exist_ok=True)
    return d


def _judged_by_check(lib, entry, doc: dict, kind="defects") -> Path:
    """A check verb's result recorded against an entry, as LibraryStore.record_output does:
    the JSON in the output bin, a line in the entry's exports.jsonl."""
    out = Path(lib) / "output" / kind / f"{entry.name}_{len(list(entry.glob('*')))}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc), encoding="utf-8")
    with open(entry / "exports.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"kind": kind, "file": out.relative_to(lib).as_posix()}) + "\n")
    return out


def _judged_by_run(lib, sid, disposition="REJECT", run_id="20261013T112800Z_line_0000beef"):
    """An inspection's mark on a scan, as the hook writes it (badparts.stamp)."""
    return badparts.stamp(lib, {"run_id": run_id, "part": {"scan_id": sid},
                                "capture": {"geometry_id": GEO}, "disposition": disposition},
                          Path(lib) / "runs" / run_id, disposition)


# ================================================================== the bad-part output

def test_a_rejected_and_a_pulled_part_each_get_a_folder_and_an_index_row(tmp_path):
    """A REJECT and a PULL: a folder each, named by the part ID, with the run's files; one
    index row each, REJECT and PULL told apart, with the values out of limit. A good part
    writes nothing. The event log is off (as for every test): the output does not need it."""
    root, runs = tmp_path / "bad", tmp_path / "runs"
    assert not eventlog.enabled()
    _job(runs, root=root)                                   # CONTINUE
    assert not root.exists()
    rid, rdir, rej = _job(runs, root=root, result="FAIL", max_dev=0.31, codes=["SN-0042"],
                          profiles=True)
    pid, pdir, pul = _job(runs, root=root, result="INVALID", codes=["SN-0043"])
    assert (rej["disposition"], pul["disposition"]) == ("REJECT", "PULL")
    assert (D.exit_code(rej), D.exit_code(pul)) == (1, 5)  # the judgments are the runs' own
    assert rej["warnings"] == [] and pul["warnings"] == []
    assert sorted(p.name for p in root.iterdir()) == [
        "BAD_PART_OUTPUT.txt", "SN-0042", "SN-0043", "bad_parts.lock", "bad_parts_v1.csv"]
    assert "is ever used as a reference" in (root / "BAD_PART_OUTPUT.txt").read_text()
    assert sorted(p.name for p in (root / "SN-0042").iterdir()) == [
        "overlay.png", "profile_x.csv", "profile_y.csv", "result.json"]
    assert (root / "SN-0042" / "result.json").read_bytes() == (rdir / "result.json").read_bytes()
    assert (root / "SN-0042" / "overlay.png").read_bytes() == PNG
    assert sorted(p.name for p in (root / "SN-0043").iterdir()) == ["overlay.png", "result.json"]
    # the index: one row per bad part, as a spreadsheet reads it
    raw = (root / badparts.INDEX).read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf" + ",".join(badparts.INDEX_FIELDS).encode() + b"\r\n")
    r1, r2 = _index(root)
    assert list(r1) == list(badparts.INDEX_FIELDS)
    assert (r1["part_id"], r1["inspection_id"], r1["program"], r1["operating_mode"],
            r1["hardware"], r1["disposition"], r1["folder"]) == (
        "SN-0042", rid, "plate_check.json", "station", "MOCK", "REJECT", "SN-0042")
    assert r1["reason"] == "deviation_mm.max 0.31 mm is outside its limits [..0.2]"
    assert r1["out_of_limit"] == "metric: deviation_mm.max=0.31 mm [..0.2]"
    assert r1["time"] == eventlog.local_time(json.loads(
        (rdir / "result.json").read_text(encoding="utf-8"))["finished"])
    assert (r2["part_id"], r2["disposition"], r2["folder"]) == ("SN-0043", "PULL", "SN-0043")
    assert r2["reason"].startswith("invalid: result INVALID") and r2["out_of_limit"] == ""
    # the run says where: its result.json names the output, its folder the part's folder
    on_disk = json.loads((rdir / "result.json").read_text(encoding="utf-8"))
    assert on_disk["outputs"]["bad_part_output"] == str(root.resolve())
    done = json.loads((rdir / badparts.DONE_FILE).read_text(encoding="utf-8"))
    assert Path(done["folder"]).resolve() == (root / "SN-0042").resolve()
    assert done["disposition"] == "REJECT"
    assert not (runs / "events").exists()                   # the event log stayed off


def test_the_folder_is_the_part_id_made_safe_else_the_inspection_id_never_reused(tmp_path):
    root, runs = tmp_path / "bad", tmp_path / "runs"
    fn = badparts.folder_name
    assert fn("SN 42/7", "run1") == "SN_42_7" and fn("  ..x.. ", "run1") == "x"
    assert fn("CON", "run1") == "_CON" and fn("nul.txt", "run1") == "_nul.txt"
    assert fn("", "run1") == fn(None, "run1") == fn("///", "run1") == "run1"
    assert fn("=cmd|' /C calc'!A0", "run1") == "cmd_C_calc_A0"
    assert fn("pending", "run1") == "_pending"              # the output's own folder
    for own in ("bad_parts_v1.csv", "BAD_PART_OUTPUT.txt", "Bad_Parts.lock",
                "bad_parts_pending.lock"):                  # ... and files: never blocked
        assert fn(own, "run1") == "_" + own
    assert len(fn("A" * 300, "run1")) == badparts.MAX_NAME
    a, _, _ = _job(runs, root=root, result="FAIL", max_dev=0.3, codes=["SN-1"])
    b, _, _ = _job(runs, root=root, result="FAIL", max_dev=0.3, codes=["SN-1"])   # again
    c, _, _ = _job(runs, root=root, result="INVALID")                             # no code
    d, _, _ = _job(runs, root=root, result="INVALID", codes=["=SN|2"])            # foreign
    assert _folders(root) == sorted(["SN-1", "SN-1-2", c, "SN_2"])
    rows = _index(root)
    assert [(r["inspection_id"], r["folder"]) for r in rows] == [
        (a, "SN-1"), (b, "SN-1-2"), (c, c), (d, "SN_2")]
    assert rows[3]["part_id"] == "'=SN|2"                   # a spreadsheet does not run it
    # the first bad part of an output, with the index's own name for a code: the index stays
    e, _, _ = _job(runs, root=tmp_path / "bad2", result="INVALID", codes=["bad_parts_v1.csv"])
    assert _folders(tmp_path / "bad2") == ["_bad_parts_v1.csv"]
    assert [r["inspection_id"] for r in _index(tmp_path / "bad2")] == [e]


def test_nothing_for_a_good_part_a_measurement_or_a_check_inside_a_cycle(eng, tmp_path,
                                                                          bad_part_output):
    """SAS_BAD_PART_OUTPUT set (this computer's default): a good part, a measurement (it judges
    nothing) and the check of a work state inside a cycle (its cycle is the inspection) write
    nothing there. The check inside the cycle still marks its scan when it judged it bad."""
    root, runs = bad_part_output, tmp_path / "runs"
    assert os.environ[BAD_PART_ENV] == str(root)
    _job(runs)
    p = SimpleNamespace(name="p1", geometry_id=GEO, rig=SimpleNamespace(mount="fixed"))
    spec = MM.spec_from({"what": "flatness", "roi": [0, 0, 10, 10]})
    mid, mdir = R.new_run(runs, "measure_flatness")
    doc = MM.empty_doc(run_id=mid, run_folder=str(mdir), mock=True, preset=p, spec=spec,
                       label="m")
    doc.update(status="MEASURED", reason=None)
    MM._write_doc(Path(mdir), doc)                          # Measure's own hook
    assert badparts.record(mdir) is None
    # a cycle and the REJECT of its first work state's check
    sid = "20261013T120000_000001_t_long"
    entry = _entry(eng.store.root, "metric", sid)
    cid, cdir = R.new_run(runs, "line1")
    clog = R.RunLog(cdir / "run.log.jsonl", cid)
    clog.event("run_start", station="line1", mode="MOCK", job="flow")
    _, sdir, sres = _job(cdir / FV.STEPS_DIR, "line1_loaded", result="FAIL", max_dev=0.3,
                         scan_id=sid, job={"type": "recipe", "work_state": "loaded",
                                           "config": "line1.json"})
    clog.close()
    assert sres["disposition"] == "REJECT" and eventlog.in_cycle(sdir, sres)
    assert not root.exists()
    (mark,) = badparts._jsonl(entry / badparts.STAMP)       # its own judgment marks its scan
    assert (mark["inspection_id"], mark["disposition"]) == (sres["run_id"], "REJECT")


def test_a_failed_copy_never_changes_the_judgment_is_said_and_is_copied_later(tmp_path):
    """The output cannot be written (here: a file stands where its folder should be): the
    run's judgment, result.json and exit code are what they would have been; the run's output
    says it; the copy waits and is made with the next bad part - once."""
    blocker = tmp_path / "share"
    blocker.write_text("not a folder")
    root, runs = blocker / "bad", tmp_path / "runs"
    rid, rdir, res = _job(runs, root=root, result="FAIL", max_dev=0.31, codes=["SN-9"])
    assert (res["disposition"], D.exit_code(res)) == ("REJECT", 1)
    (warn,) = res["warnings"]                               # the run's output says it
    assert warn.startswith(f"bad-part output: inspection {rid} (REJECT) NOT copied to "
                           f"{root.resolve()}: ")
    assert "it waits" in warn and "the judgment is unchanged" in warn
    on_disk = json.loads((rdir / "result.json").read_text(encoding="utf-8"))
    assert on_disk["warnings"] == [] and on_disk["disposition"] == "REJECT"   # as written
    note = runs / badparts.QUEUE_DIR / f"{rid}.json"        # no session: beside the runs
    assert json.loads(note.read_text(encoding="utf-8"))["inspection_id"] == rid
    assert [w["inspection_id"] for w in badparts.waiting(runs)] == [rid]
    blocker.unlink()                                        # the share is back
    rid2, _, res2 = _job(runs, root=root, result="INVALID", codes=["SN-10"])
    assert res2["warnings"] == [] and not note.exists() and badparts.waiting(runs) == []
    assert _folders(root) == ["SN-10", "SN-9"]
    assert [r["inspection_id"] for r in _index(root)] == [rid, rid2]
    # never twice: the hook again, a retry, a copy by hand
    badparts.record(rdir)
    assert badparts.retry(runs) == {"done": 0, "waiting": 0}
    assert badparts.copy_run(rdir)["folder"].resolve() == (root / "SN-9").resolve()
    assert _folders(root) == ["SN-10", "SN-9"] and len(_index(root)) == 2


def test_waiting_copies_are_copied_by_the_command_line(tmp_path, capsys):
    blocker = tmp_path / "share"
    blocker.write_text("not a folder")
    root, runs = blocker / "bad", tmp_path / "runs"
    rid, rdir, _ = _job(runs, root=root, result="INVALID")
    assert badparts.main(["--retry", "--library", str(runs)]) == 1         # still blocked
    assert "0 copied; 1 still waiting" in capsys.readouterr().out
    blocker.unlink()
    assert badparts.main(["--retry", "--library", str(runs)]) == 0
    assert "1 copied; 0 still waiting" in capsys.readouterr().out
    assert _folders(root) == [rid]
    # by hand: a good part is refused, a bad one goes where it is told
    gid, gdir, _ = _job(runs)
    assert badparts.main(["--copy", str(gdir), "--to", str(tmp_path / "x")]) == 2
    assert "only a part judged REJECT or PULL" in capsys.readouterr().err
    assert badparts.main(["--copy", str(rdir), "--to", str(tmp_path / "other")]) == 0
    assert _folders(tmp_path / "other") == [rid]


def test_an_index_open_in_a_spreadsheet_program_makes_its_row_wait(tmp_path, monkeypatch):
    root, runs = tmp_path / "bad", tmp_path / "runs"
    real = badparts._open

    def refusing(path, mode):
        if Path(path) == root / badparts.INDEX and "b" in mode and "a" in mode:
            raise PermissionError(13, "the file is open in another program", str(path))
        return real(path, mode)
    a, _, _ = _job(runs, root=root, result="INVALID", codes=["SN-1"])
    with monkeypatch.context() as m:
        m.setattr(badparts, "_open", refusing)
        b, _, res = _job(runs, root=root, result="FAIL", max_dev=0.3, codes=["SN-2"])
    assert res["disposition"] == "REJECT" and _folders(root) == ["SN-1", "SN-2"]
    (warn,) = res["warnings"]
    assert warn.startswith(f"bad-part output: {root.resolve() / 'SN-2'} written; its row waits")
    assert [r["inspection_id"] for r in _index(root)] == [a]
    pending = root / badparts.PENDING_DIR / badparts.INDEX
    assert [r["inspection_id"] for r in eventlog.read_rows(pending)] == [b]
    c, _, _ = _job(runs, root=root, result="INVALID", codes=["SN-3"])     # closed again
    assert [r["inspection_id"] for r in _index(root)] == [a, b, c] and not pending.exists()


def test_abort_after_the_verdict_makes_a_good_part_a_pull_copied_once(tmp_path):
    root, runs = tmp_path / "bad", tmp_path / "runs"
    gid, gdir, _ = _job(runs, root=root, codes=["SN-5"])
    assert not root.exists()
    R.append_event(gdir / "run.log.jsonl", gid, eventlog.ABORTED_EVENT,
                   message="ABORT was pressed after this run wrote its verdict")
    (row,) = _index(root)
    assert (row["inspection_id"], row["disposition"], row["folder"]) == (gid, "PULL", "SN-5")
    assert row["reason"].startswith("aborted: ABORT was pressed")
    pid, pdir, _ = _job(runs, root=root, result="INVALID", codes=["SN-6"])
    R.append_event(pdir / "run.log.jsonl", pid, eventlog.ABORTED_EVENT)
    assert _folders(root) == ["SN-5", "SN-6"] and len(_index(root)) == 2  # once each


@pytest.mark.skipif(sys.platform != "win32", reason="the Windows path limit")
def test_a_path_longer_than_windows_allows_is_refused_with_its_reason(tmp_path, monkeypatch):
    root, runs = tmp_path / "bad", tmp_path / "runs"
    monkeypatch.setattr(badparts, "MAX_PATH", len(str(root)) + 10)
    rid, _, res = _job(runs, root=root, result="INVALID", codes=["SN-0000000042"])
    (warn,) = res["warnings"]
    assert "characters long: Windows allows" in warn and "choose a shorter" in warn
    assert res["disposition"] == "PULL" and badparts.waiting(runs)[0]["inspection_id"] == rid


def test_where_it_is_set_the_command_line_then_the_configuration_then_the_computer(
        tmp_path, monkeypatch):
    here = tmp_path / "here"
    dest = badparts.destination
    assert os.environ[BAD_PART_ENV] == "off" and dest({"outputs": {}}) is None
    monkeypatch.setenv(BAD_PART_ENV, str(tmp_path / "machine"))
    assert dest({"outputs": {}}) == str((tmp_path / "machine").resolve())
    assert dest({"outputs": {"bad_part_output": str(here)}}) == str(here.resolve())
    assert dest({"outputs": {"bad_part_output": "OFF"}}) is None      # off beats the default
    badparts.set_command_line(str(tmp_path / "cli"))
    assert dest({"outputs": {"bad_part_output": str(here)}}) == str((tmp_path / "cli").resolve())
    badparts.set_command_line("off")
    assert dest({"outputs": {"bad_part_output": str(here)}}) is None
    badparts.set_command_line(None)
    assert badparts.configured("bad", tmp_path) == str((tmp_path / "bad").resolve())
    assert badparts.configured(" ") is None and badparts.configured(None) is None
    with pytest.raises(ValueError, match="folder path"):
        badparts.configured(5)
    # a recipe: outputs.bad_part_output, relative to its file; the station's wins
    doc = {"schema_version": 1, "name": "lid", "preset": PRESET,
           "zones": [{"id": "z", "shape": "rect", "units": "part_mm", "coords": [-30, -20, 30, 20],
                      "checks": [{"type": "presence", "params": {"min_height_mm": 5}}]}],
           "outputs": {"bad_part_output": "bins/bad"}}
    rec = recipe_schema.validate(doc, source=tmp_path / "recipes" / "lid.json")
    res = {"outputs": {"bad_part_output": None}}
    runner.bad_part_output(res, rec)
    want = str((tmp_path / "recipes" / "bins" / "bad").resolve())
    assert res["outputs"]["bad_part_output"] == want
    res = {"outputs": {"bad_part_output": "off"}}           # the station said off
    runner.bad_part_output(res, rec)
    assert res["outputs"]["bad_part_output"] == "off"
    with pytest.raises(recipe_schema.RecipeError, match="outputs.bad_part_output"):
        recipe_schema.validate({**doc, "outputs": {"bad_part_output": 5}})
    # a single-job station config: beside runs_dir, relative to its file
    cfg = RJ.config_from_doc({"station_id": "st", "bad_part_output": "bad",
                              "job": {"type": "metric", "threshold_mm": 1}}, tmp_path / "st")
    assert cfg.bad_part_output == str((tmp_path / "st" / "bad").resolve())
    res = RJ.empty_result(cfg, "run1", "runs/run1", True, False)
    assert res["outputs"]["bad_part_output"] == cfg.bad_part_output
    with pytest.raises(RJ.StationConfigError, match="bad_part_output"):
        RJ.config_from_doc({"station_id": "st", "bad_part_output": ["x"]}, tmp_path)
    # a station configuration with a flow: parameters.bad_part_output, never at the top
    src = conftest.ROOT / "stations" / "example_tube_line.json"      # (read, never written)
    flow = json.loads(src.read_text(encoding="utf-8"))
    flow["parameters"] = {**flow.get("parameters", {}), "bad_part_output": "line_bad"}
    fcfg = FS.validate(flow, source=src)
    assert FV.bad_part_output(fcfg) == str((src.parent / "line_bad").resolve())
    with pytest.raises(FS.StationConfigError, match="parameters.bad_part_output"):
        FS.validate({**flow, "parameters": {**flow["parameters"], "bad_part_output": 1}},
                    source=src)
    with pytest.raises(FS.StationConfigError, match="belong under"):
        FS.validate({**flow, "bad_part_output": "x"}, source=src)
    legacy = FS.lift_legacy({"station_id": "st", "bad_part_output": "b", "job": {}})
    assert legacy.parameters["bad_part_output"] == "b"


def _share(monkeypatch, share: Path) -> list:
    """A network folder whose path the OS cannot resolve (realpath raises an error it does
    not take, as a share does that answers late: WinError 121), and every attempt to make a
    folder there noted. -> the notes [(what, path)]."""
    touched = []
    real_realpath, real_mkdir = os.path.realpath, Path.mkdir

    def realpath(p, *a, **k):
        if str(p).startswith(str(share)):
            touched.append(("realpath", str(p)))
            raise OSError(22, "The semaphore timeout period has expired", str(p), 121)
        return real_realpath(p, *a, **k)

    def mkdir(self, *a, **k):
        if str(self).startswith(str(share)):
            touched.append(("mkdir", str(self)))
        return real_mkdir(self, *a, **k)
    monkeypatch.setattr(os.path, "realpath", realpath)
    monkeypatch.setattr(Path, "mkdir", mkdir)
    return touched


def test_the_setting_is_text_until_a_bad_part_is_copied_and_never_stops_a_run(tmp_path,
                                                                             monkeypatch):
    """A folder whose path the OS cannot resolve: every place that reads the setting takes it
    as text (a good part never touches the folder), every run writes its result.json with its
    own judgment, and a bad part is copied. A setting that is no folder path copies nothing
    and says so: it never falls through to the next level."""
    share = tmp_path / "share"
    touched = _share(monkeypatch, share)
    want = os.path.abspath(share / "bad")
    monkeypatch.setenv(BAD_PART_ENV, str(share / "bad"))
    runs = tmp_path / "runs"
    gid, gdir, good = _job(runs, codes=["SN-OK"])
    assert (good["disposition"], D.exit_code(good), good["warnings"]) == ("CONTINUE", 0, [])
    assert touched == []                                    # not even looked at
    on_disk = json.loads((gdir / "result.json").read_text(encoding="utf-8"))
    assert on_disk["outputs"]["bad_part_output"] == want
    rid, rdir, bad = _job(runs, result="FAIL", max_dev=0.3, codes=["SN-1"])
    assert (bad["disposition"], D.exit_code(bad), bad["warnings"]) == ("REJECT", 1, [])
    assert _folders(share / "bad") == ["SN-1"]
    # the configurations that name it load as ever
    cfg = RJ.config_from_doc({"station_id": "st", "bad_part_output": str(share / "bad"),
                              "job": {"type": "metric", "threshold_mm": 1}}, tmp_path)
    assert cfg.bad_part_output == want
    doc = {"schema_version": 1, "name": "lid", "preset": PRESET,
           "zones": [{"id": "z", "shape": "rect", "units": "part_mm", "coords": [-30, -20, 30, 20],
                      "checks": [{"type": "presence", "params": {"min_height_mm": 5}}]}],
           "outputs": {"bad_part_output": "../share/bad"}}
    rec = recipe_schema.validate(doc, source=tmp_path / "recipes" / "lid.json")
    assert runner.recipe_bad_part_output(rec) == want
    res = {"outputs": {"bad_part_output": None}}
    runner.bad_part_output(res, rec)
    assert res["outputs"]["bad_part_output"] == want
    flow = SimpleNamespace(parameters={"bad_part_output": "share/bad"}, source=tmp_path / "l.json")
    assert FV.bad_part_output(flow) == want
    # no folder path at the level that says: nothing copied, said - not the computer's
    res = {"outputs": {"bad_part_output": 5}, "warnings": []}
    badparts.settle(res)
    assert res["outputs"]["bad_part_output"] is None
    assert res["warnings"] == ["bad-part output: outputs.bad_part_output: bad_part_output must "
                               "be a folder path, \"off\" or null, got 5: no part is copied"]


def test_a_folder_that_is_holds_or_lies_in_the_library_or_a_runs_folder_is_refused(tmp_path,
                                                                                  capsys):
    """Else everything the scanner checks against would lie in a bad-part output (every
    later part a PULL), or the bad parts would be cleaned with the runs. Nothing is written
    there, nothing waits; the judgment is the run's own."""
    runs = tmp_path / "data" / "runs"           # no session: the runs folder is the library
    for root, how in ((runs, "is"), (tmp_path / "data", "holds"), (runs / "bad", "lies inside")):
        rid, rdir, res = _job(runs, root=root, result="FAIL", max_dev=0.3, codes=["SN-1"])
        assert (res["disposition"], D.exit_code(res)) == ("REJECT", 1)
        (warn,) = res["warnings"]
        assert warn.startswith(f"bad-part output: inspection {rid} (REJECT) NOT copied to "
                               f"{root}: the folder {how} the library {runs}: a bad-part output "
                               "must be a folder of its own, outside the library and the runs "
                               "folders"), warn
        assert "the judgment is unchanged" in warn
    assert list(tmp_path.rglob(badparts.MARKER)) == [] and badparts.waiting(runs) == []
    assert not (runs / "bad").exists()
    with pytest.raises(ValueError, match="must be a folder of its own"):
        badparts.copy_run(rdir, runs)
    assert badparts.main(["--copy", str(rdir), "--to", str(tmp_path)]) == 2
    assert "must be a folder of its own" in capsys.readouterr().err


def test_a_library_inside_a_bad_part_output_says_how_to_get_out(eng, tmp_path):
    """A marker above the library (put there by hand): every entry is refused - fail closed -
    and the refusal says that the whole library lies in the output, and what to do."""
    lib = eng.store.root
    _entry(lib, "metric", "s1")
    (tmp_path / badparts.MARKER).write_text("x", encoding="utf-8")
    with pytest.raises(badparts.BadPartReference) as e:
        badparts.require_good(eng.store, "recipe golden", ["s1"], sweep=True)
    assert f"s1 lives in a bad-part output ({tmp_path}), and so does the whole library {lib}" \
        in str(e.value)
    assert "set the bad-part output to a folder outside the library" in str(e.value)


def test_a_folder_that_does_not_answer_holds_a_bad_part_up_by_one_attempt(tmp_path,
                                                                         monkeypatch):
    """The output is a network folder that is down: each bad part tries it once, however many
    copies wait; a good part never. When it answers again, the next bad part copies the
    oldest waiting ones first (RETRY_MAX of them), then its own; `--retry` the rest."""
    runs, share = tmp_path / "runs", tmp_path / "share"
    root = share / "bad"
    attempts, down = [], [True]
    real_mkdir = Path.mkdir

    def mkdir(self, *a, **k):
        if str(self).startswith(str(share)):
            attempts.append(str(self))
            if down[0]:
                raise OSError(22, "The network path was not found", str(self), 53)
        return real_mkdir(self, *a, **k)
    monkeypatch.setattr(Path, "mkdir", mkdir)
    ids = []
    for i in range(badparts.RETRY_MAX + 2):
        n = len(attempts)
        rid, _, res = _job(runs, root=root, result="FAIL", max_dev=0.3, codes=[f"SN-{i}"])
        assert len(attempts) - n == 1                       # once, however many wait
        assert res["disposition"] == "REJECT" and "NOT copied" in res["warnings"][0]
        ids.append(rid)
    n = len(attempts)
    _job(runs, root=root, codes=["SN-OK"])
    assert len(attempts) == n                               # a good part: never
    assert len(badparts.waiting(runs)) == len(ids)
    down[0] = False                                         # the folder answers again
    rid, _, res = _job(runs, root=root, result="INVALID", codes=["SN-NEW"])
    assert res["warnings"] == []
    assert [r["inspection_id"] for r in _index(root)] == ids[:badparts.RETRY_MAX] + [rid]
    assert len(badparts.waiting(runs)) == len(ids) - badparts.RETRY_MAX
    assert badparts.main(["--retry", "--library", str(runs)]) == 0
    assert sorted(r["inspection_id"] for r in _index(root)) == sorted(ids + [rid])
    assert len(_folders(root)) == len(ids) + 1 and badparts.waiting(runs) == []


def test_a_copy_cut_short_is_never_taken_for_a_whole_one(tmp_path, monkeypatch):
    """The disk fills up half way through a file: no file of the part's folder is ever half
    a file (each is copied under a temporary name, then renamed); the retry fills up the same
    folder - never a second one - and makes again a file that is not whole."""
    root, runs = tmp_path / "bad", tmp_path / "runs"
    real = shutil.copyfileobj
    fail = {"name": None}

    def half(s, d, *a, **k):
        if fail["name"] and Path(d.name).name.startswith(fail["name"]):
            d.write(s.read()[:10])
            raise OSError(28, "No space left on device")
        return real(s, d, *a, **k)
    monkeypatch.setattr(shutil, "copyfileobj", half)
    fail["name"] = "result.json"
    rid, rdir, res = _job(runs, root=root, result="FAIL", max_dev=0.3, codes=["SN-DF"],
                          profiles=True)
    assert res["disposition"] == "REJECT" and "No space left on device" in res["warnings"][0]
    assert list((root / "SN-DF").iterdir()) == []          # nothing half there
    (note,) = badparts.waiting(runs)
    assert Path(note["folder"]) == root / "SN-DF"
    fail["name"] = "profile_y.csv"
    assert badparts.retry(runs) == {"done": 0, "waiting": 1}
    assert sorted(p.name for p in (root / "SN-DF").iterdir()) == [
        "overlay.png", "profile_x.csv", "result.json"]
    (root / "SN-DF" / "overlay.png").write_bytes(PNG[:5])   # a file cut short (not ours)
    fail["name"] = None
    assert badparts.retry(runs) == {"done": 1, "waiting": 0}
    assert _folders(root) == ["SN-DF"]                      # filled up, never made twice
    assert sorted(p.name for p in (root / "SN-DF").iterdir()) == [
        "overlay.png", "profile_x.csv", "profile_y.csv", "result.json"]
    for f in (root / "SN-DF").iterdir():
        assert f.read_bytes() == (rdir / f.name).read_bytes()
    assert [r["inspection_id"] for r in _index(root)] == [rid]


def test_an_index_changed_in_another_program_is_moved_aside_and_goes_on(tmp_path):
    """bad_parts_v1.csv opened in a spreadsheet program, a column added and saved: the next
    bad part moves it aside as it is (never deleted), carries its rows on in a new index
    under the v1 header, adds its own and says so - the index never stops growing."""
    root, runs = tmp_path / "bad", tmp_path / "runs"
    a, _, _ = _job(runs, root=root, result="INVALID", codes=["SN-1"])
    b, _, _ = _job(runs, root=root, result="INVALID", codes=["SN-2"])
    index = root / badparts.INDEX
    edited = eventlog._BOM + eventlog._encode(
        badparts.INDEX_FIELDS + ("qa_note",), [{**r, "qa_note": "seen"} for r in _index(root)],
        header=True)
    index.write_bytes(edited)
    c, _, res = _job(runs, root=root, result="FAIL", max_dev=0.3, codes=["SN-3"])
    assert res["disposition"] == "REJECT"
    (warn,) = res["warnings"]
    assert warn.startswith(f"bad-part output: {index} was changed in another program "
                           "(column(s) added: qa_note): it was moved to ")
    assert "open a copy of the index in another program, never the index itself" in warn
    (aside,) = root.glob(f"replaced_*/{badparts.INDEX}")
    assert aside.read_bytes() == edited                    # as it was
    rows = _index(root)
    assert list(rows[0]) == list(badparts.INDEX_FIELDS)
    assert [r["inspection_id"] for r in rows] == [a, b, c]
    # a header that is none of the index's: moved aside, a new index begins
    index.write_bytes(b"a,b\r\n1,2\r\n")
    d, _, res = _job(runs, root=root, result="INVALID", codes=["SN-4"])
    assert f"{index} had a header that is not the v1 header (a, b)" in res["warnings"][0]
    assert [r["inspection_id"] for r in _index(root)] == [d]
    assert len(list(root.glob(f"replaced_*/{badparts.INDEX}"))) == 2


# ================================================================== what counts as judged bad

def test_what_counts_as_a_part_judged_bad(eng, tmp_path):
    lib = eng.store.root
    f = badparts.findings
    plain = _entry(lib, "metric", "plain")
    assert f(eng.store, [plain]) == [] and f(eng.store, ["plain", None]) == []
    # an inspection's mark (the hook): REJECT and PULL alike
    for sid, disp in (("rej", "REJECT"), ("pul", "PULL")):
        _entry(lib, "metric", sid)
        _judged_by_run(lib, sid, disp)
        (hit,) = f(eng.store, [sid])
        assert (hit["item"], hit["disposition"]) == (sid, disp)
        assert hit["why"].startswith(f"was judged {disp} by inspection ")
    # a check's result recorded against the entry: its disposition (face / station check),
    # else - a verb that gives none - a definite bad verdict; a verdict it could not give
    # (uncalibrated, no band, an empty bank) judged no part
    cases = [({"verdict": "FAIL", "disposition": "REJECT"}, "REJECT"),
             ({"verdict": "INCOMPLETE", "disposition": "PULL"}, "PULL"),
             ({"verdict": "PASS", "disposition": "CONTINUE"}, None),
             ({"result": "FAIL"}, "REJECT"), ({"result": "PASS"}, None),          # compare
             ({"result": "INDETERMINATE"}, None),                                 # (px, no band)
             ({"verdict": "FLAG"}, "REJECT"), ({"verdict": "CLEAR"}, None),       # dents
             ({"result": "FLAG", "bank": "b", "n_ref": 3}, "REJECT"),            # bank check
             ({"result": "INDETERMINATE", "bank": "b", "n_ref": 0}, None),       # an empty bank
             ({"status": "MEASURED", "disposition": "NONE"}, None)]              # a measurement
    for i, (doc, want) in enumerate(cases):
        e = _entry(lib, "facescans", f"fs{i}")
        _judged_by_check(lib, e, doc, "anomalies" if "bank" in doc else "defects")
        got = f(eng.store, [e])
        assert [x["disposition"] for x in got] == ([want] if want else []), doc
    e = _entry(lib, "facescans", "measured")                # not a judging kind of output
    _judged_by_check(lib, e, {"result": "FAIL"}, "measurements")
    assert f(eng.store, [e]) == []
    e = _entry(lib, "facescans", "unreadable")              # a record that cannot be read: a doubt
    _judged_by_check(lib, e, {}).write_text("{not json", encoding="utf-8")
    assert [x["disposition"] for x in f(eng.store, [e])] == ["PULL"]
    e = _entry(lib, "facescans", "gone")                    # a record that is gone: nothing says
    _judged_by_check(lib, e, {"verdict": "FAIL"}).unlink()
    assert f(eng.store, [e]) == []
    # where it lives: a bad-part output, a run folder judged REJECT / PULL
    out = tmp_path / "bad"
    _job(tmp_path / "runs", root=out, result="INVALID", codes=["SN-1"])
    (hit,) = f(eng.store, [out / "SN-1" / "result.json"])
    assert hit["why"] == f"lives in a bad-part output ({out.resolve()})"
    _, rdir, _ = _job(tmp_path / "runs", result="FAIL", max_dev=0.3)
    _, gdir, _ = _job(tmp_path / "runs")
    (hit,) = f(eng.store, [rdir / "overlay.png"])
    assert hit["why"] == f"lies in the run folder of inspection {rdir.name}, judged REJECT"
    assert f(eng.store, [gdir / "overlay.png"]) == []
    # a run made before the marks existed: found when a reference is made (sweep)
    _entry(lib, "metric", "old")
    rid, _, _ = _job(lib / "runs", result="FAIL", max_dev=0.3, scan_id="old")
    (lib / "metric" / GEO / "old" / badparts.STAMP).unlink()     # as if before the marks
    assert f(eng.store, ["old"]) == []
    (hit,) = f(eng.store, ["old"], sweep=True)
    assert (hit["disposition"], hit["why"]) == ("REJECT", f"was judged REJECT by inspection {rid}")


def test_the_refusal_states_the_rule_and_has_no_override(eng):
    _entry(eng.store.root, "metric", "s1")
    _judged_by_run(eng.store.root, "s1", "PULL", run_id="r1")
    with pytest.raises(badparts.BadPartReference) as e:
        badparts.require_good(eng.store, "bank_add (bank 'b')", ["s1"], how="use another")
    assert str(e.value) == ("bank_add (bank 'b'): s1 was judged PULL by inspection r1. The rule: "
                            f"{RULE}: a part judged REJECT or PULL is kept for tracking, never "
                            "checked against. Use another.")
    assert isinstance(e.value, rules.RuleViolation)        # a refusal: exit 2
    words = set(inspect.signature(badparts.require_good).parameters)
    assert words == {"store", "what", "items", "sweep", "how"}
    assert not [w for w in words if w in ("force", "allow", "override", "anyway")]
    # it says why the inspection judged it so: a golden checked as the part is a PULL too
    _entry(eng.store.root, "metric", "g1")
    why = "the golden g1 is the scan being checked: a part is never compared with itself"
    badparts.stamp(eng.store.root, {"run_id": "r2", "part": {"scan_id": "g1"},
                                    "disposition": "PULL", "result": "INVALID",
                                    "pull_reasons": [{"reason": "golden", "message": why}]},
                   eng.store.root / "runs" / "r2", "PULL")
    with pytest.raises(badparts.BadPartReference) as e:
        badparts.require_good(eng.store, "the golden of recipe r", ["g1"], how=badparts.HOW_USE)
    assert f"g1 was judged PULL by inspection r2: golden: {why}. The rule: " in str(e.value)


def _made_from(lib, sid, src) -> Path:
    """A metric scan made from another one (triangulated again): its meta names the source."""
    d = _entry(lib, "metric", sid)
    (d / "meta.json").write_text(json.dumps({"id": sid, "geometry_id": GEO, "source_scan": src}),
                                 encoding="utf-8")
    return d


def test_a_scan_made_from_a_bad_part_is_a_bad_part_too(eng):
    """A scan triangulated again (after a recalibration) is the same part: it has the
    findings of the scan it was made from, through any number of steps; and a scan made from
    another one that an inspection judged bad marks that one too, so its other copies are
    bad parts as well. A loop of links ends."""
    lib = eng.store.root
    _entry(lib, "metric", "raw")
    _judged_by_run(lib, "raw", "REJECT", run_id="r_raw")
    _made_from(lib, "tri", "raw")
    _made_from(lib, "tri2", "tri")
    for sid in ("tri", "tri2"):
        (hit,) = badparts.findings(eng.store, [sid])
        assert (hit["item"], hit["disposition"], hit["why"]) == (
            sid, "REJECT", "is made from scan raw, which was judged REJECT by inspection r_raw")
    with pytest.raises(badparts.BadPartReference, match="tri2 is made from scan raw"):
        badparts.require_good(eng.store, "recipe golden", ["tri2"], sweep=True)
    _made_from(lib, "loop_a", "loop_b")
    _made_from(lib, "loop_b", "loop_a")
    assert badparts.findings(eng.store, ["loop_a"], sweep=True) == []
    # a copy judged bad marks what it was made from: its other copies are bad parts too
    _entry(lib, "metric", "raw2")
    _made_from(lib, "copy_a", "raw2")
    _made_from(lib, "copy_b", "raw2")
    _judged_by_run(lib, "copy_a", "PULL", run_id="r_a")
    (mark,) = badparts._jsonl(lib / "metric" / GEO / "raw2" / badparts.STAMP)
    assert (mark["inspection_id"], mark["scan_id"]) == ("r_a", "copy_a")
    (hit,) = badparts.findings(eng.store, ["copy_b"])
    assert hit["why"] == ("is made from scan raw2, which was judged PULL by inspection r_a (as "
                          "scan copy_a, made from it)")
    # a run before the marks that judged the source: found for the copy
    _entry(lib, "metric", "raw3")
    _made_from(lib, "copy_c", "raw3")
    rid, _, _ = _job(lib / "runs", result="FAIL", max_dev=0.3, scan_id="raw3")
    (lib / "metric" / GEO / "raw3" / badparts.STAMP).unlink()
    assert badparts.findings(eng.store, ["copy_c"]) == []
    (hit,) = badparts.findings(eng.store, ["copy_c"], sweep=True)
    assert hit["why"] == f"is made from scan raw3, which was judged REJECT by inspection {rid}"


def test_a_scan_judged_before_the_marks_is_refused_where_it_is_used_searched_once(
        eng, tmp_path, monkeypatch):
    """A run made before the marks existed (or by an older build) judged a scan REJECT: a
    golden named by its id, a job's golden, a compare - every USE searches the runs folders
    for it, once: what is found is marked, a scan nothing was found for has the search noted
    in its entry and is not searched again; a runs folder that appears later is searched
    then (that one only)."""
    lib = eng.store.root
    for sid in ("old", "good"):
        _entry(lib, "metric", sid)
    rid, _, _ = _job(lib / "runs", result="FAIL", max_dev=0.3, scan_id="old")
    mark = lib / "metric" / GEO / "old" / badparts.STAMP
    mark.unlink()                                           # as a run before the marks
    searched = []
    real = badparts._run_findings
    monkeypatch.setattr(badparts, "_run_findings",
                        lambda library, ids, dirs: searched.append(list(dirs)) or
                        real(library, ids, dirs))
    with pytest.raises(badparts.BadPartReference,
                       match=f"old was judged REJECT by inspection {rid}"):
        badparts.require_good(eng.store, "the golden", ["old"], how=badparts.HOW_USE)
    assert mark.is_file()                                   # found: marked
    assert not (lib / "metric" / GEO / "old" / badparts.SWEPT).exists()
    badparts.require_good(eng.store, "the golden", ["good"])
    noted = json.loads((lib / "metric" / GEO / "good" / badparts.SWEPT).read_text(encoding="utf-8"))
    assert badparts._key(lib / "runs") in noted["runs_dirs"]
    n = len(searched)
    badparts.require_good(eng.store, "the golden", ["good"])
    assert len(searched) == n                               # searched once
    extra = tmp_path / "line_runs"                          # a station's runs folder, new
    _job(extra, result="INVALID", scan_id="good")
    (lib / "metric" / GEO / "good" / badparts.STAMP).unlink()   # its mark: as before the marks
    monkeypatch.setenv("SCANNER_RUNS_DIR", str(extra))
    with pytest.raises(badparts.BadPartReference, match="good was judged PULL by inspection"):
        badparts.require_good(eng.store, "the golden", ["good"])
    assert [badparts._key(d) for d in searched[-1]] == [badparts._key(extra)]


# ================================================================== the verbs: plain entries

def _scan(lib, sid, kind="objects") -> ObjectScan:
    return ObjectScan(id=sid, label=sid, geometry_id=GEO, path=_entry(lib, kind, sid),
                      created="", base_id="b0")


def test_the_memory_bank_takes_good_parts_and_is_never_checked_against_a_bad_one(eng):
    lib = eng.store.root
    bad = _scan(lib, "o_bad")
    _judged_by_run(lib, "o_bad", "REJECT")
    with pytest.raises(badparts.BadPartReference, match=RULE):
        bank_mod.add(eng, bad, "plates")
    assert not (lib / "banks").exists()                     # nothing was added
    # a member judged bad after it was added: the bank is not checked against
    good = _scan(lib, "o_good")
    (good.path / "relief.npz").write_bytes(b"")             # a member that exists
    d = lib / "banks" / GEO / "plates"
    d.mkdir(parents=True)
    (d / "members.jsonl").write_text(json.dumps({"op": "add", "scan_id": "o_good",
                                                 "path": f"objects/o_good"}) + "\n")
    _judged_by_check(lib, good.path, {"result": "FLAG", "bank": "other", "n_ref": 4},
                     "anomalies")
    with pytest.raises(badparts.BadPartReference, match="bank-remove"):
        bank_mod.check(eng, _scan(lib, "o_test"), "plates")


def test_an_alignment_reference_is_taught_on_a_good_part_only(eng):
    lib = eng.store.root
    p = SimpleNamespace(name="p1", geometry_id=GEO, mock=True)
    bad = _scan(lib, "o_bad")
    _judged_by_run(lib, "o_bad", "PULL")
    with pytest.raises(badparts.BadPartReference, match=RULE):
        virtual.teach_alignment(eng, p, "seat", bad)
    # taught on a part judged bad since: never checked against
    taught = _scan(lib, "o_taught")
    virtual._write_version(eng, GEO, "seat", {"kind": "alignment", "name": "seat",
                                              "geometry_id": GEO, "scan_id": taught.id,
                                              "scan_path": str(taught.path)})
    _judged_by_check(lib, taught.path, {"verdict": "FAIL"})
    with pytest.raises(badparts.BadPartReference, match="Teach 'seat' again on a good part"):
        virtual.check_alignment(eng, p, "seat", _scan(lib, "o_test"))


def _profile(fs_id, scans=()) -> FaceProfile:
    return FaceProfile(label=fs_id, geometry_id=GEO, calibration_id="c1", face_id="f1",
                       facescan_id=fs_id, metric_scan_ids=list(scans), face_diameter_mm=21.05,
                       grid_mm=0.25, stats={}, radial=[], centre_mm=[0, 0, 0], normal=[0, 0, 1],
                       files={"npz": None}, _height=np.zeros((3, 3)), _grid=np.zeros((3, 3)))


def test_the_golden_end_face_is_made_from_good_placements_only(eng, tmp_path):
    lib = eng.store.root
    for fs in ("fs_good", "fs_dented"):
        _entry(lib, "facescans", fs)
    _judged_by_check(lib, lib / "facescans" / GEO / "fs_dented",
                     {"verdict": "FAIL", "disposition": "REJECT"})   # check_face found a dent
    with pytest.raises(badparts.BadPartReference, match="fs_dented was judged REJECT by defects"):
        face_verbs.face_golden(eng, [_profile("fs_good"), _profile("fs_dented")])
    # a placement whose repeat scan an inspection pulled
    _entry(lib, "metric", "m_pulled")
    _judged_by_run(lib, "m_pulled", "PULL")
    with pytest.raises(badparts.BadPartReference, match="m_pulled was judged PULL"):
        face_verbs.face_golden(eng, [_profile("fs_good"), _profile("fs_x", ["m_pulled"])])
    # a profile file that lives in a bad-part output
    out = tmp_path / "bad" / "SN-1"
    out.mkdir(parents=True)
    (out.parent / badparts.MARKER).write_text("x")
    (out / "face.npz").write_bytes(b"")
    with pytest.raises(badparts.BadPartReference, match="lives in a bad-part output"):
        face_verbs.face_golden(eng, [_profile("fs_good"), str(out / "face.npz")])
    # checked against a golden one of whose placements was judged bad since: refused
    gd = _entry(lib, "face_golden", "g1")
    gh = FaceGolden(id="g1", label="g1", geometry_id=GEO, path=gd, created="",
                    face_diameter_mm=21.05, calibration_id="c1", n_placements=2, params={},
                    meta={"placements": [{"facescan_id": "fs_good", "metric_scans": []},
                                         {"facescan_id": "fs_dented", "metric_scans": []}]})
    with pytest.raises(badparts.BadPartReference, match="Make the reference again"):
        face_verbs.check_face(eng, _profile("fs_test"), gh)


def test_the_station_golden_is_made_from_good_measurements_only(eng):
    lib = eng.store.root
    _entry(lib, "metric", "m_bad")
    _judged_by_check(lib, lib / "metric" / GEO / "m_bad",
                     {"kind": "station_check", "verdict": "FAIL", "disposition": "REJECT"})
    m = StationMeasurement(label="m1", scan_id="m_bad", geometry_id=GEO, calibration_id="c1",
                           cells=[], pairs=[], ct=None, zones=[], flood_source="", params={})
    with pytest.raises(badparts.BadPartReference, match="station_golden measurement: m_bad"):
        station_verbs.station_golden(eng, [m])


def test_the_dent_null_is_calibrated_on_good_cells_only(eng, monkeypatch):
    lib = eng.store.root
    flagged = _entry(lib, "dentsets", "ds_flagged")
    _judged_by_check(lib, flagged, {"verdict": "FLAG", "n_detections": 1})
    ds = SimpleNamespace(id="ds_flagged", path=flagged, zone="z1", meta={})
    monkeypatch.setattr(dent_verbs, "as_dentset", lambda s, d: d)
    monkeypatch.setattr(dent_verbs, "_metric_for",
                        lambda s, d, m: ([], SimpleNamespace(id="c1"), GEO, None, []))
    with pytest.raises(badparts.BadPartReference, match="ds_flagged was judged REJECT"):
        dent_verbs.calibrate_dent_null(eng, [ds], 21.0)
    # a null made on a cell judged bad since: never detected against
    monkeypatch.setattr(dent_verbs.ds_mod, "load",
                        lambda path, subset="all": SimpleNamespace(pattern_signature=lambda: "s"))
    null = {"kind": "dent_null", "id": f"{GEO}/n1", "geometry_id": GEO,
            "sources": [{"dentset": "ds_flagged", "path": str(flagged), "metric_scans": []}]}
    with pytest.raises(badparts.BadPartReference, match="Calibrate the null again"):
        dent_verbs.detect_dents(eng, ds, null_calibration=null)


def test_a_relief_compare_never_takes_a_bad_part_as_its_golden(eng):
    lib = eng.store.root
    bad = _scan(lib, "o_bad")
    _judged_by_run(lib, "o_bad", "REJECT")
    with pytest.raises(badparts.BadPartReference, match="compare: the golden: o_bad"):
        api.compare(_scan(lib, "o_test"), bad)


# ================================================================== real runs

@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    """A mock library with the bench's TRUE stereo calibration, the golden of the long tube
    and synthetic scans of a good and a dented one (recipe_synth: nothing is captured)."""
    root = tmp_path_factory.mktemp("badparts") / "library"
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
def real(lib):
    root, scans = lib
    s = api.configure(root, mock=True)
    yield s, scans
    api._session = None


def _reject_dent(s, scans, runs) -> dict:
    out = api.recipe_run("tube_long_end_face", scan=scans["dent"].id, project_feedback=False,
                         runs_dir=runs)
    assert out["exit_code"] == 1 and out["result"]["disposition"] == "REJECT"
    return out


def test_a_real_rejected_part_is_copied_with_its_picture_profiles_and_3d_file(
        real, bad_part_output, tmp_path, eventlog_on):
    """A recipe run of a dented tube (offline: the stored scan), the computer's default output
    set and the event log on: both get the inspection, and the dented scan is marked."""
    s, scans = real
    root = bad_part_output
    out = _reject_dent(s, scans, tmp_path / "runs")
    rid, run_dir = out["run_id"], Path(out["run_dir"])
    assert out["result"]["warnings"] == [w for w in out["result"]["warnings"]
                                         if not w.startswith("bad-part output")]
    assert _folders(root) == [rid]                          # no code read: the inspection id
    assert sorted(p.name for p in (root / rid).iterdir()) == [
        "overlay.png", "profile_x.csv", "profile_y.csv", "result.json", "scan.ply"]
    (row,) = _index(root)
    assert (row["inspection_id"], row["program"], row["operating_mode"], row["disposition"]) == (
        rid, "tube_long_end_face", "recipe_run", "REJECT")
    assert row["out_of_limit"].startswith("end_face/defect: max_depth_mm=")
    assert row["out_of_limit"].endswith("mm [..0.15]")
    ev = [r for r in eventlog.read_rows(s.store.root / "events" / f"inspections_v1_"
                                        f"{eventlog.month_of(row['time'])}.csv")
          if r["inspection_id"] == rid]
    assert ev and ev[0]["judgment"] == "FAIL"               # the event log has it too
    ply = s.store.root / ev[0]["data_3d_file"]
    assert (root / rid / "scan.ply").read_bytes() == ply.read_bytes()
    marks = badparts._jsonl(scans["dent"].path / badparts.STAMP)
    assert rid in [m["inspection_id"] for m in marks]
    shutil.rmtree(s.store.root / "events", ignore_errors=True)


def test_the_run_commands_take_the_option_and_it_wins(real, bad_part_output, tmp_path):
    s, scans = real
    root = s.store.root
    runs = tmp_path / "runs"
    mine = tmp_path / "mine"
    base = ["--mock", "--library", str(root), "recipe-run", "tube_long_end_face",
            "--scan", scans["dent"].id, "--no-feedback", "--runs-dir", str(runs)]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert cli.main([*base, "--bad-part-output", str(mine)]) == 1
    api._session = None
    assert len(_folders(mine)) == 1 and not bad_part_output.exists()
    assert badparts._COMMAND_LINE is None                   # for this command only
    with contextlib.redirect_stdout(io.StringIO()):
        assert cli.main([*base, "--bad-part-output", "off"]) == 1
    api._session = None
    assert len(_folders(mine)) == 1 and not bad_part_output.exists()
    with contextlib.redirect_stdout(io.StringIO()):         # not given: the computer's default
        assert cli.main(base) == 1
    api._session = None
    assert len(_folders(bad_part_output)) == 1
    for name in ("run", "recipe-run", "quick-check"):
        assert "--bad-part-output" in cli.build_parser()._subparsers._group_actions[0] \
            .choices[name].format_help()


def test_a_scan_judged_bad_is_never_a_golden_and_never_compared_with(real, tmp_path):
    s, scans = real
    p = api.load_preset(PRESET)
    out = _reject_dent(s, scans, tmp_path / "runs")
    dent, ok = scans["dent"], scans["ok"]
    recipe = api.recipe_load("tube_long_alignment")
    with pytest.raises(badparts.BadPartReference, match=RULE) as e:
        runner.record_golden(s, recipe, p, dent)
    assert f"{dent.id} was judged REJECT by inspection " in str(e.value)
    (mine,) = [f["why"] for f in badparts.findings(s.store, [dent])
               if f"by inspection {out['run_id']}: " in f["why"]]
    assert f"{out['run_id']}: end_face/defect: " in mine            # with the inspection's reason
    with pytest.raises(badparts.BadPartReference, match="the golden of recipe"):
        runner.find_golden(s, recipe, dent.geometry_id, dent.id)
    assert runner.find_golden(s, recipe, ok.geometry_id, ok.id).id == ok.id     # a good one is
    with pytest.raises(badparts.BadPartReference, match="compare_metric: the golden"):
        api.compare_metric(ok, dent)
    m = api.measure_run("deviation", p, golden=dent.id, scans=[ok.id], runs_dir=tmp_path / "m")
    doc = m["result"]
    assert (doc["status"], doc["reason"], m["exit_code"]) == ("NOT_MEASURABLE", "golden", 5)
    assert RULE in doc["message"]
    # a station's golden compare (here a quick check): refused, the part PULLed
    g = StationGolden(id="sg1", label="sg1", geometry_id=ok.geometry_id, path=tmp_path,
                      created="", calibration_id=ok.calibration_id, n_measurements=1,
                      limits={}, nominal={}, files={"json": tmp_path / "golden.json"})
    (tmp_path / "golden.json").write_text(json.dumps(
        {"measurements": [{"label": "m", "scan_id": dent.id, "file": ""}]}))
    with pytest.raises(badparts.BadPartReference, match=f"its measurement: {dent.id}"):
        station_verbs.station_check(s, ok, golden=g)


def test_a_station_golden_compare_with_a_bad_part_pulls_the_part(real, tmp_path, monkeypatch):
    s, scans = real
    _reject_dent(s, scans, tmp_path / "runs0")
    monkeypatch.setattr(api, "scan_fast", lambda zone, p, label, steps, periods, average, **kw:
                        scans["ok"])
    monkeypatch.setattr(cli, "_mock_place_metric", lambda ctx: None)
    runs = tmp_path / "runs"
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        code = cli.main(["--mock", "--library", str(s.store.root), "quick-check", PRESET,
                         "--check", "golden", "--golden", scans["dent"].id, "--threshold",
                         "0.5", "--runs-dir", str(runs)])
    api._session = None
    assert code == 2                                        # refused: it says why
    (run_dir,) = [p for p in runs.iterdir() if p.is_dir()]
    doc = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    assert doc["disposition"] == "PULL" and RULE in doc["invalid_reason"]


def _recipe_file(tmp_path, name: str, bad_part_output: str) -> Path:
    """A copy of a built-in recipe (same name) whose outputs name a bad-part output."""
    doc = json.loads((conftest.ROOT / "recipes" / f"{name}.json").read_text(encoding="utf-8"))
    doc["outputs"]["bad_part_output"] = bad_part_output
    f = tmp_path / f"{name}.json"
    f.write_text(json.dumps(doc), encoding="utf-8")
    return f


def test_a_recipe_naming_a_folder_the_os_cannot_resolve_judges_as_ever(real, tmp_path,
                                                                      monkeypatch):
    """Real recipe runs whose recipe names a folder whose path the OS cannot resolve: the
    good part is CONTINUE (exit 0), the dented one REJECT (exit 1), both write their
    result.json naming the folder as text, and the dented one is copied."""
    s, scans = real
    f = _recipe_file(tmp_path, "tube_long_end_face", "share/bad")
    _share(monkeypatch, tmp_path / "share")
    runs = tmp_path / "runs"
    outs = [api.recipe_run(str(f), scan=scans[v].id, project_feedback=False, runs_dir=runs)
            for v in ("ok", "dent")]
    assert [o["exit_code"] for o in outs] == [0, 1]
    for o in outs:
        doc = json.loads((Path(o["run_dir"]) / "result.json").read_text(encoding="utf-8"))
        assert doc["outputs"]["bad_part_output"] == os.path.abspath(tmp_path / "share" / "bad")
    assert _folders(tmp_path / "share" / "bad") == [outs[1]["run_id"]]


def test_a_cycle_takes_its_recipes_bad_part_output_when_the_station_names_none(
        real, tmp_path, monkeypatch):
    """A cycle of the example line on a dented tube (kicked out at the end face), its folder
    path one the OS cannot resolve: the end face recipe's outputs.bad_part_output when the
    station names none; when two of its recipes name different ones, the one of the work
    state that speaks for the cycle, and a warning; the station's own wins over both."""
    s, scans = real
    _share(monkeypatch, tmp_path / "share")
    cfg = json.loads((conftest.ROOT / "stations" / "example_tube_line.json").read_text(
        encoding="utf-8"))
    cfg["flow"] = str(conftest.ROOT / "flows" / "example_tube_line.json")
    cfg["parameters"].update(mock_library=None, runs_dir=str(tmp_path / "runs"))
    recipes = cfg["profiles"]["long_tube"]["recipes"]
    recipes["end_face"] = str(_recipe_file(tmp_path, "tube_long_end_face", "share/end"))
    station = tmp_path / "line.json"

    def cycle() -> dict:
        station.write_text(json.dumps(cfg), encoding="utf-8")
        out = api.flow_run(station, "long_tube", inputs={"part_present": True},
                           runs_dir=tmp_path / "runs", acknowledge=True, project_faults=False,
                           scans={k: scans["dent"].id for k in ("loaded", "aligned", "end_face")})
        assert (out["exit_code"], out["result"]["disposition"]) == (1, "REJECT")
        return out
    out = cycle()
    end = tmp_path / "share" / "end"
    assert out["result"]["outputs"]["bad_part_output"] == os.path.abspath(end)
    assert _folders(end) == [out["cycle_id"]]
    assert not [w for w in out["result"]["warnings"] if w.startswith("bad-part output")]
    (row,) = _index(end)
    assert (row["operating_mode"], row["disposition"]) == ("station", "REJECT")
    recipes["alignment"] = str(_recipe_file(tmp_path, "tube_long_alignment", "share/align"))
    out = cycle()
    assert out["result"]["outputs"]["bad_part_output"] == os.path.abspath(end)
    (warn,) = [w for w in out["result"]["warnings"] if w.startswith("bad-part output")]
    assert "name different bad-part outputs" in warn and "the one of work state end_face" in warn
    assert len(_folders(end)) == 2 and not (tmp_path / "share" / "align").exists()
    cfg["parameters"]["bad_part_output"] = "station_bad"   # relative to the configuration
    out = cycle()
    assert out["result"]["outputs"]["bad_part_output"] == os.path.abspath(tmp_path / "station_bad")
    assert _folders(tmp_path / "station_bad") == [out["cycle_id"]] and len(_folders(end)) == 2


def test_a_bad_part_output_that_holds_the_library_is_refused_and_judges_nothing(real, tmp_path):
    s, scans = real
    above = s.store.root.parent
    badparts.set_command_line(str(above))
    try:
        out = _reject_dent(s, scans, tmp_path / "runs")
    finally:
        badparts.set_command_line(None)
    (warn,) = [w for w in out["result"]["warnings"] if w.startswith("bad-part output")]
    assert "NOT copied" in warn and "the folder holds the library" in warn
    assert not (above / badparts.MARKER).exists()
    good = api.recipe_run("tube_long_alignment", scan=scans["ok"].id, project_feedback=False,
                          runs_dir=tmp_path / "runs")
    assert (good["exit_code"], good["result"]["disposition"]) == (0, "CONTINUE")


def test_a_golden_whose_run_came_before_the_marks_is_refused_where_it_is_named(real, tmp_path):
    """A dented tube judged REJECT by a run whose mark is not there (a run before this build,
    or by an older one): named by its id as a recipe's golden, in compare_metric and in
    Measure's deviation, it is refused each time; a good scan named by id is searched once."""
    s, scans = real
    p = api.load_preset(PRESET)
    recipe = api.recipe_load("tube_long_alignment")
    old = RS.oracle_scan(s, p, "scene:tube_long:dent", "t_long_dent_old", seed=41)
    out = api.recipe_run("tube_long_end_face", scan=old.id, project_feedback=False,
                         runs_dir=s.store.root / "runs")
    assert out["result"]["disposition"] == "REJECT"
    mark = old.path / badparts.STAMP
    mark.unlink()
    with pytest.raises(badparts.BadPartReference,
                       match=f"{old.id} was judged REJECT by inspection {out['run_id']}"):
        runner.find_golden(s, recipe, old.geometry_id, old.id)
    assert mark.is_file()                                   # what the search found is marked
    mark.unlink()
    with pytest.raises(badparts.BadPartReference, match="compare_metric: the golden"):
        api.compare_metric(scans["ok"], old)
    mark.unlink()
    m = api.measure_run("deviation", p, golden=old.id, scans=[scans["ok"].id],
                        runs_dir=tmp_path / "m")
    assert (m["result"]["status"], m["result"]["reason"]) == ("NOT_MEASURABLE", "golden")
    ok = scans["ok"]
    assert runner.find_golden(s, recipe, ok.geometry_id, ok.id).id == ok.id
    assert (ok.path / badparts.SWEPT).is_file()


def test_a_bad_part_triangulated_again_is_still_a_bad_part(real, tmp_path):
    s, scans = real
    p = api.load_preset(PRESET)
    _reject_dent(s, scans, tmp_path / "runs")
    new = api.triangulate(scans["dent"])["scan"]
    assert new.meta["source_scan"] == scans["dent"].id
    with pytest.raises(badparts.BadPartReference,
                       match=f"is made from scan {scans['dent'].id}, which was judged REJECT"):
        runner.record_golden(s, api.recipe_load("tube_long_alignment"), p, new)
    with pytest.raises(badparts.BadPartReference, match="compare_metric: the golden"):
        api.compare_metric(scans["ok"], new)


def test_a_golden_judged_bad_after_it_was_made_is_refused_at_the_check(real, tmp_path):
    """A good part recorded as the recipe's golden; later the line counts a run on that very
    part ABORTED after its verdict (a PULL): the recipe's checks against it are refused - the
    run is a PULL, and says why."""
    s, scans = real
    p = api.load_preset(PRESET)
    recipe = api.recipe_load("tube_long_alignment")
    g2 = RS.oracle_scan(s, p, "scene:tube_long", "tube_long_alignment_golden", seed=21)
    runner.record_golden(s, recipe, p, g2)                  # the newest golden of that label
    runs = tmp_path / "runs"
    before = api.recipe_run(recipe, scan=scans["ok"].id, project_feedback=False, runs_dir=runs)
    assert before["result"]["disposition"] == "CONTINUE"
    on_golden = api.recipe_run("tube_long_end_face", scan=g2.id, project_feedback=False,
                               runs_dir=runs)
    assert on_golden["result"]["disposition"] == "CONTINUE"
    R.append_event(Path(on_golden["run_dir"]) / "run.log.jsonl", on_golden["run_id"],
                   eventlog.ABORTED_EVENT, message="ABORT after the verdict")
    after = api.recipe_run(recipe, scan=scans["ok"].id, project_feedback=False, runs_dir=runs)
    res = after["result"]
    assert (res["disposition"], after["exit_code"]) == ("PULL", 5)
    (why,) = [w for w in res["warnings"] if w.startswith("golden: ")]
    assert RULE in why and on_golden["run_id"] in why


# ================================================================== the test harness guard

def test_the_guard_refuses_a_bad_part_output_outside_a_temporary_folder(tmp_path, monkeypatch):
    out = tmp_path / "bad"
    _job(tmp_path / "runs", root=out, result="INVALID", codes=["SN-1"])
    assert conftest.take_isolation_notes() == []            # in a temporary folder: fine
    monkeypatch.setattr(conftest._Guard, "temp",
                        {f + os.sep for f in conftest._forms(tmp_path / "temp")})
    for attempt in (lambda: open(out / "SN-2" / "result.json", "w"),
                    lambda: open(tmp_path / "x" / badparts.MARKER, "w"),
                    lambda: open(tmp_path / "x" / badparts.INDEX, "ab"),
                    lambda: open(tmp_path / "lib" / badparts.QUEUE_DIR / "r.json", "w"),
                    lambda: open(tmp_path / "run" / badparts.DONE_FILE, "x"),
                    lambda: open(tmp_path / "scan" / badparts.STAMP, "a")):
        with pytest.raises(conftest.IsolationError, match="bad-part output"):
            attempt()
    assert len(conftest.take_isolation_notes()) == 6
    with open(out / "SN-1" / "result.json", encoding="utf-8") as f:    # reading: never refused
        assert f.read()
    assert os.environ[BAD_PART_ENV] == "off"                # the computer's default: never used

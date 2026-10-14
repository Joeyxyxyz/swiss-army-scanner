"""Library, run and output paths longer than the OS takes (engine/library/longpath.py).

The first real-rig day: a mock Quick check and a mock Measure run on a library deep in the
file tree crashed half way with "unexpected FileNotFoundError: [WinError 206]" (exit 4).
Now the session refuses a library folder that is too long or has no room for its session
log, a run folder with no room for its files is refused before anything is measured (and
not left behind), the store refuses an entry or output path it cannot make, and any other
too-long path the command line meets is a refusal (exit 2) that names the path, its length
and the limit.

What the OS really does decides, never a length alone: each deep-path test first asks the
OS itself (plain os.makedirs / open, not the code under test) whether it takes such a path,
and where it does (Windows with long paths on, another system) it expects the same command
to run instead of being refused. Deep paths are made under tmp_path only."""

import errno
import json
import os
import shutil
from pathlib import Path

import pytest

from engine import api, rules
from engine import runs as R
from engine.library import longpath
from engine.library.store import SESSION_DIR_CHARS, SESSION_LOG_CHARS, LibraryStore, timestamp
from ui import cli

FIXTURE_PRESETS = Path(__file__).resolve().parent / "fixtures" / "presets"


@pytest.fixture(autouse=True)
def _fresh_engine():
    yield
    api._session = None


def _deep(base, n: int) -> Path:
    """A folder path exactly n characters long (absolute) under base, in names of at most
    100 characters (made by nobody: only its text)."""
    p = os.path.abspath(base)
    assert len(p) + 2 <= n, f"tmp_path is already {len(p)} characters long"
    while len(p) < n:
        r = n - len(p)
        k = min(100, r - 1)
        if r - 1 - k == 1:                  # never leave one character (a separator alone)
            k -= 1
        p = os.path.join(p, "d" * k)
    assert len(p) == n
    return Path(p)


def _os_takes(base, n: int) -> bool:
    """The OS's own answer: can a file whose whole path is n characters long be made under
    base (its folders made as deep as needed)? Everything is removed again."""
    probe = Path(base) / "os_probe"
    folder = _deep(probe, n - 21)               # room for a 20-character file name
    try:
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "f" * (n - len(str(folder)) - 1)), "x"):
            pass
        return True
    except OSError:
        return False
    finally:
        shutil.rmtree(("\\\\?\\" if os.name == "nt" else "") + str(probe.resolve()),
                      ignore_errors=True)


def _os_makes_folder(base, n: int) -> bool:
    """The OS's own answer: can a folder whose path is n characters long be made?"""
    probe = Path(base) / "os_probe_dir"
    try:
        os.makedirs(_deep(probe, n))
        return True
    except OSError:
        return False
    finally:
        shutil.rmtree(("\\\\?\\" if os.name == "nt" else "") + str(probe.resolve()),
                      ignore_errors=True)


def _session_fits(base, n: int) -> bool:
    """The OS's own answer for a library folder n characters long: can its session log be
    written, the new folder sessions/<time> and log.jsonl in it?"""
    return (_os_makes_folder(base, n + 1 + SESSION_DIR_CHARS)
            and _os_takes(base, n + 1 + SESSION_LOG_CHARS))


def _cli(capsys, *argv) -> tuple:
    code = cli.main(["--mock", *argv])
    api._session = None
    out = capsys.readouterr()
    return code, out.out, out.err


QUICK = ("quick-check", "--check", "flatness", "--threshold", "2.0", "--roi", "0,340,160,500",
         "--operator-confirmed")
MEASURE = ("measure-run", "--what", "flatness", "--repeats", "1", "--label", "lp")


# ================================================================== what the OS really says

def test_only_an_error_that_happened_is_judged_too_long(monkeypatch, tmp_path):
    """too_long() reads the OS error: ENAMETOOLONG always; on Windows WinError 206, and "not
    found" for a path of 260 characters or more only while long paths are off. A short path
    not found, or a long one with long paths on, is the error it was."""
    long_name = str(_deep(tmp_path, 300) / "x.json")
    assert longpath.too_long(OSError(errno.ENAMETOOLONG, "File name too long", "x"))
    assert not longpath.too_long(FileNotFoundError(errno.ENOENT, "not found", str(tmp_path / "x")))
    assert not longpath.too_long(PermissionError(errno.EACCES, "denied", long_name))
    assert not longpath.too_long(ValueError("not an OS error"))
    assert longpath.too_long(rules.PathTooLong("said already"))
    if os.name == "nt":
        assert longpath.too_long(OSError(errno.ENOENT, "The filename or extension is too long",
                                         str(tmp_path), 206))
        monkeypatch.setattr(longpath, "long_paths", lambda: False)
        assert longpath.too_long(FileNotFoundError(errno.ENOENT, "not found", long_name))
        assert longpath.too_long(FileNotFoundError(errno.ENOENT, "not found"), long_name)
        monkeypatch.setattr(longpath, "long_paths", lambda: True)    # a real missing folder
        assert not longpath.too_long(FileNotFoundError(errno.ENOENT, "not found", long_name))
    else:
        assert not longpath.too_long(FileNotFoundError(errno.ENOENT, "not found", long_name))


def test_the_refusal_names_the_path_its_length_and_the_limit(monkeypatch, tmp_path):
    p = _deep(tmp_path, 280)
    e = longpath.refusal("the run folder", p, fix=longpath.FIX_RUNS, done=R.NOT_MEASURED)
    msg = str(e)
    assert isinstance(e, rules.RuleViolation) and isinstance(e, FileNotFoundError)
    assert msg.startswith("the run folder cannot be written: its path is 280 characters long")
    assert str(p) in msg and msg.endswith("Nothing was measured.") and "[Errno" not in msg
    assert "runs_dir" in msg and "long paths" in msg and e.chars == 280 and e.errno == errno.ENOENT
    r = str(longpath.refusal("the library folder", p, room=41, holds="the session log"))
    assert "has no room for the session log" in r and "reach 322 characters" in r
    assert "folder among them" not in r
    r = str(longpath.refusal("the library folder", p, room=41, new_folder=31, holds="the log"))
    assert "reach 322 characters (a folder among them 312);" in r
    if os.name == "nt":
        monkeypatch.setattr(longpath, "long_paths", lambda: False)
        assert "at most 259 characters for a file's path and 247 for a folder's, and long " \
               "paths are off for this program on this PC" in str(longpath.refusal("x", p))
        monkeypatch.setattr(longpath, "long_paths", lambda: None)   # Windows can't be asked
        assert "247 for a folder's unless long paths are on." in str(longpath.refusal("x", p))
    monkeypatch.setattr(longpath, "long_paths", lambda: True)
    said = str(longpath.refusal("x", p, error=OSError(errno.ENAMETOOLONG, "too long", "y")))
    assert "the file system did not take a path that long" in said and "259" not in said


def test_the_command_line_turns_any_too_long_os_error_into_a_refusal(tmp_path):
    """A too-long OS error no guard caught is exit 2 (PathTooLong, the length named), never
    "unexpected" (exit 1, or 4 for a run); any other FileNotFoundError stays unexpected."""
    p = str(_deep(tmp_path, 270))
    err = OSError(errno.ENAMETOOLONG, "File name too long", p)
    if os.name == "nt":
        err = OSError(errno.ENOENT, "The filename or extension is too long", p, 206)
    code, line = cli._classify_error(err, "scan")
    assert code == cli.EXIT_REFUSED and line.startswith("PathTooLong: a file or folder cannot "
                                                        "be written: its path is 270 characters")
    assert "unexpected" not in line and "long paths" in line
    code, line = cli._classify_error(longpath.refusal("the run folder", p), "scan")
    assert code == cli.EXIT_REFUSED and line.startswith("PathTooLong: the run folder")
    code, line = cli._classify_error(FileNotFoundError(errno.ENOENT, "No such file", "x.json"),
                                     "scan")
    assert code == cli.EXIT_NOT_OK and line.startswith("unexpected FileNotFoundError")


def test_a_run_folder_has_room_for_every_file_a_run_writes():
    """RUN_FILE_ROOM is the longest file name a run folder holds: a new name longer than it
    must raise it (new_run checks that room before anything is measured)."""
    from engine import badparts, eventlog
    from engine.modes import measure, quick
    from engine.recipes import runner
    names = ["result.json", "run.log.jsonl", "deviation.png", badparts.DONE_FILE,
             runner.FEEDBACK_IMAGE, runner.FEEDBACK_PHOTO, runner.OVERLAY_IMAGE,
             quick.CONFIG_FILE, measure.LOG_FILE, measure.SPEC_FILE, measure.VALUES_FILE,
             measure.OVERLAY_FILE, measure.PLY_FILE, eventlog.PROFILE_X, eventlog.PROFILE_Y]
    assert max(len(n) for n in names) == R.RUN_FILE_ROOM == len(badparts.DONE_FILE)
    # a work state's run folder in a cycle fits when its files do (steps_room counts files)
    assert 1 + R.RUN_FILE_ROOM > longpath.FILE_MAX - longpath.DIR_MAX
    # and the session log (sessions/<timestamp>/log.jsonl, in a new folder) is what the
    # library folder needs
    assert SESSION_LOG_CHARS == len(os.path.join("sessions", timestamp(), "log.jsonl"))
    assert SESSION_DIR_CHARS == len(os.path.join("sessions", timestamp()))


def test_a_new_folder_keeps_room_only_where_the_os_does(monkeypatch):
    """folder_slack: Windows without long paths keeps 12 characters free in a new folder
    (247 against 259), so a probe for a file in a new folder is that much longer; where long
    paths work, or on another system, a folder has the file's limit and nothing is added."""
    monkeypatch.setattr(longpath, "long_paths", lambda: False)
    assert longpath.folder_slack() == (12 if os.name == "nt" else 0)
    monkeypatch.setattr(longpath, "long_paths", lambda: None)        # Windows can't be asked
    assert longpath.folder_slack() == (12 if os.name == "nt" else 0)
    monkeypatch.setattr(longpath, "long_paths", lambda: True)
    assert longpath.folder_slack() == 0


def test_the_room_probe_leaves_nothing_behind(tmp_path):
    lib = tmp_path / "library"
    api.configure(lib, mock=True)
    assert sorted(p.name for p in lib.iterdir()) == ["README.md"]
    run_id, run_dir = R.new_run(tmp_path / "runs", "probe")
    assert list(run_dir.iterdir()) == [] and len(run_id) == R.run_id_chars("probe")
    out = LibraryStore(lib).output_path("previews", "x", ".png")
    assert not out.exists() and list(out.parent.iterdir()) == []


# ================================================================== deep paths, for real

def test_a_library_folder_too_long_is_refused_when_the_session_opens(tmp_path, capsys):
    """A 300-character library: refused at once (api.configure, exit 2 from any command),
    nothing half made - where the OS takes such a path, used as it is."""
    lib = _deep(tmp_path / "lib", 300)
    if _os_takes(tmp_path, 300 + 1 + SESSION_LOG_CHARS):
        api.configure(lib, mock=True)               # long paths work here: never refused
        assert (lib / "README.md").is_file()
        return
    with pytest.raises(rules.PathTooLong) as ei:
        api.configure(lib, mock=True)
    msg = str(ei.value)
    assert msg.startswith("the library folder cannot be written: its path is 300 characters")
    assert "--library" in msg and "long paths" in msg
    code, out, err = _cli(capsys, "--library", str(lib), "--json", *QUICK)
    doc = json.loads(out)
    assert code == cli.EXIT_REFUSED and doc["error"]["type"] == "PathTooLong"
    assert doc["disposition"] == "PULL" and doc["pull_reasons"][0]["reason"] == "refused"
    assert "300 characters" in err and "unexpected" not in err and "Traceback" not in err


@pytest.mark.parametrize("n", [215, 216, 217, 230])
def test_a_library_folder_with_no_room_for_its_session_log_is_refused(tmp_path, capsys, n):
    """The library folder itself can be made, but not the session log every command writes
    (the new folder sessions/<time>, then log.jsonl in it): refused when the session opens,
    not after the work. 216 / 217 characters: the file would fit (258 / 259) but its folder
    does not (248 / 249 against 247 without long paths); a logged verb did its work and was
    refused afterwards. Where the OS takes the session log (215 here; any length with long
    paths on), the session opens and a logged verb writes it."""
    lib = _deep(tmp_path / "lib", n)
    if _session_fits(tmp_path, n):
        api.configure(lib, mock=True, presets_dir=FIXTURE_PRESETS)
        api.load_preset("cell_wall_2p7ft_vstripe")             # a logged verb
        assert [p.name for p in lib.glob("sessions/*/log.jsonl")] == ["log.jsonl"]
        return
    with pytest.raises(rules.PathTooLong, match="no room for the session log every command "
                                                f"writes .*its path is {n} characters long") as ei:
        api.configure(lib, mock=True, presets_dir=FIXTURE_PRESETS)
    assert f"reach {n + 1 + SESSION_LOG_CHARS} characters (a folder among them " \
           f"{n + 1 + SESSION_DIR_CHARS})" in str(ei.value)
    assert not (lib / "sessions").exists()
    assert not [p for p in lib.iterdir() if p.name.startswith(longpath.PROBE)]
    code, _, err = _cli(capsys, "--library", str(lib), *MEASURE)
    assert code == cli.EXIT_REFUSED and "PathTooLong: the library folder has no room" in err
    assert not (lib / "sessions").exists() and not (lib / "runs").exists()  # nothing done


@pytest.mark.parametrize("argv, run_name", [(QUICK, "quick_check"), (MEASURE, "measure_flatness")])
@pytest.mark.parametrize("folder_len", [243, 253])
def test_quick_check_and_measure_on_a_deep_library_refuse_before_measuring(
        tmp_path, capsys, argv, run_name, folder_len):
    """The two crashes of the first real-rig day, on the mock: a library deep enough that
    its run folder (243 / 253 characters) cannot be made, or has no room for the files a run
    writes, while the session opens (its log fits) -> exit 2 "PathTooLong: the run folder
    ..." with nothing measured and no run folder left behind, never exit 4. Where the OS
    takes it the run happens (uncalibrated here: PULL / NOT MEASURABLE, exit 5)."""
    n_lib = folder_len - len("/runs/") - R.run_id_chars(run_name)
    lib = _deep(tmp_path / "lib", n_lib)
    need = folder_len + 1 + R.RUN_FILE_ROOM
    assert n_lib + 1 + SESSION_LOG_CHARS <= longpath.FILE_MAX       # the session log fits,
    assert n_lib + 1 + SESSION_DIR_CHARS <= longpath.DIR_MAX        # its folder too
    code, out, err = _cli(capsys, "--library", str(lib), *argv)
    runs = lib / "runs"
    if _os_takes(tmp_path, need):
        assert code == 5 and "PathTooLong" not in err
        assert [p.name for p in runs.iterdir()][0].split("_", 1)[1].startswith(run_name)
        return
    assert code == cli.EXIT_REFUSED, (out, err)
    if _os_makes_folder(tmp_path, folder_len):
        assert "PathTooLong: the run folder has no room for the files a run writes" in err
        assert f"its path is {folder_len} characters long" in err
        assert f"reach {need} characters" in err
    else:
        assert f"PathTooLong: the run folder cannot be written: its path is {folder_len} " \
               "characters long" in err
    assert "Nothing was measured." in err and "runs_dir" in err
    assert "unexpected" not in err and "Traceback" not in err
    assert not runs.exists() or list(runs.iterdir()) == []     # nothing left behind
    assert (lib / "sessions").is_dir()                          # the session itself was fine


def test_a_cycle_folder_needs_room_for_its_work_states_run_folders(tmp_path):
    """A flow's cycle folder holds steps/<run id>/<files>: new_run(room=steps_room(...))
    refuses a runs folder where a plain run fits but a work state's run would not."""
    names = ["st.inspect_left"]
    room = R.steps_room("steps", names)
    assert room == len("steps/") + R.run_id_chars(names[0]) + 1 + R.RUN_FILE_ROOM
    assert R.steps_room("steps", []) == R.RUN_FILE_ROOM
    runs = _deep(tmp_path / "runs", 190)
    plain = 190 + 1 + R.run_id_chars("st") + 1 + R.RUN_FILE_ROOM
    deep = 190 + 1 + R.run_id_chars("st") + 1 + room
    if not _os_takes(tmp_path, plain):
        pytest.skip("this OS does not even take a plain run folder at 190 characters")
    run_id, run_dir = R.new_run(runs, "st")                   # a plain run: room enough
    assert run_dir.is_dir()
    if _os_takes(tmp_path, deep):
        assert R.new_run(runs, "st", room=room)[1].is_dir()
        return
    with pytest.raises(rules.PathTooLong, match="no room for the files and the work states' "
                                                f"run folders it holds.*reach {deep} char"):
        R.new_run(runs, "st", room=room)
    assert [p.name for p in runs.iterdir()] == [run_id]      # the refused one is not left


def test_a_bad_part_output_too_long_says_so_and_never_changes_the_judgment(tmp_path):
    """The bad-part output (engine/badparts.py) never stops a run: a folder too long for the
    OS is said in the run's warnings with its length and the limit; the REJECT stands."""
    from engine import disposition as D
    root = _deep(tmp_path / "bad", 280)
    runs = tmp_path / "runs"
    run_id, run_dir = R.new_run(runs, "plate_check")
    res = R.empty_result(run_id=run_id, run_folder=str(run_dir), mock=True,
                         operator_confirmed=False, station="plate_check",
                         job={"type": "metric", "base": None, "zone": None, "config": None},
                         part={"role": "test", "label": "plate"}, threshold_mm=0.2)
    res["outputs"]["bad_part_output"] = str(root)
    res.update(result="FAIL", engine_verdict="FAIL", invalid_reason=None)
    res["deviation_mm"].update(max=0.3, method="plane_fit")
    D.settle(res)
    R.write_result(run_dir, res)
    on_disk = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    assert res["disposition"] == on_disk["disposition"] == "REJECT" and D.exit_code(res) == 1
    if _os_takes(tmp_path, 280 + 40):
        assert not [w for w in res["warnings"] if "NOT copied" in w]
        return
    w = [x for x in res["warnings"] if "NOT copied" in x]
    assert len(w) == 1 and "characters long" in w[0] and "shorter path" in w[0], res["warnings"]
    assert "the judgment is unchanged" in w[0]
    if os.name == "nt":
        assert "at most 259 characters" in w[0]


def test_the_store_refuses_an_entry_or_output_path_it_cannot_make(tmp_path):
    """A 215-character library opens (its session log and its folder just fit), but a
    metric scan entry or an output-bin file of a long label does not: PathTooLong from
    new_dir / output_path, nothing made - where the OS takes them, made."""
    lib = _deep(tmp_path / "lib", 215)
    if not _session_fits(tmp_path, 215):
        pytest.skip("this OS does not take the session log at 215 characters")
    s = api.configure(lib, mock=True).store
    label = "a_rather_long_label_for_a_face_scan"
    entry_len = len(str(s.root / "metric" / "geo_v1")) + 1 + 22 + 1 + len(label)
    if _os_takes(tmp_path, entry_len + 1 + len("cloud.npz")):
        d = s.new_dir("metric", "geo_v1", label=label)
        assert d.is_dir() and s.output_path("pointclouds", label, ".ply").parent.is_dir()
        return
    with pytest.raises(rules.PathTooLong, match="a new library entry under metric/geo_v1 "
                                                "cannot be written"):
        s.new_dir("metric", "geo_v1", label=label)
    assert not (s.root / "metric" / "geo_v1").exists() or \
        list((s.root / "metric" / "geo_v1").iterdir()) == []
    with pytest.raises(rules.PathTooLong, match="the output bin folder output/pointclouds has "
                                                "no room for the new file"):
        s.output_path("pointclouds", label, ".ply")
    assert list((s.root / "output" / "pointclouds").iterdir()) == []

import os
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ------------------------------------------------- no rig, no machine files
# Three environment variables, set before anything of the engine is imported and
# inherited by every child process a test starts. Engine code reads them when it
# acts (not once at import): the fixtures below change them for single tests.
#
# SAS_NO_HARDWARE=1
#     This process must not open a real device: no camera SDK, no USB / HID.
#     A value already set is kept.
# SAS_PROJECTOR_CONTROL_FILE=<path>
#     Where the projector rig file is read instead of the machine-wide one beside
#     the rig lock. Here a path that does not exist: a missing rig file means
#     control is off, nothing is read from or written to the projector and nothing
#     is refused. The `rig_file` fixture gives one test its own file. Meant to be
#     honoured only together with SAS_NO_HARDWARE=1, so that a real session always
#     reads the machine's file.
# SAS_EVENTLOG=off
#     The inspection event log writes nothing, anywhere: no row, no waiting row,
#     no picture copy. The `eventlog_on` fixture sets "on" for one test.
NO_HARDWARE_ENV = "SAS_NO_HARDWARE"
RIG_FILE_ENV = "SAS_PROJECTOR_CONTROL_FILE"
EVENTLOG_ENV = "SAS_EVENTLOG"
RIG_FILE_NAME = "projector_control.json"

os.environ.setdefault(NO_HARDWARE_ENV, "1")
os.environ[RIG_FILE_ENV] = str(Path(tempfile.gettempdir()) / f"sas_tests_{os.getpid()}"
                               / RIG_FILE_NAME)
os.environ[EVENTLOG_ENV] = "off"

from engine import api  # noqa: E402
from engine.capture import riglock  # noqa: E402
from engine.capture.mock_bench import Box, MockBench, VALIDATION_OBJECT  # noqa: E402,F401

# frozen copies: bench tuning of presets/ must not change test expectations
FIXTURE_PRESETS = ROOT / "tests" / "fixtures" / "presets"
PRESET = "cell_wall_2p7ft_vstripe"   # fringes vertical in camera = correct for a right-side projector


# ------------------------------------------------- isolation guard
# Two things no test may do, whatever the code under test makes of the variables:
#   - open the machine's projector rig file (beside the rig lock);
#   - write an event-log file (events/inspections_v1_*, events/inspection_values_v1_*,
#     events/last_v1*, the two lock files, events/pending/, events/img/, a rebuild's
#     events/rebuild_*/ and events/replaced_*/) without the `eventlog_on` fixture, or
#     outside a temporary folder.
# An audit hook sees every open() and rename in this process. It refuses the call
# (IsolationError) and notes it; the note fails the running test at its end, also
# when the code under test swallowed the refusal. Child processes are not watched:
# they inherit the variables above.

class IsolationError(AssertionError):
    """A test reached a file it must not touch (the audit hook in tests/conftest.py)."""


def _forms(path) -> set:
    """A path as the guard compares it: absolute, and with links / short names resolved."""
    p = os.fspath(path)
    return {os.path.normcase(os.path.abspath(p)), os.path.normcase(os.path.realpath(p))}


class _Guard:
    eventlog_on = False
    notes = []
    # taken now: a test may point PROGRAMDATA or the temp folder somewhere else later
    machine = set().union(*(_forms(p.with_name(RIG_FILE_NAME))
                            for p in riglock.lock_candidates()))
    temp = {f + os.sep for f in _forms(tempfile.gettempdir())}


_EVENT_FILES = ("inspections_v1_", "inspection_values_v1_", "last_v1", "eventlog.lock",
                "pending.lock")
_EVENT_DIRS = ("pending", "img")
_EVENT_WORK_DIRS = ("rebuild_", "replaced_")        # a rebuild's work folders


def _is_eventlog(full: str) -> bool:
    parts = [p.lower() for p in Path(full).parts]
    for i, part in enumerate(parts[:-1]):
        rest = parts[i + 1:]
        if part == "events" and (rest[0].startswith(_EVENT_FILES) or (
                len(rest) > 1 and (rest[0] in _EVENT_DIRS
                                   or rest[0].startswith(_EVENT_WORK_DIRS)))):
            return True
    return False


def _writes(mode, flags) -> bool:
    if isinstance(mode, str):
        return any(c in mode for c in "wax+")
    return bool(int(flags or 0) & (os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT
                                   | os.O_TRUNC))


def _refused(event, args):
    """The reason this open / rename is refused, or None."""
    if event == "open":
        path, mode, flags = args[0], args[1], args[2]
    else:
        path, mode, flags = args[1], "w", 0              # the name it gets
    try:
        text = os.fsdecode(path)
    except TypeError:                                    # a file descriptor
        return None
    low = text.lower()
    if RIG_FILE_NAME not in low and "events" not in low:
        return None
    forms = _forms(text)
    if forms & _Guard.machine:
        return (f"the machine's projector rig file was opened ({text}): a test reads the "
                f"path in {RIG_FILE_ENV} (the rig_file fixture gives it its own)")
    if any(_is_eventlog(f) for f in forms) and _writes(mode, flags):
        if not _Guard.eventlog_on:
            return (f"an event-log file was written ({text}) while the event log is off for "
                    f"tests ({EVENTLOG_ENV}=off): the code must write nothing then; a test "
                    "that wants rows uses the eventlog_on fixture")
        if not any(f.startswith(root) for f in forms for root in _Guard.temp):
            return (f"an event-log file was written outside a temporary folder ({text}): "
                    "keep the test's library under tmp_path")
    return None


def _audit(event, args):
    if event != "open" and event != "os.rename":
        return
    try:
        why = _refused(event, args)
    except Exception:  # noqa: BLE001 - the guard itself must never break an open()
        return
    if why:
        _Guard.notes.append(why)
        raise IsolationError(why)


sys.addaudithook(_audit)


def take_isolation_notes() -> list:
    """What the guard refused since the last call, forgotten on return (for the tests
    of the guard itself)."""
    notes, _Guard.notes[:] = list(_Guard.notes), []
    return notes


@pytest.fixture(scope="session", autouse=True)
def _temp_roots(tmp_path_factory):
    _Guard.temp |= {f + os.sep for f in _forms(tmp_path_factory.getbasetemp())}


@pytest.fixture(autouse=True)
def _isolation_check():
    yield
    notes = take_isolation_notes()
    if notes:
        pytest.fail("test isolation: " + "; ".join(dict.fromkeys(notes)), pytrace=False)


@pytest.fixture
def rig_file(tmp_path, monkeypatch):
    """This test's projector rig file: a path under tmp_path, put into
    SAS_PROJECTOR_CONTROL_FILE. Nothing is written: the test writes the JSON it
    needs (no file means control is off)."""
    path = tmp_path / RIG_FILE_NAME
    monkeypatch.setenv(RIG_FILE_ENV, str(path))
    return path


@pytest.fixture
def eventlog_on(monkeypatch):
    """The inspection event log on for this test (SAS_EVENTLOG=on). Its files may
    only be written in a temporary folder: keep the library under tmp_path (the
    `eng` fixture does)."""
    monkeypatch.setenv(EVENTLOG_ENV, "on")
    _Guard.eventlog_on = True
    yield
    _Guard.eventlog_on = False


@pytest.fixture
def eng(tmp_path):
    """Engine in mock mode on a throwaway library."""
    session = api.configure(tmp_path / "library", mock=True, bench=MockBench(seed=1),
                            presets_dir=FIXTURE_PRESETS)
    yield session
    api._session = None


@pytest.fixture
def preset(eng):
    return api.load_preset(PRESET)

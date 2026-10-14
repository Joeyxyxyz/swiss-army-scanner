"""
Paths longer than the operating system takes.

Windows without long paths takes 260 characters for a path, the end mark included: at most
259 for a file's whole path and 247 for a folder's (a new folder keeps room for a short
file name). Past that a folder is not made ([WinError 206] "The filename or extension is
too long", [WinError 3] from 260 on) and a file is not opened ([Errno 2]): a library or a
runs folder deep in the file tree crashed a run half way ("unexpected FileNotFoundError",
exit 4).

Nothing here refuses a path for its length alone: where long paths work (Windows with long
paths on, any other system) the same paths are used and nothing is refused. What is judged
is what really happened:
  * too_long(e): an OS error that says the path was too long (WinError 206; ENAMETOOLONG;
    on Windows without long paths, "not found" for a path of 260 characters or more);
  * check_room(folder, room): a probe file `room` characters long is made in a folder that
    exists, and is gone again when it is closed: a folder that cannot hold the files it is
    made for is found before anything is measured. A file that goes in a new folder below
    it (new_folder=: that folder's length) needs that folder to fit too: Windows without
    long paths keeps 12 characters free in a new folder (folder_slack), so the probe is at
    least the folder's length plus those 12.
Either becomes rules.PathTooLong, a refusal (exit 2) that names the path, its length, the
limit and what to do.

Where: the session opening the library (LibraryStore.ensure_readme: the library folder and
room for the session log), a new run folder (engine/runs.new_run: room for the files a run
writes; a flow's cycle folder: room for its work states' run folders), a new library entry
or output-bin folder and the store's own file writes (LibraryStore); the command line turns
any other too-long OS error into the same refusal (ui/cli._classify_error).
"""

from __future__ import annotations

import errno
import functools
import os
import uuid
from contextlib import contextmanager
from typing import Optional

from .. import rules

MAX_PATH = 260                  # Windows without long paths: a path's characters, end mark included
FILE_MAX = MAX_PATH - 1         # 259: the longest file path
DIR_MAX = MAX_PATH - 13         # 247: the longest folder path (room kept for a short file name)
ERROR_FILENAME_EXCED_RANGE = 206    # WinError: "The filename or extension is too long"
PROBE = "~room"                 # the probe file's name starts so (it is gone when closed)

FIX_LIBRARY = ("use a library folder with a shorter path, nearer the drive root (--library), "
               "or have Windows long paths turned on (an administrator's setting, "
               "LongPathsEnabled), then run again")
FIX_RUNS = ("use a library or runs folder with a shorter path, nearer the drive root "
            "(--library; a station configuration's runs_dir or SCANNER_RUNS_DIR), or have "
            "Windows long paths turned on (an administrator's setting, LongPathsEnabled), then "
            "run again")
FIX_ANY = ("use a folder with a shorter path, nearer the drive root (the library: --library; "
           "runs: runs_dir or SCANNER_RUNS_DIR; the bad-part output), or have Windows long "
           "paths turned on (an administrator's setting, LongPathsEnabled), then run again")


@functools.lru_cache(maxsize=1)
def long_paths() -> Optional[bool]:
    """Windows: may this program use paths past 260 characters? Windows' own answer
    (RtlAreLongPathsEnabled: the LongPathsEnabled setting and the program's manifest).
    False on a Windows too old to have long paths; None on another system, or when Windows
    cannot be asked. Read once: Windows reads it when the program starts."""
    if os.name != "nt":
        return None
    try:
        import ctypes
        fn = ctypes.WinDLL("ntdll").RtlAreLongPathsEnabled
    except AttributeError:
        return False                    # before Windows 10 1607: no long paths at all
    except OSError:
        return None
    fn.restype = ctypes.c_ubyte
    fn.argtypes = []
    try:
        return bool(fn())
    except OSError:
        return None


def chars(path) -> int:
    """A path's length as the OS takes it: absolute, in characters."""
    try:
        return len(os.path.abspath(os.fsdecode(path)))
    except (TypeError, ValueError, OSError):
        return len(str(path))


def too_long(e: BaseException, path=None) -> bool:
    """Did `e` happen because a path was too long for the OS? Only an error that happened is
    judged (never a length alone): WinError 206, ENAMETOOLONG, or - on Windows without long
    paths - "not found" for a path of MAX_PATH characters or more (e's own file names, else
    `path`), where nothing can be found or made."""
    if isinstance(e, rules.PathTooLong):
        return True
    if not isinstance(e, OSError):
        return False
    if getattr(e, "winerror", None) == ERROR_FILENAME_EXCED_RANGE:
        return True
    if e.errno == errno.ENAMETOOLONG:
        return True
    if os.name == "nt" and e.errno == errno.ENOENT and long_paths() is not True:
        names = [p for p in (e.filename, getattr(e, "filename2", None), path) if p is not None]
        return any(chars(p) >= MAX_PATH for p in names)
    return False


def folder_slack() -> int:
    """What a new folder's path leaves free below a file's limit: on Windows without long
    paths a folder keeps room for a short file name, FILE_MAX - DIR_MAX = 12 characters;
    elsewhere a folder's path has the limit a file's has, 0."""
    return FILE_MAX - DIR_MAX if os.name == "nt" and long_paths() is not True else 0


def _limit(error: Optional[OSError]) -> str:
    lp = long_paths()
    if os.name == "nt" and lp is not True:
        why = ", and long paths are off for this program on this PC" if lp is False else \
            " unless long paths are on"
        return (f"Windows takes at most {FILE_MAX} characters for a file's path and "
                f"{DIR_MAX} for a folder's{why}")
    said = " ".join(str(error).split()) if error is not None else ""
    return "the file system did not take a path that long" + (f" ({said})" if said else "")


def refusal(what: str, path, *, room: Optional[int] = None, holds: str = "the files it holds",
            fix: str = FIX_LIBRARY, error: Optional[OSError] = None,
            done: str = "", new_folder: int = 0) -> rules.PathTooLong:
    """The PathTooLong for `what` at `path`: its length, the limit, what to do. room: the
    folder exists but cannot hold a name `room` characters long (`holds`: what needs it);
    new_folder: the length of the new folder below it that name goes in (0: none). done: what
    was not done ("Nothing was measured.")."""
    n = chars(path)
    if room is None:
        head = f"{what} cannot be written: its path is {n} characters long ({path})"
    else:
        head = (f"{what} has no room for {holds}: its path is {n} characters long ({path}), "
                f"and with their names those paths reach {n + 1 + int(room)} characters"
                + (f" (a folder among them {n + 1 + int(new_folder)})" if new_folder else ""))
    msg = f"{head}; {_limit(error)}. {fix[:1].upper()}{fix[1:]}." + (f" {done}" if done else "")
    e = rules.PathTooLong(msg)
    e.errno, e.filename = errno.ENOENT, os.fspath(path)
    e.chars, e.room = n, room
    return e


def explain(e: OSError, path=None, fix: str = FIX_ANY) -> str:
    """One message for a too-long OS error that no guard caught (the command line, the
    bad-part output's warnings)."""
    if isinstance(e, rules.PathTooLong):
        return str(e)
    return str(refusal("a file or folder", e.filename or path or "?", fix=fix, error=e))


@contextmanager
def guard(what: str, path=None, *, fix: str = FIX_LIBRARY, done: str = ""):
    """Turn an OS error that says a path was too long into rules.PathTooLong (any other
    error passes as it was). The message names the longer of `path` and the error's own
    file name (a folder made with its parents fails at the first parent too long)."""
    try:
        yield
    except rules.PathTooLong:
        raise
    except OSError as e:
        if not too_long(e, path):
            raise
        where = max((p for p in (e.filename, path) if p is not None), key=chars)
        raise refusal(what, where, fix=fix, error=e, done=done) from e


def check_room(folder, room: int, what: str, *, holds: str = "the files it holds",
               fix: str = FIX_LIBRARY, done: str = "", new_folder: int = 0) -> None:
    """Refuse (rules.PathTooLong) when `folder` - it exists - cannot hold a file whose name
    is `room` characters long (`holds`: what needs that room, for the message): a probe
    file of that name is made and is gone again when it is closed (Windows deletes it on
    close; elsewhere it is unlinked at once). new_folder: that name is a path whose new
    folder (`new_folder` characters, e.g. sessions/<time>) must be made too: the probe is at
    least new_folder + folder_slack() long, so a folder the OS would not make is refused here
    as well. A probe that fails for any other reason (no write permission, a full disk) is
    left to the real write, which says what it is. One file name holds at most 255
    characters: room is taken up to that."""
    need = max(int(room), int(new_folder) + folder_slack()) if new_folder else int(room)
    n = min(max(need, len(PROBE) + 8), 255)
    name = (PROBE + uuid.uuid4().hex).ljust(n, "0")[:n]
    p = os.path.join(os.fspath(folder), name)
    flags = (os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_TEMPORARY", 0)
             | getattr(os, "O_NOINHERIT", 0))
    try:
        fd = os.open(p, flags)
    except OSError as e:
        if too_long(e, p):
            raise refusal(what, folder, room=room, holds=holds, fix=fix, error=e,
                          done=done, new_folder=new_folder) from e
        return
    try:
        if not getattr(os, "O_TEMPORARY", 0):
            os.unlink(p)
    except OSError:
        pass
    finally:
        os.close(fd)

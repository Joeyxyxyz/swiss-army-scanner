"""
Rig lock — one process at a time drives the real camera + DMD.

The camera and the projector window are single physical devices. Two
processes scanning at once (the web UI and a CLI, or Claude and Grok)
fight over the HDMI window and the camera buffers and both get garbage
frames — the stale-frame failure seen on the bench 2026-09-22. The UI
serialises its own jobs on one worker thread; this lock serialises
PROCESSES — including the DLPC350 USB tool (dmd_control), whose write
commands make the DMD re-acquire its HDMI input.

It is an OS file lock (msvcrt on Windows, fcntl elsewhere) on a fixed,
machine-wide path, so it holds across library folders and is released by
the OS if the holding process dies — no stale-lock cleanup, and never a
process kill. Real hardware only; mock runs don't take it.

Where: %PROGRAMDATA%\\swiss_army_scanner\\rig.lock — one path for every
user and every process on the machine, whatever TMP / TEMP say (a process
started with another TEMP, e.g. from a service, an IDE or another user's
shell, used to get a different lock file and no mutual exclusion). Only when
PROGRAMDATA is missing or unwritable does it fall back to the temp dir. A
lock file another user created (and this one may not write) is still
locked, opened read-only.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

from .. import rules

LOCK_DIR_NAME = "swiss_army_scanner"
LOCK_FILE_NAME = "rig.lock"
FALLBACK_LOCK_PATH = Path(tempfile.gettempdir()) / "swiss_army_scanner_rig.lock"


def machine_lock_path() -> Optional[Path]:
    """%PROGRAMDATA%\\swiss_army_scanner\\rig.lock, or None without PROGRAMDATA."""
    base = os.environ.get("PROGRAMDATA")
    return Path(base) / LOCK_DIR_NAME / LOCK_FILE_NAME if base else None


def lock_candidates() -> list:
    """Lock paths in order of preference: machine-wide, then the temp-dir fallback."""
    out = [p for p in (machine_lock_path(), FALLBACK_LOCK_PATH) if p is not None]
    return list(dict.fromkeys(out))


# the machine-wide path (what RigLock() uses unless PROGRAMDATA is unusable)
LOCK_PATH = lock_candidates()[0]


class RigLock:
    def __init__(self, path: Optional[Path] = None, holder: str = ""):
        """path None: the machine-wide lock (lock_candidates), resolved at
        acquire(); an explicit path is used as given (tests, tools)."""
        self._candidates = [Path(path)] if path is not None else lock_candidates()
        self.path = self._candidates[0]
        self.info_path = self.path.with_suffix(".info.json")
        self.holder = holder or f"{Path(sys.argv[0]).name or 'python'}"
        self._fh = None

    def _open(self):
        """Open the first usable candidate. Only a candidate whose folder or
        file can't be created / opened at all is skipped (PROGRAMDATA
        unwritable); a lock HELD by someone else is never a reason to move on."""
        last: Optional[OSError] = None
        for p in self._candidates:
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
                try:
                    fh = open(p, "a+b")
                except PermissionError:
                    if not p.exists():
                        raise
                    fh = open(p, "rb")          # another user's lock file: lockable read-only
            except OSError as e:
                last = e
                continue
            self.path, self.info_path = p, p.with_suffix(".info.json")
            return fh
        raise OSError(f"no usable rig lock path ({', '.join(map(str, self._candidates))}): "
                      f"{last}") from last

    def acquire(self) -> "RigLock":
        fh = self._open()
        try:
            _lock(fh)
        except OSError:
            fh.close()
            raise rules.RigBusy(
                "the camera + DMD are in use by another process "
                f"({self._describe_holder()}). Only one program may drive the rig at a "
                "time - wait for it to finish (or use the running web UI's job queue).") from None
        self._fh = fh
        try:
            self.info_path.write_text(json.dumps(
                {"pid": os.getpid(), "holder": self.holder,
                 "since": time.strftime("%Y-%m-%dT%H:%M:%S")}), encoding="utf-8")
        except OSError:
            pass
        return self

    def release(self) -> None:
        if self._fh is None:
            return
        try:
            _unlock(self._fh)
        finally:
            self._fh.close()
            self._fh = None

    def _describe_holder(self) -> str:
        try:
            d = json.loads(self.info_path.read_text(encoding="utf-8"))
            return f"pid {d.get('pid')} '{d.get('holder')}' since {d.get('since')}"
        except (OSError, ValueError):
            return "unknown holder"

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *exc):
        self.release()

    @property
    def held(self) -> bool:
        return self._fh is not None


def _lock(fh) -> None:
    if os.name == "nt":
        import msvcrt
        fh.seek(0)
        msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
    else:  # pragma: no cover - bench is Windows
        import fcntl
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock(fh) -> None:
    if os.name == "nt":
        import msvcrt
        fh.seek(0)
        msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
    else:  # pragma: no cover
        import fcntl
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def holder_info(path: Optional[Path] = None) -> Optional[dict]:
    """Who holds (or last held) the lock — for status displays. Without a
    path, the first candidate (machine-wide, then fallback) that has a record."""
    for p in ([Path(path)] if path is not None else lock_candidates()):
        try:
            return json.loads(p.with_suffix(".info.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
    return None

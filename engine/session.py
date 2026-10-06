"""
ScanSession: owns the library store, the session log and (in mock mode)
the simulated bench. Every verb runs through `run()`, which logs one
structured record — success or failure — to sessions/<ts>/log.jsonl.
"""

from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Optional

from .capture.camera import open_camera
from .capture.mock_bench import MockBench
from .capture.projector import Projector
from .capture.sync import FreeRunSync
from .library.store import LibraryStore
from .log import records


def default_holder() -> str:
    """Rig-lock holder name when the caller set none: the program + pid."""
    return f"{Path(sys.argv[0]).name or 'python'} (scan engine, pid {os.getpid()})"


class ScanSession:
    def __init__(self, root: Path, *, mock: bool, presets_dir: Optional[Path] = None,
                 bench: Optional[MockBench] = None, holder: str = ""):
        self.store = LibraryStore(root)
        self.mock = mock
        self.presets_dir = presets_dir
        self.bench = (bench or MockBench()) if mock else None
        self.preset = None
        # who drives the rig, as another process's RigBusy message names it
        # ("web UI", "cli", an agent's name); "" = default_holder()
        self.holder = holder
        self._log_path: Optional[Path] = None

    @property
    def log_path(self) -> Path:
        if self._log_path is None:
            self._log_path = self.store.new_dir("sessions") / "log.jsonl"
        return self._log_path

    def run(self, verb: str, args: dict, call: Callable):
        t0 = time.perf_counter()
        rec = records.Record(verb=verb, args={**args, "mock": self.mock})
        try:
            result = call()
        except Exception as e:
            rec.ok, rec.error = False, f"{type(e).__name__}: {e}"
            rec.duration_s = round(time.perf_counter() - t0, 4)
            records.append(self.log_path, rec)
            raise
        rec.result = result
        rec.duration_s = round(time.perf_counter() - t0, 4)
        records.append(self.log_path, rec)
        return result

    @contextmanager
    def rig(self, preset, zone_mask=None):
        """Camera first, then projector: a camera failure never leaves a
        fullscreen window up on the EVM. On the way out every part is closed
        on its own (a camera.close() error can't skip the projector or the
        lock), and a cleanup error never hides the error that ended the run."""
        camera = None
        projector = None
        lock = None
        primary = None
        try:
            if not self.mock:
                from .capture.riglock import RigLock
                # one process drives the rig (machine-wide lock, capture/riglock.py)
                lock = RigLock(holder=getattr(self, "holder", "") or default_holder()).acquire()
            if self.mock:
                projector = Projector(preset.projection, mock=True)
                camera = open_camera(preset.camera, mock=True, bench=self.bench,
                                     window=projector.window)
            else:
                camera = open_camera(preset.camera, mock=False)
                projector = Projector(preset.projection, mock=False)
            projector.zone_mask = zone_mask
            sync = FreeRunSync(projector, camera, average=preset.camera.average,
                               flush=0 if self.mock else 4)
            yield sync
        except BaseException as e:
            primary = e
            raise
        finally:
            _release_rig(camera, projector, lock, primary)

    def acquire_field(self, preset, label: str, zone_mask=None, axis=None):
        """Project the preset's bases, capture, decode -> DeformationField.

        Mock and real run the same path; only the rig differs.
        """
        from . import rules
        from .bases.hybrid import from_preset_bases
        # a full scan is a sustained projection: rule 3's average LED limits apply
        rules.check_led_duration(preset.projection.led, None, "a full scan")
        if any(b.get("name") == "multifreq" for b in preset.bases) and not any(
                b.get("name") in ("hybrid", "graycode") for b in preset.bases):
            from .bases.multifreq import from_spec
            basis = from_spec(next(b for b in preset.bases if b["name"] == "multifreq"),
                              axis=axis)
        else:
            basis = from_preset_bases(preset.bases, axis=axis)
        t0 = time.perf_counter()
        with self.rig(preset, zone_mask=zone_mask) as sync:
            frames = [sync.shoot(p) for p in basis.generate()]
        fld = basis.decode(frames, preset.geometry_id,
                           contrast_min=preset.decode.contrast_min,
                           conf_min=preset.decode.conf_min)
        fld.meta.update({"preset": preset.name, "label": label, "mock": self.mock,
                         "camera": vars(preset.camera), "bases": preset.bases,
                         "capture_s": round(time.perf_counter() - t0, 2)})
        return fld


def _release_rig(camera, projector, lock, primary: Optional[BaseException]) -> None:
    """Close camera, then projector, then release the rig lock — each step
    runs even if an earlier one raised. With no primary error the first
    cleanup error is raised (after every step ran); with one, cleanup errors
    are only noted on it and logged to stderr, so the real cause survives."""
    errors = []
    for what, obj, method in (("camera", camera, "close"), ("projector", projector, "close"),
                              ("rig lock", lock, "release")):
        if obj is None:
            continue
        try:
            getattr(obj, method)()
        except Exception as e:          # noqa: BLE001 - every part must still be released
            errors.append((what, e))
    if not errors:
        return
    notes = [f"rig cleanup: {what} {type(e).__name__}: {e}" for what, e in errors]
    if primary is not None:
        for n in notes:
            print(n, file=sys.stderr)
            if hasattr(primary, "add_note"):
                primary.add_note(n)
        return
    first = errors[0][1]
    for n in notes[1:]:
        print(n, file=sys.stderr)
        if hasattr(first, "add_note"):
            first.add_note(n)
    raise first

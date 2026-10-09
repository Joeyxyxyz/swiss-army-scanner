"""
Check registry: every check type a recipe zone can name, by id.

Built-ins (engine/recipes/checks/builtin.py): presence, position, height, flatness, defect,
edge, code. Add your own with the decorator (docs/CHECKS.md has a 20-line example):

    from engine.recipes import register
    from engine.recipes.checks.base import Check, CheckResult, Param

    @register
    class MyCheck(Check):
        id = "my_check"
        ...

or ship it in a package with an entry point in the group `swiss_army_scanner.checks`
(pyproject: [project.entry-points."swiss_army_scanner.checks"] my_check = "pkg.mod:MyCheck");
entry points are loaded the first time the registry is read. The built-in ids are reserved
(BUILTIN_IDS): a plugin under one of them is refused at once, whatever the order or thread,
and the built-ins are loaded (and checked complete) the first time the registry is read.
No lock is held while a module imports (no deadlock with Python's import locks); a plugin
module must not read the registry (get / available) while it is being imported.
"""

from __future__ import annotations

import importlib
import logging
import sys
import threading

from .base import Check, CheckResult, Param, ParamError  # noqa: F401  (public)

log = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "swiss_army_scanner.checks"
BUILTIN_MODULE = __name__ + ".builtin"
BUILTIN_IDS = ("presence", "position", "height", "flatness", "defect", "edge", "code")
REGISTRY: dict = {}
_LOADED = {"builtin": False, "entry_points": False}
_LOCK = threading.Lock()                    # REGISTRY / _LOADED only: never held over an import
_EP_LOCK = threading.Lock()                 # one entry-point loader at a time (no registry lock)
LOAD_ERRORS: list = []                      # entry points that failed to load (listed, not fatal)


def _is_builtin(cls) -> bool:
    return getattr(cls, "__module__", None) == BUILTIN_MODULE


def register(cls):
    """Register a Check subclass under its `id` (usable as a class decorator). Re-registering
    the same class is a no-op; a built-in id is refused to any class but the built-in's own
    (a re-import of the built-in module replaces its classes); another class under a taken id
    is refused."""
    if not (isinstance(cls, type) and issubclass(cls, Check)):
        raise TypeError(f"register() takes a Check subclass, got {cls!r}")
    cid = getattr(cls, "id", "")
    if not isinstance(cid, str) or not cid or not cid.replace("_", "").isalnum():
        raise ValueError(f"{cls.__name__}.id must be a non-empty [A-Za-z0-9_] name, got {cid!r}")
    for name, spec in (cls.params or {}).items():
        if not isinstance(spec, Param):
            raise TypeError(f"{cls.__name__}.params[{name!r}] must be a Param")
    if cid in BUILTIN_IDS and not _is_builtin(cls):
        raise ValueError(f"check id {cid!r} is already registered by the built-in check "
                         f"{cid!r} ({BUILTIN_MODULE}): give yours another id")
    with _LOCK:
        have = REGISTRY.get(cid)
        if have is not None and have is not cls and not (_is_builtin(have) and _is_builtin(cls)):
            raise ValueError(f"check id {cid!r} is already registered by check "
                             f"{have.__module__}.{have.__name__}: give yours another id")
        REGISTRY[cid] = cls
    return cls


def unregister(cid: str) -> None:
    """Remove a check type (tests, plugin reloads)."""
    with _LOCK:
        REGISTRY.pop(cid, None)
        if cid in BUILTIN_IDS:
            _LOADED["builtin"] = False          # the next read restores it


def _load_builtin() -> None:
    """Import the built-in checks (they register themselves) and make sure every built-in id
    holds the FINISHED module's class: marked loaded only then, so a direct import of the
    module that failed or is still running (in this thread) is never taken for loaded, and
    a stale half-registered set from a failed import is replaced. A failed import takes
    its half-registered checks out and raises; the next read retries it whole."""
    if _LOADED["builtin"]:
        return
    try:
        mod = importlib.import_module(BUILTIN_MODULE)   # no registry lock held over it
    except BaseException:
        with _LOCK:
            for k in [k for k, v in REGISTRY.items() if _is_builtin(v)]:
                REGISTRY.pop(k, None)
        raise
    if getattr(getattr(mod, "__spec__", None), "_initializing", False) or             sys.modules.get(BUILTIN_MODULE) is not mod:
        return                                  # still being imported (this thread): not yet
    own = {c.id: c for c in vars(mod).values()
           if isinstance(c, type) and issubclass(c, Check) and _is_builtin(c)
           and getattr(c, "id", None) in BUILTIN_IDS}
    with _LOCK:
        for cid, cls in own.items():
            REGISTRY[cid] = cls                 # the finished module's classes win
        if all(_is_builtin(REGISTRY.get(cid)) for cid in BUILTIN_IDS):
            _LOADED["builtin"] = True


def _load() -> None:
    _load_builtin()
    if not _LOADED["entry_points"]:
        with _EP_LOCK:
            _load_entry_points()


def _load_entry_points() -> None:
    if _LOADED["entry_points"]:
        return
    try:
        from importlib.metadata import entry_points
        eps = entry_points(group=ENTRY_POINT_GROUP)
    except Exception as e:                  # noqa: BLE001 - no metadata: built-ins only
        LOAD_ERRORS.append(f"entry points: {type(e).__name__}: {e}")
        _LOADED["entry_points"] = True
        return
    for ep in eps:
        try:
            obj = ep.load()
            if isinstance(obj, type) and issubclass(obj, Check):
                register(obj)
        except Exception as e:              # noqa: BLE001 - a broken plugin is listed, not fatal
            LOAD_ERRORS.append(f"{ep.name}: {type(e).__name__}: {e}")
            log.warning("check plugin %s failed to load: %s", ep.name, e)
    _LOADED["entry_points"] = True


def get(cid: str):
    """The Check class registered as `cid` (KeyError with the known ids)."""
    _load()
    try:
        return REGISTRY[cid]
    except KeyError:
        raise KeyError(f"no check type {cid!r} (known: {', '.join(sorted(REGISTRY))})") from None


def available() -> dict:
    """{id: description} of every registered check (built-ins + plugins)."""
    _load()
    return {k: REGISTRY[k].describe() for k in sorted(REGISTRY)}

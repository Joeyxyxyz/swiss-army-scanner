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
entry points are loaded the first time the registry is read.
"""

from __future__ import annotations

import logging

from .base import Check, CheckResult, Param, ParamError  # noqa: F401  (public)

log = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "swiss_army_scanner.checks"
REGISTRY: dict = {}
_LOADED = {"builtin": False, "entry_points": False}
LOAD_ERRORS: list = []                      # entry points that failed to load (listed, not fatal)


def register(cls):
    """Register a Check subclass under its `id` (usable as a class decorator). Re-registering
    the same class is a no-op; another class under a taken id is refused."""
    if not (isinstance(cls, type) and issubclass(cls, Check)):
        raise TypeError(f"register() takes a Check subclass, got {cls!r}")
    cid = getattr(cls, "id", "")
    if not isinstance(cid, str) or not cid or not cid.replace("_", "").isalnum():
        raise ValueError(f"{cls.__name__}.id must be a non-empty [A-Za-z0-9_] name, got {cid!r}")
    for name, spec in (cls.params or {}).items():
        if not isinstance(spec, Param):
            raise TypeError(f"{cls.__name__}.params[{name!r}] must be a Param")
    have = REGISTRY.get(cid)
    if have is not None and have is not cls:
        raise ValueError(f"check id {cid!r} is already registered by {have.__module__}."
                         f"{have.__name__}")
    REGISTRY[cid] = cls
    return cls


def unregister(cid: str) -> None:
    """Remove a check type (tests, plugin reloads)."""
    REGISTRY.pop(cid, None)


def _load() -> None:
    if not _LOADED["builtin"]:
        _LOADED["builtin"] = True
        from . import builtin  # noqa: F401  (registers the built-in checks)
    if not _LOADED["entry_points"]:
        _LOADED["entry_points"] = True
        try:
            from importlib.metadata import entry_points
            eps = entry_points(group=ENTRY_POINT_GROUP)
        except Exception as e:              # noqa: BLE001 - no metadata: built-ins only
            LOAD_ERRORS.append(f"entry points: {type(e).__name__}: {e}")
            return
        for ep in eps:
            try:
                obj = ep.load()
                if isinstance(obj, type) and issubclass(obj, Check):
                    register(obj)
            except Exception as e:          # noqa: BLE001 - a broken plugin is listed, not fatal
                LOAD_ERRORS.append(f"{ep.name}: {type(e).__name__}: {e}")
                log.warning("check plugin %s failed to load: %s", ep.name, e)


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

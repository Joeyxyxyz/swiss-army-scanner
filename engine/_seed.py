"""
Load proven modules from the `swiss_army_sensor` package (shipped in this
repo next to `engine/`, installed by `pip install -e .`) WITHOUT
executing its package __init__ files.

Why: swiss_army_sensor/__init__.py and drivers/__init__.py both import
drivers/dlp_driver.py (the DLPC350 USB-HID driver). Rule §9.1 says that
driver is never imported in the scan path, so the engine loads only the
individual, numpy-only modules it needs straight from their files.

`importlib.util.find_spec` on a top-level name locates the package without
importing it.
"""

from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path

# name -> path inside the package. Anything touching USB is absent on purpose.
SEED_MODULES = {
    "patterns": "patterns.py",
    "decoders": "decoders.py",
    "camera_driver": "drivers/camera_driver.py",
}


@lru_cache(maxsize=None)
def package_dir() -> Path:
    spec = importlib.util.find_spec("swiss_army_sensor")
    if spec is None or not spec.submodule_search_locations:
        raise ImportError(
            "swiss_army_sensor is not importable — the engine is seeded from it. "
            "It ships in this repo: run `pip install -e .` from the repo root "
            "(or run from the repo root so it is on sys.path)")
    return Path(list(spec.submodule_search_locations)[0])


@lru_cache(maxsize=None)
def load(name: str):
    rel = SEED_MODULES[name]
    modname = f"engine._seed_{name}"
    spec = importlib.util.spec_from_file_location(modname, package_dir() / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod

"""
Recipes: what the DLP checks to verify a work state, portable to any line with a DMD
projector + a camera.

    preset (the rig) -> recipe -> zones -> checks -> feedback (projected) -> outputs (I/O)

A station configuration's profiles name the recipes of each part variant; the work states of
its flow name which one verifies them (engine/flow/).

A recipe (schema.py, versioned JSON / YAML) pins a preset, says how to scan, where the zones
are (camera px, projector px or part-frame mm) and which checks run in each; the runner
(runner.py) scans, runs the checks, writes a run folder (result.json + run.log.jsonl) and
paints the verdict back on the part with the projector (feedback.py): green outline = PASS,
red fill + a cross at the fault = FAIL. Built-in recipes live in the repo's recipes/ folder
(read-only); user recipes in the library (library.py). Checks are plugins (checks/):

    from engine.recipes import register
    @register
    class MyCheck(Check): ...

The engine verbs are api.recipe_run / recipe_golden / recipe_feedback / recipe_add /
recipe_select / recipe_delete (+ recipe_list, recipe_show, recipe_active, recipe_checks,
recipe_validate); the CLI `scan recipe-*` and `scan run` (a single job with job.type recipe,
or the work states of a configuration with profiles + a flow).
"""

from .checks import available, get, register, unregister  # noqa: F401
from .checks.base import Check, CheckResult, Param, ParamError  # noqa: F401
from .schema import SCHEMA_VERSION, Recipe, RecipeError, load, validate  # noqa: F401

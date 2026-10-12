"""
Operating modes: three ways to use one triangulation rig.

The layering (each layer only adds to the one before it):

    TRIANGULATION PRESET   the basic DLP + camera setup: rig geometry, exposure, fringes and
      (fixed | mobile)     the stereo calibration bound to that geometry (presets/, `scan
                           preset-add`, `scan calibrate-stereo`). `rig.mount` says how far
                           the calibration is trusted: fixed = until a rig check says MOVED /
                           UNKNOWN (then until that is answered); mobile = only once it was
                           verified in this session. A geometry_id names one rig: a mobile
                           preset makes every preset at its geometry mobile
                           (engine/rigstate.py)
      -> [optional] STATION CONFIGURATION   what a station is and knows (stations/*.json)
           -> PROFILES                      what each part variant looks like
                -> WORK STATES              the step flow: sensor claim, DLP check, permission

    measure       MEASURE: an instrument. Scan and report numbers - no tolerance, no pass /
                  fail, no permission, no disposition; only a validity statement (MEASURED /
                  NOT_MEASURABLE). Needs only a triangulation preset with a valid stereo
                  calibration (measure.py; `scan measure-run`, api.measure_run)
    quick_check   QUICK CHECK: a triangulation preset + ONE simple check (a plane flatness
                  limit, a golden compare, or one recipe) -> CONTINUE / REJECT / PULL. No
                  sensors, no permissions: a one-step flow through the station job runner
                  (quick.py builds its minimal single-job station config; `scan quick-check`)
    station       STATION: station configuration -> profile -> work states (engine/flow/,
                  the command bus). Unchanged; it is the mode the bus runs cycles in

The mode is selected on the operator page (/quick), over HTTP (POST /api/cmd/mode) or by a
hardware input (the command bus's SELECT_MODE). Outside Station mode the bus starts no cycle,
shows no PASS and grants no permission: a measurement never reaches the line's outputs.

Pure: constants only (the command bus imports this).
"""

from __future__ import annotations

MODES = ("station", "quick_check", "measure")
DEFAULT_MODE = "station"
TITLES = {"station": "Station", "quick_check": "Quick check", "measure": "Measure"}
ALIASES = {"quick": "quick_check", "quick-check": "quick_check", "quickcheck": "quick_check"}
# what a mode needs on top of the triangulation preset
NEEDS = {"measure": "a triangulation preset + a valid stereo calibration",
         "quick_check": "a triangulation preset + one check (flatness limit, golden compare "
                        "or one recipe)",
         "station": "a station configuration: profiles + work states"}
RESULT_KEY = "operating_mode"       # in result.json: absent = a station run


def check_mode(v) -> str:
    """A mode name (any case; quick / quick-check for quick_check) -> its MODES name, or
    ValueError."""
    k = str(v).strip().lower().replace(" ", "_") if isinstance(v, str) else ""
    k = ALIASES.get(k, k)
    if k not in MODES:
        raise ValueError(f"operating mode {v!r} is not one of {', '.join(MODES)}")
    return k

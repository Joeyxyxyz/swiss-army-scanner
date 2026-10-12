"""
Station configuration -> profiles -> work states -> checks -> dispositions.

    station configuration   stations/*.json: the parameters of one station and the PROFILE
                            LIBRARY SET it can call on (schema.py)
    profile                 one part variant of that set (long tube, short tube): what it
                            looks like - recipes (zones + checks), preset, golden. The
                            operator / PLC / HTTP selects the active one (SELECT_PROFILE)
    work state              one step of the flow: a claim of the line's own sensors, the DLP
                            check that confirms or corrects it, and the permission it grants
                            (cycle.py: CONFIRMED / CORRECTED / NOT_MET / DEFECT / DOUBT)
    check                   what the DLP measures there: a recipe of the active profile, or
                            inline zone checks (verify.py -> engine/recipes/)
    disposition             what happens to the part: CONTINUE only when every required
                            work state was CONFIRMED, else REJECT / PULL (engine/disposition.py)

schema.py and cycle.py are pure (the command bus imports them); verify.py runs the DLP
check through the recipe runner and writes the cycle folder.
"""

from .cycle import OUTCOME_CODES, Cycle, CycleError, classify  # noqa: F401
from .schema import (LEGACY_PROFILE, LEGACY_STEP, OUTCOMES, SCHEMA_VERSION,  # noqa: F401
                     Profile, StationConfigError, StationConfiguration, Step, is_legacy,
                     lift_legacy, load, validate)

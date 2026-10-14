"""
Painted zones: shapes the operator draws ON THE PART with the mouse.

The projector is a second monitor, so the mouse cursor is projected onto the real part. The
painter (`scan zone-paint`) lets the operator drag rectangles, ellipses and polygons out over
the areas that matter, move / resize / rotate them until they fit, name them, and confirm.
The result is a ZONE SET in the library, in projector px, bound to the rig geometry it was
painted at:

    <library>/zones/<geometry_id>/<name>/   zones.json + masks.npz + preview.png (+ camera.png)

and it is used wherever a lit zone is named (text "zoneset:<set>[:Z1,Z3]"): `scan scan-fast
--zones`, `scan measure-run --zones`, a recipe's scan.zone, and a recipe zone's
"ref": "zoneset:<set>/<id>".

    model.py     Zone / ZoneSet, validation, rasterising to DMD masks (pure geometry)
    painter.py   PainterModel: the whole interaction as a display-free state machine
                 (mouse / key events -> state), render(state) -> the DMD image
    session.py   the thin window shell around it (the rig, the rig lock, polarity), and
                 zone-show (project a saved set, report where each zone lands)
    store.py     save / load / list / delete (to trash) in the library
    use.py       scan masks, per-zone point counts, recipe zones and references, regions;
                 several sets in one text ("zoneset:<a>+<b>:Z1"), zones keyed "<set>/<id>"
    selection.py which sets (and zones) are switched on at a rig placement: the stored
                 selection, the word `selection` wherever a zone text is taken

Limits: zones are PROJECTOR px, valid for the rig placement they were painted at (the same
geometry_id + fingerprint is enforced; a rig that moved without a new geometry_id cannot be
detected here: re-paint, or `scan zone-show` to check, after the rig moves). Where a zone's
mirrors land in the CAMERA image depends on the depth of what they hit, so camera px are
derived per scan from that scan's own decode, never stored.
"""

from .model import (DMD_SHAPE, MIN_SIZE_PX, SHAPES, Zone, ZoneError, ZoneSet,  # noqa: F401
                    next_id, zone_problems)
from .painter import KEYS, PainterModel, PainterState, render  # noqa: F401

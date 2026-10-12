# Recipe checks: built-ins and how to add your own

A recipe zone names the checks that run there (`recipes/README.md` has the schema). A check
is a small class: it reads ONE zone of ONE metric scan and answers **PASS / FAIL / INVALID**
with its numbers and the fault points the projector marks. It never touches the rig.

`scan recipe-checks` (or `api.recipe_checks()`, `GET /api/recipes`) lists every check with
its parameters, units and defaults.

## Built-in checks

| type | what it measures | key parameters (units in the names) | reuses |
|---|---|---|---|
| `presence` | part present / absent: points at least `min_height_mm` above the reference plane cover `min_area_pct` of the zone's whole extent | `expect` present / absent, `min_height_mm`, `min_area_pct`, `min_valid_pct`, `min_points` (an unseen zone is INVALID, never "absent": see coverage below) | the part frame |
| `position` | alignment vs the golden part: in-plane offset (centroid), rotation about the plane normal (principal axis), height offset | `tol_mm`, `tol_deg`, `tol_height_mm`, `min_height_mm` | the recipe's golden scan |
| `height` | a statistic of the zone's heights vs the reference plane, or vs another zone (`ref: "zone:<id>"`: a step, the height difference of two faces); headline `deviation_mm` = height - the band's centre, limit = the band's half width | `nominal_mm` + `tol_mm`, or `band_mm` [lo, hi]; `stat` median / mean / p95 / p5 / max / min | |
| `flatness` | robust plane fit: peak-to-valley between the trimmed percentiles, RMS. "Flat" needs coverage (see below) | `tol_mm`, `rms_tol_mm`, `trim_pct`, `min_height_mm`, `min_valid_pct` | `engine/measure/endface/profile.robust_plane` |
| `defect` | local dents / bumps. `face` mode (circle zone = an end face): the MF-CFAR matched-filter detector, self-referenced; `local` mode: residual vs the zone's own smooth surface, connected blobs. "No defect" needs coverage (see below) | `depth_mm`, `min_area_mm2`, `mode` auto / face / local, `sign`, `smooth_mm`, `z_thresh`, `min_valid_pct`, `max_gap_mm2` | `engine/measure/endface/mfcfar.py` |
| `edge` | the part's long edge(s): sub-pixel edge in the flood image (or the height map), robust line fit in the part frame; straightness, and position / angle vs the golden (lines compared undirected) | `side` top / bottom / both (every side asked for must be measured, in the scan and the golden), `straightness_tol_mm`, `position_tol_mm`, `angle_tol_deg` | `engine/recipes/checks/builtin.py` `edge_points` (gradient-peak profiles across the part, parabolic sub-pixel) |
| `code` | a barcode / QR / Data Matrix in the zone: the codes read under DMD light before the scan (`scan.codes: true`), else the zone of the flood image decoded. A code of the expected kind read with the wrong text FAILs (REJECT); nothing of that kind read at all FAILs with `pull: "code_unreadable"` (PULL: unreadable is not wrong) | `expect`, `pattern` (regex), `kind`, `min_count` | `engine/sense/codes.py` |

Verdict rules: a check that cannot measure (too few points, too little of the zone measured,
no golden, no image) answers **INVALID**; a check that raises, or answers anything but a
`CheckResult` with PASS / FAIL / INVALID, is recorded as INVALID with its error (a bug is
never a verdict). A zone is FAIL when any of its checks fails, else PASS when every check
passed, else INVALID; the run the same over its zones. On top of the verdict the run gets a
disposition (`README.md`, *Dispositions*): REJECT for a definite FAIL, PULL (quality
inspection) for any doubt - an INVALID or crashed check, a FAIL that carries a `pull`
reason, a PASS value within the recipe's `pull.guard_band_pct` of a limit in its
`tolerances` - else CONTINUE.

Coverage is judged over a zone's whole extent in its OWN units, as cells (`engine/recipes/
geometry.Coverage`): camera px for camera_px; native DMD px, binned to the camera's sampling,
for projector_px; part-frame (u, v) mm at half the camera's sampling for part_mm. A cell is
seen when a measured camera px falls in it (its own px, its decoded DMD px, or its (u, v)).
Whatever of the zone is off the camera image, occluded (a tall object outside the zone hides
it), in shadow or undecoded (a dark part), at whatever depth, is a cell nothing falls in:
unseen. An unmeasured part never enters a part_mm or projector_px zone's points, so only its
unseen cells show that it may be there. `presence` calls a zone "not present" (absent, or a
missing part) only when even every unseen cell counted as part would stay under
`min_area_pct`, and at least `min_valid_pct` was measured; `defect` local calls a zone clean
only when at least `min_valid_pct` (70 %) of its inspected extent (less what was measured
below `min_height_mm`) was measured and no unseen region of the zone - enclosed by measured
points or not: a dropout reaching the zone border, the part edge or a glint streak counts -
can hold a round defect of `max_gap_mm2` (default `min_area_mm2`); a thin streak that cannot
hold one does not stop a PASS. `flatness` calls a zone flat only with `min_valid_pct` (70 %)
of that extent measured. Otherwise INVALID; a defect or a warp that was found stays a FAIL.
So draw a defect zone inside the face it inspects: a zone reaching over a projector shadow
beside a raised part is INVALID (the shadow could hide a defect).

Golden (`golden.scan`, for `position` and the `edge` golden tolerances): a golden recorded
by `scan recipe-golden` (by its label; `latest` = the newest golden of this recipe at this
geometry) or a metric scan named by its exact id. Never the scan being checked, never a plain
metric scan matched by label (an earlier run's test scan carries the part's label), never
one made at other geometry-bound settings or with another stereo calibration: INVALID.

## Checks and work states: what the DLP saw

A work state of a station configuration's flow (`stations/README.md`) may `expect` a signal of
the line's own sensors (`part_present`) and names the check that observes it (`verify.observe`:
`"<zone>/<check>"`, required for every expected signal: no check is picked by default). What
that check **saw** is compared with what the sensor says: the DLP confirms or corrects the
sensor.

- `presence` reports what it saw whatever it expected: `details.present` (true / false) on a
  PASS or a FAIL. An INVALID presence (unseen, too few points) saw nothing: unknown.
- Any other check observes "true" when it PASSes and "false" when it FAILs (`"invert": true`
  flips it); INVALID, or a FAIL / PASS carrying a `pull` reason, is unknown.
- A reading the run itself holds in doubt (a value in the recipe's guard band) is unknown too.

Unknown is never a sensor mismatch and never a confirmation: the work state is a DOUBT (PULL).
A plugin check needs nothing special for this; set `details["present"]` only if you write
another presence-like check under the type id `presence`.

## The context a check reads

| | |
|---|---|
| `ctx.points()` | (N, 3) measured points of the zone, mm, camera frame (x right, y down, z forward) |
| `ctx.uvh()` / `ctx.heights()` | the same in the part frame: u, v (mm in the reference plane), h (mm above it, + toward the camera) |
| `ctx.index()` | (ys, xs) camera px of those points, same order |
| `ctx.fault_px(i, label)` | a fault point at point i, for `fault_points` |
| `ctx.px_of(xyz)` | camera px of a camera-frame point (the stereo calibration's intrinsics) |
| `ctx.mm_per_px()` | the zone's lateral sampling, mm per camera px |
| `ctx.zone` / `ctx.zones` | this zone (mask, measured px, outline, id) / every zone by id |
| `ctx.scan` | the cloud, the decoded DMD (col, row) per camera px, the flood pair (`ctx.scan.flood`) |
| `ctx.frame` | the part frame: reference plane, origin, axes e1 / e2 / n |
| `ctx.golden` | the golden scan's view (scan, frame, the same zones), or None; `ctx.need_golden()` |
| `ctx.codes` | codes read before the scan, or None |

## Add your own check

Subclass `Check`, declare its parameters (always with units), return a `CheckResult`:

```python
import numpy as np
from engine.recipes import register
from engine.recipes.checks.base import Check, CheckResult, Param


@register
class MaxHeight(Check):
    """Highest point of the zone above the reference plane vs a limit."""
    id = "max_height"
    title = "Highest point vs a limit"
    params = {"limit_mm": Param("number", unit="mm", min=0, doc="FAIL above this height"),
              "min_points": Param("int", 30, "count", min=1, doc="fewer points: INVALID")}

    def run(self, ctx):
        h = ctx.heights()                  # mm above the reference plane, + toward the camera
        if h.size < self.p["min_points"]:
            return CheckResult.invalid(f"only {h.size} points in zone {ctx.zone.id}")
        i = int(np.argmax(h))
        top, limit = float(h[i]), self.p["limit_mm"]
        bad = top > limit
        return CheckResult(verdict="FAIL" if bad else "PASS",
                           values={"max_height_mm": round(top, 4), "n_points": int(h.size)},
                           key="max_height_mm", limit=limit,
                           fault_points=[ctx.fault_px(i, "too high")] if bad else [],
                           message=f"highest point {top:.2f} mm (limit {limit:g} mm)")
```

Then use it in a zone like any built-in:

```json
{"id": "lid", "shape": "rect", "units": "part_mm", "coords": [-40, -25, 40, 25],
 "checks": [{"type": "max_height", "params": {"limit_mm": 41.5}}]}
```

(`tests/test_recipes_mock.py` runs this example exactly as written.)

Rules of thumb:

- **Units in every name**: values `..._mm`, `..._deg`, `..._mm2`, `..._pct`, `n_...`;
  parameters declare `unit=`. The result's `units` are derived from the names.
- `key` + `limit` name the headline number, in deviation semantics: a deviation and its
  tolerance (|value| > limit is out of tolerance; not an absolute height against a band
  edge). The run's `deviation_mm` / `threshold_mm` (and the operator page's key number)
  come from a FAILing check when there is one (the largest |value| / limit among them),
  else from the mm value closest to its limit.
- `tolerances` (optional): every value the check held to a limit,
  `[engine.disposition.tolerance(name, value, lo, hi, span=None)]` (PASS means lo <= value
  <= hi; None on a side = no limit there). Give the value the check judged, not a rounded
  copy. `span` (optional) is the size of the tolerance the guard band takes its % of when
  the default does not fit (two-sided: the half width; one-sided: |limit|), e.g. a share
  that can't pass 100 %: `part_pct >= 92` has span 8. The recipe's guard band reads them;
  without them the guard band uses `key` / `limit` as |value| <= limit. A malformed list
  makes the check INVALID (a bug is never a verdict). A PASS whose tolerance (or `key`
  value) is missing, not a finite number or outside its own limits is PULLed (`invalid`:
  the verdict and its number disagree), guard band or not; a one-sided limit at 0 needs a
  `span`, else the guard band can't clear any value of it.
- `pull` (optional): a reason from `engine.disposition.REASONS` when the verdict is in
  doubt, e.g. `"code_unreadable"` on a FAIL that read nothing: the part is PULLed for
  quality inspection instead of rejected. An unknown reason is recorded as `invalid`.
- `fault_points`: `{"px": [x, y]}` camera px or `{"xyz_mm": [x, y, z]}` (camera frame, mm:
  the runner projects it to camera px); the projector draws a cross at each one of a FAIL
  check, on the surface the camera saw there. A point that cannot be placed is listed in the
  check's warnings, never dropped silently.
- Answer INVALID, never PASS, when the data cannot support a verdict.
- `needs = ("golden",)` (or `needs_for()`) makes the validator demand `golden.scan`.
- `check_params(cls, p)` (classmethod) for cross-parameter rules; `zone_refs()` returns the
  zone ids a check refers to (validated to exist).

### Shipping a check in another package

Load it with an entry point instead of an import (the registry reads the group the first
time it is used):

```toml
[project.entry-points."swiss_army_scanner.checks"]
max_height = "my_line_checks.height:MaxHeight"
```

A plugin that fails to load is listed in `engine.recipes.checks.LOAD_ERRORS`, not fatal.
A check id is registered once: another class under a taken id is refused. The built-in ids
(`engine.recipes.checks.BUILTIN_IDS`) are reserved, so a plugin can never take one, whatever
the import order or thread; the first registry read loads the built-ins and checks they are
all there (a failed import is retried whole). No lock is held while a module imports, so a
plugin module must not read the registry (`get` / `available`) while it is being imported.

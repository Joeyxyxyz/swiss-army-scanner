# Recipe checks: built-ins and how to add your own

A recipe zone names the checks that run there (`recipes/README.md` has the schema). A check
is a small class: it reads ONE zone of ONE metric scan and answers **PASS / FAIL / INVALID**
with its numbers and the fault points the projector marks. It never touches the rig.

`scan recipe-checks` (or `api.recipe_checks()`, `GET /api/recipes`) lists every check with
its parameters, units and defaults.

## Built-in checks

| type | what it measures | key parameters (units in the names) | reuses |
|---|---|---|---|
| `presence` | part present / absent: points at least `min_height_mm` above the reference plane cover `min_area_pct` of the zone | `expect` present / absent, `min_height_mm`, `min_area_pct`, `min_valid_pct` (an unseen zone is INVALID, never "absent") | the part frame |
| `position` | alignment vs the golden part: in-plane offset (centroid), rotation about the plane normal (principal axis), height offset | `tol_mm`, `tol_deg`, `tol_height_mm`, `min_height_mm` | the recipe's golden scan |
| `height` | a statistic of the zone's heights vs the reference plane, or vs another zone (`ref: "zone:<id>"`: a step, the height difference of two faces) | `nominal_mm` + `tol_mm`, or `band_mm` [lo, hi]; `stat` median / mean / p95 / p5 / max / min | |
| `flatness` | robust plane fit: peak-to-valley between the trimmed percentiles, RMS | `tol_mm`, `rms_tol_mm`, `trim_pct`, `min_height_mm` | `engine/measure/endface/profile.robust_plane` |
| `defect` | local dents / bumps. `face` mode (circle zone = an end face): the MF-CFAR matched-filter detector, self-referenced; `local` mode: residual vs the zone's own smooth surface, connected blobs | `depth_mm`, `min_area_mm2`, `mode` auto / face / local, `sign`, `smooth_mm`, `z_thresh` | `engine/measure/endface/mfcfar.py` |
| `edge` | the part's long edge: sub-pixel edge in the flood image (or the height map), robust line fit in the part frame; straightness, and position / angle vs the golden | `side` top / bottom / both, `straightness_tol_mm`, `position_tol_mm`, `angle_tol_deg` | the camber prototype's edge method |
| `code` | a barcode / QR / Data Matrix in the zone: the codes read under DMD light before the scan (`scan.codes: true`), else the zone of the flood image decoded | `expect`, `pattern` (regex), `kind`, `min_count` | `engine/sense/codes.py` |

Verdict rules: a check that cannot measure (too few points, no golden, no image) answers
**INVALID**; a check that raises is recorded as INVALID with its error (a bug is never a
verdict). A zone is FAIL when any of its checks fails, else INVALID when any is invalid,
else PASS; the run the same over its zones.

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
- `key` + `limit` name the headline number: the run's `deviation_mm` / `threshold_mm` (and the
  operator page's key number) is the mm value closest to, or furthest past, its limit.
- `fault_points`: `{"px": [x, y]}` camera px or `{"xyz_mm": [x, y, z]}`; the projector draws a
  cross at each one of a FAIL check, on the surface the camera saw there.
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
A check id is registered once: another class under a taken id is refused.

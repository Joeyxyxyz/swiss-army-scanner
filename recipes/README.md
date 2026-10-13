# Recipes (the checks that verify a work state)

A recipe is **what the DLP checks**: which rig preset to scan with, where the zones are,
which checks run in each zone, and what the projector paints on the part after the verdict.
It is data (JSON, or YAML with PyYAML), versioned by `schema_version`, so the same engine
runs any part on any line with a DMD projector + a camera.

```
preset (the rig) -> recipe -> zones -> checks -> verdict -> feedback (projected) -> outputs (I/O)
```

Where a recipe sits (`stations/README.md`): a **station configuration** holds the station's
parameters and its **profile library set**; each **profile** (a part variant: long tube /
short tube) names the recipes that say what that part looks like; each **work state** (a
step of the configuration's flow) is verified with one of them; the outcome grants or
withholds that work state's permission. A recipe is not a work state and not a profile: it
is the check a work state runs for the active profile.

The files here are the **built-in** recipes: read-only for the recipe commands (never edited
or deleted by them). User recipes live in the library (`<library>/recipes/<name>/recipe.json`
+ `meta.json`, `recipes/active.jsonl` = the selected one), added / selected / deleted like
user presets:

| | CLI | Python | Web |
|---|---|---|---|
| list both tiers | `scan recipe-list` | `api.recipe_list()` | `GET /api/recipes` |
| show one | `scan recipe-show NAME` | `api.recipe_show(name)` | `GET /api/recipes/NAME` |
| validate (every problem listed) | `scan recipe-validate NAME` (or a file) | `api.recipe_validate(r)` | - |
| check types (params, units) | `scan recipe-checks` | `api.recipe_checks()` | `GET /api/recipes` |
| add (copy + override) | `scan recipe-add NAME --from BASE [--set key=value ...] [--desc TEXT] [--replace]` (BASE: a recipe or a file) | `api.recipe_add(name, base, overrides={...})` | verb `recipe_add` |
| select (the default) | `scan recipe-select NAME` | `api.recipe_select(name)`, `api.recipe_active()` | verb `recipe_select` |
| delete (to trash) | `scan recipe-delete NAME --reason TEXT [--force]` | `api.recipe_delete(name, reason)` | verb `recipe_delete` (`confirm=true`) |
| golden (a GOOD part) | `scan recipe-golden NAME` | `api.recipe_golden(r)` | verb `recipe_golden` |
| run | `scan recipe-run NAME [--scan ID] [--no-feedback] [--hold S]` | `api.recipe_run(r, ...)` | verb `recipe_run`; a station configuration (below) |

`--set` keys walk into sections and lists: `feedback.hold_s=10`,
`zones.0.checks.1.params.tol_mm=0.4`, `golden.scan=my_good_part`. A user recipe is validated
like a built-in one and its preset must exist; a built-in name is refused, an existing user
name too unless `--replace` (the old one moves to `trash/recipes/`). Delete refuses built-ins,
the selected recipe and a recipe a station config's `job.recipe` names (`--force` overrides).

**From a measurement.** A measurement spec (the `spec.json` a `scan measure-run` leaves in
its run folder: what was measured, in which regions) becomes a recipe by adding tolerances:
`scan measure-to-check spec.json --tol flatness_p2_p98=0.3 --out my_check.json` (flatness ->
a `flatness` check on the roi; `--tol step_height=19.5,20.5` -> a `height` check of region B
above region A's plane). Run it alone with `scan quick-check PRESET --check recipe --recipe
my_check.json`, or add it to the library (`scan recipe-add NAME --from my_check.json`).

**In a station configuration.** Two ways (`stations/README.md`):

- A single-job station config with `"job": {"type": "recipe", "recipe": "box_lid"}`: its one
  work state is verified by that recipe. On the operator page (`/quick`) and the command bus,
  selecting the configuration loads the recipe's preset (the station's own `preset` wins when
  set), START / TRIGGER run it, the outputs carry PASS / FAIL / ALARM and the recipe's result
  code. See `stations/example_recipe_box_lid.json`.
- A configuration with profiles and a flow: each profile lists its recipes under keys
  (`"recipes": {"presence": "tube_long_presence", ...}`) and each work state verifies with
  one key (`"verify": {"recipe": "presence"}`). A presence check also tells the cycle what
  the DLP **saw** (`details.present`), which is compared with what the line's part-present
  sensor says: the DLP confirms or corrects the sensor. See
  `stations/example_tube_line.json` (`tube_long_*` / `tube_short_*` here are its recipes).

## Example (the minimum)

```json
{
  "schema_version": 1,
  "name": "lid_present",
  "preset": "bench_9in_metric",
  "zones": [
    {"id": "lid", "shape": "rect", "units": "part_mm", "coords": [-40, -25, 40, 25],
     "checks": [
       {"type": "presence", "params": {"min_height_mm": 20}},
       {"type": "flatness", "params": {"tol_mm": 0.5, "min_height_mm": 20}}
     ]}
  ]
}
```

Everything else has a default: a fast two-axis scan, the reference plane = the dominant plane
of the scan (the fixture), the verdict projected for 5 s.

## Schema (version 1)

| key | meaning |
|---|---|
| `schema_version` | required, `1` |
| `name` | required, `[A-Za-z0-9_.-]`; a built-in's file name and a user recipe's folder name |
| `description` | text |
| `preset` | required: the rig preset the recipe pins - rig geometry + capture settings; its stereo calibration gives the mm (a station configuration's profile may scan with its own) |
| `scan.kind` | `fast` (HDMI fast scan, default), `metric` (both axes, the preset's full set), `trigger` (hardware-triggered flash pack: `scan.pack`, and `recipe-run --pattern-mode-set-by-human`) |
| `scan.axis` | `xy` (default; projected feedback and projector-px zones need both DMD axes) or `x` |
| `scan.zone` | light only these grid cells (`F10:J12`, `F10:J12@96`), an atlas region, or a painted zone set (`zoneset:<set>`, `zoneset:<set>:Z1,Z3`: `scan zone-paint`); null = whole field |
| `scan.average`, `scan.steps`, `scan.periods`, `scan.hold_s` | fast-scan settings (null = the engine's defaults) |
| `scan.codes` | `true`: read codes under bracketed DMD floods before the scan (for `code` checks) |
| `scan.rig_check` | `true`: the HOME rig check first; MOVED / UNKNOWN -> INVALID |
| `part.label` | the scan label (default the recipe name) |
| `part.frame.reference` | `dominant_plane` (RANSAC over the scan: the fixture / background) or `{"zone": "<id>"}` (a plane fitted in a datum zone, given in camera_px or projector_px) |
| `part.frame.plane_band_mm` | the plane's inlier band (default 1.0 mm) |
| `part.frame.origin_px` | `[x, y]`: the part frame's origin is the reference-plane point seen at this camera px (default: the optical axis) |
| `part.mock_scene` | MOCK only: what the simulated human places (`scene:<name>[:<variant>]`) |
| `golden.scan` | for `position` / `edge` position checks: a golden recorded by `scan recipe-golden` (its label - newest golden wins - or `latest` = the newest golden of THIS recipe at this geometry) or an exact metric scan id; never the scan being checked, never a plain metric scan with that label (an earlier run's test scan), never one made at other rig settings or with another stereo calibration (INVALID) |
| `zones[]` | see below |
| `feedback` | see below |
| `outputs.runs_dir` | where run folders go (relative to the recipe file; default `$SCANNER_RUNS_DIR`, else `<library>/runs`) |
| `outputs.result_codes` | register values for PASS / FAIL / INVALID / ABORTED / PULL (default 1 / 2 / 3 / 4 / 5; 0 = no result yet). PULL may be left out: it gets 5, or the next code the others leave free (a recipe numbered `{"PASS": 5, "FAIL": 6, "INVALID": 7, "ABORTED": 8}` gets PULL 9); no doubt (INVALID, ABORTED, PULL) and no FAIL may share the PASS code, and PULL needs its own (not 0, not FAIL's, INVALID's or ABORTED's) |
| `outputs.overlay` | write `overlay.png` (camera view with the zones and the verdict) |
| `pull.guard_band_pct` | the guard band: a PASSing value within this % of the tolerance from a limit is PULLed for quality inspection (two-sided: % of the half width from each limit; one-sided: % of the limit; `presence` expect present: % of the room above `min_area_pct`, up to 100 %). A value on its limit is always in the band. `0` (default) = off; `0 <= pct < 100`, finite. The built-in examples use 10 |
| `pull.invalid_reasons_to_reject` | doubt reasons this line REJECTS instead of pulling (default `[]`: every doubt is a PULL): any of `uncalibrated`, `stale_calibration`, `px_not_mm`, `too_few_points`, `zone_unresolved`, `golden`, `ambiguous_faces`, `incomplete`, `invalid`, `check_crashed`, `code_unreadable`, `rig_check`; `aborted`, `guard_band`, `refused`, `error`, `rig_unverified` (a mobile rig not verified in this session), `size_mismatch` (a part of known size reads the wrong size: the calibration or the lens, not the part) and `exposure` never reject, and a refused or crashed run is never rejected through this list |

**Part frame.** Heights are mm above the reference plane (+ toward the camera); u, v are mm in
that plane, u ~ camera right, v ~ camera down, from the origin. A `part_mm` zone holds the
measured points whose (u, v) falls in its shape: a part standing proud of the fixture is
still in its zone (3-D membership, no parallax).

**Zones.** `{"id", "label", "shape", "units", "coords", "checks", "datum", "feedback"}`

| shape | coords | units |
|---|---|---|
| `rect` | `[x0, y0, x1, y1]` (x1 > x0, y1 > y0; inclusive) | `camera_px`, `projector_px` (native DMD col, row) or `part_mm` |
| `circle` | `{"center": [x, y], "radius": r}` | same |
| `polygon` | `[[x, y], ...]` (>= 3 points) | same |
| `grid-cells` | `"F10:J12"` / `"F10:J12@96"` (the projected grid's cells) | `projector_px` |

**A zone painted on the part.** `{"id", "label", "ref": "zoneset:<set>/<zone id>", "checks",
"datum", "feedback"}` refers to a zone of a painted zone set (`scan zone-paint --name <set>`: the
operator draws it on the part with the projected mouse cursor; README, *Zones: paint them on
the part*). It gives no `shape` / `units` / `coords` of its own: it is in `projector_px`, and
its shape is the set's, read when the recipe is loaded - paint the set again and the recipe
looks at the new shape. The reference is checked when the recipe is loaded (the set and the
zone id must exist in the library) and bound when it runs: the set painted at the scan's
`geometry_id`, with the same geometry-bound rig settings, is used; if there is none (painted
for another rig placement only) the zone is unresolved, its checks are INVALID and the part
is PULLed. `scan zones <set> --recipe` prints a set as plain `projector_px` zones instead
(coordinates copied; `--by-ref` as references).

Units are always said: there is no default for `units` (except grid cells). `datum: true` =
a reference-only zone (no checks: e.g. the plane for `part.frame.reference`);
`feedback: false` = never drawn by the projector.

**Checks.** `{"type": "<check id>", "id": "<optional, unique in the zone>", "params": {...},
"enabled": true}` - built-ins `presence`, `position`, `height`, `flatness`, `defect`, `edge`,
`code`, plus plugins: [`docs/CHECKS.md`](../docs/CHECKS.md). Coverage is judged over the
zone's whole extent in its own units (camera px, DMD px or part-frame mm), so whatever of a
zone is off the image, occluded, in shadow or undecoded counts as unmeasured: `presence`
(`min_points`) calls a part absent only when even its unseen extent, all counted as part,
could not make it present; `defect` local (`min_valid_pct` 70, `max_gap_mm2`) and `flatness`
(`min_valid_pct` 70) never pass a partly measured zone, and a `defect` gap that can hold the
defect counts whether or not measured px enclose it. The run headline
`deviation_mm` / `threshold_mm` comes from the failing check, as a deviation (height: from
the band centre vs the half width).

**Feedback** (`feedback.*`, what the projector paints after the verdict):

| key | default | |
|---|---|---|
| `enabled` | true | needs `scan.axis` xy |
| `pass_colour`, `fail_colour`, `invalid_colour`, `pull_colour` | green, red, amber, amber | a name or `[b, g, r]` 0-255 (BGR, as the HDMI window draws) |
| `line_px` | 4 | outline width, DMD px |
| `fill_fail`, `fill_level` | true, 0.6 | failing zones filled at this fraction of the fail colour |
| `cross_px` | 24 | cross half-length at each fault point, DMD columns |
| `text`, `text_px` | true, 28 | "FAIL right" next to the zones, camera px tall |
| `hold_s` | 5 | seconds on the part (0..300); ABORT clears it, the web UI's next command too |
| `max_fit_px` | 2.0 | camera -> DMD consistency limit (local affine residual, DMD px): a zone above it is not drawn; a zone decoded too sparsely for the 16 px tile check (under half its px in checkable tiles) must pass the whole-zone fit instead; a FAIL / INVALID zone that cannot be drawn frames the whole DMD field in the verdict colour |
| `photo` | true | photograph the projected verdict (`feedback_photo.png`) |

The paint goes where the scan's own decoded correspondences say each camera px was lit
from, so it lands on the surfaces the camera saw, at any depth. It shows the disposition:
CONTINUE green; REJECT red at the failing zones with a cross at each definite fault; PULL an
amber frame round the whole part (every drawn zone) + "PULL", and amber on the zones that
caused it (no red, no cross: a doubt is not a defect).

**Pull** (`pull.*`, the disposition on top of the verdict):

```json
"pull": {"guard_band_pct": 10, "invalid_reasons_to_reject": []}
```

Each check lists the values it held to a limit in `tolerances` (`[{name, value, lo, hi}]`);
the guard band reads them (a plugin without `tolerances`: its `key` / `limit`, |value| <=
limit). A FAIL is a REJECT only when it is definite: a `code` check that read nothing of the
expected kind FAILs with `pull: "code_unreadable"` (PULL), a code read with the wrong text
is a REJECT.

## The run folder

`api.recipe_run` / `scan recipe-run` / `scan run` (job.type recipe) write a NEW folder:
`result.json` (the station run schema, 0.3: `zones`, `recipe`, `feedback`, `disposition`,
`pull_reasons` added), `run.log.jsonl` (run_start, scan, zone, verdict, feedback_shown /
feedback_cleared, run_end; `verdict` and `run_end` carry the disposition), `overlay.png`,
`feedback_dmd.png` (the DMD image), `feedback_photo.png`. Exit codes (the disposition):
0 CONTINUE, 1 REJECT, 5 PULL; 2 refused, 4 crashed, ABORTED 1 (all three PULL in
result.json).

Verdict: any check FAIL -> FAIL; else any check INVALID -> INVALID (never a pass: no stereo
calibration, a scan made under another calibration, too few points in a zone, no golden);
else PASS. Disposition: a definite FAIL anywhere -> REJECT (even with an INVALID zone);
else any doubt (an INVALID check, a crashed check, an unreadable code, the rig check MOVED /
UNKNOWN, ABORTED, a PASS in the guard band) -> PULL; else CONTINUE. The result code follows
the disposition: PASS only for CONTINUE, FAIL for REJECT, INVALID / ABORTED as such, PULL for
a PASS or FAIL in doubt.

## The built-in examples (all runnable in `--mock`)

| recipe | part | zones -> checks | mock scenes |
|---|---|---|---|
| `box_lid` | a box with its lid | left / right half: presence, height 40 +- 1.5 mm, flatness 0.5 mm (guard band 10 %, as every built-in) | `scene:box_lid` PASS, `:ajar` FAIL (right half red), `:missing` FAIL (both red) |
| `flat_plate` | a flat plate | top face: flatness 0.3 mm, local defects 0.3 mm; top edge: straightness 0.8 mm | `scene:plate` PASS, `:bump` FAIL, `:burr` FAIL (cross at the burr) |
| `cylinder_end_faces` | two cylinders, end faces toward the camera | each face: MF-CFAR dents >= 0.15 mm; right face height vs the left 0 +- 0.3 mm | `scene:endfaces` PASS, `:dent` FAIL, `:proud` FAIL |
| `ribbed_tube_alignment` | a ribbed tube | left / middle / right section: position and top-edge position vs the golden, 2 mm | `scene:tube` PASS, `:misaligned` FAIL (right section red) |

```bat
scan.bat --mock --library library_mock --preset bench_9in_metric calibrate-stereo --mock-poses 12
scan.bat --mock --library library_mock --mock-part scene:box_lid:ajar recipe-run box_lid
scan.bat --mock --library library_mock recipe-golden ribbed_tube_alignment
scan.bat --mock --library library_mock --mock-part scene:tube:misaligned recipe-run ribbed_tube_alignment
```

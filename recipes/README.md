# Recipes (work-state checks)

A recipe is one work-state check: which preset (profile) to scan with, where the zones are,
which checks run in each zone, and what the projector paints on the part after the verdict.
It is data (JSON, or YAML with PyYAML), versioned by `schema_version`, so the same engine
runs any part on any line with a DMD projector + a camera.

```
preset (profile) -> recipe -> zones -> checks -> verdict -> feedback (projected) -> outputs (I/O)
```

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
| run | `scan recipe-run NAME [--scan ID] [--no-feedback] [--hold S]` | `api.recipe_run(r, ...)` | verb `recipe_run`; work state (below) |

`--set` keys walk into sections and lists: `feedback.hold_s=10`,
`zones.0.checks.1.params.tol_mm=0.4`, `golden.scan=my_good_part`. A user recipe is validated
like a built-in one and its preset must exist; a built-in name is refused, an existing user
name too unless `--replace` (the old one moves to `trash/recipes/`). Delete refuses built-ins,
the selected recipe and a recipe a station config's `job.recipe` names (`--force` overrides).

**Work states.** A station config with `"job": {"type": "recipe", "recipe": "box_lid"}` IS
that recipe on the operator page (`/quick`) and the command bus: selecting it loads the
recipe's preset (the station's own `preset` wins when set), START / TRIGGER run it, the
outputs carry PASS / FAIL / ALARM and the recipe's result code. See
`stations/example_recipe_box_lid.json`.

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
| `preset` | required: the preset (profile) the recipe pins - rig geometry + capture settings; its stereo calibration gives the mm |
| `scan.kind` | `fast` (HDMI fast scan, default), `metric` (both axes, the preset's full set), `trigger` (hardware-triggered flash pack: `scan.pack`, and `recipe-run --pattern-mode-set-by-human`) |
| `scan.axis` | `xy` (default; projected feedback and projector-px zones need both DMD axes) or `x` |
| `scan.zone` | light only these grid cells (`F10:J12`, `F10:J12@96`) or an atlas region; null = whole field |
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
| `outputs.result_codes` | register values for PASS / FAIL / INVALID / ABORTED (default 1 / 2 / 3 / 4; 0 = no result yet) |
| `outputs.overlay` | write `overlay.png` (camera view with the zones and the verdict) |

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
| `pass_colour`, `fail_colour`, `invalid_colour` | green, red, amber | a name or `[b, g, r]` 0-255 (BGR, as the HDMI window draws) |
| `line_px` | 4 | outline width, DMD px |
| `fill_fail`, `fill_level` | true, 0.6 | failing zones filled at this fraction of the fail colour |
| `cross_px` | 24 | cross half-length at each fault point, DMD columns |
| `text`, `text_px` | true, 28 | "FAIL right" next to the zones, camera px tall |
| `hold_s` | 5 | seconds on the part (0..300); ABORT clears it, the web UI's next command too |
| `max_fit_px` | 2.0 | camera -> DMD consistency limit (local affine residual, DMD px): a zone above it is not drawn; a zone decoded too sparsely for the 16 px tile check (under half its px in checkable tiles) must pass the whole-zone fit instead; a FAIL / INVALID zone that cannot be drawn frames the whole DMD field in the verdict colour |
| `photo` | true | photograph the projected verdict (`feedback_photo.png`) |

The paint goes where the scan's own decoded correspondences say each camera px was lit
from, so it lands on the surfaces the camera saw, at any depth.

## The run folder

`api.recipe_run` / `scan recipe-run` / `scan run` (job.type recipe) write a NEW folder:
`result.json` (the station run schema, 0.2: `zones`, `recipe`, `feedback` added), 
`run.log.jsonl` (run_start, scan, zone, verdict, feedback_shown / feedback_cleared, run_end),
`overlay.png`, `feedback_dmd.png` (the DMD image), `feedback_photo.png`. Exit codes: 0 PASS,
1 FAIL, 2 refused, 4 crashed, 5 INVALID.

Verdict: any check FAIL -> FAIL; else any check INVALID -> INVALID (never a pass: no stereo
calibration, a scan made under another calibration, too few points in a zone, no golden);
else PASS.

## The built-in examples (all runnable in `--mock`)

| recipe | part | zones -> checks | mock scenes |
|---|---|---|---|
| `box_lid` | a box with its lid | left / right half: presence, height 40 +- 1.5 mm, flatness 0.5 mm | `scene:box_lid` PASS, `:ajar` FAIL (right half red), `:missing` FAIL (both red) |
| `flat_plate` | a flat plate | top face: flatness 0.3 mm, local defects 0.3 mm; top edge: straightness 0.8 mm | `scene:plate` PASS, `:bump` FAIL, `:burr` FAIL (cross at the burr) |
| `cylinder_end_faces` | two cylinders, end faces toward the camera | each face: MF-CFAR dents >= 0.15 mm; right face height vs the left 0 +- 0.3 mm | `scene:endfaces` PASS, `:dent` FAIL, `:proud` FAIL |
| `ribbed_tube_alignment` | a ribbed tube | left / middle / right section: position and top-edge position vs the golden, 2 mm | `scene:tube` PASS, `:misaligned` FAIL (right section red) |

```bat
scan.bat --mock --library library_mock --preset bench_9in_metric calibrate-stereo --mock-poses 12
scan.bat --mock --library library_mock --mock-part scene:box_lid:ajar recipe-run box_lid
scan.bat --mock --library library_mock recipe-golden ribbed_tube_alignment
scan.bat --mock --library library_mock --mock-part scene:tube:misaligned recipe-run ribbed_tube_alignment
```

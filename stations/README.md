# Station configs

`scan run <config>` runs one job from a station config and writes a new run folder:

```
<runs_dir>/<UTC timestamp>_<station_id>_<8 hex>/
    result.json      machine-readable result (schema below)
    run.log.jsonl    one JSON event per line; every line has run_id
```

```bat
scan.bat --mock run stations\example_cell_station.json
scan.bat run stations\my_station.json --operator-confirmed
```

Exit codes: `0` PASS, `1` FAIL, `2` refused (bad config, rule, rig busy, no base,
real rig without operator confirmation), `4` crashed, `5` INVALID. A refused or
crashed run that got as far as creating its folder still writes `result.json`
with `"result": "INVALID"` and the reason.

Metric job (mm from the stereo calibration, no base): `example_metric_check.json`.
Without a stereo calibration at the preset's geometry the run is INVALID and says how to
calibrate (`charuco`, `calib-pose` x 12-20, `calibrate-stereo --square MM`).

The verdict comes from out-of-tolerance clusters, never from a statistic: every camera px
whose |deviation| exceeds `threshold_mm` is grouped with the others within 3 px of it (a
patch of small pits is ONE cluster), a cluster smaller than `job.min_defect_px` (default
5 px, at most 50) is dropped as an isolated stray triangulated point, and any other cluster
is a defect (FAIL), however small its share of the field. Dropped stray px are counted: more
than `job.max_outlier_pct` (default 0.1 %, at most 5 %) of the measured px (golden: of both
scans together, or of either scan's own once they are two defect clusters' worth) is INVALID
(noisy data or a pitted surface), never a PASS; so are dropped px that are dense in one
place (5 or more, and 2 % or more of a 21 px square: pits a few px apart, a noisy patch). `deviation_mm.max` is the robust max (the
largest |deviation| outside the dropped clusters); `deviation_mm` also carries `rms`,
`method` (`plane_fit` / `compare_metric_golden`), `n_points`, `n_defects`, `largest_defect`
(`side`, `n_px`, `area_mm2`, `max_mm`, `bbox_px`), `outlier_px` / `outlier_clusters` /
`outlier_pct` (what was dropped; golden: the worse side) and `dense_strays` (where dropped
px were dense, else null), and `outputs.deviation_map` is `deviation.png` in the run
folder.

- `plane`: a RANSAC plane in `roi`. Fewer decoded points than `job.min_valid_pct` (default
  30 %) of the roi is INVALID. Plane mode measures whatever surface is in the roi (the empty
  wall is flat too); `job.plane_range_mm: [near, far]` adds a presence check: the plane must
  lie that far from the camera, else INVALID (part absent or misplaced).
- `golden`: every foreground component in the roi (not only the largest) is compared with
  the golden in BOTH directions: test -> golden finds deviating and extra material, golden ->
  test finds missing material (`side` `test` / `golden`). The background (the wall or
  fixture plane behind the part) is the datum: the part is placed onto the golden only by a
  turn and a shift ALONG that plane (from its surface and its outline against the
  background), so a good part re-placed a little PASSes, while a part or one component that
  sits higher, lower or tilted (a cell 0.8 mm proud, a thinner part, a lid ajar) keeps that
  offset in the deviation. A background plane more than 2 deg / 5 mm off the golden's (the
  rig or the fixture moved) is INVALID. A point past the other part's outline against the
  background (extra or missing material, a component moved sideways, a part too long or
  short) counts by how far it lies beyond that outline along its outward normal, in full -
  no sample spacing is taken off it; an outline is only as sharp as a camera px (~0.35 mm on
  the bench), so in-plane edges are not judged much finer than that, and a threshold near
  it FAILs re-placed good parts at their edges. Only well-decoded px seen within 75 deg of
  face-on are judged (isolated decodes in shadows or glints, and grazing side walls, are
  not; the face-on test uses the surface around a px, so a pit or a spike is still judged),
  the same on both scans; a foreground region (decoded px joined across small gaps) of 50
  px or more with under 20 % of them judged (decoded too sparsely: a saturating or dark
  face) is INVALID, never left out unseen. Less than `job.min_coverage_pct` (default 95 %)
  of either surface within the threshold, without a defect cluster, or a placement that did
  not converge, is INVALID. `job.golden` must name the golden metric scan (a GOOD part
  scanned once) by id or label: `latest` and this station's `part.label` are refused, since
  every run saves its own test scan under that label. The comparison report (datum, placement, defects, dropped px)
  goes to `outputs.report`.

`result` may also be `ABORTED` (an operator ABORT from the operator page / command bus):
the run stopped before its next capture; exit 1.

Recipe job (`example_recipe_box_lid.json`): `"job": {"type": "recipe", "recipe": "box_lid"}`
runs a recipe (`recipes/README.md`: its preset, scan, zones, checks, projected feedback);
`preset: null` = the recipe's own preset. The work state then IS the recipe: result.json
adds `zones` (per zone: verdict, checks with values / limits / fault points), `recipe` (name,
preset, part frame, result code) and `feedback` (what the projector painted); the run folder
also holds `overlay.png`, `feedback_dmd.png` and `feedback_photo.png`. After result.json is
written the verdict is projected on the part (`job.feedback: false` or `run --no-feedback`
skip it; the operator page shows it as its own job, cleared by the next command).
`mock_part` (MOCK only) is what the simulated human places, e.g. `scene:box_lid:ajar`
(`--mock-part` wins).

The operator page (`/quick`) lists every `*.json` here as a work state (`io_*.json` are
command I/O mappings, see `io_example.json`, not stations). Work states are numbered 1..n in
file-name order, invalid configs included: an invalid one keeps its number and is shown as
unusable, so a PLC work-state number never shifts because another config broke; adding or
removing files while idle does renumber. A config with `preset: null` that is not a recipe
runs the preset currently loaded in the web UI (refused if none is loaded); when the
library's selected preset no longer exists (deleted, renamed) the web UI loads none in its
place (and shows why), and such a run is refused (INVALID, naming it) until a preset is
selected in Setup, whatever preset is loaded instead.

Copy `example_cell_station.json` (or the `.yaml`, which needs PyYAML) and change
`station_id`, `preset`, the threshold and the job. The station id comes only from
this file.

## Keys

| Key | Meaning |
|---|---|
| `station_id` | required; letters, digits, `_ . -` |
| `preset` | rig preset (`--preset` on the command line wins); `null` = the recipe's preset (recipe jobs), else the preset selected (`preset-select`) in the library the run uses (`library` / `mock_library`): a selection that no longer exists is refused, never replaced by a default. A preset file path naming a device (`CON`, `CONIN$`, a `\\.\` prefix, a trailing dot or space, a `:`) is refused |
| `library`, `mock_library` | library root for real / `--mock` runs; `null` = `--library`, `$SWISS_SCANNER_LIBRARY` or `<repo>/library` |
| `runs_dir` | where run folders go; `$SCANNER_RUNS_DIR` overrides |
| `operator_confirmed` | a human confirmed the physical setup; the real rig refuses without it (or `--operator-confirmed`). `true` / `false` only: a quoted `"false"` is refused |
| `require_mm` | `true`: a px-only result (no calibration at this geometry) is INVALID (`true` / `false` only) |
| `threshold_mm` | FAIL threshold in mm (a positive, finite number); `null` = the preset's `tolerance.ch_fail_mm` (metric jobs: required, INVALID without it) |
| `job.type` | `inspect` (defect blobs vs the base) or `compare` (vs `job.golden`): base-relief depth calibration; `metric`: stereo calibration (`calibrate-stereo`); `recipe`: a recipe (stereo calibration) |
| `job.recipe`, `job.feedback` | recipe: a recipe name (`scan recipe-list`) or a recipe file (relative to this config); `feedback` true / false / null (null: the recipe's `feedback.enabled`) |
| `mock_part` | MOCK only: what the simulated human places (as `--mock-part`, e.g. `scene:tube:misaligned`) |
| `job.scan`, `job.axis`, `job.steps`, `job.periods` | metric: `fast` (`scan_fast`, axis `xy` default) or `metric` (`scan_metric`) |
| `job.measure`, `job.roi`, `job.golden` | metric: `plane` (flatness in `roi` = `[x0, y0, x1, y1]` camera px, `null` = whole field) or `golden` (vs a golden metric scan id / label; never `latest` or this station's `part.label`) |
| `job.min_defect_px`, `job.max_outlier_pct`, `job.min_valid_pct`, `job.min_coverage_pct`, `job.plane_range_mm` | metric: smallest out-of-tolerance cluster that FAILs (default 5 camera px, 1..50; smaller isolated ones are stray points); share of the measured px that may be dropped as stray points before the run is INVALID (default 0.1 %, 0..5); plane: decoded share of the roi below which the run is INVALID (default 30 %), optional `[near, far]` plane distance from the camera (mm, presence check); golden: share of each surface within the threshold below which the run is INVALID (default 95 %) |
| `job.base` | `latest` usable base, or `capture` a new one (the scene must be empty). In `--mock`, a missing base is captured automatically. |
| `job.zone`, `job.average`, `job.n_sigma`, `job.min_area_px` | passed to the engine verbs |
| `part.role`, `part.label` | `golden` / `test`, and the scan label |

## result.json

```json
{
  "schema_version": "0.2",
  "run_id": "20261001T120000Z_example_cell_station_1a2b3c4d",
  "run_folder": "runs/20261001T120000Z_example_cell_station_1a2b3c4d",
  "mode": "MOCK",
  "station": "example_cell_station",
  "operator_confirmed": false,
  "job": {"type": "inspect", "base": "latest", "zone": null, "config": "example_cell_station.json"},
  "calibration": {"calibration_id": "...", "same_geometry": true,
                  "reprojection_error_px": {"camera": null, "projector": null}},
  "capture": {"preset": "...", "geometry_id": "...", "patterns": "...", "exposure_us": 16667,
              "average": 8, "zone": null},
  "part": {"role": "test", "label": "part", "scan_id": "..."},
  "deviation_mm": {"max": 0.0, "mean": null, "rms": null, "method": "relief_inspect",
                   "n_points": null, "n_defects": null, "largest_defect": null,
                   "outlier_px": null, "outlier_clusters": null, "outlier_pct": null,
                   "dense_strays": null},
  "deviation_units": "mm",
  "threshold_mm": 0.7,
  "result": "PASS",
  "invalid_reason": null,
  "engine_verdict": "PASS",
  "warnings": [],
  "outputs": {"mesh": [], "deviation_map": "...png", "report": "...json", "log": "run.log.jsonl"},
  "recipe": null,
  "zones": [],
  "feedback": null,
  "software": {"package_version": "0.1.0", "git_commit": null},
  "started": "...", "finished": "..."
}
```

Schema 0.2 added `recipe`, `zones` and `feedback` (null / [] for the other job types), and
`deviation_mm.n_defects` / `largest_defect` / `outlier_px` / `outlier_clusters` /
`outlier_pct` / `dense_strays` (metric jobs; null for the others); every 0.1 key is
unchanged.

`INVALID` means the numbers cannot be trusted (no mm calibration at this geometry,
an INDETERMINATE engine verdict, a refused or crashed run). Never report an
INVALID run as a pass or a fail.

A fresh `--mock` library has no calibration, so the first mock run is INVALID.
Calibrate the mock geometry once with the validation object, then runs give
PASS / FAIL:

```bat
scan.bat --mock --library library_mock base
scan.bat --mock --library library_mock scan cal --known 106.95,56.5,21.9
scan.bat --mock --library library_mock calibrate cal
scan.bat --mock run stations\example_cell_station.json
```

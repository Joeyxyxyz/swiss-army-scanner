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
calibrate (`charuco`, `calib-pose` x 12-20, `calibrate-stereo --square MM`). Max deviation is
the 99.9th percentile of |deviation| (robust to stray triangulated points); `deviation_mm`
also carries `rms`, `method` (`plane_fit` / `compare_metric_golden`) and `n_points`, and
`outputs.deviation_map` is `deviation.png` in the run folder.

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
command I/O mappings, see `io_example.json`, not stations).

Copy `example_cell_station.json` (or the `.yaml`, which needs PyYAML) and change
`station_id`, `preset`, the threshold and the job. The station id comes only from
this file.

## Keys

| Key | Meaning |
|---|---|
| `station_id` | required; letters, digits, `_ . -` |
| `preset` | rig preset (`--preset` on the command line wins) |
| `library`, `mock_library` | library root for real / `--mock` runs; `null` = `--library`, `$SWISS_SCANNER_LIBRARY` or `<repo>/library` |
| `runs_dir` | where run folders go; `$SCANNER_RUNS_DIR` overrides |
| `operator_confirmed` | a human confirmed the physical setup; the real rig refuses without it (or `--operator-confirmed`) |
| `require_mm` | `true`: a px-only result (no calibration at this geometry) is INVALID |
| `threshold_mm` | FAIL threshold in mm; `null` = the preset's `tolerance.ch_fail_mm` (metric jobs: required, INVALID without it) |
| `job.type` | `inspect` (defect blobs vs the base) or `compare` (vs `job.golden`): base-relief depth calibration; `metric`: stereo calibration (`calibrate-stereo`); `recipe`: a recipe (stereo calibration) |
| `job.recipe`, `job.feedback` | recipe: a recipe name (`scan recipe-list`) or a recipe file (relative to this config); `feedback` true / false / null (null: the recipe's `feedback.enabled`) |
| `mock_part` | MOCK only: what the simulated human places (as `--mock-part`, e.g. `scene:tube:misaligned`) |
| `job.scan`, `job.axis`, `job.steps`, `job.periods` | metric: `fast` (`scan_fast`, axis `xy` default) or `metric` (`scan_metric`) |
| `job.measure`, `job.roi`, `job.golden` | metric: `plane` (flatness in `roi` = `[x0, y0, x1, y1]` camera px, `null` = whole field) or `golden` (vs a golden metric scan id / label / `latest`) |
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
  "deviation_mm": {"max": 0.0, "mean": null, "rms": null, "method": "relief_inspect"},
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

Schema 0.2 added `recipe`, `zones` and `feedback` (null / [] for the other job types); every
0.1 key is unchanged.

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

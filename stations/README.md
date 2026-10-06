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
| `threshold_mm` | FAIL threshold in mm; `null` = the preset's `tolerance.ch_fail_mm` |
| `job.type` | `inspect` (defect blobs vs the base) or `compare` (vs `job.golden`) |
| `job.base` | `latest` usable base, or `capture` a new one (the scene must be empty). In `--mock`, a missing base is captured automatically. |
| `job.zone`, `job.average`, `job.n_sigma`, `job.min_area_px` | passed to the engine verbs |
| `part.role`, `part.label` | `golden` / `test`, and the scan label |

## result.json

```json
{
  "schema_version": "0.1",
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
  "part": {"role": "test", "label": "cell", "scan_id": "..."},
  "deviation_mm": {"max": 0.0, "mean": null, "rms": null, "method": "relief_inspect"},
  "deviation_units": "mm",
  "threshold_mm": 0.7,
  "result": "PASS",
  "invalid_reason": null,
  "engine_verdict": "PASS",
  "warnings": [],
  "outputs": {"mesh": [], "deviation_map": "...png", "report": "...json", "log": "run.log.jsonl"},
  "software": {"package_version": "0.1.0", "git_commit": null},
  "started": "...", "finished": "..."
}
```

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

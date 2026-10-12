# Station configurations

A file here is a **station configuration**: what one station is and knows. It is **not** a
work state.

| word | what it is |
|---|---|
| **Station configuration / parameters** | the station's parameters (rig preset, library, run folders, the line's sensor inputs, defaults) and the **profile library set** it can call on |
| **Profile** | one part variant of that set (long tube / short tube): what the part **looks like** - its recipes (zones + checks), preset, named zones, golden. Selected by the operator, a PLC or HTTP (`SELECT_PROFILE`) |
| **Work state** | one **step** of the flow: a claim of the line's own sensors, the DLP check that confirms or corrects it, the permission that lets the line continue |
| **Check** | what the DLP measures to verify a work state (`recipes/README.md`, `docs/CHECKS.md`) |
| **Disposition** | CONTINUE / REJECT / PULL: what happens to the part (`README.md` *Dispositions*) |

Two forms load:

- **profiles + flow** (`example_tube_line.json`): the configuration, its profile library set
  and its work states. Below.
- **a single job** (`example_cell_station.json`, `example_metric_check.json`,
  `example_recipe_box_lid.json`): the older form, still read and run exactly as before. It is
  a configuration with one implicit profile (`default`) and a one-step flow (its job). See
  *Single-job station configs*.

`scan run <config>` runs either; `scan flow-show <config>` prints and validates either.
`io_*.json` files here are command I/O mappings (`io_example.json`), not stations.

A station configuration is the **Station** operating mode (`README.md` *Operating modes*):
the layer on top of a triangulation preset that adds profiles, work states, sensor claims and
permissions. The other two modes need no file here: **Measure** (`scan measure-run`) only
reports numbers, and **Quick check** (`scan quick-check PRESET --check flatness|golden|recipe`)
generates its own one-job config - a single-job station config marked `"operating_mode":
"quick_check"`, kept in its run folder as `quick_check.json` and runnable again with `scan
run`. A configuration or profile that pins a **mobile** preset (`rig.mount`; or a preset at
the geometry of one) ends its runs and work states PULL (`rig_unverified`: "mobile rig not
verified: run rig-check / re-home / calibrate" - never a reject, whatever the recipe's
`invalid_reasons_to_reject` lists) until the rig was verified in the current session; a fixed
rig is not gated in Station mode.

## Profiles and work states

```json
{
  "schema_version": 2,
  "station_id": "example_tube_line",
  "parameters": {
    "preset": "bench_9in_metric_bright",
    "library": null, "mock_library": "../library_mock", "runs_dir": "../runs",
    "default_profile": "long_tube",
    "io": {"inputs": {"part_present": {"description": "the seat's part-present sensor"}}},
    "defaults": {"retries": 1, "timeout_s": null, "trigger": "step"}
  },
  "profiles": {
    "long_tube":  {"title": "Long tube (60 mm)",
                   "recipes": {"presence": "tube_long_presence", "alignment": "tube_long_alignment",
                               "end_face": "tube_long_end_face"}},
    "short_tube": {"title": "Short tube (38 mm)",
                   "recipes": {"presence": "tube_short_presence", "alignment": "tube_short_alignment",
                               "end_face": "tube_short_end_face"}}
  },
  "flow": "../flows/example_tube_line.json"
}
```

### `parameters`

| Key | Meaning |
|---|---|
| `preset` | the rig preset every profile scans with (a profile's own `preset` wins; `null`: each recipe's pinned preset) |
| `library`, `mock_library`, `runs_dir`, `operator_confirmed` | as in a single-job config (below); paths are relative to this file |
| `default_profile` | the profile selected when the configuration is (default: the first) |
| `io.inputs` | the sensor signals of the line the work states may `expect`: `{"<signal>": {"description", "source"}}` (or a list of names). A work state naming another signal is refused, and so is a claim for a name no configuration declares. The claims arrive at run time: `POST /api/io/inputs {"<signal>": true}`, `scan run --input <signal>=true`, an input backend's `signals` map. **A claim belongs to one cycle**: every claim is forgotten when a cycle ends (and on ABORT, HOME, FAULT, a configuration / profile change), so the line reports its sensors for every part |
| `io.max_age_s` | optional: the oldest a claim may be (seconds) when a work state's scan starts; an older one is unknown (DOUBT). `null` (default): no limit beyond "reported for this cycle" |
| `defaults` | `retries` (1), `timeout_s` (null), `trigger` (`step`) for work states that do not say |

### `profiles` (the profile library set)

One entry per part variant. Every profile must be able to verify every work state.

| Key | Meaning |
|---|---|
| `title`, `description` | shown on the operator page |
| `preset` | this profile's rig preset (`null`: `parameters.preset`, else each recipe's own) |
| `recipes` | `{"<key>": "<recipe>"}`: what this part looks like, under the keys the flow's work states verify with. A recipe is a name (`scan recipe-list`), a recipe file (relative to this config) or a recipe mapping |
| `zones`, `scan`, `part`, `golden`, `feedback`, `pull` | for work states with **inline checks**: the profile's named zones `{"<id>": {shape, units, coords}}` and the recipe sections (as in a recipe) they are scanned with |
| `mock_scene` | MOCK only: the good part of this profile (`scene:<name>`) |

### `flow` (the work states)

A list of steps, `{"steps": [...]}`, or a file relative to this config (`flows/README.md`).
Each step is one work state:

| Key | Meaning |
|---|---|
| `id`, `title` | the work state ("Tube loaded?") |
| `expects` | `{"<signal>": true \| false}`: what the line's sensors claim at this step (may be empty: the DLP alone decides) |
| `verify` | the DLP check: `{"recipe": "<key of the profile's recipes>"}`, or inline `{"checks": [{"zone": "<profile zone>", "type": "presence", "params": {...}}]}`. `"observe": {"<signal>": "<zone>/<check>"}` names the check whose answer IS that signal (`{"zone", "check", "invert": true}` flips it). **Required for every expected signal**: nothing is guessed - a signal no check was named for is refused (a door switch is never "confirmed" by a part-presence check because it was the only one) |
| `permit` | the permission granted when the work state is CONFIRMED: bus output `PERMIT_<NAME>` |
| `on_confirm` | `next` (default), `done`, or a step id |
| `on_mismatch`, `on_not_met`, `on_defect`, `on_doubt` | action lists (below). `on_not_met` defaults to the `on_mismatch` list, `on_doubt` to `kickout PULL` (a doubt list takes no `goto` and no `kickout REJECT`) |
| `retries` | re-scans the `retry` action / the RETRY command may use (default 1) |
| `timeout_s` | the longest the work state may wait for STEP, or for an ACK (each wait counted from its start; the scan itself is not counted); then its cycle ends PULL (`null`: no limit) |
| `trigger` | `step` (verified on STEP / TRIGGER) or `auto` (verified as soon as it is the current one) |
| `required`, `allow_skip` | `required: false` work states do not count for CONTINUE and may `allow_skip` (SKIP never grants the permission). At least one work state must be required. One left in DOUBT or CORRECTED is never passed over or skipped, required or not |
| `revoke` | permissions of other work states dropped when this one is not confirmed (a list, or `"all"`) |

**Outcomes.** Per expected signal three values meet: what the work state expects, what the
sensor says, what the DLP saw.

| outcome | when | permission | actions |
|---|---|---|---|
| `CONFIRMED` | sensor and DLP agree, on what the work state expects, and every check passes with no doubt | granted | `on_confirm` |
| `CORRECTED` | the DLP disagrees with a sensor: `sensor_mismatch` `[{signal, sensor, dlp, expected}]` | withheld | `on_mismatch` |
| `NOT_MET` | sensor and DLP agree, on something the work state does not expect | withheld | `on_not_met` |
| `DEFECT` | the part is there, a check definitely fails | withheld | `on_defect` |
| `DOUBT` | INVALID / unmeasured / a guard band / a sensor that never reported or changed during the scan / a crashed check | withheld | `on_doubt` |

**Actions** (a short text, or a mapping; `"when_dlp": {"<signal>": true}` limits one to what
the DLP saw):

| action | what it does |
|---|---|
| `project_fault` | paints the fault / the missing-part zone on the part (the recipe's projected feedback). Recorded as `requested` (the cycle asked for it; `scan run` adds `shown`: what the projector did); nothing is painted when every check passed (`why`) |
| `prompt <message>` | a message to the operator / system (`PROMPT` on, the banner on `/quick`); the list waits for **ACK** |
| `hold` | waits for ACK (the prompt before it, else its own), then verifies the work state again |
| `retry` | verifies again at once, up to `retries`; then the list goes on |
| `kickout REJECT\|PULL [reason]` | ends the cycle with that disposition (REJECT only on a definite finding: under DOUBT it is a PULL). The cycle's `kickout` records the reason and, as `detail`, what the DLP found |
| `goto <step id>` | makes another work state the current one (never out of a DOUBT). A work state reached by jumping ahead grants no permission while a required work state before it is unconfirmed |

A required work state whose list ends without confirming it ends the cycle **PULL** (fail
closed), and so does any work state left in DOUBT or CORRECTED - required or not: the line
never continues past an open question. A work state that is not required may be left
unconfirmed only on a definite finding the sensor agrees with (NOT_MET / DEFECT): the flow
goes on, its permission stays withheld, `cycle.unconfirmed` lists it. The cycle is
**CONTINUE only when every required work state was CONFIRMED** (and at least one is).
`KICKOUT` is on with FAIL or PULL: every cycle that does not CONTINUE says in `cycle.kickout`
where and why its part leaves the line, kickout action or not.

**A sensor says a part is there when it should not be.** The same three values, with
`expects` false: a work state that needs the seat EMPTY, verified by a presence check that
expects `absent` (here inline, on a zone of the profile):

```json
{"id": "clear", "title": "Seat clear?",
 "expects": {"part_present": false},
 "verify": {"checks": [{"zone": "seat", "type": "presence",
                        "params": {"expect": "absent", "min_height_mm": 10}}],
            "observe": {"part_present": "seat/presence"}},
 "permit": "LOAD_OK",
 "on_mismatch": [
   {"prompt": "the sensor says a part is there, the seat is empty: check the sensor",
    "when_dlp": {"part_present": false}},
   {"prompt": "a part is on the seat and the sensor missed it: take it off and check the sensor",
    "when_dlp": {"part_present": true}},
   "hold"],
 "on_not_met": ["project_fault", "prompt a part is on the seat: take it off", "hold"]}
```

A sensor mismatch has two directions; `when_dlp` gives each its own prompt (one prompt for
both would tell the operator the seat is empty while a part sits on it).

| the sensor says | the DLP says | outcome | then |
|---|---|---|---|
| no part | no part | `CONFIRMED` | `LOAD_OK` granted |
| a part | no part | `CORRECTED` (`sensor_mismatch`) | the sensor is wrong: its prompt, hold |
| a part | a part | `NOT_MET`: it really is there | the unexpected part is painted on the seat, the operator is prompted to take it off (or `kickout PULL`) |
| no part | a part | `CORRECTED` (`sensor_mismatch`) | the sensor missed it: the other prompt, hold |

**Permissions.** `PERMIT_<NAME>` is true only while its work state is CONFIRMED in the
current cycle and every required work state before it is. All drop when a new cycle starts,
when the cycle ends anything but CONTINUE, on ABORT, FAULT, HOME, ARM, DISARM, a
configuration / profile change, a flow edited on disk - and on a START / TRIGGER that starts
no cycle (refused, or ignored because the station is not armed): that part was announced and
not checked, so the last cycle's permissions, its PASS and its work-state registers never
stand in for it. A permission of a configuration that is no longer selected stays published
as false. Nothing grants one on a doubt.

**The cycle folder** (`<runs_dir>/<UTC timestamp>_<station_id>_<8 hex>/`): `result.json`
(the run schema below, 0.4: `station_config`, `profile`, `steps`, `permissions`, `cycle`),
`run.log.jsonl` (cycle_start, work_state, prompt, ack, permission, cycle_end) and
`steps/<one run folder per DLP check>` (each a normal recipe run: result.json, overlay,
fault image). One step:

```json
{"id": "loaded", "title": "Tube loaded?", "index": 1, "required": true,
 "expects": {"part_present": true}, "sensor_inputs": {"part_present": true},
 "dlp_result": {"result": "FAIL", "disposition": "REJECT", "observed": {"part_present": false},
                "run_id": "...", "recipe": "tube_long_presence", "zones": [...]},
 "outcome": "CORRECTED",
 "sensor_mismatch": [{"signal": "part_present", "sensor": true, "dlp": false, "expected": true}],
 "permit": "LOAD_DONE", "permission_granted": false, "permission_note": null,
 "actions": [{"action": "project_fault", "requested": true},
             {"action": "prompt", "message": "no tube found: load or clear the sensor",
              "acknowledged": true}],
 "attempts": 1, "disposition": null, "message": "sensor part_present says true, the DLP says false"}
```

`permission_granted` says whether the permission still stood when the cycle ended (the same
as `permissions`); `permission_note` says why a CONFIRMED work state's permission was dropped
or withheld. `cycle` holds how it ended: `ended_by`, `kickout` (`{step, disposition, reason,
detail, by}`; null only for CONTINUE), `sensor_mismatch` (a sensor and the DLP disagreed at
any verify, also one put right since), `sensor_mismatches` (each one: step, attempt, signal,
both values), `unconfirmed` (the work states that are not CONFIRMED).

**Running it.**

```bat
scan.bat flow-show stations\example_tube_line.json
scan.bat --mock run stations\example_tube_line.json --profile short_tube
scan.bat --mock run stations\example_tube_line.json --mock-case sensor_true_part_absent
scan.bat run stations\my_line.json --profile long --input part_present=true --operator-confirmed
```

`scan run` walks one whole cycle in line: `--input SIGNAL=true|false` are the sensor claims
(a work state that expects a signal nobody gave is a DOUBT); a prompt nobody acknowledges
ends the cycle PULL (`--acknowledge` answers every prompt, `--interactive` asks; a prompt is
answered at most `retries` + 1 times per work state, then the unattended cycle ends PULL
rather than acknowledge for ever). Exit codes:
0 CONTINUE, 1 REJECT, 5 PULL, 2 refused. The operator page (`/quick`) and a PLC walk the same
cycle step by step through the command bus: START, STEP, ACK, `POST /api/io/inputs`.
`flow-show` prints, per work state, the check that observes each sensor signal.

**MOCK.** `mock.cases` name the simulated line: the scene the simulated human places (the
profile's `mock_scene`, its `:<variant>`, an explicit `scene`) and what the simulated sensors
say; `--mock-case NAME`, or the case selector on `/quick`. The simulated sensors report for
every cycle, as a line's gateway must; a signal posted to `POST /api/io/inputs` is the line's
own from then on (the simulated sensor of that name goes quiet until a case is chosen
again), so a gateway tested against the mock server has to report for every part too. The
example ships `all_good`,
`sensor_true_part_absent` (CORRECTED), `sensor_false_part_present` (CORRECTED), `misaligned`
(DEFECT at *Tube aligned?*) and `dented` (DEFECT at *End face OK?*, kicked out). Once per mock
library:

```bat
scan.bat --mock --library library_mock --preset bench_9in_metric_bright calibrate-stereo --mock-poses 12
scan.bat --mock --library library_mock recipe-golden tube_long_alignment
scan.bat --mock --library library_mock recipe-golden tube_short_alignment
```

Built and mock-tested (`tests/test_flow_mock.py`, `tests/test_flow_ui_mock.py`); not yet run on the real rig. Hardware
input backends are placeholders: on a line, the sensor claims come in over HTTP from a
gateway until one is implemented.

## Single-job station configs

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

Exit codes (the disposition, `README.md` *Dispositions*): `0` CONTINUE (PASS), `1`
REJECT (FAIL), `5` PULL (quality inspection: INVALID, or a metric PASS inside
`job.guard_band_pct`), `2` refused (bad config, rule, rig busy, no base, real rig without
operator confirmation), `4` crashed. A refused or crashed run that got as far as creating
its folder still writes `result.json` with `"result": "INVALID"`, the reason and
`"disposition": "PULL"`.

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
the run stopped before its next capture; exit 1, disposition PULL.

Guard band (metric jobs): `job.guard_band_pct` (default 0 = off; `0 <= pct < 100`) pulls a
PASS whose `deviation_mm.max` is within that % of `threshold_mm` (e.g. 10: a PASS above
0.45 mm against 0.5 mm is PULL, exit 5, `pull_reasons` `guard_band`). Other job types refuse
the key (recipe jobs set `pull.guard_band_pct` in the recipe). A golden FAIL whose placement
did not converge (or whose surfaces disagree below `job.min_coverage_pct` with no defect
cluster to account for it) stays `FAIL` but is PULL (exit 5, `incomplete`): never a reject
on numbers the job marks untrusted.

Recipe job (`example_recipe_box_lid.json`): `"job": {"type": "recipe", "recipe": "box_lid"}`
runs a recipe (`recipes/README.md`: its preset, scan, zones, checks, projected feedback);
`preset: null` = the recipe's own preset. The config's one work state is verified by that
recipe: result.json
adds `zones` (per zone: verdict, checks with values / limits / fault points), `recipe` (name,
preset, part frame, result code) and `feedback` (what the projector painted); the run folder
also holds `overlay.png`, `feedback_dmd.png` and `feedback_photo.png`. After result.json is
written the verdict is projected on the part (`job.feedback: false` or `run --no-feedback`
skip it; the operator page shows it as its own job, cleared by the next command).
`mock_part` (MOCK only) is what the simulated human places, e.g. `scene:box_lid:ajar`
(`--mock-part` wins).

The operator page (`/quick`) lists every `*.json` here as a station configuration
(`io_*.json` are command I/O mappings, see `io_example.json`, not stations). Configurations
are numbered 1..n in file-name order, invalid configs included: an invalid one keeps its
number and is shown as unusable, so a PLC's configuration number (the `station_config`
register) never shifts because another config broke; adding or removing files while idle
does renumber. A config with `preset: null` that is not a recipe
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
| `job.guard_band_pct` | metric: a PASS within this % of `threshold_mm` is PULLed for quality inspection (default 0 = off; `0 <= pct < 100`); refused for other job types |
| `job.min_defect_px`, `job.max_outlier_pct`, `job.min_valid_pct`, `job.min_coverage_pct`, `job.plane_range_mm` | metric: smallest out-of-tolerance cluster that FAILs (default 5 camera px, 1..50; smaller isolated ones are stray points); share of the measured px that may be dropped as stray points before the run is INVALID (default 0.1 %, 0..5); plane: decoded share of the roi below which the run is INVALID (default 30 %), optional `[near, far]` plane distance from the camera (mm, presence check); golden: share of each surface within the threshold below which the run is INVALID (default 95 %) |
| `job.base` | `latest` usable base, or `capture` a new one (the scene must be empty). In `--mock`, a missing base is captured automatically. |
| `job.zone`, `job.average`, `job.n_sigma`, `job.min_area_px` | passed to the engine verbs |
| `part.role`, `part.label` | `golden` / `test`, and the scan label |

## result.json

```json
{
  "schema_version": "0.4",
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
  "disposition": "CONTINUE",
  "pull_reasons": [],
  "warnings": [],
  "outputs": {"mesh": [], "deviation_map": "...png", "report": "...json", "log": "run.log.jsonl"},
  "recipe": null,
  "zones": [],
  "feedback": null,
  "station_config": "example_cell_station.json",
  "profile": "default",
  "steps": [{"id": "run", "title": "inspect job", "index": 1, "required": true, "expects": {},
             "sensor_inputs": {}, "dlp_result": {"result": "PASS", "disposition": "CONTINUE",
                                                 "run_id": "..."},
             "outcome": "CONFIRMED", "sensor_mismatch": [], "permit": null,
             "permission_granted": false, "permission_note": null, "actions": [],
             "disposition": "CONTINUE"}],
  "permissions": {},
  "cycle": null,
  "software": {"package_version": "0.1.0", "git_commit": null},
  "started": "...", "finished": "..."
}
```

Schema 0.2 added `recipe`, `zones` and `feedback` (null / [] for the other job types), and
`deviation_mm.n_defects` / `largest_defect` / `outlier_px` / `outlier_clusters` /
`outlier_pct` / `dense_strays` (metric jobs; null for the others); every 0.1 key is
unchanged. Schema 0.3 added `disposition` (CONTINUE / REJECT / PULL) and `pull_reasons`
(`[{zone?, check?, reason, message}]`: every doubt found, also under REJECT); `result` and
every 0.2 key are unchanged. Schema 0.4 added the work states: `station_config`, `profile`,
`steps`, `permissions` and `cycle` (every 0.3 key unchanged). A single job is ONE implicit
work state `run` whose outcome is the job's own disposition (CONTINUE -> `CONFIRMED`, REJECT
-> `DEFECT`, anything else `DOUBT`), with no sensor claim and no permission; a cycle of a
flow lists every work state (*Profiles and work states* above), `cycle` says how it ended
(`ended_by`, `kickout`, `sensor_mismatch`, `sensor_mismatches`, `unconfirmed`) and its
`pull_reasons` may carry `step`.

`INVALID` means the numbers cannot be trusted (no mm calibration at this geometry,
an INDETERMINATE engine verdict, a refused or crashed run). Never report an
INVALID run as a pass or a fail: its disposition is PULL (route the part to quality
inspection).

A fresh `--mock` library has no calibration, so the first mock run is INVALID.
Calibrate the mock geometry once with the validation object, then runs give
PASS / FAIL:

```bat
scan.bat --mock --library library_mock base
scan.bat --mock --library library_mock scan cal --known 106.95,56.5,21.9
scan.bat --mock --library library_mock calibrate cal
scan.bat --mock run stations\example_cell_station.json
```

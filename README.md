# Swiss Army Scanner

One TI DLP LightCrafter 4500 (DMD) and one FLIR Blackfly S mono camera become many virtual sensors: 3D scanning, measurement, defect detection, a good-part memory bank, 2D profiles, barcode/QR reading, spectral material ID, part-present and alignment sensors, and projected grids you can point at ("scan zone H8:I11").

It is one engine with one small verb API (`engine/api.py`). A web UI, a CLI and AI agents (Claude, Grok) all drive the same verbs. The engine owns the hardware, patterns, decoding, calibration, the library and the safety rules; callers never touch the rig directly.

- How it measures, in plain words (one page): [`docs/HOW_IT_MEASURES.md`](docs/HOW_IT_MEASURES.md)
- Spec: [`docs/ENGINE_ARCHITECTURE.md`](docs/ENGINE_ARCHITECTURE.md)
- Agent guide: [`ui/agent.md`](ui/agent.md)
- Operating modes (Measure / Quick check / Station) and the mount (fixed / mobile): *Operating modes* below
- Station configurations, profiles, work states: [`stations/README.md`](stations/README.md), [`flows/README.md`](flows/README.md)
- Recipes (what the DLP checks to verify a work state): [`recipes/README.md`](recipes/README.md), checks + plugins [`docs/CHECKS.md`](docs/CHECKS.md), porting to another line [`docs/PORTING.md`](docs/PORTING.md)
- Seed package: [`swiss_army_sensor/README.md`](swiss_army_sensor/README.md)
- Blue-lines fix: [`docs/HDMI_TIMING_FIX.md`](docs/HDMI_TIMING_FIX.md)

---

## Operating modes

The **triangulation preset** - the basic DLP + camera setup: rig geometry, exposure, fringes, and the stereo calibration bound to that geometry - is its own layer. Three operating modes sit on top of it, each adding to the one before:

| mode | what it is | needs | answers | CLI | HTTP |
|---|---|---|---|---|---|
| **Measure** | an **instrument**: scan and report numbers. No tolerance, no pass / fail, no permission, no disposition - a validity statement instead | a triangulation preset + a valid stereo calibration | values with units, n points and uncertainty; `MEASURED`, or `NOT_MEASURABLE` with the reason and how to fix it | `scan measure-run PRESET --what ...` (exit 0 / 5) | `POST /api/measure` |
| **Quick check** | the preset + **ONE** simple check: a plane flatness limit, a golden compare, or one recipe. No sensors, no permissions | + the check and its threshold | CONTINUE / REJECT / PULL | `scan quick-check PRESET --check ...` (exit 0 / 1 / 5) | `POST /api/quick-check` |
| **Station** | station configuration -> profile -> work states: the line's cycle (everything from *Terminology* on; unchanged) | + a station configuration | per work state CONFIRMED / CORRECTED / ..., permissions, CONTINUE / REJECT / PULL | `scan run stations\<config>.json` | `POST /api/cmd/start`, `/step`, ... |

```
triangulation preset (rig.mount: fixed | mobile)  ->  [optional] station configuration  ->  profiles  ->  work states
        Measure, Quick check                                         Station
```

**Mount: fixed or mobile** (`rig.mount` of the preset, default `fixed`; `scan preset-add NAME --from BASE --set rig.mount=mobile`, shown by `scan preset-show` and `scan presets`). It says how far the stored calibration is trusted, in every mode (`engine/rigstate.py`):

- **fixed**: the calibration is trusted until a rig check says MOVED / UNKNOWN (as before: HOME then puts the bus in FAULT). Measure and Quick check also read the record: such an objection stands, and they give no mm, until it is **answered** (below). A stored scan made while it stood, or just before it (the rig moved at some time before that check), stays in doubt after the answer: scan again. The engine's own scans (`scan-metric`, `scan-fast`, `scan-flash`, `scan-trigger`, `dent-scan`, `face-locate --zone`, `face-scan`, ...) are not refused on a fixed rig, but while an objection stands they never give their mm silently: a printed warning `fixed rig not confirmed: these mm may be off (Measure and Quick check give none)` with the rig check's own words and the way out (a rig check OK again, a re-home, a new calibration, or `--set-reference` with the rig known good), the same first in the entry's meta `warnings`, a `rig` stamp `{"mount": "fixed", "trusted": false, ...}` (`trusted: true` while nothing objects), and both at the top of `--json`. `face-profile`, `station`, `face-locate`, `accuracy` and `accuracy-block` say the same of the scans they measure: one made while the objection stood or just before it. None of this changes a verdict or an exit code.
- **mobile**: the rig is **assumed moved since its last use**. No mm until it was verified **in this session** - by one of the answers below, with no rig check MOVED / UNKNOWN since. A session starts at the newest session marker (`scan rig-session --new`: the rig was set down / powered up / carried; the web UI server's start is one) and is never older than 12 h, than the start of this computer (where the operating system reports it) or than the start of the web UI server that owns the rig. The rig alone being carried or switched off can't be seen: say so with `scan rig-session --new`. Unverified: Measure answers `NOT_MEASURABLE`, Quick check and Station **PULL**, and the engine's own scans (`scan-metric`, `scan-fast`, `accuracy`, ...) are refused (exit 2), each with `mobile rig not verified: run rig-check / re-home / calibrate`, before anything is scanned. The reason is always `rig_unverified` (also after a MOVED: it says nothing about the part, so no recipe can turn it into a reject). A stored scan counts only when it was made after the verification. `scan rig-session` shows the state.
- **What answers** (fixed: clears the objection; mobile: verifies the rig), always for the stereo calibration in use: **a calibration solved from poses captured since** (solving stored poses again writes a new entry and answers nothing); **a rig check OK at the default tolerance (0.5 px) or tighter** - a looser `--tol` is reported and proves nothing - against a reference that belongs to that calibration and was **trusted when it was stored** (mobile: the rig was verified at that moment; fixed: no objection stood, or the operator asked for it); **a re-home** fitted against such a reference; on a fixed rig also **`scan rig-check --set-reference`** (the operator's statement that the rig is known good). A reference stored while the rig was in doubt - by the first check of a calibration, or with `--set-reference` on a carried rig - verifies nothing, in that session or any later one: calibrate where the rig stands, then store the reference.
- **One rig, one mount.** A `geometry_id` names one rig: when any preset at a geometry says mobile, every preset at that geometry is gated as mobile (a `fixed` copy of a mobile preset is no way round; `scan rig-session` names the preset that says so). Give a rig that is really bolted down its own `geometry_id` (`--geometry`) and calibration.
- **The record.** `calibrations\<geometry>\rig_state.jsonl` (append-only): every rig check with its tolerance and the reference it used, every calibration pose and calibration, every session marker. The ORDER of its lines decides what answers what, not their clock times (a clock set back can't put an old OK after a newer MOVED), and only its newest lines are read.

**Measure** (`engine/modes/measure.py`, `api.measure_run`). `--what`:

| what | values | regions |
|---|---|---|
| `flatness` | RMS, peak-to-valley (p2-p98), tilt vs the optical axis, plane distance | `--roi` / `--zone` (default: the whole field) |
| `step` | height of region B above region A's plane (plane to plane), the angle between the two faces | `--roi` + `--roi2` (or `--zone` / `--zone2`) |
| `distance` | the distance between two picked camera px (their 3-D points); a point to the plane of a region. A picked px on an edge (two surfaces within 3 px) is not one point: pick a few px inside the face | `--p1 X,Y` + `--p2 X,Y`, and / or `--roi` (the plane) |
| `cylinder` | radius, diameter, fit RMS, axis (`--radius` fixes the radius, `--axis` hints the axis). Not a cylinder, not a number: a surface that is flat within the noise, or points that do not lie on one cylinder | `--roi` (default: the part in front of the background plane; when the scan shows none, the largest continuous surface) |
| `endface` | diameter and flatness of a cylinder's circular end face toward the camera (`--diameter` = the expected one, `--faces N`) | the scan |
| `dims` | bounding L x W x H of the part above the reference (dominant) plane. The whole part or nothing: one cut by the image border or by the roi is not measurable | `--roi` |
| `deviation` | RMS / mean / p2-p98 / extremes of the deviation vs `--golden` (values only). Points further than 5 mm from the golden are not compared: when they form a patch (a boss, another part) the largest deviation is not in the numbers, so nothing is reported | `--roi` |

Each value has its units, the number of points behind it and an **uncertainty**: with one scan the predicted precision at that distance (`engine/measure/precision.py`, from the scan's own fringe levels) or the measured depth noise **of the px the value rests on** (its region, the surroundings of a picked point - not the whole scan's: a dark part in a quiet scan is noisier than the scan), whichever is larger, and the camera px size for lateral values; where a depth error counts more than once in the value (the radius of a weakly seen arc) it is multiplied accordingly. With `--repeats N` the repeatability over the N scans, but never less than what one scan resolves / sqrt(N) (two scans that happen to agree are not a measurement of the spread; a lateral value also keeps the camera px size: a part that does not move cannot average it away). The repeats are different scans: the same stored scan named twice is refused. Noise only: `scan accuracy-block` proves the absolute mm. A value within 3 x that depth noise or the px size is flagged `near_resolution_limit` with a note: it is mostly the instrument, not the part. `NOT_MEASURABLE` with the reason and how to fix it - no number, never a px presented as mm - without a calibration (`uncalibrated`), with a stale one (`stale_calibration`), on an unverified mobile rig (`rig_unverified`), after a fixed rig's check said MOVED (`rig_check`), with too few points (`too_few_points`), when the number would not be the whole of what it names (`incomplete`: a cut part, an uncompared patch) or not what was asked for (`invalid`: no cylinder, a pick on an edge). A grid `--zone` cuts a stored scan only by a projector map (`scan map`) or when the scan itself was lit through that zone; otherwise it is refused (it would be the whole scan).

A measure run writes a new run folder: `result.json` (`operating_mode: "measure"`, `status`, `reason`, `how_to_fix`, `validity`, `values[]`, `precision`, **`result: null`, `disposition: "NONE"`** - a measurement decides nothing, and whatever reads the document as a run gets a PULL, never a CONTINUE), `values.csv`, `overlay.png` (the camera view with what was measured, 1 image px = 1 camera px), `scan.ply` and `spec.json`. The spec runs again with `scan measure-run --spec FILE`, and is **promoted to a check** by adding tolerances: `scan measure-to-check spec.json --tol step_height=19.5,20.5 --out step_check.json` (flatness / step -> a recipe judged by the existing `flatness` / `height` checks, deviation -> a one-job station config; `api.measure_to_check`).

```bat
scan.bat --mock --library library_mock --preset bench_9in_metric calibrate-stereo --mock-poses 12   :: once
scan.bat --mock --library library_mock --mock-part 80,50,20 measure-run bench_9in_metric --what dims
scan.bat measure-run my_rig --what step --roi 300,400,500,600 --roi2 700,400,900,600 --repeats 5
scan.bat measure-run my_rig --what distance --p1 640,512 --p2 820,512
```

**Quick check** (`engine/modes/quick.py`). Not a second engine: the minimal one-job station config of the check is generated and run by the station job runner, so the verdict, the disposition, the guard band, `result.json` and the exit codes are a station run's. The config is kept in the run folder (`quick_check.json`; `scan run` on it runs the same check again) and the result carries `operating_mode: "quick_check"`.

```bat
scan.bat --mock --library library_mock quick-check bench_9in_metric --check flatness --threshold 0.5 --guard-band 10
scan.bat quick-check my_rig --check golden --golden good_part --threshold 0.3 --operator-confirmed
scan.bat quick-check --check recipe --recipe my_part --operator-confirmed
```

**On the bus and the operator page.** The bus holds the mode (`SELECT_MODE`, `POST /api/cmd/mode {"mode": "measure" | "quick_check" | "station"}`; refused while a cycle runs). Outside Station mode it starts no cycle and keeps READY, PASS and every permission off: nothing a measurement or a quick check says reaches the line's outputs, and `POST /api/measure` / `POST /api/quick-check` are refused (409) in any other mode. `/quick` has the switch at the top (**Measure | Quick check | Station**). Measure: preset, what to measure, a region or two clicks on the picture of the last measurement (camera px), **MEASURE** (key **M**), a table of values with their uncertainty and a neutral *MEASURED* / *NOT MEASURABLE* - never a lamp verdict. Quick check: preset, check, threshold, **START**, the lamp shows PASS (CONTINUE) / FAIL (REJECT) / PULL. HOME is the rig check in every mode. The picker always shows the LOADED preset (the one HOME checks and the measurement runs with). The station's keys (1-9, P, T, A, N, K) work in Station mode only, and the operator's confirmation of the physical setup is for one setup: a mode change or another preset clears it. `GET /api/cmd/state` carries the rig's state (`rig`: may it give mm now); it is cached while the rig-state record is unchanged, so polling it costs two file stats. Built and mock-tested; **not yet run on the real rig**.

---

## Terminology: station configuration, profiles, work states

Five words, in this order. They are different things:

| word | what it is | where |
|---|---|---|
| **Station configuration** | what the station **is and knows**: its parameters (rig preset, library, run folders, the line's sensor inputs, defaults) and the **profile library set** it can call on. A station configuration is **not** a work state. | `stations/*.json` (`stations/README.md`) |
| **Profile** | one part variant of the profile library set (long tube / short tube): **what that part looks like**: its recipes (zones + checks), preset, named zones, golden. The operator, a PLC or HTTP selects the active one (`SELECT_PROFILE`). | `profiles` of the station configuration |
| **Work state** | one **step** of the flow: a claim the line's own sensors make ("a tube is loaded"), the DLP check that **confirms or corrects** it, and the **permission** that lets the line continue to the next thing. | `flow` of the station configuration (inline, or a file in `flows/`) |
| **Check** | what the DLP measures to verify a work state: a recipe of the active profile, or inline zone checks (presence, position, defect, ...). | `recipes/`, `docs/CHECKS.md` |
| **Disposition** | what happens to the part: CONTINUE (every required work state was CONFIRMED), REJECT, or PULL for quality inspection (any doubt). | *Dispositions* below |

**The DLP confirms or corrects the sensor.** A line sensor says a part is there. Is it really there, or not? Is that tube there, is that dent there? Each work state is verified with three values per signal: what the work state **expects**, what the **sensor says** (`POST /api/io/inputs`), what the **DLP says** (the scan):

| outcome | when | permission | then |
|---|---|---|---|
| **CONFIRMED** | sensor and DLP agree, it is what the work state expects, every check passes with no doubt | **granted** (`PERMIT_<NAME>`) | the next work state |
| **CORRECTED** | the DLP disagrees with a sensor (sensor says present, the DLP says absent, or the reverse): `sensor_mismatch` names the signal and both values | withheld | `on_mismatch` |
| **NOT_MET** | sensor and DLP agree, on something the work state does not expect (no part where one is needed, a part where none may be) | withheld | `on_not_met` (default: the `on_mismatch` list) |
| **DEFECT** | the part is there, and a check definitely fails (misaligned, dented, not this profile's part) | withheld | `on_defect` |
| **DOUBT** | the DLP cannot tell: INVALID, unmeasured, a value in the guard band, a sensor that never reported for this cycle or changed during the scan | withheld, always | `on_doubt` (default: kickout PULL). Never a permission, never a reject, never passed over |

Actions: `project_fault` (paint the fault / the missing-part zone on the part), `prompt` (a message; waits for **ACK**), `hold` (ACK, then verify again), `retry` (scan again, up to `retries`), `kickout REJECT|PULL` (ends the cycle), `goto <step>`. `KICKOUT` is on with every FAIL and every PULL. Permissions are true only while their work state is CONFIRMED in the current cycle (and every required work state before it is); they all drop on ABORT, FAULT, HOME, ARM, DISARM, a configuration / profile change, a START / TRIGGER that starts no cycle (that part was not checked), and when the cycle ends anything but CONTINUE. The cycle is CONTINUE only when every required work state was CONFIRMED; one left in DOUBT or CORRECTED ends it PULL, required or not.

Fail closed, by construction: each expected sensor signal is observed by the check the work state **names** for it (`verify.observe`; nothing is guessed); a sensor claim belongs to **one cycle** (forgotten when it ends, so a silent gateway is a DOUBT at the next part, never an old claim read as agreement); only the answer to the verify the bus asked for confirms anything.

```bat
scan.bat flow-show stations\example_tube_line.json                              :: parameters, profiles, work states
scan.bat --mock run stations\example_tube_line.json --profile short_tube         :: one whole cycle, in line
scan.bat --mock run stations\example_tube_line.json --mock-case dented           :: DEFECT at "End face OK?" -> kickout REJECT
scan.bat --mock run stations\example_tube_line.json --mock-case sensor_true_part_absent   :: CORRECTED: the DLP corrects the sensor
```

The example (`stations/example_tube_line.json` + `flows/example_tube_line.json`): profiles `long_tube` / `short_tube`, work states *Tube loaded?* (permit `LOAD_DONE`), *Tube aligned?* (position vs the golden, permit `CLAMP`), *End face OK?* (dents, permit `TRANSFER`). Every one expects `part_present` and asks the same named check (`seat/presence`): a tube that went missing at any step is a missing tube (the sensor is corrected, the operator prompted), never a dent. Mock setup, once per mock library: `calibrate-stereo --mock-poses 12` with the profile's preset, then `recipe-golden tube_long_alignment` and `recipe-golden tube_short_alignment` (`stations/README.md`). Built and mock-tested; **not yet run on the real rig**.

An old single-job station config (`station_id`, `preset`, `job`, `part`) still loads and runs exactly as before: it is a configuration with one implicit profile (`default`) and a one-step flow (its job).

---

## Quick start (operator UI)

The page follows the DLP workflow reduced to its core, **Lights, Camera, Action**:

- **Lights**: set up the illumination. Projector / LEDs ready, aim the grid (*Project grid*), assign the areas (the station configuration's zone; *Show zone* is a placeholder for now).
- **Camera**: aim the camera at the objects and choose what to capture (*Live preview* to aim and focus, *Signal check* for contrast and focus).
- **Action**: run the cycle: **START**, then each work state is verified (**STEP**).

Lights and Camera are setup panels the operator opens when needed (only while the station is idle); Action is the production cycle.

1. Start the server: `run_ui.bat` (real rig) or `run_ui_mock.bat` (simulated bench, port 8766).
2. Open **http://127.0.0.1:8765/quick** (mock: http://127.0.0.1:8766/quick). The full engineering UI stays at `/`; each links to the other.
3. Top of the page: pick the **station configuration** (one entry per `stations/*.json`) and the **profile** (which part variant the work states check; a single-job config has one, `default`).
4. Middle of the page: the **work states**, the steps of the configuration's flow, one row each with three chips, *sensor says* / *DLP says* / *permission*, and the outcome; the current work state is highlighted. Press **START** to begin a cycle, **STEP** to verify the current work state. When a work state asks for the operator, an amber **prompt banner** appears with an **ACK** button. In MOCK the page can play the line: click a sensor chip to toggle the simulated sensor, or pick a case of the simulated line (the simulated sensors report for every cycle, as a line's gateway must). A command the bus refused is shown above the buttons until the next one is accepted, and the prompt banner with its ACK button stays in view while the page is scrolled.
5. The lamp shows READY / RUNNING / `STEP n/N` (a cycle waits for STEP) / `ACK NEEDED` / **PASS** (green: CONTINUE) / **FAIL** (red: REJECT) / **PULL – quality inspection** (amber: any doubt - INVALID, ABORTED, a value in the guard band, an unreadable code, a work state that was not confirmed - with the reasons; never a pass), with the verdict, the disposition, the key number (max deviation in mm, or the current work state), the time, a link to `result.json` and the deviation map when there is one. The lamp follows the cycle, never a single work state's own verdict. The *Lights / Camera / Action* strip follows a single job (queued, capturing, verdict). See *Dispositions* below.


The **mode switch** at the very top (*Measure | Quick check | Station*) selects the operating mode (*Operating modes* above); the steps above are Station mode. In Measure mode the page shows the preset, what to measure and the **MEASURE** button and answers with a table of values and their uncertainty; in Quick check mode the preset, the check and its threshold, and START runs that one check.

| Button | What it does |
|---|---|
| **START** | begins a cycle of the selected station configuration with the selected profile. A single-job config runs its job at once, exactly like `scan run` (a new run folder with `result.json` + `run.log.jsonl`); a flow waits at its first work state for STEP (a work state with `trigger: auto` is verified as soon as it is current) |
| **STEP** | verifies the current work state: the sensor claims are read, the DLP scans, the work state is CONFIRMED (permission granted, next work state) or its actions run |
| **ACK** (prompt banner) | acknowledges a prompt: "I did what it asks" (loaded the tube, removed the part); a `hold` then verifies the work state again |
| **ABORT** | stops the run before its next capture; the rig is closed (projector idle, rig lock released) and the run is recorded as **ABORTED** |
| **HOME** | the **rig check**: a fresh flood + a seeded random block pattern compared with the reference stored for this geometry. It measures shift + roll + zoom (phase correlation per tile, then a similarity fit) and judges the largest image motion over the frame. **OK** only with the pattern matched and its roll / zoom determined, every motion within `--tol` (0.5 px; a finite number). **MOVED** with the numbers and **which device moved**: a camera move shifts the scene in the camera's view and the projected pattern with it, a projector move (or a move of its picture: focus / zoom, the picture the computer sends) shifts the pattern against the scene, so the camera's motion is the flood's and the projector's the pattern's motion relative to the flood, tile by tile (`camera_motion_px`, `projector_motion_px`; a part of the frame that moved with neither, e.g. a part placed since an empty-scene reference, is set aside: `tiles_set_aside`). Each device counts as moved beyond `--tol` (two moves each within it but together beyond it: both, only when every part of the frame still matches the reference and the two account for the motion). E.g. "camera moved ~2.3 px (~0.70 mm at 640 mm) (...; the projected pattern moved with the scene): a small camera turn can be re-homed, else recalibrate before measuring"; "projector (or its picture) moved ~18.6 px against the scene, the camera's view 0.3 px (...): a projector move can't be re-homed: put the projector and its picture back (...) and check again, else recalibrate"; "camera and projector moved: the camera's view ~0.7 px, the projector's picture ~18.6 px against the scene (...): recalibrate before measuring"; "camera or projector moved" when it can't tell (the scene's own texture is not matched, or neither moved beyond the tolerance on its own: two small moves together, or has the scene changed since the reference?). Also MOVED when the pattern is visible but no longer matches the reference (projector moved or rolled beyond range). **UNKNOWN** when the pattern is not seen (projector off or blocked), when roll / zoom could not be determined (the pattern matched in too few parts of the frame), or when the flood shows texture but no longer matches (scene changed). The first HOME at a geometry stores the reference (do it right after base / calibration, station in its usual empty state), and the message says why one was stored: the first at this geometry, "new calibration → new reference stored" (the previous reference belonged to another calibration; it stays in the library), or on request. mm need a stereo calibration and a metric scan (the working distance); otherwise px only. Roll / zoom detection is tested on the mock bench only so far. CLI: `scan rig-check [--set-reference] [--tol PX]`. After a small **camera** nudge (MOVED, flood and pattern moving together, ≤ ~5 px) the check says that `scan rig-check --rehome` can take it out: see *Calibration* below (CLI and `api.rig_check(rehome=True)` only; HOME never re-homes) |

**Real rig: operator confirmation.** START (and ARM) stay disabled until the operator ticks *"I confirm the setup"* (the `--operator-confirmed` of `scan run`); the tick clears after each run. Mock mode needs none. Over HTTP, `operator_confirmed` must be a JSON boolean: only `true` confirms, any other type (`"false"`, `"true"`, `1`, ...) gets a 400 on START / TRIGGER / ARM. The same rule applies to the job verbs' `operator_confirmed`, `confirm` and `empty_confirmed`. ABORT, DISARM, HOME and RESET ignore the field: they are never refused because of it.

**Commands and sources.** Every source sends the same commands through one state machine (`engine/control/commands.py`):

| Command | Button | Key | HTTP (`POST`) | Hardware input (placeholder) |
|---|---|---|---|---|
| START | START | Space / Enter (only when no button, link or summary has focus) | `/api/cmd/start` `{"operator_confirmed": true, "config": "<config>.json" or 1..n, "profile": "<name>" or 1..n}` | serial `START`, DI |
| ABORT | ABORT | Esc | `/api/cmd/abort` | camera GPIO Line3, DI1, serial `STOP` |
| HOME | HOME | H | `/api/cmd/home` | DI2, serial `HOME` |
| TRIGGER (part present) | TRIGGER | T | `/api/cmd/trigger` | camera GPIO Line0, DI0, serial `TRIG` |
| ARM / DISARM | ARM | A | `/api/cmd/arm`, `/api/cmd/disarm` | DI3 |
| STEP (verify the current work state) | STEP | N | `/api/cmd/step` `{"step": "<id>" or n}` (optional: the one the sender expects) | a PLC bit (placeholder) |
| ACK (acknowledge a prompt) | ACK | K | `/api/cmd/ack` | a PLC bit (placeholder) |
| RETRY / SKIP | - | - | `/api/cmd/retry` (scan the current work state again: its `retries` budget), `/api/cmd/skip` (only a work state with `allow_skip`; never grants) | - |
| SELECT_CONFIG (station configuration) | configuration list | 1-9 (Station mode) | `/api/cmd/config` `{"config": "<config>.json" or 1..n}` | a PLC register (placeholder) |
| SELECT_PROFILE (the active profile) | profile list | P (next) | `/api/cmd/profile` `{"profile": "<name>" or 1..n}` | a PLC register (placeholder) |
| SELECT_MODE (the operating mode) | mode switch | - | `/api/cmd/mode` `{"mode": "measure"}` (or `"quick_check"`, `"station"`; refused while a cycle runs; outside Station mode START / TRIGGER / ARM are refused and READY is off) | a PLC bit (placeholder) |
| MEASURE (Measure mode only; not a bus command) | MEASURE | M, Space / Enter | `/api/measure` `{"what": "flatness", "preset": ..., "roi": [x0, y0, x1, y1], ...}` -> a job (`GET /api/jobs/<id>`) | - |
| quick check (Quick check mode only; not a bus command) | START | Space / Enter | `/api/quick-check` `{"check": "flatness", "threshold_mm": 0.5, "operator_confirmed": true, ...}` -> a job | - |
| sensor claims (inputs) | sensor chips (MOCK: click sets the simulated sensor) | - | `POST /api/io/inputs` `{"part_present": true}` for **every part** (a claim belongs to one cycle; an undeclared signal name is a 400); `GET /api/io/inputs` | an input backend's `signals` map (placeholder) |
| RESET (acknowledge) | - | - | `/api/cmd/reset` | a PLC bit (placeholder) |
| state + signals | lamp | - | `GET /api/cmd/state` | outputs: READY / BUSY / PASS / FAIL / ALARM / PULL (stack light; PULL = quality-inspection lane) + KICKOUT (on with FAIL or PULL) / PROMPT / SENSOR_MISMATCH + `PERMIT_<NAME>` per permission (a deselected configuration's stay published as false); registers `result_code`, `state_code`, `station_config`, `profile`, `work_state`, `work_state_code` |

States: READY → START → RUNNING → READY (a single job); with a flow READY → START → WAITING (the current work state waits for STEP, or a prompt for ACK) → STEP → RUNNING (the DLP verifies it) → WAITING ... → READY when the cycle ends; READY → ARM → ARMED → TRIGGER → RUNNING → ARMED (next part; TRIGGER also verifies a work state that waits for STEP). **ABORT always wins** and is never refused: in every state it also stops the projected verdict, the live view and station / recipe runs or rig checks submitted via `/api/jobs` (RUNNING / HOMING → ABORTING → READY; WAITING → READY: the open cycle is ABORTED, PULL), and every permission drops. An aborted or cancelled TRIGGER cycle does not re-arm. **HOME** only when idle (READY / ARMED / FAULT) or queued behind an ABORT: a HOME pressed while aborting runs when the aborted job ends, and a further ABORT cancels it; HOMING → READY, or FAULT when the rig moved (HOME again, or RESET). **ALARM** is on in FAULT, and while idle after an INVALID cycle until it is acknowledged: **RESET** (accepted in FAULT and while idle with the ALARM on; refused with 409 when there is nothing to acknowledge) or a good HOME switches it off, and the next INVALID result or FAULT raises it again. Acknowledging changes nothing else: the last cycle's PULL / FAIL output and its result code stay until the next cycle starts (they release nothing: PULL means "this part goes to quality inspection", not "the station is faulted"), and a PASS never comes back. **TRIGGER** is ignored unless ARMED or a work state waits for it. A START / TRIGGER that starts no cycle while none is open (refused, or ignored) releases the last cycle: every permission drops and a PASS is forgotten with its result code and work-state registers (that part was announced and not checked; a FAIL / PULL stays until the next cycle); ARM and DISARM do the same, so a PLC acts only on what follows a command the bus accepted. START / TRIGGER / ARM may carry the station configuration and profile the sender expects (`"config"`, `"profile"`); they are refused (409) when another one is selected, and the /quick page always sends them. The configuration and the profile can't change while a cycle runs; changing either forgets the last cycle (its PASS / FAIL / PULL outputs, its result code, its sensor claims), drops every permission and, while ARMED, disarms (ARM again). A profile is selected by name (also a name made of digits) or, with a JSON number, by its 1-based position. Configuration numbers (keys 1-9, PLC numbers, the `station_config` register) are every station config in file-name order, invalid ones included: an invalid or vanished config keeps its number, is marked unusable (`unusable` in `GET /api/cmd/state`) and START / TRIGGER on it are refused with result INVALID. The selected configuration and profile are never switched automatically, and the list is frozen while ARMED or in a cycle. A work state with `timeout_s` that waits longer for STEP / ACK ends its cycle PULL. `SELECT_WORKSTATE` (`/api/cmd/workstate`, the `workstate` body key) is the **deprecated** old name of `SELECT_CONFIG`: it always selected a station configuration, never a work state; it is still accepted and its reply says so. A refused HTTP command answers 409 with the reason. Hardware inputs (camera GPIO lines, serial / USB-IO, a 24 V digital I/O module) and outputs are **placeholders** with a config-driven mapping (`stations/io_example.json`): not implemented yet, disabled in the example.

---

## Dispositions: CONTINUE / REJECT / PULL (quality inspection)

Every run gets a **disposition** on top of its verdict (`engine/disposition.py`): what happens to the part. When in doubt, the part is pulled for quality inspection; a doubt never passes and is never rejected on numbers nobody can trust.

| disposition | when | exit code | outputs | projected on the part |
|---|---|---|---|---|
| **CONTINUE** | the verdict is PASS and nothing about it is in doubt | 0 | PASS + the PASS result code | green outlines |
| **REJECT** | a definite measured FAIL (a trusted measurement found a bad part) | 1 | FAIL + the FAIL code | red fill + a cross at each fault |
| **PULL** | any doubt (below) | 5 | PULL (never PASS) + the INVALID / ABORTED code, or the PULL code (5) for a PASS / FAIL in doubt | an amber frame round the whole part + "PULL", amber on the zones that caused it |

**PULL reasons** (`pull_reasons` in `result.json`: `[{zone?, check?, reason, message}]`): INVALID of any kind (`uncalibrated`, `stale_calibration` (a scan made at other rig settings or with another calibration), `px_not_mm`, `too_few_points` (points / coverage), `zone_unresolved`, `golden` (missing or mismatched), `ambiguous_faces`, `incomplete`, `invalid`), `check_crashed`, `code_unreadable` (an expected code that could not be read at all: unreadable is not wrong; a code read with the wrong text is a REJECT), `size_mismatch` (a part of known size - an end face, a round part in a circle zone - reads more than its tolerance small / large: the calibration or the lens is suspect, nothing on that scan is trusted and nothing is rescaled; it can never be turned into a REJECT), `exposure` (no camera exposure shows the part well enough: it was not scanned), `rig_check` (the rig check a recipe runs before the cycle said MOVED / UNKNOWN; for a quick check on a fixed rig: one on record did and was not answered since), `rig_unverified` (a MOBILE rig, preset `rig.mount`, not verified in this session - also after its rig check said MOVED: *Operating modes*), `aborted`, `refused` / `error` (a run refused or crashed before its verdict), and the optional **guard band** (`guard_band`): a PASS value within `guard_band_pct` % of the tolerance from a limit (two-sided: of the half width, from each limit; one-sided: of the limit, or of the tolerance's `span` when it gives one - part present >= 92 % of a share that can't pass 100 % has span 8) is pulled; a value on its limit is always in the band, and a tolerance with no size (zero width, a one-sided limit at 0 without a span) clears nothing. Recipes set it in `pull.guard_band_pct` (the built-in examples: 10), station metric jobs in `job.guard_band_pct` (default 0 = off). A PASS whose own numbers do not hold it up - a value missing, not a finite number, or outside its own limits (the verdict and the number disagree) - is pulled too (`invalid`), guard band or not. The checks hold up the values they judged, not rounded copies; the `defect` check's face mode reports the deepest significant dent still under `depth_mm` on a PASS, so its guard band sees a dent just under the limit.

**Precedence** (one run): any definite measured FAIL -> **REJECT**, even when another zone is INVALID (the doubt is still listed in `pull_reasons`); otherwise any doubt -> **PULL**; otherwise **CONTINUE**. One doubt goes before a FAIL: a `size_mismatch` (a part of known size read the wrong size on this scan) makes every mm of the scan suspect, so the run is a **PULL** whatever else it found - none of its FAILs is definite, and none of its doubts is rejected through a recipe's list. A FAIL is definite only when nothing about it is in doubt: a check FAIL carrying a `pull` reason, or a metric golden FAIL whose placement did not converge (or whose surfaces disagree with no defect cluster to account for it), is a PULL. A recipe may list doubt reasons its line rejects instead (`pull.invalid_reasons_to_reject`, default none); `aborted`, `guard_band`, `refused`, `error`, `rig_unverified`, `size_mismatch` and `exposure` never reject, and a run that was refused or crashed is never rejected through the list (its reason is `refused` / `error`, never guessed from the error text). `result` (PASS / FAIL / INVALID / ABORTED) is unchanged; `result.json` (schema 0.3) and `run.log.jsonl` (`verdict`, `run_end`) add `disposition` and `pull_reasons`. Refused (2), crashed (4) and ABORTED (1) runs keep their exit codes; their disposition is PULL. `face-check` / `station-check` exit 5 (PULL) for INCOMPLETE and for ambiguous end faces. `--json` names the disposition at the top level for `run`, `recipe-run`, `quick-check`, `station-check` and `face-check`, also when the command was refused or crashed. The command bus publishes a **PULL** output for the PLC's quality-inspection lane ([`docs/PORTING.md`](docs/PORTING.md)); `/quick` shows it as an amber lamp, *PULL – quality inspection*, with the reasons. The lamp follows the bus: it never shows a PASS the outputs do not give. An ABORT pressed while a cycle runs makes it ABORTED (PULL) on the line even when the run had already judged it: up to `result.json` the run itself records ABORTED; after that, the job's result (`line_result` ABORTED, disposition PULL) and `run.log.jsonl` (`aborted_after_verdict`) say so, and no verdict is projected. A HOME (rig check) ends a PASS: its output and code never come back for that cycle.

---

## Install

Windows 10/11, Python 3.12 (a conda env is the tested setup):

```bat
conda create -n scan python=3.12 -y
conda activate scan
git clone https://github.com/<you>/swiss-army-scanner.git
cd swiss-army-scanner
pip install -e .                 :: engine + swiss_army_sensor + numpy, scipy, opencv-python
pip install -e .[dev]            :: + pytest
pip install -e .[codes]          :: optional: zxing-cpp for stronger QR / Data Matrix / 1D decoding
```

- **Mock mode needs nothing else.** `run_ui_mock.bat` starts the simulator UI at http://127.0.0.1:8766 (its own `library_mock`), and `scan.bat --mock ...` drives the simulated bench from the CLI.
- **Real rig:** install FLIR's **Spinnaker SDK** and the matching **PySpin** wheel from FLIR (it is not on PyPI; the wheel's `cpXX` must match your Python, e.g. cp312). `pip install -e .[hardware]` adds `hidapi`, used only by the separate USB diagnostics tool (`engine/capture/dmd_control.py`).
- **Which Python the launchers use:** `scan.bat`, `run_ui.bat` and `run_ui_mock.bat` run `%SCAN_PYTHON%` if it is set (the full path to your env's `python.exe`), else the `python` on PATH, so activate the env first or `set SCAN_PYTHON=...`. The real rig needs PySpin in **that** Python: if it cannot import PySpin, a real-rig command stops with one line (`real rig needs PySpin (FLIR Spinnaker SDK) in this Python ...; set SCAN_PYTHON to the env that has it, or use --mock`, exit 3). Check with `"%SCAN_PYTHON%" -c "import PySpin"`.
- Node is optional (only for the UI's JS tests).
- The TI documents the code cites (DLPU010G, DLPU011E, DLPA036/037) are not in the repo; download them from ti.com.

## Start here

**Web UI** (recommended). Double-click one of these; it opens at http://127.0.0.1:8765:

| Launcher | What it drives |
|---|---|
| `run_ui.bat` | the **real rig** |
| `run_ui_mock.bat` | a simulated bench, for practice: its own `library_mock` and port 8766 |

**CLI.** `scan.bat --help` lists every command. `--json` gives agent-friendly output, `--mock` uses the simulator.

```bash
scan.bat signal
scan.bat base
scan.bat scan part1 --known 106.95,56.5,21.9
scan.bat measure part1
scan.bat map
scan.bat grid
scan.bat scan boxA --zone H8:I11@96
```

**Before a real session:**
1. Close the LightCrafter GUI.
2. Check the EVM display is 912×1140 at desktop offset (1920, 0).
3. Keep the camera and projector untouched between a base and its scans.

Only one program drives the rig at a time: a machine-wide rig lock refuses a second one with `RigBusy`.

**Flow-ready job.** One command runs one job from a station config and writes a new run folder with `result.json` (the same keys every run: PASS / FAIL / INVALID, the disposition CONTINUE / REJECT / PULL and its reasons, deviation in mm, calibration id, outputs) and `run.log.jsonl` (one JSON event per line). The exit code is the disposition: 0 CONTINUE (PASS), 1 REJECT (FAIL), 5 PULL (INVALID or any other doubt), 2 refused, 4 crashed. See [`stations/README.md`](stations/README.md).

```bat
scan.bat --mock run stations\example_cell_station.json
scan.bat run stations\my_station.json --operator-confirmed
```

Which calibration each job type needs:

| `job.type` | measures | needs |
|---|---|---|
| `inspect`, `compare` | relief vs a base (blobs) / vs a golden scan | base-relief depth calibration (`scan calibrate <scan>` with a known object) |
| `metric` | mm by triangulation: flatness of a plane in a camera-px `roi`, or deviation from a golden metric scan | **stereo calibration** (`calibrate-stereo`) |

**Metric job** (`stations/example_metric_check.json`): `scan_fast` (default `axis: xy`) or `scan: "metric"` (`scan_metric`), then `measure: "plane"` (RANSAC plane in `roi`; INVALID when less than `job.min_valid_pct` (30 %) of the roi decoded; optional `job.plane_range_mm` [near, far] presence check of the plane's distance) or `measure: "golden"` (every foreground component in the roi compared with `golden: "<id or label>"` in both directions, so missing and extra material count; `latest` and the station's own `part.label` are refused, they are the previous run's test scan). The background plane is the datum: the part is placed on the golden only by a turn and a shift along that plane, so a re-placed good part passes while a part or one component that sits higher, lower or tilted keeps that offset (a background more than 2° / 5 mm off the golden's is INVALID: the rig or the fixture moved). A point past the other part's outline (extra or missing material, a component moved sideways, a part too long) counts by how far it lies beyond that outline, in full: an outline is only as sharp as a camera px (~0.35 mm on the bench), so in-plane edges are not judged much finer than that. Only well-decoded px seen within 75° of face-on are judged (the rest is still the reference surface); a foreground region decoded too sparsely to judge, less than `job.min_coverage_pct` (95 %) of the golden or the test surface agreeing, or a placement that does not converge is INVALID. The verdict comes from out-of-tolerance clusters: out-of-tolerance px within 3 px of each other are one cluster, and any cluster of `job.min_defect_px` (5, at most 50) camera px or more beyond `threshold_mm` FAILs; smaller isolated ones are dropped as stray points, but more than `job.max_outlier_pct` (0.1 %, at most 5 %) of the measured px dropped (golden: of both scans together, or of either scan's own once they are two defect clusters' worth), or dropped px dense in one place (5 or more and 2 % of a 21 px square: pits a few px apart, a noisy patch), is INVALID; `deviation_mm.max` is the max over everything not dropped (no percentile can hide a defect). `threshold_mm` is required (INVALID without it); no stereo calibration at the preset's geometry is INVALID with the commands to fix it. result.json adds `deviation_mm.method`, `rms`, `n_points`, `n_defects`, `largest_defect`, `outlier_px` / `outlier_clusters` / `outlier_pct` / `dense_strays` (what was dropped), the metric scan id and the calibration id.

### Calibration (stereo, for metric jobs)

1. `scan.bat charuco --square 25`: print the PDF at 100 %, measure a printed square with calipers.
2. `scan.bat parity` once per rig (diamond DMD row parity).
3. `scan.bat calib-pose` per board pose: 12-20 poses, tilted up to ±30°, filling the measuring volume. Each pose prints its **projector corners** (corners that got a projector position / corners found) and warns under 60 %: fringes too fine for the print, the focus or the distance decode nothing on the board. The default fringe periods (8,80,1280 DMD px) are then captured again coarser (16,160,1280, then 24,192,1280) unless `--periods` was given; the first accepted pose fixes the periods of its pose set (one period set per set, never mixed). `scan.bat calib-poses` lists the poses; `scan.bat delete <pose id> --yes --reason WHY` moves a bad one to the library trash.
4. `scan.bat calibrate-stereo --square <measured mm>`: camera RMS < 0.3 px, projector RMS < 0.5 px; it lists which poses gave the projector fit its points. It is **rejected, nothing saved, exit 1** when the fit is not a calibration: too few poses with projector points, degenerate projector intrinsics (a focal length ≤ 0 or absurd, f_v / f_u far from both 2, a diamond DMD, and 1, square pixels), or a stereo RMS above 2 px. The diamond parity line says *measured* (also when the measurement found no half-column offset: a square grid), *set*, or *NOT measured* (the only case that warns).
5. `scan.bat rig-check --set-reference` with the station empty, so HOME can detect a later nudge. Between rig checks every scan that decodes both stripe directions checks itself: its camera and projector rays should meet, and their median miss (`consistency`) is held to 5 × what the calibration's own fit residual (its stereo RMS) is at the scan's distance, never below 0.3 mm (`meta.consistency_check`). Above that it warns, printed and first after any rig warning in `warnings`: "the calibration may not fit this rig any more: run the rig check" - the mm are still given. On the bench: 0.04-0.25 mm while the calibration fitted (its 0.26 px are 0.09 mm at 766 mm: limit 0.47 mm), 1.0-1.8 mm after the projector was nudged with no rig check run. Measure's `endface` adds the same hint to its NOT_MEASURABLE reason when faces were set aside because too little of the disc was measured on such a scan.
6. After a small camera nudge (rig check MOVED, "camera": the projected pattern moved with the scene, ≤ ~5 px): `scan.bat rig-check --rehome` fits the camera's rotation from the reference flood vs the fresh one and writes a **new** stereo calibration with that rotation taken out (R1 = R0 · Rd, t unchanged), flagged "re-homed (rotation only) from <calibration id>"; the fresh frames become its reference and the original calibration is kept (`scan.bat delete <re-homed id> --yes` returns to it). Refused when the projector moved, the motion is larger, or the registration is poor. Rotation only: a camera translation is not corrected, so the absolute depth may keep a small offset, and goldens / null calibrations bound to the old calibration id must be recorded again. A full calibration is the reference: recalibrate when you can.

Each step uses the selected preset (or `--preset NAME`): calibrate with the preset your station configs and recipes use, its geometry is what the calibration is bound to. Recalibrate after any zoom / focus / mount change. Mock: `scan.bat --mock --preset bench_9in_metric calibrate-stereo --mock-poses 12` (the preset the built-in recipes pin).

**Faster, without hand-holding (mock-tested only, not yet used on a real rig):** a printed **calibration fixture**: four tilted faces, each with its own ChArUco board (disjoint marker ids). One capture is one view per face, so 3 placements replace the 12-20 hand-held poses, refined as one rigid fixture. A printed **check block** (step gauge, flat, half-round bar, dome) then verifies the mm against your caliper values. Use `python tools/make_calib_fixture.py --out <folder>` (STL + stickers + specs), then `scan.bat calib-pose --fixture <spec> --set fx` x3, `scan.bat calibrate-stereo --set fx --fixture <spec>` and `scan.bat accuracy-block --block <spec> --measured <values>`. See [`docs/CALIBRATION_FIXTURE.md`](docs/CALIBRATION_FIXTURE.md).

### Presets (profiles)

A preset holds every setting for one rig geometry: the **profile** (capture settings + geometry). Different parts on the same line (boxes, cylinders, panels) can each get their own preset, so the line knows which settings go with which part; a **recipe** (next section) pins one and adds the part's zones and checks.

- **Built-in presets** (`presets/*.json`, in the repo) are **read-only** for the commands below: never edited or deleted by them.
- **User presets** live in the library: `<library>/presets/<name>/preset.json` + `meta.json` (created, author, description, copied from). They are runtime data like the rest of the library (not in git).

| | CLI | Python | Web |
|---|---|---|---|
| list both tiers (tier, selected, geometry, stereo calibration, description) | `scan presets` | `api.preset_list()` | Setup → Preset library |
| show one | `scan preset-show NAME` | `api.preset_show(name)` | `GET /api/presets/NAME` |
| add (copy + override) | `scan preset-add NAME --from BASE [--set key=value ...] [--geometry ID] [--desc TEXT] [--replace]` | `api.preset_add(name, base, overrides={...}, geometry_id=None, description="")` | Add preset; Setup editor → *Save as user preset* (`POST /api/presets {preset, base, overwrite}`) |
| select (the default) | `scan preset-select NAME` | `api.preset_select(name)`, `api.preset_active()` | Select |
| delete (to trash) | `scan preset-delete NAME --reason TEXT [--force]` | `api.preset_delete(name, reason, force=False)` | Delete (asks for a reason; `confirm=true` is required) |

- **Add** copies a built-in or user preset and overrides fields (`--set camera.exposure_us=33333 --set projection.duty=0.8 --set projection.invert=false`; values are read as JSON when they parse). The result goes through the same validation as `load_preset` (unknown keys, ranges, LED limits; tolerances must be finite numbers). A built-in name is refused; an existing user name too, unless `--replace` (the old one moves to `trash/presets/`); `--replace` keeps the geometry rule (a geometry-bound change under the same `geometry_id` is refused). Names are case-sensitive ids that are the same folder name on every platform: a name differing from an existing one only by case, a Windows device name (CON, NUL, COM1 ...), a trailing dot or a `.json` / `.yaml` ending is refused. The web Setup editor's *Save as user preset* writes a user preset to the library the same way, never `presets/*.json`; built-in names are refused there too.
- **Select** records the library's active preset (`<library>/presets/active.jsonl`, append-only, the last line wins; unreadable lines are skipped with a warning). The CLI uses it when no `--preset` is given; the web server loads it at start. If the selected preset no longer exists, CLI commands that load it stop with an error (select another or pass `--preset`), and the web server says so at start and loads none in its place (select one in Setup). CLI options must be spelled out (`--pres` is not `--preset`).
- **Delete** moves a user preset to `<library>/trash/presets/` (recoverable). Refused for built-ins, for the selected or loaded preset (select another first), for the preset of the selected recipe, and (unless `--force`) for a preset a station config in `stations/`, a recipe or a recipe station config uses (listed). A YAML station config or recipe that can't be parsed here (PyYAML is optional) counts as using it when its text mentions the preset, a recipe pinning it or a recipe file that pins it: never skipped, an over-match only asks for `--force` (`recipe-delete` reads such a station the same way).
- **A new rig geometry** = a new `geometry_id` (`--geometry rig2_v1`) + its own calibration. Nothing is copied or fitted for it: the new geometry starts **uncalibrated** (no bases, no stereo calibration) until you run the calibration steps above with that preset. Changing a geometry-bound setting (rotate90, DMD size, coded axis, projector side, baseline, standoff) under an existing `geometry_id` is refused.
- **Mount** (`rig.mount`: `fixed`, the default, or `mobile`; `--set rig.mount=mobile`). Not geometry-bound: it says how far the stored calibration is trusted. A mobile rig gives no mm until it was verified in this session, and a `geometry_id` names one rig: every preset at a geometry that has a mobile preset is gated as mobile (*Operating modes*).
- **Station configurations and profiles pin presets.** A station config (`stations/*.json`) names its own `preset` (`parameters.preset`, or a profile's own), and that always wins over the selected one; selecting a configuration or a profile loads it. A station config naming no preset (and not a recipe) runs the preset LOADED in the web server, and is refused when none is loaded. When the library's selected preset no longer exists (deleted, renamed) the server loads none in its place and says so (a banner on both pages, `warnings` in `GET /api/cmd/state` and `/api/status`); such runs are refused until a preset is selected in Setup. The operator page (`/quick`) shows *Preset START runs* (`run_preset` in `GET /api/cmd/state`). Preset changes (select, load, delete) are refused while a station run is busy.

### Recipes (the checks that verify a work state)

A recipe is **what the DLP checks** to verify a work state (a profile of a station configuration names the recipes that say what its part looks like; a single-job config names one in `job.recipe`). It ports to any line with a DMD projector + a camera: what to scan, where to look, what is recognised there, and what the projector paints on the part afterwards. It is versioned data (JSON / YAML): nothing part-specific lives in code.

| layer | what it holds | where |
|---|---|---|
| preset (the rig) | rig geometry + capture settings; its stereo calibration gives the mm | `presets/`, `scan preset-add` |
| recipe | pins a preset; scan kind (fast / metric / trigger), lit zone, part frame (reference plane), golden scan | `recipes/*.json` (built-in, read-only), `<library>/recipes/` (user) |
| zones | rect / circle / polygon / grid cells, in camera px, projector (DMD) px or part-frame mm (always said) | `zones[]` |
| checks | what is recognised in a zone: presence, position vs golden, height / step, flatness, defect (end-face MF-CFAR or local), edge straightness / position, code read, or a plugin | `zones[].checks[]`, [`docs/CHECKS.md`](docs/CHECKS.md) |
| verdict | any FAIL -> FAIL; else any INVALID -> INVALID (never a pass); else PASS | `result.json` `zones[]` |
| disposition | CONTINUE / REJECT (a definite FAIL) / PULL (any doubt, the guard band) | `pull.*`, `result.json` `disposition`, `pull_reasons` |
| feedback | the disposition painted ON the part: green outline = PASS; red fill + a cross at each fault = REJECT; an amber frame round the part + "PULL", amber on the zones that caused it = PULL | `feedback.*` |
| outputs | run folder (`result.json` + `run.log.jsonl` + overlay + DMD image), PASS / FAIL / PULL / ALARM + a result-code register on the I/O bus | `outputs.*`, `stations/io_example.json` |

```json
{"schema_version": 1, "name": "lid_present", "preset": "bench_9in_metric",
 "zones": [{"id": "lid", "shape": "rect", "units": "part_mm", "coords": [-40, -25, 40, 25],
            "checks": [{"type": "presence", "params": {"min_height_mm": 20}},
                       {"type": "flatness", "params": {"tol_mm": 0.5, "min_height_mm": 20}}]}]}
```

**The projector is also the work-state indicator.** After the verdict the DMD lights the part itself: a misaligned tube gets its misaligned section filled red with a cross where it is furthest off, a missing part leaves its zone red, a good part is outlined green. The operator sees where to act without reading a screen, which shortens recovery. The light goes where the scan's own decoded correspondences say each camera pixel was lit from, so it lands on the surfaces the camera saw at any depth; a zone whose camera -> DMD map is not consistent (local affine residual above `feedback.max_fit_px`) is not drawn, and the run says why. The hold is a normal pumped HDMI hold; ABORT clears it, and on the operator page so does the next command (START, STEP, TRIGGER, HOME, another configuration or profile). In a flow the `project_fault` action paints it: the fault of the work state that was not confirmed.

```bat
scan.bat --mock --library library_mock --preset bench_9in_metric calibrate-stereo --mock-poses 12   :: once
scan.bat --mock --library library_mock --mock-part scene:box_lid:ajar recipe-run box_lid
scan.bat recipe-run my_part --operator-confirmed           :: exit 0 CONTINUE, 1 REJECT, 5 PULL, 2 refused, 4 crashed
scan.bat run stations\example_recipe_box_lid.json          :: a single-job station config verified by a recipe
```

| | CLI | Python |
|---|---|---|
| list / show / check types / validate | `recipe-list`, `recipe-show NAME`, `recipe-checks`, `recipe-validate NAME` | `api.recipe_list()`, `recipe_show`, `recipe_checks`, `recipe_validate` |
| add (copy + override) / select / delete (to trash) | `recipe-add NAME --from BASE --set key=value`, `recipe-select NAME` (a built-in or library recipe name only: `recipe-add` a file first), `recipe-delete --reason` | `api.recipe_add`, `recipe_select`, `recipe_delete` |
| golden (a good part) / run | `recipe-golden NAME` (recorded as the recipe's golden), `recipe-run NAME [--scan ID] [--no-feedback]` (`--scan ID --no-feedback`: an offline re-check of a stored scan, no confirmation needed on the real rig: nothing is captured or projected) | `api.recipe_golden`, `api.recipe_run(recipe, preset=None, operator_confirmed=False, project_feedback=True)` |

Built-in examples (all runnable in `--mock` with `--mock-part scene:...`): `box_lid` (presence, height, flatness per lid half), `flat_plate` (flatness, local defects, edge straightness), `cylinder_end_faces` (end-face dents, height difference between faces), `ribbed_tube_alignment` (section position + edge position vs the golden), and the recipes of the example tube line's two profiles, `tube_long_*` / `tube_short_*` (`_presence`: a tube on the seat and this profile's height; `_alignment`: position vs the golden; `_end_face`: dents; mock scenes `scene:tube_long` / `scene:tube_short` with `:absent`, `:misaligned`, `:dent`). Schema and run folder: [`recipes/README.md`](recipes/README.md). Porting to another line (projector, camera, calibration, recipe, station configuration, I/O, and how to prove the numbers on site: calibration check, known-object check, Gauge R&R): [`docs/PORTING.md`](docs/PORTING.md).

### Zones: paint them on the part

The projector is a second monitor, so the Windows mouse cursor is **projected onto the real part**. The zone painter turns that into a drawing tool: move the cursor onto the part, drag a rectangle (or an ellipse, or click a polygon) over the area that matters, fit it, add more, confirm. The shapes are a **zone set**: where the scanner lights, what a scan counts, where a recipe looks.

**Workflow: paint → confirm → scan → reuse in recipes.**

```bat
scan.bat --preset bench_9in_metric zone-paint --name lid_zones    :: draw on the part; SPACE saves, ESC cancels
scan.bat zones                                    :: the saved sets (zones lid_zones: one set's zones)
scan.bat zone-show lid_zones                      :: project it again, photograph, where each zone lands
scan.bat scan-fast part7 --zones lid_zones:Z1,Z3  :: light + decode only those zones; points per zone
scan.bat measure-run bench_9in_metric --what flatness --zones lid_zones:Z1
scan.bat zones lid_zones --recipe                 :: the zones as a recipe's "zones" (--by-ref: references)
scan.bat zone-delete lid_zones --reason "fixture changed"         :: to the library trash
```

1. **Paint.** `zone-paint --name SET` opens the engine's projector window with the mouse enabled (the real rig: the machine-wide rig lock is taken first; `RigBusy` and nothing is shown when another program has the rig). What you see on the part is thin bright outlines, the handles of the selected shape, each zone's id (`Z1`, `Z2`, ...) and name, a crosshair at the cursor and a small legend strip at the bottom edge of the DMD image, on black: never a lit field, so the part stays visible. `--from SET` starts from a saved set.
2. **Confirm.** `Space` saves the set (`Esc` cancels, nothing is written). One photo of the outlines on the part is stored with it. An existing name is refused before the window opens unless `--replace` (the old set moves to `trash/zones/`).
3. **Scan.** `scan-fast LABEL --zones SET[:Z1,Z3]` lights only those zones (the lit mask is their union; every other mirror is dark, so less stray light) and reports the points decoded **per zone** (also `zones.json` in the scan's library entry). With `--zones` the scan decodes both DMD axes (`--axis xy`) unless you ask for `--axis x`: with columns only, zones that share DMD columns cannot be told apart, and their counts are reported unknown rather than guessed. `measure-run ... --zones SET:Z1` (Measure mode) takes the zone as the region: only it is lit, and the camera px that decode into it are measured.
4. **Reuse in recipes.** A recipe zone can refer to a painted zone, and a recipe's scan can be lit through a set:

```json
{"schema_version": 1, "name": "lid_painted", "preset": "bench_9in_metric",
 "scan": {"kind": "fast", "axis": "xy", "zone": "zoneset:lid_zones"},
 "zones": [{"id": "lid", "ref": "zoneset:lid_zones/Z1",
            "checks": [{"type": "presence", "params": {"min_height_mm": 20}}]}]}
```

A `ref` stays a reference: paint the set again (`--replace`) and the recipe looks at the new shape. `scan zones SET --recipe` prints the same zones with their coordinates copied instead (`projector_px` rect / circle / polygon), `--mask-out FILE.npy` writes the union as a `scan-fast --mask` file.

| key / mouse | does |
|---|---|
| drag on empty space | a new shape with the current tool: rectangle (default), `E` ellipse |
| `P`, then clicks | polygon: one click per corner; double-click, `Enter` or a click on the first corner closes it |
| click a shape | select it (the smallest shape under the cursor); `Tab` selects the next one |
| drag inside / a handle | move / resize (a rectangle: its edges resize too; a polygon: drag a corner). A small shape keeps a middle that moves it (the pick radius of its handles shrinks with it) |
| wheel, `[` `]`, `{` `}` | rotate the selected shape: 5° a notch (Shift: 1°), 1°, 15° |
| arrows | nudge the selected shape 1 px |
| Shift + drag | a new shape even when the drag starts inside one |
| `R` / `E` / `P` | tool: rectangle / ellipse / polygon |
| `N` (Shift+`N` back) | cycle the selected zone's name: part, datum, end face, tube, box, edge, fixture, and back to no name (`--labels a,b,c` sets your own: letters, digits, space, `_ . -`, checked before the window opens) |
| `Delete`, `Backspace` | remove the selected zone (while drawing a polygon: its last corner) |
| `Z`, `Ctrl+Z` | undo |
| `G` / `H` | lettered grid underlay (the cells of `scan grid`, at the spacing of the grid last projected for this geometry: the cells `--zone C4` means) / legend strip on, off (a zone or the cursor in the strip's rows is drawn over it) |
| `Space` | confirm and save |
| `Esc`, right click | drop the shape being drawn; `Esc` again cancels without saving |

A key or the wheel that edits the selected zone (`N`, `[` `]`, the arrows, `Delete`, `Tab`) while it is being dragged ends the drag where it is; `P` during a drag drops the shape being dragged out (a polygon is clicked). A quick second press (which Windows reports as a double click) is a press: click a zone, then at once drag it.

| | CLI | Python | Web (`/quick`, Lights) |
|---|---|---|---|
| paint | `zone-paint --name SET [--from SET] [--replace]` | `api.zone_paint(name, source=None, replace=False)` | **Paint zones** (a new name, or the picked set edited); the page shows the saved photo / preview |
| list / show | `zones`, `zones SET[:IDS] [--recipe] [--by-ref] [--mask-out F.npy]` | `api.zone_sets()`, `api.zone_set(spec)`, `api.zone_recipe(spec)`, `api.zone_mask(spec)` | the zone-set picker (`GET /api/zonesets`) |
| project + where it lands | `zone-show SET[:IDS] [--hold S]` (exit 1 when a zone is not seen by the camera) | `api.zone_show(spec, hold_s=5)` | **Show zones** |
| scan / measure | `scan-fast LABEL --zones SET[:IDS]`, `measure-run P --what ... --zones SET[:IDS]` | `api.scan_fast(zones="SET:Z1,Z3")`, `api.measure_run(..., zone="zoneset:SET:Z1")` | the picked set is the lit + measured region of MEASURE |
| delete (to trash) | `zone-delete SET --reason TEXT [--force]` | `api.zone_delete(name, reason)` | `zone_delete` job (`confirm=true`) |

Anywhere the engine takes a lit zone as text (`--zone`, a recipe's `scan.zone`, a measurement's `zone`) a set is `zoneset:<set>`, `zoneset:<set>:Z1,Z3` or `zoneset:<set>/Z1`. The advanced web page has no picker for sets yet: type that text into *Grid & zones → Scan this zone* (it is sent as typed, never pinned to a grid).

**What is saved:** `<library>/zones/<geometry_id>/<name>/` with `zones.json` (the shapes: rect / ellipse `center`, `size`, `rotation_deg`; polygon `points`; all in projector px `[col, row]`, plus the geometry and its fingerprint), `masks.npz` (`union`, `ids`, one bool 1140 × 912 mask per zone), `preview.png` (the zones as a DMD-space picture) and `camera.png` (the photo). A mirror belongs to a zone when its centre is inside the shape.

**Mock / scripted sessions.** In `--mock` the painter runs headless from a list of events, so a session can be replayed in tests and docs: `scan.bat --mock --preset bench_9in_metric zone-paint --name demo --events demo` (a built-in session) or `--events session.json`:

```json
[{"type": "drag", "from": [180, 260], "to": [420, 470]},
 {"type": "key", "key": "e"}, {"type": "drag", "from": [240, 620], "to": [460, 800]},
 {"type": "key", "key": "p"}, {"type": "click", "at": [560, 620]}, {"type": "click", "at": [760, 660]},
 {"type": "dblclick", "at": [720, 840]}, {"type": "key", "key": "space"}]
```

All of the interaction is a display-free state machine (`engine/zones/painter.py`: `PainterModel` events → state, `render(state)` → the DMD image); the window is a thin shell around it.

**Limits.**

- Zones are in **projector px**: they are valid for the rig placement they were painted at. The set is bound to the preset's `geometry_id` and its geometry fingerprint: a set painted at another geometry or other rig settings is refused (scans), or leaves the recipe zone unresolved (its checks INVALID, the part PULLed) - never used at the wrong placement. What cannot be detected is a rig or a fixture that moved under the same `geometry_id`: **re-paint, or re-home the rig and check with `zone-show`, after the rig moves.**
- **Camera px are derived per scan.** Which camera px a projector zone covers depends on the depth of what its mirrors hit (parallax between the projector and the camera), so a zone has no fixed camera-px outline: per-zone counts, Measure regions and recipe zones take the camera px whose decoded (col, row) is in the zone, from that scan's own decode (both DMD axes). `zone-show` reports where the zones land for the scene as it is now.
- The painter holds the rig (and its lock) while it is open. It gives the rig back after `--idle-timeout` (600 s) without input, saving nothing; on the operator page ABORT ends it.
- Keys go to the projector window: click on it once so it has the keyboard focus. The window must cover the projector display exactly (912 × 1140 at the preset's `monitor_offset`, 100 % display scaling), as for every scan. A key pressed while the picture is being redrawn (the mouse is moving) is kept and handled in order, never dropped.
- Polarity: like the grid, the painter is sent raw after a raw black vs raw white measurement, so "bright" is physically bright whichever way the HDMI path lands. That measurement shows one lit field for one hold before the drawing starts; `--polarity normal|inverted` skips it. When the measurement cannot tell (white and black look the same to the camera: projector off, not aimed at the scene) the painter does **not** start - on an inverting path a wrong guess would hold a lit field on the part for the whole session: fix the projector, or say the polarity.
- The picture is lines on black: about 2 % of the mirrors in an ordinary session. It grows with the drawing - the most a set can hold (64 zones that each span the field, grid on) lights about 40 % of the mirrors at half level or more: still outlines, never a filled field.
- A zone is at least 8 DMD px each way, a set has at most 64 zones, a polygon does not cross itself and has at most 256 corners.
- **Status: mock-tested only** (`tests/test_zones_*.py`: the state machine, the scripted sessions, scans and recipes on the simulated bench). The painter has not yet been run on a real rig: the first session there should start with `zone-show` on a simple set.

## Modes (UI tabs ↔ verbs)

| Mode | Verbs | Notes |
|---|---|---|
| Setup | `load_preset`, presets | Toggles for camera (exposure, gain, averaging), projection (duty, invert, rotate90, gamma, offset) and **rig geometry** (projector left/right/above/below, baseline, projection distance, throw ratio). Shows the live nominal mm/px and triangulation angle. |
| Focus & signal | `check_signal`, `tune`, `measure_gamma`, live view | Live focus score. The **orientation check** flags fringes parallel to the baseline. The scan assistant recommends exposure; prefer whole 60 Hz frames (16 667 µs). |
| Base library | `capture_base`, `reject_base`, `delete` | An empty-scene datum per geometry. `zone=` makes a zone-lit base. Bases under 5% valid are auto-rejected. |
| 3D scan | `capture_object`, `reconstruct`, `profile` | Relief vs a base, `.ply` point cloud with a 3D viewer, and **2D profiles** with `step_height` (for end-face height). |
| Measure | `calibrate`, `measure` | px → mm from known heights / widths (gauge blocks flat on the base). |
| Defects | `inspect`, `compare`, `bank_add`, `bank_check`, `bank_remove` | Local blobs vs the 0.7 mm trigger. `inspect` only says PASS when the noise lets it see a trigger-sized defect. Compare against a golden scan, or the **good-memory bank** (flags anything unlike known-good parts). |
| Codes | `read_codes` | QR/1D/Data Matrix (OpenCV + zxing-cpp, every decoder on the raw, CLAHE and 2× CLAHE image) under bracketed brightness and **red**/green/blue/white light; one entry per symbology + data (`EAN_13` = `EAN13`), and an EAN-13 starting with 0 is reported as the 12-digit UPC-A printed on the label (`raw` keeps the EAN-13). Needs about 2.5–3 camera px per module. Exit 0 when a code is found, 1 when none. **Experimental:** `codes --anti-glare [--strip-cols C0,C1]` for a glossy label with a saturated specular stripe: many-level flood bracket → HDR (unsaturated samples, frames ordered by measured brightness), optional narrow DMD column-strip sweep (each pixel keeps its own strip's energy: no bloom), de-veil (local min/max envelope, ~9 × 3 px: gloss is an additive offset) → the normal decoder (`engine/sense/antiglare.py`). Mock-tested on synthetic glare; one bench trial. |
| Spectral | `spectral_capture`, `learn_material`, `classify`, `separate_light` | R/G/B LED reflectance signatures → material map. Direct vs global light from shifted checkerboards. |
| Grid & zones | `capture_map`, `project_grid`, `project_zones` | A lettered grid (A1, B2, …) photographed with upright labels. Say "H8:I11@96" to light and scan just those cells. Custom machine-vision zones. |
| Custom patterns | `project_capture` | Your own DMD images (1140 × 912 uint8, grey or 3-channel, at most 256) projected one by one through the normal rig (HDMI window only, never USB; the rig lock; pumped holds) and photographed in order. `raw=True` sends them as given (no duty / polarity / gamma chain); `average` / `hold_s` override the preset for this run only. `frames.npz` (+ the patterns) and `meta.json` under `library/readings/`; no decoding. CLI `project-capture <npz | folder of PNGs> --label L [--average N] [--hold S] [--raw]`. |
| Sensors | `spot_size`, `project_spot`, `teach_presence`, `check_presence`, `teach_alignment`, `check_alignment` | A laser-spot-sensor-style spot of a chosen **diameter in mm** (nominal or your **measured** mm/mirror). **Part present** (photoelectric, or background suppression by triangulation). **Alignment** (dx/dy/dθ vs a taught pose). |
| Cylinder & 3D | `export_mesh`, `cell_unwrap`, `cell_dents`, `bake_maps`, `checker_scan` | STL/OBJ (relief, base, open-arc cylinder), cylinder/tube unwrap → dent map + end-face height, surface-map stack (cavity, normal, albedo, subsurface, dust). |
| Surfer + fast mode | `import_atlas`, `flood_verify`, `anti_glare_sweep`, `multi_orientation_scan`, `make_flash_pack` | Atlas regions as zones (`atlas:part/part`), flood-verify with a black surround, glare-dodging patterns, H/V/diagonal fusion; flash pattern packs for the triggered fast path (`docs/FLASH_PATTERN_MODE.md`). |
| Flash fast scan (1-bit, no trigger cable) | `make_fastscan_pack` | The column-axis multifreq set (4 steps × T 8 / 80 / 1280 + white / black) as 16 one-bit bit-planes of ONE 24-bit 912×1140 flash image (G0–G7, R0–R7), with two sync markers; full-field + zone-masked images (face ellipses + margin, a camera band mapped to the DMD). Fine level: column-exact ordered dither; mid / coarse: Floyd-Steinberg strips, decoded after a 3 px camera low-pass. The DLPC350 plays it on its internal timer (10 ms per slot, 160 ms loop) and the free-running camera is sorted in software, ~0.5 s per scene. Files only: a human loads the images and plays the sequence in the 4500 MKII GUI (`docs/FLASH_FASTSCAN_4500.md`: backup via the GUI's firmware link, firmware build, sequence settings, LED limits, return to video mode, rollback). CLI `fastscan-pack`; `tools/make_flash_pack.py --fastscan [--zone-ring NPY]… [--zone-band-cam NPY --zone-band-field NPZ] [--calibration DIR] [--study]`, `--unpack-check DIR`. |
| Flash soft-sync scan (camera only) | `fast_scan_flash`, `sort_flash_recording` | While a HUMAN plays the fast-scan pack in the 4500 MKII GUI (loaded with its firmware link; Pattern Sequence, Flash, Internal/External, Play Once off), the engine opens ONLY the camera: free run, a window around the zone (`roi` / `zone`, or the pack's zone through a projector map), fixed exposure (3 ms), the window's top frame rate, OldestFirst buffers, every frame with its camera timestamp and frame id. The frames are sorted in software: loop period by folding, pattern boundaries from the frames that straddle a change (fitted to µs), clean frames only (exposure inside one pattern, 250 µs guard), the sync markers + content checks anchor the pack's order, then the clean frames are averaged per pattern → the unchanged multifreq decode (mid / coarse prefiltered; px where that low-pass mixes two surfaces, and in the zone image px on the zone's edge found with the full-field sync white, are left undecoded; optional `fine_prefilter_px`) → ray-plane mm → a normal metric scan (meta `mode: flash_softsync`, per-pattern frame counts, confidence). The pack's manifest must describe exactly the loop the engine sorts (pattern version, dither, slot k on bit k = G0–R7, timing), else it is refused before the camera. Refused with the reason when the loop is not playing, stopped, misses / adds a pattern, plays out of order, runs another period, or the camera dropped frames. 10 ms slots: 2 loops = 0.34 s of frames in a zone window (321 fps), 0.51 s full frame. `dry_run` = the timing plan; the raw frames go to `output/flashscans` for `sort_flash_recording` (offline). CLI `scan-flash`, `sort-flash`; `docs/FLASH_SOFTSYNC.md`. Mock-tested only. |
| Metric calibration | `make_charuco`, `row_parity`, `capture_calib_pose`, `calibrate_stereo` | Printable ChArUco board (true-size PDF), diamond DMD row-parity check, board poses (flood + both-axis `multifreq` decode), then camera + projector (inverse camera, native DMD col/row, f_v ≈ 2 f_u) + `stereoCalibrate`. Reports per-pose and per-axis RMS and, per pose, the corners that got a projector position (a pose warns under 60 %; the default fringe periods are retried coarser; one period set per pose set); saved as a geometry-bound calibration - or rejected with nothing saved (`CalibrationRejected`) when the projector fit is degenerate, the stereo RMS is above 2 px or too few poses gave projector points. `rig_check(rehome=True)` writes a re-homed copy (rotation only, the original kept) after a small camera nudge. |
| Metric scan (mm) | `scan_metric`, `triangulate`, `export_metric_mesh` | Point cloud in mm (ray-plane from columns, ray-ray with both axes + a per-pixel consistency-error map), `.ply`, open-surface STL/OBJ in mm. Refused without a stereo calibration at the same geometry and fingerprint. |
| Metric measure | `cylinder_deviation`, `compare_metric`, `accuracy_check`, `predict_precision` | Cylinder fit → radial deviation map in mm with dent depth/area, point-to-plane ICP to a golden scan, known-object check (plane / step / cylinder / sphere: mean/max error, repeatability, GO/NO-GO), predicted σ_z before scanning. |
| Cylinder dents (shiny parts) | `capture_dent_set`, `calibrate_dent_null`, `detect_dents` | One pose = a long (66 667 µs: floods, sin0 T12/T36, sin90 T12/T36) and a short (16 667 µs: floods incl. 128, the same sinusoids, lines_v) pattern pass + one metric scan of the zone (`pattern_set="minimal"` drops sin90 T36: fewer frames, but it scored a small dent lower in the bench ablation). The two exposure presets may differ only in exposure. A fixed-radius cylinder from the metric cloud maps pixels to (θ, s) on the cylinder (`s_mm` from the axis point level with the camera's optical axis, `s_from_top_mm` from the top of the decoded cylinder). Four finders (pattern warp, photometric, HDR phase depth, reflected-line kink) are calibrated on a **known-good part** of the same type in the same fixture (1.0 = its strongest response; a finder that does not respond there is excluded, with a warning) and fused: detection > 1.25 (a margin over the one good part's own maximum), watch 0.8–1.25, edge. Each candidate: px + (θ, s), score, depth in mm (+ = pushed in; flagged unreliable in specular / dark-band zones, significant above 1.5 × the scan's depth-noise p90), size, per-finder evidence, kind dent / scratch / speck (advisory), overlay PNG. Legacy offline dent sets (`dataset.json`) work through the engine API and CLI (not the web UI). Bench numbers: the re-placed pose ("pose 2") was used while tuning, so it is not held-out data. |
| End faces (cylinder end face toward the camera) | `locate_face`, `locate_scan`, `locate_face_live`, `aim_ring`, `check_ring`, `auto_intensity`, `face_exposure`, `targeted_face_scan`, `face_profile`, `face_golden`, `check_face` | A cylinder lying with its circular **end face** toward the rig. `locate_face` on a ray-plane metric scan: the face is the nearest face-shaped depth-continuous surface (nearer non-face objects are set aside), a neighbour only a few mm behind the face (a ribbed tube) is split off by the face's own plane (sequential RANSAC planes; rim = surface at least `near_gap_mm` = 2 mm behind that plane), and when the depth rim still fails the face outline in the flood (the scan's flood pair, else the decode modulation) is used (`rim_source` in the target), its rim = contour points against far background (the true silhouette, not an undecoded band), a fixed-radius circle (the known diameter) in the face plane, the camera face mask, and a camera → DMD **affine** fitted on the decoded face pixels (residual reported, warning > 0.5 DMD px; the calibration's projector row axis is not used for aiming). `aim_ring` projects the ring + cross (the operator's placement guide, pumped hold); `check_ring` says ON_FACE / OFF (face flood coverage + where the ring lands). `auto_intensity` picks `projection.duty` for a ~200 DN face at a whole-frame exposure (never the LED currents). `targeted_face_scan` lights only the face ellipse + margin: N repeats of the depth set (ray-plane) + polar rings / spokes / floods, at a **frame-locked camera exposure chosen for this seating** (auto exposure: the face flood's p95 inside 0.8 R near 170 DN with < 0.5 % clipped, 16.7–100 ms in whole video frames, a per-capture camera override - the preset is not edited; works with inverted presets; when no exposure fits, the face is not scanned: PULL). `face_profile`: height map in the face frame (0.25 mm grid; u right, v down, + toward the camera), radial profile, sections, flatness, repeatability, PNG / CSV / JSON / STL. `face_golden` (append-only, bound to geometry, face diameter, grid, stereo calibration) holds every detector's model; default detector **MF-CFAR + reflectance FIR** (`mf_fir`): per-repeat planes removed, deviation from the good placements' median **radial** profile, a dent-shaped matched filter bank (1 / 1.5 / 2 / 3 mm) with a local plane, z against the face's own robust noise (radial-band MAD), trigger at z ≥ 5.5 inside the inspected zone (r ≤ 0.926 × the face radius: 9.75 mm on a 21.05 mm face - **every radius derives from the face diameter** unless given, one rim radius for every detector, the resolved set stored with the golden and in each check); first the height error predicted from the flood pair by a 7×7 camera-px FIR fitted on the good placements (coefficients stored with the golden; not applied to a placement whose flood saturates > 2 % of the face) is removed. `mf` = without the FIR; `band` = the prototype (median map, band max(4σ, 0.10 mm), rim ×2). Leave-one-out false triggers of every detector + a synthetic-dent set (dents injected per camera pixel). `check_face` → dents (zone, r, θ, `depth_centre_mm` = the 3 × 3 grid-cell average at the dent's deepest place against the local background: THE depth, thresholds apply to it; `depth_min_mm` = the single deepest point, which over-reads; `width_half_depth_mm`; z, area), **edge dents** (zone `rim_outline`: a notch / bulge of the flood outline beyond 0.4 mm over ≥ 1.5 mm of rim), the **known-size check** (the outline's diameter vs the face diameter: > 4 % off = `size_mismatch`), PASS / FAIL / INCOMPLETE (< 90 % of the face measured, a size mismatch or an unjudged rim is never passed) + the disposition, overlay PNG; `detector=` picks another detector or a band trigger function. Bench (2 good parts, 4 placements): flatness 0.044–0.070 mm RMS (r < 9 mm), repeatability 0.011–0.051 mm, placement pairs 0.050–0.076 mm RMS; leave-one-out: `mf_fir` 0 false triggers (band 5 + 3 rim), detection at z 5.5 ≈ 45 % at 0.15 × 1.5 mm, 78 % at 0.15 × 2 mm and 0.20 × 1.5 mm, 89 % at 0.20 × 2 mm; 1 mm wide dents are below the camera's resolution (0.38 mm/px). Reliable: ≥ 0.2 mm deep × ≥ 2 mm wide, or ≥ 0.15 × 3 mm. |
| HDMI fast zone scan | `scan_fast` | The column-axis multifreq set only (default 4 steps × T 8 / 80 / 1280 DMD px + white / black = 14 patterns) lit through a zone (grid cells, an atlas region, a DMD mask `.npy`, and / or located faces' DMD ellipses + a margin), with the camera average and the HDMI hold overridden for this run only (the preset file is untouched; average < 2 is raised to 2 with a warning: over HDMI one frame per pattern flickers), ray-plane triangulated (the column axis carries the depth) → a normal metric scan that also keeps its flood pair (`flood.npz`) for the station measurement. Bench: average 2, hold 0.12 s = 5.3 s for the whole station scene. CLI `scan-fast`. |
| Station (cylinders + ribbed tube) | `station_measure`, `station_golden`, `station_check` | Cylinders lying with their end faces toward the camera, held in a ribbed/serpentine tube (the tube edge: a thin ribbon in a trapezoid wave, seen edge-on a few mm behind the faces). From one metric scan + a flood pair (the scan's own from `scan_fast`, another scan's, a file; else the decode's fringe modulation, with a warning): cylinders C1..Cn (image order) found on the flood (Hough; brightest discs that are flat and proud in depth), face planes (robust), face centres (sub-pixel flood rim edges back-projected onto the face plane, circle fitted there; `centre_method="ellipse_ray"` = the prototype's ray through the rim ellipse centre), the common face frame; height of each face vs the mean face plane (mm, + toward the camera), `Cj-Ci` along the common normal, spacing in the face plane. Tube front edge: thin runs per camera column a few mm behind the face plane, outside every cylinder body (height band found automatically, or `ct_band_mm`), a trapezoid-wave fit of its centreline (pitch, amplitude, plateau, phase vs the cylinder-pair centre), part-fixed zones T1..Tn (one plateau + two half-ramps) with the edge height per zone (median / p5 / p95). PNG + JSON + CSV in `output/measurements`. `station_golden` (append-only, bound to geometry + fingerprint + stereo calibration): nominal values (mean of the good measurements), the zone model + numbering, user-set limits (tolerance around nominal and / or absolute bands, per item if wanted), and each end face's position. `station_check` re-measures with the golden's parameters and zones → PASS / FAIL / INCOMPLETE (a golden item not measured is never passed; every end face must lie within `face_position_tol_mm`, default the face diameter, of its golden position, so a disc found elsewhere - a background disc with the part missing - FAILs; a golden that compares nothing is INCOMPLETE) with the triggered items, PNG + JSON + CSV in `output/defects`. CLI `station`, `station-golden`, `station-check`, `stations`. Bench reproduction (saved scans): with the prototype's settings, its end-face height difference and spacing to ≤ 0.001 mm and most tube zone heights to ≤ 0.005 mm (largest gaps 0.04 mm on one scan and 0.12 mm on the other: the prototype pooled its tube-edge mask over both scans); the fast scans (average 2) repeat the slow reference within 0.03 mm (C2-C1) and 0.11 mm (tube zones); a real re-seat (0.27 mm) and a tube move (-3.2 mm) FAIL. These are repeatability and agreement numbers, not absolute accuracy: no known-size object has been measured through the station path yet. |
| Operating modes | `measure_run`, `measure_to_check`, `quick_check_config`, `rig_session`, `rig_state` | Measure (numbers with uncertainty, no verdict), Quick check (one check through the station job runner) and the rig's state for mm results (fixed / mobile mount): *Operating modes*. CLI `measure-run`, `measure-to-check`, `quick-check`, `rig-session`. |
| Output bin | `list_library("outputs")` | Every result, by type, with absolute paths. |

Every numeric result carries `units` ("px" or "mm") and `warnings`. mm needs a calibration at the same geometry (rule 6).

**Dent path (shiny cylinders).** Stereo calibration first (the metric path below) → `dent-scan` a KNOWN-GOOD part in the fixture's zone (`--preset` the dark 66 667 µs preset, `--short-preset` the 16 667 µs one) → `dent-null <it> --diameter MM` → `dent-scan` each part → `dent-detect <it>` (exit 0 CLEAR, 1 FLAG, 4 the detector failed; the cylinder diameter defaults to the matching null's). A null calibration is bound to the geometry, the cylinder diameter, the fixture (the zone, or a name given with `dent-null --fixture` that `dent-detect --fixture` must then repeat), the pattern set and the stereo calibration; recalibrate when any changes. `--subset production` evaluates a richer legacy capture with only the production patterns.

**End-face path.** Stereo calibration first → the human places the part → `face-locate --zone H10:I12 --diameter D` (D = the diameter of the bright flat the flood shows; `--outer-diameter` = the part's outer diameter when the rim around the flat returns depth points, else found from the rim; the locate scan keeps its flood pair and is repeated at a longer / shorter frame-locked exposure when the located face is dark / clipped, the scans of the seating then merged per pixel - the longest exposure that did not saturate it - so a bright surround that saturates at a dim face's exposure is still what the face is found against; or `face-locate <metric scan>`; the scan must decode x **and** y: `scan-fast --axis xy` or `scan-metric`, an x-only `scan-fast` is refused with that hint) → `face-aim --check` (exit 0 ON_FACE) → `face-scan cN` (auto exposure by default when the preset's exposure is whole video frames; `--exposure-us N` forces one, `--no-auto-exposure` keeps the preset's, `--auto-intensity` tunes the duty first; exit 5 and nothing stored when no exposure fits this seating) → `face-profile cN`. Golden: `face-golden g1 g2 g3 g4` from GOOD placements (MF-CFAR + reflectance FIR by default; `--detector band`, `--z 5.0` for ~10× more false triggers and ~10 points more detection); per cell `face-check cN` (exit 0 PASS, 1 FAIL, 5 PULL: INCOMPLETE, a size mismatch, a rim outline in its guard band or not judged all the way round, a profile off its outline's centre). Keep the flood unsaturated (`--auto-intensity`, which needs a preset with `projection.invert` false, e.g. `--preset bench_9in_metric_bright`; on an inverted preset it is refused with that hint): a saturated face is noisier and loses the FIR. `face-aim --hold 30` keeps the ring up while the next part is placed; `faces` lists targets, face scans and goldens.

**End-face limits (bench 2026-10-12: a 21 mm class cylinder whose flat end face is 19.5 mm across inside a rounded rim; projector top-down, camera ≈ 18°).**
- *The face diameter is the flat's.* Give `--diameter 19.5`, not the cylinder's 21: the flood shows the bright flat, the rounded rim is dark. Every detector radius derives from it (inspect / fit 0.926 R, rim / plane 0.855 R, noise bands scaled; `face-golden --inspect-r / --rim-r` override, `--rim-r` is one radius for every detector; a radius given is used as given or refused when it does not fit the face - never dropped, never clamped: `face-profile --r-flat 9.0` is 9.0).
- *The centre is the flat's too.* When the rounded rim returns depth points, the depth silhouette is the cylinder's 21.05 mm circle, and a 19.5 mm circle fitted to it slides until it touches one side: the located centre landed 0.75–0.83 mm off (mock model of the part), the inspected disc then reached the rim on one side, and 3 of 10 good placements were REJECTed. So `face-locate` fits its rim circle at the silhouette's own size - the outer diameter given (`--outer-diameter 21.05`), else the free circle through the rim points when they cover ≥ 150° and lie on a clean circle 2–20 % larger than the face (said in the result: `silhouette_diameter_mm`) - and `face-profile` centres the face frame on the **flood outline of the bright flat** whenever the whole outline shows (`stats.centre`: how far that was from the located circle; more than 0.035 × the face radius is said). A profile that is not centred on its outline (one stored before this, a face that leaves the camera frame) is never passed and never a definite FAIL (PULL), and a golden refuses it. Limit: a rim brighter than half the flat reads as part of the outline - the size check then says "large" (PULL), it is not inspected as flat. With the radii of a 21.05 mm face on the 19.5 mm flat the detector inspected the rim: good parts read dent z 5.8–8.2 against the 5.5 threshold (false REJECTs); scaled, five good placements read max z 3.4–4.3 and the dented part 9.2. On smaller faces the inspected zone ends less than 2 camera px (0.36 mm/px) inside the flat's edge: read the golden's leave-one-out, and pull `--inspect-r` in if good parts trigger at the rim.
- *Exposure depends on the seating.* The same bare-metal face needed 33 / 50 / 66.7 / 83 ms: too dark 6–10° off the mirror direction at 33 ms (1 % decoded), saturated at 83 ms when it mirrors the projector, and 2 % clipped pixels were enough for a false rim trigger. `face-scan` therefore picks a frame-locked exposure per face (above) and refuses a seating no exposure fits (too dark at 100 ms, or a glint that clips before the face is bright enough: re-seat the part).
- *Known size.* A lens zoomed after the calibration made every lateral size read 0.84× (depths 0.70×): a 19.5 mm face read 16.3 mm. Wherever a nominal size is known (the face diameter of locate / the golden, a station's `--diameter`, a recipe's circle zone or `nominal_diameter_mm`) the flood outline is held to it: > 4 % off (`--size-tol`) is a warning in every result and a PULL (`size_mismatch`: "part reads N % small / large: calibration or lens suspect - run rig-check / recalibrate"). Nothing is ever rescaled, and a FAIL found on a mis-scaled scan is not a definite FAIL; no recipe can turn a `size_mismatch` into a REJECT. A scan without a flood pair (`scan-metric`) is not checked at locate (said, also per face by the station); the targeted scan always has one. `--size-tol 0` switches the check off and every result then says so.
- *Edge dents.* The inspected flat stops short of the rim, so a dent on the edge shows as the flood OUTLINE moving: sub-pixel radial edge on 360 rays in the face plane, an ellipse removed (residual plane tilt), residual radius vs angle. Good parts: 0.06–0.09 mm rms, 0.20–0.28 mm max at 0.36 mm/px → a notch / bulge is > 0.4 mm over ≥ 1.5 mm of rim (`--rim-tol`, `--rim-arc`; within 20 % below the tolerance: PULL). The golden learns the outline's per-angle median and spread from the good placements. Shorter than 1.5 mm of rim (≈ 4 camera px) is reported in the max, not as a finding. A notch only a few camera px wide reads ≈ 0.1 mm shallow (the edge is a blurred px): 0.5 mm deep over 3 mm reads 0.43 mm, and one of exactly 0.4 mm read 0.31 mm on the mock bench - just under the guard band, passed.
  - *Wide dents.* The curve the residual is taken against is the one the best 60 % of the rays lie on (least trimmed squares), so a notch, a bulge or a flat is not part of it: 0.5–1.0 mm deep over 3–20 mm of a 19.5 mm face's rim reads its depth and FAILs, with the diameter and the centre unmoved (an ordinary trimmed fit bent into anything wider than ≈ 10 mm: 0.6 mm over 12 mm passed, 0.8 mm read as "5 % small"). Limit: a deviation over more than ≈ 40 % of the rim (≈ 25 mm of 61) is no longer told from the part's centre, size and ovality - only the size check (4 %) holds it.
  - *Deep chips.* A ray whose own level is under half its neighbours' takes their level (inside a chip it is only the light that leaks in), and a ray already dark where the search starts (0.72 × the radius) has its edge there: a notch "at least" that deep. Rays with no edge do not split a notch.
  - *Never passed unjudged.* ≥ 1.5 mm of rim with no edge (a glint within 3 px of the edge, a dark stretch with nothing out of tolerance beside it), an outline that shows only in part, or one with more than 20 % of its rim off the curve is `incomplete`: PULL, with or without a golden outline (a recipe's `rim_check: auto` too). Clipped flood pixels are never turned into a finding; only a flood that shows no outline at all (no flood pair, no bright disc on a dark surround) is skipped with a warning.
- *Dent depth.* The deepest single grid point over-reads (the dent plus the most negative noise at its bottom): the same dent read 0.57–0.65 mm deepest on two rigs, 0.49 mm as the 3 × 3 centre average, 0.46 mm by a depth gauge (1.33 mm wide; the scanner's width at half depth 1.5–1.64 mm: about one camera px wide). `depth_centre_mm` (= `depth_mm`) is the depth and what `--min-depth` / a recipe's `depth_mm` judge; `depth_min_mm` is kept, flagged "single point, over-reads".

**Fast face scan (experimental, method contributed by Grok (xAI)).** `face-scan cN --mode tilephase` (`targeted_face_scan(mode="tilephase")`) replaces the multifreq depth set's two coarse phase levels with 5 Gray-code tile planes per axis: 12 steps at T = 8 DMD px + 5 code bits per axis + white / black = 36 frames instead of 74 (about 2× faster; one bench session: no fringe-order errors against multifreq, the same dent depth within the repeat spread). The default is unchanged. Limits: the fringe order is unique only inside a 2^(bits−1) × 8 DMD px window (128 px with `--code-bits 5`) centred on the face, so it relies on only the face being lit (a face + margin that does not fit is refused); binary code planes are more sensitive to defocus, steep tilt and glints than sinusoids; only a two-code consistency check guards the order; motion during the code frames corrupts it. Mock-tested (decode equivalence with multifreq on the synthetic face); see `engine/bases/tilephase.py`.

**Station path.** Stereo calibration first → the human places the cylinders in the ribbed tube → `scan-fast st1 --zone F10:J12` (≈ 5 s over HDMI; keeps its flood pair) → `station st1` (end faces, `C2-C1`, spacing, tube zones T1..Tn, PNG) → golden from GOOD placements `station-golden st1 st2 --cell-tol 0.15 --ct-tol 0.3` (or absolute bands: `--cell-band LO,HI`, `--ct-band LO,HI`, `--band T3=LO,HI`) → per part `scan-fast` + `station-check <it>` (exit 0 PASS, 1 FAIL / INCOMPLETE, 2 refused, e.g. an end face not found, 4 crashed). Zone numbering follows the golden; `--skip-zones 1,0` (on `station`, or on `station-golden` for metric-scan items, with `--edge-band`, `--roi`, `--centre`, `--cells`, `--diameter`) drops a lead-in zone that is not part of the wave. A face cut by the zone edge keeps its height but is warned about: its centre, so the spacing, moves. A ribbed tube only a few mm behind a face no longer stops `face-locate` (plane split; flood-outline fallback). Strict end-face gates (default on): a candidate disc is set aside, with its reasons in the report's `set_aside` list, when its flood rim circle rms is > 1.0 mm (`max_outline_rms_mm`), or < 70 % of it is measured (`strict_min_coverage`) without a clean rim circle (rms ≤ 0.3 mm, `strict_partial_outline_rms_mm`; never below 50 %: real faces with glint or zone-edge dropouts measured 50-70 % with rims of 0.09-0.13 mm on the bench, the background patches seen there had no rim circle or one of 0.6-0.8 mm; one real face measured 69.9 % on the decode-modulation flood had a 0.49 mm rim and is refused - a looser rim lets a partly measured background disc near the faces' depth stand in for a face). The pair gates are judged over the whole set of candidate discs, not against a face accepted first: the set is chosen among the discs that pass the per-disc checks; any two faces within 2 × the diameter in height (`max_pair_dz_diam`) and at least 0.9 × the diameter apart (`min_pair_spacing_diam`); each face within 6 × the diameter of a neighbour (`max_pair_spacing_diam`), so a longer row is fine. A disc that passes every per-disc check but not the set, and is as bright as the faces or (at another depth) goes with another such disc or makes a set of its own, makes the run **ambiguous**; so does one that goes with the set - continuing its row (within the bound of one face, past it of another) or, with `n_cells` > 1, among its faces - when the pair gates let it go with every face (more end-face discs together than `n_cells`; only the brightness rule keeping it out counts too when it makes a full set with other discs), and two or more such discs at the faces' depth with no face within the bound (a group of their own): refused, both named (pass `cell_px_hint` or scan a tighter zone); a dimmer lone one is set aside. With `n_cells` = 1 the brightest disc of a compact group is the face. This stops a bright background patch on a deep scene being taken for an end face, or the part being guessed when a face of it is missing. Too few end faces left → refused with the reasons. `station-golden` accepts a measurement .json older than the gates (or made with `--no-strict`) together with gated ones by measuring its scan again with the gates (written to the output bin): the re-measurement stands in only when it finds the same end faces (warned, and the golden then checks with the gates); the golden is refused when the gates refuse that scan, find other end faces, or the measurement can't be measured again. An end face further from the measurements' mean position than the face position tolerance is refused, never averaged in. Known size (`size_tol_pct`, default 4 %; `--size-tol`): each end face chosen is held to `--diameter` by its flood outline (an ellipse-tolerant fit on ≥ 80 % of its rim, rms ≤ 0.3 mm; a face cut by the zone edge is not a size reading): its mean diameter within the tolerance, one axis of its ellipse alone within twice it (a small tilt of the disc's fitted plane moves one axis). One that is further off refuses the measurement as a size mismatch ("part reads N % small / large: calibration or lens suspect - run rig-check / recalibrate"), so a scan through a lens zoomed after the calibration is not measured and `station-check` exits 5 (PULL, reason `size_mismatch`); `--diameter` is the diameter of the bright flat the flood shows, and nothing is ever rescaled. Which discs are the end faces is still decided by the gates above, not by size. `station --no-strict` (or `strict: false`) keeps only the flat / proud checks; the end faces are then measured and flagged (`size_check` per face): the check is INCOMPLETE = PULL and a golden from such a measurement is refused.

**Metric path (stereo calibration).** `make_charuco` → print at 100 %, caliper a square → `row_parity` → `capture_calib_pose` × 12–20 (tilts ±30°; or `fixture=<spec>` × 3 placements of the printed calibration fixture, mock-tested only) → `calibrate_stereo --square MM` (camera RMS < 0.3 px, projector < 0.5 px per axis) → `accuracy_check` on a known object (or `accuracy_block` on the printed check block) → `scan_metric` → `cylinder_deviation` / `compare_metric` → `export_metric_mesh`. CLI: `charuco`, `parity`, `calib-pose`, `calibrate-stereo`, `scan-metric`, `triangulate`, `cyldev`, `accuracy`, `precision`, `metric-mesh`, `metric`; dents: `dent-scan`, `dent-null`, `dent-detect`, `dentsets`. In mock mode `calibrate-stereo --mock-poses 12` lets the simulator place the board (with `--preset` the preset the jobs use: the calibration is bound to its geometry).

## Where the data goes

Everything is under `library\` (the default is next to this README; override with `SWISS_SCANNER_LIBRARY`):

| Folder | Contents |
|---|---|
| `bases\<geometry>\` | **base library** (empty-scene references) |
| `objects\` | 3D scans: field, relief, meta, exports.jsonl |
| `calibrations\<geometry>\` | Z / XY calibration fits, stereo calibrations, diamond row parity, dent null calibrations (`dent_null.json`), rig references, and `rig_state.jsonl` (append-only, read in the order of its lines: every rig check's result, every calibration pose and calibration, the rig session markers) |
| `calib_poses\<geometry>\` | ChArUco board poses for `calibrate_stereo` |
| `metric\<geometry>\` | metric scans: field, cloud (xyz mm + consistency), meta |
| `dentsets\<geometry>\` | dent-detector captures: long / short exposure frames (cropped to the zone), meta naming the metric scan |
| `faces\<geometry>\` | located end faces: face plane, rim circle, camera face mask, camera → DMD affine + aim ring |
| `facescans\<geometry>\` | targeted end-face scans: face DMD mask, polar pattern frames, meta naming the repeat metric scans |
| `face_golden\<geometry>\` | golden end faces: MF radial profile + reflectance FIR coefficients, golden + band maps, placements' height maps, the detector settings, leave-one-out + synthetic set |
| `station_golden\<geometry>\` | station goldens: nominal end-face height differences, spacings, tube zone heights, the zone model + numbering, the limits |
| `banks\<geometry>\<bank>\` | good-memory banks |
| `maps\<geometry>\` | projector maps (camera ↔ DMD pixel) |
| `sensors\<geometry>\<name>\` | taught part-present and alignment sensors |
| `readings\` | code-reading frames |
| `spectral\` | captures, `materials\`, `light\` |
| `output\` | **the output bin** |
| &nbsp;&nbsp;`output\pointclouds\` | `.ply` / `.xyz` |
| &nbsp;&nbsp;`output\defects\` | inspect `.json` + annotated `.png`; `detect_dents` `_dents.json` / `.png` overlay / `.npz` maps; `check_face` `_face_check.json` / `.png` / `.npz`; `station_check` `_station_check.json` / `.csv` / `.png` |
| &nbsp;&nbsp;`output\measurements\` | measurements; `station_measure` `_station.json / .csv / .png` |
| &nbsp;&nbsp;`output\profiles\` | `.csv` + plot `.png`; end-face height maps (`_face.png/.json/.npz`, radial + section `.csv`) |
| &nbsp;&nbsp;`output\anomalies\` | bank checks |
| &nbsp;&nbsp;`output\comparisons\` | scan-vs-golden verdicts |
| &nbsp;&nbsp;`output\codes\` | decoded codes |
| &nbsp;&nbsp;`output\spectral\` | material maps |
| &nbsp;&nbsp;`output\previews\` | grid, zone, relief and spot images |
| &nbsp;&nbsp;`output\tuning\` | assistant and gamma reports |
| &nbsp;&nbsp;`output\calibration\` | ChArUco boards to print, stereo calibration and row-parity reports |
| &nbsp;&nbsp;`output\flashscans\` | flash soft-sync recordings (camera frames + timestamps + the loop layout) and offline sorts |
| &nbsp;&nbsp;`output\sensors\` | sensor checks |
| &nbsp;&nbsp;`output\index.csv` | one row per output file: time, kind, label, geometry, source |
| `recipes\` | user recipes (`<name>\recipe.json` + `meta.json`), `active.jsonl` = the selected recipe |
| `zones\<geometry>\<name>\` | painted zone sets (`zone-paint`): `zones.json` (shapes in projector px), `masks.npz`, `preview.png`, `camera.png` |
| `runs\` | recipe run folders from `recipe-run` (`result.json`, `run.log.jsonl`, `overlay.png`, `feedback_dmd.png`, `feedback_photo.png`); measure runs (`result.json`, `values.csv`, `overlay.png`, `scan.ply`, `spec.json`) and quick checks (`result.json`, `run.log.jsonl`, `quick_check.json`) |
| `events\` | **inspection event log**, for a spreadsheet (`engine/eventlog.py`): `inspections_v1_<YYYY-MM>.csv`, one row per inspection that wrote a `result.json` (recipe run, quick check, station job, station cycle, measurement) with its time, part id (the code read from the part), program, judgment (PASS / FAIL / PULL) and reason, values and file links; `inspection_values_v1_<YYYY-MM>.csv`, one row per value with its limits; `img\`, a copy of each run's picture; `pending\`, rows that wait while a month's file is open in another program; `last_v1.csv`, the newest row alone. One file pair per month, append-only; an inspection id can have two rows (ABORT after its verdict) and the last one counts. Page: `/static/events.html`. On for every run unless the environment says `SAS_EVENTLOG=off`; a failed append never changes a result. `python -m engine.eventlog --rebuild` writes the files again from the run folders (`result.json` stays the source of truth), `--merge` adds the waiting rows |
| `sessions\` | `log.jsonl` of every action |
| `trash\` | "deleted" entries (recoverable) |

Nothing is overwritten. "Delete" moves an entry to `trash\`; emptying it is your call.

## The rig (bench)

| Part | Detail |
|---|---|
| Projector | DLPC350 / DLP4500 912×1140 diamond DMD, driven as an HDMI extended display (video mode) |
| Camera | FLIR BFS-U3-16S2M, 1440×1080 mono, NewestOnly buffering |
| Geometry (current, `bench_9in_*`) | Projector 9 in (228.6 mm) to the right of the camera, standoff ≈ 762 mm, triangulation angle ≈ 16.7°, throw ratio 1.4 |
| Geometry (earlier, `cell_wall_2p7ft*`) | Projector 11 in (279.4 mm) to the right, standoff ≈ 823 mm, triangulation angle 18.8° |

**Presets:**
- `bench_9in_metric`: the current geometry, for the metric (stereo-calibrated) path, 16 667 µs.
- `bench_9in_metric_bright`: the same geometry and exposure with brighter projection settings.
- `bench_9in_metric_dark`: the same geometry at 66 667 µs, the long-exposure preset for shiny cylinders (dent sets).
- `cell_wall_2p7ft_vstripe`: correct orientation, but invert + duty 0.55 leaves a 45% light floor.
- `cell_wall_2p7ft_vstripe_fullrange`: the same geometry at duty 1.0. **Preferred for trials; run `tune` first.**
- `cell_wall_2p7ft`: the old rotate90 orientation, **reference only** (fringes along the baseline, so almost no depth).

**Measured on the real rig (earlier `cell_wall_2p7ft` geometry):**

| Measurement | Result |
|---|---|
| Signal | gc_conf ≈ 52, 32–41% valid, sharp focus, orientation check OK |
| Repeatability | relief σ = 0.20 px per pixel (unchanged scene, 3 ms, 8-frame average); nominal ≈ 0.38 mm per pixel before calibration |
| Projector map | 0.3 px round-trip error |
| Zone scans | work: "H8:I11" lit and cut exactly |
| Exposure | 8000 µs saturates 31% of the lit pixels |

**Known limits:**
- The QR codes in view were detected but too small to decode at 2.7 ft (they need ≥ 70 mm, or more zoom).
- The neutral-grey objects in view were spectrally indistinguishable; spectral ID needs coloured materials.

## Limits & sizing (measured on the bench setup)

**Setup.** FLIR BFS-U3-16S2M, 1440 × 1080 mono, 3.45 µm pixels; 2.8–12 mm varifocal lens set to ≈ 8 mm (consistent with the measured 0.30 mm/px at ≈ 640 mm: f ≈ 3.45 µm × 640 mm ÷ 0.30 mm ≈ 7.4 mm). Changing the zoom or focus changes every number below and invalidates the calibration: recalibrate after any lens adjustment, and lock the zoom/focus rings; DLP4500 projector at 228.6 mm baseline, 762 mm standoff. Field of view at ≈ 640 mm ≈ 430 × 320 mm.

**Codes (2D).**

| Distance | Resolution | Result |
|---|---|---|
| ≈ 330 mm | ≈ 0.16 mm/px | QR read OK |
| ≈ 640 mm | ≈ 0.30 mm/px | ≈ 5.4 mm Data Matrix (≈ 0.39 mm modules, ≈ 1.3 px/module): found, **not decodable** |

Rule: **mm/px ≈ pixel pitch × distance ÷ focal length**; keep **≥ 3 px per module** (so module ≥ 3 × mm/px).

Options for that 0.39 mm-module code at ≈ 640 mm. **Estimates**, from the rule above, not measured:

| Option | mm/px | px/module | Field of view |
|---|---|---|---|
| Today (1.6 MP, ≈ 8 mm) | ≈ 0.30 | ≈ 1.3 | ≈ 430 × 320 mm |
| 5 MP camera (≈ 2450 px wide), lens chosen for the same field | ≈ 0.18 | ≈ 2.2 (marginal) | same |
| 12 MP camera (≈ 4000 px wide), same field | ≈ 0.11 | ≈ 3.5 | same |
| Longer lens (≈ 25 mm) on today's camera | ≈ 0.09 | ≈ 4.4 | ≈ 127 × 95 mm |
| Close-up second camera (≈ 8 mm at ≈ 150 mm) | ≈ 0.065 | ≈ 6 | ≈ 95 × 70 mm |

**3D, measured on this setup.**
- End-face dents: detection floor ≈ 0.2 mm deep and ≥ 1.5 mm wide. The depth reported is the 3 × 3 grid-cell average at the dent's centre (0.75 mm square, ≈ 2 camera px): a 0.46 mm deep, 1.33 mm wide dent read 0.49 mm (its deepest single point 0.57–0.65 mm), its width at half depth 1.5–1.64 mm (≈ one camera px wide).
- End-face size and rim (a 21 mm class cylinder, 19.5 mm flat face): the flood outline's diameter is held to the nominal diameter within 4 % (a lens zoomed after the calibration read 0.84×: a PULL, not a small part); its roundness on good parts is 0.06–0.09 mm rms, 0.20–0.28 mm max, so edge dents are found from 0.4 mm over 1.5 mm of rim, up to about 40 % of the rim (wider: the size check only); rim that cannot be judged is a PULL. The inspected flat ends at 0.926 × the face radius (9.03 mm on the 19.5 mm flat): what lies beyond is judged by the outline only.
- End-face exposure: 16.7–100 ms in whole 60 Hz video frames, chosen per seating (the same face needed 33–83 ms); a seating that is too dark at 100 ms or clips > 0.5 % of the face before it is bright enough is not scanned.
- Flatness of a good face: 0.03–0.05 mm rms.
- Thermal drift: heights *within* one scan are stable, but absolute heights drift ≈ 0.006 mm/min until the rig is warm; warm up ≈ 20 min before golden or absolute work.
- Fine debris (e.g. a ≈ 40 µm hair) is detectable below one pixel from the before/after flood ratio against a clean reference; its width is only roughly estimable (experimental).
- Visible light cannot see under opaque coatings: that needs UV fluorescence, SWIR or a laser method.

## Decode, in one paragraph

Gray code (coarse) plus 4-step phase shift (fine) gives the projector column seen by each camera pixel. Relief is object minus base. Along the way:
- **Polarity is measured, never assumed.** On this bench the HDMI path lands patterns inverted, and inverted fringes are phase-corrected by π. That fix removed dense ±24 px unwrap errors present since v10.
- Clipped fringe samples are dropped.
- `coarse_bits` and `fix_period_jumps` are optional preset settings.
- The camera uses NewestOnly buffering, with a pumped HDMI hold and a warm-up, which fixed the stale frames.

## Safety rules (`engine/rules.py`, enforced, not advisory)

1. **HDMI-only scan path.** No process that scans loads the DLPC350 USB driver or `hid`. USB display commands raise `SafetyViolation`. `engine/capture/dmd_control.py` is the separate, human-run USB diagnostics tool: every command takes the rig lock, a write (`--relock`, `--source`, `--bitdepth`) needs `--yes`, and it is never imported by the engine (a test enforces this). No scan, recipe, station or command-line tool switches the projector display (video / pattern sequence), and the seed driver refuses that write on a real device, always.
2. **Pumped holds.** The projector window is kept painted with `cv2.waitKey`; there are no bare sleeps (test-enforced).
3. **LED thermal limits.** Burst R180 G255 B230, average R120 G180 B160, plus duration limits (`check_led_duration`). Record the GUI currents in `projection.led` so this rule can be checked.
4. **Mono camera ⇒ bw scheme.** Colour schemes are flagged.
5. **Base and scan share a geometry.** That means the same `geometry_id` **and** the same geometry fingerprint: rotate90, size, coded axis, rig side/baseline/standoff.
6. **mm only with a calibration at that geometry and fingerprint**, read from the library. A trashed calibration stops giving mm immediately. Mock data uses `mock-<geometry_id>`. A calibration is also bound to the **projector display** it was made in (video, or pattern sequence): a capture in the other display gives no mm, and a pattern-display calibration is never used for a normal scan. The display comes from the board poses: `calib-pose` captures in video display, and a pose whose capture does not say its display is not stored. A pattern-display calibration counts only while no video calibration of the geometry is newer (the rig check and the re-home renew the video one only): after a re-home or a new calibration it is made again. So the two scans that capture in pattern-sequence display (`scan-trigger`, `scan-flash`) are refused on a real rig until a calibration made in that display exists; their stored scans give no mm either. On a mobile rig they are refused for now: the rig is verified for its video calibration only. The simulated bench has no offset between the two displays, so mock runs are not affected.
7. **Append-only library.** Writes use exclusive-create. `delete` is one atomic move to `trash/` (bases, scans, calibrations, calibration poses, rig references, ...; the reason is recorded with the entry); an entry with an open file raises `EntryInUse` and is left untouched.
8. **One driver.** A machine-wide rig lock at `%PROGRAMDATA%\swiss_army_scanner\rig.lock` is released by the OS if the holder dies. The web UI serializes its own jobs on one worker.
9. **Physical steps are human-only.** Agents ask and wait. Part-present sensors are taught only with `empty_confirmed` after the human confirms. The command line itself never waits for input: the answer is an argument (`teach-presence --empty-confirmed`, `delete --yes`), and without it the command is refused at once (exit 2). Only `--interactive` lets a command ask (`scan wizard --interactive`), and a question gives up after `--prompt-timeout` seconds with nothing done.

## Tests

```bash
python -m pytest -q                                   # full suite, mock only
python -m pytest -q tests/test_rules.py tests/test_cli.py tests/test_run_job_mock.py
python -m pytest -q tests_sensor                       # the seed package, in its own process
```

Optional bench-data tests skip unless you point them at real files: `SCANNER_TEST_ATLAS` (a held-part atlas `.json`) and `SCANNER_TEST_REAL_CAL` (a stereo calibration folder).

The full suite takes about 35 min. It is mock only, apart from tests that assert refusals before any hardware is touched. The simulated bench has real parallax, shadows, colour and QR labels, and draws fresh camera noise for every capture.

## Status

| Area | Status |
|---|---|
| Engine: capture, decode (x/y, complementary, coarse bits), relief, library, calibration, measure, inspect, compare | ✅ built, mock-tested; core paths real-rig tested |
| Web UI (11 tabs + Sensors), CLI, output bin, trash, rig lock | ✅ built, mock-tested; real-rig smoke tested through the engine |
| Grid zones, projector map, repeatability, tune, profiles | ✅ real-rig tested |
| Spectral, light separation, code reading | ✅ real-rig run; limited by the scene (see above) |
| Anti-glare code reading (`codes --anti-glare`) | 🧪 experimental: synthetic mock test; one bench trial |
| Virtual sensors (spot, part present, alignment), good-memory bank | ✅ mock-tested; real-rig trials next |
| Flow-ready job (`run`: station config → `result.json` + JSON-lines log, exit code = verdict) | ✅ built, mock-tested (`tests/test_run_job_mock.py`); not yet run on the rig |
| Recipes (the checks that verify a work state: zones, checks, verdict painted on the part; `recipe-*`, `run` job.type recipe, result-code register) | ✅ built, mock-tested (`tests/test_recipes_mock.py`, `tests/test_recipe_run_mock.py`: synthetic scans of the mock scenes + the full mock capture path); **not yet run on the rig**: tolerances in the examples are illustrative, prove them on site (`docs/PORTING.md`) |
| Station configuration → profiles → work states (step flow: sensor claims confirmed or corrected by the DLP, permissions `PERMIT_*`, prompt / ACK / hold / retry / kickout; `run` on a configuration, `flow-show`, `/quick`, `POST /api/io/inputs`, cycle folder `result.json` schema 0.4) | ✅ built, mock-tested (`tests/test_flow_mock.py`, `tests/test_flow_ui_mock.py`: the schema, the cycle state machine, the bus, synthetic scans of the mock tube scenes, the operator page, cycles through the full mock capture path); **not yet run on the rig**; sensor inputs arrive over HTTP only (hardware backends are placeholders) |
| Operating modes: Measure (`measure-run`: values + uncertainty, NOT_MEASURABLE), Quick check (`quick-check`), Station; preset mount fixed / mobile with the rig-state gate; the mode switch on `/quick` and the bus (`SELECT_MODE`) | ✅ built, mock-tested (`tests/test_modes_measure_mock.py`, `test_modes_quick_mock.py`, `test_modes_rig_mock.py`, `test_modes_ui.py`: synthetic scans with known truth + the full mock capture path); **not yet run on the rig**: the uncertainty is noise only (prove the mm with `accuracy-block`), a mobile rig's rig check needs the reference scene it was stored on (else: calibrate with the fixture), and what a carried rig does to the calibration between two rig checks is only as visible as those checks |
| Painted zones (`zone-paint`: draw zones on the part with the projected mouse cursor; `zones`, `zone-show`, `scan-fast --zones`, `measure-run --zones`, recipe zone `ref`, `/quick` Lights) | 🧪 built, mock-tested only (`tests/test_zones_*.py`: the painter's state machine, scripted sessions, scans and recipes on the simulated bench); **not yet run on a real rig** |
| Industrial I/O (Modbus TCP, OPC UA, EtherNet/IP), camera GPIO, serial, digital I/O | placeholders: clean interfaces + intended mapping, not implemented |
| Hardware-trigger pattern mode (USB, guarded) | future (architecture step 10) |
| Hardware trigger (full speed): TRIG_OUT_1 → camera, one frame per pattern (`trigger-test`, `scan-trigger`, packs with `--trigger-mode trigout --period-us 10000/5000/4500`) | 🧪 experimental: mock-tested (`tests/test_trigger_fullspeed_mock.py`); **not yet run on the rig**. `scan-trigger` captures in pattern-sequence display: on a real rig it is refused (no mm, rule 6) until a calibration made in that display exists; `trigger-test` needs none. Bench checklist with wiring: [`docs/TRIGGER_FULLSPEED_TEST.md`](docs/TRIGGER_FULLSPEED_TEST.md) |
| Flash soft-sync fast scan (1-bit pack, free-running camera, software sort) | ✅ built, mock-tested (simulated DLP loop + free-running camera, GUI mistakes, dropped frames; pack → loop → camera → sort → decode → metric scan end to end vs an ideal synchronous capture and the HDMI fringes, `tests/test_flash_e2e_mock.py`); PySpin free-run code and the real loop not yet run on the rig. It captures in pattern-sequence display: on a real rig it is refused (no mm, rule 6) until a calibration made in that display exists |
| Stereo XYZ calibration, metric scan, cylinder deviation, accuracy check, metric mesh | ✅ built, mock-tested (pinhole stereo mock bench with known K/R/t); real-rig trials next |
| Calibration fixture (3 placements, rigid-fixture refinement, face-angle check) + check block (`tools/make_calib_fixture.py`, `calib-pose --fixture`, `accuracy-block`) | 🧪 mock-tested only (`tests/test_calib_fixture_mock.py`); **not yet printed or used on a real rig**: [`docs/CALIBRATION_FIXTURE.md`](docs/CALIBRATION_FIXTURE.md) |
| Dent detection on shiny cylinders (dent sets, null calibration, fused finders) | ✅ built; synthetic + mock tests; reproduced offline on the saved bench dent sets (see `engine/measure/dents/`); production capture not yet run on the rig |
| End faces (locate, aim / check ring, auto intensity / auto exposure, targeted scan, profile, golden, check; radii from the face diameter, known-size check, rim outline, centre depth) | ✅ built; synthetic + mock tests; reproduced offline on the saved bench end-face scans (`engine/measure/endface/`); the productised capture not yet run on the rig |
| HDMI fast zone scan, station measurement (cylinders + ribbed tube: heights, spacing, zones T1..Tn, golden, check) | ✅ built; synthetic + mock tests; reproduced offline on the saved bench station scans (`engine/measure/station/`); `scan-fast` and `station` run on the rig through the engine |

The engine verbs (`api.*`, the `scan` CLI, the web UI) are the only supported path to the hardware.

## Acknowledgements

Developed by Joey Messina with AI collaborators Claude (Anthropic) and Grok (xAI). Grok contributed pattern/grid methods (including the tile+phase fast scan), bug reports and independent testing.

## License

Apache License 2.0: see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). The rig safety rules above are guidance; operating the hardware safely is your responsibility.

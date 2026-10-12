# Porting to your line

The scanner is a DMD projector + a camera + this engine. Nothing in the engine is tied to one
part, one station or one vendor's hardware: the part knowledge lives in **recipes** (zones +
checks + tolerances, `recipes/README.md`), the rig knowledge in **presets** (geometry +
capture settings) and their **calibrations**, the line integration in the **command bus** and
its I/O backends. Porting = five adapters and one validation procedure:

| layer | what you provide | where |
|---|---|---|
| projector | a display output the engine can paint full screen | `engine/capture/projector.py`, interface `engine/capture/interfaces.py` (DisplayWindow) |
| camera | four methods: exposure, flush, grab, close | `engine/capture/camera.py`, interface CameraDriver |
| geometry + calibration | a preset per rig geometry, its stereo calibration | `presets/`, `scan preset-add`, `scan calibrate-stereo` |
| recipe | the part's zones, checks, tolerances, feedback | `recipes/`, `scan recipe-add` |
| I/O | start / abort / home / trigger / work state in; ready / busy / pass / fail / alarm / pull + result code out | `engine/control/commands.py` |

Then prove the numbers on site (the last section) before anybody trusts a PASS.

## 1. Projector

The engine drives the projector as an **extended display** (HDMI / DisplayPort): a full-screen
window at the display's desktop offset, patterns pushed as images, a **pumped hold** that keeps
the window painted (never a bare sleep). Any DLP / LCoS / LCD projector or DMD board that
shows a video input works the same way.

- Size and offset: `projection.size` (native DMD columns x rows, e.g. 912 x 1140) and
  `projection.monitor_offset` in the preset; the window's rect is checked against them and a
  mismatch is reported (wrong monitor, display scaling).
- Polarity is **measured**, never assumed (raw black vs raw white): an inverting video path
  is fine. Gamma (`projection.gamma`, `scan gamma`) and duty are preset settings.
- A diamond-pixel DMD (rows half a column apart) needs the row parity once per rig
  (`scan parity`); a square-pixel device does not.
- Optional faster modes (an on-board pattern sequence fired by a trigger line, flash
  packs) are separate, human-entered modes (`docs/FLASH_PATTERN_MODE.md`,
  `docs/TRIGGER_FULLSPEED_TEST.md`); the HDMI path stays the default.
- The engine never sends USB / vendor control commands to the projector during a scan
  (safety rule 1). LED currents are set by a human in the vendor's tool and recorded in
  `projection.led` so the thermal limits (rule 3) can be checked.

A new display driver implements `show(image)`, `hold(seconds)` and `close()`
(`DisplayWindow`), and `Projector` uses it instead of `HDMIWindow`.

## 2. Camera

The engine needs a mono camera in free run with fixed exposure: `set_exposure_us`, `flush`
(drop buffered frames), `grab` (one (H, W) frame) and `close` (`CameraDriver`). The bench
uses the FLIR Spinnaker SDK (`RealCamera`); any **GenICam** camera (**GigE Vision**,
**USB3 Vision**) fits the same four methods: `ExposureTime`, `GainAuto` off,
`AcquisitionMode` Continuous, newest-only buffering. A hardware trigger adds
`TriggerMode` / `TriggerSource` / `LineSelector` (`engine/reconstruct/trigger_scan.py`).

- Exposure: whole video frames over HDMI (16 667 us at 60 Hz) avoid the projector's
  colour / PWM sequence beating with the exposure (`scan tune` recommends one).
- Resolution sets the floor: keep >= 3 camera px per feature you must see (a code module, a
  dent's width); `README.md` "Limits & sizing" has the arithmetic.
- Lock zoom and focus: any change invalidates the calibration.

## 3. Geometry and calibration

1. `scan preset-add my_line --from <closest preset> --geometry my_line_v1 --set ...`: the new
   geometry starts **uncalibrated** (nothing is copied).
2. `scan signal`, `scan tune`: contrast, focus, exposure; fringes must vary across the
   projector-camera baseline (the orientation check says so).
3. Stereo calibration, with the new preset selected (`scan preset-select my_line`, or
   `--preset my_line` on every step: the calibration is bound to its geometry):
   `scan charuco` (print at 100 %, caliper a square) -> `scan parity` ->
   `scan calib-pose` x 12-20 (tilted up to +-30 deg, filling the measuring volume) ->
   `scan calibrate-stereo --square <measured mm>`: camera RMS < 0.3 px, projector < 0.5 px.
   Watch the **projector corners** each pose prints (corners with a projector position /
   corners found): under 60 % the fringes are not resolved on the board (a printed board,
   an opened lens, a longer distance) and the pose is captured again with coarser fringes
   (16,160,1280, then 24,192,1280 DMD px; `--periods` sets them yourself). A pose set keeps
   one period set. `calibrate-stereo` lists the poses that gave projector points and is
   rejected with nothing saved when the fit is not a calibration (degenerate projector
   intrinsics, stereo RMS > 2 px, too few poses with projector points).
4. `scan rig-check --set-reference` with the station empty: HOME then detects a nudged
   camera or projector (shift, roll and zoom; OK only when the projected pattern is matched;
   a pattern that is not seen, or roll / zoom that cannot be determined, is UNKNOWN, never
   OK; roll / zoom detection is tested on the mock bench only so far). After a small camera
   nudge `scan rig-check --rehome` writes a re-homed copy of the calibration (the camera's
   rotation taken out, the original kept; rotation only, so recalibrate when you can).

Every mm the engine reports comes from the stereo calibration of the SAME geometry and
fingerprint (rule 6); without one a recipe run is INVALID and says how to calibrate.

## 4. Recipe

Start from the closest built-in (`scan recipe-list`) and copy it: `scan recipe-add my_part
--from box_lid --set preset=my_line ...`, or write one (`recipes/README.md`). Per zone:

- **Units**: `camera_px` (fast to draw on a photo), `projector_px` / grid cells (what the
  projector lights), or `part_mm` (mm in the part frame: robust to the part standing proud of
  the fixture). Prefer `part_mm` for checks, with a datum zone on the fixture when the part is
  bigger than the background (`part.frame.reference`).
- **Checks**: presence, position, height / step, flatness, defect, edge, code - or your own
  (`docs/CHECKS.md`). Tolerances are in mm / deg in the parameter names.
- **Golden**: position and edge-position checks compare with a good part:
  `scan recipe-golden my_part` on a known-good placement (recorded as the recipe's golden;
  only recorded goldens or an exact scan id are used, never a run's own test scan).
- **Feedback**: the disposition painted on the part (green outline PASS, red fill + a cross
  at the fault REJECT, an amber frame round the part + "PULL" for quality inspection): set
  the hold to what the operator needs to act.
- **Pull** (`pull.*`): `guard_band_pct` (a PASS value this close to a limit, in % of the
  tolerance, is PULLed: start with 10 and tighten once Gauge R&R says the measurement is
  better than that) and `invalid_reasons_to_reject` (doubts your line rejects instead of
  inspecting, e.g. `code_unreadable`; default none).

`scan recipe-validate my_part` lists every problem before anything runs; `scan recipe-run
my_part --scan <stored scan> --no-feedback` re-checks a stored scan offline while you tune
tolerances: nothing is captured or projected, so the real rig needs no operator confirmation
for it. Without `--no-feedback` the verdict would be painted on whatever is in front of the
projector now, so that form needs `--operator-confirmed` like a live run.

## 5. I/O

One state machine (`engine/control/commands.py`) takes START / ABORT / HOME / TRIGGER / ARM /
DISARM / RESET / work-state commands from every source (the operator page, keys, HTTP
`POST /api/cmd/*`, hardware inputs) and publishes READY / BUSY / PASS / FAIL / ALARM / PULL
plus three registers: `result_code` (the recipe's `outputs.result_codes`: PASS 1, FAIL 2,
INVALID 3, ABORTED 4, PULL 5 by default; 0 = none yet), `state_code` (index of READY, ARMED,
RUNNING, ABORTING, HOMING, FAULT) and `workstate` (1-based): the position among ALL station
configs in file-name order, invalid ones included (marked unusable, START / TRIGGER on them
refused with result INVALID). The number never shifts because a config broke, and the list is frozen while
ARMED or busy; adding or removing config files while idle does renumber, so keep the station
config file names stable on a line driven by a PLC.

Backends are config-driven (`stations/io_example.json`); the hardware ones are
**placeholders** with the intended mapping in their docstrings and errors:

| kind | in (PLC -> scanner) | out (scanner -> PLC) |
|---|---|---|
| `camera_gpio` | Line0 trigger, Line3 abort (camera inputs only: never Line2, the 3.3 V output) | Line1 busy |
| `serial_io` | text tokens START / TRIG / STOP / HOME | - |
| `digital_io` | DI0..DI15 rising edges | DO0..DO15 |
| `modbus_tcp` | coils / discrete inputs -> commands, a holding register -> work state | coils <- signals, holding registers <- result / state / work state |
| `opc_ua` | Boolean nodes (subscription) -> commands, an Int node -> work state | Boolean nodes <- signals, Int nodes <- registers |
| `ethernet_ip` | BOOL tags / input-assembly bits -> commands, a DINT tag -> work state | BOOL tags / output bits <- signals, DINT tags <- registers |

Implement `start(bus)` (deliver `bus.command(name, source=...)`) for an input and
`publish(signals)` / `publish_registers(registers)` for an output with your site's stack,
register the class under its kind, enable it in the I/O config. A typical handshake:
PLC sets the work state -> ARM -> part in place -> TRIGGER -> BUSY -> PASS / FAIL / PULL +
result code -> the PLC acts (and the operator sees red on the part where it failed, or an
amber frame when it is pulled). START / TRIGGER / ARM may carry the work state the sender
expects (`{"workstate": ...}` over HTTP): refused when another one is selected. Changing the work state while ARMED disarms (ARM
again). Over HTTP, `operator_confirmed` must be a JSON boolean (only `true` confirms; any
other type is a 400 on START / TRIGGER / ARM); ABORT, DISARM, HOME and RESET ignore it.

**Routing PULL to a quality-inspection lane.** Every cycle ends in exactly one of three
dispositions (`README.md`, *Dispositions*): CONTINUE (PASS output), REJECT (FAIL output) or
PULL (PULL output: any doubt - INVALID, ABORTED, the rig check moved, a crashed check, an
unreadable code, a PASS inside the guard band). Wire the PLC so that:

- **PASS** -> the part continues down the line; nothing else may release it. PULL is never
  set together with PASS, and a pulled cycle never writes the PASS result code (also when
  the run crashes or is aborted after it judged the part). A HOME (rig check) ends a PASS:
  the PASS bit and code never come back for that cycle, after HOME OK or a FAULT and
  RESET. An ABORT while the station is BUSY makes the cycle ABORTED (PULL), even when the
  run had already judged the part.
- **FAIL** -> the reject lane / bin.
- **PULL** -> a **quality-inspection lane** (a hold station, a divert gate to a QI bench, a
  red-tag shelf), never back to the line and never to the reject bin by default: a person
  inspects the part and decides. Map the PULL bit (`stations/io_example.json`: `DO5`,
  `coil:21`, `Scanner.Pull`, `Scan_Pull`) to that gate. With registers only, route the
  INVALID (3), ABORTED (4) and PULL (5) codes there - anything that is not exactly the PASS
  or the FAIL code.
- **No answer** (BUSY never clears, the scanner is unreachable, READY dropped, ALARM / FAULT)
  -> treat the part in the station as PULL too: fail closed.

`result.json` (`disposition`, `pull_reasons`) records why each part was pulled, so the QI
bench sees what to look at (zone, check, reason). A line that would rather reject a given
doubt (e.g. a missing label = `code_unreadable`) lists it in the recipe's
`pull.invalid_reasons_to_reject`; that is a line decision to make with quality, not a
default.

## 6. Validation: prove the numbers on your site

Do this per rig geometry, after every calibration, and keep the records with the line's
quality documents. Recipe runs write everything needed (`result.json` per run; the values
carry their units).

1. **Calibration check against the board.** After `calibrate-stereo`, check the RMS numbers
   (camera < 0.3 px, projector < 0.5 px per axis), then measure a calibration-board pose the
   fit did not use: square size and planarity in mm vs the printed (calipered) size.
2. **Known-object check.** Scan objects of known size spanning the measuring volume - a gauge
   block step, a precision flat, a cylinder or sphere of known diameter - with
   `scan accuracy plane|step|cylinder|sphere --repeats 5`: mean and max error, repeatability,
   GO / NO-GO against your tolerance. Use objects with a finish like the part's (shiny parts
   measure differently from matte gauges).
3. **Repeatability / Gauge R&R.** Take n parts (10 is the usual minimum, spanning the
   tolerance range: good, near-limit and bad) x m repeats (2-3) x k operators (2-3), every
   placement a real re-load of the part. Run the recipe each time and collect each check's
   headline value from `result.json` (`zones[].checks[].values`). Analyse with the ANOVA
   method: repeatability (equipment variation), reproducibility (appraiser variation), part
   variation, %GRR = 100 x GRR / total variation (or / tolerance width). Common acceptance:
   < 10 % good, 10-30 % conditionally acceptable, > 30 % not acceptable; number of distinct
   categories >= 5.
4. **Attribute agreement (the verdict).** For PASS / FAIL checks (presence, defect), run a
   known set of good and seeded-bad parts several times: false-reject and false-accept rates
   and the agreement with the reference decision. Seed defects at, above and below the
   tolerance to see where detection starts.
5. **Drift.** Repeat a known-object check over a shift (warm-up: the bench drifts until the
   rig is warm, `README.md`), and run HOME (`scan rig-check`) at every start of shift.

A number that has not been through steps 2-3 on the site's rig is an estimate, not a
measurement. The bench numbers in `README.md` are the bench's, not yours.

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
| I/O | start / abort / home / trigger / work state in; ready / busy / pass / fail / alarm + result code out | `engine/control/commands.py` |

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
3. Stereo calibration: `scan charuco` (print at 100 %, caliper a square) -> `scan parity` ->
   `scan calib-pose` x 12-20 (tilted up to +-30 deg, filling the measuring volume) ->
   `scan calibrate-stereo --square <measured mm>`: camera RMS < 0.3 px, projector < 0.5 px.
4. `scan rig-check --set-reference` with the station empty: HOME then detects a nudged
   camera or projector.

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
  `scan recipe-golden my_part` on a known-good placement.
- **Feedback**: the verdict painted on the part (green outline PASS, red fill + a cross at
  the fault FAIL): set the hold to what the operator needs to act.

`scan recipe-validate my_part` lists every problem before anything runs; `scan recipe-run
my_part --scan <stored scan>` re-checks a stored scan offline while you tune tolerances.

## 5. I/O

One state machine (`engine/control/commands.py`) takes START / ABORT / HOME / TRIGGER / ARM /
DISARM / RESET / work-state commands from every source (the operator page, keys, HTTP
`POST /api/cmd/*`, hardware inputs) and publishes READY / BUSY / PASS / FAIL / ALARM plus three
registers: `result_code` (the recipe's `outputs.result_codes`: PASS 1, FAIL 2, INVALID 3,
ABORTED 4 by default; 0 = none yet), `state_code` (index of READY, ARMED, RUNNING, ABORTING,
HOMING, FAULT) and `workstate` (1-based).

Backends are config-driven (`stations/io_example.json`); the hardware ones are
**placeholders** with the intended mapping in their docstrings and errors:

| kind | in (PLC -> scanner) | out (scanner -> PLC) |
|---|---|---|
| `camera_gpio` | Line0 trigger, Line2 abort | Line1 busy |
| `serial_io` | text tokens START / TRIG / STOP / HOME | - |
| `digital_io` | DI0..DI15 rising edges | DO0..DO15 |
| `modbus_tcp` | coils / discrete inputs -> commands, a holding register -> work state | coils <- signals, holding registers <- result / state / work state |
| `opc_ua` | Boolean nodes (subscription) -> commands, an Int node -> work state | Boolean nodes <- signals, Int nodes <- registers |
| `ethernet_ip` | BOOL tags / input-assembly bits -> commands, a DINT tag -> work state | BOOL tags / output bits <- signals, DINT tags <- registers |

Implement `start(bus)` (deliver `bus.command(name, source=...)`) for an input and
`publish(signals)` / `publish_registers(registers)` for an output with your site's stack,
register the class under its kind, enable it in the I/O config. A typical handshake:
PLC sets the work state -> ARM -> part in place -> TRIGGER -> BUSY -> PASS / FAIL + result
code -> the PLC acts (and the operator sees red on the part where it failed).

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

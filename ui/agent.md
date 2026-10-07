# Driving the scanner as an AI agent

Your whole world is `engine.api`: the verbs listed in the README plus the dataclasses they return. The same verbs work in mock mode and on the real rig.

## Contract

- **Use the API only.** Never import `engine.capture.*`, `swiss_army_sensor`, `hid` or `cv2`, and never open or edit library files directly. The engine enforces the §9 safety rules, and bypassing it bypasses them.
- **Physical steps are human-only.** Placing or removing an object, focus, iris, gauge blocks and entering true dimensions are done by a person. Prompt the human, wait for them to confirm, then call `check_signal()` to verify the setup. Never assume the physical state. (In mock mode, `api.mock_bench().place(...)` / `.clear()` stand in for the human.)
- **Read numbers, not pictures.** Use `SignalReport.gc_conf` / `valid_pct` / `focus_hint`, `Dimensions.H/W/L` + `units` + `error`, and `Verdict.result`. Always check `units` and `warnings`: `px` means the result is uncalibrated (rule 6).
- **Never move the rig between base and object.** If the rig moved, capture a new base and recalibrate. The engine refuses to mix geometries.

## Typical flow

1. `load_preset(name)`. Read `preset.warnings` and `preset.calibration`.
2. Ask the human to clear the scene. Then `check_signal()` and repeat until `ok`. Relay `focus_hint`.
3. `capture_base(preset, "empty")`.
4. Ask the human to place the part, without touching the camera or projector. Then `capture_object(preset, base, label, known_dims=...)`, with known dims only if the human supplies them.
5. `measure(scan)`, or `calibrate(preset, [scans with known dims])` and then `measure`.
6. `compare(scan, golden)` for PASS/FAIL.

gc_conf bands: below 8 is unusable, about 18 marginal, about 30 usable, and 120+ excellent.

## From the command line

`scan.bat [--mock] [--json] <command>` (= `python -m ui.cli`). Add `--json` and read one JSON document: `ok`, `exit_code`, the result with its `units` and `warnings`, and `outputs_written`. Branch on the exit code:

- `0`: ok (PASS, signal ok, spot found, PRESENT, ALIGNED)
- `1`: not ok (FAIL / FLAG / INDETERMINATE, ABSENT, NOT_ALIGNED, base auto-rejected, cancelled)
- `2`: refused (rule, geometry, config or bad input; `RigBusy` means another program holds the rig, so wait or queue the job in the web UI)
- `3`: mode not built yet

Zone flow: `map` (once per rig position) → `grid` → read the grid photo named in `camera_png` and find the cells the part covers → `scan part --zone H8:I11@96`. Keep the `@96`: it pins the cell names to the grid in that photo.

Virtual sensors: `spot-size 6.5 [--measured K]`, `spot H8 --diameter 6.5`, `teach-presence pe H8 --diameter 6.5 --empty-confirmed` (pass the flag only after the human confirms that the sensor's field is empty), `check-presence pe` (exit 0 PRESENT), `teach-align fixture <scan>`, `check-align fixture <scan>` (exit 0 ALIGNED), `sensors`. Positions: a cell `H8`, `dmd:COL,ROW` or `cam:X,Y`.

Cylinders and 3D (on a stored scan): `mesh <scan> --kind relief|base|cylinder --fmt stl|obj`, `unwrap <scan> --radius 24.05` (seen arc, end-face height, dent heatmap), `dents <scan>` (exit 0 PASS; 1 FAIL or INDETERMINATE: uncalibrated px never passes, and only the seen open arc is inspected), `maps <scan> [--spectral ID --light ID --golden ID]`, `checker`. Read `units` first: `px` means no mm (rule 6).

Dents on shiny cylinders (needs a stereo calibration at the geometry): the human places a KNOWN-GOOD part, then `--preset DARK dent-scan good1 --zone H9:I12 --short-preset SHORT` and `dent-null good1 --diameter 21.05`; per cell `dent-scan cellN ...` and `dent-detect cellN` (exit 0 CLEAR, 1 FLAG, 4 the detector itself failed: never read 4 as a verdict; the diameter comes from the matching null). Read each candidate's `level` (detection above 1.25 x the good part's maximum / watch / edge), `kind` (dent / scratch / speck, advisory), `depth_mm` with `depth_reliable` and `depth_significant`, and its place on the cylinder (`theta_deg`, `s_from_top_mm`; `s_mm` is from the axis point level with the camera, not a cylinder end). The null calibration is per cylinder diameter, fixture, pattern set and stereo calibration: recalibrate on a good part when any of them changes (a null made with `--fixture NAME` needs `dent-detect --fixture NAME`). Read the report's `warnings` (e.g. a finder excluded by the null, no mirrored room line on this cell).

End faces (a cylinder lying with its circular END FACE toward the camera; needs a stereo calibration): the human places the part, then `face-locate --zone H10:I12` (a ray-plane locate scan of the zone, then the face: plane, rim circle of the known `--diameter`, camera -> DMD affine; read `affine.rms_col_px` / `rms_row_px`, a warning above 0.5 DMD px) (`face-locate <metric scan>` needs x AND y decoded: `scan-fast --axis xy` or `scan-metric`; an x-only scan is refused; `--auto-intensity` needs a preset with projection.invert false, e.g. a `*_bright` preset), `face-aim --check` (exit 0 ON_FACE; 1 OFF means the part moved: re-locate), `face-scan cN --auto-intensity` (only the face lit, 3 repeats + polar set; the duty is for this scan only, the LED currents are never touched), `face-profile cN` (flatness, repeatability, `radial`, PNG / CSV / STL). A golden needs several GOOD placements: `face-golden good1 good2 good3 good4` (default detector `mf_fir`: MF-CFAR + the reflectance FIR from the flood pair; read `leave_one_out` - any trigger on a good placement is a false trigger - and `sensitivity.by_size`, the synthetic-dent detection per depth x width). Per cell `face-check cN` (exit 0 PASS, 1 FAIL or INCOMPLETE - less than 90 % of the face measured is never passed): each trigger has `zone` face | rim, `r_mm`, `theta_deg` (0 = camera right, 90 = camera down), `depth_mm` (matched-filter dent depth, positive = a dent), `width_mm`, `z` (significance; trigger at 5.5), `dev_mm` (negative = below the golden) and `area_mm2`; `stats.noise_floor_mm` is the 1-sigma dent depth per width. Warnings about a saturated flood mean: re-scan with `--auto-intensity`. `--detector band` uses the prototype band instead. `face-aim --hold 30` keeps the ring up as the human's placement guide. `faces` lists targets, scans and goldens.

Surfer atlas: `atlas-import <path.json> <name>` (the source is only read), `atlases`, then use a region anywhere a zone goes: `scan part --zone atlas:part/part`, `flood atlas:part/part`, `antiglare atlas:part/part` (a pattern choice, not wavefront shaping), `multiorient atlas:part/part`. `flash-pack --axes x` only writes files. A human switches the DLPC350 into pattern mode (docs/FLASH_PATTERN_MODE.md). Fast scan without a trigger cable: `fastscan-pack` (files only), a human loads it with the 4500 MKII GUI's firmware link and plays it, then `scan-flash <label> --pack DIR [--roi X0,Y0,X1,Y1]` records with the camera only and sorts the frames in software (exit 2 with the reason when the loop is not playing or not in order; `--dry-run` = the timing plan; `sort-flash <recording.npz>` re-sorts offline; docs/FLASH_SOFTSYNC.md). Never ask the engine to start or stop the sequence: it cannot.

Library: `bases`, `scans`, `calibrations`, `readings`, `materials`, `banks`, `maps`, `sensors`, `faces`, `outputs`. `delete <id> --yes` moves an entry to trash/; nothing is destroyed. `scan.bat --help` lists everything.

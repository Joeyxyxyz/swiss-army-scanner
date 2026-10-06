# Swiss Army Scanner

One TI DLP LightCrafter 4500 (DMD) and one FLIR Blackfly S mono camera become many virtual sensors: 3D scanning, measurement, defect detection, a good-part memory bank, 2D profiles, barcode/QR reading, spectral material ID, part-present and alignment sensors, and projected grids you can point at ("scan zone H8:I11").

It is one engine with one small verb API (`engine/api.py`). A web UI, a CLI and AI agents (Claude, Grok) all drive the same verbs. The engine owns the hardware, patterns, decoding, calibration, the library and the safety rules; callers never touch the rig directly.

- Spec: [`docs/ENGINE_ARCHITECTURE.md`](docs/ENGINE_ARCHITECTURE.md)
- Agent guide: [`ui/agent.md`](ui/agent.md)
- Flow-ready jobs: [`stations/README.md`](stations/README.md)
- Seed package: [`swiss_army_sensor/README.md`](swiss_army_sensor/README.md)
- Blue-lines fix: [`docs/HDMI_TIMING_FIX.md`](docs/HDMI_TIMING_FIX.md)

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

**Flow-ready job.** One command runs one job from a station config and writes a new run folder with `result.json` (the same keys every run: PASS / FAIL / INVALID, deviation in mm, calibration id, outputs) and `run.log.jsonl` (one JSON event per line). The exit code is the verdict: 0 PASS, 1 FAIL, 2 refused, 4 crashed, 5 INVALID. See [`stations/README.md`](stations/README.md).

```bat
scan.bat --mock run stations\example_cell_station.json
scan.bat run stations\my_station.json --operator-confirmed
```

## Modes (UI tabs ↔ verbs)

| Mode | Verbs | Notes |
|---|---|---|
| Setup | `load_preset`, presets | Toggles for camera (exposure, gain, averaging), projection (duty, invert, rotate90, gamma, offset) and **rig geometry** (projector left/right/above/below, baseline, projection distance, throw ratio). Shows the live nominal mm/px and triangulation angle. |
| Focus & signal | `check_signal`, `tune`, `measure_gamma`, live view | Live focus score. The **orientation check** flags fringes parallel to the baseline. The scan assistant recommends exposure; prefer whole 60 Hz frames (16 667 µs). |
| Base library | `capture_base`, `reject_base`, `delete` | An empty-scene datum per geometry. `zone=` makes a zone-lit base. Bases under 5% valid are auto-rejected. |
| 3D scan | `capture_object`, `reconstruct`, `profile` | Relief vs a base, `.ply` point cloud with a 3D viewer, and **2D profiles** with `step_height` (for cell height). |
| Measure | `calibrate`, `measure` | px → mm from known heights / widths (gauge blocks flat on the base). |
| Defects | `inspect`, `compare`, `bank_add`, `bank_check`, `bank_remove` | Local blobs vs the 0.7 mm trigger. `inspect` only says PASS when the noise lets it see a trigger-sized defect. Compare against a golden scan, or the **good-memory bank** (flags anything unlike known-good parts). |
| Codes | `read_codes` | QR/1D/Data Matrix (OpenCV + zxing-cpp, every decoder on the raw, CLAHE and 2× CLAHE image) under bracketed brightness and **red**/green/blue/white light; one entry per symbology + data (`EAN_13` = `EAN13`), and an EAN-13 starting with 0 is reported as the 12-digit UPC-A printed on the label (`raw` keeps the EAN-13). Needs about 2.5–3 camera px per module. Exit 0 when a code is found, 1 when none. **Experimental:** `codes --anti-glare [--strip-cols C0,C1]` for a glossy label with a saturated specular stripe: many-level flood bracket → HDR (unsaturated samples, frames ordered by measured brightness), optional narrow DMD column-strip sweep (each pixel keeps its own strip's energy: no bloom), de-veil (local min/max envelope, ~9 × 3 px: gloss is an additive offset) → the normal decoder (`engine/sense/antiglare.py`). Mock-tested on synthetic glare; one bench trial. |
| Spectral | `spectral_capture`, `learn_material`, `classify`, `separate_light` | R/G/B LED reflectance signatures → material map. Direct vs global light from shifted checkerboards. |
| Grid & zones | `capture_map`, `project_grid`, `project_zones` | A lettered grid (A1, B2, …) photographed with upright labels. Say "H8:I11@96" to light and scan just those cells. Custom machine-vision zones. |
| Sensors | `spot_size`, `project_spot`, `teach_presence`, `check_presence`, `teach_alignment`, `check_alignment` | A laser-spot-sensor-style spot of a chosen **diameter in mm** (nominal or your **measured** mm/mirror). **Part present** (photoelectric, or background suppression by triangulation). **Alignment** (dx/dy/dθ vs a taught pose). |
| Cell & 3D | `export_mesh`, `cell_unwrap`, `cell_dents`, `bake_maps`, `checker_scan` | STL/OBJ (relief, base, open-arc cylinder), cell/tube unwrap → dent map + cell height, surface-map stack (cavity, normal, albedo, subsurface, dust). |
| Surfer + fast mode | `import_atlas`, `flood_verify`, `anti_glare_sweep`, `multi_orientation_scan`, `make_flash_pack` | Atlas regions as zones (`atlas:cell/cell`), flood-verify with a black surround, glare-dodging patterns, H/V/diagonal fusion; flash pattern packs for the triggered fast path (`docs/FLASH_PATTERN_MODE.md`). |
| Flash fast scan (1-bit, no trigger cable) | `make_fastscan_pack` | The column-axis multifreq set (4 steps × T 8 / 80 / 1280 + white / black) as 16 one-bit bit-planes of ONE 24-bit 912×1140 flash image (G0–G7, R0–R7), with two sync markers; full-field + zone-masked images (face ellipses + margin, a camera band mapped to the DMD). Fine level: column-exact ordered dither; mid / coarse: Floyd-Steinberg strips, decoded after a 3 px camera low-pass. The DLPC350 plays it on its internal timer (10 ms per slot, 160 ms loop) and the free-running camera is sorted in software, ~0.5 s per scene. Files only: a human loads the images and plays the sequence in the 4500 MKII GUI (`docs/FLASH_FASTSCAN_4500.md`: backup via the GUI's firmware link, firmware build, sequence settings, LED limits, return to video mode, rollback). CLI `fastscan-pack`; `tools/make_flash_pack.py --fastscan [--zone-ring NPY]… [--zone-band-cam NPY --zone-band-field NPZ] [--calibration DIR] [--study]`, `--unpack-check DIR`. |
| Flash soft-sync scan (camera only) | `fast_scan_flash`, `sort_flash_recording` | While a HUMAN plays the fast-scan pack in the 4500 MKII GUI (loaded with its firmware link; Pattern Sequence, Flash, Internal/External, Play Once off), the engine opens ONLY the camera: free run, a window around the zone (`roi` / `zone`, or the pack's zone through a projector map), fixed exposure (3 ms), the window's top frame rate, OldestFirst buffers, every frame with its camera timestamp and frame id. The frames are sorted in software: loop period by folding, pattern boundaries from the frames that straddle a change (fitted to µs), clean frames only (exposure inside one pattern, 250 µs guard), the sync markers + content checks anchor the pack's order, then the clean frames are averaged per pattern → the unchanged multifreq decode (mid / coarse prefiltered; px where that low-pass mixes two surfaces, and in the zone image px on the zone's edge found with the full-field sync white, are left undecoded; optional `fine_prefilter_px`) → ray-plane mm → a normal metric scan (meta `mode: flash_softsync`, per-pattern frame counts, confidence). The pack's manifest must describe exactly the loop the engine sorts (pattern version, dither, slot k on bit k = G0–R7, timing), else it is refused before the camera. Refused with the reason when the loop is not playing, stopped, misses / adds a pattern, plays out of order, runs another period, or the camera dropped frames. 10 ms slots: 2 loops = 0.34 s of frames in a zone window (321 fps), 0.51 s full frame. `dry_run` = the timing plan; the raw frames go to `output/flashscans` for `sort_flash_recording` (offline). CLI `scan-flash`, `sort-flash`; `docs/FLASH_SOFTSYNC.md`. Mock-tested only. |
| Metric calibration | `make_charuco`, `row_parity`, `capture_calib_pose`, `calibrate_stereo` | Printable ChArUco board (true-size PDF), diamond DMD row-parity check, board poses (flood + both-axis `multifreq` decode), then camera + projector (inverse camera, native DMD col/row, f_v ≈ 2 f_u) + `stereoCalibrate`. Reports per-pose and per-axis RMS; saved as a geometry-bound calibration. |
| Metric scan (mm) | `scan_metric`, `triangulate`, `export_metric_mesh` | Point cloud in mm (ray-plane from columns, ray-ray with both axes + a per-pixel consistency-error map), `.ply`, open-surface STL/OBJ in mm. Refused without a stereo calibration at the same geometry and fingerprint. |
| Metric measure | `cylinder_deviation`, `compare_metric`, `accuracy_check`, `predict_precision` | Cylinder fit → radial deviation map in mm with dent depth/area, point-to-plane ICP to a golden scan, known-object check (plane / step / cylinder / sphere: mean/max error, repeatability, GO/NO-GO), predicted σ_z before scanning. |
| Cell dents (shiny cells) | `capture_dent_set`, `calibrate_dent_null`, `detect_dents` | One pose = a long (66 667 µs: floods, sin0 T12/T36, sin90 T12/T36) and a short (16 667 µs: floods incl. 128, the same sinusoids, lines_v) pattern pass + one metric scan of the zone (`pattern_set="minimal"` drops sin90 T36: fewer frames, but it scored a small dent lower in the bench ablation). The two exposure presets may differ only in exposure. A fixed-radius cylinder from the metric cloud maps pixels to (θ, s) on the cell (`s_mm` from the axis point level with the camera's optical axis, `s_from_top_mm` from the top of the decoded cell). Four finders (pattern warp, photometric, HDR phase depth, reflected-line kink) are calibrated on a **known-good cell** of the same type in the same fixture (1.0 = its strongest response; a finder that does not respond there is excluded, with a warning) and fused: detection > 1.25 (a margin over the one good cell's own maximum), watch 0.8–1.25, edge. Each candidate: px + (θ, s), score, depth in mm (+ = pushed in; flagged unreliable in specular / dark-band zones, significant above 1.5 × the scan's depth-noise p90), size, per-finder evidence, kind dent / scratch / speck (advisory), overlay PNG. Legacy offline dent sets (`dataset.json`) work through the engine API and CLI (not the web UI). Bench numbers: the re-placed pose ("pose 2") was used while tuning, so it is not held-out data. |
| End faces (cell end face toward the camera) | `locate_face`, `aim_ring`, `check_ring`, `auto_intensity`, `targeted_face_scan`, `face_profile`, `face_golden`, `check_face` | A cell lying with its circular **end face** toward the rig. `locate_face` on a ray-plane metric scan: the face is the nearest face-shaped depth-continuous surface (nearer non-face objects are set aside), a neighbour only a few mm behind the face (a cooling tube) is split off by the face's own plane (sequential RANSAC planes; rim = surface at least `near_gap_mm` = 2 mm behind that plane), and when the depth rim still fails the face outline in the flood (the scan's flood pair, else the decode modulation) is used (`rim_source` in the target), its rim = contour points against far background (the true silhouette, not an undecoded band), a fixed-radius circle (the known diameter) in the face plane, the camera face mask, and a camera → DMD **affine** fitted on the decoded face pixels (residual reported, warning > 0.5 DMD px; the calibration's projector row axis is not used for aiming). `aim_ring` projects the ring + cross (the operator's placement guide, pumped hold); `check_ring` says ON_FACE / OFF (face flood coverage + where the ring lands). `auto_intensity` picks `projection.duty` for a ~200 DN face at a whole-frame exposure (never the LED currents). `targeted_face_scan` lights only the face ellipse + margin: N repeats of the depth set (ray-plane) + polar rings / spokes / floods. `face_profile`: height map in the face frame (0.25 mm grid; u right, v down, + toward the camera), radial profile, sections, flatness, repeatability, PNG / CSV / JSON / STL. `face_golden` (append-only, bound to geometry, face diameter, grid, stereo calibration) holds every detector's model; default detector **MF-CFAR + reflectance FIR** (`mf_fir`): per-repeat planes removed, deviation from the good placements' median **radial** profile, a dent-shaped matched filter bank (1 / 1.5 / 2 / 3 mm) with a local plane, z against the face's own robust noise (radial-band MAD), trigger at z ≥ 5.5 inside r ≤ 9.75 mm; first the height error predicted from the flood pair by a 7×7 camera-px FIR fitted on the good placements (coefficients stored with the golden; not applied to a placement whose flood saturates > 2 % of the face) is removed. `mf` = without the FIR; `band` = the prototype (median map, band max(4σ, 0.10 mm), rim ×2). Leave-one-out false triggers of every detector + a synthetic-dent battery (dents injected per camera pixel). `check_face` → dents (zone, r, θ, depth, width, z, dev, area), PASS / FAIL / INCOMPLETE (< 90 % of the face measured is never passed), overlay PNG; `detector=` picks another detector or a band trigger function. Bench (2 good cells, 4 placements): flatness 0.044–0.070 mm RMS (r < 9 mm), repeatability 0.011–0.051 mm, placement pairs 0.050–0.076 mm RMS; leave-one-out: `mf_fir` 0 false triggers (band 5 + 3 rim), detection at z 5.5 ≈ 45 % at 0.15 × 1.5 mm, 78 % at 0.15 × 2 mm and 0.20 × 1.5 mm, 89 % at 0.20 × 2 mm; 1 mm wide dents are below the camera's resolution (0.38 mm/px). Reliable: ≥ 0.2 mm deep × ≥ 2 mm wide, or ≥ 0.15 × 3 mm. |
| HDMI fast zone scan | `scan_fast` | The column-axis multifreq set only (default 4 steps × T 8 / 80 / 1280 DMD px + white / black = 14 patterns) lit through a zone (grid cells, an atlas region, a DMD mask `.npy`, and / or located faces' DMD ellipses + a margin), with the camera average and the HDMI hold overridden for this run only (the preset file is untouched; average < 2 is raised to 2 with a warning: over HDMI one frame per pattern flickers), ray-plane triangulated (the column axis carries the depth) → a normal metric scan that also keeps its flood pair (`flood.npz`) for the station measurement. Bench: average 2, hold 0.12 s = 5.3 s for the whole station scene. CLI `scan-fast`. |
| Station (cells + cooling tube) | `station_measure`, `station_golden`, `station_check` | Cells lying with their end faces toward the camera, held in a cooling tube (CT: a thin ribbon in a trapezoid wave, seen edge-on a few mm behind the faces). From one metric scan + a flood pair (the scan's own from `scan_fast`, another scan's, a file; else the decode's fringe modulation, with a warning): cells C1..Cn (image order) found on the flood (Hough; brightest discs that are flat and proud in depth), face planes (robust), face centres (sub-pixel flood rim edges back-projected onto the face plane, circle fitted there; `centre_method="ellipse_ray"` = the prototype's ray through the rim ellipse centre), the common face frame; height of each face vs the mean face plane (mm, + toward the camera), `Cj-Ci` along the common normal, spacing in the face plane. CT front edge: thin runs per camera column a few mm behind the face plane, outside every cell body (height band found automatically, or `ct_band_mm`), a trapezoid-wave fit of its centreline (pitch, amplitude, plateau, phase vs the cell-pair centre), part-fixed zones T1..Tn (one plateau + two half-ramps) with the edge height per zone (median / p5 / p95). PNG + JSON + CSV in `output/measurements`. `station_golden` (append-only, bound to geometry + fingerprint + stereo calibration): nominal values (mean of the good measurements), the zone model + numbering, user-set limits (tolerance around nominal and / or absolute bands, per item if wanted). `station_check` re-measures with the golden's parameters and zones → PASS / FAIL / INCOMPLETE (a golden item not measured is never passed) with the triggered items, PNG + JSON + CSV in `output/defects`. CLI `station`, `station-golden`, `station-check`, `stations`. Bench reproduction (saved scans): with the prototype's settings, its cell height difference and spacing to ≤ 0.001 mm and most CT zone heights to ≤ 0.005 mm (largest gaps 0.04 mm on one scan and 0.12 mm on the other: the prototype pooled its CT mask over both scans); the fast scans (average 2) repeat the slow reference within 0.03 mm (C2-C1) and 0.11 mm (CT zones); a real re-seat (0.27 mm) and a CT move (-3.2 mm) FAIL. These are repeatability and agreement numbers, not absolute accuracy: no known-size object has been measured through the station path yet. |
| Output bin | `list_library("outputs")` | Every result, by type, with absolute paths. |

Every numeric result carries `units` ("px" or "mm") and `warnings`. mm needs a calibration at the same geometry (rule 6).

**Dent path (shiny cells).** Stereo calibration first (the metric path below) → `dent-scan` a KNOWN-GOOD cell in the fixture's zone (`--preset` the dark 66 667 µs preset, `--short-preset` the 16 667 µs one) → `dent-null <it> --diameter MM` → `dent-scan` each cell → `dent-detect <it>` (exit 0 CLEAR, 1 FLAG, 4 the detector failed; the cell diameter defaults to the matching null's). A null calibration is bound to the geometry, the cell diameter, the fixture (the zone, or a name given with `dent-null --fixture` that `dent-detect --fixture` must then repeat), the pattern set and the stereo calibration; recalibrate when any changes. `--subset production` evaluates a richer legacy capture with only the production patterns.

**End-face path.** Stereo calibration first → the human places the cell → `face-locate --zone H10:I12` (or `face-locate <metric scan>`; the scan must decode x **and** y: `scan-fast --axis xy` or `scan-metric`, an x-only `scan-fast` is refused with that hint) → `face-aim --check` (exit 0 ON_FACE) → `face-scan cN --auto-intensity` → `face-profile cN`. Golden: `face-golden g1 g2 g3 g4` from GOOD placements (MF-CFAR + reflectance FIR by default; `--detector band`, `--z 5.0` for ~10× more false triggers and ~10 points more detection); per cell `face-check cN` (exit 0 PASS, 1 FAIL / INCOMPLETE). Keep the flood unsaturated (`--auto-intensity`, which needs a preset with `projection.invert` false, e.g. `--preset bench_9in_metric_bright`; on an inverted preset it is refused with that hint): a saturated face is noisier and loses the FIR. `face-aim --hold 30` keeps the ring up while the next cell is placed; `faces` lists targets, face scans and goldens.

**Station path.** Stereo calibration first → the human places the cells in the cooling tube → `scan-fast st1 --zone F10:J12` (≈ 5 s over HDMI; keeps its flood pair) → `station st1` (cells, `C2-C1`, spacing, CT zones T1..Tn, PNG) → golden from GOOD placements `station-golden st1 st2 --cell-tol 0.15 --ct-tol 0.3` (or absolute bands: `--cell-band LO,HI`, `--ct-band LO,HI`, `--band T3=LO,HI`) → per part `scan-fast` + `station-check <it>` (exit 0 PASS, 1 FAIL / INCOMPLETE, 2 refused, e.g. a cell not found, 4 crashed). Zone numbering follows the golden; `--skip-zones 1,0` (on `station`, or on `station-golden` for metric-scan items, with `--edge-band`, `--roi`, `--centre`, `--cells`, `--diameter`) drops a lead-in zone that is not part of the wave. A face cut by the zone edge keeps its height but is warned about: its centre, so the spacing, moves. A cooling tube only a few mm behind a face no longer stops `face-locate` (plane split; flood-outline fallback).

**Metric path (stereo calibration).** `make_charuco` → print at 100 %, caliper a square → `row_parity` → `capture_calib_pose` × 12–20 (tilts ±30°) → `calibrate_stereo --square MM` (camera RMS < 0.3 px, projector < 0.5 px per axis) → `accuracy_check` on a known object → `scan_metric` → `cylinder_deviation` / `compare_metric` → `export_metric_mesh`. CLI: `charuco`, `parity`, `calib-pose`, `calibrate-stereo`, `scan-metric`, `triangulate`, `cyldev`, `accuracy`, `precision`, `metric-mesh`, `metric`; dents: `dent-scan`, `dent-null`, `dent-detect`, `dentsets`. In mock mode `calibrate-stereo --mock-poses 12` lets the simulator place the board.

## Where the data goes

Everything is under `library\` (the default is next to this README; override with `SWISS_SCANNER_LIBRARY`):

| Folder | Contents |
|---|---|
| `bases\<geometry>\` | **base library** (empty-scene references) |
| `objects\` | 3D scans: field, relief, meta, exports.jsonl |
| `calibrations\<geometry>\` | Z / XY calibration fits, stereo calibrations, diamond row parity, dent null calibrations (`dent_null.json`) |
| `calib_poses\<geometry>\` | ChArUco board poses for `calibrate_stereo` |
| `metric\<geometry>\` | metric scans: field, cloud (xyz mm + consistency), meta |
| `dentsets\<geometry>\` | dent-detector captures: long / short exposure frames (cropped to the zone), meta naming the metric scan |
| `faces\<geometry>\` | located end faces: face plane, rim circle, camera face mask, camera → DMD affine + aim ring |
| `facescans\<geometry>\` | targeted end-face scans: face DMD mask, polar pattern frames, meta naming the repeat metric scans |
| `face_golden\<geometry>\` | golden end faces: MF radial profile + reflectance FIR coefficients, golden + band maps, placements' height maps, the detector settings, leave-one-out + synthetic battery |
| `station_golden\<geometry>\` | station goldens: nominal cell height differences, spacings, cooling-tube zone heights, the zone model + numbering, the limits |
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
- `bench_9in_metric_dark`: the same geometry at 66 667 µs, the long-exposure preset for shiny cells (dent sets).
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

**Setup.** FLIR BFS-U3-16S2M, 1440 × 1080 mono, 3.45 µm pixels; lens ≈ 8 mm (*derived*: 0.30 mm/px measured at ≈ 640 mm gives f ≈ 3.45 µm × 640 mm ÷ 0.30 mm ≈ 7.4 mm); DLP4500 projector at 228.6 mm baseline, 762 mm standoff. Field of view at ≈ 640 mm ≈ 430 × 320 mm.

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
- End-face dents: detection floor ≈ 0.2 mm deep and ≥ 1.5 mm wide.
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

1. **HDMI-only scan path.** The engine never imports the DLPC350 USB driver or `hid`. USB display commands raise `SafetyViolation`. `engine/capture/dmd_control.py` is the separate, human-run USB diagnostics tool; it takes the rig lock for write commands and is never imported by the engine (a test enforces this).
2. **Pumped holds.** The projector window is kept painted with `cv2.waitKey`; there are no bare sleeps (test-enforced).
3. **LED thermal limits.** Burst R180 G255 B230, average R120 G180 B160, plus duration limits (`check_led_duration`). Record the GUI currents in `projection.led` so this rule can be checked.
4. **Mono camera ⇒ bw scheme.** Colour schemes are flagged.
5. **Base and scan share a geometry.** That means the same `geometry_id` **and** the same geometry fingerprint: rotate90, size, coded axis, rig side/baseline/standoff.
6. **mm only with a calibration at that geometry and fingerprint**, read from the library. A trashed calibration stops giving mm immediately. Mock data uses `mock-<geometry_id>`.
7. **Append-only library.** Writes use exclusive-create. `delete` is one atomic move to `trash/`; an entry with an open file raises `EntryInUse` and is left untouched.
8. **One driver.** A machine-wide rig lock at `%PROGRAMDATA%\swiss_army_scanner\rig.lock` is released by the OS if the holder dies. The web UI serializes its own jobs on one worker.
9. **Physical steps are human-only.** Agents ask and wait. Part-present sensors are taught only with `empty_confirmed` after the human confirms.

## Tests

```bash
python -m pytest -q                                   # full suite, mock only
python -m pytest -q tests/test_rules.py tests/test_cli.py tests/test_run_job_mock.py
python -m pytest -q tests_sensor                       # the seed package, in its own process
```

Optional bench-data tests skip unless you point them at real files: `SCANNER_TEST_ATLAS` (a held-cell atlas `.json`) and `SCANNER_TEST_REAL_CAL` (a stereo calibration folder).

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
| Hardware-trigger pattern mode (USB, guarded) | future (architecture step 10) |
| Flash soft-sync fast scan (1-bit pack, free-running camera, software sort) | ✅ built, mock-tested (simulated DLP loop + free-running camera, GUI mistakes, dropped frames; pack → loop → camera → sort → decode → metric scan end to end vs an ideal synchronous capture and the HDMI fringes, `tests/test_flash_e2e_mock.py`); PySpin free-run code and the real loop not yet run on the rig |
| Stereo XYZ calibration, metric scan, cylinder deviation, accuracy check, metric mesh | ✅ built, mock-tested (pinhole stereo mock bench with known K/R/t); real-rig trials next |
| Dent detection on shiny cells (dent sets, null calibration, fused finders) | ✅ built; synthetic + mock tests; reproduced offline on the saved bench dent sets (see `engine/measure/dents/`); production capture not yet run on the rig |
| End faces (locate, aim / check ring, auto intensity, targeted scan, profile, golden, check) | ✅ built; synthetic + mock tests; reproduced offline on the saved bench end-face scans (`engine/measure/endface/`); the productised capture not yet run on the rig |
| HDMI fast zone scan, station measurement (cells + cooling tube: heights, spacing, zones T1..Tn, golden, check) | ✅ built; synthetic + mock tests; reproduced offline on the saved bench station scans (`engine/measure/station/`); `scan-fast` and `station` run on the rig through the engine |

The engine verbs (`api.*`, the `scan` CLI, the web UI) are the only supported path to the hardware.

## License

Apache License 2.0: see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). The rig safety rules above are guidance; operating the hardware safely is your responsibility.

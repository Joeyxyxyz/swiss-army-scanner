# Calibration fixture and check block

> **Status: mock-tested only.** Neither the fixture nor the check block has been printed or
> used on a real rig yet. Everything below is validated on the simulated stereo bench
> (`engine/capture/mock_stereo.py`, `tests/test_calib_fixture_mock.py`). Prove it on your rig
> (section 6) before you rely on it.

## 1. Why

The stereo calibration (`calibrate-stereo`) needs 12-20 views of a ChArUco board at
different tilts. Holding a board still, by hand, 12-20 times is slow and the coverage depends
on the operator.

The **fixture** is a printed bracket with N flat faces (default 4) at known, different angles.
Each face carries its **own** ChArUco board with a **disjoint marker id range**, so one camera
frame shows N boards and each is detected on its own. Each face is one planar view, so one
capture of the fixture gives N views, and **3 placements give 12 views**. That takes about
1-2 minutes, and nobody holds anything.

The **check block** is a printed object of known geometry: a step gauge, a flat reference
face, a half-round bar and a dome. `scan accuracy-block` scans it and compares what the
scanner measures with what **your print** measures with calipers. This tells you whether the
calibration gives true millimetres.

## 2. Print files: `tools/make_calib_fixture.py`

```
python tools/make_calib_fixture.py --out <folder>                 # fixture + check block
python tools/make_calib_fixture.py --out <folder> --faces 3 --square 18 --squares 5,4
python tools/make_calib_fixture.py --out <folder> --face-style recess --bed 250,210
python tools/make_calib_fixture.py --out <folder> --no-block | --block-only
```

The tool writes into a fresh folder, `<folder>/<timestamp>_<name>/`, and never overwrites a
file. It needs only numpy, plus cv2 for the sticker images:

| file | what |
|---|---|
| `fixture_spec.json` | faces (board, marker id range, tilt, pose of the board in the fixture frame, outward normal, `square_mm_measured`), printable parts, design parameters |
| `fixture_face<k>_ids<a>-<b>.png` / `.pdf` | one sticker per face at exact print scale (the PNG records its dpi, the PDF page is the sticker size) |
| `fixture_base.stl` (`_tile<i>` when split), `fixture_wedge_<k>.stl`, `fixture_splice_x<n>.stl` | binary STL (`--ascii` for ASCII), mm, closed (every edge in exactly two triangles), already in print orientation |
| `checkblock_spec.json`, `checkblock.stl` | the check block: nominal dimensions in its own frame, tolerances |
| `checkblock_measured_TEMPLATE.json` | the values you measure on the print (fill in, save under another name) |
| `README.txt` | print, assembly and measuring steps, the mesh sizes |

Defaults (all of them are parameters):

| | default | why |
|---|---|---|
| faces | 4, tilted 30 deg, each looking outward (-x, +y, +x, -y) | four orientations per capture; no face hides another from the front |
| board per face | 6 x 5 squares of **16 mm**, markers 0.7 x square, `DICT_5X5_250`, ids from 100 (face k: 100 + 15k ...) | at ~700 mm with a 1440 x 1080 camera and an ~8 mm lens a 16 mm square is ~55 camera px. Keep squares at least ~15 mm at 650-800 mm (`--square`). Ids from 100 never collide with the standard single board (`DICT_5X5_100`, ids 0-34) |
| sticker | board + 3 mm white margin, 0.3 mm thick (paper + laminate + adhesive) | the margin is the quiet zone around the outer markers |
| face | sticker + 2 mm border; `ridges` (2 mm ridges, 2 mm grooves, 0.4 mm deep) | the sticker rests on the ridge tops, and glue lumps sink into the grooves |
| base plate | 199 x 199 x 8 mm, one 3 mm pocket per wedge (0.2 mm clearance per side) | fits a 220 x 220 mm bed; with more faces the plate is split into tiles with glued splice plates |
| wedges | 106 x 90 mm face, up to ~59 mm tall, **printed face-down** | the face is printed on the bed, so it is as flat as the bed |
| check block | 148 x 64 mm, 8 mm slab; steps 25/20/15/10/5 mm; D-profile bar dia 20 on a 3 mm pedestal; dome dia 24 | fits the bed; every feature is far enough from the others for a clean fit |

## 3. Print and assemble

1. **Material.** Use matte WHITE PLA or PETG. Glossy or silk filament glints in the fringes.
   Print at 0.12-0.2 mm layers with 3 perimeters, 15-20 % infill and no supports. Each wedge
   is roughly 270 cm3 of solid, so plan the filament.
2. **Wedges face-down** (`ridges` / `flat`). Print on a flat glass or PEI bed and don't
   squish the first layer. Remove any elephant-foot lip at the face edge. With
   `--face-style recess` the wedges print bottom-down, so the face is a printed slope that is
   flat only to the layer steps. Use that style only when you must.
3. **Stickers.** Print each `fixture_face<k>_ids<a>-<b>.pdf` at **100 % / actual size**, never
   "fit to page", on matte paper (a laser printer is preferred). Laminate with **matte** film,
   or not at all, because glossy film glints. Glue each sticker flat with spray adhesive,
   without bubbles or wrinkles. Stick face k's sticker on wedge k, roughly centred. Where it
   sits on the face, and how it is turned, does not matter to the calibration.
4. **Assemble.** Drop wedge k into the pocket the spec gives face k, with its **low edge
   toward the outside** so every face looks outward. A drop of glue per wedge holds it. The
   fixture must not flex or rattle (section 4).
5. **Check flatness.** Lay a steel straightedge across each face, along its length and both
   diagonals. A 0.1 mm feeler gauge should not fit under it anywhere.
6. **Caliper one square per face.** Measure over 5 squares and divide by 5. Write the value
   into `faces[k].square_mm_measured` of `fixture_spec.json`. The printer's scale error goes
   straight into every mm the scanner reports otherwise.

## 4. Calibrate in 3 placements

```
scan parity                                                   once per rig (diamond DMD)
scan calib-pose --fixture fixture_spec.json --set fx          placement 1
scan calib-pose --fixture fixture_spec.json --set fx          placement 2
scan calib-pose --fixture fixture_spec.json --set fx          placement 3
scan calibrate-stereo --set fx --fixture fixture_spec.json
```

**Placements.** Set the fixture down facing the camera, capture, then move it:

- put it **left, right and centre** of the field;
- **turn it about its own axis by 20-45 deg** differently each time;
- tip it by about 10-15 deg (prop one edge up);
- change the distance by 30-70 mm.

On the mock rig the turn matters most. Placements turned 20-45 deg recover the focal length
within 0.3 px after the refinement below. Placements turned only up to +-20 deg still left it
1-6 px off. All faces should show in every placement. `calib-pose` lists the corners per
face and exits 1 when fewer than 2 faces are usable. Each capture is the normal flood +
both-axis multifreq sequence, and the placement is stored once, with all its boards
(`calib_poses/<geometry>/<ts>_fx/`: `boards`, `fixture` with the spec and the corners per face).

**What `calibrate-stereo` does with a fixture set:**

1. Every detected face of every placement becomes one planar view, and the unchanged
   calibration runs on them: `cv2.calibrateCamera`, the projector as an inverse camera, and
   `stereoCalibrate`.
2. **Rigid-fixture refinement** (`engine/calibrate/rigid.py`). A ~100 mm face at ~700 mm is
   a weak view on its own: its image is nearly affine, so focal length trades against
   distance. On the mock, the independent fit lands 2-15 px off in focal length, 2-18 px off
   in principal point and 0.1-0.6 deg off in R.

   The fixture is rigid, though: face k sits at the same pose relative to face 0 in every
   placement, whatever the print made of it. The refinement estimates those face-to-face
   poses as unknowns shared by all placements. It does this jointly with the camera
   intrinsics and distortion, the projector intrinsics and the stereo R, t, in one bundle
   adjustment (scipy least squares). In effect the fixture becomes a 3D target.

   It never assumes the PRINTED angles: the spec poses are only the starting point. On the
   mock, 3 placements then recover f within 0.3 px, c within ~1.3 px, R within 0.03 deg and
   t within 0.1 mm. That is as good as, or better than, the 8-pose single-board test.

   The refinement fails closed. If the joint fit is clearly worse than the independent views
   (camera or projector RMS > 1.5x, or > +0.05 px), the fixture moved or flexed between
   placements. The independent result is then kept and the summary says so. `--no-rigid`
   turns the refinement off.
3. **Face-angle check** (a print / assembly sanity check, NOT a constraint). The angle between
   every pair of face normals, as estimated, is compared with the spec. A difference over
   1 deg is a warning: check the print, the wedges in their pockets and that each sticker
   lies flat. The calibration itself does not rest on these angles.

`--square` is refused for a fixture set: each face takes its own caliper value from the spec.
The calibration is saved like any other (`calibrations/<geometry>/<ts>_stereo/`). Its `board`
records the fixture and its faces, and `report.json` adds the views, the refinement and the
face check.

Mock: `scan --mock calibrate-stereo --mock-fixture 3` places the fixture 3 times. Repeated
`scan --mock calib-pose --fixture SPEC` places it at the next placement of the same plan
(`engine/capture/mock_stereo.FIXTURE_PLAN`).

## 5. Verify with the check block

1. **Print it bottom-down,** matte white, 0.12-0.2 mm layers, 4 perimeters, 30 % infill.
2. **Measure the print** and fill in `checkblock_measured_TEMPLATE.json`, saved under another
   name:
   - each step's height above the reference face next to it (caliper depth rod or height
     gauge, 3 places, averaged);
   - the bar's diameter across its flats (calipers) and its top height;
   - the dome's apex height.

   If `dome_diameter_mm` is left null, it is taken as 2 x the dome height (a hemisphere).
3. **Place it** with the reference face to the camera and the long side horizontal. Put the
   **tallest step on the side away from the projector**, so no step shadows the next.
4. Run:

   ```
   scan accuracy-block --block checkblock_spec.json --measured <your values>.json [--repeats 3]
   ```

The scan is registered to the nominal model: the reference plane first, then point-to-plane
ICP from four in-plane turns. Each feature is then cut out by its footprint with a margin, and
the scanner measures:

| row | how | tolerance (default, spec / options) |
|---|---|---|
| `reference_flatness_rms_mm` | plane fit RMS over the bare reference face (points > 0.5 mm off are counted as outliers; > 2 % outliers = not measured) | <= 0.10 (`--flatness-max`) |
| `stepN_height_mm` | median distance of the step top above the reference plane | +-0.10 (`--tol-height`) |
| `cylinder_diameter_mm` / `_height_mm` | cylinder fit on the arc / its top line above the reference | +-0.15 / +-0.10 |
| `dome_diameter_mm` / `_height_mm` | trimmed sphere fit / its apex above the reference | +-0.15 / +-0.10 |

Results:

| disposition | exit | meaning |
|---|---|---|
| CONTINUE | 0 | the rig measures the block within tolerance |
| REJECT | 1 | a measured row is out of tolerance against your caliper values: recalibrate |
| PULL | 5 | doubt: no `--measured` (compared with NOMINAL values: a print is never exactly its model), a feature not seen in every scan, the block not found, too many outliers |

The report (rows, registration, dispositions) goes to `output/measurements/`. With repeats,
every scan must pass, and each row also reports the mean and sigma.

The flatness row also catches decode errors. On the mock, a calibration made without
`scan parity` (diamond rows decoded as a square grid) measured the step heights correctly,
because the median cancels the zigzag, but showed a 0.3 mm flatness RMS: FAIL.

## 6. Limitations (read before trusting it)

- **Mock-tested only.** Not yet printed, assembled, captured or verified on a real rig. The
  mock has exact geometry, ideal matte surfaces and simple camera noise.
- **Print flatness.** A face is only as flat as the print. Face-down on glass is the best
  case; check with a straightedge. A warped wedge breaks the "planar view" assumption of its
  face.
- **Sticker flatness.** Bubbles, a curled corner or a thick glue lump are local bumps in the
  board. Glossy laminate glints in the fringes and can cost corners.
- **Measure the printed object.** The check block is only as true as your calipers and your
  care (+-0.02-0.03 mm for good calipers). The fixture's squares are only as right as the
  value you enter in the spec.
- **Rigidity.** The refinement assumes the fixture does not change between placements: glue
  the wedges and don't touch the stickers. If it flexes, the refinement is rejected and the
  weaker independent fit is used, with a warning.
- **Shadows.** A wedge's high wall can shadow part of a neighbouring face from the projector.
  On the mock, corners in shadow are lost (fewer corners), not biased. On the rig a shadowed
  face may decode poorly; turn the fixture so all faces show.
- **The face-angle check is a sanity check,** never a constraint. A 1 deg warning means
  "look at the fixture", not "the calibration is wrong".
- **Thermal and creep.** PLA creeps when warm. Keep the fixture and block out of the sun and
  re-measure them after a knock or a hot day.

## 7. Tests

`tests/test_calib_fixture_mock.py` covers:

- the generator: closed meshes on the bed, specs that round-trip, stickers at exact scale
  that detect only their own ids, face poses on the wedge tops;
- one capture giving one view per face with disjoint ids;
- the 3-placement calibration against the mock rig's K / R / t, with the same tolerances as
  `tests/test_stereo_calibration_mock.py`;
- the refinement beating the independent face views;
- the face-angle check;
- the check block: caliper values -> CONTINUE, out of tolerance -> REJECT, nominal -> PULL,
  not in view -> PULL;
- the engine end to end with camera noise.

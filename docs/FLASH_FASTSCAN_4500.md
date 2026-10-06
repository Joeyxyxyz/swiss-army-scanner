# Fast scan from flash: 1-bit patterns, internal timer, soft sync (4500 MKII)

**Status:** the patterns, the pack tool and this procedure exist and are tested offline (`tests/test_flash_fastscan.py`). The capture side is `docs/FLASH_SOFTSYNC.md`, and the two are tested together end to end in mock (`tests/test_flash_e2e_mock.py`, §9). **Nothing has run on the rig yet.** The HDMI video path is still the default scan path.

**Goal:** the column-axis multifreq measurement of the fast HDMI scan (4 steps × periods 8 / 80 / 1280 DMD columns + white / black, 5.3 s for the two cell end faces and the cooling-tube band) in about **0.5 s**. The DLPC350 plays the patterns from flash on its own timer. The camera free-runs faster than the pattern rate, and the frames are sorted in software. There is no trigger cable.

| | HDMI video mode (today) | Flash fast scan (this doc) |
|---|---|---|
| Pattern source | desktop window over HDMI, 8-bit fringes | one 24-bit image in flash, 16 one-bit bit-planes |
| Per pattern | about 0.38 s (display latency + hold) | 10 ms |
| Whole scene | 5.3 s | 160 ms per loop; 3 loops = 0.48 s |
| Sync | software hold, flush, grab | none: camera free-runs, frames sorted by timestamp and content |
| Who switches modes | nobody (always video) | **a human, in the 4500 MKII GUI, never mid-scan** |

**Why 1-bit.** In pattern mode an 8-bit pattern is shown as a burst of PWM bit-planes inside its exposure, so an unsynchronised camera catches a random slice of it. A 1-bit pattern is constant for its whole exposure. Any camera frame that lies inside one pattern slot is a clean sample of that pattern. The sinusoids are made from dithered stripes, and the projector blur and the camera pixel low-pass them.

**Rule 1 still holds.** The engine never sends the projector a USB or HID command in the scan path. `engine/bases/onebit.py` and `tools/make_flash_pack.py` write files only; `dmd_control` is not imported. Flashing, entering pattern mode, Play / Stop and going back to video mode are all done by a human in the GUI.

## 1. Make the pack

```
python tools/make_flash_pack.py --fastscan                                  # full-field image only
python tools/make_flash_pack.py --fastscan --label bench_faces_tube ^
    --zone-ring <work>\face_c1\face_dmd_ring.npy ^
    --zone-ring <work>\face_c2\face_dmd_ring.npy ^
    --zone-band-cam <work>\ct_mask_cam.npy ^
    --zone-band-field library\metric\<geometry>\<locate scan id>\field.npz ^
    --calibration library\calibrations\<geometry>\<stereo calibration id> --study
python tools/make_flash_pack.py --unpack-check <pack folder>                # re-verify from disk
```

The same build is the `make_fastscan_pack` verb (web UI) and `scan fastscan-pack` in the CLI (`--zone-ring`, `--zone-mask`, `--band-cam`, `--band-field`, `--margin`, `--calibration`, `--study`, `--label`).

- `--zone-ring` takes a `face_dmd_ring.npy` (DMD col, row points on a face rim). The tool fits an ellipse and adds `--zone-margin` (default 6 DMD px) on each side.
- `--zone-band-cam` + `--zone-band-field` map a camera-space band (the cooling tube) to the DMD through a **both-axes** decoded field (`corr_x` and `corr_y`). Small specks are dropped and the band is closed.
- `--zone-mask` adds any DMD-shaped mask `.npy`.
- `--calibration` gives the real scale: **1.54 camera px per DMD column** and **1.84 mm of range per DMD column** at 780 mm on the camera axis (1.47 px and 1.94 mm on the faces, measured). The 0.36 figure sometimes quoted for this rig is **mm per camera pixel** (0.364 mm at 780 mm), not camera px per column.
- `--led R,G,B` sets the start currents written in the procedure (default R35 G50 B45). Values above the continuous limits are refused.
- `--singles` also writes each slot as its own 912×1140 BMP, for the GUI's Create Images tool (see §4).

The pack goes to the library output bin, `library\output\flashpacks\<timestamp>_<label>\`, with a row in `library\output\index.csv`, or under `--out`. A folder is never reused.

| File | What |
|---|---|
| `images/flash_00_full.bmp` | the full-field image: 912×1140, 24-bit, uncompressed |
| `images/flash_01_zone.bmp` | the zone image (only with zone sources): decode slots lit only inside the zone |
| `manifest.json` | slot → image, bit-plane, meaning, exposure / period, timing, LED, decode, zone provenance (with sha256), round-trip result, study |
| `procedure.txt` | the click list for this pack in the GUI's own words (the short form of §5) |
| `preview_full.png`, `preview_zone.png`, `zone_mask.png` | contact sheets of the 16 slots and the DMD zone |

Before writing, the tool checks that every bit-plane unpacks exactly to its pattern and that the unused planes (B0–B7) are zero. It does this in memory and again after reading the BMPs back from disk. `--unpack-check` repeats the check later and also checks the BMP hashes.

## 2. The pattern set

One loop is 16 one-bit patterns. Slot *k* is bit *k* of the 24-bit image: G0–G7, then R0–R7. B0–B7 are unused.

| Slot | Plane | Pattern | Decode index |
|---|---|---|---|
| 0 | G0 | **SYNC A:** every mirror on, full field. The brightest slot. | – |
| 1 | G1 | **SYNC B:** horizontal bars, 16 rows on / 16 off, full field. The only slot with structure along camera y. It always follows SYNC A. | – |
| 2 | G2 | decode white | 0 |
| 3 | G3 | decode black | 1 |
| 4–7 | G4–G7 | fine fringe, T = 8 columns, steps 0–3 (+0°, 90°, 180°, 270°) | 2–5 |
| 8–11 | R0–R3 | mid fringe, T = 80, steps 0–3 | 6–9 |
| 12–15 | R4–R7 | coarse fringe, T = 1280, steps 0–3 | 10–13 |

Slots 2–15 are exactly the 14 frames of `MultiFreqBasis(steps=4, periods_px=(8, 80, 1280), axis="x")`, in its order. The captured frames go through the unchanged multifreq decoder (`onebit.decode`, which first low-passes the mid and coarse frames by 3 camera px).

**Sync markers.** The loop repeats forever and the camera starts anywhere in it. SYNC A → SYNC B is a pair that appears once per loop: SYNC B is the only frame whose image varies along y rather than x, and SYNC A, the full-field white, is immediately before it (in the full image it is exactly as bright as slot 2; in the zone image it is the only full-field white, which is also what lets the decode find the zone's edge). Black (slot 3) is the darkest. The sorter uses these to find the start and to confirm the order. It also checks that the four steps of each fringe level have near-equal means. The sync markers stay full field in the zone variant, so they can be found in any region of interest.

**Zone variant.** For a fixed fixture: the same 16 slots, but slots 2–15 are black outside the zone (the two face ellipses + 6 px, and the tube band). Both images use the same plane layout, so switching variant in the GUI means only a different Image Index. With the bench faces and the tube band, the zone lights about 1 % of the DMD.

## 3. Why these patterns: the offline study

`onebit.study()` (and `--study`, which writes it into the manifest) renders every slot the way the camera sees it: mirror box, Gaussian projector defocus σ = 0.5–2 DMD columns, camera pixel box, sampled at the calibrated 1.54 camera px per column on a grid rotated 3°. It decodes with the multifreq formula and compares against the ideal phase 2πx/T. The fine level is also compared against the **HDMI 8-bit fringe seen through the same optics**, because that is what the stereo calibration was made with. This is done on the square grid and on the diamond DMD with either row parity. The results below are in DMD columns and in mm of range at 780 mm (× 1.84 mm per column).

**Fine level (T = 8): it sets the depth precision.**

| Projector σ (cols) | Dithered, rms | Square wave, rms (simulated / closed form) | Dithered + 1 px fine prefilter | Diamond, vs HDMI (dithered) | Modulation, dithered / square |
|---|---|---|---|---|---|
| 0.5 (sharp) | 0.018 col = 0.033 mm | 0.168 / 0.130 col = 0.31 mm | 0.004 col = 0.007 mm | ±0.043 col bias, 0.037 col rms | 0.46 / 0.59 |
| 0.75 | 0.005 col = 0.010 mm | 0.070 / 0.056 col = 0.13 mm | 0.002 col = 0.003 mm | ±0.043, 0.016 | 0.42 / 0.53 |
| 1.0 | 0.002 col = 0.003 mm | 0.023 / 0.019 col = 0.043 mm | 0.001 col | ±0.043, 0.006 | 0.36 / 0.46 |
| 1.5 | < 0.001 col | 0.001 col | < 0.001 col | ±0.043, 0.002 | 0.25 / 0.31 |
| 2.0 | < 0.001 col | < 0.001 col | < 0.001 col | ±0.043, 0.003 | 0.14 / 0.18 |

- **Chosen: column-exact ordered dither.** Every column carries one level of the 8-sample profile (4, 3, 2, 0, 0, 0, 2, 3) / 4. That profile is symmetric, so it has no phase bias, and it was searched (`best_fine_profile`) to minimise DFT bin 3. Bin 3 is the only bin a 4-step decode of a period-8 profile is sensitive to (harmonics 3 and 5). Its |F3|/|F1| is 2.9 %. A level *j*/4 lights *j* of every 4 rows. Every step is an exact 2-column translate, so the texture is common to all four steps.
- **The binary square wave is worse wherever the projector is sharp.** 4-step phase shifting on a square wave has a known harmonic error. Harmonics 3, 5, 7, … (every *h* = 4*m* ± 1) leak into a ripple at 4× the fringe phase. `onebit.square_wave_error` gives it in closed form from the optics MTF. It is 0.39 col peak (0.72 mm) with no defocus, and 0.13 col rms at σ = 0.5. It only becomes negligible (about 0.001 col) for σ ≥ 1.5 columns. There, its 27 % higher modulation gives slightly lower noise than the dither. That would need a visibly defocused projector, and **refocusing voids the stereo calibration**, so the dither stays.
- **The bench projector is sharp.** The HDMI fast scan's fine-level modulation (median B 20.8 DN at a mean level of 19.1 DN) means little blur, so σ ≈ 0.5 is the row that applies. At that focus the dither texture leaves 0.018 col rms. The optional **fine prefilter** removes most of it: `onebit.decode(..., fine_px=1.0)`, a 1 camera px Gaussian on the fine frames only. In the scan it is `scan-flash --fine-prefilter 1` (`fine_prefilter_px`). It brings the texture error to 0.004 col (0.007 mm), but it is a lateral blur of about 0.36 mm. In the mock end-to-end test, a 1.5 mm FWHM dent reads about 15 % shallower with it. The manifest records it as `fine_sigma_camera_px_recommended`. It is **off by default**. Compare both on the bench before using it for dents.
- **Diamond DMD.** On the DLP4500, alternate rows sit half a column to the right. The calibration used parity null (square grid), and the HDMI fringes light every row of a column equally. The 1-bit tile lights different rows per level. Against the HDMI reference, the fine level therefore reads a **constant offset of ±0.043 col (±0.08 mm of range)**, with its sign set by the row parity, plus a little extra ripple. The offset is common to the whole field, so relative results (C2 − C1 height, dent depth against the local face) do not see it. Absolute range compared with an HDMI scan may differ by up to 0.08 mm. A tile that lights as many even as odd rows at every level has no offset, but it needs 8-row tiles and doubles the texture error at σ = 0.5 (simulated offline), so it was rejected.
- **Mid and coarse (T = 80, 1280): they only fix fringe orders.** Their tolerances are T_finer / 4, so 2 and 20 columns. They use Floyd-Steinberg error diffusion (serpentine) of one wide strip, and step *k* is that strip shifted by *k*T/4 columns, so the steps are exact translates. Their texture is visible at a sharp projector, so they are decoded from frames low-passed by 3 camera px. After that, mid is 0.076 col rms (p99 0.19, tolerance 2) and coarse is 1.2 col rms (p99 3.9, tolerance 20). End to end (14 frames → `MultiFreqBasis.decode`), unwrap rejects 0.11 % of pixels at σ = 0.5 and none at σ ≥ 0.75 on a flat scene.
- **Depth edges.** Where a face rim stands against the wall behind it, the 3 px low-pass mixes two surfaces. The mixed mid phase can then pick the wrong fine order, an 8-column error (about 15 mm). The 8-bit HDMI decode has no low-pass and does not make this error. So the decode leaves undecoded every pixel whose mid or coarse phasor loses coherence under the low-pass (`onebit.phasor_coherence` < 0.86). In the mock, every wrong-order pixel was below 0.83, and smooth surfaces were above 0.85 at p0.1. This removes a few pixels along each rim.
- **Noise.** This is an estimate: 1.5 DN camera noise per frame at a 200 DN white, and about 4 clean frames averaged per slot (3 loops × 1.47). That gives a random fine-level error of about **0.014 mm per camera pixel**, or 0.004 mm with the 1 px fine prefilter. At the same currents, pattern-mode white gives each 3 ms frame about half the light of a 16.7 ms HDMI frame, and about 4 frames are averaged instead of the HDMI scan's 2. The total is roughly the HDMI fast scan's signal.

**Expected error at 780 mm (sharp projector, per camera pixel):** systematic 0.03 mm rms (0.007 mm with the fine prefilter), plus random about 0.014 mm, plus the diamond ripple (0.07 mm rms raw, 0.02 mm with the prefilter) if the row parity is not null. On top of that sits a constant absolute offset of up to ±0.08 mm. Face means (about 1800 px) and the C2 − C1 difference average the random and ripple parts down well below 0.01 mm. The 1.5 mm FWHM dent sits on the same length scale as the ripple (2 columns ≈ 1.1 mm on the part). The fine prefilter removes the ripple, but it also makes the dent read shallower (see above). On the bench, scan the dent both ways and compare with the HDMI fast scan before choosing.

## 4. Packing: bit-planes and BMP bytes

- 912 × 1140, 24-bit, uncompressed, bottom-up rows, B-G-R bytes per pixel (standard BMP). The Firmware Build tab only accepts 912×1140 24-bit BMPs.
- Bit *p* of the packed 24-bit value → plane: 0–7 = G0–G7, 8–15 = R0–R7, 16–23 = B0–B7. G is stored in the BMP's **green** byte, R in red and B in blue, with bit 0 as the LSB. This follows the DLPC350 pattern-number mapping (DLPU010G Table 2-70) and the TI SDK packer (`lcr4500.cpp`: `g = bits 0–7`, `r = bits 8–15`, `b = bits 16–23`).
- Slot *k* = bit *k*. A 1-bit pattern is its bit (mirror on = 1).
- **Which byte does the GUI call G?** The user's guide notes that its Create Images preview shows G-planes as red and R-planes as green. That could mean the GUI's naming is the other way round. **It does not matter for this loop.** If the GUI's G0–G7 read our red byte, the controller plays slots 8–15 then 0–7. That is the same repeating loop rotated by 8 slots, and the sorter anchors on SYNC A → SYNC B, not on the first frame (`plane_swap_play_order`, tested). The **plane check** in §5.4 shows which case the unit is in. If the bits are reversed inside a byte (the GUI's G1 shows our bit 6), the fix is in the sequence alone: enter the planes G7…G0, R7…R0 instead. If the check shows anything else (for example R planes all black), make the pack with `--singles` and compose the image in the GUI's own Image / Firmware → Create Images tool: slot *k* at bit position *k*, bit depth 1. The GUI then packs it consistently with its own firmware builder.
- The engine checks its side of this contract before the camera opens. `softsync.layout_from_manifest` refuses a pack whose manifest differs from the loop the engine sorts and decodes. It checks the pattern version, the dither design, slot *k* on bit *k*, the decode order, legal 1-bit timing and one image per variant. The zone mask must also match its recorded sha256.
- The whole loop is 16 planes of **one** 24-bit image. The DLPC350 preloads up to 48 planes (two images), so the loop replays from internal memory with no flash-load pause. Loading a buffer from flash takes up to about 200 ms, and it happens only when the Image Index changes.

## 5. Operator procedure in the 4500 MKII GUI

Use the GUI build for the **continuous** LED limits (the one labelled R120 G180 B160), not the short-burst build. The names below are the GUI's own labels. `procedure.txt` in each pack is the same list with that pack's file names.

### 5.0 Before you start

- The HDMI scan must be idle: no engine capture running. **Modes are changed only in the GUI, and never mid-scan.** The engine does not switch modes and does not know which mode the projector is in. A capture run in the wrong mode fails its content checks. It does not produce a wrong answer.
- Do this once per unit: back up (§5.1), add the images (§5.2), and do the plane check (§5.4). After that, a scan day is §5.3 → §5.5 → §5.6 → §5.8.

### 5.1 Back up first (nothing is changed yet)

1. **System Control:** write down the **Firmware Version** and **Firmware Tag**.
2. **LED Driver Control:** write down the **LED Current** R / G / B. The HDMI bench preset is R35 G50 B45.
3. **Save Solution:** save a `.ini` of today's video-mode settings next to your notes.
4. **Image / Firmware tab → the firmware link** ("Download the latest Firmware release from …"). The link opens the DLPC350 firmware (DLPR350) download page. Download the firmware `.bin` and keep it with the `.ini`. **Its version must equal the Firmware Version that System Control showed in step 1.** The "latest" release may be newer than what the unit runs. If the versions differ, stop here: the unit runs its supplier's firmware, so the base for §5.2 and the rollback image must be the supplier's `.bin`. If the unit came with its own firmware `.bin`, use that file as the base and the rollback image in any case. The unit's operating notes say to use only the supplied GUI and firmware.

The GUI can write the flash but cannot read it back, so the saved `.bin` **is** the backup. Do not go on without it.

### 5.2 Put the images into flash

**Image / Firmware → Firmware Build:**

1. **Firmware File:** Browse → the backup `.bin`: the supplier's `.bin` if you have it, otherwise the firmware-link `.bin` whose version matched in §5.1. Note **"# Original Images"** = *N*.
2. Add `images/flash_00_full.bmp`, then `images/flash_01_zone.bmp`, **after** the original images (**Add**; **Change** / **Remove** act on the image selected in the index list). **"Images Added"** must read 2 (1 without a zone) and **"Total Images"** *N* + 2. Keep all the original images: the splash / default images are still needed for video mode. The full image becomes flash index *N*, and the zone image *N* + 1. The guide warns that packing can renumber images, so the plane check (§5.4) confirms the index.
3. **Add Firmware Tag:** for example `fastscan1bit` (32 characters at most). It shows under Firmware Version when the GUI connects, which tells you which firmware is in the unit.
4. **Illumination Control:** RGB Color. **Select .ini File:** leave empty.
5. **Save Updates** → a new `.bin` name. **Never overwrite the backup.**

**Image / Firmware → Firmware Upload:**

6. **Browse** → the new `.bin`. Tick **Skip Bootloader Update**: it leaves the first 128 KB, the bootloader, untouched, so an interrupted upload can be retried over USB. Leave **Fast Flash Update** unticked for the first upload.
7. **Upload.** The flash is erased, then rewritten. Wait for the end. **Do not unplug USB, power off or close the GUI.**
8. After the unit reconnects, System Control shows the new Firmware Tag. Check that video mode still shows the HDMI desktop and that the LED currents match §5.1. Fix them in LED Driver Control if not.

The Firmware Build subtab also has a **Download Pattern** button. It appears to write the added images straight into the unit's flash without a full firmware upload. That path has not been tried here. The `.bin` route above is the documented one. The rollback (§5.9) is the same for both.

### 5.3 Pattern sequence settings (Sequence Settings)

1. **Operating Mode:** Pattern Sequence (not the variable-exposure mode).
2. **Pattern Source:** Flash. **Trigger Mode:** Internal/External.
3. **Pattern Exposure (us):** 10000. **Pattern Period (us):** 10000. Equal exposure and period are legal for 1-bit patterns, and they leave no dark gap between patterns. See §6 for other values.
4. Clear the pattern list, then add the 16 rows below **in this order**. For each row set:
   - **Choose Color:** White (red + green + blue, the same colour mix as the HDMI `bw` scans the calibration was made with).
   - **Internal Trigger.**
   - **Image Index:** *N* (full) or *N* + 1 (zone). **Bit Depth:** 1. **Bit Plane:** the row's plane.
   - **Invert Pattern Data:** off. **Clear DMD after exposure:** off.
   - **Add Pattern to Sequence.**

   | # | Plane | Pattern |
   |---|---|---|
   | 0 | G0 | SYNC A (all on) |
   | 1 | G1 | SYNC B (horizontal bars) |
   | 2 | G2 | white |
   | 3 | G3 | black |
   | 4–7 | G4, G5, G6, G7 | fine T = 8, steps 0–3 |
   | 8–11 | R0, R1, R2, R3 | mid T = 80, steps 0–3 |
   | 12–15 | R4, R5, R6, R7 | coarse T = 1280, steps 0–3 |

5. **Play Once:** off. The loop repeats until Stop.
6. **Send**, then **Validate Sequence**. All the Data Validation indicators must be clear and "Sequence Validated" must show. The loop is 16 × 10 ms = 160 ms. **Get Load Image Timing** is not needed: the loop never loads from flash while it plays.
7. Optional: **Save Solution** as `fastscan_full.ini` (and after switching the index, `fastscan_zone.ini`). Next time, **Apply Solution** sets the whole sequence up in one step.

### 5.4 Plane check (once per unit, before the first scan)

Build a **one-pattern** sequence: Image Index *N*, Bit Depth 1, **Bit Plane G1**, the scan's own Pattern Exposure = Pattern Period (10000 µs), Play Once off. One pattern on repeat is a still image, and it uses only values that §5.3 validates anyway. Send, Validate, **Play**, and look at the projection:

| You see (G1) | Meaning |
|---|---|
| **Horizontal bars**, 16 mirror rows on / 16 off | the planes are as the pack assumes. Then check **R0**: vertical stripes about 80 mirrors apart (mid step 0). And **G2**: all white (full image), or only the zones (zone image, index *N* + 1). |
| **Vertical stripes about 80 mirrors apart** | the GUI reads G and R the other way round. Harmless: the loop plays 8 slots later and the sync markers fix the order. Note it in the scan log. |
| A fine grey texture (stripes 8 mirrors apart) | the bits are reversed inside the byte (G1 shows our bit 6). Enter the 16 rows of §5.3 with planes G7, G6 … G0, R7 … R0 (slot *k* on G(7 − *k*), R(15 − *k*)). |
| Bars on G1, but **R0 black** | the GUI's R is not the BMP's red byte. Use the `--singles` + Create Images route (§4). |
| Anything else (a splash image, all black) | wrong Image Index (renumbered at build). Step the index until G1 shows the bars. |

Then Stop and build the 16-pattern sequence (§5.3).

### 5.5 LED currents

- **LED Driver Control → LED Current:** start at **R35 G50 B45** (the HDMI bench values), then **Set**. Raise them only if the camera signal is too low.
- **Hard limit: the continuous limits R120 G180 B160.** The pack tool refuses anything higher. The short-burst values (R180 G255 B230, short-burst GUI only) are **never** used for a repeating loop.
- Pattern mode keeps the selected LEDs on for the whole exposure, and White means all three at once. That is about 3× the per-LED duty of video mode at the same setting. Above **R40 G60 B53** the LEDs therefore run hotter than they do in HDMI at the continuous limit. That is still allowed, and the tool warns. Keep the fan at max and **Stop** the sequence between scans.

### 5.6 Scan

1. **Pattern Sequence Start/Stop/Pause → Play.**
2. Run the fast-scan capture on the engine side (`docs/FLASH_SOFTSYNC.md`). Use the preset the HDMI scans use, because its stereo calibration is required (rule 6):

   ```
   scan.bat --preset bench_9in_metric_bright scan-flash faces_flash --pack <pack folder> --variant zone --roi X0,Y0,X1,Y1 --dry-run
   scan.bat --preset bench_9in_metric_bright scan-flash faces_flash --pack <pack folder> --variant zone --roi X0,Y0,X1,Y1
   ```

   The engine opens only the camera. The camera free-runs with a fixed exposure of 3000 µs at 154 fps or more: 226 fps full frame, about 320 fps in a window around the faces. Every frame is kept with its timestamp, using OldestFirst buffering, not NewestOnly. In a window the capture records 2 loops (0.34 s); full frame, 3 loops (0.51 s). It does not need to start at the sync.
3. **Stop** right after.

If the sorter reports the loop as "not playing", the sequence is stopped or still in video mode: press Play in the GUI. The engine will not do it.

### 5.7 Switching between full and zone

The planes are identical, so only the Image Index changes: *N* for full, *N* + 1 for zone. Rebuild the 16 rows with the other index, or **Apply Solution** with the saved `.ini`. Stop the sequence first.

### 5.8 Back to HDMI video mode

1. **Stop** the sequence.
2. **Operating Mode:** Video Mode.
3. **Input Configuration → Source Select:** Parallel RGB (the HDMI input).
4. Check that the laptop desktop is on the DMD.
5. **LED Current** back to the values from §5.1 → **Set**.
6. Only now run HDMI scans.

If the DMD shows blue lines or the internal test pattern instead of the desktop, follow `docs/HDMI_TIMING_FIX.md`. So far only a power cycle has recovered it. Mode switches are exactly the event that triggers it with the marginal HDMI timing, so switch rarely: batch the fast scans.

### 5.9 Rollback

1. **Image / Firmware → Firmware Upload → Browse** → the backup `.bin` from §5.1 (the supplier's file if you have it).
2. Tick Skip Bootloader Update, then **Upload**. Wait for the end.
3. **System Control** shows the original Firmware Version and Firmware Tag.
4. **Video Mode**, then **Apply Solution** with the §5.1 `.ini`, or re-enter the LED currents.

If the unit no longer connects over USB after a failed upload: with Skip Bootloader Update ticked, the bootloader was not touched, so power-cycle and upload the backup again. The GUI's JTAG link is the last resort for a damaged bootloader. Do not attempt it without the supplier.

## 6. Timing

Constraints: 1-bit exposure ≥ 235 µs, and exposure = period, or period − exposure ≥ 230 µs. A camera frame is clean when its whole exposure fits inside one slot with 250 µs to spare at each end. So the camera frame period must be ≤ period − camera exposure − 500 µs, which gives at least one clean frame per slot per loop whatever the phase.

| Pattern period (= exposure) | Loop | 3 loops | Max camera exposure at 226 fps | Clean frames per slot per loop (3 ms exposure) |
|---|---|---|---|---|
| 8 ms | 128 ms | 0.38 s | 3.07 ms | 1.0 |
| **10 ms (default)** | **160 ms** | **0.48 s** | 5.07 ms | 1.5 |
| 12 ms | 192 ms | 0.58 s | 7.07 ms | 1.9 |

Use `--period-us`, `--camera-exposure-us`, `--camera-fps` and `--loops` to change them. The tool refuses a combination that could miss a slot. With a region of interest around the faces and the band, the camera runs much faster than 226 fps, which allows shorter periods. Keep the period in the GUI and the pack the same: the sorter takes the pattern period from the manifest.

## 7. Troubleshooting

| Symptom | Cause / action |
|---|---|
| Validate: **Exposure/Period OOR** | exposure below 235 µs, or a typo. Use 10000 / 10000. |
| Validate: **Period, Exposure diff < 230** | exposure ≠ period but less than 230 µs apart. Make them equal. |
| Validate: **Pattern Number OOR** | Bit Depth is not 1, or the wrong plane group is selected. |
| Nothing projected after Play | still in Video Mode, or Play Once with Trigger 2 Patterns per Pulse. Check Operating Mode and Play Once (off). |
| Projection flickers or shows a pause once per loop | the Image Index differs between rows. All 16 rows must use the same index. |
| Sorter: loop not found / order inconsistent | sequence stopped, a row missing or out of order, or LED Color not White on every row. Compare with the table in §5.3. |
| Everything about 3× brighter than expected | that is the full-duty White (§5.5). Lower the currents, not the camera gain. |
| Blue lines after going back to Video Mode | `docs/HDMI_TIMING_FIX.md` |
| Engine: "the pack is not the 16-slot loop this engine sorts and decodes" | the manifest differs from the engine's loop (another pattern version, planes or timing). Rebuild the pack with this engine and load its images. |
| Engine: signal (white − black p99) low | raise the camera gain first (`--gain`), then the LED currents within §5.5. A longer camera exposure needs a slower frame rate, which means fewer clean frames per slot. |

## 8. Files

- `engine/bases/onebit.py`: the 16 slots, the fine profile search, the Floyd-Steinberg strips, the sync markers, `decode` / `prefilter_frames`, and the simulation (`observe`, `study`, `square_wave_error`, `noise_rms_cols`).
- `tools/make_flash_pack.py --fastscan`: zones, calibration scale, timing and LED rules, packing, round trip, manifest, `procedure.txt`, `--unpack-check`.
- `engine/api.py make_fastscan_pack`, `scan fastscan-pack`, web verb `make_fastscan_pack`.
- `engine/capture/softsync.py`, `engine/reconstruct/flash_scan.py`, `engine/capture/mock_flashloop.py`: the free-running capture, the software sort, the decode and the mock DLP loop. This is the capture side of the feature (`docs/FLASH_SOFTSYNC.md`).
- `tests/test_flash_fastscan.py`: patterns, dithering statistics, pack → BMP → unpack round trip, the G/R rotation, zones, timing, LED, the study numbers, and the wiring.
- `tests/test_softsync_mock.py`: the sort against the simulator's ground truth, the GUI mistakes, and the verbs.
- `tests/test_flash_e2e_mock.py`: the pack and the capture together (§9).
- `docs/FLASH_PATTERN_MODE.md`: the older 8-bit, hardware-triggered flash pack (trigger cable). This doc replaces it for the no-cable fast scan.

## 9. Pack and capture together (mock, end to end)

`tests/test_flash_e2e_mock.py` runs the whole chain on the pinhole stereo mock. The pack is built by this tool, and its zone is the two face rims on the DMD, as for the bench pack. A simulated DLPC350 plays that pack's own BMP bit-planes, read back and unpacked, on its internal timer. The free-running camera records it, then the frames are sorted and decoded, and `fast_scan_flash` stores a metric scan. The scene is bench-like: two cell end faces tilted 20°, a 0.6 mm deep cosine dent (FWHM 1.5 mm) in C1, and C2 0.48 mm proud of C1. The same planes are also captured directly, held on the DMD for the whole exposure and averaged as often as the soft sync averaged them: this is the ideal synchronous capture. The HDMI 8-bit multifreq fringes go through the same optics as a third reference.

**The contract.** For both images, the planes in the BMP equal the loop the sorter and decoder assume: the same order, sync markers and dithering. A manifest with another bit order, pattern version, dither, sync bars, decode order, timing, trigger or image index is refused before the camera opens, and so is a changed zone mask.

**Results** (camera window 256 × 160 px at 321 fps, 2 loops = 0.34 s of frames, confidence 0.88, at least 4 clean frames per pattern):

| | Truth | Flash, zone image | Direct 1-bit, zone | HDMI 8-bit, zone | Flash, full image | Direct 1-bit, full | HDMI 8-bit, full |
|---|---|---|---|---|---|---|---|
| C2 − C1, mean fitted normal (mm) | 0.480 | 0.477 | 0.484 | 0.485 | 0.476 | 0.471 | 0.485 |
| C1 dent depth, bowl fit (mm) | 0.600 | 0.597 | 0.599 | 0.605 | 0.605 | 0.595 | 0.605 |
| face plane residual (mm rms per px) | 0 | 0.039 | 0.039 | 0.051 | 0.047 | 0.048 | 0.051 |
| decoded px off by > 1 column | – | 0 | 0 | 51 (zone edge) | 0 | 0 | 0 |

- The flash scan matches the ideal synchronous capture: projector columns differ by a median of 0.0003 column (p95 0.05 to 0.07). The face heights at the face centres agree within 0.003 mm.
- C2 − C1 scatters by about ±0.008 mm between captures. This comes from the tilt of the fitted planes (0.02° over the 26.6 mm between the centres is 0.01 mm), the same size as the bench's ±0.01 mm repeatability.
- Two decode guards came out of this test. Without them, pixels on the zone's defocused edge carried errors of up to 17 columns, and pixels on the face rims carried 8-column wrong-order errors from the 3 px mid / coarse low-pass. The capture now leaves both undecoded: the zone edge with the full-field SYNC A white, the rims with the phasor coherence (§3). The HDMI zone scan, which has no full-field white, keeps a few wrong orders at the zone edge.
- **Fine prefilter** (`--fine-prefilter 1`): the per-pixel residual drops about 3× (0.014 mm), and C2 − C1 and the face heights keep. But the dent reads 11–14 % shallower (0.52–0.54 mm instead of 0.60). Keep it off unless the bench comparison says otherwise.

## 10. Not verified yet

- Nothing here has run on the rig. The GUI labels were read from the GUI build on the bench PC. Behaviour that the user's guide does not describe (Download Pattern, the G/R naming) is covered by the plane check and the rollback.
- The diamond row parity of this unit is unknown (calibration parity null). Measuring it (`row_parity`) sets the sign of the ±0.08 mm offset.
- The projector σ is inferred from the HDMI modulation, not measured.
- LED thermal behaviour at full duty has not been measured. Start low and watch the fan.
- The end-to-end numbers of §9 are from the mock (1.5 DN read noise, 1.2 column projector blur). The bench checklist (§11) compares the first real flash scan with the 5.3 s HDMI fast scan of the same scene.

## 11. First bench run: checklist

**Pack.** `library\output\flashpacks\<ts>_<label>` (the newest build). If it was rebuilt, its BMPs should be byte-identical to the previous build; only its `procedure.txt` and manifest text change (the plane check, the firmware version rule, the scan command). Before the run, `python tools\make_flash_pack.py --unpack-check <pack>` must print `round trip OK`.

**Engine, before anything is switched** (opens nothing):

```
scan.bat --preset bench_9in_metric_bright scan-flash faces_flash --pack <pack> --variant zone --dry-run
```

Expect: the geometry's latest stereo calibration (`<geometry>/<ts>_stereo`). The camera window comes from the pack's zone through the latest projector map: 448 × 228 at (368, 488). Also expect 3000 µs at about 321 fps (exposure-limited), a 160 ms loop, 2 loops = 110 frames = 0.34 s, 4.2 clean frames per pattern, and no warnings. Without a usable projector map, add `--roi 368,488,816,716`.

**Operator, in the 4500 MKII GUI** (the build labelled R120 G180 B160, not the Short Burst one):

1. §5.1 back up. The firmware-link `.bin` must have the same version as System Control; if it does not, stop and use the supplier's `.bin`.
2. §5.2 Firmware Build + Upload, with Skip Bootloader Update ticked. Afterwards the new tag shows, video mode still works, and the LED currents are unchanged.
3. §5.4 plane check. G1 = horizontal bars, R0 = stripes about 80 mirrors apart, G2 on index *N* + 1 = the zones only. Write down which row of the table applied.
4. §5.3 sequence: Image Index *N* + 1 (zone), 16 rows, Play Once off, Validate. §5.5 LEDs R35 G50 B45, fan at max. **Play.**

**Engine, while it plays** (camera only), three repeats for repeatability, then the operator presses **Stop** and returns to video mode (§5.8):

```
scan.bat --preset bench_9in_metric_bright scan-flash faces_flash_r1 --pack <pack> --variant zone
scan.bat --preset bench_9in_metric_bright scan-flash faces_flash_r2 --pack <pack> --variant zone
scan.bat --preset bench_9in_metric_bright scan-flash faces_flash_r3 --pack <pack> --variant zone
scan.bat --preset bench_9in_metric_bright scan-flash faces_flash_fine --pack <pack> --variant zone --fine-prefilter 1
```

**Success looks like this:**

- **The sync is found.** Exit 0. `variant seen` is zone. Confidence is at least 0.7. In the scan's meta, `flash.anchor` has `sync_code` one slot after `sync_white`, with a separation of at least 2.
- **All patterns are sorted.** 0 frames dropped and 0 inconsistent. At least 3 clean frames per pattern (4 expected), with the rest straddling a change. The measured loop is 160.0 ms (± 0.1), and `timing_fit` is refined with a boundary rms well under 125 µs.
- **The signal is sufficient.** Signal (white − black p99) of about 40 DN or more. Below about 25 DN, raise `--gain` by 6 dB first, then the LEDs up to R40 G60 B53 (§5.5).
- **The decode is valid.** `valid` is roughly 1 % of the full frame, because the zone is 1 % of the DMD. `zone edge` is a band a few pixels wide around each zone, mostly in its 6-mirror margin (about a quarter of the zone's pixels in the mock). `depth edges` is tens to hundreds of pixels along the rims.
- **The time is short.** Capture 0.34 s of frames, and about 1.5–2 s in all, against 5.1–5.3 s of capture for the HDMI fast scan.

Then evaluate each scan with your bench evaluation script (face points, C2 − C1, dent, flatness against a flood mask), and compare with three HDMI fast scans of the same scene:

| | HDMI fast scan (3 runs) | Flash scan should give |
|---|---|---|
| face points C1 / C2, coverage | 1820 / 2323, 1.0 | the same (± 2 %), 1.0 |
| C2 − C1 | 0.479, 0.476, 0.466 mm | within the same ± 0.01 mm spread |
| C1 dent at (1.75, −0.75) mm | −0.55 … −0.63 mm, FWHM 1.5 mm | the same position and depth within about 0.05 mm |
| C1 / C2 flatness rms (9 mm) | 0.08–0.09 / 0.05–0.06 mm | similar; less with `--fine-prefilter 1`, but a shallower dent |

The absolute range can differ from HDMI by up to ±0.08 mm, from the diamond offset (§3). C2 − C1 and the dent depth do not see it.

A refusal is exit 2 with the reason, and the recording is kept in `output\flashscans`. Fix the GUI side, then re-sort it offline: `scan.bat sort-flash <recording.npz> --decode`.

For a fresh HDMI reference in the same session, run `scan-fast` on the same zone right before or after the flash scans.

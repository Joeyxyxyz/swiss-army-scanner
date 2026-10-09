# Flash soft-sync fast scan (1-bit loop + free-running camera)

**Status:** built and mock-tested (`tests/test_softsync_mock.py`, and together with the pack end to end in `tests/test_flash_e2e_mock.py`). **Nothing has run on the real rig yet.** The HDMI video scan stays the default. The PySpin free-run code (`PySpinFreeRunCamera` in `engine/capture/softsync.py`) is bench-only and has never been executed.

The pack side (the 16 one-bit patterns, the 24-bit flash images, the GUI procedure) is in `docs/FLASH_FASTSCAN_4500.md` and `tools/make_flash_pack.py --fastscan`. This page is the engine side: recording, sorting, decoding.

## The idea

The DLPC350 plays the fast-scan pack out of ONE 24-bit flash image on its **internal timer**, on repeat: 16 one-bit patterns, exposure = period = 10 ms, a 160 ms loop. There is **no trigger cable**. The camera free-runs faster than the pattern rate, and every frame is labelled afterwards from its timestamp and its content.

- **1-bit only.** In pattern mode an 8-bit pattern is a PWM sequence of bit-planes inside its exposure, so an unsynchronised camera integrates a random slice of it. A 1-bit pattern is constant for its whole exposure: any frame whose exposure lies inside one pattern is a clean sample of it.
- **Clean frames.** With camera exposure E, frame period 1/F and pattern exposure X, a frame is clean when `[t, t + E]` lies inside one pattern with a guard (250 µs) at both ends. Every pattern gets at least one clean frame per loop when `X ≥ E + 1/F + 2 guard`. Frames across a pattern change ("straddling") are never used for the image, only for timing.
- **Sync markers.** Slot 0 `sync_white` (all on, full field) and slot 1 `sync_code` (horizontal bars, the only pattern structured along camera y). They anchor the pack's order in a stream that can start anywhere in the loop.

## Who does what

| Step | Who | Where |
|---|---|---|
| Build the pack | engine (files only) | `fastscan-pack` / `make_fastscan_pack` |
| Back up firmware, add the flash images, upload | **human** | EKB 4500 MKII GUI, its firmware link (`procedure.txt` in the pack) |
| Pattern Sequence: Pattern Source Flash, Trigger Mode Internal/External, bit depth 1, exposure = period = 10000 µs, Play Once off (the loop repeats), the 16 planes in order, LEDs within the continuous limits (R120 G180 B160), Validate, Play | **human** | 4500 MKII GUI |
| Record, sort, decode, triangulate, store | engine, **camera only** | `scan-flash` / `fast_scan_flash` |
| Stop, back to video mode | **human** | 4500 MKII GUI |

The engine never sends a DLPC350 USB/HID command and never opens the HDMI projector window in this path (rule 1; `dmd_control` is not imported; `rules.check_scan_path_usb_free()` runs before the camera is opened). It only checks, from the images, that the expected loop is playing.

## Commands

```
scan scan-flash part --pack library\output\flashpacks\<ts>_<label> --variant zone --roi 500,300,940,780
scan scan-flash --pack <pack dir> --dry-run          # the timing plan; opens nothing
scan scan-flash part --pack <pack dir> --variant zone --fine-prefilter 1   # optional fine low-pass
scan sort-flash library\output\flashscans\<ts>_part.npz --decode   # offline re-sort
```

```python
r = api.fast_scan_flash(pack_dir, "zone", "part", roi=[500, 300, 940, 780])
r["scan"]                       # a normal MetricScan (meta mode 'flash_softsync')
r["frame_stats"], r["confidence"], r["times_s"], r["recording"]
api.sort_flash_recording(r["recording"], decode=True)
```

Web: verbs `fast_scan_flash` (takes the rig: the live view stops first) and `sort_flash_recording` (rig-free).

Arguments: `pack` (the folder the GUI plays; without it the pack defaults are assumed and a warning says so), `variant` (`full` / `zone`: which image the GUI plays; without it the sorter reports what it sees), `roi` (camera px) or `zone` (grid cells / atlas region; needs a projector map) for the camera window. With neither, the zone variant's mask is mapped through the latest projector map; otherwise the camera records the full frame. `loops` (default: the fewest ≥ 2 that give about 4 clean frames per pattern), `exposure_us` (default: the pack's 3000), `fps` (default: the window's maximum), `gain_db`, `keep_recording`, `dry_run`, `fine_prefilter_px` (0–3 camera px, default off; see *Decode and store*).

**The pack contract.** The loop the engine expects is exactly the packed one. `layout_from_manifest` checks the manifest against `engine/bases/onebit.py` before anything opens: the slot names, the pattern version, the dither design (fine levels and row order, the Floyd-Steinberg periods, the sync bars), slot *k* on bit *k* = G0…G7, R0…R7 in one image per variant, the decode order and basis parameters, and legal 1-bit timing (exposure = period, or at least 230 µs shorter; internal trigger; repeat). A mismatch is refused with the list of differences ("rebuild it with tools/make_flash_pack.py --fastscan"). The zone mask must match its sha256 in the manifest. A pack made for another `geometry_id` gives a warning: its zone image lights that rig position's zones.

Rule 6 is checked before the camera opens: without a stereo calibration at this geometry and fingerprint the scan is refused.

## The camera (`FlashFreeRunSync`, `PySpinFreeRunCamera`)

The engine's `RealCamera` opens the Blackfly S as for any scan (it resets ROI, binning, gamma, trigger). Then the soft-sync settings go on top:

- TriggerMode Off, ExposureAuto Off, ExposureMode Timed, fixed Gain.
- The ROI: OffsetX/Y = 0, then Width/Height, then OffsetX/Y, rounded to the camera's own node increments (`GetInc()`).
- ExposureTime (default 3000 µs), AcquisitionMode Continuous, AcquisitionFrameRateEnable + AcquisitionFrameRate (capped at the camera's maximum, read after the ROI and exposure are set).
- Chunk data FrameID + Timestamp (image leader values if chunks are not available).
- **StreamBufferHandlingMode OldestFirst**, StreamBufferCountManual ≥ frames + 16. The HDMI path's NewestOnly would drop frames here.

Every frame is kept with its frame id, the camera's timestamp and a host timestamp. Frame-id gaps count as dropped frames (more than 5 %: refused). On the way out the camera goes back to full frame, no frame-rate cap, NewestOnly, and is closed; the rig lock is released.

The camera timestamp is taken as the exposure start. Any constant offset does not matter: the sorter fits the pattern boundaries from the data.

## The sort (`sort_frames`, pure: no rig, no library)

1. **Lit pixels.** Pixels whose range over the recording clears the camera noise (measured from the most similar frame pairs) and 20 % of the p99 range. In the zone variant, pixels lit in many frames (the zone) get half of the 3000 sampled features.
2. **Not playing / stopped.** Refused when the frames do not change (median frame distance < 1.5 × the noise pair distance) or stay the same for more than 2 pattern periods (stopped, paused, started late).
3. **Loop period** by phase-dispersion minimisation: fold the frames at T, order by phase, and sum the squared steps between phase neighbours. The pack's loop (16 × 10 ms) gives the search window. A loop of a different length means patterns missing / added, or another period in the GUI. A count of equal slots in the loop tells them apart, and the refusal says which.
4. **Boundaries.** A coarse loop start from the within-slot dispersion, then the straddling frames' mixture fractions (`x = a A + (1 − a) B`, or `a A + b B + d dark` when there is dark between patterns) give every pattern change to a few µs. A line fit over all loops follows. Two models are tried: seamless, and a dark gap at the loop end, whose start is located by content. The fit residual must be under max(guard / 2, 10 µs). A GUI exposure shorter than its period (dark between patterns) is measured from the dark share of the straddling frames. Only the lit part is used then, with a warning.
5. **Clean / straddle / gap / inconsistent.** Every frame gets a slot and a status. A clean frame that does not match its slot's other frames (> 4 σ of the measured noise model) is rejected. More than 10 % such frames: refused.
6. **Anchor.** Per-slot mean images are made, and every rotation of the pack's order is scored on them. Scored: every pattern inside [black, white]; `sync_white` at white level; `sync_code` the most y-structured slot; each fringe group averaging its level (0.4375 for the fine profile, 0.5 for the Floyd-Steinberg cosines); complementary steps summing to twice that. In a zone variant only the px where the decode white reaches 80 % of `sync_white` count. The best rotation must fit (< 0.15 RMS in white = 1 units) and beat the next one 2 ×.
7. **Content checks.** These are made on the sorted images (mid / coarse low-passed like the decode). Marker levels; the sync bars ≥ 2 × more y-structured than any other slot; group means and complementary pairs; the fringe **phase slope** of each period group, which must run Tb / Ta faster in the finer group, so swapped groups are caught; and one phase direction for all groups, so a reversed group is caught. Any failure: refused.
8. **Result.** The per-pattern mean of the clean frames over all loops, counts per pattern, loops seen, a confidence (the worst of fold quality, clean-in-every-loop, consistency, anchor margin, boundary fit), and the variant seen (`zone` when the decode white lights < 80 % of `sync_white`'s area).

Every refusal is a `SequenceError` (a `ValueError`: CLI exit 2) with the reason and what was measured (`.report`). The recording is kept (`output/flashscans/`) so it can be re-sorted offline.

## Decode and store

`decode_sorted`: white, black and the 12 fringes in `MultiFreqBasis` order go through the **unchanged** multifreq decode (4 steps, periods 8 / 80 / 1280, axis x, the pack's diamond parity). The mid and coarse frames are first low-passed 3 camera px (`onebit.prefilter_frames`). Only the rows / columns where white and black differ are decoded, and the result is placed in the full camera frame. Two kinds of pixel are left undecoded; both were found by the end-to-end mock test, and the counts are in the field's meta:

- **Depth edges** (`prefilter_edge`). Where the 3 px low-pass straddles two surfaces (a face rim against the wall), the mixed mid phase can pick the wrong fine order, an 8-column error. The HDMI 8-bit decode has no low-pass and does not make it. A pixel is dropped when the mid or coarse phasor loses coherence under the low-pass: `|G*z| / (G*|z|)` < 0.86 (`onebit.phasor_coherence`). In the mock, every wrong-order pixel was below 0.83, and smooth surfaces were above 0.85 at p0.1, both at a sharp and at a defocused projector.
- **The zone's edge** (`zone_edge`, zone image only). Pixels whose decode white is below 0.9 × the full-field SYNC A white sit on the zone mask's defocused edge, where the fringes are clipped (errors of up to 17 columns in the mock). The 1-bit pack can find them because SYNC A is full field; an HDMI zone scan cannot.

`fine_prefilter_px` (optional, off by default) low-passes the fine fringes too. That is the pack's `fine_sigma_camera_px_recommended` for a sharp projector. It cuts the per-pixel scatter about 3× in the mock, but it is a lateral blur: a 1.5 mm FWHM dent reads 11–14 % shallower. Compare both on the bench before using it. Then comes ray-plane triangulation (`plane`), and the scan is stored like `scan_metric` (`metric/<geometry>/<ts>_<label>/`, `.ply`). Its meta carries `mode: flash_softsync` and `flash`: pack, variant asked / seen, camera (roi, exposure, fps, gain), `frame_stats` (per-pattern clean frames, straddling / rejected / dropped frames, confidence), the timing fit, the content checks, the plan and the recording path.

## Expected scan time

`flash_scan.expected_times()` / `plan_timing()`, 16 patterns, guard 250 µs (60 µs below 5 ms periods), Blackfly S estimates:

| Pattern period P | Camera exposure E | Camera window | fps (limit) | Loops | Clean frames / pattern | Frames recorded |
|---|---|---|---|---|---|---|
| **10 ms (the pack)** | 3 ms | full frame | 215 (USB3) | 3 | 4.2 | **0.51 s** |
| **10 ms (the pack)** | 3 ms | zone window, e.g. 480 × 352 | 321 (exposure) | 2 | 4.2 | **0.34 s** |
| 10 ms | 1.5 ms | zone window | 634 (exposure) | 2 | 10.2 | 0.34 s |
| 5 ms | 1.5 ms | zone window | 629 (exposure) | 3 | 5.7 | 0.25 s |
| 2.5 ms | 0.75 ms | 480 × 352 | 770 (sensor) | 4 | 5.0 | 0.17 s |
| 2.5 ms | 0.75 ms | 480 × 200 band | 1260 (exposure) | 2 | 4.1 | 0.09 s |

Two loops is the minimum (the loop period comes from the repeats), so with the pack's 10 ms slots the floor is 0.32 s of frames. On top of the frames, the processing times below were measured on this PC in mock and are estimates for the bench:

| | Zone window (~100 k px) | Full frame |
|---|---|---|
| camera settings + start (PySpin, estimate) | ~0.15 s | ~0.15 s |
| sort | 0.3–0.5 s | 0.9–1.4 s |
| decode | 0.03–0.2 s | 0.1–0.9 s |
| triangulate + store (cloud, preview, `.ply`) | ~0.3–0.9 s | ~4 s (1.5 M points) |
| **total, pack timing** | **~1.2–2 s** (0.34 s of it with the scene lit) | ~6–7 s |

Compare with the validated HDMI setting (14 patterns, average 2, hold 0.12 s): 5.3 s with the scene lit. Use a camera window: it is what makes the fast path fast. Shorter pattern periods need a shorter camera exposure, so less light per frame. Stay within the continuous LED limits (R120 G180 B160) and raise the gain or the averaged loops instead. The short-burst ceiling belongs to the separate Short Burst GUI and is not part of this procedure: a loop that plays until Stop is continuous operation.

## Mock tests (`tests/test_softsync_mock.py`)

`MockFlashLoop` plays the pack's patterns on a simulated internal timer. The patterns are the pack's own BMPs unpacked with `make_flash_pack`, or `onebit.fastscan_patterns`. The loop has a random start phase, a Gaussian defocus, an optional clock error, a gap at the loop end, a GUI exposure shorter than the period, and GUI mistakes: not running, stopped, a pattern missing or added, steps / groups / sync markers out of order, a group reversed, another period. `MockFreeRunCamera` integrates the DMD over `[t, t + E]`, so mixtures across a change are real. It adds read + shot noise and timestamp jitter, and can drop frames.

Checked:

- every clean frame is the pattern the loop showed for its whole exposure, and every mixed frame is rejected (against the simulator's ground truth, 4 phase / clock / noise / gap cases);
- the sorted images equal the noise-free pattern images within the averaging noise, with no bias;
- the decode matches a synchronous capture of the same 1-bit set with the same frame counts (median difference < 0.005 column), and the bench truth;
- the zone variant is sorted and recognised; a dark between patterns is measured;
- each GUI mistake is refused with its reason, and so are dropped frames and a too-short recording;
- `fast_scan_flash` on the pinhole stereo mock gives a metric scan in mm (decode vs oracle: median < 0.02, p95 < 0.1 column) with the frame stats in its meta and the recording saved. `dry_run` opens nothing. Not playing is refused and nothing is stored. No calibration: refused before the camera. The projector is never constructed. A loaded `hid` module is refused;
- CLI `scan-flash` / `sort-flash` and the web verbs.

`tests/test_flash_e2e_mock.py` puts the pack and the capture together. For both images, the pack's BMP planes equal the loop the sorter and decoder assume, and tampered manifests are refused before the camera. The chain pack → simulated DLP loop → free-running camera → sort → decode → metric scan runs on a bench-like scene: two tilted end faces, a 0.6 mm dent, and C2 0.48 mm proud of C1. It is compared with an ideal synchronous capture of the same planes, with the HDMI 8-bit fringes and with the truth. The numbers are in `docs/FLASH_FASTSCAN_4500.md` §9.

## Open points for the bench

- The PySpin free-run settings, chunk timestamps and the reachable frame rate for the zone window must be checked on the camera. `applied` in the recording's meta records what the camera took.
- Whether the DLPC350 inserts a gap at the loop end on repeat. The sorter handles it either way (the `gap` model) and reports it.
- How dim a 3 ms exposure is at the continuous LED limits with 1-bit patterns. The mock sorts down to ~27 DN of white; the decode noise then grows.
- The pack's fringes are drawn for diamond parity None, which is what the bench calibration measured. With another parity the scan warns: expect a ±0.25 column zigzag.
